"""Isaac SDK child의 종료와 결과 파일을 별도 supervisor에서 닫는 Play worker."""
from __future__ import annotations

import argparse
import stat
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import record_vision_evidence_v1 as R
import vision_execution_evidence_v1 as EVIDENCE
import vision_shutdown_policy_v1 as SHUTDOWN

SUPERVISOR_FAILURE_SCHEMA = "vision_play_supervisor_failure_v1"


def _regular(path: Path, label: str) -> Path:
    target = Path(path)
    if (not target.is_file() or target.is_symlink()
            or any(parent.is_symlink() for parent in target.parents)):
        raise ValueError(f"{label}가 regular file이 아니다")
    metadata = target.stat(follow_symlinks=False)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise ValueError(f"{label}가 단일 regular file이 아니다")
    return target


def _read(path: Path) -> dict[str, Any]:
    target = _regular(path, path.name)
    return R.read_object(target)


def _failure_paths(job_path: Path, output: Path) -> tuple[Path, ...]:
    return (
        output / "failure.json",
        output / "shutdown_failure.json",
        output / "supervisor_failure.json",
        job_path.parent / f"{job_path.stem}.failure" / "failure.json",
    )


def _verify_result(validated: Mapping[str, Any], job_path: Path) -> dict[str, Any]:
    output = Path(validated["output_directory"])
    if any(path.exists() or path.is_symlink() for path in _failure_paths(job_path, output)):
        raise ValueError("성공 결과와 failure 증거가 함께 있다")
    result_path = _regular(output / "result.json", "result")
    result = _read(result_path)
    if (result.get("schema") != R.RESULT_SCHEMA or result.get("completed") is not True
            or result.get("job_sha256") != validated["job_sha256"]):
        raise ValueError("result 완료 또는 job SHA 결합이 다르다")
    artifacts = result.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("result artifacts가 없다")
    artifact_paths: dict[str, Path] = {}
    seen: set[str] = set()
    for logical_id, descriptor in artifacts.items():
        if (not isinstance(logical_id, str) or not isinstance(descriptor, dict)
                or set(descriptor) != {"file", "sha256", "bytes"}):
            raise ValueError("artifact descriptor 형식이 다르다")
        name, digest, size = descriptor["file"], descriptor["sha256"], descriptor["bytes"]
        if (not isinstance(name, str) or Path(name).name != name or name in seen
                or not isinstance(digest, str) or len(digest) != 64
                or type(size) is not int or size <= 0):
            raise ValueError("artifact descriptor 값이 다르다")
        matches = [path for path in output.rglob(name) if path.is_file() and not path.is_symlink()]
        if len(matches) != 1:
            raise ValueError("artifact 파일이 없거나 중복됐다")
        artifact = _regular(matches[0], f"artifact {logical_id}")
        if artifact.stat().st_size != size or R.sha256_file(artifact) != digest:
            raise ValueError("artifact bytes 또는 SHA가 다르다")
        seen.add(name)
        artifact_paths[logical_id] = artifact
    primary_sha = result.get("primary_artifact_sha256")
    if not isinstance(primary_sha, str) or len(primary_sha) != 64:
        raise ValueError("primary artifact SHA가 없다")
    primary_matches = [
        path for path in output.rglob("*")
        if path.is_file() and not path.is_symlink() and path.name != "result.json"
        and R.sha256_file(path) == primary_sha
    ]
    if len(primary_matches) != 1:
        raise ValueError("primary artifact가 없거나 중복됐다")
    expected = R.result_body(validated, artifact_paths, _regular(primary_matches[0], "primary artifact"))
    if result != expected:
        raise ValueError("result provenance가 현재 job/artifact와 다르다")
    return result


def _write_supervisor_failure(job_path: Path, validated: Mapping[str, Any] | None,
                              error: BaseException, child_exit_code: int | None) -> None:
    output = (Path(validated["output_directory"]) if validated is not None
              else job_path.parent / f"{job_path.stem}.failure")
    output.mkdir(parents=True, exist_ok=True)
    path = output / "supervisor_failure.json"
    if path.exists() or path.is_symlink():
        return
    R.write_new(path, {
        "schema": SUPERVISOR_FAILURE_SCHEMA, "completed": False,
        "stage": "verify_sdk_child_exit", "error_type": type(error).__name__,
        "error_message": str(error),
        "job_sha256": None if validated is None else validated["job_sha256"],
        "child_exit_code": child_exit_code,
    })


def supervise(job_path: Path, *, popen: Callable[..., Any] = subprocess.Popen) -> int:
    """SDK child를 같은 process group에서 끝까지 기다린 뒤 결과를 재검증한다."""
    job = _regular(Path(job_path).resolve(strict=True), "job")
    before = job.read_bytes()
    validated: Mapping[str, Any] | None = None
    exit_code: int | None = None
    try:
        validated = R.validate_job(job)
        process = popen(
            [sys.executable, "-B", str(Path(__file__).resolve()), "--sdk-child", "--job", str(job)],
            shell=False, start_new_session=False,
        )
        exit_code = process.wait()
        if job.read_bytes() != before or R.sha256_file(job) != validated["job_sha256"]:
            raise ValueError("SDK child 실행 중 job bytes가 바뀌었다")
        if exit_code != 0:
            raise RuntimeError(f"SDK child가 nonzero로 종료됐다: {exit_code}")
        _verify_result(validated, job)
        result_path = Path(validated["output_directory"]) / "result.json"
        SHUTDOWN.verify(job, validated, result_path)
        EVIDENCE.write_manifest(job, result_path, sdk_child_exit_code=exit_code)
    except BaseException as error:  # noqa: BLE001 - supervisor 실패를 write-once로 보존한다.
        _write_supervisor_failure(job, validated, error, exit_code)
        return 1
    return 0


def _sdk_app_factory(validated: Mapping[str, Any]) -> Any:
    from isaaclab.app import AppLauncher  # pyright: ignore[reportMissingImports]

    launcher = R.fixed_profile()["launcher"]
    app = AppLauncher(**launcher).app
    try:
        import carb.settings  # pyright: ignore[reportMissingImports]

        return SHUTDOWN.AppProxy(app, validated, carb.settings.get_settings())
    except BaseException:
        app.close(wait_for_replicator=False)
        raise


def sdk_child(job_path: Path) -> int:
    import vision_capture_readiness_v1 as CAPTURE
    import vision_play_worker_v2 as V2

    validated = R.validate_job(job_path)
    return V2.execute(job_path, app_factory=lambda: _sdk_app_factory(validated),
                      runner=CAPTURE.run_capture)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk-child", action="store_true")
    parser.add_argument("--job", required=True, type=Path)
    args = parser.parse_args(argv)
    return sdk_child(args.job) if args.sdk_child else supervise(args.job)


if __name__ == "__main__":
    raise SystemExit(main())
