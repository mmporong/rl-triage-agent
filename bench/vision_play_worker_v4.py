"""Atomic Writer capture와 종료 증거를 결합하는 Play worker."""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import record_vision_evidence_v1 as R
import vision_play_worker_v3 as V3


def run_capture(validated: Mapping[str, Any], app: Any) -> tuple[dict[str, Path], Path]:
    import vision_atomic_capture_v1 as CAPTURE
    import vision_atomic_smoke_v1 as SMOKE
    import vision_play_worker_v1 as V1

    factory = CAPTURE.make_capture_factory(validated, app)
    if validated["job"]["mode"] == R.RENDER_SYNC_SMOKE_MODE:
        return V1.run_render_sync_smoke(
            validated,
            app,
            camera_capture_factory=factory,
            smoke_collector=lambda capture: SMOKE.collect(
                capture, Path(validated["job_path"])
            ),
        )
    return V1.run_real(validated, app, camera_capture_factory=factory)


def supervise(job_path: Path, *, popen: Callable[..., Any] = subprocess.Popen) -> int:
    import vision_execution_evidence_v2 as EVIDENCE

    job = V3._regular(Path(job_path).resolve(strict=True), "job")
    before = job.read_bytes()
    validated: Mapping[str, Any] | None = None
    exit_code: int | None = None
    try:
        validated = R.validate_job(job)
        process = popen(
            [
                sys.executable,
                "-B",
                str(Path(__file__).resolve()),
                "--sdk-child",
                "--job",
                str(job),
            ],
            shell=False,
            start_new_session=False,
        )
        exit_code = process.wait()
        if job.read_bytes() != before or R.sha256_file(job) != validated["job_sha256"]:
            raise ValueError("SDK child 실행 중 job bytes가 바뀌었다")
        if exit_code != 0:
            raise RuntimeError(f"SDK child가 nonzero로 종료됐다: {exit_code}")
        V3._verify_result(validated, job)
        result_path = Path(validated["output_directory"]) / "result.json"
        V3.SHUTDOWN.verify(job, validated, result_path)
        EVIDENCE.write_manifest(job, result_path, sdk_child_exit_code=exit_code)
    except BaseException as error:  # noqa: BLE001 - SDK가 삼킨 실패도 부모에서 보존한다.
        V3._write_supervisor_failure(job, validated, error, exit_code)
        return 1
    return 0


def sdk_child(job_path: Path) -> int:
    import vision_play_worker_v2 as V2

    validated = R.validate_job(job_path)
    return V2.execute(
        job_path, app_factory=lambda: V3._sdk_app_factory(validated), runner=run_capture
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk-child", action="store_true")
    parser.add_argument("--job", required=True, type=Path)
    args = parser.parse_args(argv)
    return sdk_child(args.job) if args.sdk_child else supervise(args.job)


if __name__ == "__main__":
    raise SystemExit(main())
