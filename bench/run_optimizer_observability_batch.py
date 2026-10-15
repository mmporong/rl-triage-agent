"""optimizer observability 개발 측정 23개를 한 태그에서 순차 실행한다.

`plan`은 GPU와 파일을 건드리지 않는다. `run`은 결합 소스가 같은 HEAD에
커밋됐고 GPU 입장 검사가 통과한 뒤 태그를 한 번 소비한다.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
sys.path.insert(0, str(ROOT / "evals"))
import inspect_optimizer_observability as INSPECT  # noqa: E402
import optimizer_observability as OBSERVE  # noqa: E402
import record_evidence_media as GPU  # noqa: E402
import run_runtime_readback_smoke as OWNED  # noqa: E402


SCHEMA = "optimizer_observability_batch_v1"
TASK = "Isaac-Velocity-Flat-Unitree-Go2-v0"
NUM_ENVS = 4096
UPDATES = 3
EPOCHS = 5
MINI_BATCHES = 4
WATCHDOG_S = 900
TAG_RE = re.compile(r"[a-z0-9](?:[a-z0-9_-]{0,47})\Z")
DEFAULT_OUTPUT_ROOT = ROOT / "bench/private/codex_support_20261015/task2/gpu_batches"
TASK2_PROTOCOL = ROOT / "bench/protocols/optimizer_observability_v1.json"
BOUND_FILES = (
    Path(__file__).resolve(),
    ROOT / "bench/inspect_optimizer_observability.py",
    ROOT / "evals/optimizer_observability.py",
    ROOT / "bench/hidden_faults.py",
    ROOT / "bench/record_evidence_media.py",
    ROOT / "bench/run_runtime_readback_smoke.py",
    TASK2_PROTOCOL,
)


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    arm: str
    seed: int
    runtime_set: str | None
    hidden_fault: str | None


def registered_runs() -> tuple[RunSpec, ...]:
    runs = [
        RunSpec("none_s2042", "NONE", 2042, None, "NONE"),
        RunSpec("r1_s2042", "R1", 2042, "num_learning_epochs=1", None),
    ]
    faults = {
        "NONE": "NONE", "h01": "F5", "h02": "F1", "h03": "F6",
        "h04": "F4", "h05": "F3", "h06": "F2",
    }
    for seed in (2026, 2027, 2028):
        for arm, hidden_fault in faults.items():
            runs.append(RunSpec(f"{arm.lower()}_s{seed}", arm, seed, None, hidden_fault))
    if len(runs) != 23 or len({run.run_id for run in runs}) != 23:
        raise AssertionError("등록 실행은 서로 다른 23개여야 한다")
    return tuple(runs)


def sha256_lf(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def write_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def validate_tag(tag: str) -> str:
    if not TAG_RE.fullmatch(tag):
        raise ValueError("tag는 [a-z0-9][a-z0-9_-]{0,47} 형식이어야 한다")
    return tag


def git_head() -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], check=True,
        capture_output=True, text=True, encoding="utf-8", errors="strict",
    ).stdout.strip()


def verify_committed_sources() -> str:
    relatives = [path.relative_to(ROOT).as_posix() for path in BOUND_FILES]
    subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "--error-unmatch", "--", *relatives],
        check=True, capture_output=True,
    )
    status = subprocess.run(
        ["git", "-C", str(ROOT), "status", "--porcelain", "--", *relatives],
        check=True, capture_output=True,
    )
    if status.stdout:
        raise ValueError("배치 실행기·계측 소스·protocol을 같은 HEAD에 먼저 커밋해야 한다")
    return git_head()


def build_plan(commit: str | None = None, isaaclab: Path | None = None) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "measurement_kind": "development",
        "source_commit_sha": commit or git_head(),
        "task2_protocol": {
            "path": TASK2_PROTOCOL.relative_to(ROOT).as_posix(),
            "sha256_lf": sha256_lf(TASK2_PROTOCOL),
        },
        "source_sha256_lf": {
            path.relative_to(ROOT).as_posix(): sha256_lf(path) for path in BOUND_FILES[:-1]
        },
        "fixed_training": {
            "task": TASK, "num_envs": NUM_ENVS, "max_iterations": UPDATES,
            "configured_learning_epochs": EPOCHS,
            "configured_mini_batches": MINI_BATCHES,
        },
        "gpu_admission_profile": GPU.EXPECTED_GPU_ADMISSION["policy"],
        "isaaclab_root": str(isaaclab.resolve()) if isaaclab is not None else None,
        "sequential": True,
        "stop_on_first_failure": True,
        "tag_write_once": True,
        "runs": [asdict(run) for run in registered_runs()],
    }


def validate_plan(plan: Any) -> dict[str, Any]:
    if not isinstance(plan, dict) or plan.get("schema") != SCHEMA:
        raise ValueError("batch plan schema가 다르다")
    if plan.get("source_commit_sha") != git_head():
        raise ValueError("batch plan commit이 현재 HEAD와 다르다")
    expected_protocol = {
        "path": TASK2_PROTOCOL.relative_to(ROOT).as_posix(),
        "sha256_lf": sha256_lf(TASK2_PROTOCOL),
    }
    expected_sources = {
        path.relative_to(ROOT).as_posix(): sha256_lf(path) for path in BOUND_FILES[:-1]
    }
    if plan.get("task2_protocol") != expected_protocol or plan.get("source_sha256_lf") != expected_sources:
        raise ValueError("batch plan source/protocol commitment가 현재 파일과 다르다")
    if plan.get("runs") != [asdict(run) for run in registered_runs()]:
        raise ValueError("batch plan 실행 목록이 고정 23개와 다르다")
    fixed = plan.get("fixed_training")
    if fixed != {
        "task": TASK, "num_envs": NUM_ENVS, "max_iterations": UPDATES,
        "configured_learning_epochs": EPOCHS, "configured_mini_batches": MINI_BATCHES,
    }:
        raise ValueError("batch plan 학습 설정이 다르다")
    if (plan.get("measurement_kind") != "development" or plan.get("sequential") is not True
            or plan.get("stop_on_first_failure") is not True or plan.get("tag_write_once") is not True):
        raise ValueError("batch plan 실행 계약이 다르다")
    isaaclab_root = plan.get("isaaclab_root")
    if not isinstance(isaaclab_root, str) or not Path(isaaclab_root).is_absolute():
        raise ValueError("실행 batch plan에는 절대 Isaac Lab 경로가 필요하다")
    return plan


def validate_gpu_snapshot(snapshot: Any) -> dict[str, Any]:
    if not isinstance(snapshot, dict):
        raise ValueError("GPU 입장 검사 결과가 객체가 아니다")
    required = {
        "busy": bool, "busy_reasons": list, "admission_policy": str,
        "processes": list, "observed_at_utc": str,
    }
    for key, expected_type in required.items():
        if type(snapshot.get(key)) is not expected_type:
            raise ValueError(f"GPU 입장 검사 필드가 잘못됐다: {key}")
    if any(not isinstance(row, dict) or type(row.get("pid")) is not int for row in snapshot["processes"]):
        raise ValueError("GPU process 목록 형식이 잘못됐다")
    if snapshot["admission_policy"] != GPU.EXPECTED_GPU_ADMISSION["policy"]:
        raise ValueError("GPU 입장 검사 profile이 다르다")
    return snapshot


def verify_execution(execution: dict[str, Any]) -> None:
    required = {
        "job_assigned_before_release": True,
        "natural_exit": True,
        "timed_out": False,
        "error_type": None,
        "exit_code": 0,
        "cleanup_verified": True,
    }
    if not isinstance(execution, dict):
        raise ValueError("owned execution receipt가 객체가 아니다")
    mismatches = [key for key, value in required.items() if execution.get(key) != value]
    identities = execution.get("owned_process_identities")
    if not isinstance(identities, dict):
        mismatches.append("owned_process_identities")
    if mismatches:
        raise RuntimeError(f"owned 실행이 유효하게 끝나지 않았다: {sorted(set(mismatches))}")


def _upstream_args(spec: RunSpec) -> list[str]:
    return [
        "--task", TASK, "--num_envs", str(NUM_ENVS), "--max_iterations", str(UPDATES),
        "--seed", str(spec.seed), "--headless", "--run_name", f"optimizer_observe_{spec.run_id}",
    ]


def _sdk_args(isaaclab: Path, spec: RunSpec, manifest: Path, records: Path) -> list[str]:
    args = [
        str(isaaclab / "_isaac_sim/python.bat"), "-B", str(ROOT / "bench/inspect_optimizer_observability.py"),
        "train", "--manifest", str(manifest), "--records", str(records),
    ]
    if spec.runtime_set is not None:
        args.extend(("--runtime-set", spec.runtime_set))
    if spec.hidden_fault is not None:
        args.extend(("--hidden-fault", spec.hidden_fault))
    return [*args, "--", *_upstream_args(spec)]


def inspect_records(manifest_path: Path, records_path: Path) -> dict[str, Any]:
    manifest = INSPECT.read_json(manifest_path)
    INSPECT.verify_commitment(manifest)
    return OBSERVE.assemble_inspection(INSPECT.read_jsonl(records_path), manifest)


def validate_run_artifacts(manifest_path: Path, records_path: Path, report: Any) -> None:
    manifest = INSPECT.read_json(manifest_path)
    rows = INSPECT.read_jsonl(records_path)
    expected = manifest.get("runtime_updates_to_observe")
    if expected != UPDATES or [row.get("update_index") for row in rows] != list(range(UPDATES)):
        raise ValueError("계측 update가 고정 3개보다 적거나 순서가 다르다")
    if any(row.get("error") is not None for row in rows):
        raise RuntimeError("계측 record에 학습 예외가 기록됐다")
    if not isinstance(report, dict) or report.get("records") != UPDATES:
        raise ValueError("inspection report의 record 수가 고정 3개와 다르다")
    if report.get("status") not in {"normal", "abnormal"}:
        raise ValueError("계측 결과가 normal/abnormal로 완결되지 않았다")


def aggregate(tag_dir: Path, plan: dict[str, Any]) -> dict[str, Any]:
    validate_plan(plan)
    stored_plan = INSPECT.read_json(tag_dir / "batch_plan.json")
    if stored_plan != plan:
        raise ValueError("저장 batch plan이 실행 plan과 다르다")
    initial_gpu_path = tag_dir / "gpu_admission_initial.json"
    if not initial_gpu_path.is_file():
        raise ValueError("초기 GPU 입장 receipt가 없다")
    initial_gpu = validate_gpu_snapshot(INSPECT.read_json(initial_gpu_path))
    if initial_gpu["busy"]:
        raise ValueError("저장 초기 GPU 입장 상태가 busy다")
    expected = {run.run_id: asdict(run) for run in registered_runs()}
    completed: dict[str, dict[str, Any]] = {}
    for run_id, registered in expected.items():
        folder = tag_dir / "runs" / run_id
        manifest_path = folder / "manifest.json"
        records_path = folder / "records.jsonl"
        report_path, execution_path = folder / "report.json", folder / "execution.json"
        launch_path = folder / "launch.json"
        gpu_before_path, gpu_after_path = folder / "gpu_before.json", folder / "gpu_after.json"
        artifacts = (manifest_path, records_path, report_path, execution_path,
                     launch_path, gpu_before_path, gpu_after_path)
        if not all(path.is_file() for path in artifacts):
            raise ValueError(f"필수 산출물 누락: {run_id}")
        manifest = INSPECT.read_json(manifest_path)
        INSPECT.verify_commitment(manifest)
        if manifest.get("source_commit_sha") != plan["source_commit_sha"]:
            raise ValueError(f"manifest commit과 batch plan이 다르다: {run_id}")
        for key in ("arm", "seed"):
            if manifest.get(key) != registered[key]:
                raise ValueError(f"등록과 manifest가 다르다: {run_id}:{key}")
        execution = INSPECT.read_json(execution_path)
        verify_execution(execution)
        before = validate_gpu_snapshot(INSPECT.read_json(gpu_before_path))
        after = validate_gpu_snapshot(INSPECT.read_json(gpu_after_path))
        if before["busy"] or after["busy"]:
            raise ValueError(f"저장 GPU 입장/종료 상태가 busy다: {run_id}")
        owned_pids = set(execution["owned_process_identities"])
        if any(str(row["pid"]) in owned_pids for row in after["processes"]):
            raise ValueError(f"저장 종료 GPU 상태에 owned PID가 남았다: {run_id}")
        spec = RunSpec(**registered)
        launch = INSPECT.read_json(launch_path)
        expected_launch = {
            "args": _sdk_args(Path(plan["isaaclab_root"]), spec, manifest_path, records_path),
            "cwd": plan["isaaclab_root"],
        }
        if launch != expected_launch:
            raise ValueError(f"SDK launch argv/cwd가 등록과 다르다: {run_id}")
        fresh = inspect_records(manifest_path, records_path)
        validate_run_artifacts(manifest_path, records_path, fresh)
        stored = INSPECT.read_json(report_path)
        if stored != fresh:
            raise ValueError(f"저장 report 재계산 불일치: {run_id}")
        completed[run_id] = {
            "arm": registered["arm"], "seed": registered["seed"],
            "status": fresh.get("status"), "report_sha256_lf": sha256_lf(report_path),
            "artifact_sha256_lf": {
                path.name: sha256_lf(path) for path in artifacts
            },
        }
    extra = sorted(
        path.name for path in (tag_dir / "runs").iterdir()
        if path.is_dir() and path.name not in expected
    )
    if extra:
        raise ValueError(f"등록 밖 실행 폴더가 있다: {extra}")
    r1_pair = {key: completed[key]["status"] for key in ("none_s2042", "r1_s2042")}
    specificity = [row for key, row in completed.items() if key not in r1_pair]
    return {
        "schema": "optimizer_observability_batch_summary_v1",
        "status": "complete",
        "measurement_kind": "development",
        "run_count": len(completed),
        "batch_artifact_sha256_lf": {
            "batch_plan.json": sha256_lf(tag_dir / "batch_plan.json"),
            "gpu_admission_initial.json": sha256_lf(initial_gpu_path),
        },
        "r1_pair": r1_pair,
        "specificity_status_counts": {
            status: sum(row["status"] == status for row in specificity)
            for status in ("normal", "abnormal", "unknown")
        },
        "runs": completed,
        "claim_limit": "이미 본 결함과 seed의 개발 측정이며 보류 성능이나 일반화 근거가 아니다.",
    }


def run_batch(
    tag: str,
    isaaclab: Path,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    *,
    gpu_probe: Callable[[str], dict[str, Any]] | None = None,
    owned_runner: Callable[[Path, list[str], Path, float], dict[str, Any]] | None = None,
    source_verifier: Callable[[], str] | None = None,
    manifest_builder: Callable[..., dict[str, Any]] | None = None,
    record_inspector: Callable[[Path, Path], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    validate_tag(tag)
    launcher = isaaclab / "_isaac_sim/python.bat"
    upstream = isaaclab / "scripts/reinforcement_learning/rsl_rl/train.py"
    if not launcher.is_file() or not upstream.is_file():
        raise FileNotFoundError("Isaac Lab launcher 또는 상류 train.py가 없다")
    tag_dir = output_root / tag
    if tag_dir.exists():
        raise FileExistsError(f"이미 소비된 tag다: {tag}")

    gpu_probe = gpu_probe or GPU.query_gpu
    owned_runner = owned_runner or OWNED.run_owned
    source_verifier = source_verifier or verify_committed_sources
    manifest_builder = manifest_builder or INSPECT.build_manifest
    record_inspector = record_inspector or inspect_records
    commit = source_verifier()
    plan = build_plan(commit, isaaclab)
    first_gpu = validate_gpu_snapshot(gpu_probe(plan["gpu_admission_profile"]))
    if first_gpu["busy"]:
        raise RuntimeError(f"GPU가 사용 중이다: {first_gpu['busy_reasons']}")

    tag_dir.mkdir(parents=True, exist_ok=False)
    write_new(tag_dir / "batch_plan.json", plan)
    write_new(tag_dir / "gpu_admission_initial.json", first_gpu)
    for index, spec in enumerate(registered_runs()):
        folder = tag_dir / "runs" / spec.run_id
        folder.mkdir(parents=True, exist_ok=False)
        try:
            validate_plan(plan)
            snapshot = validate_gpu_snapshot(gpu_probe(plan["gpu_admission_profile"]))
            write_new(folder / "gpu_before.json", snapshot)
            if snapshot["busy"]:
                raise RuntimeError(f"GPU가 사용 중이다: {snapshot['busy_reasons']}")
            manifest_path = folder / "manifest.json"
            records_path = folder / "records.jsonl"
            report_path = folder / "report.json"
            manifest = manifest_builder(
                EPOCHS, MINI_BATCHES, UPDATES, arm=spec.arm, seed=spec.seed,
                task=TASK, num_envs=NUM_ENVS, max_iterations=UPDATES,
            )
            INSPECT.write_new(manifest_path, manifest)
            INSPECT.verify_commitment(manifest, require_committed=True)
            if manifest.get("source_commit_sha") != plan["source_commit_sha"]:
                raise ValueError("arm manifest commit과 batch plan이 다르다")
            execution = owned_runner(folder, _sdk_args(isaaclab, spec, manifest_path, records_path),
                                     isaaclab, WATCHDOG_S)
            write_new(folder / "execution.json", execution)
            after = validate_gpu_snapshot(gpu_probe(plan["gpu_admission_profile"]))
            write_new(folder / "gpu_after.json", after)
            owned_pids = set(execution.get("owned_process_identities", {})) if isinstance(execution, dict) else set()
            residual = sorted(
                str(row.get("pid")) for row in after["processes"]
                if isinstance(row, dict) and str(row.get("pid")) in owned_pids
            )
            if residual:
                raise RuntimeError(f"owned GPU process가 종료 뒤 남았다: {residual}")
            if after["busy"]:
                raise RuntimeError(f"실행 뒤 GPU 입장 상태가 깨졌다: {after['busy_reasons']}")
            verify_execution(execution)
            report = record_inspector(manifest_path, records_path)
            validate_run_artifacts(manifest_path, records_path, report)
            write_new(report_path, report)
        except BaseException as error:
            write_new(folder / "failure.json", {
                "run_id": spec.run_id, "index": index, "type": type(error).__name__, "message": str(error),
            })
            raise
    summary = aggregate(tag_dir, plan)
    write_new(tag_dir / "summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("plan")
    run = sub.add_parser("run")
    run.add_argument("--tag", required=True)
    run.add_argument("--isaaclab", type=Path, required=True)
    run.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            print(json.dumps(build_plan(), ensure_ascii=False, indent=2, allow_nan=False))
        else:
            result = run_batch(args.tag, args.isaaclab.resolve(), args.output_root.resolve())
            print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    except (ValueError, RuntimeError, FileNotFoundError, FileExistsError,
            subprocess.CalledProcessError, json.JSONDecodeError) as error:
        print(f"거부: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
