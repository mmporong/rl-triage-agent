"""R1 optimizer 관측 검사의 새 seed 개발 확인 실행기.

prepare/verify는 CPU 전용이다. execute는 seed freshness를 한 번 확인한 뒤
300-update 학습 2개와 별도 3-update 계측 학습 3개를 순차 실행한다.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "bench", ROOT / "evals"):
    sys.path.insert(0, str(directory))

import inspect_optimizer_confirmation as INSPECT  # noqa: E402
import optimizer_observability as OBSERVE  # noqa: E402
import optimizer_faults as OPT_FAULT  # noqa: E402
import record_evidence_media as GPU  # noqa: E402
import run_runtime_readback_smoke as OWNED  # noqa: E402

PROTOCOL = ROOT / "bench/protocols/novel_optimizer_confirmation_v1.json"
PRIVATE = ROOT / "bench/private/codex_support_20261015/task11/r1"
OUTPUT_ROOT = PRIVATE / "runs"
CONSUMED = PRIVATE / "consumed"
TASK = "Isaac-Velocity-Flat-Unitree-Go2-v0"
SEEDS = (2060, 2061)
NUM_ENVS = 4096
TRAIN_UPDATES = 300
INSPECT_UPDATES = 3
EPOCHS = 5
MINI_BATCHES = 4
FINAL_CHECKPOINT = "model_299.pt"
GPU_PROFILE = GPU.EXPECTED_GPU_ADMISSION["policy"]
WATCHDOG_TRAIN_S = 2700
WATCHDOG_INSPECT_S = 900
TAG_RE = re.compile(r"novel_optimizer_confirmation_[0-9]{8}[a-z0-9_]{0,12}\Z")

BOUND = (
    Path(__file__).resolve(),
    ROOT / "bench/inspect_optimizer_confirmation.py",
    ROOT / "bench/inspect_optimizer_observability.py",
    ROOT / "evals/optimizer_observability.py",
    ROOT / "bench/train_with_optimizer_fault.py",
    ROOT / "bench/optimizer_faults.py",
    ROOT / "bench/hidden_faults.py",
    ROOT / "bench/record_evidence_media.py",
    ROOT / "bench/run_runtime_readback_smoke.py",
    ROOT / "evals/preregistered_seed_guard.py",
    PROTOCOL,
)
RUNTIME_FILES = {
    "train": "scripts/reinforcement_learning/rsl_rl/train.py",
    "go2_agent": "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/agents/rsl_rl_ppo_cfg.py",
    "ppo": "_isaac_sim/kit/python/Lib/site-packages/rsl_rl/algorithms/ppo.py",
    "runner": "_isaac_sim/kit/python/Lib/site-packages/rsl_rl/runners/on_policy_runner.py",
}


@dataclass(frozen=True)
class InspectionSpec:
    run_id: str
    arm: str
    seed: int
    runtime_set: str | None
    hidden_fault: str | None


def inspections() -> tuple[InspectionSpec, ...]:
    return (
        InspectionSpec("none_s2060", "NONE", 2060, None, "NONE"),
        InspectionSpec("r1_s2060", "R1", 2060, "num_learning_epochs=1", None),
        InspectionSpec("none_specificity_s2061", "NONE", 2061, None, "NONE"),
    )


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def sha_raw(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args], check=True, capture_output=True,
                          text=True, encoding="utf-8", errors="strict").stdout.strip()


def source_hashes() -> dict[str, str]:
    return {path.relative_to(ROOT).as_posix(): sha(path) for path in BOUND[:-1]}


def fixed_protocol() -> dict[str, Any]:
    return {
        "schema": "novel_optimizer_confirmation_protocol_v1",
        "measurement_kind": "post_result_design_novel_seed_development_confirmation",
        "design_coupling": "R1 seed 2042 결과를 본 뒤 optimizer 갱신 관측 검사를 설계했다.",
        "question": "처음 사용하는 seed에서 R1의 실제 PPO 갱신 수 감소와 정상 조건의 불발을 다시 확인할 수 있는가",
        "training": {"task": TASK, "num_envs": NUM_ENVS, "max_iterations": TRAIN_UPDATES,
                     "seed": 2060, "arms": ["NONE", "R1"], "final_checkpoint": FINAL_CHECKPOINT},
        "inspection": {"max_iterations": INSPECT_UPDATES, "configured_learning_epochs": EPOCHS,
                       "configured_mini_batches": MINI_BATCHES,
                       "expected_steps_per_update": EPOCHS * MINI_BATCHES,
                       "runs": [asdict(spec) for spec in inspections()],
                       "checkpoint_relation": "300-update 체크포인트를 재개하거나 읽지 않는다. 같은 결함·seed 조건에서 새로 초기화한 3-update 학습의 PPO 갱신을 계측한다."},
        "prediction": {"none_s2060": {"status": "normal", "steps": [20, 20, 20], "minibatches": [20, 20, 20]},
                       "r1_s2060": {"status": "abnormal", "steps": [4, 4, 4], "minibatches": [4, 4, 4]},
                       "none_specificity_s2061": {"status": "normal", "steps": [20, 20, 20], "minibatches": [20, 20, 20]}},
        "decision": {"confirmed": "세 계측이 모두 예측과 일치하고 300-update 학습 두 개가 정상 종료했다.",
                     "mismatch": "완결된 normal/abnormal 결과가 예측과 다르므로 이번 새 seed 개발 확인은 불일치다.",
                     "invalid": "실행 실패, 누락 또는 unknown이 있어 판정하지 않는다."},
        "limits": ["새 seed 두 개의 개발 확인이며 보류 성능·일반화·LLM 우월·비용 절감 근거가 아니다.",
                   "정상 특이성 한 건은 optimizer epoch/update 검사 범위의 음성 대조다."],
        "fresh_seed_rule": {"seeds": list(SEEDS), "check_once": "태그·소비·GPU 조회·실행 파일 생성 전"},
        "gpu": {"profile": GPU_PROFILE, "sequential": True, "fail_stop": True},
        "write_once": True,
    }


def runtime_sources(isaaclab: Path) -> dict[str, str]:
    out = {}
    for name, relative in RUNTIME_FILES.items():
        path = isaaclab / relative
        if not path.is_file():
            raise FileNotFoundError(f"Isaac Lab runtime source가 없다: {name}")
        out[f"<isaaclab>/{relative}"] = sha(path)
    return out


def build_protocol(isaaclab: Path) -> dict[str, Any]:
    missing = [str(path) for path in BOUND[:-1] if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"결합 소스가 없다: {missing}")
    return {**fixed_protocol(), "source_sha256_lf": source_hashes(),
            "runtime_source_sha256_lf": runtime_sources(isaaclab)}


def prepare(isaaclab: Path) -> dict[str, Any]:
    expected = build_protocol(isaaclab)
    if PROTOCOL.exists():
        if read(PROTOCOL) != expected:
            raise ValueError("기존 protocol이 현재 사전등록과 다르다")
    else:
        write_new(PROTOCOL, expected)
    return {"protocol": PROTOCOL.relative_to(ROOT).as_posix(), "sha256_lf": sha(PROTOCOL)}


def _blob(commit: str, relative: str) -> str:
    return git("rev-parse", f"{commit}:{relative}")


def verify_sources(anchor: str | None = None) -> str:
    protocol = read(PROTOCOL)
    expected = {**fixed_protocol(), "source_sha256_lf": source_hashes(),
                "runtime_source_sha256_lf": protocol.get("runtime_source_sha256_lf")}
    if protocol != expected:
        raise ValueError("protocol/source commitment가 현재 파일과 다르다")
    head = git("rev-parse", "HEAD")
    anchor = anchor or head
    if subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", anchor, head]).returncode != 0:
        raise ValueError("실험 commit이 현재 HEAD의 조상이 아니다")
    relatives = [path.relative_to(ROOT).as_posix() for path in BOUND]
    subprocess.run(["git", "-C", str(ROOT), "ls-files", "--error-unmatch", "--", *relatives], check=True,
                   capture_output=True)
    if git("status", "--porcelain", "--", *relatives):
        raise ValueError("결합 소스와 protocol을 먼저 커밋해야 한다")
    for relative in relatives:
        if _blob(anchor, relative) != _blob(head, relative):
            raise ValueError(f"실험 commit 뒤 결합 blob이 바뀌었다: {relative}")
    return anchor


def verify_runtime(isaaclab: Path, protocol: dict[str, Any]) -> None:
    if protocol.get("runtime_source_sha256_lf") != runtime_sources(isaaclab):
        raise ValueError("Isaac Lab runtime source commitment가 다르다")


def validate_gpu(value: Any, *, require_idle: bool) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("admission_policy") != GPU_PROFILE:
        raise ValueError("GPU admission receipt 형식/profile이 다르다")
    if type(value.get("busy")) is not bool or not isinstance(value.get("processes"), list):
        raise ValueError("GPU admission receipt 필드가 다르다")
    if require_idle and value["busy"]:
        raise RuntimeError(f"GPU가 사용 중이다: {value.get('busy_reasons')}")
    return value


def valid_execution(value: Any) -> None:
    required = {"job_assigned_before_release": True, "natural_exit": True, "timed_out": False,
                "error_type": None, "exit_code": 0, "cleanup_verified": True}
    if not isinstance(value, dict) or any(value.get(k) != v for k, v in required.items()):
        raise RuntimeError("소유 실행 receipt가 성공 조건을 충족하지 않는다")
    if not isinstance(value.get("owned_process_identities"), dict):
        raise RuntimeError("소유 프로세스 identity가 없다")


def _train_command(isaaclab: Path, arm: str, run_name: str) -> list[str]:
    return [str(isaaclab / "_isaac_sim/python.bat"), "-B", str(ROOT / "bench/train_with_optimizer_fault.py"),
            "--task", TASK, "--num_envs", str(NUM_ENVS), "--max_iterations", str(TRAIN_UPDATES),
            "--seed", "2060", "--headless", "--run_name", run_name]


def _inspect_command(isaaclab: Path, spec: InspectionSpec, manifest: Path, records: Path) -> list[str]:
    args = [str(isaaclab / "_isaac_sim/python.bat"), "-B", str(ROOT / "bench/inspect_optimizer_confirmation.py"),
            "train", "--manifest", str(manifest), "--records", str(records)]
    if spec.runtime_set:
        args += ["--runtime-set", spec.runtime_set]
    if spec.hidden_fault:
        args += ["--hidden-fault", spec.hidden_fault]
    return args + ["--", "--task", TASK, "--num_envs", str(NUM_ENVS), "--max_iterations", str(INSPECT_UPDATES),
                   "--seed", str(spec.seed), "--headless", "--run_name", f"novel_optimizer_inspect_{spec.run_id}"]


def _run_dirs(isaaclab: Path, run_name: str) -> list[Path]:
    root = isaaclab / "logs/rsl_rl/unitree_go2_flat"
    return [] if not root.is_dir() else sorted(path for path in root.iterdir() if path.is_dir() and path.name.endswith("_" + run_name))


def inspect_report(manifest: Path, records: Path) -> dict[str, Any]:
    with INSPECT.binding() as base:
        data = base.read_json(manifest)
        base.verify_commitment(data)
        rows = base.read_jsonl(records)
    report = OBSERVE.assemble_inspection(rows, data)
    if [row.get("update_index") for row in rows] != [0, 1, 2] or any(row.get("error") is not None for row in rows):
        raise RuntimeError("3개 update 계측이 완결되지 않았다")
    if report.get("records") != 3 or report.get("status") not in {"normal", "abnormal"}:
        raise RuntimeError("계측 결과가 normal/abnormal로 완결되지 않았다")
    return report


def _counts(records_path: Path) -> tuple[list[int], list[int]]:
    rows = INSPECT.BASE.read_jsonl(records_path)
    optimizer = [row.get("observed", {}).get("optimizer_steps") for row in rows]
    minibatches = [row.get("observed", {}).get("minibatches") for row in rows]
    if (len(rows) != INSPECT_UPDATES or any(type(value) is not int for value in optimizer + minibatches)):
        raise ValueError("계측 count evidence가 없다")
    return optimizer, minibatches


def aggregate(folder: Path, protocol: dict[str, Any], anchor: str) -> dict[str, Any]:
    verify_sources(anchor)
    freshness = read(folder / "seed_freshness.json")
    if (freshness.get("schema") != "preregistered_seed_freshness_v1"
            or freshness.get("status") != "fresh" or freshness.get("seeds") != list(SEEDS)
            or freshness.get("evidence") != []):
        raise ValueError("fresh-seed receipt가 다르다")
    validate_gpu(read(folder / "gpu_before.json"), require_idle=True)
    after = validate_gpu(read(folder / "gpu_after.json"), require_idle=False)
    if read(folder / "protocol_receipt.json") != {"path": PROTOCOL.relative_to(ROOT).as_posix(), "sha256_lf": sha(PROTOCOL), "experiment_commit": anchor}:
        raise ValueError("protocol receipt가 다르다")
    trainings = {}
    runtime_root = None
    for arm in ("NONE", "R1"):
        stage = folder / "training" / arm.lower()
        execution = read(stage / "execution.json")
        valid_execution(execution)
        validate_gpu(read(stage / "gpu_before.json"), require_idle=True)
        request = read(stage / "request.json")
        if runtime_root is None:
            runtime_root = Path(request["cwd"])
            verify_runtime(runtime_root, protocol)
        elif Path(request["cwd"]) != runtime_root:
            raise ValueError("300-update 실행의 Isaac Lab root가 서로 다르다")
        expected_command = _train_command(Path(request["cwd"]), arm, f"novel_optimizer_{arm.lower()}_s2060")
        if request != {"args": expected_command, "cwd": request["cwd"], "fault": arm}:
            raise ValueError(f"300-update request가 다르다: {arm}")
        if read(stage / "launch.json") != {"args": expected_command, "cwd": request["cwd"]}:
            raise ValueError(f"300-update launch가 다르다: {arm}")
        evidence = read(stage / "evidence.json")
        if not OPT_FAULT.readback_ok(evidence.get("readback"), arm, EPOCHS, MINI_BATCHES):
            raise ValueError(f"300-update fault readback이 다르다: {arm}")
        checkpoint = Path(evidence["checkpoint_path"])
        if not checkpoint.is_file() or sha_raw(checkpoint) != evidence["checkpoint_sha256"]:
            raise ValueError(f"300-update checkpoint가 다르다: {arm}")
        trainings[arm] = {k: evidence[k] for k in ("run_name", "checkpoint_sha256")}
    reports = {}
    predictions = protocol["prediction"]
    matches = []
    for spec in inspections():
        stage = folder / "inspection" / spec.run_id
        execution = read(stage / "execution.json")
        valid_execution(execution)
        validate_gpu(read(stage / "gpu_before.json"), require_idle=True)
        manifest_path, records_path = stage / "manifest.json", stage / "records.jsonl"
        launch = read(stage / "launch.json")
        if Path(launch["cwd"]) != runtime_root:
            raise ValueError(f"3-update 실행의 Isaac Lab root가 다르다: {spec.run_id}")
        expected_launch = {"args": _inspect_command(Path(launch["cwd"]), spec, manifest_path, records_path),
                           "cwd": launch["cwd"]}
        if launch != expected_launch:
            raise ValueError(f"3-update launch가 다르다: {spec.run_id}")
        fresh = inspect_report(stage / "manifest.json", stage / "records.jsonl")
        if fresh != read(stage / "report.json"):
            raise ValueError(f"저장 계측 보고서 재계산 불일치: {spec.run_id}")
        steps, minibatches = _counts(stage / "records.jsonl")
        predicted = predictions[spec.run_id]
        match = fresh["status"] == predicted["status"] and steps == predicted["steps"] and minibatches == predicted["minibatches"]
        matches.append(match)
        reports[spec.run_id] = {"status": fresh["status"], "optimizer_steps": steps,
                                "minibatches": minibatches, "prediction_match": match,
                                "report_sha256_lf": sha(stage / "report.json")}
    owned_pids = {pid for stage in list((folder / "training").iterdir()) + list((folder / "inspection").iterdir())
                  for pid in read(stage / "execution.json").get("owned_process_identities", {})}
    if any(str(row.get("pid")) in owned_pids for row in after["processes"]):
        raise ValueError("gpu_after에 owned PID가 남았다")
    outcome = "confirmed" if all(matches) else "mismatch"
    return {"schema": "novel_optimizer_confirmation_result_v1", "outcome": outcome,
            "measurement_kind": protocol["measurement_kind"], "source_commit_sha": anchor,
            "protocol_sha256_lf": sha(PROTOCOL), "training": trainings, "inspection": reports,
            "wording": protocol["decision"][outcome], "limits": protocol["limits"]}


def execute(tag: str, isaaclab: Path,
            *, seed_guard: Callable[..., dict[str, Any]] | None = None,
            gpu_probe: Callable[[str], dict[str, Any]] | None = None,
            owned: Callable[[Path, list[str], Path, float], dict[str, Any]] | None = None) -> dict[str, Any]:
    if not TAG_RE.fullmatch(tag):
        raise ValueError("tag 형식이 다르다")
    if not (isaaclab / "_isaac_sim/python.bat").is_file():
        raise FileNotFoundError("Isaac Lab SDK python이 없다")
    anchor = verify_sources()
    protocol = read(PROTOCOL)
    verify_runtime(isaaclab, protocol)
    folder = OUTPUT_ROOT / tag
    consumed = CONSUMED / f"{sha(PROTOCOL)[:16]}_{tag}.json"
    if folder.exists() or consumed.exists():
        raise FileExistsError("tag/protocol 실행은 한 번만 소비한다")
    if seed_guard is None:
        from preregistered_seed_guard import assert_fresh_seeds
        seed_guard = assert_fresh_seeds
    freshness = seed_guard(SEEDS, isaaclab, repo=ROOT)
    if not isinstance(freshness, dict) or freshness.get("status") != "fresh" or freshness.get("seeds") != list(SEEDS):
        raise ValueError("fresh-seed receipt가 다르다")
    gpu_probe = gpu_probe or GPU.query_gpu
    first_gpu = validate_gpu(gpu_probe(GPU_PROFILE), require_idle=True)
    owned = owned or OWNED.run_owned
    folder.mkdir(parents=True, exist_ok=False)
    write_new(folder / "seed_freshness.json", freshness)
    write_new(folder / "gpu_before.json", first_gpu)
    write_new(folder / "protocol_receipt.json", {"path": PROTOCOL.relative_to(ROOT).as_posix(), "sha256_lf": sha(PROTOCOL), "experiment_commit": anchor})
    write_new(consumed, {"tag": tag, "protocol_sha256_lf": sha(PROTOCOL), "experiment_commit": anchor})
    try:
        for arm in ("NONE", "R1"):
            verify_sources(anchor)
            stage = folder / "training" / arm.lower()
            stage.mkdir(parents=True, exist_ok=False)
            write_new(stage / "gpu_before.json", validate_gpu(gpu_probe(GPU_PROFILE), require_idle=True))
            run_name = f"novel_optimizer_{arm.lower()}_s2060"
            if _run_dirs(isaaclab, run_name):
                raise FileExistsError(f"SDK run name이 이미 있다: {run_name}")
            command = _train_command(isaaclab, arm, run_name)
            write_new(stage / "request.json", {"args": command, "cwd": str(isaaclab), "fault": arm})
            previous = os.environ.get("TRIAGE_OPTIMIZER_FAULT")
            os.environ["TRIAGE_OPTIMIZER_FAULT"] = arm
            try:
                receipt = owned(stage, command, isaaclab, WATCHDOG_TRAIN_S)
            finally:
                if previous is None:
                    os.environ.pop("TRIAGE_OPTIMIZER_FAULT", None)
                else:
                    os.environ["TRIAGE_OPTIMIZER_FAULT"] = previous
            write_new(stage / "execution.json", receipt)
            valid_execution(receipt)
            run_dirs = _run_dirs(isaaclab, run_name)
            if len(run_dirs) != 1 or not (run_dirs[0] / FINAL_CHECKPOINT).is_file():
                raise RuntimeError(f"300-update run/checkpoint가 유일하지 않다: {arm}")
            checkpoint = run_dirs[0] / FINAL_CHECKPOINT
            readback = OPT_FAULT.parse((stage / "stdout.log").read_text(encoding="utf-8", errors="replace"))
            if not OPT_FAULT.readback_ok(readback, arm, EPOCHS, MINI_BATCHES):
                raise RuntimeError(f"300-update fault readback이 다르다: {arm}")
            write_new(stage / "evidence.json", {"run_name": run_name, "checkpoint_path": str(checkpoint),
                                                 "checkpoint_sha256": sha_raw(checkpoint), "readback": readback})
        for spec in inspections():
            verify_sources(anchor)
            stage = folder / "inspection" / spec.run_id
            stage.mkdir(parents=True, exist_ok=False)
            write_new(stage / "gpu_before.json", validate_gpu(gpu_probe(GPU_PROFILE), require_idle=True))
            manifest_path, records_path = stage / "manifest.json", stage / "records.jsonl"
            with INSPECT.binding() as base:
                manifest = base.build_manifest(EPOCHS, MINI_BATCHES, INSPECT_UPDATES, arm=spec.arm,
                                               seed=spec.seed, task=TASK, num_envs=NUM_ENVS,
                                               max_iterations=INSPECT_UPDATES)
                base.write_new(manifest_path, manifest)
                base.verify_commitment(manifest, require_committed=True)
            command = _inspect_command(isaaclab, spec, manifest_path, records_path)
            receipt = owned(stage, command, isaaclab, WATCHDOG_INSPECT_S)
            write_new(stage / "execution.json", receipt)
            valid_execution(receipt)
            write_new(stage / "report.json", inspect_report(manifest_path, records_path))
        after = validate_gpu(gpu_probe(GPU_PROFILE), require_idle=False)
        owned_pids = {pid for stage in list((folder / "training").iterdir()) + list((folder / "inspection").iterdir())
                      for pid in read(stage / "execution.json").get("owned_process_identities", {})}
        if any(str(row.get("pid")) in owned_pids for row in after["processes"]):
            raise RuntimeError("종료 뒤 owned GPU PID가 남았다")
        write_new(folder / "gpu_after.json", after)
        result = aggregate(folder, protocol, anchor)
        write_new(folder / "result.json", result)
        return result
    except BaseException as error:
        write_new(folder / "failure.json", {"type": type(error).__name__, "message": str(error)})
        raise


def verify(isaaclab: Path, tag: str | None = None) -> dict[str, Any]:
    anchor = verify_sources()
    verify_runtime(isaaclab, read(PROTOCOL))
    result = {"status": "ready", "experiment_commit": anchor, "protocol_sha256_lf": sha(PROTOCOL)}
    if tag:
        folder = OUTPUT_ROOT / tag
        result["result"] = aggregate(folder, read(PROTOCOL), read(folder / "protocol_receipt.json")["experiment_commit"])
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("prepare", "verify", "execute"))
    parser.add_argument("--isaaclab", type=Path)
    parser.add_argument("--tag")
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            if args.isaaclab is None:
                raise ValueError("prepare에는 --isaaclab이 필요하다")
            output = prepare(args.isaaclab.resolve())
        else:
            if args.isaaclab is None:
                raise ValueError("verify/execute에는 --isaaclab이 필요하다")
            if args.command == "verify":
                output = verify(args.isaaclab.resolve(), args.tag)
            else:
                if not args.tag:
                    raise ValueError("execute에는 --tag가 필요하다")
                output = execute(args.tag, args.isaaclab.resolve())
        print(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False))
    except (ValueError, RuntimeError, FileNotFoundError, FileExistsError, subprocess.CalledProcessError,
            json.JSONDecodeError) as error:
        print(f"거부: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
