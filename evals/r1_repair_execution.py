"""R1 종단 비교의 private 실행 backend.

학습·3 update 검사·동결 고정 평가·동결 probe를 소유 프로세스로 실행한다.
호출자는 공개 결과를 따로 구성하며, 이 모듈은 전달받은 stage 아래 원시 산출물만 쓴다.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Callable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
for _directory in (ROOT / "bench", ROOT / "evals", ROOT / "src"):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

import optimizer_observability as OBSERVE  # noqa: E402
import r1_repair_training as WRAPPER  # noqa: E402
import record_evidence_media as GPU  # noqa: E402
import run_runtime_readback_smoke as OWNED  # noqa: E402
from rl_triage.probe_loop import PROBES  # noqa: E402
from rl_triage.triage_tools import _TolerantLoader  # noqa: E402
import yaml  # noqa: E402

TASK = "Isaac-Velocity-Flat-Unitree-Go2-v0"
FIXED_EVAL_PROTOCOL = ROOT / "bench/protocols/fixed_eval_v1.json"
GPU_PROFILE = "desktop_shared_recording_v1"
WATCHDOG = {"train": 2700, "inspect": 900, "extract": 300, "fixed_eval": 900, "probe": 900}


def sha(path: Path, lf: bool = False) -> str:
    data = path.read_bytes()
    if lf:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def write_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _runner(owned):
    return owned or OWNED.run_owned


def _finished(receipt: Mapping[str, Any]) -> bool:
    return (receipt.get("job_assigned_before_release") is True
            and receipt.get("natural_exit") is True and receipt.get("timed_out") is False
            and receipt.get("error_type") is None and receipt.get("exit_code") == 0
            and receipt.get("cleanup_verified") is True)


def _name(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_]{0,79}", value):
        raise ValueError("run_name은 영문·숫자·밑줄 형식이어야 한다")
    return value


def _query_gpu() -> dict[str, Any]:
    snapshot = GPU.query_gpu(GPU_PROFILE)
    if not isinstance(snapshot, dict) or snapshot.get("admission_policy") != GPU_PROFILE:
        raise ValueError("GPU admission receipt 형식 또는 profile이 다르다")
    if type(snapshot.get("busy")) is not bool:
        raise ValueError("GPU admission busy 필드 형식이 다르다")
    return snapshot


def _gpu_clear() -> dict[str, Any]:
    snapshot = _query_gpu()
    if snapshot.get("busy") is not False:
        raise RuntimeError(f"GPU가 사용 중이다: {snapshot.get('busy_reasons')}")
    return snapshot


def _snapshot(stage: Path, name: str) -> dict[str, Any]:
    snapshot = _query_gpu()
    write_new(stage / f"gpu_{name}.json", snapshot)
    if snapshot["busy"]:
        raise RuntimeError(f"GPU가 사용 중이다: {snapshot.get('busy_reasons')}")
    return snapshot


def _verify_runtime(isaaclab: Path, binding: Mapping[str, Any]) -> None:
    if binding.get("task") != TASK:
        raise ValueError("task가 고정 allowlist와 다르다")
    hashes = binding.get("runtime_source_sha256_lf")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("runtime_source_sha256_lf binding이 없다")
    for relative, expected in hashes.items():
        path = isaaclab / relative
        if not WRAPPER._safe_relative(relative) or not path.is_file() or sha(path, True) != expected:
            raise ValueError(f"Isaac Lab runtime source가 binding과 다르다: {relative}")


def _verify_binding(manifest: Mapping[str, Any], isaaclab: Path) -> None:
    WRAPPER.verify_repository_binding(manifest)
    WRAPPER.verify_runtime_binding(manifest, isaaclab)


def _validate_readback(value: Any, arm: str) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("arm") != arm:
        raise ValueError("optimizer readback arm이 다르다")
    before, after_fault, after = value.get("before"), value.get("after_fault"), value.get("after")
    if before != {"learning_epochs": 5, "mini_batches": 4}:
        raise ValueError("optimizer 적용 전 readback이 5×4와 다르다")
    expected_fault = 5 if arm == "NONE" else 1
    expected_after = 1 if arm == "R1" else 5
    if after_fault != {"learning_epochs": expected_fault, "mini_batches": 4}:
        raise ValueError("optimizer 결함 적용 readback이 다르다")
    if after != {"learning_epochs": expected_after, "mini_batches": 4}:
        raise ValueError("optimizer 최종 readback이 다르다")
    restoration = value.get("restoration")
    if arm == "CORRECTED":
        if restoration != {"registry_value": 5, "runtime_value": 5}:
            raise ValueError("CORRECTED registry 복원 readback이 다르다")
    elif restoration is not None:
        raise ValueError("NONE/R1에는 registry 복원 readback이 없어야 한다")
    return value


def _manifest(binding: Mapping[str, Any], *, arm: str, seed: int, run_name: str,
              mode: str, records: Path, readback: Path) -> dict[str, Any]:
    data = {
        "schema": WRAPPER.SCHEMA, "mode": mode, "arm": arm, "seed": seed,
        "run_name": run_name, "task": binding.get("task", TASK),
        "num_envs": binding.get("num_envs"), "max_iterations": binding.get("max_iterations"),
        "configured_learning_epochs": binding.get("configured_learning_epochs"),
        "configured_mini_batches": binding.get("configured_mini_batches"),
        "runtime_updates_to_observe": binding.get("runtime_updates_to_observe"),
        "source_commit_sha": binding.get("source_commit_sha"),
        "protocol_path": binding.get("protocol_path"),
        "protocol_sha256_lf": binding.get("protocol_sha256_lf"),
        "source_sha256_lf": binding.get("source_sha256_lf"),
        "runtime_source_sha256_lf": binding.get("runtime_source_sha256_lf"),
        "records_path": str(records.resolve()), "readback_path": str(readback.resolve()),
        "source_checkpoint_sha256": binding.get("source_checkpoint_sha256"),
    }
    return WRAPPER.validate_manifest(data)


def _single_run_dir(isaaclab: Path, run_name: str) -> Path:
    root = isaaclab / "logs/rsl_rl/unitree_go2_flat"
    matches = [] if not root.is_dir() else sorted(
        item for item in root.iterdir()
        if item.is_dir() and re.fullmatch(rf"\d{{4}}-\d\d-\d\d_\d\d-\d\d-\d\d_{re.escape(run_name)}", item.name)
    )
    if len(matches) != 1:
        raise ValueError(f"{run_name}: 학습 실행 폴더가 하나가 아니다({len(matches)})")
    return matches[0]


def _verify_saved_configuration(run_dir: Path, seed: int, iterations: int) -> dict[str, Any]:
    path = run_dir / "params/agent.yaml"
    if not path.is_file():
        raise FileNotFoundError("저장된 agent.yaml이 없다")
    cfg = yaml.load(path.read_text(encoding="utf-8"), Loader=_TolerantLoader)
    if not isinstance(cfg, dict) or not isinstance(cfg.get("algorithm"), dict):
        raise ValueError("저장된 agent.yaml 구조가 잘못됐다")
    algorithm = cfg["algorithm"]
    if algorithm.get("num_learning_epochs") != 5 or algorithm.get("num_mini_batches") != 4:
        raise ValueError("저장된 optimizer 설정이 epoch 5, mini-batch 4와 다르다")
    if cfg.get("seed") != seed or cfg.get("max_iterations") != iterations:
        raise ValueError("저장된 seed 또는 iteration 예산이 manifest와 다르다")
    return {"seed": cfg["seed"], "max_iterations": cfg["max_iterations"],
            "num_learning_epochs": algorithm["num_learning_epochs"],
            "num_mini_batches": algorithm["num_mini_batches"]}


def _inspection(records: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    rows = [json.loads(line) for line in records.read_text(encoding="utf-8").splitlines()]
    if [row.get("update_index") for row in rows] != [0, 1, 2] or any(row.get("error") for row in rows):
        raise ValueError("첫 3 update 계측이 완결되지 않았다")
    report = OBSERVE.assemble_inspection(rows, WRAPPER.observer_manifest(manifest))
    if report.get("status") not in {"normal", "abnormal"}:
        raise ValueError("optimizer inspection 판정이 완결되지 않았다")
    return {"status": report["status"],
            "optimizer_steps": [row["observed"]["optimizer_steps"] for row in rows],
            "minibatches": [row["observed"]["minibatches"] for row in rows],
            "records": rows, "report": report}


def _run_training(stage: Path, isaaclab: Path, arm: str, seed: int, run_name: str,
                  binding: Mapping[str, Any], owned, mode: str) -> dict[str, Any]:
    stage, isaaclab = Path(stage).resolve(), Path(isaaclab).resolve()
    _name(run_name)
    if arm not in WRAPPER.ARMS:
        raise ValueError("arm은 NONE/R1/CORRECTED 중 하나여야 한다")
    if stage.exists() or _single_run_candidates(isaaclab, run_name):
        raise FileExistsError("stage 또는 같은 run_name 실행이 이미 있다")
    expected_iterations = WRAPPER.MODES[mode]
    if binding.get("max_iterations") != expected_iterations:
        raise ValueError("binding max_iterations가 실행 mode와 다르다")
    local_binding = dict(binding)
    records, readback = stage / "records.jsonl", stage / "readback.json"
    manifest = _manifest(local_binding, arm=arm, seed=seed, run_name=run_name,
                         mode=mode, records=records, readback=readback)
    _verify_binding(manifest, isaaclab)
    stage.mkdir(parents=True, exist_ok=False)
    _snapshot(stage, "before")
    manifest_path = stage / "manifest.json"
    write_new(manifest_path, manifest)
    argv = [str(isaaclab / "_isaac_sim/python.bat"), "-B", str(ROOT / "bench/r1_repair_training.py"),
            "--manifest", str(manifest_path)]
    execution = _runner(owned)(stage / "owned", argv, isaaclab, WATCHDOG["inspect" if mode == "inspection" else "train"])
    write_new(stage / "execution.json", execution)
    if not _finished(execution) or not records.is_file() or not readback.is_file():
        raise RuntimeError("학습 실행 또는 첫 3 update 계측이 완결되지 않았다")
    run_dir = _single_run_dir(isaaclab, run_name)
    configuration = _verify_saved_configuration(run_dir, seed, expected_iterations)
    checkpoint = run_dir / f"model_{expected_iterations - 1}.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError("마지막 체크포인트가 없다")
    measured = _inspection(records, manifest)
    readback_value = _validate_readback(_read(readback), arm)
    checkpoint_receipt = stage / "checkpoint.json"
    checkpoint_execution = _runner(owned)(stage / "checkpoint_owned", [
        str(isaaclab / "_isaac_sim/python.bat"), "-B", str(ROOT / "bench/r1_repair_training.py"),
        "--inspect-checkpoint", str(checkpoint), "--first-checkpoint", str(run_dir / "model_0.pt"),
        "--expected-final-iteration", str(expected_iterations - 1), "--output", str(checkpoint_receipt),
    ], isaaclab, WATCHDOG["inspect"])
    write_new(stage / "checkpoint_execution.json", checkpoint_execution)
    if not _finished(checkpoint_execution) or not checkpoint_receipt.is_file():
        raise RuntimeError("CPU checkpoint 검사가 완결되지 않았다")
    checkpoint_evidence = _read(checkpoint_receipt)
    if checkpoint_evidence.get("final", {}).get("sha256") != sha(checkpoint):
        raise ValueError("CPU checkpoint 검사 SHA가 파일과 다르다")
    base = {"run_name": run_name, "seed": seed, "arm": arm,
            "checkpoint": {"path": str(checkpoint), "sha256": sha(checkpoint)},
            "run_dir": str(run_dir), "inspection": {key: measured[key] for key in ("status", "optimizer_steps", "minibatches")},
            "execution": execution, "readback": readback_value,
            "records_path": str(records), "report": measured["report"], "configuration": configuration,
            "checkpoint_evidence": checkpoint_evidence, "checkpoint_execution": checkpoint_execution}
    if mode == "inspection":
        report_path = stage / "inspection.json"
        write_new(report_path, measured["report"])
        _snapshot(stage, "after")
        return {**base, "measurement": measured, "report_path": str(report_path)}
    telemetry = stage / "telemetry.json"
    extract = _runner(owned)(stage / "extract", [str(isaaclab / "_isaac_sim/python.bat"),
        str(ROOT / "bench/extract_tb.py"), str(run_dir), str(telemetry)], isaaclab, WATCHDOG["extract"])
    write_new(stage / "extract_execution.json", extract)
    if not _finished(extract) or not telemetry.is_file():
        raise RuntimeError("TensorBoard 텔레메트리 추출이 완결되지 않았다")
    _snapshot(stage, "after")
    return {**base, "telemetry": {"path": str(telemetry), "sha256": sha(telemetry),
                                   "summary": _read(telemetry).get("summary")},
            "extract_execution": extract}


def _single_run_candidates(isaaclab: Path, run_name: str) -> list[Path]:
    root = isaaclab / "logs/rsl_rl/unitree_go2_flat"
    return [] if not root.is_dir() else [item for item in root.iterdir()
        if item.is_dir() and item.name.endswith("_" + run_name)]


def train(stage: Path, isaaclab: Path, arm: str, seed: int, run_name: str,
          binding: Mapping[str, Any], owned: Callable[..., Mapping[str, Any]] | None = None) -> dict[str, Any]:
    return _run_training(stage, isaaclab, arm, seed, run_name, binding, owned, "train")


def inspect(stage: Path, isaaclab: Path, arm: str, seed: int, run_name: str,
            binding: Mapping[str, Any], owned: Callable[..., Mapping[str, Any]] | None = None) -> dict[str, Any]:
    return _run_training(stage, isaaclab, arm, seed, run_name, binding, owned, "inspection")


def _artifact_map(artifacts: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    values = list(artifacts.values()) if isinstance(artifacts, Mapping) and "arm" not in artifacts else list(artifacts) if not isinstance(artifacts, Mapping) else [artifacts]
    out = {row["arm"]: row for row in values}
    if len(out) != len(values) or not out:
        raise ValueError("artifact arm이 비었거나 중복됐다")
    return out


def fixed_evaluate(stage: Path, isaaclab: Path, artifacts: Mapping[str, Any] | Sequence[Mapping[str, Any]],
                   run_name: str, binding: Mapping[str, Any], owned=None) -> dict[str, Any]:
    stage, isaaclab = Path(stage).resolve(), Path(isaaclab).resolve()
    _name(run_name)
    rows = _artifact_map(artifacts)
    if stage.exists():
        raise FileExistsError("fixed evaluation stage가 이미 있다")
    _verify_runtime(isaaclab, binding)
    stage.mkdir(parents=True, exist_ok=False)
    _snapshot(stage, "before")
    report_paths = {arm: stage / f"{arm.lower()}.json" for arm in rows}
    for arm, artifact in rows.items():
        checkpoint = Path(artifact["checkpoint"]["path"])
        if not checkpoint.is_file() or sha(checkpoint) != artifact["checkpoint"]["sha256"]:
            raise ValueError(f"{arm} 체크포인트가 바뀌었다")
    jobs = stage / "jobs.json"
    write_new(jobs, {"action_scale": 0.25, "jobs": [
        {"name": f"{run_name}_{arm.lower()}", "checkpoint": row["checkpoint"]["path"],
         "output": str(report_paths[arm])} for arm, row in rows.items()]})
    execution = _runner(owned)(stage / "owned", [str(isaaclab / "_isaac_sim/python.bat"),
        str(ROOT / "evals/fixed_eval.py"), "--protocol", str(FIXED_EVAL_PROTOCOL),
        "--jobs", str(jobs), "--headless"], isaaclab, WATCHDOG["fixed_eval"])
    write_new(stage / "execution.json", execution)
    if not _finished(execution) or not all(path.is_file() for path in report_paths.values()):
        raise RuntimeError("동결 고정 평가가 완결되지 않았다")
    reports = {arm: _read(path) for arm, path in report_paths.items()}
    for arm, report in reports.items():
        if report.get("name") != f"{run_name}_{arm.lower()}" or report.get("task") != binding.get("task", TASK):
            raise ValueError(f"{arm} 고정 평가 name/task binding이 다르다")
        if report.get("checkpoint", {}).get("sha256") != rows[arm]["checkpoint"]["sha256"]:
            raise ValueError(f"{arm} 고정 평가 checkpoint binding이 다르다")
        if report.get("protocol_sha256_lf") != sha(FIXED_EVAL_PROTOCOL, True):
            raise ValueError(f"{arm} 고정 평가 protocol binding이 다르다")
    _snapshot(stage, "after")
    return {"reports": reports, "report_paths": {arm: str(path) for arm, path in report_paths.items()},
            "execution": execution}


def legacy_probe(stage: Path, isaaclab: Path, artifacts: Mapping[str, Any] | Sequence[Mapping[str, Any]],
                 run_name: str, binding: Mapping[str, Any], owned=None, *, probes=None) -> dict[str, Any]:
    stage, isaaclab = Path(stage).resolve(), Path(isaaclab).resolve()
    _name(run_name)
    rows = _artifact_map(artifacts)
    if len(rows) != 1:
        raise ValueError("legacy_probe 한 stage에는 artifact 하나만 둔다")
    arm, artifact = next(iter(rows.items()))
    selected = sorted(PROBES) if probes is None and arm == "NONE" else list(probes or ())
    if arm != "NONE" and len(selected) != 1:
        raise ValueError("source fault probe는 승인된 하나만 실행한다")
    if arm == "NONE" and probes is not None and len(selected) != 1:
        raise ValueError("명시 probe는 하나만 선택한다")
    if not selected or len(selected) != len(set(selected)) or any(item not in PROBES for item in selected):
        raise ValueError("probe 선택이 비었거나 등록 목록과 다르다")
    if stage.exists():
        raise FileExistsError("probe stage가 이미 있다")
    _verify_runtime(isaaclab, binding)
    checkpoint = Path(artifact["checkpoint"]["path"])
    if not checkpoint.is_file() or sha(checkpoint) != artifact["checkpoint"]["sha256"]:
        raise ValueError("probe checkpoint가 바뀌었다")
    stage.mkdir(parents=True, exist_ok=False)
    _snapshot(stage, "before")
    report = stage / "report.json"
    jobs = stage / "jobs.json"
    write_new(jobs, {"task": binding.get("task", TASK), "fault": "NONE", "seed": artifact["seed"], "jobs": [
        {"name": run_name, "checkpoint": str(checkpoint), "probes": selected, "output": str(report)}]})
    execution = _runner(owned)(stage / "owned", [str(isaaclab / "_isaac_sim/python.bat"),
        str(ROOT / "evals/probes_task.py"), "--jobs", str(jobs), "--headless"], isaaclab, WATCHDOG["probe"])
    write_new(stage / "execution.json", execution)
    if not _finished(execution) or not report.is_file():
        raise RuntimeError("동결 legacy probe가 완결되지 않았다")
    value = _read(report)
    if value.get("name") != run_name or value.get("task") != binding.get("task", TASK) or value.get("seed") != artifact["seed"]:
        raise ValueError("probe report name/task/seed binding이 다르다")
    observed = value.get("probes")
    if not isinstance(observed, dict) or set(observed) != set(selected):
        raise ValueError("probe report의 측정 목록이 등록 선택과 다르다")
    if any(not isinstance(row, dict) or row.get("exit") != 0 for row in observed.values()):
        raise RuntimeError("probe 하나 이상이 exit 0으로 끝나지 않았다")
    if value.get("checkpoint", {}).get("sha256") != artifact["checkpoint"]["sha256"]:
        raise ValueError("probe report checkpoint binding이 다르다")
    _snapshot(stage, "after")
    return {"report": value, "report_path": str(report), "probes": selected, "execution": execution}
