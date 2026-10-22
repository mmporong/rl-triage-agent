"""PPO ratio 계산 오류 종단 실행의 private backend.

학습·3 update 검사만 새 진입점(bench/ppo_ratio_repair_training.py)으로 실행한다. 동결 고정 평가, 동결 legacy probe,
경로 사전 점검과 소유 프로세스 실행은 R1 backend(r1_repair_execution·r1_repair_execution_v2)를 그대로 쓴다.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Callable, Mapping

ROOT = Path(__file__).resolve().parents[1]
for _directory in (ROOT / "bench", ROOT / "evals", ROOT / "src"):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

import ppo_gradient_invariant_v2 as GRADIENT  # noqa: E402
import ppo_ratio_repair_training as WRAPPER  # noqa: E402
import r1_repair_execution as V1  # noqa: E402
import r1_repair_execution_v2 as V2  # noqa: E402

TASK = V1.TASK
FIXED_EVAL_PROTOCOL = V1.FIXED_EVAL_PROTOCOL
sha, write_new = V1.sha, V1.write_new
preflight = V2.preflight
COUNT_PRIMITIVES = ("optimizer_step_count", "minibatch_count")


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


def count_status(report: Mapping[str, Any]) -> str:
    statuses = [report["primitives"][name]["status"] for name in COUNT_PRIMITIVES]
    if "abnormal" in statuses:
        return "abnormal"
    return "normal" if all(status == "normal" for status in statuses) else "unknown"


def inspection(records: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    """첫 3 update 기록에서 갱신 횟수 판정과 surrogate gradient 판정을 따로 만든다."""
    rows = [json.loads(line) for line in records.read_text(encoding="utf-8").splitlines()]
    if [row.get("update_index") for row in rows] != [0, 1, 2]:
        raise ValueError("첫 3 update 계측이 완결되지 않았다")
    report = GRADIENT.inspect_records(rows, WRAPPER.observer_manifest(manifest))
    gradient = report["primitives"][GRADIENT.PRIMITIVE]
    return {"count_status": count_status(report), "gradient_status": gradient["status"],
            "first_mismatch_update": gradient.get("first_mismatch_update"), "gradient_batches": gradient.get("batches"),
            "optimizer_steps": [row["observed"]["optimizer_steps"] for row in rows],
            "minibatches": [row["observed"]["minibatches"] for row in rows],
            "records": rows, "report": report}


def _run_training(stage: Path, isaaclab: Path, arm: str, seed: int, run_name: str,
                  binding: Mapping[str, Any], owned, mode: str) -> dict[str, Any]:
    stage, isaaclab = Path(stage).resolve(), Path(isaaclab).resolve()
    V1._name(run_name)
    if arm not in WRAPPER.ARMS:
        raise ValueError("arm은 NONE/INVERSE_RATIO/CORRECTED 중 하나여야 한다")
    if stage.exists() or V1._single_run_candidates(isaaclab, run_name):
        raise FileExistsError("stage 또는 같은 run_name 실행이 이미 있다")
    expected_iterations = WRAPPER.MODES[mode]
    if binding.get("max_iterations") != expected_iterations:
        raise ValueError("binding max_iterations가 실행 mode와 다르다")
    records, readback = stage / "records.jsonl", stage / "readback.json"
    manifest = _manifest(dict(binding), arm=arm, seed=seed, run_name=run_name,
                         mode=mode, records=records, readback=readback)
    WRAPPER.verify_repository_binding(manifest)
    WRAPPER.verify_runtime_binding(manifest, isaaclab)
    stage.mkdir(parents=True, exist_ok=False)
    V1._snapshot(stage, "before")
    manifest_path = stage / "manifest.json"
    write_new(manifest_path, manifest)
    runner = V2._prepared_owned(owned)
    entry = str(ROOT / "bench/ppo_ratio_repair_training.py")
    argv = [str(isaaclab / "_isaac_sim/python.bat"), "-B", entry, "--manifest", str(manifest_path)]
    execution = runner(stage / "owned", argv, isaaclab, V1.WATCHDOG["inspect" if mode == "inspection" else "train"])
    write_new(stage / "execution.json", execution)
    if not V1._finished(execution) or not records.is_file() or not readback.is_file():
        raise RuntimeError("학습 실행 또는 첫 3 update 계측이 완결되지 않았다")
    run_dir = V1._single_run_dir(isaaclab, run_name)
    configuration = V1._verify_saved_configuration(run_dir, seed, expected_iterations)
    checkpoint = run_dir / f"model_{expected_iterations - 1}.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError("마지막 체크포인트가 없다")
    measured = inspection(records, manifest)
    readback_value = WRAPPER.validate_readback(V1._read(readback), arm)
    checkpoint_receipt = stage / "checkpoint.json"
    checkpoint_execution = runner(stage / "checkpoint_owned", [
        str(isaaclab / "_isaac_sim/python.bat"), "-B", entry,
        "--inspect-checkpoint", str(checkpoint), "--first-checkpoint", str(run_dir / "model_0.pt"),
        "--expected-final-iteration", str(expected_iterations - 1), "--output", str(checkpoint_receipt),
    ], isaaclab, V1.WATCHDOG["inspect"])
    write_new(stage / "checkpoint_execution.json", checkpoint_execution)
    if not V1._finished(checkpoint_execution) or not checkpoint_receipt.is_file():
        raise RuntimeError("CPU checkpoint 검사가 완결되지 않았다")
    checkpoint_evidence = V1._read(checkpoint_receipt)
    if checkpoint_evidence.get("final", {}).get("sha256") != sha(checkpoint):
        raise ValueError("CPU checkpoint 검사 SHA가 파일과 다르다")
    summary = {key: measured[key] for key in ("count_status", "gradient_status", "first_mismatch_update",
                                              "gradient_batches", "optimizer_steps", "minibatches")}
    base = {"run_name": run_name, "seed": seed, "arm": arm,
            "checkpoint": {"path": str(checkpoint), "sha256": sha(checkpoint)},
            "run_dir": str(run_dir), "inspection": summary,
            "execution": execution, "readback": readback_value,
            "records_path": str(records), "report": measured["report"], "configuration": configuration,
            "checkpoint_evidence": checkpoint_evidence, "checkpoint_execution": checkpoint_execution}
    if mode == "inspection":
        report_path = stage / "inspection.json"
        write_new(report_path, measured["report"])
        V1._snapshot(stage, "after")
        return {**base, "measurement": summary, "report_path": str(report_path)}
    telemetry = stage / "telemetry.json"
    extract = runner(stage / "extract", [str(isaaclab / "_isaac_sim/python.bat"),
        str(ROOT / "bench/extract_tb.py"), str(run_dir), str(telemetry)], isaaclab, V1.WATCHDOG["extract"])
    write_new(stage / "extract_execution.json", extract)
    if not V1._finished(extract) or not telemetry.is_file():
        raise RuntimeError("TensorBoard 텔레메트리 추출이 완결되지 않았다")
    V1._snapshot(stage, "after")
    return {**base, "telemetry": {"path": str(telemetry), "sha256": sha(telemetry),
                                   "summary": V1._read(telemetry).get("summary")},
            "extract_execution": extract}


def train(stage: Path, isaaclab: Path, arm: str, seed: int, run_name: str,
          binding: Mapping[str, Any], owned: Callable[..., Mapping[str, Any]] | None = None) -> dict[str, Any]:
    return _run_training(stage, isaaclab, arm, seed, run_name, binding, owned, "train")


def inspect(stage: Path, isaaclab: Path, arm: str, seed: int, run_name: str,
            binding: Mapping[str, Any], owned: Callable[..., Mapping[str, Any]] | None = None) -> dict[str, Any]:
    return _run_training(stage, isaaclab, arm, seed, run_name, binding, owned, "inspection")


fixed_evaluate = V2.fixed_evaluate
legacy_probe = V2.legacy_probe
