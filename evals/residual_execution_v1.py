"""복합 F1/R1 두 번째 진단 고리의 private GPU 실행 backend."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import r1_repair_execution as V1
import r1_repair_execution_v2 as V2
import residual_faults_v1 as FAULTS
import residual_training_v1 as WRAPPER

ROOT = Path(__file__).resolve().parents[1]
TASK = V1.TASK
FIXED_EVAL_PROTOCOL = V1.FIXED_EVAL_PROTOCOL
WATCHDOG = V1.WATCHDOG
PROBES = V1.PROBES
REQUIRED_RUNTIME_SOURCES = {
    "scripts/reinforcement_learning/rsl_rl/train.py",
    "source/isaaclab/isaaclab/envs/mdp/rewards.py",
}

sha = V1.sha
write_new = V1.write_new
fixed_evaluate = V2.fixed_evaluate


def preflight(isaaclab: Path) -> dict[str, Any]:
    result = V2.preflight(isaaclab)
    reward_source = Path(isaaclab).resolve() / "source/isaaclab/isaaclab/envs/mdp/rewards.py"
    if not reward_source.is_file():
        raise FileNotFoundError("F1 원본 Isaac Lab rewards.py가 없다")
    return {**result, "private_root": str((ROOT / "bench/private/codex_support_20261015/task29/runs").resolve()),
            "runtime_reward_source": str(reward_source.resolve())}


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
        "original_faults": binding.get("original_faults"),
        "applied_corrections": binding.get("applied_corrections"),
    }
    return WRAPPER.validate_manifest(data)


def _verify_binding(manifest: Mapping[str, Any], isaaclab: Path) -> None:
    runtime = manifest.get("runtime_source_sha256_lf")
    if not isinstance(runtime, dict) or not REQUIRED_RUNTIME_SOURCES <= set(runtime):
        raise ValueError("train.py와 원본 rewards.py runtime source binding이 필요하다")
    WRAPPER.verify_source_binding(manifest, isaaclab)


def _validate_readback(value: Any, manifest: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("schema") != "residual_training_readback_v1":
        raise ValueError("복합 학습 readback schema가 다르다")
    faults = list(FAULTS.normalize_faults(manifest["original_faults"]))
    corrections = list(FAULTS.normalize_corrections(manifest["applied_corrections"]))
    remaining = list(FAULTS.remaining_faults(faults, corrections))
    if value.get("original_faults") != faults or value.get("applied_corrections") != corrections:
        raise ValueError("복합 학습 readback의 결함/교정 mask가 다르다")
    if value.get("remaining_faults") != remaining:
        raise ValueError("복합 학습 readback의 잔여 결함 계산이 다르다")
    reward = value.get("reward")
    if not isinstance(reward, dict) or reward.get("target") != "isaaclab.envs.mdp.rewards.track_lin_vel_xy_exp":
        raise ValueError("reward target readback이 다르다")
    before = reward.get("before_code_sha256")
    after_fault = reward.get("after_fault_code_sha256")
    after = reward.get("after_code_sha256")
    if not all(isinstance(item, str) and len(item) == 64 for item in (before, after_fault, after)):
        raise ValueError("reward code fingerprint가 없다")
    if (after_fault != before) != ("F1" in faults) or (after != before) != ("F1" in remaining):
        raise ValueError("F1 주입/복원 readback이 mask와 다르다")
    if reward.get("fault_present_after") != ("F1" in remaining):
        raise ValueError("F1 최종 상태 readback이 다르다")
    optimizer = value.get("optimizer")
    if not isinstance(optimizer, dict) or optimizer.get("before") != {"learning_epochs": 5, "mini_batches": 4}:
        raise ValueError("optimizer 적용 전 readback이 5×4와 다르다")
    fault_epochs = 1 if "R1" in faults else 5
    final_epochs = 1 if "R1" in remaining else 5
    if optimizer.get("after_fault") != {"learning_epochs": fault_epochs, "mini_batches": 4}:
        raise ValueError("R1 주입 readback이 mask와 다르다")
    if optimizer.get("after") != {"learning_epochs": final_epochs, "mini_batches": 4}:
        raise ValueError("R1 최종 readback이 mask와 다르다")
    if optimizer.get("fault_present_after") != ("R1" in remaining):
        raise ValueError("R1 최종 상태 readback이 다르다")
    manager = value.get("reward_manager_binding")
    if not isinstance(manager, dict) or type(manager.get("match_count")) is not int or manager["match_count"] < 1:
        raise ValueError("reward manager target identity binding이 없다")
    if value.get("observer_manifest") != WRAPPER.observer_manifest(manifest):
        raise ValueError("optimizer observer manifest readback이 다르다")
    return value


def _inspection(records: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    rows = [json.loads(line) for line in records.read_text(encoding="utf-8").splitlines()]
    if [row.get("update_index") for row in rows] != [0, 1, 2] or any(row.get("error") for row in rows):
        raise ValueError("첫 3 update 계측이 완결되지 않았다")
    report = WRAPPER.OBSERVE.assemble_inspection(rows, WRAPPER.observer_manifest(manifest))
    remaining = FAULTS.remaining_faults(manifest["original_faults"], manifest["applied_corrections"])
    expected = "abnormal" if "R1" in remaining else "normal"
    if report.get("status") != expected:
        raise ValueError("optimizer 계측 판정이 남은 R1 상태와 다르다")
    return {"status": expected,
            "optimizer_steps": [row["observed"]["optimizer_steps"] for row in rows],
            "minibatches": [row["observed"]["minibatches"] for row in rows],
            "records": rows, "report": report}


def _run_training(stage: Path, isaaclab: Path, arm: str, seed: int, run_name: str,
                  binding: Mapping[str, Any], owned: Callable[..., Mapping[str, Any]] | None,
                  mode: str) -> dict[str, Any]:
    stage, isaaclab = Path(stage).resolve(), Path(isaaclab).resolve()
    V1._name(run_name)
    if arm not in WRAPPER.ARMS:
        raise ValueError("등록되지 않은 복합 학습 arm이다")
    if stage.exists() or V1._single_run_candidates(isaaclab, run_name):
        raise FileExistsError("stage 또는 같은 run_name 실행이 이미 있다")
    iterations = WRAPPER.MODES[mode]
    if binding.get("max_iterations") != iterations:
        raise ValueError("binding max_iterations가 실행 mode와 다르다")
    records, readback = stage / "records.jsonl", stage / "readback.json"
    manifest = _manifest(binding, arm=arm, seed=seed, run_name=run_name, mode=mode,
                         records=records, readback=readback)
    _verify_binding(manifest, isaaclab)
    stage.mkdir(parents=True, exist_ok=False)
    V1._snapshot(stage, "before")
    manifest_path = stage / "manifest.json"
    write_new(manifest_path, manifest)
    runner = V2._prepared_owned(owned)
    execution = runner(stage / "owned", [str(isaaclab / "_isaac_sim/python.bat"), "-B",
        str(ROOT / "bench/residual_training_v1.py"), "--manifest", str(manifest_path)],
        isaaclab, WATCHDOG["inspect" if mode == "inspection" else "train"])
    write_new(stage / "execution.json", execution)
    if not V1._finished(execution) or not records.is_file() or not readback.is_file():
        raise RuntimeError("복합 학습 또는 첫 3 update 계측이 완결되지 않았다")
    run_dir = V1._single_run_dir(isaaclab, run_name)
    configuration = V1._verify_saved_configuration(run_dir, seed, iterations)
    checkpoint = run_dir / f"model_{iterations - 1}.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError("마지막 체크포인트가 없다")
    measured = _inspection(records, manifest)
    readback_value = _validate_readback(V1._read(readback), manifest)
    checkpoint_receipt = stage / "checkpoint.json"
    checkpoint_execution = runner(stage / "checkpoint_owned", [
        str(isaaclab / "_isaac_sim/python.bat"), "-B", str(ROOT / "bench/r1_repair_training.py"),
        "--inspect-checkpoint", str(checkpoint), "--first-checkpoint", str(run_dir / "model_0.pt"),
        "--expected-final-iteration", str(iterations - 1), "--output", str(checkpoint_receipt),
    ], isaaclab, WATCHDOG["inspect"])
    write_new(stage / "checkpoint_execution.json", checkpoint_execution)
    if not V1._finished(checkpoint_execution) or not checkpoint_receipt.is_file():
        raise RuntimeError("CPU checkpoint 검사가 완결되지 않았다")
    checkpoint_evidence = V1._read(checkpoint_receipt)
    if checkpoint_evidence.get("final", {}).get("sha256") != sha(checkpoint):
        raise ValueError("CPU checkpoint 검사 SHA가 파일과 다르다")
    remaining = list(FAULTS.remaining_faults(manifest["original_faults"], manifest["applied_corrections"]))
    base = {
        "run_name": run_name, "seed": seed, "arm": arm,
        "original_faults": list(manifest["original_faults"]),
        "applied_corrections": list(manifest["applied_corrections"]),
        "remaining_faults": remaining, "checkpoint_loaded": False,
        "source_checkpoint_sha256": manifest["source_checkpoint_sha256"],
        "checkpoint": {"path": str(checkpoint), "sha256": sha(checkpoint)},
        "run_dir": str(run_dir), "manifest_path": str(manifest_path),
        "readback_path": str(readback),
        "inspection": {key: measured[key] for key in ("status", "optimizer_steps", "minibatches")},
        "execution": execution, "readback": readback_value, "records_path": str(records),
        "report": measured["report"], "configuration": configuration,
        "checkpoint_evidence": checkpoint_evidence, "checkpoint_execution": checkpoint_execution,
    }
    if mode == "inspection":
        report_path = stage / "inspection.json"
        write_new(report_path, measured["report"])
        V1._snapshot(stage, "after")
        return {**base, "measurement": measured, "report_path": str(report_path)}
    telemetry = stage / "telemetry.json"
    extract = runner(stage / "extract", [str(isaaclab / "_isaac_sim/python.bat"),
        str(ROOT / "bench/extract_tb.py"), str(run_dir), str(telemetry)],
        isaaclab, WATCHDOG["extract"])
    write_new(stage / "extract_execution.json", extract)
    if not V1._finished(extract) or not telemetry.is_file():
        raise RuntimeError("TensorBoard 텔레메트리 추출이 완결되지 않았다")
    V1._snapshot(stage, "after")
    return {**base, "telemetry": {"path": str(telemetry), "sha256": sha(telemetry),
                                    "summary": V1._read(telemetry).get("summary")},
            "extract_execution": extract}


def _verify_repository_provenance(manifest: Mapping[str, Any]) -> None:
    """GPU runtime 파일을 열지 않고 protocol과 저장소 source 결합만 다시 검증한다."""
    commit = manifest["source_commit_sha"]
    WRAPPER._git("cat-file", "-e", f"{commit}^{{commit}}")
    if WRAPPER.subprocess.run(
            ["git", "-C", str(ROOT), "merge-base", "--is-ancestor", commit, "HEAD"]).returncode:
        raise ValueError("artifact source commit이 현재 HEAD의 조상이 아니다")
    commitments = {manifest["protocol_path"]: manifest["protocol_sha256_lf"],
                   **manifest["source_sha256_lf"]}
    for relative, expected in commitments.items():
        path = ROOT / relative
        if not path.is_file() or WRAPPER.sha256_lf(path) != expected:
            raise ValueError(f"artifact 저장소 source hash가 다르다: {relative}")
        blob = WRAPPER._git("show", f"{commit}:{relative}").stdout.encode().replace(b"\r\n", b"\n")
        if WRAPPER.hashlib.sha256(blob).hexdigest() != expected:
            raise ValueError(f"artifact source commit blob이 다르다: {relative}")


def validate_artifact(artifact: Mapping[str, Any], binding: Mapping[str, Any],
                      arm: str, seed: int) -> dict[str, Any]:
    """학습 receipt를 CPU에서 원본 manifest·readback·파일 SHA까지 재검증한다."""
    if not isinstance(artifact, Mapping) or artifact.get("arm") != arm or artifact.get("seed") != seed:
        raise ValueError("artifact arm 또는 seed가 다르다")
    if artifact.get("checkpoint_loaded") is not False:
        raise ValueError("artifact는 source checkpoint를 불러오지 않은 fresh 학습이어야 한다")
    manifest_path = Path(str(artifact.get("manifest_path", "")))
    readback_path = Path(str(artifact.get("readback_path", "")))
    records_path = Path(str(artifact.get("records_path", "")))
    if not all(path.is_absolute() and path.is_file() for path in (manifest_path, readback_path, records_path)):
        raise FileNotFoundError("artifact manifest/readback/records 원본이 없다")
    manifest = WRAPPER.validate_manifest(V1._read(manifest_path))
    if manifest["mode"] != "train" or manifest["max_iterations"] != 300:
        raise ValueError("receipt gate는 300회 train artifact만 허용한다")
    run_name = artifact.get("run_name")
    if not isinstance(run_name, str):
        raise ValueError("artifact run_name이 없다")
    expected = _manifest(binding, arm=arm, seed=seed, run_name=run_name, mode="train",
                         records=records_path, readback=readback_path)
    if manifest != expected:
        raise ValueError("artifact manifest가 사전등록 binding과 다르다")
    _verify_repository_provenance(manifest)
    readback = _validate_readback(V1._read(readback_path), manifest)
    measured = _inspection(records_path, manifest)
    if artifact.get("readback") != readback:
        raise ValueError("artifact 내장 readback이 원본과 다르다")
    summary = {key: measured[key] for key in ("status", "optimizer_steps", "minibatches")}
    if artifact.get("inspection") != summary or artifact.get("report") != measured["report"]:
        raise ValueError("artifact optimizer inspection이 원본 records와 다르다")
    if artifact.get("original_faults") != manifest["original_faults"]:
        raise ValueError("artifact original_faults가 manifest와 다르다")
    if artifact.get("applied_corrections") != manifest["applied_corrections"]:
        raise ValueError("artifact applied_corrections가 manifest와 다르다")
    remaining = list(FAULTS.remaining_faults(manifest["original_faults"], manifest["applied_corrections"]))
    if artifact.get("remaining_faults") != remaining:
        raise ValueError("artifact remaining_faults가 계산값과 다르다")
    expected_configuration = {"seed": seed, "max_iterations": 300,
                              "num_learning_epochs": 5, "num_mini_batches": 4}
    if artifact.get("configuration") != expected_configuration:
        raise ValueError("artifact 학습 설정 readback이 다르다")
    run_dir = Path(str(artifact.get("run_dir", "")))
    if not run_dir.is_absolute() or V1._verify_saved_configuration(run_dir, seed, 300) != expected_configuration:
        raise ValueError("저장된 agent.yaml 설정이 artifact와 다르다")
    checkpoint = artifact.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise ValueError("artifact checkpoint가 없다")
    checkpoint_path = Path(str(checkpoint.get("path", "")))
    if not checkpoint_path.is_file() or sha(checkpoint_path) != checkpoint.get("sha256"):
        raise ValueError("artifact checkpoint SHA가 다르다")
    if artifact.get("checkpoint_evidence", {}).get("final", {}).get("sha256") != checkpoint["sha256"]:
        raise ValueError("CPU checkpoint evidence SHA가 다르다")
    telemetry = artifact.get("telemetry")
    if not isinstance(telemetry, Mapping):
        raise ValueError("artifact telemetry가 없다")
    telemetry_path = Path(str(telemetry.get("path", "")))
    if not telemetry_path.is_file() or sha(telemetry_path) != telemetry.get("sha256"):
        raise ValueError("artifact telemetry SHA가 다르다")
    if telemetry.get("summary") != V1._read(telemetry_path).get("summary"):
        raise ValueError("artifact telemetry summary가 원본과 다르다")
    for name in ("execution", "checkpoint_execution", "extract_execution"):
        if not isinstance(artifact.get(name), Mapping) or not V1._finished(artifact[name]):
            raise ValueError(f"artifact {name} receipt가 완결되지 않았다")
    if artifact.get("source_checkpoint_sha256") != manifest["source_checkpoint_sha256"]:
        raise ValueError("artifact source checkpoint provenance가 manifest와 다르다")
    return {"valid": True, "arm": arm, "seed": seed, "remaining_faults": remaining,
            "checkpoint_sha256": checkpoint["sha256"], "telemetry_sha256": telemetry["sha256"],
            "manifest_sha256": sha(manifest_path), "readback_sha256": sha(readback_path),
            "records_sha256": sha(records_path)}


def train(stage: Path, isaaclab: Path, arm: str, seed: int, run_name: str,
          binding: Mapping[str, Any], owned: Callable[..., Mapping[str, Any]] | None = None) -> dict[str, Any]:
    return _run_training(stage, isaaclab, arm, seed, run_name, binding, owned, "train")


def inspect(stage: Path, isaaclab: Path, arm: str, seed: int, run_name: str,
            binding: Mapping[str, Any], owned: Callable[..., Mapping[str, Any]] | None = None) -> dict[str, Any]:
    return _run_training(stage, isaaclab, arm, seed, run_name, binding, owned, "inspection")


def legacy_probe(stage: Path, isaaclab: Path,
                 artifacts: Mapping[str, Any] | Sequence[Mapping[str, Any]], run_name: str,
                 binding: Mapping[str, Any], owned=None, *, probes=None) -> dict[str, Any]:
    stage, isaaclab = Path(stage).resolve(), Path(isaaclab).resolve()
    V1._name(run_name)
    rows = V1._artifact_map(artifacts)
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
    V1._verify_runtime(isaaclab, binding)
    checkpoint = Path(artifact["checkpoint"]["path"])
    if not checkpoint.is_file() or sha(checkpoint) != artifact["checkpoint"]["sha256"]:
        raise ValueError("probe checkpoint가 바뀌었다")
    remaining = FAULTS.remaining_faults(artifact["original_faults"], artifact["applied_corrections"])
    expected_fault = "F1" if "F1" in remaining else "NONE"
    manifest_path = Path(artifact["manifest_path"])
    if not manifest_path.is_file():
        raise FileNotFoundError("probe source manifest가 없다")
    stage.mkdir(parents=True, exist_ok=False)
    V1._snapshot(stage, "before")
    report, jobs = stage / "report.json", stage / "jobs.json"
    write_new(jobs, {"task": binding.get("task", TASK), "fault": expected_fault,
        "seed": artifact["seed"], "jobs": [{"name": run_name, "checkpoint": str(checkpoint),
        "probes": selected, "output": str(report)}]})
    runner = V2._prepared_owned(owned)
    execution = runner(stage / "owned", [str(isaaclab / "_isaac_sim/python.bat"), "-B",
        str(ROOT / "bench/residual_probe_v1.py"), "legacy", "--manifest", str(manifest_path),
        "--", "--jobs", str(jobs), "--headless"], isaaclab, WATCHDOG["probe"])
    write_new(stage / "execution.json", execution)
    if not V1._finished(execution) or not report.is_file():
        raise RuntimeError("동결 legacy probe가 완결되지 않았다")
    value = V1._read(report)
    if (value.get("name") != run_name or value.get("task") != binding.get("task", TASK)
            or value.get("seed") != artifact["seed"]):
        raise ValueError("probe report name/task/seed binding이 다르다")
    observed = value.get("probes")
    if not isinstance(observed, dict) or set(observed) != set(selected):
        raise ValueError("probe report의 측정 목록이 등록 선택과 다르다")
    if any(not isinstance(row, dict) or row.get("exit") != 0 for row in observed.values()):
        raise RuntimeError("probe 하나 이상이 exit 0으로 끝나지 않았다")
    if value.get("checkpoint", {}).get("sha256") != artifact["checkpoint"]["sha256"]:
        raise ValueError("probe report checkpoint binding이 다르다")
    V1._snapshot(stage, "after")
    return {"report": value, "report_path": str(report), "probes": selected,
            "fault": expected_fault, "remaining_faults": list(remaining), "execution": execution}
