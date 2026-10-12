"""재개 학습 교정 부모의 판정, 입력 결합, GPU admission 회귀."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "bench"))

import run_training_correction as H  # noqa: E402
from rl_triage import grounded_recovery as C  # noqa: E402


SHA = "1" * 64
OTHER_SHA = "2" * 64
VERSIONS = {"torch": "2.7", "gpu": "GPU", "isaaclab": "x", "isaaclab_tasks": "x",
            "isaaclab_rl": "x", "rsl-rl-lib": "2.3.3", "isaacsim": None}
BANDS = {"survival_ratio_min": 0.9, "err_xy_ratio_max": 1.1513,
         "err_yaw_ratio_max": 1.1719, "fall_frac_max": 0.05}


def _report(name: str, *, lin: float = 0.2, yaw: float = 0.2,
            checkpoint_file: str = "model_598.pt") -> dict:
    alive = 40_000
    rows = []
    for command in C.GRID:
        rows.append({"command": list(command), "fall_rate": 0.0,
                     "lin_vel_rmse_mps": lin, "yaw_rate_rmse_radps": yaw,
                     "num_envs": 40, "fall_count": 0, "alive_steps_sum": alive,
                     "lin_error_sq_sum": lin * lin * alive,
                     "yaw_error_sq_sum": yaw * yaw * alive})
    total_alive = alive * len(rows)
    return {"protocol": C.PROTOCOL, "protocol_sha256_lf": H.sha(H.FIXED_EVAL, True),
            "task": C.TASK, "eval_seed": C.EVAL_SEED, "num_envs": C.NUM_ENVS,
            "horizon_steps": C.HORIZON_STEPS, "step_dt": C.STEP_DT,
            "action_scale": H.FIXED["action_scale"], "standstill": dict(C.STANDSTILL),
            "versions": dict(VERSIONS), "name": name,
            "checkpoint": {"file": checkpoint_file, "sha256": SHA},
            "metrics": {"fall_rate": 0.0,
                        "mean_survival_s": total_alive / C.NUM_ENVS * C.STEP_DT,
                        "lin_vel_rmse_mps": lin, "yaw_rate_rmse_radps": yaw},
            "by_condition": rows, "elapsed_s": 10.0,
            "started_at": "2026-10-12T00:00:00+00:00",
            "finished_at": "2026-10-12T00:00:10+00:00"}


def _initials() -> dict:
    common = {"policy_state": {"sha256": "3" * 64},
              "optimizer_state": {"sha256": "4" * 64},
              "resume_checkpoint": {"path": "C:/source/model_299.pt", "sha256": "5" * 64},
              "rng_state_before_learn": {"cpu_sha256": "6" * 64,
                                         "cuda_all_sha256": ["7" * 64],
                                         "cuda_device_count": 1},
              "runtime_versions": {"python": "3.10.15", "python_cache_tag": "cpython-310",
                                   "torch_distribution": "2.7.0", "isaaclab_distribution": "0.40.0",
                                   "rsl_rl_distribution": "2.3.3",
                                   "isaaclab_source_commit": H.FIXED["isaaclab_commit"]}}
    return {arm: copy.deepcopy(common) for arm in H.ARMS}


def _summary_inputs(*, source_lin: float = 0.32, clean_lin: float = 0.2,
                    sham_lin: float = 0.32, sham_yaw: float = 0.2,
                    corrected_lin: float = 0.2) -> tuple[dict, dict, dict, list[str]]:
    reports = {
        "clean_reference": _report("clean_reference", lin=clean_lin),
        "fault_sham": _report("fault_sham", lin=sham_lin, yaw=sham_yaw),
        "corrected": _report("corrected", lin=corrected_lin),
    }
    original = {
        "clean_reference": _report("baseline_p0c_s2027", checkpoint_file="model_299.pt"),
        "fault": _report("h02_s2027", lin=source_lin, checkpoint_file="model_299.pt"),
    }
    failed = [] if source_lin == 0.2 else ["err_xy_ratio"]
    return reports, _initials(), original, failed


@pytest.mark.parametrize(("changes", "status"), [
    ({"source_lin": 0.2}, "source_failure_absent"),
    ({"clean_lin": 0.4, "sham_lin": 0.64, "corrected_lin": 0.4}, "baseline_invalid"),
    ({"sham_lin": 0.2}, "no_failure_observed"),
    ({"sham_lin": 0.2, "sham_yaw": 0.32}, "sham_shifted"),
    ({"corrected_lin": 0.32}, "no_recovery"),
    ({}, "restoration_observed_single_run"),
])
def test_summarize_covers_registered_outcomes(changes, status):
    reports, initials, original, failed = _summary_inputs(**changes)
    result = H.summarize(reports, initials, original, BANDS, failed)
    assert result["status"] == status
    assert result["wording"] == H.FIXED["result_wording"][status]


@pytest.mark.parametrize("field", ["policy_state", "optimizer_state"])
def test_summarize_rejects_sham_corrected_initial_state_mismatch(field):
    reports, initials, original, failed = _summary_inputs()
    initials["corrected"][field]["sha256"] = OTHER_SHA
    with pytest.raises(ValueError, match="initial policy/optimizer"):
        H.summarize(reports, initials, original, BANDS, failed)


def test_summarize_rejects_source_checkpoint_mismatch():
    reports, initials, original, failed = _summary_inputs()
    initials["corrected"]["resume_checkpoint"]["sha256"] = OTHER_SHA
    with pytest.raises(ValueError, match="same checkpoint"):
        H.summarize(reports, initials, original, BANDS, failed)


def test_summarize_rejects_rng_state_mismatch():
    reports, initials, original, failed = _summary_inputs()
    initials["corrected"]["rng_state_before_learn"]["cuda_all_sha256"] = [OTHER_SHA]
    with pytest.raises(ValueError, match="CPU/CUDA RNG"):
        H.summarize(reports, initials, original, BANDS, failed)


def test_summarize_rejects_training_runtime_mismatch():
    reports, initials, original, failed = _summary_inputs()
    initials["corrected"]["runtime_versions"]["torch_distribution"] = "changed"
    with pytest.raises(ValueError, match="training runtimes"):
        H.summarize(reports, initials, original, BANDS, failed)


def test_summarize_rejects_fixed_evaluation_runtime_mismatch():
    reports, initials, original, failed = _summary_inputs()
    reports["corrected"]["versions"]["torch"] = "changed"
    with pytest.raises(ValueError, match="fixed evaluation runtimes"):
        H.summarize(reports, initials, original, BANDS, failed)


def test_source_contract_rejects_different_python_patch_version(tmp_path):
    contract = {"schema": "correction_source_contract_v1", "targets": copy.deepcopy(H.A.TARGETS),
                "runtime": {"implementation": "CPython", "version": [3, 10, 14], "cache_tag": "cpython-310"}}
    with pytest.raises(ValueError, match="registered Isaac Sim Python runtime"):
        H.check_contract(contract, tmp_path)


def test_wrong_correction_cannot_remove_fault():
    clean = {key: str(index + 1) * 64 for index, key in enumerate(H.A.TARGETS)}
    injected = dict(clean)
    fault_target = H.TARGET_BY_FAULT["F1"]
    injected[fault_target] = "9" * 64
    wrong = H.CASES["case_01"]["sham"]
    row = {"schema": "correction_readback_v1", "selected_correction": wrong,
           "before_fault": clean, "after_fault": injected,
           "after_correction": dict(clean), "changed_targets": [fault_target],
           "restored_targets": [fault_target]}
    with pytest.raises(ValueError, match="removed the sham fault"):
        H.validate_readback(row, "F1", wrong)


def _write_yaml(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=True), encoding="utf-8")


def _configuration_fixture(tmp_path: Path) -> tuple[Path, dict, dict, dict]:
    source_env = {"seed": H.FIXED["seed"], "scene": {"num_envs": H.FIXED["num_envs"]},
                  "observations": {"policy": "fixed"}}
    source_agent = {"seed": H.FIXED["seed"], "num_steps_per_env": H.FIXED["num_steps_per_env"],
                    "algorithm": {"learning_rate": 0.001}, "run_name": "source",
                    "resume": False, "load_run": ".*", "load_checkpoint": "model_.*.pt",
                    "max_iterations": 300}
    env_path, agent_path = tmp_path / "source/env.yaml", tmp_path / "source/agent.yaml"
    _write_yaml(env_path, source_env)
    _write_yaml(agent_path, source_agent)
    manifest = {"protocol_path": str(tmp_path / "protocol.json"), "fault": "F1",
                "resume_env_config_path": str(env_path), "resume_agent_config_path": str(agent_path),
                "run_name": "tag_fault_sham", "load_run_regex": "^source\\+run$",
                "checkpoint_regex": "^model_299\\.pt$", "max_iterations": H.FIXED["max_iterations"]}
    protocol = {"resume_inputs": {"runs": {"F1": {"hashes": {
        "environment": H.sha(env_path), "agent": H.sha(agent_path)}}}}}
    Path(manifest["protocol_path"]).write_text(json.dumps(protocol), encoding="utf-8")
    current_agent = copy.deepcopy(source_agent)
    current_agent.update({"run_name": manifest["run_name"], "resume": True,
                          "load_run": manifest["load_run_regex"],
                          "load_checkpoint": manifest["checkpoint_regex"],
                          "max_iterations": manifest["max_iterations"]})
    log_dir = tmp_path / "target"
    _write_yaml(log_dir / "params/env.yaml", source_env)
    _write_yaml(log_dir / "params/agent.yaml", current_agent)
    return log_dir, manifest, source_env, current_agent


def test_training_configuration_allows_only_registered_resume_fields(tmp_path):
    log_dir, manifest, _, _ = _configuration_fixture(tmp_path)
    H.validate_training_configuration(log_dir, manifest)


@pytest.mark.parametrize("mutation", ["source_sha", "environment", "learning_rate", "resume_regex"])
def test_training_configuration_rejects_drift(tmp_path, mutation):
    log_dir, manifest, source_env, current_agent = _configuration_fixture(tmp_path)
    if mutation == "source_sha":
        Path(manifest["resume_agent_config_path"]).write_text("tampered: true\n", encoding="utf-8")
    elif mutation == "environment":
        source_env["scene"]["num_envs"] -= 1
        _write_yaml(log_dir / "params/env.yaml", source_env)
    elif mutation == "learning_rate":
        current_agent["algorithm"]["learning_rate"] = 0.002
        _write_yaml(log_dir / "params/agent.yaml", current_agent)
    else:
        current_agent["load_run"] = "source.*"
        _write_yaml(log_dir / "params/agent.yaml", current_agent)
    with pytest.raises(ValueError):
        H.validate_training_configuration(log_dir, manifest)


def test_manifest_anchors_and_escapes_exact_source_run(monkeypatch, tmp_path):
    source_run = tmp_path / "logs" / "2026-10-13_12-00-00_source+[run]"
    checkpoint = source_run / "model_299.pt"
    paths = {"checkpoint": checkpoint, "environment": source_run / "params/env.yaml",
             "agent": source_run / "params/agent.yaml"}
    monkeypatch.setattr(H, "source_paths", lambda isaaclab, fault: paths)
    monkeypatch.setattr(H, "read", lambda path: {"resume_inputs": {"runs": {
        "F1": {"hashes": {"checkpoint": SHA}}}}})
    folder = tmp_path / "tag"
    request = {"case": "case_01", "source_commit_sha": OTHER_SHA,
               "protocol_sha256_lf": SHA, "isaaclab_root": str(tmp_path)}
    manifest = H.manifest_for(folder, request, "corrected")
    assert manifest["load_run_regex"] == r"^2026\-10\-13_12\-00\-00_source\+\[run\]$"
    assert manifest["checkpoint_regex"] == r"^model_299\.pt$"


def _cpu_contract() -> dict:
    targets = {}
    for index, (target, spec) in enumerate(H.A.TARGETS.items(), start=1):
        targets[target] = {**spec, "source_sha256": str(index) * 64,
                           "clean_fingerprint": str(index + 3) * 64}
    return {"schema": "correction_source_contract_v1",
            "runtime": {"implementation": "CPython", "version": [3, 10, 15],
                        "cache_tag": "cpython-310"}, "targets": targets}


def _write_cpu_matrix(root: Path, contract: dict) -> Path:
    folder = root / "bench/private/cpu_matrix"
    folder.mkdir(parents=True)
    asset = root / H.CPU.GO2_SOURCE
    asset.parent.mkdir(parents=True)
    asset.write_text('UNITREE_GO2_CFG = ArticulationCfg(actuators={"base_legs": DCMotorCfg('
                     'effort_limit=23.5, saturation_effort=23.5, velocity_limit=30.0)})\n', encoding="utf-8")
    checks = {key: True for key in {
        "clean_differs_from_fault", "cuda_remained_uninitialized",
        "matching_readback_restores_selected", "matching_restore_changes_fault",
        "matching_restore_returns_clean", "wrong_target_digest_change_count_zero",
        "wrong_target_retains_fault_behavior",
    }}
    clean = {target: row["clean_fingerprint"] for target, row in contract["targets"].items()}
    raw_hashes = {target: row["source_sha256"] for target, row in contract["targets"].items()}
    for spec in H.CASES.values():
        fault = spec["fault"]
        target = H.TARGET_BY_FAULT[fault]
        injected = dict(clean)
        injected[target] = "f" * 64
        for suffix, correction in (("match", target), ("wrong", spec["sham"])):
            matching = suffix == "match"
            after = dict(injected)
            changed = []
            if matching:
                after[target] = clean[target]
                changed = [target]
            row = {
                "schema": "correction_cpu_check_v1", "fault": fault,
                "correction_id": correction, "passed": True,
                "matching_correction": matching, "source_contract": contract,
                "runtime_fingerprint": contract["runtime"],
                "verification_type": "extracted_upstream_function_cpu_check",
                "torch": {"cuda_initialized": False, "version": H.FIXED["torch_version"]}, "checks": checks,
                "inputs": copy.deepcopy(H.CPU.INPUTS[fault]),
                "input_binding": H.CPU._go2_motor_binding(root) if fault == "F2" else None,
                "scope": {"extracted_target_functions_executed": True,
                          "full_sdk_modules_imported": False, "gpu_used": False,
                          "isaac_app_started": False, "performance_measurement": False,
                          "physx_started": False},
                "correction_readback": {
                    "schema": "correction_readback_v1", "selected_correction": correction,
                    "before_fault": clean, "after_fault": injected, "after_correction": after,
                    "changed_targets": changed, "restored_targets": changed,
                },
                "outputs": {"clean": [1], "fault": [0],
                            "corrected": [1] if matching else [0]},
                "raw_source_hashes": raw_hashes,
            }
            (folder / f"{fault.lower()}_{suffix}.json").write_text(
                json.dumps(row), encoding="utf-8")
    return folder


def test_cpu_checks_accepts_six_bound_passes(monkeypatch, tmp_path):
    contract = _cpu_contract()
    folder = _write_cpu_matrix(tmp_path, contract)
    monkeypatch.setattr(H, "ROOT", tmp_path)
    result = H.cpu_checks(folder, contract, tmp_path)
    assert result["folder"] == "bench/private/cpu_matrix"
    assert result["passed_count"] == 6
    assert set(result["sha256"]) == {
        "f1_match.json", "f1_wrong.json", "f2_match.json",
        "f2_wrong.json", "f6_match.json", "f6_wrong.json",
    }


def test_cpu_checks_rejects_missing_file(monkeypatch, tmp_path):
    contract = _cpu_contract()
    folder = _write_cpu_matrix(tmp_path, contract)
    (folder / "f6_wrong.json").unlink()
    monkeypatch.setattr(H, "ROOT", tmp_path)
    with pytest.raises(FileNotFoundError):
        H.cpu_checks(folder, contract, tmp_path)


@pytest.mark.parametrize("mutation", ["failed", "source_contract", "wrong_behavior", "inputs", "motor_binding", "torch_version"])
def test_cpu_checks_rejects_failed_unbound_or_behaviorally_wrong_evidence(
        monkeypatch, tmp_path, mutation):
    contract = _cpu_contract()
    folder = _write_cpu_matrix(tmp_path, contract)
    filename = "f1_wrong.json" if mutation == "wrong_behavior" else "f2_match.json" if mutation == "motor_binding" else "f1_match.json"
    path = folder / filename
    row = json.loads(path.read_text(encoding="utf-8"))
    if mutation == "failed":
        row["passed"] = False
    elif mutation == "source_contract":
        row["source_contract"]["runtime"]["cache_tag"] = "changed"
    elif mutation == "inputs":
        row["inputs"]["std"] = 1.0
    elif mutation == "motor_binding":
        row["input_binding"]["source_sha256"] = "9" * 64
    elif mutation == "torch_version":
        row["torch"]["version"] = "2.8.0"
    else:
        row["outputs"]["corrected"] = row["outputs"]["clean"]
    path.write_text(json.dumps(row), encoding="utf-8")
    monkeypatch.setattr(H, "ROOT", tmp_path)
    with pytest.raises(ValueError):
        H.cpu_checks(folder, contract, tmp_path)


def test_verify_protocol_rejects_bound_source_hash_change(monkeypatch, tmp_path):
    source_map = {"source_files": {"a": SHA}, "sdk_sources": {}, "rsl_sources": {}}
    protocol = {**copy.deepcopy(H.FIXED), **source_map,
                "correction_source_contract": {}, "fixed_eval_sha256_lf": SHA,
                "resume_inputs": {"summary_sha256_lf": SHA, "bands": BANDS, "runs": {}},
                "cpu_checks": {"folder": "bench/private/cpu_matrix", "sha256": {},
                               "passed_count": 6}}
    monkeypatch.setattr(H, "read", lambda path: protocol)
    monkeypatch.setattr(H, "check_contract", lambda contract, isaaclab: None)
    monkeypatch.setattr(H, "source_maps", lambda isaaclab: {
        "source_files": {"a": OTHER_SHA}, "sdk_sources": {}, "rsl_sources": {}})
    monkeypatch.setattr(H, "source_inputs", lambda isaaclab: protocol["resume_inputs"])
    monkeypatch.setattr(H, "sha", lambda path, lf=False: SHA)
    with pytest.raises(ValueError, match="source hashes changed"):
        H.verify_protocol(tmp_path)


def test_gpu_busy_returns_before_sdk_launch(monkeypatch, tmp_path):
    isaaclab = tmp_path / "IsaacLab"
    (isaaclab / "logs/rsl_rl/unitree_go2_flat").mkdir(parents=True)
    output = tmp_path / "private" / "busy_tag"
    monkeypatch.setattr(H, "verify_protocol", lambda root: ({"fixed": True}, OTHER_SHA))
    monkeypatch.setattr(H, "folder_for", lambda tag: output)
    monkeypatch.setattr(H, "PRIVATE", tmp_path / "private")
    monkeypatch.setattr(H, "sha", lambda path, lf=False: SHA)
    monkeypatch.setattr(H.M, "query_gpu", lambda profile: {"busy": True})
    monkeypatch.setattr(H.P, "run_owned", lambda *args, **kwargs: pytest.fail("SDK process launched"))
    result = H.execute("busy_tag", "case_01", isaaclab)
    assert result == {"status": "gpu_busy_no_launch", "sdk_jobs_launched": 0}
    assert not output.exists()
