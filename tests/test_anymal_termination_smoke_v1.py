from copy import deepcopy
import os
from pathlib import Path

import pytest

from evals import anymal_termination_smoke_v1 as S


def report(xy=0.1, fall=0.0):
    return {"metrics": {"lin_vel_rmse_mps": xy, "yaw_rate_rmse_radps": 0.1,
                        "mean_survival_s": 20.0, "fall_rate": fall},
            "horizon_steps": 1000, "step_dt": 0.02, "standstill": {"lin_vel_rmse_mps": 1.0}}


def test_registered_band_is_reused_without_recalibration():
    inputs = S.band_inputs()
    assert inputs["binding"]["outcome"] == "pass"
    assert inputs["bands"] == S.read(S.ROOT / S.E.STAGE1_RESULT)["analysis"]["bands"]


@pytest.mark.parametrize("none,fault,outcome", [
    (report(), report(0.8), "failure_present"),
    (report(), report(0.1), "source_failure_absent"),
    (report(0.9), report(1.0), "baseline_invalid"),
])
def test_only_behavior_failure_with_walking_control_allows_next_stage(none, fault, outcome):
    result = S.E.E4.source_gate(none, fault, S.band_inputs()["bands"])
    assert result["outcome"] == outcome


def test_protocol_edits_fail_before_gpu(monkeypatch, tmp_path):
    protocol = {**deepcopy(S.FIXED), "cpu_semantics": {"applicable": True}}
    protocol["seed"] += 1
    path = tmp_path / "protocol.json"
    S.write_new(path, protocol)
    monkeypatch.setattr(S, "PROTOCOL", path)
    monkeypatch.setattr(S, "dynamic", lambda _: {})
    monkeypatch.setattr(S.GPU, "query_gpu", lambda _: pytest.fail("GPU admission must not run"))
    with pytest.raises(ValueError, match="사전등록"):
        S.execute(tmp_path, S.TAG)


def test_uncommitted_inputs_are_rejected_before_seed_consumption(monkeypatch, tmp_path):
    def reject(*args):
        raise ValueError("uncommitted")
    monkeypatch.setattr(S, "check", reject)
    monkeypatch.setattr(S.SEEDS, "assert_fresh_seeds", lambda *a, **k: pytest.fail("seed consumed"))
    with pytest.raises(ValueError, match="uncommitted"):
        S.execute(tmp_path, S.TAG)


def test_consumed_run_cannot_be_reused(monkeypatch, tmp_path):
    folder = tmp_path / "run"
    folder.mkdir()
    monkeypatch.setattr(S, "check", lambda *a: ({}, "commit"))
    monkeypatch.setattr(S, "folder_for", lambda _: folder)
    monkeypatch.setattr(S.SEEDS, "assert_fresh_seeds", lambda *a, **k: pytest.fail("freshness run after consumption"))
    with pytest.raises(FileExistsError):
        S.execute(tmp_path, S.TAG)


def test_training_failure_restores_fault_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("TRIAGE_FAULT", "earlier_value")
    monkeypatch.setattr(S.G, "find_run_dir", lambda *a: [])
    monkeypatch.setattr(S, "admit", lambda _: None)
    def failure(stage, argv, isaaclab, timeout):
        assert os.environ["TRIAGE_FAULT"] == "F6"
        assert argv[argv.index("--task") + 1] == "Isaac-Velocity-Flat-Anymal-C-v0"
        assert argv[argv.index("--seed") + 1] == "2100"
        raise RuntimeError("child failed")
    monkeypatch.setattr(S, "run", failure)
    with pytest.raises(RuntimeError, match="child failed"):
        S.train_arm(tmp_path, "F6", tmp_path, S.FIXED)
    assert os.environ["TRIAGE_FAULT"] == "earlier_value"


@pytest.mark.parametrize("key,value", [("exit_code", 1), ("natural_exit", False),
    ("cleanup_verified", False), ("timed_out", True), ("job_assigned_before_release", False)])
def test_invalid_owned_execution_is_not_valid(key, value):
    row = {"exit_code": 0, "natural_exit": True, "cleanup_verified": True,
           "timed_out": False, "error_type": None, "job_assigned_before_release": True}
    assert S.finished(row)
    row[key] = value
    assert not S.finished(row)


def test_missing_or_changed_cpu_meaning_cannot_prepare(monkeypatch, tmp_path):
    monkeypatch.setattr(S, "PROTOCOL", tmp_path / "protocol.json")
    monkeypatch.setattr(S, "dynamic", lambda _: {})
    source = tmp_path / "cpu.json"
    S.write_new(source, {"outcome": "not_applicable"})
    with pytest.raises(ValueError):
        S.prepare(tmp_path, source)
    assert not S.PROTOCOL.exists()


def test_tag_cannot_escape_or_choose_another_seed():
    for tag in ("../escape", S.TAG + "_retry", "anymal_f6_smoke_s2101_20261017"):
        with pytest.raises(ValueError):
            S.folder_for(tag)


def telemetry_pair():
    return {arm: {"series": {"Train/mean_episode_length": [length] * 300,
                              "Episode_Termination/time_out": [0.9] * 300}}
            for arm, length in (("NONE", 900.0), ("F6", 50.0))}


def test_registered_termination_signature_checks_training_observation():
    result = S.injection_signature(telemetry_pair())
    assert all(result["checks"].values())
    assert result["f6_episode_steps_max"] == 50.0


@pytest.mark.parametrize("arm,key,value", [
    ("F6", "Train/mean_episode_length", 1000.0),
    ("F6", "Episode_Termination/time_out", 0.0),
    ("NONE", "Train/mean_episode_length", 50.0),
    ("F6", "Train/mean_episode_length", float("nan")),
])
def test_fault_name_alone_is_insufficient_runtime_evidence(arm, key, value):
    rows = telemetry_pair()
    rows[arm]["series"][key] = [value] * 300
    with pytest.raises(ValueError):
        S.injection_signature(rows)


@pytest.mark.parametrize("mutation", [None, "failed", "missing_run", "cwd", "missing_receipt"])
def test_cpu_phase_receipts_are_replayed_and_reject_tampering(tmp_path, mutation):
    stage = tmp_path / "cpu"
    isaac = tmp_path / "isaac"
    execution = {"exit_code": 0, "natural_exit": True, "cleanup_verified": True,
                 "timed_out": False, "error_type": None, "job_assigned_before_release": True}
    launch = {"args": S.cpu_command(isaac, stage), "cwd": str(isaac)}
    if mutation == "failed": execution["exit_code"] = 1
    elif mutation == "missing_run": launch["args"].remove("run")
    elif mutation == "cwd": launch["cwd"] = str(tmp_path)
    if mutation != "missing_receipt": S.write_new(stage / "execution.json", execution)
    S.write_new(stage / "launch.json", launch)
    if mutation is None:
        S.validate_cpu_execution(stage, isaac)
    else:
        with pytest.raises((ValueError, FileNotFoundError)):
            S.validate_cpu_execution(stage, isaac)
