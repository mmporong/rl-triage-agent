"""복합 결함 학습·검사 진입점의 CPU 계약을 검증한다."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "bench"), str(ROOT / "evals")]
import residual_faults_v1 as F  # noqa: E402
import residual_probe_v1 as P  # noqa: E402
import residual_training_v1 as T  # noqa: E402


@pytest.mark.parametrize(("fault", "correction"), [("F1", "restore_body_frame_reward"),
                                                    ("R1", "restore_configured_ppo_epochs")])
def test_single_fault_first_repair_has_matching_manifest(tmp_path, fault, correction):
    value = _manifest(tmp_path, "FIRST")
    value.update(original_faults=[fault], applied_corrections=[correction])
    assert T.validate_manifest(value) == value
    value["applied_corrections"] = [next(key for key in F.CORRECTIONS if key != correction)]
    with pytest.raises(ValueError, match="arm"):
        T.validate_manifest(value)


def _manifest(tmp_path: Path, arm: str = "COMPOUND", mode: str = "train") -> dict:
    states = {
        "NONE": ([], []), "F1": (["F1"], []), "R1": (["R1"], []),
        "COMPOUND": (["F1", "R1"], []),
        "FIRST": (["F1", "R1"], ["restore_body_frame_reward"]),
        "BOTH": (["F1", "R1"], ["restore_body_frame_reward", "restore_configured_ppo_epochs"]),
    }
    faults, corrections = states[arm]
    needs_source = mode == "inspection" or arm in {"FIRST", "BOTH"}
    sources = {name: "c" * 64 for name in T.MINIMUM_BOUND_SOURCES}
    return {
        "schema": T.SCHEMA, "mode": mode, "arm": arm, "seed": 2120,
        "run_name": f"residual_{arm.lower()}_s2120", "task": T.TASK,
        "num_envs": 4096, "max_iterations": T.MODES[mode],
        "configured_learning_epochs": 5, "configured_mini_batches": 4,
        "runtime_updates_to_observe": 3, "source_commit_sha": "a" * 40,
        "protocol_path": "bench/protocols/residual_loop_v1.json",
        "protocol_sha256_lf": "b" * 64, "source_sha256_lf": sources,
        "runtime_source_sha256_lf": {"scripts/reinforcement_learning/rsl_rl/train.py": "d" * 64},
        "records_path": str((tmp_path / "records.jsonl").resolve()),
        "readback_path": str((tmp_path / "readback.json").resolve()),
        "source_checkpoint_sha256": "e" * 64 if needs_source else None,
        "original_faults": faults, "applied_corrections": corrections,
    }


class PPO:
    def __init__(self):
        self.num_learning_epochs = 5
        self.num_mini_batches = 4
        self.schedule = "adaptive"
        self.desired_kl = 0.01


def _body(*_args):
    return "body"


@pytest.mark.parametrize(("faults", "corrections", "remaining", "epochs"), [
    ([], [], [], 5),
    (["F1"], [], ["F1"], 5),
    (["R1"], [], ["R1"], 1),
    (["F1", "R1"], [], ["F1", "R1"], 1),
    (["F1", "R1"], ["restore_body_frame_reward"], ["R1"], 1),
    (["F1", "R1"], ["restore_configured_ppo_epochs"], ["F1"], 5),
    (["F1", "R1"], ["restore_body_frame_reward", "restore_configured_ppo_epochs"], [], 5),
    (["F1", "R1"], ["restore_configured_ppo_epochs", "restore_body_frame_reward"], [], 5),
])
def test_fault_and_correction_states_preserve_other_fault(faults, corrections, remaining, epochs):
    original = _body.__code__
    try:
        reward = F.configure_reward(_body, faults, corrections)
        ppo = PPO()
        optimizer = F.configure_ppo(ppo, faults, corrections)
        assert list(F.remaining_faults(faults, corrections)) == remaining
        assert reward["fault_present_after"] is ("F1" in remaining)
        assert optimizer["fault_present_after"] is ("R1" in remaining)
        assert ppo.num_learning_epochs == epochs
    finally:
        _body.__code__ = original


def test_none_is_noop_and_cpu_matrix_is_complete():
    original = _body.__code__
    reward = F.configure_reward(_body, [], [])
    assert _body.__code__ is original and reward["before_code_sha256"] == reward["after_code_sha256"]
    report = T.cpu_check()
    assert report["status"] == "pass" and report["cases"] == 20
    assert report["order_independent_both"] and report["none_noop"] and report["other_fault_preserved"]
    assert set(report["source_sha256_lf"]) == T.MINIMUM_BOUND_SOURCES


@pytest.mark.parametrize("arm", T.ARMS)
def test_manifest_accepts_registered_arm_states(tmp_path, arm):
    assert T.validate_manifest(_manifest(tmp_path, arm))["arm"] == arm


def test_manifest_rejects_arm_mismatch_duplicate_correction_and_bad_checkpoint_lineage(tmp_path):
    value = _manifest(tmp_path)
    value["applied_corrections"] = ["restore_body_frame_reward"]
    with pytest.raises(ValueError, match="arm과"):
        T.validate_manifest(value)
    value = _manifest(tmp_path, "FIRST")
    value["applied_corrections"] *= 2
    with pytest.raises(ValueError, match="서로 다른"):
        T.validate_manifest(value)
    value = _manifest(tmp_path, "FIRST")
    value["source_checkpoint_sha256"] = None
    with pytest.raises(ValueError, match="checkpoint"):
        T.validate_manifest(value)


def test_manifest_preserves_registered_correction_order(tmp_path):
    first = _manifest(tmp_path, "FIRST")
    first["applied_corrections"] = ["restore_configured_ppo_epochs"]
    assert T.validate_manifest(first)["applied_corrections"] == ["restore_configured_ppo_epochs"]
    both = _manifest(tmp_path, "BOTH")
    both["applied_corrections"].reverse()
    assert T.validate_manifest(both)["applied_corrections"][0] == "restore_configured_ppo_epochs"


def test_readback_computes_remaining_faults_instead_of_trusting_input():
    row = F.receipt(["F1", "R1"], ["restore_body_frame_reward"], {}, {}, {})
    assert row["remaining_faults"] == ["R1"]
    assert "remaining_faults" not in T.REQUIRED


def test_reward_manager_binding_requires_target_identity():
    target = lambda: None
    manager = SimpleNamespace(_term_names=["track", "other"],
                              _term_cfgs=[SimpleNamespace(func=target), SimpleNamespace(func=lambda: None)])
    env = SimpleNamespace(unwrapped=SimpleNamespace(reward_manager=manager))
    assert F.reward_manager_binding(env, target) == {"target_identity_matches": ["track"], "match_count": 1}
    with pytest.raises(ValueError, match="결합되지"):
        F.reward_manager_binding(env, lambda: None)


def test_legacy_probe_jobs_follow_only_remaining_f1(tmp_path):
    jobs = tmp_path / "jobs.json"
    jobs.write_text(json.dumps({"task": T.TASK, "fault": "F1", "jobs": []}), encoding="utf-8")
    assert P.validate_legacy_jobs(jobs, ("F1", "R1"))["fault"] == "F1"
    with pytest.raises(ValueError, match="NONE"):
        P.validate_legacy_jobs(jobs, ("R1",))


def test_optimizer_probe_launches_separate_inspection_process(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(_manifest(tmp_path, "COMPOUND", "inspection")), encoding="utf-8")
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
        observer = T.observer_manifest(manifest_data)
        binding = T.OBSERVE.manifest_binding(observer)
        fingerprint = T.OBSERVE.manifest_fingerprint(observer)
        rows = []
        for index in range(3):
            rows.append({
                "schema": T.OBSERVE.SCHEMA, "manifest_sha256": fingerprint, "binding": binding,
                "update_index": index, "expected": {"optimizer_steps": 20, "minibatches": 20},
                "runtime_configuration": {"learning_epochs": 1, "mini_batches": 4},
                "observed": {"optimizer_steps": 4, "minibatches": 4, "adam_step_before": {},
                             "adam_step_after": {}, "adam_step_delta": {}, "gradient_present_steps": {},
                             "pre_step_gradient_norm": [], "learning_rates": [], "approx_kl": None,
                             "clip_fraction": None},
                "unavailable": {}, "error": None,
            })
        Path(manifest_data["records_path"]).write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        Path(manifest_data["readback_path"]).write_text(json.dumps({
            "schema": "residual_training_readback_v1", "remaining_faults": ["F1", "R1"]}),
            encoding="utf-8")
        return SimpleNamespace(returncode=0, args=argv)

    monkeypatch.setattr(P.subprocess, "run", run)
    report = P.run_optimizer(manifest)
    assert report["status"] == "abnormal" and report["inspection"]["records"] == 3
    assert len(calls) == 1 and calls[0][0][-2:] == ["--manifest", str(manifest.resolve())]
    assert calls[0][1]["encoding"] == "utf-8" and calls[0][1]["errors"] == "replace"


def test_cpu_check_cli_writes_once(tmp_path):
    output = tmp_path / "cpu.json"
    assert T.main(["--cpu-check", "--output", str(output)]) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "pass"
    assert T.main(["--cpu-check", "--output", str(output)]) == 2
