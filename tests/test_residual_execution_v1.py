"""복합 결함 실행 backend를 GPU 없이 검증한다."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "bench"), str(ROOT / "evals")]
import residual_execution_v1 as E  # noqa: E402
import residual_training_v1 as W  # noqa: E402


def _binding(*, iterations=300, faults=None, corrections=None, checkpoint=None):
    return {
        "source_commit_sha": "a" * 40,
        "protocol_path": "bench/protocols/residual_loop_v1.json",
        "protocol_sha256_lf": "b" * 64,
        "source_sha256_lf": {name: "c" * 64 for name in W.MINIMUM_BOUND_SOURCES},
        "runtime_source_sha256_lf": {
            "scripts/reinforcement_learning/rsl_rl/train.py": "d" * 64,
            "source/isaaclab/isaaclab/envs/mdp/rewards.py": "e" * 64,
        },
        "source_checkpoint_sha256": checkpoint,
        "task": E.TASK, "num_envs": 4096, "max_iterations": iterations,
        "configured_learning_epochs": 5, "configured_mini_batches": 4,
        "runtime_updates_to_observe": 3,
        "original_faults": ["F1", "R1"] if faults is None else faults,
        "applied_corrections": [] if corrections is None else corrections,
    }


def _execution():
    return {"job_assigned_before_release": True, "natural_exit": True, "timed_out": False,
            "error_type": None, "exit_code": 0, "cleanup_verified": True,
            "owned_process_identities": {}, "wall_s": 1.0}


def _records(manifest, steps):
    observer = W.observer_manifest(manifest)
    fingerprint = W.OBSERVE.manifest_fingerprint(observer)
    binding = W.OBSERVE.manifest_binding(observer)
    rows = []
    for index in range(3):
        rows.append({
            "schema": W.OBSERVE.SCHEMA, "manifest_sha256": fingerprint, "binding": binding,
            "update_index": index, "expected": {"optimizer_steps": 20, "minibatches": 20},
            "runtime_configuration": {"learning_epochs": steps // 4, "mini_batches": 4},
            "observed": {"optimizer_steps": steps, "minibatches": steps,
                         "adam_step_before": {}, "adam_step_after": {}, "adam_step_delta": {},
                         "gradient_present_steps": {}, "pre_step_gradient_norm": [],
                         "learning_rates": [], "approx_kl": None, "clip_fraction": None},
            "unavailable": {}, "error": None,
        })
    return rows


def _readback(manifest):
    faults = manifest["original_faults"]
    corrections = manifest["applied_corrections"]
    remaining = list(E.FAULTS.remaining_faults(faults, corrections))
    return {
        "schema": "residual_training_readback_v1",
        "original_faults": faults, "applied_corrections": corrections,
        "remaining_faults": remaining,
        "reward": {"target": "isaaclab.envs.mdp.rewards.track_lin_vel_xy_exp",
                   "before_code_sha256": "1" * 64,
                   "after_fault_code_sha256": "2" * 64 if "F1" in faults else "1" * 64,
                   "after_code_sha256": "2" * 64 if "F1" in remaining else "1" * 64,
                   "fault_present_after": "F1" in remaining},
        "optimizer": {"before": {"learning_epochs": 5, "mini_batches": 4},
                      "after_fault": {"learning_epochs": 1 if "R1" in faults else 5,
                                      "mini_batches": 4},
                      "after": {"learning_epochs": 1 if "R1" in remaining else 5,
                                "mini_batches": 4},
                      "fault_present_after": "R1" in remaining},
        "reward_manager_binding": {"target_identity_matches": ["track_lin_vel_xy_exp"],
                                   "match_count": 1},
        "observer_manifest": W.observer_manifest(manifest),
    }


def _isaac(tmp_path):
    root = tmp_path / "IsaacLab"
    (root / "_isaac_sim").mkdir(parents=True)
    (root / "_isaac_sim/python.bat").write_text("", encoding="utf-8")
    (root / "scripts/reinforcement_learning/rsl_rl").mkdir(parents=True)
    (root / "scripts/reinforcement_learning/rsl_rl/train.py").write_text("", encoding="utf-8")
    (root / "source/isaaclab/isaaclab/envs/mdp").mkdir(parents=True)
    (root / "source/isaaclab/isaaclab/envs/mdp/rewards.py").write_text("", encoding="utf-8")
    (root / "logs/rsl_rl/unitree_go2_flat").mkdir(parents=True)
    return root


def _fake_owned(isaac, run_name, seed):
    def owned(_folder, argv, _cwd, _watchdog):
        if "residual_training_v1.py" in " ".join(argv):
            manifest = json.loads(Path(argv[-1]).read_text(encoding="utf-8"))
            remaining = E.FAULTS.remaining_faults(manifest["original_faults"], manifest["applied_corrections"])
            steps = 4 if "R1" in remaining else 20
            Path(manifest["records_path"]).write_text(
                "".join(json.dumps(row) + "\n" for row in _records(manifest, steps)), encoding="utf-8")
            Path(manifest["readback_path"]).write_text(json.dumps(_readback(manifest)), encoding="utf-8")
            run = isaac / "logs/rsl_rl/unitree_go2_flat" / f"2026-10-17_12-00-00_{run_name}"
            (run / "params").mkdir(parents=True)
            iterations = manifest["max_iterations"]
            (run / "params/agent.yaml").write_text(
                f"seed: {seed}\nmax_iterations: {iterations}\nalgorithm:\n  num_learning_epochs: 5\n  num_mini_batches: 4\n",
                encoding="utf-8")
            (run / "model_0.pt").write_bytes(b"first")
            (run / f"model_{iterations - 1}.pt").write_bytes(b"final")
        elif "--inspect-checkpoint" in argv:
            checkpoint = Path(argv[argv.index("--inspect-checkpoint") + 1])
            output = Path(argv[argv.index("--output") + 1])
            output.write_text(json.dumps({"device": "cpu", "first": {"iteration": 0},
                "final": {"iteration": int(checkpoint.stem.split("_")[-1]),
                          "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}}), encoding="utf-8")
        elif "extract_tb.py" in " ".join(argv):
            Path(argv[-1]).write_text(json.dumps({"summary": {"reward": {"last": 1.0}}}), encoding="utf-8")
        return _execution()
    return owned


def test_manifest_requires_runtime_reward_source_and_preserves_masks(tmp_path):
    manifest = E._manifest(_binding(), arm="COMPOUND", seed=2120, run_name="compound_s2120",
                           mode="train", records=tmp_path / "r.jsonl", readback=tmp_path / "b.json")
    assert manifest["original_faults"] == ["F1", "R1"] and manifest["applied_corrections"] == []
    bad = _binding()
    del bad["runtime_source_sha256_lf"]["source/isaaclab/isaaclab/envs/mdp/rewards.py"]
    with pytest.raises(ValueError, match="rewards.py"):
        E._verify_binding(E._manifest(bad, arm="COMPOUND", seed=2120, run_name="compound_s2120",
                                     mode="train", records=tmp_path / "x", readback=tmp_path / "y"), tmp_path)


def test_train_returns_mask_receipt_checkpoint_and_telemetry(tmp_path, monkeypatch):
    isaac = _isaac(tmp_path)
    monkeypatch.setattr(E, "_verify_binding", lambda *_: None)
    monkeypatch.setattr(E, "_verify_repository_provenance", lambda *_: None)
    monkeypatch.setattr(E.V1, "_snapshot", lambda *_: {"busy": False})
    result = E.train(tmp_path / "stage", isaac, "COMPOUND", 2120, "compound_s2120",
                     _binding(), _fake_owned(isaac, "compound_s2120", 2120))
    assert result["original_faults"] == ["F1", "R1"] and result["remaining_faults"] == ["F1", "R1"]
    assert result["inspection"] == {"status": "abnormal", "optimizer_steps": [4, 4, 4],
                                    "minibatches": [4, 4, 4]}
    assert result["checkpoint_loaded"] is False and result["source_checkpoint_sha256"] is None
    assert result["telemetry"]["summary"]["reward"]["last"] == 1.0
    assert Path(result["manifest_path"]).is_file()
    receipt = E.validate_artifact(result, _binding(), "COMPOUND", 2120)
    assert receipt["valid"] is True and receipt["remaining_faults"] == ["F1", "R1"]
    Path(result["telemetry"]["path"]).write_text(json.dumps({"summary": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="telemetry SHA"):
        E.validate_artifact(result, _binding(), "COMPOUND", 2120)


def test_inspect_uses_three_updates_without_loading_source_checkpoint(tmp_path, monkeypatch):
    isaac = _isaac(tmp_path)
    monkeypatch.setattr(E, "_verify_binding", lambda *_: None)
    monkeypatch.setattr(E.V1, "_snapshot", lambda *_: {"busy": False})
    binding = _binding(iterations=3, checkpoint="f" * 64)
    result = E.inspect(tmp_path / "inspect", isaac, "COMPOUND", 2121, "inspect_s2121", binding,
                       _fake_owned(isaac, "inspect_s2121", 2121))
    assert result["checkpoint_loaded"] is False
    assert result["source_checkpoint_sha256"] == "f" * 64
    assert result["measurement"]["status"] == "abnormal" and "telemetry" not in result


def test_readback_rejects_mask_tamper(tmp_path):
    manifest = E._manifest(_binding(), arm="COMPOUND", seed=2120, run_name="compound_s2120",
                           mode="train", records=tmp_path / "r", readback=tmp_path / "b")
    value = _readback(manifest)
    value["remaining_faults"] = ["F1"]
    with pytest.raises(ValueError, match="잔여 결함"):
        E._validate_readback(value, manifest)


def test_legacy_probe_routes_remaining_f1_through_residual_entry(tmp_path, monkeypatch):
    isaac = _isaac(tmp_path)
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"policy")
    manifest = tmp_path / "source_manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    artifact = {"arm": "COMPOUND", "seed": 2120,
                "original_faults": ["F1", "R1"], "applied_corrections": [],
                "manifest_path": str(manifest),
                "checkpoint": {"path": str(checkpoint), "sha256": E.sha(checkpoint)}}
    monkeypatch.setattr(E.V1, "_verify_runtime", lambda *_: None)
    monkeypatch.setattr(E.V1, "_snapshot", lambda *_: {"busy": False})
    calls = []

    def owned(_folder, argv, _cwd, _watchdog):
        calls.append(argv)
        jobs = json.loads(Path(argv[argv.index("--jobs") + 1]).read_text(encoding="utf-8"))
        job = jobs["jobs"][0]
        Path(job["output"]).write_text(json.dumps({
            "name": job["name"], "task": E.TASK, "seed": jobs["seed"],
            "checkpoint": {"sha256": E.sha(checkpoint)},
            "probes": {"P_reward": {"measurement": {}, "exit": 0}}}), encoding="utf-8")
        return _execution()

    result = E.legacy_probe(tmp_path / "probe", isaac, artifact, "probe_s2120", _binding(), owned,
                            probes=["P_reward"])
    assert result["fault"] == "F1" and result["remaining_faults"] == ["F1", "R1"]
    assert "residual_probe_v1.py" in " ".join(calls[0]) and "evals/probes_task.py" not in " ".join(calls[0])
