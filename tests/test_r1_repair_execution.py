"""R1 교정 wrapper와 private 실행 backend를 GPU 없이 검증한다."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import types

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "bench"), str(ROOT / "evals")]
import r1_repair_execution as E  # noqa: E402
import r1_repair_training as W  # noqa: E402


def _binding(iterations=300, source_checkpoint=None):
    return {
        "source_commit_sha": "a" * 40,
        "protocol_path": "bench/protocols/r1_repair_demo_v1.json",
        "protocol_sha256_lf": "b" * 64,
        "source_sha256_lf": {"bench/r1_repair_training.py": "c" * 64,
                             "bench/optimizer_faults.py": "c" * 64,
                             "evals/optimizer_observability.py": "c" * 64},
        "runtime_source_sha256_lf": {"scripts/reinforcement_learning/rsl_rl/train.py": "d" * 64},
        "source_checkpoint_sha256": source_checkpoint,
        "task": E.TASK, "num_envs": 4096, "max_iterations": iterations,
        "configured_learning_epochs": 5, "configured_mini_batches": 4,
        "runtime_updates_to_observe": 3,
    }


def _manifest(tmp_path, arm="NONE", mode="train"):
    binding = _binding(W.MODES[mode], "e" * 64 if arm == "CORRECTED" or mode == "inspection" else None)
    return E._manifest(binding, arm=arm, seed=2070, run_name=f"r1_{arm.lower()}", mode=mode,
                       records=tmp_path / "records.jsonl", readback=tmp_path / "readback.json")


class PPO:
    def __init__(self):
        self.num_learning_epochs = 5
        self.num_mini_batches = 4
        self.schedule = "adaptive"
        self.desired_kl = 0.01


@pytest.mark.parametrize(("arm", "after", "fault", "restored"), [
    ("NONE", 5, 5, None), ("R1", 1, 1, None), ("CORRECTED", 5, 1, 5),
])
def test_configure_ppo_applies_fault_then_optional_registry_restore(arm, after, fault, restored):
    ppo = PPO()
    receipt = W.configure_ppo(ppo, arm)
    assert receipt["before"] == {"learning_epochs": 5, "mini_batches": 4}
    assert receipt["after_fault"]["learning_epochs"] == fault
    assert receipt["after"]["learning_epochs"] == after == ppo.num_learning_epochs
    assert (receipt["restoration"] or {}).get("runtime_value") == restored


def test_manifest_is_strict_and_correction_is_fresh_restart(tmp_path):
    manifest = _manifest(tmp_path, "CORRECTED")
    assert manifest["max_iterations"] == 300 and manifest["source_checkpoint_sha256"] == "e" * 64
    assert not any(key in manifest for key in ("checkpoint", "load_run", "resume"))
    manifest["extra"] = True
    with pytest.raises(ValueError, match="고정 계약"):
        W.validate_manifest(manifest)


def test_manifest_rejects_path_traversal_and_arm_checkpoint_mix(tmp_path):
    manifest = _manifest(tmp_path)
    manifest["source_sha256_lf"]["../outside.py"] = "f" * 64
    with pytest.raises(ValueError, match="source_sha256"):
        W.validate_manifest(manifest)
    manifest = _manifest(tmp_path)
    manifest["source_checkpoint_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="NONE/R1"):
        W.validate_manifest(manifest)


def test_configure_ppo_rejects_unknown_arm_and_readback_is_revalidated():
    with pytest.raises(ValueError, match="arm"):
        W.configure_ppo(PPO(), "OTHER")
    with pytest.raises(ValueError, match="적용 전"):
        E._validate_readback({"arm": "R1", "before": {"learning_epochs": 1, "mini_batches": 4}}, "R1")


def test_checkpoint_inspection_loads_first_and_final_on_cpu(tmp_path, monkeypatch):
    first, final = tmp_path / "model_0.pt", tmp_path / "model_299.pt"
    first.write_bytes(b"first")
    final.write_bytes(b"final")
    torch = types.ModuleType("torch")
    loads = []
    def load(path, **kwargs):
        loads.append(kwargs)
        return {"iter": 0 if Path(path) == first else 299,
                "model_state_dict": {"weight": 1}, "optimizer_state_dict": {"state": {}}}
    torch.load = load
    monkeypatch.setitem(sys.modules, "torch", torch)
    receipt = W.inspect_checkpoint(final, first, 299)
    assert receipt["device"] == "cpu" and receipt["first"]["iteration"] == 0
    assert receipt["final"]["sha256"] == hashlib.sha256(b"final").hexdigest()
    assert loads == [{"map_location": "cpu", "weights_only": True}] * 2


def _execution():
    return {"job_assigned_before_release": True, "natural_exit": True, "timed_out": False,
            "error_type": None, "exit_code": 0, "cleanup_verified": True,
            "owned_process_identities": {}, "wall_s": 1.0}


def test_train_returns_checkpoint_telemetry_and_separate_first_three_counts(tmp_path, monkeypatch):
    isaac = tmp_path / "IsaacLab"
    stage = tmp_path / "private" / "train"
    run_name = "corrected_s2070"
    monkeypatch.setattr(E, "_query_gpu", lambda: {"busy": False})
    monkeypatch.setattr(E, "_verify_binding", lambda *_: None)
    monkeypatch.setattr(E, "_inspection", lambda *_: {
        "status": "normal", "optimizer_steps": [20, 20, 20], "minibatches": [20, 20, 20],
        "records": [{"update_index": i} for i in range(3)], "report": {"status": "normal", "records": 3},
    })

    def owned(folder, argv, cwd, watchdog):
        folder = Path(folder)
        folder.mkdir(parents=True)
        if "extract_tb.py" in " ".join(argv):
            target = Path(argv[-1])
            target.write_text(json.dumps({"summary": {"loss": {"mean_last_20pct": 1.25}}}), encoding="utf-8")
        elif "--inspect-checkpoint" in argv:
            final = Path(argv[argv.index("--inspect-checkpoint") + 1])
            output = Path(argv[argv.index("--output") + 1])
            output.write_text(json.dumps({"device": "cpu", "first": {"iteration": 0},
                                          "final": {"iteration": 299, "sha256": E.sha(final)}}), encoding="utf-8")
        else:
            manifest = json.loads(Path(argv[-1]).read_text(encoding="utf-8"))
            Path(manifest["records_path"]).write_text("{}\n{}\n{}\n", encoding="utf-8")
            Path(manifest["readback_path"]).write_text(json.dumps({
                "arm": "CORRECTED", "before": {"learning_epochs": 5, "mini_batches": 4},
                "after_fault": {"learning_epochs": 1, "mini_batches": 4},
                "restoration": {"registry_value": 5, "runtime_value": 5},
                "after": {"learning_epochs": 5, "mini_batches": 4},
            }), encoding="utf-8")
            run = isaac / "logs/rsl_rl/unitree_go2_flat" / f"2026-10-16_12-00-00_{run_name}"
            (run / "params").mkdir(parents=True)
            (run / "params/agent.yaml").write_text(
                "seed: 2070\nmax_iterations: 300\nalgorithm:\n  num_learning_epochs: 5\n  num_mini_batches: 4\n",
                encoding="utf-8",
            )
            (run / "model_299.pt").write_bytes(b"checkpoint")
            (run / "model_0.pt").write_bytes(b"first")
        return _execution()

    result = E.train(stage, isaac, "CORRECTED", 2070, run_name, _binding(source_checkpoint="e" * 64), owned)
    assert result["inspection"] == {"status": "normal", "optimizer_steps": [20, 20, 20], "minibatches": [20, 20, 20]}
    assert result["telemetry"]["summary"]["loss"]["mean_last_20pct"] == 1.25
    assert result["checkpoint"]["sha256"] == hashlib.sha256(b"checkpoint").hexdigest()


def test_fixed_eval_and_probe_use_frozen_entries_and_bind_checkpoint(tmp_path, monkeypatch):
    isaac = tmp_path / "IsaacLab"
    checkpoint = tmp_path / "model_299.pt"
    checkpoint.write_bytes(b"policy")
    digest = hashlib.sha256(b"policy").hexdigest()
    artifact = {"arm": "NONE", "seed": 2070, "checkpoint": {"path": str(checkpoint), "sha256": digest}}
    monkeypatch.setattr(E, "_query_gpu", lambda: {"busy": False})
    monkeypatch.setattr(E, "_verify_runtime", lambda *_: None)
    calls = []

    def owned(folder, argv, cwd, watchdog):
        calls.append(argv)
        jobs_path = Path(argv[argv.index("--jobs") + 1])
        jobs = json.loads(jobs_path.read_text(encoding="utf-8"))
        for job in jobs["jobs"]:
            Path(job["output"]).write_text(json.dumps({
                "name": job["name"], "task": E.TASK, "seed": 2070,
                "checkpoint": {"sha256": digest},
                "protocol_sha256_lf": E.sha(E.FIXED_EVAL_PROTOCOL, True),
                "probes": {probe: {"measurement": {}, "exit": 0} for probe in job.get("probes", [])},
            }), encoding="utf-8")
        return _execution()

    fixed = E.fixed_evaluate(tmp_path / "fixed", isaac, {"NONE": artifact}, "eval2070", _binding(), owned)
    probed = E.legacy_probe(tmp_path / "probe", isaac, artifact, "probe2070", _binding(), owned)
    assert fixed["reports"]["NONE"]["checkpoint"]["sha256"] == digest
    assert probed["probes"] == sorted(E.PROBES) and len(probed["probes"]) == 6
    assert any("fixed_eval.py" in " ".join(argv) for argv in calls)
    assert any("probes_task.py" in " ".join(argv) for argv in calls)


def test_fault_probe_requires_exactly_one_approved_selection(tmp_path, monkeypatch):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"x")
    artifact = {"arm": "R1", "seed": 3070,
                "checkpoint": {"path": str(checkpoint), "sha256": hashlib.sha256(b"x").hexdigest()}}
    monkeypatch.setattr(E, "_verify_runtime", lambda *_: None)
    monkeypatch.setattr(E, "_query_gpu", lambda: {"busy": False})
    with pytest.raises(ValueError, match="하나만"):
        E.legacy_probe(tmp_path / "probe", tmp_path / "isaac", artifact, "probe3070", _binding(),
                       lambda *_: _execution(), probes=["P_value", "P_reward"])
