"""PPO ratio 학습 진입점과 private backend를 GPU 없이 검증한다."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "bench"), str(ROOT / "evals")]
import optimizer_observability as O  # noqa: E402
import ppo_gradient_invariant_v2 as G  # noqa: E402
import ppo_ratio_repair_execution as E  # noqa: E402
import ppo_ratio_repair_training as W  # noqa: E402

SOURCES = {name: "c" * 64 for name in sorted(W.MINIMUM_BOUND_SOURCES)}


def _binding(iterations=300, source_checkpoint=None):
    return {
        "source_commit_sha": "a" * 40, "protocol_path": "bench/protocols/ppo_ratio_pilot_v1.json",
        "protocol_sha256_lf": "b" * 64, "source_sha256_lf": dict(SOURCES),
        "runtime_source_sha256_lf": {"scripts/reinforcement_learning/rsl_rl/train.py": "d" * 64},
        "source_checkpoint_sha256": source_checkpoint, "task": E.TASK, "num_envs": 4096,
        "max_iterations": iterations, "configured_learning_epochs": 5, "configured_mini_batches": 4,
        "runtime_updates_to_observe": 3,
    }


def _manifest(tmp_path, arm="NONE", mode="train"):
    binding = _binding(W.MODES[mode], "e" * 64 if arm == "CORRECTED" or mode == "inspection" else None)
    return E._manifest(binding, arm=arm, seed=2160, run_name=f"ratio_{arm.lower()}", mode=mode,
                       records=tmp_path / "records.jsonl", readback=tmp_path / "readback.json")


class PPO:
    num_learning_epochs, num_mini_batches, clip_param = 5, 4, 0.2

    def update(self):
        actions_log_prob_batch = old_actions_log_prob_batch = None
        ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))  # noqa: F821
        return ratio


@pytest.mark.parametrize(("arm", "active", "restored"), [
    ("NONE", False, False), ("INVERSE_RATIO", True, False), ("CORRECTED", False, True)])
def test_configure_ppo_installs_registered_mutation_per_arm_and_restores(arm, active, restored):
    ppo = PPO()
    restore, row = W.configure_ppo(ppo, arm)
    mutation = row["mutation"]
    assert (mutation["active_mutation"], mutation["restored_original"]) == (active, restored)
    assert W.validate_readback(json.loads(json.dumps(row)), arm) == row
    assert ("update" in ppo.__dict__) is (arm != "CORRECTED")
    restore()
    assert "update" not in ppo.__dict__


def test_readback_from_another_arm_or_configuration_is_rejected():
    _, row = W.configure_ppo(PPO(), "INVERSE_RATIO")
    for arm in ("NONE", "CORRECTED"):
        with pytest.raises(ValueError):
            W.validate_readback({**row, "arm": arm}, arm)
    with pytest.raises(ValueError, match="구성"):
        W.validate_readback({**row, "configuration": {**row["configuration"], "learning_epochs": 1}}, "INVERSE_RATIO")
    ppo = PPO()
    ppo.clip_param = 0.3
    with pytest.raises(ValueError, match="등록값"):
        W.configure_ppo(ppo, "NONE")


def test_manifest_is_strict_about_arm_mode_and_checkpoint_lineage(tmp_path):
    assert _manifest(tmp_path, "CORRECTED")["source_checkpoint_sha256"] == "e" * 64
    assert _manifest(tmp_path, "INVERSE_RATIO", "inspection")["max_iterations"] == 3
    with pytest.raises(ValueError, match="3 update"):
        _manifest(tmp_path, "CORRECTED", "inspection")
    manifest = _manifest(tmp_path)
    manifest["source_checkpoint_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="NONE/INVERSE_RATIO"):
        W.validate_manifest(manifest)
    manifest = _manifest(tmp_path)
    manifest["source_sha256_lf"].pop("bench/ppo_ratio_mutation_v1.py")
    with pytest.raises(ValueError, match="binding"):
        W.validate_manifest(manifest)


def test_observer_manifest_fixes_the_v5_invariant_and_three_primitives(tmp_path):
    bound = W.observer_manifest(_manifest(tmp_path, "INVERSE_RATIO"))
    assert bound["gradient_invariant"] == W.INVARIANT and bound["runtime_updates_to_observe"] == 3
    assert bound["inspection_primitives"] == ["optimizer_step_count", "minibatch_count", G.PRIMITIVE]


def _records(manifest, gradient, steps=20):
    bound = W.observer_manifest(manifest)
    summary = {"max_abs_error": 0.0 if gradient == "normal" else 1.0, "mismatch_count": int(gradient != "normal"),
               "nonzero_expected_count": 1, "finite": True, "n": 1}
    return [{"schema": O.SCHEMA, "manifest_sha256": O.manifest_fingerprint(bound),
             "binding": O.manifest_binding(bound), "update_index": index,
             "expected": {"optimizer_steps": 20, "minibatches": 20},
             "observed": {"optimizer_steps": steps, "minibatches": steps,
                          G.PRIMITIVE: {"status": gradient, "batches": [summary] * steps}},
             "error": None} for index in range(3)]


@pytest.mark.parametrize(("gradient", "steps", "count", "status"), [
    ("normal", 20, "normal", "normal"), ("abnormal", 20, "normal", "abnormal"), ("normal", 4, "abnormal", "normal")])
def test_inspection_separates_count_and_gradient_statuses(tmp_path, gradient, steps, count, status):
    manifest = _manifest(tmp_path, "INVERSE_RATIO", "inspection")
    path = tmp_path / "records.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in _records(manifest, gradient, steps)), encoding="utf-8")
    measured = E.inspection(path, manifest)
    assert (measured["count_status"], measured["gradient_status"]) == (count, status)
    assert measured["optimizer_steps"] == [steps] * 3


def test_incomplete_inspection_is_rejected(tmp_path):
    manifest = _manifest(tmp_path, "INVERSE_RATIO", "inspection")
    path = tmp_path / "records.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in _records(manifest, "abnormal")[:2]), encoding="utf-8")
    with pytest.raises(ValueError, match="완결"):
        E.inspection(path, manifest)


def _execution():
    return {"job_assigned_before_release": True, "natural_exit": True, "timed_out": False,
            "error_type": None, "exit_code": 0, "cleanup_verified": True,
            "owned_process_identities": {}, "wall_s": 1.0}


@pytest.mark.parametrize(("arm", "mode"), [("CORRECTED", "train"), ("INVERSE_RATIO", "inspection")])
def test_training_runs_new_entry_and_returns_checkpoint_and_separate_statuses(tmp_path, monkeypatch, arm, mode):
    isaac = tmp_path / "IsaacLab"
    stage = tmp_path / "private" / "stage"
    run_name = f"ratio_{arm.lower()}_{mode}"
    iterations = W.MODES[mode]
    # 준비 runner는 실행 파일과 cwd가 있어야 소유 프로세스를 시작한다.
    (isaac / "_isaac_sim").mkdir(parents=True)
    (isaac / "_isaac_sim/python.bat").write_text("", encoding="utf-8")
    monkeypatch.setattr(E.V1, "_query_gpu", lambda: {"busy": False})
    monkeypatch.setattr(E.WRAPPER, "verify_repository_binding", lambda *_: None)
    monkeypatch.setattr(E.WRAPPER, "verify_runtime_binding", lambda *_: None)
    entries = []

    def owned(folder, argv, cwd, watchdog):
        entries.append(argv[2])
        if "extract_tb.py" in " ".join(argv):
            Path(argv[-1]).write_text(json.dumps({"summary": {"loss": {"mean_last_20pct": 1.0}}}), encoding="utf-8")
        elif "--inspect-checkpoint" in argv:
            final = Path(argv[argv.index("--inspect-checkpoint") + 1])
            Path(argv[argv.index("--output") + 1]).write_text(json.dumps(
                {"device": "cpu", "final": {"iteration": iterations - 1, "sha256": E.sha(final)}}), encoding="utf-8")
        else:
            manifest = json.loads(Path(argv[-1]).read_text(encoding="utf-8"))
            gradient = "abnormal" if arm == "INVERSE_RATIO" else "normal"
            Path(manifest["records_path"]).write_text(
                "".join(json.dumps(row) + "\n" for row in _records(manifest, gradient)), encoding="utf-8")
            Path(manifest["readback_path"]).write_text(json.dumps(W.configure_ppo(PPO(), arm)[1]), encoding="utf-8")
            run = isaac / "logs/rsl_rl/unitree_go2_flat" / f"2026-10-22_12-00-00_{run_name}"
            (run / "params").mkdir(parents=True)
            (run / "params/agent.yaml").write_text(
                f"seed: 2160\nmax_iterations: {iterations}\nalgorithm:\n  num_learning_epochs: 5\n  num_mini_batches: 4\n",
                encoding="utf-8")
            (run / f"model_{iterations - 1}.pt").write_bytes(b"checkpoint")
            (run / "model_0.pt").write_bytes(b"first")
        return _execution()

    call = E.train if mode == "train" else E.inspect
    result = call(stage, isaac, arm, 2160, run_name,
                  _binding(iterations, "e" * 64 if arm == "CORRECTED" or mode == "inspection" else None), owned)
    assert result["checkpoint"]["sha256"] == hashlib.sha256(b"checkpoint").hexdigest()
    expected = "abnormal" if arm == "INVERSE_RATIO" else "normal"
    assert (result["inspection"]["count_status"], result["inspection"]["gradient_status"]) == ("normal", expected)
    assert entries[0].endswith("ppo_ratio_repair_training.py") and entries[1].endswith("ppo_ratio_repair_training.py")
    assert ("telemetry" in result) is (mode == "train") and ("measurement" in result) is (mode == "inspection")
    with pytest.raises(FileExistsError):
        call(stage, isaac, arm, 2160, run_name, _binding(iterations, "e" * 64), owned)


def test_backend_reuses_frozen_evaluation_probe_and_preflight():
    assert E.fixed_evaluate is E.V2.fixed_evaluate and E.legacy_probe is E.V2.legacy_probe
    assert E.preflight is E.V2.preflight
