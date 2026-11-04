from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from bench import registered_epoch_recovery_training_v1 as CHILD
from evals import registered_epoch_recovery_execution_v1 as SUBJECT


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: dict) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return {"path": str(path.resolve()), "sha256": _sha(path)}


def _environment(tmp_path: Path) -> dict:
    target = {**CHILD.TARGET, "source_sha256_lf": "1" * 64,
              "clean_fingerprint": "2" * 64, "expected_runtime_fingerprint": "3" * 64}
    secondary_target = {**CHILD.SECONDARY_TARGET, "source_sha256_lf": "6" * 64,
                        "clean_fingerprint": "7" * 64, "expected_runtime_fingerprint": "7" * 64}
    hidden_sha = CHILD._sha(CHILD.ROOT / "bench/hidden_faults.py", lf=True)
    setup_ref = _write(tmp_path / "source.json", {
        "schema": "source-child-manifest-v1", "arm": "COMPOUND", "task": CHILD.BASE.TASK,
        "seed": 2556, "source_commit_sha": "4" * 40,
        "source_sha256_lf": {"bench/hidden_faults.py": hidden_sha},
        "runtime_source_sha256_lf": {target["source"]: target["source_sha256_lf"],
                                      secondary_target["source"]:
                                      secondary_target["source_sha256_lf"]}})
    env_path = tmp_path / "env.yaml"
    env_path.write_text("rewards:\n  track_lin_vel_xy_exp:\n    weight: 1.0\n", encoding="utf-8")
    env_ref = {"path": str(env_path.resolve()), "sha256": _sha(env_path)}
    contract_ref = _write(tmp_path / "code-contract.json", {
        "schema": "registered_environment_recipe_code_contract_v1",
        "provenance_kind": "compiled_code_contract_not_historical_runtime_observation",
        "runtime": {"implementation": "CPython", "version": [3, 10, 15], "cache_tag": "cpython-310"},
        "reward_targets": {"world_frame_reward": target},
        "motor_targets": {"nominal": secondary_target},
        "module_imports": {"carb": False, "isaaclab": False, "omni": False, "torch": False},
        "gpu_launches": 0, "model_api_calls": 0,
        "hidden_recipe_source_sha256_lf": hidden_sha})
    recipe_ref = _write(tmp_path / "recipe.json", {
        "schema": CHILD.EXPECTED_SCHEMA, "provenance_kind": CHILD.PROVENANCE_KIND,
        "task": CHILD.BASE.TASK, "source_arm": "COMPOUND", "source_seed": 2556, "training_seed": 5556,
        "source_checkpoint_sha256": "5" * 64, "source_setup_sha256": setup_ref["sha256"],
        "source_env_config_sha256": env_ref["sha256"], "mode": "world_frame_reward",
        "code_contract_sha256": contract_ref["sha256"], "target": target,
        "secondary_target": secondary_target})
    return {"schema": CHILD.SCHEMA, "mode": "world_frame_reward", "source_setup": setup_ref,
            "source_env_config": env_ref, "expected_reward": recipe_ref, "target": target,
            "secondary_target": secondary_target, "code_contract": contract_ref}


def test_train_replaces_only_training_child_and_preserves_other_children(monkeypatch, tmp_path: Path):
    stage = tmp_path / "stage"
    seen: list[list[str]] = []

    def owned(folder, argv, cwd, watchdog):
        seen.append(list(argv))
        return {"exit_code": 0}

    def base_train(stage_arg, isaaclab, arm, seed, run_name, binding, owned_adapter):
        stage_arg.mkdir()
        base = str((SUBJECT.ROOT / "bench/r1_repair_training.py").resolve())
        owned_adapter(stage_arg / "owned", ["python", "-B", base, "--manifest", "m.json"], isaaclab, 1)
        owned_adapter(stage_arg / "checkpoint", ["python", "-B", base, "--inspect-checkpoint", "cp"], isaaclab, 1)
        environment = CHILD.validate_environment_binding(binding_full["environment_preservation"])
        (stage_arg / "environment-readback.json").write_text(json.dumps({
            "schema": "registered_epoch_environment_readback_v1", "mode": environment["mode"],
            "source_setup": environment["source_setup"],
            "source_env_config": environment["source_env_config"],
            "expected_reward": environment["expected_reward"],
            "code_contract": environment["code_contract"],
            "provenance_kind": environment["provenance_kind"],
            "runtime_fingerprint": environment["target"]["expected_runtime_fingerprint"],
            "target_identity_preserved": True,
            "motor_runtime_fingerprint": environment["secondary_target"]["expected_runtime_fingerprint"],
            "motor_target_identity_preserved": True, "make_call": 1}), encoding="utf-8")
        initial_ref = {"path": str((stage_arg / "environment-readback.json").resolve()),
                       "sha256": _sha(stage_arg / "environment-readback.json")}
        (stage_arg / "environment-final-readback.json").write_text(json.dumps({
            "schema": "registered_epoch_environment_final_readback_v1", "mode": environment["mode"],
            "runtime_fingerprint": environment["target"]["expected_runtime_fingerprint"],
            "target_identity_preserved": True,
            "motor_runtime_fingerprint": environment["secondary_target"]["expected_runtime_fingerprint"],
            "motor_target_identity_preserved": True, "initial_readback": initial_ref}), encoding="utf-8")
        run_dir = tmp_path / "run"
        (run_dir / "params").mkdir(parents=True)
        (run_dir / "params/env.yaml").write_text(
            json.dumps({"rewards": {"track_lin_vel_xy_exp": {"weight": 1.0}}}), encoding="utf-8")
        checkpoint = tmp_path / "model.pt"
        checkpoint.write_bytes(b"checkpoint")
        return {"arm": arm, "checkpoint": {"path": str(checkpoint.resolve()),
                                             "sha256": _sha(checkpoint)},
                "run_dir": str(run_dir)}

    monkeypatch.setattr(SUBJECT.V1, "train", base_train)
    base_binding = {"placeholder": "base"}
    binding_full = {**base_binding, "environment_preservation": _environment(tmp_path)}
    result = SUBJECT.train(stage, tmp_path, "CORRECTED", 5556, "run", binding_full, owned)

    assert Path(seen[0][2]).name == "registered_epoch_recovery_training_v1.py"
    assert seen[0][-2] == "--environment-binding"
    assert seen[1][2].endswith("r1_repair_training.py")
    verified = SUBJECT.verify_preservation(result["environment_preservation"], result["checkpoint"])
    assert verified["mode"] == "world_frame_reward"
    assert "environment_preservation" not in base_binding
    readback_ref = result["environment_preservation"]["runtime_readback"]
    readback_path = Path(readback_ref["path"])
    tampered = json.loads(readback_path.read_text(encoding="utf-8"))
    tampered["runtime_fingerprint"] = "0" * 64
    readback_path.write_text(json.dumps(tampered), encoding="utf-8")
    readback_ref["sha256"] = _sha(readback_path)
    with pytest.raises(ValueError, match="runtime readback differs"):
        SUBJECT.verify_preservation(result["environment_preservation"], result["checkpoint"])


def test_train_rejects_seed_or_noncorrected_arm(tmp_path: Path):
    binding = {"environment_preservation": _environment(tmp_path)}
    with pytest.raises(ValueError, match="seed"):
        SUBJECT.train(tmp_path / "a", tmp_path, "CORRECTED", 5557, "run", binding)
    with pytest.raises(ValueError, match="corrected"):
        SUBJECT.train(tmp_path / "b", tmp_path, "R1", 5556, "run", binding)


def test_owned_adapter_rejects_different_manifest_entry(tmp_path: Path):
    state = {"training_replacements": 0}
    adapter = SUBJECT._owned_adapter(lambda *args: {}, tmp_path / "binding.json", state)
    with pytest.raises(ValueError, match="unexpected"):
        adapter(tmp_path, ["python", "-B", "other.py", "--manifest", "m.json"], tmp_path, 1)
