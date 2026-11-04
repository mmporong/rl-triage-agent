from __future__ import annotations

import hashlib
import json
import subprocess
import sys
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


def _environment(tmp_path: Path, *, training_seed: int = 5556,
                 reward_fingerprint: str = "3" * 64) -> dict:
    target = {**CHILD.TARGET, "source_sha256_lf": "1" * 64,
              "clean_fingerprint": "2" * 64,
              "expected_runtime_fingerprint": reward_fingerprint}
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
        "task": CHILD.BASE.TASK, "source_arm": "COMPOUND", "source_seed": 2556,
        "training_seed": training_seed,
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

    names = []

    def base_train(stage_arg, isaaclab, arm, seed, run_name, binding, owned_adapter):
        names.append(run_name)
        stage_arg.mkdir()
        base = str((SUBJECT.ROOT / "bench/r1_repair_training.py").resolve())
        owned_adapter(stage_arg / "owned", [sys.executable, "-B", base, "--manifest", "m.json"], isaaclab, 1)
        owned_adapter(stage_arg / "checkpoint", [sys.executable, "-B", base, "--inspect-checkpoint", "cp"], isaaclab, 1)
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
    assert names == [SUBJECT.training_run_name("run", 5556,
                                               binding_full["environment_preservation"])]
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


def test_training_run_name_binds_request_seed_and_environment(tmp_path: Path):
    request_id = "785427445a0a"
    first = _environment(tmp_path / "first")
    same = SUBJECT.training_run_name(request_id, 5556, first)
    assert same == SUBJECT.training_run_name(request_id, 5556, first)
    assert same.startswith(request_id + "_s5556_") and len(same) <= SUBJECT.RUN_NAME_LIMIT

    other_seed = _environment(tmp_path / "other-seed", training_seed=5557)
    other_recipe = _environment(tmp_path / "other-recipe", reward_fingerprint="8" * 64)
    assert SUBJECT.training_run_name(request_id, 5557, other_seed) != same
    assert SUBJECT.training_run_name(request_id, 5556, other_recipe) != same


def _repository_bound_training(tmp_path: Path, isaaclab: Path) -> dict:
    protocol = SUBJECT.ROOT / "bench/protocols/r1_repair_demo_v1.json"
    sources = {
        relative: CHILD.BASE.sha256_lf(SUBJECT.ROOT / relative)
        for relative in CHILD.BASE.MINIMUM_BOUND_SOURCES
    }
    runtime_relative = "scripts/reinforcement_learning/rsl_rl/train.py"
    runtime = isaaclab / runtime_relative
    runtime.parent.mkdir(parents=True, exist_ok=True)
    runtime.write_text("# namespace regression runtime\n", encoding="utf-8")
    return {
        "source_commit_sha": subprocess.run(
            ["git", "-C", str(SUBJECT.ROOT), "rev-parse", "HEAD"], check=True,
            capture_output=True, text=True, encoding="utf-8").stdout.strip(),
        "protocol_path": protocol.relative_to(SUBJECT.ROOT).as_posix(),
        "protocol_sha256_lf": CHILD.BASE.sha256_lf(protocol),
        "source_sha256_lf": sources,
        "runtime_source_sha256_lf": {
            runtime_relative: CHILD.BASE.sha256_lf(runtime),
        },
        "source_checkpoint_sha256": "9" * 64,
        "task": SUBJECT.V1.TASK,
        "num_envs": 4096,
        "max_iterations": 300,
        "configured_learning_epochs": 5,
        "configured_mini_batches": 4,
        "runtime_updates_to_observe": 3,
    }


def test_real_v1_training_collision_uses_verified_namespace_before_sdk(monkeypatch,
                                                                      tmp_path: Path):
    class SdkBoundaryReached(RuntimeError):
        pass

    request_id = "785427445a0a"
    isaaclab = tmp_path / "IsaacLab"
    python_entry = isaaclab / "_isaac_sim/python.bat"
    python_entry.parent.mkdir(parents=True)
    python_entry.write_text("@exit /b 99\n", encoding="utf-8")
    binding = {"environment_preservation": _environment(tmp_path / "environment"),
               **_repository_bound_training(tmp_path, isaaclab)}
    log_root = isaaclab / "logs/rsl_rl/unitree_go2_flat"
    (log_root / f"2026-11-04_12-00-00_{request_id}").mkdir(parents=True)
    monkeypatch.setattr(SUBJECT.V1, "_query_gpu", lambda: {"busy": False})
    reached = []

    def stop_before_sdk(folder, argv, cwd, watchdog):
        reached.append((Path(folder), list(argv), Path(cwd), watchdog))
        raise SdkBoundaryReached("SDK process was not started")

    stage = tmp_path / "attempt-one" / "training"
    with pytest.raises(SdkBoundaryReached, match="not started"):
        SUBJECT.train(stage, isaaclab, "CORRECTED", 5556, request_id, binding,
                      stop_before_sdk)
    manifest = json.loads((stage / "manifest.json").read_text(encoding="utf-8"))
    namespaced = SUBJECT.training_run_name(request_id, 5556,
                                           binding["environment_preservation"])
    assert manifest["run_name"] == namespaced != request_id
    assert len(reached) == 1 and reached[0][1][2].endswith(
        "registered_epoch_recovery_training_v1.py")

    (log_root / f"2026-11-04_12-01-00_{namespaced}").mkdir()
    with pytest.raises(FileExistsError, match="같은 run_name"):
        SUBJECT.train(tmp_path / "attempt-two" / "training", isaaclab, "CORRECTED",
                      5556, request_id, binding, stop_before_sdk)
    assert len(reached) == 1


def test_owned_adapter_rejects_different_manifest_entry(tmp_path: Path):
    state = {"training_replacements": 0}
    adapter = SUBJECT._owned_adapter(lambda *args: {}, tmp_path / "binding.json", state)
    with pytest.raises(ValueError, match="unexpected"):
        adapter(tmp_path, ["python", "-B", "other.py", "--manifest", "m.json"], tmp_path, 1)


def test_default_owned_adapter_prepares_directory_before_production_runner(monkeypatch, tmp_path: Path):
    class LaunchBoundaryReached(RuntimeError):
        pass

    binding = tmp_path / "binding.json"
    state = {"training_replacements": 0}
    work = tmp_path / "training" / "owned"
    work.parent.mkdir()
    base = str((SUBJECT.ROOT / "bench/r1_repair_training.py").resolve())
    seen = []

    def production_runner(folder, argv, cwd, watchdog):
        # The owned runner opens launch.json before starting its child.
        with (folder / "launch.json").open("x", encoding="utf-8") as stream:
            json.dump({"argv": argv}, stream)
        seen.append((folder, argv, cwd, watchdog))
        raise LaunchBoundaryReached("SDK child not started")

    monkeypatch.setattr(SUBJECT.V1.OWNED, "run_owned", production_runner)
    adapter = SUBJECT._owned_adapter(None, binding, state)
    with pytest.raises(LaunchBoundaryReached, match="SDK child not started"):
        adapter(work, [sys.executable, "-B", base, "--manifest", "m.json"], tmp_path, 10)
    assert work.is_dir()
    assert state == {"training_replacements": 1}
    assert seen[0][1] == [sys.executable, "-B", str(Path(CHILD.__file__).resolve()),
                           "--manifest", "m.json", "--environment-binding", str(binding)]


def test_default_owned_adapter_runs_real_cpu_child_in_fresh_directory(tmp_path: Path):
    work = tmp_path / "owned"
    adapter = SUBJECT._owned_adapter(None, tmp_path / "binding.json", {"training_replacements": 0})
    result = adapter(work, [sys.executable, "-c", "print('epoch-owned-cpu-boundary')"], tmp_path, 15)
    assert result["exit_code"] == 0
    assert result["cleanup_verified"] is True
    assert "epoch-owned-cpu-boundary" in (work / "stdout.log").read_text(encoding="utf-8")
