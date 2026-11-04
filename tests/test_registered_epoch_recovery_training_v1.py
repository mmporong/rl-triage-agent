from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import textwrap
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from bench import correction_adapter as ADAPTER
from bench import hidden_faults as FAULTS
from bench import registered_epoch_recovery_training_v1 as SUBJECT
from evals import registered_epoch_recovery_execution_v1 as EXECUTION


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: dict) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return {"path": str(path.resolve()), "sha256": _sha(path)}


def _body_reward(env, std, command_name, asset_cfg=None):
    return env, std, command_name, asset_cfg


_body_reward.__module__ = "isaaclab.envs.mdp.rewards"
_body_reward.__qualname__ = "track_lin_vel_xy_exp"


def _clean_motor(self, effort):
    return effort


_clean_motor.__module__ = "isaaclab.actuators.actuator_pd"
_clean_motor.__qualname__ = "DCMotor._clip_effort"


def _binding(tmp_path: Path, runtime_root: Path, mode: str = "world_frame_reward") -> dict:
    source = runtime_root / SUBJECT.TARGET["source"]
    source.parent.mkdir(parents=True)
    source.write_text("def track_lin_vel_xy_exp(): pass\n", encoding="utf-8")
    motor_source = runtime_root / SUBJECT.SECONDARY_TARGET["source"]
    motor_source.parent.mkdir(parents=True, exist_ok=True)
    motor_source.write_text("class DCMotor: pass\n", encoding="utf-8")
    target = {**SUBJECT.TARGET, "source_sha256_lf": SUBJECT._sha(source, lf=True),
              "clean_fingerprint": ADAPTER.fingerprint(_body_reward),
              "expected_runtime_fingerprint": ADAPTER.fingerprint(
                  _body_reward if mode == "nominal_body_frame" else FAULTS._track_lin_vel_world.__code__)}
    secondary_target = {**SUBJECT.SECONDARY_TARGET,
                        "source_sha256_lf": SUBJECT._sha(motor_source, lf=True),
                        "clean_fingerprint": ADAPTER.fingerprint(_clean_motor),
                        "expected_runtime_fingerprint": ADAPTER.fingerprint(_clean_motor)}
    source_arm = "R1" if mode == "nominal_body_frame" else "COMPOUND"
    hidden_sha = SUBJECT._sha(SUBJECT.ROOT / "bench/hidden_faults.py", lf=True)
    setup = {"schema": "source-child-manifest-v1", "arm": source_arm, "task": SUBJECT.BASE.TASK,
             "seed": 2556, "source_commit_sha": "a" * 40,
             "source_sha256_lf": {"bench/hidden_faults.py": hidden_sha},
             "runtime_source_sha256_lf": {target["source"]: target["source_sha256_lf"],
                                           secondary_target["source"]:
                                           secondary_target["source_sha256_lf"]}}
    setup_ref = _write(tmp_path / "source.json", setup)
    env_path = tmp_path / "env.yaml"
    env_path.write_text("rewards:\n  track_lin_vel_xy_exp:\n    weight: 1.0\n", encoding="utf-8")
    env_ref = {"path": str(env_path.resolve()), "sha256": _sha(env_path)}
    contract_ref = _write(tmp_path / "code-contract.json", {
        "schema": "registered_environment_recipe_code_contract_v1",
        "provenance_kind": "compiled_code_contract_not_historical_runtime_observation",
        "runtime": {"implementation": "CPython", "version": [3, 10, 15], "cache_tag": "cpython-310"},
        "reward_targets": {mode: target},
        "motor_targets": {"nominal": secondary_target,
                          "preserve_source_envelope": secondary_target},
        "module_imports": {"carb": False, "isaaclab": False, "omni": False, "torch": False},
        "gpu_launches": 0, "model_api_calls": 0,
        "hidden_recipe_source_sha256_lf": hidden_sha})
    recipe = {"schema": SUBJECT.EXPECTED_SCHEMA, "provenance_kind": SUBJECT.PROVENANCE_KIND,
              "task": SUBJECT.BASE.TASK, "source_seed": 2556, "training_seed": 5556,
              "source_arm": source_arm,
              "source_checkpoint_sha256": "b" * 64, "source_setup_sha256": setup_ref["sha256"],
              "source_env_config_sha256": env_ref["sha256"], "mode": mode, "target": target,
              "code_contract_sha256": contract_ref["sha256"], "secondary_target": secondary_target}
    return {"schema": SUBJECT.SCHEMA, "mode": mode, "source_setup": setup_ref,
            "source_env_config": env_ref, "expected_reward": _write(tmp_path / "recipe.json", recipe),
            "code_contract": contract_ref, "target": target, "secondary_target": secondary_target}


def _install_fake_runtime(monkeypatch):
    gym = ModuleType("gymnasium")
    rewards = ModuleType("isaaclab.envs.mdp.rewards")
    rewards.__dict__["track_lin_vel_xy_exp"] = _body_reward
    mdp = ModuleType("isaaclab.envs.mdp")
    mdp.__dict__["rewards"] = rewards
    envs = ModuleType("isaaclab.envs")
    envs.__dict__["mdp"] = mdp
    isaaclab = ModuleType("isaaclab")
    isaaclab.__dict__["envs"] = envs
    actuator_pd = ModuleType("isaaclab.actuators.actuator_pd")
    actuator_pd.__dict__["DCMotor"] = type("DCMotor", (), {"_clip_effort": _clean_motor})
    actuators = ModuleType("isaaclab.actuators")
    actuators.__dict__["actuator_pd"] = actuator_pd
    isaaclab.__dict__["actuators"] = actuators
    term = SimpleNamespace(func=_body_reward)
    close = lambda: "closed"
    environment = SimpleNamespace(unwrapped=SimpleNamespace(
        cfg=SimpleNamespace(rewards=SimpleNamespace(track_lin_vel_xy_exp=term))), close=close)
    gym.__dict__["make"] = lambda *args, **kwargs: environment
    runners = ModuleType("rsl_rl.runners")

    class OnPolicyRunner:
        def learn(self, *args, **kwargs):
            return args, kwargs

    runners.__dict__["OnPolicyRunner"] = OnPolicyRunner
    rsl_rl = ModuleType("rsl_rl")
    rsl_rl.__dict__["runners"] = runners
    for name, module in {"gymnasium": gym, "isaaclab": isaaclab, "isaaclab.envs": envs,
                         "isaaclab.envs.mdp": mdp,
                         "isaaclab.envs.mdp.rewards": rewards, "isaaclab.actuators": actuators,
                         "isaaclab.actuators.actuator_pd": actuator_pd, "rsl_rl": rsl_rl,
                         "rsl_rl.runners": runners}.items():
        monkeypatch.setitem(sys.modules, name, module)
    return gym, rewards, environment


def test_environment_scope_preserves_world_reward_and_restores(monkeypatch, tmp_path: Path):
    runtime = tmp_path / "runtime"
    binding = _binding(tmp_path, runtime)
    gym, rewards, environment = _install_fake_runtime(monkeypatch)
    original_make, original_code = gym.make, rewards.track_lin_vel_xy_exp.__code__
    monkeypatch.chdir(runtime)

    readback = tmp_path / "readback.json"
    with SUBJECT.environment_scope(binding, readback):
        assert gym.make("task") is environment
        assert ADAPTER.fingerprint(rewards.track_lin_vel_xy_exp) == binding["target"]["expected_runtime_fingerprint"]

    assert gym.make is original_make
    assert rewards.track_lin_vel_xy_exp.__code__ is original_code
    row = json.loads(readback.read_text(encoding="utf-8"))
    assert row["mode"] == "world_frame_reward"
    assert row["target_identity_preserved"] is True
    assert row["provenance_kind"] == SUBJECT.PROVENANCE_KIND


def test_environment_scope_nominal_keeps_body_reward(monkeypatch, tmp_path: Path):
    runtime = tmp_path / "runtime"
    binding = _binding(tmp_path, runtime, "nominal_body_frame")
    gym, rewards, _ = _install_fake_runtime(monkeypatch)
    monkeypatch.chdir(runtime)
    with SUBJECT.environment_scope(binding, tmp_path / "readback.json"):
        gym.make("task")
        assert rewards.track_lin_vel_xy_exp.__code__ is _body_reward.__code__


def test_environment_scope_preserves_f2_motor_recipe(monkeypatch, tmp_path: Path):
    runtime = tmp_path / "runtime"
    binding = _binding(tmp_path, runtime, "nominal_body_frame")
    gym, _, _ = _install_fake_runtime(monkeypatch)
    monkeypatch.setitem(sys.modules, "torch", ModuleType("torch"))
    actuator = sys.modules["isaaclab.actuators.actuator_pd"]
    original_motor = actuator.DCMotor._clip_effort
    FAULTS._before_env("F2")
    fault_fingerprint = ADAPTER.fingerprint(actuator.DCMotor._clip_effort)
    actuator.DCMotor._clip_effort = original_motor

    setup_path = Path(binding["source_setup"]["path"])
    setup = json.loads(setup_path.read_text(encoding="utf-8"))
    setup["arm"] = "F2"
    setup_path.write_text(json.dumps(setup), encoding="utf-8")
    binding["source_setup"]["sha256"] = _sha(setup_path)
    binding["secondary_target"]["expected_runtime_fingerprint"] = fault_fingerprint
    contract_path = Path(binding["code_contract"]["path"])
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["motor_targets"]["preserve_source_envelope"] = binding["secondary_target"]
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    binding["code_contract"]["sha256"] = _sha(contract_path)
    recipe_path = Path(binding["expected_reward"]["path"])
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    recipe["source_arm"] = "F2"
    recipe["source_setup_sha256"] = binding["source_setup"]["sha256"]
    recipe["secondary_target"] = binding["secondary_target"]
    recipe["code_contract_sha256"] = binding["code_contract"]["sha256"]
    recipe_path.write_text(json.dumps(recipe), encoding="utf-8")
    binding["expected_reward"]["sha256"] = _sha(recipe_path)
    monkeypatch.chdir(runtime)

    with SUBJECT.environment_scope(binding, tmp_path / "f2-readback.json"):
        gym.make("task")
        assert ADAPTER.fingerprint(actuator.DCMotor._clip_effort) == fault_fingerprint
        assert actuator.DCMotor._clip_effort is not original_motor
    assert actuator.DCMotor._clip_effort is original_motor


def test_binding_rejects_recipe_claimed_as_historical_observation(tmp_path: Path):
    binding = _binding(tmp_path, tmp_path / "runtime")
    recipe_path = Path(binding["expected_reward"]["path"])
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    recipe["provenance_kind"] = "historical_runtime_observation"
    recipe_path.write_text(json.dumps(recipe), encoding="utf-8")
    binding["expected_reward"]["sha256"] = _sha(recipe_path)
    with pytest.raises(ValueError, match="schema"):
        SUBJECT.validate_environment_binding(binding)


def test_execute_wraps_frozen_r1_flow_and_seals_live_readback(monkeypatch, tmp_path: Path):
    runtime = tmp_path / "runtime"
    stage = tmp_path / "stage"
    stage.mkdir()
    binding = _binding(stage, runtime)
    binding_path = stage.parent / f".{stage.name}-environment-binding.json"
    binding_path.write_text(json.dumps(binding), encoding="utf-8")
    manifest = {"task": SUBJECT.BASE.TASK, "seed": 5556,
                "source_checkpoint_sha256": "b" * 64, "arm": "CORRECTED", "mode": "train"}
    manifest_path = stage / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    gym, rewards, _ = _install_fake_runtime(monkeypatch)
    monkeypatch.chdir(runtime)
    monkeypatch.setattr(SUBJECT.BASE, "validate_manifest", lambda value: dict(value))

    def frozen_execute(path):
        assert path == manifest_path.resolve()
        gym.make("task")
        sys.modules["rsl_rl.runners"].OnPolicyRunner().learn(300, reset=False)
        assert ADAPTER.fingerprint(rewards.track_lin_vel_xy_exp) == binding["target"]["expected_runtime_fingerprint"]

    monkeypatch.setattr(SUBJECT.BASE, "execute", frozen_execute)
    row = SUBJECT.execute(manifest_path, binding_path)
    assert row["mode"] == "world_frame_reward"
    assert (stage / "environment-readback.json").is_file()
    assert (stage / "environment-final-readback.json").is_file()


def test_environment_close_seals_before_delegating_and_restores(monkeypatch, tmp_path: Path):
    runtime = tmp_path / "runtime"
    binding = _binding(tmp_path / "inputs", runtime)
    gym, _, environment = _install_fake_runtime(monkeypatch)
    original_close = environment.close
    runner_class = sys.modules["rsl_rl.runners"].OnPolicyRunner
    original_learn = runner_class.learn
    monkeypatch.chdir(runtime)
    readback = tmp_path / "readback.json"
    final = tmp_path / "final.json"

    with SUBJECT.environment_scope(binding, readback, final):
        assert gym.make("task") is environment
        assert runner_class().learn(300, reset=False) == ((300,), {"reset": False})
        assert environment.close() == "closed"
        assert final.is_file()
        with pytest.raises(ValueError, match="more than once"):
            environment.close()

    assert environment.close is original_close
    assert runner_class.learn is original_learn
    assert json.loads(final.read_text(encoding="utf-8"))["initial_readback"]["sha256"] == _sha(
        readback)


def test_environment_close_rejects_changed_recipe_before_original_close(monkeypatch, tmp_path: Path):
    runtime = tmp_path / "runtime"
    binding = _binding(tmp_path / "inputs", runtime)
    gym, rewards, environment = _install_fake_runtime(monkeypatch)
    calls: list[str] = []
    environment.close = lambda: calls.append("closed")
    monkeypatch.chdir(runtime)
    final = tmp_path / "final.json"
    clean_code = rewards.track_lin_vel_xy_exp.__code__

    with SUBJECT.environment_scope(binding, tmp_path / "readback.json", final):
        gym.make("task")
        sys.modules["rsl_rl.runners"].OnPolicyRunner().learn()
        rewards.track_lin_vel_xy_exp.__code__ = clean_code
        with pytest.raises(ValueError, match="changed during training"):
            environment.close()

    assert calls == []
    assert not final.exists()


def test_environment_close_before_learn_return_is_not_final(monkeypatch, tmp_path: Path):
    runtime = tmp_path / "runtime"
    binding = _binding(tmp_path / "inputs", runtime)
    gym, _, environment = _install_fake_runtime(monkeypatch)
    calls: list[str] = []
    environment.close = lambda: calls.append("closed")
    monkeypatch.chdir(runtime)
    final = tmp_path / "final.json"

    with SUBJECT.environment_scope(binding, tmp_path / "readback.json", final):
        gym.make("task")
        with pytest.raises(ValueError, match="did not return"):
            environment.close()

    assert calls == []
    assert not final.exists()


def test_learn_exception_is_preserved_and_does_not_seal(monkeypatch, tmp_path: Path):
    runtime = tmp_path / "runtime"
    binding = _binding(tmp_path / "inputs", runtime)
    gym, _, environment = _install_fake_runtime(monkeypatch)
    runner_class = sys.modules["rsl_rl.runners"].OnPolicyRunner

    def failed_learn(_self, *_args, **_kwargs):
        raise RuntimeError("learning failed")

    runner_class.learn = failed_learn
    monkeypatch.chdir(runtime)
    final = tmp_path / "final.json"
    with SUBJECT.environment_scope(binding, tmp_path / "readback.json", final):
        gym.make("task")
        with pytest.raises(RuntimeError, match="learning failed"):
            runner_class().learn(300)
        with pytest.raises(ValueError, match="did not return"):
            environment.close()

    assert runner_class.learn is failed_learn
    assert not final.exists()


def test_close_seal_survives_immediate_process_exit(tmp_path: Path):
    readback = tmp_path / "readback.json"
    final = tmp_path / "final.json"
    readback.write_text('{"sealed":true}\n', encoding="utf-8")
    script = textwrap.dedent(
        f"""
        import os
        from pathlib import Path
        from types import SimpleNamespace
        from bench import registered_epoch_recovery_training_v1 as subject

        def reward():
            pass
        def motor(self, value):
            return value
        class Motor:
            _clip_effort = motor
        class Environment:
            def close(self):
                os._exit(0)

        subject.ADAPTER.fingerprint = lambda value: (
            "reward" if value is reward else "motor" if value is Motor._clip_effort else "other")
        state = {{
            "function": reward, "motor_class": Motor, "term_function": reward,
            "original_motor": Motor._clip_effort, "readback_written": True,
            "final_readback_written": False, "close_calls": 0,
            "learn_calls": 1, "learn_returned": True,
        }}
        binding = {{
            "mode": "nominal_body_frame",
            "target": {{"expected_runtime_fingerprint": "reward"}},
            "secondary_target": {{"expected_runtime_fingerprint": "motor"}},
            "source_provenance": {{"source_arm": "R1"}},
        }}
        environment = Environment()
        subject._install_close_seal(
            environment, state, binding, Path({str(readback)!r}), Path({str(final)!r}))
        environment.close()
        raise RuntimeError("unreachable after application close")
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script], cwd=SUBJECT.ROOT, check=False,
        capture_output=True, text=True, encoding="utf-8", errors="replace")

    assert completed.returncode == 0, completed.stderr
    assert final.is_file()
    assert json.loads(final.read_text(encoding="utf-8"))["runtime_fingerprint"] == "reward"


def test_execute_rejects_environment_binding_path_swap(tmp_path: Path):
    stage = tmp_path / "stage"
    stage.mkdir()
    wrong = stage / "environment-binding.json"
    wrong.write_text("{}", encoding="utf-8")
    manifest = stage / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="registered sibling"):
        SUBJECT.execute(manifest, wrong)


def test_execute_rejects_recipe_change_during_training(monkeypatch, tmp_path: Path):
    runtime = tmp_path / "runtime"
    stage = tmp_path / "stage"
    stage.mkdir()
    binding = _binding(tmp_path / "inputs", runtime)
    binding_path = stage.parent / f".{stage.name}-environment-binding.json"
    binding_path.write_text(json.dumps(binding), encoding="utf-8")
    manifest = {"task": SUBJECT.BASE.TASK, "seed": 5556,
                "source_checkpoint_sha256": "b" * 64, "arm": "CORRECTED", "mode": "train"}
    manifest_path = stage / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    gym, rewards, _ = _install_fake_runtime(monkeypatch)
    clean_code = rewards.track_lin_vel_xy_exp.__code__
    monkeypatch.chdir(runtime)
    monkeypatch.setattr(SUBJECT.BASE, "validate_manifest", lambda value: dict(value))

    def changed_during_training(_path):
        gym.make("task")
        sys.modules["rsl_rl.runners"].OnPolicyRunner().learn()
        rewards.track_lin_vel_xy_exp.__code__ = clean_code

    monkeypatch.setattr(SUBJECT.BASE, "execute", changed_during_training)
    with pytest.raises(ValueError, match="changed during training"):
        SUBJECT.execute(manifest_path, binding_path)


def test_environment_scope_delays_isaac_import_until_first_make(monkeypatch, tmp_path: Path):
    runtime = tmp_path / "runtime"
    binding = _binding(tmp_path, runtime)
    gym, _, _ = _install_fake_runtime(monkeypatch)
    monkeypatch.chdir(runtime)
    original_import = SUBJECT.importlib.import_module
    calls: list[str] = []

    def tracked(name):
        calls.append(name)
        return original_import(name)

    monkeypatch.setattr(SUBJECT.importlib, "import_module", tracked)
    with SUBJECT.environment_scope(binding, tmp_path / "readback.json"):
        assert all(not name.startswith("isaaclab") for name in calls)
        gym.make("task")
        assert "isaaclab.envs.mdp.rewards" in calls


def test_execution_adapter_invokes_real_child_with_registered_sibling(
        monkeypatch, tmp_path: Path):
    runtime = tmp_path / "runtime"
    isaaclab = tmp_path / "IsaacLab"
    isaaclab.mkdir()
    environment = _binding(tmp_path / "inputs", runtime)
    gym, _, _ = _install_fake_runtime(monkeypatch)
    monkeypatch.chdir(runtime)
    monkeypatch.setattr(SUBJECT.BASE, "validate_manifest", lambda value: dict(value))
    def execute_with_learn(_path):
        gym.make("task")
        sys.modules["rsl_rl.runners"].OnPolicyRunner().learn()

    monkeypatch.setattr(SUBJECT.BASE, "execute", execute_with_learn)

    def owned(_folder, argv, _cwd, _watchdog):
        assert Path(argv[2]).resolve() == Path(SUBJECT.__file__).resolve()
        SUBJECT.execute(Path(argv[4]), Path(argv[6]))
        return {"exit_code": 0}

    def frozen_train(stage, _isaaclab, arm, seed, _run_name, _binding_value, owned_adapter):
        stage.mkdir()
        manifest = stage / "manifest.json"
        manifest.write_text(json.dumps({"task": SUBJECT.BASE.TASK, "seed": seed,
            "source_checkpoint_sha256": "b" * 64, "arm": arm, "mode": "train"}), encoding="utf-8")
        base = str((EXECUTION.ROOT / "bench/r1_repair_training.py").resolve())
        owned_adapter(stage / "owned", [sys.executable, "-B", base, "--manifest", str(manifest)],
                      isaaclab, 1)
        run_dir = tmp_path / "run"
        (run_dir / "params").mkdir(parents=True)
        source_env = Path(environment["source_env_config"]["path"]).read_bytes()
        (run_dir / "params/env.yaml").write_bytes(source_env)
        checkpoint = tmp_path / "model.pt"
        checkpoint.write_bytes(b"checkpoint")
        return {"arm": arm, "checkpoint": {"path": str(checkpoint.resolve()),
                                              "sha256": _sha(checkpoint)},
                "run_dir": str(run_dir)}

    monkeypatch.setattr(EXECUTION.V1, "train", frozen_train)
    result = EXECUTION.train(tmp_path / "stage", isaaclab, "CORRECTED", 5556, "run",
                             {"base": True, "environment_preservation": environment}, owned)
    checked = EXECUTION.verify_preservation(result["environment_preservation"], result["checkpoint"])
    assert checked["mode"] == "world_frame_reward"


def test_binding_rejects_source_or_fingerprint_swap(tmp_path: Path):
    binding = _binding(tmp_path, tmp_path / "runtime")
    binding["target"] = {**binding["target"], "expected_runtime_fingerprint": "0" * 64}
    with pytest.raises(ValueError, match="differs"):
        SUBJECT.validate_environment_binding(binding)


def test_binding_rejects_world_mode_for_nominal_source_arm(tmp_path: Path):
    binding = _binding(tmp_path, tmp_path / "runtime")
    setup_path = Path(binding["source_setup"]["path"])
    setup = json.loads(setup_path.read_text(encoding="utf-8"))
    setup["arm"] = "R1"
    setup_path.write_text(json.dumps(setup), encoding="utf-8")
    binding["source_setup"]["sha256"] = _sha(setup_path)
    recipe_path = Path(binding["expected_reward"]["path"])
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    recipe["source_arm"] = "R1"
    recipe["source_setup_sha256"] = binding["source_setup"]["sha256"]
    recipe_path.write_text(json.dumps(recipe), encoding="utf-8")
    binding["expected_reward"]["sha256"] = _sha(recipe_path)
    with pytest.raises(ValueError, match="registered source recipe"):
        SUBJECT.validate_environment_binding(binding)


def test_scope_rejects_second_environment(monkeypatch, tmp_path: Path):
    runtime = tmp_path / "runtime"
    binding = _binding(tmp_path, runtime)
    gym, _, _ = _install_fake_runtime(monkeypatch)
    monkeypatch.chdir(runtime)
    with pytest.raises(ValueError, match="more than one"), SUBJECT.environment_scope(
            binding, tmp_path / "readback.json"):
        gym.make("task")
        gym.make("task")


def test_import_does_not_load_sdk_modules():
    source = Path(SUBJECT.__file__).read_text(encoding="utf-8")
    prefix = source.split("def environment_scope", 1)[0]
    assert "import gymnasium" not in prefix
    assert "from isaaclab" not in prefix
