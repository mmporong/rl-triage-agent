from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _module():
    spec = importlib.util.spec_from_file_location(
        "train_with_correction_under_test", ROOT / "bench" / "train_with_correction.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeArray:
    def __init__(self, raw: bytes, *, dtype="float32", shape=(1,)):
        self.raw, self.dtype, self.shape = raw, dtype, shape

    def detach(self): return self
    def cpu(self): return self
    def contiguous(self): return self
    def numpy(self): return SimpleNamespace(tobytes=lambda order="C": self.raw)


def test_state_digest_binds_tensor_content_dtype_shape_and_scalars():
    module = _module()
    first = module.state_digest({"w": FakeArray(b"one", shape=(1, 3)), "step": 2, "lr": 0.1})
    same = module.state_digest({"lr": 0.1, "step": 2, "w": FakeArray(b"one", shape=(1, 3))})
    changed = module.state_digest({"w": FakeArray(b"two", shape=(1, 3)), "step": 2, "lr": 0.1})
    assert first == same
    assert first["sha256"] != changed["sha256"]
    assert first["tensors"] == [{
        "path": '$["w"]', "dtype": "float32", "shape": [1, 3],
        "content_sha256": hashlib.sha256(b"one").hexdigest(),
    }]
    assert first["scalar_count"] == 2


def _manifest(tmp_path):
    resume = tmp_path / "failed_run" / "model_299.pt"
    resume.parent.mkdir(parents=True, exist_ok=True)
    if not resume.exists():
        resume.write_bytes(b"resume")
    return {
        "schema": "training_correction_manifest_v1", "tag": "tag", "case": "h01_s1", "arm": "corrected",
        "source_commit_sha": "a" * 40, "protocol_sha256_lf": "b" * 64,
        "isaaclab_root": str(tmp_path), "protocol_path": str(tmp_path / "protocol.json"),
        "fault": "F1", "correction_id": "restore_body_frame_reward", "seed": 1,
        "num_envs": 8, "max_iterations": 2, "task": "Task-v0", "run_name": "fixed",
        "output_dir": str(tmp_path / "evidence"),
        "resume_checkpoint_path": str(resume),
        "resume_checkpoint_sha256": hashlib.sha256(b"resume").hexdigest(),
        "resume_iteration": 299, "load_run_regex": "^failed_run$", "checkpoint_regex": "^model_299\\.pt$",
        "resume_env_config_path": str(tmp_path / "failed_run" / "params" / "env.yaml"),
        "resume_agent_config_path": str(tmp_path / "failed_run" / "params" / "agent.yaml"),
    }


def test_upstream_argv_has_only_fixed_flags_and_no_hydra_overrides(tmp_path):
    module = _module()
    argv = module._upstream_argv(tmp_path / "train.py", _manifest(tmp_path))
    assert argv == [
        str(tmp_path / "train.py"), "--task", "Task-v0", "--seed", "1", "--num_envs", "8",
        "--max_iterations", "2", "--headless", "--run_name", "fixed", "--resume",
        "--load_run", "^failed_run$", "--checkpoint", "^model_299\\.pt$",
    ]
    assert not any("=" in item for item in argv)


class FakeH:
    def __init__(self): self.writes = {}
    def write_new(self, path, value):
        path = Path(path)
        if path in self.writes: raise FileExistsError(path)
        self.writes[path] = value

    def sha(self, path, lf=False):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def validate_training_configuration(self, log_dir, manifest):
        log_dir = Path(log_dir)
        if not (log_dir / "params/env.yaml").is_file() or not (log_dir / "params/agent.yaml").is_file():
            raise ValueError("configuration missing")


def _fake_runner_class(*, fail_step=False, short=False, loaded_iteration=299):
    class Runner:
        def __init__(self, env, train_cfg, log_dir=None, device=None):
            self.env, self.num_steps_per_env = env, 2
            self.alg = SimpleNamespace(
                policy=SimpleNamespace(state_dict=lambda: {"weight": FakeArray(b"policy")}),
                optimizer=SimpleNamespace(state_dict=lambda: {"state": {0: {"step": 0}}}),
            )
            self.current_learning_iteration = 0

        def load(self, path, load_optimizer=True):
            self.current_learning_iteration = loaded_iteration
            return {"loaded": True}

        def learn(self, num_learning_iterations, init_at_random_ep_len):
            calls = num_learning_iterations * self.num_steps_per_env - (1 if short else 0)
            for index in range(calls):
                if fail_step and index == 1:
                    raise RuntimeError("step failed")
                self.env.step(index)
            return "done"
    return Runner


def _install_fake_runner(monkeypatch, runner_class):
    module = SimpleNamespace(OnPolicyRunner=runner_class)
    monkeypatch.setattr("importlib.import_module", lambda name: module if name == "rsl_rl.runners" else None)

    _install_fake_torch(monkeypatch)


def _install_fake_torch(monkeypatch):
    torch = types.ModuleType("torch")
    torch.load = lambda path, map_location=None, weights_only=None: {
        "iter": 300 if Path(path).name == "model_300.pt" else 299,
        "model_state_dict": {"weight": FakeArray(b"policy")},
        "optimizer_state_dict": {"state": {0: {"step": 0}}},
    }
    torch.get_rng_state = lambda: FakeArray(b"cpu-rng", dtype="uint8", shape=(7,))
    torch.cuda = SimpleNamespace(get_rng_state_all=lambda: [FakeArray(b"cuda-rng", dtype="uint8", shape=(8,))])
    monkeypatch.setitem(sys.modules, "torch", torch)


def _write_target_configs(tmp_path):
    params = tmp_path / "logs" / "params"
    params.mkdir(parents=True, exist_ok=True)
    (params / "env.yaml").write_text("env", encoding="utf-8")
    (params / "agent.yaml").write_text("agent", encoding="utf-8")


def _env(seed=1, num_envs=8):
    calls = []
    env = SimpleNamespace(
        unwrapped=SimpleNamespace(cfg=SimpleNamespace(
            seed=seed, scene=SimpleNamespace(num_envs=num_envs),
            actions=SimpleNamespace(joint_pos=SimpleNamespace(scale=0.25)),
        )),
        step=lambda value: calls.append(value) or value,
    )
    return env, calls


def test_runner_hooks_record_initial_state_and_exact_successful_step_budget(tmp_path, monkeypatch):
    module = _module()
    runner_class = _fake_runner_class()
    _install_fake_runner(monkeypatch, runner_class)
    manifest, H, runtime = _manifest(tmp_path), FakeH(), {}
    original_init, original_load, original_learn, hooked = module._install_runner_hooks(
        manifest, H, runtime, expected_action_scale=0.25,
    )
    env, calls = _env()
    runner = hooked(env, {"seed": 1, "max_iterations": 2}, log_dir=tmp_path / "logs")
    assert H.writes == {}
    runner.load(manifest["resume_checkpoint_path"])
    _write_target_configs(tmp_path)
    assert runner.learn(num_learning_iterations=2, init_at_random_ep_len=True) == "done"
    assert calls == [0, 1, 2, 3]
    assert runtime["actual_control_steps"] == runtime["expected_control_steps"] == 4
    initial = H.writes[tmp_path / "evidence" / "initial.json"]
    assert initial["policy_state"]["tensors"][0]["content_sha256"] == hashlib.sha256(b"policy").hexdigest()
    assert initial["observed"]["action_scale"] == 0.25
    assert hooked.__init__ is not original_init and hooked.load is not original_load and hooked.learn is not original_learn


@pytest.mark.parametrize("bad", ["learn_args", "short", "seed"])
def test_runner_hooks_reject_budget_and_seed_mismatches(tmp_path, monkeypatch, bad):
    module = _module()
    runner_class = _fake_runner_class(short=bad == "short")
    _install_fake_runner(monkeypatch, runner_class)
    manifest, H, runtime = _manifest(tmp_path), FakeH(), {}
    module._install_runner_hooks(manifest, H, runtime, expected_action_scale=0.25)
    env, _ = _env(seed=2 if bad == "seed" else 1)
    if bad == "seed":
        with pytest.raises(ValueError, match="differs"):
            runner_class(env, {"seed": 1, "max_iterations": 2}, log_dir=tmp_path / "logs")
        return
    runner = runner_class(env, {"seed": 1, "max_iterations": 2}, log_dir=tmp_path / "logs")
    runner.load(manifest["resume_checkpoint_path"])
    _write_target_configs(tmp_path)
    if bad == "learn_args":
        with pytest.raises(ValueError, match="budget"):
            runner.learn(num_learning_iterations=1, init_at_random_ep_len=True)
    else:
        with pytest.raises(RuntimeError, match="steps differ"):
            runner.learn(num_learning_iterations=2, init_at_random_ep_len=True)


def test_failed_step_is_not_counted_and_original_step_is_restored(tmp_path, monkeypatch):
    module = _module()
    runner_class = _fake_runner_class(fail_step=True)
    _install_fake_runner(monkeypatch, runner_class)
    manifest, H, runtime = _manifest(tmp_path), FakeH(), {}
    module._install_runner_hooks(manifest, H, runtime, expected_action_scale=0.25)
    env, calls = _env()
    original_step = env.step
    runner = runner_class(env, {"seed": 1, "max_iterations": 2}, log_dir=tmp_path / "logs")
    runner.load(manifest["resume_checkpoint_path"])
    _write_target_configs(tmp_path)
    with pytest.raises(RuntimeError, match="step failed"):
        runner.learn(num_learning_iterations=2, init_at_random_ep_len=True)
    assert calls == [0]
    assert env.step is original_step
    assert "learning_completed" not in runtime


@pytest.mark.parametrize("failure", ["wrong_path", "wrong_hash", "optimizer_false", "iteration", "missing_load"])
def test_resume_checkpoint_contract_rejects_invalid_or_missing_load(tmp_path, monkeypatch, failure):
    module = _module()
    runner_class = _fake_runner_class(loaded_iteration=298 if failure == "iteration" else 299)
    _install_fake_runner(monkeypatch, runner_class)
    manifest, H, runtime = _manifest(tmp_path), FakeH(), {}
    if failure == "wrong_hash":
        manifest["resume_checkpoint_sha256"] = "0" * 64
    module._install_runner_hooks(manifest, H, runtime, expected_action_scale=0.25)
    env, _ = _env()
    runner = runner_class(env, {"seed": 1, "max_iterations": 2}, log_dir=tmp_path / "logs")
    if failure == "missing_load":
        with pytest.raises(RuntimeError, match="requires"):
            runner.learn(num_learning_iterations=2, init_at_random_ep_len=True)
    elif failure == "wrong_path":
        wrong = tmp_path / "failed_run" / "model_298.pt"
        wrong.write_bytes(b"resume")
        with pytest.raises(ValueError, match="path differs"):
            runner.load(wrong)
    elif failure == "optimizer_false":
        with pytest.raises(ValueError, match="optimizer"):
            runner.load(manifest["resume_checkpoint_path"], load_optimizer=False)
    elif failure == "wrong_hash":
        with pytest.raises(ValueError, match="SHA differs"):
            runner.load(manifest["resume_checkpoint_path"])
    else:
        with pytest.raises(ValueError, match="iteration differs"):
            runner.load(manifest["resume_checkpoint_path"])
    assert H.writes == {}


def test_manifest_field_contract_is_exact(tmp_path):
    module = _module()
    manifest = _manifest(tmp_path)
    assert set(manifest) == module.MANIFEST_FIELDS
    manifest["unexpected"] = True
    assert set(manifest) != module.MANIFEST_FIELDS


class FullH:
    def __init__(self, manifest, protocol, events):
        self.manifest, self.protocol, self.events = manifest, protocol, events

    def validate_manifest(self, path):
        return self.manifest, self.protocol

    def validate_readback(self, readback, fault, correction):
        self.events.append(("validate", fault, correction))

    def write_new(self, path, value):
        path = Path(path)
        if path.exists(): raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
        self.events.append(("write", path.name))

    def sha(self, path, lf=False):
        path = Path(path)
        if path == Path(self.manifest["protocol_path"]):
            return self.manifest["protocol_sha256_lf"]
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def validate_training_configuration(self, log_dir, manifest):
        assert (Path(log_dir) / "params/env.yaml").is_file()
        assert (Path(log_dir) / "params/agent.yaml").is_file()
        self.events.append(("validate", "configuration"))


def _full_fake_modules(monkeypatch, tmp_path, events, *, fail_upstream=False):
    gym = types.ModuleType("gymnasium")
    base_env, step_calls = _env()
    gym.make = lambda *args, **kwargs: events.append(("env", "create")) or base_env
    monkeypatch.setitem(sys.modules, "gymnasium", gym)

    hidden = types.ModuleType("hidden_faults")
    hidden._before_env = lambda fault: events.append(("fault", fault))

    def arm(fault):
        original_make = gym.make

        def make(*args, **kwargs):
            hidden._before_env(fault)
            return original_make(*args, **kwargs)

        gym.make = make
    hidden.arm = arm
    monkeypatch.setitem(sys.modules, "hidden_faults", hidden)

    adapter = types.ModuleType("correction_adapter")
    adapter.capture = lambda root, contract: events.append(("capture", contract["schema"])) or "snapshot"
    def apply(snapshot, correction):
        events.append(("apply", correction))
        changed = [correction] if correction == "restore_body_frame_reward" else []
        return {"selected_correction": correction, "changed_targets": changed, "restored_targets": changed}
    adapter.apply = apply
    monkeypatch.setitem(sys.modules, "correction_adapter", adapter)

    runner_class = _fake_runner_class()
    runners = types.ModuleType("rsl_rl.runners")
    runners.OnPolicyRunner = runner_class
    monkeypatch.setitem(sys.modules, "rsl_rl.runners", runners)
    _install_fake_torch(monkeypatch)

    smoke = types.ModuleType("runtime_readback_smoke")
    smoke.process_identity = lambda: {"pid": 1, "creation_time_ns": 2}
    monkeypatch.setitem(sys.modules, "runtime_readback_smoke", smoke)

    def fake_run(path, run_name):
        env = gym.make("Task-v0")
        if fail_upstream:
            raise RuntimeError("upstream failed")
        runner = runners.OnPolicyRunner(
            env, {"seed": 1, "max_iterations": 2}, log_dir=tmp_path / "logs", device="cpu",
        )
        runner.load(str(tmp_path / "failed_run" / "model_299.pt"))
        log_dir = tmp_path / "logs"
        (log_dir / "params").mkdir(parents=True)
        (log_dir / "params" / "env.yaml").write_text("env", encoding="utf-8")
        (log_dir / "params" / "agent.yaml").write_text("agent", encoding="utf-8")
        (log_dir / "model_300.pt").write_bytes(b"checkpoint")
        runner.learn(num_learning_iterations=2, init_at_random_ep_len=True)
    return fake_run, step_calls


def test_execute_orders_capture_fault_correction_before_one_environment_and_writes_receipt_last(
        tmp_path, monkeypatch):
    module = _module()
    manifest, events = _manifest(tmp_path), []
    protocol = {
        "correction_source_contract": {"schema": "portable"},
        "action_scale": 0.25, "rsl_rl_version": "unavailable", "isaaclab_commit": "sdk-commit",
    }
    H = FullH(manifest, protocol, events)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    upstream = tmp_path / "scripts/reinforcement_learning/rsl_rl/train.py"
    upstream.parent.mkdir(parents=True)
    upstream.write_text("# fake", encoding="utf-8")
    fake_run, steps = _full_fake_modules(monkeypatch, tmp_path, events)
    monkeypatch.setattr(module.runpy, "run_path", fake_run)
    monkeypatch.chdir(tmp_path)

    receipt = module.execute(manifest_path, H)

    assert events[:5] == [
        ("capture", "portable"), ("fault", "F1"),
        ("apply", "restore_body_frame_reward"),
        ("validate", "F1", "restore_body_frame_reward"), ("write", "correction.json"),
    ]
    assert events.count(("env", "create")) == 1
    assert steps == [0, 1, 2, 3]
    assert receipt["completed"] is True and receipt["actual_control_steps"] == 4
    assert receipt["aggregate_env_steps"] == 32
    assert receipt["resume_iteration"] == 299 and receipt["final_iteration"] == 300
    assert receipt["source_checkpoint_state"]["iteration"] == 299
    assert receipt["final_checkpoint_state"]["iteration"] == 300
    initial = json.loads((tmp_path / "evidence" / "initial.json").read_text(encoding="utf-8"))
    assert initial["rng_state_before_learn"]["cuda_device_count"] == 1
    assert events.index(("validate", "configuration")) < events.index(("write", "initial.json"))
    assert events[-1] == ("write", "receipt.json")


def test_execute_never_writes_receipt_when_upstream_fails(tmp_path, monkeypatch):
    module = _module()
    manifest, events = _manifest(tmp_path), []
    H = FullH(manifest, {
        "correction_source_contract": {"schema": "portable"},
        "action_scale": 0.25, "rsl_rl_version": "unavailable", "isaaclab_commit": "sdk-commit",
    }, events)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    upstream = tmp_path / "scripts/reinforcement_learning/rsl_rl/train.py"
    upstream.parent.mkdir(parents=True)
    upstream.write_text("# fake", encoding="utf-8")
    fake_run, _ = _full_fake_modules(monkeypatch, tmp_path, events, fail_upstream=True)
    monkeypatch.setattr(module.runpy, "run_path", fake_run)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError, match="upstream failed"):
        module.execute(manifest_path, H)
    assert not (tmp_path / "evidence" / "receipt.json").exists()


def test_execute_preserves_wrong_target_readback_instead_of_forcing_matching_restore(tmp_path, monkeypatch):
    module = _module()
    manifest, events = _manifest(tmp_path), []
    manifest["correction_id"] = "restore_motor_torque_speed"
    H = FullH(manifest, {
        "correction_source_contract": {"schema": "portable"},
        "action_scale": 0.25, "rsl_rl_version": "unavailable", "isaaclab_commit": "sdk-commit",
    }, events)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    upstream = tmp_path / "scripts/reinforcement_learning/rsl_rl/train.py"
    upstream.parent.mkdir(parents=True)
    upstream.write_text("# fake", encoding="utf-8")
    fake_run, _ = _full_fake_modules(monkeypatch, tmp_path, events)
    monkeypatch.setattr(module.runpy, "run_path", fake_run)
    monkeypatch.chdir(tmp_path)
    module.execute(manifest_path, H)
    correction = json.loads((tmp_path / "evidence" / "correction.json").read_text(encoding="utf-8"))
    assert correction == {
        "selected_correction": "restore_motor_torque_speed", "changed_targets": [], "restored_targets": [],
    }
