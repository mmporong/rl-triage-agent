from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "bench"), str(ROOT / "evals")]
import lineage_recovery_training_v1 as T


def _states(arm: str) -> tuple[list[str], list[str]]:
    return {
        "NONE": ([], []), "F1": (["F1"], []), "R1": (["R1"], []),
        "COMPOUND": (["F1", "R1"], []),
        "FIRST": (["F1", "R1"], ["restore_body_frame_reward"]),
        "BOTH": (["F1", "R1"],
                 ["restore_body_frame_reward", "restore_configured_ppo_epochs"]),
    }[arm]


def _manifest(tmp_path: Path, arm: str = "BOTH", mode: str = "inspection") -> dict:
    faults, corrections = _states(arm)
    needs_source = mode == "inspection" or arm in {"FIRST", "BOTH"}
    return {
        "schema": T.SCHEMA, "mode": mode, "arm": arm, "seed": 235,
        "run_name": f"lineage_recovery_{mode}_{arm.lower()}", "task": T.RESIDUAL.TASK,
        "num_envs": 4096, "max_iterations": T.RESIDUAL.MODES[mode],
        "configured_learning_epochs": 5, "configured_mini_batches": 4,
        "runtime_updates_to_observe": 3, "source_commit_sha": "a" * 40,
        "protocol_path": "bench/protocols/residual_loop_v1.json",
        "protocol_sha256_lf": "b" * 64,
        "source_sha256_lf": {name: "c" * 64 for name in T.BOUND_SOURCES},
        "runtime_source_sha256_lf": {
            "scripts/reinforcement_learning/rsl_rl/train.py": "d" * 64},
        "records_path": str((tmp_path / "stage" / "records.jsonl").resolve()),
        "readback_path": str((tmp_path / "stage" / "readback.json").resolve()),
        "source_checkpoint_sha256": "e" * 64 if needs_source else None,
        "original_faults": faults, "applied_corrections": corrections,
    }


def test_manifest_keeps_residual_schema_and_requires_all_wrapper_sources(tmp_path: Path) -> None:
    value = _manifest(tmp_path)
    assert T.validate_manifest(value) == value
    assert set(T.BOUND_SOURCES) <= set(value["source_sha256_lf"])
    canonical = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                           allow_nan=False).encode("utf-8")
    assert T.manifest_fingerprint(value) == hashlib.sha256(canonical).hexdigest()
    value["source_sha256_lf"].pop("bench/isaac_training_guard_v1.py")
    with pytest.raises(ValueError, match="lineage recovery 필수 source"):
        T.validate_manifest(value)


def test_verify_binding_uses_residual_verifier_with_resolved_lab(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    value = _manifest(tmp_path)
    calls = []
    monkeypatch.setattr(T.RESIDUAL, "verify_source_binding",
                        lambda data, runtime_root=None: calls.append((data, runtime_root)))
    T.verify_binding(value, lab=tmp_path / "lab")
    assert calls == [(value, (tmp_path / "lab").resolve())]


def test_residual_artifact_hashes_use_raw_bytes_not_lf_normalization(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    value = _manifest(tmp_path, "NONE", "inspection")
    records = Path(value["records_path"])
    readback = Path(value["readback_path"])
    records.parent.mkdir(parents=True)
    records.write_bytes(b'{"record":1}\r\n{"record":2}\r\n{"record":3}\r\n')
    readback.write_bytes(json.dumps({
        "schema": "residual_training_readback_v1", "original_faults": [],
        "applied_corrections": [], "remaining_faults": [],
    }).encode("utf-8") + b"\r\n")
    monkeypatch.setattr(T.RESIDUAL.OBSERVE, "assemble_inspection",
                        lambda *_args: {"records": 3, "status": "normal"})

    result = T._validate_residual_outputs(value)

    raw_records = hashlib.sha256(records.read_bytes()).hexdigest()
    lf_records = hashlib.sha256(records.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    assert raw_records != lf_records
    assert result["records"]["sha256"] == raw_records
    assert result["readback"]["sha256"] == hashlib.sha256(readback.read_bytes()).hexdigest()


@pytest.mark.parametrize(("arm", "faults", "corrections"), [
    ("NONE", [], []),
    ("FIRST", ["F1", "R1"], ["restore_configured_ppo_epochs"]),
    ("BOTH", ["F1", "R1"],
     ["restore_configured_ppo_epochs", "restore_body_frame_reward"]),
])
def test_none_first_both_preserve_residual_manifest_meaning(
        tmp_path: Path, arm: str, faults: list[str], corrections: list[str]) -> None:
    value = _manifest(tmp_path, arm, "train")
    value["original_faults"], value["applied_corrections"] = faults, corrections
    if arm == "NONE":
        value["source_checkpoint_sha256"] = None
    assert T.validate_manifest(value)["arm"] == arm


def test_validate_only_has_zero_isaac_torch_gym_imports(tmp_path: Path) -> None:
    value = _manifest(tmp_path)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    names = ("torch", "gymnasium", "rsl_rl", "isaaclab")
    previous = {name: sys.modules.pop(name) for name in names if name in sys.modules}
    try:
        assert T.main(["--manifest", str(path), "--lab", str(tmp_path),
                       "--out", str(tmp_path / "out"), "--validate-only"]) == 0
        assert all(name not in sys.modules for name in names)
    finally:
        sys.modules.update(previous)


class _FakeObserver:
    def __init__(self, ppo, manifest, sink):
        self.ppo, self.manifest, self.sink = ppo, manifest, sink
        self.index, self.previous = 0, None

    def install(self):
        self.previous = self.ppo.update

        def observed(_this):
            result = self.previous()
            if self.index < self.manifest["runtime_updates_to_observe"]:
                effective = self.ppo.num_learning_epochs * self.ppo.num_mini_batches
                expected = self.manifest["expected_optimizer_steps_per_update"]
                self.sink({
                    "schema": T.RESIDUAL.OBSERVE.SCHEMA,
                    "manifest_sha256": T.RESIDUAL.OBSERVE.manifest_fingerprint(self.manifest),
                    "binding": T.RESIDUAL.OBSERVE.manifest_binding(self.manifest),
                    "update_index": self.index,
                    "expected": {"optimizer_steps": expected, "minibatches": expected},
                    "runtime_configuration": {
                        "learning_epochs": self.ppo.num_learning_epochs,
                        "mini_batches": self.ppo.num_mini_batches},
                    "observed": {"optimizer_steps": effective, "minibatches": effective,
                                 "adam_step_before": {}, "adam_step_after": {},
                                 "adam_step_delta": {}, "gradient_present_steps": {},
                                 "pre_step_gradient_norm": [], "learning_rates": [],
                                 "approx_kl": None, "clip_fraction": None},
                    "unavailable": {}, "error": None,
                })
            self.index += 1
            return result

        self.ppo.update = MethodType(observed, self.ppo)
        return self

    def remove(self):
        if self.previous is not None:
            self.ppo.update = self.previous


def _runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, arm: str = "BOTH",
             mode: str = "inspection", failure: str | None = None):
    lab, out = tmp_path / "lab", tmp_path / "sealed"
    upstream = lab / "scripts/reinforcement_learning/rsl_rl/train.py"
    upstream.parent.mkdir(parents=True)
    iterations = T.RESIDUAL.MODES[mode]
    environment = ("try:\n        gymnasium.make('task')\n    except RuntimeError:\n        pass"
                   if failure == "swallowed_environment" else "gymnasium.make('task')")
    learn = f"runner.learn(num_learning_iterations={iterations}, init_at_random_ep_len=True)"
    if failure == "swallowed_learn":
        learn = f"try:\n        {learn}\n    except RuntimeError:\n        pass"
    body = (
        "import gymnasium\nfrom fake_training import Runner, simulation_app\n"
        "if __name__ == '__main__':\n"
        f"    {environment}\n    runner = Runner()\n    {learn}\n    simulation_app.close()\n"
    )
    if failure == "early_zero":
        body = "raise SystemExit(0)\nif __name__ == '__main__':\n    simulation_app.close()\n"
    elif failure == "bad_close_ast":
        body = "if __name__ == '__main__':\n    print('no close')\n"
    upstream.write_text(body, encoding="utf-8")

    value = _manifest(tmp_path, arm, mode)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(value), encoding="utf-8")
    monkeypatch.setattr(T.RESIDUAL, "verify_source_binding", lambda *args, **kwargs: None)
    monkeypatch.setattr(T.RESIDUAL.OBSERVE, "RuntimeObserver", _FakeObserver)

    events = {"closed": False, "sealed_at_close": False, "runner": None, "epochs": None}
    gym = types.ModuleType("gymnasium")

    def base_make(_name):
        if failure == "swallowed_environment":
            raise RuntimeError("environment failed")
        target = rewards.track_lin_vel_xy_exp
        manager = SimpleNamespace(_term_names=["track"], _term_cfgs=[SimpleNamespace(func=target)])
        return SimpleNamespace(unwrapped=SimpleNamespace(reward_manager=manager))

    gym.make = base_make
    monkeypatch.setitem(sys.modules, "gymnasium", gym)

    def reward(*_args):
        return 0.0

    rewards = types.ModuleType("isaaclab.envs.mdp.rewards")
    rewards.track_lin_vel_xy_exp = reward
    mdp = types.ModuleType("isaaclab.envs.mdp"); mdp.rewards = rewards
    envs = types.ModuleType("isaaclab.envs"); envs.mdp = mdp
    isaaclab = types.ModuleType("isaaclab"); isaaclab.envs = envs
    for name, module in (("isaaclab", isaaclab), ("isaaclab.envs", envs),
                         ("isaaclab.envs.mdp", mdp), ("isaaclab.envs.mdp.rewards", rewards)):
        monkeypatch.setitem(sys.modules, name, module)

    class PPO:
        def __init__(self):
            self.num_learning_epochs, self.num_mini_batches = 5, 4
            self.schedule, self.desired_kl = "adaptive", 0.01

        def update(self):
            return "updated"

    class Runner:
        def __init__(self):
            self.alg = PPO()
            self.log_dir = str(tmp_path / "logs")
            events["runner"] = self

        def learn(self, *args, **kwargs):
            if failure == "swallowed_learn":
                raise RuntimeError("learn failed")
            if failure == "load":
                self.load("source.pt")
            events["epochs"] = self.alg.num_learning_epochs
            count = iterations - 1 if failure == "under_update" else iterations + 1 if failure == "over_update" else iterations
            for _ in range(count):
                self.alg.update()
            directory = Path(self.log_dir); directory.mkdir(exist_ok=True)
            iteration = iterations if failure == "wrong_checkpoint" else iterations - 1
            (directory / f"model_{iteration}.pt").write_bytes(b"checkpoint")
            return {"learn": "returned"}

        def load(self, _path):
            return None

    originals = {"ppo_init": PPO.__init__, "runner_learn": Runner.learn, "runner_load": Runner.load,
                 "gym_make": gym.make, "runpy": T.RESIDUAL.runpy}
    ppo_mod = types.ModuleType("rsl_rl.algorithms.ppo"); ppo_mod.PPO = PPO
    algorithms = types.ModuleType("rsl_rl.algorithms"); algorithms.ppo = ppo_mod
    runners = types.ModuleType("rsl_rl.runners"); runners.OnPolicyRunner = Runner
    rsl = types.ModuleType("rsl_rl"); rsl.algorithms = algorithms; rsl.runners = runners
    for name, module in (("rsl_rl", rsl), ("rsl_rl.algorithms", algorithms),
                         ("rsl_rl.algorithms.ppo", ppo_mod), ("rsl_rl.runners", runners)):
        monkeypatch.setitem(sys.modules, name, module)

    class App:
        def close(self):
            events["sealed_at_close"] = all((out / name).is_file() for name in (
                T.GUARD.EVIDENCE, T.GUARD.RECORDS, T.GUARD.CLOSE_READY))
            events["closed"] = True

    fake = types.ModuleType("fake_training"); fake.Runner = Runner; fake.simulation_app = App()
    monkeypatch.setitem(sys.modules, "fake_training", fake)
    return manifest, lab, out, value, events, originals, PPO, Runner, gym


@pytest.mark.parametrize(("arm", "mode", "epochs", "remaining"), [
    ("NONE", "inspection", 5, []),
    ("FIRST", "inspection", 1, ["R1"]),
    ("BOTH", "inspection", 5, []),
    ("BOTH", "train", 5, []),
])
def test_execute_counts_returns_and_seals_real_residual_outputs_before_close(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, arm: str, mode: str,
        epochs: int, remaining: list[str]) -> None:
    manifest, lab, out, value, events, originals, PPO, Runner, gym = _runtime(
        tmp_path, monkeypatch, arm=arm, mode=mode)
    result = T.execute(manifest, lab, out)
    evidence = json.loads((out / T.GUARD.EVIDENCE).read_text(encoding="utf-8"))
    completion = json.loads((out / T.GUARD.RECORDS).read_text(encoding="utf-8"))
    assert result["status"] == "complete" and events["closed"] and events["sealed_at_close"]
    assert evidence["schema"] == T.EVIDENCE_SCHEMA
    assert evidence["updates"] == evidence["update_count"] == value["max_iterations"]
    assert evidence["remaining_faults"] == remaining and evidence["source_checkpoint_sha256"] == "e" * 64
    assert evidence["fresh_initialization"] is True and evidence["checkpoint_loaded"] is False
    assert evidence["learn_returned"] is True and evidence["recorded_before_app_close"] is True
    assert evidence["ppo_readback"]["effective"] == {"learning_epochs": epochs, "mini_batches": 4}
    assert completion["schema"] == T.COMPLETION_SCHEMA
    assert completion["learning_completed"] is True
    assert completion["recorded_before_app_close"] is True
    assert Path(evidence["residual_records"]["path"]) == Path(value["records_path"])
    assert evidence["residual_records"]["sha256"] == hashlib.sha256(
        Path(value["records_path"]).read_bytes()).hexdigest()
    assert PPO.__init__ is originals["ppo_init"] and Runner.learn is originals["runner_learn"]
    assert Runner.load is originals["runner_load"] and gym.make is originals["gym_make"]
    assert T.RESIDUAL.runpy is originals["runpy"] and "update" not in events["runner"].alg.__dict__


@pytest.mark.parametrize("failure", [
    "swallowed_environment", "swallowed_learn", "under_update", "over_update", "load",
    "wrong_checkpoint", "early_zero", "bad_close_ast",
])
def test_execute_rejects_incomplete_false_zero_and_restores_scopes(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    manifest, lab, out, _, events, originals, PPO, Runner, gym = _runtime(
        tmp_path, monkeypatch, failure=failure)
    with pytest.raises((RuntimeError, ValueError, FileNotFoundError, SystemExit)):
        T.execute(manifest, lab, out)
    assert (out / "failure.json").is_file()
    assert not events["closed"]
    assert PPO.__init__ is originals["ppo_init"] and Runner.learn is originals["runner_learn"]
    assert Runner.load is originals["runner_load"] and gym.make is originals["gym_make"]
    assert T.RESIDUAL.runpy is originals["runpy"]
    if events["runner"] is not None:
        assert "update" not in events["runner"].alg.__dict__


def test_preexisting_out_and_cli_failure_are_nonzero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "out"; out.mkdir()
    with pytest.raises(FileExistsError, match="fresh"):
        T.execute(tmp_path / "manifest.json", tmp_path / "lab", out)
    value = _manifest(tmp_path)
    manifest = tmp_path / "manifest.json"; manifest.write_text(json.dumps(value), encoding="utf-8")
    monkeypatch.setattr(T.RESIDUAL, "verify_source_binding", lambda *args, **kwargs: None)
    failed = tmp_path / "failed"
    assert T.main(["--manifest", str(manifest), "--lab", str(tmp_path / "missing"),
                   "--out", str(failed)]) == 1
    assert json.loads((failed / "failure.json").read_text(encoding="utf-8"))["status"] == "invalid"
