from __future__ import annotations

import hashlib
import importlib
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
import lineage_source_training_v1 as T


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")).hexdigest()


def _job(*, purpose: str = "source", arm: str = "NONE", seed: int = 17) -> dict:
    return {"job_id": f"job-{purpose}-{arm}", "case_id": "case-a", "arm": arm, "seed": seed,
            "run_name": f"lineage_{purpose}_{arm}", "task": T.TASK, "num_envs": 4096,
            "max_iterations": 1 if purpose == "smoke" else 300}


def _manifest(*, purpose: str = "source", arm: str = "NONE", seed: int = 17) -> dict:
    return {"schema": T.SCHEMA, "purpose": purpose, **_job(purpose=purpose, arm=arm, seed=seed),
            "source_commit": "a" * 40, "protocol_relative": "protocol.json",
            "protocol_sha256_lf": "b" * 64,
            "source_sha256_lf": {name: "c" * 64 for name in T.SOURCE_FILES},
            "runtime_source_sha256_lf": {"runtime.py": "d" * 64}}


def _protocol(job: dict, purpose: str) -> dict:
    filler = [{**_job(arm="NONE"), "job_id": f"filler-{index}"} for index in range(17)]
    return {"smoke_job": job if purpose == "smoke" else _job(purpose="smoke"),
            "training_jobs": [job, *filler] if purpose == "source" else [_job(), *filler]}


def test_validate_contract_and_validate_only_has_no_heavy_imports(tmp_path: Path) -> None:
    value = _manifest()
    assert T.validate_manifest(value)["max_iterations"] == 300
    names = ("torch", "gymnasium", "rsl_rl", "isaaclab")
    previous = {name: sys.modules.pop(name) for name in names if name in sys.modules}
    try:
        path = tmp_path / "manifest.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        assert T.main(["--manifest", str(path), "--lab", str(tmp_path), "--out", str(tmp_path / "out"),
                       "--validate-only"]) == 0
        assert all(name not in sys.modules for name in names)
    finally:
        sys.modules.update(previous)


@pytest.mark.parametrize("change", ["resume", "checkpoint", "long_name", "budget", "smoke_arm"])
def test_manifest_rejects_resume_checkpoint_and_policy_specific_changes(change: str) -> None:
    value = _manifest(purpose="smoke" if change == "smoke_arm" else "source")
    if change in {"resume", "checkpoint"}:
        value[change] = True
    elif change == "long_name":
        value["run_name"] = "x" * 81
    elif change == "budget":
        value["max_iterations"] = 299
    else:
        value["arm"] = "F1"
    with pytest.raises(ValueError):
        T.validate_manifest(value)


def test_verify_binding_checks_commit_blob_protocol_job_and_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo, lab = tmp_path / "repo", tmp_path / "lab"
    repo.mkdir(); lab.mkdir()
    value = _manifest()
    protocol = _protocol({key: value[key] for key in T.JOB_FIELDS}, "source")
    (repo / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
    for name in T.SOURCE_FILES:
        path = repo / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(name, encoding="utf-8")
    (lab / "runtime.py").write_text("runtime", encoding="utf-8")
    value["protocol_sha256_lf"] = _sha(repo / "protocol.json")
    value["source_sha256_lf"] = {name: _sha(repo / name) for name in T.SOURCE_FILES}
    value["runtime_source_sha256_lf"] = {"runtime.py": _sha(lab / "runtime.py")}
    blobs = {value["protocol_relative"]: (repo / value["protocol_relative"]).read_bytes(),
             **{name: (repo / name).read_bytes() for name in T.SOURCE_FILES}}

    def git(_repo: Path, *args: str, check: bool = True):
        if args[0] == "show":
            return types.SimpleNamespace(returncode=0, stdout=blobs[args[1].split(":", 1)[1]])
        return types.SimpleNamespace(returncode=0, stdout=b"")

    monkeypatch.setattr(T, "_git", git)
    T.verify_binding(value, repo=repo, lab=lab)
    (repo / T.SOURCE_FILES[0]).write_text("tamper", encoding="utf-8")
    with pytest.raises(ValueError, match="현재 source"):
        T.verify_binding(value, repo=repo, lab=lab)
    (repo / T.SOURCE_FILES[0]).write_bytes(blobs[T.SOURCE_FILES[0]])
    (repo / "protocol.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="현재 source"):
        T.verify_binding(value, repo=repo, lab=lab)


def _runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, purpose: str = "source",
             arm: str = "NONE", mode: str = "normal") -> tuple[Path, Path, dict, dict]:
    lab, out = tmp_path / "lab", tmp_path / "out"
    upstream = lab / "scripts/reinforcement_learning/rsl_rl/train.py"
    upstream.parent.mkdir(parents=True)
    body = """
import gymnasium
from fake_training import Runner, simulation_app
if __name__ == "__main__":
    {environment}
    runner = Runner()
    {learn}
    simulation_app.close()
""".format(
        environment=("try:\n        gymnasium.make('task')\n    except RuntimeError:\n        pass"
                     if mode == "swallow_env" else "gymnasium.make('task')"),
        learn=("pass" if mode == "missing_learn" else
               "runner.learn(num_learning_iterations=" + ("1" if purpose == "smoke" else "300")
               + ", init_at_random_ep_len=True)"),
    )
    upstream.write_text(body, encoding="utf-8")
    value = _manifest(purpose=purpose, arm=arm, seed=2399 if purpose == "smoke" else 17)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(value), encoding="utf-8")
    monkeypatch.setattr(T, "verify_binding", lambda *args, **kwargs: None)

    events: dict = {"closed": False, "sealed_at_close": False, "epochs": None, "runner": None}
    gym = types.ModuleType("gymnasium")

    def make(_name):
        if mode == "swallow_env":
            raise RuntimeError("environment failed")
        return object()

    gym.make = make
    monkeypatch.setitem(sys.modules, "gymnasium", gym)
    hidden = importlib.import_module("hidden_faults")
    monkeypatch.setattr(hidden, "_before_env", lambda _fault: None)
    monkeypatch.setattr(hidden, "_after_env", lambda _fault, _env: {})

    class PPO:
        def __init__(self):
            self.num_learning_epochs = 5
            self.num_mini_batches = 4

        def update(self):
            return "updated"

    class Runner:
        def __init__(self):
            self.alg = PPO()
            self.log_dir = str(tmp_path / "logs")
            events["runner"] = self

        def learn(self, *args, **kwargs):
            events["epochs"] = self.alg.num_learning_epochs
            if mode == "learn_exception":
                raise RuntimeError("learn failed")
            if mode == "load":
                self.load("checkpoint.pt")
            expected = 1 if purpose == "smoke" else 300
            count = expected - 1 if mode == "under_update" else expected + 1 if mode == "over_update" else expected
            for _ in range(count):
                self.alg.update()
            log = Path(self.log_dir); log.mkdir()
            iteration = 0 if purpose == "smoke" else 299
            (log / f"model_{iteration}.pt").write_bytes(b"checkpoint")
            return {"learn": "returned"}

        def load(self, _path):
            return None

    events["original_load"] = Runner.load

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
    return manifest_path, lab, value, events


@pytest.mark.parametrize(("purpose", "arm", "epochs", "iteration"), [
    ("source", "NONE", 5, 299), ("source", "COMPOUND", 1, 299), ("smoke", "NONE", 5, 0),
])
def test_execute_uses_guard_and_seals_completion_before_close(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, purpose: str, arm: str,
        epochs: int, iteration: int) -> None:
    manifest, lab, _, events = _runtime(tmp_path, monkeypatch, purpose=purpose, arm=arm)
    out = tmp_path / "out"
    result = T.execute(manifest, lab, out)
    evidence = json.loads((out / T.GUARD.EVIDENCE).read_text(encoding="utf-8"))
    record = json.loads((out / T.GUARD.RECORDS).read_text(encoding="utf-8"))
    assert result["status"] == "complete"
    assert events["closed"] is True and events["sealed_at_close"] is True and events["epochs"] == epochs
    assert "update" not in events["runner"].alg.__dict__
    assert type(events["runner"]).load is events["original_load"]
    assert evidence["learn_returned"] is True
    assert evidence["update_count"] == (1 if purpose == "smoke" else 300)
    assert evidence["checkpoint"]["iteration"] == iteration
    assert evidence["patch_readback"]["effective_epochs"] == epochs
    assert record["resume_requested"] is False and record["checkpoint_input"] is False


@pytest.mark.parametrize("mode", [
    "missing_learn", "learn_exception", "swallow_env", "under_update", "over_update", "load",
])
def test_execute_rejects_incomplete_or_swallowed_failures(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    manifest, lab, _, events = _runtime(tmp_path, monkeypatch, mode=mode)
    out = tmp_path / "out"
    with pytest.raises((RuntimeError, ValueError)):
        T.execute(manifest, lab, out)
    assert (out / "failure.json").is_file()
    assert not events["closed"]
    if events["runner"] is not None:
        assert "update" not in events["runner"].alg.__dict__
        assert type(events["runner"]).load is events["original_load"]


def test_execute_rejects_preexisting_output(tmp_path: Path) -> None:
    out = tmp_path / "out"; out.mkdir()
    with pytest.raises(FileExistsError, match="fresh"):
        T.execute(tmp_path / "manifest.json", tmp_path / "lab", out)


def test_main_returns_nonzero_and_writes_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    value = _manifest()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(value), encoding="utf-8")
    monkeypatch.setattr(T, "verify_binding", lambda *args, **kwargs: None)
    out = tmp_path / "out"
    assert T.main(["--manifest", str(manifest), "--lab", str(tmp_path / "missing-lab"),
                   "--out", str(out)]) == 1
    assert json.loads((out / "failure.json").read_text(encoding="utf-8"))["status"] == "invalid"
