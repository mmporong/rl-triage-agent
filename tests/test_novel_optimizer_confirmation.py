from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def runner():
    return load_module(ROOT / "evals/novel_optimizer_confirmation.py", "novel_optimizer_confirmation_test")


def test_protocol_freezes_novel_seed_and_checkpoint_relation(runner):
    fixed = runner.fixed_protocol()
    assert fixed["training"] == {
        "task": runner.TASK, "num_envs": 4096, "max_iterations": 300,
        "seed": 2060, "arms": ["NONE", "R1"], "final_checkpoint": "model_299.pt",
    }
    assert [row["run_id"] for row in fixed["inspection"]["runs"]] == [
        "none_s2060", "r1_s2060", "none_specificity_s2061",
    ]
    assert fixed["inspection"]["max_iterations"] == 3
    assert fixed["inspection"]["expected_steps_per_update"] == 20
    assert "재개하거나 읽지 않는다" in fixed["inspection"]["checkpoint_relation"]
    assert fixed["fresh_seed_rule"]["seeds"] == [2060, 2061]


def test_adapter_binds_new_protocol():
    adapter = load_module(ROOT / "bench/inspect_optimizer_confirmation.py", "inspect_optimizer_confirmation_test")
    previous = adapter.BASE.PROTOCOL
    with adapter.binding():
        assert adapter.BASE.PROTOCOL == ROOT / "bench/protocols/novel_optimizer_confirmation_v1.json"
        assert Path(adapter.__file__).resolve() in adapter.BASE.BOUND_SOURCES
    assert adapter.BASE.PROTOCOL == previous


def test_counts_are_read_from_three_runtime_records(tmp_path, runner):
    path = tmp_path / "records.jsonl"
    path.write_text("\n".join(json.dumps({"observed": {"optimizer_steps": step, "minibatches": step}})
                                      for step in (4, 4, 4)) + "\n", encoding="utf-8")
    assert runner._counts(path) == ([4, 4, 4], [4, 4, 4])


def test_execute_checks_freshness_before_gpu_and_tag_consumption(tmp_path, monkeypatch, runner):
    isaaclab = tmp_path / "IsaacLab"
    (isaaclab / "_isaac_sim").mkdir(parents=True)
    (isaaclab / "_isaac_sim/python.bat").write_text("", encoding="utf-8")
    monkeypatch.setattr(runner, "OUTPUT_ROOT", tmp_path / "results")
    monkeypatch.setattr(runner, "CONSUMED", tmp_path / "consumed")
    monkeypatch.setattr(runner, "verify_sources", lambda anchor=None: "a" * 40)
    monkeypatch.setattr(runner, "verify_runtime", lambda *_args: None)
    monkeypatch.setattr(runner, "read", lambda path: runner.fixed_protocol() if path == runner.PROTOCOL else json.loads(path.read_text(encoding="utf-8")))
    order = []

    def guard(seeds, sdk, *, repo):
        order.append("fresh")
        assert not (tmp_path / "results").exists()
        return {"schema": "preregistered_seed_freshness_v1", "status": "fresh", "seeds": list(seeds),
                "checked_at_utc": "x", "checked_roots": [], "checked_file_count": 0, "evidence": []}

    def gpu(_profile):
        order.append("gpu")
        return {"admission_policy": runner.GPU_PROFILE, "busy": False, "busy_reasons": [], "processes": []}

    def fail_owned(*_args):
        order.append("launch")
        raise RuntimeError("stop")

    with pytest.raises(RuntimeError, match="stop"):
        runner.execute("novel_optimizer_confirmation_20261016", isaaclab,
                       seed_guard=guard, gpu_probe=gpu, owned=fail_owned)
    assert order == ["fresh", "gpu", "gpu", "launch"]
    assert len(list((tmp_path / "consumed").glob("*.json"))) == 1
    failure = json.loads((tmp_path / "results/novel_optimizer_confirmation_20261016/failure.json").read_text(encoding="utf-8"))
    assert failure["type"] == "RuntimeError"


def test_gpu_receipt_allows_unrelated_post_work_but_rejects_owned_pid(runner):
    after = {"admission_policy": runner.GPU_PROFILE, "busy": True, "busy_reasons": ["external"],
             "processes": [{"pid": 91}]}
    assert runner.validate_gpu(after, require_idle=False) == after
    with pytest.raises(RuntimeError):
        runner.validate_gpu(after, require_idle=True)


def test_tag_is_write_once_before_freshness(tmp_path, monkeypatch, runner):
    isaaclab = tmp_path / "IsaacLab"
    (isaaclab / "_isaac_sim").mkdir(parents=True)
    (isaaclab / "_isaac_sim/python.bat").write_text("", encoding="utf-8")
    monkeypatch.setattr(runner, "OUTPUT_ROOT", tmp_path / "results")
    monkeypatch.setattr(runner, "CONSUMED", tmp_path / "consumed")
    monkeypatch.setattr(runner, "verify_sources", lambda anchor=None: "a" * 40)
    monkeypatch.setattr(runner, "verify_runtime", lambda *_args: None)
    monkeypatch.setattr(runner, "read", lambda path: runner.fixed_protocol())
    tag = "novel_optimizer_confirmation_20261016"
    (runner.OUTPUT_ROOT / tag).mkdir(parents=True)
    called = False

    def guard(*_args, **_kwargs):
        nonlocal called
        called = True
        return {}

    with pytest.raises(FileExistsError):
        runner.execute(tag, isaaclab, seed_guard=guard)
    assert called is False


def test_source_anchor_allows_unrelated_commit_and_rejects_bound_change(tmp_path, monkeypatch, runner):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
    source, protocol = repo / "source.py", repo / "protocol.json"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    monkeypatch.setattr(runner, "ROOT", repo)
    monkeypatch.setattr(runner, "PROTOCOL", protocol)
    monkeypatch.setattr(runner, "BOUND", (source, protocol))
    protocol.write_text(json.dumps({**runner.fixed_protocol(), "source_sha256_lf": runner.source_hashes(),
                                    "runtime_source_sha256_lf": {}}, ensure_ascii=False), encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "source.py", "protocol.json"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "base"], check=True)
    anchor = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True,
                            capture_output=True, text=True).stdout.strip()
    (repo / "unrelated.txt").write_text("x\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "unrelated.txt"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "unrelated"], check=True)
    assert runner.verify_sources(anchor) == anchor
    source.write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="commitment"):
        runner.verify_sources(anchor)
    subprocess.run(["git", "-C", str(repo), "checkout", "--", "source.py"], check=True)
    tree = subprocess.run(["git", "-C", str(repo), "write-tree"], check=True,
                          capture_output=True, text=True).stdout.strip()
    unrelated = subprocess.run(["git", "-C", str(repo), "commit-tree", tree, "-m", "side"], check=True,
                               capture_output=True, text=True).stdout.strip()
    with pytest.raises(ValueError, match="조상이 아니다"):
        runner.verify_sources(unrelated)
