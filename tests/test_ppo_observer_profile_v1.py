from __future__ import annotations

import cProfile
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import types

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "bench")]
import profile_ppo_observer_v1 as P  # noqa: E402


def _fake_exercise(output: Path, arm: str, observed: bool, updates: int = 3, *, fail=False):
    output.mkdir()
    if fail:
        (output / "training_failure.json").write_text('{"status":"invalid"}\n', encoding="utf-8")
        return {"failure_preserved": True}
    inspection = {"status": "abnormal" if arm == "INVERSE_RATIO" else "normal"}
    evidence = {"update_wall_s": [0.01] * updates}
    (output / "training_evidence.json").write_text(json.dumps(evidence), encoding="utf-8")
    (output / "records.jsonl").write_text(
        "".join(json.dumps({"update": index, "value": 1.0}) + "\n" for index in range(updates)),
        encoding="utf-8")
    return {"state": {"value": 1}, "losses": [[1.0]] * updates, "updates": updates,
            "completion_before_close": True, "inspection": inspection}


@pytest.fixture
def cpu_environment(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    monkeypatch.setenv("RSL_CPU_SYNTHETIC_ONLY", "1")


def test_environment_requires_both_cpu_guards(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("RSL_CPU_SYNTHETIC_ONLY", raising=False)
    with pytest.raises(RuntimeError, match="CUDA_VISIBLE_DEVICES"):
        P.run("baseline")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    with pytest.raises(RuntimeError, match="RSL_CPU_SYNTHETIC_ONLY"):
        P.run("baseline")


def test_profile_counts_exact_target_names_without_getitem_false_positives():
    class Calls:
        def item(self):
            return 1

        def getitem(self):
            return 2

        def _summarize_batch(self):
            return 3

        def _gradient_snapshot(self):
            return 4

        def _batch_summaries(self):
            return 5

    calls = Calls()
    profiler = cProfile.Profile()
    profiler.enable()
    json.dumps({"profile": True})
    hashlib.sha256(b"profile").hexdigest()
    for _ in range(3):
        calls.item()
        calls.getitem()
    for _ in range(2):
        calls._summarize_batch()
        calls._gradient_snapshot()
    calls._batch_summaries()
    profiler.disable()
    summary = P._profile_summary(profiler)
    assert summary["target_call_counts"]["item"] == 3
    assert summary["target_call_counts"]["summarize"] == 3
    assert summary["target_call_counts"]["grad_snapshot"] == 2
    assert summary["target_call_counts"]["json_serialization"] > 0
    assert summary["target_call_counts"]["sha256"] > 0


def test_baseline_has_separate_profile_warmup_and_six_measurements(
        monkeypatch, cpu_environment):
    calls = []
    monkeypatch.setattr(P.F, "exercise", lambda *args, **kwargs: (
        calls.append((args, kwargs)) or _fake_exercise(*args, **kwargs)))
    result = P.run("baseline")
    assert len(calls) == 8
    assert len(result["profiles"]) == 1 and len(result["measurements"]) == 6
    assert result["profiled_separately_from_wall_measurements"] is True
    assert result["medians"]["v4_update_wall_s"] == pytest.approx(0.03)
    assert all(row["version"] == "v4" for row in result["measurements"])


def test_compare_alternates_order_and_validates_exact_behavior(monkeypatch, cpu_environment):
    fake_v5 = types.ModuleType(P.V5)
    monkeypatch.setitem(sys.modules, P.V5, fake_v5)
    monkeypatch.setattr(P, "find_spec", lambda name: object())
    order = []

    def exercise(*args, **kwargs):
        order.append(P.F.T.__name__)
        return _fake_exercise(*args, **kwargs)

    monkeypatch.setattr(P.F, "exercise", exercise)
    result = P.run("compare")
    measured = result["measurements"]
    assert len(measured) == 12 and result["v4_v5_exact_pairs"] == [True] * 6
    assert [[row["version"] for row in measured[index:index + 2]] for index in range(0, 12, 2)] == [
        ["v4", "v5"], ["v5", "v4"], ["v4", "v5"],
        ["v5", "v4"], ["v4", "v5"], ["v5", "v4"],
    ]
    assert result["v5_validation"]["failure_preserved"] is True
    assert P.F.T.__name__ == P.V4
    assert len(order) == 2 + 2 + 12 + 6


def test_compare_rejects_any_record_difference(monkeypatch, cpu_environment):
    fake_v5 = types.ModuleType(P.V5)
    monkeypatch.setitem(sys.modules, P.V5, fake_v5)
    monkeypatch.setattr(P, "find_spec", lambda name: object())

    def exercise(*args, **kwargs):
        result = _fake_exercise(*args, **kwargs)
        if P.F.T.__name__ == P.V5 and not kwargs.get("fail"):
            path = args[0] / "records.jsonl"
            path.write_text('{"update":0,"value":1.0000000000001}\n', encoding="utf-8")
        return result

    monkeypatch.setattr(P.F, "exercise", exercise)
    with pytest.raises(ValueError, match="ON 결과"):
        P.run("compare")


def test_cli_is_utf8_write_once_and_contains_no_host_path(tmp_path, monkeypatch, cpu_environment):
    monkeypatch.setattr(P.F, "exercise", _fake_exercise)
    output = tmp_path / "profile.json"
    monkeypatch.setattr(sys, "argv", ["profile", "--mode", "baseline", "--output", str(output)])
    P.main()
    text = output.read_text(encoding="utf-8")
    assert json.loads(text)["status"] == "complete"
    assert str(Path.home()) not in text
    with pytest.raises(FileExistsError):
        P.main()


def test_actual_isaac_python_runs_real_rsl_compare(tmp_path):
    isaac = Path.home() / "IsaacLab" / "_isaac_sim"
    python = isaac / "python.bat"
    if not python.is_file():
        pytest.skip("이 호스트에 Isaac Python이 없다")
    output = tmp_path / "actual-profile.json"
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES="-1", RSL_CPU_SYNTHETIC_ONLY="1", PYTHONUTF8="1")
    result = subprocess.run([
        str(python), str(ROOT / "bench/profile_ppo_observer_v1.py"),
        "--mode", "compare", "--output", str(output),
    ], cwd=ROOT, env=environment, encoding="utf-8", errors="replace", timeout=120)
    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["actual_rsl_runner"] is True and payload["gpu_executions"] == 0
    assert payload["python_version"][:2] == [3, 10]
    assert len(payload["measurements"]) == 12
    assert payload["v4_v5_exact_pairs"] == [True] * 6
