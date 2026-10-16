from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("vl_worker_test", ROOT / "evals/vl_aux_worker_v1.py")
W = importlib.util.module_from_spec(spec)
spec.loader.exec_module(W)


def paths(tmp_path):
    packet = tmp_path / "packet.json"
    packet.write_text("{}", encoding="utf-8")
    return packet, tmp_path / "result.json"


def test_saved_result_requires_finished_owned_worker(tmp_path, monkeypatch):
    packet, output = paths(tmp_path)
    calls = []
    monkeypatch.setattr(W, "wsl_path", lambda path: "/test/" + Path(path).name)
    def runner(argv, **kwargs):
        calls.append((argv, kwargs))
        if len(calls) == 1:
            output.write_text("{}", encoding="utf-8")
            output.with_suffix(".pid.json").write_text("{}", encoding="utf-8")
            return SimpleNamespace(returncode=0)
        return SimpleNamespace(returncode=0, stdout=json.dumps({"cleanup_verified": True, "was_running": False}))
    W.invoke_worker("diagnosis", packet, output, runner=runner)
    assert len(calls) == 2
    assert calls[0][0][:4] == ["wsl", "-d", "Ubuntu", "--"]
    assert calls[0][1]["timeout"] == 700 and calls[1][1]["timeout"] == 15
    assert "source" not in calls[0][0]  # credential sourcing stays inside the quoted shell command
    assert '"$HOME/.config/nvidia/env"' in calls[0][0][-1]


def test_timeout_requires_identity_cleanup(tmp_path, monkeypatch):
    packet, output = paths(tmp_path)
    calls = []
    monkeypatch.setattr(W, "wsl_path", lambda path: "/test/" + Path(path).name)
    def runner(argv, **kwargs):
        calls.append(argv)
        if len(calls) == 1:
            output.with_suffix(".pid.json").write_text("{}", encoding="utf-8")
            raise subprocess.TimeoutExpired(argv, kwargs["timeout"])
        return SimpleNamespace(returncode=0, stdout=json.dumps({"cleanup_verified": True, "was_running": True}))
    with pytest.raises(TimeoutError, match="cleanup verified"):
        W.invoke_worker("correction", packet, output, runner=runner)
    assert len(calls) == 2


def test_missing_identity_does_not_issue_cleanup_command(tmp_path, monkeypatch):
    packet, output = paths(tmp_path)
    monkeypatch.setattr(W, "wsl_path", lambda path: "/test/" + Path(path).name)
    calls = []
    def runner(argv, **kwargs):
        calls.append(argv)
        raise subprocess.TimeoutExpired(argv, kwargs["timeout"])
    with pytest.raises(RuntimeError, match="identity unavailable"):
        W.invoke_worker("diagnosis", packet, output, runner=runner)
    assert len(calls) == 1


def test_cleanup_failure_does_not_accept_a_saved_output(tmp_path, monkeypatch):
    packet, output = paths(tmp_path)
    monkeypatch.setattr(W, "wsl_path", lambda path: "/test/" + Path(path).name)
    calls = []
    def runner(argv, **kwargs):
        calls.append(argv)
        if len(calls) == 1:
            output.write_text("{}", encoding="utf-8")
            output.with_suffix(".pid.json").write_text("{}", encoding="utf-8")
            return SimpleNamespace(returncode=0)
        return SimpleNamespace(returncode=2, stdout="")
    with pytest.raises(RuntimeError, match="cleanup not verified"):
        W.invoke_worker("observation", packet, output, tmp_path / "ledger", runner=runner)


def test_reused_output_or_unregistered_kind_is_rejected_before_launch(tmp_path):
    packet, output = paths(tmp_path)
    with pytest.raises(ValueError, match="unregistered"):
        W.invoke_worker("other", packet, output)
    output.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="output exists"):
        W.invoke_worker("diagnosis", packet, output)
