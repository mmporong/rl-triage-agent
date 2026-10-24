from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))
import ppo_resumable_transport_v1 as T  # pyright: ignore[reportMissingImports]


def packet() -> dict:
    return {"schema": "packet", "facts": {"오차": 1.2}}


def binding() -> dict:
    return {"run_id": "실행-1"}


def pid_value(packet_path: Path) -> dict:
    return {"pid": 123, "start_ticks": "456",
            "packet_sha256": hashlib.sha256(packet_path.read_bytes()).hexdigest()}


def test_normal_worker_uses_utf8_exact_args_and_requires_pid_release(tmp_path: Path, monkeypatch):
    folder = tmp_path / "한글 모델"
    seen = []

    def runner(args, **kwargs):
        seen.append((args, kwargs))
        assert args[:6] == ["wsl.exe", "-d", "Ubuntu", "--", "bash", "-lc"]
        command = args[-1]
        assert '. "$HOME/.config/nvidia/env"' in command
        assert "ppo_resumable_model_v1.py" in command and "--pid-file" in command
        assert kwargs["encoding"] == "utf-8" and kwargs["errors"] == "strict"
        return SimpleNamespace(returncode=0, stdout='{"status":"paused_infra"}', stderr="")

    monkeypatch.setattr(T.subprocess, "run", runner)
    result = T.model_decision(folder, packet(), binding())
    assert result == {"status": "paused_infra"} and len(seen) == 1
    receipt = T.MODEL._read_object(folder / "transport/receipts/0001.json")
    assert receipt["status"] == "complete" and receipt["cleanup"] == "natural_exit_pid_released"
    assert not list((folder / "transport/pids").glob("*.json"))


def test_timeout_cleans_only_verified_worker_and_preserves_receipt(tmp_path: Path, monkeypatch):
    folder = tmp_path / "run"
    calls = 0

    def runner(args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            packet_path = folder / "packet.json"
            _, pid_file, _ = T._inventory(folder)
            T.MODEL._write_new(pid_file, pid_value(packet_path))
            raise subprocess.TimeoutExpired(args, kwargs["timeout"])
        assert "worker name differs" in args[-1] and "--packet" in args[-1]
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(T.subprocess, "run", runner)
    with pytest.raises(TimeoutError, match="예약 원장"):
        T.model_decision(folder, packet(), binding())
    receipt = T.MODEL._read_object(folder / "transport/receipts/0001.json")
    assert calls == 2 and receipt["cleanup"] == "verified"
    assert receipt["error_type"] == "TimeoutExpired" and receipt["status"] == "invalid"


def test_cleanup_identity_mismatch_is_fail_closed_and_pid_is_preserved(tmp_path: Path, monkeypatch):
    folder = tmp_path / "run"
    calls = 0

    def runner(args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            packet_path = folder / "packet.json"
            _, pid_file, _ = T._inventory(folder)
            T.MODEL._write_new(pid_file, pid_value(packet_path))
            raise subprocess.TimeoutExpired(args, kwargs["timeout"])
        raise subprocess.CalledProcessError(1, args)

    monkeypatch.setattr(T.subprocess, "run", runner)
    with pytest.raises(T.CleanupUnverified):
        T.model_decision(folder, packet(), binding())
    receipt = T.MODEL._read_object(folder / "transport/receipts/0001.json")
    assert receipt["cleanup"] == "unverified" and receipt["cleanup_error_type"] == "CalledProcessError"
    assert (folder / "transport/pids/0001.json").is_file()
    monkeypatch.setattr(T.subprocess, "run", lambda *_a, **_k: pytest.fail("new worker launched"))
    with pytest.raises(T.CleanupUnverified, match="과거 worker"):
        T.model_decision(folder, packet(), binding())


def test_keyboard_interrupt_cleans_verified_worker_then_propagates(tmp_path: Path, monkeypatch):
    folder = tmp_path / "run"
    calls = 0

    def runner(args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            packet_path = folder / "packet.json"
            _, pid_file, _ = T._inventory(folder)
            T.MODEL._write_new(pid_file, pid_value(packet_path))
            raise KeyboardInterrupt
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(T.subprocess, "run", runner)
    with pytest.raises(KeyboardInterrupt):
        T.model_decision(folder, packet(), binding())
    receipt = T.MODEL._read_object(folder / "transport/receipts/0001.json")
    assert calls == 2 and receipt["cleanup"] == "verified"
    assert receipt["error_type"] == "KeyboardInterrupt"


def test_natural_exit_with_pid_file_is_cleaned_but_fails_closed(tmp_path: Path, monkeypatch):
    folder = tmp_path / "run"
    calls = 0

    def runner(args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            packet_path = folder / "packet.json"
            _, pid_file, _ = T._inventory(folder)
            T.MODEL._write_new(pid_file, pid_value(packet_path))
            return SimpleNamespace(returncode=0, stdout="{}", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(T.subprocess, "run", runner)
    with pytest.raises(T.CleanupUnverified, match="자연 종료"):
        T.model_decision(folder, packet(), binding())
    receipt = T.MODEL._read_object(folder / "transport/receipts/0001.json")
    assert receipt["cleanup"] == "verified" and receipt["status"] == "invalid"


def test_existing_input_change_and_stray_pid_stop_before_launch(tmp_path: Path, monkeypatch):
    folder = tmp_path / "run"
    T.MODEL._preserve(folder / "packet.json", packet())
    T.MODEL._preserve(folder / "binding.json", binding())
    with pytest.raises(T.MODEL.J.JournalIntegrityError, match="packet"):
        T.model_decision(folder, {"schema": "changed"}, binding())

    _, pid_file, _ = T._inventory(folder)
    T.MODEL._write_new(pid_file, {"pid": 1})
    monkeypatch.setattr(T.subprocess, "run", lambda *_a, **_k: pytest.fail("worker launched"))
    with pytest.raises(T.MODEL.J.JournalIntegrityError, match="PID"):
        T.model_decision(folder, packet(), binding())


def test_invalid_utf8_json_output_records_failure_without_reusing_slot(tmp_path: Path, monkeypatch):
    folder = tmp_path / "run"
    monkeypatch.setattr(T.subprocess, "run", lambda *_a, **_k:
                        SimpleNamespace(returncode=0, stdout="not-json", stderr=""))
    with pytest.raises(json.JSONDecodeError):
        T.model_decision(folder, packet(), binding())
    receipt = T.MODEL._read_object(folder / "transport/receipts/0001.json")
    assert receipt["error_type"] == "JSONDecodeError" and receipt["cleanup"] == "natural_exit_pid_released"
    assert T._inventory(folder)[0] == 2
