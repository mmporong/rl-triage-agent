"""과거 provider 시각을 현재 임대 시각으로 쓰지 않는지 실제 OS 잠금과 fake GPU query로 확인한다."""
from __future__ import annotations

import argparse
import os
import sys
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / p) for p in ("bench", "evals", "tests", "src")]
import run_agentic_capture_unit_v3 as U
from test_agentic_play_job_v1 import _close_leases  # noqa: F401 - fixture가 heartbeat·OS lock을 해제한다
from test_agentic_play_job_v2 import build_world_v2


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.setattr(U.J, "REQUIRED_SOURCE_FILES", U.SOURCE_FILES)
    value, _ = build_world_v2(tmp_path, monkeypatch)
    monkeypatch.setattr(U, "ROOT", value.repo)
    snapshot = U.J._read(Path(value.lease["path"]))
    current = U.J._utc(snapshot["acquired_at_utc"], "lease acquired") + timedelta(seconds=1)
    monkeypatch.setattr(U, "_now", lambda: current)
    calls = []

    def query(command):
        calls.append(command)
        return SimpleNamespace(returncode=0, stderr="", stdout="GPU-AGENTIC\n" if command == U.J.REMOTE.GPU_COMMAND else "")

    monkeypatch.setattr(U, "_query", query)
    if os.name != "posix":
        monkeypatch.setattr(U, "_archive_lease_raw", lambda *_: None)
    return value, current, calls


def test_provider_before_acquire_is_fresh_but_lease_is_checked_now(world):
    value, current, calls = world
    provider_at = current - timedelta(seconds=11)
    protocol = U._protocol(value.protocol_path)
    with pytest.raises(U.J.LeaseError, match="유효 기간"):
        U.J.verify_lease(value.lease, protocol=protocol,
                         protocol_sha=U.J._sha(value.protocol_path, lf=True), now=provider_at)
    provider = value.root / "admission/provider.json"
    proof_path = U.admission(value.protocol_path, value.root / "v3-admission", provider,
                             provider_at.isoformat(), value.lease)
    proof = U.J._read(proof_path)
    receipt = U.J._read(proof_path.parent / "admission-times.json")
    assert proof["observed_at_utc"] == current.isoformat()
    assert receipt["provider_observed_at_utc"] == provider_at.isoformat()
    assert receipt["lease_acquired_at_utc"] == U.J._read(Path(value.lease["path"]))["acquired_at_utc"]
    assert receipt["provider_inventory_sha256"] == U.J._sha(provider)
    assert receipt["resource_sha256"] == U.J._sha(proof_path)
    assert calls == [U.J.REMOTE.GPU_COMMAND, U.J.REMOTE.PROCESS_COMMAND]
    if os.name == "posix":
        assert (proof_path.parent / "proc-locks.txt").is_file()


@pytest.mark.parametrize("offset_s", [-31, 1])
def test_stale_or_future_provider_is_rejected_before_query(world, offset_s):
    value, current, calls = world
    with pytest.raises(ValueError, match="관찰"):
        U.admission(value.protocol_path, value.root / "bad-time", value.root / "admission/provider.json",
                    (current + timedelta(seconds=offset_s)).isoformat(), value.lease)
    assert calls == [] and not (value.root / "bad-time").exists()


def test_provider_that_ages_during_query_is_rejected(world, monkeypatch):
    value, current, calls = world
    times = iter([current] * 5 + [current + timedelta(seconds=31)])
    monkeypatch.setattr(U, "_now", lambda: next(times))
    with pytest.raises(ValueError, match="관찰"):
        U.admission(value.protocol_path, value.root / "aged", value.root / "admission/provider.json",
                    current.isoformat(), value.lease)
    assert len(calls) == 2 and not (value.root / "aged/resource.json").exists()


def test_new_source_closure_is_required_before_admission(world, monkeypatch):
    value, _, calls = world
    protocol = U._protocol(value.protocol_path)
    protocol["source_sha256_lf"].pop("bench/run_agentic_capture_unit_v3.py")
    monkeypatch.setattr(U.J, "validate_protocol", lambda *_args, **_kwargs: protocol)
    with pytest.raises(ValueError, match="source closure"):
        U._protocol(value.protocol_path)
    assert calls == []


def test_expired_lease_is_not_hidden_by_fresh_provider(world, monkeypatch):
    value, current, calls = world
    expired = current + timedelta(hours=3)
    monkeypatch.setattr(U, "_now", lambda: expired)
    with pytest.raises(U.J.LeaseError, match="유효 기간"):
        U.admission(value.protocol_path, value.root / "expired", value.root / "admission/provider.json",
                    expired.isoformat(), value.lease)
    assert calls == [] and not (value.root / "expired/resource.json").exists()


def test_query_error_does_not_issue_resource_proof(world, monkeypatch):
    value, current, _ = world
    monkeypatch.setattr(U, "_query", lambda _: SimpleNamespace(returncode=1, stdout="", stderr="fixture query failed"))
    with pytest.raises(RuntimeError, match="query 실패"):
        U.admission(value.protocol_path, value.root / "query-error", value.root / "admission/provider.json",
                    current.isoformat(), value.lease)
    assert not (value.root / "query-error/resource.json").exists()


def test_provider_mutation_during_query_is_rejected(world, monkeypatch):
    value, current, _ = world
    provider = value.root / "admission/provider.json"

    def query(command):
        provider.write_bytes(provider.read_bytes() + b" ")
        return SimpleNamespace(returncode=0, stderr="", stdout="GPU-AGENTIC\n" if command == U.J.REMOTE.GPU_COMMAND else "")

    monkeypatch.setattr(U, "_query", query)
    with pytest.raises(ValueError, match="바뀌었다"):
        U.admission(value.protocol_path, value.root / "mutated", provider, current.isoformat(), value.lease)
    assert not (value.root / "mutated/resource.json").exists()


def test_capture_scope_restores_after_failure(monkeypatch):
    monkeypatch.setattr(U.V1, "os", SimpleNamespace(name="posix"))
    monkeypatch.setattr(U, "_protocol", lambda *_: {})

    def fail(_):
        assert U.V1.admission is U.admission
        raise RuntimeError("fixture failure")

    monkeypatch.setattr(U.V2, "execute", fail)
    with pytest.raises(RuntimeError, match="fixture failure"):
        U.execute(argparse.Namespace(protocol=Path("fixture.json")))
    assert U.V1.admission is U._BASE_ADMISSION


def test_windows_entry_is_rejected_before_source_or_scope(monkeypatch):
    monkeypatch.setattr(U.V1, "os", SimpleNamespace(name="nt"))
    with pytest.raises(RuntimeError, match="Linux"):
        U.execute(argparse.Namespace())
    assert U.V1.admission is U._BASE_ADMISSION
