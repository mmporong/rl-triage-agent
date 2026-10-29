from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / part) for part in ("bench", "evals", "src")]

import agentic_brev_repair_host_v1 as V1
import agentic_brev_repair_host_v2 as V2
import agentic_brev_runtime_host_v6 as H6


def _runtime(tmp_path: Path):
    runtime = object.__new__(V1.R.Runtime)
    runtime.reg = {"allow_fixture": True}
    runtime.folder = tmp_path
    return runtime


def test_repair_host2_requires_exact_host6_and_same_runtime(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(V1.BrevRepairRuntimeHost, "_sources", lambda self: None)
    runtime = _runtime(tmp_path)
    exact = object.__new__(H6.BrevRuntimeHost)
    exact.runtime = runtime
    host = V2.BrevRepairRuntimeHost(
        runtime, exact, object(), {}, [], "approver", probe_seals=[], probe_records=[],
        admitter=lambda *_: None, allow_fixture=True,
    )
    assert host.probe_host is exact

    class Derived(H6.BrevRuntimeHost):
        pass

    derived = object.__new__(Derived)
    derived.runtime = runtime
    with pytest.raises(ValueError, match="exact Host6"):
        V2.BrevRepairRuntimeHost(
            runtime, derived, object(), {}, [], "approver", probe_seals=[], probe_records=[],
            admitter=lambda *_: None, allow_fixture=True,
        )
    other = _runtime(tmp_path / "other")
    with pytest.raises(ValueError, match="same Runtime"):
        V2.BrevRepairRuntimeHost(
            other, exact, object(), {}, [], "approver", probe_seals=[], probe_records=[],
            admitter=lambda *_: None, allow_fixture=True,
        )


def test_host6_scope_restores_repair_globals_after_error() -> None:
    before = V1.H5, V1.SOURCES
    with pytest.raises(RuntimeError, match="stop"), V2._host6_scope():
        assert V1.H5 is H6 and V1.SOURCES == V2.SOURCES
        raise RuntimeError("stop")
    assert (V1.H5, V1.SOURCES) == before
