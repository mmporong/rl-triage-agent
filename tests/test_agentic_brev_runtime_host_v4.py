from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / part) for part in ("bench", "evals", "src")]

import agentic_brev_runtime_host_v4 as M


def test_host4_source_and_parent_binding() -> None:
    assert M.SOURCES >= M.V3.SOURCES
    assert M.P6.SCHEMA == "agentic_brev_probe_pair_parent_v6"
    assert M.SCHEMA == "agentic_brev_runtime_host_v4"


def test_host4_scope_restores_host3_globals() -> None:
    before = M.V3.P2, M.V3.SCHEMA, M.V3.SOURCES
    with M._host4_scope():
        assert M.V3.P2 is M.P6
        assert M.V3.SCHEMA == M.SCHEMA
    assert (M.V3.P2, M.V3.SCHEMA, M.V3.SOURCES) == before
