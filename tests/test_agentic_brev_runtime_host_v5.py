from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / part) for part in ("bench", "evals", "src")]

import agentic_brev_runtime_host_v4 as V4
import agentic_brev_runtime_host_v5 as V5
import run_agentic_brev_probe_parent_v7 as P7


def test_host5_scopes_parent7_without_changing_registration_schema() -> None:
    original = V4.P6
    with V5._parent7_scope():
        assert V4.P6 is P7
        assert V4.SOURCES == V5.SOURCES
        assert callable(V4.P6.fresh_selection)
    assert V4.P6 is original
    assert V5.SCHEMA == V4.SCHEMA


def test_parent7_scope_restores_globals_after_nested_error() -> None:
    before = V4.P6, V4.SOURCES
    try:
        with V5._parent7_scope(), V5._parent7_scope():
            assert V4.P6 is P7
            raise ValueError("nested binding refused")
    except ValueError:
        pass
    assert (V4.P6, V4.SOURCES) == before
