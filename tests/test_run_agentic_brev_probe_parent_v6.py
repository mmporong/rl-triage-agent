from __future__ import annotations

import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / part) for part in ("bench", "evals", "src")]

import run_agentic_brev_probe_parent_v6 as M


def test_fresh_prepare_has_no_capsule_guard() -> None:
    source = inspect.getsource(M._prepare_impl)
    assert "selection_capsule" not in source
    assert "selection_result" not in source
    assert "fresh_selection" in source
    assert M.SOURCES >= M.V5.SOURCES


def test_parent6_scope_applies_worker_schema_selection_and_prepare() -> None:
    original = M.V1._selection, M.V1.SCHEMA, M.V1.prepare
    with M.fresh_parent_scope():
        assert M.V5.BASE.BACKEND.WORKER == M.V5.BASE.WORKER1
        assert M.V1._selection is M.V5.fresh_selection
        assert M.V1.SCHEMA == M.SCHEMA
        assert M.V1.prepare is M._prepare_impl
    assert (M.V1._selection, M.V1.SCHEMA, M.V1.prepare) == original


