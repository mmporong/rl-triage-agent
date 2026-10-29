from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("bench", "evals", "src")]

import run_agentic_brev_probe_parent_v5 as M


def test_selection_scope_restores_parent_function() -> None:
    original, schema = M.V1._selection, M.V1.SCHEMA
    with M.fresh_parent_scope():
        assert M.V1._selection is M.fresh_selection
        assert M.V1.SCHEMA == M.SCHEMA
        assert M.BASE.BACKEND.WORKER == M.BASE.WORKER1
    assert M.V1._selection is original and M.V1.SCHEMA == schema


def test_selection_rejects_old_capsule_shape() -> None:
    try:
        M.fresh_selection({"selection_capsule": {}}, object())
    except ValueError as error:
        assert "fresh selection source schema" in str(error)
    else:
        raise AssertionError("old capsule shape accepted")
