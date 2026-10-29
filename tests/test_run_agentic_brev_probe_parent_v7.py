from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / part) for part in ("bench", "evals", "src")]

import run_agentic_brev_probe_parent_v7 as P7


def test_fresh_selection_is_explicit_and_parent_schema_stays_v6() -> None:
    assert P7.fresh_selection is P7.V6.V5.fresh_selection
    assert P7.SCHEMA == P7.V6.SCHEMA
    assert "bench/run_agentic_brev_probe_parent_v7.py" in P7.SOURCES
