from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("evals", "bench", "src")]

import agentic_brev_runtime_host_v3 as M


def test_host_v2_uses_fresh_parent_contract() -> None:
    assert M.P2.SCHEMA == "agentic_brev_probe_pair_parent_v5"
    assert M.SCHEMA == "agentic_brev_runtime_host_v3"
    assert "evals/agentic_brev_runtime_host_v3.py" in M.SOURCES
