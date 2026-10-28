from __future__ import annotations

import sys
from pathlib import Path

ROOT = next(path for path in Path(__file__).resolve().parents if (path / "AGENTS.md").is_file() and (path / "bench").is_dir())
sys.path[:0] = [str(ROOT / "bench"), str(ROOT / "evals"), str(ROOT / "src")]

import agentic_brev_probe_transport_remote_v4 as R  # pyright: ignore[reportMissingImports]


def test_pair_hold_is_terminal() -> None:
    value = {"status": "pending", "pair": {"status": "hold", "result": {"sha256": "a" * 64}}}
    terminal = R._terminal_pair_status(value)
    assert terminal["status"] == "hold"
    assert terminal["terminal_source"] == "pair"


def test_pending_pair_remains_pending() -> None:
    value = {"status": "pending", "pair": {"status": "pending"}}
    assert R._terminal_pair_status(value)["status"] == "pending"
