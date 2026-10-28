from __future__ import annotations

import sys
from pathlib import Path

ROOT = next(path for path in Path(__file__).resolve().parents if (path / "AGENTS.md").is_file() and (path / "bench").is_dir())
sys.path[:0] = [str(ROOT / "bench"), str(ROOT / "evals"), str(ROOT / "src")]

import agentic_brev_probe_transport_remote_v5 as R  # noqa: E402  # pyright: ignore[reportMissingImports]


def test_pair_hold_is_terminal() -> None:
    value = {"status": "pending", "pair": {"status": "hold", "result": {"sha256": "a" * 64}}}
    terminal = R._terminal_pair_status(value)
    assert terminal["status"] == "hold"
    assert terminal["terminal_source"] == "pair"


def test_pending_pair_remains_pending() -> None:
    value = {"status": "pending", "pair": {"status": "pending"}}
    assert R._terminal_pair_status(value)["status"] == "pending"


def test_remote5_does_not_import_python311_only_remote4() -> None:
    source = (ROOT / R.SOURCE).read_text(encoding="utf-8")
    assert "agentic_brev_probe_transport_remote_v4" not in source
    assert "from datetime import UTC" not in source
    assert "datetime.now(timezone.utc)" in source
