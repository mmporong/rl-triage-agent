from __future__ import annotations

import sys
from pathlib import Path

ROOT = next(path for path in Path(__file__).resolve().parents if (path / "AGENTS.md").is_file())
CANDIDATE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(CANDIDATE / "bench"), str(CANDIDATE / "evals")]
sys.path.extend(str(ROOT / name) for name in ("bench", "evals", "src"))

import agentic_brev_probe_transport_remote_v7 as R  # noqa: E402  # pyright: ignore[reportMissingImports]


def test_pair_hold_is_terminal_and_preserves_raw_result() -> None:
    result = {"path": "/runtime/outcome.json", "sha256": "a" * 64}
    pair = {"status": "hold", "result": result}
    value = {"status": "pending", "arm": "NONE", "pair": pair}
    terminal = R._terminal_status({"transport_folder": Path("unused")}, value)
    assert terminal == {
        "status": "hold", "arm": "NONE", "pair": pair, "terminal_source": "pair",
    }
    assert terminal["pair"] is pair
    assert terminal["pair"]["result"] is result


def test_live_pending_is_preserved() -> None:
    value = {"status": "pending", "pair": {"status": "pending"}}
    prepared = {"transport_folder": Path("unused")}
    original = R.V6._dead_parent_status
    try:
        R.V6._dead_parent_status = lambda _prepared, current: current
        assert R._terminal_status(prepared, value) is value
    finally:
        R.V6._dead_parent_status = original


def test_arm_complete_is_preserved_without_dead_parent_lookup() -> None:
    value = {"status": "completed", "arm": "NONE", "pair": {"status": "complete"}}
    prepared = {"transport_folder": Path("unused")}
    original = R.V6._dead_parent_status
    try:
        R.V6._dead_parent_status = lambda *_args: (_ for _ in ()).throw(AssertionError())
        assert R._terminal_status(prepared, value) is value
    finally:
        R.V6._dead_parent_status = original


def test_dead_parent_hold_is_still_delegated() -> None:
    value = {"status": "pending", "pair": {"status": "pending"}}
    expected = {**value, "status": "hold", "terminal_source": "dead_parent_without_outcome"}
    prepared = {"transport_folder": Path("unused")}
    original = R.V6._dead_parent_status
    try:
        R.V6._dead_parent_status = lambda got_prepared, current: (
            expected if got_prepared is prepared and current is value else current
        )
        assert R._terminal_status(prepared, value) == expected
    finally:
        R.V6._dead_parent_status = original
