from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("bench", "evals", "src")]

import run_agentic_brev_probe_parent_v6 as V6  # pyright: ignore[reportMissingImports]
import run_agentic_brev_probe_parent_v8 as P8  # pyright: ignore[reportMissingImports]


def test_parent8_exports_v6_schema_selection_and_complete_source_closure() -> None:
    assert P8.SCHEMA == V6.SCHEMA
    assert P8.fresh_selection is V6.V5.fresh_selection
    assert V6.SOURCES < P8.SOURCES
    assert {P8.SOURCE, P8.TEST_SOURCE} <= P8.SOURCES


def test_parent8_main_runs_v6_runtime_inside_fresh_scope(monkeypatch) -> None:
    observed = []

    def main() -> None:
        observed.append((V6.V1.SCHEMA, V6.V1._selection))

    monkeypatch.setattr(V6.V1, "main", main)
    P8.main()
    assert observed == [(P8.SCHEMA, P8.fresh_selection)]
