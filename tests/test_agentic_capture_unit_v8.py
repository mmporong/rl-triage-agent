"""UNIT8 closure와 JOB4·RUN6 실행 경계 회귀."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest  # pyright: ignore[reportMissingImports]

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / "bench", ROOT / "evals", ROOT / "src", ROOT / "tests"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import agentic_play_job_v4 as JOB4  # pyright: ignore[reportMissingImports]
import run_agentic_capture_unit_v1 as V1  # pyright: ignore[reportMissingImports]
import run_agentic_capture_unit_v3 as V3  # pyright: ignore[reportMissingImports]
import run_agentic_capture_unit_v4 as V4  # pyright: ignore[reportMissingImports]
import run_agentic_capture_unit_v7 as UNIT7  # pyright: ignore[reportMissingImports]
import run_agentic_capture_unit_v8 as UNIT8  # pyright: ignore[reportMissingImports]
import run_agentic_play_record_v6 as RUN6  # pyright: ignore[reportMissingImports]


def _args() -> argparse.Namespace:
    return argparse.Namespace(protocol=Path("protocol.json"), provider_inventory=Path("inventory.json"),
                              provider_observed_at="2026-10-28T00:00:00+00:00", isaaclab=Path("lab"),
                              python=Path("python"), stage=Path("stage"), logical_id="source-s2221-none",
                              mode="record")


def test_closure_adds_exactly_six_files_to_unit7() -> None:
    expected = {
        "bench/agentic_play_job_v4.py", "tests/test_agentic_play_job_v4.py",
        "bench/run_agentic_play_record_v6.py", "tests/test_run_agentic_play_record_v6.py",
        "bench/run_agentic_capture_unit_v8.py", "tests/test_agentic_capture_unit_v8.py",
    }
    assert UNIT7.SOURCE_FILES <= UNIT8.SOURCE_FILES
    assert UNIT8.SOURCE_FILES - UNIT7.SOURCE_FILES == expected
    assert len(UNIT8.SOURCE_FILES) == len(UNIT7.SOURCE_FILES) + 6


def test_execute_binds_run6_job4_and_restores(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(V1, "os", SimpleNamespace(name="posix"))
    monkeypatch.setattr(V3, "_protocol", lambda path: {
        "source_sha256_lf": dict.fromkeys(UNIT8.SOURCE_FILES, "0" * 64)
    })
    seen: list[tuple[Any, ...]] = []

    def execute(args: argparse.Namespace) -> dict[str, bool]:
        seen.append((V1.RUN, JOB4.scope_active(), args.logical_id))
        return {"completed": True}

    monkeypatch.setattr(V4, "execute", execute)
    assert UNIT8.execute(_args()) == {"completed": True}
    assert seen == [(RUN6, True, "source-s2221-none")]
    assert V1.RUN is UNIT8._BASE_RUN and not JOB4.scope_active()


def test_missing_new_closure_stops_before_unit4(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(V1, "os", SimpleNamespace(name="posix"))
    monkeypatch.setattr(V3, "_protocol", lambda path: {
        "source_sha256_lf": dict.fromkeys(UNIT7.SOURCE_FILES, "0" * 64)
    })
    monkeypatch.setattr(V4, "execute", lambda args: pytest.fail("UNIT4에 도달하면 안 된다"))
    with pytest.raises(ValueError, match="v8 source closure"):
        UNIT8.execute(_args())
    assert V1.RUN is UNIT8._BASE_RUN and not JOB4.scope_active()


def test_windows_rejected_before_protocol(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(V1, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(V3, "_protocol", lambda path: pytest.fail("protocol을 읽으면 안 된다"))
    with pytest.raises(RuntimeError, match="Linux"):
        UNIT8.execute(_args())
