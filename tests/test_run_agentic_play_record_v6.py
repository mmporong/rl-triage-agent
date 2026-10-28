"""RUN6의 JOB4 scope와 자식 entry 결합 회귀."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest  # pyright: ignore[reportMissingImports]

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / "bench", ROOT / "evals", ROOT / "src", ROOT / "tests"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import agentic_play_job_v1 as JOB1  # pyright: ignore[reportMissingImports]
import agentic_play_job_v4 as JOB4  # pyright: ignore[reportMissingImports]
import run_agentic_play_record_v1 as RUN1  # pyright: ignore[reportMissingImports]
import run_agentic_play_record_v4 as RUN4  # pyright: ignore[reportMissingImports]
import run_agentic_play_record_v6 as RUN6  # pyright: ignore[reportMissingImports]


def test_scope_binds_job4_and_run6_entry_then_restores() -> None:
    base = (RUN1.PLAYBACK, RUN4.ENTRY, JOB1.PROTOCOL_SCHEMA)
    with RUN6.aws_f1_record_scope():
        with RUN6.aws_f1_record_scope():
            assert JOB4.scope_active()
            assert JOB1.PROTOCOL_SCHEMA == JOB4.PROTOCOL_SCHEMA
            assert RUN4.ENTRY == RUN6.ENTRY
            assert RUN1.PLAYBACK is JOB4.PLAYBACK
    assert (RUN1.PLAYBACK, RUN4.ENTRY, JOB1.PROTOCOL_SCHEMA) == base


def test_entry_points_run_inside_job4_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[str, bool, Path]] = []
    monkeypatch.setattr(
        RUN4, "main", lambda argv=None: seen.append(("main", JOB4.scope_active(), RUN4.ENTRY)) or 3
    )
    monkeypatch.setattr(
        RUN4, "child_main", lambda flag, job: seen.append(("child", JOB4.scope_active(), RUN4.ENTRY)) or 4
    )
    assert RUN6.main(["--aaba-child", "--job", "j"]) == 3
    assert RUN6.child_main("--sdk-child", Path("j")) == 4
    assert seen == [("main", True, RUN6.ENTRY), ("child", True, RUN6.ENTRY)]
    assert RUN4.ENTRY == RUN6._BASE_ENTRY and not JOB4.scope_active()


def test_foreign_entry_is_rejected_without_leaking_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(RUN4, "ENTRY", Path("foreign.py"))
    with pytest.raises(RuntimeError, match="이미 교체"):
        with RUN6.aws_f1_record_scope():
            pass
    assert not JOB4.scope_active() and RUN6._STATE["depth"] == 0


def test_run_job_uses_run4_under_v6_entry(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen: list[Any] = []
    monkeypatch.setattr(
        RUN4, "run_job",
        lambda job_path, **kwargs: seen.append((job_path, kwargs, JOB4.scope_active(), RUN4.ENTRY)) or {"ok": True},
    )
    path = tmp_path / "job.json"
    assert RUN6.run_job(path, marker="x") == {"ok": True}
    assert seen == [(path, {"marker": "x"}, True, RUN6.ENTRY)]
    assert RUN4.ENTRY == RUN6._BASE_ENTRY
