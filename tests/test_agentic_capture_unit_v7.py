"""capture unit v7: 실제 UNIT4→3→2 흐름이 F1 scope 안에서 runner를 RUN5로 묶고, UNIT5·UNIT6 교체와 충돌하지 않는다."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest  # pyright: ignore[reportMissingImports]

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / "bench", ROOT / "evals", ROOT / "src", ROOT / "tests"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import agentic_f1_playback_v1 as PB  # pyright: ignore[reportMissingImports]
import agentic_play_job_v2 as JOB2  # pyright: ignore[reportMissingImports]
import agentic_play_job_v3 as JOB3  # pyright: ignore[reportMissingImports]
import record_vision_evidence_v1 as R  # pyright: ignore[reportMissingImports]
import run_agentic_capture_unit_v1 as V1  # pyright: ignore[reportMissingImports]
import run_agentic_capture_unit_v3 as V3  # pyright: ignore[reportMissingImports]
import run_agentic_capture_unit_v4 as V4  # pyright: ignore[reportMissingImports]
import run_agentic_capture_unit_v6 as UNIT6  # pyright: ignore[reportMissingImports]
import run_agentic_capture_unit_v7 as UNIT7  # pyright: ignore[reportMissingImports]
import run_agentic_play_record_v1 as RUN1  # pyright: ignore[reportMissingImports]
import run_agentic_play_record_v4 as RUN4  # pyright: ignore[reportMissingImports]
import run_agentic_play_record_v5 as RUN5  # pyright: ignore[reportMissingImports]
import test_agentic_capture_unit_v5 as U5T  # pyright: ignore[reportMissingImports]
import test_agentic_play_job_v3 as J3T  # pyright: ignore[reportMissingImports]
import test_run_agentic_play_record_v5 as R5T  # pyright: ignore[reportMissingImports]
from test_agentic_play_job_v1 import _close_leases  # noqa: F401  # pyright: ignore[reportMissingImports]

NEW_FILES = ("bench/agentic_f1_playback_v1.py", "tests/test_agentic_f1_playback_v1.py",
             "bench/agentic_play_job_v3.py", "tests/test_agentic_play_job_v3.py",
             "bench/run_agentic_play_record_v5.py", "tests/test_run_agentic_play_record_v5.py",
             "bench/run_agentic_capture_unit_v7.py", "tests/test_agentic_capture_unit_v7.py")
BASES = (V1.RUN, V1._state_smoke, V1.admission, RUN1._leased_reader, RUN1.PLAYBACK, RUN4.ENTRY)


def _bases() -> tuple[Any, ...]:
    return V1.RUN, V1._state_smoke, V1.admission, RUN1._leased_reader, RUN1.PLAYBACK, RUN4.ENTRY


def _protocol(path: Path) -> dict[str, Any]:
    return {"source_sha256_lf": dict.fromkeys(UNIT7.SOURCE_FILES, "0" * 64)}


def test_closure_adds_eight_files_to_unit6() -> None:
    assert UNIT6.SOURCE_FILES <= UNIT7.SOURCE_FILES and set(NEW_FILES) <= UNIT7.SOURCE_FILES
    assert len(UNIT7.SOURCE_FILES - UNIT6.SOURCE_FILES) == len(NEW_FILES)
    assert JOB3.SOURCE_FILES <= UNIT7.SOURCE_FILES and "bench/vision_transferred_pull_job_v1.py" in UNIT7.SOURCE_FILES


def test_windows_is_rejected_before_protocol(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(V1, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(V3, "_protocol", lambda path: pytest.fail("protocol을 읽으면 안 된다"))
    with pytest.raises(RuntimeError, match="Linux"):
        UNIT7.execute(U5T._args(Path("p.json"), Path("stage")))
    assert _bases() == BASES and not JOB3.scope_active()


def test_real_unit_chain_binds_run5_and_records_f1_with_render_sync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    chain = R5T.ChainF1(tmp_path, monkeypatch)
    monkeypatch.setattr(V3, "_protocol", _protocol)
    monkeypatch.setattr(V1, "os", SimpleNamespace(name="posix"))
    seen: list[Any] = []

    def unit1(args: argparse.Namespace) -> dict[str, Any]:  # UNIT2.execute가 부르는 자리(입장·lease는 chain world)
        seen.append((V1.RUN is RUN5, V1._state_smoke is V4._state_smoke, V1.admission is V3.admission,
                     JOB2.scope_active_root() is not None, JOB3.scope_active(), RUN1.PLAYBACK is PB))
        job, final = chain.run_real_reader(J3T.F1_ID, runner=V1.RUN.run_job)
        return {"final": final, "stage": job.parent}

    monkeypatch.setattr(V1, "execute", unit1)
    result = UNIT7.execute(U5T._args(Path("protocol.json"), tmp_path / "stage"))
    assert seen == [(True, True, True, True, True, True)]
    assert result["final"]["final_technical_complete"] is True and result["final"]["fault_id"] == "F1"
    assert [command[7] for command in chain.commands] == [R5T.ENTRY5, R5T.ENTRY5]
    assert RUN4.verify_render_sync(result["stage"], frames=R.FRAME_COUNT)["calls"] == R.FRAME_COUNT
    assert _bases() == BASES


def test_missing_closure_stops_before_unit4(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(V1, "os", SimpleNamespace(name="posix"))
    partial = {"source_sha256_lf": dict.fromkeys(UNIT6.SOURCE_FILES, "0" * 64)}
    monkeypatch.setattr(V3, "_protocol", lambda path: partial)
    monkeypatch.setattr(V4, "execute", lambda args: pytest.fail("UNIT4에 도달하면 안 된다"))
    with pytest.raises(ValueError, match="v7 source closure"):
        UNIT7.execute(U5T._args(Path("p.json"), Path("stage")))
    assert _bases() == BASES and not JOB3.scope_active()


@pytest.mark.parametrize("other", ["unit6", "unit5"])
def test_older_runner_swap_inside_unit7_is_refused(monkeypatch: pytest.MonkeyPatch, other: str) -> None:
    import run_agentic_capture_unit_v5 as UNIT5  # pyright: ignore[reportMissingImports]

    monkeypatch.setattr(V1, "os", SimpleNamespace(name="posix"))
    monkeypatch.setattr(V3, "_protocol", _protocol)
    nested = UNIT6.execute if other == "unit6" else UNIT5.execute
    calls: list[Any] = []
    monkeypatch.setattr(V4, "execute", lambda args: calls.append(V1.RUN) or nested(args))
    with pytest.raises(RuntimeError, match="이미 교체"):
        UNIT7.execute(U5T._args(Path("p.json"), Path("stage")))
    assert calls == [RUN5] and _bases() == BASES


def test_main_uses_unit4_arguments(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    seen: list[argparse.Namespace] = []
    monkeypatch.setattr(UNIT7, "execute", lambda args: seen.append(args) or {"completed": True})
    monkeypatch.setattr(sys, "argv", ["unit", "--mode", "record", "--protocol", "p.json", "--provider-inventory", "i.json",
                                      "--provider-observed-at", "2026-10-28T00:00:00+00:00", "--isaaclab", "lab",
                                      "--python", "py", "--stage", "stage", "--logical-id", J3T.F1_ID])
    UNIT7.main()
    assert json.loads(capsys.readouterr().out) == {"completed": True} and seen[0].logical_id == J3T.F1_ID
