"""capture unit v6: 실제 UNIT4→3→2 흐름에서 runner가 RUN4로 묶이고 UNIT5의 RUN3 교체와 충돌하지 않는지 확인한다."""
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

import agentic_play_job_v2 as JOB2  # pyright: ignore[reportMissingImports]
import record_vision_evidence_v1 as R  # pyright: ignore[reportMissingImports]
import run_agentic_capture_unit_v1 as V1  # pyright: ignore[reportMissingImports]
import run_agentic_capture_unit_v3 as V3  # pyright: ignore[reportMissingImports]
import run_agentic_capture_unit_v4 as V4  # pyright: ignore[reportMissingImports]
import run_agentic_capture_unit_v5 as UNIT5  # pyright: ignore[reportMissingImports]
import run_agentic_capture_unit_v6 as UNIT6  # pyright: ignore[reportMissingImports]
import run_agentic_play_record_v1 as RUN1  # pyright: ignore[reportMissingImports]
import run_agentic_play_record_v4 as RUN4  # pyright: ignore[reportMissingImports]
import test_agentic_capture_unit_v5 as U5T  # pyright: ignore[reportMissingImports]
import test_agentic_play_job_v1 as FIX  # pyright: ignore[reportMissingImports]
import test_run_agentic_play_record_v4 as R4T  # pyright: ignore[reportMissingImports]
from test_agentic_play_job_v1 import _close_leases  # noqa: F401  # pyright: ignore[reportMissingImports]

NEW_FILES = ("bench/run_agentic_play_record_v4.py", "tests/test_run_agentic_play_record_v4.py",
             "bench/run_agentic_capture_unit_v6.py", "tests/test_agentic_capture_unit_v6.py")
BASES = (V1.RUN, V1._state_smoke, V1.admission, RUN1._leased_reader)


def _bases() -> tuple[Any, ...]:
    return V1.RUN, V1._state_smoke, V1.admission, RUN1._leased_reader


def test_closure_adds_run4_unit6_and_readiness_helper() -> None:
    assert UNIT5.SOURCE_FILES <= UNIT6.SOURCE_FILES and set(NEW_FILES) <= UNIT6.SOURCE_FILES
    assert {"bench/vision_capture_readiness_v1.py", "tests/test_vision_capture_readiness_v1.py"} <= UNIT6.SOURCE_FILES
    assert set(RUN4.SOURCE_FILES) <= UNIT6.SOURCE_FILES
    for helper in ("bench/run_agentic_play_record_v3.py", "bench/vision_pull_record_capture_v2.py",
                   "bench/record_agentic_state_sidecar_v1.py", "bench/agentic_play_job_v2.py"):
        assert helper in UNIT6.SOURCE_FILES


def test_windows_is_rejected_before_protocol(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(V1, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(V3, "_protocol", lambda path: pytest.fail("protocol을 읽으면 안 된다"))
    with pytest.raises(RuntimeError, match="Linux"):
        UNIT6.execute(U5T._args(Path("p.json"), Path("stage")))
    assert _bases() == BASES


def test_real_unit_chain_binds_run4_and_records_with_render_sync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    chain = R4T.ChainV4(tmp_path, monkeypatch)
    monkeypatch.setattr(V3, "_protocol", lambda path: {"source_sha256_lf": dict.fromkeys(UNIT6.SOURCE_FILES, "0" * 64)})
    monkeypatch.setattr(V1, "os", SimpleNamespace(name="posix"))
    seen: list[Any] = []

    def unit1(args: argparse.Namespace) -> dict[str, Any]:  # UNIT2.execute가 부르는 자리(입장·lease는 chain world)
        seen.append((V1.RUN is RUN4, V1._state_smoke is V4._state_smoke, V1.admission is V3.admission,
                     JOB2.scope_active_root() is not None))
        job, final = chain.run_real_reader(FIX.NONE_ID, runner=V1.RUN.run_job)
        return {"final": final, "stage": job.parent}

    monkeypatch.setattr(V1, "execute", unit1)
    result = UNIT6.execute(U5T._args(Path("protocol.json"), tmp_path / "stage"))
    assert seen == [(True, True, True, True)]
    assert result["final"]["final_technical_complete"] is True and result["final"]["live_evidence"] is False
    assert [command[7] for command in chain.commands] == [R4T.ENTRY4, R4T.ENTRY4]
    assert RUN4.verify_render_sync(result["stage"], frames=R.FRAME_COUNT)["calls"] == R.FRAME_COUNT
    assert _bases() == BASES


def test_unit5_runner_swap_cannot_override_run4(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(V1, "os", SimpleNamespace(name="posix"))
    monkeypatch.setattr(V3, "_protocol", lambda path: {"source_sha256_lf": dict.fromkeys(UNIT6.SOURCE_FILES, "")})
    calls: list[Any] = []
    monkeypatch.setattr(V4, "execute", lambda args: calls.append(V1.RUN) or UNIT5.execute(args))
    with pytest.raises(RuntimeError, match="이미 교체"):  # UNIT6 안에서 UNIT5가 다시 RUN3로 바꾸려 하면 거부
        UNIT6.execute(U5T._args(Path("p.json"), Path("stage")))
    assert calls == [RUN4] and _bases() == BASES
    monkeypatch.setattr(V1, "RUN", SimpleNamespace(run_job=None))
    with pytest.raises(RuntimeError, match="이미 교체"):
        UNIT6.execute(U5T._args(Path("p.json"), Path("stage")))


@pytest.mark.parametrize("case", ["intact", "missing-new-file", "tampered-copy"])
def test_real_protocol_check_requires_v6_closure_and_rejects_scratch_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    monkeypatch.setattr(U5T, "UNIT5", SimpleNamespace(SOURCE_FILES=UNIT6.SOURCE_FILES))  # helper는 그대로 재사용
    monkeypatch.setattr(U5T, "NEW_FILES", NEW_FILES)
    omit = "bench/run_agentic_play_record_v4.py" if case == "missing-new-file" else None
    repo, protocol = U5T._scratch_repo(tmp_path, monkeypatch, omit=omit)
    if case == "tampered-copy":  # 공유 원본이 아니라 scratch 복제본만 바꾼다
        FIX._write(repo / "bench/run_agentic_play_record_v4.py", "# changed after commit\n")
    reached: list[bool] = []
    monkeypatch.setattr(V4, "execute", lambda args: reached.append(V1.RUN is RUN4) or {"reached": True})
    if case == "intact":
        assert UNIT6.execute(U5T._args(protocol, tmp_path / "stage")) == {"reached": True} and reached == [True]
    else:
        with pytest.raises(ValueError) as error:
            UNIT6.execute(U5T._args(protocol, tmp_path / "stage"))
        assert ("v6 source closure" in str(error.value)) is (case == "missing-new-file")
        assert reached == []
    assert _bases() == BASES
    assert (ROOT / "bench/run_agentic_play_record_v4.py").read_bytes() != b"# changed after commit\n"


def test_main_uses_unit4_arguments(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    seen: list[argparse.Namespace] = []
    monkeypatch.setattr(UNIT6, "execute", lambda args: seen.append(args) or {"completed": True})
    monkeypatch.setattr(sys, "argv", ["unit", "--mode", "record", "--protocol", "p.json", "--provider-inventory", "i.json",
                                      "--provider-observed-at", "2026-10-28T00:00:00+00:00", "--isaaclab", "lab",
                                      "--python", "py", "--stage", "stage", "--logical-id", FIX.F2_ID])
    UNIT6.main()
    assert json.loads(capsys.readouterr().out) == {"completed": True} and seen[0].logical_id == FIX.F2_ID
