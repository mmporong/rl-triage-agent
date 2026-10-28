"""capture unit v5: 실제 UNIT4→3→2 흐름에서 녹화 runner만 RUN3로 묶이고, source closure·변조 거부가 유지되는지 확인한다."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest  # pyright: ignore[reportMissingImports]

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / "bench", ROOT / "evals", ROOT / "src", ROOT / "tests"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import agentic_fault_playback_v1 as PLAYBACK  # pyright: ignore[reportMissingImports]
import agentic_play_job_v2 as JOB2  # pyright: ignore[reportMissingImports]
import record_agentic_state_sidecar_v1 as SIDECAR1  # pyright: ignore[reportMissingImports]
import run_agentic_capture_unit_v1 as V1  # pyright: ignore[reportMissingImports]
import run_agentic_capture_unit_v3 as V3  # pyright: ignore[reportMissingImports]
import run_agentic_capture_unit_v4 as V4  # pyright: ignore[reportMissingImports]
import run_agentic_capture_unit_v5 as UNIT5  # pyright: ignore[reportMissingImports]
import run_agentic_play_record_v1 as RUN1  # pyright: ignore[reportMissingImports]
import run_agentic_play_record_v3 as RUN3  # pyright: ignore[reportMissingImports]
import test_agentic_play_job_v1 as FIX  # pyright: ignore[reportMissingImports]
import test_agentic_play_job_v2 as T2  # pyright: ignore[reportMissingImports]
import test_run_agentic_play_record_v3 as R3T  # pyright: ignore[reportMissingImports]
from test_agentic_play_job_v1 import _close_leases  # noqa: F401  # pyright: ignore[reportMissingImports]

JOB1 = JOB2.JOB1
NEW_FILES = ("bench/run_agentic_play_record_v3.py", "tests/test_run_agentic_play_record_v3.py",
             "bench/run_agentic_capture_unit_v5.py", "tests/test_agentic_capture_unit_v5.py")
BASES = (V1.RUN, V1._state_smoke, V1.admission, RUN1._leased_reader)


def _bases() -> tuple[Any, ...]:
    return V1.RUN, V1._state_smoke, V1.admission, RUN1._leased_reader


def _args(protocol: Path, stage: Path) -> argparse.Namespace:
    return argparse.Namespace(mode="record", protocol=protocol, provider_inventory=Path("provider.json"),
                              provider_observed_at="2026-10-28T00:00:00+00:00", isaaclab=Path("lab"),
                              python=Path(sys.executable), stage=stage, logical_id=FIX.NONE_ID)


def test_closure_contains_unit4_and_new_files() -> None:
    assert V4.SOURCE_FILES <= UNIT5.SOURCE_FILES and set(NEW_FILES) <= UNIT5.SOURCE_FILES
    assert set(RUN3.SOURCE_FILES) <= UNIT5.SOURCE_FILES and len(UNIT5.SOURCE_FILES) == len(V4.SOURCE_FILES) + 4


def test_windows_is_rejected_before_protocol(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(V1, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(V3, "_protocol", lambda path: pytest.fail("protocol을 읽으면 안 된다"))
    with pytest.raises(RuntimeError, match="Linux"):
        UNIT5.execute(_args(Path("p.json"), Path("stage")))
    assert _bases() == BASES


def test_real_unit_chain_binds_run3_and_records_with_real_reader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    chain = R3T.ChainV3(tmp_path, monkeypatch)
    closure = {"source_sha256_lf": dict.fromkeys(UNIT5.SOURCE_FILES, "0" * 64)}
    monkeypatch.setattr(V3, "_protocol", lambda path: closure)  # 실제 검증은 아래 scratch repo 테스트가 맡는다
    monkeypatch.setattr(V1, "os", SimpleNamespace(name="posix"))
    seen: list[Any] = []

    def unit1(args: argparse.Namespace) -> dict[str, Any]:  # UNIT2.execute가 부르는 자리(입장·lease는 chain world)
        seen.append((V1.RUN is RUN3, V1._state_smoke is V4._state_smoke, V1.admission is V3.admission,
                     JOB2.scope_active_root() is not None, UNIT5.V1.RUN is RUN3))
        _job, final = chain.run_real_reader(FIX.NONE_ID, runner=V1.RUN.run_job)
        return {"final": final}

    monkeypatch.setattr(V1, "execute", unit1)
    result = UNIT5.execute(_args(Path("protocol.json"), tmp_path / "stage"))
    assert seen == [(True, True, True, True, True)]
    assert result["final"]["final_technical_complete"] is True and result["final"]["live_evidence"] is False
    assert (tmp_path / "stage-lease-root.json").is_file()  # UNIT2 관측은 그대로 남는다
    assert _bases() == BASES


def test_foreign_runner_swap_is_refused_and_restored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(V1, "os", SimpleNamespace(name="posix"))
    monkeypatch.setattr(V3, "_protocol", lambda path: {"source_sha256_lf": dict.fromkeys(UNIT5.SOURCE_FILES, "")})
    monkeypatch.setattr(V1, "RUN", SimpleNamespace(run_job=None))
    with pytest.raises(RuntimeError, match="이미 교체"):
        UNIT5.execute(_args(Path("p.json"), Path("stage")))


# --- 실제 protocol 검증(scratch repo 복제본) -------------------------------------------------------


def _scratch_repo(tmp: Path, monkeypatch: pytest.MonkeyPatch, *, omit: str | None = None) -> tuple[Path, Path]:
    """새 scratch repo에 v5 closure를 커밋한다. 새 네 파일과 sidecar는 실제 바이트, 나머지는 fixture 문자열."""
    T2.linked_home(tmp, monkeypatch)
    now = datetime.now(timezone.utc)  # noqa: UP017 - Python 3.10
    repo = tmp / "repo"
    repo.mkdir()
    for args in (("init", "-q"), ("config", "user.email", "fixture@example.com"), ("config", "user.name", "fixture"),
                 ("config", "core.autocrlf", "false")):
        FIX._git(repo, *args)
    files = sorted(set(JOB2.REQUIRED_SOURCE_FILES) | UNIT5.SOURCE_FILES)
    for relative in files:
        if relative in NEW_FILES or relative == SIDECAR1.SOURCE_PATH:
            FIX._write(repo / relative, (ROOT / relative).read_bytes().replace(b"\r\n", b"\n"))
        else:
            FIX._write(repo / relative, f"# fixture {relative}\n")
    FIX._git(repo, "add", "-A")
    FIX._git(repo, "commit", "-q", "-m", "fixture")
    commit = FIX._git(repo, "rev-parse", "HEAD")
    sources = {relative: JOB1._sha(repo / relative, lf=True) for relative in files if relative != omit}
    monkeypatch.setattr(PLAYBACK, "ROOT", repo.resolve())
    _lab, runtime, rows = FIX._lab(tmp)
    protocol = JOB2.build_protocol(
        protocol_id="agentic-unit5-20261028", registered_at_utc=(now - timedelta(minutes=1)).isoformat(),
        source_commit=commit, source_sha256_lf=sources, runtime_source_sha256_lf=runtime,
        recording_target=FIX.TARGET, compute_quote_usd_per_h="1.74", batch_budget_usd="2.00",
        deadline_utc=(now + timedelta(hours=2)).isoformat(), checkpoints=rows,
        faults={FIX.F2_ID: "F2", FIX.NONE_ID: "NONE"},
        pairs=[{"pair_id": "go2-s2033", "seed": 2033, "case": FIX.F2_ID, "reference": FIX.NONE_ID}],
    )
    monkeypatch.setattr(V3, "ROOT", repo)
    monkeypatch.setattr(V1, "os", SimpleNamespace(name="posix"))
    return repo, FIX.commit_protocol(repo, "agentic_unit5_fixture.json", protocol)


@pytest.mark.parametrize("case", ["intact", "missing-new-file", "tampered-copy"])
def test_real_protocol_check_requires_v5_closure_and_rejects_scratch_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    omit = "bench/run_agentic_play_record_v3.py" if case == "missing-new-file" else None
    repo, protocol = _scratch_repo(tmp_path, monkeypatch, omit=omit)
    if case == "tampered-copy":  # 공유 원본이 아니라 scratch 복제본만 바꾼다
        FIX._write(repo / "bench/run_agentic_play_record_v3.py", "# changed after commit\n")
    reached: list[bool] = []
    monkeypatch.setattr(V4, "execute", lambda args: reached.append(V1.RUN is RUN3) or {"reached": True})
    if case == "intact":
        assert UNIT5.execute(_args(protocol, tmp_path / "stage")) == {"reached": True} and reached == [True]
    else:
        with pytest.raises(ValueError) as error:
            UNIT5.execute(_args(protocol, tmp_path / "stage"))
        assert ("v5 source closure" in str(error.value)) is (case == "missing-new-file")  # 변조는 protocol 검증이 거부
        assert reached == []
    assert _bases() == BASES
    assert (ROOT / "bench/run_agentic_play_record_v3.py").read_bytes() != b"# changed after commit\n"


def test_main_uses_unit4_arguments(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    seen: list[argparse.Namespace] = []
    monkeypatch.setattr(UNIT5, "execute", lambda args: seen.append(args) or {"completed": True})
    monkeypatch.setattr(sys, "argv", ["unit", "--mode", "record", "--protocol", "p.json", "--provider-inventory", "i.json",
                                      "--provider-observed-at", "2026-10-28T00:00:00+00:00", "--isaaclab", "lab",
                                      "--python", "py", "--stage", "stage", "--logical-id", FIX.F2_ID])
    UNIT5.main()
    assert json.loads(capsys.readouterr().out) == {"completed": True} and seen[0].logical_id == FIX.F2_ID
