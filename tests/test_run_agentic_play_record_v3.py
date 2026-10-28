"""RUN3: 실제 StrictRemoteGpuReader의 관찰 원문 참조가 RUN5 ``_gpu_reference`` 실제 검증까지 전달되는지 CPU에서 확인한다."""
from __future__ import annotations

import json
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import pytest  # pyright: ignore[reportMissingImports]

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / "bench", ROOT / "evals", ROOT / "src", ROOT / "tests"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import run_agentic_play_record_v1 as RUN1  # pyright: ignore[reportMissingImports]
import run_agentic_play_record_v2 as RUN2  # pyright: ignore[reportMissingImports]
import run_agentic_play_record_v3 as RUN3  # pyright: ignore[reportMissingImports]
import test_agentic_play_job_v1 as FIX  # pyright: ignore[reportMissingImports]
import test_run_agentic_play_record_v2 as R2T  # pyright: ignore[reportMissingImports]
from test_agentic_play_job_v1 import _close_leases  # noqa: F401  # pyright: ignore[reportMissingImports]

BASE = RUN1.PREVIOUS.BASE
REMOTE = RUN1.PREVIOUS.REMOTE
REAL_GPU_REFERENCE = BASE._gpu_reference  # Chain fixture가 바꾸기 전의 실제 검증 함수
ORIGINAL_LEASED_READER = RUN1._leased_reader
GPU_UUID = "GPU-AGENTIC"


def fake_query(calls: list[list[str]], *, processes: str = "") -> Any:
    """``nvidia-smi`` 대신 원문 문자열을 돌려주는 fake CLI(GPU 없음)."""

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert kwargs == {"capture_output": True, "text": True, "check": False, "timeout": 30}
        calls.append(list(command))
        stdout = GPU_UUID + "\n" if command == REMOTE.GPU_COMMAND else processes
        return subprocess.CompletedProcess(command, 0, stdout, "")

    return run


def real_reader(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, dict[str, Any], list[list[str]]]:
    world = FIX.build_world(tmp_path, monkeypatch)
    capture = json.loads(world.proof.read_text(encoding="utf-8"))["capture"]
    calls: list[list[str]] = []
    reader = REMOTE.StrictRemoteGpuReader(capture, tmp_path / "gpu-observations", run=fake_query(calls))
    validated = {"resource": {"resource_id": capture["resource_id"], "gpu_uuid": capture["gpu_uuid"]}}
    return reader, validated, calls


# --- proxy ----------------------------------------------------------------------------------------


def test_old_closure_loses_receipt_and_proxy_passes_real_reference_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reader, validated, calls = real_reader(tmp_path, monkeypatch)
    checks: list[str] = []
    old = ORIGINAL_LEASED_READER(reader, lambda: checks.append("old"))
    assert old() == [] and getattr(old, "last_receipt", None) is None
    with pytest.raises(TypeError, match="원문 참조가 없다"):  # v8 실제 실패와 같은 지점
        REAL_GPU_REFERENCE(old, validated)
    new = RUN3.LeasedReader(reader, lambda: checks.append("new"))
    assert new() == []
    reference = REAL_GPU_REFERENCE(new, validated)
    assert reference is reader.last_receipt and set(reference) == {"path", "sha256"}
    assert Path(reference["path"]).name == "000002.receipt.json" and len(calls) == 4
    assert checks == ["old", "new", "new"]  # 호출과 참조 접근 모두 lease를 확인한다


def test_receipt_follows_latest_read_without_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    reader, validated, _calls = real_reader(tmp_path, monkeypatch)
    proxy = RUN3.LeasedReader(reader, lambda: None)
    proxy()
    first = REAL_GPU_REFERENCE(proxy, validated)
    proxy()
    second = REAL_GPU_REFERENCE(proxy, validated)
    assert Path(first["path"]).name == "000001.receipt.json" and Path(second["path"]).name == "000002.receipt.json"
    assert second is reader.last_receipt and first is not second


@pytest.mark.parametrize("case", ["no-read", "raw-sha", "receipt-sha", "other-resource", "lease-lost", "busy"])
def test_reference_rejections_are_kept(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str) -> None:
    reader, validated, calls = real_reader(tmp_path, monkeypatch)
    if case == "busy":
        reader.run = fake_query(calls, processes=f"{GPU_UUID}, 77\n")
    lease = {"active": True}

    def check() -> None:
        if not lease["active"]:
            raise RUN1.JOB.LeaseError("lease heartbeat가 활성 상태가 아니다")

    proxy = RUN3.LeasedReader(reader, check)
    if case == "no-read":
        with pytest.raises(TypeError, match="원문 참조가 없다"):
            REAL_GPU_REFERENCE(proxy, validated)
        return
    if case == "busy":
        assert proxy() == [77]  # reader는 원문을 남기고 PID를 돌려준다
        with pytest.raises(ValueError, match="compute process"):
            REAL_GPU_REFERENCE(proxy, validated)
        return
    proxy()
    folder = tmp_path / "gpu-observations"
    if case == "raw-sha":
        (folder / "000001.gpu.csv").write_text(GPU_UUID + "\n\n", encoding="utf-8")
    elif case == "receipt-sha":
        (folder / "000001.receipt.json").write_text("{}\n", encoding="utf-8")
    elif case == "other-resource":
        validated = {"resource": dict(validated["resource"], resource_id="other-l40s")}
    else:
        lease["active"] = False
    with pytest.raises((ValueError, RUN1.JOB.LeaseError)):
        REAL_GPU_REFERENCE(proxy, validated)


# --- scope ----------------------------------------------------------------------------------------


def test_scope_restores_on_exception_nesting_and_refuses_foreign_swap() -> None:
    with RUN3.leased_reader_scope():
        assert RUN1._leased_reader is RUN3._leased_reader
        with RUN3.leased_reader_scope():
            assert RUN1._leased_reader is RUN3._leased_reader
        assert RUN1._leased_reader is RUN3._leased_reader
    assert RUN1._leased_reader is ORIGINAL_LEASED_READER
    with pytest.raises(RuntimeError, match="boom"), RUN3.leased_reader_scope():
        raise RuntimeError("boom")
    assert RUN1._leased_reader is ORIGINAL_LEASED_READER
    RUN1._leased_reader = lambda reader, check: reader
    try:
        with pytest.raises(RuntimeError, match="이미 교체"), RUN3.leased_reader_scope():
            pass
    finally:
        RUN1._leased_reader = ORIGINAL_LEASED_READER


def test_other_thread_waits_for_scope_release() -> None:
    entered, release, seen = threading.Event(), threading.Event(), []

    def other() -> None:
        entered.wait(5)
        with RUN3.leased_reader_scope():
            seen.append(release.is_set())

    thread = threading.Thread(target=other)
    with RUN3.leased_reader_scope():
        thread.start()
        entered.set()
        thread.join(0.3)
        assert thread.is_alive() and seen == []  # 다른 thread는 scope가 끝날 때까지 기다린다
        release.set()
    thread.join(5)
    assert seen == [True] and RUN1._leased_reader is ORIGINAL_LEASED_READER


# --- 실제 RUN5 흐름 ------------------------------------------------------------------------------


class ChainV3(R2T.ChainV2):
    """ChainV2 흐름에 실제 StrictRemoteGpuReader와 실제 ``_gpu_reference`` 검증을 다시 넣는다."""

    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **kwargs: Any) -> None:
        super().__init__(tmp_path, monkeypatch, **kwargs)
        self.queries: list[list[str]] = []
        monkeypatch.setattr(BASE, "_gpu_reference", REAL_GPU_REFERENCE)
        monkeypatch.setattr(BASE, "_gpu_query_run", lambda deadline: fake_query(self.queries))

    def run_real_reader(self, identity: str, runner: Any = RUN3.run_job, *, production: bool = False,
                        identify: Any = None) -> tuple[Path, dict[str, Any]]:
        job = FIX.write_job(self.world, identity)
        popen, cleanup = self.popen, self.cleanup  # bound method는 접근마다 새 객체라 한 번만 잡는다
        identify = identify or self.identify
        if production:
            for key, value in (("popen", popen), ("identify", identify), ("cleanup", cleanup)):
                self.monkeypatch.setitem(RUN1.PRODUCTION, key, value)
        return job, runner(job, popen=popen, identify=identify, cleanup=cleanup)


def test_v8_failure_reproduces_with_run2_real_reader(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    old = ChainV3(tmp_path, monkeypatch)
    job, final = old.run_real_reader(FIX.NONE_ID, runner=RUN2.run_job)
    execution = json.loads((job.parent / "execution.json").read_text(encoding="utf-8"))
    assert execution["error_type"] == "TypeError" and "원문 참조가 없다" in execution["error_message"]
    assert [command[8] for command in old.commands] == ["--aaba-child"] and final["final_technical_complete"] is False


def test_run3_carries_real_receipt_through_run5(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    new = ChainV3(tmp_path, monkeypatch)
    job, final = new.run_real_reader(FIX.NONE_ID)
    assert final["final_technical_complete"] is True
    assert [command[8] for command in new.commands] == ["--aaba-child", "--sdk-child"]
    assert [command[7] for command in new.commands] == [R2T.ENTRY2, R2T.ENTRY2]
    receipts = sorted((job.parent / "gpu-observations").glob("*.receipt.json"))
    assert len(receipts) >= 2 and len(new.queries) == 2 * len(receipts)
    assert RUN1._leased_reader is ORIGINAL_LEASED_READER


@pytest.mark.parametrize(("production", "live"), [(True, True), (False, False)])
def test_live_judgement_and_approval_flags_are_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, production: bool, live: bool
) -> None:
    chain = ChainV3(tmp_path, monkeypatch)
    _job, final = chain.run_real_reader(FIX.F2_ID, production=production)
    assert final["final_technical_complete"] is True and final["live_evidence"] is live
    assert final["acceptable_as_live_evidence"] is (live and final["lease"]["live_lock_proof"])
    assert final["behavior_label"] is None and final["used_for_decision"] is False
    if sys.platform == "win32":
        assert final["acceptable_as_live_evidence"] is False


def test_entry_points_keep_run2_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert RUN3.PRODUCTION is RUN1.PRODUCTION and RUN3.FINAL_NAME == RUN1.FINAL_NAME
    seen: list[Any] = []
    monkeypatch.setattr(RUN2, "main", lambda argv=None: seen.append(("main", RUN1._leased_reader)) or 5)
    monkeypatch.setattr(RUN2, "child_main", lambda flag, job: seen.append(("child", RUN1._leased_reader)) or 6)
    assert RUN3.main(["--aaba-child", "--job", "j.json"]) == 5 and RUN3.child_main("--aaba-child", Path("j")) == 6
    assert seen == [("main", RUN3._leased_reader), ("child", RUN3._leased_reader)]
    assert RUN1._leased_reader is ORIGINAL_LEASED_READER
