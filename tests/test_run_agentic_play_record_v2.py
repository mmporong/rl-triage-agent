from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest  # pyright: ignore[reportMissingImports]

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / "bench", ROOT / "evals", ROOT / "src", ROOT / "tests"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import agentic_play_job_v2 as JOB2  # pyright: ignore[reportMissingImports]
import run_agentic_play_record_v1 as RUN1  # pyright: ignore[reportMissingImports]
import run_agentic_play_record_v2 as RUN2  # pyright: ignore[reportMissingImports]
import test_agentic_play_job_v1 as FIX  # pyright: ignore[reportMissingImports]
import test_agentic_play_job_v2 as T2  # pyright: ignore[reportMissingImports]
import test_run_agentic_play_record_v1 as RT  # pyright: ignore[reportMissingImports]
from test_agentic_play_job_v1 import _close_leases  # noqa: F401  # pyright: ignore[reportMissingImports]

JOB1 = JOB2.JOB1
PREVIOUS = RUN1.PREVIOUS
ENTRY2 = str(JOB2.RUN2_ENTRY)


def _globals() -> tuple[Any, Any, Any]:
    return JOB1.LEASE_ROOT, JOB1.REQUIRED_SOURCE_FILES, RUN1._child_launcher


@contextmanager
def fresh_child(layout: T2.LinkedHome) -> Iterator[None]:
    """같은 프로세스에서 새 자식 프로세스의 import 직후 상태(v1 alias root·scope 없음)를 흉내 낸다."""
    state = dict(JOB2._STATE)
    saved = _globals()
    JOB2._STATE.update(depth=0, root=None, saved=None, run1=None)
    JOB1.LEASE_ROOT, JOB1.REQUIRED_SOURCE_FILES, RUN1._child_launcher = layout.alias, *T2.ORIGINAL[1:]
    try:
        yield
    finally:
        JOB2._STATE.update(state)
        JOB1.LEASE_ROOT, JOB1.REQUIRED_SOURCE_FILES, RUN1._child_launcher = saved


class ChainV2(RT.Chain):
    """RT.Chain의 실제 RUN5 흐름을 v2 world(canonical root·v2 closure)와 RUN2 entry로 지난다."""

    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, child_entry: str = "v2",
                 **kwargs: Any) -> None:
        layouts: list[T2.LinkedHome] = []

        def build(root: Path, patch: pytest.MonkeyPatch) -> Any:
            world, layout = T2.build_world_v2(root, patch)
            layouts.append(layout)
            return world

        monkeypatch.setattr(FIX, "build_world", build)  # RT.Chain.__init__이 부르는 world 조립만 바꾼다
        super().__init__(tmp_path, monkeypatch, **kwargs)
        self.layout = layouts[0]
        self.child_entry = child_entry
        self.child_roots: list[Path] = []

    def popen(self, command: list[str], **kwargs: Any) -> Any:
        if command[8] != "--sdk-child":
            return super().popen(command, **kwargs)
        self.commands.append(list(command))
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "0" and kwargs["shell"] is False
        entry = RUN2.main if self.child_entry == "v2" else RUN1.main
        original = JOB1.verify_lease

        def verify(*args: Any, **kwargs: Any) -> Any:  # 자식이 lease를 검증할 때 쓴 root만 기록한다
            self.child_roots.append(Path(JOB1.LEASE_ROOT))
            return original(*args, **kwargs)

        JOB1.verify_lease = verify
        try:
            with fresh_child(self.layout):
                code = entry([command[8], "--job", command[10]])
        finally:
            JOB1.verify_lease = original
        return RT.FakeProcess(code)

    def run(self, identity: str, mode: str = "injected", runner: Any = None) -> tuple[Path, dict[str, Any]]:
        runner = runner or RUN2.run_job
        job = FIX.write_job(self.world, identity)
        if mode == "injected":
            return job, runner(job, popen=self.popen, gpu_reader=list, identify=self.identify, cleanup=self.cleanup)
        popen, identify, cleanup = self.popen, self.identify, self.cleanup
        for key, value in (("popen", popen), ("identify", identify), ("cleanup", cleanup)):
            self.monkeypatch.setitem(RUN1.PRODUCTION, key, value)
        self.monkeypatch.setattr(PREVIOUS.REMOTE, "StrictRemoteGpuReader", lambda capture, folder, run: list)
        self.monkeypatch.setattr(PREVIOUS.BASE, "_gpu_query_run", lambda deadline: None)
        if mode == "identify-swapped":
            identify = lambda pid: {"pid": pid}
        elif mode == "cleanup-swapped":
            cleanup = lambda launch, process: self.cleanup(launch, process)
        return job, runner(job, popen=popen, identify=identify, cleanup=cleanup)


def test_v2_keeps_run1_production_contract() -> None:
    assert RUN2.PRODUCTION is RUN1.PRODUCTION and RUN2.FINAL_NAME == RUN1.FINAL_NAME
    assert RUN2.JOB1 is JOB1 and Path(ENTRY2) == Path(RUN2.__file__).resolve()


@pytest.mark.parametrize(("identity", "fault"), [(FIX.F2_ID, "F2"), (FIX.NONE_ID, "NONE")])
def test_real_run5_flow_launches_v2_entry_and_fresh_child_uses_canonical_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, identity: str, fault: str
) -> None:
    chain = ChainV2(tmp_path, monkeypatch)
    job, final = chain.run(identity)
    assert [command[7] for command in chain.commands] == [ENTRY2, ENTRY2]
    assert [command[8] for command in chain.commands] == ["--aaba-child", "--sdk-child"]
    assert chain.child_roots and set(chain.child_roots) == {chain.layout.canonical}
    assert final["final_technical_complete"] is True and final["fault_id"] == fault
    assert final["live_evidence"] is False and final["acceptable_as_live_evidence"] is False
    snapshot = json.loads(Path(chain.world.lease["path"]).read_text(encoding="utf-8"))
    assert snapshot["lock_path"] == str(chain.layout.canonical / f"{FIX.TARGET['resource_id']}.lock")
    assert final["lease"]["active_at_finalize"] is True and final["lease"]["lease_id"] == snapshot["lease_id"]
    assert json.loads((job.parent / RUN2.FINAL_NAME).read_text(encoding="utf-8")) == final


def test_v1_entry_in_fresh_child_cannot_verify_canonical_parent_lease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    chain = ChainV2(tmp_path, monkeypatch, child_entry="v1")
    job, final = chain.run(FIX.F2_ID)
    assert set(chain.child_roots) == {chain.layout.alias}  # v1 자식은 alias root로 parent lease를 찾는다
    execution = json.loads((job.parent / "execution.json").read_text(encoding="utf-8"))
    assert execution["completed"] is False and execution["error_type"] == "LeaseError"
    assert final["native_technical_complete"] is False and final["final_technical_complete"] is False


def test_run1_run_job_inside_scope_also_launches_v2_entry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    chain = ChainV2(tmp_path, monkeypatch)  # world lease가 scope를 잡고 있다(UNIT1에 JOB2를 결합한 경우와 같다)
    _job, final = chain.run(FIX.NONE_ID, runner=RUN1.run_job)
    assert [command[7] for command in chain.commands] == [ENTRY2, ENTRY2]
    assert final["final_technical_complete"] is True


def test_lease_loss_between_aaba_and_record_stops_record_child(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    chain = ChainV2(tmp_path, monkeypatch, drop_lease_after_aaba=True)
    job, final = chain.run(FIX.F2_ID)
    assert [command[8] for command in chain.commands] == ["--aaba-child"]
    execution = json.loads((job.parent / "execution.json").read_text(encoding="utf-8"))
    assert execution["completed"] is False and execution["error_type"] == "LeaseError"
    assert final["lease"]["active_at_finalize"] is False and final["final_technical_complete"] is False


@pytest.mark.parametrize(("mode", "live"), [("production", True), ("identify-swapped", False),
                                            ("cleanup-swapped", False)])
def test_live_judgement_uses_first_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str, live: bool
) -> None:
    chain = ChainV2(tmp_path, monkeypatch)
    _job, final = chain.run(FIX.NONE_ID, mode)
    assert final["final_technical_complete"] is True and final["live_evidence"] is live
    assert final["acceptable_as_live_evidence"] is (live and final["lease"]["live_lock_proof"])
    if sys.platform == "win32":
        assert final["acceptable_as_live_evidence"] is False


def test_main_and_child_main_enter_scope_then_restore(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    layout = T2.linked_home(tmp_path, monkeypatch)
    seen: list[Any] = []

    def aaba(job_path: Path, **_kwargs: Any) -> int:
        seen.append((job_path, JOB1.LEASE_ROOT, PREVIOUS.JOB is JOB1))
        return 3

    monkeypatch.setattr(PREVIOUS, "aaba_child", aaba)
    job = tmp_path / "job.json"
    assert RUN2.main(["--aaba-child", "--job", str(job)]) == 3
    assert RUN2.child_main("--aaba-child", job) == 3
    assert seen == [(job, layout.canonical, True)] * 2
    assert _globals() == T2.ORIGINAL
    with pytest.raises(ValueError, match="flag"):
        RUN2.child_main("--other", job)
    assert _globals() == T2.ORIGINAL


@pytest.mark.skipif(sys.platform != "win32", reason="Windows에서 기본 Popen 거부를 확인한다")
def test_default_popen_is_refused_off_posix_and_scope_is_restored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    T2.linked_home(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match="Linux"):
        RUN2.run_job(tmp_path / "job.json")
    assert _globals() == T2.ORIGINAL
