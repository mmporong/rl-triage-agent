"""RUN4: capture 직전 렌더 sync(forward 1회·물리 snapshot 동일)와 실제 RUN5·sidecar 흐름의 증거 봉인을 CPU에서 확인한다."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

import pytest  # pyright: ignore[reportMissingImports]

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / "bench", ROOT / "evals", ROOT / "src", ROOT / "tests"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import agentic_play_job_v2 as JOB2  # pyright: ignore[reportMissingImports]
import record_vision_evidence_v1 as R  # pyright: ignore[reportMissingImports]
import run_agentic_play_record_v1 as RUN1  # pyright: ignore[reportMissingImports]
import run_agentic_play_record_v2 as RUN2  # pyright: ignore[reportMissingImports]
import run_agentic_play_record_v4 as RUN4  # pyright: ignore[reportMissingImports]
import test_agentic_play_job_v1 as FIX  # pyright: ignore[reportMissingImports]
import test_run_agentic_play_record_v1 as RT  # pyright: ignore[reportMissingImports]
import test_run_agentic_play_record_v2 as R2T  # pyright: ignore[reportMissingImports]
import test_run_agentic_play_record_v3 as R3T  # pyright: ignore[reportMissingImports]
from test_agentic_play_job_v1 import _close_leases  # noqa: F401  # pyright: ignore[reportMissingImports]

CAPTURE = RUN4.CAPTURE
REAL_MAKER = CAPTURE.make_capture_factory
ORIGINAL_BASE = RT._base
ORIGINAL_LEASED_READER = RUN1._leased_reader
ENTRY4 = str(RUN4.ENTRY)


class Tensor(RT.FakeTensor):
    """tuple index·flatten을 더한 fake tensor(CPU fixture)."""

    def __getitem__(self, index: Any) -> Tensor:
        value = self.value
        for part in index if isinstance(index, tuple) else (index,):
            value = value[part]
        return Tensor(value)

    def flatten(self) -> Tensor:
        out: list[Any] = []

        def walk(item: Any) -> None:
            if isinstance(item, list):
                for child in item:
                    walk(child)
            else:
                out.append(item)

        walk(self.value)
        return Tensor(out)


class PhysxView:
    def __init__(self) -> None:
        self.dof = {"positions": [[0.1, 0.2]], "velocities": [[0.0, 0.5]]}
        self.root = {"transforms": [[0.1, -0.5, 0.4, 0.0, 0.0, 0.0, 1.0]], "velocities": [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]}

    def get_dof_positions(self) -> Tensor:
        return Tensor(copy.deepcopy(self.dof["positions"]))

    def get_dof_velocities(self) -> Tensor:
        return Tensor(copy.deepcopy(self.dof["velocities"]))

    def get_root_transforms(self) -> Tensor:
        return Tensor(copy.deepcopy(self.root["transforms"]))

    def get_root_velocities(self) -> Tensor:
        return Tensor(copy.deepcopy(self.root["velocities"]))


def synced_base(*, fabric: bool = True, mutate: bool = False, has_forward: bool = True) -> Any:
    """기존 Chain base에 readiness snapshot·PhysX view·fabric·forward를 더한다."""
    base = ORIGINAL_BASE()
    robot = base.scene["robot"]
    for name, value in (("root_pos_w", [[0.1, -0.5, 0.4]]), ("root_lin_vel_b", [[1.0, 0.0, 0.0]]),
                        ("root_ang_vel_b", [[0.0, 0.0, 0.0]]), ("projected_gravity_b", [[0.0, 0.0, -1.0]])):
        setattr(robot.data, name, Tensor(value))
    view = PhysxView()
    robot.root_physx_view = view
    base.forward_calls = []
    base.sim.is_fabric_enabled = lambda: fabric

    def forward() -> None:
        base.forward_calls.append(base.common_step_counter)
        if mutate:
            view.dof["positions"][0][0] += 1e-3

    if has_forward:
        base.sim.forward = forward
    return base


def direct_factory(tmp_path: Path, events: list[Any], *, cleanup_error: bool = False) -> tuple[Any, Path]:
    job = FIX._write(tmp_path / "stage" / "job.json", {"fixture": True})

    def native_factory(base: Any, rep: Any, render_product: Any, profile: Any) -> tuple[Any, Any]:
        def capture(capture_base: Any, *args: Any) -> dict[str, Any]:
            events.append(("capture", len(capture_base.forward_calls)))
            return {"ordinal": args[-1]}

        def cleanup() -> None:
            events.append("native.cleanup")
            if cleanup_error:
                raise RuntimeError("native cleanup failed")

        return capture, cleanup

    maker = RUN4.render_sync_capture_factory(lambda job_value, app: native_factory)
    return maker({"job_path": str(job)}, object()), job.parent


# --- capture 감싸기 ------------------------------------------------------------------------------


def test_each_capture_runs_forward_once_before_native_and_seals_on_cleanup(tmp_path: Path) -> None:
    events: list[Any] = []
    factory, stage = direct_factory(tmp_path, events)
    base = synced_base()
    capture, cleanup = factory(base, "rep", "rp", {})
    for ordinal in range(3):
        capture(base, "rep", [], {}, [0.0] * 3, [0.0] * 3, ordinal)
    cleanup()
    cleanup()  # 두 번째 cleanup은 증거를 다시 쓰지 않는다
    assert events == [("capture", 1), ("capture", 2), ("capture", 3), "native.cleanup", "native.cleanup"]
    receipt = json.loads((stage / RUN4.RENDER_SYNC_NAME).read_text(encoding="utf-8"))
    assert receipt["status"] == "closed" and receipt["calls"] == 3 and receipt["frames_expected"] == R.FRAME_COUNT
    assert receipt["fabric_enabled"] is True and receipt["physics_unchanged"] is True and receipt["first_error"] is None
    assert receipt["job_sha256"] == R.sha256_file(stage / "job.json")
    with pytest.raises(RUN4.RenderSyncError, match="완료 계약"):  # 625 frame이 아니면 수용하지 않는다
        RUN4.verify_render_sync(stage, frames=R.FRAME_COUNT)


@pytest.mark.parametrize("case", ["no-forward", "fabric-false", "physics-changed"])
def test_sync_failure_blocks_native_capture_and_seals_failure(tmp_path: Path, case: str) -> None:
    events: list[Any] = []
    factory, stage = direct_factory(tmp_path, events)
    base = synced_base(has_forward=case != "no-forward", fabric=case != "fabric-false",
                       mutate=case == "physics-changed")
    capture, cleanup = factory(base, "rep", "rp", {})
    with pytest.raises(RUN4.RenderSyncError):
        capture(base, "rep", [], {}, [0.0] * 3, [0.0] * 3, 0)
    with pytest.raises(RUN4.RenderSyncError, match="이미 실패"):
        capture(base, "rep", [], {}, [0.0] * 3, [0.0] * 3, 1)
    cleanup()
    assert [item for item in events if item != "native.cleanup"] == []  # native capture는 한 번도 불리지 않았다
    receipt = json.loads((stage / RUN4.RENDER_SYNC_NAME).read_text(encoding="utf-8"))
    assert receipt["status"] == "failed" and receipt["calls"] == 0 and receipt["first_error"]["type"] == "RenderSyncError"
    assert receipt["physics_unchanged"] is False and receipt["fabric_enabled"] is (case == "physics-changed")
    with pytest.raises(RUN4.RenderSyncError):
        RUN4.verify_render_sync(stage, frames=0)


def test_native_cleanup_error_still_seals_evidence(tmp_path: Path) -> None:
    events: list[Any] = []
    factory, stage = direct_factory(tmp_path, events, cleanup_error=True)
    base = synced_base()
    capture, cleanup = factory(base, "rep", "rp", {})
    capture(base, "rep", [], {}, [0.0] * 3, [0.0] * 3, 0)
    with pytest.raises(RuntimeError, match="native cleanup failed"):
        cleanup()
    assert json.loads((stage / RUN4.RENDER_SYNC_NAME).read_text(encoding="utf-8"))["calls"] == 1


def test_verify_rejects_missing_and_job_change(tmp_path: Path) -> None:
    stage = tmp_path / "stage"
    stage.mkdir()
    with pytest.raises(RUN4.RenderSyncError, match="없다"):
        RUN4.verify_render_sync(stage, frames=R.FRAME_COUNT)
    FIX._write(stage / "job.json", {"fixture": True})
    FIX._write(stage / RUN4.RENDER_SYNC_NAME, {
        "schema": RUN4.SCHEMA, "method": RUN4.METHOD, "status": "closed", "calls": R.FRAME_COUNT,
        "frames_expected": R.FRAME_COUNT, "fabric_enabled": True, "physics_unchanged": True,
        "snapshot_chain_sha256": "0" * 64, "first_error": None, "job_sha256": R.sha256_file(stage / "job.json")})
    assert RUN4.verify_render_sync(stage, frames=R.FRAME_COUNT)["calls"] == R.FRAME_COUNT
    FIX._write(stage / "job.json", {"fixture": "changed"})
    with pytest.raises(RUN4.RenderSyncError, match="완료 계약"):
        RUN4.verify_render_sync(stage, frames=R.FRAME_COUNT)


# --- scope·entry ----------------------------------------------------------------------------------


def test_render_sync_scope_swaps_once_and_restores() -> None:
    with RUN4.render_sync_scope():
        assert CAPTURE.make_capture_factory is not REAL_MAKER
        with pytest.raises(RUN4.RenderSyncError, match="이미 교체"), RUN4.render_sync_scope():
            pass
    assert CAPTURE.make_capture_factory is REAL_MAKER
    with pytest.raises(RuntimeError, match="boom"), RUN4.render_sync_scope():
        raise RuntimeError("boom")
    assert CAPTURE.make_capture_factory is REAL_MAKER
    CAPTURE.make_capture_factory = lambda job, app: None
    try:
        with pytest.raises(RUN4.RenderSyncError, match="이미 교체"), RUN4.render_sync_scope():
            pass
    finally:
        CAPTURE.make_capture_factory = REAL_MAKER


def test_retarget_requires_exactly_one_run2_entry_and_jobs2_scope() -> None:
    seen: list[list[str]] = []

    def original(popen: Any, job_path: Path, *, before_launch: Any = None) -> Any:
        return popen(["py", "-B", str(JOB2.RUN2_ENTRY), "--sdk-child"])

    RUN4._retarget(original)(lambda command, **_k: seen.append(command), Path("job.json"))
    assert seen == [["py", "-B", ENTRY4, "--sdk-child"]]
    for command in (["py", "-B", "bench/record_agentic_state_sidecar_v2.py"], ["py", str(JOB2.RUN2_ENTRY), str(JOB2.RUN2_ENTRY)]):
        with pytest.raises(ValueError, match="RUN2 entry"):
            RUN4._retarget(lambda popen, job, *, before_launch=None, c=command: popen(c))(lambda *_a, **_k: None, Path("j"))
    with pytest.raises(RUN4.RenderSyncError, match="JOB2"), RUN4._entry_scope():
        pass


def test_child_entry_only_sdk_child_enters_sync_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[Any] = []
    monkeypatch.setattr(RUN2, "main", lambda argv: seen.append(("main", argv[0], RUN4._STATE["sync_active"])) or 0)
    monkeypatch.setattr(RUN2, "child_main", lambda flag, job: seen.append(("child", flag, RUN4._STATE["sync_active"])) or 0)
    RUN4.main(["--aaba-child", "--job", "j"])
    RUN4.main(["--sdk-child", "--job", "j"])
    RUN4.child_main("--aaba-child", Path("j"))
    RUN4.child_main("--sdk-child", Path("j"))
    assert seen == [("main", "--aaba-child", False), ("main", "--sdk-child", True),
                    ("child", "--aaba-child", False), ("child", "--sdk-child", True)]
    assert CAPTURE.make_capture_factory is REAL_MAKER


# --- 실제 RUN5·sidecar 흐름 ----------------------------------------------------------------------


class ChainV4(R3T.ChainV3):
    """ChainV3(실제 reader·_gpu_reference)에 RUN4 자식 entry와 렌더 sync fake base를 더한다."""

    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, child: str = "v4", **base_options: Any) -> None:
        super().__init__(tmp_path, monkeypatch)
        self.child = child
        self.bases: list[Any] = []
        self.fake_maker = self._fake_maker  # bound method는 접근마다 새 객체라 한 번만 잡는다
        monkeypatch.setattr(CAPTURE, "make_capture_factory", self.fake_maker)  # Chain native factory를 기준 maker로 둔다
        monkeypatch.setattr(RUN4, "_BASE_MAKER", self.fake_maker)

        def build() -> Any:
            self.bases.append(synced_base(**base_options))
            return self.bases[-1]

        monkeypatch.setattr(RT, "_base", build)

    def _fake_maker(self, validated: Any, app: Any) -> Any:
        return self.native_factory

    def popen(self, command: list[str], **kwargs: Any) -> Any:
        if command[8] != "--sdk-child":
            return RT.Chain.popen(self, command, **kwargs)
        self.commands.append(list(command))
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "0" and kwargs["shell"] is False
        entry = RUN4.main if self.child == "v4" else RUN2.main
        with R2T.fresh_child(self.layout):
            code = entry([command[8], "--job", command[10]])
        return RT.FakeProcess(code)


def test_real_flow_launches_v4_children_and_seals_625_syncs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    chain = ChainV4(tmp_path, monkeypatch)
    launcher = RUN1._child_launcher
    job, final = chain.run_real_reader(FIX.NONE_ID, runner=RUN4.run_job)
    assert [command[7] for command in chain.commands] == [ENTRY4, ENTRY4]
    assert [command[8] for command in chain.commands] == ["--aaba-child", "--sdk-child"]
    assert final["final_technical_complete"] is True and final["live_evidence"] is False
    assert len(chain.bases) == 1 and len(chain.bases[0].forward_calls) == R.FRAME_COUNT
    receipt = RUN4.verify_render_sync(job.parent, frames=R.FRAME_COUNT)
    acceptance = json.loads((job.parent / RUN4.ACCEPTANCE_NAME).read_text(encoding="utf-8"))
    assert acceptance["render_sync"] == {"verified": True, **receipt}
    assert acceptance["final_sha256"] == R.sha256_file(job.parent / RUN4.FINAL_NAME)
    assert RUN1._child_launcher is launcher and RUN1._leased_reader is ORIGINAL_LEASED_READER
    assert CAPTURE.make_capture_factory is chain.fake_maker  # 자식 scope가 끝나 기준 maker로 원복됐다
    assert not RUN4._STATE["sync_active"] and RUN4._STATE["entry_depth"] == 0


def test_child_falling_back_to_run2_has_no_evidence_and_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    chain = ChainV4(tmp_path, monkeypatch, child="v2")
    with pytest.raises(RUN4.RenderSyncError, match="render-sync 증거가 없다"):
        chain.run_real_reader(FIX.NONE_ID, runner=RUN4.run_job)
    stage = tmp_path / FIX.NONE_ID
    acceptance = json.loads((stage / RUN4.ACCEPTANCE_NAME).read_text(encoding="utf-8"))
    assert acceptance["final_technical_complete"] is True and acceptance["render_sync"]["verified"] is False
    assert json.loads((stage / RUN4.FINAL_NAME).read_text(encoding="utf-8"))["final_technical_complete"] is True


def test_fabric_disabled_fails_record_and_keeps_final(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    chain = ChainV4(tmp_path, monkeypatch, fabric=False)
    job, final = chain.run_real_reader(FIX.NONE_ID, runner=RUN4.run_job)
    assert final["final_technical_complete"] is False
    receipt = json.loads((job.parent / RUN4.RENDER_SYNC_NAME).read_text(encoding="utf-8"))
    assert receipt["status"] == "failed" and receipt["calls"] == 0
    acceptance = json.loads((job.parent / RUN4.ACCEPTANCE_NAME).read_text(encoding="utf-8"))
    assert acceptance["render_sync"]["verified"] is False


def test_live_judgement_is_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    chain = ChainV4(tmp_path, monkeypatch)
    _job, final = chain.run_real_reader(FIX.F2_ID, runner=RUN4.run_job, production=True)
    assert final["final_technical_complete"] is True and final["live_evidence"] is True
    assert final["acceptable_as_live_evidence"] is (final["lease"]["live_lock_proof"] is True)
    assert final["behavior_label"] is None and final["used_for_decision"] is False


def test_constants_follow_run1() -> None:
    assert RUN4.PRODUCTION is RUN1.PRODUCTION and RUN4.FINAL_NAME == RUN1.FINAL_NAME
    assert set(RUN4.SOURCE_FILES) == {"bench/run_agentic_play_record_v4.py", "tests/test_run_agentic_play_record_v4.py"}
