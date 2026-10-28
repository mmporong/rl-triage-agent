"""agentic play 녹화 실행기 v4: native capture 직전에 ``SimulationContext.forward``를 1회 불러 렌더 장면을 맞춘다.

v9 원문에서 렌더된 로봇 형상이 0번 frame 위치에 머물렀다(카메라·sim root는 정상). 이 판은 카메라를 바꾸지 않고
record 자식의 capture 직전에만 렌더용 articulation 갱신을 요청한다. 원인과 효과는 GPU 실행 전 미확정이다.

- 자식(``--sdk-child``)만 ``render_sync_scope()`` 안에서 sidecar worker가 조회하는 capture factory를 감싼다.
  capture마다 fabric 활성 확인 → 물리 snapshot → ``sim.forward()`` 1회 → 같은 snapshot 재확인 → 원래 capture 순서다.
- cleanup(``run_real`` finally, sidecar 봉인·SDK close 전)에서 ``render-sync.json``을 write-once로 남긴다.
- 부모는 RUN3 GPU reader scope와 JOB2 canonical scope 안에서 자식 entry를 RUN4로 재지정하고 RUN1을 그대로 부른다.
  끝난 뒤 final 원문은 두고 ``render-sync-acceptance.json``을 따로 쓴다.
"""

from __future__ import annotations

import hashlib
import json
import sys
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _folder in (ROOT / "bench", ROOT / "evals", ROOT / "src"):
    if str(_folder) not in sys.path:
        sys.path.insert(0, str(_folder))

import agentic_play_job_v2 as JOB2
import record_vision_evidence_v1 as R
import run_agentic_play_record_v1 as RUN1
import run_agentic_play_record_v2 as RUN2
import run_agentic_play_record_v3 as RUN3
import vision_capture_readiness_v1 as READY
import vision_pull_record_capture_v2 as CAPTURE

SOURCE_FILES = ("bench/run_agentic_play_record_v4.py", "tests/test_run_agentic_play_record_v4.py")
PRODUCTION = RUN1.PRODUCTION
FINAL_NAME = RUN1.FINAL_NAME
RENDER_SYNC_NAME = "render-sync.json"
ACCEPTANCE_NAME = "render-sync-acceptance.json"
SCHEMA = "agentic_render_sync_v1"
ACCEPTANCE_SCHEMA = "agentic_render_sync_acceptance_v1"
METHOD = "isaaclab.sim.SimulationContext.forward"
ENTRY = Path(__file__).resolve()
_SCOPE_LOCK = threading.RLock()
_ENTRY_LOCK = threading.RLock()
_BASE_MAKER = CAPTURE.make_capture_factory
_STATE: dict[str, Any] = {"sync_active": False, "entry_depth": 0, "entry_saved": None}


class RenderSyncError(RuntimeError):
    """렌더 sync 계약 위반. 성공으로 바꾸지 않는다."""


def _snapshot(base: Any) -> dict[str, Any]:
    view = base.scene["robot"].root_physx_view
    return {
        "readiness": READY._physics_snapshot(base),
        "dof_positions": READY._tensor_values(view.get_dof_positions()[0]),
        "dof_velocities": READY._tensor_values(view.get_dof_velocities()[0]),
        "root_transform": READY._tensor_values(view.get_root_transforms()[0]),
        "root_velocity": READY._tensor_values(view.get_root_velocities()[0]),
    }


def _sync(base: Any, state: dict[str, Any]) -> str:
    """fabric 확인 → 전 snapshot → forward 1회 → 후 snapshot 동일 확인. 같은 snapshot의 SHA를 돌려준다."""
    sim = base.sim
    enabled = getattr(sim, "is_fabric_enabled", None)
    forward = getattr(sim, "forward", None)
    if not callable(enabled) or not callable(forward):
        raise RenderSyncError("SimulationContext에 is_fabric_enabled·forward가 없다")
    if enabled() is not True:
        state["fabric_enabled"] = False
        raise RenderSyncError("fabric이 활성 상태가 아니다")
    if state["fabric_enabled"] is None:
        state["fabric_enabled"] = True
    before = _snapshot(base)
    forward()
    after = _snapshot(base)
    if after != before:
        raise RenderSyncError("forward 전후 물리 snapshot이 다르다")
    return hashlib.sha256(json.dumps(before, sort_keys=True, allow_nan=False).encode()).hexdigest()


def render_sync_capture_factory(maker: Callable[[Mapping[str, Any], Any], Any]) -> Callable[..., Any]:
    """``(job, app) -> factory`` maker를 감싼다. capture마다 렌더 sync를 먼저 하고 cleanup에서 증거를 봉인한다."""

    def synced_maker(job: Mapping[str, Any], app: Any) -> Callable[..., Any]:
        job_path = Path(job["job_path"])
        stage, job_sha = job_path.parent, R.sha256_file(job_path)
        factory = maker(job, app)

        def synced_factory(base: Any, rep: Any, render_product: Any, profile: Mapping[str, Any]) -> tuple[Any, Any]:
            capture, cleanup = factory(base, rep, render_product, profile)
            if not callable(capture) or not callable(cleanup):
                raise TypeError("capture factory는 callback과 cleanup을 반환해야 한다")
            state: dict[str, Any] = {"calls": 0, "chain": hashlib.sha256(), "error": None, "sealed": False,
                                     "fabric_enabled": None}

            def synced_capture(capture_base: Any, *args: Any) -> Any:
                if state["error"] is None:
                    try:
                        digest = _sync(capture_base, state)
                    except BaseException as error:  # 첫 실패를 기록하고 다시 올려 capture를 멈춘다.
                        state["error"] = {"type": type(error).__name__, "message": str(error)}
                        raise
                    state["calls"] += 1
                    state["chain"].update(digest.encode())
                    return capture(capture_base, *args)
                raise RenderSyncError("렌더 sync가 이미 실패했다")

            def synced_cleanup() -> None:
                try:
                    cleanup()
                finally:
                    if not state["sealed"]:
                        state["sealed"] = True
                        failed = state["error"] is not None
                        R.write_new(stage / RENDER_SYNC_NAME, {
                            "schema": SCHEMA, "method": METHOD, "status": "failed" if failed else "closed",
                            "calls": state["calls"], "frames_expected": R.FRAME_COUNT,
                            "fabric_enabled": state["fabric_enabled"] is True,
                            "physics_unchanged": not failed and state["calls"] > 0,
                            "snapshot_chain_sha256": state["chain"].hexdigest(),
                            "first_error": state["error"], "job_sha256": job_sha,
                        })

            return synced_capture, synced_cleanup

        return synced_factory

    return synced_maker


@contextmanager
def render_sync_scope() -> Iterator[None]:
    with _SCOPE_LOCK:
        if _STATE["sync_active"] or CAPTURE.make_capture_factory is not _BASE_MAKER:
            raise RenderSyncError("capture factory가 이미 교체됐다")
        CAPTURE.make_capture_factory = render_sync_capture_factory(_BASE_MAKER)
        _STATE["sync_active"] = True
        try:
            yield
        finally:
            CAPTURE.make_capture_factory = _BASE_MAKER
            _STATE["sync_active"] = False


def verify_render_sync(stage: Path, *, frames: int) -> dict[str, Any]:
    path = Path(stage) / RENDER_SYNC_NAME
    if path.is_symlink() or not path.is_file():
        raise RenderSyncError("render-sync 증거가 없다")
    try:
        value = R.read_object(path)
    except (OSError, TypeError, ValueError) as error:
        raise RenderSyncError(f"render-sync 증거를 읽을 수 없다: {error}") from error
    job_sha = R.sha256_file(Path(stage) / "job.json")
    if (value.get("schema") != SCHEMA or value.get("method") != METHOD or value.get("status") != "closed"
            or value.get("calls") != frames or value.get("frames_expected") != frames
            or value.get("fabric_enabled") is not True or value.get("physics_unchanged") is not True
            or value.get("first_error") is not None or value.get("job_sha256") != job_sha):
        raise RenderSyncError("render-sync 증거가 완료 계약과 다르다")
    return {"path": str(path), "sha256": R.sha256_file(path), "calls": value["calls"],
            "snapshot_chain_sha256": value["snapshot_chain_sha256"]}


def _retarget(original: Callable[..., Any]) -> Callable[..., Any]:
    old, new = str(JOB2.RUN2_ENTRY), str(ENTRY)

    def launcher(popen: Callable[..., Any], job_path: Path, *, before_launch: Callable[[], Any] | None = None) -> Any:
        def retarget(command: Any, **kwargs: Any) -> Any:
            values = list(command)
            if values.count(old) != 1:
                raise ValueError("v4 entry로 바꿀 RUN2 entry가 정확히 하나가 아니다")
            values[values.index(old)] = new
            return popen(values, **kwargs)

        return original(retarget, job_path, before_launch=before_launch)

    return launcher


@contextmanager
def _entry_scope() -> Iterator[None]:
    with _ENTRY_LOCK:
        if _STATE["entry_depth"] == 0:
            if JOB2.scope_active_root() is None:
                raise RenderSyncError("자식 entry 재지정은 JOB2 canonical scope 안에서만 한다")
            _STATE["entry_saved"] = RUN1._child_launcher
            RUN1._child_launcher = _retarget(_STATE["entry_saved"])
        _STATE["entry_depth"] += 1
        try:
            yield
        finally:
            _STATE["entry_depth"] -= 1
            if _STATE["entry_depth"] == 0:
                RUN1._child_launcher = _STATE["entry_saved"]
                _STATE["entry_saved"] = None


def run_job(job_path: Path, **kwargs: Any) -> Mapping[str, Any]:
    with JOB2.canonical_lease_scope(), RUN3.leased_reader_scope(), _entry_scope():
        final = RUN1.run_job(job_path, **kwargs)
    stage = Path(job_path).absolute().parent
    try:
        render_sync: dict[str, Any] = {"verified": True, **verify_render_sync(stage, frames=R.FRAME_COUNT)}
    except RenderSyncError as error:
        render_sync = {"verified": False, "error": str(error)}
    R.write_new(stage / ACCEPTANCE_NAME, {
        "schema": ACCEPTANCE_SCHEMA, "final_sha256": R.sha256_file(stage / FINAL_NAME),
        "final_technical_complete": final.get("final_technical_complete") is True, "render_sync": render_sync,
    })
    if final.get("final_technical_complete") is True and render_sync["verified"] is not True:
        raise RenderSyncError(f"기술 완료 녹화에 render-sync 증거가 맞지 않는다: {render_sync['error']}")
    return final


def child_main(flag: str, job_path: Path) -> int:
    if flag != "--sdk-child":
        return RUN2.child_main(flag, job_path)
    with render_sync_scope():
        return RUN2.child_main(flag, job_path)


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if "--sdk-child" not in args:
        return RUN2.main(args)
    with render_sync_scope():
        return RUN2.main(args)


if __name__ == "__main__":
    raise SystemExit(main())
