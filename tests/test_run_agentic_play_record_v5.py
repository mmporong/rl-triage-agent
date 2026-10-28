"""RUN5: F1·정상 정책이 RUN4 렌더 sync·RUN3 reader·JOB2 lease·sidecar·AABA 흐름을 그대로 지나고, 자식은 RUN5 entry로 온다."""
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

import agentic_f1_playback_v1 as PB  # pyright: ignore[reportMissingImports]
import agentic_play_job_v2 as JOB2  # pyright: ignore[reportMissingImports]
import agentic_play_job_v3 as JOB3  # pyright: ignore[reportMissingImports]
import record_vision_evidence_v1 as R  # pyright: ignore[reportMissingImports]
import run_agentic_play_record_v1 as RUN1  # pyright: ignore[reportMissingImports]
import run_agentic_play_record_v4 as RUN4  # pyright: ignore[reportMissingImports]
import run_agentic_play_record_v5 as RUN5  # pyright: ignore[reportMissingImports]
import test_agentic_play_job_v2 as T2  # pyright: ignore[reportMissingImports]
import test_agentic_play_job_v3 as J3T  # pyright: ignore[reportMissingImports]
import test_run_agentic_play_record_v1 as RT  # pyright: ignore[reportMissingImports]
import test_run_agentic_play_record_v2 as R2T  # pyright: ignore[reportMissingImports]
import test_run_agentic_play_record_v4 as R4T  # pyright: ignore[reportMissingImports]
from test_agentic_play_job_v1 import _close_leases  # noqa: F401  # pyright: ignore[reportMissingImports]

ENTRY5 = str(RUN5.ENTRY)
BASE = (RUN1.PLAYBACK, RUN1._clip_consistent, RUN4.ENTRY)
NONE_CLIP = {"actuators": {"base_legs": {"clip_effort_function": RUN1.NONE_CLIP_FUNCTION}}}
F2_CLIP = {"actuators": {"base_legs": {"clip_effort_function": "hidden_faults._before_env.<locals>._clip_effort"}}}


def _current() -> tuple[Any, Any, Any]:
    return RUN1.PLAYBACK, RUN1._clip_consistent, RUN4.ENTRY


@contextmanager
def fresh_child_f1(layout: T2.LinkedHome) -> Iterator[None]:
    """새 자식 프로세스의 import 직후 상태: JOB2·F1·RUN5 scope가 모두 없다."""
    state = dict(RUN5._STATE)
    saved = _current()
    RUN5._STATE.update(depth=0)
    RUN1.PLAYBACK, RUN1._clip_consistent, RUN4.ENTRY = BASE
    try:
        with R2T.fresh_child(layout), J3T.f1_fresh():
            yield
    finally:
        RUN5._STATE.update(state)
        RUN1.PLAYBACK, RUN1._clip_consistent, RUN4.ENTRY = saved


def test_scope_swaps_playback_clip_entry_and_restores() -> None:
    with RUN5.f1_record_scope():
        with RUN5.f1_record_scope():
            assert _current() == (PB, RUN5.clip_consistent, RUN5.ENTRY) and JOB3.scope_active()
        assert RUN5._STATE["depth"] == 1
    assert _current() == BASE and not JOB3.scope_active()


def test_scope_refuses_foreign_swap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(RUN4, "ENTRY", Path("other.py"))
    with pytest.raises(RuntimeError, match="이미 교체"), RUN5.f1_record_scope():
        pass
    assert RUN1.PLAYBACK is BASE[0] and not JOB3.scope_active()


def test_clip_rule_maps_f1_to_original_motor_function() -> None:
    assert RUN5.clip_consistent(NONE_CLIP, "F1") is True and RUN5.clip_consistent(F2_CLIP, "F1") is False
    assert RUN1._clip_consistent(F2_CLIP, "F1") is True  # 기존 규칙은 F1에서 어떤 함수든 통과시킨다
    assert RUN5.clip_consistent(F2_CLIP, "F2") is True and RUN5.clip_consistent(NONE_CLIP, "NONE") is True


def test_child_command_chain_reaches_run5_entry_only_inside_all_scopes(tmp_path: Path) -> None:
    job = str((tmp_path / "job.json").absolute())
    command = ["py", "-I", "-S", "-c", "gate", "py", "-B", RT._run5(), "--sdk-child", "--job", job]
    seen: list[list[str]] = []
    with JOB2.canonical_lease_scope(), RUN5.f1_record_scope(), RUN4._entry_scope():
        RUN1._child_launcher(lambda values, **_k: seen.append(values), Path(job))(list(command))
    with JOB2.canonical_lease_scope(), RUN4._entry_scope():  # F1 scope가 없으면 RUN4 entry에서 멈춘다
        RUN1._child_launcher(lambda values, **_k: seen.append(values), Path(job))(list(command))
    assert [values[7] for values in seen] == [ENTRY5, str(R4T.RUN4.ENTRY)]
    assert _current() == BASE


class ChainF1(R4T.ChainV4):
    """ChainV4(실제 reader·render sync fake base)에 F1 world와 RUN5 자식 entry를 넣는다."""

    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, child: str = "v5",
                 installer: Any = None, **base_options: Any) -> None:
        monkeypatch.setattr(T2, "build_world_v2", J3T.build_world_f1)
        super().__init__(tmp_path, monkeypatch, **base_options)
        self.child_f1 = child
        self.child_errors: list[str] = []
        make = PB.make_run_real
        f1_installer = installer or (lambda checked: {"fault_id": checked["fault_id"], "class_method_changed": False,
                                                       "playback_injection": "none"})
        monkeypatch.setattr(PB, "make_run_real", lambda registration: make(
            registration, runner=self.play,
            installer=self.install if registration["fault_id"] == "NONE" else f1_installer))

    def popen(self, command: list[str], **kwargs: Any) -> Any:
        if command[8] != "--sdk-child":
            return RT.Chain.popen(self, command, **kwargs)
        self.commands.append(list(command))
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "0" and kwargs["shell"] is False
        entry = RUN5.main if self.child_f1 == "v5" else RUN4.main
        with fresh_child_f1(self.layout):
            try:
                code = entry([command[8], "--job", command[10]])
            except Exception as error:  # noqa: BLE001 - 실제 자식이면 잡히지 않은 예외로 exit 1이 된다
                self.child_errors.append(f"{type(error).__name__}: {error}")
                code = 1
        return RT.FakeProcess(code)


@pytest.mark.parametrize(("identity", "fault"), [(J3T.F1_ID, "F1"), (J3T.NONE_ID, "NONE")])
def test_real_flow_records_f1_pair_with_render_sync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, identity: str, fault: str
) -> None:
    chain = ChainF1(tmp_path, monkeypatch)
    job, final = chain.run_real_reader(identity, runner=RUN5.run_job)
    assert chain.child_errors == []
    assert [command[7] for command in chain.commands] == [ENTRY5, ENTRY5]
    assert [command[8] for command in chain.commands] == ["--aaba-child", "--sdk-child"]
    assert final["final_technical_complete"] is True and final["live_evidence"] is False
    assert final["fault_id"] == fault and final["pair"] == J3T.PAIR and final["interpretation"] == JOB3.INTERPRETATION
    assert final["actuator_clip_consistent_with_registration"] is True and final["behavior_label"] is None
    assert ("play.clip", "isaaclab.actuators.actuator_pd") in chain.events
    playback = json.loads((job.parent / "playback-receipt.json").read_text(encoding="utf-8"))
    assert playback["installation"]["fault_id"] == fault and playback["installation"]["class_method_changed"] is False
    receipt = RUN4.verify_render_sync(job.parent, frames=R.FRAME_COUNT)
    acceptance = json.loads((job.parent / RUN4.ACCEPTANCE_NAME).read_text(encoding="utf-8"))
    assert acceptance["render_sync"] == {"verified": True, **receipt}
    assert len(chain.bases[0].forward_calls) == R.FRAME_COUNT
    assert _current() == BASE and RUN5._STATE["depth"] == 0 and RUN4._STATE["entry_depth"] == 0


def test_child_falling_back_to_run4_cannot_validate_f1_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    chain = ChainF1(tmp_path, monkeypatch, child="v4")
    job, final = chain.run_real_reader(J3T.F1_ID, runner=RUN5.run_job)
    execution = json.loads((job.parent / "execution.json").read_text(encoding="utf-8"))
    assert final["final_technical_complete"] is False and execution["completed"] is False
    assert len(chain.child_errors) == 1 and "agentic play job" in chain.child_errors[0]
    assert not (job.parent / RUN4.RENDER_SYNC_NAME).exists()
    acceptance = json.loads((job.parent / RUN4.ACCEPTANCE_NAME).read_text(encoding="utf-8"))
    assert acceptance["render_sync"]["verified"] is False


def test_unexpected_motor_change_in_f1_play_is_not_complete(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def inject(checked: Any) -> Any:  # F1 재생에 모터 함수가 바뀌어 들어온 상황
        evidence, restore = RT.Chain.install(chain, dict(checked, fault_id="F2"))
        monkeypatch.setattr(chain, "restore_f1", restore, raising=False)
        return dict(evidence, fault_id="F1")

    chain = ChainF1(tmp_path, monkeypatch, installer=lambda checked: inject(checked))
    _job, final = chain.run_real_reader(J3T.F1_ID, runner=RUN5.run_job)
    chain.restore_f1()  # type: ignore[attr-defined]
    assert final["actuator_clip_consistent_with_registration"] is False
    assert final["final_technical_complete"] is False


def test_entry_points_open_f1_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[Any] = []
    monkeypatch.setattr(RUN4, "main", lambda argv=None: seen.append(("main", _current()[0] is PB)) or 3)
    monkeypatch.setattr(RUN4, "child_main", lambda flag, job: seen.append(("child", _current()[0] is PB)) or 4)
    assert RUN5.main(["--aaba-child", "--job", "j"]) == 3 and RUN5.child_main("--sdk-child", Path("j")) == 4
    assert seen == [("main", True), ("child", True)] and _current() == BASE
    assert RUN5.PRODUCTION is RUN1.PRODUCTION and RUN5.FINAL_NAME == RUN1.FINAL_NAME
