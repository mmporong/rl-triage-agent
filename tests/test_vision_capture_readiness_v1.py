from __future__ import annotations

import importlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "bench"), str(ROOT / "evals")]
C = importlib.import_module("vision_capture_readiness_v1")


class Tensor:
    def __init__(self, values: list[float]) -> None:
        self.values = values

    def __getitem__(self, key: object) -> Tensor:
        del key
        return self

    def detach(self) -> Tensor:
        return self

    def cpu(self) -> Tensor:
        return self

    def flatten(self) -> Tensor:
        return self

    def tolist(self) -> list[float]:
        return self.values


class Sim:
    def __init__(self, events: list[str]) -> None:
        self.current_time = 4.0
        self.events = events
        self.play_simulations = True

    def get_setting(self, name: str) -> bool:
        assert name == C.PLAYER_SETTING
        return self.play_simulations

    def set_setting(self, name: str, value: bool) -> None:
        assert name == C.PLAYER_SETTING
        self.play_simulations = value

    def set_camera_view(self, **kwargs: object) -> None:
        del kwargs
        self.events.append("camera")


class Base:
    def __init__(self, events: list[str]) -> None:
        self.sim = Sim(events)
        self.common_step_counter = 7
        data = SimpleNamespace(
            root_pos_w=Tensor([1.0, 2.0, 3.0]),
            root_lin_vel_b=Tensor([0.1, 0.2, 0.3]),
            root_ang_vel_b=Tensor([0.4, 0.5, 0.6]),
            projected_gravity_b=Tensor([0.0, 0.0, -1.0]),
        )
        self.scene = {"robot": SimpleNamespace(data=data)}


class EventStream:
    def __init__(self) -> None:
        self.callback: Any = None

    def create_subscription_to_pop(self, callback: Any, *, name: str) -> object:
        assert name == "rl-triage-capture-readiness"
        self.callback = callback
        return object()


class Context:
    def __init__(
        self, events: list[str], statuses: list[tuple[str, int, int, bool]],
        events_by_update: dict[int, list[int]] | None = None,
    ) -> None:
        self.events = events
        self.statuses = statuses
        self.index = 0
        self.stream = EventStream()
        self.events_by_update = events_by_update or {1: [1], 2: [2]}
        self.setting_reads: list[str] = []

    def advance(self) -> None:
        self.index = min(self.index + 1, len(self.statuses) - 1)
        if self.stream.callback is not None:
            for event_type in self.events_by_update.get(self.index, []):
                self.stream.callback(SimpleNamespace(type=event_type))

    def get_stage_loading_status(self) -> tuple[str, int, int]:
        row = self.statuses[self.index]
        return row[0], row[1], row[2]

    def get_stage_streaming_status(self) -> bool:
        return self.statuses[self.index][3]

    def get_stage_event_stream(self) -> EventStream:
        return self.stream

    def reset_renderer_accumulation(self) -> None:
        self.events.append("reset")

    def get_attached_hydra_engine_names(self) -> list[str]:
        return ["rtx"]


class App:
    def __init__(
        self, context: Context, base: Base, *, mutate: bool = False,
        update_error: BaseException | None = None,
    ) -> None:
        self.context = context
        self.base = base
        self.mutate = mutate
        self.update_error = update_error

    def update(self) -> None:
        assert self.base.sim.play_simulations is False
        self.context.events.append("update")
        if self.update_error is not None:
            raise self.update_error
        self.context.advance()
        if self.mutate:
            self.base.sim.current_time += 0.1


EVENT_TYPES = SimpleNamespace(ASSETS_LOADING=1, ASSETS_LOADED=2, ASSETS_LOAD_ABORTED=3)
PROFILE = {
    "camera_prim_path": "/OmniverseKit_Persp", "capture_primitive": "replicator_orchestrator_step_v1",
    "rt_subframes": 1, "pause_timeline": False, "delta_time_s": 0.0,
}


def _install_capture(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, mode: str = "record",
    statuses: list[tuple[str, int, int, bool]] | None = None,
    events_by_update: dict[int, list[int]] | None = None,
) -> tuple[dict[str, Any], Context, Base, App, list[str]]:
    events: list[str] = []
    context = Context(events, statuses or [
        ("loading", 1, 2, True), ("loading", 2, 2, True),
        ("", 0, 0, False), ("", 0, 0, False), ("", 0, 0, False),
    ], events_by_update)
    base = Base(events)
    app = App(context, base)
    job = tmp_path / "job.json"
    job.write_text("{}\n", encoding="utf-8")
    validated = {"job": {"mode": mode}, "job_path": job, "output_directory": tmp_path / "output"}
    monkeypatch.setattr(C, "_usd_context", lambda: (context, EVENT_TYPES))
    settings = SimpleNamespace(get=lambda key: context.setting_reads.append(key))
    monkeypatch.setattr(C, "_carb_settings", lambda: settings)

    def journal_factory(current: Mapping[str, Any], *, preserve_rgb: bool) -> Any:
        del current
        events.append(f"journal-smoke-{preserve_rgb}")

        def journal(
            proxy: Any, rep: Any, annotators: list[Any], camera_profile: dict[str, Any],
            eye: list[float], lookat: list[float], ordinal: int,
        ) -> dict[str, Any]:
            del rep, annotators, eye, lookat, ordinal
            proxy.sim.set_camera_view(camera_prim_path=camera_profile["camera_prim_path"])
            events.append("orchestrator-step")
            return {"rgb": b"rgb", "read_times": {}, "reference_time": {}}

        return journal

    monkeypatch.setattr(C.V2, "_journaled_camera_capture", journal_factory)
    return validated, context, base, app, events


@pytest.mark.parametrize("mode,selected,smoke", [("record", "real", False), (C.R.RENDER_SYNC_SMOKE_MODE, "smoke", True)])
def test_run_capture_waits_once_resets_each_camera_and_preserves_runner_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str, selected: str, smoke: bool,
) -> None:
    validated, context, base, app, events = _install_capture(monkeypatch, tmp_path, mode=mode)
    primary = tmp_path / "primary.json"
    primary.write_text("{}\n", encoding="utf-8")

    def runner(current: dict[str, Any], selected_app: App, *, camera_capture: Any) -> tuple[dict[str, Path], Path]:
        assert current is validated and selected_app is app
        events.append(selected)
        for ordinal in range(2):
            camera_capture(base, object(), [], PROFILE, [1.0] * 3, [0.0] * 3, ordinal)
        return {"primary": primary}, primary

    monkeypatch.setattr(C.V1, "run_real" if selected == "real" else "run_render_sync_smoke", runner)
    artifacts, returned = C.run_capture(validated, app)

    assert artifacts == {"primary": primary} and returned == primary
    assert events.count("update") == 4 and events.count("camera") == events.count("reset") == 2
    assert events.index("camera") < events.index("reset") < events.index("orchestrator-step")
    assert events[0] == f"journal-smoke-{smoke}" and context.index == 4
    receipt = json.loads((tmp_path / "job.capture-readiness.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "ready"
    assert [row["event"] for row in receipt["event_history"]] == ["assets_loading", "assets_loaded"]
    assert receipt["event_counts"] == {"loading": 1, "loaded": 1, "aborted": 0, "synthetic_loading": 0}
    assert receipt["physics_before"] == receipt["physics_after"]
    assert receipt["renderer"]["attached_hydra_engine_names"] == ["rtx"]
    assert receipt["renderer"]["carb_settings"] == {key: None for key in C.CARB_SETTING_KEYS}
    assert context.setting_reads == list(C.CARB_SETTING_KEYS)
    assert receipt["bounds"] == {
        "max_updates": C.MAX_UPDATES, "polling_elapsed_budget_s": C.TIMEOUT_S,
        "polling_elapsed_budget_scope": "checked_between_synchronous_app_updates_not_a_hard_watchdog",
        "required_consecutive_idle_updates": C.REQUIRED_IDLE_UPDATES,
    }
    for ordinal in range(2):
        reset = json.loads((tmp_path / f"job.capture-{ordinal:06d}.renderer-reset.json").read_text(encoding="utf-8"))
        assert reset["completed"] is True and reset["capture_ordinal"] == ordinal


def test_timeout_stops_before_camera_or_orchestrator(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, _, base, app, events = _install_capture(
        monkeypatch, tmp_path, statuses=[("loading", 1, 2, True)],
    )
    monkeypatch.setattr(C, "MAX_UPDATES", 2)
    receipt = tmp_path / "timeout.json"
    with pytest.raises(TimeoutError, match="상한"):
        C._wait_capture_ready(base, app, app.context, EVENT_TYPES, receipt, {})
    assert events == ["update", "update"]
    assert json.loads(receipt.read_text(encoding="utf-8"))["reason"] == "poll_budget_exhausted"


def test_app_update_physics_change_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, context, base, _, _ = _install_capture(
        monkeypatch, tmp_path, statuses=[("", 0, 0, False)] * 4,
    )
    app = App(context, base, mutate=True)
    receipt = tmp_path / "physics.json"
    with pytest.raises(RuntimeError, match="physics 상태|simulation time"):
        C._wait_capture_ready(base, app, context, EVENT_TYPES, receipt, {})
    body = json.loads(receipt.read_text(encoding="utf-8"))
    assert body["reason"] == "physics_or_simulation_time_changed"
    assert body["physics_after"]["simulation_time_s"] > body["physics_before"]["simulation_time_s"]


def test_app_update_error_preserves_error_and_seals_final_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, context, base, _, _ = _install_capture(
        monkeypatch, tmp_path, statuses=[("", 0, 0, False)] * 4,
    )
    app = App(context, base, update_error=ValueError("update failed"))
    receipt = tmp_path / "update-error.json"
    with pytest.raises(ValueError, match="update failed"):
        C._wait_capture_ready(base, app, context, EVENT_TYPES, receipt, {})
    body = json.loads(receipt.read_text(encoding="utf-8"))
    assert body["status"] == "failed"
    assert body["reason"] == "ValueError: update failed"
    assert body["updates"] == 0
    assert body["physics_before"] == body["physics_after"]


def test_existing_sidecar_fails_before_journal_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    validated, _, base, app, events = _install_capture(
        monkeypatch, tmp_path, statuses=[("", 0, 0, False)] * 4,
    )
    sidecar = tmp_path / "job.capture-readiness.json"
    sidecar.write_text("{}\n", encoding="utf-8")

    def runner(current: dict[str, Any], selected_app: App, *, camera_capture: Any) -> tuple[dict[str, Path], Path]:
        del current, selected_app
        camera_capture(base, object(), [], PROFILE, [1.0] * 3, [0.0] * 3, 0)
        raise AssertionError("unreachable")

    monkeypatch.setattr(C.V1, "run_real", runner)
    with pytest.raises(FileExistsError):
        C.run_capture(validated, app)
    assert "camera" not in events and "orchestrator-step" not in events


def test_nested_loading_waits_for_both_loaded_completions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, context, base, app, _ = _install_capture(
        monkeypatch, tmp_path, statuses=[("", 0, 0, False)] * 6,
        events_by_update={1: [1, 1], 2: [2], 3: [2]},
    )
    receipt = tmp_path / "nested.json"
    body = C._wait_capture_ready(base, app, context, EVENT_TYPES, receipt, {})
    assert body["updates"] == 5
    assert body["event_counts"] == {"loading": 2, "loaded": 2, "aborted": 0, "synthetic_loading": 0}
    assert body["loading_history"][1]["pending_asset_batches"] == 1


def test_aborted_load_is_terminal_but_never_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, context, base, app, _ = _install_capture(
        monkeypatch, tmp_path, statuses=[("", 0, 0, False)] * 6,
        events_by_update={1: [1, 1], 2: [2], 3: [3]},
    )
    receipt = tmp_path / "aborted.json"
    with pytest.raises(RuntimeError, match="asset load"):
        C._wait_capture_ready(base, app, context, EVENT_TYPES, receipt, {})
    body = json.loads(receipt.read_text(encoding="utf-8"))
    assert body["status"] == "failed"
    assert body["updates"] == 3
    assert body["event_counts"] == {"loading": 2, "loaded": 1, "aborted": 1, "synthetic_loading": 0}
    assert body["loading_history"][-1]["pending_asset_batches"] == 0


def test_orphan_loaded_event_records_synthetic_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, context, base, app, _ = _install_capture(
        monkeypatch, tmp_path, statuses=[("", 0, 0, False)] * 4,
        events_by_update={1: [2]},
    )
    body = C._wait_capture_ready(base, app, context, EVENT_TYPES, tmp_path / "orphan.json", {})
    assert body["event_counts"] == {"loading": 1, "loaded": 1, "aborted": 0, "synthetic_loading": 1}
    assert [row["event"] for row in body["event_history"]] == [
        "synthetic_assets_loading", "assets_loaded",
    ]


def test_elapsed_budget_is_checked_between_updates_and_seals_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, context, base, app, _ = _install_capture(
        monkeypatch, tmp_path, statuses=[("loading", 1, 2, True)] * 2,
        events_by_update={},
    )
    ticks = iter([0.0, 0.0, 31.0, 31.0])
    receipt = tmp_path / "elapsed.json"
    with pytest.raises(TimeoutError):
        C._wait_capture_ready(base, app, context, EVENT_TYPES, receipt, {}, clock=lambda: next(ticks))
    body = json.loads(receipt.read_text(encoding="utf-8"))
    assert body["updates"] == 1 and body["elapsed_s"] == 31.0
    assert body["bounds"]["polling_elapsed_budget_scope"].endswith("not_a_hard_watchdog")


@pytest.mark.parametrize("initial", [True, False])
def test_asset_update_restores_initial_setting_even_on_app_error(initial: bool) -> None:
    base = Base([])
    base.sim.play_simulations = initial
    class AppFailure:
        def update(self) -> None:
            assert base.sim.play_simulations is False
            raise ValueError("asset update failed")
    with pytest.raises(ValueError, match="asset update failed"):
        C._asset_update(base, AppFailure())
    assert base.sim.play_simulations is initial


def test_asset_update_rejects_failed_restore() -> None:
    base = Base([])
    def set_setting(name: str, value: bool) -> None:
        assert name == C.PLAYER_SETTING
        if value is False:
            base.sim.play_simulations = False
    base.sim.set_setting = set_setting
    with pytest.raises(RuntimeError, match="복원"):
        C._asset_update(base, SimpleNamespace(update=lambda: None))
