from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import record_vision_evidence_v1 as RECORD  # pyright: ignore[reportMissingImports]
import test_vision_annotator_pull_collector_v1 as FIXTURE  # pyright: ignore[reportMissingImports]
import vision_annotator_pull_worker_v2 as WORKER  # pyright: ignore[reportMissingImports]


class _Command:
    def new_tensor(self, value: list[float]) -> list[float]:
        return value

    def __setitem__(self, _key: Any, _value: Any) -> None:
        pass


class _Sim:
    current_time = 0.01

    def __init__(self, state: dict[str, Any]) -> None:
        self._state = state

    def set_camera_view(self, *, eye: list[float], **_kwargs: Any) -> None:
        self._state["camera_kind"] = "A" if eye[0] >= 0.0 else "B"


class _Base:
    def __init__(self, state: dict[str, Any]) -> None:
        self.sim = _Sim(state)
        self.command_manager = SimpleNamespace(
            get_command=lambda _name: _Command()
        )

    def reset(self, *, seed: int) -> None:
        assert seed >= 0


class _Env:
    def __init__(self, state: dict[str, Any]) -> None:
        self.unwrapped = _Base(state)
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _SimulationApp:
    """실제 run_worker app 인자처럼 update만 제공한다."""

    def __init__(self, state: dict[str, Any]) -> None:
        self._state = state

    def update(self) -> None:
        self._state["updates"] += 1


class _NativeApp:
    def __init__(self, state: dict[str, Any]) -> None:
        self._state = state

    def get_update_number(self) -> int:
        return int(self._state["updates"])


class _Context:
    def get_stage(self) -> Any:
        return SimpleNamespace(GetPrimAtPath=lambda path: path)

    def reset_renderer_accumulation(self) -> None:
        pass


class _Attr:
    def Set(self, _value: Any) -> None:
        pass


class _Camera:
    def GetProjectionAttr(self) -> _Attr:
        return _Attr()

    def GetHorizontalApertureAttr(self) -> _Attr:
        return _Attr()

    def GetFocalLengthAttr(self) -> _Attr:
        return _Attr()

    def GetClippingRangeAttr(self) -> _Attr:
        return _Attr()


class _RenderProduct:
    path = "/Render/Products/RP"

    def __str__(self) -> str:
        return self.path

    def destroy(self) -> None:
        pass


class _Marker:
    def __init__(self, state: dict[str, Any]) -> None:
        self._state = state

    def bind_and_validate(self, _stage: Any) -> None:
        pass

    def set_code(self, _stage: Any, code: int) -> None:
        self._state.update(code=code, ordinal=code)


def _policy_factory(
    _validated: Any,
    make_env: Any,
    state: dict[str, bool],
    probe: dict[str, Any],
) -> Any:
    def create(task: str, cfg: Any) -> Any:
        env = make_env(task, cfg)
        original_close = env.close

        def close() -> None:
            assert state == {"capture_completed": True, "annotators_detached": True}
            probe.update(
                {
                    "environment_steps": 1,
                    "annotators_detached_before_policy_step": True,
                }
            )
            original_close()

        env.close = close
        return env

    return create


def _case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, Any], _SimulationApp, _NativeApp, dict[str, Any]]:
    profile = RECORD.fixed_profile()
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    job_path = tmp_path / "job.json"
    job_path.write_text("{}", encoding="utf-8")
    validated = {
        "job": {"mode": WORKER.MODE, "profile": profile},
        "job_path": str(job_path),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": "a" * 64,
    }
    state: dict[str, Any] = {
        "ordinal": -1,
        "code": 0,
        "camera_kind": "A",
        "frame": 57,
        "simulation_time": 0.01,
        "events": [],
        "payloads": {},
        "updates": 0,
    }
    registry = FIXTURE.FakeRegistry(state)
    orchestrator = FIXTURE.FakeOrchestrator(state)
    rep = SimpleNamespace(
        AnnotatorRegistry=registry,
        orchestrator=orchestrator,
        create=SimpleNamespace(render_product=lambda *_args: _RenderProduct()),
    )
    deps = {name: object() for name in WORKER.PREPARED._DEPENDENCY_KEYS} | {
        "profile": lambda: profile,
        "settings": lambda: object(),
        "enable_extension": lambda _name: True,
        "startup": lambda _rep, _settings: {"before": {}, "after": {}},
        "registration_probe": lambda: {
            name: True for name in WORKER.SDK.ANNOTATORS
        },
        "usd_context": lambda: (_Context(), object()),
        "make_env": lambda _task, _cfg: _Env(state),
        "rep": rep,
        "camera": lambda _prim: _Camera(),
        "root_position": lambda _base: [0.0, 0.0, 0.0],
        "marker": lambda _rep, _root: _Marker(state),
        "fxaa_required": lambda _job: False,
        "renderer_readback": lambda *_args: {"renderer": "RayTracedLighting"},
        "readiness": lambda *_args: None,
        "physics_snapshot": lambda _base: {
            "root_position_world_m": [0.0, 0.0, 0.0]
        },
    }
    monkeypatch.setattr(WORKER.SDK, "_configure_env", lambda _profile, _deps: object())
    monkeypatch.setattr(WORKER.R, "WIDTH", 6)
    monkeypatch.setattr(WORKER.R, "HEIGHT", 6)
    return validated | {"dependencies": deps}, _SimulationApp(state), _NativeApp(state), state


def _run(
    validated: dict[str, Any],
    simulation_app: _SimulationApp,
    native_app: Any,
) -> dict[str, Any]:
    deps = validated.pop("dependencies")
    return WORKER.run_worker(
        validated,
        simulation_app,
        dependencies=deps,
        policy_env_factory=_policy_factory,
        native_app_factory=lambda: native_app,
    )


def test_run_worker_counts_native_updates_from_internal_render_steps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    validated, simulation_app, native_app, _ = _case(tmp_path, monkeypatch)

    report = _run(validated, simulation_app, native_app)

    assert report["diagnostic_complete"] is True
    assert report["schema"] == WORKER.SCHEMA
    assert report["setup"]["app_update_counter"] == {
        "source": "omni.kit.app.IApp.get_update_number",
        "method": "app.get_update_number()",
        "initial_value": 0,
        "after_extension_update": 1,
        "fail_closed": True,
    }
    assert [
        row["render_window_app_updates"] for row in report["collection"]["captures"]
    ] == [2, 2, 2, 2]
    assert [
        row["pull_window_app_updates"] for row in report["collection"]["captures"]
    ] == [0, 0, 0, 0]


def test_run_worker_rejects_native_update_during_sequential_pull(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    validated, simulation_app, native_app, state = _case(tmp_path, monkeypatch)
    state["update_during_pull"] = (2, "CameraParams")

    with pytest.raises(WORKER.AnnotatorPullWorkerFailure) as caught:
        _run(validated, simulation_app, native_app)

    assert isinstance(caught.value.primary, ValueError)
    assert "연속 annotator pull" in str(caught.value.primary)
    assert caught.value.report["collection"]["captures"][2][
        "pull_window_app_updates"
    ] == 1


@pytest.mark.parametrize(
    ("native_app", "expected_type"),
    [
        (SimpleNamespace(), TypeError),
        (SimpleNamespace(get_update_number=lambda: True), RuntimeError),
    ],
)
def test_run_worker_fails_closed_without_valid_native_counter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    native_app: Any,
    expected_type: type[Exception],
) -> None:
    validated, simulation_app, _, _ = _case(tmp_path, monkeypatch)

    with pytest.raises(expected_type, match="get_update_number"):
        _run(validated, simulation_app, native_app)
