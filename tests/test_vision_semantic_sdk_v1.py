from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
S = importlib.import_module("vision_semantic_sdk_v1")
R = importlib.import_module("record_vision_evidence_v1")
P = importlib.import_module("run_vision_semantic_diagnostic_v1")


@pytest.fixture(autouse=True)
def _fixed_test_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(R, "WIDTH", 4)
    monkeypatch.setattr(R, "HEIGHT", 3)


def _payload(*, empty: bool = False) -> dict[str, Any]:
    rp = "RenderProduct_0"
    semantic = (
        np.array([], dtype=np.float64) if empty else np.ones((3, 4), dtype=np.uint32)
    )
    times = {
        "durationDenominator": 60,
        "durationNumerator": 1,
        "externalTimeOfSimNs": 2_000_000_000,
        "frameNumber": 55,
        "rationalTimeOfSimDenominator": 30,
        "rationalTimeOfSimNumerator": 55,
        "sampleTimeOffsetInSimFrames": 0,
        "simulationTime": 2.0,
        "simulationTimeMonotonic": 2.0,
        "swhFrameNumber": 0,
        "systemTime": 1000.0,
    }
    return {
        "swhFrameNumber": 77,
        "reference_time": (55, 30),
        "distribution_outputs": {},
        "trigger_outputs": {},
        "named_outputs": {},
        "annotators": {
            "camera": {rp: R.fixed_profile()["camera"]["camera_prim_path"]},
            "resolution": {rp: np.array([4, 3], dtype=np.int32)},
            "rgb": {rp: {"data": np.zeros((3, 4, 4), dtype=np.uint8)}},
            "semantic_segmentation": {
                rp: {"data": semantic, "idToLabels": {"1": {"class": "hidden"}}}
            },
            "CameraParams": {
                rp: {
                    "cameraViewTransform": np.arange(16, dtype=np.float64),
                    "cameraProjection": np.arange(16, dtype=np.float32) + 1,
                    "renderProductResolution": np.array([4, 3], dtype=np.int32),
                }
            },
            "IsaacReadTimes": {rp: times},
        },
    }


def _validated(tmp_path: Path, *, mode: str = S.MODE) -> dict[str, Any]:
    return {
        "job_path": tmp_path / "semantic-job.json",
        "output_directory": tmp_path / "output",
        "job_sha256": "a" * 64,
        "job": {"mode": mode, "source": {"files_lf": {}}},
    }


class _CommandBuffer:
    def __init__(self) -> None:
        self.data = np.zeros((1, 3), dtype=np.float32)

    def __getitem__(self, key: Any) -> Any:
        return self.data[key]

    def __setitem__(self, key: Any, value: Any) -> None:
        self.data[key] = value

    def new_tensor(self, value: Any) -> np.ndarray[Any, Any]:
        return np.asarray(value, dtype=np.float32)


class _Attr:
    def __init__(self, log: list[str], name: str) -> None:
        self.log = log
        self.name = name

    def Set(self, _value: Any) -> None:
        self.log.append(f"camera_{self.name}")


class _Camera:
    def __init__(self, log: list[str]) -> None:
        self.log = log

    def GetProjectionAttr(self) -> _Attr:
        return _Attr(self.log, "projection")

    def GetHorizontalApertureAttr(self) -> _Attr:
        return _Attr(self.log, "aperture")

    def GetFocalLengthAttr(self) -> _Attr:
        return _Attr(self.log, "focal")

    def GetClippingRangeAttr(self) -> _Attr:
        return _Attr(self.log, "clipping")


class _CommandManager:
    def __init__(self) -> None:
        self.command = _CommandBuffer()

    def get_command(self, name: str) -> _CommandBuffer:
        assert name == "base_velocity"
        return self.command


class _Sim:
    def __init__(self, log: list[str]) -> None:
        self.log = log

    def set_camera_view(self, **_kwargs: Any) -> None:
        self.log.append("camera_view")


class _Base:
    def __init__(self, log: list[str]) -> None:
        self.log = log
        self.state = 0
        self.sim = _Sim(log)
        self.command_manager = _CommandManager()

    def reset(self, *, seed: int) -> None:
        assert seed == 42
        self.log.append("reset")


class _Env:
    def __init__(
        self, log: list[str], *, close_error: BaseException | None = None
    ) -> None:
        self.log = log
        self.unwrapped = _Base(log)
        self.close_error = close_error

    def close(self) -> None:
        self.log.append("env_close")
        if self.close_error is not None:
            raise self.close_error


class _Stage:
    def GetPrimAtPath(self, _path: str) -> object:
        return object()


class _Context:
    def __init__(self, log: list[str]) -> None:
        self.log = log
        self.stage = _Stage()

    def get_stage(self) -> _Stage:
        return self.stage

    def reset_renderer_accumulation(self) -> None:
        self.log.append("renderer_reset")


class _RenderProduct:
    def __init__(
        self, log: list[str], *, destroy_error: BaseException | None = None
    ) -> None:
        self.log = log
        self.destroy_error = destroy_error

    def destroy(self) -> None:
        self.log.append("render_product_destroy")
        if self.destroy_error is not None:
            raise self.destroy_error


class _Writer:
    def __init__(
        self,
        rep: Any,
        observer: Any,
        log: list[str],
        *,
        attach_payload: dict[str, Any] | None,
        detach_payload: dict[str, Any] | None,
        detach_error: BaseException | None,
    ) -> None:
        self.rep = rep
        self.observer = observer
        self.log = log
        self.attach_payload = attach_payload
        self.detach_payload = detach_payload
        self.detach_error = detach_error

    def attach(self, _render_product: Any, *, trigger: Any) -> None:
        assert trigger == ("custom", S.EVENT_NAME)
        self.log.append("writer_attach")
        self.rep.writer = self
        if self.attach_payload is not None:
            self.log.append("attach_callback")
            self.observer.observe(self.attach_payload)

    def detach(self) -> None:
        self.log.append("writer_detach")
        if self.detach_payload is not None:
            self.log.append("detach_callback")
            self.observer.observe(self.detach_payload)
        if self.detach_error is not None:
            raise self.detach_error


class _Marker:
    def __init__(self, log: list[str]) -> None:
        self.log = log

    def bind_and_validate(self, _stage: Any) -> None:
        self.log.append("marker_bind")

    def set_code(self, _stage: Any, code: int) -> None:
        assert code == 0
        self.log.append("marker_code_0")


class _App:
    def __init__(self, log: list[str] | None = None) -> None:
        self.log = log

    def update(self) -> None:
        if self.log is not None:
            self.log.append("app_update")

    def close(self) -> None:
        raise AssertionError("run_one_view must not close app")


def _env_cfg() -> Any:
    ranges = SimpleNamespace(lin_vel_x=None, lin_vel_y=None, ang_vel_z=None)
    command = SimpleNamespace(
        heading_command=None,
        rel_standing_envs=None,
        resampling_time_range=None,
        ranges=ranges,
    )
    return SimpleNamespace(
        seed=None,
        actions=SimpleNamespace(joint_pos=SimpleNamespace(scale=None)),
        commands=SimpleNamespace(base_velocity=command),
    )


def _dependencies(
    log: list[str],
    *,
    payloads: list[dict[str, Any]] | None = None,
    startup_error: BaseException | None = None,
    source_error: BaseException | None = None,
    attach_payload: dict[str, Any] | None = None,
    detach_payload: dict[str, Any] | None = None,
    detach_error: BaseException | None = None,
    destroy_error: BaseException | None = None,
    close_error: BaseException | None = None,
    setup_physics_change: bool = False,
    physics_change: bool = False,
    callbacks_per_event: int = 1,
    registrations: dict[str, bool] | None = None,
) -> dict[str, Any]:
    selected_payloads = payloads or [_payload(empty=True), _payload()]
    context = _Context(log)
    env = _Env(log, close_error=close_error)
    render_product = _RenderProduct(log, destroy_error=destroy_error)
    event_index = 0

    class Utils:
        @staticmethod
        def send_og_event(name: str) -> None:
            nonlocal event_index
            assert name == S.EVENT_NAME
            log.append(f"event_{event_index}")
            for _ in range(callbacks_per_event):
                rep.writer.observer.observe(selected_payloads[event_index])
            event_index += 1

    rep = SimpleNamespace(
        writer=None,
        utils=Utils,
        orchestrator=object(),
        trigger=SimpleNamespace(on_custom_event=lambda name: ("custom", name)),
        create=SimpleNamespace(
            render_product=lambda _path, _resolution: (
                log.append("render_product_create") or render_product
            )
        ),
    )

    def startup(_rep: Any, _settings: Any) -> dict[str, Any]:
        log.append("startup")
        if startup_error is not None:
            raise startup_error
        return {"before": {"enabled": False}, "after": {"enabled": True}}

    def capture_render_step(_orchestrator: Any, _profile: Any) -> None:
        log.append(f"render_{event_index - 1}")
        if physics_change:
            env.unwrapped.state += 1
        if source_error is not None:
            raise source_error

    def writer(_rep: Any, observer: Any) -> _Writer:
        return _Writer(
            rep,
            observer,
            log,
            attach_payload=attach_payload,
            detach_payload=detach_payload,
            detach_error=detach_error,
        )

    profile = R.fixed_profile()
    return {
        "profile": lambda: profile,
        "startup": startup,
        "parse_env_cfg": lambda _task, **_kwargs: _env_cfg(),
        "make_env": lambda _task, _cfg: env,
        "rep": rep,
        "usd_context": lambda: (context, object()),
        "settings": lambda: object(),
        "enable_extension": lambda name: log.append(f"enable_{name}") or True,
        "registration_probe": lambda: (
            log.append("registrations")
            or registrations
            or {name: True for name in S.ANNOTATORS}
        ),
        "camera": lambda _prim: _Camera(log),
        "readiness": lambda *_args: (
            log.append("readiness")
            or (
                setattr(env.unwrapped, "state", env.unwrapped.state + 1)
                if setup_physics_change
                else None
            )
            or {"status": "ready"}
        ),
        "renderer_readback": lambda *_args: {"renderer": "fake"},
        "fxaa_required": lambda _job: False,
        "fxaa_apply": lambda *_args: None,
        "fxaa_readback": lambda _settings: {},
        "fxaa_verify": lambda _value: None,
        "marker": lambda _rep, _root: log.append("marker_create") or _Marker(log),
        "root_position": lambda _base: [0.0, 0.0, 0.5],
        "writer": writer,
        "physics_snapshot": lambda base: {"state": base.state},
        "capture_render_step": capture_render_step,
    }


def test_success_seals_after_detach_before_cleanup_and_keeps_late_callback(
    tmp_path: Path,
) -> None:
    log: list[str] = []
    sealed: list[dict[str, Any]] = []

    def seal(report: Any) -> None:
        log.append("seal_report")
        sealed.append(report)

    result = S.run_one_view(
        _validated(tmp_path),
        _App(log),
        seal_report=seal,
        dependencies=_dependencies(
            log,
            attach_payload=_payload(),
            detach_payload=_payload(empty=True),
        ),
    )

    assert log.index("camera_view") < log.index("renderer_reset")
    assert log.index(f"enable_{S.EXTENSION}") < log.index("app_update")
    assert log.index("app_update") < log.index("startup")
    assert log.index("startup") < log.index("registrations")
    assert log.index("marker_create") < log.index("readiness")
    assert log.index("readiness") < log.index("marker_bind")
    assert log.index("renderer_reset") < log.index("writer_attach")
    assert log.index("writer_attach") < log.index("event_0") < log.index("event_1")
    assert log.index("event_1") < log.index("writer_detach")
    assert log.index("detach_callback") < log.index("seal_report")
    assert log.index("seal_report") < log.index("render_product_destroy")
    assert log.index("render_product_destroy") < log.index("env_close")
    assert len([item for item in log if item.startswith("event_")]) == 2
    assert sealed[0]["diagnostic"]["callbacks_observed"] == 4
    assert sealed[0]["setup_evidence"] == result["setup_evidence"]
    setup = result["setup_evidence"]
    assert setup["extension"] == {
        "name": S.EXTENSION,
        "enable_result": True,
        "app_update_completed": True,
    }
    assert setup["startup"] == {
        "before": {"enabled": False},
        "after": {"enabled": True},
    }
    assert setup["registrations"] == {name: True for name in S.ANNOTATORS}
    assert setup["callback_count_before_detach"] == 3
    assert setup["callback_count_after_detach"] == 4
    assert result["diagnostic"]["callbacks"][0]["window"] == "attaching"
    assert result["diagnostic"]["callbacks"][-1]["window"] == "detaching"
    assert setup["phase_order"].index("marker_create") < setup["phase_order"].index(
        "capture_readiness"
    )
    assert result["diagnostic"]["collection_scope"]["starts"] == (
        "before_writer_attach"
    )
    assert result["diagnostic"]["collection_scope"]["ends"] == (
        "after_writer_detach_return_before_render_product_cleanup"
    )
    assert sealed[0]["lifecycle"] == {
        "setup_completed": True,
        "writer_attach_attempted": True,
        "writer_attached": True,
        "writer_detach_attempted": True,
        "writer_detached": True,
        "report_sealed": False,
        "render_product_destroyed": False,
        "environment_closed": False,
        "errors": [],
    }
    assert result["diagnostic"]["callbacks_observed"] == 4
    assert result["diagnostic"]["events_attempted"] == 2
    assert result["diagnostic"]["events_completed"] == 2
    assert result["diagnostic"]["callbacks_preserved"] <= 16
    assert result["diagnostic"]["observations"]["physics_changed"] is False
    assert result["diagnostic"]["observations"]["source_error"] is False
    assert result["diagnostic"]["observations"]["overflow"] is False
    assert result["lifecycle"]["report_sealed"] is True
    assert result["lifecycle"]["render_product_destroyed"] is True
    assert result["lifecycle"]["environment_closed"] is True
    assert result["lifecycle"]["errors"] == []
    assert result["no_capture_approved"] is True
    assert set(result) == {
        "schema",
        "diagnostic",
        "lifecycle",
        "renderer",
        "setup_evidence",
        "no_capture_approved",
    }
    assert P._successful_sdk_envelope(result) is True


def test_source_error_preserves_primary_and_still_seals_and_cleans(
    tmp_path: Path,
) -> None:
    log: list[str] = []
    sealed: list[dict[str, Any]] = []
    primary = RuntimeError("source failure")

    with pytest.raises(S.SDKDiagnosticFailure) as caught:
        S.run_one_view(
            _validated(tmp_path),
            _App(),
            seal_report=lambda report: sealed.append(dict(report)),
            dependencies=_dependencies(log, source_error=primary),
        )

    assert caught.value.primary is primary
    assert caught.value.__cause__ is primary
    assert caught.value.report["diagnostic"]["callbacks_preserved"] == 1
    assert log[-2:] == ["render_product_destroy", "env_close"]
    assert len(sealed) == 1


def test_cleanup_errors_are_distinguished_and_all_cleanup_is_attempted(
    tmp_path: Path,
) -> None:
    log: list[str] = []
    sealed: list[dict[str, Any]] = []
    destroy_error = RuntimeError("destroy failure")
    close_error = ValueError("close failure")

    with pytest.raises(S.SDKDiagnosticFailure) as caught:
        S.run_one_view(
            _validated(tmp_path),
            _App(),
            seal_report=lambda report: sealed.append(dict(report)),
            dependencies=_dependencies(
                log, destroy_error=destroy_error, close_error=close_error
            ),
        )

    assert caught.value.primary is destroy_error
    lifecycle = caught.value.report["lifecycle"]
    assert lifecycle["render_product_destroyed"] is False
    assert lifecycle["environment_closed"] is False
    assert lifecycle["errors"][-2:] == [
        {
            "phase": "render_product_destroy",
            "error_type": "builtins.RuntimeError",
        },
        {"phase": "environment_close", "error_type": "builtins.ValueError"},
    ]
    assert sealed[0]["lifecycle"]["render_product_destroyed"] is False
    assert sealed[0]["lifecycle"]["environment_closed"] is False
    assert log[-2:] == ["render_product_destroy", "env_close"]


def test_setup_failure_seals_empty_partial_report(tmp_path: Path) -> None:
    log: list[str] = []
    sealed: list[dict[str, Any]] = []
    primary = RuntimeError("startup failure")

    with pytest.raises(S.SDKDiagnosticFailure) as caught:
        S.run_one_view(
            _validated(tmp_path),
            _App(),
            seal_report=lambda report: sealed.append(dict(report)),
            dependencies=_dependencies(log, startup_error=primary),
        )

    assert caught.value.primary is primary
    assert sealed[0]["diagnostic"]["callbacks_observed"] == 0
    lifecycle = caught.value.report["lifecycle"]
    assert lifecycle["setup_completed"] is False
    assert lifecycle["writer_attach_attempted"] is False
    assert lifecycle["writer_detach_attempted"] is False
    assert lifecycle["report_sealed"] is True
    assert lifecycle["render_product_destroyed"] is False
    assert lifecycle["environment_closed"] is False
    assert log == [f"enable_{S.EXTENSION}", "startup"]


def test_missing_required_annotator_stops_before_environment_source(
    tmp_path: Path,
) -> None:
    log: list[str] = []
    sealed: list[dict[str, Any]] = []
    registered = {name: True for name in S.ANNOTATORS}
    registered["semantic_segmentation"] = False

    with pytest.raises(S.SDKDiagnosticFailure) as caught:
        S.run_one_view(
            _validated(tmp_path),
            _App(log),
            seal_report=lambda report: sealed.append(dict(report)),
            dependencies=_dependencies(log, registrations=registered),
        )

    assert isinstance(caught.value.primary, RuntimeError)
    assert "reset" not in log and "writer_attach" not in log
    assert sealed[0]["setup_evidence"]["registrations"] == registered
    assert caught.value.report["lifecycle"]["errors"][0] == {
        "phase": "annotator_registrations",
        "error_type": "builtins.RuntimeError",
    }


def test_physics_change_is_hold_failure_after_two_bounded_events(
    tmp_path: Path,
) -> None:
    log: list[str] = []
    sealed: list[dict[str, Any]] = []

    with pytest.raises(S.SDKDiagnosticFailure) as caught:
        S.run_one_view(
            _validated(tmp_path),
            _App(),
            seal_report=lambda report: sealed.append(dict(report)),
            dependencies=_dependencies(log, physics_change=True),
        )

    assert isinstance(caught.value.primary, RuntimeError)
    assert caught.value.report["diagnostic"]["observations"]["physics_changed"] is True
    assert len([item for item in log if item.startswith("event_")]) == 2
    assert len(sealed) == 1


def test_setup_physics_change_preserves_before_and_after_snapshots(
    tmp_path: Path,
) -> None:
    log: list[str] = []
    sealed: list[dict[str, Any]] = []

    with pytest.raises(S.SDKDiagnosticFailure) as caught:
        S.run_one_view(
            _validated(tmp_path),
            _App(),
            seal_report=lambda report: sealed.append(dict(report)),
            dependencies=_dependencies(log, setup_physics_change=True),
        )

    evidence = caught.value.report["setup_evidence"]["setup_physics_before_after"]
    assert evidence == {
        "before": {"state": 0},
        "after": {"state": 1},
        "equal": False,
    }
    assert caught.value.report["lifecycle"]["writer_attach_attempted"] is False
    assert len(sealed) == 1


def test_detach_failure_seals_partial_callbacks_and_cleanup_continues(
    tmp_path: Path,
) -> None:
    log: list[str] = []
    sealed: list[dict[str, Any]] = []
    primary = RuntimeError("detach failure")

    with pytest.raises(S.SDKDiagnosticFailure) as caught:
        S.run_one_view(
            _validated(tmp_path),
            _App(),
            seal_report=lambda report: sealed.append(dict(report)),
            dependencies=_dependencies(
                log,
                detach_payload=_payload(empty=True),
                detach_error=primary,
            ),
        )

    assert caught.value.primary is primary
    assert sealed[0]["diagnostic"]["callbacks_observed"] == 3
    assert sealed[0]["lifecycle"]["writer_detach_attempted"] is True
    assert sealed[0]["lifecycle"]["writer_detached"] is False
    assert log[-2:] == ["render_product_destroy", "env_close"]


def test_callback_overflow_is_hold_failure_and_stops_second_event(
    tmp_path: Path,
) -> None:
    log: list[str] = []
    sealed: list[dict[str, Any]] = []

    with pytest.raises(S.SDKDiagnosticFailure) as caught:
        S.run_one_view(
            _validated(tmp_path),
            _App(),
            seal_report=lambda report: sealed.append(dict(report)),
            dependencies=_dependencies(log, callbacks_per_event=17),
        )

    diagnostic = caught.value.report["diagnostic"]
    assert diagnostic["callbacks_observed"] == 17
    assert diagnostic["callbacks_preserved"] == 16
    assert diagnostic["observations"]["overflow"] is True
    assert len([item for item in log if item.startswith("event_")]) == 1
    assert len(sealed) == 1


def test_seal_failure_is_primary_and_cleanup_still_runs(tmp_path: Path) -> None:
    log: list[str] = []
    primary = RuntimeError("seal failure")

    def seal(_report: Any) -> None:
        log.append("seal_report")
        raise primary

    with pytest.raises(S.SDKDiagnosticFailure) as caught:
        S.run_one_view(
            _validated(tmp_path),
            _App(),
            seal_report=seal,
            dependencies=_dependencies(log),
        )

    assert caught.value.primary is primary
    assert log[-3:] == ["seal_report", "render_product_destroy", "env_close"]


def test_mode_mismatch_rejects_before_dependencies_or_seal(tmp_path: Path) -> None:
    calls: list[str] = []

    class ExplodingDependencies(dict[str, Any]):
        def __iter__(self):  # type: ignore[no-untyped-def]
            calls.append("dependencies")
            raise AssertionError("dependencies must not be read")

    with pytest.raises(ValueError, match="mode"):
        S.run_one_view(
            _validated(tmp_path, mode="record"),
            _App(),
            seal_report=lambda _report: calls.append("seal"),
            dependencies=ExplodingDependencies(),
        )
    assert calls == []


def test_injected_dependencies_avoid_sdk_loader(
    tmp_path: Path, monkeypatch: Any
) -> None:
    log: list[str] = []
    monkeypatch.setattr(
        S,
        "_load_dependencies",
        lambda: (_ for _ in ()).throw(AssertionError("SDK import path used")),
    )

    result = S.run_one_view(
        _validated(tmp_path),
        _App(),
        seal_report=lambda _report: None,
        dependencies=_dependencies(log),
    )

    assert result["schema"] == S.SDK_REPORT_SCHEMA
    assert len([item for item in log if item.startswith("event_")]) == 2


def test_keyboard_interrupt_is_rethrown_after_seal_and_cleanup(tmp_path: Path) -> None:
    log: list[str] = []
    sealed: list[dict[str, Any]] = []

    with pytest.raises(KeyboardInterrupt) as caught:
        S.run_one_view(
            _validated(tmp_path),
            _App(),
            seal_report=lambda report: sealed.append(dict(report)),
            dependencies=_dependencies(log, source_error=KeyboardInterrupt()),
        )

    assert hasattr(caught.value, "semantic_sdk_envelope")
    assert len(sealed) == 1
    assert log[-2:] == ["render_product_destroy", "env_close"]
