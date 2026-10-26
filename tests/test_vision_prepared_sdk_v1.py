from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
S = importlib.import_module("vision_prepared_sdk_v1")
F = importlib.import_module("vision_frame_sync_v1")

WIDTH = 10
HEIGHT = 10
RP = "RenderProduct_0"


@pytest.fixture(autouse=True)
def _small_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(S.R, "WIDTH", WIDTH)
    monkeypatch.setattr(S.R, "HEIGHT", HEIGHT)


def _times(frame: int) -> dict[str, Any]:
    return {
        "durationDenominator": 60,
        "durationNumerator": 1,
        "externalTimeOfSimNs": 2_000_000_000,
        "frameNumber": frame,
        "rationalTimeOfSimDenominator": 30,
        "rationalTimeOfSimNumerator": frame,
        "sampleTimeOffsetInSimFrames": 0,
        "simulationTime": 2.0,
        "simulationTimeMonotonic": 2.0,
        "swhFrameNumber": 0,
        "systemTime": 1000.0 + frame,
    }


def _payload(
    frame: int, marker: int, *, empty: bool = False, view: str = "A"
) -> dict[str, Any]:
    camera_path = S.R.fixed_profile()["camera"]["camera_prim_path"]
    rgba = np.zeros((HEIGHT, WIDTH, 4), dtype=np.uint8)
    rgba[:, :, 3] = 255
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint32)
    labels: dict[str, Any] = {"0": {"class": "robot/body"}}
    for index, label in enumerate(sorted(F.marker_labels(marker)), 1):
        mask[index, 1:9] = index
        labels[str(index)] = {"class": label}
        role = "sentinel" if label == F.SENTINEL else label.rsplit("/", 1)[-1]
        rgba[index, 1:9, :3] = F.PALETTE[role]
    return {
        "swhFrameNumber": 0,
        "reference_time": (frame, 30),
        "distribution_outputs": {},
        "trigger_outputs": {},
        "named_outputs": {},
        "annotators": {
            "camera": {RP: camera_path},
            "resolution": {RP: np.array([WIDTH, HEIGHT], dtype=np.int32)},
            "rgb": {RP: {"data": rgba}},
            "semantic_segmentation": {
                RP: {
                    "data": np.array([], dtype=np.float64) if empty else mask,
                    "idToLabels": labels,
                }
            },
            "CameraParams": {
                RP: {
                    "cameraViewTransform": np.arange(16, dtype=np.float64)
                    + (100 if view == "B" else 0),
                    "cameraProjection": np.eye(4).reshape(16),
                    "renderProductResolution": np.array(
                        [WIDTH, HEIGHT], dtype=np.int32
                    ),
                }
            },
            "IsaacReadTimes": {RP: _times(frame)},
        },
    }


def _validated(tmp_path: Path, mode: str = S.MODE) -> dict[str, Any]:
    job = tmp_path / "job.json"
    job.write_text("{}", encoding="utf-8")
    output = tmp_path / "output"
    output.mkdir()
    return {
        "job_path": job,
        "output_directory": output,
        "job_sha256": "a" * 64,
        "job": {"mode": mode},
    }


class _Command:
    def __getitem__(self, _key: Any) -> _Command:
        return self

    def __setitem__(self, _key: Any, _value: Any) -> None:
        return None

    def new_tensor(self, value: Any) -> Any:
        return value


class _Sim:
    def __init__(self, log: list[str]) -> None:
        self.log = log
        self.current_time = 2.0

    def set_camera_view(self, **_kwargs: Any) -> None:
        self.log.append("camera_view")


class _Base:
    def __init__(self, log: list[str]) -> None:
        self.log = log
        self.state = 0
        self.sim = _Sim(log)
        self.command_manager = SimpleNamespace(get_command=lambda _name: _Command())

    def reset(self, *, seed: int) -> None:
        self.log.append(f"reset:{seed}")


class _Env:
    def __init__(
        self, log: list[str], close_error: BaseException | None = None
    ) -> None:
        self.log = log
        self.unwrapped = _Base(log)
        self.close_error = close_error

    def close(self) -> None:
        self.log.append("env_close")
        if self.close_error is not None:
            raise self.close_error


class _Context:
    def __init__(self, log: list[str]) -> None:
        self.log = log
        self.stage = SimpleNamespace(GetPrimAtPath=lambda _path: object())

    def get_stage(self) -> Any:
        return self.stage

    def reset_renderer_accumulation(self) -> None:
        self.log.append("renderer_reset")


class _Attr:
    def Set(self, _value: Any) -> None:
        return None


class _Camera:
    GetProjectionAttr = lambda self: _Attr()
    GetHorizontalApertureAttr = lambda self: _Attr()
    GetFocalLengthAttr = lambda self: _Attr()
    GetClippingRangeAttr = lambda self: _Attr()


class _Marker:
    def __init__(self, log: list[str]) -> None:
        self.log = log

    def bind_and_validate(self, _stage: Any) -> None:
        self.log.append("marker_bind")

    def set_code(self, _stage: Any, code: int) -> None:
        self.log.append(f"marker:{code}")


class _RenderProduct:
    path = "/Render/RenderProduct_0"

    def __init__(self, log: list[str], error: BaseException | None = None) -> None:
        self.log = log
        self.error = error

    def destroy(self) -> None:
        self.log.append("render_product_destroy")
        if self.error is not None:
            raise self.error


class _Writer:
    def __init__(
        self,
        rep: Any,
        mux: Any,
        log: list[str],
        *,
        attach_error: BaseException | None,
        detach_error: BaseException | None,
        detach_payload: dict[str, Any] | None,
    ) -> None:
        self.rep = rep
        self.mux = mux
        self.log = log
        self.attach_error = attach_error
        self.detach_error = detach_error
        self.detach_payload = detach_payload

    def attach(self, _rp: Any, *, trigger: Any) -> None:
        self.log.append("writer_attach")
        if self.attach_error is not None:
            raise self.attach_error
        assert trigger == ("custom", S.EVENT_NAME)
        self.rep.writer = self

    def detach(self) -> None:
        self.log.append("writer_detach")
        if self.detach_payload is not None:
            self.mux.observe(self.detach_payload)
        if self.detach_error is not None:
            raise self.detach_error

    def raise_callback_error(self) -> None:
        return None


class _App:
    def __init__(
        self, log: list[str], update_error: BaseException | None = None
    ) -> None:
        self.log = log
        self.update_error = update_error

    def update(self) -> None:
        self.log.append("app_update")
        if self.update_error is not None:
            raise self.update_error

    def close(self) -> None:
        raise AssertionError("run_prepared_smoke must not close app")


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
    prepare_empty: bool = True,
    missing_arm: int | None = None,
    extra_arm: int | None = None,
    attach_error: BaseException | None = None,
    detach_error: BaseException | None = None,
    detach_payload: dict[str, Any] | None = None,
    destroy_error: BaseException | None = None,
    close_error: BaseException | None = None,
    render_error: BaseException | None = None,
    physics_change: bool = False,
) -> dict[str, Any]:
    env = _Env(log, close_error)
    context = _Context(log)
    render_product = _RenderProduct(log, destroy_error)
    event = 0

    class Utils:
        @staticmethod
        def send_og_event(name: str) -> None:
            nonlocal event
            assert name == S.EVENT_NAME
            log.append(f"event:{event}")
            payloads = (
                (
                    [_payload(60, 3, empty=True), _payload(60, 3)]
                    if prepare_empty
                    else [_payload(60, 3)]
                )
                if event == 0
                else [_payload(60 + event, event - 1, view="B" if event == 3 else "A")]
            )
            arm = event - 1
            if arm == missing_arm:
                payloads = []
            if arm == extra_arm:
                payloads = [*payloads, _payload(70 + event, arm)]
            for payload in payloads:
                rep.writer.mux.observe(payload)
            event += 1

    rep = SimpleNamespace(
        writer=None,
        Writer=object,
        AnnotatorRegistry=SimpleNamespace(
            get_annotator=lambda *_args, **_kwargs: object()
        ),
        trigger=SimpleNamespace(on_custom_event=lambda name: ("custom", name)),
        utils=Utils,
        orchestrator=object(),
        create=SimpleNamespace(render_product=lambda *_args: render_product),
    )

    def writer(_rep: Any, mux: Any) -> _Writer:
        selected = _Writer(
            rep,
            mux,
            log,
            attach_error=attach_error,
            detach_error=detach_error,
            detach_payload=detach_payload,
        )
        selected.mux = mux
        return selected

    def render(_orchestrator: Any, _profile: Any) -> None:
        log.append("render")
        if physics_change:
            env.unwrapped.state += 1
        if render_error is not None:
            raise render_error

    profile = S.R.fixed_profile()
    return {
        "profile": lambda: profile,
        "startup": lambda *_args: {
            "before": {"enabled": False},
            "after": {"enabled": True},
        },
        "parse_env_cfg": lambda *_args, **_kwargs: _env_cfg(),
        "make_env": lambda *_args: env,
        "rep": rep,
        "usd_context": lambda: (context, object()),
        "settings": lambda: object(),
        "enable_extension": lambda _name: True,
        "registration_probe": lambda: {name: True for name in S.SDK.ANNOTATORS},
        "camera": lambda _prim: _Camera(),
        "readiness": lambda *_args: log.append("readiness"),
        "renderer_readback": lambda *_args: {"renderer": "fake"},
        "fxaa_required": lambda _job: False,
        "fxaa_apply": lambda *_args: None,
        "fxaa_readback": lambda _settings: {},
        "fxaa_verify": lambda _value: None,
        "marker": lambda *_args: _Marker(log),
        "root_position": lambda _base: [0.0, 0.0, 0.5],
        "writer": writer,
        "physics_snapshot": lambda base: {
            "root_position_world_m": [0.0, 0.0, 0.5],
            "state": base.state,
        },
        "capture_render_step": render,
    }


@pytest.mark.parametrize("prepare_empty", [False, True])
def test_success_cleans_before_seal_and_writes_only_four_journals(
    tmp_path: Path, prepare_empty: bool
) -> None:
    log: list[str] = []
    sealed: list[dict[str, Any]] = []
    result = S.run_prepared_smoke(
        _validated(tmp_path),
        _App(log),
        seal_report=lambda report: (
            log.append("seal_report"),
            sealed.append(dict(report)),
        ),
        dependencies=_dependencies(log, prepare_empty=prepare_empty),
    )
    assert (
        log.index("writer_detach")
        < log.index("render_product_destroy")
        < log.index("env_close")
        < log.index("seal_report")
    )
    assert result["approved"] is True
    assert result["renderer_smoke_only"] is True
    assert set(result) == {
        "schema",
        "prepared",
        "capture_report",
        "artifacts",
        "lifecycle",
        "renderer",
        "setup_evidence",
        "renderer_smoke_only",
        "approved",
    }
    assert len(result["artifacts"]["journals"]) == 4
    assert len(list(tmp_path.glob("job.atomic-*.json"))) == 4
    assert result["prepared"]["preparation"]["journal_written"] is False
    assert sealed[0]["lifecycle"]["envelope_sealed"] is False
    assert result["lifecycle"]["envelope_sealed"] is True
    assert result["lifecycle"]["report_verified"] is True


@pytest.mark.parametrize("case", ["missing", "extra", "detach_late"])
def test_callback_cardinality_and_detach_late_fail_closed(
    tmp_path: Path, case: str
) -> None:
    log: list[str] = []
    kwargs: dict[str, Any] = {}
    if case == "missing":
        kwargs["missing_arm"] = 1
    elif case == "extra":
        kwargs["extra_arm"] = 2
    else:
        kwargs["detach_payload"] = _payload(65, 3)
    sealed: list[dict[str, Any]] = []
    with pytest.raises(S.PreparedSDKFailure) as caught:
        S.run_prepared_smoke(
            _validated(tmp_path),
            _App(log),
            seal_report=lambda report: sealed.append(dict(report)),
            dependencies=_dependencies(log, **kwargs),
        )
    assert caught.value.report["approved"] is False
    assert caught.value.report["lifecycle"]["environment_closed"] is True
    assert len(sealed) == 1
    assert not list(tmp_path.glob("job.atomic-*.json"))


@pytest.mark.parametrize("phase", ["attach", "render", "detach"])
def test_sdk_phase_failure_preserves_primary_and_cleanup(
    tmp_path: Path, phase: str
) -> None:
    log: list[str] = []
    primary = RuntimeError(f"{phase} failed")
    kwargs: dict[str, Any] = {
        "attach_error": primary if phase == "attach" else None,
        "render_error": primary if phase == "render" else None,
        "detach_error": primary if phase == "detach" else None,
    }
    with pytest.raises(S.PreparedSDKFailure) as caught:
        S.run_prepared_smoke(
            _validated(tmp_path),
            _App(log),
            seal_report=lambda _report: None,
            dependencies=_dependencies(log, **kwargs),
        )
    assert caught.value.primary is primary
    assert "env_close" in log


def test_cleanup_errors_are_all_attempted_and_hold(tmp_path: Path) -> None:
    log: list[str] = []
    destroy = RuntimeError("destroy")
    close = ValueError("close")
    with pytest.raises(S.PreparedSDKFailure) as caught:
        S.run_prepared_smoke(
            _validated(tmp_path),
            _App(log),
            seal_report=lambda _report: None,
            dependencies=_dependencies(log, destroy_error=destroy, close_error=close),
        )
    assert caught.value.primary is destroy
    assert log[-2:] == ["render_product_destroy", "env_close"]
    assert caught.value.report["lifecycle"]["environment_closed"] is False


def test_physics_change_is_hold_and_no_journal(tmp_path: Path) -> None:
    with pytest.raises(S.PreparedSDKFailure) as caught:
        S.run_prepared_smoke(
            _validated(tmp_path),
            _App([]),
            seal_report=lambda _report: None,
            dependencies=_dependencies([], physics_change=True),
        )
    assert caught.value.report["approved"] is False
    assert caught.value.report["artifacts"]["journals"] == []


def test_seal_failure_is_primary_after_cleanup(tmp_path: Path) -> None:
    log: list[str] = []
    primary = RuntimeError("seal")
    with pytest.raises(S.PreparedSDKFailure) as caught:
        S.run_prepared_smoke(
            _validated(tmp_path),
            _App(log),
            seal_report=lambda _report: (_ for _ in ()).throw(primary),
            dependencies=_dependencies(log),
        )
    assert caught.value.primary is primary
    assert log[-2:] == ["render_product_destroy", "env_close"]


def test_mode_mismatch_stops_before_dependencies(tmp_path: Path) -> None:
    class Exploding(dict[str, Any]):
        def __iter__(self):  # type: ignore[no-untyped-def]
            raise AssertionError("dependencies read")

    with pytest.raises(ValueError, match="mode"):
        S.run_prepared_smoke(
            _validated(tmp_path, "record"),
            _App([]),
            seal_report=lambda _report: None,
            dependencies=Exploding(),
        )


def test_injected_dependencies_do_not_use_sdk_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        S,
        "_load_dependencies",
        lambda: (_ for _ in ()).throw(AssertionError("SDK loader")),
    )
    result = S.run_prepared_smoke(
        _validated(tmp_path),
        _App([]),
        seal_report=lambda _report: None,
        dependencies=_dependencies([]),
    )
    assert result["approved"] is True


def test_cancellation_is_rethrown_with_envelope_after_cleanup(tmp_path: Path) -> None:
    log: list[str] = []
    cancellation = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt) as caught:
        S.run_prepared_smoke(
            _validated(tmp_path),
            _App(log, update_error=cancellation),
            seal_report=lambda _report: None,
            dependencies=_dependencies(log),
        )
    assert hasattr(caught.value, "prepared_sdk_envelope")
    envelope = cast(Any, caught.value).prepared_sdk_envelope
    assert envelope["approved"] is False
