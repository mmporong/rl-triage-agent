from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np  # pyright: ignore[reportMissingImports]
import pytest  # pyright: ignore[reportMissingImports]

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
import vision_frame_sync_v1 as FRAME  # pyright: ignore[reportMissingImports]
import vision_frame_trigger_checkpoint_v2 as D  # pyright: ignore[reportMissingImports]
import vision_prepared_capture_v1 as MUX  # pyright: ignore[reportMissingImports]

RP = "RenderProduct_0"
CAMERA = "/World/Camera"
PHYSICS = {"joint": [0.1, 0.2], "step": 7}


def _payload(frame: int, marker: int) -> dict[str, Any]:
    width = height = 10
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[:, :, 3] = 255
    mask = np.zeros((height, width), dtype=np.uint32)
    labels: dict[str, Any] = {"0": {"class": "robot/body"}}
    for index, label in enumerate(sorted(FRAME.marker_labels(marker)), 1):
        mask[index, 1:9] = index
        labels[str(index)] = {"class": label}
        role = "sentinel" if label == FRAME.SENTINEL else label.rsplit("/", 1)[-1]
        rgba[index, 1:9, :3] = FRAME.PALETTE[role]
    times = {
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
        "systemTime": 1_000.0 + frame,
    }
    return {
        "swhFrameNumber": 0,
        "reference_time": (frame, 30),
        "distribution_outputs": {"node": "distribution"},
        "trigger_outputs": {"frame": frame},
        "named_outputs": {"node": "named"},
        "annotators": {
            "camera": {RP: CAMERA},
            "resolution": {RP: np.array([width, height], dtype=np.int32)},
            "rgb": {RP: {"data": rgba}},
            "semantic_segmentation": {RP: {"data": mask, "idToLabels": labels}},
            "CameraParams": {
                RP: {
                    "cameraViewTransform": np.arange(16, dtype=np.float64),
                    "cameraProjection": np.eye(4, dtype=np.float64).reshape(16),
                    "renderProductResolution": np.array(
                        [width, height], dtype=np.int32
                    ),
                }
            },
            "IsaacReadTimes": {RP: times},
        },
    }


def _bootstrap(frame: int) -> dict[str, Any]:
    value = _payload(frame, 3)
    value["annotators"]["semantic_segmentation"][RP]["data"] = np.zeros(
        (0,), dtype=np.float64
    )
    return value


class _Settings:
    def __init__(self) -> None:
        self.values = {D.BOUNDARY.STARTUP.CAPTURE_ON_PLAY: True}

    def get(self, key: str) -> Any:
        return self.values.get(key)

    def set_bool(self, key: str, value: bool) -> None:
        self.values[key] = value


class _WriterBase:
    rep: Any = None

    def attach(self, _render_product: Any, *args: Any, **kwargs: Any) -> None:
        self.rep.order.append("attach")
        self.rep.attach_calls.append({"args": args, "kwargs": kwargs})
        self.rep.writer = self

    def detach(self) -> None:
        self.rep.order.append("detach")

    def schedule_write(self) -> None:
        self.rep.schedule_calls += 1


class _Orchestrator:
    def __init__(
        self, rep: Any, callbacks_per_capture: int, baseline_callbacks: int
    ) -> None:
        self.rep = rep
        self.callbacks_per_capture = callbacks_per_capture
        self.baseline_callbacks = baseline_callbacks
        self.frame = 57
        self.steps = 0

    def step(self, *_args: Any, **_kwargs: Any) -> None:
        self.rep.order.append("step")
        self.steps += 1
        self.frame += 2 if self.steps == 2 else 1
        self.rep.scheduled.append([self.frame, 30])
        if self.steps == 1:
            for _ in range(self.baseline_callbacks):
                self.rep.writer.write(_payload(self.frame, 3))
            return
        marker = 3 if self.steps == 2 else self.steps - 3
        if self.steps == 2 and self.callbacks_per_capture:
            self.rep.writer.write(_bootstrap(self.frame))
        for _ in range(self.callbacks_per_capture):
            self.rep.writer.write(_payload(self.frame, marker))


def _deps(
    *, callbacks_per_capture: int = 1, baseline_callbacks: int = 0
) -> dict[str, Any]:
    settings = _Settings()
    rep = SimpleNamespace(
        writer=None,
        Writer=_WriterBase,
        AnnotatorRegistry=SimpleNamespace(
            get_annotator=lambda *_args, **_kwargs: object()
        ),
        trigger=SimpleNamespace(on_custom_event=lambda name: ("custom", name)),
        attach_calls=[],
        trigger_factory_calls=0,
        event_calls=0,
        schedule_calls=0,
        scheduled=[],
        order=[],
    )

    def send_event(_name: str) -> None:
        rep.event_calls += 1

    def custom_event(_name: str) -> object:
        rep.trigger_factory_calls += 1
        return object()

    rep.trigger = SimpleNamespace(on_custom_event=custom_event)
    rep.utils = SimpleNamespace(send_og_event=send_event)
    _WriterBase.rep = rep
    rep.orchestrator = _Orchestrator(
        rep, callbacks_per_capture, baseline_callbacks
    )
    return {name: object() for name in D.PREPARED._DEPENDENCY_KEYS} | {
        "rep": rep,
        "settings": lambda: settings,
        "startup": lambda _rep, selected: (
            selected.set_bool(D.BOUNDARY.STARTUP.CAPTURE_ON_PLAY, False),
            {"before": {"capture_on_play": True}, "after": {"capture_on_play": False}},
        )[1],
        "writer": D.PREPARED._create_writer,
        "physics_snapshot": lambda _base: PHYSICS,
        "capture_render_step": lambda orchestrator, _profile: orchestrator.step(
            delta_time=0.0, rt_subframes=1, pause_timeline=False
        ),
    }


def _validated(tmp_path: Path) -> dict[str, Any]:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    return {
        "job": {
            "mode": D.MODE,
            "callback_variant": D.VARIANT,
            "profile": {
                "delta_time_s": 0.0,
                "rt_subframes": 1,
                "pause_timeline": False,
            },
        },
        "checkpoint_path": checkpoint,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    }


def _run_checkpoint() -> Any:
    def run(
        validated: dict[str, Any],
        _app: Any,
        *,
        seal_report: Any,
        dependencies: dict[str, Any],
    ) -> dict[str, Any]:
        assert validated["job"]["mode"] == D.CHECKPOINT.MODE
        settings = dependencies["settings"]()
        dependencies["startup"](dependencies["rep"], settings)
        mux = MUX.PreparedCaptureMux(
            render_product_ref=RP, camera_path=CAMERA, resolution=[10, 10]
        )
        writer = dependencies["writer"](dependencies["rep"], mux)
        writer.attach(
            RP, trigger=dependencies["rep"].trigger.on_custom_event("capture")
        )
        base = SimpleNamespace(sim=SimpleNamespace(current_time=2.0))
        dependencies["physics_snapshot"](base)
        mux.begin_prepare(PHYSICS, 2.0, marker_code=3)
        dependencies["rep"].utils.send_og_event("capture")
        dependencies["capture_render_step"](dependencies["rep"].orchestrator, {})
        writer.raise_callback_error()
        mux.end_prepare(PHYSICS, 2.0)
        try:
            for ordinal in range(4):
                mux.begin_arm(ordinal, PHYSICS, 2.0, marker_code=ordinal)
                dependencies["rep"].utils.send_og_event("capture")
                dependencies["capture_render_step"](
                    dependencies["rep"].orchestrator, {}
                )
                writer.raise_callback_error()
                mux.end_arm(ordinal, PHYSICS, 2.0)
        except BaseException as error:
            writer.detach()
            report = {
                "schema": D.CHECKPOINT.SCHEMA,
                "prepared_smoke": {"approved": False},
                "policy_probe": None,
                "approved": False,
            }
            seal_report(report)
            raise D.CHECKPOINT.CheckpointSmokeSDKFailure(report, error) from error
        writer.detach()
        dependencies["rep"].order.append("policy_probe")
        report = {
            "schema": D.CHECKPOINT.SCHEMA,
            "prepared_smoke": {"approved": True},
            "policy_probe": {
                "environment_steps": 1,
                "capture_completed_before_policy_step": True,
                "writer_detached_before_policy_step": True,
                "actual_environment_observation_used": True,
                "pure_checkpoint_read_only": False,
            },
            "approved": True,
        }
        seal_report(report)
        return report

    return run


def test_default_attach_runs_zero_callback_baseline_then_strict_captures(
    tmp_path: Path,
) -> None:
    deps = _deps()
    sealed: list[dict[str, Any]] = []
    result = D.run_frame_trigger_diagnostic(
        _validated(tmp_path),
        object(),
        seal_report=lambda value: sealed.append(dict(value)),
        dependencies=deps,
        run_sdk=_run_checkpoint(),
        fabric_time_reader=lambda rep: list(rep.scheduled[-1]),
    )
    rep = deps["rep"]
    assert rep.attach_calls == [{"args": (), "kwargs": {}}]
    assert rep.trigger_factory_calls == 0
    assert rep.event_calls == 0
    assert rep.schedule_calls == 0
    assert rep.orchestrator.steps == 6
    assert rep.order == ["attach", *(["step"] * 6), "detach", "policy_probe"]
    assert result["baseline"] == {
        "attempted": True,
        "complete": True,
        "callback_count_before": 0,
        "callback_count_after": 0,
        "callback_count_delta": 0,
        "physics_sha256_before": result["baseline"]["physics_sha256_after"],
        "physics_sha256_after": result["baseline"]["physics_sha256_before"],
        "physx_time_before_s": 2.0,
        "physx_time_after_s": 2.0,
        "physics_unchanged": True,
        "physx_time_unchanged": True,
        "scheduled_reference_time": [58, 30],
    }
    assert result["fabric_time"]["captures"] == [
        [60, 30],
        [61, 30],
        [62, 30],
        [63, 30],
        [64, 30],
    ]
    assert result["fabric_time"]["callbacks_match_scheduled_reference_time"] is True
    assert result["dispatch_counts"] == {
        "on_custom_event_requests_suppressed": 1,
        "on_custom_event_underlying_calls": 0,
        "send_og_event_requests_suppressed": 5,
        "send_og_event_underlying_calls": 0,
        "schedule_write_calls": 0,
        "orchestrator_step_calls": 6,
    }
    assert result["diagnostic_complete"] is True
    assert result["approved"] is False
    accepted = [
        row for row in result["callbacks"] if row["classification"] == "valid"
    ]
    assert [row["window"] for row in accepted] == [
        "preparing",
        "arming0",
        "arming1",
        "arming2",
        "arming3",
    ]
    assert [row["marker_code"] for row in accepted] == [3, 0, 1, 2, 3]
    assert sealed == [result]


@pytest.mark.parametrize("callbacks_per_capture", [0, 2])
def test_zero_or_duplicate_capture_callback_is_preserved_and_rejected(
    tmp_path: Path, callbacks_per_capture: int
) -> None:
    deps = _deps(callbacks_per_capture=callbacks_per_capture)
    sealed: list[dict[str, Any]] = []
    with pytest.raises(D.FrameTriggerDiagnosticFailure) as caught:
        D.run_frame_trigger_diagnostic(
            _validated(tmp_path),
            object(),
            seal_report=lambda value: sealed.append(dict(value)),
            dependencies=deps,
            run_sdk=_run_checkpoint(),
            fabric_time_reader=lambda rep: list(rep.scheduled[-1]),
        )
    report = caught.value.report
    assert report["diagnostic_complete"] is False
    assert report["approved"] is False
    assert report["diagnostic_complete"] is False
    assert deps["rep"].event_calls == 0
    assert deps["rep"].schedule_calls == 0
    assert sealed == [report]


def test_baseline_callback_is_preserved_and_rejected(tmp_path: Path) -> None:
    deps = _deps(baseline_callbacks=1)
    sealed: list[dict[str, Any]] = []
    with pytest.raises(D.FrameTriggerDiagnosticFailure) as caught:
        D.run_frame_trigger_diagnostic(
            _validated(tmp_path),
            object(),
            seal_report=lambda value: sealed.append(dict(value)),
            dependencies=deps,
            run_sdk=_run_checkpoint(),
            fabric_time_reader=lambda rep: list(rep.scheduled[-1]),
        )
    report = caught.value.report
    assert len(report["callbacks"]) == 1
    assert report["baseline"]["attempted"] is True
    assert report["diagnostic_complete"] is False
    assert sealed == [report]


def test_default_fabric_reader_uses_sdk_scheduled_reference_time() -> None:
    rep = SimpleNamespace(
        orchestrator=SimpleNamespace(
            _orchestrator=SimpleNamespace(_sim_times_to_write=[(116, 60)])
        )
    )
    assert D._scheduled_reference_time(rep) == [29, 15]


def test_sdk_reader_matches_equivalent_unreduced_callback_time(tmp_path: Path) -> None:
    deps = _deps()
    rep = deps["rep"]
    rep.orchestrator._orchestrator = SimpleNamespace(
        _sim_times_to_write=rep.scheduled
    )
    result = D.run_frame_trigger_diagnostic(
        _validated(tmp_path),
        object(),
        seal_report=lambda _value: None,
        dependencies=deps,
        run_sdk=_run_checkpoint(),
    )
    assert result["baseline"]["scheduled_reference_time"] == [29, 15]
    assert result["fabric_time"]["captures"][0] == [2, 1]
    assert result["callbacks"][1]["reference_time"] == [60, 30]
    assert result["fabric_time"]["callbacks_match_scheduled_reference_time"] is True
    assert result["diagnostic_complete"] is True


def test_sdk_reader_rejects_different_callback_time(tmp_path: Path) -> None:
    deps = _deps()
    with pytest.raises(D.FrameTriggerDiagnosticFailure) as caught:
        D.run_frame_trigger_diagnostic(
            _validated(tmp_path),
            object(),
            seal_report=lambda _value: None,
            dependencies=deps,
            run_sdk=_run_checkpoint(),
            fabric_time_reader=lambda rep: [rep.scheduled[-1][0] + 1, 30],
        )
    assert caught.value.report["diagnostic_complete"] is False
    assert (
        caught.value.report["fabric_time"]["callbacks_match_scheduled_reference_time"]
        is False
    )
