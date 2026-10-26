from __future__ import annotations

# ruff: noqa: E402

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
import vision_frame_trigger_checkpoint_v1 as D  # pyright: ignore[reportMissingImports]
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
        self.rep.attach_calls.append({"args": args, "kwargs": kwargs})
        self.rep.writer = self

    def detach(self) -> None:
        return None

    def schedule_write(self) -> None:
        self.rep.schedule_calls += 1


class _Orchestrator:
    def __init__(self, rep: Any, duplicate_arms: bool) -> None:
        self.rep = rep
        self.duplicate_arms = duplicate_arms
        self.frame = 40
        self.steps = 0

    def step(self, *_args: Any, **_kwargs: Any) -> None:
        self.steps += 1
        self.frame += 1
        if self.steps == 1:
            self.rep.writer.write(_bootstrap(self.frame))
            self.rep.writer.write(_payload(self.frame, 3))
            return
        count = 2 if self.duplicate_arms else 1
        for _ in range(count):
            self.rep.writer.write(_payload(self.frame, self.steps - 2))


def _deps(*, duplicate_arms: bool = False) -> dict[str, Any]:
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
    )

    def send_event(_name: str) -> None:
        rep.event_calls += 1

    def custom_event(_name: str) -> object:
        rep.trigger_factory_calls += 1
        return object()

    rep.trigger = SimpleNamespace(on_custom_event=custom_event)
    rep.utils = SimpleNamespace(send_og_event=send_event)
    _WriterBase.rep = rep
    rep.orchestrator = _Orchestrator(rep, duplicate_arms)
    return {name: object() for name in D.PREPARED._DEPENDENCY_KEYS} | {
        "rep": rep,
        "settings": lambda: settings,
        "startup": lambda _rep, selected: (
            selected.set_bool(D.BOUNDARY.STARTUP.CAPTURE_ON_PLAY, False),
            {"before": {"capture_on_play": True}, "after": {"capture_on_play": False}},
        )[1],
        "writer": D.PREPARED._create_writer,
        "capture_render_step": lambda orchestrator, _profile: orchestrator.step(
            delta_time=0.0, rt_subframes=1, pause_timeline=False
        ),
    }


def _validated(tmp_path: Path) -> dict[str, Any]:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    return {
        "job": {"mode": D.MODE, "callback_variant": D.VARIANT},
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


def test_default_attach_omits_trigger_and_uses_step_only(tmp_path: Path) -> None:
    deps = _deps()
    sealed: list[dict[str, Any]] = []
    result = D.run_frame_trigger_diagnostic(
        _validated(tmp_path),
        object(),
        seal_report=lambda value: sealed.append(dict(value)),
        dependencies=deps,
        run_sdk=_run_checkpoint(),
    )
    rep = deps["rep"]
    assert rep.attach_calls == [{"args": (), "kwargs": {}}]
    assert rep.trigger_factory_calls == 0
    assert rep.event_calls == 0
    assert rep.schedule_calls == 0
    assert rep.orchestrator.steps == 5
    assert result["dispatch_counts"] == {
        "on_custom_event_requests_suppressed": 1,
        "on_custom_event_underlying_calls": 0,
        "send_og_event_requests_suppressed": 5,
        "send_og_event_underlying_calls": 0,
        "schedule_write_calls": 0,
        "orchestrator_step_calls": 5,
    }
    assert result["diagnostic_complete"] is True
    assert result["approved"] is False
    assert [row["window"] for row in result["callbacks"][-4:]] == [
        "arming0",
        "arming1",
        "arming2",
        "arming3",
    ]
    assert sealed == [result]


def test_duplicate_default_frame_callback_is_preserved_and_rejected(
    tmp_path: Path,
) -> None:
    deps = _deps(duplicate_arms=True)
    sealed: list[dict[str, Any]] = []
    with pytest.raises(D.FrameTriggerDiagnosticFailure) as caught:
        D.run_frame_trigger_diagnostic(
            _validated(tmp_path),
            object(),
            seal_report=lambda value: sealed.append(dict(value)),
            dependencies=deps,
            run_sdk=_run_checkpoint(),
        )
    report = caught.value.report
    assert report["diagnostic_complete"] is False
    assert report["approved"] is False
    assert len(report["callbacks"]) == 4
    assert report["callbacks"][-1]["mux_failure"]["code"] in {
        "callback_cardinality_invalid",
        "arm_callback_cardinality_invalid",
    }
    assert deps["rep"].event_calls == 0
    assert deps["rep"].schedule_calls == 0
    assert sealed == [report]
