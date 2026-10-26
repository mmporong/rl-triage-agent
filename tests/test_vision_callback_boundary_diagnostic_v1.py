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
import vision_callback_boundary_diagnostic_v1 as D  # pyright: ignore[reportMissingImports]
import vision_frame_sync_v1 as FRAME  # pyright: ignore[reportMissingImports]
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
    view = np.arange(16, dtype=np.float64)
    return {
        "swhFrameNumber": 0,
        "reference_time": (frame, 30),
        "distribution_outputs": {"node": "distribution"},
        "trigger_outputs": {"event": marker},
        "named_outputs": {"node": "named"},
        "annotators": {
            "camera": {RP: CAMERA},
            "resolution": {RP: np.array([width, height], dtype=np.int32)},
            "rgb": {RP: {"data": rgba}},
            "semantic_segmentation": {RP: {"data": mask, "idToLabels": labels}},
            "CameraParams": {
                RP: {
                    "cameraViewTransform": view,
                    "cameraProjection": np.eye(4, dtype=np.float64).reshape(16),
                    "renderProductResolution": np.array(
                        [width, height], dtype=np.int32
                    ),
                }
            },
            "IsaacReadTimes": {RP: times},
        },
    }


class _Settings:
    def __init__(self) -> None:
        self.values = {D.STARTUP.CAPTURE_ON_PLAY: True}

    def get(self, key: str) -> Any:
        return self.values.get(key)

    def set_bool(self, key: str, value: bool) -> None:
        self.values[key] = value


class _WriterBase:
    rep: Any = None

    def attach(self, _render_product: Any, *, trigger: Any) -> None:
        self.trigger = trigger
        self.rep.writer = self

    def detach(self) -> None:
        return None

    def schedule_write(self) -> None:
        self.rep.scheduled = True


class _Orchestrator:
    def __init__(self, rep: Any, callbacks_per_step: int) -> None:
        self.rep = rep
        self.callbacks_per_step = callbacks_per_step
        self.frame = 40

    def step(self, *_args: Any, **_kwargs: Any) -> None:
        count = self.callbacks_per_step
        if self.rep.writer.trigger is None:
            count = 1 if self.rep.scheduled else 0
        self.rep.scheduled = False
        self.frame += 1
        for _ in range(count):
            self.rep.writer.write(_payload(self.frame, 3))


def _deps(callbacks_per_step: int) -> dict[str, Any]:
    settings = _Settings()
    rep = SimpleNamespace(
        writer=None,
        Writer=_WriterBase,
        AnnotatorRegistry=SimpleNamespace(
            get_annotator=lambda *_args, **_kwargs: object()
        ),
        trigger=SimpleNamespace(on_custom_event=lambda name: ("custom", name)),
        utils=SimpleNamespace(send_og_event=lambda _name: None),
        scheduled=False,
    )
    _WriterBase.rep = rep
    rep.orchestrator = _Orchestrator(rep, callbacks_per_step)
    return {name: object() for name in D.PREPARED._DEPENDENCY_KEYS} | {
        "rep": rep,
        "settings": lambda: settings,
        "startup": lambda _rep, selected: (
            selected.set_bool(D.STARTUP.CAPTURE_ON_PLAY, False),
            {
                "before": {"capture_on_play": True},
                "after": {"capture_on_play": False},
            },
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
        "job": {"mode": D.MODE},
        "checkpoint_path": checkpoint,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    }


def _sdk_probe(expected_callbacks: int) -> Any:
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
            RP,
            trigger=dependencies["rep"].trigger.on_custom_event(D.PREPARED.EVENT_NAME),
        )
        mux.begin_prepare(PHYSICS, 2.0, marker_code=3)
        dependencies["rep"].utils.send_og_event(D.PREPARED.EVENT_NAME)
        dependencies["capture_render_step"](
            dependencies["rep"].orchestrator,
            {"delta_time_s": 0.0, "rt_subframes": 1, "pause_timeline": False},
        )
        writer.raise_callback_error()
        if expected_callbacks == 1:
            mux.end_prepare(PHYSICS, 2.0)
        writer.detach()
        report = {
            "schema": D.CHECKPOINT.SCHEMA,
            "prepared_smoke": {"approved": expected_callbacks == 1},
            "policy_probe": None,
            "approved": False,
        }
        seal_report(report)
        if expected_callbacks != 1:
            error = D.CHECKPOINT.CheckpointSmokeSDKFailure(
                report, RuntimeError("callback_cardinality_invalid")
            )
            raise error
        return report

    return run


@pytest.mark.parametrize(
    ("variant", "callbacks_per_step", "expected_callbacks", "expected_schedule"),
    [("custom_event", 2, 2, 0), ("manual_schedule", 2, 1, 1)],
)
def test_actual_writer_and_strict_mux_preserve_callback_boundary(
    tmp_path: Path,
    variant: str,
    callbacks_per_step: int,
    expected_callbacks: int,
    expected_schedule: int,
) -> None:
    sealed: list[dict[str, Any]] = []
    result = D.run_boundary_diagnostic(
        _validated(tmp_path),
        object(),
        variant=variant,
        seal_report=lambda value: sealed.append(dict(value)),
        dependencies=_deps(callbacks_per_step),
        run_sdk=_sdk_probe(expected_callbacks),
    )
    assert set(result) == D.TOP_KEYS
    assert result["diagnostic_complete"] is True
    assert result["approved"] is False
    assert result["used_for_decision"] is False
    assert len(result["callbacks"]) == expected_callbacks
    assert [row["ordinal"] for row in result["callbacks"]] == list(
        range(expected_callbacks)
    )
    assert all(row["window"] == "preparing" for row in result["callbacks"])
    names = [row["name"] for row in result["operations"]["timeline"]]
    assert names.count("schedule_write") == expected_schedule
    assert names.count("orchestrator_step") == 1
    assert names.index("send_og_event") < names.index("orchestrator_step")
    attach = next(
        row
        for row in result["operations"]["timeline"]
        if row["name"] == "writer_attach"
    )
    assert attach["evidence"]["trigger_is_none"] is (variant == "manual_schedule")
    step = next(
        row
        for row in result["operations"]["timeline"]
        if row["name"] == "orchestrator_step"
    )
    assert step["evidence"]["kwargs"] == {
        "delta_time": 0.0,
        "pause_timeline": False,
        "rt_subframes": 1,
    }
    counts = result["operations"]["counts"]
    assert counts["orchestrator_step"] == {
        "started": 1,
        "complete": 1,
        "failed": 0,
    }
    assert (
        result["callbacks"][0]["operation_counts_before"]["orchestrator_step"][
            "started"
        ]
        == 1
    )
    if variant == "custom_event":
        assert result["primary_error"]["error_message"] == (
            "callback_cardinality_invalid"
        )
        assert result["callbacks"][-1]["mux_failure"]["code"] == (
            "prepare_callback_sequence_invalid"
        )
    else:
        assert result["primary_error"] is None
    assert result["capture_on_play"] == {
        "before": True,
        "after_startup": False,
        "restore_changed": True,
        "restored": True,
    }
    assert sealed == [result]


def test_missing_render_boundary_fails_closed(tmp_path: Path) -> None:
    def no_render(
        _validated: dict[str, Any],
        _app: Any,
        *,
        seal_report: Any,
        dependencies: dict[str, Any],
    ) -> dict[str, Any]:
        report = {"schema": D.CHECKPOINT.SCHEMA, "approved": False}
        seal_report(report)
        return report

    sealed: list[dict[str, Any]] = []
    with pytest.raises(D.CallbackBoundaryDiagnosticFailure) as caught:
        D.run_boundary_diagnostic(
            _validated(tmp_path),
            object(),
            variant="custom_event",
            seal_report=lambda value: sealed.append(dict(value)),
            dependencies=_deps(1),
            run_sdk=no_render,
        )
    assert caught.value.report["diagnostic_complete"] is False
    assert caught.value.report["approved"] is False
    assert sealed == [caught.value.report]
