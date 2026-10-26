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
import vision_prepared_capture_v1 as MUX  # pyright: ignore[reportMissingImports]
import vision_semantic_settle_checkpoint_v1 as D  # pyright: ignore[reportMissingImports]

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


def _bootstrap(frame: int) -> dict[str, Any]:
    value = _payload(frame, 3)
    value["annotators"]["semantic_segmentation"][RP]["data"] = np.zeros(
        (0,), dtype=np.float64
    )
    return value


class _Attribute:
    def __init__(self, name: str, value: Any) -> None:
        self.name, self.value = name, value

    def GetName(self) -> str:
        return self.name

    def Get(self) -> Any:
        return self.value


class _Prim:
    def __init__(self, label: str) -> None:
        self.visibility = _Attribute("visibility", "invisible")
        self.semantic = _Attribute("semantic:class", label)

    def IsValid(self) -> bool:
        return True

    def GetAttribute(self, name: str) -> _Attribute:
        assert name == "visibility"
        return self.visibility

    def GetAttributes(self) -> list[_Attribute]:
        return [self.visibility, self.semantic]


class _Stage:
    def __init__(self) -> None:
        self.prims = {
            f"/Marker/{index}": _Prim(f"marker/{index}") for index in range(9)
        }

    def GetPrimAtPath(self, path: str) -> _Prim:
        return self.prims[path]


class _Marker:
    def __init__(self, rep: Any) -> None:
        self.rep = rep
        self._paths = {str(index): f"/Marker/{index}" for index in range(9)}

    def bind_and_validate(self, _stage: _Stage) -> None:
        return None

    def set_code(self, stage: _Stage, code: int) -> None:
        for index, path in enumerate(self._paths.values()):
            stage.GetPrimAtPath(path).visibility.value = (
                "inherited"
                if index in {(code + offset) % 9 for offset in range(5)}
                else "invisible"
            )
        self.rep.pending_marker = code


class _Settings:
    def __init__(self) -> None:
        self.values = {D.BOUNDARY.STARTUP.CAPTURE_ON_PLAY: True}

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
    def __init__(self, rep: Any, *, unsolicited: bool, advance_physics: bool) -> None:
        self.rep = rep
        self.unsolicited = unsolicited
        self.advance_physics = advance_physics
        self.frame = 40
        self.first_scheduled = True

    def step(self, *_args: Any, **_kwargs: Any) -> None:
        if not self.rep.scheduled:
            self.rep.current_marker = self.rep.pending_marker
            if self.advance_physics:
                self.rep.base.sim.current_time += 0.1
            if self.unsolicited:
                self.rep.writer.write(_payload(self.frame, self.rep.current_marker))
            return
        self.rep.scheduled = False
        self.frame += 1
        if self.first_scheduled:
            self.first_scheduled = False
            self.rep.writer.write(_bootstrap(self.frame))
            self.rep.current_marker = self.rep.pending_marker
        self.rep.writer.write(_payload(self.frame, self.rep.current_marker))


def _deps(
    *, unsolicited: bool = False, advance_physics: bool = False
) -> dict[str, Any]:
    settings = _Settings()
    base = SimpleNamespace(sim=SimpleNamespace(current_time=2.0))
    rep = SimpleNamespace(
        writer=None,
        Writer=_WriterBase,
        AnnotatorRegistry=SimpleNamespace(
            get_annotator=lambda *_args, **_kwargs: object()
        ),
        trigger=SimpleNamespace(on_custom_event=lambda name: ("custom", name)),
        utils=SimpleNamespace(send_og_event=lambda _name: None),
        scheduled=False,
        pending_marker=3,
        current_marker=3,
        base=base,
    )
    _WriterBase.rep = rep
    rep.orchestrator = _Orchestrator(
        rep, unsolicited=unsolicited, advance_physics=advance_physics
    )
    return {name: object() for name in D.PREPARED._DEPENDENCY_KEYS} | {
        "rep": rep,
        "settings": lambda: settings,
        "startup": lambda _rep, selected: (
            selected.set_bool(D.BOUNDARY.STARTUP.CAPTURE_ON_PLAY, False),
            {"before": {"capture_on_play": True}, "after": {"capture_on_play": False}},
        )[1],
        "writer": D.PREPARED._create_writer,
        "marker": lambda selected, _root: _Marker(selected),
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
        stage = _Stage()
        marker = dependencies["marker"](dependencies["rep"], object())
        marker.bind_and_validate(stage)
        mux = MUX.PreparedCaptureMux(
            render_product_ref=RP, camera_path=CAMERA, resolution=[10, 10]
        )
        writer = dependencies["writer"](dependencies["rep"], mux)
        writer.attach(
            RP, trigger=dependencies["rep"].trigger.on_custom_event("capture")
        )
        marker.set_code(stage, 3)
        before = dependencies["physics_snapshot"](dependencies["rep"].base)
        mux.begin_prepare(before, 2.0, marker_code=3)
        dependencies["rep"].utils.send_og_event("capture")
        dependencies["capture_render_step"](
            dependencies["rep"].orchestrator, validated["job"]["profile"]
        )
        writer.raise_callback_error()
        mux.end_prepare(dependencies["physics_snapshot"](dependencies["rep"].base), 2.0)
        for ordinal in range(4):
            marker.set_code(stage, ordinal)
            before = dependencies["physics_snapshot"](dependencies["rep"].base)
            mux.begin_arm(ordinal, before, 2.0, marker_code=ordinal)
            dependencies["rep"].utils.send_og_event("capture")
            dependencies["capture_render_step"](
                dependencies["rep"].orchestrator, validated["job"]["profile"]
            )
            writer.raise_callback_error()
            mux.end_arm(
                ordinal, dependencies["physics_snapshot"](dependencies["rep"].base), 2.0
            )
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


def test_unscheduled_settle_removes_stale_semantic_duplicate_before_each_arm(
    tmp_path: Path,
) -> None:
    sealed: list[dict[str, Any]] = []
    result = D.run_semantic_settle_diagnostic(
        _validated(tmp_path),
        object(),
        seal_report=lambda value: sealed.append(dict(value)),
        dependencies=_deps(),
        run_sdk=_run_checkpoint(),
    )
    assert result["diagnostic_complete"] is True
    assert result["approved"] is False
    assert result["used_for_decision"] is False
    assert [row["callback_delta"] for row in result["settles"]] == [0, 0, 0, 0]
    assert [row["phase"] for row in result["marker_states"]] == [
        "bound",
        "prepare",
        "arm_0",
        "arm_1",
        "arm_2",
        "arm_3",
    ]
    callbacks = result["boundary"]["callbacks"]
    assert [row["classification"] for row in callbacks] == [
        "bootstrap_empty",
        "valid",
        "valid",
        "valid",
        "valid",
        "valid",
    ]
    assert [row["window"] for row in callbacks[-4:]] == [
        "arming0",
        "arming1",
        "arming2",
        "arming3",
    ]
    assert sealed == [result]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"unsolicited": True}, "unscheduled semantic settle 중 callback"),
        ({"advance_physics": True}, "physics 또는 simulation time"),
    ],
)
def test_settle_callback_or_physics_advance_fails_closed(
    tmp_path: Path, kwargs: dict[str, bool], message: str
) -> None:
    sealed: list[dict[str, Any]] = []
    with pytest.raises(D.SemanticSettleDiagnosticFailure) as caught:
        D.run_semantic_settle_diagnostic(
            _validated(tmp_path),
            object(),
            seal_report=lambda value: sealed.append(dict(value)),
            dependencies=_deps(**kwargs),
            run_sdk=_run_checkpoint(),
        )
    report = caught.value.report
    assert report["diagnostic_complete"] is False
    assert report["approved"] is False
    assert message in report["primary_error"]["error_message"]
    assert sealed == [report]
