from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "bench"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import record_vision_evidence_v1 as RECORD  # pyright: ignore[reportMissingImports]
import vision_annotator_pull_collector_v1 as COLLECTOR  # pyright: ignore[reportMissingImports]
import vision_annotator_pull_validator_v1 as VALIDATOR  # pyright: ignore[reportMissingImports]
import vision_frame_sync_v1 as FRAME  # pyright: ignore[reportMissingImports]


class FakeAnnotator:
    def __init__(self, name: str, state: dict[str, Any]) -> None:
        self.name = name
        self.state = state
        self.attached: list[Any] = []
        self.detached: list[Any] = []
        self.calls: list[dict[str, Any]] = []

    def attach(self, render_product: Any) -> None:
        self.attached.append(render_product)
        self.state["events"].append(("attach", self.name))

    def get_data(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        self.state["events"].append(("pull", self.state["ordinal"], self.name))
        if self.state.get("fail_pull") == (self.state["ordinal"], self.name):
            raise LookupError("pull failed")
        if self.state.get("update_during_pull") == (
            self.state["ordinal"],
            self.name,
        ):
            self.state["updates"] += 1
        return self.state["payloads"][self.name]

    def detach(self, render_product: Any) -> None:
        self.detached.append(render_product)
        self.state["events"].append(("detach", self.name))


class FakeRegistry:
    def __init__(self, state: dict[str, Any]) -> None:
        self.state = state
        self.annotators: dict[str, FakeAnnotator] = {}
        self.requests: list[tuple[str, dict[str, Any]]] = []

    def get_annotator(self, name: str, **kwargs: Any) -> FakeAnnotator:
        self.requests.append((name, kwargs))
        annotator = FakeAnnotator(name, self.state)
        self.annotators[name] = annotator
        return annotator


class FakeOrchestrator:
    def __init__(self, state: dict[str, Any]) -> None:
        self.state = state
        self.steps: list[dict[str, Any]] = []

    def step(self, **kwargs: Any) -> None:
        self.steps.append(kwargs)
        self.state["events"].append(("step", self.state["ordinal"]))
        self.state["updates"] += 2
        self.state["frame"] += self.state.get("frame_increment", 1)
        self.state["payloads"] = _payloads(self.state)


def _view(kind: str) -> list[float]:
    matrix = np.eye(4, dtype=np.float64)
    matrix[3, 0] = 1.0 if kind == "A" else 2.0
    return matrix.reshape(-1).tolist()


def _payloads(state: dict[str, Any]) -> dict[str, Any]:
    code = state["code"]
    labels = sorted(FRAME.marker_labels(code))
    mask = np.zeros((6, 6), dtype=np.uint32)
    rgba = np.zeros((6, 6, 4), dtype=np.uint8)
    rgba[:, :, 3] = 255
    id_to_labels: dict[str, Any] = {}
    for runtime_id, label in enumerate(labels, start=1):
        row = runtime_id - 1
        mask[row, :] = runtime_id
        id_to_labels[str(runtime_id)] = {"class": label}
        role = "sentinel" if label == FRAME.SENTINEL else label.rsplit("/", 1)[1]
        rgba[row, :, :3] = FRAME.PALETTE[role]
    frame = state["frame"]
    read_times: dict[str, Any] = {
        "durationDenominator": 30,
        "durationNumerator": 1,
        "externalTimeOfSimNs": frame * 33_333_333,
        "frameNumber": frame,
        "rationalTimeOfSimDenominator": 30,
        "rationalTimeOfSimNumerator": frame,
        "sampleTimeOffsetInSimFrames": 0,
        "simulationTime": state["simulation_time"],
        "simulationTimeMonotonic": state["simulation_time"],
        "swhFrameNumber": frame,
        "systemTime": 1000.0 + frame,
    }
    assert set(read_times) == set(RECORD.ISAAC_READ_TIMES_FIELDS)
    return {
        "rgb": rgba,
        "semantic_segmentation": {"data": mask, "info": {"idToLabels": id_to_labels}},
        "CameraParams": {
            "cameraViewTransform": _view(state["camera_kind"]),
            "cameraProjection": np.eye(4).reshape(-1).tolist(),
            "renderProductResolution": [6, 6],
            "cameraAperture": [20.955, 15.29],
        },
        "IsaacReadTimes": read_times,
    }


def _fixture() -> tuple[dict[str, Any], Any, FakeRegistry, FakeOrchestrator, Any]:
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
    registry = FakeRegistry(state)
    rep = SimpleNamespace(AnnotatorRegistry=registry)
    orchestrator = FakeOrchestrator(state)
    active_base = SimpleNamespace(sim=SimpleNamespace(current_time=state["simulation_time"]))
    return state, rep, registry, orchestrator, active_base


def _collect(**changes: Any) -> tuple[dict[str, Any], dict[str, Any], FakeRegistry, FakeOrchestrator]:
    state, rep, registry, orchestrator, active_base = _fixture()
    state.update(changes)
    profile = RECORD.fixed_profile()["camera"]

    def prepare(ordinal: int, camera_kind: str, marker_code: int) -> dict[str, Any]:
        state.update(ordinal=ordinal, camera_kind=camera_kind, code=marker_code)
        return {
            "ordinal": ordinal,
            "camera_kind": camera_kind,
            "marker_code": marker_code,
            "marker_labels": sorted(FRAME.marker_labels(marker_code)),
            "camera": {
                "eye_world_m": [1.0 if camera_kind == "A" else -1.0, 2.0, 3.0],
                "lookat_world_m": [0.0, 0.0, 0.0],
                "resolution": [6, 6],
            },
            "marker_receipt": {"semantic_attribute_written": True},
        }

    report = COLLECTOR.collect_aaba(
        rep=rep,
        render_product="/Render/Products/RP",
        orchestrator=orchestrator,
        camera_profile=profile,
        active_base=active_base,
        physics_snapshot=lambda: {"root_position": [1.0, 2.0, 3.0]},
        prepare_arm=prepare,
        app_update_count=lambda: state["updates"],
    )
    return report, state, registry, orchestrator


def test_collect_and_validate_aaba_uses_actual_sdk_shape_and_camera_profile() -> None:
    report, state, registry, orchestrator = _collect()
    validated = VALIDATOR.validate(report)

    assert validated["approved"] is True
    assert validated["camera_sequence"] == ["A", "A", "B", "A"]
    assert validated["marker_codes"] == [0, 1, 2, 3]
    assert validated["frame_numbers"] == [58, 59, 60, 61]
    assert [
        capture["render_window_app_updates"] for capture in report["captures"]
    ] == [2, 2, 2, 2]
    assert [capture["pull_window_app_updates"] for capture in report["captures"]] == [
        0,
        0,
        0,
        0,
    ]
    assert orchestrator.steps == [
        {"rt_subframes": 1, "pause_timeline": False, "delta_time": 0.0}
    ] * 4
    assert registry.requests == [
        ("rgb", {}),
        ("semantic_segmentation", {"init_params": {"colorize": False}}),
        ("CameraParams", {}),
        ("IsaacReadTimes", {}),
    ]
    for annotator in registry.annotators.values():
        assert annotator.calls == [{"do_array_copy": True}] * 4
        assert annotator.attached == ["/Render/Products/RP"]
        assert annotator.detached == ["/Render/Products/RP"]
    expected = []
    for ordinal in range(4):
        expected.append(("step", ordinal))
        expected.extend(("pull", ordinal, name) for name in COLLECTOR.ANNOTATORS)
    assert [event for event in state["events"] if event[0] in {"step", "pull"}] == expected
    source = state["payloads"]["rgb"]
    captured = report["captures"][-1]["raw"]["rgb"]
    source[0, 0, 0] = 255
    assert captured[0, 0, 0] != 255
    assert "callbacks" not in report


def test_validator_rejects_stale_semantic_without_dropping_raw_capture() -> None:
    report, _, _, _ = _collect()
    stale = deepcopy(report["captures"][0]["raw"]["semantic_segmentation"])
    report["captures"][1]["raw"]["semantic_segmentation"] = stale

    with pytest.raises(ValueError, match="expected code"):
        VALIDATOR.validate(report)

    assert len(report["captures"]) == 4
    assert report["captures"][1]["raw"]["semantic_segmentation"] is stale


@pytest.mark.parametrize("mutation", ["physics", "time", "update", "camera", "frame"])
def test_validator_rejects_changed_receipts(mutation: str) -> None:
    report, _, _, _ = _collect()
    capture = report["captures"][2]
    if mutation == "physics":
        capture["physics_after"]["snapshot"]["root_position"][0] = 9.0
    elif mutation == "time":
        capture["raw"]["IsaacReadTimes"]["simulationTime"] = 0.02
    elif mutation == "update":
        capture["app_update_counts"]["after_pull"] += 1
        capture["pull_window_app_updates"] = 1
    elif mutation == "camera":
        capture["raw"]["CameraParams"]["cameraViewTransform"] = _view("A")
    else:
        capture["raw"]["IsaacReadTimes"]["frameNumber"] = 59

    with pytest.raises((TypeError, ValueError)):
        VALIDATOR.validate(report)


def test_render_may_advance_multiple_frames_but_pull_window_must_not_update() -> None:
    report, _, _, _ = _collect(frame_increment=2)

    validated = VALIDATOR.validate(report)

    assert validated["frame_numbers"] == [59, 61, 63, 65]
    assert all(capture["step"]["call_count"] == 1 for capture in report["captures"])


def test_validator_rejects_app_update_during_sequential_pulls() -> None:
    state, rep, _, orchestrator, active_base = _fixture()
    state["update_during_pull"] = (2, "CameraParams")
    profile = RECORD.fixed_profile()["camera"]

    def prepare(ordinal: int, camera_kind: str, marker_code: int) -> dict[str, Any]:
        state.update(ordinal=ordinal, camera_kind=camera_kind, code=marker_code)
        return {
            "ordinal": ordinal,
            "camera_kind": camera_kind,
            "marker_code": marker_code,
            "marker_labels": sorted(FRAME.marker_labels(marker_code)),
            "camera": {
                "eye_world_m": [1.0 if camera_kind == "A" else -1.0, 2.0, 3.0],
                "lookat_world_m": [0.0, 0.0, 0.0],
                "resolution": [6, 6],
            },
        }

    report = COLLECTOR.collect_aaba(
        rep=rep,
        render_product="/Render/Products/RP",
        orchestrator=orchestrator,
        camera_profile=profile,
        active_base=active_base,
        physics_snapshot=lambda: {"root_position": [1.0, 2.0, 3.0]},
        prepare_arm=prepare,
        app_update_count=lambda: state["updates"],
    )

    assert report["captures"][2]["pull_window_app_updates"] == 1
    with pytest.raises(ValueError, match="연속 annotator pull"):
        VALIDATOR.validate(report)


def test_pull_failure_preserves_first_error_and_detaches_all_annotators() -> None:
    state, rep, registry, orchestrator, active_base = _fixture()
    state["fail_pull"] = (1, "CameraParams")
    profile = RECORD.fixed_profile()["camera"]

    def prepare(ordinal: int, camera_kind: str, marker_code: int) -> dict[str, Any]:
        state.update(ordinal=ordinal, camera_kind=camera_kind, code=marker_code)
        return {
            "ordinal": ordinal,
            "camera_kind": camera_kind,
            "marker_code": marker_code,
            "marker_labels": sorted(FRAME.marker_labels(marker_code)),
            "camera": {
                "eye_world_m": [1.0 if camera_kind == "A" else -1.0, 2.0, 3.0],
                "lookat_world_m": [0.0, 0.0, 0.0],
                "resolution": [6, 6],
            },
        }

    with pytest.raises(COLLECTOR.AnnotatorPullFailure) as caught:
        COLLECTOR.collect_aaba(
            rep=rep,
            render_product="/Render/Products/RP",
            orchestrator=orchestrator,
            camera_profile=profile,
            active_base=active_base,
            physics_snapshot=lambda: {"root_position": [1.0, 2.0, 3.0]},
            prepare_arm=prepare,
            app_update_count=lambda: state["updates"],
        )

    assert isinstance(caught.value.primary, LookupError)
    assert caught.value.report["status"] == "failed"
    assert caught.value.report["capture_count"] == 1
    assert caught.value.report["partial_capture"]["ordinal"] == 1
    assert set(caught.value.report["partial_capture"]["raw"]) == {
        "rgb",
        "semantic_segmentation",
    }
    assert caught.value.report["cleanup_errors"] == []
    assert len(orchestrator.steps) == 2
    assert all(item.detached == ["/Render/Products/RP"] for item in registry.annotators.values())
