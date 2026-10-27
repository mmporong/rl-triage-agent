from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import record_vision_evidence_v1 as RECORD  # pyright: ignore[reportMissingImports]
import vision_pull_record_capture_v1 as CAPTURE  # pyright: ignore[reportMissingImports]


class _Annotator:
    def __init__(self, name: str, state: dict[str, Any]) -> None:
        self.name = name
        self.state = state

    def attach(self, render_product: Any) -> None:
        self.state["events"].append(("attach", self.name, render_product))

    def get_data(self, **kwargs: Any) -> Any:
        assert kwargs == {"device": "cpu", "do_array_copy": True}
        self.state["events"].append(("pull", self.state["ordinal"], self.name))
        if self.state.get("update_during_pull") == self.name:
            self.state["updates"] += 1
        return self.state["payloads"][self.name]

    def detach(self, render_product: Any) -> None:
        self.state["events"].append(("detach", self.name, render_product))
        if self.state.get("detach_failure") == self.name:
            raise RuntimeError(f"detach failed: {self.name}")


class _Registry:
    def __init__(self, state: dict[str, Any]) -> None:
        self.state = state
        self.items: dict[str, _Annotator] = {}

    def get_annotator(self, name: str, **kwargs: Any) -> _Annotator:
        if name == "semantic_segmentation":
            assert kwargs == {"init_params": {"colorize": False}}
        else:
            assert kwargs == {}
        item = _Annotator(name, self.state)
        self.items[name] = item
        return item


class _ReferenceRegistry:
    def __init__(self, state: dict[str, Any], items: dict[str, _Annotator]) -> None:
        self.state, self.items = state, items

    def get(self, name: str) -> _Annotator:
        assert name == "ReferenceTime"
        item = _Annotator(name, self.state)
        self.items[name] = item
        return item


class _Orchestrator:
    def __init__(self, state: dict[str, Any]) -> None:
        self.state = state

    def step(self, **kwargs: Any) -> None:
        assert kwargs == {"rt_subframes": 1, "pause_timeline": False, "delta_time": 0.0}
        self.state["events"].append(("step", self.state["ordinal"]))
        self.state["updates"] += 2
        if not self.state.get("stale_frame"):
            self.state["frame"] += 1
        self.state["payloads"] = _payloads(self.state)


class _NativeApp:
    def __init__(self, state: dict[str, Any]) -> None:
        self.state = state

    def get_update_number(self) -> int:
        return int(self.state["updates"])


class _Sim:
    current_time = 0.01

    def __init__(self, state: dict[str, Any]) -> None:
        self.state = state

    def set_camera_view(self, **kwargs: Any) -> None:
        assert kwargs["camera_prim_path"]
        self.state["events"].append(("camera", self.state["ordinal"]))


class _Context:
    def __init__(self, state: dict[str, Any]) -> None:
        self.state = state

    def reset_renderer_accumulation(self) -> None:
        self.state["events"].append(("reset", self.state["ordinal"]))


def _payloads(state: dict[str, Any]) -> dict[str, Any]:
    frame = state["frame"]
    rgba = np.zeros((3, 4, 4), dtype=np.uint8)
    rgba[:, :, :3] = frame % 255
    rgba[:, :, 3] = 255
    mask = np.zeros((3, 4), dtype=np.uint32)
    read_times = {
        "durationDenominator": 30,
        "durationNumerator": 1,
        "externalTimeOfSimNs": frame * 33_333_333,
        "frameNumber": frame,
        "rationalTimeOfSimDenominator": 30,
        "rationalTimeOfSimNumerator": frame,
        "sampleTimeOffsetInSimFrames": 0,
        "simulationTime": 0.01,
        "simulationTimeMonotonic": 0.01,
        "swhFrameNumber": frame,
        "systemTime": 1000.0 + frame,
    }
    assert set(read_times) == set(RECORD.ISAAC_READ_TIMES_FIELDS)
    return {
        "rgb": rgba,
        "semantic_segmentation": {
            "data": mask,
            "info": {"idToLabels": {"0": {"class": "robot"}}},
        },
        "CameraParams": {
            "cameraViewTransform": np.eye(4),
            "cameraProjection": np.eye(4),
            "renderProductResolution": [4, 3],
            "cameraAperture": [20.955, 15.29],
        },
        "IsaacReadTimes": read_times,
        "ReferenceTime": {
            "referenceTimeNumerator": frame,
            "referenceTimeDenominator": 30,
        },
    }


def _setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, use_fxaa: bool = False
) -> tuple[Any, Any, dict[str, Any], Any, Any]:
    monkeypatch.setattr(CAPTURE.R, "WIDTH", 4)
    monkeypatch.setattr(CAPTURE.R, "HEIGHT", 3)
    job_path = tmp_path / "job.json"
    job_path.write_text("{}", encoding="utf-8")
    validated = {"job": {"mode": RECORD.RECORD_MODE}, "job_path": job_path}
    state: dict[str, Any] = {
        "ordinal": 0,
        "frame": 57,
        "updates": 10,
        "events": [],
        "payloads": {},
    }
    registry = _Registry(state)
    rep = SimpleNamespace(
        AnnotatorRegistry=registry,
        annotators=_ReferenceRegistry(state, registry.items),
        orchestrator=_Orchestrator(state),
    )
    base = SimpleNamespace(sim=_Sim(state))
    context = _Context(state)

    def physics(_base: Any) -> dict[str, Any]:
        suffix = state["updates"] if state.get("physics_changes") else 0
        return {
            "simulation_time_s": 0.01,
            "common_step_counter": 7,
            "root_position_world_m": [float(suffix), 0.0, 0.0],
        }

    deps = {
        "usd_context": lambda: (context, object()),
        "settings": lambda: object(),
        "readiness": lambda *_args: None,
        "renderer_readback": lambda *_args: {
            "renderer_accumulation_reset": "omni.usd.UsdContext.reset_renderer_accumulation"
        },
        "physics_snapshot": physics,
        "native_app": lambda: _NativeApp(state),
        "fxaa_required": lambda _job: use_fxaa,
        "fxaa_apply": lambda *_args: {"selected": "fxaa"},
        "fxaa_readback": lambda _settings: {"observed": "fxaa"},
        "fxaa_verify": lambda value: state["events"].append(("fxaa_verify", value)),
    }
    factory = CAPTURE.make_capture_factory(validated, object(), dependencies=deps)
    capture, cleanup = factory(base, rep, "/Render/Products/RP", RECORD.fixed_profile()["camera"])
    return capture, cleanup, state, base, rep


def _capture(capture: Any, state: dict[str, Any], base: Any, rep: Any, ordinal: int) -> Any:
    state["ordinal"] = ordinal
    return capture(
        base,
        rep,
        [],
        RECORD.fixed_profile()["camera"],
        [3.0, 3.0, 2.0],
        [0.0, 0.0, 0.35],
        ordinal,
    )


def test_factory_pulls_five_actual_sources_for_two_strict_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture, cleanup, state, base, rep = _setup(tmp_path, monkeypatch)

    first = _capture(capture, state, base, rep, 0)
    second = _capture(capture, state, base, rep, 1)
    cleanup()

    assert len(first["rgb"]) == 4 * 3 * 3
    assert first["read_times"]["frameNumber"] == 58
    assert second["read_times"]["frameNumber"] == 59
    assert first["reference_time"] == {
        "referenceTimeNumerator": 58,
        "referenceTimeDenominator": 30,
    }
    assert first["pull_npz"].is_file()
    assert first["pull_raw_receipt"].is_file()
    receipt = json.loads(first["pull_receipt"].read_text(encoding="utf-8"))
    assert receipt["annotator_order"] == list(CAPTURE.ANNOTATORS)
    assert receipt["app_update_counter"]["render_window_updates"] == 2
    assert receipt["app_update_counter"]["pull_window_updates"] == 0
    assert receipt["atomicity_claimed"] is False
    assert receipt["capture_approved"] is False
    assert receipt["used_for_decision"] is False
    expected = []
    for ordinal in range(2):
        expected.extend(
            [("camera", ordinal), ("reset", ordinal), ("step", ordinal)]
        )
        expected.extend(("pull", ordinal, name) for name in CAPTURE.ANNOTATORS)
    assert [row for row in state["events"] if row[0] in {"camera", "reset", "step", "pull"}] == expected
    assert [row[1] for row in state["events"] if row[0] == "detach"] == list(
        reversed(CAPTURE.ANNOTATORS)
    )


def test_pull_window_native_update_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture, cleanup, state, base, rep = _setup(tmp_path, monkeypatch)
    state["update_during_pull"] = "CameraParams"

    with pytest.raises(RuntimeError, match="연속 annotator pull"):
        _capture(capture, state, base, rep, 0)
    cleanup()


def test_renderer_and_fxaa_readback_are_retained(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture, cleanup, state, base, rep = _setup(
        tmp_path, monkeypatch, use_fxaa=True
    )

    result = _capture(capture, state, base, rep, 0)
    cleanup()

    receipt = json.loads(result["pull_receipt"].read_text(encoding="utf-8"))
    assert receipt["renderer"]["render_policy"] == {"selected": "fxaa"}
    assert receipt["renderer"]["fxaa_readback"] == {"observed": "fxaa"}
    assert receipt["fxaa_readback_verified"] is True
    assert ("fxaa_verify", {"observed": "fxaa"}) in state["events"]


@pytest.mark.parametrize("failure", ["missing_reference", "bad_reference_type"])
def test_missing_or_wrong_type_actual_reference_time_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    capture, cleanup, state, base, rep = _setup(tmp_path, monkeypatch)
    original_step = rep.orchestrator.step

    def malformed_step(**kwargs: Any) -> None:
        original_step(**kwargs)
        if failure == "missing_reference":
            state["payloads"].pop("ReferenceTime")
        else:
            state["payloads"]["ReferenceTime"]["referenceTimeNumerator"] = 58.0

    rep.orchestrator.step = malformed_step
    with pytest.raises((KeyError, TypeError, ValueError)):
        _capture(capture, state, base, rep, 0)
    cleanup()


@pytest.mark.parametrize("value", [True, -1, 1.5])
def test_native_counter_wrong_type_or_range_is_rejected(value: Any) -> None:
    with pytest.raises(RuntimeError, match="update number"):
        CAPTURE._counter(SimpleNamespace(get_update_number=lambda: value))


@pytest.mark.parametrize("failure", ["stale_frame", "physics_changes"])
def test_stale_timing_and_physics_change_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    capture, cleanup, state, base, rep = _setup(tmp_path, monkeypatch)
    _capture(capture, state, base, rep, 0)
    state[failure] = True

    with pytest.raises((RuntimeError, ValueError)):
        _capture(capture, state, base, rep, 1)
    cleanup()


def test_cleanup_attempts_reverse_detach_and_surfaces_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, cleanup, state, _, _ = _setup(tmp_path, monkeypatch)
    state["detach_failure"] = "CameraParams"

    with pytest.raises(RuntimeError, match="detach failed: CameraParams"):
        cleanup()

    assert [row[1] for row in state["events"] if row[0] == "detach"] == list(
        reversed(CAPTURE.ANNOTATORS)
    )
