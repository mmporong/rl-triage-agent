from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
P = importlib.import_module("vision_writer_payload_v1")
R = importlib.import_module("record_vision_evidence_v1")


@pytest.fixture(autouse=True)
def _fixed_test_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(R, "WIDTH", 4)
    monkeypatch.setattr(R, "HEIGHT", 3)


def payload() -> dict[str, Any]:
    rp = "RenderProduct_0"
    rgba = np.zeros((3, 4, 4), dtype=np.uint8)
    mask = np.ones((3, 4), dtype=np.uint32)
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
        "swhFrameNumber": 0,
        "reference_time": (55, 30),
        "distribution_outputs": {},
        "trigger_outputs": {},
        "named_outputs": {},
        "annotators": {
            "camera": {rp: R.fixed_profile()["camera"]["camera_prim_path"]},
            "resolution": {rp: np.array([4, 3], dtype=np.int32)},
            "rgb": {rp: {"data": rgba}},
            "semantic_segmentation": {
                rp: {"data": mask, "idToLabels": {"1": {"class": "secret-label"}}}
            },
            "CameraParams": {
                rp: {
                    "cameraViewTransform": np.arange(16, dtype=np.float64),
                    "cameraProjection": np.eye(4).reshape(16),
                    "renderProductResolution": np.array([4, 3], dtype=np.int32),
                }
            },
            "IsaacReadTimes": {rp: times},
        },
    }


@pytest.mark.parametrize("legacy", ["rgb", "semantic"])
def test_decoder_rejects_legacy_annotator_shapes(legacy: str) -> None:
    raw = payload()
    if legacy == "rgb":
        raw["annotators"]["rgb"]["RenderProduct_0"] = np.zeros(
            (3, 4, 4), dtype=np.uint8
        )
    else:
        semantic = raw["annotators"]["semantic_segmentation"]["RenderProduct_0"]
        raw["annotators"]["semantic_segmentation"]["RenderProduct_0"] = {
            "data": semantic["data"],
            "info": {"idToLabels": semantic["idToLabels"]},
        }
    with pytest.raises(ValueError, match="nonlegacy"):
        P.decode_writer_payload(raw)


@pytest.mark.parametrize(
    "mutation", ["none", "missing", "unknown", "integer_id", "invalid_label"]
)
def test_nonlegacy_data_and_metadata_are_exact(mutation: str) -> None:
    raw = payload()
    rgb = raw["annotators"]["rgb"]["RenderProduct_0"]
    semantic = raw["annotators"]["semantic_segmentation"]["RenderProduct_0"]
    if mutation == "none":
        rgb["data"] = None
    elif mutation == "missing":
        semantic.pop("idToLabels")
    elif mutation == "unknown":
        semantic["unknown"] = None
    elif mutation == "integer_id":
        semantic["idToLabels"] = {1: {"class": "label"}}
    else:
        semantic["idToLabels"] = {"1": {"class": None}}
    descriptor = P.describe_writer_payload(raw)
    if mutation == "none":
        described = descriptor["annotators"]["rgb"]["render_products"][
            "RenderProduct_0"
        ]["data"]
        assert described["is_none"] is True
        assert described["dtype"] is None and described["shape"] is None
    with pytest.raises((TypeError, ValueError)):
        P.decode_writer_payload(raw)


def test_descriptor_has_structure_without_pixel_or_label_bodies() -> None:
    descriptor = P.describe_writer_payload(payload())
    rgb = descriptor["annotators"]["rgb"]["render_products"]["RenderProduct_0"]
    semantic = descriptor["annotators"]["semantic_segmentation"]["render_products"][
        "RenderProduct_0"
    ]
    assert descriptor["schema"] == "vision_writer_payload_descriptor_v1"
    assert rgb["mapping_keys"] == ["data"]
    assert rgb["data"] == {
        "type": "numpy.ndarray",
        "is_none": False,
        "dtype": "uint8",
        "shape": [3, 4, 4],
        "device": None,
    }
    assert semantic["mapping_keys"] == ["data", "idToLabels"]
    assert semantic["fields"]["idToLabels"] == {
        "type": "builtins.dict",
        "is_none": False,
    }
    assert "secret-label" not in json.dumps(descriptor, sort_keys=True)


def test_descriptor_and_writer_never_request_array_conversion() -> None:
    class ArrayTrap:
        dtype = "uint8"
        shape = (3, 4, 4)
        device = "cpu"

        def __array__(self) -> np.ndarray:
            raise AssertionError("descriptor must not request array conversion")

    class Writer:
        pass

    class Registry:
        @staticmethod
        def get_annotator(name: str, **kwargs: Any) -> str:
            return f"{name}:{kwargs}"

    raw = payload()
    raw["annotators"]["rgb"]["RenderProduct_0"]["data"] = ArrayTrap()
    descriptor = P.describe_writer_payload(raw)
    data = descriptor["annotators"]["rgb"]["render_products"]["RenderProduct_0"]["data"]
    assert data["shape"] == [3, 4, 4] and data["device"] == "cpu"
    writer = P.create_payload_writer(
        SimpleNamespace(Writer=Writer, AnnotatorRegistry=Registry), lambda _: None
    )
    writer.write(raw)
    original_error = writer.callback_error
    original_descriptor = writer.last_payload_descriptor
    assert isinstance(original_error, TypeError)
    assert original_descriptor == descriptor
    writer.write({"different": None})
    assert writer.callback_error is original_error
    assert writer.last_payload_descriptor is original_descriptor
    assert writer.callback_count == 2
