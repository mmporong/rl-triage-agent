from __future__ import annotations

import copy
import importlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
F = importlib.import_module("vision_callback_fingerprint_v1")
R = importlib.import_module("record_vision_evidence_v1")


@pytest.fixture(autouse=True)
def _fixed_test_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(R, "WIDTH", 4)
    monkeypatch.setattr(R, "HEIGHT", 3)


def _times(frame: int = 33) -> dict[str, Any]:
    value: dict[str, Any] = {
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
        "systemTime": 1000.0,
    }
    assert set(value) == set(R.ISAAC_READ_TIMES_FIELDS)
    return value


def _payload(*, rp: str = "RenderProduct_0") -> dict[str, Any]:
    rgba = np.zeros((3, 4, 4), dtype=np.uint8)
    rgba[:, :, :3] = (5, 6, 7)
    rgba[:, :, 3] = 255
    mask = np.ones((3, 4), dtype=np.uint32)
    camera = {
        "cameraViewTransform": np.arange(16, dtype=np.float64),
        "cameraProjection": np.arange(16, dtype=np.float32) + 1,
        "renderProductResolution": np.array([4, 3], dtype=np.int32),
        "cameraFocalLength": np.float32(24.0),
    }
    return {
        "swhFrameNumber": 77,
        "reference_time": (33, 30),
        "distribution_outputs": {"distribution": np.uint64(2**63 + 7)},
        "trigger_outputs": {"armed": True},
        "named_outputs": {"name": "diagnostic"},
        "annotators": {
            "camera": {rp: R.fixed_profile()["camera"]["camera_prim_path"]},
            "resolution": {rp: np.array([4, 3], dtype=np.int32)},
            "rgb": {rp: {"data": rgba}},
            "semantic_segmentation": {
                rp: {"data": mask, "idToLabels": {"1": {"class": "robot/body"}}}
            },
            "CameraParams": {rp: camera},
            "IsaacReadTimes": {rp: _times()},
        },
    }


def _changed(
    payload: dict[str, Any], mutate: Any
) -> tuple[dict[str, Any], dict[str, Any]]:
    before = F.fingerprint(payload)
    changed = copy.deepcopy(payload)
    mutate(changed)
    return before, F.fingerprint(changed)


def test_identical_frame_33_payloads_have_same_whole_and_component_hashes() -> None:
    first = F.fingerprint(_payload())
    second = F.fingerprint(copy.deepcopy(_payload()))
    assert first == second
    assert first["schema"] == F.SCHEMA
    assert first["no_capture_approved"] is True
    assert set(first["components"]) == {
        "distribution_outputs",
        "trigger_outputs",
        "named_outputs",
        "reference_time",
        "writer_swh_frame_number",
        "render_product_identity",
        "camera_identity",
        "resolution",
        "rgb_data",
        "semantic_segmentation",
        "semantic_mask",
        "semantic_labels",
        "camera_params",
        "isaac_read_times",
    }
    rgb = first["components"]["rgb_data"]["metadata"]
    assert rgb == {
        "type": "numpy.ndarray",
        "dtype": "|u1",
        "shape": [3, 4, 4],
        "nbytes": 48,
    }
    serialized = json.dumps(first, sort_keys=True)
    assert "robot/body" not in serialized


@pytest.mark.parametrize(
    ("component", "mutate"),
    [
        (
            "rgb_data",
            lambda value: value["annotators"]["rgb"]["RenderProduct_0"][
                "data"
            ].__setitem__((0, 0, 0), 4),
        ),
        (
            "semantic_labels",
            lambda value: value["annotators"]["semantic_segmentation"][
                "RenderProduct_0"
            ]["idToLabels"]["1"].__setitem__("class", "robot/arm"),
        ),
        (
            "camera_params",
            lambda value: value["annotators"]["CameraParams"][
                "RenderProduct_0"
            ].__setitem__("cameraFocalLength", np.float32(25.0)),
        ),
        (
            "isaac_read_times",
            lambda value: value["annotators"]["IsaacReadTimes"][
                "RenderProduct_0"
            ].__setitem__("systemTime", 1001.0),
        ),
        (
            "trigger_outputs",
            lambda value: value["trigger_outputs"].__setitem__("armed", False),
        ),
        (
            "camera_identity",
            lambda value: value["annotators"]["camera"].__setitem__(
                "RenderProduct_0", "/World/OtherCamera"
            ),
        ),
    ],
)
def test_component_hash_separates_each_raw_change(component: str, mutate: Any) -> None:
    before, after = _changed(_payload(), mutate)
    assert before["whole_payload_sha256"] != after["whole_payload_sha256"]
    assert (
        before["components"][component]["sha256"]
        != after["components"][component]["sha256"]
    )


def test_render_product_identity_change_is_preserved() -> None:
    before = F.fingerprint(_payload(rp="RenderProduct_0"))
    after = F.fingerprint(_payload(rp="RenderProduct_1"))
    assert (
        before["components"]["render_product_identity"]["sha256"]
        != after["components"]["render_product_identity"]["sha256"]
    )


def test_mapping_keys_and_scalar_types_are_not_coerced() -> None:
    hashes = []
    for trigger in (
        {"1": "x"},
        {1: "x"},
        {"value": True},
        {"value": 1},
        {"value": 1.0},
    ):
        payload = _payload()
        payload["trigger_outputs"] = trigger
        hashes.append(F.fingerprint(payload)["components"]["trigger_outputs"]["sha256"])
    assert len(set(hashes)) == len(hashes)
    numpy_payload = _payload()
    numpy_payload["trigger_outputs"] = {"value": np.uint64(2**64 - 1)}
    python_payload = _payload()
    python_payload["trigger_outputs"] = {"value": 2**64 - 1}
    assert (
        F.fingerprint(numpy_payload)["whole_payload_sha256"]
        != F.fingerprint(python_payload)["whole_payload_sha256"]
    )


def test_hash_is_snapshot_of_bytes_before_later_mutation() -> None:
    payload = _payload()
    frozen = F.fingerprint(payload)
    payload["annotators"]["rgb"]["RenderProduct_0"]["data"][0, 0, 0] = 99
    assert (
        frozen["whole_payload_sha256"] != F.fingerprint(payload)["whole_payload_sha256"]
    )


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), object()])
def test_nonfinite_and_unsupported_values_fail_closed(bad: object) -> None:
    payload = _payload()
    payload["trigger_outputs"] = {"bad": bad}
    with pytest.raises((TypeError, ValueError)):
        F.fingerprint(payload)


def test_cycle_and_cap_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _payload()
    cycle: dict[str, Any] = {}
    cycle["self"] = cycle
    payload["trigger_outputs"] = cycle
    with pytest.raises(ValueError, match="cycle"):
        F.fingerprint(payload)
    monkeypatch.setattr(F, "MAX_CANONICAL_BYTES", 32)
    with pytest.raises(ValueError, match="byte 상한"):
        F.fingerprint(_payload())


def test_buffer_cap_is_checked_before_ndarray_tobytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SpyArray(np.ndarray):
        tobytes_called = False

        def tobytes(self, *args: Any, **kwargs: Any) -> bytes:
            type(self).tobytes_called = True
            return super().tobytes(*args, **kwargs)

    payload = _payload()
    spy = payload["annotators"]["rgb"]["RenderProduct_0"]["data"].view(SpyArray)
    payload["annotators"]["rgb"]["RenderProduct_0"]["data"] = spy
    monkeypatch.setattr(F, "MAX_CANONICAL_BYTES", 32)
    with pytest.raises(ValueError, match="byte 상한"):
        F.fingerprint(payload)
    assert SpyArray.tobytes_called is False


def test_buffer_cap_is_checked_before_numpy_scalar_tobytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SpyString(np.str_):
        tobytes_called = False

        def tobytes(self, *args: Any, **kwargs: Any) -> bytes:
            type(self).tobytes_called = True
            return super().tobytes(*args, **kwargs)

    key = SpyString("large-key")
    monkeypatch.setattr(F, "MAX_CANONICAL_BYTES", 16)
    with pytest.raises(ValueError, match="상한"):
        F._canonical({key: None})
    assert SpyString.tobytes_called is False

    value = SpyString("x" * 100)
    monkeypatch.setattr(F, "MAX_CANONICAL_BYTES", 128)
    with pytest.raises(ValueError, match="byte 상한"):
        F._canonical(value)
    assert SpyString.tobytes_called is False


def test_string_cap_is_checked_from_character_count_before_encoding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(F, "MAX_CANONICAL_BYTES", 8)
    with pytest.raises(ValueError, match="byte 상한"):
        F._canonical("한글문자열")


def test_mapping_keys_share_global_node_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(F, "MAX_CANONICAL_NODES", 4)
    with pytest.raises(ValueError, match="node 상한"):
        F._canonical({(1, 2): None, (3, 4): None})


def test_canonical_result_uses_zero_copy_view() -> None:
    canonical = F._canonical({"value": 1})
    assert isinstance(canonical, memoryview)


def test_empty_semantic_and_missing_schema_key_never_fingerprint_as_valid() -> None:
    empty = _payload()
    empty["annotators"]["semantic_segmentation"]["RenderProduct_0"]["data"] = np.array(
        [], dtype=np.float64
    )
    with pytest.raises(ValueError, match="semantic mask"):
        F.fingerprint(empty)
    missing = _payload()
    missing.pop("named_outputs")
    with pytest.raises(ValueError, match="top-level"):
        F.fingerprint(missing)
