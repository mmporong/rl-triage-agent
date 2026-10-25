from __future__ import annotations

import importlib
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
S = importlib.import_module("vision_frame_sync_v1")


def _times(frame: int, simulation_time: float = 2.0) -> dict[str, Any]:
    return {
        "durationDenominator": 60,
        "durationNumerator": 1,
        "externalTimeOfSimNs": 2_000_000_000,
        "frameNumber": frame,
        "rationalTimeOfSimDenominator": 30,
        "rationalTimeOfSimNumerator": frame,
        "sampleTimeOffsetInSimFrames": 0,
        "simulationTime": simulation_time,
        "simulationTimeMonotonic": simulation_time,
        "swhFrameNumber": 0,
        "systemTime": 1_000.0 + frame,
    }


def _view(kind: str) -> np.ndarray:
    if kind == "A":
        return np.array(
            [[1, 2, 3, 4], [5, 7, 11, 13], [17, 19, 23, 29], [31, 37, 41, 43]],
            dtype=float,
        )
    return np.array(
        [[2, 3, 5, 7], [11, 13, 17, 19], [23, 29, 31, 37], [41, 43, 47, 53]],
        dtype=float,
    )


def _marker_arrays(
    code: int, id_base: int = 10, background: int = 0
) -> tuple[np.ndarray, dict[str, Any]]:
    rgb = np.full((10, 10, 3), background, dtype=np.uint8)
    mask = np.zeros((10, 10), dtype=np.int32)
    labels = sorted(S.marker_labels(code))
    mapping = {}
    for index, label in enumerate(labels):
        runtime_id = id_base + index
        row = index
        mask[row, 1:9] = runtime_id
        mapping[runtime_id] = {"class": label}
        role = "sentinel" if label == S.SENTINEL else label.rsplit("/", 1)[-1]
        rgb[row, 1:9] = S.PALETTE[role]
    return rgb, {"mask": mask, "idToLabels": mapping}


def _payload(
    ordinal: int,
    *,
    code: int | None = None,
    id_base: int = 10,
    view: str | None = None,
    background: int = 0,
    marker: bool = True,
) -> dict[str, Any]:
    callback_ref = f"callback-{ordinal}"
    if marker:
        rgb, semantic = _marker_arrays(
            ordinal if code is None else code, id_base, background
        )
    else:
        rgb = np.full((10, 10, 3), background, dtype=np.uint8)
        semantic = {
            "mask": np.zeros((10, 10), dtype=np.int32),
            "idToLabels": {0: {"class": "robot/body"}},
        }
    camera = {
        "view_matrix": _view(view or ("B" if ordinal == 2 else "A")),
        "projection": np.eye(4),
        "resolution": [10, 10],
    }
    values = {
        "rgb": rgb,
        "semantic": semantic,
        "camera": camera,
        "read_times": _times(55 + ordinal),
        "reference_time": {
            "referenceTimeNumerator": 55 + ordinal,
            "referenceTimeDenominator": 30,
        },
    }
    return {
        "adapter_callback_ref": callback_ref,
        "render_product_ref": "render-product-0",
        **{
            name: {"adapter_callback_ref": callback_ref, "data": value}
            for name, value in values.items()
        },
    }


def _capture(
    writer: Any, ordinal: int, payload: dict[str, Any], code: int | None = None
) -> dict[str, Any]:
    state = {"joint": [0.1, 0.2], "step": 7}
    writer.arm(
        ordinal,
        ordinal if code is None and writer.mode == S.SMOKE_MODE else code,
        state,
        2.0,
    )
    writer.before_step(ordinal)
    writer.callback(payload)
    return writer.close(ordinal, state, 2.0)


def _valid_smoke() -> tuple[Any, list[dict[str, Any]]]:
    writer = S.AtomicFrameWriter(S.SMOKE_MODE, 4)
    rows = [
        _capture(
            writer,
            ordinal,
            _payload(
                ordinal,
                id_base=10 + ordinal * 20,
                background=ordinal * 3,
            ),
        )
        for ordinal in range(4)
    ]
    return writer, rows


def test_valid_atomic_smoke_allows_background_noise_and_runtime_id_changes() -> None:
    writer, rows = _valid_smoke()
    assert writer.finish() == rows
    assert [row["decoded_code"] for row in rows] == [0, 1, 2, 3]
    assert len({tuple(row["id_to_labels"]) for row in rows}) == 4
    assert rows[0]["camera"]["view_matrix"] == _view("A").tolist()
    assert rows[2]["camera"]["view_matrix"] == _view("B").tolist()


def test_channel_order_allows_brightness_change_but_rejects_ties() -> None:
    payload = _payload(0)
    semantic = payload["semantic"]["data"]
    rgb = payload["rgb"]["data"]
    colors = {"sentinel": (5, 90, 50), "active": (90, 5, 50), "inactive": (5, 50, 90)}
    for runtime_id, descriptor in semantic["idToLabels"].items():
        label = descriptor["class"]
        role = "sentinel" if label == S.SENTINEL else label.rsplit("/", 1)[-1]
        rgb[semantic["mask"] == runtime_id] = colors[role]
    writer = S.AtomicFrameWriter(S.SMOKE_MODE, 4)
    _capture(writer, 0, payload)

    tied = _payload(0)
    tied["rgb"]["data"][0, 1:9] = (20, 20, 20)
    writer = S.AtomicFrameWriter(S.SMOKE_MODE, 4)
    with pytest.raises(ValueError, match="channel order"):
        _capture(writer, 0, tied)


def test_hidden_marker_metadata_is_allowed_but_hidden_variant_pixels_are_not() -> None:
    hidden_label = f"{S.MARKER_PREFIX}bit0/rail1/active"
    payload = _payload(0)
    payload["semantic"]["data"]["idToLabels"][999] = {"class": hidden_label}
    writer = S.AtomicFrameWriter(S.SMOKE_MODE, 4)
    _capture(writer, 0, payload)

    visible = _payload(0)
    visible["semantic"]["data"]["idToLabels"][999] = {"class": hidden_label}
    visible["semantic"]["data"]["mask"][9, 9] = 999
    writer = S.AtomicFrameWriter(S.SMOKE_MODE, 4)
    with pytest.raises(ValueError, match="exact set"):
        _capture(writer, 0, visible)


def test_callback_deep_copies_ndarray_and_nested_mapping() -> None:
    writer = S.AtomicFrameWriter(S.SMOKE_MODE, 4)
    payload = _payload(0)
    original_rgb = payload["rgb"]["data"].copy()
    original_labels = deepcopy(payload["semantic"]["data"]["idToLabels"])
    state = {"step": 1}
    writer.arm(0, 0, state, 2.0)
    writer.before_step(0)
    writer.callback(payload)
    payload["rgb"]["data"][:] = 0
    payload["semantic"]["data"]["idToLabels"].clear()
    row = writer.close(0, state, 2.0)
    assert row["rgb"]["sha256"] == S._sha(original_rgb.tobytes(order="C"))
    assert row["id_to_labels"] == {
        str(key): value["class"] for key, value in original_labels.items()
    }


def test_callback_cardinality_unsolicited_and_skip_are_rejected() -> None:
    writer = S.AtomicFrameWriter(S.SMOKE_MODE, 4)
    with pytest.raises(ValueError, match="unsolicited"):
        writer.callback(_payload(0))
    with pytest.raises(ValueError, match="순서대로"):
        writer.arm(1, 1, {}, 2.0)
    writer.arm(0, 0, {}, 2.0)
    writer.before_step(0)
    with pytest.raises(ValueError, match="callback 정확히 하나"):
        writer.close(0, {}, 2.0)
    writer.callback(_payload(0))
    with pytest.raises(ValueError, match="정확히 하나"):
        writer.callback(_payload(0))


@pytest.mark.parametrize("stale_kind", ["rgb", "semantic", "code1_as_code3"])
def test_stale_or_subset_marker_evidence_is_rejected(stale_kind: str) -> None:
    payload = _payload(3)
    if stale_kind == "rgb":
        payload["rgb"]["data"] = _payload(1)["rgb"]["data"]
    else:
        payload["semantic"]["data"] = _payload(1)["semantic"]["data"]
        if stale_kind == "code1_as_code3":
            payload["rgb"]["data"] = _payload(1)["rgb"]["data"]
    writer = S.AtomicFrameWriter(S.SMOKE_MODE, 4)
    for ordinal in range(3):
        _capture(writer, ordinal, _payload(ordinal))
    with pytest.raises(ValueError, match="channel order|exact set"):
        _capture(writer, 3, payload)


def test_black_code0_and_registered_without_pixels_are_rejected() -> None:
    black = _payload(0)
    black["rgb"]["data"][:] = 0
    writer = S.AtomicFrameWriter(S.SMOKE_MODE, 4)
    with pytest.raises(ValueError, match="channel order"):
        _capture(writer, 0, black)
    no_pixels = _payload(0)
    no_pixels["semantic"]["data"]["mask"][:] = 0
    writer = S.AtomicFrameWriter(S.SMOKE_MODE, 4)
    with pytest.raises(ValueError, match="exact set"):
        _capture(writer, 0, no_pixels)


def test_whole_bundle_replay_and_non_strict_frame_are_rejected() -> None:
    smoke, bundles = _valid_smoke()
    smoke.finish()
    replayed = deepcopy(bundles)
    replayed[3] = deepcopy(replayed[1])
    replayed_raw = list(smoke._raw)
    replayed_raw[3] = replayed_raw[1]
    with pytest.raises(ValueError, match="replay"):
        S.verify_sequence(replayed, replayed_raw)

    writer = S.AtomicFrameWriter(S.RECORD_MODE, 2)
    first = _payload(0, marker=False)
    _capture(writer, 0, first, None)
    replay = _payload(1, marker=False)
    replay["read_times"]["data"] = deepcopy(first["read_times"]["data"])
    replay["reference_time"]["data"] = deepcopy(first["reference_time"]["data"])
    _capture(writer, 1, replay, None)
    with pytest.raises(ValueError, match="frame 순서|timing fraction"):
        writer.finish()


def test_smoke_frame_must_increment_one_and_timing_fractions_must_match() -> None:
    mismatch = _payload(0)
    mismatch["reference_time"]["data"]["referenceTimeNumerator"] += 1
    writer = S.AtomicFrameWriter(S.SMOKE_MODE, 4)
    with pytest.raises(ValueError, match="분수"):
        _capture(writer, 0, mismatch)

    negative = _payload(0)
    negative["reference_time"]["data"].update(
        referenceTimeNumerator=-55,
        referenceTimeDenominator=-30,
    )
    writer = S.AtomicFrameWriter(S.SMOKE_MODE, 4)
    with pytest.raises(ValueError, match="분수"):
        _capture(writer, 0, negative)

    writer = S.AtomicFrameWriter(S.SMOKE_MODE, 4)
    for ordinal in range(4):
        payload = _payload(ordinal)
        if ordinal >= 2:
            frame = 56 + ordinal
            payload["read_times"]["data"]["frameNumber"] = frame
            payload["read_times"]["data"]["rationalTimeOfSimNumerator"] = frame
            payload["reference_time"]["data"]["referenceTimeNumerator"] = frame
        _capture(writer, ordinal, payload)
    with pytest.raises(ValueError, match="frame 순서|timing fraction"):
        writer.finish()


@pytest.mark.parametrize(
    "mutation", ["cross_capture_state", "bad_state_sha", "raw_resolution"]
)
def test_evidence_revalidation_rejects_cross_capture_physics_and_raw_shape(
    mutation: str,
) -> None:
    writer, bundles = _valid_smoke()
    writer.finish()
    changed = deepcopy(bundles)
    if mutation == "cross_capture_state":
        changed[3]["physics"]["state_sha256"] = "a" * 64
    elif mutation == "bad_state_sha":
        changed[0]["physics"]["state_sha256"] = "NOT-A-SHA"
    else:
        changed[2]["camera"]["resolution"] = [999, 999]
    with pytest.raises(ValueError, match="physics|state/time|resolution"):
        S.verify_sequence(changed, writer._raw)


@pytest.mark.parametrize(
    "change", ["B_equals_A", "Areturn_differs", "projection", "resolution"]
)
def test_camera_cycle_projection_and_resolution_contract(change: str) -> None:
    writer = S.AtomicFrameWriter(S.SMOKE_MODE, 4)
    for ordinal in range(4):
        payload = _payload(ordinal)
        if change == "B_equals_A" and ordinal == 2:
            payload["camera"]["data"]["view_matrix"] = _view("A")
        if change == "Areturn_differs" and ordinal == 3:
            payload["camera"]["data"]["view_matrix"] = _view("B") + 1
        if change == "projection" and ordinal == 3:
            payload["camera"]["data"]["projection"][0, 0] = 2
        if change == "resolution" and ordinal == 3:
            payload["camera"]["data"]["resolution"] = [11, 10]
        if change == "resolution" and ordinal == 3:
            with pytest.raises(ValueError, match="resolution"):
                _capture(writer, ordinal, payload)
            return
        _capture(writer, ordinal, payload)
    with pytest.raises(ValueError, match="CameraParams|projection/resolution"):
        writer.finish()


def test_record_uses_same_atomic_bundle_and_rejects_reserved_marker() -> None:
    writer = S.AtomicFrameWriter(S.RECORD_MODE, 1)
    row = _capture(writer, 0, _payload(0, marker=False), None)
    assert row["expected_code"] is row["decoded_code"] is None
    assert writer.finish()[0]["mode"] == S.RECORD_MODE
    writer = S.AtomicFrameWriter(S.RECORD_MODE, 1)
    with pytest.raises(ValueError, match="reserved marker"):
        _capture(writer, 0, _payload(0), None)


def test_single_bundle_verifier_supports_nonzero_ordinal_and_rejects_raw_tamper() -> None:
    writer, bundles = _valid_smoke()
    checked = S.verify_bundle(bundles[2], writer._raw[2], 2)
    assert checked["frame"] == 57 and checked["mode"] == S.SMOKE_MODE

    rgb, mask = writer._raw[2]
    changed = bytearray(rgb)
    changed[0] ^= 1
    with pytest.raises(ValueError, match="SHA/bytes"):
        S.verify_bundle(bundles[2], (bytes(changed), mask), 2)


def test_physics_time_and_shared_callback_reference_are_required() -> None:
    writer = S.AtomicFrameWriter(S.RECORD_MODE, 1)
    payload = _payload(0, marker=False)
    payload["camera"]["adapter_callback_ref"] = "other"
    with pytest.raises(ValueError, match="adapter callback reference"):
        _capture(writer, 0, payload, None)
    writer = S.AtomicFrameWriter(S.RECORD_MODE, 1)
    payload = _payload(0, marker=False)
    payload["render_product_ref"] = "other-rp"
    with pytest.raises(ValueError, match="단일 RenderProduct"):
        _capture(writer, 0, payload, None)
    writer = S.AtomicFrameWriter(S.RECORD_MODE, 1)
    state = {"step": 1}
    writer.arm(0, None, state, 2.0)
    writer.before_step(0)
    writer.callback(_payload(0, marker=False))
    with pytest.raises(ValueError, match="physics state/time"):
        writer.close(0, {"step": 2}, 2.0)


@pytest.mark.parametrize("mutation", ["reference", "fraction_type", "mode", "empty_callback", "empty_product"])
def test_evidence_revalidation_rejects_fraction_mode_and_identity_mutation(mutation: str):
    writer, bundles = _valid_smoke()
    changed = deepcopy(bundles)
    if mutation == "reference":
        changed[-1]["timing"]["reference_time"]["referenceTimeNumerator"] += 1
    elif mutation == "fraction_type":
        changed[-1]["timing_fraction"]["reference_numerator"] = float(
            changed[-1]["timing_fraction"]["reference_numerator"])
    elif mutation == "mode":
        for row in changed:
            row["mode"] = "bogus"
    elif mutation == "empty_callback":
        changed[0]["adapter_callback_ref"] = ""
    else:
        for row in changed:
            row["render_product_ref"] = ""
    with pytest.raises(ValueError):
        S.verify_sequence(changed, writer._raw)


@pytest.mark.parametrize("mode", [S.SMOKE_MODE, S.RECORD_MODE])
@pytest.mark.parametrize("direction", [0, -1])
def test_frame_increase_cannot_hide_frozen_or_reversed_rational_time(mode: str, direction: int):
    count = 4 if mode == S.SMOKE_MODE else 2
    writer = S.AtomicFrameWriter(mode, count)
    for ordinal in range(count):
        payload = _payload(ordinal, marker=mode == S.SMOKE_MODE)
        frame_time = 55 + ordinal * direction
        payload["read_times"]["data"]["rationalTimeOfSimNumerator"] = frame_time
        payload["reference_time"]["data"]["referenceTimeNumerator"] = frame_time
        _capture(writer, ordinal, payload, ordinal if mode == S.SMOKE_MODE else None)
    with pytest.raises(ValueError, match="timing fraction"):
        writer.finish()
