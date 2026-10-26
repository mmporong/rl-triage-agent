from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
M = importlib.import_module("vision_prepared_record_v1")
F = importlib.import_module("vision_frame_sync_v1")
R = importlib.import_module("record_vision_evidence_v1")

WIDTH = 10
HEIGHT = 10
RP = "RenderProduct_0"
CAMERA = "/World/Camera"


def _times(frame: int) -> dict[str, Any]:
    return {
        "durationDenominator": 60,
        "durationNumerator": 1,
        "externalTimeOfSimNs": frame * 1_000_000,
        "frameNumber": frame,
        "rationalTimeOfSimDenominator": 30,
        "rationalTimeOfSimNumerator": frame,
        "sampleTimeOffsetInSimFrames": 0,
        "simulationTime": frame / 30,
        "simulationTimeMonotonic": frame / 30,
        "swhFrameNumber": frame,
        "systemTime": 1000.0 + frame,
    }


def _payload(
    frame: int,
    *,
    empty: bool = False,
    rp: str = RP,
    camera: str = CAMERA,
    projection_offset: float = 0.0,
    callback_ref: str | None = None,
) -> dict[str, Any]:
    rgba = np.zeros((HEIGHT, WIDTH, 4), dtype=np.uint8)
    rgba[:, :, :3] = frame % 251
    rgba[:, :, 3] = 255
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint32)
    return {
        "swhFrameNumber": frame,
        "reference_time": (frame, 30),
        "distribution_outputs": {},
        "trigger_outputs": {},
        "named_outputs": {"callback_ref": callback_ref or f"callback-{frame}"},
        "annotators": {
            "camera": {rp: camera},
            "resolution": {rp: np.array([WIDTH, HEIGHT], dtype=np.int32)},
            "rgb": {rp: {"data": rgba}},
            "semantic_segmentation": {
                rp: {
                    "data": np.array([], dtype=np.float64) if empty else mask,
                    "idToLabels": {"0": {"class": "robot/body"}},
                }
            },
            "CameraParams": {
                rp: {
                    "cameraViewTransform": np.arange(16, dtype=np.float64),
                    "cameraProjection": np.eye(4).reshape(16) + projection_offset,
                    "renderProductResolution": np.array(
                        [WIDTH, HEIGHT], dtype=np.int32
                    ),
                }
            },
            "IsaacReadTimes": {rp: _times(frame)},
        },
    }


def _physics(step: int = 0) -> dict[str, Any]:
    return {"joint": [step, step + 1], "root": [float(step)]}


def _mux(count: int = 2, receipts: list[dict[str, Any]] | None = None) -> Any:
    sink = receipts if receipts is not None else []

    def commit(ordinal: int, _decoded: Any, bundle: Any) -> dict[str, Any]:
        row = {
            "ordinal": ordinal,
            "frame": bundle["timing"]["read_times"]["frameNumber"],
        }
        sink.append(row)
        return {"journal": f"frame-{ordinal}.json", "sha256": "a" * 64}

    return M.PreparedRecordMux(
        render_product_ref=RP,
        camera_path=CAMERA,
        resolution=[WIDTH, HEIGHT],
        expected_count=count,
        commit_callback=commit,
    )


def _prepare(mux: Any, *, empty: bool = False, frame: int = 10) -> None:
    capture_time = frame / 30
    mux.begin_prepare(_physics(), capture_time)
    if empty:
        mux.observe(_payload(frame - 1, empty=True))
    mux.observe(_payload(frame))
    mux.end_prepare(_physics(), capture_time)


def _arm(mux: Any, ordinal: int, frame: int, *, step: int | None = None) -> None:
    policy_step = ordinal + 1 if step is None else step
    capture_time = frame / 30
    mux.begin_arm(ordinal, _physics(policy_step), capture_time)
    mux.observe(_payload(frame))
    mux.end_arm(ordinal, _physics(policy_step), capture_time)


def _finish(mux: Any, step: int) -> dict[str, Any]:
    mux.begin_detach()
    mux.end_detach(_physics(step), float(step))
    mux.resolve()
    return mux.seal()


@pytest.mark.parametrize("empty", [False, True])
def test_prepare_then_streams_record_frames_without_prepare_journal(
    empty: bool,
) -> None:
    receipts: list[dict[str, Any]] = []
    mux = _mux(receipts=receipts)
    _prepare(mux, empty=empty)
    _arm(mux, 0, 11)
    _arm(mux, 1, 13)
    result = _finish(mux, 2)
    assert result["approved"] is True
    assert result["committed_count"] == 2
    assert [row["ordinal"] for row in receipts] == [0, 1]
    assert result["preparation"]["bootstrap_empty_observed"] is empty
    assert "journal" not in result["preparation"]
    assert [row["frame_number"] for row in result["manifests"]] == [11, 13]
    assert all("decoded" not in row for row in result["manifests"])


def test_same_input_array_is_frozen_before_later_mutation() -> None:
    captured: list[str] = []

    def commit(_ordinal: int, decoded: Any, bundle: Any) -> dict[str, Any]:
        captured.append(bundle["rgb"]["sha256"])
        decoded["rgb"].fill(255)
        return {"ok": True}

    mux = M.PreparedRecordMux(
        render_product_ref=RP,
        camera_path=CAMERA,
        resolution=[WIDTH, HEIGHT],
        expected_count=1,
        commit_callback=commit,
    )
    _prepare(mux)
    payload = _payload(11)
    mux.begin_arm(0, _physics(1), 11 / 30)
    mux.observe(payload)
    payload["annotators"]["rgb"][RP]["data"].fill(99)
    mux.end_arm(0, _physics(1), 11 / 30)
    result = _finish(mux, 1)
    assert result["approved"] is True
    assert result["manifests"][0]["rgb"]["sha256"] == captured[0]


@pytest.mark.parametrize("kind", ["zero", "empty_only", "invalid_then_valid"])
def test_prepare_rejects_missing_or_nonregistered_sequences(kind: str) -> None:
    mux = _mux(1)
    mux.begin_prepare(_physics(), 10 / 30)
    if kind == "empty_only":
        mux.observe(_payload(9, empty=True))
    elif kind == "invalid_then_valid":
        invalid = _payload(9)
        del invalid["annotators"]["CameraParams"]
        mux.observe(invalid)
        mux.observe(_payload(10))
    with pytest.raises(RuntimeError):
        mux.end_prepare(_physics(), 10 / 30)


def test_arm_requires_exactly_one_valid_callback() -> None:
    mux = _mux(1)
    _prepare(mux)
    mux.begin_arm(0, _physics(1), 11 / 30)
    mux.observe(_payload(11))
    mux.observe(_payload(12))
    with pytest.raises(RuntimeError):
        mux.end_arm(0, _physics(1), 11 / 30)


@pytest.mark.parametrize("window", ["outside", "detach"])
def test_outside_and_detach_callbacks_are_sticky_hold(window: str) -> None:
    mux = _mux(1)
    if window == "outside":
        mux.observe(_payload(10))
    else:
        _prepare(mux)
        _arm(mux, 0, 11)
        mux.begin_detach()
        mux.observe(_payload(12))
        mux.end_detach(_physics(1), 1.0)
    if window == "outside":
        mux.begin_detach()
        mux.end_detach(_physics(), 0.0)
    mux.resolve()
    assert mux.seal()["approved"] is False


@pytest.mark.parametrize("change", ["frame", "rp", "camera", "projection"])
def test_cross_frame_identity_and_timing_changes_fail_closed(change: str) -> None:
    mux = _mux(1)
    _prepare(mux)
    kwargs: dict[str, Any] = {}
    frame = 10 if change == "frame" else 11
    if change == "rp":
        kwargs["rp"] = "other"
    elif change == "camera":
        kwargs["camera"] = "/World/Other"
    elif change == "projection":
        kwargs["projection_offset"] = 1.0
    mux.begin_arm(0, _physics(1), frame / 30)
    mux.observe(_payload(frame, **kwargs))
    with pytest.raises((RuntimeError, ValueError)):
        mux.end_arm(0, _physics(1), frame / 30)


def test_physics_and_time_may_change_between_arms_but_not_inside_arm() -> None:
    mux = _mux(2)
    _prepare(mux)
    _arm(mux, 0, 11, step=4)
    mux.begin_arm(1, _physics(9), 12 / 30)
    mux.observe(_payload(12))
    with pytest.raises(RuntimeError):
        mux.end_arm(1, _physics(10), 12 / 30)


def test_commit_failure_is_sticky_and_is_not_retried() -> None:
    calls: list[int] = []

    def commit(ordinal: int, _decoded: Any, _bundle: Any) -> dict[str, Any]:
        calls.append(ordinal)
        raise OSError("disk")

    mux = M.PreparedRecordMux(
        render_product_ref=RP,
        camera_path=CAMERA,
        resolution=[WIDTH, HEIGHT],
        expected_count=1,
        commit_callback=commit,
    )
    _prepare(mux)
    mux.begin_arm(0, _physics(1), 11 / 30)
    mux.observe(_payload(11))
    with pytest.raises(OSError):
        mux.end_arm(0, _physics(1), 11 / 30)
    with pytest.raises(RuntimeError):
        mux.begin_arm(0, _physics(1), 1.0)
    assert calls == [0]


def test_625_frames_retain_only_lightweight_manifests() -> None:
    mux = _mux(M.MAX_RECORD_FRAMES)
    _prepare(mux, frame=100)
    for ordinal in range(M.MAX_RECORD_FRAMES):
        _arm(mux, ordinal, 101 + ordinal)
    result = _finish(mux, M.MAX_RECORD_FRAMES)
    assert result["approved"] is True
    assert len(result["manifests"]) == M.MAX_RECORD_FRAMES
    assert not any(
        isinstance(value, np.ndarray)
        for manifest in result["manifests"]
        for value in manifest.values()
    )
    assert set(mux.__dict__).isdisjoint({"_raw", "_bundles", "_arms"})


def test_raw_overflow_records_bounded_rows_and_holds() -> None:
    mux = _mux(1)
    for frame in range(20):
        mux.observe(_payload(frame + 1))
    assert mux.raw_callback_count == 20
    assert len(mux.callbacks) == mux.raw_limit + M.MAX_ERROR_ROWS
    mux.begin_detach()
    mux.end_detach(_physics(), 0.0)
    mux.resolve()
    result = mux.seal()
    assert result["approved"] is False
    assert result["callback_rows_dropped"] > 0


def test_per_callback_snapshot_budget_is_reused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(M.PREPARED, "MAX_SNAPSHOT_TOTAL_ARRAY_BYTES", 8)
    mux = _mux(1)
    mux.begin_prepare(_physics(), 10 / 30)
    mux.observe(_payload(10))
    with pytest.raises(RuntimeError):
        mux.end_prepare(_physics(), 10 / 30)
    assert mux.callbacks[0]["classification"] == "invalid"


def test_source_error_and_invalid_count_fail_closed() -> None:
    with pytest.raises(ValueError):
        _mux(M.MAX_RECORD_FRAMES + 1)
    mux = _mux(1)
    mux.source_error(RuntimeError("source"))
    with pytest.raises(RuntimeError):
        mux.begin_prepare(_physics(), 10 / 30)


def test_receipt_must_be_small_json_mapping() -> None:
    def commit(_ordinal: int, _decoded: Any, _bundle: Any) -> dict[str, Any]:
        return {"large": "x" * M.MAX_RECEIPT_BYTES}

    mux = M.PreparedRecordMux(
        render_product_ref=RP,
        camera_path=CAMERA,
        resolution=[WIDTH, HEIGHT],
        expected_count=1,
        commit_callback=commit,
    )
    _prepare(mux)
    mux.begin_arm(0, _physics(1), 11 / 30)
    mux.observe(_payload(11))
    with pytest.raises(ValueError):
        mux.end_arm(0, _physics(1), 11 / 30)


def test_before_prepare_and_after_detach_physics_are_both_preserved() -> None:
    mux = _mux(1)
    _prepare(mux)
    _arm(mux, 0, 11, step=5)
    result = _finish(mux, 7)
    assert result["before_prepare_physics"]["snapshot"] == _physics()
    assert result["after_detach_physics"]["snapshot"] == _physics(7)
    assert (
        result["before_prepare_physics"]["sha256"]
        != result["after_detach_physics"]["sha256"]
    )
