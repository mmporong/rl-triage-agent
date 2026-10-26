from __future__ import annotations

import copy
import hashlib
import importlib
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
M = importlib.import_module("vision_prepared_capture_v1")
F = importlib.import_module("vision_frame_sync_v1")

WIDTH = 10
HEIGHT = 10
RP = "RenderProduct_0"
CAMERA = "/World/Camera"
PHYSICS = {"joint": [0.1, 0.2], "step": 7}


def _times(frame: int) -> dict[str, Any]:
    return {
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


def _view(kind: str) -> np.ndarray:
    base = np.arange(16, dtype=np.float64)
    return base + (100.0 if kind == "B" else 0.0)


def _payload(
    *,
    frame: int,
    marker: int,
    view: str = "A",
    rp: str = RP,
    empty: bool = False,
    background: int = 0,
) -> dict[str, Any]:
    rgba = np.full((HEIGHT, WIDTH, 4), background, dtype=np.uint8)
    rgba[:, :, 3] = 255
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint32)
    labels: dict[str, Any] = {"0": {"class": "robot/body"}}
    for index, label in enumerate(sorted(F.marker_labels(marker)), 1):
        mask[index, 1:9] = index
        labels[str(index)] = {"class": label}
        role = "sentinel" if label == F.SENTINEL else label.rsplit("/", 1)[-1]
        rgba[index, 1:9, :3] = F.PALETTE[role]
    semantic = np.array([], dtype=np.float64) if empty else mask
    return {
        "swhFrameNumber": 0,
        "reference_time": (frame, 30),
        "distribution_outputs": {},
        "trigger_outputs": {},
        "named_outputs": {},
        "annotators": {
            "camera": {rp: CAMERA},
            "resolution": {rp: np.array([WIDTH, HEIGHT], dtype=np.int32)},
            "rgb": {rp: {"data": rgba}},
            "semantic_segmentation": {rp: {"data": semantic, "idToLabels": labels}},
            "CameraParams": {
                rp: {
                    "cameraViewTransform": _view(view),
                    "cameraProjection": np.eye(4, dtype=np.float64).reshape(16),
                    "renderProductResolution": np.array(
                        [WIDTH, HEIGHT], dtype=np.int32
                    ),
                }
            },
            "IsaacReadTimes": {rp: _times(frame)},
        },
    }


def _mux() -> Any:
    return M.PreparedCaptureMux(
        render_product_ref=RP,
        camera_path=CAMERA,
        resolution=[WIDTH, HEIGHT],
    )


def _prepare(mux: Any, callbacks: list[dict[str, Any]]) -> None:
    mux.begin_prepare(PHYSICS, 2.0, marker_code=3)
    for payload in callbacks:
        mux.observe(payload)
    mux.end_prepare(PHYSICS, 2.0)


def _arm(
    mux: Any,
    ordinal: int,
    callbacks: list[dict[str, Any]],
    *,
    after: dict[str, Any] = PHYSICS,
    after_time: float = 2.0,
) -> None:
    mux.begin_arm(ordinal, PHYSICS, 2.0, marker_code=ordinal)
    for payload in callbacks:
        mux.observe(payload)
    mux.end_arm(ordinal, after, after_time)


def _complete(
    *,
    bootstrap: bool = False,
    mutate: Any = None,
) -> tuple[Any, dict[str, Any]]:
    mux = _mux()
    prep = _payload(frame=60, marker=3)
    preparation = (
        [_payload(frame=60, marker=3, empty=True), prep] if bootstrap else [prep]
    )
    _prepare(mux, preparation)
    for ordinal in range(4):
        payload = _payload(
            frame=61 + ordinal,
            marker=ordinal,
            view="B" if ordinal == 2 else "A",
            background=ordinal * 3,
        )
        if mutate is not None:
            mutate(ordinal, payload)
        _arm(mux, ordinal, [payload])
    mux.begin_detach()
    mux.end_detach()
    mux.resolve()
    return mux, mux.seal()


def _seal_failure(mux: Any) -> dict[str, Any]:
    if mux.state != "detaching" and mux.state != "detached":
        mux.begin_detach()
    if mux.state == "detaching":
        mux.end_detach()
    mux.resolve()
    return mux.seal()


@pytest.mark.parametrize("bootstrap", [False, True])
def test_success_accepts_only_registered_prepare_shapes(bootstrap: bool) -> None:
    _, result = _complete(bootstrap=bootstrap)
    assert result["approved"] is True
    assert result["failure"] is None
    assert len(result["bundles"]) == 4
    assert result["preparation"] == {
        "logical_ordinal": 4,
        "journal_written": False,
        "marker_code": 3,
        "bootstrap_empty_observed": bootstrap,
        "valid_raw_index": 1 if bootstrap else 0,
        "frame_number": 60,
    }
    assert result["raw_callback_count"] == (6 if bootstrap else 5)
    assert [row["frame_number"] for row in result["arm_sources"]] == [61, 62, 63, 64]


@pytest.mark.parametrize(
    ("callbacks", "reason"),
    [
        ([], "prepare_callback_sequence_invalid"),
        (
            [_payload(frame=60, marker=3, empty=True)],
            "prepare_callback_sequence_invalid",
        ),
        (
            [
                {"bad": "envelope"},
                _payload(frame=60, marker=3),
            ],
            "prepare_callback_sequence_invalid",
        ),
        (
            [_payload(frame=60, marker=3), _payload(frame=61, marker=3)],
            "prepare_callback_sequence_invalid",
        ),
        (
            [
                _payload(frame=60, marker=3, empty=True),
                _payload(frame=60, marker=3),
                _payload(frame=61, marker=3),
            ],
            "prepare_callback_sequence_invalid",
        ),
    ],
)
def test_prepare_cardinality_and_order_fail_closed(
    callbacks: list[dict[str, Any]], reason: str
) -> None:
    mux = _mux()
    mux.begin_prepare(PHYSICS, 2.0, marker_code=3)
    for callback in callbacks:
        mux.observe(callback)
    with pytest.raises(RuntimeError, match="prepare callback"):
        mux.end_prepare(PHYSICS, 2.0)
    result = _seal_failure(mux)
    assert result["approved"] is False
    assert result["failure"]["code"] == reason
    assert result["raw_callback_count"] == len(callbacks)


@pytest.mark.parametrize("case", ["none", "empty", "extra"])
def test_each_arm_requires_exactly_one_valid_callback(case: str) -> None:
    mux = _mux()
    _prepare(mux, [_payload(frame=60, marker=3)])
    mux.begin_arm(0, PHYSICS, 2.0, marker_code=0)
    if case == "empty":
        mux.observe(_payload(frame=61, marker=0, empty=True))
    elif case == "extra":
        mux.observe(_payload(frame=61, marker=0))
        mux.observe(_payload(frame=62, marker=0))
    with pytest.raises(RuntimeError, match="arm callback"):
        mux.end_arm(0, PHYSICS, 2.0)
    result = _seal_failure(mux)
    assert result["approved"] is False
    assert result["bundles"] == []


def test_callback_between_arms_is_whole_failure() -> None:
    mux = _mux()
    _prepare(mux, [_payload(frame=60, marker=3)])
    _arm(mux, 0, [_payload(frame=61, marker=0)])
    mux.observe(_payload(frame=62, marker=1))
    with pytest.raises(RuntimeError, match="재개"):
        mux.begin_arm(1, PHYSICS, 2.0, marker_code=1)
    assert _seal_failure(mux)["failure"]["code"] == "unsolicited_callback"


@pytest.mark.parametrize("window", ["detaching", "detached"])
def test_callback_during_or_after_detach_is_whole_failure(window: str) -> None:
    mux, _ = _complete()
    # 새 성공 mux로 seal 이전 detach 경계를 재구성한다.
    mux = _mux()
    _prepare(mux, [_payload(frame=60, marker=3)])
    for ordinal in range(4):
        _arm(
            mux,
            ordinal,
            [
                _payload(
                    frame=61 + ordinal,
                    marker=ordinal,
                    view="B" if ordinal == 2 else "A",
                )
            ],
        )
    mux.begin_detach()
    if window == "detached":
        mux.end_detach()
    mux.observe(_payload(frame=65, marker=3))
    if window == "detaching":
        mux.end_detach()
    mux.resolve()
    result = mux.seal()
    assert result["approved"] is False
    assert result["failure"]["code"] in {
        "callback_during_detach",
        "unsolicited_callback",
    }


def test_same_buffer_is_frozen_inside_callback() -> None:
    mux = _mux()
    prep = _payload(frame=60, marker=3)
    expected = prep["annotators"]["rgb"][RP]["data"].copy()
    mux.begin_prepare(PHYSICS, 2.0, marker_code=3)
    mux.observe(prep)
    prep["annotators"]["rgb"][RP]["data"][:] = 0
    prep["annotators"]["semantic_segmentation"][RP]["idToLabels"].clear()
    mux.end_prepare(PHYSICS, 2.0)
    for ordinal in range(4):
        raw = _payload(
            frame=61 + ordinal, marker=ordinal, view="B" if ordinal == 2 else "A"
        )
        _arm(mux, ordinal, [raw])
    mux.begin_detach()
    mux.end_detach()
    mux.resolve()
    result = mux.seal()
    assert result["approved"] is True
    assert np.any(expected[:, :, :3])


def test_arm_buffer_is_frozen_before_source_reuses_it() -> None:
    mux = _mux()
    _prepare(mux, [_payload(frame=60, marker=3)])
    raw = _payload(frame=61, marker=0)
    expected = raw["annotators"]["rgb"][RP]["data"][:, :, :3].tobytes(order="C")
    mux.begin_arm(0, PHYSICS, 2.0, marker_code=0)
    mux.observe(raw)
    raw["annotators"]["rgb"][RP]["data"][:] = 0
    raw["annotators"]["semantic_segmentation"][RP]["data"][:] = 0
    mux.end_arm(0, PHYSICS, 2.0)
    for ordinal in range(1, 4):
        _arm(
            mux,
            ordinal,
            [
                _payload(
                    frame=61 + ordinal,
                    marker=ordinal,
                    view="B" if ordinal == 2 else "A",
                )
            ],
        )
    mux.begin_detach()
    mux.end_detach()
    mux.resolve()
    result = mux.seal()
    assert result["approved"] is True
    assert result["bundles"][0]["rgb"]["sha256"] == hashlib.sha256(expected).hexdigest()


def test_two_render_product_group_is_invalid() -> None:
    raw = _payload(frame=60, marker=3)
    raw["annotators"]["rgb"]["RenderProduct_1"] = copy.deepcopy(
        raw["annotators"]["rgb"][RP]
    )
    mux = _mux()
    mux.begin_prepare(PHYSICS, 2.0, marker_code=3)
    mux.observe(raw)
    with pytest.raises(RuntimeError):
        mux.end_prepare(PHYSICS, 2.0)
    assert _seal_failure(mux)["callbacks"][0]["classification"] == "invalid"


@pytest.mark.parametrize("operation", ["arm_skip", "detach_early", "resolve_early"])
def test_stage_skip_cannot_produce_approval(operation: str) -> None:
    mux = _mux()
    _prepare(mux, [_payload(frame=60, marker=3)])
    with pytest.raises(ValueError):
        if operation == "arm_skip":
            mux.begin_arm(1, PHYSICS, 2.0, marker_code=1)
        elif operation == "detach_early":
            mux.begin_detach()
        else:
            mux.resolve()
    if mux.state != "detaching":
        result = _seal_failure(mux)
        assert result["approved"] is False


def test_prepare_marker_must_be_code_three() -> None:
    mux = _mux()
    _prepare(mux, [_payload(frame=60, marker=0)])
    for ordinal in range(4):
        _arm(
            mux,
            ordinal,
            [
                _payload(
                    frame=61 + ordinal,
                    marker=ordinal,
                    view="B" if ordinal == 2 else "A",
                )
            ],
        )
    mux.begin_detach()
    mux.end_detach()
    mux.resolve()
    result = mux.seal()
    assert result["approved"] is False
    assert result["failure"]["code"] == "resolve_validation_failed"


def test_prepare_marker_mapping_with_extra_fields_is_not_normalized_away() -> None:
    raw = _payload(frame=60, marker=3)
    labels = raw["annotators"]["semantic_segmentation"][RP]["idToLabels"]
    marker_id = next(key for key, value in labels.items() if key != "0" and value)
    labels[marker_id]["unexpected"] = "value"
    mux = _mux()
    _prepare(mux, [raw])
    for ordinal in range(4):
        _arm(
            mux,
            ordinal,
            [
                _payload(
                    frame=61 + ordinal,
                    marker=ordinal,
                    view="B" if ordinal == 2 else "A",
                )
            ],
        )
    mux.begin_detach()
    mux.end_detach()
    mux.resolve()
    assert mux.seal()["approved"] is False


@pytest.mark.parametrize("case", ["rp", "frame_gap", "duplicate_frame"])
def test_resolve_rejects_identity_and_frame_progression(case: str) -> None:
    def mutate(ordinal: int, payload: dict[str, Any]) -> None:
        if case == "rp" and ordinal == 1:
            for annotator in payload["annotators"].values():
                annotator["OtherRP"] = annotator.pop(RP)
        if case == "frame_gap" and ordinal == 2:
            payload["annotators"]["IsaacReadTimes"][RP]["frameNumber"] = 70
            payload["annotators"]["IsaacReadTimes"][RP][
                "rationalTimeOfSimNumerator"
            ] = 70
            payload["reference_time"] = (70, 30)
        if case == "duplicate_frame" and ordinal == 1:
            payload["annotators"]["IsaacReadTimes"][RP] = _times(61)
            payload["reference_time"] = (61, 30)

    _, result = _complete(mutate=mutate)
    assert result["approved"] is False
    assert result["failure"]["code"] == "resolve_validation_failed"


@pytest.mark.parametrize("case", ["B_equals_A", "Areturn_differs", "bad_pixel"])
def test_atomic_frame_contract_remains_authoritative(case: str) -> None:
    def mutate(ordinal: int, payload: dict[str, Any]) -> None:
        if case == "B_equals_A" and ordinal == 2:
            payload["annotators"]["CameraParams"][RP]["cameraViewTransform"] = _view(
                "A"
            )
        if case == "Areturn_differs" and ordinal == 3:
            payload["annotators"]["CameraParams"][RP]["cameraViewTransform"] = (
                _view("B") + 1
            )
        if case == "bad_pixel" and ordinal == 1:
            payload["annotators"]["rgb"][RP]["data"][1, 1:9, :3] = (20, 20, 20)

    _, result = _complete(mutate=mutate)
    assert result["approved"] is False
    assert result["bundles"] == []


@pytest.mark.parametrize("case", ["physics", "time"])
def test_phase_physics_and_time_change_is_sticky(case: str) -> None:
    mux = _mux()
    _prepare(mux, [_payload(frame=60, marker=3)])
    after = {"joint": [9.0], "step": 7} if case == "physics" else PHYSICS
    after_time = 3.0 if case == "time" else 2.0
    with pytest.raises(RuntimeError, match="arm callback"):
        _arm(mux, 0, [_payload(frame=61, marker=0)], after=after, after_time=after_time)
    assert _seal_failure(mux)["failure"]["code"] == "physics_or_time_changed"


def test_cross_phase_time_change_is_rejected_at_resolve() -> None:
    mux = _mux()
    _prepare(mux, [_payload(frame=60, marker=3)])
    for ordinal in range(4):
        time = 3.0 if ordinal == 2 else 2.0
        mux.begin_arm(ordinal, PHYSICS, time, marker_code=ordinal)
        mux.observe(
            _payload(
                frame=61 + ordinal, marker=ordinal, view="B" if ordinal == 2 else "A"
            )
        )
        mux.end_arm(ordinal, PHYSICS, time)
    mux.begin_detach()
    mux.end_detach()
    mux.resolve()
    assert mux.seal()["failure"]["code"] == "resolve_validation_failed"


def test_prepare_to_arm_timing_fraction_must_advance_one() -> None:
    prep = _payload(frame=60, marker=3)
    times = prep["annotators"]["IsaacReadTimes"][RP]
    times["rationalTimeOfSimNumerator"] = 999
    prep["reference_time"] = (999, 30)
    mux = _mux()
    _prepare(mux, [prep])
    for ordinal in range(4):
        _arm(
            mux,
            ordinal,
            [
                _payload(
                    frame=61 + ordinal,
                    marker=ordinal,
                    view="B" if ordinal == 2 else "A",
                )
            ],
        )
    mux.begin_detach()
    mux.end_detach()
    mux.resolve()
    result = mux.seal()
    assert result["approved"] is False
    assert result["failure"]["code"] == "resolve_validation_failed"


def test_source_error_before_callback_is_sticky_and_preserved() -> None:
    mux = _mux()
    mux.begin_prepare(PHYSICS, 2.0, marker_code=3)
    mux.source_error(OSError("source failed"))
    mux.observe(_payload(frame=60, marker=3))
    with pytest.raises(RuntimeError):
        mux.end_prepare(PHYSICS, 2.0)
    result = _seal_failure(mux)
    assert result["failure"] == {
        "code": "source_error_before_callback",
        "state": "preparing",
        "raw_callback_index": None,
        "error_type": "builtins.OSError",
    }
    assert result["raw_callback_count"] == 1


def test_snapshot_array_cap_fails_before_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mux = _mux()
    mux.begin_prepare(PHYSICS, 2.0, marker_code=3)
    monkeypatch.setattr(M, "MAX_SNAPSHOT_TOTAL_ARRAY_BYTES", 8)
    mux.observe(_payload(frame=60, marker=3))
    assert mux.callbacks[0]["descriptor"] is None
    assert (
        _seal_failure(mux)["failure"]["code"]
        == "callback_snapshot_or_classification_failed"
    )


def test_cumulative_array_cap_rejects_overflowing_array_before_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SpyArray(np.ndarray):
        copy_calls = 0

        def copy(self, *args: Any, **kwargs: Any) -> np.ndarray:
            type(self).copy_calls += 1
            return super().copy(*args, **kwargs)

    raw = _payload(frame=60, marker=3)
    raw["distribution_outputs"] = {
        "first": np.zeros(8, dtype=np.uint8).view(SpyArray),
        "overflow": np.zeros(8, dtype=np.uint8).view(SpyArray),
    }
    monkeypatch.setattr(M, "MAX_SNAPSHOT_TOTAL_ARRAY_BYTES", 12)
    mux = _mux()
    mux.begin_prepare(PHYSICS, 2.0, marker_code=3)
    mux.observe(raw)
    assert SpyArray.copy_calls == 1
    assert mux.callbacks[0]["descriptor"] is None


def test_bootstrap_dummy_reserves_shared_budget_before_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _payload(frame=60, marker=3, empty=True)
    mux = _mux()
    mux.begin_prepare(PHYSICS, 2.0, marker_code=3)
    monkeypatch.setattr(M, "MAX_SNAPSHOT_TOTAL_ARRAY_BYTES", 700)
    allocations = 0

    def forbidden_zeros(*args: Any, **kwargs: Any) -> np.ndarray:
        nonlocal allocations
        allocations += 1
        raise AssertionError((args, kwargs))

    monkeypatch.setattr(M.np, "zeros", forbidden_zeros)
    mux.observe(raw)
    assert allocations == 0
    assert mux._failure["code"] == "callback_snapshot_or_classification_failed"


def test_huge_declared_bootstrap_resolution_is_rejected_before_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _payload(frame=60, marker=3, empty=True)
    huge = np.array([2**31, 2**31], dtype=np.int64)
    raw["annotators"]["resolution"][RP] = huge
    raw["annotators"]["CameraParams"][RP]["renderProductResolution"] = huge.copy()
    allocations = 0

    def forbidden_zeros(*args: Any, **kwargs: Any) -> np.ndarray:
        nonlocal allocations
        allocations += 1
        raise AssertionError((args, kwargs))

    monkeypatch.setattr(M.np, "zeros", forbidden_zeros)
    mux = _mux()
    mux.begin_prepare(PHYSICS, 2.0, marker_code=3)
    mux.observe(raw)
    assert allocations == 0
    assert mux.callbacks[0]["classification"] == "invalid"


@pytest.mark.parametrize("case", ["depth", "entries", "nodes"])
def test_snapshot_structure_budgets_fail_closed(
    monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    raw = _payload(frame=60, marker=3)
    if case == "depth":
        value: Any = "leaf"
        for _ in range(5):
            value = {"next": value}
        raw["distribution_outputs"] = value
        monkeypatch.setattr(M, "MAX_SNAPSHOT_DEPTH", 4)
    elif case == "entries":
        raw["distribution_outputs"] = {str(index): index for index in range(5)}
        monkeypatch.setattr(M, "MAX_SNAPSHOT_ENTRIES", 8)
    else:
        raw["distribution_outputs"] = {"a": [1, 2, 3]}
        monkeypatch.setattr(M, "MAX_SNAPSHOT_NODES", 8)
    mux = _mux()
    mux.begin_prepare(PHYSICS, 2.0, marker_code=3)
    mux.observe(raw)
    assert mux.callbacks[0]["descriptor"] is None
    assert mux._failure["code"] == "callback_snapshot_or_classification_failed"


def test_seal_is_write_once_in_memory() -> None:
    mux, result = _complete()
    assert result["state"] == "sealed"
    with pytest.raises(ValueError, match="한 번만"):
        mux.seal()


def test_no_bundle_is_available_before_detach_and_resolve() -> None:
    mux = _mux()
    _prepare(mux, [_payload(frame=60, marker=3)])
    for ordinal in range(4):
        _arm(
            mux,
            ordinal,
            [
                _payload(
                    frame=61 + ordinal,
                    marker=ordinal,
                    view="B" if ordinal == 2 else "A",
                )
            ],
        )
    assert mux._bundles == []
    with pytest.raises(ValueError, match="detach"):
        mux.resolve()
    result = _seal_failure(mux)
    assert result["approved"] is False
    assert result["bundles"] == []


def test_callback_evidence_rows_are_bounded_after_failure() -> None:
    mux = _mux()
    for _ in range(M.MAX_RAW_CALLBACKS + 3):
        mux.observe({"bad": "envelope"})
    result = _seal_failure(mux)
    assert result["raw_callback_count"] == M.MAX_RAW_CALLBACKS + 3
    assert len(result["callbacks"]) == M.MAX_RAW_CALLBACKS
    assert result["callback_rows_dropped"] == 3


@pytest.mark.parametrize(
    "kwargs",
    [
        {"render_product_ref": "", "camera_path": CAMERA, "resolution": [10, 10]},
        {"render_product_ref": RP, "camera_path": "", "resolution": [10, 10]},
        {"render_product_ref": RP, "camera_path": CAMERA, "resolution": [10, 0]},
    ],
)
def test_constructor_rejects_incomplete_registered_envelope(
    kwargs: dict[str, Any],
) -> None:
    with pytest.raises(ValueError):
        M.PreparedCaptureMux(**kwargs)


def test_success_bundle_rgb_sha_comes_from_frozen_atomic_payload() -> None:
    _, result = _complete()
    arm0 = _payload(frame=61, marker=0)
    expected = arm0["annotators"]["rgb"][RP]["data"][:, :, :3].tobytes(order="C")
    assert result["bundles"][0]["rgb"]["sha256"] == hashlib.sha256(expected).hexdigest()
