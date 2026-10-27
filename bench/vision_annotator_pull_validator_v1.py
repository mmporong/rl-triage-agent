"""Writer-free annotator pull receipt의 AABA·semantic·시간·물리 불변식을 검증한다."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from fractions import Fraction
from itertools import pairwise
from typing import Any, cast

import numpy as np
import record_vision_evidence_v1 as R
import vision_annotator_pull_collector_v1 as COLLECTOR
import vision_frame_sync_v1 as FRAME

SCHEMA = "vision_annotator_pull_validation_v1"


def _integer(value: Any, anchor: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{anchor}는 정수여야 한다")
    converted = int(value)
    if positive and converted <= 0:
        raise ValueError(f"{anchor}는 양수여야 한다")
    return converted


def _finite(value: Any, anchor: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{anchor}는 수치여야 한다")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{anchor}는 finite여야 한다")
    return converted


def _matrix(value: Any, anchor: str) -> np.ndarray:
    array = np.asarray(value)
    if array.size != 16 or array.dtype.kind not in "fiu":
        raise ValueError(f"{anchor}는 finite flat16/4x4 수치 행렬이어야 한다")
    converted = array.astype(np.float64, copy=False).reshape(4, 4)
    if not np.isfinite(converted).all():
        raise ValueError(f"{anchor}는 finite flat16/4x4 수치 행렬이어야 한다")
    return converted


def _rgb(value: Any) -> tuple[np.ndarray, np.ndarray]:
    if (
        not isinstance(value, np.ndarray)
        or value.dtype != np.uint8
        or value.ndim != 3
        or value.shape[2] not in (3, 4)
        or value.size == 0
    ):
        raise ValueError("rgb는 비어 있지 않은 uint8 RGB/RGBA ndarray여야 한다")
    raw = value.copy(order="C")
    return raw, raw[:, :, :3].copy(order="C")


def _semantic(value: Any, rgb: np.ndarray, marker_code: int) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"data", "info"}:
        raise ValueError("semantic_segmentation payload 형식이 다르다")
    info = value["info"]
    if not isinstance(info, Mapping) or "idToLabels" not in info:
        raise ValueError("semantic_segmentation.info.idToLabels가 없다")
    mask = value["data"]
    if isinstance(mask, np.ndarray) and mask.shape == (*rgb.shape[:2], 1):
        mask = mask[:, :, 0]
    checked, raw, labels, decoded = FRAME._semantic(
        {"mask": mask, "idToLabels": info["idToLabels"]},
        rgb,
        marker_code,
        FRAME.SMOKE_MODE,
    )
    return {
        "mask": checked,
        "mask_sha256": hashlib.sha256(raw).hexdigest(),
        "id_to_labels": labels,
        "decoded_code": decoded,
    }


def _camera(value: Any, expected: Mapping[str, Any], rgb: np.ndarray) -> dict[str, Any]:
    required = {"cameraViewTransform", "cameraProjection", "renderProductResolution"}
    if not isinstance(value, Mapping) or not required <= set(value):
        raise ValueError("CameraParams 필수 필드가 없다")
    view = _matrix(value["cameraViewTransform"], "cameraViewTransform")
    projection = _matrix(value["cameraProjection"], "cameraProjection")
    resolution_raw = value["renderProductResolution"]
    expected_resolution_raw = expected.get("resolution")
    if isinstance(resolution_raw, (str, bytes)) or isinstance(
        expected_resolution_raw, (str, bytes)
    ):
        raise TypeError("CameraParams/arm resolution이 sequence가 아니다")
    try:
        resolution = list(cast(Any, resolution_raw))
        expected_resolution = list(cast(Any, expected_resolution_raw))
    except TypeError as error:
        raise TypeError("CameraParams/arm resolution이 sequence가 아니다") from error
    actual_resolution = [rgb.shape[1], rgb.shape[0]]
    if resolution != expected_resolution or resolution != actual_resolution:
        raise ValueError("CameraParams resolution이 arm/RGB와 다르다")
    eye = expected.get("eye_world_m")
    lookat = expected.get("lookat_world_m")
    for name, vector in (("eye_world_m", eye), ("lookat_world_m", lookat)):
        if (
            not isinstance(vector, Sequence)
            or isinstance(vector, (str, bytes))
            or len(vector) != 3
            or any(not math.isfinite(float(item)) for item in vector)
        ):
            raise ValueError(f"arm camera {name} receipt가 finite vector3가 아니다")
    assert isinstance(eye, Sequence) and not isinstance(eye, (str, bytes))
    assert isinstance(lookat, Sequence) and not isinstance(lookat, (str, bytes))
    return {
        "view": view.tolist(),
        "projection": projection.tolist(),
        "resolution": resolution,
        "commanded_eye_world_m": list(eye),
        "commanded_lookat_world_m": list(lookat),
    }


def _timing(value: Any, simulation_time_s: float) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("IsaacReadTimes는 mapping이어야 한다")
    required = set(R.ISAAC_READ_TIMES_FIELDS)
    if set(value) not in (required, required | {"execOut"}):
        raise ValueError("IsaacReadTimes는 등록 11필드 또는 execOut 포함 12필드여야 한다")
    frame = _integer(value["frameNumber"], "frameNumber")
    numerator = _integer(value["rationalTimeOfSimNumerator"], "rational numerator")
    denominator = _integer(
        value["rationalTimeOfSimDenominator"], "rational denominator", positive=True
    )
    observed_time = _finite(value["simulationTime"], "IsaacReadTimes.simulationTime")
    if observed_time != simulation_time_s:
        raise ValueError("IsaacReadTimes simulationTime과 physics 시각이 다르다")
    return {
        "frame_number": frame,
        "rational_numerator": numerator,
        "rational_denominator": denominator,
        "rational": Fraction(numerator, denominator),
        "simulation_time_s": observed_time,
    }


def _physics(capture: Mapping[str, Any]) -> dict[str, Any]:
    before = capture.get("physics_before")
    after = capture.get("physics_after")
    if not isinstance(before, Mapping) or not isinstance(after, Mapping):
        raise TypeError("physics before/after receipt가 mapping이 아니다")
    if before != after:
        raise ValueError("capture step/pull 동안 physics state/time이 바뀌었다")
    return dict(before)


def validate(report: Mapping[str, Any]) -> dict[str, Any]:
    """수집 report를 strict AABA 대체 경계로 승인하거나 예외로 거부한다."""
    if not isinstance(report, Mapping) or report.get("schema") != COLLECTOR.REPORT_SCHEMA:
        raise ValueError("annotator pull report schema가 다르다")
    if report.get("status") != "collected_unvalidated":
        raise ValueError("성공 수집 report만 검증할 수 있다")
    if report.get("atomicity_claimed") is not False:
        raise ValueError("annotator pull은 원자성을 주장할 수 없다")
    captures = report.get("captures")
    if not isinstance(captures, Sequence) or isinstance(captures, (str, bytes)):
        raise TypeError("captures는 sequence여야 한다")
    if len(captures) != len(COLLECTOR.AABA):
        raise ValueError("AABA capture는 정확히 네 건이어야 한다")
    trace = report.get("operation_trace")
    if not isinstance(trace, list):
        raise TypeError("annotator pull operation trace가 list가 아니다")
    expected_trace: list[dict[str, Any]] = [
        {"operation": "attach", "annotator": name}
        for name in COLLECTOR.ANNOTATORS
    ]
    for ordinal in range(len(COLLECTOR.AABA)):
        expected_trace.append({"operation": "step", "ordinal": ordinal})
        expected_trace.extend(
            {"operation": "pull", "ordinal": ordinal, "annotator": name}
            for name in COLLECTOR.ANNOTATORS
        )
    expected_trace.extend(
        {"operation": "detach", "annotator": name}
        for name in reversed(COLLECTOR.ANNOTATORS)
    )
    if trace != expected_trace:
        raise ValueError("attach→step1→연속 pull→detach operation trace가 다르다")

    frames: list[int] = []
    fractions: list[Fraction] = []
    physics: list[dict[str, Any]] = []
    views: list[list[list[float]]] = []
    projections: list[list[list[float]]] = []
    decoded: list[int] = []
    for ordinal, (capture, (camera_kind, marker_code)) in enumerate(
        zip(captures, COLLECTOR.AABA, strict=True)
    ):
        if not isinstance(capture, Mapping) or capture.get("schema") != COLLECTOR.SCHEMA:
            raise ValueError("capture schema가 다르다")
        if capture.get("ordinal") != ordinal:
            raise ValueError("capture ordinal이 순서와 다르다")
        if capture.get("pull_order") != list(COLLECTOR.ANNOTATORS):
            raise ValueError("annotator pull 순서가 계약과 다르다")
        if capture.get("do_array_copy") is not True:
            raise ValueError("annotator 원문은 do_array_copy=True로 수집해야 한다")
        update_counts = capture.get("app_update_counts")
        if not isinstance(update_counts, Mapping) or set(update_counts) != {
            "before_render",
            "after_render_before_pull",
            "after_pull",
        }:
            raise ValueError("render/pull app update counter receipt가 다르다")
        before_render = _integer(update_counts["before_render"], "before render update")
        after_render = _integer(
            update_counts["after_render_before_pull"], "after render update"
        )
        after_pull = _integer(update_counts["after_pull"], "after pull update")
        if (
            capture.get("render_window_app_updates")
            != after_render - before_render
        ):
            raise ValueError("render window app update counter 계산이 다르다")
        if capture.get("pull_window_app_updates") != after_pull - after_render:
            raise ValueError("pull window app update counter 계산이 다르다")
        if before_render > after_render or after_render > after_pull:
            raise ValueError("app update counter가 단조 증가하지 않았다")
        if capture.get("pull_window_app_updates") != 0:
            raise ValueError("연속 annotator pull 사이에 app update가 있었다")
        if capture.get("atomicity_claimed") is not False:
            raise ValueError("capture가 annotator 원자성을 주장했다")
        step = capture.get("step")
        if not isinstance(step, Mapping) or step != {
            "call_count": 1,
            "delta_time_s": 0.0,
            "pause_timeline": False,
            "rt_subframes": 1,
        }:
            raise ValueError("capture step은 고정 delta0 profile이어야 한다")
        arm = capture.get("arm")
        if (
            not isinstance(arm, Mapping)
            or arm.get("ordinal") != ordinal
            or arm.get("camera_kind") != camera_kind
            or arm.get("marker_code") != marker_code
            or set(arm.get("marker_labels", ())) != FRAME.marker_labels(marker_code)
        ):
            raise ValueError("arm camera/marker receipt가 AABA 계약과 다르다")
        raw = capture.get("raw")
        if not isinstance(raw, Mapping) or set(raw) != set(COLLECTOR.ANNOTATORS):
            raise ValueError("raw annotator payload가 네 종류를 보존하지 않았다")
        rgba, rgb = _rgb(raw["rgb"])
        semantic = _semantic(raw["semantic_segmentation"], rgb, marker_code)
        camera_expected = arm.get("camera")
        if not isinstance(camera_expected, Mapping):
            raise TypeError("arm camera receipt가 mapping이 아니다")
        camera = _camera(raw["CameraParams"], camera_expected, rgb)
        physics_row = _physics(capture)
        timing = _timing(raw["IsaacReadTimes"], physics_row["simulation_time_s"])
        if semantic["decoded_code"] != marker_code:
            raise ValueError("semantic marker code가 현재 arm과 다르다")
        frames.append(timing["frame_number"])
        fractions.append(timing["rational"])
        physics.append(physics_row)
        views.append(camera["view"])
        projections.append(camera["projection"])
        decoded.append(marker_code)
        if rgba.size == 0:
            raise ValueError("RGB 원문이 비었다")

    if any(right <= left for left, right in pairwise(frames)):
        raise ValueError("Isaac frameNumber가 AABA 순서로 진행하지 않았다")
    if any(right <= left for left, right in pairwise(fractions)):
        raise ValueError("Isaac rational time이 AABA 순서로 진행하지 않았다")
    if any(row != physics[0] for row in physics[1:]):
        raise ValueError("AABA 동안 physics state/time이 바뀌었다")
    if not (views[0] == views[1] == views[3] and views[2] != views[0]):
        raise ValueError("CameraParams view가 AABA를 입증하지 않는다")
    if any(value != projections[0] for value in projections[1:]):
        raise ValueError("AABA 동안 CameraParams projection이 바뀌었다")
    return {
        "schema": SCHEMA,
        "approved": True,
        "capture_count": len(captures),
        "camera_sequence": [camera for camera, _ in COLLECTOR.AABA],
        "marker_codes": decoded,
        "frame_numbers": frames,
        "rational_times": [
            [fraction.numerator, fraction.denominator] for fraction in fractions
        ],
        "physics_sha256": physics[0]["sha256"],
        "atomicity_claimed": False,
    }
