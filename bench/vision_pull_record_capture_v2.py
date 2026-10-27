"""IsaacReadTimes 원문은 보존하고 Play 소비값만 엄격 정규화한다."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import vision_play_worker_v1 as PLAY
import vision_pull_record_capture_v1 as PREVIOUS
import vision_timing_adapter_v1 as TIMING

SCHEMA = PREVIOUS.SCHEMA
RAW_SCHEMA = PREVIOUS.RAW_SCHEMA
ANNOTATORS = PREVIOUS.ANNOTATORS
RETURN_KEYS = {
    "rgb",
    "read_times",
    "reference_time",
    "pull_receipt",
    "pull_raw_receipt",
    "pull_npz",
}


def normalize_capture_result(
    value: Any, *, simulation_time_s: float
) -> dict[str, Any]:
    """등록된 11/12키 raw만 받아 기존 11키 consumer 계약으로 바꾼다."""
    if not isinstance(value, Mapping) or set(value) != RETURN_KEYS:
        raise ValueError("pull capture 반환 필드가 계약과 다르다")
    read_times = value["read_times"]
    reference_time = value["reference_time"]
    if not isinstance(read_times, Mapping) or not isinstance(reference_time, Mapping):
        raise TypeError("pull capture timing 원문이 mapping이 아니다")
    normalized = TIMING.normalize_render_timing(
        read_times, reference_time, simulation_time_s
    )
    return {
        **value,
        "read_times": normalized["read_times"],
        "reference_time": normalized["reference_time"],
    }


def make_capture_factory(
    validated: Mapping[str, Any],
    app: Any,
    *,
    dependencies: Mapping[str, Any] | None = None,
) -> PLAY.CameraCaptureFactory:
    """v1 원문 기록 뒤 Play에 전달되는 timing만 정규화한다."""
    previous_factory = PREVIOUS.make_capture_factory(
        validated, app, dependencies=dependencies
    )

    def factory(
        base: Any, rep: Any, render_product: Any, profile: Mapping[str, Any]
    ) -> tuple[PLAY.CameraCapture, Callable[[], None]]:
        capture, cleanup = previous_factory(base, rep, render_product, profile)

        def normalized_capture(
            capture_base: Any,
            capture_rep: Any,
            annotators: list[Any],
            camera_profile: Mapping[str, Any],
            eye: list[float],
            lookat: list[float],
            ordinal: int,
        ) -> dict[str, Any]:
            raw_result = capture(
                capture_base,
                capture_rep,
                annotators,
                camera_profile,
                eye,
                lookat,
                ordinal,
            )
            simulation_time = float(capture_base.sim.current_time)
            return normalize_capture_result(
                raw_result, simulation_time_s=simulation_time
            )

        return normalized_capture, cleanup

    return factory
