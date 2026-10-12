"""정책 rollout 없이 DC motor 토크-속도 허용 곡선을 읽고 비교한다."""
from __future__ import annotations

import importlib
import math
import re
from typing import Any


VELOCITY_GRID = (0.0, 3.0, -3.0, 6.0, -6.0, 9.0, -9.0, 12.0, -12.0, 18.0, -18.0, 30.0, -30.0)
CONFIG_KEYS = ("effort_limit", "saturation_effort", "velocity_limit")
ABS_TOL = 1.0e-6
REL_TOL = 1.0e-6
LIMITATIONS = {
    "policy_rollout_used": False,
    "cause_confirmed": False,
    "llm_superiority_claimed": False,
    "scope": "device_agnostic_policy_free_bounded_curve_readback",
    "agreement_scope": "registered_input_grid_only",
    "current_regression_coverage": "cpu_fake_tensor_only",
}


def _invalid(
    reason: str,
    provenance: dict,
    measurements: list[dict] | None = None,
    requested_efforts: dict[str, float] | None = None,
) -> dict:
    return {
        "schema": "actuator_envelope_readback_v1",
        "status": "invalid",
        "curve_matches_expected": None,
        "invalid_reason": reason,
        "requested_efforts": requested_efforts,
        "measurements": measurements or [],
        "config_provenance": provenance,
        "limitations": dict(LIMITATIONS),
    }


def _expected_config(value: Any) -> dict[str, float] | None:
    if not isinstance(value, dict) or set(value) != set(CONFIG_KEYS):
        return None
    result = {}
    for key in CONFIG_KEYS:
        item = value[key]
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item) or item <= 0:
            return None
        result[key] = float(item)
    return result


def _flat_numbers(value: Any) -> list[float] | None:
    try:
        detached = value.detach().cpu().reshape(-1).tolist()
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None
    if not isinstance(detached, list) or not detached:
        return None
    result = []
    for item in detached:
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item):
            return None
        result.append(float(item))
    return result


def _runtime_scalar(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(value) else None
    numbers = _flat_numbers(value)
    if numbers is None or any(not math.isclose(item, numbers[0], rel_tol=REL_TOL, abs_tol=ABS_TOL) for item in numbers):
        return None
    return numbers[0]


def _documented_limits(velocity: float, config: dict[str, float]) -> tuple[float, float]:
    effort = config["effort_limit"]
    saturation = config["saturation_effort"]
    velocity_limit = config["velocity_limit"]
    corner = velocity_limit * (1.0 + effort / saturation)
    bounded_velocity = max(-corner, min(corner, velocity))
    top = saturation * (1.0 - bounded_velocity / velocity_limit)
    bottom = saturation * (-1.0 - bounded_velocity / velocity_limit)
    return max(bottom, -effort), min(top, effort)


def inspect_actuator_envelope(
    actuator: Any,
    expected_config: dict[str, float],
    *,
    config_source_sha256: str,
    tensor_module: Any | None = None,
) -> dict:
    """고정 속도 격자에서 ``_clip_effort``의 양·음 포화 출력을 측정한다.

    이 함수는 측정값이 기대 곡선과 같은지만 기록한다. 입력이나 측정이 유효하지
    않으면 결함 판정으로 승격하지 않고 ``invalid``를 반환한다.
    """
    config = _expected_config(expected_config)
    provenance = {
        "caller_asserted_source_sha256": config_source_sha256 if isinstance(config_source_sha256, str) else None,
        "source_binding_verified": False,
        "expected_config": config,
        "velocity_grid_rad_s": list(VELOCITY_GRID),
    }
    if not isinstance(config_source_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", config_source_sha256):
        return _invalid("invalid_config_source_sha256", provenance)
    if config is None:
        return _invalid("invalid_expected_config", provenance)
    if actuator is None or not callable(getattr(actuator, "_clip_effort", None)) or not hasattr(actuator, "_joint_vel"):
        return _invalid("missing_actuator_interface", provenance)

    runtime = {
        "effort_limit": _runtime_scalar(getattr(actuator, "effort_limit", None)),
        "saturation_effort": _runtime_scalar(getattr(actuator, "_saturation_effort", None)),
        "velocity_limit": _runtime_scalar(getattr(actuator, "velocity_limit", None)),
    }
    provenance["runtime_config_readback"] = runtime
    if any(runtime[key] is None or not math.isclose(runtime[key], config[key], rel_tol=REL_TOL, abs_tol=ABS_TOL)
           for key in CONFIG_KEYS):
        return _invalid("runtime_config_mismatch", provenance)

    original_velocity = actuator._joint_vel
    shape = getattr(original_velocity, "shape", None)
    if shape is None or _flat_numbers(original_velocity) is None:
        return _invalid("invalid_joint_velocity_buffer", provenance)
    torch = tensor_module
    if torch is None:
        try:
            torch = importlib.import_module("torch")
        except ImportError:
            return _invalid("tensor_module_unavailable", provenance)

    measurements: list[dict] = []
    magnitude = max(config["effort_limit"], config["saturation_effort"]) * 1000.0
    if not math.isfinite(magnitude):
        return _invalid("nonfinite_request_effort", provenance)
    requested_efforts = {"positive": magnitude, "negative": -magnitude}
    try:
        for velocity in VELOCITY_GRID:
            actuator._joint_vel = torch.full_like(original_velocity, velocity)
            positive = actuator._clip_effort(torch.full_like(original_velocity, magnitude))
            negative = actuator._clip_effort(torch.full_like(original_velocity, -magnitude))
            if getattr(positive, "shape", None) != shape or getattr(negative, "shape", None) != shape:
                return _invalid("clip_output_shape_mismatch", provenance, measurements, requested_efforts)
            maximum = _flat_numbers(positive)
            minimum = _flat_numbers(negative)
            if maximum is None or minimum is None:
                return _invalid("nonfinite_clip_output", provenance, measurements, requested_efforts)
            expected_minimum, expected_maximum = _documented_limits(velocity, config)
            matches = (all(math.isclose(value, expected_maximum, rel_tol=REL_TOL, abs_tol=ABS_TOL)
                           for value in maximum)
                       and all(math.isclose(value, expected_minimum, rel_tol=REL_TOL, abs_tol=ABS_TOL)
                               for value in minimum))
            measurements.append({
                "velocity_rad_s": velocity,
                "expected_min_effort": expected_minimum,
                "expected_max_effort": expected_maximum,
                "measured_min_effort": minimum,
                "measured_max_effort": maximum,
                "matches_expected": matches,
            })
    except Exception as error:
        return _invalid(f"measurement_error:{type(error).__name__}", provenance, measurements, requested_efforts)
    finally:
        actuator._joint_vel = original_velocity

    return {
        "schema": "actuator_envelope_readback_v1",
        "status": "valid",
        "curve_matches_expected": all(row["matches_expected"] for row in measurements),
        "invalid_reason": None,
        "requested_efforts": requested_efforts,
        "measurements": measurements,
        "config_provenance": provenance,
        "limitations": dict(LIMITATIONS),
    }
