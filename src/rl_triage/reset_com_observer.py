"""Reset 뒤 PhysX base COM 변화를 읽기만 하는 bounded 관측기."""

from __future__ import annotations

import hashlib
import json
import math
import re
from copy import deepcopy
from typing import Mapping


SOURCE_SHA_KEYS = {
    "task_sha256", "configuration_sha256", "isaaclab_sha256", "observer_sha256",
}
MEASUREMENT_DEFINITIONS = {
    "base_com_max_abs_delta_xy": {"unit": "simulation_length_native"},
    "base_com_outside_range_fraction": {"unit": "fraction"},
}
SAMPLE_FIELD_DEFINITIONS = {
    "delta_x": {"unit": "simulation_length_native", "scope": "reset_records.samples"},
    "delta_y": {"unit": "simulation_length_native", "scope": "reset_records.samples"},
}


def _canonical_sha(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


MEASUREMENT_DEFINITIONS_SHA256 = _canonical_sha(MEASUREMENT_DEFINITIONS)


def _is_int(value: object) -> bool:
    return type(value) is int


def _finite(value: object) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, TypeError, ValueError):
        return False


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 hex digest")
    return value


def _cpu_copy(value: object) -> object:
    current = value
    for method in ("detach", "cpu"):
        operation = getattr(current, method, None)
        if callable(operation):
            current = operation()
    to_list = getattr(current, "tolist", None)
    if callable(to_list):
        current = to_list()
    return deepcopy(current)


def _normalise_coms(value: object, *, num_envs: int, body_count: int,
                    expected_shape: tuple[int, int, int] | None = None) -> tuple[list, tuple[int, int, int]]:
    copied = _cpu_copy(value)
    if not isinstance(copied, (list, tuple)) or len(copied) != num_envs:
        raise ValueError("PhysX COM readback environment dimension is invalid")
    rows = []
    component_count = None
    for env_row in copied:
        if not isinstance(env_row, (list, tuple)) or len(env_row) != body_count:
            raise ValueError("PhysX COM readback body dimension is invalid")
        bodies = []
        for body_row in env_row:
            if not isinstance(body_row, (list, tuple)) or len(body_row) < 3:
                raise ValueError("PhysX COM readback component dimension is invalid")
            if component_count is None:
                component_count = len(body_row)
            if len(body_row) != component_count:
                raise ValueError("PhysX COM readback components must be rectangular")
            bodies.append([float(item) if _finite(item) else item for item in body_row])
        rows.append(bodies)
    shape = (num_envs, body_count, component_count or 0)
    if expected_shape is not None and shape != expected_shape:
        raise ValueError("PhysX COM readback shape changed after baseline capture")
    return rows, shape


def _all_finite(coms: list) -> bool:
    return all(_finite(value) for env in coms for body in env for value in body)


class BaseComResetObserver:
    """Read fresh PhysX COM values after caller-completed resets without mutating the environment."""

    def __init__(
        self, env: object, *, source_binding: Mapping[str, str],
        expected_reset_sequences: int = 20, expected_resets_per_env: int = 10,
        allowed_xy_range: float = 0.05, tolerance: float = 1.0e-4,
    ):
        num_envs = getattr(env, "num_envs", None)
        if not _is_int(num_envs) or num_envs <= 0:
            raise ValueError("env.num_envs must be a positive integer")
        if not _is_int(expected_reset_sequences) or expected_reset_sequences <= 0:
            raise ValueError("expected_reset_sequences must be a positive integer")
        if not _is_int(expected_resets_per_env) or expected_resets_per_env <= 0:
            raise ValueError("expected_resets_per_env must be a positive integer")
        if not _finite(allowed_xy_range) or float(allowed_xy_range) < 0.0:
            raise ValueError("allowed_xy_range must be a finite non-negative number")
        if not _finite(tolerance) or float(tolerance) < 0.0:
            raise ValueError("tolerance must be a finite non-negative number")
        if not isinstance(source_binding, Mapping) or set(source_binding) != SOURCE_SHA_KEYS:
            raise ValueError("source_binding must contain task/configuration/isaaclab/observer SHA-256 values")

        try:
            robot = env.scene["robot"]  # type: ignore[attr-defined,index]
            body_names = tuple(robot.body_names)
            view = robot.root_physx_view
        except Exception as exc:
            raise ValueError("env must expose scene robot body names and root PhysX view") from exc
        if not body_names or any(not isinstance(name, str) or not name for name in body_names):
            raise ValueError("robot body_names must be non-empty strings")
        base_indices = [index for index, name in enumerate(body_names) if name == "base"]
        if len(base_indices) != 1:
            raise ValueError("robot body_names must contain exactly one base body")

        baseline, shape = _normalise_coms(
            view.get_coms(), num_envs=num_envs, body_count=len(body_names),
        )
        if not _all_finite(baseline):
            raise ValueError("baseline PhysX COM readback must be finite")
        self._env = env
        self._robot = robot
        self._view = view
        self._num_envs = num_envs
        self._body_names = body_names
        self._base_index = base_indices[0]
        self._baseline = baseline
        self._baseline_shape = shape
        self._baseline_sha256 = _canonical_sha({
            "body_names": body_names, "shape": shape, "coms": baseline,
        })
        self._source_binding = {
            key: _sha256(source_binding[key], key) for key in sorted(SOURCE_SHA_KEYS)
        }
        self._expected_sequences = expected_reset_sequences
        self._expected_resets_per_env = expected_resets_per_env
        self._allowed_xy_range = float(allowed_xy_range)
        self._tolerance = float(tolerance)
        self._records: list[dict] = []

    def _validate_body_order(self) -> None:
        try:
            current = tuple(self._robot.body_names)
        except Exception as exc:
            raise ValueError("robot body order is unavailable") from exc
        if current != self._body_names:
            raise ValueError("robot body order changed after baseline capture")
        if [index for index, name in enumerate(current) if name == "base"] != [self._base_index]:
            raise ValueError("base body binding changed after baseline capture")

    def record_completed_reset(self, *, reset_sequence: int, reset_env_ids: object) -> dict:
        """Record one caller-declared completed reset using a fresh ``get_coms()`` read."""
        if len(self._records) >= self._expected_sequences:
            raise OverflowError("completed reset sequence capacity exceeded")
        if not _is_int(reset_sequence) or reset_sequence < 0:
            raise ValueError("reset_sequence must be a non-negative integer, not a boolean")
        if self._records and reset_sequence != self._records[-1]["reset_sequence"] + 1:
            raise ValueError("reset_sequence must increase by exactly one")
        if not isinstance(reset_env_ids, (list, tuple)) or not reset_env_ids:
            raise ValueError("reset_env_ids must be a non-empty explicit list")
        if any(not _is_int(env_id) for env_id in reset_env_ids):
            raise ValueError("reset_env_ids must contain integers, not booleans")
        env_ids = tuple(reset_env_ids)
        if len(set(env_ids)) != len(env_ids):
            raise ValueError("reset_env_ids contain duplicates")
        if any(env_id < 0 or env_id >= self._num_envs for env_id in env_ids):
            raise ValueError("reset_env_ids contain an out-of-range environment")
        self._validate_body_order()

        diagnostic = None
        current = None
        try:
            raw_current = self._view.get_coms()
        except Exception as exc:
            diagnostic = {"exception_type": type(exc).__name__}
        else:
            try:
                current, _ = _normalise_coms(
                    raw_current, num_envs=self._num_envs,
                    body_count=len(self._body_names), expected_shape=self._baseline_shape,
                )
            except ValueError:
                raise
            except Exception as exc:
                diagnostic = {"exception_type": type(exc).__name__}

        samples = []
        if current is not None and _all_finite(current):
            derived_unavailable = False
            for env_id in env_ids:
                baseline = self._baseline[env_id][self._base_index]
                observed = current[env_id][self._base_index]
                delta_x = observed[0] - baseline[0]
                delta_y = observed[1] - baseline[1]
                max_abs_delta = max(abs(delta_x), abs(delta_y))
                if not all(_finite(value) for value in (delta_x, delta_y, max_abs_delta)):
                    derived_unavailable = True
                    break
                samples.append({
                    "env_id": env_id,
                    "baseline_x": baseline[0], "baseline_y": baseline[1],
                    "current_x": observed[0], "current_y": observed[1],
                    "delta_x": delta_x, "delta_y": delta_y,
                    "max_abs_delta_xy": max_abs_delta,
                    "outside_allowed_range": max_abs_delta > self._allowed_xy_range + self._tolerance,
                })
            if derived_unavailable:
                samples = []
                state = "unavailable"
                diagnostic = {"exception_type": "NonFiniteDerivedComDelta"}
            else:
                state = "present"
        else:
            state = "unavailable"
            diagnostic = diagnostic or {"exception_type": "NonFiniteComReadback"}

        record = {
            "reset_sequence": reset_sequence, "reset_env_ids": list(env_ids),
            "measurement_state": state, "samples": samples,
            **({"diagnostic": diagnostic} if diagnostic is not None else {}),
        }
        self._records.append(deepcopy(record))
        return deepcopy(record)

    def aggregate(self) -> dict:
        """Return raw reset readbacks and coverage without classifying a fault or cause."""
        declared_counts = [0] * self._num_envs
        finite_counts = [0] * self._num_envs
        per_env_samples: list[list[dict]] = [[] for _ in range(self._num_envs)]
        unavailable_sequences = []
        for record in self._records:
            for env_id in record["reset_env_ids"]:
                declared_counts[env_id] += 1
            if record["measurement_state"] != "present":
                unavailable_sequences.append(record["reset_sequence"])
                continue
            for sample in record["samples"]:
                env_id = sample["env_id"]
                finite_counts[env_id] += 1
                per_env_samples[env_id].append(sample)

        per_environment = []
        all_finite_samples = []
        for env_id, samples in enumerate(per_env_samples):
            all_finite_samples.extend(samples)
            worst = max((sample["max_abs_delta_xy"] for sample in samples), default=None)
            per_environment.append({
                "env_id": env_id,
                "baseline_x": self._baseline[env_id][self._base_index][0],
                "baseline_y": self._baseline[env_id][self._base_index][1],
                "declared_reset_count": declared_counts[env_id],
                "finite_reset_measurement_count": finite_counts[env_id],
                "worst_max_abs_delta_xy": worst,
                "measurement_state": "present" if worst is not None else "unavailable",
            })

        expected_sample_count = self._num_envs * self._expected_resets_per_env
        finite_sample_count = len(all_finite_samples)
        sequence_complete = len(self._records) == self._expected_sequences
        count_complete = all(count == self._expected_resets_per_env for count in declared_counts)
        finite_complete = all(count == self._expected_resets_per_env for count in finite_counts)
        complete = sequence_complete and count_complete and finite_complete and not unavailable_sequences
        if not self._records:
            coverage_status = "not_observed"
        elif finite_sample_count == 0:
            coverage_status = "unavailable"
        elif complete:
            coverage_status = "complete"
        else:
            coverage_status = "partial"

        overall_worst = max(
            (sample["max_abs_delta_xy"] for sample in all_finite_samples), default=None,
        )
        finite_environment_worst = [
            row["worst_max_abs_delta_xy"] for row in per_environment
            if row["worst_max_abs_delta_xy"] is not None
        ]
        outside_count = sum(
            worst > self._allowed_xy_range + self._tolerance
            for worst in finite_environment_worst
        )
        outside_denominator = len(finite_environment_worst)
        outside_fraction = outside_count / outside_denominator if outside_denominator else None
        measurement_state = "present" if finite_sample_count else "unavailable"
        measurements = {
            "base_com_max_abs_delta_xy": {
                "measurement_id": "base_com_max_abs_delta_xy",
                "unit": MEASUREMENT_DEFINITIONS["base_com_max_abs_delta_xy"]["unit"],
                "state": measurement_state, "value": overall_worst,
                "coverage_status": coverage_status,
            },
            "base_com_outside_range_fraction": {
                "measurement_id": "base_com_outside_range_fraction",
                "unit": MEASUREMENT_DEFINITIONS["base_com_outside_range_fraction"]["unit"],
                "state": measurement_state, "value": outside_fraction,
                "numerator": outside_count, "denominator": outside_denominator,
                "aggregation": "fraction_of_environments_whose_worst_max_abs_delta_xy_exceeds_threshold",
                "coverage_status": coverage_status,
            },
        }
        baseline_xy = [{
            "env_id": env_id,
            "x": self._baseline[env_id][self._base_index][0],
            "y": self._baseline[env_id][self._base_index][1],
        } for env_id in range(self._num_envs)]
        return deepcopy({
            "schema": "base_com_reset_observer_v1",
            "measurement_status": coverage_status,
            "source_binding": {
                **self._source_binding,
                "measurement_definitions_sha256": MEASUREMENT_DEFINITIONS_SHA256,
                "baseline_all_env_all_body_com_sha256": self._baseline_sha256,
            },
            "body_binding": {
                "body_names": list(self._body_names), "base_body_index": self._base_index,
                "baseline_shape": list(self._baseline_shape),
                "read_source": "robot.root_physx_view.get_coms()",
                "cached_body_com_pose_b_used": False,
            },
            "baseline_snapshot": {
                "body_names": list(self._body_names),
                "shape": list(self._baseline_shape), "coms": self._baseline,
            },
            "measurement_definitions": deepcopy(MEASUREMENT_DEFINITIONS),
            "sample_field_definitions": deepcopy(SAMPLE_FIELD_DEFINITIONS),
            "caller_provenance": {
                "baseline_capture_timing_claim": "after_environment_creation_before_first_reset",
                "baseline_capture_timing_verified_by_observer": False,
                "reset_completion_claim_source": "caller",
                "reset_completion_verified_by_observer": False,
            },
            "threshold": {
                "allowed_xy_range": self._allowed_xy_range, "tolerance": self._tolerance,
                "outside_rule": "max_abs_delta_xy_strictly_greater_than_range_plus_tolerance",
            },
            "coverage": {
                "expected_reset_sequence_count": self._expected_sequences,
                "observed_reset_sequence_count": len(self._records),
                "expected_resets_per_env": self._expected_resets_per_env,
                "expected_reset_measurement_count": expected_sample_count,
                "finite_reset_measurement_count": finite_sample_count,
                "unavailable_reset_sequences": unavailable_sequences,
                "status": coverage_status,
            },
            "baseline_base_xy": baseline_xy,
            "per_environment": per_environment,
            "measurements": measurements,
            "reset_records": self._records,
            "cause_confirmed": False, "health_confirmed": False,
            "recovery_confirmed": False, "cleanup_confirmed": False,
            "model_selected": False,
        })
