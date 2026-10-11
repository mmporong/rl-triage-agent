"""Isaac Lab 상태를 바꾸지 않는 bounded runtime readback 수집기."""

from __future__ import annotations

import hashlib
import json
import math
import re
from copy import deepcopy
from typing import Mapping

from rl_triage.grounded_evidence import READ_ONLY_CHECKS


SOURCE_SHA_KEYS = {
    "task_sha256", "checkpoint_sha256", "configuration_sha256", "observer_sha256",
}
PHASES = {"completed_control_no_reset", "terminal_pre_reset", "post_auto_reset"}
MEASUREMENT_SOURCES = {
    "com_position_z": "scene.robot.data.root_com_pos_w[..., 2]",
    "com_velocity_z": "scene.robot.data.root_com_lin_vel_w[..., 2]",
    "body_com_local_position_max_abs": "scene.robot.data.body_com_pos_b",
    "reward_mean": "caller.completed_env_step_reward",
    "reward_min": "caller.completed_env_step_reward",
    "contact_force_max": "scene.contact_forces.data.net_forces_w",
    "contact_event_fraction": "norm(scene.contact_forces.data.net_forces_w) > 0",
}
AGGREGATIONS = {
    "com_position_z": "mean_over_observed_env_samples",
    "com_velocity_z": "mean_over_observed_env_samples",
    "body_com_local_position_max_abs": "max_over_observed_env_samples_and_bodies",
    "reward_mean": "mean_of_completed_env_step_rewards",
    "reward_min": "minimum_of_completed_env_step_rewards",
    "contact_force_max": "max_vector_norm_over_observed_env_samples_and_bodies",
    "contact_event_fraction": "fraction_of_claimable_env_samples_with_any_nonzero_contact_norm",
}


def _canonical_sha(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


REGISTRY_SHA256 = _canonical_sha(READ_ONLY_CHECKS)


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


def _cpu_value(value: object) -> object:
    """Detach and copy a tensor-like value without importing torch."""
    if value is None:
        return None
    current = value
    for method in ("detach", "cpu"):
        operation = getattr(current, method, None)
        if callable(operation):
            current = operation()
    to_list = getattr(current, "tolist", None)
    if callable(to_list):
        current = to_list()
    return deepcopy(current)


def _index(value: object, env_index: int) -> object:
    return _cpu_value(value[env_index])  # type: ignore[index]


def _nested_floats(value: object) -> list[float] | None:
    result: list[float] = []

    def visit(item: object) -> bool:
        if isinstance(item, (list, tuple)):
            return all(visit(child) for child in item)
        if not _finite(item):
            return False
        result.append(float(item))
        return True

    return result if visit(value) and result else None


def _measurement(value: object, *, diagnostic: dict | None = None) -> dict:
    if value is None:
        row = {"state": "unavailable", "value": None, "reason": "null_or_missing"}
    elif not _finite(value):
        row = {"state": "unavailable", "value": None, "reason": "nonfinite_or_malformed"}
    else:
        row = {"state": "present", "value": float(value)}
    if diagnostic is not None:
        row["diagnostic"] = deepcopy(diagnostic)
    return row


def _safe_mean(values: list[float]) -> float | None:
    """Average finite values without overflowing their intermediate sum."""
    if not values:
        return None
    scale = max(abs(value) for value in values)
    if scale == 0.0:
        return 0.0
    try:
        value = scale * (math.fsum(item / scale for item in values) / len(values))
    except (OverflowError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _aggregate_values(measurement_id: str, values: list[float], empty_reason: str) -> dict:
    if not values:
        return {"state": "unavailable", "value": None, "reason": empty_reason}
    if measurement_id in {"body_com_local_position_max_abs", "contact_force_max"}:
        return {"state": "present", "value": max(values)}
    if measurement_id == "reward_min":
        return {"state": "present", "value": min(values)}
    mean = _safe_mean(values)
    if mean is None:
        return {"state": "unavailable", "value": None, "reason": "nonfinite_aggregate"}
    return {"state": "present", "value": mean}


def _coverage_status(expected: int, finite: int) -> str:
    if expected == 0:
        return "not_observed"
    if finite == 0:
        return "unavailable"
    if finite < expected:
        return "partial"
    return "complete"


def _combined_status(statuses: list[str]) -> str:
    if not statuses or all(status == "not_observed" for status in statuses):
        return "not_observed"
    if all(status == "complete" for status in statuses):
        return "complete"
    if all(status == "unavailable" for status in statuses):
        return "unavailable"
    return "partial"


def _read(callable_) -> tuple[object, dict | None]:
    try:
        return callable_(), None
    except Exception as exc:  # Runtime adapters must fail closed without serializing private exception text.
        return None, {"exception_type": type(exc).__name__}


def _exact_env_mapping(value: object, env_indices: tuple[int, ...], label: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping keyed by observed environment index")
    keys = tuple(value.keys())
    if any(not _is_int(key) for key in keys) or set(keys) != set(env_indices):
        raise ValueError(f"{label} must map every observed environment index exactly once")
    return value


class RuntimeReadbackCollector:
    """Collect a fixed, bounded window from already-completed environment steps.

    The caller owns stepping and reset timing. This class never calls ``env.step``, reward
    compute/reset/term functions, physics, RNG, or configuration mutation APIs.
    """

    def __init__(self, env: object, *, observed_env_indices: list[int] | tuple[int, ...],
                 max_completed_control_samples: int, source_binding: Mapping[str, str]):
        if not _is_int(max_completed_control_samples) or max_completed_control_samples <= 0:
            raise ValueError("max_completed_control_samples must be a positive integer")
        if not isinstance(observed_env_indices, (list, tuple)) or not observed_env_indices:
            raise ValueError("observed_env_indices must be a non-empty explicit list")
        if any(not _is_int(index) for index in observed_env_indices):
            raise ValueError("observed environment indices must be integers, not booleans")
        indices = tuple(observed_env_indices)
        if len(set(indices)) != len(indices):
            raise ValueError("observed environment indices contain duplicates")
        num_envs = getattr(env, "num_envs", None)
        if not _is_int(num_envs) or num_envs <= 0:
            raise ValueError("env.num_envs must be a positive integer")
        if any(index < 0 or index >= num_envs for index in indices):
            raise ValueError("observed environment index is out of range")
        if not isinstance(source_binding, Mapping) or set(source_binding) != SOURCE_SHA_KEYS:
            raise ValueError("source_binding must contain task/checkpoint/configuration/observer SHA-256 values")
        self._env = env
        self._indices = indices
        self._max_samples = max_completed_control_samples
        self._source_binding = {key: _sha256(source_binding[key], key) for key in sorted(SOURCE_SHA_KEYS)}
        self._samples: list[dict] = []

    @property
    def sample_count(self) -> int:
        return len(self._samples)

    def observe_completed_control_sample(
        self, *, step_metadata: Mapping[str, int], episode_metadata: Mapping[int, Mapping[str, object]],
        completed_rewards: Mapping[int, object],
    ) -> dict:
        """Snapshot one completed control sample using caller-supplied clocks and reset phase."""
        if len(self._samples) >= self._max_samples:
            raise OverflowError("completed control sample ring capacity exceeded")
        if not isinstance(step_metadata, Mapping) or set(step_metadata) != {
                "control_step_index", "physics_step_index"}:
            raise ValueError("step_metadata must contain exact control and physics step indices")
        control_step = step_metadata["control_step_index"]
        physics_step = step_metadata["physics_step_index"]
        if not _is_int(control_step) or not _is_int(physics_step) or control_step < 0 or physics_step < 0:
            raise ValueError("step indices must be non-negative integers, not booleans")
        if self._samples:
            previous = self._samples[-1]
            if control_step <= previous["control_step_index"] or physics_step <= previous["physics_step_index"]:
                raise ValueError("caller-supplied control and physics step indices must increase")
        episodes = _exact_env_mapping(episode_metadata, self._indices, "episode_metadata")
        rewards = _exact_env_mapping(completed_rewards, self._indices, "completed_rewards")

        robot_data, robot_diagnostic = _read(lambda: self._env.scene["robot"].data)
        contact_data, contact_diagnostic = _read(lambda: self._env.scene["contact_forces"].data.net_forces_w)
        rows = []
        for env_index in self._indices:
            metadata = episodes[env_index]
            if not isinstance(metadata, Mapping) or set(metadata) != {"episode_id", "reset_occurred", "phase"}:
                raise ValueError("each episode metadata row must contain episode_id, reset_occurred, and phase")
            episode_id = metadata["episode_id"]
            reset_occurred = metadata["reset_occurred"]
            phase = metadata["phase"]
            if isinstance(episode_id, bool) or not isinstance(episode_id, (int, str)) or episode_id == "":
                raise ValueError("episode_id must be a non-empty string or integer")
            if type(reset_occurred) is not bool or not isinstance(phase, str) or phase not in PHASES:
                raise ValueError("reset metadata contains an invalid reset flag or phase")
            if (reset_occurred and phase == "completed_control_no_reset") or (
                    not reset_occurred and phase != "completed_control_no_reset"):
                raise ValueError("reset metadata cannot alias reset state as a no-reset sample")

            measurements: dict[str, dict] = {}
            if robot_data is None:
                for name in ("com_position_z", "com_velocity_z", "body_com_local_position_max_abs"):
                    measurements[name] = _measurement(None, diagnostic=robot_diagnostic)
            else:
                pos, pos_diag = _read(lambda: _index(robot_data.root_com_pos_w, env_index))
                vel, vel_diag = _read(lambda: _index(robot_data.root_com_lin_vel_w, env_index))
                body, body_diag = _read(lambda: _index(robot_data.body_com_pos_b, env_index))
                pos_values = _nested_floats(pos)
                vel_values = _nested_floats(vel)
                body_values = _nested_floats(body)
                measurements["com_position_z"] = _measurement(
                    pos_values[2] if pos_values is not None and len(pos_values) >= 3 else None,
                    diagnostic=pos_diag,
                )
                measurements["com_velocity_z"] = _measurement(
                    vel_values[2] if vel_values is not None and len(vel_values) >= 3 else None,
                    diagnostic=vel_diag,
                )
                measurements["body_com_local_position_max_abs"] = _measurement(
                    max(map(abs, body_values)) if body_values is not None else None,
                    diagnostic=body_diag,
                )

            for name in ("com_position_z", "com_velocity_z", "body_com_local_position_max_abs"):
                measurements[name]["phase_provenance"] = phase

            reward = rewards[env_index]
            measurements["completed_reward"] = _measurement(reward)
            measurements["completed_reward"]["phase_provenance"] = "completed_control_reward"
            measurements["completed_reward"]["environment_state_phase"] = phase

            reward_terms, terms_diagnostic = _read(
                lambda: self._env.reward_manager.get_active_iterable_terms(env_index))
            term_rows = []
            if reward_terms is not None:
                try:
                    seen = set()
                    for item in reward_terms:  # type: ignore[union-attr]
                        if not isinstance(item, (list, tuple)) or len(item) != 2:
                            raise ValueError
                        name, values = item
                        if not isinstance(name, str) or not name or name in seen:
                            raise ValueError
                        seen.add(name)
                        flattened = _nested_floats(_cpu_value(values))
                        if flattened is None:
                            term_rows.append({"term": name, "state": "unavailable", "values": None})
                        else:
                            term_rows.append({"term": name, "state": "present", "values": flattened})
                except (TypeError, ValueError):
                    term_rows = []
                    terms_diagnostic = {"exception_type": "InvalidRewardTermMapping"}
            terms_state = "present" if term_rows and terms_diagnostic is None else "unavailable"

            if phase == "post_auto_reset":
                measurements["contact_force_max"] = {
                    "state": "unavailable", "value": None, "reason": "post_auto_reset_state",
                }
                measurements["contact_present"] = {
                    "state": "unavailable", "value": None, "reason": "post_auto_reset_state",
                }
            else:
                forces, forces_diag = (None, contact_diagnostic) if contact_data is None else _read(
                    lambda: _index(contact_data, env_index))
                force_values = _nested_floats(forces)
                if force_values is None or len(force_values) % 3:
                    measurements["contact_force_max"] = _measurement(None, diagnostic=forces_diag)
                    measurements["contact_present"] = _measurement(None, diagnostic=forces_diag)
                else:
                    norms = [math.hypot(*(force_values[offset + axis] for axis in range(3)))
                             for offset in range(0, len(force_values), 3)]
                    measurements["contact_force_max"] = _measurement(max(norms))
                    measurements["contact_present"] = _measurement(float(any(norm > 0.0 for norm in norms)))
            measurements["contact_force_max"]["phase_provenance"] = phase
            measurements["contact_present"]["phase_provenance"] = phase

            rows.append({
                "env_index": env_index, "episode_id": deepcopy(episode_id),
                "reset_occurred": reset_occurred, "phase": phase,
                "measurements": measurements,
                "reward_terms": {
                    "state": terms_state,
                    "semantics": "weighted_pre_dt_step_reward_from_public_reward_manager_readback",
                    "values": term_rows if terms_state == "present" else None,
                    **({"diagnostic": terms_diagnostic} if terms_diagnostic is not None else {}),
                },
            })
        sample = {
            "control_step_index": control_step, "physics_step_index": physics_step,
            "observed_env_indices": list(self._indices), "environments": rows,
        }
        self._samples.append(deepcopy(sample))
        return deepcopy(sample)

    def aggregate(self) -> dict:
        """Return registry-bound window aggregates and immutable CPU sample copies."""
        common_scope = {
            "window_start": None if not self._samples else {
                "control_step_index": self._samples[0]["control_step_index"],
                "physics_step_index": self._samples[0]["physics_step_index"],
            },
            "window_end": None if not self._samples else {
                "control_step_index": self._samples[-1]["control_step_index"],
                "physics_step_index": self._samples[-1]["physics_step_index"],
            },
            "completed_control_sample_count": len(self._samples),
            "observed_env_indices": list(self._indices),
            "sample_phases": sorted({row["phase"] for sample in self._samples
                                      for row in sample["environments"]}),
            "control_step_indices": [sample["control_step_index"] for sample in self._samples],
            "actual_physics_step_indices": [sample["physics_step_index"] for sample in self._samples],
        }
        raw: dict[str, list[float]] = {name: [] for name in MEASUREMENT_SOURCES}
        raw_by_phase: dict[str, dict[str, list[float]]] = {
            name: {phase: [] for phase in PHASES} for name in MEASUREMENT_SOURCES
        }
        expected_by_phase: dict[str, dict[str, int]] = {
            name: {phase: 0 for phase in PHASES} for name in MEASUREMENT_SOURCES
        }
        unavailable_reasons: dict[str, dict[str, int]] = {name: {} for name in MEASUREMENT_SOURCES}
        unavailable_reasons_by_phase: dict[str, dict[str, dict[str, int]]] = {
            name: {phase: {} for phase in PHASES} for name in MEASUREMENT_SOURCES
        }
        unavailable = {name: 0 for name in MEASUREMENT_SOURCES}
        for sample in self._samples:
            for row in sample["environments"]:
                samples = row["measurements"]
                mapping = {
                    "com_position_z": samples["com_position_z"],
                    "com_velocity_z": samples["com_velocity_z"],
                    "body_com_local_position_max_abs": samples["body_com_local_position_max_abs"],
                    "reward_mean": samples["completed_reward"],
                    "reward_min": samples["completed_reward"],
                    "contact_force_max": samples["contact_force_max"],
                    "contact_event_fraction": samples["contact_present"],
                }
                for name, measurement in mapping.items():
                    phase = row["phase"]
                    expected_by_phase[name][phase] += 1
                    if measurement["state"] == "present":
                        raw[name].append(measurement["value"])
                        raw_by_phase[name][phase].append(measurement["value"])
                    else:
                        unavailable[name] += 1
                        reason = measurement.get("reason", "read_unavailable")
                        unavailable_reasons[name][reason] = unavailable_reasons[name].get(reason, 0) + 1
                        phase_reasons = unavailable_reasons_by_phase[name][phase]
                        phase_reasons[reason] = phase_reasons.get(reason, 0) + 1

        registry_results = {}
        for registry_id, registry in READ_ONLY_CHECKS.items():
            result_rows = {}
            for measurement_id, definition in registry["measurements"].items():
                values = raw[measurement_id]
                result = _aggregate_values(
                    measurement_id, values,
                    "not_observed" if not self._samples else "no_finite_measurement",
                )
                expected_count = sum(expected_by_phase[measurement_id].values())
                coverage_status = _coverage_status(expected_count, len(values))
                phase_value_counts = {
                    phase: len(raw_by_phase[measurement_id][phase])
                    for phase in common_scope["sample_phases"]
                }
                values_by_phase = {
                    phase: {
                        "expected_value_count": expected_by_phase[measurement_id][phase],
                        "finite_value_count": len(phase_values),
                        "unavailable_value_count": (
                            expected_by_phase[measurement_id][phase] - len(phase_values)
                        ),
                        "coverage_status": _coverage_status(
                            expected_by_phase[measurement_id][phase], len(phase_values)),
                        "unavailable_reason_counts": deepcopy(
                            unavailable_reasons_by_phase[measurement_id][phase]),
                        **_aggregate_values(measurement_id, phase_values, "no_finite_measurement"),
                    }
                    for phase, phase_values in raw_by_phase[measurement_id].items()
                    if expected_by_phase[measurement_id][phase] > 0
                }
                result_rows[measurement_id] = {
                    "measurement_id": measurement_id, "unit": definition["unit"],
                    "source": MEASUREMENT_SOURCES[measurement_id],
                    "aggregation": AGGREGATIONS[measurement_id],
                    "expected_value_count": expected_count,
                    "finite_value_count": len(values), "unavailable_value_count": unavailable[measurement_id],
                    "coverage_status": coverage_status,
                    "unavailable_reason_counts": deepcopy(unavailable_reasons[measurement_id]),
                    "finite_value_count_by_environment_state_phase": phase_value_counts,
                    "values_by_environment_state_phase": values_by_phase,
                    **result,
                }
            check_status = _combined_status([
                row["coverage_status"] for row in result_rows.values()
            ])
            registry_results[registry_id] = {
                "registry_id": registry_id, "adapter_id": registry["adapter_id"],
                "safety": registry["safety"], "source": "runtime_readback_collector",
                "measurement_status": check_status,
                "scope": deepcopy(common_scope), "measurements": result_rows,
            }
        observed = bool(self._samples)
        measurement_status = _combined_status([
            check["measurement_status"] for check in registry_results.values()
        ])
        return deepcopy({
            "schema": "runtime_readback_v1",
            "window_status": "observed" if observed else "not_observed",
            "measurement_status": measurement_status,
            "source_binding": {**self._source_binding, "registry_sha256": REGISTRY_SHA256},
            "scope": common_scope, "read_only_checks": registry_results,
            "samples": self._samples,
            "cause_confirmed": False, "health_confirmed": False, "recovery_confirmed": False,
            "cleanup_confirmed": False,
            "cleanup_confirmation_required_from_parent_receipt": True,
        })
