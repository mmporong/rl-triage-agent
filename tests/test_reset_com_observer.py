from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from rl_triage import reset_com_observer as R


SHA = "a" * 64
SOURCE = {
    "task_sha256": SHA,
    "configuration_sha256": "b" * 64,
    "isaaclab_sha256": "c" * 64,
    "observer_sha256": "d" * 64,
}


class FakePhysxView:
    def __init__(self, values):
        self.values = deepcopy(values)
        self.get_calls = 0
        self.raise_next = None
        self.write_calls = 0

    def get_coms(self):
        self.get_calls += 1
        if self.raise_next is not None:
            error = self.raise_next
            self.raise_next = None
            raise error
        return deepcopy(self.values)

    def set_coms(self, *args, **kwargs):
        self.write_calls += 1
        raise AssertionError("observer must not write COM state")


class ForbiddenEnvironment:
    def reset(self, *args, **kwargs):
        raise AssertionError("observer must not reset the environment")

    def step(self, *args, **kwargs):
        raise AssertionError("observer must not step the environment")


class BadCpuCopy:
    def tolist(self):
        raise RuntimeError("private tensor conversion detail")


def make_env(num_envs=4, body_names=("base", "leg")):
    values = []
    for env_id in range(num_envs):
        values.append([
            [env_id * 0.01, -env_id * 0.01, 0.2, 1.0, 0.0, 0.0, 0.0],
            [100.0 + env_id, -100.0 - env_id, 1.0, 1.0, 0.0, 0.0, 0.0],
        ][:len(body_names)])
    view = FakePhysxView(values)
    robot = SimpleNamespace(
        body_names=list(body_names), root_physx_view=view,
        data=SimpleNamespace(body_com_pose_b="cached_value_must_not_be_used"),
    )
    env = ForbiddenEnvironment()
    env.num_envs = num_envs
    env.scene = {"robot": robot}
    return env, robot, view


def observer(env, **kwargs):
    return R.BaseComResetObserver(env, source_binding=SOURCE, **kwargs)


def set_base_delta(view, env_ids, baseline, *, dx, dy=0.0):
    for env_id in env_ids:
        view.values[env_id][0][0] = baseline[env_id][0] + dx
        view.values[env_id][0][1] = baseline[env_id][1] + dy


def baseline_xy(view):
    return [(row[0][0], row[0][1]) for row in view.values]


def test_constructor_captures_immutable_all_env_all_body_baseline_and_source_binding():
    env, _, view = make_env()
    initial = deepcopy(view.values)
    collector = observer(env)
    first = collector.aggregate()
    view.values[0][0][0] = 9.0
    second = collector.aggregate()

    assert view.get_calls == 1
    assert first["baseline_snapshot"] == {
        "body_names": ["base", "leg"], "shape": [4, 2, 7], "coms": initial,
    }
    assert R._canonical_sha(first["baseline_snapshot"]) == first["source_binding"]["baseline_all_env_all_body_com_sha256"]
    first["baseline_snapshot"]["coms"][0][0][0] = 999.0
    assert second["baseline_snapshot"]["coms"] == initial
    assert first["source_binding"] == second["source_binding"]
    assert first["source_binding"]["baseline_all_env_all_body_com_sha256"] == R._canonical_sha({
        "body_names": ("base", "leg"), "shape": (4, 2, 7), "coms": initial,
    })
    assert second["baseline_base_xy"][0] == {"env_id": 0, "x": 0.0, "y": 0.0}
    assert first["caller_provenance"]["baseline_capture_timing_verified_by_observer"] is False
    assert first["measurement_status"] == "not_observed"


def test_record_reads_fresh_physx_base_only_and_other_body_offset_cannot_hide_it():
    env, robot, view = make_env()
    collector = observer(env, expected_reset_sequences=1, expected_resets_per_env=1)
    view.values[0][0][0] = 0.02
    view.values[0][0][1] = -0.03
    view.values[0][1][0] = 1000000.0
    robot.data.body_com_pose_b = [[[999999.0, 999999.0, 999999.0]]]
    record = collector.record_completed_reset(reset_sequence=0, reset_env_ids=[0])
    result = collector.aggregate()

    assert view.get_calls == 2
    assert view.write_calls == 0
    assert record["samples"][0] == {
        "env_id": 0, "baseline_x": 0.0, "baseline_y": 0.0,
        "current_x": 0.02, "current_y": -0.03,
        "delta_x": 0.02, "delta_y": -0.03,
        "max_abs_delta_xy": 0.03, "outside_allowed_range": False,
    }
    assert result["measurements"]["base_com_max_abs_delta_xy"]["value"] == 0.03
    assert result["body_binding"]["cached_body_com_pose_b_used"] is False
    assert result["body_binding"]["read_source"] == "robot.root_physx_view.get_coms()"
    assert result["measurement_definitions"] == R.MEASUREMENT_DEFINITIONS
    assert set(result["measurements"]) == set(result["measurement_definitions"])
    for measurement_id, measurement in result["measurements"].items():
        assert measurement["unit"] == result["measurement_definitions"][measurement_id]["unit"]
    for field in ("delta_x", "delta_y"):
        assert field in record["samples"][0]
        assert result["sample_field_definitions"][field] == {
            "unit": "simulation_length_native", "scope": "reset_records.samples",
        }


def test_standard_64_env_twenty_alternating_resets_are_complete_with_ten_per_env():
    env, _, view = make_env(num_envs=64)
    baseline = baseline_xy(view)
    collector = observer(env)
    for sequence in range(20):
        ids = list(range(sequence % 2, 64, 2))
        set_base_delta(view, ids, baseline, dx=0.04, dy=-0.02)
        collector.record_completed_reset(reset_sequence=sequence, reset_env_ids=ids)
    result = collector.aggregate()

    assert result["measurement_status"] == "complete"
    assert result["coverage"] == {
        "expected_reset_sequence_count": 20,
        "observed_reset_sequence_count": 20,
        "expected_resets_per_env": 10,
        "expected_reset_measurement_count": 640,
        "finite_reset_measurement_count": 640,
        "unavailable_reset_sequences": [],
        "status": "complete",
    }
    assert {row["declared_reset_count"] for row in result["per_environment"]} == {10}
    assert {row["finite_reset_measurement_count"] for row in result["per_environment"]} == {10}
    assert result["measurements"]["base_com_outside_range_fraction"]["value"] == 0.0


def test_additive_and_reset_from_baseline_patterns_remain_raw_measurement_differences():
    ref_env, _, ref_view = make_env()
    ref_baseline = baseline_xy(ref_view)
    reference = observer(ref_env)
    for sequence in range(20):
        ids = list(range(sequence % 2, 4, 2))
        set_base_delta(ref_view, ids, ref_baseline, dx=0.04)
        reference.record_completed_reset(reset_sequence=sequence, reset_env_ids=ids)

    add_env, _, add_view = make_env()
    additive = observer(add_env)
    for sequence in range(20):
        ids = list(range(sequence % 2, 4, 2))
        for env_id in ids:
            add_view.values[env_id][0][0] += 0.01
        additive.record_completed_reset(reset_sequence=sequence, reset_env_ids=ids)

    reference_result = reference.aggregate()
    additive_result = additive.aggregate()
    assert reference_result["measurements"]["base_com_max_abs_delta_xy"]["value"] == pytest.approx(0.04)
    assert additive_result["measurements"]["base_com_max_abs_delta_xy"]["value"] == pytest.approx(0.10)
    assert reference_result["measurements"]["base_com_outside_range_fraction"]["value"] == 0.0
    assert additive_result["measurements"]["base_com_outside_range_fraction"]["value"] > 0.0
    for result in (reference_result, additive_result):
        assert result["cause_confirmed"] is result["health_confirmed"] is False
        assert result["recovery_confirmed"] is result["cleanup_confirmed"] is False
        assert result["model_selected"] is False


def test_outside_range_uses_strict_range_plus_tolerance_boundary():
    env, _, view = make_env(num_envs=2)
    baseline = baseline_xy(view)
    collector = observer(env, expected_reset_sequences=2, expected_resets_per_env=1)
    threshold = 0.05 + 1.0e-4
    set_base_delta(view, [0], baseline, dx=threshold)
    first = collector.record_completed_reset(reset_sequence=0, reset_env_ids=[0])
    set_base_delta(view, [1], baseline, dx=threshold + 1.0e-6)
    second = collector.record_completed_reset(reset_sequence=1, reset_env_ids=[1])

    assert first["samples"][0]["outside_allowed_range"] is False
    assert second["samples"][0]["outside_allowed_range"] is True
    result = collector.aggregate()
    assert result["measurements"]["base_com_outside_range_fraction"]["value"] == 0.5
    assert result["threshold"] == {
        "allowed_xy_range": 0.05, "tolerance": 1.0e-4,
        "outside_rule": "max_abs_delta_xy_strictly_greater_than_range_plus_tolerance",
    }


def test_outside_fraction_uses_per_environment_worst_not_reset_sample_fraction():
    env, _, view = make_env(num_envs=2)
    baseline = baseline_xy(view)
    collector = observer(env, expected_reset_sequences=4, expected_resets_per_env=2)
    steps = [(0, [0], 0.06), (1, [1], 0.01), (2, [0], 0.01), (3, [1], 0.01)]
    for sequence, ids, delta in steps:
        set_base_delta(view, ids, baseline, dx=delta)
        collector.record_completed_reset(reset_sequence=sequence, reset_env_ids=ids)
    measurement = collector.aggregate()["measurements"]["base_com_outside_range_fraction"]

    assert measurement["value"] == 0.5
    assert measurement["numerator"] == 1 and measurement["denominator"] == 2
    assert measurement["aggregation"].startswith("fraction_of_environments")


@pytest.mark.parametrize("body_names", [(), ("leg",), ("base", "base")])
def test_missing_duplicate_or_empty_base_binding_is_rejected(body_names):
    env, _, _ = make_env(body_names=body_names)
    with pytest.raises(ValueError, match="body_names|exactly one base"):
        observer(env)


def test_body_order_and_shape_drift_are_rejected():
    env, robot, view = make_env()
    collector = observer(env)
    robot.body_names[:] = ["leg", "base"]
    with pytest.raises(ValueError, match="body order"):
        collector.record_completed_reset(reset_sequence=0, reset_env_ids=[0, 2])

    env, _, view = make_env()
    collector = observer(env)
    view.values[0].append([0.0] * 7)
    with pytest.raises(ValueError, match="body dimension"):
        collector.record_completed_reset(reset_sequence=0, reset_env_ids=[0, 2])


@pytest.mark.parametrize("ids, message", [
    ([], "non-empty"),
    ([True], "not booleans"),
    ([0, 0], "duplicates"),
    ([-1], "out-of-range"),
    ([4], "out-of-range"),
])
def test_reset_environment_ids_are_strictly_validated(ids, message):
    env, _, _ = make_env()
    collector = observer(env)
    with pytest.raises(ValueError, match=message):
        collector.record_completed_reset(reset_sequence=0, reset_env_ids=ids)


def test_reset_sequence_must_be_monotonic_bounded_and_non_boolean():
    env, _, _ = make_env()
    collector = observer(env, expected_reset_sequences=2, expected_resets_per_env=1)
    with pytest.raises(ValueError, match="not a boolean"):
        collector.record_completed_reset(reset_sequence=True, reset_env_ids=[0])
    collector.record_completed_reset(reset_sequence=4, reset_env_ids=[0, 2])
    with pytest.raises(ValueError, match="exactly one"):
        collector.record_completed_reset(reset_sequence=6, reset_env_ids=[1, 3])
    collector.record_completed_reset(reset_sequence=5, reset_env_ids=[1, 3])
    with pytest.raises(OverflowError, match="capacity"):
        collector.record_completed_reset(reset_sequence=6, reset_env_ids=[0, 2])


def test_missing_reset_subset_is_partial_and_cannot_be_success():
    env, _, view = make_env(num_envs=64)
    baseline = baseline_xy(view)
    collector = observer(env)
    for sequence in range(20):
        ids = list(range(sequence % 2, 64, 2))
        if sequence == 0:
            ids.remove(0)
        set_base_delta(view, ids, baseline, dx=0.01)
        collector.record_completed_reset(reset_sequence=sequence, reset_env_ids=ids)
    result = collector.aggregate()

    assert result["measurement_status"] == "partial"
    assert result["per_environment"][0]["declared_reset_count"] == 9
    assert result["coverage"]["finite_reset_measurement_count"] == 639
    assert result["cause_confirmed"] is False
    assert result["health_confirmed"] is False


def test_unavailable_and_nonfinite_reads_are_recorded_without_exception_text():
    env, _, view = make_env()
    collector = observer(env, expected_reset_sequences=2, expected_resets_per_env=1)
    view.raise_next = RuntimeError("private source path and secret")
    first = collector.record_completed_reset(reset_sequence=0, reset_env_ids=[0, 2])
    view.values[1][1][0] = float("nan")
    second = collector.record_completed_reset(reset_sequence=1, reset_env_ids=[1, 3])
    result = collector.aggregate()

    assert first["diagnostic"] == {"exception_type": "RuntimeError"}
    assert second["diagnostic"] == {"exception_type": "NonFiniteComReadback"}
    assert "secret" not in repr(result)
    assert result["measurement_status"] == "unavailable"
    assert result["measurements"]["base_com_outside_range_fraction"]["value"] is None
    assert result["measurements"]["base_com_outside_range_fraction"]["denominator"] == 0


def test_finite_inputs_with_overflowing_delta_are_unavailable_and_strict_json_safe():
    env, _, view = make_env()
    view.values[0][0][0] = -1.0e308
    collector = observer(env, expected_reset_sequences=1, expected_resets_per_env=1)
    view.values[0][0][0] = 1.0e308
    record = collector.record_completed_reset(reset_sequence=0, reset_env_ids=[0])
    result = collector.aggregate()

    assert record["measurement_state"] == "unavailable"
    assert record["samples"] == []
    assert record["diagnostic"] == {"exception_type": "NonFiniteDerivedComDelta"}
    assert result["measurement_status"] == "unavailable"
    assert result["measurements"]["base_com_max_abs_delta_xy"]["value"] is None
    assert result["per_environment"][0]["worst_max_abs_delta_xy"] is None
    json.dumps(result, allow_nan=False)


def test_cpu_copy_failure_is_unavailable_without_private_exception_text():
    env, _, view = make_env()
    collector = observer(env, expected_reset_sequences=1, expected_resets_per_env=1)
    view.values = BadCpuCopy()
    record = collector.record_completed_reset(reset_sequence=0, reset_env_ids=[0])
    result = collector.aggregate()

    assert record == {
        "reset_sequence": 0, "reset_env_ids": [0],
        "measurement_state": "unavailable", "samples": [],
        "diagnostic": {"exception_type": "RuntimeError"},
    }
    assert "private tensor" not in repr(result)
    assert result["measurement_status"] == "unavailable"


def test_observer_does_not_mutate_com_state_and_aggregate_returns_a_copy():
    env, robot, view = make_env()
    collector = observer(env, expected_reset_sequences=1, expected_resets_per_env=1)
    before = deepcopy(view.values)
    collector.record_completed_reset(reset_sequence=0, reset_env_ids=[0])
    result = collector.aggregate()
    result["reset_records"][0]["samples"][0]["current_x"] = 99.0

    assert view.values == before
    assert view.write_calls == 0
    assert robot.data.body_com_pose_b == "cached_value_must_not_be_used"
    assert collector.aggregate()["reset_records"][0]["samples"][0]["current_x"] == 0.0


@pytest.mark.parametrize("field,value", [
    ("task_sha256", "A" * 64),
    ("configuration_sha256", "g" * 64),
    ("isaaclab_sha256", "short"),
    ("observer_sha256", None),
])
def test_source_binding_requires_exact_lowercase_sha256(field, value):
    env, _, _ = make_env()
    source = dict(SOURCE)
    source[field] = value
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        R.BaseComResetObserver(env, source_binding=source)


@pytest.mark.parametrize("kwargs", [
    {"expected_reset_sequences": True},
    {"expected_reset_sequences": 0},
    {"expected_resets_per_env": True},
    {"expected_resets_per_env": 0},
    {"allowed_xy_range": float("nan")},
    {"tolerance": -1.0},
])
def test_constructor_rejects_invalid_counts_and_thresholds(kwargs):
    env, _, _ = make_env()
    with pytest.raises(ValueError):
        observer(env, **kwargs)
