from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from rl_triage import grounded_evidence as G
from rl_triage import runtime_observers as R


SHA = "a" * 64


class RewardManager:
    def __init__(self):
        self.calls = []

    def get_active_iterable_terms(self, env_idx):
        self.calls.append(env_idx)
        return [("tracking", [10.0 + env_idx]), ("penalty", [-2.0])]

    def compute(self, dt):
        raise AssertionError("compute must not be called")

    def reset(self, env_ids=None):
        raise AssertionError("reset must not be called")


def fixture():
    robot_data = SimpleNamespace(
        root_com_pos_w=[[0.0, 0.0, 0.5], [0.0, 0.0, 1.5], [0.0, 0.0, 9.0]],
        root_com_lin_vel_w=[[0.0, 0.0, -1.0], [0.0, 0.0, 3.0], [0.0, 0.0, 9.0]],
        body_com_pos_b=[[[0.1, -0.2, 0.3], [0.0, 0.4, 0.0]],
                        [[0.0, 0.0, 0.6]], [[9.0, 9.0, 9.0]]],
    )
    forces = [[[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]],
              [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
              [[9.0, 9.0, 9.0]]]
    reward_manager = RewardManager()
    env = SimpleNamespace(
        num_envs=3,
        scene={"robot": SimpleNamespace(data=robot_data),
               "contact_forces": SimpleNamespace(data=SimpleNamespace(net_forces_w=forces))},
        reward_manager=reward_manager,
    )
    collector = R.RuntimeReadbackCollector(
        env, observed_env_indices=[0, 1], max_completed_control_samples=2,
        source_binding={
            "task_sha256": SHA, "checkpoint_sha256": "b" * 64,
            "configuration_sha256": "c" * 64, "observer_sha256": "d" * 64,
        },
    )
    return env, collector


def metadata(phase0="completed_control_no_reset", reset0=False):
    return {
        0: {"episode_id": 7, "reset_occurred": reset0, "phase": phase0},
        1: {"episode_id": 8, "reset_occurred": False, "phase": "completed_control_no_reset"},
    }


def observe(collector, *, control=4, physics=19, episodes=None, rewards=None):
    return collector.observe_completed_control_sample(
        step_metadata={"control_step_index": control, "physics_step_index": physics},
        episode_metadata=metadata() if episodes is None else episodes,
        completed_rewards={0: 2.0, 1: -1.0} if rewards is None else rewards,
    )


def test_registry_bound_aggregates_use_actual_completed_rewards_and_public_reward_readback():
    env, collector = fixture()
    observe(collector)
    result = collector.aggregate()

    assert result["source_binding"]["registry_sha256"] == R._canonical_sha(G.READ_ONLY_CHECKS)
    assert result["window_status"] == "observed"
    assert result["measurement_status"] == "complete"
    com = result["read_only_checks"]["runtime_com_state"]["measurements"]
    reward = result["read_only_checks"]["runtime_reward_window"]["measurements"]
    contact = result["read_only_checks"]["runtime_contact_window"]["measurements"]
    assert (com["com_position_z"]["value"], com["com_velocity_z"]["value"]) == (1.0, 1.0)
    assert com["body_com_local_position_max_abs"]["value"] == 0.6
    assert (reward["reward_mean"]["value"], reward["reward_min"]["value"]) == (0.5, -1.0)
    assert (contact["contact_force_max"]["value"], contact["contact_event_fraction"]["value"]) == (5.0, 0.5)
    assert all(check["measurement_status"] == "complete" for check in result["read_only_checks"].values())
    assert all(row["coverage_status"] == "complete"
               for check in result["read_only_checks"].values()
               for row in check["measurements"].values())
    assert env.reward_manager.calls == [0, 1]
    assert result["samples"][0]["environments"][0]["reward_terms"]["semantics"].startswith("weighted_pre_dt")
    assert result["cause_confirmed"] is result["health_confirmed"] is result["recovery_confirmed"] is False
    assert result["cleanup_confirmed"] is False


def test_samples_are_deep_cpu_snapshots_without_live_aliases():
    env, collector = fixture()
    sample = observe(collector)
    env.scene["robot"].data.root_com_pos_w[0][2] = 99.0
    env.scene["contact_forces"].data.net_forces_w[0][0][0] = 99.0
    sample["environments"][0]["measurements"]["com_position_z"]["value"] = -99.0

    stored = collector.aggregate()["samples"][0]["environments"][0]["measurements"]
    assert stored["com_position_z"]["value"] == 0.5
    assert stored["contact_force_max"]["value"] == 5.0


def test_post_auto_reset_contact_is_unavailable_and_phase_provenance_is_preserved():
    _, collector = fixture()
    episodes = metadata("post_auto_reset", True)
    observe(collector, episodes=episodes)
    result = collector.aggregate()
    assert result["measurement_status"] == "partial"
    first = result["samples"][0]["environments"][0]
    assert first["phase"] == "post_auto_reset"
    assert first["measurements"]["contact_force_max"] == {
        "state": "unavailable", "value": None, "reason": "post_auto_reset_state",
        "phase_provenance": "post_auto_reset"}
    assert first["measurements"]["com_position_z"]["phase_provenance"] == "post_auto_reset"
    assert first["measurements"]["completed_reward"]["phase_provenance"] == "completed_control_reward"
    assert first["measurements"]["completed_reward"]["environment_state_phase"] == "post_auto_reset"
    contact = result["read_only_checks"]["runtime_contact_window"]["measurements"]
    assert contact["contact_event_fraction"]["finite_value_count"] == 1
    assert contact["contact_event_fraction"]["coverage_status"] == "partial"
    assert result["read_only_checks"]["runtime_contact_window"]["measurement_status"] == "partial"
    assert contact["contact_event_fraction"]["values_by_environment_state_phase"]["post_auto_reset"] == {
        "expected_value_count": 1, "finite_value_count": 0, "unavailable_value_count": 1,
        "coverage_status": "unavailable",
        "unavailable_reason_counts": {"post_auto_reset_state": 1},
        "state": "unavailable", "value": None, "reason": "no_finite_measurement",
    }
    com = result["read_only_checks"]["runtime_com_state"]["measurements"]
    assert com["com_position_z"]["finite_value_count_by_environment_state_phase"] == {
        "completed_control_no_reset": 1, "post_auto_reset": 1}
    assert com["com_position_z"]["values_by_environment_state_phase"]["post_auto_reset"] == {
        "expected_value_count": 1, "finite_value_count": 1, "unavailable_value_count": 0,
        "coverage_status": "complete", "unavailable_reason_counts": {},
        "state": "present", "value": 0.5,
    }
    assert result["scope"]["sample_phases"] == ["completed_control_no_reset", "post_auto_reset"]


def test_missing_nonfinite_and_read_exception_stay_unavailable_without_exception_text():
    env, collector = fixture()
    env.scene["robot"].data.root_com_pos_w[0][2] = float("nan")
    del env.scene["contact_forces"]
    sample = observe(collector, rewards={0: None, 1: float("inf")})
    first = sample["environments"][0]["measurements"]
    assert first["com_position_z"]["state"] == "unavailable"
    assert first["completed_reward"]["state"] == "unavailable"
    assert first["contact_force_max"]["diagnostic"] == {"exception_type": "KeyError"}
    serialized = repr(sample)
    assert "contact_forces" not in serialized
    reward = collector.aggregate()["read_only_checks"]["runtime_reward_window"]["measurements"]
    assert reward["reward_mean"]["state"] == "unavailable"
    assert reward["reward_mean"]["value"] is None


def test_empty_window_is_not_observed_and_has_explicit_actual_step_scope():
    _, collector = fixture()
    result = collector.aggregate()
    assert result["window_status"] == "not_observed"
    assert result["measurement_status"] == "not_observed"
    assert result["scope"] == {
        "window_start": None, "window_end": None, "completed_control_sample_count": 0,
        "observed_env_indices": [0, 1], "sample_phases": [],
        "control_step_indices": [], "actual_physics_step_indices": [],
    }
    assert all(row["state"] == "unavailable"
               for check in result["read_only_checks"].values()
               for row in check["measurements"].values())
    assert all(row["coverage_status"] == "not_observed"
               for check in result["read_only_checks"].values()
               for row in check["measurements"].values())


@pytest.mark.parametrize("indices", [[0, 0], [0, 3], [True]])
def test_observed_environment_indices_reject_duplicates_out_of_range_and_bool(indices):
    env, _ = fixture()
    with pytest.raises(ValueError):
        R.RuntimeReadbackCollector(
            env, observed_env_indices=indices, max_completed_control_samples=1,
            source_binding={"task_sha256": SHA, "checkpoint_sha256": SHA,
                            "configuration_sha256": SHA, "observer_sha256": SHA})


@pytest.mark.parametrize("count", [True, 0, 1.5])
def test_sample_count_rejects_bool_zero_and_non_integer(count):
    env, _ = fixture()
    with pytest.raises(ValueError):
        R.RuntimeReadbackCollector(
            env, observed_env_indices=[0], max_completed_control_samples=count,
            source_binding={"task_sha256": SHA, "checkpoint_sha256": SHA,
                            "configuration_sha256": SHA, "observer_sha256": SHA})


def test_step_metadata_reset_alias_reward_mapping_and_ring_overflow_fail_closed():
    _, collector = fixture()
    with pytest.raises(ValueError, match="reset state"):
        observe(collector, episodes=metadata("completed_control_no_reset", True))
    with pytest.raises(ValueError, match="completed_rewards"):
        observe(collector, rewards={0: 1.0})
    with pytest.raises(ValueError, match="step indices"):
        collector.observe_completed_control_sample(
            step_metadata={"control_step_index": True, "physics_step_index": 1},
            episode_metadata=metadata(), completed_rewards={0: 1.0, 1: 2.0})
    observe(collector)
    observe(collector, control=5, physics=23)
    with pytest.raises(OverflowError, match="capacity"):
        observe(collector, control=6, physics=27)


def test_reward_mapping_validation_does_not_call_private_or_term_functions():
    env, collector = fixture()
    env.reward_manager.get_active_iterable_terms = lambda env_idx: [
        ("dup", [1.0]), ("dup", [2.0])]
    sample = observe(collector)
    terms = sample["environments"][0]["reward_terms"]
    assert terms["state"] == "unavailable"
    assert terms["diagnostic"] == {"exception_type": "InvalidRewardTermMapping"}
    assert "raw" not in repr(terms).lower()


def test_aggregate_result_is_a_copy():
    _, collector = fixture()
    observe(collector)
    result = collector.aggregate()
    altered = deepcopy(result)
    altered["samples"][0]["environments"][0]["episode_id"] = "changed"
    assert collector.aggregate()["samples"][0]["environments"][0]["episode_id"] == 7


@pytest.mark.parametrize("bad_sha", ["+" + "a" * 63, "-" + "a" * 63, " " + "a" * 63,
                                      "A" * 64, "g" * 64])
def test_source_binding_rejects_noncanonical_sha256_text(bad_sha):
    env, _ = fixture()
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        R.RuntimeReadbackCollector(
            env, observed_env_indices=[0], max_completed_control_samples=1,
            source_binding={"task_sha256": bad_sha, "checkpoint_sha256": SHA,
                            "configuration_sha256": SHA, "observer_sha256": SHA})


def test_unrepresentable_integer_and_large_contact_norm_fail_closed_without_overflow():
    env, collector = fixture()
    huge_integer = 10 ** 400
    env.scene["robot"].data.root_com_pos_w[0][2] = huge_integer
    env.scene["contact_forces"].data.net_forces_w[0][0] = [1.0e308, 0.0, 0.0]
    sample = observe(collector, rewards={0: huge_integer, 1: 1.0})
    first = sample["environments"][0]["measurements"]
    assert first["com_position_z"]["state"] == "unavailable"
    assert first["completed_reward"]["state"] == "unavailable"
    assert first["contact_force_max"]["value"] == 1.0e308


def test_large_finite_mean_stays_finite_and_strict_json_serializable():
    _, collector = fixture()
    observe(collector, rewards={0: 1.0e308, 1: 1.0e308})
    result = collector.aggregate()
    reward_mean = result["read_only_checks"]["runtime_reward_window"]["measurements"]["reward_mean"]
    assert reward_mean["state"] == "present"
    assert reward_mean["value"] == 1.0e308
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("bad_phase", [["completed_control_no_reset"], None, 1])
def test_unsupported_phase_types_are_rejected_without_type_error(bad_phase):
    _, collector = fixture()
    episodes = metadata()
    episodes[0]["phase"] = bad_phase
    with pytest.raises(ValueError, match="invalid reset flag or phase"):
        observe(collector, episodes=episodes)


def test_sample_with_all_measurements_unavailable_reports_unavailable_not_observed():
    env, collector = fixture()
    del env.scene["robot"]
    del env.scene["contact_forces"]
    env.reward_manager.get_active_iterable_terms = lambda env_idx: []
    result_sample = observe(collector, rewards={0: None, 1: None})
    assert result_sample["environments"]
    result = collector.aggregate()
    assert result["window_status"] == "observed"
    assert result["measurement_status"] == "unavailable"
    assert all(check["measurement_status"] == "unavailable"
               for check in result["read_only_checks"].values())
    assert all(row["expected_value_count"] == 2 and row["finite_value_count"] == 0
               and row["unavailable_value_count"] == 2 and row["coverage_status"] == "unavailable"
               for check in result["read_only_checks"].values()
               for row in check["measurements"].values())
