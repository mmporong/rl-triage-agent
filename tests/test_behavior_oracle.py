"""P1 행동 판정: 서 있는 기준·생존자 편향·조건 불일치·불필요 개입을 검사한다."""
from copy import deepcopy
import hashlib
import itertools
import json
from pathlib import Path

import pytest

from rl_triage import behavior_oracle as O

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_SHA = hashlib.sha256((ROOT / "bench/protocols/fixed_eval_v1.json").read_bytes().replace(b"\r\n", b"\n")).hexdigest()
GATES = {"min_survival_fraction": 0.9, "max_fall_rate": 0.05,
         "max_lin_vel_rmse_mps": 0.5, "max_yaw_rate_rmse_radps": 0.5}


def report(name="reference", checkpoint="a" * 64, fall=0.0, survival=20.0, lin=0.15, yaw=0.3,
           raw=True):
    grid = [list(c) for c in itertools.product((-1.0, 0.0, 1.0), repeat=3) if c != (0.0, 0.0, 0.0)]
    envs_per_condition = 40
    fall_count = round(fall * envs_per_condition)
    alive_steps_sum = round(survival / 0.02 * envs_per_condition)
    rows = []
    for command in grid:
        row = {"command": command, "fall_rate": fall, "lin_vel_rmse_mps": lin,
               "yaw_rate_rmse_radps": yaw}
        if raw:
            row.update({"num_envs": envs_per_condition, "fall_count": fall_count,
                        "alive_steps_sum": alive_steps_sum,
                        "lin_error_sq_sum": lin * lin * alive_steps_sum,
                        "yaw_error_sq_sum": yaw * yaw * alive_steps_sum})
        rows.append(row)
    return {"protocol": "fixed_eval_v1", "protocol_sha256_lf": PROTOCOL_SHA,
            "task": "Isaac-Velocity-Flat-Unitree-Go2-v0", "eval_seed": 2026, "num_envs": 1040,
            "horizon_steps": 1000, "step_dt": 0.02, "action_scale": 0.25, "name": name,
            "versions": {"torch": "2.7.0+cu128", "isaaclab": "0.41.3", "isaaclab_tasks": "0.10.36",
                         "isaaclab_rl": "0.1.8", "rsl-rl-lib": "2.3.3", "isaacsim": None},
            "checkpoint": {"file": "model_299.pt", "sha256": checkpoint},
            "metrics": {"fall_rate": fall, "mean_survival_s": survival,
                        "lin_vel_rmse_mps": lin, "yaw_rate_rmse_radps": yaw},
            "by_condition": rows}


def test_reward_magnitude_does_not_change_behavior_oracle():
    ref = report()
    contract = O.register_contract(ref, [], GATES)
    candidate = report("candidate", lin=0.8)
    first = O.assess(candidate, ref, contract)
    candidate["Train/mean_reward"] = 1e30
    assert O.assess(candidate, ref, contract) == first
    assert first["label"] == "unhealthy"


def test_reference_must_walk_and_gates_must_beat_standstill():
    with pytest.raises(O.OracleError, match="기준 정책"):
        O.register_contract(report(lin=1.175, yaw=0.823), [], GATES)
    with pytest.raises(O.OracleError, match="제자리"):
        O.register_contract(report(), [], {**GATES, "max_lin_vel_rmse_mps": 2.0})


def test_low_surviving_step_error_does_not_hide_falls_or_short_survival():
    ref = report()
    contract = O.register_contract(ref, [], GATES)
    fallen = report("fallen", fall=0.5, survival=11, lin=0.05, yaw=0.1)
    assessment = O.assess(fallen, ref, contract)
    assert assessment["label"] == "unhealthy"
    assert {"absolute.fall_rate", "absolute.survival_fraction"} <= set(assessment["failed"])


@pytest.mark.parametrize("change", [
    {"task": "another-task"}, {"eval_seed": 7}, {"num_envs": 1300}, {"horizon_steps": 1200},
    {"step_dt": 0.025}, {"protocol_sha256_lf": "0" * 64},
])
def test_mixed_fixed_eval_conditions_are_rejected(change):
    ref = report()
    contract = O.register_contract(ref, [], GATES)
    with pytest.raises(O.OracleError, match="일치하지"):
        O.assess({**report("candidate"), **change}, ref, contract)


def test_changed_grid_or_runtime_is_rejected_but_policy_interface_is_recorded():
    ref = report()
    contract = O.register_contract(ref, [], GATES)
    candidate = report("candidate")
    candidate["by_condition"][0]["command"][0] = -0.5
    with pytest.raises(O.OracleError, match="일치하지"):
        O.assess(candidate, ref, contract)
    candidate = report("candidate")
    candidate["versions"]["torch"] = "another-version"
    with pytest.raises(O.OracleError, match="일치하지"):
        O.assess(candidate, ref, contract)
    candidate = report("candidate")
    candidate["action_scale"] = 0.5
    assert O.assess(candidate, ref, contract)["action_scale"] == 0.5


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, True, None])
def test_invalid_metrics_fail_closed(value):
    ref = report()
    contract = O.register_contract(ref, [], GATES)
    candidate = report("candidate")
    candidate["metrics"]["lin_vel_rmse_mps"] = value
    with pytest.raises(O.OracleError):
        O.assess(candidate, ref, contract)


def test_incomplete_conditions_or_metrics_are_rejected():
    ref = report()
    contract = O.register_contract(ref, [], GATES)
    for candidate in (report("empty"), report("missing"), report("duplicate")):
        if candidate["name"] == "empty":
            candidate["by_condition"] = []
        elif candidate["name"] == "missing":
            del candidate["metrics"]["fall_rate"]
        else:
            candidate["by_condition"][1] = deepcopy(candidate["by_condition"][0])
        with pytest.raises(O.OracleError):
            O.assess(candidate, ref, contract)


def test_aggregate_and_condition_fall_rates_must_match_raw_counts():
    ref = report()
    contract = O.register_contract(ref, [], GATES)
    aggregate = report("aggregate")
    aggregate["metrics"]["fall_rate"] = 0.1
    with pytest.raises(O.OracleError, match="전체 fall_rate"):
        O.assess(aggregate, ref, contract)
    condition = report("condition")
    condition["by_condition"][0]["fall_rate"] = 1.0
    with pytest.raises(O.OracleError, match="조건별 fall_rate"):
        O.assess(condition, ref, contract)


def test_global_and_condition_rmse_must_match_raw_sums():
    ref = report()
    contract = O.register_contract(ref, [], GATES)
    aggregate = report("aggregate")
    aggregate["metrics"]["lin_vel_rmse_mps"] = 0.2
    with pytest.raises(O.OracleError, match="전체 lin_vel_rmse_mps"):
        O.assess(aggregate, ref, contract)
    condition = report("condition")
    condition["by_condition"][0]["yaw_rate_rmse_radps"] = 0.4
    with pytest.raises(O.OracleError, match="조건별 yaw_rate_rmse_radps"):
        O.assess(condition, ref, contract)


def test_raw_statistics_reject_tampering_partial_rows_and_invalid_counts():
    ref = report()
    contract = O.register_contract(ref, [], GATES)
    altered = report("altered")
    altered["by_condition"][0]["alive_steps_sum"] -= 1
    with pytest.raises(O.OracleError):
        O.assess(altered, ref, contract)
    partial = report("partial")
    del partial["by_condition"][0]["fall_count"]
    with pytest.raises(O.OracleError, match="일부만"):
        O.assess(partial, ref, contract)
    out_of_range = report("range")
    out_of_range["by_condition"][0]["fall_count"] = 41
    with pytest.raises(O.OracleError, match="범위"):
        O.assess(out_of_range, ref, contract)


def test_metrics_verified_requires_raw_candidate_reference_and_calibration():
    ref = report()
    contract = O.register_contract(ref, [report("normal")], GATES)
    assert contract["reference_metrics_verified"] is True
    assert contract["calibration_metrics_verified"] is True
    assert O.assess(report("candidate"), ref, contract)["metrics_verified"] is True
    assert O.assess(report("legacy-candidate", raw=False), ref, contract)["metrics_verified"] is False

    legacy_ref = report("legacy-ref", raw=False)
    legacy_contract = O.register_contract(legacy_ref, [], GATES)
    legacy_assessment = O.assess(report("legacy-candidate", raw=False), legacy_ref, legacy_contract)
    assert legacy_contract["reference_metrics_verified"] is False
    assert legacy_assessment["label"] == "healthy"
    assert legacy_assessment["metrics_verified"] is False

    mixed_contract = O.register_contract(ref, [report("legacy-normal", raw=False)], GATES)
    assert mixed_contract["calibration_metrics_verified"] is False
    assert O.assess(report("candidate"), ref, mixed_contract)["metrics_verified"] is False


@pytest.mark.parametrize("field, value, match", [
    ("fall_rate", 0.1, "전체 fall_rate"),
    ("lin_vel_rmse_mps", 0.9, "조건별 RMSE 범위"),
    ("mean_survival_s", 10.0, "생존 하한"),
])
def test_legacy_reports_still_reject_easy_aggregate_contradictions(field, value, match):
    ref = report(raw=False)
    contract = O.register_contract(ref, [], GATES)
    candidate = report("legacy", fall=0.5, survival=11, lin=0.15, yaw=0.3, raw=False)
    candidate["metrics"][field] = value
    with pytest.raises(O.OracleError, match=match):
        O.assess(candidate, ref, contract)


def test_large_or_overflowing_values_fail_closed():
    ref = report()
    contract = O.register_contract(ref, [], GATES)
    huge = report("huge")
    huge["horizon_steps"] = 10 ** 10000
    with pytest.raises(O.OracleError):
        O.assess(huge, ref, contract)
    overflow = report("overflow")
    for row in overflow["by_condition"]:
        row["lin_error_sq_sum"] = 1e308
        row["lin_vel_rmse_mps"] = (1e308 / row["alive_steps_sum"]) ** 0.5
    with pytest.raises(O.OracleError, match="합계"):
        O.assess(overflow, ref, contract)


def test_missing_result_abstention_recovery_and_unnecessary_intervention_are_distinct():
    ref = report()
    contract = O.register_contract(ref, [], GATES)
    bad = report("bad", lin=0.8)
    good = report("good", checkpoint="c" * 64)
    assert O.recovery_outcome(bad, good, ref, contract)["outcome"] == "recovered"
    assert O.recovery_outcome(bad, bad, ref, contract)["outcome"] == "not_recovered"
    assert O.recovery_outcome(good, good, ref, contract)["outcome"] == "unnecessary_intervention"
    assert O.recovery_outcome(good, bad, ref, contract)["outcome"] == "regression"
    assert O.recovery_outcome(bad, None, ref, contract)["outcome"] == "undetermined"
    abstained = O.recovery_outcome(good, None, ref, contract, intervened=False)
    assert abstained["outcome"] == "abstained" and not abstained["causal_confirmation"]
    with pytest.raises(O.OracleError):
        O.recovery_outcome(bad, good, ref, contract, intervened=False)


def test_reference_or_contract_changes_cannot_relax_the_gate():
    ref = report()
    contract = O.register_contract(ref, [], GATES)
    altered = deepcopy(contract)
    altered["absolute_gates"]["max_lin_vel_rmse_mps"] = 0.9
    with pytest.raises(O.OracleError, match="해시"):
        O.assess(report(), ref, altered)
    changed_ref = report(lin=0.16)
    with pytest.raises(O.OracleError, match="사전등록"):
        O.assess(report(), changed_ref, contract)
    with pytest.raises(O.OracleError, match="정상 보정"):
        O.register_contract(ref, [report(lin=1.0)], GATES)


def test_saved_walking_reference_is_compatible_and_old_standing_reference_is_rejected():
    folder = ROOT / "evals/results/p0c_calibration_20261009/runs"
    ref = json.loads((folder / "p0ccal_NONE_s7.json").read_text(encoding="utf-8"))
    contract = O.register_contract(ref, [], GATES)
    assessment = O.assess(ref, ref, contract)
    assert assessment["label"] == "healthy" and assessment["metrics_verified"] is False
    assert contract["reference_metrics_verified"] is False
    standing = json.loads((ROOT / "evals/results/p0b2_fixed_eval_20261008/runs/baseline_s42.json").read_text(encoding="utf-8"))
    with pytest.raises(O.OracleError, match="기준 정책"):
        O.register_contract(standing, [], GATES)
