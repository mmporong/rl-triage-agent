"""동일 동결 정책 3-arm 고정 평가 본문 검증과 회복 판정 회귀."""
import copy
import math

import pytest

from rl_triage import grounded_recovery as GR


SHA = "1" * 64
PROTO_SHA = "2" * 64
VERSIONS = {"torch": "2.7", "gpu": "GPU", "isaaclab": "x", "isaaclab_tasks": "x", "isaaclab_rl": "x",
            "rsl-rl-lib": "x", "isaacsim": None}


def _report(name, *, lin=0.2, yaw=0.2, fall_count=0, alive=40000, checkpoint=SHA, scale=0.25):
    rows = []
    for command in GR.GRID:
        lin_sq = lin * lin * alive
        yaw_sq = yaw * yaw * alive
        rows.append({"command": list(command), "fall_rate": fall_count / 40,
                     "lin_vel_rmse_mps": lin, "yaw_rate_rmse_radps": yaw, "num_envs": 40,
                     "fall_count": fall_count, "alive_steps_sum": alive,
                     "lin_error_sq_sum": lin_sq, "yaw_error_sq_sum": yaw_sq})
    total_alive = alive * len(rows)
    return {"protocol": GR.PROTOCOL, "protocol_sha256_lf": PROTO_SHA, "task": GR.TASK,
            "eval_seed": GR.EVAL_SEED, "num_envs": GR.NUM_ENVS, "horizon_steps": GR.HORIZON_STEPS,
            "step_dt": GR.STEP_DT, "action_scale": scale, "standstill": dict(GR.STANDSTILL),
            "versions": dict(VERSIONS), "name": name, "checkpoint": {"file": "model_299.pt", "sha256": checkpoint},
            "metrics": {"fall_rate": fall_count / 40, "mean_survival_s": total_alive / GR.NUM_ENVS * GR.STEP_DT,
                        "lin_vel_rmse_mps": lin, "yaw_rate_rmse_radps": yaw},
            "by_condition": rows, "elapsed_s": 10.0,
            "started_at": "2026-10-12T00:00:00+00:00", "finished_at": "2026-10-12T00:00:10+00:00"}


def _expected(name, **changes):
    return {"name": name, "checkpoint_file": "model_299.pt", "checkpoint_sha256": SHA,
            "action_scale": 0.25, "protocol_sha256_lf": PROTO_SHA, **changes}


def _validated_triplet(before=None, sham=None, fixed=None):
    inputs = {"fault_before": before or _report("fault_before", lin=0.32),
              "fault_sham": sham or _report("fault_sham", lin=0.32),
              "callback_fixed": fixed or _report("callback_fixed", lin=0.2)}
    return {arm: GR.validate_fixed_report(report, _expected(arm)) for arm, report in inputs.items()}


def test_validate_exact_report_and_copy():
    source = _report("fault_before")
    result = GR.validate_fixed_report(source, _expected("fault_before"))
    assert result == source and result is not source


@pytest.mark.parametrize("mutation", [
    lambda r: r["checkpoint"].update(sha256="3" * 64),
    lambda r: r.update(action_scale=0.5),
    lambda r: r["metrics"].update(lin_vel_rmse_mps=0.3),
    lambda r: r["by_condition"][0].update(fall_count=1),
    lambda r: r["by_condition"][0].update(num_envs=True),
    lambda r: r["by_condition"][0].update(lin_error_sq_sum=float("nan")),
    lambda r: r.pop("elapsed_s"),
    lambda r: r["by_condition"].__setitem__(1, copy.deepcopy(r["by_condition"][0])),
])
def test_validate_rejects_binding_aggregation_counts_nan_missing_and_grid(mutation):
    report = _report("fault_before")
    mutation(report)
    with pytest.raises(ValueError):
        GR.validate_fixed_report(report, _expected("fault_before"))


def test_validate_rejects_changed_fixed_setting_and_bad_time_binding():
    report = _report("fault_before")
    report["eval_seed"] = 7
    with pytest.raises(ValueError, match="eval_seed"):
        GR.validate_fixed_report(report, _expected("fault_before"))
    report = _report("fault_before")
    report["finished_at"] = "2026-10-12T00:00:20+00:00"
    with pytest.raises(ValueError, match="elapsed"):
        GR.validate_fixed_report(report, _expected("fault_before"))


def test_improved_fixed_with_unchanged_fault_and_sham_meets_numeric_conditions_only():
    summary = GR.summarize_triplet(_validated_triplet())
    assert summary["status"] == "recovery_conditions_met" and summary["recovery_conditions_met"]
    assert summary["execution_provenance_verified"] is False
    assert "recovery_proven" not in summary
    assert summary["corrected_walks"] and summary["faults_both_unhealthy"] and summary["sham_equivalent"]
    assert summary["failed_metrics"] == {"fault_before": ["err_xy_ratio"], "fault_sham": ["err_xy_ratio"]}
    assert summary["same_nonempty_failed_metrics"]
    assert not any(summary["claims"].values())


def test_walking_arms_without_relative_failure_are_no_failure_observed():
    same = _report("fault_before", lin=0.2)
    sham = _report("fault_sham", lin=0.2)
    result = GR.summarize_triplet(_validated_triplet(same, sham))
    assert result["status"] == "no_failure_observed" and not result["faults_both_unhealthy"]


def test_corrected_must_walk():
    fixed = _report("callback_fixed", lin=0.7)
    result = GR.summarize_triplet(_validated_triplet(_report("fault_before", lin=1.0),
                                                     _report("fault_sham", lin=1.0), fixed))
    assert result["status"] == "no_recovery" and not result["corrected_walks"]


def test_failed_metric_sets_must_match_and_be_nonempty():
    before = _report("fault_before", lin=0.32)
    sham = _report("fault_sham", lin=0.2, yaw=0.32)
    result = GR.summarize_triplet(_validated_triplet(before, sham))
    assert result["faults_both_unhealthy"] and not result["same_nonempty_failed_metrics"]
    assert result["status"] == "control_invalid"


def test_sham_change_and_one_direction_only_hold():
    before = _report("fault_before", lin=0.32)
    sham = _report("fault_sham", lin=0.36)
    result = GR.summarize_triplet(_validated_triplet(before, sham))
    assert result["sham_equivalence_checks"]["sham_vs_before"]["label"] == "unhealthy"
    assert result["sham_equivalence_checks"]["before_vs_sham"]["label"] == "healthy"
    assert not result["sham_equivalent"] and result["status"] == "control_invalid"


def test_summarize_rejects_different_checkpoint_and_common_setting():
    reports = _validated_triplet()
    reports["fault_sham"]["checkpoint"]["sha256"] = "4" * 64
    with pytest.raises(ValueError, match="checkpoint"):
        GR.summarize_triplet(reports)
    reports = _validated_triplet()
    reports["fault_sham"]["versions"]["gpu"] = "other"
    with pytest.raises(ValueError, match="versions"):
        GR.summarize_triplet(reports)


def test_summarize_revalidates_rows_and_arm_names():
    reports = _validated_triplet()
    reports["fault_before"]["by_condition"][0]["alive_steps_sum"] -= 1
    with pytest.raises(ValueError, match="재집계"):
        GR.summarize_triplet(reports)
    reports = _validated_triplet()
    reports["fault_before"]["name"] = "wrong"
    with pytest.raises(ValueError, match="name"):
        GR.summarize_triplet(reports)


def test_summarize_requires_exact_three_arms():
    reports = _validated_triplet()
    reports["extra"] = reports["fault_before"]
    with pytest.raises(ValueError, match="세 arm"):
        GR.summarize_triplet(reports)


def test_report_row_rmse_fixture_is_self_consistent():
    report = _report("fault_before", lin=0.31, yaw=0.27, alive=12345)
    row = report["by_condition"][0]
    assert math.sqrt(row["lin_error_sq_sum"] / row["alive_steps_sum"]) == pytest.approx(0.31)
