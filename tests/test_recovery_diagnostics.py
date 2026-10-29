"""회복 라벨 보존, 기준 민감도, 집계에 가려지는 조건별 낙상 검사."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from rl_triage import recovery_diagnostics as D

ROOT = Path(__file__).resolve().parents[1]
BASE = {"survival_s": 20.0, "horizon_s": 20.0, "fall_frac": 0.0, "err_xy": 0.2, "err_yaw": 0.3}
BANDS = {"survival_ratio_min": 0.9, "fall_frac_max": 0.05,
         "err_xy_ratio_max": 1.15, "err_yaw_ratio_max": 1.17}


def test_margin_boundary_and_missing_conditions_preserve_registered_label():
    run = {**BASE, "fall_frac": 0.05}
    before = copy.deepcopy((run, BASE, BANDS))
    result = D.diagnose(run, BASE, BANDS)
    assert result["registered_verdict"]["label"] == "healthy"
    assert result["margins"]["fall_frac"]["signed_margin"] == 0
    assert result["condition_warnings"] == {
        "status": "unavailable", "reason": "condition_rows_not_provided",
        "conditions_checked": None, "all_fell": None,
    }
    assert (run, BASE, BANDS) == before
    assert result["used_for_decision"] is False


def test_aggregate_healthy_does_not_hide_one_all_fell_condition():
    rows = [{"command": [-1, 0, 1], "fall_count": 40, "num_envs": 40},
            {"command": [1, 0, 0], "fall_count": 0, "num_envs": 1000}]
    result = D.diagnose({**BASE, "fall_frac": 40 / 1040}, BASE, BANDS, conditions=rows)
    assert result["registered_verdict"]["label"] == "healthy"
    assert result["condition_warnings"]["all_fell"] == [
        {"condition_index": 0, "command": [-1.0, 0.0, 1.0], "fall_count": 40, "num_envs": 40},
    ]


def test_short_horizon_survival_margin_is_not_used_as_failure():
    result = D.diagnose({**BASE, "survival_s": 5.0, "horizon_s": 5.0}, BASE, BANDS)
    assert result["registered_verdict"]["label"] == "undetermined"
    assert result["margins"]["survival_ratio"]["signed_margin"] < 0
    assert result["margins"]["survival_ratio"]["applies_to_verdict"] is False


@pytest.mark.parametrize("change", [{"err_xy": float("nan")}, {"survival_s": float("inf")},
                                    {"fall_frac": True}, {"fall_frac": 1.1}, {"err_yaw": -1}])
def test_invalid_measurements_are_rejected(change):
    with pytest.raises(ValueError):
        D.diagnose({**BASE, **change}, BASE, BANDS)


def test_zero_reference_and_invalid_conditions_are_rejected():
    with pytest.raises(ValueError):
        D.diagnose(BASE, {**BASE, "err_xy": 0.0}, BANDS)
    for rows in ([], [{"command": [1, 0, 0], "fall_count": 41, "num_envs": 40}],
                 [{"command": [1, 0, float("nan")], "fall_count": 0, "num_envs": 40}]):
        with pytest.raises(ValueError):
            D.diagnose(BASE, BASE, BANDS, conditions=rows)


def test_seen_seed_reference_sensitivity_does_not_relabel_saved_outcome():
    folder = ROOT / "evals/results"
    path = folder / "reset_interventions_v2_20261015/seeds/seed_2028.json"
    raw = path.read_bytes()
    saved = json.loads(raw)
    value = saved["arms"]["actor_last_layer_reset"]
    metrics = value["fixed_evaluation"]
    run = {"survival_s": metrics["mean_survival_s"], "horizon_s": 20.0,
           "fall_frac": metrics["fall_rate"], "err_xy": metrics["lin_vel_rmse_mps"],
           "err_yaw": metrics["yaw_rate_rmse_radps"]}
    original = json.loads((folder / "reward_v4_s2028_20261013/result.json").read_text(encoding="utf-8"))["numeric"]
    resumed = D.diagnose(run, original["behavior"]["clean_resume_reference"], original["bands"])
    initial = D.diagnose(run, original["original_behavior"]["clean_reference"], original["bands"])
    assert resumed["registered_verdict"]["label"] == "unhealthy"
    assert initial["registered_verdict"]["label"] == "healthy"
    assert value["label"] == "not_recovered"
    assert path.read_bytes() == raw
