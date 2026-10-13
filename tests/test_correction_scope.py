"""관측 기반 교정 범위 점검: 공개 P_reward 기록에서 F1 교정이 범위 부족으로 나오지 않는지, U1형 측정에서 남는 항."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl_triage import correction_scope as S  # noqa: E402
from rl_triage import probe_loop as L  # noqa: E402

TARGET = ["track_lin_vel_xy_exp"]  # restore_body_frame_reward가 되돌리는 함수(bench/correction_adapter.TARGETS)


def _public_p_reward() -> dict[str, dict]:
    """홀드아웃 21개와 종단 시연 고리 5개의 공개 P_reward 측정값."""
    rows = {}
    for path in sorted((ROOT / "evals/results/p1a_probes_holdout_20261009/probes").glob("*.json")):
        rows[path.stem] = json.loads(path.read_text(encoding="utf-8"))["probes"]["P_reward"]["measurement"]
    for path in sorted((ROOT / "evals/results").glob("e2e_*_loop/probes/*__P_reward.json")):
        rows[path.stem] = json.loads(path.read_text(encoding="utf-8"))["probes"]["P_reward"]["measurement"]
    return rows


def test_public_records_never_leave_a_term_outside_the_registered_reward_correction():
    rows = _public_p_reward()
    assert len(rows) == 26
    verdicts = {name: S.check("P_reward", m, TARGET) for name, m in rows.items()}
    assert {v["verdict"] for v in verdicts.values()} == {"scope_sufficient"}
    abnormal = sorted(name for name, v in verdicts.items() if v["abnormal_terms"])
    # 이상 항이 있는 기록은 F1(h02) 실행뿐이고 그 항은 등록 표적 하나다
    assert abnormal and all(name.startswith("h02_") for name in abnormal)
    assert all(verdicts[name]["abnormal_terms"] == TARGET for name in abnormal)


def test_u1_like_measurement_leaves_the_yaw_tracking_term_outside_the_correction():
    result = S.check("P_reward", {"track_lin_vel_xy_exp_rel_error": 0.41, "track_ang_vel_z_exp_rel_error": 0.93}, TARGET)
    assert (result["verdict"], result["abnormal_terms"], result["uncovered_terms"]) == (
        "scope_insufficient", ["track_ang_vel_z_exp", "track_lin_vel_xy_exp"], ["track_ang_vel_z_exp"])
    assert result["threshold"] == L.THRESHOLDS["P_reward"][2] == 0.05
    covered = S.check("P_reward", {"track_lin_vel_xy_exp_rel_error": 0.41, "track_ang_vel_z_exp_rel_error": 0.93},
                      ["track_lin_vel_xy_exp", "track_ang_vel_z_exp"])
    assert covered["verdict"] == "scope_sufficient"


def test_the_frozen_threshold_is_strict_like_classify():
    at = {"track_lin_vel_xy_exp_rel_error": 0.3, "track_ang_vel_z_exp_rel_error": 0.05}
    assert S.check("P_reward", at, TARGET)["verdict"] == "scope_sufficient"
    assert L.classify("P_reward", {"track_lin_vel_xy_exp_rel_error": 0.0, "track_ang_vel_z_exp_rel_error": 0.05}, None) == "normal"


@pytest.mark.parametrize("measurement", [None, {}, {"track_lin_vel_xy_exp_rel_error": 0.3},
                                         {"track_lin_vel_xy_exp_rel_error": 0.3, "track_ang_vel_z_exp_rel_error": float("nan")},
                                         {"track_lin_vel_xy_exp_rel_error": "0.3", "track_ang_vel_z_exp_rel_error": 0.1}])
def test_missing_or_non_numeric_measurements_are_unknown(measurement):
    assert S.check("P_reward", measurement, TARGET)["verdict"] == "scope_unknown"


def test_probes_without_per_term_outputs_are_not_applicable():
    assert S.check("P_torque", {"low_speed_saturation": 0.08}, ["DCMotor._clip_effort"])["verdict"] == "not_applicable"


def test_contrast_categories():
    assert S.contrast("scope_insufficient", "not_recovered") == "insufficient_and_not_recovered"
    assert S.contrast("scope_insufficient", "recovered") == "insufficient_but_recovered"
    assert S.contrast("scope_sufficient", "not_recovered") == "sufficient_but_not_recovered"
    assert S.contrast("scope_sufficient", "recovered") == "sufficient_and_recovered"
    assert S.contrast("scope_unknown", "recovered") is None and S.contrast("scope_insufficient", "unknown") is None
