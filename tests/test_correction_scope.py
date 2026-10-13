"""관측 기반 교정 범위 점검: 커밋된 공개 P_reward 기록 전부에서 규칙이 내는 판정(설계자 결함은 범위 부족 0, U1 시연은 회전
추종 항이 등록 표적 밖), U1형 측정에서 남는 항."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl_triage import correction_scope as S  # noqa: E402
from rl_triage import probe_loop as L  # noqa: E402

TARGET = ["track_lin_vel_xy_exp"]  # restore_body_frame_reward가 되돌리는 함수(bench/correction_adapter.TARGETS)


HOLDOUT = "evals/results/p1a_probes_holdout_20261009/probes/*.json"
LOOPS = "evals/results/e2e_*_loop/probes/*__P_reward.json"


def _public_p_reward() -> dict[str, dict]:
    """커밋된 공개 P_reward 측정값 전부(홀드아웃 probe 묶음, 종단 시연 고리 probe). 진행 중인 실행의 미커밋 산출물은 뺀다.
    이름은 파일 이름(홀드아웃은 실행 이름, 고리는 '실행 이름__P_reward')이다."""
    listed = subprocess.run(["git", "-C", str(ROOT), "ls-files", "--", HOLDOUT, LOOPS], capture_output=True, text=True,
                            check=True).stdout.split()
    return {Path(relative).stem: json.loads((ROOT / relative).read_text(encoding="utf-8"))["probes"]["P_reward"]["measurement"]
            for relative in listed}


def test_every_committed_public_record_gets_the_expected_scope_verdict():
    """개수를 고정하지 않는다. 설계자 결함 기록(보상·모터·홀드아웃)은 범위 부족 0, U1(설계자 밖) 기록은 회전 항이 표적 밖."""
    verdicts = {name: S.check("P_reward", m, TARGET) for name, m in _public_p_reward().items()}
    designer = {name: v for name, v in verdicts.items() if not name.startswith("u1_")}
    upstream = {name: v for name, v in verdicts.items() if name.startswith("u1_")}
    assert any("__" not in name for name in designer) and any("__" in name for name in designer)  # 홀드아웃·고리 모두 읽었다
    assert {v["verdict"] for v in designer.values()} == {"scope_sufficient"}
    abnormal = sorted(name for name, v in designer.items() if v["abnormal_terms"])
    # 이상 항이 있는 설계자 결함 기록은 F1(h02) 실행뿐이고 그 항은 등록 표적 하나다
    assert abnormal and all(name.startswith("h02_") and designer[name]["abnormal_terms"] == TARGET for name in abnormal)
    # U1 시연(4e13082) 기록: 두 추종 항이 모두 이상이고 회전 추종 항이 등록 표적 밖에 남는다
    assert "u1_s2040__P_reward" in upstream
    for v in upstream.values():
        assert (v["verdict"], v["abnormal_terms"], v["uncovered_terms"]) == (
            "scope_insufficient", ["track_ang_vel_z_exp", "track_lin_vel_xy_exp"], ["track_ang_vel_z_exp"])


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
