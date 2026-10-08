"""P0-B1 행동 기반 회복 판정: 정규화·허용 범위·라벨 규칙과 저장 실행 재판정 기록을 검사한다."""
import json

import pytest

from rl_triage import recovery as RC
from rl_triage import rules as R
from test_offline_replay import ROOT, USER_PATH, _offline

CASES = {c["case_id"]: c["overrides"] for c in json.loads((ROOT / "bench" / "cases.json").read_text(encoding="utf-8"))}
REF = {"env.sim.dt": 0.005, "env.decimation": 4, "env.episode_length_s": 20.0,
       "env.commands.base_velocity.resampling_time_range": [10.0, 10.0], "env.rewards.a.weight": 1.0}


def _s(mean):
    return {"mean_last_20pct": mean}


def _summary(steps, to, bc, exy, eyaw):
    return {RC.LEN: _s(steps), "Episode_Termination/time_out": _s(to), "Episode_Termination/base_contact": _s(bc),
            "Metrics/base_velocity/error_vel_xy": _s(exy), "Metrics/base_velocity/error_vel_yaw": _s(eyaw)}


def test_run_overrides_follow_run_names():
    v2 = {"S13": "env.episode_length_s=5.0"}
    assert RC.run_overrides("baseline_s7", CASES, ["b"], v2) == []
    assert RC.run_overrides("s02_baseline_s42", CASES, ["b"], v2) == []
    assert RC.run_overrides("benign_all_s7", CASES, ["b1", "b2"], v2) == ["b1", "b2"]
    assert RC.run_overrides("c01_s123", CASES, [], v2) == CASES["c01"]
    assert RC.run_overrides("c01_revert_ch3_s42", CASES, [], v2) == CASES["c01"][:2]
    assert RC.run_overrides("v2_S13_s7", CASES, [], v2) == ["env.episode_length_s=5.0"]
    assert RC.run_overrides("mystery_s7", CASES, [], v2) is None and RC.seed_of("c01_revert_ch3_s42") == 42


def test_behavior_uses_each_runs_own_step_and_command_period():
    """Isaac Lab은 오차를 max_command_step(= 재샘플 시간 ÷ step_dt)으로 나눠 에피소드 동안 누적한다."""
    cfg = RC.run_config(REF, ["env.sim.dt=0.02", "env.commands.base_velocity.resampling_time_range=[8.0,8.0]"],
                        R.apply_overrides)
    assert cfg == {"step_dt": 0.08, "horizon_s": 20.0, "resample_max_s": 8.0}
    # 50스텝 동안 매 스텝 오차 0.5 → 누적 기록값 = 0.5 × 50 / (8 / 0.08) = 0.25
    b = RC.behavior(_summary(50, 1.0, 3.0, 0.25, 0.1), cfg)
    assert b["survival_s"] == pytest.approx(4.0) and b["fall_frac"] == 0.75
    assert b["err_xy"] == pytest.approx(0.5) and b["err_yaw"] == pytest.approx(0.2)


def test_bands_come_from_normal_runs_plus_margin():
    normal = [{"survival_ratio": 0.98, "err_xy_ratio": 1.03, "err_yaw_ratio": 0.99, "fall_frac": 0.006},
              {"survival_ratio": 1.01, "err_xy_ratio": 0.97, "err_yaw_ratio": 1.02, "fall_frac": 0.001}]
    b = RC.derive_bands(normal)
    assert b["survival_ratio_min"] == pytest.approx(0.88) and b["err_xy_ratio_max"] == pytest.approx(1.13)
    assert b["err_yaw_ratio_max"] == pytest.approx(1.12) and b["fall_frac_max"] == pytest.approx(0.056)
    with pytest.raises(ValueError):
        RC.derive_bands([])


@pytest.mark.parametrize("change, label, failed", [
    ({}, "healthy", []),
    ({"fall_frac": 0.3}, "unhealthy", ["fall_frac"]),
    ({"err_yaw_ratio": 1.5}, "unhealthy", ["err_yaw_ratio"]),
    ({"horizon_ratio": 0.5}, "undetermined", []),
    ({"horizon_ratio": 0.05, "survival_ratio": 0.05}, "undetermined", []),
    ({"horizon_ratio": 0.05, "survival_ratio": 0.05, "fall_frac": 0.9}, "unhealthy", ["fall_frac"]),
])
def test_verdict_labels(change, label, failed):
    bands = {"survival_ratio_min": 0.88, "err_xy_ratio_max": 1.13, "err_yaw_ratio_max": 1.12, "fall_frac_max": 0.056}
    c = {"survival_ratio": 1.0, "fall_frac": 0.005, "err_xy_ratio": 1.0, "err_yaw_ratio": 1.0, "horizon_ratio": 1.0,
         **change}
    v = RC.verdict(c, bands)
    assert (v["label"], v["failed"]) == (label, failed)


def test_zero_reference_value_is_an_error_not_a_ratio():
    ref = {"survival_s": 20.0, "err_xy": 0.0, "err_yaw": 0.4, "horizon_s": 20.0, "fall_frac": 0.0}
    with pytest.raises(ValueError, match="err_xy"):
        RC.compare(dict(ref, err_xy=0.5), ref)


def test_reward_weight_does_not_move_the_behavior_verdict():
    """보상 항 값이 1000배가 돼도 행동 지표가 같으면 라벨이 같다(현행 판정은 mean_reward로 움직인다)."""
    cfg = RC.run_config(REF, [], R.apply_overrides)
    base = _summary(1000, 1.7, 0.01, 0.36, 0.23)
    run = {**base, "Train/mean_reward": _s(-370.0), "Episode_Reward/dof_torques_l2": _s(-50.0)}
    assert RC.behavior(run, cfg) == RC.behavior(base, cfg)


@pytest.fixture(scope="module")
def relabel_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("p0b1") / "relabel"
    proc = _offline("evals/recovery_relabel.py", "--out", out)
    assert proc.returncode == 0, proc.stderr
    return json.loads((out / "relabel.json").read_text(encoding="utf-8")), proc


def test_relabel_covers_every_stored_run_once(relabel_run):
    report, proc = relabel_run
    files = sorted((ROOT / "bench" / "runs").glob("*.telemetry.json"))
    assert len(report["runs"]) + len(report["duplicates"]) == len(files)
    assert report["duplicates"] == [{"name": "s02_baseline_s42", "same_run_as": "baseline_s42"}]
    assert {r["run"] for r in report["runs"] if r["is_reference"]} == {"baseline_s7", "baseline_s42", "baseline_s123"}
    assert all(r["behavior_label"] == "healthy" for r in report["runs"] if r["is_reference"])
    assert report["environment"]["model_calls"] == 0
    text = proc.stdout + json.dumps(report, ensure_ascii=False)
    assert not USER_PATH.search(text) and str(ROOT) not in text


def test_short_horizon_runs_are_undetermined_not_recovered(relabel_run):
    report, _ = relabel_run
    by = {r["run"]: r for r in report["runs"]}
    for name in ("c01_s7", "c01_revert_ch1_s42", "v2_S13_s7"):
        assert by[name]["short_horizon"] and by[name]["behavior_label"] in ("undetermined", "unhealthy"), name
    assert by["c01_revert_ch3_s42"]["behavior_label"] == "healthy"  # 해로운 변경(ch3)을 되돌린 실행
