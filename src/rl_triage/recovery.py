"""P0-B1 행동 기반 회복 판정(docs/P0-B1-RECOVERY.md). 학습 보상 크기를 쓰지 않는다.

현행 eval_bridge.RECOVERY_BAND는 Train/mean_reward 비를 쓰므로 보상 가중치를 바꾸는 개입이 판정 자체를 움직인다.
여기서는 각 실행의 실제 설정(step_dt·제한 시간·명령 재샘플 주기)으로 정규화한 행동 지표만 쓴다.

- survival_s: Train/mean_episode_length × step_dt
- fall_frac: base_contact ÷ (base_contact + time_out). Episode_Termination/<항>은 리셋 개수다
- err_xy·err_yaw: Metrics/base_velocity/error_vel_* × (재샘플 최대 시간 ÷ step_dt) ÷ 에피소드 길이.
  Isaac Lab v2.1.1 UniformVelocityCommand는 스텝 오차를 max_command_step으로 나눠 에피소드 동안 누적한다.

표준 라이브러리만 쓴다.
"""
from __future__ import annotations

import re

LEN = "Train/mean_episode_length"
BEHAVIOR_KEYS = ("survival_ratio", "fall_frac", "err_xy_ratio", "err_yaw_ratio")
# 정상 실행 사이에서 관측한 최대 차이에 더하는 여유. 낙상 비율은 절대값(비율 포인트)이다.
MARGIN = {"ratio": 0.10, "fall_frac": 0.05}


def run_overrides(name: str, cases: dict, benign: list[str], v2: dict) -> list[str] | None:
    """텔레메트리 이름(<id>_s<seed>)의 실행에 적용된 override. 모르는 이름은 None."""
    if re.fullmatch(r"(s\d\d_)?baseline_s\d+", name):
        return []
    if re.fullmatch(r"benign_all_s\d+", name):
        return list(benign)
    m = re.fullmatch(r"(c\d\d)_revert_ch(\d+)_s\d+", name)
    if m and m[1] in cases:
        return [o for i, o in enumerate(cases[m[1]]) if i != int(m[2]) - 1]
    m = re.fullmatch(r"(c\d\d)_s\d+", name)
    if m and m[1] in cases:
        return list(cases[m[1]])
    m = re.fullmatch(r"v2_(S\d\d)_s\d+", name)
    if m and m[1] in v2:
        return [v2[m[1]]]
    return None


def seed_of(name: str) -> int:
    return int(name.rsplit("_s", 1)[1])


def run_config(ref_flat: dict, overrides: list[str], apply) -> dict:
    """기준 평탄화 설정에 override를 적용해 정규화에 쓰는 값만 뽑는다. apply는 rules.apply_overrides."""
    flat = apply(ref_flat, overrides)
    return {"step_dt": flat["env.sim.dt"] * flat["env.decimation"], "horizon_s": flat["env.episode_length_s"],
            "resample_max_s": flat["env.commands.base_velocity.resampling_time_range"][1]}


def _m(summary: dict, tag: str) -> float:
    return summary[tag]["mean_last_20pct"]


def behavior(summary: dict, cfg: dict) -> dict:
    steps = _m(summary, LEN)
    if steps <= 0:
        raise ValueError(f"{LEN}이 0 이하다")
    to, bc = _m(summary, "Episode_Termination/time_out"), _m(summary, "Episode_Termination/base_contact")
    per_step = cfg["resample_max_s"] / cfg["step_dt"] / steps
    return {"survival_s": steps * cfg["step_dt"], "horizon_s": cfg["horizon_s"],
            "fall_frac": bc / (to + bc) if to + bc > 0 else 0.0,
            "err_xy": _m(summary, "Metrics/base_velocity/error_vel_xy") * per_step,
            "err_yaw": _m(summary, "Metrics/base_velocity/error_vel_yaw") * per_step}


def compare(run: dict, ref: dict) -> dict:
    """같은 seed 기준 실행 대비 행동 지표. 기준값이 0 이하면 비를 정의할 수 없어 ValueError다."""
    bad = [k for k in ("survival_s", "err_xy", "err_yaw", "horizon_s") if not ref[k] > 0]
    if bad:
        raise ValueError(f"기준 실행의 {bad} 값이 0 이하라 비를 계산할 수 없다")
    return {"survival_ratio": run["survival_s"] / ref["survival_s"], "fall_frac": run["fall_frac"],
            "err_xy_ratio": run["err_xy"] / ref["err_xy"], "err_yaw_ratio": run["err_yaw"] / ref["err_yaw"],
            "horizon_ratio": run["horizon_s"] / ref["horizon_s"]}


def derive_bands(normal: list[dict]) -> dict:
    """정상 실행(서로 다른 seed의 기준 실행끼리, 무해 변경 실행 대 같은 seed 기준)의 비교값으로 허용 범위를 정한다.

    비 지표는 1에서 가장 먼 정상값까지의 거리에 MARGIN["ratio"]를 더한다. 생존 비는 아래쪽만, 오차 비는 위쪽만 본다.
    낙상 비율은 정상 최댓값에 MARGIN["fall_frac"]를 더한다.
    """
    if not normal:
        raise ValueError("정상 실행 비교값이 없다")
    dev = {k: max(abs(c[k] - 1.0) for c in normal) for k in ("survival_ratio", "err_xy_ratio", "err_yaw_ratio")}
    return {"survival_ratio_min": 1.0 - dev["survival_ratio"] - MARGIN["ratio"],
            "err_xy_ratio_max": 1.0 + dev["err_xy_ratio"] + MARGIN["ratio"],
            "err_yaw_ratio_max": 1.0 + dev["err_yaw_ratio"] + MARGIN["ratio"],
            "fall_frac_max": max(c["fall_frac"] for c in normal) + MARGIN["fall_frac"]}


def verdict(c: dict, bands: dict) -> dict:
    """healthy / unhealthy / undetermined.

    실행의 제한 시간이 기준보다 짧으면 기준 시간 동안 버티는지 관측할 수 없다. 다른 지표가 모두 범위 안이면
    undetermined로 둔다(고정 평가 조건 실측은 P0-B2).
    """
    checks = {
        "survival_ratio": c["survival_ratio"] >= bands["survival_ratio_min"],
        "fall_frac": c["fall_frac"] <= bands["fall_frac_max"],
        "err_xy_ratio": c["err_xy_ratio"] <= bands["err_xy_ratio_max"],
        "err_yaw_ratio": c["err_yaw_ratio"] <= bands["err_yaw_ratio_max"],
    }
    short = c["horizon_ratio"] < 1.0
    failed = [k for k, ok in checks.items() if not ok and not (short and k == "survival_ratio")]
    if failed:
        label = "unhealthy"
    elif short:
        label = "undetermined"
    else:
        label = "healthy"
    return {"label": label, "failed": failed, "short_horizon": short, "checks": checks}
