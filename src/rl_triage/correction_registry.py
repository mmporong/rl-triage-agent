"""E2 교정 등록 표와 회복 방식 규칙(사양 SPEC-correction-step 구성 1). 표준 라이브러리만 쓴다.

확정된 메커니즘 하나는 등록 교정 하나로만 이어진다. 등록 밖 메커니즘은 제안하지 않고 HOLD다.
회복 방식(restart/resume) 규칙은 v4 실행 전에 사전등록한 값이며 실행 결과를 보고 바꾸지 않는다.
"""
from __future__ import annotations

import math

from rl_triage import probe_loop as L

HOLD = "HOLD"
RESTART_RATIO = 0.9  # standstill_ratio가 이 값 이상이면 reward 메커니즘은 restart

# adapter_target은 bench/correction_adapter.TARGETS의 module:qualname(테스트가 일치를 검사한다).
# protocol_case는 bench/protocols/training_correction_v3의 결함 대응 case다.
REGISTRY = {
    "reward": {"correction_id": "restore_body_frame_reward",
               "adapter_target": "isaaclab.envs.mdp.rewards:track_lin_vel_xy_exp",
               "protocol_case": "case_01", "recovery_modes": ["restart", "resume"]},
    "actuator": {"correction_id": "restore_motor_torque_speed",
                 "adapter_target": "isaaclab.actuators.actuator_pd:DCMotor._clip_effort",
                 "protocol_case": "case_02", "recovery_modes": ["resume"]},
    "termination": {"correction_id": "restore_configured_timeout",
                    "adapter_target": "isaaclab.envs.mdp.terminations:time_out",
                    "protocol_case": "case_03", "recovery_modes": ["resume"]},
}
RECOVERY_MODES = ("restart", "resume")


def registry_digest(registry: dict | None = None) -> str:
    return L.digest(REGISTRY if registry is None else registry)


def protocol_case_arm(mechanism: str, registry: dict | None = None) -> dict:
    return {"case": (registry or REGISTRY)[mechanism]["protocol_case"], "arm": "corrected"}


def standstill_ratio(source_report: dict) -> float:
    """fixed_eval_v1 report의 lin_vel_rmse_mps / 제자리 기준 lin_vel_rmse_mps."""
    try:
        num = source_report["metrics"]["lin_vel_rmse_mps"]
        den = source_report["standstill"]["lin_vel_rmse_mps"]
    except (KeyError, TypeError) as e:
        raise ValueError(f"fixed_eval_v1 report에 필요한 필드가 없다: {e!r}") from e
    for v in (num, den):
        if type(v) not in (int, float) or not math.isfinite(v):
            raise ValueError("lin_vel_rmse_mps가 유한한 수가 아니다")
    if den <= 0:
        raise ValueError("제자리 기준 lin_vel_rmse_mps가 0 이하다")
    return num / den


def recommend_recovery_mode(mechanism: str, source_report: dict, registry: dict | None = None) -> dict:
    """등록 메커니즘이면 {mode, correction_id, standstill_ratio, rule, ...}, 아니면 mode=HOLD."""
    reg = REGISTRY if registry is None else registry
    if mechanism not in reg:
        return {"mode": HOLD, "mechanism": mechanism, "correction_id": None, "standstill_ratio": None,
                "rule": "unregistered_mechanism"}
    ratio = standstill_ratio(source_report)
    if mechanism == "reward" and ratio >= RESTART_RATIO:
        mode, rule = "restart", f"reward and standstill_ratio >= {RESTART_RATIO}"
    else:
        mode, rule = "resume", "default"
    if mode not in reg[mechanism]["recovery_modes"]:
        mode, rule = HOLD, f"mode {mode} not registered for {mechanism}"
    return {"mode": mode, "mechanism": mechanism, "correction_id": reg[mechanism]["correction_id"],
            "standstill_ratio": ratio, "rule": rule, "restart_ratio": RESTART_RATIO}
