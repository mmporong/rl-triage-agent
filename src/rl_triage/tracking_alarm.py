"""추종 경고 R1: 과거 20회 평균 추종 오차가 같은 시점 정상 참조 최대값의 2배를 넘는 상태가 10회 연속이면 경고한다.

상수는 이미 본 seed 2026~2028 자료에서 사후에 고른 후보(bench/private/codex_support_reward_early_20261013/
rule_candidates.json)와 같다. 최적화·검증한 임계값이 아니며 원인을 가리지 않는다. 같은 자료에서 F1 외 결함
12/15에도 울렸고 짧은 에피소드 결함(F6)은 놓쳤다. 표준 라이브러리만 쓴다.
"""
from __future__ import annotations

import math
import statistics

METRIC = "Metrics/base_velocity/error_vel_xy"
R1 = {"metric": METRIC, "smoothing": "trailing_mean", "window_iterations": 20, "eligible_iteration_min": 60,
      "persistence_iterations": 10, "error_ratio": 2.0,
      "reference": "같은 iteration의 정상 참조 실행별 과거 평균 중 최대값"}


def smooth(values: list[float], window: int) -> list[float | None]:
    """iteration t의 값은 t를 포함한 과거 window개 평균. 처음 window-1개는 None."""
    return [None if t < window - 1 else statistics.fmean(values[t - window + 1:t + 1]) for t in range(len(values))]


def alerts(flags: list[bool], persistence: int) -> list[int]:
    """조건이 persistence회 연속 참이 되는 iteration마다 이벤트 하나. 끊긴 뒤 다시 채우면 새 이벤트."""
    streak, events = 0, []
    for t, flag in enumerate(flags):
        streak = streak + 1 if flag else 0
        if streak == persistence:
            events.append(t)
    return events


def _series(values, label: str) -> list[float]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"{label}: 비어 있지 않은 시계열이어야 한다")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
        raise ValueError(f"{label}: 유한한 수만 허용한다")
    return [float(v) for v in values]


def r1(series: list[float], references: list[list[float]], rule: dict = R1) -> dict:
    """한 실행의 추종 오차 시계열을 정상 참조 시계열들과 비교한 R1 결과."""
    run = _series(series, "series")
    refs = [_series(r, f"reference[{i}]") for i, r in enumerate(references)]
    if not refs or any(len(r) != len(run) for r in refs):
        raise ValueError("참조 실행이 없거나 iteration 수가 다르다")
    window = rule["window_iterations"]
    smoothed, smoothed_refs = smooth(run, window), [smooth(r, window) for r in refs]
    flags = [t >= rule["eligible_iteration_min"] and smoothed[t] is not None
             and smoothed[t] > rule["error_ratio"] * max(r[t] for r in smoothed_refs)
             for t in range(len(run))]
    events = alerts(flags, rule["persistence_iterations"])
    return {"warned": bool(events), "alert_iterations": events,
            "first_alert_iteration": events[0] if events else None,
            "qualifying_iteration_count": sum(flags), "iterations": len(run)}
