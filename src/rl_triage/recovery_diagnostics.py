"""동결 회복 판정에 경계 여유와 조건별 낙상 경고를 덧붙이는 읽기 전용 진단."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from rl_triage import recovery as RC

SCHEMA = "recovery_diagnostics_v1"


def _number(value: Any, name: str, *, minimum: float = 0.0,
            maximum: float | None = None, positive: bool = False) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name}: 유한한 수가 필요하다")
    if value < minimum or (positive and value <= 0) or (maximum is not None and value > maximum):
        raise ValueError(f"{name}: 허용 범위 밖이다")
    return float(value)


def _behavior(value: Mapping[str, Any], name: str, *, reference: bool) -> dict[str, float]:
    result = {
        key: _number(value[key], f"{name}.{key}", positive=reference or key == "horizon_s")
        for key in ("survival_s", "horizon_s", "err_xy", "err_yaw")
    }
    result["fall_frac"] = _number(value["fall_frac"], f"{name}.fall_frac", maximum=1.0)
    return result


def condition_warnings(conditions: Sequence[Mapping[str, Any]] | None) -> dict[str, Any]:
    """조건별 전원 낙상을 표시한다. 누락 자료를 0회 낙상으로 채우지 않는다."""
    if conditions is None:
        return {"status": "unavailable", "reason": "condition_rows_not_provided",
                "conditions_checked": None, "all_fell": None}
    if not conditions:
        raise ValueError("조건 목록이 비었다")
    warnings = []
    seen = set()
    for index, row in enumerate(conditions):
        command = row["command"]
        if not isinstance(command, (list, tuple)) or len(command) != 3:
            raise ValueError("command: 세 성분이 필요하다")
        values = tuple(_number(x, "command", minimum=-math.inf) for x in command)
        if values in seen:
            raise ValueError("중복 평가 조건")
        seen.add(values)
        count, total = row["fall_count"], row["num_envs"]
        if type(count) is not int or type(total) is not int or not 0 <= count <= total or total <= 0:
            raise ValueError("fall_count/num_envs: 유효한 정수 개수가 필요하다")
        if count == total:
            warnings.append({"condition_index": index, "command": list(values),
                             "fall_count": count, "num_envs": total})
    return {"status": "available", "conditions_checked": len(conditions), "all_fell": warnings}


def diagnose(run: Mapping[str, Any], reference: Mapping[str, Any], bands: Mapping[str, Any], *,
             conditions: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """기존 라벨과 부호 있는 여유를 함께 반환한다. 양수·0은 허용 범위 안이다.

    이 출력은 기존 기준·라벨을 바꾸거나 회복 실행 권한을 만들지 않는다.
    conditions는 run의 조건별 관측이며, 실행 계보 증명은 호출자의 책임이다.
    """
    run_values = _behavior(run, "run", reference=False)
    ref_values = _behavior(reference, "reference", reference=True)
    limits = {key: _number(bands[key], key, maximum=1.0 if key == "fall_frac_max" else None)
              for key in ("survival_ratio_min", "err_xy_ratio_max", "err_yaw_ratio_max", "fall_frac_max")}
    comparison = RC.compare(run_values, ref_values)
    for key, value in comparison.items():
        _number(value, key)
    decision = RC.verdict(comparison, limits)
    margins = {}
    for key, limit, direction in (
        ("survival_ratio", "survival_ratio_min", "minimum"),
        ("fall_frac", "fall_frac_max", "maximum"),
        ("err_xy_ratio", "err_xy_ratio_max", "maximum"),
        ("err_yaw_ratio", "err_yaw_ratio_max", "maximum"),
    ):
        value, bound = comparison[key], limits[limit]
        margins[key] = {
            "observed": value, "bound": bound, "direction": direction,
            "signed_margin": value - bound if direction == "minimum" else bound - value,
            "applies_to_verdict": not (decision["short_horizon"] and key == "survival_ratio"),
            "unit": "fraction_points" if key == "fall_frac" else "ratio",
        }
    return {"schema": SCHEMA, "used_for_decision": False, "comparison": comparison,
            "registered_verdict": decision, "margins": margins,
            "condition_warnings": condition_warnings(conditions)}
