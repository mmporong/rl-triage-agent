"""P1 고리용 고정 행동 평가. P0-B1 상대 판정에 사전등록한 절대 행동 기준을 함께 적용한다.

학습 보상과 원인 정답은 읽지 않는다. 기존 P0-B1/P0-B2 판정과 결과는 변경하지 않는다.
"""
from __future__ import annotations

import math
import re

from . import recovery as RC
from .probe_loop import digest

GATE_KEYS = {"min_survival_fraction", "max_fall_rate", "max_lin_vel_rmse_mps", "max_yaw_rate_rmse_radps"}
METRICS = ("fall_rate", "mean_survival_s", "lin_vel_rmse_mps", "yaw_rate_rmse_radps")
RAW_KEYS = {"num_envs", "fall_count", "alive_steps_sum", "lin_error_sq_sum", "yaw_error_sq_sum"}
REL_TOL = 1e-5
ABS_TOL = 1e-8
SURVIVAL_ABS_TOL = 1e-4


class OracleError(ValueError):
    """비교할 수 없는 평가 입력 또는 사전등록 계약."""


def _number(value, name: str, positive: bool = False, signed: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OracleError(f"{name}: 유한한 수가 필요하다")
    try:
        finite = math.isfinite(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise OracleError(f"{name}: 유한한 수가 필요하다") from exc
    if not finite:
        raise OracleError(f"{name}: 유한한 수가 필요하다")
    if (not signed and value < 0) or (positive and value == 0):
        raise OracleError(f"{name}: {'양수' if positive else '0 이상의 수'}가 필요하다")
    return float(value)


def _integer(value, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise OracleError(f"{name}: {minimum} 이상의 정수가 필요하다")
    _number(value, name)
    return value


def _add(left: int | float, right: int | float, name: str) -> int | float:
    try:
        value = left + right
        finite = math.isfinite(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise OracleError(f"{name}: 합계가 유한한 수의 범위를 벗어났다") from exc
    if not finite:
        raise OracleError(f"{name}: 합계가 유한한 수의 범위를 벗어났다")
    return value


def _mul(left: int | float, right: int | float, name: str) -> float:
    try:
        value = left * right
        finite = math.isfinite(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise OracleError(f"{name}: 곱이 유한한 수의 범위를 벗어났다") from exc
    if not finite:
        raise OracleError(f"{name}: 곱이 유한한 수의 범위를 벗어났다")
    return float(value)


def _div(numerator: int | float, denominator: int | float, name: str) -> float:
    if denominator <= 0:
        raise OracleError(f"{name}: 분모가 양수여야 한다")
    try:
        value = numerator / denominator
        finite = math.isfinite(value)
    except (OverflowError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise OracleError(f"{name}: 비가 유한한 수의 범위를 벗어났다") from exc
    if not finite:
        raise OracleError(f"{name}: 비가 유한한 수의 범위를 벗어났다")
    return float(value)


def _close(actual: float, expected: float, name: str, abs_tol: float = ABS_TOL) -> None:
    if not math.isclose(actual, expected, rel_tol=REL_TOL, abs_tol=abs_tol):
        raise OracleError(f"{name}: 원시 통계 또는 조건별 지표와 일치하지 않는다")


def _sha(value, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise OracleError(f"{name}: 소문자 SHA256이 필요하다")
    return value


def _validated(report: dict) -> tuple[dict, dict, bool]:
    """메타데이터와 행동 값을 검증한다. 원시 합계가 없으면 제한된 일관성 검사만 수행한다."""
    try:
        for k in ("protocol", "task"):
            if not isinstance(report[k], str) or not report[k]:
                raise OracleError(f"{k}: 빈 문자열을 쓸 수 없다")
        eval_seed = _integer(report["eval_seed"], "eval_seed")
        num_envs = _integer(report["num_envs"], "num_envs", minimum=1)
        horizon_steps = _integer(report["horizon_steps"], "horizon_steps", minimum=1)
        dt = _number(report["step_dt"], "step_dt", positive=True)
        _number(report["action_scale"], "action_scale", positive=True)
        horizon = _mul(horizon_steps, dt, "평가 horizon")
        protocol_sha = _sha(report["protocol_sha256_lf"], "protocol_sha256_lf")
        _sha(report["checkpoint"]["sha256"], "checkpoint.sha256")
        versions = report["versions"]
        if not isinstance(versions, dict) or not versions:
            raise OracleError("평가 실행 환경 버전이 없다")
        for k in ("torch", "isaaclab", "isaaclab_tasks", "isaaclab_rl", "rsl-rl-lib"):
            if not isinstance(versions.get(k), str) or not versions[k]:
                raise OracleError(f"평가 실행 환경의 {k} 버전이 없다")
        m = {k: _number(report["metrics"][k], k) for k in METRICS}
        if m["fall_rate"] > 1 or m["mean_survival_s"] <= 0:
            raise OracleError("낙상 비율 또는 생존 시간이 평가 범위를 벗어났다")
        if m["mean_survival_s"] > horizon and not math.isclose(
                m["mean_survival_s"], horizon, rel_tol=REL_TOL, abs_tol=SURVIVAL_ABS_TOL):
            raise OracleError("생존 시간이 평가 horizon과 일치하지 않는다")

        rows = report["by_condition"]
        if not isinstance(rows, list) or not rows:
            raise OracleError("평가 조건이 없다")
        raw_presence = [set(row).intersection(RAW_KEYS) for row in rows if isinstance(row, dict)]
        if len(raw_presence) != len(rows):
            raise OracleError("조건별 평가 행은 객체여야 한다")
        if any(keys and keys != RAW_KEYS for keys in raw_presence):
            raise OracleError("조건별 원시 통계 다섯 항목 중 일부만 있다")
        has_raw = all(keys == RAW_KEYS for keys in raw_presence)
        if not has_raw and any(raw_presence):
            raise OracleError("조건별 원시 통계가 행마다 일관되지 않다")

        conditions = []
        row_metrics = []
        if num_envs % len(rows):
            raise OracleError("평가 환경 수가 명령 조건 수의 배수가 아니다")
        expected_envs_int = num_envs // len(rows)
        total_falls = 0
        total_alive = 0
        total_lin_sq = 0.0
        total_yaw_sq = 0.0
        for index, row in enumerate(rows):
            cmd = row["command"]
            if not isinstance(cmd, list) or len(cmd) != 3:
                raise OracleError("평가 명령은 선속도 두 값과 회전 속도 한 값이어야 한다")
            values = tuple(_number(v, "평가 명령", signed=True) for v in cmd)
            conditions.append(values)
            row_fall = _number(row["fall_rate"], "조건별 fall_rate")
            row_lin = _number(row["lin_vel_rmse_mps"], "조건별 lin_vel_rmse_mps")
            row_yaw = _number(row["yaw_rate_rmse_radps"], "조건별 yaw_rate_rmse_radps")
            if row_fall > 1:
                raise OracleError("조건별 낙상 비율이 1보다 크다")
            row_metrics.append((row_fall, row_lin, row_yaw))
            if has_raw:
                row_envs = _integer(row["num_envs"], f"by_condition[{index}].num_envs", minimum=1)
                if row_envs != expected_envs_int:
                    raise OracleError("조건별 num_envs가 report.num_envs와 일치하지 않는다")
                falls = _integer(row["fall_count"], f"by_condition[{index}].fall_count")
                if falls > row_envs:
                    raise OracleError("조건별 fall_count가 num_envs 범위를 벗어났다")
                alive = _integer(row["alive_steps_sum"], f"by_condition[{index}].alive_steps_sum", minimum=1)
                lower_alive = (row_envs - falls) * horizon_steps + falls
                upper_alive = row_envs * horizon_steps
                if not lower_alive <= alive <= upper_alive:
                    raise OracleError("조건별 alive_steps_sum이 생존·낙상 count와 일치하지 않는다")
                lin_sq = _number(row["lin_error_sq_sum"], f"by_condition[{index}].lin_error_sq_sum")
                yaw_sq = _number(row["yaw_error_sq_sum"], f"by_condition[{index}].yaw_error_sq_sum")
                _close(row_fall, _div(falls, row_envs, "조건별 fall_rate"), "조건별 fall_rate")
                _close(row_lin, math.sqrt(_div(lin_sq, alive, "조건별 lin RMSE")), "조건별 lin_vel_rmse_mps")
                _close(row_yaw, math.sqrt(_div(yaw_sq, alive, "조건별 yaw RMSE")), "조건별 yaw_rate_rmse_radps")
                total_falls = _add(total_falls, falls, "전체 fall_count")
                total_alive = _add(total_alive, alive, "전체 alive_steps_sum")
                total_lin_sq = _add(total_lin_sq, lin_sq, "전체 lin_error_sq_sum")
                total_yaw_sq = _add(total_yaw_sq, yaw_sq, "전체 yaw_error_sq_sum")

        if len(set(conditions)) != len(conditions):
            raise OracleError("평가 조건이 중복됐다")
        if has_raw:
            _close(m["fall_rate"], _div(total_falls, num_envs, "전체 fall_rate"), "전체 fall_rate")
            mean_steps = _div(total_alive, num_envs, "전체 mean survival steps")
            _close(m["mean_survival_s"], _mul(mean_steps, dt, "전체 mean_survival_s"),
                   "전체 mean_survival_s", abs_tol=SURVIVAL_ABS_TOL)
            _close(m["lin_vel_rmse_mps"], math.sqrt(_div(total_lin_sq, total_alive, "전체 lin RMSE")),
                   "전체 lin_vel_rmse_mps")
            _close(m["yaw_rate_rmse_radps"], math.sqrt(_div(total_yaw_sq, total_alive, "전체 yaw RMSE")),
                   "전체 yaw_rate_rmse_radps")
        else:
            fall_sum = 0.0
            for row_fall, _, _ in row_metrics:
                fall_sum = _add(fall_sum, row_fall, "조건별 fall_rate 합계")
            mean_fall = _div(fall_sum, len(row_metrics), "조건 평균 fall_rate")
            _close(m["fall_rate"], mean_fall, "전체 fall_rate")
            for metric, position in (("lin_vel_rmse_mps", 1), ("yaw_rate_rmse_radps", 2)):
                low = min(row[position] for row in row_metrics)
                high = max(row[position] for row in row_metrics)
                if m[metric] < low and not math.isclose(m[metric], low, rel_tol=REL_TOL, abs_tol=ABS_TOL):
                    raise OracleError(f"전체 {metric}가 조건별 RMSE 범위와 일치하지 않는다")
                if m[metric] > high and not math.isclose(m[metric], high, rel_tol=REL_TOL, abs_tol=ABS_TOL):
                    raise OracleError(f"전체 {metric}가 조건별 RMSE 범위와 일치하지 않는다")
            nonfall_survival = _mul(1 - m["fall_rate"], horizon, "비낙상 생존 하한")
            fall_survival = _mul(m["fall_rate"], dt, "낙상 생존 하한")
            survival_lower = _add(nonfall_survival, fall_survival, "전체 생존 하한")
            if (m["mean_survival_s"] < survival_lower and not math.isclose(
                    m["mean_survival_s"], survival_lower, rel_tol=REL_TOL, abs_tol=SURVIVAL_ABS_TOL)):
                raise OracleError("전체 mean_survival_s가 낙상률 기반 생존 하한과 일치하지 않는다")

        commands = sorted(conditions)
        identity = {"protocol": report["protocol"], "protocol_sha256_lf": protocol_sha,
                    "task": report["task"], "eval_seed": eval_seed, "num_envs": num_envs,
                    "horizon_steps": horizon_steps, "step_dt": dt, "versions": dict(versions),
                    "commands": [list(c) for c in commands]}
        behavior = {"survival_s": m["mean_survival_s"], "horizon_s": horizon,
                    "fall_frac": m["fall_rate"], "err_xy": m["lin_vel_rmse_mps"],
                    "err_yaw": m["yaw_rate_rmse_radps"]}
        return identity, behavior, has_raw
    except (KeyError, TypeError) as exc:
        raise OracleError(f"고정 평가 입력이 불완전하다: {exc}") from exc


def _gates(gates: dict, identity: dict) -> dict:
    if not isinstance(gates, dict) or set(gates) != GATE_KEYS:
        raise OracleError(f"절대 행동 기준은 {sorted(GATE_KEYS)} 네 항목이 필요하다")
    out = {k: _number(v, k) for k, v in gates.items()}
    if not 0 < out["min_survival_fraction"] <= 1 or out["max_fall_rate"] > 1:
        raise OracleError("생존 비율은 (0, 1], 낙상 비율은 [0, 1] 범위여야 한다")
    commands = identity["commands"]
    lin_sum = 0.0
    yaw_sum = 0.0
    for x, y, z in commands:
        lin_sq = _add(_mul(x, x, "제자리 선속도 오차"), _mul(y, y, "제자리 선속도 오차"),
                      "제자리 선속도 오차")
        lin_sum = _add(lin_sum, lin_sq, "제자리 선속도 오차")
        yaw_sum = _add(yaw_sum, _mul(z, z, "제자리 회전속도 오차"), "제자리 회전속도 오차")
    lin_still = math.sqrt(_div(lin_sum, len(commands), "제자리 선속도 RMSE"))
    yaw_still = math.sqrt(_div(yaw_sum, len(commands), "제자리 회전속도 RMSE"))
    if not (out["max_lin_vel_rmse_mps"] < lin_still and out["max_yaw_rate_rmse_radps"] < yaw_still):
        raise OracleError("추종 오차의 절대 상한은 제자리 정책의 오차보다 작아야 한다")
    return out


def _absolute_checks(behavior: dict, gates: dict) -> dict:
    return {"survival_fraction": _div(behavior["survival_s"], behavior["horizon_s"], "생존 비율") >= gates["min_survival_fraction"],
            "fall_rate": behavior["fall_frac"] <= gates["max_fall_rate"],
            "lin_vel_rmse_mps": behavior["err_xy"] <= gates["max_lin_vel_rmse_mps"],
            "yaw_rate_rmse_radps": behavior["err_yaw"] <= gates["max_yaw_rate_rmse_radps"]}


def register_contract(reference: dict, normals: list[dict], absolute_gates: dict) -> dict:
    """정상 자료와 명시한 절대 기준만으로 계약을 만든다. 결함·개입 후 결과를 받지 않는다.

    정상 반복 자료가 없으면 정상 변동 0 + 기존 RC.MARGIN을 쓴다. 변동을 실측했다는 뜻은 아니다.
    """
    identity, ref, reference_verified = _validated(reference)
    gates = _gates(absolute_gates, identity)
    if not all(_absolute_checks(ref, gates).values()):
        raise OracleError("기준 정책이 절대 행동 기준을 통과하지 못했다")
    if any(ref[k] <= 0 for k in ("survival_s", "err_xy", "err_yaw")):
        raise OracleError("기준 정책의 상대 판정 분모가 0 이하다")
    comparisons = [RC.compare(ref, ref)]
    for key, value in comparisons[0].items():
        _number(value, f"기준 상대 행동 지표 {key}")
    calibration_verified = True
    for normal in normals:
        normal_id, behavior, normal_verified = _validated(normal)
        calibration_verified = calibration_verified and normal_verified
        if normal_id != identity:
            raise OracleError("정상 보정 자료의 프로토콜·seed·격자·환경 수가 기준과 다르다")
        if not all(_absolute_checks(behavior, gates).values()):
            raise OracleError("정상 보정 자료가 절대 행동 기준을 통과하지 못했다")
        comparison = RC.compare(behavior, ref)
        for key, value in comparison.items():
            _number(value, f"정상 보정 상대 행동 지표 {key}")
        comparisons.append(comparison)
    bands = RC.derive_bands(comparisons)
    for key, value in bands.items():
        _number(value, f"상대 행동 허용 범위 {key}")
    contract = {"schema": "behavior_oracle_v1", "evaluation": identity,
                "reference_sha256": digest(reference),
                "reference_checkpoint_sha256": reference["checkpoint"]["sha256"],
                "calibration_sha256": [digest(r) for r in normals], "normal_repeats": len(normals),
                "reference_metrics_verified": reference_verified,
                "calibration_metrics_verified": calibration_verified,
                "margin": dict(RC.MARGIN), "bands": bands, "absolute_gates": gates}
    return {**contract, "contract_sha256": digest(contract)}


def assess(report: dict | None, reference: dict, contract: dict) -> dict:
    """고정 조건에서의 행동 판정. 결과가 없으면 undetermined이며 비교 불일치는 입력 오류다."""
    try:
        payload = {k: v for k, v in contract.items() if k != "contract_sha256"}
        if contract["schema"] != "behavior_oracle_v1" or digest(payload) != contract["contract_sha256"]:
            raise OracleError("행동 계약 해시 또는 버전이 다르다")
        ref_id, ref, reference_verified = _validated(reference)
        if digest(reference) != contract["reference_sha256"] or ref_id != contract["evaluation"]:
            raise OracleError("사전등록한 기준 평가 자료와 다르다")
        if type(contract["reference_metrics_verified"]) is not bool or type(contract["calibration_metrics_verified"]) is not bool:
            raise OracleError("행동 계약의 지표 검증 상태가 유효하지 않다")
        if reference_verified != contract["reference_metrics_verified"]:
            raise OracleError("기준 평가의 지표 검증 상태가 계약과 다르다")
        gates = _gates(contract["absolute_gates"], ref_id)
        if not all(_absolute_checks(ref, gates).values()):
            raise OracleError("기준 정책이 절대 행동 기준을 통과하지 못했다")
        if report is None:
            return {"label": "undetermined", "reason": "missing_fixed_eval", "failed": [],
                    "metrics_verified": False}
        identity, behavior, candidate_verified = _validated(report)
        if identity != ref_id:
            raise OracleError("평가 프로토콜·seed·격자·환경 수·관측 시간이 일치하지 않는다")
        compared = RC.compare(behavior, ref)
        for key, value in compared.items():
            _number(value, f"상대 행동 지표 {key}")
        relative = RC.verdict(compared, contract["bands"])
        absolute = _absolute_checks(behavior, gates)
        failed = [f"relative.{k}" for k in relative["failed"]]
        failed += [f"absolute.{k}" for k, passed in absolute.items() if not passed]
        return {"label": "unhealthy" if failed else relative["label"], "failed": failed,
                "relative": relative, "absolute_checks": absolute, "compared": compared,
                "metrics": {k: report["metrics"][k] for k in METRICS},
                "metrics_verified": bool(candidate_verified and reference_verified
                                         and contract["calibration_metrics_verified"]),
                "checkpoint_sha256": report["checkpoint"]["sha256"], "action_scale": report["action_scale"]}
    except (KeyError, TypeError, ZeroDivisionError) as exc:
        raise OracleError(f"행동 계약 또는 판정 입력이 불완전하다: {exc}") from exc


def recovery_outcome(before: dict, after: dict | None, reference: dict, contract: dict,
                     intervened: bool = True) -> dict:
    """행동 회복과 불필요 개입을 구분한다. 회복만으로 유일 원인이나 개입의 인과성을 확정하지 않는다."""
    if not intervened and after is not None:
        raise OracleError("보류에는 개입 후 결과를 붙일 수 없다")
    pre = assess(before, reference, contract)
    post = assess(after, reference, contract)
    if not intervened:
        outcome = "abstained"
    elif post["label"] == "undetermined" or pre["label"] == "undetermined":
        outcome = "undetermined"
    elif pre["label"] == "healthy":
        outcome = "regression" if post["label"] == "unhealthy" else "unnecessary_intervention"
    else:
        outcome = "recovered" if post["label"] == "healthy" else "not_recovered"
    return {"outcome": outcome, "before": pre, "after": post, "intervened": intervened,
            "causal_confirmation": False}
