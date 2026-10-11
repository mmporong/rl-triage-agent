"""동일 동결 정책의 3-arm 고정 평가 결과를 엄격히 검증하고 회복을 판정한다.

GPU 실행과 결과 파일의 출처·프로세스 소유권 검증은 실행 부모가 맡는다. 이 모듈은
``fixed_eval_v1``이 기록한 JSON 본문만 검증하며 모델·네트워크·GPU를 사용하지 않는다.
"""
from __future__ import annotations

import copy
import math
from datetime import datetime, timezone
from typing import Any

from rl_triage import recovery as RC


PROTOCOL = "fixed_eval_v1"
TASK = "Isaac-Velocity-Flat-Unitree-Go2-v0"
EVAL_SEED = 2026
ENVS_PER_CONDITION = 40
HORIZON_STEPS = 1000
STEP_DT = 0.02
NUM_ENVS = 1040
WALK_FRACTION = 0.5
REF_FALL_MAX = 0.05
REL_TOL = 3.0e-6
ABS_TOL = 1.0e-6

ARMS = ("fault_before", "fault_sham", "callback_fixed")
GRID = tuple(
    (float(vx), float(vy), float(yaw))
    for vx in (-1, 0, 1)
    for vy in (-1, 0, 1)
    for yaw in (-1, 0, 1)
    if (vx, vy, yaw) != (0, 0, 0)
)
STANDSTILL = {
    "lin_vel_rmse_mps": math.sqrt(sum(vx * vx + vy * vy for vx, vy, _ in GRID) / len(GRID)),
    "yaw_rate_rmse_radps": math.sqrt(sum(yaw * yaw for _, _, yaw in GRID) / len(GRID)),
}

TOP_KEYS = {
    "protocol", "protocol_sha256_lf", "task", "eval_seed", "num_envs", "horizon_steps", "step_dt",
    "action_scale", "standstill", "versions", "name", "checkpoint", "metrics", "by_condition", "elapsed_s",
    "started_at", "finished_at",
}
METRIC_KEYS = {"fall_rate", "mean_survival_s", "lin_vel_rmse_mps", "yaw_rate_rmse_radps"}
ROW_KEYS = {
    "command", "fall_rate", "lin_vel_rmse_mps", "yaw_rate_rmse_radps", "num_envs", "fall_count",
    "alive_steps_sum", "lin_error_sq_sum", "yaw_error_sq_sum",
}
VERSION_KEYS = {"torch", "gpu", "isaaclab", "isaaclab_tasks", "isaaclab_rl", "rsl-rl-lib", "isaacsim"}
CLAIMS = {
    "policy_weights_repaired": False,
    "cause_confirmed": False,
    "model_selected_intervention": False,
    "superiority_proven": False,
    "generalization_proven": False,
    "statistical_significance_proven": False,
}
SCOPE = "동일 동결 정책·한 고정 task/seed/grid의 등록 콜백 교정 후 행동 회복"


def _mapping(value: Any, label: str, keys: set[str]) -> dict:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{label} 필드가 고정 스키마와 다르다")
    return value


def _number(value: Any, label: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label}은 유한한 숫자여야 한다")
    result = float(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{label}은 {minimum} 이상이어야 한다")
    return result


def _integer(value: Any, label: str, *, minimum: int, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label}은 정수여야 한다")
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{label} 범위가 잘못됐다")
    return value


def _close(actual: Any, expected: float, label: str) -> float:
    value = _number(actual, label)
    if not math.isclose(value, expected, rel_tol=REL_TOL, abs_tol=ABS_TOL):
        raise ValueError(f"{label} 재집계 값이 다르다")
    return value


def _utc(value: Any, label: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{label}은 UTC ISO-8601 문자열이어야 한다")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{label}은 UTC ISO-8601 문자열이어야 한다") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError(f"{label}은 UTC 시각이어야 한다")
    return parsed


def _strict_equal(actual: Any, expected: Any, label: str) -> None:
    if isinstance(actual, bool) or isinstance(expected, bool) or actual != expected or type(actual) is not type(expected):
        raise ValueError(f"{label}이 기대값과 다르다")


def validate_fixed_report(report: dict, expected: dict) -> dict:
    """``fixed_eval_v1`` 결과 본문과 호출자가 고정한 실행 결합값을 검증한다."""
    r = _mapping(report, "report", TOP_KEYS)
    required = {"name", "checkpoint_file", "checkpoint_sha256", "action_scale", "protocol_sha256_lf"}
    if not isinstance(expected, dict) or not required.issubset(expected):
        raise ValueError("expected 필수 결합값이 없다")

    fixed = {
        "protocol": PROTOCOL, "task": TASK, "eval_seed": EVAL_SEED, "num_envs": NUM_ENVS,
        "horizon_steps": HORIZON_STEPS, "step_dt": STEP_DT,
    }
    for key, value in fixed.items():
        _strict_equal(r[key], value, key)
    for key in ("name", "protocol_sha256_lf", "action_scale"):
        _strict_equal(r[key], expected[key], key)
    _number(r["action_scale"], "action_scale")
    for key in ("protocol_sha256_lf",):
        if not isinstance(r[key], str) or len(r[key]) != 64 or any(c not in "0123456789abcdef" for c in r[key]):
            raise ValueError(f"{key}은 소문자 SHA-256이어야 한다")

    checkpoint = _mapping(r["checkpoint"], "checkpoint", {"file", "sha256"})
    _strict_equal(checkpoint["file"], expected["checkpoint_file"], "checkpoint.file")
    _strict_equal(checkpoint["sha256"], expected["checkpoint_sha256"], "checkpoint.sha256")
    if not isinstance(checkpoint["file"], str) or not checkpoint["file"] or "/" in checkpoint["file"] or "\\" in checkpoint["file"]:
        raise ValueError("checkpoint.file은 파일 이름이어야 한다")
    if not isinstance(checkpoint["sha256"], str) or len(checkpoint["sha256"]) != 64 \
            or any(c not in "0123456789abcdef" for c in checkpoint["sha256"]):
        raise ValueError("checkpoint.sha256은 소문자 SHA-256이어야 한다")

    standstill = _mapping(r["standstill"], "standstill", set(STANDSTILL))
    for key, value in STANDSTILL.items():
        _close(standstill[key], value, f"standstill.{key}")

    versions = _mapping(r["versions"], "versions", VERSION_KEYS)
    for key, value in versions.items():
        if value is not None and (not isinstance(value, str) or not value):
            raise ValueError(f"versions.{key}은 문자열 또는 null이어야 한다")
    if not isinstance(versions["torch"], str) or not isinstance(versions["gpu"], str):
        raise ValueError("torch와 gpu 버전 결합값이 없다")

    rows = r["by_condition"]
    if not isinstance(rows, list) or len(rows) != len(GRID):
        raise ValueError("by_condition은 순서가 고정된 26개 행이어야 한다")
    total_falls = 0
    total_alive = 0
    total_lin_sq = 0.0
    total_yaw_sq = 0.0
    for index, (row_value, command) in enumerate(zip(rows, GRID, strict=True)):
        row = _mapping(row_value, f"by_condition[{index}]", ROW_KEYS)
        if not isinstance(row["command"], list) or len(row["command"]) != 3:
            raise ValueError(f"by_condition[{index}].command 형식이 잘못됐다")
        for axis, (actual, wanted) in enumerate(zip(row["command"], command, strict=True)):
            _strict_equal(actual, wanted, f"by_condition[{index}].command[{axis}]")
        count = _integer(row["num_envs"], f"by_condition[{index}].num_envs", minimum=40, maximum=40)
        falls = _integer(row["fall_count"], f"by_condition[{index}].fall_count", minimum=0, maximum=count)
        alive = _integer(row["alive_steps_sum"], f"by_condition[{index}].alive_steps_sum",
                         minimum=1, maximum=ENVS_PER_CONDITION * HORIZON_STEPS)
        lin_sq = _number(row["lin_error_sq_sum"], f"by_condition[{index}].lin_error_sq_sum", minimum=0.0)
        yaw_sq = _number(row["yaw_error_sq_sum"], f"by_condition[{index}].yaw_error_sq_sum", minimum=0.0)
        _close(row["fall_rate"], falls / count, f"by_condition[{index}].fall_rate")
        _close(row["lin_vel_rmse_mps"], math.sqrt(lin_sq / alive), f"by_condition[{index}].lin_vel_rmse_mps")
        _close(row["yaw_rate_rmse_radps"], math.sqrt(yaw_sq / alive), f"by_condition[{index}].yaw_rate_rmse_radps")
        total_falls += falls
        total_alive += alive
        total_lin_sq += lin_sq
        total_yaw_sq += yaw_sq

    metrics = _mapping(r["metrics"], "metrics", METRIC_KEYS)
    recomputed = {
        "fall_rate": total_falls / NUM_ENVS,
        "mean_survival_s": total_alive / NUM_ENVS * STEP_DT,
        "lin_vel_rmse_mps": math.sqrt(total_lin_sq / total_alive),
        "yaw_rate_rmse_radps": math.sqrt(total_yaw_sq / total_alive),
    }
    for key, value in recomputed.items():
        _close(metrics[key], value, f"metrics.{key}")

    elapsed = _number(r["elapsed_s"], "elapsed_s", minimum=0.0)
    started = _utc(r["started_at"], "started_at")
    finished = _utc(r["finished_at"], "finished_at")
    duration = (finished - started).total_seconds()
    if duration < 0 or abs(duration - elapsed) > 1.1:
        raise ValueError("시작·종료 시각과 elapsed_s가 결합되지 않는다")
    return copy.deepcopy(r)


def _behavior(report: dict) -> dict:
    metrics = report["metrics"]
    return {
        "survival_s": metrics["mean_survival_s"],
        "horizon_s": report["horizon_steps"] * report["step_dt"],
        "fall_frac": metrics["fall_rate"],
        "err_xy": metrics["lin_vel_rmse_mps"],
        "err_yaw": metrics["yaw_rate_rmse_radps"],
    }


def summarize_triplet(reports: dict) -> dict:
    """세 본문의 수치 조건을 비교한다. 실행 출처와 회복 증명은 부모가 결합한다."""
    if not isinstance(reports, dict) or set(reports) != set(ARMS):
        raise ValueError("reports는 fault_before, fault_sham, callback_fixed 세 arm이어야 한다")
    checked = {}
    for arm in ARMS:
        report = reports[arm]
        if not isinstance(report, dict):
            raise ValueError("검증되지 않은 report 스키마가 섞여 있다")
        try:
            checkpoint = report["checkpoint"]
            expected = {"name": arm, "checkpoint_file": checkpoint["file"],
                        "checkpoint_sha256": checkpoint["sha256"], "action_scale": report["action_scale"],
                        "protocol_sha256_lf": report["protocol_sha256_lf"]}
        except (KeyError, TypeError) as exc:
            raise ValueError("검증되지 않은 report 스키마가 섞여 있다") from exc
        checked[arm] = validate_fixed_report(report, expected)
    reports = checked
    common_keys = (
        "protocol", "protocol_sha256_lf", "task", "eval_seed", "num_envs", "horizon_steps", "step_dt",
        "action_scale", "standstill", "versions", "checkpoint",
    )
    reference = reports[ARMS[0]]
    for arm in ARMS[1:]:
        for key in common_keys:
            if reports[arm][key] != reference[key]:
                raise ValueError(f"3-arm 공통값 {key}이 다르다")

    behavior = {arm: _behavior(reports[arm]) for arm in ARMS}
    corrected = behavior["callback_fixed"]
    bands = RC.derive_bands([RC.compare(corrected, corrected)])
    relative = {
        arm: RC.verdict(RC.compare(behavior[arm], corrected), bands)
        for arm in ("fault_before", "fault_sham")
    }
    failed_before = relative["fault_before"]["failed"]
    failed_sham = relative["fault_sham"]["failed"]
    same_nonempty_failure = bool(failed_before) and set(failed_before) == set(failed_sham)
    both_unhealthy = all(relative[arm]["label"] == "unhealthy" for arm in relative)

    sham_checks = {
        "sham_vs_before": RC.verdict(
            RC.compare(behavior["fault_sham"], behavior["fault_before"]),
            RC.derive_bands([RC.compare(behavior["fault_before"], behavior["fault_before"])]),
        ),
        "before_vs_sham": RC.verdict(
            RC.compare(behavior["fault_before"], behavior["fault_sham"]),
            RC.derive_bands([RC.compare(behavior["fault_sham"], behavior["fault_sham"])]),
        ),
    }
    sham_equivalent = all(item["label"] == "healthy" for item in sham_checks.values())
    corrected_walks = (
        corrected["err_xy"] < WALK_FRACTION * reports["callback_fixed"]["standstill"]["lin_vel_rmse_mps"]
        and corrected["fall_frac"] <= REF_FALL_MAX
    )
    conditions_met = corrected_walks and both_unhealthy and same_nonempty_failure and sham_equivalent
    control_valid = sham_equivalent and set(failed_before) == set(failed_sham)
    if not control_valid:
        status = "control_invalid"
    elif not corrected_walks:
        status = "no_recovery"
    elif not both_unhealthy or not same_nonempty_failure:
        status = "no_failure_observed"
    else:
        status = "recovery_conditions_met"
    return {
        "status": status,
        "recovery_conditions_met": conditions_met,
        "execution_provenance_verified": False,
        "control_valid": control_valid,
        "corrected_walks": corrected_walks,
        "faults_both_unhealthy": both_unhealthy,
        "same_nonempty_failed_metrics": same_nonempty_failure,
        "sham_equivalent": sham_equivalent,
        "failed_metrics": {arm: relative[arm]["failed"] for arm in relative},
        "relative_to_corrected": relative,
        "sham_equivalence_checks": sham_checks,
        "behavior": behavior,
        "bands": bands,
        "claims": dict(CLAIMS),
        "scope": SCOPE,
    }
