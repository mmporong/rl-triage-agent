"""실험 비용 벡터와 비교 공정성 예산을 검증한다.

비용은 통화나 단일 점수로 합치지 않는다. ``gpu_wall_s``는 GPU 프로세스가
차지한 벽시계 구간이며 실제 장치 busy 측정이 아니다. ``execution_wall_s``는
개별 실행 시간의 합이고 전체 과정의 end-to-end elapsed를 뜻하지 않는다.
"""
from __future__ import annotations

import math
import re
from numbers import Real


class CostError(ValueError):
    """비용 기록이나 공정 비교 계약이 유효하지 않을 때 발생한다."""


COST_KEYS = (
    "model_calls",
    "input_tokens",
    "output_tokens",
    "gpu_wall_s",
    "cpu_s",
    "human_review_s",
    "execution_wall_s",
    "simulator_steps",
)
COUNT_KEYS = {"model_calls", "input_tokens", "output_tokens", "simulator_steps"}
STATUSES = ("completed", "failed", "timeout", "unknown", "cancelled", "rejected")
_SHA256 = re.compile(r"[0-9a-fA-F]{64}\Z")


def _number(value: object, label: str, *, integer: bool = False) -> int | float:
    if isinstance(value, bool) or not isinstance(value, Real) or (integer and not isinstance(value, int)):
        raise CostError(f"{label}은 유한한 음수 아닌 수여야 한다")
    try:
        finite = math.isfinite(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise CostError(f"{label}은 유한한 음수 아닌 수여야 한다") from exc
    if not finite or value < 0:
        raise CostError(f"{label}은 유한한 음수 아닌 수여야 한다")
    return value


def _add(left: int | float, right: int | float, label: str) -> int | float:
    try:
        value = left + right
        finite = math.isfinite(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise CostError(f"{label} 합계가 유한한 수의 범위를 벗어났다") from exc
    if not finite:
        raise CostError(f"{label} 합계가 유한한 수의 범위를 벗어났다")
    return value


def _cost_vector(costs: object, label: str) -> dict[str, int | float | None]:
    if not isinstance(costs, dict):
        raise CostError(f"{label}은 객체여야 한다")
    unknown = sorted(set(costs) - set(COST_KEYS))
    if unknown:
        raise CostError(f"{label}에 알 수 없는 비용 키가 있다: {unknown}")
    clean: dict[str, int | float | None] = {}
    for key in COST_KEYS:
        value = costs.get(key)
        clean[key] = None if value is None else _number(value, f"{label}.{key}", integer=key in COUNT_KEYS)
    return clean


def _limits(limits: object) -> dict[str, int | float]:
    if not isinstance(limits, dict):
        raise CostError("limits는 객체여야 한다")
    unknown = sorted(set(limits) - set(COST_KEYS))
    if unknown:
        raise CostError(f"limits에 알 수 없는 비용 키가 있다: {unknown}")
    return {key: _number(value, f"limits.{key}", integer=key in COUNT_KEYS) for key, value in limits.items()}


def _events(events: object) -> list[tuple[str, str, dict[str, int | float | None]]]:
    if not isinstance(events, list):
        raise CostError("events는 목록이어야 한다")
    seen: set[str] = set()
    clean = []
    for index, event in enumerate(events):
        if not isinstance(event, dict):
            raise CostError(f"events[{index}]는 객체여야 한다")
        attempt_id = event.get("attempt_id")
        if not isinstance(attempt_id, str) or not attempt_id.strip():
            raise CostError(f"events[{index}].attempt_id는 빈 문자열이 아닌 문자열이어야 한다")
        if attempt_id in seen:
            raise CostError(f"attempt_id가 중복됐다: {attempt_id}")
        seen.add(attempt_id)
        status = event.get("status")
        if status not in STATUSES:
            raise CostError(f"events[{index}].status가 유효하지 않다: {status!r}")
        clean.append((attempt_id, status, _cost_vector(event.get("costs"), f"events[{index}].costs")))
    return clean


def _budget_result(
    known_totals: dict[str, int | float],
    missing_attempts: dict[str, list[str]],
    limits: dict[str, int | float],
) -> tuple[dict[str, bool | None], str]:
    over_budget: dict[str, bool | None] = {}
    for key, limit in limits.items():
        if known_totals[key] > limit:
            over_budget[key] = True
        elif missing_attempts[key]:
            over_budget[key] = None
        else:
            over_budget[key] = False
    if any(value is True for value in over_budget.values()):
        status = "exceeded"
    elif any(value is None for value in over_budget.values()):
        status = "unknown"
    else:
        status = "within"
    return over_budget, status


def summarize_costs(events: list[dict], limits: dict) -> dict:
    """성공 여부와 무관하게 모든 시도의 비용을 합산하고 누락을 보존한다."""
    clean_events = _events(events)
    clean_limits = _limits(limits)
    known_totals: dict[str, int | float] = {key: 0 for key in COST_KEYS}
    missing_attempts = {key: [] for key in COST_KEYS}
    by_status = {status: 0 for status in STATUSES}

    for attempt_id, status, costs in clean_events:
        by_status[status] += 1
        for key, value in costs.items():
            if value is None:
                missing_attempts[key].append(attempt_id)
            else:
                known_totals[key] = _add(known_totals[key], value, key)

    totals = {
        key: None if missing_attempts[key] else known_totals[key]
        for key in COST_KEYS
    }
    over_budget, budget_status = _budget_result(known_totals, missing_attempts, clean_limits)
    return {
        "totals": totals,
        "known_totals": known_totals,
        "missing_attempts": missing_attempts,
        "limits": clean_limits,
        "over_budget": over_budget,
        "budget_status": budget_status,
        "attempts": len(clean_events),
        "by_status": by_status,
    }


def can_afford(events: list[dict], limits: dict, next_cost: dict) -> dict:
    """다음 시도의 비용 상한을 단순 가산한 추정치로 예산 진입 여부를 판정한다."""
    current = summarize_costs(events, limits)
    estimate = _cost_vector(next_cost, "next_cost")
    projected_known = dict(current["known_totals"])
    projected_missing = {key: list(ids) for key, ids in current["missing_attempts"].items()}
    for key, value in estimate.items():
        if value is None:
            projected_missing[key].append("<next_cost>")
        else:
            projected_known[key] = _add(projected_known[key], value, key)
    projected_totals = {
        key: None if projected_missing[key] else projected_known[key]
        for key in COST_KEYS
    }
    over_budget, reason = _budget_result(projected_known, projected_missing, current["limits"])
    return {
        "allowed": reason == "within",
        "reason": reason,
        "projected": {
            "totals": projected_totals,
            "known_totals": projected_known,
            "missing_attempts": projected_missing,
            "limits": current["limits"],
            "over_budget": over_budget,
            "budget_status": reason,
            "is_estimate": True,
            "estimate_kind": "simple_additive_upper_bound",
        },
    }


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise CostError(f"{label}는 64자리 SHA-256 hex 문자열이어야 한다")
    return value.lower()


def compare_runs(runs: list[dict]) -> dict:
    """동일 정보·사전등록 계약·예산을 쓴 실행들의 비용 벡터를 비교한다."""
    if not isinstance(runs, list):
        raise CostError("runs는 목록이어야 한다")
    names: set[str] = set()
    normalized = []
    for index, run in enumerate(runs):
        if not isinstance(run, dict):
            raise CostError(f"runs[{index}]는 객체여야 한다")
        method = run.get("method")
        if not isinstance(method, str) or not method.strip():
            raise CostError(f"runs[{index}].method는 빈 문자열이 아닌 문자열이어야 한다")
        if method in names:
            raise CostError(f"method가 중복됐다: {method}")
        names.add(method)
        information_sha256 = _sha256(run.get("information_sha256"), f"runs[{index}].information_sha256")
        contract_sha256 = _sha256(run.get("contract_sha256"), f"runs[{index}].contract_sha256")
        limits = _limits(run.get("limits"))
        summary = summarize_costs(run.get("events"), limits)
        normalized.append({
            "method": method,
            "information_sha256": information_sha256,
            "contract_sha256": contract_sha256,
            "summary": summary,
        })

    reasons = []
    if len(normalized) < 2:
        reasons.append("insufficient_runs")
    if normalized:
        first = normalized[0]
        if any(run["information_sha256"] != first["information_sha256"] for run in normalized[1:]):
            reasons.append("information_sha256_mismatch")
        if any(run["contract_sha256"] != first["contract_sha256"] for run in normalized[1:]):
            reasons.append("contract_sha256_mismatch")
        if any(run["summary"]["limits"] != first["summary"]["limits"] for run in normalized[1:]):
            reasons.append("limits_mismatch")
    for run in normalized:
        budget_status = run["summary"]["budget_status"]
        if budget_status != "within":
            reasons.append(f"{run['method']}:budget_{budget_status}")
    return {"comparable": not reasons, "reasons": reasons, "runs": normalized}
