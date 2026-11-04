"""Agent와 baseline의 자원 사용을 공정 계약 아래 비교한다.

금액, 벽시계, GPU 점유 시간을 하나의 점수로 합치지 않는다. 이 모듈은
실측 결과를 생성하지 않고, 이미 기록된 receipt를 검증하고 요약한다.
"""
from __future__ import annotations

import hashlib
import json
import math
from numbers import Real

from rl_triage.experiment_cost import (
    COST_KEYS,
    CostError,
    compare_runs,
    summarize_costs,
)


class ResourceComparisonError(ValueError):
    """자원 비교 기록이나 공정성 계약이 유효하지 않을 때 발생한다."""


CONTRACT_FIELDS = (
    "case_id",
    "reference_sha256",
    "input_sha256",
    "tool_sha256",
    "completion_sha256",
    "runtime_sha256",
    "source_sha256",
    "global_budget_sha256",
)
CACHE_STATES = {"cold", "warm", "unknown"}
LOAD_STATES = {"controlled", "shared", "unknown"}
EXECUTION_CLASSES = {"actual", "recorded", "fixture"}
OUTCOMES = (
    "normal_model_misproposal",
    "host_blocked",
    "hold",
    "confirmed",
    "recovered",
    "timeout",
    "infra_failure",
    "failed",
)


def _finite(value: object, label: str, *, allow_none: bool = False) -> float | None:
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ResourceComparisonError(f"{label}은 유한한 음수 아닌 수여야 한다")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ResourceComparisonError(f"{label}은 유한한 음수 아닌 수여야 한다")
    return number


def _digest(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _events(
    value: object,
    label: str,
    receipt_ids: set[str],
    attempt_ids: set[str],
) -> list[dict]:
    if not isinstance(value, list):
        raise ResourceComparisonError(f"{label}은 목록이어야 한다")
    clean: list[dict] = []
    for index, event in enumerate(value):
        if not isinstance(event, dict):
            raise ResourceComparisonError(f"{label}[{index}]는 객체여야 한다")
        receipt_id = event.get("receipt_id")
        if not isinstance(receipt_id, str) or not receipt_id.strip():
            raise ResourceComparisonError(f"{label}[{index}].receipt_id가 필요하다")
        if receipt_id in receipt_ids:
            raise ResourceComparisonError(f"receipt_id가 중복됐다: {receipt_id}")
        receipt_ids.add(receipt_id)
        attempt_id = event.get("attempt_id")
        if not isinstance(attempt_id, str) or not attempt_id.strip():
            raise ResourceComparisonError(f"{label}[{index}].attempt_id가 필요하다")
        if attempt_id in attempt_ids:
            raise ResourceComparisonError(f"attempt_id가 중복됐다: {attempt_id}")
        attempt_ids.add(attempt_id)
        clean.append({key: event.get(key) for key in ("attempt_id", "status", "costs")})
    return clean


def _span(value: object, label: str) -> tuple[dict[str, float], float]:
    if not isinstance(value, dict):
        raise ResourceComparisonError(f"{label}은 객체여야 한다")
    start = _finite(value.get("start_s"), f"{label}.start_s")
    end = _finite(value.get("end_s"), f"{label}.end_s")
    assert start is not None and end is not None
    if end < start:
        raise ResourceComparisonError(f"{label} 종료가 시작보다 빠르다")
    return {"start_s": start, "end_s": end}, end - start


def _execution_spans(value: object, label: str) -> tuple[list[dict[str, float]], float]:
    if not isinstance(value, list):
        raise ResourceComparisonError(f"{label}은 목록이어야 한다")
    spans: list[dict[str, float]] = []
    total = 0.0
    for index, item in enumerate(value):
        span, elapsed = _span(item, f"{label}[{index}]")
        spans.append(span)
        total += elapsed
    return spans, total


def _contract(run: dict, label: str) -> dict[str, str]:
    clean: dict[str, str] = {}
    for field in CONTRACT_FIELDS:
        value = run.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ResourceComparisonError(f"{label}.{field}가 필요하다")
        if field.endswith("sha256") and (len(value) != 64 or any(c not in "0123456789abcdefABCDEF" for c in value)):
            raise ResourceComparisonError(f"{label}.{field}는 SHA-256 hex여야 한다")
        clean[field] = value.lower() if field.endswith("sha256") else value
    return clean


def _pricing(value: object) -> dict | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ResourceComparisonError("pricing_snapshot은 객체여야 한다")
    for field in ("snapshot_id", "source"):
        if not isinstance(value.get(field), str) or not value[field].strip():
            raise ResourceComparisonError(f"pricing_snapshot.{field}가 필요하다")
    if value.get("currency") != "USD":
        raise ResourceComparisonError("pricing_snapshot.currency는 USD여야 한다")
    input_rate = _finite(value.get("input_usd_per_million_tokens"), "pricing_snapshot.input rate")
    output_rate = _finite(value.get("output_usd_per_million_tokens"), "pricing_snapshot.output rate")
    assert input_rate is not None and output_rate is not None
    if input_rate == 0 or output_rate == 0:
        raise ResourceComparisonError("가격 미상이나 무료 추정을 0달러 가격표로 기록할 수 없다")
    return {
        "snapshot_id": value["snapshot_id"],
        "source": value["source"],
        "currency": "USD",
        "input_usd_per_million_tokens": input_rate,
        "output_usd_per_million_tokens": output_rate,
    }


def _arm(
    run: object,
    label: str,
    receipt_ids: set[str],
    attempt_ids: set[str],
    pricing: dict | None,
) -> dict:
    if not isinstance(run, dict):
        raise ResourceComparisonError(f"{label}은 객체여야 한다")
    method = run.get("method")
    if not isinstance(method, str) or not method.strip():
        raise ResourceComparisonError(f"{label}.method가 필요하다")
    contract = _contract(run, label)
    cache_state = run.get("cache_state")
    if cache_state not in CACHE_STATES:
        raise ResourceComparisonError(f"{label}.cache_state가 유효하지 않다")
    load_state = run.get("load_state")
    if load_state not in LOAD_STATES:
        raise ResourceComparisonError(f"{label}.load_state가 유효하지 않다")
    execution_class = run.get("execution_class")
    if execution_class not in EXECUTION_CLASSES:
        raise ResourceComparisonError(f"{label}.execution_class가 유효하지 않다")
    outcome = run.get("outcome")
    if outcome not in OUTCOMES:
        raise ResourceComparisonError(f"{label}.outcome이 유효하지 않다")
    success = run.get("success")
    if not isinstance(success, bool):
        raise ResourceComparisonError(f"{label}.success는 bool이어야 한다")
    events = _events(run.get("events"), f"{label}.events", receipt_ids, attempt_ids)
    limits = run.get("limits")
    if not isinstance(limits, dict):
        raise ResourceComparisonError(f"{label}.limits는 객체여야 한다")
    try:
        summary = summarize_costs(events, limits)
    except CostError as exc:
        raise ResourceComparisonError(str(exc)) from exc
    end_to_end, end_to_end_elapsed = _span(run.get("end_to_end"), f"{label}.end_to_end")
    execution_spans, individual_execution_sum = _execution_spans(
        run.get("execution_spans"), f"{label}.execution_spans"
    )
    calls = summary["totals"]["model_calls"]
    input_tokens = summary["totals"]["input_tokens"]
    output_tokens = summary["totals"]["output_tokens"]
    if calls == 0 and ((input_tokens not in (0, None)) or (output_tokens not in (0, None))):
        raise ResourceComparisonError(f"{label}은 API 호출 0인데 token 사용이 0이 아니다")
    estimate = None
    if pricing is not None and input_tokens is not None and output_tokens is not None:
        estimate = (
            input_tokens * pricing["input_usd_per_million_tokens"]
            + output_tokens * pricing["output_usd_per_million_tokens"]
        ) / 1_000_000
    actual_billing = _finite(run.get("actual_billing_usd"), f"{label}.actual_billing_usd", allow_none=True)
    return {
        "method": method,
        **contract,
        "cache_state": cache_state,
        "load_state": load_state,
        "execution_class": execution_class,
        "outcome": outcome,
        "success": success,
        "summary": summary,
        "end_to_end": end_to_end,
        "end_to_end_elapsed_s": end_to_end_elapsed,
        "execution_spans": execution_spans,
        "individual_execution_sum_s": individual_execution_sum,
        "api": {
            "model_calls": calls,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "reproducible_estimate_usd": estimate,
            "actual_billing_usd": actual_billing,
            "provider_gpu_measurement": None,
        },
    }


def _delta(agent: dict, baseline: dict) -> dict[str, float | int | None]:
    delta: dict[str, float | int | None] = {}
    for key in COST_KEYS:
        left = agent["summary"]["totals"][key]
        right = baseline["summary"]["totals"][key]
        delta[key] = None if left is None or right is None else left - right
    for key in ("end_to_end_elapsed_s", "individual_execution_sum_s"):
        delta[key] = agent[key] - baseline[key]
    for key in ("reproducible_estimate_usd", "actual_billing_usd"):
        left = agent["api"][key]
        right = baseline["api"][key]
        delta[key] = None if left is None or right is None else left - right
    return delta


def _cost_per_success(arms: list[dict]) -> dict[str, float | int | None]:
    successes = sum(arm["success"] for arm in arms)
    totals: dict[str, float | int | None] = {}
    for key in COST_KEYS:
        values = [arm["summary"]["totals"][key] for arm in arms]
        totals[key] = None if successes == 0 or any(value is None for value in values) else sum(values) / successes
    return totals


def _aggregate_costs(arms: list[dict]) -> dict[str, float | int | None]:
    totals: dict[str, float | int | None] = {}
    for key in COST_KEYS:
        values = [arm["summary"]["totals"][key] for arm in arms]
        totals[key] = None if any(value is None for value in values) else sum(values)
    return totals


def compare_resource_pairs(pairs: list[dict], *, pricing_snapshot: dict | None = None) -> dict:
    """공정 계약을 만족하는 agent-baseline pair의 raw 자원 차이를 반환한다."""
    if not isinstance(pairs, list):
        raise ResourceComparisonError("pairs는 목록이어야 한다")
    pricing = _pricing(pricing_snapshot)
    pair_ids: set[str] = set()
    receipt_ids: set[str] = set()
    attempt_ids: set[str] = set()
    normalized: list[dict] = []
    all_agent: list[dict] = []
    all_baseline: list[dict] = []
    for index, pair in enumerate(pairs):
        if not isinstance(pair, dict):
            raise ResourceComparisonError(f"pairs[{index}]는 객체여야 한다")
        pair_id = pair.get("pair_id")
        if not isinstance(pair_id, str) or not pair_id.strip() or pair_id in pair_ids:
            raise ResourceComparisonError(f"pairs[{index}].pair_id가 없거나 중복됐다")
        pair_ids.add(pair_id)
        common = pair.get("common_preparation")
        if not isinstance(common, dict):
            raise ResourceComparisonError(f"pairs[{index}].common_preparation은 객체여야 한다")
        common_events = _events(
            common.get("events"),
            f"pairs[{index}].common_preparation.events",
            receipt_ids,
            attempt_ids,
        )
        common_limits = common.get("limits")
        if not isinstance(common_limits, dict):
            raise ResourceComparisonError(f"pairs[{index}].common_preparation.limits는 객체여야 한다")
        try:
            common_summary = summarize_costs(common_events, common_limits)
        except CostError as exc:
            raise ResourceComparisonError(str(exc)) from exc
        agent = _arm(pair.get("agent"), f"pairs[{index}].agent", receipt_ids, attempt_ids, pricing)
        baseline = _arm(pair.get("baseline"), f"pairs[{index}].baseline", receipt_ids, attempt_ids, pricing)
        reasons: list[str] = []
        for field in CONTRACT_FIELDS:
            if agent[field] != baseline[field]:
                reasons.append(f"{field}_mismatch")
        if agent["cache_state"] != baseline["cache_state"]:
            reasons.append("cache_state_mismatch")
        elif agent["cache_state"] == "unknown":
            reasons.append("cache_state_unknown")
        if agent["load_state"] != baseline["load_state"]:
            reasons.append("load_state_mismatch")
        elif agent["load_state"] == "unknown":
            reasons.append("load_state_unknown")
        if agent["execution_class"] != baseline["execution_class"]:
            reasons.append("execution_class_mismatch")
        info_fields = CONTRACT_FIELDS[:4] + CONTRACT_FIELDS[5:7]
        comparison_input = []
        for arm in (agent, baseline):
            comparison_input.append({
                "method": arm["method"],
                "information_sha256": _digest({field: arm[field] for field in info_fields}),
                "contract_sha256": _digest({
                    "completion_sha256": arm["completion_sha256"],
                    "global_budget_sha256": arm["global_budget_sha256"],
                }),
                "limits": arm["summary"]["limits"],
                "events": [
                    {"attempt_id": event["attempt_id"], "status": event["status"], "costs": event["costs"]}
                    for event in (pair["agent"]["events"] if arm is agent else pair["baseline"]["events"])
                ],
            })
        try:
            legacy_comparison = compare_runs(comparison_input)
        except CostError as exc:
            raise ResourceComparisonError(str(exc)) from exc
        reasons.extend(reason for reason in legacy_comparison["reasons"] if reason not in reasons)
        result = {
            "pair_id": pair_id,
            "eligible": not reasons,
            "ineligibility_reasons": reasons,
            "common_preparation": common_summary,
            "agent": agent,
            "baseline": baseline,
            "agent_minus_baseline": _delta(agent, baseline),
        }
        normalized.append(result)
        if result["eligible"]:
            all_agent.append(agent)
            all_baseline.append(baseline)

    def outcomes(arms: list[dict]) -> dict[str, int]:
        return {outcome: sum(arm["outcome"] == outcome for arm in arms) for outcome in OUTCOMES}

    eligible_n = len(all_agent)
    return {
        "pairs": normalized,
        "eligible_pairs": eligible_n,
        "ineligible_pairs": len(normalized) - eligible_n,
        "pricing_snapshot": pricing,
        "aggregate": {
            "agent": {
                "outcomes": outcomes(all_agent),
                "all_attempt_costs": _aggregate_costs(all_agent),
                "successful_subset_costs": _aggregate_costs([arm for arm in all_agent if arm["success"]]),
                "cost_per_success": _cost_per_success(all_agent),
            },
            "baseline": {
                "outcomes": outcomes(all_baseline),
                "all_attempt_costs": _aggregate_costs(all_baseline),
                "successful_subset_costs": _aggregate_costs([arm for arm in all_baseline if arm["success"]]),
                "cost_per_success": _cost_per_success(all_baseline),
            },
        },
        "uncertainty": {
            "classification": "small_n_descriptive_only" if eligible_n < 20 else "descriptive_only",
            "n_pairs": eligible_n,
            "statistical_superiority_claimed": False,
        },
        "notes": {
            "common_preparation_excluded_from_arm_delta": True,
            "end_to_end_and_individual_execution_sum_are_distinct": True,
            "provider_gpu_and_human_time_unmeasured_remain_null": True,
            "cost_advantage_requires_eligible_pair": True,
        },
    }
