"""실험 비용 벡터 합산, 예산 차단, 비교 공정성 계약을 검사한다."""
import math

import pytest

from rl_triage.experiment_cost import CostError, can_afford, compare_runs, summarize_costs


INFO = "a" * 64
CONTRACT = "b" * 64


def _event(attempt_id, status="completed", **costs):
    return {"attempt_id": attempt_id, "status": status, "costs": costs}


def _run(method, *, info=INFO, contract=CONTRACT, limits=None, events=None):
    return {"method": method, "information_sha256": info, "contract_sha256": contract,
            "limits": {"model_calls": 3} if limits is None else limits,
            "events": [_event(f"{method}-1", model_calls=1)] if events is None else events}


def test_all_attempts_including_failures_and_timeouts_are_charged():
    events = [
        _event("ok", model_calls=1, input_tokens=100),
        _event("bad", "failed", model_calls=2, input_tokens=30),
        _event("slow", "timeout", model_calls=1, input_tokens=20),
    ]
    result = summarize_costs(events, {"model_calls": 4, "input_tokens": 150})
    assert result["totals"]["model_calls"] == 4
    assert result["totals"]["input_tokens"] == 150
    assert result["by_status"]["completed"] == 1
    assert result["by_status"]["failed"] == 1
    assert result["by_status"]["timeout"] == 1
    assert result["attempts"] == 3 and result["budget_status"] == "within"


def test_missing_cost_stays_unknown_while_known_partial_sum_is_preserved():
    result = summarize_costs([
        _event("known", model_calls=2, gpu_wall_s=7.5),
        _event("missing", "failed", model_calls=None),
    ], {"model_calls": 5})
    assert result["totals"]["model_calls"] is None
    assert result["known_totals"]["model_calls"] == 2
    assert result["missing_attempts"]["model_calls"] == ["missing"]
    assert result["budget_status"] == "unknown"
    assert result["over_budget"]["model_calls"] is None


def test_known_overspend_wins_even_when_another_attempt_is_missing():
    result = summarize_costs([
        _event("over", model_calls=6),
        _event("missing", "unknown", model_calls=None),
    ], {"model_calls": 5})
    assert result["budget_status"] == "exceeded"
    assert result["over_budget"]["model_calls"] is True


@pytest.mark.parametrize("events, limits", [
    ([_event("same", model_calls=1), _event("same", model_calls=1)], {}),
    ([_event("x", model_calls=True)], {}),
    ([_event("x", model_calls=-1)], {}),
    ([_event("x", model_calls=math.inf)], {}),
    ([_event("x", typo=1)], {}),
    ([_event("x", status="retried", model_calls=1)], {}),
    ([_event("x", model_calls=1)], {"model_call": 1}),
    ([_event("x", model_calls=1)], {"model_calls": math.nan}),
])
def test_invalid_or_duplicate_cost_records_are_rejected(events, limits):
    with pytest.raises(CostError):
        summarize_costs(events, limits)


def test_budget_boundary_and_projected_upper_bound():
    events = [_event("used", model_calls=2, gpu_wall_s=4)]
    exact = can_afford(events, {"model_calls": 3}, {"model_calls": 1})
    assert exact["allowed"] and exact["reason"] == "within"
    assert exact["projected"]["totals"]["model_calls"] == 3
    assert exact["projected"]["is_estimate"]
    assert exact["projected"]["estimate_kind"] == "simple_additive_upper_bound"

    over = can_afford(events, {"model_calls": 2}, {"model_calls": 1})
    assert not over["allowed"] and over["reason"] == "exceeded"


@pytest.mark.parametrize("where", ["event", "limit", "next"])
@pytest.mark.parametrize("invalid", [1.5, True])
def test_count_metrics_require_integers_everywhere(where, invalid):
    events = [_event("used", model_calls=1)]
    limits = {"model_calls": 3}
    next_cost = {"model_calls": 1}
    if where == "event":
        events[0]["costs"]["model_calls"] = invalid
    elif where == "limit":
        limits["model_calls"] = invalid
    else:
        next_cost["model_calls"] = invalid
    with pytest.raises(CostError):
        can_afford(events, limits, next_cost)


def test_count_totals_remain_integers_and_fractional_seconds_are_valid():
    result = can_afford(
        [_event("used", model_calls=1, input_tokens=5, gpu_wall_s=0.25)],
        {"model_calls": 2, "gpu_wall_s": 1.0},
        {"model_calls": 1, "input_tokens": 3, "gpu_wall_s": 0.5},
    )
    assert result["allowed"]
    assert result["projected"]["totals"]["model_calls"] == 2
    assert isinstance(result["projected"]["totals"]["model_calls"], int)
    assert result["projected"]["totals"]["gpu_wall_s"] == 0.75


def test_numeric_overflow_is_rejected_in_values_and_sums():
    with pytest.raises(CostError):
        summarize_costs([_event("huge", simulator_steps=10 ** 10000)], {})
    with pytest.raises(CostError, match="합계"):
        summarize_costs([
            _event("first", gpu_wall_s=1e308),
            _event("second", gpu_wall_s=1e308),
        ], {})
    with pytest.raises(CostError, match="합계"):
        can_afford([_event("used", gpu_wall_s=1e308)], {}, {"gpu_wall_s": 1e308})


def test_unrecorded_limited_metric_blocks_next_attempt():
    existing_missing = can_afford([_event("old", model_calls=None)], {"model_calls": 4}, {"model_calls": 1})
    next_missing = can_afford([_event("old", model_calls=1)], {"model_calls": 4}, {})
    assert (existing_missing["allowed"], existing_missing["reason"]) == (False, "unknown")
    assert (next_missing["allowed"], next_missing["reason"]) == (False, "unknown")
    assert "<next_cost>" in next_missing["projected"]["missing_attempts"]["model_calls"]


@pytest.mark.parametrize("changed, reason", [
    ({"info": "c" * 64}, "information_sha256_mismatch"),
    ({"contract": "d" * 64}, "contract_sha256_mismatch"),
    ({"limits": {"model_calls": 4}}, "limits_mismatch"),
])
def test_fair_comparison_rejects_information_contract_or_budget_mismatch(changed, reason):
    result = compare_runs([_run("agent"), _run("baseline", **changed)])
    assert not result["comparable"] and reason in result["reasons"]
    assert len(result["runs"]) == 2


def test_fair_comparison_rejects_over_budget_or_unknown_run():
    over = _run("over", limits={"model_calls": 1}, events=[_event("over-1", model_calls=2)])
    unknown = _run("unknown", limits={"model_calls": 1}, events=[_event("unknown-1", model_calls=None)])
    result = compare_runs([over, unknown])
    assert not result["comparable"]
    assert "over:budget_exceeded" in result["reasons"]
    assert "unknown:budget_unknown" in result["reasons"]


def test_matching_runs_are_comparable_without_collapsing_cost_vector():
    result = compare_runs([
        _run("agent", events=[_event("a1", model_calls=1, input_tokens=100, gpu_wall_s=3)]),
        _run("baseline", events=[_event("b1", model_calls=1, input_tokens=80, gpu_wall_s=4)]),
    ])
    assert result["comparable"] and result["reasons"] == []
    assert result["runs"][0]["summary"]["known_totals"]["input_tokens"] == 100
    assert result["runs"][1]["summary"]["known_totals"]["gpu_wall_s"] == 4


@pytest.mark.parametrize("runs", [[], [_run("only")]])
def test_comparison_requires_at_least_two_runs(runs):
    result = compare_runs(runs)
    assert not result["comparable"]
    assert result["reasons"] == ["insufficient_runs"]


def test_duplicate_method_and_invalid_hash_are_rejected():
    with pytest.raises(CostError, match="중복"):
        compare_runs([_run("same"), _run("same")])
    with pytest.raises(CostError, match="SHA-256"):
        compare_runs([_run("agent", info="short")])
