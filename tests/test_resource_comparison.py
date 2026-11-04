from copy import deepcopy

import pytest

from rl_triage.resource_comparison import (
    ResourceComparisonError,
    compare_resource_pairs,
)

SHA = {
    name: f"{index:x}" * 64
    for index, name in enumerate(
        ("reference", "input", "tool", "completion", "runtime", "source", "budget"), start=1
    )
}


def _costs(**updates):
    value = {
        "model_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "gpu_wall_s": 0.0,
        "cpu_s": 1.0,
        "human_review_s": None,
        "execution_wall_s": 4.0,
        "simulator_steps": 10,
    }
    value.update(updates)
    return value


def _arm(method: str, receipt: str, *, calls: int, outcome: str, success: bool) -> dict:
    return {
        "method": method,
        "case_id": "case-01",
        "reference_sha256": SHA["reference"],
        "input_sha256": SHA["input"],
        "tool_sha256": SHA["tool"],
        "completion_sha256": SHA["completion"],
        "runtime_sha256": SHA["runtime"],
        "source_sha256": SHA["source"],
        "global_budget_sha256": SHA["budget"],
        "cache_state": "cold",
        "load_state": "controlled",
        "execution_class": "actual",
        "limits": {"model_calls": 5, "execution_wall_s": 100},
        "events": [
            {
                "attempt_id": f"{method}-failed",
                "receipt_id": f"{receipt}-failed",
                "status": "failed",
                "costs": _costs(cpu_s=2.0, execution_wall_s=3.0),
            },
            {
                "attempt_id": f"{method}-final",
                "receipt_id": receipt,
                "status": "completed",
                "costs": _costs(
                    model_calls=calls,
                    input_tokens=100 * calls,
                    output_tokens=20 * calls,
                    cpu_s=3.0,
                    execution_wall_s=7.0,
                ),
            },
        ],
        "outcome": outcome,
        "success": success,
        "end_to_end": {"start_s": 10.0, "end_s": 18.0},
        "execution_spans": [{"start_s": 10.0, "end_s": 16.0}, {"start_s": 12.0, "end_s": 18.0}],
        "actual_billing_usd": None,
    }


def _pair() -> dict:
    return {
        "pair_id": "pair-01",
        "common_preparation": {
            "limits": {"cpu_s": 10},
            "events": [{
                "attempt_id": "prepare",
                "receipt_id": "prepare-receipt",
                "status": "completed",
                "costs": _costs(cpu_s=4.0, execution_wall_s=0.0),
            }],
        },
        "agent": _arm("agent", "agent-receipt", calls=1, outcome="recovered", success=True),
        "baseline": _arm("baseline", "baseline-receipt", calls=0, outcome="hold", success=False),
    }


def test_comparison_keeps_failures_shared_preparation_and_timing_semantics_separate() -> None:
    report = compare_resource_pairs(
        [_pair()],
        pricing_snapshot={
            "snapshot_id": "price-2026-11-04",
            "source": "provider price page snapshot",
            "currency": "USD",
            "input_usd_per_million_tokens": 2.0,
            "output_usd_per_million_tokens": 4.0,
        },
    )

    pair = report["pairs"][0]
    assert pair["eligible"] is True
    assert pair["common_preparation"]["totals"]["cpu_s"] == 4.0
    assert pair["agent"]["summary"]["attempts"] == 2
    assert pair["agent"]["summary"]["totals"]["cpu_s"] == 5.0
    assert pair["agent"]["end_to_end_elapsed_s"] == 8.0
    assert pair["agent"]["individual_execution_sum_s"] == 12.0
    assert pair["agent"]["api"]["reproducible_estimate_usd"] == pytest.approx(0.00028)
    assert pair["baseline"]["api"]["model_calls"] == 0
    assert pair["baseline"]["api"]["reproducible_estimate_usd"] == 0
    assert pair["agent_minus_baseline"]["model_calls"] == 1
    assert report["aggregate"]["baseline"]["cost_per_success"]["cpu_s"] is None
    assert report["aggregate"]["agent"]["all_attempt_costs"]["cpu_s"] == 5.0
    assert report["aggregate"]["agent"]["successful_subset_costs"]["cpu_s"] == 5.0
    assert report["aggregate"]["baseline"]["successful_subset_costs"]["cpu_s"] == 0
    assert report["uncertainty"]["classification"] == "small_n_descriptive_only"


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        (lambda pair: pair["baseline"].update({"cache_state": "warm"}), "cache_state_mismatch"),
        (lambda pair: pair["baseline"].update({"input_sha256": "f" * 64}), "input_sha256_mismatch"),
        (lambda pair: pair["baseline"].update({"execution_class": "fixture"}), "execution_class_mismatch"),
        (lambda pair: pair["baseline"].update({"load_state": "shared"}), "load_state_mismatch"),
        (lambda pair: pair["baseline"].update({"cache_state": "unknown"}) or pair["agent"].update({"cache_state": "unknown"}), "cache_state_unknown"),
    ],
)
def test_materially_different_conditions_are_ineligible(mutation, reason) -> None:
    pair = _pair()
    mutation(pair)
    result = compare_resource_pairs([pair])["pairs"][0]
    assert result["eligible"] is False
    assert reason in result["ineligibility_reasons"]


def test_duplicate_attempt_receipt_nan_and_zero_price_are_rejected() -> None:
    pair = _pair()
    pair["agent"]["events"][1]["attempt_id"] = pair["agent"]["events"][0]["attempt_id"]
    with pytest.raises(ResourceComparisonError, match="attempt_id"):
        compare_resource_pairs([pair])

    pair = _pair()
    pair["baseline"]["events"][0]["receipt_id"] = "agent-receipt"
    with pytest.raises(ResourceComparisonError, match="receipt_id"):
        compare_resource_pairs([pair])

    pair = _pair()
    pair["agent"]["events"][0]["costs"]["cpu_s"] = float("nan")
    with pytest.raises(ResourceComparisonError, match="유한"):
        compare_resource_pairs([pair])

    with pytest.raises(ResourceComparisonError, match="0달러"):
        compare_resource_pairs(
            [_pair()],
            pricing_snapshot={
                "snapshot_id": "unknown",
                "source": "none",
                "currency": "USD",
                "input_usd_per_million_tokens": 0,
                "output_usd_per_million_tokens": 0,
            },
        )


def test_unknown_price_and_missing_provider_gpu_or_human_measurement_stay_null() -> None:
    report = compare_resource_pairs([_pair()])
    agent = report["pairs"][0]["agent"]
    assert agent["api"]["reproducible_estimate_usd"] is None
    assert agent["api"]["actual_billing_usd"] is None
    assert agent["api"]["provider_gpu_measurement"] is None
    assert agent["summary"]["totals"]["human_review_s"] is None


def test_api_tokens_without_a_call_are_rejected() -> None:
    pair = deepcopy(_pair())
    pair["baseline"]["events"][1]["costs"]["input_tokens"] = 1
    with pytest.raises(ResourceComparisonError, match="API 호출 0"):
        compare_resource_pairs([pair])


def test_receipts_and_attempts_cannot_be_reused_across_pairs() -> None:
    second = deepcopy(_pair())
    second["pair_id"] = "pair-02"
    with pytest.raises(ResourceComparisonError, match="receipt_id|attempt_id"):
        compare_resource_pairs([_pair(), second])


def test_missing_contract_field_is_rejected() -> None:
    pair = _pair()
    del pair["agent"]["runtime_sha256"]
    with pytest.raises(ResourceComparisonError, match="runtime_sha256"):
        compare_resource_pairs([pair])
