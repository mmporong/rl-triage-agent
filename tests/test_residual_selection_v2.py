from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))
import residual_selection_v2 as S  # noqa: E402


INPUT_BINDINGS = {
    "case": {"checkpoint_sha256": "a" * 64, "telemetry_sha256": "b" * 64},
    "reference": {"checkpoint_sha256": "c" * 64, "telemetry_sha256": "d" * 64},
}


def packet(epoch=1, applied=None, *, prior: dict | str | None = "auto", budget=7):
    applied = [] if applied is None else applied
    if prior == "auto":
        prior = None if epoch == 1 else {
            "verdict": "not_recovered", "correction_id": applied[0],
            "result_sha256": "a" * 64, "checkpoint_sha256": "b" * 64,
        }
    assert isinstance(prior, dict) or prior is None
    return S.build_packet(epoch=epoch, applied_corrections=applied, prior_receipt=prior,
                          facts={"telemetry:error": 1.2, "telemetry:alive": 20},
                          input_bindings=INPUT_BINDINGS,
                          budget_left=budget, execution_budget_s=1200)


def content(probe="P_value", facts=None, action="continue"):
    return json.dumps({"action": action, "probe": probe if action == "continue" else None,
                       "fact_ids": ["telemetry:error"] if facts is None else facts})


def test_prompt_uses_subtype_candidates_and_three_field_output_contract():
    messages = S.diagnosis_messages(packet())
    body = json.loads(messages[-1]["content"])
    ids = [subtype for subtypes in body["candidate_groups"].values() for subtype in subtypes]
    assert set(ids) == set(S.C1.registry()["mechanisms"])
    assert set(body["candidate_groups"]) <= set(S.PUBLIC_CATEGORIES)
    assert "subtype_registry" not in body and "expected_outcomes" not in messages[0]["content"]
    assert "ranking" not in messages[0]["content"]
    assert all(set(row) == {"probe", "measures", "cost_gpu_s"} for row in body["eligible_probes"])
    assert "input_bindings" not in body
    assert not any(value in messages[-1]["content"]
                   for role in INPUT_BINDINGS.values() for value in role.values())


def test_input_bindings_are_required_exact_and_change_packet_digest():
    value = packet()
    original = S.select(S.assess(content(), value), value)["packet_digest"]
    changed = json.loads(json.dumps(value))
    changed["input_bindings"]["case"]["telemetry_sha256"] = "e" * 64
    revised = S.select(S.assess(content(), changed), changed)["packet_digest"]
    assert original != revised

    bad_rows = []
    missing = json.loads(json.dumps(value))
    del missing["input_bindings"]
    bad_rows.append(missing)
    extra = json.loads(json.dumps(value))
    extra["input_bindings"]["case"]["extra"] = "f" * 64
    bad_rows.append(extra)
    uppercase = json.loads(json.dumps(value))
    uppercase["input_bindings"]["reference"]["checkpoint_sha256"] = "A" * 64
    bad_rows.append(uppercase)
    short = json.loads(json.dumps(value))
    short["input_bindings"]["case"]["checkpoint_sha256"] = "a" * 63
    bad_rows.append(short)
    for bad in bad_rows:
        with pytest.raises(ValueError, match="입력 SHA 결합|packet 필드"):
            S.validate_packet(bad)


def test_old_seven_vs_six_output_is_format_failure_and_falls_back_to_rule():
    old = json.dumps({"ranking": ["reward", "actuator", "exploration", "optimizer_value",
                                       "physics", "termination", "optimizer_update_count"],
                      "probe": "P_reward", "action": "continue",
                      "fact_ids": ["telemetry:error"]})
    review = S.assess(old, packet())
    assert review["status"] == "invalid" and review["codes"] == ["F_FIELDS"]
    selection = S.select(review, packet())
    assert selection["selected_by"] == "registered_rule"
    assert selection["probe"] == S.M._next(S.C1.registry(), S.C1.registry()["mechanisms"], {})
    assert selection["fact_ids"] == [] and selection["host_decision"]["claims"] == []
    assert len(selection["packet_digest"]) == len(selection["registry_digest"]) == 64


def test_unknown_fact_and_unregistered_probe_are_semantic_failures():
    unknown_fact = S.assess(content(facts=["missing"]), packet())
    assert unknown_fact["format_status"] == "valid" and unknown_fact["codes"] == ["S_FACT_UNKNOWN"]
    unknown_probe = S.assess(content(probe="P_missing", facts=["missing"]), packet())
    assert unknown_probe["codes"] == ["S_FACT_UNKNOWN", "S_PROBE_UNKNOWN"]


def test_format_review_collects_independent_errors_for_one_feedback():
    review = S.format_review(json.dumps({"action": "bad", "probe": 3, "fact_ids": []}), packet())
    assert review["codes"] == ["F_ACTION", "F_FACT_IDS"]
    feedback = S.feedback_message(review)["content"]
    assert "F_ACTION" in feedback and "F_FACT_IDS" in feedback


def test_nonpartitioning_reward_probe_after_reward_repair_uses_p_value_fallback():
    value = packet(2, [S.C1.REWARD])
    shown = json.loads(S.diagnosis_messages(value)[-1]["content"])["eligible_probes"]
    assert "P_reward" not in {row["probe"] for row in shown}
    review = S.assess(content(probe="P_reward"), value)
    assert review["codes"] == ["S_PROBE_UNINFORMATIVE"]
    selected = S.select(review, value)
    assert selected["selected_by"] == "registered_rule" and selected["probe"] == "P_value"
    assert selected["host_decision"]["expected_outcomes"] == {
        key: S.R.public_catalog()["P_value"]["expect"][key] for key in S.PUBLIC_CATEGORIES[:3]}


def test_semantic_helper_rejects_already_observed_probe():
    value = packet()
    decision = {"action": "continue", "probe": "P_value", "fact_ids": ["telemetry:error"]}
    result = S.semantic_review(decision, value, observed={"P_value": "normal"})
    assert result["status"] == "invalid" and result["codes"] == ["S_PROBE_OBSERVED"]


def test_valid_model_selection_keeps_host_values_outside_model_decision():
    value = packet()
    review = S.assess(content(probe="P_value"), value)
    selected = S.select(review, value)
    assert selected["selected_by"] == "model" and selected["probe"] == "P_value"
    assert selected["host_decision"]["claims"] == [{"id": "telemetry:error", "value": 1.2}]
    assert "expected_outcomes" not in review["decision"]
    assert "args" not in review["decision"]


def test_valid_model_hold_does_not_switch_to_registered_rule():
    value = packet()
    review = S.assess(content(action="hold"), value)
    selected = S.select(review, value)
    assert selected["selected_by"] == "hold" and selected["reason"] == "model_hold"
    assert selected["probe"] is None and selected["host_decision"] is None


@pytest.mark.parametrize("status", ["infra_error", "unknown"])
def test_infrastructure_or_unknown_review_holds_without_fallback(status):
    selected = S.select({"status": status}, packet())
    assert selected["selected_by"] == "hold" and selected["reason"] == "infra_or_unknown"


def test_invalid_without_format_or_semantic_proof_holds_and_valid_tampering_raises():
    assert S.select({"status": "invalid", "codes": []}, packet())["reason"] == "unusable_review"
    valid = S.assess(content(), packet())
    valid["decision"]["probe"] = "P_missing"
    with pytest.raises(ValueError, match="재검증"):
        S.select(valid, packet())


def test_empty_budget_holds_even_when_model_selection_is_valid():
    value = packet(budget=0)
    selected = S.select(S.assess(content(), value), value)
    assert selected["selected_by"] == "hold" and selected["reason"] == "empty_budget"


def test_single_candidate_allows_only_registered_abnormal_confirmation():
    value = packet()
    decision = {"action": "continue", "probe": S.C1.NEW_PROBE, "fact_ids": ["telemetry:error"]}
    allowed = S.semantic_review(decision, value, hypotheses=[S.C1.COUNT])
    rejected = S.semantic_review({**decision, "probe": "P_value"}, value,
                                 hypotheses=[S.C1.COUNT])
    assert allowed["status"] == "valid"
    assert rejected["status"] == "invalid" and rejected["codes"] == ["S_PROBE_UNINFORMATIVE"]


def test_bad_receipt_and_candidate_tampering_are_packet_errors_not_fallbacks():
    with pytest.raises(ValueError, match="receipt"):
        packet(2, [S.C1.REWARD], prior={"verdict": "not_recovered"})
    value = packet()
    value["remaining_candidates"] = ["reward"]
    with pytest.raises(ValueError, match="남은 후보"):
        S.select({"status": "invalid"}, value)


def test_host_decision_initial_state_stays_compatible_with_v1_and_removes_reward():
    value = packet(2, [S.C1.REWARD])
    selected = S.select(S.assess(content(probe="P_value"), value), value)
    state = S.initial_state(selected["host_decision"], [S.C1.REWARD])
    assert state["next_probe"] == "P_value" and "reward" not in state["hypotheses"]
    assert set(state["hypotheses"]) == set(value["remaining_candidates"])
