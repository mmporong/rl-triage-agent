from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))
import correction_model_call as CM  # noqa: E402
import residual_contract_v1 as C  # noqa: E402


PUBLIC = list(C.L.MECHANISMS)


def packet(epoch=1, applied=None, *, prior: object = "auto"):
    applied = [] if applied is None else applied
    if prior == "auto":
        prior = None if epoch == 1 else {
            "verdict": "not_recovered", "correction_id": applied[0],
            "result_sha256": "a" * 64, "checkpoint_sha256": "b" * 64,
        }
    return {"schema": C.SCHEMA, "facts": {"telemetry:error": 1.2}, "observed": [],
            "budget_left": C.MAX_PROBES, "execution_budget_s": 1200,
            "epoch": epoch, "applied_corrections": applied, "prior_receipt": prior}


def decision(probe, first):
    ranking = [first, *[name for name in PUBLIC if name != first]]
    catalog = C.R.public_catalog()[probe]
    return {"ranking": ranking, "action": "continue", "probe": probe,
            "reason": "등록 검사를 요청한다",
            "claims": [{"id": "telemetry:error", "value": 1.2}],
            "expected_outcomes": {name: catalog["expect"][name] for name in ranking[:3]}}


def confirmed(probe, first, applied):
    state = C.initial_state(decision(probe, first), applied)
    return C.advance(state, probe, "abnormal")


def test_reward_then_update_count_reopens_only_unresolved_components():
    reward = confirmed("P_reward", "reward", [])
    assert C.confirmed_component(reward, []) == "reward"
    assert reward["hypotheses"] == ["reward"]

    second = C.initial_state(decision(C.NEW_PROBE, "optimizer"), [C.REWARD])
    assert second["observed"] == {} and second["history"] == []
    assert "reward" not in second["hypotheses"]
    assert {"actuator", "exploration", "optimizer_value", "physics", "termination", C.COUNT} <= set(
        second["hypotheses"])
    second = C.advance(second, C.NEW_PROBE, "abnormal")
    assert C.confirmed_component(second, [C.REWARD]) == C.COUNT


def test_update_count_then_reward_keeps_global_unresolved_after_local_elimination():
    count = confirmed(C.NEW_PROBE, "optimizer", [])
    assert C.confirmed_component(count, []) == C.COUNT
    assert count["hypotheses"] == [C.COUNT]

    second = C.initial_state(decision("P_reward", "reward"), [C.EPOCHS])
    assert C.COUNT not in second["hypotheses"]
    assert {"reward", "actuator", "exploration", "optimizer_value", "physics", "termination"} <= set(
        second["hypotheses"])
    second = C.advance(second, "P_reward", "abnormal")
    assert C.confirmed_component(second, [C.EPOCHS]) == "reward"


def test_applied_corrections_are_distinct_registered_and_at_most_two():
    assert C.validate_applied([C.REWARD, C.EPOCHS]) == [C.REWARD, C.EPOCHS]
    assert C.unresolved_components([C.REWARD, C.EPOCHS]) == [
        name for name in C.registry()["mechanisms"] if name not in {"reward", C.COUNT}]
    for bad in ([C.REWARD, C.REWARD], [C.REWARD, C.EPOCHS, C.REWARD], ["unregistered"]):
        with pytest.raises(ValueError, match="최대 두 개"):
            C.validate_applied(bad)


def test_unknown_receipt_is_hold_and_cannot_create_correction_packet():
    state = C.initial_state(decision("P_reward", "reward"), [])
    state = C.advance(state, "P_reward", "unknown")
    assert state["status"] == "unidentifiable" and C.confirmed_component(state, []) is None
    with pytest.raises(ValueError, match="HOLD"):
        C.correction_packet(state, {"probe": "P_reward", "outcome": "unknown", "measurement": {},
                                    "request_id": "r"},
                            {"protocol": "fixed_eval_v1", "task": "Go2", "metrics": {}, "standstill": {}},
                            "a" * 64, "b" * 64, [])


def test_diagnosis_requires_epoch_receipt_and_redacts_unregistered_packet_fields():
    first = packet()
    first["hidden_case"] = "must-not-reach-model"
    messages = C.diagnosis_messages(first)
    text = json.dumps(messages, ensure_ascii=False)
    assert "must-not-reach-model" not in text
    body = json.loads(messages[-1]["content"])
    assert body["epoch"] == 1 and body["applied_corrections"] == [] and body["prior_receipt"] is None

    second = packet(2, [C.REWARD])
    body = json.loads(C.diagnosis_messages(second)[-1]["content"])
    assert body["prior_receipt"] == second["prior_receipt"] and "reward" not in body["unresolved_components"]
    bad_rows = [
        {**first, "schema": "other"},
        packet(1, [], prior={"verdict": "not_recovered"}),
        packet(2, [C.REWARD], prior=None),
        packet(2, [], prior=None),
    ]
    for bad in bad_rows:
        with pytest.raises(ValueError):
            C.diagnosis_messages(bad)


@pytest.mark.parametrize(("probe", "first", "applied", "expected"), [
    ("P_reward", "reward", [], C.REWARD),
    (C.NEW_PROBE, "optimizer", [], C.EPOCHS),
])
def test_correction_packet_is_accepted_by_existing_model_contract(probe, first, applied, expected):
    state = confirmed(probe, first, applied)
    receipt = {"probe": probe, "outcome": "abnormal", "measurement": {"value": 1},
               "request_id": "request"}
    report = {"protocol": "fixed_eval_v1", "task": "Go2", "metrics": {}, "standstill": {}}
    built = C.correction_packet(state, receipt, report, "a" * 64, "b" * 64, applied)
    proposal = {"correction_id": expected, "recovery_mode": "restart", "hold_reason": None,
                "cited_facts": ["confirmed_subtype"],
                "checklist": {key: "rule_mode" for key in built["checklist"]}}
    verdict = CM.validate_model_proposal(proposal, built, built["registry"])
    assert verdict["accepted"] is True and verdict["decision"] == "propose"
    assert verdict["agrees_with_rule"] is True


def test_second_correction_closes_scope_and_duplicate_confirmation_is_hold():
    both = [C.REWARD, C.EPOCHS]
    assert C.unresolved_components(both)
    reward_state = confirmed("P_reward", "reward", [])
    assert C.confirmed_component(reward_state, [C.REWARD]) is None
    with pytest.raises(ValueError, match="HOLD"):
        C.correction_packet(reward_state,
                            {"probe": "P_reward", "outcome": "abnormal", "measurement": {},
                             "request_id": "request"},
                            {"protocol": "fixed_eval_v1", "task": "Go2", "metrics": {}, "standstill": {}},
                            "a" * 64, "b" * 64, both)
