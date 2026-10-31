from __future__ import annotations

import json

import pytest
import test_evidence_guided_selector_v1 as FIXTURE

from evals import prospective_agentic_comparison_v1 as C


def test_policy_rotation_balances_first_policy() -> None:
    orders = [C.policy_order(index) for index in range(3)]
    assert {order[0] for order in orders} == set(C.POLICIES)
    assert all(set(order) == set(C.POLICIES) for order in orders)
    with pytest.raises(ValueError):
        C.policy_order(-1)


def test_dialogue_preserves_invalid_attempt_and_feedback_once(monkeypatch) -> None:
    monkeypatch.setattr(C.SEL, "messages", lambda _s: [{"role": "user", "content": "facts"}])
    monkeypatch.setattr(C.SEL, "assess", lambda content, _s: {
        "status": "valid" if content == "good" else "invalid", "codes": ["S_FACT_UNREAD"]})
    action = {"action": "hold", "target": None, "fact_ids": []}
    monkeypatch.setattr(C.SEL, "select_model", lambda _r, _s: action)
    requests, checks = [], []

    def complete(messages, params):
        requests.append((messages, params))
        return {"provider_model": C.CALL.MODEL, "finish_reason": "stop",
                "content": "bad" if len(requests) == 1 else "good"}

    result = C.dialogue({"state_digest": "a" * 64}, complete,
                        verify_context=lambda: checks.append(True))
    assert len(requests) == 2 and len(checks) == 4
    assert result["calls"][0]["result"]["content"] == "bad"
    assert result["model_valid"] is True and result["semantic_feedback_used"] is True
    assert json.loads(requests[1][0][-1]["content"])["validation_errors"] == ["S_FACT_UNREAD"]
    assert requests[0][1] == requests[1][1] == C.CALL.F.request_params(C.CALL.MODEL)


def test_provider_failure_does_not_trigger_semantic_retry(monkeypatch) -> None:
    monkeypatch.setattr(C.SEL, "messages", lambda _s: [{"role": "user", "content": "facts"}])
    monkeypatch.setattr(C.SEL, "select_model", lambda _r, _s: {
        "action": "hold", "target": None, "fact_ids": []})
    sent = []

    def complete(messages, _params):
        sent.append(messages)
        return {"error_type": "TimeoutError", "content": None}

    result = C.dialogue({"state_digest": "a" * 64}, complete, verify_context=lambda: None)
    assert len(sent) == 1
    assert result["model_valid"] is False
    assert result["final_review"]["codes"] == ["PROVIDER_UNAVAILABLE"]


def test_context_mutation_stops_before_model_request(monkeypatch) -> None:
    monkeypatch.setattr(C.SEL, "messages", lambda _s: [])

    def fail():
        raise ValueError("source changed")

    with pytest.raises(ValueError, match="source changed"):
        C.dialogue({}, lambda *_a: pytest.fail("must not send"), verify_context=fail)


def test_actual_selector_reads_then_holds_without_executing_probe(
    tmp_path, monkeypatch,
) -> None:
    state, _path, ledger = FIXTURE.selector.__wrapped__(tmp_path, monkeypatch)
    packet = state["initial_packet"]
    envelope = {"packet": packet, "source_binding": ledger["source_binding"]}
    monkeypatch.setattr(C.INPUT, "make_packet", lambda *_a, **_k: envelope)
    monkeypatch.setattr(C.INPUT, "checked", lambda _p: ({}, {"runtime_facts": {
        "case": {"path": "case.json"}, "reference": {"path": "reference.json"}}}))
    monkeypatch.setattr(C.FACTS, "build", lambda *_a, **_k: ledger)
    seen = []

    def factory(_slot, current):
        def complete(messages, _params):
            view = json.loads(messages[1]["content"])
            seen.append(view)
            action = ({"action": "read_evidence", "target": C.SEL.RESOURCE, "fact_ids": []}
                      if not current["evidence_read"]
                      else {"action": "hold", "target": None, "fact_ids": []})
            return {"provider_model": C.CALL.MODEL, "finish_reason": "stop",
                    "content": json.dumps(action)}
        return complete

    result = C.run_epoch(tmp_path / "case", tmp_path / "reference",
        policy="evidence_guided_model", run_id="fixture-read-hold", out=tmp_path / "epoch",
        verify_context=lambda: None, complete_factory=factory,
        execute_pair=lambda *_a: pytest.fail("HOLD must not execute a probe"))
    assert "runtime:case:optimizer_steps_min" not in seen[0]["evidence_facts"]
    assert seen[1]["evidence_facts"]["runtime:case:optimizer_steps_min"] == 4
    assert result["decision_outcome"] == "HOLD"
    assert result["request_count"] == 2 and result["probe_count"] == 0
    assert result["claims"]["recovery"] is False
