# pyright: reportMissingImports=false
from __future__ import annotations

import json
from copy import deepcopy

import pytest
import test_evidence_guided_selector_v1 as FIXTURE

from evals import registry_action_options_v1 as O


@pytest.fixture
def state(tmp_path, monkeypatch):
    value, _path, _ledger = FIXTURE.selector.__wrapped__(tmp_path, monkeypatch)
    return value


def test_menu_contains_only_feasible_actions_and_no_private_state(state):
    context = O.context(state)
    options = {row["option_id"] for row in context["action_options"]}
    assert "read_optimizer_counts" in options
    assert f"probe:{O.SEL.NEW_PROBE}" not in options
    assert O.SEL.NEW_PROBE in O.SEL.context(state)["eligible_probes"]
    assert context["count_evidence_groups"] == []
    assert "fact_ledger" not in context and "initial_packet" not in context
    assert "runtime:case:optimizer_steps_min" not in context["evidence_facts"]
    assert context["output_schema"]["properties"]["option_id"]["enum"] == [
        row["option_id"] for row in context["action_options"]]


def test_read_maps_to_existing_action_then_count_requires_exact_groups(state):
    read = O.assess(json.dumps({"option_id": "read_optimizer_counts", "fact_ids": []}), state)
    action = O.select_model(read, state)
    assert action == {"action": "read_evidence", "target": O.SEL.RESOURCE, "fact_ids": []}
    state = O.SEL.advance(state, action)
    context = O.context(state)
    options = {row["option_id"] for row in context["action_options"]}
    assert "read_optimizer_counts" not in options
    assert f"probe:{O.SEL.NEW_PROBE}" in options
    assert len(context["count_evidence_groups"]) == 4
    cited = context["count_evidence_groups"][0]
    review = O.assess(json.dumps({"option_id": f"probe:{O.SEL.NEW_PROBE}", "fact_ids": cited}), state)
    assert review["status"] == "valid"
    action = O.select_model(review, state)
    assert action == {"action": "run_probe", "target": O.SEL.NEW_PROBE, "fact_ids": cited}
    request = O.SEL.request_spec(state, action)
    assert request["approval_granted"] is False
    partial = O.assess(json.dumps({"option_id": f"probe:{O.SEL.NEW_PROBE}",
                                  "fact_ids": cited[:2]}), state)
    assert partial["codes"] == ["S_COUNT_FACT_PAIR"]


@pytest.mark.parametrize("response,code", [
    (('{"action":"P_recorded_optimizer_updates","target":"evidence:optimizer-counts",'
      '"fact_ids":[]}'), "F_OPTION_FIELDS"),
    ('{"option_id":"hold","option_id":"read_optimizer_counts","fact_ids":[]}', "F_OPTION_JSON"),
    ('{"option_id":"hold","fact_ids":[],"arguments":{}}', "F_OPTION_FIELDS"),
    ('{"option_id":null,"fact_ids":[]}', "F_OPTION_ID"),
    ('{"option_id":"hold","fact_ids":[true]}', "F_FACT_IDS"),
    ('{"option_id":"hold","fact_ids":["x","x"]}', "F_FACT_IDS"),
    ('{"option_id":"invented","fact_ids":[]}', "S_OPTION_UNAVAILABLE"),
    ('{"option_id":"read_optimizer_counts","fact_ids":["telemetry:case:mean_reward"]}',
     "S_OPTION_EMPTY_CITATIONS"),
    ('{"option_id":"probe:P_reward","fact_ids":["unknown"]}', "S_FACT_UNREAD"),
    ('{"option_id":"probe:P_reward","fact_ids":[]}', "S_FACT_IDS"),
])
def test_invalid_contract_is_not_reinterpreted(state, response, code):
    review = O.assess(response, state)
    assert review["status"] == "invalid" and code in review["codes"]
    assert O.select_model(review, state) == {"action": "hold", "target": None, "fact_ids": []}


def test_review_is_bound_to_current_state_and_untampered_response(state):
    review = O.assess('{"option_id":"hold","fact_ids":[]}', state)
    changed = deepcopy(review)
    changed["action"]["target"] = O.SEL.NEW_PROBE
    with pytest.raises(ValueError, match="review"):
        O.select_model(changed, state)
    after_read = O.SEL.advance(state, O.SEL.select_strong_rule(state))
    with pytest.raises(ValueError, match="review"):
        O.select_model(review, after_read)


def test_feedback_exposes_enum_preserves_invalid_and_uses_same_provider_params(state):
    sent = []
    checks = []

    def complete(messages, params):
        sent.append((messages, params))
        content = ('{"action":"P_reward","target":"P_reward","fact_ids":[]}'
                   if len(sent) == 1 else '{"option_id":"read_optimizer_counts","fact_ids":[]}')
        return {"provider_model": O.CALL.MODEL, "finish_reason": "stop", "content": content}

    result = O.dialogue(state, complete, verify_context=lambda: checks.append(True))
    assert len(sent) == 2 and len(checks) == 4
    assert result["calls"][0]["review"]["codes"] == ["F_OPTION_FIELDS"]
    assert result["model_valid"] is True
    assert result["action"]["action"] == "read_evidence"
    feedback = json.loads(sent[1][0][-1]["content"])
    assert "read_optimizer_counts" in feedback["output_schema"]["properties"]["option_id"]["enum"]
    assert sent[0][1] == sent[1][1] == O.CALL.F.request_params(O.CALL.MODEL)
    assert result["claims"] == {"diagnosis_improvement": False, "recovery": False}


def test_provider_failure_and_changed_context_do_not_retry_or_execute(state):
    calls = []

    def complete(*_args):
        calls.append(True)
        return {"content": None, "error_type": "TimeoutError"}

    result = O.dialogue(state, complete, verify_context=lambda: None)
    assert len(calls) == 1 and result["model_valid"] is False
    assert result["final_review"]["codes"] == ["PROVIDER_UNAVAILABLE"]

    def fail():
        raise ValueError("source changed")

    with pytest.raises(ValueError, match="source changed"):
        O.dialogue(state, lambda *_args: pytest.fail("no HTTP"), verify_context=fail)
