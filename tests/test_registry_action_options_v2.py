# pyright: reportMissingImports=false
from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, cast

import pytest
import test_evidence_guided_selector_v1 as FIXTURE

from evals import registry_action_options_v1 as O1
from evals import registry_action_options_v2 as O


@pytest.fixture
def state(tmp_path, monkeypatch):
    value, _path, _ledger = cast(Any, FIXTURE.selector).__wrapped__(tmp_path, monkeypatch)
    return value


def test_context_preserves_options_and_adds_registered_candidate_partitions(state):
    old, new = O1.context(state), O.context(state)
    assert [row["option_id"] for row in new["action_options"]] == [
        row["option_id"] for row in old["action_options"]]
    assert "format_example" not in new
    probes = [row for row in new["action_options"] if row["action"] == "run_probe"]
    assert probes
    for row in probes:
        partition = row["candidate_outcome_partition"]
        assert set(partition) == {"normal", "abnormal", "unspecified"}
        flattened = [candidate for group in partition.values() for candidate in group]
        assert sorted(flattened) == sorted(new["remaining_candidates"])
        assert row["outcome_semantics"]["unknown"] == "confirms_no_candidate"
        assert "paired_reference_contract" in row["outcome_semantics"]
        assert row["outcome_semantics"][
            "unspecified_candidates_remain_after_normal_or_abnormal"] is True
        assert row["outcome_semantics"]["registered_predictions_are_not_observations"] is True
    assert new["contract"]["fault_truth_exposed"] is False
    assert new["contract"]["probability_or_eig"] is False


def test_read_semantics_and_recorded_partition_are_explicit_without_ranking(state):
    before = O.context(state)
    read = next(row for row in before["action_options"]
                if row["option_id"] == "read_optimizer_counts")
    assert read["evidence_semantics"] == {
        "values_hidden_until_read": True, "additional_sdk_launches": 0,
        "condition": "recorded_training_counts_already_exist"}
    state = O.SEL.advance(state, O.SEL.select_strong_rule(state))
    after = O.context(state)
    count = next(row for row in after["action_options"]
                 if row["option_id"] == f"probe:{O.SEL.NEW_PROBE}")
    partition = count["candidate_outcome_partition"]
    assert partition["abnormal"] == ["optimizer_update_count"]
    assert "optimizer_update_count" not in partition["normal"]
    assert count["outcome_semantics"]["paired_reference_contract"] == (
        "reference_outcome_must_be_normal_else_unknown")
    rendered_options = json.dumps(after["action_options"], sort_keys=True)
    assert "recommend" not in rendered_options and "probability" not in rendered_options


def test_assessment_is_bound_to_v2_context_and_reuses_v1_semantics(state):
    response = '{"option_id":"hold","fact_ids":[]}'
    review = O.assess(response, state)
    assert review["schema"] == O.REVIEW_SCHEMA and review["status"] == "valid"
    assert O.select_model(review, state) == {"action": "hold", "target": None, "fact_ids": []}
    changed = deepcopy(review)
    changed["context_digest"] = "0" * 64
    with pytest.raises(ValueError, match="V2 option review"):
        O.select_model(changed, state)


def test_dialogue_feedback_uses_v2_context_and_never_retries_provider_failure(state):
    calls = []

    def complete(messages, _params):
        calls.append(messages)
        content = ('{"action":"hold","target":null,"fact_ids":[]}' if len(calls) == 1
                   else '{"option_id":"hold","fact_ids":[]}')
        return {"provider_model": O.CALL.MODEL, "finish_reason": "stop", "content": content}

    result = O.dialogue(state, complete, verify_context=lambda: None)
    assert len(calls) == 2 and result["schema"] == "registry_action_options_dialogue_v2"
    feedback = json.loads(calls[1][-1]["content"])
    assert any("candidate_outcome_partition" in row for row in feedback["action_options"]
               if row["action"] == "run_probe")

    failed = O.dialogue(state, lambda *_: {"error_type": "TimeoutError"},
                        verify_context=lambda: None)
    assert len(failed["calls"]) == 1 and failed["action"]["action"] == "hold"
