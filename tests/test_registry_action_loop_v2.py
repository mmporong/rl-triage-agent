# pyright: reportMissingImports=false
from __future__ import annotations

import json
from pathlib import Path

import pytest
import test_registry_action_loop_v1 as V1T

from evals import registry_action_loop_v2 as L

selector = V1T.selector


def _provider(content: str) -> dict:
    return {"provider_model": L.O.CALL.MODEL, "finish_reason": "stop", "content": content}


def test_v2_loop_uses_partition_context_and_hold_has_no_sdk(selector, tmp_path: Path):
    state, _ledger, _checked = selector
    seen = []

    def factory(_slot, _state):
        def complete(messages, _params):
            body = json.loads(messages[1]["content"])
            seen.append(body)
            return _provider('{"option_id":"hold","fact_ids":[]}')
        return complete

    result = L.run_epoch(state, run_id="v2_hold", out=tmp_path / "hold",
                         verify_context=lambda: None, complete_factory=factory,
                         execute_pair=lambda *_: pytest.fail("hold must not execute SDK"))
    assert result["schema"] == L.SCHEMA and result["paired_probes"] == 0
    assert result["decision_outcome"] == "HOLD"
    assert result["selection_subject"].startswith("model_option_id_v2")
    assert any("candidate_outcome_partition" in row for row in seen[0]["action_options"]
               if row["action"] == "run_probe")


def test_v2_loop_requires_approval_receipt_proof_for_probe(selector, tmp_path: Path):
    state, _ledger, _checked = selector
    fact_id = next(iter(L.O.context(state)["evidence_facts"]))
    target = next(row["target"] for row in L.O.context(state)["action_options"]
                  if row["action"] == "run_probe")

    def factory(_slot, _state):
        return lambda *_: _provider(json.dumps({
            "option_id": f"probe:{target}", "fact_ids": [fact_id]}))

    with pytest.raises(ValueError, match="plan/receipt"):
        L.run_epoch(state, run_id="v2_no_proof", out=tmp_path / "no-proof",
                    verify_context=lambda: None, complete_factory=factory,
                    execute_pair=lambda *_: {})


def test_v2_loop_approved_recorded_receipt_confirms(selector, tmp_path: Path, monkeypatch):
    state, _ledger, _checked = selector
    state = L.SEL.advance(state, L.SEL.select_strong_rule(state))
    ids = L.O.context(state)["count_evidence_groups"][0]

    def factory(_slot, _state):
        return lambda *_: _provider(json.dumps({
            "option_id": f"probe:{L.SEL.NEW_PROBE}", "fact_ids": ids}))

    result = L.run_epoch(state, run_id="v2_confirm", out=tmp_path / "confirmed",
                         verify_context=lambda: None, complete_factory=factory,
                         execute_pair=V1T._recorded_pair(tmp_path, monkeypatch))
    assert result["decision_outcome"] == "confirmed"
    assert result["confirmed_family"] == "optimizer_update_count"
    assert result["paired_probes"] == 1


def test_unknown_receipt_stops_without_confirmation(selector, tmp_path: Path, monkeypatch):
    state, _ledger, _checked = selector
    state = L.SEL.advance(state, L.SEL.select_strong_rule(state))
    ids = L.O.context(state)["count_evidence_groups"][0]
    execute = V1T._recorded_pair(tmp_path, monkeypatch)
    original_checked = L.SEL.RECORDED.checked_receipt

    def unknown(plan, receipt):
        value = original_checked(plan, receipt)
        return {**value, "outcome": "unknown"}

    monkeypatch.setattr(L.SEL.RECORDED, "checked_receipt", unknown)

    def factory(_slot, _state):
        return lambda *_: _provider(json.dumps({
            "option_id": f"probe:{L.SEL.NEW_PROBE}", "fact_ids": ids}))

    result = L.run_epoch(state, run_id="v2_unknown", out=tmp_path / "unknown",
                         verify_context=lambda: None, complete_factory=factory,
                         execute_pair=execute)
    assert result["decision_outcome"] == "HOLD"
    assert result["final_status"] == "unidentifiable"
