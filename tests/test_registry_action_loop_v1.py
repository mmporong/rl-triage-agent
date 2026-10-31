# pyright: reportMissingImports=false
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
import test_evidence_guided_selector_v1 as FIXTURE

from evals import registry_action_loop_v1 as L

selector = FIXTURE.selector


def _provider(content: str) -> dict:
    return {"provider_model": L.O.CALL.MODEL, "finish_reason": "stop", "content": content}


def _recorded_pair(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def execute(state: dict, action: dict, out: Path) -> dict:
        spec = L.SEL.request_spec(state, action)
        paths = {}
        plans = {}
        for role in ("case", "reference"):
            folder = out / role
            plan = folder / "plan/plan.json"
            receipt = folder / "receipt.json"
            plan.parent.mkdir(parents=True)
            plan.write_text(role + "-plan", encoding="utf-8")
            receipt.write_text(role + "-receipt", encoding="utf-8")
            paths[f"{role}_plan"], paths[f"{role}_receipt"] = str(plan), str(receipt)
            plans[role] = L.ref(plan)
        approval = {
            "schema": L.HOST.APPROVAL_SCHEMA, "actor": "fixture",
            "selector_contract": "new_evidence_guided_selector_not_legacy_model_dialogue",
            "state_digest": state["state_digest"], "action": deepcopy(action),
            "request_spec": spec, "plans": plans, "approval_granted": True,
        }
        (out / "root-approval.json").write_text(json.dumps(approval), encoding="utf-8")
        return {**paths, "parent_wall_s": 0.25, "owned_child_wall_s": 0.0}

    def checked(plan: Path, _receipt: Path) -> dict:
        role = "reference" if "reference" in str(plan) else "case"
        state = execute.current_state
        action = execute.current_action
        spec = L.SEL.request_spec(state, action)
        return {"role": role, "probe": L.SEL.NEW_PROBE,
                "request_id": spec["request_ids"][role],
                "source_binding": {
                    "checkpoint_sha256": ("b" if role == "reference" else "a") * 64,
                    "runtime_binding_digest": ("d" if role == "reference" else "c") * 64},
                "outcome": "normal" if role == "reference" else "abnormal"}

    def wrapped(state: dict, action: dict, out: Path) -> dict:
        execute.current_state, execute.current_action = state, action
        return execute(state, action, out)

    execute.current_state = {}
    execute.current_action = {}
    monkeypatch.setattr(L.SEL.RECORDED, "checked_receipt", checked)
    return wrapped


def test_read_then_recorded_probe_confirms_with_actual_selector_proof(
    selector, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, _ledger, _checked = selector
    calls = []

    def factory(_slot: str, current: dict):
        def complete(_messages, _params):
            calls.append(current["state_digest"])
            context = L.O.context(current)
            if not current["evidence_read"]:
                return _provider('{"option_id":"read_optimizer_counts","fact_ids":[]}')
            ids = context["count_evidence_groups"][0]
            return _provider(json.dumps({"option_id": f"probe:{L.SEL.NEW_PROBE}",
                                         "fact_ids": ids}))
        return complete

    result = L.run_epoch(state, run_id="option_loop_confirm", out=tmp_path / "run",
                         verify_context=lambda: True, complete_factory=factory,
                         execute_pair=_recorded_pair(tmp_path, monkeypatch))
    assert result["decision_outcome"] == "confirmed"
    assert result["confirmed_family"] == "optimizer_update_count"
    assert result["model_actions"] == 2 and result["paired_probes"] == 1
    assert result["steps"][0]["selected_option_id"] == "read_optimizer_counts"
    assert result["steps"][1]["selected_option_id"] == f"probe:{L.SEL.NEW_PROBE}"
    final = json.loads((tmp_path / "run/final-state.json").read_text(encoding="utf-8"))
    assert L.SEL.context(final)["status"] == "confirmed"


def test_invalid_response_feedback_holds_without_probe(selector, tmp_path: Path) -> None:
    state, _ledger, _checked = selector
    calls = []

    def factory(_slot, _state):
        def complete(*_args):
            calls.append(True)
            return _provider('{"action":"run_probe","target":"P_reward","fact_ids":[]}')
        return complete

    result = L.run_epoch(state, run_id="option_loop_invalid", out=tmp_path / "invalid",
                         verify_context=lambda: None, complete_factory=factory,
                         execute_pair=lambda *_args: pytest.fail("probe must not run"))
    assert len(calls) == 2
    assert result["decision_outcome"] == "HOLD"
    assert result["stop_reason"] == "model_invalid_hold"
    assert result["paired_probes"] == 0 and result["model_actions"] == 1


def test_provider_failure_has_no_feedback_or_probe(selector, tmp_path: Path) -> None:
    state, _ledger, _checked = selector
    calls = []

    def factory(_slot, _state):
        def complete(*_args):
            calls.append(True)
            return {"content": None, "error_type": "TimeoutError"}
        return complete

    result = L.run_epoch(state, run_id="option_loop_provider_hold", out=tmp_path / "provider",
                         verify_context=lambda: None, complete_factory=factory,
                         execute_pair=lambda *_args: pytest.fail("probe must not run"))
    assert len(calls) == 1 and result["paired_probes"] == 0
    assert result["decision_outcome"] == "HOLD"


def test_receipt_tamper_and_duplicate_budget_are_rejected(
    selector, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, _ledger, _checked = selector
    state = L.SEL.advance(state, L.SEL.select_strong_rule(state))
    context = L.O.context(state)
    ids = context["count_evidence_groups"][0]

    def factory(_slot, _state):
        return lambda *_args: _provider(json.dumps({
            "option_id": f"probe:{L.SEL.NEW_PROBE}", "fact_ids": ids}))

    result = L.run_epoch(state, run_id="option_loop_tamper", out=tmp_path / "tamper",
                         verify_context=lambda: None, complete_factory=factory,
                         execute_pair=_recorded_pair(tmp_path, monkeypatch))
    receipt = tmp_path / "tamper/step-01/probe/case/receipt.json"
    receipt.write_text("tampered", encoding="utf-8")
    final = json.loads((tmp_path / "tamper/final-state.json").read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="proof bytes"):
        L.SEL.context(final)
    assert result["model_actions"] <= L.MAX_MODEL_ACTIONS

    over = deepcopy(state)
    over["budget_left"] = 4
    over = L.SEL._seal(over)
    with pytest.raises(ValueError, match="shape|상한"):
        L.run_epoch(over, run_id="option_loop_over", out=tmp_path / "over",
                    verify_context=lambda: None, complete_factory=factory,
                    execute_pair=lambda *_args: {})


@pytest.mark.parametrize("recorded", [False, True])
def test_pair_proof_accepts_both_registered_host_layouts(
    selector, tmp_path: Path, recorded: bool,
) -> None:
    state, _ledger, _checked = selector
    if recorded:
        state = L.SEL.advance(state, L.SEL.select_strong_rule(state))
        context = L.O.context(state)
        action = {"action": "run_probe", "target": L.SEL.NEW_PROBE,
                  "fact_ids": context["count_evidence_groups"][0]}
    else:
        action = {"action": "run_probe", "target": L.SEL._eligible(state)[0],
                  "fact_ids": ["telemetry:case:mean_reward"]}
    out = tmp_path / ("recorded" if recorded else "legacy")
    paths = {}
    plans = {}
    for role in ("case", "reference"):
        plan = out / role / ("plan/plan.json" if recorded else "plan.json")
        receipt = out / role / "receipt.json"
        plan.parent.mkdir(parents=True, exist_ok=True)
        plan.write_text(role + "-plan", encoding="utf-8")
        receipt.write_text(role + "-receipt", encoding="utf-8")
        paths[f"{role}_plan"], paths[f"{role}_receipt"] = str(plan), str(receipt)
        plans[role] = L.ref(plan)
    approval = {"schema": L.HOST.APPROVAL_SCHEMA, "actor": "fixture",
                "state_digest": state["state_digest"], "action": action,
                "request_spec": L.SEL.request_spec(state, action), "plans": plans,
                "approval_granted": True}
    (out / "root-approval.json").write_text(json.dumps(approval), encoding="utf-8")
    proof = L._pair_proof(paths, state, action, out)
    assert proof["root_approval"] == L.ref(out / "root-approval.json")


def test_pair_proof_rejects_reference_outside_explicit_probe_out(selector, tmp_path: Path) -> None:
    state, _ledger, _checked = selector
    action = {"action": "run_probe", "target": L.SEL._eligible(state)[0],
              "fact_ids": ["telemetry:case:mean_reward"]}
    out, outside = tmp_path / "probe", tmp_path / "outside.json"
    outside.write_text("outside", encoding="utf-8")
    paths, plans = {}, {}
    for role in ("case", "reference"):
        plan = out / role / "plan.json"
        receipt = out / role / "receipt.json"
        plan.parent.mkdir(parents=True, exist_ok=True)
        plan.write_text(role + "-plan", encoding="utf-8")
        receipt.write_text(role + "-receipt", encoding="utf-8")
        paths[f"{role}_plan"], paths[f"{role}_receipt"] = str(plan), str(receipt)
        plans[role] = L.ref(plan)
    paths["reference_receipt"] = str(outside)
    approval = {"schema": L.HOST.APPROVAL_SCHEMA, "actor": "fixture",
                "state_digest": state["state_digest"], "action": action,
                "request_spec": L.SEL.request_spec(state, action), "plans": plans,
                "approval_granted": True}
    (out / "root-approval.json").write_text(json.dumps(approval), encoding="utf-8")
    with pytest.raises(ValueError, match="이탈"):
        L._pair_proof(paths, state, action, out)
