# pyright: reportMissingImports=false
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from evals import evidence_guided_selector_v1 as E


def _facts(case_observed: int = 4, reference_observed: int = 20) -> dict[str, object]:
    value: dict[str, object] = {}
    for role, observed in (("case", case_observed), ("reference", reference_observed)):
        prefix = f"runtime:{role}:"
        value.update({
            prefix + "available": True, prefix + "updates_observed": 3,
            prefix + "expected_optimizer_steps": 20,
            prefix + "optimizer_steps_min": observed,
            prefix + "optimizer_steps_max": observed,
            prefix + "expected_minibatches": 20,
            prefix + "minibatches_min": observed,
            prefix + "minibatches_max": observed,
            f"derived:{role}:step_count_matches_config": observed == 20,
            f"derived:{role}:minibatch_count_matches_config": observed == 20,
        })
    return value


@pytest.fixture
def selector(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[dict, Path, dict]:
    ledger = tmp_path / "facts.json"
    ledger.write_text("facts", encoding="utf-8")
    checked = {"source_binding": {
                   "case": {"checkpoint_sha256": "a" * 64, "binding_digest": "c" * 64},
                   "reference": {"checkpoint_sha256": "b" * 64, "binding_digest": "d" * 64}},
               "model_view": {"facts": _facts()}}
    monkeypatch.setattr(E.LEDGER, "checked", lambda path: deepcopy(checked))
    packet = {"facts": {"telemetry:case:mean_reward": 1.0,
                         "runtime:case:optimizer_steps_min": 999},
              "remaining_candidates": list(E.S.C1.registry()["mechanisms"]),
              "observed": {}, "budget_left": 3,
              "input_bindings": {
                  "case": {"checkpoint_sha256": "a" * 64},
                  "reference": {"checkpoint_sha256": "b" * 64}}}
    monkeypatch.setattr(E.S, "validate_packet", lambda value: deepcopy(value))
    state = E.initialize(packet, ledger)
    return state, ledger, checked


def test_context_hides_private_source_and_exposes_equal_catalog(selector: tuple[dict, Path, dict]) -> None:
    state, _ledger, _checked = selector
    context = E.context(state)
    encoded = json.dumps(context)
    assert E.RESOURCE in encoded
    assert "runtime:case:optimizer_steps_min" not in context["evidence_facts"]
    assert "path" not in encoded.lower()
    assert "fault" not in encoded.lower()
    assert [row["target"] for row in context["probe_catalog"]][-1] == E.NEW_PROBE
    assert len(context["probe_catalog"]) == 8
    assert context["tie_break"] == "equal registered choices use ascending registry_order"
    assert context["action_contract"]["optimizer_triplet"] == [
        "expected_optimizer_steps", "optimizer_steps_min", "optimizer_steps_max"]
    assert "expected plus observed min and max" in context["action_contract"][
        "P_recorded_optimizer_updates"]
    assert "run_probe action requires at least one fact ID" in E.SYSTEM
    assert "read_evidence action requires fact_ids=[]" in E.SYSTEM


def test_read_observe_replan_and_strong_rule_prioritizes_recorded_probe(
    selector: tuple[dict, Path, dict]
) -> None:
    state, _ledger, _checked = selector
    first = E.select_strong_rule(state)
    assert first == {"action": "read_evidence", "target": E.RESOURCE, "fact_ids": []}
    state = E.advance(state, first)
    assert state["evidence_read"] is True
    assert state["visible_facts"]["runtime:case:optimizer_steps_min"] == 4
    second = E.select_strong_rule(state)
    assert second["target"] == E.NEW_PROBE
    assert len(second["fact_ids"]) == 12
    spec = E.request_spec(state, second)
    assert spec["host_args"]["params"] == E.RECORDED.PARAMS
    assert spec["approval_granted"] is False
    assert spec["model_generated_args"] is False
    with pytest.raises(ValueError, match="semantic"):
        E.advance(state, first)


def test_model_assessor_rejects_unread_null_partial_duplicate_and_unknown(
    selector: tuple[dict, Path, dict]
) -> None:
    state, _ledger, checked = selector
    unread = json.dumps({"action": "run_probe", "target": E.NEW_PROBE,
                         "fact_ids": ["runtime:case:expected_optimizer_steps"]})
    assert "S_FACT_UNREAD" in E.assess(unread, state)["codes"]
    checked["model_view"]["facts"]["runtime:reference:optimizer_steps_min"] = None
    state = E.advance(state, E.select_strong_rule(state))
    partial = json.dumps({"action": "run_probe", "target": E.NEW_PROBE,
                          "fact_ids": ["runtime:case:expected_optimizer_steps",
                                       "runtime:case:optimizer_steps_min"]})
    assert "S_COUNT_FACT_PAIR" in E.assess(partial, state)["codes"]
    unknown = json.dumps({"action": "run_probe", "target": "P_missing",
                          "fact_ids": ["telemetry:case:mean_reward"]})
    assert "S_PROBE_INELIGIBLE" in E.assess(unknown, state)["codes"]
    null_action = json.dumps({"action": "run_probe", "target": E.NEW_PROBE,
                              "fact_ids": E._count_ids()})
    assert "S_FACT_NULL" in E.assess(null_action, state)["codes"]
    duplicate = json.dumps({"action": "read_evidence", "target": E.RESOURCE, "fact_ids": []})
    assert "S_EVIDENCE_DUPLICATE" in E.assess(duplicate, state)["codes"]


def test_valid_model_action_and_invalid_model_hold(selector: tuple[dict, Path, dict]) -> None:
    state, _ledger, _checked = selector
    state = E.advance(state, E.select_strong_rule(state))
    action = {"action": "run_probe", "target": E.NEW_PROBE, "fact_ids": E._count_ids()}
    review = E.assess(json.dumps(action), state)
    assert review["status"] == "valid"
    assert E.select_model(review, state) == action
    assert E.select_model({"status": "invalid"}, state) == {
        "action": "hold", "target": None, "fact_ids": []}


def test_recorded_receipts_drive_state_and_replay_proof(
    selector: tuple[dict, Path, dict], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state, _ledger, _checked = selector
    state = E.advance(state, E.select_strong_rule(state))
    action = E.select_strong_rule(state)
    request_spec = E.request_spec(state, action)
    paths = []
    for name in ("case-plan", "case-receipt", "reference-plan", "reference-receipt"):
        path = tmp_path / f"{name}.json"
        path.write_text(name, encoding="utf-8")
        paths.append(path)

    def checked_receipt(plan: Path, receipt: Path) -> dict:
        role = "reference" if "reference" in plan.name else "case"
        return {"role": role, "probe": E.NEW_PROBE,
                "request_id": request_spec["request_ids"][role],
                "source_binding": {
                    "checkpoint_sha256": ("b" if role == "reference" else "a") * 64,
                    "runtime_binding_digest": ("d" if role == "reference" else "c") * 64},
                "outcome": "normal" if role == "reference" else "abnormal"}

    monkeypatch.setattr(E.RECORDED, "checked_receipt", checked_receipt)
    state = E.advance(state, action, case_plan=paths[0], case_receipt=paths[1],
                      reference_plan=paths[2], reference_receipt=paths[3])
    assert state["status"] == "confirmed"
    assert state["hypotheses"] == ["optimizer_update_count"]
    assert state["history"][-1]["outcome"] == "abnormal"
    assert E.context(state)["eligible_probes"] == []
    paths[1].write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="proof bytes"):
        E.context(state)


def test_state_digest_rejects_self_tamper_and_rule_falls_back_to_registry(
    selector: tuple[dict, Path, dict]
) -> None:
    state, _ledger, checked = selector
    tampered = deepcopy(state)
    tampered["budget_left"] = 0
    with pytest.raises(ValueError, match="digest"):
        E.context(tampered)
    resealed = deepcopy(state)
    resealed["hypotheses"] = ["reward"]
    resealed = E._seal(resealed)
    with pytest.raises(ValueError, match="재생"):
        E.context(resealed)
    checked["model_view"]["facts"]["derived:case:step_count_matches_config"] = True
    checked["model_view"]["facts"]["derived:case:minibatch_count_matches_config"] = True
    state = E.advance(state, E.select_strong_rule(state))
    action = E.select_strong_rule(state)
    assert action["action"] == "run_probe"
    assert action["target"] in E.S.C1.registry()["probes"]


def test_initialize_and_receipt_reject_wrong_source_checkpoint(
    selector: tuple[dict, Path, dict], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state, ledger, checked = selector
    packet = deepcopy(state["initial_packet"])
    packet["input_bindings"]["case"]["checkpoint_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="case packet checkpoint"):
        E.initialize(packet, ledger)

    state = E.advance(state, E.select_strong_rule(state))
    action = E.select_strong_rule(state)
    paths = []
    for name in ("case-plan", "case-receipt", "reference-plan", "reference-receipt"):
        path = tmp_path / f"wrong-{name}.json"
        path.write_text(name, encoding="utf-8")
        paths.append(path)

    def wrong_receipt(plan: Path, receipt: Path) -> dict:
        role = "reference" if "reference" in plan.name else "case"
        return {"role": role, "probe": E.NEW_PROBE, "request_id": f"wrong-{role}",
                "source_binding": {"checkpoint_sha256": "e" * 64,
                                   "runtime_binding_digest": checked["source_binding"][role][
                                       "binding_digest"]},
                "outcome": "normal"}

    monkeypatch.setattr(E.RECORDED, "checked_receipt", wrong_receipt)
    with pytest.raises(ValueError, match="checkpoint/runtime binding"):
        E.advance(state, action, case_plan=paths[0], case_receipt=paths[1],
                  reference_plan=paths[2], reference_receipt=paths[3])


def _legacy_pair(
    selector: tuple[dict, Path, dict], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    *, wrong_runtime: bool = False, wrong_collected: bool = False, wrong_scope: bool = False,
) -> tuple[dict, dict]:
    state, _ledger, checked = selector
    target = E._eligible(state)[0]
    action = {"action": "run_probe", "target": target, "fact_ids": []}
    spec = E.request_spec(state, action)
    root = tmp_path / "legacy"
    authorization = root / "authorization.json"
    authorization.parent.mkdir(parents=True)
    authorization.write_text("authorization", encoding="utf-8")
    shared = {"path": str(authorization.resolve()), "sha256": E.RECORDED.sha(authorization),
              "request_id": "cf1f420105e6"}
    plans: dict[str, Path] = {}
    receipts: dict[str, Path] = {}
    sources: dict[str, dict] = {}
    private_refs = {}
    for role in ("case", "reference"):
        folder = root / role
        collected = folder / "collected.json"
        runtime = folder / "runtime.json"
        collected.parent.mkdir(parents=True)
        collected.write_text(role + "-collected", encoding="utf-8")
        runtime.write_text(role + "-runtime", encoding="utf-8")
        plan = folder / "plan.json"
        plan.write_text(json.dumps({"collected_input": {
            "path": str(collected.resolve()), "sha256": E.RECORDED.sha(collected)}}), encoding="utf-8")
        receipt = folder / "probe_receipt.json"
        receipt.write_text(json.dumps({"authorization": shared}), encoding="utf-8")
        plans[role], receipts[role] = plan, receipt
        private_refs[role] = {"path": str(runtime.resolve()), "sha256": E.RECORDED.sha(runtime)}
        sources[role] = {
            "checkpoint_sha256": ("a" if role == "case" else "b") * 64,
            "runtime_binding_digest": (("9" if wrong_runtime and role == "case" else
                                        "c" if role == "case" else "d") * 64),
            "collected_binding_digest": (("8" if wrong_collected and role == "case" else
                                          "1" if role == "case" else "2") * 64),
            "telemetry_sha256": ("e" if role == "case" else "f") * 64,
        }
    checked["private_source_refs"] = private_refs

    def verify(plan: Path, _receipt: Path) -> dict:
        role = "reference" if "reference" in str(plan) else "case"
        return {"role": role, "probe": target, "status": "complete",
                "checkpoint_sha256": ("a" if role == "case" else "b") * 64,
                "binding_digest": ("1" if role == "case" else "2") * 64,
                "telemetry_sha256": ("e" if role == "case" else "f") * 64,
                "measurement": {"probe_value": 1.0}}

    monkeypatch.setattr(E.OBS, "verify_receipt", verify)
    monkeypatch.setattr(E.RECORDED, "_source", lambda collected, runtime, role: deepcopy(sources[role]))
    monkeypatch.setattr(E.OBS, "classify_raw", lambda *_args: "abnormal")
    approval = {
        "schema": "evidence_guided_probe_host_approval_v1", "actor": "fixture",
        "selector_contract": "new_evidence_guided_selector_not_legacy_model_dialogue",
        "state_digest": state["state_digest"], "action": deepcopy(action),
        "request_spec": deepcopy(spec),
        "plans": {role: {"path": str(plans[role].resolve()),
                          "sha256": E.RECORDED.sha(plans[role])}
                  for role in ("case", "reference")},
        "approval_granted": True,
    }
    if wrong_scope:
        approval["request_spec"]["request_scope_digest"] = "0" * 64
    (root / "root-approval.json").write_text(json.dumps(approval), encoding="utf-8")
    result = E.advance(state, action, case_plan=plans["case"], case_receipt=receipts["case"],
                       reference_plan=plans["reference"],
                       reference_receipt=receipts["reference"])
    return result, action


def test_legacy_pair_accepts_shared_authorization_and_replays_root_approval(
    selector: tuple[dict, Path, dict], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state, _action = _legacy_pair(selector, tmp_path, monkeypatch)
    row = state["history"][-1]
    assert row["request_ids"]["case"] == row["request_ids"]["reference"] == "cf1f420105e6"
    assert "root_approval" in row["proof"]
    assert E.context(state)["status"] == state["status"]


@pytest.mark.parametrize("wrong", ["runtime", "collected"])
def test_legacy_pair_rejects_wrong_typed_binding(
    selector: tuple[dict, Path, dict], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    wrong: str,
) -> None:
    with pytest.raises(ValueError, match="typed binding"):
        _legacy_pair(selector, tmp_path, monkeypatch,
                     wrong_runtime=wrong == "runtime", wrong_collected=wrong == "collected")


def test_legacy_pair_rejects_wrong_root_approval_scope(
    selector: tuple[dict, Path, dict], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ValueError, match="request scope"):
        _legacy_pair(selector, tmp_path, monkeypatch, wrong_scope=True)
