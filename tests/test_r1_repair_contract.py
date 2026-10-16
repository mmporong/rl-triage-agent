from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evals"))
import r1_repair_contract as C
import r1_repair_model_call as F


def packet():
    return {"schema": "r1_repair_diagnosis_packet_v1", "facts": {"telemetry:case:error": 1.2},
            "observed": [], "budget_left": 7, "execution_budget_s": 1200}


def decision(probe="P_value"):
    ranking = ["reward", "actuator", "exploration", "optimizer", "physics", "termination"]
    return {"ranking": ranking, "action": "continue", "probe": probe, "reason": "등록 검사를 요청한다",
            "claims": [{"id": "telemetry:case:error", "value": 1.2}],
            "expected_outcomes": {h: C.public_catalog()[probe]["expect"][h] for h in ranking[:3]}}


def response(content):
    return {"content": content, "provider_model": C.MODEL, "finish_reason": "stop", "wall_s": 0.2,
            "input_tokens": 10, "output_tokens": 5, "total_tokens": 15}


def test_normal_p_value_keeps_update_count_even_when_optimizer_ranked_fourth():
    state = C.advance(C.initial_state(decision()), "P_value", "normal")
    assert C.VALUE not in state["hypotheses"]
    assert C.COUNT in state["hypotheses"]
    while state["status"] == "open":
        probe = state["next_probe"]
        state = C.advance(state, probe, "abnormal" if probe == C.NEW_PROBE else "normal")
    assert state["status"] == "confirmed"
    assert C.correction_allowed(state)
    assert len(state["observed"]) <= 7


def test_legacy_registry_expectations_and_measurement_thresholds_unchanged():
    before = deepcopy(C.L.PROBES)
    registry = C.registry()
    for name, spec in before.items():
        row = deepcopy(registry["probes"][name])
        row["expect"]["optimizer"] = row["expect"].pop(C.VALUE)
        row["expect"].pop(C.COUNT)
        assert row == spec
    assert C.L.PROBES == before
    assert C.public_catalog()["P_value"]["expect"]["optimizer"] is None
    assert C.public_catalog()[C.NEW_PROBE]["expect"]["optimizer"] is None


def test_count_normal_does_not_declare_optimizer_healthy():
    state = C.advance(C.initial_state(decision(C.NEW_PROBE)), C.NEW_PROBE, "normal")
    assert C.COUNT not in state["hypotheses"]
    assert C.VALUE in state["hypotheses"]
    assert state["status"] == "open"
    assert not C.correction_allowed(state)


def test_unknown_stops_and_value_confirmation_cannot_restore_epoch():
    state = C.advance(C.initial_state(decision()), "P_value", "unknown")
    assert state["status"] == "unidentifiable"
    assert len(state["hypotheses"]) == 7
    value = C.advance(C.initial_state(decision()), "P_value", "abnormal")
    assert value["status"] == "confirmed"
    assert not C.correction_allowed(value)


def test_new_probe_is_accepted_by_new_contract_without_frozen_mutation():
    value = decision(C.NEW_PROBE)
    assert C.validate_diagnosis(value, packet()) == value
    assert C.NEW_PROBE not in C.L.PROBES
    assert "R1" not in json.dumps(C.diagnosis_messages(packet()))


@pytest.mark.parametrize("change", [
    {"probe": {}}, {"action": "confirm"}, {"claims": [{"id": "invented", "value": 1.2}]},
    {"claims": [{"id": "telemetry:case:error", "value": 1.3}]}, {"ranking": ["optimizer"] * 6},
])
def test_untrusted_decision_is_rejected(change):
    with pytest.raises(ValueError):
        C.validate_diagnosis({**decision(), **change}, packet())


def test_feedback_only_once_and_reviewer_recomputes():
    data = json.dumps(packet()).encode()
    outputs = iter([response('{}'), response(json.dumps(decision(C.NEW_PROBE)))])
    row = F.dialogue("diagnosis", data, lambda *_: next(outputs))
    assert row["requests"] == 2
    assert F.review(row, data, "diagnosis")["status"] == "valid"
    altered = deepcopy(row)
    altered["feedback"]["text"] += "extra hint"
    with pytest.raises(ValueError):
        F.review(altered, data, "diagnosis")
    altered = deepcopy(row)
    altered["calls"][-1]["provider_model"] = "other"
    with pytest.raises(ValueError):
        F.review(altered, data, "diagnosis")


def test_second_invalid_is_terminal_and_infrastructure_does_not_get_semantic_feedback():
    data = json.dumps(packet()).encode()
    row = F.dialogue("diagnosis", data, lambda *_: response('{}'))
    assert row["requests"] == 2
    assert F.review(row, data, "diagnosis")["status"] == "invalid"
    infra = F.dialogue("diagnosis", data, lambda *_: {"infra_error": "timeout"})
    assert infra["requests"] == 1 and infra["feedback"] is None
    assert F.review(infra, data, "diagnosis")["status"] == "infra_error"


def test_correction_contract_requires_count_subtype_and_retains_checklist():
    state = C.advance(C.initial_state(decision(C.NEW_PROBE)), C.NEW_PROBE, "abnormal")
    confirming = {"request_id": "request", "outcome": "abnormal", "measurement": {"steps": [4, 4, 4]}}
    report = {"protocol": "fixed_eval_v1", "task": "Go2", "metrics": {}, "standstill": {}}
    built = C.correction_packet(state, confirming, report, "a" * 64, "b" * 64)
    proposal = {"correction_id": C.CORRECTION, "recovery_mode": "restart", "hold_reason": None,
                "cited_facts": ["confirmed_subtype"], "checklist": {key: "rule_mode" for key in built["checklist"]}}
    data = json.dumps(built).encode()
    row = F.dialogue("correction", data, lambda *_: response(json.dumps(proposal)))
    assert F.review(row, data, "correction")["decision"]["agrees_with_rule"]
    value = C.advance(C.initial_state(decision()), "P_value", "abnormal")
    with pytest.raises(ValueError):
        C.correction_packet(value, confirming, report, "a" * 64, "b" * 64)


def test_call_uses_one_packet_snapshot_for_reservation_and_api(tmp_path, monkeypatch):
    path, output = tmp_path / "packet.json", tmp_path / "output.json"
    original = json.dumps(packet()).encode()
    path.write_bytes(original)
    make_reservation = F.reservation_for
    original_dialogue = F.dialogue
    sent = []
    def swap_after_snapshot(kind, data):
        reservation = make_reservation(kind, data)
        path.write_bytes(b'{"changed":true}')
        return reservation
    def fake_dialogue(kind, data):
        sent.append(data)
        return original_dialogue(kind, data, lambda *_: response(json.dumps(decision())))
    monkeypatch.setattr(F, "reservation_for", swap_after_snapshot)
    monkeypatch.setattr(F, "dialogue", fake_dialogue)
    assert F.call("diagnosis", path, output) == 0
    reservation = json.loads(output.with_suffix(".reserved.json").read_text())
    record = json.loads(output.read_text(encoding="utf-8"))
    assert sent == [original]
    assert F.review(record, original, "diagnosis", reservation)["status"] == "valid"
    with pytest.raises(ValueError):
        F.review(record, path.read_bytes(), "diagnosis", reservation)
    reservation["at"] += "changed"
    with pytest.raises(ValueError):
        F.review(record, original, "diagnosis", reservation)
