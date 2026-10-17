"""적용성 상태·원장 경계·관문 제거의 CPU 수용 검사."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / "evals", ROOT / "tests"):
    sys.path.insert(0, str(folder))
import combination_contract_v1 as K
from test_grounded_recovery import _report


def make_case(tmp_path, fault="U1", robot="Go2"):
    task, protocol, scale, protocol_path = K.ROBOTS[robot]
    mechanism, names = K.FAULTS[fault]
    correction = K.registry()[mechanism]["correction_id"]
    reports = {key: _report(key, lin=lin, scale=scale) for key, lin in (("none", .15), ("fault", 1.2))}
    for row in reports.values():
        row.update(task=task, protocol=protocol, protocol_sha256_lf=K.sha(ROOT / protocol_path))
    stages = {
        "applicability": {"targets": {name: {"path": path, "exists": True, "active": True, "effect_reachable": True}
                                       for name, path in K.fault_paths(fault).items()}},
        "injection": {"before": {name: "a" * 64 for name in names}, "after": {name: "b" * 64 for name in names},
                      "bound": {name: True for name in names}},
        "source": {**reports, "bands": K.registered_bands(robot)},
        "scope": {"probe": "P_reward", "measurement": {"track_lin_vel_xy_exp_rel_error": 1.0,
                                                         "track_ang_vel_z_exp_rel_error": 0.0}},
        "noop": {"before": {name: "a" * 64 for name in names}, "after": {name: "a" * 64 for name in names},
                 "unrelated_changed": []},
        "recovery": {"mode": "restart"},
    }
    m = K.make_manifest(robot, task, fault, correction, stages, [])
    refresh_evidence(m, tmp_path)
    return m


def refresh_evidence(manifest, root):
    evidence = root / "evidence.json"
    evidence.write_text(json.dumps({"schema": "combination_observation_v1", "development_fixture": True,
        "combination": {key: manifest[key] for key in ("robot", "task", "fault", "correction_id")},
        "stages": manifest["stages"]}), encoding="utf-8")
    manifest["evidence"] = [{"path": "evidence.json", "sha256_lf": K.sha(evidence)}]


def state_for(manifest):
    return {"status": "confirmed", "hypotheses": [K.FAULTS[manifest["fault"]][0]],
            "observed": {"P_reward": "abnormal"}, "selection_actor": "registered_rule"}


@pytest.mark.parametrize("robot", ["Go2", "ANYmal-C"])
def test_ready_requires_full_family_and_preserves_approval_consume_receipt(tmp_path, robot):
    manifest = make_case(tmp_path, robot=robot)
    state = state_for(manifest)
    ledger = K.CorrectionLedger(tmp_path / "ledger.jsonl", registry=K.registry())
    result = K.evaluate(manifest, tmp_path)
    assert result["status"] == "ready"
    assert [row["stage"] for row in result["trace"]] == list(K.ORDER)
    proposed = K.propose(manifest, state, ledger, root=tmp_path)
    rid = proposed["request_id"]
    assert proposed["evidence"]["restoration_targets"] == sorted(K.F.FUNCTIONS)
    with pytest.raises(K.LedgerError):
        K.consume(manifest, state, ledger, rid, root=tmp_path)
    ledger.approve(rid, "registered_operator")
    K.consume(manifest, state, ledger, rid, root=tmp_path)
    with pytest.raises(K.LedgerError):
        K.consume(manifest, state, ledger, rid, root=tmp_path)
    ledger.receipt(rid, "cpu_fixture", "recovered", "c" * 64, 1.0)
    assert [event["event"] for event in ledger.events()] == ["proposed", "approved", "consumed", "receipt"]
    with pytest.raises(K.LedgerError):
        K.propose(manifest, state, ledger, root=tmp_path)


def rejected_case(tmp_path, gate):
    m = make_case(tmp_path)
    if gate == "applicability":
        m["stages"][gate]["targets"][K.F.FUNCTIONS[0]]["effect_reachable"] = False
        status = "not_applicable"
    elif gate == "injection":
        m["stages"][gate]["after"] = deepcopy(m["stages"][gate]["before"])
        status = "invalid"
    elif gate == "source":
        m["stages"][gate]["fault"] = deepcopy(m["stages"][gate]["none"])
        status = "source_failure_absent"
    elif gate == "scope":
        m["correction_id"] = "restore_body_frame_reward"
        m["stages"]["noop"]["before"] = {K.F.FUNCTIONS[0]: "a" * 64}
        m["stages"]["noop"]["after"] = {K.F.FUNCTIONS[0]: "a" * 64}
        status = "scope_insufficient"
    elif gate == "noop":
        m["stages"][gate]["after"][K.F.FUNCTIONS[0]] = "c" * 64
        status = "invalid"
    else:
        m["stages"][gate]["mode"] = "resume"
        status = "invalid"
    refresh_evidence(m, tmp_path)
    return m, status


def assert_gate_contract(m, expected, gate, tmp_path):
    result = K.evaluate(m, tmp_path)
    assert result["status"] == expected
    assert [entry["stage"] for entry in result["trace"]] == list(K.ORDER[:K.ORDER.index(gate) + 1])


@pytest.mark.parametrize("gate", K.ORDER)
def test_each_gate_blocks_and_disabling_it_breaks_acceptance(tmp_path, monkeypatch, gate):
    m, expected = rejected_case(tmp_path, gate)
    assert_gate_contract(m, expected, gate, tmp_path)
    ledger = K.CorrectionLedger(tmp_path / "ledger.jsonl", registry=K.registry())
    event = K.propose(m, state_for(m), ledger, root=tmp_path)
    assert event["event"] == "held" and ledger.requests() == {}
    monkeypatch.setitem(K.GATES, gate, lambda m, row: (None, {"mutation": "gate_disabled"}))
    # 같은 수용식이 실제 관문 제거에서 실패해야 한다. production에는 제거 스위치가 없다.
    with pytest.raises(AssertionError):
        assert_gate_contract(m, expected, gate, tmp_path)
    assert K.evaluate(m, tmp_path)["status"] == "ready"


def test_stopped_gate_does_not_read_or_invent_later_stage_results(tmp_path, monkeypatch):
    m, _ = rejected_case(tmp_path, "applicability")
    m["stages"] = {"applicability": m["stages"]["applicability"]}
    refresh_evidence(m, tmp_path)
    for stage in K.ORDER[1:]:
        monkeypatch.setitem(K.GATES, stage, lambda *_: pytest.fail("뒤 관문을 호출했다"))
    result = K.evaluate(m, tmp_path)
    assert result["status"] == "not_applicable" and len(result["trace"]) == 1
    assert result["restoration_targets"] == [] and result["recovery_mode"] is None


@pytest.mark.parametrize("mutation", [
    lambda m: m.update(status="ready"),
    lambda m: m.update(task="wrong-task"),
    lambda m: m["stages"]["injection"]["bound"].update(track_lin_vel_xy_exp=False),
    lambda m: m["stages"]["source"]["fault"]["metrics"].update(lin_vel_rmse_mps=9.0),
    lambda m: m["stages"]["source"]["bands"].update(err_xy_ratio_max=float("nan")),
    lambda m: m["stages"]["source"]["bands"].update(err_xy_ratio_max=99.0),
    lambda m: m["stages"]["noop"].update(unrelated_changed=["unselected_function"]),
    lambda m: m["stages"].pop("noop"),
    lambda m: m["stages"]["scope"].update(measurement={}),
    lambda m: m["evidence"][0].update(sha256_lf="0" * 64),
    lambda m: m["evidence"][0].update(path="../outside.json"),
])
def test_malformed_or_inconsistent_evidence_never_creates_request(tmp_path, mutation):
    m = make_case(tmp_path)
    mutation(m)
    ledger = K.CorrectionLedger(tmp_path / "ledger.jsonl", registry=K.registry())
    assert K.evaluate(m, tmp_path)["status"] == "invalid"
    assert K.propose(m, state_for(m), ledger, root=tmp_path)["event"] == "held"
    assert not ledger.requests()


def test_none_noop_target_clean_fingerprint_must_match_injection_snapshot(tmp_path):
    m = make_case(tmp_path)
    for key in ("before", "after"):
        m["stages"]["noop"][key] = {name: "c" * 64 for name in K.F.FUNCTIONS}
    refresh_evidence(m, tmp_path)
    assert K.evaluate(m, tmp_path)["status"] == "invalid"


def test_unknown_diagnosis_holds_even_with_ready_applicability(tmp_path):
    m = make_case(tmp_path)
    ledger = K.CorrectionLedger(tmp_path / "ledger.jsonl", registry=K.registry())
    state = state_for(m)
    state["status"] = "active"
    assert K.propose(m, state, ledger, root=tmp_path)["reason"] == "diagnosis_unconfirmed"
    assert ledger.requests() == {}


def test_post_approval_source_or_state_change_and_registry_substitution_are_rejected(tmp_path):
    m = make_case(tmp_path)
    state = state_for(m)
    ledger = K.CorrectionLedger(tmp_path / "ledger.jsonl", registry=K.registry())
    rid = K.propose(m, state, ledger, root=tmp_path)["request_id"]
    ledger.approve(rid, "operator")
    changed = {**state, "selection_actor": "model"}
    with pytest.raises(K.LedgerError):
        K.consume(m, changed, ledger, rid, root=tmp_path)
    (tmp_path / "evidence.json").write_text("{}", encoding="utf-8")
    with pytest.raises(K.LedgerError):
        K.consume(m, state, ledger, rid, root=tmp_path)
    assert ledger.requests()[rid]["state"] == "approved"


def test_readback_selection_is_explicitly_rule_owned(tmp_path):
    m = make_case(tmp_path)
    m["stages"]["scope"] = {"probe": None, "measurement": None, "verification_kind": "registered_target_readback"}
    refresh_evidence(m, tmp_path)
    state = {"status": "confirmed", "hypotheses": ["reward_family"],
             "observed": {"target_readback": "abnormal"}, "selection_actor": "model"}
    ledger = K.CorrectionLedger(tmp_path / "ledger.jsonl", registry=K.registry())
    assert K.propose(m, state, ledger, root=tmp_path)["event"] == "held"
    state["selection_actor"] = "registered_rule"
    assert K.propose(m, state, ledger, root=tmp_path)["event"] == "proposed"


def test_unchanged_source_sha_does_not_attest_rewritten_manifest_facts(tmp_path):
    m, _ = rejected_case(tmp_path, "applicability")
    m["stages"]["applicability"]["targets"][K.F.FUNCTIONS[0]]["effect_reachable"] = True
    result = K.evaluate(m, tmp_path)
    assert result["status"] == "invalid" and result["trace"] == []


@pytest.mark.parametrize("fault", ["F2", "F6"])
def test_unregistered_scope_does_not_accept_reward_probe_for_another_mechanism(tmp_path, fault):
    manifest = make_case(tmp_path, fault=fault)
    result = K.evaluate(manifest, tmp_path)
    assert result["status"] == "invalid" and result["reason"] == "scope"
    assert result["trace"][-1]["details"]["reason"] == "scope_verification_unregistered"
    assert [row["stage"] for row in result["trace"]] == list(K.ORDER[:4])
    ledger = K.CorrectionLedger(tmp_path / "ledger.jsonl", registry=K.registry())
    assert K.propose(manifest, state_for(manifest), ledger, root=tmp_path)["event"] == "held"
    assert ledger.requests() == {}
