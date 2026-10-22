"""PPO ratio 판의 검사 등록·고리 확정·교정 범위·모델 계약을 GPU·API 없이 검증한다."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))
import ppo_ratio_repair_contract as C  # noqa: E402
import r1_repair_contract as R1  # noqa: E402

EXPECTED = {**{name: "normal" for name in C.registry()["probes"]}, C.GRADIENT_PROBE: "abnormal"}


def packet(**changes):
    return {"schema": C.PACKET_SCHEMA, "facts": {"telemetry:case:loss": 1.5}, "observed": [],
            "budget_left": C.MAX_PROBES, "execution_budget_s": 1200} | changes


def decision(probe="P_reward", ranking=None):
    ranking = ranking or ["reward", "actuator", "optimizer", "exploration", "physics", "termination"]
    return {"ranking": ranking, "probe": probe, "action": "continue", "reason": "검사",
            "claims": [{"id": "telemetry:case:loss", "value": 1.5}],
            "expected_outcomes": {name: C.public_catalog()[probe]["expect"][name] for name in ranking[:3]}}


def test_registry_adds_one_subtype_and_one_probe_without_changing_r1_entries():
    reg, old = C.registry(), R1.registry()
    assert reg["mechanisms"] == [*old["mechanisms"], C.SURROGATE]
    assert set(reg["probes"]) == set(old["probes"]) | {C.GRADIENT_PROBE}
    for name, spec in old["probes"].items():
        expect = dict(reg["probes"][name]["expect"])
        assert expect.pop(C.SURROGATE) == "normal"
        assert {**reg["probes"][name], "expect": expect} == spec
    assert reg["fixed_order"] == [*old["fixed_order"], C.GRADIENT_PROBE]


def test_each_registered_probe_confirms_exactly_one_subtype():
    reg = C.registry()
    confirming = {C.M._confirming(reg, name): name for name in reg["mechanisms"]}
    assert confirming[C.GRADIENT_PROBE] == C.SURROGATE
    assert confirming[C.COUNT_PROBE] == C.COUNT and confirming["P_value"] == C.VALUE
    assert None not in confirming and len(confirming) == len(reg["mechanisms"])


@pytest.mark.parametrize("first", sorted(C.registry()["probes"]))
def test_expected_observations_confirm_surrogate_from_any_first_probe(first):
    result = C.replay_loop(first, EXPECTED)
    assert result["status"] == "confirmed" and result["conclusion"] == C.SURROGATE
    assert result["probes"][-1] == C.GRADIENT_PROBE and len(result["probes"]) <= C.MAX_PROBES


@pytest.mark.parametrize(("changed", "status", "conclusion"), [
    ({"P_value": "abnormal"}, "confirmed", C.VALUE),
    ({"P_episode": "unknown"}, "unidentifiable", None),
    ({C.GRADIENT_PROBE: "normal"}, "none_supported", None),
])
def test_competing_observations_never_confirm_surrogate(changed, status, conclusion):
    result = C.replay_loop("P_reward", {**EXPECTED, **changed})
    assert (result["status"], result["conclusion"]) == (status, conclusion)
    state = {"status": result["status"], "hypotheses": [conclusion] if conclusion else [],
             "observed": {probe: {**EXPECTED, **changed}[probe] for probe in result["probes"]}}
    assert not C.correction_allowed(state)


def test_gradient_probe_alone_confirms_and_correction_scope_requires_gradient_abnormal():
    state = C.advance(C.initial_state(decision(C.GRADIENT_PROBE)), C.GRADIENT_PROBE, "abnormal")
    assert state["status"] == "confirmed" and state["hypotheses"] == [C.SURROGATE]
    assert C.correction_allowed(state)
    assert not C.correction_allowed({**state, "observed": {C.GRADIENT_PROBE: "unknown"}})
    assert not C.correction_allowed({**state, "hypotheses": [C.COUNT]})


def test_unknown_stops_and_advance_rejects_out_of_order_probe():
    state = C.initial_state(decision())
    with pytest.raises(ValueError):
        C.advance(state, "P_value", "normal")
    stopped = C.advance(state, "P_reward", "unknown")
    assert stopped["status"] == "unidentifiable" and stopped["next_probe"] is None


def test_initial_state_expands_optimizer_into_three_subtypes():
    state = C.initial_state(decision())
    assert state["hypotheses"].count(C.SURROGATE) == 1
    assert set(C.SUBTYPES) <= set(state["hypotheses"]) and len(state["hypotheses"]) == 8


def test_public_catalog_hides_differing_optimizer_subtypes():
    catalog = C.public_catalog()
    assert catalog[C.GRADIENT_PROBE]["expect"]["optimizer"] is None
    assert catalog["P_reward"]["expect"]["optimizer"] == "normal"


def test_correction_registry_changes_only_the_optimizer_entry():
    reg = C.correction_registry()
    assert reg["optimizer"]["correction_id"] == C.CORRECTION and reg["optimizer"]["recovery_modes"] == ["restart"]
    assert {k: v for k, v in reg.items() if k != "optimizer"} == {
        k: v for k, v in R1.correction_registry().items() if k != "optimizer"}


def test_diagnosis_messages_state_the_gradient_rule_and_packet_contract():
    messages = C.diagnosis_messages(packet())
    body = json.loads(messages[1]["content"])
    assert body["surrogate_gradient_rule"] == C.GRADIENT_RULE and body["budget_left"] == C.MAX_PROBES
    assert "three registered subtypes" in messages[0]["content"]
    for bad in (packet(schema=R1.L.digest({})), packet(budget_left=7), packet(observed=["P_reward"])):
        with pytest.raises(ValueError):
            C.diagnosis_messages(bad)


@pytest.mark.parametrize("mutation", [
    lambda d: d.update(ranking=d["ranking"][:5]),
    lambda d: d.update(probe="P_unregistered"),
    lambda d: d.update(action="confirm"),
    lambda d: d.update(claims=[{"id": "telemetry:case:loss", "value": 2.0}]),
    lambda d: d["expected_outcomes"].update(actuator="abnormal"),
])
def test_invalid_diagnosis_is_rejected(mutation):
    assert C.validate_diagnosis(decision(), packet())["probe"] == "P_reward"
    bad = deepcopy(decision())
    mutation(bad)
    with pytest.raises(ValueError):
        C.validate_diagnosis(bad, packet())


def test_correction_packet_requires_confirmed_surrogate():
    state = C.advance(C.initial_state(decision(C.GRADIENT_PROBE)), C.GRADIENT_PROBE, "abnormal")
    confirming = {"request_id": "r1", "outcome": "abnormal", "measurement": {"case": {"gradient_status": "abnormal"}}}
    report = {"protocol": "fixed_eval_v1", "task": "Go2", "metrics": {"x": 1.0}, "standstill": {"y": 1.0}}
    built = C.correction_packet(state, confirming, report, "a" * 64, "b" * 64)
    assert built["rule_result"] == {"mode": "restart", "correction_id": C.CORRECTION}
    assert built["facts"]["confirmed_subtype"] == C.SURROGATE
    with pytest.raises(ValueError):
        C.correction_packet({**state, "hypotheses": [C.COUNT]}, confirming, report, "a" * 64, "b" * 64)
