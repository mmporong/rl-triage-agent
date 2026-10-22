"""PPO ratio 계약 v2가 P_torque 하류 증상으로 actuator를 잘못 확정하지 않고, 실제 actuator 관측은 그대로 확정하는지 본다."""
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))
import ppo_ratio_repair_contract as V1  # noqa: E402
import ppo_ratio_repair_contract_v2 as C  # noqa: E402

PROBES = sorted(C.registry()["probes"])
ACTUATOR = {**{probe: "normal" for probe in PROBES}, "P_torque": "abnormal"}


def test_only_the_torque_surrogate_expectation_changes():
    old, new = V1.registry(), C.registry()
    assert new["probes"]["P_torque"]["expect"][C.SURROGATE] is None
    assert old["probes"]["P_torque"]["expect"][C.SURROGATE] == "normal"
    new["probes"]["P_torque"]["expect"][C.SURROGATE] = "normal"
    assert new == old


def test_v1_confirms_actuator_from_the_pilot_symptom_but_v2_does_not():
    assert V1.replay_loop("P_torque", C.PILOT_OUTCOMES)["conclusion"] == "actuator"
    for first in PROBES:
        result = C.replay_loop(first, C.PILOT_OUTCOMES)
        assert result["status"] == "confirmed" and result["conclusion"] == C.SURROGATE
        assert result["probes"][-1] == C.GRADIENT_PROBE
    assert C.replay_loop("P_torque", C.PILOT_OUTCOMES)["probes"] == ["P_torque", "P_noise", C.GRADIENT_PROBE]


@pytest.mark.parametrize("first", PROBES)
def test_true_actuator_observations_still_confirm_actuator(first):
    result = C.replay_loop(first, ACTUATOR)
    assert result["status"] == "confirmed" and result["conclusion"] == "actuator"
    assert len(result["probes"]) == len(V1.replay_loop(first, ACTUATOR)["probes"]) + (first == "P_torque")


def test_unknown_gradient_after_torque_symptom_stops_without_confirmation():
    outcomes = {**C.PILOT_OUTCOMES, C.GRADIENT_PROBE: "unknown"}
    assert C.replay_loop("P_torque", outcomes)["status"] == "unidentifiable"


def test_model_must_copy_the_null_optimizer_expectation_for_torque():
    packet = {"schema": C.PACKET_SCHEMA, "facts": {"x": 1.0}, "observed": [], "budget_left": C.MAX_PROBES,
              "execution_budget_s": 1200}
    ranking = ["optimizer", "actuator", "reward", "exploration", "physics", "termination"]
    decision = {"ranking": ranking, "probe": "P_torque", "action": "continue", "reason": "검사",
                "claims": [{"id": "x", "value": 1.0}],
                "expected_outcomes": {"optimizer": None, "actuator": "abnormal", "reward": "normal"}}
    assert C.validate_diagnosis(decision, packet) == decision
    with pytest.raises(ValueError, match="expected_outcomes"):
        C.validate_diagnosis({**decision, "expected_outcomes": {**decision["expected_outcomes"], "optimizer": "normal"}},
                             packet)
    with pytest.raises(ValueError, match="expected_outcomes"):
        V1.validate_diagnosis(decision, packet)
    assert "A null subtype expectation" in C.diagnosis_messages(packet)[0]["content"]


def test_correction_scope_is_unchanged():
    assert C.correction_registry() == V1.correction_registry()
    state = {"status": "confirmed", "hypotheses": [C.SURROGATE], "observed": {C.GRADIENT_PROBE: "abnormal"}}
    assert C.correction_allowed(state) and not C.correction_allowed({**state, "hypotheses": ["actuator"]})
