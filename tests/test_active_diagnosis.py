import json
from typing import Any, cast

import pytest

from rl_triage import active_diagnosis as A
from rl_triage import probe_loop as L


RANKING = list(L.MECHANISMS)


def packet(*receipts, budget_left=2):
    return {
        "facts": {"telemetry:return": -1.25, "reference:healthy": True},
        "observed": list(receipts),
        "budget_left": budget_left,
        "hidden_truth": "must-not-leak",
    }


def receipt(rid="r1", probe="P_reward", outcome="abnormal", measurement=None):
    return {
        "request_id": rid,
        "probe": probe,
        "outcome": outcome,
        "measurement": measurement if measurement is not None else {"relative_error": 0.4},
    }


def decision(**changes):
    out = {
        "ranking": RANKING,
        "probe": "P_value",
        "action": "continue",
        "reason": "공개 텔레메트리를 기준으로 다음 관측을 선택한다.",
        "claims": [{"id": "telemetry:return", "value": -1.25}],
    }
    out.update(changes)
    if "expected_outcomes" not in changes:
        if out["action"] == "continue" and out["probe"] in L.PROBES:
            out["expected_outcomes"] = {
                mechanism: L.PROBES[out["probe"]]["expect"][mechanism]
                for mechanism in out["ranking"][:3]
            }
        else:
            out["expected_outcomes"] = {}
    return out


def test_prompt_is_blind_and_requires_deterministic_receipt_grounding():
    messages = A.decision_messages(packet())
    assert [m["role"] for m in messages] == ["system", "user"]
    assert "temperature: 0" in messages[1]["content"]
    assert "must-not-leak" not in messages[1]["content"]
    assert "반드시 판단을 바꿔라" not in A.SYSTEM_PROMPT
    assert "not proof of\nruntime behavior" in A.SYSTEM_PROMPT
    payload = json.loads(messages[1]["content"].split("\n", 1)[1])
    assert set(payload) == {
        "evidence_facts", "observations", "budget_left", "execution_budget_s",
        "decision_context", "mechanisms", "probe_catalog", "thresholds",
    }
    assert payload["decision_context"] is None


def test_masked_and_observed_prompts_keep_same_phase_and_sanitized_previous_decision():
    previous = decision(secret="drop-me")
    actual = packet(receipt()) | {
        "phase": "after_receipt",
        "previous_decision": previous,
        "execution_budget_s": 180,
    }
    masked = actual | {"observed": []}
    actual_prompt = json.loads(A.build_prompt(actual).split("\n", 1)[1])
    masked_prompt = json.loads(A.build_prompt(masked).split("\n", 1)[1])
    assert A.build_prompt(actual).startswith("Decision phase: after_receipt")
    assert A.build_prompt(masked).startswith("Decision phase: after_receipt")
    assert actual_prompt["decision_context"] == masked_prompt["decision_context"]
    assert "secret" not in actual_prompt["decision_context"]
    assert actual_prompt["thresholds"] == {k: list(v) for k, v in L.THRESHOLDS.items()}


def test_receipt_facts_flattens_scalars_and_ignores_nonfinite_or_lists():
    facts = A.receipt_facts(receipt(measurement={
        "ratio": 2.0, "nested": {"count": 3, "ok": True}, "bad": float("nan"), "series": [1, 2]
    }))
    assert facts == {
        "receipt:r1:ratio": 2.0,
        "receipt:r1:nested.count": 3,
        "receipt:r1:nested.ok": True,
    }


@pytest.mark.parametrize("bad_probe", ["../P_reward", "P_reward/../../x", "", "P_missing", 3])
def test_continue_rejects_non_catalog_or_repeated_probe(bad_probe):
    with pytest.raises(ValueError, match="미관측 public probe"):
        A.validate_decision(decision(probe=bad_probe), packet())
    seen = receipt(probe="P_value", outcome="unknown", measurement={})
    post = decision(
        claims=[{"id": "telemetry:return", "value": -1.25}],
        probe=None,
        action="hold",
    )
    assert A.validate_decision(post, packet(seen))["action"] == "hold"


def test_continue_cannot_repeat_a_successful_probe():
    seen = receipt(probe="P_value", outcome="normal", measurement={"critic_change": 0.8})
    repeated = decision(
        probe="P_value",
        claims=[{"id": "receipt:r1:critic_change", "value": 0.8}],
    )
    with pytest.raises(ValueError, match="미관측 public probe"):
        A.validate_decision(repeated, packet(seen))


def test_validation_returns_fresh_object_and_enforces_strict_permutation():
    raw = decision()
    clean = A.validate_decision(raw, packet())
    assert clean == raw and clean is not raw and clean["ranking"] is not raw["ranking"]
    duplicate = decision(ranking=["reward"] * 6)
    with pytest.raises(ValueError, match="중복 없는 순열"):
        A.validate_decision(duplicate, packet())


def test_claims_must_be_known_exact_typed_finite_and_unique():
    for claims in (
        [{"id": "new:value", "value": 1}],
        [{"id": "telemetry:return", "value": -1.0}],
        [{"id": "reference:healthy", "value": 1}],
        [{"id": "telemetry:return", "value": float("inf")}],
        [
            {"id": "telemetry:return", "value": -1.25},
            {"id": "telemetry:return", "value": -1.25},
        ],
    ):
        with pytest.raises(ValueError):
            A.validate_decision(decision(claims=claims), packet())


def test_post_decision_must_cite_latest_actual_receipt():
    observed = receipt(measurement={"relative_error": 0.4})
    with pytest.raises(ValueError, match="최신 receipt"):
        A.validate_decision(decision(probe="P_value"), packet(observed))
    clean = A.validate_decision(
        decision(
            probe="P_value",
            claims=[{"id": "receipt:r1:relative_error", "value": 0.4}],
        ),
        packet(observed),
    )
    assert clean["claims"][0]["value"] == 0.4


def test_duplicate_receipts_and_fabricated_receipt_values_are_rejected():
    r = receipt()
    with pytest.raises(ValueError, match="중복 receipt"):
        A.evidence_state([r, dict(r)])
    forged = decision(
        probe="P_value",
        claims=[{"id": "receipt:r1:relative_error", "value": 0.5}],
    )
    with pytest.raises(ValueError, match="evidence fact와 다르다"):
        A.validate_decision(forged, packet(r))


def test_unknown_observation_does_not_eliminate_any_hypothesis():
    state = A.evidence_state([receipt(outcome="unknown", measurement={})], budget_left=1)
    assert state["hypotheses"] == list(L.MECHANISMS)
    assert state["dropped"] == []
    assert state["status"] == "open"


def test_initial_and_false_confirmations_are_rejected():
    initial = decision(action="confirm", probe=None)
    with pytest.raises(ValueError, match="초기 판단"):
        A.validate_decision(initial, packet())

    normal = receipt(outcome="normal")
    false = decision(
        action="confirm",
        probe=None,
        claims=[{"id": "receipt:r1:relative_error", "value": 0.4}],
    )
    with pytest.raises(ValueError, match="frozen evidence"):
        A.validate_decision(false, packet(normal))


def test_confirm_requires_frozen_confirmation_and_raw_model_top_match():
    abnormal = receipt()
    correct = decision(
        ranking=["reward"] + [m for m in L.MECHANISMS if m != "reward"],
        action="confirm",
        probe=None,
        claims=[{"id": "receipt:r1:relative_error", "value": 0.4}],
    )
    assert A.validate_decision(correct, packet(abnormal))["action"] == "confirm"

    wrong_top = dict(correct, ranking=["actuator", "reward", "exploration", "optimizer", "physics", "termination"])
    with pytest.raises(ValueError, match="raw ranking 1위"):
        A.validate_decision(wrong_top, packet(abnormal))


def test_raw_ranking_is_preserved_while_frozen_rules_are_separate():
    observed = receipt()
    raw = decision(
        ranking=["physics", "reward", "actuator", "exploration", "optimizer", "termination"],
        action="hold",
        probe=None,
        claims=[{"id": "receipt:r1:relative_error", "value": 0.4}],
    )
    clean = A.validate_decision(raw, packet(observed))
    assert clean["ranking"][0] == "physics"
    assert A.evidence_state([observed])["hypotheses"] == ["reward"]


def test_contrast_reports_actual_rejection_ablation_and_safe_confirm():
    observed = [receipt()]
    before = decision(ranking=["actuator", "reward", "exploration", "optimizer", "physics", "termination"])
    after = decision(
        ranking=["reward", "actuator", "exploration", "optimizer", "physics", "termination"],
        action="confirm",
        probe=None,
        claims=[{"id": "receipt:r1:relative_error", "value": 0.4}],
    )
    ablated = decision(ranking=["actuator", "reward", "exploration", "optimizer", "physics", "termination"])
    assert A.contrast(before, after, ablated, observed) == {
        "leader_changed": True,
        "ranking_changed": True,
        "latest_receipt_cited": True,
        "changed_leader_rejected_by_actual_observation": True,
        "after_leader_survives_actual_observation": True,
        "ablation_leader_diff": True,
        "ablation_preserves_before_leader": True,
        "safe_confirm": True,
    }


def test_terminal_actions_require_null_probe_and_continue_requires_budget():
    with pytest.raises(ValueError, match="probe는 null"):
        A.validate_decision(decision(action="hold"), packet())
    with pytest.raises(ValueError, match="남은 예산"):
        A.validate_decision(decision(), packet(budget_left=0))


def test_decision_rejects_extra_or_missing_fields():
    with pytest.raises(ValueError, match="decision 필드"):
        A.validate_decision(decision(shell_command="untrusted"), packet())
    missing = decision()
    del missing["reason"]
    with pytest.raises(ValueError, match="decision 필드"):
        A.validate_decision(missing, packet())


def test_readonly_receipt_sequence_matches_list_and_rejects_text():
    observed = [receipt()]
    assert A.evidence_state(tuple(observed)) == A.evidence_state(observed)
    assert A.evidence_state(()) == A.evidence_state([])
    initial = decision()
    assert A.contrast(initial, initial, initial, ()) == A.contrast(initial, initial, initial, [])
    with pytest.raises(ValueError, match="receipt 목록"):
        A.evidence_state(cast(Any, "not-receipts"))


def test_unknown_failed_or_over_budget_receipt_forces_hold_and_hides_metrics():
    for changes in (
        {"outcome": "unknown"},
        {"outcome": "abnormal", "over_budget": True},
        {"outcome": "abnormal", "exit_code": 1},
    ):
        bad = receipt(measurement={"relative_error": 999.0})
        bad.update(changes)
        assert A.receipt_facts(bad) == {}
        assert A.evidence_state([bad], 1)["hypotheses"] == list(L.MECHANISMS)
        hold = decision(action="hold", probe=None)
        assert A.validate_decision(hold, packet(bad))["action"] == "hold"
        with pytest.raises(ValueError, match="hold만 허용"):
            A.validate_decision(decision(probe="P_value"), packet(bad))


def test_expected_outcomes_are_bound_to_selected_probe_and_top_three():
    raw = decision()
    assert A.validate_decision(raw, packet())["expected_outcomes"] == {
        mechanism: L.PROBES["P_value"]["expect"][mechanism]
        for mechanism in RANKING[:3]
    }
    raw["expected_outcomes"][RANKING[0]] = "abnormal"
    with pytest.raises(ValueError, match="공개 카탈로그"):
        A.validate_decision(raw, packet())
