from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))
import ppo_selection_adapter_v1 as A  # pyright: ignore[reportMissingImports]


def packet() -> dict:
    return {
        "schema": A.C.PACKET_SCHEMA,
        "facts": {"telemetry:error": 1.2, "telemetry:alive": 20},
        "observed": [],
        "budget_left": A.C.MAX_PROBES,
        "execution_budget_s": 1200,
    }


def choice(probe: str = "P_reward", *, action: str = "continue", facts=None) -> dict:
    return {"action": action, "probe": probe if action == "continue" else None,
            "fact_ids": ["telemetry:error"] if facts is None else facts}


def response(body: object, *, provider: str = A.MODEL, infra: str | None = None) -> dict:
    if infra:
        return {"infra_error": infra}
    return {"response_id": "r", "provider_model": provider, "finish_reason": "stop",
            "content": body if isinstance(body, str) else json.dumps(body),
            "reasoning_content": None, "input_tokens": 2, "output_tokens": 3,
            "total_tokens": 5, "wall_s": 0.1}


def test_snapshot_and_prompt_hide_answers_arguments_and_private_paths():
    value = A.build_snapshot(packet())
    assert len(value["eligible_probes"]) == 8
    assert set(value["subtype_registry"]) == set(A.C.L.MECHANISMS)
    assert all(set(row) == {"probe", "measures", "cost_gpu_s"} for row in value["eligible_probes"])
    text = json.dumps(A.messages(value), ensure_ascii=False)
    assert "expected_outcomes" not in text and '"expect"' not in text and '"args"' not in text
    assert "ranking" not in A.messages(value)[0]["content"]
    assert "C:\\" not in text and "/home/" not in text


@pytest.mark.parametrize("first", list(A.C.public_catalog()))
def test_every_first_probe_reaches_surrogate_with_registered_followup(first):
    replay = A.C.replay_loop(first, A.C.PILOT_OUTCOMES)
    assert replay["status"] == "confirmed"
    assert replay["conclusion"] == A.C.SURROGATE


def test_torque_abnormal_first_does_not_confirm_actuator():
    replay = A.C.replay_loop("P_torque", A.C.PILOT_OUTCOMES)
    assert replay["conclusion"] == A.C.SURROGATE
    assert replay["probes"][0] == "P_torque" and len(replay["probes"]) > 1


@pytest.mark.parametrize("bad", [
    {"action": "continue", "probe": "P_reward", "fact_ids": ["telemetry:error"], "ranking": []},
    {"action": "continue", "probe": "P_missing", "fact_ids": ["telemetry:error"]},
    {"action": "continue", "probe": "P_reward", "fact_ids": ["missing"]},
    {"action": "continue", "probe": "P_reward", "fact_ids": ["telemetry:error"], "args": {}},
])
def test_unknown_or_extra_model_fields_are_rejected(bad):
    with pytest.raises(ValueError):
        A.validate_choice(A.build_snapshot(packet()), json.dumps(bad))


@pytest.mark.parametrize("body", ["", "not-json", "[]"])
def test_non_object_output_is_rejected(body):
    with pytest.raises(ValueError):
        A.validate_choice(A.build_snapshot(packet()), body)


@pytest.mark.parametrize("key", ["action", "probe", "fact_ids"])
def test_duplicate_output_key_is_rejected(key):
    pairs = [
        '"action":"continue"',
        '"probe":"P_reward"',
        '"fact_ids":["telemetry:error"]',
    ]
    duplicate = {
        "action": '"action":"hold"',
        "probe": '"probe":"P_value"',
        "fact_ids": '"fact_ids":["telemetry:alive"]',
    }[key]
    with pytest.raises(ValueError, match="F_FIELDS"):
        A.validate_choice(A.build_snapshot(packet()), "{" + ",".join([*pairs, duplicate]) + "}")


def test_host_generates_canonical_ranking_expectations_and_claim_values():
    snapshot = A.build_snapshot(packet())
    hosted = A.host_diagnosis(snapshot, choice("P_torque"))
    decision, metadata = hosted["decision"], hosted["metadata"]
    assert decision["ranking"] == list(A.C.L.MECHANISMS)
    assert decision["expected_outcomes"] == {
        name: A.C.public_catalog()["P_torque"]["expect"][name]
        for name in decision["ranking"][:3]
    }
    assert decision["expected_outcomes"]["exploration"] is None
    assert decision["claims"] == [{"id": "telemetry:error", "value": 1.2}]
    assert metadata == {
        "selection_fields_generated_by": "model",
        "ranking_generated_by": "host_registry",
        "expected_outcomes_generated_by": "host_registry",
        "claim_values_generated_by": "host_packet",
        "reason_generated_by": "host_template",
    }


def test_invalid_then_fixed_feedback_then_valid_choice():
    rows = [response({"ranking": []}), response(choice("P_value"))]
    sent = []

    def complete(messages, params):
        sent.append((messages, params))
        return rows.pop(0)

    result = A.engine(A.build_snapshot(packet()), complete)
    assert result["status"] == "choice_ready" and len(result["calls"]) == 2
    assert result["selection"]["probe"] == "P_value"
    assert set(result["host_decision"]) == {
        "ranking", "probe", "action", "reason", "claims", "expected_outcomes",
    }
    assert result["host_metadata"]["expected_outcomes_generated_by"] == "host_registry"
    assert sent[1][0][-1]["role"] == "user" and "F_FIELDS" in sent[1][0][-1]["content"]
    assert all(params == A.CALL.F.request_params(A.MODEL) for _, params in sent)


def test_invalid_after_one_feedback_holds_without_rule_fallback():
    result = A.engine(A.build_snapshot(packet()), lambda *_: response({"ranking": []}))
    assert result["status"] == "terminal_hold" and result["reason"] == "selection_invalid"
    assert result["selection"] is None and result["host_decision"] is None
    assert len(result["calls"]) == 2


def test_true_model_hold_stays_hold_without_rule_fallback():
    result = A.engine(A.build_snapshot(packet()), lambda *_: response(choice(action="hold")))
    assert result["status"] == "terminal_hold" and result["reason"] == "model_hold"
    assert result["selection"]["action"] == "hold" and result["host_decision"] is None


def test_one_infrastructure_retry_then_success_and_retry_limit():
    rows = [response({}, infra="ServiceUnavailable"), response(choice("P_episode"))]
    result = A.engine(A.build_snapshot(packet()), lambda *_: rows.pop(0))
    assert result["status"] == "choice_ready" and len(result["calls"]) == 2

    failed = A.engine(A.build_snapshot(packet()), lambda *_: response({}, infra="ServiceUnavailable"))
    assert failed["status"] == "terminal_hold" and failed["reason"] == "transport_infra_error"
    assert len(failed["calls"]) == 2


def test_timeout_exception_is_sanitized_and_retried_once():
    count = 0

    def complete(*_):
        nonlocal count
        count += 1
        if count == 1:
            raise TimeoutError("secret provider detail")
        return response(choice("P_physics"))

    result = A.engine(A.build_snapshot(packet()), complete)
    assert result["status"] == "choice_ready" and len(result["calls"]) == 2
    assert result["calls"][0]["result"] == {"error_type": "TimeoutError"}


def test_infra_then_invalid_then_invalid_stays_within_three_calls():
    rows = [response({}, infra="ServiceUnavailable"), response({}), response({})]
    result = A.engine(A.build_snapshot(packet()), lambda *_: rows.pop(0))
    assert result["status"] == "terminal_hold" and len(result["calls"]) == 3


def test_unregistered_provider_is_unknown_and_never_falls_back():
    result = A.engine(A.build_snapshot(packet()),
                      lambda *_: response(choice(), provider="nvidia/another-model"))
    assert result["status"] == "terminal_hold" and result["reason"] == "transport_unknown"
    assert len(result["calls"]) == 1 and result["host_decision"] is None


def test_registered_provider_alias_is_accepted():
    alias = "private/nvidia/nemotron-3.5-super-120b-a12b"
    result = A.engine(A.build_snapshot(packet()), lambda *_: response(choice(), provider=alias))
    assert result["status"] == "choice_ready"


def test_snapshot_tampering_is_rejected_before_provider_call():
    snapshot = A.build_snapshot(packet())
    snapshot["evidence_facts"]["telemetry:error"] = 9.9
    with pytest.raises(ValueError, match="digest"):
        A.engine(snapshot, lambda *_: pytest.fail("provider called"))
