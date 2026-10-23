from __future__ import annotations

import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))
import ppo_ratio_repair_model_call_v2 as M  # noqa: E402


def packet() -> dict:
    return {
        "schema": M.C.PACKET_SCHEMA,
        "facts": {"telemetry:error": 1.2},
        "observed": [],
        "budget_left": M.C.MAX_PROBES,
        "execution_budget_s": 1200,
    }


def valid_decision(value: dict, probe: str = "P_reward") -> dict:
    ranking = ["reward", "actuator", "optimizer", "exploration", "physics", "termination"]
    return {
        "ranking": ranking,
        "probe": probe,
        "action": "continue",
        "reason": "등록 검사를 선택한다",
        "claims": [{"id": "telemetry:error", "value": value["facts"]["telemetry:error"]}],
        "expected_outcomes": {
            mechanism: M.C.public_catalog()[probe]["expect"][mechanism]
            for mechanism in ranking[:3]
        },
    }


def encoded(value: dict) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")


def response(content: object, provider: str = M.MODEL) -> dict:
    text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
    return {
        "response_id": "response",
        "provider_model": provider,
        "finish_reason": "stop",
        "content": text,
        "reasoning_content": None,
        "input_tokens": 10,
        "output_tokens": 5,
        "total_tokens": 15,
        "wall_s": 0.1,
    }


@pytest.mark.parametrize("provider", sorted(M.ACCEPTED_PROVIDER_MODELS))
def test_requested_and_registered_provider_alias_are_accepted(provider):
    value = packet()

    def complete(_messages, params):
        assert params == M.F.request_params(M.MODEL)
        return response(valid_decision(value), provider)

    record = M.dialogue("diagnosis", encoded(value), complete)
    reviewed = M.review(record, encoded(value), "diagnosis")
    assert record["request"]["model"] == M.MODEL
    assert reviewed["status"] == "valid" and reviewed["requests"] == 1


def test_unregistered_provider_is_rejected_without_retry_or_fallback():
    value = packet()
    data = encoded(value)
    record = M.dialogue(
        "diagnosis",
        data,
        lambda *_: response(valid_decision(value), "nvidia/another-model"),
    )
    assert record["requests"] == 1 and record["final"] is None
    assert M.review(record, data, "diagnosis") == {
        "status": "call_error",
        "requests": 1,
        "reason": "ProviderModelMismatch",
    }

    changed = deepcopy(record)
    changed["calls"][0]["provider_model"] = M.MODEL
    with pytest.raises(ValueError, match="불일치 근거"):
        M.review(changed, data, "diagnosis")


def test_request_params_and_response_provider_tampering_are_rejected():
    value = packet()
    data = encoded(value)
    record = M.dialogue("diagnosis", data, lambda *_: response(valid_decision(value)))

    changed_request = deepcopy(record)
    changed_request["request"]["model"] = "nvidia/another-model"
    with pytest.raises(ValueError, match="요청 결합"):
        M.review(changed_request, data, "diagnosis")

    changed_provider = deepcopy(record)
    changed_provider["calls"][0]["provider_model"] = "private/nvidia/another-model"
    with pytest.raises(ValueError, match="응답 모델"):
        M.review(changed_provider, data, "diagnosis")


def test_invalid_first_response_gets_one_fixed_feedback_then_valid_final():
    value, calls = packet(), []

    def complete(messages, _params):
        calls.append(messages)
        content = {} if len(calls) == 1 else valid_decision(value)
        return response(content, "private/nvidia/nemotron-3.5-super-120b-a12b")

    data = encoded(value)
    record = M.dialogue("diagnosis", data, complete)
    reviewed = M.review(record, data, "diagnosis")
    assert reviewed["status"] == "valid" and reviewed["retried"] is True
    assert record["feedback"]["codes"] == record["calls"][0]["codes"]
    assert calls[1][-2] == {"role": "assistant", "content": record["calls"][0]["content"]}
    assert calls[1][-1] == {"role": "user", "content": record["feedback"]["text"]}

    changed = deepcopy(record)
    changed["feedback"]["text"] += " changed"
    with pytest.raises(ValueError, match="피드백 결합"):
        M.review(changed, data, "diagnosis")


def test_second_invalid_response_remains_invalid_for_controller_hold():
    data = encoded(packet())
    record = M.dialogue("diagnosis", data, lambda *_: response({}))
    reviewed = M.review(record, data, "diagnosis")
    assert record["requests"] == 2 and record["final"]["step"] == "retry"
    assert reviewed["status"] == "invalid" and reviewed["retried"] is True
    assert "decision" not in reviewed


def test_final_and_reservation_binding_are_recomputed():
    value = packet()
    data = encoded(value)
    record = M.dialogue("diagnosis", data, lambda *_: response(valid_decision(value)))
    reservation = {
        "kind": "diagnosis",
        "packet_sha256": hashlib.sha256(data).hexdigest(),
        "at": "2026-10-23T00:00:00+00:00",
    }
    record["reservation_sha256"] = M.C.L.digest(reservation)
    assert M.review(record, data, "diagnosis", reservation)["status"] == "valid"

    changed_final = deepcopy(record)
    changed_final["final"]["step"] = "retry"
    with pytest.raises(ValueError, match="최종 모델 판정"):
        M.review(changed_final, data, "diagnosis", reservation)

    changed_reservation = {**reservation, "packet_sha256": "0" * 64}
    with pytest.raises(ValueError, match="예약"):
        M.review(record, data, "diagnosis", changed_reservation)
