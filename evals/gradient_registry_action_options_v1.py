"""8-subtype gradient 상태를 finite model option으로 투영한다."""
# pyright: reportMissingImports=false
from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "src")]

import gradient_hypothesis_contract_v1 as GH
import selection_comparison_call_v2 as CALL

CONTEXT_SCHEMA = "gradient_registry_action_context_v1"
REVIEW_SCHEMA = "gradient_registry_action_review_v1"
DIALOGUE_SCHEMA = "gradient_registry_action_dialogue_v1"
SYSTEM = (
    "Select exactly one current option_id and return one JSON object with exactly option_id "
    "and fact_ids. Use only exact currently visible registry fact IDs. A probe requires all "
    "required_fact_ids listed on that option. Hold requires fact_ids=[]. Do not invent a cause, "
    "repair, probability, rank, recommendation, argument, or observation. Registered probe "
    "semantics are not observations. A normal count result excludes only optimizer_update_count, "
    "not optimizer_surrogate. Unknown confirms no subtype. The host separately approves and "
    "revalidates any recorded pair before consumption."
)


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def digest(value: object) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                     allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _facts(probe: str, spec: dict[str, Any]) -> dict[str, object]:
    target = spec["confirms_on_abnormal"]
    return {
        f"registry:{probe}:metric": spec["metric"],
        f"registry:{probe}:normal_effect": {
            "eliminates_only": deepcopy(spec["eliminates_on_normal"])},
        f"registry:{probe}:abnormal_effect": {"confirms_only": target},
    }


def context(state: dict) -> dict[str, Any]:
    """검증된 8-subtype 상태에서 현재 실행 가능한 probe와 HOLD만 만든다."""
    base = GH.public_context(state)
    require(state.get("subtype_schema") == GH.SUBTYPE_SCHEMA,
            "8-subtype state만 option 계약에 사용할 수 있다")
    registry = GH.registry()
    visible_facts: dict[str, object] = {}
    options: list[dict[str, Any]] = [{
        "option_id": "hold", "action": "hold", "target": None,
        "required_fact_ids": [], "description": "Stop without consuming a recorded pair.",
    }]
    if base["status"] == "open" and base["budget_left"] > 0:
        for probe in base["eligible_probes"]:
            spec = registry["probes"][probe]
            facts = _facts(probe, spec)
            visible_facts.update(facts)
            options.append({
                "option_id": f"probe:{probe}", "action": "run_probe", "target": probe,
                "required_fact_ids": list(facts), "description": spec["metric"],
                "candidate_outcome_partition": {
                    "normal_eliminates": deepcopy(spec["eliminates_on_normal"]),
                    "abnormal_confirms": spec["confirms_on_abnormal"],
                    "unknown_confirms": None,
                },
                "paired_reference_contract": "reference_must_be_normal_else_unknown",
            })
    option_ids = [row["option_id"] for row in options]
    visible_ids = sorted(visible_facts)
    return {
        "schema": CONTEXT_SCHEMA,
        "state_digest": state["state_digest"],
        "status": base["status"],
        "budget_left": base["budget_left"],
        "remaining_candidates": deepcopy(state["hypotheses"]),
        "observations": deepcopy(base["observations"]),
        "visible_facts": visible_facts,
        "action_options": options,
        "output_schema": {
            "type": "object", "additionalProperties": False,
            "required": ["option_id", "fact_ids"],
            "properties": {
                "option_id": {"type": "string", "enum": option_ids},
                "fact_ids": {"type": "array", "uniqueItems": True,
                             "items": ({"type": "string", "enum": visible_ids}
                                       if visible_ids else False)},
            },
        },
        "evidence_kind": base["evidence_kind"],
        "function_permissions": deepcopy(base["function_permissions"]),
        "contract": {
            "registered_predictions_are_not_observations": True,
            "count_normal_does_not_remove_optimizer_surrogate": True,
            "host_approval_required": True,
            "correction_authority": False,
            "recovery_authority": False,
        },
    }


def messages(state: dict) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": json.dumps(
            context(state), ensure_ascii=False, sort_keys=True, allow_nan=False)},
    ]


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON key")
        value[key] = item
    return value


def assess(content: object, state: dict) -> dict[str, Any]:
    current = context(state)
    review: dict[str, Any] = {
        "schema": REVIEW_SCHEMA, "status": "invalid", "format_status": "invalid",
        "semantic_status": "not_run", "codes": [], "state_digest": state["state_digest"],
        "context_digest": digest(current), "response_text": content,
    }
    try:
        if not isinstance(content, str):
            raise TypeError("text JSON required")
        value = json.loads(content, object_pairs_hook=_unique_object,
                           parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    except (TypeError, ValueError):
        return {**review, "codes": ["F_OPTION_JSON"]}
    if not isinstance(value, dict) or set(value) != {"option_id", "fact_ids"}:
        return {**review, "codes": ["F_OPTION_FIELDS"]}
    if not isinstance(value["option_id"], str):
        return {**review, "codes": ["F_OPTION_ID"]}
    fact_ids = value["fact_ids"]
    if (not isinstance(fact_ids, list)
            or any(not isinstance(item, str) or not item for item in fact_ids)
            or len(fact_ids) != len(set(fact_ids))):
        return {**review, "codes": ["F_FACT_IDS"]}
    review["format_status"] = "valid"
    options = {row["option_id"]: row for row in current["action_options"]}
    option = options.get(value["option_id"])
    if option is None:
        return {**review, "semantic_status": "invalid", "codes": ["S_OPTION_UNAVAILABLE"]}
    required = option["required_fact_ids"]
    if set(fact_ids) != set(required):
        return {**review, "semantic_status": "invalid", "codes": ["S_FACT_IDS_EXACT"]}
    if any(item not in current["visible_facts"] for item in fact_ids):
        return {**review, "semantic_status": "invalid", "codes": ["S_FACT_ID_UNKNOWN"]}
    action = {"action": option["action"], "target": option["target"],
              "fact_ids": deepcopy(fact_ids)}
    return {**review, "status": "valid", "semantic_status": "valid", "codes": [],
            "selected_option_id": value["option_id"], "action": action}


def select_model(review: dict, state: dict) -> dict[str, Any]:
    fresh = assess(review.get("response_text"), state)
    require(fresh == review, "gradient option review가 현재 state/context/원문과 다르다")
    if fresh["status"] != "valid":
        return {"action": "hold", "target": None, "fact_ids": []}
    return deepcopy(fresh["action"])


def dialogue(state: dict, complete: Callable[[list, dict], dict], *,
             verify_context: Callable[[], object]) -> dict[str, Any]:
    """등록된 3.5 provider 응답 한 건만 평가하고 invalid는 HOLD로 닫는다."""
    sent = messages(state)
    verify_context()
    result = complete(deepcopy(sent), CALL.F.request_params(CALL.MODEL))
    verify_context()
    require(isinstance(result, dict), "provider result 형식")
    provider_ok = (result.get("provider_model") in CALL.ACCEPTED_PROVIDER_MODELS
                   and result.get("finish_reason") == "stop"
                   and not result.get("error_type") and not result.get("infra_error"))
    review = (assess(result.get("content"), state) if provider_ok else {
        "schema": REVIEW_SCHEMA, "status": "invalid", "format_status": "not_run",
        "semantic_status": "not_run", "codes": ["PROVIDER_UNAVAILABLE"],
        "state_digest": state["state_digest"], "context_digest": digest(context(state)),
        "response_text": result.get("content"),
    })
    action = (select_model(review, state) if provider_ok else
              {"action": "hold", "target": None, "fact_ids": []})
    return {
        "schema": DIALOGUE_SCHEMA, "state_digest": state["state_digest"],
        "context_digest": digest(context(state)),
        "call": {"messages": sent, "messages_sha256": digest(sent),
                 "result": deepcopy(result), "review": deepcopy(review)},
        "action": action, "model_valid": review["status"] == "valid",
        "feedback_calls": 0,
        "provider_kind": result.get("provider_kind", "actual_model"),
        "claims": {"provider_response_consumed": True, "diagnosis_improvement": False,
                   "recovery": False},
    }


__all__ = ["CONTEXT_SCHEMA", "DIALOGUE_SCHEMA", "REVIEW_SCHEMA", "assess", "context",
           "dialogue", "messages", "select_model"]
