"""현재 허용된 행동 ID를 고르게 하고 기존 진단·승인 검증을 유지한다.

기존 action/target 응답을 보정하는 모듈이 아니다. 새 입력 계약에서 선택한
option_id와 fact_ids만 받고, 호스트가 등록된 action/target으로 변환한다.
"""
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

import evidence_guided_selector_v1 as SEL
import selection_comparison_call_v2 as CALL

SCHEMA = "registry_action_options_context_v1"
REVIEW_SCHEMA = "registry_action_options_review_v1"
SYSTEM = (
    "Select exactly one option_id from the current action_options. Return one JSON object "
    "with exactly option_id and fact_ids, matching output_schema. Do not invent an action, "
    "target, argument, diagnosis, repair, probability, or explanation. fact_ids must be exact "
    "currently visible non-null IDs. Reading evidence and holding require fact_ids=[]. "
    "A probe requires at least one supporting fact ID. For recorded counts, cite at least "
    "one complete listed count_evidence_group; if a group is touched, cite all three IDs. "
    "The host maps your option to a registered action, verifies evidence, and requires "
    "approval before execution. The options are possibilities, not recommendations. "
    "Choose hold if no available evidence supports a useful probe."
)


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _groups(facts: dict) -> list[list[str]]:
    groups = []
    for role in ("case", "reference"):
        if facts.get(f"runtime:{role}:available") is not True:
            continue
        for suffixes in (("expected_optimizer_steps", "optimizer_steps_min", "optimizer_steps_max"),
                         ("expected_minibatches", "minibatches_min", "minibatches_max")):
            ids = [f"runtime:{role}:{suffix}" for suffix in suffixes]
            if all(facts.get(name) is not None for name in ids):
                groups.append(ids)
    return groups


def context(state: dict) -> dict[str, Any]:
    """기존 원문·history를 검증한 뒤 실행 가능한 공개 선택지만 제공한다."""
    base = SEL.context(state)
    facts = base["evidence_facts"]
    groups = _groups(facts)
    options = [{"option_id": "hold", "action": "hold", "target": None,
                "citation_rule": "empty", "description": "Stop without a probe or repair."}]
    if base["status"] == "open" and base["budget_left"] > 0:
        if base["available_evidence"]:
            options.append({"option_id": "read_optimizer_counts", "action": "read_evidence",
                            "target": SEL.RESOURCE, "citation_rule": "empty",
                            "description": base["available_evidence"][0]["description"]})
        catalog = {row["target"]: row for row in base["probe_catalog"]}
        for target in base["eligible_probes"]:
            if target == SEL.NEW_PROBE and not groups:
                continue
            options.append({"option_id": f"probe:{target}", "action": "run_probe", "target": target,
                            "citation_rule": ("complete_count_group" if target == SEL.NEW_PROBE
                                              else "at_least_one_visible_fact"),
                            "description": catalog[target]["measures"],
                            "registered_gpu_cost_s": catalog[target]["cost_gpu_s"]})
    visible = sorted(name for name, value in facts.items() if value is not None)
    schema: dict[str, Any] = {
        "type": "object", "additionalProperties": False, "required": ["option_id", "fact_ids"],
        "properties": {
            "option_id": {"type": "string", "enum": [row["option_id"] for row in options]},
            "fact_ids": {"type": "array", "uniqueItems": True,
                         "items": ({"type": "string", "enum": visible} if visible else False)},
        },
    }
    return {"schema": SCHEMA, "state_digest": state["state_digest"],
            "evidence_facts": deepcopy(facts), "remaining_candidates": base["remaining_candidates"],
            "observations": base["observations"], "budget_left": base["budget_left"],
            "status": base["status"], "action_options": options,
            "count_evidence_groups": groups, "output_schema": schema,
            "format_example": {"option_id": "hold", "fact_ids": []},
            "contract": {"host_generated_action_target": True, "host_generated_fact_citations": False,
                         "old_response_reinterpretation": False, "approval_granted": False,
                         "recovery_authority": False}}


def messages(state: dict) -> list[dict[str, str]]:
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": json.dumps(context(state), ensure_ascii=False,
                                                     sort_keys=True, allow_nan=False)}]


def _unique_object(pairs: list[tuple[str, Any]]) -> dict:
    value = {}
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
        "context_digest": _digest(current), "response_text": content,
    }
    try:
        if not isinstance(content, str):
            raise TypeError("text JSON required")
        value = json.loads(content, object_pairs_hook=_unique_object,
                           parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    except (ValueError, TypeError):
        return {**review, "codes": ["F_OPTION_JSON"]}
    if not isinstance(value, dict) or set(value) != {"option_id", "fact_ids"}:
        return {**review, "codes": ["F_OPTION_FIELDS"]}
    if not isinstance(value["option_id"], str):
        return {**review, "codes": ["F_OPTION_ID"]}
    ids = value["fact_ids"]
    if (not isinstance(ids, list) or any(not isinstance(item, str) or not item for item in ids)
            or len(ids) != len(set(ids))):
        return {**review, "codes": ["F_FACT_IDS"]}
    review["format_status"] = "valid"
    options = {row["option_id"]: row for row in current["action_options"]}
    if value["option_id"] not in options:
        return {**review, "semantic_status": "invalid", "codes": ["S_OPTION_UNAVAILABLE"]}
    option = options[value["option_id"]]
    if option["citation_rule"] == "empty" and ids:
        return {**review, "semantic_status": "invalid", "codes": ["S_OPTION_EMPTY_CITATIONS"]}
    action = {"action": option["action"], "target": option["target"], "fact_ids": ids}
    inherited = SEL.assess(json.dumps(action, ensure_ascii=False, allow_nan=False), state)
    return {**review, "status": inherited["status"], "semantic_status": inherited["status"],
            "codes": inherited["codes"], "selected_option_id": value["option_id"],
            "action": action, "base_review": inherited}


def select_model(review: dict, state: dict) -> dict:
    """현재 원문을 다시 검증하며 옵션 ID로 승인·인자를 생성하지 않는다."""
    fresh = assess(review.get("response_text"), state)
    SEL.require(fresh == review, "option review가 현재 state/원문과 다르다")
    if fresh["status"] != "valid":
        return {"action": "hold", "target": None, "fact_ids": []}
    return SEL.select_model(fresh["base_review"], state)


def dialogue(state: dict, complete: Callable[[list, dict], dict], *,
             verify_context: Callable[[], object]) -> dict:
    """새 옵션 계약의 실제 응답과 최대 한 번의 피드백을 보존한다."""
    sent, calls = messages(state), []
    provider_ok = False
    for attempt in range(2):
        verify_context()
        result = complete(deepcopy(sent), CALL.F.request_params(CALL.MODEL))
        verify_context()
        SEL.require(isinstance(result, dict), "provider result 형식")
        provider_ok = (result.get("provider_model") in CALL.ACCEPTED_PROVIDER_MODELS
                       and result.get("finish_reason") == "stop"
                       and not result.get("error_type") and not result.get("infra_error"))
        review = (assess(result.get("content"), state) if provider_ok else {
            "status": "invalid", "format_status": "not_run", "semantic_status": "not_run",
            "codes": ["PROVIDER_UNAVAILABLE"]})
        calls.append({"attempt": attempt + 1, "messages": deepcopy(sent),
                      "messages_sha256": _digest(sent), "result": deepcopy(result), "review": review})
        if review["status"] == "valid" or not provider_ok:
            break
        if attempt == 0:
            current = context(state)
            sent.extend([{"role": "assistant", "content": result.get("content") or ""},
                         {"role": "user", "content": json.dumps({
                             "validation_errors": review["codes"],
                             "output_schema": current["output_schema"],
                             "action_options": current["action_options"],
                             "count_evidence_groups": current["count_evidence_groups"],
                             "required": "Use one current option_id and exact visible fact_ids only."},
                             ensure_ascii=False, sort_keys=True)}])
    final = calls[-1]["review"]
    action = (select_model(final, state) if provider_ok else
              {"action": "hold", "target": None, "fact_ids": []})
    return {"schema": "registry_action_options_dialogue_v1", "state_digest": state["state_digest"],
            "context_digest": _digest(context(state)), "calls": calls, "action": action,
            "final_review": deepcopy(final), "model_valid": final["status"] == "valid",
            "semantic_feedback_used": len(calls) == 2,
            "host_mapping": "new_option_contract_only_no_old_response_reinterpretation",
            "claims": {"diagnosis_improvement": False, "recovery": False}}
