"""등록 검사 결과가 현재 후보를 어떻게 나누는지 공개하는 option 계약."""
# pyright: reportMissingImports=false
from __future__ import annotations

import json
import sys
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "src")]

import registry_action_options_v1 as O1

SEL, CALL = O1.SEL, O1.CALL
SCHEMA = "registry_action_options_context_v2"
REVIEW_SCHEMA = "registry_action_options_review_v2"
SYSTEM = (
    "Select exactly one option_id from the current action_options. Return one JSON object "
    "with exactly option_id and fact_ids, matching output_schema. Do not invent an action, "
    "target, argument, diagnosis, repair, probability, rank, recommendation, or explanation. "
    "fact_ids must be exact currently visible non-null IDs. Reading evidence and holding require "
    "fact_ids=[]. A probe requires at least one supporting fact ID. For recorded counts, cite at "
    "least one complete listed count_evidence_group; if a group is touched, cite all three IDs. "
    "Before a probe, distinguish insufficient confirmation evidence from a probe having no useful "
    "candidate partition. A symptom is not a confirmed cause. An allowed probe with possible known "
    "outcomes that split current candidates may be selected, or hold may be selected at any time. "
    "Unknown outcomes confirm no candidate. For recorded counts specifically, a non-normal "
    "reference makes the paired outcome unknown. Legacy probes retain their registered raw paired "
    "classifier. The host maps the option, verifies evidence, and requires approval before execution."
)


def _partition(target: str, candidates: list[str]) -> dict[str, list[str]]:
    if target == SEL.NEW_PROBE:
        predicted = {name: ("abnormal" if name == "optimizer_update_count" else "normal")
                     for name in candidates}
    else:
        registry = SEL.S.C1.registry()
        predicted = {name: registry["probes"][target]["expect"][name] for name in candidates}
    return {
        outcome: [name for name in candidates if predicted[name] == value]
        for outcome, value in (("normal", "normal"), ("abnormal", "abnormal"),
                               ("unspecified", None))
    }


def context(state: dict) -> dict[str, Any]:
    """O1 state 검증 뒤 현재 후보에 한정한 등록 outcome partition을 덧붙인다."""
    base = O1.context(state)
    candidates = list(base["remaining_candidates"])
    options = deepcopy(base["action_options"])
    for option in options:
        if option["action"] == "run_probe":
            option["candidate_outcome_partition"] = _partition(option["target"], candidates)
            option["outcome_semantics"] = {
                "unknown": "confirms_no_candidate",
                "paired_reference_contract": (
                    "reference_outcome_must_be_normal_else_unknown"
                    if option["target"] == SEL.NEW_PROBE
                    else "registered_classifier_uses_role_bound_raw_reference_measurement"),
                "unspecified_candidates_remain_after_normal_or_abnormal": True,
                "registered_predictions_are_not_observations": True,
                "registered_gpu_cost": "budget_value_not_measured_runtime",
            }
        elif option["action"] == "read_evidence":
            option["evidence_semantics"] = {
                "values_hidden_until_read": True,
                "additional_sdk_launches": 0,
                "condition": "recorded_training_counts_already_exist",
            }
    return {
        "schema": SCHEMA,
        "state_digest": base["state_digest"],
        "evidence_facts": deepcopy(base["evidence_facts"]),
        "remaining_candidates": candidates,
        "observations": deepcopy(base["observations"]),
        "budget_left": base["budget_left"],
        "status": base["status"],
        "action_options": options,
        "count_evidence_groups": deepcopy(base["count_evidence_groups"]),
        "output_schema": deepcopy(base["output_schema"]),
        "contract": {**deepcopy(base["contract"]),
                     "candidate_partitions_from_registered_semantics": True,
                     "fault_truth_exposed": False,
                     "probability_or_eig": False,
                     "probe_ranking_or_recommendation": False},
    }


def messages(state: dict) -> list[dict[str, str]]:
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": json.dumps(
                context(state), ensure_ascii=False, sort_keys=True, allow_nan=False)}]


def assess(content: object, state: dict) -> dict[str, Any]:
    current = context(state)
    base = O1.assess(content, state)
    return {
        "schema": REVIEW_SCHEMA,
        "status": base["status"],
        "format_status": base["format_status"],
        "semantic_status": base["semantic_status"],
        "codes": deepcopy(base["codes"]),
        "state_digest": state["state_digest"],
        "context_digest": O1._digest(current),
        "response_text": content,
        **({"selected_option_id": base["selected_option_id"],
            "action": deepcopy(base["action"])} if "selected_option_id" in base else {}),
        "base_review": base,
    }


def select_model(review: dict, state: dict) -> dict:
    fresh = assess(review.get("response_text"), state)
    SEL.require(fresh == review, "V2 option review가 현재 state/context/원문과 다르다")
    if fresh["status"] != "valid":
        return {"action": "hold", "target": None, "fact_ids": []}
    return O1.select_model(fresh["base_review"], state)


def dialogue(state: dict, complete: Callable[[list, dict], dict], *,
             verify_context: Callable[[], object]) -> dict:
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
                      "messages_sha256": O1._digest(sent), "result": deepcopy(result),
                      "review": review})
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
    return {"schema": "registry_action_options_dialogue_v2",
            "state_digest": state["state_digest"],
            "context_digest": O1._digest(context(state)), "calls": calls, "action": action,
            "final_review": deepcopy(final), "model_valid": final["status"] == "valid",
            "semantic_feedback_used": len(calls) == 2,
            "host_mapping": "v2_option_contract_only_no_old_response_reinterpretation",
            "claims": {"diagnosis_improvement": False, "recovery": False}}


__all__ = ["REVIEW_SCHEMA", "SCHEMA", "SYSTEM", "assess", "context", "dialogue",
           "messages", "select_model"]
