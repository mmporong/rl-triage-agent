"""실제 중첩 fact 경로를 열거하는 계단 인용 계약. 과제·학습·판정은 v1과 같다."""
from __future__ import annotations

import stairs_agent_repair_v1 as V1

MODEL, ACCEPTED_MODELS = V1.MODEL, V1.ACCEPTED_MODELS
PROBES, ADAPT, TASK, RECIPE, ACCEPTANCE = V1.PROBES, V1.ADAPT, V1.TASK, V1.RECIPE, V1.ACCEPTANCE
sha, write_new, assess, probe, digest = V1.sha, V1.write_new, V1.assess, V1.probe, V1.digest


def _paths(value: dict, prefix: str = "") -> list[str]:
    found = []
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else key
        found.append(path)
        if isinstance(item, dict):
            found.extend(_paths(item, path))
    return found


def packet(source: dict, receipts: list[dict]) -> dict:
    result = V1.packet(source, receipts)
    result["schema"] = "stairs_agent_decision_packet_v2"
    result["citable_fact_ids"] = sorted(_paths(result["facts"]))
    result["citation_contract"] = "열거된 실제 fact 경로만 인용한다. 없는 경로는 거부한다."
    return result


def _validated(value: dict, context: dict) -> dict:
    if not isinstance(value, dict) or not isinstance(value.get("cited_facts"), list):
        raise TypeError("JSON 객체와 cited_facts 목록이 필요하다")
    cited = value["cited_facts"]
    if any(not isinstance(c, str) or c not in context["citable_fact_ids"] for c in cited):
        raise ValueError("열거된 실제 fact 경로가 아니다")
    if len(set(cited)) != len(cited):
        raise ValueError("중복 인용")
    normalized = {**value, "cited_facts": list(dict.fromkeys(c.split(".")[0] for c in cited))}
    V1.validate_decision(normalized, context)
    return normalized


def validate_decision(value: dict, context: dict) -> dict:
    _validated(value, context)
    return value


def messages(context: dict) -> list[dict]:
    result = V1.messages(context)
    result[0]["content"] += (
        " The exact allowed citation IDs are listed in citable_fact_ids. "
        "For example source.observed_symptom is allowed only when listed. "
        "The task-integrity check may already have been run by the host as an execution prerequisite. "
        "Select a remaining eligible check; after its receipt, choose adaptation or HOLD. "
        "For adaptation cite evidence from BOTH P_stairs_task_integrity and P_stairs_stability."
    )
    return result


def authorize_adaptation(source: dict, receipts: list[dict], decision: dict,
                         registered_operator_approval: bool) -> dict:
    normalized = _validated(decision, packet(source, receipts))
    value = V1.authorize_adaptation(source, receipts, normalized, registered_operator_approval)
    value["decision_digest"] = digest(decision)
    value["citation_contract"] = "actual_fact_paths_v2"
    return value
