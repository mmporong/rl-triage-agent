"""의미 검증은 v1을 유지하고 등록 대상·근거 선택 규칙을 명시한 개발 계약."""
from __future__ import annotations

from copy import deepcopy

import agentic_investigation_contract_v1 as A1

C, V = A1.C, A1.V
prepare, checked, assess = A1.prepare, A1.checked, A1.assess
public_facts, probe_choice = A1.public_facts, A1.probe_choice
SYSTEM = """Investigate the user's RL training failure using only the supplied public facts and registered tools.
Return exactly one JSON object with three fields: action, target, fact_ids. No extra fields or prose.
Choose action from available_actions. For any action except hold, target MUST be one element of
available_actions[action]. NEVER choose an action whose target list is empty. target=null is allowed
ONLY for hold; hold requires target=null. A missing target is not a request to load evidence.
fact_ids MUST be a nonempty list of distinct IDs that exist in the CURRENT evidence_facts object,
including when choosing hold or requesting evidence. Cite facts already visible, not facts you hope
to retrieve. Do not return an empty fact_ids list or invent IDs.
Training facts already in evidence_facts are loaded and available now. Do not request them again.
If read_evidence has no targets, choose an eligible registered probe, an available frame target,
or hold using visible facts. Evidence retrieval is optional, not a mandatory first step.
read_evidence retrieves a listed training target; inspect_frames retrieves a listed frame target.
run_probe proposes a listed registered inspection. propose_correction requires a host-confirmed
component and a listed registered correction. Visual descriptions are interpretations, not confirmed
causes or recovery verdicts. The host supplies arguments, thresholds, approval, execution and evaluation.
Never generate commands, expected values, approval, new corrections, or a recovery verdict.
Budget exhaustion, unknown observations or insufficient evidence require hold. Output JSON only."""
STRUCTURED_OUTPUT_POLICY = {
    "type": "json_schema", "name": "agentic_investigation_action_v2",
    "fields": ["action", "target", "fact_ids"], "required": ["action", "target", "fact_ids"],
    "additionalProperties": False, "minItems": 1,
    "enums_from": "current_public_available_actions_targets_and_evidence_fact_ids",
    "action_target_dependency_and_unique_facts": "host_A1.assess",
    "hosted_keyword_enforcement": "not_claimed_without_measurement",
}


def messages(context: dict) -> list[dict]:
    """기존 공개 user facts를 보존하고 system 한 개만 v2로 고정한다."""
    value = deepcopy(A1.messages(context))
    value[0] = {"role": "system", "content": SYSTEM}
    return value


def response_format_from_view(view: dict) -> dict:
    """현재 공개 enum만 쓰는 JSON schema. action/target 종속·중복 근거는 호스트가 검증한다."""
    actions = sorted(action for action, targets in view["available_actions"].items() if targets)
    targets: list = []
    for action in actions:
        for target in view["available_actions"][action]:
            if target not in targets:
                targets.append(target)
    targets = sorted(target for target in targets if target is not None) + ([None] if None in targets else [])
    facts = sorted(view["evidence_facts"])
    if not facts or not actions or "hold" not in actions or None not in targets:
        raise ValueError("출력 schema를 만들 현재 공개 근거/행동이 부족하다")
    return {"type": "json_schema", "json_schema": {
        "name": STRUCTURED_OUTPUT_POLICY["name"],
        "schema": {"type": "object", "properties": {
            "action": {"type": "string", "enum": actions},
            "target": {"type": ["string", "null"], "enum": targets},
            "fact_ids": {"type": "array", "items": {"type": "string", "enum": facts}, "minItems": 1},
        }, "required": ["action", "target", "fact_ids"], "additionalProperties": False},
    }}


def response_format(context: dict) -> dict:
    return response_format_from_view(checked(context)["model_view"])
