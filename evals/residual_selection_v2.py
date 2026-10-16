"""남은 subtype에 대한 모델 선택과 등록 규칙 전환을 분리한다.

모델은 검사와 근거 fact ID만 선택한다. 검사 기대값, 실행 인자, 기존 진단
고리에 전달할 decision은 호스트가 등록 표에서 생성한다. 형식 또는 의미가
무효인 모델 선택만 등록 규칙으로 전환하며, packet 손상과 실행 불확실성은
전환하지 않는다.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "evals"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import residual_contract_v1 as C1


MODEL = C1.MODEL
M, R = C1.M, C1.R
SCHEMA = "residual_selection_packet_v2"
OUTPUT_FIELDS = {"action", "probe", "fact_ids"}
PACKET_FIELDS = {
    "schema", "epoch", "applied_corrections", "prior_receipt", "facts",
    "observed", "budget_left", "execution_budget_s", "remaining_candidates", "input_bindings",
}
PUBLIC_CATEGORIES = list(C1.L.MECHANISMS)


SYSTEM = """You select the next registered diagnostic probe for unresolved RL failure subtypes.
Return one JSON object with exactly action, probe, fact_ids.
action is continue or hold. For continue, probe is one registered probe ID. For hold, probe is null.
fact_ids is a nonempty list of distinct IDs copied from evidence_facts. Do not copy fact values.
Candidate IDs are subtype IDs. public_category is classification metadata, not another candidate.
The host owns expected outcomes, probe arguments, numeric fact values, execution, and approval.
Do not infer hidden case names, add fields, or claim a confirmed cause. Output JSON only."""

FEEDBACK = {
    "F_EMPTY": "Return one nonempty JSON object.",
    "F_JSON": "Return valid JSON with no prose or code fence.",
    "F_FIELDS": "Use exactly action, probe, and fact_ids.",
    "F_ACTION": "action must be continue or hold.",
    "F_PROBE": "continue requires a probe ID; hold requires null.",
    "F_FACT_IDS": "fact_ids must be a nonempty list of distinct strings.",
    "S_DECISION": "The decision must pass the three-field format contract.",
    "S_FACT_UNKNOWN": "Every fact_ids entry must exist in evidence_facts.",
    "S_PROBE_UNKNOWN": "For continue, select one eligible registered probe.",
    "S_PROBE_OBSERVED": "Do not select an already observed probe.",
    "S_PROBE_UNINFORMATIVE": ("Select a probe that partitions the remaining candidates, or the registered "
                              "abnormal confirmation probe for one remaining candidate."),
}


def _finite_scalar(value: object) -> bool:
    return (value is None or isinstance(value, (str, bool))
            or (isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(value)))


def _valid_sha(value: object) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _validate_epoch(epoch: object, applied: object, prior: object) -> list[str]:
    applied_copy = C1.validate_applied(applied)
    if type(epoch) is not int or epoch not in (1, 2) or len(applied_copy) != epoch - 1:
        raise ValueError("epoch와 누적 교정 수가 다르다")
    if epoch == 1 and prior is not None:
        raise ValueError("첫 epoch에는 앞선 receipt가 없다")
    if epoch == 2:
        if (not isinstance(prior, dict)
                or set(prior) != {"verdict", "correction_id", "result_sha256", "checkpoint_sha256"}
                or prior.get("verdict") != "not_recovered"
                or prior.get("correction_id") != applied_copy[0]
                or not _valid_sha(prior.get("result_sha256"))
                or not _valid_sha(prior.get("checkpoint_sha256"))):
            raise ValueError("두 번째 epoch에는 검증한 비회복 receipt가 필요하다")
    return applied_copy


def _validate_input_bindings(value: object) -> dict:
    roles = {"case", "reference"}
    fields = {"checkpoint_sha256", "telemetry_sha256"}
    if (not isinstance(value, dict) or set(value) != roles
            or any(not isinstance(value[role], dict) or set(value[role]) != fields
                   for role in roles)
            or any(not _valid_sha(value[role][field]) for role in roles for field in fields)):
        raise ValueError("case·reference 입력 SHA 결합이 고정 계약과 다르다")
    return deepcopy(value)


def build_packet(*, epoch: int, applied_corrections: list[str], prior_receipt: dict | None,
                 facts: dict[str, object], input_bindings: dict, budget_left: int = C1.MAX_PROBES,
                 execution_budget_s: int | float = 1200) -> dict:
    """호스트 입력에서 canonical v2 packet을 만든다."""
    packet = {
        "schema": SCHEMA,
        "epoch": epoch,
        "applied_corrections": list(applied_corrections),
        "prior_receipt": deepcopy(prior_receipt),
        "facts": deepcopy(facts),
        "input_bindings": _validate_input_bindings(input_bindings),
        "observed": {},
        "budget_left": budget_left,
        "execution_budget_s": execution_budget_s,
        "remaining_candidates": C1.unresolved_components(applied_corrections),
    }
    return validate_packet(packet)


def validate_packet(packet: dict) -> dict:
    """Packet 결합을 검증한다. 손상된 packet에는 규칙 전환을 허용하지 않는다."""
    if not isinstance(packet, dict) or set(packet) != PACKET_FIELDS or packet.get("schema") != SCHEMA:
        raise ValueError("selection packet 필드가 고정 계약과 다르다")
    applied = _validate_epoch(packet.get("epoch"), packet.get("applied_corrections"),
                              packet.get("prior_receipt"))
    facts = packet.get("facts")
    if (not isinstance(facts, dict) or not facts
            or any(not isinstance(key, str) or not key or not _finite_scalar(value)
                   for key, value in facts.items())):
        raise ValueError("facts는 비지 않은 유한 scalar 표여야 한다")
    _validate_input_bindings(packet.get("input_bindings"))
    if packet.get("observed") != {}:
        raise ValueError("각 epoch의 모델 선택 packet은 빈 관측에서 시작한다")
    budget = packet.get("budget_left")
    if type(budget) is not int or not 0 <= budget <= C1.MAX_PROBES:
        raise ValueError("검사 예산이 등록 범위를 벗어났다")
    seconds = packet.get("execution_budget_s")
    if (isinstance(seconds, bool) or not isinstance(seconds, (int, float))
            or not math.isfinite(seconds) or seconds < 0):
        raise ValueError("실행 예산은 유한한 음이 아닌 수여야 한다")
    expected = C1.unresolved_components(applied)
    if packet.get("remaining_candidates") != expected:
        raise ValueError("남은 후보가 누적 교정과 다르다")
    return deepcopy(packet)


def diagnosis_messages(packet: dict) -> list[dict]:
    value = validate_packet(packet)
    registry = C1.registry()
    candidate_groups = {
        category: [subtype for subtype in value["remaining_candidates"] if R.category(subtype) == category]
        for category in PUBLIC_CATEGORIES
    }
    candidate_groups = {category: subtypes for category, subtypes in candidate_groups.items() if subtypes}
    probes = [
        {"probe": probe, "measures": spec["measures"], "cost_gpu_s": spec["cost_gpu_s"]}
        for probe, spec in registry["probes"].items()
        if _probe_semantic(probe, value["remaining_candidates"], value["observed"])[0]
    ]
    body = {
        "epoch": value["epoch"],
        "applied_corrections": value["applied_corrections"],
        "prior_receipt": value["prior_receipt"],
        "evidence_facts": value["facts"],
        "candidate_groups": candidate_groups,
        "eligible_probes": probes,
        "budget_left": value["budget_left"],
        "execution_budget_s": value["execution_budget_s"],
    }
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": json.dumps(body, ensure_ascii=False, sort_keys=True)}]


def _review(status: str, codes: list[str], reasons: list[str], decision: dict | None = None) -> dict:
    result = {"status": status, "codes": codes, "reasons": reasons}
    if decision is not None:
        result["decision"] = deepcopy(decision)
    return result


def format_review(content: object, packet: dict) -> dict:
    validate_packet(packet)
    if not isinstance(content, str) or not content.strip():
        return _review("invalid", ["F_EMPTY"], ["모델 응답이 비어 있다"])
    try:
        decision = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return _review("invalid", ["F_JSON"], ["모델 응답은 JSON 객체 하나여야 한다"])
    if not isinstance(decision, dict):
        return _review("invalid", ["F_JSON"], ["모델 응답은 JSON 객체 하나여야 한다"])
    codes, reasons = [], []
    if set(decision) != OUTPUT_FIELDS:
        codes.append("F_FIELDS")
        reasons.append("모델 출력 필드가 고정 계약과 다르다")
    action, probe, fact_ids = decision.get("action"), decision.get("probe"), decision.get("fact_ids")
    if action not in ("continue", "hold"):
        codes.append("F_ACTION")
        reasons.append("action은 continue 또는 hold여야 한다")
    if ((action == "continue" and (not isinstance(probe, str) or not probe))
            or (action == "hold" and probe is not None)):
        codes.append("F_PROBE")
        reasons.append("action과 probe 형식이 맞지 않는다")
    if (not isinstance(fact_ids, list) or not fact_ids
            or any(not isinstance(item, str) or not item for item in fact_ids)
            or len(set(fact_ids)) != len(fact_ids)):
        codes.append("F_FACT_IDS")
        reasons.append("fact_ids는 비지 않은 중복 없는 문자열 목록이어야 한다")
    if codes:
        return _review("invalid", codes, reasons)
    return _review("valid", [], [], decision)


def feedback_message(review: dict) -> dict:
    """검증 오류를 packet별 정답 없이 한 번에 되돌려 준다."""
    codes = review.get("codes")
    if (review.get("status") != "invalid" or not isinstance(codes, list) or not codes
            or any(code not in FEEDBACK for code in codes)):
        raise ValueError("invalid 검증 결과에만 고정 피드백을 만들 수 있다")
    text = "\n".join(["The previous JSON was rejected.",
                       *(f"- {code}: {FEEDBACK[code]}" for code in codes),
                       "Return one corrected JSON object only."])
    return {"role": "user", "content": text}


def _probe_semantic(probe: str, hypotheses: list[str], observed: dict[str, str]) -> tuple[bool, str]:
    registry = C1.registry()
    if probe not in registry["probes"]:
        return False, "등록되지 않은 probe다"
    if probe in observed:
        return False, "이미 관측한 probe다"
    if not hypotheses or any(item not in registry["mechanisms"] for item in hypotheses):
        return False, "현재 가설 집합이 등록 범위와 다르다"
    if len(hypotheses) == 1:
        candidate = hypotheses[0]
        confirming = M._confirming(registry, candidate)
        if confirming == probe and registry["probes"][probe]["expect"][candidate] == "abnormal":
            return True, "단일 후보의 등록 확정 검사다"
        return False, "단일 후보의 등록 확정 검사가 아니다"
    outcomes = {registry["probes"][probe]["expect"][item] for item in hypotheses}
    if {"normal", "abnormal"} <= outcomes:
        return True, "현재 후보를 normal과 abnormal로 나누는 검사다"
    return False, "현재 후보를 나누지 못하는 검사다"


def semantic_review(decision: dict, packet: dict, *, hypotheses: list[str] | None = None,
                    observed: dict[str, str] | None = None) -> dict:
    value = validate_packet(packet)
    if not isinstance(decision, dict) or set(decision) != OUTPUT_FIELDS:
        return _review("invalid", ["S_DECISION"], ["형식 검증을 통과한 decision이 아니다"])
    codes, reasons = [], []
    unknown = [item for item in decision["fact_ids"] if item not in value["facts"]]
    if unknown:
        codes.append("S_FACT_UNKNOWN")
        reasons.append("packet에 없는 fact ID가 선택됐다")
    if decision["action"] == "hold":
        return (_review("invalid", codes, reasons) if codes else _review("valid", [], [], decision))
    candidates = list(value["remaining_candidates"] if hypotheses is None else hypotheses)
    seen = dict(value["observed"] if observed is None else observed)
    allowed, reason = _probe_semantic(decision["probe"], candidates, seen)
    if not allowed:
        code = "S_PROBE_UNKNOWN" if decision["probe"] not in C1.registry()["probes"] else "S_PROBE_UNINFORMATIVE"
        if decision["probe"] in seen:
            code = "S_PROBE_OBSERVED"
        codes.append(code)
        reasons.append(reason)
    if codes:
        return _review("invalid", codes, reasons)
    return _review("valid", [], [], decision)


def assess(content: object, packet: dict) -> dict:
    formatted = format_review(content, packet)
    if formatted["status"] != "valid":
        return {**formatted, "format_status": "invalid", "semantic_status": "not_run"}
    semantic = semantic_review(formatted["decision"], packet)
    return {**semantic, "format_status": "valid", "semantic_status": semantic["status"]}


def _host_decision(action: str, probe: str | None, fact_ids: list[str], packet: dict,
                   selected_by: str) -> dict:
    ranking = list(PUBLIC_CATEGORIES)
    facts = packet["facts"]
    claims = [{"id": key, "value": deepcopy(facts[key])} for key in fact_ids]
    expected = ({key: R.public_catalog()[probe]["expect"][key] for key in ranking[:3]}
                if action == "continue" and probe is not None else {})
    return {
        "ranking": ranking,
        "probe": probe,
        "action": action,
        "reason": f"{selected_by} 선택을 등록 표와 host 값으로 직렬화했다",
        "claims": claims,
        "expected_outcomes": expected,
    }


def select(review: dict, packet: dict) -> dict:
    """유효 모델 선택을 쓰거나 invalid 선택만 등록 규칙으로 전환한다."""
    value = validate_packet(packet)
    hypotheses = list(value["remaining_candidates"])
    common = {"candidate_count": len(hypotheses), "model_status": review.get("status"),
              "format_status": review.get("format_status"),
              "semantic_status": review.get("semantic_status"),
              "codes": list(review.get("codes", [])),
              "packet_digest": C1.L.digest(value),
              "registry_digest": M.registry_digest(C1.registry())}
    if value["budget_left"] == 0:
        return {**common, "selected_by": "hold", "reason": "empty_budget", "probe": None,
                "fact_ids": [], "host_decision": None}
    status = review.get("status")
    if status == "valid":
        decision = review.get("decision")
        try:
            frozen_decision = json.dumps(decision, ensure_ascii=False, sort_keys=True, allow_nan=False)
        except (TypeError, ValueError):
            formatted = _review("invalid", ["F_JSON"], ["저장된 decision을 JSON으로 재검증할 수 없다"])
        else:
            formatted = format_review(frozen_decision, value)
        semantic = (semantic_review(formatted["decision"], value)
                    if formatted["status"] == "valid" else formatted)
        if semantic["status"] != "valid":
            raise ValueError("저장된 valid 모델 판정이 재검증 결과와 다르다")
        decision = semantic["decision"]
        if semantic["status"] == "valid" and decision["action"] == "continue":
            host = _host_decision("continue", decision["probe"], decision["fact_ids"], value, "model")
            return {**common, "selected_by": "model", "reason": "model_valid", "probe": decision["probe"],
                    "fact_ids": list(decision["fact_ids"]), "host_decision": host}
        if semantic["status"] == "valid" and decision["action"] == "hold":
            return {**common, "selected_by": "hold", "reason": "model_hold", "probe": None,
                    "fact_ids": list(decision["fact_ids"]), "host_decision": None}
    invalid_model_choice = (status == "invalid"
                            and (review.get("format_status") == "invalid"
                                 or review.get("semantic_status") == "invalid"))
    if not invalid_model_choice:
        reason = "infra_or_unknown" if status in ("infra_error", "unknown") else "unusable_review"
        return {**common, "selected_by": "hold", "reason": reason, "probe": None,
                "fact_ids": [], "host_decision": None}
    probe = M._next(C1.registry(), hypotheses, value["observed"])
    if probe is None:
        return {**common, "selected_by": "hold", "reason": "no_registered_rule_probe", "probe": None,
                "fact_ids": [], "host_decision": None}
    fact_ids: list[str] = []
    allowed, _ = _probe_semantic(probe, hypotheses, value["observed"])
    if not allowed:
        return {**common, "selected_by": "hold", "reason": "registered_rule_probe_rejected",
                "probe": None, "fact_ids": [], "host_decision": None}
    host = _host_decision("continue", probe, fact_ids, value, "registered_rule")
    return {**common, "selected_by": "registered_rule", "reason": "model_invalid",
            "probe": probe, "fact_ids": fact_ids, "host_decision": host}


def initial_state(host_decision: dict, applied: list[str]) -> dict:
    """호스트가 만든 기존 계약 decision만 v1 고리에 전달한다."""
    return C1.initial_state(host_decision, applied)
