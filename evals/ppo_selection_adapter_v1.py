"""PPO inverse-ratio 초기 진단의 모델 출력을 검사·근거 선택으로 제한한다.

모델은 ``action``, ``probe``, ``fact_ids``만 반환한다. 기존 여섯 범주 순위,
검사 기대값, claim 값과 이후 등록 검사 순서는 호스트가 동결 등록 표에서 만든다.
이 파일의 fixture 검증은 CPU 계약 개발이며 검사·회복 성능 측정이 아니다.
"""
from __future__ import annotations

import json
import sys
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _directory in (ROOT / "src", ROOT / "evals"):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

import model_decision_journal_v1 as JOURNAL
import ppo_ratio_repair_contract_v2 as C
import selection_comparison_call_v2 as CALL

MODEL = "nvidia/nemotron-3.5-super-vl-120b-a12b"
ACCEPTED_PROVIDER_MODELS = CALL.ACCEPTED_PROVIDER_MODELS
SNAPSHOT_SCHEMA = "ppo_selection_snapshot_v1"
OUTPUT_FIELDS = {"action", "probe", "fact_ids"}
MAX_CALLS = 3

SYSTEM = """You select the first registered diagnostic probe for an unresolved RL failure.
Return one JSON object with exactly action, probe, fact_ids.
action is continue or hold. For continue, probe is one eligible probe ID. For hold, probe is null.
fact_ids is a nonempty list of distinct IDs copied from evidence_facts. Do not copy fact values.
The host owns category order, expected outcomes, probe arguments, numeric fact values, and later probes.
Do not add fields, infer hidden case names, or claim a confirmed cause. Output JSON only."""

FEEDBACK = {
    "F_EMPTY": "Return one nonempty JSON object.",
    "F_JSON": "Return valid JSON with no prose or code fence.",
    "F_FIELDS": "Use exactly action, probe, and fact_ids.",
    "F_ACTION": "action must be continue or hold.",
    "F_PROBE": "continue requires one eligible probe ID; hold requires null.",
    "F_FACT_IDS": "fact_ids must be a nonempty list of distinct available evidence fact IDs.",
    "S_FACT_UNKNOWN": "Every fact_ids entry must exist in evidence_facts.",
    "S_PROBE_UNKNOWN": "For continue, select one ID from eligible_probes.",
}


def _source_packet(snapshot: dict) -> dict:
    return {
        "schema": snapshot["source_schema"],
        "facts": deepcopy(snapshot["evidence_facts"]),
        "observed": [],
        "budget_left": snapshot["budget_left"],
        "execution_budget_s": snapshot["execution_budget_s"],
    }


def _public_subtypes() -> dict[str, list[str]]:
    grouped = {name: [] for name in C.L.MECHANISMS}
    for subtype in C.registry()["mechanisms"]:
        grouped[C.category(subtype)].append(subtype)
    return grouped


def _eligible_probes() -> list[dict]:
    return [
        {"probe": name, "measures": spec["measures"], "cost_gpu_s": spec["cost_gpu_s"]}
        for name, spec in C.registry()["probes"].items()
    ]


def build_snapshot(packet: dict) -> dict:
    """기존 초기 packet을 검증하고 정답·실행 인자를 뺀 모델 입력 view를 만든다."""
    C.diagnosis_messages(packet)
    snapshot = {
        "schema": SNAPSHOT_SCHEMA,
        "source_schema": packet["schema"],
        "evidence_facts": deepcopy(packet["facts"]),
        "subtype_registry": _public_subtypes(),
        "eligible_probes": _eligible_probes(),
        "budget_left": packet["budget_left"],
        "execution_budget_s": packet["execution_budget_s"],
        "source_digest": C.L.digest(packet),
        "registry_digest": C.M.registry_digest(C.registry()),
    }
    _validate_snapshot(snapshot)
    return snapshot


def _validate_snapshot(snapshot: dict) -> dict:
    fields = {
        "schema", "source_schema", "evidence_facts", "subtype_registry", "eligible_probes",
        "budget_left", "execution_budget_s", "source_digest", "registry_digest",
    }
    if not isinstance(snapshot, dict) or set(snapshot) != fields or snapshot.get("schema") != SNAPSHOT_SCHEMA:
        raise ValueError("snapshot 필드가 고정 계약과 다르다")
    packet = _source_packet(snapshot)
    C.diagnosis_messages(packet)
    if snapshot["source_digest"] != C.L.digest(packet):
        raise ValueError("source packet digest가 다르다")
    if snapshot["registry_digest"] != C.M.registry_digest(C.registry()):
        raise ValueError("등록 표 digest가 다르다")
    if snapshot["subtype_registry"] != _public_subtypes() or snapshot["eligible_probes"] != _eligible_probes():
        raise ValueError("공개 subtype 또는 probe view가 등록 표와 다르다")
    return deepcopy(snapshot)


def messages(snapshot: dict) -> list[dict]:
    value = _validate_snapshot(snapshot)
    body = {
        "evidence_facts": value["evidence_facts"],
        "subtype_registry": value["subtype_registry"],
        "eligible_probes": value["eligible_probes"],
        "budget_left": value["budget_left"],
        "execution_budget_s": value["execution_budget_s"],
    }
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": json.dumps(body, ensure_ascii=False, sort_keys=True)},
    ]


def _choice_error(code: str) -> ValueError:
    return ValueError(code)


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    value = {}
    for key, item in pairs:
        if key in value:
            raise _choice_error("F_FIELDS")
        value[key] = item
    return value


def validate_choice(snapshot: dict, content: object) -> dict:
    value = _validate_snapshot(snapshot)
    if not isinstance(content, str) or not content.strip():
        raise _choice_error("F_EMPTY")
    try:
        choice = json.loads(content, object_pairs_hook=_unique_object)
    except (json.JSONDecodeError, TypeError) as error:
        raise _choice_error("F_JSON") from error
    if not isinstance(choice, dict):
        raise _choice_error("F_JSON")
    if set(choice) != OUTPUT_FIELDS:
        raise _choice_error("F_FIELDS")
    action, probe, fact_ids = choice["action"], choice["probe"], choice["fact_ids"]
    if action not in ("continue", "hold"):
        raise _choice_error("F_ACTION")
    probe_ids = {row["probe"] for row in value["eligible_probes"]}
    if action == "continue":
        if not isinstance(probe, str):
            raise _choice_error("F_PROBE")
        if probe not in probe_ids:
            raise _choice_error("S_PROBE_UNKNOWN")
    elif probe is not None:
        raise _choice_error("F_PROBE")
    if (not isinstance(fact_ids, list) or not fact_ids
            or any(not isinstance(item, str) or not item for item in fact_ids)
            or len(set(fact_ids)) != len(fact_ids)):
        raise _choice_error("F_FACT_IDS")
    if any(item not in value["evidence_facts"] for item in fact_ids):
        raise _choice_error("S_FACT_UNKNOWN")
    return deepcopy(choice)


def host_diagnosis(snapshot: dict, choice: dict) -> dict:
    """세 필드 선택을 기존 여섯 필드 계약으로 직렬화하고 생성 주체를 따로 기록한다."""
    value = _validate_snapshot(snapshot)
    checked = validate_choice(value, json.dumps(choice, ensure_ascii=False, allow_nan=False))
    ranking = list(C.L.MECHANISMS)
    probe = checked["probe"]
    expected = ({name: C.public_catalog()[probe]["expect"][name] for name in ranking[:3]}
                if checked["action"] == "continue" else {})
    decision = {
        "ranking": ranking,
        "probe": probe,
        "action": checked["action"],
        "reason": "호스트가 등록 순위·기대값·근거 값을 생성했다",
        "claims": [
            {"id": fact_id, "value": deepcopy(value["evidence_facts"][fact_id])}
            for fact_id in checked["fact_ids"]
        ],
        "expected_outcomes": expected,
    }
    validated = C.validate_diagnosis(decision, _source_packet(value))
    return {
        "decision": validated,
        "metadata": {
            "selection_fields_generated_by": "model",
            "ranking_generated_by": "host_registry",
            "expected_outcomes_generated_by": "host_registry",
            "claim_values_generated_by": "host_packet",
            "reason_generated_by": "host_template",
        },
    }


def _feedback(code: str) -> dict:
    if code not in FEEDBACK:
        raise ValueError("등록되지 않은 검증 코드")
    text = "\n".join((
        "The previous JSON was rejected.",
        f"- {code}: {FEEDBACK[code]}",
        "Return one corrected JSON object only.",
    ))
    return {"role": "user", "content": text}


def _safe_complete(complete: Callable[[list[dict], dict], dict], sent: list[dict], params: dict) -> dict:
    try:
        raw = complete(deepcopy(sent), deepcopy(params))
    except JOURNAL.JournalError:
        raise
    except Exception as error:  # noqa: BLE001 - provider 경계에서는 예외 class만 보존한다.
        raw = {"error_type": type(error).__name__}
    return CALL.safe_result(raw)


def _transport(result: dict) -> str:
    status = CALL.transport(result)
    if status == "unknown" and CALL.response_status(result) == "timeout":
        return "infra_error"
    return status


def engine(snapshot: dict, complete: Callable[[list[dict], dict], dict]) -> dict:
    """인프라 재시도 1회와 의미 피드백 1회를 포함한 최대 세 호출 고리."""
    value = _validate_snapshot(snapshot)
    base = messages(value)
    sent = base
    params = CALL.F.request_params(MODEL)
    calls: list[dict] = []
    infra_retried = False
    feedback_used = False
    while len(calls) < MAX_CALLS:
        result = _safe_complete(complete, sent, params)
        status = _transport(result)
        row = {"attempt": len(calls) + 1, "transport": status, "result": result}
        calls.append(row)
        if status == "infra_error" and not infra_retried and len(calls) < MAX_CALLS:
            infra_retried = True
            continue
        if status != "completed":
            return {"status": "terminal_hold", "reason": f"transport_{status}", "selection": None,
                    "host_decision": None, "calls": calls}
        try:
            choice = validate_choice(value, result.get("content"))
        except ValueError as error:
            code = str(error)
            row["validation"] = {"status": "invalid", "code": code}
            if not feedback_used and code in FEEDBACK and len(calls) < MAX_CALLS:
                feedback_used = True
                sent = [*base, {"role": "assistant", "content": result.get("content")}, _feedback(code)]
                continue
            return {"status": "terminal_hold", "reason": "selection_invalid", "selection": None,
                    "host_decision": None, "calls": calls}
        row["validation"] = {"status": "valid"}
        if choice["action"] == "hold":
            return {"status": "terminal_hold", "reason": "model_hold", "selection": choice,
                    "host_decision": None, "calls": calls}
        hosted = host_diagnosis(value, choice)
        return {"status": "choice_ready", "reason": "model_choice_valid", "selection": choice,
                "host_decision": hosted["decision"], "host_metadata": hosted["metadata"], "calls": calls}
    return {"status": "terminal_hold", "reason": "call_budget_exhausted", "selection": None,
            "host_decision": None, "calls": calls}
