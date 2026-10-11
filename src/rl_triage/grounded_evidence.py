"""해시로 묶인 단일 실행 근거와 구조화된 가설 전제 검증."""
from __future__ import annotations

import hashlib
import json
import math
import re
from copy import deepcopy
from datetime import datetime, timezone

from rl_triage import single_run as S
from rl_triage import triage_tools as T


SUMMARY_FIELDS = (
    "n", "first", "last", "min", "max", "mean_first_20pct", "mean_last_20pct",
)
RELATIONS = {"eq", "ne", "lt", "lte", "gt", "gte"}
HYPOTHESIS_KEYS = {"id", "rank", "narrative", "premises"}
PREMISE_KEYS = {"evidence_id", "relation", "operand"}
OPERAND_KEYS = {"value", "unit"}
NEXT_CHECK_KEYS = {"registry_id", "hypothesis_predictions"}
HYPOTHESIS_PREDICTION_KEYS = {"hypothesis_id", "predictions"}
PREDICTION_KEYS = {"measurement_id", "relation", "operand"}
SAFE_ID = re.compile(r"[A-Za-z0-9_.:-]+")

READ_ONLY_CHECKS = {
    "runtime_com_state": {
        "safety": "read_only", "adapter_id": "host.runtime_com_state.v1",
        "measurements": {
            "com_position_z": {"unit": "simulation_length_native", "value_type": "number"},
            "com_velocity_z": {"unit": "simulation_velocity_native", "value_type": "number"},
            "body_com_local_position_max_abs": {"unit": "simulation_length_native", "value_type": "number"},
        },
    },
    "runtime_reward_window": {
        "safety": "read_only", "adapter_id": "host.runtime_reward_window.v1",
        "measurements": {
            "reward_mean": {"unit": "reward_native", "value_type": "number"},
            "reward_min": {"unit": "reward_native", "value_type": "number"},
        },
    },
    "runtime_contact_window": {
        "safety": "read_only", "adapter_id": "host.runtime_contact_window.v1",
        "measurements": {
            "contact_force_max": {"unit": "simulation_force_native", "value_type": "number"},
            "contact_event_fraction": {"unit": "fraction", "value_type": "number"},
        },
    },
}


def _canonical_sha(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _value_type(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "list"
    raise ValueError(f"지원하지 않는 근거 값 형식입니다: {type(value).__name__}")


def _finite_number(value: object, label: str, *, integer: bool = False) -> None:
    valid = type(value) is int if integer else type(value) in (int, float)
    if not valid or (isinstance(value, float) and not math.isfinite(value)):
        expected = "정수" if integer else "유한한 숫자"
        raise ValueError(f"{label}은 {expected}여야 합니다")


def _validate_full_telemetry(full: dict) -> None:
    summary, series = full["summary"], full["series"]
    for tag, row in summary.items():
        if not isinstance(tag, str) or not isinstance(row, dict):
            raise ValueError("telemetry summary는 문자열 tag와 객체 행으로 구성되어야 합니다")
        for field in SUMMARY_FIELDS:
            if field not in row or row[field] is None:
                continue
            _finite_number(row[field], f"summary {tag}.{field}", integer=field == "n")
    for tag, values in series.items():
        if not isinstance(tag, str) or not isinstance(values, list):
            raise ValueError("telemetry series는 문자열 tag와 숫자 목록으로 구성되어야 합니다")
        for index, value in enumerate(values):
            _finite_number(value, f"series {tag}[{index}]")


def evidence_catalog(case_id: str) -> dict:
    """선택된 설정 사실과 전체 telemetry 원본에 묶인 작은 근거 카탈로그를 만든다."""
    full = S.read_full_telemetry(case_id)
    _validate_full_telemetry(full)
    context = S.get_context(case_id)
    source_sha = full["telemetry_sha256"]
    if context["status"] == "provided" and context["telemetry_sha256"] != source_sha:
        raise ValueError("context와 catalog의 telemetry_sha256이 다릅니다")
    evidence = []
    tags = sorted(set(full["summary"]) | set(full["series"]))
    for tag in tags:
        row = full["summary"].get(tag)
        if row is not None and not isinstance(row, dict):
            raise ValueError(f"summary 항목은 객체여야 합니다: {tag}")
        for field in SUMMARY_FIELDS:
            present = isinstance(row, dict) and field in row
            value = row[field] if present else None
            state = "missing" if not present else ("null" if value is None else "present")
            evidence.append({
                "evidence_id": f"telemetry.summary::{tag}::{field}",
                "kind": "telemetry_summary", "state": state, "value": value,
                "value_type": "count" if field == "n" else ("number" if state == "present" else "unknown"),
                "unit": "count" if field == "n" else "telemetry_native",
                "source": {"telemetry_sha256": source_sha, "tag": tag, "field": field, "derivation": "original_summary"},
            })
    for fact in context["facts"]:
        value = fact["value"]
        evidence.append({
            "evidence_id": f"configuration::{fact['id']}", "kind": "configuration",
            "state": "null" if value is None else "present", "value": value,
            "value_type": _value_type(value), "unit": "configuration_value",
            "source": {
                "configuration_sha256": context["configuration_sha256"],
                "context_sha256": context["context_sha256"], "path": fact["path"],
            },
        })
    body = {
        "case_id": case_id, "telemetry_sha256": source_sha,
        "context_status": context["status"], "evidence": evidence,
        "read_only_checks": deepcopy(READ_ONLY_CHECKS),
    }
    body["catalog_sha256"] = _canonical_sha(body)
    return body


def _validate_operand(operand: object, expected_unit: str, expected_type: str, label: str) -> object:
    if not isinstance(operand, dict) or set(operand) != OPERAND_KEYS:
        raise ValueError(f"{label} operand는 value와 unit만 가져야 합니다")
    if operand["unit"] != expected_unit:
        raise ValueError(f"{label} operand 단위가 근거와 다릅니다")
    value = operand["value"]
    actual_type = _value_type(value)
    if expected_type == "list":
        S._validate_selected_value(value, label)
    if expected_type in ("number", "count"):
        _finite_number(value, f"{label} operand", integer=expected_type == "count")
        actual_type = expected_type
    if actual_type != expected_type:
        raise ValueError(f"{label} operand 타입이 근거와 다릅니다")
    return value


def _relation_holds(actual: object, relation: str, operand: object) -> bool:
    if relation == "eq":
        return S._same_claim_value(actual, operand)
    if relation == "ne":
        return not _relation_holds(actual, "eq", operand)
    if type(actual) not in (int, float) or type(operand) not in (int, float):
        raise ValueError(f"{relation} 관계는 숫자에만 사용할 수 있습니다")
    return {"lt": actual < operand, "lte": actual <= operand,
            "gt": actual > operand, "gte": actual >= operand}[relation]


def _checked_hypotheses(catalog: dict, hypotheses: object) -> tuple[list[dict], list[str]]:
    if not isinstance(hypotheses, list) or len(hypotheses) < 2:
        raise ValueError("구조화된 가설이 두 개 이상 필요합니다")
    available = {item["evidence_id"]: item for item in catalog["evidence"]}
    seen_ids, seen_ranks, checked, unavailable = set(), set(), [], []
    for hypothesis in hypotheses:
        if not isinstance(hypothesis, dict) or set(hypothesis) != HYPOTHESIS_KEYS:
            raise ValueError("각 hypothesis는 id, rank, narrative, premises만 가져야 합니다")
        hypothesis_id, rank = hypothesis["id"], hypothesis["rank"]
        if not isinstance(hypothesis_id, str) or not SAFE_ID.fullmatch(hypothesis_id) or hypothesis_id in seen_ids:
            raise ValueError(f"중복되거나 잘못된 hypothesis id입니다: {hypothesis_id!r}")
        if type(rank) is not int or rank <= 0 or rank in seen_ranks:
            raise ValueError(f"중복되거나 잘못된 hypothesis rank입니다: {rank!r}")
        if not isinstance(hypothesis["narrative"], str) or not hypothesis["narrative"].strip():
            raise ValueError("hypothesis narrative는 비어 있지 않은 문자열이어야 합니다")
        premises = hypothesis["premises"]
        if not isinstance(premises, list) or not premises:
            raise ValueError("각 hypothesis에는 premise가 하나 이상 필요합니다")
        seen_ids.add(hypothesis_id)
        seen_ranks.add(rank)
        checked_premises = []
        seen_evidence = set()
        for premise in premises:
            if not isinstance(premise, dict) or set(premise) != PREMISE_KEYS:
                raise ValueError("각 premise는 evidence_id, relation, operand만 가져야 합니다")
            evidence_id, relation = premise["evidence_id"], premise["relation"]
            if not isinstance(evidence_id, str) or not isinstance(relation, str):
                raise ValueError("premise evidence_id와 relation은 문자열이어야 합니다")
            if evidence_id in seen_evidence:
                raise ValueError(f"중복 premise evidence id입니다: {evidence_id}")
            if evidence_id not in available:
                raise ValueError(f"catalog에 없는 evidence id입니다: {evidence_id}")
            seen_evidence.add(evidence_id)
            if relation not in RELATIONS:
                raise ValueError(f"허용되지 않은 relation입니다: {relation}")
            source = available[evidence_id]
            if source["state"] != "present":
                unavailable.append(evidence_id)
                continue
            operand = _validate_operand(premise["operand"], source["unit"], source["value_type"], evidence_id)
            if not _relation_holds(source["value"], relation, operand):
                raise ValueError(f"근거 관계가 참이 아닙니다: {evidence_id} {relation}")
            checked_premises.append({**premise, "actual": source["value"], "premise_verified": True})
        checked.append({
            "id": hypothesis_id, "rank": rank, "narrative": hypothesis["narrative"].strip(),
            "premises": checked_premises, "premises_verified": len(checked_premises) == len(premises),
            "model_narrative_unverified": True,
        })
    if seen_ranks != set(range(1, len(hypotheses) + 1)):
        raise ValueError("hypothesis rank는 1부터 연속되어야 합니다")
    return sorted(checked, key=lambda item: item["rank"]), sorted(set(unavailable))


def _exclusive_predictions(left: dict, right: dict) -> bool:
    left_relation, right_relation = left["relation"], right["relation"]
    left_value = left["operand"]["value"]
    right_value = right["operand"]["value"]
    if left_relation == "ne" or right_relation == "ne":
        return (
            left_relation == "eq" and right_relation == "ne" and left_value == right_value
        ) or (
            left_relation == "ne" and right_relation == "eq" and left_value == right_value
        )

    def interval(relation: str, value: int | float):
        if relation == "eq":
            return (value, True, value, True)
        if relation == "lt":
            return (None, False, value, False)
        if relation == "lte":
            return (None, False, value, True)
        if relation == "gt":
            return (value, False, None, False)
        return (value, True, None, False)

    left_lower, left_lower_inclusive, left_upper, left_upper_inclusive = interval(left_relation, left_value)
    right_lower, right_lower_inclusive, right_upper, right_upper_inclusive = interval(right_relation, right_value)
    lower, lower_inclusive = (left_lower, left_lower_inclusive)
    if lower is None or (right_lower is not None and right_lower > lower):
        lower, lower_inclusive = right_lower, right_lower_inclusive
    elif right_lower == lower and lower is not None:
        lower_inclusive = left_lower_inclusive and right_lower_inclusive
    upper, upper_inclusive = (left_upper, left_upper_inclusive)
    if upper is None or (right_upper is not None and right_upper < upper):
        upper, upper_inclusive = right_upper, right_upper_inclusive
    elif right_upper == upper and upper is not None:
        upper_inclusive = left_upper_inclusive and right_upper_inclusive
    return lower is not None and upper is not None and (
        lower > upper or (lower == upper and not (lower_inclusive and upper_inclusive))
    )


def _checked_next_check(next_check: object, hypothesis_ids: set[str]) -> dict:
    if not isinstance(next_check, dict) or set(next_check) != NEXT_CHECK_KEYS:
        raise ValueError("next_check는 registry_id와 hypothesis_predictions만 가져야 합니다")
    registry_id = next_check["registry_id"]
    if not isinstance(registry_id, str) or registry_id not in READ_ONLY_CHECKS:
        raise ValueError(f"등록되지 않은 read-only next check입니다: {registry_id}")
    predictions = next_check["hypothesis_predictions"]
    if not isinstance(predictions, list) or len(predictions) < 2:
        raise ValueError("next_check에는 두 가설 이상의 사전 prediction이 필요합니다")
    registry = READ_ONLY_CHECKS[registry_id]
    measurements = registry["measurements"]
    seen_hypotheses, checked = set(), []
    prediction_map: dict[str, dict[str, dict]] = {}
    for item in predictions:
        if not isinstance(item, dict) or set(item) != HYPOTHESIS_PREDICTION_KEYS:
            raise ValueError("hypothesis prediction은 hypothesis_id와 predictions만 가져야 합니다")
        hypothesis_id = item["hypothesis_id"]
        if not isinstance(hypothesis_id, str) or hypothesis_id not in hypothesis_ids or hypothesis_id in seen_hypotheses:
            raise ValueError(f"중복되거나 없는 hypothesis id입니다: {hypothesis_id}")
        rows = item["predictions"]
        if (not isinstance(rows, list) or len(rows) != len(measurements)
                or any(not isinstance(row, dict) or set(row) != PREDICTION_KEYS for row in rows)):
            raise ValueError("각 가설은 registry의 measurement를 정확히 한 번씩 예측해야 합니다")
        measurement_ids = [row["measurement_id"] for row in rows]
        if any(not isinstance(measurement_id, str) for measurement_id in measurement_ids) \
                or set(measurement_ids) != set(measurements):
            raise ValueError("각 가설은 registry의 measurement를 정확히 한 번씩 예측해야 합니다")
        seen_hypotheses.add(hypothesis_id)
        checked_rows = []
        prediction_map[hypothesis_id] = {}
        for row in rows:
            if not isinstance(row["relation"], str) or row["relation"] not in RELATIONS:
                raise ValueError("각 prediction은 measurement_id, relation, operand 계약을 따라야 합니다")
            measurement_id = row["measurement_id"]
            measurement = measurements[measurement_id]
            value = _validate_operand(row["operand"], measurement["unit"], measurement["value_type"], measurement_id)
            prediction_map[hypothesis_id][measurement_id] = row
            checked_rows.append({**row, "prediction_unverified": True})
        checked.append({"hypothesis_id": hypothesis_id, "predictions": checked_rows})
    selected_ids = sorted(prediction_map)
    for index, left_id in enumerate(selected_ids):
        for right_id in selected_ids[index + 1:]:
            if not any(_exclusive_predictions(
                    prediction_map[left_id][measurement_id], prediction_map[right_id][measurement_id]
            ) for measurement_id in measurements):
                raise ValueError(f"가설 {left_id}와 {right_id}의 prediction이 서로 배타적이지 않습니다")
    return {
        "registry_id": registry_id, "safety": registry["safety"],
        "adapter_id": registry["adapter_id"],
        "registered_not_executed": True, "execution_status": "approval_pending",
        "measurement_status": "not_observed", "assessment_status": "hold",
        "hypothesis_predictions": checked,
        "prediction_unverified": True,
    }


def write_grounded_assessment(case_id: str, catalog_sha256: str,
                              hypotheses: object, next_check: object) -> dict:
    """값 관계만 검증한 가설과 실행 전 read-only check 등록을 배타적으로 저장한다."""
    try:
        catalog = evidence_catalog(case_id)
        if catalog_sha256 != catalog["catalog_sha256"]:
            raise ValueError("catalog_sha256이 현재 근거와 다릅니다")
        checked_hypotheses, unavailable = _checked_hypotheses(catalog, hypotheses)
        checked_next = _checked_next_check(next_check, {item["id"] for item in checked_hypotheses})
        if unavailable:
            return {"status": "hold", "reason": "evidence_unavailable", "unavailable_evidence": unavailable}
    except (FileNotFoundError, PermissionError, ValueError) as exc:
        return {"error": str(exc)}
    doc = {
        "case_id": case_id, "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "hypotheses_only", "catalog_sha256": catalog["catalog_sha256"],
        "telemetry_sha256": catalog["telemetry_sha256"],
        "context_status": catalog["context_status"], "premises_verified": True,
        "model_narrative_unverified": True, "hypotheses": checked_hypotheses,
        "next_check": checked_next,
    }
    out_dir = T._safe(T.WORKSPACE / "preregistrations" / "grounded_assessments")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = T._safe(out_dir / f"{case_id}.json")
    try:
        with out_path.open("x", encoding="utf-8") as output:
            json.dump(doc, output, ensure_ascii=False, indent=1)
    except FileExistsError:
        return {"error": f"기존 평가를 덮어쓸 수 없습니다: grounded_assessments/{case_id}.json"}
    return {"saved": f"preregistrations/grounded_assessments/{case_id}.json", **doc}
