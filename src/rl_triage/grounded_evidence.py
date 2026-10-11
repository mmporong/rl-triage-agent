"""해시로 묶인 단일 실행 근거와 구조화된 가설 전제 검증."""
from __future__ import annotations

import hashlib
import json
import math
import re
import secrets
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
GROUP_INDEX_PAYLOAD_BYTE_CAP = 65_536
GROUP_PAGE_LIMIT = 64
DETAIL_ID_LIMIT = 16
SELECTION_BINDING_KEYS = {
    "selection_id", "case_id", "catalog_sha256", "selected_evidence_ids", "receipt_sha256",
}
SELECTION_ID = re.compile(r"sel_[0-9a-f]{32}")

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
        "context_status": context["status"],
        "configuration_sha256": context.get("configuration_sha256"),
        "context_sha256": context.get("context_sha256"),
        "evidence": evidence,
        "read_only_checks": deepcopy(READ_ONLY_CHECKS),
    }
    body["catalog_sha256"] = _canonical_sha(body)
    return body


def _source_bindings(catalog: dict) -> dict:
    configuration = {
        (item["source"]["configuration_sha256"], item["source"]["context_sha256"])
        for item in catalog["evidence"] if item["kind"] == "configuration"
    }
    if len(configuration) > 1:
        raise ValueError("configuration 근거의 source binding이 일치하지 않습니다")
    bindings = {
        "telemetry": {"telemetry_sha256": catalog["telemetry_sha256"]},
        "configuration": {
            "status": catalog["context_status"],
            "configuration_sha256": catalog["configuration_sha256"],
            "context_sha256": catalog["context_sha256"],
        },
    }
    if configuration:
        configuration_sha256, context_sha256 = next(iter(configuration))
        if (configuration_sha256, context_sha256) != (
                catalog["configuration_sha256"], catalog["context_sha256"]):
            raise ValueError("configuration 근거의 source binding이 catalog와 일치하지 않습니다")
    return bindings


def _index_row(item: dict) -> dict:
    source = item["source"]
    if item["kind"] == "telemetry_summary":
        path = [source["tag"], source["field"]]
        label = source["field"]
        source_ref = "telemetry"
    elif item["kind"] == "configuration":
        path = deepcopy(source["path"])
        label = item["evidence_id"].split("::", 1)[1]
        source_ref = "configuration"
    else:
        raise ValueError(f"지원하지 않는 근거 kind입니다: {item['kind']!r}")
    return {
        "evidence_id": item["evidence_id"], "kind": item["kind"],
        "label": label, "path": path, "state": item["state"],
        "value_type": item["value_type"], "unit": item["unit"],
        "source_ref": source_ref,
    }


def _group_prefix(item: dict) -> list:
    source = item["source"]
    if item["kind"] == "telemetry_summary":
        return [source["tag"]]
    if item["kind"] == "configuration":
        return deepcopy(source["path"][:3])
    raise ValueError(f"지원하지 않는 근거 kind입니다: {item['kind']!r}")


def _catalog_groups(catalog: dict) -> dict[str, dict]:
    grouped: dict[str, dict] = {}
    seen_evidence_ids = set()
    for item in catalog["evidence"]:
        evidence_id = item["evidence_id"]
        if evidence_id in seen_evidence_ids:
            raise ValueError(f"catalog에 중복 evidence id가 있습니다: {evidence_id}")
        seen_evidence_ids.add(evidence_id)
        prefix = _group_prefix(item)
        identity = {"kind": item["kind"], "path_prefix": prefix}
        group_id = f"grp_{_canonical_sha(identity)[:16]}"
        existing = grouped.setdefault(group_id, {**identity, "items": []})
        if existing["kind"] != item["kind"] or existing["path_prefix"] != prefix:
            raise ValueError(f"catalog group id 충돌입니다: {group_id}")
        existing["items"].append(item)
    for group in grouped.values():
        group["items"].sort(key=lambda item: item["evidence_id"])
    return dict(sorted(grouped.items()))


def _group_index_payload(catalog: dict) -> tuple[dict, dict[str, dict]]:
    groups = _catalog_groups(catalog)
    rows = []
    for group_id, group in groups.items():
        evidence_ids = [item["evidence_id"] for item in group["items"]]
        rows.append({
            "group_id": group_id, "kind": group["kind"],
            "label": str(group["path_prefix"][-1]),
            "path_prefix": deepcopy(group["path_prefix"]),
            "evidence_count": len(evidence_ids),
            "evidence_id_range": {"first": evidence_ids[0], "last": evidence_ids[-1]},
        })
    payload = {
        "case_id": catalog["case_id"], "catalog_sha256": catalog["catalog_sha256"],
        "source_bindings": _source_bindings(catalog),
        "total_evidence_count": len(catalog["evidence"]),
        "group_count": len(rows), "groups": rows,
        "read_only_checks": deepcopy(catalog["read_only_checks"]),
        "values_included": False,
    }
    return payload, groups


def evidence_catalog_group_index(case_id: str) -> dict:
    """값을 싣지 않은 결정적 경로 그룹 인덱스를 크기 제한 안에서 반환한다."""
    catalog = evidence_catalog(case_id)
    payload, _ = _group_index_payload(catalog)
    payload_bytes = len(json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
    ).encode("utf-8"))
    if payload_bytes > GROUP_INDEX_PAYLOAD_BYTE_CAP:
        raise ValueError(
            f"group index payload가 {GROUP_INDEX_PAYLOAD_BYTE_CAP} byte 제한을 넘었습니다: {payload_bytes}"
        )
    return {
        **payload, "group_index_sha256": _canonical_sha(payload),
        "payload_utf8_bytes": payload_bytes,
        "payload_byte_cap": GROUP_INDEX_PAYLOAD_BYTE_CAP,
        "coverage_status": "all_evidence_grouped_values_not_loaded",
        "comprehensive_diagnosis_supported": False,
    }


def _bounded_integer(value: object, label: str, *, minimum: int, maximum: int) -> int:
    if type(value) is not int or value < minimum or value > maximum:
        raise ValueError(f"{label}은 {minimum} 이상 {maximum} 이하 정수여야 합니다")
    return value


def _current_group(case_id: str, catalog_sha256: str, group_id: str) -> tuple[dict, dict, dict[str, dict]]:
    catalog = evidence_catalog(case_id)
    if not isinstance(catalog_sha256, str) or catalog_sha256 != catalog["catalog_sha256"]:
        raise ValueError("catalog_sha256이 현재 근거와 다릅니다")
    if not isinstance(group_id, str):
        raise ValueError("group_id는 문자열이어야 합니다")
    payload, groups = _group_index_payload(catalog)
    if group_id not in groups:
        raise ValueError(f"catalog에 없는 group id입니다: {group_id}")
    return catalog, payload, groups


def evidence_catalog_group_page(case_id: str, catalog_sha256: str, group_id: str,
                                offset: int = 0, limit: int = GROUP_PAGE_LIMIT) -> dict:
    """선택한 중립 경로 그룹에서 값 없는 근거 ID 페이지를 반환한다."""
    catalog, payload, groups = _current_group(case_id, catalog_sha256, group_id)
    group = groups[group_id]
    group_total = len(group["items"])
    clean_limit = _bounded_integer(limit, "limit", minimum=1, maximum=GROUP_PAGE_LIMIT)
    clean_offset = _bounded_integer(offset, "offset", minimum=0, maximum=group_total - 1)
    end = min(clean_offset + clean_limit, group_total)
    rows = [_index_row(item) for item in group["items"][clean_offset:end]]
    next_offset = end if end < group_total else None
    selected_complete = clean_offset == 0 and end == group_total
    return {
        "case_id": case_id, "catalog_sha256": catalog["catalog_sha256"],
        "group_index_sha256": _canonical_sha(payload),
        "source_bindings": _source_bindings(catalog),
        "group_id": group_id, "kind": group["kind"],
        "path_prefix": deepcopy(group["path_prefix"]),
        "group_evidence_count": group_total,
        "offset": clean_offset, "limit": clean_limit, "count": len(rows),
        "next_offset": next_offset,
        "unvisited_before_count": clean_offset,
        "unvisited_after_count": group_total - end,
        "unvisited_group_count": len(groups) - 1,
        "coverage_status": "complete_selected_group" if selected_complete else "partial_selected_group",
        "catalog_coverage_status": "partial",
        "comprehensive_diagnosis_supported": False,
        "values_included": False, "evidence": rows,
    }


def evidence_catalog_details(case_id: str, catalog_sha256: str, evidence_ids: object) -> dict:
    """선택한 소수 근거 ID의 현재 원문 행을 전체 catalog 해시에 묶어 반환한다."""
    catalog = evidence_catalog(case_id)
    if not isinstance(catalog_sha256, str) or catalog_sha256 != catalog["catalog_sha256"]:
        raise ValueError("catalog_sha256이 현재 근거와 다릅니다")
    if not isinstance(evidence_ids, list) or not evidence_ids:
        raise ValueError("evidence_ids는 비어 있지 않은 목록이어야 합니다")
    if len(evidence_ids) > DETAIL_ID_LIMIT:
        raise ValueError(f"evidence_ids는 최대 {DETAIL_ID_LIMIT}개까지 허용됩니다")
    if any(not isinstance(evidence_id, str) for evidence_id in evidence_ids):
        raise ValueError("모든 evidence id는 문자열이어야 합니다")
    if len(set(evidence_ids)) != len(evidence_ids):
        raise ValueError("중복 evidence id는 허용되지 않습니다")
    available = {item["evidence_id"]: item for item in catalog["evidence"]}
    unknown = sorted(set(evidence_ids) - set(available))
    if unknown:
        raise ValueError(f"catalog에 없는 evidence id입니다: {unknown[0]}")
    selected_ids = sorted(evidence_ids)
    selected = [deepcopy(available[evidence_id]) for evidence_id in selected_ids]
    selection_id = f"sel_{secrets.token_hex(16)}"
    receipt = {
        "schema": "grounded_evidence_selection_v1",
        "selection_id": selection_id,
        "case_id": case_id,
        "catalog_sha256": catalog["catalog_sha256"],
        "source_bindings": _source_bindings(catalog),
        "selected_evidence_ids": selected_ids,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    receipt_sha256 = _canonical_sha(receipt)
    out_dir = T._safe(T.WORKSPACE / "preregistrations" / "evidence_selections")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = T._safe(out_dir / f"{case_id}__{selection_id}.json")
    try:
        with out_path.open("x", encoding="utf-8") as output:
            json.dump(receipt, output, ensure_ascii=False, indent=1)
    except FileExistsError as exc:
        raise ValueError("새 evidence selection id를 만들지 못했습니다") from exc
    selection_binding = {
        "selection_id": selection_id,
        "case_id": case_id,
        "catalog_sha256": catalog["catalog_sha256"],
        "selected_evidence_ids": selected_ids,
        "receipt_sha256": receipt_sha256,
    }
    return {
        "case_id": case_id, "catalog_sha256": catalog["catalog_sha256"],
        "source_bindings": _source_bindings(catalog),
        "selection_binding": selection_binding,
        "selected_evidence_count": len(selected),
        "unselected_evidence_count": len(catalog["evidence"]) - len(selected),
        "catalog_coverage_status": "selected_evidence_only",
        "comprehensive_diagnosis_supported": False,
        "evidence": selected,
    }


def _validated_selection(case_id: str, catalog_sha256: str,
                         selection_binding: object) -> tuple[dict, list[str]]:
    if not isinstance(selection_binding, dict) or set(selection_binding) != SELECTION_BINDING_KEYS:
        raise ValueError(
            "selection_binding은 selection_id, case_id, catalog_sha256, "
            "selected_evidence_ids, receipt_sha256만 가져야 합니다"
        )
    selection_id = selection_binding["selection_id"]
    if not isinstance(selection_id, str) or not SELECTION_ID.fullmatch(selection_id):
        raise ValueError("selection_id가 host 발급 형식과 다릅니다")
    if selection_binding["case_id"] != case_id or selection_binding["catalog_sha256"] != catalog_sha256:
        raise ValueError("selection_binding의 case 또는 catalog가 요청과 다릅니다")
    selected_ids = selection_binding["selected_evidence_ids"]
    if (not isinstance(selected_ids, list) or not selected_ids or len(selected_ids) > DETAIL_ID_LIMIT
            or any(not isinstance(item, str) for item in selected_ids)
            or selected_ids != sorted(set(selected_ids))):
        raise ValueError("selection_binding의 evidence id 목록이 잘못되었습니다")
    receipt_sha256 = selection_binding["receipt_sha256"]
    if not isinstance(receipt_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", receipt_sha256):
        raise ValueError("selection receipt SHA-256 형식이 잘못되었습니다")

    catalog = evidence_catalog(case_id)
    if catalog_sha256 != catalog["catalog_sha256"]:
        raise ValueError("catalog_sha256이 현재 근거와 다릅니다")
    receipt_dir = T._safe(T.WORKSPACE / "preregistrations" / "evidence_selections")
    receipt_path = T._safe(receipt_dir / f"{case_id}__{selection_id}.json")
    if receipt_path.parent != receipt_dir or receipt_path.is_symlink() or not receipt_path.is_file():
        raise ValueError("host가 발급한 evidence selection receipt가 없습니다")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("evidence selection receipt를 읽을 수 없습니다") from exc
    if not isinstance(receipt, dict) or _canonical_sha(receipt) != receipt_sha256:
        raise ValueError("evidence selection receipt SHA-256이 다릅니다")
    expected = {
        "schema": "grounded_evidence_selection_v1",
        "selection_id": selection_id,
        "case_id": case_id,
        "catalog_sha256": catalog_sha256,
        "source_bindings": _source_bindings(catalog),
        "selected_evidence_ids": selected_ids,
    }
    if any(receipt.get(key) != value for key, value in expected.items()) \
            or set(receipt) != {*expected, "created_at"} \
            or not isinstance(receipt.get("created_at"), str):
        raise ValueError("evidence selection receipt 내용이 현재 요청과 다릅니다")
    return catalog, selected_ids


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


def _checked_hypotheses(catalog: dict, hypotheses: object,
                        selected_evidence_ids: set[str] | None = None) -> tuple[list[dict], list[str]]:
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
            if selected_evidence_ids is not None and evidence_id not in selected_evidence_ids:
                raise ValueError(f"selection에 포함되지 않은 premise evidence id입니다: {evidence_id}")
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


def _checked_next_check(next_check: object, hypothesis_ids: set[str],
                        registry: dict | None = None) -> dict:
    registry = READ_ONLY_CHECKS if registry is None else registry
    if not isinstance(next_check, dict) or set(next_check) != NEXT_CHECK_KEYS:
        raise ValueError("next_check는 registry_id와 hypothesis_predictions만 가져야 합니다")
    registry_id = next_check["registry_id"]
    if not isinstance(registry_id, str) or registry_id not in registry:
        raise ValueError(f"등록되지 않은 read-only next check입니다: {registry_id}")
    predictions = next_check["hypothesis_predictions"]
    if not isinstance(predictions, list) or len(predictions) < 2:
        raise ValueError("next_check에는 두 가설 이상의 사전 prediction이 필요합니다")
    check_definition = registry[registry_id]
    measurements = check_definition["measurements"]
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
        "registry_id": registry_id, "safety": check_definition["safety"],
        "adapter_id": check_definition["adapter_id"],
        "registered_not_executed": True, "execution_status": "approval_pending",
        "measurement_status": "not_observed", "assessment_status": "hold",
        "hypothesis_predictions": checked,
        "prediction_unverified": True,
    }


def write_grounded_assessment(case_id: str, catalog_sha256: str,
                              selection_binding: object,
                              hypotheses: object, next_check: object) -> dict:
    """값 관계만 검증한 가설과 실행 전 read-only check 등록을 배타적으로 저장한다."""
    try:
        catalog, selected_ids = _validated_selection(case_id, catalog_sha256, selection_binding)
        checked_hypotheses, unavailable = _checked_hypotheses(
            catalog, hypotheses, set(selected_ids),
        )
        checked_next = _checked_next_check(next_check, {item["id"] for item in checked_hypotheses})
        if unavailable:
            return {"status": "hold", "reason": "evidence_unavailable", "unavailable_evidence": unavailable}
    except (FileNotFoundError, PermissionError, ValueError) as exc:
        return {"error": str(exc)}
    doc = {
        "case_id": case_id, "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "hypotheses_only", "catalog_sha256": catalog["catalog_sha256"],
        "telemetry_sha256": catalog["telemetry_sha256"],
        "selection_binding": deepcopy(selection_binding),
        "selected_evidence_ids": selected_ids,
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
