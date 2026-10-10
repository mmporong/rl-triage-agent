"""정상 기준 실행이 없는 단일 RL 학습 로그의 제한적 진단 도구."""
from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

from rl_triage import triage_tools as T


_SUMMARY_FIELDS = (
    "n",
    "first",
    "last",
    "min",
    "max",
    "mean_first_20pct",
    "mean_last_20pct",
)
_CONTEXT_KEYS = {"schema", "telemetry_sha256", "configuration_sha256", "facts"}
_FACT_KEYS = {"id", "path"}
_CLAIM_KEYS = {"id", "value"}
_ALLOWED_CONFIGURATION_ROOTS = {
    "task", "simulation", "rewards", "observations", "sensors", "scene", "actions",
    "terminations", "commands", "curriculum", "events",
}
_LOWER_SHA256 = re.compile(r"[0-9a-f]{64}")
_SAFE_FACT_ID = re.compile(r"[A-Za-z0-9_.:-]+")


def _validate_absent_reference() -> None:
    """명시적으로 비어 있는 참조만 받아 자기 비교를 정상 기준으로 오인하지 않게 한다."""
    reference = T._load_json(T._safe(T.WORKSPACE / "reference" / "telemetry.json"))
    if reference != {"reference_status": "absent", "summary": {}, "series": {}}:
        raise ValueError("single-run 진단에는 reference_status='absent'인 빈 reference만 허용됩니다")


def _case_dir(case_id: str) -> Path:
    if (not isinstance(case_id, str) or not case_id or case_id in (".", "..")
            or "/" in case_id or "\\" in case_id):
        raise PermissionError(f"case 경계를 벗어난 case_id: {case_id!r}")
    cases_root = T._safe(T.WORKSPACE / "cases").resolve()
    case_path = cases_root / case_id
    if case_path.is_symlink():
        raise PermissionError(f"case 디렉터리 심볼릭 링크는 허용되지 않습니다: {case_id!r}")
    case_dir = T._case_dir(case_id)
    if case_dir.parent != cases_root:
        raise PermissionError(f"case 경계를 벗어난 case_id: {case_id!r}")
    if not case_dir.is_dir():
        raise FileNotFoundError(f"없는 case_id: {case_id}")
    return case_dir


def _json_bytes(path: Path) -> tuple[bytes, object]:
    if path.is_symlink():
        raise PermissionError(f"심볼릭 링크 입력은 허용되지 않습니다: {path.name}")
    safe_path = T._safe(path)
    if not safe_path.is_file():
        raise FileNotFoundError(f"필수 입력 파일이 없습니다: {path.name}")
    raw = safe_path.read_bytes()

    def reject_constant(value: str):
        raise ValueError(f"유한하지 않은 JSON 숫자는 허용되지 않습니다: {value}")

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"중복 JSON 키는 허용되지 않습니다: {key}")
            result[key] = value
        return result

    try:
        parsed = json.loads(raw.decode("utf-8"), parse_constant=reject_constant, object_pairs_hook=unique_object)
    except UnicodeDecodeError as exc:
        raise ValueError(f"UTF-8 JSON이 아닙니다: {path.name}") from exc
    return raw, parsed


def _case_telemetry(case_id: str) -> dict:
    case_dir = _case_dir(case_id)
    _, telemetry = _json_bytes(case_dir / "telemetry.json")
    if not isinstance(telemetry, dict) or not isinstance(telemetry.get("summary"), dict) \
            or not isinstance(telemetry.get("series"), dict):
        raise ValueError("telemetry.json에는 summary와 series 객체가 필요합니다")
    return telemetry


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _validate_selected_value(value, label: str):
    def scalar(item) -> bool:
        return item is None or isinstance(item, (str, bool, int, float))

    values = value if isinstance(value, list) else [value]
    if not scalar(value) and not isinstance(value, list):
        raise ValueError(f"선택한 설정값은 JSON scalar 또는 flat scalar list여야 합니다: {label}")
    if isinstance(value, list) and any(not scalar(item) for item in value):
        raise ValueError(f"선택한 설정값은 JSON scalar 또는 flat scalar list여야 합니다: {label}")
    if any(isinstance(item, float) and not math.isfinite(item) for item in values):
        raise ValueError(f"선택한 설정값은 유한해야 합니다: {label}")
    return value


def _resolve_fact(configuration: object, path: list, fact_id: str):
    if not isinstance(path, list) or not path:
        raise ValueError(f"fact path는 비어 있지 않은 목록이어야 합니다: {fact_id}")
    if not isinstance(path[0], str) or path[0] not in _ALLOWED_CONFIGURATION_ROOTS:
        raise ValueError(f"허용되지 않은 configuration 경로입니다: {fact_id}")
    value = configuration
    for segment in path:
        if isinstance(value, dict):
            if not isinstance(segment, str) or segment not in value:
                raise ValueError(f"configuration에 없는 fact path입니다: {fact_id}")
            value = value[segment]
        elif isinstance(value, list):
            if type(segment) is not int or segment < 0 or segment >= len(value):
                raise ValueError(f"configuration에 없는 fact path입니다: {fact_id}")
            value = value[segment]
        else:
            raise ValueError(f"configuration에 없는 fact path입니다: {fact_id}")
    return _validate_selected_value(value, fact_id)


def get_context(case_id: str) -> dict:
    """선택된 설정 사실만 원본 해시와 함께 반환한다. 설정은 측정된 동작을 증명하지 않는다."""
    _validate_absent_reference()
    case_dir = _case_dir(case_id)
    telemetry_raw, telemetry = _json_bytes(case_dir / "telemetry.json")
    if not isinstance(telemetry, dict) or not isinstance(telemetry.get("summary"), dict) \
            or not isinstance(telemetry.get("series"), dict):
        raise ValueError("telemetry.json에는 summary와 series 객체가 필요합니다")

    context_path = case_dir / "context.json"
    if context_path.is_symlink():
        raise PermissionError("심볼릭 링크 입력은 허용되지 않습니다: context.json")
    if not context_path.exists():
        return {"case_id": case_id, "status": "not_provided", "facts": []}

    context_raw, manifest = _json_bytes(context_path)
    if not isinstance(manifest, dict) or set(manifest) != _CONTEXT_KEYS:
        raise ValueError("context.json 최상위 키가 single_run_context_v1 계약과 다릅니다")
    if manifest["schema"] != "single_run_context_v1":
        raise ValueError("지원하지 않는 context schema입니다")
    for key in ("telemetry_sha256", "configuration_sha256"):
        if not isinstance(manifest[key], str) or not _LOWER_SHA256.fullmatch(manifest[key]):
            raise ValueError(f"{key}는 64자리 소문자 SHA-256이어야 합니다")
    if manifest["telemetry_sha256"] != _sha256(telemetry_raw):
        raise ValueError("context의 telemetry_sha256이 현재 telemetry.json과 다릅니다")

    configuration_raw, configuration = _json_bytes(case_dir / "configuration.json")
    if manifest["configuration_sha256"] != _sha256(configuration_raw):
        raise ValueError("context의 configuration_sha256이 현재 configuration.json과 다릅니다")
    if not isinstance(configuration, dict):
        raise ValueError("configuration.json 최상위 값은 객체여야 합니다")
    if not isinstance(manifest["facts"], list):
        raise ValueError("context facts는 목록이어야 합니다")

    seen = set()
    facts = []
    for fact in manifest["facts"]:
        if not isinstance(fact, dict) or set(fact) != _FACT_KEYS:
            raise ValueError("각 context fact는 id와 path만 가져야 합니다")
        fact_id = fact["id"]
        if not isinstance(fact_id, str) or not _SAFE_FACT_ID.fullmatch(fact_id):
            raise ValueError("fact id는 비어 있지 않은 안전한 토큰이어야 합니다")
        if fact_id in seen:
            raise ValueError(f"중복 fact id입니다: {fact_id}")
        seen.add(fact_id)
        value = _resolve_fact(configuration, fact["path"], fact_id)
        facts.append({"id": fact_id, "path": fact["path"], "value": value, "kind": "configuration"})
    return {
        "case_id": case_id,
        "status": "provided",
        "context_sha256": _sha256(context_raw),
        "telemetry_sha256": _sha256(telemetry_raw),
        "configuration_sha256": _sha256(configuration_raw),
        "facts": facts,
    }


def telemetry_overview(case_id: str) -> dict:
    """단일 실행의 스칼라 요약을 반환한다. 정상 기준 비율이나 정상성 판정은 만들지 않는다."""
    _validate_absent_reference()
    summary = _case_telemetry(case_id)["summary"]
    rows = []
    for tag in sorted(summary):
        values = summary[tag]
        if not isinstance(values, dict):
            raise ValueError(f"summary 항목은 객체여야 합니다: {tag}")
        rows.append({"tag": tag, **{field: values.get(field) for field in _SUMMARY_FIELDS}})
    return {"case_id": case_id, "reference_status": "absent", "rows": rows}


def get_series(case_id: str, tag: str, points: int = 20) -> dict:
    """단일 실행 곡선 하나를 샘플링한다. 참조 곡선은 반환하지 않는다."""
    _validate_absent_reference()
    telemetry = _case_telemetry(case_id)
    series = telemetry["series"]
    if tag not in series:
        return {"error": f"없는 지표: {tag}"}
    result = T.get_series(case_id, tag, points=points)
    result.pop("reference", None)
    result["n"] = len(series[tag])
    return result


def run_analysis(code: str, case_id: str, timeout_s: int = 30) -> dict:
    """기존 격리 분석기를 사용한다. 분석 코드의 ``ref``는 항상 빈 객체다."""
    _validate_absent_reference()
    _case_telemetry(case_id)
    return T.run_analysis(code, case_id, timeout_s=timeout_s)


def _same_claim_value(actual, claimed) -> bool:
    if isinstance(actual, bool) or isinstance(claimed, bool):
        return type(actual) is type(claimed) and actual == claimed
    if isinstance(actual, (int, float)) and isinstance(claimed, (int, float)):
        actual_finite = not isinstance(actual, float) or math.isfinite(actual)
        claimed_finite = not isinstance(claimed, float) or math.isfinite(claimed)
        return actual_finite and claimed_finite and actual == claimed
    if isinstance(actual, list) and isinstance(claimed, list):
        return len(actual) == len(claimed) and all(
            _same_claim_value(left, right) for left, right in zip(actual, claimed))
    return type(actual) is type(claimed) and actual == claimed


def _checked_fact_claims(context: dict, fact_claims: list[dict] | None) -> list[dict]:
    if fact_claims is None:
        fact_claims = []
    if not isinstance(fact_claims, list):
        raise ValueError("fact_claims는 목록이어야 합니다")
    if context["status"] == "not_provided":
        if fact_claims:
            raise ValueError("context가 없으면 configuration fact를 인용할 수 없습니다")
        return []
    if context["facts"] and not fact_claims:
        raise ValueError("선택된 configuration fact가 있으면 하나 이상 인용해야 합니다")

    available = {fact["id"]: fact for fact in context["facts"]}
    seen = set()
    checked = []
    for claim in fact_claims:
        if not isinstance(claim, dict) or set(claim) != _CLAIM_KEYS:
            raise ValueError("각 fact claim은 id와 value만 가져야 합니다")
        fact_id = claim["id"]
        if not isinstance(fact_id, str) or fact_id in seen:
            raise ValueError(f"중복되거나 잘못된 fact claim id입니다: {fact_id!r}")
        if fact_id not in available:
            raise ValueError(f"context에 없는 fact id입니다: {fact_id}")
        seen.add(fact_id)
        _validate_selected_value(claim["value"], fact_id)
        source = available[fact_id]
        if not _same_claim_value(source["value"], claim["value"]):
            raise ValueError(f"configuration fact 값이 일치하지 않습니다: {fact_id}")
        checked.append({
            "id": fact_id,
            "path": source["path"],
            "value": source["value"],
            "kind": "configuration",
            "configuration_sha256": context["configuration_sha256"],
        })
    return checked


def write_assessment(case_id: str, hypotheses: list[str], evidence: str,
                     limitations: str, next_check: str,
                     fact_claims: list[dict] | None = None) -> dict:
    """인과 확정 없이 가설과 다음 계측 하나를 배타적으로 저장한다."""
    _validate_absent_reference()
    _case_telemetry(case_id)
    if not isinstance(hypotheses, list) or not hypotheses or any(
            not isinstance(item, str) or not item.strip() for item in hypotheses):
        return {"error": "hypotheses에는 비어 있지 않은 문자열 가설이 하나 이상 필요합니다"}
    fields = {"evidence": evidence, "limitations": limitations, "next_check": next_check}
    if any(not isinstance(value, str) or not value.strip() for value in fields.values()):
        return {"error": "evidence, limitations, next_check는 비어 있지 않은 문자열이어야 합니다"}

    try:
        context = get_context(case_id)
        checked_claims = _checked_fact_claims(context, fact_claims)
    except (FileNotFoundError, PermissionError, ValueError) as exc:
        return {"error": str(exc)}

    doc = {
        "case_id": case_id,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "hypotheses_only",
        "hypotheses": [item.strip() for item in hypotheses],
        "evidence": evidence.strip(),
        "limitations": limitations.strip(),
        "next_check": next_check.strip(),
        "context": {
            key: context[key] for key in (
                "status", "context_sha256", "telemetry_sha256", "configuration_sha256"
            ) if key in context
        },
        "verified_fact_claims": checked_claims,
    }
    out_dir = T._safe(T.WORKSPACE / "preregistrations" / "assessments")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = T._safe(out_dir / f"{case_id}.json")
    try:
        with out_path.open("x", encoding="utf-8") as output:
            json.dump(doc, output, ensure_ascii=False, indent=1)
    except FileExistsError:
        return {"error": f"기존 평가를 덮어쓸 수 없습니다: assessments/{case_id}.json"}
    return {"saved": f"preregistrations/assessments/{case_id}.json", **doc}
