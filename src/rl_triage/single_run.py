"""정상 기준 실행이 없는 단일 RL 학습 로그의 제한적 진단 도구."""
from __future__ import annotations

import json
from datetime import datetime, timezone

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


def _validate_absent_reference() -> None:
    """명시적으로 비어 있는 참조만 받아 자기 비교를 정상 기준으로 오인하지 않게 한다."""
    reference = T._load_json(T._safe(T.WORKSPACE / "reference" / "telemetry.json"))
    if reference != {"reference_status": "absent", "summary": {}, "series": {}}:
        raise ValueError("single-run 진단에는 reference_status='absent'인 빈 reference만 허용됩니다")


def _case_telemetry(case_id: str) -> dict:
    if (not isinstance(case_id, str) or not case_id or case_id in (".", "..")
            or "/" in case_id or "\\" in case_id):
        raise PermissionError(f"case 경계를 벗어난 case_id: {case_id!r}")
    case_dir = T._case_dir(case_id)
    cases_root = T._safe(T.WORKSPACE / "cases")
    if case_dir.parent != cases_root.resolve():
        raise PermissionError(f"case 경계를 벗어난 case_id: {case_id!r}")
    if not case_dir.is_dir():
        raise FileNotFoundError(f"없는 case_id: {case_id}")
    telemetry = T._load_json(case_dir / "telemetry.json")
    if not isinstance(telemetry.get("summary"), dict) or not isinstance(telemetry.get("series"), dict):
        raise ValueError("telemetry.json에는 summary와 series 객체가 필요합니다")
    return telemetry


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


def write_assessment(case_id: str, hypotheses: list[str], evidence: str,
                     limitations: str, next_check: str) -> dict:
    """인과 확정 없이 가설과 다음 계측 하나를 배타적으로 저장한다."""
    _validate_absent_reference()
    _case_telemetry(case_id)
    if not isinstance(hypotheses, list) or not hypotheses or any(
            not isinstance(item, str) or not item.strip() for item in hypotheses):
        return {"error": "hypotheses에는 비어 있지 않은 문자열 가설이 하나 이상 필요합니다"}
    fields = {"evidence": evidence, "limitations": limitations, "next_check": next_check}
    if any(not isinstance(value, str) or not value.strip() for value in fields.values()):
        return {"error": "evidence, limitations, next_check는 비어 있지 않은 문자열이어야 합니다"}

    doc = {
        "case_id": case_id,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "hypotheses_only",
        "hypotheses": [item.strip() for item in hypotheses],
        "evidence": evidence.strip(),
        "limitations": limitations.strip(),
        "next_check": next_check.strip(),
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
