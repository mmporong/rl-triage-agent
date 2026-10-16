"""B2 D3 기존 줄 번호 계약을 공통 Nemotron 비교 전송기에 연결하는 순수 어댑터."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "evals") not in sys.path:
    sys.path.insert(0, str(ROOT / "evals"))

import d3_agent as D3  # noqa: E402
import d3_model_call as D3_CALL  # noqa: E402

BOTTLENECK = "B2"
KIND = "d3"
SOURCE = ROOT / "bench/private/d3_experiment/ced4302df56bfc4e"
HISTORICAL_CALLS = ROOT / "bench/private/codex_support_20261015/task4/calls"
PACKET_IDS = ("C1_B4", "C1_B5", "C2_B4", "C2_B5", "C3_B4", "C3_B5", "D0_B4", "D0_B5")
TRUE_TOP1 = {"C1": "change_2", "C2": "change_7", "C3": "change_2", "D0": "change_1"}
SOURCE_FILES = {
    "evals/nemotron_bottleneck_d3_v2.py",
    "evals/d3_agent.py",
    "evals/d3_model_call.py",
    "tests/test_nemotron_bottleneck_d3_v2.py",
}


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def sha_raw(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def packet_path(packet_id: str) -> Path:
    if packet_id not in PACKET_IDS:
        raise ValueError(f"등록하지 않은 D3 packet: {packet_id}")
    case_id, stage = packet_id.split("_", 1)
    return SOURCE / case_id / "model" / f"{stage}.packet.json"


def historical_source_sha(packet_id: str) -> str:
    """동일 packet의 기존 old 3회 응답 원문 SHA 목록을 경로 없이 결합한다."""
    rows = []
    for path in sorted(HISTORICAL_CALLS.glob("c*_old_r*_a1.json"), key=lambda item: item.name):
        record = json.loads(path.read_text(encoding="utf-8"))
        entry = record.get("entry") or {}
        if entry.get("packet_id") == packet_id and entry.get("arm") == "old":
            rows.append({"repeat": entry.get("repeat"), "response_sha256_raw": sha_raw(path)})
    if sorted(row["repeat"] for row in rows) != [1, 2, 3]:
        raise ValueError(f"기존 old 응답 3회 결합이 다르다: {packet_id}")
    return canonical_sha(sorted(rows, key=lambda row: row["repeat"]))


def build_cases() -> list[dict[str, Any]]:
    """공통 전송기가 3회×2모델 schedule을 만들기 전의 unique 8개 입력."""
    rows = []
    for packet_id in PACKET_IDS:
        path = packet_path(packet_id)
        data = path.read_bytes()
        packet = json.loads(data)
        if packet.get("schema") != D3.SCHEMA:
            raise ValueError(f"D3 packet schema가 다르다: {packet_id}")
        case_id, stage = packet_id.split("_", 1)
        messages = D3.messages(packet)
        rows.append({
            "id": packet_id,
            "bottleneck": BOTTLENECK,
            "kind": KIND,
            "messages": messages,
            "request_params": deepcopy(D3_CALL.PARAMS),
            "binding": {
                "case_id": case_id,
                "stage": stage,
                "packet_sha256_raw": hashlib.sha256(data).hexdigest(),
                "messages_sha256": canonical_sha(messages),
                "truth_sha256": canonical_sha({"case_id": case_id, "top1_label": TRUE_TOP1[case_id]}),
                "historical_old_responses_sha256": historical_source_sha(packet_id),
                "contract": "d3_old_line_number_v1",
            },
            "packet": packet,
            "truth": {"top1_label": TRUE_TOP1[case_id]},
            "source_path": path,
        })
    if [row["id"] for row in rows] != list(PACKET_IDS):
        raise ValueError("D3 packet 순서가 고정 목록과 다르다")
    return rows


def public_case(case: dict[str, Any]) -> dict[str, Any]:
    """공개 프로토콜용 논리 결합. 원 packet·메시지·정답·비공개 경로는 제외한다."""
    return {key: deepcopy(case[key]) for key in ("id", "bottleneck", "kind", "request_params", "binding")}


def _ranking_if_permutation(output: Any, packet: dict[str, Any]) -> list[str] | None:
    if not isinstance(output, dict) or not isinstance(output.get("ranking"), list):
        return None
    ranking = output["ranking"]
    labels = [change["label"] for change in packet["changes"]]
    if len(ranking) != len(labels) or sorted(map(str, ranking)) != sorted(labels):
        return None
    return list(ranking)


def validate_response(case: dict[str, Any], content: str) -> dict[str, Any]:
    """원시 content를 기존 parse/validate로 판정하고 계약과 원인 순위를 따로 기록한다."""
    if case.get("bottleneck") != BOTTLENECK or case.get("kind") != KIND or case.get("id") not in PACKET_IDS:
        raise ValueError("B2 D3 case 형식이 다르다")
    packet = case.get("packet")
    truth = case.get("truth")
    if not isinstance(packet, dict) or not isinstance(truth, dict):
        raise ValueError("검증용 packet·정답 metadata가 없다")
    output = D3.parse(content)
    try:
        errors = D3.validate(output, packet)
    except (TypeError, ValueError) as error:
        errors = [f"validator_{type(error).__name__}"]
    ranking = _ranking_if_permutation(output, packet)
    top1 = ranking[0] if ranking else None
    citation_errors = [error for error in errors
                       if error == "reasons" or error == "top_reason" or error.startswith("reason")]
    return {
        "valid": not errors,
        "errors": errors,
        "decision_key": ranking,
        "top1": top1,
        "ranking_parsed": ranking is not None,
        "true_cause_top1": None if top1 is None else top1 == truth["top1_label"],
        "valid_true_cause_top1": None if errors else top1 == truth["top1_label"],
        "evidence_citation_errors": citation_errors,
        "evidence_citation_error": bool(citation_errors),
    }


def _historical_records() -> list[Path]:
    paths = sorted(HISTORICAL_CALLS.glob("c*_old_r*_a1.json"), key=lambda path: path.name)
    if len(paths) != 24:
        raise ValueError(f"기존 old 계약 응답은 24개여야 한다: {len(paths)}")
    return paths


def historical_summary() -> dict[str, Any]:
    """기존 모델 old arm 24개를 같은 어댑터 판정으로 재검증한다."""
    cases = {case["id"]: case for case in build_cases()}
    rows = []
    for path in _historical_records():
        record = json.loads(path.read_text(encoding="utf-8"))
        entry = record.get("entry")
        if (not isinstance(entry, dict) or entry.get("arm") != "old" or entry.get("packet_id") not in cases
                or entry.get("repeat") not in (1, 2, 3) or record.get("status") != "answered"):
            raise ValueError(f"기존 D3 응답 metadata가 다르다: {path.name}")
        judged = validate_response(cases[entry["packet_id"]], record.get("content"))
        rows.append({"packet_id": entry["packet_id"], "repeat": entry["repeat"], **judged,
                     "record_sha256_raw": sha_raw(path)})
    keys = {(row["packet_id"], row["repeat"]) for row in rows}
    if len(keys) != 24:
        raise ValueError("기존 D3 응답 논리 키가 중복되거나 빠졌다")
    errors = Counter(error for row in rows for error in row["errors"])
    invalid = sum(not row["valid"] for row in rows)
    return {
        "schema": "nemotron_bottleneck_d3_historical_v2",
        "model": "nvidia/nemotron-3-super-120b-a12b",
        "contract": "d3_old_line_number_v1",
        "responses": 24,
        "valid": 24 - invalid,
        "invalid": invalid,
        "evidence_citation_invalid": sum(row["evidence_citation_error"] for row in rows),
        "ranking_parsed": sum(row["ranking_parsed"] for row in rows),
        "true_cause_top1_all_parsed": sum(row["true_cause_top1"] is True for row in rows),
        "valid_true_cause_top1": sum(row["valid_true_cause_top1"] is True for row in rows),
        "error_counts": dict(sorted(errors.items())),
        "rows": rows,
    }
