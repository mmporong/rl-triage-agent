"""Blind quadruped localization followed by a deterministic host consistency gate."""
from __future__ import annotations

import argparse
import base64
import copy
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evals import feedback_model_call as FEEDBACK
from evals import visual_integrity_experiment_v1 as V1

MODEL = V1.MODEL
REQUEST_PARAMS = copy.deepcopy(V1.REQUEST_PARAMS)
SOURCE_FILES = V1.SOURCE_FILES + (
    "evals/visual_localization_experiment_v2.py",
    "tests/test_visual_localization_experiment_v2.py",
)
SYSTEM = (
    "Locate only the actual quadruped body in the provided image. Ignore colored arrows and markers. "
    "Return exactly one JSON object containing only frame_id, visibility, and bbox_1000. "
    "frame_id must equal the provided frame ID. "
    "visibility must be visible, absent, or unclear. For visible, bbox_1000 is required as four integer values "
    "[xmin,ymin,xmax,ymax] normalized to 0..1000, with xmin<xmax and ymin<ymax. For absent or unclear, "
    "bbox_1000 must be null. Do not decide consistency, fault, cause, or recovery."
)

_HEX = re.compile(r"[0-9a-f]{64}")
_GIT = re.compile(r"[0-9a-f]{40}")
_FRAME_KEYS = {"frame_id", "visibility", "bbox_1000"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()  # noqa: UP017 - Python 3.10 contract


def _read(path: Path) -> Any:
    return V1._read(path)


def _write_new(path: Path, value: Any) -> None:
    V1._write_new(path, value)


def _sha(path: Path) -> str:
    return V1._sha(path)


def _digest(value: Any) -> str:
    return V1._digest(value)


def messages_for_case(case_dir: Path, frame_id: str) -> list[dict[str, Any]]:
    if frame_id not in {"f03", "f06"}:
        raise ValueError("frame_id는 f03 또는 f06이어야 한다")
    case_dir = Path(case_dir).resolve()
    packet = V1.validate_packet(case_dir)
    index = V1.FRAME_IDS.index(frame_id)
    sample, frame = packet["samples"][index], packet["frames"][index]
    content: list[dict[str, Any]] = [{
        "type": "text",
        "text": json.dumps({
            "frame_id": frame_id, "timestamp_s": sample["timestamp_s"], "image_size": [640, 480],
            "bbox_coordinate_system": "normalized_0_1000",
        }, separators=(",", ":")),
    }]
    image_path = V1._inside(case_dir, V1._relative(frame["file"], "frame.file"))
    content.append({
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64," + base64.b64encode(image_path.read_bytes()).decode("ascii")},
    })
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]


def validate_localization(content: Any) -> dict[str, Any]:
    format_reasons: list[str] = []
    semantic_reasons: list[str] = []
    value: Any = None
    if not isinstance(content, str):
        format_reasons.append("content_not_string")
    else:
        try:
            value = json.loads(content)
        except (json.JSONDecodeError, ValueError):
            format_reasons.append("json_parse_error")
    if not format_reasons and (not isinstance(value, dict) or set(value) != _FRAME_KEYS):
        format_reasons.append("response_shape_invalid")
    if not format_reasons:
        if not isinstance(value["frame_id"], str) or value["frame_id"] not in {"f03", "f06"}:
            semantic_reasons.append("frame_id_invalid")
        visibility = value["visibility"]
        if not isinstance(visibility, str) or visibility not in {"visible", "absent", "unclear"}:
            format_reasons.append("visibility_invalid")
        else:
            bbox = value["bbox_1000"]
            if visibility == "visible":
                if (not isinstance(bbox, list) or len(bbox) != 4
                        or any(isinstance(item, bool) or not isinstance(item, int) for item in bbox)):
                    format_reasons.append("visible_bbox_invalid")
                elif any(item < 0 or item > 1000 for item in bbox):
                    semantic_reasons.append("bbox_out_of_range")
                elif bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
                    semantic_reasons.append("bbox_reversed")
            elif bbox is not None:
                semantic_reasons.append("nonvisible_bbox_must_be_null")
    valid = not format_reasons and not semantic_reasons
    return {
        "valid": valid,
        "format_reasons": format_reasons,
        "semantic_reasons": semantic_reasons,
        "localization": value if valid else None,
    }


def derive_host_decision(packet: dict[str, Any], localization: dict[str, Any]) -> dict[str, Any]:
    frame_id = localization["frame_id"]
    sample = packet["samples"][V1.FRAME_IDS.index(frame_id)]
    expected_x, expected_y = (float(value) for value in sample["expected_root_pixel"])
    visibility = localization["visibility"]
    distance: float | None = None
    if visibility == "visible":
        bbox = localization["bbox_1000"]
        center_x = ((bbox[0] + bbox[2]) / 2) * 640 / 1000
        center_y = ((bbox[1] + bbox[3]) / 2) * 480 / 1000
        distance = math.hypot(center_x - expected_x, center_y - expected_y)
        if distance <= 30:
            status, gate, category = "consistent", "reviewed_consistent", "near"
        elif distance > 60:
            status, gate, category = "inconsistent", "quarantine_visual_evidence", "mismatch"
        else:
            status, gate, category = "unclear", "HOLD_visual_evidence", "abstain_margin"
    elif visibility == "absent" and 0 <= expected_x < 640 and 0 <= expected_y < 480:
        status, gate, category = "inconsistent", "quarantine_visual_evidence", "mismatch_absent"
    else:
        status, gate, category = "unclear", "HOLD_visual_evidence", "abstain"
    return {
        "schema": "visual_localization_host_decision_v2", "status": status, "gate": gate,
        "frame_id": frame_id, "visibility": visibility, "center_error_px": distance, "host_category": category,
        "thresholds_px": {"near_max_inclusive": 30, "far_min_exclusive": 60},
        "decision_maker": "deterministic_host_geometry_v2",
    }


def _validate_protocol_shape(protocol: Any) -> dict[str, Any]:
    if not isinstance(protocol, dict):
        raise TypeError("protocol은 object여야 한다")
    required = {
        "schema", "status", "source_anchor", "source_sha256_lf", "catalog_sha256", "labels_sha256",
        "model", "request_params", "budget", "scope", "image_detail",
    }
    if not required.issubset(protocol):
        raise ValueError("protocol 필수 필드가 없다")
    if (protocol["schema"] != "visual_localization_experiment_protocol_v2"
            or protocol["status"] != "registered" or protocol["model"] != MODEL
            or protocol["request_params"] != REQUEST_PARAMS
            or protocol["budget"] != {"cases": 4, "timestamps_s": [3, 6], "request_cap": 8,
                                      "automatic_retries": 0}
            or protocol["scope"] != "development_blind_visual_localization_only"
            or protocol["image_detail"] is not None):
        raise ValueError("protocol 고정 조건이 다르다")
    if not _GIT.fullmatch(str(protocol["source_anchor"])):
        raise ValueError("source_anchor 형식이 다르다")
    if set(protocol["source_sha256_lf"]) != set(SOURCE_FILES):
        raise ValueError("source map 범위가 다르다")
    if any(not _HEX.fullmatch(str(value)) for value in protocol["source_sha256_lf"].values()):
        raise ValueError("source hash 형식이 다르다")
    if not _HEX.fullmatch(str(protocol["catalog_sha256"])) or not _HEX.fullmatch(str(protocol["labels_sha256"])):
        raise ValueError("catalog 또는 labels hash 형식이 다르다")
    encoded = json.dumps(protocol, ensure_ascii=False)
    if any(token in encoded for token in (
        "C:\\Users\\", "C:/Users/", "\\Users\\", "/home/", "/Users/", "/mnt/", "bench/private"
    )):
        raise ValueError("공개 protocol에 개인 경로가 있다")
    return protocol


def _verify_source_guard(protocol: dict[str, Any]) -> None:
    anchor = protocol["source_anchor"]
    subprocess.run(["git", "merge-base", "--is-ancestor", anchor, "HEAD"], cwd=ROOT, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    for relative in SOURCE_FILES:
        expected = protocol["source_sha256_lf"][relative]
        current = V1._inside(ROOT, Path(relative)).read_bytes()
        if V1._lf_sha_bytes(current) != expected:
            raise ValueError(f"현재 source hash가 다르다: {relative}")
        anchored = V1._git("show", f"{anchor}:{relative}")
        if V1._lf_sha_bytes(anchored) != expected:
            raise ValueError(f"anchor source blob이 다르다: {relative}")


def _load_protocol(path: Path, expected_sha: str | None = None) -> dict[str, Any]:
    path = Path(path).resolve()
    if expected_sha is not None and (_sha(path) != expected_sha or not _HEX.fullmatch(expected_sha)):
        raise ValueError("protocol SHA가 다르다")
    protocol = _validate_protocol_shape(_read(path))
    _verify_source_guard(protocol)
    return protocol


def _expected_inputs(run_dir: Path, protocol: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    catalog_path = run_dir / "cases" / "catalog.json"
    if _sha(catalog_path) != protocol["catalog_sha256"]:
        raise ValueError("run catalog SHA가 protocol과 다르다")
    catalog = V1.validate_catalog(catalog_path)
    cases, slots = [], []
    for row in catalog["cases"]:
        cases.append({"case_id": row["case_id"], "packet_sha256": row["packet_sha256"]})
        for frame_id, timestamp_s in (("f03", 3), ("f06", 6)):
            messages_sha = _digest(messages_for_case(run_dir / "cases" / row["case_id"], frame_id))
            slots.append({
                "slot_id": f"{row['case_id']}-{frame_id}", "case_id": row["case_id"],
                "frame_id": frame_id, "timestamp_s": timestamp_s,
                "packet_sha256": row["packet_sha256"], "messages_sha256": messages_sha,
                "ledger_packet_sha256": _digest({"original_packet_sha": row["packet_sha256"],
                                                  "frame_id": frame_id}),
            })
    return cases, slots


def _validate_plan(run_dir: Path, protocol: dict[str, Any], protocol_sha: str) -> dict[str, Any]:
    plan = _read(run_dir / "plan.json")
    if not isinstance(plan, dict) or set(plan) != {
        "schema", "protocol_sha256", "catalog_sha256", "source_anchor", "cases", "slots", "plan_sha256"
    }:
        raise ValueError("plan 필드가 다르다")
    unsigned = {key: value for key, value in plan.items() if key != "plan_sha256"}
    cases, slots = _expected_inputs(run_dir, protocol)
    if (plan["schema"] != "visual_localization_plan_v2" or plan["protocol_sha256"] != protocol_sha
            or plan["catalog_sha256"] != protocol["catalog_sha256"]
            or plan["source_anchor"] != protocol["source_anchor"] or plan["plan_sha256"] != _digest(unsigned)
            or plan["cases"] != cases or plan["slots"] != slots):
        raise ValueError("plan이 frozen 입력과 다르다")
    return plan


def prepare(catalog_path: Path, protocol_path: Path, run_dir: Path) -> dict[str, Any]:
    catalog_path, protocol_path, run_dir = map(Path, (catalog_path, protocol_path, run_dir))
    protocol = _load_protocol(protocol_path)
    if _sha(catalog_path) != protocol["catalog_sha256"]:
        raise ValueError("등록된 catalog SHA와 입력이 다르다")
    catalog = V1.validate_catalog(catalog_path)
    if run_dir.exists():
        raise FileExistsError(run_dir)
    (run_dir / "cases").mkdir(parents=True)
    shutil.copyfile(catalog_path, run_dir / "cases" / "catalog.json")
    for row in catalog["cases"]:
        shutil.copytree(catalog_path.parent / row["case_id"], run_dir / "cases" / row["case_id"])
    cases, slots = _expected_inputs(run_dir, protocol)
    plan = {
        "schema": "visual_localization_plan_v2", "protocol_sha256": _sha(protocol_path),
        "catalog_sha256": _sha(catalog_path), "source_anchor": protocol["source_anchor"],
        "cases": cases, "slots": slots,
    }
    plan["plan_sha256"] = _digest(plan)
    _write_new(run_dir / "plan.json", plan)
    return plan


def _safe_result(raw: Any) -> dict[str, Any]:
    return V1._safe_result(raw)


def _transport_ok(saved: dict[str, Any]) -> bool:
    return V1._transport_ok(saved)


def _verified_provider(saved: Any) -> dict[str, Any]:
    return V1._verified_provider(saved)


def _validate_record(record: Any, slot: dict[str, Any], packet: dict[str, Any]) -> dict[str, Any]:
    required = {
        "schema", "slot_id", "case_id", "frame_id", "timestamp_s", "packet_sha256", "messages_sha256",
        "execution_kind",
        "provider", "transport_ok", "localization_validator", "derived_host_decision", "gate",
        "approval_granted", "used_for_behavior_or_recovery_decision", "record_sha256",
    }
    if (not isinstance(record, dict) or set(record) != required
            or record.get("schema") != "visual_localization_record_v2"):
        raise ValueError("record schema가 다르다")
    for key in ("slot_id", "case_id", "frame_id", "timestamp_s", "packet_sha256", "messages_sha256"):
        if record.get(key) != slot[key]:
            raise ValueError("record와 slot 결합이 다르다")
    if record["record_sha256"] != _digest({key: value for key, value in record.items() if key != "record_sha256"}):
        raise ValueError("record digest가 다르다")
    provider = _verified_provider(record["provider"])
    if (record["transport_ok"] is not _transport_ok(provider)
            or record["execution_kind"] not in {"live_provider", "cpu_fake"}
            or record["approval_granted"] is not False
            or record["used_for_behavior_or_recovery_decision"] is not False):
        raise ValueError("record 실행 경계가 다르다")
    validator = validate_localization(provider.get("content")) if record["transport_ok"] else {
        "valid": False, "format_reasons": ["transport_not_ok"], "semantic_reasons": [], "localization": None,
    }
    decision = (derive_host_decision(packet, validator["localization"]) if validator["valid"] else {
        "schema": "visual_localization_host_decision_v2", "status": "unclear",
        "gate": "HOLD_visual_evidence", "reason": "invalid_or_transport",
    })
    if (record["localization_validator"] != validator or record["derived_host_decision"] != decision
            or record["gate"] != decision["gate"]):
        raise ValueError("localization 또는 host decision이 다르다")
    return record


def execute(run_dir: Path, protocol_path: Path, expected_protocol_sha: str,
            sender: Callable[[list[dict[str, Any]], dict[str, Any]], dict[str, Any]] | None = None) -> dict[str, Any]:
    run_dir, protocol_path = Path(run_dir).resolve(), Path(protocol_path).resolve()
    protocol = _load_protocol(protocol_path, expected_protocol_sha)
    plan = _validate_plan(run_dir, protocol, expected_protocol_sha)
    live = sender is None
    if live and os.name != "posix":
        raise RuntimeError("live provider 실행은 POSIX에서만 허용된다")
    if live and not os.environ.get("NVIDIA_API_KEY"):
        raise RuntimeError("NVIDIA_API_KEY가 없다")
    complete = FEEDBACK.openai_complete if sender is None else sender
    ledger = V1.AUX.RequestLedger(run_dir / "api")
    if any(row["response"] is None for row in ledger.audit()):
        raise V1.AUX.LedgerStop("pending 예약이 있어 재호출하지 않는다")
    calls = 0
    for slot in plan["slots"]:
        case_dir = run_dir / "cases" / slot["case_id"]
        packet = V1.validate_packet(case_dir)
        record_path = run_dir / "records" / f"{slot['slot_id']}.json"
        if record_path.exists():
            _validate_record(_read(record_path), slot, packet)
            continue
        if any(row["reservation"]["packet_sha256"] == slot["ledger_packet_sha256"] for row in ledger.audit()):
            raise V1.AUX.LedgerStop("완료 원장은 있으나 record가 없어 fail closed한다")
        messages = messages_for_case(case_dir, slot["frame_id"])
        if _digest(messages) != slot["messages_sha256"]:
            raise ValueError("execute messages가 plan과 다르다")
        params = {"model": MODEL, **copy.deepcopy(REQUEST_PARAMS)}
        request_sha = _digest({"messages": messages, "params": params})
        reservation = ledger.reserve(slot["ledger_packet_sha256"], slot["slot_id"], 1, request_sha)
        started = time.perf_counter()
        try:
            safe = _safe_result(V1.ENDPOINT.provider_view(complete(messages, params), MODEL))
            transport_status = "ok" if _transport_ok(safe) else (
                "infra_error" if safe.get("infra_error") else "call_error")
        except Exception as exc:  # noqa: BLE001 - provider exceptions are redacted by type
            safe = {"error_type": type(exc).__name__, "wall_s": time.perf_counter() - started}
            transport_status = "call_error"
        ledger.finish(reservation, safe, transport_status)
        transport_ok = transport_status == "ok"
        validator = validate_localization(safe.get("content")) if transport_ok else {
            "valid": False, "format_reasons": ["transport_not_ok"], "semantic_reasons": [],
            "localization": None,
        }
        decision = (derive_host_decision(packet, validator["localization"]) if validator["valid"] else {
            "schema": "visual_localization_host_decision_v2", "status": "unclear",
            "gate": "HOLD_visual_evidence", "reason": "invalid_or_transport",
        })
        record = {
            "schema": "visual_localization_record_v2", "slot_id": slot["slot_id"],
            "case_id": slot["case_id"], "frame_id": slot["frame_id"], "timestamp_s": slot["timestamp_s"],
            "packet_sha256": slot["packet_sha256"],
            "messages_sha256": slot["messages_sha256"], "execution_kind": "live_provider" if live else "cpu_fake",
            "provider": safe, "transport_ok": transport_ok, "localization_validator": validator,
            "derived_host_decision": decision, "gate": decision["gate"], "approval_granted": False,
            "used_for_behavior_or_recovery_decision": False,
        }
        record["record_sha256"] = _digest(record)
        _write_new(record_path, record)
        calls += 1
    rows = ledger.audit()
    if len(rows) != 8 or any(row["response"] is None for row in rows):
        raise ValueError("8개 요청 원장이 완결되지 않았다")
    result = {
        "schema": "visual_localization_execution_v2", "status": "complete", "records": 8,
        "calls_this_invocation": calls, "execution_kind": "live_provider" if live else "cpu_fake",
        "live_evidence": live, "completed_at": _now(),
    }
    output = run_dir / "execution.json"
    if output.exists():
        old = _read(output)
        if (old.get("status") != "complete" or old.get("records") != 8
                or old.get("execution_kind") != result["execution_kind"]
                or old.get("live_evidence") is not result["live_evidence"]):
            raise ValueError("기존 execution 결과가 다르다")
    else:
        _write_new(output, result)
    return result


def _validated_records(run_dir: Path, protocol: dict[str, Any], protocol_sha: str) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    plan = _validate_plan(run_dir, protocol, protocol_sha)
    ledger_rows = V1.AUX.RequestLedger(run_dir / "api").audit()
    execution = _read(run_dir / "execution.json")
    if (len(ledger_rows) != 8 or any(row["response"] is None for row in ledger_rows)
            or execution.get("schema") != "visual_localization_execution_v2"
            or execution.get("status") != "complete" or execution.get("records") != 8
            or execution.get("execution_kind") not in {"live_provider", "cpu_fake"}
            or execution.get("live_evidence") is not (execution.get("execution_kind") == "live_provider")):
        raise ValueError("execution 또는 원장 경계가 다르다")
    records = []
    for index, slot in enumerate(plan["slots"]):
        case_dir = run_dir / "cases" / slot["case_id"]
        record = _validate_record(_read(run_dir / "records" / f"{slot['slot_id']}.json"), slot,
                                  V1.validate_packet(case_dir))
        reservation, response = ledger_rows[index]["reservation"], ledger_rows[index]["response"]
        messages = messages_for_case(case_dir, slot["frame_id"])
        request_sha = _digest({"messages": messages, "params": {"model": MODEL, **copy.deepcopy(REQUEST_PARAMS)}})
        expected_transport = "ok" if record["transport_ok"] else (
            "infra_error" if record["provider"].get("infra_error") else "call_error")
        if (reservation["packet_sha256"] != slot["ledger_packet_sha256"]
                or reservation["clip_id"] != slot["slot_id"] or reservation["attempt"] != 1
                or reservation["request_sha256"] != request_sha or response["result"] != record["provider"]
                or response["transport_status"] != expected_transport
                or record["execution_kind"] != execution["execution_kind"]):
            raise ValueError("record와 요청 원장 결합이 다르다")
        records.append(record)
    return plan, records, execution


def _case_gate(records: list[dict[str, Any]]) -> str:
    if len(records) != 2 or {record["frame_id"] for record in records} != {"f03", "f06"}:
        raise ValueError("case에는 f03과 f06 record가 하나씩 있어야 한다")
    statuses = [record["derived_host_decision"]["status"] for record in records]
    if statuses == ["consistent", "consistent"]:
        return "reviewed_consistent"
    if statuses == ["inconsistent", "inconsistent"]:
        return "quarantine_visual_evidence"
    return "HOLD_visual_evidence"


def score(run_dir: Path, protocol_path: Path, labels_path: Path, output_path: Path) -> dict[str, Any]:
    run_dir, protocol_path, labels_path, output_path = map(Path, (run_dir, protocol_path, labels_path, output_path))
    if output_path.exists():
        raise FileExistsError(output_path)
    protocol = _load_protocol(protocol_path)
    if _sha(labels_path) != protocol["labels_sha256"]:
        raise ValueError("등록된 labels SHA와 입력이 다르다")
    labels = V1._validate_labels(labels_path)
    plan, records, execution = _validated_records(run_dir, protocol, _sha(protocol_path))
    valid = stale_quarantined = fresh_consistent = 0
    rows = []
    for slot, record in zip(plan["slots"], records, strict=True):
        expected = labels["labels"][slot["case_id"]]["expected_status"]
        valid += int(record["localization_validator"]["valid"])
        status = record["derived_host_decision"]["status"]
        stale_ok = expected == "inconsistent" and status == "inconsistent" and record["gate"] == "quarantine_visual_evidence"
        fresh_ok = expected == "consistent" and status == "consistent" and record["gate"] == "reviewed_consistent"
        stale_quarantined += int(stale_ok)
        fresh_consistent += int(fresh_ok)
        rows.append({"slot_id": slot["slot_id"], "case_id": slot["case_id"], "frame_id": slot["frame_id"],
                     "timestamp_s": slot["timestamp_s"],
                     "expected_status": expected, "host_status": status, "valid_localization": record["localization_validator"]["valid"],
                     "gate": record["gate"]})
    confirmed = valid == 8 and stale_quarantined == 4 and fresh_consistent == 4
    live = execution["live_evidence"] is True
    outcome = ("synthetic_cpu_only" if not live else
               "feature_confirmed_development_only" if confirmed else "partial_or_not_confirmed")
    result = {
        "schema": "visual_localization_score_v2", "outcome": outcome,
        "execution_kind": execution["execution_kind"], "live_evidence": live,
        "primary_criterion": {
            "valid_localizations": {"value": valid, "required": 8},
            "stale_host_quarantined": {"value": stale_quarantined, "required": 4},
            "fresh_host_consistent": {"value": fresh_consistent, "required": 4},
        },
        "coverage": {"unique_cases": 4, "checkpoint_lineages": 3, "timestamp_calls": 8,
                     "scope": "development_blind_visual_localization_only"},
        "case_gates": {case_id: _case_gate([record for record in records if record["case_id"] == case_id])
                       for case_id in V1.CASE_IDS},
        "intervention_bundle": {
            "changes": ["state_coordinates_hidden", "single_image_request", "image_detail_omitted"],
            "individual_causal_attribution": False,
        },
        "model_role": "bbox_localization_only", "decision_role": "deterministic_host_geometry_v2",
        "used_for_behavior_or_recovery_decision": False, "llm_superiority_claimed": False,
        "rows": rows, "scored_at": _now(),
    }
    _write_new(output_path, result)
    return result


def reviewed_payload(run_dir: Path, protocol_path: Path, case_id: str,
                     consumer: Callable[[list[dict[str, Any]]], Any]) -> Any | None:
    if case_id not in V1.CASE_IDS:
        raise ValueError("case_id가 등록 범위 밖이다")
    run_dir, protocol_path = Path(run_dir).resolve(), Path(protocol_path).resolve()
    protocol = _load_protocol(protocol_path)
    _, records, _ = _validated_records(run_dir, protocol, _sha(protocol_path))
    selected = [record for record in records if record["case_id"] == case_id]
    if (_case_gate(selected) != "reviewed_consistent"
            or any(record["gate"] != "reviewed_consistent" for record in selected)):
        return None
    content = []
    for frame_id in ("f03", "f06"):
        content.extend(messages_for_case(run_dir / "cases" / case_id, frame_id)[1]["content"])
    return consumer([{"role": "user", "content": content}])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Blind VL body localization development experiment")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--catalog", type=Path, required=True)
    prepare_parser.add_argument("--protocol", type=Path, required=True)
    prepare_parser.add_argument("--run", type=Path, required=True)
    execute_parser = commands.add_parser("execute")
    execute_parser.add_argument("--run", type=Path, required=True)
    execute_parser.add_argument("--protocol", type=Path, required=True)
    execute_parser.add_argument("--expected-protocol-sha", required=True)
    score_parser = commands.add_parser("score")
    score_parser.add_argument("--run", type=Path, required=True)
    score_parser.add_argument("--protocol", type=Path, required=True)
    score_parser.add_argument("--labels", type=Path, required=True)
    score_parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        prepare(args.catalog, args.protocol, args.run)
    elif args.command == "execute":
        execute(args.run, args.protocol, args.expected_protocol_sha)
    else:
        score(args.run, args.protocol, args.labels, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
