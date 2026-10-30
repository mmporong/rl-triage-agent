"""Blind VL render/state consistency experiment with a fresh eight-call ledger."""
from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import math
import os
import re
import shutil
import struct
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
from evals import nemotron_endpoint_ledger_v3 as ENDPOINT
from evals import vl_aux_observer_v1 as AUX

MODEL = "nvidia/nemotron-3.5-super-vl-120b-a12b"
CASE_IDS = tuple(f"clip-{index:02d}" for index in range(1, 5))
FRAME_IDS = tuple(f"f{index:02d}" for index in range(1, 7))
REQUEST_PARAMS = {
    "temperature": 0,
    "max_tokens": 2048,
    "stream": False,
    "extra_body": {"reasoning_effort": "none"},
}
SOURCE_FILES = (
    "evals/visual_integrity_experiment_v1.py",
    "tests/test_visual_integrity_experiment_v1.py",
    "evals/vl_aux_observer_v1.py",
    "evals/vl_aux_worker_v1.py",
    "evals/feedback_model_call.py",
    "evals/nemotron_endpoint_ledger_v3.py",
    "evals/nemotron_bottleneck_ledger_v2.py",
    "evals/nemotron_bottleneck_vision_v2.py",
)
SYSTEM = (
    "You inspect robot-render consistency only. The camera follows the simulated root with a fixed offset. "
    "The actual quadruped body should remain near expected_root_pixel; small shape or centroid offsets are "
    "allowed. Ignore velocity arrows and markers. A missing robot or a large body/projection mismatch is "
    "inconsistent. Insufficient or occluded visual evidence is unclear. Numeric consistency is not visual "
    "consistency. Do not guess causes and do not judge recovery. Return exactly one JSON object with only: "
    "status ('consistent', 'inconsistent', or 'unclear'); frame_ids (unique IDs from the input only); "
    "reason_code ('robot_matches_projection', 'robot_displaced_from_state', 'robot_not_visible', or "
    "'insufficient_visual_evidence'); and a non-empty explanation of at most 400 characters. Use "
    "robot_matches_projection only with consistent and cite at least two frame IDs. Use "
    "robot_displaced_from_state or robot_not_visible only with inconsistent and cite at least one frame ID. "
    "Use insufficient_visual_evidence only with unclear; unclear may cite no frames."
)

_HEX = re.compile(r"[0-9a-f]{64}")
_GIT = re.compile(r"[0-9a-f]{40}")
_LINE_REF = re.compile(r"(?:\bline|줄)\s*#?\s*\d+", re.IGNORECASE)
_PACKET_KEYS = {
    "schema", "image_size", "camera_mode", "samples", "frames", "numeric_binding_checks"
}
_SAMPLE_KEYS = {
    "frame_id", "timestamp_s", "root_position_world_m", "expected_root_pixel", "camera_eye_world_m"
}
_FRAME_KEYS = {"frame_id", "file", "sha256", "bytes"}
_RESPONSE_KEYS = {"status", "frame_ids", "reason_code", "explanation"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()  # noqa: UP017 - Python 3.10 contract


def _read(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_new(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _lf_sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def _digest(value: Any) -> str:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _finite_vector(value: Any, length: int, name: str) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{name} shape이 다르다")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item)
           for item in value):
        raise ValueError(f"{name}에 유한하지 않은 수가 있다")
    return [float(item) for item in value]


def _relative(value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{name} 상대 경로가 잘못됐다")
    path = Path(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{name} 경로가 범위를 벗어난다")
    return path


def _inside(root: Path, relative: Path) -> Path:
    root = root.resolve()
    target = (root / relative).resolve()
    if target != root and root not in target.parents:
        raise ValueError("상대 경로가 입력 루트를 벗어난다")
    return target


def _png_size(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    if len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
        raise ValueError("frame이 PNG가 아니다")
    return struct.unpack(">II", data[16:24])


def validate_packet(case_dir: Path) -> dict[str, Any]:
    case_dir = Path(case_dir).resolve()
    packet_path = case_dir / "packet.json"
    packet = _read(packet_path)
    if not isinstance(packet, dict) or set(packet) != _PACKET_KEYS:
        raise ValueError("packet 필드가 고정 계약과 다르다")
    if packet["schema"] != "visual_integrity_case_v1":
        raise ValueError("packet schema가 다르다")
    if packet["image_size"] != [640, 480]:
        raise ValueError("image_size는 640x480이어야 한다")
    if packet["camera_mode"] != "follows_simulated_root_with_fixed_offset":
        raise ValueError("camera_mode가 다르다")
    samples, frames = packet["samples"], packet["frames"]
    if not isinstance(samples, list) or not isinstance(frames, list) or len(samples) != 6 or len(frames) != 6:
        raise ValueError("sample과 frame은 각각 6개여야 한다")
    for index, (sample, frame) in enumerate(zip(samples, frames, strict=True), 1):
        frame_id = FRAME_IDS[index - 1]
        if not isinstance(sample, dict) or set(sample) != _SAMPLE_KEYS:
            raise ValueError("sample 필드가 다르다")
        timestamp = sample["timestamp_s"]
        if (sample["frame_id"] != frame_id or isinstance(timestamp, bool)
                or not isinstance(timestamp, (int, float)) or timestamp != index):
            raise ValueError("sample 순서 또는 시각이 다르다")
        _finite_vector(sample["root_position_world_m"], 3, "root_position_world_m")
        _finite_vector(sample["expected_root_pixel"], 2, "expected_root_pixel")
        _finite_vector(sample["camera_eye_world_m"], 3, "camera_eye_world_m")
        if not isinstance(frame, dict) or set(frame) != _FRAME_KEYS or frame["frame_id"] != frame_id:
            raise ValueError("frame 필드 또는 순서가 다르다")
        relative = _relative(frame["file"], "frame.file")
        if relative.as_posix() != f"frames/{frame_id}.png":
            raise ValueError("frame 파일명이 frame_id와 다르다")
        path = _inside(case_dir, relative)
        if not path.is_file() or isinstance(frame["bytes"], bool) or not isinstance(frame["bytes"], int):
            raise ValueError("frame 파일 또는 bytes가 잘못됐다")
        if path.stat().st_size != frame["bytes"] or not _HEX.fullmatch(str(frame["sha256"])):
            raise ValueError("frame 크기 또는 hash 형식이 다르다")
        if _sha(path) != frame["sha256"]:
            raise ValueError("frame hash가 다르다")
        if _png_size(path) != (640, 480):
            raise ValueError("실제 PNG 크기가 image_size와 다르다")
    checks = packet["numeric_binding_checks"]
    if not isinstance(checks, dict) or set(checks) != {
        "frames", "camera_trace_eye_max_error_m", "timestamps_exact"
    }:
        raise ValueError("numeric_binding_checks 필드가 다르다")
    error = checks["camera_trace_eye_max_error_m"]
    if (checks["frames"] != 625 or checks["timestamps_exact"] is not True
            or isinstance(error, bool) or not isinstance(error, (int, float)) or not math.isfinite(error)
            or not 0 <= error <= 1e-8):
        raise ValueError("수치 결합 검사가 고정 조건을 통과하지 않았다")
    return packet


def validate_catalog(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    catalog = _read(path)
    if not isinstance(catalog, dict) or set(catalog) != {"schema", "cases"}:
        raise ValueError("catalog 필드가 다르다")
    if catalog["schema"] != "visual_integrity_catalog_v1" or not isinstance(catalog["cases"], list):
        raise ValueError("catalog schema 또는 cases가 다르다")
    if [row.get("case_id") for row in catalog["cases"] if isinstance(row, dict)] != list(CASE_IDS):
        raise ValueError("catalog는 clip-01부터 clip-04까지 순서대로 포함해야 한다")
    for case_id, row in zip(CASE_IDS, catalog["cases"], strict=True):
        if set(row) != {"case_id", "packet", "packet_sha256"}:
            raise ValueError("catalog case 필드가 다르다")
        relative = _relative(row["packet"], "catalog packet")
        if relative.as_posix() != f"{case_id}/packet.json" or not _HEX.fullmatch(str(row["packet_sha256"])):
            raise ValueError("catalog packet 결합이 다르다")
        packet_path = _inside(path.parent, relative)
        if _sha(packet_path) != row["packet_sha256"]:
            raise ValueError("catalog packet hash가 다르다")
        validate_packet(packet_path.parent)
    return catalog


def messages_for_case(case_dir: Path) -> list[dict[str, Any]]:
    case_dir = Path(case_dir).resolve()
    packet = validate_packet(case_dir)
    content: list[dict[str, Any]] = [{
        "type": "text",
        "text": (
            "Inspect the six ordered observations. Use only the supplied state projection and pixels. "
            "Return the required JSON object."
        ),
    }]
    for sample, frame in zip(packet["samples"], packet["frames"], strict=True):
        public_sample = {key: sample[key] for key in (
            "frame_id", "timestamp_s", "root_position_world_m", "expected_root_pixel", "camera_eye_world_m"
        )}
        content.append({"type": "text", "text": json.dumps(public_sample, separators=(",", ":"))})
        image = _inside(case_dir, _relative(frame["file"], "frame.file")).read_bytes()
        content.append({
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64," + base64.b64encode(image).decode("ascii"), "detail": "low"},
        })
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]


def validate_response(content: Any, allowed_frames: tuple[str, ...] = FRAME_IDS) -> dict[str, Any]:
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
    if not format_reasons:
        if not isinstance(value, dict):
            format_reasons.append("response_not_object")
        elif set(value) != _RESPONSE_KEYS:
            format_reasons.append("response_fields_mismatch")
    if not format_reasons:
        if not isinstance(value["status"], str) or value["status"] not in {
            "consistent", "inconsistent", "unclear"
        }:
            format_reasons.append("status_not_registered")
        if not isinstance(value["reason_code"], str) or value["reason_code"] not in {
            "robot_matches_projection", "robot_displaced_from_state", "robot_not_visible",
            "insufficient_visual_evidence",
        }:
            format_reasons.append("reason_code_not_registered")
        frame_ids = value["frame_ids"]
        if (not isinstance(frame_ids, list) or any(not isinstance(item, str) for item in frame_ids)):
            format_reasons.append("frame_ids_not_string_list")
        explanation = value["explanation"]
        if not isinstance(explanation, str) or not explanation.strip() or len(explanation) > 400:
            format_reasons.append("explanation_invalid")
    if not format_reasons:
        frame_ids = value["frame_ids"]
        if len(frame_ids) != len(set(frame_ids)):
            semantic_reasons.append("duplicate_frame_id")
        if any(frame_id not in allowed_frames for frame_id in frame_ids):
            semantic_reasons.append("unknown_frame_id")
        expected_reason = {
            "consistent": {"robot_matches_projection"},
            "inconsistent": {"robot_displaced_from_state", "robot_not_visible"},
            "unclear": {"insufficient_visual_evidence"},
        }[value["status"]]
        if value["reason_code"] not in expected_reason:
            semantic_reasons.append("status_reason_incompatible")
        if value["status"] == "consistent" and len(frame_ids) < 2:
            semantic_reasons.append("consistent_needs_two_frames")
        if value["status"] == "inconsistent" and not frame_ids:
            semantic_reasons.append("inconsistent_needs_frame")
        if _LINE_REF.search(value["explanation"]):
            semantic_reasons.append("line_reference_forbidden")
    valid = not format_reasons and not semantic_reasons
    return {
        "valid": valid,
        "format_reasons": format_reasons,
        "semantic_reasons": semantic_reasons,
        "decision": value if valid else None,
    }


def _validate_protocol_shape(protocol: Any) -> dict[str, Any]:
    if not isinstance(protocol, dict):
        raise TypeError("protocol은 object여야 한다")
    required = {
        "schema", "status", "source_anchor", "source_sha256_lf", "catalog_sha256", "labels_sha256",
        "model", "request_params", "budget", "scope",
    }
    if not required.issubset(protocol):
        raise ValueError("protocol 필수 필드가 없다")
    if (protocol["schema"] != "visual_integrity_experiment_protocol_v1"
            or protocol["status"] != "registered"
            or protocol["model"] != MODEL
            or protocol["request_params"] != REQUEST_PARAMS
            or protocol["budget"] != {"cases": 4, "repeats": 2, "request_cap": 8, "automatic_retries": 0}
            or protocol["scope"] != "development_render_integrity_only"):
        raise ValueError("protocol 고정 조건이 다르다")
    if not _GIT.fullmatch(str(protocol["source_anchor"])):
        raise ValueError("source_anchor 형식이 다르다")
    if set(protocol["source_sha256_lf"]) != set(SOURCE_FILES):
        raise ValueError("source map 범위가 다르다")
    if any(not _HEX.fullmatch(str(value)) for value in protocol["source_sha256_lf"].values()):
        raise ValueError("source hash 형식이 다르다")
    if not _HEX.fullmatch(str(protocol["catalog_sha256"])) or not _HEX.fullmatch(str(protocol["labels_sha256"])):
        raise ValueError("catalog 또는 labels hash 형식이 다르다")
    forbidden = ("C:\\Users\\", "C:/Users/", "\\Users\\", "/home/", "/Users/", "/mnt/", "bench/private")
    encoded = json.dumps(protocol, ensure_ascii=False)
    if any(token in encoded for token in forbidden):
        raise ValueError("공개 protocol에 개인 경로가 있다")
    return protocol


def _git(*args: str) -> bytes:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True).stdout


def _verify_source_guard(protocol: dict[str, Any]) -> None:
    anchor = protocol["source_anchor"]
    subprocess.run(["git", "merge-base", "--is-ancestor", anchor, "HEAD"], cwd=ROOT, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    for relative in SOURCE_FILES:
        expected = protocol["source_sha256_lf"][relative]
        current = _inside(ROOT, Path(relative)).read_bytes()
        if _lf_sha_bytes(current) != expected:
            raise ValueError(f"현재 source hash가 다르다: {relative}")
        anchored = _git("show", f"{anchor}:{relative}")
        if _lf_sha_bytes(anchored) != expected:
            raise ValueError(f"anchor source blob이 다르다: {relative}")


def _load_protocol(path: Path, expected_sha: str | None = None) -> dict[str, Any]:
    path = Path(path).resolve()
    if expected_sha is not None and (_sha(path) != expected_sha or not _HEX.fullmatch(expected_sha)):
        raise ValueError("protocol SHA가 다르다")
    protocol = _validate_protocol_shape(_read(path))
    _verify_source_guard(protocol)
    return protocol


def prepare(catalog_path: Path, protocol_path: Path, run_dir: Path) -> dict[str, Any]:
    catalog_path, protocol_path, run_dir = map(Path, (catalog_path, protocol_path, run_dir))
    protocol = _load_protocol(protocol_path)
    if _sha(catalog_path) != protocol["catalog_sha256"]:
        raise ValueError("등록된 catalog SHA와 입력이 다르다")
    catalog = validate_catalog(catalog_path)
    if run_dir.exists():
        raise FileExistsError(run_dir)
    (run_dir / "cases").mkdir(parents=True)
    shutil.copyfile(catalog_path, run_dir / "cases" / "catalog.json")
    case_rows = []
    slots = []
    for row in catalog["cases"]:
        source_dir = catalog_path.parent / row["case_id"]
        target_dir = run_dir / "cases" / row["case_id"]
        shutil.copytree(source_dir, target_dir)
        if _sha(target_dir / "packet.json") != row["packet_sha256"]:
            raise ValueError("복사된 packet SHA가 다르다")
        messages_sha = _digest(messages_for_case(target_dir))
        case_rows.append({
            "case_id": row["case_id"], "packet_sha256": row["packet_sha256"],
            "messages_sha256": messages_sha,
        })
        for repeat in (1, 2):
            slot_id = f"{row['case_id']}-r{repeat}"
            slots.append({
                "slot_id": slot_id, "case_id": row["case_id"], "repeat": repeat,
                "packet_sha256": row["packet_sha256"], "messages_sha256": messages_sha,
                "ledger_packet_sha256": _digest({"original_packet_sha": row["packet_sha256"], "repeat": repeat}),
            })
    plan = {
        "schema": "visual_integrity_plan_v1",
        "protocol_sha256": _sha(protocol_path),
        "catalog_sha256": _sha(catalog_path),
        "source_anchor": protocol["source_anchor"],
        "cases": case_rows,
        "slots": slots,
    }
    plan["plan_sha256"] = _digest(plan)
    _write_new(run_dir / "plan.json", plan)
    return plan


def _expected_plan_inputs(run_dir: Path, protocol: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    catalog_path = run_dir / "cases" / "catalog.json"
    if _sha(catalog_path) != protocol["catalog_sha256"]:
        raise ValueError("run catalog SHA가 protocol과 다르다")
    catalog = validate_catalog(catalog_path)
    cases: list[dict[str, Any]] = []
    slots: list[dict[str, Any]] = []
    for row in catalog["cases"]:
        messages_sha = _digest(messages_for_case(run_dir / "cases" / row["case_id"]))
        case = {
            "case_id": row["case_id"], "packet_sha256": row["packet_sha256"],
            "messages_sha256": messages_sha,
        }
        cases.append(case)
        for repeat in (1, 2):
            slots.append({
                "slot_id": f"{row['case_id']}-r{repeat}", "case_id": row["case_id"], "repeat": repeat,
                "packet_sha256": row["packet_sha256"], "messages_sha256": messages_sha,
                "ledger_packet_sha256": _digest({"original_packet_sha": row["packet_sha256"], "repeat": repeat}),
            })
    return cases, slots


def _validate_plan(run_dir: Path, protocol: dict[str, Any], protocol_sha: str) -> dict[str, Any]:
    plan = _read(run_dir / "plan.json")
    if not isinstance(plan, dict) or set(plan) != {
        "schema", "protocol_sha256", "catalog_sha256", "source_anchor", "cases", "slots", "plan_sha256"
    }:
        raise ValueError("plan 필드가 다르다")
    digest_input = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if (plan["schema"] != "visual_integrity_plan_v1" or plan["protocol_sha256"] != protocol_sha
            or plan["plan_sha256"] != _digest(digest_input) or len(plan["slots"]) != 8
            or plan["source_anchor"] != protocol["source_anchor"]
            or plan["catalog_sha256"] != protocol["catalog_sha256"]):
        raise ValueError("plan 결합이 다르다")
    expected_cases, expected_slots = _expected_plan_inputs(run_dir, protocol)
    if plan["cases"] != expected_cases or plan["slots"] != expected_slots:
        raise ValueError("plan case 또는 slot이 frozen catalog 입력과 다르다")
    return plan


def _safe_result(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {"error_type": "NonObjectProviderResult"}
    fields = (
        "response_id", "provider_model", "provider_model_raw", "provider_identity_registered",
        "identity_policy", "content", "finish_reason", "input_tokens", "output_tokens", "total_tokens",
        "wall_s", "error_type", "status_code", "infra_error",
    )
    return {key: raw[key] for key in fields if key in raw}


def _verified_provider(saved: Any) -> dict[str, Any]:
    if not isinstance(saved, dict):
        raise TypeError("provider record가 object가 아니다")
    raw = {key: saved[key] for key in (
        "response_id", "content", "finish_reason", "input_tokens", "output_tokens", "total_tokens",
        "wall_s", "error_type", "status_code", "infra_error",
    ) if key in saved}
    if "provider_model" in saved or "provider_model_raw" in saved:
        raw["provider_model"] = saved.get("provider_model_raw", saved.get("provider_model"))
    expected = _safe_result(ENDPOINT.provider_view(raw, MODEL))
    if saved != expected:
        raise ValueError("provider alias view가 저장 record와 다르다")
    return expected


def _transport_ok(saved: dict[str, Any]) -> bool:
    return (saved.get("provider_identity_registered") is True and isinstance(saved.get("content"), str)
            and not saved.get("error_type") and not saved.get("infra_error")
            and saved.get("status_code") in (None, 200) and saved.get("finish_reason") != "length")


def _gate(validator: dict[str, Any]) -> str:
    if not validator["valid"]:
        return "HOLD_visual_evidence"
    status = validator["decision"]["status"]
    if status == "inconsistent":
        return "quarantine_visual_evidence"
    if status == "consistent":
        return "reviewed_consistent"
    return "HOLD_visual_evidence"


def _validate_record(record: Any, slot: dict[str, Any]) -> dict[str, Any]:
    required = {
        "schema", "slot_id", "case_id", "repeat", "packet_sha256", "messages_sha256", "execution_kind",
        "provider", "transport_ok", "validator", "gate", "approval_granted",
        "used_for_behavior_or_recovery_decision", "record_sha256",
    }
    if (not isinstance(record, dict) or set(record) != required
            or record.get("schema") != "visual_integrity_record_v1"):
        raise ValueError("record schema가 다르다")
    for key in ("slot_id", "case_id", "repeat", "packet_sha256", "messages_sha256"):
        if record.get(key) != slot[key]:
            raise ValueError("record와 slot 결합이 다르다")
    if record.get("record_sha256") != _digest({key: value for key, value in record.items() if key != "record_sha256"}):
        raise ValueError("record digest가 다르다")
    provider = _verified_provider(record.get("provider"))
    if (record.get("transport_ok") is not _transport_ok(provider)
            or record.get("execution_kind") not in {"live_provider", "cpu_fake"}
            or record.get("approval_granted") is not False
            or record.get("used_for_behavior_or_recovery_decision") is not False):
        raise ValueError("record 실행 경계가 다르다")
    recalculated = validate_response(record.get("provider", {}).get("content")) if record.get("transport_ok") else {
        "valid": False, "format_reasons": ["transport_not_ok"], "semantic_reasons": [], "decision": None,
    }
    if record.get("validator") != recalculated or record.get("gate") != _gate(recalculated):
        raise ValueError("record validator 또는 gate가 다르다")
    return record


def execute(run_dir: Path, protocol_path: Path, expected_protocol_sha: str,
            sender: Callable[[list[dict[str, Any]], dict[str, Any]], dict[str, Any]] | None = None) -> dict[str, Any]:
    run_dir, protocol_path = Path(run_dir).resolve(), Path(protocol_path).resolve()
    protocol = _load_protocol(protocol_path, expected_protocol_sha)
    plan = _validate_plan(run_dir, protocol, expected_protocol_sha)
    if plan["source_anchor"] != protocol["source_anchor"] or plan["catalog_sha256"] != protocol["catalog_sha256"]:
        raise ValueError("plan과 protocol 결합이 다르다")
    live = sender is None
    if live and os.name != "posix":
        raise RuntimeError("live provider 실행은 POSIX에서만 허용된다")
    if live and not os.environ.get("NVIDIA_API_KEY"):
        raise RuntimeError("NVIDIA_API_KEY가 없다")
    complete = FEEDBACK.openai_complete if sender is None else sender
    ledger = AUX.RequestLedger(run_dir / "api")
    existing_rows = ledger.audit()
    if any(row["response"] is None for row in existing_rows):
        raise AUX.LedgerStop("pending 예약이 있어 재호출하지 않는다")
    calls = 0
    for slot in plan["slots"]:
        record_path = run_dir / "records" / f"{slot['slot_id']}.json"
        if record_path.exists():
            _validate_record(_read(record_path), slot)
            continue
        if any(row["reservation"]["packet_sha256"] == slot["ledger_packet_sha256"] for row in ledger.audit()):
            raise AUX.LedgerStop("완료 원장은 있으나 record가 없어 fail closed한다")
        messages = messages_for_case(run_dir / "cases" / slot["case_id"])
        if _digest(messages) != slot["messages_sha256"]:
            raise ValueError("execute messages가 plan과 다르다")
        params = {"model": MODEL, **copy.deepcopy(REQUEST_PARAMS)}
        request_sha = _digest({"messages": messages, "params": params})
        reservation = ledger.reserve(slot["ledger_packet_sha256"], slot["slot_id"], 1, request_sha)
        started = time.perf_counter()
        try:
            raw = complete(messages, params)
            safe = _safe_result(ENDPOINT.provider_view(raw, MODEL))
            transport_status = "ok" if _transport_ok(safe) else (
                "infra_error" if safe.get("infra_error") else "call_error")
        except Exception as exc:  # noqa: BLE001 - provider exceptions are redacted by type
            safe = {"error_type": type(exc).__name__, "wall_s": time.perf_counter() - started}
            transport_status = "call_error"
        ledger.finish(reservation, safe, transport_status)
        transport_ok = transport_status == "ok"
        validator = validate_response(safe.get("content")) if transport_ok else {
            "valid": False, "format_reasons": ["transport_not_ok"], "semantic_reasons": [], "decision": None,
        }
        record = {
            "schema": "visual_integrity_record_v1",
            "slot_id": slot["slot_id"], "case_id": slot["case_id"], "repeat": slot["repeat"],
            "packet_sha256": slot["packet_sha256"], "messages_sha256": slot["messages_sha256"],
            "execution_kind": "live_provider" if live else "cpu_fake",
            "provider": safe, "transport_ok": transport_ok, "validator": validator, "gate": _gate(validator),
            "approval_granted": False, "used_for_behavior_or_recovery_decision": False,
        }
        record["record_sha256"] = _digest(record)
        _write_new(record_path, record)
        calls += 1
    rows = ledger.audit()
    if len(rows) != 8 or any(row["response"] is None for row in rows):
        raise ValueError("8개 요청 원장이 완결되지 않았다")
    result = {
        "schema": "visual_integrity_execution_v1", "status": "complete", "records": 8,
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


def _validate_labels(path: Path) -> dict[str, Any]:
    labels = _read(path)
    if not isinstance(labels, dict) or set(labels) != {"schema", "labels"}:
        raise ValueError("labels 필드가 다르다")
    if labels["schema"] != "visual_integrity_labels_v1" or set(labels["labels"]) != set(CASE_IDS):
        raise ValueError("labels schema 또는 case 범위가 다르다")
    expected = []
    for case_id in CASE_IDS:
        row = labels["labels"][case_id]
        if not isinstance(row, dict) or set(row) != {"expected_status", "lineage_id", "source_refs"}:
            raise ValueError("label row 필드가 다르다")
        if row["expected_status"] not in {"inconsistent", "consistent"}:
            raise ValueError("expected_status가 다르다")
        if not isinstance(row["lineage_id"], str) or not row["lineage_id"]:
            raise ValueError("lineage_id가 없다")
        refs = row["source_refs"]
        if (not isinstance(refs, dict) or not refs
                or any(not isinstance(key, str) or not key or not _HEX.fullmatch(str(value))
                       for key, value in refs.items())):
            raise ValueError("source_refs가 잘못됐다")
        expected.append(row["expected_status"])
    if expected.count("inconsistent") != 2 or expected.count("consistent") != 2:
        raise ValueError("labels는 stale 2개와 fresh 2개여야 한다")
    if len({labels["labels"][case_id]["lineage_id"] for case_id in CASE_IDS}) != 3:
        raise ValueError("labels는 checkpoint lineage 3개를 결합해야 한다")
    return labels


def score(run_dir: Path, protocol_path: Path, labels_path: Path, output_path: Path) -> dict[str, Any]:
    run_dir, protocol_path, labels_path, output_path = map(Path, (run_dir, protocol_path, labels_path, output_path))
    if output_path.exists():
        raise FileExistsError(output_path)
    protocol = _load_protocol(protocol_path)
    if _sha(labels_path) != protocol["labels_sha256"]:
        raise ValueError("등록된 labels SHA와 입력이 다르다")
    labels = _validate_labels(labels_path)
    plan = _validate_plan(run_dir, protocol, _sha(protocol_path))
    ledger_rows = AUX.RequestLedger(run_dir / "api").audit()
    if len(ledger_rows) != 8 or any(row["response"] is None for row in ledger_rows):
        raise ValueError("score 입력 원장이 8개 완결 응답이 아니다")
    execution = _read(run_dir / "execution.json")
    if (not isinstance(execution, dict) or execution.get("schema") != "visual_integrity_execution_v1"
            or execution.get("status") != "complete" or execution.get("records") != 8
            or execution.get("execution_kind") not in {"live_provider", "cpu_fake"}
            or execution.get("live_evidence") is not (execution.get("execution_kind") == "live_provider")):
        raise ValueError("execution 경계가 다르다")
    valid = stale_quarantined = fresh_consistent = 0
    rows = []
    for index, slot in enumerate(plan["slots"]):
        record = _validate_record(_read(run_dir / "records" / f"{slot['slot_id']}.json"), slot)
        ledger_row = ledger_rows[index]
        reservation, response = ledger_row["reservation"], ledger_row["response"]
        messages = messages_for_case(run_dir / "cases" / slot["case_id"])
        request_sha = _digest({"messages": messages, "params": {"model": MODEL, **copy.deepcopy(REQUEST_PARAMS)}})
        expected_transport = "ok" if record["transport_ok"] else (
            "infra_error" if record["provider"].get("infra_error") else "call_error")
        if (reservation["packet_sha256"] != slot["ledger_packet_sha256"]
                or reservation["clip_id"] != slot["slot_id"] or reservation["attempt"] != 1
                or reservation["request_sha256"] != request_sha or response["result"] != record["provider"]
                or response["transport_status"] != expected_transport
                or record["execution_kind"] != execution["execution_kind"]):
            raise ValueError("record와 요청 원장 결합이 다르다")
        expected = labels["labels"][slot["case_id"]]["expected_status"]
        decision = record["validator"]["decision"] if record["validator"]["valid"] else None
        valid += int(decision is not None)
        stale_ok = (expected == "inconsistent" and decision is not None
                    and decision["status"] == "inconsistent" and record["gate"] == "quarantine_visual_evidence")
        fresh_ok = (expected == "consistent" and decision is not None
                    and decision["status"] == "consistent" and record["gate"] == "reviewed_consistent")
        stale_quarantined += int(stale_ok)
        fresh_consistent += int(fresh_ok)
        rows.append({
            "slot_id": slot["slot_id"], "case_id": slot["case_id"], "repeat": slot["repeat"],
            "expected_status": expected, "observed_status": decision["status"] if decision else None,
            "valid": decision is not None, "gate": record["gate"],
        })
    confirmed = valid == 8 and stale_quarantined == 4 and fresh_consistent == 4
    live = execution["live_evidence"] is True
    outcome = ("synthetic_cpu_only" if not live else
               "feature_confirmed_development_only" if confirmed else "partial_or_not_confirmed")
    result = {
        "schema": "visual_integrity_score_v1",
        "outcome": outcome,
        "execution_kind": execution["execution_kind"],
        "live_evidence": live,
        "primary_criterion": {
            "valid_calls": {"value": valid, "required": 8},
            "stale_calls_quarantined": {"value": stale_quarantined, "required": 4},
            "fresh_calls_consistent": {"value": fresh_consistent, "required": 4},
        },
        "coverage": {
            "unique_cases": 4, "checkpoint_lineages": 3, "repeated_calls": 8,
            "scope": "development_render_integrity_only",
        },
        "numeric_baseline": {
            "structure_and_camera_binding_passed": True,
            "predicts_visual_integrity": False,
            "independent_accuracy_advantage": False,
            "note": "Mask geometry shares the label method and is not an independent accuracy baseline.",
        },
        "used_for_behavior_or_recovery_decision": False,
        "llm_superiority_claimed": False,
        "rows": rows,
        "scored_at": _now(),
    }
    _write_new(output_path, result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="VL render/state consistency development experiment")
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
