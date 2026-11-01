"""Local-pair driver 앞단의 입력·transport·busy HOLD 분류 경계."""
from __future__ import annotations

import os
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any, NoReturn

SCHEMA = "local_pair_input_runtime_guard_v1"
FAULT_SCHEMA = "local_pair_input_runtime_fault_v1"
PENDING_NO_RESEND = "기존 dispatch가 미완료이므로 request를 다시 보내지 않는다"
STAGES = {"input", "quality", "geometry", "transport", "busy", "ready"}


class GuardFatalInvalid(ValueError):
    """등록된 HOLD 범위가 아닌 schema·programming 오류."""


def _fatal(reason: str) -> NoReturn:
    raise GuardFatalInvalid(reason)


def _result(decision: str, stage: str, reason: str, evidence: dict | None = None) -> dict:
    if decision not in {"continue", "HOLD"} or stage not in STAGES or not reason:
        _fatal("guard result contract")
    return {"schema": SCHEMA, "decision": decision, "stage": stage, "reason": reason,
            "approval_granted": False, "dispatch_granted": False,
            "automatic_retry": False, "evidence": deepcopy(evidence or {})}


def _registered_input_error(error: BaseException, registered_input_roots: object) -> bool:
    if not isinstance(registered_input_roots, (list, tuple)):
        _fatal("registered input roots contract")
    filename = getattr(error, "filename", None)
    if not isinstance(filename, (str, bytes)) or not filename:
        return False
    candidate = Path(os.fsdecode(filename)).resolve()
    for raw_root in registered_input_roots:
        if not isinstance(raw_root, (str, bytes, Path)):
            _fatal("registered input root type")
        root = Path(os.fsdecode(raw_root)).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            continue
        return True
    return False


def classify_exception(stage: str, error: BaseException, *,
                       registered_input_roots: tuple[Path | str, ...] | list[Path | str] = ()) -> dict:
    """HOLD 허용 예외를 좁게 분류하며 나머지는 fatal-invalid로 표시한다."""
    if stage not in STAGES or not isinstance(error, BaseException):
        _fatal("exception classifier input")
    disposition, reason = "fatal_invalid", "unregistered_exception"
    if (stage in {"input", "quality"} and isinstance(error, (FileNotFoundError, PermissionError))
            and _registered_input_error(error, registered_input_roots)):
        disposition, reason = "HOLD", "registered_input_unavailable"
    elif stage == "transport" and isinstance(error, RuntimeError) and str(error) == PENDING_NO_RESEND:
        disposition, reason = "HOLD", "transport_pending_no_resend"
    elif stage == "transport" and isinstance(error, (TimeoutError, ConnectionError)):
        disposition, reason = "HOLD", "typed_transport_error"
    return {"schema": FAULT_SCHEMA, "stage": stage, "disposition": disposition,
            "reason": reason, "error_type": type(error).__name__}


def invoke_guarded(stage: str, callback: Callable[..., Any], *args: Any,
                   registered_input_roots: tuple[Path | str, ...] | list[Path | str] = (),
                   **kwargs: Any) -> Any:
    """Expected failure만 HOLD로 바꾸고 programming/schema 오류는 다시 올린다."""
    if not callable(callback):
        _fatal("callback required")
    try:
        return callback(*args, **kwargs)
    except Exception as error:
        fault = classify_exception(stage, error, registered_input_roots=registered_input_roots)
        if fault["disposition"] == "HOLD":
            return _result("HOLD", stage, fault["reason"], {"fault": fault})
        raise


def gate_quality(receipt: dict) -> dict:
    if (not isinstance(receipt, dict)
            or receipt.get("schema") != "agentic_frame_quality_guard_v1_evidence_receipt"
            or receipt.get("approval_granted") is not False
            or receipt.get("visual_outcome") not in {"available", "unknown"}
            or not isinstance(receipt.get("frame_refs"), list)
            or type(receipt.get("image_count")) is not int
            or not isinstance(receipt.get("quality_receipts"), list)):
        _fatal("quality receipt contract")
    outcomes = [row.get("outcome") for row in receipt["quality_receipts"]
                if isinstance(row, dict)]
    if len(outcomes) != len(receipt["quality_receipts"]) or not outcomes:
        _fatal("quality outcomes missing")
    if receipt["visual_outcome"] == "unknown":
        if receipt["frame_refs"] or receipt["image_count"] != 0 or all(x == "available" for x in outcomes):
            _fatal("unknown quality did not quarantine frames")
        return _result("HOLD", "quality", "visual_quality_unknown",
                       {"outcomes": outcomes, "image_count": 0})
    if (receipt["image_count"] != len(receipt["frame_refs"])
            or receipt["image_count"] <= 0 or any(x != "available" for x in outcomes)):
        _fatal("available quality receipt mismatch")
    return _result("continue", "quality", "visual_quality_available",
                   {"outcomes": outcomes, "image_count": receipt["image_count"]})


def gate_geometry(decision: dict, numeric_status: str) -> dict:
    gates = {"consistent": "reviewed_consistent", "inconsistent": "quarantine_visual_evidence",
             "unclear": "HOLD_visual_evidence"}
    if (not isinstance(decision, dict)
            or decision.get("schema") != "visual_localization_host_decision_v2"
            or decision.get("status") not in gates
            or decision.get("gate") != gates[decision["status"]]
            or numeric_status not in {"consistent", "inconsistent", "unknown"}):
        _fatal("geometry/numeric contract")
    visual = decision["status"]
    if visual == "unclear" or numeric_status == "unknown":
        return _result("HOLD", "geometry", "visual_or_numeric_unknown",
                       {"visual_status": visual, "numeric_status": numeric_status})
    if visual != numeric_status:
        return _result("HOLD", "geometry", "verified_visual_numeric_conflict",
                       {"visual_status": visual, "numeric_status": numeric_status})
    if visual == "inconsistent":
        return _result("HOLD", "geometry", "visual_evidence_quarantined",
                       {"visual_status": visual, "numeric_status": numeric_status})
    return _result("continue", "geometry", "visual_numeric_consistent",
                   {"visual_status": visual, "numeric_status": numeric_status})


def gate_transport(result: dict) -> dict:
    pairs = {"completed": "semantic_valid", "unknown": "semantic_invalid",
             "infra_error": "infrastructure_unknown"}
    if (not isinstance(result, dict)
            or result.get("schema") not in {"local_pair_nim_verified_call_v1",
                                             "lineage_five_policy_nim_verified_call_v1",
                                             "five_policy_recovery_nim_verified_call_v1"}
            or result.get("status") not in pairs
            or result.get("outcome") != pairs[result["status"]]
            or type(result.get("live")) is not bool
            or not isinstance(result.get("call_ref"), dict)):
        _fatal("verified transport contract")
    evidence = {"status": result["status"], "outcome": result["outcome"],
                "live": result["live"], "call_ref": deepcopy(result["call_ref"])}
    if result["status"] == "completed":
        if not isinstance(result.get("content"), str) or not result["content"].strip():
            _fatal("completed transport content")
        return _result("continue", "transport", "verified_transport_completed", evidence)
    return _result("HOLD", "transport", (
        "verified_provider_unknown" if result["status"] == "unknown"
        else "verified_transport_unknown"), evidence)


def gate_busy(observation: dict) -> dict:
    if (not isinstance(observation, dict) or set(observation) != {"status", "observation_id"}
            or observation.get("status") not in {"available", "busy"}
            or not isinstance(observation.get("observation_id"), str)
            or not observation["observation_id"]):
        _fatal("busy observation contract")
    if observation["status"] == "busy":
        return _result("HOLD", "busy", "gpu_resource_busy", observation)
    return _result("continue", "busy", "gpu_resource_available", observation)


def pre_dispatch_gate(*, quality_receipt: dict | None = None,
                      geometry_decision: dict | None = None,
                      numeric_status: str | None = None,
                      transport_result: dict | None = None,
                      busy_reader: Callable[[str, dict], dict] | None = None,
                      operation_id: str | None = None, intent: dict | None = None) -> dict:
    """Driver가 승인/dispatch 전에 호출하는 무상태 합성 gate."""
    checks = []
    if quality_receipt is not None:
        checks.append(gate_quality(quality_receipt))
    if geometry_decision is not None or numeric_status is not None:
        if geometry_decision is None or numeric_status is None:
            _fatal("geometry pair required")
        checks.append(gate_geometry(geometry_decision, numeric_status))
    if transport_result is not None:
        checks.append(gate_transport(transport_result))
    for check in checks:
        if check["decision"] == "HOLD":
            return check
    if busy_reader is not None:
        if not callable(busy_reader) or not isinstance(operation_id, str) or not operation_id \
                or not isinstance(intent, dict):
            _fatal("busy callback inputs")
        busy = gate_busy(busy_reader(operation_id, deepcopy(intent)))
        if busy["decision"] == "HOLD":
            return busy
    return _result("continue", "ready", "registered_guards_clear",
                   {"checks": [row["reason"] for row in checks]})
