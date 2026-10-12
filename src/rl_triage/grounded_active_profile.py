"""승인된 시뮬레이션 reset 계측을 source-bound 단일 실행으로 기록한다."""
from __future__ import annotations

import json
import hashlib
import math
import os
import re
import secrets
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from rl_triage import grounded_evidence as G
from rl_triage import single_run as S
from rl_triage import triage_tools as T


SHA256_RE = re.compile(r"[0-9a-f]{64}")
COMMIT_RE = re.compile(r"[0-9a-f]{40}")
SAFE_TEXT = re.compile(r"[A-Za-z0-9_.:-]+")
ACTIVE_REGISTRY_ID = "grounded_active_v1"
HOST_EXECUTABLE_CHECK_IDS = ("reset_base_com_stability",)
ACTIVE_CHECKS = deepcopy(G.READ_ONLY_CHECKS)
ACTIVE_CHECKS["reset_base_com_stability"] = {
    "safety": "approved_simulation_reset",
    "adapter_id": "host.base_com_reset_readback.v1",
    "measurements": {
        "base_com_max_abs_delta_xy": {
            "unit": "simulation_length_native", "value_type": "number",
        },
        "base_com_outside_range_fraction": {
            "unit": "fraction", "value_type": "number",
        },
    },
}
FIXED_EXECUTION_ARGUMENTS = {
    "num_envs": 64, "reset_sequences": 20, "resets_per_env": 10,
    "seed": 20261012, "allowed_xy_range": 0.05, "tolerance": 0.0001,
}
DECISION_KEYS = {"schema", "case_id", "effect_key", "budget_left", "status", "ranking", "receipt_citation"}
MANIFEST_KEYS = {
    "schema", "case_id", "public_case_alias", "sources", "assessment_sha256", "task", "seed",
    "protocol_sha256", "source_commit_sha", "private_executor_binding_sha256",
}
BINDING_KEYS = MANIFEST_KEYS | {"case_source_binding_sha256"}
REQUEST_COMPONENT_KEYS = {
    "case_id", "registry_id", "assessment_sha256", "registry_sha256",
    "definition_sha256", "source_binding_sha256", "arguments_sha256",
}
REQUEST_KEYS = {
    "schema", "created_at", "effect_key", *REQUEST_COMPONENT_KEYS, "execution_limit",
}
APPROVAL_KEYS = {
    "schema", "created_at", "case_id", "effect_key", "request_sha256",
    "source_binding_sha256", "actor", "identity_attested", "approved",
}
CONSUMED_KEYS = {
    "schema", "consumed_at", "case_id", "effect_key", "request_sha256",
    "approval_sha256", "case_source_binding_sha256", "side_effect_started",
}


def _sha(value: object) -> str:
    return G._canonical_sha(value)


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _require_sha(value: object, label: str, *, commit: bool = False) -> str:
    pattern = COMMIT_RE if commit else SHA256_RE
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ValueError(f"{label} 형식이 잘못되었습니다")
    return value


def _case_id(case_id: str) -> str:
    S._case_dir(case_id)
    return case_id


def _write_once(directory: str, filename: str, value: dict) -> tuple[Path, str]:
    out_dir = T._safe(T.WORKSPACE / "preregistrations" / directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = T._safe(out_dir / filename)
    raw = (json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
    temporary = T._safe(out_dir / f".{filename}.{os.getpid()}.{secrets.token_hex(8)}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, target)
    except FileExistsError:
        raise FileExistsError(f"기존 기록을 덮어쓸 수 없습니다: {directory}/{filename}")
    finally:
        temporary.unlink(missing_ok=True)
    return target, G._canonical_sha(value)


def _load_bound(directory: str, filename: str, expected_sha: str | None = None) -> dict:
    path = T._safe(T.WORKSPACE / "preregistrations" / directory / filename)
    value = T._load_json(path)
    if expected_sha is not None and _sha(value) != expected_sha:
        raise ValueError(f"{directory} 기록 SHA가 다릅니다")
    return value


def _revalidate_case_sources(case_id: str, binding: dict) -> None:
    expected_paths = {
        name: f"cases/{case_id}/{name}.json"
        for name in ("telemetry", "configuration", "context")
    }
    sources = binding.get("sources")
    if not isinstance(sources, dict) or set(sources) != set(expected_paths):
        raise ValueError("case source 목록이 잘못되었습니다")
    for name, relative in expected_paths.items():
        row = sources[name]
        if not isinstance(row, dict) or set(row) != {"path", "sha256"} or row["path"] != relative:
            raise ValueError("case source 경로가 잘못되었습니다")
        path = T._safe(T.WORKSPACE / relative)
        if _sha_bytes(path.read_bytes()) != _require_sha(row["sha256"], f"{name}.sha256"):
            raise ValueError("case source SHA가 다릅니다")


def _load_case_binding(case_id: str, binding_sha256: object) -> dict:
    binding_sha = _require_sha(binding_sha256, "case_source_binding_sha256")
    binding = _load_bound("active_case_bindings", f"{case_id}__{binding_sha}.json")
    if (set(binding) != BINDING_KEYS
            or binding.get("schema") != "grounded_active_case_manifest_v1"
            or binding.get("case_id") != case_id
            or binding.get("case_source_binding_sha256") != binding_sha):
        raise ValueError("case source binding 계약이 잘못되었습니다")
    manifest = {key: binding[key] for key in MANIFEST_KEYS}
    if _sha(manifest) != binding_sha:
        raise ValueError("case source binding canonical SHA가 다릅니다")
    alias = binding["public_case_alias"]
    if not isinstance(alias, str) or SAFE_TEXT.fullmatch(alias) is None:
        raise ValueError("case source binding alias가 잘못되었습니다")
    if not isinstance(binding["task"], str) or not binding["task"] or type(binding["seed"]) is not int:
        raise ValueError("case source binding task/seed가 잘못되었습니다")
    for field in ("assessment_sha256", "protocol_sha256", "private_executor_binding_sha256"):
        _require_sha(binding[field], field)
    _require_sha(binding["source_commit_sha"], "source_commit_sha", commit=True)
    _revalidate_case_sources(case_id, binding)
    return binding


def _validate_request_document(case_id: str, effect: str, request: object) -> dict:
    if not isinstance(request, dict) or set(request) != REQUEST_KEYS:
        raise ValueError("execution request 계약이 잘못되었습니다")
    components = {key: request[key] for key in REQUEST_COMPONENT_KEYS}
    if (request["schema"] != "grounded_active_request_v1"
            or not isinstance(request["created_at"], str) or not request["created_at"]
            or request["case_id"] != case_id or request["effect_key"] != effect
            or _sha(components) != effect
            or type(request["execution_limit"]) is not int
            or request["execution_limit"] != 1
            or request["registry_id"] != "reset_base_com_stability"
            or request["registry_sha256"] != _sha(ACTIVE_CHECKS)
            or request["definition_sha256"] != _sha(ACTIVE_CHECKS["reset_base_com_stability"])
            or request["arguments_sha256"] != _sha(FIXED_EXECUTION_ARGUMENTS)):
        raise ValueError("execution request 의미 결속이 잘못되었습니다")
    _load_case_binding(case_id, request["source_binding_sha256"])
    assessment = _load_bound("grounded_active_assessments", f"{case_id}.json")
    if (request["assessment_sha256"] != _sha(assessment)
            or assessment.get("next_check", {}).get("registry_id") != request["registry_id"]):
        raise ValueError("execution request assessment 결속이 잘못되었습니다")
    return request


def _validate_approval_document(case_id: str, effect: str, request_sha: str,
                                source_binding_sha: str, approval: object) -> dict:
    if not isinstance(approval, dict) or set(approval) != APPROVAL_KEYS:
        raise ValueError("approval 기록 계약이 잘못되었습니다")
    actor = approval["actor"]
    if (approval["schema"] != "grounded_active_approval_v1"
            or not isinstance(approval["created_at"], str) or not approval["created_at"]
            or approval["case_id"] != case_id or approval["effect_key"] != effect
            or approval["request_sha256"] != request_sha
            or approval["source_binding_sha256"] != source_binding_sha
            or not isinstance(actor, str) or not actor.strip() or actor != actor.strip()
            or len(actor) > 128
            or approval["approved"] is not True
            or approval["identity_attested"] is not False):
        raise ValueError("approval 기록 의미 결속이 잘못되었습니다")
    return approval


def _validate_consumed_document(case_id: str, effect: str, source_binding_sha: str,
                                consumed: object) -> dict:
    if not isinstance(consumed, dict) or set(consumed) != CONSUMED_KEYS:
        raise ValueError("consumed 기록 계약이 잘못되었습니다")
    request_sha = _require_sha(consumed["request_sha256"], "request_sha256")
    approval_sha = _require_sha(consumed["approval_sha256"], "approval_sha256")
    if (consumed["schema"] != "grounded_active_consumed_v1"
            or not isinstance(consumed["consumed_at"], str) or not consumed["consumed_at"]
            or consumed["case_id"] != case_id or consumed["effect_key"] != effect
            or consumed["case_source_binding_sha256"] != source_binding_sha
            or consumed["side_effect_started"] is not False):
        raise ValueError("consumed 기록 의미 결속이 잘못되었습니다")
    request = _load_bound("active_requests", f"{case_id}__{effect}.json", request_sha)
    approval = _load_bound("active_approvals", f"{case_id}__{effect}.json", approval_sha)
    _validate_request_document(case_id, effect, request)
    _validate_approval_document(
        case_id, effect, request_sha, source_binding_sha, approval)
    if request["source_binding_sha256"] != source_binding_sha:
        raise ValueError("consumed request/source 결속이 잘못되었습니다")
    _load_case_binding(case_id, source_binding_sha)
    return consumed


def active_registry_profile() -> dict:
    return {
        "schema": "grounded_active_registry_v1", "registry_id": ACTIVE_REGISTRY_ID,
        "registry_sha256": _sha(ACTIVE_CHECKS), "checks": deepcopy(ACTIVE_CHECKS),
        "execution_status": "not_requested", "model_may_execute": False,
        "host_execution": {
            "executor_profile": "run_grounded_active",
            "supported_task": "Isaac-Velocity-Flat-Unitree-Go2-v0",
            "executable_check_ids": list(HOST_EXECUTABLE_CHECK_IDS),
            "unsupported_check_result": "HOLD_without_execution",
            "approval_required": True,
            "case_admission_verified": False,
        },
    }


def write_grounded_active_assessment(case_id: str, catalog_sha256: str,
                                     selection_binding: object,
                                     hypotheses: object, next_check: object) -> dict:
    try:
        catalog, selected_ids = G._validated_selection(case_id, catalog_sha256, selection_binding)
        checked_hypotheses, unavailable = G._checked_hypotheses(catalog, hypotheses, set(selected_ids))
        checked_next = G._checked_next_check(
            next_check, {item["id"] for item in checked_hypotheses}, ACTIVE_CHECKS,
        )
        if unavailable:
            return {"status": "hold", "reason": "evidence_unavailable", "unavailable_evidence": unavailable}
    except (FileNotFoundError, PermissionError, ValueError) as exc:
        return {"error": str(exc)}
    if checked_next["registry_id"] not in HOST_EXECUTABLE_CHECK_IDS:
        checked_next["execution_status"] = "host_execution_unsupported"
    document = {
        "schema": "grounded_active_assessment_v1", "case_id": case_id,
        "created_at": _utc(), "status": "hypotheses_only",
        "catalog_sha256": catalog["catalog_sha256"], "telemetry_sha256": catalog["telemetry_sha256"],
        "selection_binding": deepcopy(selection_binding), "selected_evidence_ids": selected_ids,
        "context_status": catalog["context_status"], "premises_verified": True,
        "model_narrative_unverified": True, "hypotheses": checked_hypotheses,
        "next_check": checked_next, "registry_id": ACTIVE_REGISTRY_ID,
        "registry_sha256": _sha(ACTIVE_CHECKS),
    }
    try:
        _, canonical = _write_once("grounded_active_assessments", f"{case_id}.json", document)
    except FileExistsError as exc:
        return {"error": str(exc)}
    return {"saved": f"preregistrations/grounded_active_assessments/{case_id}.json",
            "assessment_sha256": canonical, **document}


def validate_case_manifest(case_id: str, manifest: object) -> dict:
    _case_id(case_id)
    if not isinstance(manifest, dict) or set(manifest) != MANIFEST_KEYS:
        raise ValueError("case manifest 계약이 잘못되었습니다")
    if manifest["schema"] != "grounded_active_case_manifest_v1" or manifest["case_id"] != case_id:
        raise ValueError("case manifest case 결속이 잘못되었습니다")
    alias = manifest["public_case_alias"]
    if not isinstance(alias, str) or SAFE_TEXT.fullmatch(alias) is None:
        raise ValueError("public_case_alias 형식이 잘못되었습니다")
    _revalidate_case_sources(case_id, manifest)
    assessment = _load_bound("grounded_active_assessments", f"{case_id}.json")
    if _sha(assessment) != _require_sha(manifest["assessment_sha256"], "assessment_sha256"):
        raise ValueError("initial assessment SHA가 다릅니다")
    if (assessment.get("schema") != "grounded_active_assessment_v1"
            or assessment.get("case_id") != case_id
            or assessment.get("registry_id") != ACTIVE_REGISTRY_ID
            or assessment.get("registry_sha256") != _sha(ACTIVE_CHECKS)):
        raise ValueError("initial assessment profile 결속이 잘못되었습니다")
    if not isinstance(manifest["task"], str) or not manifest["task"] or type(manifest["seed"]) is not int:
        raise ValueError("task/seed 계약이 잘못되었습니다")
    for field in ("protocol_sha256", "private_executor_binding_sha256"):
        _require_sha(manifest[field], field)
    _require_sha(manifest["source_commit_sha"], "source_commit_sha", commit=True)
    binding = deepcopy(manifest)
    binding["case_source_binding_sha256"] = _sha(manifest)
    filename = f"{case_id}__{binding['case_source_binding_sha256']}.json"
    try:
        _write_once("active_case_bindings", filename, binding)
    except FileExistsError:
        if _load_bound("active_case_bindings", filename) != binding:
            raise ValueError("기존 case source binding 내용이 다릅니다")
    return binding


def request_execution(case_id: str, case_source_binding_sha256: str,
                      registry_id: str, arguments: object) -> dict:
    _case_id(case_id)
    source_binding = _load_case_binding(case_id, case_source_binding_sha256)
    if registry_id != "reset_base_com_stability" or arguments != FIXED_EXECUTION_ARGUMENTS:
        raise ValueError("host fixed execution 계약이 다릅니다")
    assessment = _load_bound("grounded_active_assessments", f"{case_id}.json")
    if (_sha(assessment) != source_binding["assessment_sha256"]
            or assessment.get("next_check", {}).get("registry_id") != registry_id):
        raise ValueError("initial assessment가 요청한 검사와 다릅니다")
    definition = ACTIVE_CHECKS[registry_id]
    components = {
        "case_id": case_id, "registry_id": registry_id,
        "assessment_sha256": _sha(assessment), "registry_sha256": _sha(ACTIVE_CHECKS),
        "definition_sha256": _sha(definition), "source_binding_sha256": case_source_binding_sha256,
        "arguments_sha256": _sha(arguments),
    }
    effect_key = _sha(components)
    document = {"schema": "grounded_active_request_v1", "created_at": _utc(),
                "effect_key": effect_key, **components, "execution_limit": 1}
    _, receipt_sha = _write_once("active_requests", f"{case_id}__{effect_key}.json", document)
    return {"effect_key": effect_key, "request_sha256": receipt_sha,
            "case_source_binding_sha256": case_source_binding_sha256}


def approve_execution(case_id: str, request_binding: object, actor: str) -> dict:
    if (not isinstance(request_binding, dict) or set(request_binding) != {
            "effect_key", "request_sha256", "case_source_binding_sha256"}):
        raise ValueError("request binding 계약이 잘못되었습니다")
    effect = _require_sha(request_binding["effect_key"], "effect_key")
    request_sha = _require_sha(request_binding["request_sha256"], "request_sha256")
    request = _load_bound("active_requests", f"{case_id}__{effect}.json",
                          request_sha)
    _validate_request_document(case_id, effect, request)
    source_binding_sha = _require_sha(
        request_binding["case_source_binding_sha256"], "case_source_binding_sha256")
    if request["source_binding_sha256"] != source_binding_sha:
        raise ValueError("request source binding이 바뀌었습니다")
    if not isinstance(actor, str) or not actor.strip() or len(actor) > 128:
        raise ValueError("approval actor가 잘못되었습니다")
    document = {"schema": "grounded_active_approval_v1", "created_at": _utc(),
                "case_id": case_id, "effect_key": effect,
                "request_sha256": request_sha,
                "source_binding_sha256": source_binding_sha,
                "actor": actor.strip(), "identity_attested": False, "approved": True}
    _, approval_sha = _write_once("active_approvals", f"{case_id}__{effect}.json", document)
    return {"effect_key": effect, "request_sha256": request_sha,
            "approval_sha256": approval_sha,
            "case_source_binding_sha256": source_binding_sha}


def consume_approval(case_id: str, approval_binding: object) -> dict:
    keys = {"effect_key", "request_sha256", "approval_sha256", "case_source_binding_sha256"}
    if not isinstance(approval_binding, dict) or set(approval_binding) != keys:
        raise ValueError("approval binding 계약이 잘못되었습니다")
    effect = _require_sha(approval_binding["effect_key"], "effect_key")
    request_sha = _require_sha(approval_binding["request_sha256"], "request_sha256")
    approval_sha = _require_sha(approval_binding["approval_sha256"], "approval_sha256")
    source_binding_sha = _require_sha(
        approval_binding["case_source_binding_sha256"], "case_source_binding_sha256")
    request = _load_bound(
        "active_requests", f"{case_id}__{effect}.json", request_sha)
    approval = _load_bound(
        "active_approvals", f"{case_id}__{effect}.json", approval_sha)
    _validate_request_document(case_id, effect, request)
    _validate_approval_document(
        case_id, effect, request_sha, source_binding_sha, approval)
    if request["source_binding_sha256"] != source_binding_sha:
        raise ValueError("approval/request/source 결속이 다릅니다")
    _load_case_binding(case_id, source_binding_sha)
    document = {"schema": "grounded_active_consumed_v1", "consumed_at": _utc(),
                "case_id": case_id, **approval_binding, "side_effect_started": False}
    _, consumed_sha = _write_once("active_consumed", f"{case_id}__{effect}.json", document)
    return {**approval_binding, "consumed_sha256": consumed_sha}


def normalize_execution_receipt(case_id: str, case_source_binding_sha256: str,
                                effect_key: str, receipt: object) -> dict:
    source_binding_sha = _require_sha(
        case_source_binding_sha256, "case_source_binding_sha256")
    effect = _require_sha(effect_key, "effect_key")
    consumed = _load_bound("active_consumed", f"{case_id}__{effect}.json")
    _validate_consumed_document(case_id, effect, source_binding_sha, consumed)
    receipt_keys = {
        "schema", "case_id", "effect_key", "case_source_binding_sha256", "request_sha256",
        "approval_sha256", "consumed_sha256", "registry_id", "check_definition_sha256",
        "parent_status", "parent_receipt_sha256", "body_sha256", "natural_exit",
        "cleanup_verified", "source_verified", "coverage_status", "measurements",
        "model_selected", "diagnosis_confirmed", "cause_confirmed", "recovery_confirmed",
        "policy_recovery",
    }
    structurally_valid = isinstance(receipt, dict) and set(receipt) == receipt_keys
    valid = (
        structurally_valid
        and consumed.get("case_id") == case_id
        and consumed.get("effect_key") == effect
        and consumed.get("case_source_binding_sha256") == source_binding_sha
    )
    if valid:
        valid = (
            receipt["schema"] == "grounded_active_execution_receipt_v1"
            and receipt["case_id"] == case_id and receipt["effect_key"] == effect
            and receipt["case_source_binding_sha256"] == source_binding_sha
            and receipt["request_sha256"] == consumed["request_sha256"]
            and receipt["approval_sha256"] == consumed["approval_sha256"]
            and receipt["consumed_sha256"] == _sha(consumed)
            and receipt["registry_id"] == "reset_base_com_stability"
            and receipt["check_definition_sha256"] == _sha(ACTIVE_CHECKS["reset_base_com_stability"])
            and receipt["parent_status"] == "pass" and receipt["natural_exit"] is True
            and receipt["cleanup_verified"] is True and receipt["source_verified"] is True
            and receipt["coverage_status"] == "complete"
            and all(receipt[key] is False for key in (
                "model_selected", "diagnosis_confirmed", "cause_confirmed",
                "recovery_confirmed", "policy_recovery"))
        )
        try:
            _require_sha(receipt["parent_receipt_sha256"], "parent_receipt_sha256")
            _require_sha(receipt["body_sha256"], "body_sha256")
        except ValueError:
            valid = False
    measurement_map = {}
    if valid and isinstance(receipt["measurements"], list) and len(receipt["measurements"]) == 2:
        for row in receipt["measurements"]:
            if not isinstance(row, dict) or set(row) != {"measurement_id", "unit", "state", "value"}:
                valid = False
                break
            measurement_map[row["measurement_id"]] = row
    else:
        valid = False
    expected = ACTIVE_CHECKS["reset_base_com_stability"]["measurements"]
    if set(measurement_map) != set(expected):
        valid = False
    if valid:
        for measurement_id, definition in expected.items():
            row = measurement_map[measurement_id]
            value = row["value"]
            if (row["unit"] != definition["unit"] or row["state"] != "present"
                    or type(value) not in (int, float) or not math.isfinite(float(value))):
                valid = False
                break
            if measurement_id == "base_com_max_abs_delta_xy" and value < 0:
                valid = False
                break
            if measurement_id.endswith("fraction") and not 0 <= value <= 1:
                valid = False
                break
    normalized_measurements = [{
        "measurement_id": measurement_id, "unit": definition["unit"],
        "state": "present" if valid else "unavailable",
        "value": measurement_map[measurement_id]["value"] if valid else None,
    } for measurement_id, definition in sorted(expected.items())]
    normalized = {
        "schema": "grounded_active_normalized_receipt_v1", "case_id": case_id,
        "effect_key": effect, "case_source_binding_sha256": source_binding_sha,
        "registry_id": "reset_base_com_stability",
        "check_definition_sha256": _sha(ACTIVE_CHECKS["reset_base_com_stability"]),
        "execution_valid": valid, "coverage_status": "complete" if valid else "unavailable",
        "measurements": normalized_measurements, "model_selected": False,
        "diagnosis_confirmed": False, "cause_confirmed": False,
        "recovery_confirmed": False, "policy_recovery": False,
        "judgment_only_not_correctness_or_recovery": True,
    }
    _, normalized_sha = _write_once("active_receipts", f"{case_id}__{effect}.json", normalized)
    return {**normalized, "normalized_receipt_sha256": normalized_sha}


def _post_packet(case_id: str, effect: str, stored: dict, value_masked: bool) -> dict:
    assessment = _load_bound("grounded_active_assessments", f"{case_id}.json")
    hypotheses = []
    selected_evidence = {}
    prediction_rows = {row["hypothesis_id"]: row["predictions"]
                       for row in assessment["next_check"]["hypothesis_predictions"]}
    for hypothesis in sorted(assessment["hypotheses"], key=lambda item: item["id"]):
        hypotheses.append({"hypothesis_id": hypothesis["id"],
                           "preregistered_predictions": prediction_rows[hypothesis["id"]]})
        for premise in hypothesis.get("premises", []):
            selected_evidence[premise["evidence_id"]] = {
                "evidence_id": premise["evidence_id"], "value": premise["actual"],
                "unit": premise["operand"]["unit"],
            }
    receipt = deepcopy(stored)
    if value_masked:
        for row in receipt["measurements"]:
            row["value"] = None
    return {
        "schema": "grounded_active_post_packet_v1", "case_id": case_id,
        "effect_key": effect, "budget_left": 0,
        "normalized_receipt_sha256": _sha(stored),
        "hypotheses": hypotheses,
        "selected_evidence_ids": assessment["selected_evidence_ids"],
        "selected_evidence": [selected_evidence[key] for key in sorted(selected_evidence)],
        "check_definition": deepcopy(ACTIVE_CHECKS["reset_base_com_stability"]),
        "normalized_receipt": receipt,
    }


def build_post_measurement_packet(case_id: str, normalized_receipt: object,
                                  value_masked: bool = False) -> dict:
    if not isinstance(normalized_receipt, dict) or type(value_masked) is not bool:
        raise ValueError("normalized receipt와 value_masked bool이 필요합니다")
    effect = _require_sha(normalized_receipt.get("effect_key"), "effect_key")
    stored = _load_bound("active_receipts", f"{case_id}__{effect}.json")
    supplied = {key: value for key, value in normalized_receipt.items()
                if key != "normalized_receipt_sha256"}
    if supplied != stored:
        raise ValueError("normalized receipt가 저장된 최신 receipt와 다릅니다")
    return _post_packet(case_id, effect, stored, value_masked)


def validate_final_decision(packet: object, decision: object) -> dict:
    packet_keys = {
        "schema", "case_id", "effect_key", "budget_left", "hypotheses",
        "selected_evidence_ids", "selected_evidence", "check_definition", "normalized_receipt",
        "normalized_receipt_sha256",
    }
    if (not isinstance(packet, dict) or set(packet) != packet_keys
            or packet.get("schema") != "grounded_active_post_packet_v1"):
        raise ValueError("post packet 계약이 잘못되었습니다")
    case_id = packet.get("case_id")
    effect = _require_sha(packet.get("effect_key"), "effect_key")
    stored = _load_bound("active_receipts", f"{case_id}__{effect}.json")
    if packet not in (_post_packet(case_id, effect, stored, False),
                      _post_packet(case_id, effect, stored, True)):
        raise ValueError("post packet이 저장된 최신 assessment/receipt와 다릅니다")
    if not isinstance(decision, dict) or set(decision) != DECISION_KEYS:
        raise ValueError("final decision 계약이 잘못되었습니다")
    if (decision["schema"] != "grounded_active_final_decision_v1"
            or decision["case_id"] != packet["case_id"]
            or decision["effect_key"] != packet["effect_key"]
            or decision["budget_left"] != 0 or decision["status"] != "HOLD"):
        raise ValueError("final decision 결속/HOLD 계약이 잘못되었습니다")
    ids = [row["hypothesis_id"] for row in packet["hypotheses"]]
    ranking = decision["ranking"]
    if (not isinstance(ranking, list) or any(not isinstance(item, str) for item in ranking)
            or len(ranking) != len(ids) or len(set(ranking)) != len(ranking)
            or set(ranking) != set(ids)):
        raise ValueError("final ranking은 hypothesis id를 정확히 한 번씩 포함해야 합니다")
    citation = decision["receipt_citation"]
    if citation != {"effect_key": packet["effect_key"],
                    "normalized_receipt_sha256": packet["normalized_receipt_sha256"]}:
        raise ValueError("latest normalized receipt를 정확히 인용해야 합니다")
    return deepcopy(decision)


def compare_decision_rankings(initial_assessment: dict,
                              actual_decision: dict, masked_decision: dict) -> dict:
    initial_top = min(initial_assessment["hypotheses"], key=lambda row: row["rank"])["id"]
    actual_top = actual_decision["ranking"][0]
    masked_top = masked_decision["ranking"][0]
    return {
        "initial_top": initial_top, "actual_top": actual_top, "masked_top": masked_top,
        "ranking_changed_with_values": actual_top != initial_top,
        "fixed_no_update": masked_top == initial_top,
    }


def build_judgment_influence_proof(initial_assessment: object,
                                   actual_packet: dict, actual_decision: dict,
                                   masked_packet: dict, masked_decision: dict) -> dict:
    actual = validate_final_decision(actual_packet, actual_decision)
    masked = validate_final_decision(masked_packet, masked_decision)
    case_id = actual_packet.get("case_id")
    effect = actual_packet.get("effect_key")
    stored_receipt = _load_bound("active_receipts", f"{case_id}__{effect}.json")
    actual_role_valid = actual_packet == _post_packet(case_id, effect, stored_receipt, False)
    masked_role_valid = masked_packet == _post_packet(case_id, effect, stored_receipt, True)
    actual_values = [row["value"] for row in actual_packet["normalized_receipt"]["measurements"]]
    masked_values = [row["value"] for row in masked_packet["normalized_receipt"]["measurements"]]
    values_valid = (
        all(type(value) in (int, float) and math.isfinite(float(value)) for value in actual_values)
        and all(value is None for value in masked_values)
    )
    stored_assessment = _load_bound(
        "grounded_active_assessments", f"{case_id}.json")
    if (not isinstance(initial_assessment, dict) or initial_assessment != stored_assessment
            or not actual_role_valid or not masked_role_valid or not values_valid
            or actual_packet["effect_key"] != masked_packet["effect_key"]
            or not actual_packet["normalized_receipt"]["execution_valid"]
            or not masked_packet["normalized_receipt"]["execution_valid"]):
        return {"judgment_influence_proven": False, "reason": "invalid_or_unbound_inputs"}
    actual_copy, masked_copy = deepcopy(actual_packet), deepcopy(masked_packet)
    for row in actual_copy["normalized_receipt"]["measurements"]:
        row["value"] = None
    only_values_differ = actual_copy == masked_copy
    comparison = compare_decision_rankings(initial_assessment, actual, masked)
    initial_top = comparison["initial_top"]
    new_top, masked_top = comparison["actual_top"], comparison["masked_top"]
    predictions = {row["hypothesis_id"]: row["predictions"]
                   for row in initial_assessment["next_check"]["hypothesis_predictions"]}
    measured = {row["measurement_id"]: row for row in actual_packet["normalized_receipt"]["measurements"]}
    def survives(hypothesis_id: str) -> bool:
        return all(G._relation_holds(measured[row["measurement_id"]]["value"], row["relation"],
                                     row["operand"]["value"])
                   for row in predictions[hypothesis_id])
    rule_update = (not survives(initial_top) and survives(new_top)
                   and new_top != initial_top and masked_top == initial_top)
    return {
        "judgment_influence_proven": bool(only_values_differ and rule_update),
        "only_measurement_values_differ": only_values_differ,
        "actual_receipt_cited": actual["receipt_citation"],
        "initial_top": initial_top, "actual_top": new_top, "masked_top": masked_top,
        "rule_update": rule_update, "fixed_no_update": comparison["fixed_no_update"],
        "correctness_proven": False, "superiority_proven": False,
        "policy_recovery_proven": False,
    }
