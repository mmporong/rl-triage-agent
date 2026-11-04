"""Source-bound normal/intervention governance without execution authority."""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from .correction_registry import REGISTRY

SOURCE_SCHEMA = "intervention_observation_sources_v1"
NORMALIZED_SCHEMA = "intervention_observation_normalized_v1"
DECISION_SCHEMA = "intervention_governor_decision_v1"
MODEL_SCHEMA = "intervention_model_proposal_v1"
HOST_SOURCE_SCHEMA = "intervention_host_gate_source_v1"
FIXTURE_VERIFIER_ID = "cpu_fixture_fixed_eval_bundle_v1"
_SHA_RE = re.compile(r"[0-9a-f]{64}")
_HIDDEN_KEYS = frozenset({"fault_label", "fault_labels", "benchmark_score", "benchmark_scoring",
                          "original_faults", "remaining_faults"})
_ACTIONS = frozenset({"no_intervention", "investigate", "correct", "reference_review"})
_METRICS = ("fall_rate", "mean_survival_s", "lin_vel_rmse_mps", "yaw_rate_rmse_radps")
_SEAL = object()


@dataclass(frozen=True, slots=True)
class VerifiedObservation:
    """Opaque value produced only after registered source verification."""
    _payload: dict
    _payload_sha256: str
    _seal: object

    def __post_init__(self) -> None:
        if self._seal is not _SEAL:
            raise TypeError("VerifiedObservation must come from verify_observation_receipt")
        if self._payload_sha256 != _canonical_sha(self._payload):
            raise TypeError("VerifiedObservation payload integrity check failed")


@dataclass(frozen=True, slots=True)
class VerifiedHostGate:
    """Opaque host policy envelope; it is not an execution approval."""
    _payload: dict
    _payload_sha256: str
    _seal: object

    def __post_init__(self) -> None:
        if self._seal is not _SEAL:
            raise TypeError("VerifiedHostGate must come from verify_host_gate_receipt")
        if self._payload_sha256 != _canonical_sha(self._payload):
            raise TypeError("VerifiedHostGate payload integrity check failed")


def _canonical_sha(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(raw).hexdigest()


def _verified_observation(payload: dict) -> VerifiedObservation:
    """Seal a normalized bundle produced by a registered in-package verifier."""
    if not isinstance(payload, dict) or payload.get("schema") != NORMALIZED_SCHEMA:
        raise ValueError("registered verifier did not produce a normalized observation bundle")
    payload = deepcopy(payload)
    return VerifiedObservation(payload, _canonical_sha(payload), _SEAL)


def _reject_hidden(value: object, path: str = "input") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} contains a non-string key")
            if key.lower() in _HIDDEN_KEYS:
                raise ValueError(f"{path}.{key} is hidden benchmark information")
            _reject_hidden(child, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _reject_hidden(child, f"{path}[{index}]")


def _exact(value: object, keys: set[str], label: str) -> Mapping:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"{label} fields do not match the registered contract")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _sha(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA256")
    return value


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _refs(value: object, label: str) -> list[str]:
    if (not isinstance(value, list) or not value
            or any(not isinstance(item, str) or not item.strip() for item in value)
            or len(set(value)) != len(value)):
        raise ValueError(f"{label} must contain unique evidence references")
    return list(value)


def _source_ref(value: object, label: str) -> dict | None:
    if value is None:
        return None
    row = _exact(value, {"path", "sha256"}, label)
    return {"path": _text(row["path"], f"{label}.path"),
            "sha256": _sha(row["sha256"], f"{label}.sha256")}


def _load_source(ref: dict, label: str) -> dict:
    path = Path(ref["path"])
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"{label} source is unavailable") from exc
    if hashlib.sha256(raw).hexdigest() != ref["sha256"]:
        raise ValueError(f"{label} source SHA does not match")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} source is not UTF-8 JSON") from exc
    _reject_hidden(value, label)
    if not isinstance(value, dict):
        raise TypeError(f"{label} source must be a JSON object")
    return value


def _verify_eval(value: dict, role: str, criteria_sha: str) -> dict:
    keys = {"schema", "role", "task", "protocol", "evaluation_criteria_sha256",
            "source_binding_sha256", "seed", "metrics", "refs"}
    row = _exact(value, keys, f"{role}_eval")
    if row["schema"] != "cpu_fixture_fixed_eval_result_v1" or row["role"] != role:
        raise ValueError(f"{role}_eval is not a registered CPU fixture result")
    if row["evaluation_criteria_sha256"] != criteria_sha:
        raise ValueError(f"{role}_eval criteria binding does not match")
    if isinstance(row["seed"], bool) or not isinstance(row["seed"], int) or row["seed"] < 0:
        raise ValueError(f"{role}_eval.seed must be a non-negative integer")
    metrics = _exact(row["metrics"], set(_METRICS), f"{role}_eval.metrics")
    return {"task": _text(row["task"], f"{role}_eval.task"),
            "protocol": _text(row["protocol"], f"{role}_eval.protocol"),
            "source_binding_sha256": _sha(row["source_binding_sha256"], f"{role}_eval.source_binding_sha256"),
            "seed": row["seed"], "metrics": {name: _number(metrics[name], name) for name in _METRICS},
            "refs": _refs(row["refs"], f"{role}_eval.refs")}


def _verify_criteria(value: dict) -> dict:
    row = _exact(value, {"schema", "task", "protocol", "source_binding_sha256", "metrics", "refs"},
                 "evaluation_criteria")
    if row["schema"] != "cpu_fixture_evaluation_criteria_v1":
        raise ValueError("evaluation criteria is not registered for the CPU fixture verifier")
    metrics = _exact(row["metrics"], set(_METRICS), "evaluation_criteria.metrics")
    normalized = {}
    for name in _METRICS:
        spec = _exact(metrics[name], {"direction", "absolute", "relative"}, f"criteria.{name}")
        if spec["direction"] not in {"min", "max"}:
            raise ValueError(f"criteria.{name}.direction is unsupported")
        normalized[name] = {"direction": spec["direction"],
                            "absolute": _number(spec["absolute"], f"criteria.{name}.absolute"),
                            "relative": _number(spec["relative"], f"criteria.{name}.relative")}
    return {"task": _text(row["task"], "criteria.task"),
            "protocol": _text(row["protocol"], "criteria.protocol"),
            "source_binding_sha256": _sha(row["source_binding_sha256"], "criteria.source_binding_sha256"),
            "metrics": normalized, "refs": _refs(row["refs"], "criteria.refs")}


def _metric_checks(candidate: dict, reference: dict, criteria: dict) -> tuple[str, list[dict]]:
    checks, absolute_failed, relative_failed = [], False, False
    for name in _METRICS:
        value, baseline, spec = candidate["metrics"][name], reference["metrics"][name], criteria["metrics"][name]
        if spec["direction"] == "max":
            absolute_passed, relative_passed = value <= spec["absolute"], value <= baseline * spec["relative"]
        else:
            absolute_passed, relative_passed = value >= spec["absolute"], value >= baseline * spec["relative"]
        absolute_failed |= not absolute_passed
        relative_failed |= not relative_passed
        checks.append({"metric": name, "value": value, "reference_value": baseline,
                       "direction": spec["direction"], "absolute_passed": absolute_passed,
                       "relative_passed": relative_passed})
    outcome = "failed" if absolute_failed else "below_reference" if relative_failed else "normal"
    return outcome, checks


def _verify_check(value: dict, expected_binding: str) -> dict:
    row = _exact(value, {"schema", "check_id", "source_binding_sha256", "outcome", "refs"},
                 "confirmation_check")
    if row["schema"] != "cpu_fixture_confirmation_check_v1":
        raise ValueError("confirmation check is not a registered CPU fixture receipt")
    if row["source_binding_sha256"] != expected_binding:
        raise ValueError("confirmation check source binding does not match")
    if row["outcome"] not in {"normal", "abnormal", "unknown"}:
        raise ValueError("confirmation check outcome is unsupported")
    return {"check_id": _text(row["check_id"], "confirmation_check.check_id"),
            "outcome": row["outcome"], "refs": _refs(row["refs"], "confirmation_check.refs")}


def _verify_cause(value: dict, expected_binding: str) -> dict:
    row = _exact(value, {"schema", "confirmation_id", "source_binding_sha256", "status",
                         "mechanism", "registered_correction_id", "refs"}, "cause_confirmation")
    if row["schema"] != "cause_confirmation_receipt_v1" or row["source_binding_sha256"] != expected_binding:
        raise ValueError("cause confirmation source binding or schema does not match")
    if row["status"] not in {"confirmed", "not_confirmed", "unknown"}:
        raise ValueError("cause confirmation status is unsupported")
    correction_id = row["registered_correction_id"]
    registered = REGISTRY.get(row["mechanism"])
    if (row["status"] == "confirmed"
            and (registered is None or correction_id != registered["correction_id"])):
        raise ValueError("confirmed cause does not name its registered correction")
    return {"confirmation_id": _text(row["confirmation_id"], "confirmation_id"),
            "status": row["status"], "mechanism": _text(row["mechanism"], "mechanism"),
            "registered_correction_id": None if correction_id is None else _text(correction_id, "correction_id"),
            "refs": _refs(row["refs"], "cause_confirmation.refs")}


def _verify_correction(value: dict, expected_binding: str) -> dict:
    row = _exact(value, {"schema", "correction_id", "source_binding_sha256", "status", "refs"},
                 "correction_receipt")
    if row["schema"] != "correction_execution_receipt_v1" or row["source_binding_sha256"] != expected_binding:
        raise ValueError("correction receipt source binding or schema does not match")
    if row["status"] not in {"applied", "failed", "unknown"}:
        raise ValueError("correction receipt status is unsupported")
    if row["correction_id"] not in {entry["correction_id"] for entry in REGISTRY.values()}:
        raise ValueError("correction receipt does not name a registered correction")
    return {"correction_id": _text(row["correction_id"], "correction_id"), "status": row["status"],
            "refs": _refs(row["refs"], "correction_receipt.refs")}


def verify_observation_receipt(raw_receipt: object, *, verifier_id: str = FIXTURE_VERIFIER_ID) -> VerifiedObservation:
    """Verify source artifacts; the bundled verifier is CPU-fixture-only, not a new seed measurement."""
    _reject_hidden(raw_receipt)
    if verifier_id != FIXTURE_VERIFIER_ID:
        raise ValueError("unregistered observation verifier")
    keys = {"schema", "receipt_id", "candidate_eval", "reference_eval", "evaluation_criteria",
            "confirmation_checks", "cause_confirmation_receipt", "correction_receipt"}
    row = _exact(raw_receipt, keys, "observation_sources")
    if row["schema"] != SOURCE_SCHEMA:
        raise ValueError("observation source schema is unsupported")
    receipt_id = _text(row["receipt_id"], "receipt_id")
    candidate_ref = _source_ref(row["candidate_eval"], "candidate_eval")
    criteria_ref = _source_ref(row["evaluation_criteria"], "evaluation_criteria")
    reference_ref = _source_ref(row["reference_eval"], "reference_eval")
    if candidate_ref is None or criteria_ref is None:
        raise ValueError("candidate eval and evaluation criteria sources are required")
    criteria = _verify_criteria(_load_source(criteria_ref, "evaluation_criteria"))
    candidate = _verify_eval(_load_source(candidate_ref, "candidate_eval"), "candidate", criteria_ref["sha256"])
    reference, behavior_outcome, metric_checks, lineage_matches = None, "unknown", [], False
    if reference_ref is not None:
        reference = _verify_eval(_load_source(reference_ref, "reference_eval"), "reference", criteria_ref["sha256"])
        lineage_matches = all((candidate["task"] == reference["task"] == criteria["task"],
                               candidate["protocol"] == reference["protocol"] == criteria["protocol"],
                               candidate["source_binding_sha256"] == reference["source_binding_sha256"]
                               == criteria["source_binding_sha256"]))
        if lineage_matches:
            behavior_outcome, metric_checks = _metric_checks(candidate, reference, criteria)
    check_refs = row["confirmation_checks"]
    if not isinstance(check_refs, list):
        raise TypeError("confirmation_checks must be a list")
    checks, seen = [], set()
    for index, raw_ref in enumerate(check_refs):
        ref = _source_ref(raw_ref, f"confirmation_checks[{index}]")
        if ref is None:
            raise ValueError("confirmation check source cannot be null")
        check = _verify_check(_load_source(ref, f"confirmation_checks[{index}]"), candidate["source_binding_sha256"])
        if check["check_id"] in seen:
            raise ValueError("duplicate confirmation check")
        seen.add(check["check_id"])
        checks.append(check)
    cause_ref = _source_ref(row["cause_confirmation_receipt"], "cause_confirmation_receipt")
    correction_ref = _source_ref(row["correction_receipt"], "correction_receipt")
    cause = None if cause_ref is None else _verify_cause(
        _load_source(cause_ref, "cause_confirmation_receipt"), candidate["source_binding_sha256"])
    correction = None if correction_ref is None else _verify_correction(
        _load_source(correction_ref, "correction_receipt"), candidate["source_binding_sha256"])
    if (cause is not None and correction is not None and cause["registered_correction_id"] is not None
            and correction["correction_id"] != cause["registered_correction_id"]):
        raise ValueError("correction receipt does not match the confirmed cause")
    payload = {"schema": NORMALIZED_SCHEMA, "receipt_id": receipt_id, "verifier_id": verifier_id,
               "verification_scope": "cpu_fixture_development_only", "candidate": candidate,
               "reference": reference, "criteria": criteria, "reference_available": reference is not None,
               "lineage_matches": lineage_matches, "behavior_outcome": behavior_outcome,
               "metric_checks": metric_checks, "confirmation_checks": checks,
               "cause_confirmation_receipt": cause, "correction_receipt": correction,
               "source_refs": {"candidate_eval": candidate_ref, "reference_eval": reference_ref,
                               "evaluation_criteria": criteria_ref}}
    return _verified_observation(payload)


def verify_host_gate_receipt(raw_source: object) -> VerifiedHostGate:
    """Verify a source-bound host policy envelope without granting approval."""
    _reject_hidden(raw_source, "host_gate_source")
    row = _exact(raw_source, {"schema", "receipt"}, "host_gate_source")
    if row["schema"] != HOST_SOURCE_SCHEMA:
        raise ValueError("host gate source schema is unsupported")
    ref = _source_ref(row["receipt"], "host_gate_receipt")
    if ref is None:
        raise ValueError("host gate receipt source is required")
    gate = _exact(_load_source(ref, "host_gate_receipt"),
                  {"schema", "gate_id", "allowed_actions", "refs"}, "host_gate_receipt")
    if gate["schema"] != "host_gate_receipt_v1":
        raise ValueError("host gate receipt schema is unsupported")
    allowed = gate["allowed_actions"]
    if (not isinstance(allowed, list) or len(set(allowed)) != len(allowed)
            or any(action not in _ACTIONS for action in allowed)):
        raise ValueError("host gate allowed_actions is invalid")
    payload = {"gate_id": _text(gate["gate_id"], "gate_id"), "allowed_actions": list(allowed),
               "refs": _refs(gate["refs"], "host_gate.refs"), "source_ref": ref}
    payload = deepcopy(payload)
    return VerifiedHostGate(payload, _canonical_sha(payload), _SEAL)


def _proposal(value: object) -> dict | None:
    if value is None:
        return None
    _reject_hidden(value, "raw_model_proposal")
    row = _exact(value, {"schema", "action", "reason", "refs"}, "raw_model_proposal")
    if row["schema"] != MODEL_SCHEMA or row["action"] not in _ACTIONS:
        raise ValueError("model proposal schema or action is unsupported")
    refs = row["refs"]
    if not isinstance(refs, list) or any(not isinstance(ref, str) or not ref for ref in refs):
        raise ValueError("model proposal refs are invalid")
    return {"schema": MODEL_SCHEMA, "action": row["action"],
            "reason": _text(row["reason"], "model reason"), "refs": list(refs)}


def build_model_packet(observation: VerifiedObservation) -> dict:
    if not isinstance(observation, VerifiedObservation) or observation._seal is not _SEAL:
        raise TypeError("verified observation is required")
    if observation._payload_sha256 != _canonical_sha(observation._payload):
        raise ValueError("verified observation was mutated")
    return {"schema": "intervention_model_packet_v1", "observation": deepcopy(observation._payload),
            "allowed_proposal_actions": sorted(_ACTIONS), "execution_authority": False}


def decision_messages(observation: VerifiedObservation) -> list[dict[str, str]]:
    packet = build_model_packet(observation)
    return [{"role": "system", "content": (
                "Assess observable evidence only. Do not infer hidden labels or execution authority. "
                "Return one intervention_model_proposal_v1 JSON object.")},
            {"role": "user", "content": json.dumps(packet, sort_keys=True, separators=(",", ":"))}]


def decide_intervention(observation: VerifiedObservation, raw_model_proposal: object,
                        host_gate: VerifiedHostGate) -> dict:
    """Make a pure proposal decision; approvals remain owned by the existing host."""
    if not isinstance(observation, VerifiedObservation) or observation._seal is not _SEAL:
        raise TypeError("verified observation is required")
    if not isinstance(host_gate, VerifiedHostGate) or host_gate._seal is not _SEAL:
        raise TypeError("verified host gate is required")
    if observation._payload_sha256 != _canonical_sha(observation._payload):
        raise ValueError("verified observation was mutated")
    if host_gate._payload_sha256 != _canonical_sha(host_gate._payload):
        raise ValueError("verified host gate was mutated")
    value, gate, model = observation._payload, host_gate._payload, _proposal(raw_model_proposal)
    checks, correction, cause = (value["confirmation_checks"], value["correction_receipt"],
                                 value["cause_confirmation_receipt"])
    requested, status, reason = "no_intervention", "host_no_intervention", "verified_normal_bundle"
    investigation_need, correctability, additional = "not_needed", "not_applicable", []
    if not value["reference_available"]:
        requested, status, reason = "investigate", "hold", "reference_missing"
        investigation_need, additional = "undetermined", ["provide_source_bound_reference"]
    elif not value["lineage_matches"]:
        requested, status, reason = "investigate", "hold", "reference_lineage_mismatch"
        investigation_need, additional = "undetermined", ["match_task_protocol_criteria_and_source_binding"]
    elif not checks:
        requested, status, reason = "investigate", "hold", "confirmation_checks_missing"
        investigation_need, additional = "undetermined", ["run_registered_confirmation_checks"]
    else:
        unresolved = [row["check_id"] for row in checks if row["outcome"] == "unknown"]
        abnormal = [row["check_id"] for row in checks if row["outcome"] == "abnormal"]
        all_normal = all(row["outcome"] == "normal" for row in checks)
        if unresolved:
            requested, status, reason = "investigate", "hold", "confirmation_check_unknown"
            investigation_need = "undetermined"
            additional = [f"resolve_check:{check_id}" for check_id in unresolved]
        elif (correction is not None and correction["status"] == "applied" and all_normal
              and value["behavior_outcome"] == "below_reference"):
            requested, status, reason = ("reference_review", "reference_review_required",
                                         "post_correction_reference_only_gap")
            investigation_need, correctability = "not_needed", "reference_review_only"
            additional = ["review_reference_and_evaluation_criteria"]
            if (value.get("post_confirmation_scope") is not None
                    and value.get("confirmation_catalog_complete") is not True):
                reason = "post_correction_relative_gap_with_scoped_checks"
                investigation_need, correctability = "undetermined", "reference_review_first"
                additional.append("verify_unobserved_registered_components")
        elif value["behavior_outcome"] == "normal" and all_normal:
            pass
        elif value["behavior_outcome"] == "unknown":
            requested, status, reason = "investigate", "hold", "behavior_bundle_unknown"
            investigation_need, additional = "undetermined", ["obtain_comparable_fixed_eval_bundle"]
        elif abnormal or value["behavior_outcome"] in {"failed", "below_reference"}:
            investigation_need = "needed"
            if (cause is not None and cause["status"] == "confirmed"
                    and cause["registered_correction_id"] is not None):
                requested, status, reason = "correct", "proposal_eligible", "confirmed_cause_registered_correction"
                correctability = "receipt_confirmed"
            else:
                requested, status, reason = "investigate", "investigation_required", "failure_without_confirmed_cause"
                correctability, additional = "unconfirmed", ["obtain_cause_confirmation_receipt"]
    host_blocked = requested not in gate["allowed_actions"]
    effective = requested if not host_blocked and status != "hold" else "hold"
    if host_blocked:
        status, reason = "hold", f"host_gate_blocked:{reason}"
        additional = list(dict.fromkeys(additional + [f"host_allow:{requested}"]))
    refs = list(dict.fromkeys(value["candidate"]["refs"] + value["criteria"]["refs"]
        + ([] if value["reference"] is None else value["reference"]["refs"])
        + [ref for row in checks for ref in row["refs"]]
        + ([] if cause is None else cause["refs"])
        + ([] if correction is None else correction["refs"]) + gate["refs"]))
    return {"schema": DECISION_SCHEMA, "status": status, "investigation_need": investigation_need,
            "correctability": correctability, "raw_model_proposal": deepcopy(model),
            "host_gate_id": gate["gate_id"], "host_blocked": host_blocked,
            "requested_action": requested, "effective_action": effective, "reason_code": reason,
            "evidence_refs": refs, "additional_checks": additional,
            "proposal_eligible": status == "proposal_eligible" and not host_blocked,
            "approval_granted": False, "authority_created": False,
            "verification_scope": value["verification_scope"]}
