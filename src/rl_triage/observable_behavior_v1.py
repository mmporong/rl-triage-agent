"""절대 보행 자격과 상대 fixed26 band를 분리하는 관측 검증기."""
from __future__ import annotations

import ast
import hashlib
import json
import math
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path

from . import grounded_recovery as FIXED
from . import intervention_governor as GOVERNOR
from . import recovery as RECOVERY

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "bench/protocols/observable_behavior_qualification_v1.json"
VERIFIER_ID = "observable_fixed26_walk_qualification_v1"
RULE_ID = "fixed26_walks_then_p0c_relative_bands_v1"
_BAND_KEYS = {"survival_ratio_min", "fall_frac_max", "err_xy_ratio_max", "err_yaw_ratio_max"}
_HIDDEN_KEYS = {"classification", "source_type", "fault_label", "fault_labels", "original_faults", "remaining_faults"}


def _sha(path: Path, *, lf: bool = False) -> str:
    raw = path.read_bytes()
    if lf:
        raw = raw.replace(b"\r\n", b"\n")
    return hashlib.sha256(raw).hexdigest()


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda token: (
            _ for _ in ()).throw(ValueError(f"non-finite JSON number: {token}")))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"JSON source is unavailable or invalid: {path}") from exc
    if not isinstance(value, dict):
        raise TypeError("JSON source must be an object")
    return value


def _reject_hidden(value: object, label: str = "value") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{label} contains a non-string key")
            if key.lower() in _HIDDEN_KEYS:
                raise ValueError(f"{label}.{key} contains hidden benchmark information")
            _reject_hidden(child, f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_hidden(child, f"{label}[{index}]")


def _function_sha(path: Path, function: str, lines: list[int]) -> str:
    source = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    tree = ast.parse(source)
    matches = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == function]
    if len(matches) != 1 or [matches[0].lineno, matches[0].end_lineno] != lines:
        raise ValueError("registered qualification function location changed")
    fragment = "\n".join(source.splitlines()[lines[0] - 1:lines[1]]) + "\n"
    return hashlib.sha256(fragment.encode()).hexdigest()


def _contract() -> tuple[dict, str]:
    value = _read_json(PROTOCOL)
    expected = {"schema", "rule_id", "scope", "absolute_qualification", "relative_qualification",
                "development_evidence", "outcomes", "claims"}
    if set(value) != expected or value["schema"] != "observable_behavior_qualification_v1" \
            or value["rule_id"] != RULE_ID:
        raise ValueError("observable behavior qualification protocol changed")
    absolute = value["absolute_qualification"]
    source = absolute["source"]
    source_path = ROOT / source["path"]
    implementation = absolute["implementation_source"]
    if (_sha(source_path, lf=True) != source["sha256_lf"]
            or _function_sha(source_path, "walks", source["function_lines"])
            != source["function_sha256_lf"]
            or _sha(ROOT / implementation["path"], lf=True) != implementation["sha256_lf"]
            or absolute["lin_error_vs_standstill_strict_max"] != FIXED.WALK_FRACTION
            or absolute["fall_frac_max"] != FIXED.REF_FALL_MAX):
        raise ValueError("registered absolute walk qualification changed")
    relative = value["relative_qualification"]
    summary_ref, fixed_ref = relative["summary"], relative["fixed26_protocol"]
    summary_path, fixed_path = ROOT / summary_ref["path"], ROOT / fixed_ref["path"]
    summary = _read_json(summary_path)
    if (_sha(summary_path) != summary_ref["sha256"]
            or _sha(summary_path, lf=True) != summary_ref["sha256_lf"]
            or _sha(fixed_path, lf=True) != fixed_ref["sha256_lf"]
            or summary.get("bands") != relative["bands"]
            or set(relative["bands"]) != _BAND_KEYS):
        raise ValueError("registered relative behavior criteria changed")
    return value, _sha(PROTOCOL)


def _unwrap(raw: object, label: str) -> dict:
    if not isinstance(raw, dict):
        raise TypeError(f"{label} fixed26 report is missing")
    _reject_hidden(raw, label)
    if raw.get("schema") == "lineage_source_collection_public_v2_fixed26":
        if set(raw) != {"schema", "source_id", "conditions", "report", "execution"} \
                or raw["conditions"] != 26:
            raise ValueError(f"{label} fixed26 wrapper differs")
        execution = raw["execution"]
        if (not isinstance(execution, dict) or execution.get("exit_code") != 0
                or execution.get("natural_exit") is not True
                or execution.get("cleanup_verified") is not True
                or execution.get("timed_out") is not False or execution.get("error_type") is not None):
            raise ValueError(f"{label} fixed26 execution is not a verified completion")
        source_id = raw["source_id"]
        raw = raw["report"]
        if not isinstance(raw, dict) or raw.get("name") != source_id:
            raise ValueError(f"{label} wrapper source ID differs from its report")
    if not isinstance(raw, dict):
        raise TypeError(f"{label} report is missing")
    checkpoint = raw.get("checkpoint")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"{label} checkpoint binding is missing")
    expected = {"name": raw.get("name"), "checkpoint_file": checkpoint.get("file"),
                "checkpoint_sha256": checkpoint.get("sha256"), "action_scale": raw.get("action_scale"),
                "protocol_sha256_lf": raw.get("protocol_sha256_lf")}
    return FIXED.validate_fixed_report(raw, expected)


def _absolute(report: dict) -> dict:
    behavior = FIXED._behavior(report)
    standstill = report.get("standstill")
    if not isinstance(standstill, dict) or "lin_vel_rmse_mps" not in standstill:
        raise ValueError("standstill linear RMSE is missing")
    stationary = standstill["lin_vel_rmse_mps"]
    if isinstance(stationary, bool) or not isinstance(stationary, (int, float)) \
            or not math.isfinite(stationary) or stationary <= 0:
        raise ValueError("standstill linear RMSE must be finite and positive")
    threshold = FIXED.WALK_FRACTION * float(stationary)
    walks = behavior["err_xy"] < threshold and behavior["fall_frac"] <= FIXED.REF_FALL_MAX
    return {"qualified": walks, "err_xy": float(behavior["err_xy"]),
            "strict_err_xy_max": threshold, "fall_frac": float(behavior["fall_frac"]),
            "fall_frac_max": FIXED.REF_FALL_MAX}


def _unknown(reason: str) -> dict:
    return {"behavior_outcome": "unknown", "reason_code": reason, "metric_checks": [],
            "absolute_qualification": {"candidate": None, "reference": None},
            "evidence_scope": "criteria_review_only_not_causal_or_recovery_confirmation"}


def classify_actual_fixed26(candidate_raw: object, reference_raw: object, bands: object) -> dict:
    """Classify validated fixed26 behavior without letting relative ratios bless stationary policies."""
    try:
        if not isinstance(bands, dict) or set(bands) != _BAND_KEYS:
            return _unknown("criteria_bands_missing_or_invalid")
        clean_bands = {}
        for key in _BAND_KEYS:
            value = bands[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                return _unknown("criteria_bands_missing_or_invalid")
            clean_bands[key] = float(value)
        candidate, reference = _unwrap(candidate_raw, "candidate"), _unwrap(reference_raw, "reference")
        candidate_absolute, reference_absolute = _absolute(candidate), _absolute(reference)
    except (KeyError, TypeError, ValueError):
        return _unknown("fixed26_missing_or_unverifiable")

    absolute = {"candidate": candidate_absolute, "reference": reference_absolute}
    scope = "criteria_review_only_not_causal_or_recovery_confirmation"
    if not reference_absolute["qualified"]:
        return {"behavior_outcome": "unknown", "reason_code": "reference_unqualified",
                "metric_checks": [], "absolute_qualification": absolute, "evidence_scope": scope}
    if not candidate_absolute["qualified"]:
        return {"behavior_outcome": "failed", "reason_code": "candidate_unqualified",
                "metric_checks": [], "absolute_qualification": absolute, "evidence_scope": scope}
    compared = RECOVERY.compare(FIXED._behavior(candidate), FIXED._behavior(reference))
    verdict = RECOVERY.verdict(compared, clean_bands)
    checks = [{"metric": name, "value": float(compared[name]),
               "reference_value": 1.0 if name != "fall_frac" else float(reference["metrics"]["fall_rate"]),
               "passed": bool(verdict["checks"][name])}
              for name in ("survival_ratio", "fall_frac", "err_xy_ratio", "err_yaw_ratio")]
    outcome = {"healthy": "normal", "unhealthy": "below_reference", "undetermined": "unknown"}[
        verdict["label"]]
    reason = {"normal": "absolute_and_relative_passed", "below_reference": "relative_band_failed",
              "unknown": "relative_comparison_undetermined"}[outcome]
    return {"behavior_outcome": outcome, "reason_code": reason, "metric_checks": checks,
            "absolute_qualification": absolute, "evidence_scope": scope}


def _load_ref(ref: object, label: str) -> tuple[Path, dict]:
    if not isinstance(ref, dict) or set(ref) != {"path", "sha256"}:
        raise ValueError(f"{label} reference fields differ")
    path = Path(ref["path"]).resolve(strict=True)
    if not isinstance(ref["sha256"], str) or _sha(path) != ref["sha256"]:
        raise ValueError(f"{label} source SHA differs")
    return path, _read_json(path)


def _fixed_ref_from_payload(row: dict, label: str) -> dict:
    paths, refs = row.get("paths"), row.get("refs")
    if not isinstance(paths, dict) or not isinstance(paths.get("fixed26"), str) or not isinstance(refs, list):
        raise TypeError(f"{label} source binding is incomplete")
    path = Path(paths["fixed26"]).resolve(strict=True)
    digest = _sha(path)
    if f"sha256:{digest}" not in refs:
        raise ValueError(f"{label} fixed26 SHA is not present in its verified refs")
    return {"path": str(path), "sha256": digest}


def _same_verified_report(report: dict, row: dict, label: str) -> None:
    if (report["checkpoint"].get("sha256") != row.get("checkpoint_sha256")
            or report.get("task") != row.get("task") or report.get("protocol") != row.get("protocol")
            or report.get("eval_seed") != row.get("seed") or report.get("metrics") != row.get("metrics")):
        raise ValueError(f"{label} report differs from the sealed source payload")


def qualify_source_observation(observation: GOVERNOR.VerifiedObservation) -> GOVERNOR.VerifiedObservation:
    """Re-open a sealed actual pair, bind its raw reports, and add absolute walk qualification."""
    packet = GOVERNOR.build_model_packet(observation)
    payload = packet["observation"]
    if payload.get("verifier_id") != "recorded_public_fixed26_v1" \
            or payload.get("verification_scope") != "recorded_actual_source_cpu_revalidation":
        raise ValueError("source observation is not the registered actual fixed26 verifier output")
    candidate_ref = _fixed_ref_from_payload(payload["candidate"], "candidate")
    reference_ref = _fixed_ref_from_payload(payload["reference"], "reference")
    _, candidate_raw = _load_ref(candidate_ref, "candidate")
    _, reference_raw = _load_ref(reference_ref, "reference")
    candidate_report, reference_report = _unwrap(candidate_raw, "candidate"), _unwrap(reference_raw, "reference")
    _same_verified_report(candidate_report, payload["candidate"], "candidate")
    _same_verified_report(reference_report, payload["reference"], "reference")
    contract, contract_sha = _contract()
    result = classify_actual_fixed26(candidate_raw, reference_raw, payload["criteria"].get("bands"))
    if result["reason_code"] in {"fixed26_missing_or_unverifiable", "criteria_bands_missing_or_invalid"}:
        raise ValueError("sealed actual fixed26 pair could not be qualified")
    qualified = deepcopy(payload)
    qualified.update({"verifier_id": VERIFIER_ID,
                      "verification_scope": "recorded_actual_fixed26_with_absolute_walk_qualification",
                      "behavior_outcome": result["behavior_outcome"],
                      "metric_checks": result["metric_checks"],
                      "absolute_qualification": result["absolute_qualification"],
                      "evidence_scope": result["evidence_scope"],
                      "qualification_reason_code": result["reason_code"]})
    qualified["criteria"] = {**qualified["criteria"], "qualification_rule_id": RULE_ID,
                             "qualification_protocol_sha256": contract_sha,
                             "absolute": contract["absolute_qualification"]}
    qualified["source_refs"] = {**qualified["source_refs"],
                                "candidate_fixed26": candidate_ref, "reference_fixed26": reference_ref,
                                "qualification_protocol": {"path": str(PROTOCOL), "sha256": contract_sha}}
    _reject_hidden(qualified, "qualified_observation")
    return GOVERNOR._verified_observation(qualified)


def classify_post_observation(
    source_verified_observation: GOVERNOR.VerifiedObservation,
    post_fixed_report_ref: dict,
    reference_report_ref: dict,
    new_checkpoint_ref: dict,
) -> GOVERNOR.VerifiedObservation:
    """Seal a new-checkpoint fixed26 result; no pre-correction confirmation receipt is copied."""
    source_packet = GOVERNOR.build_model_packet(source_verified_observation)["observation"]
    if source_packet.get("verifier_id") != VERIFIER_ID:
        raise ValueError("post observation requires an absolute-qualified source observation")
    expected_reference = source_packet["source_refs"].get("reference_fixed26")
    if reference_report_ref != expected_reference:
        raise ValueError("post observation reference was swapped")
    post_path, post_raw = _load_ref(post_fixed_report_ref, "post fixed26")
    _, reference_raw = _load_ref(reference_report_ref, "reference fixed26")
    checkpoint_path = Path(new_checkpoint_ref.get("path", "")).resolve(strict=True) \
        if isinstance(new_checkpoint_ref, dict) else Path()
    if (not isinstance(new_checkpoint_ref, dict) or set(new_checkpoint_ref) != {"path", "sha256"}
            or _sha(checkpoint_path) != new_checkpoint_ref["sha256"]):
        raise ValueError("new checkpoint reference differs")
    post_report, reference_report = _unwrap(post_raw, "post"), _unwrap(reference_raw, "reference")
    if (post_report["task"] != source_packet["candidate"]["task"]
            or post_report["protocol"] != source_packet["candidate"]["protocol"]
            or post_report["eval_seed"] != source_packet["candidate"]["seed"]
            or any(post_report[key] != reference_report[key]
                   for key in ("task", "protocol", "protocol_sha256_lf", "eval_seed", "num_envs",
                               "horizon_steps", "step_dt", "action_scale", "standstill"))):
        raise ValueError("post report lineage differs from the source observation")
    checkpoint = post_report["checkpoint"]
    if checkpoint.get("file") != checkpoint_path.name or checkpoint.get("sha256") != new_checkpoint_ref["sha256"]:
        raise ValueError("post report does not bind the new checkpoint")
    if checkpoint["sha256"] == source_packet["candidate"]["checkpoint_sha256"]:
        raise ValueError("post report reuses the source checkpoint")
    _same_verified_report(reference_report, source_packet["reference"], "reference")
    result = classify_actual_fixed26(post_raw, reference_raw, source_packet["criteria"]["bands"])
    if result["reason_code"] in {"fixed26_missing_or_unverifiable", "criteria_bands_missing_or_invalid"}:
        raise ValueError("post fixed26 pair could not be qualified")
    binding = GOVERNOR._canonical_sha({"post": post_fixed_report_ref, "reference": reference_report_ref,
                                      "checkpoint": new_checkpoint_ref,
                                      "qualification": source_packet["criteria"]["qualification_protocol_sha256"]})
    candidate = deepcopy(source_packet["candidate"])
    candidate.update({"source_id": post_report["name"], "task": post_report["task"],
                      "protocol": post_report["protocol"], "seed": post_report["eval_seed"],
                      "checkpoint_sha256": checkpoint["sha256"], "metrics": deepcopy(post_report["metrics"]),
                      "refs": [f"sha256:{post_fixed_report_ref['sha256']}",
                               f"sha256:{new_checkpoint_ref['sha256']}"],
                      "paths": {"fixed26": str(post_path), "checkpoint": str(checkpoint_path)},
                      "source_binding_sha256": binding})
    reference = deepcopy(source_packet["reference"])
    reference["source_binding_sha256"] = binding
    post_payload = deepcopy(source_packet)
    post_payload.update({"receipt_id": f"post:{checkpoint['sha256'][:16]}", "candidate": candidate,
                         "reference": reference, "behavior_outcome": result["behavior_outcome"],
                         "metric_checks": result["metric_checks"],
                         "absolute_qualification": result["absolute_qualification"],
                         "evidence_scope": result["evidence_scope"],
                         "qualification_reason_code": result["reason_code"],
                         "confirmation_checks": [], "cause_confirmation_receipt": None,
                         "correction_receipt": None,
                         "source_refs": {"post_fixed26": deepcopy(post_fixed_report_ref),
                                         "reference_fixed26": deepcopy(reference_report_ref),
                                         "new_checkpoint": deepcopy(new_checkpoint_ref),
                                         "qualification_protocol": deepcopy(
                                             source_packet["source_refs"]["qualification_protocol"])}})
    _reject_hidden(post_payload, "post_observation")
    return GOVERNOR._verified_observation(post_payload)
