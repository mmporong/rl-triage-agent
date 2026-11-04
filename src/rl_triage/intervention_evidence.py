"""Production adapter for recorded public fixed26 and recovery evidence.

This adapter revalidates already-recorded public artifacts on CPU.  It does not
run a policy, score a benchmark, infer a fault class, or authorize execution.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from copy import deepcopy
from pathlib import Path

from evals import export_lineage_source_collection_v2 as SOURCE_EXPORT

from . import grounded_recovery as FIXED
from . import intervention_governor as GOVERNOR
from . import recovery as RECOVERY
from .correction_registry import REGISTRY

ROOT = Path(__file__).resolve().parents[2]
SOURCE_COLLECTION = ROOT / "evals/results/five_type_sources_s2550_s2559_20261101_v1"
RECOVERY_COLLECTION = ROOT / "evals/results/five_type_recovery_s4552_s5557_20261101_v1"
CRITERIA_SUMMARY = ROOT / "evals/results/p0c_fixed_eval_20261009/summary.json"
FIXED_PROTOCOL = ROOT / "bench/protocols/fixed_eval_v1.json"

SOURCE_MANIFEST_SHA256 = "141a72f9b7c15e7a684a70811ddcf03f1c813981be8e8b356a0bceff69b4ce60"
RECOVERY_MANIFEST_SHA256 = "4095f25b14e27c97731877df982e8ecb58d44cf58ab6b3d4190f38c3810bd667"
CRITERIA_SHA256 = "7589a132d4e1a55e159527389a8b635a1766472ecde558da0a1a704c286f3f23"
FIXED_PROTOCOL_SHA256 = "8d24652eeed47f462d78244fceec785164b7695e5b757ae8a4e9bd7947aa8a46"
VERIFIER_ID = "recorded_public_fixed26_v1"
RECOVERY_VERIFIER_ID = "recorded_public_recovery_miss_v1"
_SHA_RE = re.compile(r"[0-9a-f]{64}")
_METRICS = ("fall_rate", "mean_survival_s", "lin_vel_rmse_mps", "yaw_rate_rmse_radps")
DEFAULT_REGISTRATION = {
    "collection_manifest_sha256": SOURCE_MANIFEST_SHA256,
    "criteria_sha256": CRITERIA_SHA256,
    "fixed_protocol_sha256": FIXED_PROTOCOL_SHA256,
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda token: (
            _ for _ in ()).throw(ValueError(f"non-finite JSON number: {token}")))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"recorded source is unavailable or invalid: {path.name}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"recorded source must be a JSON object: {path.name}")
    return value


def _registered_path(value: Path, expected: Path, expected_sha: str, label: str) -> Path:
    path = Path(value).resolve(strict=True)
    if path != expected.resolve(strict=True) or _sha(path) != expected_sha:
        raise ValueError(f"{label} is not the registered recorded source")
    return path


def _registration(value: dict | None) -> dict[str, str]:
    result = DEFAULT_REGISTRATION if value is None else value
    if (not isinstance(result, dict) or set(result) != set(DEFAULT_REGISTRATION)
            or any(not isinstance(item, str) or _SHA_RE.fullmatch(item) is None
                   for item in result.values())):
        raise ValueError("fixed26 source registration fields differ")
    if (result["criteria_sha256"] != CRITERIA_SHA256
            or result["fixed_protocol_sha256"] != FIXED_PROTOCOL_SHA256):
        raise ValueError("fixed26 criteria or protocol registration changed")
    return dict(result)


def _finite_metrics(value: object, label: str) -> dict[str, float]:
    if not isinstance(value, dict) or set(value) != set(_METRICS):
        raise ValueError(f"{label} must contain the four fixed26 behavior metrics")
    result = {}
    for name in _METRICS:
        item = value[name]
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item):
            raise ValueError(f"{label}.{name} must be finite")
        result[name] = float(item)
    return result


def _criteria(criteria_summary: Path, fixed_protocol: Path) -> dict:
    summary_path = _registered_path(criteria_summary, CRITERIA_SUMMARY, CRITERIA_SHA256, "criteria summary")
    protocol_path = _registered_path(fixed_protocol, FIXED_PROTOCOL, FIXED_PROTOCOL_SHA256, "fixed protocol")
    summary = _read(summary_path)
    if (summary.get("kind") != "fixed_eval_summary" or summary.get("protocol") != FIXED.PROTOCOL
            or summary.get("protocol_sha256_lf") != FIXED_PROTOCOL_SHA256):
        raise ValueError("registered criteria is not bound to fixed_eval_v1")
    bands = summary.get("bands")
    expected = {"survival_ratio_min", "fall_frac_max", "err_xy_ratio_max", "err_yaw_ratio_max"}
    if not isinstance(bands, dict) or set(bands) != expected:
        raise ValueError("registered criteria bands are incomplete")
    clean_bands = {}
    for name in sorted(expected):
        value = bands[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"criteria band {name} is not finite")
        clean_bands[name] = float(value)
    return {"protocol": FIXED.PROTOCOL, "protocol_sha256_lf": FIXED_PROTOCOL_SHA256,
            "bands": clean_bands,
            "refs": [f"sha256:{CRITERIA_SHA256}", f"sha256:{FIXED_PROTOCOL_SHA256}"],
            "paths": {"summary": str(summary_path), "protocol": str(protocol_path)}}


def _fixed_report(root: Path, inventory: list[dict], source_id: str, criteria: dict) -> dict:
    matches = [row for row in inventory if row.get("source_id") == source_id]
    if len(matches) != 1:
        raise ValueError(f"registered source inventory does not contain exactly one {source_id}")
    item = matches[0]
    fixed_path = (root / item["fixed26"]).resolve(strict=True)
    provenance_path = (root / item["provenance"]).resolve(strict=True)
    if root not in fixed_path.parents or root not in provenance_path.parents:
        raise ValueError("source inventory path escaped the registered collection")
    fixed = _read(fixed_path)
    provenance = _read(provenance_path)
    if (set(fixed) != {"schema", "source_id", "conditions", "report", "execution"}
            or fixed["schema"] != "lineage_source_collection_public_v2_fixed26"
            or fixed["source_id"] != source_id or fixed["conditions"] != 26):
        raise ValueError("fixed26 public wrapper is invalid")
    execution = fixed["execution"]
    if (not isinstance(execution, dict) or execution.get("exit_code") != 0
            or execution.get("natural_exit") is not True or execution.get("cleanup_verified") is not True
            or execution.get("timed_out") is not False or execution.get("error_type") is not None):
        raise ValueError("fixed26 execution receipt is not a verified natural completion")
    report = fixed["report"]
    if not isinstance(report, dict):
        raise TypeError("fixed26 report is missing")
    checkpoint = report.get("checkpoint")
    if not isinstance(checkpoint, dict):
        raise TypeError("fixed26 checkpoint binding is missing")
    expected = {"name": source_id, "checkpoint_file": checkpoint.get("file"),
                "checkpoint_sha256": checkpoint.get("sha256"), "action_scale": report.get("action_scale"),
                "protocol_sha256_lf": criteria["protocol_sha256_lf"]}
    checked = FIXED.validate_fixed_report(report, expected)
    if (provenance.get("schema") != "lineage_source_collection_public_v2_source_provenance"
            or provenance.get("source_id") != source_id
            or provenance.get("sha256_only", {}).get("checkpoint") != checked["checkpoint"]["sha256"]):
        raise ValueError("fixed26 report does not match its public source provenance")
    metrics = _finite_metrics(checked["metrics"], f"{source_id}.metrics")
    return {"source_id": source_id, "task": checked["task"], "protocol": checked["protocol"],
            "seed": checked["eval_seed"], "checkpoint_sha256": checked["checkpoint"]["sha256"],
            "metrics": metrics,
            "refs": [f"sha256:{_sha(fixed_path)}", f"sha256:{_sha(provenance_path)}"],
            "paths": {"fixed26": str(fixed_path), "provenance": str(provenance_path)}}


def _behavior(candidate: dict, reference: dict, bands: dict) -> tuple[str, list[dict]]:
    run = {"survival_s": candidate["metrics"]["mean_survival_s"], "horizon_s": 20.0,
           "fall_frac": candidate["metrics"]["fall_rate"],
           "err_xy": candidate["metrics"]["lin_vel_rmse_mps"],
           "err_yaw": candidate["metrics"]["yaw_rate_rmse_radps"]}
    ref = {"survival_s": reference["metrics"]["mean_survival_s"], "horizon_s": 20.0,
           "fall_frac": reference["metrics"]["fall_rate"],
           "err_xy": reference["metrics"]["lin_vel_rmse_mps"],
           "err_yaw": reference["metrics"]["yaw_rate_rmse_radps"]}
    compared = RECOVERY.compare(run, ref)
    verdict = RECOVERY.verdict(compared, bands)
    metric_checks = [{"metric": name, "value": float(compared[name]),
                      "passed": bool(verdict["checks"][name])}
                     for name in ("survival_ratio", "fall_frac", "err_xy_ratio", "err_yaw_ratio")]
    outcome = {"healthy": "normal", "unhealthy": "failed", "undetermined": "unknown"}[verdict["label"]]
    return outcome, metric_checks


def verify_public_fixed26_pair(
    collection_dir: Path,
    *,
    case_id: str,
    criteria_summary: Path = CRITERIA_SUMMARY,
    fixed_protocol: Path = FIXED_PROTOCOL,
    registration: dict | None = None,
) -> GOVERNOR.VerifiedObservation:
    """Revalidate an actual public case/reference pair and seal its numeric bundle."""
    root = Path(collection_dir).resolve(strict=True)
    if registration is None and root != SOURCE_COLLECTION.resolve(strict=True):
        raise ValueError("source collection is not the registered actual public export")
    registered = _registration(registration)
    if _sha(root / "manifest.json") != registered["collection_manifest_sha256"]:
        raise ValueError("source collection is not the registered actual public export")
    SOURCE_EXPORT.verify_export(root)
    criteria = _criteria(criteria_summary, fixed_protocol)
    summary = _read(root / "summary.json")
    rows = summary.get("case_plan")
    if not isinstance(rows, list):
        raise TypeError("public source case plan is missing")
    selected = [row for row in rows if isinstance(row, dict) and row.get("case_id") == case_id]
    if len(selected) != 1:
        raise ValueError("case_id does not identify exactly one registered pair")
    pair = selected[0]
    source_id, reference_id = pair.get("source_id"), pair.get("reference_id")
    if not isinstance(source_id, str) or not isinstance(reference_id, str):
        raise TypeError("registered pair does not name source and reference IDs")
    inventory = summary.get("sources")
    if not isinstance(inventory, list):
        raise TypeError("public source inventory is missing")
    candidate = _fixed_report(root, inventory, source_id, criteria)
    reference = _fixed_report(root, inventory, reference_id, criteria)
    lineage_matches = bool(candidate["task"] == reference["task"] == FIXED.TASK
                           and candidate["protocol"] == reference["protocol"] == criteria["protocol"])
    if not lineage_matches:
        raise ValueError("case/reference task or protocol does not match registered criteria")
    outcome, metric_checks = _behavior(candidate, reference, criteria["bands"])
    binding = GOVERNOR._canonical_sha({"manifest": registered["collection_manifest_sha256"],
                                      "criteria": registered["criteria_sha256"],
                                      "protocol": registered["fixed_protocol_sha256"]})
    for row in (candidate, reference):
        row["source_binding_sha256"] = binding
    payload = {"schema": GOVERNOR.NORMALIZED_SCHEMA, "receipt_id": f"actual:{case_id}",
               "verifier_id": VERIFIER_ID, "verification_scope": "recorded_actual_source_cpu_revalidation",
               "candidate": candidate, "reference": reference,
               "criteria": {"protocol": criteria["protocol"], "bands": criteria["bands"],
                            "refs": criteria["refs"]},
               "reference_available": True, "lineage_matches": True,
               "behavior_outcome": outcome, "metric_checks": metric_checks,
               "confirmation_checks": [
                   {"check_id": "candidate_fixed26_integrity", "outcome": "normal", "refs": candidate["refs"]},
                   {"check_id": "reference_fixed26_integrity", "outcome": "normal", "refs": reference["refs"]},
               ],
               "cause_confirmation_receipt": None, "correction_receipt": None,
               "source_refs": {"manifest": {"path": str(root / "manifest.json"),
                                             "sha256": registered["collection_manifest_sha256"]},
                               "candidate": candidate["paths"], "reference": reference["paths"]}}
    return GOVERNOR._verified_observation(payload)


def _verify_recovery_export(root: Path) -> tuple[dict, dict]:
    if root != RECOVERY_COLLECTION.resolve(strict=True) or _sha(root / "manifest.json") != RECOVERY_MANIFEST_SHA256:
        raise ValueError("recovery collection is not the registered actual public export")
    manifest = _read(root / "manifest.json")
    expected_files = {"cases.json", "provenance.json", "summary.json"}
    if (manifest.get("schema") != "five_policy_recovery_public_manifest_v1"
            or manifest.get("status") != "complete" or set(manifest.get("files", {})) != expected_files):
        raise ValueError("recovery public manifest is invalid")
    actual = {path.name for path in root.iterdir() if path.is_file()}
    if actual != expected_files | {"manifest.json"}:
        raise ValueError("recovery public export has unregistered files")
    for name, ref in manifest["files"].items():
        path = root / name
        # Historical exports store an absolute public path; SHA and size remain authoritative.
        if (not isinstance(ref, dict) or ref.get("sha256") != _sha(path)
                or ref.get("size_bytes") != path.stat().st_size):
            raise ValueError(f"recovery public file binding changed: {name}")
    summary, cases = _read(root / "summary.json"), _read(root / "cases.json")
    if (summary.get("schema") != "five_policy_recovery_public_summary_v1"
            or summary.get("scope") != "actual_backend" or summary.get("status") != "complete"
            or cases.get("schema") != "five_policy_recovery_public_cases_v1"
            or cases.get("status") != "complete"):
        raise ValueError("recovery public summary or cases schema is invalid")
    return summary, cases


def verify_public_recovery_miss(
    source_collection: Path,
    recovery_collection: Path,
    *,
    case_id: str,
    policy: str,
    criteria_summary: Path = CRITERIA_SUMMARY,
    fixed_protocol: Path = FIXED_PROTOCOL,
) -> GOVERNOR.VerifiedObservation:
    """Consume a recorded correction miss without fabricating absent post metrics."""
    source = verify_public_fixed26_pair(source_collection, case_id=case_id,
                                        criteria_summary=criteria_summary, fixed_protocol=fixed_protocol)
    root = Path(recovery_collection).resolve(strict=True)
    _summary, cases = _verify_recovery_export(root)
    rows = cases.get("rows")
    selected = [row for row in rows if isinstance(row, dict)
                and row.get("case_id") == case_id and row.get("policy") == policy] if isinstance(rows, list) else []
    if len(selected) != 1:
        raise ValueError("recovery case/policy does not identify exactly one recorded row")
    row = selected[0]
    candidate = source._payload["candidate"]
    training = row.get("first_training")
    correction_id = row.get("first_correction")
    registered_ids = {entry["correction_id"] for entry in REGISTRY.values()}
    if (row.get("first_outcome") != "not_recovered" or not isinstance(training, dict)
            or training.get("fixed_conditions") != 26
            or row.get("original_source_checkpoint_sha256") != candidate["checkpoint_sha256"]
            or correction_id not in registered_ids
            or _SHA_RE.fullmatch(str(row.get("first_receipt_sha256"))) is None
            or _SHA_RE.fullmatch(str(row.get("first_new_checkpoint_sha256"))) is None):
        raise ValueError("recorded correction miss is not bound to the source case and fixed26 receipt")
    payload = deepcopy(source._payload)
    payload.update({"receipt_id": f"actual-recovery-miss:{case_id}:{policy}",
                    "verifier_id": RECOVERY_VERIFIER_ID,
                    "candidate": {"task": candidate["task"], "protocol": candidate["protocol"],
                                  "seed": row["first_registered_seed"],
                                  "checkpoint_sha256": row["first_new_checkpoint_sha256"],
                                  "source_binding_sha256": candidate["source_binding_sha256"],
                                  "metrics": {},
                                  "refs": [f"sha256:{row['first_receipt_sha256']}"]},
                    "behavior_outcome": "unknown", "metric_checks": [],
                    "confirmation_checks": [{"check_id": "post_correction_fixed26_metrics",
                                             "outcome": "unknown",
                                             "refs": [f"sha256:{row['first_receipt_sha256']}"]}],
                    "correction_receipt": {"correction_id": correction_id, "status": "applied",
                                           "refs": [f"sha256:{row['first_receipt_sha256']}"],
                                           "missing_reason": "public_recovery_export_omits_post_correction_metrics"},
                    "missing_reason": "public_recovery_export_omits_post_correction_metrics",
                    "source_refs": {**payload["source_refs"],
                                    "recovery_manifest": {"path": str(root / "manifest.json"),
                                                          "sha256": RECOVERY_MANIFEST_SHA256},
                                    "recovery_cases": {"path": str(root / "cases.json"),
                                                       "sha256": _sha(root / "cases.json")}}})
    return GOVERNOR._verified_observation(payload)
