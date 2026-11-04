"""Recompute registered corrected-training checks from source-bound CPU evidence."""
from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _folder in (ROOT, ROOT / "bench", ROOT / "evals", ROOT / "src"):
    if str(_folder) not in sys.path:
        sys.path.insert(0, str(_folder))

from bench import registered_recovery_training_v1 as ACTUATOR_TRAIN
from evals import ppo_ratio_repair_execution as RATIO
from evals import r1_repair_execution as R1

SCHEMA = "registered_recovery_confirmation_v1"
RUNNERS = {"r1_epochs_v2", "r1_epochs_preserving_environment_v1", "ppo_ratio_v1", "actuator_resume_v1"}
CATALOG_CHECKS = frozenset({"optimizer_update_count", "ppo_surrogate_gradient", "actuator_mapping"})
_OPTIMIZER_REQUIRED = {"manifest", "readback", "records", "checkpoint"}
_OPTIMIZER_OPTIONAL = {"checkpoint_execution", "telemetry", "telemetry_execution"}
_ACTUATOR_REQUIRED = {"manifest", "readback", "training_receipt"}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda token: (
            _ for _ in ()).throw(ValueError(f"non-finite JSON number: {token}")))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"evidence JSON is unavailable or invalid: {path}") from exc
    if not isinstance(value, dict):
        raise TypeError("evidence JSON must be an object")
    return value


def _ref(value: object, label: str, *, json_body: bool = True) -> tuple[Path, dict | None]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} reference fields differ")
    path = Path(value["path"]).resolve(strict=True)
    if path.is_symlink() or not isinstance(value["sha256"], str) or _sha(path) != value["sha256"]:
        raise ValueError(f"{label} reference changed")
    return path, _json(path) if json_body else None


def _checkpoint(value: object) -> tuple[Path, dict]:
    path, _ = _ref(value, "new checkpoint", json_body=False)
    assert isinstance(value, Mapping)
    return path, {"path": str(path), "sha256": value["sha256"]}


def _check(check_id: str, outcome: str, refs: list[str]) -> dict:
    if outcome not in {"normal", "abnormal", "unknown"}:
        raise ValueError(f"{check_id} returned an unsupported outcome")
    return {"check_id": check_id, "outcome": outcome, "refs": refs}


def _optimizer(
    runner_id: str,
    evidence: Mapping[str, Any],
    checkpoint: dict,
) -> tuple[list[dict], dict[str, dict], bool]:
    keys = set(evidence)
    if not _OPTIMIZER_REQUIRED <= keys or not keys <= _OPTIMIZER_REQUIRED | _OPTIMIZER_OPTIONAL:
        raise ValueError("optimizer evidence fields differ")
    loaded: dict[str, dict] = {}
    refs: dict[str, dict] = {}
    for name in sorted(keys):
        path, body = _ref(evidence[name], name, json_body=name != "records")
        refs[name] = {"path": str(path), "sha256": evidence[name]["sha256"]}
        if body is not None:
            loaded[name] = body
    records = Path(refs["records"]["path"])
    manifest, readback, cpu_checkpoint = loaded["manifest"], loaded["readback"], loaded["checkpoint"]
    if manifest.get("arm") != "CORRECTED" or manifest.get("mode") != "train":
        raise ValueError("optimizer confirmation requires the CORRECTED training arm")
    if (Path(manifest.get("records_path", "")).resolve() != records
            or Path(manifest.get("readback_path", "")).resolve() != Path(refs["readback"]["path"])):
        raise ValueError("optimizer manifest evidence paths differ")
    if cpu_checkpoint.get("final", {}).get("sha256") != checkpoint["sha256"]:
        raise ValueError("CPU checkpoint evidence differs from the new checkpoint")
    common_refs = [f"sha256:{refs[name]['sha256']}" for name in ("manifest", "readback", "records", "checkpoint")]
    if runner_id in {"r1_epochs_v2", "r1_epochs_preserving_environment_v1"}:
        R1.WRAPPER.validate_manifest(manifest)
        R1._validate_readback(readback, "CORRECTED")
        inspection = R1._inspection(records, manifest)
        checks = [_check("optimizer_update_count", inspection["status"], common_refs)]
    else:
        RATIO.WRAPPER.validate_manifest(manifest)
        RATIO.WRAPPER.validate_readback(readback, "CORRECTED")
        inspection = RATIO.inspection(records, manifest)
        checks = [
            _check("optimizer_update_count", inspection["count_status"], common_refs),
            _check("ppo_surrogate_gradient", inspection["gradient_status"], common_refs),
        ]
    return checks, refs, False


def _actuator(
    evidence: Mapping[str, Any],
    checkpoint: dict,
) -> tuple[list[dict], dict[str, dict], bool]:
    keys = set(evidence)
    if keys != _ACTUATOR_REQUIRED:
        raise ValueError("actuator evidence fields differ")
    refs: dict[str, dict] = {}
    loaded: dict[str, dict] = {}
    for name in sorted(keys):
        path, body = _ref(evidence[name], name)
        refs[name] = {"path": str(path), "sha256": evidence[name]["sha256"]}
        assert body is not None
        loaded[name] = body
    manifest, readback, receipt = loaded["manifest"], loaded["readback"], loaded["training_receipt"]
    if (manifest.get("schema") != "training_correction_manifest_v1"
            or manifest.get("arm") != "corrected" or manifest.get("fault") != "F2"
            or manifest.get("correction_id") != ACTUATOR_TRAIN.CORRECTION):
        raise ValueError("actuator confirmation requires the corrected F2 training arm")
    ACTUATOR_TRAIN.LEGACY.validate_readback(readback, "F2", ACTUATOR_TRAIN.CORRECTION)
    if (receipt.get("schema") != "training_correction_receipt_v1"
            or receipt.get("completed") is not True or receipt.get("arm") != "corrected"
            or receipt.get("manifest_sha256") != refs["manifest"]["sha256"]
            or receipt.get("correction_sha256") != refs["readback"]["sha256"]
            or receipt.get("checkpoint") != checkpoint):
        raise ValueError("actuator training receipt differs from the corrected checkpoint")
    base_refs = [f"sha256:{refs[name]['sha256']}" for name in ("manifest", "readback", "training_receipt")]
    return [_check("actuator_mapping", "unknown", base_refs)], refs, False


def verify_corrected_runtime(
    runner_id: str,
    evidence: Mapping[str, Any],
    new_checkpoint_ref: Mapping[str, Any],
) -> dict:
    """Recompute registered component checks; it does not test final policy behavior."""
    if runner_id not in RUNNERS:
        raise ValueError("runner_id is not registered")
    if not isinstance(evidence, Mapping):
        raise TypeError("evidence must be a mapping")
    _checkpoint_path, checkpoint = _checkpoint(new_checkpoint_ref)
    if runner_id == "actuator_resume_v1":
        checks, refs, mapping_observed = _actuator(evidence, checkpoint)
    else:
        checks, refs, mapping_observed = _optimizer(runner_id, evidence, checkpoint)
    outcomes = [row["outcome"] for row in checks]
    outcome = "abnormal" if "abnormal" in outcomes else "normal" if all(
        item == "normal" for item in outcomes) else "unknown"
    source_refs = deepcopy(refs)
    source_refs["new_checkpoint"] = checkpoint
    observed = {row["check_id"] for row in checks if row["outcome"] != "unknown"}
    return {
        "schema": SCHEMA,
        "runner_id": runner_id,
        "outcome": outcome,
        "checks": checks,
        "scope": ("corrected_training_first_three_updates" if runner_id != "actuator_resume_v1"
                  else "correction_readback_without_registered_post_mapping"),
        "checkpoint_sha256": checkpoint["sha256"],
        "source_refs": source_refs,
        "catalog_coverage_complete": observed == CATALOG_CHECKS,
        "unobserved_components": sorted(CATALOG_CHECKS - observed),
        "final_checkpoint_weights_tested": False,
        "unobserved_updates_after_index": 2 if runner_id != "actuator_resume_v1" else None,
        "actuator_post_mapping_observed": mapping_observed,
        "behavior_recovery_tested": False,
        "cause_confirmed": False,
        "approval_granted": False,
        "correction_authority_created": False,
    }
