"""Verify immutable completed recovery effects across registered continuations."""
from __future__ import annotations

import math
import re
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Never

from evals import registered_recovery_continuation_v3 as V3

EFFECT_SCHEMA = "registered_completed_effect_proof_v2"
LEGACY_EFFECT_SCHEMA = V3.EFFECT_SCHEMA
_IMPORT_FIELDS = {
    "artifact_id", "kind", "case_id", "method", "proof_sha256", "prior_plan_sha256",
    "diagnostic_manifest_sha256", "source_checkpoint_sha256", "completed_checkpoint_sha256",
    "seed", "task", "correction_id", "source_anchor", "registration_anchor",
}
_PROOF_FIELDS = {
    "schema", "status", "case_id", "method", "source_anchor", "registration_anchor",
    "request_id", "recovery_folder", "identity", "refs", "claims",
}
_IDENTITY_FIELDS = {
    "prior_plan_sha256", "diagnostic_manifest_sha256", "source_checkpoint_sha256",
    "final_checkpoint_sha256", "seed", "task", "correction_id", "source_anchor",
    "registration_anchor",
}
_CLAIMS = {
    "new_dispatch_started": False,
    "ledger_reconsumed": False,
    "full_environment_state_restored": False,
    "rng_state_restored_from_checkpoint": False,
}
_FORBIDDEN_SUBTREE = "ri_codex2_tokenfactory_live_160"


class ContinuationProofError(V3.ContinuationProofError):
    """Raised when a completed effect is not closed, immutable, and registered."""


def _fail(message: str) -> Never:
    raise ContinuationProofError(message)


def _reject_forbidden_path(value: Any, *, label: str) -> None:
    raw = value.get("path") if isinstance(value, Mapping) else value
    if not isinstance(raw, (str, Path)):
        return
    parts = [part.casefold() for part in re.split(r"[\\/]", str(raw)) if part]
    if _FORBIDDEN_SUBTREE.casefold() in parts:
        _fail(f"{label} enters forbidden subtree")


def _arm(manifest: Mapping[str, Any], case_id: str, method: str) -> Mapping[str, Any]:
    arms = manifest.get("arms")
    if not isinstance(arms, list):
        _fail("prepared arms missing")
    matches = [
        arm for arm in arms
        if isinstance(arm, Mapping)
        and arm.get("case_id") == case_id
        and arm.get("method") == method
    ]
    if len(matches) != 1:
        _fail("completed arm identity is not unique")
    return matches[0]


def _only_ref(value: Any, *, label: str) -> dict[str, str]:
    _reject_forbidden_path(value, label=label)
    path = V3._validate_ref(value, label=label)
    return {"path": str(path), "sha256": V3._sha(path)}


def _git_ancestor(repo: Path, ancestor: str, descendant: str) -> None:
    result = subprocess.run(
        ["git", "-C", str(repo), "merge-base", "--is-ancestor", ancestor, descendant],
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        _fail("historical source ancestry differs")


def _historical_closure(
    repo: Path,
    source_anchor: str,
    registration_anchor: str,
    closure: Mapping[str, Any],
) -> dict[str, Any]:
    for relative in closure:
        parts = [part.casefold() for part in str(relative).replace("\\", "/").split("/") if part]
        if len(parts) >= 2 and parts[:2] == ["bench", "private"]:
            _fail("historical source closure enters private subtree")
    resolved = Path(repo).resolve(strict=True)
    _git_ancestor(resolved, source_anchor, registration_anchor)
    _git_ancestor(resolved, registration_anchor, "HEAD")
    validation = V3._historical_closure(resolved, source_anchor, closure)
    return {
        **validation,
        "registration_anchor": registration_anchor,
        "current_worktree_authoritative": False,
    }


def build_completed_effect_proof(
    old_prepared: Path,
    case_id: str,
    method: str,
    output: Path,
) -> dict[str, str]:
    """Seal one closed effect and the registered evidence required to revalidate it."""
    _reject_forbidden_path(old_prepared, label="old prepared root")
    root = Path(old_prepared).resolve(strict=True)
    manifest_path = root / "manifest.json"
    manifest = V3._json(manifest_path)
    arm = _arm(manifest, case_id, method)
    recovery = Path(str(arm.get("recovery_folder"))).resolve(strict=True)
    prepared_path = recovery / "manifest.json"
    prepared = V3._json(prepared_path)
    context = prepared.get("recovery_context")
    if not isinstance(context, Mapping):
        _fail("recovery context missing")
    protocol = V3._json(V3._validate_ref(context.get("protocol"), label="recovery protocol"))
    diagnostic = protocol.get("diagnostic")
    operation = protocol.get("operation")
    artifacts = context.get("artifacts")
    if not isinstance(diagnostic, Mapping) or not isinstance(operation, Mapping):
        _fail("registered protocol identity missing")
    if not isinstance(artifacts, Mapping):
        _fail("registered artifact resolver missing")

    resources = sorted((recovery / "resources").glob("*.json"))
    receipts = sorted((recovery / "receipts").glob("*.json"))
    if len(resources) != 1 or len(receipts) != 1 or resources[0].stem != receipts[0].stem:
        _fail("completed effect request files are not unique")
    request_id = resources[0].stem
    resource = V3._json(resources[0])
    ledger_receipt = V3._json(receipts[0])
    if resource.get("status") != "complete" or ledger_receipt.get("request_id") != request_id:
        _fail("completed effect receipt is not closed")
    backend_ref = ledger_receipt.get("receipt_ref")
    backend_path = V3._validate_ref(backend_ref, label="actual backend receipt")
    backend_receipt = V3._json(backend_path)
    if backend_receipt.get("exit_code") != 0 or backend_receipt.get("request_id") != request_id:
        _fail("actual backend receipt is incomplete")

    result_path = root / "results" / case_id / f"{method}.json"
    result = V3._json(result_path)
    controller_resource_path = root / "resources" / case_id / f"{method}.json"
    controller_resource = V3._json(controller_resource_path)
    if result.get("recovery_started") is not True or controller_resource.get("status") != "complete":
        _fail("completed controller effect missing")

    diagnostic_ref = arm.get("diagnostic_manifest")
    diagnostic_path = V3._validate_ref(diagnostic_ref, label="diagnostic manifest")
    diagnostic_manifest = V3._json(diagnostic_path)
    source_ref = diagnostic_manifest.get("source")
    V3._validate_ref(source_ref, label="diagnostic source")
    training_id = operation.get("training_binding_artifact_id")
    training_artifact = artifacts.get(training_id) if isinstance(training_id, str) else None
    training_body = training_artifact.get("body") if isinstance(training_artifact, Mapping) else None
    if not isinstance(training_body, Mapping):
        _fail("registered training binding missing")

    plan_ref = _only_ref(manifest.get("plan"), label="prior plan")
    refs: dict[str, dict[str, str]] = {
        "prepared_manifest": V3._ref(manifest_path),
        "prepared_anchor": V3._ref(root / "anchor.json"),
        "prior_plan": plan_ref,
        "arm_result": V3._ref(result_path),
        "controller_resource": V3._ref(controller_resource_path),
        "diagnostic_manifest": _only_ref(diagnostic_ref, label="diagnostic manifest"),
        "diagnostic_source": _only_ref(source_ref, label="diagnostic source"),
        "arm_approval": _only_ref(arm.get("approval_ref"), label="arm approval"),
        "recovery_manifest": V3._ref(prepared_path),
        "recovery_anchor": V3._ref(recovery / "anchor.json"),
        "recovery_ledger": V3._ref(recovery / "recovery.jsonl"),
        "approval": V3._ref(recovery / f"approvals/{request_id}.json"),
        "admission": V3._ref(recovery / f"admissions/{request_id}.json"),
        "intent": V3._ref(recovery / f"intents/{request_id}.json"),
        "resource": V3._ref(resources[0]),
        "ledger_receipt": V3._ref(receipts[0]),
        "protocol": _only_ref(context.get("protocol"), label="recovery protocol"),
        "backend_receipt": _only_ref(backend_ref, label="actual backend receipt"),
        "fixed_report": _only_ref(backend_receipt.get("fixed_report"), label="fixed report"),
        "new_checkpoint": _only_ref(backend_receipt.get("new_checkpoint"), label="new checkpoint"),
    }
    for name, value in (backend_receipt.get("training_evidence") or {}).items():
        if isinstance(value, Mapping) and set(value) == {"path", "sha256"}:
            refs[f"training_evidence:{name}"] = _only_ref(
                value, label=f"training evidence {name}"
            )

    registration_anchor = manifest.get("registration_anchor")
    source_anchor = protocol.get("source_anchor")
    if not isinstance(registration_anchor, str) or not isinstance(source_anchor, str):
        _fail("registered source anchors missing")
    proof = {
        "schema": EFFECT_SCHEMA,
        "status": "completed_effect",
        "case_id": case_id,
        "method": method,
        "source_anchor": source_anchor,
        "registration_anchor": registration_anchor,
        "request_id": request_id,
        "recovery_folder": str(recovery),
        "identity": {
            "prior_plan_sha256": plan_ref["sha256"],
            "diagnostic_manifest_sha256": refs["diagnostic_manifest"]["sha256"],
            "source_checkpoint_sha256": diagnostic["checkpoint_sha256"],
            "final_checkpoint_sha256": backend_receipt["new_checkpoint"]["sha256"],
            "seed": operation["training_seed"],
            "task": training_body["task"],
            "correction_id": operation["correction_id"],
            "source_anchor": source_anchor,
            "registration_anchor": registration_anchor,
        },
        "refs": refs,
        "claims": dict(_CLAIMS),
    }
    return V3._write_new(Path(output), proof)


def _legacy_verify(
    proof_ref: Mapping[str, Any], expected_import: Mapping[str, Any], repo: Path
) -> dict[str, Any]:
    if (
        expected_import.get("case_id") != V3.CASE_ID
        or expected_import.get("method") != V3.EFFECT_METHOD
    ):
        _fail("legacy effect scope differs")
    return V3.verify_completed_effect(
        proof_ref,
        expected_case_id=str(expected_import["case_id"]),
        expected_method=str(expected_import["method"]),
        expected_prior_plan_sha256=str(expected_import["prior_plan_sha256"]),
        expected_diagnostic_manifest_sha256=str(
            expected_import["diagnostic_manifest_sha256"]
        ),
        expected_source_checkpoint_sha256=str(expected_import["source_checkpoint_sha256"]),
        repo=repo,
    )


def verify_completed_effect(
    proof_ref: Mapping[str, Any],
    *,
    expected_import: Mapping[str, Any],
    repo: Path,
) -> dict[str, Any]:
    """Revalidate a whitelisted completed effect without consuming its request again."""
    _reject_forbidden_path(proof_ref, label="completed effect proof")
    proof_path, proof = V3._load_ref(proof_ref, label="completed effect proof")
    if set(expected_import) != _IMPORT_FIELDS or expected_import.get("kind") != "completed_effect":
        _fail("completed effect stage import whitelist differs")
    if proof_ref.get("sha256") != expected_import["proof_sha256"]:
        _fail("completed effect proof SHA differs from whitelist")
    if proof.get("schema") == LEGACY_EFFECT_SCHEMA:
        legacy_identity = proof.get("identity")
        legacy_refs = proof.get("refs")
        if not isinstance(legacy_identity, Mapping):
            _fail("legacy completed effect identity missing")
        if not isinstance(legacy_refs, Mapping):
            _fail("legacy completed effect refs missing")
        for name, value in legacy_refs.items():
            if isinstance(value, Mapping) and "path" in value:
                _reject_forbidden_path(value, label=f"legacy refs.{name}")
        expected_legacy = {
            "prior_plan_sha256": expected_import["prior_plan_sha256"],
            "diagnostic_manifest_sha256": expected_import["diagnostic_manifest_sha256"],
            "source_checkpoint_sha256": expected_import["source_checkpoint_sha256"],
            "final_checkpoint_sha256": expected_import["completed_checkpoint_sha256"],
            "seed": expected_import["seed"],
            "task": expected_import["task"],
            "correction_id": expected_import["correction_id"],
            "source_anchor": expected_import["source_anchor"],
            "registration_anchor": expected_import["registration_anchor"],
        }
        if (
            proof.get("case_id") != expected_import["case_id"]
            or proof.get("method") != expected_import["method"]
            or legacy_identity != expected_legacy
        ):
            _fail("legacy completed effect differs from whitelist")
        return _legacy_verify(proof_ref, expected_import, Path(repo))
    if set(proof) != _PROOF_FIELDS or proof.get("schema") != EFFECT_SCHEMA:
        _fail("completed effect proof fields differ")
    if proof.get("status") != "completed_effect" or proof.get("claims") != _CLAIMS:
        _fail("completed effect proof is not closed")
    identity = proof.get("identity")
    refs = proof.get("refs")
    if not isinstance(identity, Mapping) or set(identity) != _IDENTITY_FIELDS:
        _fail("completed effect identity fields differ")
    if not isinstance(refs, Mapping):
        _fail("completed effect refs missing")
    expected_identity = {
        "prior_plan_sha256": expected_import["prior_plan_sha256"],
        "diagnostic_manifest_sha256": expected_import["diagnostic_manifest_sha256"],
        "source_checkpoint_sha256": expected_import["source_checkpoint_sha256"],
        "final_checkpoint_sha256": expected_import["completed_checkpoint_sha256"],
        "seed": expected_import["seed"],
        "task": expected_import["task"],
        "correction_id": expected_import["correction_id"],
        "source_anchor": expected_import["source_anchor"],
        "registration_anchor": expected_import["registration_anchor"],
    }
    if identity != expected_identity:
        _fail("completed effect identity differs from stage import")
    if (
        proof.get("case_id") != expected_import["case_id"]
        or proof.get("method") != expected_import["method"]
        or proof.get("source_anchor") != expected_import["source_anchor"]
        or proof.get("registration_anchor") != expected_import["registration_anchor"]
        or proof_ref.get("sha256") != expected_import["proof_sha256"]
    ):
        _fail("completed effect whitelist binding differs")

    loaded: dict[str, dict[str, Any]] = {}
    for name, value in refs.items():
        _reject_forbidden_path(value, label=f"refs.{name}")
        path = V3._validate_ref(value, label=f"refs.{name}")
        if path.suffix == ".json":
            loaded[name] = V3._json(path)
        elif path.suffix == ".jsonl":
            loaded[name] = {"rows": V3._jsonl(path)}
    manifest = loaded["prepared_manifest"]
    arm = _arm(manifest, str(expected_import["case_id"]), str(expected_import["method"]))
    if manifest.get("plan") != refs["prior_plan"] or refs["prior_plan"]["sha256"] != identity["prior_plan_sha256"]:
        _fail("prior registered plan differs")
    if arm.get("diagnostic_manifest") != refs["diagnostic_manifest"]:
        _fail("diagnostic session binding differs")
    if arm.get("recovery_folder") != proof["recovery_folder"]:
        _fail("historical recovery folder differs")

    protocol = loaded["protocol"]
    diagnostic = protocol.get("diagnostic")
    operation = protocol.get("operation")
    if not isinstance(diagnostic, Mapping) or not isinstance(operation, Mapping):
        _fail("registered protocol fields missing")
    if (
        diagnostic.get("case_id") != expected_import["case_id"]
        or diagnostic.get("session_manifest_sha256")
        != expected_import["diagnostic_manifest_sha256"]
        or diagnostic.get("checkpoint_sha256") != expected_import["source_checkpoint_sha256"]
        or operation.get("training_seed") != expected_import["seed"]
        or operation.get("correction_id") != expected_import["correction_id"]
        or protocol.get("source_anchor") != expected_import["source_anchor"]
    ):
        _fail("registered protocol identity differs")
    closure = protocol.get("source_closure")
    if not isinstance(closure, Mapping):
        _fail("historical source closure missing")
    source_validation = _historical_closure(
        Path(repo),
        str(expected_import["source_anchor"]),
        str(expected_import["registration_anchor"]),
        closure,
    )

    result = loaded["arm_result"]
    controller = loaded["controller_resource"]
    resource = loaded["resource"]
    ledger_receipt = loaded["ledger_receipt"]
    if (
        result.get("case_id") != expected_import["case_id"]
        or result.get("method") != expected_import["method"]
        or result.get("recovery_started") is not True
        or result.get("outcome") not in {"recovered", "not_recovered"}
        or result.get("new_checkpoint") != refs["new_checkpoint"]
        or result.get("fixed_report") != refs["fixed_report"]
        or controller.get("status") != "complete"
        or resource.get("status") != "complete"
        or resource.get("request_id") != proof["request_id"]
        or ledger_receipt.get("request_id") != proof["request_id"]
    ):
        _fail("completed effect result/receipt differs")
    request_id = str(proof["request_id"])
    rows = [
        row for row in loaded["recovery_ledger"]["rows"]
        if row.get("request_id") == request_id
    ]
    if [row.get("event") for row in rows] != ["proposed", "approved", "consumed", "receipt"]:
        _fail("completed recovery lifecycle differs")
    if rows[-1].get("verdict") != result["outcome"] or ledger_receipt.get("verdict") != result["outcome"]:
        _fail("completed recovery verdict differs")

    from evals import registered_recovery_backend_v1 as backend
    from rl_triage import investigation_recovery

    backend_verified = backend.verify_receipt(
        {**loaded["backend_receipt"], "receipt_ref": refs["backend_receipt"]},
        loaded["intent"],
    )
    post = investigation_recovery.verified_post_observation(
        Path(str(proof["recovery_folder"])), request_id
    )
    if post.get("new_checkpoint") != refs["new_checkpoint"] or post.get("fixed_report") != refs["fixed_report"]:
        _fail("verified post observation differs")

    costs = result.get("costs")
    if not isinstance(costs, Mapping):
        _fail("completed effect costs missing")
    training_s = costs.get("training_s")
    fixed_s = costs.get("fixed_s")
    if (
        isinstance(training_s, bool)
        or not isinstance(training_s, (int, float))
        or not math.isfinite(training_s)
        or isinstance(fixed_s, bool)
        or not isinstance(fixed_s, (int, float))
        or not math.isfinite(fixed_s)
    ):
        _fail("completed effect scientific cost differs")
    prior_reuse = loaded["backend_receipt"].get("training_reuse")
    current_launches = resource.get("simulator_sdk_launches")
    if type(current_launches) is not int or current_launches < 1:
        _fail("completed effect SDK accounting differs")
    historical_launches = current_launches + (1 if isinstance(prior_reuse, Mapping) else 0)
    if historical_launches != 2:
        _fail("completed effect whole-effect SDK accounting differs")
    historical_costs = {
        "simulator_sdk_launches": historical_launches,
        "owned_gpu_s": float(training_s) + float(fixed_s),
        "training_wall_s": float(training_s),
        "fixed_wall_s": float(fixed_s),
        "phase_wall_s": controller.get("phase_wall_s"),
        "phase_measurement_ref": refs["controller_resource"],
    }
    return {
        "execution_origin": "verified_prior_effect",
        "result": result,
        "prior_resource_ref": refs["resource"],
        "prior_admission_ref": refs["admission"],
        "recovery_folder": proof["recovery_folder"],
        "post_observation": post,
        "proof_ref": {"path": str(proof_path), "sha256": V3._sha(proof_path)},
        "historical_costs": historical_costs,
        "new_dispatch_costs": {"sdk_launches": 0, "owned_gpu_s": None},
        "historical_source_validation": source_validation,
        "backend_receipt_validation": backend_verified,
        "old_refs": dict(refs),
    }
