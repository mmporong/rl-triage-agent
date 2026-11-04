"""Verify the immutable V12 pre-SDK request-namespace failure evidence."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

CONTEXT_SHA256 = "83d46f5e1d2a640cf24d46b127db946b77ee8fe7e65960e2f082d1240fe480f2"
SOURCE_ANCHOR = "36f3afb39f793c1d470ed3dd268e1bd4887e6f69"
REGISTRATION_ANCHOR = "72f6d4d66da8d179826b42e947da62c756db58ab"
PLAN_SHA256 = "ee8b272ac0eddb2b4fecaaad61b21c44182655e44f46d7b24c2f084852cd0abd"
REQUEST_ID = "785427445a0a"
REFERENCE_SHA256 = {
    "error": "54d805143991a1d1025054285e8486e7cc41850e613f419adcf53104d4d3257c",
    "plan": PLAN_SHA256,
    "preflight": "ae70bf3c90d89119300dee93bfe77d306b82dc5d04a6b1e6d99badf7ee4f14ed",
    "resource": "b968bc94bf62b45fe2f8227fbe949c95fe77747788ed8b44836cb300cebd7947",
    "supervisor": "3cf5960e58f6f6963a102b7899738f509714cd1f4ce9b2163e5cbe281734b484",
}
_SHA = re.compile(r"[0-9a-f]{64}\Z")


class NamespaceFailureError(ValueError):
    """The registered V12 namespace failure evidence differs."""


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise NamespaceFailureError("namespace failure evidence must be an object")
    return value


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_failure_context(reference: dict) -> dict:
    if (not isinstance(reference, dict) or set(reference) != {"path", "sha256"}
            or reference.get("sha256") != CONTEXT_SHA256):
        raise NamespaceFailureError("namespace failure context reference differs")
    path = Path(str(reference["path"])).resolve(strict=True)
    if _sha(path) != CONTEXT_SHA256:
        raise NamespaceFailureError("namespace failure context SHA differs")
    body = _read(path)
    if (
        body.get("schema") != "registered_run_name_collision_evidence_v1"
        or body.get("status") != "pre_sdk_infrastructure_failure"
        or body.get("source_anchor") != SOURCE_ANCHOR
        or body.get("registration_anchor") != REGISTRATION_ANCHOR
        or body.get("case_id") != "case-07"
        or body.get("method") != "one_shot_llm"
        or body.get("request_id") != REQUEST_ID
        or body.get("registered_training_seed") != 6066
        or body.get("planned_next_restart_seeds_not_executed") != [6266, 6267]
        or body.get("training_executed") is not False
        or body.get("fixed_evaluation_executed") is not False
        or body.get("same_consumed_request_reexecute_allowed") is not False
        or body.get("prior_record_preserved") is not True
        or body.get("SDK") != 0
        or body.get("API") != 0
        or body.get("CPU_helper_children") != 0
    ):
        raise NamespaceFailureError("namespace failure identity/outcome differs")
    error = body.get("observed_error")
    if (not isinstance(error, dict)
            or error.get("error_type") != "FileExistsError"
            or error.get("reexecute_allowed") is not False):
        raise NamespaceFailureError("namespace failure error differs")
    refs = body.get("references")
    if not isinstance(refs, dict) or set(refs) != set(REFERENCE_SHA256):
        raise NamespaceFailureError("namespace failure raw references differ")
    for role, expected_sha in REFERENCE_SHA256.items():
        row = refs.get(role)
        if (not isinstance(row, dict) or set(row) != {"path", "sha256"}
                or row.get("sha256") != expected_sha
                or _SHA.fullmatch(str(row.get("sha256"))) is None):
            raise NamespaceFailureError(f"namespace failure {role} reference differs")
        raw = Path(str(row["path"])).resolve(strict=True)
        if _sha(raw) != expected_sha:
            raise NamespaceFailureError(f"namespace failure {role} raw SHA differs")
    return {"context_ref": {"path": str(path), "sha256": CONTEXT_SHA256}, "context": body}
