"""새 촬영 자원 승인과 AN 실패 비용을 결합한 정책 녹화 계약."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import vision_pull_record_job_v5 as PREVIOUS
import vision_resource_issuer_v2 as ISSUER

COMMON = PREVIOUS.COMMON
RECORD = PREVIOUS.RECORD
RECORD_JOB = PREVIOUS.RECORD_JOB
PREPARED = PREVIOUS.PREPARED
ROOT = PREVIOUS.ROOT
MODE = PREVIOUS.MODE
PROTOCOL_SCHEMA = "vision_pull_record_protocol_v6"
PROTOCOL_ID = "native-policy-record-20261027ao"
JOB_SCHEMA = "vision_pull_record_job_v2"
RESULT_SCHEMA = PREVIOUS.RESULT_SCHEMA
FAILURE_SCHEMA = PREVIOUS.FAILURE_SCHEMA
CHECKPOINT_IDS = PREVIOUS.CHECKPOINT_IDS
PROTOCOL_KEYS = PREVIOUS.PROTOCOL_KEYS
JOB_KEYS = PREVIOUS.JOB_KEYS | {"resource_authorization", "resource_issuance"}
LIMITS = {
    **PREVIOUS.LIMITS,
    "max_batch_compute_usd": "4.912076",
}
FIXED_FLAGS = PREVIOUS.FIXED_FLAGS
REQUIRED_SOURCE_FILES = PREVIOUS.REQUIRED_SOURCE_FILES | {
    "bench/vision_resource_authorization_v1.py",
    "bench/vision_resource_issuer_v2.py",
    "tests/test_vision_resource_deadline_v1.py",
    "bench/vision_pull_record_job_v6.py",
    "bench/run_vision_pull_record_v6.py",
    "tests/test_vision_pull_record_job_v6.py",
    "tests/test_run_vision_pull_record_v6.py",
}
EXPECTED_PRIOR = Decimal("20.087924")
EXPECTED_FRESH = Decimal("4.912076")
EXPECTED_TOTAL = Decimal("25.000000")
EXPECTED_PREVIOUS_PROTOCOL = {
    "path": "bench/protocols/vision_pull_record_20261027_v6.json",
    "sha256_lf": "388e558c6f9f7b7cd81c106edf4ec11d4b4bd170808f324909b1167a51ee8b87",
}
EXPECTED_PREVIOUS_RESULT = {
    "path": "evals/results/vision_pull_record_dev_20261027an/summary.json",
    "sha256": "2e5ec5b2bb138b749ab64bb09deeec46a6954716563e2dcd0695dbf3b7a1d5c8",
}

_read = COMMON._read
_sha = COMMON._sha
_regular = COMMON._regular
write_new = COMMON.write_new
validation_deadline = COMMON.validation_deadline


def _reference(value: Any, anchor: str, *, lf: bool) -> dict[str, str]:
    key = "sha256_lf" if lf else "sha256"
    row = COMMON._exact(value, {"path", key}, anchor)
    COMMON._relative(row["path"], f"{anchor} path")
    COMMON._hash(row[key], f"{anchor} SHA")
    return row


def _previous_attempt(protocol: Mapping[str, Any], repo: Path) -> dict[str, Any]:
    predecessor = COMMON._exact(
        protocol["predecessor"],
        {"diagnostic", "appended_stops"},
        "pull record v6 predecessor",
    )
    previous_ref = EXPECTED_PREVIOUS_PROTOCOL
    previous_path = _regular(repo / previous_ref["path"], "previous protocol")
    previous = PREVIOUS.validate_protocol(previous_path, repo=repo)
    if predecessor["diagnostic"] != previous["predecessor"]["diagnostic"]:
        raise ValueError("pull record v6 diagnostic 계보가 다르다")
    appended = predecessor["appended_stops"]
    if (not isinstance(appended, list)
            or appended != [*previous["predecessor"]["appended_stops"], EXPECTED_PREVIOUS_RESULT]):
        raise ValueError("pull record v6의 7 STOP+AN 공개 결과 순서가 다르다")
    result_ref = _reference(appended[-1], "previous result", lf=False)
    result_path = _regular(repo / result_ref["path"], "previous result")
    if (_sha(previous_path, lf=True) != previous_ref["sha256_lf"]
            or _sha(result_path) != result_ref["sha256"]):
        raise ValueError("pull record v6 이전 protocol/result SHA가 다르다")
    result = _read(result_path)
    cost = result.get("cost")
    failure = result.get("failure")
    cleanup = result.get("cleanup")
    if (
        result.get("schema") != "vision_pull_record_infrastructure_failure_v1"
        or result.get("protocol_id") != PREVIOUS.PROTOCOL_ID
        or result.get("status") != "HOLD"
        or result.get("protocol") != EXPECTED_PREVIOUS_PROTOCOL
        or not isinstance(cost, Mapping)
        or PREPARED._amount(cost.get("failed_lifecycle_estimate_usd"), "AN failed cost")
        != Decimal("0.272854")
        or PREPARED._amount(cost.get("read_only_recovery_estimate_usd"), "AN recovery cost")
        != Decimal("0.064157")
        or PREPARED._amount(cost.get("cumulative_program_estimate_usd"), "AN cumulative")
        != EXPECTED_PRIOR
        or PREPARED._amount(cost.get("remaining_program_estimate_usd"), "AN remaining")
        != EXPECTED_FRESH
        or PREPARED._amount(cost.get("program_cap_usd"), "AN cap") != EXPECTED_TOTAL
        or not isinstance(failure, Mapping)
        or failure.get("stage") != "resource_registration"
        or failure.get("resource_proof_created") is not False
        or not isinstance(cleanup, Mapping)
        or cleanup.get("vm_status") != "STOPPED"
        or result.get("actual_recordings") != 0
        or result.get("new_training_actions") != 0
        or result.get("model_api_requests") != 0
        or result.get("capture_approved") is not False
        or result.get("used_for_decision") is not False
    ):
        raise ValueError("pull record AN 실패·비용·STOP 공개 증거가 다르다")
    previous["_result"] = result
    return previous


def _resources(protocol: Mapping[str, Any], previous: Mapping[str, Any]) -> None:
    resources = COMMON._exact(protocol["resources"], COMMON.RESOURCE_KEYS, "pull record v6 resources")
    old = PREVIOUS._resource_shape_v5(previous["resources"])
    prior = PREPARED._amount(resources["prior_compute_estimate_usd"], "prior")
    fresh = PREPARED._amount(resources["fresh_lifecycle_budget_usd"], "fresh", positive=True)
    total = PREPARED._amount(resources["total_program_budget_usd"], "total")
    if (
        resources["owned_instance_id"] != old["owned_instance_id"]
        or resources["owned_instance_name"] != old["owned_instance_name"]
        or resources["max_owned_instances"] != 1
        or PREPARED._amount(resources["compute_quote_usd_per_h"], "quote")
        != Decimal("1.74")
        or prior != EXPECTED_PRIOR
        or fresh != EXPECTED_FRESH
        or total != EXPECTED_TOTAL
        or prior + fresh != total
        or resources["prior_stop_receipts"]
        != [*old["prior_stop_receipts"], EXPECTED_PREVIOUS_RESULT]
    ):
        raise ValueError("pull record v6 누적 비용·소유 자원 계약이 다르다")


def validate_protocol(path: Path, *, repo: Path | None = None) -> dict[str, Any]:
    COMMON._check_deadline()
    repo = COMMON._directory(Path(repo or ROOT), "repo")
    path = _regular(Path(path).absolute(), "pull record v6 protocol")
    COMMON._verify_head_protocol(repo, path)
    protocol = COMMON._exact(_read(path), PROTOCOL_KEYS, "pull record v6 protocol")
    if (
        protocol["schema"] != PROTOCOL_SCHEMA
        or protocol["protocol_id"] != PROTOCOL_ID
        or protocol["mode"] != MODE
        or protocol["registered_before_measurement"] is not True
        or protocol["limits"] != LIMITS
    ):
        raise ValueError("pull record v6 기본 계약이 다르다")
    commit = protocol["source_commit"]
    if (
        not isinstance(commit, str)
        or COMMON.HEX40.fullmatch(commit) is None
        or COMMON._git(repo, "merge-base", "--is-ancestor", commit, "HEAD").returncode
    ):
        raise ValueError("pull record v6 source commit이 HEAD 조상이 아니다")
    sources = COMMON._source_map(protocol["source_sha256_lf"])
    if not REQUIRED_SOURCE_FILES <= sources.keys():
        raise ValueError("pull record v6 source closure가 부족하다")
    COMMON._verify_source_blobs(repo, commit, sources)
    previous = _previous_attempt(protocol, repo)
    for key in ("base_record_protocol", "source_protocol", "checkpoints"):
        if protocol[key] != previous[key]:
            raise ValueError(f"pull record v6가 이전 {key} 계보를 바꿨다")
    if any(sources.get(name) != digest for name, digest in previous["source_sha256_lf"].items()):
        raise ValueError("pull record v6가 이전 source closure를 보존하지 않는다")
    prereg = COMMON._exact(
        protocol["preregistration"], {"prediction", "outcomes", "fixed_flags"}, "preregistration"
    )
    outcomes = prereg["outcomes"]
    if (
        not isinstance(prereg["prediction"], str)
        or not prereg["prediction"].strip()
        or not isinstance(outcomes, dict)
        or set(outcomes) != {"recording_complete", "capture_hold", "infrastructure_failure"}
        or any(not isinstance(value, str) or not value.strip() for value in outcomes.values())
        or prereg["fixed_flags"] != FIXED_FLAGS
    ):
        raise ValueError("pull record v6 사전등록이 다르다")
    _resources(protocol, previous)
    protocol["checkpoints"] = RECORD_JOB._checkpoints(protocol["checkpoints"])
    protocol["_base"] = previous["_base"]
    protocol["_source"] = previous["_source"]
    protocol["_previous"] = previous
    return protocol


def _bound_reference(value: Any, keys: set[str], anchor: str) -> dict[str, str]:
    row = COMMON._exact(value, keys, anchor)
    for key in keys:
        if key == "path":
            if not isinstance(row[key], str) or not Path(row[key]).is_absolute():
                raise ValueError(f"{anchor} path는 절대 경로여야 한다")
        else:
            COMMON._hash(row[key], f"{anchor} {key}")
    return row


def validate_job(job_path: Path, *, now: datetime | None = None,
                 require_fresh: bool = True) -> dict[str, Any]:
    job_path = _regular(Path(job_path).absolute(), "pull record v6 job")
    job = COMMON._exact(_read(job_path), JOB_KEYS, "pull record v6 job")
    if job["schema"] != JOB_SCHEMA or job["mode"] != MODE:
        raise ValueError("pull record v6 job schema/mode가 다르다")
    source = COMMON._exact(job["source"], {"repo_root", "commit", "files_lf"}, "source")
    repo = COMMON._directory(Path(source["repo_root"]), "source repo")
    protocol_ref = _bound_reference(job["protocol"], {"path", "sha256_lf"}, "protocol")
    protocol_path = _regular(Path(protocol_ref["path"]), "protocol path")
    if _sha(protocol_path, lf=True) != protocol_ref["sha256_lf"]:
        raise ValueError("pull record v6 protocol SHA가 다르다")
    protocol = validate_protocol(protocol_path, repo=repo)
    if source != {"repo_root": str(repo), "commit": protocol["source_commit"],
                  "files_lf": protocol["source_sha256_lf"]} or job["profile"] != RECORD.fixed_profile():
        raise ValueError("pull record v6 job source/profile이 protocol과 다르다")
    lab = COMMON._directory(Path(job["isaaclab"]), "isaaclab")
    checkpoint = COMMON._exact(job["checkpoint"], {"path", "sha256", "logical_id"}, "checkpoint")
    identity = checkpoint["logical_id"]
    if identity not in CHECKPOINT_IDS:
        raise ValueError("pull record v6 checkpoint가 등록 두 건 밖이다")
    row, checkpoint_path = RECORD_JOB._checkpoint(protocol, identity, lab)
    if checkpoint != {"path": str(checkpoint_path), "sha256": row["sha256"], "logical_id": identity}:
        raise ValueError("pull record v6 checkpoint 결합이 다르다")
    if job["job_id"] != f"{PROTOCOL_ID}-{identity}":
        raise ValueError("pull record v6 job ID가 다르다")
    resource_ref = _bound_reference(job["resource_proof"], {"path", "sha256"}, "resource")
    authorization_ref = _bound_reference(
        job["resource_authorization"], {"path", "sha256_lf"}, "resource authorization"
    )
    issuance_ref = _bound_reference(
        job["resource_issuance"], {"path", "sha256"}, "resource issuance"
    )
    current = now or datetime.now(timezone.utc)  # noqa: UP017 - Python 3.10
    if (_sha(Path(authorization_ref["path"]), lf=True) != authorization_ref["sha256_lf"]
            or _sha(Path(issuance_ref["path"])) != issuance_ref["sha256"]):
        raise ValueError("pull record v6 authorization/issuance SHA가 다르다")
    ISSUER.verify_job_binding(
        Path(authorization_ref["path"]), Path(issuance_ref["path"]),
        Path(resource_ref["path"]), job_path, now=current, repo=repo,
    )
    resource = COMMON._resource(
        _regular(Path(resource_ref["path"]), "resource proof"), resource_ref["sha256"],
        protocol["_source"], protocol["source_protocol"]["sha256_lf"], current, require_fresh,
    )
    COMMON._python_executable(Path(job["python"]))
    COMMON._verify_runtime_sources(lab, protocol["_source"])
    output = Path(job["output_directory"])
    if not output.is_absolute() or output != job_path.parent / "output":
        raise ValueError("pull record v6 output 경계가 다르다")
    COMMON._check_ancestors(output.parent, "pull record v6 output")
    present = output.exists() or output.is_symlink()
    if require_fresh and present:
        raise FileExistsError(output)
    if not require_fresh and present:
        output = COMMON._directory(output, "pull record v6 output")
    return {
        "job_path": job_path, "job_sha256": _sha(job_path), "job": job,
        "protocol": protocol, "protocol_path": protocol_path,
        "protocol_sha256": protocol_ref["sha256_lf"], "output_directory": output,
        "repo": repo, "resource": resource, "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": row["sha256"],
    }


def create_job(protocol_path: Path, authorization_path: Path, issuance_path: Path,
               resource_path: Path, resource_sha: str, isaaclab: Path, python: Path,
               stage: Path, checkpoint_logical_id: str) -> Path:
    protocol_path = Path(protocol_path).absolute()
    protocol = validate_protocol(protocol_path)
    if checkpoint_logical_id not in CHECKPOINT_IDS:
        raise ValueError("pull record v6 checkpoint가 등록값이 아니다")
    lab = COMMON._directory(Path(isaaclab).absolute(), "isaaclab")
    row, checkpoint = RECORD_JOB._checkpoint(protocol, checkpoint_logical_id, lab)
    stage = Path(stage).absolute()
    COMMON._check_ancestors(stage, "pull record v6 stage")
    if stage.exists() or stage.is_symlink():
        raise FileExistsError(stage)
    stage.mkdir(parents=True)
    job_path = stage / "job.json"
    write_new(job_path, {
        "schema": JOB_SCHEMA, "mode": MODE,
        "job_id": f"{PROTOCOL_ID}-{checkpoint_logical_id}",
        "source": {"repo_root": str(ROOT), "commit": protocol["source_commit"],
                   "files_lf": protocol["source_sha256_lf"]},
        "protocol": {"path": str(protocol_path), "sha256_lf": _sha(protocol_path, lf=True)},
        "resource_authorization": {"path": str(Path(authorization_path).absolute()),
                                   "sha256_lf": _sha(authorization_path, lf=True)},
        "resource_issuance": {"path": str(Path(issuance_path).absolute()),
                              "sha256": _sha(issuance_path)},
        "resource_proof": {"path": str(Path(resource_path).absolute()), "sha256": resource_sha},
        "isaaclab": str(lab), "python": str(Path(python).absolute()),
        "output_directory": str(stage / "output"), "profile": RECORD.fixed_profile(),
        "checkpoint": {"path": str(checkpoint), "sha256": row["sha256"],
                       "logical_id": row["logical_id"]},
    })
    validate_job(job_path)
    return job_path
