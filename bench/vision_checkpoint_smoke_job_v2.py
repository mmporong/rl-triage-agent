"""checkpoint별 A-A-B-A smoke 뒤 정책 1 step 실행 계약."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import vision_record_job_v1 as RECORD_JOB

COMMON = RECORD_JOB.COMMON
SOURCE = RECORD_JOB.SOURCE
RECORD = RECORD_JOB.RECORD
ROOT = RECORD_JOB.ROOT
MODE = "prepared_checkpoint_smoke_after_capture_only"
PROTOCOL_SCHEMA = "vision_checkpoint_smoke_protocol_v2"
JOB_SCHEMA = "vision_checkpoint_smoke_job_v2"
RESULT_SCHEMA = "vision_checkpoint_smoke_result_v2"
FAILURE_SCHEMA = "vision_checkpoint_smoke_failure_v2"
LIMITS = {
    "max_wall_s": 600,
    "logical_frames": 4,
    "policy_environment_steps": 1,
    "callbacks": 6,
    "training_actions": 0,
    "model_api_requests": 0,
    "full_recordings": 0,
}
PROTOCOL_KEYS = {
    "schema",
    "mode",
    "protocol_id",
    "registered_before_measurement",
    "source_commit",
    "source_sha256_lf",
    "base_record_protocol",
    "limits",
    "preregistration",
}
JOB_KEYS = COMMON.JOB_KEYS | {"checkpoint", "profile"}

_read = COMMON._read
_sha = COMMON._sha
_regular = COMMON._regular
write_new = COMMON.write_new
validation_deadline = COMMON.validation_deadline


def validate_protocol(path: Path, *, repo: Path | None = None) -> dict[str, Any]:
    COMMON._check_deadline()
    repo = COMMON._directory(Path(repo or ROOT), "repo")
    path = _regular(Path(path).absolute(), "checkpoint smoke protocol")
    COMMON._verify_head_protocol(repo, path)
    protocol = COMMON._exact(_read(path), PROTOCOL_KEYS, "checkpoint smoke protocol")
    if (
        protocol["schema"] != PROTOCOL_SCHEMA
        or protocol["mode"] != MODE
        or protocol["registered_before_measurement"] is not True
        or not isinstance(protocol["protocol_id"], str)
        or COMMON.SLUG.fullmatch(protocol["protocol_id"]) is None
        or protocol["limits"] != LIMITS
    ):
        raise ValueError("checkpoint smoke protocol 기본 계약이 다르다")
    commit = protocol["source_commit"]
    if not isinstance(commit, str) or COMMON.HEX40.fullmatch(commit) is None:
        raise ValueError("checkpoint smoke source commit이 잘못됐다")
    if COMMON._git(repo, "merge-base", "--is-ancestor", commit, "HEAD").returncode:
        raise ValueError("checkpoint smoke source commit이 HEAD 조상이 아니다")
    source = COMMON._source_map(protocol["source_sha256_lf"])
    required = {
        "bench/vision_checkpoint_smoke_job_v2.py",
        "bench/vision_checkpoint_smoke_sdk_v2.py",
        "bench/run_vision_checkpoint_smoke_v2.py",
        "tests/test_vision_checkpoint_smoke_job_v2.py",
        "tests/test_vision_checkpoint_smoke_sdk_v2.py",
        "tests/test_run_vision_checkpoint_smoke_v2.py",
    }
    if not required <= source.keys():
        raise ValueError("checkpoint smoke source closure가 부족하다")
    COMMON._verify_source_blobs(repo, commit, source)
    base_ref = COMMON._exact(
        protocol["base_record_protocol"], {"path", "sha256_lf"}, "base record protocol"
    )
    base_path = _regular(
        repo / COMMON._relative(base_ref["path"], "base protocol"), "base protocol"
    )
    if _sha(base_path, lf=True) != COMMON._hash(
        base_ref["sha256_lf"], "base protocol SHA"
    ):
        raise ValueError("base record protocol SHA가 다르다")
    base = RECORD_JOB.validate_protocol(base_path, repo=repo)
    if COMMON._git(
        repo, "merge-base", "--is-ancestor", base["source_commit"], commit
    ).returncode or any(
        source.get(k) != v for k, v in base["source_sha256_lf"].items()
    ):
        raise ValueError(
            "checkpoint smoke source가 base record source를 보존하지 않는다"
        )
    prereg = COMMON._exact(
        protocol["preregistration"],
        {"prediction", "success_phrase", "failure_phrase", "fixed_flags"},
        "checkpoint smoke preregistration",
    )
    flags = prereg["fixed_flags"]
    expected_flags = {
        "checkpoint_specific": True,
        "used_for_decision": False,
        "no_universal_zero_lag_claim": True,
        "recording_labels_forbidden": True,
    }
    if (
        any(
            not isinstance(prereg[key], str) or not prereg[key].strip()
            for key in ("prediction", "success_phrase", "failure_phrase")
        )
        or prereg["success_phrase"] == prereg["failure_phrase"]
        or flags != expected_flags
    ):
        raise ValueError("checkpoint smoke 사전등록이 다르다")
    protocol["_base"] = base
    protocol["_base_path"] = base_path
    return protocol


def validate_job(
    job_path: Path, *, now: datetime | None = None, require_fresh: bool = True
) -> dict[str, Any]:
    job_path = _regular(Path(job_path).absolute(), "checkpoint smoke job")
    job = COMMON._exact(_read(job_path), JOB_KEYS, "checkpoint smoke job")
    if job["schema"] != JOB_SCHEMA or job["mode"] != MODE:
        raise ValueError("checkpoint smoke job schema/mode가 다르다")
    source = COMMON._exact(job["source"], {"repo_root", "commit", "files_lf"}, "source")
    repo = COMMON._directory(Path(source["repo_root"]), "source repo")
    protocol_ref = COMMON._exact(job["protocol"], {"path", "sha256_lf"}, "protocol ref")
    protocol_path = _regular(Path(protocol_ref["path"]), "protocol path")
    protocol_sha = COMMON._hash(protocol_ref["sha256_lf"], "protocol SHA")
    if _sha(protocol_path, lf=True) != protocol_sha:
        raise ValueError("checkpoint smoke protocol SHA가 다르다")
    protocol = validate_protocol(protocol_path, repo=repo)
    if source != {
        "repo_root": str(repo),
        "commit": protocol["source_commit"],
        "files_lf": protocol["source_sha256_lf"],
    }:
        raise ValueError("checkpoint smoke job source가 protocol과 다르다")
    if job["profile"] != RECORD.fixed_profile():
        raise ValueError("checkpoint smoke Play profile이 다르다")
    lab = COMMON._directory(Path(job["isaaclab"]), "isaaclab")
    checkpoint = COMMON._exact(
        job["checkpoint"], {"path", "sha256", "logical_id"}, "checkpoint"
    )
    row, checkpoint_path = RECORD_JOB._checkpoint(
        protocol["_base"], checkpoint["logical_id"], lab
    )
    if checkpoint != {
        "path": str(checkpoint_path),
        "sha256": row["sha256"],
        "logical_id": row["logical_id"],
    }:
        raise ValueError("checkpoint smoke checkpoint 결합이 다르다")
    expected_id = f"{protocol['protocol_id']}-{row['logical_id']}"
    if job["job_id"] != expected_id or COMMON.SLUG.fullmatch(expected_id) is None:
        raise ValueError("checkpoint smoke job ID가 다르다")
    resource_ref = COMMON._exact(
        job["resource_proof"], {"path", "sha256"}, "resource ref"
    )
    resource_path = _regular(Path(resource_ref["path"]), "resource proof")
    base = protocol["_base"]
    parent = SOURCE.validate_protocol(
        _read(repo / base["parent_source_protocol"]["path"])
    )
    current = now or datetime.now(timezone.utc)  # noqa: UP017 - Python 3.10 지원
    resource = COMMON._resource(
        resource_path,
        COMMON._hash(resource_ref["sha256"], "resource SHA"),
        parent,
        base["parent_source_protocol"]["sha256_lf"],
        current,
        require_fresh,
    )
    COMMON._python_executable(Path(job["python"]))
    COMMON._verify_runtime_sources(lab, parent)
    output = Path(job["output_directory"])
    if not output.is_absolute() or output != job_path.parent / "output":
        raise ValueError("checkpoint smoke output은 job 옆 output이어야 한다")
    COMMON._check_ancestors(output.parent, "checkpoint smoke output")
    if require_fresh and (output.exists() or output.is_symlink()):
        raise FileExistsError(output)
    if not require_fresh:
        output = COMMON._directory(output, "checkpoint smoke output")
    return {
        "job_path": job_path,
        "job_sha256": _sha(job_path),
        "job": job,
        "protocol": protocol,
        "protocol_path": protocol_path,
        "protocol_sha256": protocol_sha,
        "output_directory": output,
        "repo": repo,
        "resource": resource,
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": row["sha256"],
    }


def create_job(
    protocol_path: Path,
    resource_path: Path,
    resource_sha: str,
    isaaclab: Path,
    python: Path,
    stage: Path,
    checkpoint_logical_id: str,
) -> Path:
    protocol_path = Path(protocol_path).absolute()
    protocol = validate_protocol(protocol_path)
    lab = COMMON._directory(Path(isaaclab).absolute(), "isaaclab")
    row, checkpoint = RECORD_JOB._checkpoint(
        protocol["_base"], checkpoint_logical_id, lab
    )
    stage = Path(stage).absolute()
    COMMON._check_ancestors(stage, "checkpoint smoke stage")
    if stage.exists() or stage.is_symlink():
        raise FileExistsError(stage)
    stage.mkdir(parents=True, exist_ok=False)
    job_path = stage / "job.json"
    write_new(
        job_path,
        {
            "schema": JOB_SCHEMA,
            "mode": MODE,
            "job_id": f"{protocol['protocol_id']}-{checkpoint_logical_id}",
            "source": {
                "repo_root": str(ROOT),
                "commit": protocol["source_commit"],
                "files_lf": protocol["source_sha256_lf"],
            },
            "protocol": {
                "path": str(protocol_path),
                "sha256_lf": _sha(protocol_path, lf=True),
            },
            "resource_proof": {
                "path": str(Path(resource_path).absolute()),
                "sha256": resource_sha,
            },
            "isaaclab": str(lab),
            "python": str(Path(python).absolute()),
            "output_directory": str(stage / "output"),
            "profile": RECORD.fixed_profile(),
            "checkpoint": {
                "path": str(checkpoint),
                "sha256": row["sha256"],
                "logical_id": checkpoint_logical_id,
            },
        },
    )
    validate_job(job_path)
    return job_path
