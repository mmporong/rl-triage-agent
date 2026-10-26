"""준비 분리 정책 녹화의 source·checkpoint·GPU·누적 비용 계약."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / "bench", ROOT / "evals", ROOT / "src"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import vision_prepared_job_v1 as PREPARED

COMMON = PREPARED.COMMON
SOURCE = COMMON.SOURCE
RECORD = COMMON.RECORD
MODE = "prepared_policy_record_only"
PROTOCOL_SCHEMA = "vision_prepared_record_protocol_v1"
JOB_SCHEMA = "vision_prepared_record_job_v1"
RESULT_SCHEMA = "vision_prepared_record_result_v1"
FAILURE_SCHEMA = "vision_prepared_record_failure_v1"
PROTOCOL_KEYS = PREPARED.PROTOCOL_KEYS | {"checkpoints"}
JOB_KEYS = COMMON.JOB_KEYS | {"checkpoint", "profile"}
LIMITS = {
    "views": 1,
    "max_wall_s": 600,
    "frames": 625,
    "fps": 50,
    "callbacks": 627,
    "max_registered_checkpoints": 6,
    "training_actions": 0,
    "model_api_requests": 0,
    "full_recordings_per_job": 1,
}
FIXED_FLAGS = {
    "recording_only": True,
    "used_for_decision": False,
    "no_universal_zero_lag_claim": True,
    "no_duplicate_normalization": True,
    "smoke_required_before_model_use": True,
}
REQUIRED_SOURCE_FILES = PREPARED.REQUIRED_SOURCE_FILES | {
    "bench/protocols/vision_brev_source_20261026_v4.json",
    "bench/protocols/vision_prepared_capture_20261026_v1.json",
    "evals/results/vision_prepared_capture_dev_20261026t/summary.json",
    "bench/vision_prepared_record_v1.py",
    "bench/vision_prepared_record_sdk_v1.py",
    "bench/vision_record_job_v1.py",
    "bench/run_vision_prepared_record_v1.py",
    "tests/test_vision_prepared_record_v1.py",
    "tests/test_vision_prepared_record_sdk_v1.py",
    "tests/test_vision_record_job_v1.py",
    "tests/test_run_vision_prepared_record_v1.py",
    "bench/run_vision_source_prefix_v1.py",
    "tests/test_run_vision_source_prefix_v1.py",
}
_read = COMMON._read
_sha = COMMON._sha
_regular = COMMON._regular
_check_ancestors = COMMON._check_ancestors
_verify_current_files = COMMON._verify_current_files
_verify_runtime_sources = COMMON._verify_runtime_sources
_python_executable = COMMON._python_executable
write_new = COMMON.write_new
validation_deadline = COMMON.validation_deadline


def validate_cumulative_resources(
    protocol: Mapping[str, Any],
    parent: Mapping[str, Any],
    *,
    repo: Path,
    source_commit: str,
) -> None:
    """정책 녹화/학습 prefix plate도 기존 smoke 비용 계보에 append한다."""
    current_resources = PREPARED._resource_shape(protocol["resources"])
    parent_resources = parent["resources"]
    if (
        parent_resources["owned_instance_id"] != current_resources["owned_instance_id"]
        or type(parent_resources["max_owned_instances"]) is not int
        or parent_resources["max_owned_instances"] != 1
        or PREPARED._amount(parent_resources["compute_quote_usd_per_h"], "source quote")
        != PREPARED._amount(current_resources["compute_quote_usd_per_h"], "quote")
        or PREPARED._amount(parent_resources["whole_batch_budget_usd"], "source fresh")
        != PREPARED._amount(current_resources["fresh_lifecycle_budget_usd"], "fresh")
    ):
        raise ValueError("record parent resource 계약이 다르다")
    current = protocol
    seen: set[str] = set()
    while True:
        predecessor = COMMON._exact(
            current["predecessor"], {"semantic", "appended_stop"}, "predecessor"
        )
        previous = PREPARED._protocol_reference(
            predecessor["semantic"], repo, source_commit
        )
        if previous.get("schema") not in {
            PROTOCOL_SCHEMA,
            "vision_source_prefix_protocol_v1",
        }:
            PREPARED._history(current, repo, source_commit, set())
            break
        identity = str(predecessor["semantic"])
        if identity in seen or len(seen) >= 16:
            raise ValueError("record budget history 순환 또는 상한 초과")
        seen.add(identity)
        previous_resources = PREPARED._resource_shape(previous.get("resources"))
        resources = PREPARED._resource_shape(current["resources"])
        previous_parent = PREPARED._parent(previous, repo, source_commit)
        if PREPARED._amount(
            previous_parent["resources"]["whole_batch_budget_usd"], "previous parent"
        ) != PREPARED._amount(
            previous_resources["fresh_lifecycle_budget_usd"], "previous fresh"
        ):
            raise ValueError("previous parent budget이 다르다")
        appended = COMMON._exact(
            predecessor["appended_stop"], {"path", "sha256"}, "appended stop"
        )
        path = _regular(
            repo / COMMON._relative(appended["path"], "appended path"), "appended stop"
        )
        if _sha(path) != COMMON._hash(appended["sha256"], "appended SHA"):
            raise ValueError("appended STOP SHA가 다르다")
        stop = _read(path)
        execution = stop.get("execution_identity")
        if (
            not isinstance(execution, Mapping)
            or not isinstance(previous.get("protocol_id"), str)
            or not previous["protocol_id"]
            or execution.get("run_id") != previous["protocol_id"]
            or execution.get("protocol_sha256")
            != previous["parent_source_protocol"]["sha256_lf"]
        ):
            raise ValueError("record STOP과 predecessor identity가 다르다")
        if resources["prior_stop_receipts"] != [
            *previous_resources["prior_stop_receipts"],
            appended,
        ]:
            raise ValueError("record STOP history가 append-only가 아니다")
        cost = stop.get("cost_estimate")
        if not isinstance(cost, Mapping):
            raise TypeError("record STOP 비용이 없다")
        if PREPARED._amount(resources["prior_compute_estimate_usd"], "prior") != (
            PREPARED._amount(
                previous_resources["prior_compute_estimate_usd"], "previous prior"
            )
            + PREPARED._amount(cost.get("estimated_compute_cost_usd"), "append cost")
        ):
            raise ValueError("record STOP 누적 비용이 다르다")
        current = previous
    COMMON._validate_prior_stops(
        current_resources, repo, source=parent, source_commit=source_commit
    )


def _checkpoints(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not 1 <= len(value) <= 6:
        raise ValueError("등록 checkpoint는 1~6건이어야 한다")
    result, ids, paths = [], set(), set()
    for raw in value:
        row = COMMON._exact(
            raw, {"logical_id", "isaaclab_relative", "sha256"}, "checkpoint"
        )
        identity = row["logical_id"]
        if not isinstance(identity, str) or COMMON.SLUG.fullmatch(identity) is None:
            raise ValueError("checkpoint logical_id가 잘못됐다")
        relative = COMMON._relative(row["isaaclab_relative"], "checkpoint relative")
        if not relative.startswith("logs/rsl_rl/") or not relative.endswith(".pt"):
            raise ValueError("checkpoint는 Isaac Lab logs/rsl_rl의 pt여야 한다")
        COMMON._hash(row["sha256"], "checkpoint SHA")
        if identity in ids or relative in paths:
            raise ValueError("checkpoint ID 또는 경로가 중복됐다")
        ids.add(identity)
        paths.add(relative)
        result.append(row)
    return result


def _preregistration(value: Any) -> dict[str, Any]:
    row = COMMON._exact(
        value, {"prediction", "observation_phrases", "fixed_flags"}, "preregistration"
    )
    phrases = row["observation_phrases"]
    if (
        not isinstance(row["prediction"], str)
        or not row["prediction"].strip()
        or not isinstance(phrases, dict)
        or set(phrases)
        != {
            "record_complete",
            "callback_hold",
            "alignment_hold",
            "infrastructure_failure",
        }
        or any(
            not isinstance(text, str) or not text.strip() for text in phrases.values()
        )
        or len(set(phrases.values())) != 4
        or row["fixed_flags"] != FIXED_FLAGS
    ):
        raise ValueError("녹화 사전등록의 문구·flags가 다르다")
    return row


def validate_protocol(path: Path, *, repo: Path | None = None) -> dict[str, Any]:
    COMMON._check_deadline()
    repo = COMMON._directory(Path(repo or ROOT), "repo")
    path = _regular(Path(path).absolute(), "record protocol")
    COMMON._verify_head_protocol(repo, path)
    protocol = COMMON._exact(_read(path), PROTOCOL_KEYS, "record protocol")
    if (
        protocol["schema"] != PROTOCOL_SCHEMA
        or protocol["mode"] != MODE
        or protocol["registered_before_measurement"] is not True
        or not isinstance(protocol["protocol_id"], str)
        or COMMON.SLUG.fullmatch(protocol["protocol_id"]) is None
    ):
        raise ValueError("record protocol schema/mode/등록/ID가 다르다")
    commit = protocol["source_commit"]
    if not isinstance(commit, str) or COMMON.HEX40.fullmatch(commit) is None:
        raise ValueError("source commit 형식이 다르다")
    if COMMON._git(repo, "merge-base", "--is-ancestor", commit, "HEAD").returncode:
        raise ValueError("source commit이 HEAD의 조상이 아니다")
    parent = PREPARED._parent(protocol, repo, commit)
    sources = COMMON._source_map(protocol["source_sha256_lf"])
    if not REQUIRED_SOURCE_FILES <= sources.keys() or any(
        sources.get(name) != digest
        for name, digest in parent["source_sha256_lf"].items()
    ):
        raise ValueError("record source closure가 다르다")
    COMMON._verify_source_blobs(repo, commit, sources)
    if (
        protocol["profile_sha256"] != RECORD.digest(RECORD.fixed_profile())
        or protocol["limits"] != LIMITS
    ):
        raise ValueError("녹화 profile 또는 실행 상한이 다르다")
    validate_cumulative_resources(protocol, parent, repo=repo, source_commit=commit)
    protocol["checkpoints"] = _checkpoints(protocol["checkpoints"])
    protocol["preregistration"] = _preregistration(protocol["preregistration"])
    COMMON._check_deadline()
    return protocol


def _checkpoint(
    protocol: Mapping[str, Any], identity: str, lab: Path
) -> tuple[dict[str, Any], Path]:
    rows = [row for row in protocol["checkpoints"] if row["logical_id"] == identity]
    if len(rows) != 1:
        raise ValueError("checkpoint ID가 사전등록되지 않았다")
    row = rows[0]
    path = _regular(lab / row["isaaclab_relative"], "checkpoint")
    path.relative_to(lab)
    if _sha(path) != row["sha256"]:
        raise ValueError("checkpoint SHA가 사전등록과 다르다")
    return row, path


def validate_job(
    job_path: Path, *, now: datetime | None = None, require_fresh: bool = True
) -> dict[str, Any]:
    COMMON._check_deadline()
    job_path = _regular(Path(job_path).absolute(), "record job")
    job = COMMON._exact(_read(job_path), JOB_KEYS, "record job")
    if job["schema"] != JOB_SCHEMA or job["mode"] != MODE:
        raise ValueError("record job schema/mode가 다르다")
    source = COMMON._exact(job["source"], {"repo_root", "commit", "files_lf"}, "source")
    repo = COMMON._directory(Path(source["repo_root"]), "source repo")
    reference = COMMON._exact(job["protocol"], {"path", "sha256_lf"}, "protocol ref")
    path = _regular(Path(reference["path"]), "protocol path")
    expected = COMMON._hash(reference["sha256_lf"], "protocol SHA")
    if _sha(path, lf=True) != expected:
        raise ValueError("protocol LF SHA가 다르다")
    protocol = validate_protocol(path, repo=repo)
    if (
        source["commit"] != protocol["source_commit"]
        or COMMON._source_map(source["files_lf"]) != protocol["source_sha256_lf"]
    ):
        raise ValueError("job source가 사전등록과 다르다")
    if job["profile"] != RECORD.fixed_profile():
        raise ValueError("job Play profile이 다르다")
    lab = COMMON._directory(Path(job["isaaclab"]), "Isaac Lab")
    checkpoint = COMMON._exact(
        job["checkpoint"], {"path", "sha256", "logical_id"}, "job checkpoint"
    )
    row, checkpoint_path = _checkpoint(protocol, checkpoint["logical_id"], lab)
    if checkpoint != {
        "path": str(checkpoint_path),
        "sha256": row["sha256"],
        "logical_id": row["logical_id"],
    }:
        raise ValueError("job checkpoint 결합이 다르다")
    expected_job_id = f"{protocol['protocol_id']}-{row['logical_id']}"
    if (
        job["job_id"] != expected_job_id
        or COMMON.SLUG.fullmatch(expected_job_id) is None
    ):
        raise ValueError("job ID가 protocol/checkpoint와 다르다")
    resource_ref = COMMON._exact(
        job["resource_proof"], {"path", "sha256"}, "resource ref"
    )
    resource_path = _regular(Path(resource_ref["path"]), "resource proof")
    parent = SOURCE.validate_protocol(
        _read(repo / protocol["parent_source_protocol"]["path"])
    )
    current = now or datetime.now(timezone.utc)  # noqa: UP017 - Python 3.10
    if current.tzinfo is None or current.utcoffset() is None:
        raise ValueError("현재 시각에 timezone이 필요하다")
    resource = COMMON._resource(
        resource_path,
        COMMON._hash(resource_ref["sha256"], "resource SHA"),
        parent,
        protocol["parent_source_protocol"]["sha256_lf"],
        current,
        require_fresh,
    )
    _python_executable(Path(job["python"]))
    _verify_runtime_sources(lab, parent)
    output = Path(job["output_directory"])
    if not output.is_absolute() or output != job_path.parent / "output":
        raise ValueError("output directory는 job 옆 output으로 고정된다")
    _check_ancestors(output.parent, "record output")
    if require_fresh:
        if output.exists() or output.is_symlink():
            raise FileExistsError(output)
    else:
        output = COMMON._directory(output, "record output")
    COMMON._check_deadline()
    return {
        "job_path": job_path,
        "job_sha256": _sha(job_path),
        "job": job,
        "protocol": protocol,
        "protocol_sha256": expected,
        "protocol_path": path,
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
    lab = COMMON._directory(Path(isaaclab).absolute(), "Isaac Lab")
    row, checkpoint = _checkpoint(protocol, checkpoint_logical_id, lab)
    stage = Path(stage).absolute()
    _check_ancestors(stage, "record stage")
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
