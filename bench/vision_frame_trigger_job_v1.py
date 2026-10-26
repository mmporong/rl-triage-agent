"""실제 checkpoint의 frame trigger 단일 후보 진단 계약."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import vision_prepared_job_v1 as PREPARED
import vision_record_job_v1 as RECORD_JOB
import vision_semantic_settle_job_v1 as PREVIOUS

COMMON = RECORD_JOB.COMMON
SOURCE = RECORD_JOB.SOURCE
RECORD = RECORD_JOB.RECORD
ROOT = RECORD_JOB.ROOT
MODE = "checkpoint_frame_trigger_diagnostic_only"
PROTOCOL_SCHEMA = "vision_frame_trigger_protocol_v1"
JOB_SCHEMA = "vision_frame_trigger_job_v1"
RESULT_SCHEMA = "vision_frame_trigger_result_v1"
FAILURE_SCHEMA = "vision_frame_trigger_failure_v1"
VARIANTS = ("default_frame_step",)
LIMITS = {
    "variants": 1,
    "runs_per_variant": 1,
    "max_wall_s_per_variant": 600,
    "max_batch_compute_usd": "1.000000",
    "training_actions": 0,
    "model_api_requests": 0,
    "capture_approval": False,
}
PROTOCOL_KEYS = {
    "schema",
    "mode",
    "protocol_id",
    "registered_before_measurement",
    "source_commit",
    "source_sha256_lf",
    "source_protocol",
    "base_record_protocol",
    "checkpoint_logical_id",
    "variants",
    "limits",
    "resources",
    "predecessor",
    "preregistration",
}
JOB_KEYS = COMMON.JOB_KEYS | {"checkpoint", "profile", "callback_variant"}
REQUIRED_SOURCE_FILES = PREVIOUS.REQUIRED_SOURCE_FILES | {
    "bench/vision_frame_trigger_checkpoint_v1.py",
    "bench/vision_frame_trigger_job_v1.py",
    "bench/run_vision_frame_trigger_v1.py",
    "tests/test_vision_frame_trigger_checkpoint_v1.py",
    "tests/test_vision_frame_trigger_job_v1.py",
    "tests/test_run_vision_frame_trigger_v1.py",
}
EXPECTED_TOTAL = Decimal("20.000000")

_read = COMMON._read
_sha = COMMON._sha
_regular = COMMON._regular
write_new = COMMON.write_new
validation_deadline = COMMON.validation_deadline


def _reference(value: Any, name: str) -> dict[str, str]:
    row = COMMON._exact(value, {"path", "sha256_lf"}, name)
    return {
        "path": COMMON._relative(row["path"], name),
        "sha256_lf": COMMON._hash(row["sha256_lf"], name + " SHA"),
    }


def _budget(
    protocol: Mapping[str, Any], previous: Mapping[str, Any], repo: Path
) -> None:
    resources = PREPARED._resource_shape(protocol["resources"])
    previous_resources = PREPARED._resource_shape(previous["resources"])
    predecessor = COMMON._exact(
        protocol["predecessor"],
        {"diagnostic", "appended_stop"},
        "frame trigger predecessor",
    )
    previous_ref = _reference(predecessor["diagnostic"], "previous diagnostic")
    if previous_ref != {
        "path": previous["_protocol_path"].relative_to(repo).as_posix(),
        "sha256_lf": previous["_protocol_sha256"],
    }:
        raise ValueError("frame trigger predecessor가 이전 진단과 다르다")
    appended = COMMON._exact(
        predecessor["appended_stop"], {"path", "sha256"}, "frame trigger STOP"
    )
    stop_path = _regular(
        repo / COMMON._relative(appended["path"], "frame trigger STOP"),
        "frame trigger STOP",
    )
    if _sha(stop_path) != COMMON._hash(appended["sha256"], "frame trigger STOP SHA"):
        raise ValueError("frame trigger STOP SHA가 다르다")
    stop = _read(stop_path)
    identity, cost = stop.get("execution_identity"), stop.get("cost_estimate")
    stop_cost = (
        PREPARED._amount(
            cost.get("estimated_compute_cost_usd"), "frame trigger STOP cost"
        )
        if isinstance(cost, Mapping)
        else Decimal(-1)
    )
    previous_prior = PREPARED._amount(
        previous_resources["prior_compute_estimate_usd"], "previous diagnostic prior"
    )
    prior = PREPARED._amount(
        resources["prior_compute_estimate_usd"], "diagnostic v2 prior"
    )
    fresh = PREPARED._amount(
        resources["fresh_lifecycle_budget_usd"], "diagnostic v2 fresh"
    )
    total = PREPARED._amount(
        resources["total_program_budget_usd"], "diagnostic v2 total"
    )
    if (
        stop.get("status") != "STOPPED"
        or not isinstance(identity, Mapping)
        or identity.get("run_id") != previous["protocol_id"]
        or identity.get("protocol_sha256") != previous["source_protocol"]["sha256_lf"]
        or stop_cost < 0
        or prior != previous_prior + stop_cost
        or total != EXPECTED_TOTAL
        or fresh != total - prior
        or resources["prior_stop_receipts"]
        != [*previous_resources["prior_stop_receipts"], appended]
    ):
        raise ValueError("frame trigger 16 STOP 누적 원장이 다르다")


def validate_protocol(path: Path, *, repo: Path | None = None) -> dict[str, Any]:
    COMMON._check_deadline()
    repo = COMMON._directory(Path(repo or ROOT), "repo")
    path = _regular(Path(path).absolute(), "callback diagnostic protocol")
    COMMON._verify_head_protocol(repo, path)
    protocol = COMMON._exact(_read(path), PROTOCOL_KEYS, "callback diagnostic protocol")
    if (
        protocol["schema"] != PROTOCOL_SCHEMA
        or protocol["mode"] != MODE
        or protocol["registered_before_measurement"] is not True
        or not isinstance(protocol["protocol_id"], str)
        or COMMON.SLUG.fullmatch(protocol["protocol_id"]) is None
        or protocol["variants"] != list(VARIANTS)
        or protocol["limits"] != LIMITS
        or protocol["checkpoint_logical_id"] != "source-s2220-f1"
    ):
        raise ValueError("callback diagnostic protocol 기본 계약이 다르다")
    commit = protocol["source_commit"]
    if (
        not isinstance(commit, str)
        or COMMON.HEX40.fullmatch(commit) is None
        or COMMON._git(repo, "merge-base", "--is-ancestor", commit, "HEAD").returncode
    ):
        raise ValueError("callback diagnostic source commit이 HEAD 조상이 아니다")
    source_map = COMMON._source_map(protocol["source_sha256_lf"])
    if not REQUIRED_SOURCE_FILES <= source_map.keys():
        raise ValueError("callback diagnostic source closure가 부족하다")
    COMMON._verify_source_blobs(repo, commit, source_map)

    predecessor = COMMON._exact(
        protocol["predecessor"],
        {"diagnostic", "appended_stop"},
        "frame trigger predecessor",
    )
    previous_ref = _reference(predecessor["diagnostic"], "previous diagnostic")
    previous_path = _regular(repo / previous_ref["path"], "previous diagnostic")
    if _sha(previous_path, lf=True) != previous_ref["sha256_lf"]:
        raise ValueError("previous diagnostic protocol SHA가 다르다")
    previous = PREVIOUS.validate_protocol(previous_path, repo=repo)
    previous["_protocol_path"] = previous_path
    previous["_protocol_sha256"] = previous_ref["sha256_lf"]
    if any(
        source_map.get(key) != value
        for key, value in previous["source_sha256_lf"].items()
    ):
        raise ValueError("frame trigger가 semantic settle source를 보존하지 않는다")

    base_ref = _reference(protocol["base_record_protocol"], "base record protocol")
    if base_ref != previous["base_record_protocol"]:
        raise ValueError("frame trigger base가 semantic settle과 다르다")
    base_path = _regular(repo / base_ref["path"], "base record protocol")
    if _sha(base_path, lf=True) != base_ref["sha256_lf"]:
        raise ValueError("base record protocol SHA가 다르다")
    base = RECORD_JOB.validate_protocol(base_path, repo=repo)

    source_ref = _reference(protocol["source_protocol"], "source protocol")
    if (
        source_ref != previous["source_protocol"]
        or source_ref != base["parent_source_protocol"]
    ):
        raise ValueError("frame trigger source protocol 계보가 다르다")
    source_path = _regular(repo / source_ref["path"], "source protocol")
    if _sha(source_path, lf=True) != source_ref["sha256_lf"]:
        raise ValueError("source protocol SHA가 다르다")
    source_protocol = SOURCE.validate_protocol(_read(source_path))
    _budget(protocol, previous, repo)
    row = next(
        (
            item
            for item in base["checkpoints"]
            if item["logical_id"] == protocol["checkpoint_logical_id"]
        ),
        None,
    )
    if row is None:
        raise ValueError("diagnostic checkpoint가 base catalog에 없다")
    prereg = COMMON._exact(
        protocol["preregistration"],
        {"prediction", "outcomes", "fixed_flags"},
        "diagnostic preregistration",
    )
    if (
        not isinstance(prereg["prediction"], str)
        or not prereg["prediction"].strip()
        or not isinstance(prereg["outcomes"], dict)
        or set(prereg["outcomes"])
        != {
            "exact_one_all_arms",
            "duplicate_callback",
            "changed_physics_or_timing",
            "infrastructure_failure",
        }
        or any(
            not isinstance(value, str) or not value.strip()
            for value in prereg["outcomes"].values()
        )
        or prereg["fixed_flags"]
        != {
            "development_diagnostic": True,
            "used_for_decision": False,
            "callback_drops": 0,
            "default_writer_trigger": "OgnOnFrame",
            "send_og_event_underlying_calls": 0,
            "schedule_write_calls": 0,
            "capture_approval": False,
        }
    ):
        raise ValueError("callback diagnostic 사전등록이 다르다")
    (
        protocol["_base"],
        protocol["_source"],
        protocol["_base_path"],
        protocol["_previous"],
    ) = base, source_protocol, base_path, previous
    return protocol


def validate_job(
    job_path: Path, *, now: datetime | None = None, require_fresh: bool = True
) -> dict[str, Any]:
    job_path = _regular(Path(job_path).absolute(), "callback diagnostic job")
    job = COMMON._exact(_read(job_path), JOB_KEYS, "callback diagnostic job")
    if (
        job["schema"] != JOB_SCHEMA
        or job["mode"] != MODE
        or job["callback_variant"] not in VARIANTS
    ):
        raise ValueError("callback diagnostic job schema/mode/variant가 다르다")
    source = COMMON._exact(job["source"], {"repo_root", "commit", "files_lf"}, "source")
    repo = COMMON._directory(Path(source["repo_root"]), "source repo")
    protocol_ref = COMMON._exact(job["protocol"], {"path", "sha256_lf"}, "protocol ref")
    protocol_path = _regular(Path(protocol_ref["path"]), "protocol path")
    protocol_sha = COMMON._hash(protocol_ref["sha256_lf"], "protocol SHA")
    if _sha(protocol_path, lf=True) != protocol_sha:
        raise ValueError("callback diagnostic protocol SHA가 다르다")
    protocol = validate_protocol(protocol_path, repo=repo)
    if source != {
        "repo_root": str(repo),
        "commit": protocol["source_commit"],
        "files_lf": protocol["source_sha256_lf"],
    }:
        raise ValueError("callback diagnostic job source가 protocol과 다르다")
    if job["profile"] != RECORD.fixed_profile():
        raise ValueError("callback diagnostic profile이 다르다")
    lab = COMMON._directory(Path(job["isaaclab"]), "isaaclab")
    checkpoint = COMMON._exact(
        job["checkpoint"], {"path", "sha256", "logical_id"}, "checkpoint"
    )
    row, checkpoint_path = RECORD_JOB._checkpoint(
        protocol["_base"], protocol["checkpoint_logical_id"], lab
    )
    if checkpoint != {
        "path": str(checkpoint_path),
        "sha256": row["sha256"],
        "logical_id": row["logical_id"],
    }:
        raise ValueError("callback diagnostic checkpoint 결합이 다르다")
    expected_id = f"{protocol['protocol_id']}-{job['callback_variant']}"
    if job["job_id"] != expected_id or COMMON.SLUG.fullmatch(expected_id) is None:
        raise ValueError("callback diagnostic job ID가 다르다")
    resource_ref = COMMON._exact(
        job["resource_proof"], {"path", "sha256"}, "resource ref"
    )
    resource_path = _regular(Path(resource_ref["path"]), "resource proof")
    current = now or datetime.now(timezone.utc)  # noqa: UP017 - Isaac Python 3.10
    resource = COMMON._resource(
        resource_path,
        COMMON._hash(resource_ref["sha256"], "resource SHA"),
        protocol["_source"],
        protocol["source_protocol"]["sha256_lf"],
        current,
        require_fresh,
    )
    COMMON._python_executable(Path(job["python"]))
    COMMON._verify_runtime_sources(lab, protocol["_source"])
    output = Path(job["output_directory"])
    if not output.is_absolute() or output != job_path.parent / "output":
        raise ValueError("callback diagnostic output 경계가 다르다")
    COMMON._check_ancestors(output.parent, "callback diagnostic output")
    if require_fresh and (output.exists() or output.is_symlink()):
        raise FileExistsError(output)
    if not require_fresh:
        output = COMMON._directory(output, "callback diagnostic output")
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
    callback_variant: str,
) -> Path:
    protocol_path = Path(protocol_path).absolute()
    protocol = validate_protocol(protocol_path)
    if callback_variant not in VARIANTS:
        raise ValueError("callback diagnostic variant가 등록값이 아니다")
    lab = COMMON._directory(Path(isaaclab).absolute(), "isaaclab")
    row, checkpoint = RECORD_JOB._checkpoint(
        protocol["_base"], protocol["checkpoint_logical_id"], lab
    )
    stage = Path(stage).absolute()
    COMMON._check_ancestors(stage, "callback diagnostic stage")
    if stage.exists() or stage.is_symlink():
        raise FileExistsError(stage)
    stage.mkdir(parents=True, exist_ok=False)
    job_path = stage / "job.json"
    write_new(
        job_path,
        {
            "schema": JOB_SCHEMA,
            "mode": MODE,
            "job_id": f"{protocol['protocol_id']}-{callback_variant}",
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
                "logical_id": row["logical_id"],
            },
            "callback_variant": callback_variant,
        },
    )
    validate_job(job_path)
    return job_path
