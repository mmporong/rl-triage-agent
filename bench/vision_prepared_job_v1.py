"""준비 프레임을 분리한 renderer smoke의 출처·예산·입장 계약."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import vision_semantic_fingerprint_job_v1 as COMMON

ROOT = COMMON.ROOT
RECORD = COMMON.RECORD
SOURCE = COMMON.SOURCE
HISTORY_PROTOCOL_PATH = COMMON.HISTORY_PROTOCOL_PATH
MODE = "prepared_capture_smoke_only"
PROTOCOL_SCHEMA = "vision_prepared_capture_protocol_v1"
JOB_SCHEMA = "vision_prepared_capture_job_v1"
RESULT_SCHEMA = "vision_prepared_capture_result_v1"
FAILURE_SCHEMA = "vision_prepared_capture_failure_v1"
ANCHOR = {
    "path": "bench/protocols/vision_semantic_sdk_20261026_v3.json",
    "sha256_lf": "269c124ce18a534683f4708723f258efe40738291d49104208ffb1f4a6ffbd90",
}
LIMITS = {
    "views": 1,
    "events": 5,
    "callbacks": 16,
    "logical_frames": 4,
    "max_wall_s": 600,
    "training_actions": 0,
    "model_api_requests": 0,
    "full_recordings": 0,
    "renderer_smoke_only": True,
}
FIXED_FLAGS = {
    "development_measurement": True,
    "used_for_decision": False,
    "no_universal_zero_lag_claim": True,
    "no_full_recording": True,
    "no_duplicate_normalization": True,
}
REQUIRED_SOURCE_FILES = COMMON.REQUIRED_SOURCE_FILES | {
    ANCHOR["path"],
    "bench/protocols/vision_brev_source_20261026_v3.json",
    "bench/vision_prepared_capture_v1.py",
    "bench/vision_prepared_sdk_v1.py",
    "bench/vision_prepared_job_v1.py",
    "bench/run_vision_prepared_capture_v1.py",
    "tests/test_vision_prepared_capture_v1.py",
    "tests/test_vision_prepared_sdk_v1.py",
    "tests/test_vision_prepared_job_v1.py",
    "tests/test_run_vision_prepared_capture_v1.py",
    "evals/results/vision_semantic_fingerprint_dev_20261026s/summary.json",
}
PROTOCOL_KEYS = COMMON.PROTOCOL_KEYS | {"predecessor"}

# 고정 helper를 합성한다. 이 파일과 helper 원문은 새 protocol closure에 함께 결합한다.
_read = COMMON._read
_regular = COMMON._regular
_sha = COMMON._sha
_check_ancestors = COMMON._check_ancestors
_verify_current_files = COMMON._verify_current_files
_verify_runtime_sources = COMMON._verify_runtime_sources
_python_executable = COMMON._python_executable
write_new = COMMON.write_new
validation_deadline = COMMON.validation_deadline


def _amount(value: Any, name: str, *, positive: bool = False) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise TypeError(f"{name} 비용 형식이 잘못됐다")
    try:
        parsed = Decimal(str(value))
    except InvalidOperation as error:
        raise ValueError(f"{name} 비용을 읽을 수 없다") from error
    if not parsed.is_finite() or parsed < 0 or (positive and parsed == 0):
        raise ValueError(f"{name} 비용이 유효 범위 밖이다")
    return parsed


def _resource_shape(value: Any) -> dict[str, Any]:
    resources = COMMON._exact(value, COMMON.RESOURCE_KEYS, "prepared resources")
    prior = _amount(resources["prior_compute_estimate_usd"], "prior")
    fresh = _amount(resources["fresh_lifecycle_budget_usd"], "fresh", positive=True)
    if (
        resources["owned_instance_id"] != "3655a80oo"
        or resources["owned_instance_name"] != "rl-triage-e2e-20261024-c9f4"
        or type(resources["max_owned_instances"]) is not int
        or resources["max_owned_instances"] != 1
        or _amount(resources["compute_quote_usd_per_h"], "quote") != Decimal("1.74")
        or _amount(resources["total_program_budget_usd"], "total") != Decimal(20)
        or prior + fresh != Decimal(20)
    ):
        raise ValueError("prepared resource budget/owner 계약이 다르다")
    return resources


def _protocol_reference(value: Any, repo: Path, commit: str) -> dict[str, Any]:
    reference = COMMON._exact(value, {"path", "sha256_lf"}, "previous protocol")
    relative = COMMON._relative(reference["path"], "previous protocol.path")
    expected = COMMON._hash(reference["sha256_lf"], "previous protocol.sha256_lf")
    COMMON._verify_blob(repo, commit, relative, expected)
    return _read(repo / relative)


def _parent(protocol: Mapping[str, Any], repo: Path, commit: str) -> dict[str, Any]:
    parent = _protocol_reference(protocol["parent_source_protocol"], repo, commit)
    parent = SOURCE.validate_protocol(parent)
    if not isinstance(parent.get("protocol_id"), str) or not parent["protocol_id"]:
        raise ValueError("parent source protocol ID가 없다")
    return parent


def _history(
    protocol: Mapping[str, Any], repo: Path, commit: str, seen: set[str]
) -> None:
    """기록 누락으로 fresh 예산이 늘지 않도록 고정 s 판까지 역으로 연결한다."""
    COMMON._check_deadline()
    predecessor = COMMON._exact(
        protocol["predecessor"], {"semantic", "appended_stop"}, "predecessor"
    )
    reference = predecessor["semantic"]
    previous = _protocol_reference(reference, repo, commit)
    identity = str(reference)
    if identity in seen or len(seen) >= 16:
        raise ValueError("predecessor chain 순환 또는 깊이 상한 초과")
    seen.add(identity)
    previous_resources = _resource_shape(previous.get("resources"))
    current_resources = _resource_shape(protocol.get("resources"))
    previous_parent = _parent(previous, repo, commit)
    if _amount(
        previous_parent["resources"]["whole_batch_budget_usd"], "previous source"
    ) != _amount(previous_resources["fresh_lifecycle_budget_usd"], "previous fresh"):
        raise ValueError("predecessor source 예산이 다르다")
    if reference == ANCHOR:
        if previous.get("schema") != COMMON.PROTOCOL_SCHEMA:
            raise ValueError("고정 budget anchor schema가 다르다")
    elif previous.get("schema") == PROTOCOL_SCHEMA:
        _history(previous, repo, commit, seen)
    else:
        raise ValueError("고정 budget anchor까지 이어지지 않는다")

    appended = COMMON._exact(
        predecessor["appended_stop"], {"path", "sha256"}, "appended stop"
    )
    relative = COMMON._relative(appended["path"], "appended stop.path")
    expected = COMMON._hash(appended["sha256"], "appended stop.sha256")
    path = _regular(repo / relative, "appended stop")
    if _sha(path) != expected:
        raise ValueError("appended STOP SHA가 다르다")
    stop = _read(path)
    execution = stop.get("execution_identity")
    if (
        not isinstance(execution, Mapping)
        or execution.get("run_id") != previous.get("protocol_id")
        or not isinstance(previous.get("protocol_id"), str)
        or execution.get("protocol_sha256")
        != previous["parent_source_protocol"]["sha256_lf"]
    ):
        raise ValueError("appended STOP과 predecessor 실행 결합이 다르다")
    if current_resources["prior_stop_receipts"] != [
        *previous_resources["prior_stop_receipts"],
        appended,
    ]:
        raise ValueError("prior STOP 목록이 predecessor append-only 계약과 다르다")
    cost = stop.get("cost_estimate")
    if not isinstance(cost, Mapping):
        raise TypeError("appended STOP 비용이 없다")
    if _amount(current_resources["prior_compute_estimate_usd"], "prior") != (
        _amount(previous_resources["prior_compute_estimate_usd"], "previous prior")
        + _amount(cost.get("estimated_compute_cost_usd"), "appended cost")
    ):
        raise ValueError("predecessor 누적 비용 계산이 다르다")


def validate_cumulative_resources(
    protocol: Mapping[str, Any],
    parent: Mapping[str, Any],
    *,
    repo: Path,
    source_commit: str,
) -> dict[str, str]:
    resources = _resource_shape(protocol["resources"])
    parent_resources = parent["resources"]
    if (
        parent_resources["owned_instance_id"] != resources["owned_instance_id"]
        or type(parent_resources["max_owned_instances"]) is not int
        or parent_resources["max_owned_instances"] != 1
        or _amount(parent_resources["compute_quote_usd_per_h"], "source quote")
        != Decimal("1.74")
        or _amount(parent_resources["whole_batch_budget_usd"], "source fresh")
        != _amount(resources["fresh_lifecycle_budget_usd"], "fresh")
    ):
        raise ValueError("parent source resource 계약이 다르다")
    _history(protocol, repo, source_commit, set())
    COMMON._validate_prior_stops(
        resources, repo, source=parent, source_commit=source_commit
    )
    return {
        "prior_usd": str(_amount(resources["prior_compute_estimate_usd"], "prior")),
        "fresh_usd": str(_amount(resources["fresh_lifecycle_budget_usd"], "fresh")),
    }


def _preregistration(value: Any) -> dict[str, Any]:
    prereg = COMMON._exact(
        value, {"prediction", "observation_phrases", "fixed_flags"}, "preregistration"
    )
    phrases = prereg["observation_phrases"]
    if (
        not isinstance(prereg["prediction"], str)
        or not prereg["prediction"].strip()
        or not isinstance(phrases, dict)
        or set(phrases)
        != {"sync_pass", "callback_hold", "sync_hold", "infrastructure_failure"}
        or any(
            not isinstance(text, str) or not text.strip() for text in phrases.values()
        )
        or len(set(phrases.values())) != 4
        or prereg["fixed_flags"] != FIXED_FLAGS
    ):
        raise ValueError("prepared preregistration 내용/문구/flags가 다르다")
    return prereg


def validate_protocol(path: Path, *, repo: Path | None = None) -> dict[str, Any]:
    COMMON._check_deadline()
    repo = COMMON._directory(Path(repo or ROOT), "repo")
    path = _regular(Path(path).absolute(), "protocol")
    COMMON._verify_head_protocol(repo, path)
    protocol = COMMON._exact(_read(path), PROTOCOL_KEYS, "prepared protocol")
    if (
        protocol["schema"] != PROTOCOL_SCHEMA
        or protocol["mode"] != MODE
        or protocol["registered_before_measurement"] is not True
        or not isinstance(protocol["protocol_id"], str)
        or COMMON.SLUG.fullmatch(protocol["protocol_id"]) is None
    ):
        raise ValueError("prepared protocol schema/mode/등록/ID가 다르다")
    commit = protocol["source_commit"]
    if not isinstance(commit, str) or COMMON.HEX40.fullmatch(commit) is None:
        raise ValueError("source_commit 형식이 다르다")
    if COMMON._git(repo, "merge-base", "--is-ancestor", commit, "HEAD").returncode:
        raise ValueError("source_commit이 HEAD의 조상이 아니다")
    parent = _parent(protocol, repo, commit)
    sources = COMMON._source_map(protocol["source_sha256_lf"])
    if not REQUIRED_SOURCE_FILES <= sources.keys() or any(
        sources.get(name) != digest
        for name, digest in parent["source_sha256_lf"].items()
    ):
        raise ValueError("prepared source closure가 누락되거나 parent와 다르다")
    COMMON._verify_source_blobs(repo, commit, sources)
    if (
        protocol["profile_sha256"] != RECORD.digest(RECORD.fixed_profile())
        or protocol["limits"] != LIMITS
    ):
        raise ValueError("고정 profile 또는 실행 상한이 다르다")
    validate_cumulative_resources(protocol, parent, repo=repo, source_commit=commit)
    protocol["preregistration"] = _preregistration(protocol["preregistration"])
    protocol["source_sha256_lf"] = sources
    COMMON._check_deadline()
    return protocol


def validate_job(
    job_path: Path, *, now: datetime | None = None, require_fresh: bool = True
) -> dict[str, Any]:
    COMMON._check_deadline()
    job_path = _regular(Path(job_path).absolute(), "job")
    job = COMMON._exact(_read(job_path), COMMON.JOB_KEYS, "job")
    if job["schema"] != JOB_SCHEMA or job["mode"] != MODE:
        raise ValueError("prepared job schema/mode가 다르다")
    if (
        not isinstance(job["job_id"], str)
        or COMMON.SLUG.fullmatch(job["job_id"]) is None
    ):
        raise ValueError("job ID가 잘못됐다")
    source = COMMON._exact(job["source"], {"repo_root", "commit", "files_lf"}, "source")
    repo = COMMON._directory(Path(source["repo_root"]), "source.repo_root")
    protocol_ref = COMMON._exact(job["protocol"], {"path", "sha256_lf"}, "protocol")
    protocol_path = _regular(Path(protocol_ref["path"]), "protocol path")
    if _sha(protocol_path, lf=True) != COMMON._hash(
        protocol_ref["sha256_lf"], "protocol SHA"
    ):
        raise ValueError("protocol LF SHA가 다르다")
    protocol = validate_protocol(protocol_path, repo=repo)
    if job["job_id"] != protocol["protocol_id"]:
        raise ValueError("job ID가 protocol 실행 ID와 다르다")
    if (
        source["commit"] != protocol["source_commit"]
        or COMMON._source_map(source["files_lf"]) != protocol["source_sha256_lf"]
    ):
        raise ValueError("job source와 protocol 결합이 다르다")
    COMMON._verify_current_files(repo, protocol["source_sha256_lf"])
    resource_ref = COMMON._exact(job["resource_proof"], {"path", "sha256"}, "resource")
    resource_path = _regular(Path(resource_ref["path"]), "resource proof path")
    resource_sha = COMMON._hash(resource_ref["sha256"], "resource SHA")
    parent = SOURCE.validate_protocol(
        _read(repo / protocol["parent_source_protocol"]["path"])
    )
    current = now or datetime.now(timezone.utc)  # noqa: UP017 - Python 3.10
    if current.tzinfo is None or current.utcoffset() is None:
        raise ValueError("현재 시각에 timezone이 필요하다")
    resource = COMMON._resource(
        resource_path,
        resource_sha,
        parent,
        protocol["parent_source_protocol"]["sha256_lf"],
        current.astimezone(timezone.utc),  # noqa: UP017 - Python 3.10
        require_fresh,
    )
    lab = COMMON._directory(Path(job["isaaclab"]), "isaaclab")
    COMMON._python_executable(Path(job["python"]))
    COMMON._verify_runtime_sources(lab, parent)
    output = Path(job["output_directory"])
    if not output.is_absolute() or output != job_path.parent / "output":
        raise ValueError("output_directory는 job 옆 output으로 고정된다")
    _check_ancestors(output.parent, "output directory")
    if require_fresh:
        if output.exists() or output.is_symlink():
            raise FileExistsError(output)
    else:
        output = COMMON._directory(output, "output directory")
    COMMON._check_deadline()
    return {
        "job_path": job_path,
        "output_directory": output,
        "job_sha256": _sha(job_path),
        "job": job,
        "protocol": protocol,
        "resource": resource,
        "repo": repo,
    }
