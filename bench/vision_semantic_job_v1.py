"""Semantic diagnostic 전용 protocol과 job의 fail-closed 결합 검증기."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _folder in (ROOT / "bench", ROOT / "evals", ROOT / "src"):
    if str(_folder) not in sys.path:
        sys.path.insert(0, str(_folder))

import record_vision_evidence_v1 as RECORD
import run_vision_atomic_compatibility_v1 as ATOMIC
import run_vision_brev_batch_v1 as SOURCE

MODE = "semantic_diagnostic_only"
PROTOCOL_SCHEMA = "vision_semantic_sdk_protocol_v1"
JOB_SCHEMA = "vision_semantic_sdk_job_v1"
RESULT_SCHEMA = "vision_semantic_sdk_result_v1"
FAILURE_SCHEMA = "vision_semantic_sdk_failure_v1"
HISTORY_PROTOCOL_PATH = "bench/protocols/vision_atomic_compatibility_20261025_v7.json"
HISTORY_PROTOCOL_SHA256_LF = "375eb7992de62183885ae674c72d70c1d4567be9c5a7546a3c1b8ded33756b8a"
METADATA_LIFECYCLE_SCHEMA = "vision_metadata_lifecycle_v1"

HEX40 = re.compile(r"[0-9a-f]{40}\Z")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
SLUG = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
PROTOCOL_KEYS = {
    "schema", "protocol_id", "mode", "registered_before_measurement", "source_commit",
    "source_sha256_lf", "parent_source_protocol", "profile_sha256", "limits", "resources",
    "preregistration",
}
JOB_KEYS = {
    "schema", "mode", "job_id", "source", "protocol", "resource_proof", "isaaclab",
    "python", "output_directory",
}
LIMITS = {
    "views": 1,
    "events": 2,
    "callbacks": 16,
    "max_wall_s": 600,
    "training_actions": 0,
    "model_api_requests": 0,
    "full_recordings": 0,
    "no_capture_approved": True,
}
FIXED_FLAGS = {
    "diagnostic_only": True,
    "no_cause_claim": True,
    "no_repair_success_claim": True,
    "no_capture": True,
}
RESOURCE_KEYS = {
    "owned_instance_id", "owned_instance_name", "compute_quote_usd_per_h", "max_owned_instances",
    "prior_compute_estimate_usd", "fresh_lifecycle_budget_usd", "total_program_budget_usd",
    "prior_stop_receipts",
}
REQUIRED_SOURCE_FILES = {
    HISTORY_PROTOCOL_PATH,
    "bench/vision_semantic_sdk_v1.py",
    "bench/vision_semantic_job_v1.py",
    "bench/run_vision_semantic_diagnostic_v1.py",
    "bench/vision_semantic_diagnostic_v1.py",
    "tests/test_vision_semantic_sdk_v1.py",
    "tests/test_vision_semantic_job_v1.py",
    "tests/test_run_vision_semantic_diagnostic_v1.py",
    "tests/test_vision_semantic_diagnostic_v1.py",
}
_VALIDATION_DEADLINE: ContextVar[float | None] = ContextVar(
    "vision_semantic_validation_deadline", default=None,
)
MAX_BATCH_BLOB_BYTES = 64 * 1024 * 1024
MAX_SINGLE_BLOB_BYTES = 16 * 1024 * 1024
MAX_BATCH_ENTRIES = 4096
MAX_GIT_STDERR_READ_BYTES = 4096


@contextmanager
def validation_deadline(deadline_monotonic: float):
    if not isinstance(deadline_monotonic, (int, float)) or not math.isfinite(deadline_monotonic):
        raise ValueError("validation deadline은 유한한 monotonic 시각이어야 한다")
    token = _VALIDATION_DEADLINE.set(float(deadline_monotonic))
    try:
        yield
    finally:
        _VALIDATION_DEADLINE.reset(token)


def _check_deadline() -> None:
    deadline = _VALIDATION_DEADLINE.get()
    if deadline is not None and deadline - time.monotonic() <= 0:
        raise TimeoutError("semantic validation deadline을 넘었다")


def _subprocess_timeout() -> float:
    deadline = _VALIDATION_DEADLINE.get()
    if deadline is None:
        return 30.0
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("semantic validation deadline을 넘었다")
    return min(30.0, remaining)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _sha(path: Path, *, lf: bool = False) -> str:
    data = path.read_bytes()
    if lf:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def _pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in rows:
        if key in result:
            raise ValueError(f"중복 JSON key: {key}")
        result[key] = value
    return result


def _read(path: Path) -> dict[str, Any]:
    def reject(value: str) -> None:
        raise ValueError(f"비유한 JSON 값: {value}")

    value = json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=_pairs, parse_constant=reject
    )
    if not isinstance(value, dict):
        raise TypeError("JSON object가 필요하다")
    return value


def _is_reparse(info: os.stat_result) -> bool:
    flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(getattr(info, "st_file_attributes", 0) & flag)


def _check_ancestors(path: Path, anchor: str) -> None:
    absolute = path.absolute()
    existing: list[Path] = []
    cursor = absolute
    while True:
        if cursor.exists() or cursor.is_symlink():
            existing.append(cursor)
        if cursor == cursor.parent:
            break
        cursor = cursor.parent
    for item in existing:
        info = item.lstat()
        if stat.S_ISLNK(info.st_mode) or _is_reparse(info):
            raise ValueError(f"{anchor} 조상에 symlink/reparse point가 있다")


def _regular(path: Path, anchor: str) -> Path:
    path = Path(path)
    if not path.is_absolute():
        raise ValueError(f"{anchor}는 절대 경로여야 한다")
    _check_ancestors(path, anchor)
    try:
        info = path.lstat()
    except FileNotFoundError as error:
        raise ValueError(f"{anchor} 파일이 없다") from error
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or _is_reparse(info):
        raise ValueError(f"{anchor}는 단일 regular file이어야 한다")
    return path.resolve(strict=True)


def _directory(path: Path, anchor: str) -> Path:
    path = Path(path)
    if not path.is_absolute():
        raise ValueError(f"{anchor}는 절대 경로여야 한다")
    _check_ancestors(path, anchor)
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or _is_reparse(info):
        raise ValueError(f"{anchor}는 실제 directory여야 한다")
    return path.resolve(strict=True)


def _python_executable(path: Path) -> Path:
    """venv 문맥의 lexical Python 경로를 보존하며 실제 실행 파일만 검증한다."""
    path = Path(path)
    if not path.is_absolute():
        raise ValueError("python은 절대 경로여야 한다")
    lexical = path.absolute()
    _check_ancestors(lexical.parent, "python")
    try:
        target = lexical.resolve(strict=True)
        info = target.lstat()
    except FileNotFoundError as error:
        raise ValueError("python 실행 파일이 없다") from error
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or _is_reparse(info):
        raise ValueError("python 대상은 단일 regular file이어야 한다")
    current = Path(sys.executable).absolute()
    if os.path.normcase(str(lexical)) != os.path.normcase(str(current)):
        raise ValueError("job python이 현재 실행 Python과 다르다")
    return lexical


def write_new(path: Path, value: Any) -> None:
    path = Path(path)
    if not path.is_absolute():
        path = path.absolute()
    _check_ancestors(path.parent, "output")
    path.parent.mkdir(parents=True, exist_ok=True)
    _check_ancestors(path.parent, "output")
    raw = _canonical(value)
    with path.open("xb") as stream:
        stream.write(raw + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    _regular(path, "output")


def _exact(value: Any, keys: set[str], anchor: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{anchor} 필드가 계약과 다르다")
    return value


def _hash(value: Any, anchor: str) -> str:
    if not isinstance(value, str) or HEX64.fullmatch(value) is None:
        raise ValueError(f"{anchor} SHA-256이 잘못됐다")
    return value


def _relative(value: Any, anchor: str) -> str:
    if (not isinstance(value, str) or not value or "\\" in value
            or any(marker in value for marker in ("\0", "\n", "\r"))):
        raise ValueError(f"{anchor}는 POSIX 상대 경로여야 한다")
    path = Path(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{anchor} 상대 경로가 잘못됐다")
    return value


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    timeout = _subprocess_timeout()
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *args], capture_output=True, check=False, text=True,
            encoding="utf-8", errors="replace", timeout=timeout,
        )
    except subprocess.TimeoutExpired as error:
        raise TimeoutError("semantic validation Git 조회가 시간 상한을 넘었다") from error
    _check_deadline()
    return result


def _parse_batch_blobs(
    raw: bytes, rows: list[tuple[str, str]], metadata: list[tuple[bytes, int]] | None = None,
) -> None:
    offset = 0
    total = 0
    for index, (relative, expected) in enumerate(rows):
        end = raw.find(b"\n", offset)
        if end < 0 or end - offset > 256:
            raise ValueError(f"git batch header가 잘렸거나 과도하다: {relative}")
        header = raw[offset:end]
        offset = end + 1
        parts = header.split(b" ")
        if len(parts) == 2 and parts[1] == b"missing":
            raise ValueError(f"source commit에 파일이 없다: {relative}")
        if (len(parts) != 3 or re.fullmatch(rb"[0-9a-f]{40}", parts[0]) is None
                or parts[1] != b"blob" or re.fullmatch(rb"[0-9]+", parts[2]) is None):
            raise ValueError(f"git batch object header가 blob 계약과 다르다: {relative}")
        size = int(parts[2])
        if metadata is not None and (parts[0], size) != metadata[index]:
            raise ValueError(f"git batch blob이 사전 확인 metadata와 다르다: {relative}")
        if size > MAX_SINGLE_BLOB_BYTES or total + size > MAX_BATCH_BLOB_BYTES:
            raise ValueError(f"git batch blob 출력이 상한을 넘는다: {relative}")
        body_end = offset + size
        if body_end > len(raw) or raw[body_end:body_end + 1] != b"\n":
            raise ValueError(f"git batch blob이 짧거나 separator가 없다: {relative}")
        body = raw[offset:body_end]
        actual = hashlib.sha256(body.replace(b"\r\n", b"\n")).hexdigest()
        if actual != expected:
            raise ValueError(f"source commit blob이 현재 결합과 다르다: {relative}")
        total += size
        offset = body_end + 1
    if offset != len(raw):
        raise ValueError("git batch 출력에 후행 데이터가 있다")


def _parse_batch_check(raw: bytes, rows: list[tuple[str, str]]) -> list[tuple[bytes, int]]:
    if not raw.endswith(b"\n"):
        raise ValueError("git batch-check metadata가 잘렸다")
    headers = raw[:-1].split(b"\n")
    if len(headers) != len(rows):
        raise ValueError("git batch-check metadata 행 수 또는 후행 데이터가 다르다")
    metadata: list[tuple[bytes, int]] = []
    total = 0
    for header, (relative, _) in zip(headers, rows, strict=True):
        if len(header) > 256:
            raise ValueError(f"git batch-check header가 과도하다: {relative}")
        parts = header.split(b" ")
        if len(parts) == 2 and parts[1] == b"missing":
            raise ValueError(f"source commit에 파일이 없다: {relative}")
        if (len(parts) != 3 or re.fullmatch(rb"[0-9a-f]{40}", parts[0]) is None
                or parts[1] != b"blob" or re.fullmatch(rb"[0-9]+", parts[2]) is None):
            raise ValueError(f"git batch-check object header가 blob 계약과 다르다: {relative}")
        size = int(parts[2])
        if size > MAX_SINGLE_BLOB_BYTES or total + size > MAX_BATCH_BLOB_BYTES:
            raise ValueError(f"git batch-check blob 크기가 상한을 넘는다: {relative}")
        total += size
        metadata.append((parts[0], size))
    return metadata


def _source_rows(sources: Mapping[str, str]) -> list[tuple[str, str]]:
    rows = list(sources.items())
    if not rows or len(rows) > MAX_BATCH_ENTRIES:
        raise ValueError("source blob batch 항목 수가 상한 밖이다")
    paths = [_relative(relative, "source path") for relative, _ in rows]
    if len(paths) != len(set(paths)):
        raise ValueError("source blob batch 경로가 중복됐다")
    return [
        (relative, _hash(expected, f"source[{relative}]"))
        for relative, (_, expected) in zip(paths, rows, strict=True)
    ]


def _verify_current_files(repo: Path, sources: Mapping[str, str]) -> list[tuple[str, str]]:
    rows = _source_rows(sources)
    total = 0
    for relative, expected in rows:
        path = _regular(repo / relative, f"source file {relative}")
        path.relative_to(repo)
        size = path.stat().st_size
        if size > MAX_SINGLE_BLOB_BYTES:
            raise ValueError(f"현재 source 파일이 batch 단일 상한을 넘는다: {relative}")
        total += size
        if total > MAX_BATCH_BLOB_BYTES:
            raise ValueError("현재 source 파일 합계가 batch 상한을 넘는다")
        if _sha(path, lf=True) != expected:
            raise ValueError(f"source LF SHA가 다르다: {relative}")
    return rows


def _run_cat_file(repo: Path, mode: str, request: bytes, output_cap: int) -> bytes:
    timeout = _subprocess_timeout()
    try:
        with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
            result = subprocess.run(
                ["git", "-C", str(repo), "cat-file", mode], input=request,
                stdout=stdout, stderr=stderr, check=False, timeout=timeout,
            )
            _check_deadline()
            if result.returncode != 0:
                stderr.seek(0)
                detail = stderr.read(MAX_GIT_STDERR_READ_BYTES).decode("utf-8", errors="replace")
                raise ValueError(f"git {mode} 조회 실패: {detail.strip()}")
            if stdout.tell() > output_cap:
                raise ValueError(f"git {mode} 전체 출력이 상한을 넘는다")
            stdout.seek(0)
            return stdout.read()
    except subprocess.TimeoutExpired as error:
        raise TimeoutError(f"semantic validation Git {mode} 조회가 시간 상한을 넘었다") from error


def _verify_commit_blobs(repo: Path, commit: str, rows: list[tuple[str, str]]) -> None:
    request = b"".join(f"{commit}:{relative}\n".encode() for relative, _ in rows)
    if len(request) > 1024 * 1024:
        raise ValueError("git batch 요청이 상한을 넘는다")
    metadata_raw = _run_cat_file(repo, "--batch-check", request, len(rows) * 257)
    metadata = _parse_batch_check(metadata_raw, rows)
    # immutable commit의 blob 합계를 batch-check에서 제한한 뒤에만 contents를 요청한다.
    raw = _run_cat_file(
        repo, "--batch", request, MAX_BATCH_BLOB_BYTES + len(rows) * 320,
    )
    _parse_batch_blobs(raw, rows, metadata)


def _verify_source_blobs(repo: Path, commit: str, sources: Mapping[str, str]) -> None:
    _verify_commit_blobs(repo, commit, _verify_current_files(repo, sources))


def _verify_blob(repo: Path, commit: str, relative: str, expected: str) -> None:
    path = _regular(repo / relative, f"source file {relative}")
    path.relative_to(repo)
    if _sha(path, lf=True) != expected:
        raise ValueError(f"source LF SHA가 다르다: {relative}")
    blob = _git(repo, "show", f"{commit}:{relative}")
    if blob.returncode != 0:
        raise ValueError(f"source commit에 파일이 없다: {relative}")
    actual = hashlib.sha256(blob.stdout.encode("utf-8").replace(b"\r\n", b"\n")).hexdigest()
    if actual != expected:
        raise ValueError(f"source commit blob이 현재 결합과 다르다: {relative}")


def _verify_head_protocol(repo: Path, path: Path) -> None:
    relative = path.relative_to(repo).as_posix()
    current = _sha(path, lf=True)
    blob = _git(repo, "show", f"HEAD:{relative}")
    if blob.returncode != 0:
        raise ValueError("semantic protocol이 현재 HEAD에 커밋되지 않았다")
    committed = hashlib.sha256(
        blob.stdout.encode("utf-8").replace(b"\r\n", b"\n")
    ).hexdigest()
    if current != committed:
        raise ValueError("semantic protocol 현재 파일이 HEAD committed LF blob과 다르다")


def _source_map(value: Any) -> dict[str, str]:
    if not isinstance(value, dict) or not value:
        raise ValueError("source_sha256_lf가 비었다")
    result: dict[str, str] = {}
    for relative, expected in value.items():
        result[_relative(relative, "source path")] = _hash(expected, f"source[{relative}]")
    return result


def _validate_preregistration(value: Any) -> dict[str, Any]:
    prereg = _exact(value, {"prediction", "observation_phrases", "fixed_flags"}, "preregistration")
    if not isinstance(prereg["prediction"], str) or not prereg["prediction"].strip():
        raise ValueError("preregistration prediction이 비었다")
    phrases = prereg["observation_phrases"]
    if (not isinstance(phrases, dict) or not phrases
            or any(not isinstance(key, str) or not key.strip() or not isinstance(text, str)
                   or not text.strip() for key, text in phrases.items())):
        raise ValueError("preregistration observation 문구가 비었다")
    if prereg["fixed_flags"] != FIXED_FLAGS:
        raise ValueError("진단 전용 주장 금지 flags가 다르다")
    return prereg


def _validate_prior_stops(
    resources: Mapping[str, Any], repo: Path, *, source: Mapping[str, Any], source_commit: str,
) -> None:
    references = resources["prior_stop_receipts"]
    if not isinstance(references, list) or not references:
        raise ValueError("prior STOPPED receipt 목록이 비었다")
    rows: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for item in references:
        reference = _exact(item, {"path", "sha256"}, "prior stop receipt reference")
        relative = _relative(reference["path"], "prior stop receipt.path")
        expected = _hash(reference["sha256"], "prior stop receipt.sha256")
        path = _regular(repo / relative, "prior stop receipt")
        if _sha(path) != expected:
            raise ValueError("prior STOPPED receipt SHA가 다르다")
        rows.append((reference, _read(path)))

    if any(raw.get("schema") == METADATA_LIFECYCLE_SCHEMA for _, raw in rows):
        _verify_blob(
            repo, source_commit, HISTORY_PROTOCOL_PATH, HISTORY_PROTOCOL_SHA256_LF,
        )
        history = _read(repo / HISTORY_PROTOCOL_PATH)
        history_resources = history.get("resources")
        if not isinstance(history_resources, Mapping):
            raise ValueError("고정 lifecycle history resources가 다르다")
        contract = history_resources.get("metadata_lifecycle_contract")
        if not isinstance(contract, Mapping):
            raise ValueError("고정 metadata lifecycle 계약이 없다")
        mixed_resources = {
            **resources,
            "prior_lifecycle_compute_usd": resources["prior_compute_estimate_usd"],
            "metadata_lifecycle_contract": contract,
        }
        ATOMIC._validate_mixed_prior_stops(mixed_resources, source, repo)
        return

    total = Decimal(0)
    seen_hashes: set[str] = set()
    seen_runs: set[str] = set()
    for reference, stop in rows:
        expected = reference["sha256"]
        if expected in seen_hashes:
            raise ValueError("prior STOPPED receipt가 중복됐다")
        owner = stop.get("owned_instance")
        if (stop.get("status") != "STOPPED" or not isinstance(owner, Mapping)
                or owner.get("id") != resources["owned_instance_id"]
                or owner.get("name") != resources["owned_instance_name"]):
            raise ValueError("prior STOPPED receipt의 owner/status가 다르다")
        identity = stop.get("execution_identity")
        run_id = identity.get("run_id") if isinstance(identity, Mapping) else stop.get("run_id")
        if not isinstance(run_id, str) or not run_id.strip() or run_id in seen_runs:
            raise ValueError("prior STOPPED lifecycle run이 비었거나 중복됐다")
        cost = stop.get("cost_estimate")
        amount = cost.get("estimated_compute_cost_usd") if isinstance(cost, Mapping) else stop.get(
            "estimated_compute_cost_usd"
        )
        try:
            parsed = Decimal(str(amount))
        except Exception as error:
            raise ValueError("prior STOPPED 비용을 읽을 수 없다") from error
        if not parsed.is_finite() or parsed < 0:
            raise ValueError("prior STOPPED 비용이 유한한 비음수가 아니다")
        total += parsed
        seen_hashes.add(expected)
        seen_runs.add(run_id)
    if total != Decimal(str(resources["prior_compute_estimate_usd"])):
        raise ValueError("prior STOPPED 비용 합계가 등록값과 다르다")


def validate_protocol(path: Path, *, repo: Path | None = None) -> dict[str, Any]:
    _check_deadline()
    repo = _directory(Path(repo or ROOT), "repo")
    protocol_path = _regular(Path(path).absolute(), "protocol")
    protocol_path.relative_to(repo)
    _verify_head_protocol(repo, protocol_path)
    protocol = _exact(_read(protocol_path), PROTOCOL_KEYS, "protocol")
    if (protocol["schema"] != PROTOCOL_SCHEMA or protocol["mode"] != MODE
            or protocol["registered_before_measurement"] is not True):
        raise ValueError("semantic protocol schema/mode/등록 상태가 다르다")
    if not isinstance(protocol["protocol_id"], str) or SLUG.fullmatch(protocol["protocol_id"]) is None:
        raise ValueError("protocol_id가 잘못됐다")
    commit = protocol["source_commit"]
    if not isinstance(commit, str) or HEX40.fullmatch(commit) is None:
        raise ValueError("source_commit이 잘못됐다")
    if _git(repo, "merge-base", "--is-ancestor", commit, "HEAD").returncode != 0:
        raise ValueError("source_commit이 현재 HEAD의 조상이 아니다")

    parent_ref = _exact(protocol["parent_source_protocol"], {"path", "sha256_lf"}, "parent protocol")
    parent_relative = _relative(parent_ref["path"], "parent protocol.path")
    parent_sha = _hash(parent_ref["sha256_lf"], "parent protocol.sha256_lf")
    parent_path = _regular(repo / parent_relative, "parent protocol")
    if _sha(parent_path, lf=True) != parent_sha:
        raise ValueError("parent protocol LF SHA가 다르다")
    _verify_blob(repo, commit, parent_relative, parent_sha)
    parent = SOURCE.validate_protocol(SOURCE.read(parent_path))

    source_map = _source_map(protocol["source_sha256_lf"])
    parent_sources = parent["source_sha256_lf"]
    if any(source_map.get(relative) != expected for relative, expected in parent_sources.items()):
        raise ValueError("parent source closure가 보존되지 않았다")
    if not REQUIRED_SOURCE_FILES <= set(source_map):
        raise ValueError("semantic diagnostic source closure가 빠졌다")
    _check_deadline()
    _verify_source_blobs(repo, commit, source_map)
    _check_deadline()

    if protocol["profile_sha256"] != RECORD.digest(RECORD.fixed_profile()):
        raise ValueError("고정 vision profile SHA가 다르다")
    if protocol["limits"] != LIMITS:
        raise ValueError("semantic diagnostic 실행 상한이 다르다")

    resources = _exact(protocol["resources"], RESOURCE_KEYS, "resources")
    if (resources["owned_instance_id"] != "3655a80oo"
            or resources["owned_instance_name"] != "rl-triage-e2e-20261024-c9f4"
            or Decimal(str(resources["compute_quote_usd_per_h"])) != Decimal("1.74")
            or resources["max_owned_instances"] != 1
            or Decimal(str(resources["prior_compute_estimate_usd"])) != Decimal("8.500674")
            or Decimal(str(resources["fresh_lifecycle_budget_usd"])) != Decimal("11.499326")
            or Decimal(str(resources["total_program_budget_usd"])) != Decimal("20.000000")
            or Decimal(str(resources["prior_compute_estimate_usd"]))
            + Decimal(str(resources["fresh_lifecycle_budget_usd"])) != Decimal("20.000000")):
        raise ValueError("semantic diagnostic resource budget이 다르다")
    parent_resources = parent["resources"]
    if (parent_resources["owned_instance_id"] != resources["owned_instance_id"]
            or parent_resources["max_owned_instances"] != 1
            or Decimal(str(parent_resources["compute_quote_usd_per_h"])) != Decimal("1.74")
            or Decimal(str(parent_resources["whole_batch_budget_usd"])) != Decimal("11.499326")):
        raise ValueError("parent source resource 계약이 다르다")
    _validate_prior_stops(resources, repo, source=parent, source_commit=commit)
    protocol["preregistration"] = _validate_preregistration(protocol["preregistration"])
    protocol["source_sha256_lf"] = source_map
    _check_deadline()
    return protocol


def _verify_runtime_sources(isaaclab: Path, parent: Mapping[str, Any]) -> None:
    _check_deadline()
    for relative, expected in parent["runtime_source_sha256_lf"].items():
        path = _regular(isaaclab / _relative(relative, "runtime source"), f"runtime source {relative}")
        path.relative_to(isaaclab)
        if _sha(path, lf=True) != expected:
            raise ValueError(f"Isaac Lab runtime source SHA가 다르다: {relative}")
    _check_deadline()


def _resource(
    resource_path: Path, resource_sha: str, parent: Mapping[str, Any], parent_sha: str,
    current: datetime, require_fresh: bool,
) -> dict[str, Any]:
    if _sha(resource_path) != resource_sha:
        raise ValueError("resource proof SHA가 다르다")
    batch = SOURCE.SourceBatch.__new__(SOURCE.SourceBatch)
    batch.protocol = dict(parent)
    batch.protocol_sha = parent_sha
    batch.now = lambda: current
    resource = SOURCE.SourceBatch._verify_resource(batch, SOURCE.read(resource_path))
    issued = SOURCE._stamp(resource["issued_at_utc"])
    age = (current - issued).total_seconds()
    if require_fresh and not 0.0 <= age <= 30.0:
        raise ValueError("resource proof issued 시각이 30초 freshness를 벗어났다")
    return resource


def validate_job(
    job_path: Path, *, now: datetime | None = None, require_fresh: bool = True,
) -> dict[str, Any]:
    _check_deadline()
    job_path = _regular(Path(job_path).absolute(), "job")
    job = _exact(_read(job_path), JOB_KEYS, "job")
    if job["schema"] != JOB_SCHEMA or job["mode"] != MODE:
        raise ValueError("job schema/mode가 등록 범위 밖이다")
    if not isinstance(job["job_id"], str) or SLUG.fullmatch(job["job_id"]) is None:
        raise ValueError("job_id가 잘못됐다")

    source = _exact(job["source"], {"repo_root", "commit", "files_lf"}, "source")
    repo = _directory(Path(source["repo_root"]), "source.repo_root")
    commit = source["commit"]
    if not isinstance(commit, str) or HEX40.fullmatch(commit) is None:
        raise ValueError("job source commit이 잘못됐다")
    source_map = _source_map(source["files_lf"])

    protocol_ref = _exact(job["protocol"], {"path", "sha256_lf"}, "protocol reference")
    protocol_path = _regular(Path(protocol_ref["path"]), "protocol reference.path")
    protocol_sha = _hash(protocol_ref["sha256_lf"], "protocol reference.sha256_lf")
    if _sha(protocol_path, lf=True) != protocol_sha:
        raise ValueError("protocol reference LF SHA가 다르다")
    protocol = validate_protocol(protocol_path, repo=repo)
    if commit != protocol["source_commit"] or source_map != protocol["source_sha256_lf"]:
        raise ValueError("job source가 semantic protocol과 다르다")
    _check_deadline()
    _verify_current_files(repo, source_map)
    _check_deadline()

    resource_ref = _exact(job["resource_proof"], {"path", "sha256"}, "resource proof")
    resource_path = _regular(Path(resource_ref["path"]), "resource proof.path")
    resource_sha = _hash(resource_ref["sha256"], "resource proof.sha256")
    parent_path = repo / protocol["parent_source_protocol"]["path"]
    parent = SOURCE.validate_protocol(SOURCE.read(parent_path))
    parent_sha = protocol["parent_source_protocol"]["sha256_lf"]
    current = now or datetime.now(timezone.utc)  # noqa: UP017 - Python 3.10
    if current.tzinfo is None or current.utcoffset() is None:
        raise ValueError("현재 시각에 timezone이 필요하다")
    current = current.astimezone(timezone.utc)  # noqa: UP017 - Python 3.10
    resource = _resource(resource_path, resource_sha, parent, parent_sha, current, require_fresh)

    isaaclab = _directory(Path(job["isaaclab"]), "isaaclab")
    _python_executable(Path(job["python"]))
    _verify_runtime_sources(isaaclab, parent)
    output = Path(job["output_directory"])
    if not output.is_absolute() or output != job_path.parent / "output":
        raise ValueError("output_directory는 job 옆 output으로 고정된다")
    _check_ancestors(output.parent, "output_directory")
    if require_fresh:
        if output.exists() or output.is_symlink():
            raise FileExistsError(output)
    else:
        output = _directory(output, "output_directory")
    _check_deadline()
    return {
        "job_path": job_path,
        "output_directory": output,
        "job_sha256": _sha(job_path),
        "job": job,
        "protocol": protocol,
        "resource": resource,
        "repo": repo,
    }
