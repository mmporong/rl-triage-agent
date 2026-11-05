"""검증된 CPU SDK 경계 영수증 하나만 seed 실행 기록에서 분리한다."""
from __future__ import annotations

import hashlib
import json
import os
import stat
import threading
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import core_seed_guard_v1 as V1
import core_seed_guard_v2 as V2

SOURCE_FILES = (
    "evals/core_seed_guard_v3.py",
    "tests/test_core_seed_guard_v3.py",
)
SCHEMA = "core_seed_freshness_v3"
PROOF_SCHEMA = "registered_epoch_actual_request_cpu_preflight_v2"
KNOWN_HELPER_SHA256 = "128fbe1e9478f2da2a152c123e47a840f797f216c88146e2bb69665bd9e40d36"
FORBIDDEN = V1.EXCLUDED_RELATIVE.name.casefold()
EXPECTED_CHECKS = {
    "source_unchanged",
    "head_unchanged",
    "binding_unchanged",
    "preflight_processes_started_zero",
    "preflight_gpu_queries_zero",
    "owned_folder_exists",
    "launch_exists",
    "sdk_boundary_count_one",
    "fixture_snapshot_count_one",
    "entry_matches",
    "environment_option_matches",
    "environment_binding_valid",
    "manifest_seed_matches",
    "manifest_run_name_matches",
    "owned_receipt_records_boundary",
    "outer_error_is_expected_incomplete_training",
}
PROOF_FIELDS = {
    "schema", "status", "binding", "output", "original_request_id", "effective_run_name",
    "training_seed", "isaaclab_preflight", "git_head_before", "git_head_after",
    "source_sha256_before", "source_sha256_after", "outer_error", "boundary", "checks",
    "refs", "resource_counts",
}
REF_FIELDS = {
    "binding", "generated_environment_binding", "manifest", "launch", "execution",
    "outer_traceback", "sdk_boundary_stack",
}
POSITIVE_EVIDENCE_KEYS = {
    "actual_completed_training", "completed_training", "checkpoint", "checkpoint_ref",
    "fixed_evaluation", "fixed_result", "owned_pid", "child_pid", "wrapper_pid",
    "gpu_admission", "admission_ref", "recovery_outcome",
}
_LOCK = threading.RLock()


def _sha(path: Path, *, lf: bool = False) -> str:
    data = path.read_bytes()
    if lf:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def _forbidden_before_stat(value: object, label: str) -> Path:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError(f"{label} 경로 형식이 잘못됐다")
    lexical = Path(os.path.abspath(os.path.normpath(value)))
    if FORBIDDEN in {part.casefold() for part in lexical.parts}:
        raise ValueError(f"{label} 경로가 제외 provider 영역을 가리킨다")
    return lexical


def _repo_file(value: object, repo: Path, label: str) -> Path:
    lexical = _forbidden_before_stat(value, label)
    try:
        relative = lexical.relative_to(repo)
    except ValueError as error:
        raise ValueError(f"{label} 경로가 저장소 밖이다") from error
    current = repo
    for part in relative.parts:
        current = current / part
        try:
            info = os.lstat(current)
        except OSError as error:
            raise ValueError(f"{label} 경로를 읽을 수 없다") from error
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError(f"{label} 경로에 link 또는 reparse point가 있다")
    if not current.is_file():
        raise ValueError(f"{label} 경로가 파일이 아니다")
    return current


def _reference(value: object, repo: Path, label: str) -> Path:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} 참조 필드가 잘못됐다")
    path = _repo_file(value["path"], repo, label)
    expected = value["sha256"]
    if not isinstance(expected, str) or len(expected) != 64 or _sha(path) != expected:
        raise ValueError(f"{label} SHA가 원문과 다르다")
    return path


def _json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} JSON을 읽을 수 없다") from error
    if not isinstance(value, dict):
        raise TypeError(f"{label} JSON object가 필요하다")
    return value


def _reject_positive_evidence(value: object) -> None:
    if isinstance(value, Mapping):
        names = {str(key).casefold() for key in value}
        if names & POSITIVE_EVIDENCE_KEYS:
            raise ValueError("CPU proof에 실제 학습·평가·GPU 실행 양성 증거가 섞였다")
        for child in value.values():
            _reject_positive_evidence(child)
    elif isinstance(value, list):
        for child in value:
            _reject_positive_evidence(child)


def _verify_sources(proof: Mapping[str, Any], repo: Path) -> None:
    before, after = proof["source_sha256_before"], proof["source_sha256_after"]
    if not isinstance(before, dict) or before != after or not before:
        raise ValueError("CPU proof source 전후 결합이 다르다")
    for relative, receipt in before.items():
        if (not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts
                or not isinstance(receipt, dict)
                or set(receipt) != {"sha256", "sha256_lf"}):
            raise ValueError("CPU proof source receipt 형식이 잘못됐다")
        path = _repo_file(str(repo / relative), repo, f"source {relative}")
        if _sha(path) != receipt["sha256"] or _sha(path, lf=True) != receipt["sha256_lf"]:
            raise ValueError("CPU proof source 구현 bytes가 달라졌다")


def _verify_execution_receipts(proof: Mapping[str, Any], refs: Mapping[str, Path]) -> None:
    execution = _json(refs["execution"], "execution receipt")
    expected_execution = {
        "job_assigned_before_release": False,
        "natural_exit": False,
        "timed_out": False,
        "error_type": "SdkBoundaryReached",
        "exit_code": None,
        "cleanup_verified": False,
        "owned_process_identities": {},
    }
    if any(execution.get(key) != value for key, value in expected_execution.items()):
        raise ValueError("SDK 차단 execution receipt가 실제 실행 부재를 증명하지 못한다")
    launch = _json(refs["launch"], "owned launch receipt")
    boundary = proof["boundary"]
    command = boundary.get("command") if isinstance(boundary, dict) else None
    if (not isinstance(command, list) or len(command) != 4 or command[2] != "--_worker"
            or Path(str(command[3])).resolve() != refs["launch"]
            or not isinstance(launch.get("args"), list)):
        raise ValueError("SDK worker callback 차단 결합이 다르다")
    stack = refs["sdk_boundary_stack"].read_text(encoding="utf-8")
    if "guarded_popen" not in stack or "subprocess.Popen" not in stack:
        raise ValueError("SDK worker callback 차단 stack이 없다")


def _verify_nonexecution_proof(
    logical_id: str, value: object, repo: Path,
) -> dict[str, Any]:
    if not isinstance(logical_id, str) or not logical_id:
        raise ValueError("nonexecution proof logical id가 비었다")
    proof_path = _reference(value, repo, f"nonexecution proof {logical_id}")
    proof = _json(proof_path, f"nonexecution proof {logical_id}")
    if (set(proof) != PROOF_FIELDS or proof.get("schema") != PROOF_SCHEMA
            or proof.get("status") != "pass"):
        raise ValueError("CPU proof schema 또는 필드가 다르다")
    _reject_positive_evidence(proof)
    checks = proof.get("checks")
    if not isinstance(checks, dict) or set(checks) != EXPECTED_CHECKS \
            or any(value is not True for value in checks.values()):
        raise ValueError("CPU proof check 집합이 완결되지 않았다")
    resources = proof.get("resource_counts")
    if (not isinstance(resources, dict)
            or resources != {"sdk_launches": 0, "gpu_queries": 0, "api_calls": 0,
                             "fixture_gpu_snapshots": 1}):
        raise ValueError("CPU proof resource count가 비실행 범위가 아니다")
    preflight = proof.get("isaaclab_preflight")
    if (not isinstance(preflight, dict) or preflight.get("processes_started") != 0
            or preflight.get("gpu_queries") != 0):
        raise ValueError("CPU proof Isaac preflight가 비실행이 아니다")
    if (proof.get("git_head_before") != proof.get("git_head_after")
            or proof.get("outer_error") != {
                "type": "RuntimeError", "message": "학습 실행 또는 첫 3 update 계측이 완결되지 않았다",
            }
            or proof.get("boundary", {}).get("count") != 1):
        raise ValueError("CPU proof 경계 결합이 다르다")
    _verify_sources(proof, repo)
    refs_value = proof.get("refs")
    if not isinstance(refs_value, dict) or set(refs_value) != REF_FIELDS:
        raise ValueError("CPU proof refs 집합이 다르다")
    refs = {name: _reference(refs_value[name], repo, f"proof ref {name}") for name in REF_FIELDS}
    _verify_execution_receipts(proof, refs)
    output = _forbidden_before_stat(proof["output"], "CPU proof output")
    helper = _repo_file(str(output.parent / "actual_request_preflight_v13.py"), repo, "CPU helper")
    if _sha(helper) != KNOWN_HELPER_SHA256:
        raise ValueError("동결 CPU helper 구현 bytes가 다르다")
    return {
        "logical_id": logical_id,
        "proof": {"path": str(proof_path), "sha256": _sha(proof_path)},
        "excluded_record": "<repo>/" + proof_path.relative_to(repo).as_posix(),
        "helper_sha256": KNOWN_HELPER_SHA256,
        "training_seed": proof["training_seed"],
        "reason": "verified_cpu_sdk_boundary_nonexecution",
    }


def verify_nonexecution_proofs(
    refs: Mapping[str, object], repo: Path,
) -> list[dict[str, Any]]:
    """검증된 CPU 비실행 proof 목록을 정규화한다."""
    root = Path(repo).resolve(strict=True)
    if not isinstance(refs, Mapping) or not refs:
        raise ValueError("검증할 nonexecution proof가 필요하다")
    verified = [
        _verify_nonexecution_proof(logical_id, value, root)
        for logical_id, value in refs.items()
    ]
    paths = {row["proof"]["path"] for row in verified}
    if len(paths) != len(verified):
        raise ValueError("nonexecution proof 경로가 중복됐다")
    return verified


@contextmanager
def _exclude_verified_records(paths: set[Path]) -> Iterator[None]:
    original_assert = V1.ORIGINAL.assert_fresh_seeds

    def asserted(seeds: Iterable[int], isaaclab: Path, *, repo: Path):
        inherited_files = V1.ORIGINAL._files

        def files(root: Path):
            for directory, names in inherited_files(root):
                yield directory, [name for name in names if (directory / name).resolve() not in paths]

        V1.ORIGINAL._files = files
        try:
            return original_assert(seeds, isaaclab, repo=repo)
        finally:
            V1.ORIGINAL._files = inherited_files

    with _LOCK, V2._LOCK, V1._LOCK:
        old = V1.ORIGINAL.assert_fresh_seeds
        V1.ORIGINAL.assert_fresh_seeds = asserted
        try:
            yield
        finally:
            V1.ORIGINAL.assert_fresh_seeds = old


def assert_fresh_seeds(
    seeds: Iterable[int],
    isaaclab: Path,
    *,
    repo: Path = V1.ORIGINAL.ROOT,
    nonexecution_proofs: Mapping[str, object],
) -> dict[str, Any]:
    """V2 전체 검사를 유지하고 검증된 CPU proof 파일만 후보에서 제외한다."""
    repo = Path(repo).resolve(strict=True)
    verified = verify_nonexecution_proofs(nonexecution_proofs, repo)
    paths = {Path(row["proof"]["path"]).resolve(strict=True) for row in verified}
    if len(paths) != len(verified):
        raise ValueError("nonexecution proof 경로가 중복됐다")
    with _exclude_verified_records(paths):
        result = V2.assert_fresh_seeds(seeds, isaaclab, repo=repo)
    return {
        **result,
        "schema": SCHEMA,
        "verified_nonexecution_proofs": verified,
        "isolated_skipped_record_count": len(verified),
    }
