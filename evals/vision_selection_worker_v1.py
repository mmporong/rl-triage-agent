"""동결된 24개 영상 선택 비교를 재개하는 WSL 전용 worker."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import re
import stat
import subprocess
import sys
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "evals") not in sys.path:
    sys.path.insert(0, str(ROOT / "evals"))

import grounded_vision_observer_v1 as OBSERVER
import model_decision_journal_v1 as JOURNAL
import vision_selection_call_v1 as CALL
import vision_selection_comparison_v1 as COMPARISON

PROTOCOL_SCHEMA = "vision_selection_execution_protocol_v1"
VERIFIED_INPUTS_SCHEMA = "vision_selection_verified_inputs_v1"
RECEIPT_SCHEMA = "vision_selection_worker_invocation_v1"
EXIT_CODES = {"choice_ready": 0, "terminal_hold": 2, "paused_infra": 3, "blocked_pending": 4}
REQUIRED_SOURCES = frozenset({
    "bench/record_vision_evidence_v1.py",
    "bench/run_vision_brev_batch_v1.py",
    "bench/vision_play_worker_v1.py",
    "bench/vision_render_attestation_v1.py",
    "bench/vision_render_context_v1.py",
    "evals/grounded_vision_observer_v1.py",
    "evals/model_decision_journal_v1.py",
    "evals/vision_selection_call_v1.py",
    "evals/vision_selection_comparison_v1.py",
    "evals/vision_selection_contract_v1.py",
    "evals/vision_selection_loop_v1.py",
    "evals/vision_selection_sources_v1.py",
    "evals/vision_selection_worker_v1.py",
    "evals/vision_remote_gpu_v1.py",
})
_HEX40 = re.compile(r"[0-9a-f]{40}")
_HEX64 = re.compile(r"[0-9a-f]{64}")
_IDENTIFIER = re.compile(r"[a-z0-9][a-z0-9._-]{0,79}")


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode("utf-8")


def _read(path: Path) -> dict[str, Any]:
    target = Path(path)
    if not target.is_file() or target.is_symlink() or any(parent.is_symlink() for parent in target.parents):
        raise ValueError(f"{target.name}은 symlink가 아닌 regular file이어야 한다")
    metadata = target.stat(follow_symlinks=False)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise ValueError(f"{target.name}은 단일 regular file이어야 한다")
    pairs = json.loads(
        target.read_text(encoding="utf-8"), object_pairs_hook=lambda rows: rows,
        parse_constant=lambda raw: (_ for _ in ()).throw(ValueError(f"JSON constant 거부: {raw}")),
    )

    def restore(value: Any) -> Any:
        if isinstance(value, list) and all(isinstance(row, tuple) and len(row) == 2 for row in value):
            result: dict[str, Any] = {}
            for key, item in value:
                if not isinstance(key, str) or key in result:
                    raise ValueError("JSON object의 키가 중복됐거나 문자열이 아니다")
                result[key] = restore(item)
            return result
        if isinstance(value, list):
            return [restore(item) for item in value]
        return value

    restored = restore(pairs)
    if not isinstance(restored, dict):
        raise TypeError(f"{target.name}은 JSON object여야 한다")
    return restored


def _sha_lf(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _git(root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=check, capture_output=True,
        text=True, encoding="utf-8", errors="strict",
    )


def _relative_file(root: Path, raw: object, label: str) -> tuple[str, Path]:
    if not isinstance(raw, str) or not raw or "\\" in raw or raw.startswith("/") or ".." in raw.split("/"):
        raise ValueError(f"{label} 경로가 canonical repo-relative 형식이 아니다")
    path = (root / raw).resolve(strict=True)
    if path.relative_to(root).as_posix() != raw or not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} 경로가 저장소 regular file이 아니다")
    return raw, path


def _hash_map(value: object, label: str) -> dict[str, str]:
    if (not isinstance(value, dict) or not value
            or any(not isinstance(name, str) or not name or "\\" in name or name.startswith("/")
                   or ".." in name.split("/") or _HEX64.fullmatch(str(digest)) is None
                   for name, digest in value.items())):
        raise ValueError(f"{label} SHA map이 잘못됐다")
    return deepcopy(value)


def _protocol_reference(root: Path, value: object, label: str) -> tuple[str, Path, dict[str, Any]]:
    if not isinstance(value, dict) or set(value) != {"path", "sha256_lf"}:
        raise ValueError(f"{label} reference 형식이 다르다")
    name, path = _relative_file(root, value["path"], label)
    if _HEX64.fullmatch(str(value["sha256_lf"])) is None or _sha_lf(path) != value["sha256_lf"]:
        raise ValueError(f"{label} LF SHA가 다르다")
    return name, path, _read(path)


def _parent_hashes(value: dict[str, Any], *, method: bool) -> dict[str, str]:
    if method:
        if value.get("schema") != "vision_selection_comparison_method_v1":
            raise ValueError("method protocol schema가 다르다")
        tables = (
            _hash_map(value.get("source_sha256_lf"), "method source_sha256_lf"),
            _hash_map(value.get("evidence_sha256_lf"), "method evidence_sha256_lf"),
        )
    else:
        if value.get("schema") != "vision_brev_source_protocol_v1":
            raise ValueError("source protocol schema가 다르다")
        recording = value.get("recording")
        if not isinstance(recording, dict) or set(recording) != {"source_files_lf"}:
            raise ValueError("source protocol recording 결합이 다르다")
        # 두 map은 각각 Isaac Lab checkout과 Python site-packages namespace다.
        # repo-relative source map과 합치지 않고 형식만 확인한 뒤, 실제 snapshot
        # equality는 vision_selection_sources_v1이 검사한다.
        _hash_map(value.get("runtime_source_sha256_lf"), "source runtime_source_sha256_lf")
        _hash_map(value.get("python_package_source_sha256_lf"), "source python_package_source_sha256_lf")
        tables = (
            _hash_map(value.get("source_sha256_lf"), "source source_sha256_lf"),
            _hash_map(recording.get("source_files_lf"), "source recording.source_files_lf"),
        )
    merged: dict[str, str] = {}
    for table in tables:
        for name, digest in table.items():
            if name in merged and merged[name] != digest:
                raise ValueError("부모 protocol 내부 경로 SHA가 충돌한다")
            merged[name] = digest
    return merged


def verify_protocol(protocol_path: Path, expected_sha256_lf: str, *, root: Path = ROOT) -> dict[str, Any]:
    """실행 protocol과 두 부모 protocol의 결합을 API 실행 전에 검사한다."""
    repository = Path(root).resolve(strict=True)
    protocol_file = Path(protocol_path).resolve(strict=True)
    if _HEX64.fullmatch(expected_sha256_lf) is None or _sha_lf(protocol_file) != expected_sha256_lf:
        raise ValueError("실행 protocol LF SHA가 다르다")
    protocol_relative = protocol_file.relative_to(repository).as_posix()
    protocol = _read(protocol_file)
    required = {
        "schema", "protocol_id", "execution_id", "phase", "method_protocol", "source_protocol",
        "rows", "source_sha256_lf", "evidence_sha256_lf", "input_locations_sha256", "api",
    }
    if set(protocol) != required:
        raise ValueError("실행 protocol 필드가 다르다")
    expected_api = {
        "total_cap": 72, "logical_decisions": 24, "max_calls_per_decision": 3,
        "semantic_feedback": 1, "infra_retry": 1, "sdk_retries": 0,
    }
    if (protocol.get("schema") != PROTOCOL_SCHEMA
            or _IDENTIFIER.fullmatch(str(protocol.get("protocol_id"))) is None
            or _IDENTIFIER.fullmatch(str(protocol.get("execution_id"))) is None
            or protocol.get("phase") != "comparison" or protocol.get("api") != expected_api
            or _HEX64.fullmatch(str(protocol.get("input_locations_sha256"))) is None):
        raise ValueError("실행 protocol ID·phase·API 계약이 다르다")
    method_name, _, method = _protocol_reference(repository, protocol["method_protocol"], "method protocol")
    source_name, _, source = _protocol_reference(repository, protocol["source_protocol"], "source protocol")
    sources = _hash_map(protocol["source_sha256_lf"], "source_sha256_lf")
    evidence = _hash_map(protocol["evidence_sha256_lf"], "evidence_sha256_lf")
    if set(sources) & set(evidence) or not REQUIRED_SOURCES <= set(sources):
        raise ValueError("실행 protocol source/evidence 경계가 중복됐거나 필수 runtime source가 없다")
    combined = {**sources, **evidence}
    for parent_union in (_parent_hashes(method, method=True), _parent_hashes(source, method=False)):
        if any(combined.get(name) != digest for name, digest in parent_union.items()):
            raise ValueError("부모 protocol 결합 SHA를 실행 protocol이 덮지 못했다")
    rows = protocol.get("rows")
    if not isinstance(rows, list):
        raise TypeError("실행 protocol rows가 목록이 아니다")
    if any(not isinstance(row, dict) or set(row) != COMPARISON.INPUT_FIELDS for row in rows):
        raise ValueError("실행 protocol row가 고정 INPUT_FIELDS와 다르다")
    method_sha = protocol["method_protocol"]["sha256_lf"]
    source_sha = protocol["source_protocol"]["sha256_lf"]
    checked_rows = COMPARISON.validate_inputs(
        rows, method_protocol_sha256=method_sha, source_protocol_sha256=source_sha,
    )
    if protocol_relative in combined or method_name not in combined or source_name not in combined:
        raise ValueError("실행 protocol 자체 또는 부모 protocol 결합 위치가 잘못됐다")
    return {**deepcopy(protocol), "rows": checked_rows, "_protocol_relative": protocol_relative}


def verify_source(protocol: dict[str, Any], protocol_path: Path, source_anchor: str,
                  *, root: Path = ROOT) -> dict[str, Any]:
    """anchor·HEAD·index blob 및 clean working LF bytes를 함께 확인한다."""
    repository = Path(root).resolve(strict=True)
    if _HEX40.fullmatch(source_anchor) is None:
        raise ValueError("source anchor 형식이 다르다")
    expected = {**protocol["source_sha256_lf"], **protocol["evidence_sha256_lf"]}
    protocol_relative = Path(protocol_path).resolve(strict=True).relative_to(repository).as_posix()
    expected[protocol_relative] = _sha_lf(Path(protocol_path))
    if len(expected) != len(protocol["source_sha256_lf"]) + len(protocol["evidence_sha256_lf"]) + 1:
        raise ValueError("source/evidence/protocol 경로가 중복됐다")
    for name, digest in expected.items():
        _, path = _relative_file(repository, name, "결합 파일")
        if _sha_lf(path) != digest:
            raise ValueError(f"결합 파일 LF SHA가 다르다: {name}")
    head = _git(repository, "rev-parse", "HEAD").stdout.strip()
    if _git(repository, "merge-base", "--is-ancestor", source_anchor, head, check=False).returncode != 0:
        raise ValueError("source anchor가 현재 HEAD의 조상이 아니다")
    names = sorted(expected)
    _git(repository, "ls-files", "--error-unmatch", "--", *names)
    if _git(repository, "status", "--porcelain", "--", *names).stdout.strip():
        raise ValueError("API 결합 파일의 index 또는 working tree가 clean 상태가 아니다")
    for name in names:
        anchor_blob = _git(repository, "rev-parse", f"{source_anchor}:{name}").stdout.strip()
        head_blob = _git(repository, "rev-parse", f"{head}:{name}").stdout.strip()
        index_blob = _git(repository, "rev-parse", f":{name}").stdout.strip()
        if not anchor_blob == head_blob == index_blob:
            raise ValueError(f"source anchor 뒤 결합 blob이 변경됐다: {name}")
    return {"source_anchor": source_anchor, "head": head, "verified_files": len(names)}


def _load_verified_inputs(root: Path, protocol: dict[str, Any], input_locations: Path,
                          verifier: Callable[[Path, dict[str, Any], Path], dict] | None) -> dict[str, Any]:
    if verifier is None:
        module = importlib.import_module("vision_selection_sources_v1")
        loaded = getattr(module, "verify_inputs", None)
        if not callable(loaded):
            raise RuntimeError("입력 검증 helper verify_inputs를 찾을 수 없다")
        checked_verifier = cast(Callable[[Path, dict[str, Any], Path], dict], loaded)
    else:
        checked_verifier = verifier
    value = checked_verifier(root, protocol, Path(input_locations))
    required = {
        "schema", "execution_id", "rows", "input_manifest_sha256", "source_sha256_lf",
        "evidence_sha256_lf", "input_locations_sha256",
    }
    if not isinstance(value, dict) or set(value) != required or value.get("schema") != VERIFIED_INPUTS_SCHEMA:
        raise ValueError("입력 검증 helper 반환 형식이 다르다")
    call_ids = [row["call_id"] for row in protocol["rows"]]
    plan_rows = value.get("rows")
    if (value.get("execution_id") != protocol["execution_id"]
            or value.get("source_sha256_lf") != protocol["source_sha256_lf"]
            or value.get("evidence_sha256_lf") != protocol["evidence_sha256_lf"]
            or value.get("input_locations_sha256") != protocol["input_locations_sha256"]
            or not isinstance(plan_rows, list) or len(plan_rows) != 24
            or [row.get("call_id") for row in plan_rows] != call_ids
            or any(not isinstance(row, dict) or set(row) != {"call_id", "run_id", "capsule"}
                   for row in plan_rows)):
        raise ValueError("입력 검증 helper 결합이 실행 protocol과 다르다")
    manifest_map = value.get("input_manifest_sha256")
    if (not isinstance(manifest_map, dict) or set(manifest_map) != set(call_ids)
            or any(_HEX64.fullmatch(str(digest)) is None for digest in manifest_map.values())):
        raise ValueError("입력 manifest SHA map이 고정 call 순서와 다르다")
    return deepcopy(value)


def prepare(
    protocol_path: Path,
    protocol_sha256_lf: str,
    source_anchor: str,
    input_locations: Path,
    *,
    root: Path = ROOT,
    input_verifier: Callable[[Path, dict[str, Any], Path], dict] | None = None,
) -> dict[str, Any]:
    """모든 source와 원본 입력을 검증한 뒤 frozen caller plan만 생성한다."""
    protocol = verify_protocol(protocol_path, protocol_sha256_lf, root=root)
    public_protocol = {key: value for key, value in protocol.items() if not key.startswith("_")}
    source = verify_source(protocol, protocol_path, source_anchor, root=root)
    inputs = _load_verified_inputs(Path(root).resolve(), public_protocol, input_locations, input_verifier)
    plan = CALL.initialize(
        root, protocol["execution_id"], "comparison", source_anchor,
        protocol_sha256_lf, inputs["rows"],
    )
    return {"protocol": public_protocol, "source": source, "inputs": inputs, "plan": plan}


def _start_ticks(pid: int) -> str:
    path = Path(f"/proc/{pid}/stat")
    if not path.is_file():
        raise RuntimeError("Linux process identity를 읽을 수 없다")
    return path.read_text(encoding="utf-8").rsplit(")", 1)[1].split()[19]


def _identity(stage: Path, call_id: str) -> dict[str, Any]:
    pid = os.getpid()
    getpgid, getsid = getattr(os, "getpgid", None), getattr(os, "getsid", None)
    if not callable(getpgid) or not callable(getsid):
        raise TypeError("Linux process group API를 찾을 수 없다")
    pgid, sid = getpgid(pid), getsid(pid)
    if pid != pgid or pid != sid:
        raise RuntimeError("worker는 setsid로 만든 전용 Linux process group이 필요하다")
    return {"pid": pid, "pgid": pgid, "sid": sid, "start_ticks": _start_ticks(pid),
            "stage": str(stage.resolve()), "call_id": call_id}


def _write_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_canonical(value))
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def _receipt_directory(stage: Path) -> Path:
    """Return the one canonical receipt directory without following links."""
    canonical_stage = Path(stage)
    if (not canonical_stage.is_absolute() or not canonical_stage.is_dir()
            or canonical_stage.is_symlink()
            or any(parent.is_symlink() for parent in canonical_stage.parents)
            or canonical_stage.resolve(strict=True) != canonical_stage):
        raise JOURNAL.JournalIntegrityError("selection stage가 canonical directory가 아니다")
    folder = canonical_stage / "worker-receipts"
    if folder.exists() or folder.is_symlink():
        if (not folder.is_dir() or folder.is_symlink()
                or folder.resolve(strict=True) != folder
                or folder.parent != canonical_stage):
            raise JOURNAL.JournalIntegrityError("worker receipt directory가 canonical child가 아니다")
    else:
        folder.mkdir()
    if (folder.is_symlink() or any(parent.is_symlink() for parent in folder.parents)
            or folder.resolve(strict=True) != folder or folder.parent != canonical_stage):
        raise JOURNAL.JournalIntegrityError("worker receipt directory가 쓰기 전에 변경됐다")
    return folder


def _write_receipt_exclusive(stage: Path, path: Path, value: dict[str, Any]) -> None:
    folder = _receipt_directory(stage)
    target = Path(path)
    if target.parent != folder or target.name != Path(target.name).name:
        raise JOURNAL.JournalIntegrityError("worker receipt 경로가 고정 directory 밖이다")
    # 쓰기 직전에도 parent와 모든 ancestor가 link가 아닌지 다시 확인한다.
    if folder.is_symlink() or any(parent.is_symlink() for parent in folder.parents):
        raise JOURNAL.JournalIntegrityError("worker receipt parent가 쓰기 전에 변경됐다")
    if os.name != "posix":
        _write_exclusive(target, value)
        return

    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptors: list[int] = []
    created = False
    verified = False
    folder_descriptor: int | None = None
    try:
        current = Path(stage).anchor
        descriptor = os.open(current, directory_flags)
        descriptors.append(descriptor)
        for component in Path(stage).parts[1:]:
            next_descriptor = os.open(component, directory_flags, dir_fd=descriptor)
            descriptors.append(next_descriptor)
            descriptor = next_descriptor
        stage_descriptor = descriptor
        folder_descriptor = os.open("worker-receipts", directory_flags, dir_fd=stage_descriptor)
        descriptors.append(folder_descriptor)
        folder_identity = os.fstat(folder_descriptor)
        file_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
        file_descriptor = os.open(target.name, file_flags, 0o600, dir_fd=folder_descriptor)
        created = True
        with os.fdopen(file_descriptor, "wb") as stream:
            stream.write(_canonical(value))
            stream.flush()
            os.fsync(stream.fileno())
            file_identity = os.fstat(stream.fileno())
        visible_folder = os.open("worker-receipts", directory_flags, dir_fd=stage_descriptor)
        descriptors.append(visible_folder)
        visible_identity = os.fstat(visible_folder)
        visible_file = os.open(target.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=visible_folder)
        descriptors.append(visible_file)
        visible_file_identity = os.fstat(visible_file)
        if ((folder_identity.st_dev, folder_identity.st_ino)
                != (visible_identity.st_dev, visible_identity.st_ino)
                or (file_identity.st_dev, file_identity.st_ino)
                != (visible_file_identity.st_dev, visible_file_identity.st_ino)):
            raise JOURNAL.JournalIntegrityError("worker receipt canonical 이름이 쓰기 중 변경됐다")
        verified = True
    except OSError as error:
        raise JOURNAL.JournalIntegrityError("worker receipt directory가 쓰기 중 변경됐다") from error
    finally:
        if created and not verified and folder_descriptor is not None:
            try:
                os.unlink(target.name, dir_fd=folder_descriptor)
            except OSError:
                pass
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _owned(path: Path, identity: dict[str, Any]) -> bool:
    try:
        getpgid, getsid = getattr(os, "getpgid", None), getattr(os, "getsid", None)
        if not callable(getpgid) or not callable(getsid):
            return False
        return (_read(path) == identity and _start_ticks(identity["pid"]) == identity["start_ticks"]
                and getpgid(identity["pid"]) == identity["pgid"]
                and getsid(identity["pid"]) == identity["sid"])
    except (OSError, RuntimeError, ValueError, TypeError):
        return False


def _next_receipt(folder: Path) -> Path:
    if not folder.is_dir() or folder.is_symlink() or folder.resolve(strict=True) != folder:
        raise JOURNAL.JournalIntegrityError("worker receipt directory가 canonical path가 아니다")
    rows = sorted(folder.iterdir())
    expected = [f"{index:04d}.json" for index in range(1, len(rows) + 1)]
    if ([row.name for row in rows] != expected
            or any(not row.is_file() or row.is_symlink() for row in rows)):
        raise JOURNAL.JournalIntegrityError("worker invocation receipt 연속성이 깨졌다")
    return folder / f"{len(rows) + 1:04d}.json"


def _preflight_provider() -> None:
    if not os.environ.get("NVIDIA_API_KEY"):
        raise RuntimeError("NVIDIA_API_KEY가 없다")
    if importlib.util.find_spec("openai") is None:
        raise RuntimeError("openai package를 찾을 수 없다")


def _default_complete(messages: list[dict], params: dict) -> dict[str, Any]:
    return OBSERVER.default_complete(messages, params)


def advance(
    protocol_path: Path,
    protocol_sha256_lf: str,
    source_anchor: str,
    input_locations: Path,
    call_id: str,
    *,
    root: Path = ROOT,
    input_verifier: Callable[[Path, dict[str, Any], Path], dict] | None = None,
    provider_preflight: Callable[[], None] = _preflight_provider,
) -> tuple[dict[str, Any], int]:
    """고정 순서의 한 선택을 advance하고 호출 receipt를 append-only로 남긴다."""
    if not sys.platform.startswith("linux"):
        raise RuntimeError("영상 선택 worker는 등록 WSL Linux에서만 실행한다")
    prepared = prepare(
        protocol_path, protocol_sha256_lf, source_anchor, input_locations,
        root=root, input_verifier=input_verifier,
    )
    if call_id not in [row["call_id"] for row in prepared["protocol"]["rows"]]:
        raise ValueError("실행 protocol에 등록되지 않은 call_id다")
    stage = Path(prepared["plan"]["path"]).parent
    receipt_folder = _receipt_directory(stage)
    terminal_path = stage / "decisions" / call_id / "terminal.json"
    if not terminal_path.is_file():
        provider_preflight()
    identity = _identity(stage, call_id)
    identity_path, lease_path = stage / "worker.identity.json", stage / ".active"
    if identity_path.exists() or identity_path.is_symlink() or lease_path.exists() or lease_path.is_symlink():
        return {"status": "blocked_pending", "reason": "active_or_stale_worker_preserved",
                "call_id": call_id, "provider_calls": 0}, EXIT_CODES["blocked_pending"]
    try:
        _write_exclusive(identity_path, identity)
    except FileExistsError:
        return {"status": "blocked_pending", "reason": "active_or_stale_worker_preserved",
                "call_id": call_id, "provider_calls": 0}, EXIT_CODES["blocked_pending"]
    try:
        try:
            _write_exclusive(lease_path, identity)
        except FileExistsError:
            return {"status": "blocked_pending", "reason": "active_or_stale_lease_preserved",
                    "call_id": call_id, "provider_calls": 0}, EXIT_CODES["blocked_pending"]
        try:
            result = CALL.run(prepared["plan"], call_id, _default_complete)
            status = result.get("journal", {}).get("status")
            if status not in EXIT_CODES:
                raise RuntimeError("caller가 등록되지 않은 상태를 반환했다")
            receipt = {
                "schema": RECEIPT_SCHEMA, "call_id": call_id, "status": status,
                "plan_sha256": prepared["plan"]["sha256"],
                "protocol_sha256_lf": protocol_sha256_lf,
                "input_locations_sha256": prepared["inputs"]["input_locations_sha256"],
                "api": deepcopy(result["journal"]["api"]),
            }
            _write_receipt_exclusive(stage, _next_receipt(receipt_folder), receipt)
            return result, EXIT_CODES[status]
        finally:
            if not _owned(lease_path, identity):
                raise JOURNAL.JournalIntegrityError("worker active lease가 실행 중 변경됐다")
            lease_path.unlink()
    finally:
        if not _owned(identity_path, identity):
            raise JOURNAL.JournalIntegrityError("worker process identity가 실행 중 변경됐다")
        identity_path.unlink()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify", "advance"))
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256-lf", required=True)
    parser.add_argument("--source-anchor", required=True)
    parser.add_argument("--input-locations", type=Path, required=True)
    parser.add_argument("--call-id")
    args = parser.parse_args(argv)
    if args.command == "verify":
        if args.call_id is not None:
            parser.error("verify에는 --call-id를 쓰지 않는다")
        result = prepare(args.protocol, args.protocol_sha256_lf, args.source_anchor, args.input_locations)
        print(json.dumps({"status": "verified", "plan": result["plan"]}, ensure_ascii=False,
                         sort_keys=True, allow_nan=False))
        return 0
    if args.call_id is None:
        parser.error("advance에는 --call-id가 필요하다")
    result, code = advance(
        args.protocol, args.protocol_sha256_lf, args.source_anchor,
        args.input_locations, args.call_id,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
