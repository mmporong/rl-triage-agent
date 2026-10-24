"""사전등록 source 결합 뒤 grounded vision caller를 한 번 advance하는 WSL worker."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "evals") not in sys.path:
    sys.path.insert(0, str(ROOT / "evals"))

import grounded_vision_observer_v1 as OBSERVER

PROTOCOL_SCHEMA = "grounded_vision_development_protocol_v1"
RECEIPT_SCHEMA = "grounded_vision_worker_receipt_v1"
STAGE_SCHEMA = "grounded_vision_stage_manifest_v1"
COMPLETION_SCHEMA = "grounded_vision_input_completion_v1"
API_CAP = 6
REQUIRED_SOURCE_FILES = frozenset({
    "evals/feedback_model_call.py",
    "evals/grounded_vision_observer_v1.py",
    "evals/grounded_vision_worker_v1.py",
    "evals/model_decision_journal_v1.py",
    "evals/nemotron_bottleneck_ledger_v2.py",
    "evals/nemotron_endpoint_ledger_v3.py",
    "evals/vision_evidence_contract_v1.py",
})
EXIT_CODES = {
    "observation_ready": 0,
    "terminal_hold": 2,
    "paused_infra": 3,
    "blocked_pending": 4,
}
_HEX40 = re.compile(r"[0-9a-f]{40}")
_HEX64 = re.compile(r"[0-9a-f]{64}")
_IDENTIFIER = re.compile(r"[a-z0-9][a-z0-9._-]{0,79}")


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(
        Path(path).read_text(encoding="utf-8"),
        parse_constant=lambda raw: (_ for _ in ()).throw(ValueError(f"JSON constant 거부: {raw}")),
    )
    if not isinstance(value, dict):
        raise TypeError(f"{Path(path).name}은 JSON object여야 한다")
    return value


def _sha_lf(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _git(root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=check,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
    )


def _hash_map(value: object, label: str) -> dict[str, str]:
    if (not isinstance(value, dict) or not value
            or any(not isinstance(name, str) or not name or name.startswith("/")
                   or "\\" in name or ".." in name.split("/")
                   or not isinstance(digest, str) or _HEX64.fullmatch(digest) is None
                   for name, digest in value.items())):
        raise ValueError(f"protocol {label} 결합이 잘못됐다")
    return deepcopy(value)


def verify_protocol(
    protocol_path: Path,
    contract_path: Path,
    binding: dict[str, Any],
) -> dict[str, Any]:
    protocol = _read(protocol_path)
    required = {
        "schema", "protocol_id", "model", "provider_aliases", "api",
        "development_inputs", "fixed_order", "registration",
        "source_sha256_lf", "evidence_sha256_lf",
    }
    if set(protocol) != required:
        raise ValueError("개발 protocol 필드가 다르다")
    expected_api = {
        "total_cap": API_CAP,
        "dialogue_cap": OBSERVER.MAX_CALLS,
        "infra_retry": 1,
        "semantic_feedback": 1,
        "sdk_retries": 0,
    }
    if (protocol.get("schema") != PROTOCOL_SCHEMA
            or _IDENTIFIER.fullmatch(str(protocol.get("protocol_id"))) is None
            or protocol.get("model") != OBSERVER.MODEL
            or protocol.get("provider_aliases") != sorted(OBSERVER.ACCEPTED_PROVIDER_MODELS)
            or protocol.get("api") != expected_api):
        raise ValueError("개발 protocol 모델·API 계약이 다르다")
    sources = _hash_map(protocol.get("source_sha256_lf"), "source_sha256_lf")
    _hash_map(protocol.get("evidence_sha256_lf"), "evidence_sha256_lf")
    registration = protocol.get("registration")
    if not isinstance(registration, dict) or not registration:
        raise ValueError("protocol registration이 비어 있거나 object가 아니다")
    OBSERVER.JOURNAL._validate_json(registration, "protocol registration")
    if not REQUIRED_SOURCE_FILES <= set(sources):
        raise ValueError("protocol source 결합에 관찰 runtime 필수 파일이 없다")
    contract = OBSERVER.EVIDENCE.verify(contract_path)
    checkpoint_sha = contract["provenance"]["checkpoint"]["sha256"]
    inputs = protocol.get("development_inputs")
    if (not isinstance(inputs, list) or not 1 <= len(inputs) <= API_CAP
            or any(not isinstance(row, dict)
                   or set(row) != {"logical_id", "contract_sha256", "checkpoint_sha256"}
                   or _IDENTIFIER.fullmatch(str(row.get("logical_id"))) is None
                   or _HEX64.fullmatch(str(row.get("contract_sha256"))) is None
                   or _HEX64.fullmatch(str(row.get("checkpoint_sha256"))) is None
                   for row in inputs)
            or len({row["logical_id"] for row in inputs}) != len(inputs)
            or len({row["contract_sha256"] for row in inputs}) != len(inputs)
            or protocol.get("fixed_order") != [row["logical_id"] for row in inputs]):
        raise ValueError("protocol의 개발 입력 목록이 잘못됐다")
    matched = [row for row in inputs
               if row["contract_sha256"] == OBSERVER.EVIDENCE.sha256(contract_path)
               and row["checkpoint_sha256"] == checkpoint_sha]
    if len(matched) != 1:
        raise ValueError("protocol에 등록되지 않은 개발 contract다")
    if binding.get("run_id") != f"v02-dev-{matched[0]['logical_id']}":
        raise ValueError("binding run_id가 protocol canonical 입력과 다르다")
    if binding.get("checkpoint_sha256") != checkpoint_sha:
        raise ValueError("protocol의 개발 checkpoint SHA가 다르다")
    if binding.get("protocol_sha256") != _sha_lf(protocol_path):
        raise ValueError("binding protocol SHA가 현재 protocol과 다르다")
    return protocol


def _registered_input(protocol: dict[str, Any], contract_path: Path) -> tuple[int, dict[str, str]]:
    contract = OBSERVER.EVIDENCE.verify(contract_path)
    contract_sha = OBSERVER.EVIDENCE.sha256(contract_path)
    checkpoint_sha = contract["provenance"]["checkpoint"]["sha256"]
    rows = protocol["development_inputs"]
    matches = [(index, row) for index, row in enumerate(rows)
               if row["contract_sha256"] == contract_sha
               and row["checkpoint_sha256"] == checkpoint_sha]
    if len(matches) != 1:
        raise ValueError("protocol 개발 입력을 하나로 확정할 수 없다")
    return matches[0]


def _runtime_paths(root: Path, protocol: dict[str, Any], contract_path: Path) -> dict[str, Path]:
    _, row = _registered_input(protocol, contract_path)
    stage = Path(root).absolute() / ".triage-runtime" / "grounded-vision" / protocol["protocol_id"]
    logical_id = row["logical_id"]
    return {
        "stage": stage,
        "api": stage / "api",
        "folder": stage / logical_id,
        "decision": stage / logical_id / "decision",
        "receipts": stage / f"{logical_id}.receipts",
        "completions": stage / "completions",
        "pid": stage / f"{logical_id}.identity.json",
    }


def _reject_symlink_chain(path: Path, root: Path) -> None:
    repository = Path(root).absolute()
    candidate = Path(path).absolute()
    try:
        relative = candidate.relative_to(repository)
    except ValueError as error:
        raise ValueError("runtime 경로가 저장소 밖을 가리킨다") from error
    current = repository
    if current.is_symlink():
        raise ValueError("저장소 runtime 경로의 symlink를 거부한다")
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise ValueError("runtime 경로 부모의 symlink를 거부한다")


def _require_path(actual: Path, expected: Path, label: str, root: Path) -> None:
    _reject_symlink_chain(actual, root)
    if Path(actual).absolute() != Path(expected).absolute():
        raise ValueError(f"{label}가 protocol의 canonical runtime 경로와 다르다")


def _stage_manifest(protocol: dict[str, Any], protocol_path: Path,
                    binding: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": STAGE_SCHEMA,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256_lf": _sha_lf(protocol_path),
        "source_anchor": binding["source_anchor"],
        "fixed_order": deepcopy(protocol["fixed_order"]),
        "development_inputs": deepcopy(protocol["development_inputs"]),
    }


def _prepare_stage(stage: Path, expected: dict[str, Any]) -> Path:
    stage.mkdir(parents=True, exist_ok=True)
    path = stage / "manifest.json"
    if path.exists():
        if OBSERVER.JOURNAL._read(path) != expected:
            raise OBSERVER.JOURNAL.JournalIntegrityError("기존 stage manifest 결합이 다르다")
    else:
        unknown = {item.name for item in stage.iterdir()}
        if unknown:
            raise OBSERVER.JOURNAL.JournalIntegrityError("manifest 없는 stage 상태를 인수하지 않는다")
        try:
            OBSERVER.JOURNAL._write_new(path, expected)
        except FileExistsError:
            if OBSERVER.JOURNAL._read(path) != expected:
                raise OBSERVER.JOURNAL.JournalIntegrityError("경합 중 생성된 stage manifest가 다르다")
    return path


def _check_fixed_order(stage: Path, protocol: dict[str, Any], contract_path: Path) -> None:
    index, _ = _registered_input(protocol, contract_path)
    for logical_id in protocol["fixed_order"][:index]:
        _verify_completion(stage, protocol, logical_id)


def _next_receipt_path(folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    if folder.is_symlink():
        raise ValueError("receipt directory symlink를 거부한다")
    rows = sorted(folder.iterdir())
    expected = [f"{index:04d}.json" for index in range(1, len(rows) + 1)]
    if [row.name for row in rows] != expected or any(not row.is_file() or row.is_symlink() for row in rows):
        raise OBSERVER.JOURNAL.JournalIntegrityError("worker receipt 연속성이 깨졌다")
    return folder / f"{len(rows) + 1:04d}.json"


def _numbered_files(folder: Path, count: int) -> list[dict[str, str]]:
    rows = []
    for index in range(1, count + 1):
        name = f"{index:04d}.json"
        path = folder / name
        if not path.is_file() or path.is_symlink():
            raise OBSERVER.JOURNAL.JournalIntegrityError(f"봉인 대상 원장 파일이 없다: {folder.name}/{name}")
        rows.append({"name": name, "sha256": OBSERVER.JOURNAL.file_sha(path)})
    return rows


def _completion_value(stage: Path, protocol: dict[str, Any], protocol_path: Path,
                      logical_id: str, binding: dict[str, Any], status: str) -> dict[str, Any]:
    row = next(item for item in protocol["development_inputs"] if item["logical_id"] == logical_id)
    decision_root = stage / logical_id / "decision"
    api_root = stage / "api"
    manifest_path = decision_root / "manifest.json"
    terminal_path = decision_root / "terminal.json"
    if not manifest_path.is_file() or not terminal_path.is_file():
        raise OBSERVER.JOURNAL.JournalIntegrityError("terminal 완료 원장 파일이 없다")
    journal = OBSERVER.JOURNAL.DecisionJournal(
        decision_root, api_root, binding, OBSERVER.REQUEST_PARAMS,
        retry_delay_s=30, api_cap=API_CAP,
    )
    journal._verify_prepared()
    audit = journal._audit_api()
    if audit["pending"]:
        raise OBSERVER.JOURNAL.JournalIntegrityError("완료 시점 전역 API 원장에 pending이 있다")
    terminal = OBSERVER.JOURNAL._read(terminal_path)
    if (terminal.get("schema") != "model_decision_terminal_v1"
            or terminal.get("manifest_sha256") != journal.manifest_sha256
            or terminal.get("status") not in OBSERVER.JOURNAL.TERMINAL_STATUSES):
        raise OBSERVER.JOURNAL.JournalIntegrityError("완료 terminal 결합이 잘못됐다")
    expected_status = "terminal_hold" if terminal["status"] == "terminal_hold" else "observation_ready"
    if status != expected_status:
        raise OBSERVER.JOURNAL.JournalIntegrityError("worker 상태와 terminal 상태가 다르다")
    events = journal._events()
    if len(events) != len(audit["own"]):
        raise OBSERVER.JOURNAL.JournalIntegrityError("decision events와 API 호출 수가 다르다")
    prefix_count = audit["reserved"]
    api_prefix = {
        name: _numbered_files(api_root / name, prefix_count)
        for name in ("reservations", "responses", "receipts")
    }
    event_rows = _numbered_files(decision_root / "events", len(events))
    return {
        "schema": COMPLETION_SCHEMA,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256_lf": _sha_lf(protocol_path),
        "logical_id": logical_id,
        "contract_sha256": row["contract_sha256"],
        "checkpoint_sha256": row["checkpoint_sha256"],
        "binding": deepcopy(binding),
        "binding_sha256": OBSERVER.JOURNAL.digest(binding),
        "decision_manifest_sha256": OBSERVER.JOURNAL.file_sha(manifest_path),
        "terminal_sha256": OBSERVER.JOURNAL.file_sha(terminal_path),
        "events": event_rows,
        "events_sha256": OBSERVER.JOURNAL.digest(event_rows),
        "api_prefix_count": prefix_count,
        "api_prefix": api_prefix,
        "api_prefix_sha256": OBSERVER.JOURNAL.digest(api_prefix),
        "worker_status": status,
    }


def _verify_file_rows(folder: Path, rows: object, label: str) -> None:
    if (not isinstance(rows, list)
            or any(not isinstance(row, dict) or set(row) != {"name", "sha256"}
                   or row.get("name") != f"{index:04d}.json"
                   or _HEX64.fullmatch(str(row.get("sha256"))) is None
                   for index, row in enumerate(rows, 1))):
        raise OBSERVER.JOURNAL.JournalIntegrityError(f"완료 receipt {label} 목록이 잘못됐다")
    for row in rows:
        path = folder / row["name"]
        if (not path.is_file() or path.is_symlink()
                or OBSERVER.JOURNAL.file_sha(path) != row["sha256"]):
            raise OBSERVER.JOURNAL.JournalIntegrityError(f"완료 receipt {label} 봉인이 다르다")


def _verify_completion(stage: Path, protocol: dict[str, Any], logical_id: str) -> dict[str, Any]:
    path = stage / "completions" / f"{logical_id}.json"
    if not path.is_file() or path.is_symlink():
        raise ValueError("protocol fixed_order 앞 입력의 완료 receipt가 없다")
    value = OBSERVER.JOURNAL._read(path)
    stage_manifest = OBSERVER.JOURNAL._read(stage / "manifest.json")
    row = next(item for item in protocol["development_inputs"] if item["logical_id"] == logical_id)
    required = {
        "schema", "protocol_id", "protocol_sha256_lf", "logical_id", "contract_sha256",
        "checkpoint_sha256", "binding", "binding_sha256",
        "decision_manifest_sha256", "terminal_sha256", "events", "events_sha256",
        "api_prefix_count", "api_prefix", "api_prefix_sha256", "worker_status",
    }
    if (set(value) != required or value.get("schema") != COMPLETION_SCHEMA
            or value.get("protocol_id") != protocol["protocol_id"]
            or value.get("protocol_sha256_lf") != stage_manifest.get("protocol_sha256_lf")
            or value.get("logical_id") != logical_id
            or value.get("contract_sha256") != row["contract_sha256"]
            or value.get("checkpoint_sha256") != row["checkpoint_sha256"]
            or value.get("worker_status") not in {"observation_ready", "terminal_hold"}
            or value.get("binding_sha256") != OBSERVER.JOURNAL.digest(value.get("binding"))):
        raise OBSERVER.JOURNAL.JournalIntegrityError("앞 입력 완료 receipt 결합이 다르다")
    binding = value["binding"]
    OBSERVER.JOURNAL._binding(binding)
    if (binding.get("source_anchor") != stage_manifest.get("source_anchor")
            or binding.get("protocol_sha256") != stage_manifest.get("protocol_sha256_lf")
            or binding.get("checkpoint_sha256") != row["checkpoint_sha256"]
            or binding.get("run_id") != f"v02-dev-{logical_id}"):
        raise OBSERVER.JOURNAL.JournalIntegrityError("앞 입력 binding과 protocol 결합이 다르다")
    decision_root = stage / logical_id / "decision"
    manifest_path = decision_root / "manifest.json"
    terminal_path = decision_root / "terminal.json"
    if (not manifest_path.is_file() or OBSERVER.JOURNAL.file_sha(manifest_path) != value["decision_manifest_sha256"]
            or not terminal_path.is_file() or OBSERVER.JOURNAL.file_sha(terminal_path) != value["terminal_sha256"]):
        raise OBSERVER.JOURNAL.JournalIntegrityError("앞 입력 decision/terminal 봉인이 다르다")
    terminal = OBSERVER.JOURNAL._read(terminal_path)
    expected_status = "terminal_hold" if terminal.get("status") == "terminal_hold" else "observation_ready"
    if value.get("worker_status") != expected_status:
        raise OBSERVER.JOURNAL.JournalIntegrityError("앞 입력 worker/terminal 상태 결합이 다르다")
    _verify_file_rows(decision_root / "events", value["events"], "events")
    if value.get("events_sha256") != OBSERVER.JOURNAL.digest(value["events"]):
        raise OBSERVER.JOURNAL.JournalIntegrityError("앞 입력 events 최종 SHA가 다르다")
    prefix_count = value.get("api_prefix_count")
    api_prefix = value.get("api_prefix")
    if (type(prefix_count) is not int or prefix_count < 0 or not isinstance(api_prefix, dict)
            or set(api_prefix) != {"reservations", "responses", "receipts"}
            or any(len(rows) != prefix_count for rows in api_prefix.values())
            or value.get("api_prefix_sha256") != OBSERVER.JOURNAL.digest(api_prefix)):
        raise OBSERVER.JOURNAL.JournalIntegrityError("앞 입력 API prefix 형식이 다르다")
    api_root = stage / "api"
    for name, rows in api_prefix.items():
        _verify_file_rows(api_root / name, rows, f"API {name}")
    journal = OBSERVER.JOURNAL.DecisionJournal(
        decision_root, api_root, binding, OBSERVER.REQUEST_PARAMS,
        retry_delay_s=30, api_cap=API_CAP,
    )
    journal._verify_prepared()
    audit = journal._audit_api()
    if (terminal.get("manifest_sha256") != journal.manifest_sha256
            or terminal.get("status") not in OBSERVER.JOURNAL.TERMINAL_STATUSES
            or len(journal._events()) != len(audit["own"])
            or audit["reserved"] < prefix_count):
        raise OBSERVER.JOURNAL.JournalIntegrityError("앞 입력 완료 원장 재감사가 실패했다")
    return value


def _record_completion(stage: Path, protocol: dict[str, Any], protocol_path: Path,
                       contract_path: Path, binding: dict[str, Any], status: str) -> None:
    _, row = _registered_input(protocol, contract_path)
    path = stage / "completions" / f"{row['logical_id']}.json"
    if path.exists():
        _verify_completion(stage, protocol, row["logical_id"])
        return
    value = _completion_value(
        stage, protocol, protocol_path, row["logical_id"], binding, status,
    )
    OBSERVER.JOURNAL._write_new(path, value)
    if _verify_completion(stage, protocol, row["logical_id"]) != value:
        raise OBSERVER.JOURNAL.JournalIntegrityError("기록한 완료 receipt 재검증이 다르다")


def verify_source(
    binding: dict[str, Any],
    protocol_path: Path,
    contract_path: Path,
    *,
    root: Path = ROOT,
) -> dict[str, Any]:
    """API 호출 전에 protocol, 결합 파일, anchor blob, clean 상태를 검사한다."""
    repository = Path(root).resolve()
    protocol_file = Path(protocol_path).resolve()
    try:
        protocol_relative = protocol_file.relative_to(repository).as_posix()
    except ValueError as error:
        raise ValueError("protocol은 저장소 안에 있어야 한다") from error
    protocol = verify_protocol(protocol_file, contract_path, binding)
    tables = [
        _hash_map(protocol["source_sha256_lf"], "source_sha256_lf"),
        _hash_map(protocol["evidence_sha256_lf"], "evidence_sha256_lf"),
    ]
    expected = {name: digest for table in tables for name, digest in table.items()}
    if len(expected) != sum(len(table) for table in tables):
        raise ValueError("source와 evidence 결합 경로가 중복됐다")
    verified = []
    for name, digest in expected.items():
        candidate = (repository / name).resolve()
        try:
            relative = candidate.relative_to(repository).as_posix()
        except ValueError as error:
            raise ValueError("결합 경로가 저장소 밖을 가리킨다") from error
        if relative != name or not candidate.is_file() or _sha_lf(candidate) != digest:
            raise ValueError(f"결합 파일 LF SHA가 다르다: {name}")
        verified.append(relative)
    anchor = binding.get("source_anchor")
    if not isinstance(anchor, str) or _HEX40.fullmatch(anchor) is None:
        raise ValueError("source anchor 형식이 다르다")
    head = _git(repository, "rev-parse", "HEAD").stdout.strip()
    if _git(repository, "merge-base", "--is-ancestor", anchor, head, check=False).returncode != 0:
        raise ValueError("source anchor가 현재 HEAD의 조상이 아니다")
    bound = sorted({*verified, protocol_relative})
    _git(repository, "ls-files", "--error-unmatch", "--", *bound)
    if _git(repository, "status", "--porcelain", "--", *bound).stdout.strip():
        raise ValueError("API 결합 파일이 clean 상태가 아니다")
    for name in bound:
        anchor_blob = _git(repository, "rev-parse", f"{anchor}:{name}").stdout.strip()
        head_blob = _git(repository, "rev-parse", f"{head}:{name}").stdout.strip()
        if anchor_blob != head_blob:
            raise ValueError(f"source anchor 뒤 결합 blob이 변경됐다: {name}")
    return {
        "source_anchor": anchor,
        "head": head,
        "protocol_sha256_lf": _sha_lf(protocol_file),
        "verified_source_files": len(tables[0]),
        "verified_evidence_files": len(tables[1]),
    }


def _start_ticks(pid: int) -> str:
    stat = Path(f"/proc/{pid}/stat")
    if not stat.is_file():
        raise RuntimeError("Linux process identity를 읽을 수 없다")
    return stat.read_text(encoding="utf-8").rsplit(")", 1)[1].split()[19]


def _process_group() -> tuple[int, int, int]:
    pid = os.getpid()
    pgid = os.getpgid(pid)
    sid = os.getsid(pid)
    if pgid != pid or sid != pid:
        raise RuntimeError("grounded vision worker는 setsid로 만든 전용 process group이 필요하다")
    return pid, pgid, sid


def _identity(contract_path: Path, binding_path: Path, protocol_path: Path,
              paths: dict[str, Path]) -> dict[str, Any]:
    pid, pgid, sid = _process_group()
    return {
        "pid": pid,
        "start_ticks": _start_ticks(pid),
        "pgid": pgid,
        "sid": sid,
        "contract_sha256": OBSERVER.JOURNAL.file_sha(contract_path),
        "binding_sha256": OBSERVER.JOURNAL.file_sha(binding_path),
        "protocol_sha256": OBSERVER.JOURNAL.file_sha(protocol_path),
        "canonical_stage_path": str(paths["stage"]),
        "canonical_api_path": str(paths["api"]),
        "canonical_decision_path": str(paths["decision"]),
    }


def _verify_live_identity(path: Path, expected: dict[str, Any]) -> None:
    if OBSERVER.JOURNAL._read(path) != expected:
        raise OBSERVER.JOURNAL.JournalIntegrityError("worker process identity가 실행 중 변경됐다")
    if (_start_ticks(expected["pid"]) != expected["start_ticks"]
            or os.getpgid(expected["pid"]) != expected["pgid"]
            or os.getsid(expected["pid"]) != expected["sid"]):
        raise OBSERVER.JOURNAL.JournalIntegrityError("worker process identity가 stale 상태다")


def _default_complete(messages: list[dict], params: dict) -> dict:
    return OBSERVER.default_complete(messages, params)


def execute(
    contract_path: Path,
    binding_path: Path,
    protocol_path: Path,
    folder: Path,
    api_root: Path,
    receipt_dir: Path,
    *,
    complete=None,
    root: Path = ROOT,
) -> tuple[dict[str, Any], int]:
    """검증된 한 번의 journal advance를 실행하고 상태별 exit code receipt를 남긴다."""
    if sys.platform != "linux":
        raise RuntimeError("grounded vision worker는 등록 WSL Linux에서만 실행한다")
    binding = _read(binding_path)
    source = verify_source(binding, protocol_path, contract_path, root=root)
    OBSERVER.checked_binding(contract_path, binding)
    protocol = verify_protocol(protocol_path, contract_path, binding)
    paths = _runtime_paths(root, protocol, contract_path)
    _require_path(folder, paths["folder"], "decision folder", root)
    _require_path(api_root, paths["api"], "API root", root)
    _require_path(receipt_dir, paths["receipts"], "receipt directory", root)
    for path in paths.values():
        _reject_symlink_chain(path, root)
    manifest_path = _prepare_stage(paths["stage"], _stage_manifest(protocol, protocol_path, binding))
    owner = {"scope": "grounded_vision_stage", "manifest_sha256": OBSERVER.JOURNAL.file_sha(manifest_path)}
    provider = complete if complete is not None else _default_complete
    with OBSERVER.JOURNAL._lease(paths["stage"] / ".active", owner):
        _check_fixed_order(paths["stage"], protocol, contract_path)
        result = OBSERVER.run(
            paths["folder"],
            contract_path,
            binding,
            provider,
            api_root=paths["api"],
            api_cap=API_CAP,
        )
        status = result.get("status")
        if status not in EXIT_CODES:
            raise RuntimeError("worker가 등록되지 않은 실행 상태를 반환했다")
        exit_code = EXIT_CODES[status]
        receipt = {
            "schema": RECEIPT_SCHEMA,
            "contract_sha256": OBSERVER.EVIDENCE.sha256(contract_path),
            "binding_sha256": OBSERVER.JOURNAL.file_sha(binding_path),
            "protocol_sha256": OBSERVER.JOURNAL.file_sha(protocol_path),
            "source_verification": source,
            "status": status,
            "api": deepcopy(result["api"]),
            "exit_code": exit_code,
            "observation_valid": status == "observation_ready",
        }
        if status in {"observation_ready", "terminal_hold"}:
            _record_completion(
                paths["stage"], protocol, protocol_path, contract_path,
                binding, status,
            )
        OBSERVER.JOURNAL._write_new(_next_receipt_path(paths["receipts"]), receipt)
    return result, exit_code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--api-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--pid-file", type=Path, required=True)
    args = parser.parse_args(argv)
    if sys.platform != "linux":
        raise RuntimeError("grounded vision worker는 등록 WSL Linux에서만 실행한다")
    binding = _read(args.binding)
    verify_source(binding, args.protocol, args.contract, root=ROOT)
    OBSERVER.checked_binding(args.contract, binding)
    protocol = verify_protocol(args.protocol, args.contract, binding)
    paths = _runtime_paths(ROOT, protocol, args.contract)
    _require_path(args.folder, paths["folder"], "decision folder", ROOT)
    _require_path(args.api_root, paths["api"], "API root", ROOT)
    _require_path(args.receipt, paths["receipts"], "receipt directory", ROOT)
    _require_path(args.pid_file, paths["pid"], "PID identity", ROOT)
    _prepare_stage(paths["stage"], _stage_manifest(protocol, args.protocol, binding))
    identity = _identity(args.contract, args.binding, args.protocol, paths)
    OBSERVER.JOURNAL._write_new(paths["pid"], identity)
    try:
        result, exit_code = execute(
            args.contract,
            args.binding,
            args.protocol,
            args.folder,
            args.api_root,
            args.receipt,
            root=ROOT,
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
        return exit_code
    finally:
        _verify_live_identity(paths["pid"], identity)
        paths["pid"].unlink()


if __name__ == "__main__":
    raise SystemExit(main())
