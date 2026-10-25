"""동결된 영상 선택 입력만 호출하고 live 상태는 activation에서 다시 검사하는 CPU caller."""
from __future__ import annotations

import json
import re
import stat
import sys
from collections.abc import Callable
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _directory in (ROOT / "src", ROOT / "evals"):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

import grounded_vision_observer_v1 as OBSERVER
import model_decision_journal_v1 as JOURNAL
import vision_selection_contract_v1 as CONTRACT
import vision_selection_loop_v1 as LOOP

CAPSULE_SCHEMA = "vision_selection_frozen_capsule_v1"
PLAN_SCHEMA = "vision_selection_execution_manifest_v1"
REFERENCE_FIELDS = frozenset({"path", "sha256", "status"})
PHASE_CAPS = {"comparison": 72, "e2e": 32, "heldout": 10}
MAX_ROWS = 24
MAX_CALLS = 3
_IDENTIFIER = re.compile(r"[a-z0-9][a-z0-9._-]{0,79}")
_HEX40 = re.compile(r"[0-9a-f]{40}")
_HEX64 = re.compile(r"[0-9a-f]{64}")
_METHODS = (
    "evals/vision_selection_contract_v1.py",
    "evals/vision_selection_loop_v1.py",
    "evals/vision_selection_call_v1.py",
)


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode("utf-8")


def _read_canonical(path: Path) -> dict[str, Any]:
    target = Path(path)
    if (not target.is_absolute() or target != target.resolve(strict=True) or target.is_symlink()
            or any(parent.is_symlink() for parent in target.parents)):
        raise ValueError("동결 파일은 canonical regular path여야 한다")
    metadata = target.stat(follow_symlinks=False)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise ValueError("동결 파일은 단일 regular file이어야 한다")
    value = json.loads(target.read_text(encoding="utf-8"), parse_constant=lambda raw: (_ for _ in ()).throw(
        ValueError(f"JSON constant 거부: {raw}")))
    if not isinstance(value, dict) or target.read_bytes() != _canonical(value):
        raise ValueError("동결 파일 canonical bytes가 다르다")
    return value


def _write_new(path: Path, value: dict[str, Any]) -> None:
    target = Path(path)
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    existing = next((parent for parent in target.parents if parent.exists()), None)
    if existing is None or existing.is_symlink() or any(parent.is_symlink() for parent in existing.parents):
        raise ValueError("동결 출력 parent symlink를 거부한다")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.parent != target.parent.resolve(strict=True):
        raise ValueError("동결 출력 parent가 canonical path가 아니다")
    descriptor = target.open("xb")
    try:
        descriptor.write(_canonical(value))
    finally:
        descriptor.close()


def _reference(value: object, status: str) -> tuple[Path, dict[str, str]]:
    if not isinstance(value, dict) or set(value) != REFERENCE_FIELDS:
        raise TypeError("path·sha256·status reference가 필요하다")
    if value.get("status") != status or _HEX64.fullmatch(str(value.get("sha256"))) is None:
        raise ValueError("reference status 또는 SHA 형식이 다르다")
    path = Path(str(value.get("path")))
    if not path.is_file() or path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
        raise ValueError("reference path가 canonical regular file이 아니다")
    if JOURNAL.file_sha(path) != value["sha256"]:
        raise ValueError("reference SHA가 현재 파일과 다르다")
    _read_canonical(path)
    return path, {"path": str(path), "sha256": value["sha256"], "status": status}


def freeze(snapshot: dict, vision_contract: Path, source_binding: dict, view_kind: str,
           output_folder: Path) -> dict[str, str]:
    """live 입력을 한 번 검증해 self-contained bundle/messages/capsule로 동결한다."""
    folder = Path(output_folder).absolute()
    if folder.exists() or folder.is_symlink():
        raise FileExistsError(folder)
    bundle = CONTRACT.prepare(snapshot, vision_contract, source_binding, view_kind)
    messages = CONTRACT.messages(bundle, snapshot, vision_contract, source_binding)
    methods = {name: JOURNAL.file_sha(ROOT / name) for name in _METHODS}
    _write_new(folder / "bundle.json", bundle)
    _write_new(folder / "messages.json", {"messages": messages})
    capsule = {
        "schema": CAPSULE_SCHEMA,
        "bundle_sha256": JOURNAL.file_sha(folder / "bundle.json"),
        "messages_sha256": JOURNAL.file_sha(folder / "messages.json"),
        "method_sha256": methods,
    }
    _write_new(folder / "capsule.json", capsule)
    capsule_path = (folder / "capsule.json").resolve(strict=True)
    return {"path": str(capsule_path), "sha256": JOURNAL.file_sha(capsule_path), "status": "frozen"}


def prepared(reference: dict) -> dict[str, Any]:
    """live 원장이나 V01을 읽지 않고 등록 capsule의 동결 파일만 검증한다."""
    capsule_path, checked = _reference(reference, "frozen")
    capsule = _read_canonical(capsule_path)
    if set(capsule) != {"schema", "bundle_sha256", "messages_sha256", "method_sha256"} \
            or capsule.get("schema") != CAPSULE_SCHEMA:
        raise ValueError("동결 capsule 형식이 다르다")
    methods = capsule.get("method_sha256")
    if (not isinstance(methods, dict) or set(methods) != set(_METHODS)
            or any(_HEX64.fullmatch(str(value)) is None for value in methods.values())):
        raise ValueError("동결 method SHA가 다르다")
    bundle_path, messages_path = capsule_path.parent / "bundle.json", capsule_path.parent / "messages.json"
    bundle, message_box = _read_canonical(bundle_path), _read_canonical(messages_path)
    if (JOURNAL.file_sha(bundle_path) != capsule.get("bundle_sha256")
            or JOURNAL.file_sha(messages_path) != capsule.get("messages_sha256")
            or set(message_box) != {"messages"} or not isinstance(message_box["messages"], list)):
        raise ValueError("동결 bundle/messages SHA가 다르다")
    if (bundle.get("schema") != CONTRACT.SCHEMA or bundle.get("view_kind") not in CONTRACT.VIEW_KINDS):
        raise ValueError("동결 selection bundle 형식이 다르다")
    expected_images = 0 if bundle["view_kind"] == "text_only" else 6
    images = sum(
        part.get("type") == "image_url"
        for message in message_box["messages"] if isinstance(message, dict)
        for part in message.get("content", []) if isinstance(message.get("content"), list)
        if isinstance(part, dict)
    )
    if images != expected_images:
        raise ValueError("동결 messages 이미지 수가 view와 다르다")
    checkpoint = bundle.get("operational_base", {}).get("case_checkpoint_sha256")
    if _HEX64.fullmatch(str(checkpoint)) is None:
        raise ValueError("동결 case checkpoint가 없다")
    return {
        "reference": checked,
        "capsule": capsule,
        "bundle": bundle,
        "messages": deepcopy(message_box["messages"]),
        "view_kind": bundle["view_kind"],
        "checkpoint_sha256": checkpoint,
    }


def _feedback(code: str) -> dict[str, str]:
    return {"role": "user", "content": "\n".join((
        "The previous JSON was rejected.", f"- {code}",
        "Return one corrected JSON object with exactly action, probe, and fact_ids.",
    ))}


def engine(frozen: dict[str, Any], complete: Callable[[list[dict], dict], dict]) -> dict[str, Any]:
    """동결 messages에서 논리 최대 세 호출로 모델 선택만 검증한다."""
    if not callable(complete):
        raise TypeError("complete callback이 필요하다")
    base = deepcopy(frozen["messages"])
    sent = base
    calls, infra_used, feedback_used = [], False, False
    for attempt in range(1, MAX_CALLS + 1):
        result = OBSERVER._safe_complete(complete, sent)
        transport = OBSERVER._transport(result)
        row = {"attempt": attempt, "transport": transport, "result": result}
        calls.append(row)
        if transport == "infra_error" and not infra_used and attempt < MAX_CALLS:
            infra_used = True
            continue
        if transport != "completed":
            return {"accepted": False, "final_content": result.get("content"),
                    "model_status": "invalid", "hold_reason": f"transport_{transport}",
                    "selection": {"action": "hold", "probe": None, "fact_ids": []}, "calls": calls}
        content = result.get("content")
        try:
            decision = CONTRACT.validate_choice(frozen["bundle"], content)
        except CONTRACT.ChoiceError as error:
            code = str(error)
            row["validation"] = {"status": "invalid", "code": code}
            if not feedback_used and attempt < MAX_CALLS:
                feedback_used = True
                sent = [*base, {"role": "assistant", "content": content}, _feedback(code)]
                continue
            return {"accepted": False, "final_content": content, "model_status": "invalid",
                    "hold_reason": "selection_invalid",
                    "selection": {"action": "hold", "probe": None, "fact_ids": []}, "calls": calls}
        row["validation"] = {"status": "valid"}
        hold = decision["action"] == "hold"
        return {"accepted": True, "final_content": content, "model_status": "valid",
                "hold_reason": "model_hold" if hold else None,
                "selection": decision, "calls": calls}
    raise AssertionError("논리 호출 상한 분기가 닫히지 않았다")


def initialize(repo_root: Path, execution_id: str, phase: str, source_anchor: str,
               protocol_sha256: str, rows: list[dict[str, Any]]) -> dict[str, str]:
    """고정 순서 capsule 목록과 phase 전역 상한을 canonical stage에 봉인한다."""
    repository = Path(repo_root).resolve(strict=True)
    if (_IDENTIFIER.fullmatch(execution_id) is None or phase not in PHASE_CAPS
            or _HEX40.fullmatch(source_anchor) is None or _HEX64.fullmatch(protocol_sha256) is None
            or not isinstance(rows, list) or not 1 <= len(rows) <= MAX_ROWS):
        raise ValueError("selection execution plan 기본 필드가 다르다")
    checked_rows = []
    seen = set()
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict) or set(row) != {"call_id", "run_id", "capsule"}:
            raise ValueError("selection plan row 형식이 다르다")
        call_id = row["call_id"]
        if (_IDENTIFIER.fullmatch(str(call_id)) is None or call_id in seen
                or row["run_id"] != f"vision-selection-{execution_id}-{call_id}"):
            raise ValueError("selection plan call_id/run_id가 다르다")
        seen.add(call_id)
        frozen = prepared(row["capsule"])
        capsule_path = Path(frozen["reference"]["path"])
        try:
            relative = capsule_path.relative_to(repository).as_posix()
        except ValueError as error:
            raise ValueError("동결 capsule은 저장소 안에 있어야 한다") from error
        checked_rows.append({
            "ordinal": index, "call_id": call_id, "run_id": row["run_id"],
            "capsule_path": relative, "capsule_sha256": frozen["reference"]["sha256"],
            "view_kind": frozen["view_kind"], "checkpoint_sha256": frozen["checkpoint_sha256"],
        })
    stage = repository / ".triage-runtime" / "vision-selection" / execution_id
    manifest = {
        "schema": PLAN_SCHEMA, "execution_id": execution_id, "phase": phase,
        "api_cap": PHASE_CAPS[phase], "source_anchor": source_anchor,
        "protocol_sha256": protocol_sha256, "request_params": deepcopy(OBSERVER.REQUEST_PARAMS),
        "rows": checked_rows,
    }
    path = stage / "manifest.json"
    if path.exists():
        if _read_canonical(path) != manifest:
            raise JOURNAL.JournalIntegrityError("기존 selection plan이 현재 입력과 다르다")
    else:
        _write_new(path, manifest)
    seal = {"schema": "vision_selection_execution_seal_v1",
            "manifest_sha256": JOURNAL.file_sha(path)}
    seal_path = stage / "seal.json"
    if seal_path.exists():
        if _read_canonical(seal_path) != seal:
            raise JOURNAL.JournalIntegrityError("기존 selection plan seal이 다르다")
    else:
        _write_new(seal_path, seal)
    manifest_path = path.resolve(strict=True)
    return {"path": str(manifest_path), "sha256": JOURNAL.file_sha(manifest_path),
            "status": "planned"}


def _plan(reference: dict) -> tuple[Path, Path, dict[str, Any]]:
    manifest_path, _ = _reference(reference, "planned")
    root = manifest_path.parent
    manifest = _read_canonical(manifest_path)
    seal = _read_canonical(root / "seal.json")
    if seal != {"schema": "vision_selection_execution_seal_v1",
                "manifest_sha256": JOURNAL.file_sha(manifest_path)}:
        raise JOURNAL.JournalIntegrityError("selection plan seal 검증이 실패했다")
    if manifest.get("schema") != PLAN_SCHEMA or _IDENTIFIER.fullmatch(str(manifest.get("execution_id"))) is None:
        raise ValueError("selection execution manifest가 다르다")
    repository = root.parents[2]
    expected = repository / ".triage-runtime" / "vision-selection" / manifest["execution_id"]
    if root != expected.resolve(strict=True):
        raise ValueError("selection stage가 canonical runtime 경로가 아니다")
    if (manifest.get("phase") not in PHASE_CAPS
            or manifest.get("api_cap") != PHASE_CAPS[manifest["phase"]]
            or manifest.get("request_params") != OBSERVER.REQUEST_PARAMS):
        raise ValueError("selection phase/API 계약이 다르다")
    return repository, root, manifest


def _row(repository: Path, manifest: dict[str, Any], call_id: str) -> tuple[int, dict, dict]:
    matches = [(index, row) for index, row in enumerate(manifest["rows"])
               if row.get("call_id") == call_id]
    if len(matches) != 1:
        raise ValueError("plan에 등록되지 않은 call_id다")
    index, row = matches[0]
    capsule_path = repository / row["capsule_path"]
    reference = {"path": str(capsule_path.resolve(strict=True)),
                 "sha256": row["capsule_sha256"], "status": "frozen"}
    frozen = prepared(reference)
    if (frozen["view_kind"] != row["view_kind"]
            or frozen["checkpoint_sha256"] != row["checkpoint_sha256"]):
        raise JOURNAL.JournalIntegrityError("plan row와 capsule 결합이 다르다")
    return index, row, frozen


def _journal(stage: Path, manifest: dict[str, Any], row: dict, frozen: dict) -> JOURNAL.DecisionJournal:
    binding = {
        "run_id": row["run_id"], "source_anchor": manifest["source_anchor"],
        "protocol_sha256": JOURNAL.file_sha(Path(stage) / "manifest.json"),
        "checkpoint_sha256": row["checkpoint_sha256"],
        "snapshot_sha256": row["capsule_sha256"],
        "initial_messages_sha256": JOURNAL.digest(frozen["messages"]),
    }
    return JOURNAL.DecisionJournal(
        Path(stage) / "decisions" / row["call_id"], Path(stage) / "api",
        binding, OBSERVER.REQUEST_PARAMS, retry_delay_s=30, api_cap=manifest["api_cap"],
    )


def _replay(stage: Path, manifest: dict[str, Any], row: dict, frozen: dict) -> dict[str, Any]:
    decision = Path(stage) / "decisions" / row["call_id"]
    api = Path(stage) / "api"
    for required in (decision / "manifest.json", decision / "terminal.json", api / "manifest.json"):
        if not required.is_file():
            raise ValueError("앞 selection call이 terminal 상태가 아니다")
    journal = _journal(stage, manifest, row, frozen)
    journal._verify_prepared()
    if journal._audit_api()["pending"]:
        raise JOURNAL.JournalIntegrityError("전역 API pending 상태에서는 replay하지 않는다")
    result = journal.advance(
        lambda completion: engine(frozen, completion),
        lambda *_: (_ for _ in ()).throw(AssertionError("terminal replay provider 호출")),
        lambda: datetime.now(timezone.utc),  # noqa: UP017
    )
    if result["status"] not in {"choice_ready", "terminal_hold"}:
        raise JOURNAL.JournalIntegrityError("저장 terminal replay가 완료 상태가 아니다")
    return result


def run(plan_reference: dict, call_id: str,
        complete: Callable[[list[dict], dict], dict]) -> dict[str, Any]:
    """고정 순서를 확인하고 한 call의 DecisionJournal을 advance한다."""
    repository, stage, manifest = _plan(plan_reference)
    index, row, frozen = _row(repository, manifest, call_id)
    for prior in manifest["rows"][:index]:
        _, _, prior_frozen = _row(repository, manifest, prior["call_id"])
        _replay(stage, manifest, prior, prior_frozen)
    journal = _journal(stage, manifest, row, frozen)
    result = journal.advance(
        lambda completion: engine(frozen, completion), complete,
        lambda: datetime.now(timezone.utc),  # noqa: UP017
    )
    return {"call_id": call_id, "journal": result,
            "capsule_sha256": row["capsule_sha256"], "api_cap": manifest["api_cap"]}


def activate(plan_reference: dict, call_id: str, live_snapshot: dict, live_contract: Path,
             source_binding: dict, bridge_path: Path) -> dict[str, Any]:
    """terminal 선택이 현재 live 상태와 같을 때만 기존 bridge와 host request를 만든다."""
    repository, stage, manifest = _plan(plan_reference)
    index, row, frozen = _row(repository, manifest, call_id)
    for prior in manifest["rows"][:index]:
        _, _, prior_frozen = _row(repository, manifest, prior["call_id"])
        _replay(stage, manifest, prior, prior_frozen)
    replay = _replay(stage, manifest, row, frozen)
    bundle = CONTRACT.verify(
        frozen["bundle"], live_snapshot, live_contract, source_binding,
    )
    live_messages = CONTRACT.messages(bundle, live_snapshot, live_contract, source_binding)
    if _canonical({"messages": live_messages}) != _canonical({"messages": frozen["messages"]}):
        raise ValueError("동결 messages와 현재 live selection 상태가 다르다")
    engine_result = replay["engine_result"]
    model_content = engine_result.get("final_content") if isinstance(engine_result, dict) else None
    reference = LOOP.seal_bridge(
        bridge_path, live_snapshot, live_contract, source_binding,
        frozen["view_kind"], model_content,
    )
    bridge = LOOP.verify_bridge(reference, live_snapshot, live_contract, source_binding)
    host = LOOP.host_request_spec(reference, live_snapshot, live_contract, source_binding)
    return {"call_id": call_id, "capsule_sha256": row["capsule_sha256"],
            "terminal_sha256": JOURNAL.file_sha(Path(stage) / "decisions" / call_id / "terminal.json"),
            "bridge_reference": reference, "bridge": bridge, "host": host,
            "approval_granted": False, "rule_fallback_used": False}
