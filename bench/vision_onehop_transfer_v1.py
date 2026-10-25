"""한 번의 Brev exec 안에서 자원 freshness와 등록 stage 실행을 결합한다."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import re
import stat
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, BinaryIO

UTC = timezone.utc  # noqa: UP017 - Python 3.10 호환
PACKET_SCHEMA = "vision_onehop_transfer_packet_v1"
RESULT_SCHEMA = "vision_onehop_transfer_result_v1"
TRANSFER_SCHEMA = "vision_onehop_exec_transfer_v1"
SENTINEL = "RL_TRIAGE_ONEHOP_V1:"
PACKET_PLACEHOLDER = b"__RL_TRIAGE_ONEHOP_PACKET_B64__"
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
RESOURCE_KEYS = {
    "build_status",
    "gpu",
    "health_status",
    "id",
    "instance_kind",
    "instance_type",
    "name",
    "shell_status",
    "status",
}


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _b64(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def _decode_b64(value: Any, label: str) -> bytes:
    if not isinstance(value, str):
        raise TypeError(f"{label} base64가 문자열이 아니다")
    try:
        decoded = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error) as error:
        raise ValueError(f"{label} base64가 잘못됐다") from error
    if _b64(decoded) != value:
        raise ValueError(f"{label} base64가 canonical 형식이 아니다")
    return decoded


def _utc(value: Any, label: str) -> datetime:
    if not isinstance(value, str):
        raise TypeError(f"{label} 시각이 문자열이 아니다")
    try:
        normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
        parsed = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise ValueError(f"{label} 시각이 잘못됐다") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{label} 시각에 timezone이 없다")
    return parsed.astimezone(UTC)


def _token(value: Any, label: str) -> str:
    if not isinstance(value, str) or TOKEN.fullmatch(value) is None:
        raise ValueError(f"{label} 식별자가 잘못됐다")
    return value


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or HEX64.fullmatch(value) is None:
        raise ValueError(f"{label} SHA-256이 잘못됐다")
    return value


def _inventory(raw: bytes) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Brev ls 원문 JSON이 잘못됐다") from error
    if (
        not isinstance(value, dict)
        or set(value) != {"workspaces"}
        or not isinstance(value["workspaces"], list)
    ):
        raise ValueError("Brev ls 원문 schema가 다르다")
    if any(
        not isinstance(row, dict) or set(row) != RESOURCE_KEYS
        for row in value["workspaces"]
    ):
        raise ValueError("Brev workspace 행 schema가 다르다")
    return value


def _validate_target(inventory: Mapping[str, Any], target: Mapping[str, Any]) -> None:
    if set(target) != {"id", "name", "instance_type"}:
        raise ValueError("target 필드가 다르다")
    expected = {key: _token(target[key], f"target {key}") for key in target}
    rows = inventory["workspaces"]
    by_id = [row for row in rows if row["id"] == expected["id"]]
    by_name = [row for row in rows if row["name"] == expected["name"]]
    if (
        len(by_id) != 1
        or len(by_name) != 1
        or by_id[0] != by_name[0]
        or by_id[0]["instance_type"] != expected["instance_type"]
        or by_id[0]["status"] != "RUNNING"
    ):
        raise ValueError("등록한 Brev 자원이 유일한 exact-owned RUNNING 상태가 아니다")


def build_packet(
    *,
    native_receipt: Mapping[str, Any],
    raw_stdout: bytes,
    raw_stderr: bytes,
    source_binding: Mapping[str, Any],
    protocol_binding: Mapping[str, Any],
    target: Mapping[str, Any],
    attempt_id: str,
    stage_id: str,
    stage_bytes: bytes,
) -> dict[str, Any]:
    """호스트 native receipt와 stage payload를 canonical packet으로 묶는다."""
    receipt_keys = {
        "command",
        "request_before_utc",
        "response_after_utc",
        "exit_code",
        "stdout_path",
        "stdout_sha256",
        "stderr_path",
        "stderr_sha256",
    }
    if (
        set(native_receipt) != receipt_keys
        or native_receipt["command"] != ["brev", "ls", "--json"]
        or type(native_receipt["exit_code"]) is not int
        or native_receipt["exit_code"] != 0
    ):
        raise ValueError("Brev ls native receipt가 다르다")
    requested = _utc(native_receipt["request_before_utc"], "native request")
    responded = _utc(native_receipt["response_after_utc"], "native response")
    if responded < requested:
        raise ValueError("Brev ls native receipt 시각이 역전됐다")
    if (
        not isinstance(native_receipt["stdout_path"], str)
        or not native_receipt["stdout_path"]
        or not isinstance(native_receipt["stderr_path"], str)
        or not native_receipt["stderr_path"]
    ):
        raise ValueError("Brev ls native stream ref가 다르다")
    if sha256_bytes(raw_stdout) != _sha(
        native_receipt["stdout_sha256"], "native stdout"
    ):
        raise ValueError("Brev ls native stdout SHA-256이 다르다")
    if (
        sha256_bytes(raw_stderr)
        != _sha(native_receipt["stderr_sha256"], "native stderr")
        or raw_stderr
    ):
        raise ValueError("Brev ls native stderr가 비어 있지 않거나 SHA가 다르다")
    _inventory(raw_stdout)
    if not isinstance(stage_bytes, bytes) or not stage_bytes:
        raise ValueError("stage payload가 비어 있다")
    packet = {
        "schema": PACKET_SCHEMA,
        "attempt_id": _token(attempt_id, "attempt"),
        "source": dict(source_binding),
        "protocol": dict(protocol_binding),
        "target": dict(target),
        "native": {
            "receipt": dict(native_receipt),
            "raw_stdout_b64": _b64(raw_stdout),
            "raw_stderr_b64": _b64(raw_stderr),
        },
        "query": {
            "response_after_utc": native_receipt["response_after_utc"],
            "expires_at_utc": (responded + timedelta(seconds=30)).isoformat(),
        },
        "stage": {
            "id": _token(stage_id, "stage"),
            "bytes_b64": _b64(stage_bytes),
            "sha256": sha256_bytes(stage_bytes),
        },
    }
    _validate_packet(packet)
    return packet


def encode_packet(packet: Mapping[str, Any]) -> str:
    _validate_packet(packet)
    body = canonical_json(dict(packet))
    return _b64(
        canonical_json({"packet": dict(packet), "packet_sha256": sha256_bytes(body)})
    )


def decode_packet(encoded_packet: str) -> tuple[dict[str, Any], str]:
    try:
        envelope = json.loads(_decode_b64(encoded_packet, "packet envelope"))
    except json.JSONDecodeError as error:
        raise ValueError("packet envelope JSON이 잘못됐다") from error
    if (
        not isinstance(envelope, dict)
        or set(envelope) != {"packet", "packet_sha256"}
        or not isinstance(envelope["packet"], dict)
    ):
        raise ValueError("packet envelope schema가 다르다")
    packet = envelope["packet"]
    expected = _sha(envelope["packet_sha256"], "packet")
    if sha256_bytes(canonical_json(packet)) != expected:
        raise ValueError("packet SHA-256이 다르다")
    _validate_packet(packet)
    return packet, expected


def _validate_packet(packet: Mapping[str, Any]) -> None:
    if (
        set(packet)
        != {
            "schema",
            "attempt_id",
            "source",
            "protocol",
            "target",
            "native",
            "query",
            "stage",
        }
        or packet.get("schema") != PACKET_SCHEMA
    ):
        raise ValueError("one-hop packet schema가 다르다")
    _token(packet["attempt_id"], "attempt")
    for key in ("source", "protocol"):
        value = packet[key]
        if not isinstance(value, dict) or not value:
            raise ValueError(f"{key} binding이 비어 있다")
    target = packet["target"]
    if not isinstance(target, dict):
        raise TypeError("target이 object가 아니다")
    native = packet["native"]
    expected_native = {"receipt", "raw_stdout_b64", "raw_stderr_b64"}
    if not isinstance(native, dict) or set(native) != expected_native:
        raise ValueError("packet native receipt가 다르다")
    receipt = native["receipt"]
    receipt_keys = {
        "command",
        "request_before_utc",
        "response_after_utc",
        "exit_code",
        "stdout_path",
        "stdout_sha256",
        "stderr_path",
        "stderr_sha256",
    }
    if (
        not isinstance(receipt, dict)
        or set(receipt) != receipt_keys
        or receipt["command"] != ["brev", "ls", "--json"]
        or type(receipt["exit_code"]) is not int
        or receipt["exit_code"] != 0
    ):
        raise ValueError("packet native receipt가 다르다")
    raw = _decode_b64(native["raw_stdout_b64"], "native stdout")
    stderr = _decode_b64(native["raw_stderr_b64"], "native stderr")
    if sha256_bytes(raw) != _sha(receipt["stdout_sha256"], "native stdout"):
        raise ValueError("native stdout SHA-256이 다르다")
    if (
        sha256_bytes(stderr) != _sha(receipt["stderr_sha256"], "native stderr")
        or stderr
    ):
        raise ValueError("native stderr가 비어 있지 않거나 SHA-256이 다르다")
    if (
        not isinstance(receipt["stdout_path"], str)
        or not receipt["stdout_path"]
        or not isinstance(receipt["stderr_path"], str)
        or not receipt["stderr_path"]
    ):
        raise ValueError("native stream ref가 다르다")
    requested = _utc(receipt["request_before_utc"], "native request")
    responded_native = _utc(receipt["response_after_utc"], "native response")
    if responded_native < requested:
        raise ValueError("native receipt 시각이 역전됐다")
    inventory = _inventory(raw)
    _validate_target(inventory, target)
    query = packet["query"]
    if (
        not isinstance(query, dict)
        or set(query) != {"response_after_utc", "expires_at_utc"}
        or query["response_after_utc"] != receipt["response_after_utc"]
    ):
        raise ValueError("freshness query binding이 다르다")
    responded = _utc(query["response_after_utc"], "query response")
    expires = _utc(query["expires_at_utc"], "query expiry")
    if expires - responded != timedelta(seconds=30):
        raise ValueError("freshness 창이 30초가 아니다")
    stage = packet["stage"]
    if not isinstance(stage, dict) or set(stage) != {"id", "bytes_b64", "sha256"}:
        raise ValueError("stage binding이 다르다")
    _token(stage["id"], "stage")
    payload = _decode_b64(stage["bytes_b64"], "stage")
    if sha256_bytes(payload) != _sha(stage["sha256"], "stage"):
        raise ValueError("stage payload SHA-256이 다르다")


def _write_new(path: Path, value: bytes) -> None:
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as error:
        raise ValueError("write-once state를 안전하게 만들 수 없다") from error
    with os.fdopen(descriptor, "wb") as stream:
        metadata = os.fstat(stream.fileno())
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ValueError("write-once state가 단일 regular file이 아니다")
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())


@contextmanager
def _kernel_lock(path: Path) -> Iterator[None]:
    if path.is_symlink():
        raise ValueError("attempt lock이 symlink다")
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as error:
        raise ValueError("attempt lock을 안전하게 열 수 없다") from error
    with os.fdopen(descriptor, "a+b") as stream:
        metadata = os.fstat(stream.fileno())
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ValueError("attempt lock이 단일 regular file이 아니다")
        _lock(stream)
        try:
            yield
        finally:
            _unlock(stream)


def _lock(stream: BinaryIO) -> None:
    if os.name == "nt":
        import msvcrt

        stream.seek(0)
        if stream.read(1) == b"":
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
    else:
        import fcntl

        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)


def _unlock(stream: BinaryIO) -> None:
    if os.name == "nt":
        import msvcrt

        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _read_canonical(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink():
        raise ValueError(f"{label}가 symlink다")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"{label}를 안전하게 열 수 없다") from error
    with os.fdopen(descriptor, "rb") as stream:
        metadata = os.fstat(stream.fileno())
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ValueError(f"{label}가 단일 regular file이 아니다")
        raw = stream.read()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} JSON이 잘못됐다") from error
    if not isinstance(value, dict) or canonical_json(value) + b"\n" != raw:
        raise ValueError(f"{label}가 canonical JSON이 아니다")
    return value


def _safe_attempt_directory(state_dir: Path, attempt_id: str) -> Path:
    state_dir = Path(state_dir).absolute()
    for candidate in reversed((state_dir, *state_dir.parents)):
        if candidate.is_symlink():
            raise ValueError("state_dir lexical parent가 symlink다")
    state_dir.mkdir(parents=True, exist_ok=True)
    if state_dir.is_symlink() or not state_dir.is_dir():
        raise ValueError("state_dir가 실제 directory가 아니다")
    attempt = state_dir / attempt_id
    if attempt.is_symlink() or (attempt.exists() and not attempt.is_dir()):
        raise ValueError("attempt state 경로가 실제 directory가 아니다")
    attempt.mkdir(exist_ok=True)
    if attempt.is_symlink() or not attempt.is_dir():
        raise ValueError("attempt state 경로가 실제 directory가 아니다")
    return attempt


def _validate_result(
    result: Mapping[str, Any],
    *,
    attempt_id: str,
    packet_sha256: str,
    payload_sha256: str,
    dry_run: bool,
) -> None:
    common = {
        "schema": RESULT_SCHEMA,
        "attempt_id": attempt_id,
        "packet_sha256": packet_sha256,
        "payload_sha256": payload_sha256,
        "dry_run": dry_run,
    }
    if any(result.get(key) != value for key, value in common.items()):
        raise ValueError("attempt result binding이 다르다")
    if result.get("status") == "requires_reconcile":
        if (
            set(result) != {*common, "status", "stage_executed"}
            or result.get("stage_executed") is not False
        ):
            raise ValueError("reconcile result schema가 다르다")
        return
    expected_keys = {
        *common,
        "status",
        "stage_executed",
        "stage_result",
        "result_sha256",
    }
    if (
        set(result) != expected_keys
        or result.get("status") != "complete"
        or result.get("stage_executed") is not (not dry_run)
    ):
        raise ValueError("complete result schema가 다르다")
    core = {key: value for key, value in result.items() if key != "result_sha256"}
    if result.get("result_sha256") != sha256_bytes(canonical_json(core)):
        raise ValueError("attempt result SHA-256이 다르다")


def execute_packet(
    encoded_packet: str,
    *,
    state_dir: Path,
    expected_target: Mapping[str, Any],
    expected_source_binding: Mapping[str, Any],
    expected_protocol_binding: Mapping[str, Any],
    expected_stage_id: str,
    registered_stage_sha256: str,
    run_stage: Callable[[bytes, Mapping[str, Any]], Any],
    now_utc: datetime | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """검증과 stage 실행을 한 프로세스에서 이어가며 attempt를 at-least-once로 봉인한다."""
    packet, packet_sha = decode_packet(encoded_packet)
    if packet["target"] != dict(expected_target):
        raise ValueError("target이 외부 등록값과 다르다")
    if packet["source"] != dict(expected_source_binding):
        raise ValueError("source binding이 외부 등록값과 다르다")
    if packet["protocol"] != dict(expected_protocol_binding):
        raise ValueError("protocol binding이 외부 등록값과 다르다")
    stage = packet["stage"]
    payload = _decode_b64(stage["bytes_b64"], "stage")
    if stage["id"] != _token(expected_stage_id, "expected stage") or stage[
        "sha256"
    ] != _sha(registered_stage_sha256, "registered stage"):
        raise ValueError("stage가 외부 등록값과 다르다")
    attempt = _safe_attempt_directory(state_dir, packet["attempt_id"])
    intent_path = attempt / "intent.json"
    result_path = attempt / "result.json"
    with _kernel_lock(attempt / "attempt.lock"):
        intent = {
            "schema": "vision_onehop_intent_v1",
            "attempt_id": packet["attempt_id"],
            "packet_sha256": packet_sha,
            "payload_sha256": stage["sha256"],
            "dry_run": bool(dry_run),
        }
        if intent_path.exists():
            stored = _read_canonical(intent_path, "attempt intent")
            if stored != intent:
                raise ValueError("attempt payload 또는 packet binding이 다르다")
            if result_path.exists():
                result = _read_canonical(result_path, "attempt result")
                _validate_result(
                    result,
                    attempt_id=packet["attempt_id"],
                    packet_sha256=packet_sha,
                    payload_sha256=stage["sha256"],
                    dry_run=bool(dry_run),
                )
                return result
            return {
                "schema": RESULT_SCHEMA,
                "status": "requires_reconcile",
                "attempt_id": packet["attempt_id"],
                "packet_sha256": packet_sha,
                "payload_sha256": stage["sha256"],
                "dry_run": bool(dry_run),
                "stage_executed": False,
            }
        now = (now_utc or datetime.now(UTC)).astimezone(UTC)
        responded = _utc(packet["query"]["response_after_utc"], "query response")
        expires = _utc(packet["query"]["expires_at_utc"], "query expiry")
        if not responded <= now <= expires:
            raise ValueError("Brev ls freshness 창 밖이다")
        _write_new(intent_path, canonical_json(intent) + b"\n")
        stage_result = None if dry_run else run_stage(payload, packet)
        core = {
            "schema": RESULT_SCHEMA,
            "status": "complete",
            "attempt_id": packet["attempt_id"],
            "packet_sha256": packet_sha,
            "payload_sha256": stage["sha256"],
            "dry_run": bool(dry_run),
            "stage_executed": not dry_run,
            "stage_result": stage_result,
        }
        result = {**core, "result_sha256": sha256_bytes(canonical_json(core))}
        _write_new(result_path, canonical_json(result) + b"\n")
        return result


def marker_line(result: Mapping[str, Any]) -> str:
    return SENTINEL + _b64(canonical_json(dict(result)))


def render_exec_script(
    template_bytes: bytes,
    encoded_packet: str,
    *,
    expected_template_sha256: str,
) -> bytes:
    """외부 등록 template의 유일한 자리표시자를 canonical packet literal로 치환한다."""
    if sha256_bytes(template_bytes) != _sha(
        expected_template_sha256, "expected template"
    ):
        raise ValueError("immutable @FILE template SHA가 다르다")
    if template_bytes.count(PACKET_PLACEHOLDER) != 1:
        raise ValueError("immutable @FILE template packet placeholder가 유일하지 않다")
    try:
        packet_literal = encoded_packet.encode("ascii")
    except UnicodeEncodeError as error:
        raise ValueError("packet literal이 ASCII가 아니다") from error
    decode_packet(encoded_packet)
    return template_bytes.replace(PACKET_PLACEHOLDER, packet_literal)


def parse_native_stdout(
    stdout: str,
    *,
    expected_target_name: str,
    expected_attempt_id: str,
    expected_packet_sha256: str,
    expected_payload_sha256: str,
) -> dict[str, Any]:
    """Brev가 붙이는 정확한 NAME footer 외의 출력을 모두 거절한다."""
    if not stdout.endswith("\n"):
        raise ValueError("native exec stdout 끝 newline이 없다")
    lines = stdout.splitlines()
    if (
        len(lines) != 2
        or lines[1] != expected_target_name
        or not lines[0].startswith(SENTINEL)
    ):
        raise ValueError("native exec stdout sentinel 또는 NAME footer가 다르다")
    encoded = lines[0][len(SENTINEL) :]
    try:
        result = json.loads(_decode_b64(encoded, "result marker"))
    except json.JSONDecodeError as error:
        raise ValueError("result marker JSON이 잘못됐다") from error
    if (
        not isinstance(result, dict)
        or result.get("schema") != RESULT_SCHEMA
        or result.get("attempt_id") != expected_attempt_id
        or result.get("packet_sha256")
        != _sha(expected_packet_sha256, "expected packet")
    ):
        raise ValueError("result marker binding이 다르다")
    _validate_result(
        result,
        attempt_id=expected_attempt_id,
        packet_sha256=expected_packet_sha256,
        payload_sha256=_sha(expected_payload_sha256, "expected payload"),
        dry_run=result.get("dry_run") is True,
    )
    return result


def build_exec_transfer_envelope(
    *,
    attempt_id: str,
    target_name: str,
    native_exec_receipt_bytes: bytes,
    native_exec_stdout_bytes: bytes,
    native_exec_stderr_bytes: bytes,
    intent_bytes: bytes,
    template_bytes: bytes,
    expected_template_sha256: str,
    script_bytes: bytes,
    payload_bytes: bytes,
    expected_packet_sha256: str,
    expected_payload_sha256: str,
    artifact_readback_bytes: bytes,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    """native exec 운반 증거를 copy receipt와 구분된 envelope로 묶는다."""
    attempt_id = _token(attempt_id, "attempt")
    target_name = _token(target_name, "target name")
    try:
        receipt = json.loads(native_exec_receipt_bytes)
        intent = json.loads(intent_bytes)
        readback = json.loads(artifact_readback_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("exec transfer JSON 원문이 잘못됐다") from error
    receipt_keys = {
        "command",
        "request_before_utc",
        "response_after_utc",
        "exit_code",
        "stdout_path",
        "stdout_sha256",
        "stderr_path",
        "stderr_sha256",
    }
    if (
        not isinstance(receipt, dict)
        or set(receipt) != receipt_keys
        or type(receipt["exit_code"]) is not int
        or receipt["exit_code"] != 0
    ):
        raise ValueError("native exec receipt schema 또는 exit가 다르다")
    if (
        sha256_bytes(native_exec_stdout_bytes)
        != _sha(receipt["stdout_sha256"], "exec stdout")
        or sha256_bytes(native_exec_stderr_bytes)
        != _sha(receipt["stderr_sha256"], "exec stderr")
        or native_exec_stderr_bytes
    ):
        raise ValueError("native exec stream SHA 또는 stderr가 다르다")
    if (
        not isinstance(receipt["stdout_path"], str)
        or not receipt["stdout_path"]
        or not isinstance(receipt["stderr_path"], str)
        or not receipt["stderr_path"]
    ):
        raise ValueError("native exec stream ref가 다르다")
    requested = _utc(receipt["request_before_utc"], "exec request")
    responded = _utc(receipt["response_after_utc"], "exec response")
    if responded < requested:
        raise ValueError("native exec receipt 시각이 역전됐다")
    intent_keys = {"schema", "source_path", "source_sha256", "snapshot_path", "command"}
    if (
        not isinstance(intent, dict)
        or set(intent) != intent_keys
        or intent["schema"] != "vision_atomic_exec_script_intent_v1"
        or not isinstance(intent["source_path"], str)
        or not intent["source_path"]
        or not isinstance(intent["snapshot_path"], str)
        or not intent["snapshot_path"]
    ):
        raise ValueError("immutable @FILE intent schema가 다르다")
    try:
        encoded_packet = payload_bytes.decode("ascii")
    except UnicodeDecodeError as error:
        raise ValueError("exec payload packet이 ASCII가 아니다") from error
    rendered_script = render_exec_script(
        template_bytes,
        encoded_packet,
        expected_template_sha256=expected_template_sha256,
    )
    script_sha = sha256_bytes(script_bytes)
    if script_bytes != rendered_script or intent["source_sha256"] != script_sha:
        raise ValueError(
            "immutable @FILE script가 등록 template과 packet 결합 결과와 다르다"
        )
    expected_command = ["brev", "exec", target_name, "@" + intent["snapshot_path"]]
    if intent["command"] != expected_command or receipt["command"] != expected_command:
        raise ValueError("native exec command가 intent snapshot과 다르다")
    packet, packet_sha = decode_packet(encoded_packet)
    expected_packet_sha256 = _sha(expected_packet_sha256, "expected packet")
    expected_payload_sha256 = _sha(expected_payload_sha256, "expected payload")
    if (
        packet_sha != expected_packet_sha256
        or packet["attempt_id"] != attempt_id
        or packet["target"]["name"] != target_name
        or packet["stage"]["sha256"] != expected_payload_sha256
    ):
        raise ValueError("exec payload packet binding이 다르다")
    parsed_result = parse_native_stdout(
        native_exec_stdout_bytes.decode("utf-8"),
        expected_target_name=target_name,
        expected_attempt_id=attempt_id,
        expected_packet_sha256=expected_packet_sha256,
        expected_payload_sha256=expected_payload_sha256,
    )
    if parsed_result != dict(result):
        raise ValueError("native stdout result와 host result가 다르다")
    readback_keys = {
        "schema",
        "attempt_id",
        "packet_sha256",
        "stage_id",
        "stage_sha256",
        "result_sha256",
    }
    expected_readback = {
        "schema": "vision_onehop_artifact_readback_v1",
        "attempt_id": attempt_id,
        "packet_sha256": expected_packet_sha256,
        "stage_id": packet["stage"]["id"],
        "stage_sha256": expected_payload_sha256,
        "result_sha256": sha256_bytes(canonical_json(dict(result))),
    }
    if (
        not isinstance(readback, dict)
        or set(readback) != readback_keys
        or readback != expected_readback
    ):
        raise ValueError("artifact readback이 stage/packet/attempt/result와 다르다")
    artifacts = {
        name: {"bytes": len(value), "sha256": sha256_bytes(value)}
        for name, value in {
            "native_exec_receipt": native_exec_receipt_bytes,
            "native_exec_stdout": native_exec_stdout_bytes,
            "native_exec_stderr": native_exec_stderr_bytes,
            "intent": intent_bytes,
            "template": template_bytes,
            "script": script_bytes,
            "payload": payload_bytes,
            "artifact_readback": artifact_readback_bytes,
            "result": canonical_json(dict(result)),
        }.items()
    }
    return {
        "schema": TRANSFER_SCHEMA,
        "attempt_id": attempt_id,
        "packet_sha256": expected_packet_sha256,
        "payload_sha256": expected_payload_sha256,
        "claims": {
            "artifact_readback_content_bound": True,
            "artifact_readback_provenance": "not_verified",
        },
        "artifacts": artifacts,
    }
