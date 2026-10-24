"""PPO 검사 선택기를 append-only 결정 원장과 연결하는 WSL 실행기.

동일 snapshot과 첫 메시지 결합을 검증한 뒤 완료 응답을 재생한다. 재시도 가능한
인프라 오류는 원장의 ``not_before``까지 멈추며, 재실행은 완료 호출을 다시 보내지 않는다.
API 키는 부모 WSL 환경에서 모델 호출기가 읽고 이 모듈은 파일을 읽지 않는다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
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

import model_decision_journal_v1 as J
import ppo_selection_adapter_v1 as B

CALL = B.CALL
MODEL = B.MODEL
STATUS_SCHEMA = "ppo_resumable_model_status_v1"
PROTOCOL = ROOT / "bench/protocols/ppo_resumable_demo_v1.json"


def _read_object(path: Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=lambda raw: (_ for _ in ()).throw(
        ValueError(f"JSON constant 거부: {raw}")))
    if not isinstance(value, dict):
        raise TypeError(f"{Path(path).name}은 JSON object여야 한다")
    return value


def _raw(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode("utf-8")


def _write_new(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_raw(value))
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def _preserve(path: Path, value: dict) -> None:
    if path.exists():
        if path.is_symlink() or not path.is_file() or _read_object(path) != value:
            raise J.JournalIntegrityError(f"기존 {path.name}이 현재 입력과 다르다")
        return
    _write_new(path, value)


def _checked_binding(binding: dict, snapshot: dict, initial_messages: list[dict]) -> dict:
    if not isinstance(binding, dict):
        raise TypeError("binding은 object여야 한다")
    checked = deepcopy(binding)
    actual_snapshot = J.digest(snapshot)
    actual_messages = J.digest(initial_messages)
    if checked.get("snapshot_sha256") != actual_snapshot:
        raise ValueError("binding snapshot_sha256가 실제 snapshot과 다르다")
    if checked.get("initial_messages_sha256") != actual_messages:
        raise ValueError("binding initial_messages_sha256가 실제 첫 messages와 다르다")
    # 나머지 source/protocol/checkpoint/run_id 형식과 필수성은 원장의 단일 계약이 검증한다.
    J._binding(checked)
    return checked


def _sha_lf(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _git(root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", "-C", str(root), *args], check=check, capture_output=True,
                          text=True, encoding="utf-8", errors="strict")


def verify_source(binding: dict, *, root: Path = ROOT, protocol_path: Path | None = None) -> dict:
    """API 호출 전에 공개 소스 결합만 검증한다. Isaac runtime 파일은 controller 검증 범위다."""
    repository = Path(root).resolve()
    protocol_file = Path(protocol_path or (repository / "bench/protocols/ppo_resumable_demo_v1.json")).resolve()
    try:
        protocol_relative = protocol_file.relative_to(repository).as_posix()
    except ValueError as error:
        raise ValueError("protocol은 저장소 안에 있어야 한다") from error
    if binding.get("protocol_sha256") != _sha_lf(protocol_file):
        raise ValueError("binding protocol SHA가 현재 고정 protocol과 다르다")
    protocol = _read_object(protocol_file)
    maps = []
    for key in ("source_sha256_lf", "evidence_sha256_lf"):
        value = protocol.get(key)
        if (not isinstance(value, dict) or not value
                or any(not isinstance(name, str) or not name or not isinstance(digest, str)
                       or re.fullmatch(r"[0-9a-f]{64}", digest) is None
                       for name, digest in value.items())):
            raise ValueError(f"protocol {key} 결합이 없다")
        maps.append(value)
    expected = {name: digest for table in maps for name, digest in table.items()}
    if len(expected) != sum(len(table) for table in maps):
        raise ValueError("source와 evidence 결합 경로가 중복됐다")
    paths = []
    for name, digest in expected.items():
        candidate = (repository / name).resolve()
        try:
            relative = candidate.relative_to(repository).as_posix()
        except ValueError as error:
            raise ValueError("결합 경로가 저장소 밖을 가리킨다") from error
        if relative != name.replace("\\", "/") or not candidate.is_file() or _sha_lf(candidate) != digest:
            raise ValueError(f"결합 파일 LF SHA가 다르다: {name}")
        paths.append(relative)
    anchor = binding.get("source_anchor")
    if not isinstance(anchor, str) or re.fullmatch(r"[0-9a-f]{40}", anchor) is None:
        raise ValueError("source anchor 형식이 다르다")
    head = _git(repository, "rev-parse", "HEAD").stdout.strip()
    if _git(repository, "merge-base", "--is-ancestor", anchor, head, check=False).returncode != 0:
        raise ValueError("source anchor가 현재 HEAD의 조상이 아니다")
    bound = sorted({*paths, protocol_relative})
    _git(repository, "ls-files", "--error-unmatch", "--", *bound)
    if _git(repository, "status", "--porcelain", "--", *bound).stdout.strip():
        raise ValueError("API 결합 파일이 clean 상태가 아니다")
    for name in bound:
        anchor_blob = _git(repository, "rev-parse", f"{anchor}:{name}").stdout.strip()
        head_blob = _git(repository, "rev-parse", f"{head}:{name}").stdout.strip()
        if anchor_blob != head_blob:
            raise ValueError(f"source anchor 뒤 결합 blob이 변경됐다: {name}")
    return {"source_anchor": anchor, "head": head, "protocol_sha256_lf": _sha_lf(protocol_file),
            "verified_source_files": len(paths),
            "runtime_source_verification": "excluded_controller_only"}


def _default_complete(messages: list[dict], params: dict) -> dict:
    return CALL.safe_result(CALL.F.openai_complete(messages, params))


def _journal_engine(snapshot: dict, completion: Callable[[list, dict], dict]) -> dict:
    return B.engine(snapshot, completion)


def run(
    folder: Path,
    packet: dict,
    binding: dict,
    complete: Callable[[list[dict], dict], dict] | None = None,
    now_utc: Callable[[], datetime] | None = None,
) -> dict:
    """동일 판단을 재개하고 완료 응답은 API 원장에서 재생한다."""
    root = Path(folder)
    snapshot = B.build_snapshot(packet)
    initial_messages = B.messages(snapshot)
    checked_binding = _checked_binding(binding, snapshot, initial_messages)
    if complete is None:
        verify_source(checked_binding)
    params = CALL.F.request_params(MODEL)
    journal = J.DecisionJournal(
        root=root / "decision",
        api_root=root / "api",
        binding=checked_binding,
        params=params,
        retry_delay_s=30,
        api_cap=20,
    )
    provider = complete if complete is not None else _default_complete
    clock = now_utc if now_utc is not None else lambda: datetime.now(timezone.utc)  # noqa: UP017
    return journal.advance(lambda completion: _journal_engine(snapshot, completion), provider, clock)


def cli(
    packet_path: Path,
    binding_path: Path,
    folder: Path,
    *,
    complete: Callable[[list[dict], dict], dict] | None = None,
    now_utc: Callable[[], datetime] | None = None,
) -> dict:
    """입력 사본과 매 실행 status를 덮어쓰지 않고 남긴다."""
    packet = _read_object(packet_path)
    binding = _read_object(binding_path)
    root = Path(folder)
    if root.is_symlink():
        raise J.JournalIntegrityError("실행 folder symlink를 거부한다")
    root.mkdir(parents=True, exist_ok=True)
    _preserve(root / "input_packet.json", packet)
    _preserve(root / "input_binding.json", binding)
    result = run(root, packet, binding, complete=complete, now_utc=now_utc)
    status_dir = root / "statuses"
    existing = sorted(status_dir.glob("*.json")) if status_dir.exists() else []
    if [path.name for path in existing] != [f"{index:04d}.json" for index in range(1, len(existing) + 1)]:
        raise J.JournalIntegrityError("status 파일 순서가 다르다")
    record = {"schema": STATUS_SCHEMA, "ordinal": len(existing) + 1, "result": result}
    _write_new(status_dir / f"{len(existing) + 1:04d}.json", record)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--pid-file", type=Path)
    args = parser.parse_args(argv)
    identity = None
    if args.pid_file is not None:
        stat = Path("/proc/self/stat")
        if not sys.platform.startswith("linux") or not stat.is_file():
            raise RuntimeError("PID identity는 Linux worker에서만 기록한다")
        identity = {
            "pid": os.getpid(),
            "start_ticks": stat.read_text(encoding="utf-8").rsplit(")", 1)[1].split()[19],
            "packet_sha256": hashlib.sha256(args.packet.read_bytes()).hexdigest(),
        }
        _write_new(args.pid_file, identity)
    try:
        result = cli(args.packet, args.binding, args.folder)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    finally:
        if args.pid_file is not None:
            if _read_object(args.pid_file) != identity:
                raise J.JournalIntegrityError("worker PID identity가 실행 중 변경됐다")
            args.pid_file.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
