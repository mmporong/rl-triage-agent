"""NIM transport V1의 write-once JSON을 완성 뒤 원자적으로 공개한다."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import sys
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from threading import RLock
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "evals") not in sys.path:
    sys.path.insert(0, str(ROOT / "evals"))

import lineage_nim_transport_v1 as T1

SCHEMA = "lineage_nim_transport_protocol_v2"
BINDING_SCHEMA = "lineage_nim_transport_atomic_binding_v2"
BINDING_NAME = "atomic-transport-v2.json"
SOURCE_FILES = (
    "evals/lineage_nim_transport_v2.py",
    "tests/test_lineage_nim_transport_v2.py",
)
_LOCK = RLock()
_LOCAL = threading.local()
_REGISTERED_WRITER = T1._write_new


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{Path(path).name}은 JSON object여야 한다")
    return value


def _sha(path: Path, *, lf: bool = False) -> str:
    raw = Path(path).read_bytes()
    if lf:
        raw = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(raw).hexdigest()


def _relative(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError as error:
        raise ValueError("protocol은 repo 내부여야 한다") from error


def _directory_sync(path: Path) -> None:
    """POSIX에서는 link/unlink directory entry도 durable하게 동기화한다."""
    if os.name != "posix":
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_no_clobber(temp: Path, target: Path) -> None:
    """완성된 same-directory inode를 target 이름으로 원자 공개한다."""
    try:
        os.link(temp, target)
    except FileExistsError:
        raise
    except OSError:
        if os.name != "nt":
            raise
        # Windows rename은 기존 target을 덮어쓰지 않는다. POSIX에는 이 fallback을 쓰지 않는다.
        os.rename(temp, target)
        _directory_sync(target.parent)
        return
    temp.unlink()
    _directory_sync(target.parent)


def _atomic_write_new(path: Path, value: Any) -> None:
    """private temp를 fsync한 뒤 target을 no-clobber 원자 공개한다."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"write-once target이 이미 존재한다: {target}")
    suffix = f".{target.name}.{os.getpid()}.{threading.get_ident()}.{secrets.token_hex(8)}.tmp"
    temp = target.parent / suffix
    published = False
    try:
        with temp.open("xb") as stream:
            stream.write(T1._canonical(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        _publish_no_clobber(temp, target)
        published = True
    finally:
        if temp.exists():
            temp.unlink()
            _directory_sync(target.parent)
        if not published and target.exists():
            # 다른 writer가 이겼을 수 있다. 그 target은 절대 지우거나 덮지 않는다.
            pass


@contextmanager
def atomic_writer_scope() -> Iterator[None]:
    """현재 프로세스에서 T1 write 지점만 V2 atomic writer로 잠시 교체한다."""
    with _LOCK:
        depth = getattr(_LOCAL, "depth", 0)
        if depth:
            _require(T1._write_new is _atomic_write_new, "재진입 중 T1 writer가 바뀌었다")
            _LOCAL.depth = depth + 1
            try:
                yield
            finally:
                _LOCAL.depth -= 1
            return
        _require(T1._write_new is _REGISTERED_WRITER, "T1 writer가 등록 원본이 아니다")
        T1._write_new = _atomic_write_new
        _LOCAL.depth = 1
        try:
            yield
        finally:
            _LOCAL.depth = 0
            T1._write_new = _REGISTERED_WRITER


def prepare_protocol(v1_protocol: Path) -> dict[str, Any]:
    """T1 schema/model/plan은 보존하고 V2 writer source만 추가 결합한다."""
    path = Path(v1_protocol).resolve()
    value = _read(path)
    _require(value.get("schema") == T1.SCHEMA, "component transport V1 schema가 다르다")
    return {
        "schema": SCHEMA,
        "component_v1_protocol": {"path": _relative(path), "sha256_lf": _sha(path, lf=True)},
        "source_sha256_lf": {name: _sha(ROOT / name, lf=True) for name in SOURCE_FILES},
        "model": value["model"], "accepted_provider_models": deepcopy(value["accepted_provider_models"]),
        "params": deepcopy(value["params"]), "plan": deepcopy(value["plan"]),
        "operator_approved": value["operator_approved"],
        "publication": {"same_directory_private_temp": True, "temp_fsync": True,
                        "atomic_no_clobber_publish": True, "target_never_precreated": True,
                        "partial_target_visible": False, "existing_target_overwrite": False},
        "scope": {"new_queue_only": True, "existing_v1_queue_recovery": False,
                  "provider_retry": False, "duplicate_send": False},
    }


def verify_protocol(protocol_path: Path, anchor: str) -> dict[str, Any]:
    path = Path(protocol_path).resolve()
    value = _read(path)
    component = value.get("component_v1_protocol")
    if not isinstance(component, dict):
        raise TypeError("component V1 protocol ref가 없다")
    v1_path = ROOT / component["path"]
    _require(value == prepare_protocol(v1_path), "transport V2 protocol 내용 또는 SHA가 다르다")
    T1._verify_sources(v1_path, anchor)
    T1._git("cat-file", "-e", f"{anchor}^{{commit}}")
    T1._git("merge-base", "--is-ancestor", anchor, "HEAD")
    bound = {_relative(path): _sha(path, lf=True), **value["source_sha256_lf"]}
    for name, expected in bound.items():
        current = ROOT / name
        _require(current.is_file() and _sha(current, lf=True) == expected,
                 f"현재 V2 source SHA가 다르다: {name}")
        committed = T1._git("show", f"{anchor}:{name}")
        committed = committed.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        _require(hashlib.sha256(committed).hexdigest() == expected,
                 f"V2 source commit blob이 다르다: {name}")
    return deepcopy(value)


def initialize(root: Path, protocol_path: Path, anchor: str, plan: dict) -> dict:
    """V2 protocol로 검증한 fresh queue에 atomic binding을 함께 남긴다."""
    root = Path(root).resolve()
    protocol = verify_protocol(protocol_path, anchor)
    component = ROOT / protocol["component_v1_protocol"]["path"]
    with atomic_writer_scope():
        session = T1.initialize(root, component, anchor, plan)
        binding = {"schema": BINDING_SCHEMA, "protocol": {
            "path": str(Path(protocol_path).resolve()), "sha256_lf": _sha(protocol_path, lf=True)},
            "anchor": anchor, "session_sha256": _sha(root / "session.json"),
            "component_v1_protocol": deepcopy(protocol["component_v1_protocol"]),
            "atomic_writer_source_sha256_lf": deepcopy(protocol["source_sha256_lf"])}
        _atomic_write_new(root / BINDING_NAME, binding)
    return session


def _binding(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(root).resolve()
    value = _read(root / BINDING_NAME)
    _require(value.get("schema") == BINDING_SCHEMA, "V2 queue binding schema가 다르다")
    protocol_ref = value.get("protocol")
    if not isinstance(protocol_ref, dict):
        raise TypeError("V2 queue protocol binding이 없다")
    protocol_path = Path(protocol_ref["path"]).resolve()
    protocol = verify_protocol(protocol_path, protocol_ref["anchor"])
    _require(protocol_ref["sha256_lf"] == _sha(protocol_path, lf=True)
             and protocol_ref["session_sha256"] == _sha(root / "session.json")
             and protocol_ref["component_v1_protocol"] == protocol["component_v1_protocol"]
             and protocol_ref["atomic_writer_source_sha256_lf"] == protocol["source_sha256_lf"],
             "V2 queue protocol/session/source binding이 바뀌었다")
    return value, protocol


def enqueue(root: Path, slot_id: str, messages: list[dict], params: dict,
            context_digest: str, view_digest: str) -> dict:
    _binding(root)
    with atomic_writer_scope():
        return T1.enqueue(root, slot_id, messages, params, context_digest, view_digest)


def serve_one(root: Path, request_ref: dict, *, complete: Callable | None = None) -> dict:
    _binding(root)
    with atomic_writer_scope():
        return T1.serve_one(root, request_ref, complete=complete)


def file_completion(root: Path, slot_id: str, context_digest: str,
                    view_digest: str, timeout_s: float) -> Callable[[list[dict], dict], dict]:
    _binding(root)
    complete_v1 = T1.file_completion(root, slot_id, context_digest, view_digest, timeout_s)

    def complete(messages: list[dict], params: dict) -> dict:
        _binding(root)
        with atomic_writer_scope():
            return complete_v1(messages, params)

    return complete


def run_worker(root: Path, *, max_wall_s: float, poll_s: float = 0.25,
               complete: Callable | None = None) -> dict:
    _binding(root)
    with atomic_writer_scope():
        return T1.run_worker(root, max_wall_s=max_wall_s, poll_s=poll_s, complete=complete)


def close(root: Path, reason: str) -> dict:
    _binding(root)
    with atomic_writer_scope():
        return T1.close(root, reason)


def status(root: Path) -> dict:
    _binding(root)
    return T1.status(root)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=(
        "initialize", "serve-next", "once", "close", "status", "worker"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--anchor")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--request-ref", type=Path)
    parser.add_argument("--reason")
    parser.add_argument("--max-wall-s", type=float)
    parser.add_argument("--poll-s", type=float, default=0.25)
    args = parser.parse_args(argv)
    if args.command == "initialize":
        if args.protocol is None or args.anchor is None or args.plan is None:
            parser.error("initialize에는 --protocol --anchor --plan이 필요하다")
        initialize(args.root, args.protocol, args.anchor, _read(args.plan))
        return 0
    if args.command == "close":
        if args.reason is None:
            parser.error("close에는 --reason이 필요하다")
        close(args.root, args.reason)
        return 0
    if args.command == "status":
        print(T1._canonical(status(args.root)).decode("utf-8"))
        return 0
    if args.command == "worker":
        if args.max_wall_s is None:
            parser.error("worker에는 --max-wall-s가 필요하다")
        result = run_worker(args.root, max_wall_s=args.max_wall_s, poll_s=args.poll_s)
        return 2 if result["status"] == "deadline" else 0
    _binding(args.root)
    ref = T1._next_pending(args.root) if args.command == "serve-next" else (
        _read(args.request_ref) if args.request_ref is not None
        else parser.error("once에는 --request-ref가 필요하다"))
    serve_one(args.root, ref)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
