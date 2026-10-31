"""NIM transport V2 binding의 최상위 필드를 검증해 atomic queue를 소비한다."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
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

import lineage_nim_transport_v2 as T2

SCHEMA = "lineage_nim_transport_protocol_v3"
BINDING_SCHEMA = "lineage_nim_transport_binding_v3"
BINDING_NAME = "lineage-transport-v3.json"
SOURCE_FILES = (
    "evals/lineage_nim_transport_v3.py",
    "tests/test_lineage_nim_transport_v3.py",
)
_LOCK = RLock()
_LOCAL = threading.local()
_REGISTERED_BINDING = T2._binding
_HEX40 = re.compile(r"[0-9a-f]{40}")
_HEX64 = re.compile(r"[0-9a-f]{64}")


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


def _git(*args: str) -> bytes:
    result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, check=False)
    _require(result.returncode == 0, "transport V3 commit/blob 검증 실패")
    return result.stdout


def prepare_protocol(v2_protocol: Path) -> dict[str, Any]:
    """T2 계약 전체와 V3 binding 교정 source를 새 protocol로 결합한다."""
    path = Path(v2_protocol).resolve()
    component = _read(path)
    _require(component.get("schema") == T2.SCHEMA, "component transport V2 schema가 다르다")
    return {
        "schema": SCHEMA,
        "component_v2_protocol": {"path": _relative(path), "sha256_lf": _sha(path, lf=True)},
        "source_sha256_lf": {name: _sha(ROOT / name, lf=True) for name in SOURCE_FILES},
        "model": component["model"],
        "accepted_provider_models": deepcopy(component["accepted_provider_models"]),
        "params": deepcopy(component["params"]),
        "plan": deepcopy(component["plan"]),
        "operator_approved": component["operator_approved"],
        "publication": deepcopy(component["publication"]),
        "scope": {**deepcopy(component["scope"]),
                  "binding_fix": "validate_v2_top_level_anchor_session_component_source"},
    }


def verify_protocol(protocol_path: Path, anchor: str) -> dict[str, Any]:
    """V3 protocol, component T2와 현재/anchor source blob을 모두 검증한다."""
    path = Path(protocol_path).resolve()
    value = _read(path)
    component_ref = value.get("component_v2_protocol")
    if not isinstance(component_ref, dict) or set(component_ref) != {"path", "sha256_lf"}:
        raise TypeError("component V2 protocol ref가 없다")
    component_path = ROOT / component_ref["path"]
    _require(value == prepare_protocol(component_path), "transport V3 protocol 내용 또는 SHA가 다르다")
    _require(component_ref["sha256_lf"] == _sha(component_path, lf=True),
             "component V2 protocol SHA가 다르다")
    T2.verify_protocol(component_path, anchor)
    _require(isinstance(anchor, str) and bool(_HEX40.fullmatch(anchor)), "anchor 형식이 다르다")
    _git("cat-file", "-e", f"{anchor}^{{commit}}")
    _git("merge-base", "--is-ancestor", anchor, "HEAD")
    bound = [(_relative(path), path, _sha(path, lf=True)), *(
        (name, ROOT / name, expected) for name, expected in value["source_sha256_lf"].items())]
    for name, current, expected in bound:
        _require(bool(_HEX64.fullmatch(str(expected))) and current.is_file()
                 and _sha(current, lf=True) == expected, f"현재 V3 source SHA가 다르다: {name}")
        committed = _git("show", f"{anchor}:{name}").replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        _require(hashlib.sha256(committed).hexdigest() == expected,
                 f"V3 source commit blob이 다르다: {name}")
    return deepcopy(value)


def _fixed_v2_binding(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """T2.initialize가 실제 기록한 최상위 binding 필드를 검증한다."""
    root = Path(root).resolve()
    value = _read(root / T2.BINDING_NAME)
    _require(set(value) == {"schema", "protocol", "anchor", "session_sha256",
                            "component_v1_protocol", "atomic_writer_source_sha256_lf"}
             and value.get("schema") == T2.BINDING_SCHEMA, "V2 queue binding shape가 다르다")
    protocol_ref = value.get("protocol")
    if not isinstance(protocol_ref, dict) or set(protocol_ref) != {"path", "sha256_lf"}:
        raise TypeError("V2 queue protocol binding이 없다")
    protocol_path = Path(protocol_ref["path"]).resolve()
    protocol = T2.verify_protocol(protocol_path, value["anchor"])
    session = T2.T1._read(root / "session.json")
    _require(bool(_HEX40.fullmatch(str(value["anchor"])))
             and value["anchor"] == session.get("registration_anchor")
             and protocol_ref["sha256_lf"] == _sha(protocol_path, lf=True)
             and value["session_sha256"] == _sha(root / "session.json")
             and value["component_v1_protocol"] == protocol["component_v1_protocol"]
             and value["atomic_writer_source_sha256_lf"] == protocol["source_sha256_lf"],
             "V2 queue 최상위 protocol/session/component/source binding이 바뀌었다")
    return deepcopy(value), protocol


@contextmanager
def binding_scope() -> Iterator[None]:
    """T2의 잘못된 `_binding` 조회점만 재진입 잠금 아래 교체한다."""
    with _LOCK:
        depth = getattr(_LOCAL, "depth", 0)
        if depth:
            _require(T2._binding is _fixed_v2_binding, "재진입 중 T2 binding 검증기가 바뀌었다")
            _LOCAL.depth = depth + 1
            try:
                yield
            finally:
                _LOCAL.depth -= 1
            return
        _require(T2._binding is _REGISTERED_BINDING, "T2 binding 검증기가 등록 원본이 아니다")
        T2._binding = _fixed_v2_binding
        _LOCAL.depth = 1
        try:
            yield
        finally:
            _LOCAL.depth = 0
            T2._binding = _REGISTERED_BINDING


def initialize(root: Path, protocol_path: Path, anchor: str, plan: dict) -> dict:
    """검증된 V3 protocol로 fresh T2 queue와 별도 V3 binding을 만든다."""
    root = Path(root).resolve()
    protocol = verify_protocol(protocol_path, anchor)
    component_path = ROOT / protocol["component_v2_protocol"]["path"]
    session = T2.initialize(root, component_path, anchor, plan)
    component_binding, _ = _fixed_v2_binding(root)
    binding = {
        "schema": BINDING_SCHEMA,
        "protocol": {"path": str(Path(protocol_path).resolve()),
                     "sha256_lf": _sha(protocol_path, lf=True)},
        "anchor": anchor,
        "session_sha256": _sha(root / "session.json"),
        "component_v2_protocol": deepcopy(protocol["component_v2_protocol"]),
        "transport_v3_source_sha256_lf": deepcopy(protocol["source_sha256_lf"]),
        "component_v2_binding_sha256": T2.T1._digest(component_binding),
    }
    T2._atomic_write_new(root / BINDING_NAME, binding)
    return session


def _binding(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(root).resolve()
    value = _read(root / BINDING_NAME)
    _require(set(value) == {"schema", "protocol", "anchor", "session_sha256",
                            "component_v2_protocol", "transport_v3_source_sha256_lf",
                            "component_v2_binding_sha256"}
             and value.get("schema") == BINDING_SCHEMA, "V3 queue binding shape가 다르다")
    protocol_ref = value.get("protocol")
    if not isinstance(protocol_ref, dict) or set(protocol_ref) != {"path", "sha256_lf"}:
        raise TypeError("V3 queue protocol binding이 없다")
    protocol_path = Path(protocol_ref["path"]).resolve()
    protocol = verify_protocol(protocol_path, value["anchor"])
    component_binding, _ = _fixed_v2_binding(root)
    session = T2.T1._read(root / "session.json")
    _require(bool(_HEX40.fullmatch(str(value["anchor"])))
             and value["anchor"] == session.get("registration_anchor")
             and protocol_ref["sha256_lf"] == _sha(protocol_path, lf=True)
             and value["session_sha256"] == _sha(root / "session.json")
             and value["component_v2_protocol"] == protocol["component_v2_protocol"]
             and value["transport_v3_source_sha256_lf"] == protocol["source_sha256_lf"]
             and value["component_v2_binding_sha256"] == T2.T1._digest(component_binding),
             "V3 queue protocol/session/component/source binding이 바뀌었다")
    return deepcopy(value), protocol


def enqueue(root: Path, slot_id: str, messages: list[dict], params: dict,
            context_digest: str, view_digest: str) -> dict:
    _binding(root)
    with binding_scope():
        return T2.enqueue(root, slot_id, messages, params, context_digest, view_digest)


def serve_one(root: Path, request_ref: dict, *, complete: Callable | None = None) -> dict:
    _binding(root)
    with binding_scope():
        return T2.serve_one(root, request_ref, complete=complete)


def file_completion(root: Path, slot_id: str, context_digest: str,
                    view_digest: str, timeout_s: float) -> Callable[[list[dict], dict], dict]:
    _binding(root)
    with binding_scope():
        complete_v2 = T2.file_completion(root, slot_id, context_digest, view_digest, timeout_s)

    def complete(messages: list[dict], params: dict) -> dict:
        _binding(root)
        with binding_scope():
            return complete_v2(messages, params)

    return complete


def run_worker(root: Path, *, max_wall_s: float, poll_s: float = 0.25,
               complete: Callable | None = None) -> dict:
    _binding(root)
    with binding_scope():
        return T2.run_worker(root, max_wall_s=max_wall_s, poll_s=poll_s, complete=complete)


def close(root: Path, reason: str) -> dict:
    _binding(root)
    with binding_scope():
        return T2.close(root, reason)


def status(root: Path) -> dict:
    _binding(root)
    with binding_scope():
        return T2.status(root)


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
        print(T2.T1._canonical(status(args.root)).decode("utf-8"))
        return 0
    if args.command == "worker":
        if args.max_wall_s is None:
            parser.error("worker에는 --max-wall-s가 필요하다")
        result = run_worker(args.root, max_wall_s=args.max_wall_s, poll_s=args.poll_s)
        return 2 if result["status"] == "deadline" else 0
    _binding(args.root)
    ref = T2.T1._next_pending(args.root) if args.command == "serve-next" else (
        _read(args.request_ref) if args.request_ref is not None
        else parser.error("once에는 --request-ref가 필요하다"))
    serve_one(args.root, ref)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
