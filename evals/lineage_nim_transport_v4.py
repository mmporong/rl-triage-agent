"""Windows에서 기록된 T2/T3 protocol 경로를 봉인 alias로만 해석하는 transport V4."""
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

import lineage_nim_transport_v3 as T3

T2 = T3.T2
SCHEMA = "lineage_nim_transport_protocol_v4"
BINDING_SCHEMA = "lineage_nim_transport_binding_v4"
BINDING_NAME = "lineage-transport-v4.json"
SOURCE_FILES = ("evals/lineage_nim_transport_v4.py", "tests/test_lineage_nim_transport_v4.py")
_HEX40 = re.compile(r"[0-9a-f]{40}")
_HEX64 = re.compile(r"[0-9a-f]{64}")
_LOCK = RLock()
_LOCAL = threading.local()
_REGISTERED_T3_BINDING = T3._binding
_REGISTERED_T3_V2_BINDING = T3._fixed_v2_binding


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
    _require(result.returncode == 0, "transport V4 commit/blob 검증 실패")
    return result.stdout


def prepare_protocol(v3_protocol: Path) -> dict[str, Any]:
    path = Path(v3_protocol).resolve()
    component = _read(path)
    _require(component.get("schema") == T3.SCHEMA, "component transport V3 schema가 다르다")
    return {
        "schema": SCHEMA,
        "component_v3_protocol": {"path": _relative(path), "sha256_lf": _sha(path, lf=True)},
        "source_sha256_lf": {name: _sha(ROOT / name, lf=True) for name in SOURCE_FILES},
        "model": component["model"],
        "accepted_provider_models": deepcopy(component["accepted_provider_models"]),
        "params": deepcopy(component["params"]), "plan": deepcopy(component["plan"]),
        "operator_approved": component["operator_approved"],
        "publication": deepcopy(component["publication"]),
        "scope": {**deepcopy(component["scope"]), "portable_binding_paths": "exact_alias_only"},
    }


def verify_protocol(protocol_path: Path, anchor: str) -> dict[str, Any]:
    path = Path(protocol_path).resolve()
    value = _read(path)
    component = value.get("component_v3_protocol")
    if not isinstance(component, dict) or set(component) != {"path", "sha256_lf"}:
        raise TypeError("component V3 protocol ref가 없다")
    component_path = ROOT / component["path"]
    _require(value == prepare_protocol(component_path), "transport V4 protocol 내용 또는 SHA가 다르다")
    T3.verify_protocol(component_path, anchor)
    _require(isinstance(anchor, str) and bool(_HEX40.fullmatch(anchor)), "anchor 형식이 다르다")
    _git("cat-file", "-e", f"{anchor}^{{commit}}")
    _git("merge-base", "--is-ancestor", anchor, "HEAD")
    bound = [(_relative(path), path, _sha(path, lf=True)), *(
        (name, ROOT / name, digest) for name, digest in value["source_sha256_lf"].items())]
    for name, current, expected in bound:
        _require(bool(_HEX64.fullmatch(str(expected))) and current.is_file()
                 and _sha(current, lf=True) == expected, f"현재 V4 source SHA가 다르다: {name}")
        blob = _git("show", f"{anchor}:{name}").replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        _require(hashlib.sha256(blob).hexdigest() == expected,
                 f"V4 source commit blob이 다르다: {name}")
    return deepcopy(value)


def _alias(recorded: str, logical: dict) -> dict[str, str]:
    _require(isinstance(recorded, str) and bool(recorded), "기록 protocol path가 비었다")
    _require(isinstance(logical, dict) and set(logical) == {"path", "sha256_lf"},
             "logical protocol ref 형식이 다르다")
    return {"recorded_path": recorded, "logical_path": logical["path"],
            "sha256_lf": logical["sha256_lf"]}


def _raw_bindings(root: Path) -> tuple[dict, dict]:
    root = Path(root).resolve()
    return _read(root / T3.BINDING_NAME), _read(root / T2.BINDING_NAME)


def _expected_aliases(protocol: dict, raw_v3: dict, raw_v2: dict) -> list[dict[str, str]]:
    v3_ref = protocol["component_v3_protocol"]
    v3_protocol = _read(ROOT / v3_ref["path"])
    v2_ref = v3_protocol["component_v2_protocol"]
    return [_alias(raw_v3["protocol"]["path"], v3_ref),
            _alias(raw_v2["protocol"]["path"], v2_ref)]


def _v4_binding(root: Path) -> tuple[dict, dict, dict, dict]:
    root = Path(root).resolve()
    value = _read(root / BINDING_NAME)
    required = {"schema", "protocol", "anchor", "session_sha256", "component_v3_protocol",
                "component_v3_binding_sha256", "component_v2_binding_sha256",
                "transport_v4_source_sha256_lf", "path_aliases"}
    _require(set(value) == required and value.get("schema") == BINDING_SCHEMA,
             "V4 queue binding shape가 다르다")
    protocol_ref = value.get("protocol")
    if not isinstance(protocol_ref, dict) or set(protocol_ref) != {"path", "sha256_lf"}:
        raise TypeError("V4 queue protocol binding이 없다")
    _require(isinstance(protocol_ref["path"], str) and ":" not in protocol_ref["path"]
             and "\\" not in protocol_ref["path"], "V4 protocol은 repo-relative POSIX path여야 한다")
    protocol_path = ROOT / protocol_ref["path"]
    protocol = verify_protocol(protocol_path, value["anchor"])
    raw_v3, raw_v2 = _raw_bindings(root)
    session = T2.T1._read(root / "session.json")
    expected_aliases = _expected_aliases(protocol, raw_v3, raw_v2)
    _require(bool(_HEX40.fullmatch(str(value["anchor"])))
             and value["anchor"] == session.get("registration_anchor")
             and protocol_ref["sha256_lf"] == _sha(protocol_path, lf=True)
             and value["session_sha256"] == _sha(root / "session.json")
             and value["component_v3_protocol"] == protocol["component_v3_protocol"]
             and value["component_v3_binding_sha256"] == _sha(root / T3.BINDING_NAME)
             and value["component_v2_binding_sha256"] == _sha(root / T2.BINDING_NAME)
             and value["transport_v4_source_sha256_lf"] == protocol["source_sha256_lf"]
             and value["path_aliases"] == expected_aliases,
             "V4 queue protocol/session/component/source/alias binding이 바뀌었다")
    return deepcopy(value), protocol, raw_v3, raw_v2


def _resolve_alias(binding: dict, recorded: object, logical: dict) -> Path:
    expected = _alias(recorded, logical) if isinstance(recorded, str) else None
    matches = [row for row in binding["path_aliases"] if row == expected]
    _require(len(matches) == 1, "기록 protocol path의 exact alias가 없다")
    path = ROOT / logical["path"]
    _require(path.is_file() and _sha(path, lf=True) == logical["sha256_lf"],
             "logical protocol current SHA가 다르다")
    return path


def _portable_v2_binding(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    binding, protocol4, _raw_v3, raw = _v4_binding(root)
    _require(set(raw) == {"schema", "protocol", "anchor", "session_sha256",
                          "component_v1_protocol", "atomic_writer_source_sha256_lf"}
             and raw.get("schema") == T2.BINDING_SCHEMA, "V2 queue binding shape가 다르다")
    v3_protocol = _read(ROOT / protocol4["component_v3_protocol"]["path"])
    v2_ref = v3_protocol["component_v2_protocol"]
    protocol_path = _resolve_alias(binding, raw.get("protocol", {}).get("path"), v2_ref)
    protocol = T2.verify_protocol(protocol_path, raw["anchor"])
    session = T2.T1._read(Path(root) / "session.json")
    _require(raw["protocol"].get("sha256_lf") == _sha(protocol_path, lf=True)
             and raw["anchor"] == session.get("registration_anchor")
             and raw["session_sha256"] == _sha(Path(root) / "session.json")
             and raw["component_v1_protocol"] == protocol["component_v1_protocol"]
             and raw["atomic_writer_source_sha256_lf"] == protocol["source_sha256_lf"],
             "portable V2 binding 검증 실패")
    return deepcopy(raw), protocol


def _portable_v3_binding(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    binding, protocol4, raw, _raw_v2 = _v4_binding(root)
    required = {"schema", "protocol", "anchor", "session_sha256", "component_v2_protocol",
                "transport_v3_source_sha256_lf", "component_v2_binding_sha256"}
    _require(set(raw) == required and raw.get("schema") == T3.BINDING_SCHEMA,
             "V3 queue binding shape가 다르다")
    v3_ref = protocol4["component_v3_protocol"]
    protocol_path = _resolve_alias(binding, raw.get("protocol", {}).get("path"), v3_ref)
    protocol = T3.verify_protocol(protocol_path, raw["anchor"])
    raw_v2, _ = _portable_v2_binding(root)
    session = T2.T1._read(Path(root) / "session.json")
    _require(raw["protocol"].get("sha256_lf") == _sha(protocol_path, lf=True)
             and raw["anchor"] == session.get("registration_anchor")
             and raw["session_sha256"] == _sha(Path(root) / "session.json")
             and raw["component_v2_protocol"] == protocol["component_v2_protocol"]
             and raw["transport_v3_source_sha256_lf"] == protocol["source_sha256_lf"]
             and raw["component_v2_binding_sha256"] == T2.T1._digest(raw_v2),
             "portable V3 binding 검증 실패")
    return deepcopy(raw), protocol


@contextmanager
def compatibility_scope() -> Iterator[None]:
    """T3의 두 path-sensitive 검증 함수만 portable alias 검증기로 교체한다."""
    with _LOCK:
        depth = getattr(_LOCAL, "depth", 0)
        if depth:
            _require(T3._binding is _portable_v3_binding
                     and T3._fixed_v2_binding is _portable_v2_binding,
                     "재진입 중 T3 portable 검증기가 바뀌었다")
            _LOCAL.depth = depth + 1
            try:
                yield
            finally:
                _LOCAL.depth -= 1
            return
        _require(T3._binding is _REGISTERED_T3_BINDING
                 and T3._fixed_v2_binding is _REGISTERED_T3_V2_BINDING,
                 "T3 binding 검증기가 등록 원본이 아니다")
        T3._binding, T3._fixed_v2_binding = _portable_v3_binding, _portable_v2_binding
        _LOCAL.depth = 1
        try:
            yield
        finally:
            _LOCAL.depth = 0
            T3._binding, T3._fixed_v2_binding = _REGISTERED_T3_BINDING, _REGISTERED_T3_V2_BINDING


def initialize(root: Path, protocol_path: Path, anchor: str, plan: dict) -> dict:
    root, path = Path(root).resolve(), Path(protocol_path).resolve()
    protocol = verify_protocol(path, anchor)
    component = ROOT / protocol["component_v3_protocol"]["path"]
    session = T3.initialize(root, component, anchor, plan)
    raw_v3, raw_v2 = _raw_bindings(root)
    binding = {
        "schema": BINDING_SCHEMA,
        "protocol": {"path": _relative(path), "sha256_lf": _sha(path, lf=True)},
        "anchor": anchor, "session_sha256": _sha(root / "session.json"),
        "component_v3_protocol": deepcopy(protocol["component_v3_protocol"]),
        "component_v3_binding_sha256": _sha(root / T3.BINDING_NAME),
        "component_v2_binding_sha256": _sha(root / T2.BINDING_NAME),
        "transport_v4_source_sha256_lf": deepcopy(protocol["source_sha256_lf"]),
        "path_aliases": _expected_aliases(protocol, raw_v3, raw_v2),
    }
    T2._atomic_write_new(root / BINDING_NAME, binding)
    _v4_binding(root)
    return session


def _call(root: Path, function: Callable, *args, **kwargs):
    _v4_binding(root)
    with compatibility_scope():
        return function(root, *args, **kwargs)


def enqueue(root: Path, slot_id: str, messages: list[dict], params: dict,
            context_digest: str, view_digest: str) -> dict:
    return _call(root, T3.enqueue, slot_id, messages, params, context_digest, view_digest)


def serve_one(root: Path, request_ref: dict, *, complete: Callable | None = None) -> dict:
    return _call(root, T3.serve_one, request_ref, complete=complete)


def file_completion(root: Path, slot_id: str, context_digest: str,
                    view_digest: str, timeout_s: float) -> Callable[[list[dict], dict], dict]:
    complete_v3 = _call(root, T3.file_completion, slot_id, context_digest, view_digest, timeout_s)

    def complete(messages: list[dict], params: dict) -> dict:
        _v4_binding(root)
        with compatibility_scope():
            return complete_v3(messages, params)
    return complete


def run_worker(root: Path, *, max_wall_s: float, poll_s: float = 0.25,
               complete: Callable | None = None) -> dict:
    return _call(root, T3.run_worker, max_wall_s=max_wall_s, poll_s=poll_s, complete=complete)


def close(root: Path, reason: str) -> dict:
    return _call(root, T3.close, reason)


def status(root: Path) -> dict:
    return _call(root, T3.status)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=(
        "prepare-protocol", "initialize", "serve-next", "once", "close", "status", "worker"))
    parser.add_argument("--root", type=Path)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--component-v3", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--anchor")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--request-ref", type=Path)
    parser.add_argument("--reason")
    parser.add_argument("--max-wall-s", type=float)
    parser.add_argument("--poll-s", type=float, default=0.25)
    args = parser.parse_args(argv)
    if args.command == "prepare-protocol":
        if args.component_v3 is None or args.output is None:
            parser.error("prepare-protocol에는 --component-v3 --output이 필요하다")
        T2._atomic_write_new(args.output, prepare_protocol(args.component_v3))
        return 0
    if args.root is None:
        parser.error("queue 명령에는 --root가 필요하다")
    if args.command == "initialize":
        if args.protocol is None or args.anchor is None or args.plan is None:
            parser.error("initialize에는 --protocol --anchor --plan이 필요하다")
        initialize(args.root, args.protocol, args.anchor, _read(args.plan))
    elif args.command == "close":
        if args.reason is None:
            parser.error("close에는 --reason이 필요하다")
        close(args.root, args.reason)
    elif args.command == "status":
        print(T2.T1._canonical(status(args.root)).decode("utf-8"))
    elif args.command == "worker":
        if args.max_wall_s is None:
            parser.error("worker에는 --max-wall-s가 필요하다")
        return 2 if run_worker(args.root, max_wall_s=args.max_wall_s,
                               poll_s=args.poll_s)["status"] == "deadline" else 0
    else:
        ref = T2.T1._next_pending(args.root) if args.command == "serve-next" else (
            _read(args.request_ref) if args.request_ref is not None
            else parser.error("once에는 --request-ref가 필요하다"))
        serve_one(args.root, ref)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
