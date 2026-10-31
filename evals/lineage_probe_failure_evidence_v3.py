"""Lineage probe backend 예외의 상세 증거를 별도 write-once sidecar로 보존한다."""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import threading
import traceback
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from threading import RLock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "bench"), str(ROOT / "evals"), str(ROOT / "src")]

import lineage_probe_admission_v2 as A

SCHEMA = "lineage_probe_failure_evidence_protocol_v3"
DETAIL_SCHEMA = "lineage_probe_backend_exception_detail_v3"
DETAIL_NAME = "backend-exception-detail.json"
SOURCE_FILES = (
    "evals/lineage_probe_failure_evidence_v3.py",
    "tests/test_lineage_probe_failure_evidence_v3.py",
)
_ORIGINAL_EXECUTE_PROBE = A.execute_probe
_LOCK = RLock()
_LOCAL = threading.local()


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def _sha(path: Path, *, lf: bool = False) -> str:
    raw = Path(path).read_bytes()
    if lf:
        raw = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(raw).hexdigest()


def _read(path: Path) -> dict:
    value = json.loads(
        Path(path).read_text(encoding="utf-8"),
        parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)),
    )
    _require(isinstance(value, dict), f"{Path(path).name}은 JSON 객체여야 한다")
    return value


def _write_new(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(
        value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
    ).encode("utf-8") + b"\n"
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def _relative(path: Path) -> str:
    try:
        value = Path(path).resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError as error:
        raise ValueError("protocol은 저장소 내부 파일이어야 한다") from error
    _require(bool(value) and ".." not in Path(value).parts and not Path(value).is_absolute(),
             "protocol 상대경로가 안전하지 않다")
    return value


def _ref(path: Path) -> dict:
    path = Path(path).resolve()
    _require(path.is_file(), f"증거 파일이 없다: {path.name}")
    return {"path": str(path), "sha256": _sha(path)}


def prepare_protocol(admission_protocol: Path) -> dict:
    """입장 v2 protocol을 그대로 두고 예외 증거 계층만 결합한다."""
    path = Path(admission_protocol).resolve()
    value = _read(path)
    _require(value.get("schema") == A.SCHEMA, "입장 wrapper protocol schema가 다르다")
    component = {"path": _relative(path), "sha256_lf": _sha(path, lf=True)}
    return {
        "schema": SCHEMA,
        "phase": "preserve_probe_backend_exception_without_retry",
        "component_admission_protocol": component,
        "source_sha256_lf": {name: _sha(ROOT / name, lf=True) for name in SOURCE_FILES},
        "failure_contract": {
            "sidecar": DETAIL_NAME,
            "write_once": True,
            "retry": False,
            "original_exception_reraised": True,
            "original_probe_receipt_unchanged": True,
        },
        "claims": {"measurement": False, "correction": False, "recovery": False},
    }


def verify_protocol(protocol_path: Path, *, anchor: str) -> dict:
    """v3 source와 입장 v2 component가 같은 등록 commit에 있음을 확인한다."""
    _require(isinstance(anchor, str) and re.fullmatch(r"[0-9a-f]{40}", anchor) is not None,
             "registration anchor 형식이 다르다")
    path = Path(protocol_path).resolve()
    value = _read(path)
    component = value.get("component_admission_protocol")
    if not isinstance(component, dict) or set(component) != {"path", "sha256_lf"}:
        raise ValueError("입장 wrapper protocol 결합이 없다")
    component_path = (ROOT / str(component["path"])).resolve()
    expected = prepare_protocol(component_path)
    _require(value == expected, "v3 protocol 내용 또는 source 결합이 다르다")
    A.verify_protocol(component_path, anchor=anchor)
    A._git("cat-file", "-e", f"{anchor}^{{commit}}")
    A._git("merge-base", "--is-ancestor", anchor, "HEAD")
    bound = {_relative(path): _sha(path, lf=True), **value["source_sha256_lf"]}
    for relative, expected_sha in bound.items():
        current = ROOT / relative
        _require(current.is_file() and _sha(current, lf=True) == expected_sha,
                 f"현재 source/blob이 다르다: {relative}")
        committed = A._git("show", f"{anchor}:{relative}")
        committed = committed.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        _require(hashlib.sha256(committed).hexdigest() == expected_sha,
                 f"anchor source/blob이 다르다: {relative}")
    return deepcopy(value)


def execute_probe(plan_path: Path, authorization_path: Path) -> dict:
    """입장 v2 실행을 위임하고 Exception 상세만 별도 sidecar에 봉인한다."""
    plan_path = Path(plan_path).resolve()
    authorization_path = Path(authorization_path).resolve()
    plan = _read(plan_path)
    stage = Path(str(plan.get("stage", "")))
    _require(stage.is_absolute() and plan_path == stage / "plan.json",
             "probe plan stage 또는 위치가 다르다")
    detail_path = stage / DETAIL_NAME
    if detail_path.exists():
        raise FileExistsError(f"backend exception detail이 이미 존재한다: {detail_path}")
    try:
        return _ORIGINAL_EXECUTE_PROBE(plan_path, authorization_path)
    except Exception as error:
        receipt_path = stage / "probe_receipt.json"
        owned_path = stage / "owned-result.json"
        detail = {
            "schema": DETAIL_SCHEMA,
            "status": "backend_exception",
            "error": {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
                "errno": getattr(error, "errno", None),
                "winerror": getattr(error, "winerror", None),
                "filename": getattr(error, "filename", None),
            },
            "plan": _ref(plan_path),
            "authorization": _ref(authorization_path),
            "probe_receipt": _ref(receipt_path) if receipt_path.is_file() else None,
            "owned_result": _ref(owned_path) if owned_path.is_file() else None,
            "scope": {
                "retry": False,
                "measurement_changed": False,
                "original_probe_receipt_changed": False,
            },
        }
        _write_new(detail_path, detail)
        raise


@contextmanager
def logging_scope():
    """현재 thread의 실제 기본 probe 경로에만 상세 logging을 연결한다."""
    with _LOCK:
        depth = getattr(_LOCAL, "depth", 0)
        if depth:
            _require(A.execute_probe is execute_probe,
                     "재진입 중 admission probe executor가 바뀌었다")
            _LOCAL.depth = depth + 1
            try:
                yield
            finally:
                _LOCAL.depth -= 1
            return
        _require(A.execute_probe is _ORIGINAL_EXECUTE_PROBE,
                 "admission probe executor가 등록 원본이 아니다")
        original = A.execute_probe
        A.execute_probe = execute_probe
        _LOCAL.depth = 1
        try:
            yield
        finally:
            _LOCAL.depth = 0
            A.execute_probe = original


def run_epoch(*args, failure_protocol: Path, anchor: str, **kwargs) -> dict:
    """v3 결합을 확인하고 실제 기본 probe 경로에 logging을 연결한다."""
    _require("probe_executor" not in kwargs, "v3 probe executor는 교체할 수 없다")
    protocol = verify_protocol(Path(failure_protocol), anchor=anchor)
    component = (ROOT / protocol["component_admission_protocol"]["path"]).resolve()
    with logging_scope():
        return A.run_epoch(
            *args,
            wrapper_protocol=component,
            anchor=anchor,
            **kwargs,
        )
