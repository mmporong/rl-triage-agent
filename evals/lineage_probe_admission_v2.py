"""Lineage probe를 승인·claim 전에 공유 GPU 입장 대기에 연결한다."""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
import traceback
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from threading import RLock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "bench"), str(ROOT / "evals"), str(ROOT / "src")]

import lineage_diagnosis_epoch_v1 as E
import lineage_gpu_wait_v1 as GPU_WAIT
import lineage_probe_backend_v1 as P

SCHEMA = "lineage_probe_admission_wrapper_protocol_v2"
WAIT_SCHEMA = "lineage_probe_admission_wait_receipt_v2"
FAILURE_SCHEMA = "lineage_probe_admission_failure_v2"
SOURCE_FILES = (
    "evals/lineage_probe_admission_v2.py",
    "tests/test_lineage_probe_admission_v2.py",
    "evals/lineage_gpu_wait_v1.py",
    "evals/lineage_diagnosis_epoch_v1.py",
    "evals/lineage_probe_backend_v1.py",
)
_LOCK = RLock()
_LOCAL = threading.local()
_LEASE_CHECK = GPU_WAIT.B.lease_context
_GPU_QUERY = GPU_WAIT.GPU.query_gpu
_CLOCK = time.monotonic
_SLEEP = time.sleep


class ProbeAdmissionBlocked(BaseException):
    """GPU 입장 증거가 없어 epoch와 상위 batch를 중단한다."""


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def _sha(path: Path, *, lf: bool = False) -> str:
    raw = Path(path).read_bytes()
    if lf:
        raw = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(raw).hexdigest()


def _read(path: Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"),
                       parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)))
    _require(isinstance(value, dict), f"{Path(path).name}은 JSON 객체여야 한다")
    return value


def _write_new(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                     allow_nan=False).encode("utf-8") + b"\n"
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def _relative(path: Path) -> str:
    try:
        value = Path(path).resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError as error:
        raise ValueError("protocol은 저장소 내부 파일이어야 한다") from error
    _require(value and ".." not in Path(value).parts and not Path(value).is_absolute(),
             "protocol 상대경로가 안전하지 않다")
    return value


def _git(*args: str) -> bytes:
    result = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, check=False)
    if result.returncode:
        raise ValueError("Git commit/blob 결합을 확인하지 못했다")
    return result.stdout


def prepare_protocol(probe_protocol: Path) -> dict:
    """기존 probe protocol을 그대로 두고 입장 wrapper 결합만 만든다."""
    path = Path(probe_protocol).resolve()
    value = _read(path)
    _require(value.get("schema") == P.PROTOCOL_SCHEMA, "기존 probe protocol schema가 다르다")
    component = {"path": _relative(path), "sha256_lf": _sha(path, lf=True)}
    return {
        "schema": SCHEMA,
        "phase": "gpu_admission_before_probe_prepare_and_claim",
        "component_probe_protocol": component,
        "admission": GPU_WAIT.policy(),
        "source_sha256_lf": {name: _sha(ROOT / name, lf=True) for name in SOURCE_FILES},
        "failure_contract": {
            "exception": "ProbeAdmissionBlocked(BaseException)",
            "epoch_and_batch_stop": True,
            "post_block_roles": 0,
            "gpu_jobs_started": 0,
            "write_once_private_receipts": True,
        },
        "claims": {"measurement": False, "correction": False, "recovery": False},
    }


def verify_protocol(protocol_path: Path, *, anchor: str) -> dict:
    """새 공개 wrapper와 실행 source가 같은 등록 commit에 있음을 확인한다."""
    _require(isinstance(anchor, str) and re.fullmatch(r"[0-9a-f]{40}", anchor) is not None,
             "registration anchor 형식이 다르다")
    path = Path(protocol_path).resolve()
    value = _read(path)
    component = value.get("component_probe_protocol")
    _require(isinstance(component, dict) and set(component) == {"path", "sha256_lf"},
             "기존 probe protocol 결합이 없다")
    expected = prepare_protocol(ROOT / str(component["path"]))
    _require(value == expected, "wrapper protocol 내용 또는 source 결합이 다르다")
    _git("cat-file", "-e", f"{anchor}^{{commit}}")
    _git("merge-base", "--is-ancestor", anchor, "HEAD")
    bound = {_relative(path): _sha(path, lf=True), component["path"]: component["sha256_lf"],
             **value["source_sha256_lf"]}
    for relative, expected_sha in bound.items():
        current = ROOT / relative
        _require(current.is_file() and _sha(current, lf=True) == expected_sha,
                 f"현재 source/blob이 다르다: {relative}")
        committed = _git("show", f"{anchor}:{relative}").replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        _require(hashlib.sha256(committed).hexdigest() == expected_sha,
                 f"anchor source/blob이 다르다: {relative}")
    return deepcopy(value)


def _wait_stage(stage: Path, phase: str) -> Path:
    suffix = "admission-before-prepare" if phase == "prepare" else "admission-before-claim"
    return Path(stage).resolve().parent / f"{Path(stage).name}-{suffix}"


def _admit(stage: Path, phase: str) -> dict:
    stage = Path(stage).resolve()
    stage.mkdir(parents=True, exist_ok=False)
    try:
        result = GPU_WAIT.await_clear(
            stage, max_wait_s=600, interval_s=2, consecutive_clear=2,
            lease_check=_LEASE_CHECK, query=_GPU_QUERY, clock=_CLOCK, sleep=_SLEEP,
        )
    except Exception as error:
        wait_result = stage / "gpu-wait" / "result.json"
        wait_receipt = {
            "schema": WAIT_SCHEMA,
            "status": "blocked",
            "phase": phase,
            "gpu_jobs_started": 0,
            "underlying_result": ({"path": str(wait_result), "sha256": _sha(wait_result)}
                                  if wait_result.is_file() else None),
        }
        wait_path = stage / "probe-admission-wait-receipt.json"
        _write_new(wait_path, wait_receipt)
        failure = {
            "schema": FAILURE_SCHEMA,
            "status": "blocked",
            "phase": phase,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "wait_receipt": {"path": str(wait_path), "sha256": _sha(wait_path)},
            "claim_created": False,
            "owned_processes_started": 0,
            "post_block_roles_started": 0,
        }
        _write_new(stage / "probe-admission-failure.json", failure)
        raise ProbeAdmissionBlocked(f"{phase}: {type(error).__name__}: {error}") from error
    wait_result = Path(result["receipt_path"])
    receipt = {
        "schema": WAIT_SCHEMA,
        "status": "clear",
        "phase": phase,
        "gpu_jobs_started": 0,
        "underlying_result": {"path": str(wait_result), "sha256": _sha(wait_result)},
    }
    _write_new(stage / "probe-admission-wait-receipt.json", receipt)
    return receipt


def prepare_probe(collected_input: Path, probe: str, *, stage: Path, lab: Path,
                  protocol_path: Path, anchor: str, role: str = "case") -> dict:
    """원장 승인 전에 GPU clear 두 번을 확인하고 기존 plan 생성을 호출한다."""
    _admit(_wait_stage(Path(stage), "prepare"), "prepare")
    return P.prepare_probe(collected_input, probe, stage=stage, lab=lab,
                           protocol_path=protocol_path, anchor=anchor, role=role)


def execute_probe(plan_path: Path, authorization_path: Path, *, owned=None,
                  check_lease=GPU_WAIT.B.lease_context, check_gpu=None) -> dict:
    """execution claim 전에 GPU clear 두 번을 다시 확인한다."""
    plan = _read(Path(plan_path))
    stage = Path(str(plan.get("stage", "")))
    _require(stage.is_absolute(), "probe plan stage가 절대경로가 아니다")
    _admit(_wait_stage(stage, "execute"), "execute")
    return P.execute_probe(plan_path, authorization_path, owned=owned,
                           check_lease=check_lease, check_gpu=check_gpu)


@contextmanager
def adapter_scope():
    """현재 thread에서만 E.P를 어댑터 모듈로 바꾸고 원상 복구한다."""
    module = sys.modules[__name__]
    with _LOCK:
        depth = getattr(_LOCAL, "depth", 0)
        if depth:
            _require(E.P is module, "재진입 중 epoch probe adapter가 바뀌었다")
            _LOCAL.depth = depth + 1
            try:
                yield
            finally:
                _LOCAL.depth -= 1
            return
        _require(E.P is P, "epoch probe backend가 등록 원본이 아니다")
        original = E.P
        E.P = module
        _LOCAL.depth = 1
        try:
            yield
        finally:
            _LOCAL.depth = 0
            E.P = original


def run_epoch(*args, wrapper_protocol: Path, anchor: str, **kwargs) -> dict:
    """wrapper 결합을 확인한 동안에만 기존 epoch를 어댑터로 실행한다."""
    wrapper = verify_protocol(Path(wrapper_protocol), anchor=anchor)
    component = (ROOT / wrapper["component_probe_protocol"]["path"]).resolve()
    probe_protocol = Path(kwargs.get("probe_protocol", "")).resolve()
    _require(probe_protocol == component, "epoch probe protocol이 wrapper component와 다르다")
    with adapter_scope():
        return E.run_epoch(*args, anchor=anchor, **kwargs)
