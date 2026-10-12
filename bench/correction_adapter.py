"""동결 결함 F1/F2/F6의 단일 표적을 원본 Isaac Lab 구현으로 되돌린다.

소스 계약은 파일 내용과 함수 바이트코드를 결합한다. 이 모듈은 모듈 전역이나
외부 상태 전체가 원본임을 증명하지 않으며, 등록된 세 표적만 검사하고 복원한다.
"""
from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
import hashlib
import importlib
import inspect
import json
import platform
from pathlib import Path
import sys
from types import CodeType, FunctionType, ModuleType
from typing import Any


TARGETS = {
    "restore_body_frame_reward": {
        "module": "isaaclab.envs.mdp.rewards",
        "qualname": "track_lin_vel_xy_exp",
        "source": "source/isaaclab/isaaclab/envs/mdp/rewards.py",
        "mode": "code",
    },
    "restore_motor_torque_speed": {
        "module": "isaaclab.actuators.actuator_pd",
        "qualname": "DCMotor._clip_effort",
        "source": "source/isaaclab/isaaclab/actuators/actuator_pd.py",
        "mode": "attribute",
    },
    "restore_configured_timeout": {
        "module": "isaaclab.envs.mdp.terminations",
        "qualname": "time_out",
        "source": "source/isaaclab/isaaclab/envs/mdp/terminations.py",
        "mode": "code",
    },
}


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _lf_bytes(path: Path) -> bytes:
    return path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def _constant(value: Any) -> Any:
    if isinstance(value, CodeType):
        return {"code": _code_payload(value)}
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return {"float": value.hex()}
    if isinstance(value, complex):
        return {"complex": [value.real.hex(), value.imag.hex()]}
    if isinstance(value, bytes):
        return {"bytes": value.hex()}
    if isinstance(value, tuple):
        return {"tuple": [_constant(item) for item in value]}
    if isinstance(value, frozenset):
        rows = [_constant(item) for item in value]
        rows.sort(key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")))
        return {"frozenset": rows}
    return {"type": type(value).__qualname__, "repr": repr(value)}


def _code_payload(code: CodeType) -> dict[str, Any]:
    """파일명·행 번호·디버그 테이블을 제외한 실행 의미 지문 재료."""
    payload = {
        "argcount": code.co_argcount,
        "posonlyargcount": code.co_posonlyargcount,
        "kwonlyargcount": code.co_kwonlyargcount,
        "nlocals": code.co_nlocals,
        "stacksize": code.co_stacksize,
        "flags": code.co_flags,
        "code": code.co_code.hex(),
        "consts": [_constant(value) for value in code.co_consts],
        "names": list(code.co_names),
        "varnames": list(code.co_varnames),
        "freevars": list(code.co_freevars),
        "cellvars": list(code.co_cellvars),
    }
    # Python 3.11부터 예외 처리 흐름은 바이트코드와 분리된 테이블에 저장된다.
    # 이를 빼면 try/except 의미가 다른 두 함수가 같은 지문을 가질 수 있다.
    if hasattr(code, "co_exceptiontable"):
        payload["exceptiontable"] = code.co_exceptiontable.hex()
    return payload


def fingerprint(code: CodeType | FunctionType) -> str:
    """위치 메타데이터에 의존하지 않는 함수 코드 지문을 만든다."""
    if isinstance(code, FunctionType):
        payload = {"code": _code_payload(code.__code__), "closure_names": list(code.__code__.co_freevars)}
    elif isinstance(code, CodeType):
        payload = {"code": _code_payload(code), "closure_names": list(code.co_freevars)}
    else:
        raise TypeError("fingerprint requires a function or code object")
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return _sha256(raw)


def _find_code(parent: CodeType, qualname: str) -> CodeType:
    """Python 3.10에서도 동작하도록 중첩 code의 이름 경로를 순서대로 찾는다."""
    current = parent
    for segment in qualname.split("."):
        matches = [value for value in current.co_consts
                   if isinstance(value, CodeType) and value.co_name == segment]
        if len(matches) != 1:
            raise ValueError(f"source must contain exactly one code object segment {segment!r} for {qualname!r}")
        current = matches[0]
    return current


def _runtime_binding() -> dict[str, Any]:
    version = sys.version_info
    return {
        "implementation": platform.python_implementation(),
        "version": [version.major, version.minor, version.micro],
        "cache_tag": sys.implementation.cache_tag,
    }


def source_contract(isaaclab_root: Path) -> dict[str, Any]:
    """Isaac Lab 소스를 import하지 않고 세 원본 표적의 계약을 만든다."""
    root = Path(isaaclab_root).resolve()
    targets: dict[str, dict[str, Any]] = {}
    for correction_id, spec in TARGETS.items():
        path = (root / spec["source"]).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError("target source escapes the Isaac Lab root") from error
        raw = _lf_bytes(path)
        compiled = compile(raw, spec["source"], "exec", dont_inherit=True)
        clean_code = _find_code(compiled, spec["qualname"])
        targets[correction_id] = {
            **spec,
            "source_sha256": _sha256(raw),
            "clean_fingerprint": fingerprint(clean_code),
        }
    return {"schema": "correction_source_contract_v1", "runtime": _runtime_binding(), "targets": targets}


def _validate_contract(root: Path, contract: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if not isinstance(contract, dict) or contract.get("schema") != "correction_source_contract_v1":
        raise ValueError("invalid correction source contract schema")
    if contract.get("runtime") != _runtime_binding():
        raise ValueError("correction source contract runtime differs from the executing Python runtime")
    rows = contract.get("targets")
    if not isinstance(rows, dict) or set(rows) != set(TARGETS):
        raise ValueError("correction source contract target set differs")
    for correction_id, static in TARGETS.items():
        row = rows[correction_id]
        if not isinstance(row, dict) or any(row.get(key) != value for key, value in static.items()):
            raise ValueError("correction source contract target definition differs")
        path = (root / static["source"]).resolve()
        raw = _lf_bytes(path)
        if row.get("source_sha256") != _sha256(raw):
            raise ValueError("correction source contract differs from the current Isaac Lab source")
        if not isinstance(row.get("clean_fingerprint"), str) or len(row["clean_fingerprint"]) != 64:
            raise ValueError("correction source contract clean fingerprint is invalid")
        compiled = compile(raw, static["source"], "exec", dont_inherit=True)
        clean_code = _find_code(compiled, static["qualname"])
        if row["clean_fingerprint"] != fingerprint(clean_code):
            raise ValueError("correction clean fingerprint differs from the bound source")
    return rows


def _owner_and_name(module: ModuleType, qualname: str) -> tuple[Any, str]:
    parts = qualname.split(".")
    owner: Any = module
    for part in parts[:-1]:
        owner = inspect.getattr_static(owner, part)
    return owner, parts[-1]


def _raw_target(module: ModuleType, qualname: str) -> tuple[Any, str, FunctionType]:
    owner, name = _owner_and_name(module, qualname)
    value = inspect.getattr_static(owner, name)
    if not isinstance(value, FunctionType):
        raise ValueError(f"registered target {qualname!r} is not a plain function")
    return owner, name, value


@dataclass
class _SavedTarget:
    module: ModuleType
    qualname: str
    owner: Any
    name: str
    function: FunctionType
    pristine: CodeType | FunctionType
    source_path: Path
    source_sha256: str
    clean_fingerprint: str


@dataclass
class Snapshot:
    root: Path
    contract: dict[str, Any]
    targets: dict[str, _SavedTarget]
    applied: bool = False


def capture(isaaclab_root: Path, contract: dict[str, Any]) -> Snapshot:
    """결함 주입 전에 live 표적 세 개를 검증하고 복원용 객체를 보존한다."""
    root = Path(isaaclab_root).resolve()
    rows = _validate_contract(root, contract)
    saved: dict[str, _SavedTarget] = {}
    for correction_id, spec in rows.items():
        module = importlib.import_module(spec["module"])
        module_file = getattr(module, "__file__", None)
        if not module_file:
            raise ValueError(f"live module {spec['module']!r} has no source file")
        actual_path = Path(module_file).resolve()
        expected_path = (root / spec["source"]).resolve()
        if actual_path != expected_path:
            raise ValueError("live module path differs from the bound Isaac Lab source")
        if _sha256(_lf_bytes(actual_path)) != spec["source_sha256"]:
            raise ValueError("live module source hash differs from the source contract")
        owner, name, function = _raw_target(module, spec["qualname"])
        if fingerprint(function) != spec["clean_fingerprint"]:
            raise ValueError("live target is not the clean implementation")
        pristine: CodeType | FunctionType = function.__code__ if spec["mode"] == "code" else function
        saved[correction_id] = _SavedTarget(
            module=module, qualname=spec["qualname"], owner=owner, name=name,
            function=function, pristine=pristine,
            source_path=actual_path, source_sha256=spec["source_sha256"],
            clean_fingerprint=spec["clean_fingerprint"],
        )
    return Snapshot(root=root, contract=deepcopy(contract), targets=saved)


def _current(saved: _SavedTarget) -> FunctionType:
    owner, name = _owner_and_name(saved.module, saved.qualname)
    if owner is not saved.owner or name != saved.name:
        raise ValueError("registered target owner changed after capture")
    value = inspect.getattr_static(owner, name)
    if not isinstance(value, FunctionType):
        raise ValueError("registered target slot no longer contains a plain function")
    return value


def _source_binding(snapshot: Snapshot) -> dict[str, dict[str, str]]:
    binding: dict[str, dict[str, str]] = {}
    for correction_id, saved in snapshot.targets.items():
        if Path(saved.module.__file__).resolve() != saved.source_path:
            raise ValueError("live module path changed after capture")
        actual_sha = _sha256(_lf_bytes(saved.source_path))
        if actual_sha != saved.source_sha256:
            raise ValueError("live module source changed after capture")
        binding[correction_id] = {
            "module": TARGETS[correction_id]["module"],
            "qualname": TARGETS[correction_id]["qualname"],
            "source": TARGETS[correction_id]["source"],
            "source_sha256": actual_sha,
            "expected_source_path": str(saved.source_path),
            "live_source_path": str(Path(saved.module.__file__).resolve()),
            "clean_fingerprint": saved.clean_fingerprint,
        }
    return binding


def apply(snapshot: Snapshot, correction_id: str | None) -> dict[str, Any]:
    """현재 결함 상태를 읽고 선택한 표적 하나만 원본 구현으로 복원한다."""
    if not isinstance(snapshot, Snapshot):
        raise TypeError("apply requires a capture snapshot")
    if snapshot.applied:
        raise RuntimeError("correction snapshot was already applied")
    if correction_id is not None and correction_id not in TARGETS:
        raise ValueError(f"unknown correction id {correction_id!r}")

    binding = _source_binding(snapshot)
    clean = {key: saved.clean_fingerprint for key, saved in snapshot.targets.items()}
    before_apply: dict[str, str] = {}
    for key, saved in snapshot.targets.items():
        current = _current(saved)
        before_apply[key] = fingerprint(current)
        if TARGETS[key]["mode"] == "code" and current is not saved.function:
            raise ValueError("code-mode target slot was replaced instead of mutated in place")

    if correction_id is not None:
        saved = snapshot.targets[correction_id]
        if TARGETS[correction_id]["mode"] == "code":
            assert isinstance(saved.pristine, CodeType)
            saved.function.__code__ = saved.pristine
        else:
            assert isinstance(saved.pristine, FunctionType)
            setattr(saved.owner, saved.name, saved.pristine)

    after_apply = {key: fingerprint(_current(saved)) for key, saved in snapshot.targets.items()}
    for key in TARGETS:
        if key != correction_id and after_apply[key] != before_apply[key]:
            raise RuntimeError("an unselected correction target changed")
    changed = [key for key in TARGETS if before_apply[key] != after_apply[key]]
    restored = [key for key in changed if after_apply[key] == clean[key]]
    snapshot.applied = True
    return {
        "schema": "correction_readback_v1",
        "selected_correction": correction_id,
        "before_fault": clean,
        "after_fault": before_apply,
        "after_correction": after_apply,
        "changed_targets": changed,
        "restored_targets": restored,
        "source_binding": binding,
    }
