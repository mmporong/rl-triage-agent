"""U1 보상 함수군 네 항을 하나의 등록 교정으로 원본 구현에 복원한다.

이 모듈은 기존 단일 표적 교정기를 바꾸지 않는다. 등록된 네 함수의 코드와 같은
모듈의 나머지 함수 지문을 함께 읽어, 선택 범위가 다르거나 부수 변경이 생기면
교정을 거부한다.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import platform
import sys
from types import CodeType, FunctionType, ModuleType
from typing import Any

import correction_adapter as BASE

CORRECTION_ID = "restore_reward_family_scope"
MODULE = "isaaclab.envs.mdp.rewards"
SOURCE = "source/isaaclab/isaaclab/envs/mdp/rewards.py"
FUNCTIONS = (
    "track_lin_vel_xy_exp",
    "track_ang_vel_z_exp",
    "lin_vel_z_l2",
    "ang_vel_xy_l2",
)
REGISTRY = {
    CORRECTION_ID: {
        "fault_family": "root_body_velocity_reward_family",
        "module": MODULE,
        "source": SOURCE,
        "qualnames": list(FUNCTIONS),
        "mode": "code_family",
    }
}


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _lf(path: Path) -> bytes:
    return path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def _runtime() -> dict[str, Any]:
    v = sys.version_info
    return {"implementation": platform.python_implementation(), "version": [v.major, v.minor, v.micro],
            "cache_tag": sys.implementation.cache_tag}


def _top_level_code(compiled: CodeType, name: str) -> CodeType:
    rows = [item for item in compiled.co_consts if isinstance(item, CodeType) and item.co_name == name]
    if len(rows) != 1:
        raise ValueError(f"source must contain exactly one top-level function {name!r}")
    return rows[0]


def source_contract(isaaclab_root: Path) -> dict[str, Any]:
    root = Path(isaaclab_root).resolve()
    path = (root / SOURCE).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError("reward source escapes the Isaac Lab root") from error
    raw = _lf(path)
    compiled = compile(raw, SOURCE, "exec", dont_inherit=True)
    return {
        "schema": "reward_family_source_contract_v1",
        "correction_id": CORRECTION_ID,
        "runtime": _runtime(),
        "module": MODULE,
        "source": SOURCE,
        "source_sha256": _sha(raw),
        "functions": {name: BASE.fingerprint(_top_level_code(compiled, name)) for name in FUNCTIONS},
    }


def _raw_function(module: ModuleType, name: str) -> FunctionType:
    value = inspect.getattr_static(module, name)
    if not isinstance(value, FunctionType):
        raise ValueError(f"registered reward target {name!r} is not a plain function")
    return value


def validate_contract(root: Path, contract: dict[str, Any]) -> None:
    expected = source_contract(root)
    if contract != expected:
        raise ValueError("reward family source contract differs from the current runtime or source")


@dataclass
class Snapshot:
    root: Path
    module: ModuleType
    source_path: Path
    contract: dict[str, Any]
    functions: dict[str, FunctionType]
    pristine: dict[str, CodeType]
    all_before: dict[str, str]
    applied: bool = False


def _module_fingerprints(module: ModuleType) -> dict[str, str]:
    return {name: BASE.fingerprint(value) for name, value in sorted(vars(module).items())
            if isinstance(value, FunctionType) and value.__module__ == module.__name__}


def capture(isaaclab_root: Path, contract: dict[str, Any]) -> Snapshot:
    root = Path(isaaclab_root).resolve()
    validate_contract(root, contract)
    module = importlib.import_module(MODULE)
    source_path = (root / SOURCE).resolve()
    if Path(getattr(module, "__file__", "")).resolve() != source_path:
        raise ValueError("live reward module path differs from the bound Isaac Lab source")
    functions = {name: _raw_function(module, name) for name in FUNCTIONS}
    if any(BASE.fingerprint(functions[name]) != contract["functions"][name] for name in FUNCTIONS):
        raise ValueError("live reward family is not the clean implementation")
    return Snapshot(root, module, source_path, json.loads(json.dumps(contract)), functions,
                    {name: function.__code__ for name, function in functions.items()}, _module_fingerprints(module))


def apply(snapshot: Snapshot, correction_id: str | None) -> dict[str, Any]:
    if not isinstance(snapshot, Snapshot):
        raise TypeError("apply requires a reward family snapshot")
    if snapshot.applied:
        raise RuntimeError("reward family snapshot was already applied")
    if correction_id not in (None, CORRECTION_ID):
        raise ValueError(f"unregistered reward family correction {correction_id!r}")
    if _sha(_lf(snapshot.source_path)) != snapshot.contract["source_sha256"]:
        raise ValueError("reward source changed after capture")
    for name, function in snapshot.functions.items():
        if _raw_function(snapshot.module, name) is not function:
            raise ValueError("registered reward function slot was replaced instead of mutated in place")

    injected_all = _module_fingerprints(snapshot.module)
    injected = {name: injected_all[name] for name in FUNCTIONS}
    unrelated_after_fault = sorted(name for name in set(snapshot.all_before) | set(injected_all)
                                   if name not in FUNCTIONS and snapshot.all_before.get(name) != injected_all.get(name))
    if unrelated_after_fault:
        raise RuntimeError("fault changed reward functions outside the registered family")

    if correction_id == CORRECTION_ID:
        for name, function in snapshot.functions.items():
            function.__code__ = snapshot.pristine[name]

    after_all = _module_fingerprints(snapshot.module)
    after = {name: after_all[name] for name in FUNCTIONS}
    unrelated_after_correction = sorted(name for name in set(injected_all) | set(after_all)
                                        if name not in FUNCTIONS and injected_all.get(name) != after_all.get(name))
    if unrelated_after_correction:
        raise RuntimeError("correction changed reward functions outside the registered family")
    clean = snapshot.contract["functions"]
    changed = [name for name in FUNCTIONS if injected[name] != after[name]]
    restored = [name for name in changed if after[name] == clean[name]]
    if correction_id == CORRECTION_ID and after != clean:
        raise RuntimeError("full reward family was not restored to the bound clean implementation")
    snapshot.applied = True
    return {
        "schema": "reward_family_correction_readback_v1",
        "selected_correction": correction_id,
        "registered_functions": list(FUNCTIONS),
        "before_fault": clean,
        "after_fault": injected,
        "after_correction": after,
        "changed_targets": changed,
        "restored_targets": restored,
        "unrelated_changed_after_fault": unrelated_after_fault,
        "unrelated_changed_after_correction": unrelated_after_correction,
        "source_binding": {"module": MODULE, "source": SOURCE, "source_sha256": snapshot.contract["source_sha256"],
                           "live_source_path": str(snapshot.source_path)},
    }


def environment_readback(env, contract: dict[str, Any]) -> dict[str, Any]:
    """환경 생성 뒤 실제 reward manager가 복원된 네 함수 객체를 쓰는지 확인한다."""
    module = importlib.import_module(MODULE)
    functions = {name: _raw_function(module, name) for name in FUNCTIONS}
    clean = {name: BASE.fingerprint(functions[name]) == contract["functions"][name] for name in FUNCTIONS}
    manager = env.unwrapped.reward_manager
    terms = {name: manager.get_term_cfg(name).func for name in manager.active_terms}
    bound = {name: terms.get(name) is functions[name] for name in FUNCTIONS}
    result = {"schema": "reward_family_environment_readback_v1", "clean": clean,
              "bound_to_reward_terms": bound, "ok": all(clean.values()) and all(bound.values())}
    if not result["ok"]:
        raise RuntimeError("restored reward family is not clean and bound in the environment")
    return result
