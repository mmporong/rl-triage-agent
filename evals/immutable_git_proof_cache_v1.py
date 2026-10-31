"""CPU 검증 한 구간에서 immutable Git blob 조회만 메모한다.

실행 권한, 승인, HEAD/index 스냅샷 또는 mutable Git 조회를 만들지 않는다.
"""
# pyright: reportMissingImports=false
from __future__ import annotations

import re
import subprocess
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "bench"), str(ROOT / "evals")]

import lineage_probe_backend_v1 as PROBE
import lineage_recovery_execution_v1 as EXECUTION
import residual_training_v1 as RESIDUAL
import runtime_observation_training_v1 as TRAINING
import runtime_observation_view_v1 as VIEW

ALLOWED_MODULES = frozenset({
    "residual_training_v1",
    "lineage_recovery_execution_v1",
    "lineage_probe_backend_v1",
    "runtime_observation_training_v1",
    "runtime_observation_view_v1",
})
EXPORT_MODULES = (RESIDUAL, EXECUTION, PROBE, TRAINING, VIEW)
IMMUTABLE_SPEC = re.compile(r"([0-9a-f]{40}):(.+)\Z")
_LOCK = threading.RLock()


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def _immutable_show(args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[str, str] | None:
    if len(args) != 2 or args[0] != "show" or set(kwargs) - {"check"}:
        return None
    if "check" in kwargs and type(kwargs["check"]) is not bool:
        return None
    if not isinstance(args[1], str):
        return None
    match = IMMUTABLE_SPEC.fullmatch(args[1])
    if match is None:
        return None
    commit, relative = match.groups()
    path = PurePosixPath(relative)
    if (path.is_absolute() or not path.parts or relative.startswith("./")
            or "\\" in relative or ":" in relative or ".." in path.parts):
        return None
    return commit, relative


def _successful(value: object) -> bool:
    if isinstance(value, bytes):
        return True
    if isinstance(value, subprocess.CompletedProcess):
        return value.returncode == 0 and isinstance(value.stdout, str)
    return False


@contextmanager
def proof_scope(
    modules: tuple[ModuleType, ...] = EXPORT_MODULES, *, repo: Path = ROOT,
    memoize: bool = True,
) -> Iterator[dict[str, Any]]:
    """명시 모듈의 immutable ``git show`` 성공 결과만 이 구간 안에서 재사용한다."""
    repo = Path(repo).resolve()
    _require(type(memoize) is bool, "memoize는 bool이어야 한다")
    _require(bool(modules) and len(set(modules)) == len(modules), "module allowlist가 비었거나 중복됐다")
    with _LOCK:
        originals: list[tuple[ModuleType, Any]] = []
        wrappers: list[tuple[ModuleType, Any]] = []
        cache: dict[tuple[Any, ...], object] = {}
        stats: dict[str, Any] = {
            "schema": "immutable_git_proof_cache_stats_v1",
            "memoize": memoize, "hits": 0, "misses": 0,
            "git_show_calls": 0, "mutable_calls": 0,
            "approval": False, "execution_allowed": False,
        }
        for module in modules:
            _require(isinstance(module, ModuleType) and module.__name__ in ALLOWED_MODULES,
                     "허용되지 않은 source proof module")
            module_root = Path(getattr(module, "ROOT", "")).resolve()
            _require(module_root == repo, "module ROOT와 명시 repo identity가 다르다")
            original = getattr(module, "_git", None)
            _require(callable(original), "module의 원래 _git 함수가 없다")
            originals.append((module, original))

        try:
            for module, original in originals:
                def wrapper(*args: Any, __module: ModuleType = module,
                            __original: Any = original, **kwargs: Any) -> Any:
                    immutable = _immutable_show(args, kwargs)
                    if immutable is None:
                        stats["mutable_calls"] += 1
                        return __original(*args, **kwargs)
                    commit, relative = immutable
                    key = (__module.__name__, id(__original), args,
                           tuple(sorted(kwargs.items())), str(repo), commit, relative)
                    if memoize and key in cache:
                        stats["hits"] += 1
                        return deepcopy(cache[key])
                    stats["misses"] += 1
                    stats["git_show_calls"] += 1
                    value = __original(*args, **kwargs)
                    if memoize and _successful(value):
                        cache[key] = deepcopy(value)
                    return deepcopy(value)

                module.__dict__["_git"] = wrapper
                wrappers.append((module, wrapper))
            yield stats
        finally:
            changed = []
            for (module, original), (_wrapped_module, wrapper) in zip(
                    reversed(originals), reversed(wrappers), strict=True):
                if getattr(module, "_git", None) is not wrapper:
                    changed.append(module.__name__)
                module.__dict__["_git"] = original
            if changed:
                raise RuntimeError(f"proof scope 중 _git 함수가 외부에서 바뀌었다: {sorted(changed)}")


__all__ = ["ALLOWED_MODULES", "EXPORT_MODULES", "proof_scope"]
