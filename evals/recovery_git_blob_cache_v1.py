"""R4 CPU 검증 중 immutable Git blob 조회만 scope 안에서 재사용한다.

현재 파일, checkpoint, receipt, runtime, lease, 승인과 실행 결과는 캐시하지 않는다.
"""
# pyright: reportMissingImports=false
from __future__ import annotations

import re
import subprocess
import sys
import threading
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "bench"), str(ROOT / "evals")]

import five_policy_recovery_candidate_v1 as CANDIDATE
import lineage_five_policy_nim_transport_v1 as TRANSPORT
import lineage_probe_backend_v1 as PROBE
import lineage_same_play_capture_v1 as CAPTURE
import lineage_same_play_pair_inputs_v1 as PAIR_INPUTS
import lineage_source_training_v2 as SOURCE_TRAINING
import residual_training_v1 as RESIDUAL
import runtime_observation_view_v1 as VIEW

ALLOWED_MODULE_FILES = frozenset({
    "bench/lineage_same_play_capture_v1.py",
    "bench/lineage_source_training_v2.py",
    "bench/residual_training_v1.py",
    "evals/five_policy_recovery_candidate_v1.py",
    "evals/lineage_five_policy_nim_transport_v1.py",
    "evals/lineage_probe_backend_v1.py",
    "evals/lineage_same_play_pair_inputs_v1.py",
    "evals/runtime_observation_view_v1.py",
})
DEFAULT_MODULES = (
    CAPTURE,
    PAIR_INPUTS,
    PROBE,
    RESIDUAL,
    SOURCE_TRAINING,
    VIEW,
    TRANSPORT,
    CANDIDATE,
)
IMMUTABLE_SPEC = re.compile(r"([0-9a-f]{40}):(.+)\Z")
_LOCK = threading.RLock()


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def _module_relative(module: ModuleType, repo: Path) -> str:
    source = Path(str(getattr(module, "__file__", ""))).resolve(strict=True)
    try:
        relative = source.relative_to(repo).as_posix()
    except ValueError as error:
        raise ValueError("source proof module이 명시 repo 밖에 있다") from error
    _require(relative in ALLOWED_MODULE_FILES, "허용되지 않은 source proof module")
    _require(Path(getattr(module, "ROOT", "")).resolve() == repo,
             "module ROOT와 명시 repo identity가 다르다")
    return relative


def _safe_spec(value: object) -> tuple[str, str] | None:
    if not isinstance(value, str):
        return None
    match = IMMUTABLE_SPEC.fullmatch(value)
    if match is None:
        return None
    commit, relative = match.groups()
    raw_parts = relative.split("/")
    path = PurePosixPath(relative)
    if (path.is_absolute() or not path.parts or relative.startswith("./")
            or "\\" in relative or ":" in relative or ".." in path.parts
            or any(part in {"", ".", ".."} for part in raw_parts)):
        return None
    return commit, relative


def _repo_arg(value: object, repo: Path) -> bool:
    if not isinstance(value, (str, Path)):
        return False
    return Path(value).resolve() == repo


def _immutable_invocation(
    args: tuple[Any, ...], kwargs: Mapping[str, Any], repo: Path,
) -> tuple[str, str] | None:
    """두 등록형 ``git show`` 중 정확히 immutable blob인 호출만 분류한다."""
    if len(args) != 1 or not isinstance(args[0], Sequence) or isinstance(args[0], str):
        return None
    argv = tuple(args[0])
    allowed = {
        "cwd", "check", "capture_output", "text", "encoding", "errors",
        "universal_newlines", "timeout",
    }
    if set(kwargs) - allowed:
        return None
    bool_options = {"check", "capture_output", "text", "universal_newlines"}
    if any(name in kwargs and type(kwargs[name]) is not bool for name in bool_options):
        return None
    if any(name in kwargs and kwargs[name] is not None
           and not isinstance(kwargs[name], str) for name in ("encoding", "errors")):
        return None
    if "timeout" in kwargs and kwargs["timeout"] is not None and (
            isinstance(kwargs["timeout"], bool)
            or not isinstance(kwargs["timeout"], (int, float))):
        return None
    cwd = kwargs.get("cwd")
    if len(argv) == 5 and argv[:3] == ("git", "-C", str(repo)) and argv[3] == "show":
        if cwd is not None and not _repo_arg(cwd, repo):
            return None
        return _safe_spec(argv[4])
    if len(argv) == 3 and argv[:2] == ("git", "show"):
        if cwd is None or not _repo_arg(cwd, repo):
            return None
        return _safe_spec(argv[2])
    return None


def _freeze(value: object) -> object:
    if isinstance(value, Path):
        return ("Path", str(value.resolve()))
    if isinstance(value, Mapping):
        return tuple(sorted((str(key), _freeze(item)) for key, item in value.items()))
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (str, int, float, bool, bytes, type(None))):
        return value
    return (type(value).__qualname__, repr(value))


def _successful(value: object) -> bool:
    return (isinstance(value, subprocess.CompletedProcess)
            and value.returncode == 0
            and isinstance(value.stdout, (str, bytes)))


class _SubprocessProxy:
    def __init__(self, original: ModuleType, relative: str, repo: Path,
                 memoize: bool, cache: dict[tuple[object, ...], object],
                 stats: dict[str, Any]) -> None:
        self._original = original
        self._relative = relative
        self._repo = repo
        self._memoize = memoize
        self._cache = cache
        self._stats = stats

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)

    def run(self, *args: Any, **kwargs: Any) -> Any:
        immutable = _immutable_invocation(args, kwargs, self._repo)
        if immutable is None:
            self._stats["uncached_calls"] += 1
            return self._original.run(*args, **kwargs)
        key = (self._relative, _freeze(args), _freeze(kwargs), *immutable)
        if self._memoize and key in self._cache:
            self._stats["hits"] += 1
            return deepcopy(self._cache[key])
        self._stats["misses"] += 1
        self._stats["git_show_calls"] += 1
        value = self._original.run(*args, **kwargs)
        if self._memoize and _successful(value):
            self._cache[key] = deepcopy(value)
        return deepcopy(value)


@contextmanager
def proof_scope(
    modules: tuple[ModuleType, ...] = DEFAULT_MODULES, *, repo: Path = ROOT,
    memoize: bool = True,
) -> Iterator[dict[str, Any]]:
    """허용 모듈의 immutable ``git show`` 성공만 현재 검증 scope에서 메모한다."""
    repo = Path(repo).resolve(strict=True)
    _require(type(memoize) is bool, "memoize는 bool이어야 한다")
    _require(bool(modules) and len({id(module) for module in modules}) == len(modules),
             "module allowlist가 비었거나 같은 객체가 중복됐다")
    with _LOCK:
        cache: dict[tuple[object, ...], object] = {}
        stats: dict[str, Any] = {
            "schema": "recovery_git_blob_cache_stats_v1",
            "memoize": memoize,
            "hits": 0,
            "misses": 0,
            "git_show_calls": 0,
            "uncached_calls": 0,
            "cached_entries": 0,
            "approval_granted": False,
            "execution_allowed": False,
            "cached_material": False,
        }
        originals: list[tuple[ModuleType, ModuleType, _SubprocessProxy]] = []
        try:
            for module in modules:
                _require(isinstance(module, ModuleType), "module 객체가 필요하다")
                relative = _module_relative(module, repo)
                original = getattr(module, "subprocess", None)
                _require(original is subprocess, "module subprocess가 표준 모듈 identity와 다르다")
                proxy = _SubprocessProxy(original, relative, repo, memoize, cache, stats)
                module.__dict__["subprocess"] = proxy
                originals.append((module, original, proxy))
            yield stats
        finally:
            changed = []
            for module, original, proxy in reversed(originals):
                if getattr(module, "subprocess", None) is not proxy:
                    changed.append(_module_relative(module, repo))
                module.__dict__["subprocess"] = original
            stats["cached_entries"] = len(cache)
            if changed:
                raise RuntimeError(f"cache scope 중 subprocess가 외부에서 바뀌었다: {sorted(changed)}")


scoped_cache = proof_scope

__all__ = ["ALLOWED_MODULE_FILES", "DEFAULT_MODULES", "proof_scope", "scoped_cache"]
