"""실행 기록의 공개 protocol 계획 참조와 실제 seed 사용을 구분한다."""
from __future__ import annotations

import hashlib
import os
import re
import threading
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import core_seed_guard_v1 as V1

SOURCE_FILES = (
    "evals/core_seed_guard_v2.py",
    "tests/test_core_seed_guard_v2.py",
)

_ORIGINAL_RECORD_SEEDS = V1.ORIGINAL._record_seeds
_PROTOCOL_KEY = re.compile(r"(?:^|_)protocol(?:_ref)?$")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_LOCK = threading.RLock()


def _lexical_public_protocol(value: Any, repo: Path) -> Path | None:
    if not isinstance(value, str) or not value or "\x00" in value:
        return None
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = repo / candidate
    candidate = Path(os.path.abspath(os.path.normpath(candidate)))
    protocol_root = Path(os.path.abspath(os.path.normpath(repo / "bench" / "protocols")))
    try:
        relative = candidate.relative_to(protocol_root)
    except ValueError:
        return None
    if not relative.parts or candidate.suffix != ".json":
        return None
    return candidate


def _verified_planning_reference(
    value: Any, key: str, repo: Path, seeds: set[int]
) -> Path | None:
    if not _PROTOCOL_KEY.search(key) or type(value) is not dict \
            or set(value) != {"path", "sha256"}:
        return None
    if not isinstance(value["path"], str) \
            or not V1.ORIGINAL._named_seeds(value["path"], seeds):
        return None
    candidate = _lexical_public_protocol(value["path"], repo)
    if candidate is None:
        return None
    expected = value["sha256"]
    if not isinstance(expected, str) or _SHA256.fullmatch(expected) is None:
        raise ValueError("계획 protocol 참조 SHA 형식이 잘못됐다")
    protocol_root = (repo / "bench" / "protocols").resolve(strict=True)
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(protocol_root)
    except (OSError, ValueError) as error:
        raise ValueError("계획 protocol 참조가 공개 protocol 파일이 아니다") from error
    if not resolved.is_file():
        raise ValueError("계획 protocol 참조가 파일이 아니다")
    actual = hashlib.sha256(resolved.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError("계획 protocol 참조 SHA가 원문과 다르다")
    return resolved


@contextmanager
def _planning_reference_scope(repo: Path) -> Iterator[list[str]]:
    verified: list[str] = []

    def record_seeds(value: Any, seeds: set[int], key: str = "") -> set[int]:
        protocol = _verified_planning_reference(value, key, repo, seeds)
        if protocol is not None:
            verified.append(protocol.relative_to(repo).as_posix())
            return set()
        return _ORIGINAL_RECORD_SEEDS(value, seeds, key)

    with _LOCK, V1._LOCK:
        old = V1.ORIGINAL._record_seeds
        V1.ORIGINAL._record_seeds = record_seeds
        try:
            yield verified
        finally:
            V1.ORIGINAL._record_seeds = old


def assert_fresh_seeds(
    seeds: Iterable[int],
    isaaclab: Path,
    *,
    repo: Path = V1.ORIGINAL.ROOT,
) -> dict:
    """v1 검사 전체를 유지하며 검증된 계획 protocol 파일명만 실행 seed에서 뺀다."""
    repo = Path(repo).resolve()
    with _planning_reference_scope(repo) as verified:
        result = V1.assert_fresh_seeds(seeds, isaaclab, repo=repo)
    return {
        **result,
        "schema": "core_seed_freshness_v2",
        "verified_planning_protocol_refs": sorted(set(verified)),
    }
