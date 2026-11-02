"""Runtime view 안의 source 계보 검증만 legacy selection view로 격리한다."""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from threading import RLock
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("evals", "src")]

import five_policy_recovery_epoch2_v1 as SECOND
import runtime_observation_view_v1 as RV

_LOCK = RLock()
_VIEW = RV.V
_LEGACY = (_VIEW.BASE_FACT_IDS, _VIEW.PROTOCOL, _VIEW.verify_protocol)
_ORIGINAL_CHECKED_SOURCE = SECOND.checked_source
_ORIGINAL_PROBE_SOURCE = SECOND._PROBE._source


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def _view_state() -> tuple[object, Path, Callable]:
    return (_VIEW.BASE_FACT_IDS, Path(_VIEW.PROTOCOL), _VIEW.verify_protocol)


def _is_legacy(state: tuple[object, Path, Callable]) -> bool:
    return (
        state[0] == _LEGACY[0]
        and state[1].resolve() == Path(_LEGACY[1]).resolve()
        and state[2] is _LEGACY[2]
    )


def _is_registered_runtime(state: tuple[object, Path, Callable]) -> bool:
    if state[0] != RV.BASE_FACT_IDS:
        return False
    try:
        verified = state[2](RV.file_digest(state[1]))
    except (OSError, TypeError, ValueError):
        return False
    return isinstance(verified, dict) and verified.get("schema") == RV.SCHEMA


@contextmanager
def legacy_validation_scope() -> Iterator[None]:
    """등록 runtime view 중 source 검증 구간만 legacy 3-field 상태로 복원한다."""
    with RV._LOCK, _LOCK:
        current = _view_state()
        if _is_legacy(current):
            try:
                yield
            finally:
                if not _is_legacy(_view_state()):
                    _VIEW.BASE_FACT_IDS, _VIEW.PROTOCOL, _VIEW.verify_protocol = _LEGACY
                    raise RuntimeError("legacy source 검증 중 view가 외부에서 바뀌었다")
            return
        require(_is_registered_runtime(current), "알 수 없는 selection/runtime view 상태")
        _VIEW.BASE_FACT_IDS, _VIEW.PROTOCOL, _VIEW.verify_protocol = _LEGACY
        changed = False
        try:
            yield
        finally:
            changed = not _is_legacy(_view_state())
            _VIEW.BASE_FACT_IDS, _VIEW.PROTOCOL, _VIEW.verify_protocol = current
            if changed:
                raise RuntimeError("legacy source 검증 중 view가 외부에서 바뀌었다")


@contextmanager
def source_validation_scope() -> Iterator[None]:
    """SECOND source 재검증 함수 두 개에만 legacy view scope를 적용한다."""
    with _LOCK:
        require(SECOND.checked_source is _ORIGINAL_CHECKED_SOURCE,
                "SECOND.checked_source가 이미 바뀌었다")
        require(SECOND._PROBE._source is _ORIGINAL_PROBE_SOURCE,
                "SECOND probe source가 이미 바뀌었다")

        def checked_source(
            source: dict[str, Any],
            *,
            candidate_checker: Callable = SECOND.FIRST.checked_candidate,
        ) -> dict[str, Any]:
            with legacy_validation_scope():
                return cast(dict[str, Any], _ORIGINAL_CHECKED_SOURCE(
                    deepcopy(source), candidate_checker=candidate_checker))

        def probe_source(path: Path, lab: Path, role: str) -> tuple[dict, dict, dict]:
            with legacy_validation_scope():
                return cast(tuple[dict, dict, dict], _ORIGINAL_PROBE_SOURCE(
                    Path(path), Path(lab), role))

        SECOND.__dict__["checked_source"] = checked_source
        SECOND._PROBE.__dict__["_source"] = probe_source
        try:
            yield
        finally:
            current_checked = SECOND.checked_source
            current_probe = SECOND._PROBE._source
            SECOND.__dict__["checked_source"] = _ORIGINAL_CHECKED_SOURCE
            SECOND._PROBE.__dict__["_source"] = _ORIGINAL_PROBE_SOURCE
            if current_checked is not checked_source or current_probe is not probe_source:
                raise RuntimeError("source validation wrapper가 scope 중 외부에서 바뀌었다")
