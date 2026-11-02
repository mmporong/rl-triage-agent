"""동결 recovery 판정의 outer outcome을 inner verdict 소비 ABI에 연결한다."""
from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import lineage_evaluation_gate_v1 as GATE

SOURCE_FILES = (
    "evals/recovery_outcome_contract_v1.py",
    "tests/test_recovery_outcome_contract_v1.py",
)

_OUTCOME_BY_LABEL = {
    "healthy": "recovered",
    "unhealthy": "not_recovered",
    "undetermined": "unknown",
}
_CANONICAL_RECOVERY = GATE.RESIDUAL.recovery
_LOCK = threading.RLock()


def normalized_recovery(report: dict, reference: dict, bands: dict) -> dict:
    """원 판정 결과를 검증하고 inner verdict에 같은 outcome만 추가한다."""
    raw = _CANONICAL_RECOVERY(report, reference, bands)
    if type(raw) is not dict:
        raise ValueError("canonical recovery 결과가 객체가 아니다")
    verdict = raw.get("verdict")
    if type(verdict) is not dict:
        raise ValueError("canonical recovery verdict가 객체가 아니다")
    label = verdict.get("label")
    if label not in _OUTCOME_BY_LABEL:
        raise ValueError("canonical recovery label이 등록값이 아니다")
    outcome = raw.get("outcome")
    if type(outcome) is not str or outcome != _OUTCOME_BY_LABEL[label]:
        raise ValueError("canonical recovery label/outcome이 충돌한다")
    if "outcome" in verdict and verdict["outcome"] != outcome:
        raise ValueError("canonical recovery inner/outer outcome이 충돌한다")

    normalized = raw.copy()
    normalized_verdict = verdict.copy()
    normalized_verdict["outcome"] = outcome
    normalized["verdict"] = normalized_verdict
    return normalized


@contextmanager
def recovery_outcome_scope() -> Iterator[None]:
    """공유 gate recovery를 현재 범위에서만 호환 ABI로 바꾸고 정확히 복원한다."""
    residual = GATE.RESIDUAL
    with _LOCK:
        had_local = "recovery" in residual.__dict__
        previous: Any = residual.__dict__.get("recovery")
        residual.recovery = normalized_recovery
        try:
            yield
        finally:
            if had_local:
                residual.recovery = previous
            else:
                del residual.recovery
