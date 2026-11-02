from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "bench"), str(ROOT / "evals"), str(ROOT / "src")]

import recovery_outcome_contract_v1 as C

BANDS = json.loads(
    (ROOT / "evals/results/p0c_fixed_eval_20261009/summary.json").read_text(encoding="utf-8")
)["bands"]
REFERENCE = json.loads(
    (ROOT / "evals/results/brev_recovery_v4_s2196_20261024/source_fixed_none.json").read_text(
        encoding="utf-8"
    )
)
UNHEALTHY = json.loads(
    (ROOT / "evals/results/brev_recovery_v4_s2196_20261024/source_fixed_f1.json").read_text(
        encoding="utf-8"
    )
)


def _reports() -> dict[str, dict]:
    undetermined = deepcopy(REFERENCE)
    undetermined["horizon_steps"] //= 2
    return {
        "healthy": deepcopy(REFERENCE),
        "unhealthy": deepcopy(UNHEALTHY),
        "undetermined": undetermined,
    }


@pytest.mark.parametrize(
    ("label", "outcome"),
    [("healthy", "recovered"), ("unhealthy", "not_recovered"),
     ("undetermined", "unknown")],
)
def test_real_recovery_mapping_preserves_canonical_output(label: str, outcome: str) -> None:
    report = _reports()[label]
    canonical = C._CANONICAL_RECOVERY(report, REFERENCE, BANDS)
    normalized = C.normalized_recovery(report, REFERENCE, BANDS)

    assert canonical["verdict"]["label"] == label
    assert canonical["outcome"] == outcome
    assert "outcome" not in canonical["verdict"]
    assert normalized["verdict"]["outcome"] == outcome
    restored = deepcopy(normalized)
    del restored["verdict"]["outcome"]
    assert restored == canonical


def test_scope_repairs_the_actual_inner_outcome_consumer() -> None:
    before = C.GATE.RESIDUAL.recovery(REFERENCE, REFERENCE, BANDS)["verdict"]
    assert before.get("outcome") is None

    with C.recovery_outcome_scope():
        after = C.GATE.RESIDUAL.recovery(REFERENCE, REFERENCE, BANDS)["verdict"]
        assert after.get("outcome") == "recovered"

    assert C.GATE.RESIDUAL.recovery is C._CANONICAL_RECOVERY


@pytest.mark.parametrize(
    "raw",
    [
        {"verdict": {"label": "healthy"}, "outcome": "not_recovered"},
        {"verdict": {"label": "healthy", "outcome": "unknown"}, "outcome": "recovered"},
        {"verdict": {"label": "other"}, "outcome": "unknown"},
    ],
)
def test_rejects_label_or_outcome_conflicts(monkeypatch: pytest.MonkeyPatch, raw: dict) -> None:
    monkeypatch.setattr(C, "_CANONICAL_RECOVERY", lambda *_args: deepcopy(raw))
    with pytest.raises(ValueError, match="label|outcome"):
        C.normalized_recovery(REFERENCE, REFERENCE, BANDS)


def test_nested_and_exception_scope_restore_exact_original() -> None:
    original = C.GATE.RESIDUAL.recovery
    with pytest.raises(RuntimeError, match="stop"), C.recovery_outcome_scope():
        assert C.GATE.RESIDUAL.recovery is C.normalized_recovery
        with C.recovery_outcome_scope():
            assert C.GATE.RESIDUAL.recovery is C.normalized_recovery
        assert C.GATE.RESIDUAL.recovery is C.normalized_recovery
        raise RuntimeError("stop")
    assert C.GATE.RESIDUAL.recovery is original


def test_scope_deletes_injected_attribute_when_original_was_delegated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    residual = C.GATE.RESIDUAL
    original = residual.__dict__.pop("recovery")
    delegated = object()
    monkeypatch.setattr(
        residual,
        "__getattr__",
        lambda name: delegated if name == "recovery" else (_ for _ in ()).throw(AttributeError(name)),
        raising=False,
    )
    try:
        assert residual.recovery is delegated
        with C.recovery_outcome_scope():
            assert residual.recovery is C.normalized_recovery
        assert "recovery" not in residual.__dict__
        assert residual.recovery is delegated
    finally:
        residual.recovery = original
