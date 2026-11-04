"""Registered correction table for diagnostic mechanisms with executable recovery paths."""
from __future__ import annotations

from copy import deepcopy

from rl_triage.correction_registry import REGISTRY as LEGACY_REGISTRY
from rl_triage.correction_registry import registry_digest as _digest

_OPTIMIZER = {
    "optimizer_update_count": {
        "correction_id": "restore_configured_ppo_epochs",
        "adapter_target": "rsl_rl.algorithms.ppo:PPO.__init__",
        "protocol_case": "optimizer_update_count",
        "recovery_modes": ["restart"],
    },
    "optimizer_surrogate": {
        "correction_id": "restore_clipped_ppo_surrogate",
        "adapter_target": "rsl_rl.algorithms.ppo:PPO.update",
        "protocol_case": "optimizer_surrogate",
        "recovery_modes": ["restart"],
    },
}

REGISTRY = {**deepcopy(LEGACY_REGISTRY), **deepcopy(_OPTIMIZER)}


def registry() -> dict:
    """Return a caller-mutable copy; module state remains immutable by convention."""
    return deepcopy(REGISTRY)


def registry_digest(value: dict | None = None) -> str:
    return _digest(REGISTRY if value is None else value)


def correction_for(mechanism: str) -> dict:
    try:
        return deepcopy(REGISTRY[mechanism])
    except KeyError as exc:
        raise ValueError(f"등록되지 않은 메커니즘: {mechanism}") from exc


def assert_legacy_preserved() -> None:
    if {key: REGISTRY[key] for key in LEGACY_REGISTRY} != LEGACY_REGISTRY:
        raise RuntimeError("기존 교정 등록 표가 변경됐다")


assert_legacy_preserved()
