"""복합 F1/R1 결함과 두 등록 교정을 순서 독립적으로 적용한다."""
from __future__ import annotations

import hashlib
from types import CodeType
from typing import Any, Iterable, Mapping

import hidden_faults as HIDDEN
import optimizer_faults as OPTIMIZER

FAULTS = ("F1", "R1")
CORRECTIONS = {
    "restore_body_frame_reward": "F1",
    "restore_configured_ppo_epochs": "R1",
}


def normalize_faults(values: Iterable[str]) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        raise ValueError("original_faults는 배열이어야 한다")
    items = tuple(values)
    if len(set(items)) != len(items) or any(item not in FAULTS for item in items):
        raise ValueError("original_faults에는 중복 없는 F1/R1만 허용한다")
    return tuple(item for item in FAULTS if item in items)


def normalize_corrections(values: Iterable[str]) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        raise ValueError("applied_corrections는 배열이어야 한다")
    items = tuple(values)
    if len(items) > 2 or len(set(items)) != len(items) or any(item not in CORRECTIONS for item in items):
        raise ValueError("applied_corrections에는 서로 다른 등록 교정을 최대 두 개 허용한다")
    return tuple(items)


def remaining_faults(faults: Iterable[str], corrections: Iterable[str]) -> tuple[str, ...]:
    original = normalize_faults(faults)
    applied = normalize_corrections(corrections)
    removed = {CORRECTIONS[item] for item in applied}
    return tuple(item for item in original if item not in removed)


def code_sha256(code: CodeType) -> str:
    payload = b"\0".join((code.co_code, repr(code.co_consts).encode(), repr(code.co_names).encode()))
    return hashlib.sha256(payload).hexdigest()


def configure_reward(target: Any, faults: Iterable[str], corrections: Iterable[str]) -> dict[str, Any]:
    """현재 몸체 좌표 구현을 원본으로 잡고 F1 주입과 등록 복원을 적용한다."""
    original = target.__code__
    before = code_sha256(original)
    if "F1" in normalize_faults(faults):
        HIDDEN._swap_code(target, HIDDEN._track_lin_vel_world)
    after_fault = code_sha256(target.__code__)
    if "restore_body_frame_reward" in normalize_corrections(corrections):
        target.__code__ = original
    after = code_sha256(target.__code__)
    expected_fault = "F1" in remaining_faults(faults, corrections)
    if (after != before) != expected_fault:
        raise ValueError("reward 결함/교정 readback이 기대 상태와 다르다")
    return {
        "target": "isaaclab.envs.mdp.rewards.track_lin_vel_xy_exp",
        "before_code_sha256": before,
        "after_fault_code_sha256": after_fault,
        "after_code_sha256": after,
        "fault_present_after": expected_fault,
    }


def reward_manager_binding(env: Any, target: Any) -> dict[str, Any]:
    manager = env.unwrapped.reward_manager
    names = list(getattr(manager, "_term_names", ()))
    cfgs = list(getattr(manager, "_term_cfgs", ()))
    if len(names) != len(cfgs):
        raise ValueError("reward manager term 이름과 설정 수가 다르다")
    matches = [name for name, cfg in zip(names, cfgs) if getattr(cfg, "func", None) is target]
    if not matches:
        raise ValueError("track_lin_vel_xy_exp가 reward manager에 결합되지 않았다")
    return {"target_identity_matches": matches, "match_count": len(matches)}


def configure_ppo(ppo: Any, faults: Iterable[str], corrections: Iterable[str], *,
                  configured_epochs: int = 5, mini_batches: int = 4) -> dict[str, Any]:
    original = normalize_faults(faults)
    applied = normalize_corrections(corrections)
    if ppo.num_learning_epochs != configured_epochs or ppo.num_mini_batches != mini_batches:
        raise ValueError("PPO 적용 전 설정이 등록값과 다르다")
    before = {"learning_epochs": ppo.num_learning_epochs, "mini_batches": ppo.num_mini_batches}
    event = OPTIMIZER.apply("R1" if "R1" in original else "NONE", ppo)
    after_fault = {"learning_epochs": ppo.num_learning_epochs, "mini_batches": ppo.num_mini_batches}
    if "restore_configured_ppo_epochs" in applied:
        ppo.num_learning_epochs = configured_epochs
    after = {"learning_epochs": ppo.num_learning_epochs, "mini_batches": ppo.num_mini_batches}
    expected = 1 if "R1" in remaining_faults(original, applied) else configured_epochs
    if after != {"learning_epochs": expected, "mini_batches": mini_batches}:
        raise ValueError("optimizer 결함/교정 readback이 기대 상태와 다르다")
    return {"before": before, "fault_application": event, "after_fault": after_fault, "after": after,
            "fault_present_after": expected == 1}


def receipt(faults: Iterable[str], corrections: Iterable[str], reward: Mapping[str, Any],
            optimizer: Mapping[str, Any], manager: Mapping[str, Any]) -> dict[str, Any]:
    original = list(normalize_faults(faults))
    applied = list(normalize_corrections(corrections))
    return {
        "schema": "residual_training_readback_v1",
        "original_faults": original,
        "applied_corrections": applied,
        "remaining_faults": list(remaining_faults(original, applied)),
        "reward": dict(reward),
        "optimizer": dict(optimizer),
        "reward_manager_binding": dict(manager),
    }
