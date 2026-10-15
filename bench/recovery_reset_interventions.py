"""체크포인트를 불러온 직후 적용하는 최소 재설정 개입.

각 개입은 기존 Parameter 객체와 optimizer param_groups를 보존한다. 개입 전후 상태
지문과 대상 Adam 상태 제거를 기록해 부모 실행기가 학습 전 readback을 검증할 수 있게 한다.
"""
from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import train_with_correction as T

INTERVENTIONS = ("optimizer_only", "critic_reset", "actor_last_layer_reset")


def _flat_optimizer_parameters(optimizer: Any) -> list[Any]:
    return [parameter for group in optimizer.param_groups for parameter in group["params"]]


def _optimizer_indices(optimizer: Any, parameters: list[Any]) -> list[int]:
    flat = _flat_optimizer_parameters(optimizer)
    if len({id(parameter) for parameter in flat}) != len(flat):
        raise ValueError("optimizer contains a parameter more than once")
    indices = []
    for parameter in parameters:
        matches = [index for index, candidate in enumerate(flat) if candidate is parameter]
        if len(matches) != 1:
            raise ValueError("reset parameter must appear exactly once in the optimizer")
        indices.append(matches[0])
    if len(set(indices)) != len(indices):
        raise ValueError("reset parameter inventory contains duplicates")
    return sorted(indices)


def _parameter_index_map(optimizer: Any, parameters: list[Any]) -> dict[int, int]:
    flat = _flat_optimizer_parameters(optimizer)
    _optimizer_indices(optimizer, parameters)
    return {id(parameter): next(index for index, candidate in enumerate(flat) if candidate is parameter)
            for parameter in parameters}


def _linear_modules(module: Any, torch_module) -> list[tuple[str, Any]]:
    return [(name, child) for name, child in module.named_modules()
            if isinstance(child, torch_module.nn.Linear)]


def _selection(policy: Any, optimizer: Any, intervention: str, torch_module) -> tuple[list[tuple[str, Any]], list[Any]]:
    if intervention == "optimizer_only":
        return [], _flat_optimizer_parameters(optimizer)
    if intervention == "critic_reset":
        critic = getattr(policy, "critic", None)
        if critic is None:
            raise ValueError("policy has no critic module")
        modules = [(f"critic.{name}".rstrip("."), module)
                   for name, module in _linear_modules(critic, torch_module)]
    elif intervention == "actor_last_layer_reset":
        actor = getattr(policy, "actor", None)
        if actor is None:
            raise ValueError("policy has no actor module")
        linears = _linear_modules(actor, torch_module)
        modules = [] if not linears else [(f"actor.{linears[-1][0]}".rstrip("."), linears[-1][1])]
    else:
        raise ValueError("unregistered recovery reset intervention")
    if not modules:
        raise ValueError("registered intervention found no target Linear module")
    parameters = [parameter for _, module in modules for parameter in module.parameters(recurse=False)]
    if not parameters:
        raise ValueError("registered intervention found no target parameters")
    return modules, parameters


def _rng_devices(parameters: list[Any]) -> list[int]:
    devices = set()
    for parameter in parameters:
        device = parameter.device
        if device.type == "cuda":
            devices.add(device.index if device.index is not None else 0)
    return sorted(devices)


def _state_without(optimizer_state: dict, removed: set[int]) -> dict:
    return {**optimizer_state,
            "state": {index: value for index, value in optimizer_state["state"].items()
                      if index not in removed}}


def apply_intervention(policy: Any, optimizer: Any, intervention: str, reset_seed: int,
                       torch_module) -> dict[str, Any]:
    """등록 개입을 in-place로 적용하고 좁은 상태 변화 readback을 반환한다."""
    if isinstance(reset_seed, bool) or not isinstance(reset_seed, int) or reset_seed < 0:
        raise ValueError("reset_seed must be a non-negative integer")
    modules, parameters = _selection(policy, optimizer, intervention, torch_module)
    parameter_ids = [id(parameter) for parameter in parameters]
    index_by_parameter = _parameter_index_map(optimizer, parameters)
    indices = sorted(index_by_parameter.values())
    policy_before = T.state_digest(policy.state_dict())
    optimizer_dict = optimizer.state_dict()
    optimizer_before = T.state_digest(optimizer_dict)
    group_before = T.state_digest(optimizer_dict["param_groups"])
    moments_before = sorted(index_by_parameter[id(parameter)] for parameter in parameters
                            if parameter in optimizer.state and bool(optimizer.state[parameter]))

    devices = _rng_devices(parameters)
    context = torch_module.random.fork_rng(devices=devices, enabled=True) if modules else nullcontext()
    with context:
        if modules:
            torch_module.manual_seed(reset_seed)
            for _, module in modules:
                module.reset_parameters()
        for parameter in parameters:
            optimizer.state.pop(parameter, None)

    if [id(parameter) for parameter in parameters] != parameter_ids:
        raise RuntimeError("intervention replaced a selected parameter object")
    if _optimizer_indices(optimizer, parameters) != indices:
        raise RuntimeError("intervention changed optimizer parameter identity or order")
    policy_after = T.state_digest(policy.state_dict())
    optimizer_after_dict = optimizer.state_dict()
    optimizer_after = T.state_digest(optimizer_after_dict)
    expected_optimizer = T.state_digest(_state_without(optimizer_dict, set(indices)))
    if T.state_digest(optimizer_after_dict["param_groups"])["sha256"] != group_before["sha256"]:
        raise RuntimeError("intervention changed optimizer param_groups")
    if optimizer_after["sha256"] != expected_optimizer["sha256"]:
        raise RuntimeError("intervention changed optimizer state outside selected moments")
    if any(parameter in optimizer.state and optimizer.state[parameter] for parameter in parameters):
        raise RuntimeError("selected optimizer moments remain after intervention")
    if intervention == "optimizer_only" and policy_after["sha256"] != policy_before["sha256"]:
        raise RuntimeError("optimizer-only intervention changed policy tensors")
    if intervention != "optimizer_only" and policy_after["sha256"] == policy_before["sha256"]:
        raise RuntimeError("network reset did not change policy tensors")

    return {
        "schema": "recovery_reset_intervention_v1",
        "intervention": intervention,
        "reset_seed": reset_seed,
        "target_modules": [name for name, _ in modules],
        "target_parameter_count": len(parameters),
        "optimizer_param_indices": indices,
        "optimizer_moment_indices_present_before": moments_before,
        "optimizer_moments_present_after": False,
        "parameter_identity_preserved": True,
        "policy_state_before": policy_before["sha256"],
        "policy_state_after": policy_after["sha256"],
        "optimizer_state_before": optimizer_before["sha256"],
        "optimizer_state_expected_after": expected_optimizer["sha256"],
        "optimizer_state_after": optimizer_after["sha256"],
        "optimizer_param_groups_before": group_before["sha256"],
        "optimizer_param_groups_after": T.state_digest(optimizer_after_dict["param_groups"])["sha256"],
    }
