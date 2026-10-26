"""체크포인트 정책 step 뒤 준비 분리 A-A-B-A smoke를 수행한다."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import vision_prepared_sdk_v1 as PREPARED

SCHEMA = "vision_checkpoint_smoke_sdk_v1"
MODE = "prepared_checkpoint_smoke_only"


class CheckpointSmokeSDKFailure(RuntimeError):
    """첫 오류와 봉인 가능한 checkpoint smoke envelope를 함께 보존한다."""

    def __init__(self, report: dict[str, Any], primary: BaseException) -> None:
        super().__init__(
            f"checkpoint smoke failed: {type(primary).__name__}: {primary}"
        )
        self.report = report
        self.envelope = report
        self.primary = primary


def _digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _shape(value: Any) -> list[int]:
    shape = getattr(value, "shape", None)
    if shape is None:
        raise TypeError("정책 tensor shape가 없다")
    result = [int(item) for item in shape]
    if not result or any(item <= 0 for item in result):
        raise ValueError("정책 tensor shape가 잘못됐다")
    return result


def _tensor_summary(value: Any, torch: Any) -> dict[str, Any]:
    detached = value.detach()
    if not bool(torch.isfinite(detached).all().item()):
        raise ValueError("정책 tensor에 비유한 값이 있다")
    cpu = detached.to(device="cpu", dtype=torch.float32).contiguous()
    return {
        "shape": _shape(cpu),
        "dtype": str(cpu.dtype),
        "sha256": hashlib.sha256(cpu.numpy().tobytes(order="C")).hexdigest(),
    }


def _policy_step(
    validated: Mapping[str, Any],
    task: str,
    raw: Any,
    *,
    torch: Any,
    wrapper_type: Callable[..., Any],
    load_cfg: Callable[..., Any],
    runner_type: Callable[..., Any],
) -> Any:
    """checkpoint load/step 실패에서도 생성된 환경을 닫고 첫 오류를 보존한다."""
    env: Any | None = None
    try:
        profile = PREPARED.R.fixed_profile()
        agent_cfg = load_cfg(task, "rsl_rl_cfg_entry_point")
        agent_cfg.device = "cuda:0"
        selected = wrapper_type(raw, clip_actions=agent_cfg.clip_actions)
        env = selected
        runner = runner_type(
            selected, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
        )
        checkpoint = Path(validated["checkpoint_path"])
        runner.load(str(checkpoint), load_optimizer=False)
        policy = runner.get_inference_policy(device=selected.unwrapped.device)
        base = selected.unwrapped
        base.reset(seed=profile["seed"])
        command = base.command_manager.get_command("base_velocity")
        command[0, :3] = command.new_tensor([1.0, 0.0, 0.0])
        observation, _ = selected.get_observations()
        before = int(base.common_step_counter)
        with torch.no_grad():
            action = policy(observation)
        selected.step(action)
        after = int(base.common_step_counter)
        if after != before + 1:
            raise RuntimeError("checkpoint smoke 준비 policy step 수가 1이 아니다")
        receipt = {
            "schema": "vision_checkpoint_policy_probe_v1",
            "checkpoint_sha256": validated["checkpoint_sha256"],
            "command": [1.0, 0.0, 0.0],
            "observation": _tensor_summary(observation, torch),
            "action": _tensor_summary(action, torch),
            "environment_steps": 1,
            "step_counter_before": before,
            "step_counter_after": after,
            "reset_required_before_capture": True,
            "actual_environment_observation_used": True,
            "pure_checkpoint_read_only": False,
        }
        selected._rl_triage_checkpoint_probe = receipt
        return selected
    except BaseException as error:
        try:
            (env if env is not None else raw).close()
        except BaseException as close_error:  # noqa: BLE001 - 첫 오류를 덮지 않는다.
            error.__dict__["checkpoint_probe_cleanup_error"] = {
                "error_type": type(close_error).__name__,
                "error_message": str(close_error),
            }
        raise


def _default_policy_env(
    validated: Mapping[str, Any], make_env: Callable[..., Any]
) -> Callable[..., Any]:
    """실제 env 자료로 정책 forward와 한 step을 수행한 뒤 env를 반환한다."""

    def create(task: str, env_cfg: Any) -> Any:
        import torch  # pyright: ignore[reportMissingImports]
        from isaaclab_rl.rsl_rl import (  # pyright: ignore[reportMissingImports]
            RslRlVecEnvWrapper,
        )
        from isaaclab_tasks.utils import (  # pyright: ignore[reportMissingImports]
            load_cfg_from_registry,
        )
        from rsl_rl.runners import (  # pyright: ignore[reportMissingImports]
            OnPolicyRunner,
        )

        raw = make_env(task, env_cfg)
        return _policy_step(
            validated,
            task,
            raw,
            torch=torch,
            wrapper_type=RslRlVecEnvWrapper,
            load_cfg=load_cfg_from_registry,
            runner_type=OnPolicyRunner,
        )

    return create


def _outer(
    prepared: Mapping[str, Any] | None, probe: Mapping[str, Any] | None
) -> dict[str, Any]:
    approved = (
        prepared is not None
        and prepared.get("approved") is True
        and probe is not None
        and probe.get("environment_steps") == 1
        and probe.get("actual_environment_observation_used") is True
        and probe.get("pure_checkpoint_read_only") is False
    )
    return {
        "schema": SCHEMA,
        "prepared_smoke": deepcopy(dict(prepared)) if prepared is not None else None,
        "policy_probe": deepcopy(dict(probe)) if probe is not None else None,
        "approved": approved,
    }


def run_checkpoint_smoke(
    validated: Mapping[str, Any],
    app: Any,
    *,
    seal_report: Callable[[Mapping[str, Any]], None],
    dependencies: Mapping[str, Any] | None = None,
    policy_env_factory: Callable[
        [Mapping[str, Any], Callable[..., Any]], Callable[..., Any]
    ] = _default_policy_env,
) -> dict[str, Any]:
    """정책 준비 step과 캡처 reset을 분리해 checkpoint별 smoke를 봉인한다."""
    job = validated.get("job")
    if not isinstance(job, Mapping) or job.get("mode") != MODE:
        raise ValueError("checkpoint smoke job mode가 다르다")
    if not callable(seal_report):
        raise TypeError("checkpoint smoke seal_report가 callable이 아니다")
    checkpoint = Path(validated.get("checkpoint_path", ""))
    if checkpoint.is_symlink() or not checkpoint.is_file():
        raise ValueError("checkpoint smoke checkpoint가 regular file이 아니다")

    deps = (
        dict(dependencies)
        if dependencies is not None
        else PREPARED._load_dependencies()
    )
    if set(deps) != PREPARED._DEPENDENCY_KEYS:
        raise ValueError("checkpoint smoke dependency 표면이 다르다")
    original_make_env = deps["make_env"]
    probe_holder: dict[str, Any] = {}
    wrapped_create = policy_env_factory(validated, original_make_env)

    def make_env(*args: Any, **kwargs: Any) -> Any:
        env = wrapped_create(*args, **kwargs)
        probe = getattr(env, "_rl_triage_checkpoint_probe", None)
        if not isinstance(probe, Mapping):
            raise TypeError("checkpoint smoke policy probe 원문이 없다")
        probe_holder.update(dict(probe))
        return env

    deps["make_env"] = make_env
    inner_validated = dict(validated)
    inner_validated["job"] = {**dict(job), "mode": PREPARED.MODE}

    def seal_inner(value: Mapping[str, Any]) -> None:
        seal_report(_outer(value, probe_holder or None))

    try:
        prepared = PREPARED.run_prepared_smoke(
            inner_validated, app, seal_report=seal_inner, dependencies=deps
        )
    except BaseException as error:
        inner = getattr(error, "envelope", getattr(error, "report", None))
        report = _outer(
            inner if isinstance(inner, Mapping) else None, probe_holder or None
        )
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            error.__dict__["checkpoint_smoke_sdk_envelope"] = report
            raise
        primary = getattr(error, "primary", error)
        raise CheckpointSmokeSDKFailure(report, primary) from error
    result = _outer(prepared, probe_holder)
    if result["approved"] is not True:
        raise CheckpointSmokeSDKFailure(
            result, RuntimeError("checkpoint smoke 승인 실패")
        )
    return result
