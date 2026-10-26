"""엄격 A-A-B-A 캡처를 마친 뒤 checkpoint 정책 한 step을 수행한다."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import vision_prepared_sdk_v1 as PREPARED

SCHEMA = "vision_checkpoint_smoke_sdk_v2"
MODE = "prepared_checkpoint_smoke_after_capture_only"


class CheckpointSmokeSDKFailure(RuntimeError):
    """첫 오류와 봉인 가능한 checkpoint smoke envelope를 함께 보존한다."""

    def __init__(self, report: dict[str, Any], primary: BaseException) -> None:
        super().__init__(
            f"checkpoint smoke failed: {type(primary).__name__}: {primary}"
        )
        self.report = report
        self.envelope = report
        self.primary = primary


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
        raise ValueError("정책 tensor에 유한하지 않은 값이 있다")
    cpu = detached.to(device="cpu", dtype=torch.float32).contiguous()
    return {
        "shape": _shape(cpu),
        "dtype": str(cpu.dtype),
        "sha256": hashlib.sha256(cpu.numpy().tobytes(order="C")).hexdigest(),
    }


def _capture_completed(mux: Any) -> bool:
    """동결 mux의 arm 원문을 완화 없이 확인한다."""
    return (
        getattr(mux, "state", None) == "detaching"
        and getattr(mux, "_prepare", None) is not None
        and isinstance(getattr(mux, "_arms", None), list)
        and len(mux._arms) == 4
        and getattr(mux, "_failure", None) is None
    )


class _WriterProxy:
    """기존 Writer 오류를 그대로 전달하며 detach 완료 증거만 기록한다."""

    def __init__(self, writer: Any, mux: Any, state: dict[str, bool]) -> None:
        self._writer = writer
        self._mux = mux
        self._state = state
        self._detach_returned = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._writer, name)

    def detach(self) -> Any:
        result = self._writer.detach()
        self._detach_returned = True
        return result

    def raise_callback_error(self) -> None:
        self._writer.raise_callback_error()
        if self._detach_returned:
            self._state["writer_detached"] = True
            self._state["capture_completed"] = _capture_completed(self._mux)


def _remember_cleanup(error: BaseException, close_error: BaseException) -> None:
    error.__dict__["checkpoint_probe_cleanup_error"] = {
        "error_type": type(close_error).__name__,
        "error_message": str(close_error),
    }


class _PolicyEnvProxy:
    """Prepared SDK close 지점에서 정책 한 step과 실제 env close를 직렬화한다."""

    def __init__(
        self,
        env: Any,
        policy: Callable[[Any], Any],
        torch: Any,
        checkpoint_sha256: str,
        state: dict[str, bool],
        probe_holder: dict[str, Any],
    ) -> None:
        self._env = env
        self._policy = policy
        self._torch = torch
        self._checkpoint_sha256 = checkpoint_sha256
        self._state = state
        self._probe_holder = probe_holder
        self._closed = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    def _step_once(self) -> None:
        if not self._state["capture_completed"]:
            raise RuntimeError("checkpoint smoke capture 완료 전에 policy step을 거부했다")
        if not self._state["writer_detached"]:
            raise RuntimeError("checkpoint smoke Writer detach 전에 policy step을 거부했다")
        base = self._env.unwrapped
        command = base.command_manager.get_command("base_velocity")
        command[0, :3] = command.new_tensor([1.0, 0.0, 0.0])
        observation, _ = self._env.get_observations()
        before = int(base.common_step_counter)
        with self._torch.no_grad():
            action = self._policy(observation)
        self._env.step(action)
        after = int(base.common_step_counter)
        if after != before + 1:
            raise RuntimeError("checkpoint smoke policy step 수가 1이 아니다")
        self._probe_holder.update(
            {
                "schema": "vision_checkpoint_policy_probe_v2",
                "checkpoint_sha256": self._checkpoint_sha256,
                "command": [1.0, 0.0, 0.0],
                "observation": _tensor_summary(observation, self._torch),
                "action": _tensor_summary(action, self._torch),
                "environment_steps": 1,
                "step_counter_before": before,
                "step_counter_after": after,
                "capture_completed_before_policy_step": True,
                "writer_detached_before_policy_step": True,
                "actual_environment_observation_used": True,
                "pure_checkpoint_read_only": False,
            }
        )

    def close(self) -> None:
        if self._closed:
            raise RuntimeError("checkpoint smoke 환경 close가 두 번 호출됐다")
        self._closed = True
        primary: BaseException | None = None
        try:
            self._step_once()
        except BaseException as error:  # noqa: BLE001 - close 뒤 원형으로 재발생시킨다.
            primary = error
        try:
            self._env.close()
        except BaseException as close_error:
            if primary is not None:
                _remember_cleanup(primary, close_error)
            else:
                raise
        if primary is not None:
            raise primary


def _policy_environment(
    validated: Mapping[str, Any],
    task: str,
    raw: Any,
    state: dict[str, bool],
    probe_holder: dict[str, Any],
    *,
    torch: Any,
    wrapper_type: Callable[..., Any],
    load_cfg: Callable[..., Any],
    runner_type: Callable[..., Any],
) -> _PolicyEnvProxy:
    """checkpoint는 환경 생성 때 읽고, 정책 step은 proxy close까지 미룬다."""
    selected: Any | None = None
    try:
        agent_cfg = load_cfg(task, "rsl_rl_cfg_entry_point")
        agent_cfg.device = "cuda:0"
        selected = wrapper_type(raw, clip_actions=agent_cfg.clip_actions)
        if selected is None:
            raise RuntimeError("checkpoint smoke wrapper 환경이 없다")
        runner = runner_type(
            selected, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
        )
        runner.load(str(Path(validated["checkpoint_path"])), load_optimizer=False)
        policy = runner.get_inference_policy(device=selected.unwrapped.device)
        return _PolicyEnvProxy(
            selected,
            policy,
            torch,
            str(validated["checkpoint_sha256"]),
            state,
            probe_holder,
        )
    except BaseException as error:
        try:
            (selected if selected is not None else raw).close()
        except BaseException as close_error:  # noqa: BLE001 - 첫 오류를 덮지 않는다.
            _remember_cleanup(error, close_error)
        raise


def _default_policy_env(
    validated: Mapping[str, Any],
    make_env: Callable[..., Any],
    state: dict[str, bool],
    probe_holder: dict[str, Any],
) -> Callable[..., Any]:
    """환경 생성과 checkpoint load만 수행하는 실제 dependency를 만든다."""

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
        return _policy_environment(
            validated,
            task,
            raw,
            state,
            probe_holder,
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
        and probe.get("capture_completed_before_policy_step") is True
        and probe.get("writer_detached_before_policy_step") is True
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
        [
            Mapping[str, Any],
            Callable[..., Any],
            dict[str, bool],
            dict[str, Any],
        ],
        Callable[..., Any],
    ] = _default_policy_env,
) -> dict[str, Any]:
    """캡처→detach→정책 step→환경 close 순서를 기존 Prepared SDK에 합성한다."""
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
    original_writer = deps["writer"]
    state = {"capture_completed": False, "writer_detached": False}
    probe_holder: dict[str, Any] = {}
    deps["make_env"] = policy_env_factory(
        validated, original_make_env, state, probe_holder
    )

    def writer_factory(rep: Any, mux: Any) -> _WriterProxy:
        return _WriterProxy(original_writer(rep, mux), mux, state)

    deps["writer"] = writer_factory
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
