"""P0-C 숨은 결함: 실행 설정(params/env.yaml·agent.yaml)에 드러나지 않는 코드·런타임 결함을 학습 프로세스에 넣는다.

v1 결함은 모두 hydra override라 설정 diff 한 번에 드러났다(docs/IMPLEMENTATION-ORDER.md 2절 B, P0-A2 B0 10/10).
여기의 결함은 함수 구현·클래스 메서드·생성 뒤 런타임 값을 바꾸므로 dump된 설정은 기준 실행과 같다.

bench/train_with_fault.py가 TRIAGE_FAULT 환경변수로 하나를 골라 arm()한다. 결함 정의는 공개 코드지만
어느 케이스에 어느 결함이 들어갔는지는 bench/private에만 두고 평가가 끝난 뒤 공개한다.
Isaac Sim 번들 파이썬에서만 쓰인다(isaaclab·rsl_rl은 gym.make 시점에 import한다).
"""
from __future__ import annotations

FAULTS = {
    "NONE": {"category": None, "note": "결함 없음(기준 실행도 같은 부트스트랩으로 학습)"},
    "F1": {"category": "reward", "note": "속도 추종 보상이 몸체 좌표 대신 월드 좌표 선속도를 씀"},
    "F2": {"category": "actuator", "note": "DC 모터 토크-속도 곡선의 속도 항 5배(낮은 관절 속도에서 토크 포화)"},
    "F3": {"category": "exploration", "note": "학습 중 행동 샘플링 노이즈가 기록되는 std와 무관하게 0.05"},
    "F4": {"category": "optimizer", "note": "critic 기울기를 0으로 만들어 가치 함수가 학습되지 않음(value_loss_coef는 그대로)"},
    "F5": {"category": "physics", "note": "환경 생성 뒤 로봇 접촉 재질의 정·동마찰 0.25배"},
    "F6": {"category": "termination", "note": "time_out이 최대 에피소드 길이의 5%에서 발생(episode_length_s는 그대로)"},
}


def _track_lin_vel_world(env, std, command_name, asset_cfg=None):
    # isaaclab.envs.mdp.rewards.track_lin_vel_xy_exp의 __code__로 들어간다. 이름은 그 모듈 전역(torch)에서 찾는다.
    asset = env.scene[asset_cfg.name]
    lin_vel_error = torch.sum(  # noqa: F821
        torch.square(env.command_manager.get_command(command_name)[:, :2] - asset.data.root_lin_vel_w[:, :2]),  # noqa: F821
        dim=1,
    )
    return torch.exp(-lin_vel_error / std**2)  # noqa: F821


def _time_out_early(env):
    # isaaclab.envs.mdp.terminations.time_out의 __code__로 들어간다.
    # 보정 2차(seed 7): 10%에서는 고정 평가가 깨지지 않아 docs/P0-C-HOLDOUT.md 규칙대로 5%로 키웠다.
    return env.episode_length_buf >= int(env.max_episode_length * 0.05)


def _swap_code(target, replacement) -> None:
    """함수 객체는 그대로 두고 본문만 바꾼다. 설정에 담긴 함수 참조와 dump되는 모듈 경로·이름이 바뀌지 않는다."""
    if target.__code__.co_freevars or replacement.__code__.co_freevars:
        raise RuntimeError("클로저가 있는 함수는 본문을 바꿀 수 없다")
    target.__code__ = replacement.__code__


def _before_env(fault: str) -> None:
    import torch

    if fault == "F1":
        from isaaclab.envs.mdp import rewards
        _swap_code(rewards.track_lin_vel_xy_exp, _track_lin_vel_world)
    elif fault == "F2":
        from isaaclab.actuators.actuator_pd import DCMotor

        def _clip_effort(self, effort):
            self._joint_vel[:] = torch.clip(self._joint_vel, min=-self._vel_at_effort_lim, max=self._vel_at_effort_lim)
            top = self._saturation_effort * (1.0 - 5.0 * self._joint_vel / self.velocity_limit)
            bottom = self._saturation_effort * (-1.0 - 5.0 * self._joint_vel / self.velocity_limit)
            return torch.clip(effort, min=torch.clip(bottom, min=-self.effort_limit), max=torch.clip(top, max=self.effort_limit))

        DCMotor._clip_effort = _clip_effort
    elif fault == "F3":
        from rsl_rl.modules.actor_critic import ActorCritic

        def act(self, observations, **kwargs):
            self.update_distribution(observations)
            mean = self.distribution.mean
            return mean + 0.05 * torch.randn_like(mean)

        ActorCritic.act = act
    elif fault == "F4":
        from rsl_rl.runners import OnPolicyRunner
        original_init = OnPolicyRunner.__init__

        def __init__(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            for p in self.alg.policy.critic.parameters():
                p.register_hook(lambda grad: torch.zeros_like(grad))

        OnPolicyRunner.__init__ = __init__
    elif fault == "F6":
        from isaaclab.envs.mdp import terminations
        _swap_code(terminations.time_out, _time_out_early)


def _after_env(fault: str, env) -> dict:
    if fault != "F5":
        return {}
    import torch
    robot = env.unwrapped.scene["robot"]
    mats = robot.root_physx_view.get_material_properties()
    before = mats[..., :2].mean(dim=(0, 1)).tolist()
    mats[..., :2] *= 0.25
    robot.root_physx_view.set_material_properties(mats, torch.arange(env.unwrapped.num_envs, device="cpu"))
    after = robot.root_physx_view.get_material_properties()[..., :2].mean(dim=(0, 1)).tolist()
    return {"friction_static_dynamic_before": before, "after": after}


def arm(fault: str) -> None:
    """gymnasium.make를 감싸 환경 생성 직전·직후에 결함을 넣는다. train.py는 이 시점에 이미 Isaac Sim을 띄웠다."""
    if fault not in FAULTS:
        raise SystemExit(f"모르는 TRIAGE_FAULT {fault!r}: {sorted(FAULTS)}")
    import json

    import gymnasium

    original_make = gymnasium.make
    state = {"armed": False}

    def make(*args, **kwargs):
        if state["armed"]:
            return original_make(*args, **kwargs)
        state["armed"] = True
        _before_env(fault)
        env = original_make(*args, **kwargs)
        readback = _after_env(fault, env)
        # 비공개 로그(bench/private/logs)에만 남는다. 텔레메트리·params에는 들어가지 않는다.
        print("[hidden_fault] " + json.dumps({"fault": fault, **readback}), flush=True)
        return env

    gymnasium.make = make
