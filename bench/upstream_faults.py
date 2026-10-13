"""설계자 밖 외부 결함: 다른 프레임워크에서 원인이 확정되고 수정된 결함의 의미를 Isaac Lab v2.1.1 Go2 flat에 옮긴다.

U1 — UniLab 커밋 58ec02e·PR #1254. Motrix·MuJoCo 백엔드에서 root body의 몸체 좌표 선·각속도가 0으로 읽혀,
보상 4항(tracking_lin_vel·tracking_ang_vel·lin_vel_z·ang_vel_xy)이 속도 신호 없이 학습됐다. 여기서는 Isaac Lab
v2.1.1의 같은 4항(isaaclab.envs.mdp.rewards)의 본문을 "몸체 속도 자리에 0"인 본문으로 바꾼다. 관측은 원전에 언급이
없어 바꾸지 않는다. 결함 정의는 원전에서만 정하며 결과를 보고 고치지 않는다(bench/protocols/upstream_u1_smoke_v1.json).

bench/train_with_upstream_fault.py가 TRIAGE_UPSTREAM_FAULT로 arm()한다. Isaac Sim 번들 파이썬에서만 쓰인다.
동결된 bench/hidden_faults.py·external_faults.py와 별개 모듈이다.
"""
from __future__ import annotations

FAULTS = {
    "U1": {
        "category": "reward",
        "note": "보상 4항이 읽는 root 몸체 좌표 선·각속도가 0(관측은 그대로)",
        "sources": ["https://github.com/Motphys/UniLab/commit/58ec02ea8b650e6b90db222188a770932d6b9676",
                    "https://github.com/Motphys/UniLab/pull/1254"],
    },
}
REWARDS_MODULE = "isaaclab.envs.mdp.rewards"


def _lin_vel_z_l2_zero(env, asset_cfg=None):
    # isaaclab.envs.mdp.rewards.lin_vel_z_l2의 __code__로 들어간다. 이름은 그 모듈 전역(torch)에서 찾는다.
    asset = env.scene[asset_cfg.name]
    return torch.square(torch.zeros_like(asset.data.root_lin_vel_b[:, 2]))  # noqa: F821


def _ang_vel_xy_l2_zero(env, asset_cfg=None):
    asset = env.scene[asset_cfg.name]
    return torch.sum(torch.square(torch.zeros_like(asset.data.root_ang_vel_b[:, :2])), dim=1)  # noqa: F821


def _track_lin_vel_xy_exp_zero(env, std, command_name, asset_cfg=None):
    asset = env.scene[asset_cfg.name]
    lin_vel_error = torch.sum(  # noqa: F821
        torch.square(env.command_manager.get_command(command_name)[:, :2]  # noqa: F821
                     - torch.zeros_like(asset.data.root_lin_vel_b[:, :2])),  # noqa: F821
        dim=1,
    )
    return torch.exp(-lin_vel_error / std**2)  # noqa: F821


def _track_ang_vel_z_exp_zero(env, std, command_name, asset_cfg=None):
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(  # noqa: F821
        env.command_manager.get_command(command_name)[:, 2] - torch.zeros_like(asset.data.root_ang_vel_b[:, 2]))  # noqa: F821
    return torch.exp(-ang_vel_error / std**2)  # noqa: F821


# 원본 함수 이름 → 교체 본문. 순서는 프로토콜의 매핑 표와 같다.
SWAPS = {
    "U1": {
        "track_lin_vel_xy_exp": _track_lin_vel_xy_exp_zero,
        "track_ang_vel_z_exp": _track_ang_vel_z_exp_zero,
        "lin_vel_z_l2": _lin_vel_z_l2_zero,
        "ang_vel_xy_l2": _ang_vel_xy_l2_zero,
    },
}


def parameters(code) -> tuple[str, ...]:
    return code.co_varnames[:code.co_argcount + code.co_kwonlyargcount]


def swap_code(target, replacement) -> None:
    """함수 객체는 그대로 두고 본문만 바꾼다. 인자 이름·순서가 같아야 원본의 기본값·키워드 호출이 그대로 맞는다."""
    if target.__code__.co_freevars or replacement.__code__.co_freevars:
        raise RuntimeError("클로저가 있는 함수는 본문을 바꿀 수 없다")
    if parameters(target.__code__) != parameters(replacement.__code__):
        raise RuntimeError(f"{target.__name__}: 인자 {parameters(target.__code__)}와 교체 본문 인자가 다르다")
    target.__code__ = replacement.__code__


def before_env(fault: str) -> list[str]:
    import importlib
    rewards = importlib.import_module(REWARDS_MODULE)
    for name, replacement in SWAPS[fault].items():
        swap_code(getattr(rewards, name), replacement)
    return list(SWAPS[fault])


def readback(fault: str, env) -> dict:
    """환경 생성 뒤 교체가 유지되는지와, 환경의 보상 관리자가 바로 그 함수 객체를 쓰는지 확인한다."""
    import importlib
    rewards = importlib.import_module(REWARDS_MODULE)
    manager = env.unwrapped.reward_manager
    terms = {name: manager.get_term_cfg(name).func for name in manager.active_terms}
    swapped = {name: getattr(rewards, name).__code__ is replacement.__code__ for name, replacement in SWAPS[fault].items()}
    bound = {name: terms.get(name) is getattr(rewards, name) for name in SWAPS[fault]}
    return {"fault": fault, "swapped": swapped, "bound_to_reward_terms": bound,
            "ok": all(swapped.values()) and all(bound.values())}


def arm(fault: str) -> None:
    """gymnasium.make를 감싸 환경 생성 직전에 본문을 바꾸고 직후 확인 줄을 남긴다."""
    if fault not in FAULTS:
        raise SystemExit(f"모르는 TRIAGE_UPSTREAM_FAULT {fault!r}: {sorted(FAULTS)}")
    import json

    import gymnasium

    original_make = gymnasium.make
    state = {"armed": False}

    def make(*args, **kwargs):
        if state["armed"]:
            return original_make(*args, **kwargs)
        state["armed"] = True
        before_env(fault)
        env = original_make(*args, **kwargs)
        print("[upstream_fault] " + json.dumps(readback(fault, env)), flush=True)
        return env

    gymnasium.make = make
