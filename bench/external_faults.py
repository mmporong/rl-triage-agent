"""T1 외부 원인 결함: 공개 이슈·PR로 원인이 확정된 Isaac Lab 결함을 v2.1.1 학습에 넣는다(docs/T1-EXTERNAL-FAULTS.md).

P0-C 결함(bench/hidden_faults.py)은 probe와 같은 사람이 만들었다. 여기의 결함은 상류에서 출하됐거나 보고된 코드를
그대로 옮기고, 크기는 Isaac Lab 기본값을 쓴다. 동결된 probe가 겨누지 않는 결함에서 틀린 확정 없이 멈추는지를 잰다.

- X1·X3는 짝 설계다. 기준(X1_REF·X3_REF)과 결함은 같은 설정(순정 Go2 flat에 이벤트 항 하나 추가)을 쓰고 함수 본문만
  다르다. params diff는 짝 사이에 없고, 순정 Go2 flat 대비로는 추가한 이벤트 항이 보인다.
- X2는 센서 설정 한 줄이라 순정 대비 params diff에 보인다. 기준은 순정 Go2 flat(TRIAGE_FAULT=NONE)이다.

bench/hidden_faults.py가 X로 시작하는 TRIAGE_FAULT를 이 모듈로 넘긴다. 학습(bench/train_with_fault.py)과
probe(evals/probes.py)가 같은 경로로 주입한다. Isaac Sim 번들 파이썬에서만 실행된다.
"""
from __future__ import annotations

FAULTS = {
    "X1_REF": {"category": None, "pair": "X1",
               "note": "base_com을 reset 모드로 켜고(상류 기본 범위) 매번 기본 COM에서 더한다(develop의 #7311 수정 의미)"},
    "X1": {"category": "physics", "pair": "X1",
           "note": "base_com reset 모드 + v2.2.1~v2.3.x 출하 함수(현재 COM에 더해 reset마다 누적)",
           "source": "https://github.com/isaac-sim/IsaacLab/issues/7311"},
    "X2": {"category": "termination", "pair": None,
           "note": "접촉 센서 history_length=0(ContactSensorCfg 기본값). GPU에서 지연 읽기면 끊긴 접촉의 힘이 남는다",
           "source": "https://github.com/isaac-sim/IsaacLab/issues/7613"},
    "X3_REF": {"category": None, "pair": "X3",
               "note": "push_robot을 상류 기본값(10~15초, x·y ±0.5 m/s)으로 켠다. v2.1.1 함수(속도에 더함) 그대로"},
    "X3": {"category": "physics", "pair": "X3",
           "note": "push_robot 상류 기본값 + PR #1584 이전 본문(root 속도를 표본값으로 덮어씀)",
           "source": "https://github.com/isaac-sim/IsaacLab/pull/1584"},
}

_STATE: dict = {}


def _com_from_default(env, env_ids, com_range, asset_cfg):
    # X1_REF: isaaclab.envs.mdp.events.randomize_rigid_body_com의 __code__로 들어간다(이름은 그 모듈 전역에서 찾는다).
    # develop의 #7311 수정(첫 호출의 COM을 기본값으로 두고 매번 거기에 더함)을 v2.1.1 API로 옮겼다.
    asset = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")  # noqa: F821
    else:
        env_ids = env_ids.cpu()
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")  # noqa: F821
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")  # noqa: F821
    range_list = [com_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device="cpu")  # noqa: F821
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device="cpu").unsqueeze(1)  # noqa: F821
    if getattr(asset, "_triage_default_com", None) is None:
        asset._triage_default_com = asset.root_physx_view.get_coms().clone()
    coms = asset._triage_default_com.clone()
    coms[env_ids[:, None], body_ids, :3] += rand_samples
    asset.root_physx_view.set_coms(coms, env_ids)


def _com_accumulate_v221(env, env_ids, com_range, asset_cfg):
    # X1: v2.2.1 events.py randomize_rigid_body_com 본문 그대로(현재 COM을 읽어 더한다).
    asset = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")  # noqa: F821
    else:
        env_ids = env_ids.cpu()
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")  # noqa: F821
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")  # noqa: F821
    range_list = [com_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device="cpu")  # noqa: F821
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device="cpu").unsqueeze(1)  # noqa: F821
    coms = asset.root_physx_view.get_coms().clone()
    coms[env_ids[:, None], body_ids, :3] += rand_samples
    asset.root_physx_view.set_coms(coms, env_ids)


def _push_overwrite_pre1584(env, env_ids, velocity_range, asset_cfg=None):
    # X3: v2.1.1 push_by_setting_velocity 본문에서 PR #1584가 바꾼 한 줄만 되돌렸다(+= → [:] =).
    # 기본 인자는 교체 대상 함수의 것(SceneEntityCfg("robot"))이 그대로 쓰인다.
    asset = env.scene[asset_cfg.name]
    vel_w = asset.data.root_vel_w[env_ids]
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)  # noqa: F821
    vel_w[:] = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], vel_w.shape, device=asset.device)  # noqa: F821
    asset.write_root_velocity_to_sim(vel_w, env_ids=env_ids)


# 교체 대상(모듈 경로, 함수 이름)과 v2.1.1 서명. 본문 교체는 인자 이름·개수가 같아야 기본 인자가 맞는다.
SWAPS = {
    "X1_REF": ("isaaclab.envs.mdp.events", "randomize_rigid_body_com", _com_from_default),
    "X1": ("isaaclab.envs.mdp.events", "randomize_rigid_body_com", _com_accumulate_v221),
    "X3": ("isaaclab.envs.mdp.events", "push_by_setting_velocity", _push_overwrite_pre1584),
}
UPSTREAM_ARGS = {
    "randomize_rigid_body_com": ("env", "env_ids", "com_range", "asset_cfg"),
    "push_by_setting_velocity": ("env", "env_ids", "velocity_range", "asset_cfg"),
}


def apply_cfg(fault: str, cfg, upstream=None) -> dict:
    """환경 설정 객체를 바꾸고 바꾼 항목을 돌려준다. upstream은 velocity 기본 EventCfg(크기 출처)다."""
    if upstream is None:
        from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
            EventCfg,
        )
        upstream = EventCfg()
    if fault in ("X1_REF", "X1"):
        if cfg.events.base_com is not None:
            raise RuntimeError("순정 Go2 flat은 base_com이 None이다. 다른 태스크·설정에는 넣지 않는다")
        term = upstream.base_com
        term.mode = "reset"
        cfg.events.base_com = term
        return {"events.base_com": {"mode": term.mode, "com_range": dict(term.params["com_range"])}}
    if fault in ("X3_REF", "X3"):
        if cfg.events.push_robot is not None:
            raise RuntimeError("순정 Go2 flat은 push_robot이 None이다. 다른 태스크·설정에는 넣지 않는다")
        term = upstream.push_robot
        cfg.events.push_robot = term
        return {"events.push_robot": {"mode": term.mode, "interval_range_s": list(term.interval_range_s),
                                      "velocity_range": dict(term.params["velocity_range"])}}
    if fault == "X2":
        if cfg.scene.contact_forces.history_length == 0:
            raise RuntimeError("이미 history_length=0이다. 순정 Go2 flat(3)에만 넣는다")
        cfg.scene.contact_forces.history_length = 0
        return {"scene.contact_forces.history_length": 0}
    raise SystemExit(f"모르는 외부 결함 {fault!r}: {sorted(FAULTS)}")


def before_env(fault: str, swap_code) -> None:
    """함수 본문을 바꾸고, 환경 생성 때 설정을 바꾸도록 ManagerBasedRLEnv.__init__을 감싼다."""
    if fault not in FAULTS:
        raise SystemExit(f"모르는 외부 결함 {fault!r}: {sorted(FAULTS)}")
    import importlib

    from isaaclab.envs import ManagerBasedRLEnv

    if fault in SWAPS:
        module, name, replacement = SWAPS[fault]
        swap_code(getattr(importlib.import_module(module), name), replacement)
    original = ManagerBasedRLEnv.__init__

    def __init__(self, cfg, *args, **kwargs):
        # train.py는 이 설정 객체를 gym.make 뒤에 params/env.yaml로 dump한다. 바꾼 항목이 params에 남는다.
        _STATE["changed"] = apply_cfg(fault, cfg)
        ManagerBasedRLEnv.__init__ = original
        original(self, cfg, *args, **kwargs)

    ManagerBasedRLEnv.__init__ = __init__


def after_env(fault: str, env) -> dict:
    """설정이 실제 관리자에 반영됐는지 읽어 확인한다. 반영되지 않았으면 학습 전에 멈춘다."""
    base = env.unwrapped
    checks = {}
    if fault in ("X1_REF", "X1"):
        checks["base_com_in_reset"] = "base_com" in base.event_manager.active_terms.get("reset", [])
    elif fault in ("X3_REF", "X3"):
        checks["push_robot_in_interval"] = "push_robot" in base.event_manager.active_terms.get("interval", [])
    elif fault == "X2":
        checks["history_length_0"] = base.scene["contact_forces"].cfg.history_length == 0
    if not all(checks.values()):
        raise RuntimeError(f"외부 결함 {fault} 설정이 환경에 반영되지 않았다: {checks}")
    return {"changed": _STATE.get("changed"), "checks": checks}
