"""D3 다중 변경: 변경 카탈로그와 주입기.

변경 하나는 Isaac Lab v2.1.1 소스 한 곳의 편집이다. 코드 변경은 함수 객체를 그대로 두고 __code__만 바꾸고,
설정 변경은 같은 값을 hydra override로 넣는다(train.py가 params/env.yaml·agent.yaml에 남긴다). 편집 원문
(edits)은 bench/d3_diff.py가 diff로 렌더링하고, 렌더링한 함수가 실제 교체 본문과 같은지 AST로 대조한다.

bench/train_with_d3_changes.py가 TRIAGE_D3_CHANGES(쉼표로 이은 id, 변경 없음은 NONE)로 arm()한다.
U1 교체 본문은 bench/upstream_faults.py, F1은 bench/hidden_faults.py의 것을 그대로 쓴다(두 파일은 고치지 않는다).
hidden_faults의 F2 본문은 함수 안에서 정의돼 __code__로 옮길 수 없어, 같은 식을 원본 메서드 구조로 다시 썼다
(같은 입력에서 같은 값인지는 tests/test_d3_changes.py가 확인한다).
Isaac Sim 번들 파이썬에서 arm()하고, 카탈로그·부분집합 해석은 CPU 테스트에서도 쓴다.
"""
from __future__ import annotations

from functools import reduce
import importlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hidden_faults as HF  # noqa: E402
import upstream_faults as U  # noqa: E402

ENV_VAR = "TRIAGE_D3_CHANGES"
READBACK_PREFIX = "[d3_changes] "
REWARDS = ("isaaclab.envs.mdp.rewards", "source/isaaclab/isaaclab/envs/mdp/rewards.py")
VELOCITY_REWARDS = ("isaaclab_tasks.manager_based.locomotion.velocity.mdp.rewards",
                    "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/rewards.py")
REWARD_MANAGER = ("isaaclab.managers.reward_manager", "source/isaaclab/isaaclab/managers/reward_manager.py")
ACTUATOR_PD = ("isaaclab.actuators.actuator_pd", "source/isaaclab/isaaclab/actuators/actuator_pd.py")
VELOCITY = "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity"
GO2_ROUGH = f"{VELOCITY}/config/go2/rough_env_cfg.py"
GO2_PPO = f"{VELOCITY}/config/go2/agents/rsl_rl_ppo_cfg.py"
VELOCITY_ENV = f"{VELOCITY}/velocity_env_cfg.py"
UNITREE = "source/isaaclab_assets/isaaclab_assets/robots/unitree.py"


# --- 교체 본문. 대상 모듈의 전역(torch·PrettyTable)을 쓴다. 인자 이름·순서는 원본과 같다 ---
def _flat_orientation_l2(env, asset_cfg=None):
    asset = env.scene[asset_cfg.name]
    gravity_xy = asset.data.projected_gravity_b[:, :2]
    return torch.sum(gravity_xy * gravity_xy, dim=1)  # noqa: F821


def _feet_air_time(env, command_name, sensor_cfg, threshold):
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = torch.clamp(contact_sensor.data.last_air_time[:, sensor_cfg.body_ids], max=5.0)  # noqa: F821
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)  # noqa: F821
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1  # noqa: F821
    return reward


def _reward_table(self):
    msg = f"<RewardManager> contains {len(self._term_names)} active terms.\n"
    table = PrettyTable()  # noqa: F821
    table.title = "Active Reward Terms"
    table.field_names = ["Index", "Name", "Weight"]
    table.align["Name"] = "l"
    table.align["Weight"] = "r"
    for index, (name, term_cfg) in enumerate(zip(self._term_names, self._term_cfgs)):
        table.add_row([index, name, f"{term_cfg.weight:.4g}"])
    msg += table.get_string()
    msg += "\n"
    return msg


def _action_rate_l2(env):
    action_delta = env.action_manager.action - env.action_manager.prev_action
    return torch.sum(action_delta * action_delta, dim=1)  # noqa: F821


def _clip_effort_scaled(self, effort):
    self._joint_vel[:] = torch.clip(self._joint_vel, min=-self._vel_at_effort_lim, max=self._vel_at_effort_lim)  # noqa: F821
    torque_speed_top = self._saturation_effort * (1.0 - 5.0 * self._joint_vel / self.velocity_limit)
    torque_speed_bottom = self._saturation_effort * (-1.0 - 5.0 * self._joint_vel / self.velocity_limit)
    max_effort = torch.clip(torque_speed_top, max=self.effort_limit)  # noqa: F821
    min_effort = torch.clip(torque_speed_bottom, min=-self.effort_limit)  # noqa: F821
    clamped = torch.clip(effort, min=min_effort, max=max_effort)  # noqa: F821
    return clamped


def _reward_target(module_file, name, replacement, edits) -> dict:
    module, file = module_file
    return {"module": module, "file": file, "qualname": name, "replacement": replacement,
            "bound": ("reward_term", name), "edits": edits}


_U1_EDITS = {
    "track_lin_vel_xy_exp": [("- asset.data.root_lin_vel_b[:, :2]", "- torch.zeros_like(asset.data.root_lin_vel_b[:, :2])")],
    "track_ang_vel_z_exp": [("- asset.data.root_ang_vel_b[:, 2]", "- torch.zeros_like(asset.data.root_ang_vel_b[:, 2])")],
    "lin_vel_z_l2": [("torch.square(asset.data.root_lin_vel_b[:, 2])",
                      "torch.square(torch.zeros_like(asset.data.root_lin_vel_b[:, 2]))")],
    "ang_vel_xy_l2": [("torch.square(asset.data.root_ang_vel_b[:, :2])",
                       "torch.square(torch.zeros_like(asset.data.root_ang_vel_b[:, :2]))")],
}

# role: harmless(무해 목록) / reserve(무해 후보가 smoke에서 실패하면 바꿔 넣을 예비) / harmful(해로운 후보).
# config 변경의 overrides는 {hydra 키: (기준값, 새 값)}, edits는 값이 정의된 소스 줄의 편집이다.
# range는 새 값이 Isaac Lab v2.1.1 다른 로봇 설정에 문서화된 범위 안인지와 그 근거다.
CHANGES = {
    "H1": {"role": "harmless", "kind": "code", "category": "reward",
           "definition": "flat_orientation_l2의 제곱을 torch.square 대신 같은 값의 곱으로 계산한다(같은 수식)",
           "targets": [_reward_target(REWARDS, "flat_orientation_l2", _flat_orientation_l2, [(
               "    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)",
               "    gravity_xy = asset.data.projected_gravity_b[:, :2]\n    return torch.sum(gravity_xy * gravity_xy, dim=1)")])],
           "sources": ["의미 보존 리팩터링이라 수치 출처가 없다"]},
    "H2": {"role": "harmless", "kind": "config", "category": "reward",
           "definition": "track_lin_vel_xy_exp 가중치 1.5 → 1.4",
           "overrides": {"env.rewards.track_lin_vel_xy_exp.weight": (1.5, 1.4)},
           "file": GO2_ROUGH, "scope": "UnitreeGo2RoughEnvCfg",
           "edits": [("self.rewards.track_lin_vel_xy_exp.weight = 1.5", "self.rewards.track_lin_vel_xy_exp.weight = 1.4")],
           "range": {"inside": True, "values": {"velocity_env_cfg 기본": 1.0, "go1·go2·a1 rough_env_cfg.py:56": 1.5,
                                                "cassie rough_env_cfg.py:88": 2.0}},
           "sources": [f"{VELOCITY}/velocity_env_cfg.py", f"{VELOCITY}/config/cassie/rough_env_cfg.py:88"]},
    "H3": {"role": "harmless", "kind": "code", "category": "reward",
           "definition": "feet_air_time이 읽는 발 체공 시간을 5.0 s에서 자른다(보행 체공 시간보다 훨씬 긴 경우에만 걸림)",
           "targets": [_reward_target(VELOCITY_REWARDS, "feet_air_time", _feet_air_time, [(
               "last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]",
               "last_air_time = torch.clamp(contact_sensor.data.last_air_time[:, sensor_cfg.body_ids], max=5.0)")])],
           "sources": [f"{VELOCITY}/mdp/rewards.py feet_air_time_positive_biped의 torch.clamp(reward, max=threshold)"
                       "(같은 파일의 clamp 선례). 5.0 s는 설계자 선택"]},
    "H4": {"role": "harmless", "kind": "config", "category": "actuator",
           "definition": "Go2 DC 모터 effort_limit·saturation_effort 23.5 → 23.7",
           "overrides": {"env.scene.robot.actuators.base_legs.effort_limit": (23.5, 23.7),
                         "env.scene.robot.actuators.base_legs.saturation_effort": (23.5, 23.7)},
           "file": UNITREE, "scope": "UNITREE_GO2_CFG",
           "edits": [("effort_limit=23.5,", "effort_limit=23.7,"), ("saturation_effort=23.5,", "saturation_effort=23.7,")],
           "range": {"inside": True, "values": {"go2 unitree.py:170-171": 23.5, "go1 unitree.py:38,40(spec sheet)": 23.7}},
           "sources": [f"{UNITREE}:38-40"]},
    "H5": {"role": "harmless", "kind": "config", "category": "observation",
           "definition": "관절 속도 관측 노이즈 ±1.5 → ±1.2",
           "overrides": {"env.observations.policy.joint_vel.noise.n_min": (-1.5, -1.2),
                         "env.observations.policy.joint_vel.noise.n_max": (1.5, 1.2)},
           "file": VELOCITY_ENV, "scope": "ObservationsCfg.PolicyCfg",
           "edits": [("noise=Unoise(n_min=-1.5, n_max=1.5)", "noise=Unoise(n_min=-1.2, n_max=1.2)")],
           "range": {"inside": True, "values": {"velocity_env_cfg.py:132": 1.5, "spot flat_env_cfg.py:97": 0.5}},
           "sources": [f"{VELOCITY}/config/spot/flat_env_cfg.py:96-98"]},
    "H6": {"role": "harmless", "kind": "code", "category": "logging",
           "definition": "보상 관리자 시작 표의 가중치 열을 유효숫자 4자리 문자열로 출력한다(학습 값과 무관)",
           "targets": [{"module": REWARD_MANAGER[0], "file": REWARD_MANAGER[1], "qualname": "RewardManager.__str__",
                        "replacement": _reward_table, "bound": ("reward_manager", None),
                        "edits": [("table.add_row([index, name, term_cfg.weight])",
                                   'table.add_row([index, name, f"{term_cfg.weight:.4g}"])')]}],
           "sources": ["출력 형식 변경이라 수치 출처가 없다"]},
    "H7": {"role": "harmless", "kind": "config", "category": "optimizer",
           "definition": "PPO entropy_coef 0.01 → 0.008",
           "overrides": {"agent.algorithm.entropy_coef": (0.01, 0.008)},
           "file": GO2_PPO, "scope": "UnitreeGo2RoughPPORunnerCfg",
           "edits": [("entropy_coef=0.01,", "entropy_coef=0.008,")],
           "range": {"inside": True, "values": {"go2·go1·a1·h1 ppo": 0.01, "g1 rsl_rl_ppo_cfg.py:28": 0.008,
                                                "anymal ppo": 0.005, "spot ppo": 0.0025}},
           "sources": [f"{VELOCITY}/config/g1/agents/rsl_rl_ppo_cfg.py:28"]},
    "S1": {"role": "reserve", "kind": "config", "category": "termination",
           "definition": "base_contact 종료 접촉력 임계 1.0 → 1.1",
           "overrides": {"env.terminations.base_contact.params.threshold": (1.0, 1.1)},
           "file": VELOCITY_ENV, "scope": "TerminationsCfg",
           "edits": [('body_names="base"), "threshold": 1.0}', 'body_names="base"), "threshold": 1.1}')],
           "range": {"inside": False, "values": {"velocity_env_cfg.py:273·spot·digit": 1.0}},
           "sources": ["문서화된 값은 모두 1.0이라 1.1은 설계자 선택"]},
    "S2": {"role": "reserve", "kind": "config", "category": "optimizer",
           "definition": "PPO learning_rate 1.0e-3 → 9.0e-4",
           "overrides": {"agent.algorithm.learning_rate": (0.001, 0.0009)},
           "file": GO2_PPO, "scope": "UnitreeGo2RoughPPORunnerCfg",
           "edits": [("learning_rate=1.0e-3,", "learning_rate=9.0e-4,")],
           "range": {"inside": False, "values": {"모든 velocity ppo": 0.001}},
           "sources": ["문서화된 값은 모두 1.0e-3이라 9.0e-4는 설계자 선택"]},
    "S3": {"role": "reserve", "kind": "config", "category": "reward",
           "definition": "dof_acc_l2 가중치 -2.5e-7 → -3.0e-7",
           "overrides": {"env.rewards.dof_acc_l2.weight": (-2.5e-07, -3.0e-07)},
           "file": GO2_ROUGH, "scope": "UnitreeGo2RoughEnvCfg",
           "edits": [("self.rewards.dof_acc_l2.weight = -2.5e-7", "self.rewards.dof_acc_l2.weight = -3.0e-7")],
           "range": {"inside": True, "values": {"g1 flat_env_cfg.py:31": -1.0e-07, "go2 rough_env_cfg.py:58": -2.5e-07,
                                                "cassie rough_env_cfg.py:91(기본 ×1.5)": -3.75e-07}},
           "sources": [f"{VELOCITY}/config/cassie/rough_env_cfg.py:91", f"{VELOCITY}/config/g1/flat_env_cfg.py:31"]},
    "S4": {"role": "reserve", "kind": "code", "category": "reward",
           "definition": "action_rate_l2의 제곱을 torch.square 대신 같은 값의 곱으로 계산한다(같은 수식)",
           "targets": [_reward_target(REWARDS, "action_rate_l2", _action_rate_l2, [(
               "    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)",
               "    action_delta = env.action_manager.action - env.action_manager.prev_action\n"
               "    return torch.sum(action_delta * action_delta, dim=1)")])],
           "sources": ["의미 보존 리팩터링이라 수치 출처가 없다"]},
    "U1": {"role": "harmful", "kind": "code", "category": "reward",
           "definition": U.FAULTS["U1"]["note"],
           "targets": [_reward_target(REWARDS, name, replacement, _U1_EDITS[name])
                       for name, replacement in U.SWAPS["U1"].items()],
           "sources": U.FAULTS["U1"]["sources"]},
    "F1": {"role": "harmful", "kind": "code", "category": "reward",
           "definition": HF.FAULTS["F1"]["note"],
           "targets": [_reward_target(REWARDS, "track_lin_vel_xy_exp", HF._track_lin_vel_world,
                                      [("asset.data.root_lin_vel_b[:, :2]", "asset.data.root_lin_vel_w[:, :2]")])],
           "sources": ["bench/hidden_faults.py F1(설계자 결함)"]},
    "F2": {"role": "harmful", "kind": "code", "category": "actuator",
           "definition": HF.FAULTS["F2"]["note"],
           "targets": [{"module": ACTUATOR_PD[0], "file": ACTUATOR_PD[1], "qualname": "DCMotor._clip_effort",
                        "replacement": _clip_effort_scaled, "bound": ("actuator", "base_legs"),
                        "edits": [("(1.0 - self._joint_vel / self.velocity_limit)",
                                   "(1.0 - 5.0 * self._joint_vel / self.velocity_limit)"),
                                  ("(-1.0 - self._joint_vel / self.velocity_limit)",
                                   "(-1.0 - 5.0 * self._joint_vel / self.velocity_limit)")]}],
           "sources": ["bench/hidden_faults.py F2(설계자 결함)와 같은 식"]},
}
HARMLESS = tuple(cid for cid, c in CHANGES.items() if c["role"] == "harmless")
RESERVE = tuple(cid for cid, c in CHANGES.items() if c["role"] == "reserve")
HARMFUL = tuple(cid for cid, c in CHANGES.items() if c["role"] == "harmful")

# 설계안 초안(2.2절 표)과 달라진 항목. 사전등록 전 확정 목록의 근거로 남긴다.
AMENDMENTS = [
    {"id": "H1", "draft": "track_ang_vel_z_exp의 제곱을 (a-b)*(a-b)로",
     "now": "flat_orientation_l2의 제곱을 곱으로",
     "reason": "U1이 track_ang_vel_z_exp 본문을 통째로 바꿔 같은 함수에 두 변경을 겹칠 수 없다"},
    {"id": "H4", "draft": "velocity_limit 30 → 32",
     "now": "effort_limit·saturation_effort 23.5 → 23.7",
     "reason": "32는 Isaac Lab 로봇 설정에 문서화된 속도 한계(7.5~30) 밖이다. 23.7은 같은 파일 Go1 spec sheet 값이다"},
    {"id": "H6", "draft": "보상 관리자에 추가 스칼라 기록",
     "now": "보상 관리자 시작 표의 가중치 표시 형식",
     "reason": "Episode_Reward/ 아래에 새 스칼라를 넣으면 텔레메트리 특징 코드(src/rl_triage/rules.py:73, "
               "triage_tools.py:113)가 그 항을 보상 항으로 읽는다"},
    {"id": "reserve", "draft": "예비 후보 base_contact 임계·push 간격·학습률",
     "now": "S1 base_contact 임계, S2 학습률, S3 dof_acc_l2 가중치, S4 action_rate_l2 리팩터링",
     "reason": "Go2 rough_env_cfg.py:33이 push_robot을 None으로 두어 push 간격 변경이 학습에 닿지 않는다"},
]


def code_targets(cid: str) -> list[dict]:
    return CHANGES[cid].get("targets", [])


def check_catalog() -> None:
    """무해·예비 변경끼리, 그리고 해로운 후보와 대상(함수·override 키)이 겹치지 않는지 본다.

    해로운 후보끼리는 겹쳐도 된다(U1·F1 모두 track_lin_vel_xy_exp). 한 부분집합에는 해로운 변경이 하나뿐이다.
    """
    owners: dict[tuple, str] = {}
    for cid, change in CHANGES.items():
        keys = [("code", t["file"], t["qualname"]) for t in code_targets(cid)]  # 재노출 별칭이어도 정의 파일로 겹침을 본다
        keys += [("config", key) for key in change.get("overrides", {})]
        for key in keys:
            other = owners.get(key)
            if other is not None and not (change["role"] == "harmful" and CHANGES[other]["role"] == "harmful"):
                raise ValueError(f"{cid}와 {other}가 같은 대상 {key}를 바꾼다")
            owners.setdefault(key, cid)
        if change["kind"] == "code" and (not keys or "overrides" in change):
            raise ValueError(f"{cid}: 코드 변경은 교체 대상만 가진다")
        if change["kind"] == "config" and (keys != [("config", k) for k in change["overrides"]] or not change.get("edits")):
            raise ValueError(f"{cid}: 설정 변경은 override와 소스 편집만 가진다")


check_catalog()


def parse_subset(spec: str) -> tuple[str, ...]:
    """'H1,H3,U1' → 카탈로그 순서의 id 튜플. NONE은 빈 부분집합이다."""
    if not isinstance(spec, str) or not spec.strip():
        raise ValueError(f"{ENV_VAR}가 비어 있다(변경 없음은 NONE)")
    if spec.strip() == "NONE":
        return ()
    ids = [part.strip() for part in spec.split(",")]
    unknown = [cid for cid in ids if cid not in CHANGES]
    if unknown or len(set(ids)) != len(ids):
        raise ValueError(f"모르는 id나 중복이 있다: {spec!r}")
    if sum(CHANGES[cid]["role"] == "harmful" for cid in ids) > 1:
        raise ValueError("해로운 변경은 부분집합에 하나까지다")
    return tuple(cid for cid in CHANGES if cid in ids)


def hydra_overrides(ids) -> list[str]:
    return [f"{key}={new!r}" for cid in ids for key, (_old, new) in CHANGES[cid].get("overrides", {}).items()]


def params_key(override_key: str) -> str:
    """hydra 키 → run_upstream_smoke.load_params의 평면 키(env.yaml:… / agent.yaml:…)."""
    head, _, rest = override_key.partition(".")
    return f"{head}.yaml:{rest}"


def expected_params(ids) -> dict[str, tuple]:
    """부분집합이 기준 실행 params에서 바꾸는 평면 키 → (기준값, 새 값)."""
    return {params_key(key): pair for cid in ids for key, pair in CHANGES[cid].get("overrides", {}).items()}


def _resolve(root, dotted: str):
    return reduce(lambda obj, part: obj[part] if isinstance(obj, dict) else getattr(obj, part), dotted.split("."), root)


def before_env(ids) -> None:
    for cid in ids:
        for target in code_targets(cid):
            U.swap_code(_resolve(importlib.import_module(target["module"]), target["qualname"]), target["replacement"])


def _bound(target: dict, function, env) -> bool:
    kind, name = target["bound"]
    if kind == "reward_term":
        manager = env.unwrapped.reward_manager
        return name in manager.active_terms and manager.get_term_cfg(name).func is function
    owner = env.unwrapped.reward_manager if kind == "reward_manager" else env.unwrapped.scene["robot"].actuators[name]
    return getattr(type(owner), target["qualname"].rsplit(".", 1)[1]) is function


def readback(ids, env) -> dict:
    """환경 생성 뒤 교체 유지·결합과 env 설정 값을 확인한다. agent 설정은 params/agent.yaml 대조로 확인한다."""
    code, config = {}, {}
    for cid in ids:
        rows = {}
        for target in code_targets(cid):
            function = _resolve(importlib.import_module(target["module"]), target["qualname"])
            rows[target["qualname"]] = {"swapped": function.__code__ is target["replacement"].__code__,
                                        "bound": _bound(target, function, env)}
        if rows:
            code[cid] = rows
        values = {key: (_resolve(env.unwrapped.cfg, key[len("env."):]) == new if key.startswith("env.") else "params")
                  for key, (_old, new) in CHANGES[cid].get("overrides", {}).items()}
        if values:
            config[cid] = values
    ok = (all(row["swapped"] and row["bound"] for rows in code.values() for row in rows.values())
          and all(value is True for values in config.values() for value in values.values() if value != "params"))
    return {"changes": list(ids), "code": code, "config": config, "ok": ok}


def arm(spec: str) -> list[str]:
    """gymnasium.make를 감싸 첫 환경 생성 직전에 코드를 바꾸고 직후 확인 줄을 하나 남긴다. hydra override를 돌려준다."""
    try:
        ids = parse_subset(spec)
    except ValueError as error:
        raise SystemExit(f"{ENV_VAR}: {error}") from error
    import gymnasium

    original_make = gymnasium.make
    state = {"armed": False}

    def make(*args, **kwargs):
        if state["armed"]:
            return original_make(*args, **kwargs)
        state["armed"] = True
        before_env(ids)
        env = original_make(*args, **kwargs)
        try:
            line = readback(ids, env)
        except Exception as error:  # 확인 실패는 ok=false로 남기고 학습 쪽 판정에 맡긴다
            # 확인 줄은 공개 결과에 들어가므로 예외 형만 남기고, 문구는 비공개 stderr 로그에 둔다.
            line = {"changes": list(ids), "ok": False, "error": type(error).__name__}
            print(f"[d3_changes_error] {error}", file=sys.stderr, flush=True)
        print(READBACK_PREFIX + json.dumps(line), flush=True)
        return env

    gymnasium.make = make
    return hydra_overrides(ids)
