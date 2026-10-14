"""두 번째 로봇 task 단계 0 이식 관문: ANYmal-C flat 정상 학습 1회 + 고정 평가 + 동결 probe 6종 기준 측정.

질문은 Go2 flat에서 쓴 동결 고리 부품(학습 진입·고정 평가 계약·동결 probe)을 다른 로봇 task에 수정 없이 옮길 수 있는가다.
관문 기준(걷는다·probe가 재지고 절대 기준 4종이 normal이다·학습 1회 비용)을 실행 전에 고정하고, 하나라도 어긋나면 단계 1로 가지
않는다. Go2 수치와 비교하지 않는다. 진단 성능·일반화가 아니다(bench/private/claude_reward_track_20261013/second_task_scope.md).

  python evals/run_anymal_c_gate.py prepare                                # 프로토콜 작성(커밋 전)
  python evals/run_anymal_c_gate.py verify
  python evals/run_anymal_c_gate.py execute --tag anymal_c_gate_<날짜>      # GPU: 학습·추출·고정 평가·probe·판정
  python evals/run_anymal_c_gate.py execute --tag <새 tag> --retry-of <앞 tag>   # 인프라 원인 invalid 뒤 한 번

학습은 bench/train_with_fault.py(TRIAGE_FAULT=NONE), 고정 평가는 evals/fixed_eval_task.py(동결 fixed_eval.py + 평가 환경의 push·COM
무작위화 끔), probe는 evals/probes_task.py(동결 probes.py의 TASK만 바꿈)다. 소유 Job·watchdog·GPU 확인은 U1 smoke 실행기의 도구를 쓴다.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "bench", ROOT / "evals"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import fixed_eval_task as FE  # noqa: E402
from rl_triage import grounded_recovery as C  # noqa: E402
from rl_triage import probe_loop as L  # noqa: E402
import run_upstream_smoke as U  # noqa: E402

PROTOCOL = ROOT / "bench/protocols/anymal_c_gate_v1.json"
EVAL_PROTOCOL = ROOT / "bench/protocols/fixed_eval_v1_anymal_c.json"
PRIVATE = ROOT / "bench/private/anymal_c_gate"
RESULTS = ROOT / "evals/results"
RUNS_DIR = ROOT / "bench/runs_anymal"
TASK, EXPERIMENT = "Isaac-Velocity-Flat-Anymal-C-v0", "anymal_c_flat"
SEED, NUM_ENVS, MAX_ITERATIONS, ACTION_SCALE = 2043, 4096, 300, 0.5
RUN_NAMES = {1: "anymal_c_gate_none_s2043", 2: "anymal_c_gate_none_s2043_a2"}
FINAL_CHECKPOINT, FIRST_CHECKPOINT = "model_299.pt", "model_0.pt"
TAG_RE = r"anymal_c_gate_[0-9]{8}[a-z0-9_]{0,10}"
COST_MAX_S = 1800.0
ABSOLUTE_PROBES = ("P_noise", "P_reward", "P_physics", "P_episode")
RATIO_PROBES = ("P_value", "P_torque")
PROBE_ORDER = tuple(sorted(L.PROBES))
WATCHDOG_S = {"training": 3600, "telemetry_extract": 300, "fixed_evaluation": 600, "probes": 1200}
ISAACLAB_FILES = {
    "flat_env_cfg": "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/anymal_c/flat_env_cfg.py",
    "rough_env_cfg": "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/anymal_c/rough_env_cfg.py",
    "agent_cfg": "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/anymal_c/agents/rsl_rl_ppo_cfg.py",
    "registration": "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/anymal_c/__init__.py",
    "velocity_env_cfg": "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py",
    "robot_cfg": "source/isaaclab_assets/isaaclab_assets/robots/anymal.py",
    "actuator_net": "source/isaaclab/isaaclab/actuators/actuator_net.py",
    "train": "scripts/reinforcement_learning/rsl_rl/train.py",
}
# 결합 소스에 반드시 있어야 하는 줄. 다르면 이 프로토콜이 가정한 설정이 아니다.
ISAACLAB_ANCHORS = {
    "agent_cfg": ["self.max_iterations = 300", 'self.experiment_name = "anymal_c_flat"'],
    "registration": ['id="Isaac-Velocity-Flat-Anymal-C-v0",'],
    "velocity_env_cfg": ['joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, '
                         "use_default_offset=True)", "self.episode_length_s = 20.0", "self.sim.dt = 0.005",
                         "self.decimation = 4", "push_robot = EventTerm(", "base_com = EventTerm("],
    "robot_cfg": ['actuators={"legs": ANYDRIVE_3_LSTM_ACTUATOR_CFG},'],
    "actuator_net": ["class ActuatorNetLSTM(DCMotor):"],
}
CODE = ("evals/run_anymal_c_gate.py", "evals/fixed_eval_task.py", "evals/probes_task.py", "evals/fixed_eval.py",
        "evals/probes.py", "bench/protocols/fixed_eval_v1_anymal_c.json", "bench/train_with_fault.py",
        "bench/hidden_faults.py", "bench/external_faults.py", "bench/extract_tb.py", "evals/run_upstream_smoke.py",
        "src/rl_triage/recovery.py", "src/rl_triage/grounded_recovery.py", "src/rl_triage/probe_loop.py",
        "bench/record_evidence_media.py", "bench/run_runtime_readback_smoke.py", "bench/runtime_readback_smoke.py")
SUMMARY_SERIES = ("Train/mean_reward", "Train/mean_episode_length", "Metrics/base_velocity/error_vel_xy",
                  "Metrics/base_velocity/error_vel_yaw", "Loss/learning_rate", "Policy/mean_noise_std")
FIXED = {
    "schema": "second_task_gate_protocol_v1",
    "question": "Go2 flat에서 쓴 동결 고리 부품(학습 진입·고정 평가 계약·동결 probe 6종)이 ANYmal-C flat에서 수정 없이 동작하는가: "
                "정상 학습 1회가 고정 평가에서 걷고, probe가 재지며 절대 기준 probe가 정상 정책에서 normal이고, 학습 1회 비용이 계획 안인가",
    "scope": "두 번째 로봇 task 이식 관문, 새 seed 하나의 정상 학습 1회. 진단 성능·일반화·Go2와의 수치 비교가 아니다",
    "stage": {"index": 0, "next": "통과(pass)일 때만 단계 1(정상 seed 2개 추가, 밴드·비 기준 probe·경고 오경보)을 따로 사전등록한다",
              "plan_source": "bench/private/claude_reward_track_20261013/second_task_scope.md 4.1절(사용자 결정: ANYmal-C, 로컬 RTX 3060)"},
    "task": TASK, "experiment": EXPERIMENT, "seed": SEED, "num_envs": NUM_ENVS, "max_iterations": MAX_ITERATIONS,
    "action_scale": ACTION_SCALE, "final_checkpoint": FINAL_CHECKPOINT,
    "training": {"entry": "bench/train_with_fault.py(TRIAGE_FAULT=NONE, hidden_faults가 아무것도 바꾸지 않고 [hidden_fault] 줄만 남김)",
                 "args": "--task Isaac-Velocity-Flat-Anymal-C-v0 --num_envs 4096 --max_iterations 300 --seed 2043 --headless --run_name <이름>",
                 "settings": "task 등록 학습 설정 그대로(push_robot·base_com 무작위화 켬, action scale 0.5)"},
    "fixed_evaluation": {
        "protocol": "bench/protocols/fixed_eval_v1_anymal_c.json(fixed_eval_v1과 같은 격자 26개·eval seed 2026·env 40개/조건·지평 1000·"
                    "step_dt 0.02, task만 다름)",
        "driver": "evals/fixed_eval_task.py: 동결 fixed_eval.py를 그대로 부르고 parse_env_cfg 결과에서 push_robot·base_com만 None. "
                  "[eval_env] 확인 줄 하나",
        "push_decision": "평가 환경에서 push_robot·base_com을 끈다. fixed_eval_v1의 목적은 실행 자신의 학습 설정과 무관한 고정 조건이고, "
                         "이미 add_base_mass(질량 무작위화)를 끈다. push·COM 무작위화도 같은 종류의 학습용 무작위화이고 Go2 task에는 "
                         "없던 항이다. 평가 중 push가 들어가면 추종 오차에 외란 대응이 섞여 seed 간 변동(밴드)이 커진다. 결과를 보기 "
                         "전에 정했고 바꾸지 않는다. 학습 설정은 바꾸지 않는다. base_external_force_torque는 Go2·ANYmal-C 학습 설정 "
                         "모두에 켜져 있고(PLAY 설정만 끈다) 기본 범위가 힘·토크 0이라 외란이 없다. Go2 고정 평가에서도 켜진 채였으므로 "
                         "그대로 둔다",
        "validation": "보고서의 protocol 이름·task가 이 프로토콜과 같은지 따로 대조한 뒤, 나머지(격자·env 수·지평·정지 기준·행별 재계산·"
                      "시각)는 grounded_recovery.validate_fixed_report로 검사한다(그 함수는 이름·task를 fixed_eval_v1 값으로 고정해 두었다)",
    },
    "probes": {
        "driver": "evals/probes_task.py: 동결 probes.py의 TASK만 이 task로 바꾼다. 측정 수식·인자·reset·임계값은 그대로",
        "environment": "task의 등록 학습 설정(push·COM 무작위화 켬). Go2 probe도 Go2의 학습 설정으로 쟀다. probe 환경은 바꾸지 않는다",
        "checkpoint": "model_299.pt(P_value는 model_0.pt와 대조)", "fault": "NONE",
        "classification": "절대 기준 probe(P_noise·P_reward·P_physics·P_episode)는 probe_loop.classify를 기준 없이 그대로 쓴다. 비 기준 "
                          "probe(P_value·P_torque)는 같은 seed 정상 대비라 정상 실행 하나로는 판정할 수 없어 측정값만 기록하고 단계 1(seed 간)에서 본다",
    },
    "gate_criteria": {
        "walks": "고정 평가에서 정지 대비 선속도 오차 비 < 0.5이고 낙상 비율 ≤ 0.05(grounded_recovery.WALK_FRACTION·REF_FALL_MAX, "
                 "P0-C 기준 실행 걷기 조건과 같다)",
        "probes_measured": "probe 6종이 모두 exit 0이고 측정값이 있다",
        "absolute_probes_normal": "절대 기준 probe 4종이 동결 임계값으로 normal이다",
        "cost_within_plan": f"학습 부모 벽시계 ≤ {int(COST_MAX_S)} s. 계획 기준이다: 단계 1·2의 학습 5회를 합쳐 2.5시간 안에 묶기 위한 값이며 "
                            "기술적 한계가 아니다",
    },
    "outcomes": {
        "pass": "유효 조건을 모두 채우고 관문 기준 네 개를 모두 채움 → 단계 1을 사전등록한다",
        "fail": "유효 조건은 채웠지만 관문 기준 하나 이상이 어긋남 → 멈춘다. 단계 1로 가지 않는다",
        "invalid": "유효 조건 중 하나라도 어긋남. 관문을 판정하지 않는다",
    },
    "stop_rule": "fail이면 이 task로 단계 1·2를 하지 않는다. 기준·probe·임계값·평가 환경을 바꾸거나 다른 seed로 다시 시도하지 않는다. "
                 "다른 task(예: A1)로 같은 관문을 할지는 사용자가 정한다",
    "validity_checks": {
        "pipeline_completed": "실행기 단계(학습·추출·평가 전 재확인·고정 평가·probe·사후 확인)가 예외·시간 초과·소스 변경 없이 끝남",
        "training_exit_zero": "학습 프로세스 종료 코드 0",
        "final_checkpoint_present": "실행 폴더에 model_299.pt와 model_0.pt",
        "training_readback": "학습 stdout의 [hidden_fault] 줄 하나가 {\"fault\": \"NONE\"}",
        "params_expected": "params의 seed 2043, num_envs 4096, max_iterations 300, action scale 0.5",
        "eval_env_readback": "고정 평가 stdout의 [eval_env] 줄 하나: task가 같고 push_robot·base_com이 학습 설정에서 켜져 있었고 평가에서 꺼짐",
        "fixed_report_valid": "고정 평가 보고서가 이 평가 프로토콜 이름·task·SHA와 체크포인트 SHA·action scale을 통과",
        "probe_report_bound": "probe 보고서의 이름·task·seed·체크포인트 SHA가 이 실행과 같고 probe 6종이 모두 있다",
    },
    "invalid_causes": {
        "rule": "validity_checks 순서에서 처음 어긋난 검사의 원인 하나를 invalid_cause로 적는다. 학습 뒤 단계에서 멈췄어도 학습 측 "
                "검사가 어긋났으면 그 원인을 앞세운다",
        "by_check": {"pipeline_completed": "infrastructure", "training_exit_zero": "training_process",
                     "final_checkpoint_present": "training_process", "training_readback": "injection",
                     "params_expected": "comparability", "eval_env_readback": "evaluation", "fixed_report_valid": "evaluation",
                     "probe_report_bound": "evaluation"},
        "training_timeout": f"학습이 watchdog {WATCHDOG_S['training']} s를 넘으면 infrastructure로 invalid다. 이때 비용은 관문 기준을 넘었다고 "
                            "따로 기록하지만 판정은 invalid다(멈춘 프로세스와 느린 학습을 가를 수 없다)",
    },
    "execution": {
        "owned_process": "bench/run_runtime_readback_smoke.run_owned(소유 Windows Job·watchdog), U1 smoke 실행기 도구",
        "watchdog_s": WATCHDOG_S, "gpu_admission_profile": U.GPU_PROFILE,
        "seed_unused": "시작 전에 저장소(bench/runs_*·bench/private·evals/results·evals/loops)와 Isaac Lab 로그 전체(모든 experiment 폴더 "
                       "이름·params/agent.yaml seed)에서 seed 2043 흔적을 찾는다. 시도 1은 흔적이 없어야 하고 시도 2는 시도 1 실행 이름만 허용한다",
        "skip_rule": "학습 측 검사(종료 코드·체크포인트·확인 줄·params)가 어긋나면 고정 평가·probe를 하지 않는다",
        "reverify": "고정 평가 직전과 probe 뒤에 프로토콜·결합 코드·Isaac Lab 소스를 다시 확인하고, 끝에 체크포인트 SHA를 다시 계산한다",
        "attempts": {"1": RUN_NAMES[1], "2": RUN_NAMES[2]},
        "retry_rule": "시도 2는 시도 1의 결과가 invalid이고 invalid_cause가 infrastructure일 때 한 번만, 학습부터 새 실행 이름으로 한다",
        "final_rule": "처음 나온 pass 또는 fail이 최종이다",
    },
    "prediction": {
        "outcome": "pass",
        "criteria": {"walks": "걷는다. Isaac Lab이 이 task의 등록 설정으로 4096 env × 300회 학습을 정해 두었다",
                     "probes_measured": "모두 잰다. 액추에이터 그룹이 하나(legs)이고 한계가 같아 P_torque의 그룹 가정을 통과한다",
                     "absolute_probes_normal": "normal. P_reward는 같은 몸체 좌표 함수를 다시 계산한다. P_episode는 probe 환경에 push가 켜져 "
                                               "있어 넘어지면 abnormal이 될 위험이 있다",
                     "cost_within_plan": "예측하지 않는다(이 task의 4096 env 학습 기록이 없다)"},
    },
    "no_tuning_rule": "결과를 본 뒤 관문 기준·평가 환경·probe 환경·임계값·seed를 바꾸지 않는다",
    "wording": {
        "pass": "ANYmal-C flat 정상 학습(새 seed 하나, 4096 env × 300회)이 고정 평가에서 걸었고, 동결 probe 6종이 모두 재졌으며 절대 기준 "
                "4종이 normal이었고, 학습 1회 비용이 계획 안이었다. 단계 1로 간다. 단일 seed 이식 관문이며 진단 성능·일반화가 아니다.",
        "fail": "ANYmal-C flat 이식 관문의 사전등록 기준 일부를 채우지 못해 단계 1로 가지 않는다. 동결 고리 부품을 이 task에 수정 없이 옮기지 "
                "못했다는 기록이며, 기준·probe·임계값을 바꿔 다시 시도하지 않는다.",
        "invalid": "관문 실행이 사전등록한 유효 조건을 채우지 못해 관문을 판정하지 않는다.",
    },
    "used_for": "단계 1 진행 여부와 단계 1·2 비용 추정. Go2 결과와 수치로 비교하지 않는다",
}
DYNAMIC = {"source_files", "isaaclab_sources", "eval_protocol_sha256_lf"}


# --- 입력 결합 ---
def isaaclab_sources(isaaclab: Path) -> dict:
    files = {key: Path(isaaclab) / relative for key, relative in ISAACLAB_FILES.items()}
    for key, anchors in ISAACLAB_ANCHORS.items():
        text = files[key].read_text(encoding="utf-8")
        missing = [anchor for anchor in anchors if anchor not in text]
        if missing:
            raise ValueError(f"Isaac Lab {key}에 등록 줄이 없다: {missing}")
    return {key: {"path": ISAACLAB_FILES[key], "sha256_lf": U.sha(path, True)} for key, path in files.items()}


def dynamic_fields(isaaclab: Path) -> dict:
    return {"source_files": {name: U.sha(ROOT / name, True) for name in CODE},
            "isaaclab_sources": isaaclab_sources(isaaclab), "eval_protocol_sha256_lf": U.sha(EVAL_PROTOCOL, True)}


def check_protocol(protocol: dict, dynamic: dict) -> None:
    if set(protocol) != set(FIXED) | DYNAMIC:
        raise ValueError("프로토콜 필드가 다르다")
    if any(protocol[k] != v for k, v in FIXED.items()):
        raise ValueError("고정 계약이 바뀌었다")
    if any(protocol[k] != v for k, v in dynamic.items()):
        raise ValueError("결합한 코드·Isaac Lab 소스·평가 프로토콜이 바뀌었다")


def prepare(isaaclab: Path) -> dict:
    U.write_new(PROTOCOL, {**deepcopy(FIXED), **dynamic_fields(isaaclab)})
    return {"status": "prepared_commit_before_execute", "path": PROTOCOL.relative_to(ROOT).as_posix()}


def verify(isaaclab: Path, verify_committed=None) -> tuple[dict, str]:
    protocol = U.read(PROTOCOL)
    check_protocol(protocol, dynamic_fields(isaaclab))
    if verify_committed is None:
        import run_runtime_readback_smoke as P
        verify_committed = P.verify_committed
    try:
        commit = verify_committed(ROOT, set(CODE) | {PROTOCOL.relative_to(ROOT).as_posix()})
    except subprocess.CalledProcessError as error:
        raise ValueError("결합 코드나 프로토콜이 커밋되지 않았다") from error
    return protocol, commit


def log_root(isaaclab: Path) -> Path:
    return Path(isaaclab) / "logs/rsl_rl" / EXPERIMENT


def find_run_dir(isaaclab: Path, run_name: str) -> list[Path]:
    root = log_root(isaaclab)
    if not root.is_dir():
        return []
    return sorted(d for d in root.iterdir()
                  if d.is_dir() and re.fullmatch(rf"\d{{4}}-\d\d-\d\d_\d\d-\d\d-\d\d_{re.escape(run_name)}", d.name))


def seed_evidence(isaaclab: Path, allowed: set[str] = frozenset()) -> dict:
    logs = Path(isaaclab) / "logs/rsl_rl"
    roots = {"bench/runs_anymal": RUNS_DIR, "bench/runs_p0c": ROOT / "bench/runs_p0c", "bench/runs_d3": ROOT / "bench/runs_d3",
             "bench/runs_upstream": ROOT / "bench/runs_upstream", "bench/private": ROOT / "bench/private",
             "evals/results": RESULTS, "evals/loops": ROOT / "evals/loops"}
    hits = sorted(p.name for root in roots.values() if root.exists() for p in root.iterdir() if f"_s{SEED}" in p.name)
    if logs.exists():
        for experiment in logs.iterdir():
            if not experiment.is_dir():
                continue
            hits += [f"{experiment.name}/{run.name}" for run in experiment.iterdir() if f"_s{SEED}" in run.name]
            for agent in experiment.glob("*/params/agent.yaml"):
                match = re.search(r"^seed: (\d+)\s*$", agent.read_text(encoding="utf-8", errors="replace"), re.M)
                if match and int(match[1]) == SEED:
                    hits.append(f"{experiment.name}/{agent.parent.parent.name}: seed {SEED}")
    own = sorted({h for h in hits if any(name in h for name in allowed)})
    return {"seed": SEED, "checked_roots": sorted(roots) + ["isaaclab_logs/*"], "hits": sorted(set(hits) - set(own)),
            "allowed_hits": own}


# --- 판정(GPU 없이 시험한다) ---
def readback_ok(lines: list | None) -> bool:
    return isinstance(lines, list) and len(lines) == 1 and lines[0] == {"fault": "NONE"}


def eval_env_ok(lines: list | None) -> bool:
    if not isinstance(lines, list) or len(lines) != 1 or not isinstance(lines[0], dict):
        return False
    line, names = lines[0], U.read(EVAL_PROTOCOL)["env_events_disabled"]
    # 학습 설정에서 켜져 있던 항을 끈 것이어야 한다(처음부터 없던 항이면 이 프로토콜의 가정과 다르다).
    return (line.get("task") == TASK and line.get("disabled") == names
            and line.get("was_enabled") == dict.fromkeys(names, True)
            and line.get("now_enabled") == dict.fromkeys(names, False))


def params_expected(params: dict | None) -> bool:
    if not isinstance(params, dict):
        return False
    expected = {"agent.yaml:seed": SEED, "env.yaml:seed": SEED, "env.yaml:scene.num_envs": NUM_ENVS,
                "agent.yaml:max_iterations": MAX_ITERATIONS, "env.yaml:actions.joint_pos.scale": ACTION_SCALE}
    return all(params.get(key) == value for key, value in expected.items())


def report_valid(report: dict | None, evidence: dict, protocol: dict) -> bool:
    """이름·task는 이 평가 프로토콜 값과 대조하고, 나머지는 공용 검증기로 본다(공용 검증기는 이 두 칸을 Go2 값으로 고정해 두었다)."""
    if not isinstance(report, dict) or report.get("protocol") != U.read(EVAL_PROTOCOL)["name"] or report.get("task") != TASK:
        return False
    try:
        C.validate_fixed_report({**report, "protocol": C.PROTOCOL, "task": C.TASK},
                                {"name": evidence["run_name"], "checkpoint_file": FINAL_CHECKPOINT,
                                 "checkpoint_sha256": evidence.get("checkpoint_sha256"), "action_scale": ACTION_SCALE,
                                 "protocol_sha256_lf": protocol["eval_protocol_sha256_lf"]})
    except (ValueError, KeyError, TypeError):
        return False
    return True


def probe_bound(report: dict | None, evidence: dict) -> bool:
    return (isinstance(report, dict) and report.get("name") == evidence["run_name"] and report.get("task") == TASK
            and report.get("seed") == SEED and (report.get("checkpoint") or {}).get("sha256") == evidence.get("checkpoint_sha256")
            and set(report.get("probes") or {}) == set(L.PROBES))


def training_checks(evidence: dict) -> dict[str, bool]:
    return {"training_exit_zero": evidence.get("train_exit_code") == 0,
            "final_checkpoint_present": bool(evidence.get("checkpoint_sha256")) and evidence.get("first_checkpoint") is True,
            "training_readback": readback_ok(evidence.get("readback")),
            "params_expected": params_expected(evidence.get("params"))}


def gate(report: dict, probe_report: dict, train_wall_s: float | None) -> dict:
    probes = probe_report["probes"]
    measured = {p: probes[p].get("exit") == 0 and probes[p].get("measurement") is not None for p in L.PROBES}
    absolute = {p: L.classify(p, probes[p].get("measurement"), None) if measured[p] else "unknown" for p in ABSOLUTE_PROBES}
    behavior = C._behavior(report)
    criteria = {"walks": U.walks(report), "probes_measured": all(measured.values()),
                "absolute_probes_normal": all(v == "normal" for v in absolute.values()),
                "cost_within_plan": train_wall_s is not None and train_wall_s <= COST_MAX_S}
    return {"criteria": criteria, "absolute_probes": absolute, "probes_measured": measured,
            "behavior": behavior, "standstill_ratio": U.standstill_ratio(report),
            "ratio_probe_measurements": {p: probes[p].get("measurement") for p in RATIO_PROBES}}


def judge(protocol: dict, evidence: dict) -> dict:
    checks = {"pipeline_completed": evidence.get("failure") is None, **training_checks(evidence),
              "eval_env_readback": eval_env_ok(evidence.get("eval_env")),
              "fixed_report_valid": report_valid(evidence.get("report"), evidence, protocol),
              "probe_report_bound": probe_bound(evidence.get("probe_report"), evidence)}
    checks = {name: checks[name] for name in protocol["validity_checks"]}
    failed = [name for name, ok in checks.items() if not ok]
    out = {"checks": checks, "failed_checks": failed, "gate": None, "failed_criteria": [], "invalid_cause": None}
    if failed:
        cause = protocol["invalid_causes"]["by_check"][failed[0]]
        stage = (evidence.get("failure") or {}).get("stage")
        failed_training = [name for name in training_checks(evidence) if name in failed]
        if failed[0] == "pipeline_completed" and stage not in (None, "training") and failed_training:
            cause = protocol["invalid_causes"]["by_check"][failed_training[0]]
        out.update(outcome="invalid", invalid_cause=cause)
    else:
        result = gate(evidence["report"], evidence["probe_report"], evidence.get("train_wall_s"))
        failed_criteria = [name for name, ok in result["criteria"].items() if not ok]
        out.update(gate=result, failed_criteria=failed_criteria, outcome="fail" if failed_criteria else "pass")
    prediction = protocol["prediction"]["outcome"]
    out.update(prediction=prediction, prediction_match=None if out["outcome"] == "invalid" else out["outcome"] == prediction,
               wording=protocol["wording"][out["outcome"]])
    return out


def telemetry_summary(telemetry: dict | None) -> dict | None:
    if telemetry is None:
        return None
    series, out = telemetry.get("series") or {}, {}
    for key in SUMMARY_SERIES:
        values = [v for v in (series.get(key) or []) if isinstance(v, (int, float)) and math.isfinite(v)]
        if values:
            out[key] = {"first": values[0], "last": values[-1], "median": statistics.median(values), "n": len(values)}
    return out


# --- 실행(GPU) ---
def python_bat(isaaclab: Path) -> str:
    return str(Path(isaaclab) / "_isaac_sim" / "python.bat")


def train_command(isaaclab: Path, run_name: str) -> list[str]:
    return [python_bat(isaaclab), str(ROOT / "bench" / "train_with_fault.py"), "--task", TASK, "--num_envs", str(NUM_ENVS),
            "--max_iterations", str(MAX_ITERATIONS), "--seed", str(SEED), "--headless", "--run_name", run_name]


def telemetry_path(run_name: str) -> Path:
    return RUNS_DIR / f"{run_name}.telemetry.json"


def _check_tag(tag) -> None:
    if not isinstance(tag, str) or not re.fullmatch(TAG_RE, tag):
        raise ValueError("tag는 anymal_c_gate_<YYYYMMDD>[접미]")


def _consumed(kind: str) -> Path:
    return PRIVATE / "consumed" / f"{U.sha(PROTOCOL, True)[:16]}_{kind}.json"


def _lines(path: Path, prefix: str) -> list:
    text = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
    out = []
    for line in text.splitlines():
        if line.startswith(prefix):
            try:
                out.append(json.loads(line[len(prefix):]))
            except json.JSONDecodeError:
                out.append(None)
    return out


def bound_sources_same(start: str, now: str) -> bool:
    paths = [*CODE, PROTOCOL.relative_to(ROOT).as_posix()]
    return subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet", start, now, "--", *paths],
                          capture_output=True).returncode == 0


def _reverify(ctx: dict) -> None:
    protocol, commit = verify(ctx["isaaclab"])
    if protocol != ctx["protocol"]:
        raise ValueError("실행 중 프로토콜이나 결합 소스가 바뀌었다")
    if commit != ctx["commit"] and not bound_sources_same(ctx["commit"], commit):
        raise ValueError("시작 커밋과 현재 HEAD 사이에 결합 경로가 바뀌었다")
    if commit not in ctx["commits_seen"]:
        ctx["commits_seen"].append(commit)


def _run(ctx: dict) -> None:
    isaaclab, evidence, private = ctx["isaaclab"], ctx["evidence"], ctx["private"]
    ctx["stage"] = "training"
    os.environ["TRIAGE_FAULT"] = "NONE"
    try:
        execution = U._owned(private / "train", train_command(isaaclab, evidence["run_name"]), isaaclab, WATCHDOG_S["training"])
    finally:
        os.environ.pop("TRIAGE_FAULT", None)
    ctx["executions"]["training"] = execution
    evidence["train_wall_s"] = round(execution["wall_s"], 1)
    evidence["train_exit_code"] = execution["exit_code"]
    evidence["readback"] = _lines(private / "train/stdout.log", "[hidden_fault] ")
    run_dirs = find_run_dir(isaaclab, evidence["run_name"])
    run_dir = run_dirs[0] if len(run_dirs) == 1 else None
    ctx["run_dir"] = run_dir
    checkpoint = run_dir / FINAL_CHECKPOINT if run_dir else None
    evidence["checkpoint_sha256"] = U.sha(checkpoint) if checkpoint and checkpoint.is_file() else None
    evidence["first_checkpoint"] = bool(run_dir and (run_dir / FIRST_CHECKPOINT).is_file())
    if not U.finished(execution):
        raise RuntimeError("학습 프로세스가 시간 안에 스스로 끝나지 않았거나 정리되지 않았다")
    if run_dir is None:
        return
    ctx["stage"] = "telemetry_extract"
    telemetry = telemetry_path(evidence["run_name"])
    if not telemetry.exists():
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        extract = U._owned(private / "extract", [python_bat(isaaclab), str(ROOT / "bench" / "extract_tb.py"), str(run_dir),
                                                 str(telemetry)], isaaclab, WATCHDOG_S["telemetry_extract"])
        ctx["executions"]["telemetry_extract"] = extract
        if not U.finished(extract) or not telemetry.is_file():
            raise RuntimeError("텔레메트리 추출이 끝나지 않았거나 파일이 없다")
    evidence["telemetry"], evidence["telemetry_sha256"] = U.read(telemetry), U.sha(telemetry)
    ctx["stage"] = "params"
    evidence["params"] = U.load_params(run_dir)
    if not all(training_checks(evidence).values()):
        return  # 판정에 쓰지 않을 평가·probe는 하지 않는다
    ctx["stage"] = "pre_evaluation"
    _reverify(ctx)
    U._wait_for_gpu(private)
    ctx["stage"] = "fixed_evaluation"
    report_path = ctx["out"] / "runs" / f"{evidence['run_name']}.json"
    jobs = private / "eval" / "jobs.json"
    U.write_new(jobs, {"action_scale": ACTION_SCALE, "jobs": [{"name": evidence["run_name"],
                                                               "checkpoint": str(run_dir / FINAL_CHECKPOINT),
                                                               "output": str(report_path)}]})
    execution = U._owned(private / "eval", [python_bat(isaaclab), str(ROOT / "evals" / "fixed_eval_task.py"),
                                            "--protocol", str(EVAL_PROTOCOL), "--jobs", str(jobs), "--headless"],
                         isaaclab, WATCHDOG_S["fixed_evaluation"])
    ctx["executions"]["fixed_evaluation"] = execution
    evidence["eval_wall_s"] = round(execution["wall_s"], 1)
    evidence["eval_env"] = FE.parse((private / "eval/stdout.log").read_text(encoding="utf-8", errors="replace")
                                    if (private / "eval/stdout.log").is_file() else "")
    if not U.finished(execution) or execution["exit_code"] != 0 or not report_path.is_file():
        raise RuntimeError("고정 평가가 끝나지 않았거나 보고서가 없다")
    evidence["report"] = U.read(report_path)
    ctx["stage"] = "probes"
    probe_path = ctx["out"] / "probes" / f"{evidence['run_name']}.json"
    probe_jobs = private / "probes" / "jobs.json"
    U.write_new(probe_jobs, {"task": TASK, "fault": "NONE", "seed": SEED, "jobs": [
        {"name": evidence["run_name"], "checkpoint": str(run_dir / FINAL_CHECKPOINT), "probes": list(PROBE_ORDER),
         "output": str(probe_path)}]})
    execution = U._owned(private / "probes", [python_bat(isaaclab), str(ROOT / "evals" / "probes_task.py"),
                                              "--jobs", str(probe_jobs), "--headless"], isaaclab, WATCHDOG_S["probes"])
    ctx["executions"]["probes"] = execution
    evidence["probe_wall_s"] = round(execution["wall_s"], 1)
    if not U.finished(execution) or execution["exit_code"] != 0 or not probe_path.is_file():
        raise RuntimeError("probe 프로세스가 끝나지 않았거나 보고서가 없다")
    evidence["probe_report"] = U.read(probe_path)
    ctx["stage"] = "postflight"
    _reverify(ctx)
    if U.sha(run_dir / FINAL_CHECKPOINT) != evidence["checkpoint_sha256"]:
        raise ValueError("평가·probe 중 체크포인트가 바뀌었다")
    after = U._admission()
    U.write_new(private / "gpu_after.json", after)
    owned = {pid: created for execution in ctx["executions"].values()
             for pid, created in execution["owned_process_identities"].items()}
    if any(str(row["pid"]) in owned and U._process_created(int(row["pid"])) == owned[str(row["pid"])]
           for row in after["processes"]):
        raise ValueError("소유 프로세스가 GPU에 남아 있다")


def _finish(ctx: dict) -> dict:
    evidence = ctx["evidence"]
    ctx["stage"] = "judge"
    try:
        verdict = judge(ctx["protocol"], {**evidence, "failure": ctx["failure"]})
    except Exception as error:
        ctx["failure"] = ctx["failure"] or {"stage": "judge", "error_type": type(error).__name__}
        U.write_new(ctx["private"] / "judge_error.json", {"error_type": type(error).__name__, "detail": str(error)})
        verdict = {"checks": None, "failed_checks": ["pipeline_completed"], "gate": None, "failed_criteria": [],
                   "outcome": "invalid", "invalid_cause": "runner", "prediction": FIXED["prediction"]["outcome"],
                   "prediction_match": None, "wording": FIXED["wording"]["invalid"]}
    U.write_new(ctx["private"] / "execution.json", {"run_dir": str(ctx["run_dir"]) if ctx["run_dir"] else None,
                                                    "attempt": ctx["attempt"], "train_command": train_command(ctx["isaaclab"],
                                                                                                              evidence["run_name"]),
                                                    "failure": ctx["failure"], "retry_of": ctx["retry_of"],
                                                    "commits_seen": ctx["commits_seen"]})
    result = {"schema": "second_task_gate_result_v1", "tag": ctx["tag"], "task": TASK, "seed": SEED, "attempt": ctx["attempt"],
              "run_name": evidence["run_name"], "retry_of": ctx["retry_of"], "source_commit_sha": ctx["commit"],
              "commits_seen": ctx["commits_seen"], "protocol_sha256_lf": U.sha(PROTOCOL, True),
              "seed_evidence": ctx["seed_evidence"], **verdict, "failure": ctx["failure"],
              "train_exit_code": evidence.get("train_exit_code"), "readback": evidence.get("readback"),
              "eval_env": evidence.get("eval_env"), "checkpoint_sha256": evidence.get("checkpoint_sha256"),
              "telemetry_sha256": evidence.get("telemetry_sha256"), "telemetry_summary": telemetry_summary(evidence.get("telemetry")),
              "parent_wall_s": {"train": evidence.get("train_wall_s"), "fixed_eval": evidence.get("eval_wall_s"),
                                "probes": evidence.get("probe_wall_s")},
              "train_wall_per_iteration_s": (round(evidence["train_wall_s"] / MAX_ITERATIONS, 3)
                                             if evidence.get("train_wall_s") is not None else None),
              "parent_wall_scope": "부모 프로세스 벽시계(초, 기동 포함). GPU 커널 시간이나 절감량이 아니다",
              "scope": FIXED["scope"], "model_calls": 0}
    if result["outcome"] != "invalid":
        U.write_new(_consumed("final"), {"tag": ctx["tag"], "outcome": result["outcome"]})
    U.write_new(ctx["out"] / "result.json", result)
    return {"outcome": result["outcome"], "failed_criteria": result["failed_criteria"], "invalid_cause": result["invalid_cause"],
            "failure": result["failure"], "result": f"evals/results/{ctx['tag']}/result.json"}


def execute(tag: str, isaaclab: Path, retry_of: str | None = None) -> dict:
    _check_tag(tag)
    protocol, commit = verify(isaaclab)
    attempt, allowed = 1, set()
    if retry_of is not None:
        _check_tag(retry_of)
        started = _consumed("attempt1")
        last = U.read(RESULTS / retry_of / "result.json") if (RESULTS / retry_of / "result.json").is_file() else None
        if (not started.exists() or U.read(started)["tag"] != retry_of or last is None or last.get("outcome") != "invalid"
                or last.get("invalid_cause") != "infrastructure"):
            raise ValueError("시도 2는 시도 1의 결과가 인프라 원인 invalid일 때 그 tag로만 허용된다")
        attempt, allowed = 2, {RUN_NAMES[1]}
    run_name = RUN_NAMES[attempt]
    consumed = _consumed(f"attempt{attempt}")
    if ((RESULTS / tag).exists() or (PRIVATE / tag).exists() or _consumed("final").exists() or consumed.exists()
            or telemetry_path(run_name).exists() or find_run_dir(isaaclab, run_name)):
        raise FileExistsError("같은 tag·시도·실행 이름의 산출물이 이미 있다. 덮어쓰지 않는다")
    evidence = seed_evidence(isaaclab, allowed)
    if evidence["hits"]:
        raise ValueError(f"seed {SEED}가 이미 쓰였다: {evidence['hits'][:5]}")
    gpu = U._admission()
    if gpu["busy"]:
        return {"status": "gpu_busy_no_launch", "busy_reasons": gpu["busy_reasons"], "training_launched": False}
    ctx = {"tag": tag, "isaaclab": isaaclab, "protocol": protocol, "commit": commit, "attempt": attempt,
           "out": RESULTS / tag, "private": PRIVATE / tag, "run_dir": None, "stage": "training", "failure": None,
           "executions": {}, "evidence": {"run_name": run_name}, "commits_seen": [commit], "retry_of": retry_of,
           "seed_evidence": evidence}
    U.write_new(consumed, {"tag": tag, "attempt": attempt, "run_name": run_name})
    U.write_new(ctx["private"] / "gpu_before_training.json", gpu)
    try:
        _run(ctx)
    except Exception as error:
        ctx["failure"] = {"stage": ctx["stage"], "error_type": type(error).__name__}
        U.write_new(ctx["private"] / "failure.json", {**ctx["failure"], "detail": str(error)})
    return _finish(ctx)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("prepare", "verify", "execute"))
    parser.add_argument("--tag")
    parser.add_argument("--retry-of", dest="retry_of")
    parser.add_argument("--isaaclab", type=Path, default=Path.home() / "IsaacLab")
    args = parser.parse_args(argv)
    if args.command == "execute" and not args.tag:
        parser.error("execute에는 --tag가 필요하다")
    import record_evidence_media as M
    try:
        if args.command == "prepare":
            out = prepare(args.isaaclab)
        elif args.command == "verify":
            out = {"status": "pass", "source_commit_sha": verify(args.isaaclab)[1]}
        else:
            out = execute(args.tag, args.isaaclab, args.retry_of)
    except (ValueError, KeyError, FileExistsError, FileNotFoundError, M.PreflightError,
            subprocess.CalledProcessError) as error:
        print(f"거부: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 3 if out.get("status") == "gpu_busy_no_launch" else 0


if __name__ == "__main__":
    sys.exit(main())
