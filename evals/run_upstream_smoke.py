"""설계자 밖 외부 결함 U1 smoke: 학습 1회 + 고정 평가로 실패 유무만 거른다.

U1은 UniLab 58ec02e·PR #1254가 고친 결함(보상 4항이 root 몸체 속도 0을 읽음)의 의미를 Isaac Lab v2.1.1 Go2 flat에
옮긴 것이다(bench/upstream_faults.py). 기준은 P0-C의 같은 seed 정상 실행(baseline_p0c_s2026)과 그 고정 평가이며,
판정은 P0-C 밴드로 failure_present / failure_absent / invalid 셋 중 하나다. 진단 성능이나 일반화를 재지 않는다.

  python evals/run_upstream_smoke.py prepare           # 프로토콜 작성(커밋 전)
  python evals/run_upstream_smoke.py verify            # 계약·코드·기준 입력·커밋 확인
  python evals/run_upstream_smoke.py execute --tag upstream_u1_smoke_<날짜>   # GPU: 학습·추출·고정 평가·판정
  python evals/run_upstream_smoke.py execute --tag <새 tag> --retry-of <앞 tag>   # 시도 2(인프라 원인 invalid 뒤 한 번)
  python evals/run_upstream_smoke.py resume --tag <새 tag> --from <앞 tag>        # 학습은 두고 추출·평가·판정만

학습(bench/train_with_upstream_fault.py)·텔레메트리 추출(bench/extract_tb.py)·고정 평가(evals/fixed_eval.py)는
v4·W1처럼 소유 Windows Job 안에서 watchdog과 함께 돌린다. 학습을 시작한 뒤 어느 단계에서 멈춰도 비공개 failure.json과
invalid result.json을 한 번 남긴다. 모델·네트워크는 쓰지 않는다.
"""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "bench", ROOT / "evals"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import yaml  # noqa: E402

import upstream_faults as U  # noqa: E402
from rl_triage import grounded_recovery as C  # noqa: E402
from rl_triage import recovery as RC  # noqa: E402
from rl_triage.triage_tools import _TolerantLoader  # noqa: E402

PROTOCOL = ROOT / "bench/protocols/upstream_u1_smoke_v1.json"
PRIVATE = ROOT / "bench/private/upstream_smoke"
RESULTS = ROOT / "evals/results"
RUNS_DIR = ROOT / "bench/runs_upstream"
P0C = ROOT / "evals/results/p0c_fixed_eval_20261009"
FIXED_EVAL_PROTOCOL = ROOT / "bench/protocols/fixed_eval_v1.json"
REFERENCE_TELEMETRY = ROOT / "bench/runs_p0c/baseline_p0c_s2026.telemetry.json"
TASK = "Isaac-Velocity-Flat-Unitree-Go2-v0"
FAULT, SEED, NUM_ENVS, MAX_ITERATIONS, ACTION_SCALE = "U1", 2026, 4096, 300, 0.25
RUN_NAMES = {1: "upstream_u1_s2026", 2: "upstream_u1_s2026_a2"}
REFERENCE_RUN = "baseline_p0c_s2026"
FINAL_CHECKPOINT = "model_299.pt"
TAG_RE = r"upstream_u1_smoke_[0-9]{8}[a-z0-9_]{0,10}"
ISAACLAB_FILES = {"rewards": "source/isaaclab/isaaclab/envs/mdp/rewards.py",
                  "velocity_env_cfg": "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py",
                  "go2_rough_env_cfg": "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/rough_env_cfg.py"}
FUNCTION_LINES = {"lin_vel_z_l2": [76, 80], "ang_vel_xy_l2": [83, 87],
                  "track_lin_vel_xy_exp": [297, 308], "track_ang_vel_z_exp": [311, 319]}
PENALTY_SERIES = ("Episode_Reward/lin_vel_z_l2", "Episode_Reward/ang_vel_xy_l2")
TRACKING_SERIES = ("Episode_Reward/track_lin_vel_xy_exp", "Episode_Reward/track_ang_vel_z_exp")
ALLOWED_PARAM_DIFFS = {"agent.yaml:run_name"}
GPU_PROFILE = "base_com_reset_readback_v2"
# 학습 2700 s: 경합 기록 1871.3 s(e01_s2028)를 넘기되 멈춘 프로세스는 끊는다. 평가 360 s는 v4와 같다.
WATCHDOG_S = {"training": 2700, "telemetry_extract": 300, "fixed_evaluation": 360}
EVAL_GPU_WAIT_S, EVAL_GPU_POLL_S = 1800, 60
RESUMABLE_STAGES = ("telemetry_extract", "pre_evaluation", "fixed_evaluation", "postflight")
TRAINING_CHECKS = ("training_exit_zero", "final_checkpoint_present", "injection_readback")
CODE = ("bench/upstream_faults.py", "bench/train_with_upstream_fault.py", "evals/run_upstream_smoke.py",
        "bench/extract_tb.py", "evals/fixed_eval.py", "bench/protocols/fixed_eval_v1.json",
        "src/rl_triage/recovery.py", "src/rl_triage/grounded_recovery.py", "bench/record_evidence_media.py",
        "bench/run_runtime_readback_smoke.py", "bench/runtime_readback_smoke.py")
OUTCOMES = ("failure_present", "failure_absent", "invalid")
FIXED = {
    "schema": "upstream_fault_smoke_protocol_v1",
    "question": "다른 프레임워크에서 원인이 확정·수정된 결함 U1의 의미를 옮기면 Go2 flat 4096 env × 300회 학습이 실제로 실패하는가",
    "scope": "설계자 밖 외부 결함 한 종류·seed 하나의 학습 실패 유무 거르기. 진단 성능·일반화·미관측 seed 평가가 아니다",
    "fault": {
        "id": FAULT, "category": "reward",
        "definition": "Isaac Lab v2.1.1 isaaclab.envs.mdp.rewards의 4개 보상 함수가 읽는 root 몸체 좌표 선·각속도 자리를 0으로 둔다. "
                      "함수 객체는 그대로 두고 __code__만 바꾼다(bench/upstream_faults.py). 관측 항은 바꾸지 않는다",
        "observation_unchanged_reason": "원전은 보상 항이 0을 읽었다고만 적고 관측은 언급하지 않는다. 수정 전 upstream Go2 flat 설정"
                                        "(conf/appo/task/go2_joystick_flat/base.yaml, 58ec02e의 부모 697f45a)에서 policy·critic의 "
                                        "base_ang_vel은 gyro 센서, critic의 base_lin_vel은 local_linvel 센서(builtin_sensor)를 읽었다. "
                                        "설정에서 이름이 몸체 속도 항인 것은 보상 4항뿐이고, 원전 커밋 메시지도 이 4항이 0을 읽었다고 적는다. "
                                        "다른 함수의 내부 동작은 설정에 드러나지 않아 이 근거의 범위 밖이다",
        "no_tuning_rule": "결함 정의는 원전에서만 정한다. 이번 결과를 보고 크기·항목·정의를 바꾸거나 다른 seed로 재시도하지 않는다. "
                          "시도·최종 표식은 프로토콜 SHA별이므로, 최종 결과가 나온 뒤 같은 결함으로 새 프로토콜 버전을 만들지 않는다"
                          "(만들어야 하면 그 사유를 새 프로토콜에 적는다)",
    },
    "source": {
        "framework": "UniLab(Motphys), MotrixSim·MuJoCo 백엔드. Isaac Lab 버그가 아니다",
        "workaround_commit": {"url": "https://github.com/Motphys/UniLab/commit/58ec02ea8b650e6b90db222188a770932d6b9676",
                              "author": "TATP-233",
                              "excerpts": ["The migrated quadruped flat/rough owners read tracking_lin_vel, tracking_ang_vel, lin_vel_z, and ang_vel_xy",
                                           "identically zero for the root body",
                                           "trained with no velocity feedback (reward 28.3 vs legacy 39.9, tracking stuck at 0.44, entropy never collapsed)",
                                           "velocity penalties are live again and tracking_lin_vel climbs 0.37 -> 0.78"]},
        "fix_pr": {"url": "https://github.com/Motphys/UniLab/pull/1254", "merged": "2026-08-23",
                   "squash_commit": "https://github.com/Motphys/UniLab/commit/2a3bcea728e71fa1225211ae9c4cf8dcf94be5de",
                   "title": "fix: pin get_body_*_vel_b to analytical body-frame semantics across backends",
                   "excerpts": ["the root body read exactly zero (verified empirically on both backends)",
                                "back to the generic track_lin_vel_xy_exp / track_ang_vel_z_exp / lin_vel_z_l2 / ang_vel_xy_l2 terms"]},
        "contract_discussion": "https://github.com/Motphys/UniLab/discussions/1246",
        "excerpt_note": "발췌는 gh api로 읽은 커밋 메시지 원문이다(2026-10-13 대조). PR 발췌 두 개는 PR 본문이 아니라 squash 커밋 "
                        "메시지에 있고, 두 번째 발췌는 원문 줄바꿈을 공백 하나로 합쳤다",
    },
    "mapping": [
        {"source_term": "tracking_lin_vel", "isaaclab_function": "track_lin_vel_xy_exp", "reads": "root_lin_vel_b[:, :2]",
         "go2_flat_term": "track_lin_vel_xy_exp", "weight": 1.5, "std": "sqrt(0.25)",
         "effect": "exp(-|명령 xy|²/std²): 행동과 무관한 명령 함수"},
        {"source_term": "tracking_ang_vel", "isaaclab_function": "track_ang_vel_z_exp", "reads": "root_ang_vel_b[:, 2]",
         "go2_flat_term": "track_ang_vel_z_exp", "weight": 0.75, "std": "sqrt(0.25)",
         "effect": "exp(-명령 z²/std²): 행동과 무관한 명령 함수"},
        {"source_term": "lin_vel_z", "isaaclab_function": "lin_vel_z_l2", "reads": "root_lin_vel_b[:, 2]",
         "go2_flat_term": "lin_vel_z_l2", "weight": -2.0, "effect": "항상 0"},
        {"source_term": "ang_vel_xy", "isaaclab_function": "ang_vel_xy_l2", "reads": "root_ang_vel_b[:, :2]",
         "go2_flat_term": "ang_vel_xy_l2", "weight": -0.05, "effect": "항상 0"},
    ],
    "isaaclab_locations": {"version": "v2.1.1", "commit": "90b79bb2", "rewards_function_lines": FUNCTION_LINES,
                           "velocity_env_cfg_lines": [235, 243], "go2_rough_env_cfg_weight_lines": [56, 57]},
    "task": TASK, "seed": SEED, "num_envs": NUM_ENVS, "max_iterations": MAX_ITERATIONS, "action_scale": ACTION_SCALE,
    "final_checkpoint": FINAL_CHECKPOINT,
    "training_entry": "bench/train_with_upstream_fault.py(TRIAGE_UPSTREAM_FAULT=U1), run_case.ps1과 같은 train.py 인자",
    "reference": {
        "run": REFERENCE_RUN, "fixed_report": "evals/results/p0c_fixed_eval_20261009/runs/baseline_p0c_s2026.json",
        "bands_source": "evals/results/p0c_fixed_eval_20261009/summary.json",
        "reuse": "P0-C 정상 실행(같은 task·seed·예산, bench/train_with_fault.py NONE)의 학습·고정 평가를 다시 쓰고 정상 학습은 하지 않는다",
    },
    "seed_choice": {
        "decision": "seed 2026, 기준은 P0-C baseline_p0c_s2026 재사용(학습 1회)",
        "pros": ["학습 1회로 끝난다(미사용 seed는 정상 학습까지 2회)",
                 "같은 task·seed·예산의 기준 학습·고정 평가가 이미 검증돼 있고 v4·W1이 같은 기준을 썼다",
                 "T1 X2가 P0-C baseline_p0c_s<seed>를 짝 기준으로 재사용한 선례가 있다",
                 "P0-B2에서 같은 seed·설정의 100회 학습을 다시 하면 체크포인트까지 같았다(baseline_repeat1_s42, "
                 "docs/P0-B2-FIXED-EVAL.md 75행). 4096 env × 300회에서는 확인하지 않았다"],
        "cons": ["seed 2026은 이미 본 seed(P0-C holdout·v4·W1)라 이 smoke는 미관측 seed 평가가 아니다",
                 "기준은 2026-10-09 학습이라 그사이 런타임 변화가 섞일 수 있다 → 평가 런타임 버전 일치와 params 일치를 유효 조건으로 둔다",
                 "런타임 일치는 고정 평가 보고서의 versions 필드(torch·GPU 이름·isaaclab·isaaclab_tasks·isaaclab_rl·rsl-rl-lib)만 본다. "
                 "isaacsim 값은 null이고 GPU 드라이버와 Isaac Lab 트리 전체는 결합하지 않는다(결합은 매핑 3파일 SHA)",
                 "기준은 다른 진입점(bench/train_with_fault.py, TRIAGE_FAULT=NONE)으로 학습됐다. 코드를 읽으면 NONE은 아무것도 "
                 "바꾸지 않지만 실행으로 대조하지는 않았다",
                 "U1을 뒤에 미관측 seed 평가에 쓰면 seed 2026은 그 평가에서 뺀다"],
        "alternative": "미사용 seed(예: 2041) + 같은 날 새 정상 학습: 짝의 날짜·런타임이 같지만 GPU 약 두 배이고, "
                       "이 smoke의 질문(실패가 나는가)에는 seed의 새로움이 필요하지 않다",
    },
    "execution": {
        "owned_process": "bench/run_runtime_readback_smoke.run_owned: 소유 Windows Job, 시간 초과 시 프로세스 트리 종료, stdout·stderr 분리",
        "watchdog_s": WATCHDOG_S,
        "gpu_admission_profile": GPU_PROFILE,
        "gpu_admission": "학습 전 GPU가 사용 중이면 아무것도 쓰지 않고 실행하지 않는다. 고정 평가 전에는 최대 "
                         f"{EVAL_GPU_WAIT_S}초 동안 {EVAL_GPU_POLL_S}초 간격으로 다시 확인한다",
        "fixed_evaluation_entry": "evals/fixed_eval.py --protocol bench/protocols/fixed_eval_v1.json --jobs <jobs.json> --headless(v4·W1과 같은 호출)",
        "reverify": "고정 평가 직전과 평가 뒤에 프로토콜·결합 코드를 다시 확인하고(결합 경로 clean·커밋됨, 시작 커밋과 현재 HEAD 사이 "
                    "결합 경로 diff 없음), 평가 뒤 체크포인트 SHA를 다시 계산한다. 결합 밖 파일의 커밋은 막지 않고 본 커밋을 모두 기록한다",
        "bound_infrastructure_note": "공용 실행 도구 3파일(record_evidence_media·run_runtime_readback_smoke·runtime_readback_smoke)도 "
                                     "SHA로 결합하므로, 다른 작업이 이 파일을 고치면 verify가 거부되고 새 프로토콜 버전이 필요하다",
        "skip_rule": "학습 측 검사(종료 코드·체크포인트·교체 확인 줄·텔레메트리 0·params)가 하나라도 어긋나면 판정에 쓰지 않을 고정 평가는 하지 않는다",
        "attempts": {"1": RUN_NAMES[1], "2": RUN_NAMES[2]},
        "retry_rule": "시도 2는 같은 프로토콜에서 시도 1의 마지막 결과(이어 하기를 했으면 그 결과)가 끝났고 invalid_cause가 infrastructure일 때, "
                      "그 결과의 tag를 지정해 한 번만 한다. 결함 정의·seed·판정 기준은 같다",
        "resume_rule": "학습 측 검사(종료 코드·체크포인트·교체 확인 줄)를 통과한 뒤 추출·평가 단계의 인프라 원인으로 invalid가 된 결과는 "
                       "새 tag로 같은 학습의 추출·평가·판정만 한 번 다시 한다. 학습은 다시 하지 않고, 다음 시도가 이미 시작됐으면 하지 않는다",
        "final_rule": "처음 나온 유효 결과(failure_present 또는 failure_absent)가 최종이다. 그 뒤로는 시도 2·이어 하기를 모두 거부한다",
    },
    "outcomes": {
        "failure_present": "유효 조건을 모두 채우고, U1 고정 평가가 같은 seed 정상 고정 평가 대비 p0c 밴드에서 unhealthy",
        "failure_absent": "유효 조건을 모두 채우고 healthy",
        "invalid": "아래 유효 조건 중 하나라도 어긋남(판정이 undetermined인 경우 포함). 실패 유무를 판정하지 않는다",
    },
    "validity_checks": {
        "pipeline_completed": "실행기 단계(학습·추출·params 대조·평가 전 재확인·고정 평가·사후 확인)가 예외·시간 초과·소스 변경 없이 끝남",
        "training_exit_zero": "학습 프로세스 종료 코드 0",
        "final_checkpoint_present": "실행 폴더에 model_299.pt",
        "injection_readback": "학습 stdout의 [upstream_fault] 줄 하나: 4함수 본문 교체가 유지되고 보상 관리자가 그 함수 객체를 씀",
        "penalties_zero_in_telemetry": "텔레메트리 Episode_Reward/lin_vel_z_l2·ang_vel_xy_l2가 300시점 모두 정확히 0(정상 실행에서는 0이 아님)",
        "params_match_reference": "params/env.yaml·agent.yaml이 기준 실행과 agent run_name 외에 같다",
        "reference_walks": "기준 정책이 고정 평가에서 걷는다(정지 대비 오차 비 0.5 미만, 낙상 0.05 이하)",
        "fixed_report_valid": "고정 평가 보고서가 결합한 fixed_eval_v1 계약 SHA(기준 보고서와 같음)·체크포인트 SHA·action scale을 통과",
        "runtime_matches_reference": "고정 평가 런타임 버전이 기준 보고서와 같다",
        "verdict_determined": "p0c 밴드 판정이 healthy 또는 unhealthy다",
    },
    "invalid_causes": {
        "rule": "validity_checks 순서에서 처음 어긋난 검사의 원인 하나를 invalid_cause로 적고, 어긋난 검사 전체는 failed_checks에 남긴다. "
                "학습 뒤 단계(추출 이후)에서 실행기가 멈췄어도 학습 3검사(종료 코드·체크포인트·교체 확인 줄)와, 텔레메트리를 읽었으면 "
                "벌점 0 검사 중 어긋난 것이 있으면 그 원인을 적는다",
        "by_check": {"pipeline_completed": "infrastructure", "training_exit_zero": "training_process",
                     "final_checkpoint_present": "training_process", "injection_readback": "injection",
                     "penalties_zero_in_telemetry": "injection", "params_match_reference": "comparability",
                     "reference_walks": "comparability", "fixed_report_valid": "evaluation",
                     "runtime_matches_reference": "comparability", "verdict_determined": "evaluation"},
        "fault_induced_crash": "결함 때문에 학습 프로세스가 죽거나(종료 코드≠0) 체크포인트가 없어도 training_process로 invalid다. "
                               "실행 한 번으로 결함 원인과 인프라 원인을 가를 수 없어 failure_present로 세지 않는다. "
                               "학습 단계 watchdog 초과는 원인과 관계없이 infrastructure로 기록된다",
        "judge_failure": "판정 계산 자체가 예외로 끝나면 실패 단계 judge, invalid_cause runner로 남긴다. 실행기 결함이라 재실행·이어 하기 대상이 아니다",
    },
    "prediction": {"outcome": "failure_present",
                   "reason": "원전이 같은 결함에서 추종 정체(tracking stuck at 0.44)를 보고했다. 추종 보상이 행동과 무관한 명령 함수가 되고 "
                             "속도 페널티가 사라지므로 고정 평가 선속도 오차 비가 p0c 밴드(1.1513)를 넘을 것으로 본다"},
    "later_uses": {
        "failure_present": ["동결 probe·고리 시험: P_reward는 실제 root_lin_vel_b로 다시 계산하므로 reward abnormal로 확정될 것으로 본다. "
                            "등록 교정 restore_body_frame_reward는 4항 중 track_lin_vel_xy_exp 하나만 다루므로 교정 단계의 '등록 범위 밖' 처리를 별도로 사전등록해 시험한다",
                            "D3 해로운 변경 후보(research_next 0절): 외부 원인 확정 결함 중 smoke에서 실패가 확인된 것을 우선한다",
                            "둘 다 별도 사전등록이 필요하며 이 smoke 결과로 진단 성능을 말하지 않는다"],
        "failure_absent": "이 task·예산·seed에서 실패 없음으로 기록하고 후보에서 내린다. 결함 크기나 정의를 키우지 않는다",
        "invalid": "invalid_cause와 실패 단계를 기록한다. 다시 실행하거나 이어 하는 것은 execution.retry_rule·resume_rule만 따르며 "
                   "판정 기준과 결함 정의는 바꾸지 않는다",
    },
    "used_for": "실패 유무 판정만. 고리·교정·D3에는 이번 판정을 쓰지 않는다",
    "gpu_estimate": {
        "unit": "부모 프로세스 벽시계(초). GPU 커널 시간이나 절감량이 아니다",
        "train_s": [429.3, 873.6],
        "train_basis": "P0-C·T1 4096 env × 300회 신규 학습 27건의 run_case 학습 wall(bench/private/*_s202[678].meta.json). "
                       "GPU 경합이 기록된 e01_s2028(1871.3 s)을 뺀 26건의 범위와 중앙값 506.05 s. 상한 873.6 s(ref_e01_s2028)도 "
                       "같은 GPU 메모리 점유 구간의 실행이다(docs/T1-EXTERNAL-FAULTS.md 178행)",
        "telemetry_extract_s": "측정 기록 없음(1분 미만으로 추정)",
        "fixed_eval_s": [51.2, 64.5],
        "fixed_eval_basis": "v4·W1 단일 고정 평가 parent wall(공개 training_evidence.json)",
        "total_s": [480, 1000],
    },
    "wording": {
        "failure_present": "UniLab에서 원인이 확정된 결함(보상 4항이 몸체 속도 0을 읽음)의 의미를 옮긴 학습이 같은 seed 정상 학습 대비 p0c 밴드를 벗어났다. "
                           "이미 본 seed 하나의 단일 실행이며, 진단 성능이나 다른 결함·task로의 일반화를 말하지 않는다.",
        "failure_absent": "UniLab에서 원인이 확정된 결함의 의미를 옮긴 학습이 같은 seed 정상 학습 대비 p0c 밴드 안이었다. "
                          "이 task·예산·seed에서 학습 실패가 재현되지 않았고, 결함 크기나 정의를 바꿔 다시 시도하지 않는다.",
        "invalid": "smoke가 사전등록한 유효 조건을 채우지 못해 학습 실패 유무를 판정하지 않는다.",
    },
}
DYNAMIC = {"source_files", "isaaclab_sources", "reference_inputs"}


def sha(path: Path, lf: bool = False) -> str:
    data = Path(path).read_bytes()
    return hashlib.sha256(data.replace(b"\r\n", b"\n") if lf else data).hexdigest()


def read(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_new(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=1, allow_nan=False) + "\n")


def standstill_ratio(report: dict) -> float:
    """같은 고정 평가 report 안의 선속도 RMSE ÷ 제자리 정책 RMSE."""
    return report["metrics"]["lin_vel_rmse_mps"] / report["standstill"]["lin_vel_rmse_mps"]


def walks(report: dict) -> bool:
    behavior = C._behavior(report)
    return (behavior["err_xy"] < C.WALK_FRACTION * report["standstill"]["lin_vel_rmse_mps"]
            and behavior["fall_frac"] <= C.REF_FALL_MAX)


# --- 입력 결합 ---
def function_lines(rewards_source: str) -> dict[str, list[int]]:
    tree = ast.parse(rewards_source)
    return {node.name: [node.lineno, node.end_lineno] for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in FUNCTION_LINES}


def isaaclab_sources(isaaclab: Path) -> dict:
    """로컬 Isaac Lab의 매핑 파일 SHA와 4함수 줄 범위. 줄 범위가 등록값과 다르면 거부한다."""
    files = {key: isaaclab / relative for key, relative in ISAACLAB_FILES.items()}
    lines = function_lines(files["rewards"].read_text(encoding="utf-8"))
    if lines != FUNCTION_LINES:
        raise ValueError(f"Isaac Lab rewards.py의 4함수 위치가 등록값과 다르다: {lines}")
    return {key: {"path": ISAACLAB_FILES[key], "sha256_lf": sha(path, True)} for key, path in files.items()}


def reference_paths(isaaclab: Path) -> dict[str, Path]:
    """P0-C 기준 실행의 meta·체크포인트·params·고정 평가. meta가 등록 task·seed·예산과 다르면 거부한다."""
    meta_path = ROOT / "bench/private" / f"{REFERENCE_RUN}.meta.json"
    meta = read(meta_path)
    expected = {"task": TASK, "seed": SEED, "num_envs": NUM_ENVS, "max_iterations": MAX_ITERATIONS,
                "fault": "NONE", "train_exit_code": 0}
    if any(meta.get(k) != v for k, v in expected.items()):
        raise ValueError("기준 실행 meta가 등록값과 다르다")
    log_root = (isaaclab / "logs/rsl_rl/unitree_go2_flat").resolve()
    run_dir = Path(meta["run_dir"]).resolve()
    if run_dir.parent != log_root or not run_dir.name.endswith("_" + REFERENCE_RUN):
        raise ValueError("기준 실행 폴더가 등록 로그 폴더 밖이다")
    return {"metadata": meta_path, "run_dir": run_dir, "checkpoint": run_dir / FINAL_CHECKPOINT,
            "environment": run_dir / "params/env.yaml", "agent": run_dir / "params/agent.yaml",
            "fixed_report": P0C / "runs" / f"{REFERENCE_RUN}.json", "summary": P0C / "summary.json",
            "telemetry": REFERENCE_TELEMETRY}


def reference_inputs(isaaclab: Path) -> dict:
    paths = reference_paths(isaaclab)
    summary = read(paths["summary"])
    report = read(paths["fixed_report"])
    rows = [row for row in summary["runs"] if row["run"] == REFERENCE_RUN]
    if len(rows) != 1:
        raise ValueError("P0-C 요약에서 기준 실행 행을 하나로 찾지 못했다")
    if report["protocol_sha256_lf"] != sha(FIXED_EVAL_PROTOCOL, True):
        raise ValueError("기준 고정 평가의 프로토콜 SHA가 현재 fixed_eval_v1과 다르다")
    telemetry = read(paths["telemetry"])["series"]
    if any(not any(v != 0 for v in telemetry[key]) for key in PENALTY_SERIES):
        raise ValueError("기준 텔레메트리의 속도 페널티가 이미 0이다(주입 확인 기준이 될 수 없다)")
    return {"run": REFERENCE_RUN, "bands": summary["bands"],
            "checkpoint_sha256": sha(paths["checkpoint"]),
            "fixed_eval_protocol_sha256_lf": report["protocol_sha256_lf"],
            "hashes": {key: sha(paths[key], key in ("fixed_report", "summary", "telemetry"))
                       for key in ("metadata", "environment", "agent", "fixed_report", "summary", "telemetry")},
            "runtime_versions": report["versions"], "walks": walks(report),
            "standstill_ratio": standstill_ratio(report)}


def dynamic_fields(isaaclab: Path) -> dict:
    return {"source_files": {name: sha(ROOT / name, True) for name in CODE},
            "isaaclab_sources": isaaclab_sources(isaaclab), "reference_inputs": reference_inputs(isaaclab)}


def check_protocol(protocol: dict, dynamic: dict) -> None:
    if set(protocol) != set(FIXED) | DYNAMIC:
        raise ValueError("프로토콜 필드가 다르다")
    if any(protocol[k] != v for k, v in FIXED.items()):
        raise ValueError("고정 계약이 바뀌었다")
    if any(protocol[k] != v for k, v in dynamic.items()):
        raise ValueError("결합한 코드·Isaac Lab 소스·기준 입력이 바뀌었다")


def prepare(isaaclab: Path) -> dict:
    write_new(PROTOCOL, {**deepcopy(FIXED), **dynamic_fields(isaaclab)})
    return {"status": "prepared_commit_before_execute", "path": PROTOCOL.relative_to(ROOT).as_posix()}


def verify(isaaclab: Path, verify_committed=None) -> tuple[dict, str]:
    protocol = read(PROTOCOL)
    check_protocol(protocol, dynamic_fields(isaaclab))
    if verify_committed is None:
        import run_runtime_readback_smoke as P
        verify_committed = P.verify_committed
    try:
        commit = verify_committed(ROOT, set(CODE) | {PROTOCOL.relative_to(ROOT).as_posix()})
    except subprocess.CalledProcessError as error:
        raise ValueError("결합 코드나 프로토콜이 커밋되지 않았다") from error
    return protocol, commit


# --- 판정(GPU 없이 시험한다) ---
def flat_params(values, prefix: str = "") -> dict:
    if isinstance(values, dict):
        out = {}
        for key, value in values.items():
            out.update(flat_params(value, f"{prefix}.{key}" if prefix else str(key)))
        return out
    if isinstance(values, list):
        out = {}
        for index, value in enumerate(values):
            out.update(flat_params(value, f"{prefix}[{index}]"))
        return out
    return {prefix: values}


def load_params(run_dir: Path) -> dict:
    return {f"{name}:{key}": value for name in ("env.yaml", "agent.yaml")
            for key, value in flat_params(yaml.load((run_dir / "params" / name).read_text(encoding="utf-8"),
                                                    Loader=_TolerantLoader)).items()}


def params_differences(run: dict, reference: dict) -> list[str]:
    return sorted(key for key in set(run) | set(reference) if run.get(key) != reference.get(key))


def parse_readback(log_text: str) -> dict | None:
    lines = [line for line in log_text.splitlines() if line.startswith("[upstream_fault] ")]
    if len(lines) != 1:
        return None
    try:
        return json.loads(lines[0][len("[upstream_fault] "):])
    except json.JSONDecodeError:
        return None


def readback_ok(readback: dict | None) -> bool:
    names = set(U.SWAPS[FAULT])
    return (isinstance(readback, dict) and readback.get("fault") == FAULT and readback.get("ok") is True
            and set(readback.get("swapped") or {}) == names and all(readback["swapped"].values())
            and set(readback.get("bound_to_reward_terms") or {}) == names and all(readback["bound_to_reward_terms"].values()))


def penalties_zero(telemetry: dict | None) -> bool:
    series = (telemetry or {}).get("series") or {}
    return all(isinstance(series.get(key), list) and len(series[key]) == MAX_ITERATIONS
               and all(value == 0 for value in series[key]) for key in PENALTY_SERIES)


def training_checks(evidence: dict) -> dict[str, bool]:
    """학습 산출물만으로 정해지는 유효 조건. 하나라도 어긋나면 고정 평가를 하지 않는다."""
    differences = evidence.get("params_differences")
    return {"training_exit_zero": evidence.get("train_exit_code") == 0,
            "final_checkpoint_present": bool(evidence.get("checkpoint_sha256")),
            "injection_readback": readback_ok(evidence.get("readback")),
            "penalties_zero_in_telemetry": penalties_zero(evidence.get("telemetry")),
            "params_match_reference": differences is not None and set(differences) <= ALLOWED_PARAM_DIFFS}


def judge(protocol: dict, evidence: dict) -> dict:
    """사전등록 유효 조건과 p0c 밴드로 결과 범주를 정한다. evidence는 실행에서 모은 원자료다."""
    reference = protocol["reference_inputs"]
    checks = {"pipeline_completed": evidence.get("failure") is None, **training_checks(evidence),
              "reference_walks": reference["walks"] is True,
              "fixed_report_valid": False, "runtime_matches_reference": False, "verdict_determined": None}
    checks = {name: checks[name] for name in protocol["validity_checks"]}
    report = evidence.get("report")
    if report is not None and checks["final_checkpoint_present"]:
        try:
            C.validate_fixed_report(report, {"name": evidence["run_name"], "checkpoint_file": FINAL_CHECKPOINT,
                                             "checkpoint_sha256": evidence["checkpoint_sha256"],
                                             "action_scale": ACTION_SCALE,
                                             "protocol_sha256_lf": reference["fixed_eval_protocol_sha256_lf"]})
            checks["fixed_report_valid"] = True
            checks["runtime_matches_reference"] = report.get("versions") == reference["runtime_versions"]
        except (ValueError, KeyError, TypeError):
            pass
    result = {"verdict": None, "behavior": None}
    if all(ok for name, ok in checks.items() if name != "verdict_determined"):
        behavior = C._behavior(report)
        verdict = RC.verdict(RC.compare(behavior, C._behavior(evidence["reference_report"])), reference["bands"])
        checks["verdict_determined"] = verdict["label"] in ("healthy", "unhealthy")
        result.update(verdict=verdict, behavior=behavior, standstill_ratio=standstill_ratio(report))
    failed_checks = [name for name, ok in checks.items() if ok is False]
    cause = None
    if failed_checks:
        outcome = "invalid"
        cause = protocol["invalid_causes"]["by_check"][failed_checks[0]]
        stage = (evidence.get("failure") or {}).get("stage")
        priority = [*TRAINING_CHECKS, *(["penalties_zero_in_telemetry"] if evidence.get("telemetry") is not None else [])]
        failed_priority = [name for name in priority if name in failed_checks]
        if failed_checks[0] == "pipeline_completed" and stage == "judge":
            cause = "runner"
        elif failed_checks[0] == "pipeline_completed" and stage != "training" and failed_priority:
            cause = protocol["invalid_causes"]["by_check"][failed_priority[0]]  # 학습 뒤 단계 실패보다 학습 산출물 검사가 앞선다
    else:
        outcome = "failure_present" if result["verdict"]["label"] == "unhealthy" else "failure_absent"
    prediction = protocol["prediction"]["outcome"]
    return {"checks": checks, "failed_checks": failed_checks, **result, "outcome": outcome, "invalid_cause": cause,
            "prediction": prediction, "prediction_match": None if outcome == "invalid" else outcome == prediction,
            "wording": protocol["wording"][outcome]}


def telemetry_summary(telemetry: dict | None) -> dict | None:
    """공개 결과용 요약. 비유한 값은 JSON에 쓸 수 없어 None으로 두고 개수를 센다."""
    series = (telemetry or {}).get("series") or {}
    keys = PENALTY_SERIES + TRACKING_SERIES + ("Metrics/base_velocity/error_vel_xy", "Train/mean_episode_length")
    def finite(value) -> bool:
        return isinstance(value, (int, float)) and math.isfinite(value)

    out = {}
    for key in keys:
        values = series.get(key)
        if not values:
            continue
        kept = [v for v in values if finite(v)]
        out[key] = {"first": values[0] if finite(values[0]) else None,
                    "last": values[-1] if finite(values[-1]) else None,
                    "min": min(kept, default=None), "max": max(kept, default=None),
                    "nonfinite": len(values) - len(kept)}
    return out


# --- 실행(GPU) ---
def train_command(isaaclab: Path, run_name: str) -> list[str]:
    return [str(isaaclab / "_isaac_sim" / "python.bat"), str(ROOT / "bench" / "train_with_upstream_fault.py"),
            "--task", TASK, "--num_envs", str(NUM_ENVS), "--max_iterations", str(MAX_ITERATIONS),
            "--seed", str(SEED), "--headless", "--run_name", run_name]


def telemetry_path(run_name: str) -> Path:
    return RUNS_DIR / f"{run_name}.telemetry.json"


def find_run_dir(isaaclab: Path, run_name: str) -> list[Path]:
    log_root = isaaclab / "logs/rsl_rl/unitree_go2_flat"
    if not log_root.is_dir():
        return []
    return sorted(d for d in log_root.iterdir()
                  if d.is_dir() and re.fullmatch(rf"\d{{4}}-\d\d-\d\d_\d\d-\d\d-\d\d_{run_name}", d.name))


def finished(execution: dict) -> bool:
    """소유 Job이 배정됐고 시간 초과·예외 없이 스스로 끝났으며 남은 프로세스가 없다. 종료 코드는 따로 본다."""
    return (execution.get("job_assigned_before_release") is True and execution.get("natural_exit") is True
            and execution.get("timed_out") is False and execution.get("error_type") is None
            and execution.get("cleanup_verified") is True)


def _owned(stage_dir: Path, args: list[str], cwd: Path, watchdog_s: float) -> dict:
    import run_runtime_readback_smoke as P
    stage_dir.mkdir(parents=True, exist_ok=True)
    execution = P.run_owned(stage_dir, args, cwd, watchdog_s)
    write_new(stage_dir / "execution.json", execution)
    return execution


def _admission() -> dict:
    import record_evidence_media as M
    return M.query_gpu(GPU_PROFILE)


def _wait_for_gpu(private: Path) -> None:
    started = time.monotonic()
    while True:
        gpu = _admission()
        waited = time.monotonic() - started
        if not gpu["busy"] or waited >= EVAL_GPU_WAIT_S:
            write_new(private / "gpu_before_fixed_evaluation.json", {**gpu, "waited_s": round(waited, 1)})
            if gpu["busy"]:
                raise RuntimeError(f"고정 평가 전 GPU가 {EVAL_GPU_WAIT_S}초 동안 비지 않았다: {gpu['busy_reasons']}")
            return
        time.sleep(EVAL_GPU_POLL_S)


def _check_tag(tag) -> None:
    if not isinstance(tag, str) or not re.fullmatch(TAG_RE, tag):
        raise ValueError("tag는 upstream_u1_smoke_<YYYYMMDD>[접미]")


def _consumed(kind: str) -> Path:
    return PRIVATE / "consumed" / f"{sha(PROTOCOL, True)[:16]}_{kind}.json"


def _refuse_existing(tag: str, *paths: Path) -> None:
    if (RESULTS / tag).exists() or (PRIVATE / tag).exists() or any(path.exists() for path in paths):
        raise FileExistsError("같은 tag·시도·실행 이름의 산출물이 이미 있다. 덮어쓰지 않는다")


def _context(tag: str, isaaclab: Path, protocol: dict, commit: str, attempt: int, **extra) -> dict:
    return {"tag": tag, "isaaclab": isaaclab, "protocol": protocol, "commit": commit,
            "protocol_sha256_lf": sha(PROTOCOL, True), "attempt": attempt, "run_name": RUN_NAMES[attempt],
            "out": RESULTS / tag, "private": PRIVATE / tag, "run_dir": None, "stage": "training", "failure": None,
            "executions": {}, "evidence": {"run_name": RUN_NAMES[attempt]},
            "walls": {"train_s": None, "telemetry_extract_s": None, "fixed_eval_s": None}, "commits_seen": [commit],
            "training_commit": commit, "retry_of": None, "resumed_from": None, **extra}


def bound_sources_same(start: str, now: str) -> bool:
    """두 커밋 사이에 결합 경로(CODE·프로토콜)의 diff가 없는가. git 오류는 다르다고 본다."""
    paths = [*CODE, PROTOCOL.relative_to(ROOT).as_posix()]
    return subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet", start, now, "--", *paths],
                          capture_output=True).returncode == 0


def _reverify(ctx: dict) -> None:
    """결합 경로만 본다. 같은 checkout의 다른 작업이 결합 밖 파일을 커밋해도 실행을 무효로 하지 않는다."""
    protocol, commit = verify(ctx["isaaclab"])
    if protocol != ctx["protocol"]:
        raise ValueError("실행 중 프로토콜이나 결합 소스가 바뀌었다")
    if commit != ctx["commit"] and not bound_sources_same(ctx["commit"], commit):
        raise ValueError("시작 커밋과 현재 HEAD 사이에 결합 경로가 바뀌었다")
    if commit not in ctx["commits_seen"]:
        ctx["commits_seen"].append(commit)


def _process_created(pid: int) -> int | None:
    import run_runtime_readback_smoke as P
    return P._process_creation_filetime(pid)


def _train(ctx: dict) -> None:
    isaaclab, evidence = ctx["isaaclab"], ctx["evidence"]
    os.environ["TRIAGE_UPSTREAM_FAULT"] = FAULT
    try:
        execution = _owned(ctx["private"] / "train", train_command(isaaclab, ctx["run_name"]), isaaclab,
                           WATCHDOG_S["training"])
    finally:
        os.environ.pop("TRIAGE_UPSTREAM_FAULT", None)
    ctx["executions"]["training"] = execution
    ctx["walls"]["train_s"] = round(execution["wall_s"], 1)
    evidence["train_exit_code"] = execution["exit_code"]
    evidence["readback"] = parse_readback((ctx["private"] / "train/stdout.log").read_text(encoding="utf-8", errors="replace"))
    run_dirs = find_run_dir(isaaclab, ctx["run_name"])
    ctx["run_dir"] = run_dirs[0] if len(run_dirs) == 1 else None
    checkpoint = ctx["run_dir"] / FINAL_CHECKPOINT if ctx["run_dir"] else None
    evidence["checkpoint_sha256"] = sha(checkpoint) if checkpoint and checkpoint.is_file() else None
    if not finished(execution):
        raise RuntimeError("학습 프로세스가 시간 안에 스스로 끝나지 않았거나 정리되지 않았다")


def _after_training(ctx: dict) -> None:
    """추출·params 대조·평가 전 재확인·고정 평가·사후 확인. ctx["stage"]에 진행 단계를 남기고 예외는 그대로 올린다."""
    isaaclab, evidence, run_dir, private = ctx["isaaclab"], ctx["evidence"], ctx["run_dir"], ctx["private"]
    if run_dir is None:
        return  # 체크포인트 검사에서 invalid
    ctx["stage"] = "telemetry_extract"
    telemetry = telemetry_path(ctx["run_name"])
    if not telemetry.exists():
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        execution = _owned(private / "extract", [str(isaaclab / "_isaac_sim" / "python.bat"),
                                                 str(ROOT / "bench" / "extract_tb.py"), str(run_dir), str(telemetry)],
                           isaaclab, WATCHDOG_S["telemetry_extract"])
        ctx["executions"]["telemetry_extract"] = execution
        ctx["walls"]["telemetry_extract_s"] = round(execution["wall_s"], 1)
        # 추출은 성공해도 0이 아닌 종료 코드를 남길 때가 있어 결과 파일로 판정한다(run_case.ps1과 같다).
        if not finished(execution) or not telemetry.is_file():
            raise RuntimeError("텔레메트리 추출이 끝나지 않았거나 파일이 없다")
    evidence["telemetry"] = read(telemetry)
    evidence["telemetry_sha256"] = sha(telemetry)
    ctx["stage"] = "params"
    evidence["params_differences"] = params_differences(load_params(run_dir),
                                                        load_params(reference_paths(isaaclab)["run_dir"]))
    if not all(training_checks(evidence).values()):
        return  # 판정에 쓰지 않을 고정 평가는 하지 않는다
    ctx["stage"] = "pre_evaluation"
    _reverify(ctx)
    _wait_for_gpu(private)
    ctx["stage"] = "fixed_evaluation"
    report_path = ctx["out"] / "runs" / f"{ctx['run_name']}.json"
    jobs = private / "eval" / "jobs.json"
    write_new(jobs, {"action_scale": ACTION_SCALE, "jobs": [{"name": ctx["run_name"],
                                                             "checkpoint": str(run_dir / FINAL_CHECKPOINT),
                                                             "output": str(report_path)}]})
    execution = _owned(private / "eval", [str(isaaclab / "_isaac_sim" / "python.bat"), str(ROOT / "evals" / "fixed_eval.py"),
                                          "--protocol", str(FIXED_EVAL_PROTOCOL), "--jobs", str(jobs), "--headless"],
                       isaaclab, WATCHDOG_S["fixed_evaluation"])
    ctx["executions"]["fixed_evaluation"] = execution
    ctx["walls"]["fixed_eval_s"] = round(execution["wall_s"], 1)
    if not finished(execution) or execution["exit_code"] != 0 or not report_path.is_file():
        raise RuntimeError("고정 평가가 끝나지 않았거나 보고서가 없다")
    evidence["report"] = read(report_path)
    ctx["stage"] = "postflight"
    _reverify(ctx)
    if sha(run_dir / FINAL_CHECKPOINT) != evidence["checkpoint_sha256"]:
        raise ValueError("평가 중 체크포인트가 바뀌었다")
    after = _admission()
    write_new(private / "gpu_after.json", after)
    # PID는 재사용될 수 있어 생성 시각까지 같을 때만 소유 프로세스로 본다.
    owned = {pid: created for execution in ctx["executions"].values()
             for pid, created in execution["owned_process_identities"].items()}
    if any(str(row["pid"]) in owned and _process_created(int(row["pid"])) == owned[str(row["pid"])]
           for row in after["processes"]):
        raise ValueError("소유 프로세스가 GPU에 남아 있다")


def _record_failure(ctx: dict, error: Exception) -> None:
    ctx["failure"] = {"stage": ctx["stage"], "error_type": type(error).__name__}
    write_new(ctx["private"] / "failure.json", {**ctx["failure"], "detail": str(error)})


def _finish(ctx: dict) -> dict:
    """판정하고 공개 result.json과 비공개 execution.json을 한 번씩 쓴다. 판정이 깨져도 invalid로 남긴다."""
    evidence = ctx["evidence"]
    ctx["stage"] = "judge"
    try:
        evidence["reference_report"] = read(P0C / "runs" / f"{REFERENCE_RUN}.json")
        verdict = judge(ctx["protocol"], {**evidence, "failure": ctx["failure"]})
    except Exception as error:
        if ctx["failure"] is None:
            _record_failure(ctx, error)
        else:
            write_new(ctx["private"] / "judge_error.json", {"error_type": type(error).__name__, "detail": str(error)})
        # 실패가 기록되면 판정은 기준 보고서·밴드 계산 없이 invalid로 끝난다.
        try:
            verdict = judge(ctx["protocol"], {**evidence, "failure": ctx["failure"]})
        except Exception:
            verdict = {"checks": None, "failed_checks": ["pipeline_completed"], "verdict": None, "behavior": None,
                       "outcome": "invalid", "invalid_cause": "runner", "prediction": FIXED["prediction"]["outcome"],
                       "prediction_match": None, "wording": FIXED["wording"]["invalid"]}
    write_new(ctx["private"] / "execution.json", {
        "run_dir": str(ctx["run_dir"]) if ctx["run_dir"] else None, "attempt": ctx["attempt"],
        "train_command": train_command(ctx["isaaclab"], ctx["run_name"]), "failure": ctx["failure"],
        "retry_of": ctx["retry_of"], "resumed_from": ctx["resumed_from"], "commits_seen": ctx["commits_seen"]})
    result = {"schema": "upstream_fault_smoke_result_v1", "tag": ctx["tag"], "fault": FAULT, "seed": SEED,
              "attempt": ctx["attempt"], "run_name": ctx["run_name"], "retry_of": ctx["retry_of"],
              "resumed_from": ctx["resumed_from"], "source_commit_sha": ctx["commit"],
              "training_source_commit_sha": ctx["training_commit"], "commits_seen": ctx["commits_seen"],
              "protocol_sha256_lf": ctx["protocol_sha256_lf"], "reference_run": REFERENCE_RUN,
              **verdict, "failure": ctx["failure"], "train_exit_code": evidence.get("train_exit_code"),
              "params_differences": evidence.get("params_differences"), "readback": evidence.get("readback"),
              "telemetry_sha256": evidence.get("telemetry_sha256"),
              "telemetry_summary": telemetry_summary(evidence.get("telemetry")),
              "checkpoint_sha256": evidence.get("checkpoint_sha256"), "parent_wall_s": ctx["walls"],
              "parent_wall_scope": FIXED["gpu_estimate"]["unit"], "scope": FIXED["scope"], "model_calls": 0}
    if result["outcome"] != "invalid":
        # execution.final_rule. 결과보다 먼저 써서, 그 사이에 멈춰도 다음 시도가 열리지 않게 한다.
        write_new(_consumed("final"), {"tag": ctx["tag"], "outcome": result["outcome"]})
    write_new(ctx["out"] / "result.json", result)
    return {"outcome": result["outcome"], "invalid_cause": result["invalid_cause"], "failure": result["failure"],
            "failed_checks": result["failed_checks"], "result": f"evals/results/{ctx['tag']}/result.json"}


def _previous(tag: str) -> tuple[dict, dict]:
    _check_tag(tag)
    return read(RESULTS / tag / "result.json"), read(PRIVATE / tag / "execution.json")


def _last_result(attempt: int) -> tuple[str | None, dict | None]:
    """시도의 마지막 tag와 결과. 이어 하기를 했으면 그 결과이고, 결과 파일이 아직 없으면(실행 중·중단) None이다."""
    resumed, started = _consumed(f"resume{attempt}"), _consumed(f"attempt{attempt}")
    marker = resumed if resumed.exists() else started if started.exists() else None
    if marker is None:
        return None, None
    tag = read(marker)["tag"]
    path = RESULTS / tag / "result.json"
    return tag, read(path) if path.is_file() else None


def execute(tag: str, isaaclab: Path, retry_of: str | None = None) -> dict:
    _check_tag(tag)
    protocol, commit = verify(isaaclab)
    attempt = 1
    if retry_of is not None:
        _check_tag(retry_of)
        last_tag, last = _last_result(1)
        if (retry_of != last_tag or last is None or last.get("outcome") != "invalid"
                or last.get("invalid_cause") != "infrastructure"):
            raise ValueError("시도 2는 시도 1의 마지막 결과(이어 하기를 했으면 그 결과)가 인프라 원인 invalid로 끝났을 때 그 tag로만 허용된다")
        attempt = 2
    run_name = RUN_NAMES[attempt]
    consumed = _consumed(f"attempt{attempt}")
    _refuse_existing(tag, _consumed("final"), consumed, telemetry_path(run_name), *find_run_dir(isaaclab, run_name))
    gpu = _admission()
    if gpu["busy"]:
        return {"status": "gpu_busy_no_launch", "busy_reasons": gpu["busy_reasons"], "training_launched": False}
    ctx = _context(tag, isaaclab, protocol, commit, attempt, retry_of=retry_of)
    write_new(consumed, {"tag": tag, "attempt": attempt, "run_name": run_name})
    write_new(ctx["private"] / "gpu_before_training.json", gpu)
    try:
        _train(ctx)
        _after_training(ctx)
    except Exception as error:
        _record_failure(ctx, error)
    return _finish(ctx)


def resume(tag: str, from_tag: str, isaaclab: Path) -> dict:
    """앞 실행의 학습 산출물을 그대로 두고 추출·평가·판정만 새 tag로 다시 한다(execution.resume_rule)."""
    _check_tag(tag)
    protocol, commit = verify(isaaclab)
    previous, execution = _previous(from_tag)
    failure = previous.get("failure") or {}
    if (previous.get("protocol_sha256_lf") != sha(PROTOCOL, True) or previous.get("outcome") != "invalid"
            or previous.get("invalid_cause") != "infrastructure" or failure.get("stage") not in RESUMABLE_STAGES
            or previous.get("resumed_from") is not None
            or not all(previous.get("checks", {}).get(name) is True for name in TRAINING_CHECKS)):
        raise ValueError("이어 하기는 학습 측 검사를 통과한 뒤 추출·평가 단계의 인프라 원인으로 invalid가 된 결과에 한 번만 허용된다")
    attempt = previous["attempt"]
    consumed = _consumed(f"resume{attempt}")
    later = [_consumed(f"attempt{n}") for n in RUN_NAMES if n > attempt]  # 다음 시도가 시작됐으면 이어 하지 않는다
    _refuse_existing(tag, _consumed("final"), consumed, *later)
    ctx = _context(tag, isaaclab, protocol, commit, attempt, resumed_from=from_tag,
                   training_commit=previous["training_source_commit_sha"])
    run_dir = Path(execution["run_dir"])
    if find_run_dir(isaaclab, ctx["run_name"]) != [run_dir] or sha(run_dir / FINAL_CHECKPOINT) != previous["checkpoint_sha256"]:
        raise ValueError("앞 실행의 실행 폴더나 체크포인트가 그대로가 아니다")
    telemetry = telemetry_path(ctx["run_name"])
    if telemetry.exists() and sha(telemetry) != previous.get("telemetry_sha256"):
        raise ValueError("남아 있는 텔레메트리가 앞 결과에 기록된 것과 다르다. 내용을 확인한 뒤 처리한다")
    gpu = _admission()
    if gpu["busy"]:
        return {"status": "gpu_busy_no_launch", "busy_reasons": gpu["busy_reasons"], "training_launched": False}
    previous_private = PRIVATE / from_tag
    ctx["run_dir"] = run_dir
    ctx["walls"]["train_s"] = previous["parent_wall_s"]["train_s"]
    # 앞 학습의 PID는 이미 다른 프로세스가 쓸 수 있어 GPU 잔존 검사(ctx["executions"])에 넣지 않는다.
    ctx["evidence"].update(
        train_exit_code=read(previous_private / "train/execution.json")["exit_code"],
        checkpoint_sha256=previous["checkpoint_sha256"],
        readback=parse_readback((previous_private / "train/stdout.log").read_text(encoding="utf-8", errors="replace")))
    write_new(consumed, {"tag": tag, "resumed_from": from_tag, "attempt": attempt})
    write_new(ctx["private"] / "gpu_before_resume.json", gpu)
    try:
        _after_training(ctx)
    except Exception as error:
        _record_failure(ctx, error)
    return _finish(ctx)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("prepare", "verify", "execute", "resume"))
    parser.add_argument("--tag")
    parser.add_argument("--retry-of", dest="retry_of")
    parser.add_argument("--from", dest="from_tag")
    parser.add_argument("--isaaclab", type=Path, default=Path.home() / "IsaacLab")
    args = parser.parse_args(argv)
    if args.command in ("execute", "resume") and not args.tag:
        parser.error(f"{args.command}에는 --tag가 필요하다")
    if args.command == "resume" and not args.from_tag:
        parser.error("resume에는 --from이 필요하다")
    import record_evidence_media as M
    try:
        if args.command == "prepare":
            out = prepare(args.isaaclab)
        elif args.command == "verify":
            out = {"status": "pass", "source_commit_sha": verify(args.isaaclab)[1]}
        elif args.command == "execute":
            out = execute(args.tag, args.isaaclab, args.retry_of)
        else:
            out = resume(args.tag, args.from_tag, args.isaaclab)
    except (ValueError, KeyError, FileExistsError, FileNotFoundError, M.PreflightError,
            subprocess.CalledProcessError) as error:
        print(f"거부: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 3 if out.get("status") == "gpu_busy_no_launch" else 0


if __name__ == "__main__":
    sys.exit(main())
