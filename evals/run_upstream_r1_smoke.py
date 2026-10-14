"""설계자 밖 외부 결함 R1 smoke: 같은 날 새 seed에서 정상(NONE)·R1 학습 각 1회 + 고정 평가로 실패 유무만 거른다.

R1은 rsl_rl #38(보고자 확인)의 의미, 즉 PPO가 설정의 num_learning_epochs를 쓰지 않아 롤아웃 배치를 한 번만 갱신에 쓰는
결함을 RSL-RL 2.3.3에 옮긴 것이다(bench/optimizer_faults.py). 정상 기준도 같은 진입점(TRIAGE_OPTIMIZER_FAULT=NONE)으로
같은 seed에서 학습한다. 판정은 P0-C 밴드로 failure_present / failure_absent / invalid 셋 중 하나다. 진단 성능·일반화를 재지 않는다.

  python evals/run_upstream_r1_smoke.py prepare                                   # 프로토콜 작성(커밋 전)
  python evals/run_upstream_r1_smoke.py verify                                    # 계약·코드·Isaac Lab·rsl_rl 소스·커밋 확인
  python evals/run_upstream_r1_smoke.py execute --tag upstream_r1_smoke_<날짜>     # GPU: 학습 2회·추출·고정 평가·판정
  python evals/run_upstream_r1_smoke.py execute --tag <새 tag> --retry-of <앞 tag> # 시도 2(인프라 원인 invalid 뒤 한 번)

학습·추출·고정 평가는 U1 smoke(evals/run_upstream_smoke.py)와 같은 소유 Windows Job·watchdog·GPU 확인 도구를 쓴다.
U1 실행기의 일반 도구 함수를 그대로 가져다 쓰고, 결함·seed·기준 실행에 묶인 부분만 이 파일에 둔다. 모델·네트워크는 쓰지 않는다.
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
import time

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "bench", ROOT / "evals"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import optimizer_faults as O  # noqa: E402
from rl_triage import grounded_recovery as C  # noqa: E402
from rl_triage import recovery as RC  # noqa: E402
import run_upstream_smoke as U  # noqa: E402

PROTOCOL = ROOT / "bench/protocols/upstream_r1_smoke_v1.json"
PRIVATE = ROOT / "bench/private/upstream_r1_smoke"
RESULTS = ROOT / "evals/results"
RUNS_DIR = ROOT / "bench/runs_upstream"
P0C_SUMMARY = ROOT / "evals/results/p0c_fixed_eval_20261009/summary.json"
FIXED_EVAL_PROTOCOL = ROOT / "bench/protocols/fixed_eval_v1.json"
TASK = "Isaac-Velocity-Flat-Unitree-Go2-v0"
FAULT, SEED, NUM_ENVS, MAX_ITERATIONS, ACTION_SCALE = "R1", 2042, 4096, 300, 0.25
CONFIGURED_EPOCHS, NUM_MINI_BATCHES = 5, 4
ARMS = {"reference": "NONE", "fault": FAULT}
RUN_NAMES = {1: {"reference": "upstream_r1_none_s2042", "fault": "upstream_r1_s2042"},
             2: {"reference": "upstream_r1_none_s2042_a2", "fault": "upstream_r1_s2042_a2"}}
FINAL_CHECKPOINT = "model_299.pt"
TAG_RE = r"upstream_r1_smoke_[0-9]{8}[a-z0-9_]{0,10}"
RSL_RL = "_isaac_sim/kit/python/Lib/site-packages"
SOURCE_FILES = {"ppo": f"{RSL_RL}/rsl_rl/algorithms/ppo.py",
                "rollout_storage": f"{RSL_RL}/rsl_rl/storage/rollout_storage.py",
                "on_policy_runner": f"{RSL_RL}/rsl_rl/runners/on_policy_runner.py",
                "rsl_rl_metadata": f"{RSL_RL}/rsl_rl_lib-2.3.3.dist-info/METADATA",
                "train": "scripts/reinforcement_learning/rsl_rl/train.py",
                "go2_agent_cfg": "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/agents/"
                                 "rsl_rl_ppo_cfg.py"}
# ppo.py에서 num_learning_epochs를 읽거나 쓰는 줄 전부. 다른 줄이 생기면 결함 정의의 효과가 달라질 수 있어 거부한다.
PPO_EPOCH_LINES = {105: "self.num_learning_epochs = num_learning_epochs",
                   205: "generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)",
                   207: "generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)",
                   405: "num_updates = self.num_learning_epochs * self.num_mini_batches"}
GO2_CFG_LINES = {29: "num_learning_epochs=5,", 30: "num_mini_batches=4,", 31: "learning_rate=1.0e-3,",
                 32: 'schedule="adaptive",', 35: "desired_kl=0.01,", 45: "self.max_iterations = 300"}
ALLOWED_PARAM_DIFFS = {"agent.yaml:run_name"}
EPOCH_PARAM = "agent.yaml:algorithm.num_learning_epochs"
# 학습 2700 s·평가 600 s: U1 smoke의 학습 watchdog을 그대로 쓰고, 평가는 체크포인트 2개라 단일 평가 360 s보다 늘린다.
WATCHDOG_S = {"training": 2700, "telemetry_extract": 300, "fixed_evaluation": 600}
TRAINING_CHECKS = ("reference_training_exit_zero", "reference_final_checkpoint_present", "reference_injection_readback",
                   "training_exit_zero", "final_checkpoint_present", "injection_readback")
CODE = ("bench/optimizer_faults.py", "bench/train_with_optimizer_fault.py", "evals/run_upstream_r1_smoke.py",
        "evals/run_upstream_smoke.py", "bench/extract_tb.py", "evals/fixed_eval.py", "bench/protocols/fixed_eval_v1.json",
        "src/rl_triage/recovery.py", "src/rl_triage/grounded_recovery.py", "src/rl_triage/probe_loop.py",
        "bench/record_evidence_media.py", "bench/run_runtime_readback_smoke.py", "bench/runtime_readback_smoke.py")
SUMMARY_SERIES = ("Loss/learning_rate", "Loss/value_function", "Loss/surrogate", "Metrics/base_velocity/error_vel_xy",
                  "Train/mean_episode_length", "Train/mean_reward")
FIXED = {
    "schema": "upstream_optimizer_fault_smoke_protocol_v1",
    "question": "다른 구현에서 원인이 밝혀진 옵티마이저 결함 R1(PPO가 num_learning_epochs를 쓰지 않음)의 의미를 RSL-RL 2.3.3에 옮기면 "
                "Go2 flat 4096 env × 300회 학습이 같은 seed 정상 학습 대비 실제로 실패하는가",
    "scope": "설계자 밖 외부 결함 한 종류·새 seed 하나의 학습 실패 유무 거르기. 진단 성능·일반화·고리 성능 평가가 아니다",
    "fault": {
        "id": FAULT, "category": "optimizer",
        "definition": "RSL-RL 2.3.3 PPO 인스턴스가 만들어진 직후 num_learning_epochs만 1로 바꾼다(bench/optimizer_faults.py). "
                      "설정(agent_cfg, params/agent.yaml)의 값 5는 그대로 둔다. 2.3.3 update는 이 값을 미니배치 생성기의 epoch 수와 손실 "
                      "평균의 분모에 함께 쓰므로, iteration마다 롤아웃 배치를 미니배치 4개로 한 번 돌아 경사 단계가 20회에서 4회로 줄고, "
                      "기록되는 손실은 실제 갱신 4회의 평균이다",
        "injection_point": "gymnasium.make 첫 호출(train.py가 Isaac Sim을 띄운 뒤, OnPolicyRunner가 PPO를 만들기 전)에 PPO 클래스의 "
                           "__init__·update를 감싼다. 생성 직후 확인 줄 하나, 첫 update 뒤 생성기 미니배치 수·optimizer.step 호출 수 확인 줄 하나",
        "unchanged": ["적응 학습률 규칙(미니배치마다 KL > 2·desired_kl이면 ÷1.5, < desired_kl/2면 ×1.5, 1e-5~1e-2): 원 브랜치와 2.3.3이 같다",
                      "미니배치 분할(2.3.3은 한 번 섞은 순서를 epoch마다 다시 쓰므로 epoch 1이면 원 브랜치처럼 각 표본을 한 번 쓴다)",
                      "clip·value loss·entropy·gradient clip·관측·보상·환경"],
        "not_ported": ["algorithms 브랜치의 다른 구현 차이(망 구성, 이점 정규화 위치, GAE λ 기본값 등). 보고자는 epoch만 되돌려 해결했다고 "
                       "적었으므로 원인으로 적힌 epoch 무시만 옮긴다",
                       "원 보고의 과제는 Isaac Lab flat ANYmal(2024-08)이고 여기는 Go2 flat이다. 300회·적응 학습률·미니배치 4·epoch 5 설정은 같다"],
        "cpu_check": "실제 rsl_rl 2.3.3 PPO(번들 파이썬, CUDA 숨김, Isaac 앱 없음)에 같은 주입을 걸고 무작위 롤아웃으로 update를 돌렸다. "
                     "NONE은 설정 5·실제 5·첫 update 미니배치 20·optimizer.step 20, R1은 설정 5·실제 1·미니배치 4·optimizer.step 4"
                     "(backward 호출도 따로 세어 20·4). 2026-10-14 sub_1 확인, 기록은 비공개",
        "no_tuning_rule": "결함 정의는 원전에서만 정한다. 이번 결과를 보고 epoch 수·주입 방식·seed를 바꾸거나 다른 seed로 재시도하지 않는다. "
                          "시도·최종 표식은 프로토콜 SHA별이므로, 최종 결과가 나온 뒤 같은 결함으로 새 프로토콜 버전을 만들지 않는다"
                          "(만들어야 하면 그 사유를 새 프로토콜에 적는다)",
    },
    "source": {
        "framework": "RSL-RL algorithms 브랜치(지금은 지워짐). RSL-RL 2.3.3(main)과 Isaac Lab의 버그가 아니다",
        "issue": {"url": "https://github.com/leggedrobotics/rsl_rl/issues/38",
                  "title": "Huge performance gaps between algorithms and master on base IsaacLab tasks",
                  "opened": "2024-08-30T11:00:00Z", "closed": "2024-08-30T11:31:27Z(보고자가 원인 댓글과 함께 닫음)",
                  "excerpts": ["does not learn to follow the velocity commands at all in the 300 update steps",
                               "num_learning_epochs is not used as a parameter in the new version (so each batch is used once only for updating)",
                               "If you modify the code and run it with 5 epochs (like the master branch PPO), it works fine"],
                  "second_user": "2024-12-09 다른 사용자가 같은 문제를 보고하고, 보고자 안내(배치 생성 호출을 epoch 바깥 반복으로 감싸기)로 "
                                 "해결했다고 적었다",
                  "maintainer_response": "없음. 원인 확정은 보고자의 재실행 확인이며 메인테이너 확정이나 수정 PR이 아니다"},
        "code": {"commit": "96393c41c55c0a905eff035875b97f3b837225fe",
                 "commit_date": "2024-04-03T20:25:58Z(이슈보다 앞선 algorithms 브랜치 끝 커밋)",
                 "legacy_runner_mapping": {"url": "https://github.com/leggedrobotics/rsl_rl/blob/96393c41c55c0a905eff035875b97f3b837225fe/"
                                                  "rsl_rl/runners/legacy_runner.py#L38",
                                           "excerpts": ['("num_learning_epochs", None),', '("num_mini_batches", "batch_count"),']},
                 "ppo_update": {"url": "https://github.com/leggedrobotics/rsl_rl/blob/96393c41c55c0a905eff035875b97f3b837225fe/"
                                       "rsl_rl/algorithms/ppo.py#L215",
                                "excerpt": "for idx, batch in enumerate(self.storage.batch_generator(self._batch_count, trajectories=self.recurrent)):",
                                "note": "batch_generator는 한 번 섞은 순서로 batch_count개 미니배치를 낸다. 손실은 batch_count개의 평균이다"},
                 "branch_removed": {"url": "https://github.com/leggedrobotics/rsl_rl/issues/149",
                                    "excerpt": "we switched to having a single branch, since the algorithms branch was very outdated"},
                 "access_note": "브랜치는 지워졌지만 커밋은 원 저장소에서 SHA로 읽히고 여러 포크(ClemensSchwarke/rsl_rl 등)의 algorithms "
                                "브랜치 끝이 같은 커밋이다(gh api, 2026-10-14)"},
    },
    "mapping": {
        "rsl_rl_version": "2.3.3",
        "ppo_epoch_lines": {str(k): v for k, v in PPO_EPOCH_LINES.items()},
        "go2_cfg_lines": {str(k): v for k, v in GO2_CFG_LINES.items()},
        "effect": {"gradient_steps_per_iteration": [CONFIGURED_EPOCHS * NUM_MINI_BATCHES, NUM_MINI_BATCHES],
                   "loss_mean_denominator": "실제 갱신 수(정상 20, R1 4). 원 브랜치도 batch_count개의 평균이었다",
                   "params_visibility": "agent.yaml에는 num_learning_epochs 5가 그대로 남아 설정 diff로 보이지 않는다"},
    },
    "task": TASK, "seed": SEED, "num_envs": NUM_ENVS, "max_iterations": MAX_ITERATIONS, "action_scale": ACTION_SCALE,
    "final_checkpoint": FINAL_CHECKPOINT,
    "training_entry": "bench/train_with_optimizer_fault.py(TRIAGE_OPTIMIZER_FAULT=NONE 또는 R1), U1 smoke와 같은 train.py 인자",
    "reference": {
        "arm": "같은 seed·예산·진입점의 정상 학습(TRIAGE_OPTIMIZER_FAULT=NONE)을 이 실행 안에서 R1 직전에 한다",
        "bands_source": "evals/results/p0c_fixed_eval_20261009/summary.json(bands)",
        "evaluation": "두 체크포인트를 한 고정 평가 프로세스(evals/fixed_eval.py, jobs 2개)에서 평가한다",
    },
    "seed_choice": {
        "decision": "새 seed 2042, 정상 학습을 같은 날 같은 진입점으로 새로 한다(학습 2회)",
        "pros": ["seed 2042는 저장소 결과·비공개 meta·Isaac Lab 로그 어디에도 쓰인 기록이 없다(실행 시 다시 검사해 결과에 남긴다)",
                 "정상·결함 학습의 날짜·런타임·진입점이 같다(U1 smoke는 다른 진입점·날짜의 P0-C 기준을 다시 썼다)",
                 "smoke가 failure_present면 이어지는 고리 단계가 같은 seed 정상 학습의 체크포인트(model_0.pt 포함)를 기준으로 쓸 수 있다"],
        "cons": ["학습이 2회라 U1 smoke(1회)보다 GPU 시간이 약 두 배다",
                 "정상 학습이 이 seed에서 걷지 못하면(reference_walks) 결과는 invalid이고 실패 유무를 말하지 않는다"],
        "alternative": "seed 2026 + P0-C baseline_p0c_s2026 재사용(학습 1회): GPU는 절반이지만 이미 본 seed이고 기준 진입점·날짜가 다르다",
    },
    "execution": {
        "owned_process": "bench/run_runtime_readback_smoke.run_owned: 소유 Windows Job, 시간 초과 시 프로세스 트리 종료, stdout·stderr 분리",
        "watchdog_s": WATCHDOG_S,
        "gpu_admission_profile": U.GPU_PROFILE,
        "gpu_admission": "학습 전 GPU가 사용 중이면 아무것도 쓰지 않고 실행하지 않는다. 고정 평가 전에는 최대 "
                         f"{U.EVAL_GPU_WAIT_S}초 동안 {U.EVAL_GPU_POLL_S}초 간격으로 다시 확인한다",
        "order": "정상 학습 → (정상 학습 측 검사 통과 시) R1 학습 → 두 실행 텔레메트리 추출 → params 대조 → 평가 전 재확인 → "
                 "고정 평가(jobs 2개) → 사후 확인 → 판정",
        "seed_unused": "시작 전에 저장소·비공개 meta·고리 기록·Isaac Lab 로그에서 seed 2042 흔적을 찾는다. 시도 1은 흔적이 없어야 하고, "
                       "시도 2는 시도 1의 두 실행 이름만 허용한다",
        "reverify": "고정 평가 직전과 평가 뒤에 프로토콜·결합 코드·Isaac Lab·rsl_rl 소스를 다시 확인하고(결합 경로 clean·커밋됨, 시작 "
                    "커밋과 현재 HEAD 사이 결합 경로 diff 없음), 평가 뒤 두 체크포인트 SHA를 다시 계산한다",
        "skip_rule": "정상 학습 측 검사가 하나라도 어긋나면 R1 학습을 하지 않는다. 학습 측 검사(두 실행의 종료 코드·체크포인트·확인 줄, "
                     "params)가 하나라도 어긋나면 고정 평가를 하지 않는다",
        "attempts": {"1": RUN_NAMES[1], "2": RUN_NAMES[2]},
        "retry_rule": "시도 2는 시도 1의 결과가 끝났고 invalid_cause가 infrastructure일 때 그 tag를 지정해 한 번만 한다. 두 학습을 "
                      "새 실행 이름으로 처음부터 다시 하며, 결함 정의·seed·판정 기준은 같다. 이어 하기(학습을 두고 평가만 다시)는 두지 않는다",
        "final_rule": "처음 나온 유효 결과(failure_present 또는 failure_absent)가 최종이다. 그 뒤로는 시도 2를 거부한다",
    },
    "outcomes": {
        "failure_present": "유효 조건을 모두 채우고, R1 고정 평가가 같은 seed 정상 고정 평가 대비 p0c 밴드에서 unhealthy",
        "failure_absent": "유효 조건을 모두 채우고 healthy",
        "invalid": "아래 유효 조건 중 하나라도 어긋남(판정이 undetermined인 경우 포함). 실패 유무를 판정하지 않는다",
    },
    "validity_checks": {
        "pipeline_completed": "실행기 단계가 예외·시간 초과·소스 변경 없이 끝남",
        "reference_training_exit_zero": "정상 학습 프로세스 종료 코드 0",
        "reference_final_checkpoint_present": "정상 실행 폴더에 model_299.pt",
        "reference_injection_readback": "정상 학습 stdout의 [optimizer_fault] 줄 두 개: NONE, 설정·실제 epoch 5, 첫 update 미니배치·optimizer.step 20",
        "training_exit_zero": "R1 학습 프로세스 종료 코드 0",
        "final_checkpoint_present": "R1 실행 폴더에 model_299.pt",
        "injection_readback": "R1 학습 stdout의 [optimizer_fault] 줄 두 개: R1, 설정 epoch 5·실제 1, 첫 update 미니배치·optimizer.step 4",
        "params_match_reference": "params/env.yaml·agent.yaml이 정상 실행과 agent run_name 외에 같다",
        "configured_epochs_in_params": "두 실행의 agent.yaml algorithm.num_learning_epochs가 5다(결함이 설정에 드러나지 않음)",
        "reference_walks": "정상 정책이 고정 평가에서 걷는다(정지 대비 오차 비 0.5 미만, 낙상 0.05 이하)",
        "fixed_report_valid": "두 고정 평가 보고서가 결합한 fixed_eval_v1 계약 SHA·체크포인트 SHA·action scale을 통과",
        "runtime_matches_reference": "두 보고서의 런타임 버전이 같다",
        "verdict_determined": "p0c 밴드 판정이 healthy 또는 unhealthy다",
    },
    "invalid_causes": {
        "rule": "validity_checks 순서에서 처음 어긋난 검사의 원인 하나를 invalid_cause로 적고, 어긋난 검사 전체는 failed_checks에 남긴다. "
                "학습 뒤 단계에서 실행기가 멈췄어도 학습 측 검사 중 어긋난 것이 있으면 그 원인을 적는다",
        "by_check": {"pipeline_completed": "infrastructure", "reference_training_exit_zero": "training_process",
                     "reference_final_checkpoint_present": "training_process", "reference_injection_readback": "injection",
                     "training_exit_zero": "training_process", "final_checkpoint_present": "training_process",
                     "injection_readback": "injection", "params_match_reference": "comparability",
                     "configured_epochs_in_params": "comparability", "reference_walks": "comparability",
                     "fixed_report_valid": "evaluation", "runtime_matches_reference": "comparability",
                     "verdict_determined": "evaluation"},
        "fault_induced_crash": "결함 때문에 R1 학습이 죽거나(종료 코드≠0) 체크포인트가 없어도 training_process로 invalid다. 실행 한 번으로 "
                               "결함 원인과 인프라 원인을 가를 수 없어 failure_present로 세지 않는다. 학습 watchdog 초과는 infrastructure다",
        "judge_failure": "판정 계산 자체가 예외로 끝나면 실패 단계 judge, invalid_cause runner로 남긴다. 실행기 결함이라 재실행 대상이 아니다",
    },
    "prediction": {
        "outcome": "failure_present",
        "reason": "원 보고가 같은 적응 학습률 규칙·같은 300회·같은 epoch 5·미니배치 4 설정의 Isaac Lab flat 과제에서 속도 명령을 전혀 "
                  "따르지 못했다고 적었다. iteration당 경사 단계가 20회에서 4회로 줄어 300회 안에 추종을 배우지 못하고, 고정 평가 선속도 "
                  "오차 비가 p0c 밴드를 넘을 것으로 본다",
        "competing": {
            "outcome": "failure_absent",
            "mechanism": "적응 학습률은 iteration 안의 KL을 desired_kl 근처로 맞추려 한다. 갱신이 4회뿐이면 KL이 작아 학습률이 더 오르고, "
                         "iteration당 정책 이동이 정상과 비슷해질 수 있다",
            "evidence": "P0-C 정상 학습 seed 2026의 Loss/learning_rate 중앙값은 5.85e-4이고 상한 1e-2까지 약 17배 여유가 있다. CPU 확인의 "
                        "무작위 롤아웃에서도 R1의 학습률이 정상보다 빨리 올랐다(Go2 학습이 아니다)",
            "note": "이 경로가 맞으면 failure_absent이며, 결함 크기(epoch 수)나 다른 seed로 다시 시도하지 않는다",
        },
        "auxiliary": {"statement": "R1 학습의 Loss/learning_rate 300회 중앙값이 같은 seed 정상 학습의 중앙값보다 크다",
                      "use": "기록만 한다. 판정·유효 조건에 쓰지 않고, 결과가 failure_absent여도 적응 학습률이 보상했다는 인과로 쓰지 않는다"},
    },
    "loop_stage": {
        "status": "smoke가 failure_present일 때만 별도 사전등록으로 연다. 이 smoke는 probe를 재지 않는다",
        "plan": "동결 evals/probes.py·src/rl_triage/probe_loop.py를 바꾸지 않고, 같은 seed 정상 실행과 R1 실행의 체크포인트에 6개 probe를 한 번씩 "
                "잰 뒤 fixed·exhaustive 순서로 고리를 돈다. R1은 학습 update만 바꾸고 롤아웃·환경은 바꾸지 않으므로 probe 프로세스에는 "
                "결함을 걸지 않는다(TRIAGE_FAULT=NONE). 모델 순위(discriminate) 고리는 새 모델 호출이 필요해 그 사전등록에서 따로 정한다",
        "probe_predictions": {
            "P_value": {"prediction": "normal",
                        "reason": "critic 파라미터 상대 변화(model_0.pt→마지막)가 같은 seed 정상의 10% 이상일 것이다. P0-C 정상은 1.43~1.47, "
                                  "학습률이 대부분 하한 1e-5에 머문 h05(노이즈 0.05)도 0.40~0.60(정상의 27~41%)이었고, R1은 갱신 1200회가 남는다",
                        "consequence": "optimizer 가설이 기각된다. 동결 probe로는 R1을 optimizer로 확정하지 못한다"},
            "P_noise": {"prediction": "normal", "reason": "행동 샘플링 경로를 바꾸지 않는다"},
            "P_reward": {"prediction": "normal", "reason": "보상 코드를 바꾸지 않는다"},
            "P_physics": {"prediction": "normal", "reason": "물리 설정을 바꾸지 않는다"},
            "P_episode": {"prediction": "normal",
                          "reason": "P0-C 실패 정책 중 종료 결함(h03) 밖에서는 시간 제한 종료 비가 1.0이었다",
                          "risk": "R1 정책이 넘어져 에피소드가 일찍 끝나면 abnormal → termination으로 틀린 확정"},
            "P_torque": {"prediction": "normal",
                         "reason": "정상 0.0003~0.0006이라 기준은 바닥값 0.01의 3배인 0.03이다. P0-C 실패 정책 중 h01·h02·h03은 0.006 이하였다",
                         "risk": "학습이 덜 된 정책은 저속 토크 포화가 커질 수 있다. P0-C h04(critic 기울기 0) 3개 중 1개(0.038), h05(노이즈 "
                                 "0.05) 3개 모두(0.06~0.81)가 0.03을 넘었다. 넘으면 actuator로 틀린 확정"},
        },
        "predicted_conclusion": {"fixed": "none_supported", "exhaustive": "none_supported",
                                 "meaning": "맞는 확정 0·틀린 확정 0. 등록 범주(optimizer) 안의 다른 구현을 동결 P_value가 겨누지 못해 모든 가설이 "
                                            "기각되는 경우"},
        "scenarios": [
            {"if": "P_torque abnormal(나머지 예측대로)", "fixed": "confirmed actuator", "exhaustive": "confirmed actuator",
             "meaning": "틀린 확정"},
            {"if": "P_episode abnormal(나머지 예측대로)", "fixed": "confirmed termination", "exhaustive": "confirmed termination",
             "meaning": "틀린 확정"},
            {"if": "P_value abnormal(나머지 예측대로)", "fixed": "confirmed optimizer", "exhaustive": "confirmed optimizer",
             "meaning": "맞는 확정(예측하지 않음)"},
        ],
        "scenario_rule": "결론은 src/rl_triage/probe_loop.simulate를 같은 관측표에 돌린 값이다(tests/test_upstream_r1_smoke.py가 대조)",
    },
    "later_uses": {
        "failure_present": ["고리 단계(loop_stage)를 별도로 사전등록하고 잰다. 이 smoke 결과로 진단 성능을 말하지 않는다",
                            "D3 해로운 변경 후보로 쓸지는 그 판의 사전등록에서 정한다"],
        "failure_absent": "이 task·예산·seed에서 실패 없음으로 기록하고 후보에서 내린다. epoch 수나 정의를 바꿔 키우지 않는다",
        "invalid": "invalid_cause와 실패 단계를 기록한다. 다시 실행하는 것은 execution.retry_rule만 따른다",
    },
    "used_for": "실패 유무 판정만. 고리·교정·D3에는 이번 판정을 쓰지 않는다",
    "gpu_estimate": {
        "unit": "부모 프로세스 벽시계(초). GPU 커널 시간이나 절감량이 아니다",
        "train_s_each": [429.3, 873.6],
        "train_basis": "U1 smoke 프로토콜의 P0-C·T1 4096 env × 300회 신규 학습 26건 범위(GPU 경합 1건 제외), 학습 2회",
        "fixed_eval_s": [102.4, 129.0],
        "fixed_eval_basis": "v4·W1 단일 고정 평가 51.2~64.5 s의 두 배로 잡은 추정(체크포인트 2개를 한 프로세스에서 평가한 기록은 없다)",
        "total_s": [1000, 1900],
    },
    "wording": {
        "failure_present": "rsl_rl #38에서 보고자가 원인으로 확인한 결함(PPO가 설정의 epoch 수를 쓰지 않음)의 의미를 옮긴 학습이 같은 seed "
                           "정상 학습 대비 p0c 밴드를 벗어났다. 새 seed 하나의 단일 실행이며, 진단 성능이나 다른 결함·task로의 일반화를 "
                           "말하지 않는다.",
        "failure_absent": "rsl_rl #38에서 보고자가 원인으로 확인한 결함의 의미를 옮긴 학습이 같은 seed 정상 학습 대비 p0c 밴드 안이었다. "
                          "이 task·예산·seed에서 학습 실패가 재현되지 않았고, epoch 수나 정의를 바꿔 다시 시도하지 않는다.",
        "invalid": "smoke가 사전등록한 유효 조건을 채우지 못해 학습 실패 유무를 판정하지 않는다.",
    },
}
DYNAMIC = {"source_files", "isaaclab_sources", "reference_inputs"}


# --- 입력 결합 ---
def epoch_lines(ppo_source: str) -> dict[int, str]:
    return {number: line.strip() for number, line in enumerate(ppo_source.splitlines(), 1) if "num_learning_epochs" in line
            and not line.strip().startswith(("#", "num_learning_epochs="))}


def cfg_lines(cfg_source: str) -> dict[int, str]:
    lines = cfg_source.splitlines()
    return {number: lines[number - 1].strip() for number in GO2_CFG_LINES if number <= len(lines)}


def isaaclab_sources(isaaclab: Path) -> dict:
    """로컬 Isaac Lab·rsl_rl의 결합 파일 SHA. epoch을 쓰는 ppo.py 줄과 Go2 설정 줄이 등록값과 다르면 거부한다."""
    files = {key: isaaclab / relative for key, relative in SOURCE_FILES.items()}
    if epoch_lines(files["ppo"].read_text(encoding="utf-8")) != PPO_EPOCH_LINES:
        raise ValueError("rsl_rl ppo.py에서 num_learning_epochs를 쓰는 줄이 등록값과 다르다")
    if cfg_lines(files["go2_agent_cfg"].read_text(encoding="utf-8")) != GO2_CFG_LINES:
        raise ValueError("Go2 PPO 설정 줄이 등록값과 다르다")
    if "Version: 2.3.3" not in files["rsl_rl_metadata"].read_text(encoding="utf-8").splitlines():
        raise ValueError("rsl-rl-lib 버전이 2.3.3이 아니다")
    return {key: {"path": SOURCE_FILES[key], "sha256_lf": U.sha(path, True)} for key, path in files.items()}


def reference_inputs() -> dict:
    summary = U.read(P0C_SUMMARY)
    return {"bands": summary["bands"], "bands_summary_sha256_lf": U.sha(P0C_SUMMARY, True),
            "fixed_eval_protocol_sha256_lf": U.sha(FIXED_EVAL_PROTOCOL, True)}


def dynamic_fields(isaaclab: Path) -> dict:
    return {"source_files": {name: U.sha(ROOT / name, True) for name in CODE},
            "isaaclab_sources": isaaclab_sources(isaaclab), "reference_inputs": reference_inputs()}


def check_protocol(protocol: dict, dynamic: dict) -> None:
    if set(protocol) != set(FIXED) | DYNAMIC:
        raise ValueError("프로토콜 필드가 다르다")
    if any(protocol[k] != v for k, v in FIXED.items()):
        raise ValueError("고정 계약이 바뀌었다")
    if any(protocol[k] != v for k, v in dynamic.items()):
        raise ValueError("결합한 코드·Isaac Lab·rsl_rl 소스·기준 입력이 바뀌었다")


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


def seed_evidence(isaaclab: Path, allowed: set[str] = frozenset()) -> dict:
    """seed 흔적. 이름에 _s2042가 든 산출물과 seed가 2042인 Isaac Lab agent.yaml. allowed 이름을 포함하는 흔적은 따로 둔다."""
    log_root = Path(isaaclab) / "logs/rsl_rl/unitree_go2_flat"
    roots = {"bench/runs_upstream": RUNS_DIR, "bench/runs_p0c": ROOT / "bench/runs_p0c", "bench/runs_d3": ROOT / "bench/runs_d3",
             "bench/private": ROOT / "bench/private", "evals/loops": ROOT / "evals/loops", "evals/results": RESULTS,
             "isaaclab_logs": log_root}
    hits = sorted(p.name for root in roots.values() if root.exists() for p in root.iterdir() if f"_s{SEED}" in p.name)
    for agent in (log_root.glob("*/params/agent.yaml") if log_root.exists() else ()):
        match = re.search(r"^seed: (\d+)\s*$", agent.read_text(encoding="utf-8", errors="replace"), re.M)
        if match and int(match[1]) == SEED:
            hits.append(f"{agent.parent.parent.name}: seed {SEED}")
    own = sorted({h for h in hits if any(name in h for name in allowed)})
    return {"seed": SEED, "checked_roots": sorted(roots), "hits": sorted(set(hits) - set(own)), "allowed_hits": own}


# --- 판정(GPU 없이 시험한다) ---
def readback_ok(lines: list | None, fault: str) -> bool:
    return isinstance(lines, list) and O.readback_ok(lines, fault, CONFIGURED_EPOCHS, NUM_MINI_BATCHES)


def configured_epochs_kept(params: dict | None) -> bool:
    return isinstance(params, dict) and all((params.get(arm) or {}).get(EPOCH_PARAM) == CONFIGURED_EPOCHS for arm in ARMS)


def training_checks(evidence: dict) -> dict[str, bool]:
    """학습 산출물만으로 정해지는 유효 조건. 하나라도 어긋나면 고정 평가를 하지 않는다."""
    runs = evidence.get("runs") or {}
    reference, fault = runs.get("reference") or {}, runs.get("fault") or {}
    differences = evidence.get("params_differences")
    return {"reference_training_exit_zero": reference.get("train_exit_code") == 0,
            "reference_final_checkpoint_present": bool(reference.get("checkpoint_sha256")),
            "reference_injection_readback": readback_ok(reference.get("readback"), "NONE"),
            "training_exit_zero": fault.get("train_exit_code") == 0,
            "final_checkpoint_present": bool(fault.get("checkpoint_sha256")),
            "injection_readback": readback_ok(fault.get("readback"), FAULT),
            "params_match_reference": differences is not None and set(differences) <= ALLOWED_PARAM_DIFFS,
            "configured_epochs_in_params": configured_epochs_kept(evidence.get("epoch_params"))}


def _valid_report(report: dict | None, run: dict, protocol: dict) -> bool:
    if report is None or not run.get("checkpoint_sha256"):
        return False
    try:
        C.validate_fixed_report(report, {"name": run["run_name"], "checkpoint_file": FINAL_CHECKPOINT,
                                         "checkpoint_sha256": run["checkpoint_sha256"], "action_scale": ACTION_SCALE,
                                         "protocol_sha256_lf": protocol["reference_inputs"]["fixed_eval_protocol_sha256_lf"]})
    except (ValueError, KeyError, TypeError):
        return False
    return True


def judge(protocol: dict, evidence: dict) -> dict:
    """사전등록 유효 조건과 p0c 밴드로 결과 범주를 정한다. evidence는 실행에서 모은 원자료다."""
    runs = evidence.get("runs") or {}
    reports = evidence.get("reports") or {}
    checks = {"pipeline_completed": evidence.get("failure") is None, **training_checks(evidence),
              "reference_walks": False, "fixed_report_valid": False, "runtime_matches_reference": False,
              "verdict_determined": None}
    checks = {name: checks[name] for name in protocol["validity_checks"]}
    reference_report, fault_report = reports.get("reference"), reports.get("fault")
    if reference_report is not None:
        try:
            checks["reference_walks"] = U.walks(reference_report)
        except (KeyError, TypeError, ZeroDivisionError):
            pass
    checks["fixed_report_valid"] = all(_valid_report(reports.get(arm), runs.get(arm) or {}, protocol) for arm in ARMS)
    if checks["fixed_report_valid"]:
        checks["runtime_matches_reference"] = fault_report.get("versions") == reference_report.get("versions")
    result = {"verdict": None, "behavior": None, "reference_behavior": None, "standstill_ratio": None}
    if all(ok for name, ok in checks.items() if name != "verdict_determined"):
        behavior, reference_behavior = C._behavior(fault_report), C._behavior(reference_report)
        verdict = RC.verdict(RC.compare(behavior, reference_behavior), protocol["reference_inputs"]["bands"])
        checks["verdict_determined"] = verdict["label"] in ("healthy", "unhealthy")
        result.update(verdict=verdict, behavior=behavior, reference_behavior=reference_behavior,
                      standstill_ratio={"reference": U.standstill_ratio(reference_report),
                                        "fault": U.standstill_ratio(fault_report)})
    failed_checks = [name for name, ok in checks.items() if ok is False]
    cause = None
    if failed_checks:
        outcome = "invalid"
        cause = protocol["invalid_causes"]["by_check"][failed_checks[0]]
        stage = (evidence.get("failure") or {}).get("stage")
        failed_training = [name for name in TRAINING_CHECKS if name in failed_checks]
        if failed_checks[0] == "pipeline_completed" and stage == "judge":
            cause = "runner"
        elif failed_checks[0] == "pipeline_completed" and stage not in ("reference_training", "fault_training") and failed_training:
            cause = protocol["invalid_causes"]["by_check"][failed_training[0]]  # 학습 뒤 단계 실패보다 학습 산출물 검사가 앞선다
    else:
        outcome = "failure_present" if result["verdict"]["label"] == "unhealthy" else "failure_absent"
    prediction = protocol["prediction"]["outcome"]
    return {"checks": checks, "failed_checks": failed_checks, **result, "outcome": outcome, "invalid_cause": cause,
            "prediction": prediction, "prediction_match": None if outcome == "invalid" else outcome == prediction,
            "wording": protocol["wording"][outcome]}


def telemetry_summary(telemetry: dict | None) -> dict | None:
    """공개 결과용 요약. 비유한 값은 JSON에 쓸 수 없어 None으로 두고 개수를 센다."""
    if telemetry is None:
        return None
    series = telemetry.get("series") or {}

    def finite(value) -> bool:
        return isinstance(value, (int, float)) and math.isfinite(value)

    out = {}
    for key in SUMMARY_SERIES:
        values = series.get(key)
        if not values:
            continue
        kept = [v for v in values if finite(v)]
        out[key] = {"first": values[0] if finite(values[0]) else None, "last": values[-1] if finite(values[-1]) else None,
                    "min": min(kept, default=None), "max": max(kept, default=None),
                    "median": statistics.median(kept) if kept else None, "nonfinite": len(values) - len(kept)}
    return out


def auxiliary(summaries: dict) -> dict:
    """사전등록 보조 예측(학습률 중앙값 방향). 판정에 쓰지 않는다."""
    medians = {arm: (((summaries.get(arm) or {}).get("Loss/learning_rate")) or {}).get("median") for arm in ARMS}
    higher = None if None in medians.values() else medians["fault"] > medians["reference"]
    return {"learning_rate_median": medians, "fault_higher": higher}


# --- 실행(GPU) ---
def train_command(isaaclab: Path, run_name: str) -> list[str]:
    return [str(isaaclab / "_isaac_sim" / "python.bat"), str(ROOT / "bench" / "train_with_optimizer_fault.py"),
            "--task", TASK, "--num_envs", str(NUM_ENVS), "--max_iterations", str(MAX_ITERATIONS),
            "--seed", str(SEED), "--headless", "--run_name", run_name]


def telemetry_path(run_name: str) -> Path:
    return RUNS_DIR / f"{run_name}.telemetry.json"


def _check_tag(tag) -> None:
    if not isinstance(tag, str) or not re.fullmatch(TAG_RE, tag):
        raise ValueError("tag는 upstream_r1_smoke_<YYYYMMDD>[접미]")


def _consumed(kind: str) -> Path:
    return PRIVATE / "consumed" / f"{U.sha(PROTOCOL, True)[:16]}_{kind}.json"


def _refuse_existing(tag: str, *paths: Path) -> None:
    if (RESULTS / tag).exists() or (PRIVATE / tag).exists() or any(path.exists() for path in paths):
        raise FileExistsError("같은 tag·시도·실행 이름의 산출물이 이미 있다. 덮어쓰지 않는다")


def bound_sources_same(start: str, now: str) -> bool:
    """두 커밋 사이에 결합 경로(CODE·프로토콜)의 diff가 없는가. git 오류는 다르다고 본다."""
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


def _context(tag: str, isaaclab: Path, protocol: dict, commit: str, attempt: int, **extra) -> dict:
    names = RUN_NAMES[attempt]
    return {"tag": tag, "isaaclab": isaaclab, "protocol": protocol, "commit": commit,
            "protocol_sha256_lf": U.sha(PROTOCOL, True), "attempt": attempt, "run_names": dict(names),
            "out": RESULTS / tag, "private": PRIVATE / tag, "run_dirs": {}, "stage": "reference_training", "failure": None,
            "executions": {}, "evidence": {"runs": {arm: {"run_name": names[arm]} for arm in ARMS}, "reports": {}},
            "walls": {"reference_train_s": None, "fault_train_s": None, "telemetry_extract_s": None, "fixed_eval_s": None},
            "commits_seen": [commit], "retry_of": None, "seed_evidence": None, **extra}


def _train(ctx: dict, arm: str) -> None:
    isaaclab, run = ctx["isaaclab"], ctx["evidence"]["runs"][arm]
    ctx["stage"] = f"{arm}_training"
    os.environ["TRIAGE_OPTIMIZER_FAULT"] = ARMS[arm]
    try:
        execution = U._owned(ctx["private"] / f"train_{arm}", train_command(isaaclab, run["run_name"]), isaaclab,
                             WATCHDOG_S["training"])
    finally:
        os.environ.pop("TRIAGE_OPTIMIZER_FAULT", None)
    ctx["executions"][f"{arm}_training"] = execution
    ctx["walls"][f"{arm}_train_s"] = round(execution["wall_s"], 1)
    run["train_exit_code"] = execution["exit_code"]
    run["readback"] = O.parse((ctx["private"] / f"train_{arm}/stdout.log").read_text(encoding="utf-8", errors="replace"))
    run_dirs = U.find_run_dir(isaaclab, run["run_name"])
    ctx["run_dirs"][arm] = run_dirs[0] if len(run_dirs) == 1 else None
    checkpoint = ctx["run_dirs"][arm] / FINAL_CHECKPOINT if ctx["run_dirs"][arm] else None
    run["checkpoint_sha256"] = U.sha(checkpoint) if checkpoint and checkpoint.is_file() else None
    if not U.finished(execution):
        raise RuntimeError(f"{arm} 학습 프로세스가 시간 안에 스스로 끝나지 않았거나 정리되지 않았다")


def _reference_ok(evidence: dict) -> bool:
    checks = training_checks(evidence)
    return all(checks[name] for name in TRAINING_CHECKS[:3])


def _after_training(ctx: dict) -> None:
    """추출·params 대조·평가 전 재확인·고정 평가·사후 확인. ctx["stage"]에 진행 단계를 남기고 예외는 그대로 올린다."""
    isaaclab, evidence, private = ctx["isaaclab"], ctx["evidence"], ctx["private"]
    if any(ctx["run_dirs"].get(arm) is None for arm in ARMS):
        return  # 체크포인트 검사에서 invalid
    ctx["stage"] = "telemetry_extract"
    started = time.monotonic()
    for arm in ARMS:
        run = evidence["runs"][arm]
        telemetry = telemetry_path(run["run_name"])
        if not telemetry.exists():
            RUNS_DIR.mkdir(parents=True, exist_ok=True)
            execution = U._owned(private / f"extract_{arm}", [str(isaaclab / "_isaac_sim" / "python.bat"),
                                                              str(ROOT / "bench" / "extract_tb.py"), str(ctx["run_dirs"][arm]),
                                                              str(telemetry)], isaaclab, WATCHDOG_S["telemetry_extract"])
            ctx["executions"][f"{arm}_telemetry_extract"] = execution
            # 추출은 성공해도 0이 아닌 종료 코드를 남길 때가 있어 결과 파일로 판정한다(run_case.ps1과 같다).
            if not U.finished(execution) or not telemetry.is_file():
                raise RuntimeError(f"{arm} 텔레메트리 추출이 끝나지 않았거나 파일이 없다")
        run["telemetry"] = U.read(telemetry)
        run["telemetry_sha256"] = U.sha(telemetry)
    ctx["walls"]["telemetry_extract_s"] = round(time.monotonic() - started, 1)
    ctx["stage"] = "params"
    params = {arm: U.load_params(ctx["run_dirs"][arm]) for arm in ARMS}
    evidence["params_differences"] = U.params_differences(params["fault"], params["reference"])
    evidence["epoch_params"] = {arm: {EPOCH_PARAM: params[arm].get(EPOCH_PARAM)} for arm in ARMS}
    if not all(training_checks(evidence).values()):
        return  # 판정에 쓰지 않을 고정 평가는 하지 않는다
    ctx["stage"] = "pre_evaluation"
    _reverify(ctx)
    U._wait_for_gpu(private)
    ctx["stage"] = "fixed_evaluation"
    paths = {arm: ctx["out"] / "runs" / f"{evidence['runs'][arm]['run_name']}.json" for arm in ARMS}
    jobs = private / "eval" / "jobs.json"
    U.write_new(jobs, {"action_scale": ACTION_SCALE, "jobs": [
        {"name": evidence["runs"][arm]["run_name"], "checkpoint": str(ctx["run_dirs"][arm] / FINAL_CHECKPOINT),
         "output": str(paths[arm])} for arm in ARMS]})
    execution = U._owned(private / "eval", [str(isaaclab / "_isaac_sim" / "python.bat"), str(ROOT / "evals" / "fixed_eval.py"),
                                            "--protocol", str(FIXED_EVAL_PROTOCOL), "--jobs", str(jobs), "--headless"],
                         isaaclab, WATCHDOG_S["fixed_evaluation"])
    ctx["executions"]["fixed_evaluation"] = execution
    ctx["walls"]["fixed_eval_s"] = round(execution["wall_s"], 1)
    if not U.finished(execution) or execution["exit_code"] != 0 or not all(path.is_file() for path in paths.values()):
        raise RuntimeError("고정 평가가 끝나지 않았거나 보고서가 없다")
    evidence["reports"] = {arm: U.read(path) for arm, path in paths.items()}
    ctx["stage"] = "postflight"
    _reverify(ctx)
    if any(U.sha(ctx["run_dirs"][arm] / FINAL_CHECKPOINT) != evidence["runs"][arm]["checkpoint_sha256"] for arm in ARMS):
        raise ValueError("평가 중 체크포인트가 바뀌었다")
    after = U._admission()
    U.write_new(private / "gpu_after.json", after)
    # PID는 재사용될 수 있어 생성 시각까지 같을 때만 소유 프로세스로 본다.
    owned = {pid: created for execution in ctx["executions"].values()
             for pid, created in execution["owned_process_identities"].items()}
    if any(str(row["pid"]) in owned and U._process_created(int(row["pid"])) == owned[str(row["pid"])]
           for row in after["processes"]):
        raise ValueError("소유 프로세스가 GPU에 남아 있다")


def _record_failure(ctx: dict, error: Exception) -> None:
    ctx["failure"] = {"stage": ctx["stage"], "error_type": type(error).__name__}
    U.write_new(ctx["private"] / "failure.json", {**ctx["failure"], "detail": str(error)})


def _finish(ctx: dict) -> dict:
    """판정하고 공개 result.json과 비공개 execution.json을 한 번씩 쓴다. 판정이 깨져도 invalid로 남긴다."""
    evidence = ctx["evidence"]
    ctx["stage"] = "judge"
    try:
        verdict = judge(ctx["protocol"], {**evidence, "failure": ctx["failure"]})
    except Exception as error:
        _record_failure(ctx, error)
        verdict = {"checks": None, "failed_checks": ["pipeline_completed"], "verdict": None, "behavior": None,
                   "reference_behavior": None, "standstill_ratio": None, "outcome": "invalid", "invalid_cause": "runner",
                   "prediction": FIXED["prediction"]["outcome"], "prediction_match": None, "wording": FIXED["wording"]["invalid"]}
    runs = evidence["runs"]
    summaries = {arm: telemetry_summary(runs[arm].get("telemetry")) for arm in ARMS}
    U.write_new(ctx["private"] / "execution.json", {
        "run_dirs": {arm: str(path) if path else None for arm, path in ctx["run_dirs"].items()}, "attempt": ctx["attempt"],
        "train_commands": {arm: train_command(ctx["isaaclab"], runs[arm]["run_name"]) for arm in ARMS},
        "failure": ctx["failure"], "retry_of": ctx["retry_of"], "commits_seen": ctx["commits_seen"]})
    result = {"schema": "upstream_optimizer_fault_smoke_result_v1", "tag": ctx["tag"], "fault": FAULT, "seed": SEED,
              "attempt": ctx["attempt"], "run_names": ctx["run_names"], "retry_of": ctx["retry_of"],
              "source_commit_sha": ctx["commit"], "commits_seen": ctx["commits_seen"],
              "protocol_sha256_lf": ctx["protocol_sha256_lf"], "seed_evidence": ctx["seed_evidence"],
              **verdict, "failure": ctx["failure"],
              "runs": {arm: {key: runs[arm].get(key) for key in ("run_name", "train_exit_code", "readback", "checkpoint_sha256",
                                                                  "telemetry_sha256")} for arm in ARMS},
              "params_differences": evidence.get("params_differences"), "epoch_params": evidence.get("epoch_params"),
              "telemetry_summary": summaries, "auxiliary": auxiliary(summaries), "parent_wall_s": ctx["walls"],
              "parent_wall_scope": FIXED["gpu_estimate"]["unit"], "scope": FIXED["scope"], "model_calls": 0}
    if result["outcome"] != "invalid":
        # execution.final_rule. 결과보다 먼저 써서, 그 사이에 멈춰도 다음 시도가 열리지 않게 한다.
        U.write_new(_consumed("final"), {"tag": ctx["tag"], "outcome": result["outcome"]})
    U.write_new(ctx["out"] / "result.json", result)
    return {"outcome": result["outcome"], "invalid_cause": result["invalid_cause"], "failure": result["failure"],
            "failed_checks": result["failed_checks"], "result": f"evals/results/{ctx['tag']}/result.json"}


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
            raise ValueError("시도 2는 시도 1의 결과가 인프라 원인 invalid로 끝났을 때 그 tag로만 허용된다")
        attempt, allowed = 2, set(RUN_NAMES[1].values())
    names = RUN_NAMES[attempt]
    consumed = _consumed(f"attempt{attempt}")
    _refuse_existing(tag, _consumed("final"), consumed, *(telemetry_path(name) for name in names.values()),
                     *(path for name in names.values() for path in U.find_run_dir(isaaclab, name)))
    evidence = seed_evidence(isaaclab, allowed)
    if evidence["hits"]:
        raise ValueError(f"seed {SEED}가 이미 쓰였다: {evidence['hits'][:5]}")
    gpu = U._admission()
    if gpu["busy"]:
        return {"status": "gpu_busy_no_launch", "busy_reasons": gpu["busy_reasons"], "training_launched": False}
    ctx = _context(tag, isaaclab, protocol, commit, attempt, retry_of=retry_of, seed_evidence=evidence)
    U.write_new(consumed, {"tag": tag, "attempt": attempt, "run_names": names})
    U.write_new(ctx["private"] / "gpu_before_training.json", gpu)
    try:
        _train(ctx, "reference")
        if _reference_ok(ctx["evidence"]):
            _train(ctx, "fault")
            _after_training(ctx)
    except Exception as error:
        _record_failure(ctx, error)
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
