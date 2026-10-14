"""두 번째 로봇 task 단계 2: ANYmal-C flat 새 seed의 보상 좌표 결함(F1) 학습 실패 하나를 종단 시연한다.

시연 v4(evals/e2e_demo_v4.py)와 같은 흐름·결과 범주·성공 조건·회복 방식 규칙·등록 표·교정 원장·모델 결정 경로(검증 피드백
1회 재요청)를 쓰고, task에 묶인 곳만 단계 0·1(evals/run_anymal_c_gate.py·run_anymal_c_band.py)의 ANYmal-C 진입으로 바꾼다.
v4 실행기·교정 실행기는 공개 기록 replay의 소스 결합 때문에 바꾸지 않고, v4의 순수 함수(호출 기록 검토·조건·공개 보기·교정
원장·교정 소비)를 그대로 불러 쓴다. v4와 다른 점:
- 학습: bench/train_with_fault.py(TRIAGE_FAULT=NONE|F1, --task ANYmal-C) → bench/extract_tb.py. 로그 폴더 anymal_c_flat,
  텔레메트리 bench/runs_anymal. 학습 프로세스의 [hidden_fault] 줄이 그 arm의 결함 하나여야 한다.
- 고정 평가: evals/fixed_eval_task.py + bench/protocols/fixed_eval_v1_anymal_c.json(평가 환경만 push·COM 무작위화 끔).
  평가 환경 readback([eval_env])이 단계 0과 같아야 한다.
- probe: 정상 기준은 evals/probes_task.py, 승인 고리 probe는 이 파일의 loop-run(evals/loop.py run과 같은 원장 순서)이
  evals/metered_probes_task.py로 잰다. loop.py approve·reject·recover는 그대로 쓴다.
- 진단 packet: run_active_diagnosis.input_packet과 같고 task 문구만 ANYmal-C다.
- 단계 1 결합: 판정 밴드와 R1 참조는 단계 1 결과(evals/results/anymal_c_band_20261014/result.json)에서 온다. 단계 1이
  pass가 아니거나(fail·invalid) 결과가 없으면 prepare·verify·begin을 거부한다. 단계 2를 시작하지 않는다.
- 교정 실행기 bench/demo_correction_anymal_c.py(v4 교정 실행기와 arm·manifest·학습 검증·판정이 같고 task·로그 폴더·
  action scale·고정 평가 구동기만 다르다).
- 변형 f1 하나, 새 seed 2047(재학습 3047). 사례 이름 anymal_c_none_s2047·anymal_c_h02_s2047. Go2 수치와 비교하지 않는다.

  python evals/e2e_demo_anymal_c.py stage1                                   # 단계 1 분기 확인(읽기만)
  python evals/e2e_demo_anymal_c.py prepare --variant f1
  python evals/e2e_demo_anymal_c.py verify --variant f1 [--tag <tag>]
  python evals/e2e_demo_anymal_c.py begin --variant f1 --tag <tag>
  python evals/e2e_demo_anymal_c.py train --tag <tag> --arm NONE|FAULT
  python evals/e2e_demo_anymal_c.py alarm|evaluate|reference|packet|close-loop|specificity|correction-evidence|summarize|status --tag <tag>
  python evals/e2e_demo_anymal_c.py start-loop --tag <tag> --record <진단 피드백 호출 기록>
  python evals/e2e_demo_anymal_c.py loop-run --tag <tag> --request <rid>        # evals/loop.py approve 뒤
  python evals/loop.py recover <사례> --tag <tag>_loop                            # loop-run이 끊겨 고아 요청이 남았을 때
  python evals/e2e_demo_anymal_c.py correction-propose --tag <tag> --record <교정 피드백 호출 기록>
  python evals/e2e_demo_anymal_c.py approve-correction|reject-correction|correct-run|recover-correction|correction-status ...
  python evals/e2e_demo_anymal_c.py record-harness-failure --tag <tag> --step <단계> --error <오류 문자열> --cause <파일:줄>
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
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

import e2e_demo_v4 as E4  # noqa: E402
import e2e_reward_demo_v2 as V2  # noqa: E402
import run_anymal_c_band as B  # noqa: E402
import run_runtime_readback_smoke as P  # noqa: E402
import run_training_correction as V3  # noqa: E402
import runtime_readback_smoke as S  # noqa: E402
from rl_triage import correction_registry as R  # noqa: E402
from rl_triage import grounded_recovery as C  # noqa: E402
from rl_triage import probe_loop as L  # noqa: E402
from rl_triage import tracking_alarm as A  # noqa: E402
from rl_triage.correction_ledger import CorrectionLedger  # noqa: E402

G, U, FE = B.G, B.U, B.FE
read = V3.read
write_new = V3.write_new
sha = V3.sha

CODE_ROOT = ROOT  # 실행 스크립트·동결 파일의 기준(입력·출력 위치인 ROOT와 따로 둬 테스트가 ROOT만 바꿀 수 있다)
PRIVATE = ROOT / "bench/private/e2e_demo_anymal_c"
RESULTS = ROOT / "evals/results"
PROTOCOL_DIR = ROOT / "bench/protocols"
EVAL_PROTOCOL = G.EVAL_PROTOCOL
V4_PROTOCOL = E4.V4_PROTOCOL
STAGE1_TAG = "anymal_c_band_20261014"
STAGE1_PROTOCOL = B.PROTOCOL
STAGE1_RESULT = f"evals/results/{STAGE1_TAG}/result.json"
RUNS_DIR = "bench/runs_anymal"
MODEL = E4.MODEL
STEPS = E4.STEPS
LOOP_TOP_K, LOOP_BUDGET, PROBE_EXECUTION_BUDGET_S = E4.LOOP_TOP_K, E4.LOOP_BUDGET, E4.PROBE_EXECUTION_BUDGET_S
CORRECTION_BUDGET_GPU_S = E4.CORRECTION_BUDGET_GPU_S
MAX_INFRA_ATTEMPTS = E4.MAX_INFRA_ATTEMPTS
RESTART_OFFSET = E4.RESTART_OFFSET
DECISION_PATH = "single_feedback"
TAG_RE = r"anymal_c_e2e_[a-z0-9_]{1,22}"
APPROVAL = V2.APPROVAL
GPUBusy = V2.GPUBusy
# run_active_diagnosis.input_packet의 task 문구에서 로봇 이름만 바꿨다(그 밖의 packet 내용은 같다).
TASK_TEXT = "ANYmal-C flat RSL-RL PPO; run may be normal, outside supported faults, or failed"
WATCHDOG_S = {"train": 3600, "telemetry_extract": 300, "fixed_eval": 600, "reference_probes": 1200, "loop_probe": 600,
              "correction_train": 3600, "correction_eval": 600}

# 변형: 결함·P0-C 사례 id·메커니즘·새 실패 seed. seed는 지금까지 쓴 2026~2037·2040·2042~2046·3029~3034·3040·3046·7·42·123과
# 개발 실행에 쓰지 않은 값이다(2026-10-14 저장소·bench/private·evals·Isaac Lab 로그 전체 확인, begin에서 다시 검사).
VARIANTS = {"f1": {"fault": "F1", "case_id": "h02", "mechanism": "reward", "seed": 2047}}
CODE = E4.CODE | set(B.CODE) | {"evals/e2e_demo_v4.py", "evals/e2e_demo_anymal_c.py", "bench/demo_correction_anymal_c.py",
                                "evals/metered_probes_task.py", "evals/metered_probes.py", "evals/run_active_diagnosis.py"}


@dataclass(frozen=True)
class Ctx:
    variant: str
    fault: str
    case_id: str
    mechanism: str
    seed: int

    @property
    def restart_seed(self) -> int:
        return self.seed + RESTART_OFFSET

    @property
    def cases(self) -> dict[str, str]:
        return {"NONE": f"anymal_c_none_s{self.seed}", "FAULT": f"anymal_c_{self.case_id}_s{self.seed}"}

    @property
    def correction_id(self) -> str:
        return R.REGISTRY[self.mechanism]["correction_id"]

    @property
    def protocol(self) -> Path:
        return PROTOCOL_DIR / f"e2e_demo_anymal_c_{self.variant}.json"


def ctx_of(variant: str) -> Ctx:
    if variant not in VARIANTS:
        raise ValueError(f"등록되지 않은 변형 {variant!r}")
    return Ctx(variant=variant, **VARIANTS[variant])


class Stage1NotPassed(ValueError):
    """단계 1이 pass가 아니어서(결과 없음·fail·invalid) 단계 2를 시작하지 않는 등록 분기."""


# --- 사전등록 ---
PREDICTIONS = {
    "f1": {
        "source_failure": ("F1 정책이 같은 seed NONE 대비 unhealthy이고 정지 대비 오차 비 ≥ 0.9(Go2 F1 실행에서 본 양상을 옮긴 예측이다. "
                           "이 task에서 F1 학습은 처음이다)"),
        "alarm": "R1이 F1에 울리고 NONE에는 울리지 않는다(단계 1에서 정상 실행끼리 울리지 않았다는 결과가 전제다)",
        "diagnosis": "첫 판단 최종 결정 action=continue, 상위 3가설에 reward 포함",
        "loop": ("P_reward abnormal 뒤 reward 확정(예산 4 이내). F1은 보상 함수 본문을 바꾸므로 task와 무관하게 문서 정의 재계산과 "
                 "기록 보상이 어긋난다고 본다"),
        "specificity": "게이트 없음(reward)",
        "correction_rule": "reward에서 정지 대비 오차 비 ≥ 0.9 → restart",
        "correction": "교정 정책 recovered",
    },
}
STAGE_GATE = {
    "index": 2,
    "requires": (f"단계 1 결과({STAGE1_RESULT})의 outcome이 pass이고 그 protocol_sha256_lf가 "
                 "bench/protocols/anymal_c_band_v1.json의 LF SHA와 같다. 정상 실행 셋(seed 2043·2044·2045)의 seed 기록이 유효하고 "
                 "텔레메트리 SHA가 기록값과 같다"),
    "on_stage1_not_pass": ("단계 1 결과가 없거나 outcome이 fail·invalid면 prepare·verify·begin을 거부한다. 단계 2를 시작하지 않고 "
                           "기준·임계값을 바꿔 다시 시도하지 않는다. 다음 결정은 사용자가 한다"),
    "bound_from_stage1": {"bands": "result.json analysis.bands(정상 실행 셋 순서쌍 6개, P0-C 방식)",
                          "alarm_reference_runs": "정상 실행 셋 세 개의 텔레메트리(R1 참조)",
                          "probe_reference": "쓰지 않는다. 비 기준 probe 기준은 v4처럼 같은 seed NONE 정책이다(단계 1이 정상 실행끼리 "
                                             "normal임을 확인했다)"},
    "fill": "단계 1 결과가 필요한 칸(bands·stage1)은 prepare가 결과 파일에서 채우고 SHA로 묶는다",
}
HARNESS_FAILURE_RULE = {
    **deepcopy(E4.HARNESS_FAILURE_RULE),
    "preconditions": [("멈춘 단계가 마지막 기록 단계 바로 다음 등록 단계이거나, 마지막 기록이 loop_start일 때 loop-run, 마지막 기록이 "
                       "specificity일 때 교정 명령(correction-evidence·correction-propose·approve-correction·reject-correction·"
                       "correct-run) 중 하나다") if p.startswith("멈춘 단계") else p for p in E4.HARNESS_FAILURE_RULE["preconditions"]],
    "fields": ["demo(anymal_c)" if f == "demo(v4)" else f for f in E4.HARNESS_FAILURE_RULE["fields"]]}


def fixed(variant: str) -> dict:
    ctx = ctx_of(variant)
    rule = E4.RECOVERY_RULES[ctx.mechanism]
    return {
        "schema": "e2e_demo_protocol_anymal_c_v1", "variant": variant, "decision_path": DECISION_PATH,
        "question": ("두 번째 로봇 task(ANYmal-C flat)의 새 seed 등록 결함 학습 실패 하나를 시연 v4와 같은 흐름(경고·고정 평가·새 판단"
                     "(검증 피드백 재요청 최대 1회)·승인 검사·교정 제안(같은 재요청)·승인 교정·고정 평가)으로 끝까지 연결해 같은 사례/요청 "
                     "ID로 기록할 수 있는가"),
        "scope": "second_task_single_new_seed_synthetic_fault_integration_demo_not_accuracy_generalization_or_savings",
        "stage": STAGE_GATE,
        "fault": ctx.fault, "fault_mechanism": ctx.mechanism, "case_id": ctx.case_id, "cases": ctx.cases,
        "fault_definition": f"bench/hidden_faults.py {ctx.fault}(동결, 크기·정의 변경 없음)",
        "seed": ctx.seed, "restart_seed": ctx.restart_seed,
        "seed_selection": ("지금까지 쓴 seed(2026~2037·2040·2042~2046·3029~3034·3040·3046·7·42·123)와 개발 실행에 쓰지 않은 값. 시작 시 "
                           "실패·재학습 seed 흔적이 저장소 산출물 폴더와 Isaac Lab 로그 전체에 없음을 다시 확인한다"),
        "task": G.TASK, "experiment": G.EXPERIMENT, "num_envs": G.NUM_ENVS, "max_iterations": G.MAX_ITERATIONS,
        "num_steps_per_env": 24, "action_scale": G.ACTION_SCALE, "telemetry_dir": RUNS_DIR,
        "training": ("단계 0·1과 같은 진입: bench/train_with_fault.py(TRIAGE_FAULT=NONE 또는 결함, --task ANYmal-C) → bench/extract_tb.py. "
                     "학습 프로세스의 [hidden_fault] 줄이 그 arm의 결함 하나이고 params(seed·env 수·반복·action scale)가 등록값이어야 한다"),
        "approval": APPROVAL, "source_binding": V2.SOURCE_BINDING,
        "recovery_rule": {"mechanism": ctx.mechanism, **deepcopy(rule)},
        "registry": E4.registry_v3(),
        "correction": {
            "correction_id": ctx.correction_id, "budget_gpu_s_min": CORRECTION_BUDGET_GPU_S,
            "modes": {
                "restart": {"arms": ["restart_corrected"], "training_seed": ctx.restart_seed,
                            "reference": "같은 seed NONE 원 정책(시연 고정 평가)",
                            "reason": "같은 seed 처음부터 학습은 정상 학습을 재현하므로 실패 seed + 1000"},
                "resume": {"arms": ["clean_resume_reference", "resume_corrected"], "training_seed": ctx.seed,
                           "resume_iteration": 299, "final_iteration": 598,
                           "reference": "같은 seed NONE 체크포인트를 같은 갱신량으로 이어 학습한 정책",
                           "reference_validity": "기준이 걷고 같은 seed NONE 원 정책 대비 healthy여야 한다. 아니면 correction_unknown"},
            },
            "verdict": "recovery.verdict(단계 1 밴드)",
            "budget_floor_checked_at": "correction-propose(원장에 쓰기 전). 하한 미만·정수 아님이면 요청을 만들지 않는다",
            "terminal_guard": "종료 기록이 있으면 교정 근거·제안·승인·거절을 거부한다",
            "recover": ("실행 중 끊겨 결과 없이 남은(고아) 요청만 recover-correction으로 재실행 없이 닫는다(v4와 같다. 결과 파일이 요청과 "
                        "같으면 그 판정, 없으면 unknown)"),
            "model_call": "evals/feedback_model_call.py --kind correction(WSL, temperature 0, 검증 피드백 재요청 최대 1회)",
            "auto_approval": False,
            "approval": f"evals/e2e_demo_anymal_c.py approve-correction/reject-correction(시연 등록 표 원장). {APPROVAL}",
            "ledger": ("CorrectionLedger(evals/loops/<사례>/correction_ledger.jsonl, registry=e2e_demo_v4.registry_v3())로 제안·승인·"
                       "소비·실행·조회"),
            "executor": ("bench/demo_correction_anymal_c.py(v4 교정 실행기와 arm·manifest·학습 검증·판정이 같고 task·로그 폴더·"
                         "action scale·고정 평가 구동기만 ANYmal-C)"),
        },
        "diagnosis": {"model": MODEL, "packet_schema": "active_decision_packet_v2",
                      "packet_builder": ("evals/run_active_diagnosis.input_packet와 같은 사실·카탈로그·임계값·예산. task 문구만 "
                                         f"{TASK_TEXT!r}"),
                      "decision_contract": "rl_triage.active_diagnosis.validate_decision",
                      "call": "evals/feedback_model_call.py --kind diagnosis(WSL, temperature 0, 검증 피드백 재요청 최대 1회)",
                      "max_infra_attempts": MAX_INFRA_ATTEMPTS, "blind_inputs": "사례 이름·정답·결함 정의를 넣지 않는다"},
        "feedback_policy": E4.FEEDBACK_POLICY,
        "invalid_after_retry": E4.INVALID_AFTER_RETRY,
        "model_feedback_record": E4.MODEL_FEEDBACK_RECORD,
        "feedback_adoption_rule": ("시연 v4와 같다. N2 검증 피드백 재요청 arm 공개 요약(evals/results/n2_feedback_v1/summary.json)이 "
                                   "improved일 때만 prepare하며 SHA로 묶는다. 그 요약은 Go2 packet의 개발 측정이고, 이 시연에서는 결정 "
                                   "경로를 v4와 같게 두는 근거로만 쓴다"),
        "specificity_gate": {"probes": E4.SPECIFICITY_GATES.get(ctx.mechanism, []), "basis": "게이트 없음(v4 등록 표와 같다)",
                             "rule": "게이트가 없는 메커니즘은 측정 없이 통과로 기록한다(v4와 같다)"},
        "loop": {"case": ctx.cases["FAULT"], "top_k": LOOP_TOP_K, "budget_probes": LOOP_BUDGET,
                 "first_probe": "모델이 고른 probe가 상위 3가설을 가르면 그 probe, 아니면 동결 규칙 probe(대체로 기록)",
                 "update": "동결 probe_loop.update 규칙. 관측 뒤 모델 재호출 없음", "metered": True,
                 "approval": f"evals/loop.py approve/reject. {APPROVAL}",
                 "execution": ("이 파일 loop-run: evals/loop.py run과 같은 원장 순서(고아 확인 → 승인 소비 → 측정 → receipt → 다음 제안). "
                               "측정만 evals/metered_probes_task.py로 이 task·결함 환경에서 한다. 산출물 경로·형식은 loop.py와 같아 "
                               "evals/loop.py recover로 복구한다"),
                 "probe_budget_gpu_s": "동결 등록 비용의 2배(loop.py와 같다). 넘으면 unknown",
                 "proceed_rule": f"상태 confirmed이고 유일 가설이 {ctx.mechanism}일 때만 교정 단계로 간다"},
        "alarm": {**A.R1, "reference_runs": "단계 1 정상 실행 셋 세 개(stage1.reference_runs)",
                  "status": "Go2 seed 2026~2028에서 사후에 고른 상수를 그대로 쓴다. 판정·게이트에 쓰지 않는다"},
        "evaluation": {"protocol": "bench/protocols/fixed_eval_v1_anymal_c.json", "driver": "evals/fixed_eval_task.py",
                       "env_readback": "평가 프로세스의 [eval_env] 줄이 단계 0과 같다(push_robot·base_com이 학습 설정에서 켜져 있다가 꺼짐)",
                       "bands_source": STAGE1_RESULT,
                       "failure_rule": "같은 seed NONE이 걷고(err_xy < 0.5×제자리, 낙상 ≤ 0.05) 결함 정책이 NONE 대비 unhealthy면 실패 존재"},
        "gpu_admission_profile": V3.FIXED["gpu_admission_profile"], "isaaclab_commit": V3.FIXED["isaaclab_commit"],
        "rsl_rl_version": V3.FIXED["rsl_rl_version"], "torch_version": V3.FIXED["torch_version"],
        "watchdogs_s": WATCHDOG_S,
        "predictions": PREDICTIONS[variant],
        "success_conditions": E4.SUCCESS_CONDITIONS,
        "success_rule": "결과가 recovered이고 성공 조건이 모두 참일 때만 demo_success",
        "outcomes": E4.OUTCOMES,
        "common_wording": (f"두 번째 로봇 task(ANYmal-C flat)의 새 seed {ctx.seed} 합성 결함({ctx.fault}) 1건을 시연 v4와 같은 흐름으로 끝까지 "
                           "연결한 통합 시연이다. 결함과 probe는 같은 설계자가 만들었고, 진단 정확도·일반화·교정 성공률·GPU 절감의 근거가 "
                           "아니며 Go2 결과와 수치를 비교하지 않는다."),
        "gap_wording": "다음 성공 조건을 충족하지 못했다: ",
        "semantic_retries": {"diagnosis": 1, "correction": 1, "rule": "검증 피드백 재요청만(인프라 재시도와 별개)"},
        "changes_from_v4": ["task·로그 폴더·학습 진입(train_with_fault.py)·텔레메트리 폴더를 단계 0·1의 ANYmal-C 진입으로 바꿈",
                            "고정 평가를 fixed_eval_task.py + fixed_eval_v1_anymal_c.json으로(평가 환경 readback 확인 추가)",
                            "probe를 probes_task.py·metered_probes_task.py로, 고리 측정은 이 파일 loop-run",
                            "진단 packet task 문구만 ANYmal-C",
                            "판정 밴드·R1 참조를 단계 1 결과에서 가져와 SHA로 묶고, 단계 1이 pass가 아니면 시작하지 않음",
                            "변형 f1 하나와 새 seed 2047(재학습 3047), 교정 실행기 bench/demo_correction_anymal_c.py"],
        "harness_failure_rule": HARNESS_FAILURE_RULE,
        "forbidden_claims": ["진단 정확도·우월성", "일반화", "교정 성공률", "GPU 절감", "이미 본 seed 결과와 합친 분모",
                             "Go2 시연 수치와의 비교", "승인자 문자열을 신원 인증으로 표기", "운영자 승인을 사람 승인으로 표기",
                             "재요청이 진단 내용(순위·probe)을 좋게 했다"],
    }


DYNAMIC = {"source_files", "isaaclab_sources", "eval_protocol_sha256_lf", "stage1", "bands", "v4_protocol_sha256_lf",
           "correction_source_contract", "feedback_adoption"}


def _rel(path: Path) -> str:
    return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()


def stage1_inputs() -> dict:
    """단계 1 결과(pass)와 정상 실행 셋의 텔레메트리. pass가 아니면 Stage1NotPassed(단계 2를 시작하지 않는다)."""
    path = ROOT / STAGE1_RESULT
    if not path.is_file():
        raise Stage1NotPassed("단계 1 결과 파일이 없다. 단계 1이 pass로 끝나 커밋된 뒤에만 단계 2를 시작한다")
    result = read(path)
    if (result.get("schema") != "second_task_band_result_v1" or result.get("tag") != STAGE1_TAG
            or result.get("task") != G.TASK):
        raise ValueError("단계 1 결과의 형식·tag·task가 등록값과 다르다")
    if result.get("protocol_sha256_lf") != sha(STAGE1_PROTOCOL, True):
        raise ValueError("단계 1 결과가 단계 1 프로토콜(bench/protocols/anymal_c_band_v1.json)의 산출물이 아니다")
    if result.get("outcome") != "pass":
        raise Stage1NotPassed(f"단계 1 결과가 {result.get('outcome')!r}(failed_criteria={result.get('failed_criteria')})라 "
                              "단계 2를 시작하지 않는다(사전등록 분기)")
    bands = (result.get("analysis") or {}).get("bands")
    if not isinstance(bands, dict) or not bands:
        raise ValueError("단계 1 결과에 밴드가 없다")
    stage0 = read(ROOT / B.STAGE0_FILES["result"])
    if stage0.get("outcome") != "pass" or sha(ROOT / B.STAGE0_FILES["telemetry"]) != stage0.get("telemetry_sha256"):
        raise ValueError("단계 0 정상 실행의 결과·텔레메트리가 결합값과 다르다")
    runs, records = [B.STAGE0_RUN], {}
    for seed in B.SEEDS:
        summary = (result.get("seed_records") or {}).get(str(seed)) or {}
        record_path = f"evals/results/{STAGE1_TAG}/seeds/s{seed}_a{summary.get('attempt')}.json"
        record = read(ROOT / record_path)
        if (record.get("valid") is not True or record.get("run_name") != summary.get("run_name")
                or record.get("telemetry") != f"{RUNS_DIR}/{record.get('run_name')}.telemetry.json"
                or sha(ROOT / record["telemetry"]) != record.get("telemetry_sha256")):
            raise ValueError(f"단계 1 seed {seed}의 기록·텔레메트리가 결과와 맞지 않는다")
        runs.append(record["run_name"])
        records[str(seed)] = {"path": record_path, "sha256_lf": sha(ROOT / record_path, True)}
    telemetry = {name: {"path": f"{RUNS_DIR}/{name}.telemetry.json", "sha256_lf": sha(ROOT / RUNS_DIR / f"{name}.telemetry.json", True)}
                 for name in runs}
    return {"tag": STAGE1_TAG, "outcome": "pass", "result": {"path": STAGE1_RESULT, "sha256_lf": sha(path, True)},
            "protocol_sha256_lf": result["protocol_sha256_lf"], "seed_records": records,
            "stage0_result": {"path": B.STAGE0_FILES["result"], "sha256_lf": sha(ROOT / B.STAGE0_FILES["result"], True)},
            "reference_runs": runs, "reference_telemetry": telemetry}


def stage1_status() -> dict:
    """단계 2 시작 분기(읽기만). pass면 start_stage2=True."""
    try:
        bound = stage1_inputs()
    except Stage1NotPassed as error:
        return {"start_stage2": False, "reason": str(error)}
    return {"start_stage2": True, "result": bound["result"], "reference_runs": bound["reference_runs"]}


def dynamic_fields(isaaclab: Path) -> dict:
    stage1 = stage1_inputs()
    contract = read(V4_PROTOCOL)["correction_source_contract"]
    V3.check_contract(contract, isaaclab)
    return {
        "source_files": {name: sha(ROOT / name, True) for name in sorted(CODE)},
        "isaaclab_sources": G.isaaclab_sources(isaaclab), "eval_protocol_sha256_lf": sha(EVAL_PROTOCOL, True),
        "stage1": stage1, "bands": read(ROOT / STAGE1_RESULT)["analysis"]["bands"],
        "v4_protocol_sha256_lf": sha(V4_PROTOCOL, True), "correction_source_contract": contract,
        "feedback_adoption": E4.feedback_adoption(),
    }


def check_protocol(protocol: dict, dynamic: dict) -> None:
    expected = fixed(protocol.get("variant"))
    if set(protocol) != set(expected) | DYNAMIC:
        raise ValueError("protocol fields differ")
    if any(protocol[k] != v for k, v in expected.items()):
        raise ValueError("fixed demo contract changed")
    changed = sorted(k for k, v in dynamic.items() if protocol[k] != v)
    if changed:
        raise ValueError(f"bound sources changed: {changed}")


def prepare(variant: str, isaaclab: Path) -> dict:
    ctx = ctx_of(variant)
    write_new(ctx.protocol, {**deepcopy(fixed(variant)), **dynamic_fields(isaaclab)})
    return {"status": "prepared_commit_before_begin", "path": _rel(ctx.protocol)}


def bound_files(protocol: dict) -> set[str]:
    stage1 = protocol["stage1"]
    return (CODE | {_rel(ctx_of(protocol["variant"]).protocol), _rel(EVAL_PROTOCOL), _rel(V4_PROTOCOL), _rel(STAGE1_PROTOCOL),
                    stage1["result"]["path"], stage1["stage0_result"]["path"], protocol["feedback_adoption"]["summary"],
                    *(row["path"] for row in stage1["seed_records"].values()),
                    *(row["path"] for row in stage1["reference_telemetry"].values())})


def verify_protocol(variant: str, isaaclab: Path, experiment: str | None = None) -> tuple[dict, str]:
    """프로토콜·결합 파일·단계 1 결과·Isaac Lab을 검사한다. 실험 커밋이 없으면(begin) HEAD를, 있으면 실험 커밋을 돌려준다."""
    ctx = ctx_of(variant)
    protocol = read(ctx.protocol)
    if protocol.get("variant") != variant:
        raise ValueError("프로토콜 변형이 다르다")
    check_protocol(protocol, dynamic_fields(isaaclab))
    bound = bound_files(protocol)
    try:
        if S._git_head(isaaclab) != protocol["isaaclab_commit"] or not S._git_tracked_clean(isaaclab):
            raise ValueError("Isaac Lab checkout differs")
        head = P.verify_committed(ROOT, bound)
    except subprocess.CalledProcessError as error:
        raise ValueError(f"git check failed: {' '.join(map(str, error.cmd))[:200]}") from error
    return protocol, head if experiment is None else V2.frozen_commit(experiment, head, bound)


# --- 단계 원장(v4와 같다) ---
def folder_for(tag: str) -> Path:
    if not isinstance(tag, str) or not re.fullmatch(TAG_RE, tag):
        raise ValueError("demo tag는 anymal_c_e2e_<영문 소문자·숫자·_>")
    folder = PRIVATE / tag
    if folder.is_symlink() or folder.resolve().parent != PRIVATE.resolve():
        raise ValueError("demo output escapes its root")
    return folder


def step_path(tag: str, name: str) -> Path:
    return folder_for(tag) / "steps" / f"{STEPS.index(name):02d}_{name}.json"


def step_record(tag: str, name: str) -> dict:
    path = step_path(tag, name)
    if not path.is_file():
        raise FileNotFoundError(f"{name} 단계 기록이 없다")
    return read(path)


def ctx_for(tag: str) -> Ctx:
    return ctx_of(step_record(tag, "begin")["variant"])


def experiment_commit(tag: str) -> str:
    commit = step_record(tag, "begin").get("source_commit_sha")
    if not isinstance(commit, str):
        raise ValueError("begin 기록에 실험 커밋이 없다")
    return commit


def verify_for(tag: str, isaaclab: Path) -> tuple[dict, str]:
    return verify_protocol(ctx_for(tag).variant, isaaclab, experiment_commit(tag))


def _guard(tag: str, name: str) -> Path | None:
    """앞 단계가 모두 있고 SHA 사슬이 이어지며 이 단계와 종료 기록이 없을 때만 진행한다."""
    folder = folder_for(tag)
    if (folder / "terminal.json").exists():
        raise ValueError(f"시연이 이미 종료됐다: {read(folder / 'terminal.json')['outcome']}")
    if step_path(tag, name).exists():
        raise FileExistsError(f"{name} 단계는 이미 기록됐다(한 번만 쓴다)")
    previous = None
    for earlier in STEPS[:STEPS.index(name)]:
        path = step_path(tag, earlier)
        if not path.is_file():
            raise ValueError(f"{earlier} 단계가 먼저다")
        if read(path).get("previous_sha256") != (None if previous is None else sha(previous)):
            raise ValueError(f"{earlier} 단계의 SHA 사슬이 끊겼다")
        previous = path
    return previous


def _write_step(tag: str, name: str, previous: Path | None, commit: str, protocol_path: Path, record: dict) -> dict:
    full = {"step": name, "at": V2.now(), "previous_sha256": None if previous is None else sha(previous),
            "source_commit_sha": commit, "protocol_sha256_lf": sha(protocol_path, True), **record}
    write_new(step_path(tag, name), full)
    return full


def _terminal(tag: str, outcome: str, detail: dict) -> dict:
    if outcome not in E4.TERMINAL_OUTCOMES:
        raise ValueError(f"등록되지 않은 조기 종료 {outcome}")
    record = {"outcome": outcome, "at": V2.now(), "detail": detail, "wording": E4.OUTCOMES[outcome]}
    write_new(folder_for(tag) / "terminal.json", record)
    return record


def _require_open(tag: str) -> None:
    terminal = folder_for(tag) / "terminal.json"
    if terminal.exists():
        raise ValueError(f"시연이 이미 종료됐다: {read(terminal)['outcome']}")


def _attempt_folder(tag: str, step: str) -> tuple[Path, int]:
    exec_root = folder_for(tag) / "exec"
    attempt = 1 + sum(1 for p in exec_root.glob(f"{step}_attempt*") if p.is_dir()) if exec_root.exists() else 1
    folder = exec_root / f"{step}_attempt{attempt}"
    folder.mkdir(parents=True)
    return folder, attempt


def _execute(folder: Path, argv: list[str], cwd: Path, watchdog_s: float, owned=None) -> dict:
    """Windows Job으로 프로세스 트리를 소유해 실행한다(stdout.log·stderr.log·execution.json은 folder에)."""
    owned = owned or P.run_owned
    if any(char in arg for arg in argv for char in '\r\n"%&^'):
        raise ValueError("unsafe Windows batch argument")
    folder.mkdir(parents=True, exist_ok=True)
    execution = owned(folder, argv, cwd, watchdog_s)
    execution = {**execution, "parent_wall_s": round(execution["wall_s"], 1), "folder": _rel(folder)}
    write_new(folder / "execution.json", execution)
    return execution


def _ok(execution: dict) -> bool:
    return U.finished(execution) and execution.get("exit_code") == 0


def _gpu_step_failed(tag: str, step: str, execution: dict, attempt: int, output_exists: bool) -> None:
    detail = {"step": step, "attempt": attempt, "execution": {k: execution.get(k) for k in
                                                            ("exit_code", "natural_exit", "timed_out", "error_type",
                                                             "cleanup_verified", "folder")}}
    if step in ("evaluate", "reference") and attempt == 1 and not output_exists and execution.get("cleanup_verified"):
        raise RuntimeError(f"{step} 인프라 실패, 결과 파일이 없어 1회 재시도할 수 있다: {detail}")
    _terminal(tag, "invalid", detail)
    raise RuntimeError(f"{step} 실패로 시연을 invalid로 종료했다: {detail}")


def _log_text(folder: Path) -> str:
    log = folder / "stdout.log"
    return log.read_text(encoding="utf-8", errors="replace") if log.is_file() else ""


# --- 입력 확인 ---
def seed_unused_evidence(ctx: Ctx, isaaclab: Path, tag: str) -> dict:
    """단계 0·1과 같은 범위(저장소 산출물 폴더와 Isaac Lab 로그 전체)에서 실패·재학습 seed 흔적과 이 tag의 산출물을 찾는다."""
    rows = [B.seed_evidence(isaaclab, seed, set()) for seed in (ctx.seed, ctx.restart_seed)]
    outputs = [_rel(p) for p in (folder_for(tag), RESULTS / tag, RESULTS / f"{tag}_ref", RESULTS / f"{tag}_loop",
                                 *(ROOT / "evals/loops" / case for case in ctx.cases.values())) if p.exists()]
    return {"checked_roots": rows[0]["checked_roots"], "checked_seeds": [ctx.seed, ctx.restart_seed],
            "hits": sorted(h for row in rows for h in row["hits"]), "existing_outputs": outputs}


def readback_ok(lines: list | None, fault: str) -> bool:
    return isinstance(lines, list) and len(lines) == 1 and lines[0] == {"fault": fault}


def training_artifacts(ctx: Ctx, arm: str, isaaclab: Path) -> dict:
    name = ctx.cases[arm]
    run_dirs = G.find_run_dir(isaaclab, name)
    if len(run_dirs) != 1:
        raise ValueError(f"{name}: run must match exactly one SDK log directory")
    run_dir = run_dirs[0].resolve()
    if not B.params_expected(U.load_params(run_dir), ctx.seed):
        raise ValueError(f"{name}: training params differ from the registered seed, env count, iterations or action scale")
    files = {"checkpoint": run_dir / G.FINAL_CHECKPOINT, "first_checkpoint": run_dir / G.FIRST_CHECKPOINT,
             "environment": run_dir / "params/env.yaml", "agent": run_dir / "params/agent.yaml"}
    telemetry = ROOT / RUNS_DIR / f"{name}.telemetry.json"
    series = read(telemetry)["series"].get(A.METRIC)
    if not isinstance(series, list) or len(series) != G.MAX_ITERATIONS:
        raise ValueError(f"{name}: telemetry does not cover 300 iterations")
    return {"run": name, "run_dir": str(run_dir),
            **{k: {"path": str(p.resolve()), "sha256": sha(p)} for k, p in files.items()},
            "telemetry": {"path": _rel(telemetry), "sha256_lf": sha(telemetry, True)}}


def _check_artifacts(tag: str, arm: str) -> dict:
    artifacts = step_record(tag, f"train_{arm}")["artifacts"]
    for key in ("checkpoint", "first_checkpoint", "environment", "agent"):
        if sha(Path(artifacts[key]["path"])) != artifacts[key]["sha256"]:
            raise ValueError(f"{arm} {key} changed after training")
    if sha(ROOT / artifacts["telemetry"]["path"], True) != artifacts["telemetry"]["sha256_lf"]:
        raise ValueError(f"{arm} telemetry changed after training")
    return artifacts


def validate_report(report: dict, expected: dict) -> dict:
    """이름·task는 ANYmal-C 평가 프로토콜 값과 대조하고, 나머지는 공용 검증기로 본다(공용 검증기는 두 칸을 Go2 값으로 고정했다)."""
    if not isinstance(report, dict) or report.get("protocol") != read(EVAL_PROTOCOL)["name"] or report.get("task") != G.TASK:
        raise ValueError("fixed report is not an ANYmal-C fixed_eval_v1 report")
    C.validate_fixed_report({**report, "protocol": C.PROTOCOL, "task": C.TASK}, expected)
    return report


# --- 첫 판단 packet ---
def input_packet(sources: dict) -> dict:
    """run_active_diagnosis.input_packet과 같은 packet에서 task 문구만 ANYmal-C로 바꾼다."""
    import run_active_diagnosis as RA
    built = RA.input_packet({"sources": sources}, {"max_probes": LOOP_BUDGET,
                                                   "probe_execution_budget_s": PROBE_EXECUTION_BUDGET_S})
    if "Unitree Go2" not in built["task"]:
        raise ValueError("공용 packet의 task 문구가 예상과 다르다")
    return {**built, "task": TASK_TEXT}


def call_summary(record_path: Path, record: dict) -> dict:
    """v4 call_summary와 같다(경로만 이 저장소 기준)."""
    calls = record.get("calls") or []
    keys = ("step", "status", "codes", "response_id", "provider_model", "finish_reason", "input_tokens", "output_tokens",
            "wall_s")
    return {"path": DECISION_PATH, "record": _rel(record_path), "record_sha256": sha(record_path),
            "requests": record.get("requests"), "input_tokens": E4._total(c.get("input_tokens") for c in calls),
            "output_tokens": E4._total(c.get("output_tokens") for c in calls),
            "total_tokens": E4._total(c.get("total_tokens") for c in calls),
            "execution_wall_s": record.get("execution_wall_s"), "calls": [{k: c.get(k) for k in keys} for c in calls]}


# --- 승인 고리 probe 실행(evals/loop.py run의 측정 함수를 이 task로) ---
def probe_execute(tag: str, ctx: Ctx, isaaclab: Path, protocol: dict, case: str, probe: str, owned=None):
    """승인·소비된 probe 하나를 결함 정책에서 잰다. (측정값, GPU 초, 종료 코드). 산출물은 loop.isaac_execute와 같은 경로·형식."""
    import loop
    loop_tag = f"{tag}_loop"
    out = loop._probe_path(case, probe, loop_tag)
    resources_path = out.parent.parent / "resources" / out.name
    checkpoint = Path(_check_artifacts(tag, "FAULT")["checkpoint"]["path"])
    work = folder_for(tag) / "loop" / probe
    jobs, metrics = work / "jobs.json", work / "metrics.json"
    write_new(jobs, {"task": G.TASK, "fault": ctx.fault, "seed": ctx.seed, "jobs": [
        {"name": case, "checkpoint": str(checkpoint), "probes": [probe], "output": str(out)}]})
    started_at = datetime.now(timezone.utc).isoformat()
    cpu0 = time.process_time()
    execution = _execute(work / "exec", [G.python_bat(isaaclab), str(CODE_ROOT / "evals" / "metered_probes_task.py"), "--jobs",
                                         str(jobs), "--headless", "--metrics", str(metrics)], isaaclab,
                         protocol["watchdogs_s"]["loop_probe"], owned)
    wall, parent_cpu = execution["wall_s"], time.process_time() - cpu0
    report = read(out) if out.is_file() else {}
    record = (report.get("probes") or {}).get(probe, {})
    code = (execution.get("exit_code") if U.finished(execution) else None)
    code = record.get("exit") if code == 0 else code
    report_sha = sha(out, True) if out.is_file() else None
    resource = None
    if out.is_file():
        if (report.get("name") != case or report.get("task") != G.TASK or report.get("seed") != ctx.seed
                or (report.get("checkpoint") or {}).get("sha256") != sha(checkpoint)):
            raise ValueError("probe 보고서가 사례·task·seed·결함 체크포인트와 다르다")
        meter_path = metrics.with_suffix(".post_close.json") if metrics.with_suffix(".post_close.json").is_file() else metrics
        if not meter_path.is_file():
            raise ValueError("계측 sidecar가 없다")
        meter = read(meter_path)
        rows = meter.get("measurements") or []
        if (meter.get("jobs_sha256") != sha(jobs) or meter.get("probes_sha256_lf") != sha(CODE_ROOT / "evals/probes.py", True)
                or not meter.get("completed") or len(rows) != 1 or rows[0].get("probe") != probe
                or rows[0].get("checkpoint_sha256") != report["checkpoint"]["sha256"]):
            raise ValueError("계측 sidecar가 실행 입력·산출물과 다르다")
        resource = {"schema": "probe_resources_v1", "case": case, "probe": probe,
                    "checkpoint_sha256": rows[0]["checkpoint_sha256"], "report_sha256_lf": report_sha,
                    "cpu_s": meter["cpu_s"], "launcher_exit_code": execution.get("exit_code"),
                    "cpu_scope": "isaac_python_probe_driver_process", "isaac_cpu_scope": meter["cpu_scope"],
                    "app_close_cpu_included": meter["phase"] == "after_main",
                    "simulator_steps_scope": meter["simulator_steps_scope"], "meter_sha256_lf": sha(meter_path, True),
                    "probes_sha256_lf": meter["probes_sha256_lf"], **rows[0]}
        write_new(resources_path, resource)
    meta = {"schema": "probe_execution_v1", "case": case, "probe": probe,
            "checkpoint_sha256": (report.get("checkpoint") or {}).get("sha256"), "report_sha256_lf": report_sha,
            "started_at": started_at, "finished_at": datetime.now(timezone.utc).isoformat(),
            "gpu_wall_s": round(wall, 1), "execution_wall_s": round(wall, 1), "exit_code": code,
            "parent_cpu_s": parent_cpu, "cpu_s": None if resource is None else parent_cpu + resource["cpu_s"],
            "simulator_steps": None if resource is None else resource["simulator_steps"], "metered": True}
    write_new(loop._execution_path(case, probe, loop_tag), meta)
    return (record.get("measurement") if code == 0 else None), wall, code


def loop_run(tag: str, isaaclab: Path, request_id: str, owned=None) -> dict:
    """evals/loop.py run과 같은 순서(고아 확인 → 승인 소비 → 측정 → receipt → 다음 제안)로 승인된 probe 하나를 잰다."""
    import loop
    ctx = ctx_for(tag)
    _require_open(tag)
    step_record(tag, "loop_start")
    if step_path(tag, "loop_close").exists():
        raise ValueError("고리가 이미 닫혔다")
    protocol, _ = verify_for(tag, isaaclab)
    V2._admit_gpu(protocol)
    case = ctx.cases["FAULT"]
    message = loop.run(case, request_id, lambda c, p: probe_execute(tag, ctx, isaaclab, protocol, c, p, owned),
                       preflight=lambda c, p: loop.isaac_preflight(c, p, f"{tag}_loop"))
    proposed = re.match(r"제안 (\S+):", message.splitlines()[-1])
    following = ([f"python evals/loop.py approve {case} {proposed[1]} --approver <승인자>   # 또는 reject ... --reason <사유>",
                  f"python evals/e2e_demo_anymal_c.py loop-run --tag {tag} --request {proposed[1]}   # GPU, 실행 중 커밋 금지"]
                 if proposed else [f"python evals/e2e_demo_anymal_c.py close-loop --tag {tag}"])
    return {"message": message, "next_commands": following}


def loop_summary(ctx: Ctx, tag: str, checkpoint_sha: str) -> dict:
    """끝난 고리의 원장·상태를 읽고 실행된 probe마다 산출물의 사례·체크포인트 SHA를 대조한다(v4와 같다)."""
    import loop
    case = ctx.cases["FAULT"]
    prereg, state, led = loop._load(case)
    if state["status"] == "open" or state.get("pending") or led.orphans():
        raise ValueError("고리가 아직 끝나지 않았다(대기·실행 중 요청이 있다)")
    if prereg["checkpoint_sha256"] != checkpoint_sha:
        raise ValueError("고리 사전등록의 체크포인트가 시연 결함 실행과 다르다")
    events = led.events()
    approvals = {e["request_id"]: e["approver"] for e in events if e["event"] == "approved"}
    rows, costs = [], []
    for rid, req in led.requests().items():
        row = {"request_id": rid, "probe": req["probe"], "state": req["state"], "approver": approvals.get(rid)}
        if req["state"] == "rejected":
            row["rejected_reason"] = next(e["reason"] for e in events if e["event"] == "rejected" and e["request_id"] == rid)
        if req["state"] == "done":
            receipt = req["receipt"]
            row.update(outcome=receipt["outcome"], measurement=receipt["measurement"], gpu_s=receipt["gpu_s"])
            probe_file = RESULTS / f"{tag}_loop" / "probes" / f"{case}__{req['probe']}.json"
            meta_file = RESULTS / f"{tag}_loop" / "execution" / f"{case}__{req['probe']}.json"
            if receipt["outcome"] != "unknown":
                report, meta = read(probe_file), read(meta_file)
                if (report.get("name") != case or report.get("checkpoint", {}).get("sha256") != checkpoint_sha
                        or meta.get("checkpoint_sha256") != checkpoint_sha
                        or meta.get("report_sha256_lf") != sha(probe_file, True)):
                    raise ValueError(f"{req['probe']}: probe 산출물이 사례·체크포인트와 다르다")
                if meta.get("metered") is not True:
                    raise ValueError(f"{req['probe']}: 등록과 달리 계측 없이 실행됐다")
                row["artifact"] = {"probe": _rel(probe_file), "execution": _rel(meta_file)}
                costs.append({k: meta.get(k) for k in ("gpu_wall_s", "cpu_s", "simulator_steps")} | {"probe": req["probe"]})
        rows.append(row)
    return {"status": state["status"], "hypotheses": state["hypotheses"], "observed": state["observed"],
            "history": state["history"], "requests": rows, "probe_costs": costs}


# --- 교정(v4 원장·소비 함수를 그대로 쓴다) ---
def correction_ledger(ctx: Ctx) -> CorrectionLedger:
    return E4.correction_ledger(ctx)


def approve_correction(tag: str, request_id: str, approver: str) -> dict:
    _require_open(tag)
    return correction_ledger(ctx_for(tag)).approve(request_id, approver)


def reject_correction(tag: str, request_id: str, approver: str, reason: str) -> dict:
    _require_open(tag)
    return correction_ledger(ctx_for(tag)).reject(request_id, approver, reason)


def correction_status(tag: str) -> dict:
    led = correction_ledger(ctx_for(tag))
    return {"registry_digest": led.registry_digest, "requests": led.requests(),
            "holds": [e for e in led.events() if e["event"] == "held"], "orphans": led.orphans()}


def correction_packet(ctx: Ctx, tag: str) -> dict:
    """v4 correction_packet과 같다(시연 단계 기록만 이 실행기의 것)."""
    import correct as CR
    import correction_model_call as CM
    case = ctx.cases["FAULT"]
    prereg, state, loop_led = CR._loop_state(case, CR.LOOPS)
    mech, req = CR._confirming_request(case, state, loop_led)
    report_info = step_record(tag, "evaluate")["reports"]["FAULT"]
    report = read(ROOT / report_info["path"])
    receipts = [e for e in loop_led.events() if e["event"] == "receipt"]
    confirming = {"probe": req["probe"], "request_id": req["request_id"], "outcome": req["receipt"]["outcome"],
                  "measurement": req["receipt"].get("measurement")}
    return CM.build_packet(case=case, loop_state_digest=CR.loop_state_digest(prereg, state),
                           receipts_digest=L.digest(receipts), mechanism=mech, confirming=confirming,
                           observed=state.get("observed", {}), rule=E4.recommend(mech, report), source_report=report,
                           source_report_sha256=report_info["sha256_lf"], checkpoint_sha256=prereg.get("checkpoint_sha256"),
                           registry=E4.registry_v3())


def _require_specificity(tag: str) -> None:
    if not step_record(tag, "loop_close")["proceed_to_correction"] or not step_record(tag, "specificity")["passed"]:
        raise ValueError("고리 확정과 특이성 게이트를 통과한 시연만 교정 단계로 간다")


def correction_evidence(tag: str, isaaclab: Path) -> dict:
    ctx = ctx_for(tag)
    _require_open(tag)
    verify_for(tag, isaaclab)
    _require_specificity(tag)
    path = folder_for(tag) / "model" / "correction.packet.json"
    write_new(path, correction_packet(ctx, tag))
    return {"packet": _rel(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def correction_propose(tag: str, isaaclab: Path, record_path: Path, budget_gpu_s: int = CORRECTION_BUDGET_GPU_S) -> dict:
    """확정된 고리에서 v4 규칙의 교정 요청 하나를 원장에 올린다(v4 correction_propose와 같다)."""
    import correct as CR
    ctx = ctx_for(tag)
    _require_open(tag)
    if type(budget_gpu_s) is not int or budget_gpu_s < CORRECTION_BUDGET_GPU_S:
        raise ValueError(f"교정 예산은 등록 하한 {CORRECTION_BUDGET_GPU_S} s 이상의 정수여야 한다(원장에 쓰지 않았다)")
    protocol, _ = verify_for(tag, isaaclab)
    _require_specificity(tag)
    packet_path = folder_for(tag) / "model" / "correction.packet.json"
    packet = read(packet_path)
    if packet != correction_packet(ctx, tag):
        raise ValueError("교정 packet이 현재 고리·평가와 다르다")
    review = E4.review_correction_record(protocol, Path(record_path), packet, packet_path)
    case = ctx.cases["FAULT"]
    prereg, state, loop_led = CR._loop_state(case, CR.LOOPS)
    mech, req = CR._confirming_request(case, state, loop_led)
    rule = E4.recommend(mech, read(ROOT / step_record(tag, "evaluate")["reports"]["FAULT"]["path"]))
    receipts = [e for e in loop_led.events() if e["event"] == "receipt"]
    evidence = {"confirming_probe": req["probe"], "confirming_request_id": req["request_id"],
                "standstill_ratio": rule["standstill_ratio"], "rule": rule["rule"], "rule_mode": rule["mode"],
                "source_report_sha256": step_record(tag, "evaluate")["reports"]["FAULT"]["sha256_lf"],
                "model_proposal": review, "registry_v3": E4.registry_v3()[mech]}
    event = correction_ledger(ctx).propose(CR.loop_state_digest(prereg, state), mech, L.digest(receipts),
                                           rule["correction_id"], rule["mode"], R.protocol_case_arm(mech, E4.registry_v3()),
                                           budget_gpu_s, evidence)
    return {"decision": "proposed", **event}


def _bound_proposal(request: dict, record_path: Path) -> tuple[dict, dict]:
    proposal = request["evidence"].get("model_proposal")
    record = read(record_path)
    if proposal is None or proposal.get("record_sha256") != sha(record_path, True):
        raise ValueError("교정 요청에 결합된 모델 호출 기록이 아니다")
    return proposal, record


def _correction_record(tag: str, request_id: str, request: dict, proposal: dict, record_path: Path, record: dict,
                       receipt: dict, recovered: bool = False) -> dict:
    import demo_correction_anymal_c as DC
    result_path = DC.fix_folder(tag) / "result.json"
    return {
        "request_id": request_id, "request": {k: request.get(k) for k in ("correction_id", "recovery_mode", "budget_gpu_s")},
        "rule": {k: request["evidence"].get(k) for k in ("rule_mode", "rule", "standstill_ratio")},
        "model_proposal": {k: proposal.get(k) for k in ("accepted", "decision", "agrees_with_rule", "reasons",
                                                         "unmet_checklist", "rule_visible_to_model", "feedback")},
        "call": call_summary(Path(record_path), record), "receipt": receipt,
        "result": read(result_path) if result_path.is_file() else None, "recovered_orphan": recovered}


def correction(tag: str, isaaclab: Path, request_id: str, record_path: Path) -> dict:
    import demo_correction_anymal_c as DC
    ctx = ctx_for(tag)
    previous = _guard(tag, "correction")
    protocol, commit = verify_for(tag, isaaclab)
    request = correction_ledger(ctx).requests().get(request_id)
    if request is None or request["state"] != "approved":
        raise ValueError("승인된 교정 요청이 아니다")
    proposal, record = _bound_proposal(request, record_path)
    rule = E4.recommend(ctx.mechanism, read(ROOT / step_record(tag, "evaluate")["reports"]["FAULT"]["path"]))
    if request["correction_id"] != ctx.correction_id or request["recovery_mode"] != rule["mode"]:
        return _terminal(tag, "correction_not_proposed", {
            "request_id": request_id, "reason": "사전등록 교정·회복 방식과 다른 요청",
            "correction_id": request["correction_id"], "recovery_mode": request["recovery_mode"], "rule_mode": rule["mode"]})
    if request["budget_gpu_s"] < CORRECTION_BUDGET_GPU_S:
        raise ValueError(f"교정 요청 예산 {request['budget_gpu_s']} s가 등록 하한 {CORRECTION_BUDGET_GPU_S} s보다 작다. "
                         "승인을 소비하지 않았다. 요청을 reject하고 2400으로 다시 propose한다")
    if DC.fix_folder(tag).exists():
        raise FileExistsError("교정 출력이 이미 있다")
    log_root = G.log_root(isaaclab)
    if log_root.is_dir() and any(p.name.endswith(f"_{DC.fix_tag(tag)}_{arm}") for p in log_root.iterdir()
                                 for arm in DC.ARMS[rule["mode"]]):
        raise FileExistsError("교정 arm의 SDK 로그 폴더가 이미 있다")
    V2._admit_gpu(protocol)
    receipt = E4.run_correction(ctx, request_id, DC.make_executor(tag, isaaclab), DC.fix_tag(tag))
    return _write_step(tag, "correction", previous, commit, ctx.protocol,
                       _correction_record(tag, request_id, request, proposal, record_path, record, receipt))


def recover_correction(tag: str, isaaclab: Path, request_id: str, record_path: Path) -> dict:
    """실행 중 끊겨 결과 없이 남은 교정 요청을 재실행 없이 닫고 correction 단계를 기록한다(v4와 같다)."""
    import demo_correction_anymal_c as DC
    ctx = ctx_for(tag)
    previous = _guard(tag, "correction")
    _, commit = verify_for(tag, isaaclab)
    led = correction_ledger(ctx)
    if request_id not in led.orphans():
        raise ValueError("결과 없이 남은(고아) 교정 요청이 아니다")
    request = led.requests()[request_id]
    proposal, record = _bound_proposal(request, record_path)
    result_path = DC.fix_folder(tag) / "result.json"
    artifact = None
    if result_path.is_file():
        result = read(result_path)
        expected = {"schema": DC.RESULT_SCHEMA, "tag": DC.fix_tag(tag), "correction_request_id": request_id,
                    "recovery_mode": request["recovery_mode"], "source_commit_sha": commit}
        if any(result.get(k) != v for k, v in expected.items()):
            raise ValueError("교정 결과 파일이 이 요청·시연·실험 커밋의 결과가 아니다(복구하지 않았다)")
        gpu_s = sum(v for v in result["costs"].values() if isinstance(v, (int, float)))
        artifact = {"tag": DC.fix_tag(tag), "verdict": result["label"], "result_sha": sha(result_path),
                    "gpu_s": round(gpu_s, 1), "exit_code": 0, "source": _rel(result_path)}
    receipt = led.recover(request_id, artifact)
    return _write_step(tag, "correction", previous, commit, ctx.protocol,
                       _correction_record(tag, request_id, request, proposal, record_path, record, receipt, True))


def correction_commands(tag: str) -> list[str]:
    model = f"bench/private/e2e_demo_anymal_c/{tag}/model"
    record_file = f"{model}/correction.feedback1.json"
    call = (f"(WSL) evals/feedback_model_call.py --kind correction --packet {model}/correction.packet.json "
            f"--output {record_file} --model {MODEL}")
    return [f"python evals/e2e_demo_anymal_c.py correction-evidence --tag {tag}", call,
            f"python evals/e2e_demo_anymal_c.py correction-propose --tag {tag} --record {record_file}",
            f"python evals/e2e_demo_anymal_c.py approve-correction --tag {tag} --request <rid> --approver <승인자>   "
            "# 또는 reject-correction ... --reason <사유>",
            f"python evals/e2e_demo_anymal_c.py correct-run --tag {tag} --request <rid> --correction-record {record_file}"
            "   # GPU, 실행 중 커밋 금지",
            f"(실행이 끊겨 고아 요청이 남으면) python evals/e2e_demo_anymal_c.py recover-correction --tag {tag} --request <rid> "
            f"--correction-record {record_file}"]


# --- 단계 ---
def begin(tag: str, variant: str, isaaclab: Path) -> dict:
    ctx = ctx_of(variant)
    previous = _guard(tag, "begin")
    protocol, commit = verify_protocol(variant, isaaclab)  # 이 HEAD가 실험 커밋이 된다. 단계 1이 pass가 아니면 여기서 거부
    evidence = seed_unused_evidence(ctx, isaaclab, tag)
    if evidence["hits"] or evidence["existing_outputs"]:
        raise ValueError(f"seed {ctx.seed}/{ctx.restart_seed} 또는 tag 산출물이 이미 있다: {evidence}")
    return _write_step(tag, "begin", previous, commit, ctx.protocol, {
        "tag": tag, "variant": variant, "decision_path": protocol["decision_path"],
        "isaaclab_root": str(isaaclab.resolve()), "seed_unused": evidence, "stage1": protocol["stage1"]["result"]})


def train(tag: str, arm: str, isaaclab: Path, owned=None) -> dict:
    ctx = ctx_for(tag)
    previous = _guard(tag, f"train_{arm}")
    protocol, commit = verify_for(tag, isaaclab)
    name, fault = ctx.cases[arm], "NONE" if arm == "NONE" else ctx.fault
    telemetry = ROOT / RUNS_DIR / f"{name}.telemetry.json"
    if G.find_run_dir(isaaclab, name) or telemetry.exists():
        raise FileExistsError(f"{name}의 학습 산출물이 이미 있다")
    gpu = V2._admit_gpu(protocol)
    folder, attempt = _attempt_folder(tag, f"train_{arm}")
    os.environ["TRIAGE_FAULT"] = fault
    try:
        execution = _execute(folder, B.train_command(isaaclab, name, ctx.seed), isaaclab, protocol["watchdogs_s"]["train"], owned)
    finally:
        os.environ.pop("TRIAGE_FAULT", None)
    readback = [json.loads(line[len("[hidden_fault] "):]) for line in _log_text(folder).splitlines()
                if line.startswith("[hidden_fault] ")]
    run_dirs = G.find_run_dir(isaaclab, name)
    if not _ok(execution) or not readback_ok(readback, fault) or len(run_dirs) != 1:
        _gpu_step_failed(tag, f"train_{arm}", execution, attempt, True)
    extract = _execute(folder / "extract", [G.python_bat(isaaclab), str(CODE_ROOT / "bench" / "extract_tb.py"), str(run_dirs[0]),
                                            str(telemetry)], isaaclab, protocol["watchdogs_s"]["telemetry_extract"], owned)
    if not _ok(extract) or not telemetry.is_file():
        _gpu_step_failed(tag, f"train_{arm}", extract, attempt, True)
    return _write_step(tag, f"train_{arm}", previous, commit, ctx.protocol, {
        "arm": arm, "fault": fault, "gpu_before": gpu, "execution": execution, "extract_execution": extract,
        "readback": readback, "artifacts": training_artifacts(ctx, arm, isaaclab)})


def alarm(tag: str, isaaclab: Path) -> dict:
    """R1을 기록한다. 판정·게이트가 아니므로 계산 오류도 기록하고 다음 단계로 간다. 참조는 단계 1 정상 실행 셋."""
    ctx = ctx_for(tag)
    previous = _guard(tag, "alarm")
    protocol, commit = verify_for(tag, isaaclab)
    references = protocol["stage1"]["reference_telemetry"]
    refs = [read(ROOT / row["path"])["series"][A.METRIC] for row in references.values()]
    results = {}
    for arm in ("NONE", "FAULT"):
        artifacts = _check_artifacts(tag, arm)
        try:
            results[arm] = A.r1(read(ROOT / artifacts["telemetry"]["path"])["series"][A.METRIC], refs)
        except ValueError as error:
            results[arm] = {"error": str(error)[:200]}
    return _write_step(tag, "alarm", previous, commit, ctx.protocol,
                       {"rule": A.R1, "reference_runs": list(references), "results": results})


def evaluate(tag: str, isaaclab: Path, owned=None) -> dict:
    ctx = ctx_for(tag)
    previous = _guard(tag, "evaluate")
    protocol, commit = verify_for(tag, isaaclab)
    artifacts = {arm: _check_artifacts(tag, arm) for arm in ("NONE", "FAULT")}
    outputs = {arm: RESULTS / tag / "runs" / f"{ctx.cases[arm]}.json" for arm in ("NONE", "FAULT")}
    if any(p.exists() for p in outputs.values()):
        raise FileExistsError("고정 평가 결과가 이미 있다")
    gpu = V2._admit_gpu(protocol)
    folder, attempt = _attempt_folder(tag, "evaluate")
    jobs = folder / "jobs.json"
    write_new(jobs, {"action_scale": protocol["action_scale"], "jobs": [
        {"name": ctx.cases[arm], "checkpoint": artifacts[arm]["checkpoint"]["path"], "output": str(outputs[arm])}
        for arm in ("NONE", "FAULT")]})
    argv = [G.python_bat(isaaclab), str(CODE_ROOT / "evals" / "fixed_eval_task.py"), "--protocol", str(EVAL_PROTOCOL),
            "--jobs", str(jobs), "--headless"]
    execution = _execute(folder, argv, isaaclab, protocol["watchdogs_s"]["fixed_eval"], owned)
    if not _ok(execution) or not all(p.is_file() for p in outputs.values()):
        _gpu_step_failed(tag, "evaluate", execution, attempt, any(p.exists() for p in outputs.values()))
    eval_env = FE.parse(_log_text(folder))
    if not G.eval_env_ok(eval_env):
        _terminal(tag, "invalid", {"step": "evaluate", "reason": "평가 환경 readback이 등록값과 다르다", "eval_env": eval_env})
        raise RuntimeError("평가 환경 readback이 등록값과 달라 시연을 invalid로 종료했다")
    reports, paths = {}, {}
    for arm, path in outputs.items():
        reports[arm] = validate_report(read(path), {
            "name": ctx.cases[arm], "checkpoint_file": G.FINAL_CHECKPOINT,
            "checkpoint_sha256": artifacts[arm]["checkpoint"]["sha256"], "action_scale": protocol["action_scale"],
            "protocol_sha256_lf": protocol["eval_protocol_sha256_lf"]})
        paths[arm] = {"path": _rel(path), "sha256_lf": sha(path, True)}
    versions = reports["NONE"]["versions"]
    if (versions != reports["FAULT"]["versions"] or versions["torch"] != protocol["torch_version"]
            or versions["rsl-rl-lib"] != protocol["rsl_rl_version"]):
        raise ValueError("fixed evaluation runtimes differ")
    gate = E4.source_gate(reports["NONE"], reports["FAULT"], protocol["bands"])
    record = _write_step(tag, "evaluate", previous, commit, ctx.protocol,
                         {"gpu_before": gpu, "execution": execution, "eval_env": eval_env, "reports": paths, "gate": gate})
    if gate["outcome"] != "failure_present":
        _terminal(tag, gate["outcome"], {"gate": gate})
    return record


def reference(tag: str, isaaclab: Path, owned=None) -> dict:
    ctx = ctx_for(tag)
    previous = _guard(tag, "reference")
    protocol, commit = verify_for(tag, isaaclab)
    checkpoint = _check_artifacts(tag, "NONE")["checkpoint"]
    out = RESULTS / f"{tag}_ref" / "probes" / f"{ctx.cases['NONE']}.json"
    if out.exists():
        raise FileExistsError("NONE probe 기준이 이미 있다")
    gpu = V2._admit_gpu(protocol)
    folder, attempt = _attempt_folder(tag, "reference")
    jobs = folder / "jobs.json"
    write_new(jobs, {"task": G.TASK, "fault": "NONE", "seed": ctx.seed, "jobs": [
        {"name": ctx.cases["NONE"], "checkpoint": checkpoint["path"], "probes": list(G.PROBE_ORDER), "output": str(out)}]})
    execution = _execute(folder, [G.python_bat(isaaclab), str(CODE_ROOT / "evals" / "probes_task.py"), "--jobs", str(jobs),
                                  "--headless"], isaaclab, protocol["watchdogs_s"]["reference_probes"], owned)
    if not _ok(execution) or not out.is_file():
        _gpu_step_failed(tag, "reference", execution, attempt, out.exists())
    report = read(out)
    probes = report.get("probes", {})
    if (report.get("name") != ctx.cases["NONE"] or report.get("task") != G.TASK
            or report.get("checkpoint", {}).get("sha256") != checkpoint["sha256"]
            or report.get("seed") != ctx.seed or set(probes) != set(L.PROBES)
            or any(v.get("exit") != 0 or v.get("measurement") is None for v in probes.values())):
        raise ValueError("NONE probe 기준이 여섯 probe를 같은 체크포인트로 모두 재지 못했다")
    return _write_step(tag, "reference", previous, commit, ctx.protocol, {
        "gpu_before": gpu, "execution": execution, "report": {"path": _rel(out), "sha256_lf": sha(out, True)}})


def packet(tag: str, isaaclab: Path) -> dict:
    ctx = ctx_for(tag)
    previous = _guard(tag, "packet")
    _, commit = verify_for(tag, isaaclab)
    built = input_packet({"telemetry": _check_artifacts(tag, "FAULT")["telemetry"],
                          "reference_telemetry": _check_artifacts(tag, "NONE")["telemetry"]})
    path = folder_for(tag) / "model" / "diagnosis.packet.json"
    write_new(path, built)
    return _write_step(tag, "packet", previous, commit, ctx.protocol,
                       {"packet": {"path": _rel(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}})


def loop_start(tag: str, isaaclab: Path, record_path: Path) -> dict:
    """v4 loop_start와 같다(호출 기록 검토·인프라 재시도·종료 범주). 다음 명령만 이 실행기의 loop-run."""
    ctx = ctx_for(tag)
    previous = _guard(tag, "loop_start")
    _, commit = verify_for(tag, isaaclab)
    folder = folder_for(tag)
    model_dir = (folder / "model").resolve()
    attempts = sorted(model_dir.glob("diagnosis.attempt*.json"))
    expected = model_dir / f"diagnosis.feedback{len(attempts) + 1}.json"
    if Path(record_path).resolve() != expected:
        raise ValueError(f"다음 진단 호출 기록은 {expected.name}이어야 한다(model 폴더, 순서대로)")
    later = [p.name for p in model_dir.glob("diagnosis.feedback*.json")
             if re.fullmatch(r"diagnosis\.feedback(\d+)\.json", p.name)
             and int(re.fullmatch(r"diagnosis\.feedback(\d+)\.json", p.name)[1]) > len(attempts) + 1]
    if later:
        raise ValueError(f"제출하지 않은 호출 기록이 있다: {later}")
    packet_info = step_record(tag, "packet")["packet"]
    packet_path = ROOT / packet_info["path"]
    if hashlib.sha256(packet_path.read_bytes()).hexdigest() != packet_info["sha256"]:
        raise ValueError("진단 packet이 바뀌었다")
    record = read(record_path)
    review = E4.review_feedback_record(Path(record_path), packet_path, "diagnosis")
    call = call_summary(Path(record_path), record)
    attempt = {"record": _rel(record_path), "record_sha256": sha(record_path), "status": review["status"],
               "reason": review.get("reason"), "feedback": review.get("feedback"), "call": call}
    write_new(folder / "model" / f"diagnosis.attempt{len(attempts) + 1}.json", attempt)
    if review["status"] == "infra_error":
        if len(attempts) + 1 < MAX_INFRA_ATTEMPTS:
            return {"status": "retry_allowed", "attempt": len(attempts) + 1}
        return _terminal(tag, "model_infra_failed", {"attempts": len(attempts) + 1, "last": attempt})
    if review["status"] == "call_error":
        return _terminal(tag, "model_infra_failed", {"attempt": attempt})
    if review["status"] == "invalid":
        return _terminal(tag, "model_invalid", {"attempt": attempt})
    decision, feedback = review["decision"], review["feedback"]
    if decision["action"] != "continue":
        _write_step(tag, "loop_start", previous, commit, ctx.protocol,
                    {"call": call, "feedback": feedback, "decision": decision, "started": None})
        return _terminal(tag, "diagnosis_hold", {"decision": decision})
    ref = read(ROOT / step_record(tag, "reference")["report"]["path"])
    started = V2.start_loop(ctx.cases["FAULT"], decision["ranking"], _check_artifacts(tag, "FAULT")["checkpoint"]["sha256"],
                            {p: v.get("measurement") for p, v in ref["probes"].items()}, decision["probe"])
    record_out = _write_step(tag, "loop_start", previous, commit, ctx.protocol,
                             {"call": call, "feedback": feedback, "decision": decision, "started": started})
    if started["request_id"] is None:
        _terminal(tag, "loop_not_confirmed", {"reason": "상위 가설을 가르는 probe가 없다", "started": started})
        return record_out
    rid, case = started["request_id"], ctx.cases["FAULT"]
    return {**record_out, "next_commands": [
        f"python evals/loop.py approve {case} {rid} --approver <승인자>   # 또는 reject ... --reason <사유>",
        f"python evals/e2e_demo_anymal_c.py loop-run --tag {tag} --request {rid}   # GPU, 실행 중 커밋 금지",
        "다음 제안이 나오면 approve → loop-run을 반복하고, 종료 메시지 뒤 close-loop를 실행한다"]}


def loop_close(tag: str, isaaclab: Path) -> dict:
    ctx = ctx_for(tag)
    previous = _guard(tag, "loop_close")
    _, commit = verify_for(tag, isaaclab)
    summary = loop_summary(ctx, tag, _check_artifacts(tag, "FAULT")["checkpoint"]["sha256"])
    confirmed = summary["status"] == "confirmed"
    proceed = confirmed and summary["hypotheses"] == [ctx.mechanism]
    record = _write_step(tag, "loop_close", previous, commit, ctx.protocol, {"loop": summary, "proceed_to_correction": proceed})
    if not proceed:
        _terminal(tag, "wrong_confirmation" if confirmed else "loop_not_confirmed",
                  {"loop_status": summary["status"], "hypotheses": summary["hypotheses"]})
        return record
    return {**record, "next_commands": [f"python evals/e2e_demo_anymal_c.py specificity --tag {tag}   # 게이트 없음(CPU 기록만)"]}


def specificity(tag: str, isaaclab: Path) -> dict:
    """확정 메커니즘의 특이성 게이트. 등록 변형(reward)에는 게이트가 없어 측정 없이 통과로 기록한다(v4와 같다)."""
    ctx = ctx_for(tag)
    previous = _guard(tag, "specificity")
    protocol, commit = verify_for(tag, isaaclab)
    if protocol["specificity_gate"]["probes"]:
        raise ValueError("이 실행기는 게이트 없는 변형만 등록했다")
    record = _write_step(tag, "specificity", previous, commit, ctx.protocol, {"required": [], "passed": True})
    return {**record, "next_commands": correction_commands(tag)}


# --- 판정 ---
def prediction_results(ctx: Ctx, steps: dict) -> dict:
    """사전등록 예측의 적중(True/False). 그 단계까지 가지 못했으면 None(v4와 같은 계산)."""
    evaluate_step, alarm_step = steps.get("evaluate"), steps.get("alarm")
    start, close, fix = steps.get("loop_start"), steps.get("loop_close"), steps.get("correction")
    gate = (evaluate_step or {}).get("gate") or {}
    alarms = (alarm_step or {}).get("results") or {}
    decision = (start or {}).get("decision") or {}
    observed = ((close or {}).get("loop") or {}).get("observed", {})
    failure = None
    if evaluate_step:
        failure = gate["outcome"] == "failure_present" and (ctx.mechanism != "reward" or gate["standstill_ratio"]["FAULT"] >= 0.9)
    return {
        "source_failure": failure,
        "alarm": None if not alarm_step else (alarms.get("FAULT", {}).get("warned") is True
                                              and alarms.get("NONE", {}).get("warned") is False),
        "diagnosis": None if not start else (decision.get("action") == "continue"
                                             and ctx.mechanism in decision.get("ranking", [])[:LOOP_TOP_K]),
        "loop": None if not close else (close["proceed_to_correction"]
                                        and observed.get(L.confirming_probe(ctx.mechanism)) == "abnormal"),
        "specificity": None if not steps.get("specificity") else steps["specificity"]["passed"] is True,
        "correction_rule": None if not fix else fix["rule"].get("rule_mode") == fix["request"].get("recovery_mode"),
        "correction": None if not fix else fix["receipt"].get("verdict") == "recovered",
    }


def summarize(tag: str) -> dict:
    folder = folder_for(tag)
    if (folder / "result.json").exists():
        return read(folder / "result.json")
    ctx = ctx_for(tag)
    protocol = read(ctx.protocol)
    steps = {name: step_record(tag, name) for name in STEPS if step_path(tag, name).is_file()}
    terminal = read(folder / "terminal.json") if (folder / "terminal.json").exists() else None
    ledger = {"requests": {}, "holds": []}
    if "loop_close" in steps:
        led = correction_ledger(ctx)
        ledger = {"requests": led.requests(), "holds": [e for e in led.events() if e["event"] == "held"]}
    outcome = V2.classify(steps, terminal, ledger)
    checks = E4.conditions(steps, outcome)
    success = outcome == "recovered" and all(checks.values())
    unmet = [k for k, v in checks.items() if not v]
    wording = E4.OUTCOMES[outcome] + (" " + protocol["gap_wording"] + ", ".join(unmet) if unmet and outcome != "in_progress" else "")
    result = {"schema": "e2e_demo_result_anymal_c_v1", "tag": tag, "task": G.TASK, "variant": ctx.variant, "fault": ctx.fault,
              "mechanism": ctx.mechanism, "seed": ctx.seed, "case": ctx.cases["FAULT"],
              "decision_path": protocol["decision_path"], "outcome": outcome, "demo_success": success,
              "success_conditions": checks, "unmet_conditions": unmet, "wording": wording,
              "common_wording": protocol["common_wording"], "steps_sha256": {k: sha(step_path(tag, k)) for k in steps},
              "terminal": terminal, "protocol_sha256_lf": sha(ctx.protocol, True),
              "source_commit_sha": sorted({s["source_commit_sha"] for s in steps.values()}),
              "stage1": protocol["stage1"]["result"],
              "predictions": protocol["predictions"], "prediction_results": prediction_results(ctx, steps),
              "model_feedback": E4.model_feedback(steps, terminal, ledger),
              "approver_identity_attested": False, **E4.public_view(steps)}
    if outcome != "in_progress":
        write_new(RESULTS / tag / "demo_result.json", result)
        write_new(folder / "result.json", result)
    return result


def status(tag: str) -> dict:
    folder = folder_for(tag)
    done = [name for name in STEPS if step_path(tag, name).is_file()]
    terminal = read(folder / "terminal.json") if (folder / "terminal.json").exists() else None
    following = None if terminal or len(done) == len(STEPS) else STEPS[len(done)]
    return {"tag": tag, "done": done, "next": following, "terminal": terminal}


# --- 하네스 실패(사전등록 범주, v4와 같은 조건) ---
def tool_commit() -> str:
    try:
        return P.verify_committed(ROOT, {"evals/e2e_demo_anymal_c.py"})
    except subprocess.CalledProcessError as error:
        raise ValueError(f"git check failed: {' '.join(map(str, error.cmd))[:200]}") from error


def record_harness_failure(tag: str, step: str, error: str, cause: str) -> dict:
    """시연 실행기 결함으로 멈춘 시연의 종료 기록(harness_failure)을 한 번 쓴다. 판정 범주를 새로 만들지 않는다."""
    import correct as CR
    folder = folder_for(tag)
    if not folder.is_dir():
        raise ValueError("시연 폴더가 없다")
    if (folder / "terminal.json").exists():
        raise FileExistsError("종료 기록이 이미 있다")
    recorded = {p.name.split("_", 1)[1][:-5]: p for p in sorted((folder / "steps").glob("[0-9][0-9]_*.json"))}
    if "begin" not in recorded or set(recorded) - set(STEPS):
        raise ValueError("시연 단계 기록이 등록 단계와 다르다")
    if "correction" in recorded:
        raise ValueError("교정 단계가 기록된 시연은 하네스 실패로 닫지 않는다")
    last = list(recorded)[-1]
    allowed = {STEPS[STEPS.index(last) + 1]} | (set(E4.CORRECTION_COMMANDS) if last == "specificity" else set())
    if last == "loop_start":
        allowed.add("loop-run")
    if step not in allowed:
        raise ValueError(f"멈춘 단계는 마지막 기록 단계({last}) 바로 다음 등록 단계·명령이어야 한다: {step!r}")
    if not isinstance(error, str) or not error.strip():
        raise ValueError("오류 문자열이 비었다")
    if E4.ABSOLUTE_PATH.search(error):
        raise ValueError("오류 문자열에 절대 경로를 넣지 않는다(공개 요약에 실린다)")
    begin_record = read(recorded["begin"])
    protocol_path = ctx_of(begin_record["variant"]).protocol
    if not protocol_path.is_file() or sha(protocol_path, True) != begin_record.get("protocol_sha256_lf"):
        raise ValueError("시연 프로토콜이 begin 기록과 다르다")
    case = read(protocol_path)["cases"]["FAULT"]
    ledger_path = CR._dir(case, CR.LOOPS) / "correction_ledger.jsonl"
    events = CorrectionLedger(ledger_path).events()
    if any(e["event"] in ("approved", "rejected", "consumed", "receipt", "held") for e in events):
        raise ValueError("교정 원장에 승인·거절·소비·실행·보류 사건이 있어 하네스 실패로 닫지 않는다(등록 결과 범주로 끝난다)")
    counts = {kind: sum(1 for e in events if e["event"] == kind) for kind in sorted({e["event"] for e in events})}
    loop_state = CR._dir(case, CR.LOOPS) / "state.json"
    detail = {"demo": "anymal_c", "tag": tag, "case": case, "step": step, "error": error.strip()[:500],
              "cause": E4._cause(cause, begin_record["source_commit_sha"]), "approval_consumed": False,
              "correction_ledger": {"exists": ledger_path.exists(), "events": counts},
              "probe_loop_status": read(loop_state).get("status") if loop_state.is_file() else None,
              "recorded_steps": list(recorded),
              "recorded_by": "evals/e2e_demo_anymal_c.py record-harness-failure",
              "tool_commit": tool_commit(), "tool_sha256_lf": sha(Path(__file__), True),
              "category_preregistered_for_demo": True}
    record = {"outcome": "harness_failure", "at": V2.now(), "detail": detail, "wording": E4.OUTCOMES["harness_failure"]}
    write_new(folder / "terminal.json", record)
    return record


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="두 번째 로봇 task(ANYmal-C flat) F1 종단 시연(시연 v4 흐름)")
    ap.add_argument("command", choices=("stage1", "prepare", "verify", "begin", "train", "alarm", "evaluate", "reference",
                                        "packet", "start-loop", "loop-run", "close-loop", "specificity", "correction-evidence",
                                        "correction-propose", "correct-run", "approve-correction", "reject-correction",
                                        "correction-status", "recover-correction", "record-harness-failure", "summarize",
                                        "status"))
    ap.add_argument("--isaaclab", type=Path, default=Path.home() / "IsaacLab")
    ap.add_argument("--variant", choices=tuple(VARIANTS))
    ap.add_argument("--tag")
    ap.add_argument("--arm", choices=("NONE", "FAULT"))
    ap.add_argument("--record", type=Path)
    ap.add_argument("--request")
    ap.add_argument("--correction-record", type=Path)
    ap.add_argument("--budget-gpu-s", type=int, default=CORRECTION_BUDGET_GPU_S)
    ap.add_argument("--approver")
    ap.add_argument("--reason")
    ap.add_argument("--step")
    ap.add_argument("--error")
    ap.add_argument("--cause")
    args = ap.parse_args(argv)
    required = {"stage1": (), "prepare": ("variant",), "verify": ("variant",), "begin": ("variant", "tag"),
                "train": ("tag", "arm"), "start-loop": ("tag", "record"), "loop-run": ("tag", "request"),
                "correction-propose": ("tag", "record"), "correct-run": ("tag", "request", "correction_record"),
                "recover-correction": ("tag", "request", "correction_record"),
                "approve-correction": ("tag", "request", "approver"),
                "reject-correction": ("tag", "request", "approver", "reason"),
                "record-harness-failure": ("tag", "step", "error", "cause")}
    for name in required.get(args.command, ("tag",)):
        if getattr(args, name) is None:
            ap.error(f"{args.command}에는 --{name.replace('_', '-')}가 필요하다")
    os.environ.update(V2.UTF8_ENV)  # Isaac 하위 프로세스(cp949 콘솔) 출력 오류 방지
    commands = {
        "stage1": stage1_status,
        "prepare": lambda: prepare(args.variant, args.isaaclab),
        "verify": lambda: {"status": "pass", "source_commit_sha": verify_protocol(
            args.variant, args.isaaclab, experiment_commit(args.tag) if args.tag else None)[1]},
        "begin": lambda: begin(args.tag, args.variant, args.isaaclab),
        "train": lambda: train(args.tag, args.arm, args.isaaclab),
        "alarm": lambda: alarm(args.tag, args.isaaclab),
        "evaluate": lambda: evaluate(args.tag, args.isaaclab),
        "reference": lambda: reference(args.tag, args.isaaclab),
        "packet": lambda: packet(args.tag, args.isaaclab),
        "start-loop": lambda: loop_start(args.tag, args.isaaclab, args.record),
        "loop-run": lambda: loop_run(args.tag, args.isaaclab, args.request),
        "close-loop": lambda: loop_close(args.tag, args.isaaclab),
        "specificity": lambda: specificity(args.tag, args.isaaclab),
        "correction-evidence": lambda: correction_evidence(args.tag, args.isaaclab),
        "correction-propose": lambda: correction_propose(args.tag, args.isaaclab, args.record, args.budget_gpu_s),
        "correct-run": lambda: correction(args.tag, args.isaaclab, args.request, args.correction_record),
        "approve-correction": lambda: approve_correction(args.tag, args.request, args.approver),
        "reject-correction": lambda: reject_correction(args.tag, args.request, args.approver, args.reason),
        "correction-status": lambda: correction_status(args.tag),
        "recover-correction": lambda: recover_correction(args.tag, args.isaaclab, args.request, args.correction_record),
        "record-harness-failure": lambda: record_harness_failure(args.tag, args.step, args.error, args.cause),
        "summarize": lambda: summarize(args.tag),
        "status": lambda: status(args.tag),
    }
    try:
        out = commands[args.command]()
    except GPUBusy as error:
        print(f"대기: {error}. 아무것도 기록하지 않았다", file=sys.stderr)
        return 3
    except Stage1NotPassed as error:
        print(f"단계 2 시작 안 함: {error}", file=sys.stderr)
        return 4
    except (ValueError, FileExistsError, FileNotFoundError, RuntimeError, L.LedgerError) as error:
        print(f"거부: {error}", file=sys.stderr)
        return 2
    except SystemExit as error:  # evals/loop.py의 거부(고아·없는 요청·기존 측정)는 SystemExit 문자열이다
        if isinstance(error.code, int) or error.code is None:
            raise
        print(f"거부: {error.code}", file=sys.stderr)
        return 2
    print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
