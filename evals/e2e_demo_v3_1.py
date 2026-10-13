"""새 seed 결함 종단 시연 v3.1: v3과 같은 흐름·규칙·게이트에서 교정 원장을 시연 등록 표(registry_v3)로 일관되게 만든 실행기.

v3 결과[사실]: seed 2031 F2 시연(e2e_demo_v3_f2_s2031_20261013)은 actuator 확정·특이성 통과·교정 제안(restart) 뒤
correction-propose에서 "actuator에 등록되지 않은 회복 방식 'restart'"로 멈췄다. v3의 correction_propose·correction과
evals/correct.py approve·run이 교정 원장을 기본 등록 표(actuator: resume)로 만들었기 때문이다. 원장 기록·승인 소비는 없었다.
v3 실행기·교정 실행기는 공개 기록 replay의 소스 결합 때문에 바꾸지 않는다. v3과 다른 점:
- 교정 원장은 correction_ledger(ctx) 하나로만 만든다(CorrectionLedger(evals/loops/<사례>/correction_ledger.jsonl,
  registry=registry_v3())). 제안·승인·거절·소비·실행·조회가 모두 이 원장을 쓴다. 승인·거절은 이 파일의
  approve-correction·reject-correction으로 한다(evals/correct.py approve는 기본 표 원장이라 registry digest가 달라 거부된다).
- 교정 실행은 run_correction(evals/correct.py run과 같은 순서: 고아 확인 → 승인 소비 → 실행 → receipt)으로 한다.
  교정 실행기는 bench/demo_correction_v3_1.py다. 실행 중 프로세스가 끊겨 남은 고아 요청은 recover-correction이 재실행
  없이 닫는다(교정 결과 파일이 있으면 그 값, 없으면 unknown)고 correction 단계를 같은 모양으로 기록한다.
- 교정 예산 하한(2400 s)을 원장에 쓰기 전(correction-propose)에 검사한다. 종료 기록이 있으면 제안·승인·거절을 거부한다.
- 새 seed: f2 2033(재학습 3033), f1 2034(재학습 3034). 프로토콜 bench/protocols/e2e_demo_v3_1_<변형>.json,
  출력 bench/private/e2e_demo_v3_1.
- 결과 범주 harness_failure(사전등록): 시연 실행기 결함으로 등록된 다음 단계를 진행할 수 없을 때 record-harness-failure로
  사건(단계·오류·원인 파일:줄·승인 소비 없음·기록 시각·도구 커밋)을 종료 기록에 남긴다. v3 시연 폴더에도 쓸 수 있고, 그때는
  v3 사전등록에 없던 범주를 v3.1 도구로 사후 기록했다고 표기한다. harness-summary는 그 시연의 진단·고리·특이성·교정
  제안까지를 부분 공개 요약으로 쓴다.

v3 설명(그대로 적용):
흐름은 v2와 같다: 학습 실패 입력(NONE·결함 새 학습) → 추종 경고 R1 → 고정 평가 → NONE probe 기준 → 첫 판단(새 호출) →
승인된 probe 고리 → 확정 → 교정 근거 packet → 교정 제안(새 호출) → 교정 승인 → 교정 학습·고정 평가 → 결과 문구.
v1·v2 실행기는 공개 결과의 소스 결합 때문에 바꾸지 않고, v2의 순수 함수(진단 기록 검토·고리 시작·실험 커밋 고정)만
재사용한다. v2와 다른 점:
- 변형(VARIANTS)을 begin 기록에 묶는다. 프로토콜은 변형마다 bench/protocols/e2e_demo_v3_1_<변형>.json 하나다.
- 회복 방식 규칙을 이 파일의 RECOVERY_RULES로 사전등록한다(bound correction_registry.py는 바꿀 수 없다).
  값이 비어 있는 메커니즘의 변형은 prepare를 거부한다. 모터(actuator)는 W1 rule_update change에 따라 restart다.
  교정 근거 packet·제안 검증·원장 요청은
  이 규칙과 그 규칙이 허용하는 회복 방식을 더한 등록 표(registry_v3)로 만든다.
- 교정은 restart(재학습 seed = 실패 seed + 1000, 같은 seed NONE 원 정책 대비)와 resume(같은 seed로 300회 이어 학습,
  같은 갱신량으로 이어 학습한 정상 기준 대비, v4 기준과 같다)을 모두 실행한다(bench/demo_correction_v3_1.py).
- 결정 경로 스위치(decision_path): single(v2와 같은 단일 호출) 또는 nat_v4(E4b v4 NAT 도구 호출 경로: 진단은 실수 사실
  표기 고정·표기 일치 치환, 교정은 null 없는 제출·packet 등록 표 검증). nat_v4는 E4b v4 공개 요약이 adopt이고 그 요약
  파일을 프로토콜에 SHA로 묶을 때만 prepare할 수 있다(E4b v2·v3는 do_not_adopt). NAT 교정 제출은 packet에 실린 등록 표
  (registry_v3)로 검증하므로 등록 표를 넓힌 메커니즘에도 쓸 수 있다.
- 특이성 게이트(SPECIFICITY_GATES): 확정 메커니즘에 등록된 보조 probe를 결함 정책에서 따로 재고 모두 normal일 때만
  교정 단계로 간다. 모터(actuator)는 P_noise·P_value다. 동결 P_torque는 F3 4/4·F4 1/4 실행에서도 abnormal이었고,
  "P_torque abnormal ∧ P_noise normal ∧ P_value normal"은 이미 본 F2 4/4·그 밖 0/27이었다(이미 본 seed에서 고른 시연 게이트).

  python evals/e2e_demo_v3_1.py prepare --variant f2 [--decision-path single|nat_v4] [--isaaclab <경로>]
  python evals/e2e_demo_v3_1.py verify --variant f2 [--tag <tag>]
  python evals/e2e_demo_v3_1.py begin --variant f2 --tag <tag>
  python evals/e2e_demo_v3_1.py train --tag <tag> --arm NONE|FAULT
  python evals/e2e_demo_v3_1.py alarm|evaluate|reference|packet|close-loop|correction-evidence|summarize|status --tag <tag>
  python evals/e2e_demo_v3_1.py start-loop --tag <tag> --record <진단 호출 기록>
  python evals/e2e_demo_v3_1.py correction-propose --tag <tag> --record <교정 호출 기록>
  python evals/e2e_demo_v3_1.py correct-run --tag <tag> --request <rid> --correction-record <교정 호출 기록>
  python evals/e2e_demo_v3_1.py approve-correction --tag <tag> --request <rid> --approver <승인자>
  python evals/e2e_demo_v3_1.py reject-correction --tag <tag> --request <rid> --approver <승인자> --reason <사유>
  python evals/e2e_demo_v3_1.py correction-status --tag <tag>
  python evals/e2e_demo_v3_1.py recover-correction --tag <tag> --request <rid> --correction-record <교정 호출 기록>
  python evals/e2e_demo_v3_1.py record-harness-failure --demo v3|v3_1 --tag <tag> --step <단계> --error <오류 문자열> --cause <파일:줄>
  python evals/e2e_demo_v3_1.py harness-summary --demo v3|v3_1 --tag <tag>
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
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

import e2e_reward_demo_v2 as V2  # noqa: E402
import run_training_correction as V3  # noqa: E402
import run_runtime_readback_smoke as P  # noqa: E402
import runtime_readback_smoke as S  # noqa: E402
from rl_triage import correction_registry as R  # noqa: E402
from rl_triage import grounded_recovery as C  # noqa: E402
from rl_triage import probe_loop as L  # noqa: E402
from rl_triage import recovery as RC  # noqa: E402
from rl_triage import tracking_alarm as A  # noqa: E402
from rl_triage.correction_ledger import CorrectionLedger  # noqa: E402

read = V3.read
write_new = V3.write_new
sha = V3.sha

PRIVATE = ROOT / "bench/private/e2e_demo_v3_1"
CODE_ROOT = ROOT  # 하네스 실패 원인 파일:줄의 기준(입력·출력 위치인 ROOT와 따로 둬 테스트가 ROOT만 바꿀 수 있다)
RESULTS = ROOT / "evals/results"
PROTOCOL_DIR = ROOT / "bench/protocols"
V4_PROTOCOL = V2.V4_PROTOCOL
FIXED_EVAL = V2.FIXED_EVAL
BANDS_SOURCE = V2.BANDS_SOURCE
E4B_SUMMARY = ROOT / "evals/results/e4b_nat_compare_v4/summary.json"
RUNS_DIR = V2.RUNS_DIR
REFERENCE_RUNS = V2.REFERENCE_RUNS
MODEL = V2.MODEL
STEPS = ("begin", "train_NONE", "train_FAULT", "alarm", "evaluate", "reference", "packet", "loop_start",
         "loop_close", "specificity", "correction")
LOOP_TOP_K, LOOP_BUDGET, PROBE_EXECUTION_BUDGET_S = V2.LOOP_TOP_K, V2.LOOP_BUDGET, V2.PROBE_EXECUTION_BUDGET_S
CORRECTION_BUDGET_GPU_S = 2400
MAX_INFRA_ATTEMPTS = 2
TAG_RE = V2.TAG_RE
RESTART_OFFSET = 1000
DECISION_PATHS = ("single", "nat_v4")
APPROVAL = V2.APPROVAL
GPUBusy = V2.GPUBusy

# 변형: 결함·P0-C 사례 id·메커니즘·새 실패 seed. seed는 2026~2032·3029~3032·7·42·123과 개발 실행에 쓰지 않은 값이다
# (2031·3031은 시연 v3 F2, 2032·3032는 시연 v3 F1 등록 값).
VARIANTS = {
    "f2": {"fault": "F2", "case_id": "h06", "mechanism": "actuator", "seed": 2033},
    "f1": {"fault": "F1", "case_id": "h02", "mechanism": "reward", "seed": 2034},
}
# 회복 방식 규칙(사전등록). mode가 None이면 그 메커니즘의 변형은 prepare를 거부한다.
RECOVERY_RULES = {
    "reward": {"kind": "standstill_threshold", "threshold": 0.9, "at_or_above": "restart", "below": "resume",
               "source": "v4 사전등록 규칙(seed 2027 단일 관찰에서 만듦), v4 seed 2026·2028 적중, 시연 v2 seed 2030 recovered"},
    "actuator": {"kind": "fixed", "mode": "restart",
                 "source": ("W1(bench/protocols/training_correction_motor_v1.json, 결과 evals/results/motor_w1_s2026_20261013·"
                            "motor_w1_s2028_20261013) rule_update=change, next_rule '(fault_mechanism == reward and "
                            "original_fault_standstill_ratio >= 0.9) or fault_mechanism == actuator -> restart, otherwise resume'. "
                            "교정 뒤 이어 학습 300회의 선속도 오차가 seed 2026 0.2247·2028 0.2079로 같은 갱신량 정상 기준 "
                            "0.1433·0.1327 대비 1.57배라 두 seed 모두 not_recovered였다. 중간 체크포인트 오차는 단조 감소했다"
                            "(2026: 1.017·0.877·0.613·0.383·0.225, 2028: 1.016·0.945·0.753·0.356·0.208, 이어 학습 갱신 "
                            "0·51·151·251·299회, budget_limited_candidate 2/2). W1 문구: 이번 실험은 restart를 시험하지 않았고 "
                            "이어 학습의 예산 부족 가능성도 남으므로 restart를 택할 근거로 쓰지 않는다. 이 시연이 그 restart arm의 "
                            "사전등록 시험이다")},
}
CODE = V2.CODE | {
    "evals/e2e_reward_demo_v2.py", "evals/e2e_demo_v3_1.py", "bench/demo_correction_v3_1.py",
    "evals/nat_decision_call.py", "evals/nat_decision_call_v2.py", "evals/nat_decision_call_v3.py",
    "evals/nat_decision_call_v4.py", "src/rl_triage/nat_demo_tools.py", "src/rl_triage/nat_demo_tools_v2.py",
    "src/rl_triage/nat_demo_tools_v3.py", "src/rl_triage/nat_demo_tools_v4.py",
    "configs/nat_demo_v4_diagnosis.yml", "configs/nat_demo_v4_correction.yml",
}
# 확정 메커니즘별 특이성 게이트 보조 probe(사전등록). 결함 정책에서 재고 모두 normal이어야 교정으로 간다.
SPECIFICITY_GATES = {"actuator": ["P_noise", "P_value"]}


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
        return {"NONE": f"baseline_p0c_s{self.seed}", "FAULT": f"{self.case_id}_s{self.seed}"}

    @property
    def correction_id(self) -> str:
        return R.REGISTRY[self.mechanism]["correction_id"]

    @property
    def protocol(self) -> Path:
        return PROTOCOL_DIR / f"e2e_demo_v3_1_{self.variant}.json"


def ctx_of(variant: str) -> Ctx:
    if variant not in VARIANTS:
        raise ValueError(f"등록되지 않은 변형 {variant!r}")
    return Ctx(variant=variant, **VARIANTS[variant])


# --- 회복 방식 규칙과 등록 표 ---
def rule_modes(mechanism: str) -> set[str]:
    rule = RECOVERY_RULES.get(mechanism)
    if rule is None:
        return set()
    if rule["kind"] == "standstill_threshold":
        return {rule["at_or_above"], rule["below"]}
    return set() if rule["mode"] is None else {rule["mode"]}


def registry_v3() -> dict:
    """기본 등록 표에 v3 규칙이 내는 회복 방식을 더한다. 등록 메커니즘·교정은 바꾸지 않는다."""
    return {mech: {**deepcopy(entry), "recovery_modes": sorted(set(entry["recovery_modes"]) | rule_modes(mech))}
            for mech, entry in R.REGISTRY.items()}


def recommend(mechanism: str, source_report: dict) -> dict:
    """R.recommend_recovery_mode와 같은 모양. 등록 밖이면 HOLD, 규칙 값이 비었으면 ValueError."""
    if mechanism not in R.REGISTRY or mechanism not in RECOVERY_RULES:
        return {"mode": R.HOLD, "mechanism": mechanism, "correction_id": None, "standstill_ratio": None,
                "rule": "unregistered_mechanism"}
    rule = RECOVERY_RULES[mechanism]
    ratio = R.standstill_ratio(source_report)
    if rule["kind"] == "standstill_threshold":
        mode = rule["at_or_above"] if ratio >= rule["threshold"] else rule["below"]
        text = f"{mechanism} and standstill_ratio >= {rule['threshold']} -> {rule['at_or_above']}, otherwise {rule['below']}"
    else:
        mode, text = rule["mode"], f"{mechanism} -> {rule['mode']}"
    if mode is None:
        raise ValueError(f"{mechanism} 회복 방식 규칙이 아직 비어 있다(사전등록 전)")
    return {"mode": mode, "mechanism": mechanism, "correction_id": R.REGISTRY[mechanism]["correction_id"],
            "standstill_ratio": ratio, "rule": text}


# --- 사전등록 ---
OUTCOMES = {
    "invalid": "출처·커밋·seed·산출물 결합 검증이 맞지 않아 시연 판정을 보류한다.",
    "baseline_invalid": "같은 seed 정상 정책이 고정 평가에서 걷지 못해 비교 기준이 없다. 원인 진단을 시작하지 않았다.",
    "source_failure_absent": "결함 학습 정책이 같은 seed 정상 정책 대비 고정 행동 허용 범위를 통과해 진단할 실패가 없었다.",
    "model_infra_failed": "첫 판단 호출이 인프라 오류로 등록 횟수 안에 응답을 주지 못해 진단을 보류했다.",
    "model_invalid": "첫 판단 응답이 공개 판단 계약을 어겨 검사를 제안하지 않고 보류했다.",
    "diagnosis_hold": "첫 판단이 근거 부족으로 보류해 검사를 실행하지 않았다.",
    "loop_not_confirmed": "승인된 검사들로 원인을 하나로 확정하지 못해(식별 불가 또는 지원 가설 없음) 교정을 제안하지 않았다.",
    "wrong_confirmation": "검사 고리가 등록 결함과 다른 메커니즘을 확정했다. 등록 결함과 다른 확정으로 기록하고 교정하지 않았다.",
    "confirmation_not_specific": "확정 메커니즘의 특이성 게이트 보조 측정에서 다른 메커니즘의 검사가 abnormal이었거나 판정할 수 없어(unknown) 교정하지 않았다.",
    "correction_not_proposed": "확정 뒤 교정 단계가 보류(HOLD)했거나 사전등록 회복 방식과 다른 요청이라 교정을 실행하지 않았다.",
    "correction_rejected": "승인자가 교정 요청을 거절해 교정을 실행하지 않았다.",
    "correction_unknown": "승인된 교정 실행이 실패·예산 초과·검증 불일치·기준 무효로 끝나 회복 여부를 판정하지 않았다.",
    "not_recovered": "등록 결함을 확정하고 승인된 교정까지 실행했지만, 교정 정책이 사전등록 기준 정책 대비 고정 행동 허용 범위를 통과하지 못했다.",
    "recovered": "새 판단으로 시작한 승인된 검사가 등록 결함 메커니즘을 확정했고, 승인된 교정 정책이 사전등록 기준 정책 대비 고정 행동 허용 범위를 통과했다.",
    "harness_failure": ("시연 실행기 결함으로 등록된 다음 단계를 진행할 수 없어 시연을 멈췄다. 사건(단계·오류·원인·승인 소비 "
                        "없음)을 기록했고 교정·회복 결과를 판정하지 않는다."),
    "in_progress": "아직 끝나지 않은 시연이다.",
}
TERMINAL_OUTCOMES = set(OUTCOMES) - {"recovered", "not_recovered", "correction_unknown", "in_progress"}
# 하네스 실패 기록을 쓸 수 있는 시연 판과 멈춘 단계(단계 기록 이름 또는 그 사이의 명령 이름)
HARNESS_DEMOS = ("v3_1", "v3")
CORRECTION_COMMANDS = ("correction-evidence", "correction-propose", "approve-correction", "reject-correction",
                       "correct-run")
# 오류 문자열의 절대 경로: 드라이브(C:\·C:/), UNC(\\서버), 홈(~/), 유닉스·Git Bash(/home/, /c/ 등). URL(https://)은 허용
ABSOLUTE_PATH = re.compile(r"(?<![A-Za-z])[A-Za-z]:[\\/](?!/)|\\\\|(^|[\s'\"(=])(~[\\/]|/[A-Za-z]{1,8}/)")
HARNESS_FAILURE_RULE = {
    "when": ("시연 실행기(이 파일·교정 실행기) 결함으로 등록된 다음 단계 명령이 같은 입력에서 거부되어 그 판의 등록 절차로 "
             "진행할 수 없을 때"),
    "not_for": "재시도 규칙이 있는 인프라 오류, 승인자 거절, 사전등록 결과 범주로 끝나는 경우",
    "preconditions": ["종료 기록이 없다", "교정 단계 기록이 없다",
                      ("멈춘 단계가 마지막 기록 단계 바로 다음 등록 단계이거나, 마지막 기록이 specificity일 때 교정 명령"
                       "(correction-evidence·correction-propose·approve-correction·reject-correction·correct-run) 중 하나다"),
                      "교정 원장에 approved·rejected·consumed·receipt·held 사건이 없다(승인 소비·등록 결과 범주 없음)",
                      ("원인 '저장소 상대 경로:줄'이 시연 실험 커밋에 같은 내용으로 있는 파일의 범위 안이다(bench/private·.git "
                       "제외)"), "오류 문자열에 절대 경로가 없다", "기록 도구(이 파일)가 커밋돼 있다"],
    "fields": ["demo(v3|v3_1)", "tag", "case", "step", "error", "cause(file·line·text·sha256_lf·commit)",
               "approval_consumed", "correction_ledger(exists·events)", "probe_loop_status", "recorded_steps", "at",
               "recorded_by", "tool_commit", "tool_sha256_lf", "category_preregistered_for_demo"],
    "v3_folder": ("v3 시연 폴더에 쓰면 v3 사전등록에 없던 범주를 v3.1 도구로 사후 기록한 것으로 표기한다. v3 summarize는 이 "
                  "범주를 모르므로 harness-summary로 부분 공개 요약을 쓴다"),
    "summary": ("harness-summary(v3 폴더): 진단·고리·특이성·교정 제안(단일 호출 검증)까지의 공개 수치와 하네스 실패 기록. "
                "교정·회복 판정은 없다. v3.1 시연은 summarize가 이 범주를 그대로 요약한다"),
}
SUCCESS_CONDITIONS = {
    "protocol_committed": "모든 단계가 실행 전에 커밋된 같은 프로토콜과 하나의 실험 커밋에서 돌았다",
    "seed_unused": "시작 시 실패 seed·재학습 seed의 학습·평가·고리 산출물이 없었다",
    "source_failure_present": "같은 seed 정상 정책은 걷고 결함 정책은 unhealthy였다",
    "fresh_diagnosis_call": "새 첫 판단 호출이 기록됐고 계약상 유효한 continue였다",
    "first_probe_from_model": "첫 검사 제안이 모델이 고른 probe였다(규칙 대체가 아님)",
    "probes_approved_consumed_once": "모든 실행 probe가 승인 기록 뒤 한 번만 소비됐고 산출물 체크포인트 SHA가 사례와 같았다",
    "loop_confirmed_mechanism": "동결 규칙 갱신으로 등록 결함의 메커니즘이 유일 가설로 확정됐다",
    "confirmation_specific": "확정 메커니즘의 특이성 게이트 보조 probe가 모두 normal이었거나 게이트가 없는 메커니즘이었다",
    "correction_call_accepted": "새 교정 제안 호출이 기록됐고 검증을 통과해 교정 요청의 근거로 결합됐다",
    "correction_approved_consumed_once": "승인 기록 뒤 교정 요청이 한 번 소비되고 receipt가 남았다",
    "correction_recovered": "교정 정책이 사전등록 기준 정책 대비 healthy였다",
    "costs_recorded": "학습·평가·probe 프로세스 시간과 두 모델 호출의 지연·토큰이 모두 기록됐다",
}
PREDICTIONS = {
    "f2": {
        "source_failure": ("F2가 같은 seed NONE 대비 unhealthy(P0-C seed 2026~2028 h06 모두 unhealthy, 정지 비율 0.864~0.907, "
                           "시연 v3 seed 2031도 unhealthy·정지 비율 0.871)"),
        "alarm": "R1이 F2에 울리고 NONE에는 울리지 않는다(이미 본 seed h06 세 개 모두 울림)",
        "diagnosis": "첫 판단 action=continue, 상위 3가설에 actuator 포함(시연 v3 seed 2031: continue, actuator 2위)",
        "loop": "P_torque abnormal 뒤 actuator 확정(예산 4 이내, 이미 본 F2 4개 실행과 시연 v3 seed 2031 모두 abnormal)",
        "specificity": "P_noise·P_value normal(이미 본 F2 4개 실행과 시연 v3 seed 2031 모두 normal)",
        "correction_rule": "actuator → restart(W1 rule_update change)",
        "correction": ("교정 정책 recovered. [추론] restart arm은 F2 결함에 등록 교정(restore_motor_torque_speed)을 건 채 "
                       "seed 3033으로 처음부터 학습한다. 교정이 모터 곡선을 정상으로 되돌리면 정상 학습과 seed만 다르다. 모터 "
                       "결함의 restart는 W1·이전 실험에서 시험하지 않았다(시연 v3 seed 2031은 교정 제안 단계에서 "
                       "하네스 결함으로 멈췄다)"),
    },
    "f1": {
        "source_failure": "F1이 같은 seed NONE 대비 unhealthy이고 정지 비율 ≥ 0.9",
        "alarm": "R1이 F1에 울리고 NONE에는 울리지 않는다",
        "diagnosis": "첫 판단 action=continue, 상위 3가설에 reward 포함",
        "loop": "P_reward abnormal 뒤 reward 확정(예산 4 이내)",
        "specificity": "게이트 없음",
        "correction_rule": "reward·정지 비율 ≥ 0.9 → restart",
        "correction": "교정 정책 recovered",
    },
}


def fixed(variant: str, decision_path: str) -> dict:
    ctx = ctx_of(variant)
    if decision_path not in DECISION_PATHS:
        raise ValueError(f"등록되지 않은 결정 경로 {decision_path!r}")
    rule = RECOVERY_RULES[ctx.mechanism]
    return {
        "schema": "e2e_demo_protocol_v3_1", "variant": variant, "decision_path": decision_path,
        "question": ("새 seed의 등록 결함 학습 실패 하나를 경고·고정 평가·새 판단·승인 검사·교정 제안·승인 교정·고정 평가로 "
                     "끝까지 연결해 같은 사례/요청 ID로 기록할 수 있는가"),
        "scope": "single_new_seed_synthetic_fault_integration_demo_not_accuracy_generalization_or_savings",
        "fault": ctx.fault, "fault_mechanism": ctx.mechanism, "case_id": ctx.case_id, "cases": ctx.cases,
        "fault_definition": f"bench/hidden_faults.py {ctx.fault}(동결, 크기·정의 변경 없음)",
        "seed": ctx.seed, "restart_seed": ctx.restart_seed,
        "seed_selection": ("seed 2026~2032·3029~3032·7·42·123과 개발 실행에 쓰지 않은 값. 시작 시 실패·재학습 seed 산출물 "
                           "부재를 확인한다"),
        "task": C.TASK, "num_envs": 4096, "max_iterations": 300, "num_steps_per_env": 24, "action_scale": 0.25,
        "telemetry_dir": RUNS_DIR, "approval": APPROVAL, "source_binding": V2.SOURCE_BINDING,
        "recovery_rule": {"mechanism": ctx.mechanism, **deepcopy(rule)},
        "registry": registry_v3(),
        "correction": {
            "correction_id": ctx.correction_id, "budget_gpu_s_min": CORRECTION_BUDGET_GPU_S,
            "modes": {
                "restart": {"arms": ["restart_corrected"], "training_seed": ctx.restart_seed,
                            "reference": "같은 seed NONE 원 정책(시연 고정 평가)",
                            "reason": "같은 seed 처음부터 학습은 정상 학습을 바이트 단위로 재현하므로 실패 seed + 1000"},
                "resume": {"arms": ["clean_resume_reference", "resume_corrected"], "training_seed": ctx.seed,
                           "resume_iteration": 299, "final_iteration": 598,
                           "reference": "같은 seed NONE 체크포인트를 같은 갱신량으로 이어 학습한 정책(v4 기준과 같다)",
                           "reference_validity": "기준이 걷고 같은 seed NONE 원 정책 대비 healthy여야 한다. 아니면 correction_unknown"},
            },
            "verdict": "recovery.verdict(p0c 밴드)",
            "budget_floor_checked_at": "correction-propose(원장에 쓰기 전). 하한 미만·정수 아님이면 요청을 만들지 않는다",
            "terminal_guard": "종료 기록이 있으면 교정 근거·제안·승인·거절을 거부한다",
            "recover": ("실행 중 끊겨 결과 없이 남은(고아) 요청만 recover-correction으로 재실행 없이 닫는다. 교정 결과 파일"
                        "(fix/result.json)이 있고 그 스키마·fix 태그·요청 id·회복 방식·실험 커밋이 요청과 같으면 그 판정·SHA·"
                        "비용 합으로, 없으면 unknown(orphan_without_artifact)으로 receipt를 쓰고 correction 단계를 같은 모양으로 "
                        "기록한다(recovered_orphan=true, receipt.recovered_from 공개). 결과 파일이 요청과 다르면 거부한다"),
            "model_call": {"single": "evals/correction_model_call.py(WSL, temperature 0)",
                           "nat_v4": "evals/nat_decision_call_v4.py --kind correction(WSL)"}[decision_path],
            "auto_approval": False,
            "approval": f"evals/e2e_demo_v3_1.py approve-correction/reject-correction(시연 등록 표 원장). {APPROVAL}",
            "ledger": ("CorrectionLedger(evals/loops/<사례>/correction_ledger.jsonl, registry=registry_v3())로 제안·승인·"
                       "소비·실행·조회"),
        },
        "diagnosis": {"model": MODEL, "packet_schema": "active_decision_packet_v2",
                      "decision_contract": "rl_triage.active_diagnosis.validate_decision",
                      "call": {"single": "evals/active_model_call.py(WSL, temperature 0)",
                               "nat_v4": "evals/nat_decision_call_v4.py --kind diagnosis(WSL)"}[decision_path],
                      "max_infra_attempts": MAX_INFRA_ATTEMPTS, "blind_inputs": "사례 이름·정답·결함 정의를 넣지 않는다"},
        "decision_path_rule": ("nat_v4는 E4b v4 공개 요약이 adopt일 때만 쓰며 그 파일을 SHA로 묶는다. NAT 경로는 제출 전 계약 점검 "
                               "도구(실행당 최대 2회)를 쓰므로 단일 호출과 조건이 다르다"),
        "specificity_gate": {"probes": SPECIFICITY_GATES.get(ctx.mechanism, []),
                             "rule": ("고리 확정 뒤 결함 정책에서 보조 probe를 따로 재 같은 seed NONE 기준으로 동결 classify한다. 모두 "
                                      "normal이면 교정으로 가고, 하나라도 abnormal·unknown이면 confirmation_not_specific로 끝낸다"),
                             "basis": ("이미 본 holdout·dev 실행 31개에서 P_torque abnormal은 F2 4/4와 F3 4/4·F4 1/4에서 나왔고, P_noise·P_value "
                                       "normal을 더하면 F2 4/4·그 밖 0/27이었다. 이미 본 seed에서 고른 시연 게이트이며 진단 정확도 근거가 아니다")
                             if ctx.mechanism in SPECIFICITY_GATES else "게이트 없음",
                             "approval": "고리 밖 사전등록 측정 단계(정상 기준 probe 단계처럼 승인 원장을 거치지 않는다)"},
        "loop": {"case": ctx.cases["FAULT"], "top_k": LOOP_TOP_K, "budget_probes": LOOP_BUDGET,
                 "first_probe": "모델이 고른 probe가 상위 3가설을 가르면 그 probe, 아니면 동결 규칙 probe(대체로 기록)",
                 "update": "동결 probe_loop.update 규칙. 관측 뒤 모델 재호출 없음", "metered": True,
                 "approval": f"evals/loop.py approve/reject. {APPROVAL}",
                 "proceed_rule": f"상태 confirmed이고 유일 가설이 {ctx.mechanism}일 때만 교정 단계로 간다"},
        "alarm": {**A.R1, "reference_runs": list(REFERENCE_RUNS),
                  "status": "이미 본 seed 2026~2028에서 사후에 고른 후보 상수. 판정·게이트에 쓰지 않는다"},
        "evaluation": {"protocol": "bench/protocols/fixed_eval_v1.json", "bands_source": V3.FIXED["bands_source"],
                       "failure_rule": "같은 seed NONE이 걷고(err_xy < 0.5×제자리, 낙상 ≤ 0.05) 결함 정책이 NONE 대비 unhealthy면 실패 존재"},
        "gpu_admission_profile": V3.FIXED["gpu_admission_profile"], "isaaclab_commit": V3.FIXED["isaaclab_commit"],
        "rsl_rl_version": V3.FIXED["rsl_rl_version"], "torch_version": V3.FIXED["torch_version"],
        "watchdogs_s": {"train": 1800, "fixed_eval": 600, "reference_probes": 900, "specificity_probes": 600,
                        "correction_train": 1800, "correction_eval": 600},
        "predictions": PREDICTIONS[variant],
        "success_conditions": SUCCESS_CONDITIONS,
        "success_rule": "결과가 recovered이고 성공 조건이 모두 참일 때만 demo_success",
        "outcomes": OUTCOMES,
        "common_wording": (f"새 seed {ctx.seed}의 합성 결함({ctx.fault}) 1건을 끝까지 연결한 통합 시연이다. 결함과 probe는 같은 "
                           "설계자가 만들었고, 진단 정확도·일반화·교정 성공률·GPU 절감의 근거가 아니다."),
        "gap_wording": "다음 성공 조건을 충족하지 못했다: ",
        "semantic_retries": 0,
        "changes_from_v2": ["결함 변형·회복 방식 규칙을 사전등록 값으로 고른다", "resume 교정과 같은 갱신량 정상 기준 arm",
                            "결정 경로 스위치(single·nat_v4)", "확정 뒤 특이성 게이트 단계(모터: P_noise·P_value)",
                            "결과 스키마 e2e_demo_result_v3·e2e_correction_result_v3"],
        "changes_from_v3": ["교정 원장을 제안·승인·거절·소비·실행·조회 모두 시연 등록 표(registry_v3)로 만든다",
                            "승인·거절 CLI를 이 실행기에 둔다(approve-correction·reject-correction)",
                            "교정 실행 run_correction과 교정 실행기 bench/demo_correction_v3_1.py",
                            "새 seed f2 2033·f1 2034", "결과 범주 harness_failure와 사후 기록·부분 요약 CLI",
                            "교정 예산 하한을 제안 단계에서 검사하고, 종료 뒤 교정 근거·제안·승인·거절을 거부한다",
                            "고아 교정 요청 복구 recover-correction(재실행 없음, 결과 파일 출처 대조)",
                            "결과 스키마 e2e_demo_result_v3_1·e2e_correction_result_v3_1"],
        "harness_failure_rule": HARNESS_FAILURE_RULE,
        "forbidden_claims": ["진단 정확도·우월성", "일반화", "교정 성공률", "GPU 절감", "이미 본 seed 결과와 합친 분모",
                             "승인자 문자열을 신원 인증으로 표기", "운영자 승인을 사람 승인으로 표기",
                             "NAT 경로와 단일 호출 경로를 같은 조건으로 비교"],
    }


DYNAMIC = {"source_files", "reference_telemetry", "bands", "bands_summary_sha256_lf", "fixed_eval_sha256_lf",
           "v4_protocol_sha256_lf", "correction_source_contract", "nat_adoption"}


def _rel(path: Path) -> str:
    return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()


def nat_adoption(decision_path: str) -> dict | None:
    """nat_v4면 E4b v4 공개 요약의 adopt 판정을 SHA와 함께 묶는다."""
    if decision_path != "nat_v4":
        return None
    if not E4B_SUMMARY.is_file():
        raise ValueError("E4b v4 공개 요약이 없어 NAT 경로를 쓸 수 없다")
    summary = read(E4B_SUMMARY)
    if (summary.get("adoption") or {}).get("decision") != "adopt":
        raise ValueError("E4b v4 판정이 adopt가 아니라 NAT 경로를 쓸 수 없다")
    return {"summary": _rel(E4B_SUMMARY), "sha256_lf": sha(E4B_SUMMARY, True), "decision": "adopt"}


def dynamic_fields(decision_path: str, isaaclab: Path) -> dict:
    v4 = read(V4_PROTOCOL)
    contract = v4["correction_source_contract"]
    V3.check_contract(contract, isaaclab)
    return {
        "source_files": {name: sha(ROOT / name, True) for name in sorted(CODE)},
        "reference_telemetry": {name: sha(ROOT / RUNS_DIR / f"{name}.telemetry.json", True) for name in REFERENCE_RUNS},
        "bands": read(BANDS_SOURCE)["bands"], "bands_summary_sha256_lf": sha(BANDS_SOURCE, True),
        "fixed_eval_sha256_lf": sha(FIXED_EVAL, True), "v4_protocol_sha256_lf": sha(V4_PROTOCOL, True),
        "correction_source_contract": contract, "nat_adoption": nat_adoption(decision_path),
    }


def check_protocol(protocol: dict, dynamic: dict) -> None:
    expected = fixed(protocol.get("variant"), protocol.get("decision_path"))
    if set(protocol) != set(expected) | DYNAMIC:
        raise ValueError("protocol fields differ")
    if any(protocol[k] != v for k, v in expected.items()):
        raise ValueError("fixed demo contract changed")
    changed = sorted(k for k, v in dynamic.items() if protocol[k] != v)
    if changed:
        raise ValueError(f"bound sources changed: {changed}")


def prepare(variant: str, decision_path: str, isaaclab: Path) -> dict:
    ctx = ctx_of(variant)
    if not rule_modes(ctx.mechanism):
        raise ValueError(f"{ctx.mechanism} 회복 방식 규칙이 비어 있다. W1 결과로 RECOVERY_RULES를 채운 뒤 prepare한다")
    write_new(ctx.protocol, {**deepcopy(fixed(variant, decision_path)), **dynamic_fields(decision_path, isaaclab)})
    return {"status": "prepared_commit_before_begin", "path": _rel(ctx.protocol)}


def verify_protocol(variant: str, isaaclab: Path, experiment: str | None = None) -> tuple[dict, str]:
    """프로토콜·결합 파일·Isaac Lab을 검사한다. 실험 커밋이 없으면(begin) HEAD를, 있으면 실험 커밋을 돌려준다."""
    ctx = ctx_of(variant)
    protocol = read(ctx.protocol)
    if protocol.get("variant") != variant:
        raise ValueError("프로토콜 변형이 다르다")
    check_protocol(protocol, dynamic_fields(protocol["decision_path"], isaaclab))
    bound = CODE | {_rel(ctx.protocol), _rel(FIXED_EVAL), _rel(BANDS_SOURCE), _rel(V4_PROTOCOL),
                    *(f"{RUNS_DIR}/{name}.telemetry.json" for name in REFERENCE_RUNS)}
    if protocol["nat_adoption"] is not None:
        bound.add(protocol["nat_adoption"]["summary"])
    try:
        if S._git_head(isaaclab) != protocol["isaaclab_commit"] or not S._git_tracked_clean(isaaclab):
            raise ValueError("Isaac Lab checkout differs")
        head = P.verify_committed(ROOT, bound)
    except subprocess.CalledProcessError as error:
        raise ValueError(f"git check failed: {' '.join(map(str, error.cmd))[:200]}") from error
    return protocol, head if experiment is None else V2.frozen_commit(experiment, head, bound)


# --- 단계 원장 ---
def folder_for(tag: str) -> Path:
    if not isinstance(tag, str) or not re.fullmatch(TAG_RE, tag):
        raise ValueError("invalid demo tag")
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
    if outcome not in TERMINAL_OUTCOMES:
        raise ValueError(f"등록되지 않은 조기 종료 {outcome}")
    record = {"outcome": outcome, "at": V2.now(), "detail": detail, "wording": OUTCOMES[outcome]}
    write_new(folder_for(tag) / "terminal.json", record)
    return record


def _execute(tag: str, step: str, argv: list[str], watchdog_s: float, owned=None) -> tuple[dict, int]:
    owned = owned or P.run_owned
    if any(char in arg for arg in argv for char in '\r\n"%&^'):
        raise ValueError("unsafe Windows batch argument")
    exec_root = folder_for(tag) / "exec"
    attempt = 1 + sum(1 for p in exec_root.glob(f"{step}_attempt*") if p.is_dir()) if exec_root.exists() else 1
    folder = exec_root / f"{step}_attempt{attempt}"
    folder.mkdir(parents=True)
    execution = owned(folder, argv, ROOT, watchdog_s)
    execution = {**execution, "parent_wall_s": round(execution["wall_s"], 1), "attempt": attempt, "folder": _rel(folder)}
    write_new(folder / "execution.json", execution)
    return execution, attempt


def _gpu_step_failed(tag: str, step: str, execution: dict, attempt: int, output_exists: bool) -> None:
    detail = {"step": step, "attempt": attempt, "execution": {k: execution.get(k) for k in
                                                            ("exit_code", "natural_exit", "timed_out", "error_type",
                                                             "cleanup_verified", "folder")}}
    if (step in ("evaluate", "reference", "specificity") and attempt == 1 and not output_exists
            and execution.get("cleanup_verified")):
        raise RuntimeError(f"{step} 인프라 실패, 결과 파일이 없어 1회 재시도할 수 있다: {detail}")
    _terminal(tag, "invalid", detail)
    raise RuntimeError(f"{step} 실패로 시연을 invalid로 종료했다: {detail}")


# --- 입력 확인 ---
def seed_unused_evidence(ctx: Ctx, isaaclab: Path, tag: str) -> dict:
    log_root = isaaclab / "logs/rsl_rl/unitree_go2_flat"
    roots = [ROOT / RUNS_DIR, ROOT / "bench/private", ROOT / "evals/loops", log_root]
    seeds = (ctx.seed, ctx.restart_seed)
    names = tuple(f"_s{s}" for s in seeds)
    hits = sorted(str(p) for root in roots if root.exists() for p in root.iterdir() if any(n in p.name for n in names))
    hits += sorted(f"{run}: seed {seed}" for run, seed in V2._logged_seeds(log_root).items() if seed in seeds)
    outputs = [str(p) for p in (folder_for(tag), RESULTS / tag, RESULTS / f"{tag}_ref", RESULTS / f"{tag}_loop",
                                RESULTS / f"{tag}_spec", ROOT / "bench/private/e2e_demo_v3" / tag) if p.exists()]
    return {"checked_roots": [str(r) for r in roots], "checked_seeds": list(seeds), "hits": hits,
            "existing_outputs": outputs}


def training_artifacts(ctx: Ctx, arm: str, isaaclab: Path) -> dict:
    name = ctx.cases[arm]
    meta_path = ROOT / "bench/private" / f"{name}.meta.json"
    meta = read(meta_path)
    expected = {"task": C.TASK, "seed": ctx.seed, "num_envs": 4096, "max_iterations": 300,
                "fault": "NONE" if arm == "NONE" else ctx.fault, "train_exit_code": 0}
    if any(type(meta.get(k)) is not type(v) or meta[k] != v for k, v in expected.items()):
        raise ValueError(f"{name}: training metadata differs")
    log_root = (isaaclab / "logs/rsl_rl/unitree_go2_flat").resolve()
    run_dir = Path(meta["run_dir"]).resolve()
    matches = [p.resolve() for p in log_root.iterdir() if p.is_dir() and p.name.endswith("_" + name)]
    if run_dir.parent != log_root or matches != [run_dir]:
        raise ValueError(f"{name}: run must match exactly one SDK log directory")
    files = {"checkpoint": run_dir / "model_299.pt", "first_checkpoint": run_dir / "model_0.pt",
             "environment": run_dir / "params/env.yaml", "agent": run_dir / "params/agent.yaml"}
    telemetry = ROOT / RUNS_DIR / f"{name}.telemetry.json"
    series = read(telemetry)["series"].get(A.METRIC)
    if not isinstance(series, list) or len(series) != 300:
        raise ValueError(f"{name}: telemetry does not cover 300 iterations")
    if V3.load_config(files["agent"]).get("seed") != ctx.seed:
        raise ValueError(f"{name}: agent seed differs")
    return {"run": name, "run_dir": str(run_dir), "meta": {"path": _rel(meta_path), "sha256": sha(meta_path)},
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


def source_gate(none_report: dict, fault_report: dict, bands: dict) -> dict:
    verdict = RC.verdict(RC.compare(C._behavior(fault_report), C._behavior(none_report)), bands)
    none_walks = V2.walks(none_report)
    outcome = ("baseline_invalid" if not none_walks else
               "failure_present" if verdict["label"] == "unhealthy" else "source_failure_absent")
    ratio = {arm: r["metrics"]["lin_vel_rmse_mps"] / r["standstill"]["lin_vel_rmse_mps"]
             for arm, r in (("NONE", none_report), ("FAULT", fault_report))}
    return {"none_walks": none_walks, "fault_verdict": verdict, "standstill_ratio": ratio, "outcome": outcome}


# --- 결정 호출 기록 검토(단일 호출·NAT) ---
def review_nat_record(record_path: Path, packet_path: Path, kind: str, registry: dict | None = None) -> dict:
    """E4b v4 NAT 기록을 packet·설정·도구·프롬프트에 묶어 검토한다. 상태: valid|infra_error|invalid.
    진단 제출 파일의 decision은 표기 일치 claim을 저장 값으로 바꾼 결정이다. 교정은 registry(시연 등록 표)로 검증한다."""
    import correction_model_call as CM
    import nat_decision_call as W1
    import nat_decision_call_v4 as W4
    from rl_triage import active_diagnosis as D
    record, data = read(record_path), Path(packet_path).read_bytes()
    expected = {"schema": W4.RECORD_SCHEMA, "kind": kind, "model": MODEL, "temperature": 0,
                "packet_sha256": hashlib.sha256(data).hexdigest(), "config_sha256_lf": W1.lf_sha(W4.CONFIGS[kind]),
                "tools_modules_sha256_lf": {name: W1.lf_sha(path) for name, path in W4.TOOLS_MODULES.items()},
                "system_prompt_sha256": W1.text_sha(W4.system_prompt(kind))}
    if any(record.get(k) != v for k, v in expected.items()):
        raise ValueError("NAT 기록이 이 packet·설정·도구·프롬프트와 다르다")
    status = record.get("status")
    if status == "infra_error":
        return {"status": "infra_error", "reason": str(record.get("error_type"))[:200]}
    if status != "valid" or Path(record_path).with_suffix(".submission.rejected.json").exists():
        return {"status": "invalid", "reason": str(record.get("error_type"))[:200]}
    submission = Path(record_path).with_suffix(".submission.json")
    if hashlib.sha256(submission.read_bytes()).hexdigest() != record.get("submission_sha256"):
        raise ValueError("NAT 제출 파일이 기록과 다르다")
    body, packet = read(submission), json.loads(data)
    if kind == "diagnosis":
        decision = D.validate_decision(body["decision"], packet)
        if decision != record.get("decision"):
            raise ValueError("NAT 제출 결정이 다시 검증되지 않는다")
        return {"status": "valid", "decision": decision}
    verdict = CM.validate_model_proposal(body["proposal"], packet, registry)
    return {"status": "valid" if verdict["accepted"] else "invalid", "proposal": body["proposal"], **verdict}


def call_summary(record_path: Path, record: dict) -> dict:
    """단계 기록에 남길 호출 비용·식별자. NAT는 여러 LLM 호출의 합과 실행 wall이다."""
    if record.get("schema") == "active_model_call_v1" or "execution_wall_s" in record:
        keys = ("response_id", "provider_model", "input_tokens", "output_tokens", "total_tokens", "execution_wall_s",
                "finish_reason")
        return {"path": "single", "record": _rel(record_path), "record_sha256": sha(record_path),
                **{k: record.get(k) for k in keys}}
    return {"path": "nat_v4", "record": _rel(record_path), "record_sha256": sha(record_path),
            "input_tokens": record.get("input_tokens"), "output_tokens": record.get("output_tokens"),
            "execution_wall_s": record.get("wall_s"), "llm_calls": record.get("llm_calls"),
            "precheck_calls": record.get("precheck_calls")}


def loop_summary(ctx: Ctx, tag: str, checkpoint_sha: str) -> dict:
    """끝난 고리의 원장·상태를 읽고 실행된 probe마다 산출물의 사례·체크포인트 SHA를 대조한다."""
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
                    raise ValueError(f"{req['probe']}: 등록과 달리 --metered 없이 실행됐다")
                row["artifact"] = {"probe": _rel(probe_file), "execution": _rel(meta_file)}
                costs.append({k: meta.get(k) for k in ("gpu_wall_s", "cpu_s", "simulator_steps")} | {"probe": req["probe"]})
        rows.append(row)
    return {"status": state["status"], "hypotheses": state["hypotheses"], "observed": state["observed"],
            "history": state["history"], "requests": rows, "probe_costs": costs}


# --- 교정 원장(시연 등록 표) ---
def correction_ledger(ctx: Ctx) -> CorrectionLedger:
    """이 시연의 교정 원장. 제안·승인·거절·소비·실행·조회 모두 시연 등록 표(registry_v3)로 만든다."""
    import correct as CR
    return CorrectionLedger(CR._dir(ctx.cases["FAULT"], CR.LOOPS) / "correction_ledger.jsonl", registry=registry_v3())


def run_correction(ctx: Ctx, request_id: str, execute, tag: str) -> dict:
    """evals/correct.py run과 같은 순서(고아 확인 → 승인 소비 → 실행 → receipt)를 시연 등록 표 원장으로 한다."""
    import correct as CR
    case = ctx.cases["FAULT"]
    led = correction_ledger(ctx)
    orphans = led.orphans()
    if orphans:
        raise L.LedgerError(f"결과 없이 끝난 요청 {orphans}이 있다. 먼저 recover-correction으로 닫는다")
    req = led.requests().get(request_id)
    if req is None:
        raise L.LedgerError(f"{request_id}: 없는 요청")
    prereg, state, _ = CR._loop_state(case, CR.LOOPS)
    led.consume(request_id, CR.loop_state_digest(prereg, state))  # 승인 없음·거절·재소비·상태 변경이면 여기서 멈춘다
    request = {"case": case, "request_id": request_id, "correction_id": req["correction_id"],
               "recovery_mode": req["recovery_mode"], "protocol_case_arm": req["protocol_case_arm"],
               "budget_gpu_s": req["budget_gpu_s"], "tag": tag}
    started = time.perf_counter()
    result = execute(request)  # 예외는 그대로 전파한다. 요청은 running(고아)으로 남는다
    gpu_s = result.get("gpu_s", time.perf_counter() - started)
    return led.receipt(request_id, result.get("tag", tag), result["verdict"], result.get("result_sha"), gpu_s,
                       result.get("exit_code"))


def _require_open(tag: str) -> None:
    terminal = folder_for(tag) / "terminal.json"
    if terminal.exists():
        raise ValueError(f"시연이 이미 종료됐다: {read(terminal)['outcome']}")


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


# --- 교정 근거·제안(v3 규칙과 등록 표) ---
def correction_packet(ctx: Ctx, tag: str) -> dict:
    """correct.py evidence와 같은 packet을 v3 규칙·등록 표로 만든다."""
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
                           observed=state.get("observed", {}), rule=recommend(mech, report), source_report=report,
                           source_report_sha256=report_info["sha256_lf"], checkpoint_sha256=prereg.get("checkpoint_sha256"),
                           registry=registry_v3())


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


def review_correction_record(protocol: dict, record_path: Path, packet: dict, packet_path: Path) -> dict:
    """교정 호출 기록(결정 경로별)을 packet에 묶어 검증한 원장 근거. 형식·결합 불일치는 ValueError."""
    import correction_model_call as CM
    record_sha = sha(record_path, True)
    if protocol["decision_path"] == "single":
        return CM.review_call_record(read(record_path), packet, record_sha, registry_v3())
    review = review_nat_record(record_path, packet_path, "correction", registry_v3())
    if review["status"] == "infra_error":
        raise ValueError("NAT 교정 호출이 인프라 오류다(같은 packet으로 재시도)")
    record = read(record_path)
    return {"schema": "e2e_nat_correction_review_v3_1", "record_sha256": record_sha,
            "packet_digest": CM.packet_digest(packet), "model": record.get("model"), "provider_model": None,
            "response_id": None, "rule_visible_to_model": True, "precheck_calls": record.get("precheck_calls"),
            "proposal": review.get("proposal"), **{k: review.get(k) for k in
                                                   ("accepted", "decision", "reasons", "agrees_with_rule", "unmet_checklist")}}


def correction_propose(tag: str, isaaclab: Path, record_path: Path, budget_gpu_s: int = CORRECTION_BUDGET_GPU_S) -> dict:
    """확정된 고리에서 v3 규칙의 교정 요청 하나를 원장에 올린다(correct.py propose의 v3 규칙판)."""
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
    review = review_correction_record(protocol, Path(record_path), packet, packet_path)
    case = ctx.cases["FAULT"]
    prereg, state, loop_led = CR._loop_state(case, CR.LOOPS)
    mech, req = CR._confirming_request(case, state, loop_led)
    rule = recommend(mech, read(ROOT / step_record(tag, "evaluate")["reports"]["FAULT"]["path"]))
    receipts = [e for e in loop_led.events() if e["event"] == "receipt"]
    evidence = {"confirming_probe": req["probe"], "confirming_request_id": req["request_id"],
                "standstill_ratio": rule["standstill_ratio"], "rule": rule["rule"], "rule_mode": rule["mode"],
                "source_report_sha256": step_record(tag, "evaluate")["reports"]["FAULT"]["sha256_lf"],
                "model_proposal": review, "registry_v3": registry_v3()[mech]}
    event = correction_ledger(ctx).propose(CR.loop_state_digest(prereg, state), mech, L.digest(receipts),
                                           rule["correction_id"], rule["mode"], R.protocol_case_arm(mech, registry_v3()),
                                           budget_gpu_s, evidence)
    return {"decision": "proposed", **event}


def correction_approval(ctx: Ctx, request_id: str) -> dict:
    events = correction_ledger(ctx).events()
    approved = [e for e in events if e.get("event") == "approved" and e.get("request_id") == request_id]
    if len(approved) != 1:
        raise ValueError("교정 요청의 승인 사건이 하나가 아니다")
    return {"approver": approved[0]["approver"], "at": approved[0]["at"]}


# --- 단계 ---
def begin(tag: str, variant: str, isaaclab: Path) -> dict:
    ctx = ctx_of(variant)
    previous = _guard(tag, "begin")
    protocol, commit = verify_protocol(variant, isaaclab)  # 이 HEAD가 실험 커밋이 된다
    evidence = seed_unused_evidence(ctx, isaaclab, tag)
    if evidence["hits"] or evidence["existing_outputs"]:
        raise ValueError(f"seed {ctx.seed}/{ctx.restart_seed} 또는 tag 산출물이 이미 있다: {evidence}")
    return _write_step(tag, "begin", previous, commit, ctx.protocol, {
        "tag": tag, "variant": variant, "decision_path": protocol["decision_path"],
        "isaaclab_root": str(isaaclab.resolve()), "seed_unused": evidence})


def train(tag: str, arm: str, isaaclab: Path, owned=None) -> dict:
    ctx = ctx_for(tag)
    previous = _guard(tag, f"train_{arm}")
    protocol, commit = verify_for(tag, isaaclab)
    gpu = V2._admit_gpu(protocol)
    argv = ["pwsh", "-NoProfile", "-File", str(ROOT / "bench/run_case.ps1"),
            "-CaseId", "baseline_p0c" if arm == "NONE" else ctx.case_id, "-Seed", str(ctx.seed),
            "-NumEnvs", str(protocol["num_envs"]), "-MaxIterations", str(protocol["max_iterations"]),
            "-Fault", "NONE" if arm == "NONE" else ctx.fault, "-OutDir", str(ROOT / RUNS_DIR), "-IsaacLab", str(isaaclab)]
    execution, attempt = _execute(tag, f"train_{arm}", argv, protocol["watchdogs_s"]["train"], owned)
    if not V3.execution_valid(execution):
        _gpu_step_failed(tag, f"train_{arm}", execution, attempt, True)
    return _write_step(tag, f"train_{arm}", previous, commit, ctx.protocol, {
        "arm": arm, "gpu_before": gpu, "execution": execution, "artifacts": training_artifacts(ctx, arm, isaaclab)})


def alarm(tag: str, isaaclab: Path) -> dict:
    """R1을 기록한다. 판정·게이트가 아니므로 계산 오류도 기록하고 다음 단계로 간다."""
    ctx = ctx_for(tag)
    previous = _guard(tag, "alarm")
    _, commit = verify_for(tag, isaaclab)
    refs = [read(ROOT / RUNS_DIR / f"{name}.telemetry.json")["series"][A.METRIC] for name in REFERENCE_RUNS]
    results = {}
    for arm in ("NONE", "FAULT"):
        artifacts = _check_artifacts(tag, arm)
        try:
            results[arm] = A.r1(read(ROOT / artifacts["telemetry"]["path"])["series"][A.METRIC], refs)
        except ValueError as error:
            results[arm] = {"error": str(error)[:200]}
    return _write_step(tag, "alarm", previous, commit, ctx.protocol,
                       {"rule": A.R1, "reference_runs": list(REFERENCE_RUNS), "results": results})


def evaluate(tag: str, isaaclab: Path, owned=None) -> dict:
    ctx = ctx_for(tag)
    previous = _guard(tag, "evaluate")
    protocol, commit = verify_for(tag, isaaclab)
    artifacts = {arm: _check_artifacts(tag, arm) for arm in ("NONE", "FAULT")}
    gpu = V2._admit_gpu(protocol)
    argv = [sys.executable, str(ROOT / "evals/run_fixed_eval.py"), "--tag", tag, "--runs", ctx.cases["NONE"],
            ctx.cases["FAULT"], "--isaaclab", str(isaaclab)]
    execution, attempt = _execute(tag, "evaluate", argv, protocol["watchdogs_s"]["fixed_eval"], owned)
    outputs = {arm: RESULTS / tag / "runs" / f"{ctx.cases[arm]}.json" for arm in ("NONE", "FAULT")}
    if not V3.execution_valid(execution) or not all(p.is_file() for p in outputs.values()):
        _gpu_step_failed(tag, "evaluate", execution, attempt, any(p.exists() for p in outputs.values()))
    reports, paths = {}, {}
    for arm, path in outputs.items():
        reports[arm] = C.validate_fixed_report(read(path), {
            "name": ctx.cases[arm], "checkpoint_file": "model_299.pt",
            "checkpoint_sha256": artifacts[arm]["checkpoint"]["sha256"], "action_scale": protocol["action_scale"],
            "protocol_sha256_lf": protocol["fixed_eval_sha256_lf"]})
        paths[arm] = {"path": _rel(path), "sha256_lf": sha(path, True)}
    versions = reports["NONE"]["versions"]
    if (versions != reports["FAULT"]["versions"] or versions["torch"] != protocol["torch_version"]
            or versions["rsl-rl-lib"] != protocol["rsl_rl_version"]):
        raise ValueError("fixed evaluation runtimes differ")
    gate = source_gate(reports["NONE"], reports["FAULT"], protocol["bands"])
    record = _write_step(tag, "evaluate", previous, commit, ctx.protocol,
                         {"gpu_before": gpu, "execution": execution, "reports": paths, "gate": gate})
    if gate["outcome"] != "failure_present":
        _terminal(tag, gate["outcome"], {"gate": gate})
    return record


def reference(tag: str, isaaclab: Path, owned=None) -> dict:
    ctx = ctx_for(tag)
    previous = _guard(tag, "reference")
    protocol, commit = verify_for(tag, isaaclab)
    checkpoint = _check_artifacts(tag, "NONE")["checkpoint"]["sha256"]
    out = RESULTS / f"{tag}_ref" / "probes" / f"{ctx.cases['NONE']}.json"
    if out.exists():
        raise FileExistsError("NONE probe 기준이 이미 있다")
    gpu = V2._admit_gpu(protocol)
    argv = [sys.executable, str(ROOT / "evals/run_probes.py"), "--tag", f"{tag}_ref", "--runs", ctx.cases["NONE"],
            "--isaaclab", str(isaaclab)]
    execution, attempt = _execute(tag, "reference", argv, protocol["watchdogs_s"]["reference_probes"], owned)
    if not V3.execution_valid(execution) or not out.is_file():
        _gpu_step_failed(tag, "reference", execution, attempt, out.exists())
    report = read(out)
    probes = report.get("probes", {})
    if (report.get("name") != ctx.cases["NONE"] or report.get("checkpoint", {}).get("sha256") != checkpoint
            or report.get("seed") != ctx.seed or set(probes) != set(L.PROBES)
            or any(v.get("exit") != 0 or v.get("measurement") is None for v in probes.values())):
        raise ValueError("NONE probe 기준이 여섯 probe를 같은 체크포인트로 모두 재지 못했다")
    return _write_step(tag, "reference", previous, commit, ctx.protocol, {
        "gpu_before": gpu, "execution": execution, "report": {"path": _rel(out), "sha256_lf": sha(out, True)}})


def packet(tag: str, isaaclab: Path) -> dict:
    import run_active_diagnosis as RA
    ctx = ctx_for(tag)
    previous = _guard(tag, "packet")
    _, commit = verify_for(tag, isaaclab)
    sources = {"telemetry": _check_artifacts(tag, "FAULT")["telemetry"],
               "reference_telemetry": _check_artifacts(tag, "NONE")["telemetry"]}
    built = RA.input_packet({"sources": sources}, {"max_probes": LOOP_BUDGET,
                                                   "probe_execution_budget_s": PROBE_EXECUTION_BUDGET_S})
    path = folder_for(tag) / "model" / "diagnosis.packet.json"
    write_new(path, built)
    return _write_step(tag, "packet", previous, commit, ctx.protocol,
                       {"packet": {"path": _rel(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}})


def _record_name(decision_path: str) -> str:
    return "call" if decision_path == "single" else "nat"


def loop_start(tag: str, isaaclab: Path, record_path: Path) -> dict:
    ctx = ctx_for(tag)
    previous = _guard(tag, "loop_start")
    protocol, commit = verify_for(tag, isaaclab)
    kind = _record_name(protocol["decision_path"])
    folder = folder_for(tag)
    model_dir = (folder / "model").resolve()
    attempts = sorted(model_dir.glob("diagnosis.attempt*.json"))
    expected = model_dir / f"diagnosis.{kind}{len(attempts) + 1}.json"
    if Path(record_path).resolve() != expected:
        raise ValueError(f"다음 진단 호출 기록은 {expected.name}이어야 한다(model 폴더, 순서대로)")
    later = [p.name for p in model_dir.glob(f"diagnosis.{kind}*.json")
             if re.fullmatch(rf"diagnosis\.{kind}(\d+)\.json", p.name)
             and int(re.fullmatch(rf"diagnosis\.{kind}(\d+)\.json", p.name)[1]) > len(attempts) + 1]
    if later:
        raise ValueError(f"제출하지 않은 호출 기록이 있다: {later}")
    packet_info = step_record(tag, "packet")["packet"]
    packet_path = ROOT / packet_info["path"]
    if hashlib.sha256(packet_path.read_bytes()).hexdigest() != packet_info["sha256"]:
        raise ValueError("진단 packet이 바뀌었다")
    record = read(record_path)
    review = (V2.review_diagnosis_record(record, packet_path) if protocol["decision_path"] == "single"
              else review_nat_record(Path(record_path), packet_path, "diagnosis"))
    attempt = {"record": _rel(record_path), "record_sha256": sha(record_path), "status": review["status"],
               "reason": review.get("reason")}
    write_new(folder / "model" / f"diagnosis.attempt{len(attempts) + 1}.json", attempt)
    call = call_summary(Path(record_path), record)
    if review["status"] == "infra_error":
        if len(attempts) + 1 < MAX_INFRA_ATTEMPTS:
            return {"status": "retry_allowed", "attempt": len(attempts) + 1}
        return _terminal(tag, "model_infra_failed", {"attempts": len(attempts) + 1, "last": attempt})
    if review["status"] == "call_error":
        return _terminal(tag, "model_infra_failed", {"attempt": attempt})
    if review["status"] == "invalid":
        return _terminal(tag, "model_invalid", {"attempt": attempt})
    decision = review["decision"]
    if decision["action"] != "continue":
        _write_step(tag, "loop_start", previous, commit, ctx.protocol, {"call": call, "decision": decision, "started": None})
        return _terminal(tag, "diagnosis_hold", {"decision": decision})
    ref = read(ROOT / step_record(tag, "reference")["report"]["path"])
    started = V2.start_loop(ctx.cases["FAULT"], decision["ranking"], _check_artifacts(tag, "FAULT")["checkpoint"]["sha256"],
                            {p: v.get("measurement") for p, v in ref["probes"].items()}, decision["probe"])
    record_out = _write_step(tag, "loop_start", previous, commit, ctx.protocol,
                             {"call": call, "decision": decision, "started": started})
    if started["request_id"] is None:
        _terminal(tag, "loop_not_confirmed", {"reason": "상위 가설을 가르는 probe가 없다", "started": started})
        return record_out
    rid, case = started["request_id"], ctx.cases["FAULT"]
    return {**record_out, "next_commands": [
        f"python evals/loop.py approve {case} {rid} --approver <승인자>   # 또는 reject ... --reason <사유>",
        f"python evals/loop.py run {case} {rid} --tag {tag}_loop --metered",
        "다음 제안이 나오면 approve → run을 반복하고, 종료 메시지 뒤 close-loop를 실행한다"]}


def loop_close(tag: str, isaaclab: Path) -> dict:
    ctx = ctx_for(tag)
    previous = _guard(tag, "loop_close")
    protocol, commit = verify_for(tag, isaaclab)
    summary = loop_summary(ctx, tag, _check_artifacts(tag, "FAULT")["checkpoint"]["sha256"])
    confirmed = summary["status"] == "confirmed"
    proceed = confirmed and summary["hypotheses"] == [ctx.mechanism]
    record = _write_step(tag, "loop_close", previous, commit, ctx.protocol, {"loop": summary, "proceed_to_correction": proceed})
    if not proceed:
        _terminal(tag, "wrong_confirmation" if confirmed else "loop_not_confirmed",
                  {"loop_status": summary["status"], "hypotheses": summary["hypotheses"]})
        return record
    gpu = " # GPU, 실행 중 커밋 금지" if SPECIFICITY_GATES.get(ctx.mechanism) else " # 게이트 없음(CPU 기록만)"
    return {**record, "next_commands": [f"python evals/e2e_demo_v3_1.py specificity --tag {tag}{gpu}"]}


def specificity(tag: str, isaaclab: Path, owned=None) -> dict:
    """확정 메커니즘의 보조 probe를 결함 정책에서 재 동결 classify한다. 모두 normal이면 교정으로 간다."""
    ctx = ctx_for(tag)
    previous = _guard(tag, "specificity")
    protocol, commit = verify_for(tag, isaaclab)
    required = protocol["specificity_gate"]["probes"]
    if not required:
        record = _write_step(tag, "specificity", previous, commit, ctx.protocol, {"required": [], "passed": True})
        return {**record, "next_commands": correction_commands(ctx, tag, protocol)}
    checkpoint = _check_artifacts(tag, "FAULT")["checkpoint"]["sha256"]
    out = RESULTS / f"{tag}_spec" / "probes" / f"{ctx.cases['FAULT']}.json"
    if out.exists():
        raise FileExistsError("특이성 게이트 측정이 이미 있다")
    gpu = V2._admit_gpu(protocol)
    argv = [sys.executable, str(ROOT / "evals/run_probes.py"), "--tag", f"{tag}_spec", "--runs", ctx.cases["FAULT"],
            "--probes", *required, "--isaaclab", str(isaaclab)]
    execution, attempt = _execute(tag, "specificity", argv, protocol["watchdogs_s"]["specificity_probes"], owned)
    if not V3.execution_valid(execution) or not out.is_file():
        _gpu_step_failed(tag, "specificity", execution, attempt, out.exists())
    report = read(out)
    if (report.get("name") != ctx.cases["FAULT"] or report.get("checkpoint", {}).get("sha256") != checkpoint
            or report.get("seed") != ctx.seed or set(report.get("probes", {})) != set(required)):
        raise ValueError("특이성 게이트 측정이 결함 체크포인트의 등록 probe와 다르다")
    reference_probes = read(ROOT / step_record(tag, "reference")["report"]["path"])["probes"]
    outcomes = {probe: L.classify(probe, report["probes"][probe].get("measurement") if report["probes"][probe].get("exit") == 0
                                  else None, reference_probes[probe]["measurement"]) for probe in required}
    passed = all(value == "normal" for value in outcomes.values())
    record = _write_step(tag, "specificity", previous, commit, ctx.protocol, {
        "required": required, "outcomes": outcomes, "passed": passed, "gpu_before": gpu, "execution": execution,
        "measurements": {p: report["probes"][p].get("measurement") for p in required},
        "report": {"path": _rel(out), "sha256_lf": sha(out, True)}})
    if not passed:
        _terminal(tag, "confirmation_not_specific", {"outcomes": outcomes})
        return record
    return {**record, "next_commands": correction_commands(ctx, tag, protocol)}


def correction_commands(ctx: Ctx, tag: str, protocol: dict) -> list[str]:
    model = f"bench/private/e2e_demo_v3_1/{tag}/model"
    call = (f"(WSL) evals/correction_model_call.py --evidence {model}/correction.packet.json --output {model}/correction.call1.json --model {MODEL}"
            if protocol["decision_path"] == "single" else
            f"(WSL) evals/nat_decision_call_v4.py --kind correction --packet {model}/correction.packet.json --output {model}/correction.nat1.json")
    record_file = f"{model}/correction.{_record_name(protocol['decision_path'])}1.json"
    return [f"python evals/e2e_demo_v3_1.py correction-evidence --tag {tag}", call,
            f"python evals/e2e_demo_v3_1.py correction-propose --tag {tag} --record {record_file}",
            f"python evals/e2e_demo_v3_1.py approve-correction --tag {tag} --request <rid> --approver <승인자>   "
            "# 또는 reject-correction ... --reason <사유>",
            f"python evals/e2e_demo_v3_1.py correct-run --tag {tag} --request <rid> --correction-record {record_file}",
            f"(실행이 끊겨 고아 요청이 남으면) python evals/e2e_demo_v3_1.py recover-correction --tag {tag} --request <rid> "
            f"--correction-record {record_file}"]


def correction(tag: str, isaaclab: Path, request_id: str, record_path: Path) -> dict:
    import demo_correction_v3_1 as DC
    ctx = ctx_for(tag)
    previous = _guard(tag, "correction")
    protocol, commit = verify_for(tag, isaaclab)
    request = correction_ledger(ctx).requests().get(request_id)
    if request is None or request["state"] != "approved":
        raise ValueError("승인된 교정 요청이 아니다")
    proposal, record = _bound_proposal(request, record_path)
    rule = recommend(ctx.mechanism, read(ROOT / step_record(tag, "evaluate")["reports"]["FAULT"]["path"]))
    if request["correction_id"] != ctx.correction_id or request["recovery_mode"] != rule["mode"]:
        return _terminal(tag, "correction_not_proposed", {
            "request_id": request_id, "reason": "사전등록 교정·회복 방식과 다른 요청",
            "correction_id": request["correction_id"], "recovery_mode": request["recovery_mode"], "rule_mode": rule["mode"]})
    if request["budget_gpu_s"] < CORRECTION_BUDGET_GPU_S:
        raise ValueError(f"교정 요청 예산 {request['budget_gpu_s']} s가 등록 하한 {CORRECTION_BUDGET_GPU_S} s보다 작다. "
                         "승인을 소비하지 않았다. 요청을 reject하고 2400으로 다시 propose한다")
    if DC.fix_folder(tag).exists():
        raise FileExistsError("교정 출력이 이미 있다")
    log_root = isaaclab / "logs/rsl_rl/unitree_go2_flat"
    if any(p.name.endswith(f"_{DC.fix_tag(tag)}_{arm}") for p in log_root.iterdir() for arm in DC.ARMS[rule["mode"]]):
        raise FileExistsError("교정 arm의 SDK 로그 폴더가 이미 있다")
    V2._admit_gpu(protocol)
    receipt = run_correction(ctx, request_id, DC.make_executor(tag, isaaclab), DC.fix_tag(tag))
    return _write_step(tag, "correction", previous, commit, ctx.protocol,
                       _correction_record(tag, request_id, request, proposal, record_path, record, receipt))


def _bound_proposal(request: dict, record_path: Path) -> tuple[dict, dict]:
    proposal = request["evidence"].get("model_proposal")
    record = read(record_path)
    if proposal is None or proposal.get("record_sha256") != sha(record_path, True):
        raise ValueError("교정 요청에 결합된 모델 호출 기록이 아니다")
    return proposal, record


def _correction_record(tag: str, request_id: str, request: dict, proposal: dict, record_path: Path, record: dict,
                       receipt: dict, recovered: bool = False) -> dict:
    import demo_correction_v3_1 as DC
    result_path = DC.fix_folder(tag) / "result.json"
    return {
        "request_id": request_id, "request": {k: request.get(k) for k in ("correction_id", "recovery_mode", "budget_gpu_s")},
        "rule": {k: request["evidence"].get(k) for k in ("rule_mode", "rule", "standstill_ratio")},
        "model_proposal": {k: proposal.get(k) for k in ("accepted", "decision", "agrees_with_rule", "reasons",
                                                         "unmet_checklist", "rule_visible_to_model", "precheck_calls")},
        "call": call_summary(Path(record_path), record), "receipt": receipt,
        "result": read(result_path) if result_path.is_file() else None, "recovered_orphan": recovered}


def recover_correction(tag: str, isaaclab: Path, request_id: str, record_path: Path) -> dict:
    """실행 중 끊겨 결과 없이 남은 교정 요청을 재실행 없이 닫고 correction 단계를 기록한다.

    교정 결과 파일(fix/result.json)이 있으면 그 판정·SHA·비용으로, 없으면 unknown으로 receipt를 쓴다(원장 recover).
    """
    import demo_correction_v3_1 as DC
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
        expected = {"schema": "e2e_correction_result_v3_1", "tag": DC.fix_tag(tag), "correction_request_id": request_id,
                    "recovery_mode": request["recovery_mode"], "source_commit_sha": commit}
        if any(result.get(k) != v for k, v in expected.items()):
            raise ValueError("교정 결과 파일이 이 요청·시연·실험 커밋의 결과가 아니다(복구하지 않았다)")
        gpu_s = sum(v for v in result["costs"].values() if isinstance(v, (int, float)))
        artifact = {"tag": DC.fix_tag(tag), "verdict": result["label"], "result_sha": sha(result_path),
                    "gpu_s": round(gpu_s, 1), "exit_code": 0, "source": _rel(result_path)}
    receipt = led.recover(request_id, artifact)
    return _write_step(tag, "correction", previous, commit, ctx.protocol,
                       _correction_record(tag, request_id, request, proposal, record_path, record, receipt, True))


# --- 판정 ---
def _correction_ledger_state(ctx: Ctx) -> dict:
    led = correction_ledger(ctx)
    return {"requests": led.requests(), "holds": [e for e in led.events() if e["event"] == "held"]}


def conditions(steps: dict, outcome: str) -> dict:
    begin_step, evaluate_step = steps.get("begin"), steps.get("evaluate")
    start, close, fix = steps.get("loop_start"), steps.get("loop_close"), steps.get("correction")
    spec = steps.get("specificity")
    done = [r for r in (close or {}).get("loop", {}).get("requests", []) if r["state"] == "done"]
    call = (start or {}).get("call") or {}
    fix_call = (fix or {}).get("call") or {}
    costs = [step.get("execution", {}).get("parent_wall_s") for name, step in steps.items()
             if name in ("train_NONE", "train_FAULT", "evaluate", "reference")]
    spec_cost_ok = bool(spec) and (not spec["required"]
                                   or isinstance((spec.get("execution") or {}).get("parent_wall_s"), (int, float)))
    fix_costs = ((fix or {}).get("result") or {}).get("costs", {})
    return {
        "protocol_committed": (len({s["protocol_sha256_lf"] for s in steps.values()}) == 1
                               and len({s["source_commit_sha"] for s in steps.values()}) == 1 and bool(steps)),
        "seed_unused": bool(begin_step) and not begin_step["seed_unused"]["hits"],
        "source_failure_present": bool(evaluate_step) and evaluate_step["gate"]["outcome"] == "failure_present",
        "fresh_diagnosis_call": bool(start) and bool(call.get("record_sha256")) and start["decision"]["action"] == "continue",
        "first_probe_from_model": bool(start) and bool((start.get("started") or {}).get("model_probe_used")),
        "probes_approved_consumed_once": bool(done) and all(r.get("approver") and (r.get("artifact") or r["outcome"] == "unknown")
                                                            for r in done),
        "loop_confirmed_mechanism": bool(close) and close["proceed_to_correction"],
        "confirmation_specific": bool(spec) and spec["passed"] is True,
        "correction_call_accepted": bool(fix) and fix["model_proposal"]["accepted"] is True and bool(fix_call.get("record_sha256")),
        "correction_approved_consumed_once": bool(fix) and fix["receipt"].get("event") == "receipt",
        "correction_recovered": outcome == "recovered",
        "costs_recorded": (len(costs) == 4 and all(isinstance(c, (int, float)) for c in costs) and spec_cost_ok
                           and all(isinstance(r.get("gpu_s"), (int, float)) for r in done) and bool(done)
                           and all(isinstance(fix_costs.get(k), (int, float)) for k in ("train_parent_wall_s", "eval_parent_wall_s"))
                           and all(isinstance(c.get(k), (int, float)) for c in (call, fix_call)
                                   for k in ("execution_wall_s", "input_tokens", "output_tokens"))),
    }


def prediction_results(ctx: Ctx, steps: dict) -> dict:
    """사전등록 예측의 적중(True/False). 그 단계까지 가지 못했으면 None. 성공 조건과 별개다."""
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


def public_view(steps: dict) -> dict:
    """공개 결과에 남길 수치·판정. 절대 경로·PID·로그 원문은 넣지 않는다."""
    start, close, fix = steps.get("loop_start"), steps.get("loop_close"), steps.get("correction")
    call_keys = ("path", "input_tokens", "output_tokens", "execution_wall_s", "llm_calls", "precheck_calls", "response_id")
    return {
        "training_parent_wall_s": {arm: (steps.get(f"train_{arm}") or {}).get("execution", {}).get("parent_wall_s")
                                   for arm in ("NONE", "FAULT")},
        "alarm": (steps.get("alarm") or {}).get("results"),
        "evaluation": {k: (steps.get("evaluate") or {}).get(k) for k in ("gate", "reports")},
        "reference_parent_wall_s": (steps.get("reference") or {}).get("execution", {}).get("parent_wall_s"),
        "diagnosis": None if not start else {"decision": start["decision"], "started": start.get("started"),
                                             "call": {k: start["call"].get(k) for k in call_keys}},
        "loop": None if not close else {k: close["loop"][k] for k in ("status", "hypotheses", "observed", "requests",
                                                                       "probe_costs")},
        "specificity": None if not steps.get("specificity") else {
            k: steps["specificity"].get(k) for k in ("required", "outcomes", "passed", "measurements")}
        | {"parent_wall_s": (steps["specificity"].get("execution") or {}).get("parent_wall_s")},
        "correction": None if not fix else {
            "request_id": fix["request_id"], "request": fix["request"], "model_proposal": fix["model_proposal"],
            "call": {k: fix["call"].get(k) for k in call_keys}, "receipt": fix["receipt"],
            "result": {k: (fix.get("result") or {}).get(k) for k in ("label", "verdict", "standstill_ratio", "costs",
                                                                     "recovery_mode", "reference")}},
    }


def summarize(tag: str) -> dict:
    folder = folder_for(tag)
    if (folder / "result.json").exists():
        return read(folder / "result.json")
    ctx = ctx_for(tag)
    protocol = read(ctx.protocol)
    steps = {name: step_record(tag, name) for name in STEPS if step_path(tag, name).is_file()}
    terminal = read(folder / "terminal.json") if (folder / "terminal.json").exists() else None
    ledger = _correction_ledger_state(ctx) if "loop_close" in steps else {"requests": {}, "holds": []}
    outcome = V2.classify(steps, terminal, ledger)
    checks = conditions(steps, outcome)
    success = outcome == "recovered" and all(checks.values())
    unmet = [k for k, v in checks.items() if not v]
    wording = OUTCOMES[outcome] + (" " + protocol["gap_wording"] + ", ".join(unmet) if unmet and outcome != "in_progress" else "")
    result = {"schema": "e2e_demo_result_v3_1", "tag": tag, "variant": ctx.variant, "fault": ctx.fault,
              "mechanism": ctx.mechanism, "seed": ctx.seed, "case": ctx.cases["FAULT"],
              "decision_path": protocol["decision_path"], "outcome": outcome, "demo_success": success,
              "success_conditions": checks, "unmet_conditions": unmet, "wording": wording,
              "common_wording": protocol["common_wording"], "steps_sha256": {k: sha(step_path(tag, k)) for k in steps},
              "terminal": terminal, "protocol_sha256_lf": sha(ctx.protocol, True),
              "source_commit_sha": sorted({s["source_commit_sha"] for s in steps.values()}),
              "predictions": protocol["predictions"], "prediction_results": prediction_results(ctx, steps),
              "approver_identity_attested": False, **public_view(steps)}
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


# --- 하네스 실패(사전등록 범주, v3 폴더에는 사후 기록) ---
def _harness_root(demo: str) -> Path:
    if demo not in HARNESS_DEMOS:
        raise ValueError(f"모르는 시연 판 {demo!r}")
    return PRIVATE if demo == "v3_1" else ROOT / "bench/private/e2e_demo_v3"


def harness_folder(demo: str, tag: str) -> Path:
    root = _harness_root(demo)
    if not isinstance(tag, str) or not re.fullmatch(TAG_RE, tag):
        raise ValueError("invalid demo tag")
    folder = root / tag
    if not folder.is_dir() or folder.is_symlink() or folder.resolve().parent != root.resolve():
        raise ValueError("시연 폴더가 없거나 출력 위치를 벗어난다")
    return folder


def _demo_protocol(demo: str, begin: dict) -> dict:
    """begin 기록에 묶인 그 판의 시연 프로토콜."""
    name = f"e2e_demo_v3_1_{begin.get('variant')}.json" if demo == "v3_1" else f"e2e_demo_v3_{begin.get('variant')}.json"
    path = PROTOCOL_DIR / name
    if not path.is_file() or sha(path, True) != begin.get("protocol_sha256_lf"):
        raise ValueError("시연 프로토콜이 begin 기록과 다르다")
    return read(path)


def _recorded_steps(folder: Path) -> dict[str, Path]:
    steps = {p.name.split("_", 1)[1][:-5]: p for p in sorted((folder / "steps").glob("[0-9][0-9]_*.json"))}
    if "begin" not in steps or set(steps) - set(STEPS):
        raise ValueError("시연 단계 기록이 등록 단계와 다르다")
    return steps


def committed_blob_lf(commit: str, relative: str) -> str | None:
    """커밋에 있는 파일의 LF 정규화 SHA. 커밋 형식이 틀리거나 파일이 없으면 None."""
    if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        return None
    shown = subprocess.run(["git", "-C", str(CODE_ROOT), "show", f"{commit}:{relative}"], capture_output=True)
    return hashlib.sha256(shown.stdout.replace(b"\r\n", b"\n")).hexdigest() if shown.returncode == 0 else None


def _cause(cause: str, commit: str) -> dict:
    """원인 위치 '저장소 상대 경로:줄'. 시연 실험 커밋에 같은 내용으로 있는 파일(실행된 코드)이고 줄이 범위 안이어야 한다."""
    match = re.fullmatch(r"([A-Za-z0-9_./-]+):(\d+)", cause or "")
    if match is None:
        raise ValueError("원인은 '저장소 상대 경로:줄' 형식이어야 한다")
    relative = match[1]
    if relative.startswith(("bench/private/", ".git/")) or "/.git/" in relative:
        raise ValueError(f"원인 파일로 비공개·git 내부 파일을 쓰지 않는다: {relative}")
    target = (CODE_ROOT / relative).resolve()
    if CODE_ROOT.resolve() not in target.parents or not target.is_file():
        raise ValueError(f"원인 파일이 저장소에 없다: {relative}")
    if committed_blob_lf(commit, relative) != sha(target, True):
        raise ValueError(f"원인 파일이 시연 실험 커밋 {commit[:12]}의 내용과 다르다(추적되지 않았거나 바뀌었다): {relative}")
    lines = target.read_text(encoding="utf-8").splitlines()
    line = int(match[2])
    if not 1 <= line <= len(lines):
        raise ValueError(f"원인 줄이 파일 범위 밖이다: {line}")
    return {"file": relative, "line": line, "text": lines[line - 1].strip()[:200], "sha256_lf": sha(target, True),
            "commit": commit}


def tool_commit() -> str:
    """이 기록 도구가 커밋된 상태의 HEAD. 미추적·변경 상태면 거부한다."""
    try:
        return P.verify_committed(ROOT, {"evals/e2e_demo_v3_1.py"})
    except subprocess.CalledProcessError as error:
        raise ValueError(f"git check failed: {' '.join(map(str, error.cmd))[:200]}") from error


def record_harness_failure(demo: str, tag: str, step: str, error: str, cause: str) -> dict:
    """시연 실행기 결함으로 멈춘 시연의 종료 기록(harness_failure)을 한 번 쓴다. 판정 범주를 새로 만들지 않는다."""
    import correct as CR
    folder = harness_folder(demo, tag)
    if (folder / "terminal.json").exists():
        raise FileExistsError("종료 기록이 이미 있다")
    recorded = _recorded_steps(folder)
    if "correction" in recorded:
        raise ValueError("교정 단계가 기록된 시연은 하네스 실패로 닫지 않는다")
    last = list(recorded)[-1]
    allowed = {STEPS[STEPS.index(last) + 1]} | (set(CORRECTION_COMMANDS) if last == "specificity" else set())
    if step not in allowed:
        raise ValueError(f"멈춘 단계는 마지막 기록 단계({last}) 바로 다음 등록 단계·명령이어야 한다: {step!r}")
    if not isinstance(error, str) or not error.strip():
        raise ValueError("오류 문자열이 비었다")
    if ABSOLUTE_PATH.search(error):
        raise ValueError("오류 문자열에 절대 경로를 넣지 않는다(공개 요약에 실린다)")
    begin = read(recorded["begin"])
    protocol = _demo_protocol(demo, begin)
    case = protocol["cases"]["FAULT"]
    ledger_path = CR._dir(case, CR.LOOPS) / "correction_ledger.jsonl"
    events = CorrectionLedger(ledger_path).events()  # 읽기만 한다(사건 순서 검사 포함, 등록 표와 무관)
    if any(e["event"] in ("approved", "rejected", "consumed", "receipt", "held") for e in events):
        raise ValueError("교정 원장에 승인·거절·소비·실행·보류 사건이 있어 하네스 실패로 닫지 않는다(등록 결과 범주로 끝난다)")
    counts = {kind: sum(1 for e in events if e["event"] == kind) for kind in sorted({e["event"] for e in events})}
    loop_state = CR._dir(case, CR.LOOPS) / "state.json"
    detail = {"demo": demo, "tag": tag, "case": case, "step": step, "error": error.strip()[:500],
              "cause": _cause(cause, begin["source_commit_sha"]), "approval_consumed": False,
              "correction_ledger": {"exists": ledger_path.exists(), "events": counts},
              "probe_loop_status": read(loop_state).get("status") if loop_state.is_file() else None,
              "recorded_steps": list(recorded),
              "recorded_by": ("v3.1 도구로 기록(evals/e2e_demo_v3_1.py record-harness-failure)" if demo == "v3_1" else
                              "v3.1 도구로 사후 기록(evals/e2e_demo_v3_1.py record-harness-failure)"),
              "tool_commit": tool_commit(), "tool_sha256_lf": sha(Path(__file__), True),
              "category_preregistered_for_demo": demo == "v3_1"}
    wording = OUTCOMES["harness_failure"] + ("" if demo == "v3_1" else " v3 사전등록에 없던 범주를 v3.1 도구로 사후 기록했다.")
    record = {"outcome": "harness_failure", "at": V2.now(), "detail": detail, "wording": wording}
    write_new(folder / "terminal.json", record)
    return record


def _harness_correction_proposal(folder: Path, decision_path: str) -> dict | None:
    """교정 제안 호출 기록을 packet 등록 표로 다시 검증한 공개 요약. 기록이 없으면 None."""
    import correction_model_call as CM
    model = folder / "model"
    packet_path = model / "correction.packet.json"
    record_path = model / ("correction.call1.json" if decision_path == "single" else "correction.nat1.json")
    if not packet_path.is_file() or not record_path.is_file():
        return None
    packet, record = read(packet_path), read(record_path)
    if decision_path == "single":
        review = CM.review_call_record(record, packet, sha(record_path, True), packet["registry"])
        costs = {k: record.get(k) for k in ("input_tokens", "output_tokens", "execution_wall_s")}
    else:
        review = review_nat_record(record_path, packet_path, "correction", packet["registry"])
        costs = {"input_tokens": record.get("input_tokens"), "output_tokens": record.get("output_tokens"),
                 "execution_wall_s": record.get("wall_s")}
    proposal = review.get("proposal") or {}
    return {"path": decision_path, "packet_digest": CM.packet_digest(packet), "record_sha256": sha(record_path, True),
            "validated_with": "packet registry(시연 등록 표)", "rule_result": packet.get("rule_result"),
            "accepted": review.get("accepted"), "decision": review.get("decision"),
            "correction_id": proposal.get("correction_id"), "recovery_mode": proposal.get("recovery_mode"),
            "agrees_with_rule": review.get("agrees_with_rule"), "unmet_checklist": review.get("unmet_checklist"),
            "reasons": review.get("reasons"), **costs}


def harness_summary(demo: str, tag: str) -> dict:
    """하네스 실패로 닫힌 v3 시연의 부분 공개 요약(진단·고리·특이성·교정 제안). 교정·회복은 판정하지 않는다."""
    if demo != "v3":
        raise ValueError("v3.1 시연은 summarize가 harness_failure를 요약한다(harness-summary는 v3 폴더용)")
    folder = harness_folder(demo, tag)
    terminal = read(folder / "terminal.json") if (folder / "terminal.json").exists() else None
    if terminal is None or terminal.get("outcome") != "harness_failure":
        raise ValueError("하네스 실패 종료 기록이 있는 시연만 요약한다")
    recorded = _recorded_steps(folder)
    steps = {name: read(p) for name, p in recorded.items()}
    protocol = _demo_protocol(demo, steps["begin"])
    ctx = Ctx(variant=protocol["variant"], fault=protocol["fault"], case_id=protocol["case_id"],
              mechanism=protocol["fault_mechanism"], seed=protocol["seed"])
    result = {"schema": "e2e_demo_harness_summary_v3_1", "demo": demo, "tag": tag, "variant": ctx.variant,
              "fault": ctx.fault, "mechanism": ctx.mechanism, "seed": ctx.seed, "case": ctx.cases["FAULT"],
              "decision_path": protocol["decision_path"], "outcome": "harness_failure",
              "category_preregistered_for_demo": demo == "v3_1", "wording": terminal["wording"],
              "common_wording": protocol["common_wording"], "terminal": terminal,
              "steps_sha256": {name: sha(p) for name, p in recorded.items()},
              "protocol_sha256_lf": steps["begin"]["protocol_sha256_lf"],
              "source_commit_sha": sorted({s["source_commit_sha"] for s in steps.values()}),
              "predictions": protocol["predictions"], "prediction_results": prediction_results(ctx, steps),
              "correction_proposal": _harness_correction_proposal(folder, protocol["decision_path"]),
              "correction_judged": False, "approver_identity_attested": False, **public_view(steps)}
    write_new(RESULTS / tag / "harness_failure_summary.json", result)
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="새 seed 결함 종단 시연 v3.1")
    ap.add_argument("command", choices=("prepare", "verify", "begin", "train", "alarm", "evaluate", "reference", "packet",
                                        "start-loop", "close-loop", "specificity", "correction-evidence", "correction-propose",
                                        "correct-run", "approve-correction", "reject-correction", "correction-status",
                                        "recover-correction",
                                        "record-harness-failure", "harness-summary", "summarize", "status"))
    ap.add_argument("--isaaclab", type=Path, default=Path.home() / "IsaacLab")
    ap.add_argument("--variant", choices=tuple(VARIANTS))
    ap.add_argument("--decision-path", choices=DECISION_PATHS, default="single")
    ap.add_argument("--tag")
    ap.add_argument("--arm", choices=("NONE", "FAULT"))
    ap.add_argument("--record", type=Path)
    ap.add_argument("--request")
    ap.add_argument("--correction-record", type=Path)
    ap.add_argument("--budget-gpu-s", type=int, default=CORRECTION_BUDGET_GPU_S)
    ap.add_argument("--approver")
    ap.add_argument("--reason")
    ap.add_argument("--demo", choices=HARNESS_DEMOS, default="v3_1")
    ap.add_argument("--step")
    ap.add_argument("--error")
    ap.add_argument("--cause")
    args = ap.parse_args(argv)
    required = {"prepare": ("variant",), "verify": ("variant",), "begin": ("variant", "tag"), "train": ("tag", "arm"),
                "start-loop": ("tag", "record"), "correction-propose": ("tag", "record"),
                "correct-run": ("tag", "request", "correction_record"),
                "recover-correction": ("tag", "request", "correction_record"),
                "approve-correction": ("tag", "request", "approver"),
                "reject-correction": ("tag", "request", "approver", "reason"),
                "record-harness-failure": ("tag", "step", "error", "cause")}
    for name in required.get(args.command, ("tag",)):
        if getattr(args, name) is None:
            ap.error(f"{args.command}에는 --{name.replace('_', '-')}가 필요하다")
    os.environ.update(V2.UTF8_ENV)  # Isaac 하위 프로세스(cp949 콘솔) 출력 오류 방지
    commands = {
        "prepare": lambda: prepare(args.variant, args.decision_path, args.isaaclab),
        "verify": lambda: {"status": "pass", "source_commit_sha": verify_protocol(
            args.variant, args.isaaclab, experiment_commit(args.tag) if args.tag else None)[1]},
        "begin": lambda: begin(args.tag, args.variant, args.isaaclab),
        "train": lambda: train(args.tag, args.arm, args.isaaclab),
        "alarm": lambda: alarm(args.tag, args.isaaclab),
        "evaluate": lambda: evaluate(args.tag, args.isaaclab),
        "reference": lambda: reference(args.tag, args.isaaclab),
        "packet": lambda: packet(args.tag, args.isaaclab),
        "start-loop": lambda: loop_start(args.tag, args.isaaclab, args.record),
        "close-loop": lambda: loop_close(args.tag, args.isaaclab),
        "specificity": lambda: specificity(args.tag, args.isaaclab),
        "correction-evidence": lambda: correction_evidence(args.tag, args.isaaclab),
        "correction-propose": lambda: correction_propose(args.tag, args.isaaclab, args.record, args.budget_gpu_s),
        "correct-run": lambda: correction(args.tag, args.isaaclab, args.request, args.correction_record),
        "approve-correction": lambda: approve_correction(args.tag, args.request, args.approver),
        "reject-correction": lambda: reject_correction(args.tag, args.request, args.approver, args.reason),
        "correction-status": lambda: correction_status(args.tag),
        "recover-correction": lambda: recover_correction(args.tag, args.isaaclab, args.request, args.correction_record),
        "record-harness-failure": lambda: record_harness_failure(args.demo, args.tag, args.step, args.error, args.cause),
        "harness-summary": lambda: harness_summary(args.demo, args.tag),
        "summarize": lambda: summarize(args.tag),
        "status": lambda: status(args.tag),
    }
    try:
        out = commands[args.command]()
    except GPUBusy as error:
        print(f"대기: {error}. 아무것도 기록하지 않았다", file=sys.stderr)
        return 3
    except (ValueError, FileExistsError, FileNotFoundError, RuntimeError, L.LedgerError) as error:
        print(f"거부: {error}", file=sys.stderr)
        return 2
    print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
