"""새 seed 보상 결함(F1) 종단 시연의 사전등록과 단계 실행기.

학습 실패 입력(NONE·F1 새 학습) → 추종 경고 R1 → 고정 평가 → NONE probe 기준 → Nemotron 첫 판단(새 호출) →
사람 승인 probe 고리 → 확정 시 교정 제안(새 Nemotron 호출) → 사람 승인 → restart 교정 학습·고정 평가 → 결과 문구.
같은 사례 이름(h02_s2029)과 요청 ID로 각 단계를 잇는다. 단계마다 결과 파일을 한 번만 쓰고 멈추며, 다음 단계는
앞 단계 파일의 SHA 사슬과 커밋된 프로토콜을 확인한 뒤에만 진행한다. GPU 단계는 GPU가 비어 있을 때만 시작한다.
이 파일은 모델을 호출하지 않는다. 모델 호출은 WSL에서 기존 스크립트로 하고 기록 파일만 받는다.
결과는 새 seed 하나의 통합 시연이며 진단 정확도·일반화·교정 성공률·GPU 절감의 근거가 아니다.

  python evals/e2e_reward_demo.py prepare|verify [--isaaclab <경로>]
  python evals/e2e_reward_demo.py begin|alarm|evaluate|reference|packet|close-loop|summarize|status --tag <tag>
  python evals/e2e_reward_demo.py train --tag <tag> --arm NONE|F1
  python evals/e2e_reward_demo.py start-loop --tag <tag> --record <active_model_call 기록>
  python evals/e2e_reward_demo.py correct-run --tag <tag> --request <rid> --correction-record <교정 모델 호출 기록>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from copy import deepcopy
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "bench", ROOT / "evals"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import run_training_correction as V3  # noqa: E402
import record_evidence_media as M  # noqa: E402
import run_runtime_readback_smoke as P  # noqa: E402
import runtime_readback_smoke as S  # noqa: E402
from rl_triage import grounded_recovery as C  # noqa: E402
from rl_triage import probe_loop as L  # noqa: E402
from rl_triage import recovery as RC  # noqa: E402
from rl_triage import tracking_alarm as A  # noqa: E402

read = V3.read
write_new = V3.write_new
sha = V3.sha

PROTOCOL = ROOT / "bench/protocols/e2e_reward_demo_v1.json"
PRIVATE = ROOT / "bench/private/e2e_reward_demo"
RESULTS = ROOT / "evals/results"
V4_PROTOCOL = ROOT / "bench/protocols/training_correction_v4.json"
FIXED_EVAL = V3.FIXED_EVAL
BANDS_SOURCE = ROOT / V3.FIXED["bands_source"]
SEED = 2029
RESTART_SEED = SEED + 1000  # 교정 재학습 seed. bench/demo_restart_correction.RESTART_SEED와 같다
CASES = {"NONE": f"baseline_p0c_s{SEED}", "F1": f"h02_s{SEED}"}
CASE_IDS = {"NONE": "baseline_p0c", "F1": "h02"}
RUNS_DIR = "bench/runs_p0c"
REFERENCE_RUNS = tuple(f"baseline_p0c_s{s}" for s in (2026, 2027, 2028))
MODEL = "nvidia/nemotron-3-super-120b-a12b"
STEPS = ("begin", "train_NONE", "train_F1", "alarm", "evaluate", "reference", "packet", "loop_start",
         "loop_close", "correction")
LOOP_TOP_K, LOOP_BUDGET, PROBE_EXECUTION_BUDGET_S = 3, 4, 180
CORRECTION = "restore_body_frame_reward"
CORRECTION_BUDGET_GPU_S = 2400
MAX_INFRA_ATTEMPTS = 2
TAG_RE = r"[a-z0-9][a-z0-9_-]{0,35}"
UTF8_ENV = {"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
CODE = V3.CODE | {
    "evals/e2e_reward_demo.py", "bench/demo_restart_correction.py", "src/rl_triage/tracking_alarm.py",
    "bench/run_case.ps1", "bench/train_with_fault.py", "bench/hidden_faults.py", "bench/external_faults.py",
    "bench/extract_tb.py", "evals/run_fixed_eval.py", "evals/fixed_eval.py", "evals/run_probes.py", "evals/probes.py",
    "evals/metered_probes.py", "evals/loop.py", "src/rl_triage/probe_loop.py", "src/rl_triage/active_diagnosis.py",
    "evals/active_model_call.py", "evals/run_active_diagnosis.py", "evals/correct.py", "evals/correction_model_call.py",
    "src/rl_triage/correction_registry.py", "src/rl_triage/correction_ledger.py", "bench/train_with_recovery_mode.py",
    "bench/train_with_correction.py", "bench/correction_adapter.py", "evals/run_training_correction.py",
    "bench/run_runtime_readback_smoke.py", "bench/runtime_readback_smoke.py", "bench/record_evidence_media.py",
    "src/rl_triage/grounded_recovery.py", "src/rl_triage/recovery.py",
    "bench/reference/nvidia_skills/isaaclab-debugging-rl-training/SKILL.md",
}
COMMON_WORDING = ("새 seed 2029의 합성 보상 결함 1건을 끝까지 연결한 통합 시연이다. 결함과 probe는 같은 설계자가 만들었고, "
                  "진단 정확도·일반화·교정 성공률·GPU 절감의 근거가 아니다.")
OUTCOMES = {
    "invalid": "출처·커밋·seed·산출물 결합 검증이 맞지 않아 시연 판정을 보류한다.",
    "baseline_invalid": "같은 seed 정상 정책이 고정 평가에서 걷지 못해 비교 기준이 없다. 원인 진단을 시작하지 않았다.",
    "source_failure_absent": "결함 학습 정책이 같은 seed 정상 정책 대비 고정 행동 허용 범위를 통과해 진단할 실패가 없었다.",
    "model_infra_failed": "Nemotron 호출이 인프라 오류로 등록 횟수 안에 응답을 주지 못해 진단을 보류했다.",
    "model_invalid": "Nemotron 응답이 공개 판단 계약을 어겨 검사를 제안하지 않고 보류했다.",
    "diagnosis_hold": "Nemotron이 첫 판단에서 근거 부족으로 보류해 검사를 실행하지 않았다.",
    "loop_not_confirmed": "승인된 검사들로 원인을 하나로 확정하지 못해(식별 불가 또는 지원 가설 없음) 교정을 제안하지 않았다.",
    "wrong_confirmation": "검사 고리가 보상이 아닌 다른 메커니즘을 확정했다. 등록 결함과 다른 확정으로 기록하고 교정하지 않았다.",
    "correction_not_proposed": "확정 뒤 교정 단계가 보류(HOLD)했거나 이 시연에 등록되지 않은 회복 방식이라 교정을 실행하지 않았다.",
    "correction_rejected": "사람이 교정 요청을 거절해 교정을 실행하지 않았다.",
    "correction_unknown": "승인된 교정 실행이 실패·예산 초과·검증 불일치로 끝나 회복 여부를 판정하지 않았다.",
    "not_recovered": "보상 결함을 확정하고 승인된 restart 교정까지 실행했지만, 교정 정책이 같은 seed 정상 정책 대비 고정 행동 허용 범위를 통과하지 못했다.",
    "recovered": "새 Nemotron 판단으로 시작한 사람 승인 검사가 보상 결함을 확정했고, 사람이 승인한 restart 교정 정책이 같은 seed 정상 정책 대비 고정 행동 허용 범위를 통과했다.",
    "in_progress": "아직 끝나지 않은 시연이다.",
}
TERMINAL_OUTCOMES = set(OUTCOMES) - {"recovered", "not_recovered", "correction_unknown", "in_progress"}
SUCCESS_CONDITIONS = {
    "protocol_committed": "모든 단계가 실행 전에 커밋된 같은 프로토콜·소스에서 돌았다",
    "seed_unused": "시작 시 seed 2029 학습·평가·고리 산출물이 없었다",
    "source_failure_present": "같은 seed 정상 정책은 걷고 결함 정책은 unhealthy였다",
    "fresh_diagnosis_call": "새 Nemotron 첫 판단 호출이 기록됐고 계약상 유효한 continue였다",
    "first_probe_from_model": "첫 검사 제안이 Nemotron이 고른 probe였다(규칙 대체가 아님)",
    "probes_approved_consumed_once": "모든 실행 probe가 사람 승인 기록 뒤 한 번만 소비됐고 산출물 체크포인트 SHA가 사례와 같았다",
    "loop_confirmed_reward": "동결 규칙 갱신으로 reward가 유일 가설로 확정됐다",
    "correction_call_accepted": "새 교정 제안 Nemotron 호출이 기록됐고 검증을 통과해 교정 요청의 근거로 결합됐다",
    "correction_approved_consumed_once": "사람 승인 기록 뒤 교정 요청이 한 번 소비되고 receipt가 남았다",
    "restart_recovered": "restart 교정 정책이 같은 seed 정상 정책 대비 healthy였다",
    "costs_recorded": "학습·평가·probe 프로세스 시간과 두 모델 호출의 지연·토큰이 모두 기록됐다",
}
FIXED = {
    "schema": "e2e_reward_demo_protocol_v1",
    "question": "새 seed의 보상 결함 학습 실패 하나를 경고·고정 평가·새 Nemotron 판단·사람 승인 검사·교정 제안·사람 승인 교정·고정 평가로 끝까지 연결해 같은 사례/요청 ID로 기록할 수 있는가",
    "scope": "single_new_seed_synthetic_reward_fault_integration_demo_not_accuracy_generalization_or_savings",
    "seed": SEED, "seed_selection": "seed 2026~2028·7·42·123과 개발 실행에 쓰지 않은 다음 값. 시작 시 2029·3029의 산출물·학습 설정 부재를 확인한다",
    "restart_seed": RESTART_SEED,
    "restart_seed_rule": ("교정 재학습 seed = 실패 seed + 1000. NONE 기준·F1 실행과 다른 학습 seed를 사전등록한다. "
                          "같은 seed 처음부터 학습은 결정적이라 정상 학습을 바이트 단위로 재현했다(v4 seed 2026 "
                          "restart_corrected의 model_299.pt = 원본 NONE s2026). 체크포인트 동일성을 기대하지 않으며, "
                          "판정은 서로 다른 seed 기준 쌍에서 만든 p0c 밴드로 같은 seed(2029) NONE 대비 한다"),
    "task": C.TASK, "num_envs": 4096, "max_iterations": 300, "num_steps_per_env": 24, "action_scale": 0.25,
    "cases": CASES, "case_ids": CASE_IDS, "telemetry_dir": RUNS_DIR,
    "fault": "F1", "fault_mechanism": "reward", "fault_definition": "bench/hidden_faults.py F1(동결, 크기·정의 변경 없음)",
    "steps": {
        "begin": "프로토콜·소스 커밋과 seed 미사용 확인(GPU 없음)",
        "train_NONE": "bench/run_case.ps1 -CaseId baseline_p0c -Fault NONE, 4096 env × 300회(GPU)",
        "train_F1": "bench/run_case.ps1 -CaseId h02 -Fault F1, 4096 env × 300회(GPU)",
        "alarm": "R1을 두 실행의 학습 텔레메트리에 적용(CPU). 판정 단계가 아니라 기록",
        "evaluate": "evals/run_fixed_eval.py로 두 정책 고정 평가 후 p0c 밴드로 실패 존재 판정(GPU)",
        "reference": "evals/run_probes.py로 NONE 정책의 6개 probe 기준 측정(GPU)",
        "packet": "active_decision_packet_v2(사례·NONE 텔레메트리 말기 20% 평균, 공개 probe 카탈로그) 작성(CPU)",
        "loop_start": "WSL active_model_call.py 기록을 검증하고 상위 3가설·모델 probe로 고리 시작(CPU)",
        "loop_close": "evals/loop.py approve/reject/run(사람·GPU)이 끝난 뒤 원장·산출물 대조(CPU)",
        "correction": "correct.py evidence→correction_model_call.py(WSL)→propose→approve(사람) 뒤 seed 3029 restart 교정 학습·고정 평가(GPU)",
    },
    "gpu_step_execution": ("bench/run_runtime_readback_smoke.run_owned(Windows Job)로 프로세스 트리를 소유·정리한다. "
                           "학습 실패는 invalid로 종료(seed 재사용 없음). 고정 평가·기준 probe는 결과 파일이 없을 때만 "
                           "인프라 재시도 1회"),
    "alarm": {**A.R1, "reference_runs": list(REFERENCE_RUNS),
              "status": "이미 본 seed 2026~2028에서 사후에 고른 후보 상수. 판정·게이트에 쓰지 않는다"},
    "evaluation": {"protocol": "bench/protocols/fixed_eval_v1.json", "bands_source": V3.FIXED["bands_source"],
                   "failure_rule": "같은 seed NONE이 걷고(err_xy < 0.5×제자리, 낙상 ≤ 0.05) F1이 NONE 대비 unhealthy면 실패 존재",
                   "baseline_walk_rule": "grounded_recovery.WALK_FRACTION·REF_FALL_MAX"},
    "diagnosis": {"model": MODEL, "packet_schema": "active_decision_packet_v2",
                  "decision_contract": "rl_triage.active_diagnosis.validate_decision",
                  "call": "evals/active_model_call.py(WSL, temperature 0)", "max_infra_attempts": MAX_INFRA_ATTEMPTS,
                  "record_binding": ("packet SHA·messages SHA·모델 ID·worker 정리를 대조하고, 기록은 model 폴더의 "
                                     "diagnosis.call<n>.json 순서로만 받는다(제출하지 않은 호출 기록이 있으면 거부)"),
                  "budget_left": LOOP_BUDGET, "execution_budget_s": PROBE_EXECUTION_BUDGET_S,
                  "blind_inputs": "사례 이름·정답·결함 정의를 넣지 않는다"},
    "loop": {"case": CASES["F1"], "top_k": LOOP_TOP_K, "budget_probes": LOOP_BUDGET,
             "first_probe": "Nemotron이 고른 probe가 상위 3가설을 가르면 그 probe, 아니면 동결 규칙 probe(대체로 기록)",
             "update": "동결 probe_loop.update 규칙. 관측 뒤 모델 재호출 없음",
             "request_budget_gpu_s": "probe cost_gpu_s × 2(evals/loop.py)", "result_tag_suffix": "_loop",
             "metered": True, "approval": "사람이 evals/loop.py approve/reject. 승인자 문자열은 신원 인증이 아니다"},
    "correction": {"case": CASES["F1"], "correction_id": CORRECTION, "registered_recovery_mode": "restart",
                   "training_seed": RESTART_SEED,
                   "budget_gpu_s_min": CORRECTION_BUDGET_GPU_S,
                   "budget_note": "교정 요청 예산은 2400 s 이상이어야 한다. 실제 상한은 학습 1800 s·평가 360 s 감시 시간이다",
                   "result_tag_suffix": "_fix",
                   "model_call": "evals/correction_model_call.py(WSL, temperature 0)", "auto_approval": False,
                   "reference": "같은 seed NONE의 시연 고정 평가", "verdict": "recovery.verdict(p0c 밴드)"},
    "gpu_admission_profile": V3.FIXED["gpu_admission_profile"], "isaaclab_commit": V3.FIXED["isaaclab_commit"],
    "rsl_rl_version": V3.FIXED["rsl_rl_version"], "torch_version": V3.FIXED["torch_version"],
    "watchdogs_s": {"train": 1800, "fixed_eval": 600, "reference_probes": 900},
    "budget_estimate": {
        "unit": "부모 프로세스 벽시계(초). GPU 전용 시간이나 절감량이 아니다",
        "train_NONE": [686.5, 802.3], "train_F1": [453.4, 672.3], "fixed_eval_two_runs": [80, 180],
        "reference_six_probes": [80, 180], "loop_probes": [18, 110], "restart_train": [453.4, 802.3],
        "restart_eval": [46.2, 89.5], "total": [1817.5, 2836.4],
        "sources": {
            "train": "P0-C bench/private/*_s202{6,7,8}.meta.json wall_time_s(run_case.ps1, Isaac 기동·TB 추출 포함)",
            "fixed_eval": "P0-C 일괄 평가 작업당 평균 33.5 s·최대 43.4 s와 기동 약 11 s, 단일 평가 프로세스 parent wall 46.2~89.5 s",
            "reference": "p1a NONE probe 내부 elapsed 합 약 64 s와 단일 probe 실행 wall 18~27 s",
            "loop": "evals/loops receipt gpu_s P_reward 18.3~19.2, P_noise 17.9~21.2, P_torque 26.6",
        },
        "model_calls": "진단 1회·교정 1회, 각 인프라 오류 재시도 최대 1회",
    },
    "predictions": {
        "source_failure": "F1이 같은 seed NONE 대비 unhealthy이고 standstill_ratio ≥ 0.9",
        "alarm": "R1이 F1에 울리고 NONE에는 울리지 않는다(첫 경고 iteration은 예측하지 않는다)",
        "diagnosis": "첫 판단 action=continue, 상위 3가설에 reward 포함",
        "loop": "P_reward abnormal 뒤 reward 확정(예산 4 이내)",
        "correction_rule": "reward·standstill_ratio ≥ 0.9 → restart",
        "restart": "seed 3029 restart 교정 정책 recovered",
        "basis": ("seed 2026~2028 P0-C·v3·v4 개발 관찰. v4 seed 2026 restart의 회복은 같은 seed 재현이라 자명했고, "
                  "다른 seed 재학습의 회복은 이번에 처음 판정한다. 예측 적중은 성공 조건과 따로 기록한다"),
    },
    "success_conditions": SUCCESS_CONDITIONS,
    "success_rule": "결과가 recovered이고 성공 조건이 모두 참일 때만 demo_success",
    "outcomes": OUTCOMES, "common_wording": COMMON_WORDING,
    "gap_wording": "다음 성공 조건을 충족하지 못했다: ",
    "semantic_retries": 0,
    "forbidden_claims": ["진단 정확도·우월성", "일반화", "교정 성공률", "GPU 절감", "이미 본 seed 결과와 합친 분모"],
}
DYNAMIC = {"source_files", "reference_telemetry", "bands", "bands_summary_sha256_lf", "fixed_eval_sha256_lf",
           "v4_protocol_sha256_lf", "correction_source_contract"}


class GPUBusy(RuntimeError):
    """GPU가 사용 중이라 아무것도 기록하지 않고 멈췄다."""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _rel(path: Path) -> str:
    return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()


# --- 프로토콜 ---
def dynamic_fields(isaaclab: Path) -> dict:
    v4 = read(V4_PROTOCOL)
    contract = v4["correction_source_contract"]
    V3.check_contract(contract, isaaclab)
    bands = read(BANDS_SOURCE)["bands"]
    return {
        "source_files": {name: sha(ROOT / name, True) for name in sorted(CODE)},
        "reference_telemetry": {name: sha(ROOT / RUNS_DIR / f"{name}.telemetry.json", True) for name in REFERENCE_RUNS},
        "bands": bands, "bands_summary_sha256_lf": sha(BANDS_SOURCE, True),
        "fixed_eval_sha256_lf": sha(FIXED_EVAL, True), "v4_protocol_sha256_lf": sha(V4_PROTOCOL, True),
        "correction_source_contract": contract,
    }


def check_protocol(protocol: dict, dynamic: dict) -> None:
    if set(protocol) != set(FIXED) | DYNAMIC:
        raise ValueError("protocol fields differ")
    if any(protocol[k] != v for k, v in FIXED.items()):
        raise ValueError("fixed demo contract changed")
    changed = sorted(k for k, v in dynamic.items() if protocol[k] != v)
    if changed:
        raise ValueError(f"bound sources changed: {changed}")


def prepare(isaaclab: Path) -> dict:
    write_new(PROTOCOL, {**deepcopy(FIXED), **dynamic_fields(isaaclab)})
    return {"status": "prepared_commit_before_begin", "path": _rel(PROTOCOL)}


def verify_protocol(isaaclab: Path) -> tuple[dict, str]:
    protocol = read(PROTOCOL)
    check_protocol(protocol, dynamic_fields(isaaclab))
    if S._git_head(isaaclab) != FIXED["isaaclab_commit"] or not S._git_tracked_clean(isaaclab):
        raise ValueError("Isaac Lab checkout differs")
    bound = CODE | {_rel(PROTOCOL), _rel(FIXED_EVAL), _rel(BANDS_SOURCE), _rel(V4_PROTOCOL),
                    *(f"{RUNS_DIR}/{name}.telemetry.json" for name in REFERENCE_RUNS)}
    return protocol, P.verify_committed(ROOT, bound)


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


def _guard(tag: str, name: str) -> Path:
    """앞 단계가 모두 있고 SHA 사슬이 이어지며 이 단계와 종료 기록이 없을 때만 진행한다."""
    folder = folder_for(tag)
    if (folder / "terminal.json").exists():
        raise ValueError(f"시연이 이미 종료됐다: {read(folder / 'terminal.json')['outcome']}")
    if step_path(tag, name).exists():
        raise FileExistsError(f"{name} 단계는 이미 기록됐다(한 번만 쓴다)")
    index = STEPS.index(name)
    previous = None
    for earlier in STEPS[:index]:
        path = step_path(tag, earlier)
        if not path.is_file():
            raise ValueError(f"{earlier} 단계가 먼저다")
        if read(path).get("previous_sha256") != (None if previous is None else sha(previous)):
            raise ValueError(f"{earlier} 단계의 SHA 사슬이 끊겼다")
        previous = path
    return previous


def _write_step(tag: str, name: str, previous: Path | None, commit: str, record: dict) -> dict:
    full = {"step": name, "at": now(), "previous_sha256": None if previous is None else sha(previous),
            "source_commit_sha": commit, "protocol_sha256_lf": sha(PROTOCOL, True), **record}
    write_new(step_path(tag, name), full)
    return full


def _terminal(tag: str, outcome: str, detail: dict) -> dict:
    if outcome not in TERMINAL_OUTCOMES:
        raise ValueError(f"등록되지 않은 조기 종료 {outcome}")
    record = {"outcome": outcome, "at": now(), "detail": detail, "wording": OUTCOMES[outcome]}
    write_new(folder_for(tag) / "terminal.json", record)
    return record


def _admit_gpu(protocol: dict) -> dict:
    gpu = M.query_gpu(protocol["gpu_admission_profile"])
    if gpu["busy"]:
        raise GPUBusy(f"GPU 사용 중: {gpu.get('busy_reasons')}")
    return gpu


def _execute(tag: str, step: str, argv: list[str], watchdog_s: float, owned=None) -> tuple[dict, int]:
    """Windows Job으로 프로세스 트리를 소유해 실행한다. (실행 기록, 시도 번호). 시도 폴더는 한 번만 쓴다."""
    owned = owned or P.run_owned
    if any(char in arg for arg in argv for char in '\r\n"%&^'):
        raise ValueError("unsafe Windows batch argument")
    exec_root = folder_for(tag) / "exec"
    attempt = 1 + sum(1 for p in exec_root.glob(f"{step}_attempt*") if p.is_dir()) if exec_root.exists() else 1
    folder = exec_root / f"{step}_attempt{attempt}"
    folder.mkdir(parents=True)
    execution = owned(folder, argv, ROOT, watchdog_s)
    execution = {**execution, "parent_wall_s": round(execution["wall_s"], 1), "attempt": attempt,
                 "folder": _rel(folder)}
    write_new(folder / "execution.json", execution)
    return execution, attempt


def _gpu_step_failed(tag: str, step: str, execution: dict, attempt: int, output_exists: bool) -> None:
    """GPU 단계 실패 처리. 학습은 invalid로 종료하고, 평가·기준은 결과 파일이 없을 때 인프라 재시도 1회."""
    detail = {"step": step, "attempt": attempt, "execution": {k: execution.get(k) for k in
                                                            ("exit_code", "natural_exit", "timed_out", "error_type",
                                                             "cleanup_verified", "folder")}}
    if step in ("evaluate", "reference") and attempt == 1 and not output_exists and execution.get("cleanup_verified"):
        raise RuntimeError(f"{step} 인프라 실패, 결과 파일이 없어 1회 재시도할 수 있다: {detail}")
    _terminal(tag, "invalid", detail)
    raise RuntimeError(f"{step} 실패로 시연을 invalid로 종료했다: {detail}")


# --- 입력 확인 ---
def _logged_seeds(log_root: Path) -> dict[str, int]:
    seeds = {}
    for agent in log_root.glob("*/params/agent.yaml") if log_root.exists() else ():
        match = re.search(r"^seed: (\d+)\s*$", agent.read_text(encoding="utf-8", errors="replace"), re.M)
        if match:
            seeds[agent.parent.parent.name] = int(match[1])
    return seeds


def seed_unused_evidence(isaaclab: Path, tag: str) -> dict:
    log_root = isaaclab / "logs/rsl_rl/unitree_go2_flat"
    roots = [ROOT / RUNS_DIR, ROOT / "bench/private", ROOT / "evals/loops", log_root]
    names = (f"_s{SEED}", f"_s{RESTART_SEED}")
    hits = sorted(str(p) for root in roots if root.exists() for p in root.iterdir() if any(n in p.name for n in names))
    hits += sorted(f"{run}: seed {seed}" for run, seed in _logged_seeds(log_root).items() if seed in (SEED, RESTART_SEED))
    outputs = [str(p) for p in (folder_for(tag), RESULTS / tag, RESULTS / f"{tag}_ref", RESULTS / f"{tag}_loop")
               if p.exists()]
    return {"checked_roots": [str(r) for r in roots], "checked_seeds": [SEED, RESTART_SEED], "hits": hits,
            "existing_outputs": outputs}


def training_artifacts(arm: str, isaaclab: Path) -> dict:
    name = CASES[arm]
    meta_path = ROOT / "bench/private" / f"{name}.meta.json"
    meta = read(meta_path)
    expected = {"task": C.TASK, "seed": SEED, "num_envs": FIXED["num_envs"],
                "max_iterations": FIXED["max_iterations"], "fault": arm, "train_exit_code": 0}
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
    if not isinstance(series, list) or len(series) != FIXED["max_iterations"]:
        raise ValueError(f"{name}: telemetry does not cover 300 iterations")
    if V3.load_config(files["agent"]).get("seed") != SEED:
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


def walks(report: dict) -> bool:
    behavior = C._behavior(report)
    return (behavior["err_xy"] < C.WALK_FRACTION * report["standstill"]["lin_vel_rmse_mps"]
            and behavior["fall_frac"] <= C.REF_FALL_MAX)


def source_gate(none_report: dict, fault_report: dict, bands: dict) -> dict:
    verdict = RC.verdict(RC.compare(C._behavior(fault_report), C._behavior(none_report)), bands)
    none_walks = walks(none_report)
    outcome = ("baseline_invalid" if not none_walks else
               "failure_present" if verdict["label"] == "unhealthy" else "source_failure_absent")
    ratio = {arm: r["metrics"]["lin_vel_rmse_mps"] / r["standstill"]["lin_vel_rmse_mps"]
             for arm, r in (("NONE", none_report), ("F1", fault_report))}
    return {"none_walks": none_walks, "fault_verdict": verdict, "standstill_ratio": ratio, "outcome": outcome}


# --- 진단·고리 ---
def review_diagnosis_record(record: dict, packet_path: Path) -> dict:
    """active_model_call_v1 기록을 packet에 묶어 판단 계약으로 검증한다. 상태: valid|infra_error|call_error|invalid."""
    from rl_triage import active_diagnosis as D
    data = Path(packet_path).read_bytes()
    if not isinstance(record, dict) or record.get("schema") != "active_model_call_v1":
        raise ValueError("active_model_call_v1 기록이 아니다")
    if record.get("packet_sha256") != hashlib.sha256(data).hexdigest():
        raise ValueError("호출 기록의 packet이 시연 packet과 다르다")
    if record.get("messages_sha256") != L.digest(D.decision_messages(json.loads(data))):
        raise ValueError("호출 기록의 프롬프트가 커밋된 판단 계약과 다르다")
    if (record.get("request") or {}).get("model") != MODEL or (record.get("request") or {}).get("temperature") != 0:
        raise ValueError("등록 모델·temperature 0 호출이 아니다")
    if record.get("worker_cleanup_failed"):
        raise ValueError("모델 worker 정리 실패 기록이다")
    if record.get("infra_error"):
        return {"status": "infra_error", "reason": str(record["infra_error"])[:200]}
    if record.get("error_type") or record.get("exit_code") != 0:
        return {"status": "call_error", "reason": str(record.get("error_type"))[:200]}
    text = record.get("content") or ""
    if text.startswith("```json") and text.rstrip().endswith("```"):
        text = text[7:text.rfind("```")].strip()
    try:
        return {"status": "valid", "decision": D.validate_decision(json.loads(text), json.loads(data))}
    except (json.JSONDecodeError, ValueError, TypeError) as error:
        return {"status": "invalid", "reason": str(error)[:300]}


def start_loop(case: str, ranking: list[str], checkpoint_sha: str, reference: dict, model_probe: str | None) -> dict:
    """evals/loop.py start와 같은 prereg·reference·state를 쓰되, 첫 제안은 모델 probe가 상위 가설을 가를 때 그 probe."""
    import loop
    d = loop._dir(case)
    if d.exists():
        raise FileExistsError(f"evals/loops/{case}가 이미 있다")
    if not ranking or any(m not in L.MECHANISMS for m in ranking):
        raise ValueError("ranking에 모르는 범주가 있다")
    hyps = ranking[:LOOP_TOP_K]
    prereg = {"case": case, "hypotheses": hyps, "checkpoint_sha256": checkpoint_sha,
              "expectations": {p: {h: L.PROBES[p]["expect"].get(h) for h in hyps} for p in L.PROBES},
              "thresholds": {p: list(v) for p, v in L.THRESHOLDS.items()}, "budget_probes": LOOP_BUDGET}
    d.mkdir(parents=True)
    (d / "prereg.json").write_text(json.dumps(prereg, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (d / "reference.json").write_text(json.dumps(reference, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    state = {"hypotheses": hyps, "observed": {}, "skipped": [], "budget": LOOP_BUDGET, "status": "open",
             "pending": None, "history": []}
    rule_probe = L.next_probe(hyps, {})
    usable = model_probe in L.PROBES and (L.discriminates(model_probe, hyps) or (
        len(hyps) == 1 and model_probe == L.confirming_probe(hyps[0])))
    probe = model_probe if usable else rule_probe
    rid = None
    if probe is None:
        state["status"] = "unidentifiable"
    else:
        rid = L.Ledger(d / "ledger.jsonl").propose(prereg, probe, budget_gpu_s=L.PROBES[probe]["cost_gpu_s"] * 2)["request_id"]
        state["pending"] = rid
    loop._save_state(case, state)
    return {"hypotheses": hyps, "model_probe": model_probe, "rule_probe": rule_probe, "proposed_probe": probe,
            "model_probe_used": bool(usable), "request_id": rid, "prereg_digest": L.digest(prereg)}


def loop_summary(tag: str, checkpoint_sha: str) -> dict:
    """끝난 고리의 원장·상태를 읽고 실행된 probe마다 산출물의 사례·체크포인트 SHA를 대조한다."""
    import loop
    prereg, state, led = loop._load(CASES["F1"])
    if state["status"] == "open" or state.get("pending") or led.orphans():
        raise ValueError("고리가 아직 끝나지 않았다(대기·실행 중 요청이 있다)")
    if prereg["checkpoint_sha256"] != checkpoint_sha:
        raise ValueError("고리 사전등록의 체크포인트가 시연 F1과 다르다")
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
            probe_file = RESULTS / f"{tag}_loop" / "probes" / f"{CASES['F1']}__{req['probe']}.json"
            meta_file = RESULTS / f"{tag}_loop" / "execution" / f"{CASES['F1']}__{req['probe']}.json"
            if receipt["outcome"] != "unknown":
                report, meta = read(probe_file), read(meta_file)
                if (report.get("name") != CASES["F1"] or report.get("checkpoint", {}).get("sha256") != checkpoint_sha
                        or meta.get("checkpoint_sha256") != checkpoint_sha
                        or meta.get("report_sha256_lf") != sha(probe_file, True)):
                    raise ValueError(f"{req['probe']}: probe 산출물이 사례·체크포인트와 다르다")
                if meta.get("metered") is not True:
                    raise ValueError(f"{req['probe']}: 등록과 달리 --metered 없이 실행됐다")
                row["artifact"] = {"probe": _rel(probe_file), "execution": _rel(meta_file)}
                costs.append({k: meta.get(k) for k in ("probe", "gpu_wall_s", "cpu_s", "simulator_steps")} | {"probe": req["probe"]})
        rows.append(row)
    return {"status": state["status"], "hypotheses": state["hypotheses"], "observed": state["observed"],
            "history": state["history"], "requests": rows, "probe_costs": costs}


def correction_approval(request_id: str) -> dict:
    """교정 원장에서 그 요청의 승인 사건(승인자·시각). demo_restart_correction이 기록에 결합한다."""
    import correct as CR
    events = CR._ledger(CASES["F1"], CR.LOOPS).events()
    approved = [e for e in events if e.get("event") == "approved" and e.get("request_id") == request_id]
    if len(approved) != 1:
        raise ValueError("교정 요청의 사람 승인 사건이 하나가 아니다")
    return {"approver": approved[0]["approver"], "at": approved[0]["at"]}


# --- 단계 ---
def begin(tag: str, isaaclab: Path) -> dict:
    previous = _guard(tag, "begin")
    protocol, commit = verify_protocol(isaaclab)
    evidence = seed_unused_evidence(isaaclab, tag)
    if evidence["hits"] or evidence["existing_outputs"]:
        raise ValueError(f"seed {SEED}/{RESTART_SEED} 또는 tag 산출물이 이미 있다: {evidence}")
    return _write_step(tag, "begin", previous, commit, {"tag": tag, "isaaclab_root": str(isaaclab.resolve()),
                                                       "seed_unused": evidence})


def train(tag: str, arm: str, isaaclab: Path, owned=None) -> dict:
    previous = _guard(tag, f"train_{arm}")
    protocol, commit = verify_protocol(isaaclab)
    gpu = _admit_gpu(protocol)
    argv = ["pwsh", "-NoProfile", "-File", str(ROOT / "bench/run_case.ps1"), "-CaseId", CASE_IDS[arm],
            "-Seed", str(SEED), "-NumEnvs", str(FIXED["num_envs"]), "-MaxIterations", str(FIXED["max_iterations"]),
            "-Fault", arm, "-OutDir", str(ROOT / RUNS_DIR), "-IsaacLab", str(isaaclab)]
    execution, attempt = _execute(tag, f"train_{arm}", argv, FIXED["watchdogs_s"]["train"], owned)
    if not V3.execution_valid(execution):
        _gpu_step_failed(tag, f"train_{arm}", execution, attempt, True)
    return _write_step(tag, f"train_{arm}", previous, commit, {"arm": arm, "gpu_before": gpu, "execution": execution,
                                                             "artifacts": training_artifacts(arm, isaaclab)})


def alarm(tag: str, isaaclab: Path) -> dict:
    """R1을 기록한다. 판정·게이트가 아니므로 계산 오류도 기록하고 다음 단계로 간다."""
    previous = _guard(tag, "alarm")
    protocol, commit = verify_protocol(isaaclab)
    refs = [read(ROOT / RUNS_DIR / f"{name}.telemetry.json")["series"][A.METRIC] for name in REFERENCE_RUNS]
    results = {}
    for arm in ("NONE", "F1"):
        artifacts = _check_artifacts(tag, arm)
        try:
            results[arm] = A.r1(read(ROOT / artifacts["telemetry"]["path"])["series"][A.METRIC], refs)
        except ValueError as error:
            results[arm] = {"error": str(error)[:200]}
    return _write_step(tag, "alarm", previous, commit, {"rule": A.R1, "reference_runs": list(REFERENCE_RUNS),
                                                       "results": results})


def evaluate(tag: str, isaaclab: Path, owned=None) -> dict:
    previous = _guard(tag, "evaluate")
    protocol, commit = verify_protocol(isaaclab)
    artifacts = {arm: _check_artifacts(tag, arm) for arm in ("NONE", "F1")}
    gpu = _admit_gpu(protocol)
    argv = [sys.executable, str(ROOT / "evals/run_fixed_eval.py"), "--tag", tag, "--runs", CASES["NONE"], CASES["F1"],
            "--isaaclab", str(isaaclab)]
    execution, attempt = _execute(tag, "evaluate", argv, FIXED["watchdogs_s"]["fixed_eval"], owned)
    outputs = [RESULTS / tag / "runs" / f"{CASES[arm]}.json" for arm in ("NONE", "F1")]
    if not V3.execution_valid(execution) or not all(p.is_file() for p in outputs):
        _gpu_step_failed(tag, "evaluate", execution, attempt, any(p.exists() for p in outputs))
    reports, paths = {}, {}
    for arm in ("NONE", "F1"):
        path = RESULTS / tag / "runs" / f"{CASES[arm]}.json"
        reports[arm] = C.validate_fixed_report(read(path), {
            "name": CASES[arm], "checkpoint_file": "model_299.pt", "checkpoint_sha256": artifacts[arm]["checkpoint"]["sha256"],
            "action_scale": FIXED["action_scale"], "protocol_sha256_lf": protocol["fixed_eval_sha256_lf"]})
        paths[arm] = {"path": _rel(path), "sha256_lf": sha(path, True)}
    versions = reports["NONE"]["versions"]
    if (versions != reports["F1"]["versions"] or versions["torch"] != FIXED["torch_version"]
            or versions["rsl-rl-lib"] != FIXED["rsl_rl_version"]):
        raise ValueError("fixed evaluation runtimes differ")
    gate = source_gate(reports["NONE"], reports["F1"], protocol["bands"])
    record = _write_step(tag, "evaluate", previous, commit, {"gpu_before": gpu, "execution": execution,
                                                            "reports": paths, "gate": gate})
    if gate["outcome"] != "failure_present":
        _terminal(tag, gate["outcome"], {"gate": gate})
    return record


def reference(tag: str, isaaclab: Path, owned=None) -> dict:
    previous = _guard(tag, "reference")
    protocol, commit = verify_protocol(isaaclab)
    checkpoint = _check_artifacts(tag, "NONE")["checkpoint"]["sha256"]
    out = RESULTS / f"{tag}_ref" / "probes" / f"{CASES['NONE']}.json"
    if out.exists():
        raise FileExistsError("NONE probe 기준이 이미 있다")
    gpu = _admit_gpu(protocol)
    argv = [sys.executable, str(ROOT / "evals/run_probes.py"), "--tag", f"{tag}_ref", "--runs", CASES["NONE"],
            "--isaaclab", str(isaaclab)]
    execution, attempt = _execute(tag, "reference", argv, FIXED["watchdogs_s"]["reference_probes"], owned)
    if not V3.execution_valid(execution) or not out.is_file():
        _gpu_step_failed(tag, "reference", execution, attempt, out.exists())
    report = read(out)
    probes = report.get("probes", {})
    if (report.get("name") != CASES["NONE"] or report.get("checkpoint", {}).get("sha256") != checkpoint
            or report.get("seed") != SEED or set(probes) != set(L.PROBES)
            or any(v.get("exit") != 0 or v.get("measurement") is None for v in probes.values())):
        raise ValueError("NONE probe 기준이 여섯 probe를 같은 체크포인트로 모두 재지 못했다")
    return _write_step(tag, "reference", previous, commit, {"gpu_before": gpu, "execution": execution,
                                                           "report": {"path": _rel(out), "sha256_lf": sha(out, True)}})


def packet(tag: str, isaaclab: Path) -> dict:
    import run_active_diagnosis as RA
    previous = _guard(tag, "packet")
    _, commit = verify_protocol(isaaclab)
    sources = {"telemetry": _check_artifacts(tag, "F1")["telemetry"],
               "reference_telemetry": _check_artifacts(tag, "NONE")["telemetry"]}
    built = RA.input_packet({"sources": sources}, {"max_probes": LOOP_BUDGET,
                                                   "probe_execution_budget_s": PROBE_EXECUTION_BUDGET_S})
    path = folder_for(tag) / "model" / "diagnosis.packet.json"
    write_new(path, built)
    return _write_step(tag, "packet", previous, commit, {"packet": {"path": _rel(path),
                                                                   "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}})


def loop_start(tag: str, isaaclab: Path, record_path: Path) -> dict:
    previous = _guard(tag, "loop_start")
    _, commit = verify_protocol(isaaclab)
    folder = folder_for(tag)
    model_dir = (folder / "model").resolve()
    attempts = sorted((folder / "model").glob("diagnosis.attempt*.json"))
    expected = model_dir / f"diagnosis.call{len(attempts) + 1}.json"
    if Path(record_path).resolve() != expected:
        raise ValueError(f"다음 진단 호출 기록은 {expected.name}이어야 한다(model 폴더, 순서대로)")
    later = [p.name for p in model_dir.glob("diagnosis.call*.json")
             if re.fullmatch(r"diagnosis\.call(\d+)\.json", p.name) and int(p.name[14:-5]) > len(attempts) + 1]
    if later:
        raise ValueError(f"제출하지 않은 호출 기록이 있다: {later}")
    stray = []
    for path in model_dir.glob("*.json"):
        if re.fullmatch(r"diagnosis\.(call\d+|attempt\d+|packet)\.json", path.name):
            continue
        try:
            if read(path).get("schema") == "active_model_call_v1":
                stray.append(path.name)
        except (ValueError, OSError, AttributeError):
            continue
    if stray:
        raise ValueError(f"순서 규칙 밖의 진단 호출 기록이 있다: {stray}")
    packet_info = step_record(tag, "packet")["packet"]
    packet_path = ROOT / packet_info["path"]
    if hashlib.sha256(packet_path.read_bytes()).hexdigest() != packet_info["sha256"]:
        raise ValueError("진단 packet이 바뀌었다")
    record = read(record_path)
    review = review_diagnosis_record(record, packet_path)
    attempt = {"record": _rel(record_path), "record_sha256": sha(record_path), "status": review["status"],
               "reason": review.get("reason")}
    write_new(folder / "model" / f"diagnosis.attempt{len(attempts) + 1}.json", attempt)
    call = {"record": attempt["record"], "record_sha256": attempt["record_sha256"],
            **{k: record.get(k) for k in ("response_id", "provider_model", "input_tokens", "output_tokens",
                                          "total_tokens", "execution_wall_s", "finish_reason")}}
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
        _write_step(tag, "loop_start", previous, commit, {"call": call, "decision": decision, "started": None})
        return _terminal(tag, "diagnosis_hold", {"decision": decision})
    ref = read(ROOT / step_record(tag, "reference")["report"]["path"])
    started = start_loop(CASES["F1"], decision["ranking"], _check_artifacts(tag, "F1")["checkpoint"]["sha256"],
                         {p: v.get("measurement") for p, v in ref["probes"].items()}, decision["probe"])
    record_out = _write_step(tag, "loop_start", previous, commit, {"call": call, "decision": decision, "started": started})
    if started["request_id"] is None:
        _terminal(tag, "loop_not_confirmed", {"reason": "상위 가설을 가르는 probe가 없다", "started": started})
        return record_out
    rid, case = started["request_id"], CASES["F1"]
    return {**record_out, "next_commands": [
        f"python evals/loop.py approve {case} {rid} --approver <승인자>   # 또는 reject ... --reason <사유>",
        f"python evals/loop.py run {case} {rid} --tag {tag}_loop --metered",
        "다음 제안이 나오면 approve → run을 반복하고, 종료 메시지 뒤 close-loop를 실행한다"]}


def loop_close(tag: str, isaaclab: Path) -> dict:
    previous = _guard(tag, "loop_close")
    _, commit = verify_protocol(isaaclab)
    summary = loop_summary(tag, _check_artifacts(tag, "F1")["checkpoint"]["sha256"])
    confirmed = summary["status"] == "confirmed"
    proceed = confirmed and summary["hypotheses"] == ["reward"]
    record = _write_step(tag, "loop_close", previous, commit, {"loop": summary, "proceed_to_correction": proceed})
    if not proceed:
        _terminal(tag, "wrong_confirmation" if confirmed else "loop_not_confirmed", {"loop_status": summary["status"],
                                                                                   "hypotheses": summary["hypotheses"]})
        return record
    case, model = CASES["F1"], f"bench/private/e2e_reward_demo/{tag}/model"
    source = f"evals/results/{tag}/runs/{case}.json"
    return {**record, "next_commands": [
        f"python evals/correct.py evidence {case} --source-report {source} --output {model}/correction.packet.json",
        f"(WSL) evals/correction_model_call.py --evidence {model}/correction.packet.json --output {model}/correction.call1.json --model {MODEL}",
        f"python evals/correct.py propose {case} --source-report {source} --budget-gpu-s {CORRECTION_BUDGET_GPU_S} --model-proposal {model}/correction.call1.json",
        f"python evals/correct.py approve {case} <rid> --approver <승인자>   # 또는 reject",
        f"python evals/e2e_reward_demo.py correct-run --tag {tag} --request <rid> --correction-record {model}/correction.call1.json"]}


def correction(tag: str, isaaclab: Path, request_id: str, record_path: Path) -> dict:
    import correct as CR
    import correction_model_call as CM
    import demo_restart_correction as DR
    previous = _guard(tag, "correction")
    protocol, commit = verify_protocol(isaaclab)
    request = CR._ledger(CASES["F1"], CR.LOOPS).requests().get(request_id)
    if request is None or request["state"] != "approved":
        raise ValueError("사람이 승인한 교정 요청이 아니다")
    proposal = request["evidence"].get("model_proposal")
    call_record = read(record_path)
    record_sha = hashlib.sha256(Path(record_path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    if proposal is None or proposal.get("record_sha256") != record_sha or call_record.get("schema") != CM.RECORD_SCHEMA:
        raise ValueError("교정 요청에 결합된 모델 호출 기록이 아니다")
    if request["correction_id"] != CORRECTION or request["recovery_mode"] != FIXED["correction"]["registered_recovery_mode"]:
        return _terminal(tag, "correction_not_proposed", {
            "request_id": request_id, "reason": "이 시연에 등록되지 않은 교정 또는 회복 방식",
            "correction_id": request["correction_id"], "recovery_mode": request["recovery_mode"]})
    if request["budget_gpu_s"] < CORRECTION_BUDGET_GPU_S:
        raise ValueError(f"교정 요청 예산 {request['budget_gpu_s']} s가 등록 하한 {CORRECTION_BUDGET_GPU_S} s보다 작다. "
                         "승인을 소비하지 않았다. 요청을 reject하고 --budget-gpu-s 2400으로 다시 propose한다")
    # 승인 소비 전에 실행 전제를 확인해 사전 조건 실패로 승인이 소진되지 않게 한다.
    if DR.fix_folder(tag).exists():
        raise FileExistsError("교정 arm 출력이 이미 있다")
    log_root = isaaclab / "logs/rsl_rl/unitree_go2_flat"
    if any(p.name.endswith(f"_{DR.fix_tag(tag)}_{DR.ARM}") for p in log_root.iterdir()):
        raise FileExistsError("교정 arm의 SDK 로그 폴더가 이미 있다")
    _admit_gpu(protocol)
    receipt = CR.run(CASES["F1"], request_id, DR.make_executor(tag, isaaclab), loops=CR.LOOPS, tag=DR.fix_tag(tag))
    result_path = DR.fix_folder(tag) / "result.json"
    call = {"record": str(Path(record_path).resolve()), "record_sha256": record_sha,
            **{k: call_record.get(k) for k in ("response_id", "provider_model", "input_tokens", "output_tokens",
                                               "total_tokens", "execution_wall_s")}}
    return _write_step(tag, "correction", previous, commit, {
        "request_id": request_id, "request": {k: request.get(k) for k in ("correction_id", "recovery_mode", "budget_gpu_s")},
        "rule": {k: request["evidence"].get(k) for k in ("rule_mode", "rule", "standstill_ratio")},
        "model_proposal": {k: proposal.get(k) for k in
                                                     ("accepted", "decision", "agrees_with_rule", "reasons",
                                                      "unmet_checklist", "rule_visible_to_model")},
        "call": call, "receipt": receipt,
        "result": read(result_path) if result_path.is_file() else None})


# --- 판정 ---
def _correction_ledger_state() -> dict:
    import correct as CR
    led = CR._ledger(CASES["F1"], CR.LOOPS)
    return {"requests": led.requests(), "holds": [e for e in led.events() if e["event"] == "held"]}


def classify(steps: dict, terminal: dict | None, ledger: dict) -> str:
    if terminal is not None:
        return terminal["outcome"]
    if "correction" in steps:
        verdict = steps["correction"]["receipt"]["verdict"]
        return {"recovered": "recovered", "not_recovered": "not_recovered"}.get(verdict, "correction_unknown")
    if "loop_close" in steps:
        if any(r["state"] == "rejected" for r in ledger["requests"].values()) and not any(
                r["state"] in ("pending", "approved", "running", "done") for r in ledger["requests"].values()):
            return "correction_rejected"
        if ledger["holds"] and not ledger["requests"]:
            return "correction_not_proposed"
    return "in_progress"


def conditions(steps: dict, outcome: str) -> dict:
    begin_step, evaluate_step = steps.get("begin"), steps.get("evaluate")
    start, close, fix = steps.get("loop_start"), steps.get("loop_close"), steps.get("correction")
    shas = {s["protocol_sha256_lf"] for s in steps.values()}
    done = [r for r in (close or {}).get("loop", {}).get("requests", []) if r["state"] == "done"]
    call = (start or {}).get("call") or {}
    fix_call = (fix or {}).get("call") or {}
    costs = [step.get("execution", {}).get("parent_wall_s") for name, step in steps.items()
             if name in ("train_NONE", "train_F1", "evaluate", "reference")]
    fix_costs = ((fix or {}).get("result") or {}).get("costs", {})
    return {
        "protocol_committed": len(shas) == 1 and bool(steps),
        "seed_unused": bool(begin_step) and not begin_step["seed_unused"]["hits"],
        "source_failure_present": bool(evaluate_step) and evaluate_step["gate"]["outcome"] == "failure_present",
        "fresh_diagnosis_call": bool(start) and bool(call.get("response_id")) and start["decision"]["action"] == "continue",
        "first_probe_from_model": bool(start) and bool((start.get("started") or {}).get("model_probe_used")),
        "probes_approved_consumed_once": bool(done) and all(r.get("approver") and (r.get("artifact") or r["outcome"] == "unknown")
                                                            for r in done),
        "loop_confirmed_reward": bool(close) and close["proceed_to_correction"],
        "correction_call_accepted": bool(fix) and fix["model_proposal"]["accepted"] is True and bool(fix_call.get("response_id")),
        "correction_approved_consumed_once": bool(fix) and fix["receipt"].get("event") == "receipt",
        "restart_recovered": outcome == "recovered",
        "costs_recorded": (len(costs) == 4 and all(isinstance(c, (int, float)) for c in costs)
                           and all(isinstance(r.get("gpu_s"), (int, float)) for r in done) and bool(done)
                           and all(isinstance(fix_costs.get(k), (int, float)) for k in ("train_parent_wall_s", "eval_parent_wall_s"))
                           and all(isinstance(c.get(k), (int, float)) for c in (call, fix_call)
                                   for k in ("execution_wall_s", "input_tokens", "output_tokens"))),
    }


def public_view(steps: dict) -> dict:
    """공개 결과에 남길 수치·판정. 절대 경로·PID·로그 원문은 넣지 않는다."""
    def pick(name, *keys):
        step = steps.get(name) or {}
        return {k: step.get(k) for k in keys} if step else None
    start, close, fix = steps.get("loop_start"), steps.get("loop_close"), steps.get("correction")
    call_keys = ("response_id", "provider_model", "input_tokens", "output_tokens", "total_tokens", "execution_wall_s")
    return {
        "training_parent_wall_s": {arm: (steps.get(f"train_{arm}") or {}).get("execution", {}).get("parent_wall_s")
                                   for arm in ("NONE", "F1")},
        "alarm": (pick("alarm", "results") or {}).get("results"),
        "evaluation": (pick("evaluate", "gate", "reports") or {}),
        "reference_parent_wall_s": (steps.get("reference") or {}).get("execution", {}).get("parent_wall_s"),
        "diagnosis": None if not start else {"decision": start["decision"], "started": start.get("started"),
                                             "call": {k: start["call"].get(k) for k in call_keys}},
        "loop": None if not close else {k: close["loop"][k] for k in ("status", "hypotheses", "observed", "requests",
                                                                       "probe_costs")},
        "correction": None if not fix else {
            "request_id": fix["request_id"], "model_proposal": fix["model_proposal"],
            "call": {k: fix["call"].get(k) for k in call_keys}, "receipt": fix["receipt"],
            "result": {k: (fix.get("result") or {}).get(k) for k in ("label", "verdict", "standstill_ratio", "costs")}},
    }


def prediction_results(steps: dict) -> dict:
    """사전등록 예측의 적중(True/False). 그 단계까지 가지 못했으면 None. 성공 조건과 별개다."""
    evaluate_step, alarm_step = steps.get("evaluate"), steps.get("alarm")
    start, close, fix = steps.get("loop_start"), steps.get("loop_close"), steps.get("correction")
    gate = (evaluate_step or {}).get("gate") or {}
    alarms = (alarm_step or {}).get("results") or {}
    decision = (start or {}).get("decision") or {}
    loop_state = (close or {}).get("loop") or {}
    return {
        "source_failure": None if not evaluate_step else (gate["outcome"] == "failure_present"
                                                          and gate["standstill_ratio"]["F1"] >= 0.9),
        "alarm": None if not alarm_step else (alarms.get("F1", {}).get("warned") is True
                                              and alarms.get("NONE", {}).get("warned") is False),
        "diagnosis": None if not start else (decision.get("action") == "continue"
                                             and "reward" in decision.get("ranking", [])[:LOOP_TOP_K]),
        "loop": None if not close else (close["proceed_to_correction"]
                                        and loop_state.get("observed", {}).get("P_reward") == "abnormal"),
        "correction_rule": None if not fix else fix["rule"].get("rule_mode") == "restart",
        "restart": None if not fix else fix["receipt"].get("verdict") == "recovered",
    }


def summarize(tag: str) -> dict:
    folder = folder_for(tag)
    if (folder / "result.json").exists():
        return read(folder / "result.json")
    steps = {name: step_record(tag, name) for name in STEPS if step_path(tag, name).is_file()}
    terminal = read(folder / "terminal.json") if (folder / "terminal.json").exists() else None
    outcome = classify(steps, terminal, _correction_ledger_state() if "loop_close" in steps else {"requests": {}, "holds": []})
    checks = conditions(steps, outcome)
    success = outcome == "recovered" and all(checks.values())
    unmet = [k for k, v in checks.items() if not v]
    wording = OUTCOMES[outcome] + (" " + FIXED["gap_wording"] + ", ".join(unmet) if unmet and outcome != "in_progress" else "")
    result = {"schema": "e2e_reward_demo_result_v1", "tag": tag, "seed": SEED, "case": CASES["F1"],
              "outcome": outcome, "demo_success": success, "success_conditions": checks, "unmet_conditions": unmet,
              "wording": wording, "common_wording": COMMON_WORDING,
              "steps_sha256": {k: sha(step_path(tag, k)) for k in steps},
              "terminal": terminal, "protocol_sha256_lf": sha(PROTOCOL, True) if PROTOCOL.exists() else None,
              "source_commit_sha": sorted({s["source_commit_sha"] for s in steps.values()}),
              "predictions": FIXED["predictions"], "prediction_results": prediction_results(steps),
              "approver_identity_attested": False,
              **public_view(steps)}
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="새 seed 보상 결함 종단 시연")
    ap.add_argument("command", choices=("prepare", "verify", "begin", "train", "alarm", "evaluate", "reference",
                                        "packet", "start-loop", "close-loop", "correct-run", "summarize", "status"))
    ap.add_argument("--isaaclab", type=Path, default=Path.home() / "IsaacLab")
    ap.add_argument("--tag")
    ap.add_argument("--arm", choices=("NONE", "F1"))
    ap.add_argument("--record", type=Path)
    ap.add_argument("--request")
    ap.add_argument("--correction-record", type=Path)
    args = ap.parse_args(argv)
    required = {"train": ("tag", "arm"), "start-loop": ("tag", "record"), "correct-run": ("tag", "request", "correction_record")}
    for name in required.get(args.command, ("tag",) if args.command not in ("prepare", "verify") else ()):
        if getattr(args, name) is None:
            ap.error(f"{args.command}에는 --{name.replace('_', '-')}가 필요하다")
    os.environ.update(UTF8_ENV)  # Isaac 하위 프로세스(cp949 콘솔) 출력 오류 방지. run_owned가 환경을 물려준다
    commands = {
        "prepare": lambda: prepare(args.isaaclab),
        "verify": lambda: {"status": "pass", "source_commit_sha": verify_protocol(args.isaaclab)[1]},
        "begin": lambda: begin(args.tag, args.isaaclab),
        "train": lambda: train(args.tag, args.arm, args.isaaclab),
        "alarm": lambda: alarm(args.tag, args.isaaclab),
        "evaluate": lambda: evaluate(args.tag, args.isaaclab),
        "reference": lambda: reference(args.tag, args.isaaclab),
        "packet": lambda: packet(args.tag, args.isaaclab),
        "start-loop": lambda: loop_start(args.tag, args.isaaclab, args.record),
        "close-loop": lambda: loop_close(args.tag, args.isaaclab),
        "correct-run": lambda: correction(args.tag, args.isaaclab, args.request, args.correction_record),
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
