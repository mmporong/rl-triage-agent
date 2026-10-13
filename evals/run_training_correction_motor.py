"""모터 곡선 결함(F2)을 고친 뒤 같은 300회 이어 학습이 정상 범위로 돌아오는지와 그 예산 반응을 고정 행동 평가로 본다.

결함 종류에 따라 회복 방식(이어 학습·처음부터 재학습)의 결정이 갈리는지 확인하려는 호스트 사전등록 개발 실험이다.
이미 본 seed 2026·2028의 h06(F2) 실행을 쓰며 모델 선택·진단 정확도·일반화·GPU 절감의 평가가 아니다.
같은 seed 처음부터 재학습은 교정한 코드가 정상 학습과 같아져 정상 정책을 바이트 그대로 재현하므로(v4 두 seed 관찰)
restart arm은 넣지 않는다. 대신 이어 학습의 예산 반응을 보려고 두 arm의 중간 체크포인트를 학습 없이 평가한다.
v3·v4 실행 파일은 공개 결과의 소스 결합 때문에 바꾸지 않고 함수와 학습 진입점만 재사용한다. 실행 커밋은 execute가
기록한 HEAD이며, 이후 arm·학습 자식·공개본은 그 커밋이 HEAD의 조상이고 결합 파일 blob이 같을 때만 진행한다.

  python evals/run_training_correction_motor.py prepare --contract <SDK CPU 계약 JSON>
  python evals/run_training_correction_motor.py verify
  python evals/run_training_correction_motor.py execute --seed 2026 --tag <tag>
  python evals/run_training_correction_motor.py export <tag>
  python evals/run_training_correction_motor.py replay evals/results/<tag>
  python evals/run_training_correction_motor.py aggregate evals/results/<tag 2026> evals/results/<tag 2028>
  (Isaac Python 학습 자식) python evals/run_training_correction_motor.py train-child --manifest <manifest.json>
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "bench", ROOT / "evals"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import publish_training_correction as Q  # noqa: E402
import publish_training_correction_v4 as Q4  # noqa: E402
import record_evidence_media as M  # noqa: E402
import run_runtime_readback_smoke as P  # noqa: E402
import run_training_correction as V3  # noqa: E402
import run_training_correction_v4 as V4  # noqa: E402
import runtime_readback_smoke as S  # noqa: E402
import train_with_recovery_mode as R  # noqa: E402
from rl_triage import grounded_recovery as C  # noqa: E402
from rl_triage import recovery as RC  # noqa: E402

# train_with_recovery_mode.execute가 H로 쓰는 이름(v3 함수 그대로)
read = V3.read
write_new = V3.write_new
sha = V3.sha
digest = V3.digest
load_config = V3.load_config
execution_valid = V3.execution_valid
validate_readback = V3.validate_readback
check_contract = V3.check_contract
standstill_ratio = V4.standstill_ratio
walks = V4.walks
label_from = V4.label_from

PROTOCOL = ROOT / "bench/protocols/training_correction_motor_v1.json"
PRIVATE = ROOT / "bench/private/training_correction_motor"
RESULTS = ROOT / "evals/results"
FIXED_EVAL = V3.FIXED_EVAL
ORIGINAL = V3.ORIGINAL
V3_MOTOR_PUBLIC = ROOT / "evals/results/motor_resume_dev_20261013"
V4_PUBLIC = {seed: ROOT / f"evals/results/reward_v4_s{seed}_20261013" for seed in (2026, 2028)}
SEEDS = (2026, 2028)
FAULT = "F2"
MECHANISM = "actuator"
CORRECTION = "restore_motor_torque_speed"
INPUT_RUNS = {str(seed): {"NONE": f"baseline_p0c_s{seed}", FAULT: f"h06_s{seed}"} for seed in SEEDS}
ARMS = ("clean_resume_reference", "resume_corrected")
TREATMENT = "resume_corrected"
ARM_SPECS = {
    "clean_resume_reference": {"recovery_mode": "resume", "fault": "NONE", "correction": None,
                               "role": "resume_reference"},
    "resume_corrected": {"recovery_mode": "resume", "fault": FAULT, "correction": CORRECTION,
                         "role": "treatment", "primary_reference": "clean_resume_reference"},
}
RESUME_ITERATION, FINAL_ITERATION = 299, 598
BUDGET_CHECKPOINTS = (350, 450, 550)
LABELS = ("recovered", "not_recovered", "invalid")
PREDICTIONS = {TREATMENT: "not_recovered"}
RULE_THRESHOLD = 0.9
RULE_CURRENT = "fault_mechanism == reward and original_fault_standstill_ratio >= 0.9 -> restart, otherwise resume"
RULE_NEXT_IF_CHANGED = ("(fault_mechanism == reward and original_fault_standstill_ratio >= 0.9) "
                        "or fault_mechanism == actuator -> restart, otherwise resume")
CODE = V3.CODE | {
    "evals/run_training_correction_motor.py", "evals/run_training_correction_v4.py", "bench/train_with_recovery_mode.py",
    "evals/publish_training_correction.py", "evals/publish_training_correction_v4.py",
}
SCHEMA_PUBLIC = "training_correction_motor_public_v1"
PUBLISHER_SOURCES = {"evals/run_training_correction_motor.py", "evals/run_training_correction_v4.py",
                     "evals/publish_training_correction.py", "src/rl_triage/grounded_recovery.py",
                     "src/rl_triage/recovery.py"}
GPU_ESTIMATE = {
    "unit": "부모 프로세스 벽시계(초). GPU 커널 시간이나 절감량이 아니다",
    "limit_two_seeds_s": 3600,
    "arm_total_s": [575.171, 631.516],
    "arm_total_basis": ("v4 공개 training_evidence.json의 이어 학습 arm 4개 parent wall total_s 범위"
                        "(학습 516.6~576.9 s, 고정 평가 51.2~60.4 s). 같은 4096 env·300회 이어 학습·같은 평가"),
    "budget_curve_eval_per_seed_s": [210, 290],
    "budget_curve_basis": ("평가 작업 6개를 한 프로세스에서 실행: P0-C 일괄 평가 작업당 평균 33.5 s·최대 43.4 s, "
                           "v4 단일 평가 parent wall 51.2~60.4 s에서 뺀 기동 시간 약 10~27 s"),
    "per_seed_s": [1360.3, 1553.0],
    "two_seeds_s": [2720.7, 3106.1],
}
FIXED = {
    "schema": "training_correction_motor_protocol_v1",
    "question": "모터 곡선 결함(F2)을 고친 뒤 같은 300회 이어 학습이 정상 범위로 돌아오는가, 그 예산 반응은 어떠한가",
    "purpose": ("보상 결함(F1)에서는 이어 학습이 돌아오지 않았다(v4). 결함 종류에 따라 이어 학습이 통하는지 보고, "
                "회복 방식 규칙이 결함 종류에 따라 이어 학습과 재학습을 가를 근거가 있는지 확인한다"),
    "scope": "seen_two_seed_actuator_fault_resume_and_budget_response_development",
    "excluded_seeds": {"2027": "v3 case_02(motor_resume_dev_20261013)에서 같은 이어 학습을 이미 실행해 no_recovery였다. 재실행하지 않는다"},
    "task": C.TASK, "seeds": list(SEEDS), "num_envs": 4096, "max_iterations": 300, "num_steps_per_env": 24,
    "action_scale": 0.25, "resume_iteration": RESUME_ITERATION, "final_iteration": FINAL_ITERATION,
    "fault": FAULT, "fault_mechanism": MECHANISM,
    "fault_definition": "bench/hidden_faults.py F2(동결): DC 모터 토크-속도 곡선의 속도 항 5배",
    "correction": CORRECTION, "input_runs": INPUT_RUNS, "source_cohort_already_seen": True,
    "arms": list(ARMS), "arm_specs": ARM_SPECS,
    "excluded_arms": {
        "restart_corrected": "같은 seed 처음부터 재학습은 교정 코드가 정상 학습과 같아 원본 정상 정책을 바이트 그대로 재현한다(v4 두 seed 관찰). 결함에 관한 정보를 더하지 않는다",
        "resume_corrected_std_reset": "v4 F1에서 회복 여부를 바꾸지 않았다. 이번 질문인 이어 학습의 예산 반응보다 정보가 작다",
        "resume_fault_sham": "v3 seed 2027에서 결함 유지 대조 0.826 m/s 대 교정 0.188 m/s로 교정의 효과는 이미 컸다. 두 seed 합계 약 20분을 더 써 GPU 상한을 넘긴다",
    },
    "budget_curve": {
        "checkpoints": [f"model_{k}.pt" for k in BUDGET_CHECKPOINTS],
        "updates_after_resume": [k - RESUME_ITERATION for k in BUDGET_CHECKPOINTS],
        "arms": list(ARMS), "training": False,
        "evaluation": "fixed_eval_v1과 같은 조건. seed마다 두 arm 학습이 끝난 뒤 한 프로세스에서 작업 6개(arm 2 × 체크포인트 3)",
        "comparison": "같은 iteration의 정상 이어 학습 체크포인트 대비 같은 p0c 밴드로 보조 판정한다. 원본(+0)과 최종(+299)은 기존 평가와 arm 평가를 쓴다",
        "used_for_labels": False, "used_for_rule_update": False,
        "rationale": ("v3 seed 2027에서 교정한 이어 학습은 0.826→0.188 m/s로 크게 줄었지만 밴드 밖이었다. 남은 질문은 '예산이 부족한가, "
                      "멈췄는가'다. 저장된 중간 체크포인트를 학습 없이 평가하면 두 seed 합계 약 7~10분으로 예산 반응을 볼 수 있다"),
        "budget_limited_candidate_rule": "최종 라벨이 not_recovered이고 정상 기준 대비 선속도 오차 비가 model_350보다 최종에서 작으면 기록한다. 규칙 갱신에 쓰지 않는다",
    },
    "gpu_estimate": GPU_ESTIMATE,
    "references": {
        "resume_primary": "same_seed_clean_reference_resumed_same_updates",
        "resume_secondary": "original_same_seed_clean_checkpoint",
        "budget_primary": "same_seed_clean_reference_checkpoint_at_the_same_iteration",
        "reference_validity": "이어 학습 기준은 고정 평가에서 제자리 대비 걷고(WALK_FRACTION) 낙상률 REF_FALL_MAX 이하이며 원본 정상 정책 대비 healthy여야 한다",
    },
    "resume_scope": "policy_optimizer_warm_start_not_environment_or_rng_restore",
    "known_confounders": V4.FIXED["known_confounders"],
    "prior_observations_wording": ("모터 v3 seed 2027: 교정 이어 학습 0.188 m/s, 결함 유지 대조 0.826 m/s, 정상 이어 학습 0.132 m/s(no_recovery). "
                                   "보상 v4 seed 2026·2028: 고친 이어 학습 1.156·1.137 m/s, 정상 이어 학습 0.143·0.133 m/s(not_recovered). "
                                   "수치는 prior_derivation에서 공개 파일로 다시 계산한다"),
    "predictions": PREDICTIONS,
    "secondary_predictions": {
        "final_error_below_half_of_original_fault": True,
        "final_error_below_first_budget_point": True,
    },
    "prediction_wording": ("resume_corrected는 두 seed 모두 not_recovered(근거: v3 seed 2027 교정 0.188 대 기준 0.132, 오차 비 약 1.42 > 1.1513). "
                           "보조 예측: 최종 선속도 오차가 원본 결함 오차의 절반 미만이고, budget curve 첫 점(model_350)보다 작다"),
    "labels": list(LABELS),
    "label_rule": "resume_corrected의 최종(model_598) 고정 평가가 같은 seed 정상 이어 학습 대비 healthy면 recovered, unhealthy면 not_recovered, 기준 무효·실행 실패·미실행은 invalid",
    "mechanical_signal": {
        "name": "standstill_ratio",
        "definition": "metrics.lin_vel_rmse_mps / standstill.lin_vel_rmse_mps in the same fixed_eval_v1 report",
        "new_measurement": False, "used_for_labels": False,
    },
    "recovery_mode_rule": {
        "current": RULE_CURRENT, "threshold": RULE_THRESHOLD,
        "recommendation_for_this_fault": "resume(메커니즘 actuator는 현재 규칙의 restart 조건에 해당하지 않는다)",
        "outcomes": {
            "recommended_recovered": "권고한 resume arm이 recovered",
            "recommended_not_recovered": "권고한 resume arm이 not_recovered",
            "undetermined": "권고 arm이 invalid",
        },
        "update_criteria": {
            "keep": "두 seed 모두 resume_corrected가 recovered → actuator는 resume을 유지한다(규칙 변경 없음)",
            "change": "두 seed 모두 유효하고 하나라도 not_recovered → 다음 시연에 next_rule_if_changed와 그 restart arm을 사전등록해 시험한다. 이번 실험은 restart를 시험하지 않는다",
            "undetermined": "어느 seed든 invalid이거나 두 seed가 다 없으면 → 규칙 갱신을 보류한다",
        },
        "next_rule_if_changed": RULE_NEXT_IF_CHANGED,
        "derived_from": "보상 결함 seed 2027 단일 관찰(v3)과 v4 두 seed 적중",
        "used_for_labels": False,
    },
    "python_runtime": V3.FIXED["python_runtime"], "torch_version": V3.FIXED["torch_version"],
    "bands_source": V3.FIXED["bands_source"],
    "allowed_agent_config_changes": {"resume": ["run_name", "resume", "load_run", "load_checkpoint", "max_iterations"]},
    "train_watchdog_s": 1800, "eval_watchdog_s": 360, "budget_eval_watchdog_s": 600, "semantic_retries": 0,
    "train_limit_per_arm": 1, "eval_limit_per_arm": 1, "budget_eval_limit": 1,
    "execution_unit": "one_seed_two_arms_in_registered_order_then_one_budget_curve_evaluation_stop_on_first_failure",
    "gpu_admission_profile": V3.FIXED["gpu_admission_profile"],
    "isaaclab_commit": V3.FIXED["isaaclab_commit"], "rsl_rl_version": V3.FIXED["rsl_rl_version"],
    "source_binding": ("execute가 기록한 HEAD가 실험 커밋이다. 이후 arm·학습 자식·budget 평가·공개본은 실험 커밋이 HEAD의 조상이고 "
                       "결합 파일 blob(LF 정규화 SHA256)이 두 커밋에서 같을 때만 진행하고 실험 커밋을 기록한다"),
    "cpu_evidence_reuse": "v1~v4와 같은 교정 표적의 CPU 행렬을 SHA로 재결합한다",
    "approval_actor": "claude_operator_user_authorized",
    "approval_wording": "사용자가 위임한 운영자 승인(승인자 문자열 기록, 신원 인증 아님)",
    "model_calls": 0, "model_selected_correction": False,
    "result_wording": {
        "recovered": "모터 곡선 결함을 고친 뒤 300회 이어 학습한 단일 실행이 같은 seed 정상 이어 학습 대비 고정 행동 허용 범위를 통과했다. 이미 본 seed의 합성 결함 개발 결과이며 일반화·AI 교정 선택·GPU 절감을 주장하지 않는다.",
        "not_recovered": "모터 곡선 결함을 고친 뒤 300회 이어 학습한 단일 실행이 같은 seed 정상 이어 학습 대비 고정 행동 허용 범위를 통과하지 못했다. 이미 본 seed의 합성 결함 개발 결과이며 일반화·AI 교정 선택·GPU 절감을 주장하지 않는다.",
        "invalid": "출처·개입·초기 상태·예산·평가·종료 계약이나 기준 정책 검증이 맞지 않거나 실행되지 않아 회복 판정을 보류한다.",
        "seed_summary": "이미 본 seed의 합성 결함(F2) 개발 결과다. arm마다 단일 실행이며 일반화·AI 교정 선택·GPU 절감 주장이 아니다.",
        "budget_curve": "중간 체크포인트 평가는 같은 학습 실행의 저장본을 다시 평가한 보조 기록이며 라벨과 규칙 갱신에 쓰지 않는다. 반복 실행의 예산 효과 추정이 아니다.",
        "rule_update": {
            "keep": "두 seed 모두 모터 결함의 이어 학습이 정상 범위로 돌아와, 현재 규칙(actuator는 resume)을 유지한다. 이미 본 두 seed의 단일 실행이며 결함 일반 규칙의 근거가 아니다.",
            "change": "모터 결함의 이어 학습이 등록 기준을 통과하지 못한 seed가 있어, 다음 시연에는 actuator에도 restart를 권고하는 규칙과 restart arm을 사전등록해 시험한다. 이번 실험은 restart를 시험하지 않았고 이어 학습의 예산 부족 가능성도 남으므로 restart를 택할 근거로 쓰지 않는다. 이미 본 두 seed의 단일 실행이며 결함 일반 규칙의 근거가 아니다.",
            "undetermined": "유효한 판정이 두 seed 모두 갖춰지지 않아 회복 방식 규칙을 갱신하지 않는다.",
        },
    },
    "forbidden_claims": ["이어 학습이 재학습보다 낫다", "GPU·시간 절감", "일반화", "AI가 교정이나 회복 방식을 골랐다", "최초"],
}
DYNAMIC = {"source_files", "sdk_sources", "rsl_sources", "correction_source_contract", "fixed_eval_sha256_lf",
           "resume_inputs", "prior_derivation", "cpu_checks"}


# --- 입력과 프로토콜 ---
def rule_recommendation(mechanism: str, ratio: float) -> str:
    return "restart" if mechanism == "reward" and ratio >= RULE_THRESHOLD else "resume"


def rule_outcome(recommended: str, labels: dict) -> dict:
    label = labels[TREATMENT]
    if recommended != "resume":
        outcome = "not_tested"
    elif label == "invalid":
        outcome = "undetermined"
    else:
        outcome = "recommended_" + label
    return {"recommended": recommended, "recommended_arm": TREATMENT if recommended == "resume" else None,
            "recommended_label": label if recommended == "resume" else None, "outcome": outcome}


def source_paths(isaaclab: Path, seed: int, fault: str) -> dict[str, Path]:
    name = INPUT_RUNS[str(seed)][fault]
    meta_path = ROOT / "bench/private" / f"{name}.meta.json"
    meta = read(meta_path)
    expected = {"task": FIXED["task"], "seed": seed, "num_envs": FIXED["num_envs"],
                "max_iterations": FIXED["max_iterations"], "fault": fault, "train_exit_code": 0}
    if any(type(meta.get(k)) is not type(v) or meta[k] != v for k, v in expected.items()):
        raise ValueError("original training metadata differs")
    log_root = (isaaclab / "logs/rsl_rl/unitree_go2_flat").resolve()
    run_dir = Path(meta["run_dir"]).resolve()
    if run_dir.parent != log_root or not run_dir.name.endswith("_" + name):
        raise ValueError("source run is outside the registered SDK log root")
    run_regex = "^" + re.escape(run_dir.name) + "$"
    if [p.resolve() for p in log_root.iterdir() if p.is_dir() and re.match(run_regex, p.name)] != [run_dir]:
        raise ValueError("source run must match exactly one directory")
    return {"metadata": meta_path, "checkpoint": run_dir / "model_299.pt",
            "environment": run_dir / "params/env.yaml", "agent": run_dir / "params/agent.yaml",
            "fixed_report": ORIGINAL / "runs" / f"{name}.json"}


def source_inputs(isaaclab: Path) -> dict:
    """두 seed의 기존 NONE·F2 입력의 SHA, 원본 판정, 사전 신호를 묶는다. 새 평가 결과로 세지 않는다."""
    summary = read(ORIGINAL / "summary.json")
    bands = summary["bands"]
    if set(bands) != {"survival_ratio_min", "err_xy_ratio_max", "err_yaw_ratio_max", "fall_frac_max"}:
        raise ValueError("original behavior bands differ")
    seeds, versions = {}, None
    for seed in SEEDS:
        runs, reports = {}, {}
        for fault, name in INPUT_RUNS[str(seed)].items():
            paths = source_paths(isaaclab, seed, fault)
            hashes = {key: sha(path) for key, path in paths.items()}
            report = V4._original_report(read(paths["fixed_report"]), name, hashes["checkpoint"])
            rows = [row for row in summary["runs"] if row["run"] == name]
            if len(rows) != 1 or rows[0]["metrics"] != report["metrics"] or rows[0]["checkpoint_sha256"] != hashes["checkpoint"]:
                raise ValueError("original summary differs from the raw report")
            if load_config(paths["agent"]).get("seed") != seed:
                raise ValueError("original agent seed differs")
            runs[fault] = {"run": name, "hashes": hashes, "original_label": rows[0]["fixed_label"],
                           "original_failed": rows[0]["fixed_failed"], "standstill_ratio": standstill_ratio(report)}
            reports[fault] = report
        if versions is None:
            versions = reports["NONE"]["versions"]
        if versions["torch"] != FIXED["torch_version"] or any(r["versions"] != versions for r in reports.values()):
            raise ValueError("original report runtimes differ")
        reference = C._behavior(reports["NONE"])
        for fault, report in reports.items():
            verdict = RC.verdict(RC.compare(C._behavior(report), reference), bands)
            if verdict["label"] != runs[fault]["original_label"] or verdict["failed"] != runs[fault]["original_failed"]:
                raise ValueError("original failure classification does not replay")
        if runs["NONE"]["original_label"] != "healthy" or runs[FAULT]["original_label"] != "unhealthy":
            raise ValueError("registered source failure or clean reference is absent")
        seeds[str(seed)] = {"runs": runs, "original_none_walks": walks(reports["NONE"]),
                            "rule_recommendation": rule_recommendation(MECHANISM, runs[FAULT]["standstill_ratio"])}
    return {"summary_sha256_lf": sha(ORIGINAL / "summary.json", True), "bands": bands,
            "seeds": seeds, "runtime_versions": versions}


def prior_derivation() -> dict:
    """사전 관찰(모터 v3 seed 2027, 보상 v4 두 seed)을 공개 파일에서 다시 계산한다."""
    motor = {"original_fault": ORIGINAL / "runs/h06_s2027.json", "original_none": ORIGINAL / "runs/baseline_p0c_s2027.json",
             "v3_clean_reference": V3_MOTOR_PUBLIC / "reports/clean_reference.json",
             "v3_fault_sham": V3_MOTOR_PUBLIC / "reports/fault_sham.json",
             "v3_corrected": V3_MOTOR_PUBLIC / "reports/corrected.json"}
    err = {key: C._behavior(read(path))["err_xy"] for key, path in motor.items()}
    reward = {}
    for seed, folder in V4_PUBLIC.items():
        numeric = read(folder / "result.json")["numeric"]
        reward[str(seed)] = {"resume_corrected_label": numeric["labels"]["resume_corrected"],
                             "resume_corrected_err_xy": numeric["behavior"]["resume_corrected"]["err_xy"],
                             "clean_resume_reference_err_xy": numeric["behavior"]["clean_resume_reference"]["err_xy"]}
    files = {**{f"motor_{key}": path for key, path in motor.items()},
             **{f"reward_v4_{seed}": folder / "result.json" for seed, folder in V4_PUBLIC.items()}}
    return {"motor_seed": 2027, "motor_v3_status": read(V3_MOTOR_PUBLIC / "result.json")["numeric"]["status"],
            "files": {key: {"path": path.relative_to(ROOT).as_posix(), "sha256_lf": sha(path, True)}
                      for key, path in files.items()},
            "motor_err_xy": err, "motor_corrected_to_reference_ratio": err["v3_corrected"] / err["v3_clean_reference"],
            "motor_standstill_ratio": {key: standstill_ratio(read(path)) for key, path in motor.items()},
            "reward_v4_resume": reward}


def source_maps(isaaclab: Path) -> dict:
    rsl = V3.rsl_root(isaaclab)
    names = {p.relative_to(rsl).as_posix() for p in rsl.rglob("*.py")}
    if not {"runners/on_policy_runner.py", "algorithms/ppo.py", "modules/actor_critic.py"} <= names:
        raise ValueError("RSL-RL source is missing")
    return {"source_files": {name: sha(ROOT / name, True) for name in sorted(CODE)},
            "sdk_sources": {name: sha(isaaclab / name, True) for name in sorted(V3.SDK_FILES)},
            "rsl_sources": {name: sha(rsl / name, True) for name in sorted(names)}}


def prepare(isaaclab: Path, contract_path: Path) -> dict:
    contract = read(contract_path)
    check_contract(contract, isaaclab)
    document = {**deepcopy(FIXED), **source_maps(isaaclab), "correction_source_contract": contract,
                "fixed_eval_sha256_lf": sha(FIXED_EVAL, True), "resume_inputs": source_inputs(isaaclab),
                "prior_derivation": prior_derivation(),
                "cpu_checks": V3.cpu_checks(contract_path.parent, contract, isaaclab)}
    write_new(PROTOCOL, document)
    return {"status": "prepared_commit_before_execution", "path": PROTOCOL.relative_to(ROOT).as_posix()}


def _blob_lf(commit: str, relative: str) -> str:
    try:
        shown = subprocess.run(["git", "-C", str(ROOT), "show", f"{commit}:{relative}"], check=True, capture_output=True)
    except subprocess.CalledProcessError as error:
        raise ValueError(f"bound source is missing at {commit[:12]}: {relative}") from error
    return hashlib.sha256(shown.stdout.replace(b"\r\n", b"\n").replace(b"\r", b"\n")).hexdigest()


def frozen_commit(experiment: str, head: str, bound: set[str]) -> str:
    """실험 커밋이 HEAD의 조상이고 결합 파일 blob이 두 커밋에서 같으면 실험 커밋을 돌려준다(source_binding)."""
    if not isinstance(experiment, str) or re.fullmatch(r"[0-9a-f]{40}", experiment) is None:
        raise ValueError("experiment source commit is malformed")
    if head == experiment:
        return head
    if subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", experiment, head],
                      capture_output=True).returncode != 0:
        raise ValueError("experiment source commit is not an ancestor of HEAD")
    changed = [relative for relative in sorted(bound) if _blob_lf(experiment, relative) != _blob_lf(head, relative)]
    if changed:
        raise ValueError(f"bound source differs between the experiment commit and HEAD: {changed}")
    return experiment


def bound_files() -> set[str]:
    return CODE | {PROTOCOL.relative_to(ROOT).as_posix(), FIXED_EVAL.relative_to(ROOT).as_posix()}


def verify_protocol(isaaclab: Path, experiment: str | None = None) -> tuple[dict, str]:
    """프로토콜·결합 파일·Isaac Lab을 검사한다. 실험 커밋이 없으면 HEAD를, 있으면 고정된 실험 커밋을 돌려준다."""
    protocol = read(PROTOCOL)
    if set(protocol) != set(FIXED) | DYNAMIC:
        raise ValueError("protocol fields differ")
    if any(type(protocol[k]) is not type(v) or protocol[k] != v for k, v in FIXED.items()):
        raise ValueError("fixed training contract changed")
    check_contract(protocol["correction_source_contract"], isaaclab)
    if any(protocol[k] != v for k, v in source_maps(isaaclab).items()):
        raise ValueError("training source hashes changed")
    if protocol["cpu_checks"] != V3.cpu_checks(ROOT / protocol["cpu_checks"]["folder"],
                                               protocol["correction_source_contract"], isaaclab):
        raise ValueError("CPU evidence hashes changed")
    if protocol["fixed_eval_sha256_lf"] != sha(FIXED_EVAL, True):
        raise ValueError("frozen behavior protocol changed")
    if protocol["resume_inputs"] != source_inputs(isaaclab):
        raise ValueError("bound checkpoint, config, or original evaluation changed")
    if protocol["prior_derivation"] != prior_derivation():
        raise ValueError("prior observation files changed")
    if S._git_head(isaaclab) != FIXED["isaaclab_commit"] or not S._git_tracked_clean(isaaclab):
        raise ValueError("Isaac Lab checkout differs")
    head = P.verify_committed(ROOT, bound_files())
    return protocol, head if experiment is None else frozen_commit(experiment, head, bound_files())


# --- 학습 자식(H 인터페이스)과 실행 검증 ---
def folder_for(tag: str) -> Path:
    if not isinstance(tag, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,39}", tag):
        raise ValueError("invalid training tag")
    folder = PRIVATE / tag
    if folder.is_symlink() or folder.resolve().parent != PRIVATE.resolve():
        raise ValueError("training output escapes its root")
    return folder


def selection(arm: str) -> tuple[str, str | None, str]:
    if arm not in ARM_SPECS:
        raise ValueError("unregistered arm")
    spec = ARM_SPECS[arm]
    return spec["fault"], spec["correction"], spec["recovery_mode"]


def manifest_for(folder: Path, request: dict, arm: str) -> dict:
    fault, correction, mode = selection(arm)
    seed = request["seed"]
    if seed not in SEEDS:
        raise ValueError("unregistered seed")
    paths = source_paths(Path(request["isaaclab_root"]), seed, fault)
    bound = read(PROTOCOL)["resume_inputs"]["seeds"][str(seed)]["runs"][fault]["hashes"]
    checkpoint = paths["checkpoint"].resolve()
    return {
        "schema": "training_correction_manifest_v2", "tag": folder.name, "seed": seed, "arm": arm,
        "recovery_mode": mode, "source_commit_sha": request["source_commit_sha"],
        "protocol_sha256_lf": request["protocol_sha256_lf"], "isaaclab_root": request["isaaclab_root"],
        "protocol_path": str(PROTOCOL.resolve()), "fault": fault, "correction_id": correction,
        **{key: FIXED[key] for key in ("num_envs", "max_iterations", "task")},
        "run_name": f"{folder.name}_{arm}", "output_dir": str((folder / arm / "train").resolve()),
        "final_iteration": FINAL_ITERATION, "resume_checkpoint_path": str(checkpoint),
        "resume_checkpoint_sha256": bound["checkpoint"], "resume_iteration": RESUME_ITERATION,
        "load_run_regex": "^" + re.escape(checkpoint.parent.name) + "$", "checkpoint_regex": "^model_299\\.pt$",
        "source_env_config_path": str(paths["environment"].resolve()),
        "source_agent_config_path": str(paths["agent"].resolve()),
        "config_source_fault": fault, "init_noise_std": None,
    }


def validate_training_configuration(log_dir: Path, manifest: dict) -> None:
    """원 설정과 학습 시작 직전 설정의 차이를 이어 학습 인자로 제한한다."""
    protocol = read(Path(manifest["protocol_path"]))
    hashes = protocol["resume_inputs"]["seeds"][str(manifest["seed"])]["runs"][manifest["config_source_fault"]]["hashes"]
    source_env, source_agent = Path(manifest["source_env_config_path"]), Path(manifest["source_agent_config_path"])
    for path, key in ((source_env, "environment"), (source_agent, "agent")):
        if sha(path) != hashes[key]:
            raise ValueError("source configuration changed before learning")
    original_env, original_agent = load_config(source_env), load_config(source_agent)
    current_env = load_config(Path(log_dir) / "params/env.yaml")
    current_agent = load_config(Path(log_dir) / "params/agent.yaml")
    if current_env != original_env:
        raise ValueError("environment configuration changed outside the registered code correction")
    allowed = set(FIXED["allowed_agent_config_changes"]["resume"])
    if ({k: v for k, v in original_agent.items() if k not in allowed}
            != {k: v for k, v in current_agent.items() if k not in allowed}):
        raise ValueError("agent configuration changed outside the recovery mode arguments")
    expected = {"run_name": manifest["run_name"], "resume": True, "load_run": manifest["load_run_regex"],
                "load_checkpoint": manifest["checkpoint_regex"], "max_iterations": manifest["max_iterations"]}
    if any(type(current_agent.get(k)) is not type(v) or current_agent[k] != v for k, v in expected.items()):
        raise ValueError("agent configuration differs from the recovery mode manifest")
    if (original_agent.get("seed") != manifest["seed"]
            or original_agent.get("num_steps_per_env") != FIXED["num_steps_per_env"]
            or original_agent.get("max_iterations") != FIXED["max_iterations"]
            or original_env.get("seed") != manifest["seed"]
            or original_env.get("scene", {}).get("num_envs") != FIXED["num_envs"]):
        raise ValueError("original training seed, environment count, or step count differs")


def validate_manifest(path: Path) -> tuple[dict, dict]:
    manifest = read(path)
    if set(manifest) != R.MANIFEST_FIELDS:
        raise ValueError("manifest fields differ from the fixed schema")
    folder = folder_for(manifest["tag"])
    if path.is_symlink() or path.resolve() != (folder / manifest["arm"] / "train/manifest.json").resolve():
        raise ValueError("manifest is outside its registered arm")
    request = read(folder / "request.json")
    protocol, commit = verify_protocol(Path(request["isaaclab_root"]), request["source_commit_sha"])
    if request["source_commit_sha"] != commit or request["protocol_sha256_lf"] != sha(PROTOCOL, True):
        raise ValueError("request source commitment changed")
    if manifest != manifest_for(folder, request, manifest["arm"]):
        raise ValueError("manifest differs from the approved request")
    if read(folder / "approval.json") != {"request_sha256": sha(folder / "request.json"),
                                          "actor": FIXED["approval_actor"], "identity_attested": False}:
        raise ValueError("operator authorization record differs")
    if read(folder / manifest["arm"] / "consumed.json") != {"approval_sha256": sha(folder / "approval.json"),
                                                           "arm": manifest["arm"], "train_limit": 1, "eval_limit": 1}:
        raise ValueError("arm consumption record differs")
    return manifest, protocol


def verify_training(folder: Path, arm: str, execution: dict) -> dict:
    """v4 이어 학습 분기와 같은 검사를 이 프로토콜의 상수로 한다."""
    training = folder / arm / "train"
    if not execution_valid(execution):
        raise ValueError(f"training execution failed (exit={execution.get('exit_code')}); inspect its stderr.log")
    manifest, protocol = validate_manifest(training / "manifest.json")
    if R.validate_manifest_mode(manifest) != "resume":
        raise ValueError("this protocol registers resume arms only")
    receipt, initial = read(training / "receipt.json"), read(training / "initial.json")
    if (receipt.get("completed") is not True or receipt.get("schema") != "training_correction_receipt_v2"
            or initial.get("schema") != "training_correction_initial_v2" or initial.get("recovery_mode") != "resume"):
        raise ValueError("training receipt or initial state schema differs")
    for key in ("tag", "seed", "arm", "recovery_mode", "fault", "correction_id", "source_commit_sha",
                "protocol_sha256_lf", "task", "run_name", "final_iteration"):
        if receipt.get(key) != manifest[key]:
            raise ValueError(f"training receipt binding differs: {key}")
    for file, field in (("manifest.json", "manifest_sha256"), ("initial.json", "initial_sha256"),
                        ("correction.json", "correction_sha256")):
        if receipt.get(field) != sha(training / file):
            raise ValueError("training raw artifact changed")
    identity = receipt["process_identity"]
    if execution["owned_process_identities"].get(str(identity["process_id"])) != identity["process_creation_time_filetime"]:
        raise ValueError("training receipt came from an unowned process")
    steps = FIXED["max_iterations"] * FIXED["num_steps_per_env"]
    if (receipt.get("actual_control_steps") != steps or receipt.get("environment_make_calls") != 1
            or receipt.get("aggregate_env_steps") != steps * FIXED["num_envs"]
            or receipt.get("budget") != {"seed": manifest["seed"], "num_envs": FIXED["num_envs"],
                                         "max_iterations": FIXED["max_iterations"],
                                         "num_steps_per_env": FIXED["num_steps_per_env"], "expected_control_steps": steps}):
        raise ValueError("training budget or environment count differs")
    expected_resume = {"path": manifest["resume_checkpoint_path"], "sha256": manifest["resume_checkpoint_sha256"]}
    source_state = receipt.get("source_checkpoint_state")
    if (receipt.get("resume_checkpoint") != expected_resume
            or initial.get("resume_checkpoint") != {**expected_resume, "load_optimizer": True}
            or receipt.get("checkpoint_load_calls") != 1
            or any(row.get(key) != RESUME_ITERATION for row in (receipt, initial)
                   for key in ("resume_iteration", "start_iteration"))
            or not isinstance(source_state, dict) or source_state != initial.get("source_checkpoint_state")
            or source_state.get("iteration") != RESUME_ITERATION):
        raise ValueError("training did not resume the bound policy and optimizer")
    if (initial.get("std_reset") is not None or receipt.get("std_reset_applied") is not False
            or source_state.get("policy_state_sha256") != initial["policy_state"]["sha256"]
            or source_state.get("optimizer_state_sha256") != initial["optimizer_state"]["sha256"]):
        raise ValueError("initial policy/optimizer differs from the source checkpoint")
    for state, iteration in ((source_state, RESUME_ITERATION), (receipt["final_checkpoint_state"], FINAL_ITERATION)):
        if state.get("iteration") != iteration or any(not V4._sha_text(state.get(k))
                                                      for k in ("policy_state_sha256", "optimizer_state_sha256")):
            raise ValueError("checkpoint state proof is incomplete")
    if (receipt.get("resume_scope") != {"policy_optimizer_restored": True, "full_environment_state_restored": False,
                                        "rng_state_restored_from_checkpoint": False, "fresh_initialization": False,
                                        "action_std_reset": False}
            or initial.get("init_at_random_ep_len") is not True
            or initial.get("full_environment_state_restored") is not False
            or initial.get("rng_state_restored_from_checkpoint") is not False
            or receipt.get("rng_state_before_learn") != initial.get("rng_state_before_learn")
            or not V4._rng_ok(initial.get("rng_state_before_learn"))):
        raise ValueError("resume scope, episode initialization, or RNG observation differs")
    observed = {"environment_seed": manifest["seed"], "environment_num_envs": FIXED["num_envs"],
                "action_scale": FIXED["action_scale"], "agent_seed": manifest["seed"],
                "agent_max_iterations": FIXED["max_iterations"], "runner_num_steps_per_env": FIXED["num_steps_per_env"]}
    if (receipt.get("observed") != observed or initial.get("observed") != observed
            or receipt.get("actual_action_scale") != FIXED["action_scale"]):
        raise ValueError("actual training configuration differs")
    versions, runtime = receipt["runtime_versions"], protocol["correction_source_contract"]["runtime"]
    if (versions != initial.get("runtime_versions") or versions.get("python") != ".".join(map(str, runtime["version"]))
            or versions.get("python_cache_tag") != runtime["cache_tag"]
            or versions.get("rsl_rl_distribution") != FIXED["rsl_rl_version"]
            or versions.get("torch_distribution") != protocol["resume_inputs"]["runtime_versions"]["torch"]
            or versions.get("isaaclab_distribution") != protocol["resume_inputs"]["runtime_versions"]["isaaclab"]
            or versions.get("isaaclab_source_commit") != FIXED["isaaclab_commit"]):
        raise ValueError("actual training runtime differs")
    correction = read(training / "correction.json")
    validate_readback(correction, manifest["fault"], manifest["correction_id"])
    for key, target in protocol["correction_source_contract"]["targets"].items():
        if correction["before_fault"][key] != target["clean_fingerprint"]:
            raise ValueError("clean live target differs from committed SDK source")
    log_root = (Path(manifest["isaaclab_root"]) / "logs/rsl_rl/unitree_go2_flat").resolve()
    checkpoint = Path(receipt["checkpoint"]["path"])
    if (checkpoint.resolve().parent.parent != log_root or not checkpoint.parent.name.endswith("_" + manifest["run_name"])
            or checkpoint.name != f"model_{FINAL_ITERATION}.pt"
            or Path(initial["log_dir"]).resolve() != checkpoint.parent.resolve()):
        raise ValueError("training checkpoint is outside its exact run")
    if set(receipt["configs"]) != {"environment", "agent"} or any(
            Path(receipt["configs"][key]["path"]).resolve() != (checkpoint.parent / "params" / name).resolve()
            for key, name in (("environment", "env.yaml"), ("agent", "agent.yaml"))):
        raise ValueError("training config path is outside its exact run")
    for artifact in [receipt["checkpoint"], *receipt["configs"].values()]:
        if sha(Path(artifact["path"])) != artifact["sha256"]:
            raise ValueError("checkpoint or configuration hash differs")
    validate_training_configuration(checkpoint.parent, manifest)
    return receipt


def budget_name(arm: str, iteration: int) -> str:
    return f"{arm}_model_{iteration}"


def budget_checkpoints(receipt: dict) -> dict[int, Path]:
    """검증된 학습 run 폴더 안의 중간 체크포인트. 학습 receipt의 최종 체크포인트와 같은 폴더여야 한다."""
    run = Path(receipt["checkpoint"]["path"]).resolve().parent
    paths = {k: run / f"model_{k}.pt" for k in BUDGET_CHECKPOINTS}
    if any(not path.is_file() or path.is_symlink() for path in paths.values()):
        raise ValueError("intermediate checkpoint is missing from the verified run")
    return paths


def checkpoint_iteration(path: Path) -> int | None:
    """torch 없이 체크포인트 내부의 저장 iteration을 읽는다(허용 전역만 해석하는 SK1 읽기 함수)."""
    from rl_triage.skill_report import checkpoint_shapes
    import pickle
    import zipfile
    try:
        return checkpoint_shapes(path)["iteration"]
    except (ValueError, KeyError, pickle.UnpicklingError, zipfile.BadZipFile) as error:
        raise ValueError(f"intermediate checkpoint cannot be read: {path.name}") from error


def budget_marks(receipt: dict) -> dict[str, dict]:
    """arm 학습 직후 중간 체크포인트의 파일명·SHA·내부 iteration을 기록한다. budget 평가와 공개본이 이 기록과 대조한다."""
    marks = {}
    for k, path in budget_checkpoints(receipt).items():
        iteration = checkpoint_iteration(path)
        if iteration != k:
            raise ValueError(f"intermediate checkpoint iteration differs from its file name: {path.name}")
        marks[str(k)] = {"file": path.name, "sha256": sha(path), "iteration": iteration}
    return marks


def _check_marks(receipt: dict, marks: dict) -> dict[int, Path]:
    paths = budget_checkpoints(receipt)
    if set(marks) != {str(k) for k in paths} or any(
            marks[str(k)] != {"file": path.name, "sha256": sha(path), "iteration": k} for k, path in paths.items()):
        raise ValueError("intermediate checkpoint changed after its arm was trained")
    return paths


# --- 판정 ---
def _report_ok(report: dict, name: str, checkpoint_file: str, fixed_eval_sha256_lf: str) -> dict:
    return C.validate_fixed_report(report, {
        "name": name, "checkpoint_file": checkpoint_file, "checkpoint_sha256": report["checkpoint"]["sha256"],
        "action_scale": FIXED["action_scale"], "protocol_sha256_lf": fixed_eval_sha256_lf,
    })


def _check_initials(initials: dict) -> None:
    for initial in initials.values():
        if initial.get("recovery_mode") != "resume" or not V4._rng_ok(initial.get("rng_state_before_learn")):
            raise ValueError("initial recovery mode or RNG observation differs")
        if initial.get("resume_checkpoint") is None or initial.get("std_reset") is not None:
            raise ValueError("initial checkpoint usage differs from the recovery mode")
    if set(ARMS) <= set(initials) and (initials["clean_resume_reference"]["resume_checkpoint"]
                                       == initials[TREATMENT]["resume_checkpoint"]):
        raise ValueError("clean reference resumed the failed checkpoint")


def _point(iteration: int, corrected: dict, reference: dict, bands: dict, source: str) -> dict:
    treated, base = C._behavior(corrected), C._behavior(reference)
    verdict = RC.verdict(RC.compare(treated, base), bands)
    return {"iteration": iteration, "updates_after_resume": iteration - RESUME_ITERATION, "source": source,
            "corrected_err_xy": treated["err_xy"], "reference_err_xy": base["err_xy"],
            "err_xy_ratio": treated["err_xy"] / base["err_xy"],
            "corrected_standstill_ratio": standstill_ratio(corrected), "reference_walks": walks(reference),
            "verdict": verdict["label"], "failed": verdict["failed"]}


def budget_curve(reports: dict, budget_reports: dict | None, original_reports: dict, bands: dict,
                 budget_failed: bool, fixed_eval_sha256_lf: str) -> dict:
    """원본(+0)·중간 체크포인트·최종(+299)의 교정 대 기준 비교. 라벨과 규칙 갱신에 쓰지 않는 보조 기록이다."""
    names = {budget_name(arm, k): (arm, k) for arm in ARMS for k in BUDGET_CHECKPOINTS}
    if budget_reports is None or set(reports) != set(ARMS):
        return {"status": "failed" if budget_failed else "not_executed", "points": [], "used_for_labels": False,
                "wording": FIXED["result_wording"]["budget_curve"]}
    try:
        if set(budget_reports) != set(names):
            raise ValueError("budget curve report set differs from the registered checkpoints")
        for name, (_arm, k) in names.items():
            _report_ok(budget_reports[name], name, f"model_{k}.pt", fixed_eval_sha256_lf)
            if budget_reports[name]["versions"] != original_reports["clean_reference"]["versions"]:
                raise ValueError("budget curve evaluation runtime differs")
    except (ValueError, KeyError, TypeError) as error:
        # budget curve는 라벨에 쓰지 않으므로 이 오류는 보조 기록만 무효로 둔다.
        return {"status": "invalid", "reason": type(error).__name__, "points": [], "used_for_labels": False,
                "wording": FIXED["result_wording"]["budget_curve"]}
    points = [_point(RESUME_ITERATION, original_reports["fault"], original_reports["clean_reference"], bands,
                     "original_fixed_eval")]
    points += [_point(k, budget_reports[budget_name(TREATMENT, k)], budget_reports[budget_name("clean_resume_reference", k)],
                      bands, "budget_checkpoint") for k in BUDGET_CHECKPOINTS]
    points.append(_point(FINAL_ITERATION, reports[TREATMENT], reports["clean_resume_reference"], bands, "arm_fixed_eval"))
    return {"status": "complete", "points": points, "used_for_labels": False,
            "first_healthy_iteration": next((p["iteration"] for p in points[1:]
                                             if p["verdict"] == "healthy" and p["reference_walks"]), None),
            "wording": FIXED["result_wording"]["budget_curve"]}


def summarize(seed: int, reports: dict, initials: dict, original_reports: dict, bands: dict, original_failed: list[str],
              failed_arm: str | None = None, budget_reports: dict | None = None, budget_failed: bool = False,
              fixed_eval_sha256_lf: str | None = None) -> dict:
    """완료된 arm 접두부와 budget curve의 원시 행동 수치를 판정한다. 실행 출처는 부모 검증이 별도로 결합한다."""
    fixed_eval_sha256_lf = fixed_eval_sha256_lf or sha(FIXED_EVAL, True)
    if seed not in SEEDS:
        raise ValueError("unregistered seed")
    available = [arm for arm in ARMS if arm in reports]
    if set(reports) != set(available) or available != list(ARMS[:len(available)]) or set(initials) != set(reports):
        raise ValueError("completed arms must be a prefix of the registered order")
    if failed_arm is not None and (failed_arm not in ARMS or failed_arm in reports):
        raise ValueError("failed arm must be a registered arm without a report")
    _check_initials(initials)
    versions = [initials[arm].get("runtime_versions") for arm in available]
    if any(not isinstance(value, dict) or value != versions[0] for value in versions):
        raise ValueError("training runtimes differ or are missing")
    for arm in available:
        _report_ok(reports[arm], arm, f"model_{FINAL_ITERATION}.pt", fixed_eval_sha256_lf)
    if set(original_reports) != {"clean_reference", "fault"}:
        raise ValueError("original reference and failed report are required")
    for report in original_reports.values():
        _report_ok(report, report["name"], "model_299.pt", fixed_eval_sha256_lf)
    runtime = original_reports["clean_reference"]["versions"]
    if any(report["versions"] != runtime for report in [*reports.values(), *original_reports.values()]):
        raise ValueError("fixed evaluation runtimes differ")
    original = {key: C._behavior(value) for key, value in original_reports.items()}
    source_verdict = RC.verdict(RC.compare(original["fault"], original["clean_reference"]), bands)
    if source_verdict["label"] != "unhealthy" or source_verdict["failed"] != original_failed:
        raise ValueError("original failure metrics changed")
    behavior = {arm: C._behavior(report) for arm, report in reports.items()}
    secondary = {arm: RC.verdict(RC.compare(value, original["clean_reference"]), bands) for arm, value in behavior.items()}
    if "clean_resume_reference" in reports:
        reference = {"source": FIXED["references"]["resume_primary"],
                     "walks": walks(reports["clean_resume_reference"]),
                     "secondary_label": secondary["clean_resume_reference"]["label"]}
        reference["valid"] = reference["walks"] and reference["secondary_label"] == "healthy"
    else:
        reference = {"source": FIXED["references"]["resume_primary"], "walks": None, "secondary_label": None, "valid": False}
    verdicts = {}
    if TREATMENT not in reports:
        label, reason = "invalid", "execution_failed" if failed_arm == TREATMENT else "not_executed"
    elif not reference["valid"]:
        label, reason = "invalid", "resume_reference_invalid"
    else:
        verdicts[TREATMENT] = RC.verdict(RC.compare(behavior[TREATMENT], behavior["clean_resume_reference"]), bands)
        label = label_from(verdicts[TREATMENT])
        reason = "verdict_" + verdicts[TREATMENT]["label"]
    labels, reasons = {TREATMENT: label}, {TREATMENT: reason}
    curve = budget_curve(reports, budget_reports, original_reports, bands, budget_failed, fixed_eval_sha256_lf)
    signal = {"original_none": standstill_ratio(original_reports["clean_reference"]),
              "original_fault": standstill_ratio(original_reports["fault"]),
              **{arm: standstill_ratio(report) for arm, report in reports.items()}}
    secondary_predictions = {key: None for key in FIXED["secondary_predictions"]}
    if TREATMENT in behavior:
        secondary_predictions["final_error_below_half_of_original_fault"] = (
            behavior[TREATMENT]["err_xy"] < 0.5 * original["fault"]["err_xy"])
    if curve["status"] == "complete":
        first, last = curve["points"][1], curve["points"][-1]
        secondary_predictions["final_error_below_first_budget_point"] = last["corrected_err_xy"] < first["corrected_err_xy"]
        curve["budget_limited_candidate"] = label == "not_recovered" and last["err_xy_ratio"] < first["err_xy_ratio"]
    rule = rule_outcome(rule_recommendation(MECHANISM, signal["original_fault"]), labels)
    match = None if label == "invalid" else label == PREDICTIONS[TREATMENT]
    status = "invalid" if label == "invalid" else "complete" if curve["status"] == "complete" else "partial"
    return {
        "status": status, "seed": seed, "scope": FIXED["scope"], "arm_order": list(ARMS),
        "completed_arms": available, "failed_arm": failed_arm, "labels": labels, "label_reasons": reasons,
        "predictions": PREDICTIONS, "prediction_match": {TREATMENT: match},
        "secondary_predictions": FIXED["secondary_predictions"], "secondary_prediction_results": secondary_predictions,
        "reference": reference, "behavior": behavior, "verdicts": verdicts, "secondary_verdicts": secondary,
        "original_behavior": original, "source_verdict": source_verdict, "budget_curve": curve,
        "mechanical_signal": {"name": "standstill_ratio", "values": signal, "used_for_labels": False},
        "recovery_mode_rule": {**rule, "rule": RULE_CURRENT, "used_for_labels": False}, "bands": bands,
        "full_environment_state_restored": False, "rng_state_restored_from_checkpoint": False,
        "causal_confirmation": False, "exact_training_trajectory_resumed": False, "model_selected_correction": False,
        "generalization_proven": False, "gpu_saving_claimed": False,
        "wording": FIXED["result_wording"][label], "seed_wording": FIXED["result_wording"]["seed_summary"],
    }


def invalid_result(seed: int, error: Exception | str) -> dict:
    return {"status": "invalid", "seed": seed, "scope": FIXED["scope"], "labels": {TREATMENT: "invalid"},
            "label_reasons": {TREATMENT: "seed_invalid"},
            "error_type": error if isinstance(error, str) else type(error).__name__,
            "detail": None if isinstance(error, str) else str(error),
            "wording": FIXED["result_wording"]["invalid"], "seed_wording": FIXED["result_wording"]["seed_summary"]}


def aggregate(results: list[dict]) -> dict:
    """seed별 결과를 모아 사전 기준대로 회복 방식 규칙 갱신 결정을 낸다. 새 판정을 만들지 않는다."""
    seeds = [row.get("seed") for row in results]
    if not seeds or len(seeds) != len(set(seeds)) or not set(seeds) <= set(SEEDS):
        raise ValueError("aggregate needs distinct registered seed results")
    if len({row.get("protocol_sha256_lf") for row in results}) != 1:
        raise ValueError("seed results come from different protocols")
    labels = {str(row["seed"]): row["labels"][TREATMENT] for row in results}
    complete = sorted(seeds) == list(SEEDS)
    if not complete or "invalid" in labels.values():
        decision = "undetermined"
    elif all(value == "recovered" for value in labels.values()):
        decision = "keep"
    else:
        decision = "change"
    valid = [value for value in labels.values() if value != "invalid"]
    return {
        "seeds": sorted(seeds), "complete_seed_set": complete, "labels": labels,
        "recovered": sum(value == "recovered" for value in valid), "valid_seeds": len(valid),
        "prediction": PREDICTIONS[TREATMENT],
        "prediction_held_all_seeds": None if len(valid) != len(labels) or not complete
        else all(value == PREDICTIONS[TREATMENT] for value in valid),
        "budget_limited_candidates": {str(row["seed"]): (row.get("budget_curve") or {}).get("budget_limited_candidate")
                                      for row in results},
        "rule_update": {"decision": decision, "current_rule": RULE_CURRENT,
                        "next_rule": {"keep": RULE_CURRENT, "change": RULE_NEXT_IF_CHANGED, "undetermined": None}[decision],
                        "criteria": FIXED["recovery_mode_rule"]["update_criteria"][decision],
                        "wording": FIXED["result_wording"]["rule_update"][decision]},
        "scope": FIXED["scope"], "wording": FIXED["result_wording"]["seed_summary"],
    }


# --- 실행 ---
def _verified(isaaclab: Path, protocol: dict, commit: str, stage: str) -> None:
    current, current_commit = verify_protocol(isaaclab, commit)
    if current != protocol or current_commit != commit:
        raise ValueError(f"source changed {stage}")


def _owned_gpu_check(after: dict, *executions: dict) -> None:
    owned = set().union(*(set(execution["owned_process_identities"]) for execution in executions))
    if any(str(row["pid"]) in owned for row in after["processes"]):
        raise ValueError("an owned GPU process remains")


def run_budget_curve(folder: Path, isaaclab: Path, protocol: dict, commit: str, receipts: dict) -> dict:
    """두 arm의 중간 체크포인트를 학습 없이 한 평가 프로세스로 고정 평가한다."""
    budget = folder / "budget"
    _verified(isaaclab, protocol, commit, "before the budget curve")
    gpu = M.query_gpu(FIXED["gpu_admission_profile"])
    if gpu["busy"]:
        raise RuntimeError("GPU busy before the budget curve evaluation")
    budget.mkdir()
    write_new(budget / "gpu_before.json", gpu)
    checkpoints, jobs = {}, []
    for arm in ARMS:
        if not (folder / arm / "budget_checkpoints.json").is_file():
            raise ValueError(f"intermediate checkpoint record is missing for {arm}")
        marks = read(folder / arm / "budget_checkpoints.json")
        if read(folder / arm / "parent.json").get("budget_checkpoints_sha256") != sha(folder / arm / "budget_checkpoints.json"):
            raise ValueError("arm budget checkpoint record differs from its parent binding")
        for k, path in _check_marks(receipts[arm], marks).items():
            name = budget_name(arm, k)
            checkpoints[name] = {"path": str(path), "sha256": marks[str(k)]["sha256"]}
            jobs.append({"name": name, "checkpoint": str(path), "output": str(budget / f"{name}.json")})
    write_new(budget / "checkpoints.json", checkpoints)
    write_new(budget / "jobs.json", {"action_scale": FIXED["action_scale"], "jobs": jobs})
    argv = [str(isaaclab / "_isaac_sim/python.bat"), str(ROOT / "evals/fixed_eval.py"), "--protocol",
            str(FIXED_EVAL), "--jobs", str(budget / "jobs.json"), "--headless"]
    execution = P.run_owned(budget, argv, isaaclab, FIXED["budget_eval_watchdog_s"])
    write_new(budget / "execution.json", execution)
    if not execution_valid(execution):
        raise ValueError("budget curve evaluation did not complete and clean up")
    reports = {}
    for name, row in checkpoints.items():
        reports[name] = C.validate_fixed_report(read(budget / f"{name}.json"), {
            "name": name, "checkpoint_file": Path(row["path"]).name, "checkpoint_sha256": row["sha256"],
            "action_scale": FIXED["action_scale"], "protocol_sha256_lf": protocol["fixed_eval_sha256_lf"]})
        if reports[name]["versions"] != protocol["resume_inputs"]["runtime_versions"]:
            raise ValueError("budget evaluator runtime differs from the bound original")
        if sha(Path(row["path"])) != row["sha256"]:
            raise ValueError("intermediate checkpoint changed during evaluation")
    _verified(isaaclab, protocol, commit, "during the budget curve")
    after = M.query_gpu(FIXED["gpu_admission_profile"])
    write_new(budget / "gpu_after.json", after)
    _owned_gpu_check(after, execution)
    write_new(budget / "parent.json", {"status": "pass", "checkpoints_sha256": sha(budget / "checkpoints.json"),
                                       "reports_sha256": {name: sha(budget / f"{name}.json") for name in checkpoints},
                                       "eval_parent_wall_s": execution["wall_s"]})
    return reports


def execute(tag: str, seed: int, isaaclab: Path) -> dict:
    protocol, commit = verify_protocol(isaaclab)
    if seed not in SEEDS:
        raise ValueError("seed is not registered")
    folder = folder_for(tag)
    key = digest({"protocol": sha(PROTOCOL, True), "seed": seed})
    consumption = PRIVATE / "consumed" / f"{key}.json"
    if folder.exists() or consumption.exists():
        raise FileExistsError("seed or tag has already been consumed")
    log_root = isaaclab / "logs/rsl_rl/unitree_go2_flat"
    if any(path.name.endswith(f"_{tag}_{arm}") for arm in ARMS for path in log_root.iterdir()):
        raise FileExistsError("an SDK run already exists for this arm")
    if any(char in str(path) for path in (ROOT, isaaclab, folder) for char in '\r\n"%&^'):
        raise ValueError("unsafe Windows batch path")
    if M.query_gpu(FIXED["gpu_admission_profile"])["busy"]:
        return {"status": "gpu_busy_no_launch", "sdk_jobs_launched": 0}
    request = {"tag": tag, "seed": seed, "source_commit_sha": commit, "protocol_sha256_lf": sha(PROTOCOL, True),
               "isaaclab_root": str(isaaclab.resolve()),
               "selection_source": "registered_host_resume_comparison_not_model", "bundle_key": key}
    write_new(folder / "request.json", request)
    write_new(folder / "approval.json", {"request_sha256": sha(folder / "request.json"),
                                         "actor": FIXED["approval_actor"], "identity_attested": False})
    write_new(consumption, {"request_sha256": sha(folder / "request.json"), "tag": tag})
    reports, initials, receipts, costs = {}, {}, {}, {}
    failed_arm, budget_reports, budget_failed, budget_cost = None, None, False, None
    for arm in ARMS:
        stage = "preflight"
        try:
            _verified(isaaclab, protocol, commit, "during the experiment")
            gpu = M.query_gpu(FIXED["gpu_admission_profile"])
            if gpu["busy"]:
                raise RuntimeError("GPU busy before next training arm")
            arm_dir = folder / arm
            training, evaluation = arm_dir / "train", arm_dir / "eval"
            write_new(arm_dir / "gpu_before.json", gpu)
            write_new(arm_dir / "consumed.json", {"approval_sha256": sha(folder / "approval.json"), "arm": arm,
                                                  "train_limit": 1, "eval_limit": 1})
            write_new(training / "manifest.json", manifest_for(folder, request, arm))
            argv = [str(isaaclab / "_isaac_sim/python.bat"), str(ROOT / "evals/run_training_correction_motor.py"),
                    "train-child", "--manifest", str(training / "manifest.json")]
            stage = "training"
            execution = P.run_owned(training, argv, isaaclab, FIXED["train_watchdog_s"])
            write_new(training / "execution.json", execution)
            stage = "training_validation"
            receipts[arm] = verify_training(folder, arm, execution)
            initials[arm] = read(training / "initial.json")
            try:
                write_new(arm_dir / "budget_checkpoints.json", budget_marks(receipts[arm]))
            except (ValueError, OSError) as error:
                # 중간 체크포인트는 budget curve 전용이므로 기록 실패가 arm 라벨을 막지 않는다.
                write_new(arm_dir / "budget_checkpoints_failure.json", {"error_type": type(error).__name__,
                                                                        "detail": str(error)})
            evaluation.mkdir()
            write_new(evaluation / "jobs.json", {"action_scale": FIXED["action_scale"], "jobs": [
                {"name": arm, "checkpoint": receipts[arm]["checkpoint"]["path"], "output": str(evaluation / "report.json")}]})
            gpu = M.query_gpu(FIXED["gpu_admission_profile"])
            write_new(evaluation / "gpu_before.json", gpu)
            if gpu["busy"]:
                raise RuntimeError("GPU busy before fixed evaluation")
            argv = [str(isaaclab / "_isaac_sim/python.bat"), str(ROOT / "evals/fixed_eval.py"), "--protocol",
                    str(FIXED_EVAL), "--jobs", str(evaluation / "jobs.json"), "--headless"]
            stage = "fixed_evaluation"
            eval_execution = P.run_owned(evaluation, argv, isaaclab, FIXED["eval_watchdog_s"])
            write_new(evaluation / "execution.json", eval_execution)
            if not execution_valid(eval_execution):
                raise ValueError("fixed evaluation did not complete and clean up")
            reports[arm] = C.validate_fixed_report(read(evaluation / "report.json"), {
                "name": arm, "checkpoint_file": f"model_{FINAL_ITERATION}.pt",
                "checkpoint_sha256": receipts[arm]["checkpoint"]["sha256"], "action_scale": FIXED["action_scale"],
                "protocol_sha256_lf": protocol["fixed_eval_sha256_lf"]})
            if reports[arm]["versions"] != protocol["resume_inputs"]["runtime_versions"]:
                raise ValueError("fixed evaluator runtime differs from the bound original")
            stage = "postflight_validation"
            _verified(isaaclab, protocol, commit, "during evaluation")
            if sha(Path(receipts[arm]["checkpoint"]["path"])) != receipts[arm]["checkpoint"]["sha256"]:
                raise ValueError("checkpoint changed during evaluation")
            after = M.query_gpu(FIXED["gpu_admission_profile"])
            write_new(arm_dir / "gpu_after.json", after)
            _owned_gpu_check(after, execution, eval_execution)
            costs[arm] = {"train_parent_wall_s": execution["wall_s"], "eval_parent_wall_s": eval_execution["wall_s"]}
            write_new(arm_dir / "parent.json", {"status": "pass", "training_receipt_sha256": sha(training / "receipt.json"),
                                                "fixed_report_sha256": sha(evaluation / "report.json"),
                                                "budget_checkpoints_sha256": (sha(arm_dir / "budget_checkpoints.json")
                                                                              if (arm_dir / "budget_checkpoints.json").is_file() else None),
                                                **costs[arm]})
            print(json.dumps({"seed": seed, "arm": arm, "status": "validated", **costs[arm]}), flush=True)
        except Exception as error:
            failed_arm = arm
            for store in (reports, initials, receipts, costs):
                store.pop(arm, None)
            write_new(folder / "failure.json", {"arm": arm, "stage": stage, "error_type": type(error).__name__,
                                                "detail": str(error)})
            break
    if failed_arm is None:
        try:
            budget_reports = run_budget_curve(folder, isaaclab, protocol, commit, receipts)
            budget_cost = read(folder / "budget/parent.json")["eval_parent_wall_s"]
        except Exception as error:
            budget_failed = True
            write_new(folder / "budget_failure.json", {"stage": "budget_curve", "error_type": type(error).__name__,
                                                       "detail": str(error)})
    try:
        originals = {"clean_reference": read(source_paths(isaaclab, seed, "NONE")["fixed_report"]),
                     "fault": read(source_paths(isaaclab, seed, FAULT)["fixed_report"])}
        inputs = protocol["resume_inputs"]
        result = summarize(seed, reports, initials, originals, inputs["bands"],
                           inputs["seeds"][str(seed)]["runs"][FAULT]["original_failed"], failed_arm,
                           budget_reports, budget_failed, protocol["fixed_eval_sha256_lf"])
    except (KeyError, ValueError, TypeError, OSError) as error:
        result = invalid_result(seed, error)
    result.update(tag=tag, costs=costs, budget_eval_parent_wall_s=budget_cost, model_calls=0,
                  source_commit_sha=commit, protocol_sha256_lf=request["protocol_sha256_lf"])
    write_new(folder / "result.json", result)
    return result


# --- 공개 묶음 ---
def public_files(completed: list[str], budget_published: bool) -> set[str]:
    files = {"original_reports/clean_reference.json", "original_reports/fault.json", "initial_states.json",
             "training_evidence.json", "source_inputs.json", "result.json"}
    files |= {f"reports/{arm}.json" for arm in completed}
    if budget_published:
        files |= {f"budget_reports/{budget_name(arm, k)}.json" for arm in ARMS for k in BUDGET_CHECKPOINTS}
    return files


def _eval_wall(execution: dict, recorded) -> float:
    """budget 평가 프로세스의 parent wall. v3 공개 비용과 같은 범위·유한·비음수 규칙을 쓴다."""
    value = execution.get("wall_s")
    if (execution.get("wall_scope") != Q.PARENT_WALL_SCOPE or isinstance(value, bool)
            or not isinstance(value, (int, float)) or not 0 <= value < float("inf") or value != recorded):
        raise ValueError("budget curve parent wall time differs from the owned execution record")
    return float(value)


def _publisher_sources() -> dict[str, str]:
    return {relative: sha(ROOT / relative, True) for relative in sorted(PUBLISHER_SOURCES)}


def _public_summary(documents: dict, source: dict, numeric_failed_arm, budget_failed: bool) -> dict:
    completed = [arm for arm in ARMS if f"reports/{arm}.json" in documents]
    budget = None
    if any(name.startswith("budget_reports/") for name in documents):
        budget = {budget_name(arm, k): documents[f"budget_reports/{budget_name(arm, k)}.json"]
                  for arm in ARMS for k in BUDGET_CHECKPOINTS}
    return summarize(source["seed"], {arm: documents[f"reports/{arm}.json"] for arm in completed},
                     documents["initial_states.json"]["arms"],
                     {key: documents[f"original_reports/{key}.json"] for key in ("clean_reference", "fault")},
                     source["bands"], source["runs"][FAULT]["original_failed"], numeric_failed_arm, budget,
                     budget_failed, source["fixed_eval_sha256_lf"])


def _check_public_evidence(documents: dict, completed: list[str]) -> dict:
    """공개 증거의 체크포인트 SHA·중간 체크포인트 기록·비용을 공개 보고서와 대조한다."""
    evidence, initial = documents["training_evidence.json"], documents["initial_states.json"]
    if (set(evidence) != {"schema", "arms", "budget_curve"} or evidence["schema"] != SCHEMA_PUBLIC
            or list(evidence["arms"]) != completed or set(initial) != {"schema", "arms"}
            or initial["schema"] != SCHEMA_PUBLIC or list(initial["arms"]) != completed):
        raise ValueError("public arm evidence schema differs")
    costs = Q4._costs(evidence, completed)
    for arm in completed:
        row = evidence["arms"][arm]
        report_sha = documents[f"reports/{arm}.json"]["checkpoint"]["sha256"]
        marks = row.get("budget_checkpoints")
        if row.get("checkpoint") != {"file": f"model_{FINAL_ITERATION}.pt", "sha256": report_sha}:
            raise ValueError("public arm checkpoint evidence differs from its report")
        if marks is None and "budget_checkpoints" in row and evidence["budget_curve"] is None:
            continue  # 중간 체크포인트 기록이 실패한 arm. budget curve가 없을 때만 허용한다
        if (not isinstance(marks, dict) or set(marks) != {str(k) for k in BUDGET_CHECKPOINTS}
                or any(marks[str(k)].get("file") != f"model_{k}.pt" or marks[str(k)].get("iteration") != k
                       or not V4._sha_text(marks[str(k)].get("sha256")) for k in BUDGET_CHECKPOINTS)):
            raise ValueError("public intermediate checkpoint record differs")
    budget, published = evidence["budget_curve"], any(name.startswith("budget_reports/") for name in documents)
    if budget is None:
        if published:
            raise ValueError("budget reports are published without budget evidence")
        return {"arms": costs, "budget_eval_parent_wall_s": None}
    wall = budget.get("eval_parent_wall_s") if isinstance(budget, dict) else None
    if (not published or completed != list(ARMS) or set(budget) != {"checkpoints", "eval_parent_wall_s"}
            or isinstance(wall, bool) or not isinstance(wall, (int, float)) or not 0 <= wall < float("inf")):
        raise ValueError("public budget curve evidence differs")
    for arm in ARMS:
        for k in BUDGET_CHECKPOINTS:
            name = budget_name(arm, k)
            expected = evidence["arms"][arm]["budget_checkpoints"][str(k)]["sha256"]
            if (budget["checkpoints"].get(name) != {"file": f"model_{k}.pt", "sha256": expected}
                    or documents[f"budget_reports/{name}.json"]["checkpoint"]["sha256"] != expected):
                raise ValueError("public budget report is not bound to its intermediate checkpoint")
    return {"arms": costs, "budget_eval_parent_wall_s": float(wall)}


def export(tag: str, isaaclab: Path) -> dict:
    private = folder_for(tag)
    request = read(private / "request.json")
    commit, seed = request.get("source_commit_sha"), request.get("seed")
    protocol, verified = verify_protocol(isaaclab, commit)
    if (verified != commit or request.get("tag") != tag or seed not in SEEDS
            or request.get("protocol_sha256_lf") != sha(PROTOCOL, True)
            or Path(request.get("isaaclab_root", "")).resolve() != isaaclab.resolve()):
        raise ValueError("request is not bound to the verified experiment source")
    private_result = read(private / "result.json")
    if "completed_arms" not in private_result:
        raise ValueError("seed result has no replayable evidence")
    completed, failed_arm = private_result["completed_arms"], private_result["failed_arm"]
    if completed != list(ARMS[:len(completed)]) or (failed_arm is None) == (private / "failure.json").exists():
        raise ValueError("private result arm completion record differs")
    documents, initials, evidence, receipts, marks = {}, {}, {}, {}, {}
    for arm in completed:
        training, evaluation = private / arm / "train", private / arm / "eval"
        execution = read(training / "execution.json")
        receipts[arm] = receipt = verify_training(private, arm, execution)
        eval_execution = read(evaluation / "execution.json")
        if not execution_valid(eval_execution):
            raise ValueError("fixed evaluation execution is incomplete")
        documents[f"reports/{arm}.json"] = C.validate_fixed_report(read(evaluation / "report.json"), {
            "name": arm, "checkpoint_file": f"model_{FINAL_ITERATION}.pt", "checkpoint_sha256": receipt["checkpoint"]["sha256"],
            "action_scale": FIXED["action_scale"], "protocol_sha256_lf": protocol["fixed_eval_sha256_lf"]})
        parent = read(private / arm / "parent.json")
        if (parent.get("status") != "pass" or parent.get("training_receipt_sha256") != sha(training / "receipt.json")
                or parent.get("fixed_report_sha256") != sha(evaluation / "report.json")):
            raise ValueError("arm parent binding differs")
        marks_path = private / arm / "budget_checkpoints.json"
        if marks_path.is_file():
            if parent.get("budget_checkpoints_sha256") != sha(marks_path):
                raise ValueError("arm parent binding differs")
            marks[arm] = read(marks_path)
            _check_marks(receipt, marks[arm])
        elif parent.get("budget_checkpoints_sha256") is None and (private / arm / "budget_checkpoints_failure.json").is_file():
            marks[arm] = None
        else:
            raise ValueError("intermediate checkpoint record is missing without a recorded failure")
        initials[arm] = Q4._initial_public(read(training / "initial.json"))
        evidence[arm] = {**Q4._receipt_public(receipt, sha(training / "receipt.json")),
                         "parent_wall_s": Q._parent_wall_cost(execution, eval_execution, parent),
                         "budget_checkpoints": marks[arm]}
    budget_failed = (private / "budget_failure.json").exists()
    budget_evidence = None
    if (private / "budget/parent.json").is_file():
        budget = private / "budget"
        parent, checkpoints = read(budget / "parent.json"), read(budget / "checkpoints.json")
        execution = read(budget / "execution.json")
        if any(marks.get(arm) is None for arm in ARMS):
            raise ValueError("budget curve was evaluated without intermediate checkpoint records")
        expected = {budget_name(arm, k): {"path": str(path), "sha256": marks[arm][str(k)]["sha256"]}
                    for arm in ARMS for k, path in _check_marks(receipts[arm], marks[arm]).items()}
        if (parent.get("status") != "pass" or parent.get("checkpoints_sha256") != sha(budget / "checkpoints.json")
                or not execution_valid(execution) or checkpoints != expected):
            raise ValueError("budget curve parent binding differs")
        for name, row in checkpoints.items():
            if parent["reports_sha256"].get(name) != sha(budget / f"{name}.json"):
                raise ValueError("budget report changed after evaluation")
            documents[f"budget_reports/{name}.json"] = C.validate_fixed_report(read(budget / f"{name}.json"), {
                "name": name, "checkpoint_file": Path(row["path"]).name, "checkpoint_sha256": row["sha256"],
                "action_scale": FIXED["action_scale"], "protocol_sha256_lf": protocol["fixed_eval_sha256_lf"]})
        budget_evidence = {"checkpoints": {name: {"file": Path(row["path"]).name, "sha256": row["sha256"]}
                                           for name, row in checkpoints.items()},
                           "eval_parent_wall_s": _eval_wall(execution, parent.get("eval_parent_wall_s"))}
    originals = {"clean_reference": read(source_paths(isaaclab, seed, "NONE")["fixed_report"]),
                 "fault": read(source_paths(isaaclab, seed, FAULT)["fixed_report"])}
    inputs = protocol["resume_inputs"]
    source = {"seed": seed, "fault": FAULT, "bands": deepcopy(inputs["bands"]),
              "runtime_versions": deepcopy(inputs["runtime_versions"]), "summary_sha256_lf": inputs["summary_sha256_lf"],
              "runs": deepcopy(inputs["seeds"][str(seed)]["runs"]), "fixed_eval_sha256_lf": protocol["fixed_eval_sha256_lf"]}
    documents.update({"original_reports/clean_reference.json": originals["clean_reference"],
                      "original_reports/fault.json": originals["fault"],
                      "initial_states.json": {"schema": SCHEMA_PUBLIC, "arms": initials},
                      "training_evidence.json": {"schema": SCHEMA_PUBLIC, "arms": evidence, "budget_curve": budget_evidence},
                      "source_inputs.json": {"schema": SCHEMA_PUBLIC, **source}})
    numeric = _public_summary(documents, source, failed_arm, budget_failed)
    if (any(private_result.get(key) != numeric[key] for key in
            ("status", "seed", "labels", "label_reasons", "completed_arms", "failed_arm", "budget_curve"))
            or private_result.get("tag") != tag or private_result.get("source_commit_sha") != commit):
        raise ValueError("private result differs from replayed evidence")
    publisher_sources = _publisher_sources()
    documents["result.json"] = {"schema": SCHEMA_PUBLIC, "replay_scope": Q.REPLAY_SCOPE, "tag": tag, "seed": seed,
                                "source_commit_sha": commit, "protocol_sha256_lf": request["protocol_sha256_lf"],
                                "source_commit_scope": Q.SOURCE_COMMIT_SCOPE, "budget_curve_failed": budget_failed,
                                "budget_reports_published": budget_evidence is not None,
                                "publisher_sources": publisher_sources, "numeric": numeric}
    if set(documents) != public_files(completed, budget_evidence is not None):
        raise AssertionError("public artifact set differs")
    _check_public_evidence(documents, completed)
    encoded = {name: Q._json_bytes(value) for name, value in documents.items()}
    manifest = {"schema": SCHEMA_PUBLIC, "replay_scope": Q.REPLAY_SCOPE, "publisher_sources": publisher_sources,
                "files": {name: hashlib.sha256(raw).hexdigest() for name, raw in sorted(encoded.items())}}
    for value in [*documents.values(), manifest]:
        Q._public_value(value)
    output = RESULTS / tag
    if output.exists():
        raise FileExistsError("public result folder already exists")
    output.mkdir(parents=True, exist_ok=False)
    for name, raw in [*encoded.items(), ("artifact_manifest.json", Q._json_bytes(manifest))]:
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(raw)
    return {"status": numeric["status"], "seed": seed, "labels": numeric["labels"],
            "folder": f"evals/results/{tag}", "artifact_count": len(encoded)}


def replay(folder: Path) -> dict:
    """공개 파일만으로 라벨·budget curve·규칙 결과와 증거 결합을 다시 계산한다. 실행 프로세스의 신원은 재인증하지 않는다."""
    if folder.is_symlink() or not folder.resolve().is_dir():
        raise ValueError("public result folder is missing or unsafe")
    folder = folder.resolve()
    manifest = Q._read(folder / "artifact_manifest.json")
    files = manifest.get("files")
    if (set(manifest) != {"schema", "replay_scope", "publisher_sources", "files"} or manifest["schema"] != SCHEMA_PUBLIC
            or manifest["replay_scope"] != Q.REPLAY_SCOPE or not isinstance(files, dict)):
        raise ValueError("public artifact manifest differs")
    if manifest["publisher_sources"] != _publisher_sources():
        raise ValueError("publisher source LF-SHA binding differs")
    actual = {path.relative_to(folder).as_posix() for path in folder.rglob("*") if path.is_file()}
    if actual != set(files) | {"artifact_manifest.json"}:
        raise ValueError("public artifact file set differs")
    documents = {}
    for name, expected in files.items():
        path = folder / name
        if path.is_symlink() or sha(path) != expected:
            raise ValueError("public artifact hash differs")
        documents[name] = Q._read(path)
        Q._public_value(documents[name])
    result, source = documents["result.json"], documents["source_inputs.json"]
    numeric = result.get("numeric") or {}
    completed = numeric.get("completed_arms")
    if (result.get("schema") != SCHEMA_PUBLIC or source.get("schema") != SCHEMA_PUBLIC
            or result.get("source_commit_scope") != Q.SOURCE_COMMIT_SCOPE
            or result.get("publisher_sources") != manifest["publisher_sources"] or result.get("seed") != source.get("seed")
            or not isinstance(completed, list) or completed != list(ARMS[:len(completed)])
            or set(files) != public_files(completed, result.get("budget_reports_published") is True)):
        raise ValueError("result and manifest bindings differ")
    costs = _check_public_evidence(documents, completed)
    recomputed = _public_summary(documents, source, numeric.get("failed_arm"), result.get("budget_curve_failed") is True)
    if recomputed != numeric:
        raise ValueError("published result does not replay from public numeric evidence")
    return {"status": numeric["status"], "seed": numeric["seed"], "labels": numeric["labels"],
            "behavior": numeric["behavior"], "budget_curve": numeric["budget_curve"],
            "recovery_mode_rule": numeric["recovery_mode_rule"], "parent_wall_scope": Q.PARENT_WALL_SCOPE,
            "parent_wall_s": costs, "replay_scope": Q.REPLAY_SCOPE}


def aggregate_public(folders: list[Path]) -> dict:
    rows = []
    for folder in folders:
        replay(folder)
        result = Q._read(folder.resolve() / "result.json")
        rows.append({**result["numeric"], "protocol_sha256_lf": result["protocol_sha256_lf"]})
    return {**aggregate(rows), "replay_scope": Q.REPLAY_SCOPE}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify", "execute", "export", "replay", "aggregate", "train-child"))
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--isaaclab", type=Path, default=Path.home() / "IsaacLab")
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--tag")
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args(argv)
    if args.command == "train-child":
        if args.manifest is None:
            parser.error("train-child requires --manifest")
        R.execute(args.manifest, sys.modules[__name__])
        return 0
    if args.command == "prepare" and args.contract is None:
        parser.error("prepare requires --contract from the SDK Python CPU checker")
    if args.command == "execute" and (args.tag is None or args.seed is None):
        parser.error("execute requires --tag and --seed")
    if args.command in ("export", "replay") and len(args.paths) != 1:
        parser.error(f"{args.command} takes exactly one argument")
    if args.command == "aggregate" and not args.paths:
        parser.error("aggregate takes one public result folder per seed")
    commands = {
        "prepare": lambda: prepare(args.isaaclab, args.contract),
        "verify": lambda: {"status": "pass", "source_commit_sha": verify_protocol(args.isaaclab)[1]},
        "execute": lambda: execute(args.tag, args.seed, args.isaaclab),
        "export": lambda: export(str(args.paths[0]), args.isaaclab),
        "replay": lambda: replay(args.paths[0]),
        "aggregate": lambda: aggregate_public(args.paths),
    }
    try:
        result = commands[args.command]()
    except subprocess.CalledProcessError:
        print("거부: 프로토콜이나 결합 파일이 아직 커밋되지 않았거나 Git 확인에 실패했다", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 1 if result.get("status") == "invalid" else 0


if __name__ == "__main__":
    raise SystemExit(main())
