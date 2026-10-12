"""보상 결함 교정 뒤 회복 방식(이어 학습·탐색 재설정 이어 학습·처음부터 재학습)을 고정 행동 평가로 비교한다.

호스트가 고른 합성 결함 F1의 개발 실험이다. 이미 본 seed 2026·2028을 쓰며 모델 선택, 진단 정확도,
일반화, GPU 절감의 평가가 아니다. v3 실행 코드(run_training_correction.py, train_with_correction.py)는
공개 결과 재생의 소스 결합을 지키려고 바꾸지 않고 함수만 재사용한다.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "bench", ROOT / "evals"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import run_training_correction as V3  # noqa: E402
import record_evidence_media as M  # noqa: E402
import run_runtime_readback_smoke as P  # noqa: E402
import runtime_readback_smoke as S  # noqa: E402
import train_with_recovery_mode as R  # noqa: E402
from rl_triage import grounded_recovery as C  # noqa: E402
from rl_triage import recovery as RC  # noqa: E402

read = V3.read
write_new = V3.write_new
sha = V3.sha
digest = V3.digest
load_config = V3.load_config
execution_valid = V3.execution_valid
validate_readback = V3.validate_readback
check_contract = V3.check_contract

PROTOCOL = ROOT / "bench/protocols/training_correction_v4.json"
V3_PROTOCOL = V3.PROTOCOL
FIXED_EVAL = V3.FIXED_EVAL
PRIVATE = ROOT / "bench/private/training_correction_v4"
ORIGINAL = V3.ORIGINAL
V3_PUBLIC = ROOT / "evals/results/reward_resume_dev3_20261013"
SEEDS = (2026, 2028)
FAULT = "F1"
CORRECTION = "restore_body_frame_reward"
INPUT_RUNS = {str(seed): {"NONE": f"baseline_p0c_s{seed}", "F1": f"h02_s{seed}"} for seed in SEEDS}
ARMS = ("clean_resume_reference", "resume_corrected", "resume_corrected_std_reset", "restart_corrected")
TREATMENT_ARMS = ARMS[1:]
RESUME_TREATMENTS = ("resume_corrected", "resume_corrected_std_reset")
ARM_SPECS = {
    "clean_resume_reference": {"recovery_mode": "resume", "fault": "NONE", "correction": None,
                               "role": "resume_reference"},
    "resume_corrected": {"recovery_mode": "resume", "fault": FAULT, "correction": CORRECTION,
                         "role": "treatment", "primary_reference": "clean_resume_reference"},
    "resume_corrected_std_reset": {"recovery_mode": "resume_std_reset", "fault": FAULT, "correction": CORRECTION,
                                   "role": "treatment", "primary_reference": "clean_resume_reference"},
    "restart_corrected": {"recovery_mode": "restart", "fault": FAULT, "correction": CORRECTION,
                          "role": "treatment", "primary_reference": "original_none",
                          "sham_counterpart": "original_fault_reused"},
}
FINAL_ITERATION = {"resume": 598, "resume_std_reset": 598, "restart": 299}
LABELS = ("recovered", "not_recovered", "invalid")
PREDICTIONS = {"resume_corrected": "not_recovered", "resume_corrected_std_reset": None,
               "restart_corrected": "recovered"}
RULE_THRESHOLD = 0.9
RULE_ARMS = {"restart": "restart_corrected", "resume": "resume_corrected"}
STANDSTILL_PREDICTION = {"original_fault_min": 0.9, "original_none_max": 0.3}
CODE = V3.CODE | {"evals/run_training_correction_v4.py", "bench/train_with_recovery_mode.py"}
STATUS_SEED = ("complete", "partial", "invalid")
FIXED = {
    "schema": "training_correction_protocol_v2",
    "revision": 4,
    "question": "보상 결함을 고친 뒤 이어 학습·탐색 재설정 이어 학습·처음부터 재학습 중 무엇이 같은 300회 예산으로 정상 범위에 돌아오는가",
    "scope": "seen_two_seed_reward_fault_recovery_mode_development",
    "previous_protocol": {
        "path": "bench/protocols/training_correction_v3.json",
        "protocol_sha256_lf": "abff34132c0f2fbbcfde9d4dee44c4cbd97871fb370cdf7c37d8584882d761d7",
        "tag": "reward_resume_dev3_20261013", "case": "case_01", "status": "no_recovery",
    },
    "excluded_seeds": {"2027": "v3 stop rule: no semantic retry or budget increase after no_recovery"},
    "task": C.TASK, "seeds": list(SEEDS), "num_envs": 4096, "max_iterations": 300,
    "num_steps_per_env": 24, "action_scale": 0.25, "resume_iteration": 299,
    "final_iteration_by_mode": FINAL_ITERATION,
    "fault": FAULT, "fault_mechanism": "reward", "correction": CORRECTION,
    "input_runs": INPUT_RUNS, "source_cohort_already_seen": True,
    "arms": list(ARMS), "arm_specs": ARM_SPECS,
    "recovery_modes": {
        "resume": "실패 체크포인트의 정책과 optimizer 상태를 불러와 같은 seed로 300회 이어 학습한다",
        "resume_std_reset": "resume과 같고, 체크포인트를 불러와 지문을 검증한 직후 정책 action noise std만 agent 설정의 init_noise_std로 채우고 그 파라미터의 Adam 모멘트를 지운다. 다른 파라미터의 optimizer 상태와 param_groups(lr 포함)는 재설정 시점에 바꾸지 않는다",
        "restart": "체크포인트 없이 같은 seed로 처음부터 300회 학습한다. 환경 생성 전 결함 주입 뒤 교정 하나를 복원하는 hook 순서는 v3와 같다",
    },
    "std_reset": {
        "parameter": "std", "noise_std_type": "scalar",
        "value_source": "source_agent_config.policy.init_noise_std",
        "timing": "after_checkpoint_load_and_state_verification_before_learn",
        "method": "in_place_fill_under_no_grad_same_parameter_object",
        "optimizer_state": "std_adam_moments_removed_other_parameter_states_and_param_groups_preserved",
        "readback": "before_after_values_std_mean_std_moment_presence_and_state_digests",
    },
    "known_confounders": [
        "RSL-RL 2.3.3 PPO는 resume 뒤 learning_rate를 cfg 값에서 다시 시작하고 adaptive schedule이 첫 update에서 optimizer lr을 덮어쓴다(rsl_rl #221, 상류 v5.4.2 수정). 모든 resume arm에 같게 적용되어 arm 간 차이를 만들지 않는다",
        "actor·critic·std가 Adam 하나와 전역 clip_grad_norm(max_grad_norm 1.0)을 공유한다",
    ],
    "std_reset_prior_note": "v3 학습 로그에서 F1 교정·대조 arm의 std는 0.38~0.55로 정상 기준 0.27~0.36보다 높았다. std가 낮아 탐색이 부족하다는 설명은 약하므로 이 arm은 예측 없는 탐색 arm이다",
    "references": {
        "resume_primary": "same_seed_clean_reference_resumed_same_updates",
        "resume_secondary": "original_same_seed_clean_checkpoint",
        "restart_primary": "original_same_seed_none_300_iterations_from_scratch_existing_fixed_eval",
        "restart_sham_counterpart": "original_same_seed_f1_300_iterations_from_scratch_existing_fixed_eval_reused_not_rerun",
        "reference_validity": "기준 정책은 고정 평가에서 제자리 대비 걷고(WALK_FRACTION) 낙상률 REF_FALL_MAX 이하여야 한다. 이어 학습 기준은 원본 정상 정책 대비 healthy여야 한다",
    },
    "restart_initial_state_binding": "seeded_fresh_initialization_recorded_original_initial_state_unrecorded",
    "resume_scope": "policy_optimizer_warm_start_not_environment_or_rng_restore",
    "require_resume_treatment_initial_match": True,
    "predictions": PREDICTIONS,
    "prediction_wording": "restart_corrected는 두 seed 모두 recovered, resume_corrected는 두 seed 모두 not_recovered, resume_corrected_std_reset은 예측 없음(탐색)",
    "labels": list(LABELS),
    "label_rule": "treatment arm의 고정 평가가 등록 기준 대비 healthy면 recovered, unhealthy면 not_recovered, 기준 무효·실행 실패·미실행·그 밖은 invalid",
    "mechanical_signal": {
        "name": "standstill_ratio",
        "definition": "metrics.lin_vel_rmse_mps / standstill.lin_vel_rmse_mps in the same fixed_eval_v1 report",
        "new_measurement": False,
        "predictions": STANDSTILL_PREDICTION,
        "used_for_labels": False,
    },
    "recovery_mode_rule": {
        "rule": "fault_mechanism == reward and original_fault_standstill_ratio >= 0.9 -> restart, otherwise resume",
        "threshold": RULE_THRESHOLD, "arms": RULE_ARMS,
        "outcomes": {
            "hit": "권고 arm recovered이고 대안 arm not_recovered",
            "both_recovered": "권고 arm과 대안 arm 모두 recovered",
            "recommended_not_recovered": "권고 arm이 not_recovered",
            "undetermined": "권고 또는 대안 arm이 invalid",
        },
        "derived_from": "seed 2027 단일 관찰(v3 reward_resume_dev3_20261013 no_recovery)",
        "used_for_labels": False,
    },
    "python_runtime": V3.FIXED["python_runtime"], "torch_version": V3.FIXED["torch_version"],
    "bands_source": V3.FIXED["bands_source"],
    "allowed_agent_config_changes": {
        "resume": ["run_name", "resume", "load_run", "load_checkpoint", "max_iterations"],
        "resume_std_reset": ["run_name", "resume", "load_run", "load_checkpoint", "max_iterations"],
        "restart": ["run_name"],
    },
    "train_watchdog_s": 1800, "eval_watchdog_s": 360, "semantic_retries": 0,
    "train_limit_per_arm": 1, "eval_limit_per_arm": 1,
    "execution_unit": "one_seed_four_arms_in_registered_order_stop_on_first_failure",
    "gpu_admission_profile": V3.FIXED["gpu_admission_profile"],
    "isaaclab_commit": V3.FIXED["isaaclab_commit"], "rsl_rl_version": V3.FIXED["rsl_rl_version"],
    "cpu_evidence_reuse": "v1~v3와 같은 교정 표적의 CPU 행렬을 SHA로 재결합한다",
    "approval_actor": "claude_operator_user_authorized_test",
    "model_calls": 0, "model_selected_correction": False,
    "result_wording": {
        "recovered": "보상 결함을 고친 뒤 이 회복 방식으로 300회 학습한 단일 실행이 등록 기준 대비 고정 행동 허용 범위를 통과했다. 이미 본 seed의 합성 결함 개발 결과이며 일반화·AI 교정 선택·GPU 절감을 주장하지 않는다.",
        "not_recovered": "보상 결함을 고친 뒤 이 회복 방식으로 300회 학습한 단일 실행이 등록 기준 대비 고정 행동 허용 범위를 통과하지 못했다. 이미 본 seed의 합성 결함 개발 결과이며 일반화·AI 교정 선택·GPU 절감을 주장하지 않는다.",
        "invalid": "출처·개입·초기 상태·예산·평가·종료 계약이나 기준 정책 검증이 맞지 않거나 실행되지 않아 이 arm의 회복 판정을 보류한다.",
        "seed_summary": "이미 본 seed의 합성 결함(F1) 개발 결과다. arm마다 단일 실행이며 일반화·AI 교정 선택·GPU 절감 주장이 아니다.",
        "standstill_ratio": "standstill_ratio는 같은 고정 평가의 선속도 RMSE를 제자리 정책의 RMSE로 나눈 값이다. 1에 가까우면 명령 추종이 없다는 뜻이며 정지 자체를 증명하지 않는다.",
        "recovery_mode_rule": "seed 2027 단일 관찰에서 만든 규칙이다. v4 결과로 적중 여부만 기록하며 판정 라벨에 쓰지 않는다.",
    },
}
DYNAMIC = {"source_files", "sdk_sources", "rsl_sources", "correction_source_contract",
           "fixed_eval_sha256_lf", "resume_inputs", "rule_derivation", "cpu_checks"}


def standstill_ratio(report: dict) -> float:
    """같은 고정 평가 report 안의 선속도 RMSE ÷ 제자리 정책 RMSE. 정지가 아니라 추종 없음의 신호다."""
    return report["metrics"]["lin_vel_rmse_mps"] / report["standstill"]["lin_vel_rmse_mps"]


def walks(report: dict) -> bool:
    behavior = C._behavior(report)
    return (behavior["err_xy"] < C.WALK_FRACTION * report["standstill"]["lin_vel_rmse_mps"]
            and behavior["fall_frac"] <= C.REF_FALL_MAX)


def label_from(verdict: dict) -> str:
    return {"healthy": "recovered", "unhealthy": "not_recovered"}.get(verdict["label"], "invalid")


def rule_recommendation(mechanism: str, ratio: float) -> str:
    return "restart" if mechanism == "reward" and ratio >= RULE_THRESHOLD else "resume"


def rule_outcome(recommended: str, labels: dict) -> dict:
    alternative = "resume" if recommended == "restart" else "restart"
    chosen, other = labels[RULE_ARMS[recommended]], labels[RULE_ARMS[alternative]]
    if "invalid" in (chosen, other):
        outcome = "undetermined"
    elif chosen == "recovered" and other == "not_recovered":
        outcome = "hit"
    elif chosen == "recovered":
        outcome = "both_recovered"
    else:
        outcome = "recommended_not_recovered"
    return {"recommended": recommended, "recommended_arm": RULE_ARMS[recommended],
            "recommended_label": chosen, "alternative_arm": RULE_ARMS[alternative],
            "alternative_label": other, "outcome": outcome,
            "rule_hit": None if outcome == "undetermined" else outcome == "hit"}


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
    matches = [p.resolve() for p in log_root.iterdir() if p.is_dir() and re.match(run_regex, p.name)]
    if matches != [run_dir]:
        raise ValueError("source run must match exactly one directory")
    return {"metadata": meta_path, "checkpoint": run_dir / "model_299.pt",
            "environment": run_dir / "params/env.yaml", "agent": run_dir / "params/agent.yaml",
            "fixed_report": ORIGINAL / "runs" / f"{name}.json"}


def _original_report(report: dict, name: str, checkpoint_sha: str) -> dict:
    return C.validate_fixed_report(report, {
        "name": name, "checkpoint_file": "model_299.pt", "checkpoint_sha256": checkpoint_sha,
        "action_scale": FIXED["action_scale"], "protocol_sha256_lf": sha(FIXED_EVAL, True),
    })


def source_inputs(isaaclab: Path) -> dict:
    """두 seed의 기존 NONE·F1 입력의 SHA, 원본 판정, 사전 신호를 묶는다. 새 평가 결과로 세지 않는다."""
    summary = read(ORIGINAL / "summary.json")
    bands = summary["bands"]
    if set(bands) != {"survival_ratio_min", "err_xy_ratio_max", "err_yaw_ratio_max", "fall_frac_max"}:
        raise ValueError("original behavior bands differ")
    seeds, versions, std_values = {}, None, set()
    for seed in SEEDS:
        runs, reports = {}, {}
        for fault, name in INPUT_RUNS[str(seed)].items():
            paths = source_paths(isaaclab, seed, fault)
            hashes = {key: sha(path) for key, path in paths.items()}
            report = _original_report(read(paths["fixed_report"]), name, hashes["checkpoint"])
            rows = [row for row in summary["runs"] if row["run"] == name]
            if len(rows) != 1:
                raise ValueError("original summary run is missing or duplicated")
            row = rows[0]
            if row["metrics"] != report["metrics"] or row["checkpoint_sha256"] != hashes["checkpoint"]:
                raise ValueError("original summary differs from the raw report")
            agent = load_config(paths["agent"])
            policy = agent.get("policy", {})
            if (agent.get("seed") != seed or policy.get("noise_std_type") != "scalar"
                    or isinstance(policy.get("init_noise_std"), bool)
                    or not isinstance(policy.get("init_noise_std"), (int, float))):
                raise ValueError("original agent seed or action noise configuration differs")
            std_values.add(float(policy["init_noise_std"]))
            runs[fault] = {"run": name, "hashes": hashes, "original_label": row["fixed_label"],
                           "original_failed": row["fixed_failed"], "standstill_ratio": standstill_ratio(report)}
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
        fault_ratio, none_ratio = runs[FAULT]["standstill_ratio"], runs["NONE"]["standstill_ratio"]
        seeds[str(seed)] = {
            "runs": runs,
            "original_none_walks": walks(reports["NONE"]),
            "standstill_prediction_met": {
                "original_fault": fault_ratio >= STANDSTILL_PREDICTION["original_fault_min"],
                "original_none": none_ratio <= STANDSTILL_PREDICTION["original_none_max"],
            },
            "rule_recommendation": rule_recommendation(FIXED["fault_mechanism"], fault_ratio),
        }
    if len(std_values) != 1:
        raise ValueError("source agents disagree on init_noise_std")
    return {"summary_sha256_lf": sha(ORIGINAL / "summary.json", True), "bands": bands,
            "seeds": seeds, "runtime_versions": versions, "init_noise_std": std_values.pop()}


def rule_derivation() -> dict:
    """규칙을 만든 seed 2027 관찰을 기존 원본·v3 공개 report에서 다시 계산한다."""
    files = {
        "original_fault": ORIGINAL / "runs/h02_s2027.json",
        "original_none": ORIGINAL / "runs/baseline_p0c_s2027.json",
        "v3_clean_reference": V3_PUBLIC / "reports/clean_reference.json",
        "v3_fault_sham": V3_PUBLIC / "reports/fault_sham.json",
        "v3_corrected": V3_PUBLIC / "reports/corrected.json",
    }
    return {"seed": 2027, "v3_status": "no_recovery",
            "files": {key: {"path": path.relative_to(ROOT).as_posix(), "sha256_lf": sha(path, True)}
                      for key, path in files.items()},
            "standstill_ratio": {key: standstill_ratio(read(path)) for key, path in files.items()}}


def source_maps(isaaclab: Path) -> dict:
    rsl = V3.rsl_root(isaaclab)
    names = {p.relative_to(rsl).as_posix() for p in rsl.rglob("*.py")}
    if not {"runners/on_policy_runner.py", "algorithms/ppo.py", "modules/actor_critic.py"} <= names:
        raise ValueError("RSL-RL source is missing")
    return {
        "source_files": {name: sha(ROOT / name, True) for name in sorted(CODE)},
        "sdk_sources": {name: sha(isaaclab / name, True) for name in sorted(V3.SDK_FILES)},
        "rsl_sources": {name: sha(rsl / name, True) for name in sorted(names)},
    }


def prepare(isaaclab: Path, contract_path: Path) -> dict:
    contract = read(contract_path)
    check_contract(contract, isaaclab)
    document = {**deepcopy(FIXED), **source_maps(isaaclab), "correction_source_contract": contract,
                "fixed_eval_sha256_lf": sha(FIXED_EVAL, True), "resume_inputs": source_inputs(isaaclab),
                "rule_derivation": rule_derivation(),
                "cpu_checks": V3.cpu_checks(contract_path.parent, contract, isaaclab)}
    write_new(PROTOCOL, document)
    return {"status": "prepared_commit_before_execution", "path": str(PROTOCOL)}


def verify_protocol(isaaclab: Path) -> tuple[dict, str]:
    protocol = read(PROTOCOL)
    if set(protocol) != set(FIXED) | DYNAMIC:
        raise ValueError("protocol fields differ")
    if any(type(protocol[k]) is not type(v) or protocol[k] != v for k, v in FIXED.items()):
        raise ValueError("fixed training contract changed")
    if sha(V3_PROTOCOL, True) != FIXED["previous_protocol"]["protocol_sha256_lf"]:
        raise ValueError("previous v3 protocol changed")
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
    if protocol["rule_derivation"] != rule_derivation():
        raise ValueError("recovery mode rule derivation changed")
    if S._git_head(isaaclab) != FIXED["isaaclab_commit"] or not S._git_tracked_clean(isaaclab):
        raise ValueError("Isaac Lab checkout differs")
    commit = P.verify_committed(ROOT, CODE | {PROTOCOL.relative_to(ROOT).as_posix(),
                                              FIXED_EVAL.relative_to(ROOT).as_posix()})
    return protocol, commit


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
    inputs = read(PROTOCOL)["resume_inputs"]
    bound = inputs["seeds"][str(seed)]["runs"][fault]["hashes"]
    resumed = mode != "restart"
    checkpoint = paths["checkpoint"].resolve()
    return {
        "schema": "training_correction_manifest_v2", "tag": folder.name, "seed": seed, "arm": arm,
        "recovery_mode": mode, "source_commit_sha": request["source_commit_sha"],
        "protocol_sha256_lf": request["protocol_sha256_lf"], "isaaclab_root": request["isaaclab_root"],
        "protocol_path": str(PROTOCOL.resolve()), "fault": fault, "correction_id": correction,
        **{key: FIXED[key] for key in ("num_envs", "max_iterations", "task")},
        "run_name": f"{folder.name}_{arm}", "output_dir": str((folder / arm / "train").resolve()),
        "final_iteration": FINAL_ITERATION[mode],
        "resume_checkpoint_path": str(checkpoint) if resumed else None,
        "resume_checkpoint_sha256": bound["checkpoint"] if resumed else None,
        "resume_iteration": FIXED["resume_iteration"] if resumed else None,
        "load_run_regex": "^" + re.escape(checkpoint.parent.name) + "$" if resumed else None,
        "checkpoint_regex": "^model_299\\.pt$" if resumed else None,
        "source_env_config_path": str(paths["environment"].resolve()),
        "source_agent_config_path": str(paths["agent"].resolve()),
        "config_source_fault": fault,
        "init_noise_std": inputs["init_noise_std"] if mode == "resume_std_reset" else None,
    }


def validate_training_configuration(log_dir: Path, manifest: dict) -> None:
    """원 설정과 학습 시작 직전 설정의 차이를 회복 방식별로 허용된 인자로 제한한다."""
    protocol = read(Path(manifest["protocol_path"]))
    mode = manifest["recovery_mode"]
    hashes = protocol["resume_inputs"]["seeds"][str(manifest["seed"])]["runs"][manifest["config_source_fault"]]["hashes"]
    source_env = Path(manifest["source_env_config_path"])
    source_agent = Path(manifest["source_agent_config_path"])
    for path, key in ((source_env, "environment"), (source_agent, "agent")):
        if sha(path) != hashes[key]:
            raise ValueError("source configuration changed before learning")
    original_env, original_agent = load_config(source_env), load_config(source_agent)
    current_env = load_config(Path(log_dir) / "params/env.yaml")
    current_agent = load_config(Path(log_dir) / "params/agent.yaml")
    if current_env != original_env:
        raise ValueError("environment configuration changed outside the registered code correction")
    allowed = set(FIXED["allowed_agent_config_changes"][mode])
    if ({k: v for k, v in original_agent.items() if k not in allowed}
            != {k: v for k, v in current_agent.items() if k not in allowed}):
        raise ValueError("agent configuration changed outside the recovery mode arguments")
    expected = {"run_name": manifest["run_name"]}
    if mode != "restart":
        expected.update({"resume": True, "load_run": manifest["load_run_regex"],
                         "load_checkpoint": manifest["checkpoint_regex"],
                         "max_iterations": manifest["max_iterations"]})
    if any(type(current_agent.get(k)) is not type(v) or current_agent[k] != v for k, v in expected.items()):
        raise ValueError("agent configuration differs from the recovery mode manifest")
    if (original_agent.get("seed") != manifest["seed"]
            or original_agent.get("num_steps_per_env") != FIXED["num_steps_per_env"]
            or original_agent.get("max_iterations") != FIXED["max_iterations"]
            or original_env.get("seed") != manifest["seed"]
            or original_env.get("scene", {}).get("num_envs") != FIXED["num_envs"]):
        raise ValueError("original training seed, environment count, or step count differs")
    if mode == "resume_std_reset" and original_agent.get("policy", {}).get("init_noise_std") != manifest["init_noise_std"]:
        raise ValueError("std reset value differs from the source agent configuration")


def validate_manifest(path: Path) -> tuple[dict, dict]:
    manifest = read(path)
    if set(manifest) != R.MANIFEST_FIELDS:
        raise ValueError("manifest fields differ from the fixed schema")
    folder = folder_for(manifest["tag"])
    expected_path = folder / manifest["arm"] / "train/manifest.json"
    if path.is_symlink() or path.resolve() != expected_path.resolve():
        raise ValueError("manifest is outside its registered arm")
    request = read(folder / "request.json")
    protocol, commit = verify_protocol(Path(request["isaaclab_root"]))
    if request["source_commit_sha"] != commit or request["protocol_sha256_lf"] != sha(PROTOCOL, True):
        raise ValueError("request source commitment changed")
    if manifest != manifest_for(folder, request, manifest["arm"]):
        raise ValueError("manifest differs from the approved request")
    approval = read(folder / "approval.json")
    consumed = read(folder / manifest["arm"] / "consumed.json")
    if approval != {"request_sha256": sha(folder / "request.json"), "actor": FIXED["approval_actor"], "identity_attested": False}:
        raise ValueError("operator authorization record differs")
    if consumed != {"approval_sha256": sha(folder / "approval.json"), "arm": manifest["arm"], "train_limit": 1, "eval_limit": 1}:
        raise ValueError("arm consumption record differs")
    return manifest, protocol


STD_RESET_KEYS = {
    "schema", "parameter", "noise_std_type", "init_noise_std", "parameter_target_float32", "before", "after",
    "after_matches_init", "std_mean_before", "std_mean_after", "policy_state_before_reset",
    "policy_state_after_reset", "changed_tensor_paths", "parameter_in_optimizer", "optimizer_param_index",
    "optimizer_std_moment_present_before", "optimizer_std_moment_present_after",
    "optimizer_state_before_reset", "optimizer_state_expected_after_reset", "optimizer_state_after_reset",
    "optimizer_other_state_preserved",
}


def validate_std_reset(readback, source_state: dict, initial: dict, init_noise_std: float) -> None:
    """std만 등록값으로 바뀌고 optimizer는 std의 Adam 모멘트만 빠진 원 체크포인트 상태인지 확인한다."""
    if not isinstance(readback, dict) or set(readback) != STD_RESET_KEYS:
        raise ValueError("std reset readback is missing or incomplete")
    target = R.float32(init_noise_std)
    before, after = readback["before"], readback["after"]
    if (readback["schema"] != "action_std_reset_v2" or readback["parameter"] != "std"
            or readback["noise_std_type"] != FIXED["std_reset"]["noise_std_type"]
            or readback["init_noise_std"] != init_noise_std
            or readback["parameter_target_float32"] != target or readback["after_matches_init"] is not True
            or readback["parameter_in_optimizer"] is not True
            or readback["optimizer_other_state_preserved"] is not True
            or isinstance(readback["optimizer_param_index"], bool)
            or not isinstance(readback["optimizer_param_index"], int)
            or not isinstance(before, list) or not isinstance(after, list) or not after
            or len(before) != len(after) or any(value != target for value in after)
            or readback["std_mean_before"] != sum(before) / len(before)
            or readback["std_mean_after"] != sum(after) / len(after)):
        raise ValueError("std reset readback differs from the registered initial value")
    if (readback["policy_state_before_reset"] != source_state["policy_state_sha256"]
            or readback["policy_state_after_reset"] != initial["policy_state"]["sha256"]):
        raise ValueError("std reset is not bound to the loaded checkpoint and initial policy")
    moment_removed = readback["optimizer_std_moment_present_before"]
    if (moment_removed not in (True, False) or readback["optimizer_std_moment_present_after"] is not False
            or readback["optimizer_state_before_reset"] != source_state["optimizer_state_sha256"]
            or readback["optimizer_state_after_reset"] != readback["optimizer_state_expected_after_reset"]
            or readback["optimizer_state_after_reset"] != initial["optimizer_state"]["sha256"]
            or (readback["optimizer_state_before_reset"] != readback["optimizer_state_after_reset"]) != moment_removed):
        raise ValueError("std reset changed optimizer state beyond the std Adam moments")
    if readback["changed_tensor_paths"] != ([R.STD_PATH] if before != after else []):
        raise ValueError("std reset changed a tensor other than the action std")


def _sha_text(value) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _rng_ok(rng) -> bool:
    return (isinstance(rng, dict) and set(rng) == {"cpu_sha256", "cuda_all_sha256", "cuda_device_count"}
            and _sha_text(rng["cpu_sha256"]) and isinstance(rng["cuda_all_sha256"], list)
            and rng["cuda_all_sha256"] and rng["cuda_device_count"] == len(rng["cuda_all_sha256"])
            and all(_sha_text(value) for value in rng["cuda_all_sha256"]))


def verify_training(folder: Path, arm: str, execution: dict) -> dict:
    training = folder / arm / "train"
    if not execution_valid(execution):
        raise ValueError(f"training execution failed (exit={execution.get('exit_code')}); inspect its stderr.log")
    manifest, protocol = validate_manifest(training / "manifest.json")
    mode = R.validate_manifest_mode(manifest)
    resumed = mode != "restart"
    receipt = read(training / "receipt.json")
    initial = read(training / "initial.json")
    if (receipt.get("completed") is not True or receipt.get("schema") != "training_correction_receipt_v2"
            or initial.get("schema") != "training_correction_initial_v2" or initial.get("recovery_mode") != mode):
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
            or receipt.get("aggregate_env_steps") != steps * FIXED["num_envs"]):
        raise ValueError("training budget or environment count differs")
    if receipt["budget"] != {"seed": manifest["seed"], "num_envs": FIXED["num_envs"],
                             "max_iterations": FIXED["max_iterations"],
                             "num_steps_per_env": FIXED["num_steps_per_env"], "expected_control_steps": steps}:
        raise ValueError("training budget receipt differs")
    final = FINAL_ITERATION[mode]
    if receipt["final_iteration"] != final:
        raise ValueError("final iteration differs from the recovery mode")
    source_state = receipt.get("source_checkpoint_state")
    if resumed:
        expected_resume = {"path": manifest["resume_checkpoint_path"], "sha256": manifest["resume_checkpoint_sha256"]}
        if (receipt.get("resume_checkpoint") != expected_resume
                or initial.get("resume_checkpoint") != {**expected_resume, "load_optimizer": True}
                or receipt.get("checkpoint_load_calls") != 1):
            raise ValueError("training did not resume the bound policy and optimizer")
        if (receipt.get("resume_iteration") != FIXED["resume_iteration"]
                or initial.get("resume_iteration") != FIXED["resume_iteration"]
                or receipt.get("start_iteration") != FIXED["resume_iteration"]
                or initial.get("start_iteration") != FIXED["resume_iteration"]):
            raise ValueError("resume iteration differs")
        if (not isinstance(source_state, dict) or source_state != initial.get("source_checkpoint_state")
                or source_state.get("iteration") != FIXED["resume_iteration"]):
            raise ValueError("source checkpoint state proof differs")
        if mode == "resume":
            if (initial.get("std_reset") is not None or receipt.get("std_reset_applied") is not False
                    or source_state.get("policy_state_sha256") != initial["policy_state"]["sha256"]
                    or source_state.get("optimizer_state_sha256") != initial["optimizer_state"]["sha256"]):
                raise ValueError("initial policy/optimizer differs from the source checkpoint")
        else:
            if receipt.get("std_reset_applied") is not True:
                raise ValueError("std reset arm did not record its reset")
            validate_std_reset(initial.get("std_reset"), source_state, initial, manifest["init_noise_std"])
        proofs = ((source_state, FIXED["resume_iteration"]), (receipt["final_checkpoint_state"], final))
    else:
        if (receipt.get("resume_checkpoint") is not None or initial.get("resume_checkpoint") is not None
                or receipt.get("resume_iteration") is not None or initial.get("resume_iteration") is not None
                or source_state is not None or initial.get("source_checkpoint_state") is not None
                or receipt.get("checkpoint_load_calls") != 0 or initial.get("std_reset") is not None
                or receipt.get("std_reset_applied") is not False
                or receipt.get("start_iteration") != 0 or initial.get("start_iteration") != 0):
            raise ValueError("restart arm loaded or reset a checkpoint")
        proofs = ((receipt["final_checkpoint_state"], final),)
    for state, iteration in proofs:
        if state.get("iteration") != iteration or any(not _sha_text(state.get(k))
                                                      for k in ("policy_state_sha256", "optimizer_state_sha256")):
            raise ValueError("checkpoint state proof is incomplete")
    resume_scope = {"policy_optimizer_restored": resumed, "full_environment_state_restored": False,
                    "rng_state_restored_from_checkpoint": False, "fresh_initialization": not resumed,
                    "action_std_reset": mode == "resume_std_reset"}
    if (receipt.get("resume_scope") != resume_scope or initial.get("init_at_random_ep_len") is not True
            or initial.get("full_environment_state_restored") is not False
            or initial.get("rng_state_restored_from_checkpoint") is not False):
        raise ValueError("resume scope or episode initialization differs")
    if receipt.get("rng_state_before_learn") != initial.get("rng_state_before_learn") or not _rng_ok(initial.get("rng_state_before_learn")):
        raise ValueError("training RNG observation differs")
    observed = {"environment_seed": manifest["seed"], "environment_num_envs": FIXED["num_envs"],
                "action_scale": FIXED["action_scale"], "agent_seed": manifest["seed"],
                "agent_max_iterations": FIXED["max_iterations"], "runner_num_steps_per_env": FIXED["num_steps_per_env"]}
    if receipt.get("observed") != observed or initial.get("observed") != observed or receipt.get("actual_action_scale") != FIXED["action_scale"]:
        raise ValueError("actual training configuration differs")
    versions = receipt["runtime_versions"]
    runtime = protocol["correction_source_contract"]["runtime"]
    if (versions != initial.get("runtime_versions")
            or versions.get("python") != ".".join(map(str, runtime["version"]))
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
    if checkpoint.resolve().parent.parent != log_root or not checkpoint.parent.name.endswith("_" + manifest["run_name"]):
        raise ValueError("training checkpoint is outside its exact run")
    if checkpoint.name != f"model_{final}.pt":
        raise ValueError("training selected a different checkpoint")
    if Path(initial["log_dir"]).resolve() != checkpoint.parent.resolve():
        raise ValueError("initial state came from a different run")
    if set(receipt["configs"]) != {"environment", "agent"}:
        raise ValueError("training configuration artifacts are incomplete")
    for key, filename in (("environment", "env.yaml"), ("agent", "agent.yaml")):
        if Path(receipt["configs"][key]["path"]).resolve() != (checkpoint.parent / "params" / filename).resolve():
            raise ValueError("training config path is outside its exact run")
    for artifact in [receipt["checkpoint"], *receipt["configs"].values()]:
        if sha(Path(artifact["path"])) != artifact["sha256"]:
            raise ValueError("checkpoint or configuration hash differs")
    validate_training_configuration(checkpoint.parent, manifest)
    return receipt


def _validate_report(report: dict, name: str, checkpoint_file: str) -> dict:
    return C.validate_fixed_report(report, {
        "name": name, "checkpoint_file": checkpoint_file, "checkpoint_sha256": report["checkpoint"]["sha256"],
        "action_scale": FIXED["action_scale"], "protocol_sha256_lf": sha(FIXED_EVAL, True),
    })


def _check_initials(initials: dict) -> bool | None:
    """같은 실패 체크포인트에서 시작한 두 resume arm이 정책 std 외에 같은 상태에서 출발했는지 확인한다.

    구조 불일치는 예외로 seed 판정을 막는다. 반환값은 두 arm의 학습 직전 RNG 해시 일치 여부이며
    (둘 다 없으면 None) 불일치는 std 재설정 arm의 비교 조건만 깨뜨린다.
    """
    for arm, initial in initials.items():
        mode = ARM_SPECS[arm]["recovery_mode"]
        if initial.get("recovery_mode") != mode or not _rng_ok(initial.get("rng_state_before_learn")):
            raise ValueError("initial recovery mode or RNG observation differs")
        if (mode == "restart") != (initial.get("resume_checkpoint") is None):
            raise ValueError("initial checkpoint usage differs from the recovery mode")
    if "clean_resume_reference" in initials and "resume_corrected" in initials:
        if initials["clean_resume_reference"]["resume_checkpoint"] == initials["resume_corrected"]["resume_checkpoint"]:
            raise ValueError("clean reference resumed the failed checkpoint")
    if set(RESUME_TREATMENTS) <= set(initials):
        plain, reset = (initials[arm] for arm in RESUME_TREATMENTS)
        if (plain["resume_checkpoint"] != reset["resume_checkpoint"]
                or plain["source_checkpoint_state"] != reset["source_checkpoint_state"]):
            raise ValueError("resume treatments did not start from the same checkpoint")
        if (not isinstance(reset.get("std_reset"), dict)
                or reset["std_reset"].get("policy_state_before_reset") != plain["policy_state"]["sha256"]
                or reset["std_reset"].get("optimizer_state_before_reset") != plain["optimizer_state"]["sha256"]):
            raise ValueError("resume treatment initial policy/optimizer states differ beyond the std reset")
        return plain["rng_state_before_learn"] == reset["rng_state_before_learn"]
    return None


def summarize(seed: int, reports: dict, initials: dict, original_reports: dict, bands: dict,
              original_failed: list[str], failed_arm: str | None = None) -> dict:
    """완료된 arm 접두부의 원시 행동 수치를 판정한다. 실행 출처는 부모 검증이 별도로 결합한다."""
    if seed not in SEEDS:
        raise ValueError("unregistered seed")
    available = [arm for arm in ARMS if arm in reports]
    if set(reports) != set(available) or available != list(ARMS[:len(available)]) or set(initials) != set(reports):
        raise ValueError("completed arms must be a prefix of the registered order")
    if failed_arm is not None and (failed_arm not in ARMS or failed_arm in reports):
        raise ValueError("failed arm must be a registered arm without a report")
    rng_match = _check_initials(initials)
    training_versions = [initials[arm].get("runtime_versions") for arm in available]
    if any(not isinstance(value, dict) or value != training_versions[0] for value in training_versions):
        raise ValueError("training runtimes differ or are missing")
    for arm in available:
        _validate_report(reports[arm], arm, f"model_{FINAL_ITERATION[ARM_SPECS[arm]['recovery_mode']]}.pt")
    if set(original_reports) != {"clean_reference", "fault"}:
        raise ValueError("original reference and failed report are required")
    for report in original_reports.values():
        _validate_report(report, report["name"], "model_299.pt")
    runtime = original_reports["clean_reference"]["versions"]
    if any(report["versions"] != runtime for report in [*reports.values(), *original_reports.values()]):
        raise ValueError("fixed evaluation runtimes differ")
    original = {key: C._behavior(value) for key, value in original_reports.items()}
    source_verdict = RC.verdict(RC.compare(original["fault"], original["clean_reference"]), bands)
    if source_verdict["label"] != "unhealthy" or source_verdict["failed"] != original_failed:
        raise ValueError("original failure metrics changed")
    behavior = {arm: C._behavior(report) for arm, report in reports.items()}
    secondary = {arm: RC.verdict(RC.compare(value, original["clean_reference"]), bands)
                 for arm, value in behavior.items()}
    references = {"restart": {"source": FIXED["references"]["restart_primary"],
                              "walks": walks(original_reports["clean_reference"])}}
    references["restart"]["valid"] = references["restart"]["walks"]
    if "clean_resume_reference" in reports:
        resume_walks = walks(reports["clean_resume_reference"])
        resume_secondary = secondary["clean_resume_reference"]["label"]
        references["resume"] = {"source": FIXED["references"]["resume_primary"], "walks": resume_walks,
                                "secondary_label": resume_secondary,
                                "valid": resume_walks and resume_secondary == "healthy"}
    else:
        references["resume"] = {"source": FIXED["references"]["resume_primary"], "walks": None,
                                "secondary_label": None, "valid": False}
    verdicts, labels, reasons = {}, {}, {}
    for arm in TREATMENT_ARMS:
        restart = ARM_SPECS[arm]["recovery_mode"] == "restart"
        reference = references["restart" if restart else "resume"]
        if arm not in reports:
            labels[arm] = "invalid"
            reasons[arm] = "execution_failed" if arm == failed_arm else "not_executed"
        elif not reference["valid"]:
            labels[arm] = "invalid"
            reasons[arm] = "restart_reference_invalid" if restart else "resume_reference_invalid"
        elif arm == "resume_corrected_std_reset" and rng_match is False:
            labels[arm] = "invalid"
            reasons[arm] = "resume_treatment_rng_mismatch"
        else:
            primary = original["clean_reference"] if restart else behavior["clean_resume_reference"]
            verdicts[arm] = RC.verdict(RC.compare(behavior[arm], primary), bands)
            labels[arm] = label_from(verdicts[arm])
            reasons[arm] = "verdict_" + verdicts[arm]["label"]
    signal = {"original_none": standstill_ratio(original_reports["clean_reference"]),
              "original_fault": standstill_ratio(original_reports["fault"]),
              **{arm: standstill_ratio(report) for arm, report in reports.items()}}
    signal_prediction = {
        "original_fault": signal["original_fault"] >= STANDSTILL_PREDICTION["original_fault_min"],
        "original_none": signal["original_none"] <= STANDSTILL_PREDICTION["original_none_max"],
    }
    rule = rule_outcome(rule_recommendation(FIXED["fault_mechanism"], signal["original_fault"]), labels)
    matches = {arm: None if PREDICTIONS[arm] is None or labels[arm] == "invalid" else labels[arm] == PREDICTIONS[arm]
               for arm in TREATMENT_ARMS}
    valid = [arm for arm in TREATMENT_ARMS if labels[arm] != "invalid"]
    status = "complete" if len(valid) == len(TREATMENT_ARMS) else "partial" if valid else "invalid"
    return {
        "status": status, "seed": seed, "scope": FIXED["scope"], "arm_order": list(ARMS),
        "completed_arms": available, "failed_arm": failed_arm,
        "labels": labels, "label_reasons": reasons, "predictions": PREDICTIONS, "prediction_match": matches,
        "references": references, "behavior": behavior, "verdicts": verdicts, "secondary_verdicts": secondary,
        "original_behavior": original, "source_verdict": source_verdict,
        "restart_sham_counterpart": {"run": INPUT_RUNS[str(seed)][FAULT], "reused_original_run": True,
                                     "rerun": False, "verdict": source_verdict},
        "mechanical_signal": {"name": "standstill_ratio", "values": signal,
                              "prediction_met": signal_prediction, "used_for_labels": False,
                              "wording": FIXED["result_wording"]["standstill_ratio"]},
        "recovery_mode_rule": {**rule, "used_for_labels": False,
                               "wording": FIXED["result_wording"]["recovery_mode_rule"]},
        "bands": bands, "resume_treatment_rng_hashes_match": rng_match,
        "full_environment_state_restored": False, "rng_state_restored_from_checkpoint": False,
        "causal_confirmation": False, "stochastic_effect_separated": False,
        "exact_training_trajectory_resumed": False, "model_selected_correction": False,
        "generalization_proven": False, "gpu_saving_claimed": False,
        "wording": {arm: FIXED["result_wording"][labels[arm]] for arm in TREATMENT_ARMS},
        "seed_wording": FIXED["result_wording"]["seed_summary"],
    }


def invalid_result(seed: int, error: Exception | str) -> dict:
    labels = {arm: "invalid" for arm in TREATMENT_ARMS}
    return {"status": "invalid", "seed": seed, "scope": FIXED["scope"], "labels": labels,
            "label_reasons": {arm: "seed_invalid" for arm in TREATMENT_ARMS},
            "error_type": error if isinstance(error, str) else type(error).__name__,
            "detail": None if isinstance(error, str) else str(error),
            "wording": {arm: FIXED["result_wording"]["invalid"] for arm in TREATMENT_ARMS},
            "seed_wording": FIXED["result_wording"]["seed_summary"]}


def aggregate(results: list[dict]) -> dict:
    """seed별 결과를 arm별 라벨·예측·규칙 적중으로 모은다. 새 판정을 만들지 않는다."""
    seeds = [row.get("seed") for row in results]
    if sorted(seeds) != sorted(set(seeds)) or not set(seeds) <= set(SEEDS) or not seeds:
        raise ValueError("aggregate needs distinct registered seed results")
    if len({row.get("protocol_sha256_lf") for row in results}) != 1:
        raise ValueError("seed results come from different protocols")
    arms = {}
    for arm in TREATMENT_ARMS:
        labels = {str(row["seed"]): row["labels"][arm] for row in results}
        valid = [label for label in labels.values() if label != "invalid"]
        prediction = PREDICTIONS[arm]
        arms[arm] = {"labels": labels, "recovered": sum(label == "recovered" for label in valid),
                     "valid_seeds": len(valid), "prediction": prediction,
                     "prediction_held_all_seeds": (None if prediction is None or len(valid) != len(labels)
                                                   else all(label == prediction for label in valid))}
    rules = {str(row["seed"]): row.get("recovery_mode_rule", {}).get("outcome", "undetermined") for row in results}
    return {"seeds": sorted(seeds), "complete_seed_set": sorted(seeds) == list(SEEDS), "arms": arms,
            "recovery_mode_rule_outcomes": rules, "scope": FIXED["scope"],
            "wording": FIXED["result_wording"]["seed_summary"]}


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
    for arm in ARMS:
        suffix = f"_{tag}_{arm}"
        if any(path.name.endswith(suffix) for path in log_root.iterdir()):
            raise FileExistsError("an SDK run already exists for this arm")
    for path in (ROOT, isaaclab, folder):
        if any(char in str(path) for char in '\r\n"%&^'):
            raise ValueError("unsafe Windows batch path")
    admission = M.query_gpu(FIXED["gpu_admission_profile"])
    if admission["busy"]:
        return {"status": "gpu_busy_no_launch", "sdk_jobs_launched": 0}
    request = {"tag": tag, "seed": seed, "source_commit_sha": commit,
               "protocol_sha256_lf": sha(PROTOCOL, True), "isaaclab_root": str(isaaclab.resolve()),
               "selection_source": "registered_host_recovery_mode_comparison_not_model", "bundle_key": key}
    write_new(folder / "request.json", request)
    write_new(folder / "approval.json", {"request_sha256": sha(folder / "request.json"),
                                         "actor": FIXED["approval_actor"], "identity_attested": False})
    write_new(consumption, {"request_sha256": sha(folder / "request.json"), "tag": tag})
    reports, initials, costs = {}, {}, {}
    failed_arm = None
    for arm in ARMS:
        stage = "preflight"
        try:
            current, current_commit = verify_protocol(isaaclab)
            if current != protocol or current_commit != commit:
                raise ValueError("source changed during experiment")
            gpu = M.query_gpu(FIXED["gpu_admission_profile"])
            if gpu["busy"]:
                raise RuntimeError("GPU busy before next training arm")
            arm_dir = folder / arm
            training, evaluation = arm_dir / "train", arm_dir / "eval"
            final = FINAL_ITERATION[ARM_SPECS[arm]["recovery_mode"]]
            write_new(arm_dir / "gpu_before.json", gpu)
            write_new(arm_dir / "consumed.json", {"approval_sha256": sha(folder / "approval.json"), "arm": arm,
                                                  "train_limit": 1, "eval_limit": 1})
            write_new(training / "manifest.json", manifest_for(folder, request, arm))
            argv = [str(isaaclab / "_isaac_sim/python.bat"), str(ROOT / "bench/train_with_recovery_mode.py"),
                    "--manifest", str(training / "manifest.json")]
            stage = "training"
            execution = P.run_owned(training, argv, isaaclab, FIXED["train_watchdog_s"])
            write_new(training / "execution.json", execution)
            stage = "training_validation"
            receipt = verify_training(folder, arm, execution)
            initials[arm] = read(training / "initial.json")
            evaluation.mkdir()
            jobs = {"action_scale": FIXED["action_scale"], "jobs": [{"name": arm, "checkpoint": receipt["checkpoint"]["path"],
                                                                     "output": str(evaluation / "report.json")}]}
            write_new(evaluation / "jobs.json", jobs)
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
                "name": arm, "checkpoint_file": f"model_{final}.pt",
                "checkpoint_sha256": receipt["checkpoint"]["sha256"], "action_scale": FIXED["action_scale"],
                "protocol_sha256_lf": protocol["fixed_eval_sha256_lf"],
            })
            if reports[arm]["versions"] != protocol["resume_inputs"]["runtime_versions"]:
                raise ValueError("fixed evaluator runtime differs from the bound original")
            stage = "postflight_validation"
            current, current_commit = verify_protocol(isaaclab)
            if current != protocol or current_commit != commit:
                raise ValueError("source changed during evaluation")
            if sha(Path(receipt["checkpoint"]["path"])) != receipt["checkpoint"]["sha256"]:
                raise ValueError("checkpoint changed during evaluation")
            after = M.query_gpu(FIXED["gpu_admission_profile"])
            write_new(arm_dir / "gpu_after.json", after)
            owned = set(execution["owned_process_identities"]) | set(eval_execution["owned_process_identities"])
            if any(str(row["pid"]) in owned for row in after["processes"]):
                raise ValueError("an owned GPU process remains")
            costs[arm] = {"train_parent_wall_s": execution["wall_s"], "eval_parent_wall_s": eval_execution["wall_s"]}
            write_new(arm_dir / "parent.json", {"status": "pass", "training_receipt_sha256": sha(training / "receipt.json"),
                                                "fixed_report_sha256": sha(evaluation / "report.json"), **costs[arm]})
            print(json.dumps({"seed": seed, "arm": arm, "status": "validated", **costs[arm]}), flush=True)
        except Exception as error:
            failed_arm = arm
            reports.pop(arm, None)
            initials.pop(arm, None)
            costs.pop(arm, None)
            write_new(folder / "failure.json", {"arm": arm, "stage": stage, "error_type": type(error).__name__,
                                                "detail": str(error)})
            break
    try:
        originals = {"clean_reference": read(source_paths(isaaclab, seed, "NONE")["fixed_report"]),
                     "fault": read(source_paths(isaaclab, seed, FAULT)["fixed_report"])}
        inputs = protocol["resume_inputs"]
        result = summarize(seed, reports, initials, originals, inputs["bands"],
                           inputs["seeds"][str(seed)]["runs"][FAULT]["original_failed"], failed_arm)
    except (KeyError, ValueError, TypeError, OSError) as error:
        result = invalid_result(seed, error)
    result.update(tag=tag, costs=costs, model_calls=0, source_commit_sha=commit,
                  protocol_sha256_lf=request["protocol_sha256_lf"])
    result["arm_evidence"] = {arm: {"parent_sha256": sha(folder / arm / "parent.json"),
                                    **read(folder / arm / "parent.json")}
                              for arm in ARMS if (folder / arm / "parent.json").is_file()}
    write_new(folder / "result.json", result)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify", "execute", "aggregate"))
    parser.add_argument("--isaaclab", type=Path, default=Path.home() / "IsaacLab")
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--tag")
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--result", type=Path, action="append")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        if args.contract is None:
            parser.error("prepare requires --contract from the SDK Python CPU checker")
        result = prepare(args.isaaclab, args.contract)
    elif args.command == "verify":
        _, commit = verify_protocol(args.isaaclab)
        result = {"status": "pass", "source_commit_sha": commit}
    elif args.command == "execute":
        if args.tag is None or args.seed is None:
            parser.error("execute requires --tag and --seed")
        result = execute(args.tag, args.seed, args.isaaclab)
    else:
        if not args.result:
            parser.error("aggregate requires one --result per executed seed")
        result = aggregate([read(path) for path in args.result])
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if result.get("status") == "invalid" else 0


if __name__ == "__main__":
    raise SystemExit(main())
