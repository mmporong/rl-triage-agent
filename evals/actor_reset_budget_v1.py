"""actor 마지막 층 재설정의 100/150회 회복 예산을 새 F1 seed에서 확인한다."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "bench", ROOT / "evals"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import novel_actor_reset_confirmation as H  # noqa: E402
import preregistered_seed_guard as SEED_GUARD  # noqa: E402
import record_evidence_media as GPU  # noqa: E402
import run_recovery_reset_v2 as ANCHOR  # noqa: E402
import run_runtime_readback_smoke as OWNED  # noqa: E402
import run_training_correction as V3  # noqa: E402
import run_training_correction_v4 as V4  # noqa: E402
import train_with_recovery_mode as R  # noqa: E402
from rl_triage import grounded_recovery as C  # noqa: E402
from rl_triage import recovery as RC  # noqa: E402


PROTOCOL = ROOT / "bench/protocols/actor_reset_budget_v1.json"
FIXED_EVAL = ROOT / "bench/protocols/fixed_eval_v1.json"
PRIVATE = ROOT / "bench/private/codex_support_20261015/task25"
RESULTS = PRIVATE / "runs"
TAG = "actor_reset_budget_20261016"
SEEDS = (2090, 2091)
RESET_BUDGETS = (100, 150)
CONDITIONAL_BUDGET = 200
RESTART_BUDGET = 300
FAULT = "F1"
CORRECTION = "restore_body_frame_reward"
TASK = C.TASK
NUM_ENVS, SOURCE_ITERATIONS, NUM_STEPS = 4096, 300, 24
ACTION_SCALE = 0.25
SOURCE_FINAL = 299
GPU_PROFILE = V4.FIXED["gpu_admission_profile"]
CODE = {
    "evals/actor_reset_budget_v1.py", "bench/train_actor_reset_budget_v1.py",
    "tests/test_actor_reset_budget_v1.py", "bench/run_actor_reset_budget_v1.ps1",
    "evals/novel_actor_reset_confirmation.py", "bench/train_novel_actor_reset.py",
    "bench/train_with_fault.py", "bench/hidden_faults.py", "bench/train_with_recovery_mode.py",
    "bench/train_with_correction.py", "bench/recovery_reset_interventions.py",
    "bench/correction_adapter.py", "evals/preregistered_seed_guard.py", "evals/fixed_eval.py",
    "evals/run_recovery_reset_v2.py", "evals/run_training_correction.py",
    "evals/run_training_correction_v4.py", "bench/record_evidence_media.py",
    "bench/run_runtime_readback_smoke.py", "bench/runtime_readback_smoke.py",
    "src/rl_triage/grounded_recovery.py", "src/rl_triage/recovery.py",
}
BOUND = tuple(sorted(CODE | {PROTOCOL.relative_to(ROOT).as_posix(), FIXED_EVAL.relative_to(ROOT).as_posix(),
                             V4.PROTOCOL.relative_to(ROOT).as_posix()}))
read, write_new, sha = V3.read, V3.write_new, V3.sha
validate_readback = V3.validate_readback


def validated_wall_s(execution: dict[str, Any], stage: str) -> float:
    value = execution.get("wall_s")
    if (execution.get("wall_scope") != "parent_launch_to_exit_and_owned_job_empty_verification"
            or isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value <= 0):
        raise ValueError(f"{stage} GPU 작업 벽시계 receipt가 다르다")
    return float(value)


def stage_cost(stage: Path, name: str) -> dict[str, float]:
    training = validated_wall_s(read(stage / "train/execution.json"), f"{name} 학습")
    fixed_eval = validated_wall_s(read(stage / "eval/execution.json"), f"{name} 고정 평가")
    return {"training_wall_s": training, "fixed_eval_wall_s": fixed_eval,
            "total_wall_s": training + fixed_eval}


FIXED = {
    "schema": "actor_reset_budget_protocol_v1",
    "measurement_kind": "post_result_novel_seed_budget_confirmation",
    "design_disclosure": "기존 seed 2026 회복·2028 미회복과 새 seed 2063·2064의 actor 마지막 층 재설정 300회 2/2 회복 결과를 본 뒤 더 적은 갱신 횟수를 확인하도록 설계했다",
    "seeds": list(SEEDS), "task": TASK, "fault": FAULT, "correction": CORRECTION,
    "num_envs": NUM_ENVS, "source_iterations": SOURCE_ITERATIONS,
    "num_steps_per_env": NUM_STEPS, "source_final_iteration": SOURCE_FINAL,
    "reset_budgets": list(RESET_BUDGETS), "conditional_budget": CONDITIONAL_BUDGET,
    "restart_budget": RESTART_BUDGET, "action_scale": ACTION_SCALE,
    "prediction": {"reset_100": ["not_recovered", "not_recovered"],
                   "reset_150": ["recovered", "recovered"],
                   "reset_200_if_needed": ["recovered", "recovered"],
                   "restart_300": ["recovered", "recovered"]},
    "branch_rule": "100회와 150회 중 어느 예산도 두 seed 모두 recovered가 아니면 200회를 두 seed에서 실행한다",
    "candidate_rule": "가장 작은 공통 reset 예산이 두 seed 모두 recovered이고 restart300도 두 seed 모두 recovered이며 provenance가 유효할 때, 그 reset 예산이 300보다 작으면 조건부 후보로 등록한다",
    "candidate_wording": "새 seed 두 개에서 재학습 300회와 단축 개입 N회가 모두 고정 평가를 통과했다. 재학습보다 적은 학습 갱신으로 회복한 등록 조건부 후보다.",
    "keep_wording": "등록된 단축 예산이 두 seed 공통 회복 조건을 충족하지 않아 기존 회복 규칙을 유지한다.",
    "restart_control_not_recovered_wording": "재학습 대조가 두 seed 회복 조건을 충족하지 않아 단축 회복 후보 등록을 판정하지 않는다.",
    "automatic_global_change": False, "semantic_retries": 0, "model_calls": 0,
    "independent_arms": "각 reset 예산은 같은 F1 model_299에서 따로 시작하며 더 긴 예산이 짧은 예산 결과에서 이어지지 않는다",
    "restart_control": "동일 seed에서 checkpoint를 불러오지 않는 300회 재학습을 항상 실행한다",
    "initial_equivalence": "같은 seed의 reset100·reset150·조건부 reset200은 reset seed, 초기 policy, optimizer, RNG hash가 같아야 한다",
    "time_scope": "학습 시간은 parent launch부터 자연 종료와 owned job 정리 확인까지의 GPU 작업 벽시계이며 GPU kernel time이 아니다",
    "claim_boundary": "후보 판정은 갱신 횟수 N<300에만 근거하며 시간 단축 여부는 별도 결과로 기록한다",
    "budget_unit": "N은 learning iteration 또는 rollout update 반복 수이며 optimizer.step 횟수 측정이 아니다. actual control steps=N×24로 검증한다",
    "restart_identity_disclosure": "restart300은 같은 seed와 clean 함수의 source NONE 300회와 checkpoint byte가 같을 수 있으며 독립 표본으로 해석하지 않는다",
    "gpu_admission_profile": GPU_PROFILE, "train_watchdog_s": 2700, "eval_watchdog_s": 600,
    "isaaclab_commit": V4.FIXED["isaaclab_commit"], "rsl_rl_version": V4.FIXED["rsl_rl_version"],
    "limits": ["새 seed 두 개의 개발 확인이다", "인과 메커니즘·일반화·GPU 절감·LLM 우월을 주장하지 않는다"],
}
DYNAMIC = {"source_files", "sdk_sources", "rsl_sources", "correction_source_contract",
           "fixed_eval_sha256_lf", "bands", "runtime_versions"}


def reset_seed(seed: int) -> int:
    if seed not in SEEDS:
        raise ValueError("등록되지 않은 seed")
    return 120000 + seed


def source_maps(isaaclab: Path) -> dict[str, Any]:
    maps = V4.source_maps(isaaclab)
    maps["source_files"] = {name: sha(ROOT / name, True) for name in sorted(CODE)}
    return maps


def prepare(isaaclab: Path) -> dict[str, Any]:
    base = read(V4.PROTOCOL)
    document = {**deepcopy(FIXED), **source_maps(isaaclab),
                "correction_source_contract": base["correction_source_contract"],
                "fixed_eval_sha256_lf": sha(FIXED_EVAL, True),
                "bands": base["resume_inputs"]["bands"],
                "runtime_versions": base["resume_inputs"]["runtime_versions"]}
    ANCHOR._reject_public_absolute_paths(document)
    write_new(PROTOCOL, document)
    return {"status": "prepared_commit_before_execution",
            "path": PROTOCOL.relative_to(ROOT).as_posix(), "sha256_lf": sha(PROTOCOL, True)}


def verify_protocol(isaaclab: Path, committed: bool = True) -> tuple[dict[str, Any], str | None]:
    protocol = read(PROTOCOL)
    if set(protocol) != set(FIXED) | DYNAMIC or any(protocol.get(key) != value for key, value in FIXED.items()):
        raise ValueError("고정 budget protocol이 바뀌었다")
    V3.check_contract(protocol["correction_source_contract"], isaaclab)
    if any(protocol.get(key) != value for key, value in source_maps(isaaclab).items()):
        raise ValueError("결합 source hash가 바뀌었다")
    if protocol.get("fixed_eval_sha256_lf") != sha(FIXED_EVAL, True):
        raise ValueError("고정 평가 protocol이 바뀌었다")
    if protocol.get("runtime_versions") != read(V4.PROTOCOL)["resume_inputs"]["runtime_versions"]:
        raise ValueError("결합 runtime version이 바뀌었다")
    if V3.S._git_head(isaaclab) != FIXED["isaaclab_commit"] or not V3.S._git_tracked_clean(isaaclab):
        raise ValueError("Isaac Lab checkout이 다르다")
    return protocol, ANCHOR.verify_bound_at_head(ROOT, BOUND) if committed else None


def folder_for(tag: str = TAG) -> Path:
    if not isinstance(tag, str) or tag != TAG:
        raise ValueError("등록 tag가 아니다")
    folder = RESULTS / tag
    if folder.is_symlink() or folder.resolve().parent != RESULTS.resolve():
        raise ValueError("결과 경로가 private root 밖이다")
    return folder


def arm_name(mode: str, budget: int) -> str:
    if mode == "reset" and budget in (*RESET_BUDGETS, CONDITIONAL_BUDGET):
        return f"actor_last_layer_reset_{budget}"
    if mode == "restart" and budget == RESTART_BUDGET:
        return "restart_300"
    raise ValueError("등록되지 않은 arm/budget")


def arm_spec(arm: str) -> tuple[str, int]:
    if arm == "restart_300":
        return "restart", RESTART_BUDGET
    match = re.fullmatch(r"actor_last_layer_reset_(100|150|200)", arm)
    if match:
        return "resume", int(match.group(1))
    raise ValueError("등록되지 않은 arm")


def consumption_path(tag: str) -> Path:
    key = hashlib.sha256((sha(PROTOCOL, True) + tag).encode()).hexdigest()
    return PRIVATE / "consumed" / f"{key}.json"


def validate_freshness(row: Any) -> dict[str, Any]:
    expected_keys = {"schema", "status", "seeds", "checked_at_utc", "checked_roots",
                     "checked_file_count", "evidence", "scope"}
    roots = ["<repo>/bench/**/*.telemetry.json", "<repo>/bench/private/execution_records",
             "<repo>/evals/results", "<repo>/evals/loops", "<isaaclab>/logs"]
    if (not isinstance(row, dict) or set(row) != expected_keys
            or row.get("schema") != "preregistered_seed_freshness_v1" or row.get("status") != "fresh"
            or row.get("seeds") != list(SEEDS) or row.get("checked_roots") != roots
            or type(row.get("checked_file_count")) is not int or row["checked_file_count"] < 0
            or row.get("evidence") != []
            or row.get("scope") != "현재 저장소의 실행 기록과 지정 Isaac Lab 로그에 한정한다"):
        raise ValueError("fresh-seed receipt가 다르다")
    try:
        checked = datetime.fromisoformat(row["checked_at_utc"])
    except (TypeError, ValueError) as error:
        raise ValueError("fresh-seed receipt 시간이 다르다") from error
    if checked.tzinfo is None or checked.utcoffset() != timezone.utc.utcoffset(checked):
        raise ValueError("fresh-seed receipt 시간이 UTC가 아니다")
    return row


def validate_run_request(folder: Path, isaaclab: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    request = read(folder / "request.json")
    anchor = ANCHOR.verify_source_anchor(request.get("source_anchor"), ROOT, BOUND)
    expected = {"tag": folder.name, "source_commit_sha": anchor["commit"], "source_anchor": anchor,
                "protocol_sha256_lf": sha(PROTOCOL, True), "isaaclab_root": str(isaaclab.resolve())}
    if request != expected:
        raise ValueError("result request/source anchor가 다르다")
    if read(consumption_path(folder.name)) != {"tag": folder.name,
                                               "request_sha256": sha(folder / "request.json")}:
        raise ValueError("result consumption receipt가 다르다")
    return request, anchor


def manifest_for(folder: Path, request: dict[str, Any], seed: int, arm: str) -> dict[str, Any]:
    if seed not in SEEDS:
        raise ValueError("등록되지 않은 seed")
    mode, budget = arm_spec(arm)
    source = read(folder / "source" / f"s{seed}" / "source_evidence.json")[FAULT]
    selection = read(folder / "source_selection.json")
    expected_selection = {"source_evidence_sha256": sha(folder / "source" / f"s{seed}" / "source_evidence.json"),
                          "checkpoint_sha256": source["checkpoint_sha256"]}
    if selection.get("seeds", {}).get(str(seed)) != expected_selection:
        raise ValueError("등록 source selection binding이 다르다")
    resumed = mode == "resume"
    checkpoint = Path(source["checkpoint"])
    return {
        "schema": "training_correction_manifest_v2", "tag": folder.name, "seed": seed, "arm": arm,
        "recovery_mode": mode, "source_commit_sha": request["source_commit_sha"],
        "protocol_sha256_lf": request["protocol_sha256_lf"], "isaaclab_root": request["isaaclab_root"],
        "protocol_path": str(PROTOCOL.resolve()), "fault": FAULT, "correction_id": CORRECTION,
        "num_envs": NUM_ENVS, "max_iterations": budget, "task": TASK,
        "run_name": f"{folder.name}_s{seed}_{arm}",
        "output_dir": str((folder / "arms" / f"s{seed}" / arm / "train").resolve()),
        "final_iteration": SOURCE_FINAL + budget - 1 if resumed else budget - 1,
        "resume_checkpoint_path": str(checkpoint.resolve()) if resumed else None,
        "resume_checkpoint_sha256": source["checkpoint_sha256"] if resumed else None,
        "resume_iteration": SOURCE_FINAL if resumed else None,
        "load_run_regex": "^" + re.escape(checkpoint.parent.name) + "$" if resumed else None,
        "checkpoint_regex": "^model_299\\.pt$" if resumed else None,
        "source_env_config_path": source["environment"]["path"],
        "source_agent_config_path": source["agent"]["path"], "config_source_fault": FAULT,
        "init_noise_std": None,
    }


def validate_manifest(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = read(path)
    if set(manifest) != R.MANIFEST_FIELDS:
        raise ValueError("budget manifest schema가 다르다")
    mode, _ = arm_spec(manifest.get("arm", ""))
    if R.validate_manifest_mode(manifest) != mode:
        raise ValueError("budget manifest mode가 다르다")
    tag = manifest.get("tag")
    if not isinstance(tag, str):
        raise ValueError("manifest tag가 문자열이 아니다")
    folder = folder_for(tag)
    request = read(folder / "request.json")
    anchor = ANCHOR.verify_source_anchor(request.get("source_anchor"), ROOT, BOUND)
    protocol, _ = verify_protocol(Path(request["isaaclab_root"]))
    expected_path = folder / "arms" / f"s{manifest['seed']}" / manifest["arm"] / "train/manifest.json"
    if path.is_symlink() or path.resolve() != expected_path.resolve():
        raise ValueError("manifest가 등록 arm 경로 밖이다")
    if manifest != manifest_for(folder, request, manifest["seed"], manifest["arm"]):
        raise ValueError("budget manifest가 등록 입력과 다르다")
    if manifest["source_commit_sha"] != anchor["commit"] or manifest["protocol_sha256_lf"] != sha(PROTOCOL, True):
        raise ValueError("budget manifest source anchor가 다르다")
    return manifest, protocol


def validate_training_configuration(log_dir: Path, manifest: dict[str, Any]) -> None:
    source_env, source_agent = Path(manifest["source_env_config_path"]), Path(manifest["source_agent_config_path"])
    original_env, original_agent = V3.load_config(source_env), V3.load_config(source_agent)
    current_env = V3.load_config(log_dir / "params/env.yaml")
    current_agent = V3.load_config(log_dir / "params/agent.yaml")
    if current_env != original_env:
        raise ValueError("environment config가 source와 다르다")
    mode = manifest["recovery_mode"]
    allowed = {"run_name"} if mode == "restart" else {
        "run_name", "resume", "load_run", "load_checkpoint", "max_iterations"}
    if ({key: value for key, value in original_agent.items() if key not in allowed}
            != {key: value for key, value in current_agent.items() if key not in allowed}):
        raise ValueError("agent config가 허용 범위 밖에서 바뀌었다")
    expected = {"run_name": manifest["run_name"]}
    if mode != "restart":
        expected.update({"resume": True, "load_run": manifest["load_run_regex"],
                         "load_checkpoint": manifest["checkpoint_regex"],
                         "max_iterations": manifest["max_iterations"]})
    if any(current_agent.get(key) != value for key, value in expected.items()):
        raise ValueError("agent recovery arguments가 manifest와 다르다")
    if (original_agent.get("seed") != manifest["seed"]
            or original_agent.get("num_steps_per_env") != NUM_STEPS
            or original_agent.get("max_iterations") != SOURCE_ITERATIONS
            or original_env.get("seed") != manifest["seed"]
            or original_env.get("scene", {}).get("num_envs") != NUM_ENVS):
        raise ValueError("source seed/env/update config가 다르다")


def validate_intervention(row: Any, manifest: dict[str, Any], source_state: dict[str, Any],
                          initial: dict[str, Any] | None = None) -> None:
    if manifest["recovery_mode"] != "resume" or not manifest["arm"].startswith("actor_last_layer_reset_"):
        raise ValueError("reset intervention이 아닌 manifest다")
    if not isinstance(row, dict):
        raise ValueError("actor reset readback이 객체가 아니다")
    required = {"arm", "critic_state_before", "critic_state_after", "std_state_before", "std_state_after",
                "actor_linear_states_before", "actor_linear_states_after"}
    before, after = row.get("actor_linear_states_before"), row.get("actor_linear_states_after")
    targets = row.get("target_modules")
    indices = row.get("optimizer_param_indices")
    moments_before = row.get("optimizer_moment_indices_present_before")
    if (row.get("schema") != "actor_reset_budget_intervention_v1"
            or not required <= set(row) or row.get("arm") != manifest["arm"]
            or row.get("intervention") != "actor_last_layer_reset"
            or row.get("reset_seed") != reset_seed(manifest["seed"])
            or row.get("policy_state_before") != source_state["policy_state_sha256"]
            or row.get("optimizer_state_before") != source_state["optimizer_state_sha256"]
            or row.get("critic_state_before") != row.get("critic_state_after")
            or row.get("std_state_before") != row.get("std_state_after")
            or not isinstance(before, dict) or not isinstance(after, dict) or set(before) != set(after) or not before
            or not isinstance(targets, list) or targets != [list(before)[-1]]
            or row.get("target_parameter_count") != 2
            or not isinstance(indices, list) or any(type(value) is not int for value in indices)
            or indices != sorted(set(indices)) or len(indices) != row["target_parameter_count"]
            or not isinstance(moments_before, list) or moments_before != indices
            or row.get("optimizer_moments_present_after") is not False
            or row.get("policy_state_after") == row.get("policy_state_before")
            or row.get("optimizer_state_after") != row.get("optimizer_state_expected_after")
            or (initial is not None and (
                row.get("policy_state_after") != initial.get("policy_state", {}).get("sha256")
                or row.get("optimizer_state_after") != initial.get("optimizer_state", {}).get("sha256")))
            or row.get("optimizer_param_groups_before") != row.get("optimizer_param_groups_after")
            or row.get("parameter_identity_preserved") is not True
            or [name for name in before if before[name] != after[name]] != targets):
        raise ValueError("actor 마지막 층/Adam/critic/std 보존 계약이 깨졌다")


def _training_command(isaaclab: Path, manifest: Path) -> list[str]:
    return [str(isaaclab / "_isaac_sim/python.bat"), "-B",
            str(ROOT / "bench/train_actor_reset_budget_v1.py"), "--manifest", str(manifest)]


RECEIPT_FIELDS = {
    "schema", "completed", "tag", "seed", "arm", "recovery_mode", "fault", "correction_id",
    "source_commit_sha", "protocol_sha256_lf", "protocol", "manifest_sha256", "initial_sha256",
    "correction_sha256", "actual_control_steps", "resume_checkpoint", "resume_iteration",
    "start_iteration", "final_iteration", "checkpoint_load_calls", "source_checkpoint_state",
    "final_checkpoint_state", "std_reset_applied", "budget", "aggregate_env_steps",
    "environment_make_calls", "task", "run_name", "observed", "actual_action_scale",
    "runtime_versions", "rng_state_before_learn", "resume_scope", "process_identity", "checkpoint", "configs",
}
INITIAL_FIELDS = {
    "schema", "recovery_mode", "policy_state", "optimizer_state", "observed", "runtime_versions", "log_dir",
    "resume_checkpoint", "resume_iteration", "start_iteration", "init_at_random_ep_len",
    "source_checkpoint_state", "std_reset", "rng_state_before_learn", "full_environment_state_restored",
    "rng_state_restored_from_checkpoint",
}


def _sha_text(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _rng_ok(value: Any) -> bool:
    return (isinstance(value, dict)
            and set(value) == {"cpu_sha256", "cuda_all_sha256", "cuda_device_count"}
            and _sha_text(value.get("cpu_sha256"))
            and isinstance(value.get("cuda_all_sha256"), list) and bool(value["cuda_all_sha256"])
            and type(value.get("cuda_device_count")) is int
            and value["cuda_device_count"] == len(value["cuda_all_sha256"])
            and all(_sha_text(item) for item in value["cuda_all_sha256"]))


def validate_receipt_contract(training: Path, manifest: dict[str, Any], protocol: dict[str, Any],
                              receipt: Any, initial: Any) -> Path:
    mode, budget = arm_spec(manifest["arm"])
    resumed = mode == "resume"
    expected_final = SOURCE_FINAL + budget - 1 if resumed else budget - 1
    if (not isinstance(receipt, dict) or set(receipt) != RECEIPT_FIELDS
            or not isinstance(initial, dict) or set(initial) != INITIAL_FIELDS
            or receipt.get("schema") != "training_correction_receipt_v2" or receipt.get("completed") is not True
            or initial.get("schema") != "training_correction_initial_v2"
            or initial.get("recovery_mode") != mode):
        raise ValueError("training receipt/initial schema가 다르다")
    for key in ("tag", "seed", "arm", "recovery_mode", "fault", "correction_id", "source_commit_sha",
                "protocol_sha256_lf", "task", "run_name", "final_iteration"):
        if receipt.get(key) != manifest[key]:
            raise ValueError(f"training receipt manifest 결합이 다르다: {key}")
    expected_protocol = {"path": str(Path(manifest["protocol_path"]).resolve()),
                         "sha256_lf": manifest["protocol_sha256_lf"]}
    if receipt.get("protocol") != expected_protocol:
        raise ValueError("training receipt protocol 결합이 다르다")
    for filename, field in (("manifest.json", "manifest_sha256"), ("initial.json", "initial_sha256"),
                            ("correction.json", "correction_sha256")):
        if receipt.get(field) != sha(training / filename):
            raise ValueError("training raw artifact hash가 다르다")
    steps = budget * NUM_STEPS
    expected_budget = {"seed": manifest["seed"], "num_envs": manifest["num_envs"],
                       "max_iterations": budget, "num_steps_per_env": NUM_STEPS,
                       "expected_control_steps": steps}
    if (receipt.get("budget") != expected_budget or receipt.get("actual_control_steps") != steps
            or receipt.get("aggregate_env_steps") != steps * manifest["num_envs"]
            or receipt.get("environment_make_calls") != 1):
        raise ValueError("training budget/environment receipt가 다르다")
    expected_observed = {"environment_seed": manifest["seed"], "environment_num_envs": manifest["num_envs"],
                         "action_scale": ACTION_SCALE, "agent_seed": manifest["seed"],
                         "agent_max_iterations": budget, "runner_num_steps_per_env": NUM_STEPS}
    if (receipt.get("observed") != expected_observed or initial.get("observed") != expected_observed
            or receipt.get("actual_action_scale") != ACTION_SCALE):
        raise ValueError("training runtime config observation이 다르다")
    source_state = receipt.get("source_checkpoint_state")
    if resumed:
        expected_resume: dict[str, Any] | None = {"path": manifest["resume_checkpoint_path"],
                                                  "sha256": manifest["resume_checkpoint_sha256"]}
        expected_initial_resume: dict[str, Any] | None = {**expected_resume, "load_optimizer": True}
    else:
        expected_resume = None
        expected_initial_resume = None
    if (receipt.get("resume_checkpoint") != expected_resume
            or initial.get("resume_checkpoint") != expected_initial_resume
            or receipt.get("resume_iteration") != (SOURCE_FINAL if resumed else None)
            or initial.get("resume_iteration") != (SOURCE_FINAL if resumed else None)
            or receipt.get("start_iteration") != (SOURCE_FINAL if resumed else 0)
            or initial.get("start_iteration") != (SOURCE_FINAL if resumed else 0)
            or receipt.get("checkpoint_load_calls") != (1 if resumed else 0)
            or source_state != (initial.get("source_checkpoint_state") if resumed else None)
            or (not resumed and initial.get("source_checkpoint_state") is not None)
            or receipt.get("std_reset_applied") is not False or initial.get("std_reset") is not None):
        raise ValueError("training resume/restart identity가 다르다")
    proofs = [(receipt.get("final_checkpoint_state"), expected_final)]
    if resumed:
        proofs.append((source_state, SOURCE_FINAL))
    for state, iteration in proofs:
        if (not isinstance(state, dict) or state.get("iteration") != iteration
                or not _sha_text(state.get("policy_state_sha256"))
                or not _sha_text(state.get("optimizer_state_sha256"))):
            raise ValueError("checkpoint state proof가 다르다")
    resume_scope = {"policy_optimizer_restored": resumed, "full_environment_state_restored": False,
                    "rng_state_restored_from_checkpoint": False, "fresh_initialization": not resumed,
                    "action_std_reset": False}
    if (receipt.get("resume_scope") != resume_scope or initial.get("init_at_random_ep_len") is not True
            or initial.get("full_environment_state_restored") is not False
            or initial.get("rng_state_restored_from_checkpoint") is not False):
        raise ValueError("resume scope/episode initialization이 다르다")
    rng = initial.get("rng_state_before_learn")
    if receipt.get("rng_state_before_learn") != rng or not _rng_ok(rng):
        raise ValueError("training RNG observation이 다르다")
    versions = receipt.get("runtime_versions")
    expected_versions = protocol["runtime_versions"]
    if (not isinstance(versions, dict) or versions != initial.get("runtime_versions")
            or versions.get("python") != ".".join(map(str, protocol["correction_source_contract"]["runtime"]["version"]))
            or versions.get("python_cache_tag") != protocol["correction_source_contract"]["runtime"]["cache_tag"]
            or versions.get("torch_distribution") != expected_versions["torch"]
            or versions.get("rsl_rl_distribution") != expected_versions["rsl-rl-lib"]
            or versions.get("isaaclab_distribution") != expected_versions["isaaclab"]
            or versions.get("isaaclab_source_commit") != protocol["isaaclab_commit"]):
        raise ValueError("training runtime version이 다르다")
    identity = receipt.get("process_identity")
    if (not isinstance(identity, dict) or set(identity) != {"process_id", "process_creation_time_filetime"}
            or type(identity.get("process_id")) is not int or identity["process_id"] <= 0
            or type(identity.get("process_creation_time_filetime")) is not int
            or identity["process_creation_time_filetime"] <= 0):
        raise ValueError("training process identity가 다르다")
    checkpoint_artifact = receipt.get("checkpoint")
    if not isinstance(checkpoint_artifact, dict):
        raise ValueError("training checkpoint artifact가 객체가 아니다")
    checkpoint = Path(checkpoint_artifact.get("path", ""))
    log_root = (Path(manifest["isaaclab_root"]) / "logs/rsl_rl/unitree_go2_flat").resolve()
    if (checkpoint.resolve().parent.parent != log_root or not checkpoint.parent.name.endswith("_" + manifest["run_name"])
            or checkpoint.name != f"model_{expected_final}.pt"
            or Path(initial.get("log_dir", "")).resolve() != checkpoint.parent.resolve()):
        raise ValueError("training checkpoint/log 경로가 등록 run과 다르다")
    configs = receipt.get("configs")
    if not isinstance(configs, dict) or set(configs) != {"environment", "agent"}:
        raise ValueError("training config artifact 집합이 다르다")
    artifacts = [checkpoint_artifact, *configs.values()]
    expected_paths = {"environment": checkpoint.parent / "params/env.yaml",
                      "agent": checkpoint.parent / "params/agent.yaml"}
    for key, path in expected_paths.items():
        if configs[key] != {"path": str(path.resolve()), "sha256": sha(path)}:
            raise ValueError("training config path/hash가 다르다")
    for artifact in artifacts:
        if (not isinstance(artifact, dict) or set(artifact) != {"path", "sha256"}
                or not _sha_text(artifact.get("sha256")) or sha(Path(artifact["path"])) != artifact["sha256"]):
            raise ValueError("training checkpoint/config artifact가 다르다")
    if (not isinstance(initial.get("policy_state"), dict) or not _sha_text(initial["policy_state"].get("sha256"))
            or not isinstance(initial.get("optimizer_state"), dict)
            or not _sha_text(initial["optimizer_state"].get("sha256"))):
        raise ValueError("initial policy/optimizer state proof가 다르다")
    return checkpoint


def _verify_training(folder: Path, seed: int, arm: str, execution: dict[str, Any]) -> dict[str, Any]:
    if not H._finished(execution):
        raise ValueError("budget training owned execution이 실패했다")
    training = folder / "arms" / f"s{seed}" / arm / "train"
    isaaclab = Path(read(folder / "request.json")["isaaclab_root"])
    manifest_path = training / "manifest.json"
    if read(training / "launch.json") != {"args": _training_command(isaaclab, manifest_path), "cwd": str(isaaclab)}:
        raise ValueError("budget training launch argv/cwd가 다르다")
    manifest, protocol = validate_manifest(manifest_path)
    mode, budget = arm_spec(arm)
    receipt, initial = read(training / "receipt.json"), read(training / "initial.json")
    checkpoint = validate_receipt_contract(training, manifest, protocol, receipt, initial)
    expected_final = SOURCE_FINAL + budget - 1 if mode == "resume" else budget - 1
    if (receipt.get("completed") is not True or receipt.get("checkpoint_load_calls") != (1 if mode == "resume" else 0)
            or receipt.get("resume_iteration") != (SOURCE_FINAL if mode == "resume" else None)
            or receipt.get("start_iteration") != (SOURCE_FINAL if mode == "resume" else 0)
            or receipt.get("final_iteration") != expected_final
            or receipt.get("actual_control_steps") != budget * NUM_STEPS
            or receipt.get("aggregate_env_steps") != budget * NUM_STEPS * NUM_ENVS
            or receipt.get("budget", {}).get("expected_control_steps") != budget * NUM_STEPS):
        raise ValueError("budget/update/checkpoint receipt가 다르다")
    training_wall_s = validated_wall_s(execution, "학습")
    identity = receipt.get("process_identity", {})
    if execution["owned_process_identities"].get(str(identity.get("process_id"))) != identity.get("process_creation_time_filetime"):
        raise ValueError("training receipt가 owned process에서 나오지 않았다")
    for filename, field in (("manifest.json", "manifest_sha256"), ("initial.json", "initial_sha256"),
                            ("correction.json", "correction_sha256")):
        if receipt.get(field) != sha(training / filename):
            raise ValueError("training raw artifact hash가 다르다")
    correction = read(training / "correction.json")
    validate_readback(correction, FAULT, CORRECTION)
    for key, target in protocol["correction_source_contract"]["targets"].items():
        if correction["before_fault"][key] != target["clean_fingerprint"]:
            raise ValueError("clean correction target가 source contract와 다르다")
    if mode == "resume":
        source = read(folder / "source" / f"s{seed}/source_evidence.json")[FAULT]
        expected_resume = {"path": manifest["resume_checkpoint_path"], "sha256": source["checkpoint_sha256"]}
        if (receipt.get("resume_checkpoint") != expected_resume
                or initial.get("resume_checkpoint") != {**expected_resume, "load_optimizer": True}
                or receipt.get("source_checkpoint_state") != initial.get("source_checkpoint_state")):
            raise ValueError("paired source checkpoint binding이 다르다")
        validate_intervention(read(training / "intervention.json"), manifest,
                              receipt["source_checkpoint_state"], initial)
    elif receipt.get("resume_checkpoint") is not None or initial.get("resume_checkpoint") is not None:
        raise ValueError("restart가 checkpoint를 불러왔다")
    if checkpoint.name != f"model_{expected_final}.pt" or sha(checkpoint) != receipt["checkpoint"]["sha256"]:
        raise ValueError("최종 checkpoint iteration/hash가 다르다")
    validate_training_configuration(checkpoint.parent, manifest)
    observed = {"environment_seed": seed, "environment_num_envs": NUM_ENVS,
                "action_scale": ACTION_SCALE, "agent_seed": seed,
                "agent_max_iterations": budget, "runner_num_steps_per_env": NUM_STEPS}
    if receipt.get("observed") != observed or initial.get("observed") != observed:
        raise ValueError("runtime seed/env/update readback이 다르다")
    runtime = protocol["correction_source_contract"]["runtime"]
    versions = receipt["runtime_versions"]
    if (versions.get("rsl_rl_distribution") != protocol["rsl_rl_version"]
            or versions.get("isaaclab_source_commit") != protocol["isaaclab_commit"]
            or versions.get("python") != ".".join(map(str, runtime["version"]))
            or versions.get("python_cache_tag") != runtime["cache_tag"]):
        raise ValueError("runtime version이 다르다")
    return {"receipt": receipt, "initial": initial, "training_wall_s": training_wall_s,
            "actual_learning_iterations": receipt["actual_control_steps"] // NUM_STEPS}


def _initial_signature(initial: dict[str, Any]) -> dict[str, Any]:
    return {"policy_state_sha256": initial["policy_state"]["sha256"],
            "optimizer_state_sha256": initial["optimizer_state"]["sha256"],
            "rng_state_before_learn": initial["rng_state_before_learn"]}


def validate_initial_equivalence(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if set(rows) not in ({"100", "150"}, {"100", "150", "200"}):
        raise ValueError("reset 초기상태 비교 budget 집합이 다르다")
    signatures = {budget: _initial_signature(row["initial"]) for budget, row in rows.items()}
    first = signatures["100"]
    if any(value != first for value in signatures.values()):
        raise ValueError("독립 reset arm 초기 policy/optimizer/RNG가 다르다")
    return {"status": "identical", "budgets": [int(value) for value in rows], "signature": first}


def should_run_200(labels: dict[str, dict[str, str]]) -> bool:
    return not any(all(labels[str(budget)].get(str(seed)) == "recovered" for seed in SEEDS)
                       for budget in RESET_BUDGETS)


def remaining_arm_order(labels: dict[str, dict[str, str]]) -> tuple[str, ...]:
    return (("200",) if should_run_200(labels) else ()) + ("restart_300",)


def validate_result_labels(labels: Any) -> dict[str, dict[str, str]]:
    allowed_arms = ({"100", "150", "restart_300"}, {"100", "150", "200", "restart_300"})
    expected_seeds = {str(seed) for seed in SEEDS}
    if not isinstance(labels, dict) or set(labels) not in allowed_arms:
        raise ValueError("result label arm 집합이 다르다")
    for arm, row in labels.items():
        if (not isinstance(row, dict) or set(row) != expected_seeds
                or any(label not in {"recovered", "not_recovered"} for label in row.values())):
            raise ValueError(f"result label seed/값 집합이 다르다: {arm}")
    return labels


def candidate_decision(labels: dict[str, dict[str, str]], provenance_valid: bool) -> tuple[str, int | None]:
    restart_ok = all(labels["restart_300"].get(str(seed)) == "recovered" for seed in SEEDS)
    if not restart_ok:
        return "restart_control_not_recovered", None
    common = [budget for budget in (*RESET_BUDGETS, CONDITIONAL_BUDGET)
              if str(budget) in labels
              and all(labels[str(budget)].get(str(seed)) == "recovered" for seed in SEEDS)]
    budget = min(common) if common else None
    candidate = bool(provenance_valid and restart_ok and budget is not None and budget < RESTART_BUDGET)
    return ("conditional_candidate", budget) if candidate else ("keep_current_rule", None)


def evidence_inventory(folder: Path) -> dict[str, str]:
    return {path.relative_to(folder).as_posix(): sha(path) for path in sorted(folder.rglob("*"))
            if path.is_file() and path.name not in {"result.json", "failure.json"}}


def result_document(folder: Path, protocol: dict[str, Any], commit: str, anchor: dict[str, Any],
                    labels: dict[str, dict[str, str]], verdicts: dict[str, dict[str, Any]],
                    costs: dict[str, Any], restart_matches_none: dict[str, bool],
                    provenance_valid: bool) -> dict[str, Any]:
    decision, budget = candidate_decision(labels, provenance_valid)
    timings = {arm: {seed: row["training_wall_s"] for seed, row in seeds.items()}
               for arm, seeds in costs["recovery_arms"].items()}
    time_comparison = {str(seed): {
        str(reset_budget): timings[str(reset_budget)][str(seed)] < timings["restart_300"][str(seed)]
        for reset_budget in (*RESET_BUDGETS, CONDITIONAL_BUDGET) if str(reset_budget) in timings
    } for seed in SEEDS}
    time_ratios = {str(seed): {
        str(reset_budget): timings[str(reset_budget)][str(seed)] / timings["restart_300"][str(seed)]
        for reset_budget in (*RESET_BUDGETS, CONDITIONAL_BUDGET) if str(reset_budget) in timings
    } for seed in SEEDS}
    prediction = {
        "100": dict(zip(map(str, SEEDS), protocol["prediction"]["reset_100"])),
        "150": dict(zip(map(str, SEEDS), protocol["prediction"]["reset_150"])),
        "restart_300": dict(zip(map(str, SEEDS), protocol["prediction"]["restart_300"])),
    }
    if "200" in labels:
        prediction["200"] = dict(zip(map(str, SEEDS), protocol["prediction"]["reset_200_if_needed"]))
    prediction_match = {arm: {seed: labels[arm][seed] == expected
                              for seed, expected in seeds.items()}
                        for arm, seeds in prediction.items()}
    return {"schema": "actor_reset_budget_result_v1", "status": "complete", "decision": decision,
            "candidate_budget_learning_iterations": budget, "budget_unit": protocol["budget_unit"],
            "labels": labels, "verdicts": verdicts,
            "prediction": prediction, "prediction_match": prediction_match,
            "branch_200_executed": "200" in labels, "provenance_valid": provenance_valid,
            "automatic_global_change": False,
            "wording": (protocol["candidate_wording"].replace("N회", f"{budget}회")
                        if decision == "conditional_candidate" else
                        protocol["restart_control_not_recovered_wording"]
                        if decision == "restart_control_not_recovered" else protocol["keep_wording"]),
            "costs": costs, "training_wall_scope": protocol["time_scope"],
            "reset_faster_than_same_seed_restart": time_comparison,
            "training_wall_ratio_vs_same_seed_restart": time_ratios,
            "restart_matches_source_none_sha": restart_matches_none,
            "restart_identity_used_for_candidate_decision": False,
            "time_used_for_candidate_decision": False, "model_calls": 0,
            "source_commit_sha": commit, "source_anchor": anchor,
            "protocol_sha256_lf": sha(PROTOCOL, True), "limits": protocol["limits"],
            "evidence_sha256": evidence_inventory(folder)}


def _run_arm(isaaclab: Path, folder: Path, request: dict[str, Any], anchor: dict[str, Any],
             seed: int, arm: str, reference: dict[str, Any], bands: dict[str, Any]) -> tuple[str, dict, dict]:
    ANCHOR.verify_source_anchor(anchor, ROOT, BOUND)
    arm_dir = folder / "arms" / f"s{seed}" / arm
    training = arm_dir / "train"
    write_new(arm_dir / "gpu_before_training.json", H._gpu(GPU.query_gpu(GPU_PROFILE), True))
    manifest_path = training / "manifest.json"
    write_new(manifest_path, manifest_for(folder, request, seed, arm))
    execution = OWNED.run_owned(training, _training_command(isaaclab, manifest_path), isaaclab,
                                FIXED["train_watchdog_s"])
    write_new(training / "execution.json", execution)
    evidence = _verify_training(folder, seed, arm, execution)
    report, eval_execution = H._fixed_eval(isaaclab, arm_dir / "eval", f"{arm}_s{seed}",
                                           Path(evidence["receipt"]["checkpoint"]["path"]))
    evidence["fixed_eval_wall_s"] = validated_wall_s(eval_execution, "고정 평가")
    ANCHOR.verify_source_anchor(anchor, ROOT, BOUND)
    label, verdict = H._label(report, reference, bands)
    return label, verdict, evidence


def execute(isaaclab: Path, tag: str = TAG) -> dict[str, Any]:
    protocol, commit = verify_protocol(isaaclab)
    if not isinstance(commit, str):
        raise ValueError("커밋된 source HEAD가 없다")
    folder, consumption = folder_for(tag), consumption_path(tag)
    if folder.exists() or consumption.exists():
        raise FileExistsError("Task25 tag/protocol은 한 번만 실행한다")
    freshness = SEED_GUARD.assert_fresh_seeds(SEEDS, isaaclab, repo=ROOT)
    admission = H._gpu(GPU.query_gpu(GPU_PROFILE), True)
    anchor = ANCHOR.make_source_anchor(commit, ROOT, BOUND)
    ANCHOR.verify_source_anchor(anchor, ROOT, BOUND)
    request = {"tag": tag, "source_commit_sha": commit, "source_anchor": anchor,
               "protocol_sha256_lf": sha(PROTOCOL, True), "isaaclab_root": str(isaaclab.resolve())}
    write_new(folder / "freshness.json", freshness)
    write_new(folder / "gpu_before.json", admission)
    write_new(folder / "request.json", request)
    write_new(consumption, {"tag": tag, "request_sha256": sha(folder / "request.json")})
    try:
        source_reports: dict[str, dict[str, Any]] = {}
        source_evidence: dict[str, dict[str, Any]] = {}
        source_costs: dict[str, dict[str, dict[str, float]]] = {}
        for seed in SEEDS:
            source_reports[str(seed)], source_evidence[str(seed)] = {}, {}
            source_costs[str(seed)] = {}
            seed_dir = folder / "source" / f"s{seed}"
            for fault in ("NONE", FAULT):
                ANCHOR.verify_source_anchor(anchor, ROOT, BOUND)
                evidence = H._source_train(isaaclab, seed_dir / fault.lower() / "train", seed, fault)
                report, eval_execution = H._fixed_eval(isaaclab, seed_dir / fault.lower() / "eval",
                                                        f"source_{fault.lower()}_s{seed}", Path(evidence["checkpoint"]))
                ANCHOR.verify_source_anchor(anchor, ROOT, BOUND)
                source_evidence[str(seed)][fault], source_reports[str(seed)][fault] = evidence, report
                train_execution = read(seed_dir / fault.lower() / "train/execution.json")
                train_wall_s = validated_wall_s(train_execution, "source 학습")
                eval_wall_s = validated_wall_s(eval_execution, "source 고정 평가")
                source_costs[str(seed)][fault] = {
                    "training_wall_s": train_wall_s,
                    "fixed_eval_wall_s": eval_wall_s,
                    "total_wall_s": train_wall_s + eval_wall_s,
                }
            write_new(seed_dir / "source_evidence.json", source_evidence[str(seed)])
            none, fault_report = source_reports[str(seed)]["NONE"], source_reports[str(seed)][FAULT]
            fault_verdict = RC.verdict(RC.compare(C._behavior(fault_report), C._behavior(none)), protocol["bands"])
            none_verdict = RC.verdict(RC.compare(C._behavior(none), C._behavior(none)), protocol["bands"])
            if not V4.walks(none) or none_verdict["label"] != "healthy" or fault_verdict["label"] != "unhealthy":
                raise RuntimeError("source precondition(NONE healthy walk/F1 unhealthy)이 실패했다")
            write_new(seed_dir / "precondition.json", {"none_walks": True, "none_label": "healthy",
                                                         "fault_label": "unhealthy", "fault_verdict": fault_verdict,
                                                         "none_verdict": none_verdict})
        write_new(folder / "source_selection.json", {
            "schema": "actor_reset_budget_source_selection_v1",
            "rule": "각 seed의 등록 F1 model_299를 모든 reset 예산의 공통 시작점으로 쓴다",
            "seeds": {str(seed): {
                "source_evidence_sha256": sha(folder / f"source/s{seed}/source_evidence.json"),
                "checkpoint_sha256": source_evidence[str(seed)][FAULT]["checkpoint_sha256"],
            } for seed in SEEDS},
        })
        labels: dict[str, dict[str, str]] = {str(budget): {} for budget in RESET_BUDGETS}
        labels["restart_300"] = {}
        verdicts: dict[str, dict[str, Any]] = {key: {} for key in labels}
        arm_costs: dict[str, dict[str, dict[str, float]]] = {key: {} for key in labels}
        initial_rows: dict[str, dict[str, dict[str, Any]]] = {str(seed): {} for seed in SEEDS}
        for budget in RESET_BUDGETS:
            arm = arm_name("reset", budget)
            for seed in SEEDS:
                label, verdict, evidence = _run_arm(
                    isaaclab, folder, request, anchor, seed, arm, source_reports[str(seed)]["NONE"], protocol["bands"])
                labels[str(budget)][str(seed)], verdicts[str(budget)][str(seed)] = label, verdict
                arm_costs[str(budget)][str(seed)] = {
                    "training_wall_s": evidence["training_wall_s"],
                    "fixed_eval_wall_s": evidence["fixed_eval_wall_s"],
                    "total_wall_s": evidence["training_wall_s"] + evidence["fixed_eval_wall_s"],
                }
                initial_rows[str(seed)][str(budget)] = evidence
        for key in remaining_arm_order(labels):
            budget = RESTART_BUDGET if key == "restart_300" else CONDITIONAL_BUDGET
            mode = "restart" if key == "restart_300" else "reset"
            if key == "200":
                labels[key], verdicts[key], arm_costs[key] = {}, {}, {}
            for seed in SEEDS:
                arm = arm_name(mode, budget)
                label, verdict, evidence = _run_arm(
                    isaaclab, folder, request, anchor, seed, arm, source_reports[str(seed)]["NONE"], protocol["bands"])
                labels[key][str(seed)], verdicts[key][str(seed)] = label, verdict
                arm_costs[key][str(seed)] = {
                    "training_wall_s": evidence["training_wall_s"],
                    "fixed_eval_wall_s": evidence["fixed_eval_wall_s"],
                    "total_wall_s": evidence["training_wall_s"] + evidence["fixed_eval_wall_s"],
                }
                if key == "200":
                    initial_rows[str(seed)][key] = evidence
        equivalence = {str(seed): validate_initial_equivalence(initial_rows[str(seed)]) for seed in SEEDS}
        write_new(folder / "initial_equivalence.json", equivalence)
        after = H._gpu(GPU.query_gpu(GPU_PROFILE), False)
        owned_pids = {pid for path in folder.rglob("execution.json")
                      for pid in read(path).get("owned_process_identities", {})}
        if any(str(row.get("pid")) in owned_pids for row in after["processes"]):
            raise RuntimeError("종료 뒤 owned GPU PID가 남았다")
        write_new(folder / "gpu_after.json", after)
        source_total = sum(row["total_wall_s"] for seeds in source_costs.values() for row in seeds.values())
        recovery_total = sum(row["total_wall_s"] for seeds in arm_costs.values() for row in seeds.values())
        costs = {"source_preparation": source_costs, "recovery_arms": arm_costs,
                 "source_preparation_total_wall_s": source_total,
                 "recovery_experiment_total_wall_s": recovery_total,
                 "whole_experiment_accounted_wall_s": source_total + recovery_total,
                 "whole_experiment_cheaper_claimed": False}
        restart_matches_none = {str(seed):
            read(folder / f"arms/s{seed}/restart_300/train/receipt.json")["checkpoint"]["sha256"]
            == source_evidence[str(seed)]["NONE"]["checkpoint_sha256"] for seed in SEEDS}
        result = result_document(folder, protocol, commit, anchor, labels, verdicts, costs,
                                 restart_matches_none, True)
        write_new(folder / "result.json", result)
        return result
    except BaseException as error:
        write_new(folder / "failure.json", {"type": type(error).__name__, "message": str(error)})
        raise


def _verify_source(isaaclab: Path, folder: Path, protocol: dict[str, Any]) -> dict[str, dict[str, Any]]:
    reports: dict[str, dict[str, Any]] = {}
    for seed in SEEDS:
        seed_dir = folder / "source" / f"s{seed}"
        evidence = read(seed_dir / "source_evidence.json")
        if set(evidence) != {"NONE", FAULT}:
            raise ValueError("source evidence arm 집합이 다르다")
        reports[str(seed)] = {}
        for fault in ("NONE", FAULT):
            stage = seed_dir / fault.lower()
            row = H.validate_source_evidence(isaaclab, stage / "train", seed, fault, evidence[fault])
            validated_wall_s(read(stage / "train/execution.json"), "source 학습")
            checkpoint = Path(row["checkpoint"])
            eval_execution = H.validate_fixed_eval_launch(
                isaaclab, stage / "eval", f"source_{fault.lower()}_s{seed}", checkpoint)
            validated_wall_s(eval_execution, "source 고정 평가")
            report = read(stage / "eval/report.json")
            C.validate_fixed_report(report, {"name": f"source_{fault.lower()}_s{seed}",
                                             "checkpoint_file": "model_299.pt",
                                             "checkpoint_sha256": row["checkpoint_sha256"],
                                             "action_scale": ACTION_SCALE,
                                             "protocol_sha256_lf": sha(FIXED_EVAL, True)})
            reports[str(seed)][fault] = report
        fault_verdict = RC.verdict(RC.compare(C._behavior(reports[str(seed)][FAULT]),
                                              C._behavior(reports[str(seed)]["NONE"])), protocol["bands"])
        none_verdict = RC.verdict(RC.compare(C._behavior(reports[str(seed)]["NONE"]),
                                             C._behavior(reports[str(seed)]["NONE"])), protocol["bands"])
        expected = {"none_walks": True, "none_label": "healthy", "fault_label": "unhealthy",
                    "fault_verdict": fault_verdict, "none_verdict": none_verdict}
        if (not V4.walks(reports[str(seed)]["NONE"]) or none_verdict["label"] != "healthy"
                or fault_verdict["label"] != "unhealthy" or read(seed_dir / "precondition.json") != expected):
            raise ValueError("source precondition replay가 다르다")
    expected_selection = {"schema": "actor_reset_budget_source_selection_v1",
                          "rule": "각 seed의 등록 F1 model_299를 모든 reset 예산의 공통 시작점으로 쓴다",
                          "seeds": {str(seed): {
                              "source_evidence_sha256": sha(folder / f"source/s{seed}/source_evidence.json"),
                              "checkpoint_sha256": read(folder / f"source/s{seed}/source_evidence.json")[FAULT]["checkpoint_sha256"],
                          } for seed in SEEDS}}
    if read(folder / "source_selection.json") != expected_selection:
        raise ValueError("source selection attestation이 다르다")
    return reports


def verify_result(isaaclab: Path, folder: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    request, anchor = validate_run_request(folder, isaaclab)
    validate_freshness(read(folder / "freshness.json"))
    H._gpu(read(folder / "gpu_before.json"), True)
    source_reports = _verify_source(isaaclab, folder, protocol)
    result = read(folder / "result.json")
    labels = validate_result_labels(result.get("labels"))
    if ("200" in labels) != should_run_200(labels):
        raise ValueError("조건부 200회 branch가 다르다")
    verdicts, arm_costs = {key: {} for key in labels}, {key: {} for key in labels}
    initial_rows = {str(seed): {} for seed in SEEDS}
    for key in labels:
        budget = RESTART_BUDGET if key == "restart_300" else int(key)
        arm = arm_name("restart" if key == "restart_300" else "reset", budget)
        for seed in SEEDS:
            arm_dir = folder / f"arms/s{seed}/{arm}"
            H._gpu(read(arm_dir / "gpu_before_training.json"), True)
            evidence = _verify_training(folder, seed, arm, read(arm_dir / "train/execution.json"))
            H.validate_fixed_eval_launch(isaaclab, arm_dir / "eval", f"{arm}_s{seed}",
                                         Path(evidence["receipt"]["checkpoint"]["path"]))
            report = read(arm_dir / "eval/report.json")
            C.validate_fixed_report(report, {"name": f"{arm}_s{seed}",
                                             "checkpoint_file": f"model_{evidence['receipt']['final_iteration']}.pt",
                                             "checkpoint_sha256": evidence["receipt"]["checkpoint"]["sha256"],
                                             "action_scale": ACTION_SCALE,
                                             "protocol_sha256_lf": sha(FIXED_EVAL, True)})
            label, verdict = H._label(report, source_reports[str(seed)]["NONE"], protocol["bands"])
            if labels[key].get(str(seed)) != label:
                raise ValueError("저장 recovery label이 replay와 다르다")
            eval_execution = read(arm_dir / "eval/execution.json")
            eval_wall_s = validated_wall_s(eval_execution, "고정 평가")
            verdicts[key][str(seed)] = verdict
            arm_costs[key][str(seed)] = {
                "training_wall_s": evidence["training_wall_s"],
                "fixed_eval_wall_s": eval_wall_s,
                "total_wall_s": evidence["training_wall_s"] + eval_wall_s,
            }
            if key != "restart_300":
                initial_rows[str(seed)][key] = evidence
    expected_equivalence = {str(seed): validate_initial_equivalence(initial_rows[str(seed)]) for seed in SEEDS}
    if read(folder / "initial_equivalence.json") != expected_equivalence:
        raise ValueError("초기 policy/optimizer/RNG equivalence가 다르다")
    source_costs = {str(seed): {
        fault: stage_cost(folder / f"source/s{seed}/{fault.lower()}", "source")
        for fault in ("NONE", FAULT)} for seed in SEEDS}
    source_total = sum(row["total_wall_s"] for seeds in source_costs.values() for row in seeds.values())
    recovery_total = sum(row["total_wall_s"] for seeds in arm_costs.values() for row in seeds.values())
    costs = {"source_preparation": source_costs, "recovery_arms": arm_costs,
             "source_preparation_total_wall_s": source_total,
             "recovery_experiment_total_wall_s": recovery_total,
             "whole_experiment_accounted_wall_s": source_total + recovery_total,
             "whole_experiment_cheaper_claimed": False}
    restart_matches_none = {str(seed):
        read(folder / f"arms/s{seed}/restart_300/train/receipt.json")["checkpoint"]["sha256"]
        == read(folder / f"source/s{seed}/source_evidence.json")["NONE"]["checkpoint_sha256"]
        for seed in SEEDS}
    expected = result_document(folder, protocol, anchor["commit"], anchor, labels, verdicts, costs,
                               restart_matches_none, True)
    if result != expected:
        raise ValueError("저장 result 전체가 재계산 결과와 다르다")
    after = H._gpu(read(folder / "gpu_after.json"), False)
    owned = {pid for path in folder.rglob("execution.json")
             for pid in read(path).get("owned_process_identities", {})}
    if any(str(row.get("pid")) in owned for row in after["processes"]):
        raise ValueError("result gpu_after에 owned PID가 남았다")
    return result


def verify(isaaclab: Path, tag: str | None = None) -> dict[str, Any]:
    protocol, commit = verify_protocol(isaaclab)
    out: dict[str, Any] = {"status": "ready", "source_commit_sha": commit,
                           "protocol_sha256_lf": sha(PROTOCOL, True)}
    if tag:
        result = verify_result(isaaclab, folder_for(tag), protocol)
        if result.get("status") != "complete":
            raise ValueError("저장 result가 완결되지 않았다")
        out["result"] = result
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "check", "verify", "execute"))
    parser.add_argument("--isaaclab", required=True, type=Path)
    parser.add_argument("--tag")
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare(args.isaaclab.resolve())
        elif args.command == "check":
            protocol, _ = verify_protocol(args.isaaclab.resolve(), committed=False)
            result = {"status": "pass", "committed": False,
                      "protocol_sha256_lf": sha(PROTOCOL, True), "schema": protocol["schema"]}
        elif args.command == "verify":
            result = verify(args.isaaclab.resolve(), args.tag)
        else:
            result = execute(args.isaaclab.resolve(), args.tag or TAG)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    except (ValueError, RuntimeError, FileNotFoundError, FileExistsError, subprocess.CalledProcessError) as error:
        print(f"거부: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
