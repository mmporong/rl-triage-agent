"""보상 결함 교정 뒤 세 가지 재설정 개입을 같은 v4 예산으로 비교한다.

이미 본 seed 2026·2028의 개발 실험이다. 저장된 결과를 보고 설계한 개입 분리
실험이며 메커니즘의 인과 식별, 일반화, GPU 절감을 주장하지 않는다.
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

import record_evidence_media as M  # noqa: E402
import recovery_reset_interventions as I  # noqa: E402
import run_runtime_readback_smoke as P  # noqa: E402
import runtime_readback_smoke as S  # noqa: E402
import run_training_correction as V3  # noqa: E402
import run_training_correction_v4 as V4  # noqa: E402
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

PROTOCOL = ROOT / "bench/protocols/recovery_reset_v1.json"
FIXED_EVAL = V3.FIXED_EVAL
PRIVATE = ROOT / "bench/private/codex_support_20261015/task1/runner/results"
SEEDS = (2026, 2028)
FAULT = "F1"
CORRECTION = "restore_body_frame_reward"
ARMS = I.INTERVENTIONS
ARM_SPECS = {
    "optimizer_only": {"intervention": "optimizer_only"},
    "critic_reset": {"intervention": "critic_reset"},
    "actor_last_layer_reset": {"intervention": "actor_last_layer_reset"},
}
FINAL_ITERATION = 598
CODE = V4.CODE | {
    "evals/run_recovery_reset.py",
    "bench/train_with_recovery_reset.py",
    "bench/recovery_reset_interventions.py",
}
V4_RESULTS = {seed: ROOT / f"evals/results/reward_v4_s{seed}_20261013" for seed in SEEDS}
FIXED = {
    "schema": "recovery_reset_protocol_v1",
    "revision": 1,
    "question": "보상 결함 교정 뒤 이어 학습 실패가 optimizer, critic, actor 말단 상태 중 어느 재설정에 반응하는가",
    "scope": "seen_two_seed_reward_fault_reset_intervention_development",
    "design_disclosure": "v4 seed 2026·2028 결과를 본 뒤 만든 개입 분리 실험이다",
    "mechanism_claim_limit": "저장 체크포인트 비교만으로 원인을 식별할 수 없어 세 개 최소 개입의 결과를 후속 가설 근거로만 쓴다",
    "literature_scope_limit": "참고 연구가 이 PPO 로봇 회복을 입증한 것은 아니다",
    "seeds": list(SEEDS), "task": C.TASK, "fault": FAULT, "correction": CORRECTION,
    "num_envs": 4096, "num_steps_per_env": 24, "max_iterations": 300,
    "resume_iteration": 299, "final_iteration": FINAL_ITERATION, "action_scale": 0.25,
    "arms": list(ARMS), "arm_specs": ARM_SPECS,
    "reset_seed_rule": "100000 + training_seed * 10 + registered_arm_index",
    "interventions": {
        "optimizer_only": "정책 텐서는 유지하고 공유 Adam 상태를 모두 지운다",
        "critic_reset": "critic의 모든 Linear를 재초기화하고 해당 Adam 상태를 지운다. actor와 std는 유지한다",
        "actor_last_layer_reset": "actor의 마지막 Linear만 재초기화하고 해당 Adam 상태를 지운다. critic과 std는 유지한다",
    },
    "excluded_followups": "활성값 표본이 필요한 ReDo·CBP와 추가 계수를 요구하는 shrink-and-perturb는 이 실험에 넣지 않는다",
    "reference": "동일 seed v4 clean_resume_reference의 300회 이어 학습 고정 평가",
    "registered_v4_comparisons": ["resume_corrected", "restart_corrected"],
    "recovery_mode_change_criteria": {
        "candidate": "한 개입이 두 seed 모두 recovered이고 v4 resume_corrected는 두 seed 모두 not_recovered이면 새 seed 확인 후보로 등록한다",
        "tie_break": ["optimizer_only", "actor_last_layer_reset", "critic_reset"],
        "mixed": "seed 사이 라벨이 다르면 회복 규칙을 바꾸지 않고 메커니즘 미확정으로 둔다",
        "none": "세 개입이 모두 두 seed 회복에 실패하면 현재 관찰 범위에서는 restart만 회복된 경로로 유지한다",
    },
    "cost_claim_limit": "모든 개입 arm은 300회 예산이다. 실제 시간 측정 전에는 재학습보다 싸거나 GPU를 절감한다고 쓰지 않는다",
    "label_rule": "개입 arm이 v4 clean resume 기준 대비 healthy면 recovered, unhealthy면 not_recovered, 계약 불일치면 invalid",
    "predictions": {arm: None for arm in ARMS},
    "result_wording": {
        "recovered": "이미 본 seed의 개발 실행에서 이 개입 뒤 300회 이어 학습 결과가 등록된 정상 이어 학습 기준의 고정 행동 허용 범위를 통과했다. 원인 식별·일반화·GPU 절감 주장이 아니다.",
        "not_recovered": "이미 본 seed의 개발 실행에서 이 개입 뒤 300회 이어 학습 결과가 등록된 정상 이어 학습 기준의 고정 행동 허용 범위를 통과하지 못했다. 원인 식별·일반화·GPU 절감 주장이 아니다.",
        "invalid": "출처·개입·예산·평가·종료 계약이 맞지 않거나 실행되지 않아 판정을 보류한다.",
    },
    "semantic_retries": 0, "train_limit_per_arm": 1, "eval_limit_per_arm": 1,
    "train_watchdog_s": 1800, "eval_watchdog_s": 360,
    "gpu_admission_profile": V4.FIXED["gpu_admission_profile"],
    "isaaclab_commit": V4.FIXED["isaaclab_commit"], "rsl_rl_version": V4.FIXED["rsl_rl_version"],
    "approval_actor": "claude_operator_user_authorized_test",
    "model_calls": 0, "model_selected_correction": False,
}
DYNAMIC = {"source_files", "sdk_sources", "rsl_sources", "correction_source_contract",
           "fixed_eval_sha256_lf", "resume_inputs", "v4_reference_hashes"}


def intervention_for_arm(arm: str) -> str:
    if arm not in ARM_SPECS:
        raise ValueError("unregistered recovery reset arm")
    return ARM_SPECS[arm]["intervention"]


def reset_seed(seed: int, intervention: str) -> int:
    if seed not in SEEDS or intervention not in ARMS:
        raise ValueError("unregistered seed or intervention")
    return 100000 + seed * 10 + ARMS.index(intervention)


def source_maps(isaaclab: Path) -> dict:
    maps = V4.source_maps(isaaclab)
    maps["source_files"] = {name: sha(ROOT / name, True) for name in sorted(CODE)}
    return maps


def _reference_paths(seed: int) -> dict[str, Path]:
    folder = V4_RESULTS[seed]
    return {
        "report": folder / "reports/clean_resume_reference.json",
        "original_report": folder / "original_reports/clean_reference.json",
        "result": folder / "result.json",
        "source_inputs": folder / "source_inputs.json",
        "training_evidence": folder / "training_evidence.json",
    }


def _validated_reference(seed: int) -> dict:
    paths = _reference_paths(seed)
    source = read(paths["source_inputs"])
    evidence = read(paths["training_evidence"])
    report = read(paths["report"])
    original = read(paths["original_report"])
    if source.get("seed") != seed or source.get("fixed_eval_sha256_lf") != sha(FIXED_EVAL, True):
        raise ValueError("v4 reference source binding differs")
    resume = evidence.get("arms", {}).get("clean_resume_reference", {})
    C.validate_fixed_report(report, {
        "name": "clean_resume_reference", "checkpoint_file": "model_598.pt",
        "checkpoint_sha256": resume.get("checkpoint", {}).get("sha256"),
        "action_scale": FIXED["action_scale"], "protocol_sha256_lf": sha(FIXED_EVAL, True),
    })
    C.validate_fixed_report(original, {
        "name": source.get("runs", {}).get("NONE", {}).get("run"), "checkpoint_file": "model_299.pt",
        "checkpoint_sha256": source.get("runs", {}).get("NONE", {}).get("hashes", {}).get("checkpoint"),
        "action_scale": FIXED["action_scale"], "protocol_sha256_lf": sha(FIXED_EVAL, True),
    })
    if not V4.walks(report):
        raise ValueError("v4 clean resume reference does not walk")
    verdict = RC.verdict(RC.compare(C._behavior(report), C._behavior(original)), source.get("bands"))
    if verdict.get("label") != "healthy":
        raise ValueError("v4 clean resume reference is not healthy relative to original clean")
    return report


def reference_hashes() -> dict:
    rows = {}
    for seed in SEEDS:
        paths = _reference_paths(seed)
        if not all(path.is_file() for path in paths.values()):
            raise FileNotFoundError(f"v4 reference is missing for seed {seed}")
        _validated_reference(seed)
        result = read(paths["result"])
        if result.get("numeric", {}).get("labels", {}).get("resume_corrected") != "not_recovered":
            raise ValueError("v4 development outcome changed")
        rows[str(seed)] = {name: sha(path) for name, path in paths.items()}
    return rows


def prepare(isaaclab: Path) -> dict:
    v4 = read(V4.PROTOCOL)
    contract = v4["correction_source_contract"]
    check_contract(contract, isaaclab)
    document = {
        **deepcopy(FIXED), **source_maps(isaaclab),
        "correction_source_contract": contract,
        "fixed_eval_sha256_lf": sha(FIXED_EVAL, True),
        "resume_inputs": V4.source_inputs(isaaclab),
        "v4_reference_hashes": reference_hashes(),
    }
    write_new(PROTOCOL, document)
    return {"status": "prepared_commit_before_execution", "path": str(PROTOCOL)}


def verify_protocol(isaaclab: Path) -> tuple[dict, str]:
    protocol = read(PROTOCOL)
    if set(protocol) != set(FIXED) | DYNAMIC:
        raise ValueError("protocol fields differ")
    if any(type(protocol[key]) is not type(value) or protocol[key] != value for key, value in FIXED.items()):
        raise ValueError("fixed recovery reset contract changed")
    check_contract(protocol["correction_source_contract"], isaaclab)
    if any(protocol[key] != value for key, value in source_maps(isaaclab).items()):
        raise ValueError("training source hashes changed")
    if protocol["fixed_eval_sha256_lf"] != sha(FIXED_EVAL, True):
        raise ValueError("fixed behavior protocol changed")
    if protocol["resume_inputs"] != V4.source_inputs(isaaclab):
        raise ValueError("bound checkpoint, config, or original evaluation changed")
    if protocol["v4_reference_hashes"] != reference_hashes():
        raise ValueError("v4 comparison reference changed")
    if S._git_head(isaaclab) != FIXED["isaaclab_commit"] or not S._git_tracked_clean(isaaclab):
        raise ValueError("Isaac Lab checkout differs")
    commit = P.verify_committed(ROOT, CODE | {PROTOCOL.relative_to(ROOT).as_posix(),
                                              FIXED_EVAL.relative_to(ROOT).as_posix()})
    return protocol, commit


def folder_for(tag: str) -> Path:
    if not isinstance(tag, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,39}", tag):
        raise ValueError("invalid recovery reset tag")
    folder = PRIVATE / tag
    if folder.is_symlink() or folder.resolve().parent != PRIVATE.resolve():
        raise ValueError("training output escapes its root")
    return folder


def manifest_for(folder: Path, request: dict, arm: str) -> dict:
    intervention_for_arm(arm)
    seed = request["seed"]
    if seed not in SEEDS:
        raise ValueError("unregistered seed")
    paths = V4.source_paths(Path(request["isaaclab_root"]), seed, FAULT)
    bound = read(PROTOCOL)["resume_inputs"]["seeds"][str(seed)]["runs"][FAULT]["hashes"]
    checkpoint = paths["checkpoint"].resolve()
    return {
        "schema": "training_correction_manifest_v2", "tag": folder.name, "seed": seed, "arm": arm,
        "recovery_mode": "resume", "source_commit_sha": request["source_commit_sha"],
        "protocol_sha256_lf": request["protocol_sha256_lf"], "isaaclab_root": request["isaaclab_root"],
        "protocol_path": str(PROTOCOL.resolve()), "fault": FAULT, "correction_id": CORRECTION,
        "num_envs": FIXED["num_envs"], "max_iterations": FIXED["max_iterations"], "task": FIXED["task"],
        "run_name": f"{folder.name}_{arm}", "output_dir": str((folder / arm / "train").resolve()),
        "final_iteration": FINAL_ITERATION,
        "resume_checkpoint_path": str(checkpoint), "resume_checkpoint_sha256": bound["checkpoint"],
        "resume_iteration": FIXED["resume_iteration"],
        "load_run_regex": "^" + re.escape(checkpoint.parent.name) + "$",
        "checkpoint_regex": "^model_299\\.pt$",
        "source_env_config_path": str(paths["environment"].resolve()),
        "source_agent_config_path": str(paths["agent"].resolve()),
        "config_source_fault": FAULT, "init_noise_std": None,
    }


def validate_training_configuration(log_dir: Path, manifest: dict) -> None:
    protocol = read(Path(manifest["protocol_path"]))
    hashes = protocol["resume_inputs"]["seeds"][str(manifest["seed"])]["runs"][FAULT]["hashes"]
    source_env = Path(manifest["source_env_config_path"])
    source_agent = Path(manifest["source_agent_config_path"])
    for path, key in ((source_env, "environment"), (source_agent, "agent")):
        if sha(path) != hashes[key]:
            raise ValueError("source configuration changed before learning")
    original_env, original_agent = load_config(source_env), load_config(source_agent)
    current_env = load_config(Path(log_dir) / "params/env.yaml")
    current_agent = load_config(Path(log_dir) / "params/agent.yaml")
    if current_env != original_env:
        raise ValueError("environment configuration changed outside the registered correction")
    allowed = {"run_name", "resume", "load_run", "load_checkpoint", "max_iterations"}
    if ({key: value for key, value in original_agent.items() if key not in allowed}
            != {key: value for key, value in current_agent.items() if key not in allowed}):
        raise ValueError("agent configuration changed outside registered resume arguments")
    expected = {"run_name": manifest["run_name"], "resume": True,
                "load_run": manifest["load_run_regex"], "load_checkpoint": manifest["checkpoint_regex"],
                "max_iterations": FIXED["max_iterations"]}
    if any(type(current_agent.get(key)) is not type(value) or current_agent.get(key) != value
           for key, value in expected.items()):
        raise ValueError("agent configuration differs from the reset manifest")
    if (original_agent.get("seed") != manifest["seed"]
            or original_agent.get("num_steps_per_env") != FIXED["num_steps_per_env"]
            or original_agent.get("max_iterations") != FIXED["max_iterations"]
            or original_env.get("seed") != manifest["seed"]
            or original_env.get("scene", {}).get("num_envs") != FIXED["num_envs"]):
        raise ValueError("original training seed, environment count, or step count differs")


def validate_manifest(path: Path) -> tuple[dict, dict]:
    manifest = read(path)
    if set(manifest) != R.MANIFEST_FIELDS or R.validate_manifest_mode(manifest) != "resume":
        raise ValueError("manifest fields or recovery mode differ")
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
    if approval != {"request_sha256": sha(folder / "request.json"), "actor": FIXED["approval_actor"],
                    "identity_attested": False}:
        raise ValueError("operator authorization record differs")
    if consumed != {"approval_sha256": sha(folder / "approval.json"), "arm": manifest["arm"],
                    "train_limit": 1, "eval_limit": 1}:
        raise ValueError("arm consumption record differs")
    return manifest, protocol


def _sha_text(value) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def validate_intervention(row: dict, arm: str, source_state: dict, initial: dict) -> None:
    expected_keys = {
        "schema", "intervention", "reset_seed", "target_modules", "target_parameter_count",
        "optimizer_param_indices", "optimizer_moment_indices_present_before",
        "optimizer_moments_present_after", "parameter_identity_preserved", "policy_state_before",
        "policy_state_after", "optimizer_state_before", "optimizer_state_expected_after",
        "optimizer_state_after", "optimizer_param_groups_before", "optimizer_param_groups_after",
    }
    intervention = intervention_for_arm(arm)
    if not isinstance(row, dict) or set(row) != expected_keys:
        raise ValueError("intervention readback fields differ")
    if (row["schema"] != "recovery_reset_intervention_v1" or row["intervention"] != intervention
            or row["reset_seed"] != reset_seed(initial["observed"]["environment_seed"], intervention)
            or row["policy_state_before"] != source_state["policy_state_sha256"]
            or row["optimizer_state_before"] != source_state["optimizer_state_sha256"]
            or row["policy_state_after"] != initial["policy_state"]["sha256"]
            or row["optimizer_state_after"] != initial["optimizer_state"]["sha256"]
            or row["optimizer_state_after"] != row["optimizer_state_expected_after"]
            or row["optimizer_param_groups_before"] != row["optimizer_param_groups_after"]
            or row["optimizer_moments_present_after"] is not False
            or row["parameter_identity_preserved"] is not True):
        raise ValueError("intervention is not bound to source and initial states")
    indices = row["optimizer_param_indices"]
    if (not isinstance(indices, list) or not indices or any(isinstance(item, bool) or not isinstance(item, int)
                                                            for item in indices)
            or indices != sorted(set(indices)) or row["target_parameter_count"] != len(indices)
            or any(not _sha_text(row[key]) for key in ("policy_state_before", "policy_state_after",
                                                       "optimizer_state_before", "optimizer_state_after"))):
        raise ValueError("intervention target inventory is incomplete")
    if intervention == "optimizer_only":
        if row["target_modules"] or row["policy_state_after"] != row["policy_state_before"]:
            raise ValueError("optimizer-only intervention changed policy")
    elif not row["target_modules"] or row["policy_state_after"] == row["policy_state_before"]:
        raise ValueError("network intervention did not change its policy target")


def verify_training(folder: Path, arm: str, execution: dict) -> dict:
    training = folder / arm / "train"
    if not execution_valid(execution):
        raise ValueError("training execution failed")
    manifest, protocol = validate_manifest(training / "manifest.json")
    receipt = read(training / "receipt.json")
    initial = read(training / "initial.json")
    intervention = read(training / "intervention.json")
    if (receipt.get("completed") is not True or receipt.get("schema") != "training_correction_receipt_v2"
            or initial.get("schema") != "training_correction_initial_v2" or initial.get("recovery_mode") != "resume"):
        raise ValueError("training receipt or initial schema differs")
    for key in ("tag", "seed", "arm", "recovery_mode", "fault", "correction_id", "source_commit_sha",
                "protocol_sha256_lf", "task", "run_name", "final_iteration"):
        if receipt.get(key) != manifest[key]:
            raise ValueError(f"training receipt binding differs: {key}")
    for filename, field in (("manifest.json", "manifest_sha256"), ("initial.json", "initial_sha256"),
                            ("correction.json", "correction_sha256")):
        if receipt.get(field) != sha(training / filename):
            raise ValueError("training raw artifact changed")
    identity = receipt["process_identity"]
    if execution["owned_process_identities"].get(str(identity["process_id"])) != identity["process_creation_time_filetime"]:
        raise ValueError("training receipt came from an unowned process")
    steps = FIXED["max_iterations"] * FIXED["num_steps_per_env"]
    expected_resume = {"path": manifest["resume_checkpoint_path"], "sha256": manifest["resume_checkpoint_sha256"]}
    source_state = receipt.get("source_checkpoint_state")
    if (receipt.get("actual_control_steps") != steps or receipt.get("aggregate_env_steps") != steps * FIXED["num_envs"]
            or receipt.get("environment_make_calls") != 1 or receipt.get("checkpoint_load_calls") != 1
            or receipt.get("resume_checkpoint") != expected_resume
            or initial.get("resume_checkpoint") != {**expected_resume, "load_optimizer": True}
            or receipt.get("resume_iteration") != FIXED["resume_iteration"]
            or receipt.get("start_iteration") != FIXED["resume_iteration"]
            or receipt.get("final_iteration") != FINAL_ITERATION
            or not isinstance(source_state, dict) or source_state != initial.get("source_checkpoint_state")
            or initial.get("std_reset") is not None or receipt.get("std_reset_applied") is not False):
        raise ValueError("resume checkpoint, budget, or iteration proof differs")
    expected_budget = {"seed": manifest["seed"], "num_envs": FIXED["num_envs"],
                       "max_iterations": FIXED["max_iterations"],
                       "num_steps_per_env": FIXED["num_steps_per_env"], "expected_control_steps": steps}
    if receipt.get("budget") != expected_budget:
        raise ValueError("training budget receipt differs")
    for state, iteration in ((source_state, FIXED["resume_iteration"]),
                             (receipt.get("final_checkpoint_state"), FINAL_ITERATION)):
        if (not isinstance(state, dict) or state.get("iteration") != iteration
                or any(not _sha_text(state.get(key))
                       for key in ("policy_state_sha256", "optimizer_state_sha256"))):
            raise ValueError("checkpoint state proof is incomplete")
    expected_scope = {"policy_optimizer_restored": True, "full_environment_state_restored": False,
                      "rng_state_restored_from_checkpoint": False, "fresh_initialization": False,
                      "action_std_reset": False}
    rng = initial.get("rng_state_before_learn")
    if (receipt.get("resume_scope") != expected_scope or initial.get("init_at_random_ep_len") is not True
            or initial.get("full_environment_state_restored") is not False
            or initial.get("rng_state_restored_from_checkpoint") is not False
            or receipt.get("rng_state_before_learn") != rng or not V4._rng_ok(rng)):
        raise ValueError("resume scope or RNG observation differs")
    observed = {"environment_seed": manifest["seed"], "environment_num_envs": FIXED["num_envs"],
                "action_scale": FIXED["action_scale"], "agent_seed": manifest["seed"],
                "agent_max_iterations": FIXED["max_iterations"],
                "runner_num_steps_per_env": FIXED["num_steps_per_env"]}
    if (receipt.get("observed") != observed or initial.get("observed") != observed
            or receipt.get("actual_action_scale") != FIXED["action_scale"]):
        raise ValueError("actual training configuration differs")
    versions = receipt.get("runtime_versions")
    runtime = protocol["correction_source_contract"]["runtime"]
    expected_runtime = protocol["resume_inputs"]["runtime_versions"]
    if (not isinstance(versions, dict) or versions != initial.get("runtime_versions")
            or versions.get("python") != ".".join(map(str, runtime["version"]))
            or versions.get("python_cache_tag") != runtime["cache_tag"]
            or versions.get("rsl_rl_distribution") != FIXED["rsl_rl_version"]
            or versions.get("torch_distribution") != expected_runtime["torch"]
            or versions.get("isaaclab_distribution") != expected_runtime["isaaclab"]
            or versions.get("isaaclab_source_commit") != FIXED["isaaclab_commit"]):
        raise ValueError("actual training runtime differs")
    validate_intervention(intervention, arm, source_state, initial)
    correction = read(training / "correction.json")
    validate_readback(correction, FAULT, CORRECTION)
    for key, target in protocol["correction_source_contract"]["targets"].items():
        if correction["before_fault"][key] != target["clean_fingerprint"]:
            raise ValueError("clean live target differs from committed SDK source")
    log_root = (Path(manifest["isaaclab_root"]) / "logs/rsl_rl/unitree_go2_flat").resolve()
    checkpoint = Path(receipt["checkpoint"]["path"])
    if (checkpoint.resolve().parent.parent != log_root or not checkpoint.parent.name.endswith("_" + manifest["run_name"])
            or checkpoint.name != f"model_{FINAL_ITERATION}.pt" or Path(initial["log_dir"]).resolve() != checkpoint.parent.resolve()):
        raise ValueError("training checkpoint is outside its exact run")
    if set(receipt.get("configs", {})) != {"environment", "agent"}:
        raise ValueError("training configuration artifacts are incomplete")
    for key, filename in (("environment", "env.yaml"), ("agent", "agent.yaml")):
        if Path(receipt["configs"][key]["path"]).resolve() != (checkpoint.parent / "params" / filename).resolve():
            raise ValueError("training config path is outside its exact run")
    for artifact in [receipt["checkpoint"], *receipt["configs"].values()]:
        if sha(Path(artifact["path"])) != artifact["sha256"]:
            raise ValueError("checkpoint or configuration hash differs")
    validate_training_configuration(checkpoint.parent, manifest)
    return receipt


def _validate_report(report: dict, name: str, checkpoint_sha: str) -> dict:
    return C.validate_fixed_report(report, {
        "name": name, "checkpoint_file": f"model_{FINAL_ITERATION}.pt",
        "checkpoint_sha256": checkpoint_sha, "action_scale": FIXED["action_scale"],
        "protocol_sha256_lf": sha(FIXED_EVAL, True),
    })


def summarize(seed: int, reports: dict, failed_arm: str | None = None) -> dict:
    reference = _validated_reference(seed)
    reference_behavior = C._behavior(reference)
    labels, verdicts = {}, {}
    for arm in ARMS:
        if arm not in reports:
            labels[arm] = "invalid"
        else:
            verdicts[arm] = RC.verdict(RC.compare(C._behavior(reports[arm]), reference_behavior),
                                       read(PROTOCOL)["resume_inputs"]["bands"])
            labels[arm] = "recovered" if verdicts[arm]["label"] == "healthy" else "not_recovered"
    return {
        "status": "complete" if all(label != "invalid" for label in labels.values()) else "partial",
        "seed": seed, "scope": FIXED["scope"], "failed_arm": failed_arm,
        "labels": labels, "verdicts": verdicts, "predictions": FIXED["predictions"],
        "wording": {arm: FIXED["result_wording"][labels[arm]] for arm in ARMS},
        "causal_mechanism_identified": False, "generalization_proven": False, "gpu_saving_claimed": False,
    }


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
        if any(path.name.endswith(f"_{tag}_{arm}") for path in log_root.iterdir()):
            raise FileExistsError("an SDK run already exists for this arm")
    for path in (ROOT, isaaclab, folder):
        if any(character in str(path) for character in '\r\n"%&^'):
            raise ValueError("unsafe Windows batch path")
    admission = M.query_gpu(FIXED["gpu_admission_profile"])
    if admission["busy"]:
        return {"status": "gpu_busy_no_launch", "sdk_jobs_launched": 0}
    request = {"tag": tag, "seed": seed, "source_commit_sha": commit,
               "protocol_sha256_lf": sha(PROTOCOL, True), "isaaclab_root": str(isaaclab.resolve()),
               "selection_source": "registered_host_reset_comparison_not_model", "bundle_key": key}
    write_new(folder / "request.json", request)
    write_new(folder / "approval.json", {"request_sha256": sha(folder / "request.json"),
                                         "actor": FIXED["approval_actor"], "identity_attested": False})
    write_new(consumption, {"request_sha256": sha(folder / "request.json"), "tag": tag})
    reports, costs = {}, {}
    failed_arm = None
    for arm in ARMS:
        stage = "preflight"
        try:
            current, current_commit = verify_protocol(isaaclab)
            if current != protocol or current_commit != commit:
                raise ValueError("source changed during experiment")
            before = M.query_gpu(FIXED["gpu_admission_profile"])
            if before["busy"]:
                raise RuntimeError("GPU busy before next training arm")
            arm_dir = folder / arm
            training, evaluation = arm_dir / "train", arm_dir / "eval"
            write_new(arm_dir / "gpu_before.json", before)
            write_new(arm_dir / "consumed.json", {"approval_sha256": sha(folder / "approval.json"), "arm": arm,
                                                  "train_limit": 1, "eval_limit": 1})
            write_new(training / "manifest.json", manifest_for(folder, request, arm))
            argv = [str(isaaclab / "_isaac_sim/python.bat"), str(ROOT / "bench/train_with_recovery_reset.py"),
                    "--manifest", str(training / "manifest.json")]
            stage = "training"
            execution = P.run_owned(training, argv, isaaclab, FIXED["train_watchdog_s"])
            write_new(training / "execution.json", execution)
            receipt = verify_training(folder, arm, execution)
            evaluation.mkdir()
            write_new(evaluation / "jobs.json", {"action_scale": FIXED["action_scale"], "jobs": [{
                "name": arm, "checkpoint": receipt["checkpoint"]["path"],
                "output": str(evaluation / "report.json")} ]})
            if M.query_gpu(FIXED["gpu_admission_profile"])["busy"]:
                raise RuntimeError("GPU busy before fixed evaluation")
            stage = "fixed_evaluation"
            argv = [str(isaaclab / "_isaac_sim/python.bat"), str(ROOT / "evals/fixed_eval.py"),
                    "--protocol", str(FIXED_EVAL), "--jobs", str(evaluation / "jobs.json"), "--headless"]
            eval_execution = P.run_owned(evaluation, argv, isaaclab, FIXED["eval_watchdog_s"])
            write_new(evaluation / "execution.json", eval_execution)
            if not execution_valid(eval_execution):
                raise ValueError("fixed evaluation did not complete")
            reports[arm] = _validate_report(read(evaluation / "report.json"), arm, receipt["checkpoint"]["sha256"])
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
            write_new(arm_dir / "parent.json", {
                "status": "pass", "training_receipt_sha256": sha(training / "receipt.json"),
                "intervention_sha256": sha(training / "intervention.json"),
                "fixed_report_sha256": sha(evaluation / "report.json"), **costs[arm]})
        except Exception as error:
            failed_arm = arm
            reports.pop(arm, None)
            write_new(folder / "failure.json", {"arm": arm, "stage": stage,
                                                "error_type": type(error).__name__, "detail": str(error)})
            break
    result = summarize(seed, reports, failed_arm)
    result.update(tag=tag, costs=costs, model_calls=0, source_commit_sha=commit,
                  protocol_sha256_lf=request["protocol_sha256_lf"])
    result["arm_evidence"] = {arm: {"parent_sha256": sha(folder / arm / "parent.json"),
                                    **read(folder / arm / "parent.json")}
                              for arm in ARMS if (folder / arm / "parent.json").is_file()}
    write_new(folder / "result.json", result)
    return result


def _validate_result_path(path: Path) -> tuple[dict, dict]:
    path = path.resolve()
    row = read(path)
    if not isinstance(row, dict) or row.get("tag") is None:
        raise ValueError("aggregate result is missing its tag")
    folder = folder_for(row["tag"])
    if path != (folder / "result.json").resolve():
        raise ValueError("aggregate result path is outside its registered tag")
    seed = row.get("seed")
    if (seed not in SEEDS or row.get("status") != "complete" or row.get("scope") != FIXED["scope"]
            or row.get("failed_arm") is not None or set(row.get("labels", {})) != set(ARMS)
            or any(label not in {"recovered", "not_recovered"} for label in row["labels"].values())
            or row.get("predictions") != FIXED["predictions"] or row.get("model_calls") != 0
            or set(row.get("costs", {})) != set(ARMS)
            or row.get("causal_mechanism_identified") is not False
            or row.get("generalization_proven") is not False or row.get("gpu_saving_claimed") is not False):
        raise ValueError("aggregate result contract differs")
    protocol, commit = verify_protocol(Path(read(folder / "request.json")["isaaclab_root"]))
    if (row.get("source_commit_sha") != commit or row.get("protocol_sha256_lf") != sha(PROTOCOL, True)
            or set(row.get("arm_evidence", {})) != set(ARMS)):
        raise ValueError("aggregate source or arm evidence binding differs")
    reports = {}
    for arm in ARMS:
        training = folder / arm / "train"
        evaluation = folder / arm / "eval"
        execution = read(training / "execution.json")
        eval_execution = read(evaluation / "execution.json")
        if not execution_valid(eval_execution):
            raise ValueError("aggregate fixed evaluation execution failed")
        receipt = verify_training(folder, arm, execution)
        report = _validate_report(read(evaluation / "report.json"), arm, receipt["checkpoint"]["sha256"])
        parent_path = folder / arm / "parent.json"
        parent = read(parent_path)
        expected_parent = {"parent_sha256": sha(parent_path), **parent}
        expected_cost = {"train_parent_wall_s": execution["wall_s"],
                         "eval_parent_wall_s": eval_execution["wall_s"]}
        after = read(folder / arm / "gpu_after.json")
        owned = set(execution["owned_process_identities"]) | set(eval_execution["owned_process_identities"])
        if (row["arm_evidence"][arm] != expected_parent or parent.get("status") != "pass"
                or row["costs"][arm] != expected_cost
                or any(str(process["pid"]) in owned for process in after["processes"])
                or parent.get("training_receipt_sha256") != sha(training / "receipt.json")
                or parent.get("intervention_sha256") != sha(training / "intervention.json")
                or parent.get("fixed_report_sha256") != sha(evaluation / "report.json")):
            raise ValueError("aggregate arm evidence changed")
        reports[arm] = report
    recomputed = summarize(seed, reports)
    for key in ("status", "seed", "scope", "failed_arm", "labels", "verdicts", "predictions", "wording",
                "causal_mechanism_identified", "generalization_proven", "gpu_saving_claimed"):
        if row.get(key) != recomputed.get(key):
            raise ValueError(f"aggregate result replay differs: {key}")
    return row, {"path": str(path), "sha256": sha(path)}


def aggregate(result_paths: list[Path]) -> dict:
    validated = [_validate_result_path(path) for path in result_paths]
    rows = [item[0] for item in validated]
    if sorted(row["seed"] for row in rows) != list(SEEDS) or len(rows) != len(SEEDS):
        raise ValueError("aggregate requires both registered seeds once")
    if len({row["source_commit_sha"] for row in rows}) != 1 or len({row["protocol_sha256_lf"] for row in rows}) != 1:
        raise ValueError("aggregate results come from different sources")
    v4 = {}
    for seed in SEEDS:
        numeric = read(_reference_paths(seed)["result"])["numeric"]
        v4[str(seed)] = {arm: {"label": numeric["labels"][arm],
                              "behavior": numeric["behavior"][arm],
                              "verdict": numeric["verdicts"][arm]}
                         for arm in FIXED["registered_v4_comparisons"]}
    return {"status": "complete", "seeds": list(SEEDS), "scope": FIXED["scope"],
            "source_commit_sha": rows[0]["source_commit_sha"],
            "protocol_sha256_lf": rows[0]["protocol_sha256_lf"],
            "result_artifacts": {str(row["seed"]): artifact for row, (_, artifact) in zip(rows, validated)},
            "arms": {arm: {str(row["seed"]): row["labels"][arm] for row in rows} for arm in ARMS},
            "v4_comparisons": v4,
            "recovery_mode_change_criteria": FIXED["recovery_mode_change_criteria"],
            "causal_mechanism_identified": False, "generalization_proven": False, "gpu_saving_claimed": False}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify", "execute", "aggregate"))
    parser.add_argument("--isaaclab", type=Path, default=Path.home() / "IsaacLab")
    parser.add_argument("--tag")
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--result", type=Path, action="append")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare(args.isaaclab)
    elif args.command == "verify":
        _, commit = verify_protocol(args.isaaclab)
        result = {"status": "pass", "source_commit_sha": commit}
    elif args.command == "execute":
        if args.tag is None or args.seed is None:
            parser.error("execute requires --tag and --seed")
        result = execute(args.tag, args.seed, args.isaaclab)
    else:
        if not args.result:
            parser.error("aggregate requires --result twice")
        result = aggregate(args.result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if args.command in {"execute", "aggregate"} and result.get("status") != "complete" else 0


if __name__ == "__main__":
    raise SystemExit(main())
