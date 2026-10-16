"""처음 쓰는 F1 seed에서 actor 말단 재설정 후보를 조건부 확인한다."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "bench", ROOT / "evals"):
    sys.path.insert(0, str(directory))

import preregistered_seed_guard as SEED_GUARD  # noqa: E402
import record_evidence_media as GPU  # noqa: E402
import run_runtime_readback_smoke as OWNED  # noqa: E402
import run_recovery_reset_v2 as ANCHOR  # noqa: E402
import run_training_correction as V3  # noqa: E402
import run_training_correction_v4 as V4  # noqa: E402
import train_with_recovery_mode as R  # noqa: E402
from rl_triage import grounded_recovery as C  # noqa: E402
from rl_triage import recovery as RC  # noqa: E402

PROTOCOL = ROOT / "bench/protocols/novel_actor_reset_confirmation_v1.json"
FIXED_EVAL = ROOT / "bench/protocols/fixed_eval_v1.json"
PRIVATE = ROOT / "bench/private/codex_support_20261015/task12"
RESULTS = PRIVATE / "runs"
SEEDS = (2063, 2064)
ARMS = ("continuation", "actor_last_layer_reset")
FAULT = "F1"
CORRECTION = "restore_body_frame_reward"
TASK = C.TASK
NUM_ENVS, MAX_ITERATIONS, NUM_STEPS = 4096, 300, 24
ACTION_SCALE = 0.25
FINAL_SOURCE, FINAL_CONTINUATION = 299, 598
GPU_PROFILE = V4.FIXED["gpu_admission_profile"]
TAG = "novel_actor_reset_20261016"
CODE = {
    "evals/novel_actor_reset_confirmation.py", "bench/train_novel_actor_reset.py",
    "bench/train_with_fault.py", "bench/hidden_faults.py", "bench/train_with_recovery_mode.py",
    "bench/train_with_correction.py", "bench/recovery_reset_interventions.py",
    "bench/correction_adapter.py", "evals/preregistered_seed_guard.py", "evals/fixed_eval.py",
    "evals/run_recovery_reset_v2.py", "evals/run_training_correction.py",
    "evals/run_training_correction_v4.py", "bench/record_evidence_media.py",
    "bench/run_runtime_readback_smoke.py", "bench/runtime_readback_smoke.py",
    "src/rl_triage/grounded_recovery.py", "src/rl_triage/recovery.py",
}
BOUND = tuple(sorted(CODE | {PROTOCOL.relative_to(ROOT).as_posix(), FIXED_EVAL.relative_to(ROOT).as_posix()}))
read, write_new, sha = V3.read, V3.write_new, V3.sha
validate_readback = V3.validate_readback


FIXED = {
    "schema": "novel_actor_reset_confirmation_protocol_v1",
    "measurement_kind": "post_mixed_result_novel_seed_development_confirmation",
    "design_disclosure": "기존 actor_last_layer_reset의 seed 2026 recovered·2028 not_recovered 혼합 결과를 본 뒤 만든 새 seed 확인이다",
    "question": "처음 쓰는 F1 seed 두 개에서 actor 말단 재설정만 continuation보다 일관되게 회복하는가",
    "seeds": list(SEEDS), "task": TASK, "fault": FAULT, "correction": CORRECTION,
    "num_envs": NUM_ENVS, "max_iterations": MAX_ITERATIONS, "num_steps_per_env": NUM_STEPS,
    "source_final_iteration": FINAL_SOURCE, "continuation_final_iteration": FINAL_CONTINUATION,
    "action_scale": ACTION_SCALE, "arms": list(ARMS),
    "paired_source": "seed마다 같은 F1 model_299 checkpoint SHA에서 continuation과 actor_last_layer_reset을 시작한다",
    "preconditions": ["동일 seed NONE source가 healthy이고 walks=true", "동일 seed F1 source가 NONE 대비 unhealthy"],
    "intervention_contract": "actor 마지막 Linear만 재초기화하고 그 파라미터 Adam 상태만 제거한다. critic과 policy std는 유지한다",
    "prediction": {"continuation": ["not_recovered", "not_recovered"],
                   "actor_last_layer_reset": ["recovered", "recovered"]},
    "decision_rule": {
        "conditional_candidate": "actor_last_layer_reset 2/2 recovered이고 continuation 0/2 recovered일 때만 조건부 후보로 등록한다",
        "otherwise": "그 밖 결과에서는 현재 회복 규칙을 유지하고 메커니즘 미확정으로 둔다",
        "automatic_global_change": False,
    },
    "source_selection_rule": "fresh guard를 통과한 2063·2064의 등록 이름과 checkpoint model_299를 결과와 무관하게 사용한다",
    "result_wording": {
        "candidate": "새 seed 두 개의 개발 확인에서 actor 말단 재설정만 2/2 회복하고 continuation은 0/2 회복했다. 등록 조건부 후보이며 전역 정책 변경이나 일반화 근거가 아니다.",
        "keep": "새 seed 개발 확인이 등록 조건을 충족하지 않아 현재 회복 규칙을 유지한다. 메커니즘은 미확정이다.",
        "invalid": "source·개입·평가·출처 계약이 불완전해 판정하지 않는다.",
    },
    "semantic_retries": 0, "model_calls": 0, "gpu_admission_profile": GPU_PROFILE,
    "train_watchdog_s": 2700, "eval_watchdog_s": 600,
    "isaaclab_commit": V4.FIXED["isaaclab_commit"], "rsl_rl_version": V4.FIXED["rsl_rl_version"],
    "limits": ["새 seed 두 개의 개발 확인이다", "인과 메커니즘·일반화·GPU 절감·LLM 우월을 주장하지 않는다"],
}
DYNAMIC = {"source_files", "sdk_sources", "rsl_sources", "correction_source_contract",
           "fixed_eval_sha256_lf", "bands"}


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
                "bands": base["resume_inputs"]["bands"]}
    ANCHOR._reject_public_absolute_paths(document)
    write_new(PROTOCOL, document)
    return {"status": "prepared_commit_before_execution", "path": PROTOCOL.relative_to(ROOT).as_posix(),
            "sha256_lf": sha(PROTOCOL, True)}


def verify_protocol(isaaclab: Path) -> tuple[dict[str, Any], str]:
    protocol = read(PROTOCOL)
    if set(protocol) != set(FIXED) | DYNAMIC or any(protocol.get(k) != v for k, v in FIXED.items()):
        raise ValueError("고정 actor reset protocol이 바뀌었다")
    V3.check_contract(protocol["correction_source_contract"], isaaclab)
    if any(protocol.get(k) != v for k, v in source_maps(isaaclab).items()):
        raise ValueError("결합 source hash가 바뀌었다")
    if protocol.get("fixed_eval_sha256_lf") != sha(FIXED_EVAL, True):
        raise ValueError("고정 평가 protocol이 바뀌었다")
    if V3.S._git_head(isaaclab) != FIXED["isaaclab_commit"] or not V3.S._git_tracked_clean(isaaclab):
        raise ValueError("Isaac Lab checkout이 다르다")
    return protocol, ANCHOR.verify_bound_at_head(ROOT, BOUND)


def folder_for(tag: str = TAG) -> Path:
    if tag != TAG:
        raise ValueError("등록 tag가 아니다")
    folder = RESULTS / tag
    if folder.is_symlink() or folder.resolve().parent != RESULTS.resolve():
        raise ValueError("결과 경로가 private root 밖이다")
    return folder


def _find_run(isaaclab: Path, run_name: str) -> list[Path]:
    root = isaaclab / "logs/rsl_rl/unitree_go2_flat"
    return [] if not root.is_dir() else sorted(p for p in root.iterdir() if p.is_dir() and p.name.endswith("_" + run_name))


def _finished(row: dict[str, Any]) -> bool:
    return all((row.get("job_assigned_before_release") is True, row.get("natural_exit") is True,
                row.get("timed_out") is False, row.get("error_type") is None,
                row.get("exit_code") == 0, row.get("cleanup_verified") is True,
                isinstance(row.get("owned_process_identities"), dict)))


def _gpu(value: Any, idle: bool) -> dict[str, Any]:
    if (not isinstance(value, dict) or value.get("admission_policy") != GPU_PROFILE
            or type(value.get("busy")) is not bool or not isinstance(value.get("processes"), list)):
        raise ValueError("GPU receipt 형식/profile이 다르다")
    if idle and value["busy"]:
        raise RuntimeError("GPU가 사용 중이다")
    return value


def _source_command(isaaclab: Path, seed: int, fault: str) -> tuple[str, list[str]]:
    run_name = f"novel_actor_source_{fault.lower()}_s{seed}"
    return run_name, [str(isaaclab / "_isaac_sim/python.bat"), "-B", str(ROOT / "bench/train_with_fault.py"),
                      "--task", TASK, "--num_envs", str(NUM_ENVS), "--max_iterations", str(MAX_ITERATIONS),
                      "--seed", str(seed), "--headless", "--run_name", run_name]


def _fixed_eval_command(isaaclab: Path, stage: Path) -> list[str]:
    return [str(isaaclab / "_isaac_sim/python.bat"), "-B", str(ROOT / "evals/fixed_eval.py"),
            "--protocol", str(FIXED_EVAL), "--jobs", str(stage / "jobs.json"), "--headless"]


def _fixed_eval(isaaclab: Path, stage: Path, name: str, checkpoint: Path) -> tuple[dict, dict]:
    stage.mkdir(parents=True, exist_ok=False)
    write_new(stage / "gpu_before.json", _gpu(GPU.query_gpu(GPU_PROFILE), True))
    output = stage / "report.json"
    write_new(stage / "jobs.json", {"action_scale": ACTION_SCALE, "jobs": [{
        "name": name, "checkpoint": str(checkpoint), "output": str(output)}]})
    argv = _fixed_eval_command(isaaclab, stage)
    execution = OWNED.run_owned(stage, argv, isaaclab, FIXED["eval_watchdog_s"])
    write_new(stage / "execution.json", execution)
    if not _finished(execution) or not output.is_file():
        raise RuntimeError("고정 평가가 완결되지 않았다")
    report = read(output)
    C.validate_fixed_report(report, {"name": name, "checkpoint_file": checkpoint.name,
                                     "checkpoint_sha256": sha(checkpoint), "action_scale": ACTION_SCALE,
                                     "protocol_sha256_lf": sha(FIXED_EVAL, True)})
    return report, execution


def _source_train(isaaclab: Path, stage: Path, seed: int, fault: str) -> dict[str, Any]:
    run_name, argv = _source_command(isaaclab, seed, fault)
    if _find_run(isaaclab, run_name):
        raise FileExistsError("등록 source SDK run이 이미 있다")
    stage.mkdir(parents=True, exist_ok=False)
    write_new(stage / "gpu_before.json", _gpu(GPU.query_gpu(GPU_PROFILE), True))
    old = os.environ.get("TRIAGE_FAULT")
    os.environ["TRIAGE_FAULT"] = fault
    try:
        execution = OWNED.run_owned(stage, argv, isaaclab, FIXED["train_watchdog_s"])
    finally:
        if old is None: os.environ.pop("TRIAGE_FAULT", None)
        else: os.environ["TRIAGE_FAULT"] = old
    write_new(stage / "execution.json", execution)
    if not _finished(execution):
        raise RuntimeError("source 학습이 완결되지 않았다")
    lines = [json.loads(line[len("[hidden_fault] "):]) for line in
             (stage / "stdout.log").read_text(encoding="utf-8", errors="replace").splitlines()
             if line.startswith("[hidden_fault] ")]
    if lines != [{"fault": fault}]:
        raise ValueError("source hidden fault readback이 다르다")
    runs = _find_run(isaaclab, run_name)
    if len(runs) != 1 or not (runs[0] / "model_299.pt").is_file():
        raise ValueError("source checkpoint가 유일하지 않다")
    checkpoint = runs[0] / "model_299.pt"
    environment, agent = runs[0] / "params/env.yaml", runs[0] / "params/agent.yaml"
    env_cfg, agent_cfg = V3.load_config(environment), V3.load_config(agent)
    if (env_cfg.get("seed") != seed or env_cfg.get("scene", {}).get("num_envs") != NUM_ENVS
            or agent_cfg.get("seed") != seed or agent_cfg.get("max_iterations") != MAX_ITERATIONS
            or agent_cfg.get("num_steps_per_env") != NUM_STEPS):
        raise ValueError("source params seed/env/update 설정이 다르다")
    return {"run_name": run_name, "run_dir": str(runs[0]), "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha(checkpoint),
            "environment": {"path": str(environment), "sha256": sha(environment)},
            "agent": {"path": str(agent), "sha256": sha(agent)},
            "execution_sha256": sha(stage / "execution.json")}


def validate_source_evidence(isaaclab: Path, stage: Path, seed: int, fault: str,
                             row: dict[str, Any]) -> dict[str, Any]:
    run_name, command = _source_command(isaaclab, seed, fault)
    execution_path = stage / "execution.json"
    execution = read(execution_path)
    if not _finished(execution) or row.get("execution_sha256") != sha(execution_path):
        raise ValueError("source execution receipt/hash가 다르다")
    if read(stage / "launch.json") != {"args": command, "cwd": str(isaaclab)}:
        raise ValueError("source train launch argv/cwd가 다르다")
    _gpu(read(stage / "gpu_before.json"), True)
    _source_readback(stage, fault)
    run_dir = Path(row.get("run_dir", ""))
    log_root = (isaaclab / "logs/rsl_rl/unitree_go2_flat").resolve()
    checkpoint = Path(row.get("checkpoint", ""))
    if (row.get("run_name") != run_name or run_dir.resolve().parent != log_root
            or not run_dir.name.endswith("_" + run_name) or checkpoint.resolve() != (run_dir / "model_299.pt").resolve()
            or not checkpoint.is_file() or sha(checkpoint) != row.get("checkpoint_sha256")):
        raise ValueError("source run/checkpoint binding이 다르다")
    environment, agent = run_dir / "params/env.yaml", run_dir / "params/agent.yaml"
    for key, path in (("environment", environment), ("agent", agent)):
        if row.get(key) != {"path": str(path), "sha256": sha(path)}:
            raise ValueError("source config artifact binding이 다르다")
    env_cfg, agent_cfg = V3.load_config(environment), V3.load_config(agent)
    if (env_cfg.get("seed") != seed or env_cfg.get("scene", {}).get("num_envs") != NUM_ENVS
            or agent_cfg.get("seed") != seed or agent_cfg.get("max_iterations") != MAX_ITERATIONS
            or agent_cfg.get("num_steps_per_env") != NUM_STEPS):
        raise ValueError("source config seed/env/update semantics가 다르다")
    return row


def validate_fixed_eval_launch(isaaclab: Path, stage: Path, name: str, checkpoint: Path) -> dict[str, Any]:
    _gpu(read(stage / "gpu_before.json"), True)
    expected_jobs = {"action_scale": ACTION_SCALE, "jobs": [{
        "name": name, "checkpoint": str(checkpoint), "output": str(stage / "report.json")} ]}
    if read(stage / "jobs.json") != expected_jobs:
        raise ValueError("fixed evaluation jobs binding이 다르다")
    command = _fixed_eval_command(isaaclab, stage)
    if read(stage / "launch.json") != {"args": command, "cwd": str(isaaclab)}:
        raise ValueError("fixed evaluation launch argv/cwd가 다르다")
    execution = read(stage / "execution.json")
    if not _finished(execution):
        raise ValueError("fixed evaluation owned execution이 실패했다")
    return execution


def manifest_for(folder: Path, request: dict, seed: int, arm: str) -> dict[str, Any]:
    if seed not in SEEDS or arm not in ARMS:
        raise ValueError("등록 seed/arm이 아니다")
    source = read(folder / "source" / f"s{seed}" / "source_evidence.json")[FAULT]
    selection = read(folder / "source_selection.json")
    expected_selection = {"source_evidence_sha256": sha(folder / "source" / f"s{seed}" / "source_evidence.json"),
                          "checkpoint_sha256": source["checkpoint_sha256"]}
    if selection.get("seeds", {}).get(str(seed)) != expected_selection:
        raise ValueError("등록 source selection binding이 다르다")
    checkpoint = Path(source["checkpoint"])
    return {
        "schema": "training_correction_manifest_v2", "tag": folder.name, "seed": seed, "arm": arm,
        "recovery_mode": "resume", "source_commit_sha": request["source_commit_sha"],
        "protocol_sha256_lf": request["protocol_sha256_lf"], "isaaclab_root": request["isaaclab_root"],
        "protocol_path": str(PROTOCOL.resolve()), "fault": FAULT, "correction_id": CORRECTION,
        "num_envs": NUM_ENVS, "max_iterations": MAX_ITERATIONS, "task": TASK,
        "run_name": f"{folder.name}_s{seed}_{arm}",
        "output_dir": str((folder / "continuation" / f"s{seed}" / arm / "train").resolve()),
        "final_iteration": FINAL_CONTINUATION, "resume_checkpoint_path": str(checkpoint.resolve()),
        "resume_checkpoint_sha256": source["checkpoint_sha256"], "resume_iteration": FINAL_SOURCE,
        "load_run_regex": "^" + re.escape(checkpoint.parent.name) + "$", "checkpoint_regex": "^model_299\\.pt$",
        "source_env_config_path": source["environment"]["path"],
        "source_agent_config_path": source["agent"]["path"], "config_source_fault": FAULT,
        "init_noise_std": None,
    }


def validate_manifest(path: Path) -> tuple[dict, dict]:
    manifest = read(path)
    if set(manifest) != R.MANIFEST_FIELDS or R.validate_manifest_mode(manifest) != "resume":
        raise ValueError("continuation manifest schema/mode가 다르다")
    folder = folder_for(manifest.get("tag"))
    request = read(folder / "request.json")
    anchor = ANCHOR.verify_source_anchor(request.get("source_anchor"), ROOT, BOUND)
    protocol, _ = verify_protocol(Path(request["isaaclab_root"]))
    if manifest != manifest_for(folder, request, manifest["seed"], manifest["arm"]):
        raise ValueError("continuation manifest가 등록 입력과 다르다")
    if manifest["source_commit_sha"] != anchor["commit"] or manifest["protocol_sha256_lf"] != sha(PROTOCOL, True):
        raise ValueError("continuation source anchor가 다르다")
    return manifest, protocol


def validate_training_configuration(log_dir: Path, manifest: dict) -> None:
    source_env, source_agent = Path(manifest["source_env_config_path"]), Path(manifest["source_agent_config_path"])
    original_env, original_agent = V3.load_config(source_env), V3.load_config(source_agent)
    current_env, current_agent = V3.load_config(log_dir / "params/env.yaml"), V3.load_config(log_dir / "params/agent.yaml")
    if current_env != original_env:
        raise ValueError("continuation environment config가 source와 다르다")
    allowed = {"run_name", "resume", "load_run", "load_checkpoint", "max_iterations"}
    if ({k:v for k,v in original_agent.items() if k not in allowed}
            != {k:v for k,v in current_agent.items() if k not in allowed}):
        raise ValueError("continuation agent config가 허용 범위 밖에서 바뀌었다")
    expected = {"run_name": manifest["run_name"], "resume": True, "load_run": manifest["load_run_regex"],
                "load_checkpoint": manifest["checkpoint_regex"], "max_iterations": MAX_ITERATIONS}
    if any(current_agent.get(k) != v for k,v in expected.items()):
        raise ValueError("continuation resume arguments가 다르다")


def validate_intervention(row: dict, manifest: dict, source_state: dict) -> None:
    required_extra = {"arm", "critic_state_before", "critic_state_after", "std_state_before", "std_state_after",
                      "actor_linear_states_before", "actor_linear_states_after"}
    if not isinstance(row, dict) or row.get("schema") != "novel_actor_reset_intervention_v1" or not required_extra <= set(row):
        raise ValueError("actor reset readback schema가 다르다")
    arm = manifest["arm"]
    actor_before, actor_after = row["actor_linear_states_before"], row["actor_linear_states_after"]
    if (row["arm"] != arm or row["policy_state_before"] != source_state["policy_state_sha256"]
            or row["optimizer_state_before"] != source_state["optimizer_state_sha256"]
            or row["critic_state_before"] != row["critic_state_after"]
            or row["std_state_before"] != row["std_state_after"]
            or not isinstance(actor_before, dict) or set(actor_before) != set(actor_after) or not actor_before):
        raise ValueError("critic/std/source state 보존 계약이 깨졌다")
    if arm == "continuation":
        if (row["intervention"] != "none" or row["target_modules"] or row["target_parameter_count"] != 0
                or row["policy_state_after"] != row["policy_state_before"]
                or row["optimizer_state_after"] != row["optimizer_state_before"] or actor_before != actor_after
                or row["optimizer_param_indices"] or row["optimizer_moment_indices_present_before"]
                or row["optimizer_moments_present_after"] is not None
                or row["optimizer_param_groups_before"] != row["optimizer_param_groups_after"]
                or row["parameter_identity_preserved"] is not True):
            raise ValueError("continuation no-reset 계약이 깨졌다")
    else:
        if (row["intervention"] != arm or row["reset_seed"] != reset_seed(manifest["seed"])
                or row["target_modules"] != [list(actor_before)[-1]]
                or row["target_parameter_count"] != 2
                or row["optimizer_param_indices"] != sorted(set(row["optimizer_param_indices"]))
                or len(row["optimizer_param_indices"]) != row["target_parameter_count"]
                or row["optimizer_moment_indices_present_before"] != row["optimizer_param_indices"]
                or row["optimizer_moments_present_after"] is not False
                or row["policy_state_after"] == row["policy_state_before"]
                or row["optimizer_state_after"] != row["optimizer_state_expected_after"]
                or row["optimizer_param_groups_before"] != row["optimizer_param_groups_after"]
                or row["parameter_identity_preserved"] is not True
                or [name for name in actor_before if actor_before[name] != actor_after[name]] != row["target_modules"]):
            raise ValueError("actor 마지막 층/Adam 제거 계약이 깨졌다")


def _verify_continuation(folder: Path, seed: int, arm: str, execution: dict) -> dict:
    if not _finished(execution):
        raise ValueError("continuation owned execution이 실패했다")
    training = folder / "continuation" / f"s{seed}" / arm / "train"
    isaaclab = Path(read(folder / "request.json")["isaaclab_root"])
    expected_command = [str(isaaclab / "_isaac_sim/python.bat"), "-B",
                        str(ROOT / "bench/train_novel_actor_reset.py"),
                        "--manifest", str(training / "manifest.json")]
    if read(training / "launch.json") != {"args": expected_command, "cwd": str(isaaclab)}:
        raise ValueError("continuation launch argv/cwd가 다르다")
    manifest, protocol = validate_manifest(training / "manifest.json")
    receipt, initial = read(training / "receipt.json"), read(training / "initial.json")
    intervention = read(training / "intervention.json")
    if (receipt.get("completed") is not True or receipt.get("checkpoint_load_calls") != 1
            or receipt.get("resume_iteration") != FINAL_SOURCE or receipt.get("final_iteration") != FINAL_CONTINUATION
            or receipt.get("actual_control_steps") != MAX_ITERATIONS * NUM_STEPS
            or receipt.get("aggregate_env_steps") != MAX_ITERATIONS * NUM_STEPS * NUM_ENVS
            or receipt.get("source_checkpoint_state") != initial.get("source_checkpoint_state")):
        raise ValueError("continuation budget/checkpoint receipt가 다르다")
    identity = receipt.get("process_identity", {})
    if execution["owned_process_identities"].get(str(identity.get("process_id"))) != identity.get("process_creation_time_filetime"):
        raise ValueError("continuation receipt가 owned process에서 나오지 않았다")
    for name, field in (("manifest.json", "manifest_sha256"), ("initial.json", "initial_sha256"),
                        ("correction.json", "correction_sha256")):
        if receipt.get(field) != sha(training / name):
            raise ValueError("continuation raw artifact hash가 다르다")
    expected_resume = {"path": manifest["resume_checkpoint_path"], "sha256": manifest["resume_checkpoint_sha256"]}
    if receipt.get("resume_checkpoint") != expected_resume or initial.get("resume_checkpoint") != {**expected_resume, "load_optimizer": True}:
        raise ValueError("continuation paired source checkpoint가 다르다")
    validate_intervention(intervention, manifest, receipt["source_checkpoint_state"])
    correction = read(training / "correction.json")
    validate_readback(correction, FAULT, CORRECTION)
    for key, target in protocol["correction_source_contract"]["targets"].items():
        if correction["before_fault"][key] != target["clean_fingerprint"]:
            raise ValueError("continuation clean correction target가 source contract와 다르다")
    checkpoint = Path(receipt["checkpoint"]["path"])
    if checkpoint.name != "model_598.pt" or sha(checkpoint) != receipt["checkpoint"]["sha256"]:
        raise ValueError("continuation final checkpoint가 다르다")
    validate_training_configuration(checkpoint.parent, manifest)
    runtime = protocol["correction_source_contract"]["runtime"]
    versions = receipt["runtime_versions"]
    if (versions.get("rsl_rl_distribution") != protocol["rsl_rl_version"]
            or versions.get("isaaclab_source_commit") != protocol["isaaclab_commit"]
            or versions.get("python") != ".".join(map(str, runtime["version"]))
            or versions.get("python_cache_tag") != runtime["cache_tag"]):
        raise ValueError("continuation runtime version이 다르다")
    expected_observed = {"environment_seed": manifest["seed"], "environment_num_envs": NUM_ENVS,
                         "action_scale": ACTION_SCALE, "agent_seed": manifest["seed"],
                         "agent_max_iterations": MAX_ITERATIONS, "runner_num_steps_per_env": NUM_STEPS}
    if receipt.get("observed") != expected_observed or initial.get("observed") != expected_observed:
        raise ValueError("continuation runtime seed/env/update가 다르다")
    for key, filename in (("environment", "env.yaml"), ("agent", "agent.yaml")):
        artifact = receipt.get("configs", {}).get(key, {})
        if Path(artifact.get("path", "")).resolve() != (checkpoint.parent / "params" / filename).resolve() or sha(Path(artifact["path"])) != artifact.get("sha256"):
            raise ValueError("continuation config artifact가 다르다")
    log_root = (Path(manifest["isaaclab_root"]) / "logs/rsl_rl/unitree_go2_flat").resolve()
    if checkpoint.parent.parent.resolve() != log_root or not checkpoint.parent.name.endswith("_" + manifest["run_name"]):
        raise ValueError("continuation checkpoint가 등록 SDK run 밖이다")
    return receipt


def _source_readback(stage: Path, fault: str) -> None:
    lines = [json.loads(line[len("[hidden_fault] "):]) for line in
             (stage / "stdout.log").read_text(encoding="utf-8", errors="replace").splitlines()
             if line.startswith("[hidden_fault] ")]
    if lines != [{"fault": fault}]:
        raise ValueError("source hidden fault readback이 다르다")


def consumption_path(tag: str) -> Path:
    key = hashlib.sha256((sha(PROTOCOL, True) + tag).encode()).hexdigest()
    return PRIVATE / "consumed" / f"{key}.json"


def validate_freshness(row: Any) -> dict[str, Any]:
    expected_keys = {"schema", "status", "seeds", "checked_at_utc", "checked_roots",
                     "checked_file_count", "evidence", "scope"}
    expected_roots = ["<repo>/bench/**/*.telemetry.json", "<repo>/bench/private/execution_records",
                      "<repo>/evals/results", "<repo>/evals/loops", "<isaaclab>/logs"]
    if (not isinstance(row, dict) or set(row) != expected_keys
            or row.get("schema") != "preregistered_seed_freshness_v1"
            or row.get("status") != "fresh" or row.get("seeds") != list(SEEDS)
            or row.get("checked_roots") != expected_roots
            or type(row.get("checked_file_count")) is not int or row["checked_file_count"] < 0
            or row.get("evidence") != []
            or row.get("scope") != "현재 저장소의 실행 기록과 지정 Isaac Lab 로그에 한정한다"):
        raise ValueError("result fresh-seed receipt가 다르다")
    try:
        checked_at = datetime.fromisoformat(row["checked_at_utc"])
    except (TypeError, ValueError) as error:
        raise ValueError("result fresh-seed receipt 시간이 다르다") from error
    if checked_at.tzinfo is None or checked_at.utcoffset() != timezone.utc.utcoffset(checked_at):
        raise ValueError("result fresh-seed receipt 시간이 UTC가 아니다")
    return row


def validate_run_request(folder: Path, isaaclab: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    request = read(folder / "request.json")
    anchor = ANCHOR.verify_source_anchor(request.get("source_anchor"), ROOT, BOUND)
    expected = {"tag": folder.name, "source_commit_sha": anchor["commit"], "source_anchor": anchor,
                "protocol_sha256_lf": sha(PROTOCOL, True), "isaaclab_root": str(isaaclab.resolve())}
    if request != expected:
        raise ValueError("result request/source anchor가 다르다")
    consumed = read(consumption_path(folder.name))
    if consumed != {"tag": folder.name, "request_sha256": sha(folder / "request.json")}:
        raise ValueError("result consumption receipt가 다르다")
    return request, anchor


def verify_result(isaaclab: Path, folder: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    request, anchor = validate_run_request(folder, isaaclab)
    validate_freshness(read(folder / "freshness.json"))
    _gpu(read(folder / "gpu_before.json"), True)
    source_reports = {}
    for seed in SEEDS:
        seed_dir = folder / "source" / f"s{seed}"
        evidence = read(seed_dir / "source_evidence.json")
        if set(evidence) != {"NONE", FAULT}:
            raise ValueError("source evidence arm 집합이 다르다")
        source_reports[str(seed)] = {}
        for fault in ("NONE", FAULT):
            stage = seed_dir / fault.lower()
            row = validate_source_evidence(Path(request["isaaclab_root"]), stage / "train", seed, fault,
                                           evidence[fault])
            checkpoint = Path(row["checkpoint"])
            validate_fixed_eval_launch(Path(request["isaaclab_root"]), stage / "eval",
                                       f"source_{fault.lower()}_s{seed}", checkpoint)
            report = read(stage / "eval/report.json")
            C.validate_fixed_report(report, {"name": f"source_{fault.lower()}_s{seed}",
                                             "checkpoint_file": "model_299.pt",
                                             "checkpoint_sha256": row["checkpoint_sha256"],
                                             "action_scale": ACTION_SCALE,
                                             "protocol_sha256_lf": sha(FIXED_EVAL, True)})
            source_reports[str(seed)][fault] = report
        verdict = RC.verdict(RC.compare(C._behavior(source_reports[str(seed)][FAULT]),
                                        C._behavior(source_reports[str(seed)]["NONE"])), protocol["bands"])
        none_verdict = RC.verdict(RC.compare(C._behavior(source_reports[str(seed)]["NONE"]),
                                             C._behavior(source_reports[str(seed)]["NONE"])), protocol["bands"])
        expected_precondition = {"none_walks": True, "none_label": "healthy",
                                 "fault_label": "unhealthy", "fault_verdict": verdict,
                                 "none_verdict": none_verdict}
        if (not V4.walks(source_reports[str(seed)]["NONE"]) or none_verdict["label"] != "healthy"
                or verdict["label"] != "unhealthy" or read(seed_dir / "precondition.json") != expected_precondition):
            raise ValueError("source precondition replay가 다르다")
    selection = read(folder / "source_selection.json")
    expected_selection = {"schema": "novel_actor_reset_source_selection_v1",
                          "rule": protocol["source_selection_rule"],
                          "seeds": {str(seed): {
                              "source_evidence_sha256": sha(folder / "source" / f"s{seed}/source_evidence.json"),
                              "checkpoint_sha256": read(folder / "source" / f"s{seed}/source_evidence.json")[FAULT]["checkpoint_sha256"],
                          } for seed in SEEDS}}
    if selection != expected_selection:
        raise ValueError("source selection attestation이 다르다")
    labels, verdicts = {arm: {} for arm in ARMS}, {arm: {} for arm in ARMS}
    for seed in SEEDS:
        source_sha = read(folder / "source" / f"s{seed}/source_evidence.json")[FAULT]["checkpoint_sha256"]
        for arm in ARMS:
            arm_dir = folder / "continuation" / f"s{seed}" / arm
            execution = read(arm_dir / "train/execution.json")
            _gpu(read(arm_dir / "gpu_before_training.json"), True)
            receipt = _verify_continuation(folder, seed, arm, execution)
            if receipt["resume_checkpoint"]["sha256"] != source_sha:
                raise ValueError("paired continuation source SHA가 다르다")
            validate_fixed_eval_launch(Path(request["isaaclab_root"]), arm_dir / "eval",
                                       f"{arm}_s{seed}", Path(receipt["checkpoint"]["path"]))
            report = read(arm_dir / "eval/report.json")
            C.validate_fixed_report(report, {"name": f"{arm}_s{seed}", "checkpoint_file": "model_598.pt",
                                             "checkpoint_sha256": receipt["checkpoint"]["sha256"],
                                             "action_scale": ACTION_SCALE,
                                             "protocol_sha256_lf": sha(FIXED_EVAL, True)})
            labels[arm][str(seed)], verdicts[arm][str(seed)] = _label(
                report, source_reports[str(seed)]["NONE"], protocol["bands"])
    result = read(folder / "result.json")
    validate_result_document(result, result_document(folder, protocol, anchor["commit"], anchor, labels, verdicts))
    after = _gpu(read(folder / "gpu_after.json"), False)
    owned = {pid for path in folder.rglob("execution.json") for pid in read(path).get("owned_process_identities", {})}
    if any(str(row.get("pid")) in owned for row in after["processes"]):
        raise ValueError("result gpu_after에 owned PID가 남았다")
    return result


def _label(report: dict, reference: dict, bands: dict) -> tuple[str, dict]:
    verdict = RC.verdict(RC.compare(C._behavior(report), C._behavior(reference)), bands)
    if verdict["label"] == "undetermined":
        raise ValueError("고정 행동 판정이 healthy/unhealthy로 완결되지 않았다")
    if verdict["label"] == "healthy" and V4.walks(report):
        return "recovered", verdict
    if verdict["label"] in {"healthy", "unhealthy"}:
        return "not_recovered", verdict
    raise ValueError("고정 행동 판정이 healthy/unhealthy로 완결되지 않았다")


def evidence_inventory(folder: Path) -> dict[str, str]:
    excluded = {"result.json", "failure.json"}
    return {
        path.relative_to(folder).as_posix(): sha(path)
        for path in sorted(folder.rglob("*"))
        if path.is_file() and path.name not in excluded
    }


def result_document(folder: Path, protocol: dict[str, Any], commit: str, anchor: dict[str, Any],
                    labels: dict[str, dict[str, str]], verdicts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    candidate = (all(labels["actor_last_layer_reset"][str(seed)] == "recovered" for seed in SEEDS)
                 and all(labels["continuation"][str(seed)] == "not_recovered" for seed in SEEDS))
    prediction = {arm: {str(seed): protocol["prediction"][arm][index]
                        for index, seed in enumerate(SEEDS)} for arm in ARMS}
    return {"schema": "novel_actor_reset_confirmation_result_v1", "status": "complete",
            "decision": "conditional_candidate" if candidate else "keep_current_rule",
            "labels": labels, "verdicts": verdicts, "prediction": protocol["prediction"],
            "prediction_match": labels == prediction,
            "wording": protocol["result_wording"]["candidate" if candidate else "keep"],
            "source_commit_sha": commit, "source_anchor": anchor,
            "protocol_sha256_lf": sha(PROTOCOL, True), "model_calls": 0,
            "automatic_global_change": False, "limits": protocol["limits"],
            "evidence_sha256": evidence_inventory(folder)}


def validate_result_document(row: Any, expected: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(row, dict) or row != expected:
        raise ValueError("저장 result 전체가 재계산 결과와 다르다")
    return row


def execute(isaaclab: Path, tag: str = TAG) -> dict[str, Any]:
    protocol, commit = verify_protocol(isaaclab)
    folder = folder_for(tag)
    consumption = consumption_path(tag)
    if folder.exists() or consumption.exists():
        raise FileExistsError("Task12 tag/protocol은 한 번만 실행한다")
    freshness = SEED_GUARD.assert_fresh_seeds(SEEDS, isaaclab, repo=ROOT)
    admission = _gpu(GPU.query_gpu(GPU_PROFILE), True)
    anchor = ANCHOR.make_source_anchor(commit, ROOT, BOUND)
    ANCHOR.verify_source_anchor(anchor, ROOT, BOUND)
    request = {"tag": tag, "source_commit_sha": commit, "source_anchor": anchor,
               "protocol_sha256_lf": sha(PROTOCOL, True), "isaaclab_root": str(isaaclab.resolve())}
    write_new(folder / "freshness.json", freshness)
    write_new(folder / "gpu_before.json", admission)
    write_new(folder / "request.json", request)
    write_new(consumption, {"tag": tag, "request_sha256": sha(folder / "request.json")})
    try:
        source_reports, source_evidence = {}, {}
        for seed in SEEDS:
            source_evidence[str(seed)], source_reports[str(seed)] = {}, {}
            seed_dir = folder / "source" / f"s{seed}"
            for fault in ("NONE", FAULT):
                ANCHOR.verify_source_anchor(anchor, ROOT, BOUND)
                evidence = _source_train(isaaclab, seed_dir / fault.lower() / "train", seed, fault)
                report, _ = _fixed_eval(isaaclab, seed_dir / fault.lower() / "eval", f"source_{fault.lower()}_s{seed}", Path(evidence["checkpoint"]))
                source_evidence[str(seed)][fault] = evidence
                source_reports[str(seed)][fault] = report
            write_new(seed_dir / "source_evidence.json", source_evidence[str(seed)])
            none, fault_report = source_reports[str(seed)]["NONE"], source_reports[str(seed)][FAULT]
            source_verdict = RC.verdict(RC.compare(C._behavior(fault_report), C._behavior(none)), protocol["bands"])
            none_verdict = RC.verdict(RC.compare(C._behavior(none), C._behavior(none)), protocol["bands"])
            if not V4.walks(none) or none_verdict["label"] != "healthy" or source_verdict["label"] != "unhealthy":
                raise RuntimeError("source precondition(NONE healthy walk/F1 unhealthy)이 실패했다")
            write_new(seed_dir / "precondition.json", {"none_walks": True, "none_label": "healthy",
                                                         "fault_label": "unhealthy", "fault_verdict": source_verdict,
                                                         "none_verdict": none_verdict})
        write_new(folder / "source_selection.json", {
            "schema": "novel_actor_reset_source_selection_v1", "rule": protocol["source_selection_rule"],
            "seeds": {str(seed): {
                "source_evidence_sha256": sha(folder / "source" / f"s{seed}/source_evidence.json"),
                "checkpoint_sha256": source_evidence[str(seed)][FAULT]["checkpoint_sha256"],
            } for seed in SEEDS},
        })
        labels, verdicts, reports = {arm: {} for arm in ARMS}, {arm: {} for arm in ARMS}, {}
        for seed in SEEDS:
            reports[str(seed)] = {}
            for arm in ARMS:
                ANCHOR.verify_source_anchor(anchor, ROOT, BOUND)
                arm_dir = folder / "continuation" / f"s{seed}" / arm
                training = arm_dir / "train"
                write_new(arm_dir / "gpu_before_training.json", _gpu(GPU.query_gpu(GPU_PROFILE), True))
                write_new(training / "manifest.json", manifest_for(folder, request, seed, arm))
                argv = [str(isaaclab / "_isaac_sim/python.bat"), "-B", str(ROOT / "bench/train_novel_actor_reset.py"),
                        "--manifest", str(training / "manifest.json")]
                execution = OWNED.run_owned(training, argv, isaaclab, FIXED["train_watchdog_s"])
                write_new(training / "execution.json", execution)
                receipt = _verify_continuation(folder, seed, arm, execution)
                report, _ = _fixed_eval(isaaclab, arm_dir / "eval", f"{arm}_s{seed}", Path(receipt["checkpoint"]["path"]))
                reports[str(seed)][arm] = report
                label, verdict = _label(report, source_reports[str(seed)]["NONE"], protocol["bands"])
                labels[arm][str(seed)], verdicts[arm][str(seed)] = label, verdict
        after = _gpu(GPU.query_gpu(GPU_PROFILE), False)
        owned_pids = {pid for path in folder.rglob("execution.json")
                      for pid in read(path).get("owned_process_identities", {})}
        if any(str(row.get("pid")) in owned_pids for row in after["processes"]):
            raise RuntimeError("종료 뒤 owned GPU PID가 남았다")
        write_new(folder / "gpu_after.json", after)
        result = result_document(folder, protocol, commit, anchor, labels, verdicts)
        write_new(folder / "result.json", result)
        return result
    except BaseException as error:
        write_new(folder / "failure.json", {"type": type(error).__name__, "message": str(error)})
        raise


def verify(isaaclab: Path, tag: str | None = None) -> dict[str, Any]:
    protocol, commit = verify_protocol(isaaclab)
    out = {"status": "ready", "source_commit_sha": commit, "protocol_sha256_lf": sha(PROTOCOL, True)}
    if tag:
        folder = folder_for(tag)
        result = verify_result(isaaclab, folder, protocol)
        if result.get("status") != "complete" or result.get("protocol_sha256_lf") != sha(PROTOCOL, True):
            raise ValueError("저장 result가 완결되지 않았다")
        out["result"] = result
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify", "execute"))
    parser.add_argument("--isaaclab", required=True, type=Path)
    parser.add_argument("--tag")
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare": result = prepare(args.isaaclab.resolve())
        elif args.command == "verify": result = verify(args.isaaclab.resolve(), args.tag)
        else: result = execute(args.isaaclab.resolve(), args.tag or TAG)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    except (ValueError, RuntimeError, FileNotFoundError, FileExistsError, subprocess.CalledProcessError) as error:
        print(f"거부: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
