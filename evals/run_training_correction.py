"""등록한 세 코드 복원 중 하나를 이어 학습과 동결 행동 평가로 확인한다.

호스트가 고른 합성 결함의 개발 실험이다. 모델 선택, 진단 정확도, 실사용 실패,
통계적 일반화의 평가가 아니며 원래 probe와 결함 정의는 변경하지 않는다.
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
import yaml

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "bench"):
    sys.path.insert(0, str(directory))

import correction_adapter as A
import check_correction_cpu as CPU
import record_evidence_media as M
import run_runtime_readback_smoke as P
import runtime_readback_smoke as S
from rl_triage import grounded_recovery as C
from rl_triage import recovery as RC
from rl_triage.triage_tools import _TolerantLoader

PROTOCOL = ROOT / "bench/protocols/training_correction_v1.json"
FIXED_EVAL = ROOT / "bench/protocols/fixed_eval_v1.json"
PRIVATE = ROOT / "bench/private/training_correction"
ORIGINAL = ROOT / "evals/results/p0c_fixed_eval_20261009"
INPUT_RUNS = {"NONE": "baseline_p0c_s2027", "F1": "h02_s2027",
              "F2": "h06_s2027", "F6": "h03_s2027"}
TARGET_BY_FAULT = {
    "F1": "restore_body_frame_reward",
    "F2": "restore_motor_torque_speed",
    "F6": "restore_configured_timeout",
}
CASES = {
    "case_01": {"fault": "F1", "sham": "restore_motor_torque_speed"},
    "case_02": {"fault": "F2", "sham": "restore_configured_timeout"},
    "case_03": {"fault": "F6", "sham": "restore_body_frame_reward"},
}
ARMS = ("clean_reference", "fault_sham", "corrected")
CODE = {
    "evals/run_training_correction.py", "bench/train_with_correction.py",
    "bench/correction_adapter.py", "bench/check_correction_cpu.py", "bench/hidden_faults.py",
    "bench/record_evidence_media.py", "bench/run_runtime_readback_smoke.py",
    "bench/runtime_readback_smoke.py", "evals/fixed_eval.py",
    "src/rl_triage/grounded_recovery.py", "src/rl_triage/recovery.py",
    "src/rl_triage/__init__.py", "src/rl_triage/triage_tools.py",
}
SDK_FILES = S.ISAACLAB_SOURCE_FILES | {row["source"] for row in A.TARGETS.values()} | {
    "scripts/reinforcement_learning/rsl_rl/train.py",
    "scripts/reinforcement_learning/rsl_rl/cli_args.py",
    "source/isaaclab/isaaclab/actuators/actuator_cfg.py",
    "source/isaaclab/isaaclab/managers/termination_manager.py",
    "source/isaaclab/isaaclab/utils/dict.py",
}
FIXED = {
    "schema": "training_correction_protocol_v1",
    "scope": "seen_single_seed_resume_feasibility",
    "task": C.TASK, "seed": 2027, "num_envs": 4096, "max_iterations": 300,
    "num_steps_per_env": 24, "action_scale": 0.25,
    "resume_iteration": 299, "final_iteration": 598,
    "input_runs": INPUT_RUNS, "source_cohort_already_seen": True,
    "cases": CASES, "arms": list(ARMS), "initial_match_arms": ["fault_sham", "corrected"],
    "resume_scope": "policy_optimizer_warm_start_not_environment_or_rng_restore",
    "require_sham_corrected_initial_rng_match": True,
    "python_runtime": {"implementation": "CPython", "version": [3, 10, 15], "cache_tag": "cpython-310"},
    "torch_version": "2.7.0+cu128",
    "bands_source": "evals/results/p0c_fixed_eval_20261009/summary.json",
    "primary_reference": "same_seed_clean_reference_resumed_same_updates",
    "secondary_reference": "original_same_seed_clean_checkpoint",
    "require_original_sham_failed_metric_overlap": True,
    "allowed_agent_config_changes": ["run_name", "resume", "load_run", "load_checkpoint", "max_iterations"],
    "train_watchdog_s": 1800, "eval_watchdog_s": 360,
    "semantic_retries": 0, "train_limit_per_case": 3, "eval_limit_per_case": 3,
    "gpu_admission_profile": M.BASE_COM_RESET_GPU_ADMISSION["policy"],
    "isaaclab_commit": S.ISAACLAB_COMMIT, "rsl_rl_version": "2.3.3",
    "approval_actor": "codex_operator_user_authorized_test",
    "model_calls": 0, "model_selected_correction": False,
    "baseline_reuse_between_cases": False,
    "result_wording": {
        "restoration_observed_single_run": "실패한 정책에서 등록 코드 하나를 복원하고 이어 학습한 단일 실행이 고정 행동 기준을 통과했다. 같은 체크포인트·추가 학습량의 잘못된 교정은 원래 실패 지표 중 적어도 하나를 유지했다. 이미 본 seed의 합성 결함 개발 결과이며 AI 진단·일반화·실사용 수리를 입증하지 않는다.",
        "source_failure_absent": "원본 체크포인트에서 등록한 행동 실패가 확인되지 않아 회복 효과로 집계하지 않는다.",
        "baseline_invalid": "같은 예산으로 이어 학습한 정상 기준 정책이 보행 기준을 충족하지 못해 교정 효과를 판정하지 않는다.",
        "no_failure_observed": "결함을 유지한 대조에서 등록한 행동 실패가 관찰되지 않아 회복 효과로 집계하지 않는다.",
        "sham_shifted": "결함 대조가 실패했지만 원래 실패 지표와 겹치지 않아 교정 효과로 집계하지 않는다.",
        "no_recovery": "교정 후 이어 학습한 정책이 정상 기준의 행동 허용 범위를 통과하지 못했다.",
        "invalid": "출처·개입·초기 상태·예산·평가·종료 계약이 맞지 않아 효과 판정을 보류한다.",
    },
}


def read(path: Path) -> dict:
    return S._read_json(path)


def write_new(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def sha(path: Path, lf: bool = False) -> str:
    raw = path.read_bytes()
    return hashlib.sha256(raw.replace(b"\r\n", b"\n") if lf else raw).hexdigest()


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def rsl_root(isaaclab: Path) -> Path:
    return (isaaclab / "_isaac_sim/kit/python/Lib/site-packages/rsl_rl").resolve()


def source_maps(isaaclab: Path) -> dict:
    rsl = rsl_root(isaaclab)
    names = {p.relative_to(rsl).as_posix() for p in rsl.rglob("*.py")}
    if not {"runners/on_policy_runner.py", "algorithms/ppo.py", "modules/actor_critic.py"} <= names:
        raise ValueError("RSL-RL source is missing")
    return {
        "source_files": {name: sha(ROOT / name, True) for name in sorted(CODE)},
        "sdk_sources": {name: sha(isaaclab / name, True) for name in sorted(SDK_FILES)},
        "rsl_sources": {name: sha(rsl / name, True) for name in sorted(names)},
    }


def check_contract(contract: dict, isaaclab: Path) -> None:
    if contract.get("schema") != "correction_source_contract_v1" or set(contract.get("targets", {})) != set(A.TARGETS):
        raise ValueError("correction source contract target set differs")
    if contract.get("runtime") != FIXED["python_runtime"]:
        raise ValueError("contract must be generated by the registered Isaac Sim Python runtime")
    for key, spec in A.TARGETS.items():
        row = contract["targets"][key]
        if any(row.get(k) != v for k, v in spec.items()) or row.get("source_sha256") != sha(isaaclab / spec["source"], True):
            raise ValueError("correction target source differs")
        if not re.fullmatch(r"[0-9a-f]{64}", row.get("clean_fingerprint", "")):
            raise ValueError("invalid clean fingerprint")


def source_paths(isaaclab: Path, fault: str) -> dict[str, Path]:
    name = INPUT_RUNS[fault]
    meta_path = ROOT / "bench/private" / f"{name}.meta.json"
    meta = read(meta_path)
    expected = {**{k: FIXED[k] for k in ("task", "seed", "num_envs", "max_iterations")},
                "fault": fault, "train_exit_code": 0}
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


def source_inputs(isaaclab: Path) -> dict:
    """기존 입력의 SHA와 실패 지표를 묶는다. 새 평가 결과로 세지 않는다."""
    summary = read(ORIGINAL / "summary.json")
    bands = summary["bands"]
    if set(bands) != {"survival_ratio_min", "err_xy_ratio_max", "err_yaw_ratio_max", "fall_frac_max"}:
        raise ValueError("original behavior bands differ")
    reports, inputs = {}, {}
    for fault, name in INPUT_RUNS.items():
        paths = source_paths(isaaclab, fault)
        hashes = {key: sha(path) for key, path in paths.items()}
        report = C.validate_fixed_report(read(paths["fixed_report"]), {
            "name": name, "checkpoint_file": "model_299.pt", "checkpoint_sha256": hashes["checkpoint"],
            "action_scale": FIXED["action_scale"], "protocol_sha256_lf": sha(FIXED_EVAL, True),
        })
        original_rows = [row for row in summary["runs"] if row["run"] == name]
        if len(original_rows) != 1:
            raise ValueError("original summary run is missing or duplicated")
        row = original_rows[0]
        if row["metrics"] != report["metrics"] or row["checkpoint_sha256"] != hashes["checkpoint"]:
            raise ValueError("original summary differs from the raw report")
        inputs[fault] = {"run": name, "hashes": hashes,
                         "original_label": row["fixed_label"], "original_failed": row["fixed_failed"]}
        reports[fault] = report
    reference = C._behavior(reports["NONE"])
    if reports["NONE"]["versions"]["torch"] != FIXED["torch_version"]:
        raise ValueError("original Torch runtime differs")
    for fault, report in reports.items():
        if report["versions"] != reports["NONE"]["versions"]:
            raise ValueError("original report runtimes differ")
        verdict = RC.verdict(RC.compare(C._behavior(report), reference), bands)
        if (verdict["label"] != inputs[fault]["original_label"]
                or verdict["failed"] != inputs[fault]["original_failed"]):
            raise ValueError("original failure classification does not replay")
    return {"summary_sha256_lf": sha(ORIGINAL / "summary.json", True),
            "bands": bands, "runs": inputs, "runtime_versions": reports["NONE"]["versions"]}


def cpu_checks(folder: Path, contract: dict, isaaclab: Path) -> dict:
    folder = folder.resolve()
    if not folder.is_relative_to((ROOT / "bench/private").resolve()):
        raise ValueError("CPU evidence must stay under bench/private")
    hashes = {}
    check_names = {"clean_differs_from_fault", "cuda_remained_uninitialized", "matching_readback_restores_selected",
                   "matching_restore_changes_fault", "matching_restore_returns_clean",
                   "wrong_target_digest_change_count_zero", "wrong_target_retains_fault_behavior"}
    for spec in CASES.values():
        fault = spec["fault"]
        for suffix, correction in (("match", TARGET_BY_FAULT[fault]), ("wrong", spec["sham"])):
            name = f"{fault.lower()}_{suffix}.json"
            row = read(folder / name)
            if (row.get("schema") != "correction_cpu_check_v1" or row.get("fault") != fault
                    or row.get("correction_id") != correction or row.get("passed") is not True
                    or row.get("matching_correction") is not (suffix == "match")
                    or row.get("source_contract") != contract or row.get("runtime_fingerprint") != contract["runtime"]
                    or row.get("verification_type") != "extracted_upstream_function_cpu_check"
                    or row.get("torch", {}).get("cuda_initialized") is not False
                    or row.get("torch", {}).get("version") != FIXED["torch_version"]
                    or row.get("checks") != {key: True for key in check_names}):
                raise ValueError("CPU correction matrix is missing, failed, or source-mismatched")
            binding = CPU._go2_motor_binding(isaaclab) if fault == "F2" else None
            if row.get("inputs") != CPU.INPUTS[fault] or row.get("input_binding") != binding:
                raise ValueError("CPU inputs or GO2 motor source binding differs")
            if row.get("scope") != {"extracted_target_functions_executed": True, "full_sdk_modules_imported": False,
                                   "gpu_used": False, "isaac_app_started": False,
                                   "performance_measurement": False, "physx_started": False}:
                raise ValueError("CPU evidence scope differs")
            validate_readback(row["correction_readback"], fault, correction)
            outputs = row["outputs"]
            expected = outputs["clean"] if suffix == "match" else outputs["fault"]
            if outputs["clean"] == outputs["fault"] or outputs["corrected"] != expected:
                raise ValueError("CPU behavioral restoration/no-op differs")
            for target, source in contract["targets"].items():
                if (row["raw_source_hashes"].get(target) != source["source_sha256"]
                        or row["correction_readback"]["before_fault"].get(target) != source["clean_fingerprint"]):
                    raise ValueError("CPU original source binding differs")
            hashes[name] = sha(folder / name)
    return {"folder": folder.relative_to(ROOT.resolve()).as_posix(), "sha256": hashes, "passed_count": 6}


def prepare(isaaclab: Path, contract_path: Path) -> dict:
    contract = read(contract_path)
    check_contract(contract, isaaclab)
    document = {**deepcopy(FIXED), **source_maps(isaaclab),
                "correction_source_contract": contract,
                "fixed_eval_sha256_lf": sha(FIXED_EVAL, True), "resume_inputs": source_inputs(isaaclab),
                "cpu_checks": cpu_checks(contract_path.parent, contract, isaaclab)}
    write_new(PROTOCOL, document)
    return {"status": "prepared_commit_before_execution", "path": str(PROTOCOL)}


def verify_protocol(isaaclab: Path) -> tuple[dict, str]:
    protocol = read(PROTOCOL)
    if set(protocol) != set(FIXED) | {"source_files", "sdk_sources", "rsl_sources", "correction_source_contract", "fixed_eval_sha256_lf", "resume_inputs", "cpu_checks"}:
        raise ValueError("protocol fields differ")
    if any(type(protocol[k]) is not type(v) or protocol[k] != v for k, v in FIXED.items()):
        raise ValueError("fixed training contract changed")
    check_contract(protocol["correction_source_contract"], isaaclab)
    if any(protocol[k] != v for k, v in source_maps(isaaclab).items()):
        raise ValueError("training source hashes changed")
    if protocol["cpu_checks"] != cpu_checks(ROOT / protocol["cpu_checks"]["folder"], protocol["correction_source_contract"], isaaclab):
        raise ValueError("CPU evidence hashes changed")
    if protocol["fixed_eval_sha256_lf"] != sha(FIXED_EVAL, True):
        raise ValueError("frozen behavior protocol changed")
    if protocol["resume_inputs"] != source_inputs(isaaclab):
        raise ValueError("bound resume checkpoint, config, or original evaluation changed")
    if S._git_head(isaaclab) != FIXED["isaaclab_commit"] or not S._git_tracked_clean(isaaclab):
        raise ValueError("Isaac Lab checkout differs")
    commit = P.verify_committed(ROOT, CODE | {PROTOCOL.relative_to(ROOT).as_posix(), FIXED_EVAL.relative_to(ROOT).as_posix()})
    return protocol, commit


def folder_for(tag: str) -> Path:
    if not isinstance(tag, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,39}", tag):
        raise ValueError("invalid training tag")
    folder = PRIVATE / tag
    if folder.is_symlink() or folder.resolve().parent != PRIVATE.resolve():
        raise ValueError("training output escapes its root")
    return folder


def selection(case: str, arm: str) -> tuple[str, str | None]:
    if case not in CASES or arm not in ARMS:
        raise ValueError("unregistered case or arm")
    spec = CASES[case]
    if arm == "clean_reference":
        return "NONE", None
    return spec["fault"], spec["sham"] if arm == "fault_sham" else TARGET_BY_FAULT[spec["fault"]]


def manifest_for(folder: Path, request: dict, arm: str) -> dict:
    fault, correction = selection(request["case"], arm)
    paths = source_paths(Path(request["isaaclab_root"]), fault)
    checkpoint = paths["checkpoint"].resolve()
    bound = read(PROTOCOL)["resume_inputs"]["runs"][fault]["hashes"]
    return {
        "schema": "training_correction_manifest_v1", "tag": folder.name,
        "case": request["case"], "arm": arm, "source_commit_sha": request["source_commit_sha"],
        "protocol_sha256_lf": request["protocol_sha256_lf"], "isaaclab_root": request["isaaclab_root"],
        "protocol_path": str(PROTOCOL.resolve()), "fault": fault, "correction_id": correction,
        **{key: FIXED[key] for key in ("seed", "num_envs", "max_iterations", "task")},
        "run_name": f"{folder.name}_{arm}", "output_dir": str((folder / arm / "train").resolve()),
        "resume_checkpoint_path": str(checkpoint), "resume_checkpoint_sha256": bound["checkpoint"],
        "resume_iteration": FIXED["resume_iteration"],
        "resume_env_config_path": str(paths["environment"].resolve()),
        "resume_agent_config_path": str(paths["agent"].resolve()),
        "load_run_regex": "^" + re.escape(checkpoint.parent.name) + "$",
        "checkpoint_regex": "^model_299\\.pt$",
    }


def load_config(path: Path) -> dict:
    result = yaml.load(path.read_text(encoding="utf-8"), Loader=_TolerantLoader)
    if not isinstance(result, dict):
        raise ValueError("training config is not a mapping")
    return result


def validate_training_configuration(log_dir: Path, manifest: dict) -> None:
    """원 설정과 학습 시작 직전 설정의 의미 차이를 허용된 재개 인자로 제한한다."""
    protocol = read(Path(manifest["protocol_path"]))
    hashes = protocol["resume_inputs"]["runs"][manifest["fault"]]["hashes"]
    source_env = Path(manifest["resume_env_config_path"])
    source_agent = Path(manifest["resume_agent_config_path"])
    for path, key in ((source_env, "environment"), (source_agent, "agent")):
        if sha(path) != hashes[key]:
            raise ValueError("source configuration changed before learning")
    original_env, original_agent = load_config(source_env), load_config(source_agent)
    current_env = load_config(log_dir / "params/env.yaml")
    current_agent = load_config(log_dir / "params/agent.yaml")
    if current_env != original_env:
        raise ValueError("environment configuration changed outside the registered code correction")
    allowed = set(FIXED["allowed_agent_config_changes"])
    if ({k: v for k, v in original_agent.items() if k not in allowed}
            != {k: v for k, v in current_agent.items() if k not in allowed}):
        raise ValueError("agent configuration changed outside the resume arguments")
    expected = {"run_name": manifest["run_name"], "resume": True,
                "load_run": manifest["load_run_regex"], "load_checkpoint": manifest["checkpoint_regex"],
                "max_iterations": manifest["max_iterations"]}
    if any(type(current_agent.get(k)) is not type(v) or current_agent[k] != v for k, v in expected.items()):
        raise ValueError("agent resume configuration differs from the manifest")
    if (original_agent.get("seed") != FIXED["seed"] or original_agent.get("num_steps_per_env") != FIXED["num_steps_per_env"]
            or original_env.get("seed") != FIXED["seed"] or original_env.get("scene", {}).get("num_envs") != FIXED["num_envs"]):
        raise ValueError("original training seed, environment count, or step count differs")


def validate_manifest(path: Path) -> tuple[dict, dict]:
    manifest = read(path)
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


def validate_readback(row: dict, fault: str, correction: str | None) -> None:
    targets = set(A.TARGETS)
    if row.get("schema") != "correction_readback_v1" or row.get("selected_correction") != correction:
        raise ValueError("correction selection/readback differs")
    for stage in ("before_fault", "after_fault", "after_correction"):
        if set(row.get(stage, {})) != targets or any(not re.fullmatch(r"[0-9a-f]{64}", v) for v in row[stage].values()):
            raise ValueError("incomplete correction fingerprints")
    before, injected, after = (row[k] for k in ("before_fault", "after_fault", "after_correction"))
    if fault == "NONE":
        if correction is not None or before != injected or before != after or row["changed_targets"] or row["restored_targets"]:
            raise ValueError("clean reference was changed")
        return
    target = TARGET_BY_FAULT.get(fault)
    if target is None or correction not in targets:
        raise ValueError("unregistered fault or correction")
    if {key for key in targets if before[key] != injected[key]} != {target}:
        raise ValueError("frozen fault did not change exactly its registered target")
    changed = [target] if correction == target else []
    expected_after = dict(injected)
    if correction == target:
        expected_after[target] = before[target]
    if after != expected_after or row["changed_targets"] != changed or row["restored_targets"] != changed:
        raise ValueError("correction changed an unselected target or removed the sham fault")


def execution_valid(row: dict) -> bool:
    return (row.get("job_assigned_before_release") is True and row.get("natural_exit") is True
            and type(row.get("exit_code")) is int and row["exit_code"] == 0
            and row.get("cleanup_verified") is True and row.get("timed_out") is False
            and row.get("error_type") is None)


def verify_training(folder: Path, arm: str, execution: dict) -> dict:
    training = folder / arm / "train"
    manifest, protocol = validate_manifest(training / "manifest.json")
    receipt = read(training / "receipt.json")
    initial = read(training / "initial.json")
    if (not execution_valid(execution) or receipt.get("completed") is not True
            or receipt.get("schema") != "training_correction_receipt_v1"
            or initial.get("schema") != "training_correction_initial_v1"):
        raise ValueError("training or owned process cleanup failed")
    for key in ("tag", "case", "arm", "fault", "correction_id", "source_commit_sha", "protocol_sha256_lf", "task", "run_name"):
        if receipt.get(key) != manifest[key]:
            raise ValueError(f"training receipt binding differs: {key}")
    for file, field in (("manifest.json", "manifest_sha256"), ("initial.json", "initial_sha256"), ("correction.json", "correction_sha256")):
        if receipt.get(field) != sha(training / file):
            raise ValueError("training raw artifact changed")
    identity = receipt["process_identity"]
    if execution["owned_process_identities"].get(str(identity["process_id"])) != identity["process_creation_time_filetime"]:
        raise ValueError("training receipt came from an unowned process")
    steps = FIXED["max_iterations"] * FIXED["num_steps_per_env"]
    if (receipt.get("actual_control_steps") != steps or receipt.get("environment_make_calls") != 1
            or receipt.get("aggregate_env_steps") != steps * FIXED["num_envs"]):
        raise ValueError("training budget or environment count differs")
    if receipt["budget"] != {**{k: FIXED[k] for k in ("seed", "num_envs", "max_iterations", "num_steps_per_env")}, "expected_control_steps": steps}:
        raise ValueError("training budget receipt differs")
    expected_resume = {"path": manifest["resume_checkpoint_path"], "sha256": manifest["resume_checkpoint_sha256"]}
    if receipt.get("resume_checkpoint") != expected_resume or initial.get("resume_checkpoint") != {**expected_resume, "load_optimizer": True}:
        raise ValueError("training did not resume the bound policy and optimizer")
    if (receipt.get("resume_iteration") != FIXED["resume_iteration"]
            or initial.get("resume_iteration") != FIXED["resume_iteration"]
            or receipt.get("final_iteration") != FIXED["final_iteration"]):
        raise ValueError("resume or final iteration differs")
    source_state = receipt["source_checkpoint_state"]
    if (source_state != initial.get("source_checkpoint_state") or source_state.get("iteration") != FIXED["resume_iteration"]
            or source_state.get("policy_state_sha256") != initial["policy_state"]["sha256"]
            or source_state.get("optimizer_state_sha256") != initial["optimizer_state"]["sha256"]):
        raise ValueError("initial policy/optimizer differs from the source checkpoint")
    for state, iteration in ((source_state, FIXED["resume_iteration"]), (receipt["final_checkpoint_state"], FIXED["final_iteration"])):
        if state.get("iteration") != iteration or any(not re.fullmatch(r"[0-9a-f]{64}", state.get(k, ""))
                for k in ("policy_state_sha256", "optimizer_state_sha256")):
            raise ValueError("checkpoint state proof is incomplete")
    resume_scope = {"policy_optimizer_restored": True, "full_environment_state_restored": False,
                    "rng_state_restored_from_checkpoint": False}
    if (receipt.get("resume_scope") != resume_scope or initial.get("init_at_random_ep_len") is not True
            or initial.get("full_environment_state_restored") is not False
            or initial.get("rng_state_restored_from_checkpoint") is not False):
        raise ValueError("resume scope or episode initialization differs")
    if (receipt.get("rng_state_before_learn") != initial.get("rng_state_before_learn")
            or not isinstance(initial.get("rng_state_before_learn"), dict)):
        raise ValueError("training RNG observation differs")
    observed = {"environment_seed": FIXED["seed"], "environment_num_envs": FIXED["num_envs"],
                "action_scale": FIXED["action_scale"], "agent_seed": FIXED["seed"],
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
    if checkpoint.name != f"model_{FIXED['final_iteration']}.pt":
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


def summarize(reports: dict, initials: dict, original_reports: dict, bands: dict, original_failed: list[str]) -> dict:
    """원시 행동 수치 판정. 체크포인트·실행 출처는 부모 검증이 별도로 결합한다."""
    if set(reports) != set(ARMS) or set(initials) != set(ARMS):
        raise ValueError("all three arms are required")
    for field in ("policy_state", "optimizer_state"):
        if len({initials[arm][field]["sha256"] for arm in FIXED["initial_match_arms"]}) != 1:
            raise ValueError("initial policy/optimizer states differ")
    if initials["fault_sham"]["resume_checkpoint"] != initials["corrected"]["resume_checkpoint"]:
        raise ValueError("sham and correction did not start from the same checkpoint")
    rng = initials["fault_sham"].get("rng_state_before_learn")
    if (not isinstance(rng, dict) or set(rng) != {"cpu_sha256", "cuda_all_sha256", "cuda_device_count"}
            or rng != initials["corrected"].get("rng_state_before_learn")
            or not re.fullmatch(r"[0-9a-f]{64}", rng.get("cpu_sha256", ""))
            or rng.get("cuda_device_count") != len(rng.get("cuda_all_sha256", []))
            or not rng.get("cuda_all_sha256")
            or any(not re.fullmatch(r"[0-9a-f]{64}", value) for value in rng["cuda_all_sha256"])):
        raise ValueError("initial CPU/CUDA RNG states differ or are missing")
    training_versions = [initials[arm].get("runtime_versions") for arm in ARMS]
    if not isinstance(training_versions[0], dict) or any(value != training_versions[0] for value in training_versions):
        raise ValueError("training runtimes differ or are missing")
    for arm, report in reports.items():
        C.validate_fixed_report(report, {"name": arm, "checkpoint_file": f"model_{FIXED['final_iteration']}.pt",
            "checkpoint_sha256": report["checkpoint"]["sha256"], "action_scale": FIXED["action_scale"],
            "protocol_sha256_lf": sha(FIXED_EVAL, True)})
    if set(original_reports) != {"clean_reference", "fault"}:
        raise ValueError("original reference and failed report are required")
    for report in original_reports.values():
        C.validate_fixed_report(report, {"name": report["name"], "checkpoint_file": "model_299.pt",
            "checkpoint_sha256": report["checkpoint"]["sha256"], "action_scale": FIXED["action_scale"],
            "protocol_sha256_lf": sha(FIXED_EVAL, True)})
    runtime = original_reports["clean_reference"]["versions"]
    if any(report["versions"] != runtime for report in [*reports.values(), *original_reports.values()]):
        raise ValueError("fixed evaluation runtimes differ")
    behavior = {arm: C._behavior(report) for arm, report in reports.items()}
    ref = behavior["clean_reference"]
    original = {key: C._behavior(value) for key, value in original_reports.items()}
    source_verdict = RC.verdict(RC.compare(original["fault"], original["clean_reference"]), bands)
    if source_verdict["failed"] != original_failed:
        raise ValueError("original failure metrics changed")
    verdicts = {arm: RC.verdict(RC.compare(value, ref), bands) for arm, value in behavior.items()}
    secondary = {arm: RC.verdict(RC.compare(value, original["clean_reference"]), bands) for arm, value in behavior.items()}
    walks = ref["err_xy"] < C.WALK_FRACTION * reports["clean_reference"]["standstill"]["lin_vel_rmse_mps"] and ref["fall_frac"] <= C.REF_FALL_MAX
    overlap = sorted(set(source_verdict["failed"]) & set(verdicts["fault_sham"]["failed"]))
    if source_verdict["label"] != "unhealthy":
        status = "source_failure_absent"
    elif not walks or secondary["clean_reference"]["label"] != "healthy":
        status = "baseline_invalid"
    elif verdicts["fault_sham"]["label"] != "unhealthy":
        status = "no_failure_observed"
    elif not overlap:
        status = "sham_shifted"
    elif verdicts["corrected"]["label"] != "healthy":
        status = "no_recovery"
    else:
        status = "restoration_observed_single_run"
    return {"status": status, "scope": FIXED["scope"], "behavior": behavior, "verdicts": verdicts,
            "original_behavior": original, "source_verdict": source_verdict,
            "secondary_verdicts": secondary, "original_sham_failed_overlap": overlap,
            "reference_walks": walks, "sham_corrected_initial_states_match": True, "bands": bands,
            "sham_corrected_rng_hashes_match": initials["fault_sham"].get("rng_state_before_learn") == initials["corrected"].get("rng_state_before_learn"),
            "full_environment_state_restored": False, "rng_state_restored_from_checkpoint": False,
            "causal_confirmation": False, "stochastic_effect_separated": False,
            "exact_training_trajectory_resumed": False, "arm_order": list(ARMS),
            "model_selected_correction": False, "generalization_proven": False,
            "wording": FIXED["result_wording"][status]}


def execute(tag: str, case: str, isaaclab: Path) -> dict:
    protocol, commit = verify_protocol(isaaclab)
    if case not in CASES:
        raise ValueError("case is not registered")
    folder = folder_for(tag)
    key = digest({"protocol": sha(PROTOCOL, True), "case": case})
    consumption = PRIVATE / "consumed" / f"{key}.json"
    if folder.exists() or consumption.exists():
        raise FileExistsError("case or tag has already been consumed")
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
    request = {"tag": tag, "case": case, "source_commit_sha": commit,
               "protocol_sha256_lf": sha(PROTOCOL, True), "isaaclab_root": str(isaaclab.resolve()),
               "selection_source": "registered_host_feasibility_not_model", "bundle_key": key}
    write_new(folder / "request.json", request)
    write_new(folder / "approval.json", {"request_sha256": sha(folder / "request.json"), "actor": FIXED["approval_actor"], "identity_attested": False})
    write_new(consumption, {"request_sha256": sha(folder / "request.json"), "tag": tag})
    reports, initials, costs = {}, {}, {}
    error_type = None
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
            write_new(arm_dir / "gpu_before.json", gpu)
            write_new(arm_dir / "consumed.json", {"approval_sha256": sha(folder / "approval.json"), "arm": arm, "train_limit": 1, "eval_limit": 1})
            write_new(training / "manifest.json", manifest_for(folder, request, arm))
            argv = [str(isaaclab / "_isaac_sim/python.bat"), str(ROOT / "bench/train_with_correction.py"), "--manifest", str(training / "manifest.json")]
            stage = "training"
            execution = P.run_owned(training, argv, isaaclab, FIXED["train_watchdog_s"])
            write_new(training / "execution.json", execution)
            stage = "training_validation"
            receipt = verify_training(folder, arm, execution)
            initials[arm] = read(training / "initial.json")
            evaluation.mkdir()
            jobs = {"action_scale": FIXED["action_scale"], "jobs": [{"name": arm, "checkpoint": receipt["checkpoint"]["path"], "output": str(evaluation / "report.json")} ]}
            write_new(evaluation / "jobs.json", jobs)
            gpu = M.query_gpu(FIXED["gpu_admission_profile"])
            write_new(evaluation / "gpu_before.json", gpu)
            if gpu["busy"]:
                raise RuntimeError("GPU busy before fixed evaluation")
            argv = [str(isaaclab / "_isaac_sim/python.bat"), str(ROOT / "evals/fixed_eval.py"), "--protocol", str(FIXED_EVAL), "--jobs", str(evaluation / "jobs.json"), "--headless"]
            stage = "fixed_evaluation"
            eval_execution = P.run_owned(evaluation, argv, isaaclab, FIXED["eval_watchdog_s"])
            write_new(evaluation / "execution.json", eval_execution)
            if not execution_valid(eval_execution):
                raise ValueError("fixed evaluation did not complete and clean up")
            reports[arm] = C.validate_fixed_report(read(evaluation / "report.json"), {
                "name": arm, "checkpoint_file": f"model_{FIXED['final_iteration']}.pt",
                "checkpoint_sha256": receipt["checkpoint"]["sha256"], "action_scale": FIXED["action_scale"],
                "protocol_sha256_lf": protocol["fixed_eval_sha256_lf"],
            })
            if reports[arm]["versions"]["rsl-rl-lib"] != FIXED["rsl_rl_version"]:
                raise ValueError("runtime RSL-RL version differs")
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
            write_new(arm_dir / "parent.json", {"status": "pass", "training_receipt_sha256": sha(training / "receipt.json"), "fixed_report_sha256": sha(evaluation / "report.json"), **costs[arm]})
            print(json.dumps({"arm": arm, "status": "validated", **costs[arm]}), flush=True)
        except Exception as error:
            error_type = type(error).__name__
            write_new(folder / "failure.json", {"arm": arm, "stage": stage, "error_type": error_type, "detail": str(error)})
            break
    try:
        fault = CASES[case]["fault"]
        originals = {"clean_reference": read(source_paths(isaaclab, "NONE")["fixed_report"]),
                     "fault": read(source_paths(isaaclab, fault)["fixed_report"])}
        inputs = protocol["resume_inputs"]
        result = (summarize(reports, initials, originals, inputs["bands"], inputs["runs"][fault]["original_failed"])
                  if error_type is None else {"status": "invalid", "error_type": error_type})
    except (KeyError, ValueError, TypeError) as error:
        result = {"status": "invalid", "error_type": type(error).__name__, "detail": str(error)}
    result.setdefault("wording", FIXED["result_wording"][result["status"]])
    result.setdefault("scope", FIXED["scope"])
    result.update(case=case, tag=tag, costs=costs, model_calls=0, source_commit_sha=commit,
                  protocol_sha256_lf=request["protocol_sha256_lf"])
    result["arm_evidence"] = {arm: {"parent_sha256": sha(folder / arm / "parent.json"),
                                    **read(folder / arm / "parent.json")}
                              for arm in ARMS if (folder / arm / "parent.json").is_file()}
    write_new(folder / "result.json", result)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify", "execute"))
    parser.add_argument("--isaaclab", type=Path, default=Path.home() / "IsaacLab")
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--tag")
    parser.add_argument("--case", choices=tuple(CASES))
    args = parser.parse_args(argv)
    if args.command == "prepare":
        if args.contract is None:
            parser.error("prepare requires --contract from the SDK Python CPU checker")
        result = prepare(args.isaaclab, args.contract)
    elif args.command == "verify":
        _, commit = verify_protocol(args.isaaclab)
        result = {"status": "pass", "source_commit_sha": commit}
    else:
        if args.tag is None or args.case is None:
            parser.error("execute requires --tag and --case")
        result = execute(args.tag, args.case, args.isaaclab)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if result.get("status") == "invalid" else 0


if __name__ == "__main__":
    raise SystemExit(main())
