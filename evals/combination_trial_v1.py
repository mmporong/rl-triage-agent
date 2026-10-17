"""ANYmal-C flat × U1 × 전체 보상 함수군 복원의 새 조합을 사전등록하고 실행한다."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "bench"), str(ROOT / "evals")]

import e2e_demo_anymal_c as ANYMAL  # noqa: E402
import e2e_demo_v4 as E4  # noqa: E402
import isaac_training_guard_v1 as GUARD  # noqa: E402
import ppo_gradient_check_v5 as LEASE  # noqa: E402
import preregistered_seed_guard as SEEDS  # noqa: E402
import record_evidence_media as GPU  # noqa: E402
import reward_family_correction as FAMILY  # noqa: E402
import run_runtime_readback_smoke as OWNED  # noqa: E402
import run_training_correction as V3  # noqa: E402
import train_with_recovery_mode as TRAIN  # noqa: E402
from rl_triage import grounded_recovery as BEHAVIOR  # noqa: E402
from rl_triage import recovery as RECOVERY  # noqa: E402
from rl_triage.correction_ledger import CorrectionLedger  # noqa: E402

PROTOCOL = ROOT / "bench/protocols/combination_trial_v1.json"
PRIVATE = ROOT / "bench/private/codex_support_20261015/task39/runner"
TAG = "combination_anymal_u1_s2150_20261017"
TASK = "Isaac-Velocity-Flat-Anymal-C-v0"
FAULT_SEED = 2150
RESTART_SEED = 3150
ARMS = {
    "source_none": {"fault": "NONE", "seed": FAULT_SEED, "correction_id": None},
    "source_u1": {"fault": "U1", "seed": FAULT_SEED, "correction_id": None},
    "family_restart": {"fault": "U1", "seed": RESTART_SEED, "correction_id": FAMILY.CORRECTION_ID},
}
SOURCE_ARMS = ("source_none", "source_u1")
GPU_PROFILE = "base_com_reset_readback_v2"
MAX_PARENT_WALL_S = 2700
MAX_OWNED_GPU_WALL_S = 2400
BAND_RESULT = ANYMAL.STAGE1_RESULT
CODE = set(ANYMAL.CODE) | set(E4.CODE) | {
    "bench/combination_training_v1.py", "bench/combination_semantics_v1.py",
    "evals/combination_trial_v1.py", "bench/run_combination_trial_v1.ps1",
    "tests/test_combination_trial_v1.py", "tests/test_combination_semantics_v1.py",
    "bench/reward_family_correction.py", "bench/upstream_faults.py", "bench/upstream_isaac.py",
    "bench/train_with_recovery_mode.py", "evals/combination_contract_v1.py",
    "tests/test_combination_contract_v1.py",
    "bench/train_with_correction.py", "bench/correction_adapter.py", "bench/isaac_training_guard_v1.py",
    "bench/run_runtime_readback_smoke.py", "bench/runtime_readback_smoke.py", "bench/record_evidence_media.py",
    "evals/ppo_gradient_check_v5.py", "evals/fixed_eval_task.py",
    "src/rl_triage/correction_registry.py", "src/rl_triage/grounded_recovery.py", "src/rl_triage/recovery.py",
    "src/rl_triage/correction_ledger.py", "src/rl_triage/correction_scope.py",
    "evals/fixed_eval_task.py", "evals/preregistered_seed_guard.py",
}

read = V3.read
write_new = V3.write_new
sha = V3.sha


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args], check=True, capture_output=True,
                          text=True, encoding="utf-8", errors="strict").stdout.strip()


def _git_bytes(*args: str) -> bytes:
    return subprocess.run(["git", "-C", str(ROOT), *args], check=True, capture_output=True).stdout


def _rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _reject_public_paths(value: dict) -> None:
    raw = json.dumps(value, ensure_ascii=False)
    if re.search(r"[A-Za-z]:[\\/]", raw) or re.search(r"/(?:home|mnt)/", raw):
        raise ValueError("공개 protocol에 사용자별 절대경로가 있다")


def fixed() -> dict:
    return {
        "schema": "combination_trial_protocol_v1",
        "status": "preregistered_unmeasured",
        "question": "ANYmal-C flat에서 U1 네 보상 항이 행동 실패를 만들면 등록 함수군 복원 재학습이 고정 평가를 통과하는가",
        "selection": {
            "robot": "ANYmal-C", "task": TASK, "fault": "U1",
            "correction_id": FAMILY.CORRECTION_ID,
            "reason": "네 활성 target의 범위 완결과 Go2에 등록된 같은 교정을 두 번째 로봇 task에 연결하는 계약을 확인한다. 결과 예상으로 고른 조합이 아니다.",
            "development_coupling": "Go2 U1 결과와 ANYmal-C 기존 결과를 본 뒤 설계한 개발 측정이다.",
        },
        "seeds": {"source": FAULT_SEED, "restart": RESTART_SEED,
                  "freshness": "실행 직전 저장소 산출물과 Isaac Lab 로그에서 한 번 재검사한다"},
        "budget": {"num_envs": 4096, "max_iterations": 300, "num_steps_per_env": 24,
                   "source_arms": list(SOURCE_ARMS), "conditional_restart_arms": ["family_restart"],
                   "fixed_eval_conditions": 26, "action_scale": 0.5,
                   "maximum_gpu_runs": 5, "api_calls": 0,
                   "maximum_parent_wall_s": MAX_PARENT_WALL_S,
                   "maximum_owned_gpu_process_wall_s": MAX_OWNED_GPU_WALL_S},
        "sequence": ["cpu_semantics", "freshness", "source_none_train", "source_u1_train",
                     "source_fixed_eval", "source_gate", "contract", "family_restart_train", "restart_fixed_eval", "summary"],
        "stop_rule": "source_failure_absent 또는 baseline_invalid면 family_restart를 만들거나 실행하지 않고 종료한다",
        "contract_rule": "ready 상태와 등록 규칙 요청의 유효 receipt가 있을 때만 family_restart manifest를 만들고 승인·소비한다",
        "predictions": {"semantics": "applicable", "source_gate": "failure_present",
                        "family_restart": "recovered"},
        "prediction_basis": "Go2 U1에서 같은 네 보상 항 복원과 restart 300회가 회복한 기존 개발 측정에 근거한다. 미회복도 유효 결과이며 효과를 보장하지 않는다.",
        "retry_policy": {
            "semantic_retries": 0,
            "no_tuning": "source_failure_absent이면 다른 seed·학습 크기·평가 밴드를 탐색하지 않는다",
            "invalid": "새 판과 새 tag를 사전등록하기 전에는 자동 재시도하지 않는다",
        },
        "success_conditions": {
            "source_fresh": True, "targets_active": 4, "injected_targets_changed": 4,
            "source_failure_gate": "failure_present", "combination_contract": "ready",
            "correction_consumptions": 1, "restored_targets": 4,
            "restart_fixed_eval": "recovered", "within_registered_budget": True,
        },
        "wording": {
            "source_failure_absent": "새 조합에서 등록 행동 실패를 확인하지 못해 교정 실행으로 진행하지 않았다.",
            "baseline_invalid": "정상 대조가 기존 ANYmal-C 걷기 조건을 충족하지 못해 조합을 판정하지 않았다.",
            "recovered": "ANYmal-C 새 조합에서 전체 보상 함수군 복원 재학습 정책이 고정 평가 기준을 통과했다.",
            "not_recovered": "전체 보상 함수군은 복원됐지만 ANYmal-C 새 조합의 재학습 정책이 고정 평가 기준을 통과하지 못했다.",
            "invalid": "적용성·출처·실행·평가 결합 중 하나가 사전등록 계약과 달라 조합을 판정하지 않았다.",
        },
        "evidence_scope": {"diagnosis_model": "inactive", "selection_actor": "registered_rule",
                           "scope_kind": "registered_target_readback", "api_calls": 0,
                           "behavioral_recovery": "family_restart 고정 평가에서만 판정"},
        "fixed_eval": {"protocol": "bench/protocols/fixed_eval_v1_anymal_c.json",
                       "bands": BAND_RESULT, "reuse_without_recalibration": True},
        "source_anchor": {"mode": "experiment_commit_is_ancestor_and_bound_blobs_match_head"},
        "write_once": True,
        "gpu_admission_profile": GPU_PROFILE,
        "watchdogs_s": {"train": 900, "fixed_eval": 300},
        "expected_parent_wall_s": {"basis": "ANYmal-C F6 smoke의 300회 학습 약 471~473초와 고정 평가 약 83초",
                                   "source_failure_absent": 1100, "ready_with_restart": 1700},
    }


def dynamic(isaaclab: Path, semantics: dict | None = None) -> dict:
    stage1 = ROOT / BAND_RESULT
    value = {
        "source_sha256_lf": {path: sha(ROOT / path, True) for path in sorted(CODE)},
        "runtime_source_sha256_lf": ANYMAL.G.isaaclab_sources(isaaclab),
        "fixed_eval_sha256_lf": sha(ANYMAL.EVAL_PROTOCOL, True),
        "band_result": {"path": BAND_RESULT, "sha256_lf": sha(stage1, True)},
        "lease_helper_sha256_lf": sha(Path.home() / "CodexSignals/Use-GpuLease.ps1", True),
    }
    if semantics is not None:
        if semantics.get("status") != "applicable" or not all(semantics.get("checks", {}).values()):
            raise ValueError("CPU 의미 검사가 applicable로 완결되지 않았다")
        value["cpu_semantics"] = {
            "schema": semantics["schema"], "status": semantics["status"],
            "active_targets": semantics["active_targets"], "task_terms": semantics["task_terms"],
            "checks": semantics["checks"], "source_contract": semantics["source_contract"],
            "fault_fingerprints": semantics["fault_fingerprints"],
            "none_noop": {key: semantics["none_noop"][key] for key in (
                "schema", "selected_correction", "registered_functions", "before_fault", "after_fault",
                "after_correction", "changed_targets", "restored_targets", "unrelated_changed_after_fault",
                "unrelated_changed_after_correction")},
            "task_binding": semantics["task_binding"],
        }
    return value


def prepare(isaaclab: Path, semantics_path: Path) -> dict:
    semantics = read(semantics_path)
    document = {**fixed(), **dynamic(isaaclab, semantics),
                "correction_source_contract": semantics["source_contract"]}
    _reject_public_paths(document)
    write_new(PROTOCOL, document)
    return {"status": "prepared_commit_before_execute", "protocol_sha256_lf": sha(PROTOCOL, True)}


def verify(isaaclab: Path, *, committed: bool = True) -> tuple[dict, str | None]:
    protocol = read(PROTOCOL)
    expected = {**fixed(), **dynamic(isaaclab)}
    cpu = protocol.get("cpu_semantics")
    if (set(protocol) != set(expected) | {"cpu_semantics", "correction_source_contract"}
            or any(protocol.get(k) != v for k, v in expected.items())):
        raise ValueError("사전등록 또는 결합 source가 바뀌었다")
    if (not isinstance(cpu, dict) or cpu.get("status") != "applicable"
            or cpu.get("active_targets") != list(FAMILY.FUNCTIONS) or not all(cpu.get("checks", {}).values())):
        raise ValueError("CPU 의미 계약이 다르다")
    verify_cpu_contract(protocol, sha(isaaclab / FAMILY.SOURCE, True))
    if read(ROOT / BAND_RESULT).get("outcome") != "pass":
        raise ValueError("기존 ANYmal-C 밴드가 pass가 아니다")
    if not committed:
        return protocol, None
    head = _git("rev-parse", "HEAD")
    paths = sorted(CODE | {_rel(PROTOCOL)})
    _git("ls-files", "--error-unmatch", "--", *paths)
    if _git("status", "--porcelain", "--", *paths):
        raise ValueError("결합 파일을 먼저 커밋해야 한다")
    return protocol, head


def verify_cpu_contract(protocol: dict, installed_source_sha: str) -> None:
    """공개 protocol의 CPU 계약과 실행 환경에서 읽은 source SHA를 비교한다."""
    cpu = protocol.get("cpu_semantics", {})
    correction = protocol.get("correction_source_contract")
    if (not isinstance(correction, dict) or correction != cpu.get("source_contract")
            or correction.get("runtime") != {
                "implementation": "CPython", "version": [3, 10, 15], "cache_tag": "cpython-310"}
            or correction.get("source_sha256") != installed_source_sha):
        raise ValueError("Isaac Python 3.10 보상 함수군 source 계약이 다르다")


def verify_source_anchor(anchor: str, isaaclab: Path | None = None) -> str:
    """실험 커밋이 HEAD의 조상이고 결합 blob이 같은지 확인한다."""
    head = _git("rev-parse", "HEAD")
    _git("merge-base", "--is-ancestor", anchor, head)
    for path in sorted(CODE | {_rel(PROTOCOL)}):
        if _git("rev-parse", f"{anchor}:{path}") != _git("rev-parse", f"{head}:{path}"):
            raise ValueError(f"실험 커밋 뒤 결합 blob이 바뀌었다: {path}")
        committed = hashlib.sha256(_git_bytes("show", f"{anchor}:{path}").replace(b"\r\n", b"\n").replace(b"\r", b"\n")).hexdigest()
        if sha(ROOT / path, True) != committed:
            raise ValueError(f"결합 파일 bytes가 실험 커밋과 다르다: {path}")
    protocol = read(PROTOCOL)
    if any(sha(ROOT / path, True) != digest for path, digest in protocol["source_sha256_lf"].items()):
        raise ValueError("결합 source SHA가 protocol과 다르다")
    if isaaclab is not None:
        current = dynamic(Path(isaaclab))
        for key in ("runtime_source_sha256_lf", "lease_helper_sha256_lf", "fixed_eval_sha256_lf", "band_result"):
            if current[key] != protocol[key]:
                raise ValueError(f"실행 중 runtime 결합이 바뀌었다: {key}")
    return head


def folder_for(tag: str) -> Path:
    if tag != TAG:
        raise ValueError("등록 tag가 아니다")
    path = PRIVATE / "runs" / tag
    if path.is_symlink() or path.resolve().parent != (PRIVATE / "runs").resolve():
        raise ValueError("결과 경로가 task39 runner 밖이다")
    return path


def consumed_path() -> Path:
    return PRIVATE / "consumed" / f"{sha(PROTOCOL, True)}.json"


def active_lease() -> dict:
    """기존 공용 lease의 PID 계층·heartbeat·배타 lock 검사를 현재 owner로 재사용한다."""
    previous = LEASE.LEASE_OWNER
    LEASE.LEASE_OWNER = "rl-triage-combination-trial-v1"
    try:
        return LEASE.lease_context()
    finally:
        LEASE.LEASE_OWNER = previous


def assert_lease(expected_lease_id: str) -> dict:
    current = active_lease()
    if current.get("leaseId") != expected_lease_id:
        raise RuntimeError("GPU lease가 실행 도중 바뀌었다")
    return current


def run_name(arm: str) -> str:
    if arm not in ARMS:
        raise ValueError("등록 arm이 아니다")
    return f"combination_anymal_{arm}_s{ARMS[arm]['seed']}"


def manifest_for(tag: str, arm: str, isaaclab: Path, anchor: str) -> dict:
    spec = ARMS[arm]
    output = folder_for(tag) / arm / "train"
    return {
        "schema": "training_correction_manifest_v2", "tag": tag, "seed": spec["seed"], "arm": arm,
        "recovery_mode": "restart", "source_commit_sha": anchor,
        "protocol_sha256_lf": sha(PROTOCOL, True), "isaaclab_root": str(isaaclab.resolve()),
        "protocol_path": str(PROTOCOL.resolve()), "fault": spec["fault"], "correction_id": spec["correction_id"],
        "num_envs": 4096, "max_iterations": 300, "task": TASK, "run_name": run_name(arm),
        "output_dir": str(output.resolve()), "final_iteration": 299,
        "resume_checkpoint_path": None, "resume_checkpoint_sha256": None, "resume_iteration": None,
        "load_run_regex": None, "checkpoint_regex": None, "source_env_config_path": str(PROTOCOL.resolve()),
        "source_agent_config_path": str(PROTOCOL.resolve()), "config_source_fault": spec["fault"], "init_noise_std": None,
    }


def validate_manifest(path: Path) -> tuple[dict, dict]:
    manifest = read(path)
    if set(manifest) != TRAIN.MANIFEST_FIELDS or manifest.get("arm") not in ARMS:
        raise ValueError("조합 학습 manifest schema 또는 arm이 다르다")
    expected_path = folder_for(manifest["tag"]) / manifest["arm"] / "train/manifest.json"
    if path.resolve() != expected_path.resolve() or manifest != manifest_for(
            manifest["tag"], manifest["arm"], Path(manifest["isaaclab_root"]), manifest["source_commit_sha"]):
        raise ValueError("조합 학습 manifest가 등록 입력과 다르다")
    protocol, _ = verify(Path(manifest["isaaclab_root"]), committed=False)
    verify_source_anchor(manifest["source_commit_sha"], Path(manifest["isaaclab_root"]))
    if manifest["arm"] == "family_restart":
        folder = folder_for(manifest["tag"])
        _validate_contract_consumption(folder, manifest["source_commit_sha"])
    return manifest, protocol


def validate_readback(row: dict, fault: str, correction: str | None) -> None:
    if (row.get("schema") != "reward_family_correction_readback_v1"
            or row.get("registered_functions") != list(FAMILY.FUNCTIONS)
            or row.get("selected_correction") != correction
            or row.get("unrelated_changed_after_fault") or row.get("unrelated_changed_after_correction")):
        raise ValueError("보상 함수군 readback 기본 계약이 다르다")
    clean, injected, after = (row.get(k, {}) for k in ("before_fault", "after_fault", "after_correction"))
    changed = {name for name in FAMILY.FUNCTIONS if clean.get(name) != injected.get(name)}
    expected_changed = set() if fault == "NONE" else set(FAMILY.FUNCTIONS)
    if changed != expected_changed:
        raise ValueError("결함이 등록 함수군과 다르게 바뀌었다")
    expected_after = clean if correction == FAMILY.CORRECTION_ID else injected
    if after != expected_after:
        raise ValueError("교정 뒤 함수군 상태가 등록 선택과 다르다")


def validate_training_configuration(log_dir: Path, manifest: dict) -> None:
    env = V3.load_config(log_dir / "params/env.yaml")
    agent = V3.load_config(log_dir / "params/agent.yaml")
    checks = {
        "env_seed": env.get("seed") == manifest["seed"],
        "agent_seed": agent.get("seed") == manifest["seed"],
        "num_envs": env.get("scene", {}).get("num_envs") == 4096,
        "run_name": agent.get("run_name") == manifest["run_name"],
        "steps": agent.get("num_steps_per_env") == 24,
        "iterations": agent.get("max_iterations") == 300,
        "action_scale": env.get("actions", {}).get("joint_pos", {}).get("scale") == 0.5,
    }
    if not all(checks.values()):
        raise ValueError(f"ANYmal 조합 학습 설정 불일치: {[k for k, ok in checks.items() if not ok]}")


def _admit(stage: Path) -> None:
    row = GPU.query_gpu(GPU_PROFILE)
    if row.get("admission_policy") != GPU_PROFILE or row.get("busy") is not False:
        raise RuntimeError("GPU 입장 검사가 busy이거나 등록 profile과 다르다")
    write_new(stage / "gpu_before.json", row)


def _execute_owned(stage: Path, args: list[str], cwd: Path, timeout: int, lease_id: str) -> dict:
    stage.mkdir(parents=True, exist_ok=False)
    assert_lease(lease_id)
    _admit(stage)
    execution = OWNED.run_owned(stage, args, cwd, timeout)
    write_new(stage / "execution.json", execution)
    if not V3.execution_valid(execution) or read(stage / "launch.json") != {"args": args, "cwd": str(cwd.resolve())}:
        raise RuntimeError("소유 실행기 완료·launch 결합이 다르다")
    assert_lease(lease_id)
    return execution


def _training_args(isaaclab: Path, manifest: Path, lease_id: str) -> list[str]:
    return [str((isaaclab / "_isaac_sim/python.bat").resolve()), "-B",
            str((ROOT / "bench/combination_training_v1.py").resolve()), "--manifest", str(manifest.resolve()),
            "--lease-id", lease_id]


def _run_training(folder: Path, arm: str, isaaclab: Path, anchor: str, protocol: dict,
                  lease_id: str) -> dict:
    verify_source_anchor(anchor, isaaclab)
    path = folder / arm / "train/manifest.json"
    write_new(path, manifest_for(TAG, arm, isaaclab, anchor))
    execution = _execute_owned(folder / arm / "owned", _training_args(isaaclab, path, lease_id), isaaclab,
                               protocol["watchdogs_s"]["train"], lease_id)
    validate_child_completion(folder / arm / "train", path)
    receipt = read(folder / arm / "train/receipt.json")
    manifest, _ = validate_manifest(path)
    if (receipt.get("completed") is not True or receipt.get("manifest_sha256") != sha(path)
            or receipt.get("checkpoint", {}).get("sha256") != sha(Path(receipt["checkpoint"]["path"]))
            or receipt.get("actual_control_steps") != 7200 or receipt.get("task") != TASK
            or read(folder / arm / "train/family_environment.json").get("ok") is not True):
        raise ValueError("조합 학습 receipt·체크포인트·환경 결합이 다르다")
    validate_readback(read(folder / arm / "train/correction.json"), manifest["fault"], manifest["correction_id"])
    verify_source_anchor(anchor, isaaclab)
    return {"execution": execution, "checkpoint": receipt["checkpoint"], "receipt_sha256": sha(folder / arm / "train/receipt.json")}


def validate_child_completion(output: Path, manifest_path: Path) -> None:
    GUARD.validate_completion(output, sha(manifest_path, False))


def _evaluate(folder: Path, name: str, arms: list[str], isaaclab: Path, protocol: dict,
              anchor: str, lease_id: str) -> tuple[dict, dict]:
    verify_source_anchor(anchor, isaaclab)
    assert_lease(lease_id)
    stage = folder / name
    jobs = []
    for arm in arms:
        receipt = read(folder / arm / "train/receipt.json")
        jobs.append({"name": run_name(arm), "checkpoint": receipt["checkpoint"]["path"],
                     "output": str((stage / f"{arm}.json").resolve())})
    stage.mkdir(parents=True, exist_ok=False)
    write_new(stage / "jobs.json", {"action_scale": 0.5, "jobs": jobs})
    args = [str((isaaclab / "_isaac_sim/python.bat").resolve()), str((ROOT / "evals/fixed_eval_task.py").resolve()),
            "--protocol", str(ANYMAL.EVAL_PROTOCOL.resolve()), "--jobs", str((stage / "jobs.json").resolve()), "--headless"]
    _admit(stage)
    execution = OWNED.run_owned(stage, args, isaaclab, protocol["watchdogs_s"]["fixed_eval"])
    write_new(stage / "execution.json", execution)
    if not V3.execution_valid(execution) or read(stage / "launch.json") != {"args": args, "cwd": str(isaaclab.resolve())}:
        raise RuntimeError("고정 평가 소유 실행이 완결되지 않았다")
    reports = {}
    for arm in arms:
        receipt = read(folder / arm / "train/receipt.json")
        reports[arm] = ANYMAL.validate_report(read(stage / f"{arm}.json"), {
            "name": run_name(arm), "checkpoint_file": "model_299.pt",
            "checkpoint_sha256": receipt["checkpoint"]["sha256"], "action_scale": 0.5,
            "protocol_sha256_lf": protocol["fixed_eval_sha256_lf"],
        })
    assert_lease(lease_id)
    verify_source_anchor(anchor, isaaclab)
    return reports, execution


def _contract_module():
    import combination_contract_v1 as contract
    return contract


def _contract_manifest(folder: Path, protocol: dict, reports: dict) -> tuple[dict, dict]:
    contract = _contract_module()
    semantics = protocol["cpu_semantics"]
    fault_readback = read(folder / "source_u1/train/correction.json")
    fault_binding = read(folder / "source_u1/train/family_environment.json")
    none_readback = read(folder / "source_none/train/correction.json")
    registered_noop = semantics["none_noop"]
    if (registered_noop.get("selected_correction") != FAMILY.CORRECTION_ID
            or none_readback["before_fault"] != none_readback["after_fault"]
            or none_readback["before_fault"] != registered_noop.get("before_fault")
            or registered_noop.get("before_fault") != registered_noop.get("after_correction")):
        raise ValueError("NONE clean readback과 등록 교정 no-op CPU 근거가 다르다")
    targets = {name: {"path": f"{FAMILY.MODULE}:{name}", "exists": True, "active": True,
                      "effect_reachable": True} for name in FAMILY.FUNCTIONS}
    stages = {
        "applicability": {"targets": targets},
        "injection": {"before": fault_readback["before_fault"], "after": fault_readback["after_fault"],
                      "bound": fault_binding["bound_to_reward_terms"]},
        "source": {"none": reports["source_none"], "fault": reports["source_u1"],
                   "bands": read(ROOT / BAND_RESULT)["analysis"]["bands"]},
        "scope": {"probe": None, "measurement": None, "verification_kind": "registered_target_readback"},
        "noop": {"before": registered_noop["before_fault"], "after": registered_noop["after_correction"],
                 "unrelated_changed": registered_noop["unrelated_changed_after_correction"]},
        "recovery": {"mode": "restart"},
    }
    observation = {"schema": "combination_observation_v1",
                   "combination": {"robot": "ANYmal-C", "task": TASK, "fault": "U1",
                                   "correction_id": FAMILY.CORRECTION_ID},
                   "stages": stages}
    observation_path = folder / "contract_observation.json"
    write_new(observation_path, observation)
    evidence_paths = [PROTOCOL, folder / "seed_freshness.json",
                      folder / "source_none/train/correction.json", folder / "source_none/train/family_environment.json",
                      folder / "source_u1/train/correction.json", folder / "source_u1/train/family_environment.json",
                      folder / "source_eval/source_none.json", folder / "source_eval/source_u1.json", observation_path]
    evidence = [{"path": _rel(path), "sha256_lf": sha(path, True)} for path in evidence_paths]
    manifest = contract.make_manifest("ANYmal-C", TASK, "U1", FAMILY.CORRECTION_ID, stages, evidence)
    state = contract.evaluate(manifest, root=ROOT)
    if state.get("status") != "ready":
        raise ValueError(f"조합 적용성 계약이 ready가 아니다: {state.get('reason')}")
    return manifest, state


def _proposal_state() -> dict:
    return {"status": "confirmed", "hypotheses": ["reward_family"],
            "observed": {"target_readback": "abnormal"}, "selection_actor": "registered_rule"}


def _write_contract_request(folder: Path, protocol: dict, reports: dict, anchor: str, isaaclab: Path) -> dict:
    verify_source_anchor(anchor, isaaclab)
    contract = _contract_module()
    manifest, state = _contract_manifest(folder, protocol, reports)
    write_new(folder / "combination_manifest.json", manifest)
    write_new(folder / "combination_state.json", state)
    ledger = CorrectionLedger(folder / "correction_ledger.jsonl", contract.registry())
    proposed = contract.propose(manifest, _proposal_state(), ledger, root=ROOT, budget_gpu_s=2400)
    if proposed.get("event") != "proposed" or not proposed.get("request_id"):
        write_new(folder / "contract_hold.json", proposed)
        raise ValueError("적용성 계약이 교정 요청을 만들지 않고 HOLD했다")
    request_id = proposed["request_id"]
    approved = ledger.approve(request_id, "registered_trial_controller")
    consumed = contract.consume(manifest, _proposal_state(), ledger, request_id, root=ROOT)
    write_new(folder / "contract_request.json", {"request_id": request_id, "proposed": proposed,
                                                   "approved": approved, "consumed": consumed,
                                                   "manifest_digest": state["manifest_digest"]})
    verify_source_anchor(anchor, isaaclab)
    return {"request_id": request_id, "manifest": manifest, "state": state}


def _validate_contract_consumption(folder: Path, anchor: str) -> dict:
    contract = _contract_module()
    manifest = read(folder / "combination_manifest.json")
    state = contract.evaluate(manifest, root=ROOT)
    request = read(folder / "contract_request.json")
    ledger = CorrectionLedger(folder / "correction_ledger.jsonl", contract.registry())
    row = ledger.requests().get(request["request_id"])
    proposal_state = _proposal_state()
    state_digest = contract.state_digest(manifest, proposal_state)
    entry = contract.registry()["reward_family"]
    expected_effect = ledger.effect_key(state_digest, FAMILY.CORRECTION_ID, "restart")
    expected_evidence = {"combination_contract": state,
                         "restoration_targets": state["restoration_targets"],
                         "selection_actor": "registered_rule", "manifest_digest": state["manifest_digest"]}
    events = ledger.events()
    actual_events = [event for event in events if event.get("request_id") == request["request_id"]]
    if (state.get("status") != "ready" or state.get("manifest_digest") != request.get("manifest_digest")
            or row is None or row.get("state") != "running" or row.get("correction_id") != FAMILY.CORRECTION_ID
            or row.get("recovery_mode") != "restart" or row.get("loop_state_digest") != state_digest
            or row.get("receipts_digest") != contract.L.digest(manifest["evidence"])
            or row.get("confirmed_mechanism") != "reward_family" or row.get("effect_key") != expected_effect
            or row.get("registry_digest") != ledger.registry_digest or row.get("budget_gpu_s") != 2400
            or row.get("protocol_case_arm") != {"case": entry["protocol_case"], "arm": "corrected"}
            or row.get("evidence") != expected_evidence
            or len(actual_events) != 3
            or request.get("proposed") != actual_events[0] or request.get("approved") != actual_events[1]
            or request.get("consumed") != actual_events[2]):
        raise ValueError("ready 계약 요청의 승인·소비·source 결합이 다르다")
    verify_source_anchor(anchor, Path(read(folder / "source_none/train/manifest.json")["isaaclab_root"]))
    return row


def _failure_stage(folder: Path) -> str:
    checks = (("restart_eval", folder / "restart_eval"), ("family_restart", folder / "family_restart"),
              ("contract", folder / "combination_manifest.json"), ("source_eval", folder / "source_eval"),
              ("source_u1", folder / "source_u1"), ("source_none", folder / "source_none"))
    return next((name for name, path in checks if path.exists()), "initialize")


def _receipt_outcome(behavior_label: str, receipt: dict) -> str:
    expected = {"recovered": "recovered", "not_recovered": "not_recovered"}.get(behavior_label, "unknown")
    return expected if receipt.get("verdict") == expected and receipt.get("over_budget") is False else "invalid"


def _source_next(outcome: str) -> str:
    if outcome == "failure_present":
        return "contract"
    if outcome in {"source_failure_absent", "baseline_invalid"}:
        return "stop"
    raise ValueError("등록되지 않은 source gate 결과")


def _assert_total_budget(started: float, costs: dict[str, float], protocol: dict) -> dict:
    parent_wall = time.monotonic() - started
    owned_gpu_wall = sum(costs.values())
    budget = protocol["budget"]
    if (parent_wall > budget["maximum_parent_wall_s"]
            or owned_gpu_wall > budget["maximum_owned_gpu_process_wall_s"]):
        raise RuntimeError("조합 실행의 등록 총 시간 상한을 넘었다")
    return {"parent_wall_s": parent_wall, "owned_gpu_process_wall_s": owned_gpu_wall}


def _execute(tag: str, isaaclab: Path) -> dict:
    protocol, anchor = verify(isaaclab, committed=True)
    assert anchor is not None
    folder = folder_for(tag)
    if folder.exists() or consumed_path().exists():
        raise FileExistsError("등록 조합 tag가 이미 소비됐다")
    lease = active_lease()
    lease_id = lease["leaseId"]
    started = time.monotonic()
    fresh = SEEDS.assert_fresh_seeds([FAULT_SEED, RESTART_SEED], isaaclab, repo=ROOT)
    verify_source_anchor(anchor, isaaclab)
    folder.mkdir(parents=True, exist_ok=False)
    write_new(consumed_path(), {"tag": tag, "seeds": [FAULT_SEED, RESTART_SEED],
                                "protocol_sha256_lf": sha(PROTOCOL, True), "source_commit_sha": anchor})
    write_new(folder / "lease.json", lease)
    write_new(folder / "seed_freshness.json", fresh)
    costs = {}
    for arm in SOURCE_ARMS:
        _assert_total_budget(started, costs, protocol)
        result = _run_training(folder, arm, isaaclab, anchor, protocol, lease_id)
        costs[arm] = result["execution"]["wall_s"]
        _assert_total_budget(started, costs, protocol)
    _assert_total_budget(started, costs, protocol)
    reports, source_eval = _evaluate(folder, "source_eval", list(SOURCE_ARMS), isaaclab, protocol, anchor, lease_id)
    costs["source_eval"] = source_eval["wall_s"]
    totals = _assert_total_budget(started, costs, protocol)
    gate = E4.source_gate(reports["source_none"], reports["source_u1"], read(ROOT / BAND_RESULT)["analysis"]["bands"])
    base = {"schema": "combination_trial_result_v1", "tag": tag, "source_commit_sha": anchor,
            "protocol_sha256_lf": sha(PROTOCOL, True), "seeds": [FAULT_SEED, RESTART_SEED],
            "source_gate": gate, "model_calls": 0, "selection_actor": "registered_rule",
            "owned_gpu_process_wall_s_by_stage": costs, "time_totals": totals}
    if _source_next(gate["outcome"]) == "stop":
        verify_source_anchor(anchor, isaaclab)
        totals = _assert_total_budget(started, costs, protocol)
        result = {**base, "status": "complete", "outcome": gate["outcome"], "correction_executed": False,
                  "time_totals": totals,
                  "wording": protocol["wording"][gate["outcome"]]}
        write_new(folder / "result.json", result)
        return result
    request = _write_contract_request(folder, protocol, reports, anchor, isaaclab)
    _assert_total_budget(started, costs, protocol)
    corrected = _run_training(folder, "family_restart", isaaclab, anchor, protocol, lease_id)
    costs["family_restart"] = corrected["execution"]["wall_s"]
    _assert_total_budget(started, costs, protocol)
    corrected_reports, corrected_eval = _evaluate(
        folder, "restart_eval", ["family_restart"], isaaclab, protocol, anchor, lease_id)
    costs["restart_eval"] = corrected_eval["wall_s"]
    totals = _assert_total_budget(started, costs, protocol)
    verdict = RECOVERY.verdict(RECOVERY.compare(BEHAVIOR._behavior(corrected_reports["family_restart"]),
                                                 BEHAVIOR._behavior(reports["source_none"])),
                               read(ROOT / BAND_RESULT)["analysis"]["bands"])
    behavior_label = {"healthy": "recovered", "unhealthy": "not_recovered"}.get(verdict["label"], "unknown")
    ledger = CorrectionLedger(folder / "correction_ledger.jsonl", _contract_module().registry())
    receipt = ledger.receipt(request["request_id"], tag, behavior_label,
                             sha(folder / "restart_eval/family_restart.json"), costs["family_restart"] + costs["restart_eval"], 0)
    label = _receipt_outcome(behavior_label, receipt)
    verify_source_anchor(anchor, isaaclab)
    totals = _assert_total_budget(started, costs, protocol)
    result = {**base, "status": "complete", "outcome": label, "correction_executed": True,
              "time_totals": totals,
              "contract": request["state"], "correction_request_id": request["request_id"],
              "correction_receipt": receipt, "restart_verdict": verdict, "behavior_label": behavior_label,
              "wording": protocol["wording"][label]}
    write_new(folder / "result.json", result)
    return result


def execute(tag: str, isaaclab: Path) -> dict:
    folder = folder_for(tag)
    existed_before = folder.exists()
    consumed_before = consumed_path().exists()
    try:
        return _execute(tag, isaaclab)
    except Exception as error:
        marker = read(consumed_path()) if consumed_path().is_file() else {}
        owned_failure = (not existed_before and not consumed_before and folder.is_dir()
                         and marker.get("tag") == tag
                         and marker.get("protocol_sha256_lf") == sha(PROTOCOL, True))
        if owned_failure and not (folder / "result.json").exists():
            anchor = marker.get("source_commit_sha")
            write_new(folder / "result.json", {
                "schema": "combination_trial_result_v1", "tag": tag, "status": "invalid", "outcome": "invalid",
                "failure": {"stage": _failure_stage(folder), "error_type": type(error).__name__,
                            "message": str(error)[:1000]},
                "source_commit_sha": anchor, "protocol_sha256_lf": sha(PROTOCOL, True),
                "seeds": [FAULT_SEED, RESTART_SEED], "model_calls": 0,
                "correction_executed": (folder / "family_restart/train/correction.json").is_file(),
                "wording": read(PROTOCOL)["wording"]["invalid"],
            })
        raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify", "execute", "summarize"))
    parser.add_argument("--isaaclab", type=Path, default=Path.home() / "IsaacLab")
    parser.add_argument("--semantics", type=Path, default=PRIVATE / "semantics_v2.json")
    parser.add_argument("--tag", default=TAG)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            out = prepare(args.isaaclab, args.semantics)
        elif args.command == "verify":
            protocol, anchor = verify(args.isaaclab, committed=True)
            out = {"status": "pass", "schema": protocol["schema"], "source_commit_sha": anchor}
        elif args.command == "execute":
            out = execute(args.tag, args.isaaclab)
        else:
            out = read(folder_for(args.tag) / "result.json")
    except (ValueError, KeyError, FileNotFoundError, FileExistsError, RuntimeError,
            subprocess.CalledProcessError) as error:
        print(f"거부: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    print(json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
