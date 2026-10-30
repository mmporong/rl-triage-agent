"""새 정책 계보 10사례와 같은 seed 정상 참조 8개를 순차 수집한다."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "bench"), str(ROOT / "evals"), str(ROOT / "src")]
import isaac_training_guard_v1 as GUARD
import lineage_source_training_v1 as CHILD
from lease_timestamps import parse_lease_timestamp

SCHEMA = "lineage_source_collection_protocol_v1"
PROTOCOL = "bench/protocols/lineage_source_collection_20261030_v1.json"
OWNER = "rl-triage-lineage-sources-v1"
SOURCE_FILES = (
    "evals/lineage_source_batch_v1.py", "tests/test_lineage_source_batch_v1.py",
    "bench/run_lineage_source_batch_v1.ps1", "evals/preregistered_seed_guard.py",
    "evals/lease_timestamps.py",
)
RUNTIME_FILES = (
    "scripts/reinforcement_learning/rsl_rl/train.py",
    "source/isaaclab/isaaclab/envs/mdp/rewards.py",
    "source/isaaclab/isaaclab/actuators/actuator_pd.py",
    "source/isaaclab/isaaclab/envs/mdp/terminations.py",
    "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/rough_env_cfg.py",
    "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/agents/rsl_rl_ppo_cfg.py",
    "_isaac_sim/kit/python/Lib/site-packages/rsl_rl/algorithms/ppo.py",
    "_isaac_sim/kit/python/Lib/site-packages/rsl_rl/modules/actor_critic.py",
    "_isaac_sim/kit/python/Lib/site-packages/rsl_rl/runners/on_policy_runner.py",
)


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path, *, lf: bool = False) -> str:
    raw = path.read_bytes()
    return hashlib.sha256(raw.replace(b"\r\n", b"\n") if lf else raw).hexdigest()


def source_files() -> tuple[str, ...]:
    # 실제 owned-process/GPU admission 소비 경로의 기존 closure를 재사용한다.
    import selection_comparison_demo_v2 as D
    return tuple(sorted(set(D.CODE) | set(SOURCE_FILES) | set(CHILD.SOURCE_FILES)))


def jobs() -> tuple[list[dict], list[dict], dict]:
    cases, training = [], []
    for index, arm in enumerate(("NONE", "NONE", "F1", "F1", "F2", "F2",
                                  "COMPOUND", "COMPOUND", "F5", "F3")):
        seed, case = 2301 + index, f"case-{index + 1:02d}"
        reference = f"none-s{seed}"
        case_job = reference if arm == "NONE" else f"{arm.lower()}-s{seed}"
        cases.append({"case_id": case, "seed": seed, "arm": arm,
                      "source_job": case_job, "reference_job": reference})
        for candidate in (("NONE",) if arm == "NONE" else ("NONE", arm)):
            job_id = f"{candidate.lower()}-s{seed}"
            training.append({"job_id": job_id, "case_id": case, "arm": candidate,
                             "seed": seed, "run_name": f"lineage_v1_s{seed}_{candidate.lower()}",
                             "task": CHILD.TASK, "num_envs": 4096, "max_iterations": 300})
    smoke = {"job_id": "smoke-none-s2399", "case_id": "preflight", "arm": "NONE",
             "seed": 2399, "run_name": "lineage_v1_smoke_s2399", "task": CHILD.TASK,
             "num_envs": 4096, "max_iterations": 1}
    return cases, training, smoke


def protocol_value(lab: Path) -> dict:
    cases, training, smoke = jobs()
    return {
        "schema": SCHEMA, "phase": "source_collection_only", "cases": cases,
        "training_jobs": training, "smoke_job": smoke,
        "source_sha256_lf": {p: digest(ROOT / p, lf=True) for p in source_files()},
        "runtime_source_sha256_lf": {p: digest(lab / p, lf=True) for p in RUNTIME_FILES},
        "lease_helper_sha256_lf": digest(Path.home() / "CodexSignals/Use-GpuLease.ps1", lf=True),
        "budget": {"smoke_updates": 1, "source_training_runs": 18,
                   "source_updates": 5400, "model_api_requests": 0,
                   "per_job_parent_wall_s": 1800, "total_parent_wall_s": 21600,
                   "automatic_retries": 0, "paid_cloud_gpu_starts": 0},
        "pre_registration_prediction": {
            "NONE": "정상 보행 기준 충족을 예측하며 고정 평가는 다음 등록 단계에서 실시한다.",
            "faults": "알려진 결함군으로 추종이 나빠질 것을 예측하며 실패 부재도 보존한다.",
        },
        "selection": {"fault_types_known": True, "new_seed_is_not_new_fault_type": True,
                      "heldout_proven": False, "replacement_seeds": 0,
                      "reference": "same_seed_NONE", "checkpoint_duplicates": "report_without_replacement"},
        "next_stage": {"registered": False, "fixed_evaluation": "26 conditions; unchanged bands",
                       "comparison_arms": ["registry_next", "fixed_eligible", "nemotron35_numeric"],
                       "visual_diagnostic_arm": "deferred_no_confirmed_selection_gain",
                       "api_allocation": None,
                       "note": "진단 비교·교정·회복은 이 판에서 승인하거나 실행하지 않는다"},
        "scope": "설계자가 개발 결과와 같은 seed 대조군 결과를 본 뒤 만든 새 정책 계보 수집이다. "
                 "수집 완료는 실패 검출·원인 분리·회복·일반화의 확인을 뜻하지 않는다.",
        "outcome_wording": {"complete": "계획된 source 18개와 사전 실행의 학습 증거를 수집했다.",
                            "invalid": "실행 결합 또는 학습 완료 검증이 중단되어 후속 비교를 하지 않았다."},
    }


def validate_protocol(value: dict) -> dict:
    cases, training, smoke = jobs()
    if (value.get("schema") != SCHEMA or value.get("phase") != "source_collection_only"
            or value.get("cases") != cases or value.get("training_jobs") != training
            or value.get("smoke_job") != smoke):
        raise ValueError("등록 source·같은 seed 참조·실행 순서가 고정 계약과 다르다")
    budget = value.get("budget", {})
    if budget != {"smoke_updates": 1, "source_training_runs": 18, "source_updates": 5400,
                  "model_api_requests": 0, "per_job_parent_wall_s": 1800,
                  "total_parent_wall_s": 21600, "automatic_retries": 0,
                  "paid_cloud_gpu_starts": 0}:
        raise ValueError("source 수집 예산이 고정 계약과 다르다")
    if not set(source_files()) <= set(value.get("source_sha256_lf", {})):
        raise ValueError("source closure가 누락됐다")
    if set(value.get("runtime_source_sha256_lf", {})) != set(RUNTIME_FILES):
        raise ValueError("runtime binding이 고정 계약과 다르다")
    return value


def manifest(protocol: dict, row: dict, anchor: str, *, purpose: str) -> dict:
    return CHILD.validate_manifest({
        "schema": CHILD.SCHEMA, "purpose": purpose, **row, "source_commit": anchor,
        "protocol_relative": PROTOCOL, "protocol_sha256_lf": digest(ROOT / PROTOCOL, lf=True),
        "source_sha256_lf": protocol["source_sha256_lf"],
        "runtime_source_sha256_lf": protocol["runtime_source_sha256_lf"],
    })


def verify(lab: Path, anchor: str) -> dict:
    value = validate_protocol(read(ROOT / PROTOCOL))
    CHILD.verify_binding(manifest(value, value["smoke_job"], anchor, purpose="smoke"), repo=ROOT, lab=lab)
    if digest(Path.home() / "CodexSignals/Use-GpuLease.ps1", lf=True) != value["lease_helper_sha256_lf"]:
        raise ValueError("공용 GPU lease helper가 등록본과 다르다")
    for row in value["training_jobs"]:
        CHILD.validate_manifest(manifest(value, row, anchor, purpose="source"))
    return value


def fresh(lab: Path, *, include_smoke: bool = True) -> dict:
    from preregistered_seed_guard import assert_fresh_seeds
    return assert_fresh_seeds([*range(2301, 2311), *([2399] if include_smoke else [])], lab, repo=ROOT)


def validate_lease(metadata: dict, ancestors: list[dict], caller: int, now: datetime) -> dict:
    by_pid = {row["pid"]: row for row in ancestors}
    wrapper = metadata.get("wrapperPid")
    if (metadata.get("owner") != OWNER or metadata.get("state") != "running"
            or metadata.get("childPid") != caller or metadata.get("completedUtc") is not None
            or caller not in by_pid or wrapper not in by_pid
            or by_pid[caller]["parent"] != wrapper
            or by_pid[wrapper]["name"].lower() not in {"powershell.exe", "pwsh.exe"}):
        raise ValueError("현재 controller와 공용 GPU lease 소유자·부모 신원이 다르다")
    acquired = parse_lease_timestamp(metadata["acquiredUtc"])
    heartbeat = parse_lease_timestamp(metadata["heartbeatUtc"])
    if (acquired > heartbeat or not 0 <= (now - heartbeat).total_seconds() <= 60
            or re.fullmatch(r"[a-f0-9-]{36}", metadata.get("leaseId", "")) is None):
        raise ValueError("GPU lease 기간·heartbeat·ID가 잘못됐다")
    return metadata


def lease_context() -> dict:
    if os.name != "nt":
        raise RuntimeError("이 등록 판의 실제 실행은 Windows 공용 lease 경로를 쓴다")
    signals = Path.home() / "CodexSignals"
    # wrapper의 childPid 게시가 Popen 반환 뒤인 짧은 구간만 기다린다.
    for attempt in range(20):
        metadata = read(signals / "gpu-lease.json")
        if metadata.get("childPid") == os.getpid() and metadata.get("state") == "running":
            break
        if attempt != 19:
            time.sleep(.1)
    command = (f"$nodeId={os.getpid()}; $chain=@(); for($i=0;$i -lt 4 -and $nodeId -gt 0;$i++){{"
               "$p=Get-CimInstance Win32_Process -Filter ('ProcessId='+$nodeId);"
               "if($null -eq $p){break};$chain+=@{pid=[int]$p.ProcessId;"
               "parent=[int]$p.ParentProcessId;name=$p.Name};$nodeId=[int]$p.ParentProcessId};"
               "ConvertTo-Json -InputObject @($chain) -Compress")
    response = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
                              check=True, capture_output=True, encoding="utf-8", errors="strict", timeout=30,
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    result = validate_lease(metadata, json.loads(response.stdout), os.getpid(), datetime.now(timezone.utc))
    from ppo_gradient_check_v5 import assert_exclusive_lock
    assert_exclusive_lock(signals / "gpu-lease.lock")
    return result


def completed(stage: Path, fixed: dict, execution: dict) -> dict:
    if (execution.get("exit_code") != 0 or execution.get("natural_exit") is not True
            or execution.get("cleanup_verified") is not True
            or execution.get("job_assigned_before_release") is not True
            or execution.get("timed_out") is not False or execution.get("error_type") is not None):
        raise RuntimeError("owned child의 자연 종료·종료 코드·정리가 확인되지 않았다")
    output = stage / "training"
    GUARD.validate_completion(output, CHILD.manifest_fingerprint(fixed))
    evidence = read(output / GUARD.EVIDENCE)
    if ((output / "failure.json").exists() or evidence.get("arm") != fixed["arm"]
            or evidence.get("seed") != fixed["seed"] or evidence.get("learn_returned") is not True
            or evidence.get("learning_completed") is not True
            or evidence.get("recorded_before_app_close") is not True
            or evidence.get("update_count") != fixed["max_iterations"]):
        raise ValueError("학습 완료 evidence가 등록 job과 다르다")
    checkpoint = evidence["checkpoint"]
    path = Path(checkpoint["path"])
    if (not path.is_file() or digest(path) != checkpoint["sha256"]
            or path.resolve().parent != Path(evidence["log_dir"]).resolve()):
        raise ValueError("학습 checkpoint SHA·로그 계보 결합이 다르다")
    return evidence


def verify_smoke(root: Path, value: dict, anchor: str) -> dict:
    result, intent = read(root / "result.json"), read(root / "intent.json")
    if (result.get("status") != "complete" or result.get("mode") != "smoke"
            or result.get("source_jobs_completed") != 0 or len(result.get("records", [])) != 1
            or intent.get("anchor") != anchor
            or intent.get("protocol_sha256_lf") != digest(ROOT / PROTOCOL, lf=True)):
        raise ValueError("별도 smoke 완료 결과·등록 결합이 다르다")
    row = value["smoke_job"]
    stage = root / row["job_id"]
    fixed = manifest(value, row, anchor, purpose="smoke")
    if read(stage / "manifest.json") != fixed:
        raise ValueError("smoke manifest가 현재 등록 입력과 다르다")
    evidence = completed(stage, fixed, read(stage / "owned-result.json"))
    if (result["records"][0].get("evidence_sha256") != digest(stage / "training" / GUARD.EVIDENCE)
            or result["records"][0].get("checkpoint") != evidence["checkpoint"]):
        raise ValueError("smoke 학습 원문과 결과 참조가 다르다")
    return {"path": str(root.resolve()), "result_sha256": digest(root / "result.json"),
            "evidence_sha256": digest(stage / "training" / GUARD.EVIDENCE)}


def run_sequence(root: Path, value: dict, anchor: str, lab: Path, python: Path, *,
                 mode: str, runner: Callable, check_lease: Callable = lease_context,
                 check_gpu: Callable | None = None, clock: Callable = time.monotonic,
                 smoke_reference: dict | None = None, seed_freshness: dict | None = None) -> dict:
    """단회 배치. 성공한 job도 재실행하지 않으며 실패하면 다음 job을 시작하지 않는다."""
    if mode not in {"smoke", "all"}:
        raise ValueError("mode는 smoke 또는 all이어야 한다")
    value = validate_protocol(value)
    if smoke_reference is not None and (
            mode != "all" or not isinstance(smoke_reference, dict)
            or "path" not in smoke_reference
            or verify_smoke(Path(smoke_reference["path"]), value, anchor) != smoke_reference):
        raise ValueError("smoke 재사용 참조가 원문과 다르다")
    first_lease = check_lease()
    root.mkdir(parents=True, exist_ok=False)
    if seed_freshness is not None:
        GUARD.durable_new(root / "seed-freshness.json", seed_freshness)
    GUARD.durable_new(root / "intent.json", {"mode": mode, "anchor": anchor,
                                             "protocol_sha256_lf": digest(ROOT / PROTOCOL, lf=True)})
    start, records = clock(), []
    selected = [("smoke", value["smoke_job"])]
    if mode == "all":
        if smoke_reference is not None:
            selected.clear()
        selected.extend(("source", row) for row in value["training_jobs"])
    try:
        for purpose, row in selected:
            current_lease = check_lease()
            if current_lease.get("leaseId") != first_lease.get("leaseId"):
                raise ValueError("배치 도중 GPU lease ID가 바뀌었다")
            remaining = value["budget"]["total_parent_wall_s"] - (clock() - start)
            if remaining <= 0:
                raise TimeoutError("등록된 전체 controller 예산이 소진됐다")
            stage = root / row["job_id"]
            stage.mkdir()
            if check_gpu is not None:
                check_gpu(stage)
            fixed = manifest(value, row, anchor, purpose=purpose)
            CHILD.verify_binding(fixed, repo=ROOT, lab=lab)
            GUARD.durable_new(stage / "manifest.json", fixed)
            GUARD.durable_new(stage / "lease-before.json", current_lease)
            argv = [str(python), str(ROOT / "bench/lineage_source_training_v1.py"),
                    "--manifest", str(stage / "manifest.json"), "--lab", str(lab),
                    "--out", str(stage / "training")]
            receipt = dict(runner(stage, argv, lab,
                                 min(remaining, value["budget"]["per_job_parent_wall_s"])))
            GUARD.durable_new(stage / "owned-result.json", receipt)
            if check_lease().get("leaseId") != first_lease.get("leaseId"):
                raise ValueError("child 종료 뒤 GPU lease ID가 바뀌었다")
            evidence = completed(stage, fixed, receipt)
            row_result = {"job_id": row["job_id"], "purpose": purpose, "arm": row["arm"],
                          "seed": row["seed"], "checkpoint": evidence["checkpoint"],
                          "evidence_sha256": digest(stage / "training" / GUARD.EVIDENCE),
                          "owned_parent_wall_s": receipt["wall_s"]}
            records.append(row_result)
            GUARD.durable_new(stage / "accepted.json", row_result)
        grouped: dict[str, list[str]] = {}
        for row in records:
            if row["purpose"] == "source":
                grouped.setdefault(row["checkpoint"]["sha256"], []).append(row["job_id"])
        result = {"schema": SCHEMA + "_result", "status": "complete", "mode": mode,
                  "source_jobs_completed": sum(r["purpose"] == "source" for r in records),
                  "records": records, "duplicate_checkpoint_jobs": [v for v in grouped.values() if len(v) > 1],
                  "model_api_requests": 0, "fixed_evaluation_executed": False,
                  "diagnosis_or_recovery_claim": False, "wall_s": clock() - start}
        result["smoke_reference"] = smoke_reference
        GUARD.durable_new(root / "result.json", result)
        return result
    except BaseException as error:
        GUARD.durable_new(root / "failure.json", {"status": "invalid", "completed_jobs": records,
                                                  "error_type": type(error).__name__, "error": str(error)})
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify", "fresh-seeds", "execute"))
    parser.add_argument("--lab", type=Path, required=True)
    parser.add_argument("--anchor")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--isaac-python", type=Path)
    parser.add_argument("--mode", choices=("smoke", "all"), default="smoke")
    parser.add_argument("--smoke-result", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        if args.output is None:
            parser.error("prepare에는 --output이 필요하다")
        GUARD.durable_new(args.output, protocol_value(args.lab.resolve()))
        return 0
    if args.command == "fresh-seeds":
        result = fresh(args.lab)
        if args.output:
            GUARD.durable_new(args.output, result)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    if not args.anchor:
        parser.error("verify/execute에는 --anchor가 필요하다")
    value = verify(args.lab.resolve(), args.anchor)
    if args.command == "verify":
        print(json.dumps({"status": "valid", "source_jobs": 18, "api": 0}))
        return 0
    if args.output is None or args.isaac_python is None or not args.isaac_python.is_file():
        parser.error("execute에는 새 --output과 실제 --isaac-python이 필요하다")
    smoke_reference = None
    if args.smoke_result:
        if args.mode != "all":
            parser.error("--smoke-result는 all 모드에서만 쓴다")
        smoke_reference = verify_smoke(args.smoke_result.resolve(), value, args.anchor)
    freshness = fresh(args.lab, include_smoke=smoke_reference is None)
    import ppo_ratio_repair_execution_v3 as GPU
    import run_runtime_readback_smoke as OWNED
    result = run_sequence(args.output.resolve(), value, args.anchor, args.lab.resolve(),
                          args.isaac_python.resolve(), mode=args.mode, runner=OWNED.run_owned,
                          check_gpu=lambda stage: GPU.snapshot(stage, "before"),
                          smoke_reference=smoke_reference, seed_freshness=freshness)
    print(json.dumps({"status": result["status"], "sources": result["source_jobs_completed"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
