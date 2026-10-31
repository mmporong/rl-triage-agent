"""고정 lineage epoch-1 후보의 첫 등록 교정만 실행하는 fail-closed controller."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "bench"), str(ROOT / "evals"), str(ROOT / "src")]

import isaac_training_guard_v1 as GUARD
import lineage_comparison_batch_v1 as BATCH
import lineage_gpu_wait_v1 as GPU_WAIT
import lineage_recovery_candidate_v1 as CANDIDATE
import lineage_recovery_inputs_v1 as RECOVERY_INPUTS
import lineage_recovery_training_v1 as TRAIN
import lineage_source_batch_v1 as SOURCE
import selection_comparison_controller_v1 as C

SCHEMA = "lineage_recovery_execution_protocol_v1"
MANIFEST_SCHEMA = "lineage_recovery_execution_manifest_v1"
RECEIPT_SCHEMA = "lineage_recovery_receipt_v1"
APPROVAL_SCHEMA = "lineage_recovery_operator_approval_v1"
TARGET_SCHEMA = "lineage_e2e_target_selection_before_model_v1"
TARGET_SHA256 = "a88d63b5dbccedfd281366467da5e784d9e0c00a51136ce5d906c94e4a6f8c73"
TARGETS = (
    {"case_id": "case-03", "seed": 2303, "arm": "F1",
     "source_job": "f1-s2303", "reference_job": "none-s2303"},
    {"case_id": "case-07", "seed": 2307, "arm": "COMPOUND",
     "source_job": "compound-s2307", "reference_job": "none-s2307"},
)
POLICIES = ("registry_next", "model_adaptive")
FIXED_PROTOCOL = ROOT / "bench/protocols/fixed_eval_v1.json"
BANDS = ROOT / C.D1.BANDS
OWNER = SOURCE.OWNER
WATCHDOG = {"smoke": 300, "train": 1800, "eval": 600, "total": 21600, "gpu_wait": 600}
HEX40 = re.compile(r"[0-9a-f]{40}\Z")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
SOURCE_FILES = tuple(sorted({
    *TRAIN.BOUND_SOURCES,
    "evals/lineage_recovery_execution_v1.py",
    "tests/test_lineage_recovery_execution_v1.py",
    "bench/run_lineage_recovery_v1.ps1",
    "evals/lineage_comparison_batch_v1.py",
    "evals/lineage_recovery_candidate_v1.py",
    "evals/lineage_recovery_inputs_v1.py",
    "evals/lineage_gpu_wait_v1.py",
    "evals/selection_comparison_controller_v1.py",
    "src/rl_triage/correction_ledger.py",
    "evals/residual_contract_v1.py",
    "evals/residual_demo_v1.py",
}))


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"),
                       parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)))
    _require(isinstance(value, dict), f"{Path(path).name}은 JSON 객체여야 한다")
    return value


def _sha(path: Path, *, lf: bool = False) -> str:
    raw = Path(path).read_bytes()
    if lf:
        raw = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(raw).hexdigest()


def _digest(value: object) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                     allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _write_new(path: Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _ref(path: Path) -> dict[str, str]:
    path = Path(path).resolve()
    _require(path.is_file(), f"근거 파일이 없다: {path}")
    return {"path": str(path), "sha256": _sha(path)}


def _relative(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(ROOT).as_posix()
    except ValueError as exc:
        raise ValueError("공개 component는 저장소 내부 파일이어야 한다") from exc


def _git(*args: str) -> bytes:
    result = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, check=False)
    if result.returncode:
        raise ValueError("Git anchor/blob 결합을 확인하지 못했다")
    return result.stdout


def prepare_protocol(batch_protocol: Path, fixed_protocol: Path = FIXED_PROTOCOL) -> dict[str, Any]:
    """자기 commit/hash cycle과 private 경로가 없는 공개 실행 protocol을 만든다."""
    batch_path, fixed_path = Path(batch_protocol).resolve(), Path(fixed_protocol).resolve()
    batch = BATCH._read(batch_path)
    _require(batch.get("schema") == BATCH.SCHEMA, "comparison batch protocol schema가 다르다")
    runtime = batch.get("runtime_source_sha256_lf")
    _require(isinstance(runtime, dict) and len(runtime) == 9,
             "recovery runtime source binding은 등록된 9개 파일이어야 한다")
    _require(fixed_path == FIXED_PROTOCOL.resolve() and fixed_path.is_file(),
             "동결 fixed_eval_v1 protocol만 허용한다")
    components = {
        "comparison_batch": {"path": _relative(batch_path), "sha256_lf": _sha(batch_path, lf=True)},
        "fixed_evaluation": {"path": _relative(fixed_path), "sha256_lf": _sha(fixed_path, lf=True)},
        "bands": {"path": _relative(BANDS), "sha256": _sha(BANDS)},
    }
    return {
        "schema": SCHEMA, "phase": "first_registered_correction_only",
        "scope": {"epoch": 1, "second_epoch": False, "overall_goal_complete": False,
                  "second_epoch_requires": "new_protocol_after_valid_not_recovered_receipt"},
        "targets": deepcopy(list(TARGETS)), "policies": list(POLICIES),
        "target_commitment": {"filename": "e2e-targets-before-model.json",
                              "sha256": TARGET_SHA256,
                              "selection_rule": "first_registered_F1_and_first_registered_COMPOUND"},
        "predictions": {
            "f1_valid_reward_candidate": "same_seed_fresh_restart_300_expected_recovered",
            "compound_first_single_correction": (
                "same_seed_fresh_restart_300_expected_not_recovered_due_to_remaining_fault"),
            "unconfirmed_or_unsupported_candidate": "HOLD_with_zero_training_jobs",
            "contrary_healthy_compound_result": (
                "preserve_observed_result_without_prediction_change_or_forced_second_correction"),
        },
        "outcome_wording": {
            "recovered": (
                "등록된 첫 교정과 같은 seed의 fresh 300-update 정책이 동결 고정 평가 회복 기준을 통과했다."),
            "not_recovered": (
                "등록된 첫 교정 뒤에도 동결 고정 평가 회복 기준을 충족하지 못했으며 이 판에서 두 번째 교정을 실행하지 않는다."),
            "HOLD": (
                "후보 미확정·지원 범위 밖 또는 평가 불확정으로 학습을 시작하지 않거나 새 결과를 보존하고 중단했다."),
            "invalid": (
                "입력·승인·계보·실행 완료 계약이 불완전해 결과를 회복 또는 미회복으로 판정하지 않았다."),
            "claim_scope": (
                "두 사전 고정 사례의 첫 등록 교정 짝 대조이며 원인 확정, LLM 우위, 일반화 또는 새로운 독립 seed를 주장하지 않는다."),
        },
        "training": {"mode": "fresh_restart", "iterations": 300, "num_envs": 4096,
                     "configured_learning_epochs": 5, "configured_mini_batches": 4,
                     "source_checkpoint_role": "provenance_only_never_loaded", "seed_reuse": True,
                     "seed_reuse_claim": "paired_same_seed_different_correction_not_independent_seed"},
        "smoke": {"required_before_execute": True, "arm": "NONE", "mode": "inspection",
                  "updates": 3, "corrections": 0, "model_calls": 0,
                  "success_cases": 0, "not_a_300_update_control": True},
        "evaluation": {"conditions": 26, "eval_seed": 2026, "num_envs": 1040,
                       "horizon_steps": 1000, "action_scale": 0.25,
                       "same_seed_none_report_reused": True},
        "stop_rules": {"candidate_hold_sdk_jobs": 0, "unknown": "HOLD_and_stop_no_retry",
                       "failed_job": "preserve_outputs_and_start_no_next_job",
                       "automatic_retry": False},
        "watchdogs_s": deepcopy(WATCHDOG), "gpu": {"profile": GPU_WAIT.GPU.GPU_PROFILE,
            "max_wait_s": 600, "consecutive_clear": 2, "owner": OWNER,
            "paid_cloud_start": False},
        "component_protocols": components,
        "runtime_source_sha256_lf": deepcopy(runtime),
        "source_sha256_lf": {name: _sha(ROOT / name, lf=True) for name in SOURCE_FILES},
        "claims": {"new_independent_seed": False, "overall_goal_complete": False,
                   "model_output_is_execution_authority": False, "api_requests": 0},
    }


def verify_protocol(protocol_path: Path, *, anchor: str, lab: Path) -> dict[str, Any]:
    path, lab = Path(protocol_path).resolve(), Path(lab).resolve()
    _require(HEX40.fullmatch(str(anchor)) is not None, "runtime anchor 형식이 잘못됐다")
    value = _read(path)
    components = value.get("component_protocols")
    _require(isinstance(components, dict), "component_protocols가 없다")
    expected = prepare_protocol(ROOT / components["comparison_batch"]["path"],
                                ROOT / components["fixed_evaluation"]["path"])
    _require(value == expected, "공개 recovery protocol 또는 component SHA가 다르다")
    _git("cat-file", "-e", f"{anchor}^{{commit}}")
    _git("merge-base", "--is-ancestor", anchor, "HEAD")
    bound = {_relative(path): _sha(path, lf=True), **value["source_sha256_lf"],
             **{row["path"]: row.get("sha256_lf") for row in components.values()
                if "sha256_lf" in row}}
    for name, expected_sha in bound.items():
        current = ROOT / name
        _require(current.is_file() and _sha(current, lf=True) == expected_sha,
                 f"현재 recovery source/component가 바뀌었다: {name}")
        blob = _git("show", f"{anchor}:{name}").replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        _require(hashlib.sha256(blob).hexdigest() == expected_sha,
                 f"anchor blob이 recovery protocol과 다르다: {name}")
    for name, expected_sha in value["runtime_source_sha256_lf"].items():
        _require((lab / name).is_file() and _sha(lab / name, lf=True) == expected_sha,
                 f"Isaac Lab runtime source가 바뀌었다: {name}")
    return value


def _target_record(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    value = _read(path)
    _require(_sha(path) == TARGET_SHA256 and value.get("schema") == TARGET_SCHEMA
             and value.get("targets") == list(TARGETS)
             and value.get("comparison_policies") == list(POLICIES)
             and value.get("model_responses_seen_for_this_lineage_batch") == 0,
             "model 전 target commitment가 등록 원문과 다르다")
    return value


def _completed_batch(batch_result: Path, manifest: dict, protocol: dict) -> dict[str, Any]:
    result = _read(batch_result)
    rows = result.get("rows")
    _require(result.get("schema") == BATCH.RESULT_SCHEMA and result.get("status") == "complete"
             and result.get("evidence_scope") == "actual_backend"
             and isinstance(rows, list) and len(rows) == 30
             and result.get("summary", {}).get("rows") == 30
             and result.get("correction_executed") is False
             and result.get("training_executed") is False
             and result.get("epoch2_registered") is False,
             "actual first-pass 30행 완료 result가 아니다")
    plan = [{key: row[key] for key in ("order", "case_index", "policy_index", "case_id", "policy")}
            for row in rows]
    _require(plan == protocol["latin_execution_plan"] == manifest["execution_plan"],
             "batch 30행 실행 순서가 등록 plan과 다르다")
    for row in rows:
        result_path = Path(row.get("result_path", "")).resolve()
        _require(result_path.is_file() and _sha(result_path) == row.get("result_sha256")
                 and _read(result_path) == row.get("result"), "batch row 원문/SHA가 다르다")
    return result


def _reference_reports(batch_manifest: dict, source_protocol: dict) -> dict[str, dict[str, str]]:
    index_path = Path(batch_manifest["gate_path"]).resolve().parent / "report-index.json"
    index = _read(index_path)
    jobs = {row["job_id"]: row for row in source_protocol["training_jobs"]}
    out = {}
    for target in TARGETS:
        entry = index.get(target["reference_job"])
        _require(isinstance(entry, dict), "same-seed NONE fixed report index가 없다")
        path = Path(entry["report_path"])
        path = path.resolve() if path.is_absolute() else (ROOT / path).resolve()
        report = _read(path)
        job = jobs[target["reference_job"]]
        C.G.validate_fixed_report(report, {
            "name": report["name"], "checkpoint_file": Path(job.get("checkpoint", path.name)).name
            if "checkpoint" in job else report["checkpoint"]["file"],
            "checkpoint_sha256": entry["checkpoint_sha256"], "action_scale": 0.25,
            "protocol_sha256_lf": _sha(FIXED_PROTOCOL, lf=True),
        })
        _require(_sha(path) == entry["report_sha256"] and report["checkpoint"]["sha256"]
                 == entry["checkpoint_sha256"], "same-seed NONE report SHA/CP가 다르다")
        out[target["case_id"]] = {**_ref(path), "checkpoint_sha256": entry["checkpoint_sha256"]}
    return out


def _materialize_candidate(batch_row: dict, *, case_input: Path, reference_input: Path,
                           gate_path: Path, gate_sha256: str, case_id: str,
                           candidate_path: Path) -> tuple[str, str | None, dict[str, str] | None]:
    """원래 종단 HOLD는 보존하되, confirmed 행의 증거 불일치는 감추지 않는다."""
    result = batch_row.get("result")
    eligible = (isinstance(result, dict) and result.get("schema") == CANDIDATE.E.SCHEMA
                and result.get("status") == "complete" and result.get("end_state") == "confirmed"
                and result.get("hold_reason") is None and result.get("repair_supported") is True
                and result.get("correction_executed") is False
                and result.get("training_executed") is False
                and result.get("epoch2_registered") is False)
    if not eligible:
        reason = result.get("hold_reason") if isinstance(result, dict) else None
        return "HOLD", str(reason or "epoch1_not_confirmed_or_not_supported"), None
    candidate = CANDIDATE.prepare_candidate(
        Path(batch_row["result_path"]), case_input=case_input, reference_input=reference_input,
        gate_path=gate_path, gate_sha256=gate_sha256, case_id=case_id)
    _require(candidate.get("approval_granted") is False
             and candidate.get("execution_allowed") is False
             and candidate.get("correction_executed") is False
             and candidate.get("training_executed") is False,
             "candidate 자체가 승인/실행 권한을 주장한다")
    _write_new(candidate_path, candidate)
    return "candidate", None, _ref(candidate_path)


def prepare_inputs(protocol_path: Path, batch_manifest_path: Path, batch_result_path: Path,
                   target_record_path: Path, out: Path, *, anchor: str, lab: Path,
                   isaac_python: Path) -> dict[str, Any]:
    """actual source18/fixed18/batch30과 선택 4행을 CPU에서 재검증해 봉인한다."""
    out = Path(out).resolve()
    _require(not out.exists(), "prepare-inputs out은 fresh 경로여야 한다")
    protocol = verify_protocol(protocol_path, anchor=anchor, lab=lab)
    _target_record(target_record_path)
    batch_manifest, batch_protocol, collected, gate_rows = BATCH._validate_manifest(batch_manifest_path)
    _require(Path(batch_manifest["protocol_path"]).resolve()
             == (ROOT / protocol["component_protocols"]["comparison_batch"]["path"]).resolve(),
             "batch manifest가 recovery protocol의 comparison component와 다르다")
    batch = _completed_batch(batch_result_path, batch_manifest, batch_protocol)
    source_protocol_path = ROOT / batch_protocol["component_protocols"]["source_collection"]["path"]
    source_protocol = SOURCE.validate_protocol(_read(source_protocol_path))
    report_index = Path(batch_manifest["gate_path"]).resolve().parent / "report-index.json"
    rebuilt_gate = BATCH.GATE.consume_gate(
        batch_manifest["source_result_path"], source_protocol_path, report_index,
        repo=ROOT, bands_path=BANDS)
    _require(rebuilt_gate == _read(Path(batch_manifest["gate_path"])),
             "fixed18 report를 재검증한 gate가 batch 입력과 다르다")
    references = _reference_reports(batch_manifest, source_protocol)
    jobs = {row["job_id"]: row for row in source_protocol["training_jobs"]}
    rows = []
    out.mkdir(parents=True)
    for target in TARGETS:
        gate = gate_rows[target["case_id"]]
        _require(all(gate[key] == target[key] for key in
                     ("case_id", "seed", "arm", "source_job", "reference_job")),
                 "target와 source/gate 결합이 다르다")
        for policy in POLICIES:
            selected = [row for row in batch["rows"]
                        if row["case_id"] == target["case_id"] and row["policy"] == policy]
            _require(len(selected) == 1, "고정 target case/policy batch 행이 유일하지 않다")
            batch_row = selected[0]
            candidate_path = out / "candidates" / f"{target['case_id']}_{policy}.json"
            status, hold_reason, candidate_ref = _materialize_candidate(
                batch_row, case_input=Path(collected[target["source_job"]]["path"]),
                reference_input=Path(collected[target["reference_job"]]["path"]),
                gate_path=Path(batch_manifest["gate_path"]), gate_sha256=batch_manifest["gate_sha256"],
                case_id=target["case_id"], candidate_path=candidate_path)
            source_binding = CANDIDATE.INPUT._checked_collected(
                Path(collected[target["source_job"]]["path"]))[0]["source_binding"]
            reference_binding = CANDIDATE.INPUT._checked_collected(
                Path(collected[target["reference_job"]]["path"]))[0]["source_binding"]
            _require(source_binding["checkpoint_sha256"] == jobs[target["source_job"]].get(
                "checkpoint_sha256", source_binding["checkpoint_sha256"]),
                "source collected checkpoint binding이 다르다")
            _require(reference_binding["checkpoint_sha256"] == references[target["case_id"]]["checkpoint_sha256"],
                     "same-seed NONE collected/fixed checkpoint가 다르다")
            rows.append({"case_id": target["case_id"], "policy": policy, "arm": target["arm"],
                         "seed": target["seed"], "source_job": target["source_job"],
                         "reference_job": target["reference_job"], "status": status,
                         "hold_reason": hold_reason, "candidate": candidate_ref,
                         "epoch_result": _ref(Path(batch_row["result_path"])),
                         "case_input": _ref(Path(collected[target["source_job"]]["path"])),
                         "reference_input": _ref(Path(collected[target["reference_job"]]["path"])),
                         "source_checkpoint_sha256": source_binding["checkpoint_sha256"],
                         "reference_checkpoint_sha256": reference_binding["checkpoint_sha256"],
                         "reference_report": references[target["case_id"]]})
    manifest = {
        "schema": MANIFEST_SCHEMA, "protocol": _ref(Path(protocol_path)), "anchor": anchor,
        "batch_manifest": _ref(Path(batch_manifest_path)), "batch_result": _ref(Path(batch_result_path)),
        "target_record": _ref(Path(target_record_path)), "lab": str(Path(lab).resolve()),
        "isaac_python": str(Path(isaac_python).resolve()), "rows": rows,
        "same_seed_pairing": True, "new_independent_seed": False,
        "source18_verified": True, "fixed18_verified": True, "batch30_verified": True,
        "scope": "first_registered_correction_only", "epoch2_registered": False,
    }
    _write_new(out / "manifest.json", manifest)
    return deepcopy(manifest)


def _owned_ok(value: Mapping[str, Any], watchdog: int) -> bool:
    wall = value.get("wall_s")
    return (value.get("exit_code") == 0 and value.get("natural_exit") is True
            and value.get("cleanup_verified") is True
            and value.get("job_assigned_before_release") is True
            and value.get("timed_out") is False and value.get("error_type") is None
            and isinstance(wall, (int, float)) and not isinstance(wall, bool)
            and 0 <= wall <= watchdog)


def _load_manifest(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = _read(path)
    required = {"schema", "protocol", "anchor", "batch_manifest", "batch_result",
                "target_record", "lab", "isaac_python", "rows", "same_seed_pairing",
                "new_independent_seed", "source18_verified", "fixed18_verified",
                "batch30_verified", "scope", "epoch2_registered"}
    _require(set(manifest) == required and manifest.get("schema") == MANIFEST_SCHEMA,
             "recovery execution manifest 필드/schema가 다르다")
    for name in ("protocol", "batch_manifest", "batch_result", "target_record"):
        ref = manifest[name]
        source = Path(ref.get("path", "")).resolve() if isinstance(ref, dict) else Path()
        _require(source.is_file() and _sha(source) == ref.get("sha256"), f"manifest {name} 원문이 바뀌었다")
    protocol = verify_protocol(Path(manifest["protocol"]["path"]), anchor=manifest["anchor"],
                               lab=Path(manifest["lab"]))
    _target_record(Path(manifest["target_record"]["path"]))
    _require(manifest["scope"] == "first_registered_correction_only"
             and manifest["epoch2_registered"] is False
             and manifest["same_seed_pairing"] is True
             and manifest["new_independent_seed"] is False
             and all(manifest[key] is True for key in
                     ("source18_verified", "fixed18_verified", "batch30_verified")),
             "manifest scope 또는 완료 prerequisite가 다르다")
    rows = manifest.get("rows")
    expected = {(target["case_id"], policy) for target in TARGETS for policy in POLICIES}
    _require(isinstance(rows, list) and len(rows) == 4
             and {(row.get("case_id"), row.get("policy")) for row in rows} == expected,
             "고정 2사례×2정책 입력이 다르다")
    return manifest, protocol


def _training_manifest(protocol_path: Path, protocol: dict, anchor: str, row: dict,
                       stage: Path, *, smoke_mode: bool = False) -> dict[str, Any]:
    if smoke_mode:
        arm, mode, faults, corrections = "NONE", "inspection", [], []
        source_checkpoint = row["reference_checkpoint_sha256"]
        run_name = f"lr_smoke_{row['case_id'].replace('-', '_')}_{row['seed']}"
    else:
        arm, mode = "FIRST", "train"
        faults = ["F1"] if row["arm"] == "F1" else ["F1", "R1"]
        candidate = _read(Path(row["candidate"]["path"]))
        corrections = [candidate["correction"]]
        source_checkpoint = row["source_checkpoint_sha256"]
        run_name = (f"lr_{row['case_id'].replace('-', '_')}_{row['policy']}_e1_"
                    f"s{row['seed']}")
    return TRAIN.validate_manifest({
        "schema": TRAIN.SCHEMA, "mode": mode, "arm": arm, "seed": row["seed"],
        "run_name": run_name, "task": TRAIN.RESIDUAL.TASK, "num_envs": 4096,
        "max_iterations": 3 if smoke_mode else 300,
        "configured_learning_epochs": 5, "configured_mini_batches": 4,
        "runtime_updates_to_observe": 3, "source_commit_sha": anchor,
        "protocol_path": _relative(protocol_path), "protocol_sha256_lf": _sha(protocol_path, lf=True),
        "source_sha256_lf": protocol["source_sha256_lf"],
        "runtime_source_sha256_lf": protocol["runtime_source_sha256_lf"],
        "records_path": str((stage / "observer-records.jsonl").resolve()),
        "readback_path": str((stage / "readback.json").resolve()),
        "source_checkpoint_sha256": source_checkpoint,
        "original_faults": faults, "applied_corrections": corrections,
    })


def _run_training(stage: Path, fixed: dict, manifest: dict, protocol: dict,
                  *, runner: Callable, wait: Callable, watchdog: int) -> tuple[dict, dict]:
    stage.mkdir(parents=True, exist_ok=False)
    wait(stage, max_wait_s=WATCHDOG["gpu_wait"], interval_s=2, consecutive_clear=2)
    manifest_path = stage / "manifest.json"
    _write_new(manifest_path, fixed)
    TRAIN.verify_binding(fixed, lab=Path(manifest["lab"]))
    argv = [manifest["isaac_python"], "-B", str(ROOT / "bench/lineage_recovery_training_v1.py"),
            "--manifest", str(manifest_path), "--lab", manifest["lab"],
            "--out", str(stage / "training")]
    execution = dict(runner(stage / "owned", argv, Path(manifest["lab"]), watchdog))
    _write_new(stage / "owned-result.json", execution)
    _require(_owned_ok(execution, watchdog), "owned 학습 child가 exit0·자연종료·cleanup 경계를 통과하지 못했다")
    fingerprint = TRAIN.manifest_fingerprint(fixed)
    GUARD.validate_completion(stage / "training", fingerprint)
    evidence = _read(stage / "training" / GUARD.EVIDENCE)
    expected_updates = fixed["max_iterations"]
    _require(evidence.get("arm") == fixed["arm"] and evidence.get("seed") == fixed["seed"]
             and evidence.get("updates") == evidence.get("update_count") == expected_updates
             and evidence.get("fresh_initialization") is True
             and evidence.get("checkpoint_loaded") is False
             and evidence.get("resume_requested") is False
             and evidence.get("source_checkpoint_sha256") == fixed["source_checkpoint_sha256"],
             "TRAIN/GUARD evidence가 등록 fresh 학습과 다르다")
    checkpoint = evidence.get("checkpoint")
    path = Path(checkpoint.get("path", "")) if isinstance(checkpoint, dict) else Path()
    _require(path.is_file() and _sha(path) == checkpoint.get("sha256")
             and checkpoint.get("iteration") == expected_updates - 1,
             "최종 checkpoint raw SHA/iteration이 다르다")
    return execution, evidence


def smoke(manifest_path: Path, out: Path, *, runner: Callable,
          wait: Callable = GPU_WAIT.await_clear) -> dict[str, Any]:
    """교정·모델·성공 사례 없이 동일 wrapper/close 경로를 3 update로 확인한다."""
    out = Path(out).resolve()
    _require(not out.exists(), "smoke out은 fresh 경로여야 한다")
    manifest, protocol = _load_manifest(manifest_path)
    row = next(item for item in manifest["rows"] if item["case_id"] == "case-03")
    fixed = _training_manifest(Path(manifest["protocol"]["path"]), protocol,
                               manifest["anchor"], row, out / "smoke", smoke_mode=True)
    try:
        execution, evidence = _run_training(out / "smoke", fixed, manifest, protocol,
                                            runner=runner, wait=wait, watchdog=WATCHDOG["smoke"])
        result = {"schema": SCHEMA + "_smoke_result_v1", "status": "complete",
                  "scope": "registered_infrastructure_only", "updates": 3, "arm": "NONE",
                  "seed": fixed["seed"], "corrections": 0, "model_calls": 0,
                  "success_cases": 0, "checkpoint_provenance_only": True,
                  "not_300_update_control": True, "manifest": _ref(out / "smoke/manifest.json"),
                  "evidence": _ref(out / "smoke/training" / GUARD.EVIDENCE),
                  "owned_result": _ref(out / "smoke/owned-result.json"),
                  "checkpoint": evidence["checkpoint"], "owned_wall_s": execution["wall_s"]}
        _write_new(out / "result.json", result)
        return deepcopy(result)
    except BaseException as exc:
        if out.exists() and not (out / "failure.json").exists():
            _write_new(out / "failure.json", {"status": "invalid", "error_type": type(exc).__name__,
                                               "error": str(exc), "automatic_retry": False})
        raise


def verify_smoke(path: Path, manifest: dict, protocol: dict) -> dict[str, Any]:
    root = Path(path).resolve()
    result = _read(root / "result.json")
    _require(result.get("schema") == SCHEMA + "_smoke_result_v1" and result.get("status") == "complete"
             and result.get("scope") == "registered_infrastructure_only"
             and result.get("updates") == 3 and result.get("arm") == "NONE"
             and result.get("corrections") == result.get("model_calls") == result.get("success_cases") == 0
             and result.get("checkpoint_provenance_only") is True
             and result.get("not_300_update_control") is True,
             "등록 smoke 완료 결과가 다르다")
    row = next(item for item in manifest["rows"] if item["case_id"] == "case-03")
    expected = _training_manifest(Path(manifest["protocol"]["path"]), protocol,
                                  manifest["anchor"], row, root / "smoke", smoke_mode=True)
    actual = _read(Path(result["manifest"]["path"]))
    _require(actual == expected and _sha(Path(result["manifest"]["path"])) == result["manifest"]["sha256"],
             "smoke manifest가 현재 등록 입력과 다르다")
    GUARD.validate_completion(root / "smoke/training", TRAIN.manifest_fingerprint(expected))
    for name in ("evidence", "owned_result"):
        _require(_sha(Path(result[name]["path"])) == result[name]["sha256"], f"smoke {name} 원문이 바뀌었다")
    _require(_owned_ok(_read(Path(result["owned_result"]["path"])), WATCHDOG["smoke"]),
             "smoke owned child 완료 receipt가 다르다")
    return {"path": str(root), "result_sha256": _sha(root / "result.json")}


def _approval(path: Path, rows: list[dict]) -> dict[str, Any]:
    value = _read(path)
    entries = value.get("entries")
    _require(value.get("schema") == APPROVAL_SCHEMA and isinstance(value.get("approver"), str)
             and bool(value["approver"].strip()) and isinstance(entries, list),
             "operator approval schema/approver가 다르다")
    approved = {(row.get("case_id"), row.get("policy")): row for row in entries
                if isinstance(row, dict)}
    candidates = [row for row in rows if row["status"] == "candidate"]
    _require(set(approved) == {(row["case_id"], row["policy"]) for row in candidates},
             "approval entries가 실행 가능한 candidate 집합과 다르다")
    for row in candidates:
        entry = approved[(row["case_id"], row["policy"])]
        _require(set(entry) == {"case_id", "policy", "candidate_sha256", "decision"}
                 and entry["decision"] == "approve"
                 and entry["candidate_sha256"] == row["candidate"]["sha256"],
                 "operator approval이 candidate digest와 다르다")
    return value


def _candidate(row: dict, manifest: dict) -> dict[str, Any]:
    ref = row["candidate"]
    path = Path(ref["path"]).resolve()
    _require(path.is_file() and _sha(path) == ref["sha256"], "candidate 원문이 바뀌었다")
    batch_manifest = _read(Path(manifest["batch_manifest"]["path"]))
    rebuilt = CANDIDATE.prepare_candidate(
        Path(row["epoch_result"]["path"]), case_input=Path(row["case_input"]["path"]),
        reference_input=Path(row["reference_input"]["path"]),
        gate_path=Path(batch_manifest["gate_path"]), gate_sha256=batch_manifest["gate_sha256"],
        case_id=row["case_id"])
    stored = _read(path)
    _require(rebuilt == stored and stored.get("schema") == CANDIDATE.SCHEMA
             and stored.get("status") == "validated_candidate"
             and stored.get("approval_granted") is False
             and stored.get("execution_allowed") is False,
             "epoch1 candidate 재검증 또는 비실행 scope가 다르다")
    _require(stored["case_id"] == row["case_id"] and stored["seed"] == row["seed"]
             and stored["source_arm"] == row["arm"]
             and stored["restart_candidate"]["source_checkpoint_sha256"]
             == row["source_checkpoint_sha256"], "candidate case/seed/source CP 결합이 다르다")
    return stored


def _fixed_evaluate(stage: Path, row: dict, manifest: dict, evidence: dict,
                    *, runner: Callable, wait: Callable) -> tuple[dict, dict]:
    stage.mkdir(parents=True, exist_ok=False)
    wait(stage, max_wait_s=WATCHDOG["gpu_wait"], interval_s=2, consecutive_clear=2)
    checkpoint = evidence["checkpoint"]
    checkpoint_path = Path(checkpoint["path"]).resolve()
    _require(checkpoint_path.is_file() and _sha(checkpoint_path) == checkpoint["sha256"],
             "고정 평가 입력 checkpoint raw SHA가 다르다")
    name = f"lr_{row['case_id'].replace('-', '_')}_{row['policy']}_corrected"
    report_path = stage / "corrected.json"
    jobs = {"action_scale": 0.25, "jobs": [{"name": name, "checkpoint": str(checkpoint_path),
                                               "output": str(report_path)}]}
    _write_new(stage / "jobs.json", jobs)
    argv = [manifest["isaac_python"], "-B", str(ROOT / "evals/fixed_eval.py"),
            "--protocol", str(FIXED_PROTOCOL), "--jobs", str(stage / "jobs.json"), "--headless"]
    execution = dict(runner(stage / "owned", argv, Path(manifest["lab"]), WATCHDOG["eval"]))
    _write_new(stage / "owned-result.json", execution)
    _require(_owned_ok(execution, WATCHDOG["eval"]) and report_path.is_file(),
             "owned fixed evaluation exit0·report·cleanup가 완결되지 않았다")
    report = _read(report_path)
    checked = C.G.validate_fixed_report(report, {
        "name": name, "checkpoint_file": checkpoint_path.name,
        "checkpoint_sha256": checkpoint["sha256"], "action_scale": 0.25,
        "protocol_sha256_lf": _sha(FIXED_PROTOCOL, lf=True),
    })
    _require(checked == report, "fixed report 검증 반환이 원문과 다르다")
    return report, execution


def _recovery_receipt(row: dict, candidate: dict, context: dict, ledger_event: dict,
                      training: dict, collected: dict, corrected_report: Path,
                      reference_report: Path, verdict: dict, evaluation: dict,
                      smoke_ref: dict) -> dict[str, Any]:
    outcome = verdict.get("outcome")
    state = "HOLD" if outcome == "unknown" else outcome
    _require(state in {"recovered", "not_recovered", "HOLD"}, "등록되지 않은 recovery outcome")
    return {
        "schema": RECEIPT_SCHEMA, "status": state, "scope": "first_registered_correction_only",
        "overall_goal_complete": False, "case_id": row["case_id"], "policy": row["policy"],
        "seed": row["seed"], "same_seed_pairing": True, "new_independent_seed": False,
        "source_arm": row["arm"], "candidate": deepcopy(row["candidate"]),
        "candidate_context_digest": _digest(context), "component": candidate["component"],
        "correction": candidate["correction"], "training_mode": "fresh_restart",
        "source_checkpoint_sha256": row["source_checkpoint_sha256"],
        "new_checkpoint": deepcopy(training["checkpoint"]),
        "reference_checkpoint_sha256": row["reference_checkpoint_sha256"],
        "training": deepcopy(training), "collected_recovery": deepcopy(collected),
        "corrected_report": _ref(corrected_report), "reference_report": _ref(reference_report),
        "fixed_protocol": {"path": _relative(FIXED_PROTOCOL), "sha256_lf": _sha(FIXED_PROTOCOL, lf=True)},
        "bands": {"path": _relative(BANDS), "sha256": _sha(BANDS)},
        "recovery": deepcopy(verdict), "ledger_receipt": deepcopy(ledger_event),
        "evaluation_execution": deepcopy(evaluation), "smoke": deepcopy(smoke_ref),
        "second_epoch_registered": False,
        "next_step": "new_protocol_required" if state == "not_recovered" else "stop",
        "automatic_retry": False, "model_calls": 0,
    }


def execute(manifest_path: Path, approval_path: Path, smoke_root: Path, out: Path,
            *, runner: Callable, wait: Callable = GPU_WAIT.await_clear,
            collector: Callable = RECOVERY_INPUTS.collect_recovery,
            clock: Callable = time.monotonic) -> dict[str, Any]:
    """승인된 첫 correction만 train→collect→fixed eval하고 frozen verdict를 봉인한다."""
    out = Path(out).resolve()
    _require(not out.exists(), "execute out은 fresh 경로여야 한다")
    manifest, protocol = _load_manifest(manifest_path)
    smoke_ref = verify_smoke(smoke_root, manifest, protocol)
    approval = _approval(approval_path, manifest["rows"])
    out.mkdir(parents=True)
    _write_new(out / "intent.json", {"schema": SCHEMA + "_intent", "manifest": _ref(manifest_path),
                                      "approval": _ref(approval_path), "smoke": smoke_ref,
                                      "automatic_retry": False, "model_calls": 0})
    start, receipts = clock(), []
    try:
        for row in manifest["rows"]:
            if row["status"] == "HOLD":
                receipt = {"schema": RECEIPT_SCHEMA, "status": "HOLD",
                           "scope": "first_registered_correction_only", "case_id": row["case_id"],
                           "policy": row["policy"], "reason": row["hold_reason"],
                           "sdk_jobs": 0, "approval": False, "model_calls": 0,
                           "overall_goal_complete": False, "second_epoch_registered": False}
                path = out / "receipts" / f"{row['case_id']}_{row['policy']}.json"
                _write_new(path, receipt)
                receipts.append({**receipt, "path": str(path), "sha256": _sha(path)})
                continue
            elapsed = clock() - start
            _require(isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool)
                     and 0 <= elapsed <= WATCHDOG["total"] - (
                         2 * WATCHDOG["gpu_wait"] + WATCHDOG["train"] + WATCHDOG["eval"]),
                     "다음 train+eval의 최악 상한을 포함할 전체 recovery 시간이 남지 않았다")
            candidate = _candidate(row, manifest)
            context = {"candidate_sha256": row["candidate"]["sha256"],
                       "protocol_sha256": manifest["protocol"]["sha256"],
                       "case_id": row["case_id"], "policy": row["policy"], "epoch": 1,
                       "source_checkpoint_sha256": row["source_checkpoint_sha256"]}
            ledger = C.D1.CorrectionLedger(
                out / "ledgers" / f"{row['case_id']}_{row['policy']}.jsonl",
                C.RC.correction_registry())
            entry = C.RC.correction_registry().get(candidate["component"])
            _require(isinstance(entry, dict) and entry.get("correction_id") == candidate["correction"],
                     "candidate component/correction이 frozen registry와 다르다")
            proposal = ledger.propose(
                _digest(context), candidate["component"], _digest(candidate["inspection_evidence"]),
                candidate["correction"], "restart",
                {"case": entry["protocol_case"], "arm": "corrected"}, WATCHDOG["train"] + WATCHDOG["eval"],
                evidence={"context": context, "operator_approval_sha256": _sha(approval_path)})
            request_id = proposal["request_id"]
            ledger.approve(request_id, approval["approver"])
            ledger.consume(request_id, _digest(context))

            stage = out / "runs" / f"{row['case_id']}_{row['policy']}"
            fixed = _training_manifest(Path(manifest["protocol"]["path"]), protocol,
                                       manifest["anchor"], row, stage / "train")
            training_execution, evidence = _run_training(
                stage / "train", fixed, manifest, protocol, runner=runner, wait=wait,
                watchdog=min(WATCHDOG["train"], WATCHDOG["total"] - elapsed))
            collected = collector(
                stage / "train/manifest.json", stage / "train/training",
                stage / "train/owned-result.json", stage / "collected", Path(manifest["isaac_python"]),
                lab=Path(manifest["lab"]))
            RECOVERY_INPUTS.checked_recovery(stage / "collected/collected.json")
            corrected, evaluation_execution = _fixed_evaluate(
                stage / "evaluation", row, manifest, evidence, runner=runner, wait=wait)
            _require(0 <= clock() - start <= WATCHDOG["total"],
                     "recovery train/eval이 전체 실행 시간 상한을 넘었다")
            reference_path = Path(row["reference_report"]["path"])
            _require(_sha(reference_path) == row["reference_report"]["sha256"]
                     and _read(reference_path)["checkpoint"]["sha256"]
                     == row["reference_checkpoint_sha256"],
                     "same-seed NONE report/CP가 실행 전에 바뀌었다")
            verdict = C.D1.recovery(corrected, _read(reference_path), _read(BANDS)["bands"])
            preliminary = {"training_execution": training_execution,
                           "evaluation_execution": evaluation_execution,
                           "corrected_report": _ref(stage / "evaluation/corrected.json"),
                           "reference_report": _ref(reference_path), "verdict": verdict}
            _write_new(stage / "result-bundle.json", preliminary)
            ledger_outcome = verdict["outcome"]
            ledger_event = ledger.receipt(
                request_id, f"{row['case_id']}_{row['policy']}", ledger_outcome,
                _sha(stage / "result-bundle.json"),
                training_execution["wall_s"] + evaluation_execution["wall_s"], 0,
                reason="fixed_evaluation_undetermined" if ledger_outcome == "unknown" else None)
            receipt = _recovery_receipt(
                row, candidate, context, ledger_event, evidence, collected,
                stage / "evaluation/corrected.json", reference_path, verdict,
                evaluation_execution, smoke_ref)
            receipt_path = out / "receipts" / f"{row['case_id']}_{row['policy']}.json"
            _write_new(receipt_path, receipt)
            receipts.append({**receipt, "path": str(receipt_path), "sha256": _sha(receipt_path)})
            if receipt["status"] == "HOLD":
                raise RuntimeError("고정 평가가 unknown이라 HOLD로 봉인하고 다음 학습을 시작하지 않는다")
        result = {"schema": SCHEMA + "_result_v1", "status": "complete",
                  "scope": "first_registered_correction_only", "receipts": receipts,
                  "overall_goal_complete": False, "second_epoch_registered": False,
                  "model_calls": 0, "automatic_retry": False, "wall_s": clock() - start}
        _write_new(out / "result.json", result)
        return deepcopy(result)
    except BaseException as exc:
        if not (out / "failure.json").exists():
            _write_new(out / "failure.json", {"status": "invalid", "error_type": type(exc).__name__,
                                               "error": str(exc), "completed_receipts": receipts,
                                               "automatic_retry": False})
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    public = sub.add_parser("prepare-protocol")
    public.add_argument("--batch-protocol", type=Path, required=True)
    public.add_argument("--fixed-protocol", type=Path, default=FIXED_PROTOCOL)
    public.add_argument("--out", type=Path, required=True)
    prep = sub.add_parser("prepare-inputs")
    for name in ("protocol", "batch-manifest", "batch-result", "target-record", "out", "lab",
                 "isaac-python"):
        prep.add_argument("--" + name, type=Path, required=True)
    prep.add_argument("--anchor", required=True)
    smoke_parser = sub.add_parser("smoke")
    smoke_parser.add_argument("--manifest", type=Path, required=True)
    smoke_parser.add_argument("--out", type=Path, required=True)
    run = sub.add_parser("execute")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--approval", type=Path, required=True)
    run.add_argument("--smoke-root", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare-protocol":
        _write_new(args.out, prepare_protocol(args.batch_protocol, args.fixed_protocol))
    elif args.command == "prepare-inputs":
        prepare_inputs(args.protocol, args.batch_manifest, args.batch_result, args.target_record,
                       args.out, anchor=args.anchor, lab=args.lab, isaac_python=args.isaac_python)
    else:
        import run_runtime_readback_smoke as OWNED
        if args.command == "smoke":
            smoke(args.manifest, args.out, runner=OWNED.run_owned)
        else:
            execute(args.manifest, args.approval, args.smoke_root, args.out,
                    runner=OWNED.run_owned)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
