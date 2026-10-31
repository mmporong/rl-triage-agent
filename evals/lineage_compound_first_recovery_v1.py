"""case-07 재진단의 첫 등록 교정만 승인 뒤 실행하는 fail-closed controller."""
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
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "bench"), str(ROOT / "evals"), str(ROOT / "src")]

import lineage_comparison_batch_v1 as BATCH
import lineage_compound_retry_v1 as RETRY
import lineage_lease_context_v2 as LEASE
import lineage_recovery_execution_v1 as V1
import lineage_recovery_execution_v2 as V2
import lineage_source_batch_v1 as SOURCE

SCHEMA = "lineage_compound_first_recovery_protocol_v1"
MANIFEST_SCHEMA = "lineage_compound_first_recovery_manifest_v1"
RESULT_SCHEMA = "lineage_compound_first_recovery_result_v1"
SMOKE_SCHEMA = "lineage_compound_first_recovery_smoke_v1"
TARGET_CASE = "case-07"
TARGET_SEED = 2307
POLICIES = RETRY.POLICIES
WATCHDOG = {**V1.WATCHDOG, "total": 21600}
HEX40 = re.compile(r"[0-9a-f]{40}\Z")
SOURCE_FILES = tuple(sorted({
    *V2.SOURCE_FILES,
    *LEASE.SOURCE_FILES,
    *RETRY.BOUND_SOURCE_FILES,
    "evals/lineage_compound_first_recovery_v1.py",
    "tests/test_lineage_compound_first_recovery_v1.py",
}))


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(
        Path(path).read_text(encoding="utf-8"),
        parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)),
    )
    _require(isinstance(value, dict), f"JSON 객체가 아니다: {path}")
    return value


def _sha(path: Path, *, lf: bool = False) -> str:
    raw = Path(path).read_bytes()
    if lf:
        raw = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(raw).hexdigest()


def _digest(value: object) -> str:
    raw = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
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
        return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError as error:
        raise ValueError("공개 component는 저장소 내부여야 한다") from error


def _git(*args: str) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(ROOT), *args], capture_output=True, check=False
    )
    if completed.returncode:
        raise ValueError(completed.stderr.decode("utf-8", errors="replace").strip())
    return completed.stdout


def prepare_protocol(retry_protocol: Path, recovery_protocol: Path,
                     fixed_protocol: Path = V1.FIXED_PROTOCOL) -> dict[str, Any]:
    """재진단 판과 기존 recovery V2를 결합한 첫 교정 실행 protocol을 만든다."""
    retry_path, recovery_path = Path(retry_protocol).resolve(), Path(recovery_protocol).resolve()
    retry, recovery = _read(retry_path), _read(recovery_path)
    _require(retry.get("schema") == RETRY.SCHEMA, "compound retry protocol schema가 다르다")
    _require(recovery.get("schema") == V2.SCHEMA, "recovery V2 protocol schema가 다르다")
    _require(Path(fixed_protocol).resolve() == V1.FIXED_PROTOCOL.resolve(),
             "동결 fixed_eval_v1 protocol만 허용한다")
    retry_comparison = retry["component_protocols"]["comparison"]
    recovery_comparison = recovery["component_protocols"]["comparison_batch"]
    _require(retry_comparison == recovery_comparison,
             "retry와 recovery가 같은 comparison protocol을 가리키지 않는다")
    return {
        "schema": SCHEMA,
        "phase": "compound_case07_first_registered_correction_after_fresh_retry",
        "target": {"case_id": TARGET_CASE, "seed": TARGET_SEED, "arm": "COMPOUND",
                   "policies": list(POLICIES)},
        "scope": {"epoch": 1, "first_correction_only": True, "second_epoch": False,
                  "max_distinct_corrections": 2,
                  "followup_requires": "valid_not_recovered_receipt_and_new_protocol"},
        "predictions": {
            "single_component_correction": "not_recovered_due_to_remaining_fault",
            "contrary_recovered": "preserve_recovered_and_stop_without_epoch2",
            "unknown": "HOLD_and_start_no_next_training",
        },
        "training": {"mode": "fresh_restart", "iterations": 300, "seed": TARGET_SEED,
                     "source_checkpoint_role": "provenance_only_never_loaded",
                     "arm": "FIRST", "model_calls": 0},
        "smoke": {"mode": "inspection", "arm": "NONE", "updates": 3,
                  "same_seed": TARGET_SEED, "corrections": 0,
                  "reuse_previous_smoke": False},
        "evaluation": {"conditions": 26, "fixed_protocol": "fixed_eval_v1",
                       "same_seed_none_report_reused": True},
        "stop_rules": {"unconfirmed_or_unsupported": "HOLD_zero_training",
                       "unknown": "HOLD_and_stop_remaining_rows",
                       "failed_job": "preserve_outputs_and_stop_remaining_rows",
                       "automatic_retry": False},
        "watchdogs_s": deepcopy(WATCHDOG),
        "component_protocols": {
            "compound_retry": {"path": _relative(retry_path),
                               "sha256_lf": _sha(retry_path, lf=True)},
            "recovery_v2": {"path": _relative(recovery_path),
                            "sha256_lf": _sha(recovery_path, lf=True)},
            "comparison_batch": deepcopy(retry_comparison),
            "fixed_evaluation": {"path": _relative(V1.FIXED_PROTOCOL),
                                 "sha256_lf": _sha(V1.FIXED_PROTOCOL, lf=True)},
            "bands": {"path": _relative(V1.BANDS), "sha256": _sha(V1.BANDS)},
        },
        "runtime_source_sha256_lf": deepcopy(retry["runtime_source_sha256_lf"]),
        "source_sha256_lf": {name: _sha(ROOT / name, lf=True) for name in SOURCE_FILES},
        "claims": {"api_requests": 0, "new_independent_seed": False,
                   "overall_goal_complete": False, "generalization": False},
    }


def verify_protocol(protocol_path: Path, *, anchor: str, lab: Path) -> dict[str, Any]:
    _require(isinstance(anchor, str) and HEX40.fullmatch(anchor) is not None,
             "execution anchor 형식이 다르다")
    path = Path(protocol_path).resolve()
    value = _read(path)
    components = value.get("component_protocols")
    if not isinstance(components, dict):
        raise TypeError("component protocol 표가 없다")
    expected = prepare_protocol(
        ROOT / components["compound_retry"]["path"],
        ROOT / components["recovery_v2"]["path"],
        ROOT / components["fixed_evaluation"]["path"],
    )
    _require(value == expected, "first recovery protocol 내용 또는 SHA가 다르다")
    RETRY.verify_protocol(ROOT / components["compound_retry"]["path"], anchor=anchor)
    V2.verify_protocol(ROOT / components["recovery_v2"]["path"], anchor=anchor, lab=lab)
    _git("cat-file", "-e", f"{anchor}^{{commit}}")
    _git("merge-base", "--is-ancestor", anchor, "HEAD")
    bound = {_relative(path): _sha(path, lf=True), **value["source_sha256_lf"]}
    for relative, expected_sha in bound.items():
        current = ROOT / relative
        _require(current.is_file() and _sha(current, lf=True) == expected_sha,
                 f"현재 first recovery source가 다르다: {relative}")
        committed = _git("show", f"{anchor}:{relative}")
        committed = committed.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        _require(hashlib.sha256(committed).hexdigest() == expected_sha,
                 f"anchor source blob이 다르다: {relative}")
    _require(len(value["runtime_source_sha256_lf"]) == 9,
             "Isaac runtime source 결합은 9개여야 한다")
    for relative, expected_sha in value["runtime_source_sha256_lf"].items():
        runtime = Path(lab).resolve() / relative
        _require(runtime.is_file() and _sha(runtime, lf=True) == expected_sha,
                 f"Isaac runtime source가 다르다: {relative}")
    return deepcopy(value)


def _eligible_retry_row(row: dict[str, Any]) -> bool:
    result, classification = row.get("result"), row.get("classification")
    return (
        isinstance(result, dict)
        and result.get("status") == "complete"
        and result.get("end_state") == "confirmed"
        and result.get("hold_reason") is None
        and result.get("repair_supported") is True
        and result.get("correction_executed") is False
        and result.get("training_executed") is False
        and result.get("epoch2_registered") is False
        and isinstance(classification, dict)
        and classification.get("correct_confirmation") is True
        and classification.get("compound_partial") is True
    )


def _checked_retry_result(path: Path, retry_manifest: dict[str, Any]) -> dict[str, Any]:
    result = _read(path)
    _require(result.get("schema") == RETRY.RESULT_SCHEMA and result.get("status") == "complete"
             and result.get("evidence_scope") == "actual_backend"
             and result.get("original_results_immutable") is True
             and result.get("correction_executed") is False
             and result.get("training_executed") is False
             and result.get("epoch2_registered") is False,
             "retry actual_backend 완료 결과가 아니다")
    rows = result.get("rows")
    if not isinstance(rows, list):
        raise TypeError("retry 결과 rows가 목록이 아니다")
    _require(len(rows) == 2 and [(row.get("case_id"), row.get("policy")) for row in rows]
             == [(TARGET_CASE, policy) for policy in POLICIES],
             "retry 결과는 case-07 두 정책 순서여야 한다")
    expected_digest = retry_manifest["fresh_public_view"]["digest"]
    for row in rows:
        result_path = Path(str(row.get("result_path", ""))).resolve()
        _require(result_path.is_file() and _sha(result_path) == row.get("result_sha256")
                 and _read(result_path) == row.get("result")
                 and row.get("fresh_view_digest") == expected_digest
                 and row.get("arm") == "COMPOUND",
                 "retry row raw SHA/view/arm 결합이 다르다")
    return result


def _source_inputs(original_manifest_path: Path) -> tuple[dict, dict, dict, dict, dict]:
    manifest, comparison, collected, gate_rows = BATCH._validate_manifest(original_manifest_path)
    source_path = ROOT / comparison["component_protocols"]["source_collection"]["path"]
    source_protocol = SOURCE.validate_protocol(_read(source_path))
    report_index = Path(manifest["gate_path"]).resolve().parent / "report-index.json"
    rebuilt = BATCH.GATE.consume_gate(
        manifest["source_result_path"], source_path, report_index,
        repo=ROOT, bands_path=V1.BANDS,
    )
    _require(rebuilt == _read(Path(manifest["gate_path"])),
             "원래 fixed report gate 재검증이 다르다")
    return manifest, comparison, collected, gate_rows, source_protocol


def _candidate_row(retry_row: dict[str, Any], case: dict[str, Any], collected: dict,
                   source_protocol: dict, original_manifest: dict,
                   references: dict, out: Path) -> dict[str, Any]:
    jobs = {row["job_id"]: row for row in source_protocol["training_jobs"]}
    case_input = Path(collected[case["source_job"]]["path"])
    reference_input = Path(collected[case["reference_job"]]["path"])
    source_binding = V1.CANDIDATE.INPUT._checked_collected(case_input)[0]["source_binding"]
    reference_binding = V1.CANDIDATE.INPUT._checked_collected(reference_input)[0]["source_binding"]
    _require(source_binding["checkpoint_sha256"] == jobs[case["source_job"]].get(
        "checkpoint_sha256", source_binding["checkpoint_sha256"]),
        "COMPOUND source checkpoint 결합이 다르다")
    reference = references[TARGET_CASE]
    _require(reference_binding["checkpoint_sha256"] == reference["checkpoint_sha256"],
             "none-s2307 collected/report checkpoint가 다르다")
    status, reason, candidate_ref = "HOLD", "retry_not_confirmed_or_not_supported", None
    if _eligible_retry_row(retry_row):
        candidate = V1.CANDIDATE.prepare_candidate(
            Path(retry_row["result_path"]), case_input=case_input,
            reference_input=reference_input,
            gate_path=Path(original_manifest["gate_path"]),
            gate_sha256=original_manifest["gate_sha256"], case_id=TARGET_CASE,
        )
        _require(candidate.get("approval_granted") is False
                 and candidate.get("execution_allowed") is False
                 and candidate.get("correction_executed") is False
                 and candidate.get("training_executed") is False
                 and candidate.get("component") in {"reward", V1.C.RC.COUNT},
                 "candidate 승인 경계 또는 등록 reward/optimizer 범주가 다르다")
        path = out / "candidates" / f"{TARGET_CASE}_{retry_row['policy']}.json"
        _write_new(path, candidate)
        status, reason, candidate_ref = "candidate", None, _ref(path)
    else:
        result = retry_row.get("result", {})
        reason = str(result.get("hold_reason") or "retry_not_confirmed_or_not_supported")
    return {
        "order": retry_row["order"], "case_id": TARGET_CASE,
        "policy": retry_row["policy"], "arm": "COMPOUND", "seed": TARGET_SEED,
        "source_job": case["source_job"], "reference_job": case["reference_job"],
        "status": status, "hold_reason": reason, "candidate": candidate_ref,
        "epoch_result": _ref(Path(retry_row["result_path"])),
        "case_input": _ref(case_input), "reference_input": _ref(reference_input),
        "source_checkpoint_sha256": source_binding["checkpoint_sha256"],
        "reference_checkpoint_sha256": reference_binding["checkpoint_sha256"],
        "reference_report": reference,
        "fresh_view_digest": retry_row["fresh_view_digest"],
        "classification": deepcopy(retry_row["classification"]),
    }


def prepare_inputs(out: Path, protocol_path: Path, retry_manifest_path: Path,
                   retry_result_path: Path, original_manifest_path: Path, *, lab: Path,
                   isaac_python: Path, execution_anchor: str) -> dict[str, Any]:
    """원래 18-source/gate와 새 retry 두 행을 재검증해 첫 교정 입력만 만든다."""
    out = Path(out).resolve()
    _require(not out.exists(), "prepare-inputs out은 fresh 경로여야 한다")
    protocol = verify_protocol(protocol_path, anchor=execution_anchor, lab=lab)
    retry_manifest, _ = RETRY._load_manifest(retry_manifest_path)
    _require(Path(retry_manifest["original_manifest"]["path"]).resolve()
             == Path(original_manifest_path).resolve(),
             "retry와 명시 original manifest가 다르다")
    retry_result = _checked_retry_result(retry_result_path, retry_manifest)
    original_manifest, comparison, collected, gate_rows, source_protocol = _source_inputs(
        Path(original_manifest_path)
    )
    registered_comparison = ROOT / protocol["component_protocols"]["comparison_batch"]["path"]
    _require(Path(original_manifest["protocol_path"]).resolve() == registered_comparison.resolve(),
             "원래 manifest와 first recovery comparison component가 다르다")
    case = next(row for row in comparison["cases"] if row["case_id"] == TARGET_CASE)
    _require(case["seed"] == TARGET_SEED and case["arm"] == "COMPOUND"
             and all(gate_rows[TARGET_CASE].get(key) == case[key] for key in
                     ("case_id", "seed", "arm", "source_job", "reference_job")),
             "case-07 source/gate 결합이 다르다")
    references = V1._reference_reports(original_manifest, source_protocol)
    out.mkdir(parents=True)
    rows = [
        _candidate_row(row, case, collected, source_protocol, original_manifest, references, out)
        for row in retry_result["rows"]
    ]
    _require(sum(row["status"] == "candidate" for row in rows) <= 2,
             "candidate 상한 2를 넘었다")
    manifest = {
        "schema": MANIFEST_SCHEMA, "protocol": _ref(Path(protocol_path)),
        "anchor": execution_anchor, "retry_manifest": _ref(Path(retry_manifest_path)),
        "retry_result": _ref(Path(retry_result_path)),
        "batch_manifest": _ref(Path(original_manifest_path)),
        "lab": str(Path(lab).resolve()), "isaac_python": str(Path(isaac_python).resolve()),
        "rows": rows, "same_seed_pairing": True, "new_independent_seed": False,
        "source18_verified": True, "fixed_gate_verified": True,
        "scope": "compound_case07_first_registered_correction_only",
        "epoch2_registered": False, "model_calls": 0,
    }
    _write_new(out / "manifest.json", manifest)
    _write_new(out / "intent.json", {
        "status": "prepared", "candidate_rows": sum(row["status"] == "candidate" for row in rows),
        "training_executed": False, "correction_executed": False,
        "operator_approval_required": True, "model_calls": 0,
    })
    return deepcopy(manifest)


def _load_manifest(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = _read(path)
    required = {"schema", "protocol", "anchor", "retry_manifest", "retry_result",
                "batch_manifest", "lab", "isaac_python", "rows", "same_seed_pairing",
                "new_independent_seed", "source18_verified", "fixed_gate_verified", "scope",
                "epoch2_registered", "model_calls"}
    _require(set(manifest) == required and manifest.get("schema") == MANIFEST_SCHEMA,
             "first recovery manifest 필드/schema가 다르다")
    for name in ("protocol", "retry_manifest", "retry_result", "batch_manifest"):
        ref = manifest[name]
        source = Path(str(ref.get("path", ""))).resolve()
        _require(source.is_file() and _sha(source) == ref.get("sha256"),
                 f"manifest {name} 원문이 바뀌었다")
    protocol = verify_protocol(
        Path(manifest["protocol"]["path"]), anchor=manifest["anchor"], lab=Path(manifest["lab"])
    )
    retry_manifest, _ = RETRY._load_manifest(Path(manifest["retry_manifest"]["path"]))
    retry_result = _checked_retry_result(Path(manifest["retry_result"]["path"]), retry_manifest)
    _require(manifest["scope"] == "compound_case07_first_registered_correction_only"
             and manifest["same_seed_pairing"] is True
             and manifest["new_independent_seed"] is False
             and manifest["epoch2_registered"] is False and manifest["model_calls"] == 0
             and manifest["source18_verified"] is True
             and manifest["fixed_gate_verified"] is True,
             "first recovery scope가 다르다")
    rows = manifest.get("rows")
    _require(isinstance(rows, list) and len(rows) == 2
             and [(row.get("case_id"), row.get("policy")) for row in rows]
             == [(TARGET_CASE, policy) for policy in POLICIES]
             and [row.get("fresh_view_digest") for row in rows]
             == [row["fresh_view_digest"] for row in retry_result["rows"]],
             "first recovery 두 행 또는 retry view 결합이 다르다")
    return manifest, protocol


def _lease_wait(stage: Path, *, max_wait_s: float = 600, interval_s: float = 2,
                consecutive_clear: int = 2) -> dict[str, Any]:
    counter = 0

    def lease_check() -> dict:
        nonlocal counter
        counter += 1
        return LEASE.lease_context(Path(stage) / "lease-reads" / f"{counter:04d}")

    return V1.GPU_WAIT.await_clear(
        stage, max_wait_s=max_wait_s, interval_s=interval_s,
        consecutive_clear=consecutive_clear, lease_check=lease_check,
    )


def smoke(manifest_path: Path, out: Path, *, runner: Callable,
          wait: Callable = _lease_wait) -> dict[str, Any]:
    """same-seed NONE 3-update로 실제 owned wrapper와 close 경계만 확인한다."""
    out = Path(out).resolve()
    _require(not out.exists(), "smoke out은 fresh 경로여야 한다")
    manifest, protocol = _load_manifest(manifest_path)
    row = manifest["rows"][0]
    fixed = V1._training_manifest(
        Path(manifest["protocol"]["path"]), protocol, manifest["anchor"], row,
        out / "smoke", smoke_mode=True,
    )
    try:
        execution, evidence = V1._run_training(
            out / "smoke", fixed, manifest, protocol,
            runner=V2._owned_runner(runner), wait=wait, watchdog=WATCHDOG["smoke"],
        )
        result = {
            "schema": SMOKE_SCHEMA, "status": "complete", "scope": "infrastructure_only",
            "case_id": TARGET_CASE, "seed": TARGET_SEED, "arm": "NONE", "updates": 3,
            "corrections": 0, "model_calls": 0, "not_300_update_control": True,
            "manifest": _ref(out / "smoke/manifest.json"),
            "evidence": _ref(out / "smoke/training" / V1.GUARD.EVIDENCE),
            "owned_result": _ref(out / "smoke/owned-result.json"),
            "checkpoint": evidence["checkpoint"], "owned_wall_s": execution["wall_s"],
        }
        _write_new(out / "result.json", result)
        return deepcopy(result)
    except BaseException as error:
        if out.exists() and not (out / "failure.json").exists():
            _write_new(out / "failure.json", {"status": "invalid",
                "error_type": type(error).__name__, "error": str(error), "automatic_retry": False})
        raise


def verify_smoke(path: Path, manifest: dict[str, Any], protocol: dict[str, Any]) -> dict[str, str]:
    root = Path(path).resolve()
    result = _read(root / "result.json")
    _require(result.get("schema") == SMOKE_SCHEMA and result.get("status") == "complete"
             and result.get("case_id") == TARGET_CASE and result.get("seed") == TARGET_SEED
             and result.get("arm") == "NONE" and result.get("updates") == 3
             and result.get("corrections") == result.get("model_calls") == 0
             and result.get("not_300_update_control") is True,
             "same-seed smoke 결과가 다르다")
    expected = V1._training_manifest(
        Path(manifest["protocol"]["path"]), protocol, manifest["anchor"], manifest["rows"][0],
        root / "smoke", smoke_mode=True,
    )
    actual = _read(Path(result["manifest"]["path"]))
    _require(actual == expected and _sha(Path(result["manifest"]["path"]))
             == result["manifest"]["sha256"], "smoke manifest가 등록 입력과 다르다")
    V1.GUARD.validate_completion(root / "smoke/training", V1.TRAIN.manifest_fingerprint(expected))
    _require(V1._owned_ok(_read(Path(result["owned_result"]["path"])), WATCHDOG["smoke"]),
             "smoke owned 완료 receipt가 다르다")
    return {"path": str(root), "result_sha256": _sha(root / "result.json")}


def _hold_receipt(row: dict[str, Any], reason: str) -> dict[str, Any]:
    return {"schema": V1.RECEIPT_SCHEMA, "status": "HOLD",
            "scope": "first_registered_correction_only", "case_id": row["case_id"],
            "policy": row["policy"], "reason": reason, "sdk_jobs": 0,
            "approval": False, "model_calls": 0, "overall_goal_complete": False,
            "second_epoch_registered": False, "followup_epoch_allowed": False}


def execute(manifest_path: Path, approval_path: Path, smoke_root: Path, out: Path, *,
            runner: Callable, wait: Callable = _lease_wait,
            collector: Callable = V1.RECOVERY_INPUTS.collect_recovery,
            clock: Callable = time.monotonic) -> dict[str, Any]:
    """승인된 candidate만 FIRST fresh300→fixed26으로 실행하고 unknown 뒤 중단한다."""
    out = Path(out).resolve()
    _require(not out.exists(), "execute out은 fresh 경로여야 한다")
    manifest, protocol = _load_manifest(manifest_path)
    smoke_ref = verify_smoke(smoke_root, manifest, protocol)
    approval = V1._approval(approval_path, manifest["rows"])
    out.mkdir(parents=True)
    _write_new(out / "intent.json", {
        "schema": SCHEMA + "_intent", "manifest": _ref(manifest_path),
        "approval": _ref(approval_path), "smoke": smoke_ref,
        "epoch": 1, "first_correction_only": True, "model_calls": 0,
    })
    owned_runner = V2._owned_runner(runner)
    start, receipts = clock(), []
    try:
        for row in manifest["rows"]:
            if row["status"] == "HOLD":
                receipt = _hold_receipt(row, row["hold_reason"])
                receipt_path = out / "receipts" / f"{row['case_id']}_{row['policy']}.json"
                _write_new(receipt_path, receipt)
                receipts.append({**receipt, "path": str(receipt_path), "sha256": _sha(receipt_path)})
                continue
            elapsed = clock() - start
            reserve = 2 * WATCHDOG["gpu_wait"] + WATCHDOG["train"] + WATCHDOG["eval"]
            _require(isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool)
                     and 0 <= elapsed <= WATCHDOG["total"] - reserve,
                     "다음 train+eval 최악 상한을 포함할 전체 시간이 남지 않았다")
            candidate = V1._candidate(row, manifest)
            _require(candidate["component"] in {"reward", V1.C.RC.COUNT}
                     and candidate["correction"] not in [],
                     "등록 reward/optimizer 첫 교정이 아니다")
            context = {"candidate_sha256": row["candidate"]["sha256"],
                       "protocol_sha256": manifest["protocol"]["sha256"],
                       "case_id": TARGET_CASE, "policy": row["policy"], "epoch": 1,
                       "source_checkpoint_sha256": row["source_checkpoint_sha256"],
                       "applied_corrections_before": []}
            ledger = V1.C.D1.CorrectionLedger(
                out / "ledgers" / f"{row['case_id']}_{row['policy']}.jsonl",
                V1.C.RC.correction_registry(),
            )
            entry = V1.C.RC.correction_registry().get(candidate["component"])
            _require(isinstance(entry, dict)
                     and entry.get("correction_id") == candidate["correction"],
                     "candidate와 correction registry가 다르다")
            proposal = ledger.propose(
                _digest(context), candidate["component"],
                _digest(candidate["inspection_evidence"]), candidate["correction"], "restart",
                {"case": entry["protocol_case"], "arm": "corrected"},
                WATCHDOG["train"] + WATCHDOG["eval"],
                evidence={"context": context, "operator_approval_sha256": _sha(approval_path)},
            )
            request_id = proposal["request_id"]
            ledger.approve(request_id, approval["approver"])
            ledger.consume(request_id, _digest(context))
            stage = out / "runs" / f"{row['case_id']}_{row['policy']}"
            fixed = V1._training_manifest(
                Path(manifest["protocol"]["path"]), protocol, manifest["anchor"], row,
                stage / "train",
            )
            execution, evidence = V1._run_training(
                stage / "train", fixed, manifest, protocol, runner=owned_runner, wait=wait,
                watchdog=int(min(WATCHDOG["train"], WATCHDOG["total"] - elapsed)),
            )
            collected = collector(
                stage / "train/manifest.json", stage / "train/training",
                stage / "train/owned-result.json", stage / "collected",
                Path(manifest["isaac_python"]), lab=Path(manifest["lab"]),
            )
            V1.RECOVERY_INPUTS.checked_recovery(stage / "collected/collected.json")
            corrected, eval_execution = V1._fixed_evaluate(
                stage / "evaluation", row, manifest, evidence, runner=owned_runner, wait=wait,
            )
            _require(0 <= clock() - start <= WATCHDOG["total"],
                     "first recovery 전체 실행 상한을 넘었다")
            reference_path = Path(row["reference_report"]["path"])
            _require(_sha(reference_path) == row["reference_report"]["sha256"]
                     and _read(reference_path)["checkpoint"]["sha256"]
                     == row["reference_checkpoint_sha256"],
                     "none-s2307 report/checkpoint가 바뀌었다")
            verdict = V1.C.D1.recovery(corrected, _read(reference_path), _read(V1.BANDS)["bands"])
            bundle = {"training_execution": execution, "evaluation_execution": eval_execution,
                      "corrected_report": _ref(stage / "evaluation/corrected.json"),
                      "reference_report": _ref(reference_path), "verdict": verdict}
            _write_new(stage / "result-bundle.json", bundle)
            ledger_event = ledger.receipt(
                request_id, f"{row['case_id']}_{row['policy']}", verdict["outcome"],
                _sha(stage / "result-bundle.json"), execution["wall_s"] + eval_execution["wall_s"],
                0, reason="fixed_evaluation_undetermined" if verdict["outcome"] == "unknown" else None,
            )
            receipt = V1._recovery_receipt(
                row, candidate, context, ledger_event, evidence, collected,
                stage / "evaluation/corrected.json", reference_path, verdict,
                eval_execution, smoke_ref,
            )
            receipt["followup_epoch_allowed"] = receipt["status"] == "not_recovered"
            receipt["followup_contract"] = {
                "requires_new_protocol": True, "max_distinct_corrections": 2,
                "this_controller_executed_corrections": 1,
            }
            receipt_path = out / "receipts" / f"{row['case_id']}_{row['policy']}.json"
            _write_new(receipt_path, receipt)
            receipts.append({**receipt, "path": str(receipt_path), "sha256": _sha(receipt_path)})
            if receipt["status"] == "HOLD":
                raise RuntimeError("고정 평가가 unknown이라 다음 학습을 시작하지 않는다")
        result = {"schema": RESULT_SCHEMA, "status": "complete",
                  "scope": "compound_case07_first_registered_correction_only",
                  "receipts": receipts, "overall_goal_complete": False,
                  "epoch2_registered": False, "model_calls": 0,
                  "automatic_retry": False, "wall_s": clock() - start}
        _write_new(out / "result.json", result)
        return deepcopy(result)
    except BaseException as error:
        if out.exists() and not (out / "failure.json").exists():
            _write_new(out / "failure.json", {"status": "invalid",
                "error_type": type(error).__name__, "error": str(error),
                "completed_receipts": receipts, "automatic_retry": False})
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    public = sub.add_parser("prepare-protocol")
    public.add_argument("--retry-protocol", type=Path, required=True)
    public.add_argument("--recovery-protocol", type=Path, required=True)
    public.add_argument("--fixed-protocol", type=Path, default=V1.FIXED_PROTOCOL)
    public.add_argument("--out", type=Path, required=True)
    prep = sub.add_parser("prepare-inputs")
    for name in ("out", "protocol", "retry-manifest", "retry-result", "original-manifest",
                 "lab", "isaac-python"):
        prep.add_argument("--" + name, type=Path, required=True)
    prep.add_argument("--execution-anchor", required=True)
    smoke_parser = sub.add_parser("smoke")
    smoke_parser.add_argument("--manifest", type=Path, required=True)
    smoke_parser.add_argument("--out", type=Path, required=True)
    run = sub.add_parser("execute")
    for name in ("manifest", "approval", "smoke-root", "out"):
        run.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare-protocol":
        _write_new(args.out, prepare_protocol(
            args.retry_protocol, args.recovery_protocol, args.fixed_protocol
        ))
    elif args.command == "prepare-inputs":
        prepare_inputs(
            args.out, args.protocol, args.retry_manifest, args.retry_result,
            args.original_manifest, lab=args.lab, isaac_python=args.isaac_python,
            execution_anchor=args.execution_anchor,
        )
    else:
        import run_runtime_readback_smoke as OWNED  # pyright: ignore[reportMissingImports]
        if args.command == "smoke":
            smoke(args.manifest, args.out, runner=OWNED.run_owned)
        else:
            execute(args.manifest, args.approval, args.smoke_root, args.out,
                    runner=OWNED.run_owned)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
