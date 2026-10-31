"""유효한 첫 비회복 뒤 서로 다른 둘째 등록 교정만 fresh BOTH로 실행한다."""
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

import lineage_compound_first_recovery_v1 as FIRST
import lineage_lease_context_v2 as LEASE
import lineage_post_correction_inputs_v1 as POST
import lineage_post_diagnosis_epoch_v1 as E2
import lineage_post_recovery_candidate_v1 as CANDIDATE2
import lineage_recovery_execution_v1 as V1
import lineage_recovery_execution_v2 as V2

SCHEMA = "lineage_compound_second_recovery_protocol_v1"
MANIFEST_SCHEMA = "lineage_compound_second_recovery_manifest_v1"
RESULT_SCHEMA = "lineage_compound_second_recovery_result_v1"
TARGET_CASE = "case-07"
TARGET_SEED = 2307
WATCHDOG = {**V1.WATCHDOG, "total": 21600}
HEX40 = re.compile(r"[0-9a-f]{40}\Z")
SOURCE_FILES = tuple(sorted({
    *FIRST.SOURCE_FILES,
    *E2.SOURCE_FILES,
    "evals/lineage_post_recovery_candidate_v1.py",
    "tests/test_lineage_post_recovery_candidate_v1.py",
    "evals/lineage_compound_second_recovery_v1.py",
    "tests/test_lineage_compound_second_recovery_v1.py",
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
    result = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, check=False)
    if result.returncode:
        raise ValueError(result.stderr.decode("utf-8", errors="replace").strip())
    return result.stdout


def prepare_protocol(post_diagnosis_protocol: Path, recovery_protocol: Path,
                     first_recovery_protocol: Path) -> dict[str, Any]:
    """epoch2 actual 진단과 첫 recovery 실행 계약을 둘째 교정 판에 결합한다."""
    post_path = Path(post_diagnosis_protocol).resolve()
    recovery_path = Path(recovery_protocol).resolve()
    first_path = Path(first_recovery_protocol).resolve()
    post, recovery, first = _read(post_path), _read(recovery_path), _read(first_path)
    _require(post.get("schema") == E2.PROTOCOL_SCHEMA, "post diagnosis protocol schema가 다르다")
    _require(recovery.get("schema") == V2.SCHEMA, "recovery V2 protocol schema가 다르다")
    _require(first.get("schema") == FIRST.SCHEMA, "first recovery protocol schema가 다르다")
    _require(first["component_protocols"]["recovery_v2"] == {
        "path": _relative(recovery_path), "sha256_lf": _sha(recovery_path, lf=True)},
        "first recovery와 recovery V2 component가 다르다")
    return {
        "schema": SCHEMA,
        "phase": "compound_case07_second_distinct_registered_correction",
        "target": {"case_id": TARGET_CASE, "seed": TARGET_SEED, "arm": "COMPOUND"},
        "scope": {"epoch": 2, "second_correction_only": True,
                  "max_distinct_corrections": 2, "third_epoch": False},
        "predictions": {
            "two_registered_components": "same_seed_fresh300_expected_recovered",
            "unknown": "HOLD_and_start_no_next_training",
            "prior_recovered": "reject_before_candidate_or_training",
        },
        "training": {"mode": "fresh_restart", "arm": "BOTH", "iterations": 300,
                     "seed": TARGET_SEED, "source_checkpoint_role": "provenance_only_never_loaded",
                     "fault_strength_change": False, "model_calls": 0},
        "smoke": {"strategy": "reuse_and_reverify_first_controller_smoke",
                  "same_runtime_and_training_path_required": True,
                  "new_smoke_updates": 0},
        "evaluation": {"conditions": 26, "same_seed_none_report_reused": True},
        "stop_rules": {"unconfirmed_or_unsupported": "HOLD_zero_training",
                       "unknown": "HOLD_and_stop", "failed_job": "preserve_and_stop",
                       "automatic_retry": False},
        "watchdogs_s": deepcopy(WATCHDOG),
        "component_protocols": {
            "post_diagnosis": {"path": _relative(post_path), "sha256_lf": _sha(post_path, lf=True)},
            "recovery_v2": {"path": _relative(recovery_path),
                            "sha256_lf": _sha(recovery_path, lf=True)},
            "first_recovery": {"path": _relative(first_path),
                               "sha256_lf": _sha(first_path, lf=True)},
            "fixed_evaluation": deepcopy(first["component_protocols"]["fixed_evaluation"]),
            "bands": deepcopy(first["component_protocols"]["bands"]),
        },
        "runtime_source_sha256_lf": deepcopy(first["runtime_source_sha256_lf"]),
        "source_sha256_lf": {name: _sha(ROOT / name, lf=True) for name in SOURCE_FILES},
        "claims": {"api_requests": 0, "new_independent_seed": False,
                   "overall_goal_complete": False, "generalization": False},
    }


def verify_protocol(protocol_path: Path, *, anchor: str, lab: Path) -> dict[str, Any]:
    _require(isinstance(anchor, str) and HEX40.fullmatch(anchor) is not None,
             "execution anchor 형식이 다르다")
    path, lab = Path(protocol_path).resolve(), Path(lab).resolve()
    value = _read(path)
    components = value.get("component_protocols")
    if not isinstance(components, dict):
        raise TypeError("component protocol 표가 없다")
    expected = prepare_protocol(
        ROOT / components["post_diagnosis"]["path"],
        ROOT / components["recovery_v2"]["path"],
        ROOT / components["first_recovery"]["path"],
    )
    _require(value == expected, "second recovery protocol 내용 또는 SHA가 다르다")
    E2.verify_protocol(ROOT / components["post_diagnosis"]["path"], lab=lab, anchor=anchor)
    V2.verify_protocol(ROOT / components["recovery_v2"]["path"], anchor=anchor, lab=lab)
    FIRST.verify_protocol(ROOT / components["first_recovery"]["path"], anchor=anchor, lab=lab)
    _git("cat-file", "-e", f"{anchor}^{{commit}}")
    _git("merge-base", "--is-ancestor", anchor, "HEAD")
    bound = {_relative(path): _sha(path, lf=True), **value["source_sha256_lf"]}
    for relative, expected_sha in bound.items():
        current = ROOT / relative
        _require(current.is_file() and _sha(current, lf=True) == expected_sha,
                 f"현재 second recovery source가 다르다: {relative}")
        committed = _git("show", f"{anchor}:{relative}")
        committed = committed.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        _require(hashlib.sha256(committed).hexdigest() == expected_sha,
                 f"anchor source blob이 다르다: {relative}")
    _require(len(value["runtime_source_sha256_lf"]) == 9,
             "Isaac runtime source 결합은 9개여야 한다")
    return deepcopy(value)


def _eligible_epoch2(result: dict[str, Any]) -> bool:
    return (result.get("schema") == E2.SCHEMA and result.get("status") == "complete"
            and result.get("evidence_scope") == "actual_backend" and result.get("epoch") == 2
            and result.get("epoch2_registered") is True and result.get("end_state") == "confirmed"
            and result.get("hold_reason") is None and result.get("repair_supported") is True
            and result.get("correction_executed") is False
            and result.get("training_executed") is False)


def _checked_first_inputs(post_input: Path, first_manifest_path: Path,
                          first_smoke_root: Path) -> tuple[dict, dict, dict, dict, dict]:
    post, _telemetry = POST.checked_observation(post_input)
    sources = post["private_metadata"]["sources"]
    receipt_path = Path(sources["repair_receipt"]).resolve()
    ledger_path = Path(sources["correction_ledger"]).resolve()
    receipt = _read(receipt_path)
    _require(receipt.get("status") == "not_recovered"
             and receipt.get("followup_epoch_allowed") is True
             and receipt.get("second_epoch_registered") is False,
             "둘째 교정은 유효한 첫 not_recovered receipt만 허용한다")
    ledger = POST._validate_ledger(ledger_path, receipt)
    _require([row.get("event") for row in ledger] == list(POST.LEDGER_EVENTS),
             "첫 교정 원장은 propose→approve→consume→receipt 네 사건이어야 한다")
    first_manifest, first_protocol = FIRST._load_manifest(first_manifest_path)
    smoke_ref = FIRST.verify_smoke(first_smoke_root, first_manifest, first_protocol)
    _require(first_protocol["runtime_source_sha256_lf"]
             == _read(Path(first_manifest["protocol"]["path"]))["runtime_source_sha256_lf"],
             "첫 smoke runtime/training protocol 결합이 다르다")
    return post, receipt, {"path": str(ledger_path), "sha256": _sha(ledger_path)}, first_manifest, smoke_ref


def _candidate(epoch_result_path: Path, *, post_input: Path, reference_input: Path,
               gate_path: Path, gate_sha256: str) -> tuple[str, str | None, dict | None, dict]:
    result = _read(epoch_result_path)
    if not _eligible_epoch2(result):
        return "HOLD", str(result.get("hold_reason") or "epoch2_not_confirmed_or_not_supported"), None, result
    candidate = CANDIDATE2.prepare_candidate(
        epoch_result_path, post_input=post_input, reference_input=reference_input,
        gate_path=gate_path, gate_sha256=gate_sha256, case_id=TARGET_CASE,
    )
    before, after = candidate["applied_corrections_before"], candidate["applied_corrections_after"]
    _require(len(before) == 1 and len(after) == 2 and len(set(after)) == 2
             and after[0] == before[0] and after[1] == candidate["correction"]
             and candidate.get("approval_granted") is False
             and candidate.get("execution_allowed") is False,
             "둘째 candidate는 서로 다른 등록 교정 두 개와 비승인 상태여야 한다")
    return "candidate", None, candidate, result


def prepare_inputs(out: Path, protocol_path: Path, post_input: Path,
                   epoch2_result_path: Path, reference_input: Path,
                   first_manifest_path: Path, first_smoke_root: Path, *, gate_path: Path,
                   gate_sha256: str, lab: Path, isaac_python: Path,
                   execution_anchor: str, diagnosis_anchor: str) -> dict[str, Any]:
    """첫 비회복+typed post+actual epoch2를 검증해 둘째 candidate 하나를 봉인한다."""
    out = Path(out).resolve()
    _require(not out.exists(), "prepare-inputs out은 fresh 경로여야 한다")
    protocol = verify_protocol(protocol_path, anchor=execution_anchor, lab=lab)
    E2.verify_protocol(
        ROOT / protocol["component_protocols"]["post_diagnosis"]["path"],
        lab=Path(lab), anchor=diagnosis_anchor,
    )
    post, receipt, first_ledger, first_manifest, smoke_ref = _checked_first_inputs(
        Path(post_input), Path(first_manifest_path), Path(first_smoke_root)
    )
    _require(first_manifest["anchor"] != diagnosis_anchor or diagnosis_anchor != execution_anchor,
             "first/diagnosis/execution anchor 구분이 기록되지 않았다")
    status, reason, candidate, epoch2 = _candidate(
        Path(epoch2_result_path), post_input=Path(post_input), reference_input=Path(reference_input),
        gate_path=Path(gate_path), gate_sha256=gate_sha256,
    )
    out.mkdir(parents=True)
    candidate_ref = None
    if candidate is not None:
        candidate_path = out / "candidate.json"
        _write_new(candidate_path, candidate)
        candidate_ref = _ref(candidate_path)
    row = {
        "case_id": TARGET_CASE, "policy": epoch2.get("policy"), "arm": "COMPOUND",
        "seed": TARGET_SEED, "status": status, "hold_reason": reason,
        "candidate": candidate_ref, "epoch_result": _ref(Path(epoch2_result_path)),
        "post_input": _ref(Path(post_input)), "reference_input": _ref(Path(reference_input)),
        "prior_receipt": _ref(Path(post["private_metadata"]["sources"]["repair_receipt"])),
        "first_ledger": first_ledger,
        "source_checkpoint_sha256": post["source_binding"]["checkpoint_sha256"],
        "reference_checkpoint_sha256": receipt["reference_checkpoint_sha256"],
        "reference_report": deepcopy(receipt["reference_report"]),
        "applied_corrections_before": deepcopy(
            [] if candidate is None else candidate["applied_corrections_before"]),
        "applied_corrections_after": deepcopy(
            [] if candidate is None else candidate["applied_corrections_after"]),
    }
    manifest = {
        "schema": MANIFEST_SCHEMA, "protocol": _ref(Path(protocol_path)),
        "execution_anchor": execution_anchor, "diagnosis_anchor": diagnosis_anchor,
        "first_anchor": first_manifest["anchor"], "lab": str(Path(lab).resolve()),
        "isaac_python": str(Path(isaac_python).resolve()), "gate": _ref(Path(gate_path)),
        "gate_sha256": gate_sha256, "first_manifest": _ref(Path(first_manifest_path)),
        "first_smoke": smoke_ref, "row": row, "same_seed_pairing": True,
        "new_independent_seed": False, "scope": "second_distinct_registered_correction_only",
        "third_epoch_registered": False, "model_calls": 0,
    }
    _write_new(out / "manifest.json", manifest)
    _write_new(out / "intent.json", {"status": "prepared", "candidate": status == "candidate",
        "operator_approval_required": True, "training_executed": False,
        "correction_executed": False, "model_calls": 0})
    return deepcopy(manifest)


def _load_manifest(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = _read(path)
    required = {"schema", "protocol", "execution_anchor", "diagnosis_anchor", "first_anchor",
                "lab", "isaac_python", "gate", "gate_sha256", "first_manifest", "first_smoke",
                "row", "same_seed_pairing", "new_independent_seed", "scope",
                "third_epoch_registered", "model_calls"}
    _require(set(manifest) == required and manifest.get("schema") == MANIFEST_SCHEMA,
             "second recovery manifest 필드/schema가 다르다")
    for name in ("protocol", "gate", "first_manifest"):
        source = Path(manifest[name].get("path", "")).resolve()
        _require(source.is_file() and _sha(source) == manifest[name].get("sha256"),
                 f"manifest {name} 원문이 바뀌었다")
    protocol = verify_protocol(
        Path(manifest["protocol"]["path"]), anchor=manifest["execution_anchor"],
        lab=Path(manifest["lab"]),
    )
    first_manifest, first_protocol = FIRST._load_manifest(Path(manifest["first_manifest"]["path"]))
    _require(first_manifest["anchor"] == manifest["first_anchor"]
             and FIRST.verify_smoke(manifest["first_smoke"]["path"], first_manifest, first_protocol)
             == manifest["first_smoke"], "first smoke 재검증이 다르다")
    row = manifest.get("row")
    _require(isinstance(row, dict) and row.get("case_id") == TARGET_CASE
             and row.get("seed") == TARGET_SEED and row.get("arm") == "COMPOUND"
             and row.get("status") in {"candidate", "HOLD"}, "second recovery row가 다르다")
    _require(manifest["scope"] == "second_distinct_registered_correction_only"
             and manifest["same_seed_pairing"] is True
             and manifest["new_independent_seed"] is False
             and manifest["third_epoch_registered"] is False and manifest["model_calls"] == 0,
             "second recovery scope가 다르다")
    return manifest, protocol


def _training_manifest(protocol_path: Path, protocol: dict[str, Any], anchor: str,
                       row: dict[str, Any], stage: Path) -> dict[str, Any]:
    candidate = _read(Path(row["candidate"]["path"]))
    corrections = candidate["applied_corrections_after"]
    _require(candidate["applied_corrections_before"] == row["applied_corrections_before"]
             and corrections == row["applied_corrections_after"]
             and len(corrections) == 2 and len(set(corrections)) == 2,
             "BOTH 학습의 두 교정 계보가 다르다")
    remaining = list(V1.TRAIN.RESIDUAL.FAULTS.remaining_faults(["F1", "R1"], corrections))
    _require(remaining == [], "BOTH manifest에 residual fault가 남는다")
    return V1.TRAIN.validate_manifest({
        "schema": V1.TRAIN.SCHEMA, "mode": "train", "arm": "BOTH", "seed": TARGET_SEED,
        "run_name": f"lr_case_07_{row['policy']}_e2_both_s2307",
        "task": V1.TRAIN.RESIDUAL.TASK, "num_envs": 4096, "max_iterations": 300,
        "configured_learning_epochs": 5, "configured_mini_batches": 4,
        "runtime_updates_to_observe": 3, "source_commit_sha": anchor,
        "protocol_path": _relative(protocol_path),
        "protocol_sha256_lf": _sha(protocol_path, lf=True),
        "source_sha256_lf": protocol["source_sha256_lf"],
        "runtime_source_sha256_lf": protocol["runtime_source_sha256_lf"],
        "records_path": str((stage / "observer-records.jsonl").resolve()),
        "readback_path": str((stage / "readback.json").resolve()),
        "source_checkpoint_sha256": row["source_checkpoint_sha256"],
        "original_faults": ["F1", "R1"], "applied_corrections": corrections,
    })


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


def _checked_candidate(row: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    ref = row["candidate"]
    path = Path(ref["path"]).resolve()
    _require(path.is_file() and _sha(path) == ref["sha256"], "둘째 candidate 원문이 바뀌었다")
    rebuilt = CANDIDATE2.prepare_candidate(
        Path(row["epoch_result"]["path"]), post_input=Path(row["post_input"]["path"]),
        reference_input=Path(row["reference_input"]["path"]),
        gate_path=Path(manifest["gate"]["path"]), gate_sha256=manifest["gate_sha256"],
        case_id=TARGET_CASE,
    )
    stored = _read(path)
    _require(rebuilt == stored and stored.get("status") == "validated_candidate"
             and stored.get("approval_granted") is False
             and stored.get("execution_allowed") is False
             and stored["applied_corrections_before"] == row["applied_corrections_before"]
             and stored["applied_corrections_after"] == row["applied_corrections_after"],
             "둘째 candidate 재검증 또는 비승인 scope가 다르다")
    return stored


def execute(manifest_path: Path, approval_path: Path, out: Path, *, runner: Callable,
            wait: Callable = _lease_wait,
            collector: Callable = V1.RECOVERY_INPUTS.collect_recovery,
            clock: Callable = time.monotonic) -> dict[str, Any]:
    """승인된 서로 다른 둘째 교정만 BOTH fresh300→fixed26으로 실행한다."""
    out = Path(out).resolve()
    _require(not out.exists(), "execute out은 fresh 경로여야 한다")
    manifest, protocol = _load_manifest(manifest_path)
    row = manifest["row"]
    approval = V1._approval(approval_path, [row])
    out.mkdir(parents=True)
    _write_new(out / "intent.json", {"schema": SCHEMA + "_intent",
        "manifest": _ref(manifest_path), "approval": _ref(approval_path),
        "first_smoke": deepcopy(manifest["first_smoke"]), "model_calls": 0})
    receipts: list[dict[str, Any]] = []
    start = clock()
    try:
        if row["status"] == "HOLD":
            receipt = {"schema": V1.RECEIPT_SCHEMA, "status": "HOLD",
                "scope": "second_registered_correction_only", "case_id": TARGET_CASE,
                "policy": row["policy"], "reason": row["hold_reason"], "sdk_jobs": 0,
                "approval": False, "model_calls": 0, "overall_goal_complete": False,
                "third_epoch_registered": False}
            path = out / "receipt.json"
            _write_new(path, receipt)
            receipts.append({**receipt, "path": str(path), "sha256": _sha(path)})
        else:
            elapsed = clock() - start
            reserve = 2 * WATCHDOG["gpu_wait"] + WATCHDOG["train"] + WATCHDOG["eval"]
            _require(isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool)
                     and 0 <= elapsed <= WATCHDOG["total"] - reserve,
                     "BOTH train+eval 최악 상한을 포함할 시간이 남지 않았다")
            candidate = _checked_candidate(row, manifest)
            context = {"candidate_sha256": row["candidate"]["sha256"],
                       "protocol_sha256": manifest["protocol"]["sha256"],
                       "case_id": TARGET_CASE, "policy": row["policy"], "epoch": 2,
                       "source_checkpoint_sha256": row["source_checkpoint_sha256"],
                       "prior_receipt_sha256": row["prior_receipt"]["sha256"],
                       "applied_corrections_before": row["applied_corrections_before"]}
            ledger = V1.C.D1.CorrectionLedger(
                out / "correction.ledger.jsonl", V1.C.RC.correction_registry())
            entry = V1.C.RC.correction_registry().get(candidate["component"])
            _require(isinstance(entry, dict)
                     and entry.get("correction_id") == candidate["correction"]
                     and candidate["correction"] not in candidate["applied_corrections_before"],
                     "둘째 component/correction이 registry 또는 첫 교정과 충돌한다")
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
            stage = out / "run"
            fixed = _training_manifest(
                Path(manifest["protocol"]["path"]), protocol, manifest["execution_anchor"],
                row, stage / "train",
            )
            owned_runner = V2._owned_runner(runner)
            training_execution, evidence = V1._run_training(
                stage / "train", fixed, manifest, protocol, runner=owned_runner, wait=wait,
                watchdog=int(min(WATCHDOG["train"], WATCHDOG["total"] - elapsed)),
            )
            _require(evidence.get("remaining_faults") == []
                     and evidence.get("applied_corrections") == row["applied_corrections_after"],
                     "BOTH 실제 readback에 residual fault가 남거나 교정 순서가 다르다")
            collected = collector(
                stage / "train/manifest.json", stage / "train/training",
                stage / "train/owned-result.json", stage / "collected",
                Path(manifest["isaac_python"]), lab=Path(manifest["lab"]),
            )
            V1.RECOVERY_INPUTS.checked_recovery(stage / "collected/collected.json")
            corrected, eval_execution = V1._fixed_evaluate(
                stage / "evaluation", row, manifest, evidence, runner=owned_runner, wait=wait)
            reference_path = Path(row["reference_report"]["path"])
            _require(_sha(reference_path) == row["reference_report"]["sha256"]
                     and _read(reference_path)["checkpoint"]["sha256"]
                     == row["reference_checkpoint_sha256"],
                     "same-seed NONE report/checkpoint가 바뀌었다")
            verdict = V1.C.D1.recovery(corrected, _read(reference_path), _read(V1.BANDS)["bands"])
            bundle = {"training_execution": training_execution,
                      "evaluation_execution": eval_execution,
                      "corrected_report": _ref(stage / "evaluation/corrected.json"),
                      "reference_report": _ref(reference_path), "verdict": verdict}
            _write_new(stage / "result-bundle.json", bundle)
            ledger_event = ledger.receipt(
                request_id, f"{TARGET_CASE}_{row['policy']}_epoch2", verdict["outcome"],
                _sha(stage / "result-bundle.json"),
                training_execution["wall_s"] + eval_execution["wall_s"], 0,
                reason="fixed_evaluation_undetermined" if verdict["outcome"] == "unknown" else None)
            receipt = V1._recovery_receipt(
                row, candidate, context, ledger_event, evidence, collected,
                stage / "evaluation/corrected.json", reference_path, verdict,
                eval_execution, manifest["first_smoke"])
            receipt.update({
                "scope": "second_registered_correction_only", "epoch": 2,
                "prior_not_recovered_receipt": deepcopy(row["prior_receipt"]),
                "prior_correction_ledger": deepcopy(row["first_ledger"]),
                "applied_corrections_before": deepcopy(row["applied_corrections_before"]),
                "applied_corrections_after": deepcopy(row["applied_corrections_after"]),
                "remaining_faults": deepcopy(evidence["remaining_faults"]),
                "third_epoch_registered": False, "followup_epoch_allowed": False,
                "next_step": "stop",
            })
            path = out / "receipt.json"
            _write_new(path, receipt)
            receipts.append({**receipt, "path": str(path), "sha256": _sha(path)})
            if receipt["status"] == "HOLD":
                raise RuntimeError("BOTH 고정 평가가 unknown이라 HOLD로 봉인한다")
        result = {"schema": RESULT_SCHEMA, "status": "complete",
                  "scope": "second_distinct_registered_correction_only",
                  "receipts": receipts, "overall_goal_complete": False,
                  "third_epoch_registered": False, "model_calls": 0,
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
    commands = parser.add_subparsers(dest="command", required=True)
    public = commands.add_parser("prepare-protocol")
    public.add_argument("--post-diagnosis-protocol", type=Path, required=True)
    public.add_argument("--recovery-protocol", type=Path, required=True)
    public.add_argument("--first-recovery-protocol", type=Path, required=True)
    public.add_argument("--out", type=Path, required=True)
    prep = commands.add_parser("prepare-inputs")
    for name in ("out", "protocol", "post-input", "epoch2-result", "reference-input",
                 "first-manifest", "first-smoke-root", "gate", "lab", "isaac-python"):
        prep.add_argument("--" + name, type=Path, required=True)
    prep.add_argument("--gate-sha256", required=True)
    prep.add_argument("--execution-anchor", required=True)
    prep.add_argument("--diagnosis-anchor", required=True)
    run = commands.add_parser("execute")
    for name in ("manifest", "approval", "out"):
        run.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare-protocol":
        _write_new(args.out, prepare_protocol(
            args.post_diagnosis_protocol, args.recovery_protocol,
            args.first_recovery_protocol))
    elif args.command == "prepare-inputs":
        prepare_inputs(
            args.out, args.protocol, args.post_input, args.epoch2_result,
            args.reference_input, args.first_manifest, args.first_smoke_root,
            gate_path=args.gate, gate_sha256=args.gate_sha256, lab=args.lab,
            isaac_python=args.isaac_python, execution_anchor=args.execution_anchor,
            diagnosis_anchor=args.diagnosis_anchor)
    else:
        import run_runtime_readback_smoke as OWNED  # pyright: ignore[reportMissingImports]
        execute(args.manifest, args.approval, args.out, runner=OWNED.run_owned)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
