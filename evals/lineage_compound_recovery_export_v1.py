"""복합 결함 재진단과 최대 두 번의 등록 교정 결과를 공개 artifact로 내보낸다."""
from __future__ import annotations

import argparse
import hashlib
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "bench"), str(ROOT / "evals"), str(ROOT / "src")]

import lineage_compound_first_recovery_v1 as FIRST
import lineage_compound_retry_v1 as RETRY
import lineage_compound_second_recovery_v1 as SECOND
import lineage_compound_second_recovery_v2 as SECOND2
import lineage_post_diagnosis_epoch_v2 as EPOCH2_V2
import lineage_post_diagnosis_epoch_v3 as EPOCH2_V3
import lineage_post_parent_binding_v1 as PARENT
import lineage_recovery_export_v1 as BASE

SCHEMA = "lineage_compound_recovery_public_export_v1"
RECEIPT_SCHEMA = "lineage_compound_recovery_public_receipt_v1"
MANIFEST_SCHEMA = "lineage_compound_recovery_public_artifact_manifest_v1"
POLICIES = ("registry_next", "model_adaptive")


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def _read(path: Path) -> dict[str, Any]:
    return BASE._read(Path(path))


def _sha(path: Path) -> str:
    return BASE._sha(Path(path))


def _ref_matches(ref: dict[str, Any], path: Path) -> bool:
    path = Path(path).resolve()
    return (isinstance(ref, dict) and set(ref) == {"path", "sha256"}
            and Path(str(ref["path"])).resolve() == path
            and path.is_file() and ref["sha256"] == _sha(path))


def _embedded_receipt(row: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    return BASE._raw_receipt(row)


def _retry_summary(manifest_path: Path, result_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest, _protocol = RETRY._load_manifest(Path(manifest_path))
    result = _read(result_path)
    rows = result.get("rows")
    _require(result.get("schema") == RETRY.RESULT_SCHEMA
             and result.get("status") == "complete"
             and result.get("evidence_scope") == "actual_backend"
             and result.get("measurement_scope")
             == "development_after_observed_infrastructure_failure"
             and result.get("original_results_immutable") is True
             and result.get("correction_executed") is False
             and result.get("training_executed") is False
             and result.get("epoch2_registered") is False
             and result.get("automatic_retry") is False
             and isinstance(rows, list) and len(rows) == 2,
             "완료 actual 재진단 2행 결과가 아니다")
    if not isinstance(rows, list):
        raise TypeError("재진단 rows가 목록이 아니다")
    execution_plan = manifest.get("execution_plan")
    if not isinstance(execution_plan, list):
        raise TypeError("재진단 실행 계획이 목록이 아니다")
    _require([row.get("policy") for row in rows] == list(POLICIES),
             "재진단 정책 순서가 다르다")
    public_rows: list[dict[str, Any]] = []
    for planned, row in zip(execution_plan, rows, strict=True):
        source = Path(str(row.get("result_path", ""))).resolve()
        embedded = row.get("result")
        _require(row.get("case_id") == "case-07" and row.get("arm") == "COMPOUND"
                 and row.get("policy") == planned["policy"]
                 and source.is_file() and _sha(source) == row.get("result_sha256")
                 and _read(source) == embedded,
                 "재진단 행의 계획·원문·SHA 결합이 다르다")
        assert isinstance(embedded, dict)
        tokens = embedded.get("tokens") or {
            "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
            "known_subtotal": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            "unknown_cost_requests": {"input_tokens": [], "output_tokens": [],
                                      "total_tokens": []},
        }
        wall = embedded.get("wall") or {
            "known_model_wall_subtotal_s": 0, "model_wall_s": 0,
            "probe_child_known_s": sum(
                float(role.get("execution", {}).get("wall_s", 0))
                for step in embedded.get("steps", [])
                for role in step.get("paired_receipts", {}).values()),
            "probe_parent_known_s": sum(
                float(step.get("ledger_receipt", {}).get("gpu_s", 0))
                for step in embedded.get("steps", [])),
            "unknown_model_cost_requests": [],
        }
        _require(embedded.get("probe_order") == [step.get("probe") for step in embedded.get("steps", [])]
                 and embedded.get("decision_count") == len(embedded.get("steps", []))
                 and embedded.get("requests") in {0, 2}
                 and isinstance(tokens, dict) and isinstance(wall, dict),
                 "재진단 probe/decision/request 수치가 다르다")
        classification = row.get("classification")
        _require(isinstance(classification, dict), "재진단 classification이 없다")
        public_rows.append({
            "policy": row["policy"], "probe_order": deepcopy(embedded["probe_order"]),
            "end_state": embedded.get("end_state"), "hold_reason": embedded.get("hold_reason"),
            "confirmed_subtype": embedded.get("confirmed_subtype"),
            "classification": deepcopy(classification), "requests": embedded["requests"],
            "tokens": deepcopy(tokens), "wall": deepcopy(wall),
            "selection_actor": deepcopy(embedded.get("selection_actor", [])),
        })
    _require(public_rows[0]["end_state"] == "HOLD"
             and public_rows[1]["end_state"] == "confirmed",
             "등록 규칙 HOLD와 모델 확정의 개발 측정 결과가 다르다")
    return manifest, {
        "schema": SCHEMA + "_diagnosis_v1", "case_id": "case-07", "seed": 2307,
        "arm": "COMPOUND", "measurement_scope": result["measurement_scope"],
        "rows": public_rows,
        "model_diagnostic_selection": {
            "policy": "model_adaptive",
            "probe_order": deepcopy(public_rows[1]["probe_order"]),
            "confirmed_subtype": public_rows[1]["confirmed_subtype"],
            "correction_choice": None,
        },
        "original_30_row_result_unchanged": True,
    }


def _first_hold(row: dict[str, Any], receipt: dict[str, Any]) -> dict[str, Any]:
    expected = FIRST._hold_receipt(row, row["hold_reason"])
    _require(row.get("status") == "HOLD" and receipt == expected,
             "FIRST HOLD receipt가 followup=false 계약과 다르다")
    return {
        "schema": RECEIPT_SCHEMA, "epoch": 1, "case_id": row["case_id"],
        "policy": row["policy"], "arm": row["arm"], "seed": row["seed"],
        "status": "HOLD", "reason": row["hold_reason"], "sdk_jobs": 0,
        "training_updates": 0, "fixed_evaluations": 0,
        "followup_epoch_allowed": False,
    }


def _first(manifest_path: Path, result_path: Path, retry_manifest_path: Path,
           retry_result_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, bytes], list[str]]:
    manifest, protocol = FIRST._load_manifest(Path(manifest_path))
    _require(_ref_matches(manifest["retry_manifest"], retry_manifest_path)
             and _ref_matches(manifest["retry_result"], retry_result_path),
             "FIRST 입력이 제공된 재진단 원문과 다르다")
    result = _read(result_path)
    receipts = result.get("receipts")
    rows = manifest["rows"]
    _require(result.get("schema") == FIRST.RESULT_SCHEMA and result.get("status") == "complete"
             and result.get("scope") == "compound_case07_first_registered_correction_only"
             and result.get("overall_goal_complete") is False
             and result.get("epoch2_registered") is False and result.get("model_calls") == 0
             and result.get("automatic_retry") is False
             and isinstance(receipts, list) and len(receipts) == len(rows) == 2,
             "FIRST 완료 결과의 2행/scope 계약이 다르다")
    if not isinstance(receipts, list):
        raise TypeError("FIRST receipts가 목록이 아니다")
    by_key = {(row["case_id"], row["policy"]): row for row in rows}
    public, files, corrections = [], {}, []
    smoke_ref = None
    with BASE.V2._v1_scope():
        for embedded in receipts:
            _path, receipt = _embedded_receipt(embedded)
            key = (receipt.get("case_id"), receipt.get("policy"))
            _require(key in by_key, "FIRST receipt가 등록 2행 밖이다")
            row = by_key.pop(key)
            if row["status"] == "HOLD":
                item = _first_hold(row, receipt)
            else:
                item, report_files = BASE._candidate(
                    row, receipt, manifest, Path(result_path).resolve().parent)
                files.update(report_files)
                item["schema"], item["epoch"] = RECEIPT_SCHEMA, 1
                item["followup_epoch_allowed"] = receipt.get("followup_epoch_allowed")
                item["scope"]["second_epoch_eligible"] = receipt.get("followup_epoch_allowed")
                corrections.append(item["correction"])
                smoke_ref = receipt.get("smoke")
            public.append(item)
    _require(not by_key and len(corrections) == 1 and smoke_ref is not None,
             "FIRST는 HOLD 1행과 candidate 1행이어야 한다")
    if not isinstance(smoke_ref, dict):
        raise TypeError("FIRST smoke 참조가 객체가 아니다")
    FIRST.verify_smoke(Path(smoke_ref["path"]), manifest, protocol)
    return manifest, public, files, corrections


def _second_hold(row: dict[str, Any], receipt: dict[str, Any]) -> dict[str, Any]:
    expected = {
        "schema": BASE.V1.RECEIPT_SCHEMA, "status": "HOLD",
        "scope": "second_registered_correction_only", "case_id": "case-07",
        "policy": row["policy"], "reason": row["hold_reason"], "sdk_jobs": 0,
        "approval": False, "model_calls": 0, "overall_goal_complete": False,
        "third_epoch_registered": False,
    }
    _require(row.get("status") == "HOLD" and receipt == expected,
             "SECOND HOLD receipt가 등록 계약과 다르다")
    return {
        "schema": RECEIPT_SCHEMA, "epoch": 2, "case_id": "case-07",
        "policy": row["policy"], "arm": row["arm"], "seed": row["seed"],
        "status": "HOLD", "reason": row["hold_reason"], "sdk_jobs": 0,
        "training_updates": 0, "fixed_evaluations": 0,
        "followup_epoch_allowed": False,
    }


def _second_candidate(row: dict[str, Any], receipt: dict[str, Any], manifest: dict[str, Any],
                      execution_root: Path) -> tuple[dict[str, Any], dict[str, bytes], str]:
    candidate = SECOND._checked_candidate(row, manifest)
    before, after = candidate["applied_corrections_before"], candidate["applied_corrections_after"]
    _require(receipt.get("case_id") == "case-07" and receipt.get("policy") == row["policy"]
             and receipt.get("seed") == 2307 and receipt.get("candidate") == row["candidate"]
             and receipt.get("component") == candidate["component"]
             and receipt.get("correction") == candidate["correction"]
             and receipt.get("scope") == "second_registered_correction_only"
             and receipt.get("epoch") == 2
             and receipt.get("prior_not_recovered_receipt") == row["prior_receipt"]
             and receipt.get("prior_correction_ledger") == row["first_ledger"]
             and receipt.get("applied_corrections_before") == before
             and receipt.get("applied_corrections_after") == after
             and receipt.get("remaining_faults") == []
             and receipt.get("third_epoch_registered") is False
             and receipt.get("followup_epoch_allowed") is False
             and receipt.get("next_step") == "stop" and receipt.get("model_calls") == 0,
             "SECOND receipt의 prior/교정/scope 결합이 다르다")
    _require(len(before) == 1 and len(after) == 2 and len(set(after)) == 2
             and after[0] == before[0] and after[1] == candidate["correction"],
             "서로 다른 교정 최대 2개 계약이 다르다")
    stage = Path(execution_root) / "run"
    train_manifest = _read(stage / "train/manifest.json")
    BASE.V1.TRAIN.verify_binding(train_manifest, lab=Path(manifest["lab"]))
    training_root = stage / "train/training"
    BASE.V1.GUARD.validate_completion(
        training_root, BASE.V1.TRAIN.manifest_fingerprint(train_manifest))
    evidence = _read(training_root / BASE.V1.GUARD.EVIDENCE)
    checkpoint = evidence.get("checkpoint", {})
    checkpoint_path = Path(str(checkpoint.get("path", "")))
    _require(evidence == receipt.get("training")
             and evidence.get("updates") == evidence.get("update_count") == 300
             and evidence.get("fresh_initialization") is True
             and evidence.get("remaining_faults") == []
             and evidence.get("applied_corrections") == after
             and checkpoint_path.is_file() and _sha(checkpoint_path) == checkpoint.get("sha256")
             and checkpoint.get("iteration") == 299 and receipt.get("new_checkpoint") == checkpoint,
             "SECOND guarded fresh300/readback/checkpoint가 다르다")
    collected, _telemetry = BASE.V1.RECOVERY_INPUTS.checked_recovery(
        stage / "collected/collected.json")
    _require(collected == receipt.get("collected_recovery"), "SECOND collected가 receipt와 다르다")
    corrected_path = Path(receipt["corrected_report"]["path"])
    reference_path = Path(receipt["reference_report"]["path"])
    corrected = BASE._fixed_report(corrected_path, receipt["corrected_report"]["sha256"])
    reference = BASE._fixed_report(reference_path, receipt["reference_report"]["sha256"])
    _require(reference_path.resolve() == Path(row["reference_report"]["path"]).resolve()
             and reference["checkpoint"]["sha256"] == row["reference_checkpoint_sha256"]
             and corrected["checkpoint"]["sha256"] == checkpoint["sha256"],
             "SECOND fixed report checkpoint 결합이 다르다")
    recovery = BASE.V1.C.D1.recovery(corrected, reference, _read(BASE.V1.BANDS)["bands"])
    expected_status = {"recovered": "recovered", "not_recovered": "not_recovered",
                       "unknown": "HOLD"}[recovery["outcome"]]
    _require(receipt.get("recovery") == recovery and receipt.get("status") == expected_status,
             "SECOND fixed26 재계산 결과가 receipt와 다르다")
    evaluation = receipt.get("evaluation_execution")
    _require(isinstance(evaluation, dict), "SECOND evaluation receipt가 객체가 아니다")
    if not isinstance(evaluation, dict):
        raise TypeError("SECOND evaluation receipt가 객체가 아니다")
    _require(BASE.V1._owned_ok(evaluation, BASE.V1.WATCHDOG["eval"]),
             "SECOND fixed evaluation owned receipt가 완결되지 않았다")
    ledger = BASE.V1.C.D1.CorrectionLedger(
        Path(execution_root) / "correction.ledger.jsonl", BASE.V1.C.RC.correction_registry())
    events = ledger.events()
    _require([event["event"] for event in events] == ["proposed", "approved", "consumed", "receipt"]
             and events[-1] == receipt.get("ledger_receipt"),
             "SECOND 승인·소비·결과 원장이 다르다")
    stem = f"case-07_{row['policy']}_epoch2"
    files = {
        f"reports/{stem}-corrected.json": corrected_path.read_bytes(),
        f"reports/{stem}-same-seed-reference.json": reference_path.read_bytes(),
    }
    public = {
        "schema": RECEIPT_SCHEMA, "epoch": 2, "case_id": "case-07",
        "policy": row["policy"], "arm": row["arm"], "seed": row["seed"],
        "status": receipt["status"], "component": candidate["component"],
        "correction": candidate["correction"], "applied_corrections_before": before,
        "applied_corrections_after": after,
        "training": {"mode": "fresh_restart", "updates": 300,
                     "fresh_initialization": True,
                     "new_checkpoint_sha256": checkpoint["sha256"]},
        "evaluation": {"conditions": 26,
            "corrected_report": {"path": f"reports/{stem}-corrected.json",
                                 "sha256": _sha(corrected_path)},
            "same_seed_reference_report": {
                "path": f"reports/{stem}-same-seed-reference.json",
                "sha256": _sha(reference_path)}},
        "recovery": deepcopy(recovery), "followup_epoch_allowed": False,
        "scope": {"second_distinct_registered_correction_only": True,
                  "third_epoch_registered": False, "overall_goal_complete": False},
    }
    return public, files, candidate["correction"]


def _checked_epoch2_adapters(epoch_result_path: Path, context: dict[str, Any]) -> dict[str, Any]:
    root = Path(epoch_result_path).resolve().parent
    v3_path, v2_path = root / "v3-adapter-receipt.json", root / "v2-wrapper-receipt.json"
    _require(v3_path.is_file() and v2_path.is_file(), "epoch2 v3/v2 adapter receipt가 없다")
    v3, v2 = _read(v3_path), _read(v2_path)
    parent = context["parent_binding"]
    _require(v3.get("schema") == EPOCH2_V3.WRAPPER_RECEIPT_SCHEMA
             and v3.get("status") == "complete"
             and _ref_matches(v3.get("parent_result", {}), epoch_result_path)
             and v3.get("parent_result_schema") == SECOND.E2.SCHEMA
             and v3.get("post_observation_sha256") == context["post_input"]["sha256"]
             and v3.get("parent_context") == {
                 "packet_digest": parent["packet_digest"], "run_id": context["run_id"],
                 "policy": context["policy"], "parent": parent["parent"]},
             "epoch2 v3 parent adapter receipt가 exact context/result와 다르다")
    _require(v2.get("schema") == EPOCH2_V2.WRAPPER_RECEIPT_SCHEMA
             and v2.get("status") == "complete"
             and _ref_matches(v2.get("parent_result", {}), epoch_result_path)
             and v2.get("parent_result_schema") == SECOND.E2.SCHEMA
             and v2.get("preflight_inside_output") is False
             and v2.get("diagnosis_replayed") is False
             and v2.get("training_replayed") is False,
             "epoch2 v2 wrapper receipt가 exact result/preflight 계약과 다르다")
    v3_v2, v2_protocol = v3.get("v2_protocol"), v2.get("wrapper_protocol")
    _require(isinstance(v3_v2, dict) and isinstance(v2_protocol, dict)
             and Path(str(v3_v2.get("path", ""))).resolve()
             == Path(str(v2_protocol.get("path", ""))).resolve()
             and Path(str(v2_protocol.get("path", ""))).is_file()
             and _sha(Path(v2_protocol["path"])) == v2_protocol.get("sha256")
             and EPOCH2_V2._sha(Path(v3_v2["path"]), lf=True) == v3_v2.get("sha256_lf"),
             "epoch2 v3→v2 protocol 원문/SHA 결합이 다르다")
    if not isinstance(v3_v2, dict):
        raise TypeError("epoch2 v3의 v2 protocol ref가 객체가 아니다")
    for receipt, key in ((v3, "v3_protocol"), (v2, "parent_protocol")):
        ref = receipt.get(key)
        _require(isinstance(ref, dict) and Path(str(ref.get("path", ""))).is_file(),
                 f"epoch2 {key} 원문이 없다")
        assert isinstance(ref, dict)
        expected = ref.get("sha256", ref.get("sha256_lf"))
        actual = (_sha(Path(ref["path"])) if "sha256" in ref
                  else EPOCH2_V2._sha(Path(ref["path"]), lf=True))
        _require(actual == expected, f"epoch2 {key} 원문 SHA가 다르다")
    return {
        "v3_adapter_receipt_sha256": _sha(v3_path),
        "v2_wrapper_receipt_sha256": _sha(v2_path),
        "v3_protocol_sha256": v3["v3_protocol"]["sha256"],
        "v2_protocol_sha256_lf": v3_v2["sha256_lf"],
        "parent_protocol_sha256_lf": v2["parent_protocol"]["sha256_lf"],
        "anchor": v3["anchor"],
        "preflight_inside_output": False,
        "parent_binding_verified": True,
    }


def _post_diagnosis_projection(row: dict[str, Any]) -> dict[str, Any]:
    ref = row.get("epoch_result")
    _require(isinstance(ref, dict) and _ref_matches(ref, Path(str(ref.get("path", "")))),
             "epoch2 diagnosis result 원문/SHA가 다르다")
    if not isinstance(ref, dict):
        raise TypeError("epoch2 result ref가 객체가 아니다")
    path = Path(ref["path"]).resolve()
    result = _read(path)
    steps = result.get("steps")
    _require(result.get("schema") == SECOND.E2.SCHEMA and result.get("status") == "complete"
             and result.get("evidence_scope") == "actual_backend" and result.get("epoch") == 2
             and result.get("case_id") == "case-07" and result.get("policy") == row["policy"]
             and result.get("correction_executed") is False
             and result.get("training_executed") is False
             and isinstance(steps, list) and len(steps) == result.get("decision_count")
             and result.get("probe_order") == [step.get("probe") for step in steps],
             "epoch2 actual diagnosis result 계약이 다르다")
    if not isinstance(steps, list):
        raise TypeError("epoch2 diagnosis steps가 목록이 아니다")
    events, requests = SECOND.CANDIDATE2._completed_ledger(
        Path(steps[0]["authorization_path"]).resolve().parents[2] / "probes.ledger.jsonl", steps)
    _require(len(events) == 4 * len(steps) and len(requests) == len(steps),
             "epoch2 probe 원장의 4사건/step 계약이 다르다")
    public_steps: list[dict[str, Any]] = []
    for index, step in enumerate(steps, 1):
        auth = Path(step["authorization_path"]).resolve()
        _require(step.get("decision") == index and auth.is_file()
                 and _sha(auth) == step.get("authorization_sha256"),
                 "epoch2 diagnosis step/auth 순서 또는 SHA가 다르다")
        paired = step.get("paired_receipts")
        _require(isinstance(paired, dict) and set(paired) == {"case", "reference"},
                 "epoch2 diagnosis paired receipt 역할이 다르다")
        if not isinstance(paired, dict):
            raise TypeError("epoch2 paired receipts가 객체가 아니다")
        public_roles: dict[str, Any] = {}
        for role in ("case", "reference"):
            raw_path = auth.parent / role / "probe_receipt.json"
            _require(raw_path.is_file() and _read(raw_path) == paired[role],
                     f"epoch2 {role} raw probe receipt가 embedded 값과 다르다")
            public_roles[role] = {
                "status": paired[role].get("status"),
                "measurement": deepcopy(paired[role].get("measurement")),
                "raw_receipt_sha256": _sha(raw_path),
            }
        public_steps.append({
            "decision": index, "probe": step["probe"], "outcome": step["outcome"],
            "request_id": step["request_id"], "roles": public_roles,
        })
    binding = result.get("source_binding")
    _require(isinstance(binding, dict) and set(binding) == {"case", "reference"},
             "epoch2 source binding 역할이 다르다")
    if not isinstance(binding, dict):
        raise TypeError("epoch2 source binding이 객체가 아니다")
    public_binding = {
        role: {key: binding[role].get(key) for key in
               ("checkpoint_sha256", "telemetry_sha256", "seed")}
        for role in ("case", "reference")
    }
    return {
        "schema": SCHEMA + "_post_correction_diagnosis_v1",
        "epoch": 2, "run_id": result.get("run_id"), "policy": result.get("policy"),
        "end_state": result.get("end_state"), "hold_reason": result.get("hold_reason"),
        "confirmed_subtype": result.get("confirmed_subtype"),
        "probe_order": deepcopy(result.get("probe_order")),
        "decision_count": result.get("decision_count"), "steps": public_steps,
        "requests": result.get("requests"),
        "model_validity": deepcopy(result.get("model_validity")),
        "selection_actor": deepcopy(result.get("selection_actor")),
        "tokens": deepcopy(result.get("tokens")), "wall": deepcopy(result.get("wall")),
        "source_binding": public_binding,
        "prior_not_recovered_receipt_sha256": result.get(
            "prior_not_recovered_receipt_sha256"),
        "source_result_sha256": ref["sha256"],
        "outcome_reinterpreted": False,
    }


def _second(manifest_path: Path, result_path: Path) -> tuple[
        dict[str, Any], dict[str, bytes], str | None, dict[str, Any], dict[str, Any]]:
    prepare_receipt, context = SECOND2._checked_prepare_receipt(Path(manifest_path))
    execution_root = Path(result_path).resolve().parent
    execution_receipt_path = execution_root / SECOND2.EXECUTION_RECEIPT
    execution_receipt = _read(execution_receipt_path)
    _require(execution_receipt.get("schema") == SECOND2.EXECUTION_RECEIPT_SCHEMA
             and _ref_matches(execution_receipt.get("prepare_adapter_receipt", {}),
                              Path(manifest_path).resolve().parent / SECOND2.PREPARE_RECEIPT)
             and execution_receipt.get("wrapper_protocol") == prepare_receipt["wrapper_protocol"]
             and execution_receipt.get("wrapper_anchor") == prepare_receipt["wrapper_anchor"]
             and _ref_matches(execution_receipt.get("inner_result", {}), result_path)
             and execution_receipt.get("inner_result_schema") == SECOND.RESULT_SCHEMA
             and execution_receipt.get("context_sha256") == prepare_receipt["context_sha256"]
             and execution_receipt.get("training_path") == "S1_BOTH_fresh300_fixed26",
             "SECOND 실행 adapter receipt가 prepare/result와 다르다")
    adapter = _checked_epoch2_adapters(Path(context["epoch2_result"]["path"]), context)
    adapter.update({
        "second_prepare_adapter_receipt_sha256": _sha(
            Path(manifest_path).resolve().parent / SECOND2.PREPARE_RECEIPT),
        "second_execution_adapter_receipt_sha256": _sha(execution_receipt_path),
        "second_wrapper_protocol_sha256": prepare_receipt["wrapper_protocol"]["sha256"],
        "second_wrapper_anchor": prepare_receipt["wrapper_anchor"],
    })
    with PARENT.parent_scope(
            Path(context["post_input"]["path"]), Path(context["reference_input"]["path"]),
            context["run_id"], context["policy"]) as parent_context:
        _require(parent_context == context["parent_binding"],
                 "SECOND exporter parent scope가 prepare context와 다르다")
        manifest, _protocol = SECOND._load_manifest(Path(manifest_path))
        diagnosis = _post_diagnosis_projection(manifest["row"])
        result = _read(result_path)
        receipts = result.get("receipts")
        _require(result.get("schema") == SECOND.RESULT_SCHEMA and result.get("status") == "complete"
                 and result.get("scope") == "second_distinct_registered_correction_only"
                 and result.get("overall_goal_complete") is False
                 and result.get("third_epoch_registered") is False
                 and result.get("model_calls") == 0 and result.get("automatic_retry") is False
                 and isinstance(receipts, list) and len(receipts) == 1,
                 "SECOND 완료 결과의 단일행/scope 계약이 다르다")
        if not isinstance(receipts, list):
            raise TypeError("SECOND receipts가 목록이 아니다")
        _path, receipt = _embedded_receipt(receipts[0])
        row = manifest["row"]
        if row["status"] == "HOLD":
            return _second_hold(row, receipt), {}, None, adapter, diagnosis
        public, files, correction = _second_candidate(
            row, receipt, manifest, execution_root)
        return public, files, correction, adapter, diagnosis


def _validated(retry_manifest: Path, retry_result: Path, first_manifest: Path,
               first_result: Path, second_manifest: Path | None,
               second_result: Path | None) -> tuple[dict[str, Any], dict[str, Any], dict[str, bytes]]:
    retry_input, diagnosis = _retry_summary(retry_manifest, retry_result)
    first_input, first_receipts, files, corrections = _first(
        first_manifest, first_result, retry_manifest, retry_result)
    candidate = next(item for item in first_receipts if item.get("correction"))
    first_status = candidate["status"]
    supplied = second_manifest is not None or second_result is not None
    _require((second_manifest is None) == (second_result is None),
             "SECOND manifest/result는 함께 제공해야 한다")
    _require(not (first_status == "recovered" and supplied),
             "FIRST recovered 뒤 SECOND 실행을 공개할 수 없다")
    second_receipt = None
    second_adapter = None
    post_diagnosis = None
    if supplied:
        _require(first_status == "not_recovered",
                 "SECOND는 FIRST not_recovered 뒤에만 허용한다")
        assert second_manifest is not None and second_result is not None
        second_receipt, second_files, correction, second_adapter, post_diagnosis = _second(
            second_manifest, second_result)
        files.update(second_files)
        files["post-correction-diagnosis.json"] = BASE._bytes(post_diagnosis)
        if correction is not None:
            corrections.append(correction)
    _require(len(corrections) <= 2 and len(corrections) == len(set(corrections)),
             "적용 교정은 서로 다른 최대 2개여야 한다")
    diagnosis["model_diagnostic_selection"]["correction_choice"] = {
        "selected_by": "host_registered_correction_from_confirmed_subtype",
        "correction": corrections[0],
    }
    files["diagnosis.json"] = BASE._bytes(diagnosis)
    public_refs = []
    for item in [*first_receipts, *([] if second_receipt is None else [second_receipt])]:
        relative = f"receipts/epoch{item['epoch']}-{item['policy']}.json"
        raw = BASE._bytes(item)
        files[relative] = raw
        public_refs.append({"epoch": item["epoch"], "policy": item["policy"],
                            "status": item["status"], "path": relative,
                            "sha256": hashlib.sha256(raw).hexdigest()})
    terminal = second_receipt or candidate
    summary = {
        "schema": SCHEMA, "status": "complete", "case_id": "case-07", "seed": 2307,
        "arm": "COMPOUND", "diagnosis_rows": 2,
        "first_recovery_status": first_status,
        "second_recovery_status": None if second_receipt is None else second_receipt["status"],
        "epoch2_diagnosis_outcome": None if post_diagnosis is None else {
            "end_state": post_diagnosis["end_state"],
            "hold_reason": post_diagnosis["hold_reason"],
            "confirmed_subtype": post_diagnosis["confirmed_subtype"],
            "path": "post-correction-diagnosis.json",
            "sha256": None if post_diagnosis is None else hashlib.sha256(
                files["post-correction-diagnosis.json"]).hexdigest(),
        },
        "terminal_status": terminal["status"], "distinct_corrections": corrections,
        "max_distinct_corrections": 2, "second_stage_supplied": supplied,
        "receipts": public_refs, "source_anchor": first_input["anchor"],
        "scope": {"development_after_observed_infrastructure_failure": True,
                  "same_seed_pairing": True, "original_30_row_result_unchanged": True,
                  "model_diagnosis_and_host_correction_separated": True,
                  "unknown_remains_hold": True, "generalization_claim": False},
    }
    files["summary.json"] = BASE._bytes(summary)
    provenance = {
        "schema": SCHEMA + "_provenance_v1", "source_anchor": first_input["anchor"],
        "source_retry_anchor": retry_input["execution_anchor"],
        "source_retry_manifest_sha256": _sha(retry_manifest),
        "source_retry_result_sha256": _sha(retry_result),
        "source_first_manifest_sha256": _sha(first_manifest),
        "source_first_result_sha256": _sha(first_result),
        "source_second_manifest_sha256": None if second_manifest is None else _sha(second_manifest),
        "source_second_result_sha256": None if second_result is None else _sha(second_result),
        "second_adapter": deepcopy(second_adapter),
        "raw_paths_in_public_files": False,
    }
    files["provenance.json"] = BASE._bytes(provenance)
    return summary, provenance, files


def export_completed(retry_manifest: Path, retry_result: Path, first_manifest: Path,
                     first_result: Path, *, out: Path, second_manifest: Path | None = None,
                     second_result: Path | None = None, repo: Path = ROOT) -> dict[str, Any]:
    """원문을 재검증하고 repo 안의 새 공개 폴더에 허용 파일만 쓴다."""
    out, repo = Path(out).resolve(), Path(repo).resolve()
    _require(not out.exists() and out != repo and out.is_relative_to(repo),
             "공개 export는 repo 안의 새 하위 경로여야 한다")
    summary, provenance, files = _validated(
        retry_manifest, retry_result, first_manifest, first_result,
        second_manifest, second_result)
    for name, raw in files.items():
        BASE._privacy(name, raw)
    out.mkdir(parents=True, exist_ok=False)
    for name, raw in sorted(files.items()):
        BASE._write_new(out / name, raw)
    artifact = {"schema": MANIFEST_SCHEMA,
                "files": {name: hashlib.sha256(raw).hexdigest()
                          for name, raw in sorted(files.items())},
                "manifest_self_sha256_recorded": False}
    BASE._write_new(out / "artifact-manifest.json", BASE._bytes(artifact))
    verify_export(out)
    return {"summary": summary, "provenance": provenance, "artifact_manifest": artifact}


def verify_export(out: Path) -> dict[str, Any]:
    root = Path(out).resolve()
    artifact = _read(root / "artifact-manifest.json")
    files = artifact.get("files")
    _require(artifact.get("schema") == MANIFEST_SCHEMA
             and artifact.get("manifest_self_sha256_recorded") is False
             and isinstance(files, dict) and "artifact-manifest.json" not in files,
             "공개 artifact manifest 계약이 다르다")
    if not isinstance(files, dict):
        raise TypeError("공개 artifact files가 객체가 아니다")
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    _require(actual == set(files) | {"artifact-manifest.json"},
             "공개 파일 목록이 manifest와 다르다")
    for name, expected in files.items():
        raw = (root / name).read_bytes()
        _require(hashlib.sha256(raw).hexdigest() == expected, f"공개 파일 SHA가 다르다: {name}")
        BASE._privacy(name, raw)
    summary = _read(root / "summary.json")
    _require(summary.get("schema") == SCHEMA and summary.get("status") == "complete"
             and summary.get("diagnosis_rows") == 2
             and len(summary.get("distinct_corrections", [])) <= 2
             and len(summary.get("distinct_corrections", []))
             == len(set(summary.get("distinct_corrections", [])))
             and summary.get("max_distinct_corrections") == 2,
             "공개 summary의 복합 회복 계약이 다르다")
    return deepcopy(artifact)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export")
    for name in ("retry-manifest", "retry-result", "first-manifest", "first-result", "out"):
        export.add_argument("--" + name, type=Path, required=True)
    export.add_argument("--second-manifest", type=Path)
    export.add_argument("--second-result", type=Path)
    export.add_argument("--repo", type=Path, default=ROOT)
    verify = sub.add_parser("verify")
    verify.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "verify":
        verify_export(args.out)
    else:
        export_completed(args.retry_manifest, args.retry_result,
                         args.first_manifest, args.first_result, out=args.out,
                         second_manifest=args.second_manifest,
                         second_result=args.second_result, repo=args.repo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
