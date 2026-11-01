# pyright: reportMissingImports=false
"""사전 고정한 R4 네 pipeline의 first/second 회복 결과를 공개 투영한다."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("bench", "evals", "src")]

import five_policy_recovery_candidate_v1 as FIRST
import five_policy_recovery_epoch2_v1 as SECOND
import run_five_policy_recovery_loop_v1 as PARENT

SUMMARY_SCHEMA = "five_policy_recovery_public_summary_v1"
CASES_SCHEMA = "five_policy_recovery_public_cases_v1"
PROVENANCE_SCHEMA = "five_policy_recovery_public_provenance_v1"
MANIFEST_SCHEMA = "five_policy_recovery_public_manifest_v1"
SOURCE_FILES = (
    "evals/export_five_policy_recovery_v1.py",
    "tests/test_export_five_policy_recovery_v1.py",
    "bench/run_five_policy_recovery_loop_v1.py",
    "tests/test_run_five_policy_recovery_loop_v1.py",
)
TARGETS = tuple(PARENT.TARGETS)


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(value, dict), "JSON object가 필요하다")
    return value


def _sha(path: Path, *, lf: bool = False) -> str:
    raw = Path(path).read_bytes()
    if lf:
        raw = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(raw).hexdigest()


def _digest(value: object) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _write_new(path: Path, value: object) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def _bound(value: object, label: str) -> Path:
    _require(isinstance(value, Mapping) and set(value) == {"path", "sha256"}, f"{label} ref")
    row = cast(Mapping[str, Any], value)
    path = Path(cast(str, row["path"])).resolve(strict=True)
    _require(_sha(path) == row["sha256"], f"{label} SHA")
    return path


def _public_ref(path: Path) -> dict[str, Any]:
    target = Path(path).resolve(strict=True)
    return {"sha256": _sha(target), "size_bytes": target.stat().st_size}


def _relative_protocol(path: Path, *, allow_fixture: bool) -> dict[str, str]:
    target = Path(path).resolve(strict=True)
    try:
        relative = target.relative_to(ROOT).as_posix()
    except ValueError as error:
        if allow_fixture:
            return {"sha256": _sha(target)}
        raise ValueError("protocol은 저장소 공개 파일이어야 한다") from error
    if not allow_fixture:
        _require(relative.startswith("bench/protocols/"), "production protocol 공개 경로")
    return {"path": relative, "sha256": _sha(target)}


def _approval(path: Path, *, schema: str, candidate: Path, protocol: Path,
              anchor: str, seed: int, lease_id: str) -> dict[str, Any]:
    value = _read(path)
    _require(value.get("schema") == schema and value.get("decision") == "approve"
             and isinstance(value.get("approver"), str) and bool(value["approver"])
             and value.get("candidate") == FIRST.ref(candidate)
             and value.get("protocol") == FIRST.ref(protocol)
             and value.get("anchor") == anchor and value.get("registered_seed") == seed
             and value.get("lease_id") == lease_id,
             "operator approval 결합")
    return value


def _wall(value: object, label: str) -> float | None:
    if value is None:
        return None
    _require(isinstance(value, (int, float)) and not isinstance(value, bool)
             and float(value) >= 0, f"{label} wall_s")
    return float(cast(float, value))


def _training(receipt: Mapping[str, Any], candidate: Mapping[str, Any], *,
              result_path: Path, lab: Path, second: bool,
              allow_fixture: bool) -> dict[str, Any]:
    training = receipt.get("training")
    _require(isinstance(training, Mapping), "training receipt")
    training_row = cast(Mapping[str, Any], training)
    evidence = _read(_bound(training_row.get("evidence"), "training evidence"))
    source_cp = candidate.get("source_checkpoint_sha256")
    _require(evidence.get("fresh_initialization") is True
             and evidence.get("checkpoint_loaded") is False
             and evidence.get("resume_requested") is False
             and evidence.get("source_checkpoint_sha256") == source_cp
             and receipt.get("training_mode") == "fresh_restart"
             and receipt.get("source_checkpoint_loaded") is False
             and receipt.get("source_checkpoint_sha256") == source_cp,
             "fresh300/source checkpoint 계약")
    checkpoint = training_row.get("checkpoint")
    _require(isinstance(checkpoint, Mapping) and "sha256" in checkpoint,
             "new checkpoint receipt")
    checkpoint_row = cast(Mapping[str, Any], checkpoint)
    new_cp = checkpoint_row.get("sha256")
    _require(isinstance(new_cp, str) and len(new_cp) == 64, "new checkpoint SHA")
    fixed = receipt.get("fixed26")
    _require(isinstance(fixed, Mapping) and fixed.get("conditions") == 26,
             "fixed26 receipt")
    fixed_row = cast(Mapping[str, Any], fixed)
    execution = fixed_row.get("execution")
    _require(isinstance(execution, Mapping), "fixed26 execution")
    execution_row = cast(Mapping[str, Any], execution)
    train_execution = training_row.get("execution")
    _require(isinstance(train_execution, Mapping) or allow_fixture, "training execution")
    train_owned_wall = None
    fixed_owned_wall = None
    current_cp = new_cp
    if not allow_fixture:
        manifest_path = _bound(training_row.get("manifest"), "training manifest")
        owned_path = _bound(training_row.get("owned_result"), "training owned result")
        collected_path = _bound(training_row.get("collected"), "training collected")
        collected, metadata = FIRST.TRAINING.INPUT.checked(collected_path)
        _require(Path(metadata["manifest"]).resolve() == manifest_path
                 and Path(metadata["owned_result"]).resolve() == owned_path
                 and Path(metadata["evidence_dir"]).resolve() == _bound(
                     training_row.get("evidence"), "training evidence").parent,
                 "training collected 원문 결합")
        manifest = FIRST.TRAINING.TRAIN.validate_manifest(_read(manifest_path))
        _require(manifest.get("source_checkpoint_sha256") == source_cp
                 and manifest.get("max_iterations") == 300
                 and manifest.get("arm") == ("BOTH" if second else "FIRST"),
                 "training manifest source/arm/300")
        owned = _read(owned_path)
        _require(FIRST.TRAINING.EXEC._owned_ok(
            owned, FIRST.TRAINING.EXEC.WATCHDOG["train"]), "training owned receipt")
        _require(collected.get("source_binding", {}).get("checkpoint_sha256") == new_cp,
                 "collected current checkpoint")
        current_cp = cast(str, collected["source_binding"]["checkpoint_sha256"])
        checkpoint_path = Path(cast(str, checkpoint_row.get("path"))).resolve(strict=True)
        _require(_sha(checkpoint_path) == new_cp, "training checkpoint raw SHA")
        evaluation = Path(result_path).resolve().parent / "evaluation"
        jobs_path, eval_owned_path = evaluation / "jobs.json", evaluation / "owned-result.json"
        jobs, eval_owned = _read(jobs_path), _read(eval_owned_path)
        job_rows = jobs.get("jobs")
        _require(isinstance(job_rows, list) and len(job_rows) == 1, "fixed26 jobs")
        job = cast(dict[str, Any], cast(list[Any], job_rows)[0])
        report_path = Path(cast(str, job.get("output"))).resolve(strict=True)
        _require(Path(cast(str, job.get("checkpoint"))).resolve(strict=True) == checkpoint_path
                 and _sha(checkpoint_path) == new_cp, "fixed26 checkpoint")
        _require(FIRST.TRAINING.EXEC._owned_ok(
            eval_owned, FIRST.TRAINING.EXEC.WATCHDOG["eval"]), "fixed26 owned receipt")
        _require(dict(execution_row) == eval_owned, "fixed26 execution/owned 일치")
        report = _read(report_path)
        _require(report == fixed_row.get("corrected_report"), "fixed26 report 원문")
        input_key = "reference_source_v2" if second else "reference_probe_input"
        checker = SECOND.SOURCE_INPUT if second else FIRST.SOURCE_INPUT
        _bridge, raw = checker.checked(_bound(
            cast(Mapping[str, Any], candidate["inputs"])[input_key], "reference input"))
        reference_report = _read(Path(raw["raw"]["fixed_result"]["path"])).get("report")
        _require(isinstance(reference_report, dict), "reference fixed26 report")
        reference_row = cast(dict[str, Any], reference_report)
        verdict = FIRST.GATE.RESIDUAL.recovery(
            report, reference_row, _read(FIRST.OLD_RECOVERY.BANDS)["bands"])["verdict"]
        status = "HOLD" if verdict.get("outcome") == "unknown" else verdict.get("outcome")
        _require(verdict == receipt.get("recovery") and status == receipt.get("status"),
                 "fixed26 recovery 재계산")
        train_owned_wall = _wall(owned.get("wall_s"), "training owned")
        fixed_owned_wall = _wall(eval_owned.get("wall_s"), "fixed owned")
    train_execution_wall = _wall(
        train_execution.get("wall_s") if isinstance(train_execution, Mapping) else None,
        "training execution")
    fixed_execution_wall = _wall(execution_row.get("wall_s"), "fixed execution")
    missing = [name for name, value in {
        "training_execution_wall_s": train_execution_wall,
        "training_owned_wall_s": train_owned_wall,
        "fixed_execution_wall_s": fixed_execution_wall,
        "fixed_owned_wall_s": fixed_owned_wall,
    }.items() if value is None]
    return {
        "original_source_checkpoint_sha256": source_cp,
        "source_checkpoint_loaded": False,
        "fresh_initialization": True,
        "new_checkpoint_sha256": new_cp,
        "collected_current_checkpoint_sha256": current_cp,
        "fixed_conditions": 26,
        "training_execution_wall_s": train_execution_wall,
        "training_owned_wall_s": train_owned_wall,
        "fixed_execution_wall_s": fixed_execution_wall,
        "fixed_owned_wall_s": fixed_owned_wall,
        "missing_cost_reason": None if not missing else {
            "reason": "not_recorded_in_source_receipt", "fields": missing},
    }


def _model_cost(diagnosis_path: Path, *, call_verifier: Callable[[Path], Mapping[str, Any]]) -> dict[str, Any]:
    result = _read(diagnosis_path)
    steps = result.get("steps")
    _require(isinstance(steps, list) and len(steps) <= 3, "second diagnosis steps")
    step_rows = cast(list[dict[str, Any]], steps)
    expected_calls = result.get("model_calls")
    _require(type(expected_calls) is int and 0 <= expected_calls <= 3, "second model_calls")
    model_results = sorted(diagnosis_path.parent.glob("step[0-9][0-9]/model-result.json"))
    _require(len(model_results) == cast(int, expected_calls), "model-result/model_calls 원장")
    tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    provider_wall = 0.0
    unknown_usage = 0
    seen_calls: set[str] = set()
    receipt_reuses = 0
    missing_receipts = 0
    owned_values: list[float] = []
    for step in step_rows:
        wall = step.get("owned_child_wall_s")
        if isinstance(wall, (int, float)) and not isinstance(wall, bool):
            owned_values.append(float(wall))
    for model_result_path in model_results:
        model_result = _read(model_result_path)
        ref = model_result.get("call_ref")
        if not isinstance(ref, Mapping):
            missing_receipts += 1
            continue
        ref_row = cast(Mapping[str, Any], ref)
        call_path = Path(cast(str, ref_row["path"])).resolve(strict=True)
        checked = dict(call_verifier(call_path))
        _require(checked.get("call_ref") == ref_row, "second model call receipt")
        identity = _digest(ref_row)
        if identity in seen_calls:
            receipt_reuses += 1
            continue
        seen_calls.add(identity)
        provider_wall += float(checked["wall_s"])
        usage = checked.get("usage")
        if not isinstance(usage, Mapping):
            unknown_usage += 1
            continue
        for target, aliases in {
            "prompt_tokens": ("prompt_tokens", "input_tokens"),
            "completion_tokens": ("completion_tokens", "output_tokens"),
            "total_tokens": ("total_tokens",),
        }.items():
            item = next((usage[name] for name in aliases if name in usage), None)
            _require(type(item) is int and item >= 0, "provider usage")
            tokens[target] += cast(int, item)
    usage_complete = unknown_usage == 0 and missing_receipts == 0
    return {
        "paired_probe_sdk_executions": len(step_rows) * 2,
        "model_attempts": expected_calls,
        "model_api_calls": len(seen_calls),
        "provider_receipt_reuses": receipt_reuses,
        "provider_receipt_missing_attempts": missing_receipts,
        "provider_tokens": tokens if usage_complete else None,
        "provider_tokens_missing_reason": None if usage_complete else
        "usage_or_receipt_missing",
        "provider_usage_unknown_calls": unknown_usage,
        "provider_wall_s": provider_wall if missing_receipts == 0 else None,
        "provider_wall_missing_reason": None if missing_receipts == 0 else
        "provider_receipt_missing",
        "owned_child_wall_s": sum(owned_values) if len(owned_values) == len(step_rows) else None,
        "owned_child_wall_missing_reason": None if len(owned_values) == len(step_rows)
        else "not_recorded_for_every_probe",
    }


def _assert_public(*values: object) -> None:
    bad_key = re.compile(r"^(api_?key|authorization|credential|secret|access_?token)$", re.IGNORECASE)
    bad_value = re.compile(
        r"(?:[A-Za-z]:[\\/]|\\\\|/(?:home|Users|mnt|tmp)/|bench[\\/]private|"
        r"Bearer\s+|nvapi-|sk-[A-Za-z0-9])", re.IGNORECASE)

    def visit(value: object) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                _require(not (isinstance(key, str) and bad_key.fullmatch(key)),
                         "공개 결과 credential key")
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)
        elif isinstance(value, str):
            _require(bad_value.search(value) is None, "공개 결과 private path/credential")
    for value in values:
        visit(value)


def _default_verifiers() -> dict[str, Callable[..., Any]]:
    return {
        "first_candidate": FIRST.checked_candidate,
        "first_protocol": lambda path, anchor, lab: FIRST.verify_protocol(
            path, anchor=anchor, lab=lab),
        "second_candidate": SECOND.checked_second_candidate,
        "second_protocol": lambda path, anchor, lab: SECOND.verify_loop_protocol(
            path, anchor, lab=lab),
        "second_source": lambda value: SECOND.checked_source(value),
        "verify_call": SECOND.TRANSPORT.verify_call,
        "conditional_design": lambda protocol, second_seeds=None:
            FIRST.component_conditional_design(protocol, second_seeds=second_seeds),
    }


def export(first_result: Path, out: Path, *, lab: Path, second_result: Path | None = None,
           allow_fixture: bool = False,
           verifiers: Mapping[str, Callable[..., Any]] | None = None
           ) -> dict[str, Any]:
    """first 결과와 필요한 second 결과를 검증해 네 pipeline 공개 결과를 쓴다."""
    first_path = Path(first_result).resolve(strict=True)
    second_path = Path(second_result).resolve(strict=True) if second_result is not None else None
    out = Path(out).resolve()
    _require(not out.exists(), "public output은 fresh 경로여야 한다")
    if not allow_fixture:
        _require(out.parent == (ROOT / "evals/results").resolve(),
                 "actual 공개 결과는 evals/results/<tag>에 써야 한다")
        _require(verifiers is None, "production verifier injection은 허용하지 않는다")
    verify = dict(_default_verifiers() if verifiers is None else verifiers)
    required = {"first_candidate", "first_protocol", "second_candidate", "second_protocol",
                "second_source", "verify_call", "conditional_design"}
    _require(set(verify) == required, "verifier 집합")
    original_sha = {str(first_path): _sha(first_path)}
    if second_path is not None:
        original_sha[str(second_path)] = _sha(second_path)

    first = _read(first_path)
    _require(first.get("schema") == PARENT.FIRST_RESULT_SCHEMA
             and first.get("status") == "complete" and first.get("automatic_epoch2") is False,
             "first parent result")
    first_plan_path = _bound(first.get("plan"), "first plan")
    first_plan = _read(first_plan_path)
    _require(first_plan.get("schema") == PARENT.FIRST_PLAN_SCHEMA
             and first_plan.get("automatic_epoch2") is False, "first parent plan")
    identities = [(row.get("case_id"), row.get("policy")) for row in first["records"]]
    _require(len(identities) == 4 and set(identities) == set(TARGETS), "first exact4")
    first_protocol_path = _bound(first_plan["recovery_protocol"], "first protocol")
    first_protocol = verify["first_protocol"](
        first_protocol_path, first_plan["anchor"], Path(lab).resolve())
    design_path_value, design = verify["conditional_design"](first_protocol)
    design_path = Path(design_path_value)

    first_by_key: dict[str, dict[str, Any]] = {}
    public_rows: list[dict[str, Any]] = []
    needs_second: set[str] = set()
    source_by_case: dict[str, str] = {}
    for row in first["records"]:
        key = f"{row['case_id']}:{row['policy']}"
        candidate_path = _bound(row["candidate"], "first candidate")
        candidate = verify["first_candidate"](candidate_path)
        _require(candidate["case_id"] == row["case_id"] and candidate["policy"] == row["policy"]
                 and row["registered_seed"] == first_protocol["recovery_seeds"][key],
                 "first candidate/seed")
        source_cp = candidate["source_checkpoint_sha256"]
        prior = source_by_case.setdefault(row["case_id"], source_cp)
        _require(prior == source_cp, "같은 case의 pipeline source checkpoint가 다르다")
        item: dict[str, Any] = {
            "case_id": row["case_id"], "policy": row["policy"],
            "source_arm": candidate["source_arm"], "source_seed": candidate["source_seed"],
            "original_source_checkpoint_sha256": source_cp,
            "first_new_checkpoint_sha256": None,
            "first_registered_seed": row["registered_seed"],
            "first_candidate_status": candidate["status"],
            "first_correction": candidate.get("correction"),
            "first_decision_owner": None, "first_outcome": row["outcome"],
            "first_receipt_sha256": None, "first_training": None,
            "second_required": row["outcome"] == "not_recovered",
            "second_registered_seed": None, "second_candidate_status": None,
            "second_correction": None, "second_decision_owner": None,
            "second_outcome": None, "second_receipt_sha256": None,
            "second_training": None, "second_diagnosis_cost": None,
            "final_outcome": row["outcome"], "hold_reason": candidate.get("hold_reason"),
        }
        if candidate["status"] == "HOLD":
            _require(row["outcome"] == "HOLD" and row.get("result") is None,
                     "first HOLD 삭제·실행 금지")
        else:
            result_path = _bound(row["result"], "first result")
            result = _read(result_path)
            receipt_path = _bound(result.get("receipt"), "first receipt")
            receipt = _read(receipt_path)
            _require(result.get("schema") == FIRST.RESULT_SCHEMA
                     and result.get("status") == "complete"
                     and result.get("outcome") == row["outcome"] == receipt.get("status")
                     and receipt.get("schema") == FIRST.RECEIPT_SCHEMA
                     and receipt.get("candidate") == FIRST.ref(candidate_path)
                     and receipt.get("protocol") == FIRST.ref(first_protocol_path)
                     and receipt.get("registered_seed") == row["registered_seed"],
                     "first result/receipt")
            approval_path = _bound(receipt["approval"], "first approval")
            approval = _approval(
                approval_path, schema=FIRST.APPROVAL_SCHEMA, candidate=candidate_path,
                protocol=first_protocol_path, anchor=first_plan["anchor"],
                seed=row["registered_seed"], lease_id=receipt["lease_id"])
            first_training = _training(
                receipt, candidate, result_path=result_path, lab=Path(lab),
                second=False, allow_fixture=allow_fixture)
            item.update(first_decision_owner=approval["approver"],
                        first_receipt_sha256=_sha(receipt_path),
                        first_new_checkpoint_sha256=first_training["new_checkpoint_sha256"],
                        first_training=first_training)
            if row["outcome"] == "not_recovered":
                epoch2_path = _bound(row["epoch2_input"], "epoch2 input")
                epoch2 = _read(epoch2_path)
                _require(epoch2.get("schema") == FIRST.EPOCH2_SCHEMA
                         and epoch2.get("prior_receipt") == FIRST.ref(receipt_path)
                         and epoch2.get("new_protocol_required") is True
                         and epoch2.get("execution_allowed") is False,
                         "not_recovered epoch2 input")
                needs_second.add(key)
        first_by_key[key] = item
        public_rows.append(item)
    _require(len(set(source_by_case.values())) == 2, "두 source lineage checkpoint 독립성")

    second_parent = None
    second_protocol_public = None
    if needs_second:
        _require(second_path is not None, "not_recovered pipeline의 second 결과가 없다")
    if second_path is not None:
        second_parent = _read(second_path)
        _require(second_parent.get("schema") == PARENT.SECOND_RESULT_SCHEMA
                 and second_parent.get("status") == "complete"
                 and second_parent.get("epoch3_allowed") is False,
                 "second parent result")
        second_plan_path = _bound(second_parent["plan"], "second plan")
        second_plan = _read(second_plan_path)
        _require(second_plan.get("schema") == PARENT.SECOND_PLAN_SCHEMA
                 and _bound(second_plan["first_result"], "second first result") == first_path,
                 "second plan/first result")
        second_protocol_path = _bound(second_plan["loop_protocol"], "second protocol")
        second_protocol = verify["second_protocol"](
            second_protocol_path, second_plan["anchor"], Path(lab).resolve())
        verify["conditional_design"](
            first_protocol, second_seeds=second_protocol["second_seeds"])
        second_protocol_public = _relative_protocol(second_protocol_path,
                                                    allow_fixture=allow_fixture)
        records = second_parent.get("records")
        _require(isinstance(records, list) and len(records) == 4, "second exact4")
        second_identities = [(row.get("case_id"), row.get("policy"))
                             for row in cast(list[dict[str, Any]], records)]
        _require(len(set(second_identities)) == 4 and set(second_identities) == set(TARGETS),
                 "second exact4 unique")
        for row in cast(list[dict[str, Any]], records):
            key = f"{row['case_id']}:{row['policy']}"
            item = first_by_key[key]
            if key not in needs_second:
                _require(row["status"] == "not_in_second_subset"
                         and row["outcome"] == "not_in_second_subset",
                         "불필요한 second 실행")
                continue
            _require(row["status"] == "registered"
                     and row["registered_seed"] == second_protocol["second_seeds"][key],
                     "second source/seed 등록")
            source_path = _bound(row["source"], "second source")
            source = verify["second_source"](_read(source_path))
            _require(second_protocol["source_digests"][key] == SECOND.digest(source),
                     "second source digest")
            diagnosis_path = _bound(row["diagnosis"], "second diagnosis")
            candidate_path = _bound(row["candidate"], "second candidate")
            candidate = verify["second_candidate"](candidate_path)
            _require(candidate["case_id"] == row["case_id"]
                     and candidate["policy"] == row["policy"]
                     and candidate["source_checkpoint_sha256"]
                     == item["first_new_checkpoint_sha256"],
                     "second candidate/source")
            item.update(second_registered_seed=row["registered_seed"],
                        second_candidate_status=candidate["status"],
                        second_correction=candidate.get("second_correction"),
                        second_diagnosis_cost=_model_cost(
                            diagnosis_path, call_verifier=verify["verify_call"]),
                        second_outcome=row["outcome"], final_outcome=row["outcome"],
                        hold_reason=candidate.get("hold_reason"))
            if candidate["status"] == "HOLD":
                _require(row["outcome"] == "HOLD" and row.get("result") is None,
                         "second HOLD 삭제·실행 금지")
                continue
            result_path = _bound(row["result"], "second result")
            result = _read(result_path)
            receipt_path = _bound(result["receipt"], "second receipt")
            receipt = _read(receipt_path)
            _require(result.get("schema") == SECOND.SECOND_RESULT_SCHEMA
                     and result.get("status") == "complete"
                     and result.get("outcome") == row["outcome"] == receipt.get("status")
                     and result.get("epoch3_allowed") is False
                     and receipt.get("schema") == SECOND.SECOND_RECEIPT_SCHEMA
                     and receipt.get("candidate") == FIRST.ref(candidate_path)
                     and receipt.get("protocol") == FIRST.ref(second_protocol_path)
                     and receipt.get("registered_seed") == row["registered_seed"]
                     and receipt.get("corrections")
                     == [item["first_correction"], item["second_correction"]],
                     "second result/receipt")
            approval_path = _bound(receipt["approval"], "second approval")
            approval = _approval(
                approval_path, schema=SECOND.SECOND_APPROVAL_SCHEMA,
                candidate=candidate_path, protocol=second_protocol_path,
                anchor=second_plan["anchor"], seed=row["registered_seed"],
                lease_id=receipt["lease_id"])
            second_training = _training(
                receipt, candidate, result_path=result_path, lab=Path(lab),
                second=True, allow_fixture=allow_fixture)
            item.update(second_decision_owner=approval["approver"],
                        second_receipt_sha256=_sha(receipt_path),
                        second_training=second_training)

    _require(all(_sha(Path(path)) == sha for path, sha in original_sha.items()),
             "입력 결과가 export 중 바뀌었다")
    counts = {name: sum(row["final_outcome"] == name for row in public_rows)
              for name in ("recovered", "not_recovered", "HOLD")}
    summary = {
        "schema": SUMMARY_SCHEMA, "status": "complete",
        "scope": "cpu_fixture_development_measurement" if allow_fixture else "actual_backend",
        "pipeline_denominator": 4, "independent_source_lineages": 2,
        "targets": [f"{case}:{policy}" for case, policy in TARGETS],
        "final_outcomes": counts, "first_parent_wall_s": first.get("parent_wall_s"),
        "second_parent_wall_s": second_parent.get("parent_wall_s") if second_parent else None,
        "second_required_pipelines": len(needs_second),
        "second_executed": second_parent is not None,
        "automatic_retry": False, "replacement_targets": 0, "epoch3_allowed": False,
        "claims": {"generalization": False, "model_superiority": False,
                   "recovery_outside_registered_scope": False,
                   "fixture_promoted_to_actual": False},
    }
    cases = {"schema": CASES_SCHEMA, "status": "complete", "rows": public_rows}
    provenance = {
        "schema": PROVENANCE_SCHEMA, "status": "complete",
        "first_result": _public_ref(first_path),
        "second_result": _public_ref(second_path) if second_path else None,
        "first_protocol": _relative_protocol(first_protocol_path, allow_fixture=allow_fixture),
        "second_protocol": second_protocol_public,
        "conditional_design": _relative_protocol(design_path, allow_fixture=allow_fixture),
        "conditional_design_digest": _digest(design),
        "source_files_lf": {name: _sha(ROOT / name, lf=True) for name in SOURCE_FILES},
    }
    _assert_public(summary, cases, provenance)
    out.mkdir(parents=True)
    for name, value in (("summary.json", summary), ("cases.json", cases),
                        ("provenance.json", provenance)):
        _write_new(out / name, value)
    manifest = {
        "schema": MANIFEST_SCHEMA, "status": "complete",
        "files": {name: _public_ref(out / name) for name in
                  ("summary.json", "cases.json", "provenance.json")},
        "private_paths_in_public_files": False, "overwrite_allowed": False,
    }
    _assert_public(manifest)
    _write_new(out / "manifest.json", manifest)
    return deepcopy(manifest)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-result", type=Path, required=True)
    parser.add_argument("--second-result", type=Path)
    parser.add_argument("--lab", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    export(args.first_result, args.out, lab=args.lab, second_result=args.second_result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
