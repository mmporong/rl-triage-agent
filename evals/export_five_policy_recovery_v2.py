# pyright: reportMissingImports=false
"""중단된 first 실행을 검증 가능한 continuation으로 닫은 R4 결과를 공개 투영한다."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("bench", "evals", "src")]

import export_five_policy_recovery_v1 as V1
import five_policy_recovery_candidate_v1 as FIRST
import five_policy_recovery_epoch2_v1 as SECOND
import recovery_outcome_contract_v1 as OUTCOME
import run_five_policy_recovery_loop_v1 as PARENT
import run_five_policy_recovery_resume_v2 as CONTINUATION

SUMMARY_SCHEMA = V1.SUMMARY_SCHEMA
CASES_SCHEMA = V1.CASES_SCHEMA
PROVENANCE_SCHEMA = V1.PROVENANCE_SCHEMA
MANIFEST_SCHEMA = V1.MANIFEST_SCHEMA
TARGETS = tuple(PARENT.TARGETS)
PRIOR_EXECUTION_KIND = "verified_prior_roles_finalize"
NEW_EXECUTION_KIND = "new_registered_execution"
SOURCE_FILES = (
    "evals/export_five_policy_recovery_v2.py",
    "tests/test_export_five_policy_recovery_v2.py",
    "evals/recovery_outcome_contract_v1.py",
    "tests/test_recovery_outcome_contract_v1.py",
    "bench/run_five_policy_recovery_resume_v2.py",
    "tests/test_five_policy_recovery_resume_v2.py",
)

_require = V1._require
_read = V1._read
_sha = V1._sha
_digest = V1._digest
_write_new = V1._write_new
_bound = V1._bound
_public_ref = V1._public_ref
_relative_protocol = V1._relative_protocol
_approval = V1._approval
_training = V1._training
_model_cost = V1._model_cost
_assert_public = V1._assert_public


def _default_verifiers() -> dict[str, Callable[..., Any]]:
    verifiers = V1._default_verifiers()
    verifiers["continuation_protocol"] = (
        lambda path, continuation_plan_path, execution_anchor, lab:
        CONTINUATION.verify_continuation_protocol(
            path, continuation_plan_path=continuation_plan_path,
            execution_anchor=execution_anchor, lab=lab)
    )
    verifiers["second_compatibility_protocol"] = (
        lambda path, anchor, second_plan_path, host_protocol:
        CONTINUATION.verify_second_compatibility_protocol(
            path, anchor=anchor, second_plan_path=second_plan_path,
            host_protocol=host_protocol)
    )
    return verifiers


def _continuation(
    first_path: Path,
    first: Mapping[str, Any],
    *,
    lab: Path,
    verify: Mapping[str, Callable[..., Any]],
) -> tuple[Path, dict[str, Any], Path, Path, dict[str, Any], dict[str, Any], dict[str, Any]]:
    _require(
        first.get("schema") == CONTINUATION.FIRST_CONTINUATION_RESULT_SCHEMA
        and first.get("status") == "complete"
        and first.get("automatic_epoch2") is False,
        "first continuation parent result",
    )
    first_plan_path = _bound(first.get("plan"), "first plan")
    first_plan = _read(first_plan_path)
    _require(
        first_plan.get("schema") == PARENT.FIRST_PLAN_SCHEMA
        and first_plan.get("automatic_epoch2") is False,
        "first parent plan",
    )
    phase_protocol_path = _bound(first.get("phase_protocol"), "continuation protocol")
    continuation_plan_path = _bound(
        first.get("continuation_plan"), "private continuation plan")
    execution_anchor = first.get("execution_anchor")
    _require(isinstance(execution_anchor, str) and bool(execution_anchor),
             "continuation execution anchor")
    material = dict(verify["continuation_protocol"](
        phase_protocol_path, continuation_plan_path=continuation_plan_path,
        execution_anchor=execution_anchor, lab=Path(lab).resolve()))
    _require(set(material) == {"protocol", "plan"}
             and isinstance(material["protocol"], Mapping)
             and isinstance(material["plan"], Mapping),
             "continuation public/private material")
    overlay = dict(cast(Mapping[str, Any], material["protocol"]))
    private = dict(cast(Mapping[str, Any], material["plan"]))
    _require(overlay.get("private_plan_sha256") == _sha(continuation_plan_path),
             "continuation private plan SHA")
    original = overlay.get("original")
    _require(isinstance(original, Mapping), "continuation original 결합")
    original_row = cast(Mapping[str, Any], original)
    _require(
        private.get("original_plan") == FIRST.ref(first_plan_path)
        and private.get("original_first_protocol") == first_plan.get("recovery_protocol")
        and original_row.get("anchor") == first.get("anchor") == first_plan.get("anchor"),
        "continuation original plan/protocol/anchor 결합",
    )
    records = first.get("records")
    _require(isinstance(records, list), "first continuation records")
    identities = [(row.get("case_id"), row.get("policy"))
                  for row in cast(list[dict[str, Any]], records)]
    _require(len(identities) == 4 and len(set(identities)) == 4
             and set(identities) == set(TARGETS), "first exact4")
    kinds = [row.get("execution_kind") for row in cast(list[dict[str, Any]], records)]
    _require(kinds.count(PRIOR_EXECUTION_KIND) == 1
             and kinds.count(NEW_EXECUTION_KIND) == 3,
             "continuation prior1/new3 실행 종류")
    finalized = overlay.get("prior")
    _require(isinstance(finalized, Mapping), "continuation finalized target")
    finalized_row = cast(Mapping[str, Any], finalized)
    global_lease = first.get("lease_id")
    _require(isinstance(global_lease, str) and bool(global_lease),
             "continuation global lease")
    _require(first.get("lease_id") != finalized_row.get("lease_id"),
             "continuation global lease는 old closed lease와 달라야 한다")
    for row in cast(list[dict[str, Any]], records):
        lease_id = row.get("lease_id")
        _require(isinstance(lease_id, str) and bool(lease_id), "first 행별 lease")
        if row.get("execution_kind") == PRIOR_EXECUTION_KIND:
            _require(
                row.get("case_id") == finalized_row.get("case_id")
                and row.get("policy") == finalized_row.get("policy")
                and row.get("result") == private.get("finalized_result")
                and lease_id == finalized_row.get("lease_id"),
                "continuation prior closed result 재사용 결합",
            )
        else:
            _require(lease_id == first.get("lease_id"),
                     "continuation new 행 lease/global lease 결합")
    old_wall = first.get("old_wall_s")
    new_wall = first.get("new_parent_wall_s")
    total_wall = first.get("total_parent_wall_s")
    _require(first.get("old_sdk_executions") == 2
             and first.get("new_sdk_executions") == 6
             and first.get("total_sdk_execution_upper_bound") == 8
             and first.get("api_calls") == 0
             and first.get("prior_reuse_count") == 1
             and first.get("new_executed_targets") == 3,
             "continuation 실행 예산/재사용 수")
    _require(all(isinstance(value, (int, float)) and not isinstance(value, bool)
                 and float(value) >= 0 for value in (old_wall, new_wall, total_wall))
             and float(cast(float, total_wall))
             == float(cast(float, old_wall)) + float(cast(float, new_wall))
             and float(cast(float, total_wall)) <= CONTINUATION.FIRST_WALL_S,
             "continuation wall 예산")
    _require(first.get("second_phase_requires_new_registered_protocol")
             == any(row.get("outcome") == "not_recovered"
                    for row in cast(list[dict[str, Any]], records)),
             "continuation second 필요성")
    return (first_plan_path, first_plan, phase_protocol_path, continuation_plan_path,
            overlay, private, dict(first))


def _first_training_result_path(
    row: Mapping[str, Any], result_path: Path, receipt: Mapping[str, Any],
    *, private: Mapping[str, Any],
) -> Path:
    """prior CPU finalize receipt를 원 평가 폴더의 strict 검증 경로에 결합한다."""
    if row.get("execution_kind") != PRIOR_EXECUTION_KIND:
        _require(receipt.get("continuation_provenance") is None,
                 "new FIRST receipt에 prior provenance가 있다")
        return result_path
    provenance = receipt.get("continuation_provenance")
    _require(isinstance(provenance, Mapping), "prior continuation provenance")
    provenance_row = cast(Mapping[str, Any], provenance)
    failure_path = _bound(provenance_row.get("original_failure"), "original failure")
    _require(
        provenance_row.get("kind") == PRIOR_EXECUTION_KIND
        and provenance_row.get("original_approval") == receipt.get("approval")
        and provenance_row.get("original_closed_lease") == private.get("old_closed_lease")
        and provenance_row.get("original_raw_inventory")
        == private.get("old_raw_inventory"),
        "prior approval/lease/raw provenance",
    )
    normalization = provenance_row.get("abi_normalization")
    _require(isinstance(normalization, Mapping), "prior ABI normalization provenance")
    normalization_row = cast(Mapping[str, Any], normalization)
    _require(
        normalization_row.get("source")
        == FIRST.ref(ROOT / "evals/recovery_outcome_contract_v1.py")
        and normalization_row.get("canonical_recovery_preserved") is True
        and normalization_row.get("inner_outcome_added_from_outer") is True,
        "prior ABI normalization source",
    )
    return failure_path


def export(
    first_result: Path,
    out: Path,
    *,
    lab: Path,
    second_result: Path | None = None,
    second_compatibility_protocol: Path | None = None,
    second_compatibility_anchor: str | None = None,
    second_host_protocol: Path | None = None,
    allow_fixture: bool = False,
    verifiers: Mapping[str, Callable[..., Any]] | None = None,
) -> dict[str, Any]:
    """v2 continuation과 필요한 second 결과를 검증해 v1 공개 표현을 쓴다."""
    first_path = Path(first_result).resolve(strict=True)
    second_path = Path(second_result).resolve(strict=True) if second_result is not None else None
    second_compat_path = (Path(second_compatibility_protocol).resolve(strict=True)
                          if second_compatibility_protocol is not None else None)
    second_host_path = (Path(second_host_protocol).resolve(strict=True)
                        if second_host_protocol is not None else None)
    out = Path(out).resolve()
    _require(not out.exists(), "public output은 fresh 경로여야 한다")
    if not allow_fixture:
        _require(out.parent == (ROOT / "evals/results").resolve(),
                 "actual 공개 결과는 evals/results/<tag>에 써야 한다")
        _require(verifiers is None, "production verifier injection은 허용하지 않는다")
    verify = dict(_default_verifiers() if verifiers is None else verifiers)
    required = {
        "first_candidate", "first_protocol", "second_candidate", "second_protocol",
        "second_source", "verify_call", "conditional_design", "continuation_protocol",
        "second_compatibility_protocol",
    }
    _require(set(verify) == required, "verifier 집합")
    original_sha = {str(first_path): _sha(first_path)}
    if second_path is not None:
        original_sha[str(second_path)] = _sha(second_path)
        _require(second_compat_path is not None and second_host_path is not None
                 and isinstance(second_compatibility_anchor, str)
                 and bool(second_compatibility_anchor),
                 "second compatibility protocol/anchor가 필요하다")
        original_sha[str(second_compat_path)] = _sha(second_compat_path)
        original_sha[str(second_host_path)] = _sha(second_host_path)
    else:
        _require(second_compat_path is None and second_compatibility_anchor is None
                 and second_host_path is None,
                 "second 결과 없는 compatibility 입력")

    with OUTCOME.recovery_outcome_scope():
        first_raw = _read(first_path)
        (_first_plan_path, first_plan, phase_protocol_path, continuation_plan_path,
         overlay, private, first) = _continuation(
             first_path, first_raw, lab=Path(lab), verify=verify)
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
            _require(candidate["case_id"] == row["case_id"]
                     and candidate["policy"] == row["policy"]
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
                _require(
                    result.get("schema") == FIRST.RESULT_SCHEMA
                    and result.get("status") == "complete"
                    and result.get("outcome") == row["outcome"] == receipt.get("status")
                    and receipt.get("schema") == FIRST.RECEIPT_SCHEMA
                    and receipt.get("candidate") == FIRST.ref(candidate_path)
                    and receipt.get("protocol") == FIRST.ref(first_protocol_path)
                    and receipt.get("registered_seed") == row["registered_seed"]
                    and receipt.get("lease_id") == row["lease_id"],
                    "first result/receipt/행별 lease",
                )
                approval_path = _bound(receipt["approval"], "first approval")
                approval = _approval(
                    approval_path, schema=FIRST.APPROVAL_SCHEMA, candidate=candidate_path,
                    protocol=first_protocol_path, anchor=first_plan["anchor"],
                    seed=row["registered_seed"], lease_id=row["lease_id"])
                training_result_path = _first_training_result_path(
                    row, result_path, receipt, private=private)
                first_training = _training(
                    receipt, candidate, result_path=training_result_path, lab=Path(lab),
                    second=False, allow_fixture=allow_fixture)
                item.update(
                    first_decision_owner=approval["approver"],
                    first_receipt_sha256=_sha(receipt_path),
                    first_new_checkpoint_sha256=first_training["new_checkpoint_sha256"],
                    first_training=first_training,
                )
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
            assert second_compat_path is not None
            assert second_compatibility_anchor is not None
            assert second_host_path is not None
            second_compat = verify["second_compatibility_protocol"](
                second_compat_path, anchor=second_compatibility_anchor,
                second_plan_path=second_plan_path, host_protocol=second_host_path)
            _require(second_compat.get("second_plan_sha256") == _sha(second_plan_path),
                     "second compatibility plan/loop 결합")
            second_protocol_path = _bound(second_plan["loop_protocol"], "second protocol")
            second_protocol = verify["second_protocol"](
                second_protocol_path, second_plan["anchor"], Path(lab).resolve())
            verify["conditional_design"](
                first_protocol, second_seeds=second_protocol["second_seeds"])
            second_protocol_public = _relative_protocol(
                second_protocol_path, allow_fixture=allow_fixture)
            records = second_parent.get("records")
            _require(isinstance(records, list) and len(records) == 4, "second exact4")
            second_ids = [(row.get("case_id"), row.get("policy"))
                          for row in cast(list[dict[str, Any]], records)]
            _require(len(set(second_ids)) == 4 and set(second_ids) == set(TARGETS),
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
                item.update(
                    second_registered_seed=row["registered_seed"],
                    second_candidate_status=candidate["status"],
                    second_correction=candidate.get("second_correction"),
                    second_diagnosis_cost=_model_cost(
                        diagnosis_path, call_verifier=verify["verify_call"]),
                    second_outcome=row["outcome"], final_outcome=row["outcome"],
                    hold_reason=candidate.get("hold_reason"),
                )
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
                approval = _approval(
                    _bound(receipt["approval"], "second approval"),
                    schema=SECOND.SECOND_APPROVAL_SCHEMA, candidate=candidate_path,
                    protocol=second_protocol_path, anchor=second_plan["anchor"],
                    seed=row["registered_seed"], lease_id=receipt["lease_id"])
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
        "final_outcomes": counts,
        "first_parent_wall_s": first.get("total_parent_wall_s"),
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
        "first_result_schema": CONTINUATION.FIRST_CONTINUATION_RESULT_SCHEMA,
        "first_result": _public_ref(first_path),
        "second_result": _public_ref(second_path) if second_path else None,
        "first_protocol": _relative_protocol(first_protocol_path, allow_fixture=allow_fixture),
        "continuation_protocol": _relative_protocol(
            phase_protocol_path, allow_fixture=allow_fixture),
        "continuation_plan_sha256": _sha(continuation_plan_path),
        "second_protocol": second_protocol_public,
        "second_protocol_anchor": second_plan.get("anchor") if second_parent else None,
        "second_compatibility_protocol": (
            _relative_protocol(second_compat_path, allow_fixture=allow_fixture)
            if second_compat_path else None),
        "second_compatibility_anchor": second_compatibility_anchor,
        "second_host_protocol": (
            _relative_protocol(second_host_path, allow_fixture=allow_fixture)
            if second_host_path else None),
        "conditional_design": _relative_protocol(design_path, allow_fixture=allow_fixture),
        "conditional_design_digest": _digest(design),
        "first_source_anchor": first["anchor"],
        "continuation_execution_anchor": first["execution_anchor"],
        "continuation": {
            "old_invalid_phase_preserved": True,
            "prior_completed_leaf_cpu_finalized": True,
            "prior_raw_result_changed": False,
            "prior_gpu_reexecution": False,
            "prior_execution_counted_once": True,
        },
        "continuation_parent_sources_lf": overlay.get("sources_lf"),
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
    parser.add_argument("--second-compatibility-protocol", type=Path)
    parser.add_argument("--second-compatibility-anchor")
    parser.add_argument("--second-host-protocol", type=Path)
    parser.add_argument("--lab", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    export(
        args.first_result, args.out, lab=args.lab, second_result=args.second_result,
        second_compatibility_protocol=args.second_compatibility_protocol,
        second_compatibility_anchor=args.second_compatibility_anchor,
        second_host_protocol=args.second_host_protocol,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
