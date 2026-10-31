"""등록 option 기반 단일 recovery 실행을 검증해 경로 없는 공개 결과로 내보낸다."""
# pyright: reportMissingImports=false
from __future__ import annotations

import argparse
import hashlib
import math
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

BOOTSTRAP_ROOT = Path(__file__).resolve().parents[1]
if str(BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(BOOTSTRAP_ROOT))

from evals import evidence_guided_recovery_v1 as R
from evals import export_prospective_agentic_comparison_v1 as PRIV

ROOT = PRIV.ROOT
SUMMARY_SCHEMA = "registry_action_recovery_public_summary_v1"
MANIFEST_SCHEMA = "registry_action_recovery_public_manifest_v1"
SOURCE_SCHEMA = "registry_action_recovery_public_source_v1"
METRICS = ("lin_vel_rmse_mps", "yaw_rate_rmse_radps", "mean_survival_s", "fall_rate")


def _require(ok: bool, reason: str) -> None:
    PRIV._require(ok, reason)


def _protocol(path: Path, anchor: str) -> dict[str, Any]:
    """runtime/TF tree 없이 현재 공개 source와 immutable anchor blob을 검증한다."""
    value = R.read(path)
    study_ref = value.get("selection_study")
    _require(value.get("schema") == R.PROTOCOL_SCHEMA and isinstance(study_ref, dict),
             "recovery protocol schema 불일치")
    study = ROOT / cast(dict[str, str], study_ref)["path"]
    expected = R.prepare_protocol(study, value.get("runtime_source_sha256_lf", {}))
    _require(value == expected, "recovery protocol이 현재 source closure와 다르다")
    R._git("cat-file", "-e", f"{anchor}^{{commit}}")
    R._git("merge-base", "--is-ancestor", anchor, "HEAD")
    relative = path.resolve().relative_to(ROOT).as_posix()
    for name, digest in {relative: R.sha(path, lf=True), **value["source_sha256_lf"]}.items():
        blob = R._git("show", f"{anchor}:{name}").replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        _require(hashlib.sha256(blob).hexdigest() == digest,
                 f"recovery anchor source blob 불일치: {name}")
    return value


def _approval(candidate_path: Path, approval_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    candidate = R.verify_candidate(candidate_path)
    approval = R.read(approval_path)
    _require(approval.get("schema") == R.APPROVAL_SCHEMA
             and approval.get("approval_granted") is True
             and approval.get("candidate") == R.ref(candidate_path)
             and approval.get("correction_id") == R.CORRECTION
             and approval.get("recovery_seed") == R.RECOVERY_SEED
             and isinstance(approval.get("actor"), str) and bool(approval["actor"].strip()),
             "candidate/approval 결합 불일치")
    return candidate, approval


def _owned(value: object, label: str) -> dict[str, Any]:
    _require(isinstance(value, dict) and value.get("exit_code") == 0
             and value.get("natural_exit") is True and value.get("cleanup_verified") is True
             and value.get("timed_out") is False, f"{label} owned cleanup 불일치")
    return cast(dict[str, Any], value)


def _checked_ref(ref: object, root: Path, label: str) -> tuple[Path, dict[str, Any]]:
    _require(isinstance(ref, dict) and set(ref) == {"path", "sha256"}, f"{label} ref shape")
    ref = cast(dict[str, str], ref)
    path = Path(ref["path"]).resolve()
    _require(path.is_relative_to(root.resolve()) and path.is_file() and R.sha(path) == ref["sha256"],
             f"{label} ref/SHA/root 불일치")
    return path, R.read(path)


def _training(ref: object, root: Path, *, stage: str, protocol: dict[str, Any],
              protocol_path: Path, anchor: str) -> dict[str, Any]:
    path, value = _checked_ref(ref, root, f"{stage} training")
    manifest_path, manifest = _checked_ref(value.get("manifest"), root, f"{stage} manifest")
    _require(manifest_path == path.parent / "train/manifest.json", f"{stage} manifest 경로 불일치")
    fixed = R.TRAIN.TRAIN.validate_manifest(manifest)
    _require(fixed == manifest, "manifest validation 불일치")
    _require(manifest.get("protocol_path") == protocol_path.resolve().relative_to(ROOT).as_posix()
             and manifest.get("protocol_sha256_lf") == R.sha(protocol_path, lf=True)
             and manifest.get("source_commit_sha") == anchor,
             f"{stage} protocol/anchor manifest 결합 불일치")
    if stage == "smoke":
        _require(manifest.get("mode") == "inspection" and manifest.get("arm") == "NONE"
                 and manifest.get("seed") == R.SMOKE_SEED and manifest.get("max_iterations") == 3,
                 "smoke manifest 불일치")
    else:
        _require(manifest.get("mode") == "train" and manifest.get("arm") == "FIRST"
                 and manifest.get("seed") == R.RECOVERY_SEED
                 and manifest.get("max_iterations") == 300
                 and manifest.get("original_faults") == ["R1"]
                 and manifest.get("applied_corrections") == [R.CORRECTION]
                 and manifest.get("source_checkpoint_sha256") is not None,
                 "repair R1-only fresh manifest 불일치")
    _owned(value.get("execution"), f"{stage} training")
    _, evidence = _checked_ref(value.get("evidence"), root, f"{stage} evidence")
    checkpoint = value.get("checkpoint")
    _require(isinstance(checkpoint, dict) and evidence.get("checkpoint") == checkpoint,
             f"{stage} checkpoint/evidence 불일치")
    checkpoint = cast(dict[str, Any], checkpoint)
    _require(evidence.get("fresh_initialization") is True
             and evidence.get("checkpoint_loaded") is False
             and evidence.get("resume_requested") is False,
             f"{stage}는 fresh initialization이어야 한다")
    telemetry_sha: str | None = None
    if stage == "repair":
        runtime = value.get("runtime_facts", {}).get("case")
        _require(isinstance(runtime, dict), "repair runtime fact ref가 없다")
        runtime_path, _raw = _checked_ref(runtime, root, "repair runtime facts")
        envelope, facts = R.FACT.checked(
            runtime_path, expected_checkpoint_sha256=checkpoint["sha256"])
        _require(facts["runtime:case:updates_observed"] == 3, "repair runtime updates가 3이 아니다")
        for suffix in ("expected_optimizer_steps", "optimizer_steps_min", "optimizer_steps_max",
                       "expected_minibatches", "minibatches_min", "minibatches_max"):
            _require(facts[f"runtime:case:{suffix}"] == 20, "repair first3 runtime이 20/20이 아니다")
        _require(envelope["source_binding"]["checkpoint_sha256"] == checkpoint["sha256"],
                 "repair runtime checkpoint 불일치")
        collected_ref = value.get("collected")
        collected_path, _collected_raw = _checked_ref(collected_ref, root, "repair collected")
        collected, _metadata = R.TRAIN.INPUT.checked(collected_path)
        binding = collected.get("source_binding", {})
        _require(binding.get("checkpoint_sha256") == checkpoint["sha256"]
                 and isinstance(binding.get("telemetry_sha256"), str),
                 "repair collected CP/telemetry binding 불일치")
        telemetry_sha = binding["telemetry_sha256"]
    return {"path": path, "value": value, "manifest": manifest,
            "checkpoint": cast(dict[str, Any], checkpoint), "evidence": evidence,
            "telemetry_sha256": telemetry_sha}


def _ledger(path: Path, receipt: dict[str, Any]) -> dict[str, Any]:
    ledger = R.CorrectionLedger(path, R._registry())
    events = ledger.events()
    _require([event.get("event") for event in events]
             == ["proposed", "approved", "consumed", "receipt"],
             "correction ledger 사건 순서 불일치")
    request_ids = {event.get("request_id") for event in events}
    _require(len(request_ids) == 1 and receipt.get("request_id") in request_ids,
             "ledger/receipt request 결합 불일치")
    proposed, approved, consumed, final = events
    _require(proposed.get("confirmed_mechanism") == R.COMPONENT
             and proposed.get("correction_id") == R.CORRECTION
             and proposed.get("recovery_mode") == R.RECOVERY_MODE
             and bool(approved.get("approver"))
             and consumed.get("request_id") == proposed.get("request_id")
             and final.get("result_sha") == R.sha(Path(receipt["_path"]))
             and final.get("verdict") == receipt.get("recovery", {}).get("outcome"),
             "ledger proposed→receipt 결합 불일치")
    gpu_s = final.get("gpu_s")
    _require(type(gpu_s) in (int, float) and math.isfinite(cast(float, gpu_s))
             and cast(float, gpu_s) >= 0
             and gpu_s == receipt.get("gpu_s"), "ledger/receipt parent wall 불일치")
    return {"events": events, "request_id": proposed["request_id"], "parent_wall_s": gpu_s}


def _metrics(report: dict[str, Any]) -> dict[str, float]:
    metrics = report.get("metrics")
    _require(isinstance(metrics, dict) and len(report.get("by_condition", [])) == 26,
             "fixed26 report shape 불일치")
    metrics = cast(dict[str, Any], metrics)
    result: dict[str, float] = {}
    for name in METRICS:
        value = metrics.get(name)
        _require(type(value) in (int, float) and math.isfinite(cast(float, value)),
                 f"fixed metric {name} 불일치")
        result[name] = float(cast(float, value))
    return result


def export(protocol_path: Path, anchor: str, candidate_path: Path, approval_path: Path,
           source_batch: Path, study_result: Path, execution_result: Path, *, out: Path) -> dict[str, Any]:
    protocol_path, candidate_path = Path(protocol_path).resolve(), Path(candidate_path).resolve()
    approval_path, source_batch = Path(approval_path).resolve(), Path(source_batch).resolve()
    study_result = Path(study_result).resolve()
    execution_result, out = Path(execution_result).resolve(), Path(out).resolve()
    _require(not out.exists(), "public recovery out은 fresh 경로여야 한다")
    _require(execution_result.is_file(), "complete recovery result가 없다")
    with R.RUN.COMP.CACHE.proof_scope() as cache_stats:
        protocol = _protocol(protocol_path, anchor)
        view = ROOT / R.read(R.COMPARISON_PROTOCOL)["protocols"]["view"]["path"]
        with R.RUN.RV.view_scope(view, anchor):
            candidate, _approval_value = _approval(candidate_path, approval_path)
        _require(candidate.get("diagnosis", {}).get("selection_subject") == "model_option_id"
                 and candidate.get("repair", {}).get("selected_by") == "registered_host_rule",
                 "candidate selection/repair source 불일치")
        study = R.read(study_result)
        study_rows = study.get("rows")
        _require(study.get("schema") == R.STUDY.RESULT_SCHEMA and study.get("status") == "complete"
                 and isinstance(study_rows, list) and len(study_rows) == 16
                 and study.get("planned_rows") == 16 and study.get("model_requests") == 14,
                 "selection study 16행/14요청 원문 불일치")
        selected = [row for row in cast(list[dict[str, Any]], study_rows)
                    if row.get("case_id") == R.CASE_ID and row.get("policy") == R.POLICY]
        _require(len(selected) == 1 and selected[0].get("decision_outcome") == "confirmed"
                 and selected[0].get("confirmed_family") == R.COMPONENT,
                 "selection study c03/eager 확정행 불일치")
        epoch_ref = candidate.get("diagnosis", {}).get("epoch_result", {})
        _require(Path(epoch_ref.get("path", "")).resolve().is_relative_to(study_result.parent),
                 "candidate epoch가 selection study root를 이탈했다")
        cases = R.RUN._source_cases(source_batch, R.read(R.COMPARISON_PROTOCOL), R.COMPARISON_ANCHOR)
        case = next(row for row in cases if row["case_id"] == R.CASE_ID)
        reference = next(row for row in cases if row["seed"] == R.SOURCE_SEED
                         and row["category"] == "NONE")
        root = execution_result.parent
        result = R.read(execution_result)
        _require(result.get("schema") == R.RESULT_SCHEMA and result.get("status") == "complete"
                 and result.get("source_seed") == R.SOURCE_SEED
                 and result.get("recovery_seed") == R.RECOVERY_SEED
                 and result.get("model_requests") == 0
                 and result.get("corrected_seed_is_independent") is True
                 and result.get("baseline_reused_only_as_reference") is True,
                 "complete recovery result 계약 불일치")
        stages = result.get("stages")
        _require(isinstance(stages, list) and [row.get("stage") for row in stages]
                 == ["smoke", "repair"], "smoke/repair stage 순서 불일치")
        stages = cast(list[dict[str, Any]], stages)
        _smoke = _training(stages[0]["training"], root, stage="smoke",
                           protocol=protocol, protocol_path=protocol_path, anchor=anchor)
        repair = _training(stages[1]["training"], root, stage="repair",
                           protocol=protocol, protocol_path=protocol_path, anchor=anchor)
        repairs = result.get("repairs")
        _require(isinstance(repairs, list) and len(repairs) == 1, "repair receipt 수 불일치")
        repairs = cast(list[dict[str, str]], repairs)
        receipt_path, receipt = _checked_ref(repairs[0], root, "repair receipt")
        receipt["_path"] = str(receipt_path)
        corrected_path, corrected = _checked_ref(receipt.get("corrected_report"), root,
                                                 "corrected fixed report")
        reference_path = Path(receipt.get("reference_report", {}).get("path", "")).resolve()
        _require(R.ref(reference_path) == receipt.get("reference_report")
                 and reference_path == Path(reference["fixed"]).resolve(),
                 "same-seed NONE reference ref 불일치")
        reference_report = R.read(reference_path)
        source_report = R.read(Path(case["fixed"]))
        recovery = R.C.D1.recovery(corrected, reference_report, R.read(R.RUN.BANDS)["bands"])
        _require(receipt.get("recovery") == recovery and result.get("outcome") == recovery["outcome"],
                 "fixed26 recovery raw 재계산 불일치")
        _require(receipt.get("source_checkpoint_sha256") == case["checkpoint"]["sha256"]
                 and receipt.get("reference_checkpoint_sha256") == reference["checkpoint"]["sha256"]
                 and receipt.get("output_checkpoint_sha256") == repair["checkpoint"]["sha256"]
                 and receipt.get("checkpoint_loaded") is False
                 and receipt.get("recovery_mode") == "fresh_restart"
                 and receipt.get("runtime_first3_expected_observed") == "20/20",
                 "receipt CP/restart/runtime 결합 불일치")
        ledger_path = Path(result.get("correction_ledger", {}).get("path", "")).resolve()
        _require(R.ref(ledger_path) == result.get("correction_ledger"), "ledger ref 불일치")
        ledger = _ledger(ledger_path, receipt)
        _owned(receipt.get("fixed_execution"), "fixed26")
        _require(result.get("claims", {}).get("recovery") == (recovery["outcome"] == "recovered"),
                 "result recovery claim이 measured outcome과 다르다")
    out.mkdir(parents=True, exist_ok=False)
    reports = {"source_r1_s2320": source_report, "reference_none_s2320": reference_report,
               "corrected_s3332": corrected}
    public_reports: dict[str, dict[str, Any]] = {}
    files: dict[str, dict[str, str]] = {}
    for name, report in reports.items():
        public = {"schema": SOURCE_SCHEMA, "name": name, "metrics": _metrics(report),
                  "checkpoint_sha256": report.get("checkpoint", {}).get("sha256"),
                  "raw_sha256": R.sha(corrected_path if name == "corrected_s3332" else
                                       Path(case["fixed"] if name == "source_r1_s2320" else reference["fixed"]))}
        PRIV._safe_public(public)
        target = out / "reports" / f"{name}.json"
        PRIV._write(target, public)
        files[target.relative_to(out).as_posix()] = {"sha256": PRIV._sha(target), "mode": "normalized_fixed26"}
        public_reports[name] = public
    summary = {
        "schema": SUMMARY_SCHEMA, "status": "complete", "outcome": recovery["outcome"],
        "selection": {"case_id": R.CASE_ID, "source_seed": R.SOURCE_SEED,
                      "policy": R.POLICY, "selected_by": "model_option_id",
                      "option_id": R.OPTION_ID, "component": R.COMPONENT},
        "repair": {"correction_id": R.CORRECTION, "selected_by": "registered_host_rule",
                   "mode": "fresh_restart", "recovery_seed": R.RECOVERY_SEED,
                   "independent_recovery_seed": True, "source_checkpoint_provenance_only": True,
                   "checkpoint_loaded": False},
        "reports": public_reports, "recovery": recovery,
        "runtime_first3": {"updates": 3, "expected_steps": 20, "observed_steps": 20,
                           "expected_minibatches": 20, "observed_minibatches": 20},
        "timing": {"parent_wall_s": ledger["parent_wall_s"],
                   "scope": "registered recovery parent interval; not CUDA kernel time"},
        "study_separation": {"selection_study_nim_requests": 14,
                             "recovery_api_requests": 0,
                             "healthy_reference_counted_as_new_recovery": False},
        "claims": {"post_exposure_development": True, "single_case": True,
                   "measured_single_case_recovered": recovery["outcome"] == "recovered",
                   "general_recovery": False,
                   "gpu_kernel_savings": False, "early_detection": False,
                   "model_version_advantage": False, "default_adoption": False},
    }
    PRIV._safe_public(summary)
    PRIV._write(out / "summary.json", summary)
    files["summary.json"] = {"sha256": PRIV._sha(out / "summary.json"), "mode": "verified_summary"}
    provenance = {"schema": "registry_action_recovery_public_provenance_v1",
                  "anchor": anchor, "protocol_sha256": R.sha(protocol_path),
                  "candidate_sha256": R.sha(candidate_path), "approval_sha256": R.sha(approval_path),
                  "source_batch_sha256": R.sha(source_batch),
                  "selection_study_sha256": R.sha(study_result),
                  "result_sha256": R.sha(execution_result),
                  "study_protocol_sha256": R.sha(ROOT / protocol["selection_study"]["path"]),
                  "source_checkpoint_sha256": case["checkpoint"]["sha256"],
                  "reference_checkpoint_sha256": reference["checkpoint"]["sha256"],
                  "output_checkpoint_sha256": repair["checkpoint"]["sha256"],
                  "output_telemetry_sha256": repair["telemetry_sha256"],
                  "immutable_git_blob_cache": deepcopy(cache_stats),
                  "mutable_checks_cached": False}
    PRIV._safe_public(provenance)
    PRIV._write(out / "source_provenance.json", provenance)
    files["source_provenance.json"] = {"sha256": PRIV._sha(out / "source_provenance.json"),
                                       "mode": "verified_provenance"}
    manifest = {"schema": MANIFEST_SCHEMA, "files": files,
                "summary": {"path": "summary.json", "sha256": PRIV._sha(out / "summary.json")},
                "provenance": {"path": "source_provenance.json",
                               "sha256": PRIV._sha(out / "source_provenance.json")},
                "privacy": {"checkpoint_binaries": False, "training_stdout": False,
                            "private_paths": False, "secrets": False}}
    PRIV._safe_public(manifest)
    PRIV._write(out / "artifact_manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--anchor", required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--approval", type=Path, required=True)
    parser.add_argument("--source-batch", type=Path, required=True)
    parser.add_argument("--study-result", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    export(args.protocol, args.anchor, args.candidate, args.approval,
           args.source_batch, args.study_result, args.result, out=args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["export", "main"]
