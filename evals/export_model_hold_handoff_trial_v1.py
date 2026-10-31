"""검증된 model HOLD handoff trial을 paired 원문 재생 뒤 공개한다."""
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

from evals import export_diagnostic_coverage_audit_v1 as COVER
from evals import export_prospective_agentic_comparison_v1 as PRIV
from evals import run_model_hold_handoff_trial_v1 as TRIAL

ROOT = PRIV.ROOT
SUMMARY_SCHEMA = "model_hold_handoff_trial_public_summary_v1"
MEASUREMENT_SCHEMA = "model_hold_handoff_trial_public_measurement_v1"
PROVENANCE_SCHEMA = "model_hold_handoff_trial_public_provenance_v1"
MANIFEST_SCHEMA = "model_hold_handoff_trial_public_manifest_v1"


def _require(ok: bool, reason: str) -> None:
    TRIAL._require(ok, reason)


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    return TRIAL.BASE._read(Path(path))


def _finite(value: object, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"{label}은 유한 수치여야 한다")
    return float(value)


def _checked_host_ref(value: object, execution_root: Path) -> Path:
    _require(isinstance(value, dict) and set(value) == {"path", "sha256"},
             "result host_result ref 형식 불일치")
    ref = cast(dict[str, Any], value)
    path = Path(cast(str, ref["path"])).resolve()
    expected = Path(execution_root).resolve() / "probe" / "host-result.json"
    _require(path == expected and path.is_file() and _sha(path) == ref["sha256"],
             "result host_result 경로/SHA 불일치")
    return path


def _measurement(role: str, receipt: dict[str, Any], checkpoint: str) -> dict[str, Any]:
    value = COVER._measurement(role, receipt, checkpoint)
    value["schema"] = MEASUREMENT_SCHEMA
    return value


def _proposal_authority(proposal: dict[str, Any]) -> dict[str, bool]:
    names = ("approval_granted", "execution_allowed", "correction_allowed", "automatic_resume")
    _require(all(proposal.get(name) is False for name in names),
             "원 proposal 비권한 계약 불일치")
    return {name: False for name in names}


def _verify_execution(result_path: Path, inputs: dict[str, Any]
                      ) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    result_path = Path(result_path).resolve()
    result = _read(result_path)
    expected_action = inputs["proposal"]["proposal"]
    _require(result.get("schema") == TRIAL.RESULT_SCHEMA and result.get("status") == "complete"
             and result.get("case_id") == TRIAL.CASE_ID
             and result.get("reference_case_id") == TRIAL.REFERENCE_ID
             and result.get("selector_actor") == TRIAL.SELECTOR_ACTOR
             and result.get("approval_actor") == TRIAL.APPROVER
             and result.get("original_model_selected_probe") is False
             and result.get("original_model_result_changed") is False
             and result.get("parent_model_outcome") == "HOLD"
             and result.get("proposal") == expected_action
             and result.get("api_requests") == 0 and result.get("paired_probes") == 1
             and result.get("sdk_role_executions") == 2
             and result.get("training") is False and result.get("correction") is False
             and result.get("recovery") is False
             and result.get("new_observation_claim") is False
             and result.get("model_aggregate_changed") is False,
             "hold handoff trial result 고정 계약 불일치")
    _finite(result.get("parent_wall_s"), "trial parent wall_s")
    execution_root = result_path.parent
    host_path = _checked_host_ref(result.get("host_result"), execution_root)
    host = _read(host_path)
    names = ("case_plan", "case_receipt", "reference_plan", "reference_receipt")
    probe_root = execution_root / "probe"
    _require(host.get("schema") == "evidence_guided_probe_host_result_v1"
             and host.get("scope") == "legacy_registered_paired_probe"
             and all(isinstance(host.get(name), str) for name in names),
             "trial host result schema/scope/path 불일치")
    paths = {name: Path(host[name]).resolve() for name in names}
    _require(all(path.is_file() and path.is_relative_to(probe_root) for path in paths.values()),
             "trial paired plan/receipt가 probe root를 이탈했다")
    shadow = cast(dict[str, Any], inputs["shadow"])
    action = cast(dict[str, Any], inputs["action"])
    proof = COVER.LOOP._pair_proof(paths, shadow, action, probe_root)
    approval = _read(Path(proof["root_approval"]["path"]))
    request = approval.get("request_spec")
    _require(approval.get("actor") == TRIAL.APPROVER
             and approval.get("approval_granted") is True
             and approval.get("action") == action
             and approval.get("state_digest") == shadow["state_digest"]
             and isinstance(request, dict) and request.get("action") == action
             and request.get("approval_granted") is False
             and request.get("model_generated_args") is False
             and request.get("model_generated_cause") is False
             and request.get("model_generated_repair") is False,
             "trial root approval actor/action/request 불일치")
    after = TRIAL.SEL.advance(
        shadow, action, case_plan=paths["case_plan"], case_receipt=paths["case_receipt"],
        reference_plan=paths["reference_plan"],
        reference_receipt=paths["reference_receipt"])
    outcome = after.get("observed", {}).get(TRIAL.PROBE)
    _require(outcome == result.get("probe_outcome") == "abnormal"
             and after.get("state_digest") == result.get("state_after_digest"),
             "trial outcome/state digest 재생 불일치")
    source = cast(dict[str, Any], inputs["source"])
    case = cast(dict[str, Any], source["case"])
    reference = cast(dict[str, Any], source["reference"])
    receipts = [_read(paths[f"{role}_receipt"]) for role in ("case", "reference")]
    measurements = [
        _measurement("case", receipts[0], case["checkpoint"]["sha256"]),
        _measurement("reference", receipts[1], reference["checkpoint"]["sha256"]),
    ]
    raw = {"host_result_sha256": _sha(host_path),
           **{f"{name}_sha256": _sha(path) for name, path in paths.items()},
           "root_approval_sha256": proof["root_approval"]["sha256"]}
    return result, measurements, raw


def export(result_path: Path, protocol_path: Path, source_batch: Path, mode_result: Path,
           corpus: Path, epoch_result: Path, study_result: Path, lab: Path, *,
           anchor: str, out: Path) -> dict[str, Any]:
    paths = [Path(value).resolve() for value in (
        result_path, protocol_path, source_batch, mode_result, corpus, epoch_result,
        study_result, lab, out)]
    (result_path, protocol_path, source_batch, mode_result, corpus, epoch_result,
     study_result, lab, out) = paths
    _require(not out.exists(), "public hold handoff out은 fresh 경로여야 한다")
    with TRIAL.CACHE.proof_scope() as cache_stats:
        spec = TRIAL.verify_protocol(protocol_path, anchor, corpus, lab)
        audit_path = ROOT / spec["prior_coverage_audit"]["path"]
        inputs = TRIAL.inspect_inputs(
            ROOT / spec["parent_reasoning_protocol"]["path"], source_batch, mode_result,
            corpus, epoch_result, study_result, audit_path, lab)
        proposal_authority = _proposal_authority(inputs["proposal"])
        result, measurements, raw = _verify_execution(result_path, inputs)
    out.mkdir(parents=True, exist_ok=False)
    files: dict[str, dict[str, str]] = {}

    def write(name: str, value: object, kind: str) -> None:
        PRIV._safe_public(value)
        TRIAL.BASE._write(out / name, value)
        files[name] = {"sha256": _sha(out / name), "mode": kind}

    for measurement in measurements:
        write(f"measurements/{measurement['role']}.json", measurement,
              "verified_numeric_measurement")
    summary = {
        "schema": SUMMARY_SCHEMA, "status": "complete", "case_id": TRIAL.CASE_ID,
        "reference_case_id": TRIAL.REFERENCE_ID, "parent_model_outcome": "HOLD",
        "parent_model_result_changed": False, "selector_actor": TRIAL.SELECTOR_ACTOR,
        "approval_actor": TRIAL.APPROVER, "original_model_selected_probe": False,
        "proposal": result["proposal"], "probe_outcome": result["probe_outcome"],
        "api_requests": 0, "paired_probes": 1, "sdk_role_executions": 2,
        "parent_wall_s": result["parent_wall_s"],
        "proposal_authority": proposal_authority,
        "scope": {"post_exposure_development": True,
                  "same_source_prior_audit_already_observed": True,
                  "integration_trial_not_independent_case": True,
                  "automatic_resume": False},
        "claims": {"new_observation": False, "model_aggregate_improvement": False,
                   "new_independent_case": False, "new_training": False,
                   "correction": False, "recovery": False,
                   "model_improvement": False, "speed_advantage": False},
    }
    write("summary.json", summary, "derived_summary")
    protocol_relative = protocol_path.relative_to(ROOT).as_posix()
    provenance = {
        "schema": PROVENANCE_SCHEMA, "source_anchor": anchor,
        "source_mapping": {"protocol": {"path": protocol_relative,
                                           "sha256": _sha(protocol_path)},
                           "formatter_sha256": _sha(Path(__file__))},
        "result_sha256": _sha(result_path), "source_batch_sha256": _sha(source_batch),
        "mode_result_sha256": _sha(mode_result), "corpus_sha256": _sha(corpus),
        "epoch_result_sha256": _sha(epoch_result), "study_result_sha256": _sha(study_result),
        "prior_audit_summary_sha256": _sha(audit_path), "raw_proof_sha256": raw,
        "cache_scope": deepcopy(cache_stats),
        "source_checkpoints": {row["role"]: row["checkpoint_sha256"]
                               for row in measurements},
        "privacy": {"private_paths": False, "raw_process_logs": False},
    }
    write("source_provenance.json", provenance, "derived_provenance")
    manifest = {"schema": MANIFEST_SCHEMA, "files": files,
                "artifact_inventory": sorted(files),
                "summary": {"path": "summary.json",
                            "sha256": files["summary.json"]["sha256"]},
                "privacy": {"private_paths": False, "raw_process_logs": False}}
    PRIV._safe_public(manifest)
    TRIAL.BASE._write(out / "artifact_manifest.json", manifest)
    actual = {path.relative_to(out).as_posix() for path in out.rglob("*") if path.is_file()}
    _require(actual == set(files) | {"artifact_manifest.json"}, "manifest 누락/추가 파일")
    _require(all(_sha(out / name) == ref["sha256"] for name, ref in files.items()),
             "manifest artifact SHA 재검증 실패")
    return deepcopy(manifest)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-batch", type=Path, required=True)
    parser.add_argument("--mode-result", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--epoch-result", type=Path, required=True)
    parser.add_argument("--study-result", type=Path, required=True)
    parser.add_argument("--lab", type=Path, required=True)
    parser.add_argument("--anchor", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    export(args.result, args.protocol, args.source_batch, args.mode_result, args.corpus,
           args.epoch_result, args.study_result, args.lab, anchor=args.anchor, out=args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["export", "main"]
