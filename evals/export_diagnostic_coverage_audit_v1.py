"""Operator registry coverage audit를 paired 원문 재생 뒤 간결하게 공개한다."""
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

from evals import export_prospective_agentic_comparison_v1 as PRIV
from evals import registry_action_loop_v1 as LOOP
from evals import run_diagnostic_coverage_audit_v1 as AUDIT

ROOT = PRIV.ROOT
SUMMARY_SCHEMA = "diagnostic_coverage_audit_public_summary_v1"
MEASUREMENT_SCHEMA = "diagnostic_coverage_audit_public_measurement_v1"
PROVENANCE_SCHEMA = "diagnostic_coverage_audit_public_provenance_v1"
MANIFEST_SCHEMA = "diagnostic_coverage_audit_public_manifest_v1"


def _require(ok: bool, reason: str) -> None:
    AUDIT._require(ok, reason)


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    return AUDIT.BASE._read(Path(path))


def _finite(value: object, label: str) -> float:
    _require(type(value) in (int, float) and math.isfinite(cast(float, value)),
             f"{label}은 유한 수치여야 한다")
    return float(cast(float, value))


def _ref(path: Path) -> dict[str, str]:
    return AUDIT.BASE._ref(Path(path).resolve())


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
    measurement = receipt.get("measurement")
    _require(receipt.get("schema") == "runtime_observation_probe_receipt_v1"
             and receipt.get("status") == "complete" and receipt.get("probe") == AUDIT.PROBE
             and receipt.get("role") == role
             and receipt.get("checkpoint_sha256") == checkpoint
             and isinstance(measurement, dict), f"{role} receipt 측정/CP 결합 불일치")
    checked = cast(dict[str, Any], measurement)
    _require(set(checked) == {"track_lin_vel_xy_exp_rel_error",
                             "track_ang_vel_z_exp_rel_error"},
             f"{role} P_reward measurement shape 불일치")
    numeric = {name: _finite(value, f"{role}.{name}") for name, value in checked.items()}
    return {"schema": MEASUREMENT_SCHEMA, "role": role, "probe": AUDIT.PROBE,
            "checkpoint_sha256": checkpoint, "measurement": numeric,
            "new_training": False, "correction": False, "recovery": False}


def _verify_execution(result_path: Path, inputs: dict[str, Any]
                      ) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    result_path = Path(result_path).resolve()
    result = _read(result_path)
    _require(result.get("schema") == AUDIT.RESULT_SCHEMA and result.get("status") == "complete"
             and result.get("case_id") == AUDIT.CASE_ID
             and result.get("reference_case_id") == AUDIT.REFERENCE_ID
             and result.get("selection_actor") == AUDIT.APPROVER
             and result.get("model_selected_probe") is False
             and result.get("original_model_outcome") == "HOLD"
             and result.get("probe") == AUDIT.PROBE
             and result.get("api_requests") == 0 and result.get("paired_probes") == 1
             and result.get("sdk_role_executions") == 2
             and result.get("training") is False and result.get("correction") is False
             and result.get("recovery") is False
             and result.get("other_probe_outcomes_inferred") is False,
             "coverage audit result 고정 계약 불일치")
    _finite(result.get("parent_wall_s"), "audit parent wall_s")
    execution_root = result_path.parent
    host_path = _checked_host_ref(result.get("host_result"), execution_root)
    host = _read(host_path)
    probe_root = execution_root / "probe"
    names = ("case_plan", "case_receipt", "reference_plan", "reference_receipt")
    _require(host.get("schema") == "evidence_guided_probe_host_result_v1"
             and host.get("scope") == "legacy_registered_paired_probe"
             and all(isinstance(host.get(name), str) for name in names),
             "coverage host result schema/scope/path 불일치")
    paths = {name: Path(host[name]).resolve() for name in names}
    _require(all(path.is_file() and path.is_relative_to(probe_root) for path in paths.values()),
             "coverage paired plan/receipt가 probe root를 이탈했다")
    first = cast(dict[str, Any], inputs["first"])
    action = cast(dict[str, Any], inputs["action"])
    proof = LOOP._pair_proof(paths, first["state"], action, probe_root)
    approval = _read(Path(proof["root_approval"]["path"]))
    _require(approval.get("actor") == AUDIT.APPROVER
             and approval.get("approval_granted") is True
             and approval.get("action") == action,
             "coverage root approval actor/action 불일치")
    after = AUDIT.SEL.advance(
        first["state"], action, case_plan=paths["case_plan"],
        case_receipt=paths["case_receipt"], reference_plan=paths["reference_plan"],
        reference_receipt=paths["reference_receipt"])
    outcome = after.get("observed", {}).get(AUDIT.PROBE)
    _require(outcome == result.get("outcome") == "abnormal"
             and result.get("interpretation") == AUDIT._interpret("abnormal")
             and after.get("state_digest") == result.get("state_after_digest"),
             "coverage outcome/state digest 재생 불일치")
    case = cast(dict[str, Any], inputs["case"])
    reference = cast(dict[str, Any], inputs["reference"])
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
           corpus: Path, parent_epoch: Path, lab: Path, *, anchor: str, out: Path) -> dict[str, Any]:
    paths = [Path(value).resolve() for value in (result_path, protocol_path, source_batch,
                                                  mode_result, corpus, parent_epoch, lab, out)]
    (result_path, protocol_path, source_batch, mode_result, corpus,
     parent_epoch, lab, out) = paths
    _require(not out.exists(), "public coverage audit out은 fresh 경로여야 한다")
    with AUDIT.CACHE.proof_scope() as cache_stats:
        spec = AUDIT.verify_protocol(protocol_path, anchor, corpus, lab)
        inputs = AUDIT.inspect_inputs(ROOT / spec["parent_reasoning_protocol"]["path"],
                                      source_batch, mode_result, corpus, parent_epoch, lab)
        result, measurements, raw = _verify_execution(result_path, inputs)
    out.mkdir(parents=True, exist_ok=False)
    files: dict[str, dict[str, str]] = {}

    def write(name: str, value: object, kind: str) -> None:
        PRIV._safe_public(value)
        AUDIT.BASE._write(out / name, value)
        files[name] = {"sha256": _sha(out / name), "mode": kind}

    for measurement in measurements:
        write(f"measurements/{measurement['role']}.json", measurement,
              "verified_numeric_measurement")
    summary = {"schema": SUMMARY_SCHEMA, "status": "complete",
        "case_id": AUDIT.CASE_ID, "reference_case_id": AUDIT.REFERENCE_ID,
        "probe": AUDIT.PROBE, "outcome": result["outcome"],
        "selection_actor": AUDIT.APPROVER, "model_selected_probe": False,
        "original_model_outcome": "HOLD", "original_model_result_changed": False,
        "registered_prior_prediction": "normal", "registered_prior_prediction_supported": False,
        "observed_counterevidence": "abnormal", "api_requests": 0,
        "paired_probes": 1, "sdk_role_executions": 2,
        "parent_wall_s": result["parent_wall_s"],
        "claims": {"other_probe_outcomes_inferred": False, "new_training": False,
                   "correction": False, "recovery": False,
                   "model_selection_failure": False, "diagnosis_improvement": False,
                   "speed_advantage": False}}
    write("summary.json", summary, "derived_summary")
    protocol_relative = protocol_path.relative_to(ROOT).as_posix()
    provenance = {"schema": PROVENANCE_SCHEMA, "creator_sha256": _sha(Path(__file__)),
        "source_anchor": anchor,
        "protocol": {"path": protocol_relative, "sha256": _sha(protocol_path)},
        "result_sha256": _sha(result_path), "source_batch_sha256": _sha(source_batch),
        "mode_result_sha256": _sha(mode_result), "corpus_sha256": _sha(corpus),
        "parent_epoch_sha256": _sha(parent_epoch), "raw_proof_sha256": raw,
        "cache_scope": deepcopy(cache_stats),
        "source_checkpoints": {item["role"]: item["checkpoint_sha256"]
                               for item in measurements},
        "privacy": {"private_paths": False, "raw_process_logs": False}}
    write("source_provenance.json", provenance, "derived_provenance")
    manifest = {"schema": MANIFEST_SCHEMA, "files": files,
        "artifact_inventory": sorted(files),
        "summary": {"path": "summary.json", "sha256": files["summary.json"]["sha256"]},
        "privacy": {"private_paths": False, "raw_process_logs": False}}
    PRIV._safe_public(manifest)
    AUDIT.BASE._write(out / "artifact_manifest.json", manifest)
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
    parser.add_argument("--parent-epoch", type=Path, required=True)
    parser.add_argument("--lab", type=Path, required=True)
    parser.add_argument("--anchor", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    export(args.result, args.protocol, args.source_batch, args.mode_result, args.corpus,
           args.parent_epoch, args.lab, anchor=args.anchor, out=args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["export", "main"]
