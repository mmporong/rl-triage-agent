"""V3 continuation의 완료 결과를 기존 검증 export와 V3 provenance로 공개한다."""
# pyright: reportMissingImports=false
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "src")]

import export_prospective_reasoning_confirmation_v1 as BASE  # noqa: E402
import run_prospective_reasoning_confirmation_v3 as V3  # noqa: E402

SUMMARY_SCHEMA = "prospective_reasoning_confirmation_release_summary_v2"
PROVENANCE_SCHEMA = "prospective_reasoning_confirmation_release_provenance_v2"
MANIFEST_SCHEMA = "prospective_reasoning_confirmation_release_manifest_v2"


def require(ok: object, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON object가 아니다: {Path(path).name}")
    return cast(dict[str, Any], value)


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _public_ref(path: Path) -> dict[str, str]:
    path = Path(path).resolve()
    require(path.is_file() and path.is_relative_to(ROOT), "공개 ref 경로가 아니다")
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": _sha(path)}


def _write_new(path: Path, value: object) -> None:
    BASE._safe(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def _failure_summary(value: dict[str, Any]) -> None:
    require(value.get("schema") == "prospective_confirmation_execution_failure_public_v1"
            and value.get("status") == "incomplete_execution_failure"
            and value.get("assessment") == "indeterminate"
            and value.get("planned_rows") == 24 and value.get("completed_rows") == 0
            and value.get("model_requests_consumed") == 1
            and value.get("provider_responses_preserved") == 1
            and value.get("paired_probes_completed") == 0
            and value.get("sdk_probe_executions") == 0
            and value.get("failure", {}).get("category")
            == "harness_callback_contract_mismatch"
            and value.get("first_proposal") == {
                "action": "run_probe", "case_id": "c02", "confirmation": None,
                "policy": "legacy_none", "target": "P_reward"}
            and value.get("retry") == {
                "new_http_requests_for_preserved_first_slot": 0,
                "policy_or_threshold_change": False,
                "preserve_prior_response": True,
                "requires_new_registered_execution": True},
            "공개 실패 summary가 V2 indeterminate 원문과 다르다")


def _rule_audit(value: dict[str, Any], design: dict[str, Any], protocol: dict[str, Any],
                execution_root: Path) -> dict[str, Any]:
    actions = value.get("actual_actions")
    result = execution_root / "c02/strong_registered_rule/result.json"
    source_closure = protocol.get("source_sha256_lf")
    implementation_sha = (source_closure.get("evals/evidence_guided_selector_v1.py")
                          if isinstance(source_closure, dict) else None)
    require(value.get("schema") == "registered_rule_description_execution_audit_v1"
            and value.get("scope") == "posthoc_development_audit_no_policy_mutation"
            and value.get("design_sha256") == _sha(ROOT / protocol["design_protocol"]["path"])
            and value.get("registered_text_rule_policy") == design.get("rule_policy")
            and value.get("actual_rule_implementation")
            == "evals/evidence_guided_selector_v1.py::select_strong_rule"
            and value.get("implementation_lf_sha256") == implementation_sha
            and value.get("case_id") == "c02" and isinstance(actions, list)
            and len(actions) >= 2 and result.is_file()
            and value.get("result_sha256") == _sha(result)
            and value.get("must_report_rule_code_vs_text_discrepancy") is True
            and value.get("rule_arm_omitted_or_resampled") is False
            and value.get("model_only_mode_comparison_policy_changed") is False
            and value.get("new_API_GPU") == 0,
            "등록 규칙 설명/실제 코드 audit 결합 불일치")
    actions = cast(list[dict[str, Any]], actions)
    require(all(isinstance(row, dict) and set(row) == {"action", "target", "fact_ids"}
                for row in actions), "규칙 audit action 형식 불일치")
    return {
        "status": "registered_rule_prose_code_mismatch",
        "registered_text_rule_policy": deepcopy(value["registered_text_rule_policy"]),
        "actual_rule_implementation": value["actual_rule_implementation"],
        "implementation_lf_sha256": value["implementation_lf_sha256"],
        "verified_case_id": "c02",
        "verified_action_sequence": [{"action": row["action"], "target": row["target"]}
                                     for row in actions],
        "rule_description_matches_actual_sequence": False,
        "rule_arm_unqualified_for_stated_sequence": True,
        "rule_rows_preserved_not_resampled": True,
        "measurement_or_threshold_changed": False,
    }


def _completed(raw: dict[str, Any], continuation: dict[str, Any],
               protocol: dict[str, Any], result_path: Path) -> None:
    rows = raw.get("rows")
    model_requests = raw.get("model_requests")
    require(raw.get("schema") == V3.R.RESULT_SCHEMA and raw.get("status") == "complete"
            and raw.get("planned_rows") == 24 and isinstance(rows, list) and len(rows) == 24
            and type(model_requests) is int and 1 <= model_requests <= 96,
            "V3 base result는 완료된 24행이어야 한다")
    model_requests = cast(int, model_requests)
    require(continuation.get("schema") == V3.RESULT_SCHEMA
            and continuation.get("status") == "complete"
            and continuation.get("classification")
            == protocol["continuation"]["classification"]
            and continuation.get("exposure") == protocol["continuation"]["exposure"]
            and continuation.get("prior_attempt") == protocol["failed_attempt"]
            and continuation.get("base_result_sha256") == _sha(result_path)
            and continuation.get("model_requests_including_replay") == model_requests
            and continuation.get("replayed_provider_requests") == 1
            and continuation.get("new_provider_requests") == model_requests - 1
            and continuation.get("paired_probes") == raw.get("paired_probes")
            and continuation.get("sdk_role_executions_upper_bound")
            == raw.get("sdk_role_executions_upper_bound")
            and continuation.get("assessment") == raw.get("assessment")
            and continuation.get("claims") == protocol.get("claims"),
            "V3 continuation result/원 base result 결합 불일치")
    require(continuation["new_provider_requests"] >= 0,
            "API96 또는 replay 포함 분모 불일치")


def export(execution_root: Path, protocol_path: Path, design_path: Path, bridge_path: Path,
           source_batch: Path, corpus: Path, bridge_root: Path, failure_summary: Path,
           rule_audit: Path, *,
           anchor: str, out: Path,
           verify_protocol_fn: Callable[..., dict[str, Any]] = V3.verify_protocol,
           base_export_fn: Callable[..., dict[str, Any]] = BASE.export) -> dict[str, Any]:
    """완료된 V3 실행만 fresh 공개 폴더로 내보낸다."""
    paths = [Path(item).resolve() for item in
             (execution_root, protocol_path, design_path, bridge_path, source_batch,
              corpus, bridge_root, failure_summary, rule_audit, out)]
    (execution_root, protocol_path, design_path, bridge_path, source_batch,
     corpus, bridge_root, failure_summary, rule_audit, out) = paths
    require(not out.exists(), "release output은 fresh여야 한다")
    protocol = _read(protocol_path)
    base_v2 = ROOT / protocol.get("base_v2_protocol", {}).get("path", "")
    base_v1 = ROOT / protocol.get("base_v1_protocol", {}).get("path", "")
    require(protocol.get("schema") == V3.PROTOCOL_SCHEMA
            and protocol.get("status") == "registered_only_after_commit"
            and protocol.get("base_v2_anchor") == V3.V2_ANCHOR
            and protocol.get("base_v1_anchor") == V3.V2.STAGE_B_ANCHOR,
            "V3 execution protocol 형식/anchor 불일치")
    result_path, continuation_path = execution_root / "result.json", execution_root / "v3-result.json"
    require(result_path.is_file() and continuation_path.is_file(),
            "V3 완료 result/v3-result가 모두 필요하다")
    raw, continuation = _read(result_path), _read(continuation_path)
    _completed(raw, continuation, protocol, result_path)
    failure = _read(failure_summary)
    _failure_summary(failure)
    require(failure["model_requests_consumed"] == protocol["failed_attempt"]["api_requests_consumed"]
            and failure["sdk_probe_executions"] == protocol["failed_attempt"]["sdk_role_executions"]
            and failure["assessment"] == protocol["failed_attempt"]["assessment"],
            "공개 실패 summary와 V3 failed_attempt 불일치")
    design = _read(design_path)
    policy_audit = _rule_audit(_read(rule_audit), design, protocol, execution_root)
    verified = verify_protocol_fn(protocol_path, anchor, base_v2, base_v1, design_path,
                                  bridge_path, source_batch, corpus, bridge_root)
    require(verified == protocol, "V3 protocol 검증 반환값 불일치")

    verified_base = out / "verified-base"
    base_summary = base_export_fn(
        result_path, base_v1, design_path, bridge_path, source_batch, corpus, bridge_root,
        anchor=V3.V2.STAGE_B_ANCHOR, out=verified_base)
    require(base_summary.get("schema") == BASE.SUMMARY_SCHEMA
            and base_summary.get("status") == "complete"
            and base_summary.get("assessment") == raw.get("assessment")
            and base_summary.get("planned_rows") == 24
            and base_summary.get("complete_rows") == 24,
            "verified-base summary가 완료된 24행과 다르다")
    base_manifest_path = verified_base / "artifact-manifest.json"
    base_summary_path = verified_base / "summary.json"
    require(base_manifest_path.is_file() and base_summary_path.is_file(),
            "verified-base 공개 manifest/summary 누락")

    summary = {
        "schema": SUMMARY_SCHEMA, "status": "complete",
        "assessment": raw["assessment"], "planned_rows": 24, "complete_rows": 24,
        "classification": protocol["continuation"]["classification"],
        "exposure": protocol["continuation"]["exposure"],
        "verified_base_summary_sha256": _sha(base_summary_path),
        "prior_failure": {"assessment": "indeterminate", "completed_rows": 0,
                          "api_requests": 1, "sdk_probe_executions": 0,
                          "public_summary_sha256": _sha(failure_summary)},
        "provider_request_accounting": {
            "cap_including_replay": protocol["inherited"]["api_requests"],
            "model_requests_including_replay": raw["model_requests"],
            "replayed_preserved_requests": 1,
            "new_provider_requests": raw["model_requests"] - 1,
            "first_response_reselected": False,
        },
        "uncertainty": deepcopy(base_summary["uncertainty"]),
        "policy_audit": policy_audit,
        "comparison_scope": {
            "model_mode_rows_per_mode": 6,
            "model_mode_contrast": ["legacy_none", "enabled_low"],
            "rule_rows": "actual_frozen_code_observation_not_stated_sequence_baseline",
            "auxiliary_three_policy_development_confirmation": True,
            "original_four_arm_complete": False,
            "heldout_confirmation": False,
        },
        "claims": deepcopy(protocol["claims"]),
    }
    require(summary["uncertainty"].get("independent_units") == 2
            and summary["uncertainty"].get("p_value_reported") is False
            and summary["uncertainty"].get("small_sample") is True,
            "2-seed 기술 불확실성 보존 실패")
    provenance = {
        "schema": PROVENANCE_SCHEMA, "outer_execution_anchor": anchor,
        "v2_execution_anchor": protocol["base_v2_anchor"],
        "v1_execution_anchor": protocol["base_v1_anchor"],
        "execution_protocol": _public_ref(protocol_path),
        "base_v2_protocol": _public_ref(base_v2),
        "base_v1_protocol": _public_ref(base_v1),
        "design_protocol": _public_ref(design_path),
        "bridge_protocol": _public_ref(bridge_path),
        "failure_public_summary": _public_ref(failure_summary),
        "private_rule_audit_sha256": _sha(rule_audit),
        "source_batch_sha256": _sha(source_batch), "corpus_sha256": _sha(corpus),
        "private_base_result_sha256": _sha(result_path),
        "private_continuation_result_sha256": _sha(continuation_path),
        "failed_attempt_artifacts": deepcopy(protocol["failed_attempt"]["artifacts"]),
        "source_sha256_lf": deepcopy(protocol["source_sha256_lf"]),
        "verified_base_manifest_sha256": _sha(base_manifest_path),
        "private_paths_published": False, "reasoning_body_published": False,
    }
    _write_new(out / "release-summary.json", summary)
    _write_new(out / "release-provenance.json", provenance)
    manifest = {
        "schema": MANIFEST_SCHEMA, "status": "complete",
        "files": {"release-summary.json": _sha(out / "release-summary.json"),
                  "release-provenance.json": _sha(out / "release-provenance.json")},
        "verified_base_manifest_sha256": _sha(base_manifest_path),
        "verified_base_artifact_count": _read(base_manifest_path)["artifact_count"],
    }
    _write_new(out / "release-manifest.json", manifest)
    return deepcopy(summary)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("execution-root", "protocol", "design", "bridge", "source-batch",
                 "corpus", "bridge-root", "failure-summary", "rule-audit", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--anchor", required=True)
    args = parser.parse_args()
    export(args.execution_root, args.protocol, args.design, args.bridge, args.source_batch,
           args.corpus, args.bridge_root, args.failure_summary, args.rule_audit,
           anchor=args.anchor, out=args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["export"]
