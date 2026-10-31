"""검증된 model HOLD 뒤의 등록 제안을 별도 운영자 trial로 실행한다.

부모 HOLD state와 결과는 수정하지 않는다. 부모의 HOLD 직전 shadow state에서
등록 규칙이 제안한 P_reward를 새 승인으로 한 번 실행하는 후행 개발 실험이다.
"""
# pyright: reportMissingImports=false
from __future__ import annotations

import argparse
import hashlib
import math
import re
import sys
import time
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "bench"), str(ROOT / "src")]

import evidence_guided_probe_host_v1 as HOST
import evidence_guided_selector_v1 as SEL
import model_hold_followup_v1 as FOLLOW
import run_diagnostic_coverage_audit_v1 as AUDIT
import run_prospective_agentic_comparison_v1 as RUN
import run_reasoning_diagnosis_study_v1 as R
import run_registry_action_study_v1 as BASE

CACHE = R.PARENT.COMP.CACHE
SCHEMA = "model_hold_handoff_trial_protocol_v1"
PREFLIGHT_SCHEMA = "model_hold_handoff_trial_preflight_v1"
RESULT_SCHEMA = "model_hold_handoff_trial_result_v1"
PARENT_ANCHOR = AUDIT.PARENT_ANCHOR
EPOCH_SHA256 = AUDIT.PARENT_EPOCH_SHA256
STUDY_SHA256 = "69604985fdd477aff48293d00e7646d2f0af39125929f304865d3a13ef74c632"
AUDIT_SUMMARY_SHA256 = "91df2b847e96568b67b867ec18f6d02c9534a44806584aadfedd927134a3a1e0"
CASE_ID = "c02"
REFERENCE_ID = "c01"
MODE = "enabled_low"
PROBE = "P_reward"
APPROVER = "registered_handoff_trial_operator"
SELECTOR_ACTOR = FOLLOW.SELECTION_ACTOR
MAX_WALL_S = 1800
SOURCE_FILES = (
    "evals/immutable_git_proof_cache_v1.py",
    "evals/model_hold_followup_v1.py",
    "tests/test_model_hold_followup_v1.py",
    "evals/run_diagnostic_coverage_audit_v1.py",
    "tests/test_run_diagnostic_coverage_audit_v1.py",
    "evals/run_model_hold_handoff_trial_v1.py",
    "tests/test_run_model_hold_handoff_trial_v1.py",
)


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def _checked_ref(value: object, label: str) -> Path:
    _require(isinstance(value, dict), f"{label} ref 형식")
    item = cast(dict[str, Any], value)
    path, expected = item.get("path"), item.get("sha256")
    _require(isinstance(path, str) and isinstance(expected, str), f"{label} ref 필드")
    resolved = Path(cast(str, path)).resolve()
    _require(resolved.is_file() and BASE._sha(resolved) == expected,
             f"{label} ref/SHA 불일치")
    return resolved


def _audit_summary(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    _require(BASE._sha(path) == AUDIT_SUMMARY_SHA256, "coverage audit summary SHA 불일치")
    value = BASE._read(path)
    _require(value.get("schema") == "diagnostic_coverage_audit_public_summary_v1"
             and value.get("status") == "complete"
             and value.get("case_id") == CASE_ID
             and value.get("reference_case_id") == REFERENCE_ID
             and value.get("probe") == PROBE and value.get("outcome") == "abnormal"
             and value.get("selection_actor") == AUDIT.APPROVER
             and value.get("model_selected_probe") is False
             and value.get("original_model_result_changed") is False,
             "coverage audit 고정 결과 불일치")
    return value


def prepare_protocol(parent_protocol: Path, prior_audit_summary: Path) -> dict[str, Any]:
    parent_path = Path(parent_protocol).resolve()
    audit_path = Path(prior_audit_summary).resolve()
    parent = BASE._read(parent_path)
    _require(parent.get("schema") == R.SCHEMA, "parent reasoning protocol 불일치")
    _audit_summary(audit_path)
    for relative in SOURCE_FILES:
        _require((ROOT / relative).is_file(), f"source closure 파일이 없다: {relative}")
    own = {relative: BASE._sha(ROOT / relative, lf=True) for relative in SOURCE_FILES}
    inherited = {
        **deepcopy(parent.get("inherited_source_sha256_lf", {})),
        **deepcopy(parent.get("source_sha256_lf", {})),
        parent_path.relative_to(ROOT).as_posix(): BASE._sha(parent_path, lf=True),
    }
    return {
        "schema": SCHEMA, "status": "registered_only_after_commit",
        "purpose": "registered_rule_handoff_after_verified_model_hold_development_trial",
        "parent_reasoning_protocol": BASE._public_ref(parent_path),
        "parent_reasoning_anchor": PARENT_ANCHOR,
        "prior_coverage_audit": BASE._public_ref(audit_path),
        "mode_result_sha256": R.MODE_RESULT_SHA256,
        "corpus_sha256": R.CORPUS_SHA256,
        "parent_epoch_sha256": EPOCH_SHA256,
        "parent_study_sha256": STUDY_SHA256,
        "source_sha256_lf": own,
        "inherited_source_sha256_lf": inherited,
        "design": {
            "case_id": CASE_ID, "reference_case_id": REFERENCE_ID, "seed": 2320,
            "case_category": "F1", "reference_category": "NONE", "mode": MODE,
            "proposal_builder": FOLLOW.SCHEMA, "probe": PROBE,
            "selector_actor": SELECTOR_ACTOR, "approval_actor": APPROVER,
            "original_model_selected_probe": False,
            "original_model_result_changed": False,
            "parent_model_actions": 3, "trial_actions": 1, "total_actions": 4,
            "parent_paired_probes": 1, "trial_paired_probes": 1,
            "total_paired_probes": 2,
            "resource_lease_owner": RUN.OWNER,
            "proof_cache": {"kind": "immutable_git_show", "scope": "one_trial",
                            "mutable_or_authority_cached": False},
            "api_requests": 0, "sdk_role_executions": 2,
            "training": False, "correction": False, "recovery": False,
            "automatic_resume": False, "case_substitution": False,
        },
        "bounds": {"wall_s": MAX_WALL_S, "api_requests": 0,
                   "paired_probes": 1, "sdk_role_executions": 2},
        "analysis": {
            "post_exposure_development": True,
            "prior_coverage_outcome_seen": "abnormal",
            "prediction": "P_reward outcome is abnormal",
            "purpose_is_integration_check_not_new_observation": True,
            "reporting": ("valid HOLD proposal handoff and host integration only; it does not "
                          "change the parent model result or the prior model aggregate"),
        },
        "claims": {"new_observation": False, "model_improvement": False,
                   "diagnosis_improvement": False, "new_seed_effect": False,
                   "recovery": False, "speed_advantage": False},
    }


def verify_protocol(path: Path, anchor: str, corpus: Path, lab: Path) -> dict[str, Any]:
    _require(re.fullmatch(r"[0-9a-f]{40}", anchor) is not None, "anchor 형식 불일치")
    path = Path(path).resolve()
    value = BASE._read(path)
    _require(value.get("schema") == SCHEMA, "hold handoff trial protocol schema 불일치")
    parent = ROOT / value["parent_reasoning_protocol"]["path"]
    audit = ROOT / value["prior_coverage_audit"]["path"]
    _require(value == prepare_protocol(parent, audit),
             "hold handoff trial protocol이 현재 source와 다르다")
    R.verify_protocol(parent, PARENT_ANCHOR, Path(corpus).resolve(), Path(lab).resolve())
    BASE._git("cat-file", "-e", f"{anchor}^{{commit}}")
    BASE._git("merge-base", "--is-ancestor", anchor, "HEAD")
    BASE._verify_anchor_file(path, anchor)
    BASE._verify_anchor_file(audit, anchor)
    for relative, digest in value["source_sha256_lf"].items():
        blob = BASE._git("show", f"{anchor}:{relative}").replace(b"\r\n", b"\n")
        _require(hashlib.sha256(blob).hexdigest() == digest,
                 f"anchor source blob 불일치: {relative}")
    return deepcopy(value)


def _proposal(epoch_result: Path, study_result: Path) -> dict[str, Any]:
    proposal = FOLLOW.build_proposal(
        Path(epoch_result), Path(study_result), expected_epoch_sha256=EPOCH_SHA256,
        expected_study_sha256=STUDY_SHA256)
    _require(proposal.get("schema") == FOLLOW.SCHEMA
             and proposal.get("status") == "complete"
             and proposal.get("case_id") == CASE_ID and proposal.get("mode") == MODE
             and proposal.get("selection_actor") == SELECTOR_ACTOR,
             "model HOLD proposal identity 불일치")
    _require(proposal.get("approval_granted") is False
             and proposal.get("execution_allowed") is False
             and proposal.get("correction_allowed") is False
             and proposal.get("automatic_resume") is False,
             "proposal의 비권한 계약이 바뀌었다")
    parent = proposal.get("parent_closed_HOLD")
    budgets = proposal.get("budgets")
    _require(isinstance(parent, dict) and parent.get("verified") is True
             and parent.get("decision_outcome_preserved") == "HOLD"
             and parent.get("model_valid_hold_preserved") is True,
             "검증된 parent model HOLD가 아니다")
    _require(isinstance(budgets, dict)
             and budgets.get("model_actions") == {"used": 3, "limit": 4, "remaining": 1}
             and budgets.get("paired_probes") == {"used": 1, "limit": 3, "remaining": 2}
             and budgets.get("shadow_probe_budget_left") == 2,
             "parent/trial budget 계약 불일치")
    expected = {"action": "run_probe", "target": PROBE,
                "fact_ids": list(FOLLOW.REWARD_FACT_IDS)}
    _require(proposal.get("proposal") == expected
             and proposal.get("proposal_reason")
             == "counts_normal_and_source_tracking_failure_gate_verified",
             "고정 P_reward handoff proposal 불일치")
    return proposal


def inspect_inputs(parent_protocol: Path, source_batch: Path, mode_result: Path, corpus: Path,
                   epoch_result: Path, study_result: Path, prior_audit_summary: Path,
                   lab: Path) -> dict[str, Any]:
    _audit_summary(prior_audit_summary)
    source = AUDIT.inspect_inputs(parent_protocol, source_batch, mode_result, corpus,
                                  epoch_result, lab)
    proposal = _proposal(epoch_result, study_result)
    shadow_path = _checked_ref(proposal.get("shadow_state_before"), "shadow state")
    shadow = SEL._state(BASE._read(shadow_path))
    action = cast(dict[str, Any], proposal["proposal"])
    request = SEL.request_spec(shadow, action)
    _require(request.get("action") == action
             and request.get("approval_granted") is False
             and request.get("model_generated_args") is False,
             "shadow P_reward request spec 불일치")
    binding = proposal.get("source_binding")
    _require(isinstance(binding, dict)
             and binding.get("case_checkpoint_sha256")
             == source["case"]["checkpoint"]["sha256"]
             and binding.get("reference_checkpoint_sha256")
             == source["reference"]["checkpoint"]["sha256"],
             "proposal/source checkpoint 불일치")
    return {"source": source, "proposal": proposal, "shadow": shadow,
            "action": action, "request_spec": request}


def preflight(protocol: Path, anchor: str, source_batch: Path, mode_result: Path, corpus: Path,
              epoch_result: Path, study_result: Path, lab: Path, out: Path) -> dict[str, Any]:
    out = Path(out).resolve()
    _require(not out.exists(), "preflight out은 fresh 경로여야 한다")
    spec = verify_protocol(protocol, anchor, corpus, lab)
    inputs = inspect_inputs(
        ROOT / spec["parent_reasoning_protocol"]["path"], source_batch, mode_result, corpus,
        epoch_result, study_result, ROOT / spec["prior_coverage_audit"]["path"], lab)
    result = {"schema": PREFLIGHT_SCHEMA, "status": "ready",
              "protocol": BASE._ref(protocol), "source_batch": BASE._ref(source_batch),
              "parent_epoch": BASE._ref(epoch_result), "parent_study": BASE._ref(study_result),
              "case_id": CASE_ID, "reference_case_id": REFERENCE_ID,
              "proposal": inputs["proposal"]["proposal"],
              "selector_actor": SELECTOR_ACTOR, "approval_actor": APPROVER,
              "api_requests": 0, "gpu_started": False, "execution_registered": True}
    BASE._write(out / "preflight.json", result)
    return result


def _execute(protocol: Path, anchor: str, source_batch: Path, mode_result: Path, corpus: Path,
             epoch_result: Path, study_result: Path, lab: Path, out: Path, *,
             execute_host: Callable = HOST.execute_pair) -> dict[str, Any]:
    out = Path(out).resolve()
    _require(not out.exists(), "trial out은 fresh 경로여야 한다")
    spec = verify_protocol(protocol, anchor, corpus, lab)
    audit_path = ROOT / spec["prior_coverage_audit"]["path"]
    inputs = inspect_inputs(ROOT / spec["parent_reasoning_protocol"]["path"], source_batch,
                            mode_result, corpus, epoch_result, study_result, audit_path, lab)
    source = inputs["source"]
    base_protocol = R._base_protocol(source["spec"])
    probe_protocol = ROOT / base_protocol["protocols"]["paired_probe"]["path"]
    view_protocol = ROOT / base_protocol["protocols"]["view"]["path"]
    case, reference = source["case"], source["reference"]
    case_runtime = Path(case["metadata"]["runtime_facts"]["case"]["path"]).resolve()
    reference_runtime = Path(
        reference["metadata"]["runtime_facts"]["reference"]["path"]).resolve()
    immutable_inputs = (Path(source_batch), Path(mode_result), Path(corpus), Path(epoch_result),
                        Path(study_result), audit_path, probe_protocol, view_protocol)
    source_hashes = {path: BASE._sha(path) for path in immutable_inputs}
    lease_id = RUN._lease().get("leaseId")
    _require(isinstance(lease_id, str) and bool(lease_id), "leaseId가 없다")
    started = time.monotonic()

    def verify_context(context: dict[str, Any] | None = None) -> dict[str, Any]:
        elapsed = time.monotonic() - started
        _require(math.isfinite(elapsed) and elapsed < MAX_WALL_S, "trial wall 상한 종료")
        _require(RUN._lease().get("leaseId") == lease_id, "실행 중 leaseId 변경")
        _require(all(BASE._sha(path) == digest for path, digest in source_hashes.items()),
                 "실행 중 source/proposal/protocol bytes 변경")
        role = None if context is None else context.get("role")
        _require(role is None or role in ("case", "reference"), "host role 불일치")
        return {"approved": True, "leaseId": lease_id, "elapsed_s": elapsed,
                "checked_role": role, "selection_actor": SELECTOR_ACTOR,
                "approval_actor": APPROVER}

    out.mkdir(parents=True, exist_ok=False)
    host = execute_host(
        inputs["shadow"], inputs["action"], case_input=case["input"],
        reference_input=reference["input"], case_runtime=case_runtime,
        reference_runtime=reference_runtime, probe_protocol=probe_protocol,
        view_protocol=view_protocol, anchor=anchor, lab=Path(lab).resolve(),
        out=out / "probe", verify_context=verify_context, approver=APPROVER)
    after = SEL.advance(
        inputs["shadow"], inputs["action"],
        case_plan=Path(host["case_plan"]), case_receipt=Path(host["case_receipt"]),
        reference_plan=Path(host["reference_plan"]),
        reference_receipt=Path(host["reference_receipt"]))
    outcome = after["observed"].get(PROBE)
    _require(outcome in {"normal", "abnormal", "unknown"}, "P_reward outcome이 없다")
    verify_context()
    result = {"schema": RESULT_SCHEMA, "status": "complete", "case_id": CASE_ID,
              "reference_case_id": REFERENCE_ID, "selector_actor": SELECTOR_ACTOR,
              "approval_actor": APPROVER, "original_model_selected_probe": False,
              "original_model_result_changed": False, "parent_model_outcome": "HOLD",
              "proposal": inputs["proposal"]["proposal"], "probe_outcome": outcome,
              "host_result": BASE._ref(out / "probe" / "host-result.json"),
              "state_after_digest": after["state_digest"], "api_requests": 0,
              "paired_probes": 1, "sdk_role_executions": 2,
              "new_observation_claim": False, "model_aggregate_changed": False,
              "training": False, "correction": False, "recovery": False,
              "parent_wall_s": time.monotonic() - started}
    BASE._write(out / "result.json", result)
    return result


def execute(protocol: Path, anchor: str, source_batch: Path, mode_result: Path, corpus: Path,
            epoch_result: Path, study_result: Path, lab: Path, out: Path, *,
            execute_host: Callable = HOST.execute_pair) -> dict[str, Any]:
    with CACHE.proof_scope():
        return _execute(protocol, anchor, source_batch, mode_result, corpus, epoch_result,
                        study_result, lab, out, execute_host=execute_host)


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare-protocol")
    prepare.add_argument("--parent-protocol", type=Path, required=True)
    prepare.add_argument("--prior-audit-summary", type=Path, required=True)
    prepare.add_argument("--out", type=Path, required=True)
    for name in ("preflight", "execute"):
        command = sub.add_parser(name)
        command.add_argument("--protocol", type=Path, required=True)
        command.add_argument("--anchor", required=True)
        command.add_argument("--source-batch", type=Path, required=True)
        command.add_argument("--mode-result", type=Path, required=True)
        command.add_argument("--corpus", type=Path, required=True)
        command.add_argument("--epoch-result", type=Path, required=True)
        command.add_argument("--study-result", type=Path, required=True)
        command.add_argument("--lab", type=Path, required=True)
        command.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare-protocol":
        BASE._write(args.out, prepare_protocol(args.parent_protocol, args.prior_audit_summary))
        return 0
    function = preflight if args.command == "preflight" else execute
    function(args.protocol, args.anchor, args.source_batch, args.mode_result, args.corpus,
             args.epoch_result, args.study_result, args.lab, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
