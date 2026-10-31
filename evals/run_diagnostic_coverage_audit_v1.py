"""HOLD 사례에서 등록 검사 하나의 관측 가능 범위를 분리해 확인한다.

이 실행기는 c02/F1과 같은 seed의 c01/NONE만 사용한다. 기존 모델 판단은
보존하고 운영자가 사전에 고정한 P_reward paired 검사 한 번만 실행한다.
학습, 교정, 회복, 모델 호출 권한은 없다.
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
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "bench"), str(ROOT / "src")]

import evidence_guided_probe_host_v1 as HOST
import evidence_guided_selector_v1 as SEL
import reasoning_diagnostic_loop_v1 as LOOP
import run_prospective_agentic_comparison_v1 as RUN
import run_reasoning_diagnosis_study_v1 as R
import run_registry_action_study_v1 as BASE

CACHE = R.PARENT.COMP.CACHE
SCHEMA = "diagnostic_coverage_audit_protocol_v1"
PREFLIGHT_SCHEMA = "diagnostic_coverage_audit_preflight_v1"
RESULT_SCHEMA = "diagnostic_coverage_audit_result_v1"
PARENT_ANCHOR = "44e7408ccc358a958415f383bcedc90e5b7b2fbc"
MODE_RESULT_SHA256 = R.MODE_RESULT_SHA256
CORPUS_SHA256 = R.CORPUS_SHA256
PARENT_EPOCH_SHA256 = "d103be263f15d397cf6f4a8580516f71ada7926f312fbda2b2a3d50be231f495"
CASE_ID = "c02"
REFERENCE_ID = "c01"
MODE = "enabled_low"
PROBE = "P_reward"
APPROVER = "operator_registry_coverage_audit"
MAX_WALL_S = 1800
FACT_IDS = (
    "telemetry:case:Metrics/base_velocity/error_vel_xy",
    "telemetry:reference:Metrics/base_velocity/error_vel_xy",
    "telemetry:case:Episode_Reward/track_lin_vel_xy_exp",
    "telemetry:reference:Episode_Reward/track_lin_vel_xy_exp",
)
SOURCE_FILES = (
    "evals/immutable_git_proof_cache_v1.py",
    "evals/run_diagnostic_coverage_audit_v1.py",
    "tests/test_run_diagnostic_coverage_audit_v1.py",
)


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def _epoch(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    _require(BASE._sha(path) == PARENT_EPOCH_SHA256, "고정 c02 enabled_low epoch SHA 불일치")
    value = BASE._read(path)
    _require(value.get("schema") == "reasoning_diagnostic_loop_result_v1"
             and value.get("case_id") == CASE_ID and value.get("mode") == MODE
             and value.get("status") == "complete"
             and value.get("decision_outcome") == "HOLD"
             and value.get("final_status") == "hold",
             "고정 parent HOLD 결과 불일치")
    steps = value.get("steps")
    _require(isinstance(steps, list) and len(steps) == 3
             and steps[0].get("action", {}).get("action") == "read_evidence"
             and steps[1].get("action", {}).get("target") == "P_recorded_optimizer_updates"
             and steps[2].get("action", {}).get("action") == "hold",
             "고정 parent read-count-HOLD 순서 불일치")
    return value


def _operator_action(state: dict[str, Any]) -> dict[str, Any]:
    visible = state.get("visible_facts")
    if not isinstance(visible, dict):
        raise TypeError("selector visible facts가 없다")
    for fact_id in FACT_IDS:
        value = visible.get(fact_id)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            raise ValueError(f"운영자 고정 scalar fact가 없거나 비유한 값이다: {fact_id}")
    action = {"action": "run_probe", "target": PROBE, "fact_ids": list(FACT_IDS)}
    spec = SEL.request_spec(state, action)
    _require(spec.get("action") == action and spec.get("host_args", {}).get("roles")
             == ["case", "reference"], "P_reward request spec 불일치")
    return action


def prepare_protocol(parent_protocol: Path, parent_epoch_sha256: str = PARENT_EPOCH_SHA256
                     ) -> dict[str, Any]:
    parent_path = Path(parent_protocol).resolve()
    parent = BASE._read(parent_path)
    _require(parent.get("schema") == R.SCHEMA, "parent reasoning protocol 불일치")
    _require(parent_epoch_sha256 == PARENT_EPOCH_SHA256, "parent epoch SHA 고정값 불일치")
    for relative in SOURCE_FILES:
        _require((ROOT / relative).is_file(), f"source closure 파일이 없다: {relative}")
    own = {relative: BASE._sha(ROOT / relative, lf=True) for relative in SOURCE_FILES}
    inherited = {
        **deepcopy(parent.get("inherited_source_sha256_lf", {})),
        **deepcopy(parent.get("source_sha256_lf", {})),
        parent_path.relative_to(ROOT).as_posix(): BASE._sha(parent_path, lf=True),
    }
    return {
        "schema": SCHEMA,
        "status": "registered_only_after_commit",
        "purpose": "operator_fixed_registered_probe_coverage_audit_post_exposure",
        "parent_reasoning_protocol": BASE._public_ref(parent_path),
        "parent_reasoning_anchor": PARENT_ANCHOR,
        "mode_result_sha256": MODE_RESULT_SHA256,
        "corpus_sha256": CORPUS_SHA256,
        "parent_epoch_sha256": parent_epoch_sha256,
        "source_sha256_lf": own,
        "inherited_source_sha256_lf": inherited,
        "design": {
            "case_id": CASE_ID, "reference_case_id": REFERENCE_ID,
            "seed": 2320, "case_category": "F1", "reference_category": "NONE",
            "mode": MODE, "replicate": 1, "probe": PROBE,
            "selection_actor": APPROVER, "model_selected_probe": False,
            "resource_lease_owner": RUN.OWNER,
            "proof_cache": {"kind": "immutable_git_show", "scope": "one_audit",
                            "mutable_or_authority_cached": False},
            "visible_scalar_fact_ids": list(FACT_IDS),
            "paired_probes": 1, "sdk_role_executions": 2, "api_requests": 0,
            "training": False, "correction": False, "recovery": False,
            "case_substitution": False, "reference_substitution": False,
            "original_model_result_immutable": True,
        },
        "bounds": {"wall_s": MAX_WALL_S, "paired_probes": 1,
                   "sdk_role_executions": 2, "api_requests": 0},
        "analysis": {
            "post_exposure_development": True,
            "prediction": ("primary hypothesis: P_reward is normal because near-stationary "
                           "body/world velocities can make the rollout discrepancy small; "
                           "this is a pre-execution hypothesis, not an observed outcome"),
            "reporting_rules": {
                "abnormal": "registered P_reward could distinguish this selected HOLD case",
                "normal": ("registered P_reward did not reveal this F1 state; no claim is made "
                           "that other probes are normal"),
                "unknown": "indeterminate",
            },
        },
        "claims": {"model_selection_failure": False, "diagnosis_improvement": False,
                   "new_seed_effect": False, "recovery": False, "speed_advantage": False},
    }


def verify_protocol(path: Path, anchor: str, corpus: Path, lab: Path) -> dict[str, Any]:
    _require(re.fullmatch(r"[0-9a-f]{40}", anchor) is not None, "anchor 형식 불일치")
    path = Path(path).resolve()
    value = BASE._read(path)
    _require(value.get("schema") == SCHEMA, "coverage audit protocol schema 불일치")
    parent = ROOT / value["parent_reasoning_protocol"]["path"]
    _require(value == prepare_protocol(parent, value.get("parent_epoch_sha256", "")),
             "coverage audit protocol이 현재 source와 다르다")
    R.verify_protocol(parent, PARENT_ANCHOR, Path(corpus).resolve(), Path(lab).resolve())
    BASE._git("cat-file", "-e", f"{anchor}^{{commit}}")
    BASE._git("merge-base", "--is-ancestor", anchor, "HEAD")
    BASE._verify_anchor_file(path, anchor)
    for relative, digest in value["source_sha256_lf"].items():
        blob = BASE._git("show", f"{anchor}:{relative}").replace(b"\r\n", b"\n")
        _require(hashlib.sha256(blob).hexdigest() == digest,
                 f"anchor source blob 불일치: {relative}")
    return deepcopy(value)


def inspect_inputs(parent_protocol: Path, source_batch: Path, mode_result: Path, corpus: Path,
                   parent_epoch: Path, lab: Path) -> dict[str, Any]:
    """등록 전에도 기존 등록 입력만 읽어 고정 case/action을 재검증한다."""
    _epoch(parent_epoch)
    spec, cases, gates = R._validate_inputs(
        Path(parent_protocol), PARENT_ANCHOR, Path(source_batch), Path(mode_result),
        Path(corpus), Path(lab))
    case = next(row for row in cases if row["case_id"] == CASE_ID)
    reference = next(row for row in cases if row["case_id"] == REFERENCE_ID)
    _require(case["seed"] == reference["seed"] == 2320
             and case["category"] == "F1" and reference["category"] == "NONE"
             and gates[CASE_ID]["run"] is True and gates[REFERENCE_ID]["run"] is False,
             "고정 c02/F1-c01/NONE source gate 불일치")
    first = LOOP.load_recorded_first(
        Path(mode_result), ROOT / spec["mode_plan"]["path"], Path(corpus),
        R.MODE_ANCHOR, MODE_RESULT_SHA256, CASE_ID, MODE, replicate=1)
    R._verify_first_source(first, case, reference)
    action = _operator_action(first["state"])
    return {"spec": spec, "case": case, "reference": reference,
            "first": first, "action": action}


def preflight(protocol: Path, anchor: str, source_batch: Path, mode_result: Path, corpus: Path,
              parent_epoch: Path, lab: Path, out: Path) -> dict[str, Any]:
    out = Path(out).resolve()
    _require(not out.exists(), "preflight out은 fresh 경로여야 한다")
    spec = verify_protocol(protocol, anchor, corpus, lab)
    inputs = inspect_inputs(ROOT / spec["parent_reasoning_protocol"]["path"], source_batch,
                            mode_result, corpus, parent_epoch, lab)
    result = {"schema": PREFLIGHT_SCHEMA, "status": "ready",
              "protocol": BASE._ref(protocol), "source_batch": BASE._ref(source_batch),
              "mode_result": BASE._ref(mode_result), "corpus": BASE._ref(corpus),
              "parent_epoch": BASE._ref(parent_epoch), "case_id": CASE_ID,
              "reference_case_id": REFERENCE_ID, "operator_action": inputs["action"],
              "api_requests": 0, "gpu_started": False, "execution_registered": True}
    BASE._write(out / "preflight.json", result)
    return result


def _interpret(outcome: str) -> str:
    if outcome == "abnormal":
        return "registered P_reward could distinguish this selected HOLD case"
    if outcome == "normal":
        return "registered P_reward did not reveal this F1 state; other probes remain untested"
    _require(outcome == "unknown", "등록 probe outcome 불일치")
    return "indeterminate"


def _execute(protocol: Path, anchor: str, source_batch: Path, mode_result: Path, corpus: Path,
             parent_epoch: Path, lab: Path, out: Path, *,
             execute_host: Callable = HOST.execute_pair) -> dict[str, Any]:
    out = Path(out).resolve()
    _require(not out.exists(), "audit out은 fresh 경로여야 한다")
    spec = verify_protocol(protocol, anchor, corpus, lab)
    inputs = inspect_inputs(ROOT / spec["parent_reasoning_protocol"]["path"], source_batch,
                            mode_result, corpus, parent_epoch, lab)
    base_protocol = R._base_protocol(inputs["spec"])
    probe_protocol = ROOT / base_protocol["protocols"]["paired_probe"]["path"]
    view_protocol = ROOT / base_protocol["protocols"]["view"]["path"]
    case, reference = inputs["case"], inputs["reference"]
    case_runtime = Path(case["metadata"]["runtime_facts"]["case"]["path"]).resolve()
    reference_runtime = Path(
        reference["metadata"]["runtime_facts"]["reference"]["path"]).resolve()
    source_hashes = {path: BASE._sha(path) for path in (
        Path(source_batch), Path(mode_result), Path(corpus), Path(parent_epoch),
        probe_protocol, view_protocol)}
    lease_id = RUN._lease().get("leaseId")
    _require(isinstance(lease_id, str) and bool(lease_id), "leaseId가 없다")
    started = time.monotonic()

    def verify_context(context: dict[str, Any] | None = None) -> dict[str, Any]:
        elapsed = time.monotonic() - started
        _require(math.isfinite(elapsed) and elapsed < MAX_WALL_S, "audit wall 상한 종료")
        _require(RUN._lease().get("leaseId") == lease_id, "실행 중 leaseId 변경")
        _require(all(BASE._sha(path) == digest for path, digest in source_hashes.items()),
                 "실행 중 입력/protocol bytes 변경")
        role = None if context is None else context.get("role")
        _require(role is None or role in ("case", "reference"), "host role 불일치")
        return {"approved": True, "leaseId": lease_id, "elapsed_s": elapsed,
                "checked_role": role, "selection_actor": APPROVER}

    out.mkdir(parents=True, exist_ok=False)
    host = execute_host(
        inputs["first"]["state"], inputs["action"], case_input=case["input"],
        reference_input=reference["input"], case_runtime=case_runtime,
        reference_runtime=reference_runtime, probe_protocol=probe_protocol,
        view_protocol=view_protocol, anchor=anchor, lab=Path(lab).resolve(),
        out=out / "probe", verify_context=verify_context, approver=APPROVER)
    after = SEL.advance(
        inputs["first"]["state"], inputs["action"],
        case_plan=Path(host["case_plan"]), case_receipt=Path(host["case_receipt"]),
        reference_plan=Path(host["reference_plan"]),
        reference_receipt=Path(host["reference_receipt"]))
    outcome = after["observed"].get(PROBE)
    _require(outcome in {"normal", "abnormal", "unknown"}, "P_reward outcome이 없다")
    verify_context()
    result = {"schema": RESULT_SCHEMA, "status": "complete", "case_id": CASE_ID,
              "reference_case_id": REFERENCE_ID, "selection_actor": APPROVER,
              "model_selected_probe": False, "original_model_outcome": "HOLD",
              "probe": PROBE, "outcome": outcome, "interpretation": _interpret(outcome),
              "host_result": BASE._ref(out / "probe" / "host-result.json"),
              "state_after_digest": after["state_digest"], "api_requests": 0,
              "paired_probes": 1, "sdk_role_executions": 2,
              "training": False, "correction": False, "recovery": False,
              "other_probe_outcomes_inferred": False,
              "parent_wall_s": time.monotonic() - started}
    BASE._write(out / "result.json", result)
    return result


def execute(protocol: Path, anchor: str, source_batch: Path, mode_result: Path, corpus: Path,
            parent_epoch: Path, lab: Path, out: Path, *,
            execute_host: Callable = HOST.execute_pair) -> dict[str, Any]:
    """단일 audit transaction에서 immutable ``git show``만 재사용한다."""
    with CACHE.proof_scope():
        return _execute(protocol, anchor, source_batch, mode_result, corpus, parent_epoch,
                        lab, out, execute_host=execute_host)


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare-protocol")
    prepare.add_argument("--parent-protocol", type=Path, required=True)
    prepare.add_argument("--out", type=Path, required=True)
    for name in ("preflight", "execute"):
        command = sub.add_parser(name)
        command.add_argument("--protocol", type=Path, required=True)
        command.add_argument("--anchor", required=True)
        command.add_argument("--source-batch", type=Path, required=True)
        command.add_argument("--mode-result", type=Path, required=True)
        command.add_argument("--corpus", type=Path, required=True)
        command.add_argument("--parent-epoch", type=Path, required=True)
        command.add_argument("--lab", type=Path, required=True)
        command.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare-protocol":
        BASE._write(args.out, prepare_protocol(args.parent_protocol))
        return 0
    function = preflight if args.command == "preflight" else execute
    function(args.protocol, args.anchor, args.source_batch, args.mode_result, args.corpus,
             args.parent_epoch, args.lab, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
