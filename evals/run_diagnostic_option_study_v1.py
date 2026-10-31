"""등록 outcome partition option 계약의 사후개발 진단 측정 실행기."""
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
import lineage_nim_transport_v4 as T4
import prospective_agentic_comparison_v1 as COMP
import prospective_diagnosis_inputs_v2 as INPUT
import registry_action_loop_v2 as L2
import registry_action_options_v2 as O2
import run_prospective_agentic_comparison_v1 as RUN
import run_registry_action_study_v1 as BASE

SCHEMA = "diagnostic_option_study_protocol_v1"
PREFLIGHT_SCHEMA = "diagnostic_option_study_preflight_v1"
RESULT_SCHEMA = "diagnostic_option_study_result_v1"
POLICY = "evidence_guided_model"
MAX_WALL_S = 7200
MAX_REQUESTS = 48
MAX_PAIRED_PROBES = 18
MAX_SDK_ROLE_EXECUTIONS = 36
SOURCE_FILES = (
    "evals/registry_action_options_v2.py",
    "tests/test_registry_action_options_v2.py",
    "evals/registry_action_loop_v2.py",
    "tests/test_registry_action_loop_v2.py",
    "evals/run_diagnostic_option_study_v1.py",
    "tests/test_run_diagnostic_option_study_v1.py",
)


def _run_id(case_id: str) -> str:
    BASE._require(re.fullmatch(r"c\d{2}", case_id) is not None, "case_id 형식 불일치")
    return f"{case_id}.{POLICY}"


def prepare_protocol(parent_study_protocol: Path,
                     prior_option_result_sha256: str) -> dict[str, Any]:
    parent_path = Path(parent_study_protocol).resolve()
    parent = BASE._read(parent_path)
    BASE._require(parent.get("schema") == BASE.SCHEMA, "parent option study schema 불일치")
    BASE._require(re.fullmatch(r"[0-9a-f]{64}", prior_option_result_sha256) is not None,
                  "prior option result SHA 형식 불일치")
    BASE._require(parent.get("design", {}).get("opaque_cases") ==
                  [f"c{i:02d}" for i in range(1, 9)], "parent 8 source 설계 불일치")
    BASE._require(parent.get("design", {}).get("probe_budget_per_row") == L2.MAX_PAIRED_PROBES
                  and parent.get("design", {}).get("model_action_budget_per_row") == L2.MAX_MODEL_ACTIONS,
                  "parent와 V2 action/probe 예산 불일치")
    for relative in SOURCE_FILES:
        BASE._require((ROOT / relative).is_file(), f"source closure 파일이 없다: {relative}")
    return {
        "schema": SCHEMA,
        "purpose": "post_exposure_registered_partition_guided_diagnosis_development_study",
        "parent_study_protocol": BASE._public_ref(parent_path),
        "prior_option_result_sha256": prior_option_result_sha256,
        "source_sha256_lf": {name: BASE._sha(ROOT / name, lf=True) for name in SOURCE_FILES},
        "inherited_source_sha256_lf": {
            **deepcopy(parent["inherited_source_sha256_lf"]),
            **deepcopy(parent["source_sha256_lf"]),
        },
        "design": {
            "opaque_cases": [f"c{i:02d}" for i in range(1, 9)],
            "policy": POLICY, "planned_rows": 8, "fault_rows": 6,
            "normal_host_gate_rows": 2, "same_cases_as_parent": True,
            "same_source_measurements_tools_and_budgets": True,
            "registry_prediction_metadata_newly_exposed": True,
            "probe_budget_per_row": L2.MAX_PAIRED_PROBES,
            "model_action_budget_per_row": L2.MAX_MODEL_ACTIONS,
            "registered_candidate_partitions": True,
            "probability_or_eig": False, "hold_always_allowed": True,
            "normal_gate_is_not_model_success": True, "automatic_repair": False,
        },
        "bounds": {"total_wall_s": MAX_WALL_S, "global_api_requests": MAX_REQUESTS,
                   "paired_probes": MAX_PAIRED_PROBES,
                   "sdk_role_executions": MAX_SDK_ROLE_EXECUTIONS},
        "analysis": {"first_decision_valid_denominator": 6,
                     "confirmation_and_hold_separate": True,
                     "symptom_and_confirmed_cause_separate": True,
                     "post_exposure_development": True,
                     "compare_to_parent_evidence_guided_rows": True,
                     "prediction": ("registered semantics may reduce pre-observation HOLD; "
                                    "improvement not guaranteed"),
                     "reporting": {
                         "fault_denominator": 6,
                         "separate": ["first_valid", "read", "probe", "HOLD",
                                      "correct_confirmed_family", "compound_partial",
                                      "wrong_family_confirmation"],
                         "normal_host_gate_rows": 2,
                         "normal_host_gate_is_model_success": False,
                         "incomplete_or_infrastructure": "indeterminate_with_denominator_preserved",
                     },
                     "development_candidate_rule": {
                         "normal_rows_api_and_probe_zero": True,
                         "wrong_family_confirmations": 0,
                         "first_valid_at_least": "5/6",
                         "correct_confirmed_family_strictly_above_parent_guided": "0/6",
                         "all_six_hold": "not_improved",
                         "format_only_improvement": "not_adopted",
                         "positive_result": "not_new_seed_effect_or_default_adoption",
                     },
                     "research": {
                         "learning_to_acquire_information_uai2017":
                             "https://proceedings.mlr.press/r15/pu17a.html",
                         "mediq_neurips2024": ("https://proceedings.neurips.cc/paper_files/paper/2024/"
                                              "hash/32b80425554e081204e5988ab1c97e9a-Abstract-Conference.html"),
                         "paper_effect_sizes_borrowed": False,
                     }},
        "claims": {"generalization": False, "default_mode_adoption": False,
                   "diagnosis_improvement": False, "recovery": False,
                   "model_version_advantage": False, "cost_savings": False},
    }


def verify_protocol(path: Path, anchor: str, lab: Path) -> dict[str, Any]:
    BASE._require(re.fullmatch(r"[0-9a-f]{40}", anchor) is not None, "anchor 형식 불일치")
    value = BASE._read(path)
    parent_ref = value.get("parent_study_protocol")
    BASE._require(value.get("schema") == SCHEMA and isinstance(parent_ref, dict),
                  "diagnostic option protocol schema 불일치")
    parent_path = ROOT / cast(dict[str, str], parent_ref)["path"]
    prior_sha = value.get("prior_option_result_sha256")
    BASE._require(isinstance(prior_sha, str), "prior option result SHA가 없다")
    BASE._require(value == prepare_protocol(parent_path, cast(str, prior_sha)),
                  "diagnostic option protocol이 현재 source와 다르다")
    BASE.verify_protocol(parent_path, anchor, Path(lab).resolve())
    BASE._git("cat-file", "-e", f"{anchor}^{{commit}}")
    BASE._git("merge-base", "--is-ancestor", anchor, "HEAD")
    BASE._verify_anchor_file(path, anchor)
    for relative, digest in value["source_sha256_lf"].items():
        blob = BASE._git("show", f"{anchor}:{relative}").replace(b"\r\n", b"\n")
        BASE._require(hashlib.sha256(blob).hexdigest() == digest,
                      f"anchor source blob 불일치: {relative}")
    return deepcopy(value)


def _prior(path: Path, spec: dict[str, Any]) -> dict[str, Any]:
    BASE._require(BASE._sha(path) == spec["prior_option_result_sha256"],
                  "prior option result SHA 불일치")
    value = BASE._read(path)
    BASE._require(value.get("schema") == BASE.RESULT_SCHEMA and value.get("status") == "complete",
                  "prior option result가 complete가 아니다")
    rows = value.get("rows")
    BASE._require(isinstance(rows, list) and len(rows) == 16, "prior 8×2 행이 완전하지 않다")
    checked_rows = cast(list[dict[str, Any]], rows)
    expected = {(f"c{i:02d}", policy) for i in range(1, 9) for policy in BASE.POLICIES}
    BASE._require({(row.get("case_id"), row.get("policy")) for row in checked_rows} == expected,
                  "prior option case/policy 분모가 다르다")
    parent_path = ROOT / spec["parent_study_protocol"]["path"]
    BASE._require(value.get("protocol") == BASE._ref(parent_path),
                  "prior option protocol 결합 불일치")
    return value


def _validate_inputs(protocol_path: Path, anchor: str, source_batch: Path,
                     prior_result: Path, lab: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    spec = verify_protocol(protocol_path, anchor, lab)
    prior = _prior(prior_result, spec)
    parent = BASE._read(ROOT / spec["parent_study_protocol"]["path"])
    base = BASE._read(ROOT / parent["base_comparison_protocol"]["path"])
    cases = RUN._source_cases(source_batch, protocol=base, anchor=anchor)
    BASE._require(prior.get("source_batch") == BASE._ref(source_batch),
                  "prior option/source batch 결합 불일치")
    return spec, cases, prior


def _verify_current_protocol(protocol_path: Path, spec: dict[str, Any], lab: Path) -> None:
    BASE._require(BASE._read(protocol_path) == spec, "실행 중 diagnostic option protocol 변경")
    parent_path = ROOT / spec["parent_study_protocol"]["path"]
    BASE._require(prepare_protocol(parent_path, spec["prior_option_result_sha256"]) == spec,
                  "실행 중 V2 option/source closure 변경")
    parent = BASE._read(parent_path)
    BASE._verify_current_protocol(parent_path, parent, Path(lab).resolve())


def _guarded_complete_factory(transport_factory: Callable,
                              verify_context: Callable[[], dict[str, Any]]) -> Callable:
    def factory(slot: str, state: dict) -> Callable:
        complete = transport_factory(slot, state)

        def guarded(messages: list, params: dict) -> dict:
            before = verify_context()
            BASE._require(before["requests"] < MAX_REQUESTS,
                          "새 요청 전 global API cap이 소진됐다")
            result = complete(messages, params)
            verify_context()
            return result

        return guarded

    return factory


def preflight(protocol: Path, anchor: str, source_batch: Path, prior_result: Path,
              lab: Path, out: Path, transport_root: Path) -> dict[str, Any]:
    out = Path(out).resolve()
    BASE._require(not out.exists(), "preflight out은 fresh 경로여야 한다")
    spec, cases, prior = _validate_inputs(protocol, anchor, source_batch, prior_result, lab)
    lease = RUN._lease()
    BASE._require(isinstance(lease.get("leaseId"), str) and bool(lease["leaseId"]),
                  "leaseId가 없는 공용 lease다")
    result = {"schema": PREFLIGHT_SCHEMA, "status": "ready", "protocol": BASE._ref(protocol),
              "source_batch": BASE._ref(source_batch), "prior_result": BASE._ref(prior_result),
              "prior_rows": len(prior["rows"]), "opaque_cases": [row["case_id"] for row in cases],
              "planned_rows": 8, "fault_rows": 6, "model_requests": 0,
              "paired_probes": 0, "gpu_started": False, "leaseId": lease["leaseId"],
              "transport_status": T4.status(Path(transport_root).resolve()),
              "claims": deepcopy(spec["claims"])}
    BASE._write(out / "preflight.json", result)
    return result


def execute(protocol: Path, anchor: str, source_batch: Path, prior_result: Path,
            lab: Path, out: Path, transport_root: Path, *,
            run_epoch: Callable = L2.run_epoch,
            execute_host: Callable = HOST.execute_pair) -> dict[str, Any]:
    out = Path(out).resolve()
    BASE._require(not out.exists(), "study out은 fresh 경로여야 한다")
    spec, cases, prior = _validate_inputs(protocol, anchor, source_batch, prior_result, lab)
    source_sha, prior_sha = BASE._sha(source_batch), BASE._sha(prior_result)
    references = {row["seed"]: row for row in cases if row["category"] == "NONE"}
    BASE._require(set(references) == {2320, 2321}, "same-seed NONE reference가 완전하지 않다")
    parent = BASE._read(ROOT / spec["parent_study_protocol"]["path"])
    base_protocol = BASE._read(ROOT / parent["base_comparison_protocol"]["path"])
    bands = BASE._read(RUN.BANDS)["bands"]
    lease_id = RUN._lease().get("leaseId")
    BASE._require(isinstance(lease_id, str) and bool(lease_id), "leaseId가 없다")
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    rows: list[dict[str, Any]] = []
    probe_count = 0
    active: dict[str, Any] = {}
    transport_factory = COMP.transport_factory(Path(transport_root).resolve())

    def verify_context(context: dict[str, Any] | None = None) -> dict[str, Any]:
        elapsed = time.monotonic() - started
        BASE._require(math.isfinite(elapsed) and elapsed < MAX_WALL_S, "study wall 상한 종료")
        _verify_current_protocol(protocol, spec, lab)
        BASE._require(BASE._sha(source_batch) == source_sha and BASE._sha(prior_result) == prior_sha,
                      "실행 중 source/prior 변경")
        lease = RUN._lease()
        BASE._require(lease.get("leaseId") == lease_id, "실행 중 leaseId 변경")
        status = T4.status(Path(transport_root).resolve())
        requests = status.get("request_count", status.get("requests", 0))
        BASE._require(type(requests) is int and 0 <= requests <= MAX_REQUESTS,
                      "transport request cap 불일치")
        checked_role = None
        if context is not None:
            BASE._require(context.get("schema") == "evidence_guided_probe_host_context_v1",
                          "host context schema 불일치")
            checked_role = context.get("role")
            BASE._require(checked_role in ("case", "reference"), "host role 불일치")
            expected = active[cast(str, checked_role)]
            public, metadata = INPUT.checked(expected["input"])
            binding = context.get("request_spec", {}).get("source_binding", {}).get(checked_role)
            BASE._require(isinstance(binding, dict)
                          and binding.get("checkpoint_sha256") == expected["checkpoint_sha256"]
                          == public["source_binding"]["checkpoint_sha256"],
                          "host checkpoint 불일치")
            runtime_ref = metadata["runtime_facts"][checked_role]
            BASE._require(Path(runtime_ref["path"]).resolve() == expected["runtime"]
                          and BASE._sha(expected["runtime"]) == runtime_ref["sha256"],
                          "host runtime ref 불일치")
        return {"approved": True, "elapsed_s": elapsed, "requests": requests,
                "checked_role": checked_role, "leaseId": lease_id}

    complete_factory = _guarded_complete_factory(transport_factory, verify_context)

    try:
        for case in cases:
            reference = references[case["seed"]]
            gate = RUN._gate(case, reference, bands)
            base = {"case_id": case["case_id"], "policy": POLICY,
                    "checkpoint_sha256": case["checkpoint"]["sha256"],
                    "reference_checkpoint_sha256": reference["checkpoint"]["sha256"],
                    "gate": deepcopy(gate)}
            if not gate["run"]:
                rows.append({**base, "status": ("complete" if gate["reason"] == "healthy_gate"
                                                 and gate.get("analysis_exclusion") is None else "excluded"),
                             "decision_outcome": "HOLD", "model_requests": 0,
                             "model_actions": 0, "paired_probes": 0,
                             "first_decision_valid": None,
                             "post_feedback_first_decision_valid": None,
                             "correct_confirmed_family": False, "correct_complete_family": False,
                             "compound_partial": False, "compound_complete": False,
                             "normal_gate_model_success": False})
                continue
            row_root = out / case["case_id"] / POLICY
            with COMP.CACHE.proof_scope() as cache_stats:
                inputs = BASE._inputs(case, reference, row_root, eager=False)
                O2.context(inputs["state"])
                case_runtime = cast(Path, inputs["case_runtime"])
                reference_runtime = cast(Path, inputs["reference_runtime"])
                active.clear()
                active.update({
                    "case": {"input": case["input"], "runtime": case_runtime,
                             "checkpoint_sha256": case["checkpoint"]["sha256"]},
                    "reference": {"input": reference["input"], "runtime": reference_runtime,
                                  "checkpoint_sha256": reference["checkpoint"]["sha256"]},
                })

                def pair(state: dict, action: dict, stage: Path, *, _case=case,
                         _reference=reference, _case_runtime=case_runtime,
                         _reference_runtime=reference_runtime) -> dict[str, Any]:
                    nonlocal probe_count
                    BASE._require(probe_count < MAX_PAIRED_PROBES, "paired probe global cap 종료")
                    protocols = base_protocol["protocols"]
                    result = execute_host(
                        state, action, case_input=_case["input"], reference_input=_reference["input"],
                        case_runtime=_case_runtime, reference_runtime=_reference_runtime,
                        probe_protocol=ROOT / protocols["paired_probe"]["path"],
                        view_protocol=ROOT / protocols["view"]["path"], anchor=anchor,
                        lab=Path(lab).resolve(), out=stage, verify_context=verify_context)
                    probe_count += 1
                    return result

                epoch_root = row_root / "epoch"
                epoch = run_epoch(inputs["state"], run_id=_run_id(case["case_id"]),
                                  out=epoch_root, verify_context=verify_context,
                                  complete_factory=complete_factory, execute_pair=pair)
            COMP.write_new(row_root / "proof-cache-stats.json", cache_stats)
            metrics = BASE._dialogue_metrics(epoch_root, epoch)
            rows.append({**base, "status": epoch["status"],
                         "decision_outcome": epoch["decision_outcome"],
                         "model_actions": epoch["model_actions"],
                         "paired_probes": epoch["paired_probes"],
                         "parent_probe_wall_s": epoch["parent_probe_wall_s"],
                         "owned_child_wall_s": epoch["owned_child_wall_s"],
                         "final_status": epoch["final_status"], **metrics,
                         **BASE._score(case["category"], epoch),
                         "normal_gate_model_success": False})
        verify_context()
        BASE._require(len(rows) == 8, "계획된 8행 분모가 보존되지 않았다")
        requests = sum(row["model_requests"] for row in rows)
        probes = sum(row["paired_probes"] for row in rows)
        BASE._require(requests <= MAX_REQUESTS and probes <= MAX_PAIRED_PROBES,
                      "study 전역 상한 초과")
        fault = [row for row in rows if row["gate"].get("run")]
        BASE._require(len(fault) == 6, "등록 fault 6행 분모가 다르다")
        prior_validity = prior.get("validity", {})
        BASE._require(prior_validity.get("first_decision_denominator") == 12,
                      "prior first-decision 분모 불일치")
        result = {"schema": RESULT_SCHEMA, "status": "complete", "protocol": BASE._ref(protocol),
                  "source_batch": BASE._ref(source_batch), "prior_result": BASE._ref(prior_result),
                  "rows": rows, "planned_rows": 8, "fault_rows": 6,
                  "normal_host_gate_rows": 2, "model_requests": requests,
                  "paired_probes": probes, "sdk_role_executions_upper_bound": probes * 2,
                  "cpu_parent_wall_s": time.monotonic() - started,
                  "validity": {"first_decision_valid": sum(
                      row["first_decision_valid"] is True for row in fault),
                      "first_decision_denominator": 6,
                      "prior_guided_valid_fault_rows": sum(
                          row.get("first_decision_valid") is True for row in prior["rows"]
                          if row.get("policy") == POLICY and row.get("gate", {}).get("run"))},
                  "post_exposure_development": True, "claims": deepcopy(spec["claims"]),
                  "automatic_repair": False}
        BASE._write(out / "result.json", result)
        return result
    except BaseException as error:
        BASE._write(out / "failure.json", {"schema": RESULT_SCHEMA, "status": "invalid",
                    "error_type": type(error).__name__, "error": str(error), "rows": rows,
                    "automatic_retry": False})
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare-protocol")
    prepare.add_argument("--parent-study-protocol", type=Path, required=True)
    prepare.add_argument("--prior-option-result-sha256", required=True)
    prepare.add_argument("--out", type=Path, required=True)
    for command in ("preflight", "execute"):
        item = sub.add_parser(command)
        item.add_argument("--protocol", type=Path, required=True)
        item.add_argument("--anchor", required=True)
        item.add_argument("--source-batch", type=Path, required=True)
        item.add_argument("--prior-result", type=Path, required=True)
        item.add_argument("--lab", type=Path, required=True)
        item.add_argument("--out", type=Path, required=True)
        item.add_argument("--transport-root", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare-protocol":
        BASE._write(args.out, prepare_protocol(args.parent_study_protocol,
                                               args.prior_option_result_sha256))
    elif args.command == "preflight":
        preflight(args.protocol, args.anchor, args.source_batch, args.prior_result,
                  args.lab, args.out, args.transport_root)
    else:
        execute(args.protocol, args.anchor, args.source_batch, args.prior_result,
                args.lab, args.out, args.transport_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["execute", "main", "preflight", "prepare_protocol", "verify_protocol"]
