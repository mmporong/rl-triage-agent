"""등록 option 계약을 기존 전향 source에 적용하는 development 후속 실행기다."""
# pyright: reportMissingImports=false
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
import time
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "bench"), str(ROOT / "src")]

import diagnostic_fact_ledger_v1 as FACTS
import evidence_guided_probe_host_v1 as HOST
import evidence_guided_selector_v1 as SEL
import lineage_nim_transport_v4 as T4
import prospective_agentic_comparison_v1 as COMP
import prospective_diagnosis_inputs_v2 as INPUT
import registry_action_loop_v1 as L
import registry_action_options_v1 as O
import run_prospective_agentic_comparison_v1 as RUN

SCHEMA = "registry_action_study_protocol_v1"
PREFLIGHT_SCHEMA = "registry_action_study_preflight_v1"
RESULT_SCHEMA = "registry_action_study_result_v1"
BASE_ANCHOR = "164348c2b8df8f275371193306df6aa91d5d312b"
POLICIES = ("eager_model", "evidence_guided_model")
MAX_WALL_S = 14400
MAX_REQUESTS = 96
MAX_PAIRED_PROBES = 36
MAX_SDK_ROLE_EXECUTIONS = 72
SOURCE_FILES = (
    "evals/registry_action_options_v1.py",
    "tests/test_registry_action_options_v1.py",
    "evals/registry_action_loop_v1.py",
    "tests/test_registry_action_loop_v1.py",
    "evals/run_registry_action_study_v1.py",
    "tests/test_run_registry_action_study_v1.py",
)


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=lambda token: (
        _ for _ in ()).throw(ValueError(token)))
    _require(isinstance(value, dict), "JSON root는 객체여야 한다")
    return value


def _sha(path: Path, *, lf: bool = False) -> str:
    raw = Path(path).read_bytes()
    if lf:
        raw = raw.replace(b"\r\n", b"\n")
    return hashlib.sha256(raw).hexdigest()


def _write(path: Path, value: object) -> None:
    path = Path(path).resolve()
    _require(not path.exists(), "출력은 write-once여야 한다")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def _ref(path: Path) -> dict[str, str]:
    path = Path(path).resolve()
    _require(path.is_file(), f"근거 파일이 없다: {path.name}")
    return {"path": str(path), "sha256": _sha(path)}


def _public_ref(path: Path) -> dict[str, str]:
    path = Path(path).resolve()
    _require(path.is_relative_to(ROOT), "공개 protocol 경로가 저장소를 이탈했다")
    return {"path": path.relative_to(ROOT).as_posix(), "sha256_lf": _sha(path, lf=True)}


def _git(*args: str) -> bytes:
    return subprocess.run(["git", *args], cwd=ROOT, check=True,
                          capture_output=True).stdout


def _verify_anchor_file(path: Path, anchor: str) -> None:
    """현재 공개 파일과 동일 경로의 immutable anchor blob이 같은지 확인한다."""
    path = Path(path).resolve()
    _require(path.is_relative_to(ROOT), "anchor 대상이 저장소를 이탈했다")
    relative = path.relative_to(ROOT).as_posix()
    blob = _git("show", f"{anchor}:{relative}").replace(b"\r\n", b"\n")
    _require(hashlib.sha256(blob).hexdigest() == _sha(path, lf=True),
             f"anchor protocol blob 불일치: {relative}")


def prepare_protocol(base_comparison_protocol: Path,
                     prior_comparison_result_sha256: str) -> dict[str, Any]:
    """v5 source closure와 새 option 실행기만 결합한 공개 후보를 만든다."""
    base_path = Path(base_comparison_protocol).resolve()
    base = _read(base_path)
    _require(base.get("schema") == RUN.SCHEMA, "base comparison protocol schema 불일치")
    _require(re.fullmatch(r"[0-9a-f]{64}", prior_comparison_result_sha256) is not None,
             "prior result SHA 형식 불일치")
    _require(base.get("bounds") == {
        "global_api_requests": 96, "legacy_role_executions_worst_case": 144,
        "paired_probes": 72, "total_wall_s": 14400}, "v5 실행 상한 불일치")
    _require(base.get("design", {}).get("opaque_cases") == [f"c{i:02d}" for i in range(1, 9)],
             "v5 opaque case 설계 불일치")
    for relative in SOURCE_FILES:
        _require((ROOT / relative).is_file(), f"source closure 파일이 없다: {relative}")
    return {
        "schema": SCHEMA,
        "purpose": "post_exposure_registry_action_contract_development_study",
        "base_comparison_protocol": _public_ref(base_path),
        "base_comparison_anchor": BASE_ANCHOR,
        "prior_comparison_result_sha256": prior_comparison_result_sha256,
        "source_sha256_lf": {name: _sha(ROOT / name, lf=True) for name in SOURCE_FILES},
        "inherited_source_sha256_lf": deepcopy(base["source_sha256_lf"]),
        "design": {
            "opaque_cases": [f"c{i:02d}" for i in range(1, 9)],
            "policies": list(POLICIES), "planned_rows": 16,
            "fault_rows": 12, "normal_host_gate_rows": 4,
            "same_information_tools_and_budgets": True,
            "probe_budget_per_row": L.MAX_PAIRED_PROBES,
            "model_action_budget_per_row": L.MAX_MODEL_ACTIONS,
            "normal_gate_is_not_model_success": True,
            "prior_strong_rule_reference_only": True,
            "automatic_repair": False,
        },
        "bounds": {"total_wall_s": MAX_WALL_S, "global_api_requests": MAX_REQUESTS,
                   "paired_probes": MAX_PAIRED_PROBES,
                   "sdk_role_executions": MAX_SDK_ROLE_EXECUTIONS},
        "analysis": {
            "first_decision_valid_denominator": 12,
            "feedback_denominator": "initial_invalid_fault_rows",
            "read_and_probe_actions_separate": True,
            "hold_coverage_and_correct_family_separate": True,
            "format_candidate_rule": "first_decision_valid_at_least_10_of_12_and_above_prior",
            "downstream_confirmation_is_post_exposure_development": True,
        },
        "claims": {"generalization": False, "default_mode_adoption": False,
                   "recovery": False, "model_version_advantage": False,
                   "strong_rule_speed_comparison": False},
    }


def verify_protocol(path: Path, anchor: str, lab: Path) -> dict[str, Any]:
    _require(re.fullmatch(r"[0-9a-f]{40}", anchor) is not None, "anchor 형식 불일치")
    value = _read(path)
    base_ref = value.get("base_comparison_protocol")
    _require(value.get("schema") == SCHEMA and isinstance(base_ref, dict),
             "action study protocol schema 불일치")
    base_path = ROOT / cast(dict[str, str], base_ref)["path"]
    prior_sha = value.get("prior_comparison_result_sha256")
    _require(isinstance(prior_sha, str), "prior result SHA가 없다")
    expected = prepare_protocol(base_path, cast(str, prior_sha))
    _require(value == expected, "action study protocol이 현재 source와 다르다")
    RUN.verify_protocol(base_path, anchor, Path(lab).resolve())
    _git("cat-file", "-e", f"{anchor}^{{commit}}")
    _git("merge-base", "--is-ancestor", anchor, "HEAD")
    _verify_anchor_file(path, anchor)
    for relative, digest in value["source_sha256_lf"].items():
        blob = _git("show", f"{anchor}:{relative}").replace(b"\r\n", b"\n")
        _require(hashlib.sha256(blob).hexdigest() == digest,
                 f"anchor source blob 불일치: {relative}")
    return deepcopy(value)


def _prior(path: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    _require(_sha(path) == protocol["prior_comparison_result_sha256"], "prior result SHA 불일치")
    value = _read(path)
    _require(value.get("schema") == RUN.RESULT_SCHEMA and value.get("status") == "complete",
             "prior comparison result가 complete가 아니다")
    rows = value.get("rows")
    _require(isinstance(rows, list) and len(rows) == 24, "prior 8×3 행이 완전하지 않다")
    rows = cast(list[dict[str, Any]], rows)
    identities = {(row.get("case_id"), row.get("policy")) for row in rows if isinstance(row, dict)}
    expected = {(f"c{i:02d}", policy) for i in range(1, 9) for policy in COMP.POLICIES}
    _require(identities == expected, "prior case/policy 분모가 다르다")
    base_path = ROOT / protocol["base_comparison_protocol"]["path"]
    _require(value.get("protocol") == _ref(base_path), "prior result protocol 결합 불일치")
    return value


def _inputs(case: dict[str, Any], reference: dict[str, Any], root: Path,
            *, eager: bool) -> dict[str, Any]:
    envelope = INPUT.make_packet(case["input"], reference["input"], budget_left=3)
    case_runtime = Path(case["metadata"]["runtime_facts"]["case"]["path"])
    reference_runtime = Path(reference["metadata"]["runtime_facts"]["reference"]["path"])
    ledger = FACTS.build(case_runtime, reference_runtime)
    ledger_path = root / "fact-ledger.json"
    FACTS.write_new(ledger_path, ledger)
    state = SEL.initialize(envelope["packet"], ledger_path, budget_left=3)
    if eager:
        state = SEL.advance(state, {"action": "read_evidence", "target": SEL.RESOURCE,
                                    "fact_ids": []})
    O.context(state)
    return {"state": state, "envelope": envelope, "case_runtime": case_runtime,
            "reference_runtime": reference_runtime}


def _dialogue_metrics(epoch_root: Path, epoch: dict[str, Any]) -> dict[str, Any]:
    first_valid: bool | None = None
    first_final_valid: bool | None = None
    feedback = 0
    requests = 0
    invalid_codes: list[str] = []
    reads = 0
    probes = 0
    holds = 0
    for step in epoch.get("steps", []):
        action = step.get("action", {})
        reads += int(action.get("action") == "read_evidence")
        probes += int(action.get("action") == "run_probe")
        holds += int(action.get("action") == "hold")
        dialogue_ref = step.get("dialogue")
        if not isinstance(dialogue_ref, dict):
            continue
        path = Path(dialogue_ref["path"])
        _require(path.is_relative_to(epoch_root) and _sha(path) == dialogue_ref.get("sha256"),
                 "dialogue raw ref 불일치")
        dialogue = _read(path)
        calls = dialogue.get("calls")
        _require(isinstance(calls, list) and 1 <= len(calls) <= 2, "dialogue call 수 불일치")
        calls = cast(list[dict[str, Any]], calls)
        requests += len(calls)
        initial = calls[0].get("review", {}).get("status") == "valid"
        if first_valid is None:
            first_valid = initial
            first_final_valid = dialogue.get("model_valid") is True
            invalid_codes = list(calls[0].get("review", {}).get("codes", []))
        feedback += int(dialogue.get("semantic_feedback_used") is True)
    return {"first_decision_valid": first_valid, "post_feedback_first_decision_valid": first_final_valid,
            "first_decision_invalid_codes": invalid_codes, "feedback_actions": feedback,
            "model_requests": requests, "read_actions": reads, "probe_actions": probes,
            "hold_actions": holds}


def _prior_first_decisions(prior_path: Path, rows: list[dict[str, Any]]) -> int:
    """과거 fault 행의 첫 모델 선택 중 첫 응답이 유효한 경우만 센다."""
    fault_rows = [row for row in rows
                  if row.get("policy") in POLICIES and row.get("gate", {}).get("run") is True]
    _require(len(fault_rows) == 12, "prior 첫 결정 fault 분모가 12가 아니다")
    valid = 0
    root = Path(prior_path).resolve().parent
    for row in fault_rows:
        epoch_root = root / row["case_id"] / row["policy"]
        epoch_path = epoch_root / "result.json"
        epoch = _read(epoch_path)
        _require(epoch.get("schema") == "prospective_agentic_epoch_v1"
                 and epoch.get("status") == "complete"
                 and epoch.get("run_id") == f"{row['case_id']}.{row['policy']}",
                 "prior epoch result 결합 불일치")
        aggregate = RUN._validity(epoch_root, epoch)
        _require(all(row.get(name) == aggregate[name] for name in aggregate),
                 "prior aggregate validity가 epoch 원문과 다르다")
        first_review: dict[str, Any] | None = None
        for index, step in enumerate(epoch.get("steps", []), 1):
            digest = step.get("dialogue_sha256")
            if digest is None:
                continue
            dialogue_path = epoch_root / f"step-{index:02d}" / "dialogue.json"
            _require(dialogue_path.is_file() and _sha(dialogue_path) == digest,
                     "prior 첫 model dialogue SHA 불일치")
            dialogue = _read(dialogue_path)
            calls = dialogue.get("calls")
            _require(isinstance(calls, list) and bool(calls)
                     and isinstance(calls[0], dict), "prior 첫 dialogue calls 불일치")
            checked_calls = cast(list[dict[str, Any]], calls)
            first_review = checked_calls[0].get("review")
            _require(isinstance(first_review, dict), "prior 첫 응답 review가 없다")
            first_review = cast(dict[str, Any], first_review)
            break
        _require(first_review is not None, "prior fault 행에 model dialogue가 없다")
        valid += int(cast(dict[str, Any], first_review).get("status") == "valid")
    return valid


def _score(category: str, epoch: dict[str, Any]) -> dict[str, Any]:
    score = RUN._score(category, epoch)
    return {**score, "hold": epoch.get("decision_outcome") == "HOLD"}


def _validate_inputs(protocol_path: Path, anchor: str, source_batch: Path,
                     prior_result: Path, lab: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    protocol = verify_protocol(protocol_path, anchor, lab)
    prior = _prior(prior_result, protocol)
    cases = RUN._source_cases(source_batch, protocol={
        **_read(ROOT / protocol["base_comparison_protocol"]["path"])}, anchor=anchor)
    _require(prior.get("source_batch") == _ref(source_batch), "prior/source batch 결합 불일치")
    return protocol, cases, prior


def _verify_current_protocol(protocol_path: Path, protocol: dict[str, Any], lab: Path) -> None:
    """anchor 검사는 진입 시 한 번, 실행 중에는 현재 bytes와 runtime pin만 다시 확인한다."""
    _require(_read(protocol_path) == protocol, "실행 중 action study protocol이 바뀌었다")
    base_path = ROOT / protocol["base_comparison_protocol"]["path"]
    current = prepare_protocol(base_path, protocol["prior_comparison_result_sha256"])
    _require(current == protocol, "실행 중 option/source closure가 바뀌었다")
    base = _read(base_path)
    RUN._verify_current_protocol(base_path, base, Path(lab).resolve())


def preflight(protocol: Path, anchor: str, source_batch: Path, prior_result: Path,
              lab: Path, out: Path, transport_root: Path) -> dict[str, Any]:
    out = Path(out).resolve()
    _require(not out.exists(), "preflight out은 fresh 경로여야 한다")
    value, cases, prior = _validate_inputs(protocol, anchor, source_batch, prior_result, lab)
    lease = RUN._lease()
    _require(isinstance(lease.get("leaseId"), str) and bool(lease["leaseId"]),
             "leaseId가 없는 공용 lease다")
    status = T4.status(Path(transport_root).resolve())
    result = {"schema": PREFLIGHT_SCHEMA, "status": "ready", "protocol": _ref(protocol),
              "source_batch": _ref(source_batch), "prior_result": _ref(prior_result),
              "prior_rows": len(prior["rows"]), "opaque_cases": [row["case_id"] for row in cases],
              "planned_rows": 16, "model_requests": 0, "paired_probes": 0,
              "gpu_started": False, "leaseId": lease["leaseId"],
              "transport_status": status, "claims": deepcopy(value["claims"])}
    _write(out / "preflight.json", result)
    return result


def execute(protocol: Path, anchor: str, source_batch: Path, prior_result: Path,
            lab: Path, out: Path, transport_root: Path, *,
            run_epoch: Callable = L.run_epoch, execute_host: Callable = HOST.execute_pair) -> dict[str, Any]:
    out = Path(out).resolve()
    _require(not out.exists(), "study out은 fresh 경로여야 한다")
    spec, cases, prior = _validate_inputs(protocol, anchor, source_batch, prior_result, lab)
    source_sha, prior_sha = _sha(source_batch), _sha(prior_result)
    references = {row["seed"]: row for row in cases if row["category"] == "NONE"}
    _require(set(references) == {2320, 2321}, "same-seed NONE reference가 완전하지 않다")
    bands = _read(RUN.BANDS)["bands"]
    lease_id = RUN._lease().get("leaseId")
    _require(isinstance(lease_id, str) and bool(lease_id), "leaseId가 없다")
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    rows: list[dict[str, Any]] = []
    probe_count = 0
    active: dict[str, Any] = {}
    complete_factory = COMP.transport_factory(Path(transport_root).resolve())

    def verify_context(context: dict[str, Any] | None = None) -> dict[str, Any]:
        elapsed = time.monotonic() - started
        _require(math.isfinite(elapsed) and elapsed < MAX_WALL_S, "study wall 상한 종료")
        _verify_current_protocol(protocol, spec, lab)
        _require(_sha(source_batch) == source_sha and _sha(prior_result) == prior_sha,
                 "실행 중 protocol/source/prior가 바뀌었다")
        lease = RUN._lease()
        _require(lease.get("leaseId") == lease_id, "실행 중 leaseId가 바뀌었다")
        status = T4.status(Path(transport_root).resolve())
        requests = status.get("request_count", status.get("requests", 0))
        _require(type(requests) is int and 0 <= requests <= MAX_REQUESTS,
                 "transport request cap 불일치")
        checked_role = None
        if context is not None:
            _require(context.get("schema") == "evidence_guided_probe_host_context_v1",
                     "host context schema 불일치")
            checked_role = context.get("role")
            _require(checked_role in ("case", "reference"), "host role 불일치")
            checked_role = cast(str, checked_role)
            expected = active[checked_role]
            public, metadata = INPUT.checked(expected["input"])
            binding = context.get("request_spec", {}).get("source_binding", {}).get(checked_role)
            _require(isinstance(binding, dict)
                     and binding.get("checkpoint_sha256") == expected["checkpoint_sha256"]
                     == public["source_binding"]["checkpoint_sha256"], "host checkpoint 불일치")
            runtime_ref = metadata["runtime_facts"][checked_role]
            _require(Path(runtime_ref["path"]).resolve() == expected["runtime"]
                     and _sha(expected["runtime"]) == runtime_ref["sha256"],
                     "host runtime ref 불일치")
            envelope, _facts = HOST.RECORDED.FACT.checked(
                expected["runtime"], expected_checkpoint_sha256=expected["checkpoint_sha256"])
            _require(envelope["source_binding"]["binding_digest"] == binding.get("binding_digest"),
                     "host runtime binding 불일치")
        return {"approved": True, "elapsed_s": elapsed, "requests": requests,
                "checked_role": checked_role, "leaseId": lease_id}

    try:
        for case_index, case in enumerate(cases):
            reference = references[case["seed"]]
            gate = RUN._gate(case, reference, bands)
            order = tuple(policy for policy in COMP.policy_order(case_index) if policy in POLICIES)
            for policy in order:
                base = {"case_id": case["case_id"], "policy": policy,
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
                row_root = out / case["case_id"] / policy
                with COMP.CACHE.proof_scope() as cache_stats:
                    inputs = _inputs(case, reference, row_root, eager=policy == "eager_model")
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
                        _require(probe_count < MAX_PAIRED_PROBES, "paired probe global cap 종료")
                        base_protocol = _read(ROOT / spec["base_comparison_protocol"]["path"])
                        protocols = base_protocol["protocols"]
                        result = execute_host(
                            state, action, case_input=_case["input"],
                            reference_input=_reference["input"], case_runtime=_case_runtime,
                            reference_runtime=_reference_runtime,
                            probe_protocol=ROOT / protocols["paired_probe"]["path"],
                            view_protocol=ROOT / protocols["view"]["path"],
                            anchor=anchor, lab=Path(lab).resolve(), out=stage,
                            verify_context=verify_context)
                        probe_count += 1
                        return result

                    epoch_root = row_root / "epoch"
                    epoch = run_epoch(inputs["state"], run_id=f"{case['case_id']}.{policy}",
                                      out=epoch_root, verify_context=verify_context,
                                      complete_factory=complete_factory, execute_pair=pair)
                COMP.write_new(row_root / "proof-cache-stats.json", cache_stats)
                metrics = _dialogue_metrics(epoch_root, epoch)
                rows.append({**base, "status": epoch["status"],
                             "decision_outcome": epoch["decision_outcome"],
                             "model_actions": epoch["model_actions"],
                             "paired_probes": epoch["paired_probes"],
                             "parent_probe_wall_s": epoch["parent_probe_wall_s"],
                             "owned_child_wall_s": epoch["owned_child_wall_s"],
                             "final_status": epoch["final_status"], **metrics,
                             "host_eager_read_actions": int(policy == "eager_model"),
                             **_score(case["category"], epoch),
                             "normal_gate_model_success": False})
        verify_context()
        _require(len(rows) == 16, "계획된 8×2 분모가 보존되지 않았다")
        requests = sum(row["model_requests"] for row in rows)
        probes = sum(row["paired_probes"] for row in rows)
        _require(requests <= MAX_REQUESTS and probes <= MAX_PAIRED_PROBES,
                 "study 전역 상한 초과")
        fault = [row for row in rows if row["gate"].get("run")]
        _require(len(fault) == 12, "등록된 fault 12행 분모가 다르다")
        first_valid = sum(row["first_decision_valid"] is True for row in fault)
        initial_invalid = [row for row in fault if row["first_decision_valid"] is False]
        prior_first = _prior_first_decisions(prior_result, prior["rows"])
        result = {
            "schema": RESULT_SCHEMA, "status": "complete", "protocol": _ref(protocol),
            "source_batch": _ref(source_batch), "prior_result": _ref(prior_result),
            "rows": rows, "planned_rows": 16, "fault_rows": 12, "normal_host_gate_rows": 4,
            "model_requests": requests, "paired_probes": probes,
            "sdk_role_executions_upper_bound": probes * 2,
            "cpu_parent_wall_s": time.monotonic() - started,
            "validity": {"first_decision_valid": first_valid,
                         "first_decision_denominator": 12,
                         "initial_invalid": len(initial_invalid),
                         "post_feedback_valid_among_initial_invalid": sum(
                             row["post_feedback_first_decision_valid"] is True for row in initial_invalid),
                         "prior_first_action_valid_rows": prior_first},
            "format_candidate": first_valid >= 10 and first_valid > prior_first,
            "format_candidate_only": True,
            "post_exposure_development": True,
            "claims": deepcopy(spec["claims"]), "automatic_repair": False,
        }
        _write(out / "result.json", result)
        return result
    except BaseException as error:
        _write(out / "failure.json", {"schema": RESULT_SCHEMA, "status": "invalid",
               "error_type": type(error).__name__, "error": str(error), "rows": rows,
               "automatic_retry": False})
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare-protocol")
    prepare.add_argument("--base-comparison-protocol", type=Path, required=True)
    prepare.add_argument("--prior-comparison-result-sha256", required=True)
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
        _write(args.out, prepare_protocol(args.base_comparison_protocol,
                                          args.prior_comparison_result_sha256))
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
