"""기록된 reasoning 첫 응답을 이어 실제 확정/HOLD 효과를 비교한다.

이 실행기는 reasoning option 연구의 replicate 1 응답을 첫 행동으로 그대로
소비한다. fault 행만 동일 mode로 후속 판단과 승인된 paired 검사를 수행하며,
healthy 행은 기존 host gate에서 끝낸다. 학습·교정·회복 권한은 없다.
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
import nemotron_reasoning_options_study_v1 as MODE1
import nemotron_reasoning_options_study_v2 as MODE2
import prospective_diagnosis_inputs_v2 as INPUT
import reasoning_diagnostic_loop_v1 as LOOP
import registered_reasoning_bridge_v1 as BRIDGE
import run_diagnostic_option_study_v1 as PARENT
import run_prospective_agentic_comparison_v1 as RUN
import run_registry_action_study_v1 as BASE

SCHEMA = "reasoning_diagnosis_study_protocol_v1"
PREFLIGHT_SCHEMA = "reasoning_diagnosis_study_preflight_v1"
RESULT_SCHEMA = "reasoning_diagnosis_study_result_v1"
DIAGNOSTIC_ANCHOR = "640df7a801433d745ed5151393fef4be98cbb9a5"
MODE_ANCHOR = "2b24c911536f737b02bde69a1e9fecfb5fb72c2c"
MODE_RESULT_SHA256 = "3397a80fa70f6aef513a0c9859c976bb454c24215c15d23f8b806158c13b5772"
CORPUS_SHA256 = "825b5ce27b75bbf50c3bfb18e81c1bdac8a284f3610f385cea07de197f61da24"
FAULT_CASES = ("c02", "c03", "c04", "c06", "c07", "c08")
NORMAL_CASES = ("c01", "c05")
MAX_WALL_S = 7200
MAX_NEW_REQUESTS = 72
MAX_PAIRED_PROBES = 36
MAX_SDK_ROLE_EXECUTIONS = 72
SOURCE_FILES = (
    "evals/reasoning_diagnostic_loop_v1.py",
    "tests/test_reasoning_diagnostic_loop_v1.py",
    "evals/registered_reasoning_bridge_v1.py",
    "tests/test_registered_reasoning_bridge_v1.py",
    "evals/run_reasoning_diagnosis_study_v1.py",
    "tests/test_run_reasoning_diagnosis_study_v1.py",
)


def _mode_order(case_id: str) -> tuple[str, str]:
    """원 비교의 ABBA 순서처럼 case index마다 mode 순서를 뒤집는다."""
    BASE._require(re.fullmatch(r"c0[1-8]", case_id) is not None, "case_id 형식 불일치")
    return (("legacy_none", "enabled_low") if int(case_id[1:]) % 2 else
            ("enabled_low", "legacy_none"))


def _logical_slots() -> list[dict[str, str]]:
    return [{"slot_id": f"{case_id}.{mode}.d{decision}.a{attempt}",
             "case_id": case_id, "mode": mode}
            for case_id in FAULT_CASES for mode in _mode_order(case_id)
            for decision in range(2, 5) for attempt in range(1, 3)]


def prepare_bridge(parent_mode_plan: Path) -> dict[str, Any]:
    """공개 bridge protocol 생성을 위한 고정 72-slot helper."""
    return BRIDGE.prepare_protocol(Path(parent_mode_plan).resolve(), _logical_slots(),
                                   max_requests=MAX_NEW_REQUESTS)


def prepare_protocol(parent_diag_protocol: Path, mode_plan: Path, bridge_protocol: Path,
                     mode_result_sha256: str) -> dict[str, Any]:
    parent_path = Path(parent_diag_protocol).resolve()
    mode_path = Path(mode_plan).resolve()
    bridge_path = Path(bridge_protocol).resolve()
    parent = BASE._read(parent_path)
    mode = BASE._read(mode_path)
    bridge = BASE._read(bridge_path)
    BASE._require(parent.get("schema") == PARENT.SCHEMA, "parent diagnostic protocol 불일치")
    BASE._require(mode.get("schema") == MODE2.PLAN_SCHEMA, "reasoning mode plan 불일치")
    BASE._require(bridge == prepare_bridge(mode_path), "reasoning bridge protocol 불일치")
    BASE._require(mode_result_sha256 == MODE_RESULT_SHA256,
                  "고정 reasoning mode result SHA 불일치")
    BASE._require(parent.get("design", {}).get("opaque_cases") ==
                  [f"c{i:02d}" for i in range(1, 9)], "parent 8-case 설계 불일치")
    BASE._require(parent.get("design", {}).get("fault_rows") == 6,
                  "parent fault 분모 불일치")
    for relative in SOURCE_FILES:
        BASE._require((ROOT / relative).is_file(), f"source closure 파일이 없다: {relative}")
    bridge_relative = bridge_path.relative_to(ROOT).as_posix()
    own = {name: BASE._sha(ROOT / name, lf=True) for name in SOURCE_FILES}
    own[bridge_relative] = BASE._sha(bridge_path, lf=True)
    inherited = {**deepcopy(parent.get("inherited_source_sha256_lf", {})),
                 **deepcopy(parent.get("source_sha256_lf", {})),
                 **deepcopy(mode.get("source_sha256_lf", {})),
                 parent_path.relative_to(ROOT).as_posix(): BASE._sha(parent_path, lf=True),
                 mode_path.relative_to(ROOT).as_posix(): BASE._sha(mode_path, lf=True)}
    return {
        "schema": SCHEMA,
        "status": "registered_only_after_commit",
        "purpose": "recorded_first_reasoning_mode_actual_diagnosis_followup_development",
        "parent_diagnostic_protocol": BASE._public_ref(parent_path),
        "parent_diagnostic_anchor": DIAGNOSTIC_ANCHOR,
        "mode_plan": BASE._public_ref(mode_path),
        "mode_plan_anchor": MODE_ANCHOR,
        "bridge_protocol": BASE._public_ref(bridge_path),
        "mode_result_sha256": mode_result_sha256,
        "corpus_sha256": CORPUS_SHA256,
        "source_sha256_lf": own,
        "inherited_source_sha256_lf": inherited,
        "design": {
            "opaque_cases": [f"c{i:02d}" for i in range(1, 9)],
            "modes": list(MODE1.MODES), "planned_rows": 16, "fault_rows": 12,
            "normal_host_gate_rows": 4, "fault_cases": list(FAULT_CASES),
            "normal_cases": list(NORMAL_CASES), "replicate": 1,
            "mode_order": "case_index_alternating_abba",
            "recorded_first_responses": 12, "original_mode_calls": 32,
            "new_model_requests_separate_from_original": True,
            "same_state_source_measurements_tools_and_checkpoint": True,
            "healthy_model_execution": False, "forced_probe": False,
            "automatic_retry": False, "automatic_repair": False,
            "training": False, "correction": False, "recovery": False,
        },
        "bounds": {"total_wall_s": MAX_WALL_S,
                   "new_api_requests": MAX_NEW_REQUESTS,
                   "paired_probes": MAX_PAIRED_PROBES,
                   "sdk_role_executions": MAX_SDK_ROLE_EXECUTIONS},
        "analysis": {
            "post_exposure_development": True,
            "first_proposal_and_confirmed_effect_separate": True,
            "original_first_proposal_reference": {
                "legacy_none_fault_active_non_hold": "5/12",
                "enabled_low_fault_active_non_hold": "11/12",
                "both_modes_contract_valid": "12/12"},
            "followup_replicate": 1, "good_replicate_substitution": False,
            "case_substitution": False,
            "prediction": ("enabled_low may increase information-seeking without increasing "
                           "correct confirmations"),
            "development_candidate_rule": {
                "enabled_low_correct_family_strictly_above_legacy_none": True,
                "wrong_family_confirmations": 0,
                "normal_rows_api_and_probe_zero": True,
                "complete_fault_denominators": True,
                "positive_result_is_not_default_adoption": True,
            },
            "incomplete_or_infrastructure": "indeterminate_with_denominator_preserved",
        },
        "claims": {"diagnosis_improvement": False, "default_mode_adoption": False,
                   "model_version_advantage": False, "new_seed_effect": False,
                   "recovery": False, "speed_advantage": False},
    }


def verify_protocol(path: Path, anchor: str, corpus: Path, lab: Path) -> dict[str, Any]:
    BASE._require(re.fullmatch(r"[0-9a-f]{40}", anchor) is not None, "anchor 형식 불일치")
    path = Path(path).resolve()
    value = BASE._read(path)
    BASE._require(value.get("schema") == SCHEMA, "reasoning diagnosis protocol schema 불일치")
    parent = ROOT / value["parent_diagnostic_protocol"]["path"]
    mode = ROOT / value["mode_plan"]["path"]
    bridge = ROOT / value["bridge_protocol"]["path"]
    BASE._require(value == prepare_protocol(parent, mode, bridge,
                                             value.get("mode_result_sha256", "")),
                  "reasoning diagnosis protocol이 현재 source와 다르다")
    PARENT.verify_protocol(parent, DIAGNOSTIC_ANCHOR, Path(lab).resolve())
    MODE2.verify_registered(mode, Path(corpus).resolve(), MODE_ANCHOR)
    BASE._require(BASE._sha(corpus) == CORPUS_SHA256, "reasoning corpus SHA 불일치")
    BASE._git("cat-file", "-e", f"{anchor}^{{commit}}")
    BASE._git("merge-base", "--is-ancestor", anchor, "HEAD")
    BASE._verify_anchor_file(path, anchor)
    BRIDGE._verify_registered_protocol(bridge, anchor)
    for relative, digest in value["source_sha256_lf"].items():
        blob = BASE._git("show", f"{anchor}:{relative}").replace(b"\r\n", b"\n")
        BASE._require(hashlib.sha256(blob).hexdigest() == digest,
                      f"anchor source blob 불일치: {relative}")
    return deepcopy(value)


def _base_protocol(spec: dict[str, Any]) -> dict[str, Any]:
    parent = BASE._read(ROOT / spec["parent_diagnostic_protocol"]["path"])
    option_parent = BASE._read(ROOT / parent["parent_study_protocol"]["path"])
    return BASE._read(ROOT / option_parent["base_comparison_protocol"]["path"])


def _source_and_gates(source_batch: Path, spec: dict[str, Any], anchor: str
                      ) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    cases = RUN._source_cases(source_batch, protocol=_base_protocol(spec), anchor=anchor)
    BASE._require([(row["case_id"], row["seed"], row["category"]) for row in cases] == [
        (f"c{i + 1:02d}", seed, category)
        for seed in (2320, 2321) for i, category in enumerate(
            ("NONE", "F1", "R1_ONLY", "COMPOUND"), start=(seed - 2320) * 4)],
        "source 8-case seed/category 순서 불일치")
    references = {row["seed"]: row for row in cases if row["category"] == "NONE"}
    BASE._require(set(references) == {2320, 2321}, "same-seed NONE reference 불완전")
    bands = BASE._read(RUN.BANDS)["bands"]
    gates = {row["case_id"]: RUN._gate(row, references[row["seed"]], bands) for row in cases}
    BASE._require({name for name, gate in gates.items() if gate["run"]} == set(FAULT_CASES)
                  and {name for name, gate in gates.items() if not gate["run"]} == set(NORMAL_CASES),
                  "source fault/normal gate 분모 불일치")
    return cases, gates


def _validate_inputs(protocol: Path, anchor: str, source_batch: Path, mode_result: Path,
                     corpus: Path, lab: Path
                     ) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    spec = verify_protocol(protocol, anchor, corpus, lab)
    BASE._require(BASE._sha(mode_result) == MODE_RESULT_SHA256, "reasoning mode result SHA 불일치")
    cases, gates = _source_and_gates(source_batch, spec, anchor)
    return spec, cases, gates


def _verify_first_source(first: dict[str, Any], case: dict[str, Any], reference: dict[str, Any]) -> None:
    binding = first["state"]["fact_ledger"]["source_binding"]
    initial = first["state"]["initial_packet"]["input_bindings"]
    for role, source in (("case", case), ("reference", reference)):
        collected, _metadata = INPUT.checked(source["input"])
        checkpoint = source["checkpoint"]["sha256"]
        BASE._require(binding[role]["checkpoint_sha256"] == checkpoint
                      == initial[role]["checkpoint_sha256"]
                      == collected["source_binding"]["checkpoint_sha256"],
                      f"recorded first/source {role} checkpoint 불일치")


def _bridge_request_count(root: Path) -> int:
    request_root = Path(root).resolve() / "requests"
    return len(list(request_root.glob("[0-9][0-9][0-9].json"))) if request_root.is_dir() else 0


def _metrics(epoch_root: Path, epoch: dict[str, Any]) -> dict[str, Any]:
    first_valid: bool | None = None
    format_valid: bool | None = None
    reads = probes = holds = feedback = 0
    new_provider_unusable = False
    new_contract_invalid = False
    provider_failure_codes: list[str] = []
    last_step: dict[str, Any] | None = None
    for step in epoch.get("steps", []):
        last_step = step
        action = step.get("action", {})
        reads += int(action.get("action") == "read_evidence")
        probes += int(action.get("action") == "run_probe")
        holds += int(action.get("action") == "hold")
        if first_valid is None:
            first_valid = step.get("model_valid") is True
            format_valid = step.get("format_valid") is True
        dialogue = BASE._read(Path(step["dialogue"]["path"]))
        feedback += int(dialogue.get("semantic_feedback_used") is True)
        if dialogue.get("schema") == LOOP.DIALOGUE_SCHEMA:
            calls = dialogue.get("calls")
            BASE._require(isinstance(calls, list) and bool(calls),
                          "new dialogue calls 불일치")
            checked_calls = cast(list[dict[str, Any]], calls)
            for call in checked_calls:
                result = call.get("result", {})
                if not LOOP._provider_ok(result):
                    new_provider_unusable = True
                    code = result.get("error_type") or result.get("infra_error")
                    if code and result.get("status_code") is not None:
                        code = f"{code}:{result['status_code']}"
                    code = code or result.get("finish_reason") or "provider_unavailable"
                    provider_failure_codes.append(str(code))
            new_contract_invalid |= (dialogue.get("provider_incomplete_hold") is False
                                     and dialogue.get("model_valid") is False)
    model_valid_hold = bool(epoch.get("decision_outcome") == "HOLD" and last_step
                            and last_step.get("action", {}).get("action") == "hold"
                            and last_step.get("model_valid") is True)
    return {"first_decision_valid": first_valid, "first_format_valid": format_valid,
            "read_actions": reads, "probe_actions": probes, "hold_actions": holds,
            "semantic_feedback_actions": feedback,
            "model_valid_hold": model_valid_hold,
            "new_contract_invalid": new_contract_invalid,
            "new_provider_unusable": new_provider_unusable,
            "provider_failure_codes": provider_failure_codes,
            "original_model_requests": epoch["original_model_requests"],
            "new_model_requests": epoch["new_model_requests"],
            "model_requests_including_recorded": epoch["total_model_requests"]}


def _original_prefix_summary(mode_result: Path) -> dict[str, Any]:
    result = BASE._read(mode_result)
    calls = result.get("calls", [])
    summary: dict[str, Any] = {}
    for mode in MODE1.MODES:
        active = [row for row in calls if row.get("mode") == mode
                  and row.get("evaluation_role") == "fault_development"]
        BASE._require(len(active) == 12, f"original {mode} fault-active 분모 불일치")
        summary[mode] = {"active_non_hold": sum(row.get("action") != "hold" for row in active),
                         "contract_valid": sum(row.get("valid") is True for row in active),
                         "denominator": 12}
    BASE._require(summary == {
        "legacy_none": {"active_non_hold": 5, "contract_valid": 12, "denominator": 12},
        "enabled_low": {"active_non_hold": 11, "contract_valid": 12, "denominator": 12}},
                  "등록된 original first-proposal 기준값 불일치")
    return summary


def preflight(protocol: Path, anchor: str, source_batch: Path, mode_result: Path,
              corpus: Path, lab: Path, out: Path) -> dict[str, Any]:
    out = Path(out).resolve()
    BASE._require(not out.exists(), "preflight out은 fresh 경로여야 한다")
    spec, cases, gates = _validate_inputs(protocol, anchor, source_batch, mode_result,
                                          corpus, lab)
    lease = RUN._lease()
    BASE._require(isinstance(lease.get("leaseId"), str) and bool(lease["leaseId"]),
                  "leaseId가 없다")
    result = {"schema": PREFLIGHT_SCHEMA, "status": "ready",
              "protocol": BASE._ref(protocol), "source_batch": BASE._ref(source_batch),
              "mode_result": BASE._ref(mode_result), "corpus": BASE._ref(corpus),
              "opaque_cases": [row["case_id"] for row in cases],
              "fault_rows": sum(gates[row["case_id"]]["run"] for row in cases) * 2,
              "normal_host_gate_rows": sum(not gates[row["case_id"]]["run"] for row in cases) * 2,
              "original_prefix": _original_prefix_summary(mode_result),
              "new_model_requests": 0, "paired_probes": 0,
              "sdk_role_executions": 0, "gpu_started": False,
              "leaseId": lease["leaseId"], "claims": deepcopy(spec["claims"])}
    BASE._write(out / "preflight.json", result)
    return result


def execute(protocol: Path, anchor: str, source_batch: Path, mode_result: Path,
            corpus: Path, lab: Path, out: Path, bridge_root: Path, *,
            load_first: Callable = LOOP.load_recorded_first,
            run_epoch: Callable = LOOP.run_epoch,
            execute_host: Callable = HOST.execute_pair,
            bridge_complete: Callable = BRIDGE.complete,
            bridge_initialize: Callable = BRIDGE.initialize) -> dict[str, Any]:
    out, bridge_root = Path(out).resolve(), Path(bridge_root).resolve()
    BASE._require(not out.exists(), "study out은 fresh 경로여야 한다")
    BASE._require(not bridge_root.exists(), "bridge root는 fresh 경로여야 한다")
    spec, cases, gates = _validate_inputs(protocol, anchor, source_batch, mode_result,
                                          corpus, lab)
    source_sha, mode_sha, corpus_sha = map(BASE._sha, (source_batch, mode_result, corpus))
    bridge_path = ROOT / spec["bridge_protocol"]["path"]
    bridge_initialize(bridge_root, bridge_path, anchor)
    BASE._require(_bridge_request_count(bridge_root) == 0, "bridge root는 API 0의 fresh session이어야 한다")
    references = {row["seed"]: row for row in cases if row["category"] == "NONE"}
    base_protocol = _base_protocol(spec)
    paired_protocol = ROOT / base_protocol["protocols"]["paired_probe"]["path"]
    view_protocol = ROOT / base_protocol["protocols"]["view"]["path"]
    protocol_shas = {path: BASE._sha(path) for path in (paired_protocol, view_protocol)}
    lease_id = RUN._lease().get("leaseId")
    BASE._require(isinstance(lease_id, str) and bool(lease_id), "leaseId가 없다")
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    rows: list[dict[str, Any]] = []
    probe_count = 0
    active: dict[str, Any] = {}

    def verify_context(context: dict[str, Any] | None = None) -> dict[str, Any]:
        elapsed = time.monotonic() - started
        BASE._require(math.isfinite(elapsed) and elapsed < MAX_WALL_S, "study wall 상한 종료")
        BASE._require(BASE._read(protocol) == spec, "실행 중 study protocol 변경")
        BASE._require(prepare_protocol(
            ROOT / spec["parent_diagnostic_protocol"]["path"],
            ROOT / spec["mode_plan"]["path"], bridge_path,
            spec["mode_result_sha256"]) == spec, "실행 중 protocol/source closure 변경")
        BASE._require(BASE._sha(source_batch) == source_sha
                      and BASE._sha(mode_result) == mode_sha
                      and BASE._sha(corpus) == corpus_sha, "실행 중 입력 원문 변경")
        BASE._require(all(BASE._sha(path) == digest for path, digest in protocol_shas.items()),
                      "실행 중 paired/view protocol 변경")
        BASE._require(RUN._lease().get("leaseId") == lease_id, "실행 중 leaseId 변경")
        requests = _bridge_request_count(bridge_root)
        BASE._require(0 <= requests <= MAX_NEW_REQUESTS, "new API cap 불일치")
        BASE._require(probe_count <= MAX_PAIRED_PROBES
                      and probe_count * 2 <= MAX_SDK_ROLE_EXECUTIONS,
                      "probe/SDK role cap 불일치")
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
                "paired_probes": probe_count, "checked_role": checked_role,
                "leaseId": lease_id}

    def complete_factory(slot: str, _state: dict) -> Callable:
        def complete(messages: list, params: dict) -> dict[str, Any]:
            before = verify_context()
            BASE._require(before["requests"] < MAX_NEW_REQUESTS,
                          "새 요청 전 global API cap이 소진됐다")
            result = bridge_complete(bridge_root, slot, messages, params)
            after = verify_context()
            BASE._require(after["requests"] == before["requests"] + 1,
                          "bridge 요청 영수증 수가 1 증가하지 않았다")
            return result
        return complete

    try:
        for case in cases:
            reference = references[case["seed"]]
            gate = gates[case["case_id"]]
            for mode in _mode_order(case["case_id"]):
                base = {"case_id": case["case_id"], "mode": mode,
                        "category": case["category"], "gate": deepcopy(gate),
                        "checkpoint_sha256": case["checkpoint"]["sha256"],
                        "reference_checkpoint_sha256": reference["checkpoint"]["sha256"]}
                if not gate["run"]:
                    rows.append({**base, "status": "complete", "decision_outcome": "HOLD",
                                 "model_actions": 0, "original_model_requests": 0,
                                 "new_model_requests": 0, "paired_probes": 0,
                                 "first_decision_valid": None, "first_format_valid": None,
                                 "read_actions": 0, "probe_actions": 0, "hold_actions": 0,
                                 "semantic_feedback_actions": 0,
                                 "correct_confirmed_family": False,
                                 "correct_complete_family": False,
                                 "compound_partial": False, "compound_complete": False,
                                 "wrong_family_confirmation": False,
                                 "model_valid_hold": False,
                                 "new_contract_invalid": False,
                                 "new_provider_unusable": False,
                                 "provider_failure_codes": [],
                                 "normal_gate_model_success": False})
                    continue
                first = load_first(mode_result, ROOT / spec["mode_plan"]["path"], corpus,
                                   MODE_ANCHOR, MODE_RESULT_SHA256, case["case_id"], mode,
                                   replicate=1)
                _verify_first_source(first, case, reference)
                case_runtime = Path(case["metadata"]["runtime_facts"]["case"]["path"]).resolve()
                reference_runtime = Path(
                    reference["metadata"]["runtime_facts"]["reference"]["path"]).resolve()
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
                    BASE._require(probe_count < MAX_PAIRED_PROBES
                                  and (probe_count + 1) * 2 <= MAX_SDK_ROLE_EXECUTIONS,
                                  "paired probe/SDK role global cap 종료")
                    result = execute_host(
                        state, action, case_input=_case["input"],
                        reference_input=_reference["input"], case_runtime=_case_runtime,
                        reference_runtime=_reference_runtime,
                        probe_protocol=paired_protocol, view_protocol=view_protocol, anchor=anchor,
                        lab=Path(lab).resolve(), out=stage, verify_context=verify_context)
                    probe_count += 1
                    return result

                epoch_root = out / case["case_id"] / mode / "epoch"
                epoch = run_epoch(first, run_id=f"{case['case_id']}.{mode}", mode=mode,
                                  out=epoch_root, verify_context=verify_context,
                                  complete_factory=complete_factory, execute_pair=pair)
                score = BASE._score(case["category"], epoch)
                wrong = (epoch.get("decision_outcome") == "confirmed"
                         and not score["correct_confirmed_family"])
                rows.append({**base, "status": epoch["status"],
                             "decision_outcome": epoch["decision_outcome"],
                             "model_actions": epoch["model_actions"],
                             "paired_probes": epoch["paired_probes"],
                             "parent_probe_wall_s": epoch["parent_probe_wall_s"],
                             "owned_child_wall_s": epoch["owned_child_wall_s"],
                             "final_status": epoch["final_status"],
                             **_metrics(epoch_root, epoch), **score,
                             "wrong_family_confirmation": wrong,
                             "normal_gate_model_success": False})
        verify_context()
        BASE._require(len(rows) == 16, "계획된 8×2 행 분모가 보존되지 않았다")
        fault_rows = [row for row in rows if row["gate"]["run"]]
        normal_rows = [row for row in rows if not row["gate"]["run"]]
        BASE._require(len(fault_rows) == 12 and len(normal_rows) == 4,
                      "fault12/normal4 분모 불일치")
        BASE._require(all(row["original_model_requests"] == 0
                          and row["new_model_requests"] == 0
                          and row["paired_probes"] == 0 for row in normal_rows),
                      "normal gate 행에서 모델/probe가 실행됐다")
        requests = sum(row["new_model_requests"] for row in fault_rows)
        probes = sum(row["paired_probes"] for row in fault_rows)
        BASE._require(requests == _bridge_request_count(bridge_root)
                      and requests <= MAX_NEW_REQUESTS and probes == probe_count
                      and probes <= MAX_PAIRED_PROBES, "집계/실제 전역 cap 불일치")
        by_mode = {mode: [row for row in fault_rows if row["mode"] == mode]
                   for mode in MODE1.MODES}
        BASE._require(all(len(value) == 6 for value in by_mode.values()),
                      "mode별 fault 분모 불일치")
        mode_metrics = {mode: {
            "fault_rows": 6,
            "first_valid": sum(row["first_decision_valid"] is True for row in values),
            "read_actions": sum(row["read_actions"] for row in values),
            "probe_actions": sum(row["probe_actions"] for row in values),
            "holds": sum(row["decision_outcome"] == "HOLD" for row in values),
            "correct_confirmed_family": sum(row["correct_confirmed_family"] for row in values),
            "compound_partial": sum(row["compound_partial"] for row in values),
            "wrong_family_confirmations": sum(row["wrong_family_confirmation"] for row in values),
            "model_valid_holds": sum(row["model_valid_hold"] for row in values),
            "new_contract_invalid_rows": sum(row["new_contract_invalid"] for row in values),
            "new_provider_unusable_rows": sum(row["new_provider_unusable"] for row in values),
            "new_model_requests": sum(row["new_model_requests"] for row in values),
            "paired_probes": sum(row["paired_probes"] for row in values),
        } for mode, values in by_mode.items()}
        low = mode_metrics["enabled_low"]
        legacy = mode_metrics["legacy_none"]
        effect_assessable = not any(row["new_provider_unusable"] for row in fault_rows)
        candidate = (None if not effect_assessable else
                     (low["correct_confirmed_family"] > legacy["correct_confirmed_family"]
                      and low["wrong_family_confirmations"] == 0
                      and all(row["new_model_requests"] == 0 and row["paired_probes"] == 0
                              for row in normal_rows)))
        result = {"schema": RESULT_SCHEMA, "status": "complete",
                  "protocol": BASE._ref(protocol), "source_batch": BASE._ref(source_batch),
                  "mode_result": BASE._ref(mode_result), "corpus": BASE._ref(corpus),
                  "anchor": anchor, "rows": rows, "planned_rows": 16,
                  "fault_rows": 12, "normal_host_gate_rows": 4,
                  "recorded_first_responses": 12, "original_mode_calls": 32,
                  "new_model_requests": requests, "paired_probes": probes,
                  "sdk_role_executions": probes * 2,
                  "original_prefix": _original_prefix_summary(mode_result),
                  "mode_metrics": mode_metrics,
                  "development_candidate": candidate,
                  "effect_assessment": ("complete" if effect_assessable
                                        else "indeterminate_infrastructure"),
                  "development_candidate_is_default_adoption": False,
                  "cpu_parent_wall_s": time.monotonic() - started,
                  "claims": deepcopy(spec["claims"]), "automatic_retry": False,
                  "automatic_repair": False, "training": False,
                  "correction": False, "recovery": False}
        BASE._write(out / "result.json", result)
        return result
    except BaseException as error:
        BASE._write(out / "failure.json", {
            "schema": RESULT_SCHEMA, "status": "invalid",
            "error_type": type(error).__name__, "error": str(error), "rows": rows,
            "new_model_requests": _bridge_request_count(bridge_root),
            "paired_probes": probe_count, "automatic_retry": False,
            "automatic_repair": False, "recovery": False})
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    bridge = sub.add_parser("prepare-bridge")
    bridge.add_argument("--parent-mode-plan", type=Path, required=True)
    bridge.add_argument("--out", type=Path, required=True)
    prepare = sub.add_parser("prepare-protocol")
    prepare.add_argument("--parent-diag-protocol", type=Path, required=True)
    prepare.add_argument("--mode-plan", type=Path, required=True)
    prepare.add_argument("--bridge-protocol", type=Path, required=True)
    prepare.add_argument("--mode-result-sha256", required=True)
    prepare.add_argument("--out", type=Path, required=True)
    for command in ("preflight", "execute"):
        item = sub.add_parser(command)
        item.add_argument("--protocol", type=Path, required=True)
        item.add_argument("--anchor", required=True)
        item.add_argument("--source-batch", type=Path, required=True)
        item.add_argument("--mode-result", type=Path, required=True)
        item.add_argument("--corpus", type=Path, required=True)
        item.add_argument("--lab", type=Path, required=True)
        item.add_argument("--out", type=Path, required=True)
        if command == "execute":
            item.add_argument("--bridge-root", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare-bridge":
        BASE._write(args.out, prepare_bridge(args.parent_mode_plan))
    elif args.command == "prepare-protocol":
        BASE._write(args.out, prepare_protocol(args.parent_diag_protocol, args.mode_plan,
                                               args.bridge_protocol,
                                               args.mode_result_sha256))
    elif args.command == "preflight":
        preflight(args.protocol, args.anchor, args.source_batch, args.mode_result,
                  args.corpus, args.lab, args.out)
    else:
        execute(args.protocol, args.anchor, args.source_batch, args.mode_result,
                args.corpus, args.lab, args.out, args.bridge_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["PREFLIGHT_SCHEMA", "RESULT_SCHEMA", "SCHEMA", "execute", "main",
           "preflight", "prepare_bridge", "prepare_protocol", "verify_protocol"]
