"""Reasoning 진단 후속 연구를 원문 재생 검증한 뒤 trace 없이 공개한다."""
# pyright: reportMissingImports=false
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

BOOTSTRAP_ROOT = Path(__file__).resolve().parents[1]
if str(BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(BOOTSTRAP_ROOT))

from evals import export_nemotron_reasoning_options_v1 as MODE_EXPORT
from evals import export_prospective_agentic_comparison_v1 as PRIV
from evals import reasoning_diagnostic_loop_v1 as LOOP
from evals import registry_action_loop_v1 as BASE_LOOP
from evals import registry_action_options_v2 as OPTIONS
from evals import run_reasoning_diagnosis_study_v1 as STUDY

ROOT = PRIV.ROOT
SUMMARY_SCHEMA = "reasoning_diagnosis_study_public_summary_v1"
CASE_SCHEMA = "reasoning_diagnosis_study_public_case_v1"
PROVENANCE_SCHEMA = "reasoning_diagnosis_study_public_provenance_v1"
MANIFEST_SCHEMA = "reasoning_diagnosis_study_public_manifest_v1"
MODES = ("legacy_none", "enabled_low")


def _require(ok: bool, reason: str) -> None:
    PRIV._require(ok, reason)


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    return PRIV._read(Path(path))


def _finite(value: object, label: str) -> float:
    _require(type(value) in (int, float) and math.isfinite(cast(float, value))
             and cast(float, value) >= 0, f"{label}은 유한 비음수여야 한다")
    return float(cast(float, value))


def _checked_local_ref(value: object, root: Path, label: str) -> Path:
    _require(isinstance(value, dict) and set(value) == {"path", "sha256"},
             f"{label} ref 형식 불일치")
    ref = cast(dict[str, Any], value)
    path = Path(cast(str, ref["path"])).resolve()
    root = Path(root).resolve()
    _require(path.is_file() and path.is_relative_to(root)
             and _sha(path) == ref["sha256"], f"{label} 경로/SHA 불일치")
    return path


def _rows(value: dict[str, Any]) -> list[dict[str, Any]]:
    rows = value.get("rows")
    _require(value.get("schema") == STUDY.RESULT_SCHEMA and value.get("status") == "complete"
             and isinstance(rows, list) and len(rows) == 16,
             "reasoning diagnosis는 complete 16행이어야 공개할 수 있다")
    typed = cast(list[dict[str, Any]], rows)
    expected = [(f"c{i:02d}", mode) for i in range(1, 9)
                for mode in STUDY._mode_order(f"c{i:02d}")]
    _require([(row.get("case_id"), row.get("mode")) for row in typed] == expected,
             "result row 순서·중복·mode 불일치")
    _require(value.get("planned_rows") == 16 and value.get("fault_rows") == 12
             and value.get("normal_host_gate_rows") == 4
             and value.get("recorded_first_responses") == 12
             and value.get("original_mode_calls") == 32,
             "16/12/4 및 원 응답 분모 불일치")
    return typed


def _public_response(value: dict[str, Any]) -> dict[str, Any]:
    allowed = ("provider_model", "finish_reason", "content", "wall_s", "input_tokens",
               "output_tokens", "total_tokens", "reasoning_tokens",
               "reasoning_tokens_status", "status_code", "error_type", "infra_error")
    public = {key: deepcopy(value[key]) for key in allowed if key in value}
    _require("reasoning" not in public and "reasoning_content" not in public,
             "공개 response에 reasoning body가 남았다")
    if "content" in public and isinstance(public["content"], str):
        try:
            parsed = json.loads(public["content"])
        except json.JSONDecodeError:
            parsed = None
        public["final_json"] = parsed
        public.pop("content", None)
    return public


def _verify_new_dialogue(dialogue: dict[str, Any], before: dict[str, Any], *,
                         mode: str) -> dict[str, Any]:
    _require(dialogue.get("schema") == LOOP.DIALOGUE_SCHEMA
             and dialogue.get("state_digest") == before.get("state_digest")
             and dialogue.get("context_digest") == OPTIONS.O1._digest(OPTIONS.context(before))
             and dialogue.get("mode") == mode, "후속 dialogue state/context/mode 불일치")
    calls = dialogue.get("calls")
    _require(isinstance(calls, list) and 1 <= len(calls) <= 2, "후속 dialogue call 수 불일치")
    sent = OPTIONS.messages(before)
    checked_public: list[dict[str, Any]] = []
    provider_ok = False
    for index, call in enumerate(cast(list[dict[str, Any]], calls), 1):
        _require(call.get("attempt") == index and call.get("messages") == sent
                 and call.get("messages_sha256") == OPTIONS.O1._digest(sent),
                 "후속 call attempt/messages 결합 불일치")
        params = LOOP._mode_params(sent, mode)
        result = call.get("result")
        _require(call.get("request_params") == params and isinstance(result, dict),
                 "후속 call params/result 형식 불일치")
        result = cast(dict[str, Any], result)
        LOOP._transport_ref(result.get("transport_ref"), slot=cast(str, call.get("slot")),
                            mode=mode, messages=sent, params=params,
                            returned_result=result)
        response_path = Path(cast(dict[str, Any], result["transport_ref"])["path"]).resolve()
        request_path = response_path.parent.parent / "requests" / response_path.name
        provider_ok = LOOP._provider_ok(result)
        review = (OPTIONS.assess(result.get("content"), before) if provider_ok else {
            "status": "invalid", "format_status": "not_run", "semantic_status": "not_run",
            "codes": ["length" if result.get("finish_reason") == "length"
                      else "PROVIDER_UNAVAILABLE"], "response_text": result.get("content")})
        _require(call.get("review") == review, "후속 call review 재생 불일치")
        checked_public.append({"attempt": index, "messages": deepcopy(sent),
                               "request_params": deepcopy(params),
                               "response": _public_response(result),
                               "review": deepcopy(review),
                               "transport_proof": {"request_sha256": _sha(request_path),
                                                   "response_sha256": _sha(response_path)}})
        if index == 1 and len(cast(list[Any], calls)) == 2:
            current = OPTIONS.context(before)
            sent = [*sent, {"role": "assistant", "content": result.get("content") or ""},
                    {"role": "user", "content": json.dumps({
                        "validation_errors": review["codes"],
                        "output_schema": current["output_schema"],
                        "action_options": current["action_options"],
                        "count_evidence_groups": current["count_evidence_groups"],
                        "required": "Use one current option_id and exact visible fact_ids only."},
                        ensure_ascii=False, sort_keys=True)}]
    final = cast(list[dict[str, Any]], calls)[-1]["review"]
    action = (OPTIONS.select_model(final, before) if provider_ok else
              {"action": "hold", "target": None, "fact_ids": []})
    _require(dialogue.get("final_review") == final and dialogue.get("action") == action
             and dialogue.get("model_valid") == (final["status"] == "valid")
             and dialogue.get("semantic_feedback_used") == (len(cast(list[Any], calls)) == 2)
             and dialogue.get("provider_incomplete_hold") == (not provider_ok),
             "후속 dialogue 최종 action/provider 상태 불일치")
    return {"schema": "reasoning_diagnosis_public_dialogue_v1", "mode": mode,
            "calls": checked_public, "action": deepcopy(action),
            "model_valid": dialogue["model_valid"],
            "semantic_feedback_used": dialogue["semantic_feedback_used"]}


def _advance(stage: Path, before: dict[str, Any], action: dict[str, Any],
             host: dict[str, Any]) -> dict[str, Any]:
    if action.get("action") == "run_probe":
        proof_path = stage / "paired-proof.json"
        proof = _read(proof_path)
        names = ("case_plan", "case_receipt", "reference_plan", "reference_receipt")
        paths = {name: _checked_local_ref(proof.get(name), stage / "probe", name)
                 for name in names}
        recomputed = BASE_LOOP._pair_proof(paths, before, action, stage / "probe")
        _require(proof == recomputed and host.get("paired_proof") == BASE_LOOP.ref(proof_path)
                 and host.get("root_approval") == proof["root_approval"],
                 "paired proof/root approval 재검증 불일치")
        return STUDY.BASE.SEL.advance(
            before, action, case_plan=paths["case_plan"], case_receipt=paths["case_receipt"],
            reference_plan=paths["reference_plan"],
            reference_receipt=paths["reference_receipt"])
    _require(host.get("request_spec") is None and host.get("root_approval") is None
             and host.get("paired_proof") is None, "비검사 host action에 probe proof가 있다")
    return STUDY.BASE.SEL.advance(before, action)


def _probe_execution_audit(stage: Path) -> dict[str, Any]:
    proof = _read(stage / "paired-proof.json")
    receipts = [_read(_checked_local_ref(proof[f"{role}_receipt"], stage / "probe",
                                         f"{role} receipt"))
                for role in ("case", "reference")]
    host_result = _read(stage / "probe" / "host-result.json")
    schemas = {row.get("schema") for row in receipts}
    if schemas == {"recorded_optimizer_probe_receipt_v1"}:
        _require(all(row.get("evidence_kind") == "actual_training_reuse"
                     and row.get("scope") == {
                         "reused_verified_training_records": True, "new_training": False,
                         "gpu_execution": False, "new_observation": False,
                         "recovery_claim": False} for row in receipts),
                 "recorded probe actual evidence/scope 불일치")
        _require(host_result.get("scope") == "recorded_verified_training_cpu_reuse"
                 and host_result.get("owned_child_wall_s") == 0,
                 "recorded probe host CPU reuse scope 불일치")
        return {"probe": "P_recorded_optimizer_updates",
                "evidence_kind": "actual_training_reuse",
                "actual_sdk_role_executions": 0, "planned_max_sdk_role_executions": 2,
                "reused_training_observer_records": True, "new_training": False}
    _require(schemas == {"runtime_observation_probe_receipt_v1"},
             "paired receipt schema가 등록된 recorded/fresh 검사와 다르다")
    _require(all(row.get("status") in {"complete", "unknown"}
                 and isinstance(row.get("execution"), dict) for row in receipts),
             "fresh runtime receipt owned execution 근거 불일치")
    probes = {row.get("probe") for row in receipts}
    _require(len(probes) == 1, "fresh paired receipt probe 불일치")
    _require(host_result.get("scope") == "legacy_registered_paired_probe"
             and _finite(host_result.get("owned_child_wall_s"), "fresh owned child wall") > 0,
             "fresh paired host execution scope 불일치")
    return {"probe": next(iter(probes)), "evidence_kind": "fresh_owned_execution",
            "actual_sdk_role_executions": 2, "planned_max_sdk_role_executions": 2,
            "reused_training_observer_records": False, "new_training": False}


def _verify_epoch(root: Path, row: dict[str, Any], case: dict[str, Any],
                  reference: dict[str, Any], recorded: dict[tuple[str, str], dict[str, Any]]
                  ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    row_root = Path(root) / row["case_id"] / row["mode"]
    if not row.get("gate", {}).get("run"):
        _require(not row_root.exists() and row.get("decision_outcome") == "HOLD"
                 and row.get("model_actions") == 0 and row.get("new_model_requests") == 0
                 and row.get("paired_probes") == 0
                 and row.get("normal_gate_model_success") is False,
                 "normal host gate가 모델 실행/성공으로 기록됐다")
        return [], {"first_action": None, "provider_invalid": False,
                    "contract_invalid": False, "observed_actions": [],
                    "probe_audits": [], "actual_sdk_role_executions": 0}
    epoch_root = row_root / "epoch"
    epoch = _read(epoch_root / "result.json")
    _require(epoch.get("schema") == LOOP.SCHEMA and epoch.get("status") == "complete"
             and epoch.get("run_id") == f"{row['case_id']}.{row['mode']}",
             "reasoning loop result schema/status/run 불일치")
    for field in ("decision_outcome", "model_actions", "paired_probes",
                  "parent_probe_wall_s", "owned_child_wall_s", "final_status"):
        _require(row.get(field) == epoch.get(field), f"row/epoch {field} 불일치")
    for key, value in STUDY.BASE._score(case["category"], epoch).items():
        _require(row.get(key) == value, f"row score {key} 재계산 불일치")
    _finite(epoch.get("parent_probe_wall_s"), "parent_probe_wall_s")
    _finite(epoch.get("owned_child_wall_s"), "owned_child_wall_s")
    expected_recorded = recorded[(row["case_id"], row["mode"])]
    recorded_row = epoch.get("recorded_first")
    _require(isinstance(recorded_row, dict)
             and recorded_row.get("response_is_original") is True
             and recorded_row.get("call_index") == expected_recorded["refproof"]["call_index"]
             and recorded_row.get("private") == expected_recorded["refproof"]["private"]
             and recorded_row.get("request_sha256") ==
                 expected_recorded["refproof"]["request_sha256"],
             "epoch recorded-first response/ref 불일치")
    recorded_row = cast(dict[str, Any], recorded_row)
    recorded_wall = _finite(recorded_row.get("wall_s"), "recorded first API wall_s")
    post_first_epoch_wall = _finite(epoch.get("epoch_wall_s"),
                                    "post-recorded-first epoch wall_s")
    steps = epoch.get("steps")
    _require(isinstance(steps, list) and len(steps) == epoch.get("model_actions")
             and bool(steps), "reasoning loop step/model_actions 불일치")
    public_dialogues: list[dict[str, Any]] = []
    previous: str | None = None
    new_requests = 0
    probe_count = 0
    provider_invalid = contract_invalid = False
    actions: list[str] = []
    probe_audits: list[dict[str, Any]] = []
    for index, step in enumerate(cast(list[dict[str, Any]], steps), 1):
        stage = epoch_root / f"step-{index:02d}"
        before = _read(stage / "state-before.json")
        OPTIONS.context(before)
        _require(before.get("state_digest") == step.get("state_before_digest")
                 and (previous is None or previous == before.get("state_digest")),
                 "reasoning state-before chain 불일치")
        dialogue_path = _checked_local_ref(step.get("dialogue"), epoch_root, "dialogue")
        dialogue = _read(dialogue_path)
        if index == 1:
            expected = expected_recorded
            _require(dialogue == expected["dialogue"], "recorded first dialogue 원문 불일치")
            proof_path = _checked_local_ref(step.get("reference_proof"), epoch_root,
                                            "recorded reference proof")
            _require(_read(proof_path) == expected["refproof"],
                     "recorded first intent/private/public/request proof 불일치")
            STUDY._verify_first_source({"state": before}, case, reference)
            public_dialogue = expected["public_dialogue"]
            action = expected["action"]
        else:
            _require(_read(stage / "option-context.json") == OPTIONS.context(before),
                     "후속 option context/state 불일치")
            public_dialogue = _verify_new_dialogue(dialogue, before, mode=row["mode"])
            action = public_dialogue["action"]
            new_requests += len(public_dialogue["calls"])
            provider_invalid |= dialogue.get("provider_incomplete_hold") is True
            contract_invalid |= (dialogue.get("provider_incomplete_hold") is False
                                 and dialogue.get("model_valid") is False)
        _require(action == step.get("action"), "dialogue/step action 불일치")
        selected = _read(stage / "selected-action.json")
        _require(selected.get("action") == action
                 and selected.get("model_valid") == step.get("model_valid"),
                 "selected action/model_valid 불일치")
        host = _read(_checked_local_ref(step.get("host_action"), epoch_root, "host action"))
        _require(host.get("action") == action, "host/action 불일치")
        replayed = _advance(stage, before, action, host)
        if action.get("action") == "run_probe":
            probe_audits.append(_probe_execution_audit(stage))
        after = _read(_checked_local_ref(step.get("state_after"), epoch_root, "state after"))
        _require(after == replayed and after.get("state_digest") == step.get("state_after_digest"),
                 "receipt 기반 state-after 재생 불일치")
        previous = cast(str, after["state_digest"])
        probe_count += int(action.get("action") == "run_probe")
        actions.append(cast(str, action.get("action")))
        public_dialogues.append(public_dialogue)
    final = _read(epoch_root / "final-state.json")
    OPTIONS.context(final)
    _require(final.get("state_digest") == previous
             and final.get("status") == epoch.get("final_status"), "final state chain 불일치")
    raw_metrics = STUDY._metrics(epoch_root, epoch)
    for key, value in raw_metrics.items():
        _require(row.get(key) == value, f"row metric {key} 재계산 불일치")
    _require(epoch.get("new_model_requests") == new_requests
             and epoch.get("paired_probes") == probe_count,
             "epoch request/probe 원문 집계 불일치")
    return public_dialogues, {"first_action": actions[0], "provider_invalid": provider_invalid,
                              "contract_invalid": contract_invalid,
                              "observed_actions": actions, "probe_audits": probe_audits,
                              "recorded_first_api_wall_s": recorded_wall,
                              "post_recorded_first_epoch_wall_s": post_first_epoch_wall,
                              "actual_sdk_role_executions": sum(
                                  item["actual_sdk_role_executions"] for item in probe_audits)}


def _recorded_prefix(mode_result: Path, mode_plan: Path, corpus: Path, *,
                     mode_anchor: str = STUDY.MODE_ANCHOR
                     ) -> tuple[list[dict[str, Any]], dict[tuple[str, str], dict[str, Any]],
                                list[dict[str, Any]]]:
    plan = MODE_EXPORT.V2.verify_plan(mode_plan, corpus)
    corpus_value, cases = MODE_EXPORT.V1._corpus_ref(corpus, load_payloads=True)
    rows = MODE_EXPORT._result_rows(_read(mode_result), plan)
    by_case = {row["case_id"]: row for row in cases}
    public_calls: list[dict[str, Any]] = []
    proof_rows: list[dict[str, Any]] = []
    selected: dict[tuple[str, str], dict[str, Any]] = {}
    for planned, row in zip(plan["calls"], rows, strict=True):
        case = by_case[planned["case_id"]]
        public, proofs = MODE_EXPORT._validate_call(
            mode_result.parent / "calls" / f"{planned['call_index']:02d}", planned, row, case)
        public_calls.append(public)
        proof_rows.append({"call_index": planned["call_index"], **proofs})
        if planned["replicate"] == 1 and planned["evaluation_role"] == "fault_development":
            private_path = mode_result.parent / "calls" / f"{planned['call_index']:02d}" / "private.json"
            private = _read(private_path)
            response = cast(dict[str, Any], private["response"])
            review = OPTIONS.assess(response["content"], case["state"])
            action = OPTIONS.select_model(review, case["state"])
            dialogue = {"schema": "reasoning_diagnostic_recorded_dialogue_v1",
                        "recorded_original": True,
                        "messages": deepcopy(case["messages"]),
                        "messages_digest": case["messages_digest"],
                        "request_params": LOOP._mode_params(case["messages"], planned["mode"]),
                        "raw_response": deepcopy(response), "review": deepcopy(review),
                        "action": deepcopy(action), "model_valid": review["status"] == "valid",
                        "semantic_feedback_used": False}
            selected[(planned["case_id"], planned["mode"])] = {
                "dialogue": dialogue, "action": action,
                "public_dialogue": {"schema": "reasoning_diagnosis_public_dialogue_v1",
                    "mode": planned["mode"], "recorded_original": True,
                    "messages": deepcopy(case["messages"]),
                    "response": _public_response(response), "review": deepcopy(review),
                    "action": deepcopy(action), "model_valid": review["status"] == "valid",
                    "raw_proof": {"result_sha256": _sha(mode_result),
                                  "request_sha256": MODE_EXPORT.V1.digest(
                                      MODE_EXPORT.V1.build_request(
                                          case["messages"], planned["mode"])),
                                  "intent_sha256": proofs["intent_sha256"],
                                  "private_sha256": proofs["private_sha256"],
                                  "public_sha256": proofs["public_sha256"]}},
                "refproof": {"anchor": mode_anchor,
                    "result": LOOP._json_ref(mode_result), "plan": LOOP._json_ref(mode_plan),
                    "corpus": LOOP._json_ref(corpus),
                    "intent": LOOP._json_ref(mode_result.parent / "calls" /
                                              f"{planned['call_index']:02d}" / "intent.json"),
                    "private": LOOP._json_ref(private_path),
                    "public": LOOP._json_ref(mode_result.parent / "calls" /
                                              f"{planned['call_index']:02d}" / "public.json"),
                    "fact_ledger": LOOP._verify_ledger(case["state"]),
                    "call_index": planned["call_index"],
                    "request_sha256": MODE_EXPORT.V1.digest(
                        MODE_EXPORT.V1.build_request(case["messages"], planned["mode"]))}}
    _require(len(selected) == 12, "recorded replicate-1 fault 응답 12개 불일치")
    _require(corpus_value.get("schema") == MODE_EXPORT.V1.CORPUS_SCHEMA,
             "reasoning corpus schema 불일치")
    return cases, selected, proof_rows


def export(study_result: Path, protocol_path: Path, source_batch: Path, mode_result: Path,
           corpus: Path, lab: Path, *, anchor: str, out: Path) -> dict[str, Any]:
    study_result, protocol_path, source_batch, mode_result, corpus, lab, out = map(
        Path.resolve, map(Path, (study_result, protocol_path, source_batch, mode_result,
                                corpus, lab, out)))
    _require(not out.exists(), "public reasoning diagnosis out은 fresh 경로여야 한다")
    with STUDY.BASE.COMP.CACHE.proof_scope() as cache_stats:
        spec, source_cases, gates = STUDY._validate_inputs(
            protocol_path, anchor, source_batch, mode_result, corpus, lab)
        value = _read(study_result)
        rows = _rows(value)
        _require(value.get("protocol") == STUDY.BASE._ref(protocol_path)
                 and value.get("source_batch") == STUDY.BASE._ref(source_batch)
                 and value.get("mode_result") == STUDY.BASE._ref(mode_result)
                 and value.get("corpus") == STUDY.BASE._ref(corpus)
                 and value.get("anchor") == anchor,
                 "result protocol/source/mode/corpus/anchor ref 불일치")
        mode_plan = ROOT / spec["mode_plan"]["path"]
        corpus_cases, recorded, raw_call_proofs = _recorded_prefix(
            mode_result, mode_plan, corpus, mode_anchor=spec["mode_plan_anchor"])
        _require([row["case_id"] for row in corpus_cases] == [f"c{i:02d}" for i in range(1, 9)],
                 "corpus 8-case 순서 불일치")
        references = {row["seed"]: row for row in source_cases if row["category"] == "NONE"}
        by_source = {row["case_id"]: row for row in source_cases}
        public_rows: list[dict[str, Any]] = []
        public_dialogues: list[tuple[str, str, int, dict[str, Any]]] = []
        for row in rows:
            case = by_source[row["case_id"]]
            _require(row.get("gate") == gates[row["case_id"]]
                     and row.get("category") == case["category"], "row source gate/category 불일치")
            dialogues, derived = _verify_epoch(study_result.parent, row, case,
                                                references[case["seed"]], recorded)
            for index, dialogue in enumerate(dialogues, 1):
                public_dialogues.append((row["case_id"], row["mode"], index, dialogue))
            public_rows.append({"schema": CASE_SCHEMA, "case_id": row["case_id"],
                "mode": row["mode"], "evaluation_role": ("fault_development"
                    if row["gate"]["run"] else "normal_host_gate"),
                "status": row["status"], "decision_outcome": row["decision_outcome"],
                "model_actions": row["model_actions"],
                "original_model_requests": row["original_model_requests"],
                "new_model_requests": row["new_model_requests"],
                "paired_probes": row["paired_probes"],
                "first_decision_valid": row["first_decision_valid"],
                "first_format_valid": row["first_format_valid"],
                "read_actions": row["read_actions"], "probe_actions": row["probe_actions"],
                "hold_actions": row["hold_actions"],
                "correct_confirmed_family": row["correct_confirmed_family"],
                "correct_complete_family": row["correct_complete_family"],
                "compound_partial": row["compound_partial"],
                "compound_complete": row["compound_complete"],
                "wrong_family_confirmation": row["wrong_family_confirmation"],
                "model_valid_hold": row["model_valid_hold"],
                "new_contract_invalid": row["new_contract_invalid"],
                "new_provider_unusable": row["new_provider_unusable"],
                "provider_failure_codes": deepcopy(row["provider_failure_codes"]),
                "actions": derived["observed_actions"],
                "probe_execution_audit": derived["probe_audits"],
                "actual_sdk_role_executions": derived["actual_sdk_role_executions"],
                "planned_max_sdk_role_executions": row["paired_probes"] * 2,
                "recorded_first_api_wall_s": derived.get("recorded_first_api_wall_s"),
                "post_recorded_first_epoch_wall_s":
                    derived.get("post_recorded_first_epoch_wall_s"),
                "continuous_end_to_end_latency": False,
                "raw_epoch_result_sha256": (_sha(study_result.parent / row["case_id"] /
                                                 row["mode"] / "epoch" / "result.json")
                                             if row["gate"]["run"] else None),
                "api_call_wall_observations_only": True,
                "parent_probe_wall_s": row.get("parent_probe_wall_s"),
                "owned_child_wall_s": row.get("owned_child_wall_s")})
        mode_metrics = {mode: {
            "fault_rows": 6,
            "first_valid": sum(row["first_decision_valid"] is True for row in public_rows
                               if row["mode"] == mode
                               and row["evaluation_role"] == "fault_development"),
            "read_actions": sum(row["read_actions"] for row in public_rows
                                if row["mode"] == mode
                                and row["evaluation_role"] == "fault_development"),
            "probe_actions": sum(row["probe_actions"] for row in public_rows
                                 if row["mode"] == mode
                                 and row["evaluation_role"] == "fault_development"),
            "holds": sum(row["decision_outcome"] == "HOLD" for row in public_rows
                         if row["mode"] == mode
                         and row["evaluation_role"] == "fault_development"),
            "correct_confirmed_family": sum(row["correct_confirmed_family"] is True
                                            for row in public_rows if row["mode"] == mode
                                            and row["evaluation_role"] == "fault_development"),
            "compound_partial": sum(row["compound_partial"] is True for row in public_rows
                                    if row["mode"] == mode
                                    and row["evaluation_role"] == "fault_development"),
            "wrong_family_confirmations": sum(row["wrong_family_confirmation"] is True
                                               for row in public_rows if row["mode"] == mode
                                               and row["evaluation_role"] == "fault_development"),
            "model_valid_holds": sum(row["model_valid_hold"] is True for row in public_rows
                                     if row["mode"] == mode
                                     and row["evaluation_role"] == "fault_development"),
            "new_contract_invalid_rows": sum(row["new_contract_invalid"] is True
                                             for row in public_rows if row["mode"] == mode
                                             and row["evaluation_role"] == "fault_development"),
            "new_provider_unusable_rows": sum(row["new_provider_unusable"] is True
                                              for row in public_rows if row["mode"] == mode
                                              and row["evaluation_role"] == "fault_development"),
            "new_model_requests": sum(row["new_model_requests"] for row in public_rows
                                      if row["mode"] == mode
                                      and row["evaluation_role"] == "fault_development"),
            "paired_probes": sum(row["paired_probes"] for row in public_rows
                                 if row["mode"] == mode
                                 and row["evaluation_role"] == "fault_development")}
            for mode in MODES}
        _require(mode_metrics == value.get("mode_metrics"), "공개 mode metrics/raw result 불일치")
        effect_assessable = not any(row["new_provider_unusable"] for row in public_rows
                                    if row["evaluation_role"] == "fault_development")
        _require(value.get("effect_assessment") == ("complete" if effect_assessable
                 else "indeterminate_infrastructure"), "effect-assessable 기준 재계산 불일치")
    out.mkdir(parents=True, exist_ok=False)
    files: dict[str, dict[str, str]] = {}

    def write(relative: str, payload: object, mode: str) -> None:
        PRIV._safe_public(payload)
        path = out / relative
        STUDY.BASE._write(path, payload)
        files[relative] = {"sha256": _sha(path), "mode": mode}

    for row in public_rows:
        write(f"cases/{row['case_id']}-{row['mode']}.json", row, "verified_case_projection")
    for case_id, mode, index, dialogue in public_dialogues:
        write(f"dialogues/{case_id}/{mode}/step-{index:02d}.json", dialogue,
              "reasoning_trace_removed_dialogue")
    for case in corpus_cases:
        write(f"messages/{case['case_id']}.json", case["messages"], "verified_model_messages")
    summary = {"schema": SUMMARY_SCHEMA, "status": "complete", "planned_rows": 16,
        "fault_rows": 12, "normal_host_gate_rows": 4,
        "recorded_first_responses": 12, "original_mode_calls": 32,
        "new_model_requests": value["new_model_requests"],
        "paired_probes": value["paired_probes"], "mode_metrics": deepcopy(mode_metrics),
        "planned_max_sdk_role_executions": value["sdk_role_executions"],
        "actual_sdk_role_executions": sum(row["actual_sdk_role_executions"]
                                          for row in public_rows),
        "recorded_probe_sdk_role_executions": 0,
        "development_candidate": value["development_candidate"],
        "effect_assessment": value["effect_assessment"],
        "effect_assessable": effect_assessable,
        "claims": {"post_exposure_development": True, "first_proposal_separate": True,
                   "confirmed_diagnosis_separate": True, "correction": False,
                   "recovery": False, "default_adoption": False,
                   "model_version_advantage": False, "speed_advantage": False,
                   "sdk_efficiency_advantage": False, "new_training": False,
                   "parent_gpu_time_inferred_from_api_wall": False}}
    write("summary.json", summary, "derived_summary")
    provenance = {"schema": PROVENANCE_SCHEMA, "creator_sha256": _sha(Path(__file__)),
        "source_anchor": anchor, "result_sha256": _sha(study_result),
        "protocol": {"path": protocol_path.relative_to(ROOT).as_posix(),
                     "sha256": _sha(protocol_path)},
        "mode_result_sha256": _sha(mode_result), "corpus_sha256": _sha(corpus),
        "raw_original_call_proofs": raw_call_proofs,
        "cache_scope": deepcopy(cache_stats),
        "privacy": {"private_paths": False, "reasoning_trace_body": False,
                    "final_json_and_usage_only": True}}
    write("source_provenance.json", provenance, "derived_provenance")
    manifest = {"schema": MANIFEST_SCHEMA, "files": files,
        "summary": {"path": "summary.json", "sha256": files["summary.json"]["sha256"]},
        "provenance": {"path": "source_provenance.json",
                       "sha256": files["source_provenance.json"]["sha256"]},
        "artifact_inventory": sorted(files),
        "privacy": {"private_paths": False, "reasoning_trace_body": False}}
    PRIV._safe_public(manifest)
    STUDY.BASE._write(out / "artifact_manifest.json", manifest)
    actual = {path.relative_to(out).as_posix() for path in out.rglob("*") if path.is_file()}
    _require(actual == set(files) | {"artifact_manifest.json"}, "manifest 누락/추가 파일")
    _require(all(_sha(out / name) == item["sha256"] for name, item in files.items()),
             "manifest artifact SHA 재검증 실패")
    return deepcopy(manifest)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-result", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-batch", type=Path, required=True)
    parser.add_argument("--mode-result", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--lab", type=Path, required=True)
    parser.add_argument("--anchor", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    export(args.study_result, args.protocol, args.source_batch, args.mode_result,
           args.corpus, args.lab, anchor=args.anchor, out=args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["export", "main"]
