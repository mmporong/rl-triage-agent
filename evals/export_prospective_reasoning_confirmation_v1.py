"""전향 reasoning 확인 원문을 재생 검증해 경로 없는 공개 결과로 내보낸다."""
# pyright: reportMissingImports=false
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "src")]

import run_prospective_reasoning_confirmation_v1 as R  # noqa: E402

SUMMARY_SCHEMA = "prospective_reasoning_confirmation_public_summary_v1"
CASE_SCHEMA = "prospective_reasoning_confirmation_public_case_v1"
CALL_SCHEMA = "prospective_reasoning_confirmation_public_call_v1"
PROVENANCE_SCHEMA = "prospective_reasoning_confirmation_public_provenance_v1"
MANIFEST_SCHEMA = "prospective_reasoning_confirmation_public_manifest_v1"
MODEL_POLICIES = ("legacy_none", "enabled_low")
USAGE_FIELDS = ("input_tokens", "output_tokens", "total_tokens", "reasoning_tokens")


def require(ok: object, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON object가 아니다: {Path(path).name}")
    return cast(dict[str, Any], value)


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _finite(value: object, label: str) -> float:
    require(type(value) in (int, float) and math.isfinite(float(cast(float, value)))
            and float(cast(float, value)) >= 0, f"{label}은 유한 비음수여야 한다")
    return float(cast(float, value))


def _checked_ref(value: object, expected: Path, label: str) -> dict[str, str]:
    require(isinstance(value, dict) and set(value) == {"path", "sha256"}, f"{label} ref 형식")
    ref = cast(dict[str, Any], value)
    path, expected = Path(cast(str, ref["path"])).resolve(), Path(expected).resolve()
    require(path == expected and path.is_file() and ref["sha256"] == _sha(path),
            f"{label} ref/SHA 불일치")
    return cast(dict[str, str], ref)


def _usage(result: dict[str, Any]) -> dict[str, Any]:
    values: dict[str, int | None] = {}
    for key in USAGE_FIELDS:
        value = result.get(key)
        require(value is None or (type(value) is int and value >= 0), f"provider {key} 형식")
        values[key] = cast(int | None, value)
    reported = [key for key, value in values.items() if value is not None]
    return {"status": ("complete" if len(reported) == len(values)
                       else "partial" if reported else "unknown"),
            "reported_fields": reported, "values": values}


def _provider_ok(result: dict[str, Any]) -> bool:
    return (result.get("provider_model") in R.BRIDGE.MODE.CALL.ACCEPTED_PROVIDER_MODELS
            and result.get("finish_reason") == "stop"
            and not result.get("error_type") and not result.get("infra_error")
            and isinstance(result.get("content"), str))


def _pair_proof(pair: dict[str, Any], before: dict[str, Any], action: dict[str, Any],
                probe_out: Path) -> dict[str, Any]:
    ledger_loop = getattr(R.RLOOP, "L1")
    verifier = cast(Callable[..., dict[str, Any]], getattr(ledger_loop, "_pair_proof"))
    return verifier(pair, before, action, probe_out)


def _public_call(dialogue: dict[str, Any], call: dict[str, Any], *, case_id: str,
                 policy: str, step: int) -> dict[str, Any]:
    result = cast(dict[str, Any], call["result"])
    transport = cast(dict[str, str], result["transport_ref"])
    response = Path(transport["path"]).resolve()
    request = response.parent.parent / "requests" / response.name
    final_json: dict[str, Any] | None = None
    content = result.get("content")
    if dialogue["model_valid"] and call is dialogue["calls"][-1]:
        require(isinstance(content, str), "유효 final JSON text가 없다")
        parsed = json.loads(cast(str, content))
        require(isinstance(parsed, dict) and set(parsed) == {"option_id", "fact_ids"},
                "유효 final JSON 계약 불일치")
        final_json = cast(dict[str, Any], parsed)
    provider_usable = _provider_ok(result)
    if provider_usable:
        disposition = ("semantic_valid" if call["review"].get("status") == "valid"
                       else "semantic_invalid")
    elif result.get("infra_error") or result.get("error_type"):
        disposition = "infrastructure_unknown"
    else:
        disposition = "provider_incomplete"
    return {
        "schema": CALL_SCHEMA, "case_id": case_id, "policy": policy, "step": step,
        "attempt": call["attempt"], "slot": call["slot"],
        "request_sha256": _sha(request), "response_sha256": _sha(response),
        "messages_sha256": call["messages_sha256"],
        "request_params": deepcopy(call["request_params"]),
        "provider_model": result.get("provider_model"),
        "finish_reason": result.get("finish_reason"),
        "error_type": result.get("error_type"), "infra_error": result.get("infra_error"),
        "provider_usable": provider_usable, "disposition": disposition,
        "final_json": final_json,
        "review_status": call["review"].get("status"),
        "review_codes": deepcopy(call["review"].get("codes", [])),
        "usage": _usage(result), "api_wall_s": _finite(result.get("wall_s"), "API wall"),
        "reasoning_body_included": False, "automatic_retry": False,
    }


def _verify_dialogue(value: dict[str, Any], state: dict[str, Any], *, case_id: str,
                     policy: str, step: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    require(value.get("schema") == "prospective_reasoning_mode_dialogue_v1"
            and value.get("state_digest") == state["state_digest"]
            and value.get("mode") == policy and value.get("automatic_retry") is False,
            "model dialogue state/mode/schema 불일치")
    calls = value.get("calls")
    require(isinstance(calls, list) and 1 <= len(calls) <= 2, "model call 분모 불일치")
    call_rows = cast(list[dict[str, Any]], calls)
    sent = R.O2.messages(state)
    public_calls: list[dict[str, Any]] = []
    provider_ok = False
    for index, call in enumerate(call_rows, 1):
        require(call.get("attempt") == index and call.get("messages") == sent
                and call.get("messages_sha256") == R.O2.O1._digest(sent),
                "model call attempt/messages 불일치")
        params = R.BRIDGE.MODE.build_request(sent, policy)
        require(params.pop("messages") == sent and call.get("request_params") == params,
                "model call mode params 불일치")
        result = call.get("result")
        require(isinstance(result, dict), "model call result 형식")
        result = cast(dict[str, Any], result)
        R._transport_ref_v2(result.get("transport_ref"), slot=cast(str, call.get("slot")),
                            mode=policy, messages=sent, params=params, returned=result)
        provider_ok = _provider_ok(result)
        expected_review = (R.O2.assess(result.get("content"), state) if provider_ok else {
            "status": "invalid", "format_status": "not_run", "semantic_status": "not_run",
            "codes": ["PROVIDER_UNAVAILABLE"]})
        require(call.get("review") == expected_review, "model call review 재계산 불일치")
        public_calls.append(_public_call(value, call, case_id=case_id, policy=policy, step=step))
        if provider_ok and expected_review["status"] != "valid" and index == 1:
            current = R.O2.context(state)
            sent.extend([{"role": "assistant", "content": result.get("content") or ""},
                         {"role": "user", "content": json.dumps({
                             "validation_errors": expected_review["codes"],
                             "output_schema": current["output_schema"],
                             "action_options": current["action_options"],
                             "count_evidence_groups": current["count_evidence_groups"],
                             "required": "Use one current option_id and exact visible fact_ids only."},
                             ensure_ascii=False, sort_keys=True)}])
    final = call_rows[-1]["review"]
    action = (R.O2.select_model(final, state) if provider_ok else
              {"action": "hold", "target": None, "fact_ids": []})
    require(value.get("final_review") == final and value.get("action") == action
            and value.get("model_valid") == (final["status"] == "valid")
            and value.get("semantic_feedback_used") == (len(call_rows) == 2)
            and value.get("provider_incomplete_hold") == (not provider_ok),
            "model dialogue final/action 집계 불일치")
    return action, public_calls


def _advance_from_artifacts(before: dict[str, Any], action: dict[str, Any], stage: Path,
                            host: dict[str, Any]) -> dict[str, Any]:
    if action["action"] != "run_probe":
        require(host == {"action": action, "request_spec": None,
                         "root_approval": None, "paired_proof": None},
                "비검사 host action 불일치")
        return R.SEL.advance(before, action)
    request = _read(stage / "request-spec.json")
    require(request == R.SEL.request_spec(before, action), "probe request spec 재계산 불일치")
    proof_path = stage / "paired-proof.json"
    _checked_ref(host.get("paired_proof"), proof_path, "paired proof")
    proof = _read(proof_path)
    required = ("case_plan", "case_receipt", "reference_plan", "reference_receipt")
    paths = {name: Path(cast(dict[str, str], proof[name])["path"]) for name in required}
    pair = {name: str(path) for name, path in paths.items()}
    expected = _pair_proof(pair, before, action, stage / "probe")
    require(proof == expected and host.get("root_approval") == proof["root_approval"]
            and host.get("action") == action and host.get("request_spec") == request,
            "paired proof/host approval 불일치")
    return R.SEL.advance(before, action,
                         case_plan=paths["case_plan"], case_receipt=paths["case_receipt"],
                         reference_plan=paths["reference_plan"],
                         reference_receipt=paths["reference_receipt"])


def _verify_model_epoch(root: Path, initial: dict[str, Any], row: dict[str, Any], *,
                        case_id: str, policy: str,
                        category: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    epoch = _read(root / "result.json")
    require(epoch.get("schema") == "prospective_reasoning_mode_epoch_v1"
            and epoch.get("status") == "complete" and epoch.get("run_id") == f"{case_id}.{policy}"
            and epoch.get("mode") == policy and epoch.get("automatic_retry") is False
            and epoch.get("repair") is False and epoch.get("recovery") is False,
            "model epoch schema/run/scope 불일치")
    state, requests, probes = deepcopy(initial), 0, 0
    parent_wall = child_wall = 0.0
    public_calls: list[dict[str, Any]] = []
    steps = epoch.get("steps")
    require(isinstance(steps, list) and len(steps) <= 4, "model epoch steps 상한")
    step_rows = cast(list[dict[str, Any]], steps)
    for index, record in enumerate(step_rows, 1):
        stage, before = root / f"step-{index:02d}", deepcopy(state)
        dialogue = _read(stage / "dialogue.json")
        action, calls = _verify_dialogue(dialogue, before, case_id=case_id,
                                         policy=policy, step=index)
        host = _read(stage / "host-action.json")
        state = _advance_from_artifacts(before, action, stage, host)
        require(_read(stage / "state-after.json") == state
                and record == {"step": index, "action": action,
                               "model_valid": dialogue["model_valid"],
                               "calls": len(dialogue["calls"]),
                               "state_before_digest": before["state_digest"],
                               "state_after_digest": state["state_digest"]},
                "model step/state 재생 불일치")
        requests += len(calls)
        probes += int(action["action"] == "run_probe")
        public_calls.extend(calls)
    derived = {
        "model_actions": len(step_rows), "model_requests": requests, "paired_probes": probes,
        "provider_unusable_count": sum(
            _read(root / f"step-{i:02d}/dialogue.json")["provider_incomplete_hold"]
            for i in range(1, len(step_rows) + 1)),
        "semantic_invalid_count": sum(
            not _read(root / f"step-{i:02d}/dialogue.json")["provider_incomplete_hold"]
            and not _read(root / f"step-{i:02d}/dialogue.json")["model_valid"]
            for i in range(1, len(step_rows) + 1)),
        "decision_outcome": "confirmed" if state["status"] == "confirmed" else "HOLD",
        "confirmed_family": state["hypotheses"][0] if state["status"] == "confirmed" else None,
        "final_status": state["status"],
    }
    for name in ("model_actions", "model_requests", "paired_probes", "provider_unusable_count",
                 "semantic_invalid_count", "decision_outcome", "confirmed_family", "final_status"):
        require(epoch.get(name) == derived[name], f"model epoch {name} 집계 불일치")
    parent_wall = _finite(epoch.get("parent_probe_wall_s"), "parent probe wall")
    child_wall = _finite(epoch.get("owned_child_wall_s"), "owned child wall")
    _finite(epoch.get("epoch_wall_s"), "epoch wall")
    expected_row = {"case_id": case_id, "policy": policy, "status": "complete",
                    **{name: derived[name] for name in ("decision_outcome", "confirmed_family",
                       "model_requests", "paired_probes", "provider_unusable_count",
                       "semantic_invalid_count")}, **R._score(category, epoch)}
    require(row == expected_row, "model parent row 불일치")
    return {"epoch": epoch, "parent_probe_wall_s": parent_wall,
            "owned_child_wall_s": child_wall}, public_calls


def _verify_rule_epoch(root: Path, initial: dict[str, Any], row: dict[str, Any], *,
                       case_id: str, category: str) -> dict[str, Any]:
    epoch = _read(root / "result.json")
    require(epoch.get("schema") == "prospective_agentic_epoch_v1"
            and epoch.get("status") == "complete" and epoch.get("run_id")
            == f"{case_id}.strong_registered_rule" and epoch.get("policy")
            == "strong_registered_rule" and epoch.get("request_count") == 0,
            "rule epoch schema/run/request 불일치")
    state, probes = deepcopy(initial), 0
    steps = epoch.get("steps")
    require(isinstance(steps, list), "rule steps 형식")
    for index, record in enumerate(cast(list[dict[str, Any]], steps), 1):
        stage, before = root / f"step-{index:02d}", deepcopy(state)
        require(record.get("selected_by") == "rule"
                and record.get("state_before") == before, "rule state-before 불일치")
        action = R.SEL.select_strong_rule(before)
        selected = _read(stage / "selected-action.json")
        require(record.get("action") == action
                and selected == {"state_before": before, "selected_by": "rule",
                                 "action": action}, "rule 선택 재계산 불일치")
        if action["action"] == "run_probe":
            pair = record.get("paired_receipts")
            require(isinstance(pair, dict), "rule paired receipt 형식")
            pair = cast(dict[str, Any], pair)
            required = ("case_plan", "case_receipt", "reference_plan", "reference_receipt")
            expected = _pair_proof(pair, before, action, stage / "probe")
            require(all(Path(pair[name]).is_file() for name in required), "rule paired receipt 누락")
            require(_read(stage / "probe/root-approval.json")
                    == _read(Path(expected["root_approval"]["path"])), "rule root approval 불일치")
            state = R.SEL.advance(before, action, **{name: Path(pair[name]) for name in required})
            probes += 1
            _finite(pair.get("parent_wall_s"), "rule parent wall")
            _finite(pair.get("owned_child_wall_s"), "rule child wall")
        else:
            state = R.SEL.advance(before, action)
        require(_read(stage / "state-after.json") == state
                and record.get("state_after_digest") == state["state_digest"],
                "rule state-after 불일치")
    require(_read(root / "final-state.json") == state
            and epoch.get("probe_count") == probes
            and epoch.get("model_actions") == 0
            and epoch.get("decision_outcome")
            == ("confirmed" if state["status"] == "confirmed" else "HOLD")
            and epoch.get("confirmed_family")
            == (state["hypotheses"][0] if state["status"] == "confirmed" else None),
            "rule final 집계 불일치")
    expected_row = {"case_id": case_id, "policy": "strong_registered_rule", "status": "complete",
                    "decision_outcome": epoch["decision_outcome"],
                    "confirmed_family": epoch.get("confirmed_family"),
                    "model_requests": 0, "paired_probes": probes,
                    "provider_unusable_count": 0, "semantic_invalid_count": 0,
                    **R._score(category, epoch)}
    require(row == expected_row, "rule parent row 불일치")
    return {"epoch": epoch,
            "parent_probe_wall_s": _finite(epoch.get("parent_probe_wall_s"), "rule parent wall"),
            "owned_child_wall_s": _finite(epoch.get("owned_child_wall_s"), "rule child wall")}


def _normal_row(row: dict[str, Any], case_id: str, policy: str) -> dict[str, Any]:
    expected = {"case_id": case_id, "policy": policy, "status": "complete",
                "decision_outcome": "HOLD", "model_requests": 0, "paired_probes": 0,
                "correct_confirmed_family": False, "wrong_family_confirmation": False,
                "provider_unusable_count": 0, "semantic_invalid_count": 0,
                "compound_partial": False, "compound_complete": False}
    require(row == expected, "normal host gate row 불일치")
    return expected


def _verify_parent_aggregate(result: dict[str, Any], rows: list[dict[str, Any]],
                             design: dict[str, Any]) -> None:
    require(result.get("planned_rows") == 24
            and result.get("model_requests") == sum(row["model_requests"] for row in rows)
            and result.get("paired_probes") == sum(row["paired_probes"] for row in rows)
            and result.get("sdk_role_executions_upper_bound")
            == 2 * sum(row["paired_probes"] for row in rows)
            and result.get("assessment") == R._assessment(rows)
            and result.get("claims") == design["claims"], "parent result aggregate 불일치")


def _verify_qualification(value: dict[str, Any]) -> None:
    gates = value.get("gates")
    require(value.get("schema") == "prospective_reasoning_source_qualification_v1"
            and value.get("status") == "qualified"
            and value.get("api_requests") == 0
            and value.get("replacement_seed") is False
            and value.get("comparison_outcome") is None
            and isinstance(gates, list) and len(gates) == 8,
            "source qualification이 완료·적격이 아니다")
    gates = cast(list[dict[str, Any]], gates)
    require(sum(item.get("category") == "NONE"
                and item.get("gate", {}).get("reason") == "healthy_gate"
                and item.get("gate", {}).get("run") is not True for item in gates) == 2
            and sum(item.get("category") != "NONE"
                    and item.get("gate", {}).get("run") is True for item in gates) == 6,
            "source qualification gate 2/6 분모 불일치")


def _collect(result_path: Path, source_batch: Path, corpus: Path,
             bridge_root: Path, design: dict[str, Any]) -> tuple[list[dict[str, Any]],
                                                                  list[dict[str, Any]],
                                                                  dict[str, Any]]:
    result = _read(result_path)
    rows = result.get("rows")
    require(result.get("schema") == R.RESULT_SCHEMA and result.get("status") == "complete"
            and isinstance(rows, list) and len(rows) == 24, "완료된 24행 result가 아니다")
    rows = cast(list[dict[str, Any]], rows)
    expected_order = [(f"c{i:02d}", policy) for i in range(1, 9) for policy in R.POLICIES]
    require([(row.get("case_id"), row.get("policy")) for row in rows] == expected_order,
            "parent row 순서/정책 불일치")
    source_rows = R._source_rows(source_batch, design)
    _corpus_value, corpus_rows = R._corpus(corpus)
    R._bind_rows(source_rows, corpus_rows)
    by_case = {row["case_id"]: row for row in source_rows}
    public_rows: list[dict[str, Any]] = []
    public_calls: list[dict[str, Any]] = []
    parent_wall = child_wall = 0.0
    for raw in rows:
        case_id, policy = cast(str, raw["case_id"]), cast(str, raw["policy"])
        source = by_case[case_id]
        if case_id in R.NORMAL_CASES:
            _normal_row(raw, case_id, policy)
            detail = {"parent_probe_wall_s": 0.0, "owned_child_wall_s": 0.0}
        else:
            root = result_path.parent / case_id / policy
            initial = corpus_rows[case_id]["state"]
            if policy in MODEL_POLICIES:
                detail, calls = _verify_model_epoch(root, initial, raw, case_id=case_id,
                                                    policy=policy, category=source["category"])
                public_calls.extend(calls)
            else:
                detail = _verify_rule_epoch(root, initial, raw, case_id=case_id,
                                            category=source["category"])
        parent_wall += detail["parent_probe_wall_s"]
        child_wall += detail["owned_child_wall_s"]
        public_rows.append({"schema": CASE_SCHEMA, **deepcopy(raw),
                            "seed": source["seed"], "category": source["category"],
                            "checkpoint_sha256": source["checkpoint"]["sha256"]})
    _verify_parent_aggregate(result, rows, design)
    bridge_requests = sorted((Path(bridge_root) / "requests").glob("[0-9][0-9][0-9].json"))
    bridge_responses = sorted((Path(bridge_root) / "responses").glob("[0-9][0-9][0-9].json"))
    require(len(bridge_requests) == len(bridge_responses) == len(public_calls)
            == result["model_requests"], "bridge request/response/model 분모 불일치")
    require({call["slot"] for call in public_calls}
            == {_read(path)["slot"] for path in bridge_requests}, "bridge slot 소비 불일치")
    timing = {"api_wall_s": sum(call["api_wall_s"] for call in public_calls),
              "parent_probe_wall_s": parent_wall, "owned_child_wall_s": child_wall,
              "scope": "component_sums_not_whole_execution_wall"}
    return public_rows, public_calls, timing


def _safe(value: object) -> None:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    forbidden = ("bench/private", "bench\\private", "C:\\Users\\", "/home/", "/mnt/",
                 "reasoning_content", '"reasoning"', "api_key", "authorization")
    require(not any(token.lower() in encoded.lower() for token in forbidden),
            "공개 projection에 비공개 경로·추론·자격 정보가 있다")


def _write_new(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def export(result_path: Path, protocol_path: Path, design_path: Path, bridge_path: Path,
           source_batch: Path, corpus: Path, bridge_root: Path, *, anchor: str, out: Path,
           verify_execution_fn: Callable[..., dict[str, Any]] = R.verify_execution,
           collect_fn: Callable[..., tuple[list[dict[str, Any]], list[dict[str, Any]],
                                          dict[str, Any]]] = _collect) -> dict[str, Any]:
    """등록된 완료 결과를 재검증한 뒤 fresh 공개 폴더를 생성한다."""
    paths = [Path(item).resolve() for item in (result_path, protocol_path, design_path,
                                               bridge_path, source_batch, corpus, bridge_root, out)]
    result_path, protocol_path, design_path, bridge_path, source_batch, corpus, bridge_root, out = paths
    require(not out.exists(), "공개 output은 fresh여야 한다")
    spec = verify_execution_fn(protocol_path, anchor, design_path, bridge_path,
                               source_batch, corpus)
    design = _read(design_path)
    require(spec.get("planned_rows") == 24 and spec.get("fault_rows") == 18
            and spec.get("normal_host_gate_rows") == 6, "execution 등록 분모 불일치")
    qualification = _read(corpus.parent / "source-qualification.json")
    _verify_qualification(qualification)
    session, session_protocol = R.BRIDGE._session(bridge_root)
    require(Path(session["protocol"]["path"]) == bridge_path.relative_to(ROOT)
            and session["protocol"]["sha256"] == _sha(bridge_path)
            and session["anchor"] == spec["design_anchor"]
            and session_protocol == _read(bridge_path), "bridge session/protocol/anchor 불일치")
    public_rows, public_calls, timing = collect_fn(
        result_path, source_batch, corpus, bridge_root, design)
    raw = _read(result_path)
    require(raw.get("assessment") == R._assessment([
        {key: value for key, value in row.items()
         if key not in {"schema", "seed", "category", "checkpoint_sha256"}}
        for row in public_rows]), "공개 rows assessment 재계산 불일치")
    by_policy: dict[str, dict[str, Any]] = {}
    for policy in R.POLICIES:
        selected = [row for row in public_rows if row["policy"] == policy]
        faults = [row for row in selected if row["case_id"] in R.FAULT_CASES]
        calls = [call for call in public_calls if call["policy"] == policy]
        by_policy[policy] = {
            "rows": len(selected), "fault_rows": len(faults),
            "model_requests": sum(row["model_requests"] for row in selected),
            "paired_probes": sum(row["paired_probes"] for row in selected),
            "provider_usable_calls": sum(call["provider_usable"] for call in calls),
            "semantic_valid_calls": sum(call["disposition"] == "semantic_valid"
                                         for call in calls),
            "semantic_invalid_calls": sum(call["disposition"] == "semantic_invalid"
                                           for call in calls),
            "infrastructure_unknown_calls": sum(
                call["disposition"] == "infrastructure_unknown" for call in calls),
            "provider_incomplete_calls": sum(call["disposition"] == "provider_incomplete"
                                             for call in calls),
            "provider_calls": len(calls),
            "correct_confirmed_families": sum(row["correct_confirmed_family"] for row in faults),
            "wrong_family_confirmations": sum(row["wrong_family_confirmation"] for row in faults),
            "compound_partial": sum(row["compound_partial"] for row in faults),
            "compound_complete": sum(row["compound_complete"] for row in faults),
            "holds": sum(row["decision_outcome"] == "HOLD" for row in selected),
            "provider_unusable_events": sum(row["provider_unusable_count"] for row in selected),
            "semantic_invalid_events": sum(row["semantic_invalid_count"] for row in selected),
        }
    require(all(value["rows"] == 8 and value["fault_rows"] == 6 for value in by_policy.values())
            and by_policy["strong_registered_rule"]["model_requests"] == 0,
            "정책별 8/6행 또는 rule API0 분모 불일치")
    seed_contrasts = []
    for seed in design["seeds"]:
        rows = [row for row in public_rows if row["seed"] == seed
                and row["case_id"] in R.FAULT_CASES]
        seed_contrasts.append({"seed": seed,
            "legacy_none_correct": sum(row["correct_confirmed_family"]
                                       for row in rows if row["policy"] == "legacy_none"),
            "enabled_low_correct": sum(row["correct_confirmed_family"]
                                       for row in rows if row["policy"] == "enabled_low"),
            "strong_rule_correct": sum(row["correct_confirmed_family"]
                                       for row in rows if row["policy"] == "strong_registered_rule")})
    usage = {key: {"reported_calls": sum(call["usage"]["values"][key] is not None
                                                for call in public_calls),
                   "known_sum": sum(call["usage"]["values"][key] or 0
                                    for call in public_calls)} for key in USAGE_FIELDS}
    for value in usage.values():
        value["complete"] = value["reported_calls"] == len(public_calls)
    summary = {
        "schema": SUMMARY_SCHEMA, "status": "complete", "assessment": raw["assessment"],
        "planned_rows": 24, "complete_rows": len(public_rows),
        "independent_source_seeds": len(design["seeds"]),
        "policies": by_policy, "seed_paired_contrasts": seed_contrasts,
        "model_requests": len(public_calls),
        "paired_probes": sum(row["paired_probes"] for row in public_rows),
        "sdk_role_executions_upper_bound": raw["sdk_role_executions_upper_bound"],
        "timing": timing, "usage_observation": usage,
        "uncertainty": {"independent_unit": "source_seed", "independent_units": 2,
                        "fault_rows_per_policy": 6, "fault_conditions_per_seed": 3,
                        "fault_rows_are_not_independent_seeds": True,
                        "p_value_reported": False,
                        "small_sample": True,
                        "post_exposure_design": design["analysis"]["post_exposure_design"]},
        "claims": {**deepcopy(design["claims"]), "visual_benefit": False,
                   "new_recovery": False, "model_version_advantage": False,
                   "reasoning_body_public": False},
    }
    provenance = {
        "schema": PROVENANCE_SCHEMA, "design_anchor": spec["design_anchor"],
        "execution_anchor": anchor,
        "design_protocol": {"path": design_path.relative_to(ROOT).as_posix(),
                            "sha256": _sha(design_path)},
        "bridge_protocol": {"path": bridge_path.relative_to(ROOT).as_posix(),
                            "sha256": _sha(bridge_path)},
        "execution_protocol": {"path": protocol_path.relative_to(ROOT).as_posix(),
                               "sha256": _sha(protocol_path)},
        "source_batch_sha256": _sha(source_batch), "corpus_sha256": _sha(corpus),
        "source_qualification_sha256": _sha(corpus.parent / "source-qualification.json"),
        "private_execution_result_sha256": _sha(result_path),
        "source_sha256_lf": deepcopy(design["source_sha256_lf"]),
        "request_params": deepcopy(design["bridge"]["modes"]),
        "private_paths_published": False, "reasoning_body_published": False,
    }
    artifacts: dict[str, object] = {"summary.json": summary,
                                   "source-provenance.json": provenance}
    for row in public_rows:
        artifacts[f"cases/{row['case_id']}.{row['policy']}.json"] = row
    for call in public_calls:
        artifacts[f"calls/{call['slot']}.json"] = call
    for value in artifacts.values():
        _safe(value)
    out.mkdir(parents=True, exist_ok=False)
    for relative, value in artifacts.items():
        _write_new(out / relative, value)
    inventory = {relative: _sha(out / relative) for relative in sorted(artifacts)}
    manifest = {"schema": MANIFEST_SCHEMA, "status": "complete",
                "artifact_count": len(inventory), "files": inventory}
    _safe(manifest)
    _write_new(out / "artifact-manifest.json", manifest)
    return deepcopy(summary)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("result", "protocol", "design", "bridge", "source-batch", "corpus",
                 "bridge-root", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--anchor", required=True)
    args = parser.parse_args()
    export(args.result, args.protocol, args.design, args.bridge, args.source_batch,
           args.corpus, args.bridge_root, anchor=args.anchor, out=args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["export"]
