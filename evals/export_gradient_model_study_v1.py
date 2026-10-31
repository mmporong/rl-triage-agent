"""RI311 gradient model 개발 측정을 검증해 경로 없는 공개 결과로 투영한다."""
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

import gradient_hypothesis_contract_v1 as GH
import gradient_model_study_v1 as STUDY
import gradient_registry_action_loop_v1 as LOOP
import gradient_registry_action_options_v1 as OPTIONS
import recorded_ppo_gradient_probe_v1 as REC

SCHEMA = "gradient_model_study_public_summary_v1"
CASE_SCHEMA = "gradient_model_study_public_case_v1"
CALL_SCHEMA = "gradient_model_study_public_call_v1"
PROVENANCE_SCHEMA = "gradient_model_study_public_provenance_v1"
MANIFEST_SCHEMA = "gradient_model_study_public_manifest_v1"


def require(ok: object, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON object가 아니다: {Path(path).name}")
    return cast(dict[str, Any], value)


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _ref(path: Path) -> dict[str, str]:
    path = Path(path).resolve()
    require(path.is_file(), f"ref 파일이 없다: {path.name}")
    return {"path": str(path), "sha256": _sha(path)}


def _checked_ref(value: object, expected: Path, label: str) -> dict[str, str]:
    require(isinstance(value, dict) and set(value) == {"path", "sha256"}, f"{label} ref 형식")
    value = cast(dict[str, Any], value)
    path = Path(value["path"]).resolve()
    expected = Path(expected).resolve()
    require(path == expected and path.is_file() and value["sha256"] == _sha(path),
            f"{label} ref/SHA 불일치")
    return cast(dict[str, str], value)


def _request_params(request: dict[str, Any]) -> dict[str, Any]:
    return {key: deepcopy(value) for key, value in request.items() if key != "messages"}


def _usage(response: dict[str, Any]) -> dict[str, Any]:
    values: dict[str, int | None] = {}
    for key in ("input_tokens", "output_tokens", "total_tokens", "reasoning_tokens"):
        value = response.get(key)
        if value is None:
            values[key] = None
            continue
        require(type(value) is int and value >= 0, f"provider {key} 형식")
        values[key] = value
    reported = [key for key, value in values.items() if value is not None]
    status = "complete" if len(reported) == len(values) else ("partial" if reported else "unknown")
    return {"status": status, "reported_fields": reported, "values": values}


def _call_records(root: Path, protocol_sha256: str) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    folders = sorted((root / "calls").glob("*")) if (root / "calls").is_dir() else []
    for folder in folders:
        require(folder.is_dir() and not folder.is_symlink(), "call folder 형식")
        intent_path, claim_path, result_path = (
            folder / "intent.json", folder / "claim.json", folder / "result.json")
        intent, claim, record = _read(intent_path), _read(claim_path), _read(result_path)
        require(intent.get("schema") == STUDY.INTENT_SCHEMA
                and claim == {"schema": STUDY.CLAIM_SCHEMA, "index": intent.get("index"),
                              "slot": intent.get("slot"), "intent_sha256": _sha(intent_path),
                              "exclusive": True}, "call intent/claim 불일치")
        require(record.get("schema") == STUDY.CALL_SCHEMA
                and record.get("index") == intent.get("index")
                and record.get("slot") == intent.get("slot")
                and record.get("automatic_retry") is False,
                "call result 결합 불일치")
        _checked_ref(record.get("intent"), intent_path, "call intent")
        _checked_ref(record.get("claim"), claim_path, "call claim")
        wall = record.get("wall_s")
        require(isinstance(wall, (int, float)) and not isinstance(wall, bool)
                and math.isfinite(float(wall)) and wall >= 0, "call wall 형식")
        require(intent.get("protocol_sha256") == protocol_sha256, "call protocol SHA 불일치")
        slot = intent.get("slot")
        require(isinstance(slot, str) and slot not in records, "call slot 중복/형식")
        records[cast(str, slot)] = {"intent": intent, "record": record, "folder": folder}
    return records


def _initial_state(execution_root: Path, case_id: str, plan_row: dict[str, Any]) -> dict[str, Any]:
    endpoints: dict[str, Any] = {}
    for role in ("case", "reference"):
        folder = execution_root / "sources" / case_id / role
        plan_path, receipt_path = folder / "plan.json", folder / "receipt.json"
        receipt = REC.checked_receipt(plan_path, receipt_path)
        expected = {name: plan_row[role]["raw"][name]["sha256"] for name in STUDY.RAW_FILES}
        require(receipt["source_binding"]["raw_source_sha256"] == expected,
                f"{case_id} {role} raw SHA 불일치")
        endpoints[role] = {"role": role, "plan": plan_path, "receipt": receipt_path}
    return GH.initialize({"schema": GH.PAIR_SCHEMA, "subtype_schema": GH.SUBTYPE_SCHEMA,
                          "case": endpoints["case"], "reference": endpoints["reference"]}, 2)


def _provider_valid(result: dict[str, Any]) -> bool:
    return (result.get("provider_model") in STUDY.MODE.CALL.ACCEPTED_PROVIDER_MODELS
            and result.get("finish_reason") == "stop"
            and not result.get("error_type") and not result.get("infra_error"))


def _public_call(case_id: str, step: int, slot: str, dialogue: dict[str, Any],
                 source: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    intent, record = source["intent"], source["record"]
    call = dialogue["call"]
    require(record.get("response") == call.get("result"), "call response/dialogue 불일치")
    messages = call.get("messages")
    require(isinstance(messages, list)
            and intent.get("messages_sha256") == STUDY.digest(messages), "call messages SHA 불일치")
    messages = cast(list[dict[str, str]], messages)
    request = intent.get("request")
    require(isinstance(request, dict)
            and request == STUDY.MODE.build_request(messages, STUDY.MODEL_MODE)
            and _request_params(request) == plan["request_params"], "call request params 불일치")
    response = cast(dict[str, Any], record["response"])
    review = dialogue["call"]["review"]
    final_json: dict[str, Any] | None = None
    if dialogue["model_valid"]:
        content = response.get("content")
        require(isinstance(content, str), "유효 provider final JSON text가 없다")
        parsed = json.loads(cast(str, content))
        require(isinstance(parsed, dict) and set(parsed) == {"option_id", "fact_ids"},
                "유효 provider final JSON 계약 불일치")
        final_json = cast(dict[str, Any], parsed)
    return {
        "schema": CALL_SCHEMA, "case_id": case_id, "step": step, "slot": slot,
        "intent_sha256": _sha(source["folder"] / "intent.json"),
        "claim_sha256": _sha(source["folder"] / "claim.json"),
        "call_record_sha256": _sha(source["folder"] / "result.json"),
        "request_sha256": STUDY.digest(request),
        "response_sha256": STUDY.digest(response),
        "request_params": deepcopy(plan["request_params"]),
        "provider_model": response.get("provider_model"),
        "finish_reason": response.get("finish_reason"),
        "final_json": final_json,
        "model_valid": dialogue["model_valid"],
        "validation_codes": deepcopy(review.get("codes", [])),
        "selected_option_id": review.get("selected_option_id"),
        "usage": _usage(response), "api_wall_s": float(record["wall_s"]),
        "reasoning_body_included": False, "automatic_retry": False,
    }


def _verify_epoch(execution_root: Path, plan: dict[str, Any], plan_row: dict[str, Any],
                  result_row: dict[str, Any], calls: dict[str, dict[str, Any]],
                  used_slots: set[str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    case_id = plan_row["case_id"]
    require(result_row.get("case_id") == case_id and result_row.get("status") == "complete",
            f"{case_id} result row 불일치")
    epoch_root = execution_root / "epochs" / case_id
    epoch_path = epoch_root / "result.json"
    _checked_ref(result_row.get("epoch"), epoch_path, f"{case_id} epoch")
    epoch = _read(epoch_path)
    require(epoch.get("schema") == LOOP.SCHEMA and epoch.get("status") == "complete"
            and epoch.get("run_id") == f"{case_id}.gradient_model", "epoch schema/run 불일치")
    state = _initial_state(execution_root, case_id, plan_row)
    public_calls: list[dict[str, Any]] = []
    steps = epoch.get("steps")
    require(isinstance(steps, list) and len(steps) <= 2, "epoch steps 형식/상한")
    steps = cast(list[dict[str, Any]], steps)
    for index, step in enumerate(steps, 1):
        require(isinstance(step, dict) and step.get("step") == index, "epoch step 순서")
        stage = epoch_root / f"step-{index:02d}"
        require(_read(stage / "state-before.json") == state, "state-before 재생 불일치")
        dialogue = _read(stage / "dialogue.json")
        _checked_ref(step.get("dialogue"), stage / "dialogue.json", "step dialogue")
        require(dialogue.get("schema") == OPTIONS.DIALOGUE_SCHEMA
                and dialogue.get("state_digest") == state["state_digest"]
                and dialogue.get("context_digest") == OPTIONS.digest(OPTIONS.context(state)),
                "dialogue state/context 불일치")
        provider = cast(dict[str, Any], dialogue["call"]["result"])
        review = (OPTIONS.assess(provider.get("content"), state) if _provider_valid(provider)
                  else dialogue["call"]["review"])
        require(dialogue["call"]["review"] == review, "dialogue review 재생 불일치")
        action = (OPTIONS.select_model(review, state) if _provider_valid(provider)
                  else {"action": "hold", "target": None, "fact_ids": []})
        require(dialogue.get("action") == action, "dialogue action 재생 불일치")
        slot = f"{case_id}.gradient_model.d{index}"
        require(slot in calls and slot not in used_slots, "call slot 누락/중복")
        used_slots.add(slot)
        public_calls.append(_public_call(case_id, index, slot, dialogue, calls[slot], plan))
        before = state
        if action["action"] == "run_probe":
            request = _read(stage / "request.json")
            source = before["source_pair_binding"]
            pair = {"schema": LOOP.PROOF_SCHEMA,
                    "case_plan": source["case"]["plan"]["path"],
                    "case_receipt": source["case"]["receipt"]["path"],
                    "reference_plan": source["reference"]["plan"]["path"],
                    "reference_receipt": source["reference"]["receipt"]["path"],
                    "approval": str(stage / "probe/approval.json"),
                    "claim": str(stage / "probe/claim.json"),
                    "done": str(stage / "probe/done.json")}
            proof, paths = LOOP._host_proof(pair, before, request, stage / "probe")
            require(_read(stage / "paired-proof.json") == proof, "paired proof 재생 불일치")
            _checked_ref(step.get("paired_proof"), stage / "paired-proof.json", "paired proof")
            state = (GH.advance_count(before, *paths) if action["target"] == GH.COUNT_PROBE
                     else GH.advance_gradient(before, *paths))
        else:
            require(step.get("paired_proof") is None, "HOLD에 paired proof가 있다")
        require(_read(stage / "state-after.json") == state, "state-after 재생 불일치")
        _checked_ref(step.get("state_after"), stage / "state-after.json", "state-after")
        require(step.get("state_before_digest") == before["state_digest"]
                and step.get("state_after_digest") == state["state_digest"]
                and step.get("hypotheses_after") == state["hypotheses"], "step digest/hypotheses 불일치")
    require(_read(epoch_root / "final-state.json") == state, "final state 재생 불일치")
    _checked_ref(epoch.get("final_state"), epoch_root / "final-state.json", "final state")
    decision = "confirmed" if state["status"] == "confirmed" else "HOLD"
    require(epoch.get("decision_outcome") == decision
            and epoch.get("confirmed_subtype") == state["conclusion"]
            and epoch.get("model_actions") == len(steps)
            and epoch.get("recorded_pair_consumptions") == sum(
                step["action"]["action"] == "run_probe" for step in steps),
            "epoch final 집계 불일치")
    require(result_row.get("model") == {
        "decision_outcome": epoch["decision_outcome"],
        "confirmed_subtype": epoch["confirmed_subtype"],
        "model_actions": epoch["model_actions"],
        "recorded_pair_consumptions": epoch["recorded_pair_consumptions"],
        "all_model_calls_valid": all(step["model_valid"] for step in steps),
    }, "parent model row 불일치")
    baseline = STUDY._baseline(_initial_state(execution_root, case_id, plan_row))
    require(result_row.get("rules") == baseline, "parent baseline 재계산 불일치")
    expected = GH.SURROGATE if case_id == "g02" else None
    case = {
        "schema": CASE_SCHEMA, "case_id": case_id,
        "model": deepcopy(result_row["model"]), "rules": deepcopy(baseline),
        "expected_subtype": expected,
        "correct_subtype": (epoch["confirmed_subtype"] == expected
                            if expected is not None else epoch["decision_outcome"] != "confirmed"),
        "wrong_confirmation": expected is None and epoch["decision_outcome"] == "confirmed",
        "actions": [deepcopy(step["action"]) for step in steps],
        "call_refs": [f"calls/{slot}.json" for slot in sorted(
            used_slots, key=str) if slot.startswith(f"{case_id}.")],
    }
    return case, public_calls


def _safe_public(value: object) -> None:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    forbidden = ("bench/private", "bench\\private", "reasoning_content", "private_source",
                 "C:\\Users\\", "/home/", "\\Users\\")
    require(not any(item in encoded for item in forbidden), "공개 projection에 비공개 내용이 있다")


def _write_new(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def export(
    plan_path: Path,
    anchor: str,
    result_path: Path,
    out: Path,
    *,
    verify_registered_fn: Callable[[Path, str], dict[str, Any]] = STUDY.verify_registered,
) -> dict[str, Any]:
    """완료된 RI311 원문을 재검증한 뒤 fresh 공개 폴더를 생성한다."""
    plan_path, result_path, out = Path(plan_path).resolve(), Path(result_path).resolve(), Path(out).resolve()
    require(not out.exists(), "공개 output은 fresh여야 한다")
    plan = verify_registered_fn(plan_path, anchor)
    result = _read(result_path)
    execution_root = result_path.parent
    require(result_path == execution_root / "result.json"
            and result.get("schema") == STUDY.RESULT_SCHEMA and result.get("status") == "complete",
            "완료된 study result가 아니다")
    require(result.get("claims") == plan["claims"] and result.get("scope") == {
        "model_selects_probe": True, "host_confirms_from_receipt": True,
        "reasoning_body_public": False, "automatic_retry": False}, "result scope/claims 불일치")
    rows = result.get("rows")
    require(isinstance(rows, list) and len(rows) == len(plan["cases"]) == 4,
            "계획 분모/rows 불일치")
    rows = cast(list[dict[str, Any]], rows)
    plan_cases = cast(list[dict[str, Any]], plan["cases"])
    require([row.get("case_id") for row in rows] == [row["case_id"] for row in plan_cases],
            "case 순서 불일치")
    protocol_sha = _sha(plan_path)
    calls = _call_records(execution_root, protocol_sha)
    used_slots: set[str] = set()
    public_cases: list[dict[str, Any]] = []
    public_calls: list[dict[str, Any]] = []
    for plan_row, result_row in zip(plan_cases, rows):
        case, case_calls = _verify_epoch(execution_root, plan, plan_row, result_row, calls, used_slots)
        public_cases.append(case)
        public_calls.extend(case_calls)
    require(used_slots == set(calls) and len(calls) == result.get("api_requests"),
            "call 분모/소비 불일치")
    require(STUDY._aggregate(rows) == result.get("outcome"), "study aggregate 재계산 불일치")
    normal_cases = [row for row in public_cases if row["case_id"] != "g02"]
    usage_fields = ("input_tokens", "output_tokens", "total_tokens", "reasoning_tokens")
    usage_observation = {
        key: {
            "reported_calls": sum(call["usage"]["values"][key] is not None for call in public_calls),
            "known_sum": sum(call["usage"]["values"][key] or 0 for call in public_calls),
        }
        for key in usage_fields
    }
    for row in usage_observation.values():
        row["complete"] = row["reported_calls"] == len(public_calls)
    summary = {
        "schema": SCHEMA, "status": "complete", "outcome": result["outcome"],
        "planned_cases": 4, "complete_cases": len(public_cases),
        "api_requests": len(public_calls),
        "valid_model_calls": sum(call["model_valid"] for call in public_calls),
        "model_confirmed_subtypes": sum(
            row["model"]["decision_outcome"] == "confirmed" for row in public_cases),
        "correct_subtype_confirmations": sum(
            row["expected_subtype"] is not None and row["correct_subtype"] for row in public_cases),
        "normal_wrong_confirmations": sum(row["wrong_confirmation"] for row in normal_cases),
        "model_holds": sum(row["model"]["decision_outcome"] == "HOLD" for row in public_cases),
        "total_tokens": (usage_observation["total_tokens"]["known_sum"]
                         if usage_observation["total_tokens"]["complete"] else None),
        "reasoning_tokens": (usage_observation["reasoning_tokens"]["known_sum"]
                             if usage_observation["reasoning_tokens"]["complete"] else None),
        "usage_observation": usage_observation,
        "api_wall_s": sum(call["api_wall_s"] for call in public_calls),
        "baselines": {row["case_id"]: deepcopy(row["rules"]) for row in public_cases},
        "claims": {**deepcopy(plan["claims"]), "new_fault": False,
                   "model_version_advantage": False, "efficiency": False,
                   "reasoning_body_public": False},
    }
    plan_logical = (plan_path.relative_to(ROOT).as_posix()
                    if plan_path.is_relative_to(ROOT) else "fixture/plan.json")
    provenance = {
        "schema": PROVENANCE_SCHEMA, "source_anchor": plan["source_anchor"],
        "execution_anchor": anchor,
        "protocol": {"path": plan_logical, "sha256": protocol_sha},
        "raw_inputs": {row["case_id"]: {
            role: deepcopy(row[role]["raw"]) for role in ("case", "reference")}
            for row in plan["cases"]},
        "source_sha256_lf": deepcopy(plan["source_sha256_lf"]),
        "request_params": deepcopy(plan["request_params"]),
        "private_execution_result_sha256": _sha(result_path),
        "private_paths_published": False,
    }
    artifacts: dict[str, object] = {"summary.json": summary, "source-provenance.json": provenance}
    for row in public_cases:
        artifacts[f"cases/{row['case_id']}.json"] = row
    for row in public_calls:
        artifacts[f"calls/{row['slot']}.json"] = row
    for value in artifacts.values():
        _safe_public(value)
    out.mkdir(parents=True, exist_ok=False)
    for relative, value in artifacts.items():
        _write_new(out / relative, value)
    inventory = {relative: _sha(out / relative) for relative in sorted(artifacts)}
    manifest = {"schema": MANIFEST_SCHEMA, "status": "complete",
                "artifact_count": len(inventory), "files": inventory}
    _safe_public(manifest)
    _write_new(out / "artifact-manifest.json", manifest)
    return deepcopy(summary)


def _cli() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--anchor", required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    export(args.plan, args.anchor, args.result, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())


__all__ = ["export"]
