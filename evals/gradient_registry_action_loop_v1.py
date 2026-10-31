"""모델 option과 승인된 recorded gradient pair를 8-subtype 상태에 연결한다."""
# pyright: reportMissingImports=false
from __future__ import annotations

import hashlib
import json
import re
import sys
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "src")]

import gradient_hypothesis_contract_v1 as GH
import gradient_registry_action_options_v1 as O
import recorded_ppo_gradient_probe_v1 as REC

SCHEMA = "gradient_registry_action_loop_result_v1"
REQUEST_SCHEMA = "gradient_recorded_pair_request_v1"
APPROVAL_SCHEMA = "gradient_recorded_pair_approval_v1"
CLAIM_SCHEMA = "gradient_recorded_pair_claim_v1"
DONE_SCHEMA = "gradient_recorded_pair_done_v1"
PROOF_SCHEMA = "gradient_recorded_pair_proof_v1"
MAX_MODEL_ACTIONS = 2


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ref(path: Path) -> dict[str, str]:
    resolved = Path(path).resolve()
    require(resolved.is_file(), f"근거 파일이 없다: {resolved.name}")
    return {"path": str(resolved), "sha256": sha(resolved)}


def write_new(path: Path, value: object) -> None:
    resolved = Path(path).resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON object가 아니다: {Path(path).name}")
    return cast(dict[str, Any], value)


def _source_binding(state: dict[str, Any]) -> dict[str, Any]:
    value = state.get("source_pair_binding")
    require(isinstance(value, dict), "source pair binding이 없다")
    return cast(dict[str, Any], value)


def _reconstruct_initial(state: dict[str, Any]) -> dict[str, Any]:
    require(state.get("subtype_schema") == GH.SUBTYPE_SCHEMA,
            "기존 7-subtype state는 허용하지 않는다")
    source = _source_binding(state)
    pair: dict[str, Any] = {"schema": GH.PAIR_SCHEMA, "subtype_schema": GH.SUBTYPE_SCHEMA}
    for role in ("case", "reference"):
        endpoint = source.get(role)
        require(isinstance(endpoint, dict), f"{role} endpoint가 없다")
        endpoint = cast(dict[str, Any], endpoint)
        plan, receipt = endpoint.get("plan"), endpoint.get("receipt")
        require(isinstance(plan, dict) and isinstance(receipt, dict), f"{role} refs 형식")
        plan = cast(dict[str, Any], plan)
        receipt = cast(dict[str, Any], receipt)
        pair[role] = {"role": role, "plan": plan.get("path"), "receipt": receipt.get("path")}
    budget = state.get("budget_left")
    require(type(budget) is int, "초기 budget 형식")
    rebuilt = GH.initialize(pair, cast(int, budget))
    require(rebuilt == state, "initial state가 현재 immutable pair에서 재구성되지 않는다")
    return rebuilt


def _verify_sources(state: dict[str, Any]) -> None:
    source = _source_binding(state)
    for role in ("case", "reference"):
        endpoint = source[role]
        checked = REC.checked_receipt(Path(endpoint["plan"]["path"]),
                                      Path(endpoint["receipt"]["path"]))
        require(checked.get("source_binding") == endpoint["source_binding"],
                f"{role} current source binding 불일치")
        require(checked.get("evidence_kind") == state["evidence_kind"],
                f"{role} evidence kind 불일치")


def _request(run_id: str, step: int, state: dict[str, Any], context_digest: str,
             action: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": REQUEST_SCHEMA, "run_id": run_id, "step": step,
        "state_digest": state["state_digest"], "context_digest": context_digest,
        "probe": action["target"],
        "source_pair_binding_digest": O.digest(state["source_pair_binding"]),
        "evidence_kind": state["evidence_kind"], "approval_granted": False,
    }


def _same_ref(expected: dict[str, str], supplied: object, label: str) -> Path:
    require(isinstance(supplied, (str, Path)), f"{label} 경로가 없다")
    path = Path(cast(str | Path, supplied)).resolve()
    require(str(path) == expected["path"] and path.is_file() and sha(path) == expected["sha256"],
            f"{label} immutable ref 불일치")
    return path


def _host_proof(pair: dict[str, Any], state: dict[str, Any], request: dict[str, Any],
                probe_out: Path) -> tuple[dict[str, Any], tuple[Path, Path, Path, Path]]:
    require(isinstance(pair, dict) and pair.get("schema") == PROOF_SCHEMA,
            "recorded pair proof schema 불일치")
    source = state["source_pair_binding"]
    paths = (
        _same_ref(source["case"]["plan"], pair.get("case_plan"), "case plan"),
        _same_ref(source["case"]["receipt"], pair.get("case_receipt"), "case receipt"),
        _same_ref(source["reference"]["plan"], pair.get("reference_plan"), "reference plan"),
        _same_ref(source["reference"]["receipt"], pair.get("reference_receipt"),
                  "reference receipt"),
    )
    probe_out = Path(probe_out).resolve()
    request_digest = O.digest(request)
    evidence_paths: dict[str, Path] = {}
    for name in ("approval", "claim", "done"):
        supplied = pair.get(name)
        require(isinstance(supplied, (str, Path)), f"host {name} 경로가 없다")
        evidence = Path(cast(str | Path, supplied)).resolve()
        require(evidence.is_file() and evidence.is_relative_to(probe_out),
                f"host {name}은 probe output 안의 파일이어야 한다")
        evidence_paths[name] = evidence
    plan_refs = {"case": ref(paths[0]), "reference": ref(paths[2])}
    approval = _read(evidence_paths["approval"])
    require(approval == {
        "schema": APPROVAL_SCHEMA, "approval_granted": True,
        "actor": "registered_gradient_probe_host", "request_digest": request_digest,
        "request": ref(probe_out.parent / "request.json"),
        "run_id": request["run_id"], "step": request["step"],
        "state_digest": request["state_digest"], "context_digest": request["context_digest"],
        "probe": request["probe"],
        "source_pair_binding_digest": request["source_pair_binding_digest"],
        "plans": plan_refs, "execution_authority": "recorded_cpu_pair_only",
    }, "host approval이 request/state/source/plan과 다르다")
    claim = _read(evidence_paths["claim"])
    require(claim == {
        "schema": CLAIM_SCHEMA, "status": "claimed", "request_digest": request_digest,
        "exclusive": True, "consume_count": 1,
        "approval": ref(evidence_paths["approval"]),
    }, "exclusive claim 불일치")
    receipt_refs = {"case": ref(paths[1]), "reference": ref(paths[3])}
    done = _read(evidence_paths["done"])
    require(done == {
        "schema": DONE_SCHEMA, "status": "done", "request_digest": request_digest,
        "consume_count": 1, "claim": ref(evidence_paths["claim"]),
        "receipts": receipt_refs, "new_training": False, "gpu": False,
        "sdk_role_executions": 0, "reused_development_records": True,
    }, "done record 불일치")
    proof = {
        "schema": PROOF_SCHEMA, "request_digest": request_digest,
        "approval": ref(evidence_paths["approval"]), "claim": ref(evidence_paths["claim"]),
        "done": ref(evidence_paths["done"]), "plans": plan_refs, "receipts": receipt_refs,
    }
    return proof, paths


def consume_immutable_pair(state: dict, action: dict, request: dict, out: Path) -> dict[str, Any]:
    """등록된 immutable plan/receipt pair를 단 한 번 승인·claim·done으로 소비한다."""
    out = Path(out).resolve()
    require(not out.exists(), "recorded pair consume output은 fresh여야 한다")
    run_id, step = request.get("run_id"), request.get("step")
    require(isinstance(run_id, str) and type(step) is int, "recorded pair request ID/step 형식")
    require(request == _request(cast(str, run_id), cast(int, step), state,
                                O.digest(O.context(state)), action),
            "recorded pair request가 현재 state/action과 다르다")
    require(action.get("action") == "run_probe"
            and action.get("target") in (GH.COUNT_PROBE, GH.GRADIENT_PROBE),
            "recorded pair action이 등록 probe가 아니다")
    _verify_sources(state)
    source = state["source_pair_binding"]
    case_plan = Path(source["case"]["plan"]["path"])
    case_receipt = Path(source["case"]["receipt"]["path"])
    reference_plan = Path(source["reference"]["plan"]["path"])
    reference_receipt = Path(source["reference"]["receipt"]["path"])
    request_path = out.parent / "request.json"
    require(request_path.is_file() and _read(request_path) == request,
            "recorded pair request 파일 불일치")
    out.mkdir(parents=True, exist_ok=False)
    request_digest = O.digest(request)
    approval_path, claim_path, done_path = (
        out / "approval.json", out / "claim.json", out / "done.json")
    write_new(approval_path, {
        "schema": APPROVAL_SCHEMA, "approval_granted": True,
        "actor": "registered_gradient_probe_host", "request_digest": request_digest,
        "request": ref(request_path), "run_id": request["run_id"], "step": request["step"],
        "state_digest": request["state_digest"], "context_digest": request["context_digest"],
        "probe": request["probe"],
        "source_pair_binding_digest": request["source_pair_binding_digest"],
        "plans": {"case": ref(case_plan), "reference": ref(reference_plan)},
        "execution_authority": "recorded_cpu_pair_only",
    })
    write_new(claim_path, {
        "schema": CLAIM_SCHEMA, "status": "claimed", "request_digest": request_digest,
        "exclusive": True, "consume_count": 1, "approval": ref(approval_path),
    })
    REC.checked_receipt(case_plan, case_receipt)
    REC.checked_receipt(reference_plan, reference_receipt)
    write_new(done_path, {
        "schema": DONE_SCHEMA, "status": "done", "request_digest": request_digest,
        "consume_count": 1, "claim": ref(claim_path),
        "receipts": {"case": ref(case_receipt), "reference": ref(reference_receipt)},
        "new_training": False, "gpu": False, "sdk_role_executions": 0,
        "reused_development_records": True,
    })
    return {
        "schema": PROOF_SCHEMA, "case_plan": str(case_plan.resolve()),
        "case_receipt": str(case_receipt.resolve()),
        "reference_plan": str(reference_plan.resolve()),
        "reference_receipt": str(reference_receipt.resolve()),
        "approval": str(approval_path.resolve()), "claim": str(claim_path.resolve()),
        "done": str(done_path.resolve()),
    }


def run_epoch(
    initial_state: dict,
    *,
    run_id: str,
    out: Path,
    verify_context: Callable[[], object],
    complete_factory: Callable[[str, dict], Callable],
    consume_pair: Callable[[dict, dict, dict, Path], dict],
) -> dict[str, Any]:
    """fresh initial state에서 model option→승인→recorded pair→8-subtype 전이를 수행한다."""
    require(isinstance(run_id, str) and re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", run_id) is not None,
            "run_id 형식")
    require(callable(verify_context) and callable(complete_factory) and callable(consume_pair),
            "검증·provider·recorded host callback이 필요하다")
    out = Path(out).resolve()
    require(not out.exists(), "epoch out은 fresh 폴더여야 한다")
    verify_context()
    state = _reconstruct_initial(deepcopy(initial_state))
    require(state["budget_left"] <= MAX_MODEL_ACTIONS, "gradient action budget 상한 초과")
    _verify_sources(state)
    O.context(state)
    out.mkdir(parents=True, exist_ok=False)
    steps: list[dict[str, Any]] = []
    model_actions = pair_consumptions = 0
    provider_kinds: list[str] = []
    stop_reason: str | None = None
    try:
        while state["status"] == "open" and state["budget_left"] > 0:
            require(model_actions < MAX_MODEL_ACTIONS, "model action budget 초과")
            verify_context()
            _verify_sources(state)
            option_context = O.context(state)
            model_actions += 1
            index = model_actions
            stage = out / f"step-{index:02d}"
            write_new(stage / "state-before.json", state)
            write_new(stage / "option-context.json", option_context)
            complete = complete_factory(f"{run_id}.d{index}", deepcopy(state))
            dialogue = O.dialogue(state, complete, verify_context=verify_context)
            provider_kinds.append(dialogue["provider_kind"])
            write_new(stage / "dialogue.json", dialogue)
            action = dialogue["action"]
            before = deepcopy(state)
            proof_ref: dict[str, str] | None = None
            if action["action"] == "hold":
                stop_reason = ("model_hold" if dialogue["model_valid"] else "model_invalid_hold")
            else:
                require(action["action"] == "run_probe"
                        and action["target"] in (GH.COUNT_PROBE, GH.GRADIENT_PROBE),
                        "등록되지 않은 gradient action")
                request = _request(run_id, index, before, dialogue["context_digest"], action)
                write_new(stage / "request.json", request)
                verify_context()
                _verify_sources(before)
                pair = consume_pair(deepcopy(before), deepcopy(action), deepcopy(request),
                                    stage / "probe")
                verify_context()
                _verify_sources(before)
                proof, paths = _host_proof(pair, before, request, stage / "probe")
                write_new(stage / "paired-proof.json", proof)
                proof_ref = ref(stage / "paired-proof.json")
                if action["target"] == GH.COUNT_PROBE:
                    state = GH.advance_count(before, *paths)
                else:
                    state = GH.advance_gradient(before, *paths)
                O.context(state)
                pair_consumptions += 1
                if (state["status"] == "hold" and state["history"]
                        and state["history"][-1]["outcome"] == "unknown"):
                    stop_reason = "unknown_or_reference_not_normal"
            write_new(stage / "state-after.json", state)
            steps.append({
                "step": index, "model_valid": dialogue["model_valid"],
                "selected_option_id": dialogue.get("call", {}).get("review", {}).get(
                    "selected_option_id"),
                "action": deepcopy(action), "state_before_digest": before["state_digest"],
                "state_after_digest": state["state_digest"],
                "hypotheses_after": deepcopy(state["hypotheses"]),
                "dialogue": ref(stage / "dialogue.json"), "paired_proof": proof_ref,
                "state_after": ref(stage / "state-after.json"),
            })
            if action["action"] == "hold" or state["status"] != "open":
                break
        verify_context()
        _verify_sources(state)
        O.context(state)
        write_new(out / "final-state.json", state)
        decision = "confirmed" if state["status"] == "confirmed" else "HOLD"
        if stop_reason is None and decision == "HOLD":
            stop_reason = "budget_exhausted_or_multiple_hypotheses"
        result = {
            "schema": SCHEMA, "status": "complete", "run_id": run_id,
            "decision_outcome": decision, "stop_reason": stop_reason,
            "final_hypothesis_status": state["status"],
            "confirmed_subtype": state["conclusion"] if state["status"] == "confirmed" else None,
            "model_actions": model_actions, "recorded_pair_consumptions": pair_consumptions,
            "steps": steps, "final_state": ref(out / "final-state.json"),
            "scope": {"fixture_provider": bool(provider_kinds)
                      and all(kind == "fixture" for kind in provider_kinds),
                      "actual_model": bool(provider_kinds)
                      and all(kind == "actual_model" for kind in provider_kinds),
                      "new_observation": False, "new_training": False,
                      "gpu": False, "correction": False, "recovery": False},
        }
        write_new(out / "result.json", result)
        return deepcopy(result)
    except BaseException as error:
        write_new(out / "failure.json", {
            "schema": SCHEMA, "status": "invalid", "run_id": run_id,
            "error_type": type(error).__name__, "error": str(error),
            "model_actions": model_actions, "recorded_pair_consumptions": pair_consumptions,
            "automatic_retry": False, "correction": False, "recovery": False,
        })
        raise


__all__ = ["APPROVAL_SCHEMA", "CLAIM_SCHEMA", "DONE_SCHEMA", "PROOF_SCHEMA",
           "REQUEST_SCHEMA", "SCHEMA", "consume_immutable_pair", "run_epoch"]
