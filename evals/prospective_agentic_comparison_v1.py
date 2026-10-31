"""같은 검증 입력에서 즉시 노출·근거 조회·등록 규칙의 진단 행동을 비교한다."""
from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "bench"), str(ROOT / "src")]

import diagnostic_fact_ledger_v1 as FACTS
import evidence_guided_selector_v1 as SEL
import immutable_git_proof_cache_v1 as CACHE
import lineage_nim_transport_v4 as TRANSPORT
import prospective_diagnosis_inputs_v2 as INPUT
import selection_comparison_call_v2 as CALL

POLICIES = ("eager_model", "evidence_guided_model", "strong_registered_rule")
MODEL_ACTION_BUDGET = 4
PROBE_BUDGET = 3
STUDY = ROOT / "bench/protocols/evidence_guided_study_20261101_v1.json"
SOURCE_FILES = (
    "evals/prospective_agentic_comparison_v1.py",
    "tests/test_prospective_agentic_comparison_v1.py",
    "evals/evidence_guided_selector_v1.py",
    "tests/test_evidence_guided_selector_v1.py",
    "evals/diagnostic_fact_ledger_v1.py",
    "tests/test_diagnostic_fact_ledger_v1.py",
    "evals/recorded_optimizer_probe_v1.py",
    "tests/test_recorded_optimizer_probe_v1.py",
    "evals/immutable_git_proof_cache_v1.py",
    "tests/test_immutable_git_proof_cache_v1.py",
    "evals/prospective_diagnosis_inputs_v2.py",
    "tests/test_prospective_diagnosis_inputs_v2.py",
)


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def digest(value: object) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                     allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def write_new(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def policy_order(case_index: int) -> tuple[str, ...]:
    require(type(case_index) is int and case_index >= 0, "case index 형식")
    offset = case_index % len(POLICIES)
    return POLICIES[offset:] + POLICIES[:offset]


def dialogue(state: dict, complete: Callable[[list, dict], dict], *,
             verify_context: Callable[[], object]) -> dict:
    """의미 재요청은 한 번이며 전송 원문·형식/의미 실패를 모두 보존한다."""
    sent = SEL.messages(state)
    calls: list[dict[str, Any]] = []
    for attempt in range(2):
        verify_context()
        result = complete(deepcopy(sent), CALL.F.request_params(CALL.MODEL))
        verify_context()
        require(isinstance(result, dict), "provider 응답은 객체여야 한다")
        provider_ok = (result.get("provider_model") in CALL.ACCEPTED_PROVIDER_MODELS
                       and result.get("finish_reason") == "stop"
                       and not result.get("error_type") and not result.get("infra_error"))
        review = (SEL.assess(result.get("content"), state) if provider_ok else {
            "status": "invalid", "format_status": "not_run", "semantic_status": "not_run",
            "codes": ["PROVIDER_UNAVAILABLE"], "reasons": ["완결된 등록 모델 응답이 없다"]})
        calls.append({"attempt": attempt + 1, "messages": deepcopy(sent),
                      "messages_sha256": digest(sent), "result": deepcopy(result),
                      "review": deepcopy(review)})
        if review["status"] == "valid" or not provider_ok:
            break
        if attempt == 0:
            sent.extend([{"role": "assistant", "content": result.get("content") or ""},
                         {"role": "user", "content": json.dumps({
                             "validation_errors": review["codes"],
                             "required": "Return one registered action,target,fact_ids JSON object; cite only currently visible non-null facts."},
                             ensure_ascii=False, sort_keys=True)}])
    action = SEL.select_model(calls[-1]["review"], state)
    return {"schema": "prospective_agentic_dialogue_v1", "state_digest": state["state_digest"],
            "calls": calls, "action": action, "final_review": calls[-1]["review"],
            "semantic_feedback_used": len(calls) == 2,
            "model_valid": calls[-1]["review"]["status"] == "valid"}


def _run_epoch(
    case_input: Path, reference_input: Path, *, policy: str, run_id: str,
    out: Path, verify_context: Callable[[], object],
    complete_factory: Callable[[str, dict], Callable] | None,
    execute_pair: Callable[[dict, dict, Path], dict],
) -> dict:
    """전향적 입력을 재검증하고 최대 세 paired receipt 이후 진단만 반환한다.

    실제 정상 gate와 별도 교정 승인은 호출자가 소유한다. 이 함수는 새로운
    source·교정·고정 평가·영상 관측이 실행됐다는 주장을 생성하지 않는다.
    """
    require(policy in POLICIES and isinstance(run_id, str) and bool(run_id), "policy/run ID")
    require(callable(verify_context) and callable(execute_pair), "실제 검증/검사 host가 필요하다")
    require(policy == "strong_registered_rule" or callable(complete_factory), "모델 transport가 없다")
    out = Path(out).resolve()
    require(not out.exists(), "epoch 출력은 fresh 폴더여야 한다")
    verify_context()
    envelope = INPUT.make_packet(case_input, reference_input, budget_left=PROBE_BUDGET)
    _, case_meta = INPUT.checked(case_input)
    _, reference_meta = INPUT.checked(reference_input)
    ledger = FACTS.build(Path(case_meta["runtime_facts"]["case"]["path"]),
                         Path(reference_meta["runtime_facts"]["reference"]["path"]))
    out.mkdir(parents=True, exist_ok=False)
    ledger_path = out / "fact-ledger.json"
    FACTS.write_new(ledger_path, ledger)
    state = SEL.initialize(envelope["packet"], ledger_path, budget_left=PROBE_BUDGET)
    initial_digest = state["state_digest"]
    steps: list[dict] = []
    # Eager arm이 얻는 정보원은 guided arm이 조회하는 원문과 바이트까지 같다.
    # deterministic booleans도 동일하게 제공하여 규칙 파생 정보 차이를 제거한다.
    if policy == "eager_model":
        action = {"action": "read_evidence", "target": SEL.RESOURCE, "fact_ids": []}
        state = SEL.advance(state, action)
        steps.append({"kind": "host_eager_exposure", "action": action,
                      "state_digest": state["state_digest"], "model_request": False})
    start = time.monotonic()
    model_actions = 0
    result: dict[str, Any] = {
        "schema": "prospective_agentic_epoch_v1", "status": "running", "run_id": run_id,
        "policy": policy, "initial_state_digest": initial_digest,
        "source_binding": deepcopy(envelope["source_binding"]), "steps": steps,
        "request_count": 0, "model_actions": 0, "probe_count": 0,
        "correct_family": None, "parent_probe_wall_s": 0.0, "owned_child_wall_s": 0.0,
        "claims": {"recovery": False, "model_version_advantage": False,
                   "fresh_training_observation_from_recorded_probe": False},
    }
    try:
        while state["status"] == "open" and state["budget_left"] > 0:
            verify_context()
            current = INPUT.make_packet(case_input, reference_input, budget_left=PROBE_BUDGET)
            require(current == envelope, "진단 중 source packet이 바뀌었다")
            index = len(steps) + 1
            stage = out / f"step-{index:02d}"
            before = deepcopy(state)
            record: dict[str, Any] = {"state_before": deepcopy(before), "selected_by": (
                "rule" if policy == "strong_registered_rule" else "model")}
            if policy == "strong_registered_rule":
                action = SEL.select_strong_rule(state)
            else:
                if model_actions >= MODEL_ACTION_BUDGET:
                    result["stop_reason"] = "model_action_budget_exhausted"
                    break
                model_actions += 1
                if complete_factory is None:
                    raise ValueError("모델 transport가 없다")
                completion = complete_factory(f"{run_id}.d{model_actions}", state)
                response = dialogue(state, completion, verify_context=verify_context)
                write_new(stage / "dialogue.json", response)
                record["dialogue_sha256"] = hashlib.sha256((stage / "dialogue.json").read_bytes()).hexdigest()
                record["model_valid"] = response["model_valid"]
                result["request_count"] += len(response["calls"])
                action = response["action"]
            record["action"] = deepcopy(action)
            write_new(stage / "selected-action.json", record)
            verify_context()
            if action["action"] == "run_probe":
                pair = execute_pair(before, action, stage / "probe")
                verify_context()
                required = ("case_plan", "case_receipt", "reference_plan", "reference_receipt")
                require(isinstance(pair, dict) and all(key in pair for key in required),
                        "실제 paired plan/receipt 네 경로가 필요하다")
                state = SEL.advance(before, action, **{key: Path(pair[key]) for key in required})
                result["probe_count"] += 1
                for source, target in (("parent_wall_s", "parent_probe_wall_s"),
                                       ("owned_child_wall_s", "owned_child_wall_s")):
                    value = pair.get(source)
                    if (type(value) not in (int, float) or not isinstance(value, (int, float))
                            or not math.isfinite(value) or value < 0):
                        raise ValueError("검사 시간은 확인된 비음수 수치여야 한다")
                    result[target] += value
                record["paired_receipts"] = deepcopy(pair)
            else:
                state = SEL.advance(before, action)
            write_new(stage / "state-after.json", state)
            record["state_after_digest"] = state["state_digest"]
            steps.append(record)
        verify_context()
        write_new(out / "final-state.json", state)
        result.update(status="complete", model_actions=model_actions, final_status=state["status"],
                      decision_outcome=("confirmed" if state["status"] == "confirmed" else "HOLD"),
                      remaining_candidates=state["hypotheses"], observations=state["observed"],
                      confirmed_family=(state["hypotheses"][0] if state["status"] == "confirmed" else None),
                      epoch_wall_s=time.monotonic() - start)
        write_new(out / "result.json", result)
        return deepcopy(result)
    except BaseException as error:
        write_new(out / "failure.json", {**result, "status": "invalid",
                  "error_type": type(error).__name__, "error": str(error),
                  "automatic_retry": False, "model_actions": model_actions})
        raise


def run_epoch(
    case_input: Path, reference_input: Path, *, policy: str, run_id: str,
    out: Path, verify_context: Callable[[], object],
    complete_factory: Callable[[str, dict], Callable] | None,
    execute_pair: Callable[[dict, dict, Path], dict],
) -> dict:
    """세 arm에 같은 immutable commit-blob 재사용을 적용한다. 현재 파일은 매번 읽는다."""
    with CACHE.proof_scope() as cache:
        result = _run_epoch(
            case_input, reference_input, policy=policy, run_id=run_id, out=out,
            verify_context=verify_context, complete_factory=complete_factory,
            execute_pair=execute_pair)
    write_new(Path(out) / "proof-cache-stats.json", cache)
    return result


def transport_factory(root: Path) -> Callable:
    """등록된 atomic NIM queue를 소비한다. transport worker는 호출자가 소유한다."""
    def complete(slot: str, state: dict) -> Callable:
        context = SEL.context(state)
        return TRANSPORT.file_completion(Path(root), slot, state["state_digest"],
                                         digest(context), 180)
    return complete
