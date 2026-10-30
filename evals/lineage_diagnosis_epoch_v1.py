"""Lineage source pair의 epoch 1 진단만 수행하는 fail-closed host controller.

교정, 학습, epoch 2 등록은 이 모듈의 범위가 아니다. 실제 실행은 이미 등록된
paired probe backend와 file RPC transport에만 위임한다.
"""

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
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "src")]

import lineage_nim_transport_v1 as T
import lineage_probe_backend_v1 as P
import lineage_selection_inputs_v1 as INPUT
import selection_comparison_controller_v1 as C

SCHEMA = "lineage_diagnosis_epoch_v1"
AUTH_SCHEMA = P.AUTH_SCHEMA
POLICIES = ("registry_next", "fixed_eligible", "model_adaptive")
STOP_GATE_LABELS = {"baseline_invalid", "source_failure_absent", "measurement_unknown"}


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def _read(path: Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"),
                       parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)))
    _require(isinstance(value, dict), f"{Path(path).name}은 JSON 객체여야 한다")
    return value


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_new(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _finite(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value >= 0)


def _gate(gate_path: Path, gate_sha256: str, case_id: str) -> tuple[dict, dict]:
    path = Path(gate_path).resolve()
    _require(path.is_file() and isinstance(gate_sha256, str) and _sha(path) == gate_sha256,
             "source gate 경로 또는 SHA가 다르다")
    gate = _read(path)
    _require(gate.get("schema") == "lineage_evaluation_gate_v1"
             and gate.get("status") == "complete" and isinstance(gate.get("cases"), list),
             "완료된 lineage source gate가 아니다")
    rows = [row for row in gate["cases"] if isinstance(row, dict) and row.get("case_id") == case_id]
    _require(len(rows) == 1, "gate case_id가 없거나 중복됐다")
    row = rows[0]
    required = {"case_id", "seed", "arm", "source_job", "reference_job", "label",
                "fixed_verdict", "diagnosis_eligible"}
    _require(required <= set(row) and row["label"] in {
        "baseline_invalid", "source_failure_absent", "measurement_unknown", "eligible", "out_of_support"
    }, "gate case 필드 또는 label이 다르다")
    return gate, deepcopy(row)


def _source_binding(envelope: dict, gate_case: dict) -> dict:
    private = envelope.get("private_metadata")
    _require(isinstance(private, dict), "selection packet의 비공개 source 결합이 없다")
    bound: dict[str, dict] = {}
    expected_jobs = {"case": gate_case["source_job"], "reference": gate_case["reference_job"]}
    packet_bindings = envelope["packet"]["input_bindings"]
    for role in ("case", "reference"):
        source = private.get(role, {}).get("source")
        _require(isinstance(source, dict)
                 and source.get("job_id") == expected_jobs[role]
                 and source.get("seed") == gate_case["seed"],
                 f"gate와 {role} collected job/seed 결합이 다르다")
        binding = packet_bindings[role]
        bound[role] = {"job_id": source["job_id"], "seed": source["seed"],
                       "checkpoint_sha256": binding["checkpoint_sha256"],
                       "telemetry_sha256": binding["telemetry_sha256"]}
    _require(private["reference"]["source"].get("arm") == "NONE",
             "reference collected source는 NONE이어야 한다")
    _require(private["case"]["source"].get("arm") == gate_case["arm"],
             "gate arm과 case collected source가 다르다")
    if gate_case["arm"] == "NONE":
        _require(gate_case["source_job"] == gate_case["reference_job"]
                 and bound["case"]["job_id"] == bound["reference"]["job_id"]
                 and bound["case"]["checkpoint_sha256"] == bound["reference"]["checkpoint_sha256"],
                 "NONE gate는 같은 job/checkpoint 자기 참조여야 한다")
    return bound


def _base_result(*, run_id: str, policy: str, case_id: str, gate_sha256: str,
                 gate_case: dict, binding: dict, max_decisions: int,
                 execution_budget_s: int, evidence_scope: str) -> dict:
    return {
        "schema": SCHEMA, "status": "complete", "epoch": 1, "run_id": run_id,
        "policy": policy, "case_id": case_id, "gate_sha256": gate_sha256,
        "gate": deepcopy(gate_case), "source_binding": deepcopy(binding),
        "evidence_scope": evidence_scope, "max_decisions": max_decisions,
        "execution_budget_s": execution_budget_s, "decision_count": 0,
        "probe_order": [], "steps": [], "requests": 0,
        "tokens": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
                   "known_subtotal": {"input_tokens": 0, "output_tokens": 0,
                                      "total_tokens": 0},
                   "unknown_cost_requests": {"input_tokens": [], "output_tokens": [],
                                             "total_tokens": []}},
        "wall": {"model_wall_s": 0.0, "known_model_wall_subtotal_s": 0.0,
                 "unknown_model_cost_requests": [], "probe_parent_known_s": 0.0,
                 "probe_child_known_s": 0.0, "probe_child_unknown_roles": []},
        "model_observation_use": [],
        "scope": {"observation_citation_measurement": "This does not establish causality."},
        "selection_actor": [], "model_validity": [], "end_state": "HOLD",
        "confirmed_subtype": None, "remaining_candidates": [], "hold_reason": None,
        "repair_supported": gate_case.get("label") != "out_of_support",
        "repair_metadata": ({"status": "unsupported", "source_gate_label": "out_of_support"}
                            if gate_case.get("label") == "out_of_support" else None),
        "correction_executed": False, "training_executed": False,
        "epoch2_registered": False,
    }


def _accumulate_model(result: dict, dialogue: dict, decision: int) -> None:
    result["requests"] += dialogue["requests"]
    for request_index, call in enumerate(dialogue["calls"], 1):
        raw = call["result"]
        request = {"decision": decision, "request": request_index}
        for name in ("input_tokens", "output_tokens", "total_tokens"):
            value = raw.get(name)
            if type(value) is int and value >= 0:
                result["tokens"]["known_subtotal"][name] += value
                if result["tokens"][name] is not None:
                    result["tokens"][name] += value
            else:
                result["tokens"][name] = None
                result["tokens"]["unknown_cost_requests"][name].append(deepcopy(request))
        wall = raw.get("wall_s")
        if _finite(wall):
            result["wall"]["known_model_wall_subtotal_s"] += wall
            if result["wall"]["model_wall_s"] is not None:
                result["wall"]["model_wall_s"] += wall
        else:
            result["wall"]["model_wall_s"] = None
            result["wall"]["unknown_model_cost_requests"].append(deepcopy(request))


def _record_observation_use(result: dict, choice: dict, decision: int) -> None:
    fact_ids = choice.get("fact_ids", [])
    observation_ids = [item for item in fact_ids
                       if isinstance(item, str) and item.startswith("observation:")]
    prior_prefix = None
    cited_prior = None
    if result["steps"]:
        prior = result["steps"][-1]
        prior_prefix = f"observation:{len(result['steps']) - 1}:{prior['probe']}:"
        cited_prior = any(item.startswith(prior_prefix) for item in observation_ids)
    result["model_observation_use"].append({
        "decision": decision,
        "observation_fact_id_count": len(observation_ids),
        "immediately_prior_receipt_prefix": prior_prefix,
        "cited_immediately_prior_receipt": cited_prior,
    })


def _model_choice(snapshot: dict, transport_root: Path | None, slot_id: str | None,
                  complete_factory: Callable | None) -> tuple[dict, dict]:
    inputs = C.dialogue_input(snapshot, "production")
    _require(inputs["needs_model"], "모델 정책인데 model dialogue가 필요하지 않다")
    _require(transport_root is not None and isinstance(slot_id, str) and bool(slot_id),
             "model_adaptive에는 transport_root와 decision별 slot_id가 필요하다")
    factory = T.file_completion if complete_factory is None else complete_factory
    complete = factory(Path(transport_root), slot_id,
                       C.L.digest(snapshot["binding"]), snapshot["view_digest"], 180)
    engine = C.CALL.engine(snapshot, inputs["view"], inputs["messages"], "production", complete)
    safe_results = [C.CALL.safe_result(row["result"]) for row in engine["calls"]]
    dialogue = {"schema": "selection_controller_dialogue_v1",
                "input_digest": C.L.digest(inputs), "safe_results": safe_results,
                "engine": engine}
    return C.choose(snapshot, "model_adaptive", dialogue), engine


def _plan_path(plan: dict) -> Path:
    path = Path(plan["stage"]) / "plan.json"
    _require(path.is_file(), "probe backend가 plan.json을 만들지 않았다")
    return path.resolve()


def _execute_pair(executor: Callable, paths: dict[str, Path], authorization: Path) -> tuple[dict, float]:
    started = time.perf_counter()
    receipts: dict[str, dict] = {}
    for role in ("reference", "case"):
        try:
            receipt = executor(paths[role], authorization)
        except Exception as error:  # noqa: BLE001 - backend가 봉인한 unknown receipt를 회수한다.
            receipt_path = paths[role].parent / "probe_receipt.json"
            receipt = _read(receipt_path) if receipt_path.is_file() else {
                "status": "unknown", "measurement": None, "execution": {},
                "error_type": type(error).__name__, "role": role,
            }
        _require(isinstance(receipt, dict), "probe executor receipt는 객체여야 한다")
        receipts[role] = deepcopy(receipt)
    return receipts, time.perf_counter() - started


def _settle(ledger: Any, request_id: str, probe: str, receipts: dict,
            parent_wall_s: float) -> tuple[str, dict]:
    unknown = any(receipts[role].get("status") != "complete"
                  or not isinstance(receipts[role].get("measurement"), dict)
                  for role in ("case", "reference"))
    if unknown:
        outcome, measurement, reason = "unknown", {}, "paired_backend_unknown"
    else:
        try:
            outcome = C.classify_raw(probe, receipts["case"]["measurement"],
                                     receipts["reference"]["measurement"])
            measurement, reason = receipts["case"]["measurement"], None
        except Exception:  # noqa: BLE001 - raw 계약 오류도 unknown으로 단 한 번 정산한다.
            outcome, measurement, reason = "unknown", {}, "paired_measurement_contract_invalid"
    receipt = ledger.receipt(request_id, outcome, measurement, parent_wall_s,
                             0 if outcome != "unknown" else None, reason=reason)
    return receipt["outcome"], receipt


def run_epoch(out: Path, case_input: Path, reference_input: Path, policy: str, *,
              run_id: str, lab: Path, probe_protocol: Path, anchor: str,
              gate_path: Path, gate_sha256: str, case_id: str,
              transport_root: Path | None = None, slot_ids: tuple[str, ...] = (),
              max_decisions: int = 3, execution_budget_s: int = 1200,
              approver: str = "codex-main-user-delegated",
              probe_executor: Callable | None = None,
              complete_factory: Callable | None = None) -> dict:
    """Epoch 1 first-pass 진단을 실행하고 새 output에 단일 결과를 봉인한다."""
    _require(policy in POLICIES, "epoch1 정책이 등록되지 않았다")
    _require(type(max_decisions) is int and 1 <= max_decisions <= 7,
             "max_decisions는 1..7 정수여야 한다")
    _require(type(execution_budget_s) is int and execution_budget_s >= 0,
             "execution_budget_s는 음이 아닌 정수여야 한다")
    _require(isinstance(approver, str) and bool(approver.strip()), "승인 주체가 비었다")
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    _gate_value, gate_case = _gate(Path(gate_path), gate_sha256, case_id)
    envelope = INPUT.make_selection_packet(Path(case_input), Path(reference_input),
                                           budget_left=max_decisions,
                                           execution_budget_s=execution_budget_s)
    _require(envelope.get("schema") == INPUT.PACKET_SCHEMA and isinstance(envelope.get("packet"), dict),
             "selection packet envelope 형식이 다르다")
    binding = _source_binding(envelope, gate_case)
    evidence_scope = ("actual_backend" if probe_executor is None and complete_factory is None
                      else "cpu_fixture")
    result = _base_result(run_id=run_id, policy=policy, case_id=case_id,
                          gate_sha256=gate_sha256, gate_case=gate_case, binding=binding,
                          max_decisions=max_decisions, execution_budget_s=execution_budget_s,
                          evidence_scope=evidence_scope)
    packet = envelope["packet"]
    ledger_path = out / "probes.ledger.jsonl"
    protocol_sha = C.V.file_digest(C.V.PROTOCOL)

    if gate_case["label"] in STOP_GATE_LABELS:
        result.update(hold_reason="source_gate_" + gate_case["label"],
                      remaining_candidates=list(packet["remaining_candidates"]))
        _write_new(out / "result.json", result)
        return deepcopy(result)
    if gate_case["arm"] == "NONE":
        _require(gate_case["label"] == "eligible"
                 and gate_case.get("fixed_verdict", {}).get("label") == "healthy",
                 "NONE self-reference gate가 healthy가 아니다")
        result.update(end_state="no_intervention", hold_reason="host_anomaly_gate",
                      remaining_candidates=list(packet["remaining_candidates"]))
        _write_new(out / "result.json", result)
        return deepcopy(result)
    _require(gate_case.get("diagnosis_eligible") is True,
             "gate가 diagnosis eligible이 아니다")

    executor = P.execute_probe if probe_executor is None else probe_executor
    slots = iter(slot_ids)
    for decision in range(1, max_decisions + 1):
        snapshot = C.V.build_snapshot(packet, run_id, policy, protocol_sha, ledger_path)
        view = snapshot["view"]
        if view["status"] != "open":
            break
        if policy == "model_adaptive":
            choice, dialogue = _model_choice(snapshot, transport_root, next(slots, None), complete_factory)
            _accumulate_model(result, dialogue, decision)
            _record_observation_use(result, choice, decision)
        else:
            choice, dialogue = C.choose(snapshot, policy), None
        result["selection_actor"].append(choice["selected_by"])
        result["model_validity"].append(choice["model_status"])
        if choice["action"] != "continue":
            result["hold_reason"] = choice["reason"]
            break
        try:
            spec = C.request_spec(snapshot, choice)
            step_root = out / "probes" / f"step_{decision:02d}"
            plans = {
                "case": P.prepare_probe(Path(case_input), spec["probe"], stage=step_root / "case",
                                        lab=Path(lab), protocol_path=Path(probe_protocol),
                                        anchor=anchor, role="case"),
                "reference": P.prepare_probe(Path(reference_input), spec["probe"],
                                             stage=step_root / "reference", lab=Path(lab),
                                             protocol_path=Path(probe_protocol), anchor=anchor,
                                             role="reference"),
            }
            plan_paths = {role: _plan_path(plan) for role, plan in plans.items()}
        except Exception as error:  # noqa: BLE001 - 승인 전 backend 계약 오류는 HOLD다.
            result["hold_reason"] = "probe_prepare_" + type(error).__name__
            break
        ledger = C.S.M.RegistryLedger(ledger_path, C.RC.registry())
        _require(not any(row["state"] != "done" for row in ledger.requests().values()),
                 "미완결 승인/소비 상태에서는 재실행하지 않는다")
        request = ledger.propose(spec["prereg"], spec["probe"], int(spec["budget_gpu_s"]))
        ledger.approve(request["request_id"], approver)
        ledger.consume(request["request_id"], spec["prereg"], spec["probe"])
        authorization = {
            "schema": AUTH_SCHEMA,
            "plan_sha256": {role: _sha(path) for role, path in plan_paths.items()},
            "snapshot": snapshot, "choice": choice, "request_spec": spec,
            "ledger_path": str(ledger_path), "request_id": request["request_id"],
        }
        authorization_path = step_root / "authorization.json"
        _write_new(authorization_path, authorization)
        receipts, parent_wall = _execute_pair(executor, plan_paths, authorization_path)
        outcome, ledger_receipt = _settle(ledger, request["request_id"], spec["probe"],
                                          receipts, parent_wall)
        child_known = []
        child_unknown = []
        for role in ("reference", "case"):
            wall = receipts[role].get("execution", {}).get("wall_s")
            (child_known if _finite(wall) else child_unknown).append(wall if _finite(wall) else role)
        result["wall"]["probe_parent_known_s"] += parent_wall
        result["wall"]["probe_child_known_s"] += sum(child_known)
        result["wall"]["probe_child_unknown_roles"].extend(child_unknown)
        result["decision_count"] += 1
        result["probe_order"].append(spec["probe"])
        result["steps"].append({
            "decision": decision, "probe": spec["probe"], "request_id": request["request_id"],
            "authorization_path": str(authorization_path), "authorization_sha256": _sha(authorization_path),
            "outcome": outcome, "ledger_receipt": ledger_receipt,
            "paired_receipts": receipts,
        })
        fresh = C.V.build_snapshot(packet, run_id, policy, protocol_sha, ledger_path)
        if outcome == "unknown":
            result["hold_reason"] = "probe_outcome_unknown"
            break
        if fresh["view"]["status"] != "open":
            break

    final = C.V.build_snapshot(packet, run_id, policy, protocol_sha, ledger_path)
    result["remaining_candidates"] = list(final["view"]["remaining_candidates"])
    if final["view"]["status"] == "confirmed":
        result["confirmed_subtype"] = final["view"]["remaining_candidates"][0]
        if gate_case["label"] == "out_of_support":
            result.update(end_state="HOLD", hold_reason="repair_unsupported",
                          repair_supported=False)
        else:
            result["end_state"] = "confirmed"
    elif result["hold_reason"] is None:
        result["hold_reason"] = ("no_informative_probe" if final["view"]["status"] == "none_supported"
                                 else "decision_limit" if result["decision_count"] >= max_decisions
                                 else "no_informative_probe")
    _write_new(out / "result.json", result)
    return deepcopy(result)
