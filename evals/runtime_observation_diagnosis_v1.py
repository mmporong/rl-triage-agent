"""검증된 runtime training pair의 단일 진단 epoch를 실행하는 generic host."""
from __future__ import annotations

import hashlib
import sys
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "bench"), str(ROOT / "evals"), str(ROOT / "src")]

import lineage_diagnosis_epoch_v1 as E0
import lineage_nim_transport_v4 as T3
import runtime_observation_development_v1 as DEV
import runtime_observation_view_v1 as RV
import runtime_training_inputs_v1 as INPUT
import selection_comparison_controller_v1 as C

SCHEMA = "runtime_observation_diagnosis_v1"
PREFLIGHT_SCHEMA = "runtime_observation_diagnosis_preflight_v1"
POLICY = "model_adaptive"
BUDGET_LEFT = 3
EXECUTION_BUDGET_S = 1200
MAX_MODEL_SELECTIONS = 3
MODEL_TIMEOUT_S = 180


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_new(path: Path, value: object) -> None:
    E0._write_new(Path(path), value)


def _ref(path: Path) -> dict[str, str]:
    path = Path(path).resolve()
    _require(path.is_file(), f"봉인 파일이 없다: {path.name}")
    return {"path": str(path), "sha256": _sha(path)}


def preflight_path(out: Path) -> Path:
    out = Path(out).resolve()
    return out.parent / f"{out.name}.context-preflight.json"


def _probe_module():
    import runtime_observation_probe_v1 as probe
    return probe


def _packet(case_input: Path, reference_input: Path, case_runtime_path: Path,
            reference_runtime_path: Path | None, applied_corrections: tuple[str, ...],
            prior_receipt: dict | None) -> dict:
    envelope = INPUT.make_packet(
        Path(case_input), Path(reference_input), case_runtime_path=Path(case_runtime_path),
        reference_runtime_path=(None if reference_runtime_path is None
                                else Path(reference_runtime_path)),
        budget_left=BUDGET_LEFT, applied_corrections=list(applied_corrections),
        prior_receipt=deepcopy(prior_receipt))
    _require(envelope.get("schema") == INPUT.PACKET_SCHEMA
             and isinstance(envelope.get("packet"), dict),
             "runtime training packet envelope 형식이 다르다")
    packet = C.S.validate_packet(envelope["packet"])
    _require(packet["budget_left"] == BUDGET_LEFT
             and packet["execution_budget_s"] == EXECUTION_BUDGET_S,
             "진단 budget은 검사 3회·요청 1200초여야 한다")
    return deepcopy(envelope)


def _base_result(run_id: str, packet: dict, parent: dict | None) -> dict:
    return {
        "schema": SCHEMA, "status": "complete", "run_id": run_id,
        "epoch": packet["epoch"], "policy": POLICY, "budget_left": BUDGET_LEFT,
        "execution_budget_s": EXECUTION_BUDGET_S,
        "max_model_selections": MAX_MODEL_SELECTIONS,
        "decision_count": 0, "requests": 0, "probe_order": [], "steps": [],
        "selection_actor": [], "model_validity": [], "runtime_fact_citations": [],
        "tokens": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
                   "known_subtotal": {"input_tokens": 0, "output_tokens": 0,
                                      "total_tokens": 0},
                   "unknown_cost_requests": {"input_tokens": [], "output_tokens": [],
                                             "total_tokens": []}},
        "wall": {"model_wall_s": 0.0, "known_model_wall_subtotal_s": 0.0,
                 "unknown_model_cost_requests": [], "probe_parent_known_s": 0.0,
                 "probe_child_known_s": 0.0, "probe_child_unknown_roles": []},
        "model_observation_use": [], "confirmed_subtype": None,
        "remaining_candidates": list(packet["remaining_candidates"]),
        "final_status": None, "hold_reason": None, "end_state": "HOLD",
        "initial_packet_digest": C.L.digest(packet), "parent": deepcopy(parent),
        "correction_executed": False, "training_executed": False,
        "scope": {"generic_diagnosis_epoch_only": True,
                  "source_gate_owned_by_caller": True,
                  "prior_receipt_revalidation_owned_by_caller": True,
                  "correction_registration_owned_by_caller": True,
                  "automatic_retry": False},
    }


def run_epoch(
    out: Path, case_input: Path, reference_input: Path, *,
    case_runtime_path: Path, reference_runtime_path: Path | None,
    run_id: str, view_protocol: Path, probe_protocol: Path, anchor: str,
    lab: Path, transport_root: Path, slot_ids: tuple[str, ...],
    verify_context: Callable[[], object], applied_corrections: tuple[str, ...] = (),
    prior_receipt: dict | None = None, parent: dict | None = None,
    approver: str = "codex-main-user-delegated", probe_executor: Callable | None = None,
    complete_factory: Callable | None = None, view_fixture: bool = False,
) -> dict:
    """최대 세 번의 모델 선택과 paired probe만 수행하고 correction 전에 멈춘다."""
    _require(callable(verify_context), "verify_context callback이 필요하다")
    _require(isinstance(run_id, str) and bool(run_id), "run_id가 비었다")
    _require(isinstance(approver, str) and bool(approver.strip()), "승인 주체가 비었다")
    _require(isinstance(slot_ids, tuple) and len(slot_ids) == MAX_MODEL_SELECTIONS
             and len(set(slot_ids)) == MAX_MODEL_SELECTIONS
             and all(isinstance(item, str) and item for item in slot_ids),
             "서로 다른 model slot 이름 3개가 필요하다")
    _require(isinstance(applied_corrections, tuple), "applied_corrections는 tuple이어야 한다")
    _require(type(view_fixture) is bool, "view fixture 플래그 형식")
    out = Path(out).resolve()
    preflight = preflight_path(out)
    _require(not out.exists(), "diagnosis output은 fresh 경로여야 한다")
    _require(not preflight.exists(), "diagnosis context preflight가 이미 존재한다")
    case_input, reference_input = Path(case_input).resolve(), Path(reference_input).resolve()
    case_runtime_path = Path(case_runtime_path).resolve()
    reference_runtime = (None if reference_runtime_path is None
                         else Path(reference_runtime_path).resolve())

    verify_context()
    initial = _packet(case_input, reference_input, case_runtime_path, reference_runtime,
                      applied_corrections, prior_receipt)
    packet = initial["packet"]
    if packet["epoch"] == 1:
        _require(parent is None and prior_receipt is None,
                 "epoch 1에는 parent/prior receipt가 없어야 한다")
    else:
        _require(parent is not None and prior_receipt is not None,
                 "epoch 2 이상은 검증된 prior receipt parent가 필요하다")

    def fresh_context() -> None:
        verify_context()
        current = _packet(case_input, reference_input, case_runtime_path, reference_runtime,
                          applied_corrections, prior_receipt)
        _require(current == initial, "진단 중 full input 또는 prior context가 바뀌었다")

    with RV.view_scope(Path(view_protocol), anchor, fixture=view_fixture):
        fresh_context()
        _write_new(preflight, {
            "schema": PREFLIGHT_SCHEMA, "run_id": run_id,
            "input_envelope_digest": C.L.digest(initial), "packet_digest": C.L.digest(packet),
            "view_protocol": str(Path(view_protocol).resolve()), "anchor": anchor,
            "output_was_fresh": True})
        _require(not out.exists(), "preflight가 diagnosis output을 먼저 만들었다")
        out.mkdir(parents=True, exist_ok=False)
        result = _base_result(run_id, packet, parent)
        result["preflight"] = _ref(preflight)
        ledger_path = out / "probes.ledger.jsonl"
        protocol_sha = RV.file_digest(Path(view_protocol))
        slots = iter(slot_ids)
        probe = _probe_module()
        executor = probe.execute_probe if probe_executor is None else probe_executor
        factory = T3.file_completion if complete_factory is None else complete_factory

        for decision in range(1, MAX_MODEL_SELECTIONS + 1):
            fresh_context()
            snapshot = C.V.build_snapshot(
                packet, run_id, POLICY, protocol_sha, ledger_path, parent=parent)
            step_root = out / "steps" / f"step_{decision:02d}"
            snapshot_path = step_root / "snapshot.json"
            _write_new(snapshot_path, snapshot)
            if snapshot["view"]["status"] != "open":
                result["hold_reason"] = "state_" + snapshot["view"]["status"]
                result["steps"].append({
                    "decision": decision, "snapshot": _ref(snapshot_path), "choice": None,
                    "request_spec": None, "authorization": None, "plans": {},
                    "receipts": {}, "outcome": None})
                break

            fresh_context()
            complete = factory(Path(transport_root), next(slots),
                               C.L.digest(snapshot["binding"]), snapshot["view_digest"],
                               MODEL_TIMEOUT_S)
            choice, dialogue, audit = DEV.select(snapshot, complete)
            fresh_context()
            E0._accumulate_model(result, dialogue, decision)
            E0._record_observation_use(result, choice, decision)
            runtime_ids = [item for item in choice.get("fact_ids", [])
                           if isinstance(item, str) and item.startswith("runtime:")]
            result["runtime_fact_citations"].append(
                {"decision": decision, "fact_ids": runtime_ids, "count": len(runtime_ids)})
            result["selection_actor"].append(choice["selected_by"])
            result["model_validity"].append(choice["model_status"])
            choice_path, dialogue_path = step_root / "choice.json", step_root / "dialogue.json"
            _write_new(choice_path, choice)
            _write_new(dialogue_path, {"engine": dialogue, "audit": audit})
            step = {"decision": decision, "snapshot": _ref(snapshot_path),
                    "choice": _ref(choice_path), "dialogue": _ref(dialogue_path),
                    "request_spec": None, "authorization": None,
                    "plans": {}, "receipts": {}, "outcome": None}
            if choice["action"] != "continue":
                result["hold_reason"] = choice["reason"]
                result["steps"].append(step)
                break

            spec = audit["request_spec"]
            _require(spec == C.request_spec(snapshot, choice), "host request spec 재계산 불일치")
            fresh_context()
            spec_path = step_root / "request-spec.json"
            _write_new(spec_path, spec)
            step["request_spec"] = _ref(spec_path)
            plan_paths: dict[str, Path] = {}
            for role, source in (("case", case_input), ("reference", reference_input)):
                fresh_context()
                plan = probe.prepare_probe(
                    source, spec["probe"], stage=step_root / role, lab=Path(lab),
                    protocol_path=Path(probe_protocol), anchor=anchor, role=role)
                plan_path = E0._plan_path(plan)
                plan_paths[role] = plan_path
                step["plans"][role] = _ref(plan_path)

            ledger = C.S.M.RegistryLedger(ledger_path, C.RC.registry())
            _require(not any(row["state"] != "done" for row in ledger.requests().values()),
                     "미완결 승인/소비 상태에서는 재실행하지 않는다")
            fresh_context()
            request = ledger.propose(spec["prereg"], spec["probe"], EXECUTION_BUDGET_S)
            fresh_context()
            ledger.approve(request["request_id"], approver)
            fresh_context()
            ledger.consume(request["request_id"], spec["prereg"], spec["probe"])
            authorization = {
                "schema": probe.AUTH_SCHEMA,
                "plan_sha256": {role: _sha(path) for role, path in plan_paths.items()},
                "snapshot": snapshot, "choice": choice, "request_spec": spec,
                "ledger_path": str(ledger_path), "request_id": request["request_id"]}
            fresh_context()
            authorization_path = step_root / "authorization.json"
            _write_new(authorization_path, authorization)
            step["authorization"] = _ref(authorization_path)

            def checked_executor(plan_path: Path, authorization: Path) -> dict:
                fresh_context()
                return executor(plan_path, authorization)

            receipts, parent_wall = E0._execute_pair(
                checked_executor, plan_paths, authorization_path)
            for role, receipt in receipts.items():
                fresh_context()
                sealed = step_root / f"{role}-receipt.json"
                _write_new(sealed, receipt)
                step["receipts"][role] = _ref(sealed)
            fresh_context()
            outcome, ledger_receipt = E0._settle(
                ledger, request["request_id"], spec["probe"], receipts, parent_wall)
            ledger_receipt_path = step_root / "ledger-receipt.json"
            _write_new(ledger_receipt_path, ledger_receipt)
            step.update(outcome=outcome, ledger_receipt=_ref(ledger_receipt_path),
                        request_id=request["request_id"], probe=spec["probe"])
            result["decision_count"] += 1
            result["probe_order"].append(spec["probe"])
            result["steps"].append(step)
            result["wall"]["probe_parent_known_s"] += parent_wall
            for role in ("reference", "case"):
                wall = receipts[role].get("execution", {}).get("wall_s")
                if E0._finite(wall):
                    result["wall"]["probe_child_known_s"] += wall
                else:
                    result["wall"]["probe_child_unknown_roles"].append(role)
            if outcome == "unknown":
                result["hold_reason"] = "probe_outcome_unknown"
                break

        fresh_context()
        final = C.V.build_snapshot(packet, run_id, POLICY, protocol_sha, ledger_path, parent=parent)
        final_path = out / "final-snapshot.json"
        _write_new(final_path, final)
        result["final_snapshot"] = _ref(final_path)
        result["final_status"] = final["view"]["status"]
        result["remaining_candidates"] = list(final["view"]["remaining_candidates"])
        if final["view"]["status"] == "confirmed":
            result["confirmed_subtype"] = final["view"]["remaining_candidates"][0]
            result["end_state"], result["hold_reason"] = "confirmed", None
        elif result["hold_reason"] is None:
            result["hold_reason"] = ("budget_exhausted" if final["view"]["budget_left"] == 0
                                     else "no_informative_probe")
        result["request_total"] = result["requests"]
        fresh_context()
        _write_new(out / "result.json", result)
        return deepcopy(result)
