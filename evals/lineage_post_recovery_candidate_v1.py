"""완료된 actual epoch 2에서 서로 다른 둘째 등록 교정 후보만 검증한다."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from threading import RLock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "src")]

import lineage_diagnosis_epoch_v1 as E0
import lineage_post_correction_inputs_v1 as POST
import lineage_post_diagnosis_epoch_v1 as E2
import lineage_post_probe_backend_v1 as POST_BACKEND
import lineage_recovery_candidate_v1 as CANDIDATE1
import selection_comparison_controller_v1 as C

SCHEMA = "lineage_post_recovery_candidate_v1"
ROLES = ("case", "reference")
_LOCK = RLock()
_ORIGINAL_CANDIDATE_BACKEND = CANDIDATE1.P


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


def _ref(path: Path) -> dict:
    path = Path(path).resolve()
    _require(path.is_file(), f"근거 파일이 없다: {path}")
    return {"path": str(path), "sha256": _sha(path)}


def _write_new(path: Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


@contextmanager
def typed_backend_scope() -> Iterator[None]:
    """v1 증거 helper가 epoch2 typed plan/receipt만 검증하도록 한정한다."""
    with _LOCK:
        _require(CANDIDATE1.P is _ORIGINAL_CANDIDATE_BACKEND,
                 "candidate v1 backend가 등록 원본이 아니다")
        CANDIDATE1.P = POST_BACKEND
        try:
            yield
        finally:
            CANDIDATE1.P = _ORIGINAL_CANDIDATE_BACKEND


def _completed_ledger(ledger_path: Path, steps: list[dict]) -> tuple[list[dict], dict]:
    ledger = C.S.M.RegistryLedger(ledger_path, C.RC.registry())
    events, requests = ledger.events(), ledger.requests()
    request_ids = [row.get("request_id") for row in steps]
    _require(len(requests) == len(steps) and list(requests) == request_ids
             and all(request.get("state") == "done" for request in requests.values())
             and len(events) == 4 * len(steps),
             "epoch2 ledger request 수 또는 pending/unknown 상태가 다르다")
    return events, requests


def _registered_second_correction(final: dict, applied: list[str], subtype: str) -> tuple[str, str]:
    component, correction = C.registered_correction(final, applied=applied)
    _require(component == subtype and correction not in applied,
             "둘째 교정은 첫 교정과 달라야 한다")
    return component, correction


def prepare_candidate(epoch_result_path: Path, *, post_input: Path, reference_input: Path,
                      gate_path: Path, gate_sha256: str, case_id: str) -> dict:
    """검증된 epoch2 증거에서 승인되지 않은 서로 다른 둘째 교정 후보를 만든다."""
    epoch_result_path = Path(epoch_result_path).resolve()
    post_input, reference_input = Path(post_input).resolve(), Path(reference_input).resolve()
    result = _read(epoch_result_path)
    _require(result.get("schema") == E2.SCHEMA and result.get("status") == "complete"
             and result.get("evidence_scope") == "actual_backend" and result.get("epoch") == 2
             and result.get("epoch2_registered") is True,
             "완료된 actual_backend epoch 2 결과가 아니다")
    _require(result.get("end_state") == "confirmed" and result.get("hold_reason") is None
             and result.get("repair_supported") is True
             and result.get("correction_executed") is False
             and result.get("training_executed") is False,
             "둘째 교정 후보를 만들 수 있는 confirmed 종단 상태가 아니다")
    remaining = result.get("remaining_candidates")
    _require(isinstance(remaining, list) and len(remaining) == 1
             and result.get("confirmed_subtype") == remaining[0],
             "epoch2 단일 confirmed subtype이 다르다")
    assert isinstance(remaining, list)

    _gate, gate_case = E0._gate(Path(gate_path), gate_sha256, case_id)
    _require(case_id == "case-07" and gate_case.get("arm") == "COMPOUND"
             and gate_case.get("seed") == 2307 and gate_case.get("label") == "eligible"
             and gate_case.get("diagnosis_eligible") is True
             and result.get("case_id") == case_id and result.get("gate") == gate_case
             and result.get("gate_sha256") == gate_sha256,
             "epoch2 원래 gate association이 다르다")
    envelope = POST.make_selection_packet(
        post_input, reference_input,
        budget_left=result["max_decisions"], execution_budget_s=result["execution_budget_s"],
    )
    binding = E2._checked_post_binding(envelope, gate_case)
    _require(result.get("source_binding") == binding,
             "epoch2 result의 post observation checkpoint/telemetry 결합이 다르다")
    packet = envelope["packet"]
    applied = packet["applied_corrections"]
    _require(isinstance(applied, list), "epoch2 applied correction 목록이 아니다")
    assert isinstance(applied, list)
    _require(len(applied) == 1
             and result.get("applied_corrections_before") == applied
             and result.get("prior_not_recovered_receipt_sha256")
             == binding["case"]["prior_not_recovered_receipt_sha256"]
             and result.get("current_checkpoint_gate") == binding["second_epoch_gate"],
             "epoch2 prior receipt/current checkpoint gate가 다르다")

    steps = result.get("steps")
    decision_count = result.get("decision_count")
    _require(isinstance(steps, list) and isinstance(decision_count, int)
             and len(steps) == decision_count > 0
             and result.get("probe_order") == [row.get("probe") for row in steps],
             "epoch2 step/decision/probe 수가 다르다")
    assert isinstance(steps, list)
    first_auth = _read(Path(steps[0]["authorization_path"]))
    ledger_path = Path(first_auth["ledger_path"]).resolve()
    _require(ledger_path == epoch_result_path.parent / "probes.ledger.jsonl",
             "epoch2 결과와 다른 ledger다")
    events, requests = _completed_ledger(ledger_path, steps)
    request_ids = [row.get("request_id") for row in steps]

    rows: list[dict] = []
    prefix_events: list[dict] = []
    evidence_steps = []
    with typed_backend_scope():
        for index, step in enumerate(steps, 1):
            request_id = request_ids[index - 1]
            _require(step.get("decision") == index and isinstance(request_id, str)
                     and step.get("outcome") in {"normal", "abnormal"},
                     "epoch2 step 순서 또는 outcome이 다르다")
            request = requests[request_id]
            request_events = [row for row in events if row.get("request_id") == request_id]
            auth_path = Path(step["authorization_path"]).resolve()
            auth = _read(auth_path)
            snapshot, choice = auth.get("snapshot"), auth.get("choice")
            _require(isinstance(snapshot, dict) and isinstance(choice, dict),
                     "epoch2 authorization snapshot/choice가 객체가 아니다")
            assert isinstance(snapshot, dict) and isinstance(choice, dict)
            CANDIDATE1._snapshot_at(snapshot, prefix_events, rows)
            spec = C._frozen_request_spec(snapshot, choice)
            _require(snapshot["inputs"]["packet"] == packet
                     and snapshot["inputs"]["run_id"] == result["run_id"]
                     and snapshot["inputs"]["policy"] == result["policy"]
                     and Path(snapshot["inputs"]["ledger_path"]).resolve() == ledger_path,
                     "epoch2 authorization snapshot 입력 결합이 다르다")
            auth = CANDIDATE1._done_authorization(
                auth_path, step["authorization_sha256"], snapshot, choice,
                spec, request_id, ledger_path, request, request_events,
            )
            _require(isinstance(auth.get("plan_sha256"), dict)
                     and set(auth["plan_sha256"]) == set(ROLES),
                     "epoch2 paired plan SHA map이 다르다")
            measurements, role_evidence = {}, {}
            paired = step.get("paired_receipts")
            _require(isinstance(paired, dict) and set(paired) == set(ROLES),
                     "epoch2 paired receipt 역할이 다르다")
            assert isinstance(paired, dict)
            for role, input_path in (("case", post_input), ("reference", reference_input)):
                plan_path = Path(paired[role]["plan"]["path"]).resolve()
                plan = POST_BACKEND._verify_plan(plan_path)
                _require(plan_path == auth_path.parent / role / "plan.json"
                         and auth["plan_sha256"][role] == _sha(plan_path)
                         and paired[role]["plan"]["sha256"] == _sha(plan_path)
                         and plan["role"] == role and plan["probe"] == spec["probe"]
                         and plan["seed"] == gate_case["seed"],
                         f"epoch2 {role} plan/auth/seed 결합이 다르다")
                measurements[role], role_evidence[role] = CANDIDATE1._receipt(
                    paired[role], role=role, plan_path=plan_path, plan=plan,
                    auth_path=auth_path, auth_sha=step["authorization_sha256"],
                    request_id=request_id, collected_input=input_path, source_binding=binding[role],
                )
            outcome = C.classify_raw(spec["probe"], measurements["case"], measurements["reference"])
            ledger_receipt = request["receipt"]
            _require(outcome == step["outcome"] == ledger_receipt.get("outcome")
                     and step.get("ledger_receipt") == ledger_receipt
                     and ledger_receipt.get("measurement") == measurements["case"]
                     and ledger_receipt.get("exit_code") == 0
                     and ledger_receipt.get("over_budget") is False,
                     "epoch2 raw pair 재분류와 epoch/ledger receipt가 다르다")
            evidence_steps.append({
                "decision": index, "probe": spec["probe"], "request_id": request_id,
                "authorization": _ref(auth_path), "roles": role_evidence,
                "recomputed_outcome": outcome,
            })
            rows.append({"probe": spec["probe"], "outcome": outcome,
                         "measurement": measurements["case"],
                         "gpu_s": ledger_receipt["gpu_s"]})
            prefix_events.extend(request_events)

    protocol_sha = C.V.file_digest(C.V.PROTOCOL)
    final = C.V.build_snapshot(packet, result["run_id"], result["policy"], protocol_sha, ledger_path)
    _require(final["view"]["status"] == "confirmed"
             and final["view"]["remaining_candidates"] == remaining
             and [row["probe"] for row in final["view"]["observations"]] == result["probe_order"],
             "epoch2 final snapshot과 result가 다르다")
    component, correction = _registered_second_correction(
        final, applied, result["confirmed_subtype"])

    post, _telemetry = POST.checked_observation(post_input)
    source = post["private_metadata"]["sources"]
    prior_receipt = Path(source["repair_receipt"]).resolve()
    return {
        "schema": SCHEMA, "status": "validated_candidate", "epoch": 2,
        "case_id": case_id, "source_arm": "COMPOUND", "seed": gate_case["seed"],
        "component": component, "correction": correction,
        "applied_corrections_before": deepcopy(applied),
        "applied_corrections_after": [*applied, correction],
        "restart_candidate": {
            "training_mode": "new_initialization",
            "source_checkpoint_sha256": binding["case"]["checkpoint_sha256"],
        },
        "selected_by": "registered_host_rule", "approval_granted": False,
        "execution_allowed": False, "correction_executed": False, "training_executed": False,
        "inputs": {
            "post_observation": _ref(post_input), "reference": _ref(reference_input),
            "gate": _ref(Path(gate_path)), "epoch_result": _ref(epoch_result_path),
            "ledger": _ref(ledger_path), "prior_not_recovered_receipt": _ref(prior_receipt),
            "current_checkpoint_sha256": binding["case"]["checkpoint_sha256"],
        },
        "provenance": {
            "candidate_removal": post["private_metadata"]["candidate_removal_provenance"],
            "prior_receipt_sha256": binding["case"]["prior_not_recovered_receipt_sha256"],
            "actual_applied_correction": applied[0],
            "second_correction_distinct": True,
        },
        "inspection_evidence": {
            "final_snapshot_digest": C.L.digest(final), "steps": evidence_steps,
        },
        "scope": {
            "activity": "cpu_epoch2_validation_and_candidate_materialization_only",
            "authority": False, "recovery_effect_claimed": False,
            "excluded": ["approval", "ledger_write", "correction_execution", "training", "api"],
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epoch-result", type=Path, required=True)
    parser.add_argument("--post-input", type=Path, required=True)
    parser.add_argument("--reference-input", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--gate-sha256", required=True)
    parser.add_argument("--case-id", default="case-07")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    value = prepare_candidate(
        args.epoch_result, post_input=args.post_input, reference_input=args.reference_input,
        gate_path=args.gate, gate_sha256=args.gate_sha256, case_id=args.case_id,
    )
    _write_new(args.out, value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
