"""완료된 actual lineage epoch 1에서 실행 불가 등록 교정 후보만 만든다."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "src")]

import lineage_diagnosis_epoch_v1 as E
import lineage_gpu_admission_v1 as GPU
import lineage_probe_backend_v1 as P
import lineage_selection_inputs_v1 as INPUT
import selection_comparison_controller_v1 as C

SCHEMA = "lineage_recovery_candidate_v1"
SOURCE_ARMS = {"F1", "COMPOUND"}
ROLES = ("case", "reference")


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def _read(path: Path) -> dict:
    path = Path(path).resolve()
    value = json.loads(path.read_text(encoding="utf-8"),
                       parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)))
    _require(isinstance(value, dict), f"{path.name}은 JSON 객체여야 한다")
    return value


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _ref(path: Path) -> dict:
    path = Path(path).resolve()
    _require(path.is_file(), f"근거 파일이 없다: {path}")
    return {"path": str(path), "sha256": _sha(path)}


def _write_new(path: Path, value: object) -> None:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _snapshot_at(snapshot: dict, events: list[dict], rows: list[dict]) -> dict:
    """원장 파일을 쓰지 않고 승인 직전 snapshot을 원래 event prefix에서 재구성한다."""
    inputs = snapshot.get("inputs", {})
    packet = C.V.S.validate_packet(inputs.get("packet"))
    binding = C.V.epoch_binding(packet, inputs.get("run_id"), inputs.get("policy"),
                                inputs.get("protocol_sha256"), inputs.get("parent"))
    _require(inputs.get("masked") is False, "실행 authorization은 masked snapshot일 수 없다")
    view = C.V._view(packet, rows, None, False)
    pre = {"binding": binding, "prior_observations": rows[:-1],
           "executed_probe": rows[-1]["probe"] if rows else None}
    expected = {
        "schema": "selection_snapshot_v1", "inputs": deepcopy(inputs), "binding": binding,
        "ledger_digest": C.L.digest(events), "pre_receipt_anchor": C.L.digest(pre),
        "view": view, "view_digest": C.L.digest(view), "observation_masked_control": False,
        "proposal_allowed": view["status"] == "open", "execution_authorized": False,
        "approval_granted": False,
    }
    _require(snapshot == expected, "authorization snapshot이 해당 원장 prefix와 다르다")
    return expected


def _done_authorization(path: Path, expected_sha: str, snapshot: dict, choice: dict,
                        spec: dict, request_id: str, ledger_path: Path,
                        request: dict, request_events: list[dict]) -> dict:
    path = Path(path).resolve()
    value = _read(path)
    required = {"schema", "plan_sha256", "snapshot", "choice", "request_spec",
                "ledger_path", "request_id"}
    _require(_sha(path) == expected_sha and set(value) == required
             and value.get("schema") == P.AUTH_SCHEMA, "authorization 파일 SHA 또는 형식")
    _require(value["snapshot"] == snapshot and value["choice"] == choice
             and value["request_spec"] == spec and value["request_id"] == request_id
             and Path(value["ledger_path"]).resolve() == ledger_path,
             "authorization snapshot/request/ledger 결합")
    frozen = C._frozen_request_spec(snapshot, choice)
    _require(frozen == spec and spec["args"] == C.RC.registry()["probes"][spec["probe"]]["args"],
             "authorization context/view/args 결합")
    _require(request.get("state") == "done" and request.get("probe") == spec["probe"]
             and request.get("prereg_digest") == C.L.digest(spec["prereg"])
             and request.get("args_digest") == C.L.digest(spec["args"])
             and [row.get("event") for row in request_events]
             == ["proposed", "approved", "consumed", "receipt"],
             "완료 authorization의 승인→consume→receipt 관계")
    return value


def _measurement(plan: dict) -> dict:
    return (P._legacy_measurement(plan) if plan["probe"] in P.LEGACY_PROBES
            else P._optimizer_measurement(plan))


def _receipt(receipt: dict, *, role: str, plan_path: Path, plan: dict,
             auth_path: Path, auth_sha: str, request_id: str,
             collected_input: Path, source_binding: dict) -> tuple[dict, dict]:
    receipt_path = plan_path.parent / "probe_receipt.json"
    stored = _read(receipt_path)
    _require(receipt == stored and stored.get("schema") == P.RECEIPT_SCHEMA
             and stored.get("status") == "complete" and stored.get("error_type") is None,
             f"{role} 완료 probe receipt")
    _require(stored.get("role") == role and stored.get("probe") == plan["probe"]
             and stored.get("source_arm") == plan["source_arm"]
             and stored.get("seed") == plan["seed"], f"{role} receipt 역할/source 결합")
    _require(stored.get("plan") == {"path": str(plan_path), "sha256": _sha(plan_path)}
             and stored.get("authorization") == {
                 "path": str(auth_path), "sha256": auth_sha, "request_id": request_id},
             f"{role} receipt plan/authorization 결합")
    _require(stored.get("checkpoint_sha256") == source_binding["checkpoint_sha256"]
             == plan["checkpoint"]["sha256"]
             and plan["telemetry_sha256"] == source_binding["telemetry_sha256"],
             f"{role} checkpoint/telemetry 결합")
    collected = stored.get("source_refs", {}).get("collected_input")
    _require(collected == {"path": str(collected_input), "sha256": _sha(collected_input)}
             and plan["collected_input"] == collected, f"{role} collected input 결합")
    _require(stored["source_refs"].get("source_manifest_sha256")
             == plan["source_manifest_sha256"]
             and stored["source_refs"].get("auxiliary_inputs") == plan["auxiliary_inputs"],
             f"{role} source manifest/auxiliary 결합")
    owned_result = plan_path.parent / "owned-result.json"
    _require(owned_result.is_file() and _read(owned_result) == stored.get("execution")
             and P._execution_ok(_read(owned_result))
             and P._execution_ok(stored.get("execution", {})),
             f"{role} owned exit0·natural exit·cleanup 증거")
    gpu_before_path = plan_path.parent / "gpu_before.json"
    gpu_before = _read(gpu_before_path)
    _require(gpu_before == stored.get("gpu_before")
             and gpu_before.get("admission_policy") == GPU.GPU_PROFILE
             and gpu_before.get("busy") is False,
             f"{role} GPU before 원문/profile/strict clear 결합")

    expected_raw = {name: {"path": str(Path(path).resolve()), "sha256": _sha(Path(path))}
                    for name, path in plan["raw_files"].items() if Path(path).is_file()}
    expected_raw["owned_result"] = _ref(owned_result)
    _require(stored.get("raw_refs") == expected_raw, f"{role} raw_refs SHA 결합")

    claim = stored.get("claim", {})
    claim_path = Path(str(claim.get("path", ""))).resolve()
    _require(claim_path.is_file() and claim.get("sha256") == _sha(claim_path)
             and claim.get("body") == _read(claim_path)
             and claim["body"].get("role") == role
             and claim["body"].get("request_id") == request_id
             and claim["body"].get("plan_sha256") == _sha(plan_path),
             f"{role} execution claim 결합")
    measured = _measurement(plan)
    _require(stored.get("measurement") == measured, f"{role} frozen raw 측정 재계산 불일치")
    return measured, {"receipt": _ref(receipt_path), "plan": _ref(plan_path),
                      "claim": _ref(claim_path), "gpu_before": _ref(gpu_before_path),
                      "raw_refs": deepcopy(expected_raw)}


def prepare_candidate(epoch_result_path: Path, *, case_input: Path, reference_input: Path,
                      gate_path: Path, gate_sha256: str, case_id: str) -> dict:
    """검증된 actual epoch 1 증거에서 승인되지 않은 등록 교정 후보를 반환한다."""
    epoch_result_path = Path(epoch_result_path).resolve()
    case_input, reference_input = Path(case_input).resolve(), Path(reference_input).resolve()
    result = _read(epoch_result_path)
    _require(result.get("schema") == E.SCHEMA and result.get("status") == "complete"
             and result.get("evidence_scope") == "actual_backend" and result.get("epoch") == 1,
             "완료된 actual_backend epoch 1 결과가 아니다")
    _require(result.get("end_state") == "confirmed" and result.get("hold_reason") is None
             and result.get("repair_supported") is True
             and result.get("correction_executed") is False
             and result.get("training_executed") is False
             and result.get("epoch2_registered") is False,
             "교정 후보를 만들 수 있는 confirmed 종단 상태가 아니다")
    remaining = result.get("remaining_candidates")
    _require(isinstance(remaining, list) and len(remaining) == 1
             and result.get("confirmed_subtype") == remaining[0], "단일 confirmed subtype 불일치")

    _, gate_case = E._gate(Path(gate_path), gate_sha256, case_id)
    _require(result.get("case_id") == case_id and result.get("gate_sha256") == gate_sha256
             and result.get("gate") == gate_case and gate_case.get("arm") in SOURCE_ARMS
             and gate_case.get("label") == "eligible" and gate_case.get("diagnosis_eligible") is True,
             "등록 교정 대상 source gate 결합")
    envelope = INPUT.make_selection_packet(case_input, reference_input,
                                           budget_left=result["max_decisions"],
                                           execution_budget_s=result["execution_budget_s"])
    _require(envelope.get("schema") == INPUT.PACKET_SCHEMA, "selection packet 형식")
    binding = E._source_binding(envelope, gate_case)
    _require(binding == result.get("source_binding")
             and binding["case"]["seed"] == binding["reference"]["seed"] == gate_case["seed"],
             "epoch source checkpoint/telemetry/seed 결합")
    private = envelope["private_metadata"]
    _require(Path(private["case"]["collected"]).resolve() == case_input
             and Path(private["reference"]["collected"]).resolve() == reference_input,
             "selection packet의 collected 입력 경로가 다르다")

    steps = result.get("steps")
    _require(isinstance(steps, list) and len(steps) == result.get("decision_count") > 0
             and result.get("probe_order") == [row.get("probe") for row in steps],
             "epoch step/decision/probe 수 불일치")
    first_auth = _read(Path(steps[0]["authorization_path"]))
    ledger_path = Path(first_auth["ledger_path"]).resolve()
    _require(ledger_path == epoch_result_path.parent / "probes.ledger.jsonl", "epoch 결과와 다른 ledger")
    ledger = C.S.M.RegistryLedger(ledger_path, C.RC.registry())
    events, requests = ledger.events(), ledger.requests()
    request_ids = [row.get("request_id") for row in steps]
    _require(len(requests) == len(steps) and list(requests) == request_ids
             and all(request.get("state") == "done" for request in requests.values())
             and len(events) == 4 * len(steps), "ledger request 수 또는 pending/unknown 상태")

    packet = envelope["packet"]
    rows: list[dict] = []
    prefix_events: list[dict] = []
    evidence_steps = []
    for index, step in enumerate(steps, 1):
        request_id = request_ids[index - 1]
        _require(step.get("decision") == index and isinstance(request_id, str)
                 and step.get("outcome") in {"normal", "abnormal"}, "epoch step 순서 또는 outcome")
        request = requests[request_id]
        request_events = [row for row in events if row.get("request_id") == request_id]
        auth_path = Path(step["authorization_path"]).resolve()
        auth = _read(auth_path)
        snapshot, choice = auth.get("snapshot"), auth.get("choice")
        _snapshot_at(snapshot, prefix_events, rows)
        spec = C._frozen_request_spec(snapshot, choice)
        _require(snapshot["inputs"]["packet"] == packet
                 and snapshot["inputs"]["run_id"] == result["run_id"]
                 and snapshot["inputs"]["policy"] == result["policy"]
                 and Path(snapshot["inputs"]["ledger_path"]).resolve() == ledger_path,
                 "authorization snapshot epoch 입력 결합")
        auth = _done_authorization(auth_path, step["authorization_sha256"], snapshot, choice,
                                   spec, request_id, ledger_path, request, request_events)
        _require(isinstance(auth.get("plan_sha256"), dict)
                 and set(auth["plan_sha256"]) == set(ROLES), "paired plan SHA map")
        measurements, role_evidence = {}, {}
        paired = step.get("paired_receipts")
        _require(isinstance(paired, dict) and set(paired) == set(ROLES), "paired receipt 역할")
        for role, input_path in (("case", case_input), ("reference", reference_input)):
            plan_path = Path(paired[role]["plan"]["path"]).resolve()
            plan = P._verify_plan(plan_path)
            _require(plan_path == auth_path.parent / role / "plan.json"
                     and auth["plan_sha256"][role] == _sha(plan_path)
                     and paired[role]["plan"]["sha256"] == _sha(plan_path)
                     and plan["role"] == role and plan["probe"] == spec["probe"]
                     and plan["seed"] == gate_case["seed"]
                     and plan["source_arm"] == (gate_case["arm"] if role == "case" else "NONE"),
                     f"{role} plan/auth/source 결합")
            measurements[role], role_evidence[role] = _receipt(
                paired[role], role=role, plan_path=plan_path, plan=plan,
                auth_path=auth_path, auth_sha=step["authorization_sha256"],
                request_id=request_id, collected_input=input_path, source_binding=binding[role])
        outcome = C.classify_raw(spec["probe"], measurements["case"], measurements["reference"])
        ledger_receipt = request["receipt"]
        _require(outcome == step["outcome"] == ledger_receipt.get("outcome")
                 and step.get("ledger_receipt") == ledger_receipt
                 and ledger_receipt.get("measurement") == measurements["case"]
                 and ledger_receipt.get("exit_code") == 0
                 and ledger_receipt.get("over_budget") is False,
                 "raw pair 재분류와 epoch/ledger receipt 불일치")
        evidence_steps.append({"decision": index, "probe": spec["probe"],
                               "request_id": request_id, "authorization": _ref(auth_path),
                               "roles": role_evidence, "recomputed_outcome": outcome})
        rows.append({"probe": spec["probe"], "outcome": outcome,
                     "measurement": measurements["case"], "gpu_s": ledger_receipt["gpu_s"]})
        prefix_events.extend(request_events)

    protocol_sha = C.V.file_digest(C.V.PROTOCOL)
    final = C.V.build_snapshot(packet, result["run_id"], result["policy"], protocol_sha, ledger_path)
    _require(final["view"]["status"] == "confirmed"
             and final["view"]["remaining_candidates"] == remaining
             and [row["probe"] for row in final["view"]["observations"]] == result["probe_order"],
             "현재 완료 ledger의 final snapshot과 epoch 결과가 다르다")
    component, correction = C.registered_correction(final, applied=[])
    _require(component == result["confirmed_subtype"] and gate_case["arm"] != "NONE",
             "registered correction과 confirmed subtype 불일치")

    return {
        "schema": SCHEMA, "status": "validated_candidate", "epoch": 1,
        "case_id": case_id, "source_arm": gate_case["arm"], "seed": gate_case["seed"],
        "component": component, "correction": correction,
        "restart_candidate": {"training_mode": "new_initialization",
                              "source_checkpoint_sha256": binding["case"]["checkpoint_sha256"]},
        "selected_by": "registered_host_rule", "approval_granted": False,
        "execution_allowed": False, "correction_executed": False, "training_executed": False,
        "inputs": {
            "case": {"collected": _ref(case_input), **deepcopy(binding["case"])},
            "reference": {"collected": _ref(reference_input), **deepcopy(binding["reference"])},
            "gate": _ref(Path(gate_path)), "epoch_result": _ref(epoch_result_path),
            "ledger": _ref(ledger_path),
        },
        "inspection_evidence": {"final_snapshot_digest": C.L.digest(final),
                                "steps": evidence_steps},
        "scope": {
            "activity": "cpu_validation_and_candidate_materialization_only",
            "not_selection_controller_actual_evidence": True,
            "not_residual_artifact": True,
            "raw_probe_approval_is_not_correction_approval": True,
            "recovery_effect_claimed": False,
            "excluded": ["approval", "ledger_write", "correction_execution", "training", "api"],
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epoch-result", type=Path, required=True)
    parser.add_argument("--case-input", type=Path, required=True)
    parser.add_argument("--reference-input", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--gate-sha256", required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    value = prepare_candidate(args.epoch_result, case_input=args.case_input,
                              reference_input=args.reference_input, gate_path=args.gate,
                              gate_sha256=args.gate_sha256, case_id=args.case_id)
    _write_new(args.out, value)
    print(json.dumps(value, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
