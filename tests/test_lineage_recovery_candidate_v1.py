from __future__ import annotations

import hashlib
import importlib
import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "src")]
M = importlib.import_module("lineage_recovery_candidate_v1")


def save(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return path.resolve()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    case_input = save(tmp_path / "case.json", {"fixture": "case"})
    reference_input = save(tmp_path / "reference.json", {"fixture": "reference"})
    gate = save(tmp_path / "gate.json", {
        "schema": "lineage_evaluation_gate_v1", "status": "complete",
        "cases": [{"case_id": "case-01", "seed": 2301, "arm": "F1",
                   "source_job": "f1-s2301", "reference_job": "none-s2301",
                   "label": "eligible", "fixed_verdict": {"label": "unhealthy"},
                   "diagnosis_eligible": True}],
    })
    packet = M.C.S.build_packet(
        epoch=1, applied_corrections=[], prior_receipt=None,
        facts={"telemetry:case:Train/mean_reward": 1.0},
        input_bindings={
            "case": {"checkpoint_sha256": "1" * 64, "telemetry_sha256": "2" * 64},
            "reference": {"checkpoint_sha256": "3" * 64, "telemetry_sha256": "4" * 64},
        }, budget_left=3, execution_budget_s=1200)
    envelope = {
        "schema": M.INPUT.PACKET_SCHEMA, "packet": packet,
        "private_metadata": {
            "case": {"collected": str(case_input),
                     "source": {"job_id": "f1-s2301", "seed": 2301, "arm": "F1"}},
            "reference": {"collected": str(reference_input),
                          "source": {"job_id": "none-s2301", "seed": 2301, "arm": "NONE"}},
        },
    }
    monkeypatch.setattr(M.INPUT, "make_selection_packet", lambda *_args, **_kwargs: deepcopy(envelope))

    run = tmp_path / "run"
    ledger_path = (run / "probes.ledger.jsonl").resolve()
    snapshot = M.C.V.build_snapshot(packet, "fixture_run", "registry_next",
                                    M.C.V.file_digest(M.C.V.PROTOCOL), ledger_path)
    choice = M.C.choose(snapshot, "registry_next")
    spec = M.C.request_spec(snapshot, choice)
    plans = {}
    paired = {}
    for role in M.ROLES:
        plan_path = save(run / "probes/step_01" / role / "plan.json", {"role": role})
        plans[role] = plan_path
        paired[role] = {"plan": {"path": str(plan_path), "sha256": sha(plan_path)}}
    ledger = M.C.S.M.RegistryLedger(ledger_path, M.C.RC.registry())
    request = ledger.propose(spec["prereg"], spec["probe"], spec["budget_gpu_s"])
    ledger.approve(request["request_id"], "fixture-approver")
    ledger.consume(request["request_id"], spec["prereg"], spec["probe"])
    auth = {
        "schema": M.P.AUTH_SCHEMA, "plan_sha256": {role: sha(path) for role, path in plans.items()},
        "snapshot": snapshot, "choice": choice, "request_spec": spec,
        "ledger_path": str(ledger_path), "request_id": request["request_id"],
    }
    auth_path = save(run / "probes/step_01/authorization.json", auth)
    case_measurement = {"critic_change": 0.01, "actor_change": 0.5}
    reference_measurement = {"critic_change": 1.0, "actor_change": 0.5}
    receipt = ledger.receipt(request["request_id"], "abnormal", case_measurement, 1.0, 0)
    for role in M.ROLES:
        paired[role].update({"role": role, "measurement": (
            case_measurement if role == "case" else reference_measurement)})

    result = {
        "schema": M.E.SCHEMA, "status": "complete", "evidence_scope": "actual_backend",
        "epoch": 1, "run_id": "fixture_run", "policy": "registry_next", "case_id": "case-01",
        "gate_sha256": sha(gate), "gate": json.loads(gate.read_text(encoding="utf-8"))["cases"][0],
        "source_binding": {
            "case": {"job_id": "f1-s2301", "seed": 2301, "checkpoint_sha256": "1" * 64,
                     "telemetry_sha256": "2" * 64},
            "reference": {"job_id": "none-s2301", "seed": 2301, "checkpoint_sha256": "3" * 64,
                          "telemetry_sha256": "4" * 64},
        },
        "max_decisions": 3, "execution_budget_s": 1200, "decision_count": 1,
        "probe_order": [spec["probe"]], "steps": [{
            "decision": 1, "probe": spec["probe"], "request_id": request["request_id"],
            "authorization_path": str(auth_path), "authorization_sha256": sha(auth_path),
            "outcome": "abnormal", "ledger_receipt": receipt, "paired_receipts": paired,
        }],
        "end_state": "confirmed", "hold_reason": None, "repair_supported": True,
        "remaining_candidates": ["optimizer_value"], "confirmed_subtype": "optimizer_value",
        "correction_executed": False, "training_executed": False, "epoch2_registered": False,
    }
    result_path = save(run / "result.json", result)

    def verify_plan(path: Path) -> dict:
        role = Path(path).parent.name
        return {"role": role, "probe": spec["probe"], "seed": 2301,
                "source_arm": "F1" if role == "case" else "NONE"}

    def receipt(value: dict, **kwargs):
        role = kwargs["role"]
        measurement = case_measurement if role == "case" else reference_measurement
        return deepcopy(measurement), {"receipt": {"path": f"fixture-{role}", "sha256": "a" * 64}}

    monkeypatch.setattr(M.P, "_verify_plan", verify_plan)
    monkeypatch.setattr(M, "_receipt", receipt)
    monkeypatch.setattr(M.C, "registered_correction",
                        lambda _snapshot, applied: ("optimizer_value", "C_optimizer")
                        if applied == [] else (_ for _ in ()).throw(ValueError("applied")))
    return {"result": result, "result_path": result_path, "case": case_input,
            "reference": reference_input, "gate": gate, "gate_sha": sha(gate),
            "envelope": envelope, "ledger": ledger_path, "auth": auth_path}


def call(data: dict) -> dict:
    return M.prepare_candidate(data["result_path"], case_input=data["case"],
                               reference_input=data["reference"], gate_path=data["gate"],
                               gate_sha256=data["gate_sha"], case_id="case-01")


def test_valid_completed_actual_epoch_creates_one_non_executable_candidate(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = fixture(tmp_path, monkeypatch)
    value = call(data)
    assert value["schema"] == M.SCHEMA
    assert (value["component"], value["correction"]) == ("optimizer_value", "C_optimizer")
    assert value["selected_by"] == "registered_host_rule"
    assert value["approval_granted"] is value["execution_allowed"] is False
    assert value["scope"]["raw_probe_approval_is_not_correction_approval"] is True
    assert value["scope"]["recovery_effect_claimed"] is False


@pytest.mark.parametrize(("field", "value"), [
    ("evidence_scope", "cpu_fixture"), ("end_state", "HOLD"),
    ("repair_supported", False), ("remaining_candidates", ["optimizer_value", "reward"]),
])
def test_rejects_non_actual_hold_unsupported_or_non_singleton(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, value: object) -> None:
    data = fixture(tmp_path, monkeypatch)
    data["result"][field] = value
    save(data["result_path"], data["result"])
    with pytest.raises(ValueError):
        call(data)


def test_rejects_pending_or_unknown_and_different_ledger(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pending = fixture(tmp_path / "pending", monkeypatch)
    lines = pending["ledger"].read_text(encoding="utf-8").splitlines()
    pending["ledger"].write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="pending|request|ledger"):
        call(pending)

    unknown = fixture(tmp_path / "unknown", monkeypatch)
    unknown["result"]["steps"][0]["outcome"] = "unknown"
    save(unknown["result_path"], unknown["result"])
    with pytest.raises(ValueError, match="outcome"):
        call(unknown)

    other = fixture(tmp_path / "other", monkeypatch)
    auth = json.loads(other["auth"].read_text(encoding="utf-8"))
    auth["ledger_path"] = str((tmp_path / "foreign.ledger.jsonl").resolve())
    save(other["auth"], auth)
    other["result"]["steps"][0]["authorization_sha256"] = sha(other["auth"])
    save(other["result_path"], other["result"])
    with pytest.raises(ValueError, match="ledger"):
        call(other)


def test_rejects_checkpoint_reference_and_authorization_args_tampering(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    unsupported = fixture(tmp_path / "unsupported", monkeypatch)
    gate_value = json.loads(unsupported["gate"].read_text(encoding="utf-8"))
    gate_value["cases"][0]["arm"] = "F2"
    save(unsupported["gate"], gate_value)
    unsupported["gate_sha"] = sha(unsupported["gate"])
    unsupported["result"]["gate"] = gate_value["cases"][0]
    unsupported["result"]["gate_sha256"] = unsupported["gate_sha"]
    save(unsupported["result_path"], unsupported["result"])
    with pytest.raises(ValueError, match="source gate"):
        call(unsupported)

    cp = fixture(tmp_path / "cp", monkeypatch)
    cp["result"]["source_binding"]["case"]["checkpoint_sha256"] = "9" * 64
    save(cp["result_path"], cp["result"])
    with pytest.raises(ValueError, match="source"):
        call(cp)

    reference = fixture(tmp_path / "reference", monkeypatch)
    reference["envelope"]["private_metadata"]["reference"]["source"]["arm"] = "F1"
    monkeypatch.setattr(M.INPUT, "make_selection_packet",
                        lambda *_args, **_kwargs: deepcopy(reference["envelope"]))
    with pytest.raises(ValueError, match="NONE"):
        call(reference)

    args = fixture(tmp_path / "args", monkeypatch)
    auth = json.loads(args["auth"].read_text(encoding="utf-8"))
    auth["request_spec"]["args"] = {"tampered": True}
    save(args["auth"], auth)
    args["result"]["steps"][0]["authorization_sha256"] = sha(args["auth"])
    save(args["result_path"], args["result"])
    with pytest.raises(ValueError, match="args|request"):
        call(args)


@pytest.mark.parametrize("tamper", ["owned", "raw"])
def test_receipt_recomputes_measurement_and_rejects_owned_or_raw_tampering(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tamper: str) -> None:
    stage = tmp_path / "case"
    plan_path = save(stage / "plan.json", {"fixture": True})
    collected = save(tmp_path / "collected.json", {"fixture": True})
    raw = save(stage / "raw.json", {"value": 1})
    claim = save(tmp_path / "claim.json", {"role": "case", "request_id": "rid",
                                           "plan_sha256": sha(plan_path)})
    plan = {"probe": "P_value", "role": "case", "source_arm": "F1", "seed": 1,
            "checkpoint": {"sha256": "1" * 64}, "telemetry_sha256": "2" * 64,
            "collected_input": {"path": str(collected), "sha256": sha(collected)},
            "source_manifest_sha256": "3" * 64, "auxiliary_inputs": [],
            "raw_files": {"raw": str(raw)}}
    execution = {"exit_code": 0, "natural_exit": True, "timed_out": False,
                 "error_type": None, "cleanup_verified": True,
                 "job_assigned_before_release": True}
    owned = save(stage / "owned-result.json", execution)
    gpu_before = {"admission_policy": M.GPU.GPU_PROFILE, "busy": False}
    save(stage / "gpu_before.json", gpu_before)
    receipt = {
        "schema": M.P.RECEIPT_SCHEMA, "status": "complete", "error_type": None,
        "role": "case", "probe": "P_value", "source_arm": "F1", "seed": 1,
        "plan": {"path": str(plan_path), "sha256": sha(plan_path)},
        "authorization": {"path": str((tmp_path / "auth.json").resolve()),
                          "sha256": "a" * 64, "request_id": "rid"},
        "checkpoint_sha256": "1" * 64,
        "source_refs": {"collected_input": plan["collected_input"],
                        "source_manifest_sha256": "3" * 64, "auxiliary_inputs": []},
        "execution": execution, "gpu_before": gpu_before,
        "raw_refs": {"raw": {"path": str(raw), "sha256": sha(raw)},
                     "owned_result": {"path": str(owned), "sha256": sha(owned)}},
        "claim": {"path": str(claim), "sha256": sha(claim),
                  "body": json.loads(claim.read_text(encoding="utf-8"))},
        "measurement": {"critic_change": 0.1, "actor_change": 0.5},
    }
    save(stage / "probe_receipt.json", receipt)
    monkeypatch.setattr(M, "_measurement", lambda _plan: deepcopy(receipt["measurement"]))
    M._receipt(receipt, role="case", plan_path=plan_path, plan=plan,
               auth_path=tmp_path / "auth.json", auth_sha="a" * 64, request_id="rid",
               collected_input=collected,
               source_binding={"checkpoint_sha256": "1" * 64, "telemetry_sha256": "2" * 64})
    if tamper == "owned":
        save(owned, {**execution, "cleanup_verified": False})
        reason = "owned"
    else:
        raw.write_text('{"value":2}', encoding="utf-8")
        reason = "raw_refs"
    with pytest.raises(ValueError, match=reason):
        M._receipt(receipt, role="case", plan_path=plan_path, plan=plan,
                   auth_path=tmp_path / "auth.json", auth_sha="a" * 64, request_id="rid",
                   collected_input=collected,
                   source_binding={"checkpoint_sha256": "1" * 64, "telemetry_sha256": "2" * 64})
