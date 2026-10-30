from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "src")]
M = importlib.import_module("lineage_diagnosis_epoch_v1")


def save(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gate(tmp_path: Path, *, label: str = "eligible", arm: str = "F1",
         verdict: str = "unhealthy") -> tuple[Path, str]:
    reference_job = "source-job" if arm == "NONE" else "reference-job"
    path = save(tmp_path / "gate.json", {
        "schema": "lineage_evaluation_gate_v1", "status": "complete",
        "cases": [{"case_id": "case-01", "seed": 2301, "arm": arm,
                   "source_job": "source-job", "reference_job": reference_job,
                   "label": label, "fixed_verdict": {"label": verdict},
                   "diagnosis_eligible": label in {"eligible", "out_of_support"} and arm != "NONE"}],
    })
    return path, sha(path)


def envelope(max_decisions: int, execution_budget_s: int) -> dict:
    packet = M.C.S.build_packet(
        epoch=1, applied_corrections=[], prior_receipt=None,
        facts={"telemetry:case:Train/mean_reward": 1.0},
        input_bindings={
            "case": {"checkpoint_sha256": "1" * 64, "telemetry_sha256": "2" * 64},
            "reference": {"checkpoint_sha256": "3" * 64, "telemetry_sha256": "4" * 64},
        }, budget_left=max_decisions, execution_budget_s=execution_budget_s)
    return {
        "schema": M.INPUT.PACKET_SCHEMA, "packet": packet,
        "private_metadata": {
            "case": {"source": {"job_id": "source-job", "seed": 2301, "arm": "F1"}},
            "reference": {"source": {"job_id": "reference-job", "seed": 2301, "arm": "NONE"}},
        },
    }


def install_inputs(monkeypatch: pytest.MonkeyPatch, *, arm: str = "F1") -> None:
    def packet(_case, _reference, *, budget_left, execution_budget_s):
        value = envelope(budget_left, execution_budget_s)
        value["private_metadata"]["case"]["source"]["arm"] = arm
        return value
    monkeypatch.setattr(M.INPUT, "make_selection_packet", packet)


def install_prepare(monkeypatch: pytest.MonkeyPatch) -> None:
    def prepare(_source, probe, *, stage, lab, protocol_path, anchor, role):
        value = {"schema": M.P.SCHEMA, "stage": str(Path(stage).resolve()),
                 "probe": probe, "role": role, "lab": str(lab),
                 "protocol_path": str(protocol_path), "anchor": anchor}
        save(Path(stage) / "plan.json", value)
        return value
    monkeypatch.setattr(M.P, "prepare_probe", prepare)


def measurement(probe: str, abnormal: bool) -> dict:
    if probe == M.C.RC.NEW_PROBE:
        return {"status": "abnormal" if abnormal else "normal"}
    if probe == "P_value":
        return {"critic_change": 0.01 if abnormal else 1.0, "actor_change": 0.5}
    if probe == "P_episode":
        return {"timeout_ratio": 0.1 if abnormal else 1.0}
    if probe == "P_physics":
        return {"mismatch_count": 1 if abnormal else 0}
    if probe == "P_noise":
        return {"noise_ratio": 4.0 if abnormal else 1.0}
    if probe == "P_reward":
        value = 1.0 if abnormal else 0.0
        return {"track_lin_vel_xy_exp_rel_error": value,
                "track_ang_vel_z_exp_rel_error": value}
    if probe == "P_torque":
        return {"low_speed_saturation": 1.0 if abnormal else 0.0}
    return {"timeout_ratio": 1.0}


def executor(*, abnormal: bool = True, failed: bool = False):
    def run(plan_path: Path, _authorization: Path) -> dict:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if failed:
            return {"status": "unknown", "role": plan["role"], "measurement": None,
                    "execution": {}, "error_type": "FixtureFailure"}
        return {"status": "complete", "role": plan["role"],
                "measurement": measurement(plan["probe"], abnormal if plan["role"] == "case" else False),
                "execution": {"wall_s": 0.25}, "error_type": None}
    return run


def call(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, policy: str = "registry_next", **changes):
    install_prepare(monkeypatch)
    gate_value = changes.pop("gate", None)
    gate_path, gate_sha = gate(tmp_path) if gate_value is None else gate_value
    gate_row = json.loads(gate_path.read_text(encoding="utf-8"))["cases"][0]
    install_inputs(monkeypatch, arm=gate_row["arm"])
    values = {"out": tmp_path / "out", "case_input": tmp_path / "case.json",
              "reference_input": tmp_path / "reference.json", "policy": policy,
              "run_id": "epoch_fixture", "lab": tmp_path / "lab",
              "probe_protocol": tmp_path / "protocol.json", "anchor": "a" * 40,
              "gate_path": gate_path, "gate_sha256": gate_sha, "case_id": "case-01",
              "probe_executor": executor(), "max_decisions": 3}
    values.update(changes)
    return M.run_epoch(**values)


def test_gate_stop_and_none_host_anomaly_run_no_model_or_probe(tmp_path, monkeypatch):
    stopped = call(tmp_path / "stopped", monkeypatch,
                   gate=gate(tmp_path / "stopped", label="baseline_invalid"))
    assert stopped["end_state"] == "HOLD"
    assert stopped["hold_reason"] == "source_gate_baseline_invalid"
    assert stopped["requests"] == stopped["decision_count"] == 0

    install_inputs(monkeypatch)
    gate_path, gate_sha = gate(tmp_path / "none", arm="NONE", verdict="healthy")
    env = envelope(3, 1200)
    env["private_metadata"]["case"]["source"].update(job_id="source-job", arm="NONE")
    env["private_metadata"]["reference"]["source"]["job_id"] = "source-job"
    env["packet"]["input_bindings"]["reference"] = dict(env["packet"]["input_bindings"]["case"])
    monkeypatch.setattr(M.INPUT, "make_selection_packet", lambda *_args, **_kw: env)
    result = M.run_epoch(tmp_path / "none/out", tmp_path / "case", tmp_path / "ref",
                         "model_adaptive", run_id="none_fixture", lab=tmp_path / "lab",
                         probe_protocol=tmp_path / "protocol", anchor="a" * 40,
                         gate_path=gate_path, gate_sha256=gate_sha, case_id="case-01")
    assert result["end_state"] == "no_intervention"
    assert result["hold_reason"] == "host_anomaly_gate"
    assert result["requests"] == result["decision_count"] == 0


def test_single_confirming_probe_uses_paired_execution_and_one_approval(tmp_path, monkeypatch):
    result = call(tmp_path, monkeypatch)
    assert result["end_state"] == "confirmed"
    assert result["confirmed_subtype"] == "optimizer_value"
    assert result["probe_order"] == ["P_value"]
    assert result["decision_count"] == 1
    step = result["steps"][0]
    assert set(step["paired_receipts"]) == {"case", "reference"}
    authorization = json.loads(Path(step["authorization_path"]).read_text(encoding="utf-8"))
    assert set(authorization) == {"schema", "plan_sha256", "snapshot", "choice",
                                  "request_spec", "ledger_path", "request_id"}
    events = M.C.S.M.RegistryLedger(tmp_path / "out/probes.ledger.jsonl", M.C.RC.registry()).events()
    assert [row["event"] for row in events] == ["proposed", "approved", "consumed", "receipt"]


def test_failure_is_unknown_once_without_zero_cost_masquerade(tmp_path, monkeypatch):
    result = call(tmp_path, monkeypatch, probe_executor=executor(failed=True))
    assert result["end_state"] == "HOLD"
    assert result["hold_reason"] == "probe_outcome_unknown"
    assert result["steps"][0]["outcome"] == "unknown"
    receipt = result["steps"][0]["ledger_receipt"]
    assert receipt["gpu_s"] > 0
    assert result["wall"]["probe_child_unknown_roles"] == ["reference", "case"]
    events = M.C.S.M.RegistryLedger(tmp_path / "out/probes.ledger.jsonl", M.C.RC.registry()).events()
    assert sum(row["event"] == "receipt" for row in events) == 1


def test_fixed_budget_limit_and_exhausted_information_are_distinct(tmp_path, monkeypatch):
    limited = call(tmp_path / "limited", monkeypatch, probe_executor=executor(abnormal=False))
    assert limited["decision_count"] == 3
    assert limited["hold_reason"] == "decision_limit"

    exhausted = call(tmp_path / "exhausted", monkeypatch, probe_executor=executor(abnormal=False),
                     max_decisions=7)
    assert exhausted["decision_count"] <= 7
    assert exhausted["hold_reason"] == "no_informative_probe"


def model_factory(content: str):
    def factory(_root, _slot, _context, _view, _timeout):
        def complete(_messages, _params):
            return {"provider_model": M.C.CALL.MODEL, "finish_reason": "stop",
                    "content": content, "input_tokens": 10, "output_tokens": 5,
                    "total_tokens": 15, "wall_s": 0.1}
        return complete
    return factory


def test_model_engine_valid_and_invalid_are_revalidated_without_rule_fallback(tmp_path, monkeypatch):
    valid = json.dumps({"action": "continue", "probe": "P_value",
                        "fact_ids": ["telemetry:case:Train/mean_reward"]})
    result = call(tmp_path / "valid", monkeypatch, "model_adaptive",
                  transport_root=tmp_path / "transport", slot_ids=("slot-1",),
                  complete_factory=model_factory(valid))
    assert result["selection_actor"] == ["model"]
    assert result["model_validity"] == ["valid"]
    assert result["requests"] == 1 and result["tokens"]["total_tokens"] == 15
    assert result["tokens"]["known_subtotal"]["total_tokens"] == 15
    assert result["wall"]["model_wall_s"] == 0.1
    assert result["model_observation_use"] == [{
        "decision": 1, "observation_fact_id_count": 0,
        "immediately_prior_receipt_prefix": None,
        "cited_immediately_prior_receipt": None,
    }]
    assert result["scope"]["observation_citation_measurement"] == "This does not establish causality."
    assert result["evidence_scope"] == "cpu_fixture"

    invalid = call(tmp_path / "invalid", monkeypatch, "model_adaptive",
                   transport_root=tmp_path / "transport", slot_ids=("slot-2",),
                   complete_factory=model_factory("{}"))
    assert invalid["end_state"] == "HOLD"
    assert invalid["decision_count"] == 0
    assert invalid["selection_actor"] == ["hold"]
    assert invalid["model_validity"] == ["invalid"]
    assert invalid["hold_reason"] == "model_invalid_hold"


def test_unknown_model_cost_then_valid_next_decision_preserves_subtotals_and_observation_use(
        tmp_path, monkeypatch):
    calls = {}

    def factory(_root, slot, _context, _view, _timeout):
        calls[slot] = 0

        def complete(messages, _params):
            calls[slot] += 1
            view = json.loads(messages[1]["content"])
            probe = view["eligible_probes"][0]
            observation_ids = [key for key in view["evidence_facts"] if key.startswith("observation:")]
            fact_id = (observation_ids[0] if observation_ids
                       else "telemetry:case:Train/mean_reward")
            content = json.dumps({"action": "continue", "probe": probe, "fact_ids": [fact_id]})
            if slot == "slot-1" and calls[slot] == 1:
                return {"infra_error": "TimeoutError"}
            return {"provider_model": M.C.CALL.MODEL, "finish_reason": "stop",
                    "content": content, "input_tokens": 10, "output_tokens": 5,
                    "total_tokens": 15, "wall_s": 0.1}
        return complete

    result = call(tmp_path, monkeypatch, "model_adaptive",
                  transport_root=tmp_path / "transport", slot_ids=("slot-1", "slot-2"),
                  complete_factory=factory, probe_executor=executor(abnormal=False), max_decisions=2)
    assert result["decision_count"] == 2
    assert result["requests"] == 3
    assert result["tokens"]["total_tokens"] is None
    assert result["tokens"]["known_subtotal"]["total_tokens"] == 30
    assert result["tokens"]["unknown_cost_requests"]["total_tokens"] == [
        {"decision": 1, "request": 1}]
    assert result["wall"]["model_wall_s"] is None
    assert result["wall"]["known_model_wall_subtotal_s"] == pytest.approx(0.2)
    assert result["wall"]["unknown_model_cost_requests"] == [{"decision": 1, "request": 1}]
    assert result["model_observation_use"][0]["cited_immediately_prior_receipt"] is None
    assert result["model_observation_use"][1]["observation_fact_id_count"] == 1
    assert result["model_observation_use"][1]["cited_immediately_prior_receipt"] is True


def test_out_of_support_diagnoses_but_never_enters_repair(tmp_path, monkeypatch):
    result = call(tmp_path, monkeypatch, gate=gate(tmp_path, label="out_of_support", arm="F3"))
    assert result["decision_count"] == 1
    assert result["confirmed_subtype"] == "optimizer_value"
    assert result["end_state"] == "HOLD"
    assert result["hold_reason"] == "repair_unsupported"
    assert result["repair_supported"] is False
    assert result["correction_executed"] is result["training_executed"] is False


def test_three_policies_start_from_identical_public_view(tmp_path, monkeypatch):
    digests = []
    valid = json.dumps({"action": "continue", "probe": "P_value",
                        "fact_ids": ["telemetry:case:Train/mean_reward"]})
    for policy in M.POLICIES:
        kwargs = ({"transport_root": tmp_path / "transport", "slot_ids": ("slot",),
                   "complete_factory": model_factory(valid)} if policy == "model_adaptive" else {})
        result = call(tmp_path / policy, monkeypatch, policy, **kwargs)
        auth = json.loads(Path(result["steps"][0]["authorization_path"]).read_text(encoding="utf-8"))
        digests.append(auth["snapshot"]["view_digest"])
        assert result["max_decisions"] == 3
    assert len(set(digests)) == 1


def test_budget_range_output_reuse_and_interrupted_pending_refuse_reexecution(tmp_path, monkeypatch):
    install_inputs(monkeypatch)
    install_prepare(monkeypatch)
    gate_path, gate_sha = gate(tmp_path)
    common = {"case_input": tmp_path / "case", "reference_input": tmp_path / "ref",
              "policy": "registry_next", "run_id": "pending_fixture", "lab": tmp_path / "lab",
              "probe_protocol": tmp_path / "protocol", "anchor": "a" * 40,
              "gate_path": gate_path, "gate_sha256": gate_sha, "case_id": "case-01"}
    with pytest.raises(ValueError, match="1..7"):
        M.run_epoch(tmp_path / "bad", max_decisions=0, **common)

    def interrupted(_plan, _authorization):
        raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        M.run_epoch(tmp_path / "pending", probe_executor=interrupted, **common)
    ledger = M.C.S.M.RegistryLedger(tmp_path / "pending/probes.ledger.jsonl", M.C.RC.registry())
    assert next(iter(ledger.requests().values()))["state"] == "running"
    with pytest.raises(FileExistsError):
        M.run_epoch(tmp_path / "pending", probe_executor=executor(), **common)

    result = M.run_epoch(tmp_path / "fresh", probe_executor=executor(), **common)
    assert result["status"] == "complete"
    with pytest.raises(FileExistsError):
        M.run_epoch(tmp_path / "fresh", probe_executor=executor(), **common)
