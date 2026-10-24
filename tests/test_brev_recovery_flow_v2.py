from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))
import brev_recovery_contract_v2 as P
import brev_recovery_flow_v2 as FLOW
import brev_recovery_model_v2 as MODEL

ANCHOR = "a" * 40


def execution():
    return {"schema": "brev_recovery_posix_execution_v1", "completed": True,
            "exit_code": 0, "timed_out": False, "cleanup_verified": True,
            "gpu_before": [], "gpu_after": [], "source_before": {"same": True},
            "source_after": {"same": True}, "wall_s": 1.0}


class Backend:
    def __init__(self, lab):
        self.calls = []
        self.lab = lab

    def train(self, stage, arm, seed, run_name, corrections=None, *, iterations=300,
              source_checkpoint_sha256=None):
        self.calls.append(("train", arm, iterations))
        stage.mkdir(parents=True)
        checkpoint = stage / "model.pt"
        checkpoint.write_bytes(arm.encode())
        telemetry = stage / "telemetry.json"
        summary = {"Metrics/base_velocity/error_vel_xy": {"mean_last_20pct": 1.5}}
        P.write_new(telemetry, {"summary": summary})
        receipt = execution()
        P.write_new(stage / "execution.json", receipt)
        manifest = stage / "manifest.json"
        if seed == P.DRY_SEED:
            body = {"schema": "brev_recovery_dryrun_manifest_v1", "arm": "NONE", "seed": seed, "iterations": 20}
            readback = {"remaining_faults": []}
        else:
            readback = {"remaining_faults": ["F1"] if arm == "F1" else [], "reward": {
                "before_code_sha256": "a" * 64,
                "after_fault_code_sha256": ("a" if arm == "NONE" else "b") * 64,
                "after_code_sha256": ("b" if arm == "F1" else "a") * 64,
                "fault_present_after": arm == "F1"}}
            readback_path = stage / "readback.json"
            P.write_new(readback_path, readback)
            body = {"arm": "FIRST" if arm == "CORRECTED" else arm, "seed": seed,
                    "max_iterations": 300, "mode": "train", "num_envs": 4096,
                    "original_faults": [] if arm == "NONE" else ["F1"],
                    "applied_corrections": corrections or [], "source_commit_sha": ANCHOR,
                    "protocol_sha256_lf": P.sha(P.PROTOCOL, True), "readback_path": str(readback_path),
                    "source_checkpoint_sha256": source_checkpoint_sha256}
        P.write_new(manifest, body)
        return {"arm": arm, "seed": seed, "checkpoint": {"path": str(checkpoint), "sha256": P.sha(checkpoint)},
                "telemetry": {"path": str(telemetry), "sha256": P.sha(telemetry), "summary": summary},
                "readback": readback, "execution": receipt, "manifest_path": str(manifest)}

    def evaluate(self, stage, artifacts, run_name):
        self.calls.append(("evaluate", tuple(artifacts)))
        stage.mkdir(parents=True)
        result = {"reports": {arm: {"arm": arm} for arm in artifacts}, "execution": execution()}
        P.write_new(stage / "raw.json", result)
        return result

    def probe(self, stage, artifact, probe, run_name):
        self.calls.append(("probe", artifact["arm"], probe))
        stage.mkdir(parents=True)
        value = 0.5 if artifact["arm"] == "F1" else 0.0
        measurement = {"track_lin_vel_xy_exp_rel_error": value,
                       "track_ang_vel_z_exp_rel_error": 0.0, "timeout_ratio": 1.0}
        report = {"name": run_name, "task": FLOW.R.TASK, "seed": artifact["seed"],
                  "checkpoint": {"sha256": artifact["checkpoint"]["sha256"]},
                  "probes": {probe: {"exit": 0, "measurement": measurement}}}
        result = {"measurement": measurement, "report": report, "execution": execution()}
        P.write_new(stage / "raw.json", result)
        return result


@pytest.fixture
def setup(tmp_path, monkeypatch):
    protocol = {"tag": P.TAG, "selection": {"model": MODEL.CALL.MODEL, "max_probes": 7},
                "evaluation": {"bands": "evals/results/p0c_fixed_eval_20261009/summary.json"}}
    protocol_path = tmp_path / "protocol.json"
    P.write_new(protocol_path, protocol)
    monkeypatch.setattr(P, "PROTOCOL", protocol_path)
    monkeypatch.setattr(P, "verify", lambda anchor: protocol if anchor == ANCHOR else None)
    monkeypatch.setattr(FLOW, "_reports", lambda evaluation, artifacts, folder: evaluation["reports"])
    monkeypatch.setattr(FLOW, "validate_report_set", lambda evaluation, artifacts: evaluation["reports"])
    monkeypatch.setattr(FLOW.R, "recovery", lambda report, reference, bands: {
        "outcome": "not_recovered" if report["arm"] == "F1" else "recovered"})
    monkeypatch.setattr(FLOW.R.D1.V.R1.U, "walks", lambda report: True)
    monkeypatch.setattr(FLOW.FRESH, "assert_fresh_seeds", lambda *args: fresh())
    return tmp_path / "run", Backend(tmp_path / "lab"), tmp_path / "local"


def fresh():
    return {"schema": "preregistered_seed_freshness_v1", "status": "fresh", "seeds": [P.SEED, P.DRY_SEED],
            "checked_at_utc": "2026-10-24T00:00:00+00:00", "evidence": [], "checked_file_count": 123,
            "checked_roots": ["repo1", "repo2", "repo3", "repo4", "<isaaclab>/logs"], "scope": "CPU fixture"}


def to_model(folder, backend):
    state = FLOW.advance(folder, backend, ANCHOR, freshness=fresh())
    while state["phase"] != "needs_model":
        state = FLOW.advance(folder, backend, ANCHOR)
    return state


def provider(probe="P_reward", content=None):
    return lambda messages, params: {"content": content or json.dumps({
        "action": "continue", "probe": probe, "fact_ids": ["config:action_scale"]}),
        "provider_model": MODEL.CALL.MODEL, "finish_reason": "stop", "input_tokens": 1,
        "output_tokens": 1, "total_tokens": 2, "wall_s": 0.01}


def model_response(state, local):
    return MODEL.decide(P.read(Path(state["model_request"])), local / "call1", local / "api", ANCHOR,
                        complete=provider())


def test_single_registered_fault_fixture_runs_approval_probe_restart_and_followup(setup):
    folder, backend, local = setup
    state = to_model(folder, backend)
    response = model_response(state, local)
    state = FLOW.advance(folder, backend, ANCHOR, response=response)
    assert state["probes"][0]["outcome"] == "abnormal"
    assert state["probes"][0]["selected_by"] == "model"
    assert not any(call[:2] == ("train", "CORRECTED") for call in backend.calls)
    state = FLOW.advance(folder, backend, ANCHOR)
    assert state["phase"] == "repair"
    state = FLOW.advance(folder, backend, ANCHOR)
    assert state["outcome"] == "recovered" and state["demo_success"] is True
    assert state["pending_ledgers"] == []
    assert ("train", "CORRECTED", 300) in backend.calls
    ledger = FLOW.C.D1.CorrectionLedger(folder / "corrections.ledger.jsonl", FLOW.C.RC.correction_registry())
    assert [row["event"] for row in ledger.events()] == ["proposed", "approved", "consumed", "receipt"]
    assert MODEL.audit(local / "api")["completed"] == 1
    result = P.read(folder / "result.json")
    assert result.pop("final_state_sha256") == P.sha(folder / "states" / f"{state['index']:04d}.json")
    assert result == state


def test_path_rebinding_preserves_view_and_messages_and_rejects_ledger_changes(setup):
    folder, backend, local = setup
    state = to_model(folder, backend)
    envelope = P.read(Path(state["model_request"]))
    snapshot = MODEL.rebind_snapshot(envelope, local / "good")
    assert snapshot["view"] == envelope["snapshot"]["view"]
    assert snapshot["inputs"]["ledger_path"] != envelope["snapshot"]["inputs"]["ledger_path"]
    changed = deepcopy(envelope)
    changed["ledger_events"] = [{"event": "receipt", "request_id": "invented"}]
    with pytest.raises((ValueError, KeyError)):
        MODEL.rebind_snapshot(changed, local / "bad")


@pytest.mark.parametrize("field,value", [("cleanup_verified", False), ("exit_code", 1),
                                         ("timed_out", True), ("gpu_after", [123])])
def test_noncomplete_parent_cannot_authorize_measurement(field, value):
    receipt = execution()
    receipt[field] = value
    with pytest.raises(ValueError):
        FLOW.valid_execution(receipt)


def test_changed_checkpoint_and_raw_probe_cannot_reach_correction(setup):
    folder, backend, local = setup
    state = to_model(folder, backend)
    response = model_response(state, local)
    state = FLOW.advance(folder, backend, ANCHOR, response=response)
    path = next(iter(state["probes"][0]["case"]["inventory"]))
    (folder / path).write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError):
        FLOW.advance(folder, backend, ANCHOR)
    assert not any(call[:2] == ("train", "CORRECTED") for call in backend.calls)


def test_stale_response_and_duplicate_transition_never_launch_probe_twice(setup):
    folder, backend, local = setup
    state = to_model(folder, backend)
    response = model_response(state, local)
    bad = deepcopy(response)
    bad["snapshot_sha256"] = "f" * 64
    with pytest.raises(ValueError):
        FLOW.advance(folder, backend, ANCHOR, response=bad)
    with pytest.raises(FileExistsError):
        FLOW.advance(folder, backend, ANCHOR, response=response)
    assert not any(call[0] == "probe" for call in backend.calls)


def test_valid_hold_is_terminal_without_rule_or_correction(setup):
    folder, backend, local = setup
    state = to_model(folder, backend)
    response = MODEL.decide(P.read(Path(state["model_request"])), local / "hold", local / "api", ANCHOR,
                            complete=provider(content=json.dumps({"action": "hold", "probe": None,
                                                                  "fact_ids": ["config:action_scale"]})))
    state = FLOW.advance(folder, backend, ANCHOR, response=response)
    assert state["outcome"] == "HOLD" and state["probes"] == []
    assert not any(call[0] == "probe" for call in backend.calls)


def test_invalid_model_gets_one_feedback_then_hold(setup):
    folder, backend, local = setup
    state = to_model(folder, backend)
    response = MODEL.decide(P.read(Path(state["model_request"])), local / "invalid", local / "api", ANCHOR,
                            complete=provider(content='{"made_up":true}'))
    assert len(response["safe_results"]) == 2
    state = FLOW.advance(folder, backend, ANCHOR, response=response)
    assert state["outcome"] == "HOLD" and state["probes"] == []


def test_api_pending_reservation_blocks_new_http(setup):
    folder, backend, local = setup
    state = to_model(folder, backend)
    P.write_new(local / "api/reservations/0001.json", {"pending": True})
    calls = []
    with pytest.raises(ValueError, match="미완료 예약"):
        MODEL.decide(P.read(Path(state["model_request"])), local / "blocked", local / "api", ANCHOR,
                     complete=lambda *args: calls.append(args))
    assert calls == []


def test_adaptive_two_roundtrips_have_distinct_snapshots_and_append_http_history(setup):
    folder, backend, local = setup
    state = to_model(folder, backend)
    first = MODEL.decide(P.read(Path(state["model_request"])), local / "first", local / "api", ANCHOR,
                         complete=provider("P_episode"))
    state = FLOW.advance(folder, backend, ANCHOR, response=first)
    assert state["probes"][0]["outcome"] == "normal"
    state = FLOW.advance(folder, backend, ANCHOR)
    assert state["phase"] == "needs_model"
    second = MODEL.decide(P.read(Path(state["model_request"])), local / "second", local / "api", ANCHOR,
                          complete=provider())
    assert first["snapshot_sha256"] != second["snapshot_sha256"]
    state = FLOW.advance(folder, backend, ANCHOR, response=second)
    assert state["api_reserved"] == 2
    assert [row["probe"] for row in state["probes"]] == ["P_episode", "P_reward"]


def test_different_messages_digest_prevents_http(setup):
    folder, backend, local = setup
    state = to_model(folder, backend)
    envelope = P.read(Path(state["model_request"]))
    envelope["messages_digest"] = "e" * 64
    calls = []
    with pytest.raises(ValueError):
        MODEL.decide(envelope, local / "mismatch", local / "api", ANCHOR,
                     complete=lambda *args: calls.append(args))
    assert calls == []


def test_twenty_requests_exhaust_cap_without_request_twenty_one(setup):
    folder, backend, local = setup
    state = to_model(folder, backend)
    root = local / "api"
    for index in range(1, 21):
        path = root / "reservations" / f"{index:04d}.json"
        request = {"index": index}
        P.write_new(path, request)
        P.write_new(root / "responses" / path.name,
                    {"request": request, "reservation_sha256": P.sha(path), "result": {}})
    calls = []
    result = MODEL.decide(P.read(Path(state["model_request"])), local / "limit", root, ANCHOR,
                          complete=lambda *args: calls.append(args))
    assert result["stop_reason"] == "budget_exhausted" and result["safe_results"] == []
    assert calls == [] and MODEL.audit(root)["reserved"] == 20


def test_waiting_for_model_is_read_only_and_valid_response_can_follow(setup):
    folder, backend, local = setup
    state = to_model(folder, backend)
    count = len(list((folder / "intents").glob("*.json")))
    assert FLOW.advance(folder, backend, ANCHOR) == state
    assert len(list((folder / "intents").glob("*.json"))) == count
    after = FLOW.advance(folder, backend, ANCHOR, response=model_response(state, local))
    assert len(after["probes"]) == 1


def test_source_report_tail_tamper_is_rejected(setup):
    folder, backend, _ = setup
    state = to_model(folder, backend)
    path = folder / "states" / f"{state['index']:04d}.json"
    body = P.read(path)
    body["source_reports"]["NONE"]["arm"] = "F1"
    path.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(ValueError, match="정상 참조"):
        FLOW.advance(folder, backend, ANCHOR)


def test_fake_minimal_seed_self_report_is_rejected_before_dry_run(setup):
    folder, backend, _ = setup
    with pytest.raises(ValueError, match="seed guard"):
        FLOW.advance(folder, backend, ANCHOR, freshness={"status": "fresh", "seeds": [P.SEED, P.DRY_SEED]})
    assert backend.calls == []


def test_partial_probe_failure_keeps_known_cost_and_unknown_total(setup):
    folder, backend, local = setup
    state = to_model(folder, backend)
    response = model_response(state, local)
    original = backend.probe

    def fail_case(stage, artifact, probe, run_name):
        if artifact["arm"] == "F1":
            raise ValueError("CPU fixture unknown child cost")
        return original(stage, artifact, probe, run_name)

    backend.probe = fail_case
    with pytest.raises(ValueError):
        FLOW.advance(folder, backend, ANCHOR, response=response)
    cost = P.read(folder / "failed_probe_cost.json")
    assert cost["parent_wall_s"] is None and cost["known_parent_wall_s"] == 1.0
    ledger = FLOW.C.S.M.RegistryLedger(folder / "probes.ledger.jsonl", FLOW.C.RC.registry())
    receipt = next(iter(ledger.requests().values()))["receipt"]
    assert receipt["outcome"] == "unknown" and receipt["gpu_s"] is None


def test_budget_exhaustion_during_feedback_can_be_replayed_without_extra_http(setup):
    folder, backend, local = setup
    state = to_model(folder, backend)
    root = local / "api"
    for index in range(1, 20):
        path = root / "reservations" / f"{index:04d}.json"
        request = {"index": index}
        P.write_new(path, request)
        P.write_new(root / "responses" / path.name,
                    {"request": request, "reservation_sha256": P.sha(path), "result": {}})
    response = MODEL.decide(P.read(Path(state["model_request"])), local / "last_slot", root, ANCHOR,
                            complete=provider(content='{"invalid":true}'))
    assert len(response["safe_results"]) == 1 and response["stop_reason"] == "budget_exhausted"
    snap = FLOW.snapshot(folder, state)
    assert FLOW.replay_budget_stop(snap, response["safe_results"])["engine"] == response["local_engine"]


def finish_fixture(folder, backend, local):
    state = to_model(folder, backend)
    state = FLOW.advance(folder, backend, ANCHOR, response=model_response(state, local))
    state = FLOW.advance(folder, backend, ANCHOR)
    return FLOW.advance(folder, backend, ANCHOR)


@pytest.mark.parametrize("target", ["result", "repair", "correction_ledger"])
def test_terminal_reaudit_rejects_result_repair_and_ledger_tampering(setup, target):
    folder, backend, local = setup
    state = finish_fixture(folder, backend, local)
    assert FLOW.load_state(folder) == state
    path = {"result": folder / "result.json", "repair": folder / "repair.json",
            "correction_ledger": folder / "corrections.ledger.jsonl"}[target]
    if target == "correction_ledger":
        path.write_text("", encoding="utf-8")
    else:
        body = P.read(path)
        body["forged"] = True
        path.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(ValueError):
        FLOW.load_state(folder)
