"""새 active driver의 승인 결속·HOLD·비용 경계를 GPU 없이 검증한다."""
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from rl_triage import probe_loop as L


ROOT = Path(__file__).resolve().parents[1]


def write_stage_outcome(driver, folder, stage, packet, status, decision=None, error_type=None):
    packet_path = folder / "model" / f"{stage}.packet.json"
    driver.write_new(packet_path, packet)
    decision_sha = None
    if decision is not None:
        decision_path = folder / "model" / f"{stage}.decision.json"
        driver.write_new(decision_path, decision)
        decision_sha = driver.sha(decision_path)
    error = None if error_type is None else {"type": error_type, "message": error_type}
    driver.write_new(folder / "model" / f"{stage}.outcome.json", {
        "stage": stage, "status": status, "packet_sha256_lf": driver.sha(packet_path),
        "attempts": [], "decision": decision, "decision_sha256_lf": decision_sha, "error": error})


@pytest.fixture
def driver(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("active_driver_test", ROOT / "evals/run_active_diagnosis.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    return module


def test_wsl_path_rejects_non_windows_paths(driver):
    assert driver.wsl_path(ROOT / "evals/active_model_call.py").endswith("/evals/active_model_call.py")


def test_semantic_model_error_is_not_retried(driver, tmp_path, monkeypatch):
    folder = tmp_path / "case"
    calls = []

    def worker(args, **kwargs):
        calls.append(args)
        output = folder / "model/initial.attempt1.json"
        packet = folder / "model/initial.packet.json"
        import hashlib
        driver.write_new(output, {"packet_sha256": hashlib.sha256(packet.read_bytes()).hexdigest(),
                                  "messages_sha256": driver.L.digest(driver.D.decision_messages(
                                      {"facts": {"x": 1}, "observed": [], "budget_left": 2})),
                                  "content": '{"ranking": ["fake"]}', "exit_code": 0})
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(driver.subprocess, "run", worker)
    with pytest.raises(ValueError, match="decision 필드"):
        driver.fresh_decision(folder, "initial", {"facts": {"x": 1}, "observed": [], "budget_left": 2},
                              {"model": "model", "max_infra_attempts": 2})
    assert len(calls) == 1
    assert not (folder / "model/initial.decision.json").exists()
    outcome = driver.read(folder / "model/initial.outcome.json")
    assert outcome["status"] == "semantic_invalid" and len(outcome["attempts"]) == 1
    assert outcome["decision"] is None and outcome["decision_sha256_lf"] is None


def test_messages_hash_mismatch_is_harness_fatal_without_retry(driver, tmp_path, monkeypatch):
    folder = tmp_path / "case"
    calls = []

    def worker(args, **kwargs):
        calls.append(args)
        output = folder / "model/initial.attempt1.json"
        packet = folder / "model/initial.packet.json"
        import hashlib
        driver.write_new(output, {"packet_sha256": hashlib.sha256(packet.read_bytes()).hexdigest(),
                                  "messages_sha256": "0" * 64, "content": "{}", "exit_code": 0})
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(driver.subprocess, "run", worker)
    with pytest.raises(driver.ModelHarnessError, match="해시 불일치"):
        driver.fresh_decision(folder, "initial", {"facts": {"x": 1}, "observed": [], "budget_left": 2},
                              {"model": "model", "max_infra_attempts": 2})
    assert len(calls) == 1
    assert driver.read(folder / "model/initial.outcome.json")["status"] == "harness_fatal"


def test_timeout_cleanup_failure_is_harness_fatal_without_retry(driver, tmp_path, monkeypatch):
    folder = tmp_path / "case"
    calls = []

    def timeout(*args, **kwargs):
        calls.append(args)
        raise driver.subprocess.TimeoutExpired("worker", 330)

    monkeypatch.setattr(driver.subprocess, "run", timeout)
    monkeypatch.setattr(driver, "stop_model_worker",
                        lambda *args: (_ for _ in ()).throw(RuntimeError("cleanup failed")))
    with pytest.raises(driver.ModelHarnessError, match="cleanup 실패"):
        driver.fresh_decision(folder, "initial", {"facts": {"x": 1}, "observed": [], "budget_left": 2},
                              {"model": "model", "max_infra_attempts": 2})
    assert len(calls) == 1
    outcome = driver.read(folder / "model/initial.outcome.json")
    assert outcome["status"] == "harness_fatal" and len(outcome["attempts"]) == 1


@pytest.mark.parametrize("error_class", [OSError, subprocess.SubprocessError])
def test_worker_launch_failure_writes_safe_attempt_and_harness_outcome(driver, tmp_path, monkeypatch, error_class):
    folder = tmp_path / "case"
    calls = []

    def launch_failure(*args, **kwargs):
        calls.append(args)
        raise error_class("secret-token-must-not-be-recorded")

    monkeypatch.setattr(driver.subprocess, "run", launch_failure)
    with pytest.raises(driver.ModelHarnessError, match=error_class.__name__):
        driver.fresh_decision(folder, "initial", {"facts": {"x": 1}, "observed": [], "budget_left": 2},
                              {"model": "model", "max_infra_attempts": 2})
    attempt = driver.read(folder / "model/initial.attempt1.json")
    outcome = driver.read(folder / "model/initial.outcome.json")
    serialized = json.dumps({"attempt": attempt, "outcome": outcome})
    assert len(calls) == 1 and attempt["error_type"] == error_class.__name__
    assert attempt["error_message"] == "model worker launch failed" and attempt["retryable"] is False
    assert outcome["status"] == "harness_fatal" and len(outcome["attempts"]) == 1
    assert "secret-token" not in serialized


def test_non_object_attempt_is_harness_fatal(driver, tmp_path, monkeypatch):
    folder = tmp_path / "case"

    def worker(args, **kwargs):
        driver.write_new(folder / "model/initial.attempt1.json", ["not", "an", "object"])
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(driver.subprocess, "run", worker)
    with pytest.raises(driver.ModelHarnessError, match="non-object"):
        driver.fresh_decision(folder, "initial", {"facts": {"x": 1}, "observed": [], "budget_left": 2},
                              {"model": "model", "max_infra_attempts": 2})
    assert driver.read(folder / "model/initial.outcome.json")["status"] == "harness_fatal"


def test_infrastructure_errors_use_exactly_two_attempts(driver, tmp_path, monkeypatch):
    folder = tmp_path / "case"
    calls = []

    def worker(args, **kwargs):
        calls.append(args)
        attempt = len(calls)
        output = folder / f"model/initial.attempt{attempt}.json"
        packet = folder / "model/initial.packet.json"
        import hashlib
        driver.write_new(output, {"packet_sha256": hashlib.sha256(packet.read_bytes()).hexdigest(),
                                  "messages_sha256": driver.L.digest(driver.D.decision_messages(
                                      {"facts": {"x": 1}, "observed": [], "budget_left": 2})),
                                  "infra_error": "APITimeoutError", "error_type": "APITimeoutError",
                                  "worker_cleanup_failed": False, "exit_code": 2})
        return SimpleNamespace(returncode=2)

    monkeypatch.setattr(driver.subprocess, "run", worker)
    monkeypatch.setattr(driver.time, "sleep", lambda *_: None)
    with pytest.raises(driver.ModelInfraError, match="한도 도달"):
        driver.fresh_decision(folder, "initial", {"facts": {"x": 1}, "observed": [], "budget_left": 2},
                              {"model": "model", "max_infra_attempts": 2})
    outcome = driver.read(folder / "model/initial.outcome.json")
    assert len(calls) == 2 and outcome["status"] == "infra_exhausted"
    assert [item["path"] for item in outcome["attempts"]] == [
        "model/initial.attempt1.json", "model/initial.attempt2.json"]


def test_stage_pair_masks_only_latest_measurement_and_alternates_order(driver, tmp_path, monkeypatch):
    first = {"request_id": "old", "probe": "P_reward", "outcome": "normal", "measurement": {"x": 1}}
    last = {"request_id": "new", "probe": "P_noise", "outcome": "normal", "measurement": {"x": 2}}
    packet = {"phase": "after_receipt", "facts": {"telemetry:x": 5, "receipt:old:x": 1, "receipt:new:x": 2},
              "budget_left": 0, "observed": [first, last], "previous_decision": {"ranking": list(L.MECHANISMS)}}
    called = []
    decision = {"ranking": list(L.MECHANISMS), "action": "hold", "claims": []}

    def fake(folder, stage, current, protocol):
        called.append((stage, current))
        packet_path = folder / "model" / f"{stage}.packet.json"
        driver.write_new(packet_path, current)
        driver.write_new(folder / "model" / f"{stage}.decision.json", decision)
        driver.write_new(folder / "model" / f"{stage}.outcome.json", {
            "stage": stage, "status": "valid", "packet_sha256_lf": driver.sha(packet_path),
            "attempts": [], "decision": decision,
            "decision_sha256_lf": driver.sha(folder / "model" / f"{stage}.decision.json"), "error": None})
        return decision, []

    monkeypatch.setattr(driver, "fresh_decision", fake)
    actual_decision, control_decision, comparison, pair = driver.stage_pair(tmp_path, 0, decision, packet, {})
    actual, masked = [c[1] for c in called]
    assert actual == packet
    assert masked == {**packet, "observed": [first], "facts": {"telemetry:x": 5, "receipt:old:x": 1}}
    assert pair["proof_eligible"] is True
    assert comparison == driver.D.contrast(decision, actual_decision, control_decision, packet["observed"])
    called.clear()
    second = tmp_path / "second"
    driver.stage_pair(second, 1, decision, packet, {})
    assert called[0][0] == "without_receipt_2"
    assert called[1][0] == "after_2"


def test_existing_result_is_never_overwritten(driver, tmp_path):
    target = tmp_path / "result.json"
    driver.write_new(target, {"value": 1})
    with pytest.raises(FileExistsError):
        driver.write_new(target, {"value": 2})
    assert json.loads(target.read_text()) == {"value": 1}


def test_missing_model_worker_artifact_keeps_one_attempt_with_unknown_cost(driver, tmp_path, monkeypatch):
    folder = tmp_path / "case"
    monkeypatch.setattr(driver.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=1))
    with pytest.raises(RuntimeError, match="ModelWorkerMissingArtifact"):
        driver.fresh_decision(folder, "initial", {"facts": {"x": 1}, "observed": [], "budget_left": 2},
                              {"model": "model", "max_infra_attempts": 2})
    outputs = list((folder / "model").glob("*.attempt*.json"))
    assert len(outputs) == 1
    record = driver.read(outputs[0])
    assert record["cpu_s"] is None and record["input_tokens"] is None
    assert record["response_id"] is None and record["finish_reason"] is None
    assert record["execution_wall_s"] >= 0
    assert record["retryable"] is False


def test_approval_envelope_binds_model_source_and_previous_receipt(driver, tmp_path, monkeypatch):
    folder = tmp_path / "cases/h01_s2027"
    reference = tmp_path / "reference.json"
    driver.write_new(reference, {"probes": {"P_reward": {"measurement": {}}}})
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    import hashlib
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    spec = {"case": "h01_s2027", "checkpoint_sha256": checkpoint_sha,
            "sources": {"probe_reference": {"path": "reference.json", "sha256_lf": driver.sha(reference)}}}
    ledger = L.Ledger(folder / "ledger.jsonl")
    binding = {"protocol_sha256_lf": "a" * 64, "decision_sha256_lf": "b" * 64}
    gpu_calls = []

    def ready_after_busy():
        assert ledger.events() == []
        gpu_calls.append(len(gpu_calls))
        if len(gpu_calls) == 1:
            raise driver.GPUWaitError({"at": "busy", "ready": False, "busy_reasons": ["isaac_process"]})
        return {"at": "ready", "ready": True, "busy_reasons": []}

    monkeypatch.setattr(driver, "gpu_snapshot", ready_after_busy)
    monkeypatch.setattr(driver.time, "sleep", lambda *_: None)
    monkeypatch.setattr(driver, "find_run", lambda *args: tmp_path)
    monkeypatch.setattr(driver, "last_checkpoint", lambda *args: checkpoint)
    monkeypatch.setattr(driver.loop, "isaac_preflight", lambda *args: None)

    def execute(case, probe, tag, **kwargs):
        report = tmp_path / "evals/results/tag/probes/h01_s2027__P_reward.json"
        driver.write_new(report, {"checkpoint": {"sha256": checkpoint_sha}})
        execution = tmp_path / "evals/results/tag/execution/h01_s2027__P_reward.json"
        driver.write_new(execution, {"gpu_wall_s": 5.7})
        return {"track_lin_vel_xy_exp_rel_error": 0.4, "track_ang_vel_z_exp_rel_error": 0.0}, 5.678, 0

    monkeypatch.setattr(driver.loop, "isaac_execute", execute)
    monkeypatch.setattr(driver.loop, "file_artifact", lambda *args: {"checkpoint_sha256": checkpoint_sha})
    receipt = driver.approved_probe(folder, "tag", spec, {"max_probes": 2, "probe_execution_budget_s": 180},
                                    {"observed": [{"receipt_digest": "c" * 64}]}, "P_reward", "1", ledger, binding)
    request = ledger.requests()[receipt["request_id"]]
    envelope = driver.read(folder / "approvals" / f"{receipt['request_id']}.json")
    assert envelope["binding"] == binding
    assert envelope["prior_receipt_digests"] == ["c" * 64]
    assert request["prereg_digest"] == L.digest(envelope)
    assert [e["event"] for e in ledger.events()] == ["proposed", "approved", "consumed", "receipt"]
    assert receipt["receipt_digest"] == L.digest(request["receipt"])
    assert receipt["gpu_s"] == 5.7
    assert len(gpu_calls) == 2


def test_initial_model_hold_preserves_remaining_budget(driver, tmp_path, monkeypatch):
    protocol = {"max_probes": 2, "scope": "integration", "cases": [{"case": "h01_s2027", "checkpoint_sha256": "a" * 64}]}
    path = tmp_path / "protocol.json"
    driver.write_new(path, protocol)
    monkeypatch.setattr(driver, "verify_protocol", lambda *args: None)
    monkeypatch.setattr(driver, "input_packet", lambda *args: {"facts": {"x": 1}, "observed": [], "budget_left": 2})
    hold = {"ranking": list(L.MECHANISMS), "probe": None, "action": "hold", "reason": "모호한 텔레메트리",
            "claims": [{"id": "x", "value": 1}], "expected_outcomes": {}}
    def fake_fresh(folder, stage, packet, protocol):
        packet_path = folder / "model" / f"{stage}.packet.json"
        driver.write_new(packet_path, packet)
        driver.write_new(folder / "model" / f"{stage}.decision.json", hold)
        driver.write_new(folder / "model" / f"{stage}.outcome.json", {
            "stage": stage, "status": "valid", "packet_sha256_lf": driver.sha(packet_path),
            "attempts": [], "decision": hold,
            "decision_sha256_lf": driver.sha(folder / "model" / f"{stage}.decision.json"), "error": None})
        return hold, []
    monkeypatch.setattr(driver, "fresh_decision", fake_fresh)
    result = driver.run_case(path, "hold_test", "h01_s2027")
    assert result["status"] == "hold" and result["stop_reason"] == "model_hold"
    assert result["budget_left"] == 2 and result["evidence_state"]["status"] == "open"
    assert result["observed"] == [] and result["costs"]["isaac_wall_s"] == 0
    assert result["initial_outcome"]["status"] == "valid" and result["pairs"] == []


@pytest.mark.parametrize("index", [0, 1])
def test_control_semantic_invalid_never_blocks_actual_in_either_order(driver, tmp_path, monkeypatch, index):
    observed = {"request_id": "new", "probe": "P_reward", "outcome": "abnormal",
                "measurement": {"relative_error": 0.4}}
    packet = {"facts": {"x": 1, "receipt:new:relative_error": 0.4}, "observed": [observed],
              "budget_left": 1, "previous_decision": {"ranking": list(L.MECHANISMS)}}
    before = {"ranking": list(L.MECHANISMS), "action": "continue", "claims": []}
    actual = {"ranking": list(reversed(L.MECHANISMS)), "action": "hold", "claims": []}
    calls = []

    def fake(folder, stage, current, protocol):
        calls.append(stage)
        if stage.startswith("without_receipt"):
            write_stage_outcome(driver, folder, stage, current, "semantic_invalid",
                                error_type="ModelSemanticError")
            raise driver.ModelSemanticError("invalid control")
        write_stage_outcome(driver, folder, stage, current, "valid", decision=actual)
        return actual, []

    monkeypatch.setattr(driver, "fresh_decision", fake)
    got_actual, got_control, comparison, pair = driver.stage_pair(tmp_path, index, before, packet, {})
    assert len(calls) == 2 and got_actual == actual and got_control is None
    assert comparison is None and pair["proof_eligible"] is False
    assert pair["actual"]["status"] == "valid" and pair["control"]["status"] == "semantic_invalid"


def test_actual_semantic_invalid_still_records_control_then_holds_product_decision(driver, tmp_path, monkeypatch):
    observed = {"request_id": "new", "probe": "P_reward", "outcome": "abnormal", "measurement": {}}
    packet = {"facts": {"x": 1}, "observed": [observed], "budget_left": 1,
              "previous_decision": {"ranking": list(L.MECHANISMS)}}
    before = {"ranking": list(L.MECHANISMS), "action": "continue", "claims": []}
    control = {"ranking": list(L.MECHANISMS), "action": "hold", "claims": []}

    def fake(folder, stage, current, protocol):
        if stage.startswith("after_"):
            write_stage_outcome(driver, folder, stage, current, "semantic_invalid",
                                error_type="ModelSemanticError")
            raise driver.ModelSemanticError("invalid actual")
        write_stage_outcome(driver, folder, stage, current, "valid", decision=control)
        return control, []

    monkeypatch.setattr(driver, "fresh_decision", fake)
    actual, got_control, comparison, pair = driver.stage_pair(tmp_path, 0, before, packet, {})
    assert actual is None and got_control == control and comparison is None
    assert pair["actual"]["status"] == "semantic_invalid" and pair["control"]["status"] == "valid"


def test_control_infra_exhaustion_still_calls_actual(driver, tmp_path, monkeypatch):
    observed = {"request_id": "new", "probe": "P_reward", "outcome": "abnormal", "measurement": {}}
    packet = {"facts": {"x": 1}, "observed": [observed], "budget_left": 1}
    decision = {"ranking": list(L.MECHANISMS), "action": "hold", "claims": []}
    calls = []

    def fake(folder, stage, current, protocol):
        calls.append(stage)
        if stage.startswith("without_receipt"):
            write_stage_outcome(driver, folder, stage, current, "infra_exhausted",
                                error_type="ModelInfraError")
            raise driver.ModelInfraError("infra")
        write_stage_outcome(driver, folder, stage, current, "valid", decision=decision)
        return decision, []

    monkeypatch.setattr(driver, "fresh_decision", fake)
    actual, _, _, pair = driver.stage_pair(tmp_path, 0, decision, packet, {})
    assert calls == ["without_receipt_1", "after_1"] and actual == decision
    assert pair["control"]["status"] == "infra_exhausted"


def test_harness_fatal_stops_followup_arm_but_writes_pair(driver, tmp_path, monkeypatch):
    observed = {"request_id": "new", "probe": "P_reward", "outcome": "abnormal", "measurement": {}}
    packet = {"facts": {"x": 1}, "observed": [observed], "budget_left": 1}
    calls = []

    def fake(folder, stage, current, protocol):
        calls.append(stage)
        write_stage_outcome(driver, folder, stage, current, "harness_fatal",
                            error_type="ModelHarnessError")
        raise driver.ModelHarnessError("hash mismatch")

    monkeypatch.setattr(driver, "fresh_decision", fake)
    with pytest.raises(driver.ModelHarnessError, match="hash mismatch"):
        driver.stage_pair(tmp_path, 0, {"ranking": list(L.MECHANISMS)}, packet, {})
    pair = driver.read(tmp_path / "model/pair_1.json")
    assert calls == ["without_receipt_1"] and pair["control"]["status"] == "harness_fatal"
    assert pair["actual"] is None and pair["proof_eligible"] is False


def test_gpu_wait_records_busy_then_ready_without_touching_external_process(driver, tmp_path, monkeypatch):
    busy = {"at": "busy", "ready": False, "other_isaac_pids": [94880],
            "busy_reasons": ["isaac_process"]}
    ready = {"at": "ready", "ready": True, "other_isaac_pids": [], "busy_reasons": []}
    snapshots = iter([driver.GPUWaitError(busy), ready])

    def fake_snapshot():
        item = next(snapshots)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(driver, "gpu_snapshot", fake_snapshot)
    monkeypatch.setattr(driver.time, "sleep", lambda *_: None)
    assert driver.wait_for_gpu(tmp_path, "1") == ready
    lines = [json.loads(line) for line in (tmp_path / "gpu_wait_1.jsonl").read_text().splitlines()]
    assert lines == [busy, ready]


def test_initial_semantic_invalid_is_hold_and_attempt_cost_is_preserved(driver, tmp_path, monkeypatch):
    protocol = {"max_probes": 2, "scope": "integration",
                "cases": [{"case": "h02_s2027", "checkpoint_sha256": "a" * 64}]}
    path = tmp_path / "protocol.json"
    driver.write_new(path, protocol)
    monkeypatch.setattr(driver, "verify_protocol", lambda *args: None)
    monkeypatch.setattr(driver, "input_packet", lambda *args: {"facts": {"x": 1}, "observed": [], "budget_left": 2})

    def invalid(folder, stage, packet, protocol):
        packet_path = folder / "model" / f"{stage}.packet.json"
        attempt_path = folder / "model" / f"{stage}.attempt1.json"
        driver.write_new(packet_path, packet)
        driver.write_new(attempt_path, {"input_tokens": 11, "output_tokens": 3, "execution_wall_s": 0.5,
                                        "cpu_s": 0.1})
        driver.write_new(folder / "model" / f"{stage}.outcome.json", {
            "stage": stage, "status": "semantic_invalid", "packet_sha256_lf": driver.sha(packet_path),
            "attempts": [{"path": f"model/{stage}.attempt1.json", "sha256_lf": driver.sha(attempt_path)}],
            "decision": None, "decision_sha256_lf": None,
            "error": {"type": "ModelSemanticError", "message": "invalid"}})
        raise driver.ModelSemanticError("invalid")

    monkeypatch.setattr(driver, "fresh_decision", invalid)
    result = driver.run_case(path, "invalid_initial", "h02_s2027")
    assert result["status"] == "hold" and result["stop_reason"] == "initial_model_invalid"
    assert result["initial_outcome"]["status"] == "semantic_invalid" and result["decisions"] == []
    assert result["costs"]["model_calls"] == 1 and result["costs"]["input_tokens"] == 11


def test_pair_write_failure_uses_in_memory_pair_and_preserves_hold_costs(driver, tmp_path, monkeypatch):
    protocol = {"max_probes": 1, "scope": "integration", "probe_execution_budget_s": 180,
                "cases": [{"case": "h01_s2027", "checkpoint_sha256": "a" * 64}]}
    protocol_path = tmp_path / "protocol.json"
    driver.write_new(protocol_path, protocol)
    monkeypatch.setattr(driver, "verify_protocol", lambda *args: None)
    monkeypatch.setattr(driver, "input_packet", lambda *args: {
        "facts": {"x": 1}, "observed": [], "budget_left": 1})
    ranking = list(L.MECHANISMS)
    initial = {"ranking": ranking, "probe": "P_reward", "action": "continue", "reason": "probe",
               "claims": [{"id": "x", "value": 1}], "expected_outcomes": {}}
    post = {"ranking": ranking, "probe": None, "action": "hold", "reason": "hold",
            "claims": [], "expected_outcomes": {}}

    def fake_fresh(folder, stage, packet, current_protocol):
        chosen = initial if stage == "initial" else post
        packet_path = folder / "model" / f"{stage}.packet.json"
        attempt_path = folder / "model" / f"{stage}.attempt1.json"
        decision_path = folder / "model" / f"{stage}.decision.json"
        driver.write_new(packet_path, packet)
        driver.write_new(attempt_path, {"input_tokens": 2, "output_tokens": 1,
                                        "execution_wall_s": 0.1, "cpu_s": 0.01})
        driver.write_new(decision_path, chosen)
        driver.write_new(folder / "model" / f"{stage}.outcome.json", {
            "stage": stage, "status": "valid", "packet_sha256_lf": driver.sha(packet_path),
            "attempts": [{"path": f"model/{stage}.attempt1.json", "sha256_lf": driver.sha(attempt_path)}],
            "decision": chosen, "decision_sha256_lf": driver.sha(decision_path), "error": None})
        return chosen, []

    monkeypatch.setattr(driver, "fresh_decision", fake_fresh)
    monkeypatch.setattr(driver, "approved_probe", lambda *args, **kwargs: {
        "request_id": "r1", "probe": "P_reward", "outcome": "abnormal",
        "measurement": {"relative_error": 0.4}, "receipt_digest": "b" * 64, "gpu_s": 0.0})
    original_write_new = driver.write_new

    def fail_pair_only(path, value):
        if Path(path).name == "pair_1.json":
            raise OSError("pair storage unavailable")
        return original_write_new(path, value)

    monkeypatch.setattr(driver, "write_new", fail_pair_only)
    result = driver.run_case(protocol_path, "pair_write_failure", "h01_s2027")
    case_folder = tmp_path / "evals/results/pair_write_failure/cases/h01_s2027"
    assert result["status"] == "hold" and result["error_type"] == "ModelHarnessError"
    assert len(result["pairs"]) == 1 and result["pairs"][0]["stage"] == 1
    assert result["costs"]["model_calls"] == 3 and result["costs"]["input_tokens"] == 6
    assert not (case_folder / "model/pair_1.json").exists()
    assert driver.read(case_folder / "result.json")["pairs"] == result["pairs"]


def test_initial_packet_write_failure_uses_memory_outcome_and_writes_hold_result(driver, tmp_path, monkeypatch):
    protocol = {"max_probes": 1, "scope": "integration",
                "cases": [{"case": "h01_s2027", "checkpoint_sha256": "a" * 64}]}
    protocol_path = tmp_path / "protocol.json"
    driver.write_new(protocol_path, protocol)
    monkeypatch.setattr(driver, "verify_protocol", lambda *args: None)
    monkeypatch.setattr(driver, "input_packet", lambda *args: {
        "facts": {"x": 1}, "observed": [], "budget_left": 1})
    original_write_new = driver.write_new

    def fail_initial_packet(path, value):
        if Path(path).name == "initial.packet.json":
            raise OSError(13, "secret-user-path C:/Users/private/account")
        return original_write_new(path, value)

    monkeypatch.setattr(driver, "write_new", fail_initial_packet)
    result = driver.run_case(protocol_path, "initial_packet_failure", "h01_s2027")
    case_folder = tmp_path / "evals/results/initial_packet_failure/cases/h01_s2027"
    assert result["status"] == "hold" and result["initial_outcome"]["status"] == "harness_fatal"
    assert result["initial_outcome"]["attempts"] == [] and result["costs"]["model_calls"] == 0
    assert not (case_folder / "model/initial.outcome.json").exists()
    assert (case_folder / "result.json").exists() and "C:/Users/private" not in json.dumps(result)


def test_post_packet_write_failure_records_called_arm_and_hold_result(driver, tmp_path, monkeypatch):
    protocol = {"max_probes": 1, "scope": "integration", "probe_execution_budget_s": 180,
                "cases": [{"case": "h01_s2027", "checkpoint_sha256": "a" * 64}]}
    protocol_path = tmp_path / "protocol.json"
    driver.write_new(protocol_path, protocol)
    monkeypatch.setattr(driver, "verify_protocol", lambda *args: None)
    monkeypatch.setattr(driver, "input_packet", lambda *args: {
        "facts": {"x": 1}, "observed": [], "budget_left": 1})
    decision = {"ranking": list(L.MECHANISMS), "probe": "P_reward", "action": "continue",
                "reason": "probe", "claims": [{"id": "x", "value": 1}], "expected_outcomes": {}}
    real_fresh = driver.fresh_decision

    def initial_then_real(folder, stage, packet, current_protocol):
        if stage != "initial":
            return real_fresh(folder, stage, packet, current_protocol)
        packet_path = folder / "model/initial.packet.json"
        attempt_path = folder / "model/initial.attempt1.json"
        decision_path = folder / "model/initial.decision.json"
        driver.write_new(packet_path, packet)
        driver.write_new(attempt_path, {"input_tokens": 1, "output_tokens": 1,
                                        "execution_wall_s": 0.1, "cpu_s": 0.01})
        driver.write_new(decision_path, decision)
        driver.write_new(folder / "model/initial.outcome.json", {
            "stage": "initial", "status": "valid", "packet_sha256_lf": driver.sha(packet_path),
            "attempts": [{"path": "model/initial.attempt1.json", "sha256_lf": driver.sha(attempt_path)}],
            "decision": decision, "decision_sha256_lf": driver.sha(decision_path), "error": None})
        return decision, []

    monkeypatch.setattr(driver, "fresh_decision", initial_then_real)
    monkeypatch.setattr(driver, "approved_probe", lambda *args, **kwargs: {
        "request_id": "r1", "probe": "P_reward", "outcome": "abnormal",
        "measurement": {"relative_error": 0.4}, "receipt_digest": "b" * 64, "gpu_s": 0.0})
    original_write_new = driver.write_new

    def fail_control_packet(path, value):
        if Path(path).name == "without_receipt_1.packet.json":
            raise OSError(5, "private account path")
        return original_write_new(path, value)

    monkeypatch.setattr(driver, "write_new", fail_control_packet)
    result = driver.run_case(protocol_path, "post_packet_failure", "h01_s2027")
    pair = result["pairs"][0]
    assert result["status"] == "hold" and pair["control"]["status"] == "harness_fatal"
    assert pair["actual"] is None and pair["proof_eligible"] is False
    assert result["costs"]["model_calls"] == 1 and "private account" not in json.dumps(result)


@pytest.mark.parametrize("attempt_value", [
    pytest.param("malformed", id="malformed-json"),
    pytest.param({"input_tokens": True, "output_tokens": -1, "execution_wall_s": "bad"},
                 id="missing-and-bad-fields"),
])
def test_bad_attempt_costs_are_null_but_count_and_result_survive(driver, tmp_path, monkeypatch, attempt_value):
    protocol = {"max_probes": 1, "scope": "integration",
                "cases": [{"case": "h01_s2027", "checkpoint_sha256": "a" * 64}]}
    protocol_path = tmp_path / "protocol.json"
    driver.write_new(protocol_path, protocol)
    monkeypatch.setattr(driver, "verify_protocol", lambda *args: None)
    monkeypatch.setattr(driver, "input_packet", lambda *args: {
        "facts": {"x": 1}, "observed": [], "budget_left": 1})

    def broken(folder, stage, packet, current_protocol):
        packet_path = folder / "model/initial.packet.json"
        attempt_path = folder / "model/initial.attempt1.json"
        driver.write_new(packet_path, packet)
        attempt_path.parent.mkdir(parents=True, exist_ok=True)
        if attempt_value == "malformed":
            attempt_path.write_text("{", encoding="utf-8")
        else:
            driver.write_new(attempt_path, attempt_value)
        harness = driver.ModelHarnessError("model attempt artifact 판독 실패: JSONDecodeError")
        driver.write_new(folder / "model/initial.outcome.json", {
            "stage": "initial", "status": "harness_fatal", "packet_sha256_lf": driver.sha(packet_path),
            "attempts": [{"path": "model/initial.attempt1.json", "sha256_lf": driver.sha(attempt_path)}],
            "decision": None, "decision_sha256_lf": None,
            "error": {"type": "ModelHarnessError", "message": str(harness)}})
        raise harness

    monkeypatch.setattr(driver, "fresh_decision", broken)
    suffix = "json" if attempt_value == "malformed" else "fields"
    result = driver.run_case(protocol_path, f"bad_cost_{suffix}", "h01_s2027")
    assert result["status"] == "hold" and result["costs"]["model_calls"] == 1
    assert result["costs"]["input_tokens"] is None and result["costs"]["output_tokens"] is None
    assert result["costs"]["model_wall_s"] is None and result["costs"]["model_cpu_s"] is None


def test_initial_valid_outcome_write_failure_becomes_memory_harness_fatal(driver, tmp_path, monkeypatch):
    protocol = {"max_probes": 1, "max_infra_attempts": 2, "model": "model", "scope": "integration",
                "cases": [{"case": "h01_s2027", "checkpoint_sha256": "a" * 64}]}
    protocol_path = tmp_path / "protocol.json"
    driver.write_new(protocol_path, protocol)
    monkeypatch.setattr(driver, "verify_protocol", lambda *args: None)
    monkeypatch.setattr(driver, "input_packet", lambda *args: {
        "facts": {"x": 1}, "observed": [], "budget_left": 1})
    hold = {"ranking": list(L.MECHANISMS), "probe": None, "action": "hold", "reason": "hold",
            "claims": [{"id": "x", "value": 1}], "expected_outcomes": {}}
    folder = tmp_path / "evals/results/outcome_write_initial/cases/h01_s2027"

    def worker(args, **kwargs):
        packet_path = folder / "model/initial.packet.json"
        driver.write_new(folder / "model/initial.attempt1.json", {
            "packet_sha256": driver.sha(packet_path),
            "messages_sha256": driver.L.digest(driver.D.decision_messages(driver.read(packet_path))),
            "content": json.dumps(hold), "exit_code": 0, "input_tokens": 1, "output_tokens": 1,
            "execution_wall_s": 0.1, "cpu_s": 0.01})
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(driver.subprocess, "run", worker)
    original_write_new = driver.write_new

    def fail_outcome(path, value):
        if Path(path).name == "initial.outcome.json":
            raise OSError(5, "private path")
        return original_write_new(path, value)

    monkeypatch.setattr(driver, "write_new", fail_outcome)
    result = driver.run_case(protocol_path, "outcome_write_initial", "h01_s2027")
    outcome = result["initial_outcome"]
    assert result["status"] == "hold" and outcome["status"] == "harness_fatal"
    assert outcome["decision"] is None and outcome["error"]["type"] == "ModelHarnessError"
    assert not (folder / "model/initial.outcome.json").exists() and "private path" not in json.dumps(result)


def test_post_valid_outcome_write_failure_records_fatal_called_arm(driver, tmp_path, monkeypatch):
    folder = tmp_path / "case"
    observed = {"request_id": "r1", "probe": "P_reward", "outcome": "abnormal",
                "measurement": {"relative_error": 0.4}}
    packet = {"facts": {"x": 1, "receipt:r1:relative_error": 0.4}, "observed": [observed],
              "budget_left": 0, "previous_decision": {"ranking": list(L.MECHANISMS)}}
    before = {"ranking": list(L.MECHANISMS), "action": "continue", "claims": []}
    hold = {"ranking": list(L.MECHANISMS), "probe": None, "action": "hold", "reason": "hold",
            "claims": [{"id": "x", "value": 1}], "expected_outcomes": {}}

    def worker(args, **kwargs):
        packet_path = folder / "model/without_receipt_1.packet.json"
        driver.write_new(folder / "model/without_receipt_1.attempt1.json", {
            "packet_sha256": driver.sha(packet_path),
            "messages_sha256": driver.L.digest(driver.D.decision_messages(driver.read(packet_path))),
            "content": json.dumps(hold), "exit_code": 0})
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(driver.subprocess, "run", worker)
    original_write_new = driver.write_new

    def fail_outcome(path, value):
        if Path(path).name == "without_receipt_1.outcome.json":
            raise OSError(5, "private path")
        return original_write_new(path, value)

    monkeypatch.setattr(driver, "write_new", fail_outcome)
    with pytest.raises(driver.ModelHarnessError) as raised:
        driver.stage_pair(folder, 0, before, packet, {"model": "model", "max_infra_attempts": 2})
    pair = driver.read(folder / "model/pair_1.json")
    assert raised.value.outcome["status"] == "harness_fatal"
    assert pair["control"]["status"] == "harness_fatal" and pair["control"]["decision"] is None
    assert pair["actual"] is None and pair["proof_eligible"] is False


@pytest.mark.parametrize("value", ["missing", "malformed", [], {}, float("inf"), -1, True, "bad"])
def test_sidecar_cost_validation_rejects_missing_malformed_and_bad_values(driver, tmp_path, value):
    path = tmp_path / "sidecar.json"
    if value == "malformed":
        path.write_text("{", encoding="utf-8")
    elif isinstance(value, float):
        path.write_text('{"cpu_s": Infinity}', encoding="utf-8")
    elif value != "missing":
        driver.write_new(path, value if isinstance(value, (list, dict)) else {"cpu_s": value})
    assert driver.cost_value([path], "cpu_s") == (None, True)


def test_sidecar_explicit_null_stays_null_without_validation_error(driver, tmp_path):
    path = tmp_path / "sidecar.json"
    driver.write_new(path, {"cpu_s": None})
    assert driver.cost_value([path], "cpu_s") == (None, False)


def test_malformed_execution_sidecar_writes_typed_hold_result(driver, tmp_path, monkeypatch):
    protocol = {"max_probes": 1, "scope": "integration",
                "cases": [{"case": "h01_s2027", "checkpoint_sha256": "a" * 64}]}
    protocol_path = tmp_path / "protocol.json"
    driver.write_new(protocol_path, protocol)
    monkeypatch.setattr(driver, "verify_protocol", lambda *args: None)
    monkeypatch.setattr(driver, "input_packet", lambda *args: {
        "facts": {"x": 1}, "observed": [], "budget_left": 1})

    def probe(folder, tag, spec, current_protocol, packet, probe_name, stage, ledger, binding):
        execution = tmp_path / f"evals/results/{tag}/execution/h01_s2027__{probe_name}.json"
        resources = tmp_path / f"evals/results/{tag}/resources/h01_s2027__{probe_name}.json"
        execution.parent.mkdir(parents=True, exist_ok=True)
        execution.write_text("{", encoding="utf-8")
        driver.write_new(resources, {"simulator_steps": 12})
        return {"request_id": "r1", "probe": probe_name, "outcome": "normal", "measurement": {},
                "receipt_digest": "b" * 64, "gpu_s": 0.0}

    monkeypatch.setattr(driver, "approved_probe", probe)
    result = driver.run_case(protocol_path, "bad_execution_sidecar", "h01_s2027", controller="fixed_order")
    assert result["status"] == "hold" and result["stop_reason"] == "cost_artifact_invalid"
    assert result["error_type"] == "ModelHarnessError" and result["costs"]["isaac_cpu_s"] is None
    assert result["costs"]["isaac_simulator_steps"] == 12
    assert (tmp_path / "evals/results/bad_execution_sidecar/cases/h01_s2027/result.json").exists()


def test_checkpoint_change_blocks_before_consuming_approval(driver, tmp_path, monkeypatch):
    checkpoint = tmp_path / "changed.pt"
    checkpoint.write_bytes(b"changed")
    monkeypatch.setattr(driver, "gpu_snapshot", lambda: {})
    monkeypatch.setattr(driver, "find_run", lambda *args: tmp_path)
    monkeypatch.setattr(driver, "last_checkpoint", lambda *args: checkpoint)
    ledger = L.Ledger(tmp_path / "ledger.jsonl")
    spec = {"case": "h01_s2027", "checkpoint_sha256": "a" * 64}
    with pytest.raises(ValueError, match="checkpoint"):
        driver.approved_probe(tmp_path / "case", "tag", spec, {}, {"observed": []}, "P_reward", "1", ledger, {})
    assert ledger.events() == []


def test_case_subprocess_failure_does_not_publish_command_paths(driver, tmp_path, monkeypatch):
    protocol = {"max_probes": 2, "scope": "integration",
                "cases": [{"case": "h01_s2027", "checkpoint_sha256": "a" * 64}]}
    path = tmp_path / "protocol.json"
    driver.write_new(path, protocol)
    monkeypatch.setattr(driver, "verify_protocol", lambda *args: None)
    monkeypatch.setattr(driver, "input_packet", lambda *args: {
        "facts": {"x": 1}, "observed": [], "budget_left": 2})

    def failed_probe(*args, **kwargs):
        raise subprocess.CalledProcessError(2, ["script.exe", "C:/Users/private-user/private-path"])

    monkeypatch.setattr(driver, "approved_probe", failed_probe)
    result = driver.run_case(path, "subprocess_failure", "h01_s2027", controller="fixed_order")
    assert result["status"] == "hold" and result["error_type"] == "CalledProcessError"
    assert "private-user" not in json.dumps(result)
    assert result["costs"]["model_calls"] == 0
    assert (tmp_path / "evals/results/subprocess_failure/cases/h01_s2027/result.json").exists()
