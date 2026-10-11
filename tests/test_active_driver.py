"""새 active driver의 승인 결속·HOLD·비용 경계를 GPU 없이 검증한다."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from rl_triage import probe_loop as L


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def driver(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("active_driver_test", ROOT / "evals/run_active_diagnosis.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    return module


def test_wsl_path_rejects_non_windows_paths(driver):
    assert driver.wsl_path(ROOT / "evals/active_model_call.py").endswith("/rl-triage-agent/evals/active_model_call.py")


def test_semantic_model_error_is_not_retried(driver, tmp_path, monkeypatch):
    folder = tmp_path / "case"
    calls = []

    def worker(args, **kwargs):
        calls.append(args)
        output = folder / "model/initial.attempt1.json"
        packet = folder / "model/initial.packet.json"
        import hashlib
        driver.write_new(output, {"packet_sha256": hashlib.sha256(packet.read_bytes()).hexdigest(),
                                  "content": '{"ranking": ["fake"]}', "exit_code": 0})
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(driver.subprocess, "run", worker)
    with pytest.raises(ValueError, match="decision 필드"):
        driver.fresh_decision(folder, "initial", {"facts": {"x": 1}, "observed": [], "budget_left": 2},
                              {"model": "model", "max_infra_attempts": 2})
    assert len(calls) == 1
    assert not (folder / "model/initial.decision.json").exists()


def test_stage_pair_masks_only_latest_measurement_and_alternates_order(driver, tmp_path, monkeypatch):
    first = {"request_id": "old", "probe": "P_reward", "outcome": "normal", "measurement": {"x": 1}}
    last = {"request_id": "new", "probe": "P_noise", "outcome": "normal", "measurement": {"x": 2}}
    packet = {"phase": "after_receipt", "facts": {"telemetry:x": 5, "receipt:old:x": 1, "receipt:new:x": 2},
              "budget_left": 0, "observed": [first, last], "previous_decision": {"ranking": list(L.MECHANISMS)}}
    called = []
    decision = {"ranking": list(L.MECHANISMS), "action": "hold", "claims": []}

    def fake(folder, stage, current, protocol):
        called.append((stage, current))
        return decision, []

    monkeypatch.setattr(driver, "fresh_decision", fake)
    driver.stage_pair(tmp_path, 0, decision, packet, {})
    actual, masked = [c[1] for c in called]
    assert actual == packet
    assert masked == {**packet, "observed": [first], "facts": {"telemetry:x": 5, "receipt:old:x": 1}}
    called.clear()
    driver.stage_pair(tmp_path, 1, decision, packet, {})
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
    monkeypatch.setattr(driver, "gpu_snapshot", lambda: {"idle": True})
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


def test_initial_model_hold_preserves_remaining_budget(driver, tmp_path, monkeypatch):
    protocol = {"max_probes": 2, "scope": "integration", "cases": [{"case": "h01_s2027", "checkpoint_sha256": "a" * 64}]}
    path = tmp_path / "protocol.json"
    driver.write_new(path, protocol)
    monkeypatch.setattr(driver, "verify_protocol", lambda *args: None)
    monkeypatch.setattr(driver, "input_packet", lambda *args: {"facts": {"x": 1}, "observed": [], "budget_left": 2})
    hold = {"ranking": list(L.MECHANISMS), "probe": None, "action": "hold", "reason": "모호한 텔레메트리",
            "claims": [{"id": "x", "value": 1}], "expected_outcomes": {}}
    monkeypatch.setattr(driver, "fresh_decision", lambda *args: (hold, []))
    result = driver.run_case(path, "hold_test", "h01_s2027")
    assert result["status"] == "hold" and result["stop_reason"] == "model_hold"
    assert result["budget_left"] == 2 and result["evidence_state"]["status"] == "open"
    assert result["observed"] == [] and result["costs"]["isaac_wall_s"] == 0


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
