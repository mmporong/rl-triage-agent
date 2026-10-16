"""네트워크·GPU 없이 종단 상태와 승인 소비, 실패 중단을 실행한다."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))
import r1_repair_demo as D
import r1_repair_contract as C
import r1_repair_model_call as M


class Backend:
    def __init__(self, corrected_healthy=True, count_status="abnormal", fail_corrected=False):
        self.calls = []
        self.corrected_healthy = corrected_healthy
        self.count_status = count_status
        self.fail_corrected = fail_corrected

    def train(self, stage, isaaclab, arm, seed, run_name, binding):
        self.calls.append(("train", arm))
        if arm == "CORRECTED" and self.fail_corrected:
            raise RuntimeError("mock training failure")
        stage.mkdir(parents=True)
        telemetry = stage / "telemetry.json"
        value = 1.2 if arm == "R1" else 0.15
        D.E.write_new(telemetry, {"summary": {D.ALARM.METRIC: {"mean_last_20pct": value}},
                                 "series": {D.ALARM.METRIC: [value] * 300}})
        return {"arm": arm, "seed": seed, "run_name": run_name,
                "checkpoint": {"path": str(stage / "model_299.pt"), "sha256": "a" * 64},
                "telemetry": {"path": str(telemetry), "sha256": D.E.sha(telemetry)},
                "execution": {"wall_s": 600}, "inspection": {
                    "status": "abnormal" if arm == "R1" else "normal",
                    "optimizer_steps": [4 if arm == "R1" else 20] * 3,
                    "minibatches": [4 if arm == "R1" else 20] * 3}}

    def fixed_evaluate(self, stage, isaaclab, artifacts, run_name, binding):
        self.calls.append(("evaluate", tuple(artifacts)))
        stage.mkdir(parents=True)
        reports, paths = {}, {}
        for arm in artifacts:
            healthy = arm == "NONE" or (arm == "CORRECTED" and self.corrected_healthy)
            source = "upstream_r1_none_s2042" if healthy else "upstream_r1_s2042"
            report = D.read(ROOT / f"evals/results/upstream_r1_smoke_20261014/runs/{source}.json")
            report["name"] = run_name + "_" + arm.lower()
            report["checkpoint"]["sha256"] = "a" * 64
            path = stage / f"{arm}.json"
            D.E.write_new(path, report)
            reports[arm], paths[arm] = report, str(path)
        return {"reports": reports, "report_paths": paths, "execution": {"wall_s": 30}}

    def legacy_probe(self, stage, isaaclab, artifacts, run_name, binding, probes=None):
        self.calls.append(("legacy_probe", tuple(artifacts), probes))
        stage.mkdir(parents=True)
        measurements = {
            "P_value": {"critic_change": 1.0}, "P_noise": {"noise_ratio": 1.0},
            "P_episode": {"timeout_ratio": 1.0}, "P_physics": {"mismatch_count": 0},
            "P_torque": {"low_speed_saturation": 0.0},
            "P_reward": {"track_lin_vel_xy_exp_rel_error": 0.0, "track_ang_vel_z_exp_rel_error": 0.0},
        }
        return {"report": {"probes": {name: {"measurement": measurements[name], "exit": 0}
                                      for name in (probes or C.L.PROBES)}}, "execution": {"wall_s": 20}}

    def inspect(self, stage, isaaclab, arm, seed, run_name, binding):
        self.calls.append(("inspect", arm))
        stage.mkdir(parents=True)
        status = self.count_status if arm == "R1" else "normal"
        assert binding["source_checkpoint_sha256"] == "a" * 64
        return {"measurement": {"status": status, "optimizer_steps": [4 if arm == "R1" else 20] * 3,
                                "minibatches": [4 if arm == "R1" else 20] * 3}, "execution": {"wall_s": 30}}


def model_caller(first_probe="P_value", invalid=False, correction_hold=False):
    def caller(kind, data):
        packet = json.loads(data)
        if invalid:
            content = "{}"
        elif kind == "diagnosis":
            ranking = ["reward", "actuator", "exploration", "optimizer", "physics", "termination"]
            key = next(iter(packet["facts"]))
            content = json.dumps({"ranking": ranking, "probe": first_probe, "action": "continue", "reason": "검사",
                "claims": [{"id": key, "value": packet["facts"][key]}],
                "expected_outcomes": {h: C.public_catalog()[first_probe]["expect"][h] for h in ranking[:3]}})
        else:
            content = json.dumps({"correction_id": None if correction_hold else C.CORRECTION,
                "recovery_mode": None if correction_hold else "restart", "cited_facts": ["confirmed_subtype"],
                "checklist": {key: "rule_mode" for key in packet["checklist"]},
                "hold_reason": "보류" if correction_hold else None})
        return M.dialogue(kind, data, lambda *_: {"content": content, "provider_model": C.MODEL,
            "finish_reason": "stop", "input_tokens": 20, "output_tokens": 10, "total_tokens": 30, "wall_s": 1.0})
    return caller


@pytest.fixture
def setup(tmp_path, monkeypatch):
    protocol = {**D.fixed_protocol(), "source_sha256_lf": {}, "runtime_source_sha256_lf": {}}
    path = tmp_path / "protocol.json"
    D.E.write_new(path, protocol)
    monkeypatch.setattr(D, "PRIVATE", tmp_path / "runs")
    # binding 프로토콜 경로는 실제 repo-relative를 유지하며 hash만 fixture에서 제공한다.
    original_sha = D.E.sha
    monkeypatch.setattr(D.E, "sha", lambda p, lf=False: "b" * 64 if p == D.PROTOCOL else original_sha(p, lf))
    monkeypatch.setattr(D, "verify", lambda *args, **kwargs: (protocol, "c" * 40))
    return lambda: {"status": "fresh", "seeds": [2070, 3070]}


def run(setup, backend, caller=None):
    return D.execute("r1_repair_s2070_20261016", Path("unused"), "operator", True,
                     backend=backend, caller=caller or model_caller(), seed_guard=lambda *a, **k: setup())


def test_complete_path_models_choose_then_ledgers_consume_once(setup):
    backend = Backend()
    result = run(setup, backend)
    assert result["outcome"] == "recovered" and result["demo_success"]
    assert result["api_requests"] == 2
    assert result["probe_steps"][0]["probe"] == "P_value"
    assert result["loop"]["hypotheses"] == [C.COUNT]
    folder = D.PRIVATE / result["tag"]
    ledger = D.CorrectionLedger(folder / "correction.ledger.jsonl", C.correction_registry())
    assert [r["event"] for r in ledger.events()] == ["proposed", "approved", "consumed", "receipt"]
    assert ("train", "CORRECTED") in backend.calls
    with pytest.raises(FileExistsError):
        run(setup, backend)


@pytest.mark.parametrize("status", ["unknown", "normal"])
def test_unconfirmed_count_cannot_trigger_correction(setup, status):
    backend = Backend(count_status=status)
    result = run(setup, backend, model_caller(C.NEW_PROBE))
    assert result["outcome"] == "loop_not_confirmed"
    assert not result["demo_success"]
    assert ("train", "CORRECTED") not in backend.calls


def test_invalid_diagnosis_cannot_fall_back_to_rule(setup):
    backend = Backend()
    result = run(setup, backend, model_caller(invalid=True))
    assert result["outcome"] == "model_invalid"
    assert result["api_requests"] == 2
    assert not result["probe_steps"]


def test_healthy_count_does_not_override_failed_policy(setup):
    result = run(setup, Backend(corrected_healthy=False))
    assert result["conditions"]["restored_update_count"]
    assert result["outcome"] == "not_recovered" and not result["demo_success"]


def test_correction_hold_does_not_execute(setup):
    backend = Backend()
    result = run(setup, backend, model_caller(correction_hold=True))
    assert result["outcome"] == "correction_not_proposed"
    assert ("train", "CORRECTED") not in backend.calls


def test_failed_consumed_correction_gets_unknown_receipt(setup):
    with pytest.raises(RuntimeError):
        run(setup, Backend(fail_corrected=True))
    folder = D.PRIVATE / "r1_repair_s2070_20261016"
    result = D.read(folder / "result.json")
    assert result["outcome"] == "correction_unknown"
    ledger = D.CorrectionLedger(folder / "correction.ledger.jsonl", C.correction_registry())
    assert ledger.events()[-1]["verdict"] == "unknown"
    assert ledger.orphans() == []


def test_approval_and_seed_gate_before_training(setup):
    backend = Backend()
    with pytest.raises(ValueError):
        D.execute("r1_repair_s2070_20261016", Path("unused"), "operator", False, backend=backend)
    def reused(*a, **k):
        raise ValueError("seed used")
    with pytest.raises(ValueError):
        D.execute("r1_repair_s2070_20261016", Path("unused"), "operator", True, backend=backend, seed_guard=reused)
    assert not backend.calls and not D.PRIVATE.exists()


def test_provider_model_mismatch_is_terminal_without_retry(setup, tmp_path):
    calls = []
    def caller(kind, data):
        calls.append(kind)
        return M.dialogue(kind, data, lambda *_: {"content": "{}", "provider_model": "unregistered", "finish_reason": "stop"})
    result = D.model_decision(tmp_path / "model", "diagnosis", {
        "schema": "r1_repair_diagnosis_packet_v1", "facts": {"x": 1}, "observed": [],
        "budget_left": 7, "execution_budget_s": 1200}, caller)
    assert len(calls) == 1 and result["review"]["status"] == "call_error"


@pytest.mark.parametrize("outcome", ["baseline_invalid", "source_failure_absent"])
def test_source_gate_stops_before_model_or_reference(setup, monkeypatch, outcome):
    backend = Backend()
    monkeypatch.setattr(D, "source_gate", lambda *args: {"outcome": outcome})
    result = run(setup, backend)
    assert result["outcome"] == outcome and result["api_requests"] == 0
    assert not result["model"] and not result["probe_steps"]
    assert not any(row[0] in ("legacy_probe", "inspect") for row in backend.calls)


def test_correction_model_invalid_does_not_fall_back_to_rule(setup):
    backend = Backend()
    good, bad = model_caller(), model_caller(invalid=True)
    result = run(setup, backend, lambda kind, data: good(kind, data) if kind == "diagnosis" else bad(kind, data))
    assert result["outcome"] == "model_invalid"
    assert result["api_requests"] == 3
    assert ("train", "CORRECTED") not in backend.calls


@pytest.mark.parametrize("method", ["approve", "consume"])
def test_probe_approval_failure_preserves_error_and_pending_state(setup, monkeypatch, method):
    def fail(*args):
        raise ValueError("approval failure")
    monkeypatch.setattr(C.M.RegistryLedger, method, fail)
    with pytest.raises(ValueError, match="approval failure"):
        run(setup, Backend())
    folder = D.PRIVATE / "r1_repair_s2070_20261016"
    assert D.read(folder / "result.json")["failure"]["type"] == "ValueError"
    rows = C.M.RegistryLedger(folder / "probes.ledger.jsonl", C.registry()).requests()
    assert [row["state"] for row in rows.values()] == ["pending" if method == "approve" else "approved"]


def test_packet_mutation_during_call_is_not_accepted(setup, tmp_path):
    folder = tmp_path / "model"
    def caller(kind, data):
        value = model_caller()(kind, data)
        (folder / "diagnosis.packet.json").write_text('{}', encoding="utf-8")
        return value
    with pytest.raises(ValueError, match="packet"):
        D.model_decision(folder, "diagnosis", {
            "schema": "r1_repair_diagnosis_packet_v1", "facts": {"x": 1}, "observed": [],
            "budget_left": 7, "execution_budget_s": 1200}, caller)


def test_wsl_timeout_cleans_only_owned_worker_and_does_not_retry(setup, tmp_path, monkeypatch):
    cleanup, requests = [], []
    monkeypatch.setattr(D, "wsl_path", lambda p: str(p))
    def stop(*args):
        cleanup.append(args)
        return "verified"
    monkeypatch.setattr(D, "stop_model_worker", stop)
    def timeout(*args, **kwargs):
        requests.append(kwargs)
        raise D.subprocess.TimeoutExpired(args[0], kwargs["timeout"])
    monkeypatch.setattr(D.subprocess, "run", timeout)
    with pytest.raises(TimeoutError):
        D.model_decision(tmp_path / "model", "diagnosis", {
            "schema": "r1_repair_diagnosis_packet_v1", "facts": {"x": 1}, "observed": [],
            "budget_left": 7, "execution_budget_s": 1200})
    assert len(cleanup) == len(requests) == 1
    assert requests[0]["timeout"] == 900
    assert D.read(tmp_path / "model/diagnosis.transport_error.json")["worker_cleanup"] == "verified"


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_timeout_does_not_claim_cleanup_without_verification(setup, tmp_path, monkeypatch, cleanup_fails):
    monkeypatch.setattr(D, "wsl_path", lambda p: str(p))
    def timeout(*args, **kwargs):
        raise D.subprocess.TimeoutExpired(args[0], kwargs["timeout"])
    monkeypatch.setattr(D.subprocess, "run", timeout)
    if cleanup_fails:
        def fail(*args):
            raise RuntimeError("cleanup failed")
        monkeypatch.setattr(D, "stop_model_worker", fail)
    with pytest.raises(TimeoutError):
        D.model_decision(tmp_path / "model", "diagnosis", {
            "schema": "r1_repair_diagnosis_packet_v1", "facts": {"x": 1}, "observed": [],
            "budget_left": 7, "execution_budget_s": 1200})
    receipt = D.read(tmp_path / "model/diagnosis.transport_error.json")
    assert receipt["worker_cleanup"] == ("unverified" if cleanup_fails else "unverified_no_pid")
    assert receipt.get("cleanup_error_type") == ("RuntimeError" if cleanup_fails else None)


def test_restart_reference_semantics_match_v4(setup):
    backend = Backend()
    result = run(setup, backend)
    folder = D.PRIVATE / result["tag"]
    assert D.read(folder / "artifact_none.json")["seed"] == 2070
    assert D.read(folder / "artifact_r1.json")["seed"] == 2070
    assert D.read(folder / "artifact_corrected.json")["seed"] == 3070
    assert result["recovery"]["reference_metrics"] == D.read(folder / "source_evaluation/NONE.json")["metrics"]


def test_execute_timeout_without_reservation_keeps_api_cost_unknown(setup, monkeypatch):
    backend = Backend()
    def timeout(*args, **kwargs):
        raise D.subprocess.TimeoutExpired(args[0], kwargs["timeout"])
    monkeypatch.setattr(D.subprocess, "run", timeout)
    with pytest.raises(TimeoutError):
        D.execute("r1_repair_s2070_20261016", Path("unused"), "operator", True,
                  backend=backend, seed_guard=lambda *args, **kwargs: setup())
    folder = D.PRIVATE / "r1_repair_s2070_20261016"
    result = D.read(folder / "result.json")
    assert result["outcome"] == "model_infra_failed" and not result["demo_success"]
    assert result["api_requests"] is None and result["api_requests_in_saved_records"] == 0
    assert not result["pending_model_records"] and not result["conditions"]["costs_recorded"]
    assert result["transport_errors"] == ["diagnosis.transport_error.json"]
    assert D.read(folder / "model/diagnosis.transport_error.json")["worker_cleanup"] == "unverified_no_pid"
    assert ("train", "CORRECTED") not in backend.calls
