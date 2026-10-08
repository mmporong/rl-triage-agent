"""승인 원장과 고정 평가를 연결한 CLI를 모델·네트워크·GPU 차단 환경에서 검사한다."""
import importlib.util
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from rl_triage import probe_loop as L
from rl_triage import experiment_cost as C
from test_behavior_oracle import GATES, report
from test_offline_replay import ROOT, _offline


def _write(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, allow_nan=False), encoding="utf-8")
    return path


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _cli(*args):
    return _offline("evals/audit_experiment.py", *args)


def _setup(tmp_path, method="agent", limits=None):
    folder = tmp_path / method
    folder.mkdir(parents=True)
    prereg = {"case": "fixture_s2026", "hypotheses": ["reward", "actuator"],
              "checkpoint_sha256": "b" * 64, "budget_probes": 2,
              "thresholds": {p: list(v) for p, v in L.THRESHOLDS.items()},
              "expectations": {p: {h: L.PROBES[p]["expect"].get(h) for h in ("reward", "actuator")}
                               for p in L.PROBES}}
    state = {"hypotheses": prereg["hypotheses"], "observed": {}, "budget": 2, "skipped": [],
             "status": "open", "history": []}
    _write(folder / "prereg.json", prereg)
    _write(folder / "reference.json", {"P_reward": {"track_lin_vel_xy_exp_rel_error": 0.0,
                                                  "track_ang_vel_z_exp_rel_error": 0.0}})
    ledger = L.Ledger(folder / "ledger.jsonl")
    rid = ledger.propose(prereg, "P_reward", 120)["request_id"]
    state["pending"] = rid
    _write(folder / "state.json", state)
    inputs = folder / "inputs"
    files = {"reference": _write(inputs / "reference.json", report()),
             "gates": _write(inputs / "gates.json", GATES),
             "limits": _write(inputs / "limits.json", {"gpu_wall_s": 120, "model_calls": 4} if limits is None else limits),
             "information": _write(inputs / "information.json", {"case": "fixture_s2026", "telemetry": [1, 2, 3]}),
             "before": _write(inputs / "before.json", report("fixture_s2026", "b" * 64, lin=0.9)),
             "after": _write(inputs / "after.json", report("repair_s2026", "c" * 64))}
    return {"folder": folder, "prereg": prereg, "state": state, "ledger": ledger, "rid": rid,
            "files": files, "contract": folder / "audit_contract.json", "out": folder / "audit_result.json",
            "method": method}


def _register(case, *extra):
    f = case["files"]
    return _cli("register", "--loop-dir", case["folder"], "--reference", f["reference"], "--gates", f["gates"],
                "--limits", f["limits"], "--information", f["information"], "--method", case["method"],
                "--out", case["contract"], *extra)


def _finish(case, *, code=0, gpu_s=25, receipt=True):
    led, rid, prereg = case["ledger"], case["rid"], case["prereg"]
    led.approve(rid, "fixture-reviewer")
    led.consume(rid, prereg, "P_reward")
    if receipt:
        outcome = "abnormal" if code == 0 else "unknown"
        led.receipt(rid, outcome, {"track_lin_vel_xy_exp_rel_error": 0.8,
                                   "track_ang_vel_z_exp_rel_error": 0.0}, gpu_s, code)
        state = case["state"]
        state["observed"] = {"P_reward": outcome}
        state["hypotheses"] = ["reward"] if code == 0 else ["reward", "actuator"]
        state["history"] = [{"request_id": rid, "probe": "P_reward", "outcome": outcome,
                             "dropped": ["actuator"] if code == 0 else []}]
        state["pending"] = None
        state["status"] = "confirmed" if code == 0 else "open"
        _write(case["folder"] / "state.json", state)
        after = _read(case["files"]["after"])
        after.update(started_at=datetime.now(timezone.utc).isoformat(), finished_at=datetime.now(timezone.utc).isoformat())
        _write(case["files"]["after"], after)


def _finalize(case, *extra):
    f = case["files"]
    return _cli("finalize", "--loop-dir", case["folder"], "--contract", case["contract"],
                "--reference", f["reference"], "--before", f["before"], "--out", case["out"], *extra)


def test_real_ledger_to_fixed_eval_cli_runs_offline_without_changing_inputs(tmp_path):
    case = _setup(tmp_path)
    assert _register(case).returncode == 0
    _finish(case)
    original = {p: p.read_bytes() for p in case["folder"].glob("*.json*")}
    proc = _finalize(case, "--after", case["files"]["after"])
    assert proc.returncode == 0, proc.stderr
    result = _read(case["out"])
    assert result["behavior"]["outcome"] == "recovered"
    assert result["loop"]["diagnosis_status"] == "confirmed"
    assert result["costs"]["known_totals"]["gpu_wall_s"] == 25
    assert result["costs"]["totals"]["human_review_s"] is None
    assert result["loop"]["probe_attempts"] == 1
    assert not result["repair_execution_verified"] and not result["probe_execution_checkpoint_verified"]
    assert result["environment"] == {"model_calls": 0, "network": "not used", "gpu": "not used"}
    assert all(path.read_bytes() == data for path, data in original.items())
    again = _finalize(case, "--after", case["files"]["after"])
    assert again.returncode == 2 and "덮어쓰지" in again.stderr


def test_registration_after_consumption_is_rejected(tmp_path):
    case = _setup(tmp_path)
    _finish(case, receipt=False)
    proc = _register(case)
    assert proc.returncode == 2 and "소비하기 전에" in proc.stderr
    assert not case["contract"].exists()


def test_approval_bypass_or_duplicate_receipt_is_rejected(tmp_path):
    for kind in ("consumed", "duplicate_receipt"):
        case = _setup(tmp_path, kind)
        assert _register(case).returncode == 0
        if kind == "consumed":
            ev = {"at": "2099-01-01T00:00:00+00:00", "event": "consumed", "request_id": case["rid"]}
        else:
            _finish(case)
            ev = case["ledger"].events()[-1]
        with case["ledger"].path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(ev) + "\n")
        proc = _finalize(case)
        assert proc.returncode == 2 and "순서 위반" in proc.stderr
        assert not case["out"].exists()


@pytest.mark.parametrize("change", ["prereg", "state", "checkpoint", "threshold", "contract", "pending", "status", "history"])
def test_stale_or_changed_bindings_are_rejected(tmp_path, change):
    case = _setup(tmp_path)
    assert _register(case).returncode == 0
    _finish(case)
    if change == "prereg":
        case["prereg"]["checkpoint_sha256"] = "d" * 64
        _write(case["folder"] / "prereg.json", case["prereg"])
    elif change == "state":
        state = _read(case["folder"] / "state.json")
        state["hypotheses"] = ["optimizer"]
        _write(case["folder"] / "state.json", state)
    elif change == "checkpoint":
        before = _read(case["files"]["before"])
        before["checkpoint"]["sha256"] = "d" * 64
        _write(case["files"]["before"], before)
    elif change == "threshold":
        case["prereg"]["thresholds"]["P_reward"][2] = 0.9
        _write(case["folder"] / "prereg.json", case["prereg"])
    elif change in ("pending", "status", "history"):
        state = _read(case["folder"] / "state.json")
        state[change] = {"pending": "stale-request", "status": "open", "history": []}[change]
        _write(case["folder"] / "state.json", state)
    else:
        contract = _read(case["contract"])
        contract["limits"]["gpu_wall_s"] = 10000
        _write(case["contract"], contract)
    proc = _finalize(case, "--after", case["files"]["after"])
    assert proc.returncode == 2, proc.stdout
    assert not case["out"].exists()


def test_interruption_and_failed_execution_keep_unknown_and_charge_known_cost(tmp_path):
    interrupted = _setup(tmp_path, "interrupted")
    assert _register(interrupted).returncode == 0
    _finish(interrupted, receipt=False)
    assert _finalize(interrupted).returncode == 0
    result = _read(interrupted["out"])
    assert result["loop"]["incomplete_requests"] == [interrupted["rid"]]
    assert result["costs"]["budget_status"] == "unknown"
    assert result["behavior"]["outcome"] == "undetermined"
    failed = _setup(tmp_path, "failed")
    assert _register(failed).returncode == 0
    _finish(failed, code=1, gpu_s=135)
    assert _finalize(failed).returncode == 0
    result = _read(failed["out"])
    assert result["loop"]["observed"] == {"P_reward": "unknown"}
    assert result["costs"]["budget_status"] == "exceeded"
    assert result["costs"]["by_status"]["failed"] == 1
    assert result["loop"]["request_budget_overruns"] == [failed["rid"]]


def test_missing_after_and_abstention_are_not_recovery(tmp_path):
    case = _setup(tmp_path)
    assert _register(case).returncode == 0
    _finish(case)
    assert _finalize(case, "--abstain").returncode == 0
    assert _read(case["out"])["behavior"]["outcome"] == "abstained"


def test_cost_enrichment_fills_missing_only_and_does_not_double_count(tmp_path):
    case = _setup(tmp_path)
    assert _register(case).returncode == 0
    _finish(case)
    event = {"attempt_id": f"fixture_s2026:{case['rid']}", "status": "completed",
             "costs": {"gpu_wall_s": 25, "cpu_s": 8, "human_review_s": 3, "simulator_steps": 51200}}
    extra = _write(tmp_path / "costs.json", [event, {"attempt_id": "diagnosis-1", "status": "timeout",
                                                     "costs": {"model_calls": 1, "gpu_wall_s": 0}}])
    proc = _finalize(case, "--events", extra, "--after", case["files"]["after"])
    assert proc.returncode == 0, proc.stderr
    result = _read(case["out"])
    assert result["costs"]["attempts"] == 2 and result["costs"]["known_totals"]["gpu_wall_s"] == 25
    assert result["costs"]["by_status"]["timeout"] == 1
    assert result["costs"]["totals"]["cpu_s"] is None  # 모델 timeout 쪽 CPU 미기록
    case["out"] = tmp_path / "conflict.json"
    event["costs"]["gpu_wall_s"] = 0
    _write(extra, [event])
    proc = _finalize(case, "--events", extra)
    assert proc.returncode == 2 and "실측값" in proc.stderr


def test_comparison_blocks_information_mismatch_and_incomplete_total_cost(tmp_path):
    cases = [_setup(tmp_path, method) for method in ("agent", "rules")]
    for case in cases:
        assert _register(case).returncode == 0
        _finish(case)
        assert _finalize(case, "--after", case["files"]["after"]).returncode == 0
    out = tmp_path / "compare.json"
    proc = _cli("compare", "--reports", *(c["out"] for c in cases), "--out", out)
    assert proc.returncode == 3, proc.stderr
    result = _read(out)
    assert not result["comparable"]
    assert "agent: incomplete_cost_vector" in result["reasons"]
    assert "information_sha256_mismatch" not in result["reasons"]
    assert all(d["F_failed"] == 1 and d["N_all"] == 1 for d in result["denominators"])
    assert all(d["validated_recovery_F"] is None for d in result["denominators"])
    tampered = _read(cases[0]["out"])
    tampered["behavior"]["outcome"] = "unnecessary_intervention"
    _write(cases[0]["out"], tampered)
    proc = _cli("compare", "--reports", *(c["out"] for c in cases), "--out", tmp_path / "tampered.json")
    assert proc.returncode == 2 and "내용 해시" in proc.stderr


def test_budget_cli_saves_unknown_block_and_never_executes(tmp_path):
    events = _write(tmp_path / "events.json", [{"attempt_id": "crashed", "status": "unknown", "costs": {}}])
    limits = _write(tmp_path / "limits.json", {"gpu_wall_s": 120})
    estimate = _write(tmp_path / "estimate.json", {"gpu_wall_s": 30})
    out = tmp_path / "budget.json"
    proc = _cli("budget", "--events", events, "--limits", limits, "--next-cost", estimate, "--out", out)
    assert proc.returncode == 3, proc.stderr
    assert _read(out)["reason"] == "unknown" and not _read(out)["allowed"]


def test_receipt_outcome_and_source_code_changes_are_detected(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("audit_under_test", ROOT / "evals/audit_experiment.py")
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    case = _setup(tmp_path)
    assert _register(case).returncode == 0
    _finish(case)
    events = case["ledger"].events()
    events[-1]["outcome"] = "normal"
    with pytest.raises(ValueError, match="관측"):
        audit.audit_loop(case["prereg"], _read(case["folder"] / "reference.json"), case["state"], events)
    monkeypatch.setattr(audit, "_code_sources", lambda: {"modified.py": "0" * 64})
    proc = audit.main(["finalize", "--loop-dir", str(case["folder"]), "--contract", str(case["contract"]),
                       "--reference", str(case["files"]["reference"]), "--before", str(case["files"]["before"]),
                       "--out", str(case["out"])])
    assert proc == 2 and not case["out"].exists()


def test_complete_cost_vectors_and_timeline_enable_declared_budget_comparison(tmp_path):
    cases = [_setup(tmp_path, method) for method in ("agent", "rules")]
    for case in cases:
        assert _register(case, "--wall-budget", 300).returncode == 0
        _finish(case, gpu_s=1)
        timeline = _write(case["folder"] / "timeline.json",
                          {"started_at": (datetime.now(timezone.utc) - timedelta(seconds=60)).isoformat(),
                           "finished_at": datetime.now(timezone.utc).isoformat()})
        events = _complete_costs(case)
        proc = _finalize(case, "--after", case["files"]["after"], "--events", events, "--timeline", timeline)
        assert proc.returncode == 0, proc.stderr
        assert _read(case["out"])["costs"]["totals"]["cpu_s"] == pytest.approx(0.11)
    out = tmp_path / "complete_compare.json"
    proc = _cli("compare", "--reports", *(c["out"] for c in cases), "--out", out)
    assert proc.returncode == 0, proc.stderr
    assert _read(out)["comparable"]
    assert not _read(out)["input_delivery_verified"]


def test_timeline_outside_loop_or_missing_timezone_is_rejected(tmp_path):
    case = _setup(tmp_path)
    assert _register(case, "--wall-budget", 300).returncode == 0
    _finish(case)
    timeline = _write(tmp_path / "timeline.json",
                      {"started_at": "2000-01-01T00:00:00+00:00", "finished_at": "2001-01-01T00:00:00+00:00"})
    proc = _finalize(case, "--timeline", timeline)
    assert proc.returncode == 2 and not case["out"].exists()
    _write(timeline, {"started_at": "2000-01-01T00:00:00", "finished_at": datetime.now(timezone.utc).isoformat()})
    proc = _finalize(case, "--timeline", timeline)
    assert proc.returncode == 2 and "timezone" in proc.stderr


def _complete_costs(case):
    after = _read(case["files"]["after"])
    register_at = _read(case["contract"])["registered_at"]
    receipt_at = case["ledger"].events()[-1]["at"]
    zero = {key: 0 for key in C.COST_KEYS}
    events = [{"attempt_id": f"fixture_s2026:{case['rid']}", "status": "completed",
               "costs": {"cpu_s": 0.1, "human_review_s": 0.1, "simulator_steps": 51200}},
              {"attempt_id": "diagnosis-1", "stage": "diagnosis", "status": "completed",
               "started_at": register_at, "finished_at": register_at,
               "costs": {**zero, "cpu_s": 0.01}},
              {"attempt_id": "repair-1", "stage": "repair", "status": "completed",
               "started_at": receipt_at, "finished_at": after["started_at"],
               "artifact_sha256": after["checkpoint"]["sha256"], "costs": zero},
              {"attempt_id": "fixed-eval-1", "stage": "fixed_eval", "status": "completed",
               "started_at": after["started_at"], "finished_at": after["finished_at"],
               "artifact_sha256": hashlib.sha256(case["files"]["after"].read_bytes()).hexdigest(), "costs": zero}]
    return _write(case["folder"] / "costs.json", events)


@pytest.mark.parametrize("probe", ["P_value", "P_torque"])
def test_non_discriminating_or_out_of_order_probe_is_rejected(tmp_path, probe):
    case = _setup(tmp_path)
    events = case["ledger"].events()
    events[0]["probe"] = probe
    events[0]["args_digest"] = L.digest(L.PROBES[probe]["args"])
    case["ledger"].path.write_text(json.dumps(events[0]) + "\n", encoding="utf-8")
    proc = _register(case)
    assert proc.returncode == 2 and "선택 규칙" in proc.stderr


def test_parallel_open_requests_are_rejected(tmp_path):
    case = _setup(tmp_path)
    case["ledger"].propose(case["prereg"], "P_reward", 120)
    proc = _register(case)
    assert proc.returncode == 2 and "열린 probe" in proc.stderr


def test_missing_after_for_healthy_case_does_not_count_unnecessary_intervention(tmp_path):
    cases = [_setup(tmp_path, method) for method in ("agent", "rules")]
    for case in cases:
        _write(case["files"]["before"], report("fixture_s2026", "b" * 64))
        assert _register(case).returncode == 0
        _finish(case)
        assert _finalize(case).returncode == 0
    out = tmp_path / "unknown-comparison.json"
    proc = _cli("compare", "--reports", *(c["out"] for c in cases), "--out", out)
    assert proc.returncode == 3
    for d in _read(out)["denominators"]:
        assert d["H_healthy"] == 1 and d["undetermined_N"] == 1
        assert d["behavior_unnecessary_H"] == 0 and d["unnecessary_H"] is None


@pytest.mark.parametrize("value", [None, 123, {}, []])
def test_malformed_timestamp_is_input_error_without_traceback(tmp_path, value):
    case = _setup(tmp_path)
    assert _register(case).returncode == 0
    _finish(case)
    timeline = _write(tmp_path / "bad-time.json", {"started_at": value, "finished_at": datetime.now(timezone.utc).isoformat()})
    proc = _finalize(case, "--timeline", timeline)
    assert proc.returncode == 2 and "Traceback" not in proc.stderr


def test_omitted_stages_or_wrong_final_artifact_cannot_complete_timeline(tmp_path):
    case = _setup(tmp_path)
    assert _register(case, "--wall-budget", 300).returncode == 0
    _finish(case, gpu_s=1)
    timeline = _write(tmp_path / "timeline.json",
                      {"started_at": (datetime.now(timezone.utc) - timedelta(seconds=60)).isoformat(),
                       "finished_at": datetime.now(timezone.utc).isoformat()})
    events = _complete_costs(case)
    records = _read(events)
    records[-1]["artifact_sha256"] = "0" * 64
    _write(events, records)
    proc = _finalize(case, "--after", case["files"]["after"], "--events", events, "--timeline", timeline)
    assert proc.returncode == 0, proc.stderr
    result = _read(case["out"])
    assert result["end_to_end_wall"]["budget_status"] == "unknown"
    assert "fixed_eval_receipt_matching_artifact" in result["end_to_end_wall"]["missing"]


@pytest.mark.parametrize("outside", ["external_attempt", "evaluation"])
def test_late_external_attempt_or_evaluation_is_outside_whole_timeline(tmp_path, outside):
    case = _setup(tmp_path)
    assert _register(case, "--wall-budget", 300).returncode == 0
    _finish(case, gpu_s=1)
    finish = datetime.now(timezone.utc)
    timeline = _write(tmp_path / "timeline.json",
                      {"started_at": (finish - timedelta(seconds=60)).isoformat(), "finished_at": finish.isoformat()})
    events = _complete_costs(case)
    if outside == "external_attempt":
        records = _read(events)
        records[1]["started_at"] = (finish + timedelta(seconds=10)).isoformat()
        records[1]["finished_at"] = (finish + timedelta(seconds=11)).isoformat()
        _write(events, records)
    else:
        after = _read(case["files"]["after"])
        after["finished_at"] = (finish + timedelta(seconds=10)).isoformat()
        _write(case["files"]["after"], after)
    proc = _finalize(case, "--after", case["files"]["after"], "--events", events, "--timeline", timeline)
    assert proc.returncode == 2 and not case["out"].exists()
    assert "시간" in proc.stderr or "시각" in proc.stderr
