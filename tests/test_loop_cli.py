"""P1-A 사람 승인 고리 CLI: 사전등록·제안·승인·거절 뒤 대안·실행(가짜 Isaac)·갱신·종료."""
import importlib.util
import json
from pathlib import Path

import pytest

from rl_triage import probe_loop as L
from test_offline_replay import ROOT

REF = {"P_value": {"critic_change": 0.5}, "P_torque": {"low_speed_saturation": 0.002},
       "P_physics": {"mismatch_count": 0}}


@pytest.fixture
def loop(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("loop_under_test", ROOT / "evals" / "loop.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "LOOPS", tmp_path / "loops")
    return mod


def _fake(measurements, calls):
    def execute(case, probe):
        calls.append(probe)
        return measurements[probe], 1.0, 0
    return execute


def test_reward_fault_is_confirmed_after_one_approved_probe(loop):
    msg = loop.start("h03_s2026", ["reward", "actuator"], 3, 4, "a" * 64, REF)
    assert "P_reward" in msg and "사람 승인 필요" in msg
    _, state, led = loop._load("h03_s2026")
    rid = state["pending"]
    with pytest.raises(L.LedgerError):  # 승인 전 실행은 원장이 막는다
        loop.run("h03_s2026", rid, _fake({}, []))
    led.approve(rid, "human")
    calls = []
    out = loop.run("h03_s2026", rid, _fake({"P_reward": {"track_lin_vel_xy_exp_rel_error": 0.8,
                                                         "track_ang_vel_z_exp_rel_error": 0.0}}, calls))
    assert calls == ["P_reward"] and "종료: confirmed" in out
    state = json.loads((loop.LOOPS / "h03_s2026" / "state.json").read_text(encoding="utf-8"))
    assert state["hypotheses"] == ["reward"] and state["status"] == "confirmed"
    with pytest.raises(L.LedgerError):  # 같은 승인으로 다시 실행할 수 없다
        loop.run("h03_s2026", rid, _fake({"P_reward": {}}, []))


def test_rejected_probe_is_replaced_by_an_alternative_without_spending_budget(loop):
    loop.start("h05_s2026", ["reward", "actuator"], 3, 4, "b" * 64, REF)
    _, state, _ = loop._load("h05_s2026")
    out = loop.reject("h05_s2026", state["pending"], "human", "보상 재계산 probe는 지금 못 돌린다")
    assert "P_torque" in out
    _, state, led = loop._load("h05_s2026")
    assert state["observed"] == {} and state["skipped"] == ["P_reward"]
    rid = state["pending"]
    led.approve(rid, "human")
    out = loop.run("h05_s2026", rid, _fake({"P_torque": {"low_speed_saturation": 0.2}}, []))
    assert "기각=['reward']" in out  # actuator만 남고, 이미 본 확정 probe(P_torque)라 종료
    assert "종료: confirmed" in out


def test_existing_loop_record_is_not_overwritten(loop):
    loop.start("h01_s2026", ["termination", "physics"], 3, 4, "c" * 64, REF)
    with pytest.raises(SystemExit, match="덮어쓰지 않는다"):
        loop.start("h01_s2026", ["termination", "physics"], 3, 4, "c" * 64, REF)


def test_crash_after_consume_blocks_new_runs_until_recovered(loop):
    loop.start("h04_s2027", ["reward", "actuator"], 3, 4, "d" * 64, REF)
    _, state, led = loop._load("h04_s2027")
    rid = state["pending"]
    led.approve(rid, "human")
    led.consume(rid, json.loads((loop.LOOPS / "h04_s2027" / "prereg.json").read_text(encoding="utf-8")), "P_reward")
    # 실행 프로세스가 receipt 전에 죽었다. 같은 승인으로 다시 돌릴 수도, 다른 요청을 돌릴 수도 없다
    with pytest.raises(SystemExit, match="recover"):
        loop.run("h04_s2027", rid, _fake({}, []))
    calls = []

    def artifact(case, probe, prereg):
        calls.append(probe)
        return {"measurement": {"track_lin_vel_xy_exp_rel_error": 0.5, "track_ang_vel_z_exp_rel_error": 0.0},
                "gpu_s": 40.0, "exit_code": 0, "checkpoint_sha256": prereg["checkpoint_sha256"],
                "source": "evals/results/x/probes/h04_s2027.json"}

    out = loop.recover("h04_s2027", artifact)
    assert calls == ["P_reward"] and "산출물로 복구" in out and "종료: confirmed" in out
    _, state, led = loop._load("h04_s2027")
    assert led.orphans() == [] and state["hypotheses"] == ["reward"]


def test_recover_without_artifact_records_unknown(loop):
    loop.start("h06_s2028", ["reward", "actuator"], 3, 4, "e" * 64, REF)
    _, state, led = loop._load("h06_s2028")
    rid = state["pending"]
    led.approve(rid, "human")
    led.consume(rid, json.loads((loop.LOOPS / "h06_s2028" / "prereg.json").read_text(encoding="utf-8")), "P_reward")
    out = loop.recover("h06_s2028", lambda c, p, pr: None)
    assert "unknown (산출물 없음" in out
    _, state, _ = loop._load("h06_s2028")
    assert state["observed"] == {"P_reward": "unknown"} and state["hypotheses"] == ["reward", "actuator"]


def test_two_approved_probes_execute_and_recover_from_separate_artifacts(loop, tmp_path, monkeypatch):
    """실제 CLI 연결을 유지하고 Isaac 프로세스만 대체해 case 단위 skip 회귀를 잡는다."""
    from types import SimpleNamespace

    spec = importlib.util.spec_from_file_location("probe_driver_under_test", ROOT / "evals/run_probes.py")
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    monkeypatch.setattr(loop, "ROOT", tmp_path)
    monkeypatch.setattr(driver, "ROOT", tmp_path)
    monkeypatch.setattr(driver, "PRIVATE", tmp_path / "private")
    monkeypatch.setattr(driver, "KEY_PATH", tmp_path / "absent.json")
    monkeypatch.setattr(driver, "T1_KEY_PATH", tmp_path / "absent.json")
    monkeypatch.setattr(driver, "find_run", lambda *_: tmp_path / "checkpoint")
    monkeypatch.setattr(driver, "last_checkpoint", lambda *_: tmp_path / "model_299.pt")
    measurements = {"P_value": {"critic_change": 0.5}, "P_physics": {"mismatch_count": 0}}
    calls = []

    def subprocess_run(cmd, **kwargs):
        if cmd[1].endswith("run_probes.py"):
            return SimpleNamespace(returncode=driver.main(cmd[2:]))
        jobs = json.loads(Path(cmd[cmd.index("--jobs") + 1]).read_text(encoding="utf-8"))
        for job in jobs["jobs"]:
            calls.extend(job["probes"])
            out = Path(job["output"])
            assert not out.exists()
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({"name": job["name"], "checkpoint": {"sha256": "a" * 64},
                                      "probes": {p: {"measurement": measurements[p], "exit": 0, "elapsed_s": 1.0}
                                                 for p in job["probes"]}}), encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(loop.subprocess, "run", subprocess_run)
    case, tag = "p0ccal_NONE_s7", "sequential"
    loop.start(case, ["optimizer", "physics", "reward"], 3, 4, "a" * 64, REF)
    for _ in range(2):
        _, state, led = loop._load(case)
        rid = state["pending"]
        led.approve(rid, "fake-executor-regression")
        assert loop.main(["run", case, rid, "--tag", tag]) == 0
    prereg, state, _ = loop._load(case)
    assert calls == ["P_value", "P_physics"]
    assert state["observed"] == {"P_value": "normal", "P_physics": "normal"}
    for probe in measurements:
        artifact = loop.file_artifact(case, probe, prereg, tag)
        assert artifact["measurement"] == measurements[probe]
        assert artifact["source"].endswith(f"{case}__{probe}.json")
    with pytest.raises(SystemExit, match="새 --tag"):
        loop.isaac_execute(case, "P_value", tag)
    assert calls == ["P_value", "P_physics"]
    assert driver.main(["--tag", "legacy_batch", "--runs", case, "--probes", *measurements]) == 0
    legacy = tmp_path / "evals/results/legacy_batch/probes" / f"{case}.json"
    assert set(json.loads(legacy.read_text(encoding="utf-8"))["probes"]) == set(measurements)
    before_skip = list(calls)
    assert driver.main(["--tag", "legacy_batch", "--runs", case, "--probes", *measurements]) == 0
    assert calls == before_skip
    before_invalid = list(calls)
    with pytest.raises(SystemExit) as invalid:
        driver.main(["--tag", "invalid", "--runs", case, "--probes", *measurements, "--per-probe-output"])
    assert invalid.value.code == 2
    assert calls == before_invalid
    path = tmp_path / "evals/results" / tag / "probes" / f"{case}__P_physics.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    report["checkpoint"]["sha256"] = "b" * 64
    path.write_text(json.dumps(report), encoding="utf-8")
    assert loop.file_artifact(case, "P_physics", prereg, tag) is None
    report["checkpoint"]["sha256"] = "a" * 64
    report["name"] = "another_case_s7"
    path.write_text(json.dumps(report), encoding="utf-8")
    assert loop.file_artifact(case, "P_physics", prereg, tag) is None

    stale_case = "p0ccal_NONE_s8"
    stale = path.with_name(f"{stale_case}__P_value.json")
    stale.write_text(json.dumps({"name": stale_case, "checkpoint": {"sha256": "a" * 64},
                                 "probes": {"P_value": {"measurement": measurements["P_value"], "exit": 0}}}),
                     encoding="utf-8")
    stale_bytes = stale.read_bytes()
    loop.start(stale_case, ["optimizer", "physics", "reward"], 3, 4, "a" * 64, REF)
    _, state, led = loop._load(stale_case)
    rid = state["pending"]
    led.approve(rid, "fake-executor-regression")
    before_stale = list(calls)
    with pytest.raises(SystemExit, match="새 --tag"):
        loop.main(["run", stale_case, rid, "--tag", tag])
    assert led.requests()[rid]["state"] == "approved" and led.orphans() == []
    assert loop.main(["recover", stale_case, "--tag", tag]) == 0
    assert led.requests()[rid]["state"] == "approved" and calls == before_stale
    assert not any(e["event"] == "receipt" for e in led.events())
    with pytest.raises(SystemExit, match="새 --tag"):
        driver.main(["--tag", tag, "--runs", stale_case, "--probes", "P_value", "--per-probe-output"])
    assert calls == before_stale
    assert loop.main(["run", stale_case, rid, "--tag", "fresh"]) == 0
    assert calls == [*before_stale, "P_value"] and led.orphans() == []
    assert stale.read_bytes() == stale_bytes


@pytest.mark.parametrize("artifact_present", [True, False])
def test_recovered_cli_state_passes_audit(loop, artifact_present):
    spec = importlib.util.spec_from_file_location("audit_under_test", ROOT / "evals/audit_experiment.py")
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    case = "h04_s2027"
    loop.start(case, ["reward", "actuator"], 3, 4, "c" * 64, REF)
    prereg, state, led = loop._load(case)
    rid = state["pending"]
    led.approve(rid, "fake-executor-regression")
    led.consume(rid, prereg, "P_reward")
    artifact = {"measurement": {"track_lin_vel_xy_exp_rel_error": 0.8,
                                "track_ang_vel_z_exp_rel_error": 0.0},
                "gpu_s": 1.0, "exit_code": 0, "checkpoint_sha256": "c" * 64, "source": "fixture.json"}
    loop.recover(case, lambda *_: artifact if artifact_present else None)
    pr, ref, st, events, _ = audit._loop(loop._dir(case))
    result = audit.audit_loop(pr, ref, st, events)
    assert result["incomplete_requests"] == []
    assert result["observed"]["P_reward"] == ("abnormal" if artifact_present else "unknown")


def test_over_budget_cli_receipt_remains_unknown_in_audit(loop):
    spec = importlib.util.spec_from_file_location("audit_under_test", ROOT / "evals/audit_experiment.py")
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    case = "h01_s2026"
    loop.start(case, ["reward", "actuator"], 3, 4, "d" * 64, REF)
    _, state, led = loop._load(case)
    rid = state["pending"]
    led.approve(rid, "fake-executor-regression")
    loop.run(case, rid, lambda *_: ({"track_lin_vel_xy_exp_rel_error": 0.8,
                                   "track_ang_vel_z_exp_rel_error": 0.0}, 121.0, 0))
    pr, ref, st, events, _ = audit._loop(loop._dir(case))
    result = audit.audit_loop(pr, ref, st, events)
    assert result["observed"] == {"P_reward": "unknown"}
    assert result["request_budget_overruns"] == [rid]
