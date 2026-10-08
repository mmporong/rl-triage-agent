"""P1-A 사람 승인 고리 CLI: 사전등록·제안·승인·거절 뒤 대안·실행(가짜 Isaac)·갱신·종료."""
import importlib.util
import json

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
