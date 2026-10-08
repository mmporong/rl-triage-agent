"""P1-A 사람 승인 고리 CLI: 사전등록·제안·승인·거절 뒤 대안·실행(가짜 Isaac)·갱신·종료."""
import importlib.util
import json

import pytest

from rl_triage import probe_loop as L
from test_offline_replay import ROOT

REF = {"P_value": {"value_return_corr": 0.8}, "P_torque": {"low_speed_saturation": 0.002},
       "P_slip": {"loaded_foot_speed": 0.02}}


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
    msg = loop.start("h03_s2026", ["reward", "physics", "optimizer"], 3, 4, "a" * 64, REF)
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
    loop.start("h05_s2026", ["reward", "physics", "optimizer"], 3, 4, "b" * 64, REF)
    _, state, _ = loop._load("h05_s2026")
    out = loop.reject("h05_s2026", state["pending"], "human", "보상 재계산 probe는 지금 못 돌린다")
    assert "P_slip" in out
    _, state, led = loop._load("h05_s2026")
    assert state["observed"] == {} and state["skipped"] == ["P_reward"]
    rid = state["pending"]
    led.approve(rid, "human")
    out = loop.run("h05_s2026", rid, _fake({"P_slip": {"loaded_foot_speed": 0.3}}, []))
    assert "기각=['reward', 'optimizer']" in out and "P_slip" in out  # physics만 남고, 이미 본 확정 probe라 종료
    assert "종료: confirmed" in out


def test_existing_loop_record_is_not_overwritten(loop):
    loop.start("h01_s2026", ["termination", "physics"], 3, 4, "c" * 64, REF)
    with pytest.raises(SystemExit, match="덮어쓰지 않는다"):
        loop.start("h01_s2026", ["termination", "physics"], 3, 4, "c" * 64, REF)
