"""E2 교정 제안·승인·실행 단계 테스트. GPU·모델·네트워크 없이 돈다."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "bench"))

from rl_triage import correction_registry as R  # noqa: E402
from rl_triage import probe_loop as L  # noqa: E402
from rl_triage.correction_ledger import CorrectionLedger, LedgerError  # noqa: E402


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


C = _load("correct_cli", "evals/correct.py")
M = _load("correction_model_call", "evals/correction_model_call.py")


def report(ratio, standstill=1.0):
    return {"metrics": {"lin_vel_rmse_mps": ratio * standstill}, "standstill": {"lin_vel_rmse_mps": standstill}}


def make_loop(loops, case="c1", mechanism="reward", status="confirmed", hyps=None, confirm=True):
    """ 확정(또는 다른 상태)된 loop 디렉터리를 만든다."""
    d = loops / case
    d.mkdir(parents=True)
    prereg = {"case": case, "hypotheses": [mechanism], "checkpoint_sha256": "ab" * 32}
    hyps = [mechanism] if hyps is None else hyps
    observed = {}
    led = L.Ledger(d / "ledger.jsonl")
    probe = L.confirming_probe(mechanism) if mechanism in L.MECHANISMS else None
    if confirm and probe:
        rid = led.propose(prereg, probe, 100)["request_id"]
        led.approve(rid, "tester")
        led.consume(rid, prereg, probe)
        led.receipt(rid, "abnormal", {}, 10.0, 0)
        observed[probe] = "abnormal"
    state = {"hypotheses": hyps, "observed": observed, "skipped": [], "budget": 4, "status": status,
             "pending": None, "history": []}
    (d / "prereg.json").write_text(json.dumps(prereg), encoding="utf-8")
    (d / "state.json").write_text(json.dumps(state), encoding="utf-8")
    return d


def fake_exec(verdict="recovered", exit_code=0, gpu_s=100.0, calls=None):
    def _e(request):
        if calls is not None:
            calls.append(request)
        return {"exit_code": exit_code, "gpu_s": gpu_s, "verdict": verdict, "result_sha": "f" * 64,
                "tag": request["tag"]}
    return _e


# --- 등록 표 ---
def test_registry_matches_adapter_targets():
    import correction_adapter as A
    assert {e["correction_id"] for e in R.REGISTRY.values()} == set(A.TARGETS)
    for e in R.REGISTRY.values():
        t = A.TARGETS[e["correction_id"]]
        assert e["adapter_target"] == f"{t['module']}:{t['qualname']}"
    assert set(R.REGISTRY) == {"reward", "actuator", "termination"}


@pytest.mark.parametrize("ratio,mode", [(0.89, "resume"), (0.9, "restart"), (0.91, "restart"), (1.5, "restart")])
def test_reward_standstill_boundary(ratio, mode):
    out = R.recommend_recovery_mode("reward", report(ratio))
    assert out["mode"] == mode and abs(out["standstill_ratio"] - ratio) < 1e-9
    assert out["correction_id"] == "restore_body_frame_reward"


@pytest.mark.parametrize("mech", ["actuator", "termination"])
def test_other_registered_mechanisms_always_resume(mech):
    assert R.recommend_recovery_mode(mech, report(5.0))["mode"] == "resume"


@pytest.mark.parametrize("mech", ["physics", "optimizer", "exploration", "none_supported", "unknown"])
def test_unregistered_mechanism_is_hold(mech):
    out = R.recommend_recovery_mode(mech, report(0.5))
    assert out["mode"] == R.HOLD and out["correction_id"] is None


@pytest.mark.parametrize("bad", [{}, {"metrics": {"lin_vel_rmse_mps": 1}},
                                 report(1.0, standstill=0.0), {"metrics": {"lin_vel_rmse_mps": float("nan")},
                                                              "standstill": {"lin_vel_rmse_mps": 1.0}}])
def test_bad_report_rejected(bad):
    with pytest.raises(ValueError):
        R.recommend_recovery_mode("reward", bad)


def test_registry_digest_changes_with_table():
    changed = {**R.REGISTRY, "reward": {**R.REGISTRY["reward"], "recovery_modes": ["resume"]}}
    assert R.registry_digest(changed) != R.registry_digest()


# --- propose (CLI 함수) ---
def test_propose_requires_confirmed(tmp_path):
    for status in ("open", "unidentifiable_x"):
        make_loop(tmp_path, case=f"c_{status}", status=status, hyps=["reward", "physics"], confirm=False)
        with pytest.raises(LedgerError):
            C.propose(f"c_{status}", report(1.0), loops=tmp_path)
        assert CorrectionLedger(tmp_path / f"c_{status}" / "correction_ledger.jsonl").requests() == {}


def test_propose_requires_confirming_receipt(tmp_path):
    make_loop(tmp_path, confirm=False)  # 상태만 confirmed로 조작
    with pytest.raises(LedgerError):
        C.propose("c1", report(1.0), loops=tmp_path)


def test_propose_records_evidence(tmp_path):
    make_loop(tmp_path)
    out = C.propose("c1", report(0.95), loops=tmp_path, source_report_sha="aa")
    assert out["decision"] == "proposed" and out["recovery_mode"] == "restart"
    assert out["correction_id"] == "restore_body_frame_reward"
    assert out["protocol_case_arm"] == {"case": "case_01", "arm": "corrected"}
    ev = out["evidence"]
    assert ev["confirming_probe"] == "P_reward" and ev["standstill_ratio"] == pytest.approx(0.95)
    assert ev["rule_mode"] == "restart" and ev["source_report_sha256"] == "aa" and ev["confirming_request_id"]


@pytest.mark.parametrize("mech", ["physics", "optimizer", "exploration"])
def test_unsupported_mechanism_holds_without_request(tmp_path, mech):
    make_loop(tmp_path, mechanism=mech)
    out = C.propose("c1", report(1.0), loops=tmp_path)
    assert out["decision"] == R.HOLD
    st = C.status("c1", tmp_path)
    assert st["requests"] == {} and len(st["holds"]) == 1


def test_none_supported_holds(tmp_path):
    make_loop(tmp_path, status="none_supported", hyps=[], confirm=False)
    assert C.propose("c1", report(1.0), loops=tmp_path)["decision"] == R.HOLD
    assert C.status("c1", tmp_path)["requests"] == {}


# --- 원장 ---
def test_duplicate_effect_key_rejected_and_reproposal_after_reject(tmp_path):
    make_loop(tmp_path)
    first = C.propose("c1", report(0.5), loops=tmp_path)
    with pytest.raises(LedgerError):
        C.propose("c1", report(0.5), loops=tmp_path)
    led = CorrectionLedger(tmp_path / "c1" / "correction_ledger.jsonl")
    led.approve(first["request_id"], "me")
    with pytest.raises(LedgerError):  # 승인 후에도 거부
        C.propose("c1", report(0.5), loops=tmp_path)


def test_propose_must_match_registry(tmp_path):
    led = CorrectionLedger(tmp_path / "l.jsonl")
    ok = dict(loop_state_digest="s", confirmed_mechanism="reward", receipts_digest="r",
              correction_id="restore_body_frame_reward", recovery_mode="restart",
              protocol_case_arm={"case": "case_01", "arm": "corrected"}, budget_gpu_s=10)
    for bad in ({"correction_id": "restore_motor_torque_speed"}, {"recovery_mode": "scratch"},
                {"confirmed_mechanism": "physics"}, {"protocol_case_arm": {"case": "case_02", "arm": "corrected"}},
                {"budget_gpu_s": -1}):
        with pytest.raises(LedgerError):
            led.propose(**{**ok, **bad})
    assert led.requests() == {}
    led.propose(**ok)
    assert len(led.requests()) == 1
    with pytest.raises(LedgerError):  # actuator는 restart 미등록
        led.propose("s2", "actuator", "r", "restore_motor_torque_speed", "restart",
                    {"case": "case_02", "arm": "corrected"}, 10)


def test_run_requires_approval(tmp_path):
    make_loop(tmp_path)
    rid = C.propose("c1", report(0.5), loops=tmp_path)["request_id"]
    calls = []
    with pytest.raises(LedgerError):
        C.run("c1", rid, fake_exec(calls=calls), loops=tmp_path)
    assert calls == []


def test_approved_run_consumed_once_with_receipt(tmp_path):
    make_loop(tmp_path)
    rid = C.propose("c1", report(0.5), loops=tmp_path)["request_id"]
    led = CorrectionLedger(tmp_path / "c1" / "correction_ledger.jsonl")
    led.approve(rid, "me")
    calls = []
    rec = C.run("c1", rid, fake_exec(calls=calls), loops=tmp_path, tag="t1")
    assert rec["verdict"] == "recovered" and rec["tag"] == "t1" and rec["result_sha"] == "f" * 64
    assert calls[0]["correction_id"] == "restore_body_frame_reward"
    assert calls[0]["recovery_mode"] == "resume" and calls[0]["protocol_case_arm"]["case"] == "case_01"
    with pytest.raises(LedgerError):  # 두 번째 실행은 소비됨
        C.run("c1", rid, fake_exec(calls=calls), loops=tmp_path)
    assert len(calls) == 1
    assert C.status("c1", tmp_path)["requests"][rid]["state"] == "done"


def test_rejected_request_cannot_run_or_be_approved(tmp_path):
    make_loop(tmp_path)
    rid = C.propose("c1", report(0.5), loops=tmp_path)["request_id"]
    led = CorrectionLedger(tmp_path / "c1" / "correction_ledger.jsonl")
    led.reject(rid, "me", "싫다")
    calls = []
    with pytest.raises(LedgerError):
        C.run("c1", rid, fake_exec(calls=calls), loops=tmp_path)
    with pytest.raises(LedgerError):
        led.approve(rid, "me")
    assert calls == []


def test_loop_state_change_after_approval_blocks_run(tmp_path):
    d = make_loop(tmp_path)
    rid = C.propose("c1", report(0.5), loops=tmp_path)["request_id"]
    CorrectionLedger(d / "correction_ledger.jsonl").approve(rid, "me")
    st = json.loads((d / "state.json").read_text(encoding="utf-8"))
    st["budget"] = 99
    (d / "state.json").write_text(json.dumps(st), encoding="utf-8")
    with pytest.raises(LedgerError):
        C.run("c1", rid, fake_exec(), loops=tmp_path)


def test_failed_or_over_budget_receipt_is_unknown(tmp_path):
    for n, kw in enumerate(({"exit_code": 1}, {"gpu_s": 10 ** 6})):
        case = f"k{n}"
        make_loop(tmp_path, case=case)
        rid = C.propose(case, report(0.5), loops=tmp_path)["request_id"]
        CorrectionLedger(tmp_path / case / "correction_ledger.jsonl").approve(rid, "me")
        assert C.run(case, rid, fake_exec(**kw), loops=tmp_path)["verdict"] == "unknown"


def test_executor_crash_leaves_orphan_and_recover_closes(tmp_path):
    d = make_loop(tmp_path)
    rid = C.propose("c1", report(0.5), loops=tmp_path)["request_id"]
    led = CorrectionLedger(d / "correction_ledger.jsonl")
    led.approve(rid, "me")

    def boom(request):
        raise RuntimeError("죽음")

    with pytest.raises(RuntimeError):
        C.run("c1", rid, boom, loops=tmp_path)
    assert led.orphans() == [rid]
    with pytest.raises(LedgerError):  # 고아가 있으면 새 실행 막힘
        C.run("c1", rid, fake_exec(), loops=tmp_path)
    ev = led.recover(rid, None)
    assert ev["verdict"] == "unknown" and led.orphans() == []
    # 산출물로 복구
    make_loop(tmp_path, case="c2")
    rid2 = C.propose("c2", report(0.5), loops=tmp_path)["request_id"]
    led2 = CorrectionLedger(tmp_path / "c2" / "correction_ledger.jsonl")
    led2.approve(rid2, "me")
    with pytest.raises(RuntimeError):
        C.run("c2", rid2, boom, loops=tmp_path)
    ev2 = led2.recover(rid2, {"tag": "t", "verdict": "recovered", "result_sha": "a", "source": "x"})
    assert ev2["verdict"] == "recovered" and ev2["recovered_from"] == "x"


def test_receipt_rejects_unknown_verdict(tmp_path):
    led = CorrectionLedger(tmp_path / "l.jsonl")
    rid = led.propose("s", "termination", "r", "restore_configured_timeout", "resume",
                      {"case": "case_03", "arm": "corrected"}, 10)["request_id"]
    led.approve(rid, "me")
    led.consume(rid, "s")
    with pytest.raises(LedgerError):
        led.receipt(rid, "t", "great", None, 1.0, 0)


def test_registry_change_invalidates_request(tmp_path):
    path = tmp_path / "l.jsonl"
    led = CorrectionLedger(path)
    rid = led.propose("s", "reward", "r", "restore_body_frame_reward", "restart",
                      {"case": "case_01", "arm": "corrected"}, 10)["request_id"]
    changed = {**R.REGISTRY, "reward": {**R.REGISTRY["reward"], "recovery_modes": ["restart", "resume", "x"]}}
    with pytest.raises(LedgerError):
        CorrectionLedger(path, changed).approve(rid, "me")


# --- 원장 손상·순서 역전 ---
def _ledger_with_request(tmp_path):
    path = tmp_path / "l.jsonl"
    led = CorrectionLedger(path)
    rid = led.propose("s", "reward", "r", "restore_body_frame_reward", "restart",
                      {"case": "case_01", "arm": "corrected"}, 10)["request_id"]
    led.approve(rid, "me")
    return path, led, rid


def test_corrupted_ledger_fails(tmp_path):
    path, led, rid = _ledger_with_request(tmp_path)
    with path.open("a", encoding="utf-8") as f:
        f.write("{깨진 줄\n")
    with pytest.raises(LedgerError):
        led.requests()


def test_reordered_ledger_fails(tmp_path):
    path, led, rid = _ledger_with_request(tmp_path)
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(reversed(lines)) + "\n", encoding="utf-8")
    with pytest.raises(LedgerError):
        led.requests()


def test_consumed_without_approval_fails(tmp_path):
    path, led, rid = _ledger_with_request(tmp_path)
    lines = path.read_text(encoding="utf-8").splitlines()
    forged = {**json.loads(lines[1]), "event": "consumed"}
    path.write_text(lines[0] + "\n" + json.dumps(forged) + "\n", encoding="utf-8")
    with pytest.raises(LedgerError):
        led.requests()


def test_duplicate_event_and_timestamp_reversal_fail(tmp_path):
    path, led, rid = _ledger_with_request(tmp_path)
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines + [lines[1]]) + "\n", encoding="utf-8")  # 승인 중복
    with pytest.raises(LedgerError):
        led.requests()
    early = {**json.loads(lines[1]), "at": "2000-01-01T00:00:00+00:00"}
    path.write_text(lines[0] + "\n" + json.dumps(early) + "\n", encoding="utf-8")
    with pytest.raises(LedgerError):
        led.requests()


# --- 모델 제안 검증(호출 없음) ---
EVIDENCE = {"mechanism": "reward", "facts": {"standstill_ratio": 0.95, "confirming_probe": "P_reward"},
            "rule_result": {"correction_id": "restore_body_frame_reward", "mode": "restart"}}


def prop(**kw):
    return {**{"correction_id": "restore_body_frame_reward", "recovery_mode": "restart",
               "cited_facts": ["standstill_ratio"]}, **kw}


def test_model_proposal_valid_and_rule_agreement():
    out = M.validate_model_proposal(prop(), EVIDENCE)
    assert out["accepted"] and out["agrees_with_rule"]
    out = M.validate_model_proposal(prop(recovery_mode="resume"), EVIDENCE)
    assert out["accepted"] and not out["agrees_with_rule"]


@pytest.mark.parametrize("bad", [
    prop(correction_id="rewrite_reward_code"),                 # 등록 표 밖
    prop(correction_id="restore_motor_torque_speed"),          # 다른 메커니즘의 교정
    prop(recovery_mode="scratch"),
    prop(cited_facts=[]),                                      # 인용 없음
    prop(cited_facts=["invented_fact"]),                       # 근거에 없는 인용
    prop(correction_id=["x"]),
    {"correction_id": "restore_body_frame_reward", "recovery_mode": "restart"},
    "not a dict",
])
def test_model_proposal_rejected(bad):
    out = M.validate_model_proposal(bad, EVIDENCE)
    assert not out["accepted"] and out["reasons"] and not out["agrees_with_rule"]


def test_model_proposal_unregistered_mechanism():
    assert not M.validate_model_proposal(prop(), {**EVIDENCE, "mechanism": "physics"})["accepted"]


def test_build_messages_has_no_network_and_carries_facts():
    msgs = M.build_messages(EVIDENCE)
    assert [m["role"] for m in msgs] == ["system", "user"]
    body = json.loads(msgs[1]["content"])
    assert body["facts"] == EVIDENCE["facts"] and "restore_body_frame_reward" in msgs[1]["content"]
