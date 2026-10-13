"""E2 교정 제안·승인·실행 단계 테스트. GPU·모델·네트워크 없이 돈다."""
import hashlib
import importlib.util
import json
import sys
import types
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
M = C.M  # correct.py가 불러온 같은 모듈


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


# --- E3 모델 교정 제안: skill 체크리스트 근거, 호출 없음 ---
CHECK_OK = {"1": "eval_metrics", "2": "confirmed_mechanism", "3": "confirming_probe", "4": "eval_metrics",
            "5": "unmet", "6": "registered_correction"}


def answer(**kw):
    base = {"correction_id": "restore_body_frame_reward", "recovery_mode": "restart",
            "cited_facts": ["standstill_ratio", "confirming_probe_outcome"], "checklist": dict(CHECK_OK),
            "hold_reason": None}
    return json.dumps({**base, **kw}, ensure_ascii=False)


FIXTURES = {  # 이름: (모델 응답 content, accepted, decision)
    "normal": (answer(), True, "propose"),
    "out_of_registry": (answer(correction_id="rewrite_reward_code"), False, None),
    "no_citation": (answer(cited_facts=[]), False, None),
    "mechanism_mismatch": (answer(correction_id="restore_motor_torque_speed", recovery_mode="resume"), False, None),
    "broken_json": ('{"correction_id": "restore_body_frame_reward", "recovery_mode": ', False, None),
    "hold": (answer(correction_id=None, recovery_mode=None, cited_facts=["probe_observations"],
                    hold_reason="체크리스트 5번을 충족하는 사실이 없다"), True, R.HOLD),
}


def packet_for(tmp_path, ratio=0.95, mechanism="reward", sha="aa", case="c1"):
    make_loop(tmp_path, case=case, mechanism=mechanism)
    return C.evidence_packet(case, report(ratio), loops=tmp_path, source_report_sha=sha)


def call_record(packet, content, **kw):
    return {"schema": M.RECORD_SCHEMA, "packet_digest": M.packet_digest(packet), "exit_code": 0,
            "messages_sha256": L.digest(M.build_messages(packet)), "skill_sha256": M.SKILL_SHA256,
            "request": {"model": "fixture-model", "temperature": 0}, "provider_model": "fixture-model",
            "response_id": "r1", "content": content, **kw}


def test_skill_reference_pinned_and_checklist_verbatim():
    src =json.loads((M.SKILL_DIR / "SOURCE.json").read_text(encoding="utf-8"))
    assert src["commit_sha"] == M.SKILL_COMMIT and src["files"]["SKILL.md"]["sha256"] == M.SKILL_SHA256
    for name, meta in src["files"].items():  # 상류 바이트 그대로(LICENSE 포함)
        assert hashlib.sha256((M.SKILL_DIR / name).read_bytes()).hexdigest() == meta["sha256"]
    assert src["license"] == "BSD-3-Clause"
    assert "SPDX-License-Identifier: BSD-3-Clause" in (M.SKILL_DIR / "LICENSE").read_text(encoding="utf-8")
    items = M.skill_checklist()
    assert list(items) == ["1", "2", "3", "4", "5", "6"]
    assert items["2"].startswith("The failed layer is classified as environment, reward, reset, physics")
    assert items["6"] == "Any recommended next run changes only one variable."


def test_skill_tamper_detected_and_crlf_tolerated(tmp_path):
    data = (M.SKILL_DIR / "SKILL.md").read_bytes()
    (tmp_path / "SKILL.md").write_bytes(data.replace(b"\n", b"\r\n"))
    assert M.skill_checklist(tmp_path) == M.skill_checklist()
    (tmp_path / "SKILL.md").write_bytes(data.replace(b"only one variable", b"two variables"))
    with pytest.raises(ValueError):
        M.skill_checklist(tmp_path)


def test_packet_carries_facts_checklist_costs_but_messages_hide_case(tmp_path):
    pk = packet_for(tmp_path, case="hidden_case_name")
    f = pk["facts"]
    assert f["confirmed_mechanism"] == "reward" and f["confirming_probe"] == "P_reward"
    assert f["confirming_probe_outcome"] == "abnormal" and f["standstill_ratio"] == pytest.approx(0.95)
    assert f["rule_mode"] == "restart" and f["registered_recovery_modes"] == ["restart", "resume"]
    arms = {"clean_reference", "corrected", "fault_sham"}
    assert set(f["cost_train_300_updates_parent_wall_s"]) == arms and set(f["cost_fixed_eval_parent_wall_s"]) == arms
    assert pk["cost_basis_source"] == M.COST_SOURCE
    assert pk["checklist"] == M.skill_checklist()
    assert pk["rule_result"] == {"correction_id": "restore_body_frame_reward", "mode": "restart"}
    msgs = M.build_messages(pk)
    assert [m["role"] for m in msgs] == ["system", "user"]
    body = json.loads(msgs[1]["content"])
    assert body["facts"] == f and body["skill_checklist"] == pk["checklist"]
    assert "hidden_case_name" not in msgs[1]["content"] and pk["loop_state_digest"] not in msgs[1]["content"]


def test_evidence_requires_confirmed_loop(tmp_path):
    make_loop(tmp_path, status="open", hyps=["reward", "physics"], confirm=False)
    with pytest.raises(LedgerError):
        C.evidence_packet("c1", report(1.0), loops=tmp_path)


@pytest.mark.parametrize("name", list(FIXTURES))
def test_fixture_responses(tmp_path, name):
    content, accepted, decision = FIXTURES[name]
    pk = packet_for(tmp_path)
    out = M.review_call_record(call_record(pk, content), pk, "cc")
    assert out["accepted"] is accepted and out["decision"] == decision
    assert out["record_sha256"] == "cc" and out["packet_digest"] == M.packet_digest(pk)
    assert (out["reasons"] == []) is accepted
    if not accepted:
        assert not out["agrees_with_rule"]
    if name == "normal":
        assert out["agrees_with_rule"] and out["unmet_checklist"] == ["5"]
    if name == "hold":
        assert not out["agrees_with_rule"]  # 결정적 규칙은 restart
    if name == "broken_json":
        assert out["proposal"] is None and "JSON" in out["reasons"][0]


@pytest.mark.parametrize("bad", [
    {"checklist": {k: v for k, v in CHECK_OK.items() if k != "6"}},  # 체크리스트 누락
    {"checklist": {**CHECK_OK, "2": "invented_fact"}},               # 체크리스트가 없는 사실 인용
    {"cited_facts": ["invented_fact"]},
    {"recovery_mode": "scratch"},
    {"correction_id": ["x"]},
    {"hold_reason": ""},
    {"hold_reason": "보류", "correction_id": "restore_body_frame_reward"},
])
def test_more_rejections(tmp_path, bad):
    out = M.validate_model_proposal(json.loads(answer(**bad)), packet_for(tmp_path))
    assert not out["accepted"] and out["reasons"]


def test_key_set_fence_and_rule_disagreement(tmp_path):
    pk = packet_for(tmp_path)
    obj = json.loads(answer())
    assert not M.validate_model_proposal({**obj, "approve": True}, pk)["accepted"]
    assert not M.validate_model_proposal({k: v for k, v in obj.items() if k != "hold_reason"}, pk)["accepted"]
    assert not M.validate_model_proposal("not a dict", pk)["accepted"]
    assert M.review_call_record(call_record(pk, "```json\n" + answer() + "\n```"), pk)["accepted"]
    disagree = M.validate_model_proposal(json.loads(answer(recovery_mode="resume")), pk)
    assert disagree["accepted"] and not disagree["agrees_with_rule"]


def test_review_refuses_stale_failed_or_foreign_record(tmp_path):
    pk = packet_for(tmp_path)
    for rec in (call_record(pk, answer(), packet_digest="0" * 64),
                call_record(pk, answer(), messages_sha256="0" * 64),  # 다른 프롬프트로 만든 응답
                call_record(pk, answer(), skill_sha256="0" * 64),
                call_record(pk, None, exit_code=2, infra_error="empty_or_truncated_response"),
                {**call_record(pk, answer()), "schema": "active_model_call_v1"}):
        with pytest.raises(ValueError):
            M.review_call_record(rec, pk)


def test_review_marks_rule_visibility_and_caps_stored_output(tmp_path):
    pk = packet_for(tmp_path)
    out = M.review_call_record(call_record(pk, answer()), pk)
    assert out["rule_visible_to_model"] and out["messages_sha256"] == L.digest(M.build_messages(pk))
    big = M.review_call_record(call_record(pk, answer(hold_reason="x" * 5000, correction_id=None,
                                                      recovery_mode=None)), pk)
    assert big["proposal"] is None and big["proposal_omitted"] == "too_large" and big["decision"] == R.HOLD


def test_propose_attaches_model_review_but_keeps_rule_and_needs_approval(tmp_path):
    pk = packet_for(tmp_path)
    out = C.propose("c1", report(0.95), loops=tmp_path, source_report_sha="aa",
                    model_record=call_record(pk, answer(recovery_mode="resume")), model_record_sha="bb")
    assert out["decision"] == "proposed" and out["recovery_mode"] == "restart"  # 모델 값이 아니라 규칙 값
    mp = out["evidence"]["model_proposal"]
    assert mp["accepted"] and not mp["agrees_with_rule"] and mp["proposal"]["recovery_mode"] == "resume"
    assert mp["record_sha256"] == "bb"
    assert C.status("c1", tmp_path)["requests"][out["request_id"]]["state"] == "pending"
    calls = []
    with pytest.raises(LedgerError):  # 사람 승인 전에는 실행할 수 없다
        C.run("c1", out["request_id"], fake_exec(calls=calls), loops=tmp_path)
    assert calls == []


def test_rejected_model_output_is_recorded_on_request(tmp_path):
    pk = packet_for(tmp_path)
    out = C.propose("c1", report(0.95), loops=tmp_path, source_report_sha="aa",
                    model_record=call_record(pk, FIXTURES["broken_json"][0]))
    mp = out["evidence"]["model_proposal"]
    assert out["decision"] == "proposed" and not mp["accepted"] and mp["reasons"]


def test_stale_model_record_writes_nothing(tmp_path):
    pk = packet_for(tmp_path, sha="aa")
    with pytest.raises(ValueError):  # report SHA가 바뀌면 packet digest가 달라진다
        C.propose("c1", report(0.95), loops=tmp_path, source_report_sha="zz", model_record=call_record(pk, answer()))
    st = C.status("c1", tmp_path)
    assert st["requests"] == {} and st["holds"] == []


def test_model_record_on_unconfirmed_loop_is_refused(tmp_path):
    pk = packet_for(tmp_path)
    make_loop(tmp_path, case="c2", status="none_supported", hyps=[], confirm=False)
    with pytest.raises(ValueError):
        C.propose("c2", report(1.0), loops=tmp_path, model_record=call_record(pk, answer()))
    assert C.status("c2", tmp_path)["holds"] == []


def test_without_model_record_evidence_unchanged(tmp_path):
    make_loop(tmp_path)
    out = C.propose("c1", report(0.95), loops=tmp_path, source_report_sha="aa")
    assert "model_proposal" not in out["evidence"]


def test_unregistered_mechanism_hold_keeps_model_review(tmp_path):
    pk = packet_for(tmp_path, mechanism="physics")
    assert pk["rule_result"]["mode"] == R.HOLD and pk["facts"]["registered_correction"] is None
    out = C.propose("c1", report(0.95), loops=tmp_path, source_report_sha="aa",
                    model_record=call_record(pk, FIXTURES["hold"][0]))
    assert out["decision"] == R.HOLD and out["model_proposal"]["decision"] == R.HOLD
    assert out["model_proposal"]["agrees_with_rule"]
    st = C.status("c1", tmp_path)
    assert st["requests"] == {} and st["holds"][0]["evidence"]["model_proposal"]["decision"] == R.HOLD


# --- call: 가짜 openai 모듈로만 돈다(네트워크·키 없음) ---
def fake_openai(monkeypatch, content=None, raise_connection=False):
    mod = types.ModuleType("openai")

    class APIConnectionError(Exception):
        pass

    class APITimeoutError(Exception):
        pass

    class APIStatusError(Exception):
        status_code = 500

    seen = {"requests": 0}

    def create(**kw):
        seen["requests"] += 1
        seen["request"] = kw
        if raise_connection:
            raise APIConnectionError("no network in tests")
        msg = types.SimpleNamespace(content=content, reasoning_content=None)
        return types.SimpleNamespace(id="resp-1", model="fake-model",
                                     choices=[types.SimpleNamespace(message=msg, finish_reason="stop")],
                                     usage=types.SimpleNamespace(prompt_tokens=11, completion_tokens=7,
                                                                 total_tokens=18))

    class OpenAI:
        def __init__(self, **kw):
            seen["client"] = kw
            self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=create))

    mod.OpenAI, mod.APIConnectionError = OpenAI, APIConnectionError
    mod.APITimeoutError, mod.APIStatusError = APITimeoutError, APIStatusError
    monkeypatch.setitem(sys.modules, "openai", mod)
    monkeypatch.setenv("NVIDIA_API_KEY", "placeholder-for-test")
    return seen


def write_packet(tmp_path, pk, name="packet.json"):
    p = tmp_path / name
    p.write_text(json.dumps(pk, ensure_ascii=False, indent=1), encoding="utf-8")
    return p


def test_call_records_request_response_and_tokens(tmp_path, monkeypatch):
    pk = packet_for(tmp_path)
    seen = fake_openai(monkeypatch, content=answer())
    out = tmp_path / "call" / "record.json"
    assert M.call(write_packet(tmp_path, pk), out, "fixture-model") == 0
    text = out.read_text(encoding="utf-8")
    rec = json.loads(text)
    assert rec["schema"] == M.RECORD_SCHEMA and rec["request"]["temperature"] == 0
    assert rec["packet_digest"] == M.packet_digest(pk) and rec["skill_sha256"] == M.SKILL_SHA256
    assert (rec["input_tokens"], rec["output_tokens"], rec["total_tokens"]) == (11, 7, 18)
    assert rec["exit_code"] == 0 and rec["execution_wall_s"] >= 0 and rec["content"] == answer()
    assert seen["request"]["messages"] == M.build_messages(pk) and seen["client"]["max_retries"] == 0
    assert "placeholder-for-test" not in text
    assert M.review_call_record(rec, pk)["accepted"]
    with pytest.raises(FileExistsError):  # 기록이 있으면 호출 전에 멈춘다
        M.call(write_packet(tmp_path, pk), out, "fixture-model")
    assert seen["requests"] == 1


def test_call_connection_error_is_retryable_and_review_refuses(tmp_path, monkeypatch):
    pk = packet_for(tmp_path)
    fake_openai(monkeypatch, raise_connection=True)
    out = tmp_path / "r.json"
    assert M.call(write_packet(tmp_path, pk), out, "m") == 2
    rec = json.loads(out.read_text(encoding="utf-8"))
    assert rec["infra_error"] == "APIConnectionError" and rec["retryable"] and rec["content"] is None
    with pytest.raises(ValueError):
        M.review_call_record(rec, pk)


def test_call_refuses_packet_with_altered_checklist(tmp_path, monkeypatch):
    pk = packet_for(tmp_path)
    seen = fake_openai(monkeypatch, content=answer())
    bad = {**pk, "checklist": {**pk["checklist"], "6": "Change everything at once."}}
    with pytest.raises(ValueError):
        M.call(write_packet(tmp_path, bad), tmp_path / "r.json", "m")
    assert seen["requests"] == 0


def test_cli_evidence_then_propose_with_model_record(tmp_path, monkeypatch):
    make_loop(tmp_path)
    monkeypatch.setattr(C, "LOOPS", tmp_path)
    rp = tmp_path / "report.json"
    rp.write_text(json.dumps(report(0.95)), encoding="utf-8")
    pk_path = tmp_path / "packet.json"
    assert C.main(["evidence", "c1", "--source-report", str(rp), "--output", str(pk_path)]) == 0
    assert C.main(["evidence", "c1", "--source-report", str(rp), "--output", str(pk_path)]) == 2  # 덮어쓰기 거부
    pk = json.loads(pk_path.read_text(encoding="utf-8"))
    recp = tmp_path / "record.json"
    recp.write_text(json.dumps(call_record(pk, answer())), encoding="utf-8")
    assert C.main(["propose", "c1", "--source-report", str(rp), "--model-proposal", str(recp)]) == 0
    req = next(iter(C.status("c1", tmp_path)["requests"].values()))
    assert req["state"] == "pending" and req["recovery_mode"] == "restart"
    assert req["evidence"]["model_proposal"]["accepted"] and req["evidence"]["model_proposal"]["record_sha256"]
