"""P1-A 고리 핵심: probe 선택·갱신·종료 판정과 승인 원장(한 번만 소비, digest 묶음, 재시작, 불명 결과)."""
import pytest

from rl_triage import probe_loop as L
from rl_triage import triage_tools as T

PREREG = {"case": "h03_s2026", "hypotheses": ["reward", "physics"], "checkpoint_sha256": "a" * 64}


def test_probe_expectations_cover_every_mechanism_and_each_has_a_confirming_probe():
    assert list(L.MECHANISMS) == T.MECHANISMS
    for spec in L.PROBES.values():
        assert set(spec["expect"]) == set(L.MECHANISMS)
    assert {m: L.confirming_probe(m) for m in L.MECHANISMS} == {
        "reward": "P_reward", "actuator": "P_torque", "exploration": "P_noise",
        "optimizer": "P_value", "physics": "P_slip", "termination": "P_episode"}


def test_every_pair_of_mechanisms_can_be_separated_by_some_probe():
    for a in L.MECHANISMS:
        for b in L.MECHANISMS:
            if a < b:
                assert L.select_probe([a, b]) is not None, (a, b)


def test_selection_update_and_status():
    assert L.select_probe(["reward", "physics"]) == "P_reward"  # 같은 비용이면 이름순
    keep, dropped = L.update(["reward", "physics"], "P_reward", "abnormal")
    assert (keep, dropped) == (["reward"], ["physics"])
    assert L.update(["reward", "physics"], "P_reward", "unknown") == (["reward", "physics"], [])
    assert L.update(["optimizer", "physics"], "P_value", "normal") == (["physics"], ["optimizer"])
    assert L.status(["reward"], {"P_reward": "abnormal"}, budget_left=3) == "confirmed"
    assert L.status(["reward"], {}, budget_left=3) == "open"  # 하나 남아도 확정 probe를 보기 전엔 확정 아님
    assert L.status(["reward", "physics"], {}, budget_left=0) == "unidentifiable"
    assert L.status([], {"P_reward": "normal"}, budget_left=3) == "none_supported"
    with pytest.raises(ValueError):
        L.update(["reward"], "P_reward", "maybe")


def _approved(tmp_path):
    led = L.Ledger(tmp_path / "ledger.jsonl")
    rid = led.propose(PREREG, "P_reward", budget_gpu_s=120)["request_id"]
    led.approve(rid, approver="human")
    return led, rid


def test_approval_is_consumed_once_and_bound_to_prereg_probe_and_args(tmp_path):
    led, rid = _approved(tmp_path)
    with pytest.raises(L.LedgerError, match="다르다"):
        led.consume(rid, {**PREREG, "hypotheses": ["reward"]}, "P_reward")
    with pytest.raises(L.LedgerError, match="다르다"):
        led.consume(rid, PREREG, "P_slip")
    led.consume(rid, PREREG, "P_reward")
    with pytest.raises(L.LedgerError, match="이미 소비"):
        led.consume(rid, PREREG, "P_reward")
    led.receipt(rid, "abnormal", {"ratio": 3.1}, gpu_s=58.0, exit_code=0)
    again = led.propose(PREREG, "P_reward", budget_gpu_s=120)["request_id"]  # 같은 실험도 새 요청·새 승인
    assert again != rid
    with pytest.raises(L.LedgerError):
        led.consume(again, PREREG, "P_reward")


def test_unapproved_rejected_and_unknown_results(tmp_path):
    led = L.Ledger(tmp_path / "ledger.jsonl")
    rid = led.propose(PREREG, "P_noise", budget_gpu_s=60)["request_id"]
    with pytest.raises(L.LedgerError, match="승인되지 않았"):
        led.consume(rid, PREREG, "P_noise")
    led.reject(rid, approver="human", reason="budget")
    with pytest.raises(L.LedgerError):
        led.approve(rid, approver="human")
    led2, rid2 = _approved(tmp_path / "b")
    led2.consume(rid2, PREREG, "P_reward")
    ev = led2.receipt(rid2, "abnormal", {}, gpu_s=300.0, exit_code=None)  # timeout은 실패가 아니라 불명
    assert ev["outcome"] == "unknown"


def test_state_is_rebuilt_from_the_ledger_file_after_restart(tmp_path):
    led, rid = _approved(tmp_path)
    led.consume(rid, PREREG, "P_reward")
    restarted = L.Ledger(tmp_path / "ledger.jsonl")
    assert restarted.requests()[rid]["state"] == "running"
    with pytest.raises(L.LedgerError):
        restarted.consume(rid, PREREG, "P_reward")


def test_next_probe_asks_for_the_confirming_probe_when_one_hypothesis_is_left():
    assert L.select_probe(["reward"]) is None  # 가를 대상이 없으면 None
    assert L.next_probe(["reward"], {}) == "P_reward"
    assert L.next_probe(["reward"], {"P_reward": "abnormal"}) is None
    assert L.next_probe(["reward", "physics"], {}) == "P_reward"
    assert L.next_probe(["reward", "physics"], {"P_reward": "unknown"}) == "P_slip"


def test_receipt_cannot_be_written_twice(tmp_path):
    led, rid = _approved(tmp_path)
    led.consume(rid, PREREG, "P_reward")
    led.receipt(rid, "normal", {}, gpu_s=1.0, exit_code=0)
    with pytest.raises(L.LedgerError, match="실행 중"):
        led.receipt(rid, "abnormal", {}, gpu_s=1.0, exit_code=0)
    assert not (tmp_path / "ledger.jsonl.lock").exists()  # 쓰기가 끝나면 잠금이 풀린다


def _outcomes_for(true_mechanism):
    """참 범주가 원인일 때 예상대로 나오는 probe 결과표(예상 없음은 normal)."""
    return {p: (spec["expect"].get(true_mechanism) or "normal") for p, spec in L.PROBES.items()}


@pytest.mark.parametrize("truth", L.MECHANISMS)
def test_every_strategy_confirms_the_truth_when_probes_behave_as_expected(truth):
    out = _outcomes_for(truth)
    for strategy in ("exhaustive", "fixed", "random"):
        r = L.simulate(strategy, out, seed=1)
        assert (r["status"], r["conclusion"]) == ("confirmed", truth), (strategy, r)
    # 진단 순위에 참 범주가 상위 3위 안에 있으면 가르기 방식도 확정하고, 전수보다 probe를 적게 쓴다
    ranking = [truth] + [m for m in L.MECHANISMS if m != truth]
    r = L.simulate("discriminate", out, ranking=ranking[1:3] + [truth])
    assert (r["status"], r["conclusion"]) == ("confirmed", truth) and r["probes_used"] <= 3, r
    assert L.simulate("exhaustive", out)["probes_used"] == len(L.PROBES)


def test_discriminate_cannot_recover_a_truth_outside_its_hypotheses():
    out = _outcomes_for("physics")
    r = L.simulate("discriminate", out, ranking=["reward", "optimizer", "exploration"])
    assert r["status"] == "none_supported" and r["conclusion"] is None


def test_unknown_probe_results_can_end_unidentifiable():
    out = {p: "unknown" for p in L.PROBES}
    r = L.simulate("fixed", out)
    assert r["status"] == "unidentifiable" and r["probes_used"] == len(L.PROBES)


def test_classify_thresholds_and_unknowns():
    ref = {"value_return_corr": 0.8, "low_speed_saturation": 0.002, "loaded_foot_speed": 0.02}
    assert L.classify("P_noise", {"noise_ratio": 1.1}, None) == "normal"
    assert L.classify("P_noise", {"noise_ratio": 0.1}, None) == "abnormal"
    assert L.classify("P_value", {"value_return_corr": 0.1}, ref) == "abnormal"
    assert L.classify("P_value", {"value_return_corr": 0.6}, ref) == "normal"
    assert L.classify("P_value", {"value_return_corr": 0.6}, None) == "unknown"
    assert L.classify("P_reward", {"track_lin_vel_xy_exp_rel_error": 0.0, "track_ang_vel_z_exp_rel_error": 0.2}, None) == "abnormal"
    assert L.classify("P_torque", {"low_speed_saturation": 0.02}, ref) == "normal"  # 바닥 0.01의 3배 이하
    assert L.classify("P_torque", {"low_speed_saturation": 0.2}, ref) == "abnormal"
    assert L.classify("P_slip", {"loaded_foot_speed": 0.2}, ref) == "abnormal"
    assert L.classify("P_slip", {"loaded_foot_speed": 0.05}, ref) == "normal"
    assert L.classify("P_episode", {"timeout_ratio": 0.05}, None) == "abnormal"
    assert L.classify("P_episode", {"timeout_ratio": None}, None) == "unknown"
    assert L.classify("P_slip", None, ref) == "unknown"
