"""D3 방법 구동기·요약기·B4 계약을 Isaac·모델 없이 고정한다."""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
from pathlib import Path
import random
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "bench", ROOT / "evals"):
    sys.path.insert(0, str(directory))

import d3_agent as AG  # noqa: E402
import d3_changes as D  # noqa: E402
import d3_methods as MT  # noqa: E402
import d3_summary as SU  # noqa: E402
from rl_triage import probe_loop as L  # noqa: E402

ISAACLAB = Path.home() / "IsaacLab"
needs_isaaclab = pytest.mark.skipif(not (ISAACLAB / D.REWARDS[1]).is_file(), reason="로컬 Isaac Lab v2.1.1 소스가 없다")
K = 8
CATEGORIES = ["reward", "reward", "reward", "actuator", "observation", "logging", "optimizer"]


def oracle(harmful: int, k: int = K) -> dict[int, str]:
    return {mask: "unhealthy" if mask >> harmful & 1 else "healthy" for mask in range(1 << k)}


def categories_with(harmful: int, category: str) -> list[str]:
    out = list(CATEGORIES)
    out.insert(harmful, category)
    return out


# --- 방법 구동기 ---
@pytest.mark.parametrize("harmful", range(K))
def test_every_method_confirms_the_harmful_change_with_the_designed_number_of_retrains(harmful):
    answers = oracle(harmful)
    b1 = MT.run_method("B1", K, answers)
    assert (b1["status"], b1["confirmed"]) == ("confirmed", harmful) and len(b1["queries"]) in (3, 4)
    b2 = MT.run_method("B2", K, answers)
    assert (b2["confirmed"], len(b2["queries"])) == (harmful, harmful + 1)
    for category in ("reward", "actuator"):
        cats = categories_with(harmful, category)
        b3 = MT.run_method("B3", K, answers, categories=cats, loop_conclusion=category)
        same = [i for i, c in enumerate(cats) if c == category]
        assert (b3["confirmed"], len(b3["queries"])) == (harmful, same.index(harmful) + 1)
    ranking = random.Random(harmful).sample(range(K), K)
    for method in ("B4", "B5"):
        row = MT.run_method(method, K, answers, ranking=ranking)
        assert (row["confirmed"], len(row["queries"])) == (harmful, ranking.index(harmful) + 1)
    # 모든 확정은 그 변경 하나만 되돌린 재학습을 질의한 뒤다
    for row in (b1, b2, b3):
        assert MT.single_revert(K, harmful) in row["queries"]


def test_b1_reverts_the_front_half_and_counts_a_repeated_single_revert_once():
    row = MT.run_method("B1", K, oracle(0))
    assert row["queries"] == [0b11110000, 0b11111100, 0b11111110]
    row = MT.run_method("B1", K, oracle(1))
    assert row["queries"] == [0b11110000, 0b11111100, 0b11111110, 0b11111101]


def test_b3_falls_back_to_bisection_when_the_loop_did_not_confirm():
    answers = oracle(5)
    assert MT.run_method("B3", K, answers, categories=categories_with(5, "reward"), loop_conclusion=None)["queries"] == \
        MT.run_method("B1", K, answers)["queries"]


@pytest.mark.parametrize("method", ["B1", "B2", "B3", "B4"])
def test_replaying_with_answers_arriving_one_by_one_gives_the_same_queries(method):
    full_answers = oracle(6)
    context = {"categories": categories_with(6, "reward"), "loop_conclusion": "reward", "ranking": [3, 6, 0, 1, 2, 4, 5, 7]}
    expected = MT.run_method(method, K, full_answers, **context)
    known: dict[int, str] = {}
    while (row := MT.run_method(method, K, known, **context))["status"] == "pending":
        known[row["next"]] = full_answers[row["next"]]
        known[0b10101010] = "unhealthy"  # 다른 방법이 캐시에 남긴 답은 질의 순서를 바꾸지 않는다
    assert row == expected


def test_confirmation_requires_a_healthy_single_revert_and_unknown_answers_stop_the_method():
    answers = oracle(2)
    answers[MT.single_revert(K, 2)] = "unhealthy"  # 이분 탐색이 좁힌 후보의 단일 되돌리기가 healthy가 아니다
    row = MT.run_method("B1", K, answers)
    assert (row["status"], row["reason"], row["confirmed"]) == ("unresolved", "confirmation_unhealthy", None)
    assert MT.run_method("B2", K, answers)["reason"] == "order_exhausted"
    answers = oracle(2)
    answers[MT.single_revert(K, 0)] = "unknown"
    row = MT.run_method("B2", K, answers)
    assert (row["status"], row["reason"], len(row["queries"])) == ("unresolved", "unanswered", 1)


def test_the_cap_and_an_invalid_ranking_leave_the_method_unresolved():
    row = MT.run_method("B2", K, oracle(5), cap=2)
    assert (row["status"], row["reason"], len(row["queries"])) == ("unresolved", "cap", 2)
    row = MT.run_method("B4", K, oracle(5), ranking_invalid=True)
    assert (row["status"], row["reason"], row["queries"]) == ("unresolved", "model_invalid", [])
    with pytest.raises(ValueError, match="순열"):
        MT.run_method("B4", K, oracle(5), ranking=[0, 0, 1, 2, 3, 4, 5, 6])
    with pytest.raises(ValueError, match="순위"):
        MT.run_method("B5", K, oracle(5))


def test_interaction_finds_subsets_that_contradict_the_no_interaction_assumption():
    answers = {0: "healthy", 0xFF: "unhealthy", 0b11110000: "healthy", 0b00001111: "unhealthy"}
    assert MT.interaction(answers, 0, K) == []
    answers[0b11111100] = "unhealthy"  # 해로운 변경(0)이 없는데 unhealthy
    answers[0b00000011] = "healthy"    # 있는데 healthy
    assert MT.interaction(answers, 0, K) == [0b00000011, 0b11111100]
    assert MT.interaction({0xFF: "healthy", 0: "unhealthy"}, 0, K) == []  # 실패 실행·NONE은 따로 다룬다


def test_the_lazy_loop_matches_the_frozen_fixed_order_simulation_on_every_outcome_table():
    probes = sorted(L.PROBES)
    for outcomes in itertools.product(L.OUTCOMES, repeat=len(probes)):
        table = dict(zip(probes, outcomes))
        observed = {}
        while (state := MT.loop_state(observed))["next"] is not None:
            observed[state["next"]] = table[state["next"]]
        frozen = L.simulate("fixed", table)
        assert (state["status"], state["conclusion"], [s["probe"] for s in state["steps"]]) == (
            frozen["status"], frozen["conclusion"], [s["probe"] for s in frozen["steps"]]), table


# --- 요약기 ---
def _case(name: str, harmful: int, counts: dict[str, int], *, aggregate=True, valid=True, wrong=(), unresolved=()) -> dict:
    methods = {}
    for method in MT.METHODS:
        n = counts.get(method, 4)
        status = "unresolved" if method in unresolved else "confirmed"
        confirmed = None if status == "unresolved" else (harmful + 1) % K if method in wrong else harmful
        methods[method] = {"status": status, "confirmed": confirmed, "queries": list(range(n)), "reason": None}
    return {"case": name, "aggregate": aggregate, "valid": valid, "harmful_index": harmful, "methods": methods,
            "invalid_reason": None if valid else "interaction_observed"}


def _three(b4=(1, 1, 1), b3=(4, 4, 2), **kwargs) -> list[dict]:
    return [_case(f"C{i + 1}", i, {"B1": 4, "B2": 6, "B3": b3[i], "B4": b4[i], "B5": 2}, **kwargs) for i in range(3)]


def test_the_advantage_wording_needs_all_four_conditions():
    out = SU.verdict([_case("D0", 0, {"B4": 8}, aggregate=False), *_three()])
    assert out["outcome"] == "agent_fewer_retrains" and all(out["conditions"].values())
    assert out["development_cases"] == ["D0"] and out["sums"]["B4"] == 3 and out["wording"] == SU.WORDING["agent_fewer_retrains"]


def test_one_case_where_b4_needs_more_than_b3_blocks_the_advantage():
    out = SU.verdict(_three(b4=(1, 1, 3)))  # 합은 5로 B3 10보다 작지만 C3에서 B4 3 > B3 2
    assert out["outcome"] == "no_advantage" and out["conditions"]["never_more_than_b3_in_a_case"] is False


def test_an_unresolved_b4_costs_the_cap_and_blocks_the_advantage():
    out = SU.verdict(_three(unresolved=("B4",)))
    assert out["sums"]["B4"] == 3 * MT.R and out["outcome"] == "agent_worse"
    # B3 합이 커서 B4 합(1+1+8)이 더 작아도, unresolved가 있으면 우위가 아니다
    cases = [*_three(b3=(8, 8, 8))[:2], _case("C3", 2, {"B1": 9, "B2": 9, "B3": 8}, unresolved=("B4",))]
    out = SU.verdict(cases)
    assert out["outcome"] == "no_advantage" and out["conditions"]["no_wrong_or_unresolved"] is False
    assert out["conditions"]["fewer_than_every_deterministic_sum"] is True


def test_a_wrong_b4_confirmation_or_a_larger_b4_sum_is_agent_worse():
    assert SU.verdict(_three(wrong=("B4",)))["outcome"] == "agent_worse"
    assert SU.verdict(_three(b4=(5, 5, 5)))["outcome"] == "agent_worse"
    out = SU.verdict(_three(b4=(4, 4, 2)))  # B3와 같은 합: 우위도 열세도 아니다
    assert out["outcome"] == "no_advantage"


def test_fewer_than_three_valid_cases_is_invalid():
    cases = _three()
    cases[1].update(valid=False, invalid_reason="interaction_observed")
    out = SU.verdict(cases)
    assert out["outcome"] == "invalid" and out["excluded_cases"] == {"C2": "interaction_observed"}
    assert SU.verdict(_three()[:2])["outcome"] == "invalid"


def test_wording_does_not_make_forbidden_claims():
    from build_replay_page import claim_violations
    assert not [p for text in SU.WORDING.values() for p in claim_violations(text)]


# --- B4·B5 계약 ---
FACTS = {"telemetry:case:Train/mean_reward": 1.0, "telemetry:reference:Train/mean_reward": 20.0}
DIFF = "--- a/x.py\n+++ b/x.py\n@@ -10,3 +10,4 @@\n a\n-b\n+c\n+d\n e\n"
OBS = [{"probe": "P_reward", "measures": AG.PROBE_TEXT["P_reward"], "rule": ["k", "above", 0.05], "outcome": "abnormal",
        "measurement": {"k": 0.5}, "reference_measurement": {"k": 0.0}}]


def _packet(observations=OBS, diffs=None) -> dict:
    return AG.build_packet(diffs or [DIFF] * 3, FACTS, observations)


def _output(**changes) -> dict:
    out = {"ranking": ["change_2", "change_1", "change_3"],
           "reasons": [{"change": "change_2", "facts": ["telemetry:case:Train/mean_reward"], "probes": ["P_reward"],
                        "diff_lines": [11, 12]}], "uncertain": False}
    out.update(changes)
    return out


def test_packets_carry_neutral_labels_and_b5_drops_only_the_probe_observations():
    b4, b5 = _packet(), _packet([])
    assert [c["label"] for c in b4["changes"]] == ["change_1", "change_2", "change_3"]
    assert set(b4["changes"][0]) == {"label", "diff"} and {k: v for k, v in b4.items() if k != "probe_observations"} == \
        {k: v for k, v in b5.items() if k != "probe_observations"}
    assert AG.messages(b4)[0]["content"] == AG.messages(b5)[0]["content"]
    assert AG.new_lines(DIFF) == {10, 11, 12, 13} and AG.new_lines("@@ -1 +5 @@\n") == {5}


def test_the_validator_accepts_the_contract_and_names_each_violation():
    packet = _packet()
    assert AG.validate(_output(), packet) == []
    cases = {"ranking_not_a_permutation": _output(ranking=["change_2", "change_2", "change_3"]),
             "reason0_facts": _output(reasons=[{**_output()["reasons"][0], "facts": ["telemetry:case:nope"]}]),
             "reason0_diff_lines": _output(reasons=[{**_output()["reasons"][0], "diff_lines": [99]}]),
             "top_reason": _output(reasons=[{**_output()["reasons"][0], "change": "change_1"}]),
             "uncertain_not_bool": _output(uncertain="no"), "keys": {**_output(), "extra": 1}}
    for problem, output in cases.items():
        assert problem in AG.validate(output, packet), problem
    assert "reason0_probes" in AG.validate(_output(), _packet([]))  # B5는 probe를 인용할 수 없다
    assert AG.validate(_output(reasons=[{**_output()["reasons"][0], "diff_lines": []}]), packet) == ["top_reason"]


def _record(packet: dict, content, **extra) -> tuple[dict, bytes]:
    data = json.dumps(packet).encode()
    record = {"schema": "d3_model_call_v1", "packet_sha256": hashlib.sha256(data).hexdigest(),
              "request": {"model": "m", "temperature": 0}, "exit_code": 0, "total_tokens": 10,
              "content": content if isinstance(content, str) else json.dumps(content), **extra}
    return record, data


def test_review_binds_the_record_to_the_packet_and_classifies_the_call():
    packet = _packet()
    record, data = _record(packet, "```json\n" + json.dumps(_output()) + "\n```")
    out = AG.review(record, data, "m")
    assert (out["status"], out["ranking"], out["total_tokens"]) == ("valid", ["change_2", "change_1", "change_3"], 10)
    assert AG.review(_record(packet, _output(uncertain="x"))[0], data, "m")["status"] == "invalid"
    assert AG.review(_record(packet, "not json")[0], data, "m")["status"] == "invalid"
    assert AG.review({**record, "infra_error": "APITimeoutError", "exit_code": 2}, data, "m")["status"] == "infra_error"
    assert AG.review({**record, "error_type": "BadRequestError", "exit_code": 2}, data, "m")["status"] == "call_error"
    for broken, message in (({**record, "packet_sha256": "0" * 64}, "packet"), ({**record, "request": {"model": "x",
                                                                                                       "temperature": 0}}, "모델"),
                            ({**record, "schema": "active_model_call_v1"}, "d3_model_call_v1")):
        with pytest.raises(ValueError, match=message):
            AG.review(broken, data, "m")


def test_telemetry_facts_use_the_active_diagnosis_fact_ids():
    telemetry = json.loads((ROOT / "bench/runs_p0c/baseline_p0c_s2026.telemetry.json").read_text(encoding="utf-8"))
    facts = AG.telemetry_facts(telemetry["summary"], telemetry["summary"])
    tag = "Episode_Reward/track_lin_vel_xy_exp"
    assert facts[f"telemetry:case:{tag}"] == telemetry["summary"][tag]["mean_last_20pct"] == facts[f"telemetry:reference:{tag}"]
    assert len(facts) == 2 * sum("mean_last_20pct" in v for v in telemetry["summary"].values())


@needs_isaaclab
def test_real_case_packets_have_no_clues_and_planted_ones_are_found():
    import d3_diff as R
    order = ["H1", "H2", "H3", "H4", "H5", "U1", "H6", "H7"]
    telemetry = json.loads((ROOT / "bench/runs_p0c/baseline_p0c_s2026.telemetry.json").read_text(encoding="utf-8"))
    packet = AG.build_packet([R.render(cid, ISAACLAB) for cid in order],
                             AG.telemetry_facts(telemetry["summary"], telemetry["summary"]), OBS)
    assert AG.clues(packet) == []
    planted = copy.deepcopy(packet)
    planted["facts"]["telemetry:case:U1"] = 1.0
    assert "messages:id:U1" in AG.clues(planted)
    planted = copy.deepcopy(packet)
    planted["changes"][0]["diff"] += "+    x = 1  # tidy\n"
    assert "change_1:added_comment" in AG.clues(planted)
    assert "messages:word:harm" in AG.clues({**packet, "facts": {"harmless": 1}})
