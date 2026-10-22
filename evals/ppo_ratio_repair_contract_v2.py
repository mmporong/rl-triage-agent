"""PPO ratio 계산 오류 종단 판의 검사 등록 v2. GPU와 API 부수효과가 없다.

v1(ppo_ratio_repair_contract.py)은 기존 검사 전부에서 surrogate 하위형을 normal로 예상했다. 설계 파일럿 v2
(seed 2162)에서 ratio 오류 정책은 행동 표준편차가 정상의 약 4배였고 P_torque 저속 포화가 0.746(정상 0.00067)으로
abnormal이 나왔다. 토크 포화가 원인이 아니라 하류 증상으로도 나타나므로, v1 고리는 P_torque를 먼저 고르면 actuator를
잘못 확정한다. v2는 P_torque의 surrogate 예상만 null(예측 없음)로 바꾸고 나머지 예상·임계값·고리 규칙은 v1과 같다.
v1은 파일럿 v1·v2 프로토콜이 SHA로 묶고 있어 바꾸지 않는다. value·update_count 하위형의 P_torque 예상은 관측한 적이
없어 v1 그대로 둔다.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for _directory in (ROOT / "src", ROOT / "evals"):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))
import ppo_ratio_repair_contract as V1  # noqa: E402

D, M, L = V1.D, V1.M, V1.L
MODEL, SURROGATE, SUBTYPES = V1.MODEL, V1.SURROGATE, V1.SUBTYPES
COUNT_PROBE, GRADIENT_PROBE, CORRECTION = V1.COUNT_PROBE, V1.GRADIENT_PROBE, V1.CORRECTION
MAX_PROBES, PACKET_SCHEMA, GRADIENT_RULE = V1.MAX_PROBES, V1.PACKET_SCHEMA, V1.GRADIENT_RULE
SYMPTOM_PROBE = "P_torque"
# 설계 파일럿 v2(seed 2162) 관측표. 이 표가 v2 변경의 근거이며 본 실행 seed와 다르다.
PILOT_OUTCOMES = {"P_episode": "normal", "P_noise": "normal", "P_physics": "normal", "P_reward": "normal",
                  "P_torque": "abnormal", "P_value": "normal", "P_optimizer_updates": "normal",
                  "P_ppo_gradient": "abnormal"}
category, initial_state = V1.category, V1.initial_state
correction_registry, correction_allowed = V1.correction_registry, V1.correction_allowed


def registry() -> dict:
    """v1 등록 표에서 P_torque의 surrogate 예상만 null로 바꾼다."""
    value = V1.registry()
    value["probes"][SYMPTOM_PROBE]["expect"][SURROGATE] = None
    return M.normalize(value)


def public_catalog() -> dict:
    result = {}
    for name, spec in registry()["probes"].items():
        expected = {}
        for mechanism in L.MECHANISMS:
            values = {outcome for subtype, outcome in spec["expect"].items() if category(subtype) == mechanism}
            expected[mechanism] = next(iter(values)) if len(values) == 1 else None
        result[name] = {**deepcopy(spec), "expect": expected}
    return result


def advance(state: dict, probe: str, outcome: str) -> dict:
    reg = registry()
    if state["status"] != "open" or probe != state["next_probe"] or probe in state["observed"]:
        raise ValueError("등록된 다음 검사와 다르거나 고리가 이미 닫혔다")
    hypotheses, dropped = M.update(reg, state["hypotheses"], probe, outcome)
    observed = {**state["observed"], probe: outcome}
    remaining = MAX_PROBES - len(observed)
    status = M._status(reg, hypotheses, observed, remaining)
    # 알 수 없는 실행을 다음 검사로 가려서 성공으로 바꾸지 않는다.
    if outcome == "unknown":
        status = "unidentifiable"
    next_probe = M._next(reg, hypotheses, observed) if status == "open" else None
    if status == "open" and next_probe is None:
        status = "unidentifiable"
    return {"hypotheses": hypotheses, "observed": observed, "budget_left": remaining, "status": status,
            "next_probe": next_probe,
            "history": [*state["history"], {"probe": probe, "outcome": outcome, "dropped": dropped, "left": hypotheses}]}


def replay_loop(first_probe: str, outcomes: dict[str, str]) -> dict:
    """관측표가 주어졌을 때 첫 검사에서 시작한 등록 고리의 끝을 계산한다(설계 점검용, 실행 없음)."""
    state = initial_state({"ranking": list(L.MECHANISMS), "probe": first_probe})
    while state["status"] == "open":
        probe = state["next_probe"]
        state = advance(state, probe, outcomes.get(probe, "unknown"))
    return {"first_probe": first_probe, "status": state["status"],
            "conclusion": state["hypotheses"][0] if state["status"] == "confirmed" else None,
            "probes": [row["probe"] for row in state["history"]]}


def correction_packet(state: dict, confirming: dict, report: dict, checkpoint_sha: str, report_sha: str) -> dict:
    # P_ppo_gradient 정의는 v1과 같아 v1 교정 packet을 그대로 쓴다.
    return V1.correction_packet(state, confirming, report, checkpoint_sha, report_sha)


SYSTEM = V1.SYSTEM.replace(
    "A null category expectation means the subtypes differ.",
    "A null category expectation means the subtypes differ or one subtype is not predicted.\n"
    "A null subtype expectation keeps that subtype for any outcome of the probe.")


def diagnosis_messages(packet: dict) -> list[dict]:
    if (packet.get("schema") != PACKET_SCHEMA or packet.get("observed") != []
            or packet.get("budget_left") != MAX_PROBES):
        raise ValueError("초기 진단 packet 계약이 다르다")
    facts = packet.get("facts")
    if not isinstance(facts, dict) or not facts or any(not isinstance(k, str) or not D._is_scalar(v) for k, v in facts.items()):
        raise ValueError("진단 사실은 비지 않은 유한 scalar 표여야 한다")
    body = {"evidence_facts": facts, "categories": list(L.MECHANISMS), "probe_catalog": public_catalog(),
            "subtype_registry": registry(), "thresholds": {k: list(v) for k, v in L.THRESHOLDS.items()},
            "update_count_rule": "3 complete updates; steps and minibatches equal 20 each => normal; any mismatch => abnormal; incomplete => unknown",
            "surrogate_gradient_rule": GRADIENT_RULE,
            "budget_left": MAX_PROBES, "execution_budget_s": packet["execution_budget_s"]}
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": json.dumps(body, ensure_ascii=False, sort_keys=True)}]


def validate_diagnosis(decision: dict, packet: dict) -> dict:
    diagnosis_messages(packet)
    fields = {"ranking", "probe", "action", "reason", "claims", "expected_outcomes"}
    if not isinstance(decision, dict) or set(decision) != fields:
        raise ValueError("decision 필드는 공개 계약과 정확히 일치해야 한다")
    ranking = decision["ranking"]
    if (not isinstance(ranking, list) or any(not isinstance(x, str) for x in ranking)
            or len(ranking) != 6 or set(ranking) != set(L.MECHANISMS)):
        raise ValueError("ranking은 여섯 mechanism의 중복 없는 순열이어야 한다")
    action, probe = decision["action"], decision["probe"]
    if action not in ("continue", "hold"):
        raise ValueError("초기 판단은 confirm할 수 없다")
    if not isinstance(decision["reason"], str) or not decision["reason"].strip():
        raise ValueError("reason은 비어 있지 않은 문자열이어야 한다")
    if action == "continue":
        if not isinstance(probe, str) or probe not in public_catalog():
            raise ValueError("continue probe는 미관측 public probe여야 한다")
        expected = {key: public_catalog()[probe]["expect"][key] for key in ranking[:3]}
        if decision["expected_outcomes"] != expected:
            raise ValueError("expected_outcomes는 선택 probe의 공개 카탈로그와 정확히 일치해야 한다")
    elif probe is not None or decision["expected_outcomes"] != {}:
        raise ValueError("hold/confirm의 probe는 null이어야 한다")
    claims, seen = decision["claims"], set()
    if not isinstance(claims, list) or not claims:
        raise ValueError("claims는 하나 이상의 근거 claim이어야 한다")
    for item in claims:
        if not isinstance(item, dict) or set(item) != {"id", "value"}:
            raise ValueError("claim은 id와 value만 가진 객체여야 한다")
        key = item["id"]
        if not isinstance(key, str) or key in seen or key not in packet["facts"]:
            raise ValueError("알 수 없거나 중복된 claim id")
        if not D._is_scalar(item["value"]) or not D._same_value(item["value"], packet["facts"][key]):
            raise ValueError("claim 값이 evidence fact와 다르다")
        seen.add(key)
    return deepcopy(decision)
