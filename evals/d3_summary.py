"""D3 요약기: 사례·방법별 확인 재학습 수와 결과 문구(설계안 6절).

주 지표는 해로운 변경을 확정하기까지 방법이 혼자 돌았을 때 쓴 확인 재학습 수다. unresolved·model_invalid는 상한 R로 센다.
결과는 네 가지 중 하나다.
- invalid: 집계 대상 유효 사례가 3개 미만(무해성 실패·case_invalid·interaction_observed 등)
- agent_worse: B4 합이 B3 합보다 크거나 B4가 틀린 변경을 확정했다
- agent_fewer_retrains: ① B4 합 < B1·B2·B3 합의 최소 ② 어느 사례에서도 B4 > B3 아님 ③ B4 잘못된 확정·unresolved 0 ④ 유효 사례 3
- no_advantage: 그 밖
개발 사례(aggregate False)는 집계하지 않는다. 표준 라이브러리만 쓴다.
"""
from __future__ import annotations

from d3_methods import METHODS, R

REQUIRED_CASES = 3
DETERMINISTIC = ("B1", "B2", "B3")
OUTCOMES = ("agent_fewer_retrains", "no_advantage", "agent_worse", "invalid")
WORDING = {
    "agent_fewer_retrains": "k = 8, 사전등록 사례 3건에서 변경 diff와 probe 관측으로 순위를 매긴 에이전트가 결정적 기준선보다 확인 "
                            "재학습을 적게 써서 해로운 변경을 확정했다. 사례 수가 적어 일반화·통계적 우위의 근거가 아니다.",
    "no_advantage": "이 설정에서 에이전트 순위는 결정적 기준선보다 확인 재학습을 줄이지 못했다.",
    "agent_worse": "에이전트 순위가 결정적 기준선보다 재학습을 더 쓰거나 틀린 변경을 확정했다.",
    "invalid": "사례 성립 조건을 채우지 못해 방법 비교를 판정하지 않는다.",
}
FORBIDDEN_CLAIMS = ("에이전트가 더 정확하다", "일반화", "실제 PR·프로젝트에서의 절감", "통계적 유의", "다른 실험과 합친 분모",
                    "k ≠ 8에서도 성립", "LLM 코드 리뷰가 사람보다 낫다", "B5 결과를 probe 일반의 가치로 확대")


def method_cost(row: dict, harmful: int) -> dict:
    """방법 하나의 사례 결과 → 비용(재학습 수, unresolved는 R)·잘못된 확정·unresolved 표시."""
    confirmed = row["status"] == "confirmed"
    return {"retrains": len(row["queries"]) if confirmed else R, "queried": len(row["queries"]),
            "wrong": confirmed and row["confirmed"] != harmful, "unresolved": not confirmed,
            "reason": row.get("reason")}


def case_costs(case: dict) -> dict[str, dict]:
    return {m: method_cost(case["methods"][m], case["harmful_index"]) for m in METHODS}


def verdict(cases: list[dict]) -> dict:
    """cases: {case, aggregate, valid, harmful_index, methods{B1..B5: run_method 결과}}."""
    counted = [c for c in cases if c["aggregate"]]
    valid = [c for c in counted if c["valid"]]
    costs = {c["case"]: case_costs(c) for c in valid}
    sums = {m: sum(costs[c["case"]][m]["retrains"] for c in valid) for m in METHODS}
    b4 = [costs[c["case"]]["B4"] for c in valid]
    conditions = {
        "fewer_than_every_deterministic_sum": bool(valid) and sums["B4"] < min(sums[m] for m in DETERMINISTIC),
        "never_more_than_b3_in_a_case": all(costs[c["case"]]["B4"]["retrains"] <= costs[c["case"]]["B3"]["retrains"]
                                            for c in valid),
        "no_wrong_or_unresolved": not any(row["wrong"] or row["unresolved"] for row in b4),
        "three_valid_cases": len(valid) == REQUIRED_CASES,
    }
    if len(valid) < REQUIRED_CASES:
        outcome = "invalid"
    elif sums["B4"] > sums["B3"] or any(row["wrong"] for row in b4):
        outcome = "agent_worse"
    elif all(conditions.values()):
        outcome = "agent_fewer_retrains"
    else:
        outcome = "no_advantage"
    return {"outcome": outcome, "wording": WORDING[outcome], "conditions": conditions, "sums": sums,
            "valid_cases": [c["case"] for c in valid],
            "excluded_cases": {c["case"]: c.get("invalid_reason") for c in counted if not c["valid"]},
            "development_cases": [c["case"] for c in cases if not c["aggregate"]], "costs": costs,
            "wrong_confirmations": {m: sum(costs[c][m]["wrong"] for c in costs) for m in METHODS},
            "unresolved": {m: sum(costs[c][m]["unresolved"] for c in costs) for m in METHODS}}
