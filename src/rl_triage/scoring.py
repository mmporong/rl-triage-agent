"""채점 규칙. 평가 하네스(evals/run_eval.py)와 offline replay(evals/replay.py)가 같은 구현을 쓴다.

표준 라이브러리만 쓴다. 모델 클라이언트·NAT·네트워크를 import하지 않는다.
"""
from __future__ import annotations


def score_blind(case_id: str, ranking, key: dict) -> dict:
    """과제 B: 메커니즘 순위의 1위·2위 안에 정답 범주가 있는지."""
    truth = key["cases"][case_id]["category"]
    ranking = ranking or []
    return {"truth": truth, "suspected": ranking[0] if ranking else None, "ranking": ranking,
            "correct": bool(ranking) and ranking[0] == truth, "top2": truth in ranking[:2],
            "category": truth}


def truth_change_id(overrides: list[str], harmful_override: str) -> str:
    """과제 A의 정답 change_id. list_changes와 같은 규칙(overrides 순서대로 ch1, ch2, ...)으로 매긴다."""
    harmful = harmful_override.split("=", 1)[0]
    hits = [f"ch{i + 1}" for i, ov in enumerate(overrides) if ov.split("=", 1)[0] == harmful]
    if len(hits) != 1:
        raise ValueError(f"해로운 변경 {harmful!r}이 overrides에서 {len(hits)}번 나온다")
    return hits[0]


def score_changes(case_id: str, suspected: str | None, overrides: list[str], key: dict) -> dict:
    """과제 A: 의심 change_id가 해로운 변경과 같은지."""
    truth = truth_change_id(overrides, key["cases"][case_id]["harmful_override"])
    return {"truth": truth, "suspected": suspected, "correct": suspected == truth,
            "category": key["cases"][case_id]["category"]}
