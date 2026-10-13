"""D3 방법 구동기: 변경 k개 중 학습 실패를 낸 변경 하나를 확인 재학습으로 확정하는 다섯 방법(B1~B5).

부분집합은 사례의 변경 순서에 대한 비트마스크(적용한 변경의 비트가 1)다. 답은 그 부분집합으로 같은 seed에서 처음부터
학습한 정책이 같은 seed NONE 대비 healthy인지 unhealthy인지이며, 캐시(evals/run_d3_experiment.py)가 준다.
각 방법은 처음부터 다시 재생한다. 아는 답이 있으면 쓰고, 모르는 부분집합에 닿으면 그것을 다음 질의로 돌려준다.
그래서 여러 방법이 캐시를 함께 써도 각 방법의 질의 순서·횟수는 그 방법이 혼자 돌았을 때와 같다.

- 확정(모든 방법 공통): 변경 c 하나만 되돌린 재학습(전체 − {c})이 healthy일 때만 c를 확정한다.
- 상한: 서로 다른 질의 R=8개 안에 확정하지 못하면 unresolved. 답이 unknown이면 그 자리에서 unresolved.
- B1 이분 탐색, B2 하나씩(사례 순서), B3 probe 범주 규칙(동결 고리가 확정한 범주의 변경 먼저, 미확정이면 B1),
  B4·B5 모델 순위 순서. 순위가 무효(model_invalid)면 질의 없이 unresolved이고 비용은 요약기가 R로 센다.
표준 라이브러리와 동결 probe_loop만 쓴다(Isaac·모델 없음).
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from rl_triage import probe_loop as L  # noqa: E402

R = 8
METHODS = ("B1", "B2", "B3", "B4", "B5")
ANSWERS = ("healthy", "unhealthy", "unknown")


def full(k: int) -> int:
    return (1 << k) - 1


def single_revert(k: int, index: int) -> int:
    return full(k) & ~(1 << index)


class _Pending(Exception):
    def __init__(self, mask: int):
        self.mask = mask


class _Stop(Exception):
    def __init__(self, reason: str):
        self.reason = reason


class _Asker:
    """방법 하나의 질의 기록. 같은 부분집합을 다시 물으면 횟수를 늘리지 않는다."""

    def __init__(self, answers: dict[int, str], cap: int):
        self.answers, self.cap, self.queries = answers, cap, []

    def __call__(self, mask: int) -> str:
        if mask not in self.queries:
            if len(self.queries) >= self.cap:
                raise _Stop("cap")
            self.queries.append(mask)
        answer = self.answers.get(mask)
        if answer is None:
            raise _Pending(mask)
        if answer not in ANSWERS:
            raise ValueError(f"모르는 답 {answer!r}")
        if answer == "unknown":
            raise _Stop("unanswered")
        return answer


def _in_order(k: int, indices: list[int], ask) -> tuple[str, int | None, str | None]:
    for index in indices:
        if ask(single_revert(k, index)) == "healthy":
            return "confirmed", index, None
    return "unresolved", None, "order_exhausted"


def _bisection(k: int, ask) -> tuple[str, int | None, str | None]:
    """후보의 앞쪽 절반을 되돌리고(나머지는 적용) healthy면 그 절반, 아니면 나머지 절반으로 좁힌다."""
    candidates = list(range(k))
    while len(candidates) > 1:
        half = candidates[:len(candidates) // 2]
        mask = full(k) & ~sum(1 << i for i in half)
        candidates = half if ask(mask) == "healthy" else candidates[len(candidates) // 2:]
    index = candidates[0]
    if ask(single_revert(k, index)) == "healthy":  # 마지막 단계가 이 단일 되돌리기였으면 다시 세지 않는다
        return "confirmed", index, None
    return "unresolved", None, "confirmation_unhealthy"


def category_order(categories: list[str], mechanism: str) -> list[int]:
    """B3 순서: 범주가 고리 확정 메커니즘과 같은 변경을 사례 순서로 먼저, 나머지를 사례 순서로."""
    first = [i for i, c in enumerate(categories) if c == mechanism]
    return first + [i for i in range(len(categories)) if i not in first]


def run_method(method: str, k: int, answers: dict[int, str], *, categories: list[str] | None = None,
               loop_conclusion: str | None = None, ranking: list[int] | None = None, ranking_invalid: bool = False,
               cap: int = R) -> dict:
    """방법 하나를 처음부터 재생한다. status는 confirmed / unresolved / pending(next에 다음 질의)."""
    if method not in METHODS:
        raise ValueError(f"모르는 방법 {method!r}")
    ask = _Asker(answers, cap)
    if method in ("B4", "B5") and ranking_invalid:
        return {"method": method, "status": "unresolved", "confirmed": None, "reason": "model_invalid", "queries": [],
                "next": None}
    try:
        if method == "B1" or (method == "B3" and loop_conclusion is None):
            status, index, reason = _bisection(k, ask)
        elif method == "B2":
            status, index, reason = _in_order(k, list(range(k)), ask)
        elif method == "B3":
            status, index, reason = _in_order(k, category_order(categories, loop_conclusion), ask)
        else:
            if ranking is None:
                raise ValueError(f"{method}에는 순위가 필요하다")
            if sorted(ranking) != list(range(k)):
                raise ValueError("순위가 변경 전체의 순열이 아니다")
            status, index, reason = _in_order(k, list(ranking), ask)
    except _Pending as pending:
        return {"method": method, "status": "pending", "confirmed": None, "reason": None, "queries": ask.queries,
                "next": pending.mask}
    except _Stop as stop:
        return {"method": method, "status": "unresolved", "confirmed": None, "reason": stop.reason,
                "queries": ask.queries, "next": None}
    return {"method": method, "status": status, "confirmed": index, "reason": reason, "queries": ask.queries,
            "next": None}


def interaction(answers: dict[int, str], harmful: int, k: int) -> list[int]:
    """상호작용 없음 가정과 모순인 부분집합: 해로운 변경이 없는데 unhealthy, 있는데 healthy.

    빈 부분집합(NONE 기준 자신)과 전체(실패 실행, healthy면 case_invalid로 따로 다룬다)는 보지 않는다.
    """
    out = []
    for mask, answer in sorted(answers.items()):
        if mask in (0, full(k)) or answer not in ("healthy", "unhealthy"):
            continue
        if (answer == "unhealthy") != bool(mask >> harmful & 1):
            out.append(mask)
    return out


# --- B3의 동결 고리: probe_loop.simulate("fixed")를 probe 하나씩 재는 형태로 ---
def loop_state(observed: dict[str, str]) -> dict:
    """모든 범주를 가설로 두고 FIXED_ORDER로 진행한다. 아직 안 잰 probe가 필요하면 next에 둔다.

    probe 결과는 같은 체크포인트에서 순서와 무관하므로, 다 재고 나면 L.simulate("fixed", observed)와 같은 결론이다.
    """
    hypotheses, seen, steps = list(L.MECHANISMS), {}, []
    budget = len(L.PROBES)
    while len(seen) < budget:
        if L.status(hypotheses, seen, budget - len(seen)) != "open":
            break
        probe = next((p for p in L.FIXED_ORDER if p not in seen), None)
        if probe is None:
            break
        if probe not in observed:
            return {"next": probe, "status": "open", "conclusion": None, "probes": list(seen), "steps": steps}
        seen[probe] = observed[probe]
        hypotheses, dropped = L.update(hypotheses, probe, observed[probe])
        steps.append({"probe": probe, "outcome": observed[probe], "dropped": dropped, "left": list(hypotheses)})
    final = L.status(hypotheses, seen, budget - len(seen))
    final = "unidentifiable" if final == "open" else final
    return {"next": None, "status": final, "conclusion": hypotheses[0] if final == "confirmed" else None,
            "probes": list(seen), "steps": steps}
