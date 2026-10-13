"""교정 범위 점검(관측 기반, 결정적 검사): 확정 probe가 이상으로 본 항이 등록 교정 표적 밖에 있는가.

동결 P_reward는 추종 보상 두 항(track_lin_vel_xy_exp·track_ang_vel_z_exp)의 재계산 상대 오차를 따로 남긴다. 측정 키의
항 이름은 isaaclab.envs.mdp.rewards의 함수 이름과 같다. 문턱은 동결 probe_loop.THRESHOLDS의 그 probe 값이고, classify와
같이 문턱을 넘어야(>) 이상이다. 이상 항 가운데 등록 교정이 되돌리는 함수가 아닌 것이 있으면 범위 부족이다.

에이전트 판단이 아니다. 확정 probe 측정값과 등록 교정 표적만 읽는다. 항별 출력이 없는 확정 probe(P_torque 등)는
not_applicable, 측정값이 없거나 수가 아니면 scope_unknown이다. 이 점검은 확정 probe가 재는 항만 본다(P_reward는 추종
두 항뿐이고 페널티 항은 보지 않는다).
"""
from __future__ import annotations

import math

from rl_triage import probe_loop as L

# 확정 probe → {보상 함수 이름: 그 항의 상대 오차 측정 키}
PER_TERM = {"P_reward": {"track_lin_vel_xy_exp": "track_lin_vel_xy_exp_rel_error",
                         "track_ang_vel_z_exp": "track_ang_vel_z_exp_rel_error"}}
VERDICTS = ("scope_sufficient", "scope_insufficient", "scope_unknown", "not_applicable")
# 대조 모드: 점검 판정 × 교정 정책 판정(고정 평가). 점검이 부족이라 했는데 회복하면 오경보, 충분이라 했는데 미회복이면 놓침
CONTRASTS = {("scope_insufficient", "not_recovered"): "insufficient_and_not_recovered",
             ("scope_insufficient", "recovered"): "insufficient_but_recovered",
             ("scope_sufficient", "not_recovered"): "sufficient_but_not_recovered",
             ("scope_sufficient", "recovered"): "sufficient_and_recovered"}


def check(confirming_probe: str, measurement: dict | None, target_functions: list[str]) -> dict:
    """확정 probe 측정값과 등록 교정 표적 함수 이름으로 범위를 판정한다."""
    terms = PER_TERM.get(confirming_probe)
    base = {"probe": confirming_probe, "registered_targets": sorted(target_functions)}
    if terms is None:
        return {"verdict": "not_applicable", **base, "reason": "확정 probe가 항별 측정값을 남기지 않는다"}
    threshold = L.THRESHOLDS[confirming_probe][2]
    errors = {name: (measurement or {}).get(key) for name, key in terms.items()}
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in errors.values()):
        return {"verdict": "scope_unknown", **base, "threshold": threshold, "rel_errors": errors,
                "reason": "항별 측정값이 없거나 수가 아니다"}
    abnormal = sorted(name for name, value in errors.items() if value > threshold)
    uncovered = sorted(set(abnormal) - set(target_functions))
    return {"verdict": "scope_insufficient" if uncovered else "scope_sufficient", **base, "threshold": threshold,
            "rel_errors": errors, "abnormal_terms": abnormal, "uncovered_terms": uncovered}


def contrast(verdict: str, correction_label: str | None) -> str | None:
    """대조 모드 범주. 점검이 판정을 못 했거나 교정 판정이 recovered·not_recovered가 아니면 None."""
    return CONTRASTS.get((verdict, correction_label))
