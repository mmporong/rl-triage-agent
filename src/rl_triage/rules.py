"""P0-A2 결정적 기준선. LLM 없이 과제 B와 같은 메커니즘 순위를 낸다(docs/P0-A2-BASELINES.md).

- prior: 텔레메트리를 보지 않는다. dev 정답 빈도순으로 낸다.
- features: 정규화 특징(docs/IMPLEMENTATION-ORDER.md 2절 C)에 고정 규칙을 적용한다.
- template: 같은 특징으로 가장 가까운 dev 사례의 범주를 낸다. 같은 결함 템플릿을 다시 알아보는지 진단한다.
- b0_diff: 실패 실행 설정과 기준 설정의 차이로 범주를 낸다. 설정을 볼 수 있는 조건이다(2절 B).

규칙·임계값은 dev seed 7·42만 보고 정했다. 표준 라이브러리만 쓴다. case_id는 판정에 쓰지 않는다.
"""
from __future__ import annotations

import json
import math

MECHANISMS = ("reward", "actuator", "exploration", "optimizer", "physics", "termination")
DEV_SEEDS = (7, 42)
# dev seed 7·42 정답 빈도순(reward 6, optimizer 4, physics 4, 나머지 2). 동률은 MECHANISMS 순서.
PRIOR_ORDER = ("reward", "optimizer", "physics", "actuator", "exploration", "termination")

LEN = "Train/mean_episode_length"
EPS = 1e-6

# 임계값은 dev에서 규칙이 맞아야 할 사례(양성)의 가장 약한 값과 맞지 않아야 할 사례(음성)의 가장 강한 값 사이
# 기하 평균을 유효숫자 두 자리로 정했다. 음성은 규칙의 다른 조건을 통과하면서, 점수가 더 높은 규칙이 이미
# 정답을 정하지 않는 dev 사례다. 0.5(붕괴 여부)·0.1(길이 고정)·0.5(time_out 다수)는 의미로 정한 값이다.
# 각 값의 dev 근거는 docs/P0-A2-BASELINES.md 표에 있고 tests/test_rule_baselines.py가 다시 확인한다.
THRESHOLDS = {
    "collapsed_len_ratio": 0.5,      # 이보다 짧으면 학습이 무너진 실행으로 본다
    "pinned_len": 0.1,               # (최대 길이 - 후반 평균) / 후반 평균. 모든 에피소드가 같은 길이에서 끝남
    "timeout_majority": 0.5,         # 종료 중 time_out 비율
    "noise_collapse": 0.16,          # Policy/mean_noise_std 비. 양성 0.038, 음성 최소 0.677
    "value_loss_collapse": 0.12,     # 살아남은 실행의 Loss/value_function 비. 양성 0.017, 음성 최소 0.796
    "tracking_rate_up": 2.0,         # 무너진 실행의 추종 보상 초당 비 최대. 양성 3.43, 음성 최대 1.16
    "single_term_dominance": 20.0,   # 초당 비 1위/2위. 양성 36.6, 음성 최대 11.2
    "positive_term_up": 2.3,         # 살아남은 실행의 양수 가중 보상 항 초당 비 최대. 양성 4.66, 음성 최대 1.16
    "immediate_collapse": 0.022,     # 에피소드 길이 비. 양성 0.0100, 음성 최소 0.0501
    "torque_rate_up": 1.7,           # 점진 붕괴의 dof_torques_l2 초당 비. 양성 2.89, 음성 최대 1.03
    "value_loss_up": 2.2,            # Loss/value_function 비. 양성 3.76, 음성 최대 1.32
    "value_loss_up_max_rate": 79.0,  # 보상 크기 폭증과 구분. 양성 최대 10.2, 음성 최소 620
}


def ref_config(env: dict) -> dict:
    """기준 env params에서 정규화에 쓰는 값. 과제 B 작업공간에도 있는 reference/params/env.yaml에서 읽는다."""
    weights = {k: v["weight"] for k, v in env["rewards"].items() if isinstance(v, dict) and "weight" in v}
    return {"step_dt": env["sim"]["dt"] * env["decimation"], "episode_length_s": env["episode_length_s"],
            "reward_weights": weights}


def _m(summary: dict, tag: str):
    s = summary.get(tag)
    return None if s is None else s["mean_last_20pct"]


def _ratio(a, b):
    return None if a is None or b is None or abs(b) < EPS else a / b


def features(run: dict, ref: dict, cfg: dict) -> dict:
    """텔레메트리 summary 두 개(실패 실행, 같은 seed 기준 실행)에서 정규화 특징을 계산한다.

    Episode_Reward/<항>은 에피소드 합을 max_episode_length_s로 나눈 값이라 빨리 넘어지면 모든 항이 함께 작아진다.
    그래서 에피소드 길이 비로 나눠 초당 비(rate_ratio)로 바꾼다. Episode_Termination/<항>은 리셋 개수라 비율로 바꾼다.
    """
    len_run, len_ref = _m(run, LEN), _m(ref, LEN)
    if not len_run or not len_ref:
        raise ValueError(f"{LEN}이 없거나 0이다")
    len_ratio = len_run / len_ref
    to, bc = (_m(run, f"Episode_Termination/{t}") for t in ("time_out", "base_contact"))
    to, bc = (0.0 if x is None else x for x in (to, bc))  # 종료 항이 기록되지 않은 실행은 0회로 본다
    rate_ratio = {}
    for tag in sorted(run):
        if tag.startswith("Episode_Reward/"):
            r = _ratio(_m(run, tag), _m(ref, tag))
            if r is not None:
                rate_ratio[tag.split("/", 1)[1]] = r / len_ratio
    positive = [t for t, w in cfg["reward_weights"].items() if w > 0 and t in rate_ratio]
    tracking = [t for t in positive if t.startswith("track_")]
    ranked = sorted((abs(v) for v in rate_ratio.values()), reverse=True)
    return {
        "len_ratio": len_ratio,
        "len_pinned": (run[LEN]["max"] - len_run) / len_run,
        "timeout_frac": to / (to + bc) if to + bc > 0 else 0.0,
        "noise_ratio": _ratio(_m(run, "Policy/mean_noise_std"), _m(ref, "Policy/mean_noise_std")),
        "vf_ratio": _ratio(_m(run, "Loss/value_function"), _m(ref, "Loss/value_function")),
        "rate_ratio": rate_ratio,
        "max_rate": ranked[0] if ranked else 0.0,
        "dominance": ranked[0] / ranked[1] if len(ranked) > 1 and ranked[1] > EPS else 0.0,
        "sign_flips": sorted(t for t, v in rate_ratio.items() if v < 0),
        "tracking_rate": max((rate_ratio[t] for t in tracking), default=0.0),
        "positive_term_rate": max((rate_ratio[t] for t in positive), default=0.0),
        "torque_rate": rate_ratio.get("dof_torques_l2", 0.0),
    }


def _ranked(points: dict) -> list[str]:
    return sorted(MECHANISMS, key=lambda m: (-points.get(m, 0), PRIOR_ORDER.index(m)))


def prior_ranking() -> list[str]:
    return list(PRIOR_ORDER)


def feature_ranking(f: dict) -> tuple[list[str], list[dict]]:
    """고정 규칙. 강한 증거 3점, 보조 증거 2점을 더하고 동점은 PRIOR_ORDER로 정한다. (순위, 발화한 규칙)."""
    t = THRESHOLDS
    collapsed = f["len_ratio"] < t["collapsed_len_ratio"]
    survived = not collapsed
    immediate = f["len_ratio"] <= t["immediate_collapse"]
    time_limit = collapsed and f["timeout_frac"] >= t["timeout_majority"] and f["len_pinned"] <= t["pinned_len"]
    rules = [
        ("R1", "termination", 3, time_limit,
         "짧은 같은 길이에서 time_out으로 끝남: 에피소드 제한 시간"),
        ("R2", "exploration", 3, f["noise_ratio"] is not None and f["noise_ratio"] <= t["noise_collapse"],
         "정책 노이즈가 기준보다 크게 작음: 탐색 붕괴"),
        ("R3", "optimizer", 3, survived and f["vf_ratio"] is not None and f["vf_ratio"] <= t["value_loss_collapse"],
         "살아남았는데 가치 손실이 크게 작음: 짧은 할인 지평 등 학습 목표 변화"),
        ("R4", "reward", 3, bool(f["sign_flips"]),
         "보상 항 부호가 기준과 반대: 보상 정의 변경"),
        ("R5", "physics", 3, collapsed and not time_limit and f["tracking_rate"] >= t["tracking_rate_up"],
         "무너졌는데 추종 보상 초당 값이 기준보다 큼: 시간 간격이 가정과 다름"),
        ("R6", "reward", 3, f["dominance"] >= t["single_term_dominance"],
         "한 보상 항의 초당 비가 나머지를 압도: 그 항의 가중치 변경"),
        ("R7", "reward", 2, survived and f["positive_term_rate"] >= t["positive_term_up"],
         "살아남은 실행에서 양수 보상 항이 크게 늘어남: 보상 가중치 변경"),
        ("R8", "actuator", 2, immediate,
         "보행이 생기기 전에 넘어짐: 행동→관절 대응 변경"),
        ("R9", "physics", 2, collapsed and not immediate and f["timeout_frac"] < t["timeout_majority"]
         and f["torque_rate"] >= t["torque_rate_up"],
         "점진적으로 넘어지며 초당 토크가 늘어남: 하중·물리 변경"),
        ("R10", "optimizer", 2, not immediate and f["vf_ratio"] is not None and f["vf_ratio"] >= t["value_loss_up"]
         and f["max_rate"] <= t["value_loss_up_max_rate"],
         "보상 크기 폭증 없이 가치 손실이 커짐: 가치 학습 약화"),
    ]
    fired = [{"rule": rid, "mechanism": mech, "points": pts, "why": why}
             for rid, mech, pts, cond, why in rules if cond]
    points = {}
    for r in fired:
        points[r["mechanism"]] = points.get(r["mechanism"], 0) + r["points"]
    return _ranked(points), fired


def _slog(x: float) -> float:
    return math.copysign(math.log1p(abs(x)), x)


def template_vector(f: dict, terms: list[str]) -> list[float]:
    vec = [math.log(max(f["len_ratio"], EPS)), f["timeout_frac"],
           math.log(max(f["noise_ratio"] or EPS, EPS)), math.log(max(f["vf_ratio"] or EPS, EPS))]
    return vec + [_slog(f["rate_ratio"].get(t, 0.0)) for t in terms]


def template_ranking(f: dict, labelled: list[tuple[dict, str]]) -> tuple[list[str], dict]:
    """가장 가까운 dev 사례의 범주 순(유클리드 거리). labelled는 (특징, 정답 범주) 목록이다."""
    terms = sorted({t for g, _ in labelled for t in g["rate_ratio"]})
    v = template_vector(f, terms)
    best = {}
    for g, label in labelled:
        d = math.dist(v, template_vector(g, terms))
        best[label] = min(best.get(label, math.inf), d)
    order = sorted(MECHANISMS, key=lambda m: (best.get(m, math.inf), PRIOR_ORDER.index(m)))
    return order, {m: round(best[m], 4) for m in order if m in best}


# ---------- B0: 설정 diff ----------

# 실행 이름·seed처럼 실행마다 달라지는 식별 값. 결함 후보가 아니다.
IDENTITY_KEYS = ("agent.run_name", "agent.seed", "env.seed")
# 앞에서부터 처음 맞는 접두어의 범주를 쓴다. 맞는 것이 없으면 범주 없음(체크포인트 주기·명령 샘플링 등)이다.
KEY_MECHANISM = (
    ("env.rewards.", "reward"),
    ("env.actions.", "actuator"),
    ("env.scene.robot.actuators.", "actuator"),
    ("agent.policy.init_noise_std", "exploration"),
    ("agent.policy.noise_std_type", "exploration"),
    ("agent.algorithm.entropy_coef", "exploration"),
    ("agent.algorithm.", "optimizer"),
    ("agent.num_steps_per_env", "optimizer"),
    ("env.sim.", "physics"),
    ("env.decimation", "physics"),
    ("env.events.", "physics"),
    ("env.scene.robot.", "physics"),
    ("env.scene.terrain.", "physics"),
    ("env.episode_length_s", "termination"),
    ("env.terminations.", "termination"),
)


def mechanism_of(key: str) -> str | None:
    for prefix, mech in KEY_MECHANISM:
        if key == prefix or key.startswith(prefix):
            return mech
    return None


def flatten(cfg: dict, prefix: str) -> dict:
    out = {}
    for k, v in cfg.items():
        if isinstance(v, dict):
            out.update(flatten(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def flat_config(env: dict, agent: dict) -> dict:
    return {**flatten(env, "env."), **flatten(agent, "agent.")}


def _parse_value(text: str):
    try:
        return json.loads(text)
    except ValueError:
        return text


def apply_overrides(flat: dict, overrides: list[str]) -> dict:
    """hydra override(`env.a.b=값`)를 평탄화한 설정에 적용한다. 없는 키는 ValueError다."""
    out = dict(flat)
    for ov in overrides:
        key, value = ov.split("=", 1)
        if key not in out:
            raise ValueError(f"기준 설정에 없는 키: {key}")
        out[key] = _parse_value(value)
    return out


def config_diff(ref_flat: dict, run_flat: dict) -> list[dict]:
    keys = sorted((set(ref_flat) | set(run_flat)) - set(IDENTITY_KEYS))
    return [{"key": k, "old": ref_flat.get(k), "new": run_flat.get(k)} for k in keys
            if ref_flat.get(k) != run_flat.get(k)]


def _magnitude(old, new) -> float:
    """변경 크기. 같은 부호 수치는 |ln(new/old)|, 부호 반전·0 경계는 무한대, 목록은 원소 최댓값, 그 밖은 1."""
    if isinstance(old, list) and isinstance(new, list) and len(old) == len(new):
        return max((_magnitude(a, b) for a, b in zip(old, new)), default=0.0)
    num = (int, float)
    if isinstance(old, num) and isinstance(new, num) and not isinstance(old, bool) and not isinstance(new, bool):
        if old == new:
            return 0.0
        if old == 0 or new == 0 or (old > 0) != (new > 0):
            return math.inf
        return abs(math.log(new / old))
    return 0.0 if old == new else 1.0


def b0_ranking(diff: list[dict]) -> tuple[list[str], list[dict]]:
    """바뀐 키의 범주를 변경 크기 순으로 낸다. 바뀌지 않은 범주는 PRIOR_ORDER로 뒤에 붙인다."""
    scored = []
    for d in diff:
        mech = mechanism_of(d["key"])
        mag = _magnitude(d["old"], d["new"])
        scored.append({**d, "mechanism": mech, "magnitude": "inf" if math.isinf(mag) else round(mag, 4), "_m": mag})
    best = {}
    for s in scored:
        if s["mechanism"]:
            best[s["mechanism"]] = max(best.get(s["mechanism"], -1.0), s["_m"])
    order = sorted(best, key=lambda m: (-best[m], PRIOR_ORDER.index(m)))
    order += [m for m in PRIOR_ORDER if m not in best]
    return order, [{k: v for k, v in s.items() if k != "_m"} for s in scored]
