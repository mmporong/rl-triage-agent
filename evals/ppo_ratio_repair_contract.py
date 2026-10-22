"""PPO ratio 계산 오류 종단 판의 검사 등록, 모델 계약과 교정 범위. GPU와 API 부수효과가 없다.

R1 판(r1_repair_contract.py)의 optimizer 하위형 두 개(value·update_count)에 surrogate 계산 하위형을 더한다.
기존 여섯 검사와 R1 갱신 횟수 검사의 정의·임계값은 그대로 두고, 새 검사 P_ppo_gradient만 등록한다.
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
import r1_repair_contract as R1  # noqa: E402
from rl_triage import active_diagnosis as D  # noqa: E402
from rl_triage import correction_registry as C  # noqa: E402
from rl_triage import mechanism_registry as M  # noqa: E402
from rl_triage import probe_loop as L  # noqa: E402

MODEL = R1.MODEL
VALUE, COUNT, SURROGATE = R1.VALUE, R1.COUNT, "optimizer_surrogate"
SUBTYPES = (VALUE, COUNT, SURROGATE)
COUNT_PROBE, GRADIENT_PROBE = R1.NEW_PROBE, "P_ppo_gradient"
CORRECTION = "restore_ppo_ratio_formula"
MAX_PROBES = 8
PACKET_SCHEMA = "ppo_ratio_repair_diagnosis_packet_v1"
GRADIENT_RULE = ("3 complete updates; in every minibatch the clipped-surrogate gradient of the new log-prob equals "
                 "-advantage*ratio where clipping passes it and 0 elsewhere, ratio=exp(new-old), within atol 1e-6 "
                 "and rtol 1e-5 => normal; any mismatch => abnormal; incomplete or non-finite => unknown")


def registry() -> dict:
    """R1 등록 표에 surrogate 하위형과 gradient 검사를 더한다. 기존 검사의 예상은 surrogate에 normal이다."""
    base = R1.registry()
    mechanisms = [*base["mechanisms"], SURROGATE]
    probes = deepcopy(base["probes"])
    for spec in probes.values():
        spec["expect"][SURROGATE] = "normal"
    probes[GRADIENT_PROBE] = {
        "measures": ("같은 실행 설정으로 새로 시작한 PPO 3 update에서 clipped surrogate의 새 log-prob gradient와 "
                     "저장 advantage·ratio로 계산한 기대 gradient의 불일치. 정책 체크포인트 관측이 아님"),
        "cost_gpu_s": 60,
        "args": {"num_envs": 4096, "updates": 3, "configured_learning_epochs": 5, "configured_mini_batches": 4,
                 "clip_param": 0.2, "atol": 1.0e-6, "rtol": 1.0e-5},
        "expect": {name: "abnormal" if name == SURROGATE else "normal" for name in mechanisms},
    }
    return M.normalize({"mechanisms": mechanisms, "outcomes": list(L.OUTCOMES), "probes": probes,
                        "fixed_order": [*base["fixed_order"], GRADIENT_PROBE]})


def category(name: str) -> str:
    return "optimizer" if name in SUBTYPES else name


def public_catalog() -> dict:
    result = {}
    for name, spec in registry()["probes"].items():
        expected = {}
        for mechanism in L.MECHANISMS:
            values = {outcome for subtype, outcome in spec["expect"].items() if category(subtype) == mechanism}
            expected[mechanism] = next(iter(values)) if len(values) == 1 else None
        result[name] = {**deepcopy(spec), "expect": expected}
    return result


def initial_state(decision: dict) -> dict:
    # 순위는 첫 검사 선택에 쓰며, 검사하지 않은 하위 가설을 순위만으로 기각하지 않는다.
    hypotheses = []
    for name in decision["ranking"]:
        hypotheses.extend(SUBTYPES if name == "optimizer" else [name])
    return {"hypotheses": hypotheses, "observed": {}, "history": [], "status": "open",
            "budget_left": MAX_PROBES, "next_probe": decision["probe"]}


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


def correction_registry() -> dict:
    reg = deepcopy(C.REGISTRY)
    reg["optimizer"] = {"correction_id": CORRECTION, "adapter_target": "rsl_rl.algorithms.ppo:PPO.update",
                        "protocol_case": "optimizer_surrogate_ratio", "recovery_modes": ["restart"]}
    return reg


def correction_allowed(state: dict) -> bool:
    return (state.get("status") == "confirmed" and state.get("hypotheses") == [SURROGATE]
            and state.get("observed", {}).get(GRADIENT_PROBE) == "abnormal")


SYSTEM = """You are a blind RL failure-diagnosis policy. Use only supplied evidence and the public registry.
Return one JSON object with exactly ranking, probe, action, reason, claims, expected_outcomes.
ranking is a permutation of all six public categories. action is continue or hold for this initial decision.
For continue, choose one public probe and copy its category expectations for the top three ranked categories
into expected_outcomes. For hold, probe is null and expected_outcomes is {}.
reason is a nonempty explanation. claims is a nonempty list of {id, value}, copied exactly from evidence_facts.
Configuration values are not runtime observations. Do not infer hidden case names or claim a confirmed cause.
The optimizer category has three registered subtypes. A null category expectation means the subtypes differ.
The executor preserves all categories, expands optimizer into all three subtypes, runs your chosen probe first,
then uses the unchanged registry elimination/confirmation rules. A normal value-learning or update-count probe
does not eliminate the surrogate-calculation subtype. Unknown execution stops the loop without confirmation.
Output JSON only."""


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


def correction_packet(state: dict, confirming: dict, report: dict, checkpoint_sha: str, report_sha: str) -> dict:
    import correction_model_call as CM
    if not correction_allowed(state):
        raise ValueError("surrogate 계산 하위 가설이 확정되지 않아 교정 범위가 맞지 않는다")
    rule = {"mode": "restart", "correction_id": CORRECTION}
    facts = {
        "confirmed_mechanism": "optimizer", "confirmed_subtype": SURROGATE,
        "confirming_probe": GRADIENT_PROBE, "confirming_probe_outcome": confirming["outcome"],
        "confirming_probe_measurement": confirming["measurement"],
        "confirming_probe_measures": registry()["probes"][GRADIENT_PROBE]["measures"],
        "probe_observations": state["observed"], "eval_protocol": report["protocol"],
        "eval_task": report["task"], "eval_metrics": report["metrics"],
        "eval_standstill": report["standstill"], "rule_mode": "restart",
        "rule": "optimizer_surrogate abnormal 확인 뒤 PPO.update의 ratio 식을 설치 원본으로 복원, 새 초기화 300회",
        "recovery_selection_basis": "정상 ratio 식 학습 근거에 따라 새 초기화 복원을 시험한다. 오류 정책 이어 학습과의 비교 결과는 없다",
        "registered_correction": CORRECTION, "registered_recovery_modes": ["restart"],
        "checkpoint_sha256": checkpoint_sha, "source_report_sha256": report_sha,
    }
    table = {m: {"correction_id": e["correction_id"], "recovery_modes": e["recovery_modes"]}
             for m, e in correction_registry().items()}
    return {"schema": CM.PACKET_SCHEMA, "case": "blind_optimizer_repair", "mechanism": "optimizer",
            "loop_state_digest": L.digest(state), "receipts_digest": L.digest(confirming),
            "confirming_request_id": confirming["request_id"], "facts": facts, "registry": table,
            "rule_result": rule, "checklist": CM.skill_checklist(),
            "skill_source": {"url": CM.SKILL_URL, "commit": CM.SKILL_COMMIT, "sha256": CM.SKILL_SHA256}}
