"""R1 수리 판의 검사 등록, 모델 계약과 교정 범위. GPU와 API 부수효과가 없다."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rl_triage import active_diagnosis as D
from rl_triage import correction_registry as C
from rl_triage import mechanism_registry as M
from rl_triage import probe_loop as L

MODEL = "nvidia/nemotron-3-super-120b-a12b"
VALUE = "optimizer_value"
COUNT = "optimizer_update_count"
NEW_PROBE = "P_optimizer_updates"
CORRECTION = "restore_configured_ppo_epochs"
MAX_PROBES = 7


def registry() -> dict:
    """기존 optimizer 예상은 value 하위형에 보존하고 update 횟수 하위형을 등록한다."""
    mechanisms = [VALUE if name == "optimizer" else name for name in L.MECHANISMS] + [COUNT]
    probes = deepcopy(L.PROBES)
    for spec in probes.values():
        spec["expect"][VALUE] = spec["expect"].pop("optimizer")
        spec["expect"][COUNT] = "normal"
    probes[NEW_PROBE] = {
        "measures": "같은 실행 설정으로 새로 시작한 PPO 3 update의 optimizer.step 및 미니배치 횟수. 정책 체크포인트 관측이 아님",
        "cost_gpu_s": 60,
        "args": {"num_envs": 4096, "updates": 3, "configured_learning_epochs": 5,
                 "configured_mini_batches": 4, "expected_steps_per_update": 20},
        "expect": {name: "abnormal" if name == COUNT else "normal" for name in mechanisms},
    }
    return M.normalize({"mechanisms": mechanisms, "outcomes": list(L.OUTCOMES), "probes": probes,
                        "fixed_order": [*L.FIXED_ORDER, NEW_PROBE]})


def category(name: str) -> str:
    return "optimizer" if name in (VALUE, COUNT) else name


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
        hypotheses.extend([VALUE, COUNT] if name == "optimizer" else [name])
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


def correction_registry() -> dict:
    reg = deepcopy(C.REGISTRY)
    reg["optimizer"] = {"correction_id": CORRECTION,
                        "adapter_target": "rsl_rl.algorithms.ppo:PPO.num_learning_epochs",
                        "protocol_case": "optimizer_epoch_count", "recovery_modes": ["restart"]}
    return reg


def correction_allowed(state: dict) -> bool:
    return (state.get("status") == "confirmed" and state.get("hypotheses") == [COUNT]
            and state.get("observed", {}).get(NEW_PROBE) == "abnormal")


SYSTEM = """You are a blind RL failure-diagnosis policy. Use only supplied evidence and the public registry.
Return one JSON object with exactly ranking, probe, action, reason, claims, expected_outcomes.
ranking is a permutation of all six public categories. action is continue or hold for this initial decision.
For continue, choose one public probe and copy its category expectations for the top three ranked categories
into expected_outcomes. For hold, probe is null and expected_outcomes is {}.
reason is a nonempty explanation. claims is a nonempty list of {id, value}, copied exactly from evidence_facts.
Configuration values are not runtime observations. Do not infer hidden case names or claim a confirmed cause.
The optimizer category has two registered subtypes. A null category expectation means the subtypes differ.
The executor preserves all categories, expands optimizer into both subtypes, runs your chosen probe first,
then uses the unchanged registry elimination/confirmation rules. A normal value-learning probe does not
eliminate the update-count subtype. Unknown execution stops the loop without confirmation.
Output JSON only."""


def diagnosis_messages(packet: dict) -> list[dict]:
    if (packet.get("schema") != "r1_repair_diagnosis_packet_v1" or packet.get("observed") != []
            or packet.get("budget_left") != MAX_PROBES):
        raise ValueError("초기 진단 packet 계약이 다르다")
    facts = packet.get("facts")
    if not isinstance(facts, dict) or not facts or any(not isinstance(k, str) or not D._is_scalar(v) for k, v in facts.items()):
        raise ValueError("진단 사실은 비지 않은 유한 scalar 표여야 한다")
    body = {"evidence_facts": facts, "categories": list(L.MECHANISMS), "probe_catalog": public_catalog(),
            "subtype_registry": registry(), "thresholds": {k: list(v) for k, v in L.THRESHOLDS.items()},
            "update_count_rule": "3 complete updates; steps and minibatches equal 20 each => normal; any mismatch => abnormal; incomplete => unknown",
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
        raise ValueError("epoch 횟수 하위 가설이 확정되지 않아 교정 범위가 맞지 않는다")
    rule = {"mode": "restart", "correction_id": CORRECTION}
    facts = {
        "confirmed_mechanism": "optimizer", "confirmed_subtype": COUNT,
        "confirming_probe": NEW_PROBE, "confirming_probe_outcome": confirming["outcome"],
        "confirming_probe_measurement": confirming["measurement"],
        "confirming_probe_measures": registry()["probes"][NEW_PROBE]["measures"],
        "probe_observations": state["observed"], "eval_protocol": report["protocol"],
        "eval_task": report["task"], "eval_metrics": report["metrics"],
        "eval_standstill": report["standstill"], "rule_mode": "restart",
        "rule": "optimizer_update_count abnormal 확인 뒤 configured epoch 5 복원, 새 초기화 300회",
        "recovery_selection_basis": "정상 epoch 학습 근거에 따라 새 초기화 복원을 시험한다. R1 이어 학습과의 비교 결과는 없다",
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
