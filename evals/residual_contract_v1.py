"""남은 실패 재진단: epoch 지역 가설과 누적 교정의 경계를 검증한다."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))
import r1_repair_contract as R
import correction_model_call as CM

MODEL, L, M = R.MODEL, R.L, R.M
MAX_PROBES, NEW_PROBE = R.MAX_PROBES, R.NEW_PROBE
REWARD = "restore_body_frame_reward"
COUNT = R.COUNT
EPOCHS = R.CORRECTION
CORRECTIONS = {"reward": REWARD, COUNT: EPOCHS}
SCHEMA = "residual_diagnosis_packet_v1"
registry = R.registry


def validate_applied(applied: object) -> list:
    if (not isinstance(applied, list) or len(applied) > 2
            or any(not isinstance(x, str) or x not in CORRECTIONS.values() for x in applied)
            or len(set(applied)) != len(applied)):
        raise ValueError("서로 다른 등록 교정은 최대 두 개다")
    return list(applied)


def unresolved_components(applied: list) -> list:
    validate_applied(applied)
    removed = {key for key, value in CORRECTIONS.items() if value in applied}
    return [key for key in registry()["mechanisms"] if key not in removed]


def _legacy_packet(packet: dict) -> dict:
    if packet.get("schema") != SCHEMA or type(packet.get("epoch")) is not int or packet["epoch"] not in (1, 2):
        raise ValueError("재진단 epoch 계약이 다르다")
    applied = validate_applied(packet.get("applied_corrections"))
    if len(applied) != packet["epoch"] - 1:
        raise ValueError("epoch와 앞선 교정 수가 다르다")
    prior = packet.get("prior_receipt")
    if packet["epoch"] == 1 and prior is not None:
        raise ValueError("첫 epoch에는 앞선 receipt가 없다")
    if packet["epoch"] == 2 and (not isinstance(prior, dict) or prior.get("verdict") != "not_recovered"
            or prior.get("correction_id") != applied[0]
            or set(prior) != {"verdict", "correction_id", "result_sha256", "checkpoint_sha256"}
            or any(not isinstance(prior[k], str) or len(prior[k]) != 64
                   or any(c not in "0123456789abcdef" for c in prior[k])
                   for k in ("result_sha256", "checkpoint_sha256"))):
        raise ValueError("두 번째 epoch에는 검증한 비회복 receipt가 필요하다")
    return {**packet, "schema": "r1_repair_diagnosis_packet_v1"}


def diagnosis_messages(packet: dict) -> list[dict]:
    messages = R.diagnosis_messages(_legacy_packet(packet))
    messages[0]["content"] += (" A confirmation names ONE repairable component in this epoch, not all causes. "
        "Previous local eliminations do not exclude residual causes. Each epoch starts fresh observations. "
        "Only a successfully applied correction removes its exact subtype from the next epoch candidates.")
    body = json.loads(messages[1]["content"])
    body.update(epoch=packet["epoch"], applied_corrections=packet["applied_corrections"],
                prior_receipt=packet["prior_receipt"],
                unresolved_components=unresolved_components(packet["applied_corrections"]))
    messages[1]["content"] = json.dumps(body, ensure_ascii=False, sort_keys=True)
    return messages


def validate_diagnosis(decision: dict, packet: dict) -> dict:
    return R.validate_diagnosis(decision, _legacy_packet(packet))


def initial_state(decision: dict, applied: list) -> dict:
    candidates = unresolved_components(applied)
    state = R.initial_state(decision)
    state["hypotheses"] = [x for x in state["hypotheses"] if x in candidates]
    return state


advance = R.advance


def correction_registry() -> dict:
    source = R.correction_registry()
    return {"reward": {**source["reward"], "recovery_modes": ["restart"]},
            COUNT: deepcopy(source["optimizer"])}


def confirmed_component(state: dict, applied: list) -> str | None:
    validate_applied(applied)
    if state.get("status") != "confirmed" or len(state.get("hypotheses", [])) != 1:
        return None
    component = state["hypotheses"][0]
    probe = "P_reward" if component == "reward" else NEW_PROBE if component == COUNT else None
    if (component not in CORRECTIONS or CORRECTIONS[component] in applied
            or state.get("observed", {}).get(probe) != "abnormal"):
        return None
    return component


def correction_packet(state: dict, confirming: dict, report: dict, checkpoint_sha: str,
                      report_sha: str, applied: list) -> dict:
    component = confirmed_component(state, applied)
    if component is None or len(applied) >= 2:
        raise ValueError("미확정·중복·범위 밖 교정은 HOLD다")
    probe = "P_reward" if component == "reward" else NEW_PROBE
    if confirming.get("probe") != probe or confirming.get("outcome") != "abnormal":
        raise ValueError("확정 성분의 abnormal receipt가 없다")
    mechanism, cid = R.category(component), CORRECTIONS[component]
    facts = {"confirmed_mechanism": mechanism, "confirmed_subtype": component,
             "confirming_probe": probe, "confirming_probe_outcome": confirming["outcome"],
             "confirming_probe_measurement": confirming["measurement"],
             "confirming_probe_measures": registry()["probes"][probe]["measures"],
             "probe_observations": state["observed"], "eval_protocol": report["protocol"],
             "eval_task": report["task"], "eval_metrics": report["metrics"],
             "eval_standstill": report["standstill"], "rule_mode": "restart",
             "rule": "이번 성분의 등록 교정을 누적 적용하고 같은 seed로 새 초기화 300회",
             "recovery_selection_basis": "보상 v4·R1 재학습 근거를 사용한 복합 개발 시험. 이어 학습 비교는 이 판에 없다",
             "registered_correction": cid, "registered_recovery_modes": ["restart"],
             "checkpoint_sha256": checkpoint_sha, "source_report_sha256": report_sha,
             "applied_corrections": list(applied), "confirmation_scope": "one_component_this_epoch"}
    table = {R.category(key): {"correction_id": row["correction_id"], "recovery_modes": ["restart"]}
             for key, row in correction_registry().items()}
    return {"schema": CM.PACKET_SCHEMA, "case": "blind_residual_repair", "mechanism": mechanism,
            "loop_state_digest": L.digest(state), "receipts_digest": L.digest(confirming),
            "confirming_request_id": confirming["request_id"], "facts": facts, "registry": table,
            "rule_result": {"mode": "restart", "correction_id": cid}, "checklist": CM.skill_checklist(),
            "skill_source": {"url": CM.SKILL_URL, "commit": CM.SKILL_COMMIT, "sha256": CM.SKILL_SHA256}}
