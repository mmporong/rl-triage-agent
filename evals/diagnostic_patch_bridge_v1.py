"""확인 검사와 제한된 생성 패치의 source·receipt 경계를 연결한다."""
from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable

import ppo_patch_contract_v1 as CONTRACT

PROBE = "P_optimizer_updates_cpu_adapter"
SCHEMA = "diagnostic_generated_patch_bridge_v1"


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def diagnosis_context(source: str, config: dict, *, case_id: str) -> dict:
    CONTRACT.validate_source(source)
    if type(config) is not dict or not isinstance(case_id, str) or not case_id:
        raise ValueError("source 조사 입력이 잘못됐다")
    return {"schema": "ppo_adapter_diagnosis_context_v1", "case_id": case_id,
            "source": source, "source_sha256": CONTRACT.source_sha256(source),
            "config": copy.deepcopy(config), "facts": {
                "configured.epochs": config["num_learning_epochs"],
                "configured.minibatches": config["num_mini_batches"]},
            "available_probes": [{"id": PROBE, "backend": "actual_rsl_cpu_tensor_abi",
                                  "measures": "configured versus actual PPO minibatch and optimizer update counts"}],
            "scope": "focused adapter investigation; robot behavior has not been measured"}


def validate_choice(choice: dict, context: dict) -> dict:
    if type(choice) is not dict or set(choice) != {"action", "probe", "source_sha256", "fact_ids", "rationale"}:
        raise ValueError("진단 선택 필드가 계약과 다르다")
    if choice["source_sha256"] != CONTRACT.source_sha256(context["source"]):
        raise ValueError("진단 선택 source SHA가 다르다")
    if context["source_sha256"] != choice["source_sha256"]:
        raise ValueError("진단 context source 결합이 다르다")
    ids = choice["fact_ids"]
    if not isinstance(ids, list) or not ids or len(ids) != len(set(ids)) or any(
            not isinstance(x, str) or x not in context["facts"] for x in ids):
        raise ValueError("진단 선택 fact 인용이 잘못됐다")
    if not isinstance(choice["rationale"], str) or not 0 < len(choice["rationale"]) <= 1000:
        raise ValueError("진단 선택 이유가 잘못됐다")
    if not ((choice["action"] == "inspect" and choice["probe"] == PROBE)
            or (choice["action"] == "hold" and choice["probe"] is None)):
        raise ValueError("등록된 검사 또는 HOLD만 선택할 수 있다")
    return copy.deepcopy(choice)


def bind_confirmation(context: dict, choice: dict, baseline: dict, reference: dict,
                      authorization: dict) -> dict:
    """새 CPU backend의 실제 검사 원문을 묶는다. 기존 GPU receipt로 위장하지 않는다."""
    validate_choice(choice, context)
    if choice["action"] != "inspect":
        raise ValueError("HOLD 선택으로 패치 단계를 시작할 수 없다")
    if authorization != {"approved": True, "principal": "user_delegated_host",
                         "scope": "one_cpu_adapter_inspection", "context_sha256": digest(context)}:
        raise ValueError("검사 승인 결합이 다르다")
    if authorization["approved"] is not True:
        raise ValueError("검사 승인은 bool true여야 한다")
    if baseline.get("source_sha256") != context["source_sha256"]:
        raise ValueError("원본 검사 source가 진단 입력과 다르다")
    import ppo_patch_cases_v1 as CASES

    from evals import ppo_patch_feedback_v1 as FEEDBACK
    FEEDBACK.verify_validation(context["source"], baseline)
    FEEDBACK.verify_validation(CASES.SOURCES["healthy"], reference)
    if reference.get("all_passed") is not True:
        raise ValueError("정상 참조 CPU 검사가 통과하지 못했다")
    rows = baseline.get("records")
    if not isinstance(rows, list) or not rows:
        raise ValueError("실제 검사 기록이 없다")
    first = rows[0]
    if first.get("configuration") != context["config"]:
        raise ValueError("진단 설정과 확인 검사 설정이 다르다")
    measurement = first.get("measurement")
    if not isinstance(measurement, dict) or measurement.get("status") != "passed":
        raise ValueError("원본 PPO CPU 실행이 완료되지 않았다")
    per = measurement.get("measurement", {}).get("per_update", [])
    if len(per) != 3:
        raise ValueError("확인 검사는 실제 CPU update 세 개가 필요하다")
    expected = context["config"]["num_learning_epochs"] * context["config"]["num_mini_batches"]
    counts = [(r.get("optimizer_steps"), r.get("minibatches")) for r in per]
    if any(type(x) is not int or x < 0 for pair in counts for x in pair):
        raise ValueError("원본 갱신 수가 유효하지 않다")
    abnormal = any(pair != (expected, expected) for pair in counts)
    facts = copy.deepcopy(context["facts"])
    facts["inspection.expected_steps"] = expected
    facts["inspection.actual_steps"] = [x[0] for x in counts]
    facts["inspection.actual_minibatches"] = [x[1] for x in counts]
    return {"schema": SCHEMA, "backend": "actual_rsl_cpu_tensor_abi", "probe": PROBE,
            "source_sha256": context["source_sha256"], "diagnosis_context_sha256": digest(context),
            "choice_sha256": digest(choice), "authorization_sha256": digest(authorization),
            "baseline_sha256": digest(baseline), "reference_sha256": digest(reference),
            "outcome": "abnormal" if abnormal else "normal", "facts": facts,
            "patch_eligible": abnormal and baseline.get("all_passed") is False,
            "normal_source_preserved": not abnormal, "GPU_approval_granted": False}


def patch_context(context: dict, confirmation: dict, baseline: dict, reference: dict) -> dict:
    if confirmation.get("schema") != SCHEMA or confirmation.get("patch_eligible") is not True:
        raise ValueError("유효한 abnormal 확인 검사 없이는 패치를 생성하지 않는다")
    if confirmation["diagnosis_context_sha256"] != digest(context):
        raise ValueError("확인 검사의 진단 계보가 다르다")
    if confirmation["baseline_sha256"] != digest(baseline) or confirmation["reference_sha256"] != digest(reference):
        raise ValueError("확인 검사 원문이 달라졌다")
    if baseline["source_sha256"] != CONTRACT.source_sha256(context["source"]):
        raise ValueError("패치 대상과 원본 CPU 검사의 source가 다르다")
    return {"schema": "ppo_patch_context_v1", "source": context["source"],
            "source_sha256": context["source_sha256"], "config": copy.deepcopy(context["config"]),
            "facts": copy.deepcopy(confirmation["facts"]), "diagnosis_confirmation": copy.deepcopy(confirmation),
            "observation_scope": "registered focused CPU optimizer check; robot recovery requires separate fixed evaluation"}


def verify_legacy_diagnosis(result: dict, applied: list[str], *, verifier: Callable | None = None) -> dict:
    """기존 고리의 full receipt 검증을 위임하며 summary만으로 실행 권한을 만들지 않는다."""
    if verifier is None:
        from runtime_observation_recovery_v1 import verify_diagnosis
        verifier = verify_diagnosis
    component, correction = verifier(result, applied)
    if component != "optimizer_update_count" or correction != "restore_configured_ppo_epochs":
        raise ValueError("이 연결은 확인된 optimizer update count만 지원한다")
    return {"schema": "legacy_diagnosis_generated_patch_handoff_v1", "component": component,
            "registered_alternative": correction, "diagnosis_sha256": digest(result),
            "selected_path": "generated_patch", "registered_correction_executed": False,
            "execution_allowed": False, "GPU_approval_granted": False}
