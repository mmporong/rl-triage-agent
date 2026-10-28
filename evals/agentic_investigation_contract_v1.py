"""등록된 자료 조회·검사·교정 제안을 위한 CPU 행동 계약.

모델 출력은 승인이나 실행 인자가 아니다. 기존 controller가 실제 검사·교정의
승인, raw receipt와 회복 판정을 검증한다. 이 모듈은 API/GPU를 실행하지 않는다.
"""
from __future__ import annotations

import json
import math
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "src")]
import selection_comparison_controller_v1 as C
import vision_selection_contract_v1 as V

SCHEMA = "agentic_investigation_context_v1"
OUTPUT_FIELDS = {"action", "target", "fact_ids"}
SYSTEM = """Investigate the user's RL training failure using the registered tools and evidence.
Choose one action: read_evidence, inspect_frames, run_probe, propose_correction, or hold.
Return exactly action, target, fact_ids as one JSON object. hold requires target=null.
Use only listed targets and cite nonempty distinct IDs in evidence_facts. A visual description
is an interpretation, not a confirmed cause or a recovery verdict. Probe results can change
the next action. Only host-confirmed components may receive a listed registered correction.
The host supplies arguments, thresholds, approval, execution and recovery evaluation.
Never generate commands, expected values, approval, new corrections or a recovery verdict.
Budget exhaustion and unknown observations require hold. Output JSON only."""


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def public_facts(facts: dict) -> dict:
    """기존 수치와 원본 frame 메타데이터만 허용하며 파일 경로·평가 라벨은 거부한다."""
    _require(isinstance(facts, dict), "공개 evidence facts 형식")
    for key, value in facts.items():
        _require(isinstance(key, str), "fact ID 형식")
        if re.fullmatch(r"vision:frame:f\d+", key):
            _require(isinstance(value, dict)
                     and set(value) == {"evidence_type", "timestamp_s", "source_group"}
                     and value["evidence_type"] == "raw_frame"
                     and value["source_group"] == "visual_source_1"
                     and type(value["timestamp_s"]) in (int, float)
                     and math.isfinite(value["timestamp_s"]) and value["timestamp_s"] >= 0,
                     "원본 frame 근거 형식")
        else:
            allowed = key in C.V.BASE_FACT_IDS or bool(re.fullmatch(
                r"vision:play:f\d+:(?:timestamp_s|(?:command|motion):(?:linear_x_mps|linear_y_mps|yaw_rate_rps)"
                r"|pose:(?:base_height_m|up_axis_z))", key))
            _require(allowed and (value is None or isinstance(value, bool)
                     or type(value) in (int, float) and math.isfinite(value)),
                     "공개 수치 fact가 아니거나 경로·정답이 포함됨")
    return deepcopy(facts)


def prepare(snapshot: dict, *, goal: str, evidence: dict[str, str] | None = None,
            revealed: list[str] | None = None, extra_facts: dict | None = None,
            decisions_left: int = 16) -> dict:
    """기존 snapshot의 공개 view에 조회 가능한 자료 ID만 추가한다."""
    checked = C.V.checked(snapshot)
    _require(isinstance(goal, str) and 0 < len(goal) <= 1000, "목표 형식")
    _require(type(decisions_left) is int and decisions_left >= 0, "판단 예산 형식")
    evidence, revealed = deepcopy(evidence or {}), list(revealed or [])
    _require(isinstance(evidence, dict) and all(
        isinstance(k, str) and bool(re.fullmatch(r"evidence:[a-z0-9_-]+", k))
        and v in {"training", "frames"} for k, v in evidence.items()), "조회 자료 목록 형식")
    _require(len(revealed) == len(set(revealed)) and set(revealed) <= set(evidence), "조회 자료 이력")
    view = deepcopy(checked["view"])
    additional = public_facts(extra_facts or {})
    _require(not set(additional) & set(view["evidence_facts"]), "기존 fact 덮어쓰기")
    view["evidence_facts"].update(additional)
    corrections = []
    if view["status"] == "confirmed":
        try:
            _, correction = C.registered_correction(checked, checked["inputs"]["packet"]["applied_corrections"])
            corrections = [correction]
        except ValueError:
            pass  # 등록 교정이 없으면 모델에 교정 실행 후보를 제공하지 않는다.
    active = decisions_left > 0 and view["status"] in {"open", "confirmed"}
    targets = {"read_evidence": [k for k, v in evidence.items()
                                 if active and v == "training" and k not in revealed],
               "inspect_frames": [k for k, v in evidence.items()
                                  if active and v == "frames" and k not in revealed],
               "run_probe": list(view["eligible_probes"]) if active else [],
               "propose_correction": corrections if active else [], "hold": [None]}
    value = {"schema": SCHEMA, "snapshot": checked, "goal": goal, "evidence": evidence,
             "revealed": revealed, "extra_facts": additional, "decisions_left": decisions_left,
             "model_view": {"goal": goal, **view, "available_actions": targets,
                            "decisions_left": decisions_left}}
    value["context_digest"] = C.L.digest(value)
    return value


def checked(context: dict) -> dict:
    _require(isinstance(context, dict), "행동 context 형식")
    expected = prepare(context["snapshot"], **{k: context[k] for k in (
        "goal", "evidence", "revealed", "extra_facts", "decisions_left")})
    _require(context == expected, "행동 context 결합 불일치")
    return expected


def messages(context: dict) -> list[dict]:
    value = checked(context)
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": json.dumps(value["model_view"], ensure_ascii=False,
                                                    sort_keys=True, allow_nan=False)}]


def assess(content: Any, context: dict) -> dict:
    value = checked(context)
    try:
        _require(isinstance(content, str), "응답은 JSON 문자열이어야 함")
        decision = json.loads(content, object_pairs_hook=V._unique_object)
        _require(isinstance(decision, dict) and set(decision) == OUTPUT_FIELDS, "행동 출력 필드")
        action, target, facts = (decision[k] for k in ("action", "target", "fact_ids"))
        targets = value["model_view"]["available_actions"]
        _require(isinstance(action, str) and action in targets, "미등록 행동")
        _require(target in targets[action], "등록·현재 상태·예산 밖의 대상")
        _require(isinstance(facts, list) and bool(facts)
                 and all(isinstance(item, str) for item in facts)
                 and len(set(facts)) == len(facts)
                 and set(facts) <= set(value["model_view"]["evidence_facts"]), "없는 fact 또는 중복 근거")
        return {"status": "valid", "decision": decision, "codes": [],
                "context_digest": value["context_digest"], "approval_granted": False}
    except (ValueError, TypeError, KeyError) as error:
        return {"status": "invalid", "decision": None, "codes": [str(error)],
                "context_digest": value["context_digest"], "approval_granted": False}


def probe_choice(context: dict, review: dict) -> dict:
    """행동 검증을 기존 controller의 승인 전 검사 선택으로 변환한다."""
    value = checked(context)
    _require(review.get("status") == "valid" and review.get("context_digest") == value["context_digest"]
             and review.get("approval_granted") is False, "유효한 현재 행동이 아님")
    decision = review["decision"]
    _require(decision["action"] == "run_probe", "검사 선택이 아님")
    _require(assess(json.dumps(decision), value) == review, "행동 검증 결과 변조")
    snapshot = value["snapshot"]
    return {"schema": "selection_controller_choice_v1",
            "context_digest": C.L.digest(snapshot["binding"]), "view_digest": snapshot["view_digest"],
            "policy": snapshot["inputs"]["policy"], "selected_by": "model", "action": "continue",
            "probe": decision["target"], "fact_ids": decision["fact_ids"], "model_status": "valid",
            "codes": [], "reason": "agentic_registered_action", "used_for_decision": True,
            "approval_granted": False}
