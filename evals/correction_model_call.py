"""E2 선택 구성 4: 모델 교정 제안의 프롬프트 생성과 검증. 이 파일은 모델·네트워크를 호출하지 않는다.

모델은 {correction_id, recovery_mode, cited_facts} JSON만 낸다. 검증은 등록 표와 일치하는지,
인용한 사실이 제안 근거(evidence["facts"])에 있는지만 본다. 결정적 규칙과의 일치 여부는 기록만 하고 우위를 주장하지 않는다.
실제 호출은 메인 오케스트레이터가 별도 tag로 수행한다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rl_triage.correction_registry import REGISTRY  # noqa: E402

SYSTEM = ("RL 진단이 확정된 뒤 교정 하나와 회복 방식 하나를 고른다. 등록 표 밖의 값은 쓰지 않는다. "
          "출력은 JSON 한 개 {\"correction_id\": str, \"recovery_mode\": \"restart\"|\"resume\", "
          "\"cited_facts\": [근거 이름...]}뿐이며 cited_facts는 주어진 facts의 키만 인용한다.")


def build_messages(evidence: dict, registry: dict | None = None) -> list[dict]:
    """chat 형식 메시지. evidence = {mechanism, facts: {이름: 값}} (정답·결함 정의는 넣지 않는다)."""
    reg = REGISTRY if registry is None else registry
    table = {m: {"correction_id": e["correction_id"], "recovery_modes": e["recovery_modes"]} for m, e in reg.items()}
    user = {"confirmed_mechanism": evidence["mechanism"], "facts": evidence["facts"], "registry": table}
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False, sort_keys=True)}]


def validate_model_proposal(proposal, evidence: dict, registry: dict | None = None) -> dict:
    """{accepted, reasons, agrees_with_rule}. 불일치는 거부 사유로 남기고 예외를 던지지 않는다."""
    reg = REGISTRY if registry is None else registry
    reasons: list[str] = []
    if not isinstance(proposal, dict) or set(proposal) != {"correction_id", "recovery_mode", "cited_facts"}:
        return {"accepted": False, "reasons": ["제안은 correction_id·recovery_mode·cited_facts 세 키여야 한다"],
                "agrees_with_rule": False}
    entry = reg.get(evidence.get("mechanism"))
    cid, mode, cited = proposal["correction_id"], proposal["recovery_mode"], proposal["cited_facts"]
    if not isinstance(cid, str) or not isinstance(mode, str):
        return {"accepted": False, "reasons": ["correction_id와 recovery_mode는 문자열이어야 한다"],
                "agrees_with_rule": False}
    if entry is None:
        reasons.append("확정 메커니즘이 등록 표에 없다")
    else:
        if cid not in {e["correction_id"] for e in reg.values()}:
            reasons.append(f"등록 표에 없는 교정 {cid!r}")
        elif cid != entry["correction_id"]:
            reasons.append(f"확정 메커니즘의 교정이 아니다({cid!r})")
        if mode not in entry["recovery_modes"]:
            reasons.append(f"등록되지 않은 회복 방식 {mode!r}")
    facts = evidence.get("facts") or {}
    if not isinstance(cited, list) or not cited or not all(isinstance(c, str) for c in cited):
        reasons.append("근거 인용(cited_facts)이 비었거나 형식이 틀리다")
    else:
        missing = [c for c in cited if c not in facts]
        if missing:
            reasons.append(f"제안 근거에 없는 인용 {missing}")
    rule = evidence.get("rule_result") or {}
    agrees = not reasons and cid == rule.get("correction_id") and mode == rule.get("mode")
    return {"accepted": not reasons, "reasons": reasons, "agrees_with_rule": agrees}
