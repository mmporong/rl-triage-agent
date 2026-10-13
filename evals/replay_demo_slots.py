"""발표 재생 화면(E5)의 종단 시연 자리 두 개(end_to_end·correction_proposal)를 시연 공개 결과로 채운다.

slots.json의 경로가 비어 있거나 파일이 승인되지 않았으면(없음·미커밋·변경·형식 다름) 기존 자리 표시를 그대로 둔다.
값은 승인된 demo_result.json(evals/e2e_reward_demo.py·e2e_reward_demo_v2.py summarize 산출물)에서만 옮기고,
회복 판정은 시연의 고정 평가 판정을 그대로 보여 줄 뿐 다시 계산하지 않는다. 승인된 결과에 값이 없으면
대기가 아니라 기록 없음("-")이나 판정 없음으로 표시한다. GPU·모델·네트워크를 쓰지 않는다.
"""
from __future__ import annotations

import json
from typing import Any

# 결과 schema → 그 시연 판의 프로토콜(등록 교정·회복 방식 표시에 쓴다)
DEMO_PROTOCOLS = {
    "e2e_reward_demo_result_v1": "bench/protocols/e2e_reward_demo_v1.json",
    "e2e_reward_demo_result_v2": "bench/protocols/e2e_reward_demo_v2.json",
}
END_TO_END_LABEL = "종단 시연: 새 Nemotron 호출 → 승인 → 1회 소비 → Isaac 실행 → receipt"
PROPOSAL_LABEL = "교정 제안 기록: 모델 제안 · 결정적 규칙 · 체크리스트"
NO_RECORD = "-"
NO_VERDICT = "판정 없음"


def _cell(sources: Any, value: Any, source: str, digits: int | None = None) -> dict:
    """승인된 결과의 값. 결과에 값이 없으면 기록 없음으로 둔다(대기와 구분)."""
    shown = NO_RECORD if value is None else value
    return {"value": shown if sources.ready(source) else None, "source": source, "digits": digits}


def _yes(value: Any) -> str | None:
    return None if value is None else ("예" if value is True else "아니오")


def _tokens(call: dict) -> str | None:
    if not call or call.get("input_tokens") is None:
        return None
    return f"{call['input_tokens']} / {call.get('output_tokens')}"


def _load(sources: Any, relative: str | None, kind: str) -> tuple[dict, dict | None]:
    row = sources.file(relative, kind=kind)
    if row["status"] != "ready":
        return row, None
    result = json.loads((sources.root / relative).read_text(encoding="utf-8"))
    if not isinstance(result, dict) or result.get("schema") not in DEMO_PROTOCOLS:
        row["status"], row["reason"] = "mismatch", "unexpected_schema"
        return row, None
    return row, result


def end_to_end_blocks(sources: Any, relative: str | None) -> list[dict]:
    row, result = _load(sources, relative, "end_to_end")
    if result is None:
        return [{"type": "slot", "label": END_TO_END_LABEL, "source": row["id"]}]
    sid = row["id"]
    decision = (result.get("diagnosis") or {}).get("decision") or {}
    diagnosis_call = (result.get("diagnosis") or {}).get("call") or {}
    loop = result.get("loop") or {}
    correction = result.get("correction") or {}
    proposal, receipt = correction.get("model_proposal") or {}, correction.get("receipt") or {}
    fixed = correction.get("result") or {}
    checks = result.get("success_conditions") or {}
    done = [r for r in loop.get("requests", []) if r.get("state") == "done"]
    observed = ", ".join(f"{probe} {outcome}" for probe, outcome in sorted((loop.get("observed") or {}).items()))
    label = fixed.get("label") or (NO_VERDICT if correction else None)
    steps = [
        ["시연 결과", _cell(sources, f"{result.get('outcome')} · 성공 조건 {sum(bool(v) for v in checks.values())}/{len(checks)}", sid)],
        ["첫 판단(새 Nemotron 호출)", _cell(sources, (
            f"{decision.get('action')} · 상위 3 {', '.join(decision.get('ranking', [])[:3])} · 제안 검사 {decision.get('probe')}"
            if decision else None), sid)],
        ["승인된 검사", _cell(sources, f"{len(done)}건 실행 · {observed} · {loop.get('status')}" if loop else None, sid)],
        ["교정 제안(새 Nemotron 호출)", _cell(sources, (
            f"{proposal.get('decision')} · 계약 통과 {_yes(proposal.get('accepted'))} · 규칙과 일치 {_yes(proposal.get('agrees_with_rule'))}"
            if proposal else None), sid)],
        ["승인 뒤 교정 실행 receipt", _cell(sources, f"{receipt.get('event')} · {receipt.get('verdict')}" if receipt else None, sid)],
        ["교정 정책 고정 평가 판정", _cell(sources, label, sid)],
        ["교정 정책 standstill ratio", _cell(sources, fixed.get("standstill_ratio"), sid, 3)],
        ["결과 문구", _cell(sources, result.get("wording"), sid)],
    ]
    walls = result.get("training_parent_wall_s") or {}
    costs = fixed.get("costs") or {}
    cost_rows = [
        ["NONE 학습 wall(s)", _cell(sources, walls.get("NONE"), sid, 0)],
        ["F1 학습 wall(s)", _cell(sources, walls.get("F1"), sid, 0)],
        ["승인 검사 GPU wall(s)", _cell(sources, sum(r.get("gpu_s") or 0 for r in done) if done else None, sid, 1)],
        ["교정 재학습 wall(s)", _cell(sources, costs.get("train_parent_wall_s"), sid, 0)],
        ["교정 고정 평가 wall(s)", _cell(sources, costs.get("eval_parent_wall_s"), sid, 0)],
        ["진단 호출 입력·출력 토큰", _cell(sources, _tokens(diagnosis_call), sid)],
        ["교정 호출 입력·출력 토큰", _cell(sources, _tokens(correction.get("call") or {}), sid)],
    ]
    return [
        {"type": "table", "title": f"종단 시연 seed {result.get('seed')}: 경고부터 교정 판정까지",
         "columns": ["단계", "기록"], "rows": steps, "status_sources": [sid]},
        {"type": "table", "title": "시연 비용 기록(부모 프로세스 벽시계·토큰이며 절감 비교가 아니다)",
         "columns": ["항목", "값"], "rows": cost_rows, "status_sources": [sid]},
    ]


def correction_proposal_blocks(sources: Any, relative: str | None) -> list[dict]:
    row, result = _load(sources, relative, "correction_proposal")
    if result is None:
        return [{"type": "slot", "wide": True, "label": PROPOSAL_LABEL, "source": row["id"]}]
    sid = row["id"]
    protocol_path = DEMO_PROTOCOLS[result["schema"]]
    protocol_row = sources.file(protocol_path)
    protocol = (json.loads((sources.root / protocol_path).read_text(encoding="utf-8"))
                if protocol_row["status"] == "ready" else None)
    registered = (f"{protocol['correction']['correction_id']} · {protocol['correction']['registered_recovery_mode']}"
                  if protocol else None)
    correction = result.get("correction") or {}
    proposal, receipt = correction.get("model_proposal") or {}, correction.get("receipt") or {}
    unmet = proposal.get("unmet_checklist")
    rows = [
        ["등록 교정 · 회복 방식(시연 프로토콜)", _cell(sources, registered, protocol_row["id"])],
        ["모델 제안 결정", _cell(sources, proposal.get("decision"), sid)],
        ["공개 계약 검증 통과", _cell(sources, _yes(proposal.get("accepted")), sid)],
        ["결정적 규칙과 일치", _cell(sources, _yes(proposal.get("agrees_with_rule")), sid)],
        ["규칙을 모델 입력에 보여 줌", _cell(sources, _yes(proposal.get("rule_visible_to_model")), sid)],
        ["미충족 체크리스트 항목 수", _cell(sources, None if unmet is None else len(unmet), sid, 0)],
        ["승인 뒤 1회 소비", _cell(sources, _yes(receipt.get("event") == "receipt") if receipt else None, sid)],
    ]
    return [{"type": "table", "wide": True, "title": f"교정 제안 기록(종단 시연 seed {result.get('seed')})",
             "columns": ["항목", "기록"], "rows": rows, "status_sources": [sid, protocol_row["id"]]}]
