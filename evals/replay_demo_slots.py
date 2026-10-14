"""발표 재생 화면(E5)의 종단 시연 자리(보상·모터·NAT 결정 경로·설계자 밖 결함 U1 시연)를 시연 공개 결과로 채운다.

slots.json의 경로가 비어 있거나 파일이 승인되지 않았으면(없음·미커밋·변경·형식 다름) 기존 자리 표시를 그대로 둔다.
값은 승인된 demo_result.json(evals/e2e_reward_demo.py·e2e_reward_demo_v2.py·e2e_demo_v3_1.py·e2e_demo_u1.py summarize 산출물)에서만
옮기고, 회복 판정은 시연의 고정 평가 판정을 그대로 보여 줄 뿐 다시 계산하지 않는다. 승인된 결과에 값이 없으면
대기가 아니라 기록 없음("-")이나 판정 없음으로 표시한다. NAT 도구 목록은 시연 프로토콜이 SHA로 묶은 설정에서만 읽는다.
GPU·모델·네트워크를 쓰지 않는다.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

import yaml

# 결과 schema → 그 시연 판의 프로토콜(등록 교정·회복 방식 표시에 쓴다). v3.1은 결함 변형별 프로토콜이다.
DEMO_PROTOCOLS = {
    "e2e_reward_demo_result_v1": "bench/protocols/e2e_reward_demo_v1.json",
    "e2e_reward_demo_result_v2": "bench/protocols/e2e_reward_demo_v2.json",
    "e2e_demo_result_v3_1": "bench/protocols/e2e_demo_v3_1_{variant}.json",
    "e2e_demo_result_u1_v1": "bench/protocols/e2e_demo_u1_v1.json",
}
UPSTREAM_SCHEMA = "e2e_demo_result_u1_v1"
FEEDBACK_SCHEMA = "e2e_demo_result_v4"  # 모터 시연 장면 notes에 한 줄로만 쓴다(장면·자리 없음)
VARIANTS = ("f1", "f2")
# 회복 방식을 시연 프로토콜의 등록 규칙이 고르는 판(요청한 방식을 따로 보여 준다)
RULE_CHOSEN_MODE = ("e2e_demo_result_v3_1", "e2e_demo_result_u1_v1")
END_TO_END_LABEL = "종단 시연: 새 Nemotron 호출 → 승인 → 1회 소비 → Isaac 실행 → receipt"
PROPOSAL_LABEL = "교정 제안 기록: 모델 제안 · 결정적 규칙 · 체크리스트"
MOTOR_END_TO_END_LABEL = "모터 결함 종단 시연: 새 Nemotron 호출 → 승인된 검사 → 교정 승인 → Isaac 재학습 → receipt"
MOTOR_PROPOSAL_LABEL = "모터 결함 교정 제안 기록: 모델 제안 · 결정적 규칙 · 체크리스트"
NAT_END_TO_END_LABEL = "NAT 결정 경로 종단 시연: NeMo Agent Toolkit 도구 호출 → 승인된 검사 → 교정 승인 → Isaac 재학습 → receipt"
NAT_RECORD_LABEL = "NeMo Agent Toolkit 호출 기록: LLM 호출 · 제출 전 점검 · 토큰 · 설정 도구"
UPSTREAM_END_TO_END_LABEL = ("설계자 밖 결함 U1 종단 시연: 새 Nemotron 호출 → 승인된 검사 → 교정 범위 점검 → 교정 승인 → "
                             "Isaac 재학습 → receipt")
SCOPE_LABEL = "교정 범위 점검과 재학습 대조: 확정 검사 항별 오차 · 등록 표적 · 고정 평가 판정"
SCOPE_MODES = {"contrast": "대조 모드: 판정 기록 뒤 재학습 1회", "enforce": "강제 모드: 범위 부족이면 재학습 전에 끝냄"}
# 결과의 decision_path → 화면 표기. v1·v2 결과에는 필드가 없고 단일 호출이다.
DECISION_PATHS = {"single": "새 Nemotron 호출", "nat_v4": "NeMo Agent Toolkit 도구 호출"}
# NAT 결정 경로 → 진단·교정 워크플로 설정(시연 프로토콜 source_files가 LF SHA로 묶는다)
NAT_CONFIGS = {"nat_v4": {"diagnosis": "configs/nat_demo_v4_diagnosis.yml",
                          "correction": "configs/nat_demo_v4_correction.yml"}}
NO_RECORD = "-"
NO_VERDICT = "판정 없음"


def protocol_path(result: dict) -> str | None:
    """결과가 가리키는 시연 프로토콜 경로. 모르는 변형이면 None."""
    template = DEMO_PROTOCOLS[result["schema"]]
    if "{variant}" not in template:
        return template
    return template.format(variant=result["variant"]) if result.get("variant") in VARIANTS else None


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
    if protocol_path(result) is None:
        row["status"], row["reason"] = "mismatch", "unexpected_variant"
        return row, None
    if (result.get("decision_path") or "single") not in DECISION_PATHS:
        row["status"], row["reason"] = "mismatch", "unexpected_decision_path"
        return row, None
    return row, result


def _path_label(result: dict) -> str:
    return DECISION_PATHS[result.get("decision_path") or "single"]


def _fault_label(result: dict) -> str:
    """제목 뒤에 붙는 결함 표시. v1·v2 결과에는 결함 필드가 없고 보상 결함(F1)이라 붙이지 않는다."""
    return f" {result['fault']}·{result.get('mechanism')}" if result.get("fault") else ""


def _probe_order(result: dict | None) -> list[str]:
    return [r.get("probe") for r in ((result or {}).get("loop") or {}).get("requests", [])]


def feedback_demo_note(sources: Any, relative: str | None, compare: str | None, *,
                       kind: str = "motor_feedback_end_to_end") -> str | None:
    """검증 피드백 1회 재요청 시연(v4) 결과 한 줄. 승인되지 않았거나 형식이 다르면 None(줄을 넣지 않는다).
    검사 경로가 compare 결과(기존 모터 시연)와 같을 때만 "같은 검사 경로"라고 쓴다."""
    row = sources.file(relative, kind=kind)
    if row["status"] != "ready":
        return None
    result = json.loads((sources.root / relative).read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        result = {}
    feedback = result.get("model_feedback")
    if (result.get("schema") != FEEDBACK_SCHEMA or result.get("variant") not in VARIANTS or not isinstance(feedback, dict)
            or any(not isinstance((feedback.get(step) or {}).get("retries"), int) for step in ("diagnosis", "correction"))):
        row["status"], row["reason"] = "mismatch", "unexpected_schema"
        return None
    order = _probe_order(result)
    _, other = _load(sources, compare, "motor_end_to_end")
    path = ("같은 검사 경로" if other is not None and _probe_order(other) == order else "검사 경로") + f"({' → '.join(order)})"
    retries = {step: feedback[step]["retries"] for step in ("diagnosis", "correction")}
    return (f"v4(seed {result.get('seed')}, 검증 피드백 1회 재요청)도 {path}를 거쳐 결과가 {result.get('outcome')}였다. "
            f"재요청은 {sum(retries.values())}회(첫 판단 {retries['diagnosis']}회, 교정 제안 {retries['correction']}회)였다.")


def end_to_end_blocks(sources: Any, relative: str | None, *, kind: str = "end_to_end",
                      label: str = END_TO_END_LABEL) -> list[dict]:
    row, result = _load(sources, relative, kind)
    if result is None:
        return [{"type": "slot", "label": label, "source": row["id"]}]
    sid = row["id"]
    decision = (result.get("diagnosis") or {}).get("decision") or {}
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
        [f"첫 판단({_path_label(result)})", _cell(sources, (
            f"{decision.get('action')} · 상위 3 {', '.join(decision.get('ranking', [])[:3])} · 제안 검사 {decision.get('probe')}"
            if decision else None), sid)],
        ["승인된 검사", _cell(sources, f"{len(done)}건 실행 · {observed} · {loop.get('status')}" if loop else None, sid)],
    ]
    specificity = result.get("specificity")
    if specificity is not None:  # v3.1: 확정 뒤 다른 원인 검사도 정상인지 보는 관문(변형에 따라 등록 검사가 없다)
        outcomes = ", ".join(f"{probe} {outcome}" for probe, outcome in sorted((specificity.get("outcomes") or {}).items()))
        shown = (f"{'통과' if specificity.get('passed') else '미통과'} · {outcomes}" if specificity.get("required")
                 else "등록된 보조 검사 없음")
        steps.append(["특이성 관문", _cell(sources, shown, sid)])
    steps += [
        [f"교정 제안({_path_label(result)})", _cell(sources, (
            f"{proposal.get('decision')} · 계약 통과 {_yes(proposal.get('accepted'))} · 규칙과 일치 {_yes(proposal.get('agrees_with_rule'))}"
            if proposal else None), sid)],
        ["승인 뒤 교정 실행 receipt", _cell(sources, f"{receipt.get('event')} · {receipt.get('verdict')}" if receipt else None, sid)],
        ["교정 정책 고정 평가 판정", _cell(sources, label, sid)],
        ["교정 정책 standstill ratio", _cell(sources, fixed.get("standstill_ratio"), sid, 3)],
        ["결과 문구", _cell(sources, result.get("wording"), sid)],
    ]
    return [{"type": "table", "title": f"종단 시연 seed {result.get('seed')}{_fault_label(result)}: 경고부터 교정 판정까지",
             "columns": ["단계", "기록"], "rows": steps, "status_sources": [sid]}]


COST_COLUMNS = ["시연", "NONE·결함 학습 wall(s)", "승인 검사 GPU wall(s)", "교정 재학습·평가 wall(s)",
                "진단·교정 호출 토큰(입력 / 출력)"]


def _pair(first: Any, second: Any, joiner: str = " / ") -> str | None:
    """두 값을 한 칸에 적는다. 숫자는 초 단위 정수로, 빈 값은 기록 없음으로 둔다."""
    if first is None and second is None:
        return None
    shown = [NO_RECORD if value is None else f"{value:.0f}" if isinstance(value, (int, float)) else value
             for value in (first, second)]
    return joiner.join(shown)


def cost_row(sources: Any, relative: str | None, *, kind: str, name: str) -> list:
    """비용 표의 한 줄(시연 하나). 결과가 승인되지 않았으면 그 근거 상태로 표시한다."""
    row, result = _load(sources, relative, kind)
    sid = row["id"]
    if result is None:
        return [name, *({"value": None, "source": sid, "digits": None} for _ in COST_COLUMNS[1:])]
    walls = result.get("training_parent_wall_s") or {}
    fault = result.get("fault") or "F1"
    done = [r for r in (result.get("loop") or {}).get("requests", []) if r.get("state") == "done"]
    correction = result.get("correction") or {}
    costs = (correction.get("result") or {}).get("costs") or {}
    diagnosis_call = (result.get("diagnosis") or {}).get("call") or {}
    return [f"{name} seed {result.get('seed')}",
            _cell(sources, _pair(walls.get("NONE"), walls.get("FAULT", walls.get(fault))), sid),
            _cell(sources, sum(r.get("gpu_s") or 0 for r in done) if done else None, sid, 1),
            _cell(sources, _pair(costs.get("train_parent_wall_s"), costs.get("eval_parent_wall_s")), sid),
            _cell(sources, _pair(_tokens(diagnosis_call), _tokens(correction.get("call") or {}), " · "), sid)]


def cost_block(sources: Any, entries: list[tuple[str | None, str, str]]) -> dict:
    """종단 시연 비용 표. entries는 (결과 경로, 근거 종류, 시연 이름)이다."""
    rows = [cost_row(sources, relative, kind=kind, name=name) for relative, kind, name in entries]
    return {"type": "table", "wide": True, "title": "종단 시연 비용(부모 프로세스 벽시계·토큰이며 절감 비교가 아니다)",
            "columns": COST_COLUMNS, "rows": rows, "status_sources": [r[1]["source"] for r in rows]}


def _registered(protocol: dict | None) -> str | None:
    """시연 프로토콜의 등록 교정과 회복 방식. v3.1은 방식 여러 개를 등록하고 규칙이 하나를 고른다."""
    if protocol is None:
        return None
    correction = protocol["correction"]
    modes = correction.get("registered_recovery_mode") or "/".join(correction.get("modes") or ())
    return f"{correction['correction_id']} · {modes}"


def _protocol(sources: Any, result: dict) -> tuple[dict, dict | None]:
    row = sources.file(protocol_path(result))
    return row, (json.loads((sources.root / row["path"]).read_text(encoding="utf-8")) if row["status"] == "ready" else None)


def correction_proposal_blocks(sources: Any, relative: str | None, *, kind: str = "correction_proposal",
                               label: str = PROPOSAL_LABEL, wide: bool = True) -> list[dict]:
    row, result = _load(sources, relative, kind)
    if result is None:
        return [{"type": "slot", "wide": wide, "label": label, "source": row["id"]}]
    sid = row["id"]
    protocol_row, protocol = _protocol(sources, result)
    correction = result.get("correction") or {}
    proposal, receipt = correction.get("model_proposal") or {}, correction.get("receipt") or {}
    unmet = proposal.get("unmet_checklist")
    rows = [["등록 교정 · 회복 방식(시연 프로토콜)", _cell(sources, _registered(protocol), protocol_row["id"])]]
    if result["schema"] in RULE_CHOSEN_MODE:
        rows.append(["등록 규칙이 고른 회복 방식", _cell(sources, (correction.get("request") or {}).get("recovery_mode"), sid)])
    rows += [
        ["모델 제안 결정", _cell(sources, proposal.get("decision"), sid)],
        ["공개 계약 검증 통과", _cell(sources, _yes(proposal.get("accepted")), sid)],
        ["결정적 규칙과 일치", _cell(sources, _yes(proposal.get("agrees_with_rule")), sid)],
        ["규칙을 모델 입력에 보여 줌", _cell(sources, _yes(proposal.get("rule_visible_to_model")), sid)],
        ["미충족 체크리스트 항목 수", _cell(sources, None if unmet is None else len(unmet), sid, 0)],
        ["승인 뒤 1회 소비", _cell(sources, _yes(receipt.get("event") == "receipt") if receipt else None, sid)],
    ]
    return [{"type": "table", "wide": wide, "title": f"교정 제안 기록(종단 시연 seed {result.get('seed')}{_fault_label(result)})",
             "columns": ["항목", "기록"], "rows": rows, "status_sources": [sid, protocol_row["id"]]}]


def _load_nat(sources: Any, relative: str | None, kind: str) -> tuple[dict, dict | None]:
    """NAT 결정 경로 시연 결과. 승인된 결과가 NAT 경로가 아니면 불일치로 둔다."""
    row, result = _load(sources, relative, kind)
    if result is not None and result.get("decision_path") not in NAT_CONFIGS:
        row["status"], row["reason"] = "mismatch", "not_nat_decision_path"
        return row, None
    return row, result


def _lf_sha(path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _bound_config(sources: Any, path: str, protocol: dict | None) -> dict:
    """시연 프로토콜 source_files가 묶은 LF SHA와 같은 설정 파일만 확인으로 둔다."""
    row = sources.file(path, kind="nat_config")
    if row["status"] == "ready" and protocol is None:
        row["status"], row["reason"] = "waiting", "demo_protocol_not_ready"
    elif row["status"] == "ready" and protocol.get("source_files", {}).get(path) != _lf_sha(sources.root / path):
        row["status"], row["reason"] = "mismatch", "demo_protocol_sha_mismatch"
    return row


def _call_text(call: dict) -> str | None:
    if not call or call.get("llm_calls") is None:
        return None
    parts = [f"LLM {call['llm_calls']}회"]
    if call.get("precheck_calls") is not None:
        parts.append(f"제출 전 점검 {call['precheck_calls']}회")
    if call.get("execution_wall_s") is not None:
        parts.append(f"실행 {call['execution_wall_s']:.1f} s")
    return " · ".join(parts)


def nat_record_blocks(sources: Any, relative: str | None, *, kind: str = "nat_end_to_end") -> list[dict]:
    """NAT 결정 경로의 호출 기록. 도구 호출 순서는 공개 결과에 없어 설정된 도구 목록만 보인다."""
    row, result = _load_nat(sources, relative, kind)
    if result is None:
        return [{"type": "slot", "label": NAT_RECORD_LABEL, "source": row["id"]}]
    sid = row["id"]
    protocol_row, protocol = _protocol(sources, result)
    adoption = (protocol or {}).get("nat_adoption") or {}
    diagnosis_call = (result.get("diagnosis") or {}).get("call") or {}
    correction_call = (result.get("correction") or {}).get("call") or {}
    rows = [
        ["결정 경로", _cell(sources, f"{_path_label(result)}({result['decision_path']})", sid)],
        ["경로 채택 근거(시연 프로토콜)", _cell(sources, adoption.get("decision") and f"NAT 경로 비교 요약 판정 {adoption['decision']}",
                                     protocol_row["id"])],
        ["진단 호출", _cell(sources, _call_text(diagnosis_call), sid)],
        ["진단 토큰(입력 / 출력)", _cell(sources, _tokens(diagnosis_call), sid)],
        ["교정 호출", _cell(sources, _call_text(correction_call), sid)],
        ["교정 토큰(입력 / 출력)", _cell(sources, _tokens(correction_call), sid)],
    ]
    ids = [sid, protocol_row["id"]]
    for stage, title in (("diagnosis", "진단"), ("correction", "교정")):
        path = NAT_CONFIGS[result["decision_path"]][stage]
        config_row = _bound_config(sources, path, protocol)
        tools = None
        if config_row["status"] == "ready":
            tools = ", ".join(yaml.safe_load((sources.root / path).read_text(encoding="utf-8"))["workflow"]["tool_names"])
        rows.append([f"{title} 워크플로 도구(설정 목록, 호출 순서 아님)", _cell(sources, tools, config_row["id"])])
        ids.append(config_row["id"])
    return [{"type": "table", "title": f"NeMo Agent Toolkit 호출 기록(seed {result.get('seed')})",
             "columns": ["항목", "기록"], "rows": rows, "status_sources": ids}]


def nat_role_cell(sources: Any, relative: str | None, *, kind: str = "nat_end_to_end") -> dict:
    """NVIDIA 역할 표의 NAT 행 근거: NAT 결정 경로 시연 한 건의 LLM 호출 수."""
    row, result = _load_nat(sources, relative, kind)
    if result is None:
        return {"value": None, "source": row["id"], "digits": None}
    counts = [((result.get(stage) or {}).get("call") or {}).get("llm_calls") for stage in ("diagnosis", "correction")]
    shown = [NO_RECORD if count is None else f"{count}회" for count in counts]
    return _cell(sources, f"seed {result.get('seed')} · 진단 LLM {shown[0]} · 교정 LLM {shown[1]}", row["id"])


def _readback_text(readback: dict | None) -> str | None:
    if not readback:
        return None
    return f"{readback.get('registered_target_state')} · {readback.get('note')}"


def _errors_text(scope: dict) -> str | None:
    errors = scope.get("rel_errors")
    if not errors:
        return None
    return " · ".join(f"{name} {NO_RECORD if value is None else f'{value:.3g}'}" for name, value in sorted(errors.items()))


def _hit(value: Any) -> str:
    return NO_RECORD if value is None else _yes(value)


def _terms(names: list | None) -> str:
    return ", ".join(names or []) or "없음"


def _approval_text(result: dict, protocol: dict | None) -> str | None:
    """교정 승인 성격. 요청 근거에 묶인 범위 판정·미충족 체크리스트와 프로토콜의 대조·강제 모드 규칙만 쓴다.

    대조 모드에서 범위가 충분하지 않다는 판정이 요청 근거에 묶인 채 승인·소비(receipt)됐으면 대조 측정용 승인이다.
    묶이지 않았으면 승인자가 그 판정을 보았다고 말할 수 없어 일반 승인으로 둔다. 근거는 프로토콜 scope_gate의
    contrast_run(운영자는 부족 판정을 보고 검증용으로 승인)과 enforce_outcome(강제 모드면 재학습 전 종료)이다.
    """
    correction = result.get("correction") or {}
    step = result.get("scope_check") or {}
    verdict = (step.get("result") or {}).get("verdict")
    if not correction or (correction.get("receipt") or {}).get("event") != "receipt" or protocol is None:
        return None
    unmet = "·".join((correction.get("model_proposal") or {}).get("unmet_checklist") or []) or "없음"
    evidence = f"요청 근거 {verdict} · 체크리스트 미충족 {unmet}" if correction.get("scope_check_bound") else "범위 점검 미결합"
    gate = protocol.get("scope_gate") or {}
    if (correction.get("scope_check_bound") and step.get("mode") == "contrast" and gate.get("contrast_run")
            and verdict not in ("scope_sufficient", "not_applicable")):
        return f"대조 측정용 승인(운영 기준 예외) · {evidence} · 강제 모드였다면 재학습 전 종료"
    return f"승인 · {evidence}"


def scope_blocks(sources: Any, relative: str | None, *, kind: str = "upstream_end_to_end") -> list[dict]:
    """U1 시연의 교정 범위 점검과 재학습 판정 대조. U1 결과가 아니면 불일치로 두고 자리 표시를 남긴다.

    단계 표(end_to_end_blocks)와 나란히 놓이므로 고정 평가 판정·결과 문구(대조 문구 포함)는 단계 표에만 둔다.
    """
    row, result = _load(sources, relative, kind)
    if result is not None and result["schema"] != UPSTREAM_SCHEMA:
        row["status"], row["reason"] = "mismatch", "not_upstream_result"
        result = None
    if result is None:
        return [{"type": "slot", "label": SCOPE_LABEL, "source": row["id"]}]
    sid = row["id"]
    protocol_row, protocol = _protocol(sources, result)
    step = result.get("scope_check") or {}
    scope = step.get("result") or {}
    fixed = (result.get("correction") or {}).get("result") or {}
    contrast = result.get("scope_contrast") or {}
    predicted = result.get("prediction_results") or {}
    aux = (fixed.get("correction_readback") or {}).get("aux_fingerprints") or {}
    method = f"{SCOPE_MODES.get(step.get('mode'), step.get('mode'))} · 결정적 검사" if step else None
    threshold = f"(문턱 {scope['threshold']})" if scope.get("threshold") is not None else ""
    rows = [
        ["점검 방식", _cell(sources, method, sid)],
        [f"확정 검사 항별 상대 오차{threshold}", _cell(sources, _errors_text(scope), sid)],
        ["등록 교정과 되돌리는 함수", _cell(sources, f"{step.get('correction_id')} → {_terms(scope.get('registered_targets'))}"
                                          if step else None, sid)],
        ["점검 판정 · 등록 표적 밖 이상 항", _cell(sources, f"{scope.get('verdict')} · {_terms(scope.get('uncovered_terms'))}"
                                               if scope else None, sid)],
        ["대조 범주(점검 × 고정 평가 판정)", _cell(sources, contrast.get("category"), sid)],
        ["교정 승인 성격", _cell(sources, _approval_text(result, protocol), protocol_row["id"])],
        ["사전 예측 적중(점검 · 교정 · 대조)", _cell(sources, " · ".join(
            _hit(predicted.get(key)) for key in ("scope_check", "correction", "scope_contrast")) if predicted else None, sid)],
        ["교정 readback(등록 표적만 대조)", _cell(sources, _readback_text(fixed.get("correction_readback")), sid)],
        ["보조 지문: 원본과 다른 함수(판정에 쓰지 않음)", _cell(sources, _terms(aux.get("differs_from_clean")) if aux else None, sid)],
    ]
    return [{"type": "table", "title": f"교정 범위 점검과 재학습 대조(seed {result.get('seed')})",
             "columns": ["항목", "기록"], "rows": rows, "status_sources": [sid, protocol_row["id"]]}]
