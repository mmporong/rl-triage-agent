"""NAT HITL 미들웨어를 검사·교정 요청 도구에 붙이는 어댑터. 정본 승인은 애플리케이션 원장이다.

- 요청 도구(rl_hitl_pending_probe·rl_hitl_pending_correction)는 기존 고리가 원장에 올린 대기 요청을 읽기만 한다.
  새 요청을 만들거나 고리 상태를 바꾸지 않는다(제안은 evals/loop.py·evals/correct.py가 한다).
- HITL 응답은 명시적 이진 선택 approve일 때만 원장 approve 한 번으로 이어진다. reject·텍스트·알림·다른 선택·시간 초과·
  콜백 없음은 원장을 바꾸지 않는다. 거절 기록과 다음 검사 제안은 기존 CLI(loop.py reject)가 맡는다.
- 승인 직전에 원장의 그 요청이 아직 대기이고 도구가 보여 준 effect key와 같은지 확인한다. 원장 오류는 그대로 올린다.
- 실행은 여전히 원장 consume을 거친다. 여기서 approve를 써도 실행 권한은 원장 상태에서만 나온다.
- 기록되는 approver는 설정한 이름표와 NAT 상호작용 id이며 신원 인증이 아니다. "명시적 이진 선택"은 NAT 데이터 모델
  수준의 보장이다. NAT FastAPI front-end는 클라이언트가 보낸 텍스트로 이진 응답을 만들므로, 누가 응답할 수 있는지는
  front-end 인증이 정한다.
- 단일 승인 작성자를 전제한다. probe_loop.Ledger는 상태 검사와 쓰기 사이의 경쟁을 막지 않으므로(Ledger._append), HITL
  세션 동안 같은 케이스에 loop.py·correct.py의 approve·reject를 함께 쓰지 않고 NAT 서버는 worker 하나로 돌린다.
"""
from __future__ import annotations

import json
from pathlib import Path
import re

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.context import Context
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function, register_middleware
from nat.data_models.function import FunctionBaseConfig
from nat.data_models.interactive import HumanPromptModelType, InteractionResponse
from nat.middleware.hitl.hitl_middleware import HITLMiddleware
from nat.middleware.hitl.hitl_middleware_config import HITLMiddlewareConfig
from nat.middleware.middleware import InvocationAction, InvocationContext

from rl_triage import probe_loop as L
from rl_triage.correction_ledger import CorrectionLedger

APPROVE = "approve"
NO_DECISION = "no_decision"
KINDS = ("probe", "correction")
LEDGER_FILES = {"probe": "ledger.jsonl", "correction": "correction_ledger.jsonl"}
CASE_RE = r"[A-Za-z0-9_]{1,64}"


def open_ledger(kind: str, loops_dir: str | Path, case_id: str) -> L.Ledger:
    if not isinstance(case_id, str) or not re.fullmatch(CASE_RE, case_id):
        raise ValueError(f"케이스 이름이 이상하다: {case_id!r}")
    path = Path(loops_dir) / case_id / LEDGER_FILES[kind]
    return L.Ledger(path) if kind == "probe" else CorrectionLedger(path)


def decision_from_response(response) -> str:
    """명시적 이진 선택의 값이 정확히 "approve"일 때만 APPROVE. 그 밖은 모두 NO_DECISION이다."""
    content = getattr(response, "content", None)
    if getattr(content, "type", None) != HumanPromptModelType.BINARY_CHOICE:
        return NO_DECISION
    option = getattr(content, "selected_option", None)
    return APPROVE if getattr(option, "value", None) == APPROVE else NO_DECISION


def pending_request(kind: str, loops_dir: str | Path, case_id: str) -> dict:
    """원장에서 대기 중인 요청 하나를 읽는다. 없거나 둘 이상이면 request_id None."""
    requests = open_ledger(kind, loops_dir, case_id).requests()
    pending = sorted((r for r in requests.values() if r["state"] == "pending"), key=lambda r: r["at"])
    out = {"kind": kind, "case_id": case_id, "request_id": None, "pending_count": len(pending)}
    if len(pending) != 1:
        return out
    request = pending[0]
    fields = (("probe", "budget_gpu_s") if kind == "probe"
              else ("confirmed_mechanism", "correction_id", "recovery_mode", "budget_gpu_s"))
    return {**out, "request_id": request["request_id"], "effect_key": request["effect_key"],
            **{key: request.get(key) for key in fields}}


def approve_shown_request(shown: dict, decision: str, loops_dir: str | Path, approver: str) -> dict:
    """보여 준 대기 요청에 대한 HITL 결정을 원장에 반영한다. 원장을 바꾸는 경로는 APPROVE 하나뿐이다."""
    if shown.get("kind") not in KINDS:
        raise ValueError("요청 종류를 알 수 없다")
    ledger = open_ledger(shown["kind"], loops_dir, shown["case_id"])
    rid = shown.get("request_id")
    event = None
    current = None if rid is None else ledger.requests().get(rid)
    if rid is not None and current is None:
        raise ValueError(f"{rid}: 원장에 없는 요청이다")
    if decision == APPROVE and rid is not None:
        if current.get("effect_key") != shown.get("effect_key"):
            raise ValueError(f"{rid}: 보여 준 요청과 원장의 요청이 다르다")
        event = ledger.approve(rid, approver)["event"]  # 대기가 아니면 원장이 LedgerError를 낸다
    state = None if rid is None else ledger.requests()[rid]["state"]
    return {"hitl_decision": decision, "ledger_event": event, "ledger_state": state}


def shown_request(output) -> dict:
    """요청 도구 출력(JSON 객체 문자열)을 읽는다. 형식이 다르면 원장을 건드리기 전에 ValueError."""
    try:
        shown = json.loads(output)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("요청 도구 출력이 JSON이 아니다") from error
    if not isinstance(shown, dict):
        raise ValueError("요청 도구 출력이 JSON 객체가 아니다")
    return shown


# --- NAT 함수: 대기 요청 읽기 ---
class PendingProbeConfig(FunctionBaseConfig, name="rl_hitl_pending_probe"):
    """고리 원장의 대기 검사 요청을 보여 준다. 승인은 HITL 응답과 원장이 정한다."""
    loops_dir: str = Field(default="evals/loops", description="evals/loops 형식의 고리 폴더")


class PendingCorrectionConfig(FunctionBaseConfig, name="rl_hitl_pending_correction"):
    """교정 원장의 대기 교정 요청을 보여 준다. 승인은 HITL 응답과 원장이 정한다."""
    loops_dir: str = Field(default="evals/loops", description="evals/loops 형식의 고리 폴더")


@register_function(config_type=PendingProbeConfig)
async def rl_hitl_pending_probe(config: PendingProbeConfig, builder: Builder):  # noqa: ARG001
    async def _run(case_id: str) -> str:
        return json.dumps(pending_request("probe", config.loops_dir, case_id), ensure_ascii=False)
    yield FunctionInfo.from_fn(_run, description="Show the pending probe request for a case. Does not approve it.")


@register_function(config_type=PendingCorrectionConfig)
async def rl_hitl_pending_correction(config: PendingCorrectionConfig, builder: Builder):  # noqa: ARG001
    async def _run(case_id: str) -> str:
        return json.dumps(pending_request("correction", config.loops_dir, case_id), ensure_ascii=False)
    yield FunctionInfo.from_fn(_run, description="Show the pending correction request for a case. Does not approve it.")


# --- NAT 미들웨어 ---
class LedgerHITLMiddlewareConfig(HITLMiddlewareConfig, name="rl_triage_ledger_hitl"):
    """대기 요청 도구 뒤에 운영자 확인을 묻고, 명시적 approve만 원장 approve로 넘긴다."""
    loops_dir: str = Field(default="evals/loops", description="요청 도구와 같은 고리 폴더")
    approver_label: str = Field(default="operator", description="원장 approver에 붙는 이름표(신원 인증 아님)")


class LedgerHITLMiddleware(HITLMiddleware):
    def __init__(self, config: LedgerHITLMiddlewareConfig, builder: Builder) -> None:
        super().__init__(config=config, builder=builder)
        self._ledger_config = config

    async def _on_pre_invoke_response(self, response: InteractionResponse,
                                      context: InvocationContext) -> InvocationContext | None:
        """사전 확인을 켠 경우 approve가 아니면 도구를 부르지 않는다. 원장은 건드리지 않는다."""
        if decision_from_response(response) == APPROVE:
            return None
        context.action = InvocationAction.SKIP
        return context

    async def post_invoke(self, context: InvocationContext) -> InvocationContext | None:
        """보여 줄 요청 내용을 프롬프트 문구에 붙여 묻는다(설정 문구는 바꾸지 않는다)."""
        prompt = self._hitl_config.post_invoke_prompt
        if not prompt:
            return None
        shown = shown_request(context.output)
        if shown.get("request_id") is None:
            return None
        detailed = prompt.model_copy(update={"text": prompt.text.rstrip() + "\n" + json.dumps(shown, ensure_ascii=False)})
        response = await Context.get().user_interaction_manager.prompt_user_input(detailed)
        return await self._on_post_invoke_response(response, context)

    async def _on_post_invoke_response(self, response: InteractionResponse,
                                       context: InvocationContext) -> InvocationContext | None:
        shown = shown_request(context.output)
        approver = f"nat_hitl:{self._ledger_config.approver_label}:{getattr(response, 'id', None)}"
        result = approve_shown_request(shown, decision_from_response(response), self._ledger_config.loops_dir,
                                       approver)
        context.output = json.dumps({**shown, **result}, ensure_ascii=False)
        return context


@register_middleware(config_type=LedgerHITLMiddlewareConfig)
async def rl_triage_ledger_hitl(config: LedgerHITLMiddlewareConfig, builder: Builder):
    yield LedgerHITLMiddleware(config, builder)
