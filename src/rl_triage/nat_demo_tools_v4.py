"""E4b v4: 진단 근거 인용의 실수 표기를 고정하고, 교정 검증을 packet 등록 표로 하는 NAT 도구.

v3 결과[사실]: 새 입력 h02_s2027의 진단 무효 3회는 모두 claim `telemetry:reference:Policy/mean_noise_std`의 값이
0.3697160104910533으로, 사실 값 0.36971601049105324와 다른 double이었다(실행마다 점검 호출 3회(한도 거절 1회 포함)와
제출이 모두 같은 값). 공개 검증기
(active_diagnosis.validate_decision)는 claim 값을 사실 값과 == 로 비교한다.

v4 선택: 표기 고정 + 표기 일치 치환. 사실 값 읽기 도구(e4b4_get_facts)는 실수를 유효숫자 6자리 문자열로 보여 주고,
어댑터(claims_to_contract)는 claim이 그 사실 id에 대해 보여 준 표기와 정확히 같을 때만 저장 값으로 바꾼 뒤 공개
검증기에 넘긴다. 허용 오차 대신 이 방식을 고른 이유: 검증기의 "보여 준 근거를 정확히 인용한다"는 의미가 유지되고
(모델이 본 적 없는 근사값은 통과하지 못한다), 오차 크기를 새로 정하지 않으며, 문자열은 제공자 쪽 숫자 재표기의
영향을 받지 않는다. 정수·문자열·불리언 사실은 그대로 보여 주고 바꾸지 않는다. 검증기는 수정하지 않는다.

교정은 v3의 null 없는 제출 스키마·어댑터를 그대로 쓰되, 검증 등록 표를 packet에 실린 등록 표(모델이 e4b2_registry로
보는 표)로 한다. 기본 등록 표 packet에서는 v3와 같고, 시연 v3처럼 등록 표를 넓힌 packet에서도 같은 표로 판정한다.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
import os

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig
from pydantic import Field

from rl_triage import active_diagnosis as D
from rl_triage import nat_demo_tools as V1
from rl_triage import nat_demo_tools_v2 as V2
from rl_triage import nat_demo_tools_v3 as V3

PRECHECK_LIMIT = V2.PRECHECK_LIMIT
PRECHECK_TOOLS = {"diagnosis": ("e4b4_check_decision",), "correction": ("e4b4_check_proposal",)}
DISPLAY_DIGITS = 6
SubmissionRejected = V1.SubmissionRejected


def display(value):
    """도구가 보여 주는 사실 값. 유한한 실수는 유효숫자 6자리 문자열, 그 밖은 그대로(중첩은 재귀)."""
    if isinstance(value, float) and math.isfinite(value):
        return format(value, f".{DISPLAY_DIGITS}g")
    if isinstance(value, list):
        return [display(item) for item in value]
    if isinstance(value, dict):
        return {key: display(item) for key, item in value.items()}
    return value


def _cites_display(claimed, fact: float) -> bool:
    shown = display(fact)
    if isinstance(claimed, str):
        return claimed == shown
    return (not isinstance(claimed, bool) and isinstance(claimed, (int, float)) and math.isfinite(claimed)
            and float(claimed) == float(shown))


def claims_to_contract(decision: dict, packet: dict) -> dict:
    """보여 준 실수 표기를 정확히 인용한 claim만 저장 값으로 바꾼다. 나머지는 그대로 두고 검증기가 판정한다."""
    out = deepcopy(decision)
    known = D._known_facts(packet)
    for claim in out.get("claims") or []:
        if not isinstance(claim, dict):
            continue
        fact = known.get(claim.get("id"))
        if isinstance(fact, float) and math.isfinite(fact) and _cites_display(claim.get("value"), fact):
            claim["value"] = fact
    return out


def precheck_calls(kind: str, trace_path: str | None = None) -> int:
    path = trace_path or os.environ.get("E4B_TRACE")
    if not path or not os.path.exists(path):
        return 0
    with open(path, encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip() and json.loads(line).get("tool") in PRECHECK_TOOLS[kind])


def _limit_reached(kind: str) -> dict | None:
    if precheck_calls(kind) >= PRECHECK_LIMIT:
        return {"ok": False, "checked": False,
                "reasons": [f"precheck limit of {PRECHECK_LIMIT} calls reached; submit your final object now"]}
    return None


# --- 진단 ---
def get_facts(ids: list[str]) -> dict:
    found = V1.get_facts(ids)
    return {**found, "facts": display(found["facts"]),
            "display": f"decimals are shown with {DISPLAY_DIGITS} significant digits as strings; cite them exactly as shown"}


def check_decision(decision: dict) -> dict:
    _, packet = V1.context("diagnosis")
    limited = _limit_reached("diagnosis")
    if limited:
        return limited
    try:
        D.validate_decision(claims_to_contract(decision, packet), packet)
    except (ValueError, TypeError) as error:
        return {"ok": False, "checked": True, "reasons": [str(error)]}
    return {"ok": True, "checked": True, "reasons": []}


def submit_decision(decision: dict) -> str:
    tool = "e4b4_submit_decision"
    _, packet = V1.context("diagnosis")
    V1._guard_once(tool, decision)
    try:
        canonical = D.validate_decision(claims_to_contract(decision, packet), packet)
    except (ValueError, TypeError) as error:
        V1._reject(tool, decision, [str(error)])
    return V1._save(tool, decision, {"kind": "diagnosis", "decision": canonical, "tool_args": decision})


# --- 교정 ---
def validate_proposal(proposal: dict, packet: dict) -> dict:
    """공개 계약 검증을 packet에 실린 등록 표로 한다."""
    return V1._correction_module().validate_model_proposal(proposal, packet, packet["registry"])


def _validate(args: dict) -> tuple[dict | None, dict]:
    _, packet = V1.context("correction")
    proposal, reasons = V3.to_contract(args)
    if proposal is None:
        return None, {"accepted": False, "decision": None, "reasons": reasons, "agrees_with_rule": False,
                      "unmet_checklist": []}
    return proposal, validate_proposal(proposal, packet)


def check_proposal(args: dict) -> dict:
    limited = _limit_reached("correction")
    if limited:
        return limited
    _, verdict = _validate(args)
    return {"ok": verdict["accepted"], "checked": True, "reasons": verdict["reasons"]}


def submit_proposal(args: dict) -> str:
    tool = "e4b4_submit_proposal"
    V1.context("correction")
    V1._guard_once(tool, args)
    proposal, verdict = _validate(args)
    if not verdict["accepted"]:
        V1._reject(tool, args, verdict["reasons"])
    return V1._save(tool, args, {"kind": "correction", "proposal": proposal, "validation": verdict, "tool_args": args})


# --- NAT 입력 스키마: v2와 같고 claim 설명만 v4 읽기 도구·표기에 맞춘다 ---
class DecisionInput(V2.DecisionInput):
    claims: list[V1.Claim] = Field(
        min_length=1,
        description=(f"Evidence fact ids read with e4b4_get_facts, each with its value copied exactly as e4b4_get_facts "
                     f"showed it (decimals are strings with {DISPLAY_DIGITS} significant digits, for example \"0.369716\")."))


# --- NAT 등록 ---
class ListFactIdsConfig(FunctionBaseConfig, name="e4b4_list_fact_ids"):
    pass


@register_function(config_type=ListFactIdsConfig)
async def list_fact_ids_fn(config: ListFactIdsConfig, builder: Builder):
    async def _run(request: V1.EmptyInput) -> str:
        return V1._traced("e4b4_list_fact_ids", {}, V1.list_fact_ids)
    yield FunctionInfo.from_fn(_run, input_schema=V1.EmptyInput, description=(
        "List the evidence fact ids of the packet. Read their values with e4b4_get_facts."))


class GetFactsConfig(FunctionBaseConfig, name="e4b4_get_facts"):
    pass


@register_function(config_type=GetFactsConfig)
async def get_facts_fn(config: GetFactsConfig, builder: Builder):
    async def _run(request: V1.FactIdsInput) -> str:
        return V1._traced("e4b4_get_facts", {"ids": request.ids}, lambda: get_facts(request.ids))
    yield FunctionInfo.from_fn(_run, input_schema=V1.FactIdsInput, description=(
        f"Return the values of up to {V1.MAX_FACT_IDS} fact ids. Decimals are shown as strings with "
        f"{DISPLAY_DIGITS} significant digits; cite them in claims exactly as shown. Unknown ids are listed."))


class CheckDecisionConfig(FunctionBaseConfig, name="e4b4_check_decision"):
    pass


@register_function(config_type=CheckDecisionConfig)
async def check_decision_fn(config: CheckDecisionConfig, builder: Builder):
    async def _run(request: DecisionInput) -> str:
        args = request.model_dump(mode="python")
        return V1._traced("e4b4_check_decision", args, lambda: check_decision(args))
    yield FunctionInfo.from_fn(_run, input_schema=DecisionInput, description=(
        "Check a diagnosis decision against the public contract. " + V2.CHECK_NOTE))


class SubmitDecisionConfig(FunctionBaseConfig, name="e4b4_submit_decision"):
    pass


@register_function(config_type=SubmitDecisionConfig)
async def submit_decision_fn(config: SubmitDecisionConfig, builder: Builder):
    async def _run(request: DecisionInput) -> str:
        return submit_decision(request.model_dump(mode="python"))
    yield FunctionInfo.from_fn(_run, input_schema=DecisionInput, description=(
        "Submit the diagnosis decision once. It is validated; a rejection ends the run, so never resubmit."))


class CheckProposalConfig(FunctionBaseConfig, name="e4b4_check_proposal"):
    pass


@register_function(config_type=CheckProposalConfig)
async def check_proposal_fn(config: CheckProposalConfig, builder: Builder):
    async def _run(request: V3.ProposalInput) -> str:
        args = request.model_dump(mode="python")
        return V1._traced("e4b4_check_proposal", args, lambda: check_proposal(args))
    yield FunctionInfo.from_fn(_run, input_schema=V3.ProposalInput, description=(
        "Check a correction proposal against the public contract. " + V2.CHECK_NOTE))


class SubmitProposalConfig(FunctionBaseConfig, name="e4b4_submit_proposal"):
    pass


@register_function(config_type=SubmitProposalConfig)
async def submit_proposal_fn(config: SubmitProposalConfig, builder: Builder):
    async def _run(request: V3.ProposalInput) -> str:
        return submit_proposal(request.model_dump(mode="python"))
    yield FunctionInfo.from_fn(_run, input_schema=V3.ProposalInput, description=(
        "Submit the correction proposal once. It is validated; a rejection ends the run, so never resubmit."))
