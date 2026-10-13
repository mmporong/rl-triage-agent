"""E4b v3: null을 쓰지 않는 교정 제안 제출 스키마. 진단 도구는 v2(nat_demo_tools_v2)를 그대로 쓴다.

v2 결과[사실]: 교정 제출 20회 중 19회의 도구 인자 hold_reason이 문자열 "None"이었다. ATIF 궤적(파싱된 LLM 도구 호출)과
도구 기록(도구가 받은 인자)이 호출마다 같았고(run2 세 번째 점검만 둘 다 null), langchain은 인자를 json.loads로
파싱하므로 null을 "None"으로 바꾸지 않는다. 같은 계약의 단일 호출(응답 본문 JSON)은 null을 썼다.
[추론] "None"은 제공자 쪽 도구 호출 인자 생성·파싱 단계에서 생긴다. 모델과 제공자 파서 중 어느 쪽인지는 원시 응답을
남기지 않아 가를 수 없다.

그래서 v3 제출 스키마는 null을 요구하지 않는다: decision("propose"|"hold") 판별 칸을 두고, 쓰지 않는 칸은 빈
문자열 ""로 받는다. 어댑터(to_contract)가 공개 계약(correction_model_call.validate_model_proposal) 형태로 바꾼다.
형식이 판별 칸과 맞지 않으면 고쳐 주지 않고 사유를 돌려준다(예: propose인데 hold_reason이 비어 있지 않음).
계약 검증기는 바꾸지 않는다. 읽기 도구·점검 상한·한 번 제출 구현은 v1·v2를 재사용한다.
"""
from __future__ import annotations

import json
import os
from typing import Literal

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig
from pydantic import Field

from rl_triage import nat_demo_tools as V1
from rl_triage import nat_demo_tools_v2 as V2

PRECHECK_LIMIT = V2.PRECHECK_LIMIT
PRECHECK_TOOLS = ("e4b3_check_proposal",)
SubmissionRejected = V1.SubmissionRejected


def precheck_calls(trace_path: str | None = None) -> int:
    """도구 기록에 남은 v3 교정 점검 호출 수(상한을 넘은 호출 포함)."""
    path = trace_path or os.environ.get("E4B_TRACE")
    if not path or not os.path.exists(path):
        return 0
    with open(path, encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip() and json.loads(line).get("tool") in PRECHECK_TOOLS)


class ProposalInput(V1.StrictInput):
    decision: Literal["propose", "hold"] = Field(
        description="propose: one registered correction and recovery mode. hold: no correction, give a reason.")
    correction_id: str = Field(
        description=(f"When propose: one registered correction id ({', '.join(V2.CORRECTION_IDS)}). "
                     "When hold: the empty string \"\"."))
    recovery_mode: str = Field(
        description="When propose: one registered recovery mode from the registry. When hold: the empty string \"\".")
    cited_facts: list[str] = Field(description="At least one fact id from e4b2_list_fact_ids, one id per list item.")
    checklist: dict[str, str] = Field(
        description=("One entry per skill checklist item number. Each value is exactly ONE fact id that satisfies "
                     "that item, or the string \"unmet\". Never join several ids with commas, and never use a "
                     "correction id, because a correction id is not a fact id."))
    hold_reason: str = Field(
        description="When hold: a non-empty reason. When propose: the empty string \"\".")


def to_contract(args: dict) -> tuple[dict | None, list[str]]:
    """v3 도구 인자를 공개 계약 제안 형태로 바꾼다. 판별 칸과 맞지 않는 칸은 고치지 않고 사유로 돌려준다."""
    reasons = []
    base = {"cited_facts": args["cited_facts"], "checklist": args["checklist"]}
    if args["decision"] == "propose":
        if args["hold_reason"] != "":
            reasons.append('decision이 propose면 hold_reason은 빈 문자열 ""이어야 한다')
        if not args["correction_id"].strip() or not args["recovery_mode"].strip():
            reasons.append("decision이 propose면 correction_id와 recovery_mode는 비지 않은 등록 값이어야 한다")
        proposal = {**base, "correction_id": args["correction_id"], "recovery_mode": args["recovery_mode"],
                    "hold_reason": None}
    else:
        if args["correction_id"] != "" or args["recovery_mode"] != "":
            reasons.append('decision이 hold면 correction_id와 recovery_mode는 빈 문자열 ""이어야 한다')
        if not args["hold_reason"].strip():
            reasons.append("decision이 hold면 hold_reason은 비지 않은 이유여야 한다")
        proposal = {**base, "correction_id": None, "recovery_mode": None, "hold_reason": args["hold_reason"]}
    return (None if reasons else proposal), reasons


def _validate(args: dict) -> tuple[dict | None, dict]:
    """(계약 제안, 검증 결과). 어댑터 사유가 있으면 계약 검증 전에 거부한다."""
    _, packet = V1.context("correction")
    proposal, reasons = to_contract(args)
    if proposal is None:
        return None, {"accepted": False, "decision": None, "reasons": reasons, "agrees_with_rule": False,
                      "unmet_checklist": []}
    return proposal, V1._correction_module().validate_model_proposal(proposal, packet)


def check_proposal(args: dict) -> dict:
    if precheck_calls() >= PRECHECK_LIMIT:
        return {"ok": False, "checked": False,
                "reasons": [f"precheck limit of {PRECHECK_LIMIT} calls reached; submit your final object now"]}
    _, verdict = _validate(args)
    return {"ok": verdict["accepted"], "checked": True, "reasons": verdict["reasons"]}


def submit_proposal(args: dict) -> str:
    tool = "e4b3_submit_proposal"
    V1.context("correction")
    V1._guard_once(tool, args)
    proposal, verdict = _validate(args)
    if not verdict["accepted"]:
        V1._reject(tool, args, verdict["reasons"])
    return V1._save(tool, args, {"kind": "correction", "proposal": proposal, "validation": verdict,
                                 "tool_args": args})


class CheckProposalConfig(FunctionBaseConfig, name="e4b3_check_proposal"):
    pass


@register_function(config_type=CheckProposalConfig)
async def check_proposal_fn(config: CheckProposalConfig, builder: Builder):
    async def _run(request: ProposalInput) -> str:
        args = request.model_dump(mode="python")
        return V1._traced("e4b3_check_proposal", args, lambda: check_proposal(args))
    yield FunctionInfo.from_fn(_run, input_schema=ProposalInput, description=(
        "Check a correction proposal against the public contract. " + V2.CHECK_NOTE))


class SubmitProposalConfig(FunctionBaseConfig, name="e4b3_submit_proposal"):
    pass


@register_function(config_type=SubmitProposalConfig)
async def submit_proposal_fn(config: SubmitProposalConfig, builder: Builder):
    async def _run(request: ProposalInput) -> str:
        return submit_proposal(request.model_dump(mode="python"))
    yield FunctionInfo.from_fn(_run, input_schema=ProposalInput, description=(
        "Submit the correction proposal once. It is validated; a rejection ends the run, so never resubmit."))
