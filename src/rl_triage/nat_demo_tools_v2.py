"""E4b v2: NAT 도구 호출 결정 경로의 도구를 v1 실패 형태에 맞춰 고친 판.

v1(nat_demo_tools.py)은 v1 사전등록 결과의 소스 결합 때문에 바꾸지 않고, 읽기·기록·한 번 제출 구현을 그대로 재사용한다.
v1과 다른 점:
- 도구 이름 e4b2_*(같은 프로세스에서 v1 등록과 겹치지 않게)
- 제출·점검 입력 스키마에 필드 설명(허용 값 목록 포함)을 붙였다. v1 실패[사실]는 교정 5/5의 hold_reason 문자열
  "None", 4/5의 쉼표로 이은 체크리스트, 진단 2/5의 6개 expected_outcomes였다. 모델에 가는 도구 스키마에는 필드 설명과
  실제 검증 제약만 남는다(langchain 변환이 json_schema_extra를 버린다). 입력 검증은 v1처럼 느슨하게 둬 형식이 틀린
  입력도 도구 호출 단계에서 끊기지 않고 점검·계약 검증이 사유를 남긴다.
- 제출 전 계약 점검 도구(e4b2_check_decision·e4b2_check_proposal): 제출 도구와 같은 공개 계약 검증기로 결과만 돌려주고
  저장하지 않는다. 실행당 PRECHECK_LIMIT회까지 검증하고, 넘으면 검증 없이 거부 메시지를 돌려준다. 단일 호출 경로에는
  없는 기능이며 호출은 도구 기록(E4B_TRACE)에 남는다.

  E4B_KIND=diagnosis|correction  E4B_PACKET=<packet>  E4B_PACKET_SHA256=<raw sha>
  E4B_SUBMISSION=<제출 파일>  E4B_TRACE=<도구 호출 기록 jsonl>
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

from rl_triage import active_diagnosis as D
from rl_triage import correction_registry as R
from rl_triage import nat_demo_tools as V1
from rl_triage import probe_loop as L

PRECHECK_LIMIT = 2
PRECHECK_TOOLS = ("e4b2_check_decision", "e4b2_check_proposal")
SubmissionRejected = V1.SubmissionRejected
CORRECTION_IDS = sorted({entry["correction_id"] for entry in R.REGISTRY.values()})


def precheck_calls(trace_path: str | None = None) -> int:
    """도구 기록에 남은 점검 호출 수(상한을 넘은 호출 포함)."""
    path = trace_path or os.environ.get("E4B_TRACE")
    if not path or not os.path.exists(path):
        return 0
    with open(path, encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip() and json.loads(line).get("tool") in PRECHECK_TOOLS)


def _precheck(validate) -> dict:
    if precheck_calls() >= PRECHECK_LIMIT:
        return {"ok": False, "checked": False,
                "reasons": [f"precheck limit of {PRECHECK_LIMIT} calls reached; submit your final object now"]}
    reasons = validate()
    return {"ok": not reasons, "checked": True, "reasons": reasons}


# --- 도구 구현(등록과 분리해 테스트에서 바로 부른다) ---
def check_decision(decision: dict) -> dict:
    _, packet = V1.context("diagnosis")

    def validate() -> list[str]:
        try:
            D.validate_decision(decision, packet)
        except (ValueError, TypeError) as error:
            return [str(error)]
        return []
    return _precheck(validate)


def check_proposal(proposal: dict) -> dict:
    _, packet = V1.context("correction")
    return _precheck(lambda: V1._correction_module().validate_model_proposal(proposal, packet)["reasons"])


def submit_decision(decision: dict) -> str:
    tool = "e4b2_submit_decision"
    _, packet = V1.context("diagnosis")
    V1._guard_once(tool, decision)
    try:
        canonical = D.validate_decision(decision, packet)
    except (ValueError, TypeError) as error:
        V1._reject(tool, decision, [str(error)])
    return V1._save(tool, decision, {"kind": "diagnosis", "decision": canonical})


def submit_proposal(proposal: dict) -> str:
    tool = "e4b2_submit_proposal"
    _, packet = V1.context("correction")
    V1._guard_once(tool, proposal)
    verdict = V1._correction_module().validate_model_proposal(proposal, packet)
    if not verdict["accepted"]:
        V1._reject(tool, proposal, verdict["reasons"])
    return V1._save(tool, proposal, {"kind": "correction", "proposal": proposal, "validation": verdict})


# --- NAT 입력 스키마: 검증은 v1처럼 느슨하게, 규칙은 모델이 보는 필드 설명에 ---
class DecisionInput(V1.StrictInput):
    ranking: list[str] = Field(
        description=f"All six mechanisms ({', '.join(L.MECHANISMS)}) exactly once, most likely first.")
    probe: str | None = Field(
        description=f"For continue: one probe name from the catalog ({', '.join(L.PROBES)}). For hold or confirm: JSON null.")
    action: Literal["continue", "hold", "confirm"]
    reason: str = Field(description="One short sentence.")
    claims: list[V1.Claim] = Field(
        min_length=1,
        description="Evidence fact ids read with e4b2_get_facts, each with its value copied exactly.")
    expected_outcomes: dict[str, Literal["abnormal", "normal"] | None] = Field(
        description=("For continue: keys are ONLY the top three mechanisms of your ranking (exactly three keys); each "
                     "value is copied from the selected probe's catalog 'expect' entry. Do not add the other three "
                     "mechanisms. For hold or confirm: an empty object {}."))


class ProposalInput(V1.StrictInput):
    correction_id: str | None = Field(
        description=f"One registered correction id ({', '.join(CORRECTION_IDS)}) when proposing; JSON null when holding.")
    recovery_mode: str | None = Field(
        description=(f"One registered recovery mode ({', '.join(R.RECOVERY_MODES)}) when proposing; "
                     "JSON null when holding."))
    cited_facts: list[str] = Field(
        description="At least one fact id from e4b2_list_fact_ids, one id per list item.")
    checklist: dict[str, str] = Field(
        description=("One entry per skill checklist item number. Each value is exactly ONE fact id that satisfies "
                     "that item, or the string \"unmet\". Never join several ids with commas, and never use a "
                     "correction id, because a correction id is not a fact id."))
    hold_reason: str | None = Field(
        description=("When proposing a correction this MUST be JSON null; the strings \"None\", \"null\" and \"\" are "
                     "not null. Only when holding: a non-empty reason, with correction_id and recovery_mode null."))


CHECK_NOTE = f"Returns ok and reasons without submitting. At most {PRECHECK_LIMIT} checks per run."


class ListFactIdsConfig(FunctionBaseConfig, name="e4b2_list_fact_ids"):
    pass


@register_function(config_type=ListFactIdsConfig)
async def list_fact_ids_fn(config: ListFactIdsConfig, builder: Builder):
    async def _run(request: V1.EmptyInput) -> str:
        return V1._traced("e4b2_list_fact_ids", {}, V1.list_fact_ids)
    yield FunctionInfo.from_fn(_run, input_schema=V1.EmptyInput, description=(
        "List the evidence fact ids of the packet. Read their values with e4b2_get_facts."))


class GetFactsConfig(FunctionBaseConfig, name="e4b2_get_facts"):
    pass


@register_function(config_type=GetFactsConfig)
async def get_facts_fn(config: GetFactsConfig, builder: Builder):
    async def _run(request: V1.FactIdsInput) -> str:
        return V1._traced("e4b2_get_facts", {"ids": request.ids}, lambda: V1.get_facts(request.ids))
    yield FunctionInfo.from_fn(_run, input_schema=V1.FactIdsInput, description=(
        f"Return the values of up to {V1.MAX_FACT_IDS} fact ids. Unknown ids are listed in unknown_ids."))


class ProbeCatalogConfig(FunctionBaseConfig, name="e4b2_probe_catalog"):
    pass


@register_function(config_type=ProbeCatalogConfig)
async def probe_catalog_fn(config: ProbeCatalogConfig, builder: Builder):
    async def _run(request: V1.EmptyInput) -> str:
        return V1._traced("e4b2_probe_catalog", {}, V1.probe_catalog)
    yield FunctionInfo.from_fn(_run, input_schema=V1.EmptyInput, description=(
        "Return the public probe catalog (measures, cost, args, expected outcome per mechanism) and thresholds."))


class RegistryConfig(FunctionBaseConfig, name="e4b2_registry"):
    pass


@register_function(config_type=RegistryConfig)
async def registry_fn(config: RegistryConfig, builder: Builder):
    async def _run(request: V1.EmptyInput) -> str:
        return V1._traced("e4b2_registry", {}, V1.registry)
    yield FunctionInfo.from_fn(_run, input_schema=V1.EmptyInput, description=(
        "Return the confirmed mechanism and the registered corrections and recovery modes."))


class SkillChecklistConfig(FunctionBaseConfig, name="e4b2_skill_checklist"):
    pass


@register_function(config_type=SkillChecklistConfig)
async def skill_checklist_fn(config: SkillChecklistConfig, builder: Builder):
    async def _run(request: V1.EmptyInput) -> str:
        return V1._traced("e4b2_skill_checklist", {}, V1.skill_checklist)
    yield FunctionInfo.from_fn(_run, input_schema=V1.EmptyInput, description=(
        "Return the six validation checklist items of the pinned Isaac Lab debug-rl-training skill."))


class CheckDecisionConfig(FunctionBaseConfig, name="e4b2_check_decision"):
    pass


@register_function(config_type=CheckDecisionConfig)
async def check_decision_fn(config: CheckDecisionConfig, builder: Builder):
    async def _run(request: DecisionInput) -> str:
        args = request.model_dump(mode="python")
        return V1._traced("e4b2_check_decision", args, lambda: check_decision(args))
    yield FunctionInfo.from_fn(_run, input_schema=DecisionInput, description=(
        "Check a diagnosis decision against the public contract. " + CHECK_NOTE))


class CheckProposalConfig(FunctionBaseConfig, name="e4b2_check_proposal"):
    pass


@register_function(config_type=CheckProposalConfig)
async def check_proposal_fn(config: CheckProposalConfig, builder: Builder):
    async def _run(request: ProposalInput) -> str:
        args = request.model_dump(mode="python")
        return V1._traced("e4b2_check_proposal", args, lambda: check_proposal(args))
    yield FunctionInfo.from_fn(_run, input_schema=ProposalInput, description=(
        "Check a correction proposal against the public contract. " + CHECK_NOTE))


class SubmitDecisionConfig(FunctionBaseConfig, name="e4b2_submit_decision"):
    pass


@register_function(config_type=SubmitDecisionConfig)
async def submit_decision_fn(config: SubmitDecisionConfig, builder: Builder):
    async def _run(request: DecisionInput) -> str:
        return submit_decision(request.model_dump(mode="python"))
    yield FunctionInfo.from_fn(_run, input_schema=DecisionInput, description=(
        "Submit the diagnosis decision once. It is validated; a rejection ends the run, so never resubmit."))


class SubmitProposalConfig(FunctionBaseConfig, name="e4b2_submit_proposal"):
    pass


@register_function(config_type=SubmitProposalConfig)
async def submit_proposal_fn(config: SubmitProposalConfig, builder: Builder):
    async def _run(request: ProposalInput) -> str:
        return submit_proposal(request.model_dump(mode="python"))
    yield FunctionInfo.from_fn(_run, input_schema=ProposalInput, description=(
        "Submit the correction proposal once. It is validated; a rejection ends the run, so never resubmit."))
