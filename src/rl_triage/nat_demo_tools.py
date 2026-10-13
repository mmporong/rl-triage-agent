"""E4b: 종단 시연의 두 모델 결정을 NeMo Agent Toolkit 도구 호출로 돌리기 위한 읽기·제출 도구.

도구는 환경변수로 받은 packet 파일만 읽고, 매 호출 그 SHA256을 대조한다. 제출 도구는 기존 판단 계약
(active_diagnosis.validate_decision, correction_model_call.validate_model_proposal)을 그대로 import해 통과할 때만
제출 파일을 한 번 쓴다. 거절과 두 번째 제출은 거절 기록을 남기고 예외로 워크플로를 멈춘다(의미상 재시도 없음).
증거 상태 계산 도구는 두지 않는다. 판단 계약이 모델에게 그 계산을 요구하므로 측정 대상을 바꾸지 않는다.

  E4B_KIND=diagnosis|correction  E4B_PACKET=<packet>  E4B_PACKET_SHA256=<raw sha>
  E4B_SUBMISSION=<제출 파일>  E4B_TRACE=<도구 호출 기록 jsonl>
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Literal, NoReturn

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig
from pydantic import BaseModel, ConfigDict, Field

from rl_triage import active_diagnosis as D
from rl_triage import probe_loop as L

ROOT = Path(__file__).resolve().parents[2]
MAX_FACT_IDS = 20
KINDS = ("diagnosis", "correction")


class SubmissionRejected(ValueError):
    """제출이 판단 계약을 어겨 워크플로를 멈춘다."""


def _correction_module():
    if str(ROOT / "evals") not in sys.path:
        sys.path.insert(0, str(ROOT / "evals"))
    import correction_model_call as CM  # noqa: E402  (evals 모듈, 교정 종류일 때만 불러온다)
    return CM


def context(expected_kind: str | None = None) -> tuple[str, dict]:
    """(종류, packet). packet 파일이 등록 SHA와 다르거나 종류가 맞지 않으면 예외."""
    kind = os.environ.get("E4B_KIND")
    if kind not in KINDS or (expected_kind is not None and kind != expected_kind):
        raise ValueError(f"이 도구는 {expected_kind} 결정에서만 쓴다(현재 {kind})")
    data = Path(os.environ["E4B_PACKET"]).read_bytes()
    if hashlib.sha256(data).hexdigest() != os.environ.get("E4B_PACKET_SHA256"):
        raise ValueError("packet이 등록 SHA와 다르다")
    return kind, json.loads(data)


def _dump(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _trace(tool: str, args: dict, result: str | None, error: str | None = None) -> None:
    path = os.environ.get("E4B_TRACE")
    if not path:
        return
    row = {"t": time.time(), "tool": tool, "args": args, "error": error,
           "result_sha256": None if result is None else hashlib.sha256(result.encode()).hexdigest(),
           "result_head": None if result is None else result[:400]}
    with open(path, "a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def _submission_path() -> Path:
    return Path(os.environ["E4B_SUBMISSION"])


def _reject(tool: str, args: dict, reasons: list[str]) -> NoReturn:
    rejected = _submission_path().with_suffix(".rejected.json")
    if not rejected.exists():
        with rejected.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(_dump({"tool": tool, "reasons": reasons, "submitted": args}) + "\n")
    _trace(tool, args, None, error="; ".join(reasons)[:400])
    raise SubmissionRejected(f"{tool} rejected: {'; '.join(reasons)[:300]}")


def _guard_once(tool: str, args: dict) -> None:
    """이미 제출했거나 거절된 실행에서는 다시 받지 않는다(병렬·순차 재제출 모두)."""
    if _submission_path().exists() or _submission_path().with_suffix(".rejected.json").exists():
        _reject(tool, args, ["한 실행에서 제출은 한 번뿐이다(이미 제출 또는 거절됨)"])


def _save(tool: str, args: dict, body: dict) -> str:
    path = _submission_path()
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(_dump(body) + "\n")
    out = _dump({"saved": True, "submission_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    _trace(tool, args, out)
    return out


# --- 도구 구현(등록과 분리해 테스트에서 바로 부른다) ---
def list_fact_ids() -> dict:
    _, packet = context()
    return {"fact_ids": sorted(packet["facts"])}


def get_facts(ids: list[str]) -> dict:
    _, packet = context()
    facts = packet["facts"]
    return {"facts": {i: facts[i] for i in ids if i in facts}, "unknown_ids": [i for i in ids if i not in facts]}


def probe_catalog() -> dict:
    context("diagnosis")
    return {"probe_catalog": {name: {k: spec[k] for k in ("measures", "cost_gpu_s", "args", "expect")}
                              for name, spec in L.PROBES.items()},
            "thresholds": {name: list(rule) for name, rule in L.THRESHOLDS.items()}}


def registry() -> dict:
    _, packet = context("correction")
    return {"confirmed_mechanism": packet["mechanism"], "registry": packet["registry"]}


def skill_checklist() -> dict:
    _, packet = context("correction")
    CM = _correction_module()
    if packet["checklist"] != CM.skill_checklist():
        raise ValueError("packet 체크리스트가 고정 skill 원문과 다르다")
    return {"skill_checklist": packet["checklist"], "skill_source": packet["skill_source"]}


def submit_decision(decision: dict) -> str:
    tool = "e4b_submit_decision"
    _, packet = context("diagnosis")
    _guard_once(tool, decision)
    try:
        canonical = D.validate_decision(decision, packet)
    except (ValueError, TypeError) as error:
        _reject(tool, decision, [str(error)])
    return _save(tool, decision, {"kind": "diagnosis", "decision": canonical})


def submit_proposal(proposal: dict) -> str:
    tool = "e4b_submit_proposal"
    _, packet = context("correction")
    _guard_once(tool, proposal)
    verdict = _correction_module().validate_model_proposal(proposal, packet)
    if not verdict["accepted"]:
        _reject(tool, proposal, verdict["reasons"])
    return _save(tool, proposal, {"kind": "correction", "proposal": proposal, "validation": verdict})


# --- NAT 입력 스키마 ---
class StrictInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EmptyInput(StrictInput):
    pass


class FactIdsInput(StrictInput):
    ids: list[str] = Field(min_length=1, max_length=MAX_FACT_IDS, description="e4b_list_fact_ids가 돌려준 사실 id")


class Claim(StrictInput):
    id: str
    value: str | bool | int | float


class DecisionInput(StrictInput):
    ranking: list[str] = Field(description="여섯 공개 mechanism의 순열")
    probe: str | None
    action: Literal["continue", "hold", "confirm"]
    reason: str
    claims: list[Claim] = Field(min_length=1)
    expected_outcomes: dict[str, Literal["abnormal", "normal"] | None]


class ProposalInput(StrictInput):
    correction_id: str | None
    recovery_mode: str | None
    cited_facts: list[str]
    checklist: dict[str, str] = Field(description="skill 체크리스트 번호 → 사실 id 또는 unmet")
    hold_reason: str | None


def _traced(tool: str, args: dict, fn) -> str:
    try:
        out = _dump(fn())
    except Exception as error:
        _trace(tool, args, None, error=f"{type(error).__name__}: {error}"[:400])
        raise
    _trace(tool, args, out)
    return out


class ListFactIdsConfig(FunctionBaseConfig, name="e4b_list_fact_ids"):
    pass


@register_function(config_type=ListFactIdsConfig)
async def list_fact_ids_fn(config: ListFactIdsConfig, builder: Builder):
    async def _run(request: EmptyInput) -> str:
        return _traced("e4b_list_fact_ids", {}, list_fact_ids)
    yield FunctionInfo.from_fn(_run, input_schema=EmptyInput, description=(
        "packet의 근거 사실 id 목록을 돌려준다. 값은 e4b_get_facts로 읽는다."))


class GetFactsConfig(FunctionBaseConfig, name="e4b_get_facts"):
    pass


@register_function(config_type=GetFactsConfig)
async def get_facts_fn(config: GetFactsConfig, builder: Builder):
    async def _run(request: FactIdsInput) -> str:
        return _traced("e4b_get_facts", {"ids": request.ids}, lambda: get_facts(request.ids))
    yield FunctionInfo.from_fn(_run, input_schema=FactIdsInput, description=(
        f"사실 id 최대 {MAX_FACT_IDS}개의 값을 돌려준다. 없는 id는 unknown_ids로 알려 준다. 값을 바꾸지 않는다."))


class ProbeCatalogConfig(FunctionBaseConfig, name="e4b_probe_catalog"):
    pass


@register_function(config_type=ProbeCatalogConfig)
async def probe_catalog_fn(config: ProbeCatalogConfig, builder: Builder):
    async def _run(request: EmptyInput) -> str:
        return _traced("e4b_probe_catalog", {}, probe_catalog)
    yield FunctionInfo.from_fn(_run, input_schema=EmptyInput, description=(
        "공개 probe 카탈로그(측정·비용·인자·메커니즘별 예상)와 판정 임계값을 돌려준다. 진단 결정에서만 쓴다."))


class RegistryConfig(FunctionBaseConfig, name="e4b_registry"):
    pass


@register_function(config_type=RegistryConfig)
async def registry_fn(config: RegistryConfig, builder: Builder):
    async def _run(request: EmptyInput) -> str:
        return _traced("e4b_registry", {}, registry)
    yield FunctionInfo.from_fn(_run, input_schema=EmptyInput, description=(
        "확정 메커니즘과 등록 교정·회복 방식 표를 돌려준다. 교정 제안에서만 쓴다."))


class SkillChecklistConfig(FunctionBaseConfig, name="e4b_skill_checklist"):
    pass


@register_function(config_type=SkillChecklistConfig)
async def skill_checklist_fn(config: SkillChecklistConfig, builder: Builder):
    async def _run(request: EmptyInput) -> str:
        return _traced("e4b_skill_checklist", {}, skill_checklist)
    yield FunctionInfo.from_fn(_run, input_schema=EmptyInput, description=(
        "고정한 Isaac Lab debug-rl-training skill의 검증 체크리스트 6항목을 돌려준다. 교정 제안에서만 쓴다."))


class SubmitDecisionConfig(FunctionBaseConfig, name="e4b_submit_decision"):
    pass


@register_function(config_type=SubmitDecisionConfig)
async def submit_decision_fn(config: SubmitDecisionConfig, builder: Builder):
    async def _run(request: DecisionInput) -> str:
        return submit_decision(request.model_dump(mode="python"))
    yield FunctionInfo.from_fn(_run, input_schema=DecisionInput, description=(
        "진단 결정을 한 번 제출한다. 공개 판단 계약으로 검증하며 거절되면 실행이 끝난다(다시 제출하지 않는다)."))


class SubmitProposalConfig(FunctionBaseConfig, name="e4b_submit_proposal"):
    pass


@register_function(config_type=SubmitProposalConfig)
async def submit_proposal_fn(config: SubmitProposalConfig, builder: Builder):
    async def _run(request: ProposalInput) -> str:
        return submit_proposal(request.model_dump(mode="python"))
    yield FunctionInfo.from_fn(_run, input_schema=ProposalInput, description=(
        "교정 제안을 한 번 제출한다. 등록 표·인용·체크리스트 검증을 통과해야 저장되며 거절되면 실행이 끝난다."))
