"""NAT registrations for the bounded v2 tools; old registrations stay frozen."""
from typing import Literal

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig
from pydantic import Field

from rl_triage import grounded_bounded_v2 as B
from rl_triage import grounded_evidence as G
from rl_triage.nat_functions import _dump, _trace
from rl_triage.nat_single_run import (
    GroundedActiveHypothesis,
    GroundedActiveHypothesisPrediction,
    GroundedActiveRegistryInput,
    GroundedActiveSelectionBinding,
    GroundedActiveStrictInput,
)


class NextCheck(GroundedActiveStrictInput):
    check_id: Literal["reset_base_com_stability", "runtime_com_state", "runtime_contact_window", "runtime_reward_window"]
    hypothesis_predictions: list[GroundedActiveHypothesisPrediction] = Field(min_length=2)


class AssessmentInput(GroundedActiveStrictInput):
    case_id: str
    catalog_sha256: str
    selection_binding: GroundedActiveSelectionBinding
    hypotheses: list[GroundedActiveHypothesis] = Field(min_length=2)
    next_check: NextCheck


def call(tool: str, args: dict) -> str:
    session = B.CURRENT_SESSION.get()
    if session is None:
        raise B.ToolBoundaryError("bounded_session_missing")
    try:
        result = session.call(tool, args)
    except Exception:
        _trace(tool, args, _dump({"error": "bounded_tool_rejected"}))
        raise
    return _trace(tool, args, _dump(result))


class RegistryConfig(FunctionBaseConfig, name="triage_bounded_registry_v2"):
    pass


@register_function(config_type=RegistryConfig)
async def registry_fn(config: RegistryConfig, builder: Builder):
    async def _run(request: GroundedActiveRegistryInput) -> str:
        return call("grounded_active_registry", {})
    yield FunctionInfo.from_fn(_run, input_schema=GroundedActiveRegistryInput,
                              description="Read once. Returns selectable_check_ids and hard per-session tool limits. Never executes checks.")


class IndexConfig(FunctionBaseConfig, name="triage_bounded_index_v2"):
    pass


@register_function(config_type=IndexConfig)
async def index_fn(config: IndexConfig, builder: Builder):
    async def _run(case_id: str) -> str:
        return call("grounded_evidence_group_index", {"case_id": case_id})
    yield FunctionInfo.from_fn(_run, description="Read once. Neutral group paths and source digests, without values.")


class PageConfig(FunctionBaseConfig, name="triage_bounded_page_v2"):
    pass


@register_function(config_type=PageConfig)
async def page_fn(config: PageConfig, builder: Builder):
    async def _run(case_id: str, catalog_sha256: str, group_id: str,
                   offset: int = 0, limit: int = G.GROUP_PAGE_LIMIT) -> str:
        return call("grounded_evidence_group_page", {"case_id": case_id, "catalog_sha256": catalog_sha256,
                    "group_id": group_id, "offset": offset, "limit": limit})
    yield FunctionInfo.from_fn(_run, description="At most TWO calls total. Select a group from the index. Further calls fail before backend execution.")


class DetailsConfig(FunctionBaseConfig, name="triage_bounded_details_v2"):
    pass


@register_function(config_type=DetailsConfig)
async def details_fn(config: DetailsConfig, builder: Builder):
    async def _run(case_id: str, catalog_sha256: str, evidence_ids: list[str]) -> str:
        return call("grounded_evidence_details", {"case_id": case_id, "catalog_sha256": catalog_sha256,
                    "evidence_ids": evidence_ids})
    yield FunctionInfo.from_fn(_run, description="Read once, up to 16 selected IDs. Preserve the returned host selection_binding unchanged.")


class WriteConfig(FunctionBaseConfig, name="triage_bounded_assessment_v2"):
    pass


@register_function(config_type=WriteConfig)
async def write_fn(config: WriteConfig, builder: Builder):
    async def _run(request: AssessmentInput) -> str:
        return call("write_grounded_active_assessment", request.model_dump(mode="python"))
    yield FunctionInfo.from_fn(_run, input_schema=AssessmentInput,
                              description="Write once. next_check.check_id must be one selectable_check_ids entry, NOT the registry ID. Host validates premises and exclusive predictions. No approval or execution.")
