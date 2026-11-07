"""Bounded grounded tools; this layer never approves or executes an inspection."""
from __future__ import annotations

from collections import Counter
from contextvars import ContextVar
from copy import deepcopy
from threading import RLock

from rl_triage import grounded_active_profile as A
from rl_triage import grounded_evidence as G

TOOL_LIMITS = {
    "grounded_active_registry": 1,
    "grounded_evidence_group_index": 1,
    "grounded_evidence_group_page": 2,
    "grounded_evidence_details": 1,
    "write_grounded_active_assessment": 1,
}
CHECK_IDS = tuple(sorted(A.ACTIVE_CHECKS))
CURRENT_SESSION: ContextVar[BoundedSession | None] = ContextVar("grounded_bounded_session_v2", default=None)


class ToolBoundaryError(ValueError):
    """A refused call is terminal for this workflow, before backend execution."""


class BoundedSession:
    def __init__(self, case_id: str):
        self.case_id = case_id
        self.catalog_sha256 = G.evidence_catalog(case_id)["catalog_sha256"]
        self.counts: Counter[str] = Counter()
        self.events: list[dict] = []
        self.terminal_reason: str | None = None
        self.completed = False
        self._lock = RLock()

    def _deny(self, tool: str, reason: str) -> None:
        self.events.append({"tool": tool, "status": "denied", "reason": reason,
                            "backend_entered": False})
        self.terminal_reason = self.terminal_reason or reason
        raise ToolBoundaryError(reason)

    def call(self, tool: str, arguments: dict) -> dict:
        """Serialize admission and execution so parallel tool calls cannot race a cap."""
        with self._lock:
            if self.terminal_reason or self.completed:
                self._deny(tool, "session_closed")
            if tool not in TOOL_LIMITS:
                self._deny(tool, "tool_not_registered")
            if not isinstance(arguments, dict):
                self._deny(tool, "arguments_not_object")
            if tool != "grounded_active_registry" and arguments.get("case_id") != self.case_id:
                self._deny(tool, "case_binding_mismatch")
            if self.counts[tool] >= TOOL_LIMITS[tool]:
                self._deny(tool, "tool_budget_exhausted")
            if tool in {"grounded_evidence_group_page", "grounded_evidence_details",
                        "write_grounded_active_assessment"}:
                if not self.counts["grounded_active_registry"] or not self.counts["grounded_evidence_group_index"]:
                    self._deny(tool, "registry_and_index_required")
                if arguments.get("catalog_sha256") != self.catalog_sha256:
                    self._deny(tool, "catalog_binding_mismatch")
            if tool == "grounded_evidence_details" and not self.counts["grounded_evidence_group_page"]:
                self._deny(tool, "page_required")
            if tool == "write_grounded_active_assessment" and not self.counts["grounded_evidence_details"]:
                self._deny(tool, "selection_required")
            if G.evidence_catalog(self.case_id)["catalog_sha256"] != self.catalog_sha256:
                self._deny(tool, "source_changed")
            self.counts[tool] += 1
            event = {"tool": tool, "status": "entered", "backend_entered": True}
            self.events.append(event)
            try:
                result = self._dispatch(tool, deepcopy(arguments))
                if "error" in result or result.get("status") == "hold":
                    raise ToolBoundaryError("backend_semantic_rejection")
                if tool == "write_grounded_active_assessment":
                    if result.get("status") != "hypotheses_only" or not result.get("assessment_sha256"):
                        raise ToolBoundaryError("assessment_not_saved")
                    self.completed = True
                event["status"] = "completed"
                return result
            except Exception as error:
                event.update(status="rejected", reason=type(error).__name__)
                self.terminal_reason = "backend_rejected"
                raise

    def _dispatch(self, tool: str, args: dict) -> dict:
        if tool == "grounded_active_registry":
            if args:
                raise ToolBoundaryError("registry_arguments_must_be_empty")
            profile = A.active_registry_profile()
            return {**profile, "schema": "grounded_bounded_registry_v2",
                    "selectable_check_ids": list(CHECK_IDS), "tool_limits": dict(TOOL_LIMITS),
                    "next_check_field": "check_id", "legacy_storage_field": "registry_id",
                    "registration_is_execution": False}
        if tool == "grounded_evidence_group_index":
            return G.evidence_catalog_group_index(**args)
        if tool == "grounded_evidence_group_page":
            return G.evidence_catalog_group_page(**args)
        if tool == "grounded_evidence_details":
            return G.evidence_catalog_details(**args)
        # Translate a documented field name only; never repair model values.
        next_check = args["next_check"]
        if set(next_check) != {"check_id", "hypothesis_predictions"} or next_check["check_id"] not in CHECK_IDS:
            raise ToolBoundaryError("check_id_not_registered")
        args["next_check"] = {"registry_id": next_check["check_id"],
                              "hypothesis_predictions": next_check["hypothesis_predictions"]}
        return A.write_grounded_active_assessment(**args)

    def receipt(self) -> dict:
        return {"schema": "grounded_bounded_session_v2", "case_id": self.case_id,
                "catalog_sha256": self.catalog_sha256, "limits": dict(TOOL_LIMITS),
                "executed_calls": dict(self.counts), "events": deepcopy(self.events),
                "completed": self.completed, "terminal_reason": self.terminal_reason,
                "approval_created": False, "inspection_executed": False,
                "gpu_runs": 0, "cause_confirmed": False, "recovery_confirmed": False}
