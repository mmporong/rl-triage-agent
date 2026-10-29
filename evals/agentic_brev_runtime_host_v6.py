"""Host5의 fresh parent plan 결합과 prefix 밖 요청의 HOLD 경계를 고친다."""
from __future__ import annotations

from contextlib import contextmanager

import agentic_brev_runtime_host_v1 as V1
import agentic_brev_runtime_host_v5 as V5
import run_agentic_brev_probe_parent_v7 as P7

SCHEMA = V5.SCHEMA
PARENT_SCHEMA = P7.SCHEMA
SOURCES = V5.SOURCES | P7.SOURCES | {
    "evals/agentic_brev_runtime_host_v6.py",
    "tests/test_agentic_brev_runtime_host_v6.py",
}


@contextmanager
def _parent7_evidence_scope(protocol):
    """V1 evidence 검증이 등록된 Parent7 계약을 사용하게 좁게 바꾼다."""
    V1.require(
        protocol.get("schema") == PARENT_SCHEMA + "_protocol"
        and P7.SOURCES <= protocol.get("sources_lf", {}).keys(),
        "registered Parent7 protocol required",
    )
    with V5.V4._LOCK:
        saved = V1.P
        V1.P = P7
        try:
            yield
        finally:
            V1.P = saved


class BrevRuntimeHost(V5.BrevRuntimeHost):
    """Host5 등록을 유지하며 Parent7 plan과 unsupported ordinal을 구분한다."""

    def _evidence(self, bound, arm, result_ref=None):
        with _parent7_evidence_scope(bound["protocol"]):
            return super()._evidence(bound, arm, result_ref)

    def _unsupported_fresh_ordinal(self, operation, intent):
        if intent.get("action") != "run_probe":
            return False
        expected = V1.J.digest({"manifest": self.runtime.manifest, "intent": intent})
        if operation != expected:
            return False
        steps = [
            path.parent
            for path in (self.runtime.folder / "steps").glob("*/operation.json")
            if V1.J._read(path).get("operation_id") == operation
        ]
        if len(steps) != 1:
            return False
        step = steps[0]
        ordinal = int(step.name)
        if ordinal <= len(self.runtime.reg["prefix"]):
            return False
        archived_intent = V1.J._read(step / "intent.json")
        V1.require(archived_intent == intent, "Runtime immutable intent changed")
        V1.require(
            V1.J._read(step / "operation.json")
            == {"operation_id": operation, "intent_sha256": V1.J.file_sha(step / "intent.json")},
            "Runtime immutable operation record changed",
        )
        return True

    def approve(self, operation, intent):
        self._owner()
        self._sources()
        if self._unsupported_fresh_ordinal(operation, intent):
            return {
                "status": "hold",
                "reason": "fresh_prefix_ordinal_unsupported",
                "operation_id": operation,
            }
        return super().approve(operation, intent)
