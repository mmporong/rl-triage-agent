"""Host-v4가 Parent-v7의 명시적 fresh 선택 계약을 소비한다."""
from __future__ import annotations

from contextlib import contextmanager

import agentic_brev_runtime_host_v4 as V4
import run_agentic_brev_probe_parent_v7 as P7

SCHEMA = V4.SCHEMA
SOURCES = V4.SOURCES | P7.SOURCES | {
    "evals/agentic_brev_runtime_host_v5.py",
    "tests/test_agentic_brev_runtime_host_v5.py",
}


@contextmanager
def _parent7_scope():
    with V4._LOCK:
        saved = V4.P6, V4.SOURCES
        V4.P6, V4.SOURCES = P7, SOURCES
        try:
            yield
        finally:
            V4.P6, V4.SOURCES = saved


class BrevRuntimeHost(V4.BrevRuntimeHost):
    def _binding(self, operation, intent, registration_ref=None):
        with _parent7_scope():
            return super()._binding(operation, intent, registration_ref)
