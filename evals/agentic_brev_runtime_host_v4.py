"""Parent6 prepare 계약을 fresh Runtime Host3 검증에 결합한다."""
from __future__ import annotations

import threading
from contextlib import contextmanager

import agentic_brev_runtime_host_v3 as V3
import run_agentic_brev_probe_parent_v6 as P6

SCHEMA = "agentic_brev_runtime_host_v4"
SOURCES = V3.SOURCES | P6.SOURCES | {
    "evals/agentic_brev_runtime_host_v4.py",
    "tests/test_agentic_brev_runtime_host_v4.py",
}
_LOCK = threading.RLock()


@contextmanager
def _host4_scope():
    with _LOCK:
        saved = V3.P2, V3.SCHEMA, V3.SOURCES
        V3.P2, V3.SCHEMA, V3.SOURCES = P6, SCHEMA, SOURCES
        try:
            yield
        finally:
            V3.P2, V3.SCHEMA, V3.SOURCES = saved


class BrevRuntimeHost(V3.BrevRuntimeHost):
    def _binding(self, operation, intent, registration_ref=None):
        with _host4_scope():
            return super()._binding(operation, intent, registration_ref)
