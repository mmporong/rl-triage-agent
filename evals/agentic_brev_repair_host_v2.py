"""RepairHost1을 exact Host6와 동일 Runtime에만 연결한다."""
from __future__ import annotations

import threading
from contextlib import contextmanager

import agentic_brev_repair_host_v1 as V1
import agentic_brev_runtime_host_v6 as H6

SOURCES = V1.SOURCES | H6.SOURCES | {
    "evals/agentic_brev_repair_host_v2.py",
    "tests/test_agentic_brev_repair_host_v2.py",
}
_LOCK = threading.RLock()


@contextmanager
def _host6_scope():
    with _LOCK:
        saved = V1.H5, V1.SOURCES
        V1.H5, V1.SOURCES = H6, SOURCES
        try:
            yield
        finally:
            V1.H5, V1.SOURCES = saved


class BrevRepairRuntimeHost(V1.BrevRepairRuntimeHost):
    """기존 repair 검증을 재사용하되 exact Host6를 요구한다."""

    def __init__(self, runtime, probe_host, provider, registration_ref, root_map, approver, *,
                 probe_seals, probe_records, admitter, allow_fixture=False):
        V1.E.require(
            type(probe_host) is H6.BrevRuntimeHost and probe_host.runtime is runtime,
            "exact Host6 from the same Runtime required",
        )
        with _host6_scope():
            super().__init__(runtime, probe_host, provider, registration_ref, root_map, approver,
                             probe_seals=probe_seals, probe_records=probe_records,
                             admitter=admitter, allow_fixture=allow_fixture)

    def _sources(self):
        with _host6_scope():
            return super()._sources()
