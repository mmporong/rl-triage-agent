"""Parent-v6의 fresh 선택 함수를 Host 소비자에 명시적으로 노출한다."""
from __future__ import annotations

import run_agentic_brev_probe_parent_v6 as V6

SCHEMA = V6.SCHEMA
SOURCES = V6.SOURCES | {
    "bench/run_agentic_brev_probe_parent_v7.py",
    "tests/test_run_agentic_brev_probe_parent_v7.py",
}
V1 = V6.V1
fresh_selection = V6.V5.fresh_selection


def __getattr__(name: str):
    return getattr(V6, name)
