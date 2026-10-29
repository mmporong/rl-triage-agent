"""Parent6 fresh pair 계약을 독립 원격 CLI에 명시적으로 노출한다."""
from __future__ import annotations

import run_agentic_brev_probe_parent_v6 as V6

SCHEMA = V6.SCHEMA
SOURCE = "bench/run_agentic_brev_probe_parent_v8.py"
TEST_SOURCE = "tests/test_run_agentic_brev_probe_parent_v8.py"
SOURCES = V6.SOURCES | {SOURCE, TEST_SOURCE}

V1 = V6.V1
C = V1.C
E = V1.E
ROOT = V1.ROOT
fresh_selection = V6.V5.fresh_selection


def parent8_scope():
    return V6.fresh_parent_scope()


def prepare(*args, **kwargs):
    with parent8_scope():
        return V6._prepare_impl(*args, **kwargs)


def lookup(*args, **kwargs):
    with parent8_scope():
        return V1.lookup(*args, **kwargs)


def run(*args, **kwargs):
    with parent8_scope():
        return V1.run(*args, **kwargs)


def verify_authority(*args, **kwargs):
    with parent8_scope():
        return V1.verify_authority(*args, **kwargs)


def _verify_completion(*args, **kwargs):
    with parent8_scope():
        return V1._verify_completion(*args, **kwargs)


def main() -> None:
    with parent8_scope():
        V1.main()


def __getattr__(name: str):
    return getattr(V6, name)


if __name__ == "__main__":
    main()
