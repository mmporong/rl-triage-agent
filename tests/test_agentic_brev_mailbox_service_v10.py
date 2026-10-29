from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CANDIDATE = HERE.parent
ROOT = next(path for path in HERE.parents if (path / "AGENTS.md").is_file())
sys.path[:0] = [str(CANDIDATE / "bench"), str(CANDIDATE / "evals"),
                str(ROOT / "evals"), str(ROOT / "bench"), str(ROOT / "src")]

import agentic_brev_mailbox_service_v10 as service  # pyright: ignore[reportMissingImports]


def test_service10_scope_selects_remote9_and_restores_service9() -> None:
    before = (service.V9.REMOTE, service.V9.REMOTE_SOURCE,
              service.V9.SOURCES, service.V9.__file__, service.V9._BASE)
    with service.service10_scope():
        assert service.V9.REMOTE is service.REMOTE
        assert service.V9.REMOTE_SOURCE == service.REMOTE_SOURCE
        assert service.V9.SOURCES == service.SOURCES
        assert service.V9._BASE == (
            service.V9.V8.REMOTE, service.V9.V8.REMOTE_SOURCE,
            service.V9.V8.SOURCES, service.V9.V8.__file__,
        )
    assert (service.V9.REMOTE, service.V9.REMOTE_SOURCE,
            service.V9.SOURCES, service.V9.__file__, service.V9._BASE) == before


def test_service10_closure_contains_remote9() -> None:
    assert service.REMOTE.SOURCES <= service.SOURCES
    assert service.SOURCE in service.SOURCES
    assert service.TEST_SOURCE in service.SOURCES
