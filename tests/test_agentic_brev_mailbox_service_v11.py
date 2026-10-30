from __future__ import annotations

import inspect
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CANDIDATE = HERE.parent
ROOT = next(path for path in HERE.parents if (path / "AGENTS.md").is_file())
sys.path[:0] = [str(CANDIDATE / "bench"), str(CANDIDATE / "evals"),
                str(ROOT / "evals"), str(ROOT / "bench"), str(ROOT / "src")]

import agentic_brev_mailbox_service_v11 as service


def test_service11_scope_selects_remote10_and_restores_service10() -> None:
    before = (service.V10.REMOTE, service.V10.REMOTE_SOURCE,
              service.V10.SOURCES, service.V10.__file__, service.V10._BASE)
    with service.service11_scope():
        assert service.V10.REMOTE is service.REMOTE
        assert service.V10.REMOTE_SOURCE == service.REMOTE_SOURCE
        assert service.V10.SOURCES == service.SOURCES
        assert service.V10._BASE == (
            service.V10.V9.REMOTE, service.V10.V9.REMOTE_SOURCE,
            service.V10.V9.SOURCES, service.V10.V9.__file__,
        )
    assert (service.V10.REMOTE, service.V10.REMOTE_SOURCE,
            service.V10.SOURCES, service.V10.__file__, service.V10._BASE) == before


def test_service11_methods_and_host6_closure_contract() -> None:
    assert service.REMOTE.SOURCES <= service.SOURCES
    assert {service.SOURCE, service.TEST_SOURCE} <= service.SOURCES
    source = Path(service.__file__).resolve()
    for name in ("_budget", "_validate_host_intent", "_provider", "_rpc", "_registered", "serve_once"):
        assert Path(inspect.getsourcefile(getattr(service.BrevMailboxService, name)) or "").resolve() == source
    host6 = service.V10.V9.H6
    assert host6.SOURCES <= service.SOURCES
