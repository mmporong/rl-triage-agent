from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CANDIDATE = HERE.parent
ROOT = next(path for path in HERE.parents if (path / "AGENTS.md").is_file())
sys.path[:0] = [str(CANDIDATE / "bench"), str(CANDIDATE / "evals"),
                str(ROOT / "evals"), str(ROOT / "bench"), str(ROOT / "src")]

import agentic_brev_probe_transport_remote_v10 as remote


def test_remote10_scope_selects_parent10_and_restores_remote9() -> None:
    before = (remote.V9.PARENT, remote.V9.SOURCES, remote.V9.PROVIDER_SCHEMA, remote.V9.PRODUCTION)
    with remote.parent10_scope():
        assert remote.V9.PARENT is remote.PARENT
        assert remote.V9.SOURCES == remote.SOURCES
        assert remote.V9.PROVIDER_SCHEMA == remote.PROVIDER_SCHEMA
        assert remote.V9.PRODUCTION == remote.PRODUCTION
    assert (remote.V9.PARENT, remote.V9.SOURCES,
            remote.V9.PROVIDER_SCHEMA, remote.V9.PRODUCTION) == before


def test_remote10_production_preparer_and_source_closure(monkeypatch) -> None:
    assert remote.PRODUCTION[0] is remote.PARENT.prepare
    assert remote.PARENT.SOURCES <= remote.SOURCES
    assert {remote.SOURCE, remote.TEST_SOURCE,
            remote.SERVICE_SOURCE, remote.SERVICE_TEST_SOURCE} <= remote.SOURCES
    observed = []

    def fake_run(*args, **kwargs):
        observed.append((args, kwargs, remote.V9.PARENT, remote.V9.PRODUCTION))
        return {"status": "prepared"}

    monkeypatch.setattr(remote.V9, "run_transport", fake_run)
    assert remote.run_transport(request={}, request_sha256="0" * 64) == {
        "status": "prepared", "schema": remote.SCHEMA,
    }
    assert observed[0][1]["preparer"] is remote.PRODUCTION[0]
    assert observed[0][2] is remote.PARENT and observed[0][3] == remote.PRODUCTION
