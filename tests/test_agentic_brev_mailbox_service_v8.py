from __future__ import annotations

import sys
from pathlib import Path

ROOT = next(path for path in Path(__file__).resolve().parents if (path / "AGENTS.md").is_file())
CANDIDATE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(CANDIDATE / "bench"), str(CANDIDATE / "evals")]
sys.path.extend(str(ROOT / name) for name in ("bench", "evals", "src"))

import agentic_brev_mailbox_service_v8 as S  # noqa: E402  # pyright: ignore[reportMissingImports]


def test_service8_selects_remote7_and_restores() -> None:
    before = (S.V5.REMOTE, S.V5.REMOTE_SOURCE, S.V5.SOURCES, S.V5.__file__)
    with S.service8_scope():
        assert S.V5.REMOTE is S.REMOTE
        assert S.V5.REMOTE_SOURCE == S.REMOTE.SOURCE
        assert S.V5.SOURCES is S.SOURCES
        assert S.V5.__file__ == S.__file__
    assert (S.V5.REMOTE, S.V5.REMOTE_SOURCE, S.V5.SOURCES, S.V5.__file__) == before


def test_service8_source_closure_contains_remote7_and_service8() -> None:
    assert {
        "bench/agentic_brev_probe_transport_remote_v7.py",
        "tests/test_agentic_brev_probe_transport_remote_v7.py",
        "evals/agentic_brev_mailbox_service_v8.py",
        "tests/test_agentic_brev_mailbox_service_v8.py",
    } <= S.SOURCES


def test_service8_budget_uses_native_reference_and_restores(monkeypatch) -> None:
    service = object.__new__(S.BrevMailboxService)
    service.runtime = object()
    original = {"lifecycle_protocol": {"path": "bench/protocols/x.json", "sha256": "a" * 64}}
    service.contract = original
    native = {"path": "/repo/bench/protocols/x.json", "sha256": "a" * 64}
    observed = []
    monkeypatch.setattr(S.V6, "_native_lifecycle_reference", lambda runtime, reference: native)
    monkeypatch.setattr(
        S.V5.BrevMailboxService, "_budget",
        lambda self, seconds=0: observed.append((self.contract, seconds)),
    )
    service._budget(15)
    assert observed == [({"lifecycle_protocol": native}, 15)]
    assert service.contract is original
