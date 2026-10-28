from __future__ import annotations

import inspect
import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "AGENTS.md").is_file() and (p / "bench").is_dir())
sys.path[:0] = [str(ROOT / p) for p in ("evals", "bench", "tests", "src")]

import agentic_brev_mailbox_transport_v3 as T  # pyright: ignore[reportMissingImports]
import agentic_brev_runtime_host_v1 as H  # pyright: ignore[reportMissingImports]
from test_agentic_brev_runtime_host_v1 import setup  # pyright: ignore[reportMissingImports]
from test_agentic_investigation_evidence_v1 import case  # noqa: F401  # pyright: ignore[reportMissingImports]


def test_registered_methods_are_bound_to_transport3_source() -> None:
    source = inspect.getsourcefile(T.BrevMailboxTransport)
    assert source == str(ROOT / T.SOURCE)
    for name in H.METHODS:
        method = getattr(T.BrevMailboxTransport, name)
        assert inspect.getsourcefile(method) == source


def test_frozen_host_sources_accept_transport3_fixture(case, tmp_path, monkeypatch) -> None:  # noqa: F811
    monkeypatch.setattr(H, "SOURCES", H.SOURCES | {T.SOURCE, T.TEST_SOURCE})
    monkeypatch.setattr(T.BrevMailboxTransport, "EVIDENCE_KIND", H.E.FIXTURE)
    data = setup(tmp_path, case)
    transport = object.__new__(T.BrevMailboxTransport)
    transport.runtime = data["runtime"]
    transport.contract_ref = data["contract_ref"]
    transport.contract = H.R.read_ref(data["contract_ref"])
    transport.root = data["runtime"].folder / "transport-mailbox"
    transport.results = data["archive"]
    transport.wait_s = 120
    host = H.BrevRuntimeHost(
        data["runtime"], {data["operation"]: data["binding_ref"]}, transport,
        "cpu_operator", allow_fixture=True,
    )
    host._sources()
