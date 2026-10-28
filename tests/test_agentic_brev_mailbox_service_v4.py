from __future__ import annotations

import sys
from pathlib import Path

ROOT = next(path for path in Path(__file__).resolve().parents if (path / "AGENTS.md").is_file() and (path / "bench").is_dir())
sys.path[:0] = [str(ROOT / "bench"), str(ROOT / "evals"), str(ROOT / "src")]

import agentic_brev_mailbox_service_v4 as S  # pyright: ignore[reportMissingImports]
import agentic_brev_probe_transport_remote_v4 as R  # pyright: ignore[reportMissingImports]


def test_service4_scope_uses_remote4_and_restores() -> None:
    original = (S.V3.REMOTE, S.V3.REMOTE_SOURCE, S.V3.SOURCES, S.V3.__file__)
    with S.remote4_scope():
        assert S.V3.REMOTE is R
        assert S.V3.REMOTE_SOURCE == R.SOURCE
    assert (S.V3.REMOTE, S.V3.REMOTE_SOURCE, S.V3.SOURCES, S.V3.__file__) == original
