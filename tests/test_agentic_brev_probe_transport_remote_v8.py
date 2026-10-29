from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("bench", "evals", "src")]

import agentic_brev_probe_transport_remote_v8 as R8  # pyright: ignore[reportMissingImports]


def test_parent8_scope_changes_parent_provider_schema_together_and_restores() -> None:
    before = (R8.V1.P, R8.V2.P, R8.V1.PRODUCTION, R8.V2.PRODUCTION,
              R8.V1.PROVIDER_SCHEMA)
    with R8.parent8_scope():
        assert R8.V1.P is R8.PARENT
        assert R8.V2.P is R8.PARENT
        assert R8.V1.PROVIDER_SCHEMA == R8.PARENT.SCHEMA + "_provider_packet"
        assert R8.V1.PRODUCTION is R8.PRODUCTION
    assert (R8.V1.P, R8.V2.P, R8.V1.PRODUCTION, R8.V2.PRODUCTION,
            R8.V1.PROVIDER_SCHEMA) == before


def test_start_argv_executes_parent8(monkeypatch, tmp_path: Path) -> None:
    request = tmp_path / "request.json"
    request.write_text("{}", encoding="utf-8")
    prepared = {"transport_folder": tmp_path / "transport", "python": Path(sys.executable)}
    calls = []

    class Process:
        pid = 17

    identity = {"pid": 17, "pgid": 17, "sid": 17, "start_ticks": 1,
                "posix_group_verified": True}

    def popen(argv, **kwargs):
        calls.append((argv, kwargs))
        return Process()

    with R8.parent8_scope():
        result = R8.V1._start(prepared, request, "a" * 64, popen, lambda _pid: identity)
    assert result["status"] == "running"
    assert calls[0][0][2] == str((ROOT / R8.PARENT.SOURCE).resolve())


def test_provider_schema_is_derived_from_same_executable_parent() -> None:
    assert R8.PROVIDER_SCHEMA == R8.PARENT.SCHEMA + "_provider_packet"
    assert R8.PARENT.SOURCE in R8.SOURCES
