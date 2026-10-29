from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
CANDIDATE = HERE.parent
ROOT = next(path for path in HERE.parents if (path / "AGENTS.md").is_file())
sys.path[:0] = [str(CANDIDATE / "bench"), str(CANDIDATE / "evals"),
                str(ROOT / "evals"), str(ROOT / "bench"), str(ROOT / "src")]

import agentic_brev_probe_transport_remote_v9 as remote  # pyright: ignore[reportMissingImports]


def test_remote9_scope_selects_parent9_and_restores_remote8() -> None:
    before = (remote.V8.PARENT, remote.V8.SOURCES,
              remote.V8.PROVIDER_SCHEMA, remote.V8.PRODUCTION)
    with remote.parent9_scope():
        assert remote.V8.PARENT is remote.PARENT
        assert remote.V8.SOURCES == remote.SOURCES
        assert remote.V8.PROVIDER_SCHEMA == remote.PROVIDER_SCHEMA
        assert remote.V8.PRODUCTION == remote.PRODUCTION
    assert (remote.V8.PARENT, remote.V8.SOURCES,
            remote.V8.PROVIDER_SCHEMA, remote.V8.PRODUCTION) == before


def test_remote9_closure_contains_parent9_and_service10() -> None:
    assert remote.PARENT.SOURCES <= remote.SOURCES
    assert remote.SOURCE in remote.SOURCES
    assert remote.TEST_SOURCE in remote.SOURCES
    assert remote.SERVICE_SOURCE in remote.SOURCES
    assert remote.SERVICE_TEST_SOURCE in remote.SOURCES


def test_remote9_default_preparer_is_parent9_and_explicit_value_is_preserved(monkeypatch) -> None:
    calls = []

    def fake_prepare(*args, **kwargs):
        return {"fake": True}

    original = remote.PRODUCTION
    monkeypatch.setattr(remote, "PRODUCTION", (fake_prepare, original[1], original[2]))

    def fake_internal_prepare(*args, **kwargs):
        calls.append(kwargs["preparer"])
        return {"prepared": True}

    monkeypatch.setattr(remote.V8.os, "name", "posix")
    monkeypatch.setattr(remote.V8, "_prepare", fake_internal_prepare)
    monkeypatch.setattr(remote.V8.V1, "_public_prepare",
                        lambda prepared, request, request_sha256: {"status": "prepared"})
    assert remote.run_transport(request={}, request_sha256="0" * 64) == {
        "status": "prepared", "schema": remote.SCHEMA}
    assert calls == [fake_prepare]

    explicit = object()
    try:
        remote.run_transport(request={}, request_sha256="0" * 64, preparer=explicit)
    except ValueError as error:
        assert str(error) == "actual transport dependency injection forbidden"
    else:
        raise AssertionError("explicit unregistered preparer was overwritten")
    assert calls == [fake_prepare]


def test_remote9_cli_default_never_uses_parent8_preparer() -> None:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    script = CANDIDATE / "bench/agentic_brev_probe_transport_remote_v9.py"
    with tempfile.TemporaryDirectory() as raw:
        missing = Path(raw) / "missing-request.json"
        result = subprocess.run(
            [sys.executable, "-B", str(script), "--request", str(missing),
             "--request-sha256", "0" * 64, "--mode", "prepare"],
            cwd=raw, env=env, text=True, encoding="utf-8", errors="replace",
            capture_output=True, check=False,
        )
    assert result.returncode != 0
    assert "actual transport dependency injection forbidden" not in result.stderr
    if os.name == "posix":
        assert "FileNotFoundError" in result.stderr

