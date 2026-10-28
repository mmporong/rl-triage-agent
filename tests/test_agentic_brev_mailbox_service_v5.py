from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

ROOT = next(path for path in Path(__file__).resolve().parents if (path / "AGENTS.md").is_file() and (path / "bench").is_dir())
sys.path[:0] = [str(ROOT / "bench"), str(ROOT / "evals"), str(ROOT / "src")]

import agentic_brev_mailbox_service_v5 as S  # noqa: E402  # pyright: ignore[reportMissingImports]
import agentic_brev_probe_transport_remote_v5 as R  # noqa: E402  # pyright: ignore[reportMissingImports]


def test_service5_scope_uses_remote5_and_restores() -> None:
    original = (S.V3.REMOTE, S.V3.REMOTE_SOURCE, S.V3.SOURCES, S.V3.__file__)
    with S.remote5_scope():
        assert S.V3.REMOTE is R
        assert S.V3.REMOTE_SOURCE == R.SOURCE
    assert (S.V3.REMOTE, S.V3.REMOTE_SOURCE, S.V3.SOURCES, S.V3.__file__) == original


def test_python_compat_import_has_bounded_code() -> None:
    raw = {
        "exit_code": 1,
        "stdout_base64": "",
        "stderr_base64": base64.b64encode(
            b"ImportError: cannot import name 'UTC' from 'datetime' (/usr/lib/python3.10/datetime.py)"
        ).decode(),
    }
    assert S._failure_code(raw) == "remote_python_compat_import"
    error = S.RemoteInfrastructureFailure(
        "remote_python_compat_import", {"path": "/tmp/rpc.json", "sha256": "a" * 64}
    )
    assert json.loads(str(error))["code"] == "remote_python_compat_import"


def test_other_exit_and_timeout_remain_bounded() -> None:
    assert S._failure_code({"timed_out": True}) == "remote_helper_timeout"
    raw = {"exit_code": 2, "stdout_base64": "", "stderr_base64": base64.b64encode(b"x").decode()}
    assert S._failure_code(raw) == "remote_helper_exit_nonzero"
