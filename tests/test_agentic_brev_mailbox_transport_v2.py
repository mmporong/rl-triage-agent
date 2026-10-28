from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = next(path for path in Path(__file__).resolve().parents if (path / "AGENTS.md").is_file() and (path / "bench").is_dir())
sys.path[:0] = [str(ROOT / "bench"), str(ROOT / "evals"), str(ROOT / "src")]

import agentic_brev_mailbox_transport_v1 as V1  # noqa: E402  # pyright: ignore[reportMissingImports]
import agentic_brev_mailbox_transport_v2 as T  # noqa: E402  # pyright: ignore[reportMissingImports]


def _write(path: Path, value: dict) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, sort_keys=True) + "\n").encode()
    path.write_bytes(raw)
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}


def test_unknown_preserves_bounded_infrastructure_without_status_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    operation, phase = "a" * 64, "reference"
    root = tmp_path / "mailbox"
    response_path = root / operation / phase / "response.json"
    raw_ref = _write(response_path.parent / "service/rpc-prepare-1.json", {"exit_code": 1})
    _write(response_path, {
        "status": "unknown", "error_type": "RemoteInfrastructureFailure",
        "reason": json.dumps({"code": "remote_python_compat_import", "raw_rpc": raw_ref}),
    })
    monkeypatch.setattr(V1.BrevMailboxTransport, "_lookup",
                        lambda self, *args: {"status": "unknown", "operation_id": operation})
    transport = object.__new__(T.BrevMailboxTransport)
    transport.root = root
    result = transport._lookup(phase, operation, {}, {})
    assert result["status"] == "unknown"
    assert result["infrastructure"] == {"code": "remote_python_compat_import", "raw_rpc": raw_ref}


def test_unregistered_infrastructure_code_is_rejected(tmp_path: Path) -> None:
    response_path = tmp_path / "op/reference/response.json"
    raw_ref = _write(response_path.parent / "service/rpc-prepare-1.json", {"exit_code": 1})
    response = {"status": "unknown", "error_type": "RemoteInfrastructureFailure",
                "reason": json.dumps({"code": "anything", "raw_rpc": raw_ref})}
    with pytest.raises(ValueError, match="bounded infrastructure code"):
        T._bounded_infrastructure(response_path, response)
