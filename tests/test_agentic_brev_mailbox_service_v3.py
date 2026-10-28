"""합성 CPU service로 Remote3 argv·자기 source scope·v2 metadata 해석 보존을 검증한다."""
from __future__ import annotations

import base64
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("bench", "evals", "src", "tests")]
import agentic_brev_mailbox_service_v3 as S


def test_service_scope_is_reentrant_and_restores_v2_globals() -> None:
    original = (S.V2.REMOTE, S.V2.REMOTE_SOURCE, S.V2.SOURCES, S.V2.__file__)
    with S.remote3_scope():
        assert S.V2.REMOTE is S.REMOTE and S.V2.REMOTE_SOURCE == S.REMOTE_SOURCE
        assert Path(S.V2.__file__).name == "agentic_brev_mailbox_service_v3.py"
        with S.remote3_scope():
            assert S._STATE["depth"] == 2
    assert (S.V2.REMOTE, S.V2.REMOTE_SOURCE, S.V2.SOURCES, S.V2.__file__) == original


def test_service_rpc_uses_only_remote3_entry_and_keeps_v2_lifecycle(tmp_path, monkeypatch) -> None:
    assert S.BrevMailboxService._budget is S.V2.V1.BrevMailboxService._budget
    assert S.BrevMailboxService._call is S.V2.V1.BrevMailboxService._call
    service = object.__new__(S.BrevMailboxService)
    service.contract = {"target": {"resource_name": "cpu-vm"},
                        "host_mount_projection": {"host": str(tmp_path), "wsl": "/cpu"}}
    monkeypatch.setattr(service, "_inventory", lambda _folder: None)

    def call(folder, _label, _args):
        text = next(folder.glob("rpc-prepare-*.sh")).read_text(encoding="utf-8")
        encoded = re.search(r"base64.b64decode\('([^']+)'\)", text)
        assert encoded is not None
        argv = json.loads(base64.b64decode(encoded.group(1)))
        assert argv[2] == "/repo/bench/agentic_brev_probe_transport_remote_v3.py"
        result = {"schema": S.REMOTE.SCHEMA, "status": "prepared", "authority_issued": False}
        envelope = {"exit_code": 0, "stdout_base64": base64.b64encode(json.dumps(result).encode()).decode(),
                    "stderr_base64": ""}
        output = folder / "rpc.stdout"
        output.write_text(S.V2.V1.MARKER + base64.b64encode(json.dumps(envelope).encode()).decode() + "\ncpu-vm\n",
                          encoding="utf-8")
        return {"stdout": str(output)}

    monkeypatch.setattr(service, "_call", call)
    result = service._rpc(tmp_path, {"remote_repo": "/repo", "remote_python": "/venv/bin/python",
                                     "remote_request": "/request.json"},
                          {"parent_request": {"sha256": "a" * 64}}, "prepare")
    assert result["schema"] == S.REMOTE.SCHEMA


def test_remote3_preserves_v2_archive_schema_and_old_metadata_helpers() -> None:
    assert S.REMOTE.ARCHIVE_SCHEMA == S.V2.REMOTE.ARCHIVE_SCHEMA
    assert S.REMOTE.locate is S.V2.REMOTE.locate
    assert S.REMOTE.maps_for_host is S.V2.REMOTE.maps_for_host
    assert {S.SOURCE, S.TEST_SOURCE, S.REMOTE.SOURCE, S.REMOTE.TEST_SOURCE} <= S.SOURCES


def test_same_name_foreign_remote_module_is_refused(monkeypatch) -> None:
    monkeypatch.setattr(S.V2, "REMOTE", SimpleNamespace(__name__=S._BASE_REMOTE.__name__))
    with pytest.raises(RuntimeError, match="이미 교체"), S.remote3_scope():
        pass

