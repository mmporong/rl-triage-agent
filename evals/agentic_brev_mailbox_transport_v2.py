"""Mailbox v1 상태를 유지하며 제한된 인프라 실패 원문을 전달한다."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import agentic_brev_mailbox_transport_v1 as V1

SCHEMA = V1.SCHEMA
SOURCE = "evals/agentic_brev_mailbox_transport_v2.py"
TEST_SOURCE = "tests/test_agentic_brev_mailbox_transport_v2.py"
INFRASTRUCTURE_CODES = frozenset({
    "remote_python_compat_import",
    "remote_helper_exit_nonzero",
    "remote_helper_timeout",
})


def _bounded_infrastructure(response_path: Path, response: dict[str, Any]) -> dict[str, Any] | None:
    if response.get("status") != "unknown" or response.get("error_type") != "RemoteInfrastructureFailure":
        return None
    try:
        value = json.loads(response.get("reason", ""))
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("invalid bounded infrastructure reason") from error
    V1.require(isinstance(value, dict) and set(value) == {"code", "raw_rpc"}
               and value["code"] in INFRASTRUCTURE_CODES, "bounded infrastructure code")
    reference = value["raw_rpc"]
    V1.require(isinstance(reference, dict) and set(reference) == {"path", "sha256"},
               "bounded raw RPC reference")
    raw_path = Path(reference["path"])
    V1.regular(raw_path)
    V1.require(raw_path.parent == response_path.parent / "service"
               and raw_path.name.startswith("rpc-") and raw_path.suffix == ".json",
               "raw RPC reference outside service folder")
    V1.read_ref(reference)
    return {"code": value["code"], "raw_rpc": reference}


class BrevMailboxTransport(V1.BrevMailboxTransport):
    def _lookup(self, phase: str, operation: str, intent: dict, registration_ref: dict) -> dict:
        result = super()._lookup(phase, operation, intent, registration_ref)
        if result["status"] not in {"unknown", "hold"}:
            return result
        response_path = self.root / operation / phase / "response.json"
        if not response_path.is_file():
            return result
        response = V1.E.loads(response_path.read_text(encoding="utf-8"))
        infrastructure = _bounded_infrastructure(response_path, response)
        if infrastructure is not None:
            result["infrastructure"] = infrastructure
        return result


def __getattr__(name: str) -> Any:
    return getattr(V1, name)
