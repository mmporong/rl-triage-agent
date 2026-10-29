"""Transport3에 Host6 fresh registration의 immutable binding을 추가한다."""
# pyright: reportMissingImports=false
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "AGENTS.md").is_file() and (p / "evals").is_dir())
for folder in (ROOT / "evals", ROOT / "bench", ROOT / "src"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import agentic_brev_mailbox_transport_v3 as V3
import agentic_brev_runtime_host_v6 as H6
import run_agentic_brev_probe_parent_v7 as P7

SCHEMA = V3.SCHEMA
SOURCE = "evals/agentic_brev_mailbox_transport_v4.py"
TEST_SOURCE = "tests/test_agentic_brev_mailbox_transport_v4.py"

H1 = H6.V5.V4.V3.V1
R2 = H6.V5.V4.V3.R2


class BrevMailboxTransport(V3.BrevMailboxTransport):
    """Legacy Host1은 보존하고 Host6 registration만 별도 검증한다."""

    def _base(self, operation: str, intent: dict, registration_ref: dict) -> dict:
        registration = V3.read_ref(registration_ref)
        if registration.get("schema") == H1.SCHEMA + "_registration":
            return super()._base(operation, intent, registration_ref)
        H1.require(registration.get("schema") == H6.SCHEMA + "_registration",
                   "registered Host1 or Host6 parent required")
        allowed = {"schema", "parent_protocol", "parent_binding", "parent_request", "parent_anchor"}
        H1.require(set(registration) in (allowed, allowed | {"service_plan"}),
                   "Host6 registration fields")
        for name in ("parent_protocol", "parent_binding", "parent_request"):
            H1.require(set(registration[name]) == {"path", "sha256"},
                       f"Host6 {name} raw reference")
        protocol = V3.read_ref(registration["parent_protocol"])
        binding = V3.read_ref(registration["parent_binding"])
        request = V3.read_ref(registration["parent_request"])
        if "service_plan" in registration:
            H1.require(set(registration["service_plan"]) == {"path", "sha256"},
                       "Host6 service plan raw reference")
            V3.read_ref(registration["service_plan"])
        H1.require(
            protocol.get("schema") == P7.SCHEMA + "_protocol"
            and protocol.get("registered_before_execution") is True
            and H6.SOURCES <= protocol.get("sources_lf", {}).keys()
            and protocol.get("binding_sha256") == registration["parent_binding"]["sha256"]
            and request.get("schema") == P7.SCHEMA + "_request"
            and request.get("action") == "probe"
            and request.get("protocol") == registration["parent_protocol"]
            and request.get("binding") == registration["parent_binding"]
            and request.get("source_anchor") == registration["parent_anchor"],
            "Host6 parent protocol/binding/request mismatch",
        )
        P7.V1._public(protocol)
        R2.verify_sources(
            protocol, registration["parent_anchor"], repo=self.runtime.repo,
            protocol_path=Path(registration["parent_protocol"]["path"]),
        )
        H1._ancestor(self.runtime.repo, self.runtime.anchor, registration["parent_anchor"])
        host = binding["expected_request"]["execution"]["runtime_host_binding"]
        H1.require(
            host.get("schema") == H6.SCHEMA + "_runtime_binding"
            and host.get("operation_id") == operation
            and host.get("intent_sha256") == H1.J.digest(intent)
            and host.get("runtime_manifest_sha256") == H1.J.digest(self.runtime.manifest)
            and host.get("runtime_source_anchor") == self.runtime.anchor
            and host.get("runtime_protocol_sha256") == self.runtime.protocol_ref["sha256"],
            "Host6 Runtime operation/intent/source binding",
        )
        H1.require(V3.read_ref(self.contract_ref) == self.contract, "mailbox contract changed")
        H1.require(intent.get("action") == "run_probe", "repair transport unsupported")
        return {
            "schema": SCHEMA + "_request", "operation_id": operation,
            "intent_sha256": H1.E.digest(intent), "registration": registration_ref,
            "contract_sha256": self.contract_ref["sha256"],
        }

    def prepare_pair(self, operation_id: str, intent: dict, registration_ref: dict) -> dict:
        return super().prepare_pair(operation_id, intent, registration_ref)

    def start_reference(self, operation_id: str, intent: dict, registration_ref: dict) -> dict:
        return super().start_reference(operation_id, intent, registration_ref)

    def lookup_reference(self, operation_id: str, intent: dict, registration_ref: dict) -> dict:
        return super().lookup_reference(operation_id, intent, registration_ref)

    def dispatch_case(
        self, operation_id: str, intent: dict, registration_ref: dict, authorization: dict,
    ) -> dict:
        return super().dispatch_case(operation_id, intent, registration_ref, authorization)

    def lookup_pair(self, operation_id: str, intent: dict, registration_ref: dict) -> dict:
        return super().lookup_pair(operation_id, intent, registration_ref)


def __getattr__(name: str) -> Any:
    return getattr(V3, name)
