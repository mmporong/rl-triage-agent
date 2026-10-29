"""Transport4에 등록된 원격 parent ref 투영을 제한적으로 허용한다."""
# pyright: reportMissingImports=false
from __future__ import annotations

import sys
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "AGENTS.md").is_file() and (p / "evals").is_dir())
for folder in (ROOT / "evals", ROOT / "bench", ROOT / "src"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import agentic_brev_mailbox_transport_v4 as V4
import agentic_brev_runtime_host_v6 as H6
import run_agentic_brev_probe_parent_v7 as P7

SCHEMA = V4.SCHEMA
SOURCE = "evals/agentic_brev_mailbox_transport_v5.py"
TEST_SOURCE = "tests/test_agentic_brev_mailbox_transport_v5.py"
H1, R2 = V4.H1, V4.R2


def _remote_ref(local_ref: dict, roots: list[dict]) -> dict:
    local = H1._path(local_ref["path"])
    matches = []
    for row in roots:
        root = H1._path(row["local"])
        if type(local) is type(root) and (local == root or root in local.parents):
            remote = H1._path(row["remote"])
            matches.append(remote.joinpath(*local.relative_to(root).parts))
    H1.require(len(matches) == 1, "unique Host6 local-to-remote binding projection")
    return {"path": str(matches[0]), "sha256": local_ref["sha256"]}


def _protocol_remote(local_ref: dict, runtime_repo: Path, remote_repo: str) -> dict:
    local = Path(local_ref["path"])
    H1.require(local.is_absolute() and local.resolve().is_relative_to(runtime_repo.resolve()),
               "Host6 local parent protocol outside Runtime repo")
    relative = local.resolve().relative_to(runtime_repo.resolve())
    return {"path": str(PurePosixPath(remote_repo).joinpath(*relative.parts)),
            "sha256": local_ref["sha256"]}


class BrevMailboxTransport(V4.BrevMailboxTransport):
    """동일 local ref 또는 등록 service plan이 계산한 remote ref만 받는다."""

    def _base(self, operation: str, intent: dict, registration_ref: dict) -> dict:
        registration = V4.read_ref(registration_ref)
        if registration.get("schema") == H1.SCHEMA + "_registration":
            return super()._base(operation, intent, registration_ref)
        H1.require(registration.get("schema") == H6.SCHEMA + "_registration",
                   "registered Host1 or Host6 parent required")
        protocol = V4.read_ref(registration["parent_protocol"])
        binding = V4.read_ref(registration["parent_binding"])
        request = V4.read_ref(registration["parent_request"])
        same_local = (request.get("protocol") == registration["parent_protocol"]
                      and request.get("binding") == registration["parent_binding"])
        if same_local:
            return super()._base(operation, intent, registration_ref)

        allowed = {"schema", "parent_protocol", "parent_binding", "parent_request",
                   "parent_anchor", "service_plan"}
        H1.require(set(registration) == allowed, "remote Host6 registration fields")
        service_plan = V4.read_ref(registration["service_plan"])
        H1.require(protocol.get("host_transport_service_plan_sha256")
                   == registration["service_plan"]["sha256"],
                   "Host6 service plan SHA registration")
        host = binding["expected_request"]["execution"]["runtime_host_binding"]
        expected_protocol = _protocol_remote(
            registration["parent_protocol"], self.runtime.repo, service_plan["remote_repo"]
        )
        expected_binding = _remote_ref(registration["parent_binding"], host["host_input_roots"])
        H1.require(request.get("protocol") == expected_protocol
                   and request.get("binding") == expected_binding,
                   "Host6 remote parent refs outside registered projections")
        H1.require(
            protocol.get("schema") == P7.SCHEMA + "_protocol"
            and protocol.get("registered_before_execution") is True
            and H6.SOURCES <= protocol.get("sources_lf", {}).keys()
            and protocol.get("binding_sha256") == registration["parent_binding"]["sha256"]
            and request.get("schema") == P7.SCHEMA + "_request"
            and request.get("action") == "probe"
            and request.get("source_anchor") == registration["parent_anchor"],
            "Host6 parent protocol/binding/request mismatch",
        )
        P7.V1._public(protocol)
        R2.verify_sources(
            protocol, registration["parent_anchor"], repo=self.runtime.repo,
            protocol_path=Path(registration["parent_protocol"]["path"]),
        )
        H1._ancestor(self.runtime.repo, self.runtime.anchor, registration["parent_anchor"])
        H1.require(
            host.get("schema") == H6.SCHEMA + "_runtime_binding"
            and host.get("operation_id") == operation
            and host.get("intent_sha256") == H1.J.digest(intent)
            and host.get("runtime_manifest_sha256") == H1.J.digest(self.runtime.manifest)
            and host.get("runtime_source_anchor") == self.runtime.anchor
            and host.get("runtime_protocol_sha256") == self.runtime.protocol_ref["sha256"],
            "Host6 Runtime operation/intent/source binding",
        )
        H1.require(V4.read_ref(self.contract_ref) == self.contract, "mailbox contract changed")
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
    return getattr(V4, name)
