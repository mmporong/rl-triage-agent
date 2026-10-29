"""Host5 probe receipt를 동일 Runtime의 등록 수리 Host에 연결한다."""
from __future__ import annotations

import inspect
from copy import deepcopy
from pathlib import Path
from typing import Any

import agentic_brev_repair_evidence_v1 as E
import agentic_brev_runtime_host_v2 as H2
import agentic_brev_runtime_host_v5 as H5
import agentic_investigation_runtime_v1 as R

C = E.C
SOURCES = H2.SOURCES | H5.SOURCES | {
    "evals/agentic_brev_repair_host_v1.py",
    "tests/test_agentic_brev_repair_host_v1.py",
}


def _registered_runtime_binding(registered: dict[str, Any], runtime: R.Runtime) -> None:
    """등록 경로 별칭 둘이 같은 원문과 Runtime을 가리키는지 확인한다."""
    resolver = registered["resolver"]
    raw_runtime = registered["raw"]["runtime"]
    registered_protocol = resolver.file(raw_runtime["protocol"])
    runtime_protocol = resolver.file(runtime.protocol_ref)
    registered_repo = resolver.path(raw_runtime["repo"]).resolve(strict=True)
    E.require(
        registered["protocol"] == runtime.protocol
        and registered_protocol == runtime_protocol
        and raw_runtime["protocol"]["sha256"] == runtime.protocol_ref["sha256"]
        and registered_repo == Path(runtime.repo).resolve(strict=True)
        and raw_runtime["anchor"] == runtime.anchor
        and runtime.reg["bands"] == registered["contract"]["bands"],
        "registered aliases must resolve to the same Runtime bytes",
    )


class BrevRepairRuntimeHost(H2.BrevRuntimeHost):
    """Host2 동작을 유지하며 actual probe host를 Host5로 고정한다."""

    def __init__(self, runtime, probe_host, provider, registration_ref, root_map, approver, *,
                 probe_seals, probe_records, admitter, allow_fixture=False):
        E.require(
            isinstance(runtime, R.Runtime)
            and runtime.reg["allow_fixture"] is allow_fixture
            and type(allow_fixture) is bool
            and isinstance(approver, str)
            and bool(approver.strip()),
            "host mode/approver",
        )
        E.require(
            type(probe_host) is H5.BrevRuntimeHost and probe_host.runtime is runtime,
            "exact Host5 from the same Runtime required",
        )
        self.runtime, self.provider = runtime, provider
        self.probe_host: Any = probe_host
        self.registration_ref, self.root_map = deepcopy(registration_ref), deepcopy(root_map)
        self.probe_seals, self.probe_records = deepcopy(probe_seals), deepcopy(probe_records)
        self.approver, self.admitter = approver.strip(), admitter
        self.allow_fixture = allow_fixture
        self.folder = runtime.folder / "brev-repair-host"
        self.owner: int | None = None
        self.artifacts: E.RepairArtifactScope | None = None
        self._sources()

    def _sources(self):
        registered = E.registration(
            self.registration_ref, self.root_map, allow_fixture=self.allow_fixture
        )
        _registered_runtime_binding(registered, self.runtime)
        E.require(
            SOURCES <= registered["protocol"]["sources_lf"].keys(),
            "Host5 repair source closure missing",
        )
        provider = registered["layout"]["provider"]
        for method in (getattr(self.provider, name, None) for name in ("execute", "lookup")):
            if not inspect.ismethod(method):
                raise TypeError("registered provider bound methods required")
            E.require(
                method.__self__ is self.provider
                and method.__func__ is getattr(type(self.provider), method.__name__),
                "registered provider bound methods",
            )
            source = inspect.getsourcefile(method.__func__)
            E.require(
                source is not None
                and C.digest_lf_file(Path(source)) == provider["sha256_lf"]
                and registered["protocol"]["sources_lf"].get(provider["path"])
                == provider["sha256_lf"]
                and type(self.provider).__name__ == provider["class"],
                "provider service source/class binding",
            )
        source = inspect.getsourcefile(self.admitter)
        expected = registered["layout"]["admission_service"]
        E.require(
            source is not None
            and C.digest_lf_file(Path(source)) == expected["sha256_lf"]
            and registered["protocol"]["sources_lf"].get(expected["path"])
            == expected["sha256_lf"],
            "admission service source binding",
        )
        E.require(
            getattr(self.provider, "allow_fixture", None) is self.allow_fixture
            and getattr(self.provider, "registration_ref", None) == self.registration_ref,
            "provider fixture/registration boundary",
        )
        return registered
