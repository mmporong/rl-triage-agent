# pyright: reportMissingImports=false
from __future__ import annotations

import importlib.util
import inspect
import sys
from copy import deepcopy
from pathlib import Path, PurePosixPath
from typing import Any, cast

import pytest

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "AGENTS.md").is_file() and (p / "tests").is_dir())
for folder in (ROOT / "evals", ROOT / "bench", ROOT / "tests", ROOT / "src"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

SOURCE = Path(__file__).resolve().parents[1] / "evals/agentic_brev_mailbox_transport_v5.py"
SPEC = importlib.util.spec_from_file_location("agentic_brev_mailbox_transport_v5_candidate", SOURCE)
assert SPEC and SPEC.loader
T5 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(T5)

import agentic_brev_runtime_host_v1 as H1
import test_agentic_brev_runtime_host_v6 as H6T
from test_agentic_brev_runtime_host_v1 import setup as host_setup
from test_agentic_investigation_evidence_v1 import case as evidence_case


def _save(path: Path, value: dict) -> dict:
    H1.J._write_new(path, value)
    return H1.R.ref(path)


def _transport(data: dict[str, Any]):
    value = object.__new__(T5.BrevMailboxTransport)
    value.runtime, value.contract_ref = data["runtime"], data["contract_ref"]
    value.contract = H1.R.read_ref(data["contract_ref"])
    value.root, value.results, value.wait_s = data["runtime"].folder / "transport-mailbox", data["archive"], 120
    return value


def _fresh(tmp_path: Path) -> dict[str, Any]:
    fixture = cast(Any, evidence_case).__wrapped__(tmp_path / "evidence")
    data = host_setup(tmp_path / "host", fixture)
    H6T._upgrade_full_fixture(data)
    return data


def _remote_registration(data: dict[str, Any], root: Path) -> tuple[dict, dict]:
    registration = H1.R.read_ref(data["binding_ref"])
    binding = H1.R.read_ref(registration["parent_binding"])
    host = binding["expected_request"]["execution"]["runtime_host_binding"]
    binding_path = root / "binding.json"
    host["host_input_roots"] = [{"local": str(binding_path.parent), "remote": "/remote/registration"}]
    binding_ref = _save(binding_path, binding)
    service_ref = _save(root / "service-plan.json", {"remote_repo": "/remote/repo"})
    protocol = H1.R.read_ref(registration["parent_protocol"])
    protocol["binding_sha256"] = binding_ref["sha256"]
    protocol["host_transport_service_plan_sha256"] = service_ref["sha256"]
    relative = Path(registration["parent_protocol"]["path"]).relative_to(data["runtime"].repo)
    protocol_path = data["runtime"].repo / relative
    files = {name: (ROOT / name).read_bytes() for name in protocol["sources_lf"] if (ROOT / name).is_file()}
    files[relative.as_posix()] = H1.J._bytes(protocol)
    parent_anchor = H6T.commit_objects(data["runtime"].repo, files, registration["parent_anchor"])
    protocol_ref = H1.R.ref(protocol_path)
    request = H1.R.read_ref(registration["parent_request"])
    request.update(
        protocol={"path": str(PurePosixPath("/remote/repo").joinpath(*relative.parts)),
                  "sha256": protocol_ref["sha256"]},
        binding={"path": "/remote/registration/binding.json", "sha256": binding_ref["sha256"]},
        source_anchor=parent_anchor,
    )
    request_ref = _save(root / "request.json", request)
    remote = {
        "schema": registration["schema"], "parent_protocol": protocol_ref,
        "parent_binding": binding_ref, "parent_request": request_ref,
        "parent_anchor": parent_anchor, "service_plan": service_ref,
    }
    return remote, request


def test_same_path_fresh_and_legacy_paths_are_unchanged(tmp_path: Path) -> None:
    fresh = _fresh(tmp_path / "fresh")
    assert _transport(fresh).prepare_pair(fresh["operation"], fresh["intent"], fresh["binding_ref"])["status"] == "prepared"
    fixture = cast(Any, evidence_case).__wrapped__(tmp_path / "legacy-evidence")
    legacy = host_setup(tmp_path / "legacy", fixture)
    assert _transport(legacy).prepare_pair(legacy["operation"], legacy["intent"], legacy["binding_ref"])["status"] == "prepared"
    assert all(inspect.getsourcefile(getattr(T5.BrevMailboxTransport, name)) == str(SOURCE) for name in H1.METHODS)


def test_registered_remote_protocol_and_binding_paths_pass(tmp_path: Path) -> None:
    data = _fresh(tmp_path / "fresh")
    registration, _request = _remote_registration(data, tmp_path / "remote")
    ref = _save(tmp_path / "remote-registration.json", registration)
    assert _transport(data).prepare_pair(data["operation"], data["intent"], ref)["status"] == "prepared"


def test_arbitrary_remote_path_or_service_plan_sha_is_rejected(tmp_path: Path) -> None:
    data = _fresh(tmp_path / "fresh")
    registration, request = _remote_registration(data, tmp_path / "remote")
    bad_request = deepcopy(request)
    bad_request["binding"]["path"] = "/remote/other/binding.json"
    bad_path = deepcopy(registration)
    bad_path["parent_request"] = _save(tmp_path / "bad-request.json", bad_request)
    with pytest.raises(ValueError, match="outside registered projections"):
        _transport(data).prepare_pair(data["operation"], data["intent"],
                                      _save(tmp_path / "bad-path-registration.json", bad_path))
    bad_plan = deepcopy(registration)
    bad_plan["service_plan"] = {**registration["service_plan"], "sha256": "0" * 64}
    with pytest.raises(ValueError, match="SHA changed"):
        _transport(data).prepare_pair(data["operation"], data["intent"],
                                      _save(tmp_path / "bad-plan-registration.json", bad_plan))
