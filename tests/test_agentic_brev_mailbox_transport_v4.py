# pyright: reportMissingImports=false
from __future__ import annotations

import importlib.util
import inspect
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import pytest

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "AGENTS.md").is_file() and (p / "tests").is_dir())
for folder in (ROOT / "evals", ROOT / "bench", ROOT / "tests", ROOT / "src"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

SOURCE = Path(__file__).resolve().parents[1] / "evals/agentic_brev_mailbox_transport_v4.py"
SPEC = importlib.util.spec_from_file_location("agentic_brev_mailbox_transport_v4_candidate", SOURCE)
assert SPEC and SPEC.loader
T4 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(T4)

import agentic_brev_runtime_host_v1 as H1
import agentic_brev_runtime_host_v6 as H6
import test_agentic_brev_runtime_host_v6 as H6T
from test_agentic_brev_runtime_host_v1 import setup as host_setup
from test_agentic_investigation_evidence_v1 import case as evidence_case


def _transport(data: dict[str, Any]):
    value = object.__new__(T4.BrevMailboxTransport)
    value.runtime = data["runtime"]
    value.contract_ref = data["contract_ref"]
    value.contract = H1.R.read_ref(data["contract_ref"])
    value.root = data["runtime"].folder / "transport-mailbox"
    value.results = data["archive"]
    value.wait_s = 120
    return value


def _fresh(tmp_path: Path) -> dict[str, Any]:
    fixture = cast(Any, evidence_case).__wrapped__(tmp_path / "evidence")
    data = host_setup(tmp_path / "host", fixture)
    H6T._upgrade_full_fixture(data)
    return data


def _save(path: Path, value: dict) -> dict:
    H1.J._write_new(path, value)
    return H1.R.ref(path)


def test_host6_fresh_registration_prepare_pair_passes(tmp_path: Path) -> None:
    data = _fresh(tmp_path)
    result = _transport(data).prepare_pair(data["operation"], data["intent"], data["binding_ref"])
    assert result == {"status": "prepared", "operation_id": data["operation"]}
    assert H1.R.read_ref(data["binding_ref"])["schema"] == H6.SCHEMA + "_registration"


def test_host6_changed_registration_or_runtime_ref_is_rejected(tmp_path: Path) -> None:
    data = _fresh(tmp_path)
    registration = H1.R.read_ref(data["binding_ref"])
    invalid = _save(tmp_path / "invalid-schema.json", {**registration, "schema": "unknown"})
    with pytest.raises(ValueError, match="Host1 or Host6"):
        _transport(data).prepare_pair(data["operation"], data["intent"], invalid)

    changed = deepcopy(registration)
    binding = H1.R.read_ref(changed["parent_binding"])
    binding["expected_request"]["execution"]["runtime_host_binding"]["operation_id"] = "0" * 64
    changed["parent_binding"] = _save(tmp_path / "changed-binding.json", binding)
    with pytest.raises(ValueError, match="protocol/binding/request mismatch|Runtime operation"):
        _transport(data).prepare_pair(data["operation"], data["intent"],
                                      _save(tmp_path / "changed-registration.json", changed))


def test_legacy_host1_path_and_registered_method_sources_are_unchanged(tmp_path: Path) -> None:
    fixture = cast(Any, evidence_case).__wrapped__(tmp_path / "evidence")
    data = host_setup(tmp_path / "legacy", fixture)
    result = _transport(data).prepare_pair(data["operation"], data["intent"], data["binding_ref"])
    assert result == {"status": "prepared", "operation_id": data["operation"]}
    for name in H1.METHODS:
        assert inspect.getsourcefile(getattr(T4.BrevMailboxTransport, name)) == str(SOURCE)
