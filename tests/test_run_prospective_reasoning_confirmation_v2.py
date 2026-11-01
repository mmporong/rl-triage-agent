# pyright: reportMissingImports=false
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from evals import run_prospective_reasoning_confirmation_v2 as V


BASE = V.ROOT / "bench/protocols/prospective_reasoning_confirmation_execution_20261101_v1.json"


def _lease(identifier: str = "0caa81b0-f9be-4b37-8f89-339cf007a804") -> dict:
    return {"owner": V.R.SOURCE.OWNER, "leaseId": identifier, "wrapperPid": 1,
            "childPid": 2, "acquiredUtc": "2026-11-01T00:23:19Z",
            "heartbeatUtc": "2026-11-01T00:23:20Z", "completedUtc": None,
            "state": "running", "exitCode": None}


def _install_raw(monkeypatch: pytest.MonkeyPatch, values: list[dict]):
    queue = [deepcopy(value) for value in values]

    def raw(_evidence: Path | None = None):
        return queue.pop(0) if len(queue) > 1 else deepcopy(queue[0])

    monkeypatch.setattr(V, "_REGISTERED_LEASE_CONTEXT", raw)
    monkeypatch.setattr(V.R.SOURCE.LEASE, "lease_context", raw)
    return raw


def test_running_lease_adds_nonempty_alias_without_mutating_raw() -> None:
    value = _lease()
    before = deepcopy(value)
    normalized = V.normalize_lease_metadata(value)
    assert normalized["lease_id"] == normalized["leaseId"]
    assert normalized["lease_id"] is not None
    assert value == before and "lease_id" not in value


@pytest.mark.parametrize("mutate", [
    lambda value: value.pop("leaseId"),
    lambda value: value.update(leaseId=""),
    lambda value: value.update(owner="other"),
    lambda value: value.update(state="stopped"),
    lambda value: value.update(completedUtc="2026-11-01T01:00:00Z"),
])
def test_normalizer_rejects_missing_or_unknown_identity(mutate) -> None:
    value = _lease()
    mutate(value)
    with pytest.raises(ValueError):
        V.normalize_lease_metadata(value)


def test_two_key_conflict_is_rejected_and_equal_alias_is_allowed() -> None:
    value = _lease()
    value["lease_id"] = value["leaseId"]
    assert V.normalize_lease_metadata(value)["lease_id"] == value["leaseId"]
    value["lease_id"] = "1a78ae9d-d11f-4abe-b20f-f9b884f7c6ce"
    with pytest.raises(ValueError, match="충돌"):
        V.normalize_lease_metadata(value)


def test_scope_normalizes_callback_reads_and_restores_after_exception(
        monkeypatch: pytest.MonkeyPatch) -> None:
    raw = _install_raw(monkeypatch, [_lease()])
    with pytest.raises(RuntimeError, match="boom"):
        with V.lease_adapter_scope():
            assert V.R.SOURCE.LEASE.lease_context()["lease_id"] == _lease()["leaseId"]
            raise RuntimeError("boom")
    assert V.R.SOURCE.LEASE.lease_context is raw


def test_scope_rejects_nested_entry_but_keeps_internal_callback_reads(
        monkeypatch: pytest.MonkeyPatch) -> None:
    raw = _install_raw(monkeypatch, [_lease()])
    with V.lease_adapter_scope():
        assert V.R.SOURCE.LEASE.lease_context()["lease_id"] == _lease()["leaseId"]
        assert V.R.SOURCE.LEASE.lease_context()["lease_id"] == _lease()["leaseId"]
        with pytest.raises(ValueError, match="재진입"):
            with V.lease_adapter_scope():
                pass
    assert V.R.SOURCE.LEASE.lease_context is raw


def test_uuid_change_reaches_v1_style_verify_context_and_is_rejected(
        monkeypatch: pytest.MonkeyPatch) -> None:
    first = _lease()
    second = _lease("1a78ae9d-d11f-4abe-b20f-f9b884f7c6ce")
    raw = _install_raw(monkeypatch, [first, second])
    with V.lease_adapter_scope():
        initial = V.R.SOURCE.LEASE.lease_context()["lease_id"]
        current = V.R.SOURCE.LEASE.lease_context()["lease_id"]
        with pytest.raises(ValueError, match="identity 변경"):
            V.R.require(current == initial, "실행 중 lease identity 변경")
    assert V.R.SOURCE.LEASE.lease_context is raw


def test_prepare_protocol_preserves_all_v1_policy_and_budget_contracts() -> None:
    value = V.prepare_protocol(BASE)
    assert value["base_execution_anchor"] == V.STAGE_B_ANCHOR
    assert value["base_design_anchor"] == V.STAGE_A_ANCHOR
    assert value["inherited"] == {
        "policies": ["legacy_none", "enabled_low", "strong_registered_rule"],
        "planned_rows": 24, "fault_rows": 18, "normal_host_gate_rows": 6,
        "api_requests": 96, "paired_probes": 54,
        "sdk_role_executions_upper_bound": 108,
        "claims": {"new_fault_family_generalization": False, "llm_superiority": False,
                   "default_mode_adoption": False, "repair": False, "recovery": False}}
    assert value["change"]["policy_or_prompt_changed"] is False
    assert value["change"]["threshold_or_case_changed"] is False
    assert set(V.LOCAL_SOURCE_FILES) <= set(value["source_sha256_lf"])
    assert "evals/run_prospective_reasoning_confirmation_v1.py" in value["source_sha256_lf"]


def test_execute_passes_original_v1_protocol_and_stage_b_anchor_inside_scope(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw = _install_raw(monkeypatch, [_lease()])
    wrapper = V.prepare_protocol(BASE)
    protocol = tmp_path / "outer.json"
    protocol.write_text(json.dumps(wrapper), encoding="utf-8")
    monkeypatch.setattr(V, "verify_protocol", lambda _path, _anchor: deepcopy(wrapper))
    seen = []

    def base_execute(*args):
        seen.append(args)
        lease = V.R.SOURCE.LEASE.lease_context()
        assert lease["lease_id"] == lease["leaseId"]
        return {"schema": V.R.RESULT_SCHEMA, "status": "complete"}

    placeholders = [tmp_path / name for name in
                    ("design", "bridge", "source", "corpus", "lab", "bridge-root", "out")]
    result = V.execute(protocol, "c" * 40, BASE, *placeholders, base_execute=base_execute)
    assert result["status"] == "complete"
    assert seen[0][0] == BASE.resolve() and seen[0][1] == V.STAGE_B_ANCHOR
    assert V.R.SOURCE.LEASE.lease_context is raw


def test_execute_delivers_lease_identity_change_to_v1_verification_boundary(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    first = _lease()
    second = _lease("1a78ae9d-d11f-4abe-b20f-f9b884f7c6ce")
    raw = _install_raw(monkeypatch, [first, second])
    wrapper = V.prepare_protocol(BASE)
    protocol = tmp_path / "outer.json"
    protocol.write_text(json.dumps(wrapper), encoding="utf-8")
    monkeypatch.setattr(V, "verify_protocol", lambda _path, _anchor: deepcopy(wrapper))

    def base_execute(*_args):
        initial = V.R.SOURCE.LEASE.lease_context()["lease_id"]
        current = V.R.SOURCE.LEASE.lease_context()["lease_id"]
        V.R.require(current == initial, "실행 중 lease identity 변경")
        return {"schema": V.R.RESULT_SCHEMA, "status": "complete"}

    placeholders = [tmp_path / name for name in
                    ("design", "bridge", "source", "corpus", "lab", "bridge-root", "out")]
    with pytest.raises(ValueError, match="identity 변경"):
        V.execute(protocol, "c" * 40, BASE, *placeholders, base_execute=base_execute)
    assert V.R.SOURCE.LEASE.lease_context is raw


def test_prepare_rejects_changed_v1_denominator(tmp_path: Path) -> None:
    value = json.loads(BASE.read_text(encoding="utf-8"))
    value["planned_rows"] = 23
    path = tmp_path / "base.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="base V1"):
        V.prepare_protocol(path)
