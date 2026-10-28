"""Bounded provider CPU checks; source files are changed only in private copies."""
from __future__ import annotations

from copy import deepcopy
from datetime import timedelta
from pathlib import Path

import pytest
from test_agentic_brev_repair_evidence_v1 import (
    NOW,
    Admitter,
    E,
    Services,
    U,
    host,
    world,
)


@pytest.mark.parametrize("bad", ["cleanup", "lease", "interrupt"])
def test_provider_failure_never_seals_or_restarts(tmp_path, bad):
    data = world(tmp_path)
    data.bad = bad
    with host(data) as controller:
        approval = controller.approve(data.operation, data.intent)["authorization"]
        with pytest.raises((ValueError, KeyboardInterrupt)):
            controller.dispatch(data.operation, data.intent, approval)
        result = controller.lookup(data.operation, data.intent)
        assert result["status"] == "hold"
        assert data.calls == ["train"]
        assert not list(data.runtime.folder.rglob("repair-seal.json"))
        ledger = E.C.D1.CorrectionLedger(Path(approval["ledger_path"]), E.C.RC.correction_registry())
        assert next(iter(ledger.requests().values()))["state"] == "running"
        assert len(list((data.root / "output").rglob("failure.json"))) == 1


@pytest.mark.parametrize("field,seconds", [("provider_observed_at_utc", -31), ("provider_observed_at_utc", 1),
                                         ("admission_at_utc", 1), ("lease_expires_at_utc", -1)])
def test_provider_admission_stale_future_or_expired_rejected(tmp_path, field, seconds):
    data = world(tmp_path)
    approval_ref = E.write(tmp_path / "synthetic-approval-ref.json", {"synthetic_cpu": True})
    admission_ref = Admitter(data).issue(data.operation, {}, approval_ref, data.registration_ref)
    body = data.registered["resolver"].read(admission_ref)
    body[field] = (NOW + timedelta(seconds=seconds)).isoformat()
    changed = E.write(tmp_path / "changed-admission.json", body)
    with pytest.raises(ValueError, match="times|freshness"):
        U.admission_body(changed, data.operation, data.registered, NOW, entry=True)
    assert data.calls == []


def test_provider_observation_before_lease_is_valid_if_current_admission_is_valid(tmp_path):
    data = world(tmp_path)
    admission_ref = Admitter(data).issue(data.operation, {}, {}, data.registration_ref)
    body = U.admission_body(admission_ref, data.operation, data.registered, NOW, entry=True)
    assert E.E._stamp(body["provider_observed_at_utc"]) < E.E._stamp(body["lease_acquired_at_utc"])


def test_fixture_services_cannot_be_injected_into_actual_provider(tmp_path):
    data = world(tmp_path)
    with pytest.raises(ValueError, match="actual services"):
        U.RepairProvider(data.registration_ref, data.maps, services=Services(data), allow_fixture=False)


@pytest.mark.parametrize("field,value", [("seed", 2220), ("iterations", 20), ("max_repairs", 2),
                                        ("max_eval_jobs", 2), ("budget", {"gpu_s": float("nan")})])
def test_unsupported_scope_or_nonfinite_budget_rejected(tmp_path, field, value):
    data = world(tmp_path)
    rule = deepcopy(data.runtime.protocol["repair_contract"])
    rule[field] = value
    with pytest.raises(ValueError):
        E.contract(rule)


@pytest.mark.parametrize("field,value", [("gpu_s", 1), ("compute_usd", 0.001)])
def test_execution_windows_must_fit_gpu_and_money_reservation(tmp_path, field, value):
    data = world(tmp_path)
    rule = deepcopy(data.runtime.protocol["repair_contract"])
    rule["budget"][field] = value
    with pytest.raises(ValueError, match="reservation"):
        E.contract(rule)
