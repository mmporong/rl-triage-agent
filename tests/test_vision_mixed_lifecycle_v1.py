from __future__ import annotations

import os
import shutil
from datetime import timedelta
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import pytest
import test_vision_metadata_lifecycle_v1 as MT
from test_run_vision_atomic_compatibility_v1 import (
    NOW,
    ROOT,
    C,
    cumulative_fixture,
    write,
)

pytest_plugins = ["test_run_vision_atomic_compatibility_v1"]


def mixed_fixture(fx, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    protocol = cumulative_fixture(fx)
    protocol["schema"] = C.MIXED_LIFECYCLE_SCHEMA
    owner = {"id": "resource-1", "name": "owned-vm", "instance_type": "l40s-48gb.1x"}
    monkeypatch.setattr(MT, "OWNER", owner)
    metadata = MT._fixture(tmp_path / "metadata-source")
    target = fx.repo / metadata["root"].relative_to(metadata["repo"])
    shutil.copytree(metadata["root"], target)
    envelope_path = target / "envelope.json"
    envelope = C._read(envelope_path)
    envelope["quote"]["hourly_rate_usd"] = "1.8"
    amount = (Decimal("496.25158") * Decimal("1.8") / Decimal(3600)).quantize(
        Decimal("0.000001"), rounding=ROUND_HALF_UP,
    )
    envelope["reported"]["cost_usd"] = str(amount)
    write(envelope_path, envelope)
    resources = protocol["resources"]
    resources["metadata_lifecycle_contract"] = {
        "owner": owner, "allowed_inspection_script_sha256": [metadata["script_sha"]],
        "runs": {envelope["run_id"]: {
            "envelope_sha256": C._sha(envelope_path),
            "start_receipt_sha256": envelope["evidence"]["start_receipt"]["sha256"],
            "stop_receipt_sha256": envelope["evidence"]["stop_receipt"]["sha256"],
        }},
    }
    for reference in resources["prior_stop_receipts"]:
        path = fx.repo / reference["path"]
        body = C._read(path)
        body["owned_instance"]["name"] = owner["name"]
        end = body["cost_estimate"]["stopped_observed_at_utc"]
        body["stop_call"] = {
            "command": ["brev", "stop", owner["name"]], "exit_code": 0,
            "request_before_utc": (C._stamp(end) - timedelta(seconds=2)).isoformat(),
            "response_after_utc": (C._stamp(end) - timedelta(seconds=1)).isoformat(),
            "stdout": {"sha256": "f" * 64}, "stderr": {"sha256": C.METADATA.EMPTY_SHA256},
        }
        write(path, body)
        reference["sha256"] = C._sha(path)
    resources["prior_stop_receipts"].append(C._reference(fx.repo, envelope_path))
    resources["prior_lifecycle_compute_usd"] = float(Decimal("1.8") + amount)
    resources["fresh_lifecycle_budget_usd"] = float(Decimal(20) - Decimal("1.8") - amount)
    fx.controller.source["resources"]["whole_batch_budget_usd"] = resources["fresh_lifecycle_budget_usd"]
    for relative in ("bench/vision_metadata_lifecycle_v1.py", "tests/test_vision_metadata_lifecycle_v1.py",
                     "tests/test_vision_mixed_lifecycle_v1.py"):
        path = fx.repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / relative).read_bytes())
        protocol["controller_sources_lf"][relative] = C._sha(path, lf=True)
    return protocol, envelope_path


def test_v3_preserves_original_total_and_counts_actual_metadata_only_period(fx, tmp_path, monkeypatch):
    protocol, _ = mixed_fixture(fx, tmp_path, monkeypatch)
    checked, source = C.validate_protocol(protocol, fx.repo)
    assert checked["resources"]["prior_lifecycle_compute_usd"] == 2.048126
    assert source["resources"]["whole_batch_budget_usd"] == 17.951874
    latest, leases = C._validate_lifecycle_history(checked, source, fx.repo)
    assert latest.isoformat() == "2026-10-25T13:04:34.707887+00:00"
    assert leases == {"previous-lease-0", "previous-lease-1"}


@pytest.mark.parametrize("kind", ["omit", "duplicate", "reverse", "total", "inflate", "unregistered", "owner", "inspection"])
def test_v3_rejects_missing_repeated_or_unbound_history(fx, tmp_path, monkeypatch, kind):
    protocol, _ = mixed_fixture(fx, tmp_path, monkeypatch)
    resources = protocol["resources"]
    if kind == "omit":
        resources["prior_stop_receipts"].pop()
    elif kind == "duplicate":
        resources["prior_stop_receipts"].append(resources["prior_stop_receipts"][-1])
    elif kind == "reverse":
        resources["prior_stop_receipts"].reverse()
    elif kind == "total":
        resources["prior_lifecycle_compute_usd"] = 1.8
    elif kind == "inflate":
        resources["fresh_lifecycle_budget_usd"] = 20.0
    elif kind == "unregistered":
        resources["metadata_lifecycle_contract"]["runs"] = {}
    elif kind == "owner":
        resources["metadata_lifecycle_contract"]["owner"]["id"] = "other-resource"
    else:
        resources["metadata_lifecycle_contract"]["allowed_inspection_script_sha256"] = ["0" * 64]
    with pytest.raises((ValueError, TypeError)):
        C.validate_protocol(protocol, fx.repo)


def test_v3_rejects_relabelled_metadata_even_with_rehashed_envelope(fx, tmp_path, monkeypatch):
    protocol, path = mixed_fixture(fx, tmp_path, monkeypatch)
    body = C._read(path)
    body["run_id"] = "different-run"
    write(path, body)
    protocol["resources"]["prior_stop_receipts"][-1] = C._reference(fx.repo, path)
    with pytest.raises(ValueError, match="미등록"):
        C.validate_protocol(protocol, fx.repo)


def test_v3_rejects_changed_normal_rate_even_if_receipt_cost_and_total_are_consistent(fx, tmp_path, monkeypatch):
    protocol, _ = mixed_fixture(fx, tmp_path, monkeypatch)
    resources = protocol["resources"]
    reference = resources["prior_stop_receipts"][0]
    path = fx.repo / reference["path"]
    body = C._read(path)
    body["cost_estimate"].update(hourly_compute_rate_usd="0.9", estimated_compute_cost_usd="0.450000")
    write(path, body)
    reference["sha256"] = C._sha(path)
    resources.update(prior_lifecycle_compute_usd=1.598126, fresh_lifecycle_budget_usd=18.401874)
    fx.controller.source["resources"]["whole_batch_budget_usd"] = 18.401874
    with pytest.raises(ValueError, match="요율"):
        C.validate_protocol(protocol, fx.repo)


def test_v3_rejects_same_native_start_stop_under_a_second_registered_run(fx, tmp_path, monkeypatch):
    protocol, path = mixed_fixture(fx, tmp_path, monkeypatch)
    body = C._read(path)
    body["run_id"] = "second-run"
    second = path.parent / "second-envelope.json"
    write(second, body)
    resources = protocol["resources"]
    resources["prior_stop_receipts"].append(C._reference(fx.repo, second))
    resources["metadata_lifecycle_contract"]["runs"]["second-run"] = {
        **resources["metadata_lifecycle_contract"]["runs"]["run-metadata"],
        "envelope_sha256": C._sha(second),
    }
    with pytest.raises(ValueError, match="중복"):
        C.validate_protocol(protocol, fx.repo)


def test_v3_rechecks_metadata_raw_sha_before_registration(fx, tmp_path, monkeypatch):
    protocol, path = mixed_fixture(fx, tmp_path, monkeypatch)
    envelope = C._read(path)
    raw = fx.repo / envelope["evidence"]["stop_stdout"]["path"]
    raw.write_bytes(b"another-vm\n")
    with pytest.raises(ValueError, match="raw SHA"):
        C.validate_protocol(protocol, fx.repo)


def test_v3_rejects_missing_metadata_validator_source_closure(fx, tmp_path, monkeypatch):
    protocol, _ = mixed_fixture(fx, tmp_path, monkeypatch)
    protocol["controller_sources_lf"].pop("bench/vision_metadata_lifecycle_v1.py")
    with pytest.raises(ValueError, match="source 결합"):
        C.validate_protocol(protocol, fx.repo)


@pytest.mark.skipif(os.name == "nt", reason="POSIX symlink 경계는 Linux에서 검증한다")
@pytest.mark.parametrize("index", [0, -1])
def test_v3_rejects_symlink_lifecycle_ref_before_resolving(fx, tmp_path, monkeypatch, index):
    protocol, _ = mixed_fixture(fx, tmp_path, monkeypatch)
    reference = protocol["resources"]["prior_stop_receipts"][index]
    path = fx.repo / reference["path"]
    link = path.parent / "link.json"
    link.symlink_to(path)
    reference["path"] = link.relative_to(fx.repo).as_posix()
    with pytest.raises(ValueError, match="symlink"):
        C.validate_protocol(protocol, fx.repo)


@pytest.mark.parametrize("kind", ["clock", "lease"])
def test_v3_checks_previous_stop_and_normal_lease_before_any_capture(fx, tmp_path, monkeypatch, kind):
    protocol, _ = mixed_fixture(fx, tmp_path, monkeypatch)
    protocol["resources"]["deadline_utc"] = (NOW + timedelta(hours=6)).isoformat()
    fx.controller.source["resources"]["deadline_utc"] = protocol["resources"]["deadline_utc"]
    fx.clock.value = NOW + timedelta(hours=4)
    resource = dict(fx.resource)
    resource.update(
        issued_at_utc=(fx.clock.value - timedelta(seconds=5)).isoformat(),
        compute_started_at_utc=(fx.clock.value - timedelta(seconds=20)).isoformat(),
        expires_at_utc=(fx.clock.value + timedelta(hours=2)).isoformat(),
    )
    if kind == "clock":
        resource["compute_started_at_utc"] = "2026-10-25T13:00:00+00:00"
    else:
        resource["lease_id"] = "previous-lease-0"
    write(fx.resource_path, resource)
    write(fx.protocol_path, protocol)
    controller = C.AtomicCompatibilityController(
        fx.repo, fx.protocol_path, C._sha(fx.protocol_path, lf=True), fx.resource_path,
        C._sha(fx.resource_path), fx.repo / "IsaacLab", fx.repo / "python",
        now=fx.clock, backend_factory=fx.controller.backend_factory,
    )
    with pytest.raises(ValueError, match="이전 STOPPED"):
        controller.register_checkpoint(fx.checkpoint)
    assert not controller.registration_path.exists()
    assert not fx.holder.batch.backend.calls
