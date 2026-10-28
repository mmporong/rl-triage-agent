"""Default consumer identity reachability and explicit synthetic separation."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import timedelta
from types import SimpleNamespace

import pytest
from test_agentic_brev_repair_transport_v1 import REMOTE, E, T, world


def test_default_production_factory_reaches_core_consumer_registration_guard(tmp_path):
    data = world(tmp_path)
    assert REMOTE.PRODUCTION[0] is REMOTE.UNIT.RepairProvider
    # The real constructor first accepts its real ActualServices identity, then
    # rejects this synthetic registration. No actual evidence/job is produced.
    with pytest.raises(ValueError, match="registration synthetic/actual") as error:
        REMOTE.provider(data.registration_ref, data.registration["layout"]["root_maps"]["provider"])
    frames = []
    trace = error.value.__traceback__
    while trace:
        frames.append((trace.tb_frame.f_code.co_filename, trace.tb_frame.f_code.co_name))
        trace = trace.tb_next
    assert any(name == "__init__" and file.endswith("run_agentic_brev_repair_unit_v1.py") for file, name in frames)
    assert any(name == "_registered" and file.endswith("run_agentic_brev_repair_unit_v1.py") for file, name in frames)
    assert data.calls == []


def test_arbitrary_factory_popen_and_fake_services_cannot_enter_production():
    with pytest.raises(ValueError, match="provider identity"):
        REMOTE.provider({}, [], factory=lambda *_: object())
    with pytest.raises(ValueError, match="provider identity"):
        REMOTE.provider({}, [], fixture_services=object())
    with pytest.raises(ValueError, match="launch identity"):
        REMOTE.start({}, {}, {}, None, popen=lambda *_: object())
    with pytest.raises(ValueError, match="fixture remote injection"):
        REMOTE.run_rpc("lookup", {}, fixture_engine=object())


def test_remote_exact_request_rejects_tamper_before_any_spawn(tmp_path):
    data = world(tmp_path)
    body = {"operation_id": data.operation, "registration_ref": data.registration_ref,
        "contract_ref": data.config_ref, "root_map": data.registration["layout"]["root_maps"]["provider"],
        "request_ref": {"path": str(data.remote_root / "absent.json"), "sha256": "0" * 64},
        "authorization_ref": {}, "admission_ref": None, "inventory_ref": {},
        "inventory_observed_at_utc": data.clock.isoformat(), "upload": None}
    with pytest.raises(ValueError):
        REMOTE.prepare(body, allow_fixture=True)
    assert not list(data.remote_root.rglob("launch.json"))


def test_archive_limits_are_part_of_the_registered_contract(tmp_path):
    data = world(tmp_path)
    raw = E.J._bytes({"schema": "invalid"})
    with pytest.raises(ValueError):
        T.unpack({"sha256": "0" * 64, "bytes": len(raw), "base64": "AA=="}, data.config,
                 "host", operation=data.operation, kind="remote-lookup")


def test_failed_admission_cleans_inside_lease_and_records_after_context_exit(tmp_path, monkeypatch):
    # Fault-path ordering only: no synthetic lease is accepted as actual proof.
    order = []
    basis = tmp_path / "basis.json"
    basis.write_text("{}", encoding="utf-8")
    folder = tmp_path / "control"
    folder.mkdir()
    reg = {"repo": tmp_path, "layout": {"output_root": str(tmp_path / "output"), "lease_basis": {}},
           "resolver": SimpleNamespace(path=lambda value: tmp_path / "output", file=lambda value: basis)}
    config = {"remote_stop_intent_path": str(tmp_path / "stop.json"),
              "deadline_utc": (T.clock() + timedelta(hours=1)).isoformat(), "limits": {"owner_ttl_s": 200}}
    body = {"operation_id": "a" * 64, "request_ref": {"path": "synthetic-only", "sha256": "b" * 64}}

    @contextmanager
    def scope():
        yield

    @contextmanager
    def lease(*args, **kwargs):
        order.append("lease-open")
        try:
            yield {"synthetic_cpu": True}
        finally:
            order.append("lease-closed")

    def fail_admission(*args):
        order.append("admission-failed")
        raise RuntimeError("synthetic admission fault")

    def record(*args, children_closed):
        assert children_closed is True
        order.append("release-observed")

    monkeypatch.setattr(REMOTE, "prepare", lambda value: (reg, config, {}, folder))
    monkeypatch.setattr(REMOTE.JOB, "aws_f1_play_scope", scope)
    monkeypatch.setattr(REMOTE.JOB, "resource_lease", lease)
    monkeypatch.setattr(REMOTE, "fresh_admission", fail_admission)
    monkeypatch.setattr(REMOTE, "_cleanup_execution_groups", lambda *args: order.append("children-closed"))
    monkeypatch.setattr(REMOTE, "_record_release", record)
    with pytest.raises(RuntimeError, match="synthetic admission fault"):
        REMOTE.owner(body)
    assert order == ["lease-open", "admission-failed", "children-closed", "lease-closed", "release-observed"]
    failure = E.E.loads((folder / "failure.json").read_text(encoding="utf-8"))
    assert failure["stop_required_if_unreleased"] is True
    assert not (folder / "result-ref.json").exists() and not (folder / "release.json").exists()
