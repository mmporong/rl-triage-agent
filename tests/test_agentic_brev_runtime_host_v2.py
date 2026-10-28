"""Real C approval/consumption/seal/done and frozen Runtime tool transitions."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
from test_agentic_brev_repair_evidence_v1 import C, E, host, world


@pytest.mark.parametrize("recovered", [True, False])
def test_actual_shaped_fixture_repair_to_runtime_and_fresh_lookup(tmp_path, recovered):
    data = world(tmp_path)
    data.bad = None if recovered else "not_recovered"
    with host(data) as controller:
        auth = controller.callbacks().approve(data.operation, data.intent)["authorization"]
        result = controller.callbacks().dispatch(data.operation, data.intent, auth)
        assert result["status"] == "completed"
        assert set(result["result"]) == {"seal", "ledger_path", "bands", "next_snapshot", "evidence"}
        E.write(tmp_path / "runtime-tool/host-result.json", result)
        data.runtime._tool(data.loop, data.intent, result, tmp_path / "runtime-tool")
        assert data.loop.status == ("recovered" if recovered else "active") and data.loop.pending is None
        assert data.calls == ["train", "evaluate"]
    with host(data) as fresh:
        assert fresh.lookup(data.operation, data.intent) == result
        assert fresh.dispatch(data.operation, data.intent, auth) == result
        assert fresh.approve(data.operation, data.intent) == result
        assert data.calls == ["train", "evaluate"]
        raw = E.E.loads(Path(result["result"]["seal"]["path"]).read_text(encoding="utf-8"))
        assert raw["output_artifact"]["schema"] == E.ARTIFACT
        assert raw["output_artifact"]["actual_ready"] is False
        assert raw["authorization"]["evidence_kind"] == C.FIXTURE_EVIDENCE
        ledger = C.D1.CorrectionLedger(Path(auth["ledger_path"]), C.RC.correction_registry())
        assert [e["event"] for e in ledger.events()] == ["proposed", "approved", "consumed", "receipt"]
        if not recovered:
            after = result["result"]["next_snapshot"]
            assert after["view"]["observations"] == [] and after["inputs"]["packet"]["epoch"] == 2
            assert after["inputs"]["packet"]["input_bindings"]["case"]["checkpoint_sha256"] == raw["output_artifact"]["checkpoint"]["sha256"]
            assert after["inputs"]["packet"]["input_bindings"]["reference"] == data.intent["snapshot"]["inputs"]["packet"]["input_bindings"]["reference"]


def test_probe_callbacks_are_delegated_unchanged(tmp_path):
    data = world(tmp_path)
    with host(data) as controller:
        probe = {"action": "run_probe"}
        assert controller.approve("synthetic-probe", probe)["status"] == "synthetic-probe-approved"
        assert controller.dispatch("synthetic-probe", probe, {})["status"] == "synthetic-probe-dispatched"
        assert controller.lookup("synthetic-probe", probe)["status"] == "synthetic-probe-lookup"
        assert [row[0] for row in controller.probe_host.calls] == ["approve", "dispatch", "lookup"]
        assert data.calls == []


def test_host_rejects_unregistered_args_and_missing_actual_confirmation(tmp_path):
    data = world(tmp_path)
    with host(data) as controller:
        wrong = deepcopy(data.intent)
        wrong["iterations"] = 301
        with pytest.raises(ValueError, match="operation"):
            controller.approve(data.operation, wrong)
        controller.probe_seals = []
        with pytest.raises(ValueError):
            controller.approve(data.operation, data.intent)
    assert data.calls == []


@pytest.mark.parametrize("target", ["checkpoint", "report", "response", "prefix"])
def test_fresh_lookup_rejects_tampering_in_private_fixture_only(tmp_path, target):
    data = world(tmp_path)
    with host(data) as controller:
        auth = controller.approve(data.operation, data.intent)["authorization"]
        result = controller.dispatch(data.operation, data.intent, auth)
    folder = data.runtime.folder / "brev-repair-host" / data.operation
    seal = E.E.loads(Path(result["result"]["seal"]["path"]).read_text(encoding="utf-8"))
    path = {"checkpoint": Path(seal["output_artifact"]["checkpoint"]["path"]),
            "report": Path(seal["output_report_path"]), "response": folder / "response.json",
            "prefix": folder / "approval-prefix.jsonl"}[target]
    path.write_bytes(b"{}\n")
    with host(data) as controller, pytest.raises((ValueError, KeyError)):
        controller.lookup(data.operation, data.intent)


def test_missing_source_closure_fails_before_approval(tmp_path):
    data = world(tmp_path)
    protocol = deepcopy(data.runtime.protocol)
    protocol["sources_lf"].pop("evals/agentic_brev_repair_evidence_v1.py")
    # A separate scratch protocol is rejected even before its unregistered blob can be used.
    changed = E.write(tmp_path / "changed-protocol.json", protocol)
    registration = deepcopy(data.registration)
    registration["runtime"]["protocol"] = changed
    changed_registration = E.write(tmp_path / "changed-registration.json", registration)
    with pytest.raises(ValueError, match="closure"):
        E.registration(changed_registration, data.maps, allow_fixture=True)
    assert data.calls == []


def test_lookup_recovers_interruption_between_outcome_ref_and_response_without_retraining(tmp_path, monkeypatch):
    data = world(tmp_path)
    original_write = E.write

    def interrupted_write(path, value):
        if Path(path).name == "response.json":
            assert (Path(path).parent / "outcome-ref.json").is_file()
            raise KeyboardInterrupt("synthetic interruption after outcome-ref persistence")
        return original_write(path, value)

    with host(data) as controller:
        authorization = controller.approve(data.operation, data.intent)["authorization"]
        with monkeypatch.context() as injected:
            injected.setattr(E, "write", interrupted_write)
            with pytest.raises(KeyboardInterrupt, match="outcome-ref"):
                controller.dispatch(data.operation, data.intent, authorization)
    folder = data.runtime.folder / "brev-repair-host" / data.operation
    preserved = (folder / "outcome-ref.json").read_bytes()
    assert not (folder / "response.json").exists()
    ledger = C.D1.CorrectionLedger(Path(authorization["ledger_path"]), C.RC.correction_registry())
    assert [row["event"] for row in ledger.events()] == ["proposed", "approved", "consumed", "receipt"]
    assert data.calls == ["train", "evaluate"]
    with host(data) as fresh:
        result = fresh.lookup(data.operation, data.intent)
        assert result["status"] == "completed"
        assert E.E.loads((folder / "response.json").read_text(encoding="utf-8")) == result
        assert (folder / "outcome-ref.json").read_bytes() == preserved
        assert fresh.lookup(data.operation, data.intent) == result
        assert fresh.dispatch(data.operation, data.intent, authorization) == result
        assert data.calls == ["train", "evaluate"]
        assert [row["event"] for row in ledger.events()] == ["proposed", "approved", "consumed", "receipt"]
