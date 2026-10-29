from __future__ import annotations

import shutil
import sys
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / part) for part in ("bench", "evals", "src")]

import agentic_brev_runtime_host_v1 as V1
import agentic_brev_runtime_host_v5 as V5
import agentic_brev_runtime_host_v6 as V6
import run_agentic_brev_probe_parent_v7 as P7
import test_agentic_brev_runtime_host_v1 as FULL
from test_agentic_brev_runtime_host_v1 import Transport as FullTransport
from test_agentic_brev_runtime_host_v1 import setup as full_setup
from test_agentic_investigation_evidence_v1 import case as evidence_case
from test_agentic_investigation_runtime_v1 import commit_objects


def _host(tmp_path: Path, *, ordinal: int = 2, prefix: int = 1):
    runtime = SimpleNamespace(
        manifest={"schema": "manifest"},
        folder=tmp_path,
        reg={"prefix": [{} for _ in range(prefix)]},
    )
    intent = {"action": "run_probe", "target": "P_reward"}
    operation = V1.J.digest({"manifest": runtime.manifest, "intent": intent})
    step = tmp_path / "steps" / f"{ordinal:04d}"
    step.mkdir(parents=True)
    V1.J._write_new(step / "intent.json", intent)
    V1.J._write_new(step / "operation.json", {
        "operation_id": operation,
        "intent_sha256": V1.J.file_sha(step / "intent.json"),
    })
    host = object.__new__(V6.BrevRuntimeHost)
    host.runtime = runtime
    host._owner = lambda: None
    host._sources = lambda: None
    return host, operation, intent


def _save(path: Path, value: dict) -> dict:
    V1.J._write_new(path, value)
    return V1.R.ref(path)


def _fresh_source(data: dict) -> dict:
    runtime_root = data["runtime"].folder.parent
    step = data["runtime"].folder / "steps/0002"
    capsule_ref = V1.J._read(step / "capsule-reference.json")
    _capsule_path, capsule = V1.R.CALL.prepared(capsule_ref)
    engine = V1.R.completed(capsule_ref, runtime_root / "api")
    reassessed = P7.V6.V5.A2.assess(engine["final_content"], capsule["context"])
    choice = P7.V6.V5.A2.probe_choice(capsule["context"], reassessed)
    engine = deepcopy(engine)
    engine["review"] = {**engine["review"], "controller_choice": choice}
    root = runtime_root / "api10-host6"
    root.mkdir()
    prepared = {"context": capsule["context"], "messages": capsule["messages"]}
    prepared_ref = _save(root / "prepared.json", prepared)
    projection_ref = _save(root / "projection.json", {
        "status": "PASS", "projected_prepared_sha256": prepared_ref["sha256"],
        "same_model_messages": True, "same_model_view": True, "same_prior_history": True,
    })
    binding = {
        "run_id": "host6-full-cpu",
        "snapshot_sha256": V1.E.digest(prepared["context"]["snapshot"]),
        "initial_messages_sha256": V1.E.digest(prepared["messages"]),
    }
    params = capsule["params"]
    manifest_ref = _save(root / "manifest.json", {
        "schema": "model_decision_journal_v1", "binding": binding,
        "binding_sha256": V1.E.digest(binding), "params": params,
        "params_sha256": V1.E.digest(params),
    })
    reservation_ref = _save(root / "reservation.json", {
        "schema": "model_decision_reservation_v1", "index": 10, "ordinal": 1,
        "run_id": binding["run_id"], "binding_sha256": V1.E.digest(binding),
        "manifest_sha256": manifest_ref["sha256"],
        "messages_sha256": V1.E.digest(prepared["messages"]),
        "params_sha256": V1.E.digest(params),
    })
    provider_result = {"content": engine["final_content"]}
    response_ref = _save(root / "response.json", {
        "schema": "model_decision_response_v1", "status": "completed", "index": 10,
        "reservation_sha256": reservation_ref["sha256"], "result": provider_result,
    })
    receipt_ref = _save(root / "receipt.json", {
        "reservation_sha256": reservation_ref["sha256"],
        "response_sha256": response_ref["sha256"],
    })
    event_ref = _save(root / "event.json", {
        "schema": "model_decision_event_v1", "global_index": 10, "ordinal": 1,
        "manifest_sha256": manifest_ref["sha256"],
        "reservation_sha256": reservation_ref["sha256"],
        "response_sha256": response_ref["sha256"],
    })
    terminal_ref = _save(root / "terminal.json", {
        "schema": "model_decision_terminal_v1", "status": "choice_ready",
        "manifest_sha256": manifest_ref["sha256"], "completed_calls": 1,
        "calls_sha256": V1.E.digest([provider_result]), "engine_result": engine,
    })
    actual_ref = _save(root / "actual-result.json", {"engine_result": engine})

    def logical(ref: dict) -> dict:
        path = Path(ref["path"])
        return {"path": "/runtime/" + path.relative_to(runtime_root).as_posix(),
                "sha256": ref["sha256"]}

    return {
        "schema": "fresh_selection_source_v1", "prepared": logical(prepared_ref),
        "actual_result": logical(actual_ref), "projection": logical(projection_ref),
        "manifest": logical(manifest_ref), "event": logical(event_ref),
        "terminal": logical(terminal_ref), "reservation": logical(reservation_ref),
        "response": logical(response_ref), "receipt": logical(receipt_ref),
        "api_global_index": 10,
    }


def _upgrade_full_fixture(data: dict) -> None:
    source = _fresh_source(data)
    producer = data["producer"]
    previous_registration = V1.R.read_ref(data["binding_ref"])
    reference_provider_path = producer["root"] / "host/provider-NONE.json"
    reference_provider = V1.J._read(reference_provider_path)
    reference_provider["schema"] = P7.SCHEMA + "_provider_packet"
    reference_provider_path.write_bytes(V1.J._bytes(reference_provider))
    producer["binding"]["schema"] = P7.SCHEMA + "_binding"
    pair = producer["binding"]["expected_request"]
    execution = pair["execution"]
    probe, args = data["intent"]["target"], data["intent"]["request_spec"]["args"]
    pair["probe"] = probe
    for arm in P7.ORDER:
        pair["commands"][arm]["probe"] = probe
        pair["commands"][arm]["args"] = deepcopy(args)
        execution["operations"][arm]["probe"] = probe
        execution["operations"][arm]["args"] = deepcopy(args)
        stage = producer["templates"][arm] / "stage"
        measurement = {
            "track_lin_vel_xy_exp_rel_error": 0.0 if arm == "NONE" else 0.1,
            "track_ang_vel_z_exp_rel_error": 0.0 if arm == "NONE" else 0.1,
        }
        job = V1.J._read(stage / "job.json")
        job["probe"] = probe
        (stage / "job.json").write_bytes(V1.J._bytes(job))
        report = V1.J._read(stage / "report.json")
        report["probes"] = {probe: {"exit": 0, "measurement": measurement}}
        (stage / "report.json").write_bytes(V1.J._bytes(report))
        jobs = V1.J._read(stage / "report.jobs.json")
        jobs["jobs"][0]["probes"] = [probe]
        (stage / "report.jobs.json").write_bytes(V1.J._bytes(jobs))
        worker = V1.J._read(stage / "worker.json")
        worker.update(probe=probe, measurement=measurement)
        worker["report"]["sha256"] = V1.J.file_sha(stage / "report.json")
        (stage / "worker.json").write_bytes(V1.J._bytes(worker))
        backend = V1.J._read(stage / "backend_result.json")
        backend.update(measurement=measurement, report=report)
        (stage / "backend_result.json").write_bytes(V1.J._bytes(backend))
    pair["commands_sha256"] = P7.PAIR.digest(pair["commands"])
    execution["selection_source"] = source
    execution["schema"] = P7.SCHEMA + "_execution"
    pair["selection"] = {
        "schema": "fresh_selection_pair_binding_v1",
        "api_global_index": 10,
        "actual_result_sha256": source["actual_result"]["sha256"],
        "journal_terminal_sha256": source["terminal"]["sha256"],
    }
    execution["runtime_host_binding"]["schema"] = V6.SCHEMA + "_runtime_binding"
    hashes = {
        name: sha256((ROOT / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        for name in V6.SOURCES
    }
    execution["source_closure_sha256"] = V1.E.digest(hashes)
    ledger_prefix = P7.PairLedger(producer["root"] / "host6-pairs.ledger.jsonl").authorize(
        pair, "cpu_explicit",
    )
    projection = execution["ledger_projection"]
    shutil.copyfile(ledger_prefix["path"], producer["root"] / "transport/ri21-consumed.jsonl")
    binding_ref = _save(producer["root"] / "host6-binding.json", producer["binding"])
    protocol = {
        **producer["protocol"], "schema": P7.SCHEMA + "_protocol",
        "sources_lf": hashes, "binding_sha256": binding_ref["sha256"],
    }
    protocol["allowed_probes"] = sorted(set(protocol["allowed_probes"]) | {probe})
    files = {name: (ROOT / name).read_bytes() for name in V6.SOURCES}
    for name in ("bench/protocols/probe.json", "bench/protocols/basis.json"):
        files[name] = (producer["repo"] / name).read_bytes()
    files["bench/protocols/host6-parent.json"] = V1.J._bytes(protocol)
    parent_anchor = commit_objects(
        producer["repo"], files, previous_registration["parent_anchor"],
    )
    request = deepcopy(producer["request"])
    request.update(
        schema=P7.SCHEMA + "_request", source_anchor=parent_anchor,
        protocol=V1.R.ref(producer["repo"] / "bench/protocols/host6-parent.json"),
        binding=binding_ref,
    )
    request["authority"] = {
        "schema": P7.SCHEMA + "_authority",
        "host_reference": {**ledger_prefix, "path": projection["host_path"]},
        "projection": deepcopy(projection),
    }
    request_ref = _save(producer["root"] / "host6-request.json", request)
    registration_ref = _save(data["runtime"].folder.parent / "host6-registration.json", {
        "schema": V6.SCHEMA + "_registration", "parent_protocol": request["protocol"],
        "parent_binding": binding_ref, "parent_request": request_ref,
        "parent_anchor": parent_anchor,
    })
    producer["request"], producer["request_ref"] = request, request_ref
    data["binding_ref"] = registration_ref
    native_actual = {
        "path": str(data["runtime"].folder.parent / source["actual_result"]["path"].removeprefix("/runtime/")),
        "sha256": source["actual_result"]["sha256"],
    }
    data["runtime"].reg["prefix"] = [
        {"actual_result": native_actual},
        {"actual_result": native_actual},
    ]


def test_parent7_evidence_scope_uses_registered_parent_and_restores() -> None:
    protocol = {"schema": P7.SCHEMA + "_protocol",
                "sources_lf": {name: "0" * 64 for name in P7.SOURCES}}
    original = V1.P
    with V6._parent7_evidence_scope(protocol):
        assert V1.P is P7
    assert V1.P is original
    with pytest.raises(ValueError, match="registered Parent7"), \
            V6._parent7_evidence_scope({**protocol, "schema": "other_protocol"}):
        pass
    assert V1.P is original


def test_host5_rejects_parent7_plan_while_host6_passes_that_boundary(tmp_path: Path) -> None:
    stage = tmp_path / "stage" / "raw"
    stage.mkdir(parents=True)
    plan = {
        "schema": P7.SCHEMA + "_plan",
        "identity": {"side": "reference", "arm": "NONE", "seed": 2221,
                     "probe": "P_reward", "run_name": "none"},
        "operation_id": "none-op",
        "logical_stage": "/remote/none",
        "source": {"checkpoint": {"sha256": "1" * 64}},
        "expected_source": {"cuda_available": True, "cuda_device_count": 1},
        "evidence_kind": V1.E.FIXTURE,
        "parent_request": {"sha256": "wrong"},
        "authority": {"request_sha256": "wrong", "host_reference": {}},
    }
    original = {"evidence_kind": V1.E.FIXTURE}
    plan_path, evidence_path = tmp_path / "plan.json", stage.parent / "evidence.json"
    V1.J._write_new(plan_path, plan)
    V1.J._write_new(evidence_path, original)
    outcome_path = stage.parent / "outcome.json"
    V1.J._write_new(outcome_path, {
        "status": "complete", "operation_id": "none-op",
        "plan": V1.R.ref(plan_path), "evidence": V1.R.ref(evidence_path),
    })

    class Metadata:
        @staticmethod
        def read(ref):
            return V1.R.read_ref(ref)

        @staticmethod
        def file(ref):
            return Path(ref["path"])

    bound = {
        "folder": tmp_path / "host", "metadata": Metadata(),
        "protocol": {"schema": P7.SCHEMA + "_protocol",
                     "sources_lf": {name: "0" * 64 for name in P7.SOURCES}},
        "execution": {
            "operations": {"NONE": {
                "side": "reference", "arm": "NONE", "seed": 2221,
                "probe": "P_reward", "run_name": "none", "operation_id": "none-op",
                "host_stage": str(stage), "remote_stage": "/remote/none",
                "source": plan["source"],
            }},
            "backend": {"expected_source": plan["expected_source"]},
        },
        "binding": {"evidence_kind": V1.E.FIXTURE},
        "registration": {"parent_request": {"sha256": "expected"}},
        "authority": {"request_sha256": "expected", "host_reference": {}},
    }
    host5 = object.__new__(V5.BrevRuntimeHost)
    host5.allow_fixture = True
    with pytest.raises(ValueError, match="parent plan/receipt"):
        host5._evidence(bound, "NONE", V1.R.ref(outcome_path))
    host6 = object.__new__(V6.BrevRuntimeHost)
    host6.allow_fixture = True
    with pytest.raises(ValueError, match="parent plan authority"):
        host6._evidence(bound, "NONE", V1.R.ref(outcome_path))

def test_unsupported_second_ordinal_is_hold_without_dispatch(tmp_path: Path) -> None:
    host, operation, intent = _host(tmp_path)
    result = host.approve(operation, intent)
    assert result == {"status": "hold", "reason": "fresh_prefix_ordinal_unsupported",
                      "operation_id": operation}


def test_invalid_operation_and_non_probe_are_not_hidden_as_hold(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    host, operation, intent = _host(tmp_path)
    calls = []

    def rejected(self, value, payload):
        calls.append((value, payload))
        raise ValueError("registered binding rejected")

    monkeypatch.setattr(V5.BrevRuntimeHost, "approve", rejected)
    with pytest.raises(ValueError, match="registered binding rejected"):
        host.approve("f" * 64, intent)
    repair = {**intent, "action": "run_repair"}
    with pytest.raises(ValueError, match="registered binding rejected"):
        host.approve(operation, repair)
    assert len(calls) == 2


def test_registered_prefix_path_keeps_existing_approval_flow(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    host, operation, intent = _host(tmp_path, ordinal=1, prefix=1)
    calls = []

    def approved(self, value, payload):
        calls.append((value, payload))
        return {"status": "approved", "authorization": {"operation_id": value}}

    monkeypatch.setattr(V5.BrevRuntimeHost, "approve", approved)
    first = host.approve(operation, intent)
    second = host.approve(operation, intent)
    assert first == second and len(calls) == 2


def test_operation_record_tamper_is_rejected_not_held(tmp_path: Path) -> None:
    host, operation, intent = _host(tmp_path)
    path = tmp_path / "steps/0002/operation.json"
    path.write_bytes(V1.J._bytes({"operation_id": operation, "intent_sha256": "0" * 64}))
    with pytest.raises(ValueError, match="immutable operation record"):
        host.approve(operation, intent)


def test_full_host5_negative_then_host6_approve_with_valid_parent7_receipt(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model_type = FULL.Model
    monkeypatch.setattr(
        FULL, "Model",
        lambda decisions: model_type([
            (action, "P_reward" if action == "run_probe" else target)
            for action, target in decisions
        ]),
    )
    fixture = cast(Any, evidence_case).__wrapped__(tmp_path / "evidence")
    data = full_setup(tmp_path / "full-host", fixture)
    _upgrade_full_fixture(data)
    monkeypatch.setattr(FULL, "P", P7)
    start = FullTransport._start
    run = FullTransport._run

    def scoped_start(self, *args):
        with P7.V6.fresh_parent_scope():
            return start(self, *args)

    def scoped_run(self, *args):
        with P7.V6.fresh_parent_scope():
            return run(self, *args)

    monkeypatch.setattr(FullTransport, "_start", scoped_start)
    monkeypatch.setattr(FullTransport, "_run", scoped_run)
    operation, intent = data["operation"], data["intent"]
    pair = data["producer"]["binding"]["expected_request"]
    execution = pair["execution"]
    protocol = V1.R.read_ref(V1.R.read_ref(data["binding_ref"])["parent_protocol"])
    checks = {
        "kind": data["producer"]["binding"]["evidence_kind"]
        == execution["evidence_kind"] == V1.E.FIXTURE,
        "arms": pair["arms"] == list(P7.ORDER),
        "seed": pair["seed"] == 2221,
        "order": execution["order"] == list(P7.ORDER),
        "gpu": execution["gpu_binding"] == P7.GPU_BINDING,
        "probe": pair["probe"] == P7.registered_probe(protocol, pair) == intent["target"],
        "commands": pair["commands_sha256"] == P7.PAIR.digest(pair["commands"]),
        "closure": execution["source_closure_sha256"] == V1.E.digest(protocol["sources_lf"]),
    }
    assert all(checks.values()), checks

    host5_transport = FullTransport(data)
    try:
        with V5.BrevRuntimeHost(
            data["runtime"], {operation: data["binding_ref"]}, host5_transport,
            "cpu_operator", allow_fixture=True,
        ) as host5, pytest.raises(ValueError, match="parent plan/receipt"):
            host5.approve(operation, intent)
        assert host5_transport.calls == ["prepare", "reference"]
    finally:
        host5_transport.close()
    shutil.rmtree(data["runtime"].folder / "brev-host" / operation)

    host6_transport = FullTransport(data)
    try:
        with V6.BrevRuntimeHost(
            data["runtime"], {operation: data["binding_ref"]}, host6_transport,
            "cpu_operator", allow_fixture=True,
        ) as host6:
            approved = host6.approve(operation, intent)
            duplicate = host6.approve(operation, intent)
        assert approved["status"] == duplicate["status"] == "approved"
        assert host6_transport.calls == ["prepare", "reference"]
        ledger = V1.C.S.M.RegistryLedger(
            Path(intent["snapshot"]["inputs"]["ledger_path"]), V1.C.RC.registry(),
        )
        assert [row["event"] for row in ledger.events()] == [
            "proposed", "approved", "consumed",
        ]
    finally:
        host6_transport.close()
