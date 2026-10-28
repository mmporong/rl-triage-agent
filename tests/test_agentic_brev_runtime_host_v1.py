"""기존 Runtime/C와 부모 fakeprocess를 연결하는 host CPU 회귀."""
from __future__ import annotations

import shutil
import sys
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / p) for p in ("evals", "bench", "tests", "src")]
import agentic_brev_runtime_host_v1 as H
from test_agentic_brev_probe_parent_v1 import NOW, Services
from test_agentic_brev_probe_parent_v1 import world as parent_world
from test_agentic_investigation_evidence_v1 import case  # noqa: F401 - pytest fixture
from test_agentic_investigation_runtime_v1 import Model, commit_objects, no_model

P, R, C, E, J = H.P, H.R, H.C, H.E, H.J


def save(path, value):
    J._write_new(path, value)
    return R.ref(path)


class Transport:
    EVIDENCE_KIND = E.FIXTURE

    def __init__(self, data):
        self.data = data
        self.calls = []
        self.services = Services(data["producer"])
        self.prepared = None
        self.lease_context = self.owner_context = None
        self.pending = False
        self.interrupt_case = False
        self.bad_reference = False
        self.wrong_operation = False
        self.contract_ref = data["contract_ref"]
        self.contract = R.read_ref(self.contract_ref)

    def prepare_pair(self, operation_id, intent, registration_ref):
        self.calls.append("prepare")
        assert operation_id == self.data["operation"] and registration_ref == self.data["binding_ref"]
        assert intent == self.data["intent"]
        return {"status": "prepared", "operation_id": operation_id}

    def _archive(self):
        source, archive = self.data["producer"]["root"], self.data["archive"]
        for path in source.rglob("*"):
            if not path.is_file():
                continue
            target = archive / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                assert target.read_bytes() == path.read_bytes(), "fixture archive overwrite rejected"
            else:
                shutil.copyfile(path, target)

    def _outcome(self, arm):
        op = self.data["producer"]["binding"]["expected_request"]["execution"]["operations"][arm]
        return Path(op["host_stage"]).parent / "outcome.json"

    def _start(self):
        w = self.data["producer"]
        self.prepared = P.prepare(w["request_ref"], allow_fixture=True, home=w["home"], now=NOW)
        self.prepared["folder"].mkdir(parents=True)
        J._write_new(self.prepared["folder"] / "intent.json", {"schema": P.SCHEMA + "_intent",
            "request_ref": w["request_ref"], "authority": self.prepared["authority"], "order": list(P.ORDER), "gpu_binding": P.GPU_BINDING})
        for details in self.prepared["operations"].values():
            details["unit"].mkdir(parents=True)
        self.owner_context = P._owner_scope(self.services)
        self.owner_context.__enter__()
        self.lease_context = P._acquire(self.prepared, self.services)
        self.lease = self.lease_context.__enter__()
        self.boot = self.services.boot(self.prepared["folder"] / "boot.txt")

    def _run(self, arm):
        prepared = self.prepared
        assert prepared is not None

        def check():
            assert self.services.active
            self.services.verify(prepared, self.lease, prepared["basis"])

        folder, proof, capture, raw, model = P._proof(prepared, arm, self.lease, prepared["basis"],
            self.services, check, self.boot, None if arm == "NONE" else self.uuid)
        self.uuid = proof["gpu_uuid"]
        plan, plan_ref = P._plan(prepared, arm, folder, proof, capture, raw, model, self.lease, self.boot)
        result = P._dispatch(prepared, arm, plan, plan_ref, self.services, check)
        P._verify_completion(prepared, result)
        return result

    def start_reference(self, operation_id, intent, registration_ref):
        self.calls.append("reference")
        assert operation_id == self.data["operation"] and registration_ref == self.data["binding_ref"]
        if not self._outcome("NONE").exists():
            self._start()
            self._run("NONE")
            self._archive()
        result = R.ref(self._outcome("NONE"))
        if self.bad_reference:
            source = self.data["archive"] / "NONE/source/model_299.pt"
            source.write_bytes(b"tampered fixture checkpoint")
        return {"status": "completed", "operation_id": "f" * 64 if self.wrong_operation else operation_id, "result": result}

    def lookup_reference(self, operation_id, intent, registration_ref):
        self.calls.append("lookup_reference")
        path = self._outcome("NONE")
        return {"status": "completed", "operation_id": operation_id, "result": R.ref(path)} if path.exists() else {
            "status": "unknown", "operation_id": operation_id}

    def dispatch_case(self, operation_id, intent, registration_ref, authorization):
        self.calls.append("case")
        checked = C.validate_probe_authorization(authorization, intent["snapshot"])
        assert checked["request"]["state"] == "running"
        assert self._outcome("NONE").exists() and self.services.active
        w = self.data["producer"]
        op = w["binding"]["expected_request"]["execution"]["operations"]["F1"]
        packet = w["root"] / "host/provider-F1.json"
        assert not packet.exists(), "case provider cannot be released before C consumption"
        save(packet, {"schema": P.SCHEMA + "_provider_packet", "operation_id": op["operation_id"],
            "observed_at_utc": self.services.clock().isoformat(), "inventory": {
                "path": "/host/provider.json", "sha256": w["provider_ref"]["sha256"]}})
        try:
            self._run("F1")
        finally:
            self.close()
        self._archive()
        if self.interrupt_case:
            raise KeyboardInterrupt("CPU transport response interrupted after case")
        return {"status": "pending", "operation_id": operation_id} if self.pending else {
            "status": "completed", "operation_id": operation_id, "result": R.ref(self._outcome("F1"))}

    def lookup_pair(self, operation_id, intent, registration_ref):
        self.calls.append("lookup_pair")
        path = self._outcome("F1")
        if self.pending or not path.exists():
            return {"status": "pending", "operation_id": operation_id}
        return {"status": "completed", "operation_id": operation_id, "result": R.ref(path)}

    def close(self):
        if self.lease_context is not None:
            self.lease_context.__exit__(None, None, None)
            self.lease_context = None
        if self.owner_context is not None:
            self.owner_context.__exit__(None, None, None)
            self.owner_context = None


def setup(tmp_path, case, *, mismatch=None):  # noqa: F811 - pytest fixture data
    w = parent_world(tmp_path / "producer")
    runtime_root = tmp_path / "runtime"
    api = runtime_root / "api"
    save(api / "manifest.json", {"schema": J.API_SCHEMA, "api_cap": 12})
    inputs = deepcopy(case["snapshot"]["inputs"])
    inputs["ledger_path"] = str(tmp_path / "case/epoch1/probes.ledger.jsonl")
    snapshot = C.V.build_snapshot(**inputs)
    bridge_registration = R.B.register(case["index"], case["roots"], snapshot,
        expected_index_sha256=R.B.E.index_sha256(case["index"]),
        source_lineage_id=case["registration"]["session"]["source_lineage_id"], aliases=case["registration"]["aliases"])
    bridge = R.B.EvidenceAdapter(case["index"], case["roots"], bridge_registration,
                                expected_registration_sha256=R.B.E.index_sha256(bridge_registration))
    reg = R.registration(snapshot, goal="CPU 연결", api=R.capture_api(api, baseline=0), max_decisions=4,
                         evidence=R.evidence_spec(bridge), allow_fixture=True)
    reg_ref = save(runtime_root / "registration.json", reg)
    files = {name: (ROOT / name).read_bytes() for name in H.SOURCES}
    hashes = {name: sha256(raw.replace(b"\r\n", b"\n")).hexdigest() for name, raw in files.items()}
    contract_ref = save(runtime_root / "io-contract.json", {"schema": "agentic_brev_mailbox_transport_v1_contract",
        "mailbox": str(runtime_root / "run/transport-mailbox"), "result_root": str(tmp_path / "archive"),
        "response_wait_s": 120, "service_source_sha256_lf": hashes["tests/test_agentic_brev_runtime_host_v1.py"]})
    for name in ("bench/protocols/probe.json", "bench/protocols/basis.json"):
        files[name] = (w["repo"] / name).read_bytes()
    runtime_protocol = {"schema": R.SCHEMA + "_protocol", "registration_sha256": reg_ref["sha256"],
        "api_cap": 12, "controller_extensions": [E.GPU_CONTRACT], "sources_lf": hashes,
        "host_transport_contract_sha256": contract_ref["sha256"],
        "host_transport_service_sha256_lf": hashes["tests/test_agentic_brev_runtime_host_v1.py"]}
    files["bench/protocols/ri21-runtime.json"] = J._bytes(runtime_protocol)
    anchor_r = commit_objects(w["repo"], files, w["request"]["source_anchor"])
    args = (runtime_root / "run", reg_ref, R.ref(w["repo"] / "bench/protocols/ri21-runtime.json"), anchor_r)
    model = Model([("read_evidence", "evidence:training-01"), ("run_probe", "P_episode"), ("hold", None)])
    runtime = R.Runtime(*args, repo=w["repo"])
    assert runtime.advance(complete=model)["status"] == "active"
    assert runtime.advance(complete=model)["status"] == "blocked_host"
    step = runtime.folder / "steps/0002"
    operation, intent = J._read(step / "operation.json")["operation_id"], J._read(step / "intent.json")
    execution = w["binding"]["expected_request"]["execution"]
    source_roots = case["roots"]
    for arm, role in (("NONE", "reference"), ("F1", "case")):
        template = w["templates"][arm]
        training = case["entry"]["training"][role]
        for key, destination in (("checkpoint", "model_299.pt"), ("telemetry", "telemetry.json")):
            shutil.copyfile(R.B.E._ref(training[key], source_roots), template / "source" / destination)
        cp_sha, tel_sha = J.file_sha(template / "source/model_299.pt"), J.file_sha(template / "source/telemetry.json")
        artifact_path = template / "source/adapter_artifact.json"
        artifact = J._read(artifact_path)
        artifact["checkpoint"]["sha256"], artifact["telemetry"]["sha256"] = cp_sha, tel_sha
        artifact_path.write_bytes(J._bytes(artifact))
        source = execution["operations"][arm]["source"]
        source["checkpoint"]["sha256"], source["telemetry"]["sha256"] = cp_sha, tel_sha
        source["artifact"]["sha256"] = J.file_sha(artifact_path)
        w["binding"]["expected_request"]["checkpoints"][arm] = cp_sha
        job_path, report_path, worker_path = (template / "stage" / name for name in ("job.json", "report.json", "worker.json"))
        job, report, worker = J._read(job_path), J._read(report_path), J._read(worker_path)
        job["checkpoint_sha256"], report["checkpoint"]["sha256"] = cp_sha, cp_sha
        job_path.write_bytes(J._bytes(job))
        report_path.write_bytes(J._bytes(report))
        worker["report"]["sha256"] = J.file_sha(report_path)
        worker_path.write_bytes(J._bytes(worker))
        backend_path = template / "stage/backend_result.json"
        backend = J._read(backend_path)
        backend["report"] = report
        backend_path.write_bytes(J._bytes(backend))
    roots = w["binding"]["roots"]
    roots.append({"remote": "/runtime", "local": str(runtime_root)})
    capsule = J._read(step / "capsule-reference.json")
    raw_capsule = R.read_ref(capsule)

    def logical_ref(path):
        return {"path": "/runtime/" + path.relative_to(runtime_root).as_posix(), "sha256": J.file_sha(path)}

    call = Path(capsule["path"]).parent
    execution.update(selection_capsule=logical_ref(Path(capsule["path"])), selection_result=logical_ref(call / "journal/terminal.json"),
        selection_manifest=logical_ref(call / "journal/manifest.json"),
        selection_calls=[{"event": logical_ref(call / "journal/events/0001.json"),
            "reservation": logical_ref(api / "reservations/0002.json"), "response": logical_ref(api / "responses/0002.json")}])
    snap_ref = save(runtime_root / "parent-snapshot.json", raw_capsule["context"]["snapshot"])
    execution["snapshot"] = logical_ref(Path(snap_ref["path"]))
    execution["snapshot_history_projection"] = {"schema": P.SCHEMA + "_snapshot_history_projection",
        "host_path": raw_capsule["context"]["snapshot"]["inputs"]["ledger_path"],
        "host_base_capsule_path": raw_capsule["base_reference"]["path"],
        "base_capsule": logical_ref(call / "history-binding-v1/capsule.json"),
        "history": logical_ref(call / "history-binding-v1/history.jsonl")}
    execution["windows"]["provider_wait_s"] = 120
    execution["source_closure_sha256"] = E.digest(hashes)
    archive = tmp_path / "archive"
    archive.mkdir()

    def relocated(value):
        return str(archive / Path(value).relative_to(w["root"])) if Path(value).is_relative_to(w["root"]) else value

    evidence_roots = {}
    for arm in P.ORDER:
        op = execution["operations"][arm]
        stage = w["repo"] / "parents/operations" / op["operation_id"] / "raw"
        op["host_stage"] = relocated(str(stage))
        commands = w["binding"]["expected_request"]["commands"][arm]
        commands["stage"] = op["host_stage"]
        original = op["source_roots"] + roots + [{"remote": "/parent/" + op["operation_id"], "local": str(w["folder"] / op["operation_id"])},
            {"remote": op["remote_stage"], "local": str(stage)}]
        evidence_roots[arm] = [{"remote": row["remote"], "local": relocated(row["local"])} for row in original]
    execution["runtime_host_binding"] = {"schema": H.SCHEMA + "_runtime_binding", "operation_id": operation,
        "runtime_manifest_sha256": J.digest(runtime.manifest), "intent_sha256": J.digest(intent),
        "runtime_source_anchor": anchor_r, "runtime_protocol_sha256": runtime.protocol_ref["sha256"],
        "host_input_roots": roots, "metadata_roots": [{"remote": str(w["root"]), "local": str(archive)}], "evidence_roots": evidence_roots}
    if mismatch == "operation":
        execution["runtime_host_binding"]["operation_id"] = "f" * 64
    elif mismatch == "args":
        execution["operations"]["F1"]["args"] = {"unregistered": 1}
    elif mismatch == "checkpoint":
        execution["operations"]["NONE"]["source"]["checkpoint"]["sha256"] = "0" * 64
    pair = w["binding"]["expected_request"]
    pair["selection"] = {"terminal_sha256": execution["selection_result"]["sha256"], "activation_bridge_sha256": capsule["sha256"]}
    pair["commands_sha256"] = P.PAIR.digest(pair["commands"])
    execution["ledger_projection"] = {"host_path": "Q:\\ri21\\pair.ledger.jsonl", "remote_path": "/transport/ri21-consumed.jsonl"}
    prefix = P.PairLedger(w["root"] / "ri21-pairs.ledger.jsonl").authorize(pair, "cpu_explicit")
    shutil.copyfile(prefix["path"], w["root"] / "transport/ri21-consumed.jsonl")
    binding_ref = save(w["root"] / "ri21-binding.json", w["binding"])
    protocol = {**w["protocol"], "sources_lf": hashes, "binding_sha256": binding_ref["sha256"]}
    files["bench/protocols/ri21-parent.json"] = J._bytes(protocol)
    anchor_p = commit_objects(w["repo"], files, anchor_r)
    request = {**w["request"], "source_anchor": anchor_p, "protocol": R.ref(w["repo"] / "bench/protocols/ri21-parent.json"),
        "binding": binding_ref, "authority": {"schema": P.SCHEMA + "_authority", "host_reference": {
            **prefix, "path": execution["ledger_projection"]["host_path"]}, "projection": execution["ledger_projection"]}}
    w["request_ref"] = save(w["root"] / "ri21-request.json", request)
    w["binding"]["expected_request"] = pair
    native_registration = {"schema": H.SCHEMA + "_registration", "parent_protocol": request["protocol"],
        "parent_binding": binding_ref, "parent_request": w["request_ref"], "parent_anchor": anchor_p}
    binding_ref = save(runtime_root / "host-parent-registration.json", native_registration)
    (w["root"] / "host/provider-F1.json").unlink()
    return {"producer": w, "archive": archive, "args": args, "repo": w["repo"], "operation": operation,
        "intent": intent, "binding_ref": binding_ref, "model": model, "runtime": runtime, "contract_ref": contract_ref}


def host(data, transport):
    runtime = R.Runtime(*data["args"], repo=data["repo"])
    return H.BrevRuntimeHost(runtime, {data["operation"]: data["binding_ref"]}, transport, "cpu_operator", allow_fixture=True)


def test_runtime_reference_then_C_authorization_case_seal_and_next_model(case, tmp_path):  # noqa: F811
    data = setup(tmp_path, case)
    transport = Transport(data)
    try:
        result = host(data, transport).advance(complete=no_model)
        assert result["status"] == "active", result
        assert transport.calls == ["prepare", "reference", "case"]
        assert transport.services.calls == ["NONE", "F1"] and transport.services.releases == 1
        state = data["runtime"].folder / "brev-host" / data["operation"]
        authorization = J._read(state / "authorization.json")
        ledger = C.S.M.RegistryLedger(Path(authorization["path"]).parent / "probes.ledger.jsonl", C.RC.registry())
        assert next(iter(ledger.requests().values()))["state"] == "done"
        projected = J._read(state / "NONE.projection.json")
        assert projected["original_root_map_sha256"] != projected["host_roots_sha256"]
        before = len(data["model"].calls)
        after = host(data, transport).advance(complete=data["model"])
        assert after["status"] == "hold" and len(data["model"].calls) == before + 1
        assert after["events"][-1]["latest_probe_followup"]["probe"] == "P_episode"
        assert after["events"][-1]["receipt_fact_ids_cited"]
        assert transport.services.calls == ["NONE", "F1"]
    finally:
        transport.close()


def test_approval_file_after_interruption_is_recovered_without_new_approval(case, tmp_path, monkeypatch):  # noqa: F811
    data = setup(tmp_path, case)
    transport = Transport(data)
    original = R.same_or_new

    def interrupted(path, value):
        if path.name == "authorization.json":
            raise KeyboardInterrupt("CPU after C AUTH before host acknowledgement")
        return original(path, value)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(R, "same_or_new", interrupted)
            with pytest.raises(KeyboardInterrupt):
                host(data, transport).advance(complete=no_model)
        assert transport.calls == ["prepare", "reference"] and transport.services.calls == ["NONE"]
        resumed = host(data, transport).advance(complete=no_model)
        assert resumed["status"] == "active", resumed
        assert transport.calls.count("reference") == transport.calls.count("case") == 1
        ledger = C.S.M.RegistryLedger(Path(data["intent"]["snapshot"]["inputs"]["ledger_path"]), C.RC.registry())
        assert [row["event"] for row in ledger.events()] == ["proposed", "approved", "consumed", "receipt"]
    finally:
        transport.close()


@pytest.mark.parametrize("mode", ["pending", "interrupted", "completed"])
def test_case_pending_interruption_and_completed_restart_are_lookup_only(case, tmp_path, mode):  # noqa: F811
    data = setup(tmp_path, case)
    transport = Transport(data)
    transport.pending = mode == "pending"
    transport.interrupt_case = mode == "interrupted"
    try:
        if mode == "interrupted":
            with pytest.raises(KeyboardInterrupt):
                host(data, transport).advance(complete=no_model)
        else:
            first = host(data, transport).advance(complete=no_model)
            assert first["status"] == ("blocked_host" if mode == "pending" else "active"), first
        if mode == "pending":
            assert host(data, transport).advance(complete=no_model)["status"] == "blocked_host"
        transport.pending = transport.interrupt_case = False
        if mode == "completed":
            # 완료 이력 뒤 다음 판단만 진행하며 GPU/C 승인 callback은 재호출하지 않는다.
            response = host(data, transport).advance(complete=data["model"])
            assert response["status"] == "hold", response
        else:
            response = host(data, transport).advance(complete=no_model)
            assert response["status"] == "active", response
        assert transport.services.calls == ["NONE", "F1"]
        assert transport.calls.count("reference") == transport.calls.count("case") == 1
        assert len(C.S.M.RegistryLedger(Path(data["intent"]["snapshot"]["inputs"]["ledger_path"]), C.RC.registry()).requests()) == 1
    finally:
        transport.close()


def test_interrupted_approve_without_C_auth_never_mints_approval_in_lookup(case, tmp_path, monkeypatch):  # noqa: F811
    data = setup(tmp_path, case)
    transport = Transport(data)
    original = J._write_new

    def interrupted(path, value):
        if path.name == "case-approval-intent.json":
            raise KeyboardInterrupt("CPU before C approval")
        return original(path, value)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(J, "_write_new", interrupted)
            with pytest.raises(KeyboardInterrupt):
                host(data, transport).advance(complete=no_model)
        pending = host(data, transport).advance(complete=no_model)
        assert pending["status"] == "blocked_host", pending
        ledger = C.S.M.RegistryLedger(Path(data["intent"]["snapshot"]["inputs"]["ledger_path"]), C.RC.registry())
        assert ledger.events() == [] and transport.services.calls == ["NONE"]
        # 별도 명시 settle은 새 NONE 실행 없이 이미 검증된 reference에서만 승인한다.
        instance = host(data, transport)
        with instance:
            assert instance.settle_approval(data["operation"], data["intent"])["status"] == "approved"
        assert host(data, transport).advance(complete=no_model)["status"] == "active"
        assert transport.services.calls == ["NONE", "F1"] and len(ledger.requests()) == 1
    finally:
        transport.close()


@pytest.mark.parametrize("mismatch", ["operation", "args", "checkpoint"])
def test_registered_pair_mismatch_rejected_before_any_transport_or_C_write(case, tmp_path, mismatch):  # noqa: F811
    data = setup(tmp_path, case, mismatch=mismatch)
    transport = Transport(data)
    with pytest.raises(ValueError):
        host(data, transport).advance(complete=no_model)
    assert transport.calls == []
    assert C.S.M.RegistryLedger(Path(data["intent"]["snapshot"]["inputs"]["ledger_path"]), C.RC.registry()).events() == []


@pytest.mark.parametrize("problem", ["raw_checkpoint", "operation"])
def test_bad_reference_never_reaches_C_authorization(case, tmp_path, problem):  # noqa: F811
    data = setup(tmp_path, case)
    transport = Transport(data)
    transport.bad_reference = problem == "raw_checkpoint"
    transport.wrong_operation = problem == "operation"
    try:
        with pytest.raises(ValueError):
            host(data, transport).advance(complete=no_model)
        assert "case" not in transport.calls
        assert C.S.M.RegistryLedger(Path(data["intent"]["snapshot"]["inputs"]["ledger_path"]), C.RC.registry()).events() == []
    finally:
        transport.close()


def test_completed_archive_tamper_is_rejected_on_new_host(case, tmp_path):  # noqa: F811
    data = setup(tmp_path, case)
    transport = Transport(data)
    try:
        assert host(data, transport).advance(complete=no_model)["status"] == "active"
        case_path = Path(data["producer"]["binding"]["expected_request"]["execution"]["operations"]["F1"]["host_stage"]) / "report.json"
        case_path.write_bytes(case_path.read_bytes() + b" ")
        with pytest.raises(ValueError):
            host(data, transport).advance(complete=no_model)
        assert transport.services.calls == ["NONE", "F1"] and transport.calls.count("case") == 1
    finally:
        transport.close()


@pytest.mark.parametrize("problem", ["fixture_kind", "contract", "source"])
def test_transport_kind_contract_and_source_cannot_bypass_registration(case, tmp_path, monkeypatch, problem):  # noqa: F811
    data = setup(tmp_path, case)
    transport = Transport(data)
    if problem == "fixture_kind":
        monkeypatch.setattr(Transport, "EVIDENCE_KIND", E.ACTUAL)
    elif problem == "contract":
        transport.contract_ref = {**transport.contract_ref, "sha256": "0" * 64}
    else:
        path = data["repo"] / "evals/agentic_brev_runtime_host_v1.py"
        path.write_bytes(path.read_bytes() + b"\n# changed fixture source after registration\n")
    with pytest.raises(ValueError):
        host(data, transport)
    assert transport.calls == []


def test_repair_unsupported_has_no_ledger_or_transport_side_effect(case, tmp_path):  # noqa: F811
    data = setup(tmp_path, case)
    transport = Transport(data)
    instance = host(data, transport)
    intent = {"action": "propose_correction"}
    with instance:
        for result in (instance.approve("unused", intent), instance.dispatch("unused", intent, {}),
                       instance.lookup("unused", intent), instance.settle_approval("unused", intent)):
            assert result == {"status": "hold", "reason": "repair_unsupported"}
    assert transport.calls == [] and not instance.root.exists()


def test_partial_C_proposal_cannot_be_approved_by_lookup_or_settle(case, tmp_path, monkeypatch):  # noqa: F811
    data = setup(tmp_path, case)
    transport = Transport(data)

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt("CPU after C proposed before approved")

    try:
        with monkeypatch.context() as patch:
            patch.setattr(C.S.M.RegistryLedger, "approve", interrupted)
            with pytest.raises(KeyboardInterrupt):
                host(data, transport).advance(complete=no_model)
        ledger = C.S.M.RegistryLedger(Path(data["intent"]["snapshot"]["inputs"]["ledger_path"]), C.RC.registry())
        before = ledger.events()
        assert [row["event"] for row in before] == ["proposed"]
        assert host(data, transport).advance(complete=no_model)["status"] == "blocked_host"
        instance = host(data, transport)
        with instance:
            assert instance.settle_approval(data["operation"], data["intent"])["status"] == "unknown"
        assert ledger.events() == before and transport.services.calls == ["NONE"]
        assert transport.calls.count("reference") == 1 and "case" not in transport.calls
    finally:
        transport.close()


def test_new_case_GPU_UUID_is_rejected_without_case_execution_or_C_receipt(case, tmp_path):  # noqa: F811
    data = setup(tmp_path, case)
    transport = Transport(data)
    transport.services.fail = "gpu"
    try:
        with pytest.raises(ValueError, match="^current boot resource/GPU changed$"):
            host(data, transport).advance(complete=no_model)
        ledger = C.S.M.RegistryLedger(Path(data["intent"]["snapshot"]["inputs"]["ledger_path"]), C.RC.registry())
        assert [row["event"] for row in ledger.events()] == ["proposed", "approved", "consumed"]
        assert transport.services.calls == ["NONE"] and not transport._outcome("F1").exists()
        assert not list((data["runtime"].folder / "brev-host").glob("*/completed.json"))
    finally:
        transport.close()


def test_runtime_dispatch_marker_without_host_intent_stays_lookup_only(case, tmp_path, monkeypatch):  # noqa: F811
    data = setup(tmp_path, case)
    transport = Transport(data)
    original = J._write_new

    def interrupted(path, value):
        if path.name == "case-dispatch-intent.json":
            raise KeyboardInterrupt("CPU before host dispatch intent")
        return original(path, value)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(J, "_write_new", interrupted)
            with pytest.raises(KeyboardInterrupt):
                host(data, transport).advance(complete=no_model)
        ledger = C.S.M.RegistryLedger(Path(data["intent"]["snapshot"]["inputs"]["ledger_path"]), C.RC.registry())
        before = ledger.events()
        result = host(data, transport).advance(complete=no_model)
        assert result["status"] == "blocked_host" and result["pending"]["host_status"] == "unknown"
        event = J._read(data["runtime"].folder / "steps/0002/host-events/0003.json")
        assert event["phase"] == "lookup_dispatch" and event["response"] == {
            "status": "unknown", "reason": "runtime_dispatch_without_host_intent_requires_settle"}
        assert ledger.events() == before and [row["event"] for row in before] == ["proposed", "approved", "consumed"]
        assert transport.services.calls == ["NONE"] and "case" not in transport.calls
    finally:
        transport.close()
