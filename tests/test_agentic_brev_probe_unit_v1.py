"""CPU fakeprocess로 새 단일 probe unit의 권한·원문·중복 실행 경계를 확인한다."""
from __future__ import annotations

import shutil
import sys
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / p) for p in ("bench", "evals", "tests", "src")]
import run_agentic_brev_probe_unit_v1 as U
from test_agentic_brev_probe_evidence_v1 import fixture
from test_agentic_investigation_runtime_v1 import commit_objects
from test_selection_comparison_controller_v1 import VIEW_SHA

E, C = U.E, U.C
NOW = datetime(2026, 10, 28, 0, 0, 10, tzinfo=timezone.utc)  # noqa: UP017 - Python 3.10


def save(path, value):
    E.write_new(path, value)
    return U.ref(path)


def completed_reference(root):
    """실행 전에 존재하는 NONE CPU 원문을 E13으로 검증한다. actual 권한으로 사용할 수 없다."""
    request, roots = fixture(root)
    request.update(side="reference", arm="NONE")
    for relative in ("source/adapter_artifact.json", "source/training.manifest.json", "stage/job.json"):
        path = root / relative
        data = E.loads(path.read_text(encoding="utf-8"))
        data["arm"] = "NONE"
        path.write_bytes(E.J._bytes(data))
    jobs = root / "stage/report.jobs.json"
    data = E.loads(jobs.read_text(encoding="utf-8"))
    data["fault"] = "NONE"
    jobs.write_bytes(E.J._bytes(data))
    for key, relative in (("artifact", "source/adapter_artifact.json"), ("manifest", "source/training.manifest.json")):
        request["source"][key]["sha256"] = E.J.file_sha(root / relative)
    for key, relative in (("job", "stage/job.json"), ("report_jobs", "stage/report.jobs.json")):
        request["files"][key]["sha256"] = E.J.file_sha(root / relative)
    evidence = E.build(request, roots, expected_request_sha256=E.digest(request),
        expected_roots_sha256=E.digest(roots), allow_fixture=True)
    E.verify(evidence, roots, expected_request_sha256=E.digest(request),
        expected_roots_sha256=E.digest(roots), allow_fixture=True)
    return save(root / "completion.json", evidence), evidence


def world(root, *, probe="P_episode", side="case"):
    template = root / "template"
    request, roots = fixture(template, probe)
    home = root / "home"
    repo, lab, python = home / "repo", home / "lab", home / "venv/bin/python"
    lab.mkdir(parents=True)
    python.parent.mkdir(parents=True)
    python.write_bytes(b"cpu-not-executable")
    identity = {k: request[k] for k in ("side", "arm", "seed", "probe", "run_name")}
    if side == "reference":
        identity.update(side="reference", arm="NONE")
        for name in ("adapter_artifact.json", "training.manifest.json"):
            path = template / "source" / name
            value = E.loads(path.read_text(encoding="utf-8"))
            value["arm"] = "NONE"
            path.write_bytes(E.J._bytes(value))
        request["source"]["artifact"]["sha256"] = E.J.file_sha(template / "source/adapter_artifact.json")
        request["source"]["manifest"]["sha256"] = E.J.file_sha(template / "source/training.manifest.json")
    backend = {"schema": "cpu_backend_fixture", **{k: request["expected_source"][k] for k in E.HASHES},
               "environment": request["expected_source"]["environment"]}
    backend_bytes = E.J._bytes(backend)
    backend_path = template / "control/bench/protocols/probe.json"
    backend_path.write_bytes(backend_bytes)
    request["expected_source"]["protocol"]["sha256_lf"] = E.J.file_sha(backend_path)
    for name in ("job.json", "execution.json", "backend_result.json"):
        path = template / "stage" / name
        value = E.loads(path.read_text(encoding="utf-8"))
        if name == "job.json":
            value.update(protocol_sha256_lf=E.J.file_sha(backend_path), arm=identity["arm"])
        elif name == "execution.json":
            value.update(source_before=request["expected_source"], source_after=deepcopy(request["expected_source"]))
        else:
            value["execution"].update(source_before=request["expected_source"], source_after=deepcopy(request["expected_source"]))
        path.write_bytes(E.J._bytes(value))
    if side == "reference":
        jobs = template / "stage/report.jobs.json"
        data = E.loads(jobs.read_text(encoding="utf-8"))
        data["fault"] = "NONE"
        jobs.write_bytes(E.J._bytes(data))
    lease = save(template / "control/lease.json", {"schema": "cpu_lease_fixture", "authority": False})
    capture = save(template / "control/capture-parent.json", {"schema": "cpu_capture_fixture", "authority": False})
    basis = E.J._bytes({"schema": "cpu_capture_basis_fixture"})
    basis_sha = E.J.digest({"schema": "cpu_capture_basis_fixture"})
    unit = repo / "units" / request["operation_id"]
    stage = unit / "raw"
    roots[0]["local"] = str(stage.resolve())
    source = request["source"]
    allowed = {"arm": identity["arm"], "seed": 2221, "checkpoint_sha256": source["checkpoint"]["sha256"],
               "telemetry_sha256": source["telemetry"]["sha256"], "lineage_id": source["lineage_id"]}
    protocol = {"schema": U.SCHEMA + "_protocol", "registered_before_execution": True,
        "layout": {"repo": "repo", "isaaclab": "lab", "python": "venv/bin/python"},
        "deadline_utc": "2026-10-28T01:00:00Z", "execution_root": "units", "allowed_probes": [probe],
        "inputs": {side: allowed}, "max_wall_s": 300, "resource_id": request["resource"]["resource_id"],
        "gpu_uuid": request["resource"]["gpu_uuid"], "lease_usage": "physical_mutex_only_not_probe_authority",
        "lease_basis": {"path": "bench/protocols/capture-basis.json", "sha256": basis_sha},
        "backend": {"protocol": {"path": "bench/protocols/probe.json", "sha256": E.J.file_sha(backend_path)},
                    "source_anchor": request["expected_source"]["source_anchor"]}}
    sources = {name: (ROOT / name).read_bytes() for name in U.SOURCES}
    protocol["sources_lf"] = {name: C.digest_lf_file(ROOT / name) for name in sources}
    scope_protocol = {"schema": "agentic_investigation_runtime_v1_protocol",
        "controller_extensions": [E.GPU_CONTRACT], "sources_lf": protocol["sources_lf"]}
    anchor = commit_objects(repo, {**sources, "bench/protocols/probe.json": backend_bytes,
        "bench/protocols/capture-basis.json": basis, "bench/protocols/unit.json": E.J._bytes(protocol),
        "bench/protocols/scope.json": E.J._bytes(scope_protocol)})
    source_registration = {"protocol": U.ref(repo / "bench/protocols/scope.json"),
                           "anchor": anchor, "repo": str(repo)}
    protocol_ref = U.ref(repo / "bench/protocols/unit.json")
    (template / "control/execution.json").write_bytes((repo / "bench/protocols/unit.json").read_bytes())
    request["resource"]["protocol"]["sha256"] = protocol_ref["sha256"]
    request["resource"]["protocol_sha256"] = protocol_ref["sha256"]
    proof_path = template / "control/resource.json"
    proof = E.loads(proof_path.read_text(encoding="utf-8"))
    proof.update(protocol_sha256=protocol_ref["sha256"], operation_id=request["operation_id"],
        lease_basis_sha256=basis_sha, lease_snapshot_sha256=lease["sha256"], capture_parent_sha256=capture["sha256"])
    proof_path.write_bytes(E.J._bytes(proof))
    request["resource"]["proof"]["sha256"] = E.J.file_sha(proof_path)
    plan = {"schema": U.SCHEMA + "_plan", "action": "probe", "evidence_kind": E.FIXTURE,
        "protocol": protocol_ref, "source_anchor": anchor, "identity": identity,
        "operation_id": request["operation_id"], "stage": str(stage), "logical_stage": "/stage",
        "source": source, "roots": roots, "expected_source": request["expected_source"],
        "resource": request["resource"], "lease_basis": protocol["lease_basis"],
        "lease_snapshot": lease, "capture_parent": capture, "source_registration": source_registration}
    reference_ref, reference = completed_reference(root / "completed-reference")
    arms = root / "approvals"
    ledger_path = arms / "model_adaptive/epoch1/probes.ledger.jsonl"
    packet = C.S.build_packet(epoch=1, applied_corrections=[], prior_receipt=None,
        facts={"config:action_scale": 0.25}, input_bindings={"case": {
            "checkpoint_sha256": source["checkpoint"]["sha256"], "telemetry_sha256": source["telemetry"]["sha256"]},
            "reference": {"checkpoint_sha256": reference["checkpoint_sha256"],
                          "telemetry_sha256": reference["request"]["source"]["telemetry"]["sha256"]}}, execution_budget_s=1200)
    snapshot = C.V.build_snapshot(packet, "unit_fixture", "model_adaptive", VIEW_SHA, ledger_path)
    import agentic_investigation_contract_v1 as A
    context = A.prepare(snapshot, goal="CPU")
    review = A.assess(E.J._bytes({"action": "run_probe", "target": probe, "fact_ids": ["config:action_scale"]}).decode(), context)
    choice = A.probe_choice(context, review)
    spec = C.request_spec(snapshot, choice)
    execution_plan = {"stage": str(stage), "checkpoint_sha256": source["checkpoint"]["sha256"],
        "seed": 2221, "run_name": identity["run_name"], "args": spec["args"], "gpu_contract": E.GPU_CONTRACT,
        "reference": {"evidence_path": reference_ref["path"], "evidence_sha256": reference_ref["sha256"],
            "checkpoint_sha256": reference["checkpoint_sha256"], "seed": 2221,
            "stage": str(root / "completed-reference/stage"), "run_name": reference["run_name"],
            "gpu_contract": E.GPU_CONTRACT}}
    with U.S.BrevProbeScope(allow_fixture=True, source_registration=source_registration):
        auth = C.authorize_probe(arms, "model_adaptive", 1, snapshot, choice, "cpu_explicit",
            evidence_kind=C.FIXTURE_EVIDENCE, execution_plan=execution_plan)
    ledger = C.S.M.RegistryLedger(ledger_path, C.RC.registry())
    plan["authorization_prefix_sha256"] = E.digest([row for row in ledger.events()
        if row.get("request_id") == auth["request_id"]])
    plan_ref = save(root / "plan.json", plan)
    return {"root": root, "home": home, "template": template, "plan": plan, "plan_ref": plan_ref,
            "auth_ref": {key: auth[key] for key in ("path", "sha256")}, "stage": stage, "unit": unit,
            "ledger": ledger_path, "reference_ref": reference_ref}


class FakeProcess:
    def __init__(self, world, *, cleanup=True, interrupt=False):
        self.world, self.cleanup, self.interrupt = world, cleanup, interrupt
        self.waits = 0

    def wait(self):
        self.waits += 1
        stage = self.world["stage"]
        stage.mkdir(parents=True)
        shutil.copyfile(self.world["template"] / "stage/job.json", stage / "job.json")
        if self.interrupt:
            raise KeyboardInterrupt("fixture interrupted dispatch")
        for path in (self.world["template"] / "stage").iterdir():
            if path.name not in {"backend_result.json", "job.json"}:
                shutil.copyfile(path, stage / path.name)
        if not self.cleanup:
            path = stage / "execution.json"
            data = E.loads(path.read_text(encoding="utf-8"))
            data["cleanup_verified"] = False
            path.write_bytes(E.J._bytes(data))


class FixtureServices:
    def __init__(self, world, *, cleanup=True, interrupt=False, lost_lease=False):
        self.world = world
        self.process = FakeProcess(world, cleanup=cleanup, interrupt=interrupt)
        self.calls, self.checks, self.lost_lease = 0, 0, lost_lease

    @contextmanager
    def resource(self, _prepared):
        def check():
            self.checks += 1
            if self.lost_lease and self.checks > 1:
                raise U.JOB.LeaseError("fixture parent lock lost")
        yield {"check": check, "reader": list}

    def backend(self, _prepared, _live):
        owner = self
        class Backend:
            def probe(self, stage, artifact, probe, run_name):
                owner.calls += 1
                assert stage == owner.world["stage"]
                assert artifact["checkpoint"]["sha256"] == owner.world["plan"]["source"]["checkpoint"]["sha256"]
                assert probe == owner.world["plan"]["identity"]["probe"]
                assert run_name == owner.world["plan"]["identity"]["run_name"]
                owner.process.wait()
                execution = E.loads((stage / "execution.json").read_text(encoding="utf-8"))
                raw = E.loads((owner.world["template"] / "stage/backend_result.json").read_text(encoding="utf-8"))
                raw["execution"] = execution
                return raw
        return Backend()


def run(w, services=None, *, execute=True, authorization=True):
    return U.run(w["plan_ref"], w["auth_ref"] if authorization else None, execute=execute, allow_fixture=True,
        home=w["home"], now=NOW, backend_factory=services.backend if services else None,
        resource_service=services.resource if services else None)


def test_prepare_is_readonly_and_no_approval_never_dispatches(tmp_path):
    w = world(tmp_path)
    assert run(w, execute=False)["status"] == "prepared"
    assert run(w, authorization=False)["status"] == "approval_required"
    assert not w["unit"].exists()
    prepared = U.prepare(w["plan_ref"], w["auth_ref"], allow_fixture=True, home=w["home"], now=NOW)
    assert prepared["python"] == w["home"] / "venv/bin/python"


@pytest.mark.parametrize("side", ["case", "reference"])
def test_execute_fakeprocess_seals_e13_and_duplicate_is_lookup_only(tmp_path, side):
    w = world(tmp_path, side=side)
    services = FixtureServices(w)
    result = run(w, services)
    assert result["status"] == "complete", result
    assert run(w, services) == result
    assert services.calls == services.process.waits == 1
    evidence = U.read_ref(result["evidence"])
    assert evidence["side"] == side and evidence["probe"] == "P_episode"
    assert not result["approval_granted_by_unit"]


@pytest.mark.parametrize("failure", ["cleanup", "interrupt", "lease"])
def test_incomplete_dispatch_preserves_raw_and_never_resubmits(tmp_path, failure):
    w = world(tmp_path)
    services = FixtureServices(w, cleanup=failure != "cleanup", interrupt=failure == "interrupt", lost_lease=failure == "lease")
    result = run(w, services)
    assert result["status"] == ("pending_unknown" if failure == "interrupt" else "HOLD"), result
    assert not (w["unit"] / "evidence.json").exists()
    assert (w["unit"] / "dispatch-intent.json").is_file()
    assert run(w, services) == result and services.calls == 1
    assert list(w["stage"].iterdir())


@pytest.mark.parametrize("field", ["checkpoint_sha256", "source", "auth", "plan_sha"])
def test_wrong_cp_source_approval_or_plan_rejected_before_dispatch(tmp_path, field):
    w = world(tmp_path)
    if field == "plan_sha":
        w["plan_ref"]["sha256"] = "0" * 64
    elif field == "auth":
        w["auth_ref"]["sha256"] = "0" * 64
    else:
        data = E.loads(Path(w["plan_ref"]["path"]).read_text(encoding="utf-8"))
        if field == "source":
            data["source"]["lineage_id"] = "other-lineage"
        else:
            data["source"]["checkpoint"]["sha256"] = "9" * 64
        w["plan_ref"] = save(tmp_path / "wrong-plan.json", data)
    services = FixtureServices(w)
    with pytest.raises(ValueError):
        run(w, services)
    assert services.calls == 0 and not w["unit"].exists()


def test_repair_is_hold_before_any_approval_or_backend(tmp_path):
    w = world(tmp_path)
    plan = deepcopy(w["plan"])
    plan["action"] = "repair"
    w["plan_ref"] = save(tmp_path / "repair-plan.json", plan)
    services = FixtureServices(w)
    assert run(w, services, authorization=False)["reason"] == "repair_unsupported"
    assert services.calls == services.checks == 0 and not w["unit"].exists()


def test_fixture_cannot_use_default_service_or_actual_override(tmp_path):
    w = world(tmp_path)
    with pytest.raises(ValueError, match="fixture mode"):
        U.run(w["plan_ref"], w["auth_ref"])
    with pytest.raises(ValueError, match="explicit fakeprocess"):
        run(w)
    plan = deepcopy(w["plan"])
    plan["evidence_kind"] = E.ACTUAL
    w["plan_ref"] = save(tmp_path / "actual-plan.json", plan)
    with pytest.raises(ValueError, match="HOME override"):
        run(w, FixtureServices(w))


def test_existing_partial_stage_is_pending_and_never_dispatched(tmp_path):
    w = world(tmp_path)
    w["stage"].mkdir(parents=True)
    (w["stage"] / "partial.stdout").write_bytes(b"original partial raw")
    services = FixtureServices(w)
    result = run(w, services)
    assert result["status"] == "pending_unknown" and result["redispatch"] is False
    assert services.calls == 0 and (w["stage"] / "partial.stdout").read_bytes() == b"original partial raw"


@pytest.mark.parametrize("source", ["bench/run_agentic_brev_probe_unit_v1.py",
    "src/rl_triage/mechanism_registry.py", "src/rl_triage/probe_loop.py"])
def test_registered_source_change_after_commit_fails_before_dispatch(tmp_path, source):
    w = world(tmp_path)
    path = w["home"] / "repo" / source
    path.write_bytes(path.read_bytes() + b"\n# changed after public registration\n")
    services = FixtureServices(w)
    with pytest.raises(ValueError, match="source/blob"):
        run(w, services)
    assert services.calls == 0 and not w["unit"].exists()


def test_false_host_resource_grant_rejected_before_dispatch(tmp_path):
    w = world(tmp_path)
    path = w["template"] / "control/resource.json"
    body = E.loads(path.read_text(encoding="utf-8"))
    body["status"] = "pending"
    path.write_bytes(E.J._bytes(body))
    plan = deepcopy(w["plan"])
    plan["resource"]["proof"]["sha256"] = E.J.file_sha(path)
    w["plan_ref"] = save(tmp_path / "pending-resource-plan.json", plan)
    services = FixtureServices(w)
    with pytest.raises(ValueError, match="host resource grant"):
        run(w, services)
    assert services.calls == 0 and not w["unit"].exists()


def test_actual_parent_service_rejects_non_posix_lock_proof(tmp_path, monkeypatch):
    w = world(tmp_path)
    prepared = U.prepare(w["plan_ref"], w["auth_ref"], allow_fixture=True, home=w["home"], now=NOW)
    monkeypatch.setattr(U.JOB, "validate_protocol", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(U.JOB, "validate_resource_proof", lambda *_args, **_kwargs: {
        "lease": w["plan"]["lease_snapshot"], "_lease": {"live_lock_proof": False}})
    with pytest.raises(ValueError, match="actual POSIX parent lock proof"), U.parent_resource(prepared):
        pytest.fail("fixture OS lock must not be actual permission")


@pytest.mark.parametrize("change", ["missing", "proposed", "approved", "other_request", "prefix", "registry"])
def test_authorization_requires_exact_host_ledger_before_dispatch(tmp_path, change):
    w = world(tmp_path)
    rows = [E.loads(line) for line in w["ledger"].read_text(encoding="utf-8").splitlines()]
    if change == "missing":
        w["ledger"].unlink()
    elif change == "prefix":
        w["plan"]["authorization_prefix_sha256"] = "0" * 64
        w["plan_ref"] = save(tmp_path / "wrong-prefix-plan.json", w["plan"])
    else:
        if change in {"proposed", "approved"}:
            rows = rows[:1 if change == "proposed" else 2]
        elif change == "other_request":
            for row in rows:
                row["request_id"] = "another-host-request"
        else:
            rows[0]["registry_digest"] = "0" * 64
            w["plan"]["authorization_prefix_sha256"] = E.digest(rows)
            w["plan_ref"] = save(tmp_path / "wrong-registry-plan.json", w["plan"])
        w["ledger"].write_bytes(b"".join(E.J._bytes(row) for row in rows))
    services = FixtureServices(w)
    with pytest.raises(C.L.LedgerError if change == "registry" else ValueError):
        run(w, services)
    assert services.calls == 0 and not w["unit"].exists()


@pytest.mark.parametrize("change", ["arm", "epoch", "request_id", "reference_sha", "foreign_posix_stage"])
def test_changed_auth_keeps_native_controller_path_and_reference_guards(tmp_path, change):
    w = world(tmp_path)
    auth = U.read_ref(w["auth_ref"])
    if change == "reference_sha":
        auth["execution_plan"]["reference"]["evidence_sha256"] = "0" * 64
    elif change == "foreign_posix_stage":
        auth["execution_plan"]["stage"] = "/foreign/posix/stage"
    else:
        auth[change] = "foreign" if change != "epoch" else 2
    path = Path(w["auth_ref"]["path"]).with_name("authorization_tampered.json")
    w["auth_ref"] = save(path, auth)
    services = FixtureServices(w)
    with pytest.raises(ValueError):
        run(w, services)
    assert services.calls == 0 and not w["unit"].exists()


def test_consumed_to_done_allows_only_exact_complete_lookup(tmp_path):
    w = world(tmp_path)
    services = FixtureServices(w)
    complete = run(w, services)
    assert complete["status"] == "complete", complete
    auth = U.read_ref(w["auth_ref"])
    ledger = C.S.M.RegistryLedger(w["ledger"], C.RC.registry())
    ledger.receipt(auth["request_id"], "normal", {"cpu": True}, 10, 0)
    assert run(w, services) == complete and services.calls == 1
    assert run(w, services, authorization=False) == complete and services.calls == 1
    # 소비 완료 원장은 새 raw stage로의 실행을 승인하지 않는다.
    fresh = world(tmp_path / "fresh")
    fresh_auth = U.read_ref(fresh["auth_ref"])
    C.S.M.RegistryLedger(fresh["ledger"], C.RC.registry()).receipt(
        fresh_auth["request_id"], "normal", {"cpu": True}, 10, 0)
    fresh_services = FixtureServices(fresh)
    with pytest.raises(ValueError, match="done authorization cannot dispatch"):
        run(fresh, fresh_services)
    assert fresh_services.calls == 0 and not fresh["unit"].exists()


@pytest.mark.parametrize("field", ["repo", "anchor", "protocol"])
def test_authorization_registration_is_not_a_fixture_bypass(tmp_path, field):
    w = world(tmp_path)
    plan = deepcopy(w["plan"])
    registration = plan["source_registration"]
    if field == "repo":
        registration[field] = str(tmp_path)
    elif field == "anchor":
        registration[field] = "0" * 40
    else:
        registration[field]["sha256"] = "0" * 64
    w["plan_ref"] = save(tmp_path / "wrong-registration-plan.json", plan)
    services = FixtureServices(w)
    with pytest.raises(ValueError):
        run(w, services)
    assert services.calls == 0 and not w["unit"].exists()


def test_ledger_change_while_acquiring_is_rejected_before_backend(tmp_path):
    w = world(tmp_path)
    services = FixtureServices(w)

    @contextmanager
    def changed(prepared):
        w["ledger"].write_bytes(b"")
        with services.resource(prepared) as live:
            yield live

    result = U.run(w["plan_ref"], w["auth_ref"], execute=True, allow_fixture=True,
        home=w["home"], now=NOW, backend_factory=services.backend, resource_service=changed)
    assert result["status"] == "HOLD" and result["dispatched"] is False
    assert services.calls == 0 and not (w["unit"] / "dispatch-intent.json").exists()


def test_reference_absent_blocks_controller_validation_without_placeholder(tmp_path):
    w = world(tmp_path)
    Path(w["reference_ref"]["path"]).unlink()
    before = w["ledger"].read_bytes()
    services = FixtureServices(w)
    with pytest.raises(ValueError, match="reference execution plan"):
        run(w, services)
    assert w["ledger"].read_bytes() == before
    assert services.calls == 0 and not w["unit"].exists()


def test_owned_process_preserves_group_flags_and_drops_api_keys(monkeypatch):
    seen = []
    sentinel = object()
    monkeypatch.setattr(U.subprocess, "Popen", lambda args, **kwargs: (seen.append((args, kwargs)), sentinel)[1])
    result = U._owned_popen(["registered-python", "-B", "registered-worker"], start_new_session=True,
        env={"NVIDIA_API_KEY": "fixture", "OPENAI_API_KEY": "fixture", "CUDA_VISIBLE_DEVICES": "0"})
    assert result is sentinel and seen[0][1]["start_new_session"] is True
    assert seen[0][1]["env"] == {"CUDA_VISIBLE_DEVICES": "0"}
