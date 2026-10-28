"""실제 pair ledger 함수와 fakeprocess로 부모 acquire→증거→해제 CPU 경계를 검사한다."""
from __future__ import annotations

import shutil
import subprocess
import sys
from contextlib import contextmanager
from copy import deepcopy
from datetime import timedelta
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / p) for p in ("bench", "evals", "src", "tests")]
import run_agentic_brev_probe_parent_v1 as P
from test_agentic_brev_probe_evidence_v1 import fixture
from test_agentic_brev_probe_unit_v1 import NOW, completed_reference, save
from test_agentic_investigation_runtime_v1 import commit_objects
from test_selection_comparison_controller_v1 import VIEW_SHA

E, C = P.E, P.C


def reference_request(root, probe):
    if probe == "P_episode":
        return completed_reference(root)[1]["request"]
    request, _roots = fixture(root, probe)
    request.update(side="reference", arm="NONE")
    for relative in ("source/adapter_artifact.json", "source/training.manifest.json", "stage/job.json"):
        path = root / relative
        data = E.J._read(path)
        data["arm"] = "NONE"
        path.write_bytes(E.J._bytes(data))
    jobs = root / "stage/report.jobs.json"
    data = E.J._read(jobs)
    data["fault"] = "NONE"
    jobs.write_bytes(E.J._bytes(data))
    for name, relative in (("artifact", "source/adapter_artifact.json"), ("manifest", "source/training.manifest.json")):
        request["source"][name]["sha256"] = E.J.file_sha(root / relative)
    return request


def world(root, *, probe="P_episode", model_probe=None, operation_probe=None, command_probe=None):
    home = root / "home"
    repo, lab, python = home / "repo", home / "lab", home / "venv/bin/python"
    lab.mkdir(parents=True)
    python.parent.mkdir(parents=True)
    python.write_bytes(b"fixture-not-executable")
    host, transport = root / "host", root / "transport"
    host.mkdir()
    transport.mkdir()
    request_case, roots_case = fixture(root / "F1", probe)
    request_none = reference_request(root / "NONE", probe)
    templates = {"F1": root / "F1", "NONE": root / "NONE"}
    requests = {"F1": request_case, "NONE": request_none}
    source_roots = {"F1": roots_case[1:], "NONE": [
        {"remote": "/" + key, "local": str((root / "NONE" / key).resolve())} for key in ("source", "control")]}
    expected = deepcopy(request_case["expected_source"])
    backend = {"schema": "cpu_backend_fixture", **{key: expected[key] for key in E.HASHES},
               "environment": expected["environment"]}
    backend_raw = E.J._bytes(backend)
    backend_sha = E.J.digest(backend)
    expected["protocol"]["sha256_lf"] = backend_sha
    for arm, template in templates.items():
        (template / "control/bench/protocols/probe.json").write_bytes(backend_raw)
        job = E.loads((template / "stage/job.json").read_text(encoding="utf-8"))
        job["protocol_sha256_lf"] = backend_sha
        (template / "stage/job.json").write_bytes(E.J._bytes(job))
        execution = E.loads((template / "stage/execution.json").read_text(encoding="utf-8"))
        execution.update(source_before=expected, source_after=deepcopy(expected))
        if arm == "F1":
            execution.update(started_at="2026-10-28T00:00:30Z", finished_at="2026-10-28T00:00:40Z")
        (template / "stage/execution.json").write_bytes(E.J._bytes(execution))
    snapshot_inputs = {"case": {"checkpoint_sha256": request_case["source"]["checkpoint"]["sha256"],
        "telemetry_sha256": request_case["source"]["telemetry"]["sha256"]},
        "reference": {"checkpoint_sha256": request_none["source"]["checkpoint"]["sha256"],
            "telemetry_sha256": request_none["source"]["telemetry"]["sha256"]}}
    packet = C.S.build_packet(epoch=1, applied_corrections=[], prior_receipt=None,
        facts={"config:action_scale": 0.25}, input_bindings=snapshot_inputs, execution_budget_s=1200)
    snapshot = C.V.build_snapshot(packet, "pair_cpu", "model_adaptive", VIEW_SHA, root / "not-consumed.ledger.jsonl")
    capsule_ref = P.CALL.freeze(P.CALL.A.prepare(snapshot, goal="CPU"), host / "call",
                               source_anchor="a" * 40, protocol_sha256="b" * 64)
    capsule = E.J._read(Path(capsule_ref["path"]))
    snapshot = capsule["context"]["snapshot"]
    content = E.J._bytes({"action": "run_probe", "target": model_probe or probe,
                         "fact_ids": ["config:action_scale"]}).decode()
    review = P.CALL.A.assess(content, capsule["context"])
    assert review["status"] == "valid"
    answer = P.CALL.advance(capsule_ref, host / "cpu-api", lambda *_: {
        "content": content, "finish_reason": "stop", "provider_model": P.CALL.O.MODEL, "wall_s": 0.01})
    assert answer["journal"]["status"] == "choice_ready"
    decision_ref = P.U.ref(host / "call/journal/terminal.json")
    snapshot_ref = save(host / "snapshot.json", snapshot)
    history_path = host / "call/history-binding-v1/history.jsonl"
    provider = E.loads((root / "F1/control/provider.json").read_text(encoding="utf-8"))
    provider_ref = save(host / "provider.json", provider)
    target = {"resource_id": "fixture-vm", "resource_name": "fixture", "instance_type": "fixture", "gpu_model": "L40S"}
    host_resource = {"schema": P.SCHEMA + "_host_resource", "status": "granted", "target": target,
        "gpu_binding": P.GPU_BINDING, "issued_at_utc": "2026-10-28T00:00:00Z", "expires_at_utc": "2026-10-28T01:00:00Z"}
    resource_ref = save(host / "host-resource.json", host_resource)
    sources = {name: (ROOT / name).read_bytes() for name in P.SOURCES}
    hashes = {name: sha256(raw.replace(b"\r\n", b"\n")).hexdigest() for name, raw in sources.items()}
    basis = {"schema": "cpu_lease_basis", "source_sha256_lf": {name: hashes[name] for name in P.JOB2.REQUIRED_SOURCE_FILES},
             "resources": {"deadline_utc": "2026-10-28T01:00:00Z"}}
    basis_sha = E.digest(basis)
    execution = {"schema": P.SCHEMA + "_execution", "pair_id": "a" * 64, "order": list(P.ORDER),
        "evidence_kind": E.FIXTURE,
        "unverified_child_policy": P.UNVERIFIED_CHILD_POLICY,
        "gpu_binding": P.GPU_BINDING, "source_closure_sha256": E.digest(hashes),
        "ledger_projection": {"host_path": "Q:\\authority\\pairs.ledger.jsonl", "remote_path": "/transport/consumed.jsonl"},
        "selection_capsule": {"path": "/host/call/capsule.json", "sha256": capsule_ref["sha256"]},
        "selection_result": {"path": "/host/call/journal/terminal.json", "sha256": decision_ref["sha256"]},
        "selection_manifest": {"path": "/host/call/journal/manifest.json", "sha256": E.J.file_sha(host / "call/journal/manifest.json")},
        "selection_calls": [{key: {"path": "/host/" + relative, "sha256": E.J.file_sha(host / relative)}
            for key, relative in (("event", "call/journal/events/0001.json"),
                ("reservation", "cpu-api/reservations/0001.json"), ("response", "cpu-api/responses/0001.json"))}],
        "snapshot": {"path": "/host/snapshot.json", "sha256": snapshot_ref["sha256"]},
        "snapshot_history_projection": {"schema": P.SCHEMA + "_snapshot_history_projection",
            "host_path": snapshot["inputs"]["ledger_path"], "host_base_capsule_path": capsule["base_reference"]["path"],
            "base_capsule": {"path": "/host/call/history-binding-v1/capsule.json", "sha256": capsule["base_reference"]["sha256"]},
            "history": {"path": "/host/call/history-binding-v1/history.jsonl", "sha256": E.J.file_sha(history_path)}},
        "host_resource": {"path": "/host/host-resource.json", "sha256": resource_ref["sha256"]},
        "lease_basis": {"path": "/registered/basis.json", "sha256": basis_sha},
        "source_protocol_sha256_lf": "d" * 64,
        "backend": {"protocol": {"path": "/registered/probe.json", "sha256": backend_sha},
            "source_anchor": expected["source_anchor"], "expected_source": expected},
        "windows": {"lease_ttl_s": 3600, "execution_window_s": 600, "max_wall_s": 300,
            "admission_max_age_s": P.JOB2.FRESH_CAPTURE_S, "admission_valid_s": 120, "provider_wait_s": 0.05},
        "operations": {}}
    commands = {}
    for arm, side, digit in (("NONE", "reference", "1"), ("F1", "case", "2")):
        operation = {"side": side, "arm": arm, "seed": 2221, "probe": operation_probe or probe, "operation_id": digit * 64,
            "run_name": "registered_probe", "host_stage": "Q:\\execution\\" + arm + "\\raw",
            "remote_stage": "/stage", "args": C.RC.registry()["probes"][probe]["args"],
            "source": requests[arm]["source"], "source_roots": source_roots[arm], "provider_packet_path": "/host/provider-" + arm + ".json"}
        execution["operations"][arm] = operation
        commands[arm] = {key: operation[key] for key in ("operation_id", "remote_stage", "run_name", "args")}
        commands[arm]["stage"] = operation["host_stage"]
        commands[arm]["probe"] = command_probe or probe
        stage = repo / "parents/operations" / operation["operation_id"] / "raw"
        commands[arm]["argv"] = [str(python), "-B", str(repo / "bench/brev_recovery_worker_v1.py"),
            "--job", str(stage / "job.json"), "--output", str(stage / "worker.json")]
        save(host / ("provider-" + arm + ".json"), {"schema": P.SCHEMA + "_provider_packet",
            "operation_id": operation["operation_id"], "observed_at_utc": "2026-10-28T00:00:00Z",
            "inventory": {"path": "/host/provider.json", "sha256": provider_ref["sha256"]}})
    pair = {"schema": P.PAIR.PAIR_SCHEMA, "group": "initial-s2221", "probe": probe, "seed": 2221,
        "arms": list(P.ORDER), "checkpoints": {arm: requests[arm]["source"]["checkpoint"]["sha256"] for arm in P.ORDER},
        "selection": {"terminal_sha256": decision_ref["sha256"], "activation_bridge_sha256": capsule_ref["sha256"]},
        "protocols": {"source": "d" * 64, "probe": backend_sha}, "execution": execution,
        "commands": commands, "commands_sha256": P.PAIR.digest(commands), "budget_gpu_s": 1200, "resource_sha256": resource_ref["sha256"]}
    binding = {"schema": P.SCHEMA + "_binding", "evidence_kind": E.FIXTURE, "expected_request": pair,
        "roots": [{"remote": "/host", "local": str(host.resolve())},
            {"remote": "/transport", "local": str(transport.resolve())},
            {"remote": "/registered", "local": str((repo / "bench/protocols").resolve())}]}
    binding_ref = save(root / "binding.json", binding)
    protocol = {"schema": P.SCHEMA + "_protocol", "registered_before_execution": True,
        "layout": {"repo": "repo", "isaaclab": "lab", "python": "venv/bin/python"}, "execution_root": "parents",
        "deadline_utc": "2026-10-28T01:00:00Z", "gpu_binding": P.GPU_BINDING,
        "binding_sha256": binding_ref["sha256"], "sources_lf": hashes, "provider_target": target, "allowed_probes": [probe]}
    anchor = commit_objects(repo, {**sources, "bench/protocols/pair.json": E.J._bytes(protocol),
        "bench/protocols/probe.json": backend_raw, "bench/protocols/basis.json": E.J._bytes(basis)})
    ledger = P.PairLedger(root / "host-original-pairs.ledger.jsonl")
    prefix = ledger.authorize(pair, "cpu_explicit")
    copy = transport / "consumed.jsonl"
    shutil.copyfile(ledger.path, copy)
    host_reference = {**prefix, "path": execution["ledger_projection"]["host_path"]}
    request = {"schema": P.SCHEMA + "_request", "action": "probe", "protocol": P.U.ref(repo / "bench/protocols/pair.json"),
        "source_anchor": anchor, "binding": binding_ref, "authority": {"schema": P.SCHEMA + "_authority",
            "host_reference": host_reference, "projection": execution["ledger_projection"]}}
    return {"root": root, "home": home, "repo": repo, "templates": templates, "binding": binding,
        "protocol": protocol, "request": request, "request_ref": save(root / "request.json", request),
        "ledger": ledger, "copy": copy, "folder": repo / "parents/pairs" / ("a" * 64), "provider_ref": provider_ref}


class Services:
    evidence_kind = E.FIXTURE

    def __init__(self, w, *, fail=None):
        self.w, self.fail = w, fail
        self.active, self.acquires, self.releases, self.calls, self.checks = False, 0, 0, [], 0
        self.current = NOW

    def clock(self):
        return self.current

    @contextmanager
    def scope(self):
        with P.JOB2.canonical_lease_scope() as root:
            yield root

    def observation(self):
        return {"schema": "cpu_canonical_observation", "actual_kernel_lock": False}

    def validate_basis(self, prepared):
        return prepared["basis"]

    @contextmanager
    def acquire(self, prepared):
        self.active = True
        self.acquires += 1
        ref = save(self.w["folder"] / "lease.json", {"schema": "cpu_fixture_lease", "resource_id": "fixture-vm",
            "lease_id": "fixture-physical-mutex", "protocol_sha256_lf": prepared["basis_sha"],
            "acquired_at_utc": "2026-10-28T00:00:05Z",
            "expires_at_utc": "2026-10-28T00:00:11Z" if self.fail == "short" else "2026-10-28T01:00:00Z"})
        try:
            yield ref
        finally:
            self.active = False
            self.releases += 1

    def verify(self, prepared, lease, basis):
        self.checks += 1
        if not self.active or self.fail == "inactive" or (self.fail == "lost" and self.calls):
            raise ValueError("fixture physical lock lost")
        return {"resource_id": "fixture-vm", "lease_id": "fixture-physical-mutex", "owner": {"pid": 123},
            "reference": lease, "evidence_kind": E.FIXTURE, "live_lock_proof": False,
            "lock_proof": "explicit_cpu_fixture", "heartbeat_sequence": self.checks}

    def record_lock(self, prepared, folder, lease, live):
        (folder / "cpu-lock.txt").write_bytes(b"explicit fixture; not OS evidence")

    def boot(self, path):
        path.write_bytes(b"changed-boot" if self.fail == "boot" and self.calls else b"cpu-current-boot")
        return P.raw_ref(path)

    def admission(self, prepared, folder, lease, basis, check, provider):
        assert self.active
        folder.mkdir(parents=True)
        operation_id = folder.parent.name
        arm = next(arm for arm in P.ORDER if prepared["operations"][arm]["op"]["operation_id"] == operation_id)
        template = self.w["templates"][arm]
        capture = E.loads((template / "control/resource.json").read_text(encoding="utf-8"))
        capture = {key: capture[key] for key in E.GPU.CAPTURE_KEYS}
        capture.update(provider_inventory_path="/host/provider.json", provider_inventory_sha256=self.w["provider_ref"]["sha256"])
        if self.fail == "resource":
            capture["resource_id"] = "other-vm"
        if self.fail == "gpu" and self.calls:
            capture["gpu_uuid"] = "GPU-other-boot"
        observed = self.current - timedelta(seconds=121) if self.fail == "stale" else self.current
        proof = {"schema": "cpu_fixture_capture", "lease": lease, "resource_id": capture["resource_id"],
            "gpu_uuid": capture["gpu_uuid"], "capture": capture, "observed_at_utc": observed.isoformat(),
            "expires_at_utc": (observed + timedelta(seconds=120)).isoformat()}
        lease_body = P.U.read_ref(lease)
        save(folder / "admission-times.json", {"schema": P.SCHEMA + "_admission_times", "lease": lease,
            "provider_observed_at_utc": provider["observed_at_utc"], "admission_at_utc": observed.isoformat(),
            "lease_acquired_at_utc": lease_body["acquired_at_utc"], "lease_expires_at_utc": lease_body["expires_at_utc"],
            "queries": [{"command": "explicit_cpu_fixture", "observed_at_utc": self.current.isoformat()}]})
        path = save(folder / "resource.json", proof)
        model = "wrong GPU" if self.fail == "model" else "NVIDIA L40S"
        (folder / "model.csv").write_text(capture["gpu_uuid"] + ", " + model + "\n", encoding="utf-8")
        return path, P.raw_ref(folder / "model.csv")

    def capture(self, prepared, capture_ref, basis):
        return P.U.read_ref(capture_ref)

    def reader(self, capture, folder):
        class Reader:
            last_receipt = None

            def __call__(self):
                return []
        return Reader()

    def release_observation(self, prepared, lease):
        assert not self.active
        return {"released": True, "actual_kernel_release_verified": False, "evidence_kind": E.FIXTURE, "lease": lease}

    def ensure_children_closed(self, prepared, lease):
        assert self.active

    def backend(self, prepared, live):
        owner = self
        arm = prepared["plan"]["identity"]["arm"]

        class Backend:
            def probe(self, stage, projected, probe, run_name):
                assert owner.active and probe == owner.w["binding"]["expected_request"]["probe"] and run_name == "registered_probe"
                live["reader"]()
                owner.calls.append(arm)
                stage.mkdir(parents=True)
                shutil.copyfile(owner.w["templates"][arm] / "stage/job.json", stage / "job.json")
                if owner.fail == "interrupt" and arm == "F1":
                    raise KeyboardInterrupt("fixture child interrupted")
                if owner.fail == "error":
                    raise RuntimeError("fixture child failure")
                for path in (owner.w["templates"][arm] / "stage").iterdir():
                    if path.name not in {"backend_result.json", "job.json"}:
                        shutil.copyfile(path, stage / path.name)
                raw = E.loads((owner.w["templates"][arm] / "stage/backend_result.json").read_text(encoding="utf-8"))
                raw["execution"] = E.loads((stage / "execution.json").read_text(encoding="utf-8"))
                owner.current += timedelta(seconds=10)
                live["reader"]()
                return raw

        def close():
            assert owner.active

        return Backend(), close


def run(w, services=None, *, execute=True):
    return P.run(w["request_ref"], execute=execute, allow_fixture=True, home=w["home"], now=NOW, services=services)


def test_readonly_prepare_never_acquires_or_queries(tmp_path):
    w = world(tmp_path)
    services = Services(w)
    assert run(w, services, execute=False)["status"] == "prepared"
    assert services.acquires == services.checks == 0 and not w["folder"].exists()


def test_full_parent_pair_fakeprocess_and_released_lease_lookup(tmp_path):
    w = world(tmp_path)
    services = Services(w)
    result = run(w, services)
    assert result["status"] == "complete", result
    assert services.calls == ["NONE", "F1"] and services.acquires == services.releases == 1 and not services.active
    assert run(w, services) == result and services.calls == ["NONE", "F1"]
    assert P._OWNER is None and P.JOB2.scope_active_root() is None
    assert result["host_completion_seal_required"] and result["c_loop_update_permitted"] is False
    for entry in result["completed"]:
        plan = P.U.read_ref(entry["plan"])
        proof = P.U.read_ref(plan["proof"])
        assert proof["evidence_kind"] == E.FIXTURE and proof["gpu_binding"] == P.GPU_BINDING
        assert E._stamp(proof["expires_at_utc"]) > E._stamp(proof["admission_expires_at_utc"])
        assert not proof["physical_mutex_is_probe_authority"]
        evidence = P.U.read_ref(entry["evidence"])
        assert E.verify(evidence, plan["roots"], expected_request_sha256=entry["request_sha256"],
            expected_roots_sha256=entry["roots_sha256"], allow_fixture=True) == evidence


def test_new_registered_reward_pair_matches_original_model_and_source(tmp_path):
    w = world(tmp_path, probe="P_reward")
    services = Services(w)
    result = run(w, services)
    assert result["status"] == "complete", result
    assert services.calls == ["NONE", "F1"] and w["protocol"]["allowed_probes"] == ["P_reward"]
    for entry in result["completed"]:
        evidence = P.U.read_ref(entry["evidence"])
        plan = P.U.read_ref(entry["plan"])
        assert evidence["probe"] == plan["identity"]["probe"] == "P_reward"
        source = w["binding"]["expected_request"]["execution"]["operations"][plan["identity"]["arm"]]["source"]
        assert evidence["request"]["source"] == plan["source"] == source


@pytest.mark.parametrize("field", ["model", "operation", "command"])
def test_registered_reward_requires_model_operation_and_command_probe_agreement(tmp_path, field):
    kwargs = {field + "_probe": "P_episode"}
    w = world(tmp_path, probe="P_reward", **kwargs)
    services = Services(w)
    with pytest.raises(ValueError, match="probe|identity|command"):
        run(w, services)
    assert services.acquires == 0 and services.calls == []


@pytest.mark.parametrize("kind", ["unregistered", "optimizer"])
def test_registered_probe_subset_does_not_expand_initial_or_optimizer_backend(kind):
    probe = "P_reward" if kind == "unregistered" else E.B.OPTIMIZER_PROBE
    allowed = ["P_episode"] if kind == "unregistered" else [probe]
    with pytest.raises(ValueError, match="registered allowed_probes|optimizer pair backend unsupported"):
        P.registered_probe({"allowed_probes": allowed}, {"probe": probe})


@pytest.mark.parametrize("failure", ["stale", "resource", "gpu", "boot", "inactive", "lost", "error", "interrupt", "short", "model"])
def test_bad_lifetime_or_partial_execution_releases_and_never_replays(tmp_path, failure):
    w = world(tmp_path)
    services = Services(w, fail=failure)
    result = run(w, services)
    assert result["status"] == ("pending_unknown" if failure == "interrupt" else "HOLD"), result
    assert not services.active and services.releases == services.acquires == 1
    calls = list(services.calls)
    assert run(w, services) == result and services.calls == calls
    if failure in {"gpu", "boot", "interrupt"}:
        assert len(result["completed"]) == 1 and result["completed"][0]["status"] == "complete"


@pytest.mark.parametrize("events", [0, 1, 2])
def test_missing_proposed_or_approved_prefix_cannot_dispatch(tmp_path, events):
    w = world(tmp_path)
    raw = w["copy"].read_bytes().splitlines(keepends=True)
    w["copy"].write_bytes(b"".join(raw[:events]))
    w["request"]["authority"]["host_reference"]["sha256"] = E.J.file_sha(w["copy"])
    w["request_ref"] = save(tmp_path / "partial-request.json", w["request"])
    services = Services(w)
    with pytest.raises(ValueError, match="prefix"):
        run(w, services)
    assert services.acquires == 0


@pytest.mark.parametrize("change", ["request_id", "raw", "projection", "fixture", "plan_sha", "source"])
def test_exact_authority_and_registered_source_are_required(tmp_path, change):
    w = world(tmp_path)
    if change == "request_id":
        w["request"]["authority"]["host_reference"]["request_id"] = "another-request"
    elif change == "raw":
        w["copy"].write_bytes(w["copy"].read_bytes() + b"\n")
    elif change == "projection":
        w["request"]["authority"]["projection"]["remote_path"] = "/transport/other.jsonl"
    elif change == "fixture":
        with pytest.raises(ValueError, match="fixture mode"):
            P.run(w["request_ref"])
        return
    elif change == "source":
        path = w["repo"] / "bench/run_agentic_brev_probe_parent_v1.py"
        path.write_bytes(path.read_bytes() + b"\n# source changed after registration\n")
    else:
        w["request_ref"]["sha256"] = "0" * 64
    if change in {"request_id", "projection"}:
        w["request_ref"] = save(tmp_path / "changed-request.json", w["request"])
    services = Services(w)
    with pytest.raises(ValueError):
        run(w, services)
    assert services.acquires == 0


def test_repair_holds_before_any_authority_or_resource(tmp_path):
    request = save(tmp_path / "repair.json", {"schema": P.SCHEMA + "_request", "action": "repair"})
    assert P.run(request, execute=True) == {"status": "HOLD", "reason": "repair_unsupported", "dispatched": False}


@pytest.mark.parametrize("value", [{"/restricted/fixture": "value"},
    {"outer": {"inner": "/restricted/fixture"}}, {"metadata": ["bench/private/fixture.json"]}])
def test_public_registration_rejects_private_paths_in_keys_values_and_lists(value):
    with pytest.raises(ValueError, match="private path"):
        P._public(value)


def test_bounded_wait_preserves_cleanup_wait_and_group_flags(monkeypatch):
    waits = []

    class Process:
        pid = 123

        def wait(self, timeout=None):
            waits.append(timeout)
            return 0

    seen = []
    monkeypatch.setattr(P.U, "_owned_popen", lambda args, **kwargs: (seen.append(kwargs), Process())[1])
    monkeypatch.setattr(P.U.OWNED, "_process_identity", lambda _pid: {"pid": 123, "start_ticks": 1})
    bounded = P.BoundedPopen(lambda: None, 30, ["registered-python"])
    process = bounded(["registered-python"], env={"CUDA_VISIBLE_DEVICES": "0"}, start_new_session=True)
    process.wait(timeout=900)
    process.wait(timeout=10)
    assert 0 < waits[0] <= 30 and waits[1] == 10
    assert seen[0]["start_new_session"]


@pytest.mark.parametrize("failure", ["none", "lookup", "interrupt"])
def test_unverified_child_is_quarantined_with_mutex_until_exit(tmp_path, monkeypatch, failure):
    w = world(tmp_path)
    prepared = P.prepare(w["request_ref"], allow_fixture=True, home=w["home"], now=NOW)
    prepared["folder"].mkdir(parents=True)
    resource = Services(w)
    polls = []

    class Child:
        def close(self):
            assert resource.active and resource.releases == 0
            raise RuntimeError("explicit CPU unresolved child")

        def groups_alive(self):
            assert resource.active and resource.releases == 0
            polls.append(True)
            if failure == "lookup" and len(polls) == 1:
                raise OSError("explicit CPU group observation unavailable")
            return len(polls) < 2

    actual = P.ActualServices()
    actual.children = [Child()]
    def sleep(_delay):
        if failure == "interrupt":
            raise KeyboardInterrupt("explicit CPU interrupted quarantine wait")

    monkeypatch.setattr(P.time, "sleep", sleep)
    resource.ensure_children_closed = lambda prepared, lease: actual.ensure_children_closed(prepared, lease)
    with P._acquire(prepared, resource):
        assert resource.active
    assert resource.releases == 1 and not resource.active and len(polls) == 2
    assert E.J._read(prepared["folder"] / "quarantine.json")["physical_mutex_retained"] is True
    resolved = E.J._read(prepared["folder"] / "quarantine-resolved.json")
    assert resolved["owned_groups_absent"] is True
    assert resolved["poll_error_count"] == (0 if failure == "none" else 1)


@pytest.mark.parametrize("change", ["probe", "checkpoint", "args", "operation", "stage", "seed", "lineage", "source", "budget"])
def test_consumed_other_exact_request_cannot_expand_registered_authority(tmp_path, change):
    w = world(tmp_path)
    pair = deepcopy(w["binding"]["expected_request"])
    op = pair["execution"]["operations"]["NONE"]
    if change == "probe":
        pair["probe"] = "P_reward"
    elif change == "checkpoint":
        pair["checkpoints"]["NONE"] = "0" * 64
    elif change == "args":
        op["args"] = {"num_envs": 1}
    elif change == "operation":
        op["operation_id"] = "f" * 64
    elif change == "stage":
        op["remote_stage"] = "/foreign/raw"
    elif change == "seed":
        pair["seed"] = 1
    elif change == "lineage":
        op["source"]["lineage_id"] = "foreign-lineage"
    elif change == "source":
        pair["execution"]["source_closure_sha256"] = "0" * 64
    else:
        pair["budget_gpu_s"] *= 2
    prefix = P.PairLedger(tmp_path / "other-request.ledger.jsonl").authorize(pair, "cpu_explicit")
    shutil.copyfile(prefix["path"], w["copy"])
    w["request"]["authority"]["host_reference"] = {**prefix, "path": pair["execution"]["ledger_projection"]["host_path"]}
    w["request_ref"] = save(tmp_path / "wrong-exact-request.json", w["request"])
    services = Services(w)
    with pytest.raises(ValueError, match="registered exact request"):
        run(w, services)
    assert services.acquires == 0 and services.calls == []


@pytest.mark.parametrize("source", ["bench/run_agentic_brev_probe_parent_v1.py", "tests/test_agentic_brev_probe_parent_v1.py",
    "bench/agentic_play_job_v2.py", "bench/run_agentic_capture_unit_v2.py", "src/rl_triage/mechanism_registry.py"])
def test_each_new_source_dependency_is_required_before_acquire(tmp_path, source):
    w = world(tmp_path)
    protocol = deepcopy(w["protocol"])
    del protocol["sources_lf"][source]
    files = {name: (w["repo"] / name).read_bytes() for name in P.SOURCES}
    files.update({"bench/protocols/pair.json": E.J._bytes(protocol),
        **{name: (w["repo"] / name).read_bytes() for name in ("bench/protocols/basis.json", "bench/protocols/probe.json")}})
    w["request"]["source_anchor"] = commit_objects(w["repo"], files, w["request"]["source_anchor"])
    w["request"]["protocol"] = P.U.ref(w["repo"] / "bench/protocols/pair.json")
    w["request_ref"] = save(tmp_path / "missing-source-request.json", w["request"])
    services = Services(w)
    with pytest.raises(ValueError, match="source closure"):
        run(w, services)
    assert services.acquires == 0


@pytest.mark.parametrize("kind", ["stale", "missing"])
def test_new_host_provider_packet_is_required_for_each_entry(tmp_path, kind):
    w = world(tmp_path)
    path = tmp_path / "host/provider-F1.json"
    if kind == "missing":
        path.unlink()
    else:
        body = E.J._read(path)
        body["observed_at_utc"] = "2026-10-27T00:00:00Z"
        path.write_bytes(E.J._bytes(body))
    services = Services(w)
    result = run(w, services)
    assert result["status"] == "HOLD", result
    assert services.calls == ["NONE"] and services.releases == 1 and len(result["completed"]) == 1


def test_existing_operation_is_never_redispatched_or_overwritten(tmp_path):
    w = world(tmp_path)
    stage = w["repo"] / "parents/operations" / ("1" * 64) / "raw"
    stage.mkdir(parents=True)
    (stage / "partial.stdout").write_bytes(b"preserved partial raw")
    services = Services(w)
    result = run(w, services)
    assert result["status"] == "HOLD" and services.acquires == 0 and services.calls == []
    assert (stage / "partial.stdout").read_bytes() == b"preserved partial raw"
    assert run(w, services) == result


def test_completed_raw_tamper_is_not_accepted_after_release(tmp_path):
    w = world(tmp_path)
    services = Services(w)
    result = run(w, services)
    assert result["status"] == "complete", result
    plan = P.U.read_ref(result["completed"][0]["plan"])
    Path(plan["boot"]["path"]).write_bytes(b"different boot")
    with pytest.raises(ValueError, match="raw parent changed"):
        run(w, services)
    assert services.acquires == 1 and services.calls == ["NONE", "F1"]


def test_cpu_services_cannot_be_injected_as_actual(tmp_path):
    w = world(tmp_path)
    services = Services(w)
    services.evidence_kind = E.ACTUAL
    with pytest.raises(ValueError, match="explicit CPU lifetime"):
        run(w, services)
    assert services.acquires == 0


def test_repair_cli_is_hold_without_gpu_or_authority(tmp_path):
    request = save(tmp_path / "repair.json", {"schema": P.SCHEMA + "_request", "action": "repair"})
    result = subprocess.run([sys.executable, "-B", str(ROOT / "bench/run_agentic_brev_probe_parent_v1.py"),
        "--request", request["path"], "--request-sha256", request["sha256"], "--execute"],
        capture_output=True, text=True, encoding="utf-8", timeout=30, check=False)
    assert result.returncode == 0, result.stderr
    assert E.loads(result.stdout)["reason"] == "repair_unsupported", (result.stdout, result.stderr)


def test_snapshot_history_projection_preserves_raw_and_all_public_fields(tmp_path):
    w = world(tmp_path)
    destination = tmp_path / "copied-host"
    shutil.copytree(tmp_path / "host", destination)
    roots = [{"remote": "/host", "local": str(destination)}]
    execution = w["binding"]["expected_request"]["execution"]
    before = (destination / "call/capsule.json").read_bytes()
    snapshot, spec, transport = P._selection(execution, E.Resolver(roots, E.digest(roots)))
    assert before == (destination / "call/capsule.json").read_bytes()
    assert snapshot["inputs"]["ledger_path"] == execution["snapshot_history_projection"]["host_path"]
    assert transport["raw_context_digest"] != transport["derived_context_digest"]
    assert transport["allowed_changes"] == ["snapshot.inputs.ledger_path", "context_digest"]
    assert transport["public_view_unchanged"] and spec["probe"] == "P_episode"


@pytest.mark.parametrize("change", ["history", "host_path", "context", "params", "response"])
def test_model_transport_does_not_skip_original_digest_history_or_provider_checks(tmp_path, change):
    w = world(tmp_path)
    execution = deepcopy(w["binding"]["expected_request"]["execution"])
    if change == "history":
        (tmp_path / "host/call/history-binding-v1/history.jsonl").write_bytes(b"changed raw history\n")
    elif change == "host_path":
        execution["snapshot_history_projection"]["host_path"] = "Q:\\foreign\\history.jsonl"
    elif change in {"context", "params"}:
        path = tmp_path / "host/call/capsule.json"
        data = E.J._read(path)
        if change == "context":
            data["context"]["context_digest"] = "0" * 64
        else:
            data["params"]["temperature"] = 1
        path.write_bytes(E.J._bytes(data))
        execution["selection_capsule"]["sha256"] = E.J.file_sha(path)
    else:
        (tmp_path / "host/cpu-api/responses/0001.json").write_bytes(b"{}\n")
    roots = w["binding"]["roots"]
    with pytest.raises(ValueError):
        P._selection(execution, E.Resolver(roots, E.digest(roots)))


@pytest.mark.parametrize("kind", ["provider_before_lease", "provider_stale", "provider_future", "lease_expired", "lease_future"])
def test_actual_admission_uses_current_lease_time_and_separate_provider_time(tmp_path, monkeypatch, kind):
    calls, builds = [], []
    acquired, expires = NOW - timedelta(seconds=9), NOW + timedelta(seconds=100)
    provider_observed = NOW - timedelta(seconds=20)
    if kind == "provider_stale":
        provider_observed = NOW - timedelta(seconds=31)
    elif kind == "provider_future":
        provider_observed = NOW + timedelta(seconds=1)
    elif kind == "lease_expired":
        expires = NOW - timedelta(seconds=1)
    elif kind == "lease_future":
        acquired = NOW + timedelta(seconds=1)
    lease = save(tmp_path / "lease.json", {"acquired_at_utc": acquired.isoformat(), "expires_at_utc": expires.isoformat()})
    inventory = save(tmp_path / "provider.json", {"schema": "explicit_cpu_inventory"})

    class ClockFixture(P.ActualServices):
        evidence_kind = E.FIXTURE

        def clock(self):
            return NOW

    def query(command, **_kwargs):
        calls.append(command)
        stdout = "GPU-fixture, NVIDIA L40S\n" if command == P.MODEL_COMMAND else ""
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    def builder(_path, **kwargs):
        instant = E._stamp(kwargs["observed_at_utc"])
        assert acquired <= instant < expires
        builds.append(kwargs)
        return {"schema": "cpu_admission_clock_fixture", "observed_at_utc": kwargs["observed_at_utc"]}

    monkeypatch.setattr(P.subprocess, "run", query)
    monkeypatch.setattr(P.JOB2, "build_resource_proof", builder)
    prepared = {"basis_path": tmp_path / "unused-protocol.json", "repo": tmp_path}
    basis = {"resources": {"deadline_utc": (NOW + timedelta(hours=1)).isoformat()}}
    provider = {"observed_at_utc": provider_observed.isoformat(), "inventory_native": inventory}
    folder = tmp_path / "admission"
    if kind == "provider_before_lease":
        capture, _model = ClockFixture().admission(prepared, folder, lease, basis, lambda: None, provider)
        assert E._stamp(P.U.read_ref(capture)["observed_at_utc"]) == NOW
        times = E.J._read(folder / "admission-times.json")
        assert E._stamp(times["provider_observed_at_utc"]) < E._stamp(times["lease_acquired_at_utc"]) < E._stamp(times["admission_at_utc"])
        assert len(times["queries"]) == len(calls) == 3 and len(builds) == 1
    else:
        with pytest.raises(ValueError, match="stale/future|outside acquired physical lease"):
            ClockFixture().admission(prepared, folder, lease, basis, lambda: None, provider)
        assert builds == []
