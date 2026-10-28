"""Synthetic Windows host / native CPU producer transport; no Brev or GPU calls."""
from __future__ import annotations

import base64
import shutil
import sys
import zlib
from contextlib import contextmanager
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / p) for p in ("evals", "bench", "tests", "src")]
import agentic_brev_repair_mailbox_service_v1 as S
import agentic_brev_repair_transport_remote_v1 as REMOTE
import agentic_brev_repair_transport_v1 as T
from test_agentic_brev_repair_evidence_v1 import (
    NOW,
    Admitter,
    FakeBackend,
    FakeBounded,
    ProbeHost,
    Services,
)
from test_agentic_brev_repair_evidence_v1 import world as core_world
from test_agentic_investigation_runtime_v1 import commit_objects

E, CORE = T.E, T.CORE


def world(root):
    host_root, remote_root, archive = root / "host", root / "remote", root / "archive"
    host_root.mkdir()
    remote_root.mkdir()
    archive.mkdir()
    data = core_world(host_root)
    paths = T.SOURCES | set(data.runtime.protocol["sources_lf"])
    files = {path.relative_to(data.repo).as_posix(): path.read_bytes() for path in data.repo.rglob("*")
             if path.is_file() and ".git" not in path.relative_to(data.repo).parts}
    hashes = {}
    for name in paths:
        files[name] = (ROOT / name).read_bytes()
        hashes[name] = E.C.digest_lf_file(ROOT / name)
    code_anchor = commit_objects(data.repo, files, data.runtime.anchor)
    phase = {"schema": S.OLD.SCHEMA + "_lifecycle_protocol", "registered_at_utc": (NOW - timedelta(seconds=10)).isoformat(),
        "deadline_utc": (NOW + timedelta(hours=1)).isoformat(), "resource_id": "synthetic-vm", "resource_name": "synthetic-vm",
        "instance_type": "L40S", "quoted_usd_per_h": "1", "compute_cap_usd": "100", "stop_reserve_s": 10,
        "max_lifecycle_s": 5000, "budget_origin": "immutable_registered_at_utc", "source_commit": code_anchor, "sources_lf": hashes}
    phase_path = data.repo / "bench/protocols/repair-transport-phase.json"
    E.write(phase_path, phase)
    core_host = [{"remote": str(host_root), "local": str(host_root)}, {"remote": str(remote_root), "local": str(archive)},
                 {"remote": "/synthetic-posix-input", "local": str(host_root / "sources/F1")}]
    core_provider = [{"remote": str(host_root), "local": str(remote_root)}, {"remote": str(remote_root), "local": str(remote_root)},
                     {"remote": "/synthetic-posix-input", "local": str(remote_root / "sources/F1")}]
    rows = [{"producer": str(host_root), "host": str(host_root), "remote": str(remote_root), "writable": True},
            {"producer": str(remote_root), "host": str(archive), "remote": str(remote_root), "writable": True},
            {"producer": "/synthetic-posix-input", "host": str(host_root / "sources/F1"),
             "remote": str(remote_root / "sources/F1"), "writable": True}]
    control = host_root / "io"
    control.mkdir()
    remote_control = remote_root / "io"
    remote_control.mkdir()
    config = {"schema": T.SCHEMA + "_contract", "synthetic_cpu": True, "source_anchor": code_anchor, "sources_lf": hashes,
        "service_source_sha256_lf": hashes[T.SERVICE_SOURCE], "remote_source_sha256_lf": hashes[T.REMOTE_SOURCE],
        "target": {k: phase[k] for k in ("resource_id", "resource_name", "instance_type")},
        "cli_argv_prefix": ["synthetic-not-a-Brev-command"], "host_mount_projection": {"host": str(host_root), "wsl": "/synthetic"},
        "lifecycle_protocol": E.ref(phase_path), "deadline_utc": phase["deadline_utc"],
        "lifecycle_stop_intent_path": str(host_root / "STOP.json"), "remote_stop_intent_path": str(remote_root / "STOP.json"),
        "host_control_root": str(control), "remote_control_root": str(remote_control),
        "remote_repo": "/synthetic/repo", "remote_python": "/synthetic/venv/bin/python",
        "manifest_projection": T.PROJECTION,
        "layers": {"metadata": deepcopy(rows), "evidence": deepcopy(rows)},
        "limits": {"rpc_timeout_s": 300, "admission_wait_s": 20, "owner_ttl_s": 200, "command_wait_s": 30,
                   "max_files": 1024, "max_file_bytes": 64 * 1024 * 1024, "max_archive_bytes": 128 * 1024 * 1024, "max_rpc_calls": 20}}
    config_ref = E.write(host_root / "transport-contract.json", config)
    registration = deepcopy(data.registration)
    layout = registration["layout"]
    layout["root_maps"] = {"host": core_host, "provider": core_provider}
    layout["provider"] = {"path": T.SOURCE, "sha256_lf": hashes[T.SOURCE], "class": "BrevRepairTransport"}
    layout["admission_service"] = {"path": T.SERVICE_SOURCE, "sha256_lf": hashes[T.SERVICE_SOURCE]}
    protocol = deepcopy(data.runtime.protocol)
    protocol["sources_lf"] = hashes
    protocol["repair_contract"]["layout_sha256"] = E.J.digest(layout)
    protocol["repair_transport_contract_sha256"] = config_ref["sha256"]
    protocol["lifecycle_protocol_sha256"] = config["lifecycle_protocol"]["sha256"]
    protocol["repair_manifest_projection"] = T.PROJECTION
    files["bench/protocols/repair-transport-phase.json"] = phase_path.read_bytes()
    files["bench/protocols/repair-runtime.json"] = E.J._bytes(protocol)
    anchor = commit_objects(data.repo, files, code_anchor)
    protocol_ref = E.ref(data.repo / "bench/protocols/repair-runtime.json")
    registration["runtime"] = {"protocol": protocol_ref, "anchor": anchor, "repo": str(data.repo)}
    data.registration_ref = E.write(host_root / "transport-registration.json", registration)
    data.registration, data.maps = registration, core_host
    data.registered = E.registration(data.registration_ref, core_host, allow_fixture=True)
    data.runtime.protocol, data.runtime.protocol_ref, data.runtime.anchor = protocol, protocol_ref, anchor
    data.runtime.manifest = {"synthetic_cpu": True, "source_anchor": anchor}
    data.intent["source_anchor"], data.intent["protocol"] = anchor, protocol_ref
    data.operation = E.J.digest({"manifest": data.runtime.manifest, "intent": data.intent})
    step = data.runtime.folder / "steps/0001"
    # These are synthetic fixture drafts, not original shared Runtime records.
    (step / "intent.json").write_bytes(E.J._bytes(data.intent))
    (step / "operation.json").write_bytes(E.J._bytes({"operation_id": data.operation, "intent_sha256": E.J.file_sha(step / "intent.json")}))
    for path in host_root.rglob("*"):
        if path.is_file():
            target = remote_root / path.relative_to(host_root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    data.remote_root, data.archive, data.config, data.config_ref = remote_root, archive, config, config_ref
    data.engine = FixtureRPC(data)
    return data


class FixtureRPC:
    synthetic_cpu = True

    def __init__(self, data):
        self.data = data
        self.remote = SimpleNamespace(**vars(data))
        self.remote.root = data.remote_root
        self.remote.repo = data.remote_root / data.repo.relative_to(data.root)
        self.remote.live = True
        self.methods = []
        self.owner_live = False
        self.parent_launches = 0
        self.partial_admit_once = False
        self.partial_once = False
        self.bad_reply = None
        self.before_execute = None
        self.admission = self.outcome = self.release = self.launch = None

    def clock(self):
        return self.remote.clock

    def inventory(self, folder, target):
        rows = {key: target["resource_id"] if key == "id" else target["resource_name"] if key == "name" else
                target["instance_type"] if key == "instance_type" else "RUNNING" if key == "status" else "synthetic"
                for key in E.GPU.PROVIDER_ROW_KEYS}
        return E.write(folder / "provider.json", {"workspaces": [rows]}), self.clock().isoformat()

    def __call__(self, mode, body):
        # A real POSIX producer is another process. Only the explicit CPU RPC
        # fixture suspends the Windows host's projection hook for that side.
        with patch.object(E.TRAIN, "validate_manifest", T._MANIFEST_BASE):
            return self._producer(mode, body)

    def _producer(self, mode, body):
        self.methods.append(mode)
        data, config = self.data, self.data.config
        T.unpack(body["upload"], config, "remote", operation=data.operation, kind="host-upload")
        reg = E.registration(data.registration_ref, data.registration["layout"]["root_maps"]["provider"], allow_fixture=True)
        E.verify_model(reg["resolver"].read(body["request_ref"]), reg)
        E.verify_authorization(reg["resolver"].read(body["request_ref"]), body["authorization_ref"], reg)
        folder = data.remote_root / "io" / data.operation
        folder.mkdir(parents=True, exist_ok=True)
        if mode == "admit":
            assert not self.owner_live and self.admission is None
            self.parent_launches += 1
            self.owner_live = True
            (self.remote.root / "output").mkdir(parents=True, exist_ok=True)
            self.admission = Admitter(self.remote).issue(data.operation, body["request_ref"], body["authorization_ref"], data.registration_ref)
            self.launch = E.write(folder / "synthetic-launch.json", {"synthetic_cpu": True,
                "pid": 4200, "pgid": 4200, "sid": 4200, "start_ticks": 123, "request_ref": body["request_ref"]})
            T.save(folder / "admission-ref.json", self.admission)
        elif mode == "execute":
            assert self.owner_live and self.admission == body["admission_ref"] and self.outcome is None
            try:
                if self.before_execute is not None:
                    self.before_execute(self)
                worker = REMOTE.provider(data.registration_ref, data.registration["layout"]["root_maps"]["provider"],
                                         allow_fixture=True, fixture_services=RelayServices(self.remote))
                self.outcome = worker.execute(data.operation, body["request_ref"], body["authorization_ref"], self.admission)
                assert self.owner_live
            finally:
                self.owner_live = False
                self.remote.live = False
                self.release = E.write(folder / "synthetic-release.json", {"schema": REMOTE.SCHEMA + "_release", "synthetic_cpu": True,
                    "operation_id": data.operation, "request_ref": body["request_ref"], "owned_children_closed": True,
                    "flock_released": True, "heartbeat_stopped": True})
            if self.partial_once:
                self.partial_once = False
                raise T.RepairPending("synthetic partial reply after one train/evaluation")
        elif mode == "lookup":
            assert self.launch is not None
        else:
            pytest.fail("unexpected fixture RPC mode")
        state = {"status": "completed" if self.outcome else "pending", "owner_current": self.owner_live,
                 "owner_ended": not self.owner_live, "admission": self.admission, "outcome": self.outcome,
                 "release": self.release, "launch": self.launch}
        if mode == "lookup":
            state = REMOTE._lookup_admission(folder, body, reg, state, self.clock())
        response = REMOTE._response(mode, body, reg, config, folder, state)
        if mode == "admit" and self.partial_admit_once:
            self.partial_admit_once = False
            raise T.RepairPending("synthetic lost admit response after existing owner admission")
        if self.bad_reply:
            self.bad_reply(response)
        return response


class RelayServices(Services):
    """Native runtime parameters, mirroring ActualServices.backend's projection."""

    def backend(self, registered, folder, capture, check, mode):
        native = deepcopy(registered)
        native["layout"]["fixed_eval_protocol"]["path"] = str(registered["resolver"].file(
            registered["layout"]["fixed_eval_protocol"]))
        return FakeBackend(self.data, native, check), FakeBounded()


@contextmanager
def host(data):
    service = S.BrevRepairMailboxService(data.registration_ref, data.maps, data.config_ref, allow_fixture=True, fixture_rpc=data.engine)
    transport = T.BrevRepairTransport(data.registration_ref, data.maps, data.config_ref, service, allow_fixture=True)
    with T.RepairManifestProjectionScope(data.registration_ref, data.maps, data.config_ref, allow_fixture=True), \
            CORE.BrevRuntimeHost(data.runtime, ProbeHost(), transport, data.registration_ref, data.maps, "synthetic-operator",
                probe_seals=data.seals, probe_records=[], admitter=service.admitter, allow_fixture=True) as active:
        yield active


@pytest.mark.parametrize("partial", [False, True])
def test_real_core_transport_roundtrip_and_partial_reply_lookup_never_retrains(tmp_path, partial):
    data = world(tmp_path)
    data.engine.partial_once = partial
    with host(data) as controller:
        auth = controller.approve(data.operation, data.intent)["authorization"]
        request_path = data.runtime.folder / "brev-repair-host" / data.operation / "request.json"
        original = request_path.read_bytes()
        if partial:
            with pytest.raises(T.RepairPending, match="partial"):
                controller.dispatch(data.operation, data.intent, auth)
            result = controller.lookup(data.operation, data.intent)
        else:
            result = controller.dispatch(data.operation, data.intent, auth)
        assert result["status"] == "completed" and data.calls == ["train", "evaluate"]
        assert not data.engine.owner_live and not data.engine.remote.live
        assert request_path.read_bytes() == original
        assert data.engine.methods[:2] == ["admit", "execute"]
        E.write(tmp_path / "runtime-consume/host-result.json", result)
        data.runtime._tool(data.loop, data.intent, result, tmp_path / "runtime-consume")
        assert data.loop.status == "recovered"
    before = list(data.engine.methods)
    with host(data) as fresh:
        assert fresh.lookup(data.operation, data.intent) == result
        assert fresh.dispatch(data.operation, data.intent, auth) == result
    assert data.engine.methods == before and data.calls == ["train", "evaluate"]


def test_lost_admit_reply_is_recovered_by_runtime_lookup_with_same_admission_and_one_execution(tmp_path):
    data = world(tmp_path)
    data.engine.partial_admit_once = True
    step = data.runtime.folder / "steps/0001"
    host_folder = data.runtime.folder / "brev-repair-host" / data.operation
    with host(data) as controller, pytest.raises(T.RepairPending, match="lost admit response"):
        data.runtime._operation(step, data.intent, controller.provider.callbacks(controller))
    assert (step / "dispatch-intent.json").exists() and not (host_folder / "dispatch-intent.json").exists()
    assert data.engine.methods == ["admit"] and data.engine.parent_launches == 1 and data.engine.owner_live
    original = deepcopy(data.engine.admission)
    original_bytes = Path(original["path"]).read_bytes()
    request_bytes = (host_folder / "request.json").read_bytes()
    pointer = data.remote_root / "io" / data.operation / "admission-ref.json"
    pointer_bytes = pointer.read_bytes()
    with host(data) as fresh:
        assert fresh.lookup(data.operation, data.intent)["status"] == "approved"
        result = data.runtime._operation(step, data.intent, fresh.provider.callbacks(fresh))
        assert result["status"] == "completed"
        data.runtime._tool(data.loop, data.intent, result, step)
        assert data.loop.status == "recovered"
    assert data.engine.methods == ["admit", "lookup", "execute"]
    assert data.engine.parent_launches == 1 and data.calls == ["train", "evaluate"]
    assert data.engine.admission == original and Path(original["path"]).read_bytes() == original_bytes
    assert pointer.read_bytes() == pointer_bytes and (host_folder / "request.json").read_bytes() == request_bytes
    recovered = E.E.loads((Path(data.config["host_control_root"]) / data.operation / "admission-ref.json").read_text(encoding="utf-8"))
    assert recovered == original and not data.engine.owner_live
    before = list(data.engine.methods)
    with host(data) as fresh:
        assert data.runtime._operation(step, data.intent, fresh.provider.callbacks(fresh)) == result
        assert fresh.dispatch(data.operation, data.intent, result["authorization"]) == result
    assert data.engine.methods == before and data.engine.parent_launches == 1 and data.calls == ["train", "evaluate"]


@pytest.mark.parametrize("failure", ["ended", "stale"])
def test_lost_admit_reply_ended_or_stale_owner_is_durable_hold_without_relaunch(tmp_path, failure):
    data = world(tmp_path)
    data.engine.partial_admit_once = True
    step = data.runtime.folder / "steps/0001"
    with host(data) as controller, pytest.raises(T.RepairPending, match="lost admit response"):
        data.runtime._operation(step, data.intent, controller.provider.callbacks(controller))
    original = deepcopy(data.engine.admission)
    if failure == "ended":
        data.engine.owner_live, data.engine.remote.live = False, False
    else:
        data.engine.remote.clock += timedelta(seconds=31)
    with host(data) as fresh:
        result = data.runtime._operation(step, data.intent, fresh.provider.callbacks(fresh))
        assert result["status"] == "hold" and result["reason"] == "admission_recovery_hold"
        assert data.runtime._operation(step, data.intent, fresh.provider.callbacks(fresh)) == result
        before = list(data.engine.methods)
        with pytest.raises(ValueError, match="admission recovery HOLD"):
            fresh.dispatch(data.operation, data.intent, E.E.loads((step / "approval.json").read_text(encoding="utf-8")))
    assert data.engine.methods == before == ["admit", "lookup"]
    assert data.engine.admission == original and data.engine.parent_launches == 1 and data.calls == []
    held = E.E.loads(Path(result["raw"]["path"]).read_text(encoding="utf-8"))
    assert held["intent"]["operation_id"] == data.operation and held["authority_issued"] is False
    assert not list(data.runtime.folder.rglob("repair-seal.json"))
    # The explicit stale CPU fixture has no real process; close its lifetime marker.
    data.engine.owner_live, data.engine.remote.live = False, False


def archive_fixture(root):
    for name in ("host-metadata", "host-evidence", "remote-metadata", "remote-evidence"):
        (root / name).mkdir(parents=True)
    rows = {"metadata": [{"producer": "C:/synthetic-host", "host": str(root / "host-metadata"),
                           "remote": str(root / "remote-metadata"), "writable": True}],
            "evidence": [{"producer": "/synthetic-posix", "host": str(root / "host-evidence"),
                          "remote": str(root / "remote-evidence"), "writable": True}]}
    config = {"layers": rows, "limits": {"max_files": 4, "max_file_bytes": 1024, "max_archive_bytes": 8192}}
    ref = {"path": "/synthetic-posix/model.pt", "sha256": ""}
    raw = b"SYNTHETIC raw checkpoint\r\n"
    E.write(root / "host-metadata/control.json", {"synthetic_cpu": True})
    (root / "host-evidence/model.pt").write_bytes(raw)
    ref["sha256"] = E.J.file_sha(root / "host-evidence/model.pt")
    reg = {"resolver": E.RootMap(T._maps(config, "host"))}
    return config, reg, ref, raw


def test_layered_packet_keeps_posix_raw_ref_and_windows_metadata_namespace(tmp_path):
    config, reg, ref, raw = archive_fixture(tmp_path)
    metadata = {"path": "C:/synthetic-host/control.json", "sha256": E.J.file_sha(tmp_path / "host-metadata/control.json")}
    seeds = [ref, metadata]
    packet = T.packet(seeds, reg, config, "host", operation="a" * 64, kind="synthetic")
    archive = tmp_path / "separate-raw-archive"
    assert T.unpack(packet, config, "remote", operation="a" * 64, kind="synthetic", archive_root=archive) == seeds
    assert (tmp_path / "remote-evidence/model.pt").read_bytes() == raw
    assert (archive / "evidence/0/model.pt").read_bytes() == raw
    assert (archive / "metadata/0/control.json").read_bytes() == (tmp_path / "host-metadata/control.json").read_bytes()
    assert not (archive / "metadata/0/model.pt").exists()
    assert T.unpack(packet, config, "remote", operation="a" * 64, kind="synthetic", archive_root=archive) == seeds


def test_native_case_aliases_are_duplicate_destinations_on_windows(tmp_path):
    config, reg, ref, raw = archive_fixture(tmp_path)
    packet = T.packet([ref], reg, config, "host", operation="a" * 64, kind="synthetic")
    body = E.E.loads(zlib.decompress(base64.b64decode(packet["base64"])).decode())
    alias = deepcopy(body["files"][0])
    alias["relative"], alias["reference"]["path"] = "MODEL.pt", "/synthetic-posix/MODEL.pt"
    body["files"].append(alias)
    packed = zlib.compress(E.J._bytes(body))
    envelope = {"sha256": __import__("hashlib").sha256(packed).hexdigest(), "bytes": len(packed),
                "base64": base64.b64encode(packed).decode()}
    if Path("model.pt") == Path("MODEL.pt"):
        with pytest.raises(ValueError, match="duplicate"):
            T.unpack(envelope, config, "remote", operation="a" * 64, kind="synthetic")
        assert not (tmp_path / "remote-evidence/model.pt").exists()
    else:
        T.unpack(envelope, config, "remote", operation="a" * 64, kind="synthetic")
        assert (tmp_path / "remote-evidence/model.pt").read_bytes() == raw
        assert (tmp_path / "remote-evidence/MODEL.pt").read_bytes() == raw


@pytest.mark.parametrize("bad", ["duplicate", "traversal", "link", "sha", "size", "layer", "metadata_missing", "root_map"])
def test_archive_tamper_rejected_before_any_write(tmp_path, bad):
    config, reg, ref, _ = archive_fixture(tmp_path)
    packet = T.packet([ref], reg, config, "host", operation="a" * 64, kind="synthetic")
    body = E.E.loads(zlib.decompress(base64.b64decode(packet["base64"])).decode())
    row = body["files"][0]
    if bad == "duplicate":
        body["files"].append(deepcopy(row))
    elif bad == "traversal":
        row["relative"] = "../escape"
    elif bad == "link":
        row["kind"] = "symlink"
    elif bad == "sha":
        row["reference"]["sha256"] = "0" * 64
    elif bad == "size":
        row["bytes"] = 100000
    elif bad == "layer":
        row["layer"] = "metadata"
    elif bad == "metadata_missing":
        body["files"] = []
    else:
        body["maps_sha256"] = "0" * 64
    packed = zlib.compress(E.J._bytes(body))
    envelope = {"sha256": __import__("hashlib").sha256(packed).hexdigest(), "bytes": len(packed),
                "base64": base64.b64encode(packed).decode()}
    with pytest.raises(ValueError):
        T.unpack(envelope, config, "remote", operation="a" * 64, kind="synthetic")
    assert not (tmp_path / "remote-evidence/model.pt").exists()


def test_posix_manifest_is_consumed_by_frozen_core_with_only_two_derived_paths(tmp_path):
    data = world(tmp_path)
    reg = data.registered
    original_bundle = reg["resolver"].read(reg["layout"]["sources"]["case"])
    manifest = reg["resolver"].read(original_bundle["manifest"])
    original_manifest_bytes = reg["resolver"].file(original_bundle["manifest"]).read_bytes()
    foreign = deepcopy(manifest)
    foreign["records_path"] = "/synthetic-posix-input/optimizer.records.jsonl"
    foreign["readback_path"] = "/synthetic-posix-input/training.readback.json"
    folder = data.root / "posix-manifest-fixture"
    manifest_ref = E.write(folder / "training.manifest.json", foreign)
    job = reg["resolver"].read(original_bundle["job"])
    job["manifest_path"] = manifest_ref["path"]
    worker = reg["resolver"].read(original_bundle["worker"])
    worker["manifest_path"], worker["manifest_sha256"] = manifest_ref["path"], manifest_ref["sha256"]
    worker["records"] = {**original_bundle["records"], "path": foreign["records_path"]}
    artifact = reg["resolver"].read(original_bundle["artifact"])
    artifact.update(manifest=foreign, manifest_path=manifest_ref["path"], manifest_sha256=manifest_ref["sha256"])
    bundle = {**original_bundle, "manifest": manifest_ref, "job": E.write(folder / "job.json", job),
        "worker": E.write(folder / "worker.json", worker), "artifact": E.write(folder / "artifact.json", artifact),
        "records": worker["records"], "readback": {**original_bundle["readback"], "path": foreign["readback_path"]}}
    bundle_ref = E.write(folder / "bundle.json", bundle)
    before = E.TRAIN.validate_manifest
    with T.RepairManifestProjectionScope(data.registration_ref, data.maps, data.config_ref, allow_fixture=True) as scope:
        checked = E.verify_training(bundle_ref, "case", reg)
        assert checked["manifest"] == foreign
        proof = scope.projections[E.J.digest(foreign)]
        assert proof["allowed_changes"] == ["records_path", "readback_path"]
        assert manifest_ref == E.ref(folder / "training.manifest.json")
    assert E.TRAIN.validate_manifest is before
    assert reg["resolver"].file(original_bundle["manifest"]).read_bytes() == original_manifest_bytes


def test_changed_live_host_approval_blocks_transport_lookup_before_rpc(tmp_path):
    data = world(tmp_path)
    with host(data) as controller:
        authorization = controller.approve(data.operation, data.intent)["authorization"]
        ledger = Path(authorization["ledger_path"])
        lines = ledger.read_bytes().splitlines(keepends=True)
        ledger.write_bytes(b"".join(lines[:-1]))
        request = E.ref(ledger.parent / "request.json")
        with pytest.raises(ValueError, match="host live C authority changed"):
            controller.provider.lookup(data.operation, request)
    assert data.engine.methods == [] and data.calls == []


@pytest.mark.parametrize("bad", ["lease", "boot", "admission"])
def test_changed_remote_lease_boot_or_admission_rejects_before_training_and_closes_fixture_owner(tmp_path, bad):
    data = world(tmp_path)

    def changed(engine):
        if bad == "lease":
            engine.remote.live = False
        elif bad == "boot":
            (engine.remote.root / "boot.txt").write_bytes(b"SYNTHETIC changed boot\n")
        else:
            Path(engine.admission["path"]).write_bytes(b'{"synthetic_cpu":"changed admission"}')

    data.engine.before_execute = changed
    with host(data) as controller:
        authorization = controller.approve(data.operation, data.intent)["authorization"]
        with pytest.raises(ValueError):
            controller.dispatch(data.operation, data.intent, authorization)
    assert data.calls == [] and not data.engine.owner_live
    assert not list(data.runtime.folder.rglob("repair-seal.json"))
