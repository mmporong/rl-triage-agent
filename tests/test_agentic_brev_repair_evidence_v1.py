"""Synthetic actual-shaped repair inventories; never Isaac/GPU execution evidence."""
from __future__ import annotations

import json
import sys
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / p) for p in ("evals", "bench", "tests", "src")]
import agentic_brev_repair_evidence_v1 as E
import agentic_brev_runtime_host_v2 as H
import agentic_investigation_runtime_v1 as R
import run_agentic_brev_repair_unit_v1 as U
from test_agentic_investigation_runtime_v1 import commit_objects
from test_brev_recovery_backend_v4 import protocol as backend_protocol
from test_optimizer_observability import observed
from test_selection_comparison_controller_v1 import (
    BANDS,
    FACTS,
    VIEW_SHA,
    measurement,
    model_dialogue,
    report,
)

NOW = datetime(2026, 10, 28, 0, 0, 10, tzinfo=timezone.utc)  # noqa: UP017
C = E.C


def blob(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(body)
    return E.ref(path)


def observation(folder, ordinal):
    gpu = blob(folder / f"{ordinal:06d}.gpu.csv", b"GPU-synthetic-current-boot\n")
    processes = blob(folder / f"{ordinal:06d}.process.csv", b"")
    return E.write(folder / f"{ordinal:06d}.receipt.json", {
        "schema": "vision_remote_gpu_observation_v1", "ordinal": ordinal,
        "resource_id": "synthetic-vm", "gpu_uuid": "GPU-synthetic-current-boot",
        "gpu_command": E.GPU.GPU_COMMAND, "process_command": E.GPU.PROCESS_COMMAND,
        "gpu_exit_code": 0, "process_exit_code": 0,
        "gpu_stdout_file": Path(gpu["path"]).name, "gpu_stdout_sha256": gpu["sha256"], "gpu_stderr": "",
        "process_stdout_file": Path(processes["path"]).name,
        "process_stdout_sha256": processes["sha256"], "process_stderr": ""})


def execution(data, folder, mode, offset):
    return {"schema": "brev_recovery_posix_execution_v1", "mode": mode, "completed": True,
        "started_at": (NOW + timedelta(seconds=offset)).isoformat(),
        "finished_at": (NOW + timedelta(seconds=offset + 1)).isoformat(), "exit_code": 0,
        "timed_out": False, "cleanup_verified": True,
        "launch": {"pid": 4242, "pgid": 4242, "sid": 4242, "start_ticks": 12345, "posix_group_verified": True},
        "gpu_before": [], "gpu_after": [], "wall_s": 1.0,
        "source_before": deepcopy(data.expected), "source_after": deepcopy(data.expected),
        "gpu_before_observation": observation(folder / "gpu", 1),
        "gpu_after_observation": observation(folder / "gpu", 2)}


def base_job(data, folder, mode):
    return {"schema": "brev_recovery_worker_job_v1", "mode": mode, "stage": str(folder),
        "lab": str(data.root / "lab"), "repo": str(data.repo), "source_commit_sha": data.backend_anchor,
        "protocol_path": data.backend_ref["path"], "protocol_sha256_lf": data.backend_ref["sha256"],
        **{key: data.expected[key] for key in E.E.HASHES}}


def training(data, folder, arm, source=None, *, offset=-5, run_name="synthetic_source"):
    job = {**base_job(data, folder, "train"), "arm": arm, "seed": 2221, "run_name": run_name,
        "iterations": 300, "source_checkpoint_sha256": source,
        **{key: str(folder / name) for key, name in (
            ("manifest_path", "training.manifest.json"), ("records_path", "optimizer.records.jsonl"),
            ("readback_path", "training.readback.json"), ("telemetry_path", "telemetry.json"))}}
    source_faults, applied = ([] if arm == "NONE" else ["F1"]), ([E.CORRECTION] if arm == "CORRECTED" else [])
    manifest = {"schema": E.TRAIN.SCHEMA, "mode": "train", "arm": "FIRST" if arm == "CORRECTED" else arm,
        "seed": 2221, "run_name": run_name, "task": E.B.TASK, "num_envs": 4096, "max_iterations": 300,
        "configured_learning_epochs": 5, "configured_mini_batches": 4, "runtime_updates_to_observe": 3,
        "source_commit_sha": data.backend_anchor, "protocol_path": "bench/protocols/backend.json",
        "protocol_sha256_lf": data.backend_ref["sha256"],
        "source_sha256_lf": data.expected["source_sha256_lf"], "runtime_source_sha256_lf": data.expected["runtime_source_sha256_lf"],
        "records_path": job["records_path"], "readback_path": job["readback_path"],
        "source_checkpoint_sha256": source, "original_faults": source_faults, "applied_corrections": applied}
    E.TRAIN.validate_manifest(manifest)
    observer_manifest = E.TRAIN.observer_manifest(manifest)
    ppo, rows, observer = observed(epochs=5, bound=observer_manifest)
    ppo.num_mini_batches = 4
    try:
        for _ in range(3):
            ppo.update()
    finally:
        observer.remove()
    cp = blob(folder / "model_299.pt", ("SYNTHETIC checkpoint " + arm).encode())
    telemetry = E.write(folder / "telemetry.json", {"summary": {"reward": {"mean_last_20pct": 0.2}},
                                                  "series": {}, "synthetic_cpu": True, "arm": arm})
    readback = {"remaining_faults": ["F1"] if arm == "F1" else [], "observer_manifest": observer_manifest}
    manifest_ref = E.write(folder / "training.manifest.json", manifest)
    readback_ref = E.write(folder / "training.readback.json", readback)
    records_ref = blob(folder / "optimizer.records.jsonl", b"".join(E.J._bytes(row) for row in rows))
    exe = execution(data, folder, "train", offset)
    worker = {"schema": "brev_recovery_train_worker_v1", "completed": True, "arm": arm, "seed": 2221,
        "iterations": 300, "learn_wall_s": 0.5, "wall_s": 0.9, "checkpoint": cp, "telemetry": telemetry,
        "manifest_path": manifest_ref["path"], "manifest_sha256": manifest_ref["sha256"],
        "records": records_ref, "readback": readback, "readback_sha256": readback_ref["sha256"]}
    for name, body in (("job", job), ("worker", worker), ("execution", exe)):
        E.write(folder / (name + ".json"), body)
    for name in ("stdout", "stderr"):
        blob(folder / (name + ".log"), b"")
    artifact = {"checkpoint": cp, "telemetry": telemetry, "arm": arm, "seed": 2221,
        "original_faults": source_faults, "applied_corrections": applied, "source_checkpoint_sha256": source,
        "manifest_path": manifest_ref["path"], "manifest_sha256": manifest_ref["sha256"],
        "manifest": manifest, "readback": readback, "execution": exe}
    return artifact, U.training_bundle(folder, artifact)


def world(root):
    data = SimpleNamespace(root=root, repo=root / "repo", clock=NOW, live=True, calls=[], bad=None)
    data.repo.mkdir(parents=True)
    paths = H.SOURCES | set(E.TRAIN.MINIMUM_BOUND_SOURCES)
    files = {}
    for name in paths:
        path = ROOT / name
        assert path.is_file(), name
        files[name] = path.read_bytes()
    hashes = {name: C.digest_lf_file(ROOT / name) for name in paths}
    backend = backend_protocol()
    backend["source_sha256_lf"] = {n: hashes[n] for n in E.TRAIN.MINIMUM_BOUND_SOURCES}
    backend["protocol_path"] = str(data.repo / "bench/protocols/backend.json")
    backend["protocol_sha256_lf"] = "0" * 64
    files["bench/protocols/backend.json"] = E.J._bytes(backend)
    data.backend_anchor = commit_objects(data.repo, files)
    data.backend_ref = E.ref(data.repo / "bench/protocols/backend.json")
    environment = {**backend["environment"], "cuda_available": True, "cuda_device_count": 1}
    data.expected = {"lab_head": environment["isaaclab_sha"], "lab_tracked_clean": True,
        "source_anchor": data.backend_anchor, "protocol": {"path": "bench/protocols/backend.json", "sha256_lf": data.backend_ref["sha256"]},
        "environment": environment, "selected_interpreter": {"python": environment["python"], "packages": {}},
        **{key: backend[key] for key in E.E.HASHES}}
    case, case_ref = training(data, root / "sources/F1", "F1", run_name="synthetic_f1")
    none, none_ref = training(data, root / "sources/NONE", "NONE", run_name="synthetic_none")
    inputs = {role: {"checkpoint_sha256": artifact["checkpoint"]["sha256"],
                     "telemetry_sha256": artifact["telemetry"]["sha256"]}
              for role, artifact in (("case", case), ("reference", none))}
    data.case, data.none = case, none
    maps = [{"remote": str(root), "local": str(root)}]
    (root / "output").mkdir()
    boot = blob(root / "boot.txt", b"SYNTHETIC current boot\n")
    data.boot = boot
    layout = {"root_maps": {"host": maps, "provider": maps}, "output_root": str(root / "output"),
        "backend_protocol": data.backend_ref, "backend_anchor": data.backend_anchor, "expected_source": data.expected,
        "lab": str(root / "lab"), "python": sys.executable,
        "lease_basis": E.write(root / "lease-basis.json", {"synthetic_cpu": True}),
        "provider_target": {"resource_id": "synthetic-vm", "resource_name": "synthetic-vm", "instance_type": "L40S"},
        "sources": {"case": case_ref, "reference": none_ref},
        "fixed_eval_protocol": E.ref(data.repo / "bench/protocols/fixed_eval_v1.json"),
        "provider": {"path": "bench/run_agentic_brev_repair_unit_v1.py", "sha256_lf": hashes["bench/run_agentic_brev_repair_unit_v1.py"],
                     "class": "RepairProvider"},
        "admission_service": {"path": "tests/test_agentic_brev_repair_evidence_v1.py", "sha256_lf": hashes["tests/test_agentic_brev_repair_evidence_v1.py"]}}
    rule = {"schema": E.SCHEMA + "_contract", "task": E.B.TASK, "arm": "F1", "seed": 2221,
        "correction_id": E.CORRECTION, "recovery_mode": "restart", "iterations": 300, "num_envs": 4096,
        "max_repairs": 1, "max_train_runs": 1, "max_eval_jobs": 1, "confirming_probe": "P_reward",
        "budget": {"gpu_s": 60, "wall_s": 60, "compute_usd": 1.0, "usd_per_gpu_s": 0.01},
        "timeouts": {"train": 30, "evaluate": 30}, "bands": BANDS, "source_inputs": inputs,
        "layout_sha256": E.J.digest(layout), "fixed_eval_protocol_sha256_lf": C.FIXED_EVAL_PROTOCOL_SHA256}
    protocol = {"schema": "agentic_investigation_runtime_v1_protocol", "sources_lf": hashes,
                "controller_extensions": [E.E.GPU_CONTRACT], "repair_contract": rule}
    files["bench/protocols/repair-runtime.json"] = E.J._bytes(protocol)
    anchor = commit_objects(data.repo, files, data.backend_anchor)
    protocol_ref = E.ref(data.repo / "bench/protocols/repair-runtime.json")
    registration = {"schema": E.SCHEMA + "_registration", "synthetic_cpu": True,
        "runtime": {"protocol": protocol_ref, "anchor": anchor, "repo": str(data.repo)}, "layout": layout}
    data.registration_ref = E.write(root / "registration.json", registration)
    data.maps, data.registration = maps, registration
    data.registered = E.registration(data.registration_ref, maps, allow_fixture=True)
    packet = C.S.build_packet(epoch=1, applied_corrections=[], prior_receipt=None, facts=FACTS, input_bindings=inputs)

    def snapshot():
        return C.V.build_snapshot(packet, "repair_synthetic_s2221", "registry_next", VIEW_SHA, root / "F1/epoch1/probes.ledger.jsonl")

    data.seals = []
    for index in range(8):
        snap = snapshot()
        if snap["view"]["status"] != "open":
            break
        choice = C.choose(snap, "registry_next", model_dialogue(snap) if C.dialogue_input(snap)["needs_model"] else None)
        probe = choice["probe"]
        outcome = C.RC.registry()["probes"][probe]["expect"]["reward"]
        raw = root / "probe" / str(index)
        case_path = E.write(raw / "case.json", measurement(probe, outcome))
        none_path = E.write(raw / "none.json", measurement(probe, "normal"))
        seal = C.record_probe(root, "F1", 1, snap, choice, Path(case_path["path"]), Path(none_path["path"]), "synthetic-operator", 1.0)
        data.seals.append({k: seal[k] for k in ("path", "sha256")})
    data.loop = R.L.Investigation(snapshot(), goal="synthetic RI42 repair", allow_fixture=True)
    data.context = data.loop.context()
    choice = {"action": "propose_correction", "target": E.CORRECTION,
              "fact_ids": [next(iter(data.context["model_view"]["evidence_facts"]))]}
    review = data.loop.propose(json.dumps(choice))
    runtime = object.__new__(R.Runtime)
    runtime.reg = {"allow_fixture": True, "bands": BANDS}
    runtime.protocol, runtime.protocol_ref, runtime.anchor, runtime.repo = protocol, protocol_ref, anchor, data.repo
    runtime.folder, runtime.manifest = root / "runtime", {"synthetic_cpu": True, "source_anchor": anchor}
    data.runtime = runtime
    data.intent = {"action": "propose_correction", "target": E.CORRECTION, "snapshot": snapshot(), "review": review,
                   "source_anchor": anchor, "protocol": protocol_ref, "selector": "model", "approval_granted": False}
    data.operation = E.J.digest({"manifest": runtime.manifest, "intent": data.intent})
    step = runtime.folder / "steps/0001"
    intent_ref = E.write(step / "intent.json", data.intent)
    E.write(step / "operation.json", {"operation_id": data.operation, "intent_sha256": intent_ref["sha256"]})
    E.write(step / "request.json", {"context": data.context})
    return data


class FakeBounded:
    def close(self):
        return None

    def groups_alive(self):
        return False


class Services:
    synthetic_cpu = True

    def __init__(self, data):
        self.data = data

    def clock(self):
        return self.data.clock

    def check(self, registered, admission, *, entry=False):
        assert registered["allow_fixture"] is True
        E.require(self.data.live, "synthetic lease revoked")
        E.require(registered["resolver"].file(admission["boot"]).read_bytes() == b"SYNTHETIC current boot\n", "boot changed")
        proof = registered["resolver"].read(admission["resource"])
        E.require(proof["capture"]["gpu_uuid"] == "GPU-synthetic-current-boot", "UUID changed")
        return proof["capture"]

    def backend(self, registered, folder, capture, check, mode):
        return FakeBackend(self.data, registered, check), FakeBounded()

    def boot_after(self, folder):
        return blob(folder / "boot-after.txt", b"SYNTHETIC current boot\n")


class FakeBackend:
    def __init__(self, data, registered, check):
        self.data, self.registered, self.check = data, registered, check

    def train(self, stage, arm, seed, run_name, corrections, iterations, source_checkpoint_sha256):
        assert (arm, seed, iterations, corrections) == ("CORRECTED", 2221, 300, [E.CORRECTION])
        self.check()
        self.data.calls.append("train")
        if self.data.bad == "interrupt":
            raise KeyboardInterrupt("synthetic child interrupted")
        value, _ = training(self.data, stage, arm, source_checkpoint_sha256, offset=0, run_name=run_name)
        # Producer fixture already wrote the backend-result/bundle; real provider owns these writes.
        (stage / "backend-result.json").rename(stage / "synthetic-backend-result.json")
        (stage / "bundle.json").rename(stage / "synthetic-bundle.json")
        if self.data.bad == "cleanup":
            value["execution"]["cleanup_verified"] = False
            path = stage / "execution.json"
            path.write_bytes(E.J._bytes(value["execution"]))
        if self.data.bad == "lease":
            self.data.live = False
        self.data.clock = NOW + timedelta(seconds=1)
        return value

    def evaluate(self, stage, artifacts, run_name):
        self.check()
        self.data.calls.append("evaluate")
        reports = {}
        refs = {}
        for name, source_name in (("NONE", "s2120_none.json"),
                                  ("CORRECTED", "s2120_first.json" if self.data.bad == "not_recovered" else "s2120_both.json")):
            artifact = artifacts[name]
            controller = {"checkpoint": artifact["checkpoint"]}
            path = report(stage, source_name, name, controller)
            target = stage / "reports" / (name + ".json")
            assert path == target
            refs[name], reports[name] = E.ref(path), E.E.loads(path.read_text(encoding="utf-8"))
        job = {**base_job(self.data, stage, "evaluate"), "artifacts": artifacts,
               "fixed_eval_protocol": self.registered["layout"]["fixed_eval_protocol"]["path"],
               "reports_dir": str(stage / "reports"), "jobs_path": str(stage / "fixed_eval.jobs.json")}
        worker = {"completed": True, "wall_s": 0.9, "reports": refs}
        exe = execution(self.data, stage, "evaluate", 2)
        for name, body in (("job", job), ("worker", worker), ("execution", exe)):
            E.write(stage / (name + ".json"), body)
        for name in ("stdout", "stderr"):
            blob(stage / (name + ".log"), b"")
        E.write(stage / "fixed_eval.jobs.json", {"synthetic_cpu": True, "arms": ["NONE", "CORRECTED"]})
        self.data.clock = NOW + timedelta(seconds=3)
        return {"reports": reports, "execution": exe}


class ProbeHost:
    def __init__(self):
        self.scope, self.calls = None, []

    def __enter__(self):
        self.scope = E.S.BrevProbeScope(allow_fixture=True)
        self.scope.__enter__()
        return self

    def __exit__(self, *args):
        assert self.scope is not None
        return self.scope.__exit__(*args)

    def approve(self, *args):
        self.calls.append(("approve", args))
        return {"status": "synthetic-probe-approved"}

    def dispatch(self, *args):
        self.calls.append(("dispatch", args))
        return {"status": "synthetic-probe-dispatched"}

    def lookup(self, *args):
        self.calls.append(("lookup", args))
        return {"status": "synthetic-probe-lookup"}


class Admitter:
    def __init__(self, data):
        self.data = data

    def issue(self, operation, request, authorization, registration):
        assert self.data.calls == []
        root = self.data.root / "admission"
        inventory = {"workspaces": [{key: "synthetic-vm" if key in {"id", "name"} else "RUNNING" if key == "status" else "L40S"
                                   for key in E.GPU.PROVIDER_ROW_KEYS}]}
        capture = {"resource_id": "synthetic-vm", "gpu_uuid": "GPU-synthetic-current-boot"}
        for stem, reference in (("provider_inventory", E.write(root / "provider.json", inventory)),
            ("nvidia_gpu_csv", blob(root / "gpu.csv", b"GPU-synthetic-current-boot\n")),
            ("nvidia_process_csv", blob(root / "process.csv", b""))):
            capture[stem + "_path"], capture[stem + "_sha256"] = reference["path"], reference["sha256"]
        resource = E.write(root / "resource.json", {"capture": capture, "synthetic_cpu": True})
        model = blob(root / "model.csv", b"GPU-synthetic-current-boot,NVIDIA L40S\n")
        return E.write(root / "admission.json", {"schema": E.SCHEMA + "_admission", "operation_id": operation,
            "registration": registration, "resource": resource, "boot": self.data.boot, "gpu_model": model,
            "provider_observed_at_utc": (NOW - timedelta(seconds=10)).isoformat(),
            "query_at_utc": (NOW - timedelta(seconds=1)).isoformat(), "admission_at_utc": NOW.isoformat(),
            "lease_acquired_at_utc": (NOW - timedelta(seconds=2)).isoformat(),
            "lease_expires_at_utc": (NOW + timedelta(seconds=120)).isoformat()})


def host(data):
    provider = U.RepairProvider(data.registration_ref, data.maps, services=Services(data), allow_fixture=True)
    return H.BrevRuntimeHost(data.runtime, ProbeHost(), provider, data.registration_ref, data.maps,
        "synthetic-operator", probe_seals=data.seals, probe_records=[], admitter=Admitter(data).issue, allow_fixture=True)


def test_schema_scope_keeps_existing_controller_validator_and_rejects_fixture_promotion(tmp_path):
    data = world(tmp_path)
    before = C.validate_artifact
    reg = data.registered
    wrapper = E.artifact(reg["layout"]["sources"]["case"], "case", reg, policy="registry_next")
    assert wrapper["actual_ready"] is False and wrapper["evidence_kind"] == C.FIXTURE_EVIDENCE
    with E.RepairArtifactScope({"reference": data.registration_ref, "root_map": data.maps, "allow_fixture": True},
                               list(reg["layout"]["sources"].values())):
        C.validate_artifact(wrapper, arm="F1", policy="registry_next", epoch=0, corrections=[], source_checkpoint_sha256=None)
        bad = deepcopy(wrapper)
        bad["actual_ready"] = True
        with pytest.raises(ValueError):
            C.validate_artifact(bad, arm="F1", policy="registry_next", epoch=0, corrections=[], source_checkpoint_sha256=None)
        unregistered = deepcopy(wrapper)
        unregistered["role"] = "output"
        with pytest.raises(ValueError, match="registered full repair"):
            C.validate_artifact(unregistered, arm="F1", policy="registry_next", epoch=0,
                                corrections=[], source_checkpoint_sha256=None)
        with pytest.raises(ValueError, match="artifact"):
            C.validate_artifact({"schema": "unknown"}, arm="F1", policy="registry_next", epoch=0,
                                corrections=[], source_checkpoint_sha256=None)
    assert C.validate_artifact is before


def test_registered_sources_committed_blob_and_raw_training_tamper_rejected(tmp_path):
    data = world(tmp_path)
    reg = data.registered
    for role in ("case", "reference"):
        E.verify_training(reg["layout"]["sources"][role], role, reg)
    path = data.repo / "bench/run_agentic_brev_repair_unit_v1.py"
    path.write_bytes(path.read_bytes() + b"\n# changed fixture copy\n")
    with pytest.raises(ValueError, match="source/blob"):
        E.registration(data.registration_ref, data.maps, allow_fixture=True)


def test_cross_os_root_map_preserves_windows_reference_bytes(tmp_path):
    folder = tmp_path / "archive"
    folder.mkdir()
    raw = blob(folder / "receipt.bin", b"raw Windows authority\r\n")
    windows = {"path": "C:/synthetic-host/receipt.bin", "sha256": raw["sha256"]}
    resolver = E.RootMap([{"remote": "C:/synthetic-host", "local": str(folder)}])
    assert resolver.file(windows).read_bytes() == b"raw Windows authority\r\n"
    with pytest.raises(ValueError):
        resolver.file({**windows, "path": "C:/other-host/receipt.bin"})
