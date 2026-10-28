"""Registered Brev repair evidence; raw bytes stay in their producer namespace."""
from __future__ import annotations

import hashlib
import json
import math
import re
import threading
from copy import deepcopy
from pathlib import Path, PurePosixPath, PureWindowsPath

import agentic_brev_controller_scope_v1 as S
import agentic_investigation_contract_v1 as A
import brev_recovery_backend_v4 as B
import residual_training_v1 as TRAIN
from run_agentic_development_v2 import verify_sources

C, E, J, GPU = S.C, S.E, S.E.J, S.E.GPU
SCHEMA = "agentic_brev_repair_v1"
ARTIFACT = "agentic_brev_repair_artifact_v1"
CORRECTION = "restore_body_frame_reward"
NEW_SOURCES = {
    "bench/run_agentic_brev_repair_unit_v1.py", "evals/agentic_brev_repair_evidence_v1.py",
    "evals/agentic_brev_runtime_host_v2.py", "tests/test_agentic_brev_repair_unit_v1.py",
    "tests/test_agentic_brev_repair_evidence_v1.py", "tests/test_agentic_brev_runtime_host_v2.py",
}
SOURCES = set(C.SOURCE_FILES) | set(S.SOURCE_FILES) | NEW_SOURCES | {
    "evals/vision_brev_backend_v1.py", "evals/vision_brev_training_backend_v2.py",
    "bench/agentic_play_job_v2.py", "bench/brev_recovery_worker_v1.py",
    "bench/brev_recovery_worker_v2.py", "bench/brev_recovery_worker_v3.py",
    "evals/agentic_investigation_runtime_v1.py", "evals/agentic_brev_runtime_host_v1.py",
    "evals/agentic_investigation_contract_v1.py", "evals/agentic_investigation_loop_v1.py",
}
_BASE = C.validate_artifact
_HOOK = "validate_artifact"
_ACTIVE = None


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def regular(path):
    path = Path(path)
    require(path.is_absolute() and path.is_file() and path.stat().st_nlink == 1
            and not any(p.is_symlink() for p in (path, *path.parents)), "single regular raw file")
    return path.resolve(strict=True)


def ref(path):
    path = regular(path)
    return {"path": str(path), "sha256": J.file_sha(path)}


def write(path, body):
    path = Path(path)
    require(not any(p.is_symlink() for p in (path, *path.parents)), "write ancestor symlink")
    path.parent.mkdir(parents=True, exist_ok=True)
    J._write_new(path, body)
    return ref(path)


def logical(value):
    require(isinstance(value, str) and bool(value) and "\x00" not in value, "producer path")
    p = PureWindowsPath(value) if PureWindowsPath(value).drive else PurePosixPath(value)
    require(p.is_absolute() and ".." not in p.parts and "." not in p.parts, "absolute producer path")
    return p


class RootMap:
    """Only registered path relocation; contents and all producer hashes stay raw."""

    def __init__(self, rows):
        require(isinstance(rows, list) and bool(rows), "root map required")
        self.rows, self.inventory = [], {}
        for row in rows:
            require(set(row) == {"remote", "local"}, "root map fields")
            remote, local = logical(row["remote"]), Path(row["local"])
            require(local.is_absolute() and local.is_dir() and local == local.resolve(strict=True)
                    and not any(p.is_symlink() for p in (local, *local.parents)), "native root canonical")
            require(all(type(remote) is not type(old) or
                        (remote != old and remote not in old.parents and old not in remote.parents)
                        for old, _ in self.rows), "overlapping producer roots")
            self.rows.append((remote, local))

    def path(self, value):
        p = logical(value)
        rows = [(remote, local) for remote, local in self.rows
                if type(p) is type(remote) and (p == remote or remote in p.parents)]
        require(len(rows) == 1, "path outside registered root map")
        remote, local = rows[0]
        native = local.joinpath(*p.relative_to(remote).parts)
        require(not any(x.is_symlink() for x in (native, *native.parents))
                and native.resolve().is_relative_to(local), "mapped path escaped/symlink")
        return native

    def file(self, value, *, lf=False):
        require(isinstance(value, dict) and set(value) == {"path", "sha256"}
                and re.fullmatch(r"[0-9a-f]{64}", str(value.get("sha256"))) is not None, "raw ref fields")
        path = regular(self.path(value["path"]))
        data = path.read_bytes()
        digest = hashlib.sha256(data.replace(b"\r\n", b"\n") if lf else data).hexdigest()
        require(digest == value["sha256"], "raw file SHA changed")
        actual = hashlib.sha256(data).hexdigest()
        require(value["path"] not in self.inventory or self.inventory[value["path"]] == actual,
                "raw changed during verification")
        self.inventory[value["path"]] = actual
        return path

    def read(self, value):
        return E.loads(self.file(value).read_text(encoding="utf-8"))


def contract(value):
    fixed = {"schema": SCHEMA + "_contract", "task": B.TASK, "arm": "F1", "seed": 2221,
             "correction_id": CORRECTION, "recovery_mode": "restart", "iterations": 300,
             "num_envs": 4096, "max_repairs": 1, "max_train_runs": 1, "max_eval_jobs": 1,
             "confirming_probe": "P_reward"}
    require(set(value) == set(fixed) | {"budget", "timeouts", "bands", "source_inputs", "layout_sha256",
                                        "fixed_eval_protocol_sha256_lf"}, "repair contract fields")
    require(all(value.get(k) == v and type(value[k]) is type(v) for k, v in fixed.items()),
            "unsupported repair contract; HOLD")
    budget = value["budget"]
    require(set(budget) == {"gpu_s", "wall_s", "compute_usd", "usd_per_gpu_s"}
            and all(positive(v) for v in budget.values()), "explicit finite repair budget")
    timeout = value["timeouts"]
    require(set(timeout) == {"train", "evaluate"} and all(type(v) is int and v > 0 for v in timeout.values())
            and timeout["train"] <= B.TIMEOUTS["train"] and timeout["evaluate"] <= B.TIMEOUTS["evaluate"]
            and sum(timeout.values()) <= budget["gpu_s"] <= budget["wall_s"] <= 2700
            and budget["wall_s"] * budget["usd_per_gpu_s"] <= budget["compute_usd"], "bounded train/evaluate windows/cost reservation")
    require(isinstance(value["bands"], dict) and bool(value["bands"])
            and value["fixed_eval_protocol_sha256_lf"] == C.FIXED_EVAL_PROTOCOL_SHA256, "fixed evaluation binding")
    inputs = value["source_inputs"]
    require(set(inputs) == {"case", "reference"} and all(set(r) == {"checkpoint_sha256", "telemetry_sha256"}
            and all(re.fullmatch(r"[0-9a-f]{64}", str(v)) is not None for v in r.values())
            for r in inputs.values()), "registered source inputs")
    return deepcopy(value)


def registration(registration_ref, root_map, *, allow_fixture=False):
    resolver = RootMap(root_map)
    raw = resolver.read(registration_ref)
    require(set(raw) == {"schema", "synthetic_cpu", "runtime", "layout"}
            and raw["schema"] == SCHEMA + "_registration" and type(raw["synthetic_cpu"]) is bool
            and raw["synthetic_cpu"] == allow_fixture, "registration synthetic/actual boundary")
    runtime = raw["runtime"]
    require(set(runtime) == {"protocol", "anchor", "repo"}, "runtime source registration")
    repo = resolver.path(runtime["repo"]).resolve(strict=True)
    protocol_path = resolver.file(runtime["protocol"])
    protocol = resolver.read(runtime["protocol"])
    require(protocol.get("schema") == "agentic_investigation_runtime_v1_protocol"
            and SOURCES <= protocol.get("sources_lf", {}).keys(), "repair source closure missing")
    verify_sources(protocol, runtime["anchor"], repo=repo, protocol_path=protocol_path)
    rule = contract(protocol.get("repair_contract", {}))
    layout = raw["layout"]
    require(set(layout) == {"root_maps", "output_root", "backend_protocol", "backend_anchor", "expected_source",
                           "lab", "python", "lease_basis", "provider_target", "sources", "fixed_eval_protocol",
                           "provider", "admission_service"},
            "repair layout fields")
    require(set(layout["root_maps"]) == {"host", "provider"}
            and J.digest(layout) == rule["layout_sha256"] and root_map in layout["root_maps"].values(), "registered IO layout/root map")
    resolver.file(layout["fixed_eval_protocol"])
    require(C.digest_lf_file(resolver.file(layout["fixed_eval_protocol"])) == rule["fixed_eval_protocol_sha256_lf"],
            "fixed eval source changed")
    backend_path = resolver.file(layout["backend_protocol"])
    backend = B.validate_protocol({**resolver.read(layout["backend_protocol"]),
        "protocol_path": str(backend_path), "protocol_sha256_lf": C.digest_lf_file(backend_path)})
    expected = layout["expected_source"]
    require(expected["source_anchor"] == layout["backend_anchor"] and expected["lab_tracked_clean"] is True
            and expected["lab_head"] == expected["environment"]["isaaclab_sha"]
            and all(expected[key] == backend[key] and bool(backend[key]) for key in E.HASHES)
            and all(expected["environment"].get(k) == v for k, v in backend["environment"].items())
            and expected["selected_interpreter"]["python"] == expected["environment"]["python"]
            and expected["selected_interpreter"]["packages"] == {k: v for k, v in expected["environment"].items()
                if k not in {"python", "isaaclab_sha", "cuda_available", "cuda_device_count"}}, "backend source/environment binding")
    from vision_transferred_e2e_v2 import ObjectEvidence

    store = ObjectEvidence(repo)
    todo, seen = [runtime["anchor"]], set()
    while todo:
        current = todo.pop()
        if current in seen:
            continue
        seen.add(current)
        todo.extend(store.commit(current)["parents"])
    require(layout["backend_anchor"] in seen, "backend source must precede Runtime registration")
    for name, digest in expected["source_sha256_lf"].items():
        require(C.digest_lf_file(regular(repo / name)) == digest
                and hashlib.sha256(store.file(layout["backend_anchor"], name)[1].replace(b"\r\n", b"\n")).hexdigest() == digest,
                "backend committed source mismatch")
    require(set(layout["sources"]) == {"case", "reference"}, "only preregistered input source bundles")
    return {"raw": raw, "resolver": resolver, "protocol": protocol, "contract": rule,
            "layout": layout, "backend_protocol": backend, "repo": repo, "allow_fixture": allow_fixture}


def _project_snapshot(snapshot, resolver):
    projected = deepcopy(snapshot)
    projected["inputs"]["ledger_path"] = str(resolver.path(snapshot["inputs"]["ledger_path"]).resolve())
    C.V.checked(projected)
    return projected


def verify_model(request, registered):
    resolver, rule = registered["resolver"], registered["contract"]
    intent, context = request["intent"], request["context"]
    require(set(intent) == {"action", "target", "snapshot", "review", "source_anchor", "protocol", "selector", "approval_granted"}
            and intent["action"] == "propose_correction" and intent["target"] == CORRECTION
            and intent["selector"] == "model" and intent["approval_granted"] is False, "repair action/host args")
    require(intent["source_anchor"] == registered["raw"]["runtime"]["anchor"]
            and intent["protocol"] == registered["raw"]["runtime"]["protocol"]
            and context["snapshot"] == intent["snapshot"] and context["context_digest"] == C.L.digest({
                k: v for k, v in context.items() if k != "context_digest"}), "raw model context binding")
    snapshot = _project_snapshot(intent["snapshot"], resolver)
    derived = A.prepare(snapshot, **{k: context[k] for k in ("goal", "evidence", "revealed", "extra_facts", "decisions_left")})
    expected = deepcopy(context)
    expected["snapshot"], expected["context_digest"] = snapshot, derived["context_digest"]
    require(derived == expected, "projection changed fields beyond history path/context digest")
    review = intent["review"]
    checked = A.assess(json.dumps(review.get("decision")), derived)
    require(review == {**checked, "context_digest": context["context_digest"]}
            and review["status"] == "valid" and review["decision"]["action"] == intent["action"]
            and review["decision"]["target"] == intent["target"], "model choice/fact IDs invalid")
    packet = snapshot["inputs"]["packet"]
    require(packet["epoch"] == 1 and packet["applied_corrections"] == []
            and packet["input_bindings"] == rule["source_inputs"]
            and C.registered_correction(snapshot, []) == ("reward", CORRECTION), "registered reward source/epoch")
    return snapshot


def verify_authorization(request, authorization_ref, registered, *, done=False):
    """Explicit portable C prefix contract; no raw ledger/path mutation or approval."""
    resolver = registered["resolver"]
    wrapper = resolver.read(authorization_ref)
    require(set(wrapper) == {"authorization", "ledger_path"}, "repair authorization fields")
    auth = wrapper["authorization"]
    ledger_ref = request["approval_ledger"]
    require(ledger_ref["path"] == wrapper["ledger_path"], "C approval ledger path")
    prefix = request["approval_prefix"]
    require(prefix["sha256"] == ledger_ref["sha256"], "immutable C prefix SHA")
    ledger_path = resolver.file(prefix)
    ledger = C.D1.CorrectionLedger(ledger_path, C.RC.correction_registry())
    row = ledger.requests().get(auth["request_id"], {})
    events = [e["event"] for e in ledger.events() if e.get("request_id") == auth["request_id"]]
    require(done is False and events == ["proposed", "approved", "consumed"]
            and row.get("state") == "running"
            and row.get("loop_state_digest") == C.L.digest(request["intent"]["snapshot"])
            and row.get("correction_id") == auth["correction_id"] == CORRECTION
            and auth["component"] == "reward" and row.get("evidence") == auth["evidence"]
            and row.get("recovery_mode") == "restart" and row.get("budget_gpu_s") == registered["contract"]["budget"]["gpu_s"],
            "C consumed approval exact binding")
    kind = C.FIXTURE_EVIDENCE if registered["allow_fixture"] else C.ACTUAL_EVIDENCE
    require(auth["evidence_kind"] == auth["evidence"]["evidence_kind"] == kind
            and auth["evidence"]["arm"] == "F1"
            and auth["evidence"]["probe_seals"] == request["probe_seals"], "C evidence kind/seal refs")
    verified = [verify_probe(seal, request, registered) for seal in request["probe_seals"]]
    require(any(v["probe"] == "P_reward" and v["outcome"] == "abnormal" for v in verified),
            "genuine reward abnormal and NONE normal reference required")
    return wrapper


def verify_probe(reference, request, registered):
    resolver = registered["resolver"]
    raw = resolver.read(reference)
    auth = resolver.read({"path": raw["authorization_path"], "sha256": raw["authorization_sha256"]})
    snap = raw["snapshot"]
    require(raw["schema"] == "selection_probe_seal_v1" and raw["status"] == "completed"
            and raw["snapshot_digest"] == C.L.digest(snap) == auth["snapshot_digest"]
            and auth["snapshot"] == snap and snap["binding"] == request["intent"]["snapshot"]["binding"],
            "probe seal/snapshot lineage")
    spec = C._frozen_request_spec(snap, auth["choice"])
    ledger = C.S.M.RegistryLedger(resolver.path(str(logical(raw["authorization_path"]).parent / "probes.ledger.jsonl")), C.RC.registry())
    row = ledger.requests().get(auth["request_id"], {})
    events = [e["event"] for e in ledger.events() if e.get("request_id") == auth["request_id"]]
    require(events == ["proposed", "approved", "consumed", "receipt"] and row.get("state") == "done"
            and row.get("prereg_digest") == C.L.digest(auth["prereg"]) == C.L.digest(spec["prereg"])
            and row.get("args_digest") == C.L.digest(spec["args"]) and row.get("probe") == spec["probe"]
            and row.get("receipt") == raw["receipt"], "probe approval prefix/receipt")
    measurements = []
    for side in ("case", "reference"):
        value = resolver.read({"path": raw[side + "_path"], "sha256": raw[side + "_sha256"]})
        if registered["allow_fixture"]:
            require(raw["evidence_kind"] == auth["evidence_kind"] == C.FIXTURE_EVIDENCE, "fixture probe kind")
            measurements.append(value)
        else:
            require(raw["evidence_kind"] == auth["evidence_kind"] == C.ACTUAL_EVIDENCE, "actual probe kind")
            record = next((r for r in request["probe_records"] if r["reference"] == {
                "path": raw[side + "_path"], "sha256": raw[side + "_sha256"]}), None)
            if record is None:
                raise ValueError("registered probe evidence missing")
            require(J.digest(record["roots"]) == record["roots_sha256"], "registered probe raw roots")
            roots = [{"remote": r["remote"], "local": str(resolver.path(r["local"]).resolve())} for r in record["roots"]]
            rebuilt = E.build(value["request"], roots, expected_request_sha256=record["request_sha256"],
                              expected_roots_sha256=J.digest(roots), allow_fixture=False)
            require({k: v for k, v in rebuilt.items() if k != "root_map_sha256"} == {
                k: v for k, v in value.items() if k != "root_map_sha256"}, "probe transport changed fields beyond root map digest")
            require(rebuilt["probe"] == spec["probe"] and rebuilt["seed"] == 2221, "probe/source request")
            measurements.append(rebuilt["measurement"])
    outcome = C.classify_raw(spec["probe"], *measurements)
    require(C.classify_raw(spec["probe"], measurements[1], measurements[1]) == "normal"
            and raw["receipt"]["outcome"] == outcome and raw["receipt"]["exit_code"] == 0
            and raw["receipt"]["over_budget"] is False, "probe outcome/reference normal")
    return {"probe": spec["probe"], "outcome": outcome}


def verify_training(bundle_ref, role, registered):
    resolver, rule = registered["resolver"], registered["contract"]
    bundle = resolver.read(bundle_ref)
    require(set(bundle) == {"artifact", "job", "worker", "execution", "manifest", "readback", "records", "checkpoint", "telemetry", "stdout", "stderr"},
            "complete training inventory required")
    artifact, job, worker = (resolver.read(bundle[k]) for k in ("artifact", "job", "worker"))
    manifest, readback = resolver.read(bundle["manifest"]), resolver.read(bundle["readback"])
    for name in ("checkpoint", "telemetry", "records", "stdout", "stderr"):
        resolver.file(bundle[name])
    arm = {"case": "F1", "reference": "NONE", "output": "CORRECTED"}[role]
    corrections = [CORRECTION] if role == "output" else []
    source = rule["source_inputs"]["case"]["checkpoint_sha256"] if role == "output" else None
    require(artifact["arm"] == worker["arm"] == job["arm"] == arm
            and artifact["seed"] == worker["seed"] == job["seed"] == 2221
            and job["iterations"] == worker["iterations"] == 300
            and artifact["source_checkpoint_sha256"] == job.get("source_checkpoint_sha256") == source
            and artifact["applied_corrections"] == corrections
            and artifact["original_faults"] == ([] if arm == "NONE" else ["F1"]), "training identity/restart binding")
    require(artifact["checkpoint"] == worker["checkpoint"] == bundle["checkpoint"]
            and artifact["telemetry"]["path"] == bundle["telemetry"]["path"]
            and artifact["telemetry"]["sha256"] == worker["telemetry"]["sha256"] == bundle["telemetry"]["sha256"]
            and artifact["manifest_path"] == job["manifest_path"] == worker["manifest_path"] == bundle["manifest"]["path"]
            and artifact["manifest_sha256"] == worker["manifest_sha256"] == bundle["manifest"]["sha256"]
            and artifact["manifest"] == manifest and artifact["readback"] == worker["readback"] == readback,
            "training raw inventory mismatch")
    TRAIN.validate_manifest(manifest)
    require(manifest["arm"] == ("FIRST" if role == "output" else arm)
            and manifest["source_checkpoint_sha256"] == source and manifest["max_iterations"] == 300
            and manifest["seed"] == 2221 and manifest["run_name"] == job["run_name"]
            and manifest["records_path"] == bundle["records"]["path"]
            and manifest["readback_path"] == bundle["readback"]["path"]
            and readback["remaining_faults"] == (["F1"] if role == "case" else [])
            and readback["observer_manifest"] == TRAIN.observer_manifest(manifest), "manifest/readback fresh initialization")
    require(worker["records"] == bundle["records"] and worker["readback_sha256"] == bundle["readback"]["sha256"],
            "records/readback raw worker binding")
    observer = TRAIN.observer_manifest(manifest)
    records = [E.loads(line) for line in resolver.file(bundle["records"]).read_text(encoding="utf-8").splitlines()]
    inspection = B.OPTIMIZER.assemble_inspection(records, observer)
    require(inspection["status"] == "normal", "optimizer records incomplete/abnormal")
    execution = resolver.read(bundle["execution"])
    require(artifact["execution"] == execution, "training execution embedded/raw mismatch")
    require(execution["schema"] == "brev_recovery_posix_execution_v1" and execution["mode"] == "train"
            and execution["completed"] is True and worker["completed"] is True
            and execution["exit_code"] == 0 and execution["timed_out"] is False
            and execution["cleanup_verified"] is True and execution["gpu_before"] == execution["gpu_after"] == []
            and execution["source_before"] == execution["source_after"], "training completion/cleanup provenance")
    if role != "output":
        expected = rule["source_inputs"][role]
        require(bundle["checkpoint"]["sha256"] == expected["checkpoint_sha256"]
                and bundle["telemetry"]["sha256"] == expected["telemetry_sha256"], "registered source cache SHA")
    return {"bundle": bundle, "artifact": artifact, "job": job, "worker": worker, "execution": execution,
            "manifest": manifest, "telemetry": resolver.read(bundle["telemetry"])}


def verify_execution(execution, worker, job, mode, registered, resource, admission):
    resolver, rule = registered["resolver"], registered["contract"]
    require(execution["schema"] == "brev_recovery_posix_execution_v1" and execution["mode"] == mode
            and execution["completed"] is True and worker["completed"] is True
            and type(execution["exit_code"]) is int and execution["exit_code"] == 0
            and execution["timed_out"] is False and execution["cleanup_verified"] is True
            and execution["gpu_before"] == execution["gpu_after"] == []
            and positive(execution["wall_s"]) and execution["wall_s"] <= rule["timeouts"][mode]
            and positive(worker["wall_s"]) and worker["wall_s"] <= execution["wall_s"], "execution/cleanup/time bound")
    launch = execution["launch"]
    require(type(launch["pid"]) is int and launch["pid"] > 0 and launch["pid"] == launch["sid"] == launch["pgid"]
            and launch["posix_group_verified"] is True and type(launch["start_ticks"]) is int
            and launch["start_ticks"] > 0, "owned child identity required")
    expected = registered["layout"]["expected_source"]
    require(execution["source_before"] == execution["source_after"] == expected
            and job["source_commit_sha"] == expected["source_anchor"]
            and job["protocol_sha256_lf"] == expected["protocol"]["sha256_lf"]
            and all(job[key] == expected[key] for key in E.HASHES), "source/env/packages changed")
    before, after = [], []
    for field, target in (("gpu_before_observation", before), ("gpu_after_observation", after)):
        path = resolver.file(execution[field])
        value = resolver.read(execution[field])
        for stem in ("gpu", "process"):
            raw = {"path": str(logical(execution[field]["path"]).parent / value[stem + "_stdout_file"]),
                   "sha256": value[stem + "_stdout_sha256"]}
            require(resolver.file(raw).parent == path.parent, "GPU observation sibling files")
        target.append(GPU.verify_observation_reference(ref(path), resource["resource_id"], resource["gpu_uuid"]))
    require(before[0]["ordinal"] < after[0]["ordinal"], "GPU observation order")
    started, finished = E._stamp(execution["started_at"]), E._stamp(execution["finished_at"])
    require(E._stamp(admission["admission_at_utc"]) <= started < finished <= E._stamp(admission["lease_expires_at_utc"])
            and (finished - started).total_seconds() <= rule["timeouts"][mode], "execution outside lease/window")


def verify_repair_evidence(raw_ref, *, request_ref, registration_ref, root_map):
    resolver = RootMap(root_map)
    raw, request = resolver.read(raw_ref), resolver.read(request_ref)
    registered = registration(registration_ref, root_map, allow_fixture=raw.get("synthetic_cpu") is True)
    require(raw.get("schema") == SCHEMA + "_outcome" and raw.get("status") == "completed"
            and raw["request"] == request_ref and raw["registration"] == registration_ref
            and raw["operation_id"] == request["operation_id"] and request["registration"] == registration_ref
            and raw["authorization"] == request["authorization"],
            "repair raw/request/registration binding")
    verify_model(request, registered)
    verify_authorization(request, raw["authorization"], registered)
    source = {role: verify_training(registered["layout"]["sources"][role], role, registered)
              for role in ("case", "reference")}
    output = verify_training(raw["training"], "output", registered)
    output_root = provider_path(registered["layout"]["output_root"], registered)
    stage = output_root / request["operation_id"] / "train"
    require(logical(output["job"]["stage"]) == stage
            and output["job"]["run_name"] == "repair_" + request["operation_id"][:20]
            and output["worker"]["checkpoint"]["sha256"] == output["bundle"]["checkpoint"]["sha256"],
            "new operation training stage/run binding")
    admission = resolver.read(raw["admission"])
    require(admission["schema"] == SCHEMA + "_admission" and admission["operation_id"] == request["operation_id"]
            and admission["registration"] == registration_ref, "admission binding")
    resource = resolver.read(admission["resource"])
    if not registered["allow_fixture"]:
        require(resource["schema"] == "agentic_play_resource_v1" and resource["status"] == "granted"
                and resource["recording_protocol_sha256_lf"] == C.digest_lf_file(resolver.file(registered["layout"]["lease_basis"]))
                and resource["recording_target"] == registered["layout"]["provider_target"], "actual resource proof/basis/target")
    capture = resource["capture"]
    relocated = dict(capture)
    for stem in ("provider_inventory", "nvidia_gpu_csv", "nvidia_process_csv"):
        relocated[stem + "_path"] = str(resolver.file({"path": capture[stem + "_path"], "sha256": capture[stem + "_sha256"]}))
    GPU.verify_capture_proof(relocated)
    require(capture["resource_id"] == registered["layout"]["provider_target"]["resource_id"]
            and admission["boot"]["sha256"] == raw["boot_after"]["sha256"], "resource/current boot changed")
    resolver.file(admission["boot"])
    resolver.file(raw["boot_after"])
    verify_execution(output["execution"], output["worker"], output["job"], "train", registered, capture, admission)
    evaluation = resolver.read(raw["evaluation"])
    require(set(evaluation) == {"job", "worker", "execution", "NONE", "CORRECTED", "stdout", "stderr", "jobs"}, "evaluation inventory")
    for name in ("stdout", "stderr", "jobs"):
        resolver.file(evaluation[name])
    job, worker, execution = (resolver.read(evaluation[k]) for k in ("job", "worker", "execution"))
    require(logical(job["stage"]) == output_root / request["operation_id"] / "evaluate"
            and logical(job["fixed_eval_protocol"]) == provider_path(registered["layout"]["fixed_eval_protocol"]["path"], registered)
            and job["jobs_path"] == evaluation["jobs"]["path"],
            "registered fixed evaluation stage/protocol")
    verify_execution(execution, worker, job, "evaluate", registered, capture, admission)
    none = deepcopy(source["reference"]["artifact"])
    for key in ("checkpoint", "telemetry"):
        none[key]["path"] = str(provider_path(none[key]["path"], registered))
    require(job["artifacts"] == {"NONE": none, "CORRECTED": output["artifact"]},
            "fixed evaluation source/output binding")
    reports = {}
    for name, training in (("NONE", source["reference"]), ("CORRECTED", output)):
        require(worker["reports"][name] == evaluation[name], "fixed worker/report ref")
        report = resolver.read(evaluation[name])
        cp = training["bundle"]["checkpoint"]
        reports[name] = C.G.validate_fixed_report(report, {"name": name, "checkpoint_file": logical(cp["path"]).name,
            "checkpoint_sha256": cp["sha256"], "action_scale": 0.25,
            "protocol_sha256_lf": registered["contract"]["fixed_eval_protocol_sha256_lf"]})
    elapsed = output["execution"]["wall_s"] + execution["wall_s"]
    budget = registered["contract"]["budget"]
    require(positive(raw["wall_s_total"]) and elapsed <= raw["wall_s_total"] <= budget["wall_s"]
            and elapsed <= budget["gpu_s"] and raw["wall_s_total"] * budget["usd_per_gpu_s"] <= budget["compute_usd"]
            and raw["gpu_s"] == elapsed and raw["compute_usd"] == raw["wall_s_total"] * budget["usd_per_gpu_s"], "aggregate repair budget/cost")
    require(E._stamp(output["execution"]["finished_at"]) <= E._stamp(execution["started_at"]), "train before evaluation")
    for name in ("entry", "exit"):
        lifetime = resolver.read(raw["lifetime"][name])
        require(lifetime["admission"] == raw["admission"] and lifetime["capture"] == capture
                and lifetime["synthetic_cpu"] is registered["allow_fixture"], "physical lifetime record binding")
        if not registered["allow_fixture"]:
            live = lifetime["live_lease"]
            require(live["live_lock_proof"] is True and live["reference"] == resource["lease"]
                    and live["resource_id"] == capture["resource_id"] and bool(live["owner"]), "actual live canonical lease proof")
    return {"raw": raw, "registered": registered, "request": request, "source": source, "output": output,
            "reports": reports, "evaluation": evaluation, "gpu_s": elapsed,
            "recovery": C.D1.recovery(reports["CORRECTED"], reports["NONE"], registered["contract"]["bands"])}


def provider_path(value, registered):
    """Pure producer-side projection; no foreign native path is resolved."""
    value = logical(value)
    matches = []
    for row in registered["layout"]["root_maps"]["provider"]:
        remote = logical(row["remote"])
        if type(value) is type(remote) and (remote == value or remote in value.parents):
            matches.append(logical(row["local"]).joinpath(*value.relative_to(remote).parts))
    require(len(matches) == 1, "provider path projection")
    return matches[0]


def artifact(bundle_ref, role, registered, *, policy):
    checked = verify_training(bundle_ref, role, registered)
    cp = checked["bundle"]["checkpoint"]
    value = {"schema": ARTIFACT, "evidence_kind": C.FIXTURE_EVIDENCE if registered["allow_fixture"] else C.ACTUAL_EVIDENCE,
             "actual_ready": not registered["allow_fixture"], "arm": "NONE" if role == "reference" else "F1",
             "policy": policy, "epoch": 1 if role == "output" else 0,
             "corrections": [CORRECTION] if role == "output" else [], "training_mode": "new_initialization",
             "source_checkpoint_sha256": checked["artifact"]["source_checkpoint_sha256"],
             "checkpoint": {"path": str(registered["resolver"].file(cp)), "sha256": cp["sha256"], "identity": ""},
             "bundle": deepcopy(bundle_ref), "role": role}
    value["checkpoint"]["identity"] = C.checkpoint_identity(value)
    return value


class RepairArtifactScope:
    """Single-thread narrow schema dispatcher sharing the frozen probe RLock."""

    def __init__(self, registration, evidence):
        self.registration, self.evidence = deepcopy(registration), deepcopy(evidence)
        self.owner = None
        self.completions = {}

    def register_output(self, outcome_ref, request_ref):
        require(_ACTIVE is self and self.owner == threading.get_ident(), "repair output registration owner")
        checked = verify_repair_evidence(outcome_ref, request_ref=request_ref,
            registration_ref=self.registration["reference"], root_map=self.registration["root_map"])
        bundle = checked["raw"]["training"]
        key = J.digest(bundle)
        entry = {"outcome": deepcopy(outcome_ref), "request": deepcopy(request_ref)}
        require(key not in self.completions or self.completions[key] == entry, "repair completion changed")
        self.completions[key] = entry

    def _validate(self, value, **expected):
        require(_ACTIVE is self and self.owner == threading.get_ident(), "repair scope owner")
        if value.get("schema") != ARTIFACT:
            return _BASE(value, **expected)
        reg = registration(self.registration["reference"], self.registration["root_map"],
                           allow_fixture=self.registration["allow_fixture"])
        if value["role"] == "output":
            entry = self.completions.get(J.digest(value["bundle"]))
            require(entry is not None, "output artifact requires registered full repair evidence")
            if entry is None:
                raise ValueError("repair output evidence absent")
            checked = verify_repair_evidence(entry["outcome"], request_ref=entry["request"],
                registration_ref=self.registration["reference"], root_map=self.registration["root_map"])
            require(checked["raw"]["training"] == value["bundle"], "repair output evidence/bundle")
        else:
            require(value["bundle"] in self.evidence
                    and value["bundle"] == reg["layout"]["sources"].get(value["role"]), "registered input artifact only")
        fresh = artifact(value["bundle"], value["role"], reg, policy=value["policy"])
        require(value == fresh and all(value[key] == expected[key] for key in (
            "arm", "policy", "epoch", "corrections", "source_checkpoint_sha256"))
            and (expected.get("expected_kind") is None or value["evidence_kind"] == expected["expected_kind"]),
            "repair artifact controller binding")
        return deepcopy(value)

    def __enter__(self):
        global _ACTIVE
        S._LOCK.acquire()
        try:
            require(_ACTIVE is None and C.validate_artifact is _BASE, "nested/changed repair scope")
            registration(self.registration["reference"], self.registration["root_map"],
                         allow_fixture=self.registration["allow_fixture"])
            self.owner, _ACTIVE = threading.get_ident(), self
            setattr(C, _HOOK, self._validate)
            return self
        except BaseException:
            S._LOCK.release()
            raise

    def __exit__(self, *_):
        global _ACTIVE
        require(_ACTIVE is self and self.owner == threading.get_ident(), "repair scope exit owner")
        try:
            setattr(C, _HOOK, _BASE)
            self.owner, _ACTIVE = None, None
        finally:
            S._LOCK.release()
