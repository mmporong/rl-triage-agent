"""새 pair prefix의 NONE→F1 검사와 실제 mutex lifetime을 결합한다. 기본은 읽기 검증이다."""
from __future__ import annotations

import argparse
import json
import os
import stat
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import cast

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / p) for p in ("bench", "evals", "src")]
import agentic_investigation_call_v2 as CALL
import agentic_play_job_v2 as JOB2
import run_agentic_brev_probe_unit_v1 as U
import vision_selection_brev_adapter_v1 as PAIR
from vision_selection_e2e_caller_v1 import PairLedger

E, C = U.E, U.C
SCHEMA = "agentic_brev_probe_pair_parent_v1"
GPU_BINDING = "fresh_uuid_same_current_boot_v1"
UNVERIFIED_CHILD_POLICY = "hold_mutex_until_main_stop_v1"
ORDER = ("NONE", "F1")
SOURCES = U.SOURCES | JOB2.REQUIRED_SOURCE_FILES | {
    "bench/run_agentic_brev_probe_parent_v1.py", "tests/test_agentic_brev_probe_parent_v1.py",
    "evals/vision_selection_brev_adapter_v1.py", "evals/vision_selection_e2e_caller_v1.py",
    "evals/agentic_investigation_call_v1.py", "evals/agentic_investigation_call_v2.py",
    "evals/agentic_investigation_contract_v1.py", "evals/agentic_investigation_contract_v2.py",
}
_LOCK = threading.RLock()
_OWNER = None
MODEL_COMMAND = ["nvidia-smi", "--query-gpu=uuid,name", "--format=csv,noheader,nounits"]


def require(ok: bool, reason: str) -> None:
    E.require(ok, reason)


def utc() -> datetime:
    return datetime.now(timezone.utc)  # noqa: UP017 - Python 3.10


def raw_ref(path: Path) -> dict:
    path = path.absolute()
    require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents))
            and stat.S_ISREG(path.stat().st_mode) and path.stat().st_nlink == 1, "single regular raw file")
    return {"path": str(path), "sha256": E.J.file_sha(path)}


def _directory(path: Path) -> None:
    require(not any(p.is_symlink() for p in (path, *path.parents)), "runtime ancestor symlink")
    require(not path.exists() or path.is_dir(), "runtime directory required")


def _public(value) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _public(key)
            _public(item)
    elif isinstance(value, list):
        for item in value:
            _public(item)
    elif isinstance(value, str):
        require(not PurePosixPath(value).is_absolute() and not PureWindowsPath(value).is_absolute()
                and "bench/private" not in value.replace("\\", "/"), "public registration contains private path")


def registered_probe(protocol, pair):
    allowed = protocol.get("allowed_probes")
    require(isinstance(allowed, list) and bool(allowed) and all(isinstance(item, str) for item in allowed)
            and len(allowed) == len(set(allowed)) and set(allowed) <= set(E.PROBES), "explicit registered probe subset")
    probe = pair.get("probe")
    require(probe in allowed, "pair probe is outside registered allowed_probes")
    require(probe != E.B.OPTIMIZER_PROBE, "optimizer pair backend unsupported; existing guard retained")
    return probe


def verify_authority(prepared: dict) -> dict:
    """등록 projection의 사본만 읽는다. 원장/승인/원문 내부 경로는 수정하지 않는다."""
    authority = prepared["request"]["authority"]
    require(authority.get("schema") == SCHEMA + "_authority"
            and set(authority) == {"schema", "host_reference", "projection"}, "explicit pair authority required")
    original, projection = authority["host_reference"], authority["projection"]
    require(projection == prepared["execution"]["ledger_projection"]
            and set(projection) == {"host_path", "remote_path"}
            and projection["host_path"] == original["path"], "registered host/remote ledger projection")
    require(PureWindowsPath(original["path"]).is_absolute() or PurePosixPath(original["path"]).is_absolute(),
            "host ledger absolute path")
    local = prepared["resolver"].file({"path": projection["remote_path"], "sha256": original["sha256"]})
    # 파생 reference의 path만 root-map으로 해석한다. 사본 bytes와 원래 host reference는 보존한다.
    projected = {**original, "path": str(local)}
    PairLedger(local).events()
    request = PAIR.verify_pair_prefix(local.parent, projected)
    require(request == prepared["binding"]["expected_request"], "consumed pair differs from registered exact request")
    return {"host_reference": deepcopy(original), "local_copy": raw_ref(local),
            "projection": deepcopy(projection), "request_sha256": PAIR.digest(request)}


def _transport_context(capsule, snapshot, projection, resolver):
    context = capsule["context"]
    require(context["snapshot"] == snapshot and context["context_digest"] == C.L.digest({
        key: value for key, value in context.items() if key != "context_digest"}), "raw model context digest")
    require(projection.get("schema") == SCHEMA + "_snapshot_history_projection"
            and projection["host_path"] == snapshot["inputs"]["ledger_path"]
            and projection["host_base_capsule_path"] == capsule["base_reference"]["path"]
            and projection["base_capsule"]["sha256"] == capsule["base_reference"]["sha256"], "exact snapshot history projection")
    base = resolver.read(projection["base_capsule"])
    history = resolver.file(projection["history"])
    require(base.get("schema") == CALL.V1.SCHEMA and base["context"] == context
            and base["history_sha256"] == projection["history"]["sha256"]
            and base["source_anchor"] == capsule["source_anchor"] and base["protocol_sha256"] == capsule["protocol_sha256"],
            "frozen v1 history/capsule binding")
    fresh_snapshot = deepcopy(snapshot)
    fresh_snapshot["inputs"]["ledger_path"] = str(history.resolve())
    C.V.checked(fresh_snapshot)
    fresh = CALL.A.prepare(fresh_snapshot, **{key: context[key] for key in (
        "goal", "evidence", "revealed", "extra_facts", "decisions_left")})
    expected = deepcopy(context)
    expected["snapshot"], expected["context_digest"] = fresh_snapshot, fresh["context_digest"]
    require(fresh == expected and fresh["model_view"] == context["model_view"], "model transport changed fields beyond history path/digest")
    CALL.A.checked(fresh)
    return fresh, {"schema": SCHEMA + "_model_transport", "projection": deepcopy(projection),
        "history": raw_ref(history), "base_capsule": raw_ref(resolver.file(projection["base_capsule"])),
        "raw_context_digest": context["context_digest"], "derived_context_digest": fresh["context_digest"],
        "allowed_changes": ["snapshot.inputs.ledger_path", "context_digest"], "public_view_unchanged": True}


def _model_terminal(execution, capsule, resolver):
    terminal = resolver.read(execution["selection_result"])
    manifest = resolver.read(execution["selection_manifest"])
    require(terminal.get("schema") == "model_decision_terminal_v1" and terminal["status"] == "choice_ready"
            and manifest.get("schema") == E.J.SCHEMA
            and terminal["manifest_sha256"] == execution["selection_manifest"]["sha256"]
            and manifest["binding_sha256"] == E.digest(manifest["binding"])
            and manifest["binding"]["snapshot_sha256"] == execution["selection_capsule"]["sha256"]
            and manifest["binding"]["source_anchor"] == capsule["source_anchor"]
            and manifest["binding"]["protocol_sha256"] == capsule["protocol_sha256"]
            and manifest["binding"]["initial_messages_sha256"] == capsule["messages_sha256"]
            and manifest["params"] == capsule["params"] and manifest["params_sha256"] == capsule["params_sha256"],
            "original model terminal/manifest/capsule binding")
    engine = terminal["engine_result"]
    calls, previous = [], None
    require(0 < len(execution["selection_calls"]) == terminal["completed_calls"] <= CALL.V1.MAX_CALLS,
            "model terminal call cardinality")
    for ordinal, refs in enumerate(execution["selection_calls"], 1):
        event, reservation, response = (resolver.read(refs[name]) for name in ("event", "reservation", "response"))
        require(event.get("schema") == "model_decision_event_v1" and event["ordinal"] == ordinal
                and event["previous_sha256"] == previous and event["manifest_sha256"] == terminal["manifest_sha256"]
                and event["reservation_sha256"] == refs["reservation"]["sha256"]
                and event["response_sha256"] == refs["response"]["sha256"]
                and reservation.get("schema") == "model_decision_reservation_v1"
                and reservation["index"] == event["global_index"] == response["index"]
                and reservation["ordinal"] == ordinal and reservation["binding_sha256"] == manifest["binding_sha256"]
                and reservation["manifest_sha256"] == terminal["manifest_sha256"]
                and reservation["run_id"] == manifest["binding"]["run_id"]
                and reservation["messages_sha256"] == capsule["messages_sha256"]
                and reservation["params_sha256"] == capsule["params_sha256"]
                and response.get("schema") == "model_decision_response_v1" and response["status"] == "completed"
                and response["reservation_sha256"] == refs["reservation"]["sha256"], "original provider response/event binding")
        calls.append(response["result"])
        previous = refs["event"]["sha256"]
    require(terminal["calls_sha256"] == E.digest(calls)
            and [entry["result"] for entry in engine["calls"]] == calls
            and engine["final_content"] == calls[-1].get("content"), "original provider content/terminal changed")
    return engine


def _selection(execution: dict, resolver) -> tuple[dict, dict, dict]:
    capsule, snapshot = (resolver.read(execution[name]) for name in ("selection_capsule", "snapshot"))
    context, transport = _transport_context(capsule, snapshot, execution["snapshot_history_projection"], resolver)
    require(capsule.get("schema") == CALL.SCHEMA and capsule["context"]["snapshot"] == snapshot
            and capsule["params"] == CALL.params_for(context)
            and capsule["params"] == CALL.params_for_messages(capsule["messages"])
            and capsule["system_sha256"] == E.digest(CALL.A.SYSTEM)
            and capsule["params_sha256"] == E.digest(capsule["params"])
            and capsule["messages_sha256"] == E.digest(capsule["messages"]), "frozen model capsule/snapshot binding")
    CALL.A.checked(context)
    CALL._messages(context, capsule["messages"])
    engine = _model_terminal(execution, capsule, resolver)
    review = CALL.A.assess(engine["final_content"], context)
    require(engine.get("model_status") == "valid" and engine.get("accepted") is True
            and review["status"] == "valid" and review["decision"] == engine["selection"]
            and review["decision"]["action"] == "run_probe" and review["decision"]["target"] in E.PROBES,
            "valid registered model probe decision required")
    choice = CALL.A.probe_choice(context, review)
    spec = C._frozen_request_spec(snapshot, choice)
    require(spec["probe"] == review["decision"]["target"], "model target/frozen request probe mismatch")
    return snapshot, spec, transport


def prepare(request_ref: dict, *, allow_fixture: bool = False, home: Path | None = None,
            now: datetime | None = None) -> dict:
    request = U.read_ref(request_ref)
    require(request.get("schema") == SCHEMA + "_request", "parent request schema")
    if request.get("action") != "probe":
        return {"status": "HOLD", "reason": "repair_unsupported", "dispatched": False}
    protocol, binding = U.read_ref(request["protocol"]), U.read_ref(request["binding"])
    fixture = binding.get("evidence_kind") == E.FIXTURE
    require(type(allow_fixture) is bool and binding.get("evidence_kind") in {E.FIXTURE, E.ACTUAL}
            and (not fixture or allow_fixture), "fixture mode must be explicit")
    require(fixture or (home is None and now is None), "actual HOME/clock injection forbidden")
    _public(protocol)
    require(protocol.get("schema") == SCHEMA + "_protocol" and protocol.get("registered_before_execution") is True
            and SOURCES <= protocol.get("sources_lf", {}).keys()
            and protocol["binding_sha256"] == request["binding"]["sha256"]
            and protocol["gpu_binding"] == GPU_BINDING and "gpu_uuid" not in protocol,
            "registered pair source closure/dynamic GPU binding")
    layout = protocol["layout"]
    require(set(layout) == {"repo", "isaaclab", "python"}, "HOME layout fields")
    base = (home or Path.home()).absolute()
    locations = {key: base.joinpath(*U._relative(value)).absolute() for key, value in layout.items()}
    repo, lab, python = (locations[key] for key in ("repo", "isaaclab", "python"))
    require(repo.is_dir() and lab.is_dir() and python.is_file(), "registered repo/lab/Python unavailable")
    require(fixture or repo.resolve() == ROOT.resolve(), "actual code must be registered repository")
    U.verify_sources(protocol, request["source_anchor"], repo=repo, protocol_path=Path(request["protocol"]["path"]))
    require(binding.get("schema") == SCHEMA + "_binding", "pair binding schema")
    pair = binding["expected_request"]
    probe = registered_probe(protocol, pair)
    require(set(pair) == PAIR.PAIR_KEYS and pair.get("schema") == PAIR.PAIR_SCHEMA
            and pair["arms"] == list(ORDER) and pair["seed"] == 2221,
            "registered NONE/F1 s2221 pair only")
    execution = pair["execution"]
    require(execution.get("schema") == SCHEMA + "_execution"
            and execution["order"] == list(ORDER) and execution["gpu_binding"] == GPU_BINDING
            and execution["evidence_kind"] == binding["evidence_kind"]
            and execution["unverified_child_policy"] == UNVERIFIED_CHILD_POLICY
            and execution["source_closure_sha256"] == E.digest(protocol["sources_lf"])
            and E.J.HEX64.fullmatch(str(execution["pair_id"])) is not None,
            "pair execution/source/order binding")
    root = repo.joinpath(*U._relative(protocol["execution_root"]))
    folder = root / "pairs" / execution["pair_id"]
    _directory(folder)
    current = now or utc()
    require(folder.exists() or current < E._stamp(protocol["deadline_utc"]), "pair registration expired")
    resolver = E.Resolver(binding["roots"], E.digest(binding["roots"]))
    snapshot, spec, model_transport = _selection(execution, resolver)
    require(pair["probe"] == spec["probe"], "registered pair/model target/frozen request probe mismatch")
    require(pair["selection"] == {"terminal_sha256": execution["selection_result"]["sha256"],
            "activation_bridge_sha256": execution["selection_capsule"]["sha256"]}
            and pair["commands_sha256"] == PAIR.digest(pair["commands"])
            and set(pair["commands"]) == set(ORDER)
            and pair["checkpoints"] == {"NONE": spec["reference_checkpoint_sha256"], "F1": spec["checkpoint_sha256"]},
            "pair model/commands/checkpoints binding")
    windows = execution["windows"]
    require(set(windows) == {"lease_ttl_s", "execution_window_s", "max_wall_s", "admission_max_age_s", "admission_valid_s", "provider_wait_s"}
            and all(E._positive(v) for v in windows.values())
            and windows["admission_max_age_s"] == JOB2.FRESH_CAPTURE_S and windows["admission_valid_s"] == 120
            and windows["max_wall_s"] <= windows["execution_window_s"] <= windows["lease_ttl_s"]
            and 2 * windows["max_wall_s"] <= pair["budget_gpu_s"] <= spec["budget_gpu_s"], "registered pair time/budget windows")
    resource = resolver.read(execution["host_resource"])
    require(set(protocol["provider_target"]) == {"resource_id", "resource_name", "instance_type", "gpu_model"},
            "fixed provider identity/model fields")
    require(resource.get("schema") == SCHEMA + "_host_resource" and resource.get("status") == "granted"
            and resource["gpu_binding"] == GPU_BINDING and "gpu_uuid" not in resource
            and pair["resource_sha256"] == execution["host_resource"]["sha256"]
            and resource["target"] == protocol["provider_target"]
            and protocol["provider_target"]["gpu_model"] == "L40S", "host resource is provider-only, not a UUID grant")
    if not folder.exists():
        require(E._stamp(resource["issued_at_utc"]) <= current < E._stamp(resource["expires_at_utc"]), "host resource expired")
    basis_path = resolver.file(execution["lease_basis"])
    backend_path = resolver.file(execution["backend"]["protocol"])
    basis, backend = resolver.read(execution["lease_basis"]), resolver.read(execution["backend"]["protocol"])
    require(pair["protocols"] == {"source": execution["source_protocol_sha256_lf"],
            "probe": C.digest_lf_file(backend_path)}
            and JOB2.REQUIRED_SOURCE_FILES <= basis.get("source_sha256_lf", {}).keys(), "lease/backend source closure")
    U.verify_sources({"sources_lf": basis["source_sha256_lf"]}, request["source_anchor"], repo=repo, protocol_path=basis_path)
    if not fixture:
        U.verify_sources({"sources_lf": backend["source_sha256_lf"]}, execution["backend"]["source_anchor"],
                         repo=repo)
        U.verify_sources({"sources_lf": {}}, request["source_anchor"], repo=repo, protocol_path=backend_path)
    operations = {}
    require(set(execution["operations"]) == set(ORDER), "exactly two registered operations")
    for arm, side in (("NONE", "reference"), ("F1", "case")):
        op = execution["operations"][arm]
        require(op["side"] == side and op["arm"] == arm and type(op["seed"]) is int and op["seed"] == 2221
                and op["probe"] == probe and E.J.HEX64.fullmatch(str(op["operation_id"])) is not None
                and op["args"] == spec["args"] and op["source"]["checkpoint"]["sha256"] == pair["checkpoints"][arm],
                "operation/probe/CP/args identity")
        C.BACKEND.V1._name(op["run_name"])
        unit = root / "operations" / op["operation_id"]
        stage = unit / "raw"
        _directory(stage)
        E._logical(op["remote_stage"])
        require(fixture or stage.as_posix() == op["remote_stage"], "exact registered POSIX stage projection")
        require(PureWindowsPath(op["host_stage"]).is_absolute() or PurePosixPath(op["host_stage"]).is_absolute(),
                "original host stage required")
        if not fixture:
            require(all(op["source"][key]["sha256"] == U.INITIAL[side][key + "_sha256"]
                        for key in ("checkpoint", "telemetry")), "initial checkpoint/telemetry source only")
        source_resolver = E.Resolver(op["source_roots"], E.digest(op["source_roots"]))
        combined = E.Resolver(op["source_roots"] + binding["roots"], E.digest(op["source_roots"] + binding["roots"]))
        packet = E._logical(op["provider_packet_path"])
        require(sum(remote == packet or remote in packet.parents for remote, _local in resolver.rows) == 1,
                "provider packet outside registered transport root")
        stage_logical = E._logical(op["remote_stage"])
        require(all(remote != stage_logical and remote not in stage_logical.parents and stage_logical not in remote.parents
                    for remote, _local in combined.rows), "stage overlaps source/transport root")
        checkpoint, manifest = source_resolver.file(op["source"]["checkpoint"]), source_resolver.file(op["source"]["manifest"])
        E._source({"source": op["source"], **op}, source_resolver, {"seed": 2221, "arm": arm,
            "checkpoint_path": op["source"]["checkpoint"]["path"] if fixture else str(checkpoint),
            "checkpoint_sha256": pair["checkpoints"][arm],
            "training_manifest_path": op["source"]["manifest"]["path"] if fixture else str(manifest)})
        expected = execution["backend"]["expected_source"]
        require(expected["source_anchor"] == execution["backend"]["source_anchor"]
                and expected["protocol"]["sha256_lf"] == C.digest_lf_file(backend_path)
                and all(expected[key] == backend[key] for key in E.HASHES), "backend source snapshot")
        command = [str(python), "-B", str(repo / "bench/brev_recovery_worker_v1.py"),
                   "--job", str(stage / "job.json"), "--output", str(stage / "worker.json")]
        require(pair["commands"][arm] == {"operation_id": op["operation_id"], "stage": op["host_stage"],
            "remote_stage": op["remote_stage"], "run_name": op["run_name"], "args": op["args"], "argv": command, "probe": probe},
            "exact approved command descriptor/argv")
        operations[arm] = {"op": op, "unit": unit, "stage": stage, "checkpoint": checkpoint, "manifest": manifest}
    require(operations["NONE"]["unit"] != operations["F1"]["unit"], "distinct operation/stage required")
    prepared = {"status": "prepared", "request": request, "request_ref": request_ref, "protocol": protocol,
        "binding": binding, "execution": execution, "repo": repo, "lab": lab, "python": python,
        "folder": folder, "resolver": resolver, "operations": operations, "basis_path": basis_path,
        "basis": basis, "basis_sha": C.digest_lf_file(basis_path), "backend_path": backend_path,
        "backend": backend, "resource": resource, "snapshot": snapshot,
        "fixture": fixture, "now": now, "model_transport": model_transport}
    prepared["authority"] = verify_authority(prepared)
    return prepared


class GuardedReader:
    def __init__(self, reader, check):
        self.reader, self.check = reader, check

    @property
    def last_receipt(self):
        return self.reader.last_receipt

    def __call__(self):
        self.check()
        try:
            return self.reader()
        finally:
            self.check()


class BoundedPopen:
    """첫 wait만 등록 실행 창으로 줄이고 기존 소유 group cleanup의 후속 wait는 유지한다."""
    def __init__(self, check, max_wall_s, expected_argv):
        self.check, self.deadline = check, time.monotonic() + max_wall_s
        self.expected_argv = expected_argv
        self.processes = []
        self.identities = {}

    def __call__(self, args, **kwargs):
        self.check()
        require(list(args) == self.expected_argv and kwargs.get("start_new_session") is True, "unapproved child command/group")
        require(time.monotonic() < self.deadline, "execution window exhausted before child launch")
        process = U._owned_popen(args, **kwargs)
        self.processes.append(process)
        self.identities[process.pid] = U.OWNED._process_identity(process.pid)
        self.check()
        owner = self

        class Process:
            waited = False

            def __getattr__(self, name):
                return getattr(process, name)

            def wait(self, timeout=None):
                if not self.waited:
                    self.waited = True
                    remaining = max(0.001, owner.deadline - time.monotonic())
                    timeout = remaining if timeout is None else min(timeout, remaining)
                return process.wait(timeout=timeout)

        return cast(subprocess.Popen, Process())

    def close(self):
        for process in self.processes:
            if U.OWNED._group_members(process.pid):
                identity = self.identities.get(process.pid) or U.OWNED._process_identity(process.pid)
                require(U.OWNED._safe_cleanup(identity, process), "owned child cleanup unresolved")
            require(not U.OWNED._group_members(process.pid), "owned child group still alive")

    def groups_alive(self):
        return any(U.OWNED._group_members(process.pid) for process in self.processes)


class ActualServices:
    evidence_kind = E.ACTUAL

    def __init__(self):
        self.children = []

    def clock(self):
        return utc()

    def scope(self):
        return JOB2.canonical_lease_scope()

    def observation(self):
        return JOB2.observe_lease_root()

    def validate_basis(self, prepared):
        return JOB2.validate_protocol(prepared["basis_path"], repo=prepared["repo"])

    def acquire(self, prepared):
        return JOB2.resource_lease(prepared["basis_path"], repo=prepared["repo"],
            ttl_s=prepared["execution"]["windows"]["lease_ttl_s"])

    def verify(self, prepared, lease, basis):
        require(JOB2.scope_active_root() == JOB2.canonical_lease_root(), "canonical lease alias changed during parent lifetime")
        return {**JOB2.verify_lease(lease, protocol=basis, protocol_sha=prepared["basis_sha"], now=self.clock()),
                "evidence_kind": self.evidence_kind}

    def boot(self, path):
        path.write_bytes(Path("/proc/sys/kernel/random/boot_id").read_bytes())
        return raw_ref(path)

    def admission(self, prepared, folder, lease, basis, check, provider):
        provider_observed = E._stamp(provider["observed_at_utc"])
        require(timedelta(0) <= self.clock() - provider_observed <= timedelta(seconds=JOB2.FRESH_CAPTURE_S),
                "provider observation stale/future at entry")
        folder.mkdir(parents=True, exist_ok=False)
        queries = []
        for stem, command in (("gpu", E.GPU.GPU_COMMAND), ("process", E.GPU.PROCESS_COMMAND), ("model", MODEL_COMMAND)):
            check()
            try:
                result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8",
                                        errors="strict", timeout=30, check=False)
            finally:
                check()
            (folder / (stem + ".csv")).write_bytes(result.stdout.encode("utf-8"))
            (folder / (stem + ".stderr")).write_bytes(result.stderr.encode("utf-8"))
            E.write_new(folder / (stem + ".receipt.json"), {"command": command, "exit_code": result.returncode,
                "observed_at_utc": self.clock().isoformat(), "stdout_sha256": E.J.file_sha(folder / (stem + ".csv"))})
            queries.append({"command": command, "observed_at_utc": self.clock().isoformat()})
            require(result.returncode == 0 and not result.stderr, "admission GPU query failed")
        observed = self.clock()
        require(timedelta(0) <= observed - provider_observed <= timedelta(seconds=JOB2.FRESH_CAPTURE_S),
                "provider observation stale/future after query")
        lease_body = U.read_ref(lease)
        acquired, lease_expires = E._stamp(lease_body["acquired_at_utc"]), E._stamp(lease_body["expires_at_utc"])
        require(acquired <= observed < lease_expires, "current admission outside acquired physical lease")
        expiry = min(E._stamp(basis["resources"]["deadline_utc"]), observed + timedelta(seconds=120))
        E.write_new(folder / "admission-times.json", {"schema": SCHEMA + "_admission_times",
            "provider_observed_at_utc": provider["observed_at_utc"], "queries": queries,
            "admission_at_utc": observed.isoformat(), "lease_acquired_at_utc": lease_body["acquired_at_utc"],
            "lease_expires_at_utc": lease_body["expires_at_utc"], "lease": lease})
        body = JOB2.build_resource_proof(prepared["basis_path"], provider_inventory_path=Path(provider["inventory_native"]["path"]),
            nvidia_gpu_csv_path=folder / "gpu.csv", nvidia_process_csv_path=folder / "process.csv",
            observed_at_utc=observed.isoformat(), expires_at_utc=expiry.isoformat(), lease_reference=lease, repo=prepared["repo"])
        E.write_new(folder / "resource.json", body)
        return U.ref(folder / "resource.json"), raw_ref(folder / "model.csv")

    def capture(self, prepared, capture_ref, basis):
        return JOB2.validate_resource_proof(Path(capture_ref["path"]), capture_ref["sha256"], protocol=basis,
            protocol_sha=prepared["basis_sha"], now=self.clock(), require_fresh=True)

    def record_lock(self, prepared, folder, lease, live):
        paths = JOB2.lease_paths(live["resource_id"])
        for name, path in {"proc-locks.txt": Path("/proc/locks"),
                           "owner-stat.txt": Path(f"/proc/{live['owner']['pid']}/stat"),
                           "heartbeat.json": paths["heartbeat"], "lease.json": Path(lease["path"])}.items():
            (folder / name).write_bytes(path.read_bytes())

    def reader(self, capture, folder):
        return E.GPU.StrictRemoteGpuReader(capture, folder)

    def release_observation(self, prepared, lease):
        require(os.name == "posix", "kernel release verification requires POSIX")
        snapshot = U.read_ref(lease)
        folder = prepared["folder"] / "lease-release"
        folder.mkdir()
        paths = JOB2.lease_paths(snapshot["resource_id"])
        locks = Path("/proc/locks").read_bytes()
        (folder / "proc-locks.txt").write_bytes(locks)
        (folder / "heartbeat.json").write_bytes(paths["heartbeat"].read_bytes())
        heartbeat = E.J._read(folder / "heartbeat.json")
        identity = snapshot["lock_identity"]
        owner = (snapshot["owner"]["pid"], os.major(identity["device"]), os.minor(identity["device"]), identity["inode"])
        require(heartbeat["active"] is False and heartbeat["lease_id"] == snapshot["lease_id"]
                and owner not in JOB2.proc_lock_owners(locks.decode("utf-8")), "physical lease release not verified")
        return {"released": True, "actual_kernel_release_verified": True, "lease": lease,
                "raw": [raw_ref(folder / name) for name in ("proc-locks.txt", "heartbeat.json")]}

    def backend(self, prepared, live):
        popen = BoundedPopen(live["check"], prepared["protocol"]["max_wall_s"], prepared["plan"]["approved_argv"])
        self.children.append(popen)
        backend = U.OWNED.SafeRecoveryBackend(prepared["lab"], prepared["python"], prepared["backend_protocol"],
            prepared["protocol"]["backend"]["source_anchor"], read_gpu=live["reader"], popen=popen)
        return backend, popen.close

    def ensure_children_closed(self, prepared, lease):
        """실제 종료를 확인하기 전에는 mutex context를 나가지 않는다. VM Stop은 main만 수행한다."""
        try:
            for child in self.children:
                child.close()
            return
        except BaseException as error:  # noqa: BLE001 - 종료 불명은 권한 해제 대신 격리한다.
            require(prepared["execution"]["unverified_child_policy"] == UNVERIFIED_CHILD_POLICY, "unregistered quarantine policy")
            event = {"schema": SCHEMA + "_quarantine", "status": "pending_unknown", "redispatch": False,
                "physical_mutex_retained": True, "main_vm_stop_required": True, "lease": lease,
                "reason": type(error).__name__, "observed_at_utc": self.clock().isoformat()}
            try:
                E.write_new(prepared["folder"] / "quarantine.json", event)
            except OSError:
                event["quarantine_file_write_failed"] = True
            try:
                print(json.dumps(event, ensure_ascii=False), file=sys.stderr, flush=True)
            except OSError:
                pass
        poll_error_count, last_poll_error = 0, None
        while True:
            try:
                if not any(child.groups_alive() for child in self.children):
                    break
            except BaseException as error:  # noqa: BLE001 - 확인 불가를 자식 종료로 바꾸지 않는다.
                poll_error_count += 1
                last_poll_error = type(error).__name__
            try:
                time.sleep(1)
            except BaseException as error:  # noqa: BLE001 - main은 부모 종료 전에 VM Stop을 완료해야 한다.
                poll_error_count += 1
                last_poll_error = type(error).__name__
        E.write_new(prepared["folder"] / "quarantine-resolved.json", {
            "owned_groups_absent": True, "observed_at_utc": self.clock().isoformat(), "lease": lease,
            "poll_error_count": poll_error_count, "last_poll_error": last_poll_error})


@contextmanager
def _owner_scope(services):
    global _OWNER
    with _LOCK:
        require(_OWNER is None, "nested/concurrent pair child forbidden")
        _OWNER = threading.get_ident()
        try:
            with services.scope():
                yield
        finally:
            _OWNER = None


@contextmanager
def _acquire(prepared, services):
    lease = None
    try:
        with services.acquire(prepared) as lease:
            try:
                yield lease
            finally:
                services.ensure_children_closed(prepared, lease)
    finally:
        if lease is not None:
            E.write_new(prepared["folder"] / "lease-release.json", services.release_observation(prepared, lease))


def _proof(prepared, arm, lease, basis, services, check, boot_ref, first_uuid):
    op = prepared["operations"][arm]["op"]
    folder = prepared["folder"] / op["operation_id"]
    folder.mkdir(parents=True, exist_ok=False)
    check()
    packet_path = op["provider_packet_path"]
    E._logical(packet_path)
    deadline = time.monotonic() + prepared["execution"]["windows"]["provider_wait_s"]
    logical = E._logical(packet_path)
    remote, local = next((remote, local) for remote, local in prepared["resolver"].rows
                         if remote == logical or remote in logical.parents)
    future = local.joinpath(*logical.relative_to(remote).parts)
    while not future.exists():
        check()
        require(not any(p.is_symlink() for p in (future, *future.parents)), "provider packet ancestor changed")
        require(time.monotonic() < deadline, "fresh host provider packet missing")
        time.sleep(0.25)
    local_packet = prepared["resolver"].path(packet_path)
    packet_ref = U.ref(local_packet)
    provider = U.read_ref(packet_ref)
    require(provider.get("schema") == SCHEMA + "_provider_packet" and provider["operation_id"] == op["operation_id"]
            and timedelta(0) <= services.clock() - E._stamp(provider["observed_at_utc"])
            <= timedelta(seconds=prepared["execution"]["windows"]["admission_max_age_s"]), "stale/wrong host provider packet")
    inventory_path = prepared["resolver"].file(provider["inventory"])
    provider["inventory_native"] = U.ref(inventory_path)
    E.write_new(folder / "provider-packet-binding.json", {"packet": packet_ref, "original": U.read_ref(packet_ref)})
    capture_ref, model_ref = services.admission(prepared, folder / "admission", lease, basis, check, provider)
    check()
    capture = services.capture(prepared, capture_ref, basis)
    check()
    current = services.clock()
    raw_capture = U.read_ref(capture_ref)
    observed, admission_expiry = E._stamp(raw_capture["observed_at_utc"]), E._stamp(raw_capture["expires_at_utc"])
    windows = prepared["execution"]["windows"]
    require(timedelta(0) <= current - observed <= timedelta(seconds=windows["admission_max_age_s"])
            and current < admission_expiry <= observed + timedelta(seconds=120), "stale admission capture")
    require(capture["lease"] == lease and capture["resource_id"] == prepared["protocol"]["provider_target"]["resource_id"]
            and (first_uuid is None or capture["gpu_uuid"] == first_uuid), "current boot resource/GPU changed")
    model_path = Path(model_ref["path"])
    require(E.J.file_sha(model_path) == model_ref["sha256"], "model observation SHA")
    row = model_path.read_text(encoding="utf-8").strip().split(",")
    require(len(row) == 2 and row[0].strip() == capture["gpu_uuid"] and row[1].strip() in {"L40S", "NVIDIA L40S"},
            "fresh GPU model/UUID must be L40S")
    latest_boot = services.boot(folder / "boot-current.txt")
    require(latest_boot["sha256"] == boot_ref["sha256"], "boot changed during physical lease")
    snapshot = U.read_ref(lease)
    expires = min(E._stamp(snapshot["expires_at_utc"]), E._stamp(prepared["resource"]["expires_at_utc"]),
        E._stamp(prepared["protocol"]["deadline_utc"]), current + timedelta(seconds=windows["execution_window_s"]))
    require(expires - current >= timedelta(seconds=windows["max_wall_s"]), "insufficient registered execution window")
    proof = {"schema": "brev_resource_proof_v1", "status": "granted", "evidence_kind": services.evidence_kind,
        **capture["capture"], "lease_id": snapshot["lease_id"], "protocol_sha256": prepared["request"]["protocol"]["sha256"],
        "issued_at_utc": current.isoformat(), "expires_at_utc": expires.isoformat(),
        "operation_id": op["operation_id"], "pair_id": prepared["execution"]["pair_id"],
        "lease_basis_sha256": prepared["execution"]["lease_basis"]["sha256"], "lease_snapshot_sha256": lease["sha256"],
        "capture_parent_sha256": capture_ref["sha256"], "authority_sha256": prepared["authority"]["local_copy"]["sha256"],
        "authority_request_sha256": prepared["authority"]["request_sha256"], "boot_sha256": boot_ref["sha256"],
        "gpu_model_observation_sha256": model_ref["sha256"], "gpu_binding": GPU_BINDING,
        "provider_packet_binding_sha256": E.J.file_sha(folder / "provider-packet-binding.json"),
        "admission_times_sha256": E.J.file_sha(folder / "admission/admission-times.json"),
        "model_transport_sha256": E.digest(prepared["model_transport"]),
        "admission_observed_at_utc": observed.isoformat(), "admission_expires_at_utc": admission_expiry.isoformat(),
        "admission_is_entry_freshness_only": True, "physical_mutex_is_probe_authority": False}
    check()
    verify_authority(prepared)
    E.write_new(folder / "probe-resource.json", proof)
    return folder, proof, capture_ref, capture["capture"], model_ref


def _plan(prepared, arm, folder, proof, capture_ref, capture, model_ref, lease, boot_ref):
    details = prepared["operations"][arm]
    op = details["op"]
    logical_parent = folder.as_posix() if not prepared["fixture"] else "/parent/" + op["operation_id"]
    roots = deepcopy(op["source_roots"] + prepared["binding"]["roots"])
    roots += [{"remote": logical_parent, "local": str(folder)},
              {"remote": op["remote_stage"], "local": str(details["stage"])}]
    resolver = E.Resolver(roots[:-1], E.digest(roots[:-1]))
    for prefix in ("provider_inventory", "nvidia_gpu_csv", "nvidia_process_csv"):
        resolver.file({"path": capture[prefix + "_path"], "sha256": capture[prefix + "_sha256"]})
    with (folder / "registered-protocol.json").open("xb") as stream:
        stream.write(Path(prepared["request"]["protocol"]["path"]).read_bytes())
    resource = {"proof": {"path": logical_parent + "/probe-resource.json", "sha256": E.J.file_sha(folder / "probe-resource.json")},
        "protocol": {"path": logical_parent + "/registered-protocol.json", "sha256": prepared["request"]["protocol"]["sha256"]},
        **{key: proof[key] for key in ("resource_id", "gpu_uuid", "lease_id", "protocol_sha256")}, "capture": capture}
    plan = {"schema": SCHEMA + "_plan", "authority_kind": "registered_consumed_pair_prefix",
        "evidence_kind": prepared["binding"]["evidence_kind"], "action": "probe", "identity": {
            key: op[key] for key in ("side", "arm", "seed", "probe", "run_name")}, "operation_id": op["operation_id"],
        "stage": str(details["stage"]), "logical_stage": op["remote_stage"], "source": op["source"],
        "expected_source": prepared["execution"]["backend"]["expected_source"], "roots": roots, "resource": resource,
        "approved_argv": prepared["binding"]["expected_request"]["commands"][arm]["argv"],
        "parent_request": prepared["request_ref"], "authority": prepared["authority"],
        "model_transport": prepared["model_transport"],
        "lease_basis": prepared["execution"]["lease_basis"], "lease_snapshot": lease, "capture_parent": capture_ref,
        "boot": boot_ref, "model_observation": model_ref, "proof": U.ref(folder / "probe-resource.json")}
    plan["provider_packet_binding"] = U.ref(folder / "provider-packet-binding.json")
    plan["admission_times"] = U.ref(folder / "admission/admission-times.json")
    E.write_new(folder / "plan.json", plan)
    return plan, U.ref(folder / "plan.json")


def _dispatch(prepared, arm, plan, plan_ref, services, check):
    details = prepared["operations"][arm]
    unit = details["unit"]
    E.write_new(unit / "plan-ref.json", plan_ref)
    require(not details["stage"].exists(), "raw stage already exists; redispatch forbidden")
    check()
    verify_authority(prepared)
    stage = details["stage"]
    backend_protocol = {**prepared["backend"], "protocol_path": str(prepared["backend_path"]),
                        "protocol_sha256_lf": C.digest_lf_file(prepared["backend_path"])}
    unit_view = {"plan": plan, "stage": stage, "unit": unit, "lab": prepared["lab"], "python": prepared["python"],
        "repo": prepared["repo"], "backend_protocol": backend_protocol, "protocol": {
            "max_wall_s": prepared["execution"]["windows"]["max_wall_s"], "backend": prepared["execution"]["backend"]}}
    reader = GuardedReader(services.reader(plan["resource"]["capture"], unit / "gpu-observations"), check)
    backend, close = services.backend(unit_view, {"reader": reader, "check": check})
    E.write_new(unit / "dispatch-intent.json", {"plan": plan_ref, "authority": prepared["authority"],
        "started_at_utc": services.clock().isoformat(), "host_completion_seal_required": True})
    try:
        check()
        raw = backend.probe(stage, {"arm": arm, "seed": 2221, "checkpoint": {
            "path": str(details["checkpoint"]), "sha256": plan["source"]["checkpoint"]["sha256"]},
            "manifest_path": str(details["manifest"])}, plan["identity"]["probe"], plan["identity"]["run_name"])
    finally:
        close()
        check()
    E.write_new(stage / "backend_result.json", raw)
    request = U._request(unit_view)
    evidence = E.build(request, plan["roots"], expected_request_sha256=E.digest(request),
        expected_roots_sha256=E.digest(plan["roots"]), allow_fixture=prepared["fixture"])
    E.verify(evidence, plan["roots"], expected_request_sha256=E.digest(request),
        expected_roots_sha256=E.digest(plan["roots"]), allow_fixture=prepared["fixture"])
    check()
    E.write_new(unit / "request.json", request)
    E.write_new(unit / "evidence.json", evidence)
    result = {"status": "complete", "operation_id": plan["operation_id"], "plan": plan_ref,
        "request_sha256": E.digest(request), "roots_sha256": E.digest(plan["roots"]), "evidence": U.ref(unit / "evidence.json")}
    E.write_new(unit / "outcome.json", result)
    return result


def _verify_completion(prepared, result):
    plan = U.read_ref(result["plan"])
    evidence = U.read_ref(result["evidence"])
    require(plan["parent_request"] == prepared["request_ref"] and plan["authority"] == prepared["authority"],
            "completion parent/authority binding")
    require(plan["model_transport"] == prepared["model_transport"], "completion snapshot history transport changed")
    arm = plan["identity"]["arm"]
    require(arm in ORDER, "completion arm")
    op = prepared["operations"][arm]["op"]
    folder = prepared["folder"] / op["operation_id"]
    require(result["plan"]["path"] == str(folder / "plan.json")
            and result["evidence"]["path"] == str(prepared["operations"][arm]["unit"] / "evidence.json"),
            "completion location differs from registered operation")
    require(plan["identity"] == {key: op[key] for key in ("side", "arm", "seed", "probe", "run_name")}
            and plan["operation_id"] == op["operation_id"] and plan["source"] == op["source"]
            and plan["logical_stage"] == op["remote_stage"]
            and plan["stage"] == str(prepared["operations"][arm]["stage"])
            and plan["expected_source"] == prepared["execution"]["backend"]["expected_source"]
            and plan["approved_argv"] == prepared["binding"]["expected_request"]["commands"][arm]["argv"]
            and plan["evidence_kind"] == evidence["evidence_kind"] == prepared["binding"]["evidence_kind"],
            "completed plan differs from registered operation/source/kind")
    logical_parent = folder.as_posix() if not prepared["fixture"] else "/parent/" + op["operation_id"]
    roots = op["source_roots"] + prepared["binding"]["roots"] + [
        {"remote": logical_parent, "local": str(folder)},
        {"remote": op["remote_stage"], "local": str(prepared["operations"][arm]["stage"])}]
    require(plan["roots"] == roots and result["roots_sha256"] == E.digest(roots), "completion root-map changed")
    expected = U._request({"plan": plan, "stage": prepared["operations"][arm]["stage"],
        "protocol": {"max_wall_s": prepared["execution"]["windows"]["max_wall_s"]}})
    require(evidence["request"] == expected and result["request_sha256"] == E.digest(expected),
            "completion differs from exact registered dispatch")
    proof = U.read_ref(plan["proof"])
    for key, reference in (("lease_snapshot_sha256", plan["lease_snapshot"]), ("capture_parent_sha256", plan["capture_parent"]),
                           ("boot_sha256", plan["boot"]), ("gpu_model_observation_sha256", plan["model_observation"]),
                           ("provider_packet_binding_sha256", plan["provider_packet_binding"])):
        require(E.J.file_sha(Path(reference["path"])) == reference["sha256"] == proof[key], "completion raw parent changed")
    require(proof["authority_sha256"] == prepared["authority"]["local_copy"]["sha256"]
            and proof["authority_request_sha256"] == prepared["authority"]["request_sha256"]
            and proof["protocol_sha256"] == prepared["request"]["protocol"]["sha256"]
            and proof["operation_id"] == plan["operation_id"] and proof["evidence_kind"] == plan["evidence_kind"],
            "completion proof authority/protocol/operation")
    require(proof["model_transport_sha256"] == E.digest(prepared["model_transport"]), "completion model transport SHA")
    require(plan["lease_basis"] == prepared["execution"]["lease_basis"]
            and proof["lease_basis_sha256"] == plan["lease_basis"]["sha256"]
            and proof["pair_id"] == prepared["execution"]["pair_id"] and proof["gpu_binding"] == GPU_BINDING,
            "completion lease/pair registration changed")
    packet_binding = U.read_ref(plan["provider_packet_binding"])
    require(U.read_ref(packet_binding["packet"]) == packet_binding["original"], "host provider packet changed after entry")
    admission, lease = U.read_ref(plan["capture_parent"]), U.read_ref(plan["lease_snapshot"])
    times = U.read_ref(plan["admission_times"])
    require(plan["admission_times"]["sha256"] == proof["admission_times_sha256"]
            and times["lease"] == plan["lease_snapshot"]
            and times["provider_observed_at_utc"] == packet_binding["original"]["observed_at_utc"]
            and times["admission_at_utc"] == admission["observed_at_utc"]
            and times["lease_acquired_at_utc"] == lease["acquired_at_utc"]
            and E._stamp(lease["acquired_at_utc"]) <= E._stamp(times["admission_at_utc"]) < E._stamp(lease["expires_at_utc"])
            and timedelta(0) <= E._stamp(times["admission_at_utc"]) - E._stamp(times["provider_observed_at_utc"])
            <= timedelta(seconds=prepared["execution"]["windows"]["admission_max_age_s"])
            and all(E._stamp(times["lease_acquired_at_utc"]) <= E._stamp(query["observed_at_utc"])
                    <= E._stamp(times["admission_at_utc"]) for query in times["queries"]), "separate provider/query/admission/lease times")
    require(proof["admission_observed_at_utc"] == admission["observed_at_utc"]
            and proof["admission_expires_at_utc"] == admission["expires_at_utc"]
            and admission["lease"] == plan["lease_snapshot"] and admission["capture"] == plan["resource"]["capture"],
            "completion admission changed")
    issued, expires, observed = (E._stamp(proof[name]) for name in
                                ("issued_at_utc", "expires_at_utc", "admission_observed_at_utc"))
    windows = prepared["execution"]["windows"]
    require(observed <= issued < E._stamp(admission["expires_at_utc"])
            and issued - observed <= timedelta(seconds=windows["admission_max_age_s"])
            and issued + timedelta(seconds=windows["max_wall_s"]) <= expires <= min(
                E._stamp(lease["expires_at_utc"]), E._stamp(prepared["resource"]["expires_at_utc"]),
                E._stamp(prepared["protocol"]["deadline_utc"]), issued + timedelta(seconds=windows["execution_window_s"])),
            "completed execution window differs from registered lifetime")
    E.verify(evidence, plan["roots"], expected_request_sha256=result["request_sha256"],
        expected_roots_sha256=result["roots_sha256"], allow_fixture=prepared["fixture"])
    return evidence


def _verify_pair_result(prepared, result):
    require(result.get("schema") == SCHEMA + "_result" and result.get("status") in {"complete", "HOLD", "pending_unknown"}
            and result.get("host_completion_seal_required") is True and result.get("c_loop_update_permitted") is False,
            "pair outcome cannot grant host loop authority")
    entries = result["completed"]
    require(isinstance(entries, list) and len(entries) <= 2
            and (result["status"] != "complete" or len(entries) == 2), "pair completion cardinality")
    previous, first = None, None
    for arm, entry in zip(ORDER, entries, strict=False):
        require(entry["operation_id"] == prepared["operations"][arm]["op"]["operation_id"], "pair result order/operation changed")
        evidence = _verify_completion(prepared, entry)
        plan = U.read_ref(entry["plan"])
        proof = U.read_ref(plan["proof"])
        execution = E.Resolver(plan["roots"], E.digest(plan["roots"])).read(evidence["request"]["files"]["execution"])
        identity = (proof["boot_sha256"], proof["gpu_uuid"], proof["lease_snapshot_sha256"])
        require(first is None or identity == first, "completed pair crossed boot/GPU/physical lease")
        require(previous is None or E._stamp(previous) <= E._stamp(execution["started_at"]), "pair GPU executions overlap")
        first, previous = identity, execution["finished_at"]


def lookup(prepared):
    folder = prepared["folder"]
    if not (folder / "intent.json").exists():
        return {"status": "pending_unknown", "redispatch": False}
    intent = E.J._read(folder / "intent.json")
    require(intent["request_ref"] == prepared["request_ref"] and intent["authority"] == prepared["authority"], "existing pair intent differs")
    if not (folder / "outcome.json").exists():
        return {"status": "pending_unknown", "redispatch": False}
    result = E.J._read(folder / "outcome.json")
    _verify_pair_result(prepared, result)
    if "lease_release" in result:
        U.read_ref(result["lease_release"])
    return result


def run(request_ref: dict, *, execute: bool = False, allow_fixture: bool = False,
        home: Path | None = None, now: datetime | None = None, services=None) -> dict:
    prepared = prepare(request_ref, allow_fixture=allow_fixture, home=home, now=now)
    if prepared["status"] == "HOLD":
        return prepared
    folder = prepared["folder"]
    if folder.exists():
        return lookup(prepared)
    if not execute:
        return {"status": "prepared", "dispatched": False, "authority_kind": "registered_consumed_pair_prefix"}
    fixture = prepared["fixture"]
    require(fixture or (os.name == "posix" and services is None), "actual requires POSIX/default services")
    require(not fixture or (services is not None and services.evidence_kind == E.FIXTURE), "explicit CPU lifetime services required")
    services = services or ActualServices()
    folder.mkdir(parents=True, exist_ok=False)
    E.write_new(folder / "intent.json", {"schema": SCHEMA + "_intent", "request_ref": request_ref,
        "authority": prepared["authority"], "order": list(ORDER), "gpu_binding": GPU_BINDING})
    completed = []
    result = {"schema": SCHEMA + "_result", "status": "pending_unknown", "redispatch": False,
        "completed": completed, "host_completion_seal_required": True, "c_loop_update_permitted": False}
    try:
        for details in prepared["operations"].values():
            details["unit"].mkdir(parents=True, exist_ok=False)
            E.write_new(details["unit"] / "parent-intent.json", {"request_ref": request_ref, "pair_id": prepared["execution"]["pair_id"]})
        with _owner_scope(services):
            E.write_new(folder / "canonical-lease-root.json", services.observation())
            basis = services.validate_basis(prepared)
            if not fixture:
                require(all(basis["_recording_target"][key] == prepared["protocol"]["provider_target"][key]
                            for key in ("resource_id", "resource_name", "instance_type")), "lease/provider target mismatch")
            with _acquire(prepared, services) as lease:
                snapshot = U.read_ref(lease)
                require(snapshot["resource_id"] == prepared["protocol"]["provider_target"]["resource_id"]
                        and snapshot["protocol_sha256_lf"] == prepared["basis_sha"], "acquired lease resource/protocol")
                boot_ref = services.boot(folder / "boot.txt")

                def check():
                    require(_OWNER == threading.get_ident(), "pair lifetime owner thread")
                    before = services.verify(prepared, lease, basis)
                    require(before["lease_id"] == snapshot["lease_id"] and before["resource_id"] == snapshot["resource_id"]
                            and before["evidence_kind"] == prepared["binding"]["evidence_kind"]
                            and (fixture or before["live_lock_proof"] is True), "physical mutex inactive/lost/fixture")
                    logs = folder / "lease-observations"
                    log = logs / f"{len(list(logs.glob('*'))) + 1:05d}"
                    log.mkdir(parents=True, exist_ok=False)
                    services.record_lock(prepared, log, lease, before)
                    after = services.verify(prepared, lease, basis)
                    require(all(after.get(key) == before.get(key) for key in (
                        "owner", "lease_id", "resource_id", "live_lock_proof", "evidence_kind", "reference", "lock_proof"))
                        and after.get("heartbeat_sequence", 0) >= before.get("heartbeat_sequence", 0),
                        "physical owner changed during observation")
                    E.write_new(log / "proof.json", {"physical_mutex_only": True, "probe_authority_issued": False,
                        "lease": lease, "live": after, "raw_sha256": {p.name: E.J.file_sha(p) for p in log.iterdir()}})

                first_uuid = None
                for arm in ORDER:
                    check()
                    proof_folder, proof, capture_ref, capture, model_ref = _proof(
                        prepared, arm, lease, basis, services, check, boot_ref, first_uuid)
                    first_uuid = proof["gpu_uuid"]
                    plan, plan_ref = _plan(prepared, arm, proof_folder, proof, capture_ref, capture, model_ref, lease, boot_ref)
                    completed.append(_dispatch(prepared, arm, plan, plan_ref, services, check))
                    check()
                    _verify_completion(prepared, completed[-1])
                    if len(completed) == 2:
                        first = U.read_ref(completed[0]["evidence"])["request"]["files"]["execution"]
                        second = U.read_ref(completed[1]["evidence"])["request"]["files"]["execution"]
                        first_plan, second_plan = (U.read_ref(entry["plan"]) for entry in completed)
                        before = E.Resolver(first_plan["roots"], E.digest(first_plan["roots"])).read(first)
                        after = E.Resolver(second_plan["roots"], E.digest(second_plan["roots"])).read(second)
                        require(E._stamp(before["finished_at"]) <= E._stamp(after["started_at"]), "pair GPU executions overlap")
                    check()
                result.update(status="complete", gpu_uuid=first_uuid, boot_sha256=boot_ref["sha256"])
                _verify_pair_result(prepared, result)
                check()
            result["lease_released"] = True
    except BaseException as error:  # noqa: BLE001 - 중단·부분 결과도 보존하고 자동 재실행을 막는다.
        result.update(status="HOLD" if isinstance(error, Exception) else "pending_unknown",
                      error_type=type(error).__name__, reason=str(error), raw_preserved=True)
    if (folder / "lease-release.json").exists():
        result["lease_release"] = U.ref(folder / "lease-release.json")
    E.write_new(folder / "outcome.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--request-sha256")
    parser.add_argument("--model-transport", type=Path)
    parser.add_argument("--model-transport-sha256")
    parser.add_argument("--model-root", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    try:
        if args.model_transport is not None:
            require(args.request is None and not args.execute and args.model_root is not None,
                    "model transport is read-only and cannot authorize execution")
            if args.model_root is None:
                raise ValueError("model root required")
            envelope = U.read_ref({"path": str(args.model_transport.absolute()), "sha256": args.model_transport_sha256})
            require(envelope.get("schema") == SCHEMA + "_model_transport_request", "model transport request schema")
            roots = [{"remote": "/bundle", "local": str(args.model_root.resolve(strict=True))}]
            _snapshot, _spec, transport = _selection(envelope["execution"], E.Resolver(roots, E.digest(roots)))
            result = {"status": "model_transport_valid", "transport": transport, "execution_permitted": False,
                      "probe_authority_issued": False, "root_map_sha256": E.digest(roots), "native_platform": os.name}
        else:
            require(args.request is not None and args.model_root is None, "request/explicit SHA required")
            if args.request is None:
                raise ValueError("request required")
            result = run({"path": str(args.request.absolute()), "sha256": args.request_sha256}, execute=args.execute)
    except Exception as error:  # noqa: BLE001 - CLI 실패를 명시 HOLD로 표시한다.
        result = {"status": "HOLD", "dispatched": False, "error_type": type(error).__name__, "reason": str(error)}
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
