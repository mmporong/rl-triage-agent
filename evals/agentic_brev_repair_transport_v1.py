"""Source-bound repair transport and bounded, layered raw-byte packets."""
from __future__ import annotations

import base64
import hashlib
import inspect
import stat
import threading
import zlib
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import agentic_brev_probe_transport_remote_v3 as PROBE
import agentic_brev_repair_evidence_v1 as E
import agentic_brev_runtime_host_v2 as CORE

SCHEMA = "agentic_brev_repair_transport_v1"
SOURCE = "evals/agentic_brev_repair_transport_v1.py"
SERVICE_SOURCE = "evals/agentic_brev_repair_mailbox_service_v1.py"
REMOTE_SOURCE = "bench/agentic_brev_repair_transport_remote_v1.py"
NEW_SOURCES = {SOURCE, SERVICE_SOURCE, REMOTE_SOURCE,
    "tests/test_agentic_brev_repair_transport_v1.py", "tests/test_agentic_brev_repair_mailbox_service_v1.py",
    "tests/test_agentic_brev_repair_transport_remote_v1.py"}
SOURCES = set(CORE.SOURCES) | set(PROBE.SOURCES) | NEW_SOURCES
SOURCES |= {"evals/agentic_brev_mailbox_service_v1.py", "evals/agentic_brev_mailbox_service_v2.py",
            "evals/agentic_brev_mailbox_service_v3.py", "evals/agentic_brev_mailbox_transport_v1.py"}
# Explicit transitive local imports, including registered CPU fixture helpers.
SOURCES |= {
    "bench/inspect_optimizer_observability.py",
    "bench/optimizer_faults.py",
    "bench/ppo_ratio_mutation_v1.py",
    "bench/ppo_ratio_repair_training.py",
    "bench/r1_repair_training.py",
    "bench/residual_faults_v1.py",
    "bench/run_runtime_readback_smoke.py",
    "bench/run_vision_atomic_compatibility_v1.py",
    "bench/run_vision_brev_batch_v1.py",
    "bench/run_vision_brev_preflight_v1.py",
    "bench/run_vision_brev_render_v2.py",
    "bench/run_vision_checkpoint_smoke_v1.py",
    "bench/run_vision_checkpoint_smoke_v2.py",
    "bench/run_vision_corrected_pull_record_v1.py",
    "bench/run_vision_prepared_capture_v1.py",
    "bench/run_vision_prepared_record_v1.py",
    "bench/run_vision_prepared_record_v2.py",
    "bench/runtime_readback_smoke.py",
    "bench/upstream_faults.py",
    "bench/vision_atomic_capture_v1.py",
    "bench/vision_atomic_smoke_v1.py",
    "bench/vision_checkpoint_smoke_job_v1.py",
    "bench/vision_checkpoint_smoke_job_v2.py",
    "bench/vision_checkpoint_smoke_sdk_v1.py",
    "bench/vision_checkpoint_smoke_sdk_v2.py",
    "bench/vision_execution_evidence_v1.py",
    "bench/vision_execution_evidence_v2.py",
    "bench/vision_frame_sync_v1.py",
    "bench/vision_fxaa_policy_v1.py",
    "bench/vision_metadata_lifecycle_v1.py",
    "bench/vision_play_worker_v2.py",
    "bench/vision_prepared_capture_v1.py",
    "bench/vision_prepared_job_v1.py",
    "bench/vision_prepared_record_sdk_v1.py",
    "bench/vision_prepared_record_sdk_v2.py",
    "bench/vision_prepared_record_v1.py",
    "bench/vision_prepared_sdk_v1.py",
    "bench/vision_record_source_material_v1.py",
    "bench/vision_render_attestation_v1.py",
    "bench/vision_render_context_v1.py",
    "bench/vision_replicator_startup_v1.py",
    "bench/vision_semantic_diagnostic_v1.py",
    "bench/vision_semantic_sdk_v1.py",
    "bench/vision_timing_adapter_v1.py",
    "bench/vision_writer_payload_v1.py",
    "evals/agentic_evidence_index_v1.py",
    "evals/correction_model_call.py",
    "evals/export_ppo_ratio_repair_v2.py",
    "evals/feedback_model_call.py",
    "evals/grounded_vision_observer_v1.py",
    "evals/nemotron_bottleneck_ledger_v2.py",
    "evals/nemotron_endpoint_ledger_v3.py",
    "evals/ppo_gradient_invariant_v1.py",
    "evals/ppo_gradient_invariant_v2.py",
    "evals/ppo_ratio_pilot_v1.py",
    "evals/ppo_ratio_pilot_v2.py",
    "evals/ppo_ratio_repair_contract_v2.py",
    "evals/ppo_ratio_repair_contract.py",
    "evals/ppo_ratio_repair_demo_v2.py",
    "evals/ppo_ratio_repair_demo_v3.py",
    "evals/ppo_ratio_repair_model_call_v2.py",
    "evals/preregistered_seed_guard.py",
    "evals/r1_repair_contract.py",
    "evals/r1_repair_demo_v2.py",
    "evals/r1_repair_demo.py",
    "evals/r1_repair_execution_v2.py",
    "evals/r1_repair_model_call.py",
    "evals/residual_demo_v2.py",
    "evals/residual_model_call_v1.py",
    "evals/residual_model_call_v2.py",
    "evals/run_upstream_r1_smoke.py",
    "evals/run_upstream_smoke.py",
    "evals/vision_brev_render_backend_v2.py",
    "evals/vision_brev_render_backend_v3.py",
    "evals/vision_brev_render_backend_v4.py",
    "evals/vision_brev_render_backend_v5.py",
    "evals/vision_corrected_record_service_v1.py",
    "evals/vision_evidence_contract_v1.py",
    "evals/vision_model_spool_v1.py",
    "evals/vision_record_source_readiness_v1.py",
    "evals/vision_render_readiness_v2.py",
    "evals/vision_selection_call_v1.py",
    "evals/vision_selection_comparison_v1.py",
    "evals/vision_selection_contract_v1.py",
    "evals/vision_selection_e2e_v1.py",
    "evals/vision_selection_execution_v1.py",
    "evals/vision_selection_loop_v1.py",
    "evals/vision_selection_materialize_v1.py",
    "evals/vision_selection_sources_v1.py",
    "evals/vision_selection_worker_v1.py",
    "evals/vision_transferred_input_acceptance_v1.py",
    "evals/vision_transferred_input_acceptance_v2.py",
    "evals/vision_transferred_input_acceptance_v3.py",
    "evals/vision_transferred_probe_v1.py",
    "evals/vision_transferred_selection_v1.py",
    "src/rl_triage/__init__.py",
    "src/rl_triage/active_diagnosis.py",
    "src/rl_triage/correction_registry.py",
    "src/rl_triage/grounded_evidence.py",
    "src/rl_triage/runtime_observers.py",
    "src/rl_triage/single_run.py",
    "src/rl_triage/tracking_alarm.py",
    "src/rl_triage/triage_tools.py",
    "tests/test_agentic_investigation_evidence_v1.py",
    "tests/test_brev_recovery_backend_v4.py",
    "tests/test_optimizer_observability.py",
    "tests/test_vision_selection_contract_v1.py",
}
LAYERS = ("metadata", "evidence")
require = E.require
PROJECTION = SCHEMA + "_registered_manifest_paths_v1"
_MANIFEST_HOOK = "validate_manifest"
_MANIFEST_BASE = E.TRAIN.validate_manifest
_MANIFEST_ACTIVE = None


class RepairPending(RuntimeError):
    """The original dispatch stays pending; only lookup may follow."""


def clock():
    return datetime.now(timezone.utc)  # noqa: UP017 - Python 3.10


def safe(path, *, exists=False):
    path = Path(path)
    require(path.is_absolute(), "absolute native path")
    for parent in (path, *path.parents):
        if parent.exists() or parent.is_symlink():
            info = parent.lstat()
            require(not parent.is_symlink() and not (getattr(info, "st_file_attributes", 0)
                    & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)), "link/reparse ancestor")
    if exists:
        E.regular(path)
    return path


def new_bytes(path, raw):
    path = safe(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(raw)
    except FileExistsError:
        require(safe(path, exists=True).read_bytes() == raw, "write-once raw differs")
    return E.ref(path)


def save(path, value):
    return new_bytes(path, E.J._bytes(value))


def _maps(config, side):
    rows = []
    for layer in LAYERS:
        for row in config["layers"][layer]:
            value = {"remote": row["producer"], "local": row[side]}
            if value not in rows:
                rows.append(value)
    return rows


def registered(registration_ref, root_map, contract_ref, *, allow_fixture=False):
    reg = E.registration(registration_ref, root_map, allow_fixture=allow_fixture)
    config = reg["resolver"].read(contract_ref)
    fields = {"schema", "synthetic_cpu", "source_anchor", "sources_lf", "service_source_sha256_lf",
        "remote_source_sha256_lf", "target", "cli_argv_prefix", "host_mount_projection",
        "lifecycle_protocol", "deadline_utc", "lifecycle_stop_intent_path", "remote_stop_intent_path",
        "host_control_root", "remote_control_root", "remote_repo", "remote_python", "layers", "limits", "manifest_projection"}
    require(set(config) == fields and config["schema"] == SCHEMA + "_contract"
            and config["synthetic_cpu"] is allow_fixture
            and reg["protocol"].get("repair_transport_contract_sha256") == contract_ref["sha256"]
            and reg["protocol"].get("lifecycle_protocol_sha256") == config["lifecycle_protocol"]["sha256"],
            "registered repair IO/phase contract")
    require(config["manifest_projection"] == PROJECTION
            and reg["protocol"].get("repair_manifest_projection") == PROJECTION, "explicit manifest projection registration")
    require(SOURCES <= reg["protocol"]["sources_lf"].keys() and SOURCES <= config["sources_lf"].keys()
            and all(config["sources_lf"][key] == reg["protocol"]["sources_lf"][key] for key in SOURCES),
            "transport source closure")
    CORE.R.verify_sources({"sources_lf": config["sources_lf"]}, config["source_anchor"], repo=reg["repo"])
    CORE.H1._ancestor(reg["repo"], config["source_anchor"], reg["raw"]["runtime"]["anchor"])
    require(config["target"] == reg["layout"]["provider_target"]
            and config["service_source_sha256_lf"] == config["sources_lf"][SERVICE_SOURCE]
            and config["remote_source_sha256_lf"] == config["sources_lf"][REMOTE_SOURCE], "target/service/remote source")
    limits = config["limits"]
    require(set(limits) == {"rpc_timeout_s", "admission_wait_s", "owner_ttl_s", "command_wait_s",
                            "max_files", "max_file_bytes", "max_archive_bytes", "max_rpc_calls"}
            and all(type(v) is int and v > 0 for v in limits.values())
            and limits["admission_wait_s"] <= 120 and limits["command_wait_s"] <= 120
            and reg["contract"]["budget"]["wall_s"] + limits["command_wait_s"] < limits["owner_ttl_s"]
            and limits["rpc_timeout_s"] >= limits["owner_ttl_s"]
            and limits["max_files"] <= 2048 and limits["max_file_bytes"] <= 256 * 1024 * 1024
            and limits["max_archive_bytes"] <= 512 * 1024 * 1024, "explicit bounded IO/lifetime limits")
    for key in ("host_control_root", "lifecycle_stop_intent_path"):
        E.logical(config[key])
    if not allow_fixture:
        for key in ("remote_control_root", "remote_repo", "remote_python", "remote_stop_intent_path"):
            require(isinstance(E.logical(config[key]), PurePosixPath), "registered absolute POSIX runtime path")
        require(E.logical(config["remote_repo"]) == E.provider_path(reg["raw"]["runtime"]["repo"], reg)
                and E.logical(config["remote_python"]) == E.logical(reg["layout"]["python"]), "core/transport remote install binding")
    require(E.provider_path(config["host_control_root"], reg) == E.logical(config["remote_control_root"])
            and E.provider_path(config["lifecycle_stop_intent_path"], reg) == E.logical(config["remote_stop_intent_path"]), "control/Stop map binding")
    require(set(config["layers"]) == set(LAYERS), "separate metadata/evidence layers")
    for layer in LAYERS:
        rows = config["layers"][layer]
        require(isinstance(rows, list) and bool(rows), "registered layer roots")
        seen = []
        for row in rows:
            require(set(row) == {"producer", "host", "remote", "writable"}
                    and type(row["writable"]) is bool, "layer root fields")
            producer = E.logical(row["producer"])
            require(all(type(producer) is not type(old) or (producer != old
                    and producer not in old.parents and old not in producer.parents) for old in seen), "overlapping layer roots")
            seen.append(producer)
            require(allow_fixture or isinstance(E.logical(row["remote"]), PurePosixPath), "actual native remote root must be POSIX")
    for side, key in (("host", "host"), ("remote", "provider")):
        observed = {(r["remote"], r["local"]) for r in _maps(config, side)}
        expected = {(r["remote"], r["local"]) for r in reg["layout"]["root_maps"][key]}
        require(observed == expected, "IO maps must exactly match core maps")
    return reg, config


def locate(reference, config, layer, side, *, existing=True):
    require(layer in LAYERS and side in {"host", "remote"}, "registered map side/layer")
    p = E.logical(reference["path"])
    matches = []
    for index, row in enumerate(config["layers"][layer]):
        root = E.logical(row["producer"])
        if type(root) is type(p) and (root == p or root in p.parents):
            relative = p.relative_to(root).as_posix()
            require(relative not in {"", "."}, "file below root required")
            target = Path(row[side]).joinpath(*p.relative_to(root).parts)
            safe(target, exists=existing)
            matches.append((index, relative, target, row))
    require(len(matches) == 1, "reference outside unique layer root")
    return matches[0]


def refs_in(value, origin=None):
    refs = []
    if isinstance(value, dict):
        if {"path", "sha256"} <= value.keys():
            try:
                E.logical(value["path"])
            except ValueError:
                pass
            else:
                refs.append({k: value[k] for k in ("path", "sha256")})
        for key, text in value.items():
            if key.endswith("_path") and key[:-5] + "_sha256" in value:
                refs.append({"path": text, "sha256": value[key[:-5] + "_sha256"]})
        if origin is not None:
            for stem in ("gpu", "process"):
                if stem + "_stdout_file" in value:
                    name = value[stem + "_stdout_file"]
                    require(isinstance(name, str) and PurePosixPath(name).name == name and name not in {".", ".."}, "GPU sibling name")
                    refs.append({"path": str(E.logical(origin["path"]).parent / name), "sha256": value[stem + "_stdout_sha256"]})
        for child in value.values():
            if isinstance(child, (list, dict)):
                refs.extend(refs_in(child, origin))
    elif isinstance(value, list):
        for child in value:
            refs.extend(refs_in(child, origin))
    return refs


def layer_for(path):
    path = E.logical(str(path))
    if path.name in {"approval-prefix.jsonl", "corrections.ledger.jsonl", "probes.ledger.jsonl"}:
        return "metadata"
    return "evidence" if path.suffix in {".pt", ".jsonl"} or path.name in {
        "telemetry.json", "training.manifest.json", "training.readback.json", "job.json", "worker.json",
        "execution.json", "backend-result.json", "bundle.json", "stdout.log", "stderr.log", "NONE.json", "CORRECTED.json"} else "metadata"


def collect(seeds, reg, config, side, *, extras=()):
    """Traverse declared raw refs, never a directory or the repository itself."""
    todo = list(seeds) + list(extras)
    found = {}
    while todo:
        reference = todo.pop()
        key = E.J.digest(reference)
        if key in found:
            continue
        path = reg["resolver"].file(reference)
        require(path.stat().st_size <= config["limits"]["max_file_bytes"], "raw file size cap")
        # Control namespaces use metadata; training/report/checkpoint files use evidence.
        layer = layer_for(reference["path"])
        index, relative, target, row = locate(reference, config, layer, side)
        require(target == path, "core/layer resolver disagreement")
        found[key] = {"reference": deepcopy(reference), "layer": layer, "root_index": index,
                      "relative": relative, "path": path, "writable": row["writable"]}
        require(len(found) <= config["limits"]["max_files"], "raw file count cap")
        if path.suffix == ".json":
            raw = E.E.loads(path.read_text(encoding="utf-8"))
            todo.extend(refs_in(raw, reference))
            # Snapshot history has an original bare path, not a raw-ref wrapper.
            pending = [raw]
            while pending:
                value = pending.pop()
                if isinstance(value, dict):
                    if "ledger_path" in value:
                        native = reg["resolver"].path(value["ledger_path"])
                        todo.append({"path": value["ledger_path"], "sha256": E.J.file_sha(E.regular(native))})
                    pending.extend(v for v in value.values() if isinstance(v, (dict, list)))
                elif isinstance(value, list):
                    pending.extend(value)
    return list(found.values())


def packet(seeds, reg, config, side, *, operation, kind, extras=()):
    rows, files, targets = collect(seeds, reg, config, side, extras=extras), [], set()
    for row in rows:
        target_side = "remote" if side == "host" else "host"
        root = config["layers"][row["layer"]][row["root_index"]]
        target_key = str(E.logical(root[target_side]).joinpath(*PurePosixPath(row["relative"]).parts))
        if target_key in targets:
            continue  # registered aliases resolve to one immutable native file
        targets.add(target_key)
        data = row["path"].read_bytes()
        files.append({"kind": "regular", "layer": row["layer"], "root_index": row["root_index"],
            "relative": row["relative"], "reference": row["reference"], "bytes": len(data),
            "data_base64": base64.b64encode(data).decode() if row["writable"] else None})
    body = {"schema": SCHEMA + "_packet", "operation_id": operation, "kind": kind,
            "maps_sha256": E.J.digest(config["layers"]), "seeds": deepcopy(seeds), "files": files,
            "authority_issued": False, "evidence_accepted": False}
    raw = E.J._bytes(body)
    require(len(raw) <= config["limits"]["max_archive_bytes"], "packet size cap")
    packed = zlib.compress(raw)
    return {"sha256": hashlib.sha256(packed).hexdigest(), "bytes": len(packed), "base64": base64.b64encode(packed).decode()}


def unpack(envelope, config, side, *, operation, kind, archive_root=None):
    require(set(envelope) == {"sha256", "bytes", "base64"}, "archive envelope")
    maximum = config["limits"]["max_archive_bytes"]
    require(type(envelope["bytes"]) is int and 0 < envelope["bytes"] <= maximum
            and len(envelope["base64"]) <= (maximum * 4 // 3 + 8), "compressed archive cap")
    packed = base64.b64decode(envelope["base64"], validate=True)
    require(len(packed) == envelope["bytes"] and hashlib.sha256(packed).hexdigest() == envelope["sha256"], "archive raw SHA/size")
    decoder = zlib.decompressobj()
    raw = decoder.decompress(packed, maximum + 1)
    require(len(raw) <= maximum and decoder.eof and not decoder.unused_data and not decoder.unconsumed_tail, "bounded single archive stream")
    body = E.E.loads(raw.decode("utf-8"))
    require(set(body) == {"schema", "operation_id", "kind", "maps_sha256", "seeds", "files", "authority_issued", "evidence_accepted"}
            and body["schema"] == SCHEMA + "_packet" and body["operation_id"] == operation and body["kind"] == kind
            and body["maps_sha256"] == E.J.digest(config["layers"])
            and body["authority_issued"] is False and body["evidence_accepted"] is False, "archive identity/layer binding")
    files = body["files"]
    require(isinstance(files, list) and len(files) <= config["limits"]["max_files"], "archive file count")
    staged, seen, total = [], set(), 0
    for row in files:
        require(set(row) == {"kind", "layer", "root_index", "relative", "reference", "bytes", "data_base64"}
                and row["kind"] == "regular" and type(row["bytes"]) is int
                and 0 <= row["bytes"] <= config["limits"]["max_file_bytes"], "regular bounded archive member")
        text = row["relative"]
        require(isinstance(text, str) and bool(text) and "\\" not in text and ":" not in text
                and all(p not in {"", ".", ".."} for p in text.split("/"))
                and not PurePosixPath(text).is_absolute(), "archive canonical relative name")
        index, relative, path, root = locate(row["reference"], config, row["layer"], side, existing=False)
        require(type(row["root_index"]) is int and row["root_index"] == index and relative == text
                and path not in seen and row["layer"] == layer_for(row["reference"]["path"]), "duplicate/foreign archive layer/destination")
        seen.add(path)
        total += row["bytes"]
        require(total <= maximum, "archive total uncompressed file cap")
        if row["data_base64"] is None:
            require(root["writable"] is False and path.exists(), "missing readonly mirrored raw file")
            data = safe(path, exists=True).read_bytes()
        else:
            require(root["writable"] is True and len(row["data_base64"]) <= row["bytes"] * 4 // 3 + 8, "writable archive destination")
            data = base64.b64decode(row["data_base64"], validate=True)
        require(len(data) == row["bytes"] and hashlib.sha256(data).hexdigest() == row["reference"]["sha256"], "raw archive file SHA/size")
        if path.exists():
            require(safe(path, exists=True).read_bytes() == data, "existing archive bytes differ")
        staged.append((path, data, row))
    for seed in body["seeds"]:
        require(isinstance(seed, dict) and set(seed) == {"path", "sha256"}, "required seed raw ref")
        _, _, path, _ = locate(seed, config, layer_for(seed["path"]), side, existing=False)
        require(path in seen, "required metadata/evidence absent from inventory")
    # Validate every member before making any file visible; partial copies are exact-equality replayable.
    if archive_root is not None:
        archive_root = safe(archive_root)
        # Raw archives remain split; the core's registered consumer mirror keeps
        # its original directory tree and is a separate, byte-identical copy.
        for _path, data, row in staged:
            archived = archive_root / row["layer"] / str(row["root_index"]) / row["relative"]
            new_bytes(archived, data)
        save(archive_root / "inventory.json", body)
    for path, data, _row in staged:
        new_bytes(path, data)
    maps = _maps(config, side)
    resolver = E.RootMap(maps)
    for reference in body["seeds"]:
        resolver.file(reference)
    return deepcopy(body["seeds"])


def project_manifest(value, resolver):
    """Validate a derived native-path copy, then retain every original field."""
    require(isinstance(value, dict), "manifest object required")
    raw, native = deepcopy(value), deepcopy(value)
    for key in ("records_path", "readback_path"):
        E.logical(raw[key])
        native[key] = str(resolver.path(raw[key]).resolve(strict=True))
        safe(Path(native[key]), exists=True)
    _MANIFEST_BASE(native)
    require({k: v for k, v in native.items() if k not in {"records_path", "readback_path"}} == {
        k: v for k, v in raw.items() if k not in {"records_path", "readback_path"}}, "manifest projection changed non-path field")
    return raw, {"schema": PROJECTION, "original_digest": E.J.digest(raw), "native_validation_digest": E.J.digest(native),
                 "allowed_changes": ["records_path", "readback_path"]}


class RepairManifestProjectionScope:
    """Explicit main-side contract extension, sharing the core scope RLock.

    Wrap Host2 and Runtime replay together. The POSIX producer uses the original
    core provider/validator; no callback, approval or backend identity changes.
    """

    def __init__(self, registration_ref, root_map, contract_ref, *, allow_fixture=False):
        self.registration_ref, self.root_map, self.contract_ref = deepcopy(registration_ref), deepcopy(root_map), deepcopy(contract_ref)
        self.allow_fixture, self.owner, self.reg = allow_fixture, None, None
        self.projections = {}

    def _validate(self, value):
        require(_MANIFEST_ACTIVE is self and self.owner == threading.get_ident(), "manifest projection scope owner")
        if self.reg is None:
            raise ValueError("manifest projection registration missing")
        raw, proof = project_manifest(value, self.reg["resolver"])
        self.projections[proof["original_digest"]] = proof
        return raw

    def __enter__(self):
        global _MANIFEST_ACTIVE
        E.S._LOCK.acquire()
        try:
            require(_MANIFEST_ACTIVE is None and getattr(E.TRAIN, _MANIFEST_HOOK) is _MANIFEST_BASE, "nested/foreign manifest validator")
            self.reg, _ = registered(self.registration_ref, self.root_map, self.contract_ref, allow_fixture=self.allow_fixture)
            self.owner, _MANIFEST_ACTIVE = threading.get_ident(), self
            setattr(E.TRAIN, _MANIFEST_HOOK, self._validate)
            return self
        except BaseException:
            E.S._LOCK.release()
            raise

    def __exit__(self, *_):
        global _MANIFEST_ACTIVE
        require(_MANIFEST_ACTIVE is self and self.owner == threading.get_ident(), "manifest projection scope exit owner")
        try:
            setattr(E.TRAIN, _MANIFEST_HOOK, _MANIFEST_BASE)
            self.owner, _MANIFEST_ACTIVE = None, None
        finally:
            E.S._LOCK.release()


class BrevRepairTransport:
    """Core Host2's provider facade; queue/CLI success is not evidence acceptance."""

    def __init__(self, registration_ref, root_map, contract_ref, service, *, allow_fixture=False):
        self.registration_ref, self.root_map, self.contract_ref = deepcopy(registration_ref), deepcopy(root_map), deepcopy(contract_ref)
        self.allow_fixture, self.service = allow_fixture, service
        self._registered()

    def _registered(self):
        reg, config = registered(self.registration_ref, self.root_map, self.contract_ref, allow_fixture=self.allow_fixture)
        source = inspect.getsourcefile(type(self.service))
        if source is None:
            raise ValueError("repair service source missing")
        require(type(self.service).__name__ == "BrevRepairMailboxService"
                and self.service.allow_fixture is self.allow_fixture and self.service.registration_ref == self.registration_ref
                and E.C.digest_lf_file(Path(source)) == config["service_source_sha256_lf"],
                "registered repair service identity")
        for name in ("admitter", "execute", "lookup", "production_identity"):
            method = getattr(self.service, name, None)
            require(inspect.ismethod(method) and method.__self__ is self.service
                    and method.__func__ is getattr(type(self.service), name), "service method identity changed")
        self.service.production_identity()
        return reg, config

    def _authority(self, operation, request_ref, *, complete=False):
        reg, config = self._registered()
        request = reg["resolver"].read(request_ref)
        require(request["operation_id"] == operation and request["registration"] == self.registration_ref, "transport exact operation")
        E.verify_model(request, reg)
        auth = E.verify_authorization(request, request["authorization"], reg)
        if complete:
            # Host2 owns done transition; transport cannot issue it.
            ledger = E.C.D1.CorrectionLedger(reg["resolver"].path(auth["ledger_path"]), E.C.RC.correction_registry())
            row = ledger.requests().get(auth["authorization"]["request_id"], {})
            validator = object.__new__(CORE.R.Runtime)
            validator.reg = {"allow_fixture": self.allow_fixture}
            require(row.get("state") in {"running", "done"}, "host live C authority changed")
            validator._authorization(request["intent"], auth, done=row["state"] == "done")
        else:
            require(E.J.file_sha(reg["resolver"].path(auth["ledger_path"])) == request["approval_ledger"]["sha256"], "host consumed prefix changed")
        return reg, config, request

    def execute(self, operation_id, request_ref, authorization_ref, admission_ref):
        reg, _config, request = self._authority(operation_id, request_ref)
        require(request["authorization"] == authorization_ref, "transport authorization ref")
        result = self.service.execute(operation_id, request_ref, authorization_ref, admission_ref)
        self._authority(operation_id, request_ref)
        if result is None:
            raise RepairPending("remote owner still pending; lookup only")
        raw = reg["resolver"].read(result)
        require(raw.get("operation_id") == operation_id and raw.get("request") == request_ref, "transport raw outcome identity")
        return result

    def lookup(self, operation_id, request_ref):
        self._authority(operation_id, request_ref, complete=True)
        result = self.service.lookup(operation_id, request_ref)
        self._authority(operation_id, request_ref, complete=True)
        return result

    def callbacks(self, host):
        """Resume only an interrupted, unexecuted admission in Runtime lookup."""
        self._registered()
        require(type(host) is CORE.BrevRuntimeHost and host.provider is self
                and host.registration_ref == self.registration_ref and host.allow_fixture is self.allow_fixture
                and host.admitter == self.service.admitter, "recovery Host2 binding")
        for name in ("approve", "dispatch", "lookup"):
            method = getattr(host, name)
            require(inspect.ismethod(method) and method.__self__ is host
                    and method.__func__ is getattr(CORE.BrevRuntimeHost, name), "frozen recovery callback identity")

        def lookup(operation_id, intent):
            response = host.lookup(operation_id, intent)
            if response.get("status") == "approved" and intent["action"] == "propose_correction":
                folder = host.folder / operation_id
                request_ref = E.ref(folder / "request.json")
                self._authority(operation_id, request_ref)
                claim = self.service._folder(operation_id) / "admit-intent.json"
                if claim.exists() and not (folder / "dispatch-intent.json").exists():
                    try:
                        return host.dispatch(operation_id, intent, response["authorization"])
                    except RepairPending:
                        return {"status": "pending", "authorization": response["authorization"]}
                    except ValueError:
                        hold = claim.parent / "admission-hold.json"
                        if not hold.exists():
                            raise
                        return {"status": "hold", "reason": "admission_recovery_hold", "raw": E.ref(hold)}
            return response

        return CORE.R.HostCallbacks(host.approve, host.dispatch, lookup)
