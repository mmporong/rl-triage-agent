"""Registered production backend for the resumable investigation session.

Preparation and recorded replay are CPU-only.  The actual child manifest is
validated here; process launch stays fixed to the registered child and owned runner.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from rl_triage import intervention_evidence as EVIDENCE
from rl_triage import investigation_session as SESSION
from rl_triage import mechanism_confirmation as CONFIRMATION
from rl_triage import mechanism_registry as REGISTRY_TOOLS
from rl_triage import nim_transport

BACKEND_ID = "registered_diagnostic_v1"
PROTOCOL_SCHEMA = "registered_diagnostic_protocol_v1"
APPROVAL_SCHEMA = "registered_diagnostic_approval_v1"
SOURCE_SCHEMA = "registered_diagnostic_source_v1"
RECEIPT_FIELDS = {
    "outcome", "measurement", "gpu_s", "execution_wall_s", "exit_code", "receipt_ref",
    "source_sha256", "checkpoint_sha256", "case_id", "probe", "request_id", "effect_key",
    "registry_sha256", "execution_class",
}
PREPARE_FIELDS = {
    "source_path", "case_id", "execution_class", "method", "protocol",
    "artifact_resolver", "api_root", "max_checks", "gpu_budget_per_check_s",
}
ACTUAL_CONFIG_FIELDS = {
    "schema", "operation", "component", "probe", "condition", "seed", "device", "num_envs",
    "case_binding", "source_binding", "reference_binding", "checkpoint_role",
    "source_anchor", "source_closure", "runtime_pins", "isaaclab_root", "watchdog_s",
}
_PROTOCOL_FIELDS = {
    "schema", "backend_id", "source_registration", "registry_sha256", "model",
    "registry_profile", "components", "actors",
}
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_HIDDEN = {
    "fault", "faults", "fault_id", "fault_label", "fault_labels", "hidden_label",
    "hidden_labels", "known_faults", "known_fault_types", "source_type", "classification",
    "original_faults", "remaining_faults", "scoring", "benchmark_score", "benchmark_scoring",
    "answer_key", "healthy", "label",
}
_RECORDED_VERIFIERS = {
    CONFIRMATION.COUNT_PROBE: "evals.recorded_optimizer_probe_v1",
    CONFIRMATION.GRADIENT_PROBE: "evals.recorded_ppo_gradient_probe_v1",
}


class BackendBlocked(RuntimeError):
    """A registered execution path is unavailable; no success receipt was made."""


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_ref(value: object, label: str, *, public: bool = True) -> tuple[Path, dict]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} reference fields differ")
    path = Path(value["path"]).resolve(strict=True)
    if path.is_symlink() or not isinstance(value["sha256"], str) or _SHA.fullmatch(value["sha256"]) is None:
        raise ValueError(f"{label} reference is invalid")
    if _file_sha(path) != value["sha256"]:
        raise ValueError(f"{label} SHA changed")
    parsed = json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda token: (
        _ for _ in ()).throw(ValueError(f"non-finite JSON: {token}")))
    if not isinstance(parsed, dict):
        raise TypeError(f"{label} must be a JSON object")
    if public:
        _reject_hidden(parsed, label)
    return path, parsed


def _reject_hidden(value: object, label: str) -> None:
    if isinstance(value, Mapping):
        lowered = {str(key).casefold() for key in value}
        if lowered & _HIDDEN:
            raise ValueError(f"{label} contains hidden truth")
        for child in value.values():
            _reject_hidden(child, label)
    elif isinstance(value, list):
        for child in value:
            _reject_hidden(child, label)
    _canonical(value)


def _model_params_sha() -> str:
    return _digest(nim_transport.PARAMS)


def _source_registration() -> dict:
    return {
        "collection_manifest_sha256": EVIDENCE.SOURCE_MANIFEST_SHA256,
        "criteria_sha256": EVIDENCE.CRITERIA_SHA256,
        "fixed_protocol_sha256": EVIDENCE.FIXED_PROTOCOL_SHA256,
    }


def _validate_source_registration(value: object) -> dict:
    if not isinstance(value, dict):
        raise TypeError("source registration must be an object")
    base_fields = set(_source_registration())
    if set(value) not in (base_fields, base_fields | {"collection_artifact_id"}):
        raise ValueError("source registration fields differ")
    if (value.get("criteria_sha256") != EVIDENCE.CRITERIA_SHA256
            or value.get("fixed_protocol_sha256") != EVIDENCE.FIXED_PROTOCOL_SHA256
            or _SHA.fullmatch(str(value.get("collection_manifest_sha256"))) is None):
        raise ValueError("source registration criteria, protocol, or manifest differs")
    if "collection_artifact_id" in value and (
            not isinstance(value["collection_artifact_id"], str)
            or not value["collection_artifact_id"]):
        raise ValueError("source collection artifact ID is invalid")
    return deepcopy(value)


def _registry(profile: str) -> dict:
    if profile == "full":
        return CONFIRMATION.mechanism_registry()
    if profile != "runtime_confirmation_v1":
        raise ValueError("mechanism registry profile is not registered")
    return CONFIRMATION.mechanism_registry(profile=profile)


def _verify_anchor(anchor: str) -> None:
    if _COMMIT.fullmatch(anchor) is None:
        raise ValueError("model source anchor must be a commit SHA")
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", anchor, "HEAD"], cwd=Path(__file__).resolve().parents[2],
        capture_output=True, check=False, timeout=10,
    )
    if result.returncode != 0:
        raise ValueError("model source anchor is not an ancestor of the registered source")


def _protocol(reference: object) -> tuple[dict, dict]:
    path, value = _read_ref(reference, "protocol")
    if set(value) != _PROTOCOL_FIELDS or value.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("registered diagnostic protocol fields differ")
    if value["backend_id"] != BACKEND_ID:
        raise ValueError("registered diagnostic backend identity differs")
    _validate_source_registration(value["source_registration"])
    profile = value["registry_profile"]
    if not isinstance(profile, str):
        raise TypeError("mechanism registry profile must be a string")
    registry = _registry(profile)
    registry_sha = REGISTRY_TOOLS.registry_digest(registry)
    if value["registry_sha256"] != registry_sha:
        raise ValueError("mechanism registry changed")
    model = value["model"]
    if (not isinstance(model, dict)
            or set(model) != {"source_anchor", "params_sha256", "api_cap", "api_scope_id"}
            or model["params_sha256"] != _model_params_sha()
            or type(model["api_cap"]) is not int or model["api_cap"] < 1
            or not isinstance(model["api_scope_id"], str) or not model["api_scope_id"]):
        raise ValueError("registered model binding differs")
    _verify_anchor(model["source_anchor"])
    actors = value["actors"]
    if (not isinstance(actors, list) or not actors or len(set(actors)) != len(actors)
            or any(not isinstance(actor, str) or not actor for actor in actors)):
        raise ValueError("registered actor scope differs")
    components = value["components"]
    registered_probes = set(registry["probes"])
    if not isinstance(components, dict) or set(components) != registered_probes:
        raise ValueError("registered component set differs")
    for probe, component in components.items():
        fields = {"component_id", "dispatcher_id", "execution_class", "artifacts"}
        if (not isinstance(component, dict) or set(component) != fields
                or component["execution_class"] not in {"actual", "recorded"}
                or not isinstance(component["component_id"], str) or not component["component_id"]
                or not isinstance(component["dispatcher_id"], str) or not component["dispatcher_id"]
                or not isinstance(component["artifacts"], dict)):
            raise ValueError(f"registered component differs: {probe}")
        for artifact in component["artifacts"].values():
            if (not isinstance(artifact, dict) or set(artifact) != {"artifact_id", "sha256"}
                    or not isinstance(artifact["artifact_id"], str)
                    or _SHA.fullmatch(str(artifact["sha256"])) is None):
                raise ValueError(f"registered artifact differs: {probe}")
    return value, {"path": str(path), "sha256": _file_sha(path)}


def _resolver(value: object, protocol: dict) -> dict:
    if not isinstance(value, dict):
        raise TypeError("artifact resolver must be an object")
    rows = [row for component in protocol["components"].values()
            for row in component["artifacts"].values()]
    source_registration = protocol["source_registration"]
    if "collection_artifact_id" in source_registration:
        rows.append({"artifact_id": source_registration["collection_artifact_id"],
                     "sha256": source_registration["collection_manifest_sha256"]})
    registered = {}
    for row in rows:
        existing = registered.get(row["artifact_id"])
        if existing is not None and existing != row["sha256"]:
            raise ValueError("registered artifact ID has conflicting SHAs")
        registered[row["artifact_id"]] = row["sha256"]
    if set(value) != set(registered):
        raise ValueError("artifact resolver IDs differ from the protocol")
    normalized = {}
    for artifact_id, reference in value.items():
        if not isinstance(reference, Mapping) or set(reference) != {"path", "sha256"}:
            raise ValueError(f"artifact resolver reference differs: {artifact_id}")
        path = Path(reference["path"]).resolve(strict=True)
        if (path.is_symlink() or not isinstance(reference["sha256"], str)
                or _SHA.fullmatch(reference["sha256"]) is None):
            raise ValueError(f"artifact resolver reference is invalid: {artifact_id}")
        if _file_sha(path) != registered[artifact_id]:
            raise ValueError(f"artifact resolver SHA differs: {artifact_id}")
        normalized[artifact_id] = {"path": str(path), "sha256": registered[artifact_id]}
    return normalized


def _model_binding(protocol: dict, protocol_ref: dict, method: str, api_root: object) -> dict | None:
    if method not in SESSION.MODEL_METHODS:
        if api_root is not None:
            raise ValueError("non-model method cannot bind an API root")
        return None
    if not isinstance(api_root, str) or not api_root:
        raise ValueError("model method requires a private shared API root")
    root = Path(api_root)
    if not root.is_absolute() or (root.exists() and root.is_symlink()):
        raise ValueError("private shared API root must be an absolute non-symlink path")
    model = protocol["model"]
    return {"source_anchor": model["source_anchor"], "protocol_sha256": protocol_ref["sha256"],
            "params": deepcopy(nim_transport.PARAMS), "api_cap": model["api_cap"],
            "api_root": str(root.resolve(strict=False))}


def _source_value(case_id: str, execution_class: str, protocol: dict, protocol_ref: dict,
                  resolver: dict) -> dict:
    registration = protocol["source_registration"]
    artifact_id = registration.get("collection_artifact_id")
    if artifact_id is None:
        collection = EVIDENCE.SOURCE_COLLECTION
    else:
        manifest = Path(resolver[artifact_id]["path"])
        if manifest.name != "manifest.json":
            raise ValueError("registered collection artifact must be manifest.json")
        collection = manifest.parent
    evidence_registration = {key: registration[key] for key in _source_registration()}
    observation = EVIDENCE.verify_public_fixed26_pair(
        collection, case_id=case_id, registration=evidence_registration)
    payload = observation._payload
    candidate, reference = payload["candidate"], payload["reference"]
    facts = {}
    for role, row in (("case", candidate), ("reference", reference)):
        for metric, value in row["metrics"].items():
            facts[f"behavior:{role}:{metric}"] = value
    for row in payload["metric_checks"]:
        facts[f"comparison:{row['metric']}"] = row["value"]
    binding = {"schema": "registered_diagnostic_source_binding_v1", "protocol": protocol_ref,
               "source_registration": deepcopy(registration), "case_id": case_id,
               "evidence_verifier_id": payload["verifier_id"], "execution_class": execution_class,
               "registry_profile": protocol["registry_profile"],
               "artifact_resolver": resolver,
               "component_digest": _digest(protocol["components"])}
    return {"schema": SOURCE_SCHEMA, "case_id": case_id,
            "checkpoint_sha256": candidate["checkpoint_sha256"], "facts": facts, "binding": binding}


def prepare_source(payload: dict) -> dict:
    if not isinstance(payload, dict) or set(payload) != PREPARE_FIELDS:
        raise ValueError("prepare payload fields differ")
    if payload["execution_class"] not in {"actual", "recorded"}:
        raise ValueError("execution class must be actual or recorded")
    if payload["method"] not in SESSION.METHODS:
        raise ValueError("investigation method is not registered")
    if (type(payload["max_checks"]) is not int or payload["max_checks"] <= 0
            or isinstance(payload["gpu_budget_per_check_s"], bool)
            or not isinstance(payload["gpu_budget_per_check_s"], (int, float))
            or not math.isfinite(payload["gpu_budget_per_check_s"])
            or payload["gpu_budget_per_check_s"] < 0):
        raise ValueError("investigation limits are invalid")
    protocol, protocol_ref = _protocol(payload["protocol"])
    resolver = _resolver(payload["artifact_resolver"], protocol)
    source = _source_value(payload["case_id"], payload["execution_class"], protocol, protocol_ref, resolver)
    source_path = Path(payload["source_path"]).resolve()
    if source_path.exists():
        if SESSION.read_ref({"path": str(source_path), "sha256": SESSION.file_sha(source_path)}) != source:
            raise ValueError("existing prepared source differs")
    else:
        SESSION.write_new(source_path, source)
    return {"source_path": str(source_path), "registry": _registry(protocol["registry_profile"]),
            "model_binding": _model_binding(protocol, protocol_ref, payload["method"], payload["api_root"]),
            "execution_class": payload["execution_class"]}


def verify_source(path: Path) -> dict:
    source_path = Path(path).resolve(strict=True)
    source = SESSION.read_ref({"path": str(source_path), "sha256": SESSION.file_sha(source_path)})
    if set(source) != {"schema", "case_id", "checkpoint_sha256", "facts", "binding"}:
        raise ValueError("registered source fields differ")
    binding = source["binding"]
    if not isinstance(binding, dict) or binding.get("schema") != "registered_diagnostic_source_binding_v1":
        raise ValueError("registered source binding is missing")
    protocol, protocol_ref = _protocol(binding["protocol"])
    resolver = _resolver(binding["artifact_resolver"], protocol)
    expected = _source_value(source["case_id"], binding["execution_class"], protocol, protocol_ref, resolver)
    if source != expected:
        raise ValueError("registered source no longer matches its recorded sources")
    return {"status": "verified", "verifier_id": BACKEND_ID,
            "source_sha256": SESSION.file_sha(source_path),
            "checkpoint_sha256": source["checkpoint_sha256"],
            "execution_class": binding["execution_class"], "case_id": source["case_id"],
            "binding_sha256": _digest(binding)}


def authorize(request: dict, actor: str, authorization: dict) -> dict:
    if (not isinstance(authorization, dict)
            or set(authorization) != {"protocol", "approval", "source"}):
        return {"status": "denied", "reason": "authorization_fields"}
    try:
        protocol, protocol_ref = _protocol(authorization["protocol"])
        source_path, source = _read_ref(authorization["source"], "authorization source")
        _source_verification = verify_source(source_path)
        approval_path, approval = _read_ref(authorization["approval"], "approval")
        expected_fields = {"schema", "approval_id", "protocol_sha256", "source_sha256", "case_id",
                           "checkpoint_sha256", "probe", "request_id", "effect_key", "registry_sha256",
                           "actor", "approved"}
        if set(approval) != expected_fields or approval.get("schema") != APPROVAL_SCHEMA:
            raise ValueError("approval fields differ")
        expected = {"protocol_sha256": protocol_ref["sha256"],
                    "source_sha256": SESSION.file_sha(source_path), "case_id": source["case_id"],
                    "checkpoint_sha256": source["checkpoint_sha256"], "probe": request["probe"],
                    "request_id": request["request_id"], "effect_key": request["effect_key"],
                    "registry_sha256": request["registry_digest"], "actor": actor, "approved": True}
        if any(approval.get(key) != value for key, value in expected.items()):
            raise ValueError("approval scope differs from the pending request")
        if actor not in protocol["actors"] or protocol["registry_sha256"] != request["registry_digest"]:
            raise ValueError("actor or registry is outside the protocol scope")
        return {"status": "authorized", "authority_id": approval["approval_id"], "actor": actor,
                "request_id": request["request_id"], "effect_key": request["effect_key"],
                "protocol_ref": protocol_ref,
                "approval_ref": {"path": str(approval_path), "sha256": _file_sha(approval_path)},
                "source_ref": {"path": str(source_path), "sha256": SESSION.file_sha(source_path)}}
    except (ValueError, OSError, KeyError, TypeError):
        return {"status": "denied", "reason": "authorization_binding"}


def _approval_proof(intent: dict) -> dict:
    approval_path = Path(intent["host_receipt_path"]).parents[1] / "approvals" / f"{intent['request']['request_id']}.json"
    value = json.loads(approval_path.read_text(encoding="utf-8"))
    proof = value.get("proof") if isinstance(value, dict) else None
    if not isinstance(proof, dict) or proof.get("status") != "authorized":
        raise BackendBlocked("immutable approval proof is unavailable")
    return proof


def _recorded_receipt(intent: dict, component: dict, protocol: dict) -> tuple[str, dict]:
    probe = intent["request"]["probe"]
    expected_module = _RECORDED_VERIFIERS.get(probe)
    if component["dispatcher_id"] != expected_module:
        raise BackendBlocked(f"recorded verifier is not registered for {probe}")
    artifacts = component["artifacts"]
    if set(artifacts) != {"plan", "receipt"}:
        raise BackendBlocked("recorded component requires plan and receipt artifacts")
    resolver = intent["source"]["binding"]["artifact_resolver"]
    refs = {name: resolver[row["artifact_id"]] for name, row in artifacts.items()}
    for name, row in artifacts.items():
        if refs[name]["sha256"] != row["sha256"]:
            raise ValueError("recorded artifact SHA differs from protocol")
    if probe == CONFIRMATION.COUNT_PROBE:
        from evals import recorded_optimizer_probe_v1 as verifier
    elif probe == CONFIRMATION.GRADIENT_PROBE:
        from evals import recorded_ppo_gradient_probe_v1 as verifier
    else:
        raise BackendBlocked("actuator recorded verifier is not registered")
    checked = verifier.checked_receipt(Path(refs["plan"]["path"]), Path(refs["receipt"]["path"]))
    if probe == CONFIRMATION.COUNT_PROBE:
        outcome = checked.get("outcome")
        public = {key: checked["measurement"].get(key) for key in (
            "updates_observed", "expected_optimizer_steps", "optimizer_steps_min",
            "optimizer_steps_max", "expected_minibatches", "minibatches_min", "minibatches_max")}
    else:
        outcome = checked.get("outcome")
        public = {"count_status": checked.get("count_status"), "records": checked.get("records")}
    if outcome not in {"normal", "abnormal", "unknown"}:
        raise ValueError("recorded verifier did not return a registry outcome")
    source_sha = intent["source"]["checkpoint_sha256"]
    if source_sha not in json.dumps(checked, sort_keys=True):
        raise ValueError("recorded artifact is not bound to this checkpoint")
    measurement = {"verifier_id": expected_module, "recorded": True,
                   "plan_sha256": refs["plan"]["sha256"], "receipt_sha256": refs["receipt"]["sha256"],
                   "values": public, "new_execution": False}
    _canonical(measurement)
    return outcome, measurement


def _artifact(component: dict, source: dict, slot: str) -> dict:
    registered = component["artifacts"].get(slot)
    if not isinstance(registered, dict):
        raise BackendBlocked(f"actual component has no registered {slot} artifact")
    reference = source["binding"]["artifact_resolver"].get(registered["artifact_id"])
    if not isinstance(reference, dict) or reference.get("sha256") != registered["sha256"]:
        raise BackendBlocked(f"actual {slot} artifact is not resolved")
    path = Path(reference["path"]).resolve(strict=True)
    if path.is_symlink() or _file_sha(path) != reference["sha256"]:
        raise BackendBlocked(f"actual {slot} artifact changed")
    return {"path": str(path), "sha256": reference["sha256"]}


def _actual_config(intent: dict, component: dict) -> tuple[dict, dict, dict]:
    if component["dispatcher_id"] != "bench.diagnostic_observer_child_v1":
        raise BackendBlocked("actual dispatcher is not diagnostic_observer_child_v1")
    if set(component["artifacts"]) != {"execution", "child_protocol", "checkpoint"}:
        raise BackendBlocked("actual component artifacts are incomplete")
    execution_ref = _artifact(component, intent["source"], "execution")
    _execution_path, config = _read_ref(execution_ref, "actual execution config", public=False)
    if (set(config) != ACTUAL_CONFIG_FIELDS
            or config.get("schema") != "registered_actual_dispatch_v1"):
        raise BackendBlocked("actual execution config fields differ")
    request_probe = intent["request"]["probe"]
    probes = config["probe"] if isinstance(config["probe"], list) else [config["probe"]]
    if request_probe not in probes:
        raise BackendBlocked("selected probe is outside the registered child bundle")
    if (not isinstance(config["case_binding"], Mapping)
            or config["case_binding"].get("case_id") != intent["source"]["case_id"]
            or not isinstance(config["source_binding"], Mapping)):
        raise BackendBlocked("actual config source/case binding differs")
    child_protocol = _artifact(component, intent["source"], "child_protocol")
    checkpoint = _artifact(component, intent["source"], "checkpoint")
    if checkpoint["sha256"] != intent["source"]["checkpoint_sha256"]:
        raise BackendBlocked("actual checkpoint differs from the verified source")
    _canonical(config)
    return config, child_protocol, checkpoint


def _lease_context() -> dict:
    from evals import lineage_source_batch_v1 as lease
    return lease.lease_context()


def _run_owned(folder: Path, args: list[str], cwd: Path, watchdog_s: float) -> dict:
    from bench import run_runtime_readback_smoke as owned
    return owned.run_owned(folder, args, cwd, watchdog_s)


def _write_stage_ref(path: Path, value: dict) -> dict:
    if path.exists():
        existing = SESSION.read_ref(SESSION.reference(path))
        if existing != value:
            raise BackendBlocked(f"actual stage artifact changed: {path.name}")
    else:
        SESSION.write_new(path, value)
    return SESSION.reference(path)


def _approval_lineage(intent: dict, stage: Path, proof: dict) -> dict:
    request = intent["request"]
    shared = {"request_id": request["request_id"], "effect_key": request["effect_key"]}
    return {
        "request": _write_stage_ref(stage / "request.json", {**shared, "request": request}),
        "approval": _write_stage_ref(stage / "approval.json", {**shared, "proof": proof}),
        "consumption": _write_stage_ref(
            stage / "consumption.json", {**shared, "consumption": intent["consumption"]}),
    }


def _admission_snapshot(stage: Path, live: dict) -> dict:
    required = {"owner", "leaseId", "state", "childPid", "wrapperPid", "completedUtc"}
    if (not required <= set(live) or live["state"] != "running"
            or live["completedUtc"] is not None or type(live["childPid"]) is not int):
        raise BackendBlocked("live GPU lease admission is invalid")
    snapshot = {"schema": "registered_gpu_admission_snapshot_v1",
                **{key: live[key] for key in sorted(required)}}
    ref = _write_stage_ref(stage / "gpu-admission.json", snapshot)
    return {"ref": ref, "lease_id": live["leaseId"], "owner_pid": live["childPid"]}


def _actual_observation(output: Path, manifest_sha: str, selected_probe: str) -> tuple[str, dict]:
    failure = output / "failure.json"
    if failure.exists():
        raise BackendBlocked("diagnostic child recorded a failure")
    if selected_probe == CONFIRMATION.ACTUATOR_PROBE:
        observation = SESSION.read_ref(SESSION.reference(output / "observation.json"))
        close = SESSION.read_ref(SESSION.reference(output / "close_ready.json"))
        if (observation.get("status") != "complete"
                or observation.get("manifest_sha256") != manifest_sha
                or close.get("manifest_sha256") != manifest_sha
                or close.get("environment_closed") is not True
                or close.get("app_close_pending") is not True):
            raise BackendBlocked("actuator child close/evidence binding differs")
        raw = observation.get("observation", {})
        outcome = ("normal" if raw.get("status") == "valid"
                   and raw.get("curve_matches_expected") is True
                   else "abnormal" if raw.get("status") == "valid"
                   and raw.get("curve_matches_expected") is False else "unknown")
        measurement = {"component": "actuator_mapping", "observation": raw}
        measurement["evidence_refs"] = {
            "observation": SESSION.reference(output / "observation.json"),
            "close_ready": SESSION.reference(output / "close_ready.json"),
        }
    else:
        from bench import isaac_training_guard_v1 as guard
        guard.validate_completion(output, manifest_sha)
        evidence = SESSION.read_ref(SESSION.reference(output / guard.EVIDENCE))
        inspection = evidence.get("inspection")
        if (evidence.get("schema") != "diagnostic_optimizer_observation_v1"
                or not isinstance(inspection, dict)):
            raise BackendBlocked("optimizer child evidence differs")
        primitives = inspection.get("primitives", {})
        if selected_probe == CONFIRMATION.COUNT_PROBE:
            selected = {name: primitives.get(name) for name in (
                "optimizer_step_count", "minibatch_count")}
            statuses = [row.get("status") for row in selected.values() if isinstance(row, dict)]
            outcome = ("abnormal" if "abnormal" in statuses else
                       "normal" if len(statuses) == 2 and all(row == "normal" for row in statuses)
                       else "unknown")
        elif selected_probe == CONFIRMATION.GRADIENT_PROBE:
            selected = primitives.get("ppo_surrogate_gradient", {})
            outcome = selected.get("status", "unknown") if isinstance(selected, dict) else "unknown"
        else:
            raise BackendBlocked("optimizer bundle cannot derive the selected receipt")
        if outcome not in {"normal", "abnormal", "unknown"}:
            raise BackendBlocked("optimizer inspector returned an unsupported outcome")
        measurement = {"component": "optimizer_update_bundle", "selected": selected,
                       "records": inspection.get("records"),
                       "derived_one_selected_probe_from_fresh_bundle": True,
                       "evidence_refs": {
                           "evidence": SESSION.reference(output / guard.EVIDENCE),
                           "records": SESSION.reference(output / guard.RECORDS),
                           "close_ready": SESSION.reference(output / guard.CLOSE_READY),
                       }}
    return outcome, measurement


def _execute_actual(intent: dict, component: dict, proof: dict) -> tuple[str, dict, float, int]:
    config, child_protocol, checkpoint = _actual_config(intent, component)
    from bench import diagnostic_observer_child_v1 as child

    registered_protocols = {
        child.REGISTERED_PROTOCOL.resolve(),
        *(path.resolve() for path in child.REGISTERED_PROTOCOLS),
    }
    if Path(child_protocol["path"]).resolve() not in registered_protocols:
        raise BackendBlocked("actual child protocol is not the fixed registered path")
    live = _lease_context()
    stage = (Path(intent["host_receipt_path"]).parents[1] / "actual_stages"
             / intent["request"]["request_id"])
    if stage.exists():
        raise BackendBlocked("actual request stage already exists; re-execution is forbidden")
    stage.mkdir(parents=True, exist_ok=False)
    approval = _approval_lineage(intent, stage, proof)
    outer_lease = _admission_snapshot(stage, live)
    output = stage / "output"
    manifest = child.build_manifest(
        protocol_path=Path(child_protocol["path"]), operation=config["operation"],
        request_id=intent["request"]["request_id"], effect_key=intent["request"]["effect_key"],
        probe=config["probe"], case_binding=config["case_binding"],
        source_binding={**config["source_binding"],
                        "source_sha256": intent["preregistration"]["source_sha256"]},
        checkpoint_binding={**checkpoint, "role": config["checkpoint_role"]},
        reference_binding=config["reference_binding"], source_anchor=config["source_anchor"],
        source_closure=config["source_closure"], runtime_pins=config["runtime_pins"],
        seed=config["seed"], device=config["device"], num_envs=config["num_envs"],
        condition=config["condition"], component=config["component"], outer_lease=outer_lease,
        consumed_approval=approval, output_dir=output,
    )
    manifest_path = stage / "manifest.json"
    SESSION.write_new(manifest_path, manifest)
    isaaclab = Path(config["isaaclab_root"]).resolve(strict=True)
    child.verify_manifest(
        manifest_path, repo_root=Path(__file__).resolve().parents[2], isaaclab_root=isaaclab,
        expected_protocol_path=Path(child_protocol["path"]),
    )
    owned = stage / "owned"
    owned.mkdir()
    command = [str(isaaclab / "_isaac_sim" / "python.bat"),
               str(Path(child.__file__).resolve()), "--manifest", str(manifest_path)]
    execution = _run_owned(owned, command, isaaclab, float(config["watchdog_s"]))
    current = _lease_context()
    if current.get("leaseId") != live.get("leaseId"):
        raise BackendBlocked("GPU lease changed during diagnostic child execution")
    if (execution.get("exit_code") != 0 or execution.get("natural_exit") is not True
            or execution.get("cleanup_verified") is not True
            or execution.get("job_assigned_before_release") is not True
            or execution.get("timed_out") is not False or execution.get("error_type") is not None):
        raise BackendBlocked("owned diagnostic child exit/cleanup was not verified")
    execution_ref = _write_stage_ref(stage / "execution.json", execution)
    outcome, measurement = _actual_observation(
        output, SESSION.file_sha(manifest_path), intent["request"]["probe"])
    wall = execution.get("wall_s")
    if isinstance(wall, bool) or not isinstance(wall, (int, float)) or not math.isfinite(wall):
        raise BackendBlocked("owned diagnostic child wall time is invalid")
    measurement.update({"manifest_ref": SESSION.reference(manifest_path),
                        "execution_ref": execution_ref,
                        "gpu_s_scope": "lease_held_owned_child_wall_not_kernel_busy_time"})
    return outcome, measurement, float(wall), int(execution["exit_code"])


def _verify_actual_stage(intent: dict, component: dict, artifact: dict) -> None:
    measurement = artifact.get("measurement")
    if not isinstance(measurement, dict):
        raise TypeError("actual receipt measurement is missing")
    request = intent["request"]
    stage = (Path(intent["host_receipt_path"]).parents[1] / "actual_stages"
             / request["request_id"])
    expected_paths = {"manifest_ref": stage / "manifest.json",
                      "execution_ref": stage / "execution.json"}
    documents = {}
    for key, path in expected_paths.items():
        reference = measurement.get(key)
        if (not isinstance(reference, dict) or Path(reference.get("path", "")).resolve() != path.resolve()
                or reference.get("sha256") != SESSION.file_sha(path)):
            raise ValueError(f"actual {key} binding differs")
        documents[key] = SESSION.read_ref(reference)
    execution = documents["execution_ref"]
    if (execution.get("exit_code") != 0 or execution.get("natural_exit") is not True
            or execution.get("cleanup_verified") is not True
            or execution.get("job_assigned_before_release") is not True
            or execution.get("timed_out") is not False or execution.get("error_type") is not None
            or execution.get("wall_s") != artifact["execution_wall_s"]
            or artifact["gpu_s"] != artifact["execution_wall_s"]):
        raise ValueError("stored actual owned execution differs")
    manifest = documents["manifest_ref"]
    config, child_protocol, checkpoint = _actual_config(intent, component)
    from bench import diagnostic_observer_child_v1 as child

    approval_paths = {name: stage / f"{name}.json"
                      for name in ("request", "approval", "consumption")}
    for name, path in approval_paths.items():
        reference = manifest.get("consumed_approval", {}).get(name)
        if (not isinstance(reference, dict)
                or Path(reference.get("path", "")).resolve() != path.resolve()
                or reference.get("sha256") != SESSION.file_sha(path)):
            raise ValueError(f"stored actual {name} lineage differs")
    child._verify_approval(
        manifest["consumed_approval"], request["request_id"], request["effect_key"])
    admission_path = stage / "gpu-admission.json"
    lease_ref = manifest.get("outer_lease", {}).get("ref")
    if (not isinstance(lease_ref, dict)
            or Path(lease_ref.get("path", "")).resolve() != admission_path.resolve()
            or lease_ref.get("sha256") != SESSION.file_sha(admission_path)):
        raise ValueError("stored actual admission lineage differs")
    child._verify_lease(manifest["outer_lease"])
    expected_manifest = child.build_manifest(
        protocol_path=Path(child_protocol["path"]), operation=config["operation"],
        request_id=request["request_id"], effect_key=request["effect_key"], probe=config["probe"],
        case_binding=config["case_binding"],
        source_binding={**config["source_binding"],
                        "source_sha256": intent["preregistration"]["source_sha256"]},
        checkpoint_binding={**checkpoint, "role": config["checkpoint_role"]},
        reference_binding=config["reference_binding"], source_anchor=config["source_anchor"],
        source_closure=config["source_closure"], runtime_pins=config["runtime_pins"],
        seed=config["seed"], device=config["device"], num_envs=config["num_envs"],
        condition=config["condition"], component=config["component"],
        outer_lease=manifest["outer_lease"], consumed_approval=manifest["consumed_approval"],
        output_dir=stage / "output",
    )
    if manifest != expected_manifest:
        raise ValueError("stored actual child manifest differs")
    outcome, observed = _actual_observation(
        stage / "output", measurement["manifest_ref"]["sha256"], request["probe"])
    expected_measurement = {**observed, "manifest_ref": measurement["manifest_ref"],
                            "execution_ref": measurement["execution_ref"],
                            "gpu_s_scope": "lease_held_owned_child_wall_not_kernel_busy_time"}
    if artifact.get("outcome") != outcome or measurement != expected_measurement:
        raise ValueError("actual receipt no longer matches its stage evidence")


def execute(intent: dict) -> dict:
    proof = _approval_proof(intent)
    protocol, _protocol_ref = _protocol(proof["protocol_ref"])
    probe = intent["request"]["probe"]
    component = protocol["components"][probe]
    if component["execution_class"] != intent["execution_class"]:
        raise BackendBlocked("component execution class differs from the session")
    if intent["execution_class"] == "actual":
        outcome, measurement, wall_s, exit_code = _execute_actual(intent, component, proof)
        gpu_s, execution_class = wall_s, "actual"
    else:
        outcome, measurement = _recorded_receipt(intent, component, protocol)
        gpu_s, wall_s, exit_code, execution_class = 0.0, 0.0, 0, "recorded"
    artifact = {"outcome": outcome, "measurement": measurement, "gpu_s": gpu_s,
                "execution_wall_s": wall_s, "exit_code": exit_code,
                "source_sha256": intent["preregistration"]["source_sha256"],
                "checkpoint_sha256": intent["source"]["checkpoint_sha256"],
                "case_id": intent["source"]["case_id"], "probe": probe,
                "request_id": intent["request"]["request_id"],
                "effect_key": intent["request"]["effect_key"],
                "registry_sha256": intent["registry_sha256"],
                "execution_class": execution_class}
    SESSION.write_new(Path(intent["host_receipt_path"]), artifact)
    return deepcopy(artifact)


def verify_receipt(raw: dict, intent: dict) -> dict:
    receipt_path = Path(intent["host_receipt_path"]).resolve(strict=True)
    artifact = SESSION.read_ref({"path": str(receipt_path), "sha256": SESSION.file_sha(receipt_path)})
    if raw != artifact or set(artifact) != RECEIPT_FIELDS - {"receipt_ref"}:
        raise ValueError("host receipt body differs")
    bindings = {"source_sha256": intent["preregistration"]["source_sha256"],
                "checkpoint_sha256": intent["source"]["checkpoint_sha256"],
                "case_id": intent["source"]["case_id"], "probe": intent["request"]["probe"],
                "request_id": intent["request"]["request_id"],
                "effect_key": intent["request"]["effect_key"],
                "registry_sha256": intent["registry_sha256"],
                "execution_class": intent["execution_class"]}
    if any(artifact.get(key) != value for key, value in bindings.items()):
        raise ValueError("host receipt intent binding differs")
    protocol, _ = _protocol(_approval_proof(intent)["protocol_ref"])
    component = protocol["components"][artifact["probe"]]
    if artifact["execution_class"] == "recorded":
        outcome, measurement = _recorded_receipt(intent, protocol["components"][artifact["probe"]], protocol)
        if artifact["outcome"] != outcome or artifact["measurement"] != measurement:
            raise ValueError("recorded receipt no longer matches its original verifier")
    else:
        _verify_actual_stage(intent, component, artifact)
    for name in ("gpu_s", "execution_wall_s"):
        if (isinstance(artifact[name], bool) or not isinstance(artifact[name], (int, float))
                or not math.isfinite(artifact[name]) or artifact[name] < 0):
            raise ValueError(f"{name} is invalid")
    return {**artifact, "receipt_ref": SESSION.reference(receipt_path)}


def observation_gate(context: dict) -> dict:
    source = context.get("source") if isinstance(context, dict) else None
    if not isinstance(source, dict) or source.get("schema") != SOURCE_SCHEMA:
        raise ValueError("observation gate requires a registered source")
    binding = source["binding"]
    observation = EVIDENCE.verify_public_fixed26_pair(
        EVIDENCE.SOURCE_COLLECTION, case_id=binding["case_id"])
    decision = "host_no_intervention" if observation._payload["behavior_outcome"] == "normal" else "allow"
    return {"decision": decision, "gate_id": "recorded_fixed26_behavior_gate_v1",
            "reason": "verified_normal_behavior_bundle" if decision == "host_no_intervention"
            else "registered_investigation_may_continue"}


def build_backend() -> dict[str, Any]:
    """Return the exact backend contract consumed by ``rl_triage.investigate``."""
    return {"backend_id": BACKEND_ID, "prepare_source": prepare_source,
            "verify_source": verify_source, "authorize": authorize, "execute": execute,
            "verify_receipt": verify_receipt, "complete": nim_transport.complete,
            "observation_gate": observation_gate}
