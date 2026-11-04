"""Registered host backend for one-shot correction training and fixed evaluation.

The module does not choose a correction.  It executes only an immutable protocol whose
diagnostic, correction, seed, budget, runner, and source bindings were approved earlier.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import re
import subprocess
import sys
import time
from collections.abc import Mapping
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from typing import Any

from rl_triage import grounded_recovery, recovery
from rl_triage import investigation_session as session_io
from rl_triage.correction_registry_v2 import REGISTRY, registry_digest

BACKEND_ID = "registered_recovery_v1"
SCHEMA = "registered_recovery_protocol_v1"
SCHEMA_V2 = "registered_recovery_protocol_v2"
SCHEMA_V3 = "registered_recovery_protocol_v3"
DESKTOP_PROFILE = "registered_desktop_shared_v2"
WAIT_PROFILE = "registered_desktop_shared_wait_v1"
SHARED_PROFILE = "registered_desktop_shared_baseline_v1"
RETRY_PROFILE = "registered_desktop_shared_retry_v1"
RECEIPT_SCHEMA = "registered_recovery_receipt_v1"
_HIDDEN = {"scoring", "original_faults", "remaining_faults", "hidden_label", "hidden_labels",
           "answer_key", "correct_any_family"}
_ROOT = Path(__file__).resolve().parents[1]
for _directory in (_ROOT / "bench", _ROOT / "evals", _ROOT / "src"):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

_RUNNERS = {
    "r1_epochs_v2": "r1_repair_execution_v2",
    "r1_epochs_preserving_environment_v1": "registered_epoch_recovery_execution_v1",
    "ppo_ratio_v1": "ppo_ratio_repair_execution",
    "actuator_resume_v1": "registered_recovery_training_v1",
}
_PRESERVING_RUNNER = "r1_epochs_preserving_environment_v1"
_EPOCH_RUNNERS = {"r1_epochs_v2", _PRESERVING_RUNNER}
_MECHANISM_RUNNER = {
    "optimizer_update_count": "r1_epochs_v2",
    "optimizer_surrogate": "ppo_ratio_v1",
    "actuator": "actuator_resume_v1",
}
_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_OPTIMIZER_BINDING_FIELDS = {
    "task", "num_envs", "max_iterations", "configured_learning_epochs", "configured_mini_batches",
    "runtime_updates_to_observe", "source_commit_sha", "protocol_path", "protocol_sha256_lf",
    "source_sha256_lf", "runtime_source_sha256_lf", "source_checkpoint_sha256",
}
_FIXED_BINDING_FIELDS = {"task", "runtime_source_sha256_lf", "action_scale", "protocol_sha256_lf",
                         "reference_report", "bands"}
_DESKTOP_SOURCES = {
    "evals/registered_desktop_gpu_admission_v1.py",
    "tests/test_registered_desktop_gpu_admission_v1.py",
    "bench/protocols/registered_desktop_gpu_admission_v1.json",
}
_PRIOR_SOURCES = {"evals/registered_prior_training_v1.py", "tests/test_registered_prior_training_v1.py"}
_WAIT_SOURCES = {"evals/registered_desktop_gpu_wait_v1.py", "tests/test_registered_desktop_gpu_wait_v1.py",
                 "bench/protocols/registered_desktop_gpu_wait_v1.json"}
_SHARED_SOURCES = {"evals/registered_desktop_gpu_shared_v1.py", "tests/test_registered_desktop_gpu_shared_v1.py",
                   "bench/protocols/registered_desktop_gpu_shared_v1.json"}
_RETRY_SOURCES = _SHARED_SOURCES | {
    "evals/registered_desktop_gpu_retry_v1.py", "tests/test_registered_desktop_gpu_retry_v1.py",
    "bench/protocols/registered_desktop_gpu_retry_v1.json",
}
_CONTINUATION_SOURCES = {"evals/registered_recovery_continuation_v3.py",
                         "tests/test_registered_recovery_continuation_v3.py"}


def _require(ok: object, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def _runner_module(runner_id: str) -> Any:
    return importlib.import_module(_RUNNERS[runner_id])


def _runtime_confirmation(runner_id: str, evidence: dict, checkpoint_ref: dict) -> dict:
    module = importlib.import_module("evals.registered_recovery_confirmation_v1")
    return module.verify_corrected_runtime(runner_id, evidence, checkpoint_ref)


def _verify_environment_preservation(runner_id: str, module: Any, evidence: object,
                                     checkpoint_ref: dict, training_binding: dict) -> dict | None:
    if runner_id != _PRESERVING_RUNNER:
        _require(evidence is None, "unregistered environment preservation evidence")
        return None
    _require(isinstance(evidence, dict), "environment preservation evidence missing")
    assert isinstance(evidence, dict)
    _, recorded_binding = _ref(evidence.get("binding"), "environment preservation binding")
    _require(recorded_binding == training_binding.get("environment_preservation"),
             "environment preservation differs from approved source recipe")
    verified = module.verify_preservation(evidence, checkpoint_ref)
    return {"evidence": deepcopy(evidence), "verification": verified}


def _verify_training_manifest(manifest: dict, operation: dict, diagnostic: dict) -> None:
    actuator = operation["runner_id"] == "actuator_resume_v1"
    checkpoint_key = "resume_checkpoint_sha256" if actuator else "source_checkpoint_sha256"
    _require(manifest.get("arm") == ("corrected" if actuator else "CORRECTED")
             and manifest.get("seed") == operation["training_seed"]
             and manifest.get("max_iterations") == operation["training_iterations"]
             and manifest.get(checkpoint_key) == diagnostic["checkpoint_sha256"],
             "training manifest binding mismatch")


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"),
                       parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
    _require(isinstance(value, dict), "JSON object required")
    session_io.canonical(value)
    return value


def _ref(value: object, label: str) -> tuple[Path, dict]:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} ref fields")
    path = Path(value["path"]).resolve(strict=True)
    _require(path.is_file() and not path.is_symlink() and session_io.file_sha(path) == value["sha256"],
             f"{label} ref changed")
    return path, _json(path)


def _no_hidden(value: object) -> None:
    if isinstance(value, dict):
        _require(not (_HIDDEN & {str(key).casefold() for key in value}), "hidden/scoring fields forbidden")
        for child in value.values():
            _no_hidden(child)
    elif isinstance(value, list):
        for child in value:
            _no_hidden(child)


def _verify_source_closure(anchor: object, closure: object) -> None:
    if not isinstance(anchor, str) or _HEX40.fullmatch(anchor) is None:
        raise ValueError("source anchor")
    if not isinstance(closure, dict) or not closure:
        raise ValueError("source closure")
    root = Path(__file__).resolve().parents[1]
    subprocess.run(["git", "-C", str(root), "cat-file", "-e", f"{anchor}^{{commit}}"],
                   check=True, capture_output=True)
    _require(subprocess.run(["git", "-C", str(root), "merge-base", "--is-ancestor", anchor, "HEAD"],
                            check=False, capture_output=True).returncode == 0,
             "source anchor is not an ancestor")
    for relative, expected in closure.items():
        _require(isinstance(relative, str) and relative and not Path(relative).is_absolute()
                 and "\\" not in relative and ".." not in Path(relative).parts
                 and isinstance(expected, str) and _HEX64.fullmatch(expected) is not None,
                 "source closure entry")
        path = (root / relative).resolve()
        _require(root in path.parents and path.is_file(), "source closure path")
        current = hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        blob = subprocess.run(["git", "-C", str(root), "show", f"{anchor}:{relative}"], check=True,
                              capture_output=True).stdout.replace(b"\r\n", b"\n")
        _require(current == expected == hashlib.sha256(blob).hexdigest(), "source closure blob mismatch")


def _extensions(protocol: Mapping[str, Any]) -> str | None:
    """Return the optional prior artifact after checking the registered v2 boundary."""
    operation = protocol["operation"]
    if protocol.get("schema") not in {SCHEMA_V2, SCHEMA_V3}:
        _require(not {"gpu_admission_profile", "prior_training_artifact_id"} & set(operation),
                 "v2 operation extensions require v2 protocol")
        return None
    v3 = protocol.get("schema") == SCHEMA_V3
    profile = operation.get("gpu_admission_profile")
    _require(profile in ({WAIT_PROFILE, SHARED_PROFILE, RETRY_PROFILE} if v3 else {DESKTOP_PROFILE}),
             "unregistered desktop GPU admission profile")
    _require(_DESKTOP_SOURCES <= set(protocol["source_closure"]), "desktop policy source missing")
    if v3:
        required_sources = (_RETRY_SOURCES if profile == RETRY_PROFILE else
                            _SHARED_SOURCES if profile == SHARED_PROFILE else _WAIT_SOURCES)
        _require(required_sources <= set(protocol["source_closure"]), "desktop admission policy source missing")
    prior_id = operation.get("prior_training_artifact_id")
    _require(prior_id is None or isinstance(prior_id, str) and bool(prior_id), "prior training artifact id")
    if prior_id is not None:
        _require(operation["runner_id"] == "actuator_resume_v1", "prior training runner scope")
        _require(_PRIOR_SOURCES <= set(protocol["source_closure"]), "prior training source missing")
        if v3:
            _require(_CONTINUATION_SOURCES <= set(protocol["source_closure"]), "continuation source missing")
        _require(prior_id not in {operation["training_binding_artifact_id"],
                                 operation["fixed_binding_artifact_id"]}, "prior artifact aliases binding")
    return prior_id


def _prior_training(context: Mapping[str, Any]) -> dict | None:
    protocol = context["protocol_body"]
    prior_id = _extensions(protocol)
    if prior_id is None:
        return None
    operation, diagnostic = protocol["operation"], protocol["diagnostic"]
    module_name = ("evals.registered_recovery_continuation_v3" if protocol.get("schema") == SCHEMA_V3
                   else "evals.registered_prior_training_v1")
    module = importlib.import_module(module_name)
    load = (module.load_completed_training if protocol.get("schema") == SCHEMA_V3
            else module.load_completed_actuator_training)
    binding = context["artifacts"][operation["training_binding_artifact_id"]]["body"]
    return load(
        context["artifacts"][prior_id]["ref"],
        expected_source_checkpoint_sha256=diagnostic["checkpoint_sha256"],
        expected_seed=operation["training_seed"], expected_correction_id=operation["correction_id"],
        expected_task=binding["task"],
        expected_diagnostic_manifest_sha256=diagnostic["session_manifest_sha256"],
        repo=_ROOT, isaaclab=Path(context["isaaclab_root"]), expected_binding=binding)


def _reuse_evidence(prior: dict) -> dict:
    return {key: deepcopy(value) for key, value in prior.items() if key != "training_result"}


def _gpu_scope(context: Mapping[str, Any]):
    protocol = context["protocol_body"]
    _extensions(protocol)
    if protocol.get("schema") not in {SCHEMA_V2, SCHEMA_V3}:
        return nullcontext()
    if protocol.get("schema") == SCHEMA_V3:
        name = ("evals.registered_desktop_gpu_retry_v1"
                if protocol["operation"]["gpu_admission_profile"] == RETRY_PROFILE else
                "evals.registered_desktop_gpu_shared_v1"
                if protocol["operation"]["gpu_admission_profile"] == SHARED_PROFILE
                else "evals.registered_desktop_gpu_wait_v1")
    else:
        name = "evals.registered_desktop_gpu_admission_v1"
    module = importlib.import_module(name)
    return module.snapshot_scope()


def prepare(payload: Mapping[str, Any]) -> dict:
    """Validate immutable preregistration and physical resolver without starting a process."""
    _require(isinstance(payload, Mapping), "payload mapping required")
    _no_hidden(payload)
    required = {"protocol", "artifact_resolver", "isaaclab_root", "output_root"}
    _require(set(payload) == required, "prepare payload fields")
    protocol_path, protocol = _ref(payload["protocol"], "protocol")
    _no_hidden(protocol)
    fields = {"schema", "backend_id", "source_anchor", "source_closure", "diagnostic", "operation",
              "registry_sha256", "actors", "limits"}
    _require(set(protocol) == fields and protocol["schema"] in {SCHEMA, SCHEMA_V2, SCHEMA_V3}
             and protocol["backend_id"] == BACKEND_ID, "protocol fields")
    _require(protocol["registry_sha256"] == registry_digest(), "registry binding changed")
    _verify_source_closure(protocol["source_anchor"], protocol["source_closure"])
    diagnostic = protocol["diagnostic"]
    dfields = {"session_manifest_sha256", "source_sha256", "checkpoint_sha256", "case_id",
               "mechanism", "probe", "diagnostic_receipt_sha256"}
    _require(isinstance(diagnostic, dict) and set(diagnostic) == dfields, "diagnostic binding fields")
    operation = protocol["operation"]
    ofields = {"correction_id", "recovery_mode", "runner_id", "training_seed", "training_iterations",
               "gpu_budget_s", "source_checkpoint_consumed", "training_binding_artifact_id",
               "fixed_binding_artifact_id"}
    if protocol["schema"] in {SCHEMA_V2, SCHEMA_V3}:
        ofields |= {"gpu_admission_profile", "prior_training_artifact_id"}
    _require(isinstance(operation, dict) and set(operation) == ofields, "operation fields")
    prior_id = _extensions(protocol)
    mechanism = diagnostic["mechanism"]
    _require(mechanism in _MECHANISM_RUNNER, "mechanism has no executable registered recovery")
    entry = REGISTRY[mechanism]
    _require(operation["correction_id"] == entry["correction_id"]
             and operation["recovery_mode"] in entry["recovery_modes"], "correction registry mismatch")
    allowed_runners = (_EPOCH_RUNNERS if mechanism == "optimizer_update_count"
                       else {_MECHANISM_RUNNER[mechanism]})
    _require(operation["runner_id"] in allowed_runners, "runner mismatch")
    _require(type(operation["training_seed"]) is int and operation["training_seed"] >= 0, "seed scope")
    _require(operation["training_iterations"] == 300, "full recovery training must be preregistered at 300")
    _require(type(operation["gpu_budget_s"]) is int and operation["gpu_budget_s"] >= 0, "GPU budget scope")
    _require(operation["source_checkpoint_consumed"] is (mechanism == "actuator"),
             "restart/resume checkpoint consumption mismatch")
    _require(isinstance(protocol["actors"], list) and protocol["actors"]
             and all(isinstance(actor, str) and actor for actor in protocol["actors"]), "actors")
    limits = protocol["limits"]
    _require(isinstance(limits, dict)
             and set(limits) == {"training_watchdog_s", "fixed_watchdog_s"}
             and all(type(value) in {int, float} and math.isfinite(value) and value > 0
                     for value in limits.values()), "limits")

    resolver_path, resolver = _ref(payload["artifact_resolver"], "resolver")
    _require(resolver.get("schema") == "registered_recovery_resolver_v1"
             and isinstance(resolver.get("artifacts"), dict), "resolver schema")
    artifacts: dict[str, dict] = {}
    artifact_ids = [operation["training_binding_artifact_id"], operation["fixed_binding_artifact_id"]]
    if prior_id is not None:
        artifact_ids.append(prior_id)
    for key in artifact_ids:
        _require(key in resolver["artifacts"], "resolver artifact missing")
        path, body = _ref(resolver["artifacts"][key], key)
        artifacts[key] = {"ref": session_io.reference(path), "body": body}
    isaaclab = Path(payload["isaaclab_root"]).resolve()
    output = Path(payload["output_root"]).resolve()
    _require(isaaclab.is_dir(), "Isaac Lab root missing")
    _require(not output.exists(), "output root must be fresh")
    training_body = artifacts[operation["training_binding_artifact_id"]]["body"]
    _require(training_body.get("max_iterations") == operation["training_iterations"],
             "training binding iteration mismatch")
    fixed_body = artifacts[operation["fixed_binding_artifact_id"]]["body"]
    _require(set(fixed_body) == _FIXED_BINDING_FIELDS,
             "fixed binding fields")
    _ref(fixed_body["reference_report"], "reference report")
    bands = fixed_body["bands"]
    _require(isinstance(bands, dict)
             and set(bands) == {"survival_ratio_min", "fall_frac_max", "err_xy_ratio_max", "err_yaw_ratio_max"}
             and all(type(value) in {int, float} and math.isfinite(value) for value in bands.values()),
             "fixed recovery bands")
    if operation["runner_id"] == "actuator_resume_v1":
        module = _runner_module(operation["runner_id"])
        module.validate_prepared_binding(training_body)
        expected_watchdogs = {"training_watchdog_s": float(module.V1.WATCHDOG["train"]),
                              "fixed_watchdog_s": float(module.V1.WATCHDOG["fixed_eval"])}
    else:
        preserving = operation["runner_id"] == _PRESERVING_RUNNER
        fields = _OPTIMIZER_BINDING_FIELDS | ({"environment_preservation"} if preserving else set())
        _require(set(training_body) == fields, "optimizer training binding fields")
        _require(training_body["max_iterations"] == 300
                 and training_body["source_checkpoint_sha256"] == diagnostic["checkpoint_sha256"],
                 "optimizer training provenance")
        module = _runner_module(operation["runner_id"])
        if preserving:
            environment = module.CHILD.validate_environment_binding(training_body["environment_preservation"])
            provenance = environment["source_provenance"]
            _require(provenance["training_seed"] == operation["training_seed"]
                     and provenance["source_checkpoint_sha256"] == diagnostic["checkpoint_sha256"],
                     "environment preservation seed/checkpoint differs")
            for relative in ("bench/registered_epoch_recovery_training_v1.py",
                             "evals/registered_epoch_recovery_execution_v1.py"):
                _require(training_body["source_sha256_lf"].get(relative)
                         == protocol["source_closure"].get(relative)
                         and relative in protocol["source_closure"], "preserving runner source missing")
        implementation = module.V1 if operation["runner_id"] in _EPOCH_RUNNERS else module
        base_binding = {key: value for key, value in training_body.items() if key != "environment_preservation"}
        manifest = implementation._manifest(base_binding, arm="CORRECTED", seed=operation["training_seed"],
                                             run_name="preflight", mode="train",
                                             records=output / "preflight_records.jsonl",
                                             readback=output / "preflight_readback.json")
        implementation.WRAPPER.verify_repository_binding(manifest)
        implementation.WRAPPER.verify_runtime_binding(manifest, isaaclab)
        expected_watchdogs = {"training_watchdog_s": float(module.V1.WATCHDOG["train"])
                              if hasattr(module, "V1") else float(module.WATCHDOG["train"]),
                              "fixed_watchdog_s": float(module.V1.WATCHDOG["fixed_eval"])
                              if hasattr(module, "V1") else float(module.WATCHDOG["fixed_eval"])}
    module.V1._verify_runtime(isaaclab, fixed_body)
    _require({key: float(value) for key, value in limits.items()} == expected_watchdogs,
             "watchdog binding differs from fixed runner")
    context = {"schema": "registered_recovery_prepared_v1", "protocol": session_io.reference(protocol_path),
            "resolver": session_io.reference(resolver_path), "protocol_body": deepcopy(protocol),
            "artifacts": artifacts, "isaaclab_root": str(isaaclab), "output_root": str(output),
            "processes_started": 0, "api_calls": 0}
    _prior_training(context)
    return context


def authorize(context: Mapping[str, Any], request: Mapping[str, Any], actor: str,
              approval_ref: Mapping[str, Any]) -> dict:
    """Verify a separately written approval bound to this request and protocol."""
    _require(actor in context["protocol_body"]["actors"], "actor not preregistered")
    _, approval = _ref(approval_ref, "approval")
    fields = {"schema", "actor", "request_id", "effect_key", "protocol_sha256", "registry_sha256",
              "source_sha256", "checkpoint_sha256", "mechanism", "correction_id", "recovery_mode",
              "gpu_budget_s"}
    _require(set(approval) == fields and approval["schema"] == "registered_recovery_approval_v1",
             "approval fields")
    protocol = context["protocol_body"]
    diagnostic, operation = protocol["diagnostic"], protocol["operation"]
    expected = {"actor": actor, "request_id": request["request_id"], "effect_key": request["effect_key"],
                "protocol_sha256": context["protocol"]["sha256"], "registry_sha256": registry_digest(),
                "source_sha256": diagnostic["source_sha256"],
                "checkpoint_sha256": diagnostic["checkpoint_sha256"], "mechanism": diagnostic["mechanism"],
                "correction_id": operation["correction_id"], "recovery_mode": operation["recovery_mode"],
                "gpu_budget_s": operation["gpu_budget_s"]}
    _require(all(approval.get(key) == value for key, value in expected.items()), "approval binding mismatch")
    return {"status": "authorized", "actor": actor, "request_id": request["request_id"],
            "effect_key": request["effect_key"], "authority_id": approval_ref["sha256"],
            "approval_ref": dict(approval_ref)}


def _lease(lease_reader=None) -> dict:
    if lease_reader is None:
        module = importlib.import_module("evals.lineage_source_batch_v1")
        retry = importlib.import_module("evals.lease_read_retry_v1")
        value, owner = retry.read_live_lease(module.lease_context), module.OWNER
    else:
        value, owner = lease_reader(), "rl-triage-lineage-sources-v1"
    if (not isinstance(value, dict) or value.get("owner") != owner
            or not isinstance(value.get("leaseId"), str) or not value["leaseId"]):
        raise ValueError("registered recovery requires the live shared GPU lease")
    return value


def admit(context: Mapping[str, Any], request: Mapping[str, Any], *, lease_reader=None) -> dict:
    """Verify the shared live lease before consuming a correction request."""
    protocol, lease = context["protocol_body"], _lease(lease_reader)
    diagnostic = protocol["diagnostic"]
    return {"schema": "registered_recovery_lease_admission_v1",
            "lease_id": lease["leaseId"], "owner": lease["owner"],
            "child_pid": lease.get("childPid"), "wrapper_pid": lease.get("wrapperPid"),
            "request_id": request["request_id"], "effect_key": request["effect_key"],
            "protocol_sha256": context["protocol"]["sha256"],
            "registry_sha256": protocol["registry_sha256"],
            "source_sha256": diagnostic["source_sha256"],
            "checkpoint_sha256": diagnostic["checkpoint_sha256"], "admitted_pid": os.getpid()}


def _verify_admission(intent: Mapping[str, Any], *, require_live: bool, lease_reader=None) -> dict:
    path, body = _ref(intent.get("admission_ref"), "lease admission")
    context, request = intent["context"], intent["request"]
    protocol, diagnostic = context["protocol_body"], context["protocol_body"]["diagnostic"]
    expected = {"schema": "registered_recovery_lease_admission_v1",
                "request_id": request["request_id"], "effect_key": request["effect_key"],
                "protocol_sha256": context["protocol"]["sha256"],
                "registry_sha256": protocol["registry_sha256"],
                "source_sha256": diagnostic["source_sha256"],
                "checkpoint_sha256": diagnostic["checkpoint_sha256"]}
    _require(all(body.get(key) == value for key, value in expected.items())
             and body.get("owner") == "rl-triage-lineage-sources-v1"
             and isinstance(body.get("lease_id"), str), "lease admission binding mismatch")
    if require_live:
        _require(_lease(lease_reader)["leaseId"] == body["lease_id"],
                 "shared GPU lease changed before execution")
    return {"path": str(path), "sha256": intent["admission_ref"]["sha256"]}


def _execution_wall(result: Mapping[str, Any]) -> float:
    execution = result.get("execution", {})
    _require(isinstance(execution, dict), "execution evidence missing")
    value = execution.get("wall_s", execution.get("execution_wall_s"))
    _require(type(value) in {int, float} and math.isfinite(value) and value >= 0, "execution wall")
    return float(value)


def _validate_readback(runner_id: str, module: Any, value: object) -> dict:
    if runner_id in _EPOCH_RUNNERS:
        checked = module.V1._validate_readback(value, "CORRECTED")
    elif runner_id == "ppo_ratio_v1":
        checked = module.WRAPPER.validate_readback(value, "CORRECTED")
    else:
        module.LEGACY.validate_readback(value, "F2", "restore_motor_torque_speed")
        checked = value
    if not isinstance(checked, dict):
        raise TypeError("correction readback must be an object")
    return checked


def _behavior_verdict(candidate: dict, reference: dict, bands: dict) -> tuple[str, dict, dict]:
    from rl_triage import observable_behavior_v1 as behavior

    compared = recovery.compare(grounded_recovery._behavior(candidate),
                                grounded_recovery._behavior(reference))
    result = recovery.verdict(compared, bands)
    qualification = behavior.classify_actual_fixed26(candidate, reference, bands)
    result["qualification"] = qualification
    verdict = {"normal": "recovered", "failed": "not_recovered", "below_reference": "not_recovered",
               "unknown": "unknown"}[qualification["behavior_outcome"]]
    return verdict, compared, result


def execute(intent: Mapping[str, Any], *, runner: Any | None = None, lease_reader=None) -> dict:
    """Run one registered correction once. The caller must consume its ledger before this call."""
    protocol = intent["context"]["protocol_body"]
    _verify_source_closure(protocol["source_anchor"], protocol["source_closure"])
    with _gpu_scope(intent["context"]):
        return _execute_phases(intent, runner=runner, lease_reader=lease_reader)


def _execute_phases(intent: Mapping[str, Any], *, runner: Any | None = None, lease_reader=None) -> dict:
    context, request = intent["context"], intent["request"]
    protocol = context["protocol_body"]
    operation, diagnostic = protocol["operation"], protocol["diagnostic"]
    _require(intent["authorization"]["status"] == "authorized", "authorization missing")
    _verify_source_closure(protocol["source_anchor"], protocol["source_closure"])
    admission_ref = _verify_admission(intent, require_live=True, lease_reader=lease_reader)
    _require(request["confirmed_mechanism"] == diagnostic["mechanism"], "request mechanism mismatch")
    module = runner or _runner_module(operation["runner_id"])
    output = Path(context["output_root"])
    output.mkdir(parents=True, exist_ok=False)
    binding = deepcopy(context["artifacts"][operation["training_binding_artifact_id"]]["body"])
    fixed_binding = deepcopy(context["artifacts"][operation["fixed_binding_artifact_id"]]["body"])
    _require(set(fixed_binding) == _FIXED_BINDING_FIELDS,
             "fixed binding fields")
    started = time.perf_counter()
    prior = _prior_training(context)
    if prior is None:
        training_stage = output / "training"
        training = module.train(training_stage, Path(context["isaaclab_root"]), "CORRECTED",
                                operation["training_seed"], request["request_id"], binding)
    else:
        training_stage = Path(prior["training_stage"]).resolve(strict=True)
        training = prior["training_result"]
    readback = _validate_readback(operation["runner_id"], module, training.get("readback"))
    checkpoint = training.get("checkpoint")
    _require(isinstance(checkpoint, dict) and set(checkpoint) >= {"path", "sha256"}, "training checkpoint")
    checkpoint_path = Path(checkpoint["path"]).resolve(strict=True)
    _require(session_io.file_sha(checkpoint_path) == checkpoint["sha256"], "new checkpoint SHA")
    preservation = _verify_environment_preservation(operation["runner_id"], module,
        training.get("environment_preservation"), session_io.reference(checkpoint_path), binding)
    training_wall = _execution_wall(training)
    _require(training_wall <= operation["gpu_budget_s"], "training exhausted registered GPU budget")
    fixed = module.fixed_evaluate(output / "fixed", Path(context["isaaclab_root"]), training,
                                  request["request_id"], fixed_binding)
    report_paths = fixed.get("report_paths")
    if isinstance(report_paths, dict):
        _require(len(report_paths) == 1, "one fixed report required")
        report_path = Path(next(iter(report_paths.values()))).resolve(strict=True)
    else:
        _require(isinstance(report_paths, list) and len(report_paths) == 1, "one fixed report required")
        report_path = Path(report_paths[0]).resolve(strict=True)
    report = _json(report_path)
    validated = grounded_recovery.validate_fixed_report(report, {
        "name": report["name"], "checkpoint_file": checkpoint_path.name,
        "checkpoint_sha256": checkpoint["sha256"], "action_scale": fixed_binding["action_scale"],
        "protocol_sha256_lf": fixed_binding["protocol_sha256_lf"],
    })
    _, reference = _ref(fixed_binding["reference_report"], "reference report")
    reference = grounded_recovery.validate_fixed_report(reference, {
        "name": reference["name"], "checkpoint_file": reference["checkpoint"]["file"],
        "checkpoint_sha256": reference["checkpoint"]["sha256"],
        "action_scale": fixed_binding["action_scale"],
        "protocol_sha256_lf": fixed_binding["protocol_sha256_lf"],
    })
    verdict, comparison, behavior_verdict = _behavior_verdict(validated, reference, fixed_binding["bands"])
    costs = {"prepare_s": float(intent["costs"]["prepare_s"]),
             "authority_s": float(intent["costs"]["authority_s"]),
             "queue_s": float(intent["costs"].get("queue_s", 0.0)),
             "training_s": training_wall, "fixed_s": _execution_wall(fixed),
             "postcheck_s": max(0.0, time.perf_counter() - started
                                - (training_wall if prior is None else 0.0) - _execution_wall(fixed)),
             "api_calls": 0}
    gpu_s = costs["training_s"] + costs["fixed_s"]
    if gpu_s > operation["gpu_budget_s"]:
        verdict = "unknown"
    evidence = {"manifest": session_io.reference(training_stage / "manifest.json"),
                "readback": session_io.reference(training_stage / "readback.json"
                                                  if operation["runner_id"] != "actuator_resume_v1"
                                                  else training_stage / "correction.json")}
    checkpoint_evidence = training_stage / "checkpoint.json"
    if checkpoint_evidence.is_file():
        evidence["checkpoint"] = session_io.reference(checkpoint_evidence)
    checkpoint_execution_path = training_stage / "checkpoint_execution.json"
    checkpoint_wall = None
    telemetry_wall = None
    if checkpoint_execution_path.is_file():
        checkpoint_execution = _json(checkpoint_execution_path)
        _require(checkpoint_execution.get("exit_code") == 0
                 and checkpoint_execution.get("natural_exit") is True
                 and checkpoint_execution.get("cleanup_verified") is True,
                 "checkpoint inspection execution incomplete")
        checkpoint_wall_value = checkpoint_execution.get("wall_s")
        if (isinstance(checkpoint_wall_value, bool)
                or not isinstance(checkpoint_wall_value, (int, float))
                or not math.isfinite(checkpoint_wall_value) or checkpoint_wall_value < 0):
            raise ValueError("checkpoint inspection wall missing")
        checkpoint_wall = float(checkpoint_wall_value)
        evidence["checkpoint_execution"] = session_io.reference(checkpoint_execution_path)
    if operation["runner_id"] != "actuator_resume_v1":
        _require("checkpoint" in evidence and "checkpoint_execution" in evidence,
                 "optimizer checkpoint evidence/execution missing")
        extraction_path = training_stage / "extract_execution.json"
        extraction = _json(extraction_path)
        _require(extraction.get("exit_code") == 0 and extraction.get("natural_exit") is True
                 and extraction.get("cleanup_verified") is True,
                 "telemetry extraction execution incomplete")
        telemetry_wall = _execution_wall({"execution": extraction})
        evidence["telemetry_execution"] = session_io.reference(extraction_path)
        evidence["telemetry"] = session_io.reference(training_stage / "telemetry.json")
        evidence["records"] = session_io.reference(training_stage / "records.jsonl")
    actuator_receipt = training_stage / "receipt.json"
    if actuator_receipt.is_file():
        evidence["training_receipt"] = session_io.reference(actuator_receipt)
    runtime_confirmation = _runtime_confirmation(operation["runner_id"], evidence,
                                                 session_io.reference(checkpoint_path))
    receipt = {"schema": RECEIPT_SCHEMA, "request_id": request["request_id"],
               "effect_key": request["effect_key"], "mechanism": diagnostic["mechanism"],
               "correction_id": operation["correction_id"], "recovery_mode": operation["recovery_mode"],
               "source_sha256": diagnostic["source_sha256"], "checkpoint_sha256": diagnostic["checkpoint_sha256"],
               "source_checkpoint_consumed": operation["source_checkpoint_consumed"],
               "new_checkpoint": session_io.reference(checkpoint_path), "correction_readback": readback,
               "training_evidence": evidence,
               "runtime_confirmation": runtime_confirmation,
               "environment_preservation": preservation,
               "fixed_report": session_io.reference(report_path), "metrics": validated["metrics"],
               "reference_report": fixed_binding["reference_report"], "comparison": comparison,
               "metric_checks": behavior_verdict["checks"],
               "behavior_qualification": behavior_verdict["qualification"],
               "training_stage": session_io.reference(training_stage / "execution.json"),
               "fixed_stage": session_io.reference(output / "fixed" / "execution.json"),
               "lease_admission": admission_ref,
               "owned_breakdown": {"simulator_training_wall_s": costs["training_s"],
                                   "simulator_fixed_wall_s": costs["fixed_s"],
                                   "checkpoint_inspection_owned_wall_s": checkpoint_wall,
                                   "checkpoint_cpu_process_s": None,
                                   "checkpoint_included_in_postcheck_s": checkpoint_wall is not None,
                                   "telemetry_extraction_owned_wall_s": telemetry_wall,
                                   "telemetry_included_in_postcheck_s": telemetry_wall is not None},
               "costs": costs, "gpu_s": gpu_s, "exit_code": 0, "verdict": verdict}
    if prior is not None:
        receipt["training_reuse"] = _reuse_evidence(prior)
    receipt_path = output / "receipt.json"
    session_io.write_new(receipt_path, receipt)
    return {**receipt, "receipt_ref": session_io.reference(receipt_path)}


def verify_receipt(raw: Mapping[str, Any], intent: Mapping[str, Any]) -> dict:
    _require(isinstance(raw, Mapping) and isinstance(raw.get("receipt_ref"), dict), "receipt ref")
    _, artifact = _ref(raw["receipt_ref"], "receipt")
    _require(dict(raw) == {**artifact, "receipt_ref": raw["receipt_ref"]}, "receipt body differs")
    protocol, request = intent["context"]["protocol_body"], intent["request"]
    admission_ref = _verify_admission(intent, require_live=False)
    _require(artifact.get("lease_admission") == admission_ref, "historical lease admission differs")
    diagnostic, operation = protocol["diagnostic"], protocol["operation"]
    prior = _prior_training(intent["context"])
    if prior is None:
        _require("training_reuse" not in artifact, "unregistered prior training receipt")
    else:
        stage = Path(prior["training_stage"])
        _require(artifact.get("training_reuse") == _reuse_evidence(prior)
                 and artifact.get("training_stage") == session_io.reference(stage / "execution.json")
                 and artifact.get("new_checkpoint") == prior["training_result"]["checkpoint"],
                 "prior training receipt differs")
    expected = {"request_id": request["request_id"], "effect_key": request["effect_key"],
                "mechanism": diagnostic["mechanism"], "correction_id": operation["correction_id"],
                "recovery_mode": operation["recovery_mode"], "source_sha256": diagnostic["source_sha256"],
                "checkpoint_sha256": diagnostic["checkpoint_sha256"],
                "source_checkpoint_consumed": operation["source_checkpoint_consumed"]}
    _require(all(artifact.get(key) == value for key, value in expected.items()), "receipt binding mismatch")
    checkpoint_path = Path(artifact["new_checkpoint"]["path"]).resolve(strict=True)
    _require(session_io.file_sha(checkpoint_path) == artifact["new_checkpoint"]["sha256"], "checkpoint changed")
    module = _runner_module(operation["runner_id"])
    preservation = artifact.get("environment_preservation")
    if operation["runner_id"] == _PRESERVING_RUNNER:
        _require(isinstance(preservation, dict)
                 and set(preservation) == {"evidence", "verification"},
                 "environment preservation receipt missing")
        assert isinstance(preservation, dict)
        preserved = _verify_environment_preservation(operation["runner_id"], module,
            preservation["evidence"], artifact["new_checkpoint"],
            intent["context"]["artifacts"][operation["training_binding_artifact_id"]]["body"])
        _require(preservation == preserved, "environment preservation receipt differs")
    else:
        _require(preservation is None, "unregistered environment preservation receipt")
    evidence = artifact.get("training_evidence")
    if (not isinstance(evidence, dict) or not {"manifest", "readback"} <= set(evidence)
            or not set(evidence) <= {"manifest", "readback", "checkpoint", "checkpoint_execution",
                                     "training_receipt", "records", "telemetry", "telemetry_execution"}):
        raise ValueError("training evidence refs")
    _, manifest = _ref(evidence["manifest"], "training manifest")
    _verify_training_manifest(manifest, operation, diagnostic)
    _, readback = _ref(evidence["readback"], "correction readback")
    checked_readback = _validate_readback(operation["runner_id"], module, readback)
    _require(artifact["correction_readback"] == checked_readback, "correction readback differs")
    checkpoint_execution = None
    telemetry_execution = None
    if operation["runner_id"] == "actuator_resume_v1":
        if "training_receipt" not in evidence:
            raise ValueError("actuator training receipt missing")
        _, training_receipt = _ref(evidence["training_receipt"], "actuator training receipt")
        _require(training_receipt.get("checkpoint") == artifact["new_checkpoint"]
                 and training_receipt.get("resume_checkpoint", {}).get("sha256") == diagnostic["checkpoint_sha256"],
                 "actuator checkpoint/resume binding mismatch")
    else:
        if not {"checkpoint", "checkpoint_execution", "records", "telemetry", "telemetry_execution"} <= set(evidence):
            raise ValueError("optimizer checkpoint evidence/execution missing")
        _, checkpoint_evidence = _ref(evidence["checkpoint"], "checkpoint evidence")
        _require(checkpoint_evidence.get("final", {}).get("sha256") == artifact["new_checkpoint"]["sha256"],
                 "optimizer checkpoint evidence mismatch")
        _, checkpoint_execution = _ref(evidence["checkpoint_execution"], "checkpoint execution")
        _require(checkpoint_execution.get("exit_code") == 0
                 and checkpoint_execution.get("natural_exit") is True
                 and checkpoint_execution.get("cleanup_verified") is True
                 and type(checkpoint_execution.get("wall_s")) in {int, float}
                 and math.isfinite(checkpoint_execution["wall_s"])
                 and checkpoint_execution["wall_s"] >= 0,
                 "checkpoint inspection execution differs")
        _, telemetry_execution = _ref(evidence["telemetry_execution"], "telemetry execution")
        _require(telemetry_execution.get("exit_code") == 0
                 and telemetry_execution.get("natural_exit") is True
                 and telemetry_execution.get("cleanup_verified") is True,
                 "telemetry extraction execution differs")
        _execution_wall({"execution": telemetry_execution})
        _ref(evidence["telemetry"], "training telemetry")
        records_path = Path(evidence["records"]["path"]).resolve(strict=True)
        _require(session_io.file_sha(records_path) == evidence["records"]["sha256"], "training records changed")
    _require(artifact.get("runtime_confirmation") == _runtime_confirmation(
        operation["runner_id"], evidence, artifact["new_checkpoint"]), "corrected runtime confirmation differs")
    _, report = _ref(artifact["fixed_report"], "fixed report")
    fixed_binding = intent["context"]["artifacts"][operation["fixed_binding_artifact_id"]]["body"]
    verified = grounded_recovery.validate_fixed_report(report, {
        "name": report["name"], "checkpoint_file": checkpoint_path.name,
        "checkpoint_sha256": artifact["new_checkpoint"]["sha256"],
        "action_scale": fixed_binding["action_scale"],
        "protocol_sha256_lf": fixed_binding["protocol_sha256_lf"],
    })
    _require(artifact["metrics"] == verified["metrics"], "metrics differ from fixed report")
    _require(artifact["reference_report"] == fixed_binding["reference_report"],
             "reference report ref differs from preregistration")
    _, reference = _ref(artifact["reference_report"], "reference report")
    reference = grounded_recovery.validate_fixed_report(reference, {
        "name": reference["name"], "checkpoint_file": reference["checkpoint"]["file"],
        "checkpoint_sha256": reference["checkpoint"]["sha256"],
        "action_scale": fixed_binding["action_scale"],
        "protocol_sha256_lf": fixed_binding["protocol_sha256_lf"],
    })
    verdict, comparison, behavior_verdict = _behavior_verdict(verified, reference, fixed_binding["bands"])
    _require(artifact["comparison"] == comparison
             and artifact["metric_checks"] == behavior_verdict["checks"]
             and artifact.get("behavior_qualification") == behavior_verdict["qualification"],
             "behavior verdict differs")
    stages = {}
    for key in ("training_stage", "fixed_stage"):
        _, stage = _ref(artifact[key], key)
        _require(stage.get("exit_code") == 0 and stage.get("natural_exit") is True
                 and (stage.get("cleanup_complete") is True or stage.get("cleanup_verified") is True),
                 f"{key} not complete")
        _require(type(stage.get("wall_s")) in {int, float} and math.isfinite(stage["wall_s"])
                 and stage["wall_s"] >= 0, f"{key} wall missing")
        stages[key] = stage
    costs = artifact["costs"]
    _require(set(costs) == {"prepare_s", "authority_s", "queue_s", "training_s", "fixed_s", "postcheck_s", "api_calls"}
             and costs["api_calls"] == 0, "cost fields")
    _require(all(type(costs[key]) in {int, float} and math.isfinite(costs[key]) and costs[key] >= 0
                 for key in costs if key != "api_calls"), "cost values")
    _require(math.isclose(artifact["gpu_s"], costs["training_s"] + costs["fixed_s"]), "gpu cost differs")
    _require(math.isclose(float(stages["training_stage"].get("wall_s")), costs["training_s"])
             and math.isclose(float(stages["fixed_stage"].get("wall_s")), costs["fixed_s"]),
             "owned stage wall differs from cost receipt")
    expected_verdict = "unknown" if artifact["gpu_s"] > operation["gpu_budget_s"] else verdict
    _require(artifact["verdict"] == expected_verdict, "budget-adjusted verdict differs")
    breakdown = artifact.get("owned_breakdown")
    if (not isinstance(breakdown, dict)
            or set(breakdown) != {"simulator_training_wall_s", "simulator_fixed_wall_s",
                                  "checkpoint_inspection_owned_wall_s", "checkpoint_cpu_process_s",
                                  "checkpoint_included_in_postcheck_s", "telemetry_extraction_owned_wall_s",
                                  "telemetry_included_in_postcheck_s"}
            or breakdown["simulator_training_wall_s"] != costs["training_s"]
            or breakdown["simulator_fixed_wall_s"] != costs["fixed_s"]):
        raise ValueError("owned cost breakdown differs")
    if operation["runner_id"] == "actuator_resume_v1":
        _require(breakdown["checkpoint_inspection_owned_wall_s"] is None
                 and breakdown["checkpoint_included_in_postcheck_s"] is False
                 and breakdown["telemetry_extraction_owned_wall_s"] is None
                 and breakdown["telemetry_included_in_postcheck_s"] is False,
                 "actuator must not claim a separate checkpoint child")
    else:
        assert checkpoint_execution is not None
        _require(breakdown["checkpoint_inspection_owned_wall_s"] == checkpoint_execution["wall_s"]
                 and breakdown["checkpoint_cpu_process_s"] is None
                 and breakdown["checkpoint_included_in_postcheck_s"] is True,
                 "checkpoint CPU/owned wall breakdown differs")
        assert telemetry_execution is not None
        _require(breakdown["telemetry_extraction_owned_wall_s"] == telemetry_execution["wall_s"]
                 and breakdown["telemetry_included_in_postcheck_s"] is True,
                 "telemetry CPU/owned wall breakdown differs")
    return deepcopy(dict(raw))


def preflight(payload: Mapping[str, Any]) -> dict:
    result = prepare(payload)
    return {"status": "prepared_cpu_only", "backend_id": BACKEND_ID,
            "protocol_sha256": result["protocol"]["sha256"], "processes_started": 0,
            "gpu_queries": 0, "api_calls": 0}


def build_backend() -> dict[str, Any]:
    return {"backend_id": BACKEND_ID, "prepare": prepare, "authorize": authorize, "admit": admit,
            "execute": execute, "verify_receipt": verify_receipt, "preflight": preflight}
