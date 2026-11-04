"""Resumable bridge from a confirmed investigation to registered recovery execution."""
from __future__ import annotations

import importlib
import time
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from rl_triage import intervention_governor as governor
from rl_triage import investigation_session as io
from rl_triage import observable_behavior_v1 as behavior
from rl_triage import registered_diagnostic_backend
from rl_triage.correction_ledger import CorrectionLedger
from rl_triage.correction_registry import protocol_case_arm
from rl_triage.correction_registry_v2 import REGISTRY, correction_for, registry_digest
from rl_triage.investigation_session import InvestigationSession

SCHEMA = "investigation_recovery_v1"
_PROBE = {"optimizer_update_count": "P_optimizer_count",
          "optimizer_surrogate": "P_optimizer_gradient", "actuator": "P_actuator_mapping"}


def _backend() -> dict[str, Any]:
    return importlib.import_module("evals.registered_recovery_backend_v1").build_backend()


def _resources():
    return importlib.import_module("evals.registered_recovery_resources_v1")


def _read(path: Path) -> dict:
    return io.read_ref(io.reference(path))


def _write_or_match(path: Path, value: dict) -> None:
    if path.exists():
        if _read(path) != value:
            raise ValueError(f"existing {path.name} differs")
    else:
        io.write_new(path, value)


def _diagnostic(session: InvestigationSession, mechanism: str) -> tuple[dict, dict, dict]:
    expected_probe = _PROBE.get(mechanism)
    if expected_probe is None:
        raise ValueError("confirmed mechanism has no recovery diagnostic binding")
    rows = [row for row in session.ledger.requests().values()
            if row["state"] == "done" and row["probe"] == expected_probe
            and row["receipt"]["outcome"] == "abnormal"]
    if len(rows) != 1:
        raise ValueError("exactly one confirming abnormal diagnostic receipt required")
    request = rows[0]
    consumed = [event for event in session.ledger.events()
                if event["event"] == "consumed" and event["request_id"] == request["request_id"]]
    if len(consumed) != 1:
        raise ValueError("diagnostic consumption binding missing")
    intent = session._intent(request, consumed[0])
    raw = _read(Path(intent["host_receipt_path"]))
    verified = registered_diagnostic_backend.verify_receipt(deepcopy(raw), deepcopy(intent))
    if verified["outcome"] != "abnormal" or verified["probe"] != expected_probe:
        raise ValueError("diagnostic receipt does not confirm the mechanism")
    return request, intent, verified


def prepare(session_path: Path, recovery_payload: Mapping[str, Any], folder: Path) -> dict:
    """Revalidate the completed diagnostic and registered recovery payload; start no process."""
    started = time.perf_counter()
    target = Path(folder).resolve()
    if target.exists():
        raise FileExistsError("recovery folder must be fresh")
    source_session = InvestigationSession(Path(session_path))
    source_session._verify_operation(registered_diagnostic_backend.verify_source)
    state = source_session.status()
    if state["status"] != "confirmed" or not isinstance(state["confirmed_mechanism"], str):
        raise ValueError("completed confirmed investigation required")
    mechanism = state["confirmed_mechanism"]
    request, _, diagnostic = _diagnostic(source_session, mechanism)
    backend = _backend()
    context = backend["prepare"](recovery_payload)
    protocol = context["protocol_body"]
    binding = protocol["diagnostic"]
    expected = {"session_manifest_sha256": source_session._manifest_sha256,
                "source_sha256": source_session.manifest["source"]["sha256"],
                "checkpoint_sha256": source_session.source["checkpoint_sha256"],
                "case_id": source_session.source["case_id"], "mechanism": mechanism,
                "probe": request["probe"], "diagnostic_receipt_sha256": diagnostic["receipt_ref"]["sha256"]}
    if binding != expected:
        raise ValueError("recovery protocol diagnostic binding differs")
    target.mkdir(parents=True, exist_ok=False)
    manifest = {"schema": SCHEMA, "investigation": io.reference(Path(session_path) / "manifest.json"),
                "session_path": str(Path(session_path).resolve()), "loop_state_digest": io.digest(state),
                "diagnostic_receipt": diagnostic["receipt_ref"], "mechanism": mechanism,
                "recovery_context": context, "registry_sha256": registry_digest(),
                "prepare_s": max(0.0, time.perf_counter() - started)}
    io.write_new(target / "manifest.json", manifest)
    io.write_new(target / "anchor.json", {"schema": SCHEMA + "_anchor",
                                           "manifest_sha256": io.file_sha(target / "manifest.json")})
    return status(target)


def _load(folder: Path) -> tuple[dict, CorrectionLedger]:
    root = Path(folder).resolve(strict=True)
    anchor = _read(root / "anchor.json")
    manifest_path = root / "manifest.json"
    if anchor != {"schema": SCHEMA + "_anchor", "manifest_sha256": io.file_sha(manifest_path)}:
        raise ValueError("recovery anchor changed")
    manifest = _read(manifest_path)
    if manifest["registry_sha256"] != registry_digest():
        raise ValueError("recovery registry changed")
    session = InvestigationSession(Path(manifest["session_path"]))
    session._verify_operation(registered_diagnostic_backend.verify_source)
    if io.digest(session.status()) != manifest["loop_state_digest"]:
        raise ValueError("investigation state changed")
    _diagnostic(session, manifest["mechanism"])
    return manifest, CorrectionLedger(root / "recovery.jsonl", registry=REGISTRY)


def propose(folder: Path) -> dict:
    manifest, ledger = _load(folder)
    mechanism = manifest["mechanism"]
    entry = correction_for(mechanism)
    operation = manifest["recovery_context"]["protocol_body"]["operation"]
    if operation["correction_id"] != entry["correction_id"]:
        raise ValueError("protocol correction differs from registry")
    event = ledger.propose(manifest["loop_state_digest"], mechanism,
                           manifest["diagnostic_receipt"]["sha256"], entry["correction_id"],
                           operation["recovery_mode"], protocol_case_arm(mechanism, REGISTRY),
                           operation["gpu_budget_s"], evidence={"diagnostic_receipt": manifest["diagnostic_receipt"]})
    return {"event": event, "status": status(folder)}


def approve(folder: Path, request_id: str, actor: str, approval_ref: Mapping[str, Any]) -> dict:
    started = time.perf_counter()
    manifest, ledger = _load(folder)
    request = ledger.requests().get(request_id)
    if request is None or request["state"] != "pending":
        raise ValueError("pending correction request required")
    proof = _backend()["authorize"](manifest["recovery_context"], request, actor, approval_ref)
    _write_or_match(Path(folder) / "approvals" / f"{request_id}.json",
                    {"schema": SCHEMA + "_approval", "proof": proof, "request_sha256": io.digest(request)})
    ledger.approve(request_id, actor)
    _write_or_match(Path(folder) / "authority_costs" / f"{request_id}.json",
                    {"authority_s": max(0.0, time.perf_counter() - started)})
    return status(folder)


def dispatch(folder: Path, request_id: str) -> dict:
    manifest, ledger = _load(folder)
    request = ledger.requests().get(request_id)
    if request is None or request["state"] != "approved":
        raise ValueError("approved unused correction request required")
    approval = _read(Path(folder) / "approvals" / f"{request_id}.json")
    authority_cost = _read(Path(folder) / "authority_costs" / f"{request_id}.json")
    backend = _backend()
    admission = backend["admit"](manifest["recovery_context"], request)
    admission_path = Path(folder) / "admissions" / f"{request_id}.json"
    _write_or_match(admission_path, admission)
    admission_ref = io.reference(admission_path)
    measurement = _resources().Measurement(Path(folder), request, admission_ref,
                                           manifest["recovery_context"])
    try:
        consume = ledger.consume(request_id, manifest["loop_state_digest"])
    except BaseException as exc:
        measurement.finish(raw=None, error=exc)
        raise
    intent = {"context": manifest["recovery_context"], "request": request, "consumption": consume,
              "authorization": approval["proof"], "admission_ref": admission_ref,
              "costs": {"prepare_s": manifest["prepare_s"], "authority_s": authority_cost["authority_s"],
                        "queue_s": 0.0}}
    _write_or_match(Path(folder) / "intents" / f"{request_id}.json", intent)
    raw = None
    error = None
    try:
        raw = backend["execute"](deepcopy(intent))
        verified = backend["verify_receipt"](deepcopy(raw), deepcopy(intent))
        _write_or_match(Path(folder) / "receipts" / f"{request_id}.json", verified)
        ledger.receipt(request_id, "registered_recovery", verified["verdict"],
                       verified["receipt_ref"]["sha256"], verified["gpu_s"], verified["exit_code"])
    except Exception as exc:  # noqa: BLE001 - consumed execution is never repeated automatically.
        error = exc
        _write_or_match(Path(folder) / "dispatch_errors" / f"{request_id}.json",
                        {"schema": SCHEMA + "_dispatch_error", "error_type": type(exc).__name__,
                         "reexecute_allowed": False})
    finally:
        measurement.finish(raw=raw, error=error)
    return status(folder)


def settle(folder: Path, request_id: str, receipt: Mapping[str, Any] | None) -> dict:
    """Close an ambiguous consumed request by verification or unknown, never by rerunning it."""
    _manifest, ledger = _load(folder)
    request = ledger.requests().get(request_id)
    if request is None or request["state"] != "running":
        raise ValueError("running correction request required")
    intent = _read(Path(folder) / "intents" / f"{request_id}.json")
    if receipt is None:
        ledger.recover(request_id, None)
    else:
        verified = _backend()["verify_receipt"](deepcopy(receipt), deepcopy(intent))
        _write_or_match(Path(folder) / "receipts" / f"{request_id}.json", verified)
        ledger.recover(request_id, {"tag": "registered_recovery", "verdict": verified["verdict"],
                                    "result_sha": verified["receipt_ref"]["sha256"],
                                    "gpu_s": verified["gpu_s"], "exit_code": verified["exit_code"],
                                    "source": verified["receipt_ref"]["path"]})
    return status(folder)


def verified_post_observation(folder: Path, request_id: str) -> dict:
    """Expose only reverified fixed metrics and correction evidence; it does not declare health."""
    manifest, ledger = _load(folder)
    request = ledger.requests().get(request_id)
    if request is None or request["state"] != "done":
        raise ValueError("completed recovery required")
    raw = _read(Path(folder) / "receipts" / f"{request_id}.json")
    intent = _read(Path(folder) / "intents" / f"{request_id}.json")
    verified = _backend()["verify_receipt"](raw, intent)
    return {"schema": "verified_recovery_post_observation_v1", "case_id": manifest["recovery_context"]
            ["protocol_body"]["diagnostic"]["case_id"], "mechanism": manifest["mechanism"],
            "correction_id": verified["correction_id"], "correction_readback": verified["correction_readback"],
            "metrics": verified["metrics"], "fixed_report": verified["fixed_report"],
            "reference_report": verified["reference_report"], "comparison": verified["comparison"],
            "new_checkpoint": verified["new_checkpoint"], "costs": verified["costs"],
            "runtime_confirmation": verified["runtime_confirmation"],
            "behavior_outcome": {"recovered": "normal", "not_recovered": "failed",
                                 "unknown": "unknown"}[verified["verdict"]],
            "metric_checks": verified["metric_checks"], "healthy_declared": False}


def sealed_post_observation(
    folder: Path, request_id: str, source_observation: governor.VerifiedObservation, *,
    post_confirmation_refs: list[Mapping[str, Any]] | None = None,
    post_cause_ref: Mapping[str, Any] | None = None,
) -> governor.VerifiedObservation:
    """Reverify recovery evidence and seal it for the U2 governor without inventing post checks."""
    if not isinstance(source_observation, governor.VerifiedObservation):
        raise TypeError("source VerifiedObservation required")
    source = governor.build_model_packet(source_observation)["observation"]
    post = verified_post_observation(folder, request_id)
    if source.get("verifier_id") in {"recorded_public_fixed26_v1", behavior.VERIFIER_ID}:
        if post_confirmation_refs or post_cause_ref is not None:
            raise ValueError("actual post confirmation requires a registered new-checkpoint verifier")
        qualified = (behavior.qualify_source_observation(source_observation)
                     if source["verifier_id"] == "recorded_public_fixed26_v1" else source_observation)
        qualified_source = governor.build_model_packet(qualified)["observation"]
        reference_ref = qualified_source["source_refs"]["reference_fixed26"]
        original_reference = behavior._unwrap(io.read_ref(post["reference_report"]), "recovery reference")
        public_reference = behavior._unwrap(io.read_ref(reference_ref), "source reference")
        common = ("task", "protocol", "protocol_sha256_lf", "eval_seed", "num_envs",
                  "horizon_steps", "step_dt", "action_scale", "standstill", "metrics")
        if (any(original_reference.get(key) != public_reference.get(key) for key in common)
                or original_reference["checkpoint"]["sha256"] != public_reference["checkpoint"]["sha256"]):
            raise ValueError("recovery reference differs from the qualified source reference")
        classified = behavior.classify_post_observation(
            qualified, post["fixed_report"], reference_ref, post["new_checkpoint"])
        value = governor.build_model_packet(classified)["observation"]
        runtime = post.get("runtime_confirmation")
        checks = ([{"check_id": "post_cause_confirmation", "outcome": "unknown",
                    "refs": [f"sha256:{post['fixed_report']['sha256']}"]}]
                  if runtime is None else deepcopy(runtime["checks"]))
        if post.get("behavior_outcome") == "unknown":
            checks.append({"check_id": "post_execution_verdict", "outcome": "unknown",
                           "refs": [f"sha256:{post['fixed_report']['sha256']}"]})
        value.update({"receipt_id": f"recovery:{request_id}",
                      "verifier_id": "registered_recovery_absolute_fixed26_v1",
                      "verification_scope": "registered_recovery_absolute_fixed26_post_evidence",
                      "confirmation_checks": checks,
                      "post_confirmation_scope": runtime.get("scope") if runtime is not None else None,
                      "confirmation_catalog_complete": (runtime.get("catalog_coverage_complete", False)
                                                        if runtime is not None else False),
                      "unobserved_confirmation_components": (runtime.get("unobserved_components", [])
                                                              if runtime is not None else []),
                      "correction_receipt": {"correction_id": post["correction_id"], "status": "applied",
                                             "refs": [f"sha256:{post['new_checkpoint']['sha256']}",
                                                      f"sha256:{post['fixed_report']['sha256']}"]}})
        value["source_refs"]["recovery_original_reference"] = deepcopy(post["reference_report"])
        return governor._verified_observation(value)
    reference = source.get("reference")
    criteria = source.get("criteria")
    if not isinstance(reference, dict) or not isinstance(criteria, dict):
        reference_matches = False
        behavior_outcome, metric_checks = "unknown", []
    else:
        raw_reference = io.read_ref(post["reference_report"])
        reference_matches = raw_reference.get("metrics") == reference.get("metrics")
        candidate = deepcopy(source["candidate"])
        candidate["checkpoint_sha256"] = post["new_checkpoint"]["sha256"]
        candidate["metrics"] = deepcopy(post["metrics"])
        candidate["refs"] = list(dict.fromkeys(candidate["refs"] + [
            f"sha256:{post['fixed_report']['sha256']}", f"sha256:{post['new_checkpoint']['sha256']}"]))
        if reference_matches:
            behavior_outcome, metric_checks = governor._metric_checks(candidate, reference, criteria)
        else:
            behavior_outcome, metric_checks = "unknown", []

    checks = []
    for index, ref in enumerate(post_confirmation_refs or []):
        body = io.read_ref(ref)
        check = governor._verify_check(body, source["candidate"]["source_binding_sha256"])
        if check["check_id"] in {row["check_id"] for row in checks}:
            raise ValueError("duplicate post confirmation check")
        checks.append(check)
    cause = None
    if post_cause_ref is not None:
        body = io.read_ref(post_cause_ref)
        fields = {"schema", "confirmation_id", "source_binding_sha256", "status", "mechanism",
                  "registered_correction_id", "refs"}
        if (set(body) != fields or body["schema"] != "cause_confirmation_receipt_v1"
                or body["source_binding_sha256"] != source["candidate"]["source_binding_sha256"]
                or body["status"] not in {"confirmed", "not_confirmed", "unknown"}
                or body["mechanism"] != post["mechanism"]
                or body["registered_correction_id"] != post["correction_id"]
                or not isinstance(body["refs"], list)):
            raise ValueError("post cause confirmation binding differs")
        cause = {key: deepcopy(body[key]) for key in fields - {"schema", "source_binding_sha256"}}
    if (not checks or cause is None) and not any(row["check_id"] == "post_cause_confirmation" for row in checks):
        checks.append({"check_id": "post_cause_confirmation", "outcome": "unknown",
                       "refs": [f"sha256:{post['fixed_report']['sha256']}"]})

    candidate = deepcopy(source["candidate"])
    candidate["checkpoint_sha256"] = post["new_checkpoint"]["sha256"]
    candidate["metrics"] = deepcopy(post["metrics"])
    candidate["refs"] = list(dict.fromkeys(candidate["refs"] + [
        f"sha256:{post['fixed_report']['sha256']}", f"sha256:{post['new_checkpoint']['sha256']}"]))
    correction = {"correction_id": post["correction_id"], "status": "applied",
                  "refs": [f"sha256:{post['new_checkpoint']['sha256']}",
                           f"sha256:{post['fixed_report']['sha256']}"]}
    payload = {"schema": governor.NORMALIZED_SCHEMA,
               "receipt_id": f"recovery:{request_id}", "verifier_id": "registered_recovery_v1",
               "verification_scope": "registered_recovery_fixed26_post_evidence",
               "candidate": candidate, "reference": deepcopy(reference), "criteria": deepcopy(criteria),
               "reference_available": reference is not None,
               "lineage_matches": bool(source.get("lineage_matches") and reference_matches),
               "behavior_outcome": behavior_outcome, "metric_checks": metric_checks,
               "confirmation_checks": checks, "cause_confirmation_receipt": cause,
               "correction_receipt": correction,
               "source_refs": {**deepcopy(source["source_refs"]),
                               "recovery_fixed_eval": post["fixed_report"],
                               "recovery_checkpoint": post["new_checkpoint"]}}
    return governor._verified_observation(payload)


def status(folder: Path) -> dict:
    root = Path(folder).resolve(strict=True)
    manifest = _read(root / "manifest.json")
    ledger = CorrectionLedger(root / "recovery.jsonl", registry=REGISTRY)
    requests = ledger.requests()
    live = [row for row in requests.values() if row["state"] in {"pending", "approved", "running"}]
    return {"schema": SCHEMA + "_status", "mechanism": manifest["mechanism"],
            "requests": deepcopy(requests), "pending": deepcopy(live[0]) if live else None,
            "approval_granted_by_session": False, "automatic_reexecution": False}


def review(folder: Path, request_id: str, host_gate_sources: Mapping[str, Any]) -> dict:
    """Revalidate the registered source and post evidence; return a proposal without another execution."""
    started = time.perf_counter()
    manifest, _ledger = _load(folder)
    source_session = InvestigationSession(Path(manifest["session_path"]))
    binding = source_session.source["binding"]
    protocol, _protocol_ref = registered_diagnostic_backend._protocol(binding["protocol"])
    resolver = registered_diagnostic_backend._resolver(binding["artifact_resolver"], protocol)
    source = registered_diagnostic_backend._source_observation(source_session.source["case_id"], protocol, resolver)
    observed = sealed_post_observation(folder, request_id, source)
    gate = governor.verify_host_gate_receipt(host_gate_sources)
    decision = governor.decide_intervention(observed, None, gate)
    payload = governor.build_model_packet(observed)["observation"]
    return {"schema": "registered_recovery_post_review_v1", "decision": decision,
            "behavior_outcome": payload["behavior_outcome"],
            "confirmation_scope": payload.get("post_confirmation_scope"),
            "confirmation_catalog_complete": payload.get("confirmation_catalog_complete", False),
            "unobserved_components": payload.get("unobserved_confirmation_components", []),
            "cpu_review_wall_s": max(0.0, time.perf_counter() - started),
            "api_calls": 0, "gpu_launches": 0}
