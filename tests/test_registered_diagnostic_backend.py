from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from rl_triage import investigate, nim_transport
from rl_triage import investigation_session as SESSION
from rl_triage import mechanism_confirmation as CONFIRMATION
from rl_triage import mechanism_registry as REGISTRY
from rl_triage import registered_diagnostic_backend as B


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: dict) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"path": str(path.resolve()), "sha256": _sha(path)}


def _head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True,
        encoding="utf-8",
    ).stdout.strip()


def _protocol(
    tmp_path: Path,
    *,
    execution_class: str = "actual",
    registry: dict | None = None,
    registry_profile: str = "full",
) -> dict:
    registry = registry or CONFIRMATION.mechanism_registry()
    components = {
        probe: {
            "component_id": f"component:{probe}",
            "dispatcher_id": "bench.diagnostic_observer_child_v1",
            "execution_class": execution_class,
            "artifacts": {},
        }
        for probe in registry["probes"]
    }
    value = {
        "schema": B.PROTOCOL_SCHEMA,
        "backend_id": B.BACKEND_ID,
        "source_registration": B._source_registration(),
        "registry_sha256": REGISTRY.registry_digest(registry),
        "registry_profile": registry_profile,
        "model": {
            "source_anchor": _head(),
            "params_sha256": B._model_params_sha(),
            "api_cap": 2,
            "api_scope_id": "shared_nim_calls_v1",
        },
        "components": components,
        "actors": ["registered-host"],
    }
    return _write(tmp_path / "registered_protocol.json", value)


def _prepare_payload(
    tmp_path: Path,
    protocol: dict,
    *,
    case_id: str = "case-03",
    method: str = "fixed_sequence",
    execution_class: str = "actual",
) -> dict:
    return {
        "source_path": str((tmp_path / f"source-{case_id}-{method}.json").resolve()),
        "case_id": case_id,
        "execution_class": execution_class,
        "method": method,
        "protocol": protocol,
        "artifact_resolver": {},
        "api_root": (str((tmp_path / "private-api").resolve())
                     if method in SESSION.MODEL_METHODS else None),
        "max_checks": 2,
        "gpu_budget_per_check_s": 0,
    }


def _approval(tmp_path: Path, session: SESSION.InvestigationSession, protocol: dict) -> dict:
    pending = session.status()["pending"]
    value = {
        "schema": B.APPROVAL_SCHEMA,
        "approval_id": "approval-001",
        "protocol_sha256": protocol["sha256"],
        "source_sha256": session.manifest["source"]["sha256"],
        "case_id": session.source["case_id"],
        "checkpoint_sha256": session.source["checkpoint_sha256"],
        "probe": pending["probe"],
        "request_id": pending["request_id"],
        "effect_key": pending["effect_key"],
        "registry_sha256": session.manifest["registry_sha256"],
        "actor": "registered-host",
        "approved": True,
    }
    return _write(tmp_path / "approval.json", value)


def test_build_backend_has_registered_exact_contract():
    backend = B.build_backend()
    assert set(backend) == {
        "backend_id", "prepare_source", "verify_source", "authorize", "execute",
        "verify_receipt", "complete", "observation_gate",
    }
    assert backend["backend_id"] == "registered_diagnostic_v1"
    assert backend["complete"] is nim_transport.complete
    loaded, reason = investigate._load_backend(B.BACKEND_ID, False)
    assert reason is None and loaded == backend


def test_prepare_revalidates_actual_public_source_without_hidden_truth(tmp_path):
    protocol = _protocol(tmp_path)
    prepared = B.prepare_source(_prepare_payload(tmp_path, protocol))
    assert set(prepared) == {"source_path", "registry", "model_binding", "execution_class"}
    assert prepared["model_binding"] is None and prepared["execution_class"] == "actual"
    source_path = Path(prepared["source_path"])
    source = json.loads(source_path.read_text(encoding="utf-8"))
    assert set(source) == {"schema", "case_id", "checkpoint_sha256", "facts", "binding"}
    assert len(source["facts"]) == 12
    assert all(type(value) is float for value in source["facts"].values())
    serialized = json.dumps(source, sort_keys=True)
    for hidden in ("source_type", "classification", "known_faults", "label", "scoring"):
        assert hidden not in serialized
    first = B.verify_source(source_path)
    second = B.verify_source(source_path)
    assert first == second
    assert first["status"] == "verified"
    assert first["source_sha256"] == _sha(source_path)


def test_runtime_registry_profile_exposes_exactly_three_supported_tools(tmp_path):
    probes = (CONFIRMATION.COUNT_PROBE, CONFIRMATION.GRADIENT_PROBE,
              CONFIRMATION.ACTUATOR_PROBE)
    runtime = CONFIRMATION.mechanism_registry(profile="runtime_confirmation_v1")
    protocol = _protocol(
        tmp_path, registry=runtime, registry_profile="runtime_confirmation_v1")
    prepared = B.prepare_source(_prepare_payload(tmp_path, protocol))
    assert list(prepared["registry"]["probes"]) == sorted(probes)
    assert prepared["registry"]["fixed_order"] == list(probes)
    source = json.loads(Path(prepared["source_path"]).read_text(encoding="utf-8"))
    assert source["binding"]["registry_profile"] == "runtime_confirmation_v1"

    value = json.loads(Path(protocol["path"]).read_text(encoding="utf-8"))
    value["components"]["P_episode"] = {
        "component_id": "legacy", "dispatcher_id": "legacy",
        "execution_class": "actual", "artifacts": {},
    }
    bad_protocol = _write(tmp_path / "bad-profile-protocol.json", value)
    bad_payload = _prepare_payload(tmp_path / "bad", bad_protocol)
    with pytest.raises(ValueError, match="component set"):
        B.prepare_source(bad_payload)


def test_parent_protocol_resolves_registered_collection_manifest(tmp_path):
    protocol = _protocol(tmp_path)
    value = json.loads(Path(protocol["path"]).read_text(encoding="utf-8"))
    value["source_registration"]["collection_artifact_id"] = "source-export-manifest"
    protocol = _write(tmp_path / "source-registered-protocol.json", value)
    payload = _prepare_payload(tmp_path, protocol)
    payload["artifact_resolver"] = {
        "source-export-manifest": {
            "path": str((B.EVIDENCE.SOURCE_COLLECTION / "manifest.json").resolve()),
            "sha256": B.EVIDENCE.SOURCE_MANIFEST_SHA256,
        },
    }
    prepared = B.prepare_source(payload)
    source = json.loads(Path(prepared["source_path"]).read_text(encoding="utf-8"))
    assert source["binding"]["source_registration"] == value["source_registration"]
    assert B.verify_source(Path(prepared["source_path"]))["status"] == "verified"


@pytest.mark.parametrize(
    "method", ["fixed_sequence", "static_rule", "one_shot_llm", "iterative_agent"],
)
def test_verified_normal_host_gate_precedes_every_arm_and_calls_no_model(tmp_path, method):
    protocol = _protocol(tmp_path)
    payload = _prepare_payload(tmp_path, protocol, case_id="case-01", method=method)
    session_path = tmp_path / f"session-{method}"
    prepared = investigate.run("prepare", session_path, payload,
                               backend_name=B.BACKEND_ID, fixture=False)
    assert prepared["status"] == "open"
    result = investigate.run("next", session_path, {}, backend_name=B.BACKEND_ID, fixture=False)
    assert result["status"] == "host_no_intervention"
    assert result["approval_granted_by_session"] is False
    assert not (session_path / "model").exists()
    gate = json.loads((session_path / "host_gate.json").read_text(encoding="utf-8"))
    assert gate["reason"] == "verified_normal_behavior_bundle"


def test_model_binding_uses_verified_protocol_sha_and_private_root_only(tmp_path):
    protocol = _protocol(tmp_path)
    payload = _prepare_payload(tmp_path, protocol, method="iterative_agent")
    prepared = B.prepare_source(payload)
    binding = prepared["model_binding"]
    assert binding["protocol_sha256"] == protocol["sha256"]
    assert binding["source_anchor"] == _head()
    assert binding["params"] == nim_transport.PARAMS
    assert binding["api_root"] == str((tmp_path / "private-api").resolve())
    public_protocol = json.loads(Path(protocol["path"]).read_text(encoding="utf-8"))
    assert "protocol_sha256" not in public_protocol["model"]
    assert "api_root" not in public_protocol["model"]


def test_authorization_binds_actor_protocol_source_checkpoint_probe_and_effect(tmp_path):
    protocol = _protocol(tmp_path)
    payload = _prepare_payload(tmp_path, protocol)
    prepared = B.prepare_source(payload)
    session = SESSION.InvestigationSession.prepare(
        Path(prepared["source_path"]), tmp_path / "session", registry=prepared["registry"],
        method="fixed_sequence", max_checks=2, gpu_budget_per_check_s=0,
        source_verifier=B.verify_source, model_binding=None, execution_class="actual",
    )
    state = session.next(source_verifier=B.verify_source, observation_gate=B.observation_gate)
    approval = _approval(tmp_path, session, protocol)
    authorization = {"protocol": protocol, "approval": approval, "source": session.manifest["source"]}
    proof = B.authorize(state["pending"], "registered-host", authorization)
    assert proof["status"] == "authorized"
    assert proof["request_id"] == state["pending"]["request_id"]
    assert B.authorize(state["pending"], "different-actor", authorization) == {
        "status": "denied", "reason": "authorization_binding",
    }
    tampered = dict(state["pending"])
    tampered["effect_key"] = "different-effect"
    assert B.authorize(tampered, "registered-host", authorization)["status"] == "denied"


def test_u3_flow_consumes_approval_then_blocks_unattached_actual_dispatch(tmp_path):
    protocol = _protocol(tmp_path)
    payload = _prepare_payload(tmp_path, protocol)
    session_path = tmp_path / "session"
    investigate.run("prepare", session_path, payload, backend_name=B.BACKEND_ID, fixture=False)
    proposed = investigate.run("next", session_path, {}, backend_name=B.BACKEND_ID, fixture=False)
    session = SESSION.InvestigationSession(session_path)
    approval = _approval(tmp_path, session, protocol)
    authorized = investigate.run(
        "approve", session_path,
        {"request_id": proposed["pending"]["request_id"], "actor": "registered-host",
         "authorization": {"protocol": protocol, "approval": approval,
                           "source": session.manifest["source"]}},
        backend_name=B.BACKEND_ID, fixture=False,
    )
    assert authorized["status"] == "ready_to_dispatch"
    dispatched = investigate.run(
        "dispatch", session_path, {"request_id": proposed["pending"]["request_id"]},
        backend_name=B.BACKEND_ID, fixture=False,
    )
    assert dispatched["status"] == "pending_result"
    assert dispatched["dispatch_error"] == {
        "error_type": "BackendBlocked", "reexecute_allowed": False,
    }
    assert not (session_path / "host_receipts" / f"{proposed['pending']['request_id']}.json").exists()
    events = SESSION.InvestigationSession(session_path).ledger.events()
    assert any(row["event"] == "consumed" for row in events)
    assert not any(row["event"] == "receipt" for row in events)


def test_actual_dispatch_derives_manifest_from_registered_artifacts_without_intent_extension(
    tmp_path, monkeypatch,
):
    from bench import diagnostic_observer_child_v1 as child

    checkpoint = tmp_path / "case.pt"
    checkpoint.write_bytes(b"registered checkpoint")
    child_protocol = _write(
        tmp_path / "child-protocol.json", {"schema": "diagnostic_observer_protocol_v1"})
    config = {
        "schema": "registered_actual_dispatch_v1",
        "operation": "optimizer_update_bundle",
        "component": "optimizer_update_bundle",
        "probe": [CONFIRMATION.COUNT_PROBE, CONFIRMATION.GRADIENT_PROBE],
        "condition": "R1",
        "seed": 2201,
        "device": "cuda:0",
        "num_envs": 64,
        "case_binding": {"case_id": "case-03"},
        "source_binding": {"evidence_binding": "fixed26-public-v1"},
        "reference_binding": {"case_id": "reference-03"},
        "checkpoint_role": "identity_only",
        "source_anchor": _head(),
        "source_closure": {"bench/diagnostic_observer_child_v1.py": "a" * 64},
        "runtime_pins": {"logical_root": "isaaclab", "sha256": {"runtime": "b" * 64}},
        "isaaclab_root": str(tmp_path),
        "watchdog_s": 30,
    }
    execution = _write(tmp_path / "actual-execution.json", config)
    checkpoint_ref = {"path": str(checkpoint.resolve()), "sha256": _sha(checkpoint)}
    artifacts = {
        "execution": {"artifact_id": "actual-config", "sha256": execution["sha256"]},
        "child_protocol": {"artifact_id": "child-protocol", "sha256": child_protocol["sha256"]},
        "checkpoint": {"artifact_id": "checkpoint", "sha256": checkpoint_ref["sha256"]},
    }
    resolver = {"actual-config": execution, "child-protocol": child_protocol,
                "checkpoint": checkpoint_ref}
    component = {"dispatcher_id": "bench.diagnostic_observer_child_v1", "artifacts": artifacts}
    host_receipt = tmp_path / "session" / "host_receipts" / "request-1.json"
    host_receipt.parent.mkdir(parents=True)
    intent = {
        "request": {"request_id": "request-1", "effect_key": "effect-1",
                    "probe": CONFIRMATION.COUNT_PROBE},
        "consumption": {"event": "consumed"},
        "preregistration": {"source_sha256": "c" * 64},
        "source": {"case_id": "case-03", "checkpoint_sha256": checkpoint_ref["sha256"],
                   "binding": {"artifact_resolver": resolver}},
        "host_receipt_path": str(host_receipt),
    }
    seen = {}

    def build_manifest(**kwargs):
        seen["manifest"] = kwargs
        return {"schema": "cpu-preflight-manifest",
                **{key: str(value) if isinstance(value, Path) else value
                   for key, value in kwargs.items()}}

    def verify_manifest(path, **kwargs):
        seen["verified"] = {"path": path, **kwargs}
        return {"status": "verified"}

    def run_owned(folder, args, cwd, watchdog_s):
        seen["run"] = {"folder": folder, "args": args, "cwd": cwd,
                       "watchdog_s": watchdog_s}
        return {"exit_code": 0, "natural_exit": True, "cleanup_verified": True,
                "job_assigned_before_release": True, "timed_out": False,
                "error_type": None, "wall_s": 1.25}

    lease = {"owner": "rl-triage-lineage-sources-v1", "leaseId": "1" * 36,
             "state": "running", "childPid": 4242, "wrapperPid": 4200,
             "completedUtc": None}
    monkeypatch.setattr(child, "build_manifest", build_manifest)
    monkeypatch.setattr(child, "verify_manifest", verify_manifest)
    monkeypatch.setattr(child, "REGISTERED_PROTOCOL", Path(child_protocol["path"]))
    monkeypatch.setattr(B, "_lease_context", lambda: dict(lease))
    monkeypatch.setattr(B, "_run_owned", run_owned)
    monkeypatch.setattr(
        B, "_actual_observation",
        lambda output, manifest_sha, selected: (
            "normal", {"output": str(output), "manifest": manifest_sha, "selected": selected}),
    )
    outcome, measurement, wall_s, exit_code = B._execute_actual(
        intent, component, {"status": "authorized"})
    assert outcome == "normal" and wall_s == 1.25 and exit_code == 0
    assert measurement["gpu_s_scope"] == "lease_held_owned_child_wall_not_kernel_busy_time"
    assert seen["manifest"]["source_binding"]["source_sha256"] == "c" * 64
    assert seen["manifest"]["checkpoint_binding"] == {**checkpoint_ref, "role": "identity_only"}
    assert seen["manifest"]["probe"] == [CONFIRMATION.COUNT_PROBE,
                                           CONFIRMATION.GRADIENT_PROBE]
    assert seen["run"]["args"][0].endswith("_isaac_sim\\python.bat")
    assert seen["run"]["watchdog_s"] == 30.0
    stage = tmp_path / "session" / "actual_stages" / "request-1"
    for name in ("gpu-admission.json", "request.json", "approval.json", "consumption.json",
                 "manifest.json", "execution.json"):
        assert (stage / name).is_file()
    artifact = {"outcome": outcome, "measurement": measurement, "gpu_s": wall_s,
                "execution_wall_s": wall_s, "exit_code": exit_code}
    B._verify_actual_stage(intent, component, artifact)
    (stage / "execution.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="execution_ref binding"):
        B._verify_actual_stage(intent, component, artifact)


def test_prepare_rejects_caller_truth_and_non_finite_limits(tmp_path):
    protocol = _protocol(tmp_path)
    payload = _prepare_payload(tmp_path, protocol)
    payload["healthy"] = True
    with pytest.raises(ValueError, match="prepare payload fields"):
        B.prepare_source(payload)
    payload = _prepare_payload(tmp_path, protocol)
    payload["gpu_budget_per_check_s"] = float("nan")
    with pytest.raises(ValueError, match="limits"):
        B.prepare_source(payload)
