import hashlib
import json
import subprocess
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from evals import registered_recovery_backend_v1 as backend
from rl_triage.correction_registry_v2 import registry_digest
from rl_triage.investigation_session import file_sha

r1_runner = backend._runner_module("r1_epochs_v2")


def test_retry_admission_requires_both_policy_closures_and_routes_exact_scope(monkeypatch):
    closure = backend._DESKTOP_SOURCES | backend._RETRY_SOURCES
    protocol = {"schema": backend.SCHEMA_V3, "source_closure": dict.fromkeys(closure, "a" * 64),
                "operation": {"gpu_admission_profile": backend.RETRY_PROFILE,
                              "prior_training_artifact_id": None}}
    assert backend._extensions(protocol) is None
    calls = []
    scope = object()
    module = SimpleNamespace(snapshot_scope=lambda: scope)
    monkeypatch.setattr(backend.importlib, "import_module",
                        lambda name: calls.append(name) or module)
    assert backend._gpu_scope({"protocol_body": protocol}) is scope
    assert calls == ["evals.registered_desktop_gpu_retry_v1"]
    for relative in backend._RETRY_SOURCES:
        altered = deepcopy(protocol)
        del altered["source_closure"][relative]
        with pytest.raises(ValueError, match="policy source missing"):
            backend._extensions(altered)
    protocol["schema"] = backend.SCHEMA_V2
    with pytest.raises(ValueError, match="unregistered desktop GPU"):
        backend._extensions(protocol)


def test_v4_requires_actuator_prior_training_and_v5_closure() -> None:
    closure = (backend._DESKTOP_SOURCES | backend._RETRY_SOURCES | backend._PRIOR_SOURCES
               | backend._CONTINUATION_V5_SOURCES)
    protocol = {"schema": backend.SCHEMA_V4, "source_closure": dict.fromkeys(closure, "a" * 64),
                "operation": {"gpu_admission_profile": backend.RETRY_PROFILE,
                              "prior_training_artifact_id": "prior_training",
                              "training_binding_artifact_id": "training",
                              "fixed_binding_artifact_id": "fixed",
                              "runner_id": "actuator_resume_v1"}}
    assert backend._extensions(protocol) == "prior_training"
    protocol["operation"]["prior_training_artifact_id"] = None
    with pytest.raises(ValueError, match="v4 requires"):
        backend._extensions(protocol)
    protocol["operation"]["prior_training_artifact_id"] = "prior_training"
    protocol["source_closure"].pop("evals/registered_recovery_continuation_v5.py")
    with pytest.raises(ValueError, match="v5 continuation"):
        backend._extensions(protocol)


def test_v4_prior_training_routes_to_v5_with_backend_expectations(monkeypatch: pytest.MonkeyPatch) -> None:
    closure = (backend._DESKTOP_SOURCES | backend._RETRY_SOURCES | backend._PRIOR_SOURCES
               | backend._CONTINUATION_V5_SOURCES)
    operation = {"gpu_admission_profile": backend.RETRY_PROFILE,
                 "prior_training_artifact_id": "prior_training",
                 "training_binding_artifact_id": "training",
                 "fixed_binding_artifact_id": "fixed", "runner_id": "actuator_resume_v1",
                 "training_seed": 2665, "correction_id": "restore_motor_torque_speed"}
    protocol = {"schema": backend.SCHEMA_V4, "source_closure": dict.fromkeys(closure, "a" * 64),
                "operation": operation,
                "diagnostic": {"checkpoint_sha256": "1" * 64,
                               "session_manifest_sha256": "2" * 64}}
    binding = {"task": "Isaac-Velocity-Flat-Unitree-Go2-v0"}
    calls = []

    def load(ref, **kwargs):
        calls.append((ref, kwargs))
        return {"new_training_launched": False}

    monkeypatch.setattr(backend.importlib, "import_module",
                        lambda name: SimpleNamespace(load_completed_training=load))
    context = {"protocol_body": protocol, "artifacts": {
        "training": {"body": binding},
        "prior_training": {"ref": {"path": "proof.json", "sha256": "3" * 64}},
    }, "isaaclab_root": str(Path.cwd())}
    assert backend._prior_training(context) == {"new_training_launched": False}
    assert calls[0][0] == context["artifacts"]["prior_training"]["ref"]
    assert calls[0][1]["expected_source_checkpoint_sha256"] == "1" * 64
    assert calls[0][1]["expected_diagnostic_manifest_sha256"] == "2" * 64
    assert calls[0][1]["expected_binding"] == binding


def _write(path: Path, value: dict) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return {"path": str(path.resolve()), "sha256": file_sha(path)}


def _actual_fixed_report(source_id: str = "source-01") -> dict:
    from rl_triage.intervention_evidence import SOURCE_COLLECTION

    path = SOURCE_COLLECTION / "sources" / source_id / "fixed26.json"
    return json.loads(path.read_text(encoding="utf-8"))["report"]


def _payload(tmp_path: Path, *, hidden: bool = False) -> dict:
    minimum = {name: "b" * 64 for name in r1_runner.V1.WRAPPER.MINIMUM_BOUND_SOURCES}
    training = _write(tmp_path / "training.json", {
        "task": r1_runner.V1.TASK, "num_envs": 4096, "max_iterations": 300,
        "configured_learning_epochs": 5, "configured_mini_batches": 4,
        "runtime_updates_to_observe": 3, "source_commit_sha": "a" * 40,
        "protocol_path": "bench/protocols/test.json", "protocol_sha256_lf": "b" * 64,
        "source_sha256_lf": minimum, "runtime_source_sha256_lf": {"runtime.py": "c" * 64},
        "source_checkpoint_sha256": "3" * 64,
    })
    reference = _write(tmp_path / "reference.json", {"registered_fixed26_report": True})
    fixed = _write(tmp_path / "fixed.json", {"task": r1_runner.TASK,
                                               "runtime_source_sha256_lf": {"runtime.py": "c" * 64},
                                               "action_scale": 0.25, "protocol_sha256_lf": "a" * 64,
                                               "reference_report": reference,
                                               "bands": {"survival_ratio_min": 0.8, "fall_frac_max": 0.1,
                                                         "err_xy_ratio_max": 1.2, "err_yaw_ratio_max": 1.2}})
    resolver = _write(tmp_path / "resolver.json", {
        "schema": "registered_recovery_resolver_v1",
        "artifacts": {"training": training, "fixed": fixed},
    })
    root = Path(__file__).resolve().parents[1]
    closure_path = root / "src/rl_triage/correction_registry.py"
    anchor = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True,
                            capture_output=True, text=True).stdout.strip()
    closure_sha = hashlib.sha256(closure_path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    protocol = {
        "schema": backend.SCHEMA, "backend_id": backend.BACKEND_ID,
        "source_anchor": anchor,
        "source_closure": {"src/rl_triage/correction_registry.py": closure_sha},
        "diagnostic": {"session_manifest_sha256": "1" * 64, "source_sha256": "2" * 64,
                       "checkpoint_sha256": "3" * 64, "case_id": "case03",
                       "mechanism": "optimizer_update_count", "probe": "P_optimizer_count",
                       "diagnostic_receipt_sha256": "4" * 64},
        "operation": {"correction_id": "restore_configured_ppo_epochs", "recovery_mode": "restart",
                      "runner_id": "r1_epochs_v2", "training_seed": 7, "training_iterations": 300,
                      "gpu_budget_s": 100, "source_checkpoint_consumed": False,
                      "training_binding_artifact_id": "training", "fixed_binding_artifact_id": "fixed"},
        "registry_sha256": registry_digest(), "actors": ["registered-host"],
        "limits": {"training_watchdog_s": float(r1_runner.V1.WATCHDOG["train"]),
                   "fixed_watchdog_s": float(r1_runner.V1.WATCHDOG["fixed_eval"])},
    }
    if hidden:
        protocol["limits"]["scoring"] = {"answer": True}
    protocol_ref = _write(tmp_path / "protocol.json", protocol)
    isaac = tmp_path / "isaaclab"
    isaac.mkdir()
    return {"protocol": protocol_ref, "artifact_resolver": resolver,
            "isaaclab_root": str(isaac), "output_root": str(tmp_path / "output")}


@pytest.fixture(autouse=True)
def _registered_runner_verifiers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(r1_runner.V1.WRAPPER, "verify_repository_binding", lambda _manifest: None)
    monkeypatch.setattr(r1_runner.V1.WRAPPER, "verify_runtime_binding", lambda _manifest, _root: None)
    monkeypatch.setattr(r1_runner.V1, "_verify_runtime", lambda _root, _binding: None)


def test_prepare_binds_registry_seed_budget_and_artifacts_without_launch(tmp_path: Path) -> None:
    result = backend.prepare(_payload(tmp_path))
    assert result["processes_started"] == 0
    assert result["api_calls"] == 0
    assert result["protocol_body"]["operation"]["training_iterations"] == 300


def test_build_backend_has_registered_host_operations() -> None:
    assert set(backend.build_backend()) == {
        "backend_id", "prepare", "authorize", "admit", "execute", "verify_receipt", "preflight"}


def test_prepare_rejects_hidden_scoring_fields(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="hidden/scoring"):
        backend.prepare(_payload(tmp_path, hidden=True))


def test_authorize_requires_immutable_exact_scope(tmp_path: Path) -> None:
    context = backend.prepare(_payload(tmp_path))
    request = {"request_id": "r1", "effect_key": "e1"}
    diagnostic = context["protocol_body"]["diagnostic"]
    operation = context["protocol_body"]["operation"]
    approval = _write(tmp_path / "approval.json", {
        "schema": "registered_recovery_approval_v1", "actor": "registered-host",
        "request_id": "r1", "effect_key": "e1", "protocol_sha256": context["protocol"]["sha256"],
        "registry_sha256": registry_digest(), "source_sha256": diagnostic["source_sha256"],
        "checkpoint_sha256": diagnostic["checkpoint_sha256"], "mechanism": diagnostic["mechanism"],
        "correction_id": operation["correction_id"], "recovery_mode": operation["recovery_mode"],
        "gpu_budget_s": operation["gpu_budget_s"],
    })
    proof = backend.authorize(context, request, "registered-host", approval)
    assert proof["status"] == "authorized"
    bad = dict(request, effect_key="caller-verified-true")
    with pytest.raises(ValueError, match="approval binding"):
        backend.authorize(context, bad, "registered-host", approval)


def test_full_retrain_other_than_300_is_rejected(tmp_path: Path) -> None:
    payload = _payload(tmp_path)
    path = Path(payload["protocol"]["path"])
    protocol = json.loads(path.read_text(encoding="utf-8"))
    protocol["operation"]["training_iterations"] = 299
    path.write_text(json.dumps(protocol, sort_keys=True), encoding="utf-8")
    payload["protocol"]["sha256"] = file_sha(path)
    with pytest.raises(ValueError, match="300"):
        backend.prepare(payload)


def test_lease_admission_rechecks_same_live_lease_but_historical_verify_does_not(
        tmp_path: Path) -> None:
    context = {"protocol": {"sha256": "1" * 64}, "protocol_body": {
        "registry_sha256": registry_digest(),
        "diagnostic": {"source_sha256": "2" * 64, "checkpoint_sha256": "3" * 64}}}
    request = {"request_id": "r1", "effect_key": "e1"}
    first = lambda: {"owner": "rl-triage-lineage-sources-v1", "leaseId": "lease-1"}
    admission_ref = _write(tmp_path / "admission.json", backend.admit(context, request,
                                                                       lease_reader=first))
    intent = {"context": context, "request": request, "admission_ref": admission_ref}
    assert backend._verify_admission(intent, require_live=False)["sha256"] == admission_ref["sha256"]
    changed = lambda: {"owner": "rl-triage-lineage-sources-v1", "leaseId": "lease-2"}
    with pytest.raises(ValueError, match="lease changed"):
        backend._verify_admission(intent, require_live=True, lease_reader=changed)


def test_optimizer_prepare_rejects_placeholder_binding(tmp_path: Path) -> None:
    payload = _payload(tmp_path)
    resolver_path = Path(payload["artifact_resolver"]["path"])
    resolver = json.loads(resolver_path.read_text(encoding="utf-8"))
    training_path = Path(resolver["artifacts"]["training"]["path"])
    training_path.write_text(json.dumps({"max_iterations": 300}), encoding="utf-8")
    resolver["artifacts"]["training"]["sha256"] = file_sha(training_path)
    resolver_path.write_text(json.dumps(resolver, sort_keys=True), encoding="utf-8")
    payload["artifact_resolver"]["sha256"] = file_sha(resolver_path)
    with pytest.raises(ValueError, match="optimizer training binding fields"):
        backend.prepare(payload)


def test_real_actuator_manifest_producer_is_consumed_as_resume_not_optimizer(tmp_path: Path) -> None:
    module = backend._runner_module("actuator_resume_v1")
    source = _write(tmp_path / "model_299.pt", {"fixture_checkpoint": True})
    binding = {"schema": module.SCHEMA, "source_commit_sha": "a" * 40,
               "task": module.V1.TASK, "seed": 2664,
               "num_envs": 4096, "max_iterations": 300, "resume_iteration": 299,
               "resume_checkpoint": source, "resume_env_config": {"path": str(tmp_path / "env.yaml")},
               "resume_agent_config": {"path": str(tmp_path / "agent.yaml")}}
    manifest = module.manifest_for(tmp_path / "stage", binding, tmp_path, "resume_1")
    operation = {"runner_id": "actuator_resume_v1", "training_seed": 2664, "training_iterations": 300}
    diagnostic = {"checkpoint_sha256": source["sha256"]}
    backend._verify_training_manifest(manifest, operation, diagnostic)
    assert manifest["arm"] == "corrected"
    assert manifest["resume_checkpoint_sha256"] == source["sha256"]
    manifest["arm"] = "CORRECTED"
    with pytest.raises(ValueError, match="manifest binding"):
        backend._verify_training_manifest(manifest, operation, diagnostic)


def test_execution_wall_is_required_not_defaulted_to_zero() -> None:
    with pytest.raises(ValueError, match="execution wall"):
        backend._execution_wall({"execution": {}})


def test_execute_rechecks_registered_code_before_lease_or_training(monkeypatch: pytest.MonkeyPatch) -> None:
    def changed(_anchor, _closure):
        raise ValueError("registered code changed")

    monkeypatch.setattr(backend, "_verify_source_closure", changed)
    monkeypatch.setattr(backend, "_verify_admission", lambda *_args, **_kwargs: pytest.fail("admission reached"))
    intent = {"context": {"protocol_body": {"operation": {}, "diagnostic": {},
        "source_anchor": "a" * 40, "source_closure": {"changed.py": "b" * 64}}},
        "authorization": {"status": "authorized"}, "request": {}}
    with pytest.raises(ValueError, match="registered code changed"):
        backend.execute(intent)


def test_post_metrics_are_compared_to_registered_reference_not_healthy_flag() -> None:
    reference = _actual_fixed_report()
    candidate = deepcopy(reference)
    bands = {"survival_ratio_min": 0.8, "fall_frac_max": 0.1,
             "err_xy_ratio_max": 1.2, "err_yaw_ratio_max": 1.2}
    verdict, compared, checks = backend._behavior_verdict(candidate, reference, bands)
    assert verdict == "recovered"
    assert compared["survival_ratio"] == 1.0
    assert all(checks["checks"].values())
    candidate = _actual_fixed_report("source-06")
    assert backend._behavior_verdict(candidate, reference, bands)[0] == "not_recovered"
    candidate["metrics"]["lin_vel_rmse_mps"] = reference["standstill"]["lin_vel_rmse_mps"]
    assert backend._behavior_verdict(candidate, reference, bands)[0] == "unknown"


def test_equal_stationary_policies_do_not_prove_recovery() -> None:
    stationary = _actual_fixed_report("source-06")
    bands = {"survival_ratio_min": 0.8, "fall_frac_max": 0.1,
             "err_xy_ratio_max": 1.2, "err_yaw_ratio_max": 1.2}
    verdict, comparison, checks = backend._behavior_verdict(stationary, stationary, bands)
    assert comparison["err_xy_ratio"] == 1.0
    assert verdict == "unknown"
    assert checks["qualification"]["reason_code"] == "reference_unqualified"


@pytest.mark.parametrize(("registered_budget", "expected_verdict"), [(20, "recovered"), (14, "unknown")])
@pytest.mark.parametrize("runner_id", ["r1_epochs_v2", "r1_epochs_preserving_environment_v1"])
def test_execute_constructs_stock_runner_call_and_owned_evidence(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, registered_budget: int,
        expected_verdict: str, runner_id: str) -> None:
    report = _actual_fixed_report()
    report["name"] = "r1_corrected"
    reference_path = tmp_path / "reference.json"
    reference_path.write_text(json.dumps(report), encoding="utf-8")
    reference_ref = {"path": str(reference_path), "sha256": file_sha(reference_path)}
    output = tmp_path / "output"
    calls = []
    preserving = runner_id == "r1_epochs_preserving_environment_v1"
    environment_binding = {"fixture": "approved_recipe_boundary_only"}
    valid_readback = {"arm": "CORRECTED", "before": {"learning_epochs": 5, "mini_batches": 4},
                      "after_fault": {"learning_epochs": 1, "mini_batches": 4},
                      "after": {"learning_epochs": 5, "mini_batches": 4},
                      "restoration": {"registry_value": 5, "runtime_value": 5}}

    def train(stage, isaaclab, arm, seed, run_name, binding):
        calls.append(("train", arm, seed, run_name, binding["max_iterations"]))
        stage.mkdir(parents=True)
        checkpoint = stage / "model_299.pt"
        checkpoint.write_bytes(b"checkpoint")
        _write(stage / "manifest.json", {"arm": arm, "seed": seed, "max_iterations": 300,
                                           "source_checkpoint_sha256": "3" * 64})
        _write(stage / "readback.json", valid_readback)
        _write(stage / "checkpoint.json", {"final": {"sha256": file_sha(checkpoint)}})
        execution = {"exit_code": 0, "natural_exit": True, "cleanup_verified": True, "wall_s": 10.0}
        _write(stage / "execution.json", execution)
        checkpoint_execution = {"exit_code": 0, "natural_exit": True,
                                "cleanup_verified": True, "wall_s": 1.5}
        _write(stage / "checkpoint_execution.json", checkpoint_execution)
        _write(stage / "extract_execution.json", {**checkpoint_execution, "wall_s": 0.8})
        _write(stage / "telemetry.json", {"summary": {"fixture": True}})
        (stage / "records.jsonl").write_text('{"fixture": true}\n', encoding="utf-8")
        result = {"arm": arm, "checkpoint": {"path": str(checkpoint), "sha256": file_sha(checkpoint)},
                "readback": valid_readback, "execution": execution,
                "checkpoint_execution": checkpoint_execution}
        if preserving:
            result["environment_preservation"] = {
                "binding": _write(stage / "environment-binding.json", environment_binding)}
        return result

    def fixed(stage, isaaclab, artifacts, run_name, binding):
        calls.append(("fixed", artifacts["arm"], run_name))
        stage.mkdir(parents=True)
        report_path = stage / "corrected.json"
        value = {**report, "checkpoint": {"file": "model_299.pt",
                                           "sha256": artifacts["checkpoint"]["sha256"]}}
        report_path.write_text(json.dumps(value), encoding="utf-8")
        execution = {"exit_code": 0, "natural_exit": True, "cleanup_verified": True, "wall_s": 5.0}
        _write(stage / "execution.json", execution)
        return {"report_paths": {"CORRECTED": str(report_path)}, "execution": execution}

    fake = SimpleNamespace(train=train, fixed_evaluate=fixed,
                           verify_preservation=lambda evidence, checkpoint: {
                               "boundary_fixture": True, "checkpoint_sha256": checkpoint["sha256"]},
                           V1=SimpleNamespace(_validate_readback=lambda value, _arm: value))
    monkeypatch.setattr(backend, "_runner_module", lambda _runner_id: fake)
    monkeypatch.setattr(backend, "_runtime_confirmation", lambda *_args: {
        "checks": [{"check_id": "fixture_runtime_not_observed", "outcome": "unknown", "refs": ["fixture"]}],
        "scope": "backend_callback_unit_test"})
    monkeypatch.setattr(backend, "_verify_source_closure", lambda _anchor, _closure: None)
    monkeypatch.setattr(backend.grounded_recovery, "validate_fixed_report", lambda value, _expected: value)
    context = {"protocol": {"sha256": "9" * 64}, "protocol_body": {
        "source_anchor": "a" * 40, "source_closure": {"fixture.py": "b" * 64},
        "diagnostic": {"source_sha256": "2" * 64, "checkpoint_sha256": "3" * 64,
                       "mechanism": "optimizer_update_count"},
        "operation": {"correction_id": "restore_configured_ppo_epochs", "recovery_mode": "restart",
                      "runner_id": runner_id, "training_seed": 7, "training_iterations": 300,
                      "gpu_budget_s": registered_budget, "source_checkpoint_consumed": False,
                          "training_binding_artifact_id": "training", "fixed_binding_artifact_id": "fixed"},
            "registry_sha256": registry_digest()},
        "artifacts": {"training": {"body": {"max_iterations": 300,
                              **({"environment_preservation": environment_binding} if preserving else {})}},
                      "fixed": {"body": {"task": r1_runner.TASK,
                                           "runtime_source_sha256_lf": {"runtime.py": "c" * 64},
                                           "action_scale": 0.25, "protocol_sha256_lf": "a" * 64,
                                           "reference_report": reference_ref,
                                           "bands": {"survival_ratio_min": 0.8, "fall_frac_max": 0.1,
                                                     "err_xy_ratio_max": 1.2, "err_yaw_ratio_max": 1.2}}}},
        "isaaclab_root": str(tmp_path), "output_root": str(output)}
    request = {"request_id": "r1", "effect_key": "e1", "confirmed_mechanism": "optimizer_update_count"}
    lease = lambda: {"owner": "rl-triage-lineage-sources-v1", "leaseId": "lease-1",
                     "childPid": 1, "wrapperPid": 2}
    admission_ref = _write(tmp_path / "admission.json", backend.admit(context, request,
                                                                       lease_reader=lease))
    intent = {"context": context, "request": request, "authorization": {"status": "authorized"},
              "admission_ref": admission_ref,
              "costs": {"prepare_s": 1.0, "authority_s": 2.0, "queue_s": 0.0}}
    raw = backend.execute(intent, runner=fake, lease_reader=lease)
    assert calls == [("train", "CORRECTED", 7, "r1", 300), ("fixed", "CORRECTED", "r1")]
    assert raw["gpu_s"] == 15.0
    assert raw["training_evidence"].keys() >= {"manifest", "readback", "checkpoint",
                                               "checkpoint_execution"}
    assert raw["owned_breakdown"]["checkpoint_inspection_owned_wall_s"] == 1.5
    assert raw["owned_breakdown"]["telemetry_extraction_owned_wall_s"] == 0.8
    assert raw["gpu_s"] == 15.0  # CPU checkpoint child is not GPU time.
    assert backend.verify_receipt(raw, intent)["verdict"] == expected_verdict
    if preserving:
        assert raw["environment_preservation"]["verification"]["checkpoint_sha256"] == raw["new_checkpoint"]["sha256"]
        context["artifacts"]["training"]["body"]["environment_preservation"] = {"fixture": "swapped"}
        with pytest.raises(ValueError, match="approved source recipe"):
            backend.verify_receipt(raw, intent)
        context["artifacts"]["training"]["body"]["environment_preservation"] = environment_binding
    Path(raw["training_stage"]["path"]).write_text(json.dumps({"exit_code": 0}), encoding="utf-8")
    with pytest.raises(ValueError, match="training_stage ref changed"):
        backend.verify_receipt(raw, intent)


def test_environment_evidence_cannot_be_attached_to_a_different_runner() -> None:
    with pytest.raises(ValueError, match="unregistered environment"):
        backend._verify_environment_preservation("r1_epochs_v2", SimpleNamespace(), {}, {}, {})
    with pytest.raises(ValueError, match="missing"):
        backend._verify_environment_preservation(backend._PRESERVING_RUNNER, SimpleNamespace(), None, {}, {})
