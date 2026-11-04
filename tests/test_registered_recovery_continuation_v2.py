from contextlib import contextmanager, nullcontext
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_registered_recovery_backend_v1 import _actual_fixed_report, _write

from evals import registered_prior_training_v1 as PRIOR
from evals import registered_recovery_backend_v1 as B
from evals import registered_recovery_resources_v1 as resources
from rl_triage import investigation_session as io


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    old = tmp_path / "original-training"
    cp = old / "model_598.pt"
    old.mkdir()
    cp.write_bytes(b"completed training checkpoint")
    cp_ref = io.reference(cp)
    readback = {"fixture": "actuator runtime boundary"}
    _write(old / "manifest.json", {"arm": "corrected", "seed": 2664, "max_iterations": 300,
                                   "resume_checkpoint_sha256": "3" * 64})
    _write(old / "correction.json", readback)
    execution = {"exit_code": 0, "natural_exit": True, "cleanup_verified": True, "wall_s": 12.0}
    old_execution = _write(old / "execution.json", execution)
    _write(old / "receipt.json", {"checkpoint": cp_ref, "resume_checkpoint": {"sha256": "3" * 64}})
    proof = _write(tmp_path / "prior-proof.json", {"fixture": "validated by separate loader"})
    prior = {"training_stage": str(old), "new_training_launched": False, "proof_ref": proof,
             "prior_owned_execution_ref": old_execution,
             "prior_dispatch_resource_ref": {"path": "historical-resource.json", "sha256": "f" * 64},
             "validation": {"full_environment_state_restored": False},
             "training_result": {"arm": "CORRECTED", "checkpoint": cp_ref,
                                 "readback": readback, "execution": execution}}
    output = tmp_path / "new-dispatch"
    report = _actual_fixed_report()
    reference = _write(tmp_path / "reference.json", report)
    calls = []

    def fixed(stage, _lab, training, _name, _binding):
        calls.append("fixed")
        assert training["checkpoint"] == cp_ref
        value = deepcopy(report)
        value["checkpoint"] = {"file": cp.name, "sha256": cp_ref["sha256"]}
        ref = _write(stage / "fixed.json", value)
        ex = {**execution, "wall_s": 5.0}
        _write(stage / "execution.json", ex)
        return {"report_paths": [ref["path"]], "execution": ex}

    runner = SimpleNamespace(train=lambda *_args: pytest.fail("completed training was launched again"),
                             fixed_evaluate=fixed, LEGACY=SimpleNamespace(validate_readback=lambda *_args: None))
    operation = {"correction_id": "restore_motor_torque_speed", "recovery_mode": "resume",
                 "runner_id": "actuator_resume_v1", "training_seed": 2664, "training_iterations": 300,
                 "gpu_budget_s": 100, "source_checkpoint_consumed": True,
                 "training_binding_artifact_id": "training", "fixed_binding_artifact_id": "fixed",
                 "gpu_admission_profile": B.DESKTOP_PROFILE, "prior_training_artifact_id": "prior"}
    context = {"protocol": {"sha256": "a" * 64}, "protocol_body": {
        "schema": B.SCHEMA_V2, "operation": operation, "registry_sha256": "b" * 64,
        "source_anchor": "c" * 40,
        "source_closure": {key: "d" * 64 for key in B._DESKTOP_SOURCES | B._PRIOR_SOURCES},
        "diagnostic": {"mechanism": "actuator", "checkpoint_sha256": "3" * 64,
                       "source_sha256": "4" * 64, "session_manifest_sha256": "5" * 64}},
        "artifacts": {"training": {"body": {"task": "registered-task"}},
                      "prior": {"ref": proof}, "fixed": {"body": {
                          "task": "registered-task", "action_scale": 0.25,
                          "protocol_sha256_lf": "a" * 64, "runtime_source_sha256_lf": {},
                          "reference_report": reference, "bands": {"survival_ratio_min": 0.8,
                            "fall_frac_max": 0.1, "err_xy_ratio_max": 1.2, "err_yaw_ratio_max": 1.2}}}},
        "output_root": str(output), "isaaclab_root": str(tmp_path)}
    request = {"request_id": "new-request", "effect_key": "new-effect", "confirmed_mechanism": "actuator"}
    lease = lambda: {"leaseId": "live-lease", "owner": "rl-triage-lineage-sources-v1"}
    admission = _write(tmp_path / "admission.json", B.admit(context, request, lease_reader=lease))
    intent = {"context": context, "request": request, "authorization": {"status": "authorized"},
              "admission_ref": admission, "costs": {"prepare_s": 1.0, "authority_s": 0.5}}

    def load(ref, **expected):
        calls.append("verify-prior")
        assert ref == proof
        assert expected["expected_source_checkpoint_sha256"] == "3" * 64
        assert expected["expected_seed"] == 2664
        assert expected["expected_task"] == "registered-task"
        assert expected["expected_diagnostic_manifest_sha256"] == "5" * 64
        return deepcopy(prior)

    monkeypatch.setattr(PRIOR, "load_completed_actuator_training", load)
    monkeypatch.setattr(B, "_runner_module", lambda _id: runner)
    monkeypatch.setattr(B, "_runtime_confirmation", lambda *_args: {"fixture": True})
    monkeypatch.setattr(B, "_verify_source_closure", lambda *_args: None)
    monkeypatch.setattr(B, "_gpu_scope", lambda *_args: nullcontext())
    monkeypatch.setattr(B.grounded_recovery, "validate_fixed_report", lambda value, _expected: value)
    return intent, runner, lease, prior, calls


def test_saved_training_is_verified_and_fixed_only_has_fresh_request_and_costs(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    intent, runner, lease, prior, calls = _fixture(tmp_path, monkeypatch)
    old = Path(prior["training_stage"])
    original = {path.name: path.read_bytes() for path in old.iterdir()}

    class Sampler:
        def __init__(self, output):
            self.output = Path(output)

        def start(self):
            return {"scope": "CPU fixture"}

        def stop(self):
            return {"receipt_ref": _write(self.output / "stop.json", {"scope": "CPU fixture"})}

    measurement = resources.Measurement(tmp_path / "resources", intent["request"],
        intent["admission_ref"], intent["context"], sampler_factory=Sampler)
    raw = B.execute(intent, runner=runner, lease_reader=lease)
    current_cost = resources.verify(measurement.finish(raw=raw, error=None),
        request_id="new-request", effect_key="new-effect")
    assert current_cost["simulator_sdk_launches"] == 1 and current_cost["owned_gpu_s"] == 5.0
    assert current_cost["execution_refs"]["training"] is None
    assert calls == ["verify-prior", "fixed"]
    assert raw["costs"]["training_s"] == 12.0 and raw["gpu_s"] == 17.0
    assert raw["training_stage"] == prior["prior_owned_execution_ref"]
    assert raw["training_reuse"]["new_training_launched"] is False
    assert not (Path(intent["context"]["output_root"]) / "training").exists()
    assert B.verify_receipt(raw, intent)["verdict"] == "recovered"
    assert calls == ["verify-prior", "fixed", "verify-prior"]
    assert {path.name: path.read_bytes() for path in old.iterdir()} == original
    artifact = {key: value for key, value in raw.items() if key != "receipt_ref"}
    artifact["training_reuse"]["new_training_launched"] = True
    new_ref = _write(tmp_path / "tampered-receipt.json", artifact)
    with pytest.raises(ValueError, match="prior training receipt differs"):
        B.verify_receipt({**artifact, "receipt_ref": new_ref}, intent)


def test_prior_integrity_failure_does_not_fall_back_to_training_or_fixed(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    intent, runner, lease, _prior, calls = _fixture(tmp_path, monkeypatch)

    def changed(*_args, **_kwargs):
        raise ValueError("prior checkpoint changed")

    monkeypatch.setattr(PRIOR, "load_completed_actuator_training", changed)
    with pytest.raises(ValueError, match="prior checkpoint changed"):
        B.execute(intent, runner=runner, lease_reader=lease)
    assert calls == []


@pytest.mark.parametrize("mutation", ["v1", "policy", "runner", "missing_source", "binding_alias"])
def test_continuation_extensions_fail_closed(mutation: str) -> None:
    operation = {"gpu_admission_profile": B.DESKTOP_PROFILE, "prior_training_artifact_id": "prior",
                 "runner_id": "actuator_resume_v1", "training_binding_artifact_id": "training",
                 "fixed_binding_artifact_id": "fixed"}
    protocol = {"schema": B.SCHEMA_V2, "operation": operation,
                "source_closure": {key: "a" * 64 for key in B._DESKTOP_SOURCES | B._PRIOR_SOURCES}}
    if mutation == "v1":
        protocol["schema"] = B.SCHEMA
    elif mutation == "policy":
        operation["gpu_admission_profile"] = "desktop_without_checks"
    elif mutation == "runner":
        operation["runner_id"] = "r1_epochs_v2"
    elif mutation == "missing_source":
        protocol["source_closure"].pop("evals/registered_prior_training_v1.py")
    else:
        operation["prior_training_artifact_id"] = "training"
    with pytest.raises(ValueError):
        B._extensions(protocol)


def test_operational_gpu_scope_has_no_fixture_callback_forwarding(monkeypatch: pytest.MonkeyPatch) -> None:
    from evals import registered_desktop_gpu_admission_v1 as admission

    calls = []

    @contextmanager
    def scope():
        calls.append("enter")
        try:
            yield
        finally:
            calls.append("exit")

    monkeypatch.setattr(admission, "snapshot_scope", scope)
    context = {"protocol_body": {"schema": B.SCHEMA_V2,
        "source_closure": {key: "a" * 64 for key in B._DESKTOP_SOURCES},
        "operation": {"gpu_admission_profile": B.DESKTOP_PROFILE, "prior_training_artifact_id": None}}}
    with pytest.raises(RuntimeError, match="fixed failure"), B._gpu_scope(context):
        raise RuntimeError("fixed failure")
    assert calls == ["enter", "exit"]
