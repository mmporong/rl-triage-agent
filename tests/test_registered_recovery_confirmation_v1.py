from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from evals import optimizer_observability as OBSERVE
from evals import ppo_gradient_invariant_v2 as GRADIENT
from evals import ppo_ratio_repair_execution as RATIO
from evals import r1_repair_execution as R1
from evals import registered_recovery_confirmation_v1 as SUBJECT


def _ref(path: Path, value: object | None = None, *, raw: bytes | None = None) -> dict:
    if raw is None:
        raw = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    path.write_bytes(raw)
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest()}


def _binding(ratio: bool = False) -> dict:
    minimum = (RATIO.WRAPPER.MINIMUM_BOUND_SOURCES if ratio
               else {"bench/r1_repair_training.py", "bench/optimizer_faults.py",
                     "evals/optimizer_observability.py"})
    return {
        "source_commit_sha": "a" * 40,
        "protocol_path": "bench/protocols/frozen.json",
        "protocol_sha256_lf": "b" * 64,
        "source_sha256_lf": {name: "c" * 64 for name in sorted(minimum)},
        "runtime_source_sha256_lf": {"runtime.py": "d" * 64},
        "source_checkpoint_sha256": "e" * 64,
        "task": R1.TASK,
        "num_envs": 4096,
        "max_iterations": 300,
        "configured_learning_epochs": 5,
        "configured_mini_batches": 4,
        "runtime_updates_to_observe": 3,
    }


def _r1_manifest(tmp_path: Path) -> dict:
    return R1._manifest(_binding(), arm="CORRECTED", seed=1, run_name="corrected", mode="train",
                        records=tmp_path / "records.jsonl", readback=tmp_path / "readback.json")


def _r1_rows(manifest: dict, *, steps: int = 20) -> list[dict]:
    bound = R1.WRAPPER.observer_manifest(manifest)
    rows = []
    for index in range(3):
        before = None if index == 0 else float(index * steps)
        after = float((index + 1) * steps)
        rows.append({
            "schema": OBSERVE.SCHEMA,
            "manifest_sha256": OBSERVE.manifest_fingerprint(bound),
            "binding": OBSERVE.manifest_binding(bound),
            "update_index": index,
            "expected": {"optimizer_steps": 20, "minibatches": 20},
            "runtime_configuration": {"learning_epochs": 5, "mini_batches": 4},
            "observed": {
                "minibatches": steps, "optimizer_steps": steps,
                "adam_step_before": {"actor.weight": before},
                "adam_step_after": {"actor.weight": after},
                "adam_step_delta": {"actor.weight": None if before is None else float(steps)},
                "gradient_present_steps": {"actor.weight": steps},
                "pre_step_gradient_norm": [3.0] * steps,
                "learning_rates": [[0.01]] * steps,
                "approx_kl": None, "clip_fraction": None,
            },
            "unavailable": {
                "approx_kl": "not exposed by frozen runtime",
                "clip_fraction": "not exposed by frozen runtime",
            },
            "error": None,
        })
    return rows


def _ratio_manifest(tmp_path: Path) -> dict:
    return RATIO._manifest(_binding(True), arm="CORRECTED", seed=1, run_name="corrected", mode="train",
                           records=tmp_path / "records.jsonl", readback=tmp_path / "readback.json")


def _ratio_rows(manifest: dict, gradient: str = "normal", steps: int = 20) -> list[dict]:
    bound = RATIO.WRAPPER.observer_manifest(manifest)
    summary = {"max_abs_error": 0.0 if gradient == "normal" else 1.0,
               "mismatch_count": int(gradient != "normal"), "nonzero_expected_count": 1,
               "finite": True, "n": 1}
    return [{
        "schema": OBSERVE.SCHEMA,
        "manifest_sha256": OBSERVE.manifest_fingerprint(bound),
        "binding": OBSERVE.manifest_binding(bound),
        "update_index": index,
        "expected": {"optimizer_steps": 20, "minibatches": 20},
        "observed": {"optimizer_steps": steps, "minibatches": steps,
                     GRADIENT.PRIMITIVE: {"status": gradient, "batches": [summary] * steps}},
        "error": None,
    } for index in range(3)]


def _optimizer_evidence(tmp_path: Path, runner: str, *, gradient: str = "normal",
                        steps: int = 20) -> tuple[dict, dict]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    checkpoint = _ref(tmp_path / "model_299.pt", raw=b"new checkpoint")
    if runner == "r1_epochs_v2":
        manifest = _r1_manifest(tmp_path)
        rows = _r1_rows(manifest, steps=steps)
        readback = {"arm": "CORRECTED", "before": {"learning_epochs": 5, "mini_batches": 4},
                    "after_fault": {"learning_epochs": 1, "mini_batches": 4},
                    "restoration": {"registry_value": 5, "runtime_value": 5},
                    "after": {"learning_epochs": 5, "mini_batches": 4}}
    else:
        manifest = _ratio_manifest(tmp_path)
        rows = _ratio_rows(manifest, gradient, steps)
        mutation = {"arm": "RESTORED", "mutation": RATIO.WRAPPER.MUTATION.MUTATION,
                    "changed_ratio_assignments": 1, "active_mutation": False,
                    "restored_original": True, "installed_ast_sha256": "f" * 64,
                    "original_ast_sha256": "f" * 64, "mutated_ast_sha256": "0" * 64}
        readback = {"arm": "CORRECTED",
                    "configuration": {"learning_epochs": 5, "mini_batches": 4,
                                      "clip_param": GRADIENT.CLIP_PARAM},
                    "mutation": mutation}
    records = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode()
    evidence = {
        "manifest": _ref(tmp_path / "manifest.json", manifest),
        "readback": _ref(tmp_path / "readback.json", readback),
        "records": _ref(tmp_path / "records.jsonl", raw=records),
        "checkpoint": _ref(tmp_path / "checkpoint.json",
                           {"final": {"sha256": checkpoint["sha256"]}}),
    }
    return evidence, checkpoint


@pytest.mark.parametrize("runner", ["r1_epochs_v2", "r1_epochs_preserving_environment_v1", "ppo_ratio_v1"])
def test_optimizer_confirmation_calls_frozen_inspector_and_reports_normal(
        tmp_path: Path, runner: str, monkeypatch: pytest.MonkeyPatch) -> None:
    epoch_runner = runner in {"r1_epochs_v2", "r1_epochs_preserving_environment_v1"}
    evidence, checkpoint = _optimizer_evidence(tmp_path, "r1_epochs_v2" if epoch_runner else runner)
    called = {"value": False}
    module, name = (R1, "_inspection") if epoch_runner else (RATIO, "inspection")
    original = getattr(module, name)

    def wrapped(*args):
        called["value"] = True
        return original(*args)

    monkeypatch.setattr(module, name, wrapped)
    result = SUBJECT.verify_corrected_runtime(runner, evidence, checkpoint)

    assert called["value"] is True
    assert result["outcome"] == "normal"
    assert result["scope"] == "corrected_training_first_three_updates"
    assert result["final_checkpoint_weights_tested"] is False
    assert result["behavior_recovery_tested"] is False
    assert result["unobserved_updates_after_index"] == 2


def test_ratio_count_and_gradient_are_independent(tmp_path: Path) -> None:
    evidence, checkpoint = _optimizer_evidence(tmp_path, "ppo_ratio_v1", gradient="abnormal")
    result = SUBJECT.verify_corrected_runtime("ppo_ratio_v1", evidence, checkpoint)
    assert [(row["check_id"], row["outcome"]) for row in result["checks"]] == [
        ("optimizer_update_count", "normal"), ("ppo_surrogate_gradient", "abnormal")]
    assert result["outcome"] == "abnormal"
    assert result["behavior_recovery_tested"] is False


def test_incomplete_rows_checkpoint_tamper_and_wrong_arm_are_rejected(tmp_path: Path) -> None:
    evidence, checkpoint = _optimizer_evidence(tmp_path, "r1_epochs_v2")
    records = Path(evidence["records"]["path"])
    records.write_text(records.read_text(encoding="utf-8").splitlines()[0] + "\n", encoding="utf-8")
    evidence["records"]["sha256"] = hashlib.sha256(records.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="완결"):
        SUBJECT.verify_corrected_runtime("r1_epochs_v2", evidence, checkpoint)

    evidence, checkpoint = _optimizer_evidence(tmp_path / "cp", "r1_epochs_v2")
    Path(checkpoint["path"]).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="new checkpoint reference changed"):
        SUBJECT.verify_corrected_runtime("r1_epochs_v2", evidence, checkpoint)

    evidence, checkpoint = _optimizer_evidence(tmp_path / "different", "r1_epochs_v2")
    checkpoint_evidence_path = Path(evidence["checkpoint"]["path"])
    evidence["checkpoint"] = _ref(
        checkpoint_evidence_path, {"final": {"sha256": "0" * 64}})
    with pytest.raises(ValueError, match="CPU checkpoint evidence differs"):
        SUBJECT.verify_corrected_runtime("r1_epochs_v2", evidence, checkpoint)

    evidence, checkpoint = _optimizer_evidence(tmp_path / "arm", "r1_epochs_v2")
    manifest_path = Path(evidence["manifest"]["path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["arm"] = "R1"
    evidence["manifest"] = _ref(manifest_path, manifest)
    with pytest.raises(ValueError, match="CORRECTED"):
        SUBJECT.verify_corrected_runtime("r1_epochs_v2", evidence, checkpoint)


def _actuator_readback() -> dict:
    targets = set(SUBJECT.ACTUATOR_TRAIN.LEGACY.A.TARGETS)
    target = SUBJECT.ACTUATOR_TRAIN.LEGACY.TARGET_BY_FAULT["F2"]
    before = {name: "a" * 64 for name in targets}
    injected = dict(before)
    injected[target] = "b" * 64
    return {"schema": "correction_readback_v1", "selected_correction": target,
            "before_fault": before, "after_fault": injected, "after_correction": before,
            "changed_targets": [target], "restored_targets": [target]}


def test_actuator_readback_without_post_curve_remains_unknown(tmp_path: Path) -> None:
    checkpoint = _ref(tmp_path / "model_598.pt", raw=b"actuator checkpoint")
    manifest = {"schema": "training_correction_manifest_v1", "arm": "corrected", "fault": "F2",
                "correction_id": SUBJECT.ACTUATOR_TRAIN.CORRECTION}
    manifest_ref = _ref(tmp_path / "manifest.json", manifest)
    readback_ref = _ref(tmp_path / "correction.json", _actuator_readback())
    receipt = {"schema": "training_correction_receipt_v1", "completed": True, "arm": "corrected",
               "manifest_sha256": manifest_ref["sha256"], "correction_sha256": readback_ref["sha256"],
               "checkpoint": checkpoint}
    evidence = {"manifest": manifest_ref, "readback": readback_ref,
                "training_receipt": _ref(tmp_path / "receipt.json", receipt)}

    result = SUBJECT.verify_corrected_runtime("actuator_resume_v1", evidence, checkpoint)

    assert result["outcome"] == "unknown"
    assert result["checks"] == [{"check_id": "actuator_mapping", "outcome": "unknown",
                                  "refs": [f"sha256:{manifest_ref['sha256']}",
                                           f"sha256:{readback_ref['sha256']}",
                                           f"sha256:{evidence['training_receipt']['sha256']}"]}]
    assert result["actuator_post_mapping_observed"] is False
    assert result["behavior_recovery_tested"] is False
    assert result["scope"] == "correction_readback_without_registered_post_mapping"
    with pytest.raises(ValueError, match="evidence fields differ"):
        SUBJECT.verify_corrected_runtime("actuator_resume_v1",
                                         {**evidence, "post_mapping": manifest_ref}, checkpoint)


def test_unknown_runner_extra_evidence_and_records_sha_tamper_are_rejected(tmp_path: Path) -> None:
    evidence, checkpoint = _optimizer_evidence(tmp_path, "r1_epochs_v2")
    with pytest.raises(ValueError, match="runner_id"):
        SUBJECT.verify_corrected_runtime("ppo_surrogate_v1", evidence, checkpoint)
    with pytest.raises(ValueError, match="fields differ"):
        SUBJECT.verify_corrected_runtime("r1_epochs_v2", {**evidence, "summary": evidence["manifest"]}, checkpoint)
    Path(evidence["records"]["path"]).write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="records reference changed"):
        SUBJECT.verify_corrected_runtime("r1_epochs_v2", evidence, checkpoint)
