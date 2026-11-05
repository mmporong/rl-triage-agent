import json
import subprocess
import sys
from pathlib import Path

import pytest

from evals import registered_recovery_backend_v1 as recovery_backend
from evals import registered_recovery_resources_v1 as recovery_resources
from evals import run_registered_recovery_comparison_v1 as comparison
from rl_triage.investigation_session import file_sha
from tests.registered_failure_evidence_fixture_v1 import (
    namespace_fixture,
    shutdown_fixture,
)


def test_standalone_recovery_controller_help_outside_repo(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-I", "-B", str(Path(comparison.__file__).resolve()), "--help"],
        cwd=tmp_path, capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=30, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "prepare" in result.stdout and "advance" in result.stdout


def _write(path: Path, value: dict) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return {"path": str(path.resolve()), "sha256": file_sha(path)}


def _fixture(tmp_path: Path, cases: tuple[str, ...] = ("case-05",)) -> tuple[Path, Path, Path, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    closure = repo / "controller.py"
    closure.write_text("# fixed\n", encoding="utf-8")
    inputs = []
    for role in ("diagnostic_plan", "recovery_review_policy", *(f"diagnostic_protocol:{case}" for case in cases)):
        if role == "recovery_review_policy":
            path = repo / comparison.REVIEW_POLICY_RELATIVE
            path.parent.mkdir(parents=True)
            source = Path(comparison.__file__).resolve().parents[1] / comparison.REVIEW_POLICY_RELATIVE
            path.write_bytes(source.read_bytes())
        else:
            path = repo / (role.replace(":", "_") + ".json")
            path.write_text("{}", encoding="utf-8")
        inputs.append({"role": role, "path": path.relative_to(repo).as_posix(),
                       "sha256": comparison._lf_sha(path)})
    templates = {}
    commitments = {}
    for case in cases:
        templates[case] = {}
        commitments[case] = {}
        for mechanism in ("actuator", "optimizer_update_count", "optimizer_surrogate"):
            training = _write(tmp_path / "templates" / case / mechanism / "training.json",
                              {"schema": "training-template", "case_id": case, "mechanism": mechanism})
            fixed = _write(tmp_path / "templates" / case / mechanism / "fixed.json",
                           {"schema": "fixed-template", "case_id": case})
            templates[case][mechanism] = (training, fixed)
            commitments[case][mechanism] = {
                "training_binding_sha256": training["sha256"],
                "fixed_binding_sha256": fixed["sha256"],
                "source_checkpoint_sha256": "c" * 64,
            }
    plan = {
        "schema": comparison.PLAN_SCHEMA, "source_anchor": "a" * 40,
        "source_closure": {"controller.py": comparison._lf_sha(closure)},
        "methods": list(comparison.METHODS),
        "cases": [{"case_id": case, "rotation": list(comparison.METHODS)} for case in cases],
        "limits": {"max_recoveries": len(cases) * 4, "gpu_sdk_launch_cap": len(cases) * 8,
                   "cpu_checkpoint_launch_cap": len(cases) * 4,
                   "cpu_telemetry_launch_cap": len(cases) * 4,
                   "total_owned_launch_cap": len(cases) * 16,
                   "max_parent_wall_s_per_recovery": 5000, "api_cap": 0},
        "inputs": inputs, "backend": comparison.BACKEND,
        "authorization": {"actor": "registered-host",
                          "ambiguous_consumed_policy": "settle_unknown_without_reexecution"},
        "recovery_policy": {"training_iterations": 300,
                            "seeds": {case: {"resume": 2664 + index, "restart": 5664 + index}
                                      for index, case in enumerate(cases)},
                            "same_case_same_template": True,
                            "source_cost_policy": "phase_a_shared_count_once"},
        "template_commitments": commitments,
    }
    plan_path = repo / "plan.json"
    plan_ref = _write(plan_path, plan)
    out = tmp_path / "prepared"
    arms = []
    for case in cases:
        for method in comparison.METHODS:
            source = _write(tmp_path / "phase_a" / case / method / "source.json",
                            {"schema": "registered_diagnostic_source_v1", "case_id": case,
                             "checkpoint_sha256": "c" * 64})
            diagnostic = _write(tmp_path / "phase_a" / case / method / "manifest.json",
                                {"method": method, "source": source,
                                 "source_verification": {"case_id": case,
                                                         "checkpoint_sha256": "c" * 64}})
            training, fixed = templates[case]["actuator"]
            protocol = _write(tmp_path / "registrations" / case / method / "protocol.json", {
                "source_anchor": "a" * 40, "source_closure": {"controller.py": comparison._lf_sha(closure)},
                "registry_sha256": comparison.registry_digest(),
                "diagnostic": {"case_id": case, "mechanism": "actuator",
                               "checkpoint_sha256": "c" * 64},
                "operation": {"training_seed": plan["recovery_policy"]["seeds"][case]["resume"],
                              "training_iterations": 300, "recovery_mode": "resume",
                              "correction_id": "restore_motor_torque_speed",
                              "training_binding_artifact_id": "training",
                              "fixed_binding_artifact_id": "fixed"},
                "actors": ["registered-host"]})
            resolver = _write(tmp_path / "registrations" / case / method / "resolver.json",
                              {"artifacts": {"training": training, "fixed": fixed}})
            approval = _write(tmp_path / "registrations" / case / method / "approval.json", {"approved": True})
            recovery_folder = (tmp_path / "prepared-recoveries" / case / method).resolve()
            arms.append({"case_id": case, "method": method, "diagnostic_manifest": diagnostic,
                         "recovery_folder": str(recovery_folder),
                         "recovery_payload": {"protocol": protocol, "artifact_resolver": resolver,
                                              "isaaclab_root": str((tmp_path / "isaaclab").resolve()),
                                              "output_root": str((recovery_folder / "actual").resolve())},
                         "approval_ref": approval})
    job_path = tmp_path / "job.json"
    _write(job_path, {"schema": comparison.JOB_SCHEMA, "plan": plan_ref, "arms": arms})
    return repo, plan_path, job_path, out


def test_registered_job_reconstructs_prepared_manifest_without_execution(tmp_path: Path) -> None:
    repo, plan, job, out = _fixture(tmp_path)
    comparison.prepare(plan, job, out, repo_root=repo, registration_anchor="b" * 40,
                       require_committed=False)
    verified = comparison.verify_prepared_manifest(
        out, repo_root=repo, registration_anchor="b" * 40, require_committed=False)
    assert len(verified["arms"]) == 4
    assert not (out / "resources").exists()


@pytest.mark.parametrize("field", ["session_path", "recovery_folder", "recovery_payload", "approval_ref"])
def test_resealed_manifest_arm_cannot_replace_registered_job_authority(tmp_path: Path, field: str) -> None:
    repo, plan, job, out = _fixture(tmp_path)
    comparison.prepare(plan, job, out, repo_root=repo, registration_anchor="b" * 40,
                       require_committed=False)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    arm = manifest["arms"][0]
    if field in {"recovery_payload", "approval_ref"}:
        arm[field] = {"replaced": True}
    else:
        arm[field] = str(tmp_path / "replacement")
    changed = _write(out / "manifest.json", manifest)
    _write(out / "anchor.json", {"schema": comparison.RESULT_SCHEMA + "_anchor",
                                  "manifest_sha256": changed["sha256"]})
    with pytest.raises(comparison.ComparisonContractError, match="manifest authority differs"):
        comparison.verify_prepared_manifest(
            out, repo_root=repo, registration_anchor="b" * 40, require_committed=False)


def test_resealed_job_and_manifest_still_require_registered_recovery_layout(tmp_path: Path) -> None:
    repo, plan, job, out = _fixture(tmp_path)
    comparison.prepare(plan, job, out, repo_root=repo, registration_anchor="b" * 40,
                       require_committed=False)
    job_body = json.loads(job.read_text(encoding="utf-8"))
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    other = str(tmp_path / "other-prepared-authority")
    job_body["arms"][0]["recovery_folder"] = other
    manifest["arms"][0]["recovery_folder"] = other
    manifest["job"] = _write(job, job_body)
    changed = _write(out / "manifest.json", manifest)
    _write(out / "anchor.json", {"schema": comparison.RESULT_SCHEMA + "_anchor",
                                  "manifest_sha256": changed["sha256"]})
    with pytest.raises(comparison.ComparisonContractError, match="recovery folder layout differs"):
        comparison.verify_prepared_manifest(
            out, repo_root=repo, registration_anchor="b" * 40, require_committed=False)


def _upgrade_to_v2(plan_path: Path, job_path: Path, tmp_path: Path) -> dict:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    job = json.loads(job_path.read_text(encoding="utf-8"))
    first = next(
        row for row in job["arms"]
        if row["case_id"] == "case-05" and row["method"] == "fixed_sequence"
    )
    diagnostic_sha = first["diagnostic_manifest"]["sha256"]
    prior_proof = _write(tmp_path / "prior-proof.json", {"schema": "prior-proof-fixture"})
    plan["schema"] = comparison.PLAN_SCHEMA_V2
    plan["limits"]["gpu_sdk_launch_cap"] -= 1
    plan["limits"]["total_owned_launch_cap"] -= 1
    plan["continuation"] = {
        "gpu_admission_profile": "registered_desktop_shared_v2",
        "prior_training": {
            "logical_id": "case-05/fixed_sequence/case_02/CORRECTED",
            "proof_sha256": prior_proof["sha256"],
            "source_checkpoint_sha256": "c" * 64,
            "completed_checkpoint_sha256": "d" * 64,
            "seed": plan["recovery_policy"]["seeds"]["case-05"]["resume"],
            "task": "Isaac-Velocity-Flat-Unitree-Go2-v0",
            "correction_id": "restore_motor_torque_speed",
            "diagnostic_manifest_sha256": diagnostic_sha,
        },
        "prior_consumed_gpu_sdk_launches": 1,
        "prior_training_gpu_s": 515.3279999999795,
        "expected_new_gpu_sdk_launches": 29,
        "continuation_tag": "v5_fresh",
        "prior_failed_attempt_preserved": True,
    }
    plan_ref = _write(plan_path, plan)
    for row in job["arms"]:
        protocol_path = Path(row["recovery_payload"]["protocol"]["path"])
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        protocol["schema"] = "registered_recovery_protocol_v2"
        protocol["diagnostic"]["session_manifest_sha256"] = row["diagnostic_manifest"]["sha256"]
        operation = protocol["operation"]
        operation.update({
            "runner_id": "actuator_resume_v1",
            "gpu_budget_s": 3600,
            "source_checkpoint_consumed": True,
            "gpu_admission_profile": "registered_desktop_shared_v2",
            "prior_training_artifact_id": (
                "prior_training"
                if row["case_id"] == "case-05" and row["method"] == "fixed_sequence"
                else None
            ),
        })
        row["recovery_payload"]["protocol"] = _write(protocol_path, protocol)
        resolver_path = Path(row["recovery_payload"]["artifact_resolver"]["path"])
        resolver = json.loads(resolver_path.read_text(encoding="utf-8"))
        if operation["prior_training_artifact_id"] is not None:
            resolver["artifacts"]["prior_training"] = prior_proof
        row["recovery_payload"]["artifact_resolver"] = _write(resolver_path, resolver)
    job["plan"] = plan_ref
    _write(job_path, job)
    return plan


def _upgrade_to_v3(plan_path: Path, job_path: Path, tmp_path: Path) -> tuple[dict, dict]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    job = json.loads(job_path.read_text(encoding="utf-8"))
    proofs = {
        "prior_effect": _write(tmp_path / "prior-effect.json", {"schema": "effect-fixture"}),
        "prior_training": _write(tmp_path / "prior-static-training.json", {"schema": "training-fixture"}),
    }
    by_method = {row["method"]: row for row in job["arms"] if row["case_id"] == "case-05"}
    imports = []
    for method, kind, artifact_id in (
        ("fixed_sequence", "completed_effect", "prior_effect"),
        ("static_rule", "completed_training", "prior_training"),
    ):
        imports.append({
            "artifact_id": artifact_id,
            "kind": kind,
            "case_id": "case-05",
            "method": method,
            "proof_sha256": proofs[artifact_id]["sha256"],
            "prior_plan_sha256": "e" * 64,
            "source_checkpoint_sha256": "c" * 64,
            "completed_checkpoint_sha256": "d" * 64,
            "seed": plan["recovery_policy"]["seeds"]["case-05"]["resume"],
            "task": "Isaac-Velocity-Flat-Unitree-Go2-v0",
            "correction_id": "restore_motor_torque_speed",
            "diagnostic_manifest_sha256": by_method[method]["diagnostic_manifest"]["sha256"],
            "source_anchor": "f9d5fea93cce406a325f578bc6f4597cc0bc35c5",
            "registration_anchor": "bf1632ee6770e681e9e8c4c5b1df890ebaed12a6",
        })
    plan["schema"] = comparison.PLAN_SCHEMA_V3
    plan["limits"]["gpu_sdk_launch_cap"] = 29
    plan["limits"]["total_owned_launch_cap"] = 61
    plan["continuation"] = {
        "gpu_admission_profile": "registered_desktop_shared_wait_v1",
        "stage_imports": imports,
        "prior_consumed_gpu_sdk_launches": 3,
        "expected_new_gpu_sdk_launches": 27,
        "continuation_tag": "v6_fresh",
        "prior_v5_registration_anchor": "bf1632ee6770e681e9e8c4c5b1df890ebaed12a6",
        "prior_incomplete_preserved": True,
    }
    plan_ref = _write(plan_path, plan)
    for row in job["arms"]:
        row["continuation_import"] = None
        if row is by_method["fixed_sequence"]:
            row["recovery_folder"] = None
            row["recovery_payload"] = None
            row["approval_ref"] = None
            row["continuation_import"] = {
                "kind": "completed_effect", "artifact_id": "prior_effect",
                "proof": proofs["prior_effect"],
            }
            continue
        protocol_path = Path(row["recovery_payload"]["protocol"]["path"])
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        protocol["schema"] = "registered_recovery_protocol_v3"
        protocol["diagnostic"]["session_manifest_sha256"] = row["diagnostic_manifest"]["sha256"]
        operation = protocol["operation"]
        operation.update({
            "runner_id": "actuator_resume_v1", "gpu_budget_s": 3600,
            "source_checkpoint_consumed": True,
            "gpu_admission_profile": "registered_desktop_shared_wait_v1",
            "prior_training_artifact_id": None,
        })
        if row is by_method["static_rule"]:
            operation["prior_training_artifact_id"] = "prior_training"
            row["continuation_import"] = {
                "kind": "completed_training", "artifact_id": "prior_training",
                "proof": proofs["prior_training"],
            }
        row["recovery_payload"]["protocol"] = _write(protocol_path, protocol)
        resolver_path = Path(row["recovery_payload"]["artifact_resolver"]["path"])
        resolver = json.loads(resolver_path.read_text(encoding="utf-8"))
        if row is by_method["static_rule"]:
            resolver["artifacts"]["prior_training"] = proofs["prior_training"]
        row["recovery_payload"]["artifact_resolver"] = _write(resolver_path, resolver)
    job["plan"] = plan_ref
    _write(job_path, job)
    return plan, proofs


def _upgrade_to_v4(plan_path: Path, job_path: Path, tmp_path: Path) -> tuple[dict, dict]:
    plan, old_proofs = _upgrade_to_v3(plan_path, job_path, tmp_path)
    job = json.loads(job_path.read_text(encoding="utf-8"))
    proofs = {
        "prior_effect": old_proofs["prior_effect"],
        "prior_static_effect": _write(
            tmp_path / "prior-static-effect.json", {"schema": "effect-fixture"}),
    }
    plan["schema"] = comparison.PLAN_SCHEMA_V4
    plan["limits"]["gpu_sdk_launch_cap"] = 27
    plan["limits"]["total_owned_launch_cap"] = 59
    plan["continuation"].update({
        "gpu_admission_profile": "registered_desktop_shared_baseline_v1",
        "prior_consumed_gpu_sdk_launches": 5,
        "expected_new_gpu_sdk_launches": 26,
        "cumulative_gpu_sdk_launches": 32,
        "continuation_tag": "v8_resumable",
        "prior_v7_registration_anchor": "1" * 40,
    })
    del plan["continuation"]["prior_v5_registration_anchor"]
    static_import = plan["continuation"]["stage_imports"][1]
    static_import.update({
        "artifact_id": "prior_static_effect",
        "kind": "completed_effect",
        "proof_sha256": proofs["prior_static_effect"]["sha256"],
        "registration_anchor": "1" * 40,
    })
    plan_ref = _write(plan_path, plan)
    for row in job["arms"]:
        if row["case_id"] == "case-05" and row["method"] == "static_rule":
            row.update({
                "recovery_folder": None, "recovery_payload": None, "approval_ref": None,
                "continuation_import": {
                    "kind": "completed_effect", "artifact_id": "prior_static_effect",
                    "proof": proofs["prior_static_effect"],
                },
            })
            continue
        payload = row.get("recovery_payload")
        if payload is None:
            continue
        protocol_path = Path(payload["protocol"]["path"])
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        protocol["operation"]["gpu_admission_profile"] = (
            "registered_desktop_shared_baseline_v1")
        payload["protocol"] = _write(protocol_path, protocol)
    job["plan"] = plan_ref
    _write(job_path, job)
    return plan, proofs


def _upgrade_to_v5(plan_path: Path, job_path: Path, tmp_path: Path) -> tuple[dict, dict, dict]:
    plan, _old = _upgrade_to_v4(plan_path, job_path, tmp_path)
    job = json.loads(job_path.read_text(encoding="utf-8"))
    entries = {}
    proofs = {}
    for case in ("case-05", "case-06"):
        for method in comparison.METHODS:
            artifact_id = (
                "prior_effect" if (case, method) == ("case-05", "fixed_sequence")
                else "prior_static_effect" if (case, method) == ("case-05", "static_rule")
                else f"prior_{case.replace('-', '')}_{method}"
            )
            proof = _write(tmp_path / f"{artifact_id}.json", {"schema": "effect-fixture"})
            proofs[artifact_id] = proof
            source_arm = next(
                row for row in job["arms"]
                if row["case_id"] == case and row["method"] == method
            )
            diagnostic = json.loads(
                Path(source_arm["diagnostic_manifest"]["path"]).read_text(encoding="utf-8")
            )
            entries[artifact_id] = {
                "artifact_id": artifact_id, "kind": "completed_effect",
                "case_id": case, "method": method, "proof_sha256": proof["sha256"],
                "prior_plan_sha256": "1" * 64,
                "diagnostic_manifest_sha256": source_arm["diagnostic_manifest"]["sha256"],
                "source_checkpoint_sha256": diagnostic["source_verification"]["checkpoint_sha256"],
                "completed_checkpoint_sha256": "2" * 64,
                "seed": 2664 if case == "case-05" else 2665,
                "task": "Isaac-Velocity-Flat-Unitree-Go2-v0",
                "correction_id": "restore_motor_torque_speed",
                "source_anchor": "3" * 40, "registration_anchor": "4" * 40,
            }
            source_arm.update({
                "recovery_folder": None, "recovery_payload": None, "approval_ref": None,
                "continuation_import": {
                    "kind": "completed_effect", "artifact_id": artifact_id, "proof": proof,
                },
            })
    plan["schema"] = comparison.PLAN_SCHEMA_V5
    plan["limits"].update(gpu_sdk_launch_cap=15, cpu_checkpoint_launch_cap=7,
                          cpu_telemetry_launch_cap=7, total_owned_launch_cap=29)
    plan["continuation"] = {
        "gpu_admission_profile": "registered_desktop_shared_retry_v1",
        "completed_import_profile": "motor_completed_v1",
        "completed_effect_registry_sha256_lf": comparison.COMPLETED_EFFECT_REGISTRY_SHA256_LF,
        "stage_imports": [entries[key] for key in sorted(entries)],
        "prior_consumed_gpu_sdk_launches": 17,
        "expected_new_gpu_sdk_launches": 14, "new_gpu_sdk_launch_cap": 15,
        "cumulative_gpu_sdk_launches": 32, "continuation_tag": "v10_epoch_repair",
        "prior_v9_registration_anchor": "fbd0bdd929e68933fdd2d8d354f7d5447625b101",
        "prior_incomplete_preserved": True, "v9_failed_epoch_sdk_launches": 0,
    }
    for relative, role in (
        (comparison.V10_NOTE, "completed_motor_effects_preregistration"),
        (comparison.COMPLETED_EFFECT_REGISTRY_RELATIVE, "completed_motor_effect_registry"),
    ):
        plan["source_closure"][relative] = (
            comparison.COMPLETED_EFFECT_REGISTRY_SHA256_LF
            if relative == comparison.COMPLETED_EFFECT_REGISTRY_RELATIVE else "5" * 64
        )
        plan["inputs"].append({
            "role": role, "path": relative, "sha256": plan["source_closure"][relative],
        })
    for relative in comparison.CONSISTENT_OBSERVATION_SOURCES:
        plan["source_closure"].setdefault(relative, "6" * 64)
    for row in job["arms"]:
        payload = row.get("recovery_payload")
        if payload is None:
            continue
        protocol_path = Path(payload["protocol"]["path"])
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        protocol["source_closure"] = plan["source_closure"]
        protocol["operation"]["gpu_admission_profile"] = "registered_desktop_shared_retry_v1"
        payload["protocol"] = _write(protocol_path, protocol)
    ordered = []
    for case in plan["cases"]:
        for method in case["rotation"]:
            ordered.append(next(row for row in job["arms"]
                                if row["case_id"] == case["case_id"]
                                and row["method"] == method))
    plan_ref = _write(plan_path, plan)
    job["plan"] = plan_ref
    job["arms"] = ordered
    _write(job_path, job)
    registry = {
        "schema": "registered_completed_effect_whitelist_v1",
        "profile_id": "all_completed_motor_effects_before_epoch_v1",
        "completed_cases": ["case-05", "case-06"],
        "methods": list(comparison.METHODS), "entries": entries,
        "prior_consumed_gpu_sdk_launches": 17, "expected_new_gpu_sdk_launches": 14,
        "new_gpu_sdk_launch_cap": 15, "cumulative_gpu_sdk_launch_cap": 32,
        "new_recovery_arms": 7, "v9_failed_epoch_sdk_launches": 0,
        "outcome_selection": "all_completed_effects_of_first_two_cases_regardless_of_outcome",
    }
    return plan, proofs, registry


def _upgrade_to_v6(plan_path: Path, job_path: Path, tmp_path: Path) -> tuple[dict, dict, dict]:
    plan, proofs, registry = _upgrade_to_v5(plan_path, job_path, tmp_path)
    job = json.loads(job_path.read_text(encoding="utf-8"))
    plan["schema"] = comparison.PLAN_SCHEMA_V6
    plan["limits"].update(gpu_sdk_launch_cap=14, total_owned_launch_cap=28)
    plan["recovery_policy"]["seeds"]["case-07"]["restart"] = 6066
    plan["recovery_policy"]["seeds"]["case-08"]["restart"] = 6067
    continuation = plan["continuation"]
    continuation.pop("prior_v9_registration_anchor")
    continuation.pop("v9_failed_epoch_sdk_launches")
    continuation.update({
        "completed_import_profile": "motor_completed_epoch_close_retry_v1",
        "epoch_shutdown_retry_policy_sha256_lf": (
            comparison.EPOCH_SHUTDOWN_RETRY_POLICY_SHA256_LF),
        "failure_proof_sha256": comparison.EPOCH_SHUTDOWN_FAILURE_PROOF_SHA256,
        "prior_consumed_gpu_sdk_launches": 18,
        "new_gpu_sdk_launch_cap": 14,
        "continuation_tag": "v12_epoch_close_retry",
        "prior_v11_registration_anchor": "389b3ae0c85257b5548a24926584917abbc12ccb",
        "v11_failed_epoch_sdk_launches": 1,
    })
    for relative, role in (
        (comparison.V12_NOTE, "epoch_close_retry_preregistration"),
        (comparison.EPOCH_SHUTDOWN_RETRY_POLICY_RELATIVE,
         "epoch_shutdown_retry_policy"),
    ):
        sha = (comparison.EPOCH_SHUTDOWN_RETRY_POLICY_SHA256_LF
               if relative == comparison.EPOCH_SHUTDOWN_RETRY_POLICY_RELATIVE
               else "7" * 64)
        plan["source_closure"][relative] = sha
        plan["inputs"].append({"role": role, "path": relative, "sha256": sha})
    plan["source_closure"].update({
        "evals/registered_epoch_shutdown_failure_v1.py": "8" * 64,
        "tests/test_registered_epoch_shutdown_failure_v1.py": "8" * 64,
    })
    for row in job["arms"]:
        payload = row.get("recovery_payload")
        if payload is None:
            continue
        protocol_path = Path(payload["protocol"]["path"])
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        protocol["source_closure"] = plan["source_closure"]
        if row["case_id"] in {"case-07", "case-08"}:
            protocol["operation"]["training_seed"] = plan["recovery_policy"]["seeds"][
                row["case_id"]
            ][protocol["operation"]["recovery_mode"]]
        payload["protocol"] = _write(protocol_path, protocol)
    failure_proof = _write(tmp_path / "v11-failure-proof.json", {"schema": "proof-fixture"})
    job["epoch_shutdown_failure_proof"] = failure_proof
    plan_ref = _write(plan_path, plan)
    job["plan"] = plan_ref
    _write(job_path, job)
    return plan, proofs, registry


def _upgrade_to_v7(plan_path: Path, job_path: Path, tmp_path: Path) -> tuple[dict, dict, dict]:
    plan, proofs, registry = _upgrade_to_v6(plan_path, job_path, tmp_path)
    job = json.loads(job_path.read_text(encoding="utf-8"))
    plan["schema"] = comparison.PLAN_SCHEMA_V7
    plan["recovery_policy"]["seeds"]["case-07"]["restart"] = 6266
    plan["recovery_policy"]["seeds"]["case-08"]["restart"] = 6267
    plan["continuation"].update({
        "completed_import_profile": "motor_completed_epoch_namespace_retry_v1",
        "namespace_failure_context_sha256": comparison.namespace_failure.CONTEXT_SHA256,
        "continuation_tag": "v13_epoch_namespace_retry",
    })
    plan["source_closure"].update({
        comparison.V13_NOTE: "9" * 64,
        "evals/registered_epoch_namespace_failure_v1.py": "9" * 64,
        "tests/test_registered_epoch_namespace_failure_v1.py": "9" * 64,
    })
    plan["inputs"].append({
        "role": "epoch_namespace_retry_preregistration", "path": comparison.V13_NOTE,
        "sha256": "9" * 64,
    })
    for row in job["arms"]:
        payload = row.get("recovery_payload")
        if payload is None:
            continue
        protocol_path = Path(payload["protocol"]["path"])
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        protocol["source_closure"] = plan["source_closure"]
        if row["case_id"] in {"case-07", "case-08"}:
            protocol["operation"]["training_seed"] = plan["recovery_policy"]["seeds"][
                row["case_id"]][protocol["operation"]["recovery_mode"]]
        payload["protocol"] = _write(protocol_path, protocol)
    context = _write(tmp_path / "v12-namespace-context.json", {"schema": "fixture"})
    job["epoch_namespace_failure_context"] = context
    plan_ref = _write(plan_path, plan)
    job["plan"] = plan_ref
    _write(job_path, job)
    return plan, proofs, registry


def _upgrade_to_v8(plan_path: Path, job_path: Path, tmp_path: Path) -> tuple[dict, dict]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    job = json.loads(job_path.read_text(encoding="utf-8"))
    repo = plan_path.parent
    note = repo / comparison.V14_NOTE
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text("registered balanced resource remeasurement fixture\n", encoding="utf-8")
    summary = repo / comparison.V13_PUBLIC_SUMMARY
    summary.parent.mkdir(parents=True, exist_ok=True)
    public_summary = Path(comparison.__file__).resolve().parents[1] / comparison.V13_PUBLIC_SUMMARY
    summary.write_bytes(public_summary.read_bytes())
    project_root = Path(comparison.__file__).resolve().parents[1]
    retry_sources = {}
    for relative in (
        comparison.CONSISTENT_OBSERVATION_SOURCES
        | comparison.BALANCED_ADMISSION_SOURCES
        | comparison.BALANCED_RESOURCE_SUMMARY_SOURCES
    ):
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((project_root / relative).read_bytes())
        retry_sources[relative] = comparison._lf_sha(path)
    plan["schema"] = comparison.PLAN_SCHEMA_V8
    plan["limits"] = {
        "max_recoveries": 16, "gpu_sdk_launch_cap": 32,
        "cpu_checkpoint_launch_cap": 16, "cpu_telemetry_launch_cap": 16,
        "total_owned_launch_cap": 64, "max_parent_wall_s_per_recovery": 5000,
        "api_cap": 0,
    }
    plan["cases"] = [
        {"case_id": case, "rotation": list(comparison.METHODS[index:]
                                             + comparison.METHODS[:index])}
        for index, case in enumerate(("case-05", "case-06", "case-07", "case-08"))
    ]
    plan["recovery_policy"]["seeds"] = {
        "case-05": {"resume": 2664, "restart": 5664},
        "case-06": {"resume": 2665, "restart": 5665},
        "case-07": {"resume": 2666, "restart": 6466},
        "case-08": {"resume": 2667, "restart": 6467},
    }
    plan["source_closure"].update({
        comparison.V14_NOTE: comparison._lf_sha(note),
        comparison.V13_PUBLIC_SUMMARY: comparison.V13_PUBLIC_SUMMARY_SHA256,
        **retry_sources,
    })
    plan["inputs"].extend([
        {"role": "balanced_resource_remeasurement_preregistration",
         "path": comparison.V14_NOTE, "sha256": comparison._lf_sha(note)},
        {"role": "known_primary_outcomes", "path": comparison.V13_PUBLIC_SUMMARY,
         "sha256": comparison.V13_PUBLIC_SUMMARY_SHA256},
    ])
    plan["resource_remeasurement"] = {
        "profile_id": comparison.RESOURCE_REMEASUREMENT_PROFILE,
        "scope": "already_seen_development_resource_check",
        "imports_allowed": False,
        "known_primary_outcomes": {"path": comparison.V13_PUBLIC_SUMMARY,
                                   "sha256": comparison.V13_PUBLIC_SUMMARY_SHA256},
        "expected": {"planned_arms": 16, "fresh_effects": 15, "diagnostic_hold": 1},
        "accounting": {"prior_primary_sdk_launches": 32, "prior_post_sdk_launches": 1,
                       "prior_whole_study_sdk_launches": 33,
                       "new_gpu_sdk_launch_cap": 32,
                       "cumulative_gpu_sdk_launch_cap": 65},
        "no_model_requery": True,
    }
    mechanisms = {"case-05": "actuator", "case-06": "actuator",
                  "case-07": "optimizer_update_count", "case-08": "optimizer_surrogate"}
    runner = {"actuator": "actuator_resume_v1",
              "optimizer_update_count": "r1_epochs_preserving_environment_v1",
              "optimizer_surrogate": "ppo_ratio_v1"}
    mode = {"actuator": "resume", "optimizer_update_count": "restart",
            "optimizer_surrogate": "restart"}
    by_key = {(row["case_id"], row["method"]): row for row in job["arms"]}
    ordered = []
    for case_row in plan["cases"]:
        case = case_row["case_id"]
        for method in case_row["rotation"]:
            row = by_key[(case, method)]
            if (case, method) == ("case-07", "iterative_agent"):
                row.update(recovery_folder=None, recovery_payload=None, approval_ref=None)
                ordered.append(row)
                continue
            mechanism = mechanisms[case]
            protocol_path = Path(row["recovery_payload"]["protocol"]["path"])
            protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
            protocol["schema"] = "registered_recovery_protocol_v3"
            protocol["source_closure"] = plan["source_closure"]
            protocol["diagnostic"].update({
                "mechanism": mechanism,
                "session_manifest_sha256": row["diagnostic_manifest"]["sha256"],
            })
            recovery_mode = mode[mechanism]
            protocol["operation"] = {
                "correction_id": comparison.REGISTRY[mechanism]["correction_id"],
                "recovery_mode": recovery_mode, "runner_id": runner[mechanism],
                "training_seed": plan["recovery_policy"]["seeds"][case][recovery_mode],
                "training_iterations": 300, "gpu_budget_s": 3600,
                "source_checkpoint_consumed": mechanism == "actuator",
                "training_binding_artifact_id": "training",
                "fixed_binding_artifact_id": "fixed",
                "gpu_admission_profile": "registered_desktop_shared_retry_v1",
                "prior_training_artifact_id": None,
            }
            row["recovery_payload"]["protocol"] = _write(protocol_path, protocol)
            resolver_path = Path(row["recovery_payload"]["artifact_resolver"]["path"])
            training = _write(tmp_path / "templates" / case / mechanism / "training.json",
                              {"schema": "training-template", "case_id": case,
                               "mechanism": mechanism})
            fixed = _write(tmp_path / "templates" / case / mechanism / "fixed.json",
                           {"schema": "fixed-template", "case_id": case})
            plan["template_commitments"][case][mechanism].update({
                "training_binding_sha256": training["sha256"],
                "fixed_binding_sha256": fixed["sha256"],
            })
            row["recovery_payload"]["artifact_resolver"] = _write(
                resolver_path, {"artifacts": {"training": training, "fixed": fixed}})
            ordered.append(row)
    plan_ref = _write(plan_path, plan)
    job["plan"] = plan_ref
    job["arms"] = ordered
    _write(job_path, job)
    return plan, job


def _upgrade_to_v9(plan_path: Path, job_path: Path, tmp_path: Path) -> tuple[dict, dict]:
    plan, job = _upgrade_to_v8(plan_path, job_path, tmp_path)
    repo = plan_path.parent
    project_root = Path(comparison.__file__).resolve().parents[1]
    namespace_sources = {}
    for relative in comparison.BALANCED_NAMESPACE_SOURCES:
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative == comparison.V15_NOTE:
            path.write_text("registered actuator namespace fixture\n", encoding="utf-8")
        else:
            path.write_bytes((project_root / relative).read_bytes())
        namespace_sources[relative] = comparison._lf_sha(path)
    plan["schema"] = comparison.PLAN_SCHEMA_V9
    plan["resource_remeasurement"]["profile_id"] = (
        comparison.RESOURCE_REMEASUREMENT_NAMESPACED_PROFILE)
    plan["source_closure"].update(namespace_sources)
    plan["inputs"].append({
        "role": "balanced_resource_namespace_preregistration",
        "path": comparison.V15_NOTE,
        "sha256": namespace_sources[comparison.V15_NOTE],
    })
    for row in job["arms"]:
        payload = row.get("recovery_payload")
        if payload is None:
            continue
        protocol_path = Path(payload["protocol"]["path"])
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        protocol["source_closure"] = plan["source_closure"]
        payload["protocol"] = _write(protocol_path, protocol)
        if protocol["diagnostic"]["mechanism"] != "actuator":
            continue
        resolver_path = Path(payload["artifact_resolver"]["path"])
        resolver = json.loads(resolver_path.read_text(encoding="utf-8"))
        training_path = Path(resolver["artifacts"]["training"]["path"])
        training = json.loads(training_path.read_text(encoding="utf-8"))
        training.update({
            "schema": "registered_actuator_recovery_binding_v2",
            "namespace_profile": comparison.RESOURCE_REMEASUREMENT_PROFILE,
        })
        training_ref = _write(training_path, training)
        resolver["artifacts"]["training"] = training_ref
        payload["artifact_resolver"] = _write(resolver_path, resolver)
        plan["template_commitments"][row["case_id"]]["actuator"][
            "training_binding_sha256"] = training_ref["sha256"]
    plan_ref = _write(plan_path, plan)
    job["plan"] = plan_ref
    _write(job_path, job)
    return plan, job


def _upgrade_to_v10(plan_path: Path, job_path: Path, tmp_path: Path) -> tuple[dict, dict]:
    plan, job = _upgrade_to_v9(plan_path, job_path, tmp_path)
    repo = plan_path.parent
    project_root = Path(comparison.__file__).resolve().parents[1]
    v16 = repo / comparison.V16_NOTE
    v16.parent.mkdir(parents=True, exist_ok=True)
    v16.write_text("registered manifest retry fixture\n", encoding="utf-8")
    failure = repo / comparison.V15_FAILURE_SUMMARY
    failure.parent.mkdir(parents=True, exist_ok=True)
    failure.write_bytes((project_root / comparison.V15_FAILURE_SUMMARY).read_bytes())
    plan["schema"] = comparison.PLAN_SCHEMA_V10
    plan["limits"].update(gpu_sdk_launch_cap=31, total_owned_launch_cap=63)
    plan["recovery_policy"]["seeds"]["case-07"]["restart"] = 6666
    plan["recovery_policy"]["seeds"]["case-08"]["restart"] = 6667
    plan["source_closure"].update({
        comparison.V16_NOTE: comparison._lf_sha(v16),
        comparison.V15_FAILURE_SUMMARY: comparison.V15_FAILURE_SUMMARY_SHA256,
    })
    plan["inputs"].extend([
        {"role": "balanced_resource_manifest_retry_preregistration",
         "path": comparison.V16_NOTE, "sha256": comparison._lf_sha(v16)},
        {"role": "failed_resource_remeasurement", "path": comparison.V15_FAILURE_SUMMARY,
         "sha256": comparison.V15_FAILURE_SUMMARY_SHA256},
    ])
    resource = plan["resource_remeasurement"]
    resource["profile_id"] = comparison.RESOURCE_MANIFEST_RETRY_PROFILE
    resource["known_failed_remeasurement"] = {
        "path": comparison.V15_FAILURE_SUMMARY,
        "sha256": comparison.V15_FAILURE_SUMMARY_SHA256,
    }
    resource["accounting"] = {
        "prior_primary_sdk_launches": 32, "prior_post_sdk_launches": 1,
        "prior_failed_remeasurement_sdk_launches": 1,
        "prior_whole_study_sdk_launches": 34, "new_gpu_sdk_launch_cap": 31,
        "cumulative_gpu_sdk_launch_cap": 65,
    }
    for row in job["arms"]:
        payload = row.get("recovery_payload")
        if payload is None:
            continue
        protocol_path = Path(payload["protocol"]["path"])
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        protocol["source_closure"] = plan["source_closure"]
        if row["case_id"] in {"case-07", "case-08"}:
            mode = protocol["operation"]["recovery_mode"]
            protocol["operation"]["training_seed"] = plan["recovery_policy"]["seeds"][
                row["case_id"]][mode]
        payload["protocol"] = _write(protocol_path, protocol)
        if protocol["diagnostic"]["mechanism"] != "actuator":
            continue
        resolver_path = Path(payload["artifact_resolver"]["path"])
        resolver = json.loads(resolver_path.read_text(encoding="utf-8"))
        training_path = Path(resolver["artifacts"]["training"]["path"])
        training = json.loads(training_path.read_text(encoding="utf-8"))
        training["schema"] = "registered_actuator_recovery_binding_v3"
        training_ref = _write(training_path, training)
        resolver["artifacts"]["training"] = training_ref
        payload["artifact_resolver"] = _write(resolver_path, resolver)
        plan["template_commitments"][row["case_id"]]["actuator"][
            "training_binding_sha256"] = training_ref["sha256"]
    plan_ref = _write(plan_path, plan)
    job["plan"] = plan_ref
    _write(job_path, job)
    return plan, job


def _upgrade_to_v11(plan_path: Path, job_path: Path, tmp_path: Path) -> tuple[dict, dict]:
    plan, job = _upgrade_to_v10(plan_path, job_path, tmp_path)
    repo = plan_path.parent
    project_root = Path(comparison.__file__).resolve().parents[1]
    for relative in comparison.BALANCED_CONTINUATION_SOURCES:
        target = repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        source = project_root / relative
        target.write_bytes(source.read_bytes() if source.is_file() else b"fixture\n")
        plan["source_closure"][relative] = comparison._lf_sha(target)
    plan["schema"] = comparison.PLAN_SCHEMA_V11
    plan["limits"] = {
        "max_recoveries": 16, "gpu_sdk_launch_cap": 16,
        "cpu_checkpoint_launch_cap": 8, "cpu_telemetry_launch_cap": 8,
        "total_owned_launch_cap": 32, "max_parent_wall_s_per_recovery": 5000,
        "api_cap": 0,
    }
    plan["inputs"].append({
        "role": "balanced_resource_continuation_preregistration",
        "path": comparison.V17_NOTE,
        "sha256": plan["source_closure"][comparison.V17_NOTE],
    })
    plan["inputs"].append({
        "role": "prior_recovery_timing_prefix",
        "path": comparison.V16_PREFIX,
        "sha256": plan["source_closure"][comparison.V16_PREFIX],
    })
    plan.pop("resource_remeasurement")
    by_key = {(row["case_id"], row["method"]): row for row in job["arms"]}
    for row in job["arms"]:
        row["continuation_import"] = None
    import_keys = (*comparison.V17_EFFECT_KEYS, comparison.V17_TRAINING_KEY)
    imports = []
    proof_refs = {}
    for case_id, method in import_keys:
        row = by_key[(case_id, method)]
        kind = "completed_training" if (case_id, method) == comparison.V17_TRAINING_KEY else "completed_effect"
        artifact_id = ("prior_training" if kind == "completed_training"
                       else f"prior_effect_{case_id.replace('-', '_')}_{method}")
        proof_refs[artifact_id] = _write(tmp_path / "proofs" / f"{artifact_id}.json", {"fixture": True})
        imports.append({
            "artifact_id": artifact_id, "kind": kind, "case_id": case_id, "method": method,
            "proof_sha256": proof_refs[artifact_id]["sha256"], "prior_plan_sha256": "e" * 64,
            "diagnostic_manifest_sha256": row["diagnostic_manifest"]["sha256"],
            "source_checkpoint_sha256": "c" * 64, "completed_checkpoint_sha256": "d" * 64,
            "seed": 2664 if case_id == "case-05" else 2665, "task": "task",
            "correction_id": "restore_motor_torque_speed", "source_anchor": "a" * 40,
            "registration_anchor": "b" * 40,
        })
        row["continuation_import"] = {"kind": kind, "artifact_id": artifact_id,
                                      "proof": proof_refs[artifact_id]}
        if kind == "completed_effect":
            row.update(recovery_folder=None, recovery_payload=None, approval_ref=None)
    plan["continuation"] = {
        "profile_id": comparison.RESOURCE_CONTINUATION_PROFILE,
        "gpu_admission_profile": "registered_desktop_shared_retry_v1",
        "stage_imports": imports, "prior_plan_sha256": "e" * 64,
        "prior_registration_anchor": "b" * 40, "prior_source_anchor": "a" * 40,
        "prior_completed_effects": 7, "prior_completed_training": 1,
        "prior_consumed_gpu_sdk_launches": 15, "expected_new_gpu_sdk_launches": 15,
        "new_gpu_sdk_launch_cap": 16, "prior_primary_sdk_launches": 32,
        "prior_post_sdk_launches": 1, "prior_v15_failed_sdk_launches": 1,
        "prior_v16_partial_sdk_launches": 15, "prior_whole_study_sdk_launches": 49,
        "expected_whole_study_sdk_launches": 64, "cumulative_gpu_sdk_launch_cap": 65,
        "prior_incomplete_preserved": True, "no_model_requery": True,
    }
    for row in job["arms"]:
        payload = row["recovery_payload"]
        if payload is None:
            continue
        protocol_path = Path(payload["protocol"]["path"])
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        protocol["source_closure"] = plan["source_closure"]
        payload["protocol"] = _write(protocol_path, protocol)
    training_row = by_key[comparison.V17_TRAINING_KEY]
    payload = training_row["recovery_payload"]
    protocol_path = Path(payload["protocol"]["path"])
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol["schema"] = "registered_recovery_protocol_v4"
    protocol["operation"]["prior_training_artifact_id"] = "prior_training"
    payload["protocol"] = _write(protocol_path, protocol)
    resolver_path = Path(payload["artifact_resolver"]["path"])
    resolver = json.loads(resolver_path.read_text(encoding="utf-8"))
    training_path = Path(resolver["artifacts"]["training"]["path"])
    training = json.loads(training_path.read_text(encoding="utf-8"))
    training.update(schema="registered_actuator_recovery_binding_v3",
                    namespace_profile=comparison.RESOURCE_REMEASUREMENT_PROFILE)
    training_ref = _write(training_path, training)
    resolver["artifacts"]["training"] = training_ref
    resolver["artifacts"]["prior_training"] = proof_refs["prior_training"]
    payload["artifact_resolver"] = _write(resolver_path, resolver)
    plan["template_commitments"]["case-06"]["actuator"]["training_binding_sha256"] = training_ref["sha256"]
    plan_ref = _write(plan_path, plan)
    job["plan"] = plan_ref
    _write(job_path, job)
    return plan, job


def _upgrade_v11_to_r2(plan: dict, repo: Path) -> dict:
    for relative in comparison.BALANCED_CONTINUATION_R2_SOURCES:
        target = repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        source = Path(comparison.__file__).resolve().parents[1] / relative
        target.write_bytes(source.read_bytes())
        plan["source_closure"][relative] = comparison._lf_sha(target)
    plan["inputs"].extend([
        {
            "role": "balanced_resource_continuation_r2_preregistration",
            "path": comparison.V17_R2_NOTE,
            "sha256": plan["source_closure"][comparison.V17_R2_NOTE],
        },
        {
            "role": "restart_seed_nonexecution_registry",
            "path": comparison.V17_SEED_NONEXECUTION_REGISTRY,
            "sha256": plan["source_closure"][comparison.V17_SEED_NONEXECUTION_REGISTRY],
        },
    ])
    return plan


def _upgrade_v11_r2_to_r3(plan: dict, repo: Path) -> dict:
    for relative in comparison.BALANCED_CONTINUATION_SOURCES | comparison.BALANCED_CONTINUATION_R3_SOURCES:
        target = repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        source = Path(comparison.__file__).resolve().parents[1] / relative
        target.write_bytes(source.read_bytes())
        plan["source_closure"][relative] = comparison._lf_sha(target)
    plan["inputs"].append({
        "role": "balanced_resource_continuation_r3_preregistration",
        "path": comparison.V17_R3_NOTE,
        "sha256": plan["source_closure"][comparison.V17_R3_NOTE],
    })
    return plan


def test_v11_r3_plan_requires_backend_and_postprocessing_closure(tmp_path: Path) -> None:
    repo, plan_path, job_path, _out = _fixture(
        tmp_path, cases=("case-05", "case-06", "case-07", "case-08")
    )
    plan, _job = _upgrade_to_v11(plan_path, job_path, tmp_path)
    plan = _upgrade_v11_to_r2(plan, repo)
    plan = _upgrade_v11_r2_to_r3(plan, repo)
    comparison._validate_plan(plan)
    required = (
        comparison.BALANCED_CONTINUATION_BACKEND_SOURCES
        | comparison.BALANCED_CONTINUATION_POSTPROCESSING_SOURCES
    )
    for relative in required:
        incomplete = json.loads(json.dumps(plan))
        incomplete["source_closure"].pop(relative)
        with pytest.raises(
            comparison.ComparisonContractError,
            match="v11 r3 backend extension registration differs",
        ):
            comparison._validate_plan(incomplete)


def _prepared_v7_for_advance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, dict, dict[str, str], dict[str, str]]:
    repo, plan_path, job_path, out = _fixture(
        tmp_path, cases=("case-05", "case-06", "case-07", "case-08"))
    proof_ref = shutdown_fixture(tmp_path / "failure-fixture", monkeypatch)
    context_ref = namespace_fixture(tmp_path / "failure-fixture", monkeypatch)
    monkeypatch.setattr(
        comparison, "EPOCH_SHUTDOWN_FAILURE_PROOF_SHA256", proof_ref["sha256"]
    )
    plan, proofs, registry = _upgrade_to_v7(plan_path, job_path, tmp_path)
    job = json.loads(job_path.read_text(encoding="utf-8"))
    job["epoch_shutdown_failure_proof"] = proof_ref
    job["epoch_namespace_failure_context"] = context_ref
    _write(job_path, job)
    monkeypatch.setattr(comparison, "_completed_effect_registry", lambda: registry)
    monkeypatch.setattr(comparison, "load_plan", lambda *_args, **_kwargs: plan)
    prior_resource = _write(tmp_path / "old/resource.json", {"schema": "old-resource"})
    prior_admission = _write(tmp_path / "old/admission.json", {"schema": "old-admission"})
    historical_folder = tmp_path / "old/recovery"
    historical_folder.mkdir(parents=True)

    class ProofModule:
        @staticmethod
        def verify_completed_effect(proof_ref, *, expected_import, repo):
            assert proof_ref == proofs[expected_import["artifact_id"]]
            return {
                "execution_origin": "verified_prior_effect",
                "result": {"outcome": "not_recovered", "diagnostic_status": "confirmed",
                    "confirmed_mechanism": "actuator", "recovery_started": True,
                    "new_checkpoint": {"sha256": "d" * 64},
                    "fixed_report": {"sha256": "f" * 64}, "behavior_outcome": "failed",
                    "costs": {"training_s": 1.0, "fixed_s": 1.0, "api_calls": 0}},
                "prior_resource_ref": prior_resource, "prior_admission_ref": prior_admission,
                "recovery_folder": str(historical_folder.resolve()),
                "post_observation": {"behavior_outcome": "failed", "healthy_declared": False},
                "proof_ref": proof_ref,
                "historical_costs": {"simulator_sdk_launches": 2, "owned_gpu_s": 2.0,
                                     "training_wall_s": 1.0, "fixed_wall_s": 1.0},
                "new_dispatch_costs": {"sdk_launches": 0, "owned_gpu_s": None},
            }

    monkeypatch.setattr(comparison, "_continuation_module", lambda _plan: ProofModule)
    comparison.prepare(plan_path, job_path, out, repo_root=repo,
                       registration_anchor="b" * 40, require_committed=False)
    return out, plan, proof_ref, context_ref


def _make_recovery_null(job_path: Path) -> None:
    job = json.loads(job_path.read_text(encoding="utf-8"))
    for arm in job["arms"]:
        arm["recovery_folder"] = None
        arm["recovery_payload"] = None
        arm["approval_ref"] = None
    job_path.write_text(json.dumps(job, sort_keys=True), encoding="utf-8")


def _enable_retry_fixture(repo: Path, plan: dict) -> None:
    plan["continuation"].update(gpu_admission_profile="registered_desktop_shared_retry_v1",
                                continuation_tag="v9_consistent_observation")
    for relative in comparison.CONSISTENT_OBSERVATION_SOURCES:
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("retry fixture\n", encoding="utf-8")
        plan["source_closure"][relative] = comparison._lf_sha(path)
    plan["inputs"].append({"role": "consistent_gpu_observation_preregistration",
                           "path": comparison.CONSISTENT_OBSERVATION_NOTE,
                           "sha256": plan["source_closure"][comparison.CONSISTENT_OBSERVATION_NOTE]})


def _write_projected_resource(out: Path, plan_sha: str, *, phase_wall_s: float) -> None:
    admission = _write(out / "raw/admission.json", {
        "schema": "registered_recovery_lease_admission_v1", "request_id": "r1",
        "effect_key": "e1", "owner": "rl-triage-lineage-sources-v1", "lease_id": "lease-1"})
    sampler = _write(out / "raw/sampler.json", {"schema": "sampler", "sample_count": 3})
    raw = _write(out / "raw/resource.json", {
        "schema": recovery_resources.SCHEMA, "request_id": "r1", "effect_key": "e1",
        "status": "complete", "phase_wall_s": phase_wall_s, "lease_await_s": None,
        "locked_wall_s": phase_wall_s, "owned_gpu_s": None, "owned_gpu_s_complete": True,
        "simulator_sdk_launches": 0, "cpu_checkpoint_launches": 0, "cpu_telemetry_launches": 0,
        "total_owned_children": 0, "admission_ref": admission, "sampler_ref": sampler,
        "sampler_error_type": None,
        "execution_refs": {"training": None, "fixed": None, "checkpoint": None, "telemetry": None},
        "raw_receipt_ref": None, "error_type": None})
    recovery_resources.project(raw, out / "resources/case-05/fixed_sequence.json",
                               case_id="case-05", method="fixed_sequence", plan_sha256=plan_sha,
                               request_id="r1", effect_key="e1")


def test_prepare_locks_explicit_case_matrix_without_execution(tmp_path: Path) -> None:
    repo, plan, job, out = _fixture(tmp_path, cases=("case-05", "case-06", "case-07", "case-08"))
    result = comparison.prepare(plan, job, out, repo_root=repo, registration_anchor="b" * 40,
                                require_committed=False)
    assert result["total_arms"] == 16
    assert result["limits"] == {"max_recoveries": 16, "gpu_sdk_launch_cap": 32,
                                "cpu_checkpoint_launch_cap": 16,
                                "cpu_telemetry_launch_cap": 16,
                                "total_owned_launch_cap": 64,
                                "max_parent_wall_s_per_recovery": 5000, "api_cap": 0}
    assert result["api_calls"] == 0


def test_v2_plan_accepts_one_prior_training_arm_and_reduced_sdk_cap(tmp_path: Path) -> None:
    repo, plan_path, job_path, out = _fixture(
        tmp_path, cases=("case-05", "case-06", "case-07", "case-08")
    )
    plan = _upgrade_to_v2(plan_path, job_path, tmp_path)
    result = comparison.prepare(
        plan_path,
        job_path,
        out,
        repo_root=repo,
        registration_anchor="b" * 40,
        require_committed=False,
    )
    assert result["limits"]["gpu_sdk_launch_cap"] == 31
    assert result["limits"]["total_owned_launch_cap"] == 63
    assert plan["continuation"]["expected_new_gpu_sdk_launches"] == 29
    public = plan_path.read_text(encoding="utf-8")
    assert str(tmp_path) not in public


def test_v2_rejects_prior_training_on_different_diagnostic_session(tmp_path: Path) -> None:
    repo, plan_path, job_path, out = _fixture(
        tmp_path, cases=("case-05", "case-06", "case-07", "case-08")
    )
    _upgrade_to_v2(plan_path, job_path, tmp_path)
    job = json.loads(job_path.read_text(encoding="utf-8"))
    first = next(row for row in job["arms"] if row["method"] == "fixed_sequence")
    protocol_path = Path(first["recovery_payload"]["protocol"]["path"])
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol["diagnostic"]["session_manifest_sha256"] = "0" * 64
    first["recovery_payload"]["protocol"] = _write(protocol_path, protocol)
    _write(job_path, job)
    with pytest.raises(comparison.ComparisonContractError, match="prior training arm/profile binding"):
        comparison.prepare(
            plan_path,
            job_path,
            out,
            repo_root=repo,
            registration_anchor="b" * 40,
            require_committed=False,
        )


def test_v3_materializes_completed_effect_and_preserves_new_dispatch_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, plan_path, job_path, out = _fixture(
        tmp_path, cases=("case-05", "case-06", "case-07", "case-08")
    )
    plan, proofs = _upgrade_to_v3(plan_path, job_path, tmp_path)
    prior_resource = _write(tmp_path / "old/resource.json", {"schema": "old-resource"})
    prior_admission = _write(tmp_path / "old/admission.json", {"schema": "old-admission"})
    historical_folder = tmp_path / "old/recovery"
    historical_folder.mkdir(parents=True)

    class ProofModule:
        @staticmethod
        def verify_completed_effect(*_args, **_kwargs):
            return {
                "execution_origin": "verified_prior_effect",
                "result": {
                    "outcome": "not_recovered", "diagnostic_status": "confirmed",
                    "confirmed_mechanism": "actuator", "recovery_started": True,
                    "new_checkpoint": {"sha256": "d" * 64},
                    "fixed_report": {"sha256": "f" * 64},
                    "behavior_outcome": "failed",
                    "costs": {"training_s": 515.3279999999795,
                              "fixed_s": 47.07799999997951, "api_calls": 0},
                },
                "prior_resource_ref": prior_resource,
                "prior_admission_ref": prior_admission,
                "recovery_folder": str(historical_folder.resolve()),
                "post_observation": {"behavior_outcome": "failed", "healthy_declared": False},
                "proof_ref": proofs["prior_effect"],
                "historical_costs": {
                    "simulator_sdk_launches": 2,
                    "owned_gpu_s": 562.405999999959,
                    "training_wall_s": 515.3279999999795,
                    "fixed_wall_s": 47.07799999997951,
                },
                "new_dispatch_costs": {"sdk_launches": 0, "owned_gpu_s": None},
                "historical_source_validation": {"status": "verified"},
                "backend_receipt_validation": {"status": "verified"},
                "old_refs": {},
            }

    monkeypatch.setattr(comparison, "_continuation_module", lambda _plan: ProofModule)
    result = comparison.prepare(
        plan_path, job_path, out, repo_root=repo, registration_anchor="b" * 40,
        require_committed=False,
    )
    assert result["completed_arms"] == 1 and result["total_arms"] == 16
    arm = next(row for row in result["arms"] if row["method"] == "fixed_sequence")
    assert arm["execution_origin"] == "verified_prior_effect"
    assert arm["outcome"] == "not_recovered"
    assert arm["resource_cost"]["new_dispatch_costs"]["simulator_sdk_launches"] == 0
    assert arm["resource_cost"]["historical_effect_costs"]["simulator_sdk_launches"] == 2
    assert result["limits"]["gpu_sdk_launch_cap"] == 29
    assert plan["continuation"]["expected_new_gpu_sdk_launches"] == 27


@pytest.mark.parametrize("retry_observation", [False, True])
def test_v4_materializes_two_completed_effects_without_recovery_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, retry_observation: bool
) -> None:
    repo, plan_path, job_path, out = _fixture(
        tmp_path, cases=("case-05", "case-06", "case-07", "case-08"))
    plan, proofs = _upgrade_to_v4(plan_path, job_path, tmp_path)
    if retry_observation:
        _enable_retry_fixture(repo, plan)
        job = json.loads(job_path.read_text(encoding="utf-8"))
        job["plan"] = _write(plan_path, plan)
        for arm in job["arms"]:
            payload = arm.get("recovery_payload")
            if payload is not None:
                path = Path(payload["protocol"]["path"])
                protocol = json.loads(path.read_text(encoding="utf-8"))
                protocol["operation"]["gpu_admission_profile"] = "registered_desktop_shared_retry_v1"
                protocol["source_closure"] = plan["source_closure"]
                payload["protocol"] = _write(path, protocol)
        _write(job_path, job)
    prior_resource = _write(tmp_path / "old/resource.json", {"schema": "old-resource"})
    prior_admission = _write(tmp_path / "old/admission.json", {"schema": "old-admission"})
    historical_folder = tmp_path / "old/recovery"
    historical_folder.mkdir(parents=True)

    class ProofModule:
        @staticmethod
        def verify_completed_effect(proof_ref, *, expected_import, repo):
            assert proof_ref == proofs[expected_import["artifact_id"]]
            assert repo.resolve() == Path(repo).resolve()
            return {
                "execution_origin": "verified_prior_effect",
                "result": {"outcome": "not_recovered", "diagnostic_status": "confirmed",
                    "confirmed_mechanism": "actuator", "recovery_started": True,
                    "new_checkpoint": {"sha256": "d" * 64},
                    "fixed_report": {"sha256": "f" * 64}, "behavior_outcome": "failed",
                    "costs": {"training_s": 1.0, "fixed_s": 1.0, "api_calls": 0}},
                "prior_resource_ref": prior_resource, "prior_admission_ref": prior_admission,
                "recovery_folder": str(historical_folder.resolve()),
                "post_observation": {"behavior_outcome": "failed", "healthy_declared": False},
                "proof_ref": proof_ref,
                "historical_costs": {"simulator_sdk_launches": 2, "owned_gpu_s": 2.0,
                                     "training_wall_s": 1.0, "fixed_wall_s": 1.0},
                "new_dispatch_costs": {"sdk_launches": 0, "owned_gpu_s": None},
            }

    monkeypatch.setattr(comparison, "_continuation_module", lambda _plan: ProofModule)
    result = comparison.prepare(plan_path, job_path, out, repo_root=repo,
                                registration_anchor="b" * 40, require_committed=False)
    imported = [row for row in result["arms"]
                if row.get("execution_origin") == "verified_prior_effect"]
    assert [(row["case_id"], row["method"]) for row in imported] == [
        ("case-05", "fixed_sequence"), ("case-05", "static_rule")]
    assert result["limits"]["gpu_sdk_launch_cap"] == 27
    assert plan["continuation"]["expected_new_gpu_sdk_launches"] == 26
    assert plan["continuation"]["cumulative_gpu_sdk_launches"] == 32


def test_v5_requires_rotation_and_materializes_eight_completed_effects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, plan_path, job_path, out = _fixture(
        tmp_path, cases=("case-05", "case-06", "case-07", "case-08"))
    plan, proofs, registry = _upgrade_to_v5(plan_path, job_path, tmp_path)
    monkeypatch.setattr(comparison, "_completed_effect_registry", lambda: registry)
    comparison._validate_plan(plan)
    prior_resource = _write(tmp_path / "old/resource.json", {"schema": "old-resource"})
    prior_admission = _write(tmp_path / "old/admission.json", {"schema": "old-admission"})
    historical_folder = tmp_path / "old/recovery"
    historical_folder.mkdir(parents=True)

    class ProofModule:
        @staticmethod
        def verify_completed_effect(proof_ref, *, expected_import, repo):
            assert proof_ref == proofs[expected_import["artifact_id"]]
            return {
                "execution_origin": "verified_prior_effect",
                "result": {"outcome": "not_recovered", "diagnostic_status": "confirmed",
                    "confirmed_mechanism": "actuator", "recovery_started": True,
                    "new_checkpoint": {"sha256": "d" * 64},
                    "fixed_report": {"sha256": "f" * 64}, "behavior_outcome": "failed",
                    "costs": {"training_s": 1.0, "fixed_s": 1.0, "api_calls": 0}},
                "prior_resource_ref": prior_resource, "prior_admission_ref": prior_admission,
                "recovery_folder": str(historical_folder.resolve()),
                "post_observation": {"behavior_outcome": "failed", "healthy_declared": False},
                "proof_ref": proof_ref,
                "historical_costs": {"simulator_sdk_launches": 2, "owned_gpu_s": 2.0,
                                     "training_wall_s": 1.0, "fixed_wall_s": 1.0},
                "new_dispatch_costs": {"sdk_launches": 0, "owned_gpu_s": None},
            }

    monkeypatch.setattr(comparison, "_continuation_module", lambda _plan: ProofModule)
    monkeypatch.setattr(comparison, "load_plan", lambda *_args, **_kwargs: plan)
    result = comparison.prepare(plan_path, job_path, out, repo_root=repo,
                                registration_anchor="b" * 40, require_committed=False)
    imported = [row for row in result["arms"]
                if row.get("execution_origin") == "verified_prior_effect"]
    assert len(imported) == 8
    assert result["completed_arms"] == 8 and result["total_arms"] == 16
    assert result["limits"] == {"max_recoveries": 16, "gpu_sdk_launch_cap": 15,
        "cpu_checkpoint_launch_cap": 7, "cpu_telemetry_launch_cap": 7,
        "total_owned_launch_cap": 29, "max_parent_wall_s_per_recovery": 5000,
        "api_cap": 0}
    job = json.loads(job_path.read_text(encoding="utf-8"))
    assert [(row["case_id"], row["method"]) for row in job["arms"]] == [
        (case["case_id"], method) for case in plan["cases"] for method in case["rotation"]]

    reordered = json.loads(job_path.read_text(encoding="utf-8"))
    reordered["arms"][0], reordered["arms"][1] = reordered["arms"][1], reordered["arms"][0]
    with pytest.raises(comparison.ComparisonContractError, match="rotation order"):
        comparison._validate_job(reordered, reordered["plan"], plan, job_path)


def test_v5_rejects_registry_and_cap_forgery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _repo, plan_path, job_path, _out = _fixture(
        tmp_path, cases=("case-05", "case-06", "case-07", "case-08"))
    plan, _proofs, registry = _upgrade_to_v5(plan_path, job_path, tmp_path)
    monkeypatch.setattr(comparison, "_completed_effect_registry", lambda: registry)
    changed = json.loads(json.dumps(plan))
    changed["continuation"]["stage_imports"][0]["proof_sha256"] = "0" * 64
    with pytest.raises(comparison.ComparisonContractError, match="registry binding"):
        comparison._validate_plan(changed)
    changed = json.loads(json.dumps(plan))
    changed["limits"]["gpu_sdk_launch_cap"] = 14
    with pytest.raises(comparison.ComparisonContractError, match="cap differs"):
        comparison._validate_plan(changed)


def test_v6_validates_retry_accounting_seeds_rotation_and_imports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _repo, plan_path, job_path, _out = _fixture(
        tmp_path, cases=("case-05", "case-06", "case-07", "case-08"))
    plan, _proofs, registry = _upgrade_to_v6(plan_path, job_path, tmp_path)
    monkeypatch.setattr(comparison, "_completed_effect_registry", lambda: registry)
    monkeypatch.setattr(comparison.epoch_failure, "verify_failure_proof",
        lambda reference, **_kwargs: {"proof_ref": {
            "path": reference["path"],
            "sha256": comparison.EPOCH_SHUTDOWN_FAILURE_PROOF_SHA256,
        }})
    comparison._validate_plan(plan)
    job = json.loads(job_path.read_text(encoding="utf-8"))
    normalized = comparison._validate_job(job, job["plan"], plan, job_path)
    assert len(normalized) == 16
    assert len([row for row in normalized if row["continuation_import"]]) == 8
    assert plan["limits"] == {"max_recoveries": 16, "gpu_sdk_launch_cap": 14,
        "cpu_checkpoint_launch_cap": 7, "cpu_telemetry_launch_cap": 7,
        "total_owned_launch_cap": 28, "max_parent_wall_s_per_recovery": 5000,
        "api_cap": 0}
    assert plan["continuation"]["prior_consumed_gpu_sdk_launches"] == 18
    assert plan["continuation"]["expected_new_gpu_sdk_launches"] == 14
    assert plan["recovery_policy"]["seeds"]["case-07"]["restart"] == 6066
    assert plan["recovery_policy"]["seeds"]["case-08"]["restart"] == 6067
    assert [(row["case_id"], row["method"]) for row in job["arms"]] == [
        (case["case_id"], method) for case in plan["cases"] for method in case["rotation"]]


def test_v7_validates_namespace_failure_new_seeds_and_preserves_matrix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _repo, plan_path, job_path, _out = _fixture(
        tmp_path, cases=("case-05", "case-06", "case-07", "case-08"))
    plan, _proofs, registry = _upgrade_to_v7(plan_path, job_path, tmp_path)
    monkeypatch.setattr(comparison, "_completed_effect_registry", lambda: registry)
    monkeypatch.setattr(comparison.epoch_failure, "verify_failure_proof",
        lambda reference, **_kwargs: {"proof_ref": {
            "path": reference["path"],
            "sha256": comparison.EPOCH_SHUTDOWN_FAILURE_PROOF_SHA256,
        }})
    monkeypatch.setattr(comparison.namespace_failure, "verify_failure_context",
        lambda reference: {"context_ref": {
            "path": reference["path"],
            "sha256": comparison.namespace_failure.CONTEXT_SHA256,
        }, "context": {"SDK": 0}})
    comparison._validate_plan(plan)
    job = json.loads(job_path.read_text(encoding="utf-8"))
    normalized = comparison._validate_job(job, job["plan"], plan, job_path)
    assert len(normalized) == 16
    assert len([row for row in normalized if row["continuation_import"]]) == 8
    assert plan["recovery_policy"]["seeds"]["case-07"]["restart"] == 6266
    assert plan["recovery_policy"]["seeds"]["case-08"]["restart"] == 6267
    assert plan["continuation"]["prior_consumed_gpu_sdk_launches"] == 18


def test_v8_validates_balanced_fresh_matrix_without_imports(
    tmp_path: Path,
) -> None:
    _repo, plan_path, job_path, _out = _fixture(
        tmp_path, cases=("case-05", "case-06", "case-07", "case-08"))
    plan, job = _upgrade_to_v8(plan_path, job_path, tmp_path)
    comparison._validate_plan(plan)
    normalized = comparison._validate_job(job, job["plan"], plan, job_path)
    assert len(normalized) == 16
    assert sum(row["recovery_payload"] is not None for row in normalized) == 15
    assert [(row["case_id"], row["method"]) for row in normalized
            if row["recovery_payload"] is None] == [("case-07", "iterative_agent")]
    assert all("continuation_import" not in row for row in normalized)
    assert plan["limits"]["gpu_sdk_launch_cap"] == 32
    assert plan["resource_remeasurement"]["accounting"] == {
        "prior_primary_sdk_launches": 32, "prior_post_sdk_launches": 1,
        "prior_whole_study_sdk_launches": 33, "new_gpu_sdk_launch_cap": 32,
        "cumulative_gpu_sdk_launch_cap": 65,
    }
    assert comparison.BALANCED_RESOURCE_SUMMARY_SOURCES <= set(plan["source_closure"])
    assert comparison.BALANCED_ADMISSION_SOURCES <= set(plan["source_closure"])


@pytest.mark.parametrize("missing", sorted(comparison.BALANCED_ADMISSION_SOURCES))
def test_v8_production_backend_requires_every_desktop_admission_source(
    tmp_path: Path, missing: str,
) -> None:
    _repo, plan_path, job_path, _out = _fixture(
        tmp_path, cases=("case-05", "case-06", "case-07", "case-08"))
    plan, _job = _upgrade_to_v8(plan_path, job_path, tmp_path)
    protocol = {
        "schema": recovery_backend.SCHEMA_V3,
        "source_closure": plan["source_closure"],
        "operation": {
            "gpu_admission_profile": recovery_backend.RETRY_PROFILE,
            "prior_training_artifact_id": None,
            "runner_id": "actuator_resume_v1",
            "training_binding_artifact_id": "training",
            "fixed_binding_artifact_id": "fixed",
        },
    }
    assert recovery_backend._extensions(protocol) is None
    changed = json.loads(json.dumps(plan))
    changed["source_closure"].pop(missing)
    protocol["source_closure"] = changed["source_closure"]
    with pytest.raises(ValueError, match="desktop .*policy source missing"):
        recovery_backend._extensions(protocol)
    with pytest.raises(comparison.ComparisonContractError,
                       match="resource remeasurement preregistration/closure"):
        comparison._validate_plan(changed)


def test_v9_requires_namespaced_actuator_binding_and_preserves_v8_contract(
    tmp_path: Path,
) -> None:
    _repo, plan_path, job_path, _out = _fixture(
        tmp_path, cases=("case-05", "case-06", "case-07", "case-08"))
    plan, job = _upgrade_to_v9(plan_path, job_path, tmp_path)
    comparison._validate_plan(plan)
    normalized = comparison._validate_job(job, job["plan"], plan, job_path)
    assert len(normalized) == 16
    assert plan["resource_remeasurement"]["profile_id"] == (
        comparison.RESOURCE_REMEASUREMENT_NAMESPACED_PROFILE)
    assert comparison.BALANCED_NAMESPACE_SOURCES <= set(plan["source_closure"])
    actuator = next(row for row in job["arms"] if row.get("recovery_payload")
                    and row["case_id"] == "case-05")
    resolver = json.loads(Path(actuator["recovery_payload"]["artifact_resolver"]["path"])
                          .read_text(encoding="utf-8"))
    training_path = Path(resolver["artifacts"]["training"]["path"])
    training = json.loads(training_path.read_text(encoding="utf-8"))
    training.pop("namespace_profile")
    training["schema"] = "registered_actuator_recovery_binding_v1"
    changed_training = _write(training_path, training)
    resolver["artifacts"]["training"] = changed_training
    plan["template_commitments"][actuator["case_id"]]["actuator"][
        "training_binding_sha256"] = changed_training["sha256"]
    actuator["recovery_payload"]["artifact_resolver"] = _write(
        Path(actuator["recovery_payload"]["artifact_resolver"]["path"]), resolver)
    with pytest.raises(comparison.ComparisonContractError,
                       match="v9 actuator namespace binding differs"):
        comparison._validate_job(job, job["plan"], plan, job_path)


def test_v10_retries_manifest_failure_with_v3_sidecar_binding(
    tmp_path: Path,
) -> None:
    _repo, plan_path, job_path, _out = _fixture(
        tmp_path, cases=("case-05", "case-06", "case-07", "case-08"))
    plan, job = _upgrade_to_v10(plan_path, job_path, tmp_path)
    comparison._validate_plan(plan)
    normalized = comparison._validate_job(job, job["plan"], plan, job_path)
    assert len(normalized) == 16
    assert plan["limits"]["gpu_sdk_launch_cap"] == 31
    assert plan["resource_remeasurement"]["accounting"][
        "prior_failed_remeasurement_sdk_launches"] == 1
    assert plan["recovery_policy"]["seeds"]["case-07"]["restart"] == 6666
    actuator = next(row for row in job["arms"] if row.get("recovery_payload")
                    and row["case_id"] == "case-05")
    resolver_path = Path(actuator["recovery_payload"]["artifact_resolver"]["path"])
    resolver = json.loads(resolver_path.read_text(encoding="utf-8"))
    training_path = Path(resolver["artifacts"]["training"]["path"])
    training = json.loads(training_path.read_text(encoding="utf-8"))
    training["schema"] = "registered_actuator_recovery_binding_v2"
    training_ref = _write(training_path, training)
    resolver["artifacts"]["training"] = training_ref
    actuator["recovery_payload"]["artifact_resolver"] = _write(resolver_path, resolver)
    plan["template_commitments"]["case-05"]["actuator"][
        "training_binding_sha256"] = training_ref["sha256"]
    with pytest.raises(comparison.ComparisonContractError,
                       match="v10 actuator namespace binding differs"):
        comparison._validate_job(job, job["plan"], plan, job_path)


@pytest.mark.parametrize("mutation,match", [
    ("failure_sha", "resource remeasurement contract"),
    ("seed", "v10 registered restart seed map"),
    ("cap", "recovery/GPU/API cap differs"),
    ("note", "manifest retry preregistration/closure"),
])
def test_v10_rejects_changed_failure_authority_seed_cap_or_note(
    tmp_path: Path, mutation: str, match: str,
) -> None:
    _repo, plan_path, job_path, _out = _fixture(
        tmp_path, cases=("case-05", "case-06", "case-07", "case-08"))
    plan, _job = _upgrade_to_v10(plan_path, job_path, tmp_path)
    if mutation == "failure_sha":
        plan["resource_remeasurement"]["known_failed_remeasurement"]["sha256"] = "0" * 64
    elif mutation == "seed":
        plan["recovery_policy"]["seeds"]["case-07"]["restart"] = 6466
    elif mutation == "cap":
        plan["limits"]["gpu_sdk_launch_cap"] = 32
    else:
        plan["source_closure"].pop(comparison.V16_NOTE)
    with pytest.raises(comparison.ComparisonContractError, match=match):
        comparison._validate_plan(plan)


def test_v11_accepts_seven_effects_one_training_and_remaining_matrix(tmp_path: Path) -> None:
    _repo, plan_path, job_path, _out = _fixture(
        tmp_path, cases=("case-05", "case-06", "case-07", "case-08"))
    plan, job = _upgrade_to_v11(plan_path, job_path, tmp_path)
    comparison._validate_plan(plan)
    normalized = comparison._validate_job(job, job["plan"], plan, job_path)
    assert len(normalized) == 16
    assert plan["limits"] == {
        "max_recoveries": 16, "gpu_sdk_launch_cap": 16,
        "cpu_checkpoint_launch_cap": 8, "cpu_telemetry_launch_cap": 8,
        "total_owned_launch_cap": 32, "max_parent_wall_s_per_recovery": 5000,
        "api_cap": 0,
    }
    assert plan["source_closure"][comparison.V16_PREFIX] == next(
        row["sha256"] for row in plan["inputs"]
        if row["role"] == "prior_recovery_timing_prefix"
    )
    imported = [row for row in normalized if row["continuation_import"] is not None]
    assert [row["continuation_import"]["kind"] for row in imported].count("completed_effect") == 7
    assert [row["continuation_import"]["kind"] for row in imported].count("completed_training") == 1
    active = [row for row in normalized if row["recovery_payload"] is not None]
    assert len(active) == 8
    assert next(row for row in active if (row["case_id"], row["method"])
                == comparison.V17_TRAINING_KEY)["continuation_import"]["kind"] == "completed_training"
    schemas = [
        json.loads(Path(row["recovery_payload"]["protocol"]["path"]).read_text())["schema"]
        for row in active
    ]
    assert schemas.count("registered_recovery_protocol_v4") == 1
    assert schemas.count("registered_recovery_protocol_v3") == 7


def test_v11_r2_requires_seed_nonexecution_note_registry_and_source_closure(
    tmp_path: Path,
) -> None:
    _repo, plan_path, job_path, _out = _fixture(
        tmp_path, cases=("case-05", "case-06", "case-07", "case-08"))
    plan, _job = _upgrade_to_v11(plan_path, job_path, tmp_path)
    plan = _upgrade_v11_to_r2(plan, plan_path.parent)
    comparison._validate_plan(plan)
    for missing in comparison.BALANCED_CONTINUATION_R2_SOURCES:
        changed = json.loads(json.dumps(plan))
        changed["source_closure"].pop(missing)
        with pytest.raises(comparison.ComparisonContractError, match="r2 seed"):
            comparison._validate_plan(changed)
    changed = json.loads(json.dumps(plan))
    changed["inputs"] = [
        row for row in changed["inputs"]
        if row["role"] != "restart_seed_nonexecution_registry"
    ]
    with pytest.raises(comparison.ComparisonContractError, match="r2 seed"):
        comparison._validate_plan(changed)


@pytest.mark.parametrize("mutation,match", [
    ("missing_import", "accounting/profile"),
    ("wrong_kind", "stage import"),
    ("wrong_prior", "accounting/profile"),
    ("wrong_seed", "registered restart seed"),
    ("wrong_cap", "recovery/GPU/API cap"),
    ("wrong_order", "rotation order"),
    ("missing_prefix", "preregistration/closure"),
])
def test_v11_rejects_changed_import_authority_or_remaining_contract(
    tmp_path: Path, mutation: str, match: str,
) -> None:
    _repo, plan_path, job_path, _out = _fixture(
        tmp_path, cases=("case-05", "case-06", "case-07", "case-08"))
    plan, job = _upgrade_to_v11(plan_path, job_path, tmp_path)
    if mutation == "missing_import":
        plan["continuation"]["stage_imports"].pop()
    elif mutation == "wrong_kind":
        plan["continuation"]["stage_imports"][0]["kind"] = "completed_training"
    elif mutation == "wrong_prior":
        plan["continuation"]["prior_v16_partial_sdk_launches"] = 14
    elif mutation == "wrong_seed":
        plan["recovery_policy"]["seeds"]["case-07"]["restart"] = 6466
    elif mutation == "wrong_cap":
        plan["limits"]["gpu_sdk_launch_cap"] = 15
    elif mutation == "missing_prefix":
        plan["inputs"] = [
            row for row in plan["inputs"]
            if row["role"] != "prior_recovery_timing_prefix"
        ]
    else:
        job["arms"][0], job["arms"][1] = job["arms"][1], job["arms"][0]
    with pytest.raises(comparison.ComparisonContractError, match=match):
        if mutation == "wrong_order":
            comparison._validate_job(job, job["plan"], plan, job_path)
        else:
            comparison._validate_plan(plan)


@pytest.mark.parametrize("target_schema", [
    "fresh_as_v4",
    "prior_as_v3",
])
def test_v11_protocol_v4_is_reserved_for_the_one_prior_training_arm(
    tmp_path: Path, target_schema: str,
) -> None:
    _repo, plan_path, job_path, _out = _fixture(
        tmp_path, cases=("case-05", "case-06", "case-07", "case-08"))
    plan, job = _upgrade_to_v11(plan_path, job_path, tmp_path)
    if target_schema == "prior_as_v3":
        row = next(row for row in job["arms"]
                   if (row["case_id"], row["method"]) == comparison.V17_TRAINING_KEY)
        replacement = "registered_recovery_protocol_v3"
    else:
        row = next(row for row in job["arms"]
                   if row["recovery_payload"] is not None
                   and row["continuation_import"] is None)
        replacement = "registered_recovery_protocol_v4"
    path = Path(row["recovery_payload"]["protocol"]["path"])
    protocol = json.loads(path.read_text(encoding="utf-8"))
    protocol["schema"] = replacement
    row["recovery_payload"]["protocol"] = _write(path, protocol)
    with pytest.raises(comparison.ComparisonContractError, match="per-arm recovery protocol"):
        comparison._validate_job(job, job["plan"], plan, job_path)


@pytest.mark.parametrize("mutation,match", [
    ("order", "rotation order"),
    ("import", "job arm fields"),
    ("prior_authority", "v8 fresh arm/profile"),
    ("wrong_seed", "registered restart seed map"),
    ("wrong_cap", "recovery/GPU/API cap differs"),
    ("second_hold", "expected fresh/hold matrix"),
])
def test_v8_rejects_unbalanced_or_historical_authority(
    tmp_path: Path, mutation: str, match: str,
) -> None:
    _repo, plan_path, job_path, _out = _fixture(
        tmp_path, cases=("case-05", "case-06", "case-07", "case-08"))
    plan, job = _upgrade_to_v8(plan_path, job_path, tmp_path)
    if mutation == "order":
        job["arms"][0], job["arms"][1] = job["arms"][1], job["arms"][0]
    elif mutation == "import":
        job["arms"][0]["continuation_import"] = None
    elif mutation == "prior_authority":
        donor = next(row for row in job["arms"] if row["recovery_payload"] is not None)
        protocol_path = Path(donor["recovery_payload"]["protocol"]["path"])
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        protocol["operation"]["prior_training_artifact_id"] = "prior_training"
        donor["recovery_payload"]["protocol"] = _write(protocol_path, protocol)
    elif mutation == "wrong_seed":
        plan["recovery_policy"]["seeds"]["case-07"]["restart"] = 6266
    elif mutation == "wrong_cap":
        plan["limits"]["gpu_sdk_launch_cap"] = 31
    else:
        donor = next(row for row in job["arms"] if row["recovery_payload"] is not None)
        donor.update(recovery_folder=None, recovery_payload=None, approval_ref=None)
    with pytest.raises(comparison.ComparisonContractError, match=match):
        if mutation in {"wrong_seed", "wrong_cap"}:
            comparison._validate_plan(plan)
        else:
            comparison._validate_job(job, job["plan"], plan, job_path)


def test_v7_rejects_missing_namespace_failure_before_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, plan_path, job_path, out = _fixture(
        tmp_path, cases=("case-05", "case-06", "case-07", "case-08"))
    plan, _proofs, registry = _upgrade_to_v7(plan_path, job_path, tmp_path)
    monkeypatch.setattr(comparison, "_completed_effect_registry", lambda: registry)
    monkeypatch.setattr(comparison, "load_plan", lambda *_args, **_kwargs: plan)
    job = json.loads(job_path.read_text(encoding="utf-8"))
    job.pop("epoch_namespace_failure_context")
    _write(job_path, job)
    with pytest.raises(comparison.ComparisonContractError, match="job fields"):
        comparison.prepare(plan_path, job_path, out, repo_root=repo,
                           registration_anchor="b" * 40, require_committed=False)
    assert not out.exists()


def test_v7_advance_revalidates_canonical_manifest_before_transition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out, _plan, _proof_ref, _context_ref = _prepared_v7_for_advance(tmp_path, monkeypatch)
    calls = []
    original = comparison.verify_prepared_manifest

    def verified(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(comparison, "verify_prepared_manifest", verified)
    monkeypatch.setattr(comparison, "_verified_diagnostic_status",
        lambda _session: {"status": "confirmed", "case_id": "case-07",
                          "confirmed_mechanism": "actuator"})
    backend_calls = []
    monkeypatch.setattr(comparison.investigate, "run",
        lambda command, *_args, **_kwargs: backend_calls.append(command) or {
            "mechanism": "actuator"})
    result = comparison.advance(out)
    assert calls == [True]
    assert result["halted"] is False and result["completed_arms"] == 8
    assert backend_calls == ["recovery-prepare"]


@pytest.mark.parametrize("evidence", ["namespace_raw", "v11_raw"])
def test_v7_advance_rejects_raw_evidence_change_before_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, evidence: str
) -> None:
    out, _plan, proof_ref, context_ref = _prepared_v7_for_advance(tmp_path, monkeypatch)
    backend_calls = []
    monkeypatch.setattr(comparison.investigate, "run",
        lambda *_args, **_kwargs: backend_calls.append(True))
    if evidence == "namespace_raw":
        context = json.loads(Path(context_ref["path"]).read_text(encoding="utf-8"))
        changed = Path(context["references"]["error"]["path"]).resolve()
        original = comparison.namespace_failure._sha
        monkeypatch.setattr(comparison.namespace_failure, "_sha",
            lambda path: "0" * 64 if Path(path).resolve() == changed else original(path))
    else:
        proof = json.loads(Path(proof_ref["path"]).read_text(encoding="utf-8"))
        changed = Path(next(iter(proof["refs"].values()))["path"]).resolve()
        original = comparison.epoch_failure._sha
        monkeypatch.setattr(comparison.epoch_failure, "_sha",
            lambda path, *, lf=False: "0" * 64
            if Path(path).resolve() == changed else original(path, lf=lf))
    result = comparison.advance(out)
    assert result["halted"] is True
    assert backend_calls == []


def test_v7_advance_rejects_resealed_manifest_arm_before_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out, _plan, _proof_ref, _context_ref = _prepared_v7_for_advance(tmp_path, monkeypatch)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    manifest["arms"][8]["approval_ref"] = {"path": "replaced", "sha256": "0" * 64}
    changed = _write(out / "manifest.json", manifest)
    _write(out / "anchor.json", {"schema": comparison.RESULT_SCHEMA + "_anchor",
                                  "manifest_sha256": changed["sha256"]})
    backend_calls = []
    monkeypatch.setattr(comparison.investigate, "run",
        lambda *_args, **_kwargs: backend_calls.append(True))
    result = comparison.advance(out)
    assert result["halted"] is True
    assert backend_calls == []


def test_v6_rejects_job_without_failure_proof_before_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, plan_path, job_path, out = _fixture(
        tmp_path, cases=("case-05", "case-06", "case-07", "case-08"))
    plan, _proofs, registry = _upgrade_to_v6(plan_path, job_path, tmp_path)
    monkeypatch.setattr(comparison, "_completed_effect_registry", lambda: registry)
    monkeypatch.setattr(comparison, "load_plan", lambda *_args, **_kwargs: plan)
    job = json.loads(job_path.read_text(encoding="utf-8"))
    job.pop("epoch_shutdown_failure_proof")
    _write(job_path, job)
    with pytest.raises(comparison.ComparisonContractError, match="job fields"):
        comparison.prepare(
            plan_path, job_path, out, repo_root=repo,
            registration_anchor="b" * 40, require_committed=False)
    assert not out.exists()


@pytest.mark.parametrize("mutation", ["proof", "cap", "seed", "import"])
def test_v6_rejects_failure_proof_cap_seed_and_import_forgery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    _repo, plan_path, job_path, _out = _fixture(
        tmp_path, cases=("case-05", "case-06", "case-07", "case-08"))
    plan, _proofs, registry = _upgrade_to_v6(plan_path, job_path, tmp_path)
    monkeypatch.setattr(comparison, "_completed_effect_registry", lambda: registry)
    changed = json.loads(json.dumps(plan))
    if mutation == "proof":
        changed["continuation"]["failure_proof_sha256"] = "0" * 64
    elif mutation == "cap":
        changed["limits"]["gpu_sdk_launch_cap"] = 15
    elif mutation == "seed":
        changed["recovery_policy"]["seeds"]["case-07"]["restart"] = 5866
    else:
        changed["continuation"]["stage_imports"][0]["proof_sha256"] = "0" * 64
    with pytest.raises(comparison.ComparisonContractError):
        comparison._validate_plan(changed)


@pytest.mark.parametrize("profile,tag", [
    ("registered_desktop_shared_retry_v1", "v8_resumable"),
    ("registered_desktop_shared_baseline_v1", "v9_consistent_observation"),
])
def test_v4_rejects_crossed_observation_profile_and_tag(tmp_path: Path, profile: str, tag: str):
    repo, plan_path, job_path, _out = _fixture(
        tmp_path, cases=("case-05", "case-06", "case-07", "case-08"))
    plan, _proofs = _upgrade_to_v4(plan_path, job_path, tmp_path)
    plan["continuation"].update(gpu_admission_profile=profile, continuation_tag=tag)
    _write(plan_path, plan)
    with pytest.raises(comparison.ComparisonContractError, match="v4 continuation"):
        comparison.load_plan(plan_path,
                             repo_root=repo, registration_anchor="b" * 40,
                             require_committed=False)


@pytest.mark.parametrize("missing", sorted(comparison.CONSISTENT_OBSERVATION_SOURCES) + [
    "note_role", "note_path", "note_sha",
])
def test_retry_plan_requires_note_role_and_entire_source_closure(tmp_path: Path, missing: str):
    repo, plan_path, job_path, _out = _fixture(
        tmp_path, cases=("case-05", "case-06", "case-07", "case-08"))
    plan, _proofs = _upgrade_to_v4(plan_path, job_path, tmp_path)
    _enable_retry_fixture(repo, plan)
    comparison._validate_plan(plan)
    if missing in comparison.CONSISTENT_OBSERVATION_SOURCES:
        del plan["source_closure"][missing]
    elif missing == "note_role":
        plan["inputs"].pop()
    elif missing == "note_path":
        plan["inputs"][-1]["path"] = "bench/protocols/unregistered-note.md"
    else:
        plan["inputs"][-1]["sha256"] = "0" * 64
    with pytest.raises(comparison.ComparisonContractError, match="consistent GPU observation"):
        comparison._validate_plan(plan)


def test_v3_rejects_completed_effect_with_fresh_recovery_authority(tmp_path: Path) -> None:
    repo, plan_path, job_path, out = _fixture(
        tmp_path, cases=("case-05", "case-06", "case-07", "case-08")
    )
    _plan, _proofs = _upgrade_to_v3(plan_path, job_path, tmp_path)
    job = json.loads(job_path.read_text(encoding="utf-8"))
    effect = next(row for row in job["arms"] if row["method"] == "fixed_sequence")
    donor = next(row for row in job["arms"] if row["method"] == "one_shot_llm")
    effect["recovery_folder"] = donor["recovery_folder"]
    effect["recovery_payload"] = donor["recovery_payload"]
    effect["approval_ref"] = donor["approval_ref"]
    _write(job_path, job)
    with pytest.raises(comparison.ComparisonContractError, match="must not carry recovery authority"):
        comparison.prepare(
            plan_path, job_path, out, repo_root=repo, registration_anchor="b" * 40,
            require_committed=False,
        )


def test_v3_accepts_only_registered_profile_tag_pairs(tmp_path: Path) -> None:
    _repo, plan_path, job_path, _out = _fixture(
        tmp_path, cases=("case-05", "case-06", "case-07", "case-08")
    )
    plan, _proofs = _upgrade_to_v3(plan_path, job_path, tmp_path)
    plan["continuation"]["gpu_admission_profile"] = "registered_desktop_shared_baseline_v1"
    plan["continuation"]["continuation_tag"] = "v7_shared_desktop"
    comparison._validate_plan(plan)
    plan["continuation"]["continuation_tag"] = "v6_fresh"
    with pytest.raises(comparison.ComparisonContractError, match="accounting/profile"):
        comparison._validate_plan(plan)


def test_prepare_rejects_partially_null_recovery_authority(tmp_path: Path) -> None:
    repo, plan, job_path, out = _fixture(tmp_path)
    job = json.loads(job_path.read_text(encoding="utf-8"))
    job["arms"][0]["approval_ref"] = None
    job_path.write_text(json.dumps(job, sort_keys=True), encoding="utf-8")
    with pytest.raises(comparison.ComparisonContractError, match="jointly null"):
        comparison.prepare(plan, job_path, out, repo_root=repo, registration_anchor="b" * 40,
                           require_committed=False)


def test_prepare_rejects_legacy_case_id_in_concrete_diagnostic_source(tmp_path: Path) -> None:
    repo, plan, job_path, out = _fixture(tmp_path)
    job = json.loads(job_path.read_text(encoding="utf-8"))
    manifest_path = Path(job["arms"][0]["diagnostic_manifest"]["path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_path = Path(manifest["source"]["path"])
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source["case_id"] = "case05"
    source_path.write_text(json.dumps(source, sort_keys=True), encoding="utf-8")
    manifest["source"]["sha256"] = file_sha(source_path)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    job["arms"][0]["diagnostic_manifest"]["sha256"] = file_sha(manifest_path)
    job_path.write_text(json.dumps(job, sort_keys=True), encoding="utf-8")
    with pytest.raises(comparison.ComparisonContractError, match="source case binding"):
        comparison.prepare(plan, job_path, out, repo_root=repo, registration_anchor="b" * 40,
                           require_committed=False)


def test_plan_rejects_hidden_scoring_authority(tmp_path: Path) -> None:
    repo, plan_path, _job, _out = _fixture(tmp_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["scoring"] = {"case-05": "F2"}
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(comparison.ComparisonContractError, match="hidden/scoring"):
        comparison.load_plan(plan_path, repo_root=repo, registration_anchor="b" * 40,
                             require_committed=False)


def test_controller_runs_registered_recovery_and_preserves_costs(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo, plan, job, out = _fixture(tmp_path)
    comparison.prepare(plan, job, out, repo_root=repo, registration_anchor="b" * 40,
                       require_committed=False)
    states = {}

    def run(command, session, payload, **_kwargs):
        if command == "status":
            return {"status": "confirmed", "confirmed_mechanism": "actuator"}
        key = str(session)
        if command == "recovery-prepare":
            Path(payload["folder"]).mkdir(parents=True)
            _write(Path(payload["folder"]) / "manifest.json",
                   {"recovery_context": {"output_root": payload["recovery_payload"]["output_root"]},
                    "prepare_s": 0.1})
            states[str(Path(payload["folder"]))] = None
            return {"mechanism": "actuator"}
        if command == "recovery-status":
            row = states[key]
            return {"requests": {}} if row is None else {"requests": {"r1": row}}
        if command == "recovery-propose":
            states[key] = {"request_id": "r1", "state": "pending"}
        elif command == "recovery-approve":
            states[key] = {"request_id": "r1", "state": "approved"}
        elif command == "recovery-dispatch":
            states[key] = {"request_id": "r1", "state": "done",
                           "receipt": {"verdict": "recovered", "result_sha": "a" * 64}}
            _write(Path(session) / "resources/r1.json", {"sealed_by_fixture": True})
        elif command == "recovery-post":
            return {"new_checkpoint": {"sha256": "b" * 64}, "fixed_report": {"sha256": "c" * 64},
                    "behavior_outcome": "normal", "costs": {"training_s": 10.0, "fixed_s": 5.0,
                                                               "api_calls": 0}}
        return {"status": "ok"}

    monkeypatch.setattr(comparison.investigate, "run", run)
    monkeypatch.setattr(comparison, "_verified_diagnostic_status",
                        lambda session: run("status", session, {}))
    measured = {"status": "complete", "phase_wall_s": 15.0}
    monkeypatch.setattr(comparison, "_seal_resource", lambda *_args, **_kwargs: measured)
    monkeypatch.setattr(comparison, "_resource_cost", lambda *_args, **_kwargs: measured)
    result = comparison.run_to_boundary(out)
    assert result["completed"] is True
    assert all(row["outcome"] == "recovered" for row in result["arms"])
    assert all(row["costs"] == {"training_s": 10.0, "fixed_s": 5.0, "api_calls": 0}
               for row in result["arms"])
    assert all(row["source_cost_accounting"] == "phase_a_shared_not_repeated" for row in result["arms"])


def test_prepare_accepts_producer_prepared_proposed_session_and_advance_resumes_it(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo, plan, job_path, out = _fixture(tmp_path)
    job = json.loads(job_path.read_text(encoding="utf-8"))
    first = job["arms"][0]
    recovery_folder = Path(first["recovery_folder"])
    state: dict[str, object] = {"request": None}
    calls = []

    def run(command, session, payload, **_kwargs):
        calls.append(command)
        if command == "recovery-prepare":
            Path(payload["folder"]).mkdir(parents=True)
            _write(Path(payload["folder"]) / "manifest.json",
                   {"recovery_context": {"output_root": payload["recovery_payload"]["output_root"]},
                    "prepare_s": 0.1})
            return {"mechanism": "actuator"}
        if command == "recovery-propose":
            state["request"] = {"request_id": "r1", "state": "pending"}
            return {"status": "pending"}
        if command == "recovery-status":
            return {"requests": {"r1": state["request"]}}
        if command == "recovery-approve":
            state["request"] = {"request_id": "r1", "state": "approved"}
            return {"status": "approved"}
        raise AssertionError(command)

    monkeypatch.setattr(comparison.investigate, "run", run)
    run("recovery-prepare", Path(first["diagnostic_manifest"]["path"]).parent,
        {"folder": str(recovery_folder), "recovery_payload": first["recovery_payload"]})
    run("recovery-propose", recovery_folder, {})
    producer_calls = list(calls)
    comparison.prepare(plan, job_path, out, repo_root=repo, registration_anchor="b" * 40,
                       require_committed=False)
    monkeypatch.setattr(comparison, "_verified_diagnostic_status",
                        lambda _session: {"status": "confirmed", "confirmed_mechanism": "actuator"})
    result = comparison.advance(out)
    assert producer_calls == ["recovery-prepare", "recovery-propose"]
    assert calls[2:] == ["recovery-status", "recovery-approve"]
    request = state["request"]
    assert isinstance(request, dict) and request["state"] == "approved"
    assert result["completed_arms"] == 0 and result["halted"] is False
    assert out.is_dir() and recovery_folder.is_dir()


def test_consumed_ambiguous_arm_settles_unknown_without_redispatch(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo, plan, job, out = _fixture(tmp_path)
    comparison.prepare(plan, job, out, repo_root=repo, registration_anchor="b" * 40,
                       require_committed=False)
    states = {}
    counts = {"dispatch": 0, "settle": 0}

    def run(command, session, payload, **_kwargs):
        if command == "status":
            return {"status": "confirmed", "confirmed_mechanism": "actuator"}
        key = str(session)
        if command == "recovery-prepare":
            Path(payload["folder"]).mkdir(parents=True)
            _write(Path(payload["folder"]) / "manifest.json",
                   {"recovery_context": {"output_root": payload["recovery_payload"]["output_root"]},
                    "prepare_s": 0.1})
            states[str(Path(payload["folder"]))] = None
            return {"mechanism": "actuator"}
        if command == "recovery-status":
            row = states[key]
            return {"requests": {}} if row is None else {"requests": {"r1": row}}
        if command == "recovery-propose":
            states[key] = {"request_id": "r1", "state": "pending"}
        elif command == "recovery-approve":
            states[key] = {"request_id": "r1", "state": "approved"}
        elif command == "recovery-dispatch":
            counts["dispatch"] += 1
            states[key] = {"request_id": "r1", "state": "running"}
        elif command == "recovery-settle":
            counts["settle"] += 1
            states[key] = {"request_id": "r1", "state": "done",
                           "receipt": {"verdict": "unknown", "result_sha": None,
                                       "reason": "orphan_without_artifact"}}
        return {"status": "ok"}

    monkeypatch.setattr(comparison.investigate, "run", run)
    monkeypatch.setattr(comparison, "_verified_diagnostic_status",
                        lambda session: run("status", session, {}))
    for _ in range(7):
        comparison.advance(out)
        if comparison.status(out)["completed_arms"]:
            break
    assert counts == {"dispatch": 1, "settle": 1}
    first = comparison.status(out)["arms"][0]
    assert first["outcome"] == "unknown"
    assert first["costs"]["status"] == "unknown_partial_cost"


def test_normal_hold_unsupported_and_pending_are_preserved_without_recovery(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo, plan, job, out = _fixture(tmp_path)
    _make_recovery_null(job)
    comparison.prepare(plan, job, out, repo_root=repo, registration_anchor="b" * 40,
                       require_committed=False)
    called = []

    def run(command, session, _payload, **_kwargs):
        called.append(command)
        method = Path(session).name
        states = {"fixed_sequence": ("host_no_intervention", None),
                  "static_rule": ("unidentifiable", None),
                  "one_shot_llm": ("confirmed", "reward"),
                  "iterative_agent": ("pending_result", None)}
        status, mechanism = states[method]
        return {"status": status, "confirmed_mechanism": mechanism}

    monkeypatch.setattr(comparison.investigate, "run", run)
    monkeypatch.setattr(comparison, "_verified_diagnostic_status",
                        lambda session: run("status", session, {}))
    result = comparison.status(out)
    for _ in range(4):
        result = comparison.advance(out)
    assert [row["outcome"] for row in result["arms"][:3]] == [
        "normal_or_host_gate", "hold", "unsupported"]
    assert result["resume"]["reason"] == "diagnostic_not_terminal"
    assert set(called) == {"status"}


def test_confirmed_supported_null_arm_remains_incomplete(tmp_path: Path,
                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    repo, plan, job, out = _fixture(tmp_path)
    _make_recovery_null(job)
    comparison.prepare(plan, job, out, repo_root=repo, registration_anchor="b" * 40,
                       require_committed=False)
    monkeypatch.setattr(comparison, "_verified_diagnostic_status",
                        lambda _session: {"status": "confirmed", "confirmed_mechanism": "actuator"})
    result = comparison.advance(out)
    assert result["completed_arms"] == 0
    assert result["resume"] == {"required": True, "reason": "recovery_artifacts_not_prepared",
                                "case_id": "case-05", "method": "fixed_sequence",
                                "confirmed_mechanism": "actuator"}


def test_unconfirmed_nonnull_arm_is_rejected_as_authority_confusion(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo, plan, job, out = _fixture(tmp_path)
    comparison.prepare(plan, job, out, repo_root=repo, registration_anchor="b" * 40,
                       require_committed=False)
    monkeypatch.setattr(comparison, "_verified_diagnostic_status",
                        lambda _session: {"status": "unidentifiable", "confirmed_mechanism": None})
    result = comparison.advance(out)
    assert result["halted"] is True
    assert "must not carry recovery authority" in result["halt"]["detail"]


def test_phase_a_status_revalidates_actual_registered_source(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    class Session:
        def __init__(self, path):
            self.manifest = {"execution_class": "actual"}
            calls.append(("session", path))

        @staticmethod
        def _verify_operation(verifier):
            calls.append(("verify", verifier))

        @staticmethod
        def status():
            calls.append(("status", None))
            return {"status": "confirmed"}

    monkeypatch.setattr(comparison, "InvestigationSession", Session)
    result = comparison._verified_diagnostic_status(tmp_path)
    assert result["status"] == "confirmed"
    assert calls[0] == ("session", tmp_path)
    assert calls[1][0] == "verify"
    assert calls[2][0] == "status"


def test_infrastructure_failure_halts_remaining_arms_without_retry(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo, plan, job, out = _fixture(tmp_path)
    comparison.prepare(plan, job, out, repo_root=repo, registration_anchor="b" * 40,
                       require_committed=False)
    calls = {"count": 0}

    def fail(_session):
        calls["count"] += 1
        raise RuntimeError("source runtime mismatch")

    monkeypatch.setattr(comparison, "_verified_diagnostic_status", fail)
    first = comparison.advance(out)
    second = comparison.advance(out)
    assert first["halted"] is second["halted"] is True
    assert first["halt"]["detail"] == "source runtime mismatch"
    assert first["completed_arms"] == 0 and first["resume"]["required"] is False
    assert calls["count"] == 1


def test_status_consumes_bound_sampler_lease_and_phase_cost_receipt(tmp_path: Path) -> None:
    repo, plan, job, out = _fixture(tmp_path)
    comparison.prepare(plan, job, out, repo_root=repo, registration_anchor="b" * 40,
                       require_committed=False)
    plan_sha = json.loads((out / "manifest.json").read_text(encoding="utf-8"))["plan"]["sha256"]
    _write_projected_resource(out, plan_sha, phase_wall_s=20.0)
    cost = comparison.status(out)["arms"][0]["resource_cost"]
    assert cost["phase_wall_s"] == 20.0 and cost["api_calls"] == 0
    assert cost["receipt_ref"]["sha256"] == file_sha(out / "resources/case-05/fixed_sequence.json")


def test_status_rejects_parent_wall_above_preregistered_cap(tmp_path: Path) -> None:
    repo, plan, job, out = _fixture(tmp_path)
    comparison.prepare(plan, job, out, repo_root=repo, registration_anchor="b" * 40,
                       require_committed=False)
    plan_sha = json.loads((out / "manifest.json").read_text(encoding="utf-8"))["plan"]["sha256"]
    _write_projected_resource(out, plan_sha, phase_wall_s=5000.001)
    with pytest.raises(comparison.ComparisonContractError, match="parent wall cap"):
        comparison.status(out)
