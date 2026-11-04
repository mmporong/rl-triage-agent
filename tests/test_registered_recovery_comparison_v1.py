import json
import subprocess
import sys
from pathlib import Path

import pytest

from evals import registered_recovery_resources_v1 as recovery_resources
from evals import run_registered_recovery_comparison_v1 as comparison
from rl_triage.investigation_session import file_sha


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


def _make_recovery_null(job_path: Path) -> None:
    job = json.loads(job_path.read_text(encoding="utf-8"))
    for arm in job["arms"]:
        arm["recovery_folder"] = None
        arm["recovery_payload"] = None
        arm["approval_ref"] = None
    job_path.write_text(json.dumps(job, sort_keys=True), encoding="utf-8")


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
