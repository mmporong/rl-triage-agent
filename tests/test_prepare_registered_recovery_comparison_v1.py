import hashlib
import json
from pathlib import Path

import pytest

from evals import prepare_registered_recovery_comparison_v1 as prepare


def _write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, bytes):
        path.write_bytes(value)
    else:
        path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_fixed_binding_is_consumed_by_stock_runtime_verifier_before_launch(tmp_path: Path) -> None:
    from evals import r1_repair_execution as stock
    runtime_file = _write(tmp_path / "runtime.py", b"frozen runtime source\n")
    runtime = {"runtime.py": _sha(runtime_file)}
    report = {"task": stock.TASK, "action_scale": 0.25, "protocol_sha256_lf": "a" * 64}
    binding = prepare._fixed_binding({"path": "reference", "sha256": "b" * 64}, report, {}, runtime)
    stock._verify_runtime(tmp_path, binding)
    runtime_file.write_bytes(b"different runtime source\n")
    with pytest.raises(ValueError, match="runtime source"):
        stock._verify_runtime(tmp_path, binding)


def _source_fixture(tmp_path: Path) -> tuple[Path, Path]:
    sources = {}
    index_sources = {}
    cases = []
    for offset, case in enumerate(prepare.CASES):
        old = f"case-{int(case[-2:]):02d}"
        seed = prepare.SOURCE_SEEDS[case] - 110
        source_id, reference_id = f"fault-s{seed}", f"none-s{seed}"
        cases.append({"case_id": old, "seed": seed, "source_job": source_id,
                      "reference_job": reference_id, "hidden_label": "ignored"})
        source_arm = "F2" if case in {"case-05", "case-06"} else "COMPOUND"
        for arm, job in ((source_arm, source_id), ("NONE", reference_id)):
            log = tmp_path / "logs" / job
            event = _write(log / "events.out.tfevents.test", b"events")
            checkpoint = _write(log / "model_299.pt", (job + "-checkpoint").encode())
            _write(log / "params/env.yaml", b"rewards:\n  track_lin_vel_xy_exp:\n    weight: 1.0\n")
            _write(log / "params/agent.yaml", b"agent: true\nnum_steps_per_env: 24\n")
            stage = tmp_path / "stages" / job
            report = {
                "task": "Isaac-Velocity-Flat-Unitree-Go2-v0",
                "action_scale": 0.25,
                "protocol_sha256_lf": "a" * 64,
                "versions": {"rsl-rl-lib": "2.3.3", "isaaclab": "deadbeef"},
                "metrics": {"survival": 1.0},
            }
            fixed = _write(stage / "fixed-result.json", {"report": report, "execution": {}})
            hidden_sha = prepare._sha(prepare.EPOCH_CHILD.ROOT / "bench/hidden_faults.py", lf=True)
            manifest = {"schema": "source-child-manifest-v1", "arm": arm,
                        "task": "Isaac-Velocity-Flat-Unitree-Go2-v0", "seed": seed,
                        "source_commit_sha": "a" * 40,
                        "source_sha256_lf": {"bench/hidden_faults.py": hidden_sha},
                        "runtime_source_sha256_lf": {
                            "runtime.py": "b" * 64,
                            "source/isaaclab/isaaclab/envs/mdp/rewards.py": "c" * 64,
                            "source/isaaclab/isaaclab/actuators/actuator_pd.py": "d" * 64}}
            manifest_path = _write(stage / "manifest.json", manifest)
            public_id = f"source-{len(sources) + 1:02d}"
            binding = {"checkpoint_sha256": _sha(checkpoint), "fixed_result_sha256": _sha(fixed),
                       "manifest_sha256": _sha(manifest_path)}
            sources[job] = {
                "public_id": public_id, "binding": binding,
                "material": {"event": {"path": str(event), "sha256": _sha(event)},
                             "stage": str(stage)},
                "manifest": manifest,
            }
            index_sources[public_id] = binding
    metadata = _write(tmp_path / "metadata.json", {
        "schema": "lineage_source_inputs_metadata_v2", "cases": cases, "sources": sources})
    index = _write(tmp_path / "collected.json", {
        "schema": "lineage_source_inputs_v2", "sources": index_sources,
        "metadata_ref": {"file": metadata.name, "sha256": _sha(metadata)}})
    return index, metadata


def _phase_plan(tmp_path: Path) -> tuple[Path, dict]:
    inputs = []
    for case in prepare.CASES:
        old = f"case-{int(case[-2:]):02d}"
        path = _write(tmp_path / f"{old}.json", {"case_id": old})
        inputs.append({"role": f"protocol:{old}", "path": path.name, "sha256": _sha(path)})
    plan = _write(tmp_path / "phase-plan.json", {"phase": "A"})
    return plan, {"cases": [{"case_id": f"case-{i:02d}"} for i in range(5, 9)], "inputs": inputs}


def _patch_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path, dict]:
    index, metadata = _source_fixture(tmp_path)
    phase_path, phase_body = _phase_plan(tmp_path)
    recipe = _write(tmp_path / "recipe.json", {"schema": "recipe"})
    p0c = _write(tmp_path / "p0c.json", {"bands": {"survival_ratio_min": 0.9,
        "fall_frac_max": 0.05, "err_xy_ratio_max": 1.1513, "err_yaw_ratio_max": 1.1719}})
    sdk = _write(tmp_path / "sdk.json", {"schema": "correction_source_contract_v1",
        "runtime": {"implementation": "CPython", "version": [3, 10, 15], "cache_tag": "cpython-310"},
        "targets": {}})
    hidden_sha = prepare._sha(prepare.EPOCH_CHILD.ROOT / "bench/hidden_faults.py", lf=True)
    reward_nominal = {"module": "isaaclab.envs.mdp.rewards", "qualname": "track_lin_vel_xy_exp",
        "source": "source/isaaclab/isaaclab/envs/mdp/rewards.py", "source_sha256_lf": "c" * 64,
        "clean_fingerprint": "1" * 64, "expected_runtime_fingerprint": "1" * 64}
    reward_world = {**reward_nominal, "expected_runtime_fingerprint": "2" * 64}
    motor_nominal = {"module": "isaaclab.actuators.actuator_pd", "qualname": "DCMotor._clip_effort",
        "source": "source/isaaclab/isaaclab/actuators/actuator_pd.py", "source_sha256_lf": "d" * 64,
        "clean_fingerprint": "3" * 64, "expected_runtime_fingerprint": "3" * 64}
    motor_f2 = {**motor_nominal, "expected_runtime_fingerprint": "4" * 64}
    recipe_code = _write(tmp_path / "recipe-code.json", {
        "schema": "registered_environment_recipe_code_contract_v1",
        "provenance_kind": "compiled_code_contract_not_historical_runtime_observation",
        "runtime": {"implementation": "CPython", "version": [3, 10, 15], "cache_tag": "cpython-310"},
        "reward_targets": {"nominal_body_frame": reward_nominal,
                           "world_frame_reward": reward_world},
        "motor_targets": {"nominal": motor_nominal, "preserve_source_envelope": motor_f2},
        "module_imports": {"carb": False, "isaaclab": False, "omni": False, "torch": False},
        "gpu_launches": 0, "model_api_calls": 0,
        "hidden_recipe_source_sha256_lf": hidden_sha})
    source = _write(tmp_path / "source.py", b"source\n")
    for relative in (prepare.COMPARISON.REVIEW_POLICY_RELATIVE,
                     "bench/protocols/diagnostic_recovery_comparison_20261104_v2.md",
                     "bench/protocols/diagnostic_recovery_comparison_20261104_v3.md",
                     "bench/protocols/diagnostic_recovery_comparison_20261104_v4.md"):
        _write(tmp_path / relative, (prepare.ROOT / relative).read_bytes())
    monkeypatch.setattr(prepare, "ROOT", tmp_path)
    monkeypatch.setattr(prepare, "RECIPE", recipe)
    monkeypatch.setattr(prepare, "_SOURCE_FILES", {source.name})
    monkeypatch.setattr(prepare, "SDK_CONTRACT_SHA256", _sha(sdk))
    monkeypatch.setattr(prepare, "RECIPE_CODE_CONTRACT_SHA256", _sha(recipe_code))
    monkeypatch.setattr(prepare.UPSTREAM, "P0C_SUMMARY", p0c)
    monkeypatch.setattr(prepare.PHASE_A, "load_plan", lambda *_args, **_kwargs: phase_body)
    monkeypatch.setattr(prepare.COMPARISON, "load_plan", lambda path, **_kwargs: json.loads(Path(path).read_text()))
    monkeypatch.setattr(prepare.ACTUATOR, "validate_prepared_binding", lambda body: body)
    monkeypatch.setattr(prepare, "_optimizer_binding",
        lambda anchor, cp, runtime, mechanism, closure, environment_preservation=None: {
        "task": "task", "num_envs": 4096, "max_iterations": 300,
        "configured_learning_epochs": 5, "configured_mini_batches": 4,
        "runtime_updates_to_observe": 3, "source_commit_sha": anchor,
        "protocol_path": recipe.name, "protocol_sha256_lf": _sha(recipe),
        "source_sha256_lf": closure, "runtime_source_sha256_lf": runtime,
        "source_checkpoint_sha256": cp,
        **({"environment_preservation": environment_preservation}
           if environment_preservation is not None else {})})
    return index, metadata, phase_path, phase_body


def test_templates_bind_all_cases_mechanisms_and_ignore_hidden_eligibility(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    index, metadata, phase, _ = _patch_environment(tmp_path, monkeypatch)
    sdk = tmp_path / "sdk.json"
    out = tmp_path / "templates-out"
    result = prepare.prepare_templates(source_index=index, metadata=metadata,
        diagnostic_plan=phase, diagnostic_registration_anchor="b" * 40,
        source_anchor="a" * 40, sdk_contract=sdk,
        recipe_code_contract=tmp_path / "recipe-code.json", output=out, actor="operator",
        isaaclab_commit="c" * 40,
        require_committed=False)
    plan = json.loads((out / "registered-plan.json").read_text())
    assert result["actual_training_started"] is False
    assert [row["case_id"] for row in plan["cases"]] == list(prepare.CASES)
    assert set(plan["template_commitments"]) == set(prepare.CASES)
    assert all(set(rows) == set(prepare.MECHANISMS)
               for rows in plan["template_commitments"].values())
    assert plan["recovery_policy"]["seeds"]["case-05"] == {"resume": 2664, "restart": 5664}
    f2 = json.loads((out / "templates/case-05/optimizer_update_count/training-binding.json").read_text())
    compound = json.loads((out / "templates/case-07/optimizer_update_count/training-binding.json").read_text())
    assert f2["environment_preservation"]["mode"] == "nominal_body_frame"
    assert (json.loads(Path(f2["environment_preservation"]["expected_reward"]["path"]).read_text())
            ["source_arm"] == "F2")
    assert f2["environment_preservation"]["secondary_target"]["expected_runtime_fingerprint"] == "4" * 64
    assert compound["environment_preservation"]["mode"] == "world_frame_reward"
    assert (json.loads(Path(compound["environment_preservation"]["expected_reward"]["path"]).read_text())
            ["source_arm"] == "COMPOUND")
    assert compound["environment_preservation"]["target"]["expected_runtime_fingerprint"] == "2" * 64
    serialized = (out / "registered-plan.json").read_text()
    assert "hidden_label" not in serialized and "eligible" not in serialized


def test_templates_v2_bind_prior_proof_without_private_path_in_public_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index, metadata, phase, _ = _patch_environment(tmp_path, monkeypatch)
    proof = _write(tmp_path / "prior-proof.json", {"schema": "prior-proof-fixture"})
    proof_ref = {"path": str(proof.resolve()), "sha256": _sha(proof)}
    logical = {
        "logical_id": "case-05/fixed_sequence/case_02/CORRECTED",
        "proof_sha256": proof_ref["sha256"],
        "source_checkpoint_sha256": "a" * 64,
        "completed_checkpoint_sha256": "b" * 64,
        "seed": 2664,
        "task": "Isaac-Velocity-Flat-Unitree-Go2-v0",
        "correction_id": "restore_motor_torque_speed",
        "diagnostic_manifest_sha256": "c" * 64,
    }
    monkeypatch.setattr(prepare, "_CONTINUATION_SOURCE_FILES", {"source.py"})
    monkeypatch.setattr(prepare, "_prior_training", lambda *_args: (proof_ref, logical))
    isaaclab = tmp_path / "IsaacLab"
    isaaclab.mkdir()
    out = tmp_path / "templates-v2"
    result = prepare.prepare_templates(
        source_index=index,
        metadata=metadata,
        diagnostic_plan=phase,
        diagnostic_registration_anchor="b" * 40,
        source_anchor="a" * 40,
        sdk_contract=tmp_path / "sdk.json",
        recipe_code_contract=tmp_path / "recipe-code.json",
        output=out,
        actor="operator",
        isaaclab_commit="c" * 40,
        prior_training_proof=proof,
        isaaclab_root=isaaclab,
        require_committed=False,
    )
    plan_path = out / "registered-plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert plan["schema"] == prepare.COMPARISON.PLAN_SCHEMA_V2
    assert plan["limits"]["gpu_sdk_launch_cap"] == 31
    assert plan["limits"]["total_owned_launch_cap"] == 63
    assert plan["continuation"]["expected_new_gpu_sdk_launches"] == 29
    assert result["prior_training_proof"] == proof_ref
    assert str(tmp_path) not in plan_path.read_text(encoding="utf-8")


def test_templates_reject_source_index_and_checkpoint_substitution(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    index, metadata, phase, _ = _patch_environment(tmp_path, monkeypatch)
    sdk = tmp_path / "sdk.json"
    changed = json.loads(index.read_text())
    changed["sources"]["source-01"]["checkpoint_sha256"] = "f" * 64
    index.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(prepare.PreparationError, match="source index binding"):
        prepare.prepare_templates(source_index=index, metadata=metadata,
            diagnostic_plan=phase, diagnostic_registration_anchor="b" * 40,
            source_anchor="a" * 40, sdk_contract=sdk,
            recipe_code_contract=tmp_path / "recipe-code.json", output=tmp_path / "bad-index",
            actor="operator", isaaclab_commit="c" * 40, require_committed=False)

    checkpoint_root = tmp_path / "checkpoint"
    index, metadata, phase, _ = _patch_environment(checkpoint_root, monkeypatch)
    source = json.loads(metadata.read_text())["sources"]["fault-s2554"]
    Path(source["material"]["event"]["path"]).parent.joinpath("model_299.pt").write_bytes(b"swapped")
    with pytest.raises(prepare.PreparationError, match="source checkpoint lineage"):
        prepare.prepare_templates(source_index=index, metadata=metadata,
            diagnostic_plan=phase, diagnostic_registration_anchor="b" * 40,
            source_anchor="a" * 40, sdk_contract=checkpoint_root / "sdk.json",
            recipe_code_contract=checkpoint_root / "recipe-code.json",
            output=checkpoint_root / "bad-checkpoint", actor="operator",
            isaaclab_commit="c" * 40, require_committed=False)


def test_templates_v3_bind_two_imports_and_separate_historical_sdk_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index, metadata, phase, _ = _patch_environment(tmp_path, monkeypatch)
    metadata_body = json.loads(metadata.read_text(encoding="utf-8"))
    source_checkpoint = metadata_body["sources"]["fault-s2554"]["binding"]["checkpoint_sha256"]
    common = {
        "diagnostic_manifest_sha256": "c" * 64,
        "source_checkpoint_sha256": source_checkpoint,
        "final_checkpoint_sha256": "d" * 64,
        "seed": 2664,
        "task": "Isaac-Velocity-Flat-Unitree-Go2-v0",
        "correction_id": "restore_motor_torque_speed",
        "source_anchor": prepare.PRIOR_V5_SOURCE_ANCHOR,
    }
    effect = _write(tmp_path / "effect-proof.json", {
        "schema": "registered_completed_effect_proof_v1",
        "identity": {
            "prior_plan_sha256": "e" * 64,
            **common,
            "registration_anchor": prepare.PRIOR_V5_REGISTRATION_ANCHOR,
        },
    })
    training = _write(tmp_path / "training-proof.json", {
        "schema": "registered_completed_training_proof_v1",
        "identity": {**common, "case_id": "case-05", "method": "static_rule"},
    })

    class ProofModule:
        @staticmethod
        def verify_completed_effect(*_args, **_kwargs):
            return {"execution_origin": "verified_prior_effect"}

        @staticmethod
        def load_completed_training(*_args, **_kwargs):
            return {"new_training_launched": False}

    monkeypatch.setattr(prepare, "_continuation_module", lambda: ProofModule)
    monkeypatch.setattr(prepare, "_CONTINUATION_V3_SOURCE_FILES", {"source.py"})
    isaaclab = tmp_path / "IsaacLab"
    isaaclab.mkdir()
    out = tmp_path / "templates-v3"
    result = prepare.prepare_templates(
        source_index=index,
        metadata=metadata,
        diagnostic_plan=phase,
        diagnostic_registration_anchor="b" * 40,
        source_anchor="a" * 40,
        sdk_contract=tmp_path / "sdk.json",
        recipe_code_contract=tmp_path / "recipe-code.json",
        output=out,
        actor="operator",
        isaaclab_commit="c" * 40,
        prior_completed_effect_proof=effect,
        prior_static_training_proof=training,
        isaaclab_root=isaaclab,
        require_committed=False,
    )
    plan = json.loads((out / "registered-plan.json").read_text(encoding="utf-8"))
    assert plan["schema"] == prepare.COMPARISON.PLAN_SCHEMA_V3
    assert plan["limits"]["gpu_sdk_launch_cap"] == 29
    assert plan["limits"]["total_owned_launch_cap"] == 61
    assert plan["continuation"]["prior_consumed_gpu_sdk_launches"] == 3
    assert plan["continuation"]["expected_new_gpu_sdk_launches"] == 27
    assert [(row["method"], row["kind"]) for row in plan["continuation"]["stage_imports"]] == [
        ("fixed_sequence", "completed_effect"),
        ("static_rule", "completed_training"),
    ]
    assert result["continuation_proofs"] == {
        "prior_effect": {"path": str(effect.resolve()), "sha256": _sha(effect)},
        "prior_training": {"path": str(training.resolve()), "sha256": _sha(training)},
    }
    public = (out / "registered-plan.json").read_text(encoding="utf-8")
    assert str(tmp_path) not in public

    (prepare.ROOT / "shared.py").write_text("# registered shared profile\n", encoding="utf-8")
    monkeypatch.setattr(prepare, "_CONTINUATION_V7_SOURCE_FILES", {"shared.py"})
    shared_out = tmp_path / "templates-v7"
    prepare.prepare_templates(
        source_index=index, metadata=metadata, diagnostic_plan=phase,
        diagnostic_registration_anchor="b" * 40, source_anchor="a" * 40,
        sdk_contract=tmp_path / "sdk.json", recipe_code_contract=tmp_path / "recipe-code.json",
        output=shared_out, actor="operator", isaaclab_commit="c" * 40,
        prior_completed_effect_proof=effect, prior_static_training_proof=training,
        isaaclab_root=isaaclab,
        gpu_admission_profile=prepare.GPU_ADMISSION_PROFILE_V3_SHARED,
        require_committed=False,
    )
    shared_plan = json.loads((shared_out / "registered-plan.json").read_text(encoding="utf-8"))
    assert shared_plan["continuation"]["gpu_admission_profile"] == (
        prepare.GPU_ADMISSION_PROFILE_V3_SHARED
    )
    assert shared_plan["continuation"]["continuation_tag"] == "v7_shared_desktop"
    assert "shared.py" in shared_plan["source_closure"]
    assert plan["continuation"]["gpu_admission_profile"] == prepare.GPU_ADMISSION_PROFILE_V3
    assert plan["continuation"]["continuation_tag"] == "v6_fresh"

    monkeypatch.setattr(prepare.COMPARISON, "load_plan", lambda *_args, **_kwargs: plan)
    monkeypatch.setattr(
        prepare.SEED_GUARD,
        "assert_fresh_seeds",
        lambda seeds, *_args, **_kwargs: {
            "schema": "core_seed_freshness_v1", "status": "fresh", "seeds": list(seeds),
        },
    )
    diagnostic_root = tmp_path / "diagnostics-v3"
    for case in prepare.CASES:
        checkpoint = plan["template_commitments"][case]["actuator"]["source_checkpoint_sha256"]
        for method in prepare.METHODS:
            source = _write(
                diagnostic_root / "bound" / case / method / "source.json",
                {"case_id": case, "checkpoint_sha256": checkpoint},
            )
            _write(
                diagnostic_root / "bound" / case / method / "session/manifest.json",
                {
                    "method": method, "execution_class": "actual",
                    "source": {"path": str(source), "sha256": _sha(source)},
                    "source_verification": {"case_id": case, "checkpoint_sha256": checkpoint},
                },
            )
    statuses: dict[tuple[str, str], dict] = {
        (case, method): {"status": "unidentifiable", "case_id": case}
        for case in prepare.CASES for method in prepare.METHODS
    }
    for method in ("fixed_sequence", "static_rule"):
        statuses[("case-05", method)] = {
            "status": "confirmed", "confirmed_mechanism": "actuator", "case_id": "case-05",
            "source_verification": {"checkpoint_sha256": source_checkpoint},
        }
    monkeypatch.setattr(
        prepare, "_diagnostic_status",
        lambda session: statuses[(session.parent.parent.name, session.parent.name)],
    )

    class FakeSession:
        def __init__(self, path: Path):
            self.folder = Path(path).resolve(strict=True)
            body = json.loads((self.folder / "manifest.json").read_text(encoding="utf-8"))
            self._manifest_sha256 = (
                "c" * 64
                if self.folder.parent.parent.name == "case-05"
                and self.folder.parent.name in {"fixed_sequence", "static_rule"}
                else _sha(self.folder / "manifest.json")
            )
            self.manifest = body
            self.source = json.loads(Path(body["source"]["path"]).read_text(encoding="utf-8"))

    monkeypatch.setattr(prepare, "InvestigationSession", FakeSession)
    monkeypatch.setattr(
        prepare.RECOVERY, "_diagnostic",
        lambda _session, _mechanism: (
            {"probe": "P_actuator_mapping", "request_id": "diag-1"}, {},
            {"receipt_ref": {"path": str(tmp_path / "diag.json"), "sha256": "f" * 64}},
        ),
    )

    def run(command, _session, payload, **_kwargs):
        if command == "recovery-prepare":
            Path(payload["folder"]).mkdir(parents=True)
            return {"mechanism": "actuator"}
        if command == "recovery-propose":
            return {"status": {"pending": {"request_id": "req-v3", "effect_key": "effect-v3"}}}
        raise AssertionError(command)

    monkeypatch.setattr(prepare.investigate, "run", run)
    jobs = tmp_path / "jobs-v3"
    job_result = prepare.prepare_jobs(
        templates=out, registered_plan=out / "registered-plan.json",
        registration_anchor="d" * 40, diagnostic_root=diagnostic_root,
        isaaclab_root=isaaclab, output=jobs, require_committed=False,
    )
    job = json.loads((jobs / "job.json").read_text(encoding="utf-8"))
    fixed_arm = next(row for row in job["arms"] if row["case_id"] == "case-05"
                     and row["method"] == "fixed_sequence")
    static_arm = next(row for row in job["arms"] if row["case_id"] == "case-05"
                      and row["method"] == "static_rule")
    assert job_result["recovery_prepared"] == 1
    assert fixed_arm["recovery_payload"] is None
    assert fixed_arm["continuation_import"]["kind"] == "completed_effect"
    assert static_arm["continuation_import"]["kind"] == "completed_training"
    static_protocol = json.loads(Path(static_arm["recovery_payload"]["protocol"]["path"]).read_text())
    assert static_protocol["schema"] == "registered_recovery_protocol_v3"
    assert static_protocol["operation"]["gpu_admission_profile"] == prepare.GPU_ADMISSION_PROFILE_V3
    assert static_protocol["operation"]["prior_training_artifact_id"] == "prior_training"


@pytest.mark.parametrize("retry_observation", [False, True])
def test_templates_and_jobs_v4_import_two_completed_effects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, retry_observation: bool
) -> None:
    index, metadata, phase, _ = _patch_environment(tmp_path, monkeypatch)
    source_checkpoint = json.loads(metadata.read_text(encoding="utf-8"))[
        "sources"]["fault-s2554"]["binding"]["checkpoint_sha256"]
    common = {"source_checkpoint_sha256": source_checkpoint,
              "final_checkpoint_sha256": "d" * 64, "seed": 2664,
              "task": "Isaac-Velocity-Flat-Unitree-Go2-v0",
              "correction_id": "restore_motor_torque_speed"}
    proofs = {
        "prior_effect": _write(tmp_path / "effect-v1.json", {
            "schema": "registered_completed_effect_proof_v1", "status": "completed_effect",
            "case_id": "case-05", "method": "fixed_sequence",
            "identity": {"prior_plan_sha256": "e" * 64,
                "diagnostic_manifest_sha256": "c" * 64, **common,
                "source_anchor": prepare.PRIOR_V5_SOURCE_ANCHOR,
                "registration_anchor": prepare.PRIOR_V5_REGISTRATION_ANCHOR}}),
        "prior_static_effect": _write(tmp_path / "static-effect-v2.json", {
            "schema": "registered_completed_effect_proof_v2", "status": "completed_effect",
            "case_id": "case-05", "method": "static_rule",
            "identity": {"prior_plan_sha256": "f" * 64,
                "diagnostic_manifest_sha256": "9" * 64, **common,
                "source_anchor": "a" * 40,
                "registration_anchor": prepare.PRIOR_V7_REGISTRATION_ANCHOR}}),
    }
    fixture_expected = {}
    for artifact_id, proof_path in proofs.items():
        proof = json.loads(proof_path.read_text(encoding="utf-8"))
        identity = proof["identity"]
        fixture_expected[artifact_id] = {
            "artifact_id": artifact_id, "kind": "completed_effect",
            "case_id": proof["case_id"], "method": proof["method"],
            "proof_sha256": _sha(proof_path),
            "prior_plan_sha256": identity["prior_plan_sha256"],
            "diagnostic_manifest_sha256": identity["diagnostic_manifest_sha256"],
            "source_checkpoint_sha256": identity["source_checkpoint_sha256"],
            "completed_checkpoint_sha256": identity["final_checkpoint_sha256"],
            "seed": identity["seed"], "task": identity["task"],
            "correction_id": identity["correction_id"],
            "source_anchor": identity["source_anchor"],
            "registration_anchor": identity["registration_anchor"],
        }
    monkeypatch.setattr(prepare, "V4_EXPECTED_IMPORTS", fixture_expected)

    class ProofModule:
        @staticmethod
        def verify_completed_effect(*_args, **_kwargs):
            return {"execution_origin": "verified_prior_effect"}

    monkeypatch.setattr(prepare, "_continuation_module_v4", lambda: ProofModule)
    monkeypatch.setattr(prepare, "_CONTINUATION_V3_SOURCE_FILES", {"source.py"})
    monkeypatch.setattr(prepare, "_CONTINUATION_V4_SOURCE_FILES", {"source.py"})
    monkeypatch.setattr(prepare, "_CONTINUATION_RETRY_SOURCE_FILES",
                        prepare.COMPARISON.CONSISTENT_OBSERVATION_SOURCES)
    _write(tmp_path / "bench/protocols/diagnostic_recovery_comparison_20261104_v8.md", b"v8\n")
    if retry_observation:
        for relative in prepare.COMPARISON.CONSISTENT_OBSERVATION_SOURCES:
            _write(tmp_path / relative, b"retry fixture\n")
        _write(tmp_path / "bench/protocols/diagnostic_recovery_comparison_20261104_v9.md", b"v9\n")
    isaaclab = tmp_path / "IsaacLab"
    isaaclab.mkdir()
    templates = tmp_path / "templates-v4"
    result = prepare.prepare_templates(
        source_index=index, metadata=metadata, diagnostic_plan=phase,
        diagnostic_registration_anchor="b" * 40, source_anchor="a" * 40,
        sdk_contract=tmp_path / "sdk.json", recipe_code_contract=tmp_path / "recipe-code.json",
        output=templates, actor="operator", isaaclab_commit="c" * 40,
        completed_effect_proofs=proofs, isaaclab_root=isaaclab, require_committed=False,
        gpu_admission_profile=prepare.GPU_ADMISSION_PROFILE_RETRY if retry_observation else None)
    plan = json.loads((templates / "registered-plan.json").read_text(encoding="utf-8"))
    assert plan["schema"] == prepare.COMPARISON.PLAN_SCHEMA_V4
    assert plan["limits"]["gpu_sdk_launch_cap"] == 27
    assert plan["limits"]["total_owned_launch_cap"] == 59
    assert plan["continuation"]["expected_new_gpu_sdk_launches"] == 26
    assert plan["continuation"]["cumulative_gpu_sdk_launches"] == 32
    assert plan["continuation"]["gpu_admission_profile"] == (
        prepare.GPU_ADMISSION_PROFILE_RETRY if retry_observation else prepare.GPU_ADMISSION_PROFILE_V3_SHARED)
    assert plan["continuation"]["continuation_tag"] == (
        "v9_consistent_observation" if retry_observation else "v8_resumable")
    assert result["continuation_proofs"] == {
        key: {"path": str(path.resolve()), "sha256": _sha(path)} for key, path in proofs.items()}

    monkeypatch.setattr(prepare.COMPARISON, "load_plan", lambda *_args, **_kwargs: plan)
    monkeypatch.setattr(prepare.SEED_GUARD, "assert_fresh_seeds",
                        lambda seeds, *_args, **_kwargs: {"status": "fresh", "seeds": list(seeds)})
    diagnostic_root = tmp_path / "diagnostics-v4"
    for case in prepare.CASES:
        checkpoint = plan["template_commitments"][case]["actuator"]["source_checkpoint_sha256"]
        for method in prepare.METHODS:
            source = _write(diagnostic_root / "bound" / case / method / "source.json",
                            {"case_id": case, "checkpoint_sha256": checkpoint})
            _write(diagnostic_root / "bound" / case / method / "session/manifest.json", {
                "method": method, "execution_class": "actual",
                "source": {"path": str(source), "sha256": _sha(source)},
                "source_verification": {"case_id": case, "checkpoint_sha256": checkpoint}})
    statuses: dict[tuple[str, str], dict[str, object]] = {
        (case, method): {"status": "unidentifiable", "case_id": case}
        for case in prepare.CASES for method in prepare.METHODS}
    for method in ("fixed_sequence", "static_rule"):
        statuses[("case-05", method)] = {"status": "confirmed",
            "confirmed_mechanism": "actuator", "case_id": "case-05",
            "source_verification": {"checkpoint_sha256": source_checkpoint}}
    monkeypatch.setattr(prepare, "_diagnostic_status",
        lambda session: statuses[(session.parent.parent.name, session.parent.name)])

    class FakeSession:
        def __init__(self, path: Path):
            self.folder = Path(path).resolve(strict=True)
            body = json.loads((self.folder / "manifest.json").read_text(encoding="utf-8"))
            key = (self.folder.parent.parent.name, self.folder.parent.name)
            self._manifest_sha256 = ({("case-05", "fixed_sequence"): "c" * 64,
                                      ("case-05", "static_rule"): "9" * 64}
                                     .get(key, _sha(self.folder / "manifest.json")))
            self.manifest = body
            self.source = json.loads(Path(body["source"]["path"]).read_text(encoding="utf-8"))

    monkeypatch.setattr(prepare, "InvestigationSession", FakeSession)
    jobs = tmp_path / "jobs-v4"
    job_result = prepare.prepare_jobs(
        templates=templates, registered_plan=templates / "registered-plan.json",
        registration_anchor="d" * 40, diagnostic_root=diagnostic_root,
        isaaclab_root=isaaclab, output=jobs, require_committed=False)
    job = json.loads((jobs / "job.json").read_text(encoding="utf-8"))
    imported = [row for row in job["arms"] if row["continuation_import"] is not None]
    assert job_result["recovery_prepared"] == 0
    assert [(row["method"], row["continuation_import"]["kind"]) for row in imported] == [
        ("fixed_sequence", "completed_effect"), ("static_rule", "completed_effect")]
    assert all(row["recovery_payload"] is None and row["approval_ref"] is None
               for row in imported)


def test_v4_rejects_self_consistent_unregistered_completed_effect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def proof(method: str, prior_plan: str, source_anchor: str, registration: str) -> dict:
        return {"status": "completed_effect", "case_id": "case-05", "method": method,
                "identity": {"prior_plan_sha256": prior_plan,
                    "diagnostic_manifest_sha256": "c" * 64,
                    "source_checkpoint_sha256": "d" * 64,
                    "final_checkpoint_sha256": "e" * 64, "seed": 2664, "task": "task",
                    "correction_id": "restore_motor_torque_speed",
                    "source_anchor": source_anchor, "registration_anchor": registration}}

    paths = {
        "prior_effect": _write(tmp_path / "fixed.json", proof(
            "fixed_sequence", "1" * 64, "a" * 40, "b" * 40)),
        "prior_static_effect": _write(tmp_path / "static.json", proof(
            "static_rule", "2" * 64, "f" * 40, "9" * 40)),
    }
    expected = {}
    for artifact_id, path in paths.items():
        body = json.loads(path.read_text(encoding="utf-8"))
        identity = body["identity"]
        expected[artifact_id] = {"artifact_id": artifact_id, "kind": "completed_effect",
            "case_id": body["case_id"], "method": body["method"],
            "proof_sha256": _sha(path), "prior_plan_sha256": identity["prior_plan_sha256"],
            "diagnostic_manifest_sha256": identity["diagnostic_manifest_sha256"],
            "source_checkpoint_sha256": identity["source_checkpoint_sha256"],
            "completed_checkpoint_sha256": identity["final_checkpoint_sha256"],
            "seed": identity["seed"], "task": identity["task"],
            "correction_id": identity["correction_id"],
            "source_anchor": identity["source_anchor"],
            "registration_anchor": identity["registration_anchor"]}
    monkeypatch.setattr(prepare, "V4_EXPECTED_IMPORTS", expected)

    class ProofModule:
        @staticmethod
        def verify_completed_effect(*_args, **_kwargs):
            return {"execution_origin": "verified_prior_effect"}

    monkeypatch.setattr(prepare, "_continuation_module_v4", lambda: ProofModule)
    changed = json.loads(paths["prior_static_effect"].read_text(encoding="utf-8"))
    changed["identity"]["prior_plan_sha256"] = "3" * 64
    _write(paths["prior_static_effect"], changed)
    with pytest.raises(prepare.PreparationError, match="registered basis"):
        prepare._v4_imports(paths)


def test_templates_v3_reject_cross_method_training_proof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index, metadata, phase, _ = _patch_environment(tmp_path, monkeypatch)
    common = {
        "diagnostic_manifest_sha256": "c" * 64,
        "source_checkpoint_sha256": "a" * 64,
        "final_checkpoint_sha256": "d" * 64,
        "seed": 2664, "task": "task", "correction_id": "restore_motor_torque_speed",
        "source_anchor": prepare.PRIOR_V5_SOURCE_ANCHOR,
    }
    effect = _write(tmp_path / "effect.json", {"identity": {
        "prior_plan_sha256": "e" * 64, **common,
        "registration_anchor": prepare.PRIOR_V5_REGISTRATION_ANCHOR,
    }})
    training = _write(tmp_path / "training.json", {"identity": {
        **common, "case_id": "case-05", "method": "fixed_sequence",
    }})
    isaaclab = tmp_path / "IsaacLab"
    isaaclab.mkdir()
    with pytest.raises(prepare.PreparationError, match="completed training proof logical identity"):
        prepare.prepare_templates(
            source_index=index, metadata=metadata, diagnostic_plan=phase,
            diagnostic_registration_anchor="b" * 40, source_anchor="a" * 40,
            sdk_contract=tmp_path / "sdk.json", recipe_code_contract=tmp_path / "recipe-code.json",
            output=tmp_path / "bad-v3", actor="operator", isaaclab_commit="c" * 40,
            prior_completed_effect_proof=effect, prior_static_training_proof=training,
            isaaclab_root=isaaclab, require_committed=False,
        )


def test_templates_reject_explicit_shared_profile_without_v3_proofs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index, metadata, phase, _ = _patch_environment(tmp_path, monkeypatch)
    with pytest.raises(prepare.PreparationError, match="requires v3 continuation proofs"):
        prepare.prepare_templates(
            source_index=index, metadata=metadata, diagnostic_plan=phase,
            diagnostic_registration_anchor="b" * 40, source_anchor="a" * 40,
            sdk_contract=tmp_path / "sdk.json", recipe_code_contract=tmp_path / "recipe-code.json",
            output=tmp_path / "bad-profile", actor="operator", isaaclab_commit="c" * 40,
            gpu_admission_profile=prepare.GPU_ADMISSION_PROFILE_V3_SHARED,
            require_committed=False,
        )


def test_v7_shared_profile_source_closure_is_complete() -> None:
    expected = {
        "evals/registered_desktop_gpu_shared_v1.py",
        "tests/test_registered_desktop_gpu_shared_v1.py",
        "bench/protocols/registered_desktop_gpu_shared_v1.json",
        "bench/protocols/diagnostic_recovery_comparison_20261104_v7.md",
    }
    assert prepare._CONTINUATION_V7_SOURCE_FILES == expected
    assert all((prepare.ROOT / relative).is_file() for relative in expected)
    policy = json.loads(
        (prepare.ROOT / "bench/protocols/registered_desktop_gpu_shared_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert policy["profile_id"] == prepare.GPU_ADMISSION_PROFILE_V3_SHARED


def test_templates_reject_source_manifest_or_code_contract_substitution(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    index, metadata, phase, _ = _patch_environment(tmp_path, monkeypatch)
    metadata_body = json.loads(metadata.read_text())
    source = metadata_body["sources"]["fault-s2554"]
    manifest_path = Path(source["material"]["stage"]) / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["arm"] = "NONE"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(prepare.PreparationError, match="source setup manifest reference"):
        prepare.prepare_templates(source_index=index, metadata=metadata, diagnostic_plan=phase,
            diagnostic_registration_anchor="b" * 40, source_anchor="a" * 40,
            sdk_contract=tmp_path / "sdk.json", recipe_code_contract=tmp_path / "recipe-code.json",
            output=tmp_path / "bad-source-manifest", actor="operator",
            isaaclab_commit="c" * 40, require_committed=False)

    second = tmp_path / "second"
    index, metadata, phase, _ = _patch_environment(second, monkeypatch)
    contract = second / "recipe-code.json"
    contract.write_text("{}", encoding="utf-8")
    with pytest.raises(prepare.PreparationError, match="code contract changed"):
        prepare.prepare_templates(source_index=index, metadata=metadata, diagnostic_plan=phase,
            diagnostic_registration_anchor="b" * 40, source_anchor="a" * 40,
            sdk_contract=second / "sdk.json", recipe_code_contract=contract,
            output=second / "bad-code-contract", actor="operator",
            isaaclab_commit="c" * 40, require_committed=False)

def test_jobs_preserve_all_arms_and_prepare_only_confirmed_registered_mechanism(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    index, metadata, phase, _ = _patch_environment(tmp_path, monkeypatch)
    template_root = tmp_path / "templates-out"
    prepare.prepare_templates(source_index=index, metadata=metadata, diagnostic_plan=phase,
        diagnostic_registration_anchor="b" * 40, source_anchor="a" * 40,
        sdk_contract=tmp_path / "sdk.json", recipe_code_contract=tmp_path / "recipe-code.json",
        output=template_root, actor="operator",
        isaaclab_commit="c" * 40,
        require_committed=False)
    plan_path = template_root / "registered-plan.json"
    plan = json.loads(plan_path.read_text())
    monkeypatch.setattr(prepare.COMPARISON, "load_plan", lambda *_args, **_kwargs: plan)
    monkeypatch.setattr(prepare.SEED_GUARD, "assert_fresh_seeds", lambda seeds, *_args, **_kwargs: {
        "schema": "core_seed_freshness_v1", "status": "fresh", "seeds": list(seeds)})
    diagnostic_root = tmp_path / "diagnostics"
    for case in prepare.CASES:
        checkpoint = plan["template_commitments"][case]["actuator"]["source_checkpoint_sha256"]
        for method in prepare.METHODS:
            source = _write(diagnostic_root / "bound" / case / method / "source.json", {
                "case_id": case, "checkpoint_sha256": checkpoint})
            _write(diagnostic_root / "bound" / case / method / "session" / "manifest.json", {
                "method": method, "execution_class": "actual",
                "source": {"path": str(source), "sha256": _sha(source)},
                "source_verification": {"case_id": case, "checkpoint_sha256": checkpoint}})
    statuses: dict[tuple[str, str], dict] = {
        (case, method): {"status": "unidentifiable", "case_id": case}
        for case in prepare.CASES for method in prepare.METHODS
    }
    statuses[("case-05", "fixed_sequence")] = {
        "status": "confirmed", "confirmed_mechanism": "optimizer_update_count",
        "case_id": "case-05",
        "source_verification": {"checkpoint_sha256":
            plan["template_commitments"]["case-05"]["actuator"]["source_checkpoint_sha256"]}}
    monkeypatch.setattr(prepare, "_diagnostic_status",
                        lambda session: statuses[(session.parent.parent.name, session.parent.name)])

    class FakeSession:
        def __init__(self, path: Path):
            self.folder = Path(path).resolve(strict=True)
            self._manifest_sha256 = _sha(self.folder / "manifest.json")
            body = json.loads((self.folder / "manifest.json").read_text())
            self.manifest = body
            self.source = json.loads(Path(body["source"]["path"]).read_text())

    monkeypatch.setattr(prepare, "InvestigationSession", FakeSession)
    monkeypatch.setattr(prepare.RECOVERY, "_diagnostic", lambda _session, _mechanism: (
        {"probe": "P_optimizer_updates", "request_id": "diag-1"}, {},
        {"receipt_ref": {"path": str(tmp_path / "diag.json"), "sha256": "f" * 64}}))
    calls = []

    def run(command, session, payload, **_kwargs):
        calls.append((command, Path(session), payload))
        if command == "recovery-prepare":
            Path(payload["folder"]).mkdir(parents=True)
            return {"mechanism": "optimizer_update_count"}
        if command == "recovery-propose":
            return {"status": {"pending": {"request_id": "req-1", "effect_key": "effect-1"}}}
        raise AssertionError(command)

    monkeypatch.setattr(prepare.investigate, "run", run)
    isaaclab = tmp_path / "IsaacLab"
    (isaaclab / "logs").mkdir(parents=True)
    result = prepare.prepare_jobs(templates=template_root, registered_plan=plan_path,
        registration_anchor="c" * 40, diagnostic_root=diagnostic_root,
        isaaclab_root=isaaclab, output=tmp_path / "jobs", require_committed=False)
    job = json.loads((tmp_path / "jobs/job.json").read_text())
    assert result["arms"] == 16 and result["recovery_prepared"] == 1
    assert len(job["arms"]) == 16
    active = next(row for row in job["arms"] if row["recovery_payload"] is not None)
    assert active["case_id"] == "case-05" and active["method"] == "fixed_sequence"
    protocol = json.loads(Path(active["recovery_payload"]["protocol"]["path"]).read_text())
    assert protocol["operation"]["runner_id"] == "r1_epochs_preserving_environment_v1"
    assert "prepared-recoveries" in active["recovery_folder"]
    assert [row[0] for row in calls] == ["recovery-prepare", "recovery-propose"]
    assert all(row["recovery_payload"] is None for row in job["arms"] if row is not active)
    assert json.loads((tmp_path / "jobs/restart-seed-freshness.json").read_text())["status"] == "fresh"
    approval = Path(active["approval_ref"]["path"])
    approval.write_text("{}", encoding="utf-8")
    with pytest.raises(prepare.COMPARISON.ComparisonContractError, match="approval reference changed"):
        prepare.COMPARISON._validate_job(job, job["plan"], plan, tmp_path / "jobs/job.json")


def test_jobs_v2_emit_prior_proof_only_for_exact_fixed_sequence_actuator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index, metadata, phase, _ = _patch_environment(tmp_path, monkeypatch)
    metadata_body = json.loads(metadata.read_text(encoding="utf-8"))
    source_checkpoint = metadata_body["sources"]["fault-s2554"]["binding"]["checkpoint_sha256"]
    proof = _write(tmp_path / "prior-proof.json", {"schema": "prior-proof-fixture"})
    proof_ref = {"path": str(proof.resolve()), "sha256": _sha(proof)}
    logical = {
        "logical_id": "case-05/fixed_sequence/case_02/CORRECTED",
        "proof_sha256": proof_ref["sha256"],
        "source_checkpoint_sha256": source_checkpoint,
        "completed_checkpoint_sha256": "b" * 64,
        "seed": 2664,
        "task": "Isaac-Velocity-Flat-Unitree-Go2-v0",
        "correction_id": "restore_motor_torque_speed",
        "diagnostic_manifest_sha256": "c" * 64,
    }
    monkeypatch.setattr(prepare, "_CONTINUATION_SOURCE_FILES", {"source.py"})
    monkeypatch.setattr(prepare, "_prior_training", lambda *_args: (proof_ref, logical))
    isaaclab = tmp_path / "IsaacLab"
    (isaaclab / "logs").mkdir(parents=True)
    template_root = tmp_path / "templates-v2"
    prepare.prepare_templates(
        source_index=index,
        metadata=metadata,
        diagnostic_plan=phase,
        diagnostic_registration_anchor="b" * 40,
        source_anchor="a" * 40,
        sdk_contract=tmp_path / "sdk.json",
        recipe_code_contract=tmp_path / "recipe-code.json",
        output=template_root,
        actor="operator",
        isaaclab_commit="c" * 40,
        prior_training_proof=proof,
        isaaclab_root=isaaclab,
        require_committed=False,
    )
    plan_path = template_root / "registered-plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    monkeypatch.setattr(prepare.COMPARISON, "load_plan", lambda *_args, **_kwargs: plan)
    monkeypatch.setattr(
        prepare.SEED_GUARD,
        "assert_fresh_seeds",
        lambda seeds, *_args, **_kwargs: {
            "schema": "core_seed_freshness_v1",
            "status": "fresh",
            "seeds": list(seeds),
        },
    )
    diagnostic_root = tmp_path / "diagnostics"
    for case in prepare.CASES:
        checkpoint = plan["template_commitments"][case]["actuator"]["source_checkpoint_sha256"]
        for method in prepare.METHODS:
            source = _write(
                diagnostic_root / "bound" / case / method / "source.json",
                {"case_id": case, "checkpoint_sha256": checkpoint},
            )
            _write(
                diagnostic_root / "bound" / case / method / "session/manifest.json",
                {
                    "method": method,
                    "execution_class": "actual",
                    "source": {"path": str(source), "sha256": _sha(source)},
                    "source_verification": {"case_id": case, "checkpoint_sha256": checkpoint},
                },
            )
    statuses: dict[tuple[str, str], dict[str, object]] = {
        (case, method): {"status": "unidentifiable", "case_id": case}
        for case in prepare.CASES
        for method in prepare.METHODS
    }
    statuses[("case-05", "fixed_sequence")] = {
        "status": "confirmed",
        "confirmed_mechanism": "actuator",
        "case_id": "case-05",
        "source_verification": {"checkpoint_sha256": source_checkpoint},
    }
    monkeypatch.setattr(
        prepare,
        "_diagnostic_status",
        lambda session: statuses[(session.parent.parent.name, session.parent.name)],
    )

    class FakeSession:
        def __init__(self, path: Path):
            self.folder = Path(path).resolve(strict=True)
            body = json.loads((self.folder / "manifest.json").read_text(encoding="utf-8"))
            self._manifest_sha256 = (
                "c" * 64
                if self.folder.parent.parent.name == "case-05"
                and self.folder.parent.name == "fixed_sequence"
                else _sha(self.folder / "manifest.json")
            )
            self.manifest = body
            self.source = json.loads(Path(body["source"]["path"]).read_text(encoding="utf-8"))

    monkeypatch.setattr(prepare, "InvestigationSession", FakeSession)
    monkeypatch.setattr(
        prepare.RECOVERY,
        "_diagnostic",
        lambda _session, _mechanism: (
            {"probe": "P_actuator_mapping", "request_id": "diag-1"},
            {},
            {"receipt_ref": {"path": str(tmp_path / "diag.json"), "sha256": "f" * 64}},
        ),
    )

    def run(command, _session, payload, **_kwargs):
        if command == "recovery-prepare":
            Path(payload["folder"]).mkdir(parents=True)
            return {"mechanism": "actuator"}
        if command == "recovery-propose":
            return {"status": {"pending": {"request_id": "req-1", "effect_key": "effect-1"}}}
        raise AssertionError(command)

    monkeypatch.setattr(prepare.investigate, "run", run)
    jobs = tmp_path / "jobs-v2"
    result = prepare.prepare_jobs(
        templates=template_root,
        registered_plan=plan_path,
        registration_anchor="d" * 40,
        diagnostic_root=diagnostic_root,
        isaaclab_root=isaaclab,
        output=jobs,
        require_committed=False,
    )
    job = json.loads((jobs / "job.json").read_text(encoding="utf-8"))
    active = next(row for row in job["arms"] if row["recovery_payload"] is not None)
    protocol = json.loads(Path(active["recovery_payload"]["protocol"]["path"]).read_text())
    resolver = json.loads(Path(active["recovery_payload"]["artifact_resolver"]["path"]).read_text())
    assert result["recovery_prepared"] == 1
    assert protocol["schema"] == "registered_recovery_protocol_v2"
    assert protocol["operation"]["gpu_admission_profile"] == "registered_desktop_shared_v2"
    assert protocol["operation"]["prior_training_artifact_id"] == "prior_training"
    assert resolver["artifacts"]["prior_training"] == proof_ref


def test_bound_diagnostic_sessions_use_phase_a_layout_and_reject_substitution(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cases = [{"case_id": case, "rotation": list(prepare.PHASE_A.METHODS)}
             for case in prepare.CASES]
    phase_plan = {
        "cases": cases,
        "limits": {"max_checks": 3},
        "inputs": [],
    }
    plan_ref = {"path": "phase-plan.json", "sha256": "a" * 64}
    root = (tmp_path / "comparison").resolve()
    api_root = str((root / "api").resolve())
    rows = []
    for case in prepare.CASES:
        arms = []
        for method in prepare.METHODS:
            base = root / "bound" / case / method
            source = _write(base / "source.json", {
                "schema": "registered_diagnostic_source_fixture_v1",
                "case_id": case, "checkpoint_sha256": "b" * 64,
                "facts": {}, "binding": {}})
            session = base / "session"
            arms.append({
                "method": method,
                "source_path": str(source.resolve()),
                "session_path": str(session.resolve()),
                "prepare_payload": {
                    "source_path": str(source.resolve()), "method": method,
                    "case_id": case, "max_checks": 3, "execution_class": "fixture",
                    "protocol": None,
                    "api_root": api_root if method in {"one_shot_llm", "iterative_agent"} else None,
                },
            })
        rows.append({"case_id": case, "arms": arms})
    binding = {"schema": prepare.PHASE_A.JOB_SCHEMA, "plan": plan_ref,
               "api_root": api_root, "cases": rows}
    normalized = prepare.PHASE_A._validate_binding(binding, plan_ref, phase_plan, root, tmp_path)
    assert set(normalized) == set(prepare.CASES)
    commitments = {
        case: {mechanism: {"source_checkpoint_sha256": "b" * 64}
               for mechanism in prepare.MECHANISMS}
        for case in prepare.CASES
    }
    recovery_plan = {"template_commitments": commitments}
    registry = {
        "mechanisms": ["m1"], "outcomes": ["abnormal", "normal", "unknown"],
        "probes": {"p1": {"measures": "fixture", "cost_gpu_s": 0, "args": {},
                           "expect": {"m1": "abnormal"}}},
        "fixed_order": ["p1"],
    }
    for case in prepare.CASES:
        for method in prepare.METHODS:
            base = root / "bound" / case / method
            source = base / "source.json"
            model_binding = ({"source_anchor": "c" * 40, "protocol_sha256": "d" * 64,
                              "params": {}, "api_cap": 4, "api_root": api_root}
                             if method in {"one_shot_llm", "iterative_agent"} else None)
            prepare.InvestigationSession.prepare(
                source, base / "session", registry=registry, method=method, max_checks=3,
                gpu_budget_per_check_s=0, model_binding=model_binding, execution_class="actual",
                source_verifier=lambda path, current_case=case: {
                    "status": "verified", "execution_class": "actual",
                    "verifier_id": "phase_a_fixture_v1", "source_sha256": _sha(path),
                    "checkpoint_sha256": "b" * 64, "case_id": current_case,
                },
            )
    sessions = prepare._bound_diagnostic_sessions(root, recovery_plan)
    assert sessions[("case-05", "fixed_sequence")] == (
        root / "bound/case-05/fixed_sequence/session")

    wrong = root / "bound/case-05/fixed_sequence/session/manifest.json"
    original = wrong.read_bytes()
    body = json.loads(wrong.read_text())
    body["method"] = "iterative_agent"
    wrong.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(prepare.PreparationError, match="session binding differs"):
        prepare._bound_diagnostic_sessions(root, recovery_plan)

    wrong.write_bytes(original)
    wrong.unlink()
    _write(root / "case-05/fixed_sequence/manifest.json", body)
    with pytest.raises(prepare.PreparationError, match="legacy diagnostic session layout"):
        prepare._bound_diagnostic_sessions(root, recovery_plan)

    (root / "case-05/fixed_sequence/manifest.json").unlink()
    wrong.write_bytes(original)
    body = json.loads(original)
    other_source = root / "bound/case-05/one_shot_llm/source.json"
    body["source"] = {"path": str(other_source), "sha256": _sha(other_source)}
    wrong.write_text(json.dumps(body, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    anchor = wrong.parent / "anchor.json"
    anchor.write_text(json.dumps({"schema": "triage_investigation_session_v1_anchor",
                                  "manifest_sha256": _sha(wrong)},
                                 sort_keys=True, separators=(",", ":")), encoding="utf-8")
    with pytest.raises(prepare.PreparationError, match="session binding differs"):
        prepare._bound_diagnostic_sessions(root, recovery_plan)
