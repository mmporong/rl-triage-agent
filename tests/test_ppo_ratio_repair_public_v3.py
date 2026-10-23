"""C3 공개 exporter가 v2 경계와 bounded GPU receipt를 함께 보존하는지 확인한다."""
from __future__ import annotations

import json
import shutil
import sys
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "tests")]
import export_ppo_ratio_repair_v2 as BASE  # noqa: E402
import export_ppo_ratio_repair_v3 as EXPORT  # noqa: E402
import test_ppo_ratio_repair_public_v2 as OLD  # noqa: E402


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def clear(memory: int = 2500) -> dict:
    return {
        "busy": False,
        "busy_reasons": [],
        "admission_policy": EXPORT.EXECUTION_V3.GPU_PROFILE,
        "gpu_exclusive": False,
        "performance_measurement": False,
        "utilization_percent": 0,
        "memory_used_mib": memory,
        "compute_process_count": 0,
        "processes": [],
        "active_engines": [],
    }


def transient(memory: int = 2500) -> dict:
    pid = 1320
    return {
        **clear(memory),
        "busy": True,
        "busy_reasons": [f"unapproved_active_engine:{pid}:3D", f"unverified_active_engine_process:{pid}"],
        "active_engines": [{
            "name": f"pid_{pid}_luid_0x0_phys_0_eng_0_engtype_3D",
            "utilization_percent": 1,
            "process_name": "csrss.exe",
        }],
    }


def receipt(*, resampled: bool) -> dict:
    if not resampled:
        samples, identities, reason = [clear()], [], "initial_clear"
    else:
        identity = {"ProcessId": 1320, "CreationDate": "2026-10-23T00:00:00.0000000Z", "SessionId": 1}
        samples, identities, reason = [transient(), clear(), clear()], [identity, identity, identity], \
            "two_consecutive_strict_clear"
    bounded = {
        "attempted": resampled,
        "accepted": True,
        "reason": reason,
        "interval_s": 2,
        "max_samples": 3,
        "sample_count": len(samples),
        "required_consecutive_clear": 2,
        "identities": identities,
    }
    return {**samples[-1], "bounded_resample": bounded, "samples": samples}


def rejected_receipt() -> dict:
    sample = transient()
    sample["busy_reasons"] = [
        "unapproved_gpu_process:1320",
        "unapproved_active_engine:1320:3D",
        "unverified_active_engine_process:1320",
    ]
    sample["compute_process_count"] = 1
    sample["processes"] = [{"pid": 1320, "name": "kit.exe", "type": "C+G", "known_workload": True}]
    sample["active_engines"][0].update(utilization_percent=16, process_name="kit.exe")
    bounded = {
        "attempted": False,
        "accepted": False,
        "reason": "strict_busy_not_registered_transient",
        "interval_s": 2,
        "max_samples": 3,
        "sample_count": 1,
        "required_consecutive_clear": 2,
        "identities": [],
    }
    return {**sample, "bounded_resample": bounded, "samples": [sample]}


def _rewrite_training(run: Path, result: dict, protocol: dict) -> None:
    for arm, (seed, folder_name, run_name, _source_kind) in EXPORT.EXPECTED_ARMS.items():
        folder = run / folder_name
        manifest = EXPORT.read(folder / "manifest.json")
        manifest.update(
            seed=seed,
            run_name=run_name,
            source_commit_sha=result["experiment_commit"],
            protocol_path=EXPORT.PROTOCOL.relative_to(ROOT).as_posix(),
            protocol_sha256_lf=result["protocol_sha256_lf"],
            source_sha256_lf=protocol["source_sha256_lf"],
            runtime_source_sha256_lf=protocol["runtime_source_sha256_lf"],
        )
        write(folder / "manifest.json", manifest)
        observer = EXPORT.TRAINING.observer_manifest(manifest)
        observe_module = BASE.EXECUTION.GRADIENT.O
        records_path = folder / "records.jsonl"
        rows = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines()]
        for row in rows:
            row["manifest_sha256"] = observe_module.manifest_fingerprint(observer)
            row["binding"] = observe_module.manifest_binding(observer)
        records_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
            newline="\n",
        )
        measured = BASE.EXECUTION.inspection(records_path, manifest)
        artifact_path = run / f"artifact_{arm.lower()}.json"
        artifact = EXPORT.read(artifact_path)
        artifact.update(seed=seed, run_name=run_name, report=measured["report"])
        artifact["configuration"]["seed"] = seed
        artifact["inspection"] = {
            key: measured[key]
            for key in (
                "count_status", "gradient_status", "first_mismatch_update", "gradient_batches",
                "optimizer_steps", "minibatches",
            )
        }
        write(artifact_path, artifact)


def _rewrite_ledgers(run: Path, result: dict) -> None:
    fault = EXPORT.read(run / f"artifact_{EXPORT.DEMO.FAULT.lower()}.json")
    fault_sha = fault["checkpoint"]["sha256"]
    probe_path = run / "probes.ledger.jsonl"
    probe_path.unlink()
    probes = EXPORT.CONTRACT.M.RegistryLedger(probe_path, EXPORT.CONTRACT.registry())
    prereg = {
        "checkpoint_sha256": fault_sha,
        "seed": 2164,
        "protocol_sha256_lf": result["protocol_sha256_lf"],
        "experiment_commit": result["experiment_commit"],
        "registry_digest": probes.registry_digest,
    }
    old_probe = result["probe_steps"][0]
    request = probes.propose(prereg, EXPORT.CONTRACT.GRADIENT_PROBE, 1200)
    probes.approve(request["request_id"], "fixture-operator")
    probes.consume(request["request_id"], prereg, EXPORT.CONTRACT.GRADIENT_PROBE)
    new_probe = probes.receipt(request["request_id"], "abnormal", old_probe["measurement"], 2.0, 0)
    result["probe_steps"] = [{**new_probe, "probe": EXPORT.CONTRACT.GRADIENT_PROBE}]
    diagnosis = result["model"]["diagnosis"]["review"]["decision"]
    result["loop"] = EXPORT.CONTRACT.advance(
        EXPORT.CONTRACT.initial_state(diagnosis), EXPORT.CONTRACT.GRADIENT_PROBE, "abnormal"
    )
    write(run / "loop_01.json", result["loop"])

    correction_path = run / "correction.ledger.jsonl"
    correction_path.unlink()
    ledger = EXPORT.CorrectionLedger(correction_path, EXPORT.CONTRACT.correction_registry())
    correction_packet = EXPORT.read(run / "model/correction.packet.json")
    correction_review = result["model"]["correction"]
    request = ledger.propose(
        EXPORT.CONTRACT.L.digest(result["loop"]),
        "optimizer",
        EXPORT.CONTRACT.L.digest(result["probe_steps"]),
        EXPORT.CONTRACT.CORRECTION,
        "restart",
        {"case": "optimizer_surrogate_ratio", "arm": "corrected"},
        3600,
        evidence={
            "packet_digest": EXPORT.CONTRACT.L.digest(correction_packet),
            "model_review_digest": EXPORT.CONTRACT.L.digest(correction_review),
            "protocol_sha256_lf": result["protocol_sha256_lf"],
            "checkpoint_sha256": fault_sha,
        },
    )
    ledger.approve(request["request_id"], "fixture-operator")
    ledger.consume(request["request_id"], EXPORT.CONTRACT.L.digest(result["loop"]))
    ledger.receipt(
        request["request_id"], EXPORT.TAG, "recovered", EXPORT.sha256(run / "recovery.json"), 30.0, 0
    )


def _gpu_evidence(run: Path, result: dict) -> None:
    folders = [
        run / "training/none",
        run / "training/inverse_ratio",
        run / "source_evaluation",
        run / "reference",
        run / f"probe_1_{EXPORT.CONTRACT.GRADIENT_PROBE}",
        run / "correction_training",
        run / "correction_evaluation",
    ]
    for index, folder in enumerate(folders):
        write(folder / "execution.json", {
            "natural_exit": True,
            "timed_out": False,
            "exit_code": 0,
            "cleanup_verified": True,
            "wall_s": float(index + 1),
        })
        write(folder / "gpu_before.json", receipt(resampled=index == 0))
        write(folder / "gpu_after.json", receipt(resampled=index == 1))
    assert set(result["costs"]) == {
        "train_NONE", f"train_{EXPORT.DEMO.FAULT}", "source_evaluation", "reference",
        f"probe_1_{EXPORT.CONTRACT.GRADIENT_PROBE}", "correction_training", "correction_evaluation",
    }


def complete_fixture(tmp_path: Path) -> Path:
    run = OLD.complete_fixture(tmp_path)
    protocol = EXPORT.read(EXPORT.PROTOCOL)
    result = EXPORT.read(run / "result.json")
    result.update(
        schema="ppo_ratio_repair_demo_result_v3",
        tag=EXPORT.TAG,
        seed=2164,
        restart_seed=3164,
        experiment_commit=str(EXPORT.V2.git(
            "log", "-1", "--format=%H", "--", EXPORT.PROTOCOL.relative_to(ROOT).as_posix()
        )),
        protocol_sha256_lf=EXPORT.sha256(EXPORT.PROTOCOL, True),
        wording=protocol["outcome_wording"]["recovered"],
        design_coupling=protocol["design_coupling"],
    )
    write(run / "begin.json", {
        "tag": EXPORT.TAG,
        "experiment_commit": result["experiment_commit"],
        "protocol_sha256_lf": result["protocol_sha256_lf"],
        "approver": "fixture-operator",
        "identity_attested": False,
        "delegation": "승인된 등록 검사와 교정을 순서대로 실행하는 운영자 위임",
    })
    write(run / "seed_freshness.json", {
        "schema": "preregistered_seed_freshness_v1",
        "status": "fresh",
        "seeds": [2164, 3164],
        "evidence": [],
    })
    _rewrite_training(run, result, protocol)
    for relative, name in (
        ("source_evaluation/none.json", f"{EXPORT.TAG}_none"),
        ("source_evaluation/inverse_ratio.json", f"{EXPORT.TAG}_inverse_ratio"),
        ("correction_evaluation/corrected.json", f"{EXPORT.TAG}_fixed_corrected"),
    ):
        report = EXPORT.read(run / relative)
        report["name"] = name
        write(run / relative, report)
    reports = {
        "NONE": EXPORT.read(run / "source_evaluation/none.json"),
        EXPORT.DEMO.FAULT: EXPORT.read(run / "source_evaluation/inverse_ratio.json"),
    }
    result["source_gate"] = EXPORT.DEMO.source_gate(reports, EXPORT.read(EXPORT.BANDS)["bands"])
    write(run / "source_gate.json", result["source_gate"])
    _rewrite_ledgers(run, result)
    result["conditions"] = {key: True for key in protocol["success_conditions"]}
    write(run / "result.json", result)
    _gpu_evidence(run, result)
    return run


def invalid_fixture(tmp_path: Path) -> Path:
    run = tmp_path / "invalid_run"
    protocol = EXPORT.read(EXPORT.PROTOCOL)
    commit = str(EXPORT.V2.git(
        "log", "-1", "--format=%H", "--", EXPORT.PROTOCOL.relative_to(ROOT).as_posix()
    ))
    protocol_sha = EXPORT.sha256(EXPORT.PROTOCOL, True)
    result = {
        "schema": "ppo_ratio_repair_demo_result_v3",
        "tag": EXPORT.TAG,
        "seed": 2164,
        "restart_seed": 3164,
        "experiment_commit": commit,
        "protocol_sha256_lf": protocol_sha,
        "conditions": {
            "committed_source_and_protocol": True,
            "fresh_seeds": True,
            "launch_paths_checked": True,
            "costs_recorded": False,
        },
        "outcome": "invalid",
        "model": {},
        "probe_steps": [],
        "costs": {},
        "api_requests": 0,
        "api_requests_in_saved_records": 0,
        "pending_model_records": [],
        "transport_errors": [],
        "demo_success": False,
        "wording": protocol["outcome_wording"]["invalid"],
        "design_coupling": protocol["design_coupling"],
    }
    write(run / "begin.json", {
        "tag": EXPORT.TAG,
        "experiment_commit": commit,
        "protocol_sha256_lf": protocol_sha,
        "approver": "fixture-operator",
        "identity_attested": False,
        "delegation": "승인된 등록 검사와 교정을 순서대로 실행하는 운영자 위임",
    })
    write(run / "seed_freshness.json", {
        "schema": "preregistered_seed_freshness_v1",
        "status": "fresh",
        "seeds": [2164, 3164],
        "evidence": [],
    })
    write(run / "launch_paths.json", {"processes_started": 0, "gpu_queries": 0})
    write(run / "result.json", result)
    return run


def partial_source_training_fixture(tmp_path: Path) -> Path:
    run = complete_fixture(tmp_path)
    for relative in (
        "training/inverse_ratio", "source_evaluation", "reference",
        f"probe_1_{EXPORT.CONTRACT.GRADIENT_PROBE}", "correction_training", "correction_evaluation", "model",
    ):
        shutil.rmtree(run / relative)
    for relative in (
        "artifact_none.json", f"artifact_{EXPORT.DEMO.FAULT.lower()}.json", "artifact_corrected.json",
        "source_gate.json", "probes.ledger.jsonl", "correction.ledger.jsonl", "loop_01.json", "recovery.json",
    ):
        (run / relative).unlink()
    result = EXPORT.read(run / "result.json")
    result.update(
        outcome="invalid",
        model={},
        probe_steps=[],
        costs={},
        failure={
            "stage": "source_training",
            "type": "RuntimeError",
            "site": {"file": "evals/ppo_ratio_repair_execution_v3.py", "line": 152},
        },
        api_requests=0,
        api_requests_in_saved_records=0,
        demo_success=False,
        wording=EXPORT.read(EXPORT.PROTOCOL)["outcome_wording"]["invalid"],
    )
    result.pop("loop", None)
    result.pop("recovery", None)
    result.pop("source_gate", None)
    result["conditions"] = {
        "committed_source_and_protocol": True,
        "fresh_seeds": True,
        "launch_paths_checked": True,
        "costs_recorded": False,
    }
    for name, wall_s in (("checkpoint_execution.json", 3.0), ("extract_execution.json", 2.0)):
        write(run / "training/none" / name, {
            "natural_exit": True,
            "timed_out": False,
            "exit_code": 0,
            "cleanup_verified": True,
            "wall_s": wall_s,
        })
    write(run / "training/none/telemetry.json", {
        "run_dir_name": "fixture-none",
        "summary": {"Train/mean_reward": {"n": 300, "first": 0.0, "last": 1.0}},
    })
    write(run / "training/none/gpu_after.json", rejected_receipt())
    write(run / "result.json", result)
    return run


def test_complete_fixture_reuses_v2_boundary_and_exports_v3_schemas(tmp_path):
    original = (BASE.TAG, BASE.PROTOCOL, deepcopy(BASE.EXPECTED_ARMS))
    run = complete_fixture(tmp_path)
    payloads = EXPORT.verify_and_build(run)
    assert set(payloads) == EXPORT.PUBLIC_FILES
    assert payloads["summary.json"]["outcome"] == "recovered"
    assert payloads["summary.json"]["gpu_admission_evidence"] == {
        "validated_receipts": 14,
        "initial_clear": 12,
        "resampled_clear": 2,
        "rejected": 0,
        "used_for_performance_claim": False,
    }
    assert all(value["schema"].endswith("_v3") for value in payloads.values())
    assert any(key.startswith("gpu_receipt:") for key in payloads["provenance.json"]["source_artifact_sha256"])
    assert original == (BASE.TAG, BASE.PROTOCOL, BASE.EXPECTED_ARMS)


def test_invalid_result_is_exported_without_inventing_gpu_or_stage_evidence(tmp_path):
    payloads = EXPORT.verify_and_build(invalid_fixture(tmp_path))
    assert payloads["summary.json"]["outcome"] == "invalid"
    assert payloads["summary.json"]["gpu_admission_evidence"]["validated_receipts"] == 0
    assert payloads["evaluations.json"]["arms"] == {}


def test_source_training_gpu_rejection_keeps_raw_training_partial_and_invalid(tmp_path):
    run = partial_source_training_fixture(tmp_path)
    payloads = EXPORT.verify_and_build(run)
    summary = payloads["summary.json"]
    assert summary["outcome"] == "invalid"
    assert summary["failure"]["stage"] == "source_training"
    assert summary["gpu_admission_evidence"]["rejected"] == 1
    assert summary["execution_costs"]["partial_unaccepted_stage_parent_wall_s"] == {
        "owned_training_parent_wall_s": 1.0,
        "checkpoint_readback_parent_wall_s": 3.0,
        "telemetry_extract_parent_wall_s": 2.0,
    }
    assert summary["execution_costs"]["known_total_parent_wall_s"] == 6.0
    assert payloads["evaluations.json"]["arms"] == {}
    provenance = payloads["provenance.json"]["source_artifact_sha256"]
    assert "training_records:NONE" in provenance
    assert "partial_training_execution:NONE" in provenance


@pytest.mark.parametrize("target", ["records", "checkpoint", "execution"])
def test_partial_source_training_raw_evidence_tampering_is_rejected(tmp_path, target):
    run = partial_source_training_fixture(tmp_path)
    if target == "records":
        path = run / "training/none/records.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        rows[0]["observed"]["optimizer_steps"] = 19
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    elif target == "checkpoint":
        checkpoint = EXPORT.read(run / "training/none/checkpoint.json")
        Path(checkpoint["final"]["path"]).write_bytes(b"tampered")
    else:
        path = run / "training/none/execution.json"
        value = EXPORT.read(path)
        value["cleanup_verified"] = False
        write(path, value)
    with pytest.raises(ValueError):
        EXPORT.verify_and_build(run)


@pytest.mark.parametrize("mutation", ["accepted", "identity", "sample_cap", "strict_clear", "missing"])
def test_gpu_receipt_tampering_is_rejected(tmp_path, mutation):
    run = complete_fixture(tmp_path)
    path = run / "training/none/gpu_before.json"
    value = EXPORT.read(path)
    if mutation == "accepted":
        value["bounded_resample"]["accepted"] = False
    elif mutation == "identity":
        value["bounded_resample"]["identities"][1]["SessionId"] = 2
    elif mutation == "sample_cap":
        value["samples"].append(clear())
        value["bounded_resample"]["sample_count"] = 4
    elif mutation == "strict_clear":
        value["samples"][1]["busy"] = True
    else:
        (run / "training/none/gpu_after.json").unlink()
        with pytest.raises(ValueError, match="inventory"):
            EXPORT.verify_and_build(run)
        return
    write(path, value)
    with pytest.raises(ValueError, match="GPU|CSRSS|resample"):
        EXPORT.verify_and_build(run)


def test_protocol_cap_cost_and_orphan_model_evidence_remain_strict(tmp_path):
    run = complete_fixture(tmp_path)
    protocol = EXPORT.read(EXPORT.PROTOCOL)
    protocol["model"]["total_http_request_cap"] = 5
    changed = tmp_path / "changed_protocol.json"
    write(changed, protocol)
    with pytest.raises(ValueError, match="request cap"):
        EXPORT.verify_and_build(run, changed)

    raw = run / "model/diagnosis.feedback1.json"
    value = EXPORT.read(raw)
    value["calls"][0].pop("input_tokens")
    write(raw, value)
    with pytest.raises(ValueError):
        EXPORT.verify_and_build(run)

    run = complete_fixture(tmp_path / "orphan")
    write(run / "model/diagnosis.review2.json", {})
    with pytest.raises(ValueError, match="orphan model review"):
        EXPORT.verify_and_build(run)


def test_exact_export_bytes_audit_and_overwrite_are_rejected(tmp_path):
    run = complete_fixture(tmp_path)
    payloads = EXPORT.verify_and_build(run)
    output, audit = tmp_path / "public", tmp_path / "audit"
    EXPORT.write_new(payloads, output, audit)
    assert set(EXPORT.verify_export(output, run, EXPORT.PROTOCOL, audit)) == EXPORT.PUBLIC_FILES
    with pytest.raises(FileExistsError):
        EXPORT.write_new(payloads, output, audit)
    with (output / "summary.json").open("a", encoding="utf-8") as handle:
        handle.write(" ")
    with pytest.raises(ValueError, match="artifact differs"):
        EXPORT.verify_export(output, run, EXPORT.PROTOCOL, audit)
