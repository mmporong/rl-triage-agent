from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest  # pyright: ignore[reportMissingImports]

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))
sys.path.insert(0, str(ROOT / "bench"))
import vision_pull_record_source_readiness_v5 as M  # pyright: ignore[reportMissingImports]


def _write(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, bytes):
        path.write_bytes(value)
    else:
        path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _ref(repo: Path, path: Path) -> dict[str, str]:
    return M.MATERIAL._ref(repo, path)


def _descriptor(path: Path) -> dict[str, Any]:
    return {"file": path.name, "sha256": M.MATERIAL._sha(path), "bytes": path.stat().st_size}


def _case(repo: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    base = repo / "evidence"
    checkpoint = _write(base / "checkpoint.pt", b"checkpoint")
    resource = _write(base / "resource.json", {"schema": "brev_resource_proof_v1"})
    artifacts: dict[str, Path] = {}
    leaves = {
        "video": "same_play.mp4",
        "raw_trace": "raw_trace.json",
        "encoder_receipt": "encoder_receipt.json",
        "timing_receipt": "timing_receipt.json",
        "measurement_1": "window_1_measurements.json",
        "spec_1": "window_1_v01_spec.json",
        "measurement_2": "window_2_measurements.json",
        "spec_2": "window_2_v01_spec.json",
    }
    video = _write(base / "output" / leaves["video"], b"video")
    video_sha = M.MATERIAL._sha(video)
    play = {
        "task": "Isaac-Velocity-Flat-Unitree-Go2-Play-v0",
        "camera": "viewer_camera",
        "command_name": "base_velocity",
        "command_frame": "body",
        "motion_frame": "world",
        "settings_sha256": "a" * 64,
    }
    for key, leaf in leaves.items():
        path = base / "output" / leaf
        if key == "video":
            artifacts[key] = video
        elif key.startswith("measurement"):
            artifacts[key] = _write(path, {"schema": "measurement", "play_run_id": "run-1"})
        elif key.startswith("spec"):
            index = 1 if key.endswith("1") else 2
            artifacts[key] = _write(
                path,
                {
                    "schema": M.EVIDENCE.PREPARE_SCHEMA,
                    "play_run_id": "run-1",
                    "source_lineage_id": f"lineage-{index}",
                    "parent_source_lineage_id": None,
                    "video": {"path": "/remote/video.mp4", "logical_id": "media/video.mp4", "play_run_id": "run-1"},
                    "checkpoint": {"path": "/remote/checkpoint.pt", "logical_id": "policy/checkpoint.pt", "play_run_id": "run-1"},
                    "frames": [
                        {"timestamp_s": float(i), "play_run_id": "run-1", "video_sha256": video_sha}
                        for i in range(1, 7)
                    ],
                    "measurement": {"path": f"/remote/measurement-{index}.json", "logical_id": f"measurements/{index}.json", "play_run_id": "run-1"},
                    "play": play,
                },
            )
        else:
            artifacts[key] = _write(path, {"artifact": key})
    stage = base
    job = _write(
        stage / "job.json",
        {
            "schema": M.RUN.JOB.JOB_SCHEMA,
            "mode": M.RUN.JOB.MODE,
            "checkpoint": {"path": "/remote/checkpoint.pt", "sha256": M.MATERIAL._sha(checkpoint), "logical_id": "source-s2220-none"},
            "resource_proof": {"path": "/remote/resource.json", "sha256": M.MATERIAL._sha(resource)},
            "output_directory": "/remote/output",
            "profile": {"fps": 50, "frames": 625},
        },
    )
    manifest = _write(base / "output" / M.WORKER.MANIFEST_NAME, {"manifest": "valid"})
    raw_artifacts = []
    aaba_output = stage / "aaba-output"
    for ordinal in range(4):
        array = np.asarray([ordinal, ordinal + 1], dtype=np.int64)
        npz_path = aaba_output / f"job.annotator-pull-{ordinal:06d}.npz"
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(npz_path, rgb=array)
        npz_descriptor = _descriptor(npz_path)
        receipt = _write(
            aaba_output / f"job.annotator-pull-{ordinal:06d}.json",
            {
                "schema": "vision_annotator_pull_raw_receipt_v1",
                "ordinal": ordinal,
                "complete_capture": True,
                "npz": npz_descriptor,
                "arrays": {
                    "rgb": {
                        "shape": list(array.shape),
                        "dtype": str(array.dtype),
                        "sha256": M.hashlib.sha256(array.tobytes(order="C")).hexdigest(),
                    }
                },
                "semantic_info": {},
                "camera_params_non_array": {},
                "arm": {"ordinal": ordinal},
                "physics_before": {"time": 1.0},
                "physics_after": {"time": 1.0},
                "atomicity_claimed": False,
            },
        )
        raw_artifacts.append(
            {
                "ordinal": ordinal,
                "complete_capture": True,
                "npz": npz_descriptor,
                "receipt": _descriptor(receipt),
            }
        )
    aaba_body = {
        "worker": {
            "policy_probe": {
                "checkpoint_sha256": M.MATERIAL._sha(checkpoint),
                "environment_steps": 1,
            }
        },
        "raw_artifacts": raw_artifacts,
    }
    aaba = _write(aaba_output / M.RUN.AABA_REPORT_NAME, aaba_body)
    gpu_observation = _write(stage / "gpu-observation.json", {"idle": True})
    continuation = _write(
        stage / M.RUN.AABA_CONTINUATION_NAME,
        {
            "schema": "vision_pull_record_phase_continuation_v2",
            "job_sha256": M.MATERIAL._sha(job),
            "checkpoint_sha256": M.MATERIAL._sha(checkpoint),
            "resource_proof_sha256": M.MATERIAL._sha(resource),
            "aaba_report": _descriptor(aaba),
            "aaba_diagnostic_complete": True,
            "gpu_observation": {"path": "/remote/gpu.json", "sha256": M.MATERIAL._sha(gpu_observation)},
            "capture_approved": False,
            "used_for_decision": False,
        },
    )
    shutdown_policy = _write(
        stage / M.RUN.AABA_SHUTDOWN.POLICY_NAME,
        {"schema": M.RUN.AABA_SHUTDOWN.SCHEMA, "completed": True},
    )
    close_entered = _write(
        stage / M.RUN.AABA_SHUTDOWN.CLOSE_ENTERED_NAME,
        {"schema": "vision_aaba_close_entered_v1"},
    )
    close_returned = _write(
        stage / M.RUN.AABA_SHUTDOWN.CLOSE_RETURNED_NAME,
        {"schema": "vision_aaba_close_returned_v1", "returned": True},
    )
    process_exit = _write(
        stage / M.RUN.AABA_SHUTDOWN.PROCESS_EXIT_NAME,
        {
            "schema": "vision_aaba_process_exit_v1",
            "close_returned_observed": True,
            "exit_code": 0,
            "cleanup_verified": True,
            "gpu_after": [],
        },
    )
    continuation_body = _read_json(continuation)
    continuation_body["aaba_process_exit"] = _descriptor(process_exit)
    continuation.write_text(json.dumps(continuation_body), encoding="utf-8")
    worker = {
        "startup": {
            "before": {"orchestrator_enabled": False, "capture_on_play": True},
            "after": {"orchestrator_enabled": True, "capture_on_play": False},
        },
        "artifacts": {key: _descriptor(path) for key, path in artifacts.items()},
        "native_capture_manifest": _descriptor(manifest),
    }
    result_body = {"worker": worker}
    result = _write(base / "output" / "result.json", result_body)
    execution_body = {
        "schema": M.RUN.EXECUTION_SCHEMA,
        "completed": True,
        "technical_recording_complete": True,
        "mode": M.RUN.JOB.MODE,
        "job_sha256": M.MATERIAL._sha(job),
        "exit_code": 0,
        "timed_out": False,
        "cleanup_verified": True,
        "gpu_before": [],
        "gpu_after": [],
        "error_type": None,
        "error_message": None,
        "failure_atom": None,
        "result_sha256": M.MATERIAL._sha(result),
        "wall_s": 1.0,
        "checkpoint_sha256": M.MATERIAL._sha(checkpoint),
        "resource_proof_sha256": M.MATERIAL._sha(resource),
        "per_checkpoint_aaba_required": True,
        "full_recordings": 1,
        "capture_approved": False,
        "model_ready": False,
        "used_for_decision": False,
        "aaba_receipt": {
            "checkpoint_sha256": M.MATERIAL._sha(checkpoint),
            "diagnostic_complete": True,
            "capture_approved": False,
            "used_for_decision": False,
            "report": _descriptor(aaba),
            "shutdown_policy": _descriptor(shutdown_policy),
            "close_entered": _descriptor(close_entered),
            "close_returned": _descriptor(close_returned),
            "process_exit": _descriptor(process_exit),
        },
        "phases": [
            {"name": name, "exit_code": 0, "cleanup_verified": True, "gpu_before": [], "gpu_after": []}
            for name in ("aaba", "record")
        ],
    }
    execution = _write(stage / "execution.json", execution_body)
    inventory = _write(
        base / "inventory.json",
        {
            "schema": M.INVENTORY_SCHEMA,
            "raw_job": _ref(repo, job),
            "checkpoint": _ref(repo, checkpoint),
            "result": _ref(repo, result),
            "execution": _ref(repo, execution),
            "aaba_report": _ref(repo, aaba),
            "aaba_continuation": _ref(repo, continuation),
            "aaba_shutdown_policy": _ref(repo, shutdown_policy),
            "aaba_close_entered": _ref(repo, close_entered),
            "aaba_close_returned": _ref(repo, close_returned),
            "aaba_process_exit": _ref(repo, process_exit),
            "gpu_observation": _ref(repo, gpu_observation),
            "resource_proof": _ref(repo, resource),
            "native_manifest": _ref(repo, manifest),
            "artifacts": {key: _ref(repo, path) for key, path in artifacts.items()},
        },
    )
    telemetry = _write(base / "telemetry.json", {"seed": 2220})
    source_artifact = _write(
        base / "source-training.json",
        {"seed": 2220, "arm": "NONE", "checkpoint": _ref(repo, checkpoint)},
    )
    training = _write(
        base / "training.json",
        {
            "schema": M.MATERIAL.TRAINING_SCHEMA,
            "source_seed": 2220,
            "arm": "NONE",
            "checkpoint": _ref(repo, checkpoint),
            "telemetry": _ref(repo, telemetry),
            "source_training_artifact": _ref(repo, source_artifact),
        },
    )
    source_protocol = _write(
        base / "source-protocol.json",
        {
            "schema": "vision_brev_source_protocol_v1",
            "protocol_id": "source-v1",
            "training_plan": [{"seed": 2220, "arm": "NONE"}],
        },
    )
    fixed = _write(
        base / "fixed.json",
        {
            "kind": "fixed_eval_summary",
            "bands": {"survival_ratio_min": 0.9, "err_xy_ratio_max": 1.2, "err_yaw_ratio_max": 1.2, "fall_frac_max": 0.05},
            "normal": [{"pair": "normal vs normal"}],
            "runs": [{"run": "normal-2220", "seed": 2220, "reference": "normal-2220", "is_reference": True, "fixed_label": "healthy", "checkpoint_sha256": M.MATERIAL._sha(checkpoint)}],
        },
    )
    resource_files = {key: _write(base / f"resource-{key}.json", {"key": key}) for key in M.LEGACY.RESOURCE_BUNDLE_KEYS}
    resource_files["proof"] = resource
    resource_bundle = {key: _ref(repo, path) for key, path in resource_files.items()}
    binding = _write(
        base / "binding.json",
        {
            "schema": M.BINDING_SCHEMA,
            "seed_group": "source-01",
            "source_role": "normal",
            "source_seed": 2220,
            "raw_job": _ref(repo, job),
            "result": _ref(repo, result),
            "execution": _ref(repo, execution),
            "aaba_report": _ref(repo, aaba),
            "aaba_continuation": _ref(repo, continuation),
            "aaba_shutdown_policy": _ref(repo, shutdown_policy),
            "aaba_close_entered": _ref(repo, close_entered),
            "aaba_close_returned": _ref(repo, close_returned),
            "aaba_process_exit": _ref(repo, process_exit),
            "local_remap_inventory": _ref(repo, inventory),
            "resource_bundle": resource_bundle,
            "training_binding": _ref(repo, training),
            "fixed_eval_summary": _ref(repo, fixed),
            "source_protocol": _ref(repo, source_protocol),
        },
    )
    monkeypatch.setattr(M.RUN.V1, "verify_terminal", lambda validated: result_body)
    monkeypatch.setattr(M.RUN.V1, "verify_result", lambda validated: result_body)
    monkeypatch.setattr(M.RUN, "_verify_aaba_terminal", lambda validated: _read_json(aaba))
    shutdown_calls: list[dict[str, Any]] = []

    def verify_shutdown(validated: dict[str, Any]) -> dict[str, Any]:
        shutdown_calls.append(validated)
        return _read_json(shutdown_policy)

    monkeypatch.setattr(M.RUN.AABA_SHUTDOWN, "verify", verify_shutdown)
    monkeypatch.setattr(
        M.RUN.AABA_SHUTDOWN,
        "verify_process_exit",
        lambda validated, observation: _read_json(process_exit),
    )
    monkeypatch.setattr(M.RUN.REMOTE, "verify_observation_reference", lambda *args: {})
    monkeypatch.setattr(M.WORKER, "_manifest", lambda validated, paths: {"manifest": "valid"})
    monkeypatch.setattr(
        M.LEGACY,
        "_resource",
        lambda repo_arg, value, label: (
            {key: M.MATERIAL._reference(repo_arg, value[key], key) for key in M.LEGACY.RESOURCE_BUNDLE_KEYS},
            {
                "protocol_sha256": M.LEGACY._sha_lf(source_protocol),
                "resource_id": "resource-1",
                "gpu_uuid": "GPU-1",
            },
        ),
    )
    return {
        "binding": binding,
        "inventory": inventory,
        "checkpoint": checkpoint,
        "aaba": aaba,
        "result": result,
        "execution": execution,
        "shutdown_policy": shutdown_policy,
        "close_entered": close_entered,
        "close_returned": close_returned,
        "process_exit": process_exit,
        "shutdown_calls": shutdown_calls,
        "artifacts": artifacts,
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_run5_reuses_execution_v3_and_shutdown_v1_wire() -> None:
    assert M.RUN.EXECUTION_SCHEMA == "vision_pull_record_execution_v3"
    assert M.RUN.AABA_SHUTDOWN.SCHEMA == "vision_aaba_shutdown_policy_v1"


def test_minimal_registered_input_builds_and_translates_only_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _case(tmp_path, monkeypatch)
    ready_path = tmp_path / "ready.json"
    value = M.build_readiness(tmp_path, case["binding"], ready_path)
    assert value["status"] == "READY"
    assert value["outputs"] == 2
    assert value["proof"]["native_pull_window_updates"] == 0
    assert len(case["shutdown_calls"]) == 1
    assert value["aaba_shutdown_policy"]["sha256"] == M.MATERIAL._sha(
        case["shutdown_policy"]
    )

    translated = M.derive_local_specs(tmp_path, ready_path, tmp_path / "translated")
    assert translated["status"] == "READY"
    assert translated["path_fields_changed_only"] == ["video.path", "checkpoint.path", "measurement.path"]
    original = _read_json(case["artifacts"]["spec_1"])
    derived = _read_json(tmp_path / "translated" / "window_1_local_spec.json")
    for key in ("video", "checkpoint", "measurement"):
        assert derived[key]["path"] != original[key]["path"]
        patched = copy.deepcopy(derived)
        patched[key]["path"] = original[key]["path"]
        comparison = copy.deepcopy(original)
        for other in {"video", "checkpoint", "measurement"} - {key}:
            comparison[other]["path"] = derived[other]["path"]
        assert patched == comparison


def test_missing_input_yields_hold_without_fake_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _case(tmp_path, monkeypatch)
    case["artifacts"]["video"].unlink()
    value = M.build_readiness(tmp_path, case["binding"], tmp_path / "hold.json")
    assert value == {
        "schema": M.SCHEMA,
        "status": "HOLD",
        "outputs": 0,
        "model_use_ready": False,
        "capture_approved": False,
        "used_for_decision": False,
        "reason": "missing_inputs",
        "missing": "same_play.mp4",
    }


def test_missing_process_exit_receipt_yields_hold_without_fake_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _case(tmp_path, monkeypatch)
    case["process_exit"].unlink()
    value = M.build_readiness(tmp_path, case["binding"], tmp_path / "hold.json")
    assert value["status"] == "HOLD"
    assert value["outputs"] == 0
    assert value["model_use_ready"] is False
    assert value["missing"] == M.RUN.AABA_SHUTDOWN.PROCESS_EXIT_NAME


def test_execution_shutdown_descriptor_tamper_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _case(tmp_path, monkeypatch)
    execution = _read_json(case["execution"])
    execution["aaba_receipt"]["close_returned"]["sha256"] = "0" * 64
    case["execution"].write_text(json.dumps(execution), encoding="utf-8")
    inventory = _read_json(case["inventory"])
    inventory["execution"] = _ref(tmp_path, case["execution"])
    case["inventory"].write_text(json.dumps(inventory), encoding="utf-8")
    binding = _read_json(case["binding"])
    binding["execution"] = _ref(tmp_path, case["execution"])
    binding["local_remap_inventory"] = _ref(tmp_path, case["inventory"])
    case["binding"].write_text(json.dumps(binding), encoding="utf-8")
    with pytest.raises(ValueError, match="부모 실행 증거"):
        M.build_readiness(tmp_path, case["binding"], tmp_path / "ready.json")


def test_shutdown_accepts_unobserved_python_close_return_with_parent_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage = tmp_path / "stage"
    job = _write(stage / "job.json", {})
    policy = _write(stage / M.RUN.AABA_SHUTDOWN.POLICY_NAME, {"policy": True})
    entered = _write(stage / M.RUN.AABA_SHUTDOWN.CLOSE_ENTERED_NAME, {"entered": True})
    process_exit = _write(
        stage / M.RUN.AABA_SHUTDOWN.PROCESS_EXIT_NAME,
        {"close_returned_observed": False, "exit_code": 0, "cleanup_verified": True},
    )
    validated = {"job_path": job}
    monkeypatch.setattr(M.RUN, "_aaba_validated", lambda value: value)
    monkeypatch.setattr(M.RUN.AABA_SHUTDOWN, "verify", lambda value: _read_json(policy))
    monkeypatch.setattr(
        M.RUN.AABA_SHUTDOWN,
        "verify_process_exit",
        lambda value, observation: _read_json(process_exit),
    )

    receipt = M._shutdown(
        validated, policy, entered, None, process_exit, {"sha256": "a" * 64}
    )

    assert receipt["process_exit"]["close_returned_observed"] is False


def test_tampered_reference_sha_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _case(tmp_path, monkeypatch)
    binding = _read_json(case["binding"])
    binding["result"]["sha256"] = "0" * 64
    case["binding"].write_text(json.dumps(binding), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA"):
        M.build_readiness(tmp_path, case["binding"], tmp_path / "ready.json")


def test_checkpoint_mismatch_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _case(tmp_path, monkeypatch)
    other = _write(tmp_path / "evidence" / "other.pt", b"other")
    inventory = _read_json(case["inventory"])
    inventory["checkpoint"] = _ref(tmp_path, other)
    case["inventory"].write_text(json.dumps(inventory), encoding="utf-8")
    binding = _read_json(case["binding"])
    binding["local_remap_inventory"] = _ref(tmp_path, case["inventory"])
    case["binding"].write_text(json.dumps(binding), encoding="utf-8")
    with pytest.raises(ValueError, match="checkpoint SHA"):
        M.build_readiness(tmp_path, case["binding"], tmp_path / "ready.json")


def test_invalid_aaba_embedded_checkpoint_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _case(tmp_path, monkeypatch)
    monkeypatch.setattr(
        M.RUN,
        "_verify_aaba_terminal",
        lambda validated: {"worker": {"policy_probe": {"checkpoint_sha256": "f" * 64}}},
    )
    with pytest.raises(ValueError, match="embedded checkpoint"):
        M.build_readiness(tmp_path, case["binding"], tmp_path / "ready.json")


def test_aaba_npz_byte_tamper_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _case(tmp_path, monkeypatch)
    report = _read_json(case["aaba"])
    npz_path = case["aaba"].parent / report["raw_artifacts"][0]["npz"]["file"]
    npz_path.write_bytes(npz_path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="raw 원문 SHA"):
        M.build_readiness(tmp_path, case["binding"], tmp_path / "ready.json")


def test_continuation_checkpoint_mismatch_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _case(tmp_path, monkeypatch)
    binding = _read_json(case["binding"])
    continuation_path = M.MATERIAL._reference(
        tmp_path, binding["aaba_continuation"], "continuation"
    )
    continuation = _read_json(continuation_path)
    continuation["checkpoint_sha256"] = "f" * 64
    continuation_path.write_text(json.dumps(continuation), encoding="utf-8")
    inventory = _read_json(case["inventory"])
    inventory["aaba_continuation"] = _ref(tmp_path, continuation_path)
    case["inventory"].write_text(json.dumps(inventory), encoding="utf-8")
    binding["aaba_continuation"] = _ref(tmp_path, continuation_path)
    binding["local_remap_inventory"] = _ref(tmp_path, case["inventory"])
    case["binding"].write_text(json.dumps(binding), encoding="utf-8")
    with pytest.raises(ValueError, match="연속 receipt"):
        M.build_readiness(tmp_path, case["binding"], tmp_path / "ready.json")


def test_measurement_from_another_execution_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _case(tmp_path, monkeypatch)
    other = _write(tmp_path / "evidence" / "other_measurement.json", {"other": True})
    inventory = _read_json(case["inventory"])
    inventory["artifacts"]["measurement_1"] = _ref(tmp_path, other)
    case["inventory"].write_text(json.dumps(inventory), encoding="utf-8")
    binding = _read_json(case["binding"])
    binding["local_remap_inventory"] = _ref(tmp_path, case["inventory"])
    case["binding"].write_text(json.dumps(binding), encoding="utf-8")
    with pytest.raises(ValueError, match="local artifact"):
        M.build_readiness(tmp_path, case["binding"], tmp_path / "ready.json")


@pytest.mark.parametrize(
    "clue",
    [
        {"source_role": "normal"},
        {"content": [{"type": "text", "text": "fixed label: healthy"}]},
        {"content": [{"type": "text", "text": "C:\\private\\video.mp4"}]},
        {"content": [{"type": "text", "text": "a" * 64}]},
    ],
)
def test_model_view_leakage_clues_are_rejected(clue: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="단서"):
        M._assert_public_view_safe(clue)


def test_public_model_view_shape_without_clues_is_allowed() -> None:
    M._assert_public_view_safe(
        {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "걷는가?"}]},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAAA"}},
            ]
        }
    )


def test_prepare_contracts_reuses_v01_contract_and_checks_public_view(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _case(tmp_path, monkeypatch)
    ready_path = tmp_path / "ready.json"
    M.build_readiness(tmp_path, case["binding"], ready_path)
    translated_dir = tmp_path / "translated"
    M.derive_local_specs(tmp_path, ready_path, translated_dir)
    calls: list[str] = []

    def fake_prepare(spec: Path, output: Path) -> Path:
        calls.append(Path(spec).name)
        return _write(output / "contract.json", {"schema": M.EVIDENCE.CONTRACT_SCHEMA})

    monkeypatch.setattr(M.EVIDENCE, "prepare", fake_prepare)
    monkeypatch.setattr(
        M.EVIDENCE,
        "public_model_view",
        lambda contract: {
            "schema": M.EVIDENCE.MODEL_VIEW_SCHEMA,
            "messages": [{"role": "user", "content": [{"type": "text", "text": "걷는가?"}]}],
        },
    )
    receipt = M.prepare_contracts(
        tmp_path,
        translated_dir / "translation_inventory.json",
        tmp_path / "contracts",
    )
    assert receipt["status"] == "READY"
    assert receipt["outputs"] == 2
    assert calls == ["window_1_local_spec.json", "window_2_local_spec.json"]


def test_source_pair_fixed_eval_binds_actual_arm_reference_and_bands(tmp_path: Path) -> None:
    checkpoint_sha = "b" * 64
    bands = _write(
        tmp_path / "bands.json",
        {
            "bands": {"survival_ratio_min": 0.9, "err_xy_ratio_max": 1.2, "err_yaw_ratio_max": 1.2, "fall_frac_max": 0.05},
            "normal": [{"pair": "normal"}],
        },
    )
    summary = _write(
        tmp_path / "source-summary.json",
        {
            "schema": "vision_source_prefix_public_result_v1",
            "status": "source_training_and_fixed_evaluation_complete",
            "training": [
                {"job_id": "vision-s2220-none", "seed": 2220, "arm": "NONE", "checkpoint_sha256": "a" * 64},
                {"job_id": "vision-s2220-f1", "seed": 2220, "arm": "F1", "checkpoint_sha256": checkpoint_sha},
            ],
            "evaluations": [
                {
                    "seed": 2220,
                    "metrics": {"NONE": {"error": 0.1}, "F1": {"error": 1.2}},
                    "source_gate": {"verdict": {"label": "unhealthy"}},
                }
            ],
            "scope": {"fixed_evaluation_is_behavior_label": True},
        },
    )
    path, receipt = M._fixed_eval(
        tmp_path,
        _ref(tmp_path, summary),
        2220,
        "F1",
        "unhealthy",
        checkpoint_sha,
        {
            "evaluation": {
                "bands_path": bands.relative_to(tmp_path).as_posix(),
                "bands_sha256": M.MATERIAL._sha(bands),
            }
        },
    )
    assert path == summary
    assert receipt["run"] == "vision-s2220-f1"
    assert receipt["reference"] == "vision-s2220-none"
    assert receipt["label"] == "unhealthy"
