"""영상 선택 원본 helper의 실제 V04→V01→capsule 결합 CPU 회귀."""
# pyright: reportMissingImports=false
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from copy import deepcopy
from pathlib import Path, PurePosixPath

import pytest

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / "tests", ROOT / "bench", ROOT / "evals", ROOT / "src"):
    sys.path.insert(0, str(folder))

import record_vision_evidence_v1 as RECORD
import run_vision_brev_batch_v1 as BATCH
import test_record_vision_evidence_v1 as RECORD_FIXTURE
import test_run_vision_brev_batch_v1 as BATCH_FIXTURE
import test_vision_selection_contract_v1 as SELECTION_FIXTURE
import vision_remote_gpu_v1 as REMOTE
import vision_render_context_v1 as CONTEXT
import vision_selection_call_v1 as CALL
import vision_selection_comparison_v1 as COMPARISON
import vision_selection_sources_v1 as SOURCES


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _sha(path: Path, *, lf: bool = False) -> str:
    raw = path.read_bytes().replace(b"\r\n", b"\n") if lf else path.read_bytes()
    return hashlib.sha256(raw).hexdigest()


def _ref(repo: Path, path: Path, *, frozen: bool = False) -> dict[str, str]:
    value = {"path": path.resolve().relative_to(repo.resolve()).as_posix(), "sha256": _sha(path)}
    if frozen:
        value["status"] = "frozen"
    return value


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True,
        text=True, encoding="utf-8", errors="strict",
    ).stdout.strip()


def _artifact(path: Path) -> dict[str, object]:
    return {"file": path.name, "sha256": _sha(path), "bytes": path.stat().st_size}


def _observation(repo: Path, folder: Path, ordinal: int) -> dict[str, str]:
    folder.mkdir(parents=True, exist_ok=True)
    prefix = f"{ordinal:06d}"
    gpu, process = folder / f"{prefix}.gpu.csv", folder / f"{prefix}.process.csv"
    gpu.write_text("GPU-fixture\n", encoding="utf-8", newline="")
    process.write_text("", encoding="utf-8", newline="")
    receipt = folder / f"{prefix}.receipt.json"
    _write(receipt, {
        "schema": "vision_remote_gpu_observation_v1", "ordinal": ordinal,
        "resource_id": "brev-instance-1", "gpu_uuid": "GPU-fixture",
        "gpu_command": REMOTE.GPU_COMMAND, "gpu_exit_code": 0,
        "gpu_stdout_file": gpu.name, "gpu_stdout_sha256": _sha(gpu), "gpu_stderr": "",
        "process_command": REMOTE.PROCESS_COMMAND, "process_exit_code": 0,
        "process_stdout_file": process.name, "process_stdout_sha256": _sha(process), "process_stderr": "",
    })
    return _ref(repo, receipt)


def _source_snapshot(protocol: dict, protocol_path: Path, anchor: str) -> dict:
    packages = {
        "isaacsim": "4.5.0.0", "isaaclab": "0.41.3", "isaaclab_rl": "0.1.8",
        "torch": "2.7.0+cu128", "rsl-rl-lib": "2.3.3", "gymnasium": "1.2.0",
    }
    environment = {
        "python": "3.10.12", "isaaclab_sha": "a" * 40, **packages,
        "cuda_available": True, "cuda_device_count": 1,
    }
    return {
        "lab_head": "a" * 40, "lab_tracked_clean": True, "source_anchor": anchor,
        "protocol": {"path": protocol_path.as_posix(), "sha256_lf": protocol["_verified_sha256_lf"]},
        "environment": environment,
        "selected_interpreter": {
            "launcher": "/opt/vision/.venv/bin/python", "python": "3.10.12",
            "executable": "/opt/vision/.venv/bin/python", "prefix": "/opt/vision/.venv",
            "base_prefix": "/usr", "packages": packages,
        },
        "source_sha256_lf": protocol["source_sha256_lf"],
        "runtime_source_sha256_lf": protocol["runtime_source_sha256_lf"],
        "python_package_source_sha256_lf": protocol["python_package_source_sha256_lf"],
    }


def _context(repo: Path, folder: Path, job: dict, snapshot: dict, job_sha: str) -> Path:
    folder.mkdir(parents=True)
    gpu, torch = folder / CONTEXT.GPU_FILE, folder / CONTEXT.TORCH_FILE
    gpu.write_text("GPU-fixture, NVIDIA L40S, 570.0\n", encoding="utf-8", newline="")
    torch.write_text(json.dumps({
        "version": "2.7.0+cu128", "built_cuda_version": "12.8",
        "device_name": "NVIDIA L40S", "compute_capability": [8, 9],
        "device_count": 1, "cuda_available": True,
    }, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    receipt = folder / CONTEXT.RECEIPT_FILE
    _write(receipt, {
        "schema": CONTEXT.SCHEMA, "job_sha256": job_sha,
        "source_protocol_sha256_lf": snapshot["protocol"]["sha256_lf"],
        "source_snapshot_sha256": CONTEXT._digest(snapshot), "source_snapshot": snapshot,
        "captured_at_utc": "2027-01-01T01:01:00+00:00",
        "finished_at_utc": "2027-01-01T01:02:00+00:00",
        "gpu_raw": {"argv": CONTEXT.GPU_COMMAND, "stdout_file": gpu.name,
                    "stdout_sha256": _sha(gpu), "exit_code": 0, "stderr": ""},
        "torch_raw": {"argv": [snapshot["selected_interpreter"]["launcher"], "-I", "-c", CONTEXT.TORCH_SCRIPT],
                      "stdout_file": torch.name, "stdout_sha256": _sha(torch),
                      "exit_code": 0, "stderr": ""},
    })
    CONTEXT.read_context(receipt, job, job_sha, snapshot)
    return receipt


def _smoke_artifacts(folder: Path) -> dict[str, Path]:
    captures = [
        RECORD_FIXTURE._smoke_capture("warmup", 0),
        RECORD_FIXTURE._smoke_capture("A", 1),
        RECORD_FIXTURE._smoke_capture("B", 2),
        RECORD_FIXTURE._smoke_capture("A", 1),
    ]
    captures[2]["camera"] = {
        "eye_world_m": [-3.0, 3.0, 2.5],
        "lookat_world_m": [0.0, 0.0, 0.85],
    }
    report = RECORD.assess_render_sync_smoke(captures)
    folder.mkdir(parents=True)
    report_path = folder / "render_sync_smoke.json"
    _write(report_path, report)
    outputs = {"render_sync_smoke": report_path}
    for index, capture in enumerate(captures[1:], 1):
        path = folder / f"audit_rgb_{index}.rgb"
        path.write_bytes(capture["rgb"])
        outputs[f"audit_rgb_{index}"] = path
    return outputs


def _capture(
    repo: Path,
    folder: Path,
    *,
    mode: str,
    job_id: str,
    checkpoint: Path,
    checkpoint_logical_id: str,
    anchor: str,
    source_protocol: dict,
    source_protocol_path: Path,
    resource: dict,
    resource_refs: dict[str, dict[str, str]],
    video_source: Path,
    records: list[dict],
    ordinal: int,
) -> tuple[dict[str, dict[str, str]], dict[str, Path], dict]:
    folder.mkdir(parents=True)
    remote_output = f"/task/{job_id}/output"
    protocol_path = folder / "record_protocol.json"
    record_protocol = {
        "schema": RECORD.PROTOCOL_SCHEMA, "method": RECORD.METHOD, "mode": mode,
        "profile": RECORD.fixed_profile(), "source_commit": anchor,
        "source_files_lf": source_protocol["recording"]["source_files_lf"],
        "checkpoint_sha256": _sha(checkpoint),
        "resource": {"resource_id": "brev-instance-1", "gpu_uuid": "GPU-fixture"},
    }
    _write(protocol_path, record_protocol)
    job = {
        "schema": RECORD.JOB_SCHEMA, "method": RECORD.METHOD, "mode": mode, "job_id": job_id,
        "profile": RECORD.fixed_profile(),
        "source": {"repo_root": "/task/repo", "commit": anchor,
                   "files_lf": source_protocol["recording"]["source_files_lf"]},
        "protocol": {"path": f"{remote_output}/record_protocol.json", "sha256": _sha(protocol_path)},
        "checkpoint": {"path": f"/task/checkpoints/{checkpoint.name}", "sha256": _sha(checkpoint),
                       "logical_id": checkpoint_logical_id},
        "resource": {"resource_id": "brev-instance-1", "gpu_uuid": "GPU-fixture"},
        "remote_gpu_admission": {"path": f"{remote_output}/admission.json", "sha256": "0" * 64,
                                 "resource_parent_sha256": resource_refs["resource_parent"]["sha256"],
                                 "live_lease_proof_sha256": resource_refs["live_lease_proof"]["sha256"]},
        "output_directory": remote_output,
    }
    admission_path = folder / "admission.json"
    admission = {
        "schema": RECORD.ADMISSION_SCHEMA, "status": "granted", "job_id": job_id,
        "resource_kind": "remote_gpu", "lease_id": resource["lease_id"],
        "request_binding": RECORD.request_binding(job),
        "issued_at_utc": "2027-01-01T01:00:00+00:00",
        "expires_at_utc": resource["expires_at_utc"],
        "resource_id": "brev-instance-1", "gpu_uuid": "GPU-fixture",
        "resource_parent_sha256": resource_refs["resource_parent"]["sha256"],
        "live_lease_proof_sha256": resource_refs["live_lease_proof"]["sha256"],
    }
    _write(admission_path, admission)
    job["remote_gpu_admission"]["sha256"] = _sha(admission_path)
    job_path = folder / "job.json"
    _write(job_path, job)
    snapshot = _source_snapshot(
        source_protocol, source_protocol_path.relative_to(repo), anchor,
    )
    context_path = _context(repo, folder / "context", job, snapshot, _sha(job_path))
    before = _observation(repo, folder / "gpu", ordinal * 2 - 1)
    after = _observation(repo, folder / "gpu", ordinal * 2)

    if mode == RECORD.RECORD_MODE:
        video = folder / "same_play.mp4"
        shutil.copyfile(video_source, video)
        timing = RECORD.probe_video_timing(video)
        artifacts = RECORD.build_trace_and_v01_specs(
            {"job": job, "checkpoint_sha256": _sha(checkpoint), "checkpoint_path": checkpoint},
            video, records,
            {"natural_exit": True, "returncode": 0, "frames_written": RECORD.FRAME_COUNT,
             "bytes_written": RECORD.FRAME_COUNT * RECORD.WIDTH * RECORD.HEIGHT * 3},
            timing, folder / "evidence",
        )
        for number in (1, 2):
            spec_path = artifacts[f"spec_{number}"]
            spec = json.loads(spec_path.read_text(encoding="utf-8"))
            spec["video"]["path"] = f"{remote_output}/same_play.mp4"
            spec["checkpoint"]["path"] = job["checkpoint"]["path"]
            spec["measurement"]["path"] = f"{remote_output}/evidence/{artifacts[f'measurement_{number}'].name}"
            _write(spec_path, spec)
        primary = video
    else:
        artifacts = _smoke_artifacts(folder / "smoke")
        primary = artifacts["render_sync_smoke"]
        video = None

    result = {
        "schema": RECORD.RESULT_SCHEMA, "completed": True,
        "job_sha256": _sha(job_path), "protocol_sha256": _sha(protocol_path),
        "checkpoint_sha256": _sha(checkpoint), "request_binding": RECORD.request_binding(job),
        "resource": job["resource"],
        "resource_parent_sha256": resource_refs["resource_parent"]["sha256"],
        "live_lease_proof_sha256": resource_refs["live_lease_proof"]["sha256"],
        "admission_sha256": _sha(admission_path),
        "admission_issued_at_utc": admission["issued_at_utc"],
        "admission_expires_at_utc": admission["expires_at_utc"],
        "profile_sha256": RECORD.digest(RECORD.fixed_profile()), "mode": mode,
        "primary_artifact_sha256": _sha(primary), "video_sha256": _sha(video) if video else None,
        "artifacts": {key: _artifact(path) for key, path in artifacts.items()},
        "fixed_evaluation_used_as_behavior_ground_truth": False,
        "independent_visual_evidence_source_count": 1,
        "zero_update_pixel_lag_confirmed": False,
        "gpu_smoke_required_before_model_use": mode == RECORD.RECORD_MODE,
    }
    result_path = folder / "result.json"
    _write(result_path, result)
    execution_path = folder / "execution.json"
    _write(execution_path, {
        "schema": "vision_brev_record_execution_v1", "completed": True, "exit_code": 0,
        "timed_out": False, "cleanup_verified": True, "job_sha256": _sha(job_path),
        "result_sha256": _sha(result_path), "gpu_before": [], "gpu_after": [],
        "source_before": snapshot, "source_after": snapshot,
        "gpu_before_observation": before, "gpu_after_observation": after,
        "launch": {"argv": ["fixture"]}, "wall_s": 1.0,
    })
    attestation_path = folder / "attestation.json"
    _write(attestation_path, {
        "schema": "vision_render_attestation_v1", "completed": True,
        "job_sha256": _sha(job_path),
        "wrapper_sha256_lf": source_protocol["recording"]["source_files_lf"][
            "bench/vision_render_attestation_v1.py"
        ],
        "context": {"path": "context/context.receipt.json", "sha256": _sha(context_path)},
        "execution": {"path": str(PurePosixPath(remote_output).parent / "execution.json"),
                      "sha256": _sha(execution_path)},
        "result": {"path": f"{remote_output}/result.json", "sha256": _sha(result_path)},
        "source_snapshot_sha256": CONTEXT._digest(snapshot),
        "execute_started_at_utc": "2027-01-01T01:03:00+00:00",
        "execute_finished_at_utc": "2027-01-01T01:04:00+00:00",
        "attested_at_utc": "2027-01-01T01:05:00+00:00",
        "phase_order": ["source_before", "context", "execute", "source_after", "seal"],
    })
    refs = {
        "job": _ref(repo, job_path), "protocol": _ref(repo, protocol_path),
        "admission": _ref(repo, admission_path), "result": _ref(repo, result_path),
        "execution": _ref(repo, execution_path), "context": _ref(repo, context_path),
        "attestation": _ref(repo, attestation_path),
        "gpu_observation_before": before, "gpu_observation_after": after,
        **{key: _ref(repo, path) for key, path in artifacts.items()},
    }
    if video is not None:
        refs["video"] = _ref(repo, video)
    return refs, artifacts, snapshot


def _resource(repo: Path, source_protocol_sha: str) -> tuple[dict, dict[str, dict[str, str]]]:
    folder = repo / "artifacts/resource"
    folder.mkdir(parents=True)
    provider, gpu, process = folder / "provider.json", folder / "gpu.csv", folder / "process.csv"
    provider_row: dict[str, object] = {key: None for key in REMOTE.PROVIDER_ROW_KEYS}
    provider_row.update({"id": "brev-instance-1", "name": "fixture", "status": "RUNNING"})
    _write(provider, {"workspaces": [provider_row]})
    gpu.write_text("GPU-fixture\n", encoding="utf-8", newline="")
    process.write_text("", encoding="utf-8", newline="")
    common = {
        "status": "granted", "resource_id": "brev-instance-1", "gpu_uuid": "GPU-fixture",
        "lease_id": "lease-1", "issued_at_utc": "2027-01-01T00:00:00+00:00",
        "expires_at_utc": "2027-01-02T00:00:00+00:00",
        "compute_started_at_utc": "2027-01-01T00:00:00+00:00",
        "protocol_sha256": source_protocol_sha,
        "provider_inventory_sha256": _sha(provider), "nvidia_gpu_csv_sha256": _sha(gpu),
        "nvidia_process_csv_sha256": _sha(process),
    }
    raw, budget = folder / "raw_gpu.json", folder / "budget.json"
    _write(raw, {"schema": BATCH.RAW_GPU_RECEIPT_SCHEMA, **common})
    _write(budget, {
        "schema": BATCH.BUDGET_ADMISSION_SCHEMA, **common,
        "whole_batch_budget_usd": 50.0, "compute_quote_usd_per_h": 2.0,
        "max_owned_instances": 1,
        "cost_scope": "owned_instance_elapsed_wall_includes_idle_setup_transfer_excludes_storage_v1",
    })
    proof = {
        "schema": BATCH.RESOURCE_SCHEMA, "status": "granted", "resource_kind": "remote_gpu",
        "lease_id": "lease-1", "protocol_sha256": source_protocol_sha,
        "issued_at_utc": common["issued_at_utc"], "expires_at_utc": common["expires_at_utc"],
        "compute_started_at_utc": common["compute_started_at_utc"],
        "resource_id": "brev-instance-1", "gpu_uuid": "GPU-fixture",
        "raw_gpu_receipt_path": "/task/resource/raw_gpu.json", "raw_gpu_receipt_sha256": _sha(raw),
        "budget_admission_path": "/task/resource/budget.json", "budget_admission_sha256": _sha(budget),
        "provider_inventory_path": "/task/resource/provider.json", "provider_inventory_sha256": _sha(provider),
        "nvidia_gpu_csv_path": "/task/resource/gpu.csv", "nvidia_gpu_csv_sha256": _sha(gpu),
        "nvidia_process_csv_path": "/task/resource/process.csv", "nvidia_process_csv_sha256": _sha(process),
    }
    proof_path = folder / "resource.json"
    _write(proof_path, proof)
    refs = {
        "resource_parent": _ref(repo, proof_path), "live_lease_proof": _ref(repo, raw),
        "budget_admission": _ref(repo, budget), "provider_inventory": _ref(repo, provider),
        "gpu_csv": _ref(repo, gpu), "process_csv": _ref(repo, process),
    }
    return proof, refs


def _training(
    repo: Path,
    folder: Path,
    *,
    checkpoint: Path,
    arm: str,
    seed: int,
    job_id: str,
    action_id: str,
    action_ordinal: int,
    source_protocol: dict,
    source_protocol_path: Path,
    anchor: str,
    resource_ref: dict[str, str],
) -> tuple[Path, dict[str, dict[str, str]]]:
    folder.mkdir(parents=True)
    artifact = {
        "checkpoint": {"path": f"/task/checkpoints/{checkpoint.name}", "sha256": _sha(checkpoint)},
        "telemetry": {"path": "/task/telemetry.json", "sha256": "4" * 64, "summary": {}},
        "seed": seed, "arm": arm, "original_faults": [] if arm == "NONE" else ["F1"],
        "applied_corrections": [], "readback": {},
        "execution": {"completed": True, "exit_code": 0, "timed_out": False, "cleanup_verified": True},
        "manifest_path": "/task/training/manifest.json", "manifest_sha256": "5" * 64,
        "manifest": {"job_id": job_id}, "source_checkpoint_sha256": None,
    }
    artifact_path = folder / "training_artifact.json"
    _write(artifact_path, artifact)
    intent = {
        "schema": BATCH.INTENT_SCHEMA, "action_id": action_id, "kind": "train",
        "ordinal": action_ordinal, "protocol_sha256_lf": source_protocol["_verified_sha256_lf"],
        "resource_sha256": resource_ref["sha256"], "cost_before_usd": 2.0,
        "started_at_utc": "2027-01-01T01:00:00+00:00", "reserved_max_wall_s": 1800,
        "reserved_max_compute_usd": 1.0,
        "cost_scope": source_protocol["resources"]["cost_scope"],
    }
    intent_path = folder / "intent.json"
    _write(intent_path, intent)
    raw_path = folder / "raw.json"
    _write(raw_path, {"kind": "train", "action_id": action_id, "artifact": artifact, "job_id": job_id})
    receipt_path = folder / "receipt.json"
    _write(receipt_path, {
        "schema": BATCH.RECEIPT_SCHEMA, "action_id": action_id,
        "intent_sha256": _sha(intent_path), "raw_sha256": _sha(raw_path),
        "cost_after_usd": 2.0 + 2.0 / 60.0, "finished_at_utc": "2027-01-01T01:01:00+00:00",
    })
    freshness_path = folder / "freshness.json"
    _write(freshness_path, {
        "schema": "preregistered_seed_freshness_v1", "status": "fresh",
        "seeds": [4101, 4102], "checked_at_utc": "2027-01-01T00:59:00+00:00",
        "checked_roots": ["bench"], "checked_file_count": 1, "evidence": [],
        "scope": "fixture_registered_source_roots",
    })
    runtime_path = folder / "runtime_manifest.json"
    _write(runtime_path, {
        "schema": BATCH.MANIFEST_SCHEMA, "protocol_id": source_protocol["protocol_id"],
        "protocol_path": source_protocol_path.relative_to(repo).as_posix(),
        "protocol_sha256_lf": source_protocol["_verified_sha256_lf"], "source_anchor": anchor,
        "resource_path_sha256": resource_ref["sha256"],
        "training_plan_sha256": BATCH.digest(source_protocol["training_plan"]),
        "source_action_order_sha256": BATCH.digest(source_protocol["source_action_order"]),
        "freshness_sha256": _sha(freshness_path),
    })
    return artifact_path, {
        "intent": _ref(repo, intent_path), "raw": _ref(repo, raw_path),
        "receipt": _ref(repo, receipt_path), "runtime_manifest": _ref(repo, runtime_path),
        "freshness": _ref(repo, freshness_path),
    }


def _readiness(
    repo: Path,
    folder: Path,
    *,
    source_protocol: dict,
    source_protocol_path: Path,
    anchor: str,
    resource: dict,
    resource_refs: dict[str, dict[str, str]],
    checkpoint: Path,
    arm: str,
    seed: int,
    records: list[dict],
    video: Path,
    ordinal: int,
) -> tuple[Path, dict[str, Path], dict]:
    source_job_id = f"s{seed}-{arm.lower()}"
    record_job_id = f"record-{seed}-{arm.lower()}"
    action = next(row for row in source_protocol["source_action_order"]
                  if row["kind"] == "train" and row["job_id"] == source_job_id)
    action_ordinal = source_protocol["source_action_order"].index(action)
    artifact_path, training_refs = _training(
        repo, folder / "training", checkpoint=checkpoint, arm=arm, seed=seed,
        job_id=source_job_id, action_id=action["action_id"], action_ordinal=action_ordinal,
        source_protocol=source_protocol, source_protocol_path=source_protocol_path,
        anchor=anchor, resource_ref=resource_refs["resource_parent"],
    )
    logical_id = f"policy-{seed}-{arm.lower()}"
    record_refs, record_artifacts, _ = _capture(
        repo, folder / "record", mode=RECORD.RECORD_MODE, job_id=record_job_id,
        checkpoint=checkpoint, checkpoint_logical_id=logical_id, anchor=anchor,
        source_protocol=source_protocol, source_protocol_path=source_protocol_path,
        resource=resource, resource_refs=resource_refs, video_source=video,
        records=records, ordinal=ordinal * 2 - 1,
    )
    smoke_refs, _, _ = _capture(
        repo, folder / "smoke", mode=RECORD.RENDER_SYNC_SMOKE_MODE,
        job_id=f"smoke-{seed}-{arm.lower()}", checkpoint=checkpoint,
        checkpoint_logical_id=logical_id, anchor=anchor,
        source_protocol=source_protocol, source_protocol_path=source_protocol_path,
        resource=resource, resource_refs=resource_refs, video_source=video,
        records=records, ordinal=ordinal * 2,
    )
    manifest = {
        "schema": SOURCES.READINESS_SCHEMA, "source_training_artifact": _ref(repo, artifact_path),
        "training_action": training_refs, **resource_refs,
        "record": record_refs, "smoke": smoke_refs,
        "receipt": {},
    }
    manifest["receipt"] = {
        "schema": "vision_selection_readiness_receipt_v1",
        "scope": "registered_renderer_combination_only", "model_use_ready": True,
        "record_result_sha256": record_refs["result"]["sha256"],
        "smoke_result_sha256": smoke_refs["result"]["sha256"],
        "smoke_report_sha256": smoke_refs["render_sync_smoke"]["sha256"],
        "record_context_sha256": record_refs["context"]["sha256"],
        "smoke_context_sha256": smoke_refs["context"]["sha256"],
        "record_attestation_sha256": record_refs["attestation"]["sha256"],
        "smoke_attestation_sha256": smoke_refs["attestation"]["sha256"],
        "record_job_sha256": record_refs["job"]["sha256"],
        "smoke_job_sha256": smoke_refs["job"]["sha256"],
        "resource_proof_sha256": resource_refs["resource_parent"]["sha256"],
        "live_lease_proof_sha256": resource_refs["live_lease_proof"]["sha256"],
        "training_action_receipt_sha256": training_refs["receipt"]["sha256"],
        "source_protocol_sha256_lf": source_protocol["_verified_sha256_lf"],
        "universal_zero_lag_claim": False, "smoke_raw_state_recomputed": False,
    }
    path = folder / "model_readiness.json"
    _write(path, manifest)
    return path, record_artifacts, record_refs


@pytest.fixture(scope="module")
def full_case(tmp_path_factory: pytest.TempPathFactory) -> dict:
    repo = tmp_path_factory.mktemp("vision-sources") / "repo"
    repo.mkdir()
    copied = {
        "bench/vision_render_context_v1.py",
        "bench/vision_render_attestation_v1.py",
        "bench/record_vision_evidence_v1.py",
        "bench/vision_play_worker_v1.py",
        "bench/run_vision_brev_batch_v1.py",
        *CALL._METHODS,
    }
    for relative in copied:
        target = repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())
    for name in ("runtime.py", "package.py"):
        (repo / name).write_text(f"# {name}\n", encoding="utf-8")
    bands = repo / "bench/bands.json"
    _write(bands, {"bands": {"fixture": True}})
    source_map = {name: _sha(repo / name, lf=True) for name in copied}
    recording_map = {name: source_map[name] for name in (
        "bench/vision_render_context_v1.py", "bench/record_vision_evidence_v1.py",
        "bench/vision_play_worker_v1.py", "bench/vision_render_attestation_v1.py",
    )}
    packages = {
        "isaacsim": "4.5.0.0", "isaaclab": "0.41.3", "isaaclab_rl": "0.1.8",
        "torch": "2.7.0+cu128", "rsl-rl-lib": "2.3.3", "gymnasium": "1.2.0",
    }
    source_protocol = {
        "schema": BATCH.SCHEMA, "protocol_id": "vision-source-helper-fixture",
        "registered_before_measurement": True,
        "record_profile_sha256": RECORD.digest(RECORD.fixed_profile()),
        "training_plan": BATCH_FIXTURE._plan(), "source_action_order": BATCH_FIXTURE._actions(),
        "evaluation": {"bands_path": "bench/bands.json", "bands_sha256": _sha(bands)},
        "recording": {"source_files_lf": recording_map},
        "resources": {
            "deadline_utc": "2030-01-01T00:00:00+00:00", "whole_batch_budget_usd": 50.0,
            "compute_quote_usd_per_h": 2.0, "owned_instance_id": "brev-instance-1",
            "max_owned_instances": 1,
            "action_limits": {
                "train": {"max_wall_s": 1800, "max_compute_usd": 1.0},
                "evaluate": {"max_wall_s": 900, "max_compute_usd": 0.5},
                "record": {"max_wall_s": 900, "max_compute_usd": 0.5},
            },
            "reservation_method": "owned_instance_elapsed_plus_next_max_v1",
            "cost_scope": "owned_instance_elapsed_wall_includes_idle_setup_transfer_excludes_storage_v1",
        },
        "source_sha256_lf": source_map,
        "runtime_source_sha256_lf": {"runtime.py": _sha(repo / "runtime.py", lf=True)},
        "python_package_source_sha256_lf": {"package.py": _sha(repo / "package.py", lf=True)},
        "environment": {"python": "3.10.12", "isaaclab_sha": "a" * 40, **packages},
        "fixed_eval_protocol": "bench/fixed_eval.json",
    }
    source_protocol_path = repo / "bench/protocols/vision_brev_source_v1.json"
    _write(source_protocol_path, source_protocol)
    method_protocol_path = repo / "bench/protocols/vision_selection_comparison_v1.json"
    _write(method_protocol_path, {"schema": "vision_selection_comparison_method_v1"})
    _git(repo, "init")
    _git(repo, "config", "user.email", "fixture@example.invalid")
    _git(repo, "config", "user.name", "fixture")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "fixture source")
    anchor = _git(repo, "rev-parse", "HEAD")
    source_sha = _sha(source_protocol_path, lf=True)
    source_protocol = BATCH.validate_protocol(source_protocol)
    source_protocol["_verified_sha256_lf"] = source_sha
    resource, resource_refs = _resource(repo, source_sha)

    shared_video = repo / "artifacts/shared/same_play.mp4"
    shared_video.parent.mkdir(parents=True)
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
        "testsrc=size=64x64:rate=50", "-frames:v", str(RECORD.FRAME_COUNT),
        "-c:v", "mpeg4", "-y", str(shared_video),
    ], check=True, capture_output=True)
    timing = RECORD.probe_video_timing(shared_video)
    assert timing["frame_count"] == RECORD.FRAME_COUNT
    records = []
    for index in range(RECORD.FRAME_COUNT):
        _, row = RECORD.validate_atomic_sample(RECORD_FIXTURE._sample(index), index)
        records.append({**row, "actual_timestamp_s": index / RECORD.FPS})

    role_data = {}
    for ordinal, (group, seed, arm, role) in enumerate((
        ("source-01", 4101, "NONE", "normal"),
        ("source-01", 4101, "F1", "registered-f1"),
        ("source-02", 4102, "NONE", "normal"),
        ("source-02", 4102, "F1", "registered-f1"),
    ), 1):
        folder = repo / f"artifacts/{group}-{role}"
        checkpoint = folder / "checkpoint.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(f"checkpoint-{group}-{role}".encode())
        readiness, record_artifacts, record_refs = _readiness(
            repo, folder, source_protocol=source_protocol,
            source_protocol_path=source_protocol_path, anchor=anchor,
            resource=resource, resource_refs=resource_refs, checkpoint=checkpoint,
            arm=arm, seed=seed, records=records, video=shared_video, ordinal=ordinal,
        )
        trace = record_artifacts["trace"]
        audit = folder / "camera_pose_audit.json"
        _write(audit, SOURCES.camera_pose_audit(trace))
        role_data[(group, role)] = {
            "seed": seed, "checkpoint": checkpoint, "readiness": readiness,
            "trace": trace, "audit": audit, "record_refs": record_refs,
            "measurements": {window: record_artifacts[f"measurement_{index}"]
                             for index, window in enumerate(COMPARISON.WINDOWS, 1)},
            "specs": {window: record_artifacts[f"spec_{index}"]
                      for index, window in enumerate(COMPARISON.WINDOWS, 1)},
        }

    method_sha = _sha(method_protocol_path, lf=True)
    method_hashes = {name: _sha(repo / name) for name in CALL._METHODS}
    rows, locations = [], []
    for slot in COMPARISON.design():
        source = role_data[(slot["seed_group"], slot["source_role"])]
        slot_folder = repo / f"inputs/{slot['slot_id']}"
        original_spec = json.loads(source["specs"][slot["window"]].read_text(encoding="utf-8"))
        local_spec = deepcopy(original_spec)
        local_spec["video"]["path"] = str((repo / source["record_refs"]["video"]["path"]).resolve())
        local_spec["checkpoint"]["path"] = str(source["checkpoint"].resolve())
        local_spec["measurement"]["path"] = str(source["measurements"][slot["window"]].resolve())
        local_spec_path = slot_folder / "v01_local_spec.json"
        _write(local_spec_path, local_spec)
        contract_path = CALL.CONTRACT.VISION.prepare(local_spec_path, slot_folder / "v01")
        contract = CALL.CONTRACT.VISION.verify(contract_path)
        binding = {"contract_sha256": _sha(contract_path),
                   "source_lineage_id": contract["provenance"]["source_lineage_id"]}
        snapshot = SELECTION_FIXTURE.selection_snapshot(slot_folder, _sha(source["checkpoint"]))
        snapshot_path = slot_folder / "snapshot.json"
        _write(snapshot_path, snapshot)
        manifest_path = slot_folder / "input_manifest.json"
        _write(manifest_path, {
            "schema": SOURCES.INPUT_SCHEMA, "snapshot": _ref(repo, snapshot_path),
            "contract": _ref(repo, contract_path), "trace": _ref(repo, source["trace"]),
            "camera_pose_audit": _ref(repo, source["audit"]),
            "model_readiness": _ref(repo, source["readiness"]),
        })
        for view in slot["view_order"]:
            call_id = f"choice-{len(rows) + 1:02d}"
            capsule_ref = CALL.freeze(
                snapshot, contract_path, binding, view, slot_folder / f"capsule-{view}",
            )
            capsule_path = Path(capsule_ref["path"])
            capsule = json.loads(capsule_path.read_text(encoding="utf-8"))
            rows.append({
                "slot_id": slot["slot_id"], "seed_group": slot["seed_group"],
                "source_role": slot["source_role"], "window": slot["window"],
                "view_kind": view, "call_id": call_id, "source_seed": source["seed"],
                "source_lineage_id": contract["provenance"]["source_lineage_id"],
                "parent_lineage_id": contract["provenance"]["parent_source_lineage_id"],
                "split": "comparison", "method_protocol_sha256": method_sha,
                "source_protocol_sha256": source_sha,
                "capsule_method_raw_sha256": method_hashes,
                "checkpoint_sha256": _sha(source["checkpoint"]),
                "video_sha256": contract["provenance"]["video"]["sha256"],
                "trace_sha256": _sha(source["trace"]), "contract_sha256": _sha(contract_path),
                "snapshot_sha256": _sha(snapshot_path), "capsule_sha256": _sha(capsule_path),
                "messages_sha256": capsule["messages_sha256"],
                "camera_pose_audit_sha256": _sha(source["audit"]),
            })
            locations.append({"call_id": call_id, "input_manifest": _ref(repo, manifest_path, frozen=True),
                              "capsule": _ref(repo, capsule_path, frozen=True)})

    locations_path = repo / "private/input_locations.json"
    _write(locations_path, {"schema": SOURCES.LOCATIONS_SCHEMA,
                            "execution_id": "vision-selection-source-fixture", "rows": locations})
    execution_protocol = {
        "execution_id": "vision-selection-source-fixture", "rows": rows,
        "method_protocol": {"path": method_protocol_path.relative_to(repo).as_posix(),
                            "sha256_lf": method_sha},
        "source_protocol": {"path": source_protocol_path.relative_to(repo).as_posix(),
                            "sha256_lf": source_sha},
        "source_sha256_lf": source_map,
        "evidence_sha256_lf": {"bench/bands.json": _sha(bands, lf=True)},
        "input_locations_sha256": _sha(locations_path),
    }
    return {"repo": repo, "protocol": execution_protocol, "locations": locations_path,
            "role_data": role_data}


def test_full_24_inputs_bind_v04_readiness_v01_and_capsules(full_case: dict) -> None:
    result = SOURCES.verify_inputs(
        full_case["repo"], full_case["protocol"], full_case["locations"],
    )
    assert result["schema"] == SOURCES.VERIFIED_SCHEMA
    assert len(result["rows"]) == 24
    assert [row["call_id"] for row in result["rows"]] == [f"choice-{index:02d}" for index in range(1, 25)]
    assert result["input_locations_sha256"] == full_case["protocol"]["input_locations_sha256"]
    assert all(set(row) == {"call_id", "run_id", "capsule"} for row in result["rows"])


def test_rejects_public_locations_sha_and_strict_json_tampering(full_case: dict, tmp_path: Path) -> None:
    protocol = deepcopy(full_case["protocol"])
    protocol["input_locations_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="입력 위치 SHA"):
        SOURCES.verify_inputs(full_case["repo"], protocol, full_case["locations"])

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema":"a","schema":"b"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="중복 key"):
        SOURCES._read(duplicate.resolve())


def test_camera_audit_and_artifact_alias_tampering_are_rejected(full_case: dict, tmp_path: Path) -> None:
    source = full_case["role_data"][("source-01", "normal")]
    trace = json.loads(source["trace"].read_text(encoding="utf-8"))
    trace["records"][50]["camera"]["eye_world_m"][0] += 0.1
    changed = tmp_path / "changed_trace.json"
    _write(changed, trace)
    with pytest.raises(ValueError, match="camera"):
        SOURCES.camera_pose_audit(changed.resolve())

    first = tmp_path / "one.json"
    first.write_text("{}\n", encoding="utf-8")
    result = {"artifacts": {
        "a": {"file": first.name, "sha256": _sha(first), "bytes": first.stat().st_size},
        "b": {"file": first.name, "sha256": _sha(first), "bytes": first.stat().st_size},
    }}
    refs = {"a": {"path": first.name, "sha256": _sha(first)},
            "b": {"path": first.name, "sha256": _sha(first)}}
    with pytest.raises(ValueError, match="이름|바이트|크기"):
        SOURCES._result_files(tmp_path.resolve(), refs, result, {"a", "b"})


def _validated_source_protocol(full_case: dict) -> dict:
    repo = full_case["repo"]
    reference = full_case["protocol"]["source_protocol"]
    protocol = BATCH.validate_protocol(json.loads((repo / reference["path"]).read_text(encoding="utf-8")))
    protocol["_verified_sha256_lf"] = reference["sha256_lf"]
    return protocol


@pytest.mark.parametrize("target", ["live_lease_proof", "source_training_artifact", "training_raw"])
def test_readiness_rejects_resource_and_training_evidence_tampering(
    full_case: dict, target: str,
) -> None:
    repo = full_case["repo"]
    original_path = full_case["role_data"][("source-01", "normal")]["readiness"]
    manifest = json.loads(original_path.read_text(encoding="utf-8"))
    folder = repo / f"artifacts/tamper-{target}"
    folder.mkdir(parents=True, exist_ok=True)
    if target == "live_lease_proof":
        changed = folder / "live.json"
        _write(changed, {"arbitrary": True})
        manifest[target] = _ref(repo, changed)
    elif target == "source_training_artifact":
        source = repo / manifest[target]["path"]
        body = json.loads(source.read_text(encoding="utf-8"))
        body["arm"] = "F1"
        changed = folder / "artifact.json"
        _write(changed, body)
        manifest[target] = _ref(repo, changed)
    else:
        source = repo / manifest["training_action"]["raw"]["path"]
        body = json.loads(source.read_text(encoding="utf-8"))
        body["job_id"] = "other-job"
        changed = folder / "raw.json"
        _write(changed, body)
        manifest["training_action"]["raw"] = _ref(repo, changed)
    changed_manifest = folder / "readiness.json"
    _write(changed_manifest, manifest)
    with pytest.raises(ValueError):
        SOURCES.verify_model_readiness(repo, changed_manifest.resolve(), _validated_source_protocol(full_case))


def test_smoke_camera_offset_and_strict_nonfinite_or_hardlink_are_rejected(
    full_case: dict, tmp_path: Path,
) -> None:
    repo = full_case["repo"]
    readiness = json.loads(
        full_case["role_data"][("source-01", "normal")]["readiness"].read_text(encoding="utf-8"),
    )
    smoke = readiness["smoke"]
    report_path = repo / smoke["render_sync_smoke"]["path"]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["captures"][2]["camera"] = report["captures"][1]["camera"]
    changed_report = tmp_path / "smoke.json"
    _write(changed_report, report)
    artifact_paths = {
        "render_sync_smoke": changed_report.resolve(),
        **{f"audit_rgb_{index}": (repo / smoke[f"audit_rgb_{index}"]["path"]).resolve()
           for index in range(1, 4)},
    }
    result = json.loads((repo / smoke["result"]["path"]).read_text(encoding="utf-8"))
    result["primary_artifact_sha256"] = _sha(changed_report)
    with pytest.raises(ValueError, match="offset|A-B-A"):
        SOURCES._verify_smoke({"artifact_paths": artifact_paths, "result": result})

    nonfinite = tmp_path / "nan.json"
    nonfinite.write_text('{"value":NaN}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="비유한"):
        SOURCES._read(nonfinite.resolve())

    hardlink = tmp_path / "hardlink.json"
    os.link(nonfinite, hardlink)
    with pytest.raises(ValueError, match="hardlink"):
        SOURCES._read(hardlink.resolve())
