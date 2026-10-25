from __future__ import annotations

import copy
import importlib
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import test_vision_selection_sources_v1 as FIXTURE
from test_vision_selection_sources_v1 import full_case as full_case  # noqa: PLC0414

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / "bench", ROOT / "evals", ROOT / "src"):
    sys.path.insert(0, str(folder))
F = importlib.import_module("vision_render_readiness_v2")


def _input(case: dict) -> tuple[Path, Path, str, dict]:
    repo = case["repo"]
    descriptor = case["protocol"]["source_protocol"]
    readiness = next(iter(case["role_data"].values()))["readiness"]
    material = copy.deepcopy(F.INPUTS._read(readiness))
    material.pop("receipt")
    return repo, repo / descriptor["path"], descriptor["sha256_lf"], material


def test_finalizer_checks_original_pair_and_replays_without_gpu_or_model(
    full_case: dict, monkeypatch,
) -> None:
    repo, protocol, protocol_sha, material = _input(full_case)
    output = repo / "finalized/verified.json"
    calls: list[str] = []
    real_run = subprocess.run

    def cpu_only(argv, *args, **kwargs):
        executable = Path(argv[0]).stem
        assert executable in {"git", "ffprobe"}
        calls.append(executable)
        return real_run(argv, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", cpu_only)
    reference = F.finalize_pair(repo, protocol, protocol_sha, material, output)
    original = output.read_bytes()
    assert reference == {"path": "finalized/verified.json", "sha256": F.SOURCE.sha256(output)}
    body = F.INPUTS._read(output)
    assert body["receipt"]["model_use_ready"] is True
    assert body["receipt"]["scope"] == "registered_renderer_combination_only"
    assert body["receipt"]["universal_zero_lag_claim"] is False
    assert F.finalize_pair(repo, protocol, protocol_sha, material, output) == reference
    assert output.read_bytes() == original
    assert calls and set(calls) <= {"git", "ffprobe"}
    assert list(output.parent.glob(".readiness-check-*")) == []


def test_invalid_smoke_does_not_create_readiness(full_case: dict) -> None:
    repo, protocol, protocol_sha, material = _input(full_case)
    material["smoke"]["audit_rgb_1"]["sha256"] = "0" * 64
    output = repo / "finalized/invalid-smoke.json"
    with pytest.raises(ValueError, match="SHA|바이트|원문"):
        F.finalize_pair(repo, protocol, protocol_sha, material, output)
    assert not output.exists()
    assert list(output.parent.glob(".readiness-check-*")) == []


def test_resource_start_bounds_freshness_before_any_ready_output(full_case: dict) -> None:
    repo, protocol, protocol_sha, material = _input(full_case)
    fresh = F.INPUTS._read(F.INPUTS._ref(repo, material["training_action"]["freshness"]))
    fresh["checked_at_utc"] = "2026-01-01T00:00:00+00:00"
    path = repo / "finalized/too-early-freshness.json"
    F.SOURCE.write_new(path, fresh)
    material["training_action"]["freshness"] = F._reference(repo, path)
    output = repo / "finalized/invalid-freshness.json"
    with pytest.raises(ValueError, match="자원 시작"):
        F.finalize_pair(repo, protocol, protocol_sha, material, output)
    assert not output.exists()


def test_source_sha_and_existing_target_are_not_replaced(full_case: dict) -> None:
    repo, protocol, protocol_sha, material = _input(full_case)
    output = repo / "finalized/preserved.json"
    with pytest.raises(ValueError, match="protocol LF SHA"):
        F.finalize_pair(repo, protocol, "0" * 64, material, output)
    assert not output.exists()
    F.SOURCE.write_new(output, {"preserved": True})
    before = output.read_bytes()
    with pytest.raises(ValueError, match="기존 readiness"):
        F.finalize_pair(repo, protocol, protocol_sha, material, output)
    assert output.read_bytes() == before


def test_readiness_target_cannot_escape_repo(full_case: dict) -> None:
    repo, protocol, protocol_sha, material = _input(full_case)
    output = repo.parent / "escaped-readiness.json"
    with pytest.raises(ValueError):
        F.finalize_pair(repo, protocol, protocol_sha, material, output)
    assert not output.exists()


def test_builder_derives_portable_material_from_completed_originals(full_case: dict) -> None:
    repo, protocol_path, protocol_sha, original = _input(full_case)
    protocol = F.load_source_protocol(repo, protocol_path, protocol_sha)
    record_job = F.INPUTS._read(F.INPUTS._ref(repo, original["record"]["job"]))
    record_id = record_job["job_id"]
    record_action = next(row for row in protocol["source_action_order"]
                         if row["kind"] == "record" and row["job_id"] == record_id)
    ordinal, train = next((i, row) for i, row in enumerate(protocol["source_action_order"])
                          if row["kind"] == "train" and row["job_id"] == record_action["source_job_id"])
    runtime = repo / ".triage-runtime/vision-brev" / protocol["protocol_id"]
    action_folder = runtime / "actions" / f"{ordinal:02d}_{train['action_id']}"
    action_folder.mkdir(parents=True)
    for key in ("intent", "raw", "receipt"):
        shutil.copyfile(F.INPUTS._ref(repo, original["training_action"][key]), action_folder / f"{key}.json")
    for key, filename in (("runtime_manifest", "manifest.json"), ("freshness", "freshness.json")):
        shutil.copyfile(F.INPUTS._ref(repo, original["training_action"][key]), runtime / filename)
    resource_path = F.INPUTS._ref(repo, original["resource_parent"])
    resource = F.INPUTS._read(resource_path)
    role = next(iter(full_case["role_data"].values()))
    smoke_refs, _, _ = FIXTURE._capture(
        repo, repo / "builder-smoke", mode=F.INPUTS.RECORD.RENDER_SYNC_SMOKE_MODE,
        job_id=f"smoke-{record_id}", checkpoint=role["checkpoint"],
        checkpoint_logical_id=record_action["checkpoint_logical_id"],
        anchor=record_job["source"]["commit"], source_protocol=protocol,
        source_protocol_path=protocol_path, resource=resource,
        resource_refs={key: original[key] for key in (
            "resource_parent", "live_lease_proof", "budget_admission", "provider_inventory", "gpu_csv", "process_csv",
        )}, video_source=F.INPUTS._ref(repo, original["record"]["video"]), records=[], ordinal=91,
    )
    locations = {key: original[key] for key in (
        "resource_parent", "live_lease_proof", "budget_admission", "provider_inventory", "gpu_csv", "process_csv",
    )}
    material = F.build_material(repo, protocol_path, protocol_sha, resource_path, runtime,
                                record_id, original["record"], smoke_refs, resource_locations=locations)
    assert set(material) == F.MATERIAL_KEYS
    assert "receipt" not in material
    for key in ("resource_parent", "live_lease_proof", "budget_admission", "provider_inventory", "gpu_csv", "process_csv"):
        assert F.INPUTS._ref(repo, material[key]).read_bytes() == F.INPUTS._ref(repo, original[key]).read_bytes()
    reference = F.finalize_pair(repo, protocol_path, protocol_sha, material, repo / "finalized/built-pair.json")
    assert reference["path"] == "finalized/built-pair.json"
    assert F.build_material(repo, protocol_path, protocol_sha, resource_path, runtime,
                            record_id, original["record"], smoke_refs, resource_locations=locations) == material


def test_archive_refs_map_posix_provenance_without_rewriting_originals(tmp_path: Path) -> None:
    repo = (tmp_path / "repo").resolve()
    runtime = repo / ".triage-runtime/vision-brev/fixture"
    stage = runtime / "record/smoke-record-one"
    output = stage / "output"
    output.mkdir(parents=True)
    original_root = "/origin/repo"

    def original(path: Path) -> str:
        return original_root + "/" + path.relative_to(repo).as_posix()

    protocol, admission = runtime / "inputs/protocol.json", runtime / "inputs/admission.json"
    F.SOURCE.write_new(protocol, {"fixture": "protocol"})
    F.SOURCE.write_new(admission, {"fixture": "admission"})
    job = {"job_id": "smoke-record-one", "mode": F.INPUTS.RECORD.RENDER_SYNC_SMOKE_MODE,
           "source": {"repo_root": original_root}, "protocol": {"path": original(protocol)},
           "remote_gpu_admission": {"path": original(admission)}}
    job_path = stage / "job.json"
    F.SOURCE.write_new(job_path, job)
    job_bytes = job_path.read_bytes()
    artifacts = {}
    for name in ("render_sync_smoke", "audit_rgb_1", "audit_rgb_2", "audit_rgb_3"):
        path = output / (name + ".bin")
        path.write_bytes(name.encode())
        artifacts[name] = {"file": path.name, "sha256": F.SOURCE.sha256(path), "bytes": path.stat().st_size}
    F.SOURCE.write_new(output / "result.json", {"completed": True, "job_sha256": F.SOURCE.sha256(job_path),
                                                "artifacts": artifacts})
    observations = {}
    for side in ("before", "after"):
        path = runtime / f"observations/{side}.json"
        F.SOURCE.write_new(path, {"fixture": side})
        observations[f"gpu_{side}_observation"] = {"path": original(path), "sha256": F.SOURCE.sha256(path)}
    F.SOURCE.write_new(stage / "execution.json", {
        "completed": True, "job_sha256": F.SOURCE.sha256(job_path),
        "result_sha256": F.SOURCE.sha256(output / "result.json"), **observations,
    })
    for path in (stage / "attestation/context/context.receipt.json", stage / "attestation/attestation.json"):
        F.SOURCE.write_new(path, {"fixture": True})
    references = F.archive_capture_references(repo, runtime, "smoke-record-one")
    assert references["protocol"] == F._reference(repo, protocol)
    assert references["admission"] == F._reference(repo, admission)
    assert references["gpu_observation_before"]["path"].endswith("observations/before.json")
    assert all(not Path(ref["path"]).is_absolute() for ref in references.values())
    assert job_path.read_bytes() == job_bytes
    assert not (runtime / "readiness").exists()
    (output / "audit_rgb_1.bin").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="artifact 바이트"):
        F.archive_capture_references(repo, runtime, "smoke-record-one")
    with pytest.raises(ValueError):
        F._archived_path(repo, original_root, "/origin/repo-other/protocol.json")


def test_archive_resource_locations_bind_bytes_and_reject_unknown_roots(tmp_path: Path) -> None:
    repo = (tmp_path / "repo").resolve()
    repo.mkdir()
    proof = {}
    fields = ("raw_gpu_receipt", "budget_admission", "provider_inventory", "nvidia_gpu_csv", "nvidia_process_csv")
    for field in fields:
        path = repo / f"resource/{field}.data"
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(field.encode())
        proof[field + "_path"] = "/origin/repo/" + path.relative_to(repo).as_posix()
        proof[field + "_sha256"] = F.SOURCE.sha256(path)
    resource = repo / "resource/proof.json"
    F.SOURCE.write_new(resource, proof)
    before = resource.read_bytes()
    locations = F.archive_resource_locations(repo, resource, "/origin/repo")
    assert len(locations) == 6
    assert locations["resource_parent"]["sha256"] == F.SOURCE.sha256(resource)
    assert resource.read_bytes() == before
    with pytest.raises(ValueError):
        F.archive_resource_locations(repo, resource, "/other/repo")
