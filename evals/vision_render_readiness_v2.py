"""회수한 record/smoke 원문을 CPU에서 검증하고 readiness를 새 파일로 고정한다."""
from __future__ import annotations

import hashlib
import os
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _folder in (ROOT / "bench", ROOT / "evals", ROOT / "src"):
    if str(_folder) not in sys.path:
        sys.path.insert(0, str(_folder))

import run_vision_brev_batch_v1 as SOURCE  # pyright: ignore[reportMissingImports]
import vision_selection_sources_v1 as INPUTS

MATERIAL_KEYS = {
    "schema", "source_training_artifact", "training_action", "resource_parent",
    "live_lease_proof", "budget_admission", "provider_inventory", "gpu_csv", "process_csv",
    "record", "smoke",
}


def load_source_protocol(repo: Path, path: Path, expected_sha256_lf: str) -> dict[str, Any]:
    """보관 protocol을 현재 LF 바이트와 대조한다."""
    repo = Path(repo).resolve(strict=True)
    path = INPUTS._regular(Path(path))
    path.relative_to(repo)
    if (not isinstance(expected_sha256_lf, str)
            or SOURCE.HEX64.fullmatch(expected_sha256_lf) is None
            or SOURCE.sha256(path, lf=True) != expected_sha256_lf):
        raise ValueError("source protocol LF SHA가 다르다")
    protocol = SOURCE.validate_protocol(INPUTS._read(path))
    protocol["_verified_sha256_lf"] = expected_sha256_lf
    return protocol


def _target(repo: Path, output: Path) -> Path:
    output = Path(output)
    if not output.is_absolute() or output.resolve() != output or output.suffix != ".json":
        raise ValueError("readiness output은 정규화된 절대 JSON 경로여야 한다")
    output.relative_to(repo)
    current = output
    while current != repo:
        if current.is_symlink():
            raise ValueError("readiness output 경로에 symlink가 있다")
        current = current.parent
    return output


def _reference(repo: Path, path: Path) -> dict[str, str]:
    path = INPUTS._regular(path)
    return {"path": path.relative_to(repo).as_posix(), "sha256": SOURCE.sha256(path)}


def _archived_path(repo: Path, original_root: str, original_path: str) -> Path:
    root, path = PurePosixPath(original_root), PurePosixPath(original_path)
    if (not root.is_absolute() or not path.is_absolute() or ".." in root.parts or ".." in path.parts
            or root.as_posix() != original_root or path.as_posix() != original_path):
        raise ValueError("보관 원문의 POSIX 경로가 정규형이 아니다")
    relative = path.relative_to(root)
    return INPUTS._regular(repo.joinpath(*relative.parts))


def archive_resource_locations(
    repo: Path, resource_path: Path, original_repo_root: str,
) -> dict[str, dict[str, str]]:
    """원래 저장소 안에 있던 자원 원문만 보관 위치로 대응시킨다."""
    repo = Path(repo).resolve(strict=True)
    resource_path = INPUTS._regular(Path(resource_path))
    proof = INPUTS._read(resource_path)
    mappings = {
        "live_lease_proof": "raw_gpu_receipt", "budget_admission": "budget_admission",
        "provider_inventory": "provider_inventory", "gpu_csv": "nvidia_gpu_csv",
        "process_csv": "nvidia_process_csv",
    }
    result = {"resource_parent": _reference(repo, resource_path)}
    for name, field in mappings.items():
        path = _archived_path(repo, original_repo_root, proof[f"{field}_path"])
        reference = _reference(repo, path)
        if reference["sha256"] != proof[f"{field}_sha256"]:
            raise ValueError("보관 자원 원문 SHA가 다르다")
        result[name] = reference
    return result


def archive_capture_references(
    repo: Path, runtime_root: Path, job_id: str,
) -> dict[str, dict[str, str]]:
    """완료 capture의 보관 파일 위치를 읽는다. 새 worker를 시작하지 않는다."""
    repo = Path(repo).resolve(strict=True)
    runtime = Path(runtime_root)
    if (SOURCE.SLUG.fullmatch(job_id) is None
            or runtime.parent != repo / ".triage-runtime/vision-brev"
            or runtime.resolve() != runtime):
        raise ValueError("보관 capture runtime/job이 정규형이 아니다")
    stage = runtime / "record" / job_id
    job_path, result_path, execution_path = stage / "job.json", stage / "output/result.json", stage / "execution.json"
    job, result, execution = (INPUTS._read(path) for path in (job_path, result_path, execution_path))
    if (job.get("job_id") != job_id or execution.get("completed") is not True
            or execution.get("job_sha256") != SOURCE.sha256(job_path)
            or result.get("completed") is not True or result.get("job_sha256") != SOURCE.sha256(job_path)
            or execution.get("result_sha256") != SOURCE.sha256(result_path)):
        raise ValueError("보관 capture 완료·job·result 결합이 다르다")
    original_root = job["source"]["repo_root"]
    files = {
        "job": job_path, "result": result_path, "execution": execution_path,
        "protocol": _archived_path(repo, original_root, job["protocol"]["path"]),
        "admission": _archived_path(repo, original_root, job["remote_gpu_admission"]["path"]),
        "context": stage / "attestation/context/context.receipt.json",
        "attestation": stage / "attestation/attestation.json",
    }
    import vision_render_attestation_v1 as ATTEST  # pyright: ignore[reportMissingImports]

    auxiliary = ATTEST.execution_evidence_path(execution_path, execution, job, result_path)
    if auxiliary is not None:
        files["execution_evidence"] = auxiliary
    for side in ("before", "after"):
        descriptor = execution[f"gpu_{side}_observation"]
        actual = _archived_path(repo, original_root, descriptor["path"])
        if SOURCE.sha256(actual) != descriptor["sha256"]:
            raise ValueError("보관 capture GPU 관측 원문 SHA가 다르다")
        files[f"gpu_observation_{side}"] = actual
    if job.get("mode") == INPUTS.RECORD.RECORD_MODE:
        files["video"] = stage / "output/same_play.mp4"
        artifact_keys = INPUTS._RECORD_ARTIFACTS
    elif job.get("mode") == INPUTS.RECORD.RENDER_SYNC_SMOKE_MODE:
        artifact_keys = INPUTS._SMOKE_ARTIFACTS
    else:
        raise ValueError("보관 capture mode가 다르다")
    artifacts = INPUTS._exact(result.get("artifacts"), artifact_keys, "보관 capture artifact")
    for name, descriptor in artifacts.items():
        INPUTS._exact(descriptor, {"file", "sha256", "bytes"}, "보관 artifact descriptor")
        filename = descriptor["file"]
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ValueError("보관 capture artifact 이름이 정규형이 아니다")
        matches = list((stage / "output").rglob(filename))
        if len(matches) != 1:
            raise ValueError("보관 capture artifact가 없거나 중복됐다")
        actual = INPUTS._regular(matches[0])
        if (type(descriptor["bytes"]) is not int or descriptor["bytes"] <= 0
                or actual.stat().st_size != descriptor["bytes"]
                or SOURCE.sha256(actual) != descriptor["sha256"]):
            raise ValueError("보관 capture artifact 바이트가 다르다")
        files[name] = actual
    return {name: _reference(repo, path) for name, path in files.items()}


def _receipt(repo: Path, material: dict[str, Any], source_sha: str) -> dict[str, Any]:
    record, smoke = material["record"], material["smoke"]

    def sha(reference: Any) -> str:
        return SOURCE.sha256(INPUTS._ref(repo, reference))

    return {
        "schema": "vision_selection_readiness_receipt_v1",
        "scope": "registered_renderer_combination_only", "model_use_ready": True,
        "record_result_sha256": sha(record["result"]),
        "smoke_result_sha256": sha(smoke["result"]),
        "smoke_report_sha256": sha(smoke["render_sync_smoke"]),
        "record_context_sha256": sha(record["context"]),
        "smoke_context_sha256": sha(smoke["context"]),
        "record_attestation_sha256": sha(record["attestation"]),
        "smoke_attestation_sha256": sha(smoke["attestation"]),
        "record_job_sha256": sha(record["job"]), "smoke_job_sha256": sha(smoke["job"]),
        "resource_proof_sha256": sha(material["resource_parent"]),
        "live_lease_proof_sha256": sha(material["live_lease_proof"]),
        "training_action_receipt_sha256": sha(material["training_action"]["receipt"]),
        "source_protocol_sha256_lf": source_sha,
        "universal_zero_lag_claim": False, "smoke_raw_state_recomputed": False,
    }


def _freshness_after_resource(repo: Path, material: dict[str, Any]) -> None:
    resource = INPUTS._read(INPUTS._ref(repo, material["resource_parent"]))
    fresh = INPUTS._read(INPUTS._ref(repo, material["training_action"]["freshness"]))
    if SOURCE._stamp(fresh["checked_at_utc"]) < SOURCE._stamp(resource["compute_started_at_utc"]):
        raise ValueError("freshness가 해당 자원 시작보다 이르다")


def build_material(
    repo: Path,
    source_protocol_path: Path,
    source_protocol_sha256_lf: str,
    resource_path: Path,
    runtime_root: Path,
    record_job_id: str,
    record_refs: dict[str, Any],
    smoke_refs: dict[str, Any],
    *,
    resource_locations: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """완료 원장에서 한 쌍의 portable material을 파생한다.

    resource 원문은 저장소 안으로 복사하고 내용·SHA를 보존한다. 반환 값은
    아직 readiness가 아니며 finalize_pair의 전체 검증을 거쳐야 한다.
    """
    repo = Path(repo).resolve(strict=True)
    protocol = load_source_protocol(repo, source_protocol_path, source_protocol_sha256_lf)
    runtime = Path(runtime_root)
    expected_runtime = repo / ".triage-runtime" / "vision-brev" / protocol["protocol_id"]
    if runtime != expected_runtime or runtime.resolve() != runtime:
        raise ValueError("material runtime이 등록 source 경로와 다르다")
    rows = [row for row in protocol["source_action_order"]
            if row["kind"] == "record" and row["job_id"] == record_job_id]
    if len(rows) != 1:
        raise ValueError("material record가 등록 source action이 아니다")
    record = rows[0]
    training = [(index, row) for index, row in enumerate(protocol["source_action_order"])
                if row["kind"] == "train" and row["job_id"] == record["source_job_id"]]
    if len(training) != 1:
        raise ValueError("material source 학습 action이 유일하지 않다")
    ordinal, action = training[0]
    folder = runtime / "actions" / f"{ordinal:02d}_{action['action_id']}"
    paths = {
        "intent": folder / "intent.json", "raw": folder / "raw.json",
        "receipt": folder / "receipt.json", "runtime_manifest": runtime / "manifest.json",
        "freshness": runtime / "freshness.json",
    }
    training_refs = {name: _reference(repo, path) for name, path in paths.items()}
    raw = INPUTS._read(paths["raw"])
    if (raw.get("kind") != "train" or raw.get("action_id") != action["action_id"]
            or raw.get("job_id") != action["job_id"] or not isinstance(raw.get("artifact"), dict)):
        raise ValueError("material source 원장이 등록 학습과 다르다")
    resource_path = INPUTS._regular(Path(resource_path))
    proof = INPUTS._read(resource_path)
    resource_sha = SOURCE.sha256(resource_path)
    record_job = INPUTS._read(INPUTS._ref(repo, record_refs["job"]))
    smoke_job = INPUTS._read(INPUTS._ref(repo, smoke_refs["job"]))
    if (record_job.get("job_id") != record_job_id
            or record_job.get("mode") != INPUTS.RECORD.RECORD_MODE
            or smoke_job.get("job_id") != f"smoke-{record_job_id}"
            or smoke_job.get("mode") != INPUTS.RECORD.RENDER_SYNC_SMOKE_MODE
            or record_job.get("checkpoint") != smoke_job.get("checkpoint")
            or record_job.get("checkpoint", {}).get("logical_id") != record["checkpoint_logical_id"]
            or any(job.get("remote_gpu_admission", {}).get("resource_parent_sha256") != resource_sha
                   for job in (record_job, smoke_job))):
        raise ValueError("material record/smoke/proof가 등록 쌍과 다르다")
    originals = {
        "resource_parent": resource_path,
        "live_lease_proof": Path(proof["raw_gpu_receipt_path"]),
        "budget_admission": Path(proof["budget_admission_path"]),
        "provider_inventory": Path(proof["provider_inventory_path"]),
        "gpu_csv": Path(proof["nvidia_gpu_csv_path"]),
        "process_csv": Path(proof["nvidia_process_csv_path"]),
    }
    if resource_locations is not None:
        INPUTS._exact(resource_locations, set(originals), "보관 resource 위치")
        originals = {name: INPUTS._ref(repo, reference) for name, reference in resource_locations.items()}
        if originals["resource_parent"] != resource_path:
            raise ValueError("보관 resource 위치가 지정한 원문 proof와 다르다")
    data = {name: INPUTS._regular(path).read_bytes() for name, path in originals.items()}
    expected_hashes = {
        "resource_parent": resource_sha, "live_lease_proof": proof["raw_gpu_receipt_sha256"],
        "budget_admission": proof["budget_admission_sha256"],
        "provider_inventory": proof["provider_inventory_sha256"],
        "gpu_csv": proof["nvidia_gpu_csv_sha256"], "process_csv": proof["nvidia_process_csv_sha256"],
    }
    for name, payload in data.items():
        if hashlib.sha256(payload).hexdigest() != expected_hashes[name]:
            raise ValueError("material resource 원문 SHA가 다르다")
    base = runtime / "materials-inputs" / record_job_id
    _target(repo, base / "training_artifact.json")
    base.mkdir(parents=True, exist_ok=True)
    copied = {}
    for name, payload in data.items():
        target = base / (name + (".csv" if name.endswith("csv") else ".json"))
        if target.exists():
            if INPUTS._regular(target).read_bytes() != payload:
                raise ValueError("기존 material resource 원문이 다르다")
        else:
            with target.open("xb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
        copied[name] = _reference(repo, target)
    artifact_path = base / "training_artifact.json"
    if artifact_path.exists():
        if INPUTS._read(artifact_path) != raw["artifact"]:
            raise ValueError("기존 material 학습 artifact가 다르다")
    else:
        SOURCE.write_new(artifact_path, raw["artifact"])
    return {
        "schema": INPUTS.READINESS_SCHEMA,
        "source_training_artifact": _reference(repo, artifact_path),
        "training_action": training_refs, **copied, "record": record_refs, "smoke": smoke_refs,
    }


def verify_material(
    repo: Path,
    source_protocol_path: Path,
    source_protocol_sha256_lf: str,
    material: dict[str, Any],
) -> dict[str, Any]:
    """쌍별 원문을 검증하고 최종 파일을 쓰지 않은 채 manifest 본문을 돌려준다."""
    repo = Path(repo).resolve(strict=True)
    protocol = load_source_protocol(repo, source_protocol_path, source_protocol_sha256_lf)
    INPUTS._exact(material, MATERIAL_KEYS, "readiness material")
    if material["schema"] != INPUTS.READINESS_SCHEMA:
        raise ValueError("readiness material schema가 다르다")
    _freshness_after_resource(repo, material)
    body = {**material, "receipt": _receipt(repo, material, source_protocol_sha256_lf)}
    with tempfile.TemporaryDirectory(prefix=".readiness-check-", dir=repo) as temporary:
        candidate = Path(temporary) / "candidate.json"
        SOURCE.write_new(candidate, body)
        INPUTS.verify_model_readiness(repo, candidate, protocol)
        _freshness_after_resource(repo, material)
        if _receipt(repo, material, source_protocol_sha256_lf) != body["receipt"]:
            raise ValueError("검증 중 readiness 원문이 바뀌었다")
    return body


def finalize_pair(
    repo: Path,
    source_protocol_path: Path,
    source_protocol_sha256_lf: str,
    material: dict[str, Any],
    output_path: Path,
) -> dict[str, str]:
    """상위 render/journal 검증 뒤 쌍별 readiness를 write-once로 내보낸다.

    GPU 조회·앱 시작·모델 요청·새 lease 발급은 수행하지 않는다. 보관 resource
    proof는 원래 발급 시각의 기록으로 검증하며 새 실행 권한으로 사용하지 않는다.
    """
    repo = Path(repo).resolve(strict=True)
    output_path = _target(repo, output_path)
    body = verify_material(repo, source_protocol_path, source_protocol_sha256_lf, material)
    protocol = load_source_protocol(repo, source_protocol_path, source_protocol_sha256_lf)
    if output_path.exists():
        if INPUTS._read(output_path) != body:
            raise ValueError("기존 readiness와 현재 material이 다르다")
        INPUTS.verify_model_readiness(repo, output_path, protocol)
        return _reference(repo, output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    SOURCE.write_new(output_path, body)
    INPUTS.verify_model_readiness(repo, output_path, protocol)
    return _reference(repo, output_path)
