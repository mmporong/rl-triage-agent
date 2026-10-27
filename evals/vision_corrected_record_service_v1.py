"""CORRECTED native annotator-pull 결과를 검증하고 V01 근거 계약으로 봉인한다."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import run_vision_corrected_pull_record_v1 as RUNNER
import vision_evidence_contract_v1 as V01
import vision_pull_record_worker_v3 as WORKER

SCHEMA = "vision_corrected_native_record_receipt_v1"
ARTIFACT_KEYS = {
    "video", "raw_trace", "encoder_receipt", "timing_receipt",
    "measurement_1", "spec_1", "measurement_2", "spec_2",
}
FALSE_FLAGS = {
    "capture_approved": False,
    "model_ready": False,
    "used_for_decision": False,
}


def _read(path: Path) -> dict[str, Any]:
    path = Path(path).resolve(strict=True)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"JSON이 regular file이 아니다: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("JSON root가 객체가 아니다")
    return value


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _descriptor(path: Path) -> dict[str, Any]:
    path = Path(path).resolve(strict=True)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"artifact가 regular file이 아니다: {path}")
    return {"file": path.name, "sha256": _sha(path), "bytes": path.stat().st_size}


def _reference(path: Path) -> dict[str, Any]:
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": _sha(path), "bytes": path.stat().st_size}


def _artifact_paths(output: Path, descriptors: Any) -> dict[str, Path]:
    if not isinstance(descriptors, Mapping) or set(descriptors) != ARTIFACT_KEYS:
        raise ValueError("CORRECTED native artifact 8개 계약이 다르다")
    output = Path(output).resolve(strict=True)
    result: dict[str, Path] = {}
    for key, descriptor in descriptors.items():
        if not isinstance(descriptor, Mapping) or set(descriptor) != {"file", "sha256", "bytes"}:
            raise ValueError(f"CORRECTED artifact {key} descriptor가 다르다")
        matches = [path for path in output.rglob(str(descriptor["file"])) if path.is_file() and not path.is_symlink()]
        if len(matches) != 1:
            raise ValueError(f"CORRECTED artifact {key}가 output 안에서 유일하지 않다")
        path = matches[0].resolve(strict=True)
        if output not in path.parents or _descriptor(path) != dict(descriptor):
            raise ValueError(f"CORRECTED artifact {key} 원문/SHA가 다르다")
        result[str(key)] = path
    return result


def _verify_execution(stage: Path, validated: Mapping[str, Any], result_path: Path) -> tuple[Path, dict[str, Any]]:
    execution_path = stage / "execution.json"
    execution = _read(execution_path)
    result = _read(result_path)
    if (
        execution.get("schema") != "vision_pull_record_execution_v3"
        or execution.get("completed") is not True
        or execution.get("technical_recording_complete") is not True
        or execution.get("cleanup_verified") is not True
        or execution.get("exit_code") != 0
        or execution.get("timed_out") is not False
        or execution.get("gpu_after") != []
        or execution.get("job_sha256") != validated["job_sha256"]
        or execution.get("checkpoint_sha256") != validated["checkpoint_sha256"]
        or execution.get("result_sha256") != _sha(result_path)
        or any(execution.get(key) is not expected for key, expected in FALSE_FLAGS.items())
        or not isinstance(execution.get("phases"), list)
        or [row.get("name") for row in execution["phases"]] != ["aaba", "record"]
        or not all(row.get("cleanup_verified") is True and row.get("exit_code") == 0
                   for row in execution["phases"])
    ):
        raise ValueError("CORRECTED technical execution/cleanup 결합이 다르다")
    if (
        result.get("schema") != RUNNER.RESULT_SCHEMA
        or result.get("completed") is not True
        or result.get("technical_recording_complete") is not True
        or result.get("checkpoint_logical_id") not in RUNNER.CHECKPOINT_IDS
        or result.get("checkpoint_sha256") != validated["checkpoint_sha256"]
        or result.get("job_sha256") != validated["job_sha256"]
        or result.get("training_actions") != 0
        or result.get("model_api_requests") != 0
        or any(result.get(key) is not expected for key, expected in FALSE_FLAGS.items())
    ):
        raise ValueError("CORRECTED native result 결합이 다르다")
    return execution_path, execution


def _verify_aaba(stage: Path, validated: Mapping[str, Any], execution: Mapping[str, Any]) -> tuple[Path, list[dict[str, Any]]]:
    aaba_path = stage / "aaba-output" / "native-aaba-preflight.report.json"
    aaba = _read(aaba_path)
    receipt = execution.get("aaba_receipt")
    if (
        not isinstance(receipt, Mapping)
        or receipt.get("diagnostic_complete") is not True
        or receipt.get("checkpoint_sha256") != validated["checkpoint_sha256"]
        or receipt.get("report") != _descriptor(aaba_path)
        or receipt.get("capture_approved") is not False
        or receipt.get("used_for_decision") is not False
        or aaba.get("diagnostic_complete") is not True
        or aaba.get("approved") is not False
        or aaba.get("atomicity_claimed") is not False
        or aaba.get("capture_approved") is not False
        or aaba.get("used_for_decision") is not False
    ):
        raise ValueError("CORRECTED AABA 원문 결합이 다르다")
    raw_rows = aaba.get("raw_artifacts")
    if not isinstance(raw_rows, list) or len(raw_rows) != 4:
        raise ValueError("CORRECTED AABA raw artifact가 네 건이 아니다")
    checked: list[dict[str, Any]] = []
    for ordinal, raw in enumerate(raw_rows):
        if (not isinstance(raw, Mapping) or raw.get("ordinal") != ordinal
                or raw.get("complete_capture") is not True):
            raise ValueError("CORRECTED AABA raw artifact 순서·완결성이 다르다")
        npz_descriptor, receipt_descriptor = raw.get("npz"), raw.get("receipt")
        if not isinstance(npz_descriptor, Mapping) or not isinstance(receipt_descriptor, Mapping):
            raise TypeError("CORRECTED AABA raw descriptor가 없다")
        npz_path = aaba_path.parent / str(npz_descriptor.get("file", ""))
        receipt_path = aaba_path.parent / str(receipt_descriptor.get("file", ""))
        if _descriptor(npz_path) != dict(npz_descriptor) or _descriptor(receipt_path) != dict(receipt_descriptor):
            raise ValueError("CORRECTED AABA raw 원문 SHA가 descriptor와 다르다")
        receipt_row = _read(receipt_path)
        arrays = receipt_row.get("arrays")
        if (receipt_row.get("schema") != "vision_annotator_pull_raw_receipt_v1"
                or receipt_row.get("ordinal") != ordinal
                or receipt_row.get("complete_capture") is not True
                or receipt_row.get("npz") != dict(npz_descriptor)
                or receipt_row.get("atomicity_claimed") is not False
                or receipt_row.get("physics_before") != receipt_row.get("physics_after")
                or not isinstance(arrays, Mapping) or not arrays):
            raise ValueError("CORRECTED AABA raw receipt 내용이 다르다")
        with np.load(npz_path, allow_pickle=False) as archive:
            if set(archive.files) != set(arrays):
                raise ValueError("CORRECTED AABA NPZ 배열 목록이 다르다")
            for name in archive.files:
                array = np.asarray(archive[name])
                metadata = arrays[name]
                if (not isinstance(metadata, Mapping)
                        or metadata.get("shape") != list(array.shape)
                        or metadata.get("dtype") != str(array.dtype)
                        or metadata.get("sha256") != hashlib.sha256(
                            array.copy(order="C").tobytes(order="C")
                        ).hexdigest()):
                    raise ValueError("CORRECTED AABA NPZ 배열 metadata가 다르다")
        checked.append({"ordinal": ordinal, "npz": _reference(npz_path),
                        "receipt": _reference(receipt_path)})
    return aaba_path, checked


def _verify_raw_frames(stage: Path, manifest: Mapping[str, Any]) -> None:
    frames = manifest.get("frames")
    if not isinstance(frames, list) or len(frames) != 625:
        raise ValueError("CORRECTED native raw frame 수가 625가 아니다")
    for ordinal, row in enumerate(frames):
        if not isinstance(row, Mapping) or row.get("ordinal") != ordinal:
            raise ValueError("CORRECTED native raw frame ordinal이 다르다")
        npz_path = stage / str(row.get("npz", {}).get("file", ""))
        raw_path = stage / str(row.get("raw_receipt", {}).get("file", ""))
        pull_path = stage / str(row.get("pull_receipt", {}).get("file", ""))
        if (_descriptor(npz_path) != row.get("npz") or _descriptor(raw_path) != row.get("raw_receipt")
                or _descriptor(pull_path) != row.get("pull_receipt")):
            raise ValueError("CORRECTED native raw frame descriptor가 다르다")
        raw = _read(raw_path)
        if raw.get("npz") != row.get("npz") or raw.get("capture_ordinal") != ordinal:
            raise ValueError("CORRECTED native raw receipt가 NPZ와 다르다")
        with np.load(npz_path, allow_pickle=False) as archive:
            files = set(archive.files)
            required_groups = {
                "rgb": any(name == "rgb" for name in files),
                "semantic_segmentation": any(name.startswith("semantic_") for name in files),
                "CameraParams": any(name.startswith("camera_") or name == "render_product_resolution"
                                    for name in files),
                "IsaacReadTimes": any(name.startswith("read_times__") for name in files),
                "ReferenceTime": any(name.startswith("reference_time__") for name in files),
            }
            if (raw.get("raw_fields") != list(WORKER.CAPTURE.ANNOTATORS)
                    or not files or not all(required_groups.values())):
                raise ValueError("CORRECTED native raw NPZ field가 다르다")


def _verify_worker(stage: Path, validated: Mapping[str, Any], result: Mapping[str, Any]) -> tuple[dict[str, Path], Path]:
    worker = result.get("worker")
    if (
        not isinstance(worker, Mapping)
        or worker.get("schema") != "vision_pull_record_worker_v3"
        or worker.get("completed") is not True
        or worker.get("technical_recording_complete") is not True
        or worker.get("frame_count") != 625
        or worker.get("fps") != 50
        or worker.get("writer_calls") != {"attach": 0, "callback": 0, "create": 0, "detach": 0}
        or worker.get("atomicity_claimed") is not False
        or worker.get("checkpoint_sha256") != validated["checkpoint_sha256"]
        or worker.get("job_sha256") != validated["job_sha256"]
        or any(worker.get(key) is not expected for key, expected in FALSE_FLAGS.items())
    ):
        raise ValueError("CORRECTED native worker 결과가 다르다")
    output = stage / "output"
    artifacts = _artifact_paths(output, worker.get("artifacts"))
    if not WORKER.PREVIOUS.RECORD_VERIFY._record_artifacts(validated, artifacts):
        raise ValueError("CORRECTED native 8-artifact trace/PTS 결합이 다르다")
    manifest_path = output / WORKER.MANIFEST_NAME
    manifest = _read(manifest_path)
    if worker.get("native_capture_manifest") != _descriptor(manifest_path):
        raise ValueError("CORRECTED native manifest descriptor가 다르다")
    rebuilt = WORKER._manifest(validated, artifacts)
    if manifest != rebuilt:
        raise ValueError("CORRECTED native manifest 재계산 결과가 다르다")
    _verify_raw_frames(stage, manifest)
    return artifacts, manifest_path


def _v01_contracts(
    artifacts: Mapping[str, Path], output: Path, checkpoint_sha256: str
) -> list[dict[str, Any]]:
    evidence = Path(output) / "evidence"
    evidence.mkdir(exist_ok=False)
    rows: list[dict[str, Any]] = []
    for window in (1, 2):
        contract = V01.prepare(artifacts[f"spec_{window}"], evidence / f"window-{window:02d}")
        checked = V01.verify(contract)
        if (checked["provenance"]["checkpoint"]["sha256"] != checkpoint_sha256
                or checked["provenance"]["video"]["sha256"] != _sha(artifacts["video"])
                or checked["behavior_ground_truth"] is not None
                or checked["fixed_evaluation_used_as_behavior_ground_truth"] is not False):
            raise ValueError("CORRECTED V01 checkpoint/video/ground-truth 결합이 다르다")
        rows.append({"window": window, "contract": _reference(contract), "verified": True})
    return rows


def _normal_reference(value: Mapping[str, Any] | None, seed: int, profile: Mapping[str, Any]) -> dict[str, Any] | None:
    if value is None:
        return None
    required = {"seed", "role", "checkpoint", "fixed_eval", "record_profile", "record_receipt"}
    if (set(value) != required or value.get("seed") != seed or value.get("role") != "NONE"
            or value.get("fixed_eval") != "healthy" or value.get("record_profile") != dict(profile)):
        raise ValueError("CORRECTED 동일 seed NONE 참조가 다르다")
    result: dict[str, Any] = {"seed": seed, "role": "NONE", "fixed_eval": "healthy",
                              "record_profile": dict(profile)}
    for key in ("checkpoint", "record_receipt"):
        ref = value[key]
        if not isinstance(ref, Mapping) or set(ref) != {"path", "sha256"}:
            raise ValueError(f"CORRECTED NONE {key} ref가 다르다")
        path = Path(str(ref["path"])).resolve(strict=True)
        if path.is_symlink() or _sha(path) != ref["sha256"]:
            raise ValueError(f"CORRECTED NONE {key} 원문/SHA가 다르다")
        result[key] = {"path": str(path), "sha256": ref["sha256"]}
    return result


def verify_native_result(
    protocol_path: Path,
    job_path: Path,
    correction_result: Path,
    approval_receipt: Path,
    output: Path,
    *,
    normal_reference: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """실제 RUN3 원문을 검증하고 별도 native receipt를 write-once로 만든다."""
    validated = RUNNER.validate_job(job_path, require_fresh=False)
    registered_protocol = Path(protocol_path).resolve(strict=True)
    if Path(validated["protocol_path"]).resolve(strict=True) != registered_protocol:
        raise ValueError("CORRECTED service protocol과 record job protocol이 다르다")
    stage = Path(job_path).resolve(strict=True).parent
    result_path = stage / "output" / "result.json"
    result = _read(result_path)
    execution_path, execution = _verify_execution(stage, validated, result_path)
    aaba_path, aaba_raw = _verify_aaba(stage, validated, execution)
    artifacts, manifest_path = _verify_worker(stage, validated, result)
    protocol = validated["protocol"]
    logical = result["checkpoint_logical_id"]
    binding = RUNNER._verified_binding(logical, correction_result, approval_receipt)
    row = next(item for item in protocol["checkpoints"] if item["logical_id"] == logical)
    if binding != row or binding["checkpoint"]["sha256"] != result["checkpoint_sha256"]:
        raise ValueError("CORRECTED late binding과 native result가 다르다")
    normal = _normal_reference(normal_reference, RUNNER.CATALOG[logical]["seed"], protocol["record_profile"])
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    contracts = _v01_contracts(artifacts, output, validated["checkpoint_sha256"])
    receipt = {
        "schema": SCHEMA,
        "status": "verified_native_record",
        "logical_id": logical,
        "seed": RUNNER.CATALOG[logical]["seed"],
        "correction_job_id": RUNNER.CATALOG[logical]["correction_job_id"],
        "checkpoint_sha256": result["checkpoint_sha256"],
        "record_profile": protocol["record_profile"],
        "technical_execution": _reference(execution_path),
        "native_result": _reference(result_path),
        "aaba_raw_source": {"report": _reference(aaba_path), "artifacts": aaba_raw},
        "native_capture_manifest": _reference(manifest_path),
        "artifacts": {key: _reference(path) for key, path in artifacts.items()},
        "correction_result": _reference(correction_result),
        "approval_receipt": _reference(approval_receipt),
        "v01_contracts": contracts,
        "normal_reference": normal,
        "claims": {"new_recovery_claim": False, "behavior_ground_truth": None},
        "post_observer": {"used_for_decision": False},
        **FALSE_FLAGS,
    }
    receipt_path = output / "receipt.json"
    RUNNER.JOB3.write_new(receipt_path, receipt)
    return {"receipt": receipt_path, "value": receipt}


class NativeCorrectedRecordService:
    """E2E가 새 CORRECTED recorder와 native verifier를 한 경계로 호출한다."""

    def __init__(self, protocol_path: Path):
        self.protocol_path = Path(protocol_path).resolve(strict=True)

    def record(
        self,
        *,
        resource_proof: Path,
        resource_sha256: str,
        isaaclab: Path,
        python: Path,
        stage: Path,
        logical_id: str,
        correction_result: Path,
        approval_receipt: Path,
        receipt_output: Path,
        normal_reference: Mapping[str, Any] | None = None,
        run: Callable[[Path], Mapping[str, Any]] = RUNNER.run_job,
    ) -> dict[str, Any]:
        job = RUNNER.create_job(self.protocol_path, resource_proof, resource_sha256,
                                isaaclab, python, stage, logical_id)
        result = run(job)
        if not isinstance(result, Mapping) or result.get("completed") is not True:
            return {"status": "HOLD", "reason": "native_record_incomplete", "job": str(job)}
        checked = verify_native_result(self.protocol_path, job, correction_result,
                                       approval_receipt, receipt_output,
                                       normal_reference=normal_reference)
        return {"status": "complete", "job": str(job),
                "native_receipt": _reference(checked["receipt"])}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--correction-result", type=Path, required=True)
    parser.add_argument("--approval-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    verify_native_result(args.protocol, args.job, args.correction_result,
                         args.approval_receipt, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
