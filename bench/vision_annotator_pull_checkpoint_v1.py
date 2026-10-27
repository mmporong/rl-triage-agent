"""Writer-free annotator pull 결과와 원시 NPZ를 봉인하는 checkpoint 진단 경계."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import record_vision_evidence_v1 as R
import vision_annotator_pull_worker_v1 as WORKER

SCHEMA = "vision_annotator_pull_checkpoint_v1"
MODE = "checkpoint_annotator_pull_diagnostic_only"
VARIANT = "annotator_pull"
TOP_KEYS = frozenset(
    {
        "schema",
        "variant",
        "worker",
        "raw_artifacts",
        "primary_error",
        "diagnostic_complete",
        "used_for_decision",
        "capture_approved",
        "approved",
        "atomicity_claimed",
    }
)


class AnnotatorPullDiagnosticFailure(RuntimeError):
    def __init__(self, report: dict[str, Any], primary: BaseException) -> None:
        super().__init__(f"annotator pull diagnostic failed: {type(primary).__name__}: {primary}")
        self.report = report
        self.envelope = report
        self.primary = primary


def _camera_profile(profile: Any) -> dict[str, Any]:
    if not isinstance(profile, Mapping):
        raise TypeError("annotator pull profile이 mapping이 아니다")
    camera = profile.get("camera")
    if not isinstance(camera, Mapping):
        raise TypeError("annotator pull profile.camera가 mapping이 아니다")
    selected = dict(camera)
    if (
        selected.get("rt_subframes") != 1
        or selected.get("pause_timeline") is not False
        or selected.get("delta_time_s") != 0.0
    ):
        raise ValueError("annotator pull camera render step 계약이 다르다")
    return selected


def _array_meta(value: np.ndarray) -> dict[str, Any]:
    copied = value.copy(order="C")
    return {
        "shape": list(copied.shape),
        "dtype": str(copied.dtype),
        "sha256": hashlib.sha256(copied.tobytes(order="C")).hexdigest(),
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return deepcopy(value)


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError("annotator pull raw NPZ가 이미 있다")
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)  # pyright: ignore[reportArgumentType]
        stream.flush()
        os.fsync(stream.fileno())


def _raw_rows(collection: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    captures = collection.get("captures")
    if not isinstance(captures, list):
        return []
    rows: list[Mapping[str, Any]] = [
        row for row in captures if isinstance(row, Mapping)
    ]
    partial = collection.get("partial_capture")
    if isinstance(partial, Mapping):
        rows.append(partial)
    return rows


def _write_raw_artifacts(
    validated: Mapping[str, Any], collection: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output = Path(validated["output_directory"])
    job_stem = Path(validated["job_path"]).stem
    descriptors: list[dict[str, Any]] = []
    sanitized = deepcopy(dict(collection))
    sanitized_rows = _raw_rows(sanitized)
    for index, (source, target) in enumerate(
        zip(_raw_rows(collection), sanitized_rows, strict=True)
    ):
        raw = source.get("raw")
        if not isinstance(raw, Mapping):
            continue
        semantic = raw.get("semantic_segmentation")
        camera = raw.get("CameraParams")
        times = raw.get("IsaacReadTimes")
        arrays: dict[str, np.ndarray] = {}
        rgb = raw.get("rgb")
        if isinstance(rgb, np.ndarray):
            arrays["rgb"] = rgb
        if isinstance(semantic, Mapping) and isinstance(semantic.get("data"), np.ndarray):
            arrays["semantic_data"] = semantic["data"]
        if isinstance(camera, Mapping):
            for source_key, target_key in (
                ("cameraViewTransform", "camera_view_transform"),
                ("cameraProjection", "camera_projection"),
                ("renderProductResolution", "render_product_resolution"),
            ):
                if source_key in camera:
                    arrays[target_key] = np.asarray(camera[source_key])
        if isinstance(times, Mapping):
            for key, value in times.items():
                if type(value) in (bool, int, float) or isinstance(value, np.number):
                    arrays[f"read_times__{key}"] = np.asarray(value)
        suffix = "partial" if source.get("schema") is None else f"{index:06d}"
        npz_path = output / f"{job_stem}.annotator-pull-{suffix}.npz"
        _write_npz(npz_path, arrays)
        metadata = {
            "schema": "vision_annotator_pull_raw_receipt_v1",
            "ordinal": source.get("ordinal"),
            "complete_capture": source.get("schema") is not None,
            "npz": {
                "file": npz_path.name,
                "sha256": hashlib.sha256(npz_path.read_bytes()).hexdigest(),
                "bytes": npz_path.stat().st_size,
            },
            "arrays": {key: _array_meta(value) for key, value in arrays.items()},
            "semantic_info": (
                _jsonable(dict(semantic.get("info", {})))
                if isinstance(semantic, Mapping)
                else None
            ),
            "camera_params_non_array": (
                {
                    key: _jsonable(value)
                    for key, value in camera.items()
                    if key
                    not in {
                        "cameraViewTransform",
                        "cameraProjection",
                        "renderProductResolution",
                    }
                }
                if isinstance(camera, Mapping)
                else None
            ),
            "arm": _jsonable(source.get("arm")),
            "physics_before": _jsonable(source.get("physics_before")),
            "physics_after": _jsonable(source.get("physics_after")),
            "atomicity_claimed": False,
        }
        receipt_path = output / f"{job_stem}.annotator-pull-{suffix}.json"
        R.write_new(receipt_path, metadata)
        descriptor = {
            "ordinal": source.get("ordinal"),
            "complete_capture": metadata["complete_capture"],
            "npz": metadata["npz"],
            "receipt": {
                "file": receipt_path.name,
                "sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
                "bytes": receipt_path.stat().st_size,
            },
        }
        descriptors.append(descriptor)
        if not isinstance(target, dict):
            raise TypeError("sanitized capture가 mutable dict가 아니다")
        target["raw"] = {"artifact": descriptor, "preserved": True}
    return descriptors, sanitized


def _report(
    worker: Mapping[str, Any] | None,
    artifacts: list[dict[str, Any]],
    primary: BaseException | None,
) -> dict[str, Any]:
    complete = (
        primary is None
        and worker is not None
        and worker.get("diagnostic_complete") is True
        and len(artifacts) == 4
        and all(row.get("complete_capture") is True for row in artifacts)
    )
    return {
        "schema": SCHEMA,
        "variant": VARIANT,
        "worker": deepcopy(dict(worker)) if worker is not None else None,
        "raw_artifacts": deepcopy(artifacts),
        "primary_error": (
            None
            if primary is None
            else {
                "error_type": f"{type(primary).__module__}.{type(primary).__qualname__}",
                "error_message": str(primary),
            }
        ),
        "diagnostic_complete": complete,
        "used_for_decision": False,
        "capture_approved": False,
        "approved": False,
        "atomicity_claimed": False,
    }


def run_annotator_pull_diagnostic(
    validated: Mapping[str, Any],
    app: Any,
    *,
    seal_report: Callable[[Mapping[str, Any]], None],
    dependencies: Mapping[str, Any] | None = None,
    run_worker: Callable[..., dict[str, Any]] = WORKER.run_worker,
) -> dict[str, Any]:
    job = validated.get("job")
    if (
        not isinstance(job, Mapping)
        or job.get("mode") != MODE
        or job.get("callback_variant") != VARIANT
    ):
        raise ValueError("annotator pull diagnostic job 계약이 다르다")
    _camera_profile(job.get("profile"))
    if not callable(seal_report):
        raise TypeError("annotator pull seal_report가 callable이 아니다")
    checkpoint = Path(validated.get("checkpoint_path", ""))
    if checkpoint.is_symlink() or not checkpoint.is_file():
        raise ValueError("annotator pull checkpoint가 regular file이 아니다")

    worker: dict[str, Any] | None = None
    primary: BaseException | None = None
    artifacts: list[dict[str, Any]] = []
    inner = dict(validated)
    inner["job"] = {**dict(job), "mode": WORKER.MODE}
    try:
        worker = run_worker(inner, app, dependencies=dependencies)
    except BaseException as error:  # noqa: BLE001 - raw를 쓴 뒤 최초 오류를 재발생시킨다.
        raw = getattr(error, "envelope", getattr(error, "report", None))
        if isinstance(raw, Mapping):
            worker = dict(raw)
        primary = getattr(error, "primary", error)
    collection = worker.get("collection") if isinstance(worker, Mapping) else None
    if isinstance(collection, Mapping):
        try:
            artifacts, sanitized = _write_raw_artifacts(validated, collection)
            if not isinstance(worker, dict):
                raise TypeError("annotator pull worker envelope가 dict가 아니다")
            worker = {**worker, "collection": sanitized}
        except BaseException as artifact_error:  # noqa: BLE001
            if primary is None:
                primary = artifact_error
    report = _report(worker, artifacts, primary)
    seal_report(report)
    if primary is not None:
        raise AnnotatorPullDiagnosticFailure(report, primary) from primary
    if report["diagnostic_complete"] is not True:
        error = RuntimeError("annotator pull diagnostic이 완결되지 않았다")
        raise AnnotatorPullDiagnosticFailure(report, error) from error
    return report
