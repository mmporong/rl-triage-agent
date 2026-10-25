"""Replicator 1.11.35 Writer payload를 atomic frame 계약으로 변환하고 봉인한다."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import record_vision_evidence_v1 as R
import vision_frame_sync_v1 as FRAME
import vision_timing_adapter_v1 as TIMING

SCHEMA = "vision_atomic_capture_journal_v1"
TOP_LEVEL_KEYS = frozenset(
    {
        "swhFrameNumber",
        "reference_time",
        "distribution_outputs",
        "trigger_outputs",
        "named_outputs",
        "annotators",
    }
)
ANNOTATOR_KEYS = frozenset(
    {
        "camera",
        "resolution",
        "rgb",
        "semantic_segmentation",
        "CameraParams",
        "IsaacReadTimes",
    }
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _single_render_product(value: Any, label: str) -> tuple[str, Any]:
    if not isinstance(value, Mapping) or len(value) != 1:
        raise ValueError(f"{label}은 단일 RenderProduct mapping이어야 한다")
    key, item = next(iter(value.items()))
    if not isinstance(key, str) or not key:
        raise ValueError(f"{label} RenderProduct key가 잘못됐다")
    return key, item


def _array(
    value: Any, label: str, *, shape: tuple[int, ...] | None = None
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{label}은 ndarray여야 한다")
    if shape is not None and value.shape != shape:
        raise ValueError(f"{label} shape가 다르다")
    return value.copy(order="C")


def _matrix16(value: Any, label: str) -> list[list[float]]:
    raw = np.asarray(value)
    if raw.shape != (16,) or raw.dtype.kind not in "fiu":
        raise ValueError(f"{label}은 flat16 수치 배열이어야 한다")
    matrix = raw.astype(np.float64, copy=False)
    if not np.isfinite(matrix).all():
        raise ValueError(f"{label}은 finite여야 한다")
    return matrix.reshape(4, 4).tolist()


def _json_safe(value: Any, label: str) -> Any:
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not np.isfinite(value):
            raise ValueError(f"{label}에 nonfinite 값이 있다")
        return value
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "biuf" or not np.isfinite(value).all():
            raise ValueError(f"{label} ndarray를 보존할 수 없다")
        return {
            "encoding": "ndarray",
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "values": value.tolist(),
        }
    if (
        type(value).__module__.split(".", 1)[0] == "numpy"
        and getattr(value, "shape", None) == ()
    ):
        return _json_safe(value.item(), f"{label}.item")
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError(f"{label} mapping key는 문자열이어야 한다")
        return {
            key: _json_safe(item, f"{label}.{key}")
            for key, item in sorted(value.items())
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item, f"{label}[]") for item in value]
    raise TypeError(f"{label} 값을 JSON evidence로 보존할 수 없다")


def _decode_json_safe(value: Any, label: str) -> Any:
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not np.isfinite(value):
            raise ValueError(f"{label}에 nonfinite 값이 있다")
        return value
    if isinstance(value, list):
        return [_decode_json_safe(item, f"{label}[]") for item in value]
    if not isinstance(value, dict):
        raise TypeError(f"{label} JSON evidence 형식이 다르다")
    if set(value) == {"encoding", "dtype", "shape", "values"}:
        if value.get("encoding") != "ndarray" or not isinstance(value.get("shape"), list):
            raise ValueError(f"{label} ndarray evidence 형식이 다르다")
        decoded = _decode_json_safe(value.get("values"), f"{label}.values")
        array = np.asarray(decoded)
        if list(array.shape) != value["shape"] or not isinstance(value.get("dtype"), str):
            raise ValueError(f"{label} ndarray shape가 values와 다르다")
        return decoded
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{label} mapping key가 문자열이 아니다")
    return {key: _decode_json_safe(item, f"{label}.{key}") for key, item in value.items()}


def _typed_timing(
    value: Any, required: set[str], label: str, *, allow_execution_port: bool = False
) -> dict[str, Any]:
    allowed = (required, required | {TIMING.EXECUTION_PORT}) if allow_execution_port else (required,)
    if (
        not isinstance(value, dict)
        or set(value) != {"type", "fields", "non_string_key_types"}
        or not isinstance(value.get("type"), str)
        or not value["type"]
        or not isinstance(value.get("fields"), dict)
        or value.get("non_string_key_types") != []
        or set(value["fields"]) not in allowed
    ):
        raise ValueError(f"{label} typed mapping 형식이 다르다")
    decoded = {}
    for key in required:
        scalar = value["fields"].get(key)
        if (
            not isinstance(scalar, dict)
            or set(scalar) != {"type", "encoding", "value"}
            or not isinstance(scalar.get("type"), str)
            or not isinstance(scalar.get("encoding"), str)
            or not scalar["encoding"].endswith("json_native")
        ):
            raise ValueError(f"{label}.{key} scalar를 실제 값으로 복원할 수 없다")
        decoded[key] = scalar["value"]
    return decoded


def verify_writer_bundle(
    journal: Mapping[str, Any], bundle: Mapping[str, Any], mode: str, ordinal: int
) -> None:
    """보존한 Writer 원문 metadata를 normalized atomic bundle에 다시 결합한다."""
    writer = journal.get("writer_payload")
    keys = {
        "mode", "ordinal", "callback_ref", "render_product_ref", "swh_frame_number",
        "writer_reference_time", "reference_time", "isaac_read_times", "camera_params",
        "camera", "resolution",
    }
    if not isinstance(writer, dict) or set(writer) != keys:
        raise ValueError("atomic writer payload 필드가 다르다")
    if (
        writer.get("mode") != mode
        or writer.get("ordinal") != ordinal
        or writer.get("callback_ref") != bundle.get("adapter_callback_ref")
        or writer.get("render_product_ref") != bundle.get("render_product_ref")
        or type(writer.get("swh_frame_number")) is not int
        or not isinstance(writer.get("camera"), str)
        or not writer["camera"]
    ):
        raise ValueError("atomic writer identity가 bundle과 다르다")
    read_times = _typed_timing(
        writer.get("isaac_read_times"), set(R.ISAAC_READ_TIMES_FIELDS),
        "atomic IsaacReadTimes", allow_execution_port=True,
    )
    reference = _typed_timing(
        writer.get("reference_time"), set(TIMING.REFERENCE_FIELDS), "atomic ReferenceTime"
    )
    physics = bundle.get("physics")
    if not isinstance(physics, Mapping):
        raise TypeError("atomic bundle physics가 없다")
    simulation_time = R._finite(
        physics.get("simulation_time_before_s"), "atomic bundle simulation time"
    )
    normalized = TIMING.normalize_render_timing(
        read_times, reference, simulation_time
    )
    if (
        normalized != bundle.get("timing")
        or writer.get("writer_reference_time")
        != [reference.get("referenceTimeNumerator"), reference.get("referenceTimeDenominator")]
    ):
        raise ValueError("atomic writer timing 원문이 bundle과 다르다")
    camera_params = _decode_json_safe(writer.get("camera_params"), "CameraParams")
    resolution = _decode_json_safe(writer.get("resolution"), "resolution")
    camera = bundle.get("camera")
    if not isinstance(camera_params, Mapping) or not isinstance(camera, Mapping):
        raise TypeError("atomic CameraParams 또는 bundle camera가 없다")
    try:
        view = np.asarray(camera_params["cameraViewTransform"], dtype=np.float64).reshape(4, 4)
        projection = np.asarray(camera_params["cameraProjection"], dtype=np.float64).reshape(4, 4)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("atomic CameraParams matrix가 flat16이 아니다") from error
    if (
        not np.isfinite(view).all()
        or not np.isfinite(projection).all()
        or view.tolist() != camera.get("view_matrix")
        or projection.tolist() != camera.get("projection")
        or camera_params.get("renderProductResolution") != camera.get("resolution")
        or resolution != camera.get("resolution")
    ):
        raise ValueError("atomic CameraParams 원문이 bundle과 다르다")
    if (
        writer.get("camera") != R.fixed_profile()["camera"]["camera_prim_path"]
        or resolution != [R.WIDTH, R.HEIGHT]
    ):
        raise ValueError("atomic Writer camera path 또는 resolution이 fixed profile과 다르다")


def decode_writer_payload(payload: Any) -> dict[str, Any]:
    """OgnWriter의 annotator 구조를 단일 RP atomic payload로 엄격히 복사한다."""
    if not isinstance(payload, Mapping) or set(payload) != TOP_LEVEL_KEYS:
        raise ValueError("Writer top-level payload 필드가 다르다")
    for name in ("distribution_outputs", "trigger_outputs", "named_outputs"):
        if not isinstance(payload[name], Mapping):
            raise TypeError(f"Writer {name}은 mapping이어야 한다")
    annotators = payload["annotators"]
    if not isinstance(annotators, Mapping) or set(annotators) != ANNOTATOR_KEYS:
        raise ValueError("Writer annotator payload 필드가 다르다")
    rows = {
        name: _single_render_product(annotators[name], name) for name in ANNOTATOR_KEYS
    }
    render_products = {row[0] for row in rows.values()}
    if len(render_products) != 1:
        raise ValueError("Writer annotator RenderProduct identity가 서로 다르다")
    render_product = next(iter(render_products))

    resolution = rows["resolution"][1]
    if (
        not isinstance(resolution, (list, tuple, np.ndarray))
        or len(resolution) != 2
        or any(
            isinstance(item, (bool, np.bool_))
            or not isinstance(item, (int, np.integer))
            or int(item) <= 0
            for item in resolution
        )
    ):
        raise ValueError("Writer resolution이 [width,height]가 아니다")
    width, height = (int(resolution[0]), int(resolution[1]))
    rgba = _array(rows["rgb"][1], "Writer rgb", shape=(height, width, 4))
    if rgba.dtype != np.uint8:
        raise ValueError("Writer rgb는 uint8 RGBA여야 한다")
    rgb = rgba[:, :, :3].copy(order="C")

    semantic = rows["semantic_segmentation"][1]
    if not isinstance(semantic, Mapping) or set(semantic) != {"data", "info"}:
        raise ValueError("semantic_segmentation payload 필드가 다르다")
    mask = _array(semantic["data"], "semantic mask")
    if mask.shape == (height, width, 1):
        mask = mask[:, :, 0].copy(order="C")
    if mask.shape != (height, width) or mask.dtype != np.uint32:
        raise ValueError("semantic mask는 resolution과 같은 uint32 HxW여야 한다")
    info = semantic["info"]
    if not isinstance(info, Mapping) or set(info) != {"idToLabels"}:
        raise ValueError("semantic info는 idToLabels만 가져야 한다")
    labels = deepcopy(info["idToLabels"])
    if not isinstance(labels, Mapping):
        raise TypeError("semantic idToLabels는 mapping이어야 한다")

    camera_params = rows["CameraParams"][1]
    required_camera = {
        "cameraViewTransform",
        "cameraProjection",
        "renderProductResolution",
    }
    if not isinstance(camera_params, Mapping) or not required_camera <= set(
        camera_params
    ):
        raise ValueError("CameraParams 필수 필드가 없다")
    camera_resolution = camera_params["renderProductResolution"]
    if list(camera_resolution) != [width, height]:
        raise ValueError("CameraParams resolution과 RenderProduct resolution이 다르다")
    camera = {
        "view_matrix": _matrix16(
            camera_params["cameraViewTransform"], "cameraViewTransform"
        ),
        "projection": _matrix16(camera_params["cameraProjection"], "cameraProjection"),
        "resolution": [width, height],
    }

    raw_read_times = rows["IsaacReadTimes"][1]
    if not isinstance(raw_read_times, Mapping):
        raise TypeError("IsaacReadTimes는 mapping이어야 한다")
    reference = payload["reference_time"]
    if (
        not isinstance(reference, (list, tuple))
        or len(reference) != 2
        or any(type(item) is not int for item in reference)
    ):
        raise ValueError("Writer reference_time은 정수 numerator/denominator여야 한다")
    reference_time = {
        "referenceTimeNumerator": reference[0],
        "referenceTimeDenominator": reference[1],
    }
    TIMING._shape(raw_read_times, reference_time)
    read_times = {
        field: deepcopy(raw_read_times[field]) for field in R.ISAAC_READ_TIMES_FIELDS
    }
    writer_swh_frame = deepcopy(payload["swhFrameNumber"])
    if type(writer_swh_frame) is not int:
        raise ValueError("Writer top-level swhFrameNumber가 정수가 아니다")
    frame = read_times.get("frameNumber")
    swh = read_times.get("swhFrameNumber")
    if type(frame) is not int or type(swh) is not int:
        raise ValueError("IsaacReadTimes frameNumber가 정수가 아니다")
    callback_ref = _sha(
        (
            json.dumps(
                [render_product, list(reference), frame, swh], separators=(",", ":")
            )
            + "\n"
        ).encode()
    )
    return {
        "callback_ref": callback_ref,
        "render_product_ref": render_product,
        "rgb": rgb,
        "semantic": {"mask": mask, "idToLabels": labels},
        "camera": camera,
        "camera_params_raw": deepcopy(camera_params),
        "camera_path": deepcopy(rows["camera"][1]),
        "render_product_resolution": deepcopy(rows["resolution"][1]),
        "writer_swh_frame_number": writer_swh_frame,
        "read_times": read_times,
        "read_times_observed": dict(raw_read_times),
        "read_times_raw": TIMING._typed_mapping(raw_read_times, "IsaacReadTimes"),
        "reference_time": reference_time,
        "reference_time_raw": TIMING._typed_mapping(reference_time, "ReferenceTime"),
        "writer_reference_time": list(reference),
    }


def frame_payload(decoded: Mapping[str, Any]) -> dict[str, Any]:
    callback_ref = decoded["callback_ref"]
    values = {
        "rgb": decoded["rgb"],
        "semantic": decoded["semantic"],
        "camera": decoded["camera"],
        "read_times": decoded["read_times"],
        "reference_time": decoded["reference_time"],
    }
    return {
        "adapter_callback_ref": callback_ref,
        "render_product_ref": decoded["render_product_ref"],
        **{
            name: {"adapter_callback_ref": callback_ref, "data": value}
            for name, value in values.items()
        },
    }


def create_payload_writer(rep: Any, sink: Callable[[dict[str, Any]], None]) -> Any:
    """SDK import를 호출 시점으로 제한한 custom Writer instance를 만든다."""
    if not callable(sink):
        raise TypeError("Writer sink가 callable이 아니다")

    class AtomicPayloadWriter(rep.Writer):
        data_structure = "annotator"

        def __init__(self) -> None:
            self.annotators = [
                rep.AnnotatorRegistry.get_annotator("rgb"),
                rep.AnnotatorRegistry.get_annotator(
                    "semantic_segmentation", init_params={"colorize": False}
                ),
                rep.AnnotatorRegistry.get_annotator("CameraParams"),
                rep.AnnotatorRegistry.get_annotator("IsaacReadTimes"),
            ]
            self.callback_count = 0
            self.callback_error: BaseException | None = None
            self.last_decoded: dict[str, Any] | None = None

        def write(self, data: dict[str, Any]) -> None:
            self.callback_count += 1
            if self.callback_error is not None:
                return
            try:
                decoded = decode_writer_payload(data)
                self.last_decoded = decoded
                sink(decoded)
            except BaseException as error:  # noqa: BLE001 - Kit 밖에서 다시 발생시킨다.
                self.callback_error = error

        def raise_callback_error(self) -> None:
            if self.callback_error is not None:
                raise self.callback_error

    return AtomicPayloadWriter()


def _paths(job_path: Path, ordinal: int) -> tuple[Path, Path]:
    if type(ordinal) is not int or ordinal < 0:
        raise ValueError("atomic journal ordinal이 잘못됐다")
    job = Path(job_path)
    if job.is_symlink() or not job.is_file():
        raise ValueError("atomic journal job은 regular file이어야 한다")
    stem = f"{job.stem}.atomic-{ordinal:06d}"
    return job.with_name(stem + ".json"), job.with_name(stem + ".npz")


def write_atomic_journal(
    job_path: Path,
    ordinal: int,
    mode: str,
    decoded: Mapping[str, Any],
    bundle: Mapping[str, Any],
) -> tuple[Path, Path]:
    """성공한 atomic capture의 RGB·mask 원문과 callback metadata를 write-once로 봉인한다."""
    json_path, npz_path = _paths(job_path, ordinal)
    if json_path.exists() or npz_path.exists():
        raise FileExistsError("atomic capture journal이 이미 있다")
    rgb = _array(decoded.get("rgb"), "journal rgb")
    semantic = decoded.get("semantic")
    if not isinstance(semantic, Mapping):
        raise TypeError("journal semantic payload가 없다")
    mask = _array(semantic.get("mask"), "journal mask")
    with npz_path.open("xb") as stream:
        np.savez(stream, rgb=rgb, semantic_mask=mask)
        stream.flush()
        os.fsync(stream.fileno())
    npz_bytes = npz_path.read_bytes()
    body = {
        "schema": SCHEMA,
        "bundle": deepcopy(dict(bundle)),
        "raw": {"file": npz_path.name, "sha256": _sha(npz_bytes)},
        "writer_payload": {
            "mode": mode,
            "ordinal": ordinal,
            "callback_ref": decoded.get("callback_ref"),
            "render_product_ref": decoded.get("render_product_ref"),
            "swh_frame_number": decoded.get("writer_swh_frame_number"),
            "writer_reference_time": _json_safe(
                decoded.get("writer_reference_time"), "writer_reference_time"
            ),
            "reference_time": deepcopy(decoded.get("reference_time_raw")),
            "isaac_read_times": deepcopy(decoded.get("read_times_raw")),
            "camera_params": _json_safe(
                decoded.get("camera_params_raw"), "CameraParams"
            ),
            "camera": _json_safe(decoded.get("camera_path"), "camera"),
            "resolution": _json_safe(
                decoded.get("render_product_resolution"), "resolution"
            ),
        },
    }
    try:
        R.write_new(json_path, body)
    except BaseException:
        npz_path.unlink(missing_ok=True)
        raise
    return json_path.resolve(), npz_path.resolve()


def load_capture_journal(
    job_path: Path, ordinal: int
) -> tuple[dict[str, Any], dict[str, Any], tuple[bytes, bytes]]:
    """journal·bundle·RGB/mask 원문을 재검증해 manifest에 제공한다."""
    json_path, expected_npz = _paths(job_path, ordinal)
    if (
        json_path.is_symlink()
        or not json_path.is_file()
        or json_path.stat().st_nlink != 1
    ):
        raise ValueError("atomic journal JSON은 단일 regular file이어야 한다")
    body = R.read_object(json_path)
    if (
        set(body) != {"schema", "bundle", "raw", "writer_payload"}
        or body.get("schema") != SCHEMA
    ):
        raise ValueError("atomic journal schema가 다르다")
    bundle = body.get("bundle")
    writer_payload = body.get("writer_payload")
    if not isinstance(bundle, Mapping) or not isinstance(writer_payload, Mapping):
        raise TypeError("atomic journal bundle/writer payload가 다르다")
    if writer_payload.get("ordinal") != ordinal:
        raise ValueError("atomic journal ordinal이 다르다")
    leaf = body.get("raw")
    if not isinstance(leaf, Mapping) or set(leaf) != {"file", "sha256"}:
        raise ValueError("atomic journal npz leaf가 다르다")
    if (
        leaf["file"] != expected_npz.name
        or expected_npz.is_symlink()
        or not expected_npz.is_file()
    ):
        raise ValueError("atomic journal npz 경로가 다르다")
    if expected_npz.stat().st_nlink != 1:
        raise ValueError("atomic journal npz hardlink를 거부한다")
    raw = expected_npz.read_bytes()
    if leaf["sha256"] != _sha(raw):
        raise ValueError("atomic journal npz SHA가 다르다")
    with np.load(expected_npz, allow_pickle=False) as archive:
        if set(archive.files) != {"rgb", "semantic_mask"}:
            raise ValueError("atomic journal npz entry가 다르다")
        arrays = {name: archive[name].copy(order="C") for name in archive.files}
    for field, array in (
        ("rgb", arrays["rgb"]),
        ("semantic_mask", arrays["semantic_mask"]),
    ):
        descriptor = bundle[field]
        data = array.tobytes(order="C")
        if (
            not isinstance(descriptor, Mapping)
            or descriptor.get("sha256") != _sha(data)
            or descriptor.get("bytes") != len(data)
            or descriptor.get("shape") != list(array.shape)
            or descriptor.get("dtype") != str(array.dtype)
        ):
            raise ValueError(f"atomic journal {field} 원문 결합이 다르다")
    rgb_bytes = arrays["rgb"].tobytes(order="C")
    mask_bytes = arrays["semantic_mask"].tobytes(order="C")
    _, _, _, decoded = FRAME._semantic(
        {
            "mask": arrays["semantic_mask"],
            "idToLabels": bundle.get("id_to_labels"),
        },
        arrays["rgb"],
        bundle.get("expected_code"),
        str(bundle.get("mode")),
    )
    if decoded != bundle.get("decoded_code"):
        raise ValueError("atomic journal semantic 원문과 bundle decoded code가 다르다")
    mode = bundle.get("mode")
    if mode not in {FRAME.SMOKE_MODE, FRAME.RECORD_MODE}:
        raise ValueError("atomic journal bundle mode가 다르다")
    verify_writer_bundle(body, bundle, str(mode), ordinal)
    FRAME.verify_bundle(bundle, (rgb_bytes, mask_bytes), ordinal)
    return body, dict(bundle), (rgb_bytes, mask_bytes)


def verify_atomic_journal(job_path: Path, ordinal: int) -> dict[str, Any]:
    """호환용 단일 반환 verifier."""
    journal, _, _ = load_capture_journal(job_path, ordinal)
    return journal
