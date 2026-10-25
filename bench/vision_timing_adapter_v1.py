"""Isaac render timing의 허용 shape를 정규화하고 원문 scalar를 별도 보존한다."""

from __future__ import annotations

import hashlib
import math
import os
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from typing import Any

import record_vision_evidence_v1 as R

SCHEMA = "vision_timing_raw_capture_v1"
EXECUTION_PORT = "execOut"
REFERENCE_FIELDS = frozenset({"referenceTimeNumerator", "referenceTimeDenominator"})


def _shape(read_times: Mapping[str, Any], reference_time: Mapping[str, Any]) -> bool:
    if not isinstance(read_times, Mapping) or not isinstance(reference_time, Mapping):
        raise TypeError("render timing 원문은 mapping이어야 한다")
    read_keys = set(read_times)
    base = set(R.ISAAC_READ_TIMES_FIELDS)
    if read_keys not in (base, base | {EXECUTION_PORT}):
        raise ValueError(
            "IsaacReadTimes는 등록 11필드 또는 execOut 포함 12필드여야 한다"
        )
    if set(reference_time) != REFERENCE_FIELDS:
        raise ValueError("ReferenceTime 필드가 계약과 다르다")
    return EXECUTION_PORT in read_keys


def normalize_render_timing(
    read_times: Mapping[str, Any],
    reference_time: Mapping[str, Any],
    simulation_time_s: float,
) -> dict[str, Any]:
    """execOut을 시각에서 제외하고 기존 11필드 검증 계약을 그대로 적용한다."""
    _shape(read_times, reference_time)
    timing = {
        "read_times": {field: read_times[field] for field in R.ISAAC_READ_TIMES_FIELDS},
        "reference_time": dict(reference_time),
    }
    return R.validate_render_timing(timing, simulation_time_s)


def _type_name(value: Any) -> str:
    kind = type(value)
    return f"{kind.__module__}.{kind.__qualname__}"


def _native_scalar(value: Any, anchor: str) -> tuple[Any, str]:
    if isinstance(value, Enum):
        native, encoding = _native_scalar(value.value, f"{anchor}.value")
        return native, f"enum_value:{encoding}"
    if value is None or type(value) in (bool, int, str):
        return value, "json_native"
    if type(value) is float:
        if not math.isfinite(value):
            marker = (
                "nan"
                if math.isnan(value)
                else "positive_infinity"
                if value > 0
                else "negative_infinity"
            )
            return {"nonfinite": marker}, "json_nonfinite_marker"
        return value, "json_native"
    if type(value).__module__.split(".", 1)[0] == "numpy":
        try:
            item = getattr(value, "item", None)
            if getattr(value, "ndim", None) != 0 or getattr(value, "shape", None) != ():
                return None, "unsupported"
            if not callable(item):
                return None, "unsupported"
            scalar = item()
        except Exception:  # noqa: BLE001 - 관측 원문 기록은 변환 실패로 중단하지 않는다.
            return None, "unsupported"
        native, encoding = _native_scalar(scalar, f"{anchor}.item")
        return native, f"numpy_item:{encoding}"
    return None, "unsupported"


def _typed_scalar(value: Any, anchor: str) -> dict[str, Any]:
    native, encoding = _native_scalar(value, anchor)
    return {"type": _type_name(value), "encoding": encoding, "value": native}


def _typed_mapping(value: Mapping[Any, Any], anchor: str) -> dict[str, Any]:
    fields = {
        field: _typed_scalar(item, f"{anchor}.{field}")
        for field, item in value.items()
        if isinstance(field, str)
    }
    return {
        "type": _type_name(value),
        "fields": {field: fields[field] for field in sorted(fields)},
        "non_string_key_types": sorted(
            _type_name(field) for field in value if not isinstance(field, str)
        ),
    }


def write_capture_journal(
    job_path: Path,
    capture_ordinal: int,
    read_times: Mapping[str, Any],
    reference_time: Mapping[str, Any],
    rgb_sha256: str,
    *,
    rgb_bytes: bytes | None = None,
) -> Path:
    """캡처 timing 원문 metadata와 RGB SHA만 job 옆에 write-once로 기록한다."""
    if not isinstance(read_times, Mapping) or not isinstance(reference_time, Mapping):
        raise TypeError("render timing 원문은 mapping이어야 한다")
    has_exec_out = EXECUTION_PORT in read_times
    if (
        isinstance(capture_ordinal, bool)
        or not isinstance(capture_ordinal, int)
        or capture_ordinal < 0
    ):
        raise ValueError("capture ordinal은 0 이상 정수여야 한다")
    if not R._SHA256.fullmatch(rgb_sha256):
        raise ValueError("RGB SHA-256이 잘못됐다")
    job = Path(job_path)
    if job.is_symlink() or not job.is_file():
        raise ValueError("job은 symlink가 아닌 regular file이어야 한다")
    execution_port = {
        "field": EXECUTION_PORT,
        "present": has_exec_out,
        "used_as_timing": False,
        "raw": (
            _typed_scalar(read_times[EXECUTION_PORT], "IsaacReadTimes.execOut")
            if has_exec_out
            else None
        ),
    }
    destination = job.with_name(
        f"{job.stem}.capture-{capture_ordinal:06d}.timing-raw.json"
    )
    rgb_path = job.with_name(f"{job.stem}.capture-{capture_ordinal:06d}.rgb")
    if destination.exists() or (rgb_bytes is not None and rgb_path.exists()):
        raise FileExistsError("capture journal 또는 RGB 원문이 이미 있다")
    rgb_artifact = None
    if rgb_bytes is not None:
        if not isinstance(rgb_bytes, bytes):
            raise TypeError("RGB 원문은 bytes여야 한다")
        if hashlib.sha256(rgb_bytes).hexdigest() != rgb_sha256:
            raise ValueError("RGB 원문 SHA가 등록값과 다르다")
        with rgb_path.open("xb") as stream:
            stream.write(rgb_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        rgb_artifact = {
            "file": rgb_path.name,
            "sha256": rgb_sha256,
            "bytes": len(rgb_bytes),
        }
    body = {
        "schema": SCHEMA,
        "capture_ordinal": capture_ordinal,
        "observed_read_times_key_count": len(read_times),
        "read_times": _typed_mapping(read_times, "IsaacReadTimes"),
        "reference_time": _typed_mapping(reference_time, "ReferenceTime"),
        "execution_port": execution_port,
        "rgb_sha256": rgb_sha256,
        "rgb_artifact": rgb_artifact,
    }
    R.write_new(destination, body)
    return destination.resolve()
