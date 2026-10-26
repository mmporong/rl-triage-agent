"""유효한 Writer callback 원문을 본문 노출 없이 타입 보존 해시로 요약한다."""

from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Mapping
from typing import Any

import numpy as np
import vision_writer_payload_v1 as PAYLOAD

SCHEMA = "vision_callback_fingerprint_v1"
MAX_CANONICAL_BYTES = 64 * 1024 * 1024
MAX_CANONICAL_NODES = 100_000
MAX_CANONICAL_DEPTH = 64
MAX_MAPPING_KEY_BYTES = 4096


def _type_name(value: Any) -> str:
    kind = type(value)
    return f"{kind.__module__}.{kind.__qualname__}"


class _Encoder:
    def __init__(self) -> None:
        self.output = bytearray()
        self.nodes = 0
        self.active: set[int] = set()

    def _ensure(self, count: int) -> None:
        if count < 0 or len(self.output) + count > MAX_CANONICAL_BYTES:
            raise ValueError("callback fingerprint canonical byte 상한을 넘었다")

    def _remaining(self) -> int:
        return MAX_CANONICAL_BYTES - len(self.output)

    def _append(self, value: bytes | bytearray | memoryview) -> None:
        self._ensure(len(value))
        self.output.extend(value)

    @staticmethod
    def _field_size(tag: bytes, length: int) -> int:
        return len(tag) + len(str(length)) + 1 + length

    def _field(self, tag: bytes, value: bytes | bytearray | memoryview) -> None:
        length = len(value)
        self._ensure(self._field_size(tag, length))
        self.output.extend(tag)
        self.output.extend(str(length).encode("ascii"))
        self.output.extend(b":")
        self.output.extend(value)

    def _produced_field(self, tag: bytes, length: int, producer: Any) -> None:
        self._ensure(self._field_size(tag, length))
        value = producer()
        if len(value) != length:
            raise ValueError("callback fingerprint buffer 길이가 metadata와 다르다")
        self._field(tag, value)

    def _node(self, depth: int) -> None:
        self.nodes += 1
        if self.nodes > MAX_CANONICAL_NODES:
            raise ValueError("callback fingerprint canonical node 상한을 넘었다")
        if depth > MAX_CANONICAL_DEPTH:
            raise ValueError("callback fingerprint canonical depth 상한을 넘었다")

    def _key(self, value: Any, depth: int, available: int) -> bytes:
        self._node(depth)
        limit = min(MAX_MAPPING_KEY_BYTES, available)
        if limit < 0:
            raise ValueError("callback fingerprint canonical byte 상한을 넘었다")
        if value is None:
            raw = b"N;"
        elif type(value) is bool:
            raw = b"B1;" if value else b"B0;"
        elif type(value) is int:
            estimated = (value.bit_length() * 30103) // 100000 + 3
            if estimated > limit:
                raise ValueError("callback fingerprint mapping key byte 상한을 넘었다")
            body = str(value).encode("ascii")
            raw = b"I" + str(len(body)).encode("ascii") + b":" + body
        elif type(value) is float:
            if not math.isfinite(value):
                raise ValueError("callback fingerprint mapping key가 nonfinite다")
            raw = b"F8:" + struct.pack(">d", value)
        elif type(value) is str:
            encoded_upper_bound = len(value) if value.isascii() else len(value) * 4
            if encoded_upper_bound > limit:
                raise ValueError("callback fingerprint mapping key byte 상한을 넘었다")
            body = value.encode("utf-8")
            raw = b"S" + str(len(body)).encode("ascii") + b":" + body
        elif type(value) is bytes:
            if 1 + len(str(len(value))) + 1 + len(value) > limit:
                raise ValueError("callback fingerprint mapping key byte 상한을 넘었다")
            raw = b"Y" + str(len(value)).encode("ascii") + b":" + value
        elif isinstance(value, np.generic):
            dtype = value.dtype
            if dtype.hasobject or dtype.fields is not None:
                raise TypeError(
                    "callback fingerprint numpy mapping key를 지원하지 않는다"
                )
            if dtype.kind not in "biufcSU":
                raise TypeError(
                    "callback fingerprint numpy mapping key kind를 지원하지 않는다"
                )
            dtype_raw = dtype.str.encode("ascii")
            nbytes = int(value.nbytes)
            raw_size = 1 + len(dtype_raw) + 1 + len(str(nbytes)) + 1 + nbytes
            if raw_size > limit:
                raise ValueError("callback fingerprint mapping key byte 상한을 넘었다")
            if dtype.kind in "fc" and not bool(np.isfinite(value)):
                raise ValueError("callback fingerprint numpy mapping key가 nonfinite다")
            body = value.tobytes()
            if len(body) != nbytes:
                raise ValueError("callback fingerprint buffer 길이가 metadata와 다르다")
            raw = b"G" + dtype_raw + b":" + str(len(body)).encode("ascii") + b":" + body
        else:
            raise TypeError(
                "callback fingerprint mapping key type을 지원하지 않는다: "
                + _type_name(value)
            )
        if len(raw) > limit:
            raise ValueError("callback fingerprint mapping key byte 상한을 넘었다")
        return raw

    def _enter(self, value: Any) -> int:
        identity = id(value)
        if identity in self.active:
            raise ValueError("callback fingerprint payload에 cycle이 있다")
        self.active.add(identity)
        return identity

    def _value(self, value: Any, depth: int) -> None:
        self._node(depth)
        if value is None:
            self._append(b"N;")
            return
        if type(value) is bool:
            self._append(b"B1;" if value else b"B0;")
            return
        if type(value) is int:
            estimated = (value.bit_length() * 30103) // 100000 + 3
            if estimated > self._remaining():
                raise ValueError("callback fingerprint canonical byte 상한을 넘었다")
            self._field(b"I", str(value).encode("ascii"))
            return
        if type(value) is float:
            if not math.isfinite(value):
                raise ValueError(
                    "callback fingerprint payload에 nonfinite float가 있다"
                )
            self._field(b"F", struct.pack(">d", value))
            return
        if type(value) is str:
            encoded_upper_bound = len(value) if value.isascii() else len(value) * 4
            if encoded_upper_bound > self._remaining():
                raise ValueError("callback fingerprint canonical byte 상한을 넘었다")
            self._field(b"S", value.encode("utf-8"))
            return
        if type(value) is bytes:
            self._field(b"Y", value)
            return
        if type(value) is bytearray:
            self._field(b"A", value)
            return
        if isinstance(value, memoryview):
            self._ensure(self._field_size(b"Vb", value.nbytes))
            self._field(b"Vt", _type_name(value).encode())
            self._field(b"Vf", value.format.encode())
            self._field(b"Vs", repr(value.shape).encode("ascii"))
            self._produced_field(b"Vb", value.nbytes, value.tobytes)
            return
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject or value.dtype.fields is not None:
                raise TypeError("callback fingerprint ndarray dtype을 지원하지 않는다")
            if value.dtype.kind not in "biufcSU":
                raise TypeError("callback fingerprint ndarray kind를 지원하지 않는다")
            type_raw = _type_name(value).encode()
            dtype_raw = value.dtype.str.encode("ascii")
            shape_raw = repr(tuple(int(item) for item in value.shape)).encode()
            required = sum(
                self._field_size(tag, length)
                for tag, length in (
                    (b"Dt", len(type_raw)),
                    (b"Dd", len(dtype_raw)),
                    (b"Ds", len(shape_raw)),
                    (b"Db", int(value.nbytes)),
                )
            )
            self._ensure(required)
            if value.dtype.kind in "fc" and not np.isfinite(value).all():
                raise ValueError("callback fingerprint ndarray에 nonfinite 값이 있다")
            self._field(b"Dt", type_raw)
            self._field(b"Dd", dtype_raw)
            self._field(b"Ds", shape_raw)
            self._produced_field(
                b"Db", int(value.nbytes), lambda: value.tobytes(order="C")
            )
            return
        if isinstance(value, np.generic):
            dtype = value.dtype
            if dtype.hasobject or dtype.fields is not None:
                raise TypeError(
                    "callback fingerprint numpy scalar dtype을 지원하지 않는다"
                )
            if dtype.kind not in "biufcSU":
                raise TypeError(
                    "callback fingerprint numpy scalar kind를 지원하지 않는다"
                )
            type_raw = _type_name(value).encode()
            dtype_raw = dtype.str.encode("ascii")
            nbytes = int(value.nbytes)
            required = sum(
                self._field_size(tag, length)
                for tag, length in (
                    (b"Gt", len(type_raw)),
                    (b"Gd", len(dtype_raw)),
                    (b"Gb", nbytes),
                )
            )
            self._ensure(required)
            if dtype.kind in "fc" and not bool(np.isfinite(value)):
                raise ValueError("callback fingerprint numpy scalar가 nonfinite다")
            self._field(b"Gt", type_raw)
            self._field(b"Gd", dtype_raw)
            self._produced_field(b"Gb", nbytes, value.tobytes)
            return
        if isinstance(value, Mapping):
            identity = self._enter(value)
            try:
                count = len(value)
                if self.nodes + 2 * count > MAX_CANONICAL_NODES:
                    raise ValueError(
                        "callback fingerprint canonical node 상한을 넘었다"
                    )
                if count > MAX_CANONICAL_BYTES // 4:
                    raise ValueError(
                        "callback fingerprint canonical byte 상한을 넘었다"
                    )
                rows: list[tuple[bytes, Any]] = []
                pending = 0
                for key, item in value.items():
                    key_raw = self._key(key, depth + 1, self._remaining() - pending)
                    pending += self._field_size(b"Mk", len(key_raw))
                    self._ensure(pending)
                    rows.append((key_raw, item))
                rows.sort(key=lambda row: row[0])
                type_raw = _type_name(value).encode()
                count_raw = str(count).encode("ascii")
                self._ensure(
                    self._field_size(b"Mt", len(type_raw))
                    + self._field_size(b"Mc", len(count_raw))
                    + 5
                    + pending
                )
                self._field(b"Mt", type_raw)
                self._field(b"Mc", count_raw)
                self._append(b"M{")
                for key, item in rows:
                    self._field(b"Mk", key)
                    self._value(item, depth + 1)
                self._append(b"M};")
            finally:
                self.active.remove(identity)
            return
        if isinstance(value, (list, tuple)):
            identity = self._enter(value)
            try:
                if self.nodes + len(value) > MAX_CANONICAL_NODES:
                    raise ValueError(
                        "callback fingerprint canonical node 상한을 넘었다"
                    )
                self._field(b"Qt", _type_name(value).encode())
                self._field(b"Qc", str(len(value)).encode("ascii"))
                self._append(b"Q{")
                for item in value:
                    self._value(item, depth + 1)
                self._append(b"Q};")
            finally:
                self.active.remove(identity)
            return
        raise TypeError(
            f"callback fingerprint type을 지원하지 않는다: {_type_name(value)}"
        )

    def encode(self, value: Any) -> memoryview:
        self._value(value, 0)
        return memoryview(self.output)


def _canonical(value: Any) -> memoryview:
    return _Encoder().encode(value)


def _metadata(value: Any) -> dict[str, Any]:
    result: dict[str, Any] = {"type": _type_name(value)}
    if isinstance(value, np.ndarray):
        result.update(
            {
                "dtype": value.dtype.str,
                "shape": [int(item) for item in value.shape],
                "nbytes": int(value.nbytes),
            }
        )
    elif isinstance(value, np.generic):
        result["dtype"] = np.asarray(value).dtype.str
    elif isinstance(value, Mapping):
        result["length"] = len(value)
        result["key_types"] = sorted({_type_name(key) for key in value})
    elif isinstance(value, (list, tuple, bytes, bytearray, memoryview, str)):
        result["length"] = len(value)
    elif type(value) is int:
        result["bit_length"] = value.bit_length()
    return result


def _component(value: Any) -> dict[str, Any]:
    raw = _canonical(value)
    return {"sha256": hashlib.sha256(raw).hexdigest(), "metadata": _metadata(value)}


def fingerprint(payload: Any) -> dict[str, Any]:
    """유효한 callback payload 전체를 타입 보존 SHA와 비식별 metadata로 반환한다."""
    if not isinstance(payload, Mapping):
        raise TypeError("callback fingerprint payload가 mapping이 아니다")
    whole = _canonical(payload)
    annotators = payload.get("annotators")
    if not isinstance(annotators, Mapping):
        raise TypeError("callback fingerprint annotators가 mapping이 아니다")
    raw_rows: dict[str, tuple[Any, Any]] = {}
    for name in PAYLOAD.ANNOTATOR_KEYS:
        value = annotators.get(name)
        if not isinstance(value, Mapping) or len(value) != 1:
            raise ValueError("callback fingerprint annotator RP가 단일값이 아니다")
        raw_rows[name] = next(iter(value.items()))
    render_products = {row[0] for row in raw_rows.values()}
    if len(render_products) != 1:
        raise ValueError("callback fingerprint RenderProduct identity가 다르다")
    render_product = next(iter(render_products))
    rgb = raw_rows["rgb"][1]
    semantic = raw_rows["semantic_segmentation"][1]
    if not isinstance(rgb, Mapping) or not isinstance(semantic, Mapping):
        raise TypeError(
            "callback fingerprint RGB 또는 semantic 원문이 mapping이 아니다"
        )
    snapshots = {
        "distribution_outputs": _component(payload.get("distribution_outputs")),
        "trigger_outputs": _component(payload.get("trigger_outputs")),
        "named_outputs": _component(payload.get("named_outputs")),
        "reference_time": _component(payload.get("reference_time")),
        "writer_swh_frame_number": _component(payload.get("swhFrameNumber")),
        "render_product_identity": _component(render_product),
        "camera_identity": _component(raw_rows["camera"][1]),
        "resolution": _component(raw_rows["resolution"][1]),
        "rgb_data": _component(rgb.get("data")),
        "semantic_segmentation": _component(semantic),
        "semantic_mask": _component(semantic.get("data")),
        "semantic_labels": _component(semantic.get("idToLabels")),
        "camera_params": _component(raw_rows["CameraParams"][1]),
        "isaac_read_times": _component(raw_rows["IsaacReadTimes"][1]),
    }
    PAYLOAD.decode_writer_payload(payload)
    return {
        "schema": SCHEMA,
        "whole_payload_sha256": hashlib.sha256(whole).hexdigest(),
        "components": snapshots,
        "canonical_limits": {
            "max_bytes": MAX_CANONICAL_BYTES,
            "max_nodes": MAX_CANONICAL_NODES,
            "max_depth": MAX_CANONICAL_DEPTH,
        },
        "no_capture_approved": True,
    }
