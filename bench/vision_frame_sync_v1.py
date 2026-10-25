"""한 Writer callback의 RGB·semantic·camera·timing을 원자적으로 결합하는 CPU 계약."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from itertools import pairwise
from typing import Any

import numpy as np
import record_vision_evidence_v1 as R
import vision_timing_adapter_v1 as TIMING

SCHEMA = "vision_frame_sync_bundle_v1"
SMOKE_MODE = "render_sync_smoke"
RECORD_MODE = "record"
MARKER_PREFIX = "rl_triage_frame_sync/"
SENTINEL = MARKER_PREFIX + "sentinel"
PALETTE = {
    "sentinel": (17, 251, 197),
    "active": (241, 53, 89),
    "inactive": (37, 61, 83),
}
_SOURCE_KEYS = frozenset({"rgb", "semantic", "camera", "read_times", "reference_time"})
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")


def marker_labels(code: int) -> set[str]:
    if type(code) is not int or code not in range(4):
        raise ValueError("marker code는 0..3 정수여야 한다")
    labels = {SENTINEL}
    for bit in range(2):
        active_rail = (code >> bit) & 1
        for rail in range(2):
            role = "active" if rail == active_rail else "inactive"
            labels.add(f"{MARKER_PREFIX}bit{bit}/rail{rail}/{role}")
    return labels


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _snapshot(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.copy(order="K")
    if isinstance(value, Mapping):
        return {key: _snapshot(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_snapshot(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_snapshot(item) for item in value)
    return deepcopy(value)


def _finite_matrix(value: Any, label: str) -> list[list[float]]:
    array = np.asarray(value)
    if array.shape != (4, 4) or array.dtype.kind not in "fiu":
        raise ValueError(f"{label}은 finite 4x4 수치 행렬이어야 한다")
    converted = array.astype(np.float64, copy=False)
    if not np.isfinite(converted).all():
        raise ValueError(f"{label}은 finite 4x4 수치 행렬이어야 한다")
    return converted.tolist()


def _source(payload: Mapping[str, Any], name: str, callback_ref: str) -> Any:
    value = payload.get(name)
    if (
        not isinstance(value, Mapping)
        or set(value) != {"adapter_callback_ref", "data"}
        or value.get("adapter_callback_ref") != callback_ref
    ):
        raise ValueError("모든 annotator는 같은 adapter callback reference여야 한다")
    return value["data"]


def _array_bytes(
    value: Any, label: str, *, channels: int | None = None
) -> tuple[np.ndarray, bytes]:
    if not isinstance(value, np.ndarray) or value.dtype != np.uint8:
        raise ValueError(f"{label}은 uint8 ndarray여야 한다")
    if channels is not None and (value.ndim != 3 or value.shape[2] != channels):
        raise ValueError(f"{label} shape가 다르다")
    if channels is None and value.ndim != 2:
        raise ValueError(f"{label} shape가 다르다")
    copied = value.copy(order="C")
    return copied, copied.tobytes(order="C")


def _label(value: Any) -> str:
    if isinstance(value, str):
        return value
    if (
        isinstance(value, Mapping)
        and set(value) == {"class"}
        and isinstance(value.get("class"), str)
    ):
        return value["class"]
    raise ValueError("idToLabels 값은 class label이어야 한다")


def _semantic(
    value: Any, rgb: np.ndarray, expected_code: int | None, mode: str
) -> tuple[np.ndarray, bytes, dict[str, str], int | None]:
    if not isinstance(value, Mapping) or set(value) != {"mask", "idToLabels"}:
        raise ValueError("semantic payload 형식이 다르다")
    mask_raw = value["mask"]
    if (
        not isinstance(mask_raw, np.ndarray)
        or mask_raw.ndim != 2
        or mask_raw.dtype.kind not in "ui"
    ):
        raise ValueError("semantic mask는 2D unsigned/integer ndarray여야 한다")
    mask = mask_raw.copy(order="C")
    if mask.shape != rgb.shape[:2]:
        raise ValueError("semantic mask와 RGB resolution이 다르다")
    labels_raw = value["idToLabels"]
    if not isinstance(labels_raw, Mapping):
        raise TypeError("idToLabels는 mapping이어야 한다")
    labels = {str(runtime_id): _label(item) for runtime_id, item in labels_raw.items()}
    marker_ids = {
        int(runtime_id): label
        for runtime_id, label in labels.items()
        if label.startswith(MARKER_PREFIX)
    }
    if mode == RECORD_MODE:
        if marker_ids:
            raise ValueError("record capture에는 reserved marker가 없어야 한다")
        return mask, mask.tobytes(order="C"), labels, None
    visible_marker_ids = {
        runtime_id: label
        for runtime_id, label in marker_ids.items()
        if (mask == runtime_id).any()
    }
    if (
        expected_code is None
        or len(visible_marker_ids) != 5
        or set(visible_marker_ids.values()) != marker_labels(expected_code)
    ):
        raise ValueError("smoke reserved label exact set이 expected code와 다르다")
    decoded = 0
    for runtime_id, label in visible_marker_ids.items():
        pixels = mask == runtime_id
        samples = rgb[pixels]
        median = tuple(int(item) for item in np.median(samples, axis=0))
        if label == SENTINEL:
            role = "sentinel"
        else:
            parts = label.removeprefix(MARKER_PREFIX).split("/")
            if (
                len(parts) != 3
                or parts[0] not in {"bit0", "bit1"}
                or parts[1] not in {"rail0", "rail1"}
                or parts[2] not in {"active", "inactive"}
            ):
                raise ValueError("reserved marker label 문법이 다르다")
            role = parts[2]
            if parts[2] == "active":
                bit, rail = int(parts[0][-1]), int(parts[1][-1])
                decoded |= rail << bit
        red, green, blue = median
        ordered = (
            (role == "sentinel" and green > blue > red)
            or (role == "active" and red > blue > green)
            or (role == "inactive" and blue > green > red)
        )
        if not ordered:
            raise ValueError(
                "semantic current와 RGB marker channel order가 같은 callback이 아니다"
            )
    if decoded != expected_code:
        raise ValueError("semantic marker decoded code가 expected code와 다르다")
    return mask, mask.tobytes(order="C"), labels, decoded


def _timing_fraction(read_times: Mapping[str, Any], reference: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "rational_numerator": read_times.get("rationalTimeOfSimNumerator"),
        "rational_denominator": read_times.get("rationalTimeOfSimDenominator"),
        "reference_numerator": reference.get("referenceTimeNumerator"),
        "reference_denominator": reference.get("referenceTimeDenominator"),
    }


def _verify_fraction(value: Any, timing: Mapping[str, Any]) -> dict[str, int]:
    keys = {"rational_numerator", "rational_denominator",
            "reference_numerator", "reference_denominator"}
    if (not isinstance(value, dict) or set(value) != keys
            or any(type(item) is not int for item in value.values())):
        raise ValueError("timing fraction은 네 exact 정수여야 한다")
    if (value["rational_denominator"] <= 0 or value["reference_denominator"] <= 0
            or value["rational_numerator"] * value["reference_denominator"]
            != value["reference_numerator"] * value["rational_denominator"]
            or value != _timing_fraction(timing["read_times"], timing["reference_time"])):
        raise ValueError("IsaacReadTimes와 ReferenceTime 분수 결합이 다르다")
    return value


class AtomicFrameWriter:
    """step 전에 arm하고 첫 Writer callback 하나만 받아 evidence bundle을 만든다."""

    def __init__(
        self,
        mode: str,
        expected_count: int,
        render_product_ref: str = "render-product-0",
    ) -> None:
        if mode not in {SMOKE_MODE, RECORD_MODE}:
            raise ValueError("frame sync mode가 다르다")
        if (
            type(expected_count) is not int
            or expected_count <= 0
            or (mode == SMOKE_MODE and expected_count != 4)
        ):
            raise ValueError("smoke는 4 capture, record는 양의 capture 수가 필요하다")
        if not isinstance(render_product_ref, str) or not render_product_ref:
            raise ValueError("단일 RenderProduct reference가 필요하다")
        self.mode = mode
        self.expected_count = expected_count
        self.render_product_ref = render_product_ref
        self._pending: dict[str, Any] | None = None
        self._bundles: list[dict[str, Any]] = []
        self._raw: list[tuple[bytes, bytes]] = []
        self._callback_refs: set[str] = set()

    def arm(
        self,
        ordinal: int,
        expected_code: int | None,
        state_before: Mapping[str, Any],
        simulation_time_s: float,
    ) -> None:
        if (
            self._pending is not None
            or ordinal != len(self._bundles)
            or ordinal >= self.expected_count
        ):
            raise ValueError(
                "capture ordinal은 pending 없이 0부터 순서대로 arm해야 한다"
            )
        if self.mode == SMOKE_MODE and expected_code != ordinal:
            raise ValueError("smoke ordinal의 expected code는 0,1,2,3 순서여야 한다")
        if self.mode == RECORD_MODE and expected_code is not None:
            raise ValueError("record capture에는 marker code가 없어야 한다")
        if not isinstance(state_before, Mapping) or not math.isfinite(
            float(simulation_time_s)
        ):
            raise TypeError("physics state와 simulation time이 필요하다")
        self._pending = {
            "ordinal": ordinal,
            "expected_code": expected_code,
            "state_before": _snapshot(state_before),
            "simulation_time_s": float(simulation_time_s),
            "stepped": False,
            "callback": None,
        }

    def before_step(self, ordinal: int) -> None:
        if (
            self._pending is None
            or self._pending["ordinal"] != ordinal
            or self._pending["stepped"]
        ):
            raise ValueError("step은 해당 ordinal arm 뒤 한 번만 시작해야 한다")
        self._pending["stepped"] = True

    def callback(self, payload: Mapping[str, Any]) -> None:
        if self._pending is None or not self._pending["stepped"]:
            raise ValueError("arm과 step 시작 없는 unsolicited callback을 거부한다")
        if self._pending["callback"] is not None:
            raise ValueError("한 capture에는 Writer callback이 정확히 하나여야 한다")
        if not isinstance(payload, Mapping) or set(payload) != _SOURCE_KEYS | {
            "adapter_callback_ref",
            "render_product_ref",
        }:
            raise ValueError("Writer callback payload 필드가 다르다")
        if payload.get("render_product_ref") != self.render_product_ref:
            raise ValueError("Writer callback이 등록한 단일 RenderProduct와 다르다")
        callback_ref = payload.get("adapter_callback_ref")
        if (
            not isinstance(callback_ref, str)
            or not callback_ref
            or callback_ref in self._callback_refs
        ):
            raise ValueError("callback reference가 없거나 replay됐다")
        self._pending["callback"] = _snapshot(payload)
        self._callback_refs.add(callback_ref)

    def close(
        self, ordinal: int, state_after: Mapping[str, Any], simulation_time_s: float
    ) -> dict[str, Any]:
        pending = self._pending
        if (
            pending is None
            or pending["ordinal"] != ordinal
            or not pending["stepped"]
            or pending["callback"] is None
        ):
            raise ValueError(
                "capture는 해당 ordinal의 callback 정확히 하나 뒤 닫아야 한다"
            )
        if (
            _canonical(pending["state_before"]) != _canonical(state_after)
            or float(simulation_time_s) != pending["simulation_time_s"]
        ):
            raise ValueError("capture 동안 physics state/time이 바뀌었다")
        payload = pending["callback"]
        callback_ref = payload["adapter_callback_ref"]
        rgb, rgb_bytes = _array_bytes(
            _source(payload, "rgb", callback_ref), "RGB", channels=3
        )
        semantic_data = _source(payload, "semantic", callback_ref)
        mask, mask_bytes, labels, decoded = _semantic(
            semantic_data,
            rgb,
            pending["expected_code"],
            self.mode,
        )
        camera = _source(payload, "camera", callback_ref)
        if not isinstance(camera, Mapping) or set(camera) != {
            "view_matrix",
            "projection",
            "resolution",
        }:
            raise ValueError("CameraParams 형식이 다르다")
        view = _finite_matrix(camera["view_matrix"], "CameraParams.view_matrix")
        projection = _finite_matrix(camera["projection"], "CameraParams.projection")
        resolution = camera["resolution"]
        if (
            not isinstance(resolution, Sequence)
            or isinstance(resolution, (str, bytes))
            or list(resolution) != [rgb.shape[1], rgb.shape[0]]
        ):
            raise ValueError("CameraParams resolution과 RGB가 다르다")
        read_times = _source(payload, "read_times", callback_ref)
        reference_time = _source(payload, "reference_time", callback_ref)
        timing = TIMING.normalize_render_timing(
            read_times,
            reference_time,
            pending["simulation_time_s"],
        )
        fraction = _verify_fraction(_timing_fraction(read_times, reference_time), timing)
        bundle = {
            "schema": SCHEMA,
            "mode": self.mode,
            "ordinal": ordinal,
            "callback_sequence": len(self._bundles),
            "adapter_callback_ref": callback_ref,
            "render_product_ref": self.render_product_ref,
            "expected_code": pending["expected_code"],
            "decoded_code": decoded,
            "rgb": {
                "sha256": _sha(rgb_bytes),
                "bytes": len(rgb_bytes),
                "shape": list(rgb.shape),
                "dtype": str(rgb.dtype),
            },
            "semantic_mask": {
                "sha256": _sha(mask_bytes),
                "bytes": len(mask_bytes),
                "shape": list(mask.shape),
                "dtype": str(mask.dtype),
            },
            "id_to_labels": labels,
            "camera": {
                "view_matrix": view,
                "projection": projection,
                "resolution": list(resolution),
            },
            "timing": timing,
            "timing_fraction": fraction,
            "physics": {
                "state_sha256": _sha(_canonical(pending["state_before"])),
                "simulation_time_before_s": pending["simulation_time_s"],
                "simulation_time_after_s": float(simulation_time_s),
            },
        }
        self._bundles.append(bundle)
        self._raw.append((rgb_bytes, mask_bytes))
        self._pending = None
        return deepcopy(bundle)

    def finish(self) -> list[dict[str, Any]]:
        if self._pending is not None or len(self._bundles) != self.expected_count:
            raise ValueError("capture sequence가 끝나지 않았다")
        verify_sequence(self._bundles, self._raw)
        return deepcopy(self._bundles)


def verify_bundle(
    bundle: Mapping[str, Any], artifacts: tuple[bytes, bytes], ordinal: int
) -> dict[str, Any]:
    """한 callback bundle의 원문·시각·camera·semantic 결합을 검증한다."""
    mode = bundle.get("mode")
    if mode not in {SMOKE_MODE, RECORD_MODE}:
        raise ValueError("bundle mode가 등록 enum이 아니다")
    if type(ordinal) is not int or ordinal < 0:
        raise ValueError("bundle ordinal 검증값이 잘못됐다")
    rgb_bytes, mask_bytes = artifacts
    if not isinstance(rgb_bytes, bytes) or not isinstance(mask_bytes, bytes):
        raise TypeError("bundle RGB/mask 원문은 bytes여야 한다")
    if (
        set(bundle)
        != {
            "schema",
            "mode",
            "ordinal",
            "callback_sequence",
            "adapter_callback_ref",
            "render_product_ref",
            "expected_code",
            "decoded_code",
            "rgb",
            "semantic_mask",
            "id_to_labels",
            "camera",
            "timing",
            "timing_fraction",
            "physics",
        }
        or bundle.get("schema") != SCHEMA
        or bundle.get("mode") != mode
        or bundle.get("ordinal") != ordinal
        or bundle.get("callback_sequence") != ordinal
    ):
        raise ValueError("bundle schema/order가 다르거나 replay됐다")
    callback_ref = bundle.get("adapter_callback_ref")
    if not isinstance(callback_ref, str) or not callback_ref:
        raise ValueError("callback reference가 없다")
    product_ref = bundle.get("render_product_ref")
    if not isinstance(product_ref, str) or not product_ref:
        raise ValueError("bundle RenderProduct reference가 없다")
    for field, data in (("rgb", rgb_bytes), ("semantic_mask", mask_bytes)):
        descriptor = bundle[field]
        if (
            not isinstance(descriptor, Mapping)
            or set(descriptor) != {"sha256", "bytes", "shape", "dtype"}
            or descriptor.get("sha256") != _sha(data)
            or descriptor.get("bytes") != len(data)
        ):
            raise ValueError("bundle artifact SHA/bytes가 원문과 다르다")
    try:
        rgb_shape = tuple(bundle["rgb"]["shape"])
        mask_shape = tuple(bundle["semantic_mask"]["shape"])
        rgb = np.frombuffer(rgb_bytes, dtype=np.dtype(bundle["rgb"]["dtype"])).reshape(rgb_shape)
        mask = np.frombuffer(
            mask_bytes, dtype=np.dtype(bundle["semantic_mask"]["dtype"])
        ).reshape(mask_shape)
    except (TypeError, ValueError) as error:
        raise ValueError("bundle artifact shape/dtype로 원문을 복원할 수 없다") from error
    rgb_dims = tuple(int(value) for value in rgb.shape)
    mask_dims = tuple(int(value) for value in mask.shape)
    if (
        rgb.dtype != np.uint8
        or len(rgb_dims) != 3
        or rgb_dims[2] != 3
        or mask.dtype.kind not in "ui"
        or len(mask_dims) != 2
        or mask_dims != rgb_dims[:2]
    ):
        raise ValueError("bundle artifact shape/dtype가 계약과 다르다")
    _, _, _, decoded = _semantic(
        {"mask": mask, "idToLabels": bundle["id_to_labels"]},
        rgb,
        bundle.get("expected_code"),
        str(mode),
    )
    if decoded != bundle.get("decoded_code"):
        raise ValueError("bundle decoded code가 RGB/mask 원문과 다르다")
    physics = bundle.get("physics")
    if (
        not isinstance(physics, Mapping)
        or set(physics)
        != {"state_sha256", "simulation_time_before_s", "simulation_time_after_s"}
        or _HEX64.fullmatch(str(physics.get("state_sha256"))) is None
        or physics["simulation_time_before_s"] != physics["simulation_time_after_s"]
    ):
        raise ValueError("bundle physics time 결합이 다르다")
    checked_timing = R.validate_render_timing(
        bundle.get("timing"), physics["simulation_time_before_s"]
    )
    if checked_timing != bundle.get("timing"):
        raise ValueError("bundle timing이 canonical normalized 값이 아니다")
    fraction = _verify_fraction(bundle.get("timing_fraction"), checked_timing)
    timing = checked_timing["read_times"]
    camera = bundle.get("camera")
    if not isinstance(camera, Mapping) or set(camera) != {
        "view_matrix",
        "projection",
        "resolution",
    }:
        raise ValueError("bundle CameraParams 형식이 다르다")
    view = _finite_matrix(camera["view_matrix"], "bundle view_matrix")
    projection = _finite_matrix(camera["projection"], "bundle projection")
    resolution = camera["resolution"]
    if resolution != [rgb_dims[1], rgb_dims[0]]:
        raise ValueError("bundle CameraParams resolution과 RGB 원문이 다르다")
    _canonical(dict(bundle))
    if mode == SMOKE_MODE and (
        bundle.get("expected_code") != ordinal or bundle.get("decoded_code") != ordinal
    ):
        raise ValueError("smoke expected/decoded code가 ordinal과 다르다")
    if mode == RECORD_MODE and (
        bundle.get("expected_code") is not None
        or bundle.get("decoded_code") is not None
        or any(label.startswith(MARKER_PREFIX) for label in bundle["id_to_labels"].values())
    ):
        raise ValueError("record bundle에 reserved marker가 있다")
    return {
        "mode": mode,
        "callback_ref": callback_ref,
        "render_product_ref": product_ref,
        "view": view,
        "projection": projection,
        "resolution": resolution,
        "frame": timing["frameNumber"],
        "swh_frame": timing["swhFrameNumber"],
        "physics": (physics["state_sha256"], physics["simulation_time_before_s"]),
        "fraction": fraction,
    }


def verify_sequence(
    bundles: Sequence[Mapping[str, Any]], raw: Sequence[tuple[bytes, bytes]]
) -> None:
    if len(bundles) != len(raw) or not bundles:
        raise ValueError("bundle과 RGB/mask 원문의 수가 다르다")
    mode = bundles[0].get("mode")
    if mode not in {SMOKE_MODE, RECORD_MODE}:
        raise ValueError("bundle mode가 등록 enum이 아니다")
    if mode == SMOKE_MODE and len(bundles) != 4:
        raise ValueError("smoke bundle은 네 개여야 한다")
    views, projections, resolutions = [], [], []
    frames, swh_frames, callback_refs, physics_rows = [], [], set(), []
    fractions = []
    product_ref: str | None = None
    for ordinal, (bundle, artifacts) in enumerate(zip(bundles, raw, strict=True)):
        checked = verify_bundle(bundle, artifacts, ordinal)
        if checked["mode"] != mode:
            raise ValueError("bundle sequence mode가 바뀌었다")
        if checked["callback_ref"] in callback_refs:
            raise ValueError("callback reference가 중복됐다")
        callback_refs.add(checked["callback_ref"])
        if product_ref is None:
            product_ref = checked["render_product_ref"]
        elif checked["render_product_ref"] != product_ref:
            raise ValueError("bundle이 서로 다른 RenderProduct를 사용했다")
        views.append(checked["view"])
        projections.append(checked["projection"])
        resolutions.append(checked["resolution"])
        frames.append(checked["frame"])
        swh_frames.append(checked["swh_frame"])
        physics_rows.append(checked["physics"])
        fractions.append(checked["fraction"])
    frame_steps = [right - left for left, right in pairwise(frames)]
    for previous, current in pairwise(fractions):
        for prefix in ("rational", "reference"):
            numerator, denominator = f"{prefix}_numerator", f"{prefix}_denominator"
            if mode == SMOKE_MODE:
                if (current[denominator] != previous[denominator]
                        or current[numerator] != previous[numerator] + 1):
                    raise ValueError("smoke timing fraction이 한 frame씩 진행하지 않았다")
            elif current[numerator] * previous[denominator] <= previous[numerator] * current[denominator]:
                raise ValueError("record timing fraction이 진행하지 않았다")
    if (
        (mode == SMOKE_MODE and any(step != 1 for step in frame_steps))
        or (mode == RECORD_MODE and any(step <= 0 for step in frame_steps))
        or any(right < left for left, right in pairwise(swh_frames))
    ):
        raise ValueError("Writer callback frame 순서가 등록 규칙과 다르다")
    if any(value != projections[0] for value in projections[1:]) or any(
        value != resolutions[0] for value in resolutions[1:]
    ):
        raise ValueError("capture projection/resolution이 바뀌었다")
    if mode == SMOKE_MODE and not (
        views[0] == views[1] == views[3] and views[2] != views[0]
    ):
        raise ValueError("CameraParams는 Awarm=A1=Areturn, Bdifferent여야 한다")
    if mode == SMOKE_MODE and any(row != physics_rows[0] for row in physics_rows[1:]):
        raise ValueError("smoke 네 capture의 physics state/time이 같아야 한다")
