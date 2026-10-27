"""Writer 없이 다섯 annotator를 연속 pull하는 clean 정책 녹화 factory."""

from __future__ import annotations

import hashlib
import math
import os
from collections.abc import Callable, Mapping
from copy import deepcopy
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import record_vision_evidence_v1 as R
import vision_capture_readiness_v1 as READY
import vision_frame_sync_v1 as FRAME
import vision_fxaa_policy_v1 as FXAA
import vision_play_worker_v1 as V1
import vision_timing_adapter_v1 as TIMING

SCHEMA = "vision_pull_record_capture_v1"
RAW_SCHEMA = "vision_pull_record_raw_receipt_v1"
ANNOTATORS = ("rgb", "semantic_segmentation", "CameraParams", "IsaacReadTimes", "ReferenceTime")


def _native_app() -> Any:
    import omni.kit.app  # pyright: ignore[reportMissingImports]

    return omni.kit.app.get_app()


def _counter(native_app: Any) -> tuple[Callable[[], int], dict[str, Any]]:
    method = getattr(native_app, "get_update_number", None)
    if not callable(method):
        raise TypeError("Kit IApp.get_update_number API가 없다")

    def read() -> int:
        value = method()
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RuntimeError("Kit update number가 음이 아닌 정수가 아니다")
        return value

    return read, {
        "source": "omni.kit.app.IApp.get_update_number",
        "method": "app.get_update_number()",
        "initial_value": read(),
        "fail_closed": True,
    }


def _annotators(rep: Any) -> list[tuple[str, Any]]:
    return [
        ("rgb", rep.AnnotatorRegistry.get_annotator("rgb")),
        (
            "semantic_segmentation",
            rep.AnnotatorRegistry.get_annotator(
                "semantic_segmentation", init_params={"colorize": False}
            ),
        ),
        ("CameraParams", rep.AnnotatorRegistry.get_annotator("CameraParams")),
        ("IsaacReadTimes", rep.AnnotatorRegistry.get_annotator("IsaacReadTimes")),
        ("ReferenceTime", rep.annotators.get("ReferenceTime")),
    ]


def _snapshot(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.copy(order="K")
    if isinstance(value, Mapping):
        return {str(key): _snapshot(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_snapshot(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_snapshot(item) for item in value)
    return deepcopy(value)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if value is None or type(value) in (bool, int, float, str):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("raw receipt에는 non-finite 수를 쓸 수 없다")
        return value
    raise TypeError(f"raw receipt가 지원하지 않는 형식이다: {type(value).__name__}")


def _array(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.size == 0 or array.dtype.kind not in "fiu":
        raise ValueError(f"{name}은 비어 있지 않은 수치 배열이어야 한다")
    if not np.isfinite(array.astype(np.float64, copy=False)).all():
        raise ValueError(f"{name}에 non-finite 값이 있다")
    return array.copy(order="C")


def _rgb(value: Any) -> tuple[np.ndarray, bytes]:
    if (
        not isinstance(value, np.ndarray)
        or value.dtype != np.uint8
        or value.ndim != 3
        or value.shape[2] not in (3, 4)
        or value.size == 0
    ):
        raise ValueError("RGB는 uint8 RGB/RGBA ndarray여야 한다")
    rgb = value[:, :, :3].copy(order="C")
    return rgb, rgb.tobytes(order="C")


def _semantic(value: Any, rgb: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    if not isinstance(value, Mapping) or set(value) != {"data", "info"}:
        raise ValueError("semantic payload 형식이 다르다")
    info = value["info"]
    if not isinstance(info, Mapping) or "idToLabels" not in info:
        raise ValueError("semantic idToLabels가 없다")
    mask = value["data"]
    if isinstance(mask, np.ndarray) and mask.shape == (*rgb.shape[:2], 1):
        mask = mask[:, :, 0]
    checked, _, _, decoded = FRAME._semantic(
        {"mask": mask, "idToLabels": info["idToLabels"]},
        rgb,
        None,
        FRAME.RECORD_MODE,
    )
    if decoded is not None:
        raise ValueError("clean record에 reserved marker code가 있다")
    return checked, _snapshot(dict(info))


def _camera(value: Any, rgb: np.ndarray) -> dict[str, Any]:
    required = {"cameraViewTransform", "cameraProjection", "renderProductResolution"}
    if not isinstance(value, Mapping) or not required <= set(value):
        raise ValueError("CameraParams 필수 필드가 없다")
    view = _array(value["cameraViewTransform"], "cameraViewTransform")
    projection = _array(value["cameraProjection"], "cameraProjection")
    if view.size != 16 or projection.size != 16:
        raise ValueError("CameraParams 행렬 크기가 4x4가 아니다")
    resolution = list(value["renderProductResolution"])
    if resolution != [rgb.shape[1], rgb.shape[0]] or resolution != [R.WIDTH, R.HEIGHT]:
        raise ValueError("CameraParams/RGB/fixed resolution이 다르다")
    return {
        "view": view.reshape(4, 4),
        "projection": projection.reshape(4, 4),
        "resolution": np.asarray(resolution, dtype=np.int64),
        "extra": {
            str(key): _snapshot(item)
            for key, item in value.items()
            if key not in required
        },
    }


def _integer(value: Any, name: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name}는 정수여야 한다")
    result = int(value)
    if positive and result <= 0:
        raise ValueError(f"{name}는 양수여야 한다")
    return result


def _timing(
    read_times: Any,
    reference_time: Any,
    simulation_time_s: float,
) -> tuple[dict[str, Any], int, Fraction]:
    if not isinstance(read_times, Mapping) or not isinstance(reference_time, Mapping):
        raise TypeError("IsaacReadTimes/ReferenceTime 원문은 mapping이어야 한다")
    normalized = TIMING.normalize_render_timing(
        read_times, reference_time, simulation_time_s
    )
    FRAME._verify_fraction(
        FRAME._timing_fraction(read_times, reference_time), normalized
    )
    frame = _integer(read_times.get("frameNumber"), "frameNumber")
    numerator = _integer(
        read_times.get("rationalTimeOfSimNumerator"), "rational numerator"
    )
    denominator = _integer(
        read_times.get("rationalTimeOfSimDenominator"),
        "rational denominator",
        positive=True,
    )
    return normalized, frame, Fraction(numerator, denominator)


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)  # pyright: ignore[reportArgumentType]
        stream.flush()
        os.fsync(stream.fileno())


def _descriptor(path: Path) -> dict[str, Any]:
    return {
        "file": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
    }


def _write_raw(
    job_path: Path,
    ordinal: int,
    raw: Mapping[str, Any],
    rgb: np.ndarray,
    semantic: np.ndarray,
    camera: Mapping[str, Any],
) -> tuple[Path, Path]:
    stem = f"{job_path.stem}.pull-record-{ordinal:06d}"
    npz_path = job_path.with_name(stem + ".npz")
    receipt_path = job_path.with_name(stem + ".raw.json")
    read_times = raw["IsaacReadTimes"]
    reference = raw["ReferenceTime"]
    arrays: dict[str, np.ndarray] = {
        "rgb": rgb,
        "semantic_data": semantic,
        "camera_view_transform": camera["view"],
        "camera_projection": camera["projection"],
        "render_product_resolution": camera["resolution"],
    }
    for prefix, value in (("read_times", read_times), ("reference_time", reference)):
        if not isinstance(value, Mapping):
            raise TypeError(f"{prefix} 원문은 mapping이어야 한다")
        for key, item in value.items():
            if type(item) in (bool, int, float) or isinstance(item, np.number):
                arrays[f"{prefix}__{key}"] = np.asarray(item)
    _write_npz(npz_path, arrays)
    R.write_new(
        receipt_path,
        {
            "schema": RAW_SCHEMA,
            "capture_ordinal": ordinal,
            "npz": _descriptor(npz_path),
            "raw_fields": list(ANNOTATORS),
            "semantic_info": _jsonable(raw["semantic_segmentation"]["info"]),
            "camera_extra": _jsonable(camera["extra"]),
            "read_times": _jsonable(read_times),
            "reference_time": _jsonable(reference),
            "reserved_marker_count": 0,
            "atomicity_claimed": False,
            "capture_approved": False,
            "used_for_decision": False,
        },
    )
    return npz_path.resolve(), receipt_path.resolve()


def make_capture_factory(
    validated: Mapping[str, Any],
    app: Any,
    *,
    dependencies: Mapping[str, Any] | None = None,
) -> V1.CameraCaptureFactory:
    """run_real에 넣을 clean Writer-free camera capture factory를 만든다."""
    if validated.get("job", {}).get("mode") != R.RECORD_MODE:
        raise ValueError("pull record capture는 record mode만 허용한다")
    job_path = Path(validated.get("job_path", ""))
    if job_path.is_symlink() or not job_path.is_file():
        raise ValueError("pull record job은 regular file이어야 한다")
    deps = {
        "usd_context": READY._usd_context,
        "settings": READY._carb_settings,
        "readiness": READY._wait_capture_ready,
        "renderer_readback": READY._renderer_readback,
        "physics_snapshot": READY._physics_snapshot,
        "native_app": _native_app,
        "fxaa_required": FXAA.required,
        "fxaa_apply": FXAA.apply,
        "fxaa_readback": FXAA.readback,
        "fxaa_verify": FXAA.verify_readback,
    }
    if dependencies is not None:
        deps.update(dependencies)

    def factory(
        base: Any, rep: Any, render_product: Any, profile: Mapping[str, Any]
    ) -> tuple[V1.CameraCapture, Callable[[], None]]:
        if profile != R.fixed_profile()["camera"]:
            raise ValueError("pull record camera profile이 fixed profile과 다르다")
        context, stage_events = deps["usd_context"]()
        settings = deps["settings"]()
        use_fxaa = bool(deps["fxaa_required"](validated["job"]))
        policy = deps["fxaa_apply"](rep, settings) if use_fxaa else None
        renderer = dict(deps["renderer_readback"](context, profile, settings))
        if policy is not None:
            renderer["render_policy"] = policy
        deps["readiness"](
            base,
            app,
            context,
            stage_events,
            job_path.with_suffix(".pull-record-readiness.json"),
            renderer,
        )
        fxaa_readback = None
        if use_fxaa:
            fxaa_readback = deps["fxaa_readback"](settings)
            deps["fxaa_verify"](fxaa_readback)
            renderer["fxaa_readback"] = _snapshot(fxaa_readback)
        read_counter, counter_source = _counter(deps["native_app"]())
        pairs = _annotators(rep)
        attached: list[tuple[str, Any]] = []
        try:
            for name, annotator in pairs:
                if annotator is None:
                    raise RuntimeError(f"{name} annotator가 없다")
                annotator.attach(render_product)
                attached.append((name, annotator))
        except BaseException:
            for _, annotator in reversed(attached):
                annotator.detach(render_product)
            raise

        expected_ordinal = 0
        previous_frame: int | None = None
        previous_rational: Fraction | None = None
        closed = False

        def capture(
            capture_base: Any,
            capture_rep: Any,
            annotators: list[Any],
            camera_profile: Mapping[str, Any],
            eye: list[float],
            lookat: list[float],
            ordinal: int,
        ) -> dict[str, Any]:
            nonlocal expected_ordinal, previous_frame, previous_rational
            if (
                closed
                or capture_base is not base
                or capture_rep is not rep
                or annotators
                or camera_profile != profile
                or ordinal != expected_ordinal
            ):
                raise ValueError("pull record capture identity/ordinal이 다르다")
            base.sim.set_camera_view(
                eye=eye,
                target=lookat,
                camera_prim_path=profile["camera_prim_path"],
            )
            context.reset_renderer_accumulation()
            physics_before = _snapshot(deps["physics_snapshot"](base))
            updates_before = read_counter()
            V1._capture_render_step(rep.orchestrator, profile)
            updates_after_render = read_counter()
            if updates_after_render <= updates_before:
                raise RuntimeError("render step이 native app update counter를 진행하지 않았다")
            raw: dict[str, Any] = {}
            pull_counters = [updates_after_render]
            for name, annotator in pairs:
                raw[name] = _snapshot(
                    annotator.get_data(device="cpu", do_array_copy=True)
                )
                pull_counters.append(read_counter())
            physics_after = _snapshot(deps["physics_snapshot"](base))
            if physics_after != physics_before:
                raise RuntimeError("pull record capture가 physics state/time을 바꿨다")
            if any(value != updates_after_render for value in pull_counters):
                raise RuntimeError("연속 annotator pull 사이에 app update가 있었다")
            rgb, rgb_bytes = _rgb(raw["rgb"])
            semantic, _ = _semantic(raw["semantic_segmentation"], rgb)
            camera = _camera(raw["CameraParams"], rgb)
            simulation_time = float(physics_before.get("simulation_time_s"))
            timing, frame, rational = _timing(
                raw["IsaacReadTimes"], raw["ReferenceTime"], simulation_time
            )
            if (
                (previous_frame is not None and frame <= previous_frame)
                or (previous_rational is not None and rational <= previous_rational)
            ):
                raise ValueError("pull record frame/rational time이 증가하지 않았다")
            npz_path, raw_receipt = _write_raw(
                job_path, ordinal, raw, rgb, semantic, camera
            )
            receipt_path = job_path.with_name(
                f"{job_path.stem}.pull-record-{ordinal:06d}.json"
            )
            R.write_new(
                receipt_path,
                {
                    "schema": SCHEMA,
                    "capture_ordinal": ordinal,
                    "annotator_order": list(ANNOTATORS),
                    "render_product": str(render_product),
                    "step": {
                        "call_count": 1,
                        "delta_time_s": profile["delta_time_s"],
                        "pause_timeline": profile["pause_timeline"],
                        "rt_subframes": profile["rt_subframes"],
                    },
                    "app_update_counter": {
                        **counter_source,
                        "before_render": updates_before,
                        "after_render_before_pull": updates_after_render,
                        "after_each_pull": pull_counters[1:],
                        "render_window_updates": updates_after_render - updates_before,
                        "pull_window_updates": 0,
                    },
                    "physics_before": _jsonable(physics_before),
                    "physics_after": _jsonable(physics_after),
                    "frame_number": frame,
                    "rational_time": [rational.numerator, rational.denominator],
                    "timing": _jsonable(timing),
                    "renderer": _jsonable(renderer),
                    "fxaa_readback_verified": use_fxaa,
                    "raw_npz": _descriptor(npz_path),
                    "raw_receipt": _descriptor(raw_receipt),
                    "complete_capture": True,
                    "reserved_marker_count": 0,
                    "atomicity_claimed": False,
                    "capture_approved": False,
                    "used_for_decision": False,
                },
            )
            previous_frame, previous_rational = frame, rational
            expected_ordinal += 1
            return {
                "rgb": rgb_bytes,
                "read_times": dict(raw["IsaacReadTimes"]),
                "reference_time": dict(raw["ReferenceTime"]),
                "pull_receipt": receipt_path.resolve(),
                "pull_raw_receipt": raw_receipt,
                "pull_npz": npz_path,
            }

        def cleanup() -> None:
            nonlocal closed
            if closed:
                raise RuntimeError("pull record cleanup이 두 번 호출됐다")
            closed = True
            errors: list[BaseException] = []
            for _, annotator in reversed(attached):
                try:
                    annotator.detach(render_product)
                except BaseException as error:  # noqa: BLE001
                    errors.append(error)
            if errors:
                raise errors[0]

        return capture, cleanup

    return factory
