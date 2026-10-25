"""Atomic Writer를 readiness·marker·renderer reset과 결합하는 capture factory."""

from __future__ import annotations

import hashlib
import traceback
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import record_vision_evidence_v1 as R
import vision_capture_readiness_v1 as READY
import vision_frame_sync_v1 as FRAME
import vision_fxaa_policy_v1 as FXAA
import vision_play_worker_v1 as V1
import vision_timing_adapter_v1 as TIMING
import vision_writer_payload_v1 as PAYLOAD

RESET_SCHEMA = "vision_capture_renderer_reset_v1"
DESCRIPTOR_SCHEMA = "vision_writer_payload_descriptor_v1"
MARKER_ROOT = "rl_triage_atomic"


def _root_position(base: Any) -> list[float]:
    robot = base.scene["robot"]
    return [
        float(item) for item in robot.data.root_pos_w[0, :3].detach().cpu().tolist()
    ]


def _render_product_name(render_product: Any) -> str:
    raw = (
        render_product
        if isinstance(render_product, str)
        else getattr(render_product, "path", None)
    )
    if not isinstance(raw, str) or not raw or not raw.rsplit("/", 1)[-1]:
        raise ValueError("RenderProduct path에서 Writer identity를 얻을 수 없다")
    return raw.rsplit("/", 1)[-1]


def _verify_decoded_camera(decoded: Mapping[str, Any]) -> None:
    expected = R.fixed_profile()["camera"]
    if (
        decoded.get("camera_path") != expected["camera_prim_path"]
        or decoded.get("render_product_resolution") != [R.WIDTH, R.HEIGHT]
    ):
        raise ValueError("Writer camera path 또는 resolution이 fixed profile과 다르다")


def _marker_specs(root: list[float]) -> list[dict[str, Any]]:
    offsets = {
        "sentinel": (0.0, 0.0, 1.05),
        "bit0/rail0": (-0.34, -0.18, 0.82),
        "bit0/rail1": (0.34, -0.18, 0.82),
        "bit1/rail0": (-0.34, 0.18, 1.18),
        "bit1/rail1": (0.34, 0.18, 1.18),
    }
    rows = [
        {
            "key": "sentinel",
            "label": FRAME.SENTINEL,
            "role": "sentinel",
            "position": [
                root[index] + offsets["sentinel"][index] for index in range(3)
            ],
            "visible": True,
        }
    ]
    for bit in range(2):
        for rail in range(2):
            key = f"bit{bit}/rail{rail}"
            for role in ("active", "inactive"):
                rows.append(
                    {
                        "key": f"{key}/{role}",
                        "label": f"{FRAME.MARKER_PREFIX}{key}/{role}",
                        "role": role,
                        "position": [
                            root[index] + offsets[key][index] for index in range(3)
                        ],
                        "visible": False,
                    }
                )
    return rows


class MarkerRig:
    """Replicator로 만든 아홉 marker와 실제 USD visibility를 소유한다."""

    def __init__(self, rep: Any, root: list[float]) -> None:
        colors = {
            "sentinel": (0.05, 1.0, 0.55),
            "active": (1.0, 0.05, 0.55),
            "inactive": (0.05, 0.55, 1.0),
        }
        self._items: dict[str, Any] = {}
        self._paths: dict[str, str] = {}
        for row in _marker_specs(root):
            material = rep.create.material_omnipbr(
                diffuse=colors[row["role"]],
                emissive_color=colors[row["role"]],
                emissive_intensity=500.0,
            )
            self._items[row["key"]] = rep.create.cube(
                as_mesh=False,
                semantics=[("class", row["label"])],
                material=material,
                position=row["position"],
                scale=(0.09, 0.09, 0.09),
                visible=row["visible"],
                name=f"{MARKER_ROOT}_{row['key'].replace('/', '_')}",
            )

    def bind_and_validate(self, stage: Any) -> None:
        from pxr import Usd, UsdGeom, UsdPhysics  # pyright: ignore[reportMissingImports]

        paths: dict[str, str] = {}
        for key, item in self._items.items():
            outputs = item.get_output_prims()
            prims = outputs.get("prims") if isinstance(outputs, Mapping) else None
            if not isinstance(prims, list) or len(prims) != 1:
                count = len(prims) if isinstance(prims, list) else None
                raise ValueError(
                    "marker output prim이 단일 Usd.Prim이 아니다 "
                    f"(outputs={type(outputs).__name__}, "
                    f"prims={type(prims).__name__}, count={count})"
                )
            output_prim = prims[0]
            if not isinstance(output_prim, Usd.Prim):
                raise TypeError(
                    "marker output prim이 Usd.Prim이 아니다 "
                    f"(type={type(output_prim).__name__})"
                )
            if output_prim.GetStage() != stage:
                raise ValueError("marker output prim이 현재 USD stage에 속하지 않는다")
            path = str(output_prim.GetPath())
            prim = stage.GetPrimAtPath(path)
            if (
                not prim
                or not prim.IsValid()
                or prim != output_prim
                or not UsdGeom.Imageable(prim)
            ):
                raise ValueError("marker prim이 유효한 Imageable이 아니다")
            for descendant in Usd.PrimRange(prim):
                if descendant.HasAPI(UsdPhysics.CollisionAPI) or descendant.HasAPI(
                    UsdPhysics.RigidBodyAPI
                ):
                    raise ValueError("marker 또는 자손에 physics API가 있다")
            paths[key] = path
        if len(paths) != 9:
            raise ValueError("atomic marker는 정확히 아홉 prim이어야 한다")
        self._paths = paths

    def set_code(self, stage: Any, code: int) -> None:
        from pxr import UsdGeom  # pyright: ignore[reportMissingImports]

        if not self._paths:
            raise RuntimeError("marker stage binding이 끝나지 않았다")
        if type(code) is not int or code not in range(4):
            raise ValueError("marker code는 0..3이어야 한다")
        for key, path in self._paths.items():
            imageable = UsdGeom.Imageable(stage.GetPrimAtPath(path))
            visible = key == "sentinel"
            if key != "sentinel":
                bit = int(key[3])
                rail = int(key[9])
                role = key.rsplit("/", 1)[-1]
                visible = role == (
                    "active" if ((code >> bit) & 1) == rail else "inactive"
                )
            imageable.MakeVisible() if visible else imageable.MakeInvisible()


def _write_reset(
    validated: Mapping[str, Any], ordinal: int, *, use_fxaa: bool, settings: Any
) -> Path:
    path = Path(validated["job_path"]).with_name(
        f"{Path(validated['job_path']).stem}.capture-{ordinal:06d}.renderer-reset.json"
    )
    body: dict[str, Any] = {
        "schema": RESET_SCHEMA,
        "capture_ordinal": ordinal,
        "completed": True,
        "method": "omni.usd.UsdContext.reset_renderer_accumulation",
    }
    if use_fxaa:
        observed = FXAA.readback(settings)
        FXAA.verify_readback(observed)
        body["render_policy_readback"] = observed
    R.write_new(path, body)
    return path.resolve()


def _detach(writer: Any, primary_error: BaseException | None) -> None:
    try:
        writer.detach()
    except BaseException as error:
        if primary_error is not None:
            traceback.print_exception(error)
        else:
            raise


def _write_writer_descriptor(
    validated: Mapping[str, Any], ordinal: int, writer: Any
) -> Path | None:
    """값을 제외한 실제 callback 구조를 job에 결합해 write-once로 남긴다."""
    descriptor = writer.last_payload_descriptor
    if descriptor is None:
        return None
    if not isinstance(descriptor, Mapping):
        raise TypeError("Writer callback descriptor가 mapping이 아니다")
    job_path = Path(validated["job_path"])
    if job_path.is_symlink() or not job_path.is_file():
        raise ValueError("Writer descriptor job은 regular file이어야 한다")
    error = writer.callback_error
    body = {
        "schema": DESCRIPTOR_SCHEMA,
        "capture_ordinal": ordinal,
        "job_sha256": hashlib.sha256(job_path.read_bytes()).hexdigest(),
        "callback_count": writer.callback_count,
        "callback_error_type": type(error).__name__ if error is not None else None,
        "descriptor": dict(descriptor),
    }
    path = job_path.with_name(
        f"{job_path.stem}.writer-descriptor-{ordinal:06d}.json"
    )
    R.write_new(path, body)
    return path


def make_capture_factory(
    validated: Mapping[str, Any],
    app: Any,
    *,
    dependencies: Mapping[str, Any] | None = None,
) -> V1.CameraCaptureFactory:
    """V1 worker가 RenderProduct 생성 뒤 호출할 atomic capture factory를 만든다."""
    mode = validated.get("job", {}).get("mode")
    if mode not in {R.RECORD_MODE, R.RENDER_SYNC_SMOKE_MODE}:
        raise ValueError("atomic capture job mode가 다르다")
    deps = {
        "usd_context": READY._usd_context,
        "settings": READY._carb_settings,
        "writer": PAYLOAD.create_payload_writer,
        "marker": MarkerRig,
    }
    if dependencies is not None:
        deps.update(dependencies)

    def factory(
        base: Any, rep: Any, render_product: Any, profile: Mapping[str, Any]
    ) -> tuple[V1.CameraCapture, Callable[[], None]]:
        smoke = mode == R.RENDER_SYNC_SMOKE_MODE
        expected_count = 4 if smoke else R.FRAME_COUNT
        sync = FRAME.AtomicFrameWriter(
            FRAME.SMOKE_MODE if smoke else FRAME.RECORD_MODE,
            expected_count,
            render_product_ref=_render_product_name(render_product),
        )
        context, stage_events = deps["usd_context"]()
        settings = deps["settings"]()
        use_fxaa = FXAA.required(validated["job"])
        policy = FXAA.apply(rep, settings) if use_fxaa else None
        marker = deps["marker"](rep, _root_position(base)) if smoke else None
        renderer = READY._renderer_readback(context, profile, settings)
        if policy is not None:
            renderer["render_policy"] = policy
        READY._wait_capture_ready(
            base,
            app,
            context,
            stage_events,
            Path(validated["job_path"]).with_suffix(READY.SIDECAR_SUFFIX),
            renderer,
        )
        if marker is not None:
            marker.bind_and_validate(context.get_stage())
        if use_fxaa:
            FXAA.verify_readback(FXAA.readback(settings))

        persistent_writer: Any | None = None
        persistent_event = "rl-triage-atomic-record"
        last_decoded: dict[str, Any] | None = None

        def sink(decoded: dict[str, Any]) -> None:
            nonlocal last_decoded
            last_decoded = decoded
            sync.callback(PAYLOAD.frame_payload(decoded))

        if not smoke:
            selected_writer = deps["writer"](rep, sink)
            selected_writer.attach(
                render_product, trigger=rep.trigger.on_custom_event(persistent_event)
            )
            persistent_writer = selected_writer

        def capture(
            capture_base: Any,
            capture_rep: Any,
            annotators: list[Any],
            camera_profile: Mapping[str, Any],
            eye: list[float],
            lookat: list[float],
            ordinal: int,
        ) -> dict[str, Any]:
            nonlocal last_decoded
            if capture_base is not base or capture_rep is not rep or annotators:
                raise ValueError("atomic capture factory 호출 identity가 다르다")
            if camera_profile != profile:
                raise ValueError("atomic capture camera profile이 factory와 다르다")
            writer = persistent_writer
            if writer is not None:
                try:
                    writer.raise_callback_error()
                except BaseException:
                    try:
                        _write_writer_descriptor(validated, ordinal, writer)
                    except BaseException as descriptor_error:  # noqa: BLE001
                        traceback.print_exception(descriptor_error)
                    raise
            before = READY._physics_snapshot(base)
            simulation_time = float(base.sim.current_time)
            expected_code = ordinal if smoke else None
            sync.arm(ordinal, expected_code, before, simulation_time)

            if smoke:
                if marker is None:
                    raise RuntimeError("smoke marker rig가 없다")
                marker.set_code(context.get_stage(), ordinal)

            def reset_after_camera() -> None:
                context.reset_renderer_accumulation()
                _write_reset(validated, ordinal, use_fxaa=use_fxaa, settings=settings)

            proxy = READY._BaseProxy(base, reset_after_camera)
            proxy.sim.set_camera_view(
                eye=eye,
                target=lookat,
                camera_prim_path=profile["camera_prim_path"],
            )
            event = persistent_event
            if smoke:
                event = f"rl-triage-atomic-smoke-{ordinal}"
                writer = deps["writer"](rep, sink)
                writer.attach(
                    render_product, trigger=rep.trigger.on_custom_event(event)
                )
            if writer is None:
                raise RuntimeError("atomic Writer가 생성되지 않았다")
            primary: BaseException | None = None
            descriptor_written = False
            try:
                callback_before = writer.callback_count
                last_decoded = None
                sync.before_step(ordinal)
                rep.utils.send_og_event(event)
                V1._capture_render_step(rep.orchestrator, profile)
                writer.raise_callback_error()
                if writer.callback_count != callback_before + 1 or last_decoded is None:
                    raise RuntimeError(
                        "custom event capture callback이 정확히 한 번 오지 않았다"
                    )
                if smoke or ordinal == 0:
                    _write_writer_descriptor(validated, ordinal, writer)
                    descriptor_written = True
                _verify_decoded_camera(last_decoded)
                after = READY._physics_snapshot(base)
                bundle = sync.close(ordinal, after, float(base.sim.current_time))
                journal, npz = PAYLOAD.write_atomic_journal(
                    Path(validated["job_path"]),
                    ordinal,
                    sync.mode,
                    last_decoded,
                    bundle,
                )
                rgb = last_decoded["rgb"].tobytes(order="C")
                TIMING.write_capture_journal(
                    Path(validated["job_path"]),
                    ordinal,
                    last_decoded["read_times_observed"],
                    last_decoded["reference_time"],
                    bundle["rgb"]["sha256"],
                    rgb_bytes=rgb if smoke else None,
                )
                if ordinal + 1 == expected_count:
                    sync.finish()
                return {
                    "rgb": rgb,
                    "read_times": bundle["timing"]["read_times"],
                    "reference_time": bundle["timing"]["reference_time"],
                    "atomic_bundle": bundle,
                    "atomic_journal": journal,
                    "atomic_npz": npz,
                }
            except BaseException as error:
                primary = error
                if not descriptor_written:
                    try:
                        _write_writer_descriptor(validated, ordinal, writer)
                    except BaseException as descriptor_error:  # noqa: BLE001
                        traceback.print_exception(descriptor_error)
                raise
            finally:
                if smoke:
                    _detach(writer, primary)

        def cleanup() -> None:
            errors: list[BaseException] = []
            if persistent_writer is not None:
                try:
                    persistent_writer.raise_callback_error()
                except BaseException as error:  # noqa: BLE001
                    errors.append(error)
            try:
                sync.finish()
            except BaseException as error:  # noqa: BLE001 - persistent Writer detach를 이어간다.
                errors.append(error)
            if persistent_writer is not None:
                try:
                    persistent_writer.detach()
                except BaseException as error:  # noqa: BLE001
                    errors.append(error)
            if errors:
                raise errors[0]

        return capture, cleanup

    return factory
