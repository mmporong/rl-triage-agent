"""준비 callback과 네 atomic smoke frame을 한 SDK Writer attach로 수집한다."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from copy import deepcopy
from pathlib import Path
from types import TracebackType
from typing import Any

import record_vision_evidence_v1 as R
import vision_atomic_smoke_v1 as SMOKE
import vision_frame_sync_v1 as FRAME
import vision_prepared_capture_v1 as PREPARED
import vision_semantic_sdk_v1 as SDK
import vision_writer_payload_v1 as PAYLOAD

SDK_REPORT_SCHEMA = "vision_prepared_sdk_v1"
MODE = "prepared_capture_smoke_only"
EVENT_NAME = "rl-triage-prepared-smoke-v1"


class PreparedSDKFailure(RuntimeError):
    """최종 envelope와 첫 일반 오류를 함께 전달한다."""

    def __init__(self, report: dict[str, Any], primary: Exception) -> None:
        super().__init__(f"prepared SDK smoke failed: {_type_name(primary)}")
        self.report = report
        self.envelope = report
        self.primary = primary


def _type_name(value: Any) -> str:
    kind = type(value)
    return f"{kind.__module__}.{kind.__qualname__}"


def _create_writer(rep: Any, mux: PREPARED.PreparedCaptureMux) -> Any:
    """Replicator raw payload를 decode 전에 prepared mux로 보내는 Writer를 만든다."""

    class PreparedWriter(rep.Writer):
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
            self.callback_error: BaseException | None = None

        def write(self, data: dict[str, Any]) -> None:
            if self.callback_error is not None:
                return
            try:
                mux.observe(data)
            except BaseException as error:  # noqa: BLE001 - SDK callback 밖에서 재발생시킨다.
                self.callback_error = error
                if isinstance(error, Exception):
                    mux.source_error(error)

        def raise_callback_error(self) -> None:
            if self.callback_error is not None:
                raise self.callback_error

    return PreparedWriter()


def _load_dependencies() -> dict[str, Any]:
    deps = SDK._load_dependencies()
    deps["writer"] = _create_writer
    return deps


_DEPENDENCY_KEYS = frozenset(SDK._DEPENDENCY_KEYS)


def _validate_request(
    validated: Mapping[str, Any], seal_report: Callable[[Mapping[str, Any]], None]
) -> None:
    if not isinstance(validated, Mapping):
        raise TypeError("prepared SDK validated request가 mapping이 아니다")
    if not callable(seal_report):
        raise TypeError("prepared SDK seal_report가 callable이 아니다")
    job = validated.get("job")
    if not isinstance(job, Mapping) or job.get("mode") != MODE:
        raise ValueError("prepared SDK job mode가 다르다")
    job_path = validated.get("job_path")
    output = validated.get("output_directory")
    if not isinstance(job_path, Path) or not isinstance(output, Path):
        raise TypeError("prepared SDK job/output 경로가 Path가 아니다")
    if job_path.is_symlink() or not job_path.is_file():
        raise ValueError("prepared SDK job은 regular file이어야 한다")
    if (
        output.is_symlink()
        or not output.is_dir()
        or output.resolve() != (job_path.parent / "output").resolve()
    ):
        raise ValueError("prepared SDK output과 job 경계가 다르다")
    job_sha = validated.get("job_sha256")
    if not isinstance(job_sha, str) or len(job_sha) != 64:
        raise ValueError("prepared SDK job SHA-256이 잘못됐다")


def _write_bytes_new(path: Path, data: bytes) -> Path:
    if path.exists() or path.is_symlink():
        raise FileExistsError("prepared SDK artifact가 이미 있다")
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    return path.resolve()


def _destroy_render_product(render_product: Any) -> None:
    render_product.destroy()


def _remember(
    error: BaseException,
    primary: Exception | None,
    cancellation: BaseException | None,
    cancellation_tb: TracebackType | None,
) -> tuple[Exception | None, BaseException | None, TracebackType | None]:
    return SDK._remember_failure(error, primary, cancellation, cancellation_tb)


def _errors(phases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"phase": row["name"], "error_type": row["error_type"]}
        for row in phases
        if row["status"] in {"failed", "cancelled"}
    ]


def _envelope(
    prepared: Mapping[str, Any] | None,
    capture_report: Mapping[str, Any] | None,
    artifacts: Mapping[str, Any],
    lifecycle: Mapping[str, Any],
    renderer: Mapping[str, Any] | None,
    setup_evidence: Mapping[str, Any],
    phases: list[dict[str, Any]],
) -> dict[str, Any]:
    success_fields = (
        "setup_completed",
        "writer_attached",
        "prepare_completed",
        "writer_detached",
        "render_product_destroyed",
        "environment_closed",
        "mux_resolved",
        "mux_sealed",
        "journals_written",
        "capture_report_written",
        "report_verified",
    )
    errors = _errors(phases)
    approved = (
        prepared is not None
        and prepared.get("approved") is True
        and capture_report is not None
        and capture_report.get("status") == "pass"
        and all(lifecycle.get(field) is True for field in success_fields)
        and lifecycle.get("arms_completed") == 4
        and not errors
    )
    return {
        "schema": SDK_REPORT_SCHEMA,
        "prepared": deepcopy(dict(prepared)) if prepared is not None else None,
        "capture_report": (
            deepcopy(dict(capture_report)) if capture_report is not None else None
        ),
        "artifacts": deepcopy(dict(artifacts)),
        "lifecycle": {**deepcopy(dict(lifecycle)), "errors": errors},
        "renderer": deepcopy(dict(renderer)) if renderer is not None else {},
        "setup_evidence": deepcopy(dict(setup_evidence)),
        "renderer_smoke_only": True,
        "approved": approved,
    }


def run_prepared_smoke(
    validated: Mapping[str, Any],
    app: Any,
    *,
    seal_report: Callable[[Mapping[str, Any]], None],
    dependencies: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """한 Writer attach에서 prepare와 A-A-B-A smoke를 수집하고 cleanup 뒤 봉인한다."""
    _validate_request(validated, seal_report)
    deps = dict(dependencies) if dependencies is not None else _load_dependencies()
    if set(deps) != _DEPENDENCY_KEYS:
        raise ValueError("prepared SDK dependency 표면이 다르다")

    phases: list[dict[str, Any]] = []
    phase_order: list[str] = []
    lifecycle: dict[str, Any] = {
        "setup_completed": False,
        "writer_attach_attempted": False,
        "writer_attached": False,
        "prepare_completed": False,
        "arms_completed": 0,
        "writer_detach_attempted": False,
        "writer_detached": False,
        "render_product_destroyed": False,
        "environment_closed": False,
        "mux_resolved": False,
        "mux_sealed": False,
        "journals_written": False,
        "capture_report_written": False,
        "report_verified": False,
        "envelope_sealed": False,
    }
    extension = {
        "name": SDK.EXTENSION,
        "enable_result": False,
        "app_update_completed": False,
    }
    startup: dict[str, Any] = {"before": None, "after": None}
    registrations = {name: False for name in SDK.ANNOTATORS}
    setup_physics: dict[str, Any] = {"before": None, "after": None, "equal": None}
    camera_poses: list[dict[str, Any]] = []
    renderer: dict[str, Any] | None = None
    primary: Exception | None = None
    cancellation: BaseException | None = None
    cancellation_tb: TracebackType | None = None
    env: Any | None = None
    render_product: Any | None = None
    writer: Any | None = None
    mux: PREPARED.PreparedCaptureMux | None = None
    marker: Any | None = None
    stage: Any | None = None
    profile: Mapping[str, Any] | None = None

    def phase(name: str, call: Callable[[], Any]) -> Any:
        return SDK._phase(name, phases, phase_order, call)

    try:
        profile = phase("profile", deps["profile"])
        if not isinstance(profile, Mapping):
            raise TypeError("prepared SDK profile이 mapping이 아니다")
        camera_profile = profile["camera"]
        if not isinstance(camera_profile, Mapping):
            raise TypeError("prepared SDK camera profile이 mapping이 아니다")
        if (
            camera_profile.get("delta_time_s") != 0.0
            or camera_profile.get("rt_subframes") != 1
            or camera_profile.get("pause_timeline") is not False
        ):
            raise ValueError("prepared SDK renderer step profile이 등록값과 다르다")
        rep = deps["rep"]
        settings = phase("settings", deps["settings"])

        def enable_extension() -> bool:
            if deps["enable_extension"](SDK.EXTENSION) is not True:
                raise RuntimeError("prepared SDK isaacsim.core.nodes 활성화가 실패했다")
            return True

        extension["enable_result"] = phase("extension_enable", enable_extension)
        phase("extension_app_update", app.update)
        extension["app_update_completed"] = True
        startup = phase(
            "replicator_startup",
            lambda: SDK._startup_readback(deps["startup"](rep, settings)),
        )
        registrations = phase(
            "annotator_registrations",
            lambda: SDK._registration_readback(
                deps["registration_probe"](), registrations
            ),
        )
        context, stage_events = phase("usd_context", deps["usd_context"])
        env_cfg = phase("environment_config", lambda: SDK._configure_env(profile, deps))
        env = phase(
            "environment_create", lambda: deps["make_env"](profile["task"], env_cfg)
        )
        if env is None:
            raise RuntimeError("prepared SDK environment가 없다")
        active_base = env.unwrapped
        phase("environment_reset", lambda: active_base.reset(seed=profile["seed"]))

        def configure_command() -> None:
            command = active_base.command_manager.get_command("base_velocity")
            command[0, :3] = command.new_tensor([1.0, 0.0, 0.0])

        phase("command_configure", configure_command)
        setup_physics["before"] = phase(
            "setup_physics_before",
            lambda: deepcopy(deps["physics_snapshot"](active_base)),
        )
        stage = context.get_stage()
        if stage is None:
            raise RuntimeError("prepared SDK stage가 없다")
        camera = deps["camera"](stage.GetPrimAtPath(camera_profile["camera_prim_path"]))
        phase("camera_configure", lambda: SDK._configure_camera(camera, camera_profile))
        render_product = phase(
            "render_product_create",
            lambda: rep.create.render_product(
                camera_profile["camera_prim_path"], (R.WIDTH, R.HEIGHT)
            ),
        )
        if render_product is None:
            raise RuntimeError("prepared SDK render product가 없다")
        root = phase("marker_root_position", lambda: deps["root_position"](active_base))
        marker = phase("marker_create", lambda: deps["marker"](rep, root))
        if marker is None:
            raise RuntimeError("prepared SDK marker가 없다")
        policy = None
        if deps["fxaa_required"](validated["job"]):
            policy = phase("fxaa_apply", lambda: deps["fxaa_apply"](rep, settings))

        def renderer_readback() -> dict[str, Any]:
            observed = deps["renderer_readback"](context, camera_profile, settings)
            if not isinstance(observed, Mapping):
                raise TypeError("prepared SDK renderer readback이 mapping이 아니다")
            return dict(observed)

        renderer = phase("renderer_readback", renderer_readback)
        if not isinstance(renderer, dict):
            raise TypeError("prepared SDK renderer readback 결과가 dict가 아니다")
        if policy is not None:
            renderer["render_policy"] = policy
        phase(
            "capture_readiness",
            lambda: deps["readiness"](
                active_base,
                app,
                context,
                stage_events,
                Path(validated["job_path"]).with_suffix(".prepared-readiness.json"),
                renderer,
            ),
        )
        if policy is not None:
            phase(
                "fxaa_verify",
                lambda: deps["fxaa_verify"](deps["fxaa_readback"](settings)),
            )
        phase("marker_bind", lambda: marker.bind_and_validate(stage))

        def verify_setup_physics() -> dict[str, Any]:
            after = deepcopy(deps["physics_snapshot"](active_base))
            setup_physics["after"] = after
            setup_physics["equal"] = setup_physics["before"] == after
            if setup_physics["equal"] is not True:
                raise RuntimeError("prepared SDK setup이 physics 상태를 바꿨다")
            return after

        phase("setup_physics_after", verify_setup_physics)
        lifecycle["setup_completed"] = True
        mux = PREPARED.PreparedCaptureMux(
            render_product_ref=(
                render_product
                if isinstance(render_product, str)
                else str(render_product.path).rsplit("/", 1)[-1]
            ),
            camera_path=camera_profile["camera_prim_path"],
            resolution=[R.WIDTH, R.HEIGHT],
        )
        writer = deps["writer"](rep, mux)
        if writer is None:
            raise RuntimeError("prepared SDK writer가 없다")
        lifecycle["writer_attach_attempted"] = True
        phase(
            "writer_attach",
            lambda: writer.attach(
                render_product, trigger=rep.trigger.on_custom_event(EVENT_NAME)
            ),
        )
        lifecycle["writer_attached"] = True

        def pose(kind: str) -> dict[str, list[float]]:
            eye_offset = list(camera_profile["eye_offset_m"])
            if kind == "B":
                eye_offset[0] = -eye_offset[0]
            return {
                "eye_world_m": [root[index] + eye_offset[index] for index in range(3)],
                "lookat_world_m": [
                    root[index] + camera_profile["lookat_offset_m"][index]
                    for index in range(3)
                ],
            }

        def set_pose(kind: str) -> dict[str, Any]:
            selected = pose(kind)
            active_base.sim.set_camera_view(
                eye=selected["eye_world_m"],
                target=selected["lookat_world_m"],
                camera_prim_path=camera_profile["camera_prim_path"],
            )
            context.reset_renderer_accumulation()
            return selected

        phase("prepare_marker_3", lambda: marker.set_code(stage, 3))
        prepare_pose = phase("prepare_camera_A", lambda: set_pose("A"))
        camera_poses.append({"phase": "prepare", **prepare_pose})
        before = deps["physics_snapshot"](active_base)
        mux.begin_prepare(before, float(active_base.sim.current_time), marker_code=3)
        phase("prepare_event", lambda: rep.utils.send_og_event(EVENT_NAME))
        phase(
            "prepare_render",
            lambda: deps["capture_render_step"](rep.orchestrator, camera_profile),
        )
        writer.raise_callback_error()
        mux.end_prepare(
            deps["physics_snapshot"](active_base),
            float(active_base.sim.current_time),
        )
        lifecycle["prepare_completed"] = True

        for ordinal in range(4):
            phase(
                f"arm_{ordinal}_marker",
                lambda value=ordinal: marker.set_code(stage, value),
            )
            selected_pose = phase(
                f"arm_{ordinal}_camera",
                lambda value=ordinal: set_pose("B" if value == 2 else "A"),
            )
            camera_poses.append({"phase": f"arm_{ordinal}", **selected_pose})
            before = deps["physics_snapshot"](active_base)
            mux.begin_arm(
                ordinal,
                before,
                float(active_base.sim.current_time),
                marker_code=ordinal,
            )
            phase(f"arm_{ordinal}_event", lambda: rep.utils.send_og_event(EVENT_NAME))
            phase(
                f"arm_{ordinal}_render",
                lambda: deps["capture_render_step"](rep.orchestrator, camera_profile),
            )
            writer.raise_callback_error()
            mux.end_arm(
                ordinal,
                deps["physics_snapshot"](active_base),
                float(active_base.sim.current_time),
            )
            lifecycle["arms_completed"] = ordinal + 1
    except BaseException as error:  # noqa: BLE001 - cleanup과 실패 봉인을 계속한다.
        primary, cancellation, cancellation_tb = _remember(
            error, primary, cancellation, cancellation_tb
        )

    if mux is None:
        mux = PREPARED.PreparedCaptureMux(
            render_product_ref="unavailable",
            camera_path="unavailable",
            resolution=[R.WIDTH, R.HEIGHT],
        )
    if primary is not None:
        mux.source_error(primary)
    elif cancellation is not None:
        mux.source_error(RuntimeError("prepared SDK cancellation before seal"))

    lifecycle["writer_detach_attempted"] = writer is not None

    def detach() -> None:
        mux.begin_detach()
        try:
            if writer is None:
                raise AssertionError("prepared SDK writer detach 대상이 없다")
            writer.detach()
            writer.raise_callback_error()
        finally:
            mux.end_detach()

    detach_status, detach_error = SDK._cleanup_phase(
        "writer_detach", phases, phase_order, detach if writer is not None else None
    )
    lifecycle["writer_detached"] = detach_status == "complete"
    if detach_error is not None:
        primary, cancellation, cancellation_tb = _remember(
            detach_error, primary, cancellation, cancellation_tb
        )
        if isinstance(detach_error, Exception):
            mux.source_error(detach_error)
    if writer is None:
        mux.begin_detach()
        mux.end_detach()

    destroy_call: Callable[[], Any] | None = None
    if render_product is not None:
        destroy_call = lambda: _destroy_render_product(render_product)
    destroy_status, destroy_error = SDK._cleanup_phase(
        "render_product_destroy",
        phases,
        phase_order,
        destroy_call,
    )
    lifecycle["render_product_destroyed"] = destroy_status == "complete"
    if destroy_error is not None:
        primary, cancellation, cancellation_tb = _remember(
            destroy_error, primary, cancellation, cancellation_tb
        )
    close_status, close_error = SDK._cleanup_phase(
        "environment_close", phases, phase_order, env.close if env is not None else None
    )
    lifecycle["environment_closed"] = close_status == "complete"
    if close_error is not None:
        primary, cancellation, cancellation_tb = _remember(
            close_error, primary, cancellation, cancellation_tb
        )

    prepared: dict[str, Any] | None = None
    try:
        phase("mux_resolve", mux.resolve)
        lifecycle["mux_resolved"] = True
        prepared = phase("mux_seal", mux.seal)
        if not isinstance(prepared, dict):
            raise TypeError("prepared capture mux seal 결과가 dict가 아니다")
        lifecycle["mux_sealed"] = True
        if prepared.get("approved") is not True and primary is None:
            primary = RuntimeError("prepared capture mux가 capture를 승인하지 않았다")
    except BaseException as error:  # noqa: BLE001 - envelope seal을 계속한다.
        primary, cancellation, cancellation_tb = _remember(
            error, primary, cancellation, cancellation_tb
        )

    artifacts: dict[str, Any] = {"report": None, "journals": [], "audits": {}}
    capture_report: dict[str, Any] | None = None
    if primary is None and cancellation is None and prepared is not None:
        try:
            job_path = Path(validated["job_path"])
            output = Path(validated["output_directory"])
            bundles = prepared["bundles"]
            decoded_rows = [row["decoded"] for row in mux._arms]
            rows = []
            labels = SMOKE.ORDER
            arm_poses = camera_poses[-4:]
            for ordinal, (decoded, bundle, pose_row) in enumerate(
                zip(decoded_rows, bundles, arm_poses, strict=True)
            ):
                json_path, npz_path = PAYLOAD.write_atomic_journal(
                    job_path, ordinal, FRAME.SMOKE_MODE, decoded, bundle
                )
                artifacts["journals"].append(
                    {"ordinal": ordinal, "json": json_path.name, "npz": npz_path.name}
                )
                before = mux._arms[ordinal]["before"]
                rows.append(
                    {
                        "label": labels[ordinal],
                        "state_before": deepcopy(before),
                        "state_after": deepcopy(before),
                        "simulation_time_before_s": mux._arms[ordinal]["time"],
                        "simulation_time_after_s": mux._arms[ordinal]["time"],
                        "camera": {
                            "eye_world_m": pose_row["eye_world_m"],
                            "lookat_world_m": pose_row["lookat_world_m"],
                        },
                        "atomic_bundle": bundle,
                    }
                )
            lifecycle["journals_written"] = True
            audit_paths: dict[str, Path] = {}
            for ordinal in (1, 2, 3):
                path = _write_bytes_new(
                    output / f"{job_path.stem}.prepared-audit-rgb-{ordinal}.rgb",
                    decoded_rows[ordinal]["rgb"].tobytes(order="C"),
                )
                audit_paths[f"audit_rgb_{ordinal}"] = path
                artifacts["audits"][f"audit_rgb_{ordinal}"] = path.name
            capture_report = SMOKE._assess(rows, job_path)
            report_path = output / f"{job_path.stem}.smoke-report.json"
            R.write_new(report_path, capture_report)
            artifacts["report"] = report_path.name
            lifecycle["capture_report_written"] = True
            checked = SMOKE.verify_report(report_path, job_path, audit_paths)
            if checked != capture_report:
                raise ValueError("prepared SDK smoke report 재검증 결과가 다르다")
            lifecycle["report_verified"] = True
        except BaseException as error:  # noqa: BLE001 - 실패 envelope를 봉인한다.
            primary, cancellation, cancellation_tb = _remember(
                error, primary, cancellation, cancellation_tb
            )

    setup_evidence = {
        "phase_order": list(phase_order),
        "setup_physics_before_after": deepcopy(setup_physics),
        "extension": deepcopy(extension),
        "startup": deepcopy(startup),
        "registrations": deepcopy(registrations),
        "camera_poses": deepcopy(camera_poses),
        "raw_callback_count": mux.raw_callback_count,
        "complete_drain_proven": False,
        "all_future_callbacks_collected": False,
    }
    seal_row = {"name": "seal_report", "status": "running", "error_type": None}
    phases.append(seal_row)
    phase_order.append("seal_report")
    sealed = _envelope(
        prepared,
        capture_report,
        artifacts,
        lifecycle,
        renderer,
        setup_evidence,
        phases,
    )
    try:
        seal_report(deepcopy(sealed))
    except BaseException as error:  # noqa: BLE001 - 첫 오류와 cancellation을 보존한다.
        seal_row["status"] = (
            "cancelled" if not isinstance(error, Exception) else "failed"
        )
        seal_row["error_type"] = _type_name(error)
        primary, cancellation, cancellation_tb = _remember(
            error, primary, cancellation, cancellation_tb
        )
    else:
        seal_row["status"] = "complete"
        lifecycle["envelope_sealed"] = True

    terminal = _envelope(
        prepared,
        capture_report,
        artifacts,
        lifecycle,
        renderer,
        setup_evidence,
        phases,
    )
    if cancellation is not None:
        cancellation.__dict__["prepared_sdk_envelope"] = terminal
        raise cancellation.with_traceback(cancellation_tb)
    if primary is not None:
        raise PreparedSDKFailure(terminal, primary) from primary
    return terminal
