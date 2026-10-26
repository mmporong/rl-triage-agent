"""한 view의 semantic callback을 capture 승인 없이 SDK에서 관찰한다."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from pathlib import Path
from types import TracebackType
from typing import Any

import record_vision_evidence_v1 as R
import vision_atomic_capture_v1 as ATOMIC
import vision_capture_readiness_v1 as READY
import vision_fxaa_policy_v1 as FXAA
import vision_play_worker_v1 as V1
import vision_play_worker_v2 as V2
import vision_replicator_startup_v1 as STARTUP
import vision_semantic_diagnostic_v1 as D

SDK_REPORT_SCHEMA = "vision_semantic_sdk_observation_v1"
MODE = "semantic_diagnostic_only"
EVENT_COUNT = 2
EVENT_NAME = "rl-triage-semantic-diagnostic-v1"
EXTENSION = "isaacsim.core.nodes"
ANNOTATORS = ("rgb", "semantic_segmentation", "CameraParams", "IsaacReadTimes")


class SDKDiagnosticFailure(RuntimeError):
    """최종 SDK envelope와 첫 일반 오류를 함께 전달한다."""

    def __init__(self, report: dict[str, Any], primary: Exception) -> None:
        super().__init__(f"semantic SDK diagnostic failed: {_type_name(primary)}")
        self.report = report
        self.envelope = report
        self.primary = primary


def _type_name(value: Any) -> str:
    kind = type(value)
    return f"{kind.__module__}.{kind.__qualname__}"


def _sdk_enable_extension(name: str) -> bool:
    return V2._enable_extension(name)


def _sdk_registration_probe(rep: Any) -> dict[str, bool]:
    return {
        name: rep.AnnotatorRegistry.get_annotator(name) is not None
        for name in ANNOTATORS
    }


def _load_dependencies() -> dict[str, Any]:
    """Isaac가 있는 child process에서만 SDK 모듈을 불러온다."""
    import gymnasium as gym  # pyright: ignore[reportMissingImports]
    import isaaclab_tasks  # noqa: F401  # pyright: ignore[reportMissingImports]
    import omni.replicator.core as rep  # pyright: ignore[reportMissingImports]
    from isaaclab_tasks.utils import parse_env_cfg  # pyright: ignore[reportMissingImports]
    from pxr import UsdGeom  # pyright: ignore[reportMissingImports]

    return {
        "profile": R.fixed_profile,
        "startup": lambda selected_rep, settings: STARTUP.configure_replicator_startup(
            settings=settings, replicator=selected_rep
        ),
        "parse_env_cfg": parse_env_cfg,
        "make_env": lambda task, cfg: gym.make(task, cfg=cfg),
        "rep": rep,
        "usd_context": READY._usd_context,
        "settings": READY._carb_settings,
        "enable_extension": _sdk_enable_extension,
        "registration_probe": lambda: _sdk_registration_probe(rep),
        "camera": UsdGeom.Camera,
        "readiness": READY._wait_capture_ready,
        "renderer_readback": READY._renderer_readback,
        "fxaa_required": FXAA.required,
        "fxaa_apply": FXAA.apply,
        "fxaa_readback": FXAA.readback,
        "fxaa_verify": FXAA.verify_readback,
        "marker": ATOMIC.MarkerRig,
        "root_position": ATOMIC._root_position,
        "writer": D.create_diagnostic_writer,
        "physics_snapshot": READY._physics_snapshot,
        "capture_render_step": V1._capture_render_step,
    }


_DEPENDENCY_KEYS = {
    "profile",
    "startup",
    "parse_env_cfg",
    "make_env",
    "rep",
    "usd_context",
    "settings",
    "enable_extension",
    "registration_probe",
    "camera",
    "readiness",
    "renderer_readback",
    "fxaa_required",
    "fxaa_apply",
    "fxaa_readback",
    "fxaa_verify",
    "marker",
    "root_position",
    "writer",
    "physics_snapshot",
    "capture_render_step",
}


def _validate_request(
    validated: Mapping[str, Any], seal_report: Callable[[Mapping[str, Any]], None]
) -> None:
    if not isinstance(validated, Mapping):
        raise TypeError("semantic SDK validated request가 mapping이 아니다")
    if not callable(seal_report):
        raise TypeError("semantic SDK seal_report가 callable이 아니다")
    job = validated.get("job")
    if not isinstance(job, Mapping) or job.get("mode") != MODE:
        raise ValueError("semantic SDK job mode가 다르다")
    if not isinstance(validated.get("job_path"), Path):
        raise TypeError("semantic SDK job_path가 Path가 아니다")
    if not isinstance(validated.get("output_directory"), Path):
        raise TypeError("semantic SDK output_directory가 Path가 아니다")
    job_sha = validated.get("job_sha256")
    if not isinstance(job_sha, str) or len(job_sha) != 64:
        raise ValueError("semantic SDK job SHA-256이 잘못됐다")


def _configure_env(profile: Mapping[str, Any], deps: Mapping[str, Any]) -> Any:
    env_cfg = deps["parse_env_cfg"](profile["task"], device="cuda:0", num_envs=1)
    env_cfg.seed = profile["seed"]
    env_cfg.actions.joint_pos.scale = profile["action_scale"]
    command_cfg = env_cfg.commands.base_velocity
    command_cfg.heading_command = False
    command_cfg.rel_standing_envs = 0.0
    command_cfg.resampling_time_range = tuple(
        profile["command"]["resampling_time_range_s"]
    )
    command_cfg.ranges.lin_vel_x = (1.0, 1.0)
    command_cfg.ranges.lin_vel_y = (0.0, 0.0)
    command_cfg.ranges.ang_vel_z = (0.0, 0.0)
    return env_cfg


def _configure_camera(camera: Any, profile: Mapping[str, Any]) -> None:
    camera.GetProjectionAttr().Set(profile["projection"])
    camera.GetHorizontalApertureAttr().Set(profile["horizontal_aperture_mm"])
    camera.GetFocalLengthAttr().Set(profile["focal_length_mm"])
    camera.GetClippingRangeAttr().Set(tuple(profile["clipping_range_m"]))


def _startup_readback(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, Mapping):
        raise TypeError("semantic SDK startup readback이 mapping이 아니다")
    before = value.get("before")
    after = value.get("after")
    if not isinstance(before, Mapping) or not isinstance(after, Mapping):
        raise TypeError("semantic SDK startup before/after가 mapping이 아니다")
    return {"before": dict(before), "after": dict(after)}


def _registration_readback(value: Any, evidence: dict[str, bool]) -> dict[str, bool]:
    if not isinstance(value, Mapping):
        raise TypeError("semantic SDK annotator 등록 probe가 mapping이 아니다")
    observed = {name: value.get(name) is True for name in ANNOTATORS}
    evidence.update(observed)
    missing = [name for name in ANNOTATORS if not observed[name]]
    if missing:
        raise RuntimeError(f"semantic SDK annotator 등록 누락: {','.join(missing)}")
    return observed


def _phase(
    name: str,
    phases: list[dict[str, Any]],
    phase_order: list[str],
    call: Callable[[], Any],
) -> Any:
    phase_order.append(name)
    row = {"name": name, "status": "running", "error_type": None}
    phases.append(row)
    try:
        value = call()
    except BaseException as error:
        row["status"] = "cancelled" if not isinstance(error, Exception) else "failed"
        if isinstance(error, D.DiagnosticFailure):
            row["error_type"] = _type_name(error.primary)
        else:
            row["error_type"] = _type_name(error)
        raise
    row["status"] = "complete"
    return value


def _cleanup_phase(
    name: str,
    phases: list[dict[str, Any]],
    phase_order: list[str],
    call: Callable[[], Any] | None,
) -> tuple[str, BaseException | None]:
    if call is None:
        phases.append({"name": name, "status": "not_applicable", "error_type": None})
        phase_order.append(name)
        return "not_applicable", None
    try:
        _phase(name, phases, phase_order, call)
    except BaseException as error:  # noqa: BLE001 - 다음 자원 정리를 계속한다.
        return "failed", error
    return "complete", None


def _envelope(
    diagnostic: Mapping[str, Any],
    phases: list[dict[str, Any]],
    lifecycle: Mapping[str, Any],
    renderer: Mapping[str, Any] | None,
    setup_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    errors = [
        {"phase": row["name"], "error_type": row["error_type"]}
        for row in phases
        if row["status"] in {"failed", "cancelled"}
    ]
    return {
        "schema": SDK_REPORT_SCHEMA,
        "diagnostic": deepcopy(dict(diagnostic)),
        "lifecycle": {**deepcopy(dict(lifecycle)), "errors": errors},
        "renderer": deepcopy(dict(renderer)) if renderer is not None else {},
        "setup_evidence": deepcopy(dict(setup_evidence)),
        "no_capture_approved": True,
    }


def _remember_failure(
    error: BaseException,
    primary: Exception | None,
    cancellation: BaseException | None,
    cancellation_tb: TracebackType | None,
) -> tuple[Exception | None, BaseException | None, TracebackType | None]:
    if isinstance(error, Exception):
        if primary is None:
            primary = error
    elif cancellation is None:
        cancellation = error
        cancellation_tb = error.__traceback__
    return primary, cancellation, cancellation_tb


def run_one_view(
    validated: Mapping[str, Any],
    app: Any,
    *,
    seal_report: Callable[[Mapping[str, Any]], None],
    dependencies: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """고정 view의 semantic callback 두 event를 진단 evidence로만 반환한다."""
    _validate_request(validated, seal_report)
    deps = dict(dependencies) if dependencies is not None else _load_dependencies()
    if set(deps) != _DEPENDENCY_KEYS:
        raise ValueError("semantic SDK dependency 표면이 다르다")

    observer = D.DiagnosticObserver(max_callbacks=D.MAX_CALLBACKS)
    phases: list[dict[str, Any]] = []
    phase_order: list[str] = []
    setup_physics: dict[str, Any] = {"before": None, "after": None, "equal": None}
    extension_evidence = {
        "name": EXTENSION,
        "enable_result": False,
        "app_update_completed": False,
    }
    startup_evidence: dict[str, Any] = {"before": None, "after": None}
    registrations = {name: False for name in ANNOTATORS}
    lifecycle: dict[str, Any] = {
        "setup_completed": False,
        "writer_attach_attempted": False,
        "writer_attached": False,
        "writer_detach_attempted": False,
        "writer_detached": False,
        "report_sealed": False,
        "render_product_destroyed": False,
        "environment_closed": False,
    }
    renderer: dict[str, Any] | None = None
    diagnostic: dict[str, Any] | None = None
    primary: Exception | None = None
    cancellation: BaseException | None = None
    cancellation_tb: TracebackType | None = None
    env: Any | None = None
    render_product: Any | None = None
    writer: Any | None = None
    detach_complete = False
    callback_count_before_detach = 0
    callback_count_after_detach = 0

    try:
        profile = _phase("profile", phases, phase_order, deps["profile"])
        rep = deps["rep"]
        settings = _phase("settings", phases, phase_order, deps["settings"])

        def enable_extension() -> bool:
            enabled = deps["enable_extension"](EXTENSION)
            if enabled is not True:
                raise RuntimeError("semantic SDK isaacsim.core.nodes 활성화가 실패했다")
            return True

        extension_evidence["enable_result"] = _phase(
            "extension_enable", phases, phase_order, enable_extension
        )
        _phase("extension_app_update", phases, phase_order, app.update)
        extension_evidence["app_update_completed"] = True
        startup_evidence = _phase(
            "replicator_startup",
            phases,
            phase_order,
            lambda: _startup_readback(deps["startup"](rep, settings)),
        )
        registrations = _phase(
            "annotator_registrations",
            phases,
            phase_order,
            lambda: _registration_readback(deps["registration_probe"](), registrations),
        )
        context, stage_events = _phase(
            "usd_context", phases, phase_order, deps["usd_context"]
        )
        env_cfg = _phase(
            "environment_config",
            phases,
            phase_order,
            lambda: _configure_env(profile, deps),
        )
        selected_env = _phase(
            "environment_create",
            phases,
            phase_order,
            lambda: deps["make_env"](profile["task"], env_cfg),
        )
        env = selected_env
        base = selected_env.unwrapped
        _phase(
            "environment_reset",
            phases,
            phase_order,
            lambda: base.reset(seed=profile["seed"]),
        )

        def configure_command() -> None:
            command_buffer = base.command_manager.get_command("base_velocity")
            command_buffer[0, :3] = command_buffer.new_tensor([1.0, 0.0, 0.0])

        _phase("command_configure", phases, phase_order, configure_command)
        setup_physics["before"] = _phase(
            "setup_physics_before",
            phases,
            phase_order,
            lambda: deepcopy(deps["physics_snapshot"](base)),
        )

        camera_profile = profile["camera"]
        stage = context.get_stage()
        camera_prim = stage.GetPrimAtPath(camera_profile["camera_prim_path"])
        camera = deps["camera"](camera_prim)
        _phase(
            "camera_configure",
            phases,
            phase_order,
            lambda: _configure_camera(camera, camera_profile),
        )
        render_product = _phase(
            "render_product_create",
            phases,
            phase_order,
            lambda: rep.create.render_product(
                camera_profile["camera_prim_path"], (R.WIDTH, R.HEIGHT)
            ),
        )
        root = _phase(
            "marker_root_position",
            phases,
            phase_order,
            lambda: deps["root_position"](base),
        )
        marker = _phase(
            "marker_create",
            phases,
            phase_order,
            lambda: deps["marker"](rep, root),
        )
        use_fxaa = bool(deps["fxaa_required"](validated["job"]))
        if use_fxaa:
            policy = _phase(
                "fxaa_apply",
                phases,
                phase_order,
                lambda: deps["fxaa_apply"](rep, settings),
            )
        else:
            policy = None

        def read_renderer() -> dict[str, Any]:
            observed = deps["renderer_readback"](context, camera_profile, settings)
            if not isinstance(observed, Mapping):
                raise TypeError("semantic SDK renderer readback이 mapping이 아니다")
            return dict(observed)

        selected_renderer = _phase(
            "renderer_readback",
            phases,
            phase_order,
            read_renderer,
        )
        renderer = selected_renderer
        if policy is not None:
            selected_renderer["render_policy"] = policy
        _phase(
            "capture_readiness",
            phases,
            phase_order,
            lambda: deps["readiness"](
                base,
                app,
                context,
                stage_events,
                Path(validated["job_path"]).with_suffix(READY.SIDECAR_SUFFIX),
                selected_renderer,
            ),
        )
        if use_fxaa:
            _phase(
                "fxaa_verify",
                phases,
                phase_order,
                lambda: deps["fxaa_verify"](deps["fxaa_readback"](settings)),
            )
        _phase(
            "marker_bind",
            phases,
            phase_order,
            lambda: marker.bind_and_validate(stage),
        )
        _phase("marker_code_0", phases, phase_order, lambda: marker.set_code(stage, 0))
        eye = [
            root[index] + camera_profile["eye_offset_m"][index] for index in range(3)
        ]
        lookat = [
            root[index] + camera_profile["lookat_offset_m"][index] for index in range(3)
        ]
        _phase(
            "camera_view",
            phases,
            phase_order,
            lambda: base.sim.set_camera_view(
                eye=eye,
                target=lookat,
                camera_prim_path=camera_profile["camera_prim_path"],
            ),
        )
        _phase(
            "renderer_reset",
            phases,
            phase_order,
            context.reset_renderer_accumulation,
        )

        def verify_setup_physics() -> dict[str, Any]:
            after = deepcopy(deps["physics_snapshot"](base))
            setup_physics["after"] = after
            setup_physics["equal"] = setup_physics["before"] == after
            if not setup_physics["equal"]:
                raise RuntimeError("semantic SDK setup이 physics 상태를 바꿨다")
            return after

        setup_physics["after"] = _phase(
            "setup_physics_after",
            phases,
            phase_order,
            verify_setup_physics,
        )
        lifecycle["setup_completed"] = True

        selected_writer = deps["writer"](rep, observer)
        writer = selected_writer
        lifecycle["writer_attach_attempted"] = True
        observer.window = "attaching"
        _phase(
            "writer_attach",
            phases,
            phase_order,
            lambda: selected_writer.attach(
                render_product, trigger=rep.trigger.on_custom_event(EVENT_NAME)
            ),
        )
        observer.window = "after_writer_attach_before_collection"
        lifecycle["writer_attached"] = True

        def run_event(index: int) -> None:
            def source() -> None:
                rep.utils.send_og_event(EVENT_NAME)
                deps["capture_render_step"](rep.orchestrator, camera_profile)

            _phase(f"event_{index}", phases, phase_order, source)

        def collect() -> dict[str, Any]:
            nonlocal diagnostic
            diagnostic = D.collect_diagnostic(
                observer,
                run_event=run_event,
                physics_snapshot=lambda: deps["physics_snapshot"](base),
                event_count=EVENT_COUNT,
            )
            observations = diagnostic["observations"]
            if observations["physics_changed"]:
                raise RuntimeError("semantic SDK event가 physics 상태를 바꿨다")
            if observations["overflow"]:
                raise RuntimeError("semantic SDK callback 관찰이 상한을 넘었다")
            if (
                diagnostic["events_attempted"] != EVENT_COUNT
                or diagnostic["events_completed"] != EVENT_COUNT
            ):
                raise RuntimeError(
                    "semantic SDK event 수집이 두 event를 완료하지 못했다"
                )
            return diagnostic

        _phase("diagnostic_collect", phases, phase_order, collect)
    except BaseException as error:  # noqa: BLE001 - detach와 evidence 봉인을 계속한다.
        if isinstance(error, D.DiagnosticFailure):
            diagnostic = error.report
            error = error.primary
        primary, cancellation, cancellation_tb = _remember_failure(
            error, primary, cancellation, cancellation_tb
        )

    lifecycle["writer_detach_attempted"] = writer is not None

    def detach_writer() -> None:
        nonlocal callback_count_before_detach, callback_count_after_detach
        if writer is None:
            raise AssertionError("semantic SDK writer detach 대상이 없다")
        callback_count_before_detach = observer.total_callbacks
        observer.window = "detaching"
        try:
            writer.detach()
        finally:
            callback_count_after_detach = observer.total_callbacks
            observer.window = "after_writer_detach_return_before_render_product_cleanup"

    detach_status, detach_error = _cleanup_phase(
        "writer_detach",
        phases,
        phase_order,
        detach_writer if writer is not None else None,
    )
    detach_complete = detach_status == "complete"
    lifecycle["writer_detached"] = detach_complete
    if writer is None:
        callback_count_before_detach = observer.total_callbacks
        callback_count_after_detach = observer.total_callbacks
    if detach_error is not None:
        primary, cancellation, cancellation_tb = _remember_failure(
            detach_error, primary, cancellation, cancellation_tb
        )

    events = diagnostic["events"] if diagnostic is not None else []
    try:
        diagnostic = _phase(
            "diagnostic_finalize",
            phases,
            phase_order,
            lambda: D._build_report(observer, list(events), EVENT_COUNT),
        )
    except BaseException as error:  # noqa: BLE001 - cleanup을 이어가고 원인을 보존한다.
        primary, cancellation, cancellation_tb = _remember_failure(
            error, primary, cancellation, cancellation_tb
        )
        if diagnostic is None:
            diagnostic = {
                "schema": D.SCHEMA,
                "callbacks_observed": observer.total_callbacks,
                "callbacks_preserved": len(observer.callbacks),
                "callbacks": deepcopy(observer.callbacks),
                "no_capture_approved": True,
            }

    if diagnostic is None:
        diagnostic = {
            "schema": D.SCHEMA,
            "callbacks_observed": observer.total_callbacks,
            "callbacks_preserved": len(observer.callbacks),
            "callbacks": deepcopy(observer.callbacks),
            "no_capture_approved": True,
        }
    diagnostic_snapshot: dict[str, Any] = dict(diagnostic)
    diagnostic_snapshot["collection_scope"] = {
        "starts": "before_writer_attach",
        "ends": "after_writer_detach_return_before_render_product_cleanup",
        "callbacks_before_detach_preserved": True,
        "callbacks_during_detach_included": True,
        "complete_drain_proven": False,
        "all_future_callbacks_collected": False,
    }
    diagnostic = diagnostic_snapshot
    setup_evidence = {
        "phase_order": list(phase_order),
        "setup_physics_before_after": deepcopy(setup_physics),
        "extension": deepcopy(extension_evidence),
        "startup": deepcopy(startup_evidence),
        "registrations": deepcopy(registrations),
        "callback_count_before_detach": callback_count_before_detach,
        "callback_count_after_detach": callback_count_after_detach,
        "observation_window": {
            "starts": "before_writer_attach",
            "ends": "after_writer_detach_return_before_render_product_cleanup",
            "detach_callbacks_included": True,
            "complete_drain_proven": False,
        },
    }

    seal_row = {"name": "seal_report", "status": "running", "error_type": None}
    phases.append(seal_row)
    phase_order.append("seal_report")
    sealed = _envelope(
        diagnostic,
        phases,
        lifecycle,
        renderer,
        setup_evidence,
    )
    try:
        seal_report(deepcopy(sealed))
    except BaseException as error:  # noqa: BLE001 - RP와 환경 정리를 계속한다.
        seal_row["status"] = (
            "cancelled" if not isinstance(error, Exception) else "failed"
        )
        seal_row["error_type"] = _type_name(error)
        primary, cancellation, cancellation_tb = _remember_failure(
            error, primary, cancellation, cancellation_tb
        )
    else:
        seal_row["status"] = "complete"
        lifecycle["report_sealed"] = True

    destroy_status, destroy_error = _cleanup_phase(
        "render_product_destroy",
        phases,
        phase_order,
        render_product.destroy if render_product is not None else None,
    )
    lifecycle["render_product_destroyed"] = destroy_status == "complete"
    if destroy_error is not None:
        primary, cancellation, cancellation_tb = _remember_failure(
            destroy_error, primary, cancellation, cancellation_tb
        )
    close_status, close_error = _cleanup_phase(
        "environment_close",
        phases,
        phase_order,
        env.close if env is not None else None,
    )
    lifecycle["environment_closed"] = close_status == "complete"
    if close_error is not None:
        primary, cancellation, cancellation_tb = _remember_failure(
            close_error, primary, cancellation, cancellation_tb
        )

    terminal = _envelope(
        diagnostic,
        phases,
        lifecycle,
        renderer,
        setup_evidence,
    )
    if cancellation is not None:
        cancellation.__dict__["semantic_sdk_envelope"] = terminal
        raise cancellation.with_traceback(cancellation_tb)
    if primary is not None:
        raise SDKDiagnosticFailure(terminal, primary) from primary
    return terminal
