"""Kit의 native update counter로 Writer 없는 AABA pull 경계를 실행한다."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import record_vision_evidence_v1 as R
import vision_annotator_pull_collector_v1 as COLLECTOR
import vision_annotator_pull_validator_v1 as VALIDATOR
import vision_annotator_pull_worker_v1 as V1
import vision_frame_sync_v1 as FRAME
import vision_prepared_sdk_v1 as PREPARED
import vision_semantic_sdk_v1 as SDK

SCHEMA = "vision_annotator_pull_worker_v2"
MODE = V1.MODE


class AnnotatorPullWorkerFailure(RuntimeError):
    def __init__(self, report: dict[str, Any], primary: BaseException) -> None:
        super().__init__(f"annotator pull worker failed: {type(primary).__name__}: {primary}")
        self.report = report
        self.envelope = report
        self.primary = primary


def _default_native_app() -> Any:
    import omni.kit.app  # pyright: ignore[reportMissingImports]

    return omni.kit.app.get_app()


def _native_app_update_counter(
    native_app: Any,
) -> tuple[Callable[[], int], dict[str, Any]]:
    """IApp native counter를 검증하고 계측 출처 receipt를 만든다."""
    get_update_number = getattr(native_app, "get_update_number", None)
    if not callable(get_update_number):
        raise TypeError("Kit app.get_update_number API가 없어 계측을 거부했다")

    def read() -> int:
        value = get_update_number()
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RuntimeError("Kit app.get_update_number가 음이 아닌 정수를 반환하지 않았다")
        return value

    initial = read()
    return read, {
        "source": "omni.kit.app.IApp.get_update_number",
        "method": "app.get_update_number()",
        "initial_value": initial,
        "fail_closed": True,
    }


def _report(
    collection: Mapping[str, Any] | None,
    validation: Mapping[str, Any] | None,
    policy_probe: Mapping[str, Any] | None,
    phases: list[dict[str, Any]],
    lifecycle: Mapping[str, Any],
    setup: Mapping[str, Any],
    primary: BaseException | None,
) -> dict[str, Any]:
    report = V1._report(
        collection,
        validation,
        policy_probe,
        phases,
        lifecycle,
        setup,
        primary,
    )
    report["schema"] = SCHEMA
    return report


def run_worker(
    validated: Mapping[str, Any],
    app: Any,
    *,
    dependencies: Mapping[str, Any] | None = None,
    policy_env_factory: Callable[..., Callable[..., Any]] = V1._default_policy_env,
    native_app_factory: Callable[[], Any] = _default_native_app,
) -> dict[str, Any]:
    """native Kit update counter로 AABA pull 뒤 정책 한 step을 확인한다."""
    job = validated.get("job")
    if not isinstance(job, Mapping) or job.get("mode") != MODE:
        raise ValueError("annotator pull worker mode가 다르다")
    checkpoint = Path(validated.get("checkpoint_path", ""))
    if checkpoint.is_symlink() or not checkpoint.is_file():
        raise ValueError("annotator pull checkpoint가 regular file이 아니다")
    deps = dict(dependencies) if dependencies is not None else PREPARED._load_dependencies()
    if set(deps) != PREPARED._DEPENDENCY_KEYS:
        raise ValueError("annotator pull dependency 표면이 다르다")
    native_app = native_app_factory()
    app_update_count, counter_receipt = _native_app_update_counter(native_app)

    phases: list[dict[str, Any]] = []
    phase_order: list[str] = []
    lifecycle: dict[str, Any] = {
        "annotators_attached": False,
        "annotators_detached": False,
        "render_product_destroyed": False,
        "environment_closed": False,
    }
    setup: dict[str, Any] = {"app_update_counter": counter_receipt}
    state = {"capture_completed": False, "annotators_detached": False}
    probe: dict[str, Any] = {}
    collection: dict[str, Any] | None = None
    validation: dict[str, Any] | None = None
    primary: BaseException | None = None
    env: Any | None = None
    render_product: Any | None = None

    def phase(name: str, call: Callable[[], Any]) -> Any:
        return SDK._phase(name, phases, phase_order, call)

    try:
        profile = phase("profile", deps["profile"])
        if not isinstance(profile, Mapping) or profile != job.get("profile"):
            raise ValueError("annotator pull fixed profile이 job과 다르다")
        camera_profile = profile.get("camera")
        if not isinstance(camera_profile, Mapping):
            raise TypeError("annotator pull profile.camera가 mapping이 아니다")
        if (
            camera_profile.get("delta_time_s") != 0.0
            or camera_profile.get("rt_subframes") != 1
            or camera_profile.get("pause_timeline") is not False
        ):
            raise ValueError("annotator pull camera step 계약이 다르다")
        rep = deps["rep"]
        settings = phase("settings", deps["settings"])
        phase(
            "extension_enable",
            lambda: deps["enable_extension"](SDK.EXTENSION) is True
            or (_ for _ in ()).throw(RuntimeError("isaacsim.core.nodes 활성화 실패")),
        )
        phase("extension_app_update", app.update)
        setup["app_update_counter"]["after_extension_update"] = app_update_count()
        setup["startup"] = phase(
            "replicator_startup",
            lambda: SDK._startup_readback(deps["startup"](rep, settings)),
        )
        setup["registrations"] = phase(
            "annotator_registrations",
            lambda: SDK._registration_readback(
                deps["registration_probe"](), {name: False for name in SDK.ANNOTATORS}
            ),
        )
        context, stage_events = phase("usd_context", deps["usd_context"])
        env_cfg = phase("environment_config", lambda: SDK._configure_env(profile, deps))
        make_env = policy_env_factory(validated, deps["make_env"], state, probe)
        env = phase("environment_create", lambda: make_env(profile["task"], env_cfg))
        if env is None:
            raise RuntimeError("annotator pull environment가 없다")
        active_base = env.unwrapped
        phase("environment_reset", lambda: active_base.reset(seed=profile["seed"]))
        command = active_base.command_manager.get_command("base_velocity")
        command[0, :3] = command.new_tensor([1.0, 0.0, 0.0])
        physics_before = deepcopy(deps["physics_snapshot"](active_base))
        stage = context.get_stage()
        if stage is None:
            raise RuntimeError("annotator pull USD stage가 없다")
        camera = deps["camera"](stage.GetPrimAtPath(camera_profile["camera_prim_path"]))
        phase("camera_configure", lambda: SDK._configure_camera(camera, camera_profile))
        render_product = phase(
            "render_product_create",
            lambda: rep.create.render_product(
                camera_profile["camera_prim_path"], (R.WIDTH, R.HEIGHT)
            ),
        )
        root = phase("marker_root_position", lambda: deps["root_position"](active_base))
        marker = phase("marker_create", lambda: deps["marker"](rep, root))
        if marker is None:
            raise RuntimeError("annotator pull marker가 없다")
        policy = None
        if deps["fxaa_required"](job):
            policy = phase("fxaa_apply", lambda: deps["fxaa_apply"](rep, settings))
        renderer = phase(
            "renderer_readback",
            lambda: dict(deps["renderer_readback"](context, camera_profile, settings)),
        )
        if policy is not None:
            renderer["render_policy"] = policy
        phase(
            "capture_readiness",
            lambda: deps["readiness"](
                active_base,
                app,
                context,
                stage_events,
                Path(validated["job_path"]).with_suffix(".annotator-readiness.json"),
                renderer,
            ),
        )
        if policy is not None:
            phase(
                "fxaa_verify",
                lambda: deps["fxaa_verify"](deps["fxaa_readback"](settings)),
            )
        phase("marker_bind", lambda: marker.bind_and_validate(stage))
        physics_after = deepcopy(deps["physics_snapshot"](active_base))
        if physics_before != physics_after:
            raise RuntimeError("annotator pull setup이 physics 상태를 바꿨다")

        def prepare_arm(ordinal: int, camera_kind: str, marker_code: int) -> dict[str, Any]:
            marker.set_code(stage, marker_code)
            eye_offset = list(camera_profile["eye_offset_m"])
            if camera_kind == "B":
                eye_offset[0] = -eye_offset[0]
            eye = [root[index] + eye_offset[index] for index in range(3)]
            lookat = [
                root[index] + camera_profile["lookat_offset_m"][index]
                for index in range(3)
            ]
            active_base.sim.set_camera_view(
                eye=eye,
                target=lookat,
                camera_prim_path=camera_profile["camera_prim_path"],
            )
            context.reset_renderer_accumulation()
            return {
                "ordinal": ordinal,
                "camera_kind": camera_kind,
                "marker_code": marker_code,
                "marker_labels": sorted(FRAME.marker_labels(marker_code)),
                "marker": {"usd_visibility_written": True},
                "camera": {
                    "eye_world_m": eye,
                    "lookat_world_m": lookat,
                    "resolution": [R.WIDTH, R.HEIGHT],
                },
            }

        collected = phase(
            "annotator_pull_aaba",
            lambda: COLLECTOR.collect_aaba(
                rep=rep,
                render_product=render_product,
                orchestrator=rep.orchestrator,
                camera_profile=camera_profile,
                active_base=active_base,
                physics_snapshot=lambda: deps["physics_snapshot"](active_base),
                prepare_arm=prepare_arm,
                app_update_count=app_update_count,
            ),
        )
        if not isinstance(collected, dict):
            raise TypeError("annotator pull collection 결과가 dict가 아니다")
        collection = collected
        attach_count = sum(
            row.get("operation") == "attach" for row in collection["operation_trace"]
        )
        detach_count = sum(
            row.get("operation") == "detach" for row in collection["operation_trace"]
        )
        lifecycle["annotators_attached"] = attach_count == len(COLLECTOR.ANNOTATORS)
        lifecycle["annotators_detached"] = (
            detach_count == len(COLLECTOR.ANNOTATORS)
            and not collection["cleanup_errors"]
        )
        state["annotators_detached"] = lifecycle["annotators_detached"]
        checked = phase("annotator_pull_validate", lambda: VALIDATOR.validate(collected))
        if not isinstance(checked, dict):
            raise TypeError("annotator pull validation 결과가 dict가 아니다")
        validation = checked
        state["capture_completed"] = validation.get("approved") is True
    except BaseException as error:  # noqa: BLE001 - cleanup을 계속한다.
        primary = getattr(error, "primary", error)
        raw_report = getattr(error, "report", None)
        if collection is None and isinstance(raw_report, Mapping):
            collection = dict(raw_report)

    destroy_status, destroy_error = SDK._cleanup_phase(
        "render_product_destroy",
        phases,
        phase_order,
        render_product.destroy if render_product is not None else None,
    )
    lifecycle["render_product_destroyed"] = destroy_status == "complete"
    if primary is None and destroy_error is not None:
        primary = destroy_error
    close_status, close_error = SDK._cleanup_phase(
        "environment_close", phases, phase_order, env.close if env is not None else None
    )
    lifecycle["environment_closed"] = close_status == "complete"
    if primary is None and close_error is not None:
        primary = close_error
    report = _report(collection, validation, probe or None, phases, lifecycle, setup, primary)
    if primary is not None:
        raise AnnotatorPullWorkerFailure(report, primary) from primary
    if report["diagnostic_complete"] is not True:
        error = RuntimeError("annotator pull worker 진단이 완결되지 않았다")
        raise AnnotatorPullWorkerFailure(report, error) from error
    return report
