"""실제 Isaac 환경에서 Writer 없이 AABA annotator pull 경계를 실행한다."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import record_vision_evidence_v1 as R
import vision_annotator_pull_collector_v1 as COLLECTOR
import vision_annotator_pull_validator_v1 as VALIDATOR
import vision_checkpoint_smoke_sdk_v2 as POLICY
import vision_frame_sync_v1 as FRAME
import vision_prepared_sdk_v1 as PREPARED
import vision_semantic_sdk_v1 as SDK

SCHEMA = "vision_annotator_pull_worker_v1"
MODE = "checkpoint_annotator_pull_diagnostic_only"


class AnnotatorPullWorkerFailure(RuntimeError):
    def __init__(self, report: dict[str, Any], primary: BaseException) -> None:
        super().__init__(f"annotator pull worker failed: {type(primary).__name__}: {primary}")
        self.report = report
        self.envelope = report
        self.primary = primary


class _PolicyEnv:
    def __init__(
        self,
        env: Any,
        policy: Callable[[Any], Any],
        torch: Any,
        checkpoint_sha256: str,
        state: dict[str, bool],
        probe: dict[str, Any],
    ) -> None:
        self._env = env
        self._policy = policy
        self._torch = torch
        self._checkpoint_sha256 = checkpoint_sha256
        self._state = state
        self._probe = probe
        self._closed = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    def close(self) -> None:
        if self._closed:
            raise RuntimeError("annotator pull 환경 close가 두 번 호출됐다")
        self._closed = True
        primary: BaseException | None = None
        try:
            if self._state["capture_completed"]:
                if not self._state["annotators_detached"]:
                    raise RuntimeError("annotator detach 전에 policy step을 거부했다")
                base = self._env.unwrapped
                command = base.command_manager.get_command("base_velocity")
                command[0, :3] = command.new_tensor([1.0, 0.0, 0.0])
                observation, _ = self._env.get_observations()
                before = int(base.common_step_counter)
                with self._torch.no_grad():
                    action = self._policy(observation)
                self._env.step(action)
                after = int(base.common_step_counter)
                if after != before + 1:
                    raise RuntimeError("annotator pull policy step 수가 1이 아니다")
                self._probe.update(
                    {
                        "schema": "vision_annotator_pull_policy_probe_v1",
                        "checkpoint_sha256": self._checkpoint_sha256,
                        "command": [1.0, 0.0, 0.0],
                        "observation": POLICY._tensor_summary(observation, self._torch),
                        "action": POLICY._tensor_summary(action, self._torch),
                        "environment_steps": 1,
                        "step_counter_before": before,
                        "step_counter_after": after,
                        "capture_completed_before_policy_step": True,
                        "annotators_detached_before_policy_step": True,
                        "actual_environment_observation_used": True,
                    }
                )
        except BaseException as error:  # noqa: BLE001 - env close 뒤 첫 오류를 재발생시킨다.
            primary = error
        try:
            self._env.close()
        except BaseException as close_error:
            if primary is None:
                raise
            POLICY._remember_cleanup(primary, close_error)
        if primary is not None:
            raise primary


def _default_policy_env(
    validated: Mapping[str, Any],
    make_env: Callable[..., Any],
    state: dict[str, bool],
    probe: dict[str, Any],
) -> Callable[..., Any]:
    def create(task: str, env_cfg: Any) -> Any:
        import torch  # pyright: ignore[reportMissingImports]
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # pyright: ignore[reportMissingImports]
        from isaaclab_tasks.utils import load_cfg_from_registry  # pyright: ignore[reportMissingImports]
        from rsl_rl.runners import OnPolicyRunner  # pyright: ignore[reportMissingImports]

        raw = make_env(task, env_cfg)
        selected: Any | None = None
        try:
            cfg = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
            cfg.device = "cuda:0"
            selected = RslRlVecEnvWrapper(raw, clip_actions=cfg.clip_actions)
            if selected is None:
                raise RuntimeError("annotator pull policy wrapper 환경이 없다")
            runner = OnPolicyRunner(
                selected, cfg.to_dict(), log_dir=None, device=cfg.device
            )
            runner.load(str(validated["checkpoint_path"]), load_optimizer=False)
            policy = runner.get_inference_policy(device=selected.unwrapped.device)
            return _PolicyEnv(
                selected,
                policy,
                torch,
                str(validated["checkpoint_sha256"]),
                state,
                probe,
            )
        except BaseException:
            (selected if selected is not None else raw).close()
            raise

    return create


def _report(
    collection: Mapping[str, Any] | None,
    validation: Mapping[str, Any] | None,
    policy_probe: Mapping[str, Any] | None,
    phases: list[dict[str, Any]],
    lifecycle: Mapping[str, Any],
    setup: Mapping[str, Any],
    primary: BaseException | None,
) -> dict[str, Any]:
    errors = [row for row in phases if row["status"] in {"failed", "cancelled"}]
    complete = (
        primary is None
        and validation is not None
        and validation.get("approved") is True
        and policy_probe is not None
        and policy_probe.get("environment_steps") == 1
        and policy_probe.get("annotators_detached_before_policy_step") is True
        and lifecycle.get("annotators_attached") is True
        and lifecycle.get("annotators_detached") is True
        and lifecycle.get("render_product_destroyed") is True
        and lifecycle.get("environment_closed") is True
        and not errors
    )
    return {
        "schema": SCHEMA,
        "collection": deepcopy(dict(collection)) if collection is not None else None,
        "validation": deepcopy(dict(validation)) if validation is not None else None,
        "policy_probe": deepcopy(dict(policy_probe)) if policy_probe is not None else None,
        "phases": deepcopy(phases),
        "lifecycle": deepcopy(dict(lifecycle)),
        "setup": deepcopy(dict(setup)),
        "writer_calls": {"create": 0, "attach": 0, "callback": 0, "detach": 0},
        "diagnostic_complete": complete,
    }


def run_worker(
    validated: Mapping[str, Any],
    app: Any,
    *,
    dependencies: Mapping[str, Any] | None = None,
    policy_env_factory: Callable[..., Callable[..., Any]] = _default_policy_env,
) -> dict[str, Any]:
    """실제 환경/RP/marker를 만들고 AABA pull 뒤 정책 한 step을 확인한다."""
    job = validated.get("job")
    if not isinstance(job, Mapping) or job.get("mode") != MODE:
        raise ValueError("annotator pull worker mode가 다르다")
    checkpoint = Path(validated.get("checkpoint_path", ""))
    if checkpoint.is_symlink() or not checkpoint.is_file():
        raise ValueError("annotator pull checkpoint가 regular file이 아니다")
    deps = dict(dependencies) if dependencies is not None else PREPARED._load_dependencies()
    if set(deps) != PREPARED._DEPENDENCY_KEYS:
        raise ValueError("annotator pull dependency 표면이 다르다")

    phases: list[dict[str, Any]] = []
    phase_order: list[str] = []
    lifecycle: dict[str, Any] = {
        "annotators_attached": False,
        "annotators_detached": False,
        "render_product_destroyed": False,
        "environment_closed": False,
    }
    setup: dict[str, Any] = {"explicit_app_updates": 0}
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
        setup["explicit_app_updates"] += 1
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
                app_update_count=lambda: int(setup["explicit_app_updates"]),
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
