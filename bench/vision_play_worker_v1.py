"""검증된 V04 job으로 Go2 Play 영상과 같은 상태의 수치 trace를 기록한다.

Isaac 및 GPU 모듈은 ``run_real`` 안에서만 import한다. 부모 실행기는 GPU 입장,
프로세스 그룹, 외부 timeout을 소유한다.
"""
from __future__ import annotations

import argparse
import sys
import traceback
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import record_vision_evidence_v1 as R

CameraCapture = Callable[
    [Any, Any, list[Any], Mapping[str, Any], list[float], list[float], int],
    dict[str, Any],
]


def _tensor_values(value: Any) -> list[float]:
    return [float(item) for item in value.detach().cpu().flatten().tolist()]


def _capture_render_step(orchestrator: Any, camera_profile: Mapping[str, Any]) -> None:
    """Physics 시각을 진행하거나 timeline을 멈추지 않고 한 RGB 갱신을 기다린다."""
    orchestrator.step(
        rt_subframes=camera_profile["rt_subframes"],
        pause_timeline=camera_profile["pause_timeline"],
        delta_time=camera_profile["delta_time_s"],
    )


def _create_capture_annotators(rep: Any, render_product: Any) -> list[Any]:
    annotators = [
        rep.AnnotatorRegistry.get_annotator("rgb"),
        rep.AnnotatorRegistry.get_annotator("IsaacReadTimes"),
        rep.annotators.get("ReferenceTime"),
    ]
    for annotator in annotators:
        annotator.attach(render_product)
    return annotators


def _capture_pixels_and_timing(
    rep: Any, annotators: list[Any], camera_profile: Mapping[str, Any]
) -> dict[str, Any]:
    if len(annotators) != 3:
        raise ValueError("capture annotator 묶음이 고정 profile과 다르다")
    _capture_render_step(rep.orchestrator, camera_profile)
    rgb_array = annotators[0].get_data(device="cpu", do_array_copy=True)
    if rgb_array is None:
        raise RuntimeError("Replicator RGB annotator가 frame을 반환하지 않았다")
    read_times = annotators[1].get_data(device="cpu", do_array_copy=True)
    reference_time = annotators[2].get_data(device="cpu", do_array_copy=True)
    if not isinstance(read_times, Mapping) or not isinstance(reference_time, Mapping):
        raise TypeError("Replicator timestamp annotator가 mapping을 반환하지 않았다")
    return {
        "rgb": R.rgb24_bytes(rgb_array),
        "read_times": dict(read_times),
        "reference_time": dict(reference_time),
    }


def _set_camera_and_capture(
    base: Any,
    rep: Any,
    annotators: list[Any],
    camera_profile: Mapping[str, Any],
    eye: list[float],
    lookat: list[float],
) -> dict[str, Any]:
    """record와 smoke가 공유하는 camera pose·render·annotator 경로다."""
    base.sim.set_camera_view(
        eye=eye, target=lookat, camera_prim_path=camera_profile["camera_prim_path"]
    )
    return _capture_pixels_and_timing(rep, annotators, camera_profile)


def _capture_with_ordinal(
    capture: CameraCapture | None, base: Any, rep: Any, annotators: list[Any],
    camera_profile: Mapping[str, Any], eye: list[float], lookat: list[float], ordinal: int,
) -> dict[str, Any]:
    if capture is not None:
        return capture(base, rep, annotators, camera_profile, eye, lookat, ordinal)
    return _set_camera_and_capture(base, rep, annotators, camera_profile, eye, lookat)


def _cleanup_capture_resources(
    encoder: Any, render_product: Any, annotators: list[Any], env: Any,
    primary_error: BaseException | None,
) -> None:
    errors: list[BaseException] = []
    if encoder is not None:
        try:
            if not encoder.finished:
                encoder.abort()
        except BaseException as error:  # noqa: BLE001 - 나머지 SDK 자원도 정리한다.
            errors.append(error)
    if render_product is not None:
        for annotator in annotators:
            try:
                annotator.detach(render_product)
            except BaseException as error:  # noqa: BLE001 - destroy와 env.close를 이어간다.
                errors.append(error)
        try:
            render_product.destroy()
        except BaseException as error:  # noqa: BLE001 - env.close를 이어간다.
            errors.append(error)
    if env is not None:
        try:
            env.close()
        except BaseException as error:  # noqa: BLE001 - 첫 실패를 보존한다.
            errors.append(error)
    if primary_error is not None:
        for error in errors:
            traceback.print_exception(error)
    elif errors:
        raise errors[0]


def run_real(
    validated: Mapping[str, Any], app: Any, *, camera_capture: CameraCapture | None = None,
) -> tuple[dict[str, Path], Path]:
    import gymnasium as gym  # pyright: ignore[reportMissingImports]
    import isaaclab_tasks  # noqa: F401  # pyright: ignore[reportMissingImports]
    import omni.replicator.core as rep  # pyright: ignore[reportMissingImports]
    import omni.usd  # pyright: ignore[reportMissingImports]
    import torch  # pyright: ignore[reportMissingImports]
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # pyright: ignore[reportMissingImports]
    from isaaclab_tasks.utils import (  # pyright: ignore[reportMissingImports]
        load_cfg_from_registry,
        parse_env_cfg,
    )
    from pxr import UsdGeom  # pyright: ignore[reportMissingImports]
    from rsl_rl.runners import OnPolicyRunner  # pyright: ignore[reportMissingImports]

    profile = R.fixed_profile()
    env_cfg = parse_env_cfg(profile["task"], device="cuda:0", num_envs=1)
    env_cfg.seed = profile["seed"]
    env_cfg.actions.joint_pos.scale = profile["action_scale"]
    command_cfg = env_cfg.commands.base_velocity
    command_cfg.heading_command = False
    command_cfg.rel_standing_envs = 0.0
    command_cfg.resampling_time_range = tuple(profile["command"]["resampling_time_range_s"])
    command_cfg.ranges.lin_vel_x = (1.0, 1.0)
    command_cfg.ranges.lin_vel_y = (0.0, 0.0)
    command_cfg.ranges.ang_vel_z = (0.0, 0.0)
    step_dt = float(env_cfg.sim.dt * env_cfg.decimation)
    if abs(step_dt - R.FRAME_PERIOD_S) > 1e-12:
        raise ValueError("Go2 Play control step이 고정 50fps time basis와 다르다")
    agent_cfg = load_cfg_from_registry(profile["task"], "rsl_rl_cfg_entry_point")
    agent_cfg.device = "cuda:0"
    del app
    env = None
    render_product = None
    annotators: list[Any] = []
    encoder = None
    output = validated["output_directory"]
    video = output / "same_play.mp4"
    try:
        env = RslRlVecEnvWrapper(
            gym.make(profile["task"], cfg=env_cfg), clip_actions=agent_cfg.clip_actions
        )
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(str(validated["checkpoint_path"]), load_optimizer=False)
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        base = env.unwrapped
        robot = base.scene["robot"]
        output.mkdir(parents=True, exist_ok=False)
        encoder = R.RawRgbFfmpegEncoder(video)
        base.reset(seed=profile["seed"])
        command_buffer = base.command_manager.get_command("base_velocity")
        fixed_command = torch.tensor([[1.0, 0.0, 0.0]], device=base.device)
        command_buffer.copy_(fixed_command)
        camera_profile = profile["camera"]
        camera_prim = omni.usd.get_context().get_stage().GetPrimAtPath(camera_profile["camera_prim_path"])
        camera = UsdGeom.Camera(camera_prim)
        camera.GetProjectionAttr().Set(camera_profile["projection"])
        camera.GetHorizontalApertureAttr().Set(camera_profile["horizontal_aperture_mm"])
        camera.GetFocalLengthAttr().Set(camera_profile["focal_length_mm"])
        camera.GetClippingRangeAttr().Set(tuple(camera_profile["clipping_range_m"]))
        render_product = rep.create.render_product(
            camera_profile["camera_prim_path"], (R.WIDTH, R.HEIGHT)
        )
        annotators = _create_capture_annotators(rep, render_product)
        obs, _ = env.get_observations()

        def state_snapshot() -> dict[str, Any]:
            return {
                "command": _tensor_values(command_buffer[0, :3]),
                "motion": [*_tensor_values(robot.data.root_lin_vel_b[0, :2]),
                           float(robot.data.root_ang_vel_b[0, 2].detach().cpu())],
                "pose": {
                    "base_height_m": float(robot.data.root_pos_w[0, 2].detach().cpu()),
                    "up_axis_z": -float(robot.data.projected_gravity_b[0, 2].detach().cpu()),
                },
                "root_position_world_m": _tensor_values(robot.data.root_pos_w[0, :3]),
                "simulation_time_s": float(base.sim.current_time),
                "common_step_counter": int(base.common_step_counter),
            }

        def atomic_sample(index: int) -> dict[str, Any]:
            nonlocal obs
            if index:
                command_buffer.copy_(fixed_command)
                with torch.no_grad():
                    actions = policy(obs)
                obs, _, _, _ = env.step(actions)
                command_buffer.copy_(fixed_command)
            before = state_snapshot()
            root = before["root_position_world_m"]
            eye = [root[i] + [3.0, 3.0, 2.0][i] for i in range(3)]
            lookat = [root[i] + [0.0, 0.0, 0.35][i] for i in range(3)]
            capture = _capture_with_ordinal(
                camera_capture, base, rep, annotators, camera_profile, eye, lookat, index
            )
            after = state_snapshot()
            R.validate_render_state_unchanged(before, after)
            return {
                "frame_index": index,
                "capture_ordinal": index,
                "simulation_step": after["common_step_counter"],
                "rgb": capture["rgb"],
                "state": after,
                "camera": {"eye_world_m": eye, "lookat_world_m": lookat},
                "render_timing": {
                    "read_times": capture["read_times"],
                    "reference_time": capture["reference_time"],
                },
            }

        records, encoder_receipt = R.capture_lockstep(atomic_sample, encoder)
        timing_receipt = R.probe_video_timing(video)
        pts = timing_receipt["relative_actual_pts_s"]
        bound = R.bind_actual_pts(records, pts)
        artifacts = R.build_trace_and_v01_specs(
            validated, video, bound, encoder_receipt, timing_receipt, output / "evidence"
        )
        return artifacts, video
    finally:
        _cleanup_capture_resources(
            encoder, render_product, annotators, env, sys.exc_info()[1]
        )


def run_render_sync_smoke(
    validated: Mapping[str, Any], app: Any, *, camera_capture: CameraCapture | None = None,
) -> tuple[dict[str, Path], Path]:
    """정책·physics step 없이 현재 renderer의 A-B-A capture 동기화를 검사한다."""
    import gymnasium as gym  # pyright: ignore[reportMissingImports]
    import isaaclab_tasks  # noqa: F401  # pyright: ignore[reportMissingImports]
    import omni.replicator.core as rep  # pyright: ignore[reportMissingImports]
    import omni.usd  # pyright: ignore[reportMissingImports]
    from isaaclab_tasks.utils import parse_env_cfg  # pyright: ignore[reportMissingImports]
    from pxr import UsdGeom  # pyright: ignore[reportMissingImports]

    del app
    profile = R.fixed_profile()
    env_cfg = parse_env_cfg(profile["task"], device="cuda:0", num_envs=1)
    env_cfg.seed = profile["seed"]
    env_cfg.actions.joint_pos.scale = profile["action_scale"]
    command_cfg = env_cfg.commands.base_velocity
    command_cfg.heading_command = False
    command_cfg.rel_standing_envs = 0.0
    command_cfg.resampling_time_range = tuple(profile["command"]["resampling_time_range_s"])
    command_cfg.ranges.lin_vel_x = (1.0, 1.0)
    command_cfg.ranges.lin_vel_y = (0.0, 0.0)
    command_cfg.ranges.ang_vel_z = (0.0, 0.0)
    env = None
    render_product = None
    annotators: list[Any] = []
    capture_ordinal = 0
    try:
        env = gym.make(profile["task"], cfg=env_cfg)
        base = env.unwrapped
        base.reset(seed=profile["seed"])
        robot = base.scene["robot"]
        command_buffer = base.command_manager.get_command("base_velocity")
        command_buffer[0, :3] = command_buffer.new_tensor([1.0, 0.0, 0.0])
        camera_profile = profile["camera"]
        camera_prim = omni.usd.get_context().get_stage().GetPrimAtPath(camera_profile["camera_prim_path"])
        camera = UsdGeom.Camera(camera_prim)
        camera.GetProjectionAttr().Set(camera_profile["projection"])
        camera.GetHorizontalApertureAttr().Set(camera_profile["horizontal_aperture_mm"])
        camera.GetFocalLengthAttr().Set(camera_profile["focal_length_mm"])
        camera.GetClippingRangeAttr().Set(tuple(camera_profile["clipping_range_m"]))
        render_product = rep.create.render_product(
            camera_profile["camera_prim_path"], (R.WIDTH, R.HEIGHT)
        )
        annotators = _create_capture_annotators(rep, render_product)

        def state_snapshot() -> dict[str, Any]:
            return {
                "command": _tensor_values(command_buffer[0, :3]),
                "motion": [*_tensor_values(robot.data.root_lin_vel_b[0, :2]),
                           float(robot.data.root_ang_vel_b[0, 2].detach().cpu())],
                "pose": {"base_height_m": float(robot.data.root_pos_w[0, 2].detach().cpu()),
                         "up_axis_z": -float(robot.data.projected_gravity_b[0, 2].detach().cpu())},
                "root_position_world_m": _tensor_values(robot.data.root_pos_w[0, :3]),
                "common_step_counter": int(base.common_step_counter),
            }

        def capture(label: str, eye_offset: list[float]) -> dict[str, Any]:
            nonlocal capture_ordinal
            before = state_snapshot()
            root = before["root_position_world_m"]
            eye = [root[index] + eye_offset[index] for index in range(3)]
            lookat_offset = camera_profile["lookat_offset_m"]
            lookat = [root[index] + lookat_offset[index] for index in range(3)]
            simulation_time_before = float(base.sim.current_time)
            rendered = _capture_with_ordinal(
                camera_capture, base, rep, annotators, camera_profile, eye, lookat,
                capture_ordinal,
            )
            capture_ordinal += 1
            after = state_snapshot()
            return {
                "label": label, "rgb": rendered["rgb"], "state_before": before, "state_after": after,
                "simulation_time_before_s": simulation_time_before,
                "simulation_time_after_s": float(base.sim.current_time),
                "read_times": rendered["read_times"], "reference_time": rendered["reference_time"],
                "camera": {"eye_world_m": eye, "lookat_world_m": lookat},
            }

        captures, report = R.collect_render_sync_smoke(capture)
        output = validated["output_directory"]
        output.mkdir(parents=True, exist_ok=False)
        report_path = output / "render_sync_smoke.json"
        R.write_new(report_path, report)
        artifacts: dict[str, Path] = {"render_sync_smoke": report_path}
        for index, capture_row in enumerate(captures[1:], 1):
            rgb_path = output / f"audit_{index}_{capture_row['label']}.rgb"
            rgb_path.write_bytes(capture_row["rgb"])
            artifacts[f"audit_rgb_{index}"] = rgb_path
        return artifacts, report_path
    finally:
        _cleanup_capture_resources(
            None, render_product, annotators, env, sys.exc_info()[1]
        )


def execute(
    job_path: Path,
    *,
    app_factory: Callable[[], Any] | None = None,
    runner: Callable[[Mapping[str, Any], Any], tuple[dict[str, Path], Path]] | None = None,
) -> int:
    validated: dict[str, Any] | None = None
    app = None
    stage = "validate_job"
    try:
        validated = R.validate_job(job_path)
        stage = "launch_app"
        if app_factory is None:
            from isaaclab.app import AppLauncher  # pyright: ignore[reportMissingImports]
            app_factory = lambda: AppLauncher(
                headless=True, enable_cameras=True, fast_shutdown=False
            ).app
        app = app_factory()
        stage = "record_play"
        selected_runner = runner
        if selected_runner is None:
            selected_runner = (
                run_render_sync_smoke
                if validated["job"]["mode"] == R.RENDER_SYNC_SMOKE_MODE
                else run_real
            )
        artifacts, video = selected_runner(validated, app)
        stage = "write_result_before_app_close"
        R.write_new(validated["output_directory"] / "result.json", R.result_body(validated, artifacts, video))
    except BaseException as error:  # noqa: BLE001 - SDK 종료 전에 모든 실행 실패를 봉인한다.
        failure_root = (
            validated["output_directory"] if validated is not None
            else Path(job_path).resolve().parent / f"{Path(job_path).stem}.failure"
        )
        failure_root.mkdir(parents=True, exist_ok=True)
        failure_path = failure_root / "failure.json"
        if not failure_path.exists():
            R.write_failure(
                failure_path, stage=stage, error=error,
                job_sha256=None if validated is None else validated["job_sha256"],
            )
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        if app is not None:
            try:
                app.close()
            except BaseException:  # noqa: BLE001 - 원래 실행 실패를 보존한다.
                traceback.print_exc()
        return 1
    try:
        app.close()
    except BaseException as error:  # noqa: BLE001 - SDK 종료 실패도 nonzero 증거로 남긴다.
        failure_path = validated["output_directory"] / "shutdown_failure.json"
        R.write_failure(failure_path, stage="app_close", error=error, job_sha256=validated["job_sha256"])
        traceback.print_exc()
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True, type=Path)
    args = parser.parse_args(argv)
    return execute(args.job)


if __name__ == "__main__":
    raise SystemExit(main())
