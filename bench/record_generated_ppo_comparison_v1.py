# pyright: reportMissingImports=false
"""seed 5750 PPO 결함/생성 패치 정책을 같은 flat 과제에서 15초 녹화한다."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, TextIO

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "bench")]

SCHEMA = "generated_ppo_comparison_batch_v1"
MANIFEST_SCHEMA = "generated_ppo_comparison_manifest_v1"
ENCODER_SCHEMA = "generated_ppo_comparison_exact_frames_v1"
SHUTDOWN_SCHEMA = "generated_ppo_comparison_shutdown_v1"
TASK = "Isaac-Velocity-Flat-Unitree-Go2-v0"
SEED = 5750
ARMS = ("fault", "patched")
COMMAND = (0.25, 0.0, 0.0)
FRAMES = 750
FPS = 50
RESOLUTION = (2560, 1440)
EYE = (3.5, 6.0, 2.2)
LOOKAT = (0.0, 0.0, 0.5)
ACTION_SCALE = 0.25
EXPECTED_METRICS = {
    "fault": {"lin_vel_rmse_mps": 1.1730560454811372, "fall_rate": 0.0},
    "patched": {"lin_vel_rmse_mps": 0.14578562336420092, "fall_rate": 0.0},
}
SOURCE_FILES = (
    "bench/record_generated_ppo_comparison_v1.py",
    "tests/test_record_generated_ppo_comparison_v1.py",
    "evals/lineage_lease_context_v2.py",
    "evals/lineage_lease_context_v3.py",
    "evals/lineage_source_batch_v1.py",
    "bench/protocols/fixed_eval_v1.json",
    "bench/protocols/generated_ppo_patch_behavior_s5750_20261106_v3.json",
)


def sha(path: Path, *, lf: bool = False) -> str:
    raw = Path(path).read_bytes()
    return hashlib.sha256(raw.replace(b"\r\n", b"\n") if lf else raw).hexdigest()


def read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON object가 아니다: {path}")
    return value


def write_new(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")


def _source_closure(repo: Path) -> dict[str, str]:
    observed: dict[str, str] = {}
    for relative in SOURCE_FILES:
        path = repo / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        observed[relative] = sha(path, lf=True)
    return observed


def build_batch(repo: Path, run_root: Path) -> dict[str, Any]:
    """기존 실제 checkpoint/fixed26 결과를 새 녹화 입력으로 결합한다."""
    repo, run_root = Path(repo).resolve(), Path(run_root).resolve()
    if repo != ROOT.resolve():
        raise ValueError("실행기 저장소 root가 다르다")
    public = read(repo / "evals/results/ppo_generated_patch_behavior_s5750_20261106_v1/summary.json")
    if public.get("status") != "complete" or public.get("outcome") != "recovered":
        raise ValueError("seed 5750 공개 행동 결과가 complete/recovered가 아니다")
    fixed_jobs = read(run_root / "fixed-jobs.json")
    if fixed_jobs.get("action_scale") != ACTION_SCALE:
        raise ValueError("고정 평가 action scale이 다르다")
    job_by_name = {item["name"]: item for item in fixed_jobs.get("jobs", [])}
    if set(job_by_name) != {"healthy", "fault", "patched"}:
        raise ValueError("기존 fixed26 작업 집합이 다르다")
    policies = []
    for arm in ARMS:
        job = job_by_name[arm]
        checkpoint = Path(job["checkpoint"]).resolve()
        fixed_report = Path(job["output"]).resolve()
        training_result = run_root / arm / "training/training-result.json"
        for path in (checkpoint, fixed_report, training_result):
            if not path.is_file():
                raise FileNotFoundError(path)
        if checkpoint != (run_root / arm / "training/final.pt").resolve():
            raise ValueError(f"{arm} 체크포인트 계보가 다르다")
        if sha(checkpoint) != job["checkpoint_sha256"]:
            raise ValueError(f"{arm} 체크포인트 SHA가 다르다")
        report = read(fixed_report)
        training = read(training_result)
        metrics = report.get("metrics")
        if report.get("name") != arm or report.get("checkpoint", {}).get("sha256") != sha(checkpoint):
            raise ValueError(f"{arm} fixed26 report 결합이 다르다")
        if metrics is None or any(metrics.get(key) != value for key, value in EXPECTED_METRICS[arm].items()):
            raise ValueError(f"{arm} fixed26 기준 수치가 다르다")
        if (training.get("status") != "complete" or training.get("seed") != SEED
                or training.get("task") != TASK or training.get("checkpoint_sha256") != sha(checkpoint)):
            raise ValueError(f"{arm} 실제 학습 영수증 결합이 다르다")
        policies.append({
            "arm": arm,
            "checkpoint": {"path": str(checkpoint), "sha256": sha(checkpoint)},
            "training_result": {"path": str(training_result.resolve()), "sha256": sha(training_result)},
            "fixed_report": {"path": str(fixed_report), "sha256": sha(fixed_report)},
            "fixed_metrics": {key: metrics[key] for key in (
                "fall_rate", "mean_survival_s", "lin_vel_rmse_mps", "yaw_rate_rmse_radps"
            )},
        })
    if policies[1]["checkpoint"]["sha256"] != job_by_name["healthy"]["checkpoint_sha256"]:
        raise ValueError("패치 정책이 정상 대조 checkpoint와 일치하지 않는다")
    return {
        "schema": SCHEMA,
        "development_case": True,
        "seed": SEED,
        "task": TASK,
        "command": list(COMMAND),
        "action_scale": ACTION_SCALE,
        "capture": {
            "duration_s": FRAMES / FPS,
            "frames": FRAMES,
            "fps": FPS,
            "resolution": list(RESOLUTION),
            "camera": {"eye": list(EYE), "lookat": list(LOOKAT), "origin_type": "asset_root"},
            "same_native_duration": True,
            "new_training": 0,
            "model_api": 0,
        },
        "source_sha256_lf": _source_closure(repo),
        "policies": policies,
        "claim_boundary": {
            "fault": "fixed26_tracking_failure",
            "patched": "fixed26_healthy",
            "visual_motion_label": "pending_observation",
            "fall_claim": False,
        },
    }


def validate_batch(batch: dict[str, Any], repo: Path = ROOT) -> dict[str, Any]:
    expected_top = {
        "schema", "development_case", "seed", "task", "command", "action_scale",
        "capture", "source_sha256_lf", "policies", "claim_boundary",
    }
    if set(batch) != expected_top or batch["schema"] != SCHEMA:
        raise ValueError("녹화 batch schema가 다르다")
    if (batch["development_case"] is not True or batch["seed"] != SEED or batch["task"] != TASK
            or batch["command"] != list(COMMAND) or batch["action_scale"] != ACTION_SCALE):
        raise ValueError("seed/task/명령/action scale이 등록값과 다르다")
    expected_capture = {
        "duration_s": 15.0, "frames": FRAMES, "fps": FPS, "resolution": list(RESOLUTION),
        "camera": {"eye": list(EYE), "lookat": list(LOOKAT), "origin_type": "asset_root"},
        "same_native_duration": True, "new_training": 0, "model_api": 0,
    }
    if batch["capture"] != expected_capture:
        raise ValueError("15초 HQ capture 계약이 다르다")
    if batch["source_sha256_lf"] != _source_closure(Path(repo).resolve()):
        raise ValueError("source closure가 현재 파일과 다르다")
    policies = batch["policies"]
    if not isinstance(policies, list) or [item.get("arm") for item in policies] != list(ARMS):
        raise ValueError("fault/patched 순서가 다르다")
    for policy in policies:
        arm = policy["arm"]
        if set(policy) != {"arm", "checkpoint", "training_result", "fixed_report", "fixed_metrics"}:
            raise ValueError(f"{arm} policy schema가 다르다")
        for key in ("checkpoint", "training_result", "fixed_report"):
            ref = policy[key]
            if set(ref) != {"path", "sha256"}:
                raise ValueError(f"{arm} {key} ref schema가 다르다")
            path = Path(ref["path"])
            if not path.is_absolute() or not path.is_file() or sha(path) != ref["sha256"]:
                raise ValueError(f"{arm} {key} 결합이 다르다")
        if any(policy["fixed_metrics"].get(key) != value for key, value in EXPECTED_METRICS[arm].items()):
            raise ValueError(f"{arm} 고정 평가 수치가 다르다")
    expected_boundary = {
        "fault": "fixed26_tracking_failure", "patched": "fixed26_healthy",
        "visual_motion_label": "pending_observation", "fall_claim": False,
    }
    if batch["claim_boundary"] != expected_boundary:
        raise ValueError("관측 전 라벨 경계가 다르다")
    return batch


def _policy(batch: dict[str, Any], arm: str) -> dict[str, Any]:
    if arm not in ARMS:
        raise ValueError("fault/patched만 녹화한다")
    return next(item for item in batch["policies"] if item["arm"] == arm)


def validate_owned_receipt(receipt: dict[str, Any]) -> dict[str, Any]:
    """SDK 자식이 자연 종료했고 Windows Job이 비었을 때만 수용한다."""
    if (receipt.get("exit_code") != 0 or receipt.get("natural_exit") is not True
            or receipt.get("cleanup_verified") is not True
            or receipt.get("timed_out") is not False
            or receipt.get("job_assigned_before_release") is not True
            or receipt.get("error_type") is not None):
        raise ValueError("owned SDK 자식의 실행·자연 종료·정리 영수증이 다르다")
    identities = receipt.get("owned_process_identities")
    if not isinstance(identities, dict) or not identities:
        raise ValueError("owned SDK 자식 PID 계보가 없다")
    return receipt


class TraceWriter:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.stream: TextIO | None = path.open("x", encoding="utf-8", newline="\n")
        self.rows = 0

    def append(self, row: dict[str, Any]) -> None:
        if self.stream is None:
            raise RuntimeError("trace가 이미 닫혔다")
        self.stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")
        self.stream.flush()
        self.rows += 1

    def close(self) -> None:
        if self.stream is not None:
            self.stream.close()
            self.stream = None


def install_exact_frame_encoder(writer_factory: Any = None) -> None:
    rendering = importlib.import_module("gymnasium.wrappers.rendering")

    def stop_recording(wrapper: Any) -> None:
        if not wrapper.recording or len(wrapper.recorded_frames) != FRAMES or wrapper.frames_per_sec != FPS:
            raise ValueError("등록된 750-frame/50fps encoder 입력이 아니다")
        import numpy as np

        factory = writer_factory
        if factory is None:
            factory = importlib.import_module("moviepy.video.io.ffmpeg_writer").FFMPEG_VideoWriter
        first = wrapper.recorded_frames[0]
        height, width = first.shape[:2]
        if (width, height) != RESOLUTION:
            raise ValueError("native RGB가 2560x1440이 아니다")
        video = Path(wrapper.video_folder) / f"{wrapper._video_name}.mp4"
        writer = factory(str(video), RESOLUTION, FPS, codec="libx264", preset="medium",
                         ffmpeg_params=["-crf", "16", "-movflags", "+faststart"])
        rows = []
        try:
            for index, frame in enumerate(wrapper.recorded_frames):
                if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim != 3
                        or frame.shape[2] not in (3, 4) or frame.shape[:2] != (RESOLUTION[1], RESOLUTION[0])):
                    raise ValueError("RGB 원본 프레임 shape/dtype가 다르다")
                rgb = np.ascontiguousarray(frame[:, :, :3])
                digest = hashlib.sha256()
                digest.update(str(rgb.dtype).encode("ascii"))
                digest.update(json.dumps(list(rgb.shape), separators=(",", ":")).encode("ascii"))
                digest.update(rgb.tobytes(order="C"))
                writer.write_frame(rgb)
                rows.append({"index": index, "sha256": digest.hexdigest()})
        finally:
            writer.close()
        write_new(video.with_suffix(video.suffix + ".exact-frames.json"), {
            "schema": ENCODER_SCHEMA, "frames_received": len(wrapper.recorded_frames),
            "frames_written": len(rows), "fps": FPS, "width": RESOLUTION[0],
            "height": RESOLUTION[1], "frame_sequence": rows, "video_file": video.name,
            "video_sha256": sha(video),
        })
        wrapper.recorded_frames = []
        wrapper.recording = False
        wrapper._video_name = None
        if wrapper.gc_trigger and wrapper.gc_trigger(wrapper.episode_id):
            gc.collect()

    rendering.RecordVideo.stop_recording = stop_recording


def configure_env(env_cfg: Any) -> dict[str, Any]:
    env_cfg.scene.num_envs = 1
    env_cfg.seed = SEED
    env_cfg.episode_length_s = 20.0
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.actions.joint_pos.scale = ACTION_SCALE
    cmd = env_cfg.commands.base_velocity
    cmd.heading_command = False
    cmd.rel_standing_envs = 0.0
    cmd.rel_heading_envs = 0.0
    cmd.resampling_time_range = (1.0e9, 1.0e9)
    for name in ("push_robot", "base_external_force_torque", "add_base_mass", "base_com"):
        if hasattr(env_cfg.events, name):
            setattr(env_cfg.events, name, None)
    reset = env_cfg.events.reset_base.params
    reset["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
    reset["velocity_range"] = {key: (0.0, 0.0) for key in (
        "x", "y", "z", "roll", "pitch", "yaw"
    )}
    viewer = env_cfg.viewer
    viewer.origin_type = "asset_root"
    viewer.asset_name = "robot"
    viewer.env_index = 0
    viewer.eye = EYE
    viewer.lookat = LOOKAT
    viewer.resolution = RESOLUTION
    return {
        "task": TASK, "seed": SEED, "num_envs": 1, "episode_length_s": 20.0,
        "command": list(COMMAND), "action_scale": ACTION_SCALE,
        "camera": {"origin_type": "asset_root", "asset_name": "robot", "env_index": 0,
                   "eye": list(EYE), "lookat": list(LOOKAT), "resolution": list(RESOLUTION)},
    }


def _snapshot(base: Any) -> tuple[Any, ...]:
    robot = base.scene["robot"]
    return tuple(value.detach().clone() for value in (
        robot.data.root_state_w, robot.data.joint_pos, robot.data.joint_vel,
    ))


def _same_snapshot(torch: Any, before: tuple[Any, ...], after: tuple[Any, ...]) -> bool:
    return all(torch.equal(left, right) for left, right in zip(before, after, strict=True))


def _trace_row(base: Any, frame: int, terminated: bool) -> dict[str, Any]:
    robot = base.scene["robot"]
    command = base.command_manager.get_command("base_velocity")[0].detach().cpu().tolist()
    return {
        "frame": frame, "sim_time_s": frame / FPS,
        "root_pos_w": robot.data.root_pos_w[0].detach().cpu().tolist(),
        "root_quat_w": robot.data.root_quat_w[0].detach().cpu().tolist(),
        "root_lin_vel_b": robot.data.root_lin_vel_b[0].detach().cpu().tolist(),
        "command_body": command, "terminated": terminated,
    }


def _run(batch: dict[str, Any], arm: str, out: Path) -> dict[str, Any]:
    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401
    import torch
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
    from rsl_rl.runners import OnPolicyRunner

    install_exact_frame_encoder()
    policy_ref = _policy(batch, arm)
    env_cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=1, use_fabric=True)
    environment = configure_env(env_cfg)
    agent_cfg = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
    agent_cfg.device = "cuda:0"
    base_env = gym.make(TASK, cfg=env_cfg, render_mode="rgb_array")

    class RenderSync(gym.Wrapper):
        calls = 0

        def render(self):
            before = _snapshot(self.unwrapped)
            self.unwrapped.sim.forward()
            after = _snapshot(self.unwrapped)
            if not _same_snapshot(torch, before, after):
                raise RuntimeError("render forward가 physics state를 바꿨다")
            self.calls += 1
            return self.env.render()

    synced = RenderSync(base_env)
    trace = TraceWriter(out / "state-trace.jsonl")
    env = None
    rows: list[dict[str, Any]] = []
    try:
        recorded = gym.wrappers.RecordVideo(
            synced, video_folder=str(out / "video"), step_trigger=lambda step: step == 0,
            video_length=FRAMES, disable_logger=True,
        )
        env = RslRlVecEnvWrapper(recorded, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device="cuda:0")
        runner.load(policy_ref["checkpoint"]["path"], load_optimizer=False)
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        base = env.unwrapped
        base.reset(seed=SEED)
        fixed = torch.tensor([COMMAND], dtype=torch.float32, device=base.device)
        command_buf = base.command_manager.get_command("base_velocity")
        command_buf.copy_(fixed)
        first = _trace_row(base, 0, False)
        trace.append(first)
        rows.append(first)
        with torch.no_grad():
            for frame in range(1, FRAMES + 1):
                command_buf.copy_(fixed)
                obs, _ = env.get_observations()
                obs, _, dones, _ = env.step(policy(obs))
                del obs
                terminated = bool(dones[0].item())
                row = _trace_row(base, frame, terminated)
                trace.append(row)
                rows.append(row)
                if terminated:
                    raise RuntimeError("등록된 15초 구간 전에 episode가 종료됐다")
    finally:
        trace.close()
        if env is not None:
            env.close()
        else:
            base_env.close()
    videos = sorted((out / "video").glob("*.mp4"))
    if len(videos) != 1 or len(rows) != FRAMES + 1 or synced.calls != FRAMES:
        raise RuntimeError("15초 native frame/trace/render 수가 다르다")
    encoder = videos[0].with_suffix(videos[0].suffix + ".exact-frames.json")
    encoder_value = read(encoder)
    if (encoder_value.get("frames_written") != FRAMES
            or [row.get("index") for row in encoder_value.get("frame_sequence", [])]
            != list(range(FRAMES))):
        raise RuntimeError("exact frame receipt가 다르다")
    return {
        "environment": environment,
        "video": {"path": videos[0].relative_to(out).as_posix(), "sha256": sha(videos[0])},
        "encoder": {"path": encoder.relative_to(out).as_posix(), "sha256": sha(encoder)},
        "trace": {"path": trace.path.relative_to(out).as_posix(), "sha256": sha(trace.path),
                  "rows": len(rows)},
        "render_sync": {"method": "simulation_context_forward_before_env_render",
                        "calls": synced.calls, "physics_state_unchanged": True},
    }


def _marker(out: Path, sequence: int, event: str, **extra: Any) -> None:
    write_new(out / "shutdown" / f"{sequence:02d}-{event}.json", {
        "schema": SHUTDOWN_SCHEMA, "sequence": sequence, "event": event,
        "pid": os.getpid(), "monotonic_ns": time.monotonic_ns(), **extra,
    })


def launcher_arguments() -> dict[str, Any]:
    return {"headless": True, "device": "cuda:0", "enable_cameras": True,
            "kit_args": "--/app/vulkan=false --/app/window/hideUi=true", "fast_shutdown": False}


def execute(batch_path: Path, arm: str, out: Path) -> dict[str, Any]:
    if out.exists() or out.is_symlink():
        raise FileExistsError(out)
    batch = validate_batch(read(batch_path))
    policy_ref = _policy(batch, arm)
    import lineage_lease_context_v3 as LEASE

    LEASE.lease_context(out.parent / f"{out.name}-lease-before")
    out.mkdir(parents=True)
    write_new(out / "intent.json", {
        "batch_sha256": sha(batch_path), "arm": arm,
        "checkpoint_sha256": policy_ref["checkpoint"]["sha256"],
        "new_training": 0, "model_api": 0,
    })
    app = None
    result: dict[str, Any] | None = None
    try:
        from isaaclab.app import AppLauncher

        app = AppLauncher(launcher_arguments()).app
        _marker(out, 1, "app-startup", enable_cameras=True)
        result = _run(batch, arm, out)
        manifest = {
            "schema": MANIFEST_SCHEMA, "status": "complete", "arm": arm,
            "seed": SEED, "task": TASK, "command": list(COMMAND),
            "checkpoint_sha256": policy_ref["checkpoint"]["sha256"],
            "fixed_report_sha256": policy_ref["fixed_report"]["sha256"],
            "fixed_metrics": policy_ref["fixed_metrics"],
            "source_sha256_lf": batch["source_sha256_lf"],
            "capture": batch["capture"], "claim_boundary": batch["claim_boundary"],
            **result,
        }
        write_new(out / "manifest.json", manifest)
        _marker(out, 2, "result-ready", manifest_sha256=sha(out / "manifest.json"))
        LEASE.lease_context(out / "lease-after")
        import carb
        settings = carb.settings.get_settings()
        settings.set_bool("/app/fastShutdown", True)
        if settings.get_as_bool("/app/fastShutdown") is not True:
            raise RuntimeError("fast shutdown 설정 확인 실패")
        _marker(out, 3, "app-close-entered", fast_shutdown=True, wait_for_replicator=False)
        app.close(wait_for_replicator=False)
        app = None
        _marker(out, 4, "app-close-returned")
        return manifest
    except BaseException as error:  # noqa: BLE001 - SDK 실패를 봉인한 뒤 강제 비정상 종료한다.
        if not (out / "failure.json").exists():
            write_new(out / "failure.json", {
                "error_type": f"{type(error).__module__}.{type(error).__qualname__}",
                "error_message": str(error),
                "traceback": traceback.format_exception(type(error), error, error.__traceback__),
                "partial_evidence_preserved": True,
            })
        traceback.print_exception(type(error), error, error.__traceback__, file=sys.stderr)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    finally:
        if app is not None:
            try:
                app.close(wait_for_replicator=False)
            except BaseException as close_error:  # noqa: BLE001 - 종료 실패 증거를 남긴다.
                if not (out / "close-failure.json").exists():
                    write_new(out / "close-failure.json", {
                        "error_type": f"{type(close_error).__module__}.{type(close_error).__qualname__}",
                        "error_message": str(close_error),
                        "primary_failure_preserved": (out / "failure.json").exists(),
                    })


def verify_capture(out: Path, batch_path: Path, arm: str) -> dict[str, Any]:
    batch = validate_batch(read(batch_path))
    policy = _policy(batch, arm)
    manifest = read(out / "manifest.json")
    if (manifest.get("schema") != MANIFEST_SCHEMA or manifest.get("status") != "complete"
            or manifest.get("arm") != arm or manifest.get("checkpoint_sha256") != policy["checkpoint"]["sha256"]
            or manifest.get("fixed_report_sha256") != policy["fixed_report"]["sha256"]
            or manifest.get("capture") != batch["capture"]
            or manifest.get("source_sha256_lf") != batch["source_sha256_lf"]):
        raise ValueError("capture manifest 결합이 다르다")
    for key in ("video", "encoder", "trace"):
        ref = manifest[key]
        path = out / ref["path"]
        if Path(ref["path"]).is_absolute() or ".." in Path(ref["path"]).parts or sha(path) != ref["sha256"]:
            raise ValueError(f"{key} artifact 결합이 다르다")
    encoder = read(out / manifest["encoder"]["path"])
    if (encoder.get("schema") != ENCODER_SCHEMA or encoder.get("frames_written") != FRAMES
            or encoder.get("fps") != FPS or [encoder.get("width"), encoder.get("height")] != list(RESOLUTION)
            or [row.get("index") for row in encoder.get("frame_sequence", [])] != list(range(FRAMES))):
        raise ValueError("native exact-frame 영수증이 다르다")
    if manifest.get("trace", {}).get("rows") != FRAMES + 1 or manifest.get("render_sync") != {
        "method": "simulation_context_forward_before_env_render", "calls": FRAMES,
        "physics_state_unchanged": True,
    }:
        raise ValueError("trace/render-sync 영수증이 다르다")
    shutdown = [read(path) for path in sorted((out / "shutdown").glob("*.json"))]
    if [row["event"] for row in shutdown] != [
        "app-startup", "result-ready", "app-close-entered", "app-close-returned"
    ]:
        raise ValueError("fast-close 증거가 다르다")
    return {
        "status": "complete", "arm": arm, "manifest_sha256": sha(out / "manifest.json"),
        "video_sha256": manifest["video"]["sha256"], "frames": FRAMES,
        "fps": FPS, "duration_s": FRAMES / FPS,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command_name", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--repo", type=Path, default=ROOT)
    prepare.add_argument("--run-root", required=True, type=Path)
    prepare.add_argument("--out", required=True, type=Path)
    record = sub.add_parser("record")
    record.add_argument("--batch", required=True, type=Path)
    record.add_argument("--arm", required=True, choices=ARMS)
    record.add_argument("--out", required=True, type=Path)
    check = sub.add_parser("check")
    check.add_argument("--batch", required=True, type=Path)
    check.add_argument("--arm", required=True, choices=ARMS)
    check.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.command_name == "prepare":
        write_new(args.out, build_batch(args.repo, args.run_root))
        value = {"status": "prepared", "batch": str(args.out), "sha256": sha(args.out)}
    elif args.command_name == "record":
        value = execute(args.batch, args.arm, args.out)
    else:
        value = verify_capture(args.out, args.batch, args.arm)
    print(json.dumps(value, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
