# pyright: reportMissingImports=false
"""Isaac Lab v2.1.1 공개 rough 정책을 고정 계단 한 장면에서 녹화한다.

Isaac/torch import는 AppLauncher가 열린 뒤 수행한다. 한 호출은 한 robot·방향 clip만 만들며,
실패·넘어짐도 manifest와 trace에 남긴다.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any, TextIO

SCHEMA = "stairs_playback_spec_v2"
MANIFEST_SCHEMA = "stairs_playback_manifest_v2"
SHUTDOWN_MARKER_SCHEMA = "stairs_playback_shutdown_marker_v1"
GYMNASIUM_RECORD_VIDEO_VERSION = "1.2.0"
GYMNASIUM_RECORD_VIDEO_SOURCE = (
    "https://raw.githubusercontent.com/Farama-Foundation/Gymnasium/"
    "v1.2.0/gymnasium/wrappers/rendering.py"
)
FRAMES = 625
FPS = 50
STEP_HEIGHT_M = 0.10
STEP_WIDTH_M = 0.30
COMMAND_X_M_S = 0.25
MIN_PROGRESS_M = 1.5
MIN_HEIGHT_CHANGE_M = 0.30
ROBOT_TASKS = {
    "biped": {
        "training_task": "Isaac-Velocity-Rough-H1-v0",
        "play_task": "Isaac-Velocity-Rough-H1-Play-v0",
    },
    "quadruped": {
        "training_task": "Isaac-Velocity-Rough-Unitree-Go2-v0",
        "play_task": "Isaac-Velocity-Rough-Unitree-Go2-Play-v0",
    },
}
TERRAIN_FOR_DIRECTION = {"up": "pyramid_stairs_inv", "down": "pyramid_stairs"}
COMMON_ISAACLAB_SOURCES = {
    "source/isaaclab/isaaclab/terrains/config/rough.py",
    "source/isaaclab/isaaclab/terrains/trimesh/mesh_terrains.py",
    "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py",
}
ROLE_ISAACLAB_SOURCE = {
    "biped": "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/h1/rough_env_cfg.py",
    "quadruped": "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/rough_env_cfg.py",
}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha_lf(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _hex(value: Any, length: int, label: str) -> str:
    if not isinstance(value, str) or len(value) != length:
        raise ValueError(f"{label} 형식이 다르다")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{label} 형식이 다르다") from error
    return value


def _write_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")


class TraceWriter:
    """한 행씩 write-once trace에 반영하고 즉시 flush한다."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream: TextIO | None = self.path.open("x", encoding="utf-8", newline="\n")
        self.rows = 0

    def append(self, row: dict[str, Any]) -> None:
        if self._stream is None:
            raise RuntimeError("state trace가 이미 닫혔다")
        self._stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")
        self._stream.flush()
        self.rows += 1

    def close(self) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None


def _record_video_contract(steps: int, video_length: int) -> dict[str, Any]:
    """Gymnasium 1.2.0 ``RecordVideo.step`` 순서를 그대로 계산한다."""
    if steps < 0 or video_length <= 0:
        raise ValueError("record video step/length가 양수 계약과 다르다")
    step_id = -1
    recording = False
    recorded_frames = 0
    for _ in range(steps):
        step_id += 1
        if not recording and step_id == 0:
            recording = True
        if recording:
            recorded_frames += 1
            if recorded_frames > video_length:
                recording = False
    return {
        "gymnasium_version": GYMNASIUM_RECORD_VIDEO_VERSION,
        "source": GYMNASIUM_RECORD_VIDEO_SOURCE,
        "step_trigger": "step_id == 0",
        "video_length": video_length,
        "executed_steps": steps,
        "frames_before_close": recorded_frames,
        "recording_before_close": recording,
        "close_finalizes_recording": recording,
    }


def _shutdown_marker(out: Path, sequence: int, event: str, **fields: Any) -> Path:
    path = out / "shutdown" / f"{sequence:02d}-{event}.json"
    _write_new(path, {
        "schema": SHUTDOWN_MARKER_SCHEMA,
        "event": event,
        "pid": os.getpid(),
        "sequence": sequence,
        "monotonic_ns": time.monotonic_ns(),
        **fields,
    })
    return path


def _error_value(error: BaseException) -> dict[str, Any]:
    return {
        "error_type": f"{type(error).__module__}.{type(error).__qualname__}",
        "error_message": str(error),
        "traceback": traceback.format_exception(type(error), error, error.__traceback__),
    }


def _read_spec(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("spec은 JSON object여야 한다")
    required = {
        "schema", "clip_id", "robot_role", "direction", "training_task", "play_task", "seed",
        "checkpoint", "source", "capture",
    }
    if set(value) != required or value["schema"] != SCHEMA:
        raise ValueError("stairs playback spec schema/keys가 다르다")
    role = value["robot_role"]
    direction = value["direction"]
    if role not in ROBOT_TASKS or direction not in TERRAIN_FOR_DIRECTION:
        raise ValueError("robot_role/direction이 등록 밖이다")
    if {"training_task": value["training_task"], "play_task": value["play_task"]} != ROBOT_TASKS[role]:
        raise ValueError("robot task가 등록값과 다르다")
    if not isinstance(value["seed"], int) or isinstance(value["seed"], bool):
        raise TypeError("seed는 정수여야 한다")
    checkpoint = value["checkpoint"]
    if not isinstance(checkpoint, dict) or set(checkpoint) != {"path", "sha256"}:
        raise ValueError("checkpoint ref가 다르다")
    checkpoint_path = Path(checkpoint["path"])
    if not checkpoint_path.is_absolute() or not checkpoint_path.is_file():
        raise FileNotFoundError("checkpoint 파일이 없다")
    _hex(checkpoint["sha256"], 64, "checkpoint SHA")
    if _sha(checkpoint_path) != checkpoint["sha256"]:
        raise ValueError("checkpoint SHA가 다르다")
    source = value["source"]
    if not isinstance(source, dict) or set(source) != {
        "isaaclab_commit", "isaaclab_files_lf", "runner_sha256_lf"
    }:
        raise ValueError("source binding이 다르다")
    if source["runner_sha256_lf"] != _sha_lf(Path(__file__)):
        raise ValueError("runner source SHA가 다르다")
    _hex(source["isaaclab_commit"], 40, "Isaac Lab commit")
    required_sources = COMMON_ISAACLAB_SOURCES | {ROLE_ISAACLAB_SOURCE[role]}
    if not isinstance(source["isaaclab_files_lf"], dict) or set(source["isaaclab_files_lf"]) != required_sources:
        raise ValueError("Isaac Lab source closure가 다르다")
    for name, digest in source["isaaclab_files_lf"].items():
        _hex(digest, 64, f"Isaac Lab source SHA {name}")
    capture = value["capture"]
    expected_capture = {
        "frames": FRAMES, "fps": FPS, "step_height_m": STEP_HEIGHT_M,
        "step_width_m": STEP_WIDTH_M, "command_x_m_s": COMMAND_X_M_S,
        "camera": "asset_root_three_quarter",
    }
    if capture != expected_capture:
        raise ValueError("capture 고정값이 다르다")
    return value


def verify_isaaclab_source(spec: dict[str, Any], isaaclab_module: Any) -> dict[str, Any]:
    root = Path(isaaclab_module.__file__).resolve().parents[3]
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True,
        encoding="utf-8", errors="strict", check=False,
    )
    if commit.returncode or commit.stderr or commit.stdout.strip() != spec["source"]["isaaclab_commit"]:
        raise ValueError("Isaac Lab source commit이 다르다")
    observed = {name: _sha_lf(root / name) for name in spec["source"]["isaaclab_files_lf"]}
    if observed != spec["source"]["isaaclab_files_lf"]:
        raise ValueError("Isaac Lab source file SHA가 다르다")
    return {"commit": commit.stdout.strip(), "files_lf": observed}


def configure_env(env_cfg: Any, spec: dict[str, Any]) -> dict[str, Any]:
    """Play config를 단일 고정 계단·고정 명령·고정 reset으로 바꾼다."""
    direction = spec["direction"]
    terrain_key = TERRAIN_FOR_DIRECTION[direction]
    generator = env_cfg.scene.terrain.terrain_generator
    if generator is None or terrain_key not in generator.sub_terrains:
        raise ValueError("공식 rough config에 등록 계단 지형이 없다")
    terrain = copy.deepcopy(generator.sub_terrains[terrain_key])
    terrain.proportion = 1.0
    terrain.step_height_range = (STEP_HEIGHT_M, STEP_HEIGHT_M)
    terrain.step_width = STEP_WIDTH_M
    generator.sub_terrains = {terrain_key: terrain}
    generator.num_rows = 1
    generator.num_cols = 1
    generator.curriculum = False
    generator.use_cache = False
    env_cfg.scene.terrain.max_init_terrain_level = 0
    env_cfg.scene.num_envs = 1
    env_cfg.seed = spec["seed"]
    env_cfg.episode_length_s = 20.0

    command = env_cfg.commands.base_velocity
    command.resampling_time_range = (FRAMES / FPS + 1.0, FRAMES / FPS + 1.0)
    command.rel_standing_envs = 0.0
    command.rel_heading_envs = 0.0
    command.heading_command = False
    command.ranges.lin_vel_x = (COMMAND_X_M_S, COMMAND_X_M_S)
    command.ranges.lin_vel_y = (0.0, 0.0)
    command.ranges.ang_vel_z = (0.0, 0.0)
    command.ranges.heading = (0.0, 0.0)

    reset = env_cfg.events.reset_base.params
    reset["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
    reset["velocity_range"] = {
        "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
        "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
    }
    if getattr(env_cfg.events, "reset_robot_joints", None) is not None:
        env_cfg.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
    for name in ("push_robot", "base_external_force_torque", "add_base_mass", "base_com"):
        if hasattr(env_cfg.events, name):
            setattr(env_cfg.events, name, None)
    env_cfg.observations.policy.enable_corruption = False

    viewer = env_cfg.viewer
    viewer.origin_type = "asset_root"
    viewer.asset_name = "robot"
    viewer.env_index = 0
    viewer.eye = (6.0, 6.0, 3.0)
    viewer.lookat = (0.0, 0.0, 0.5)
    return {
        "terrain_id": terrain_key,
        "terrain_seed": spec["seed"],
        "step_height_m": STEP_HEIGHT_M,
        "step_width_m": STEP_WIDTH_M,
        "spawn": "terrain_origin_center",
        "heading_rad": 0.0,
        "command": {"lin_vel_x": COMMAND_X_M_S, "lin_vel_y": 0.0, "ang_vel_z": 0.0},
        "direction_geometry": "center_low_to_outer_up" if direction == "up" else "center_high_to_outer_down",
    }


def assess_trace(spec: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) < 2:
        raise ValueError("state trace가 너무 짧다")
    first, last = rows[0], rows[-1]
    dx = float(last["root_pos_w"][0]) - float(first["root_pos_w"][0])
    dz = float(last["root_pos_w"][2]) - float(first["root_pos_w"][2])
    terminated = any(bool(row["terminated"]) for row in rows)
    height_ok = dz >= MIN_HEIGHT_CHANGE_M if spec["direction"] == "up" else dz <= -MIN_HEIGHT_CHANGE_M
    progress_ok = dx >= MIN_PROGRESS_M
    if terminated:
        outcome = "fall"
    elif height_ok and progress_ok and len(rows) == FRAMES + 1:
        outcome = "completed"
    else:
        outcome = "failed"
    return {
        "outcome": outcome, "root_dx_m": dx, "root_dz_m": dz,
        "height_condition_pass": height_ok, "progress_condition_pass": progress_ok,
        "termination_seen": terminated, "trace_rows": len(rows),
    }


def _tensor_row(base_env: Any, frame: int, terminated: bool) -> dict[str, Any]:
    robot = base_env.scene["robot"]
    pos = robot.data.root_pos_w[0].detach().cpu().tolist()
    quat = robot.data.root_quat_w[0].detach().cpu().tolist()
    lin_vel = robot.data.root_lin_vel_w[0].detach().cpu().tolist()
    terrain_origin = base_env.scene.terrain.terrain_origins[0].detach().cpu().tolist()
    return {
        "frame": frame, "sim_time_s": frame / FPS, "root_pos_w": pos, "root_quat_w": quat,
        "root_lin_vel_w": lin_vel, "terrain_origin_w": terrain_origin, "terminated": terminated,
    }


def _snapshot(base_env: Any) -> tuple[Any, Any, Any]:
    robot = base_env.scene["robot"]
    return tuple(value.detach().clone() for value in (
        robot.data.root_state_w, robot.data.joint_pos, robot.data.joint_vel,
    ))


def _same_snapshot(torch: Any, before: tuple[Any, ...], after: tuple[Any, ...]) -> bool:
    return all(torch.equal(left, right) for left, right in zip(before, after, strict=True))


def _rollout(env: Any, base_env: Any, policy: Any, obs: Any, torch: Any, trace: TraceWriter) -> list[dict[str, Any]]:
    """정책 재생 중 생성된 행을 즉시 trace에 반영한다."""
    rows = [_tensor_row(base_env, 0, False)]
    trace.append(rows[0])
    with torch.inference_mode():
        for frame in range(1, FRAMES + 1):
            obs, _, dones, _ = env.step(policy(obs))
            terminated = bool(dones[0].item())
            row = _tensor_row(base_env, frame, terminated)
            trace.append(row)
            rows.append(row)
            if terminated:
                break
    return rows


def _run(spec: dict[str, Any], out: Path, app: Any) -> dict[str, Any]:
    import gymnasium as gym
    import isaaclab
    import isaaclab_tasks  # noqa: F401
    import torch
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
    from rsl_rl.runners import OnPolicyRunner

    source_receipt = verify_isaaclab_source(spec, isaaclab)
    env_cfg = parse_env_cfg(spec["play_task"], device="cuda:0", num_envs=1, use_fabric=True)
    terrain_receipt = configure_env(env_cfg, spec)
    agent_cfg = load_cfg_from_registry(spec["training_task"], "rsl_rl_cfg_entry_point")
    agent_cfg.device = "cuda:0"
    base_env = gym.make(spec["play_task"], cfg=env_cfg, render_mode="rgb_array")

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

    synced_env = RenderSync(base_env)
    video_dir = out / "video"
    env = None
    trace = TraceWriter(out / "state-trace.jsonl")
    rows: list[dict[str, Any]] = []
    try:
        recorded_env = gym.wrappers.RecordVideo(
            synced_env, video_folder=str(video_dir), step_trigger=lambda step: step == 0,
            video_length=FRAMES, disable_logger=True,
        )
        env = RslRlVecEnvWrapper(recorded_env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(spec["checkpoint"]["path"])
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        obs, _ = env.get_observations()
        rows = _rollout(env, base_env.unwrapped, policy, obs, torch, trace)
    finally:
        try:
            trace.close()
        finally:
            if env is not None:
                env.close()
            else:
                base_env.close()
    videos = sorted(video_dir.glob("*.mp4"))
    if len(videos) != 1:
        raise RuntimeError("RecordVideo mp4가 하나가 아니다")
    trace_path = trace.path
    assessment = assess_trace(spec, rows)
    video_contract = _record_video_contract(FRAMES, FRAMES)
    if video_contract["frames_before_close"] != FRAMES or not video_contract["close_finalizes_recording"]:
        raise RuntimeError("Gymnasium RecordVideo frame 계약이 625 step/625 frame과 다르다")
    return {
        "terrain": terrain_receipt,
        "isaaclab_source": source_receipt,
        "assessment": assessment,
        "video": {"path": videos[0].relative_to(out).as_posix(), "sha256": _sha(videos[0])},
        "record_video_contract": video_contract,
        "state_trace": {"path": trace_path.relative_to(out).as_posix(), "sha256": _sha(trace_path)},
        "render_sync": {"method": "simulation_context_forward_before_env_render", "calls": synced_env.calls,
                        "physics_state_unchanged": True},
        "camera": {"origin_type": "asset_root", "asset_name": "robot", "env_index": 0,
                   "eye": [6.0, 6.0, 3.0], "lookat": [0.0, 0.0, 0.5]},
    }


def execute(spec_path: Path, out: Path) -> dict[str, Any]:
    if out.exists() or out.is_symlink():
        raise FileExistsError(out)
    spec = _read_spec(spec_path)
    out.mkdir(parents=True)
    _write_new(out / "intent.json", {
        "spec_sha256": _sha(spec_path),
        "checkpoint_sha256": spec["checkpoint"]["sha256"], "new_training": 0,
        "model_api": 0, "diagnosis_or_recovery_claim": False,
    })
    app = None
    manifest: dict[str, Any] | None = None
    primary_error: BaseException | None = None
    try:
        from isaaclab.app import AppLauncher

        app = AppLauncher(headless=True, enable_cameras=True).app
        _shutdown_marker(out, 1, "app-startup", enable_cameras=True, headless=True)
        result = _run(spec, out, app)
        manifest = {
            "schema": MANIFEST_SCHEMA, "clip_id": spec["clip_id"], "robot_role": spec["robot_role"],
            "direction": spec["direction"], "training_task": spec["training_task"],
            "play_task": spec["play_task"], "checkpoint_sha256": spec["checkpoint"]["sha256"],
            "source": spec["source"], **result,
        }
        _write_new(out / "manifest.json", manifest)
        _shutdown_marker(out, 2, "result-ready", manifest_sha256=_sha(out / "manifest.json"))
    except BaseException as error:  # noqa: BLE001 - SDK 종료 전 모든 실패를 봉인한다.
        primary_error = error
        trace_path = out / "state-trace.jsonl"
        trace_rows = trace_path.read_bytes().count(b"\n") if trace_path.is_file() else 0
        failure = {
            **_error_value(error),
            "partial_evidence_preserved": True,
            "partial_trace": None if not trace_path.is_file() else {
                "path": trace_path.relative_to(out).as_posix(),
                "sha256": _sha(trace_path),
                "rows": trace_rows,
            },
        }
        _write_new(out / "failure.json", failure)
        _shutdown_marker(out, 2, "first-error", failure_sha256=_sha(out / "failure.json"))
    close_error: BaseException | None = None
    if app is not None:
        _shutdown_marker(out, 3, "app-close-entered")
        try:
            app.close()
        except BaseException as error:  # noqa: BLE001 - SDK close 실패도 marker로 봉인한다.
            close_error = error
            _shutdown_marker(out, 4, "app-close-error", **_error_value(error))
        else:
            _shutdown_marker(out, 4, "app-close-returned")
    if primary_error is not None:
        raise primary_error.with_traceback(primary_error.__traceback__)
    if close_error is not None:
        if not (out / "failure.json").exists():
            _write_new(out / "failure.json", {
                **_error_value(close_error),
                "partial_evidence_preserved": True,
                "phase": "app-close",
            })
        raise close_error.with_traceback(close_error.__traceback__)
    if manifest is None:
        raise RuntimeError("manifest 없이 stairs playback이 끝났다")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(execute(args.spec, args.out), ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
