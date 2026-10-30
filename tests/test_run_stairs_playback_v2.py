from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace as NS

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "bench/run_stairs_playback_v2.py"
SPEC = importlib.util.spec_from_file_location("stairs_runner", SOURCE)
assert SPEC and SPEC.loader
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def _cfg() -> NS:
    terrain = NS(proportion=0.2, step_height_range=(0.05, 0.23), step_width=0.3)
    generator = NS(sub_terrains={"pyramid_stairs": terrain, "pyramid_stairs_inv": terrain},
                   num_rows=5, num_cols=5, curriculum=False, use_cache=True)
    ranges = NS(lin_vel_x=(0.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0),
                heading=(-3.14, 3.14))
    command = NS(resampling_time_range=(10.0, 10.0), rel_standing_envs=0.02,
                 rel_heading_envs=1.0, heading_command=True, ranges=ranges)
    reset = {"pose_range": {}, "velocity_range": {}}
    events = NS(reset_base=NS(params=reset), reset_robot_joints=NS(params={}), push_robot=object(),
                base_external_force_torque=object(), add_base_mass=object(), base_com=object())
    return NS(
        scene=NS(terrain=NS(terrain_generator=generator, max_init_terrain_level=None), num_envs=50),
        commands=NS(base_velocity=command), events=events,
        observations=NS(policy=NS(enable_corruption=True)),
        viewer=NS(origin_type=None, asset_name=None, env_index=None, eye=None, lookat=None),
        seed=None, episode_length_s=40.0,
    )


def _spec(direction: str) -> dict:
    return {"direction": direction, "seed": 172}


def test_configure_up_uses_single_inverted_stairs_and_fixed_command() -> None:
    cfg = _cfg()
    receipt = M.configure_env(cfg, _spec("up"))
    assert set(cfg.scene.terrain.terrain_generator.sub_terrains) == {"pyramid_stairs_inv"}
    terrain = cfg.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs_inv"]
    assert terrain.step_height_range == (0.10, 0.10)
    assert cfg.commands.base_velocity.ranges.lin_vel_x == (0.25, 0.25)
    assert cfg.events.reset_base.params["pose_range"]["x"] == (0.0, 0.0)
    assert cfg.events.push_robot is None
    assert receipt["direction_geometry"] == "center_low_to_outer_up"


def test_configure_down_uses_pyramid_and_fixed_camera() -> None:
    cfg = _cfg()
    receipt = M.configure_env(cfg, _spec("down"))
    assert set(cfg.scene.terrain.terrain_generator.sub_terrains) == {"pyramid_stairs"}
    assert cfg.viewer.origin_type == "asset_root"
    assert cfg.viewer.asset_name == "robot"
    assert receipt["direction_geometry"] == "center_high_to_outer_down"


def test_assessment_keeps_success_fall_and_failure_separate() -> None:
    up = [{"root_pos_w": [0.0, 0.0, 0.2], "terminated": False}]
    up.extend({"root_pos_w": [1.6, 0.0, 0.7], "terminated": False} for _ in range(M.FRAMES))
    assert M.assess_trace(_spec("up"), up)["outcome"] == "completed"
    fall = list(up)
    fall[-1] = {"root_pos_w": [1.6, 0.0, 0.7], "terminated": True}
    assert M.assess_trace(_spec("up"), fall)["outcome"] == "fall"
    flat = list(up)
    flat[-1] = {"root_pos_w": [1.6, 0.0, 0.25], "terminated": False}
    assert M.assess_trace(_spec("up"), flat)["outcome"] == "failed"


def test_import_does_not_load_isaac_or_torch() -> None:
    import sys

    assert "isaaclab" not in sys.modules
    assert "torch" not in sys.modules


def test_spec_binds_checkpoint_and_exact_source_closure(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint-fixture")
    role = "biped"
    sources = M.COMMON_ISAACLAB_SOURCES | {M.ROLE_ISAACLAB_SOURCE[role]}
    spec = {
        "schema": M.SCHEMA, "clip_id": "biped-stairs-up", "robot_role": role, "direction": "up",
        **M.ROBOT_TASKS[role], "seed": 172,
        "checkpoint": {"path": str(checkpoint.resolve()),
                       "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()},
        "source": {"isaaclab_commit": "9" * 40,
                   "isaaclab_files_lf": {name: "8" * 64 for name in sources},
                   "runner_sha256_lf": M._sha_lf(SOURCE)},
        "capture": {"frames": 625, "fps": 50, "step_height_m": 0.10, "step_width_m": 0.30,
                    "command_x_m_s": 0.25, "camera": "asset_root_three_quarter"},
    }
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    assert M._read_spec(path)["checkpoint"]["sha256"] == spec["checkpoint"]["sha256"]
    spec["source"]["isaaclab_files_lf"].pop(next(iter(sources)))
    path.write_text(json.dumps(spec), encoding="utf-8")
    try:
        M._read_spec(path)
    except ValueError as error:
        assert "source closure" in str(error)
    else:
        raise AssertionError("누락 source closure를 수용했다")


def test_cli_help_has_no_isaac_import_dependency(tmp_path: Path) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = ""
    result = subprocess.run(
        [sys.executable, "-B", str(SOURCE), "--help"], cwd=tmp_path,
        env=env, capture_output=True, check=False,
    )
    assert result.returncode == 0
    assert b"--spec" in result.stdout and b"--out" in result.stdout
    assert result.stderr == b""


def test_rollout_flushes_partial_trace_before_mid_step_error(tmp_path: Path) -> None:
    class Inference:
        def __enter__(self):
            return None

        def __exit__(self, *_args):
            return False

    class Torch:
        @staticmethod
        def inference_mode():
            return Inference()

    class Done:
        @staticmethod
        def item():
            return False

    class Env:
        calls = 0

        def step(self, _action):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("mid-step fixture")
            return "obs-1", None, [Done()], None

    original = M._tensor_row
    setattr(M, "_tensor_row", lambda _base, frame, terminated: {  # noqa: B010
        "frame": frame,
        "root_pos_w": [float(frame), 0.0, 0.0],
        "terminated": terminated,
    })
    trace = M.TraceWriter(tmp_path / "state-trace.jsonl")
    try:
        try:
            M._rollout(Env(), object(), lambda _obs: "action", "obs-0", Torch(), trace)
        except RuntimeError as error:
            assert str(error) == "mid-step fixture"
        else:
            raise AssertionError("중간 step 오류가 전파되지 않았다")
    finally:
        trace.close()
        setattr(M, "_tensor_row", original)  # noqa: B010
    rows = [json.loads(line) for line in trace.path.read_text(encoding="utf-8").splitlines()]
    assert [row["frame"] for row in rows] == [0, 1]
    assert trace.rows == 2


def test_gymnasium_1_2_record_video_625_steps_yields_625_frames_on_close() -> None:
    contract = M._record_video_contract(625, 625)
    assert contract == {
        "gymnasium_version": "1.2.0",
        "source": M.GYMNASIUM_RECORD_VIDEO_SOURCE,
        "step_trigger": "step_id == 0",
        "video_length": 625,
        "executed_steps": 625,
        "frames_before_close": 625,
        "recording_before_close": True,
        "close_finalizes_recording": True,
    }
    assert M._record_video_contract(626, 625)["frames_before_close"] == 626
    assert M._record_video_contract(626, 625)["recording_before_close"] is False


def test_shutdown_markers_are_write_once_and_ordered(tmp_path: Path) -> None:
    startup = M._shutdown_marker(tmp_path, 1, "app-startup")
    entered = M._shutdown_marker(tmp_path, 2, "app-close-entered")
    returned = M._shutdown_marker(tmp_path, 3, "app-close-returned")
    values = [json.loads(path.read_text(encoding="utf-8")) for path in (startup, entered, returned)]
    assert [value["sequence"] for value in values] == [1, 2, 3]
    assert [value["event"] for value in values] == [
        "app-startup", "app-close-entered", "app-close-returned",
    ]
    try:
        M._shutdown_marker(tmp_path, 1, "app-startup")
    except FileExistsError:
        pass
    else:
        raise AssertionError("shutdown marker 덮어쓰기를 수용했다")
