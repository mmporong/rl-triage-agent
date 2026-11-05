from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("go2_long", ROOT / "bench/run_go2_stairs_down_long_v1.py")
assert SPEC and SPEC.loader
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def test_extension_changes_duration_and_schema_without_changing_original():
    old = {"robot_role": "quadruped", "direction": "down", "seed": 200,
           "checkpoint": {"sha256": M.CHECKPOINT_SHA},
           "capture": {"frames": 625, "fps": 50, "command_x_m_s": 0.25,
                       "step_height_m": 0.05, "step_width_m": 0.3}}
    unchanged = copy.deepcopy(old)
    new = M.make_spec(old)
    assert old == unchanged and new["capture"]["frames"] == 1500
    assert new["capture"]["command_x_m_s"] == 0.25 and new["schema"] == M.SPEC_SCHEMA
    old["seed"] = 201
    with pytest.raises(ValueError):
        M.make_spec(old)


def test_long_configuration_disables_reset_without_touching_policy_or_stairs():
    base = M.load_base()
    terrain = NS(proportion=1.0, step_height_range=(0.05, 0.23), step_width=0.3)
    cfg = NS(scene=NS(terrain=NS(terrain_generator=NS(sub_terrains={"pyramid_stairs": terrain}))),
             commands=NS(base_velocity=NS(ranges=NS())),
             events=NS(reset_base=NS(params={}), reset_robot_joints=NS(params={})),
             observations=NS(policy=NS(enable_corruption=True)), viewer=NS(),
             terminations=NS(base_contact=object(), time_out=object()))
    value = base.configure_env(cfg, {"direction": "down", "seed": 200})
    assert cfg.terminations.base_contact is None and cfg.terminations.time_out is None
    assert cfg.episode_length_s == 31 and cfg.commands.base_velocity.resampling_time_range == (31, 31)
    assert cfg.commands.base_velocity.ranges.lin_vel_x == (0.25, 0.25)
    assert value["step_height_m"] == 0.05 and value["step_width_m"] == 0.3
    assert value["observation_extension"]["automatic_reset_disabled"] is True
    original = base._record_video_contract(1500, 1500)
    assert original["frames_before_close"] == 1500 and original["close_finalizes_recording"]
    assert M.sha(ROOT / "bench/run_stairs_playback_v2.py", lf=True) == M.BASE_SHA


def test_quaternion_observation_detects_inversion_without_claiming_cause():
    assert M.body_up_z([1, 0, 0, 0]) == 1
    assert M.body_up_z([0, 1, 0, 0]) == -1
    with pytest.raises(ValueError):
        M.body_up_z([float("nan"), 0, 0, 0])


def fixture_record(out):
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
    video = out / "video/clip.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"synthetic encoder verification fixture")
    rows = [{"frame": i, "episode_length_step": i, "sim_time_s": i / 50,
             "terminated": False, "root_quat_w": [1, 0, 0, 0], "body_up_world_z": 1,
             "root_pos_w": [i / 200, 0, -i / 3000]} for i in range(1501)]
    trace = out / "state-trace.jsonl"
    trace.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    manifest = {"schema": M.MANIFEST_SCHEMA, "clip_id": M.CLIP_ID,
                "checkpoint_sha256": M.CHECKPOINT_SHA, "source": {"fixture": True},
                "terrain": {"observation_extension": {"automatic_reset_disabled": True, "seconds": 30}},
                "render_sync": {"physics_state_unchanged": True},
                "state_trace": {"sha256": M.sha(trace)},
                "video": {"path": "video/clip.mp4", "sha256": M.sha(video)}}
    write(out / "manifest.json", manifest)
    write(video.with_suffix(".mp4.exact-frames.json"), {
        "frames_received": 1500, "frames_written": 1500, "fps": 50,
        "frame_sequence": [{"index": i, "sha256": "a" * 64} for i in range(1500)],
        "video_sha256": M.sha(video)})
    return rows, manifest


def test_frame_count_reset_and_encoder_mismatch_are_rejected(tmp_path):
    rows, manifest = fixture_record(tmp_path)
    result = M.verify_record(tmp_path, {"source": {"fixture": True}})
    assert result["duration_s"] == 30 and result["agent_recovery_evidence"] is False
    assert result["reset_count"] == 0
    rows[700]["episode_length_step"] = 0
    trace = tmp_path / "state-trace.jsonl"
    trace.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    manifest["state_trace"]["sha256"] = M.sha(trace)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="리셋"):
        M.verify_record(tmp_path, {"source": {"fixture": True}})
