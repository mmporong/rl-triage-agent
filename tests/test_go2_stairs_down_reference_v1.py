from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
SPEC = importlib.util.spec_from_file_location("reference_test", ROOT / "bench/run_go2_stairs_down_reference_v1.py")
assert SPEC and SPEC.loader
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def test_only_command_and_logical_clip_identity_change():
    old = {"robot_role": "quadruped", "direction": "down", "seed": 200,
           "checkpoint": {"sha256": M.CHECKPOINT_SHA},
           "capture": {"frames": 625, "fps": 50, "command_x_m_s": 0.25,
                       "step_height_m": 0.05, "step_width_m": 0.3}}
    value = M.make_spec(old)
    assert value["capture"]["command_x_m_s"] == 0.75
    assert value["capture"]["frames"] == 1500 and old["capture"]["command_x_m_s"] == 0.25
    assert "0p75" in value["clip_id"] and value["checkpoint"] == old["checkpoint"]


def test_reference_preserves_observation_horizon_reset_and_terrain():
    terrain = NS(proportion=1, step_height_range=(0.05, 0.23), step_width=0.3)
    cfg = NS(scene=NS(terrain=NS(terrain_generator=NS(sub_terrains={"pyramid_stairs": terrain}))),
             commands=NS(base_velocity=NS(ranges=NS())),
             events=NS(reset_base=NS(params={}), reset_robot_joints=NS(params={})),
             observations=NS(policy=NS(enable_corruption=True)), viewer=NS(),
             terminations=NS(base_contact=object(), time_out=object()))
    base = M.load_base()
    value = base.configure_env(cfg, {"direction": "down", "seed": 200})
    assert cfg.commands.base_velocity.ranges.lin_vel_x == (0.75, 0.75)
    assert cfg.episode_length_s == 31 and cfg.commands.base_velocity.resampling_time_range == (31, 31)
    assert cfg.terminations.base_contact is None and cfg.terminations.time_out is None
    assert value["step_height_m"] == 0.05 and value["observation_extension"]["seconds"] == 30
