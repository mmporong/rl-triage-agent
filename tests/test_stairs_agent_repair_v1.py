import copy
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "bench")]
C = importlib.import_module("stairs_agent_repair_v1")
TRAIN = importlib.import_module("train_stairs_agent_adaptation_v1")
TRAIN3 = importlib.import_module("train_stairs_agent_adaptation_v3")
PLAY = importlib.import_module("run_go2_stairs_agent_play_v1")


def rows():
    return [{"frame": i, "sim_time_s": i/50, "episode_length_step": i,
             "terminated": False, "root_pos_w": [i/250, 0, 0.75-i/3000],
             "root_quat_w": [1, 0, 0, 0], "body_up_world_z": 1,
             "root_lin_vel_w": [0.25, 0, 0],
             "command_body": [0.25, 0, 0]} for i in range(1501)]


def source():
    data = rows()
    for row in data[600:]:
        row.update(root_quat_w=[0, 1, 0, 0], body_up_world_z=-1)
    return {"schema": "stairs_agent_source_v1", "checkpoint_sha256": "a"*64,
            "assessment": C.assess(data), "observed_symptom": "fall"}


def decision(action, cited=None):
    return {"action": action, "cited_facts": cited or ["source"], "reason": "기록된 실패 확인"}


def test_real_measurements_require_same_command_and_no_resets():
    data = rows()
    assert C.assess(data)["accepted"]
    data[100]["command_body"] = [0.75, 0, 0]
    assert not C.assess(data)["accepted"]
    data[100]["episode_length_step"] = 0
    with pytest.raises(ValueError):
        C.assess(data)


def test_body_frame_tracking_error_rotates_world_velocity():
    data = rows()
    for row in data:
        row.update(root_quat_w=[2**-0.5,0,0,2**-0.5], root_lin_vel_w=[0,0.25,0])
    assert C.assess(data)["tracking_rmse_xy_m_s"] < 1e-12
    for row in data:
        row["root_lin_vel_w"] = [0.25,0,0]
    assert C.assess(data)["tracking_rmse_xy_m_s"] > 0.25


@pytest.mark.parametrize("field,value", [("root_pos_w", [float('nan'),0,0]),
    ("root_quat_w", [2,0,0,0]), ("command_body", [0.25]), ("body_up_world_z", -1)])
def test_invalid_observation_is_rejected(field, value):
    data = rows()
    data[20][field] = value
    with pytest.raises(ValueError):
        C.assess(data)


def test_model_chooses_checks_before_adaptation_and_probe_values_are_hidden():
    s = source()
    context = C.packet(s, [])
    assert C.ADAPT not in context["eligible_actions"]
    assert "assessment" not in context["facts"]["source"]
    with pytest.raises(ValueError):
        C.validate_decision(decision(C.ADAPT), context)
    receipts = [C.probe(p, s) for p in C.PROBES]
    assert [p["result"] for p in receipts] == ["normal", "abnormal"]
    d = decision(C.ADAPT, list(C.PROBES))
    auth = C.authorize_adaptation(s, receipts, d, True)
    assert auth["selected_by"] == "model" and auth["task"]["command_x_m_s"] == 0.25
    with pytest.raises(ValueError):
        C.authorize_adaptation(s, receipts, d, False)
    with pytest.raises(ValueError):
        C.authorize_adaptation(s, receipts, decision(C.ADAPT), True)


def test_normal_control_never_offers_training():
    s = source()
    s["assessment"] = C.assess(rows())
    context = C.packet(s, [C.probe(p, s) for p in C.PROBES])
    assert context["eligible_actions"] == ["HOLD"]


def test_unknown_tool_fake_fact_receipt_replacement_and_duplicates_rejected():
    s = source()
    with pytest.raises(ValueError):
        C.validate_decision(decision("change_speed_to_0p75"), C.packet(s, []))
    with pytest.raises(ValueError):
        C.validate_decision(decision(C.PROBES[0], ["invented"]), C.packet(s, []))
    p = C.probe(C.PROBES[0], s)
    changed = copy.deepcopy(p)
    changed["result"] = "abnormal"
    with pytest.raises(ValueError):
        C.packet(s, [changed])
    with pytest.raises(ValueError):
        C.packet(s, [p, p])


@pytest.mark.parametrize("training_module", [TRAIN, TRAIN3])
def test_training_keeps_task_and_rewards_while_preserving_training_terminations(training_module):
    terrain = NS(proportion=1, step_height_range=(0.05,0.2), step_width=0.3)
    term, timeout, rewards = object(), object(), object()
    cfg = NS(scene=NS(terrain=NS(terrain_generator=NS(sub_terrains={"pyramid_stairs":terrain}))),
        commands=NS(base_velocity=NS(ranges=NS())),events=NS(reset_base=NS(params={}),
        reset_robot_joints=NS(params={})),observations=NS(policy=NS(enable_corruption=True)),
        viewer=NS(),terminations=NS(base_contact=term,time_out=timeout),
        curriculum=NS(terrain_levels=object()),rewards=rewards)
    training_module.configure_training(cfg,{"direction":"down","seed":5710})
    assert cfg.scene.num_envs == 1024
    assert cfg.commands.base_velocity.ranges.lin_vel_x == (0.25,0.25)
    assert cfg.terminations.base_contact is term and cfg.terminations.time_out is timeout
    assert cfg.rewards is rewards and cfg.curriculum.terrain_levels is None


def test_actual_play_verifier_binds_checkpoint_terrain_and_command(tmp_path):
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
    video = tmp_path/"video/clip.mp4"
    video.parent.mkdir()
    video.write_bytes(b"synthetic verifier boundary fixture")
    trace = tmp_path/"state-trace.jsonl"
    trace.write_text("\n".join(json.dumps(r) for r in rows())+"\n", encoding="utf-8")
    m = {"schema":PLAY.MANIFEST_SCHEMA,"source":{"fixture":True},"checkpoint_sha256":"a"*64,
         "clip_id":"fixture","render_sync":{"physics_state_unchanged":True},
         "state_trace":{"sha256":C.sha(trace)},"video":{"path":"video/clip.mp4","sha256":C.sha(video)},
         "terrain":{"terrain_id":"pyramid_stairs","step_height_m":0.05,"step_width_m":0.3,
         "command":{"lin_vel_x":0.25,"lin_vel_y":0.0,"ang_vel_z":0.0},
         "observation_extension":{"seconds":30,"automatic_reset_disabled":True}}}
    write(tmp_path/"manifest.json",m)
    write(video.with_suffix('.mp4.exact-frames.json'),{"video_sha256":C.sha(video),"frames_written":1500,
         "frames_received":1500,"frame_sequence":[{"index":i} for i in range(1500)],"fps":50})
    s = {"source":{"fixture":True},"checkpoint":{"sha256":"a"*64},"clip_id":"fixture"}
    assert PLAY.verify_record(tmp_path,s)["assessment"]["accepted"]
    s["checkpoint"]["sha256"] = "b"*64
    with pytest.raises(ValueError):
        PLAY.verify_record(tmp_path,s)
    s["checkpoint"]["sha256"] = "a"*64
    m["terrain"]["command"]["lin_vel_x"] = 0.75
    write(tmp_path/"manifest.json",m)
    with pytest.raises(ValueError):
        PLAY.verify_record(tmp_path,s)
