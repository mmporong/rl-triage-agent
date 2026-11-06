# pyright: reportMissingImports=false, reportAttributeAccessIssue=false
import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
import record_generated_ppo_comparison_v1 as R


def test_actual_camera_launcher_uses_registered_windows_renderer():
    arguments = R.launcher_arguments()
    assert arguments["enable_cameras"] is True
    assert arguments["headless"] is True
    assert "--/app/vulkan=false" in arguments["kit_args"].split()
    assert arguments["fast_shutdown"] is False


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _file(path: Path, raw: bytes = b"x") -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest()}


def _batch(tmp_path: Path, monkeypatch) -> dict:
    monkeypatch.setattr(R, "_source_closure", lambda repo: {"bench/record_generated_ppo_comparison_v1.py": "a" * 64})
    policies = []
    for arm in R.ARMS:
        checkpoint = _file(tmp_path / arm / "final.pt", arm.encode())
        training = _file(tmp_path / arm / "training-result.json", f"train-{arm}".encode())
        report = _file(tmp_path / arm / "fixed.json", f"fixed-{arm}".encode())
        policies.append({
            "arm": arm, "checkpoint": checkpoint, "training_result": training,
            "fixed_report": report,
            "fixed_metrics": {"fall_rate": 0.0, "mean_survival_s": 20.0,
                              "lin_vel_rmse_mps": R.EXPECTED_METRICS[arm]["lin_vel_rmse_mps"],
                              "yaw_rate_rmse_radps": 0.5},
        })
    return {
        "schema": R.SCHEMA, "development_case": True, "seed": R.SEED, "task": R.TASK,
        "command": list(R.COMMAND), "action_scale": R.ACTION_SCALE,
        "capture": {"duration_s": 15.0, "frames": R.FRAMES, "fps": R.FPS,
                    "resolution": list(R.RESOLUTION),
                    "camera": {"eye": list(R.EYE), "lookat": list(R.LOOKAT),
                               "origin_type": "asset_root"},
                    "same_native_duration": True, "new_training": 0, "model_api": 0},
        "source_sha256_lf": {"bench/record_generated_ppo_comparison_v1.py": "a" * 64},
        "policies": policies,
        "claim_boundary": {"fault": "fixed26_tracking_failure", "patched": "fixed26_healthy",
                           "visual_motion_label": "pending_observation", "fall_claim": False},
    }


def test_import_does_not_load_sdk_modules():
    source = ROOT / "bench/record_generated_ppo_comparison_v1.py"
    script = (
        "import importlib.util,sys;"
        f"s=importlib.util.spec_from_file_location('isolated_recorder',r'{source}');"
        "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
        "assert all(x not in sys.modules for x in ('isaaclab','torch','carb','omni'))"
    )
    import subprocess
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                            encoding="utf-8", errors="replace", check=False)
    assert result.returncode == 0, result.stderr


def test_validate_accepts_only_exact_two_policy_contract(tmp_path, monkeypatch):
    batch = _batch(tmp_path, monkeypatch)
    assert R.validate_batch(batch, tmp_path) is batch
    changes = [
        ("duration", lambda value: value["capture"].update(duration_s=14.9)),
        ("resolution", lambda value: value["capture"].update(resolution=[1920, 1080])),
        ("camera", lambda value: value["capture"]["camera"].update(eye=[3.5, 3.5, 2.2])),
        ("command", lambda value: value.update(command=[0.2, 0.0, 0.0])),
        ("order", lambda value: value["policies"].reverse()),
        ("label", lambda value: value["claim_boundary"].update(visual_motion_label="walking")),
    ]
    for _, mutate in changes:
        bad = copy.deepcopy(batch)
        mutate(bad)
        with pytest.raises(ValueError):
            R.validate_batch(bad, tmp_path)


def test_checkpoint_and_source_tamper_are_rejected(tmp_path, monkeypatch):
    batch = _batch(tmp_path, monkeypatch)
    Path(batch["policies"][0]["checkpoint"]["path"]).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checkpoint"):
        R.validate_batch(batch, tmp_path)
    batch = _batch(tmp_path / "fresh", monkeypatch)
    batch["source_sha256_lf"]["bench/record_generated_ppo_comparison_v1.py"] = "b" * 64
    with pytest.raises(ValueError, match="source closure"):
        R.validate_batch(batch, tmp_path)


def test_configure_env_changes_only_registered_flat_play_values():
    events = NS(push_robot=object(), base_external_force_torque=object(), add_base_mass=object(),
                base_com=object(), reset_base=NS(params={}))
    env = NS(scene=NS(num_envs=99), seed=1, episode_length_s=10,
             observations=NS(policy=NS(enable_corruption=True)),
             actions=NS(joint_pos=NS(scale=1.0)), events=events,
             commands=NS(base_velocity=NS(heading_command=True, rel_standing_envs=1.0,
                                           rel_heading_envs=1.0, resampling_time_range=(1, 2))),
             viewer=NS(origin_type="world", asset_name=None, env_index=7,
                       eye=(6, 6, 3), lookat=(0, 0, 0), resolution=(1280, 720)))
    receipt = R.configure_env(env)
    assert receipt["command"] == [0.25, 0.0, 0.0]
    assert env.scene.num_envs == 1 and env.actions.joint_pos.scale == 0.25
    assert env.viewer.eye == (3.5, 6.0, 2.2) and env.viewer.resolution == (2560, 1440)
    assert all(getattr(events, name) is None for name in (
        "push_robot", "base_external_force_torque", "add_base_mass", "base_com"
    ))


def test_exact_encoder_fixes_native_quality_and_rejects_wrong_shape(monkeypatch):
    class RecordVideo:
        pass
    rendering = NS(RecordVideo=RecordVideo)
    original = importlib.import_module
    monkeypatch.setattr(importlib, "import_module",
                        lambda name: rendering if name == "gymnasium.wrappers.rendering" else original(name))
    calls = []

    class Writer:
        def __init__(self, *args, **kwargs):
            calls.append((args, kwargs))
        def write_frame(self, frame):
            pass
        def close(self):
            pass

    R.install_exact_frame_encoder(Writer)
    import numpy as np
    folder = Path(pytest.ensuretemp("generated-patch-encoder")) if hasattr(pytest, "ensuretemp") else None
    # tmp_path를 쓰지 않아도 되는 단일 frame shape 거부를 먼저 확인한다.
    wrapper = NS(recording=True, recorded_frames=[np.zeros((1, 1, 3), dtype=np.uint8)] * R.FRAMES,
                 frames_per_sec=R.FPS, video_folder=str(folder or Path.cwd()), _video_name="x",
                 gc_trigger=None, episode_id=0)
    with pytest.raises(ValueError, match="2560x1440"):
        RecordVideo.stop_recording(wrapper)
    assert not calls


def test_render_snapshot_equality_is_exact():
    class Value:
        def detach(self): return self
        def clone(self): return self
    robot = NS(data=NS(root_state_w=Value(), joint_pos=Value(), joint_vel=Value()))
    before = R._snapshot(NS(scene={"robot": robot}))
    fake_torch = NS(equal=lambda left, right: left is right)
    assert R._same_snapshot(fake_torch, before, before)
    assert not R._same_snapshot(fake_torch, before, (object(), *before[1:]))


def test_owned_receipt_requires_natural_exit_cleanup_and_pid_lineage():
    good = {"exit_code": 0, "natural_exit": True, "cleanup_verified": True,
            "timed_out": False, "job_assigned_before_release": True,
            "error_type": None, "owned_process_identities": {"123": 456}}
    assert R.validate_owned_receipt(good) is good
    for field, value in (("exit_code", 1), ("natural_exit", False),
                         ("cleanup_verified", False), ("timed_out", True),
                         ("job_assigned_before_release", False), ("error_type", "OSError")):
        bad = dict(good)
        bad[field] = value
        with pytest.raises(ValueError, match="owned SDK"):
            R.validate_owned_receipt(bad)
    bad = dict(good, owned_process_identities={})
    with pytest.raises(ValueError, match="PID"):
        R.validate_owned_receipt(bad)


def test_verify_capture_requires_750_native_frames_and_fast_close(tmp_path, monkeypatch):
    batch = _batch(tmp_path / "inputs", monkeypatch)
    batch_path = tmp_path / "batch.json"
    _write(batch_path, batch)
    out = tmp_path / "capture"
    video = _file(out / "video/capture.mp4", b"video")
    encoder_path = out / "video/capture.mp4.exact-frames.json"
    _write(encoder_path, {"schema": R.ENCODER_SCHEMA, "frames_written": R.FRAMES,
                          "fps": R.FPS, "width": 2560, "height": 1440,
                          "frame_sequence": [{"index": i, "sha256": "a" * 64} for i in range(R.FRAMES)]})
    trace = _file(out / "state-trace.jsonl", b"trace")
    policy = batch["policies"][0]
    manifest = {"schema": R.MANIFEST_SCHEMA, "status": "complete", "arm": "fault",
                "checkpoint_sha256": policy["checkpoint"]["sha256"],
                "fixed_report_sha256": policy["fixed_report"]["sha256"],
                "capture": batch["capture"], "source_sha256_lf": batch["source_sha256_lf"],
                "video": {"path": "video/capture.mp4", "sha256": video["sha256"]},
                "encoder": {"path": "video/capture.mp4.exact-frames.json",
                            "sha256": R.sha(encoder_path)},
                "trace": {"path": "state-trace.jsonl", "sha256": trace["sha256"], "rows": 751},
                "render_sync": {"method": "simulation_context_forward_before_env_render",
                                "calls": 750, "physics_state_unchanged": True}}
    _write(out / "manifest.json", manifest)
    for index, event in enumerate(("app-startup", "result-ready", "app-close-entered", "app-close-returned"), 1):
        _write(out / "shutdown" / f"{index:02d}-{event}.json", {
            "schema": R.SHUTDOWN_SCHEMA, "event": event, "pid": 123, "sequence": index,
            "monotonic_ns": index, "manifest_sha256": R.sha(out / "manifest.json"),
            "fast_shutdown": True, "wait_for_replicator": False})
    owned = {"exit_code": 0, "natural_exit": True, "cleanup_verified": True,
             "timed_out": False, "job_assigned_before_release": True,
             "error_type": None, "owned_process_identities": {"123": 456}}
    _write(out.parent / "owned/owned-result.json", owned)
    assert R.verify_capture(out, batch_path, "fault")["duration_s"] == 15.0
    (out / "shutdown/04-app-close-returned.json").unlink()
    assert R.verify_capture(out, batch_path, "fault")["app_close_returned"] is False
    for field, value in (("exit_code", 1), ("cleanup_verified", False)):
        _write(out.parent / "owned/owned-result.json", {**owned, field: value})
        with pytest.raises(ValueError, match="owned SDK"):
            R.verify_capture(out, batch_path, "fault")
    _write(out.parent / "owned/owned-result.json", {**owned, "owned_process_identities": {"456": 789}})
    with pytest.raises(ValueError, match="종료 PID"):
        R.verify_capture(out, batch_path, "fault")
    _write(out.parent / "owned/owned-result.json", owned)
    manifest["trace"]["rows"] = 750
    _write(out / "bad.json", manifest)
    (out / "manifest.json").write_text((out / "bad.json").read_text(), encoding="utf-8")
    with pytest.raises(ValueError, match="trace/render-sync"):
        R.verify_capture(out, batch_path, "fault")
