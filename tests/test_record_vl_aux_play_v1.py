import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
import record_vl_aux_play_v1 as R


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def files(tmp_path):
    sdk = tmp_path / "IsaacLab"
    play = sdk / R.PLAY_RELATIVE
    play.parent.mkdir(parents=True)
    play.write_text("play", encoding="utf-8")
    run = sdk / "logs/rsl_rl/unitree_go2_flat/run"
    params = run / "params"
    params.mkdir(parents=True)
    checkpoint = run / "model_299.pt"
    env = params / "env.yaml"
    agent = params / "agent.yaml"
    checkpoint.write_bytes(b"checkpoint")
    env.write_text("env", encoding="utf-8")
    agent.write_text("agent", encoding="utf-8")
    fixed = tmp_path / "fixed.json"
    fixed.write_text(json.dumps({"checkpoint": {"sha256": sha(checkpoint)}, "action_scale": 0.25}), encoding="utf-8")
    artifact = {
        "checkpoint": {"path": checkpoint, "sha256": sha(checkpoint)},
        "environment": {"path": env, "sha256": sha(env)},
        "agent": {"path": agent, "sha256": sha(agent)},
    }
    protocol = R.build_recording_contract(sdk)
    protocol["fixed_evaluation_label"] = "unhealthy"
    return sdk, artifact, fixed, protocol


def idle():
    return {"busy": False, "utilization_percent": 0, "memory_used_mib": 1}


def test_records_with_existing_recorder_and_keeps_label_out_of_item(monkeypatch, tmp_path):
    sdk, artifact, fixed, protocol = files(tmp_path)
    output = tmp_path / "private-stage"
    calls = []
    gpu_calls = []
    monkeypatch.setattr(R.M, "query_gpu", lambda profile: gpu_calls.append(profile) or idle())

    def record(passed_protocol, item, passed_sdk, private, timeout):
        calls.append((passed_protocol, item, passed_sdk, private, timeout))
        video = private / "policy_play/videos/play/clip.mp4"
        video.parent.mkdir(parents=True)
        video.write_bytes(b"video")
        return {"video": video, "probe": R.M.EXPECTED_VIDEO_CONTRACT,
                "own_pid_cleanup": True, "residual_processes_observed": 0}

    monkeypatch.setattr(R.M, "_record_clip", record)
    result = R.record_clip(artifact, fixed, output, sdk, protocol)
    assert calls[0][2:] == (sdk.resolve(), output.resolve(), 900)
    assert gpu_calls == ["desktop_shared_recording_v1", "desktop_shared_recording_v1"]
    assert "fixed_evaluation_label" not in calls[0][1]["clip"]
    assert calls[0][1]["clip"]["overrides"] == protocol["overrides"]
    assert "env.actions.joint_pos.scale=0.25" in calls[0][1]["clip"]["overrides"]
    assert result["sha256"] == hashlib.sha256(b"video").hexdigest()
    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["fixed_evaluation_label"] == "unhealthy"
    assert manifest["label_forwarded_to_player_or_model"] is False
    assert manifest["playback"] == {
        "task": "Isaac-Velocity-Flat-Unitree-Go2-Play-v0", "seed": 42,
        "num_envs": 16, "video_length": 300, "action_scale": 0.25,
        "overrides": protocol["overrides"],
    }


def test_unknown_fixed_label_is_recordable_but_not_forwarded(monkeypatch, tmp_path):
    sdk, artifact, fixed, protocol = files(tmp_path)
    protocol["fixed_evaluation_label"] = "unknown"
    output = tmp_path / "unknown-label"
    monkeypatch.setattr(R.M, "query_gpu", lambda _profile: idle())

    def record(_protocol, item, _sdk, private, _timeout):
        assert "fixed_evaluation_label" not in item["clip"]
        video = private / "policy_play/videos/play/clip.mp4"
        video.parent.mkdir(parents=True)
        video.write_bytes(b"video")
        return {"video": video, "probe": {}, "own_pid_cleanup": True,
                "residual_processes_observed": 0}

    monkeypatch.setattr(R.M, "_record_clip", record)
    result = R.record_clip(artifact, fixed, output, sdk, protocol)
    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["fixed_evaluation_label"] == "unknown"
    assert manifest["label_forwarded_to_player_or_model"] is False


def test_rejects_changed_input_before_output_or_gpu(monkeypatch, tmp_path):
    sdk, artifact, fixed, protocol = files(tmp_path)
    Path(artifact["environment"]["path"]).write_text("changed", encoding="utf-8")
    monkeypatch.setattr(R.M, "query_gpu", lambda _profile: pytest.fail("invalid input reached GPU"))
    output = tmp_path / "unused"
    with pytest.raises(R.RecordingContractError, match="digest differs"):
        R.record_clip(artifact, fixed, output, sdk, protocol)
    assert not output.exists()


def test_rejects_existing_output_without_touching_it(monkeypatch, tmp_path):
    sdk, artifact, fixed, protocol = files(tmp_path)
    output = tmp_path / "used"
    output.mkdir()
    marker = output / "marker"
    marker.write_text("keep", encoding="utf-8")
    monkeypatch.setattr(R.M, "query_gpu", lambda _profile: pytest.fail("collision reached GPU"))
    with pytest.raises(FileExistsError):
        R.record_clip(artifact, fixed, output, sdk, protocol)
    assert marker.read_text(encoding="utf-8") == "keep"


def test_busy_gpu_consumes_output_and_never_launches(monkeypatch, tmp_path):
    sdk, artifact, fixed, protocol = files(tmp_path)
    output = tmp_path / "busy"
    monkeypatch.setattr(R.M, "query_gpu", lambda _profile: {"busy": True})
    monkeypatch.setattr(R.M, "_record_clip", lambda *_args: pytest.fail("busy GPU launched Play"))
    with pytest.raises(R.RecordingContractError, match="before recording"):
        R.record_clip(artifact, fixed, output, sdk, protocol)
    assert (output / "failure.json").is_file()
    with pytest.raises(FileExistsError):
        R.record_clip(artifact, fixed, output, sdk, protocol)


def test_rejects_unverified_cleanup(monkeypatch, tmp_path):
    sdk, artifact, fixed, protocol = files(tmp_path)
    output = tmp_path / "cleanup"
    monkeypatch.setattr(R.M, "query_gpu", lambda _profile: idle())
    monkeypatch.setattr(R.M, "_record_clip", lambda *_args: {
        "video": output / "unused.mp4", "probe": {},
        "own_pid_cleanup": False, "residual_processes_observed": 1,
    })
    with pytest.raises(R.RecordingContractError, match="cleanup"):
        R.record_clip(artifact, fixed, output, sdk, protocol)
    assert json.loads((output / "failure.json").read_text())["stage"] == "cleanup"


def test_post_recording_gpu_busy_preserves_partial_and_rejects(monkeypatch, tmp_path):
    sdk, artifact, fixed, protocol = files(tmp_path)
    output = tmp_path / "post-busy"
    observations = iter([idle(), {"busy": True}])
    monkeypatch.setattr(R.M, "query_gpu", lambda _profile: next(observations))

    def record(_protocol, _item, _sdk, private, _timeout):
        video = private / "policy_play/videos/play/clip.mp4"
        video.parent.mkdir(parents=True)
        video.write_bytes(b"video")
        return {"video": video, "probe": {}, "own_pid_cleanup": True,
                "residual_processes_observed": 0}

    monkeypatch.setattr(R.M, "_record_clip", record)
    with pytest.raises(R.RecordingContractError, match="after recording"):
        R.record_clip(artifact, fixed, output, sdk, protocol)
    assert json.loads((output / "failure.json").read_text())["stage"] == "gpu_after"


def test_rejects_post_recording_source_change(monkeypatch, tmp_path):
    sdk, artifact, fixed, protocol = files(tmp_path)
    output = tmp_path / "mutated"
    monkeypatch.setattr(R.M, "query_gpu", lambda _profile: idle())

    def mutate(_protocol, item, _sdk, private, _timeout):
        Path(artifact["agent"]["path"]).write_text("mutated", encoding="utf-8")
        video = private / "policy_play/videos/play/clip.mp4"
        video.parent.mkdir(parents=True)
        video.write_bytes(b"video")
        return {"video": video, "probe": {}, "own_pid_cleanup": True,
                "residual_processes_observed": 0}

    monkeypatch.setattr(R.M, "_record_clip", mutate)
    with pytest.raises(R.RecordingContractError, match="changed during execution"):
        R.record_clip(artifact, fixed, output, sdk, protocol)
    assert json.loads((output / "failure.json").read_text())["stage"] == "source_recheck"


def test_fixed_report_must_bind_same_checkpoint(monkeypatch, tmp_path):
    sdk, artifact, fixed, protocol = files(tmp_path)
    fixed.write_text(json.dumps({"checkpoint": {"sha256": "0" * 64}, "action_scale": 0.25}), encoding="utf-8")
    monkeypatch.setattr(R.M, "query_gpu", lambda _profile: pytest.fail("bad report reached GPU"))
    with pytest.raises(R.RecordingContractError, match="checkpoint differ"):
        R.record_clip(artifact, fixed, tmp_path / "unused", sdk, protocol)
