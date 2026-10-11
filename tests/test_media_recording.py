import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


MODULE_PATH = Path(__file__).parents[1] / "bench" / "record_evidence_media.py"
SPEC = importlib.util.spec_from_file_location("record_evidence_media", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
media = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(media)


def _fixture(tmp_path: Path):
    repo = tmp_path / "repo"
    isaac = tmp_path / "IsaacLab"
    play = isaac / "scripts/reinforcement_learning/rsl_rl/play.py"
    launcher = isaac / "_isaac_sim/python.bat"
    play.parent.mkdir(parents=True)
    launcher.parent.mkdir(parents=True)
    play.write_text("play", encoding="utf-8")
    launcher.write_text("launcher", encoding="utf-8")
    source = repo / "bench/runs/source.json"
    source.parent.mkdir(parents=True)
    source.write_text("{}\n", encoding="utf-8")
    recorder = repo / "bench/record_evidence_media.py"
    recorder.write_text("recorder\n", encoding="utf-8")
    checkpoint = isaac / "logs/rsl_rl/unitree_go2_flat/run/model_99.pt"
    params = checkpoint.parent / "params"
    params.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    (params / "env.yaml").write_text("episode_length_s: 1.0\n", encoding="utf-8")
    (params / "agent.yaml").write_text("seed: 42\n", encoding="utf-8")
    protocol = {
        "task": "task", "seed": 42, "num_envs": 16, "video_length": 300, "kit_args": "d3d12",
        "play_py_sha256": media.sha256_file(play),
        "recorder_source": {"path": "bench/record_evidence_media.py", "sha256_lf": media.sha256_lf_file(recorder)},
        "video_contract": {"width": 1280, "height": 720, "frame_rate": "50/1", "frame_count": 299, "duration_s": 5.98, "duration_tolerance_s": 0.02},
        "clips": [{
            "id": "before", "run_basename": "run", "checkpoint": "model_99.pt",
            "checkpoint_sha256": media.sha256_file(checkpoint),
            "params_env_sha256": media.sha256_file(params / "env.yaml"),
            "params_agent_sha256": media.sha256_file(params / "agent.yaml"),
            "overrides": ["env.episode_length_s=1.0"],
        }],
        "public_sources": [{"path": "bench/runs/source.json", "sha256_lf": media.sha256_lf_file(source)}],
    }
    return repo, isaac, protocol, checkpoint


def _tools(monkeypatch, tmp_path: Path):
    for name in ("ffmpeg", "ffprobe"):
        tool = tmp_path / name
        tool.write_text("tool", encoding="utf-8")
    monkeypatch.setattr(shutil, "which", lambda name: str(tmp_path / name))


def test_checkpoint_digest_mismatch_rejects_before_any_capture_copy(tmp_path, monkeypatch):
    repo, isaac, protocol, _ = _fixture(tmp_path)
    _tools(monkeypatch, tmp_path)
    broken = copy.deepcopy(protocol)
    broken["clips"][0]["checkpoint_sha256"] = "0" * 64

    with pytest.raises(media.PreflightError, match="checkpoint before digest mismatch"):
        media.preflight(broken, repo, isaac, lambda: {"busy": False})

    assert not (repo / "bench/private").exists()
    assert not (repo / "docs/media").exists()


def test_busy_gpu_writes_waiting_status_without_copying_checkpoint(tmp_path, monkeypatch):
    repo, isaac, protocol, original = _fixture(tmp_path)
    _tools(monkeypatch, tmp_path)
    before = media.sha256_file(original)

    result = media.run_recording(protocol, repo, isaac, "busy-01", 60, lambda: {"busy": True})

    assert result == {"state": "waiting_gpu_idle", "reason": "gpu_busy"}
    assert json.loads((repo / "docs/media/recovery-c01-busy-01/status.json").read_text(encoding="utf-8"))["state"] == "waiting_gpu_idle"
    assert not (repo / "bench/private/media_capture_busy-01").exists()
    assert media.sha256_file(original) == before


def test_clip_gpu_guard_blocks_before_launcher_or_private_copy(tmp_path, monkeypatch):
    repo, isaac, protocol, original = _fixture(tmp_path)
    _tools(monkeypatch, tmp_path)
    states = iter(({"busy": False}, {"busy": True}))
    monkeypatch.setattr(media, "_record_clip", lambda *_args: pytest.fail("launcher must not start"))

    result = media.run_recording(protocol, repo, isaac, "became-busy", 60, lambda: next(states))

    assert result["state"] == "waiting_gpu_idle"
    assert not (repo / "bench/private/media_capture_became-busy").exists()
    assert media.sha256_file(original) == protocol["clips"][0]["checkpoint_sha256"]


def test_agent_params_tamper_is_rejected_before_launcher(tmp_path, monkeypatch):
    _repo, isaac, protocol, checkpoint = _fixture(tmp_path)
    private = tmp_path / "private"
    private.mkdir()
    params = checkpoint.parent / "params"
    (params / "agent.yaml").write_text("seed: 7\n", encoding="utf-8")
    monkeypatch.setattr(media, "WindowsJob", lambda: pytest.fail("launcher must not be constructed"))

    with pytest.raises(media.PreflightError, match="agent params before launch before digest mismatch"):
        media._record_clip(protocol, {"clip": protocol["clips"][0], "checkpoint": checkpoint, "params_dir": params}, isaac, private, 60)

    assert not (private / "before").exists()


def test_existing_public_output_is_never_overwritten(tmp_path):
    repo = tmp_path / "repo"
    public = repo / "docs/media/recovery-c01-used"
    public.mkdir(parents=True)
    clip = public / "before.mp4"
    clip.write_bytes(b"keep")

    with pytest.raises(media.PreflightError, match="public media tag already exists"):
        media.reserve_output(repo, "used", {"busy": False})

    assert clip.read_bytes() == b"keep"


def test_initial_waiting_tag_can_resume_and_reach_complete_status(tmp_path):
    repo = tmp_path / "repo"
    assert media.reserve_output(repo, "resume", {"busy": True}) is None
    reserved = media.reserve_output(repo, "resume", {"busy": False})
    assert reserved is not None
    public, _private = reserved

    media._set_status(public / "status.json", {"state": "complete", "clips": 3})

    assert json.loads((public / "status.json").read_text(encoding="utf-8")) == {"state": "complete", "clips": 3}


def test_gpu_busy_between_clips_marks_partial_tag_aborted(tmp_path, monkeypatch):
    repo, isaac, protocol, checkpoint = _fixture(tmp_path)
    _tools(monkeypatch, tmp_path)
    second = copy.deepcopy(protocol["clips"][0])
    second["id"] = "second"
    protocol["clips"].append(second)
    states = iter(({"busy": False}, {"busy": False}, {"busy": True}))
    monkeypatch.setattr(media, "_record_clip", lambda _protocol, item, *_args: {"clip": item["clip"]})

    result = media.run_recording(protocol, repo, isaac, "partial", 60, lambda: next(states))

    assert result["state"] == "aborted_gpu_busy"
    status = json.loads((repo / "docs/media/recovery-c01-partial/status.json").read_text(encoding="utf-8"))
    assert status["partial_private_clips"] == 1
    assert status["retry_requires_new_tag"] is True
    assert media.sha256_file(checkpoint) == protocol["clips"][0]["checkpoint_sha256"]


@pytest.mark.parametrize("tag", ["../escape", "Upper", "two words", "x" * 49])
def test_tag_rejects_path_and_noncanonical_values(tag):
    with pytest.raises(media.PreflightError):
        media.validate_tag(tag)


def test_protocol_rejects_larger_evaluation(tmp_path):
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({
        "schema_version": 1, "protocol_id": media.EXPECTED_PROTOCOL_ID,
        "task": media.EXPECTED_TASK, "seed": 42, "num_envs": 32, "video_length": 300,
        "kit_args": "--/app/vulkan=false --/app/window/hideUi=true", "play_py_sha256": "0" * 64,
        "recorder_source": {}, "video_contract": {},
        "clips": [{"id": name} for name in media.EXPECTED_CLIP_IDS], "public_sources": [],
        "interpretation": {"scope": "old_checkpoint_playback", "same_case_as_active": False, "retraining_shown": False, "disallowed_claims": ["x"]},
    }), encoding="utf-8")

    with pytest.raises(media.PreflightError, match="recording invariants changed"):
        media.load_protocol(protocol)


@pytest.mark.parametrize("mutation, message", [
    (lambda data: data.update(schema_version=2), "recording invariants changed"),
    (lambda data: data.update(protocol_id="same-looking-id"), "recording invariants changed"),
    (lambda data: data["interpretation"].update(same_case_as_active=True), "interpretation contract changed"),
])
def test_protocol_identity_and_interpretation_are_mandatory(tmp_path, mutation, message):
    source = Path(__file__).parents[1] / "bench/protocols/evidence_media_c01_v1.json"
    data = json.loads(source.read_text(encoding="utf-8"))
    mutation(data)
    candidate = tmp_path / "protocol.json"
    candidate.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(media.PreflightError, match=message):
        media.load_protocol(candidate)


def test_short_valid_mp4_fails_video_contract(tmp_path):
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg tools unavailable")
    video = tmp_path / "short.mp4"
    subprocess.run([
        "ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=size=1280x720:rate=50",
        "-t", "0.2", "-pix_fmt", "yuv420p", str(video),
    ], check=True, creationflags=media.CREATE_NO_WINDOW if os.name == "nt" else 0)

    probe = media._probe_video(video)
    with pytest.raises(RuntimeError, match="frame contract"):
        media._validate_video_contract(probe, {
            "width": 1280, "height": 720, "frame_rate": "50/1", "frame_count": 299,
            "duration_s": 5.98, "duration_tolerance_s": 0.02,
        })


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object lifecycle")
def test_windows_job_gates_child_launch_and_observes_empty_job(tmp_path):
    sentinel = tmp_path / "child-started"
    command = tmp_path / "command.json"
    command.write_text(json.dumps({
        "args": [sys.executable, "-c", f"from pathlib import Path; Path({str(sentinel)!r}).write_text('ok')"],
        "cwd": str(tmp_path),
    }), encoding="utf-8")
    job = media.WindowsJob()
    process = subprocess.Popen(
        [sys.executable, str(MODULE_PATH), "--_worker", str(command)],
        stdin=subprocess.PIPE,
        creationflags=media.CREATE_NO_WINDOW,
    )
    try:
        job.assign(process)
        assert not sentinel.exists()
        assert process.stdin is not None
        process.stdin.write(b"GO\n")
        process.stdin.close()
        assert process.wait(timeout=10) == 0
        assert job.wait_empty(5)
        assert job.active_processes() == 0
        assert sentinel.read_text(encoding="utf-8") == "ok"
    finally:
        if job.active_processes():
            job.terminate()
        job.close()


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object lifecycle")
def test_job_assignment_failure_cleans_gated_wrapper_without_launching_child(tmp_path, monkeypatch):
    _repo, isaac, protocol, checkpoint = _fixture(tmp_path)
    private = tmp_path / "private"
    private.mkdir()
    item = {"clip": protocol["clips"][0], "checkpoint": checkpoint, "params_dir": checkpoint.parent / "params"}
    monkeypatch.setattr(media.WindowsJob, "assign", lambda _self, _process: (_ for _ in ()).throw(OSError("denied")))

    with pytest.raises(RuntimeError, match="could not assign owned launcher"):
        media._record_clip(protocol, item, isaac, private, 60)

    execution = json.loads((private / "before/execution.json").read_text(encoding="utf-8"))
    assert execution["job_assigned"] is False
    assert execution["own_pid_cleanup"] is True
    assert execution["residual_processes_observed"] == 0
    assert not (private / "before/videos").exists()
