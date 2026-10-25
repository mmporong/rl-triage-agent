from __future__ import annotations

import hashlib
import importlib
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "bench"
sys.path.insert(0, str(BENCH))
R = importlib.import_module("record_vision_evidence_v1")
W = importlib.import_module("vision_play_worker_v1")


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _sha(path: Path, lf: bool = False) -> str:
    data = path.read_bytes().replace(b"\r\n", b"\n") if lf else path.read_bytes()
    return hashlib.sha256(data).hexdigest()


def _read_times(simulation_time: float, frame: int) -> dict[str, int | float]:
    return {
        "durationDenominator": R.FPS,
        "durationNumerator": 1,
        "externalTimeOfSimNs": frame * 20_000_000,
        "frameNumber": frame,
        "rationalTimeOfSimDenominator": R.FPS,
        "rationalTimeOfSimNumerator": frame,
        "sampleTimeOffsetInSimFrames": 0,
        "simulationTime": simulation_time,
        "simulationTimeMonotonic": simulation_time,
        "swhFrameNumber": frame,
        "systemTime": 1000.0 + simulation_time,
    }


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _job(
    tmp_path: Path,
    *,
    now: datetime | None = None,
    receipt_changes: dict[str, object] | None = None,
    mode: str = R.RECORD_MODE,
) -> Path:
    now = now or datetime.now(timezone.utc)  # noqa: UP017 - Python 3.10 호환
    repo = tmp_path / "repo"
    repo.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "test"], check=True)
    source = repo / "source.py"
    source.write_text("value = 1\n", encoding="utf-8", newline="\n")
    subprocess.run(["git", "-C", str(repo), "add", "source.py"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "fixture"], check=True)
    commit = _git(repo, "rev-parse", "HEAD")
    source_hash = _sha(source, lf=True)
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    protocol = tmp_path / "protocol.json"
    protocol_body = {
        "schema": R.PROTOCOL_SCHEMA, "method": R.METHOD, "mode": mode,
        "profile": R.fixed_profile(),
        "source_commit": commit, "source_files_lf": {"source.py": source_hash},
        "checkpoint_sha256": _sha(checkpoint),
        "resource": {"resource_id": "brev-instance-1", "gpu_uuid": "GPU-fixture-1"},
    }
    _write(protocol, protocol_body)
    job = {
        "schema": R.JOB_SCHEMA, "method": R.METHOD, "mode": mode, "job_id": "v04-test",
        "profile": R.fixed_profile(),
        "source": {"repo_root": str(repo), "commit": commit, "files_lf": {"source.py": source_hash}},
        "protocol": {"path": str(protocol), "sha256": _sha(protocol)},
        "checkpoint": {"path": str(checkpoint), "sha256": _sha(checkpoint), "logical_id": "go2-policy"},
        "resource": {"resource_id": "brev-instance-1", "gpu_uuid": "GPU-fixture-1"},
        "remote_gpu_admission": {"resource_parent_sha256": "a" * 64,
                                 "live_lease_proof_sha256": "b" * 64},
        "output_directory": str(tmp_path / "out"),
    }
    admission = tmp_path / "admission.json"
    receipt = {"schema": R.ADMISSION_SCHEMA, "status": "granted", "job_id": "v04-test",
               "resource_kind": "remote_gpu", "lease_id": "brev-gpu-1",
               "request_binding": R.request_binding(job),
               "issued_at_utc": (now - timedelta(seconds=5)).isoformat(),
               "expires_at_utc": (now + timedelta(seconds=R.EXECUTION_AUTHORIZATION_S + 5)).isoformat(),
               "resource_id": "brev-instance-1", "gpu_uuid": "GPU-fixture-1",
               "resource_parent_sha256": "a" * 64, "live_lease_proof_sha256": "b" * 64}
    receipt.update(receipt_changes or {})
    _write(admission, receipt)
    job["remote_gpu_admission"] = {
        "path": str(admission), "sha256": _sha(admission),
        "resource_parent_sha256": "a" * 64, "live_lease_proof_sha256": "b" * 64,
    }
    job_path = tmp_path / "job.json"
    _write(job_path, job)
    return job_path


def _sample(index: int, *, bad_step: bool = False, missing: bool = False) -> dict:
    if missing:
        raise RuntimeError("state missing")
    root = [index / 100.0, 0.0, 0.5]
    return {
        "frame_index": index, "capture_ordinal": index,
        "simulation_step": index + int(bad_step),
        "rgb": bytes([index % 251]) * (R.WIDTH * R.HEIGHT * 3),
        "state": {"command": [1.0, 0.0, 0.0], "motion": [0.8, 0.0, 0.0],
                  "pose": {"base_height_m": 0.5, "up_axis_z": 1.0},
                  "root_position_world_m": root,
                  "simulation_time_s": index / R.FPS,
                  "common_step_counter": index + int(bad_step)},
        "camera": {"eye_world_m": [root[0] + 3.0, 3.0, 2.5],
                   "lookat_world_m": [root[0], 0.0, 0.85]},
        "render_timing": {
            "read_times": _read_times(index / R.FPS, index),
            "reference_time": {"referenceTimeNumerator": index,
                               "referenceTimeDenominator": R.FPS},
        },
    }


class FakeEncoder:
    def __init__(self, *, fail_at: int | None = None):
        self.frames = 0
        self.fail_at = fail_at
        self.aborted = False

    def write(self, rgb: bytes) -> None:
        if self.fail_at == self.frames:
            raise RuntimeError("ffmpeg failed")
        assert len(rgb) == R.WIDTH * R.HEIGHT * 3
        self.frames += 1

    def finish(self) -> dict:
        return {"natural_exit": True, "frames_written": self.frames}

    def abort(self) -> dict:
        self.aborted = True
        return {"natural_exit": False, "frames_written": self.frames}


class _BytesSink:
    def __init__(self) -> None:
        self.data = bytearray()

    def write(self, value: bytes) -> None:
        self.data.extend(value)

    def close(self) -> None:
        return None


class _FakeProcess:
    def __init__(self, output: Path, returncode: int = 0) -> None:
        self.stdin = _BytesSink()
        self.stdout = _BytesSink()
        self.stderr = _BytesSink()
        self.pid = 123
        self.returncode = returncode
        self.output = output
        self.killed = False

    def communicate(self):
        if self.returncode == 0:
            self.output.write_bytes(b"encoded")
        return b"", b"encode failed" if self.returncode else b""

    def poll(self):
        return self.returncode

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9


def test_validate_job_binds_profile_source_protocol_checkpoint_and_admission(tmp_path: Path) -> None:
    job_path = _job(tmp_path)
    validated = R.validate_job(job_path)
    assert validated["checkpoint_sha256"] == _sha(tmp_path / "model.pt")
    for mutation in ("method", "mode", "profile", "source", "checkpoint", "resource", "job_id"):
        body = json.loads(job_path.read_text(encoding="utf-8"))
        if mutation == "method":
            body["method"] = "other"
        elif mutation == "mode":
            body["mode"] = "other"
        elif mutation == "profile":
            body["profile"]["camera"]["overlay_count"] = 1
        elif mutation == "source":
            body["source"]["files_lf"]["source.py"] = "0" * 64
        elif mutation == "checkpoint":
            body["checkpoint"]["sha256"] = "0" * 64
        elif mutation == "resource":
            body["resource"]["gpu_uuid"] = "GPU-other"
        else:
            body["job_id"] = "changed"
        changed = tmp_path / f"{mutation}.json"
        _write(changed, body)
        with pytest.raises((ValueError, FileExistsError)):
            R.validate_job(changed)


@pytest.mark.parametrize(
    ("name", "changes"),
    [
        ("stale", {"issued_at_utc": "2027-01-01T11:59:00+00:00"}),
        ("future", {"issued_at_utc": "2027-01-01T12:00:01+00:00"}),
        ("expired", {"expires_at_utc": "2027-01-01T12:14:59+00:00"}),
        ("wrong_vm", {"resource_id": "brev-instance-2"}),
        ("wrong_gpu", {"gpu_uuid": "GPU-other"}),
        ("parent_sha", {"resource_parent_sha256": "c" * 63}),
        ("live_sha", {"live_lease_proof_sha256": "d" * 63}),
    ],
)
def test_admission_rejects_stale_future_expired_wrong_resource_and_proof(
    tmp_path: Path, name: str, changes: dict[str, object]
) -> None:
    now = datetime(2027, 1, 1, 12, 0, tzinfo=timezone.utc)  # noqa: UP017 - Python 3.10 호환
    job_path = _job(tmp_path / name, now=now, receipt_changes=changes)
    with pytest.raises(ValueError, match="admission"):
        R.validate_job(job_path, now=now)


def test_admission_requires_timezone_aware_now(tmp_path: Path) -> None:
    aware = datetime(2027, 1, 1, 12, 0, tzinfo=timezone.utc)  # noqa: UP017 - Python 3.10 호환
    job_path = _job(tmp_path, now=aware)
    with pytest.raises(ValueError, match="timezone"):
        R.validate_job(job_path, now=datetime(2027, 1, 1, 12, 0))


def test_validate_job_requires_single_env_overlay_zero_and_exact_method(tmp_path: Path) -> None:
    job_path = _job(tmp_path)
    body = json.loads(job_path.read_text(encoding="utf-8"))
    body["profile"]["num_envs"] = 2
    body["profile"]["camera"]["overlay_count"] = 1
    _write(tmp_path / "bad_profile.json", body)
    with pytest.raises(ValueError, match="profile"):
        R.validate_job(tmp_path / "bad_profile.json")


def test_profile_fixes_play_camera_launch_and_intrinsics() -> None:
    profile = R.fixed_profile()
    assert profile["task"].endswith("Go2-Play-v0")
    assert profile["launcher"] == {"headless": True, "enable_cameras": True, "fast_shutdown": False}
    assert profile["camera"]["horizontal_fov_deg"] == 60.0
    assert profile["camera"]["capture_primitive"] == "replicator_orchestrator_step_v1"
    assert profile["camera"]["rt_subframes"] == 1
    assert profile["camera"]["pause_timeline"] is False
    assert profile["camera"]["delta_time_s"] == 0.0
    assert profile["camera"]["isaac_read_times_contract"] == "isaac_sim_4_5_outputs_11_v1"
    assert profile["camera"]["isaac_read_times_fields"] == list(R.ISAAC_READ_TIMES_FIELDS)
    assert profile["camera"]["overlay_count"] == 0


def test_rgba_non_contiguous_array_becomes_c_order_rgb24() -> None:
    source = np.zeros((R.HEIGHT, R.WIDTH * 2, 4), dtype=np.uint8)
    source[:, 0::2, 0] = 11
    source[:, 0::2, 1] = 22
    source[:, 0::2, 2] = 33
    source[:, 0::2, 3] = 44
    non_contiguous = source[:, 0::2, :]
    assert not non_contiguous.flags.c_contiguous
    result = R.rgb24_bytes(non_contiguous)
    assert len(result) == R.WIDTH * R.HEIGHT * 3
    assert result[:6] == bytes([11, 22, 33, 11, 22, 33])


def test_render_update_must_not_change_bound_robot_state() -> None:
    before = {"root": [0.0, 0.0, 0.5], "step": 7}
    R.validate_render_state_unchanged(before, dict(before))
    with pytest.raises(ValueError, match="render update"):
        R.validate_render_state_unchanged(before, {"root": [0.1, 0.0, 0.5], "step": 7})


def test_atomic_sample_rejects_rgb_timing_from_another_physics_state() -> None:
    sample = _sample(7)
    sample["render_timing"]["read_times"]["simulationTime"] = 0.16
    with pytest.raises(ValueError, match="IsaacReadTimes"):
        R.validate_atomic_sample(sample, 7)


def _smoke_capture(label: str, value: int, *, moved: bool = False) -> dict:
    state = {"root": [0.0, 0.0, 0.5], "step": 0}
    return {
        "label": label,
        "rgb": bytes([value]) * (R.WIDTH * R.HEIGHT * 3),
        "state_before": state,
        "state_after": {"root": [0.1 if moved else 0.0, 0.0, 0.5], "step": 0},
        "simulation_time_before_s": 1.0,
        "simulation_time_after_s": 1.0,
        "read_times": _read_times(1.0, 50),
        "reference_time": {"referenceTimeNumerator": 1, "referenceTimeDenominator": 1},
        "camera": {"eye_world_m": [3.0, 3.0, 2.5], "lookat_world_m": [0.0, 0.0, 0.85]},
    }


def test_render_sync_smoke_accepts_static_a_b_a_and_rejects_state_or_rgb_mismatch() -> None:
    captures = [_smoke_capture("warmup", 0), _smoke_capture("A", 1),
                _smoke_capture("B", 2), _smoke_capture("A", 1)]
    report = R.assess_render_sync_smoke(captures)
    assert report["status"] == "pass"
    assert report["policy_steps"] == report["capture_phase_physics_steps"] == 0
    assert report["universal_zero_lag_claim"] is False
    moved = list(captures)
    moved[2] = _smoke_capture("B", 2, moved=True)
    with pytest.raises(ValueError, match="robot state"):
        R.assess_render_sync_smoke(moved)
    same = list(captures)
    same[2] = _smoke_capture("B", 1)
    with pytest.raises(ValueError, match="구분"):
        R.assess_render_sync_smoke(same)


def test_record_and_smoke_preserve_all_sim45_read_times_fields() -> None:
    timing = {"read_times": _read_times(0.5, 25),
              "reference_time": {"referenceTimeNumerator": 25,
                                 "referenceTimeDenominator": 50}}
    normalized = R.validate_render_timing(timing, 0.5)
    assert list(normalized["read_times"]) == list(R.ISAAC_READ_TIMES_FIELDS)
    assert normalized["read_times"] == timing["read_times"]
    _, record = R.validate_atomic_sample(_sample(0), 0)
    assert record["render_timing"]["read_times"] == _read_times(0.0, 0)
    smoke = R.assess_render_sync_smoke(
        [_smoke_capture("warmup", 0), _smoke_capture("A", 1),
         _smoke_capture("B", 2), _smoke_capture("A", 1)]
    )
    assert smoke["captures"][1]["read_times"] == _read_times(1.0, 50)
    missing = {"read_times": dict(timing["read_times"]),
               "reference_time": dict(timing["reference_time"])}
    missing["read_times"].pop("durationNumerator")
    with pytest.raises(ValueError, match="IsaacReadTimes"):
        R.validate_render_timing(missing, 0.5)


def test_render_sync_smoke_cpu_adapter_uses_fixed_warmup_a_b_a() -> None:
    calls: list[tuple[str, list[float]]] = []

    def capture(label: str, offset: list[float]):
        calls.append((label, offset))
        value = 2 if label == "B" else (0 if label == "warmup" else 1)
        return _smoke_capture(label, value)

    captures, report = R.collect_render_sync_smoke(capture)
    assert len(captures) == 4
    assert [label for label, _ in calls] == ["warmup", "A", "B", "A"]
    assert calls[1][1] == calls[3][1]
    assert calls[1][1] != calls[2][1]
    assert report["status"] == "pass"


def test_record_and_smoke_capture_step_is_one_subframe_unpaused_and_zero_delta() -> None:
    calls: list[dict[str, object]] = []

    class Orchestrator:
        def step(self, **kwargs) -> None:
            calls.append(kwargs)

    W._capture_render_step(Orchestrator(), R.fixed_profile()["camera"])
    assert calls == [{"rt_subframes": 1, "pause_timeline": False, "delta_time": 0.0}]


def test_shared_capture_primitive_reads_same_rgb_and_timing_annotators() -> None:
    calls: list[tuple[str, object]] = []

    class Annotator:
        def __init__(self, name: str, value: object) -> None:
            self.name = name
            self.value = value

        def attach(self, product: object) -> None:
            calls.append((f"attach:{self.name}", product))

        def get_data(self, **kwargs):
            calls.append((f"read:{self.name}", kwargs))
            return self.value

    rgb = np.zeros((R.HEIGHT, R.WIDTH, 4), dtype=np.uint8)
    values = {
        "rgb": Annotator("rgb", rgb),
        "IsaacReadTimes": Annotator(
            "IsaacReadTimes",
            _read_times(0.0, 0),
        ),
        "ReferenceTime": Annotator(
            "ReferenceTime", {"referenceTimeNumerator": 0, "referenceTimeDenominator": 50}
        ),
    }

    class Registry:
        @staticmethod
        def get_annotator(name: str):
            return values[name]

    class ReferenceRegistry:
        @staticmethod
        def get(name: str):
            return values[name]

    class Orchestrator:
        def step(self, **kwargs) -> None:
            calls.append(("step", kwargs))

    class Rep:
        AnnotatorRegistry = Registry
        annotators = ReferenceRegistry
        orchestrator = Orchestrator()

    annotators = W._create_capture_annotators(Rep, "product")
    class Sim:
        def set_camera_view(self, **kwargs) -> None:
            calls.append(("camera", kwargs))

    class Base:
        sim = Sim()

    camera_profile = R.fixed_profile()["camera"]
    captured = W._set_camera_and_capture(
        Base, Rep, annotators, camera_profile, [3.0, 3.0, 2.5], [0.0, 0.0, 0.85]
    )
    assert len(captured["rgb"]) == R.WIDTH * R.HEIGHT * 3
    assert ("step", {"rt_subframes": 1, "pause_timeline": False, "delta_time": 0.0}) in calls
    assert ("camera", {"eye": [3.0, 3.0, 2.5], "target": [0.0, 0.0, 0.85],
                       "camera_prim_path": "/OmniverseKit_Persp"}) in calls
    assert [name for name, _ in calls if name.startswith("read:")] == [
        "read:rgb", "read:IsaacReadTimes", "read:ReferenceTime"
    ]


def test_capture_lockstep_aligns_frame_state_and_aborts_on_missing_or_ffmpeg_failure() -> None:
    encoder = FakeEncoder()
    records, receipt = R.capture_lockstep(_sample, encoder)
    assert len(records) == R.FRAME_COUNT
    assert receipt["frames_written"] == R.FRAME_COUNT
    with pytest.raises(ValueError, match="lockstep"):
        R.capture_lockstep(lambda index: _sample(index, bad_step=index == 4), FakeEncoder())
    missing_encoder = FakeEncoder()
    with pytest.raises(RuntimeError, match="missing"):
        R.capture_lockstep(lambda index: _sample(index, missing=index == 3), missing_encoder)
    assert missing_encoder.aborted
    failed_encoder = FakeEncoder(fail_at=2)
    with pytest.raises(RuntimeError, match="ffmpeg"):
        R.capture_lockstep(_sample, failed_encoder)
    assert failed_encoder.aborted


def test_raw_encoder_records_natural_exit_and_rejects_ffmpeg_failure(tmp_path: Path) -> None:
    output = tmp_path / "ok.mp4"
    process = _FakeProcess(output)
    encoder = R.RawRgbFfmpegEncoder(output, popen=lambda *args, **kwargs: process)
    encoder.write(b"rgb")
    receipt = encoder.finish()
    assert receipt["natural_exit"] is True
    assert receipt["frames_written"] == 1
    assert receipt["returncode"] == 0
    failed_output = tmp_path / "failed.mp4"
    failed = _FakeProcess(failed_output, returncode=2)
    failed_encoder = R.RawRgbFfmpegEncoder(failed_output, popen=lambda *args, **kwargs: failed)
    failed_encoder.write(b"rgb")
    with pytest.raises(RuntimeError, match="ffmpeg encode"):
        failed_encoder.finish()


def test_probe_uses_actual_pts_and_rejects_offset(tmp_path: Path) -> None:
    def runner(values: list[float]):
        return lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, json.dumps({"frames": [{"best_effort_timestamp_time": str(value)} for value in values]}), ""
        )

    pts = [10.0 + index / R.FPS for index in range(R.FRAME_COUNT)]
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    assert R.probe_frame_pts(video, runner=runner(pts))[50] == 1.0
    pts[50] += 0.01
    with pytest.raises(ValueError, match="PTS"):
        R.probe_frame_pts(video, runner=runner(pts))


def test_builds_two_six_sample_v01_inputs_from_actual_pts(tmp_path: Path) -> None:
    validated = R.validate_job(_job(tmp_path))
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    records = []
    for index in range(R.FRAME_COUNT):
        _, record = R.validate_atomic_sample(_sample(index), index)
        records.append({**record, "actual_timestamp_s": index / R.FPS})
    artifacts = R.build_trace_and_v01_specs(
        validated, video, records, {"natural_exit": True, "frames_written": R.FRAME_COUNT},
        {"source_sha256": _sha(video), "frame_count": R.FRAME_COUNT,
         "relative_actual_pts_s": [index / R.FPS for index in range(R.FRAME_COUNT)]},
        tmp_path / "evidence"
    )
    for window in (1, 2):
        measurement = json.loads(artifacts[f"measurement_{window}"].read_text(encoding="utf-8"))
        assert len(measurement["samples"]) == 6
        assert measurement["samples"][0]["timestamp_s"] == float(1 if window == 1 else 7)
        assert measurement["samples"][0]["motion"]["linear_x_mps"] == 0.8
    trace = json.loads(artifacts["trace"].read_text(encoding="utf-8"))
    assert trace["records"][50]["camera"]["eye_world_m"][0] == 3.5
    assert trace["render_alignment"]["zero_update_pixel_lag_confirmed"] is False
    assert trace["render_alignment"]["gpu_smoke_required_before_model_use"] is True
    assert "caption" not in json.dumps(trace)


def test_worker_import_is_cpu_safe_and_failure_written_before_close(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert "isaaclab.app" not in W.__dict__
    job_path = _job(tmp_path)
    events: list[str] = []

    class App:
        def close(self) -> None:
            events.append("close")
            failure = tmp_path / "out" / "failure.json"
            assert failure.is_file()

    def fail(validated, app):
        del validated, app
        raise RuntimeError("capture failed")

    assert W.execute(job_path, app_factory=lambda: App(), runner=fail) == 1
    failure = json.loads((tmp_path / "out" / "failure.json").read_text(encoding="utf-8"))
    assert failure["stage"] == "record_play"
    assert failure["completed"] is False
    assert events == ["close"]


def test_worker_success_writes_result_before_close(tmp_path: Path) -> None:
    job_path = _job(tmp_path)

    class App:
        def close(self) -> None:
            assert (tmp_path / "out" / "result.json").is_file()

    def succeed(validated, app):
        del app
        video = validated["output_directory"] / "same_play.mp4"
        video.parent.mkdir(parents=True)
        video.write_bytes(b"video")
        trace = validated["output_directory"] / "trace.json"
        trace.write_text("{}\n", encoding="utf-8")
        return {"trace": trace}, video

    assert W.execute(job_path, app_factory=lambda: App(), runner=succeed) == 0
    result = json.loads((tmp_path / "out" / "result.json").read_text(encoding="utf-8"))
    assert result["completed"] is True
    assert result["fixed_evaluation_used_as_behavior_ground_truth"] is False
    assert result["artifacts"]["trace"]["sha256"] == _sha(tmp_path / "out" / "trace.json")


def test_worker_dispatches_render_sync_smoke_mode_without_policy_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_path = _job(tmp_path, mode=R.RENDER_SYNC_SMOKE_MODE)
    calls: list[str] = []

    class App:
        def close(self) -> None:
            calls.append("close")

    def smoke_runner(validated, app):
        del app
        calls.append(validated["job"]["mode"])
        output = validated["output_directory"]
        output.mkdir(parents=True)
        report = output / "render_sync_smoke.json"
        report.write_text("{}\n", encoding="utf-8")
        return {"render_sync_smoke": report}, report

    monkeypatch.setattr(W, "run_render_sync_smoke", smoke_runner)
    assert W.execute(job_path, app_factory=lambda: App()) == 0
    result = json.loads((tmp_path / "out" / "result.json").read_text(encoding="utf-8"))
    assert result["mode"] == R.RENDER_SYNC_SMOKE_MODE
    assert result["video_sha256"] is None
    assert calls == [R.RENDER_SYNC_SMOKE_MODE, "close"]
