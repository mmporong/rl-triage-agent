from __future__ import annotations

import hashlib
import importlib
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / "bench", ROOT / "evals", ROOT / "src"):
    sys.path.insert(0, str(folder))
V = importlib.import_module("vision_brev_backend_v1")
R = importlib.import_module("record_vision_evidence_v1")


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _plan() -> list[dict]:
    rows = []
    for seed in (301, 302):
        for arm in ("NONE", "F1", "CORRECTED"):
            job_id = f"s{seed}-{arm.lower()}"
            rows.append({
                "job_id": job_id, "arm": arm, "seed": seed, "run_name": job_id,
                "stage_relative": f"training/{job_id}", "iterations": 300, "num_envs": 4096,
                "source_job_id": f"s{seed}-f1" if arm == "CORRECTED" else None,
            })
    return rows


class FakeRecovery:
    def __init__(self, tmp_path: Path, *, gpu: list[int] | None = None, popen=None):
        self.calls: list[tuple] = []
        self.lab = tmp_path
        self.python = Path(sys.executable)
        self.read_gpu = lambda: list(gpu or [])
        self.popen = popen
        self.snapshot = {"source": "fixed", "interpreter": str(self.python)}
        self.protocol = {"protocol_id": "vision-test", "training_plan": _plan()}

    def _source_snapshot(self):
        return dict(self.snapshot)

    def train(self, stage, arm, seed, run_name, corrections, iterations, source_checkpoint_sha256):
        self.calls.append((stage, arm, seed, run_name, corrections, iterations, source_checkpoint_sha256))
        stage.mkdir(parents=True)
        checkpoint = stage / "model.pt"
        checkpoint.write_bytes(f"{arm}-{seed}".encode())
        telemetry = stage / "telemetry.json"
        _write(telemetry, {"summary": {}})
        readback = stage / "training.readback.json"
        readback_body = {"remaining_faults": []}
        _write(readback, readback_body)
        records = stage / "optimizer.records.jsonl"
        records.write_text("", encoding="utf-8")
        execution = stage / "execution.json"
        execution_body = {"completed": True, "mode": "train"}
        _write(execution, execution_body)
        manifest = stage / "training.manifest.json"
        manifest_body = {"schema": "fixture", "seed": seed, "arm": arm,
                         "readback_path": str(readback), "records_path": str(records)}
        _write(manifest, manifest_body)
        return {"seed": seed, "arm": arm,
                "checkpoint": {"path": str(checkpoint), "sha256": _sha(checkpoint)},
                "telemetry": {"path": str(telemetry), "sha256": _sha(telemetry), "summary": {}},
                "manifest_path": str(manifest), "manifest_sha256": _sha(manifest),
                "manifest": manifest_body, "readback": readback_body,
                "execution": execution_body,
                "source_checkpoint_sha256": source_checkpoint_sha256}

    def probe(self, *args, **kwargs):
        return {"probe": [args, kwargs]}

    def evaluate(self, *args, **kwargs):
        return {"evaluate": [args, kwargs]}


def _backend(tmp_path: Path, recovery: FakeRecovery | None = None,
             plan: list[dict] | None = None) -> tuple[object, Path]:
    rows = plan or _plan()
    recovery = recovery or FakeRecovery(tmp_path)
    recovery.protocol = {"protocol_id": "vision-test", "training_plan": rows}
    V.ROOT = tmp_path / "repo-root"
    runtime = V.ROOT / ".triage-runtime/vision-brev/vision-test"
    return V.VisionBrevBackend(recovery, runtime, rows), runtime


def _approval(tmp_path: Path, source_sha: str) -> dict:
    from rl_triage.correction_ledger import CorrectionLedger

    path = tmp_path / "correction.jsonl"
    ledger = CorrectionLedger(path)
    proposed = ledger.propose(
        "a" * 64, "reward", "b" * 64, "restore_body_frame_reward", "restart",
        {"case": "case_01", "arm": "corrected"}, 3600,
        evidence={"source_checkpoint_sha256": source_sha},
    )
    ledger.approve(proposed["request_id"], "operator")
    ledger.consume(proposed["request_id"], "a" * 64)
    return {"path": str(path), "sha256": _sha(path), "request_id": proposed["request_id"],
            "loop_state_digest": "a" * 64}


def test_fixed_six_plan_global_reservation_cache_pending_and_tamper(tmp_path: Path) -> None:
    recovery = FakeRecovery(tmp_path)
    backend, runtime = _backend(tmp_path, recovery)
    first = backend.train("s301-none")
    assert first["arm"] == "NONE" and len(recovery.calls) == 1
    assert backend.train("s301-none") == first and len(recovery.calls) == 1
    with pytest.raises(ValueError, match="등록되지 않은"):
        backend.train("new-tag")

    pending = "s302-none"
    base = {"schema": V.LEDGER_SCHEMA, "job_id": pending, "plan_sha256": backend.plan_sha256}
    backend._append({**base, "event": "reserved", "stage_relative": f"training/{pending}"})
    backend._append({**base, "event": "started", "approval_reference": None})
    assert backend.train(pending)["reason"] == "training_pending"
    assert len(recovery.calls) == 1

    artifact = runtime / "training/s301-none/adapter_artifact.json"
    artifact.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="변조"):
        backend.train("s301-none")


def test_failed_training_reservation_cannot_be_retried_or_evaded(tmp_path: Path) -> None:
    recovery = FakeRecovery(tmp_path)
    calls = 0

    def fail(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("training failed")

    recovery.train = fail
    backend, _ = _backend(tmp_path, recovery)
    with pytest.raises(RuntimeError, match="training failed"):
        backend.train("s301-none")
    assert backend.train("s301-none")["reason"] == "training_pending"
    assert calls == 1
    with pytest.raises(ValueError, match="등록되지 않은"):
        backend.train("s301-none-retry")


def test_corrected_requires_consumed_controller_reference_and_source_artifact(tmp_path: Path) -> None:
    recovery = FakeRecovery(tmp_path)
    backend, _ = _backend(tmp_path, recovery)
    assert backend.train("s301-corrected")["reason"] == "source_training_not_completed"
    source = backend.train("s301-f1")
    with pytest.raises(ValueError, match="승인 참조"):
        backend.train("s301-corrected")
    approval = _approval(tmp_path, source["checkpoint"]["sha256"])
    corrected = backend.train("s301-corrected", approval_reference=approval)
    assert corrected["arm"] == "CORRECTED"
    assert recovery.calls[-1][-1] == source["checkpoint"]["sha256"]
    assert len(json.loads((tmp_path / "correction.jsonl").read_text().splitlines()[-1])["event"]) > 0
    assert [json.loads(line)["event"] for line in (tmp_path / "correction.jsonl").read_text().splitlines()] == [
        "proposed", "approved", "consumed"
    ]
    from rl_triage.correction_ledger import CorrectionLedger
    CorrectionLedger(Path(approval["path"])).receipt(approval["request_id"], "tag", "recovered", None)
    assert backend.train("s301-corrected") == corrected
    Path(source["checkpoint"]["path"]).write_bytes(b"tampered-parent")
    with pytest.raises(ValueError, match="checkpoint SHA"):
        backend.train("s301-corrected")

    other_recovery = FakeRecovery(tmp_path)
    other_recovery.protocol["protocol_id"] = "vision-other"
    V.ROOT = tmp_path / "repo-root-other"
    other = V.VisionBrevBackend(
        other_recovery, V.ROOT / ".triage-runtime/vision-brev/vision-other", _plan()
    )
    source2 = other.train("s301-f1")
    done = _approval(tmp_path / "done", source2["checkpoint"]["sha256"])
    CorrectionLedger(Path(done["path"])).receipt(done["request_id"], "tag", "unknown", None)
    done["sha256"] = _sha(Path(done["path"]))
    with pytest.raises(ValueError, match="consumed"):
        other.train("s301-corrected", approval_reference=done)


def _record_job(tmp_path: Path, stage: Path) -> dict:
    repo = tmp_path / "repo"
    repo.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "t@example.com"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "t"], check=True)
    source = repo / "source.py"
    source.write_text("x=1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "source.py"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "fixture"], check=True)
    commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True,
                            capture_output=True, text=True).stdout.strip()
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"model")
    source_sha = hashlib.sha256(source.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    protocol = tmp_path / "vision_protocol.json"
    _write(protocol, {"schema": R.PROTOCOL_SCHEMA, "method": R.METHOD, "mode": R.RECORD_MODE,
                      "profile": R.fixed_profile(),
                      "source_commit": commit, "source_files_lf": {"source.py": source_sha},
                      "checkpoint_sha256": _sha(checkpoint),
                      "resource": {"resource_id": "brev-test", "gpu_uuid": "GPU-test"}})
    parent_sha, lease_sha = "d" * 64, "e" * 64
    job = {"schema": R.JOB_SCHEMA, "method": R.METHOD, "mode": R.RECORD_MODE,
           "job_id": "record-1",
           "profile": R.fixed_profile(),
           "source": {"repo_root": str(repo), "commit": commit, "files_lf": {"source.py": source_sha}},
           "protocol": {"path": str(protocol), "sha256": _sha(protocol)},
           "checkpoint": {"path": str(checkpoint), "sha256": _sha(checkpoint), "logical_id": "policy"},
           "resource": {"resource_id": "brev-test", "gpu_uuid": "GPU-test"},
           "remote_gpu_admission": {"resource_parent_sha256": parent_sha,
                                    "live_lease_proof_sha256": lease_sha},
           "output_directory": str(stage / "output")}
    admission = tmp_path / "admission.json"
    now = datetime.now(timezone.utc)  # noqa: UP017 - Python 3.10 회귀
    _write(admission, {"schema": R.ADMISSION_SCHEMA, "status": "granted", "job_id": "record-1",
                       "resource_kind": "remote_gpu", "lease_id": "lease-1",
                       "request_binding": R.request_binding(job),
                       "issued_at_utc": now.isoformat(),
                       "expires_at_utc": (now + timedelta(seconds=1200)).isoformat(),
                       "resource_id": "brev-test", "gpu_uuid": "GPU-test",
                       "resource_parent_sha256": parent_sha, "live_lease_proof_sha256": lease_sha})
    job["remote_gpu_admission"].update({"path": str(admission), "sha256": _sha(admission)})
    return job


class FakeProcess:
    def __init__(self, job: dict, *, code: int = 0):
        self.pid = os.getpid()
        self.returncode = code
        self.job = job

    def wait(self, timeout):
        assert timeout == V.RECORD_TIMEOUT_S
        if self.returncode == 0:
            output = Path(self.job["output_directory"])
            job_path = output.parent / "job.json"
            validated = R.validate_job(job_path)
            (output / "evidence").mkdir(parents=True)
            video = output / "same_play.mp4"
            video.write_bytes(b"video")
            video_sha = _sha(video)
            artifacts = {}
            trace = output / "evidence/raw_trace.json"
            records = [{"actual_timestamp_s": frame / R.FPS,
                        "state": {"command": [1.0, 0.0, 0.0], "motion": [0.5, 0.0, 0.0],
                                  "pose": {"base_height_m": 0.3, "up_axis_z": 1.0}}}
                       for frame in range(R.FRAME_COUNT)]
            _write(trace, {"schema": R.TRACE_SCHEMA, "method": R.METHOD,
                           "job_id": self.job["job_id"], "profile": R.fixed_profile(),
                           "profile_sha256": R.digest(R.fixed_profile()), "video_sha256": video_sha,
                           "checkpoint_sha256": self.job["checkpoint"]["sha256"],
                           "time_basis": R.fixed_profile()["time_basis"],
                           "frame_count": R.FRAME_COUNT, "records": records,
                           "encoder_receipt": {"natural_exit": True,
                                               "frames_written": R.FRAME_COUNT},
                           "timing_receipt": {"source_sha256": video_sha,
                                              "frame_count": R.FRAME_COUNT, "fps": R.FPS,
                                              "frame_period_s": R.FRAME_PERIOD_S,
                                              "relative_actual_pts_s": [row["actual_timestamp_s"]
                                                                        for row in records]},
                           "render_alignment": {"camera_pose_applied_before_render": True,
                                                "capture_primitive": R.fixed_profile()["camera"]["capture_primitive"],
                                                "rt_subframes": R.fixed_profile()["camera"]["rt_subframes"],
                                                "pause_timeline": R.fixed_profile()["camera"]["pause_timeline"],
                                                "delta_time_s": R.fixed_profile()["camera"]["delta_time_s"],
                                                "robot_state_unchanged_across_render_enforced": True,
                                                "rgb_timing_matches_state_enforced": True,
                                                "annotator_cpu_copy": True,
                                                "zero_update_pixel_lag_confirmed": False,
                                                "gpu_smoke_required_before_model_use": True}})
            artifacts["trace"] = trace
            for index, timestamps in enumerate(R.WINDOWS, 1):
                play_id = f"{self.job['job_id']}-w{index}"
                measurement = output / f"evidence/window_{index}_measurements.json"
                _write(measurement, {"schema": R.MEASUREMENT_SCHEMA, "play_run_id": play_id,
                                     "video_sha256": video_sha,
                                     "checkpoint_sha256": self.job["checkpoint"]["sha256"],
                                     "command_name": "base_velocity",
                                     "coordinate_frames": {"command": "body", "motion": "body"},
                                     "samples": [{"timestamp_s": value,
                                                  "command": {"linear_x_mps": 1.0,
                                                              "linear_y_mps": 0.0,
                                                              "yaw_rate_rps": 0.0},
                                                  "motion": {"linear_x_mps": 0.5,
                                                             "linear_y_mps": 0.0,
                                                             "yaw_rate_rps": 0.0},
                                                  "pose": {"base_height_m": 0.3,
                                                           "up_axis_z": 1.0}}
                                                 for value in timestamps]})
                spec = output / f"evidence/window_{index}_v01_spec.json"
                _write(spec, {"schema": R.PREPARE_SCHEMA, "play_run_id": play_id,
                              "source_lineage_id": f"{self.job['job_id']}-same-play",
                              "parent_source_lineage_id": None,
                              "video": {"path": str(video.resolve()), "logical_id": "same_play_video",
                                        "play_run_id": play_id},
                              "checkpoint": {"path": str(Path(self.job["checkpoint"]["path"]).resolve()),
                                             "logical_id": self.job["checkpoint"]["logical_id"],
                                             "play_run_id": play_id},
                              "frames": [{"timestamp_s": value, "play_run_id": play_id,
                                          "video_sha256": video_sha} for value in timestamps],
                              "measurement": {"path": str(measurement.resolve()),
                                              "logical_id": f"same_play_measurements_w{index}",
                                              "play_run_id": play_id},
                              "play": {"task": R.TASK,
                                       "camera": R.fixed_profile()["camera"]["identifier"],
                                       "command_name": "base_velocity", "command_frame": "body",
                                       "motion_frame": "body",
                                       "settings_sha256": R.digest(R.fixed_profile())}})
                artifacts[f"measurement_{index}"] = measurement
                artifacts[f"spec_{index}"] = spec
            _write(output / "result.json", R.result_body(validated, artifacts, video))
        return self.returncode

    def poll(self):
        return self.returncode

    def terminate(self):
        self.returncode = -15

    def kill(self):
        self.returncode = -9


def test_record_uses_selected_interpreter_environment_cleanup_and_cache(tmp_path: Path, monkeypatch) -> None:
    launches = []

    def popen(argv, **kwargs):
        launches.append((argv, kwargs))
        return FakeProcess(job)

    recovery = FakeRecovery(tmp_path, popen=popen)
    backend, runtime = _backend(tmp_path, recovery)
    stage = runtime / "record/record-1"
    job = _record_job(tmp_path, stage)
    monkeypatch.setattr(V, "_process_identity", lambda pid: {"pid": pid, "pgid": pid, "sid": pid,
                                                             "start_ticks": 1,
                                                             "posix_group_verified": True})
    monkeypatch.setattr(V, "_safe_cleanup", lambda identity, process: True)
    result = backend.record(job)
    assert result["execution"]["completed"] is True
    assert result["status"] == "capture_complete_gpu_smoke_pending"
    assert result["model_use_ready"] is False
    argv, kwargs = launches[0]
    assert argv[:3] == [str(recovery.python), "-B", str(V.VISION_WORKER)]
    assert kwargs["start_new_session"] is True
    assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "0"
    assert backend.record(job)["result"] == result["result"]
    assert len(launches) == 1
    artifact = stage / "output/evidence/raw_trace.json"
    artifact.write_text("{\"changed\":true}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="크기 또는 SHA"):
        backend.record(job)
    artifact.write_text("{}\n", encoding="utf-8")
    (stage / "output/shutdown_failure.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="실패 증거"):
        backend.record(job)
    (stage / "output/shutdown_failure.json").unlink()
    (stage / "output/result.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="변조"):
        backend.record(job)


def test_record_blocks_busy_gpu_and_never_retries_invalid_stage(tmp_path: Path, monkeypatch) -> None:
    recovery = FakeRecovery(tmp_path, gpu=[91], popen=lambda *a, **k: pytest.fail("spawned"))
    backend, runtime = _backend(tmp_path, recovery)
    stage = runtime / "record/record-1"
    job = _record_job(tmp_path, stage)
    monkeypatch.setattr(V.B, "cleanup_group", lambda pid, process: True)
    with pytest.raises(V.B.BackendError, match="ForeignGPUBusy"):
        backend.record(job)
    with pytest.raises(V.B.BackendError, match="ForeignGPUBusy"):
        backend.record(job)


def test_record_pending_stage_rejects_different_job(tmp_path: Path) -> None:
    backend, runtime = _backend(tmp_path, FakeRecovery(tmp_path))
    stage = runtime / "record/record-1"
    job = _record_job(tmp_path, stage)
    stage.mkdir(parents=True)
    _write(stage / "job.json", job)
    assert backend.record(job)["reason"] == "record_pending"
    changed = json.loads(json.dumps(job))
    changed["checkpoint"]["logical_id"] = "other-policy"
    with pytest.raises(ValueError, match="다른 job"):
        backend.record(changed)


def test_plan_rejects_cap_bypass_and_record_rejects_output_escape(tmp_path: Path) -> None:
    bad = _plan() + [{**_plan()[0], "job_id": "extra", "stage_relative": "training/extra"}]
    with pytest.raises(ValueError, match="여섯"):
        _backend(tmp_path, FakeRecovery(tmp_path), bad)
    backend, runtime = _backend(tmp_path / "valid", FakeRecovery(tmp_path / "valid"))
    stage = runtime / "record/record-1"
    job = _record_job(tmp_path / "other", stage)
    job["output_directory"] = str(tmp_path / "escaped")
    with pytest.raises(V.B.BackendError):
        backend.record(job)


def test_record_timeout_cleans_owned_group_and_seals_failure(tmp_path: Path, monkeypatch) -> None:
    cleaned: list[int] = []

    class TimedOut(FakeProcess):
        def wait(self, timeout):
            raise subprocess.TimeoutExpired("worker", timeout)

    recovery = FakeRecovery(tmp_path, popen=lambda *a, **k: TimedOut(job))
    backend, runtime = _backend(tmp_path, recovery)
    stage = runtime / "record/record-1"
    job = _record_job(tmp_path, stage)
    monkeypatch.setattr(V, "_process_identity", lambda pid: {"pid": pid, "pgid": pid, "sid": pid,
                                                             "start_ticks": 1,
                                                             "posix_group_verified": True})
    monkeypatch.setattr(V, "_safe_cleanup", lambda identity, process: cleaned.append(identity["pid"]) or True)
    with pytest.raises(V.B.BackendError):
        backend.record(job)
    receipt = json.loads((stage / "execution.json").read_text(encoding="utf-8"))
    assert receipt["timed_out"] is True and receipt["cleanup_verified"] is True
    assert cleaned


def test_from_runtime_passes_new_protocol_and_environment_to_v4_backend(tmp_path: Path, monkeypatch) -> None:
    captured = {}

    def construct(lab, python, protocol, anchor, **kwargs):
        captured.update(lab=lab, python=python, protocol=protocol, anchor=anchor, kwargs=kwargs)
        return FakeRecovery(tmp_path)

    monkeypatch.setattr(V, "SafeRecoveryBackend", construct)
    protocol = {
        "source_sha256_lf": {"evals/vision_brev_backend_v1.py": "a" * 64},
        "runtime_source_sha256_lf": {"train.py": "b" * 64},
        "python_package_source_sha256_lf": {"ppo.py": "c" * 64},
        "environment": {"python": "3.10"},
        "protocol_path": str((tmp_path / "new-protocol.json").resolve()),
        "protocol_sha256_lf": "d" * 64,
        "protocol_id": "vision-test", "training_plan": _plan(),
    }
    V.ROOT = tmp_path / "repo-root"
    runtime = V.ROOT / ".triage-runtime/vision-brev/vision-test"
    backend = V.VisionBrevBackend.from_runtime(
        tmp_path / "lab", Path(sys.executable), protocol, "e" * 40,
        runtime, _plan(), read_gpu=list,
    )
    assert isinstance(backend, V.VisionBrevBackend)
    assert captured["protocol"] == protocol and captured["anchor"] == "e" * 40
    assert "read_gpu" in captured["kwargs"]


def test_from_runtime_builds_strict_reader_from_resource_proof(tmp_path: Path, monkeypatch) -> None:
    captured = {}
    sentinel = list

    def construct_reader(proof, audit_root):
        captured.update(proof=proof, audit_root=audit_root)
        return sentinel

    def construct(lab, python, protocol, anchor, **kwargs):
        captured["read_gpu"] = kwargs["read_gpu"]
        return FakeRecovery(tmp_path)

    monkeypatch.setattr(V.REMOTE, "StrictRemoteGpuReader", construct_reader)
    monkeypatch.setattr(V, "SafeRecoveryBackend", construct)
    protocol = {"protocol_id": "vision-test", "training_plan": _plan()}
    proof = {key: "value" for key in V.REMOTE.CAPTURE_KEYS}
    V.ROOT = tmp_path / "repo-root"
    runtime = V.ROOT / ".triage-runtime/vision-brev/vision-test"
    V.VisionBrevBackend.from_runtime(
        tmp_path / "lab", Path(sys.executable), protocol, "e" * 40, runtime, _plan(),
        resource_proof=proof,
    )
    assert captured["proof"] == proof
    assert captured["audit_root"] == runtime / "gpu-observations"
    assert captured["read_gpu"] is sentinel


@pytest.mark.parametrize("target", ["telemetry", "manifest", "readback", "records", "execution"])
def test_training_cache_rechecks_complete_artifact_inventory(tmp_path: Path, target: str) -> None:
    backend, runtime = _backend(tmp_path, FakeRecovery(tmp_path))
    backend.train("s301-none")
    stage = runtime / "training/s301-none"
    paths = {
        "telemetry": stage / "telemetry.json", "manifest": stage / "training.manifest.json",
        "readback": stage / "training.readback.json", "records": stage / "optimizer.records.jsonl",
        "execution": stage / "execution.json",
    }
    paths[target].write_bytes(paths[target].read_bytes() + b"tamper")
    with pytest.raises((ValueError, json.JSONDecodeError), match="SHA|artifact|본문|Expecting|Extra data"):
        backend.train("s301-none")


def test_gpu_execution_lock_is_shared_across_train_probe_evaluate_and_record(tmp_path: Path) -> None:
    entered, release = threading.Event(), threading.Event()
    errors: list[BaseException] = []
    recovery = FakeRecovery(tmp_path)
    original = recovery.train

    def slow(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)

    recovery.train = slow
    backend, runtime = _backend(tmp_path, recovery)
    record_job = _record_job(tmp_path / "record-input", runtime / "record/record-1")
    def run_training() -> None:
        try:
            backend.train("s301-none")
        except BaseException as error:  # noqa: BLE001 - thread 실패를 test 본문으로 전달한다.
            errors.append(error)

    thread = threading.Thread(target=run_training, daemon=True)
    thread.start()
    assert entered.wait(5)
    for call in (
        lambda: backend.train("s302-none"),
        lambda: backend.probe("stage", {}, "P_reward", "run"),
        lambda: backend.evaluate("stage", {}, "run"),
        lambda: backend.record(record_job),
    ):
        with pytest.raises(V.B.BackendError, match="GPUExecutionBusy"):
            call()
    release.set()
    thread.join(5)
    assert not thread.is_alive() and errors == []
    assert [row for row in backend._events() if row["job_id"] == "s302-none"] == []
    assert backend.train("s302-none")["arm"] == "NONE"


def test_reentrant_action_admission_checks_busy_before_training_ledger(tmp_path: Path) -> None:
    gpu = [77]
    recovery = FakeRecovery(tmp_path)
    recovery.read_gpu = lambda: list(gpu)
    backend, runtime = _backend(tmp_path, recovery)
    with pytest.raises(V.B.BackendError, match="ForeignGPUBusy"), backend.action_admission():
        backend.train("s301-none")
    assert not (runtime / "training.ledger.jsonl").exists()
    gpu.clear()
    with backend.action_admission():
        assert backend.train("s301-none")["arm"] == "NONE"


def test_training_cache_reaudits_strict_gpu_observation_references(tmp_path: Path, monkeypatch) -> None:
    class FakeStrictReader:
        def __call__(self):
            return []

    monkeypatch.setattr(V.REMOTE, "StrictRemoteGpuReader", FakeStrictReader)
    recovery = FakeRecovery(tmp_path)
    recovery.read_gpu = FakeStrictReader()
    recovery.vision_resource_proof = {"resource_id": "instance-1", "gpu_uuid": "GPU-fixture"}
    original = recovery.train

    def train(*args, **kwargs):
        artifact = original(*args, **kwargs)
        execution = {
            **artifact["execution"], "schema": "brev_recovery_posix_execution_v1",
            "gpu_before_observation": {"path": "before", "sha256": "a" * 64},
            "gpu_after_observation": {"path": "after", "sha256": "b" * 64},
        }
        artifact["execution"] = execution
        _write(Path(artifact["manifest_path"]).parent / "execution.json", execution)
        return artifact

    recovery.train = train
    checked = []
    monkeypatch.setattr(V.REMOTE, "verify_observation_reference",
                        lambda ref, resource, gpu: checked.append((ref, resource, gpu)) or {})
    backend, _ = _backend(tmp_path, recovery)
    backend.train("s301-none")
    assert len(checked) == 2
    backend.train("s301-none")
    assert len(checked) == 4


def test_gpu_advisory_lock_rejects_live_owner_and_recovers_after_owner_death(tmp_path: Path) -> None:
    lock = tmp_path / "gpu-execution.lock"
    lock.write_bytes(b"\0")
    source = (
        "import sys,time\n"
        "from pathlib import Path\n"
        "import vision_brev_backend_v1 as V\n"
        "with V._gpu_lock(Path(sys.argv[1])):\n"
        " print('LOCKED', flush=True)\n"
        " time.sleep(60)\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "evals"), str(ROOT / "bench"), str(ROOT / "src"), env.get("PYTHONPATH", "")]
    )
    process = subprocess.Popen(
        [sys.executable, "-c", source, str(lock)], stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, env=env,
    )
    try:
        assert process.stdout is not None and process.stdout.readline().strip() == "LOCKED"
        with pytest.raises(V.B.BackendError, match="GPUExecutionBusy"), V._gpu_lock(lock):
            pytest.fail("살아 있는 owner의 GPU lock을 획득했다")
    finally:
        process.kill()
        process.wait(timeout=10)
    acquired = False
    for _ in range(100):
        try:
            with V._gpu_lock(lock):
                acquired = True
            break
        except V.B.BackendError:
            time.sleep(0.05)
    assert acquired and lock.is_file()


def test_record_cache_reaudits_strict_gpu_observation_references(tmp_path: Path, monkeypatch) -> None:
    class FakeStrictReader:
        def __init__(self):
            self.last_receipt = {"path": "gpu-observation", "sha256": "a" * 64}

        def __call__(self):
            return []

    checked = []
    monkeypatch.setattr(V.REMOTE, "StrictRemoteGpuReader", FakeStrictReader)
    monkeypatch.setattr(V.REMOTE, "verify_observation_reference",
                        lambda ref, resource, gpu: checked.append((ref, resource, gpu)) or {})
    recovery = FakeRecovery(tmp_path, popen=lambda *args, **kwargs: FakeProcess(job))
    recovery.read_gpu = FakeStrictReader()
    recovery.vision_resource_proof = {"resource_id": "instance-1", "gpu_uuid": "GPU-fixture"}
    backend, runtime = _backend(tmp_path, recovery)
    stage = runtime / "record/record-1"
    job = _record_job(tmp_path, stage)
    monkeypatch.setattr(V, "_process_identity", lambda pid: {
        "pid": pid, "pgid": pid, "sid": pid, "start_ticks": 1,
        "posix_group_verified": True,
    })
    monkeypatch.setattr(V, "_safe_cleanup", lambda identity, process: True)
    backend.record(job)
    assert len(checked) == 2
    backend.record(job)
    assert len(checked) == 4


def test_protocol_plan_and_canonical_runtime_prevent_cap_reset(tmp_path: Path) -> None:
    recovery = FakeRecovery(tmp_path)
    recovery.protocol["training_plan"] = _plan()[::-1]
    V.ROOT = tmp_path / "root"
    canonical = V.ROOT / ".triage-runtime/vision-brev/vision-test"
    with pytest.raises(ValueError, match="training_plan"):
        V.VisionBrevBackend(recovery, canonical, _plan())
    recovery.protocol["training_plan"] = _plan()
    with pytest.raises(ValueError, match="canonical"):
        V.VisionBrevBackend(recovery, tmp_path / "fresh-cap-root", _plan())


def test_record_rejects_coherent_trace_only_rewrite(tmp_path: Path, monkeypatch) -> None:
    launches = []

    def popen(argv, **kwargs):
        launches.append(argv)
        return FakeProcess(job)

    recovery = FakeRecovery(tmp_path, popen=popen)
    backend, runtime = _backend(tmp_path, recovery)
    stage = runtime / "record/record-1"
    job = _record_job(tmp_path, stage)
    monkeypatch.setattr(V, "_process_identity", lambda pid: {"pid": pid, "pgid": pid, "sid": pid,
                                                             "start_ticks": 1,
                                                             "posix_group_verified": True})
    monkeypatch.setattr(V, "_safe_cleanup", lambda identity, process: True)
    backend.record(job)
    result_path, execution_path = stage / "output/result.json", stage / "execution.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["artifacts"] = {"trace": result["artifacts"]["trace"]}
    _write(result_path, result)
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    execution["result_sha256"] = _sha(result_path)
    execution_path.unlink()
    _write(execution_path, execution)
    with pytest.raises(ValueError, match="고정 다섯"):
        backend.record(job)
    assert len(launches) == 1


@pytest.mark.parametrize(
    ("field", "changed"),
    [
        ("resource", {"resource_id": "other", "gpu_uuid": "GPU-test"}),
        ("resource_parent_sha256", "a" * 64),
        ("live_lease_proof_sha256", "b" * 64),
        ("admission_sha256", "c" * 64),
        ("admission_issued_at_utc", "2020-01-01T00:00:00+00:00"),
        ("admission_expires_at_utc", "2030-01-01T00:00:00+00:00"),
    ],
)
def test_record_rejects_coherent_result_provenance_rewrite(
    tmp_path: Path, monkeypatch, field: str, changed: object
) -> None:
    recovery = FakeRecovery(tmp_path, popen=lambda *args, **kwargs: FakeProcess(job))
    backend, runtime = _backend(tmp_path, recovery)
    stage = runtime / "record/record-1"
    job = _record_job(tmp_path, stage)
    monkeypatch.setattr(V, "_process_identity", lambda pid: {
        "pid": pid, "pgid": pid, "sid": pid, "start_ticks": 1,
        "posix_group_verified": True,
    })
    monkeypatch.setattr(V, "_safe_cleanup", lambda identity, process: True)
    backend.record(job)
    result_path, execution_path = stage / "output/result.json", stage / "execution.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result[field] = changed
    result_path.unlink()
    _write(result_path, result)
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    execution["result_sha256"] = _sha(result_path)
    execution_path.unlink()
    _write(execution_path, execution)
    with pytest.raises(ValueError, match="provenance"):
        backend.record(job)


def test_safe_cleanup_never_signals_reused_or_ambiguous_pid(monkeypatch) -> None:
    signals = []

    class Process:
        def poll(self):
            return None

    monkeypatch.setattr(V.os, "name", "posix")
    monkeypatch.setattr(V, "_group_members", lambda pgid: [pgid])
    monkeypatch.setattr(V, "_identity_is_current", lambda identity: False)
    monkeypatch.setattr(V.os, "killpg", lambda pid, sig: signals.append((pid, sig)), raising=False)
    assert V._safe_cleanup({"pid": 42, "pgid": 42, "sid": 42, "start_ticks": 1}, Process()) is False
    assert signals == []


@pytest.mark.skipif(os.name != "posix", reason="POSIX /proc process-group ownership regression")
def test_posix_identity_captures_start_ticks_and_cleans_only_owned_group() -> None:
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True
    )
    try:
        identity = V._process_identity(process.pid)
        assert identity["pid"] == identity["pgid"] == identity["sid"]
        assert type(identity["start_ticks"]) is int and identity["start_ticks"] > 0
        assert V._safe_cleanup(identity, process) is True
        assert process.poll() is not None
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)
