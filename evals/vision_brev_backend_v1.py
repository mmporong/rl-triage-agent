"""Brev 안의 학습과 같은-Play 영상 기록을 한 실행 경계로 묶는다.

GPU 학습·probe·평가는 검증된 ``brev_recovery_backend_v4.Backend``에
위임한다. 이 모듈은 고정된 여섯 학습을 append-only 원장으로 한 번만
예약하고, 영상 기록 worker를 새 POSIX process group에서 실행한다.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import subprocess
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import brev_recovery_backend_v4 as B
import record_vision_evidence_v1 as R  # pyright: ignore[reportMissingImports]
import vision_remote_gpu_v1 as REMOTE  # pyright: ignore[reportMissingImports]

from rl_triage.correction_ledger import CorrectionLedger

ROOT = Path(__file__).resolve().parents[1]
VISION_WORKER = ROOT / "bench/vision_play_worker_v1.py"
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
SLUG = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}\Z")
RECORD_TIMEOUT_S = 900
PLAN_SCHEMA = "vision_brev_training_plan_v1"
LEDGER_SCHEMA = "vision_brev_training_ledger_event_v1"
APPROVAL_KEYS = {"path", "sha256", "request_id", "loop_state_digest"}
RESULT_KEYS = {
    "schema", "completed", "job_sha256", "protocol_sha256", "checkpoint_sha256",
    "request_binding", "resource", "resource_parent_sha256", "live_lease_proof_sha256",
    "admission_sha256", "admission_issued_at_utc", "admission_expires_at_utc",
    "profile_sha256", "mode", "primary_artifact_sha256",
    "video_sha256", "artifacts",
    "fixed_evaluation_used_as_behavior_ground_truth", "independent_visual_evidence_source_count",
    "zero_update_pixel_lag_confirmed", "gpu_smoke_required_before_model_use",
}
RECORD_ARTIFACTS = {"trace", "measurement_1", "spec_1", "measurement_2", "spec_2"}


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("JSON object가 필요하다")
    return value


def _regular(path: Path, anchor: str) -> Path:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{anchor}는 symlink가 아닌 regular file이어야 한다")
    return path.resolve()


def _single_regular(path: Path, anchor: str) -> Path:
    resolved = _regular(path, anchor)
    if resolved.stat().st_nlink != 1:
        raise ValueError(f"{anchor}는 hardlink가 아닌 단일 regular file이어야 한다")
    return resolved


def _start_ticks(pid: int, proc_root: Path = Path("/proc")) -> int:
    raw = (proc_root / str(pid) / "stat").read_text(encoding="utf-8")
    tail = raw[raw.rfind(")") + 2:].split()
    value = int(tail[19])
    if value <= 0:
        raise ValueError("process start_ticks가 잘못됐다")
    return value


def _process_identity(pid: int) -> dict[str, Any]:
    if type(pid) is not int or pid <= 0:
        raise ValueError("worker PID가 잘못됐다")
    if os.name != "posix":
        return {"pid": pid, "pgid": pid, "sid": pid, "start_ticks": None,
                "posix_group_verified": False}
    pgid, sid = os.getpgid(pid), os.getsid(pid)
    if pgid != pid or sid != pid:
        raise ValueError("record worker가 소유한 새 POSIX session/process group이 아니다")
    return {"pid": pid, "pgid": pgid, "sid": sid, "start_ticks": _start_ticks(pid),
            "posix_group_verified": True}


def _group_members(pgid: int, proc_root: Path = Path("/proc")) -> list[int]:
    members: list[int] = []
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if os.getpgid(int(entry.name)) == pgid:
                members.append(int(entry.name))
        except (OSError, ValueError):
            continue
    return sorted(members)


def _identity_is_current(identity: Mapping[str, Any]) -> bool:
    if os.name != "posix":
        return True
    try:
        pid = int(identity["pid"])
        return (os.getpgid(pid) == identity["pgid"] == pid
                and os.getsid(pid) == identity["sid"] == pid
                and _start_ticks(pid) == identity["start_ticks"])
    except (OSError, KeyError, TypeError, ValueError):
        return False


def _safe_cleanup(identity: Mapping[str, Any], process: subprocess.Popen[Any]) -> bool:
    """시작 tick까지 같은 소유 group만 종료한다. 불명확하면 signal을 보내지 않는다."""
    if os.name != "posix":
        return B.cleanup_group(int(identity["pid"]), process)
    pid = int(identity["pid"])
    members = _group_members(pid)
    if not members:
        return True
    if not _identity_is_current(identity):
        return False
    if process.poll() is None:
        os.killpg(pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            if not _identity_is_current(identity):
                return False
            os.killpg(pid, signal.SIGKILL)
            process.wait(timeout=10)
    deadline = time.monotonic() + 10
    while _group_members(pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    if _group_members(pid):
        if not _identity_is_current(identity):
            return False
        os.killpg(pid, signal.SIGKILL)
        deadline = time.monotonic() + 10
        while _group_members(pid) and time.monotonic() < deadline:
            time.sleep(0.05)
    return not _group_members(pid)


def _verify_record_result(result_path: Path, job: Mapping[str, Any], job_sha: str) -> dict[str, Any]:
    result = _read(_single_regular(result_path, "record result"))
    if set(result) != RESULT_KEYS:
        raise ValueError("record result 필드가 계약과 다르다")
    resource = job.get("resource")
    remote = job.get("remote_gpu_admission")
    if (not isinstance(resource, Mapping) or set(resource) != {"resource_id", "gpu_uuid"}
            or not isinstance(remote, Mapping)
            or set(remote) != {"path", "sha256", "resource_parent_sha256",
                               "live_lease_proof_sha256"}):
        raise ValueError("record job resource 결합이 잘못됐다")
    admission_path = _single_regular(Path(remote["path"]), "record admission")
    admission_sha = remote["sha256"]
    if (not isinstance(admission_sha, str) or HEX64.fullmatch(admission_sha) is None
            or R.sha256_file(admission_path) != admission_sha):
        raise ValueError("record admission SHA가 다르다")
    admission = _read(admission_path)
    if (admission.get("resource_id") != resource["resource_id"]
            or admission.get("gpu_uuid") != resource["gpu_uuid"]
            or admission.get("resource_parent_sha256") != remote["resource_parent_sha256"]
            or admission.get("live_lease_proof_sha256") != remote["live_lease_proof_sha256"]):
        raise ValueError("record admission resource 결합이 다르다")
    output = result_path.parent.resolve()
    expected = {
        "schema": R.RESULT_SCHEMA,
        "completed": True,
        "job_sha256": job_sha,
        "protocol_sha256": job["protocol"]["sha256"],
        "checkpoint_sha256": job["checkpoint"]["sha256"],
        "request_binding": R.request_binding(job),
        "resource": dict(resource),
        "resource_parent_sha256": remote["resource_parent_sha256"],
        "live_lease_proof_sha256": remote["live_lease_proof_sha256"],
        "admission_sha256": admission_sha,
        "admission_issued_at_utc": admission.get("issued_at_utc"),
        "admission_expires_at_utc": admission.get("expires_at_utc"),
        "profile_sha256": R.digest(R.fixed_profile()),
        "mode": R.RECORD_MODE,
        "fixed_evaluation_used_as_behavior_ground_truth": False,
        "independent_visual_evidence_source_count": 1,
        "zero_update_pixel_lag_confirmed": False,
        "gpu_smoke_required_before_model_use": True,
    }
    if any(result.get(key) != value for key, value in expected.items()):
        raise ValueError("record result provenance가 job과 다르다")
    for failure_name in ("failure.json", "shutdown_failure.json"):
        if (output / failure_name).exists():
            raise ValueError("record 완료 폴더에 실패 증거가 함께 있다")
    video = _single_regular(output / "same_play.mp4", "record video")
    if not isinstance(result["video_sha256"], str) or R.sha256_file(video) != result["video_sha256"]:
        raise ValueError("record video SHA가 다르다")
    if result["primary_artifact_sha256"] != result["video_sha256"]:
        raise ValueError("record primary artifact가 video와 다르다")
    artifacts = result["artifacts"]
    if not isinstance(artifacts, dict) or set(artifacts) != RECORD_ARTIFACTS:
        raise ValueError("record artifact 집합이 고정 다섯 파일과 다르다")
    names: set[str] = set()
    paths: dict[str, Path] = {}
    for logical_id, descriptor in artifacts.items():
        if (not isinstance(logical_id, str) or not logical_id or not isinstance(descriptor, dict)
                or set(descriptor) != {"file", "sha256", "bytes"}):
            raise ValueError("record artifact descriptor가 잘못됐다")
        name, expected_sha, expected_bytes = (
            descriptor["file"], descriptor["sha256"], descriptor["bytes"]
        )
        if (not isinstance(name, str) or Path(name).name != name or name in names
                or not isinstance(expected_sha, str) or HEX64.fullmatch(expected_sha) is None
                or type(expected_bytes) is not int or expected_bytes < 0):
            raise ValueError("record artifact 이름·SHA·크기가 잘못됐다")
        matches = [path for path in output.rglob(name) if path.is_file() and not path.is_symlink()]
        if len(matches) != 1:
            raise ValueError("record artifact가 없거나 같은 이름이 중복됐다")
        artifact = _owned(matches[0], output, f"record artifact {logical_id}")
        artifact = _single_regular(artifact, f"record artifact {logical_id}")
        if artifact.stat().st_size != expected_bytes or R.sha256_file(artifact) != expected_sha:
            raise ValueError("record artifact 크기 또는 SHA가 다르다")
        names.add(name)
        paths[logical_id] = artifact
    trace = _read(paths["trace"])
    _canonical(trace)
    trace_keys = {"schema", "method", "job_id", "profile", "profile_sha256", "video_sha256",
                  "checkpoint_sha256", "time_basis", "frame_count", "records",
                  "encoder_receipt", "timing_receipt", "render_alignment"}
    alignment = trace.get("render_alignment")
    encoder = trace.get("encoder_receipt")
    timing = trace.get("timing_receipt")
    if (set(trace) != trace_keys or trace.get("schema") != R.TRACE_SCHEMA or trace.get("method") != R.METHOD
            or trace.get("job_id") != job["job_id"] or trace.get("profile") != R.fixed_profile()
            or trace.get("profile_sha256") != R.digest(R.fixed_profile())
            or trace.get("video_sha256") != result["video_sha256"]
            or trace.get("checkpoint_sha256") != job["checkpoint"]["sha256"]
            or trace.get("time_basis") != R.fixed_profile()["time_basis"]
            or trace.get("frame_count") != R.FRAME_COUNT
            or not isinstance(trace.get("records"), list)
            or len(trace["records"]) != R.FRAME_COUNT
            or not isinstance(encoder, dict) or encoder.get("natural_exit") is not True
            or encoder.get("frames_written") != R.FRAME_COUNT
            or not isinstance(timing, dict) or timing.get("source_sha256") != result["video_sha256"]
            or timing.get("frame_count") != R.FRAME_COUNT or timing.get("fps") != R.FPS
            or timing.get("frame_period_s") != R.FRAME_PERIOD_S
            or timing.get("relative_actual_pts_s") != [row.get("actual_timestamp_s")
                                                        for row in trace["records"]]
            or not isinstance(alignment, dict)
            or set(alignment) != {"camera_pose_applied_before_render", "capture_primitive",
                                  "rt_subframes", "pause_timeline", "delta_time_s",
                                  "robot_state_unchanged_across_render_enforced",
                                  "rgb_timing_matches_state_enforced", "annotator_cpu_copy",
                                  "zero_update_pixel_lag_confirmed", "gpu_smoke_required_before_model_use"}
            or alignment["camera_pose_applied_before_render"] is not True
            or alignment["capture_primitive"] != R.fixed_profile()["camera"]["capture_primitive"]
            or alignment["rt_subframes"] != R.fixed_profile()["camera"]["rt_subframes"]
            or alignment["pause_timeline"] != R.fixed_profile()["camera"]["pause_timeline"]
            or alignment["delta_time_s"] != R.fixed_profile()["camera"]["delta_time_s"]
            or alignment["robot_state_unchanged_across_render_enforced"] is not True
            or alignment["rgb_timing_matches_state_enforced"] is not True
            or alignment["annotator_cpu_copy"] is not True
            or alignment["zero_update_pixel_lag_confirmed"] is not False
            or alignment["gpu_smoke_required_before_model_use"] is not True):
        raise ValueError("record trace가 video/checkpoint/Play 계약과 다르다")
    for index in (1, 2):
        measurement = _read(paths[f"measurement_{index}"])
        spec = _read(paths[f"spec_{index}"])
        _canonical(measurement)
        _canonical(spec)
        play_id = f"{job['job_id']}-w{index}"
        if (set(measurement) != {"schema", "play_run_id", "video_sha256", "checkpoint_sha256",
                                 "command_name", "coordinate_frames", "samples"}
                or measurement.get("schema") != R.MEASUREMENT_SCHEMA
                or measurement.get("play_run_id") != play_id
                or measurement.get("video_sha256") != result["video_sha256"]
                or measurement.get("checkpoint_sha256") != job["checkpoint"]["sha256"]
                or measurement.get("command_name") != "base_velocity"
                or measurement.get("coordinate_frames") != {"command": "body", "motion": "body"}
                or not isinstance(measurement.get("samples"), list)
                or len(measurement["samples"]) != len(R.WINDOWS[index - 1])):
            raise ValueError("record measurement가 같은 Play 수치 계약과 다르다")
        expected_samples = []
        try:
            for timestamp in R.WINDOWS[index - 1]:
                matches = [row for row in trace["records"]
                           if row.get("actual_timestamp_s") == timestamp]
                if len(matches) != 1:
                    raise ValueError("trace actual PTS가 유일하지 않다")
                state = matches[0]["state"]
                expected_samples.append({
                    "timestamp_s": timestamp,
                    "command": dict(zip(("linear_x_mps", "linear_y_mps", "yaw_rate_rps"),
                                        state["command"], strict=True)),
                    "motion": dict(zip(("linear_x_mps", "linear_y_mps", "yaw_rate_rps"),
                                       state["motion"], strict=True)),
                    "pose": dict(state["pose"]),
                })
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("record trace state가 measurement를 재생성할 수 없다") from error
        if measurement["samples"] != expected_samples:
            raise ValueError("record measurement 수치가 trace state와 다르다")
        expected_spec = {
            "schema": R.PREPARE_SCHEMA, "play_run_id": play_id,
            "source_lineage_id": f"{job['job_id']}-same-play", "parent_source_lineage_id": None,
            "video": {"path": str(video), "logical_id": "same_play_video", "play_run_id": play_id},
            "checkpoint": {"path": str(Path(job["checkpoint"]["path"]).resolve()),
                           "logical_id": job["checkpoint"]["logical_id"], "play_run_id": play_id},
            "frames": [{"timestamp_s": value, "play_run_id": play_id,
                        "video_sha256": result["video_sha256"]} for value in R.WINDOWS[index - 1]],
            "measurement": {"path": str(paths[f"measurement_{index}"]),
                            "logical_id": f"same_play_measurements_w{index}", "play_run_id": play_id},
            "play": {"task": R.TASK, "camera": R.fixed_profile()["camera"]["identifier"],
                     "command_name": "base_velocity", "command_frame": "body", "motion_frame": "body",
                     "settings_sha256": R.digest(R.fixed_profile())},
        }
        if spec != expected_spec:
            raise ValueError("record spec가 같은 Play 파일·수치와 다르다")
    return result


def _owned(path: Path, root: Path, anchor: str) -> Path:
    root = Path(root).resolve()
    current = Path(path).absolute()
    if current != root and root not in current.parents:
        raise ValueError(f"{anchor}가 runtime root를 벗어났다")
    relative = current.relative_to(root) if current != root else Path()
    probe = root
    if root.exists() and root.is_symlink():
        raise ValueError("runtime root는 symlink일 수 없다")
    for part in relative.parts:
        probe /= part
        if probe.exists() and probe.is_symlink():
            raise ValueError(f"{anchor} 경로에 symlink가 있다")
    return current.resolve()


def _validate_plan(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    required = {
        "job_id", "arm", "seed", "run_name", "stage_relative", "iterations", "num_envs",
        "source_job_id",
    }
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or len(rows) != 6:
        raise ValueError("학습 계획은 두 seed × NONE/F1/CORRECTED 여섯 건이어야 한다")
    out: list[dict[str, Any]] = []
    ids: set[str] = set()
    seeds: set[int] = set()
    for raw in rows:
        if not isinstance(raw, Mapping) or set(raw) != required:
            raise ValueError("학습 계획 필드가 고정 계약과 다르다")
        row = dict(raw)
        job_id, run_name = row["job_id"], row["run_name"]
        if not isinstance(job_id, str) or SLUG.fullmatch(job_id) is None or job_id in ids:
            raise ValueError("job_id가 잘못됐거나 중복됐다")
        if not isinstance(run_name, str) or SLUG.fullmatch(run_name) is None:
            raise ValueError("run_name이 잘못됐다")
        arm, seed = row["arm"], row["seed"]
        if arm not in ("NONE", "F1", "CORRECTED") or type(seed) is not int or seed < 0:
            raise ValueError("arm 또는 seed가 잘못됐다")
        relative = Path(row["stage_relative"]) if isinstance(row["stage_relative"], str) else Path("/")
        if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != f"training/{job_id}":
            raise ValueError("stage_relative은 training/<job_id>여야 한다")
        if row["iterations"] != 300 or row["num_envs"] != 4096:
            raise ValueError("본 학습은 300 iterations·4096 envs로 고정한다")
        source = row["source_job_id"]
        if (arm == "CORRECTED") != isinstance(source, str):
            raise ValueError("CORRECTED만 source_job_id를 가져야 한다")
        if source is not None and (SLUG.fullmatch(source) is None or source == job_id):
            raise ValueError("source_job_id가 잘못됐다")
        ids.add(job_id)
        seeds.add(seed)
        out.append(row)
    if len(seeds) != 2:
        raise ValueError("학습 계획에는 서로 다른 seed 두 개가 필요하다")
    by_seed = {seed: [row for row in out if row["seed"] == seed] for seed in seeds}
    for seed, group in by_seed.items():
        if sorted(row["arm"] for row in group) != ["CORRECTED", "F1", "NONE"]:
            raise ValueError(f"seed {seed}의 arm 집합이 다르다")
        fault = next(row for row in group if row["arm"] == "F1")
        corrected = next(row for row in group if row["arm"] == "CORRECTED")
        if corrected["source_job_id"] != fault["job_id"]:
            raise ValueError("CORRECTED source_job_id가 같은 seed의 F1이 아니다")
    return out


@contextmanager
def _lock(path: Path, kind: str = "TrainingLedgerBusy") -> Iterator[None]:
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise B.BackendError(kind) from error
    try:
        yield
    finally:
        os.close(fd)
        path.unlink(missing_ok=True)


@contextmanager
def _gpu_lock(path: Path) -> Iterator[None]:
    """프로세스 사망 시 OS가 해제하는 비대기 GPU advisory lock."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    try:
        if handle.seek(0, os.SEEK_END) == 0:
            handle.write(b"\0")
            handle.flush()
            os.fsync(handle.fileno())
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, BlockingIOError) as error:
            raise B.BackendError("GPUExecutionBusy") from error
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


class SafeRecoveryBackend(B.Backend):
    """기존 v4 계약을 유지하면서 child 소유권을 start tick까지 검증한다."""

    vision_resource_proof: dict[str, Any] | None = None

    def _execute(self, stage: Path, mode: str,
                 job: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        stage = Path(stage).resolve()
        if stage.exists():
            if not stage.is_dir() or any(stage.iterdir()):
                raise FileExistsError("stage는 새 빈 폴더여야 한다")
        else:
            stage.mkdir(parents=True)
        job_path, worker_output = stage / "job.json", stage / "worker.json"
        parent_output = stage / "execution.json"
        B.write_new(job_path, job)
        started = time.perf_counter()
        receipt: dict[str, Any] = {
            "schema": "brev_recovery_posix_execution_v1", "mode": mode,
            "completed": False, "started_at": B.now(), "exit_code": None,
            "timed_out": False, "cleanup_verified": True, "launch": None,
            "gpu_before": None, "gpu_after": None, "wall_s": 0.0,
            "gpu_before_observation": None, "gpu_after_observation": None,
            "source_before": None, "source_after": None,
        }
        process: subprocess.Popen[Any] | None = None
        worker: dict[str, Any] = {}
        try:
            receipt["source_before"] = self._source_snapshot()
            receipt["gpu_before"] = self.read_gpu()
            receipt["gpu_before_observation"] = getattr(self.read_gpu, "last_receipt", None)
            if receipt["gpu_before"]:
                raise B.BackendError("ForeignGPUBusy", receipt)
            with (stage / "stdout.log").open("x", encoding="utf-8") as stdout, \
                    (stage / "stderr.log").open("x", encoding="utf-8") as stderr:
                process = self.popen(
                    [str(self.python), "-B", str(B.WORKER), "--job", str(job_path),
                     "--output", str(worker_output)], cwd=str(self.lab),
                    env={**os.environ, "OMNI_KIT_ACCEPT_EULA": "YES", "PYTHONUNBUFFERED": "1",
                         "CUDA_VISIBLE_DEVICES": "0"}, stdout=stdout, stderr=stderr,
                    start_new_session=True,
                )
                identity = _process_identity(process.pid)
                receipt["launch"] = identity
                try:
                    receipt["exit_code"] = process.wait(timeout=B.TIMEOUTS[mode])
                except subprocess.TimeoutExpired:
                    receipt["timed_out"] = True
                finally:
                    receipt["cleanup_verified"] = _safe_cleanup(identity, process)
                    receipt["exit_code"] = (-signal.SIGTERM if receipt["timed_out"]
                                            else process.returncode)
            receipt["gpu_after"] = self.read_gpu()
            receipt["gpu_after_observation"] = getattr(self.read_gpu, "last_receipt", None)
            receipt["source_after"] = self._source_snapshot()
            worker = _read(worker_output) if worker_output.is_file() else {}
            receipt["completed"] = bool(
                receipt["exit_code"] == 0 and not receipt["timed_out"]
                and receipt["cleanup_verified"] and receipt["gpu_after"] == []
                and worker.get("completed") is True
                and receipt["source_before"] == receipt["source_after"]
            )
            if not receipt["completed"]:
                receipt["error_type"] = "InvalidExecutionEvidence"
        except BaseException as error:  # noqa: BLE001 - 예외도 write-once 증거로 닫는다.
            receipt["error_type"] = error.kind if isinstance(error, B.BackendError) else type(error).__name__
            if process is not None and isinstance(receipt.get("launch"), dict):
                try:
                    receipt["cleanup_verified"] = _safe_cleanup(receipt["launch"], process)
                    receipt["exit_code"] = process.returncode
                except BaseException:  # noqa: BLE001
                    receipt["cleanup_verified"] = False
            try:
                receipt["gpu_after"] = self.read_gpu()
                receipt["gpu_after_observation"] = getattr(self.read_gpu, "last_receipt", None)
            except BaseException:  # noqa: BLE001
                receipt["gpu_after"] = None
        finally:
            receipt["wall_s"] = time.perf_counter() - started
            receipt["finished_at"] = B.now()
            B.write_new(parent_output, receipt)
        if not receipt["completed"]:
            raise B.BackendError(str(receipt.get("error_type", "InvalidExecutionEvidence")), receipt)
        return worker, receipt


class VisionBrevBackend:
    """고정 학습 원장과 같은-Play record 실행을 소유하는 얇은 adapter."""

    def __init__(self, recovery: B.Backend, runtime_root: Path,
                 training_plan: Sequence[Mapping[str, Any]]):
        self.recovery = recovery
        protocol = getattr(recovery, "protocol", None)
        if not isinstance(protocol, Mapping):
            raise TypeError("recovery protocol 결합이 없다")
        protocol_id = protocol.get("protocol_id")
        if not isinstance(protocol_id, str) or SLUG.fullmatch(protocol_id) is None:
            raise ValueError("recovery protocol_id가 잘못됐다")
        plan = _validate_plan(training_plan)
        if protocol.get("training_plan") != plan:
            raise ValueError("recovery protocol training_plan이 adapter 계획과 다르다")
        self.runtime_root = Path(runtime_root).absolute()
        expected_root = (ROOT / ".triage-runtime" / "vision-brev" / protocol_id).resolve()
        if self.runtime_root.resolve() != expected_root:
            raise ValueError("runtime root가 protocol_id의 canonical 경로가 아니다")
        if self.runtime_root.exists() and (not self.runtime_root.is_dir() or self.runtime_root.is_symlink()):
            raise ValueError("runtime root는 symlink가 아닌 directory여야 한다")
        self.runtime_root.mkdir(parents=True, exist_ok=True)
        self.runtime_root = self.runtime_root.resolve()
        self.plan = plan
        self.plan_by_id = {row["job_id"]: row for row in self.plan}
        self.plan_path = self.runtime_root / "training_plan.json"
        body = {"schema": PLAN_SCHEMA, "rows": self.plan}
        if self.plan_path.exists():
            if _read(_regular(self.plan_path, "training plan")) != body:
                raise ValueError("기존 runtime의 학습 계획이 다르다")
        else:
            B.write_new(self.plan_path, body)
        self.plan_sha256 = B.sha256(self.plan_path)
        self.ledger_path = self.runtime_root / "training.ledger.jsonl"
        self.lock_path = self.runtime_root / "training.ledger.lock"
        self.gpu_lock_path = self.runtime_root / "gpu-execution.lock"
        self._gpu_owner_thread: int | None = None
        self._gpu_depth = 0

    @classmethod
    def from_runtime(cls, lab: Path, python: Path, protocol: Mapping[str, Any], anchor: str,
                     runtime_root: Path, training_plan: Sequence[Mapping[str, Any]], **kwargs: Any
                     ) -> VisionBrevBackend:
        read_gpu = kwargs.pop("read_gpu", None)
        resource_proof = kwargs.pop("resource_proof", None)
        if read_gpu is None:
            if not isinstance(resource_proof, Mapping):
                raise ValueError("기본 원격 GPU reader에는 resource_proof가 필요하다")
            capture = {key: resource_proof[key] for key in REMOTE.CAPTURE_KEYS}
            read_gpu = REMOTE.StrictRemoteGpuReader(
                capture, Path(runtime_root) / "gpu-observations"
            )
        recovery = SafeRecoveryBackend(
            lab, python, dict(protocol), anchor, read_gpu=read_gpu, **kwargs
        )
        if isinstance(resource_proof, Mapping):
            recovery.vision_resource_proof = dict(resource_proof)
        return cls(recovery, runtime_root, training_plan)

    @contextmanager
    def action_admission(self) -> Iterator[None]:
        """한 source action의 예약부터 receipt까지 같은 GPU permit을 유지한다."""
        thread_id = threading.get_ident()
        if self._gpu_owner_thread == thread_id:
            self._gpu_depth += 1
            try:
                yield
            finally:
                self._gpu_depth -= 1
            return
        with _gpu_lock(self.gpu_lock_path):
            self._gpu_owner_thread, self._gpu_depth = thread_id, 1
            try:
                if self.recovery.read_gpu():
                    raise B.BackendError("ForeignGPUBusy")
                yield
            finally:
                self._gpu_depth = 0
                self._gpu_owner_thread = None

    def _events(self) -> list[dict[str, Any]]:
        if not self.ledger_path.exists():
            return []
        if self.ledger_path.is_symlink():
            raise ValueError("학습 원장은 symlink일 수 없다")
        events: list[dict[str, Any]] = []
        states: dict[str, str] = {}
        for number, line in enumerate(self.ledger_path.read_text(encoding="utf-8").splitlines(), 1):
            value = json.loads(line)
            if not isinstance(value, dict) or value.get("schema") != LEDGER_SCHEMA:
                raise ValueError(f"학습 원장 {number}행 schema가 다르다")
            job_id_value, event_value = value.get("job_id"), value.get("event")
            if not isinstance(job_id_value, str) or not isinstance(event_value, str):
                raise TypeError("학습 원장 job_id/event 형식이 다르다")
            job_id, event = job_id_value, event_value
            if job_id not in self.plan_by_id or value.get("plan_sha256") != self.plan_sha256:
                raise ValueError("학습 원장 결합이 계획과 다르다")
            expected = {None: "reserved", "reserved": "started", "started": "completed"}.get(states.get(job_id))
            if event != expected:
                raise ValueError("학습 원장 사건 순서가 다르다")
            states[job_id] = event
            events.append(value)
        return events

    def _append(self, event: Mapping[str, Any]) -> None:
        with self.ledger_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(dict(event), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def _approval_prefix(self, approval: Mapping[str, Any] | None) -> dict[str, Any] | None:
        if approval is None:
            return None
        path = _regular(Path(approval["path"]), "correction ledger")
        request_id = approval["request_id"]
        events = [row for row in CorrectionLedger(path).events() if row.get("request_id") == request_id]
        if [row.get("event") for row in events[:3]] != ["proposed", "approved", "consumed"]:
            raise ValueError("교정 승인 prefix가 바뀌었다")
        request = CorrectionLedger(path).requests().get(request_id)
        source_sha = request.get("evidence", {}).get("source_checkpoint_sha256") if isinstance(request, dict) else None
        if not isinstance(source_sha, str) or HEX64.fullmatch(source_sha) is None:
            raise ValueError("교정 승인 prefix에 source checkpoint SHA가 없다")
        return {"path": str(path), "request_id": request_id, "count": 3,
                "sha256": _digest(events[:3]), "source_checkpoint_sha256": source_sha}

    def _training_inventory(self, artifact: Mapping[str, Any],
                            approval: Mapping[str, Any] | None) -> dict[str, Any]:
        inventory: dict[str, Any] = {}
        for key in ("checkpoint", "telemetry"):
            descriptor = artifact.get(key)
            if (not isinstance(descriptor, Mapping) or not isinstance(descriptor.get("path"), str)
                    or not isinstance(descriptor.get("sha256"), str)):
                raise TypeError(f"학습 {key} descriptor가 없다")
            path = _single_regular(Path(descriptor["path"]), f"training {key}")
            if key == "telemetry":
                path = _owned(path, self.runtime_root, "training telemetry")
            if B.sha256(path) != descriptor["sha256"]:
                raise ValueError(f"학습 {key} SHA가 다르다")
            inventory[key] = {"path": str(path), "sha256": descriptor["sha256"]}
        manifest_path = artifact.get("manifest_path")
        manifest_sha = artifact.get("manifest_sha256")
        if not isinstance(manifest_path, str) or not isinstance(manifest_sha, str):
            raise TypeError("학습 manifest descriptor가 없다")
        manifest = _single_regular(Path(manifest_path), "training manifest")
        manifest = _owned(manifest, self.runtime_root, "training manifest")
        if B.sha256(manifest) != manifest_sha or _read(manifest) != artifact.get("manifest"):
            raise ValueError("학습 manifest 파일과 본문이 다르다")
        execution = artifact.get("execution")
        readback = artifact.get("readback")
        if not isinstance(execution, Mapping) or not isinstance(readback, Mapping):
            raise TypeError("학습 execution/readback이 없다")
        if execution.get("schema") == "brev_recovery_posix_execution_v1":
            self.verify_execution_gpu_observations(execution)
        manifest_body = artifact["manifest"]
        if not isinstance(manifest_body, Mapping):
            raise TypeError("학습 manifest 본문이 없다")
        file_bindings: dict[str, Any] = {}
        for key, manifest_key, expected_body in (
            ("readback", "readback_path", readback),
            ("records", "records_path", None),
            ("execution", None, execution),
        ):
            raw_path = (manifest.parent / "execution.json" if manifest_key is None
                        else Path(str(manifest_body.get(manifest_key, ""))))
            path = _single_regular(raw_path, f"training {key}")
            path = _owned(path, self.runtime_root, f"training {key}")
            if expected_body is not None and _read(path) != dict(expected_body):
                raise ValueError(f"학습 {key} 파일과 artifact 본문이 다르다")
            file_bindings[key] = {"path": str(path), "sha256": B.sha256(path)}
        inventory.update({
            "manifest": {"path": str(manifest), "sha256": manifest_sha},
            "manifest_body_sha256": _digest(artifact["manifest"]),
            "readback_sha256": _digest(readback),
            "execution_sha256": _digest(execution),
            "required_files": file_bindings,
            "approval_prefix": self._approval_prefix(approval),
        })
        return inventory

    def verify_execution_gpu_observations(self, execution: Mapping[str, Any]) -> None:
        if not isinstance(self.recovery.read_gpu, REMOTE.StrictRemoteGpuReader):
            return
        resource = getattr(self.recovery, "vision_resource_proof", None)
        if not isinstance(resource, Mapping):
            raise TypeError("strict GPU 관측 검증에 resource_proof가 없다")
        for key in ("gpu_before_observation", "gpu_after_observation"):
            reference = execution.get(key)
            if not isinstance(reference, Mapping):
                raise TypeError(f"{key} 참조가 없다")
            REMOTE.verify_observation_reference(
                reference, str(resource.get("resource_id")), str(resource.get("gpu_uuid"))
            )

    def _verify_inventory(self, artifact: Mapping[str, Any], inventory: Any) -> None:
        if not isinstance(inventory, Mapping):
            raise TypeError("완료 학습 artifact inventory가 없다")
        expected = self._training_inventory(artifact, inventory.get("approval_prefix"))
        if dict(inventory) != expected:
            raise ValueError("완료 학습 필수 artifact inventory가 변조됐다")

    def _completed(self, job_id: str) -> dict[str, Any] | None:
        rows = [row for row in self._events() if row["job_id"] == job_id]
        if not rows or rows[-1]["event"] != "completed":
            return None
        event = rows[-1]
        artifact_path = _owned(Path(event["artifact_path"]), self.runtime_root, "artifact")
        if not artifact_path.is_file() or B.sha256(artifact_path) != event["artifact_sha256"]:
            raise ValueError("완료 학습 artifact가 변조됐다")
        artifact = _read(artifact_path)
        row = self.plan_by_id[job_id]
        if artifact.get("seed") != row["seed"] or artifact.get("arm") != row["arm"]:
            raise ValueError("완료 학습 artifact identity가 다르다")
        self._verify_inventory(artifact, event.get("inventory"))
        if row["arm"] == "CORRECTED":
            source = self._completed(row["source_job_id"])
            if source is None:
                raise ValueError("교정 학습의 F1 parent가 완료 상태가 아니다")
            source_checkpoint = source.get("checkpoint")
            source_sha = source_checkpoint.get("sha256") if isinstance(source_checkpoint, dict) else None
            approval = event.get("inventory", {}).get("approval_prefix")
            if (not isinstance(approval, Mapping) or artifact.get("source_checkpoint_sha256") != source_sha
                    or approval.get("source_checkpoint_sha256") != source_sha):
                raise ValueError("교정 학습이 F1 parent·승인 evidence와 다르다")
        return artifact

    def _approval(self, reference: Mapping[str, Any] | None, source_sha: str) -> dict[str, Any]:
        if not isinstance(reference, Mapping) or set(reference) != APPROVAL_KEYS:
            raise ValueError("CORRECTED에는 소비된 교정 승인 참조가 필요하다")
        path = _regular(Path(reference["path"]), "correction ledger")
        if not isinstance(reference["sha256"], str) or B.sha256(path) != reference["sha256"]:
            raise ValueError("교정 승인 원장 SHA가 다르다")
        request_id = reference["request_id"]
        ledger = CorrectionLedger(path)
        request = ledger.requests().get(request_id)
        if (not isinstance(request_id, str) or not isinstance(request, dict)
                or request.get("state") != "running"
                or request.get("loop_state_digest") != reference["loop_state_digest"]
                or request.get("correction_id") != "restore_body_frame_reward"
                or request.get("recovery_mode") != "restart"
                or request.get("protocol_case_arm", {}).get("arm") != "corrected"
                or request.get("evidence", {}).get("source_checkpoint_sha256") != source_sha):
            raise ValueError("교정 승인이 consumed 상태·교정·원본 checkpoint와 다르다")
        events = [row["event"] for row in ledger.events() if row.get("request_id") == request_id]
        if events != ["proposed", "approved", "consumed"]:
            raise ValueError("교정 승인은 proposed→approved→consumed 경계여야 한다")
        if B.sha256(path) != reference["sha256"]:
            raise ValueError("검증 중 교정 승인 원장이 바뀌었다")
        return {"path": str(path), "sha256": reference["sha256"], "request_id": request_id,
                "prefix_sha256": _digest([row for row in ledger.events()
                                           if row.get("request_id") == request_id])}

    def train(self, job_id: str, *, approval_reference: Mapping[str, Any] | None = None) -> dict[str, Any]:
        row = self.plan_by_id.get(job_id)
        if row is None:
            raise ValueError("등록되지 않은 학습 job_id")
        with self.action_admission():
            with _lock(self.lock_path):
                cached = self._completed(job_id)
                if cached is not None:
                    return cached
                prior = [event for event in self._events() if event["job_id"] == job_id]
                if prior:
                    return {"status": "HOLD", "reason": "training_pending", "job_id": job_id}
            return self._train_locked(job_id, row, approval_reference)

    def _train_locked(self, job_id: str, row: Mapping[str, Any],
                      approval_reference: Mapping[str, Any] | None) -> dict[str, Any]:
        with _lock(self.lock_path):
            cached = self._completed(job_id)
            if cached is not None:
                return cached
            prior = [event for event in self._events() if event["job_id"] == job_id]
            if prior:
                return {"status": "HOLD", "reason": "training_pending", "job_id": job_id}
            source_sha = None
            approval = None
            if row["arm"] == "CORRECTED":
                source = self._completed(row["source_job_id"])
                if source is None:
                    return {"status": "HOLD", "reason": "source_training_not_completed", "job_id": job_id}
                checkpoint = source.get("checkpoint")
                source_sha = checkpoint.get("sha256") if isinstance(checkpoint, dict) else None
                if not isinstance(source_sha, str) or HEX64.fullmatch(source_sha) is None:
                    raise ValueError("F1 source checkpoint SHA가 없다")
                approval = self._approval(approval_reference, source_sha)
            elif approval_reference is not None:
                raise ValueError("NONE/F1은 교정 승인 참조를 받지 않는다")
            base = {"schema": LEDGER_SCHEMA, "job_id": job_id, "plan_sha256": self.plan_sha256}
            self._append({**base, "event": "reserved", "stage_relative": row["stage_relative"]})
            self._append({**base, "event": "started", "approval_reference": approval})

        stage = _owned(self.runtime_root / row["stage_relative"], self.runtime_root, "training stage")
        artifact = self.recovery.train(
            stage, row["arm"], row["seed"], row["run_name"],
            corrections=B.ARMS[row["arm"]][1], iterations=300,
            source_checkpoint_sha256=source_sha,
        )
        inventory = self._training_inventory(artifact, approval)
        artifact_path = stage / "adapter_artifact.json"
        B.write_new(artifact_path, artifact)
        artifact_sha = B.sha256(artifact_path)
        with _lock(self.lock_path):
            prior = [event for event in self._events() if event["job_id"] == job_id]
            if [event["event"] for event in prior] != ["reserved", "started"]:
                raise ValueError("학습 완료 전 원장 상태가 바뀌었다")
            self._append({"schema": LEDGER_SCHEMA, "event": "completed", "job_id": job_id,
                          "plan_sha256": self.plan_sha256, "artifact_path": str(artifact_path),
                          "artifact_sha256": artifact_sha, "inventory": inventory})
        return artifact

    def probe(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        with self.action_admission():
            return self.recovery.probe(*args, **kwargs)

    def evaluate(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        with self.action_admission():
            return self.recovery.evaluate(*args, **kwargs)

    def _cached_record(self, stage: Path, job: Mapping[str, Any]) -> dict[str, Any] | None:
        if not stage.exists():
            return None
        if not stage.is_dir() or stage.is_symlink():
            raise ValueError("record stage가 symlink 아닌 directory가 아니다")
        job_path, receipt_path = stage / "job.json", stage / "execution.json"
        if not job_path.is_file() or job_path.is_symlink() or _read(job_path) != dict(job):
            raise ValueError("같은 record stage에 다른 job을 재사용할 수 없다")
        if not receipt_path.is_file():
            return {"status": "HOLD", "reason": "record_pending", "job_id": job.get("job_id")}
        receipt_path = _owned(receipt_path, self.runtime_root, "record execution")
        receipt = _read(_single_regular(receipt_path, "record execution"))
        if B.sha256(job_path) != receipt.get("job_sha256"):
            raise ValueError("record job이 변조됐다")
        if receipt.get("completed") is not True:
            raise B.BackendError(str(receipt.get("error_type", "InvalidRecordEvidence")), receipt)
        if (receipt.get("schema") != "vision_brev_record_execution_v1"
                or receipt.get("exit_code") != 0 or receipt.get("timed_out") is not False
                or receipt.get("cleanup_verified") is not True
                or receipt.get("gpu_before") != [] or receipt.get("gpu_after") != []
                or receipt.get("source_before") != receipt.get("source_after")
                or not isinstance(receipt.get("launch"), dict)):
            raise ValueError("완료 record execution 증거가 유효하지 않다")
        if isinstance(self.recovery.read_gpu, REMOTE.StrictRemoteGpuReader):
            self.verify_execution_gpu_observations(receipt)
        result_path = stage / "output" / "result.json"
        result_path = _owned(result_path, self.runtime_root, "record result")
        if not result_path.is_file() or B.sha256(result_path) != receipt.get("result_sha256"):
            raise ValueError("record result가 변조됐다")
        result = _verify_record_result(result_path, dict(job), receipt["job_sha256"])
        return {"status": "capture_complete_gpu_smoke_pending", "model_use_ready": False,
                "result": result, "execution": receipt}

    def record(self, job: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(job, Mapping):
            raise TypeError("record job은 mapping이어야 한다")
        job_id_value = job.get("job_id")
        if not isinstance(job_id_value, str):
            raise TypeError("record job_id가 없다")
        job_id = job_id_value
        if SLUG.fullmatch(job_id) is None:
            raise ValueError("record job_id가 잘못됐다")
        stage = _owned(self.runtime_root / "record" / job_id, self.runtime_root, "record stage")
        cached = self._cached_record(stage, job)
        if cached is not None:
            return cached
        with self.action_admission():
            cached = self._cached_record(stage, job)
            if cached is not None:
                return cached
            return self._record_new(dict(job), stage)

    def _record_new(self, job: dict[str, Any], stage: Path) -> dict[str, Any]:
        stage.mkdir(parents=True, exist_ok=False)
        job_path = stage / "job.json"
        B.write_new(job_path, dict(job))
        receipt: dict[str, Any] = {
            "schema": "vision_brev_record_execution_v1", "completed": False,
            "job_sha256": B.sha256(job_path), "exit_code": None, "timed_out": False,
            "cleanup_verified": True, "gpu_before": None, "gpu_after": None,
            "gpu_before_observation": None, "gpu_after_observation": None,
            "source_before": None, "source_after": None, "result_sha256": None,
            "launch": None,
        }
        process: subprocess.Popen[Any] | None = None
        result: dict[str, Any] = {}
        started = time.perf_counter()
        try:
            validated = R.validate_job(job_path)
            if validated["output_directory"] != (stage / "output").resolve():
                raise ValueError("record output_directory는 소유 stage/output이어야 한다")
            receipt["source_before"] = self.recovery._source_snapshot()
            receipt["gpu_before"] = self.recovery.read_gpu()
            receipt["gpu_before_observation"] = getattr(self.recovery.read_gpu, "last_receipt", None)
            if receipt["gpu_before"]:
                raise B.BackendError("ForeignGPUBusy", receipt)
            with (stage / "stdout.log").open("x", encoding="utf-8") as stdout, \
                    (stage / "stderr.log").open("x", encoding="utf-8") as stderr:
                process = self.recovery.popen(
                    [str(self.recovery.python), "-B", str(VISION_WORKER), "--job", str(job_path)],
                    cwd=str(self.recovery.lab),
                    env={**os.environ, "OMNI_KIT_ACCEPT_EULA": "YES", "PYTHONUNBUFFERED": "1",
                         "CUDA_VISIBLE_DEVICES": "0"},
                    stdout=stdout, stderr=stderr, start_new_session=True,
                )
                receipt["launch"] = _process_identity(process.pid)
                try:
                    receipt["exit_code"] = process.wait(timeout=RECORD_TIMEOUT_S)
                except subprocess.TimeoutExpired:
                    receipt["timed_out"] = True
                finally:
                    receipt["cleanup_verified"] = _safe_cleanup(receipt["launch"], process)
                    if receipt["timed_out"]:
                        receipt["exit_code"] = -signal.SIGTERM
            receipt["gpu_after"] = self.recovery.read_gpu()
            receipt["gpu_after_observation"] = getattr(self.recovery.read_gpu, "last_receipt", None)
            receipt["source_after"] = self.recovery._source_snapshot()
            if isinstance(self.recovery.read_gpu, REMOTE.StrictRemoteGpuReader):
                self.verify_execution_gpu_observations(receipt)
            result_path = stage / "output" / "result.json"
            result_path = _owned(result_path, self.runtime_root, "record result")
            result = _verify_record_result(result_path, dict(job), receipt["job_sha256"])
            receipt["result_sha256"] = B.sha256(result_path)
            receipt["completed"] = bool(
                receipt["exit_code"] == 0 and not receipt["timed_out"]
                and receipt["cleanup_verified"] and receipt["gpu_after"] == []
                and receipt["source_before"] == receipt["source_after"]
            )
            if not receipt["completed"]:
                receipt["error_type"] = "InvalidRecordEvidence"
        except BaseException as error:  # noqa: BLE001 - 실패도 write-once 실행 증거로 닫는다.
            receipt["error_type"] = type(error).__name__ if not isinstance(error, B.BackendError) else error.kind
            if process is not None:
                try:
                    launch = receipt.get("launch")
                    receipt["cleanup_verified"] = bool(
                        isinstance(launch, dict) and _safe_cleanup(launch, process)
                    )
                    receipt["exit_code"] = process.returncode
                except BaseException:  # noqa: BLE001
                    receipt["cleanup_verified"] = False
            try:
                receipt["gpu_after"] = self.recovery.read_gpu()
                receipt["gpu_after_observation"] = getattr(self.recovery.read_gpu, "last_receipt", None)
            except BaseException:  # noqa: BLE001
                receipt["gpu_after"] = None
        finally:
            receipt["wall_s"] = time.perf_counter() - started
            B.write_new(stage / "execution.json", receipt)
        if not receipt["completed"]:
            raise B.BackendError(str(receipt.get("error_type", "InvalidRecordEvidence")), receipt)
        return {"status": "capture_complete_gpu_smoke_pending", "model_use_ready": False,
                "result": result, "execution": receipt}
