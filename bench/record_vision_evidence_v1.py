"""같은 Isaac Play 상태의 RGB frame과 운동 수치를 write-once로 기록한다.

이 모듈은 Isaac Lab을 import하지 않는다. 실행 전 결합 검증, raw RGB 인코딩,
실제 영상 PTS와 상태 step의 정렬, V01 입력 생성을 CPU에서 검사할 수 있다.
"""
from __future__ import annotations

import hashlib
import json
import math
import numbers
import os
import re
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Protocol

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))
from lease_timestamps import parse_lease_timestamp  # pyright: ignore[reportMissingImports]

JOB_SCHEMA = "vision_play_record_job_v1"
PROTOCOL_SCHEMA = "vision_play_record_protocol_v1"
ADMISSION_SCHEMA = "remote_gpu_admission_receipt_v1"
TRACE_SCHEMA = "vision_play_raw_trace_v1"
RESULT_SCHEMA = "vision_play_record_result_v1"
FAILURE_SCHEMA = "vision_play_record_failure_v1"
METHOD = "same_play_raw_rgb_state_lockstep_v1"
RECORD_MODE = "record"
RENDER_SYNC_SMOKE_MODE = "render_sync_smoke"
MEASUREMENT_SCHEMA = "vision_play_measurements_v1"
PREPARE_SCHEMA = "vision_evidence_prepare_v1"
TASK = "Isaac-Velocity-Flat-Unitree-Go2-Play-v0"
FPS = 50
FRAME_COUNT = 625
DURATION_S = 12.5
FRAME_PERIOD_S = 1.0 / FPS
PTS_TOLERANCE_S = 0.0005
ADMISSION_MAX_AGE_S = 30.0
EXECUTION_AUTHORIZATION_S = 900.0
WIDTH = 640
HEIGHT = 480
WINDOWS = ((1.0, 2.0, 3.0, 4.0, 5.0, 6.0), (7.0, 8.0, 9.0, 10.0, 11.0, 12.0))
ISAAC_READ_TIMES_FIELDS = (
    "durationDenominator",
    "durationNumerator",
    "externalTimeOfSimNs",
    "frameNumber",
    "rationalTimeOfSimDenominator",
    "rationalTimeOfSimNumerator",
    "sampleTimeOffsetInSimFrames",
    "simulationTime",
    "simulationTimeMonotonic",
    "swhFrameNumber",
    "systemTime",
)
ISAAC_READ_TIMES_INTEGER_FIELDS = frozenset(
    field for field in ISAAC_READ_TIMES_FIELDS if field not in {
        "simulationTime", "simulationTimeMonotonic", "systemTime"
    }
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")
_SOURCE_PATH = re.compile(r"[A-Za-z0-9_.@+/-]+")


class Encoder(Protocol):
    def write(self, rgb: bytes) -> None: ...
    def finish(self) -> dict[str, Any]: ...
    def abort(self) -> dict[str, Any]: ...


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def sha256_file(path: Path, *, lf: bool = False) -> str:
    data = Path(path).read_bytes()
    if lf:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def read_object(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"비유한 JSON 수 {value}는 허용되지 않는다")

    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise ValueError(f"중복 JSON key: {key}")
            result[key] = value
        return result

    value = json.loads(
        Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=pairs,
        parse_constant=reject_constant,
    )
    if not isinstance(value, dict):
        raise TypeError("JSON object가 필요하다")
    return value


def write_new(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def fixed_profile() -> dict[str, Any]:
    return {
        "task": TASK,
        "seed": 42,
        "num_envs": 1,
        "action_scale": 0.25,
        "command": {
            "name": "base_velocity",
            "frame": "body",
            "linear_x_mps": 1.0,
            "linear_y_mps": 0.0,
            "yaw_rate_rps": 0.0,
            "heading_command": False,
            "relative_standing_envs": 0.0,
            "resampling_time_range_s": [DURATION_S, DURATION_S],
        },
        "motion_frame": "body",
        "duration_s": DURATION_S,
        "fps": FPS,
        "frame_count": FRAME_COUNT,
        "resolution": {"width": WIDTH, "height": HEIGHT},
        "camera": {
            "identifier": "robot_parallel_world_camera_v1",
            "mode": "parallel_track_root_position_world_axes",
            "camera_prim_path": "/OmniverseKit_Persp",
            "eye_offset_m": [3.0, 3.0, 2.0],
            "lookat_offset_m": [0.0, 0.0, 0.35],
            "rotation_tracks_robot": False,
            "projection": "perspective",
            "horizontal_fov_deg": 60.0,
            "horizontal_aperture_mm": 20.955,
            "focal_length_mm": 18.147562,
            "clipping_range_m": [0.01, 1000.0],
            "capture_primitive": "replicator_orchestrator_step_v1",
            "rt_subframes": 1,
            "pause_timeline": False,
            "delta_time_s": 0.0,
            "read_after_capture": True,
            "isaac_read_times_contract": "isaac_sim_4_5_outputs_11_v1",
            "isaac_read_times_fields": list(ISAAC_READ_TIMES_FIELDS),
            "overlay_count": 0,
        },
        "launcher": {"headless": True, "enable_cameras": True, "fast_shutdown": False},
        "sample_windows_s": [list(window) for window in WINDOWS],
        "time_basis": "relative_actual_video_frame_pts_from_first_decoded_frame",
    }


def _exact(value: Any, keys: set[str], anchor: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{anchor} 필드가 계약과 다르다")
    return value


def _hash(value: Any, anchor: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{anchor} SHA-256이 잘못됐다")
    return value


def _identifier(value: Any, anchor: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"{anchor} 식별자가 잘못됐다")
    return value


def _regular_file(path: Path, anchor: str) -> Path:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{anchor}는 symlink가 아닌 regular file이어야 한다")
    return path.resolve()


def _run_git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True,
        encoding="utf-8", errors="replace", check=False,
    )


def request_binding(job: Mapping[str, Any]) -> str:
    return digest(
        {
            "method": job["method"],
            "mode": job["mode"],
            "job_id": job["job_id"],
            "profile": job["profile"],
            "source": job["source"],
            "protocol_sha256": job["protocol"]["sha256"],
            "checkpoint_sha256": job["checkpoint"]["sha256"],
            "resource": job["resource"],
            "resource_parent_sha256": job["remote_gpu_admission"]["resource_parent_sha256"],
            "live_lease_proof_sha256": job["remote_gpu_admission"]["live_lease_proof_sha256"],
        }
    )


def validate_job(job_path: Path, *, now: datetime | None = None) -> dict[str, Any]:
    job_path = _regular_file(Path(job_path), "job")
    job = _exact(
        read_object(job_path),
        {"schema", "method", "mode", "job_id", "profile", "source", "protocol", "checkpoint", "resource",
         "remote_gpu_admission", "output_directory"},
        "job",
    )
    if job["schema"] != JOB_SCHEMA or job["method"] != METHOD:
        raise ValueError("job schema 또는 method가 다르다")
    if job["mode"] not in {RECORD_MODE, RENDER_SYNC_SMOKE_MODE}:
        raise ValueError("job mode가 등록 범위 밖이다")
    job_id = _identifier(job["job_id"], "job_id")
    if canonical(job["profile"]) != canonical(fixed_profile()):
        raise ValueError("고정 Play profile이 바뀌었다")
    resource = _exact(job["resource"], {"resource_id", "gpu_uuid"}, "resource")
    resource_id = _identifier(resource["resource_id"], "resource.resource_id")
    gpu_uuid = _identifier(resource["gpu_uuid"], "resource.gpu_uuid")
    if not gpu_uuid.startswith("GPU-"):
        raise ValueError("resource.gpu_uuid가 NVIDIA GPU UUID 형식이 아니다")

    source = _exact(job["source"], {"repo_root", "commit", "files_lf"}, "source")
    repo = Path(source["repo_root"]).resolve()
    if not (repo / ".git").exists():
        raise ValueError("source.repo_root가 Git 저장소가 아니다")
    commit = source["commit"]
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("source.commit이 40자리 commit SHA가 아니다")
    ancestor = _run_git(repo, "merge-base", "--is-ancestor", commit, "HEAD")
    if ancestor.returncode != 0:
        raise ValueError("고정 source commit이 현재 HEAD의 조상이 아니다")
    files = source["files_lf"]
    if not isinstance(files, dict) or not files:
        raise ValueError("source.files_lf가 비어 있다")
    for relative, expected in sorted(files.items()):
        if (
            not isinstance(relative, str)
            or not _SOURCE_PATH.fullmatch(relative)
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            raise ValueError("source 상대 경로가 잘못됐다")
        expected = _hash(expected, f"source.files_lf[{relative}]")
        path = _regular_file(repo / relative, f"source file {relative}")
        if sha256_file(path, lf=True) != expected:
            raise ValueError(f"source file LF SHA가 다르다: {relative}")
        anchored = _run_git(repo, "show", f"{commit}:{relative}")
        if anchored.returncode != 0:
            raise ValueError(f"source commit에 파일이 없다: {relative}")
        anchored_lf = anchored.stdout.encode("utf-8").replace(b"\r\n", b"\n")
        if hashlib.sha256(anchored_lf).hexdigest() != expected:
            raise ValueError(f"source commit blob이 현재 결합과 다르다: {relative}")

    protocol = _exact(job["protocol"], {"path", "sha256"}, "protocol")
    protocol_path = _regular_file(Path(protocol["path"]), "protocol.path")
    protocol_sha = _hash(protocol["sha256"], "protocol.sha256")
    if sha256_file(protocol_path) != protocol_sha:
        raise ValueError("protocol bytes SHA가 다르다")
    protocol_body = _exact(
        read_object(protocol_path),
        {"schema", "method", "mode", "profile", "source_commit", "source_files_lf", "checkpoint_sha256", "resource"},
        "protocol body",
    )

    checkpoint = _exact(job["checkpoint"], {"path", "sha256", "logical_id"}, "checkpoint")
    checkpoint_path = _regular_file(Path(checkpoint["path"]), "checkpoint.path")
    checkpoint_sha = _hash(checkpoint["sha256"], "checkpoint.sha256")
    _identifier(checkpoint["logical_id"], "checkpoint.logical_id")
    if sha256_file(checkpoint_path) != checkpoint_sha:
        raise ValueError("checkpoint bytes SHA가 다르다")
    expected_protocol = {
        "schema": PROTOCOL_SCHEMA,
        "method": METHOD,
        "mode": job["mode"],
        "profile": fixed_profile(),
        "source_commit": commit,
        "source_files_lf": files,
        "checkpoint_sha256": checkpoint_sha,
        "resource": dict(resource),
    }
    if canonical(protocol_body) != canonical(expected_protocol):
        raise ValueError("protocol이 job source/profile/checkpoint와 다르다")

    admission = _exact(
        job["remote_gpu_admission"],
        {"path", "sha256", "resource_parent_sha256", "live_lease_proof_sha256"},
        "remote_gpu_admission",
    )
    admission_path = _regular_file(Path(admission["path"]), "remote_gpu_admission.path")
    admission_sha = _hash(admission["sha256"], "remote_gpu_admission.sha256")
    resource_parent_sha = _hash(
        admission["resource_parent_sha256"], "remote_gpu_admission.resource_parent_sha256"
    )
    live_lease_proof_sha = _hash(
        admission["live_lease_proof_sha256"], "remote_gpu_admission.live_lease_proof_sha256"
    )
    if sha256_file(admission_path) != admission_sha:
        raise ValueError("GPU admission receipt bytes SHA가 다르다")
    receipt = _exact(
        read_object(admission_path),
        {"schema", "status", "job_id", "resource_kind", "lease_id", "request_binding",
         "issued_at_utc", "expires_at_utc", "resource_id", "gpu_uuid",
         "resource_parent_sha256", "live_lease_proof_sha256"},
        "GPU admission receipt",
    )
    if (
        receipt["schema"] != ADMISSION_SCHEMA
        or receipt["status"] != "granted"
        or receipt["job_id"] != job_id
        or receipt["resource_kind"] != "remote_gpu"
        or not _IDENTIFIER.fullmatch(str(receipt["lease_id"]))
        or receipt["request_binding"] != request_binding(job)
        or receipt["resource_id"] != resource_id
        or receipt["gpu_uuid"] != gpu_uuid
        or receipt["resource_parent_sha256"] != resource_parent_sha
        or receipt["live_lease_proof_sha256"] != live_lease_proof_sha
    ):
        raise ValueError("GPU admission receipt 결합이 다르다")
    issued = parse_lease_timestamp(receipt["issued_at_utc"])
    expires = parse_lease_timestamp(receipt["expires_at_utc"])
    current = now or datetime.now(timezone.utc)  # noqa: UP017 - Python 3.10 호환
    if current.tzinfo is None or current.utcoffset() is None:
        raise ValueError("현재 시각에는 timezone이 필요하다")
    current = current.astimezone(timezone.utc)  # noqa: UP017 - Python 3.10 호환
    age = (current - issued).total_seconds()
    if not 0.0 <= age <= ADMISSION_MAX_AGE_S:
        raise ValueError("GPU admission receipt가 오래됐거나 미래다")
    if expires <= issued or expires < current + timedelta(seconds=EXECUTION_AUTHORIZATION_S):
        raise ValueError("GPU admission expiry가 고정 실행 시작 권한을 덮지 않는다")
    output = Path(job["output_directory"]).resolve()
    if output.exists():
        raise FileExistsError(output)
    return {
        "job": job,
        "job_path": job_path,
        "job_sha256": sha256_file(job_path),
        "repo_root": repo,
        "protocol_path": protocol_path,
        "protocol_sha256": protocol_sha,
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": checkpoint_sha,
        "admission_path": admission_path,
        "admission_sha256": admission_sha,
        "request_binding": receipt["request_binding"],
        "resource": dict(resource),
        "resource_parent_sha256": receipt["resource_parent_sha256"],
        "live_lease_proof_sha256": receipt["live_lease_proof_sha256"],
        "admission_issued_at_utc": receipt["issued_at_utc"],
        "admission_expires_at_utc": receipt["expires_at_utc"],
        "output_directory": output,
    }


def _finite(value: Any, anchor: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value):
        raise ValueError(f"{anchor}는 유한 수여야 한다")
    return float(value)


def _integer(value: Any, anchor: str) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{anchor}는 정수여야 한다")
    return int(value)


def _vector3(value: Any, anchor: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{anchor}는 길이 3 벡터여야 한다")
    return [_finite(item, f"{anchor}[{index}]") for index, item in enumerate(value)]


def validate_atomic_sample(value: Any, index: int) -> tuple[bytes, dict[str, Any]]:
    row = _exact(
        value,
        {"frame_index", "capture_ordinal", "simulation_step", "rgb", "state", "camera", "render_timing"},
        "atomic sample",
    )
    simulation_step = row["simulation_step"]
    if (
        row["frame_index"] != index
        or row["capture_ordinal"] != index
        or isinstance(simulation_step, bool)
        or not isinstance(simulation_step, int)
        or simulation_step < 0
    ):
        raise ValueError("frame index, capture 순서, 실제 simulation step 결합이 잘못됐다")
    rgb = row["rgb"]
    if not isinstance(rgb, bytes) or len(rgb) != WIDTH * HEIGHT * 3:
        raise ValueError("RGB frame bytes 크기가 고정 profile과 다르다")
    state = _exact(
        row["state"],
        {"command", "motion", "pose", "root_position_world_m", "simulation_time_s", "common_step_counter"},
        "state",
    )
    command = _vector3(state["command"], "state.command")
    motion = _vector3(state["motion"], "state.motion")
    pose = _exact(state["pose"], {"base_height_m", "up_axis_z"}, "state.pose")
    pose_values = {key: _finite(pose[key], f"state.pose.{key}") for key in pose}
    root = _vector3(state["root_position_world_m"], "state.root_position_world_m")
    simulation_time = _finite(state["simulation_time_s"], "state.simulation_time_s")
    render_timing = validate_render_timing(row["render_timing"], simulation_time)
    common_step = state["common_step_counter"]
    if isinstance(common_step, bool) or not isinstance(common_step, int) or common_step != simulation_step:
        raise ValueError("state common_step_counter가 실제 simulation step과 다르다")
    if command != [1.0, 0.0, 0.0]:
        raise ValueError("실제 command가 고정 body command와 다르다")
    camera = _exact(row["camera"], {"eye_world_m", "lookat_world_m"}, "camera")
    eye = _vector3(camera["eye_world_m"], "camera.eye_world_m")
    lookat = _vector3(camera["lookat_world_m"], "camera.lookat_world_m")
    expected_eye = [root[i] + [3.0, 3.0, 2.0][i] for i in range(3)]
    expected_lookat = [root[i] + [0.0, 0.0, 0.35][i] for i in range(3)]
    if any(abs(a - b) > 1e-9 for a, b in zip(eye, expected_eye, strict=True)):
        raise ValueError("camera eye가 root 평행 추적 profile과 다르다")
    if any(abs(a - b) > 1e-9 for a, b in zip(lookat, expected_lookat, strict=True)):
        raise ValueError("camera lookat이 root 평행 추적 profile과 다르다")
    record = {
        "frame_index": index,
        "capture_ordinal": index,
        "simulation_step": simulation_step,
        "state": {
            "command": command,
            "motion": motion,
            "pose": pose_values,
            "root_position_world_m": root,
            "simulation_time_s": simulation_time,
            "common_step_counter": common_step,
        },
        "camera": {"eye_world_m": eye, "lookat_world_m": lookat},
        "render_timing": render_timing,
        "rgb_sha256": hashlib.sha256(rgb).hexdigest(),
    }
    return rgb, record


def rgb24_bytes(array: Any) -> bytes:
    """CPU RGBA/RGB 배열을 연속 C-order RGB24 bytes로 고정한다."""
    shape = getattr(array, "shape", None)
    dtype = getattr(array, "dtype", None)
    if (
        not isinstance(shape, tuple)
        or len(shape) != 3
        or shape[0] != HEIGHT
        or shape[1] != WIDTH
        or shape[2] not in (3, 4)
        or str(dtype) != "uint8"
    ):
        raise ValueError("annotator RGB/RGBA shape 또는 dtype이 고정 profile과 다르다")
    try:
        rgb = array[:, :, :3].copy(order="C")
        result = rgb.tobytes(order="C")
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("annotator RGB를 C-order RGB24 bytes로 바꿀 수 없다") from error
    if len(result) != WIDTH * HEIGHT * 3:
        raise ValueError("C-order RGB24 bytes 크기가 고정 profile과 다르다")
    return result


def validate_render_state_unchanged(before: Mapping[str, Any], after: Mapping[str, Any]) -> None:
    """render update가 결합 대상의 physics 상태를 바꾸면 frame을 거부한다."""
    if canonical(before) != canonical(after):
        raise ValueError("RGB render update 전후 robot state가 달라졌다")


def validate_render_timing(value: Any, simulation_time_s: float) -> dict[str, Any]:
    """RGB와 함께 읽은 Replicator 시각을 physics 상태 시각에 결합한다."""
    timing = _exact(value, {"read_times", "reference_time"}, "render timing")
    read_times = _exact(
        timing["read_times"],
        set(ISAAC_READ_TIMES_FIELDS),
        "IsaacReadTimes",
    )
    reference = _exact(
        timing["reference_time"],
        {"referenceTimeNumerator", "referenceTimeDenominator"},
        "ReferenceTime",
    )
    normalized_read_times = {
        key: (
            _integer(read_times[key], f"IsaacReadTimes.{key}")
            if key in ISAAC_READ_TIMES_INTEGER_FIELDS
            else _finite(read_times[key], f"IsaacReadTimes.{key}")
        )
        for key in ISAAC_READ_TIMES_FIELDS
    }
    normalized_reference = {
        key: _finite(item, f"ReferenceTime.{key}") for key, item in reference.items()
    }
    if abs(normalized_read_times["simulationTime"] - simulation_time_s) > 1e-9:
        raise ValueError("RGB render의 IsaacReadTimes와 simulation state 시각이 다르다")
    if normalized_read_times["rationalTimeOfSimDenominator"] == 0:
        raise ValueError("render timestamp 분모가 0이다")
    if normalized_reference["referenceTimeDenominator"] == 0:
        raise ValueError("render timestamp 분모가 0이다")
    return {"read_times": normalized_read_times, "reference_time": normalized_reference}


def assess_render_sync_smoke(captures: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """warmup 뒤 A-B-A 정지 capture가 해당 renderer 조합에서 재현되는지 판정한다."""
    if len(captures) != 4 or [row.get("label") for row in captures] != ["warmup", "A", "B", "A"]:
        raise ValueError("render sync smoke는 warmup-A-B-A 네 capture여야 한다")
    rows = []
    for index, raw in enumerate(captures):
        row = _exact(
            raw,
            {"label", "rgb", "state_before", "state_after", "simulation_time_before_s",
             "simulation_time_after_s", "read_times", "reference_time", "camera"},
            f"smoke capture[{index}]",
        )
        rgb = row["rgb"]
        if not isinstance(rgb, bytes) or len(rgb) != WIDTH * HEIGHT * 3:
            raise ValueError("render sync smoke RGB bytes가 잘못됐다")
        validate_render_state_unchanged(row["state_before"], row["state_after"])
        before = _finite(row["simulation_time_before_s"], "simulation_time_before_s")
        after = _finite(row["simulation_time_after_s"], "simulation_time_after_s")
        if before != after:
            raise ValueError("0-delta render capture가 simulation time을 진행했다")
        timing = validate_render_timing(
            {"read_times": row["read_times"], "reference_time": row["reference_time"]}, before
        )
        rows.append(
            {"label": row["label"], "rgb_sha256": hashlib.sha256(rgb).hexdigest(),
             "simulation_time_s": before, **timing, "camera": dict(row["camera"])}
        )
    if rows[1]["rgb_sha256"] != rows[3]["rgb_sha256"]:
        raise ValueError("A camera 복귀 RGB가 같은 정지 상태에서 재현되지 않았다")
    if rows[1]["rgb_sha256"] == rows[2]["rgb_sha256"]:
        raise ValueError("서로 다른 A/B camera RGB가 구분되지 않았다")
    return {
        "schema": "vision_render_sync_smoke_v1",
        "status": "pass",
        "capture_order": ["warmup_discard", "A", "B", "A"],
        "policy_steps": 0,
        "capture_phase_physics_steps": 0,
        "environment_reset_performed": True,
        "delta_time_s": 0.0,
        "rt_subframes": 1,
        "pause_timeline": False,
        "renderer_combination_only": True,
        "universal_zero_lag_claim": False,
        "captures": rows,
    }


def collect_render_sync_smoke(
    capture: Callable[[str, list[float]], Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    """고정 warmup-A-B-A 순서를 SDK adapter에 적용하고 CPU 판정기로 넘긴다."""
    offset_a = list(fixed_profile()["camera"]["eye_offset_m"])
    offset_b = [-offset_a[0], offset_a[1], offset_a[2]]
    captures = [capture("warmup", offset_a), capture("A", offset_a),
                capture("B", offset_b), capture("A", offset_a)]
    return captures, assess_render_sync_smoke(captures)


def capture_lockstep(
    sample: Callable[[int], Mapping[str, Any]], encoder: Encoder, *, frame_count: int = FRAME_COUNT
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if frame_count != FRAME_COUNT:
        raise ValueError("고정 frame count가 바뀌었다")
    records: list[dict[str, Any]] = []
    first_step: int | None = None
    first_time: float | None = None
    try:
        for index in range(frame_count):
            rgb, record = validate_atomic_sample(sample(index), index)
            if first_step is None:
                first_step = record["simulation_step"]
                first_time = record["state"]["simulation_time_s"]
            elif (
                record["simulation_step"] != first_step + index
                or abs(record["state"]["simulation_time_s"] - first_time - index * FRAME_PERIOD_S) > 1e-6
            ):
                raise ValueError("실제 simulation step/time과 frame capture 순서가 lockstep이 아니다")
            encoder.write(rgb)
            records.append(record)
        receipt = encoder.finish()
    except BaseException:
        encoder.abort()
        raise
    if receipt.get("natural_exit") is not True or receipt.get("frames_written") != FRAME_COUNT:
        raise RuntimeError("ffmpeg가 모든 frame을 자연 완료로 기록하지 않았다")
    return records, receipt


class RawRgbFfmpegEncoder:
    def __init__(self, output: Path, *, popen: Callable[..., Any] = subprocess.Popen):
        self.output = Path(output)
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.argv = [
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-f", "rawvideo",
            "-pixel_format", "rgb24", "-video_size", f"{WIDTH}x{HEIGHT}", "-framerate", str(FPS),
            "-i", "pipe:0", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags",
            "+faststart", "-y", str(self.output),
        ]
        self.process = popen(self.argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.frames_written = 0
        self.bytes_written = 0
        self.finished = False

    def write(self, rgb: bytes) -> None:
        if self.finished or self.process.stdin is None:
            raise RuntimeError("ffmpeg stdin을 쓸 수 없다")
        self.process.stdin.write(rgb)
        self.frames_written += 1
        self.bytes_written += len(rgb)

    def finish(self) -> dict[str, Any]:
        if self.finished:
            raise RuntimeError("ffmpeg encoder가 이미 종료됐다")
        self.finished = True
        if self.process.stdin is not None:
            self.process.stdin.close()
            self.process.stdin = None
        stdout, stderr = self.process.communicate()
        if self.process.returncode != 0:
            raise RuntimeError(f"ffmpeg encode 실패: {stderr.decode('utf-8', errors='replace')[-1000:]}")
        if not self.output.is_file() or self.output.stat().st_size <= 0:
            raise RuntimeError("ffmpeg output이 없거나 비어 있다")
        return {
            "tool": "ffmpeg",
            "argv": [*self.argv[:-1], self.output.name],
            "pid": self.process.pid,
            "returncode": self.process.returncode,
            "natural_exit": True,
            "frames_written": self.frames_written,
            "bytes_written": self.bytes_written,
            "stdout_bytes": len(stdout),
            "stderr": stderr.decode("utf-8", errors="replace"),
        }

    def abort(self) -> dict[str, Any]:
        if not self.finished:
            self.finished = True
            if self.process.stdin is not None:
                self.process.stdin.close()
                self.process.stdin = None
            if self.process.poll() is None:
                self.process.kill()
            stdout, stderr = self.process.communicate()
        else:
            stdout, stderr = b"", b""
        return {
            "tool": "ffmpeg", "pid": self.process.pid, "returncode": self.process.returncode,
            "natural_exit": False, "frames_written": self.frames_written,
            "bytes_written": self.bytes_written, "stdout_bytes": len(stdout),
            "stderr": stderr.decode("utf-8", errors="replace"),
        }


def probe_video_timing(
    video: Path, *, runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run
) -> dict[str, Any]:
    argv = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
            "frame=best_effort_timestamp_time", "-of", "json", str(video)]
    result = runner(
        argv,
        capture_output=True, text=True, encoding="utf-8", errors="replace", check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe 실패: {result.stderr[-1000:]}")
    try:
        rows = json.loads(result.stdout)["frames"]
        absolute = [float(row["best_effort_timestamp_time"]) for row in rows]
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise ValueError("ffprobe frame PTS를 읽을 수 없다") from error
    if len(absolute) != FRAME_COUNT or any(not math.isfinite(value) for value in absolute):
        raise ValueError("영상 frame 수 또는 PTS가 고정 profile과 다르다")
    first = absolute[0]
    relative = [round(value - first, 9) for value in absolute]
    for index, value in enumerate(relative):
        expected = index * FRAME_PERIOD_S
        if abs(value - expected) > PTS_TOLERANCE_S:
            raise ValueError(f"frame[{index}] 실제 PTS가 50fps state step과 다르다")
        if index and value <= relative[index - 1]:
            raise ValueError("frame PTS가 엄격한 오름차순이 아니다")
    return {
        "tool": "ffprobe",
        "argv": [*argv[:-1], Path(video).name],
        "source_sha256": sha256_file(video),
        "first_frame_pts_s": first,
        "frame_count": len(relative),
        "fps": FPS,
        "frame_period_s": FRAME_PERIOD_S,
        "pts_tolerance_s": PTS_TOLERANCE_S,
        "relative_actual_pts_s": relative,
    }


def probe_frame_pts(
    video: Path, *, runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run
) -> list[float]:
    """호출부가 필요한 실제 상대 PTS만 돌려주는 CPU 편의 API."""
    return probe_video_timing(video, runner=runner)["relative_actual_pts_s"]


def bind_actual_pts(records: Sequence[Mapping[str, Any]], pts: Sequence[float]) -> list[dict[str, Any]]:
    if len(records) != FRAME_COUNT or len(pts) != FRAME_COUNT:
        raise ValueError("state record와 video PTS 수가 다르다")
    bound: list[dict[str, Any]] = []
    for index, (record, timestamp) in enumerate(zip(records, pts, strict=True)):
        if record["frame_index"] != index or record["capture_ordinal"] != index:
            raise ValueError("state record 순서가 frame 순서와 다르다")
        if index:
            if record["simulation_step"] != records[0]["simulation_step"] + index:
                raise ValueError("실제 common simulation step이 capture마다 1씩 진행하지 않았다")
            relative_sim_time = record["state"]["simulation_time_s"] - records[0]["state"]["simulation_time_s"]
            if abs(relative_sim_time - index * FRAME_PERIOD_S) > 1e-6:
                raise ValueError("실제 simulation time이 50fps capture 순서와 다르다")
        bound.append({**record, "actual_timestamp_s": float(timestamp)})
    return bound


def _at_timestamp(records: Sequence[Mapping[str, Any]], timestamp: float) -> Mapping[str, Any]:
    matches = [row for row in records if row["actual_timestamp_s"] == timestamp]
    if len(matches) != 1:
        raise ValueError(f"actual PTS {timestamp:.2f}s에 대응하는 state가 하나가 아니다")
    return matches[0]


def build_trace_and_v01_specs(
    validated: Mapping[str, Any], video: Path, bound: Sequence[Mapping[str, Any]],
    encoder_receipt: Mapping[str, Any], timing_receipt: Mapping[str, Any], output: Path,
) -> dict[str, Path]:
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    video = _regular_file(Path(video), "video")
    video_sha = sha256_file(video)
    if (
        timing_receipt.get("source_sha256") != video_sha
        or timing_receipt.get("frame_count") != FRAME_COUNT
        or timing_receipt.get("relative_actual_pts_s") != [row["actual_timestamp_s"] for row in bound]
    ):
        raise ValueError("ffprobe timing receipt가 video와 결합된 state records와 다르다")
    checkpoint_sha = validated["checkpoint_sha256"]
    job = validated["job"]
    trace = {
        "schema": TRACE_SCHEMA,
        "method": METHOD,
        "job_id": job["job_id"],
        "profile": fixed_profile(),
        "profile_sha256": digest(fixed_profile()),
        "video_sha256": video_sha,
        "checkpoint_sha256": checkpoint_sha,
        "time_basis": fixed_profile()["time_basis"],
        "frame_count": FRAME_COUNT,
        "records": list(bound),
        "encoder_receipt": dict(encoder_receipt),
        "timing_receipt": dict(timing_receipt),
        "render_alignment": {
            "camera_pose_applied_before_render": True,
            "capture_primitive": fixed_profile()["camera"]["capture_primitive"],
            "rt_subframes": fixed_profile()["camera"]["rt_subframes"],
            "pause_timeline": fixed_profile()["camera"]["pause_timeline"],
            "delta_time_s": fixed_profile()["camera"]["delta_time_s"],
            "robot_state_unchanged_across_render_enforced": True,
            "rgb_timing_matches_state_enforced": True,
            "annotator_cpu_copy": True,
            "zero_update_pixel_lag_confirmed": False,
            "gpu_smoke_required_before_model_use": True,
        },
    }
    trace_path = output / "raw_trace.json"
    write_new(trace_path, trace)
    outputs: dict[str, Path] = {"trace": trace_path}
    for window_index, timestamps in enumerate(WINDOWS, 1):
        samples = []
        for timestamp in timestamps:
            row = _at_timestamp(bound, timestamp)
            state = row["state"]
            samples.append(
                {
                    "timestamp_s": row["actual_timestamp_s"],
                    "command": dict(zip(("linear_x_mps", "linear_y_mps", "yaw_rate_rps"), state["command"], strict=True)),
                    "motion": dict(zip(("linear_x_mps", "linear_y_mps", "yaw_rate_rps"), state["motion"], strict=True)),
                    "pose": dict(state["pose"]),
                }
            )
        measurement = {
            "schema": MEASUREMENT_SCHEMA,
            "play_run_id": f"{job['job_id']}-w{window_index}",
            "video_sha256": video_sha,
            "checkpoint_sha256": checkpoint_sha,
            "command_name": "base_velocity",
            "coordinate_frames": {"command": "body", "motion": "body"},
            "samples": samples,
        }
        measurement_path = output / f"window_{window_index}_measurements.json"
        write_new(measurement_path, measurement)
        spec = {
            "schema": PREPARE_SCHEMA,
            "play_run_id": measurement["play_run_id"],
            "source_lineage_id": f"{job['job_id']}-same-play",
            "parent_source_lineage_id": None,
            "video": {"path": str(video), "logical_id": "same_play_video", "play_run_id": measurement["play_run_id"]},
            "checkpoint": {"path": str(validated["checkpoint_path"]), "logical_id": job["checkpoint"]["logical_id"],
                           "play_run_id": measurement["play_run_id"]},
            "frames": [{"timestamp_s": value, "play_run_id": measurement["play_run_id"],
                        "video_sha256": video_sha} for value in timestamps],
            "measurement": {"path": str(measurement_path), "logical_id": f"same_play_measurements_w{window_index}",
                            "play_run_id": measurement["play_run_id"]},
            "play": {"task": TASK, "camera": fixed_profile()["camera"]["identifier"],
                     "command_name": "base_velocity", "command_frame": "body", "motion_frame": "body",
                     "settings_sha256": digest(fixed_profile())},
        }
        spec_path = output / f"window_{window_index}_v01_spec.json"
        write_new(spec_path, spec)
        outputs[f"measurement_{window_index}"] = measurement_path
        outputs[f"spec_{window_index}"] = spec_path
    return outputs


def result_body(validated: Mapping[str, Any], artifacts: Mapping[str, Path], primary: Path) -> dict[str, Any]:
    mode = validated["job"]["mode"]
    return {
        "schema": RESULT_SCHEMA,
        "completed": True,
        "job_sha256": validated["job_sha256"],
        "protocol_sha256": validated["protocol_sha256"],
        "checkpoint_sha256": validated["checkpoint_sha256"],
        "request_binding": validated["request_binding"],
        "resource": dict(validated["resource"]),
        "resource_parent_sha256": validated["resource_parent_sha256"],
        "live_lease_proof_sha256": validated["live_lease_proof_sha256"],
        "admission_sha256": validated["admission_sha256"],
        "admission_issued_at_utc": validated["admission_issued_at_utc"],
        "admission_expires_at_utc": validated["admission_expires_at_utc"],
        "profile_sha256": digest(fixed_profile()),
        "mode": mode,
        "primary_artifact_sha256": sha256_file(primary),
        "video_sha256": sha256_file(primary) if mode == RECORD_MODE else None,
        "artifacts": {
            key: {"file": path.name, "sha256": sha256_file(path), "bytes": path.stat().st_size}
            for key, path in artifacts.items()
        },
        "fixed_evaluation_used_as_behavior_ground_truth": False,
        "independent_visual_evidence_source_count": 1,
        "zero_update_pixel_lag_confirmed": False,
        "gpu_smoke_required_before_model_use": mode == RECORD_MODE,
    }


def write_failure(path: Path, *, stage: str, error: BaseException, job_sha256: str | None) -> None:
    write_new(
        path,
        {"schema": FAILURE_SCHEMA, "completed": False, "stage": stage,
         "error_type": type(error).__name__, "error_message": str(error), "job_sha256": job_sha256},
    )
