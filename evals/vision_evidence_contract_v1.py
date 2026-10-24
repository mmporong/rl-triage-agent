"""영상, checkpoint, 동기화 운동 수치를 결합하는 CPU 입력 계약.

모델 view에는 여섯 프레임과 같은 Play에서 기록한 물리 수치만 들어간다.
실행 식별자, 파일명, 평가 결과는 provenance에 남고 모델 view에는 노출되지
않는다. 여섯 프레임은 한 영상에서 뽑은 표본이므로 독립 근거 여섯 개로
계산하지 않는다.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import itertools
import json
import math
import os
import re
import shutil
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import Any

CONTRACT_SCHEMA = "vision_evidence_contract_v1"
PREPARE_SCHEMA = "vision_evidence_prepare_v1"
MEASUREMENT_SCHEMA = "vision_play_measurements_v1"
MODEL_VIEW_SCHEMA = "vision_evidence_model_view_v1"
FRAME_COUNT = 6
MAX_JPEG_BYTES = 180_000
FRAME_PTS_TOLERANCE_S = 0.05
_SHA256 = re.compile(r"[0-9a-f]{64}")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}")
_VECTOR_KEYS = ("linear_x_mps", "linear_y_mps", "yaw_rate_rps")
_POSE_KEYS = ("base_height_m", "up_axis_z")
_COMMAND_NAMES = frozenset({"base_velocity"})
_COMMAND_FRAMES = frozenset({"body"})
_MOTION_FRAMES = frozenset({"body", "world"})

SYSTEM_PROMPT = (
    "Observe only the supplied frames and synchronized numeric measurements. "
    "Describe visible locomotion evidence without declaring a fault cause, repair, "
    "fixed-evaluation outcome, or recovery. The six frames are samples from one "
    "video source and count as one visual evidence source. Missing numbers are null."
)
QUESTION = (
    "Using the ordered frames and synchronized values, report what is visibly "
    "supported about movement, staying in place, and loss of upright posture. "
    "Cite frame IDs and timestamps."
)


def canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
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


def _exact(value: Any, keys: set[str], anchor: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{anchor} 필드가 계약과 다르다")
    return value


def _identifier(value: Any, anchor: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"{anchor} 식별자가 잘못됐다")
    return value


def _logical_id(value: Any, anchor: str) -> str:
    result = _identifier(value, anchor)
    if result.startswith("/") or re.match(r"^[A-Za-z]:/", result) or ".." in result.split("/"):
        raise ValueError(f"{anchor}에는 절대·상위 경로를 쓸 수 없다")
    return result


def _hash(value: Any, anchor: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{anchor} SHA-256이 잘못됐다")
    return value


def _number_or_none(value: Any, anchor: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{anchor} 값은 유한 수 또는 null이어야 한다")
    return float(value)


def _timestamp(value: Any, anchor: str) -> float:
    result = _number_or_none(value, anchor)
    if result is None or result < 0:
        raise ValueError(f"{anchor} 시각이 잘못됐다")
    return result


def _source_file(value: Any, anchor: str, run_id: str) -> tuple[Path, str]:
    row = _exact(value, {"path", "logical_id", "play_run_id"}, anchor)
    if row["play_run_id"] != run_id:
        raise ValueError(f"{anchor}가 같은 Play 실행에 결합되지 않았다")
    path = Path(row["path"]).resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"{anchor} 파일이 없거나 비어 있다")
    return path, _logical_id(row["logical_id"], f"{anchor}.logical_id")


def _validate_jpeg(path: Path, anchor: str, *, enforce_size: bool = True) -> None:
    size = path.stat().st_size
    if size < 4 or (enforce_size and size > MAX_JPEG_BYTES):
        raise ValueError(f"{anchor} JPEG 크기가 잘못됐다")
    with path.open("rb") as stream:
        start = stream.read(2)
        stream.seek(-2, os.SEEK_END)
        end = stream.read(2)
    if start != b"\xff\xd8" or end != b"\xff\xd9":
        raise ValueError(f"{anchor}는 JPEG 바이트가 아니다")
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,width,height", "-of", "json", str(path),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    try:
        probe = json.loads(result.stdout)
        streams = probe["streams"]
        stream = streams[0]
    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as error:
        raise ValueError(f"{anchor} JPEG를 디코딩할 수 없다") from error
    if (
        result.returncode != 0
        or len(streams) != 1
        or stream.get("codec_name") != "mjpeg"
        or not isinstance(stream.get("width"), int)
        or not isinstance(stream.get("height"), int)
        or stream["width"] <= 0
        or stream["height"] <= 0
    ):
        raise ValueError(f"{anchor} JPEG를 디코딩할 수 없다")
    decoded = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-xerror", "-err_detect", "explode",
            "-i", str(path), "-frames:v", "1", "-f", "null", "-",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if decoded.returncode != 0 or decoded.stderr.strip():
        raise ValueError(f"{anchor} JPEG 전체 scan을 디코딩할 수 없다")


def _ffmpeg_version() -> str:
    result = subprocess.run(
        ["ffmpeg", "-version"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )
    first_line = result.stdout.splitlines()[0].strip() if result.stdout.splitlines() else ""
    if not first_line.startswith("ffmpeg version "):
        raise ValueError("ffmpeg 버전을 확인할 수 없다")
    return first_line


def _source_frame_pts(source: Path) -> list[float]:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "frame=best_effort_timestamp_time", "-of", "json", str(source),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )
    try:
        rows = json.loads(result.stdout)["frames"]
        values = [float(row["best_effort_timestamp_time"]) for row in rows]
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise ValueError("원본 영상 frame PTS를 읽을 수 없다") from error
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("원본 영상 frame PTS가 비어 있거나 유한하지 않다")
    if any(current <= prior for prior, current in itertools.pairwise(values)):
        raise ValueError("원본 영상 frame PTS가 엄격한 오름차순이 아니다")
    return values


def _resolve_frame_requests(source: Path, requested_timestamps: list[float]) -> tuple[float, list[dict[str, Any]]]:
    source_pts = _source_frame_pts(source)
    first_pts = source_pts[0]
    relative_pts = [round(value - first_pts, 9) for value in source_pts]
    resolved = []
    prior_index = -1
    for requested in requested_timestamps:
        index = min(range(len(relative_pts)), key=lambda candidate: (abs(relative_pts[candidate] - requested), candidate))
        actual = relative_pts[index]
        error = abs(actual - requested)
        if error > FRAME_PTS_TOLERANCE_S:
            raise ValueError(
                f"요청 frame 시각과 실제 PTS 차이가 {FRAME_PTS_TOLERANCE_S:.3f}s 허용값을 넘었다"
            )
        if index <= prior_index:
            raise ValueError("요청 frame들이 서로 다른 오름차순 source frame으로 해석되지 않았다")
        prior_index = index
        resolved.append(
            {
                "source_frame_index": index,
                "request_timestamp_s": requested,
                "actual_timestamp_s": actual,
                "source_frame_pts_s": source_pts[index],
                "absolute_error_s": round(error, 9),
            }
        )
    return first_pts, resolved


def _extract_frames(
    source: Path,
    destination: Path,
    source_first_pts_s: float,
    resolved: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    version = _ffmpeg_version()
    records = []
    commands = []
    for index, timing in enumerate(resolved, 1):
        output = destination / f"frame_{index:02d}.jpg"
        selected: tuple[int, int] | None = None
        for width, quality in ((640, 4), (480, 7)):
            command = [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(source),
                "-vf", f"select=eq(n\\,{timing['source_frame_index']}),scale='min({width},iw)':-2",
                "-vsync", "0", "-frames:v", "1", "-q:v", str(quality), "-y", str(output),
            ]
            subprocess.run(command, check=True, capture_output=True)
            _validate_jpeg(output, f"frame[{index}]", enforce_size=False)
            if output.stat().st_size <= MAX_JPEG_BYTES:
                selected = (width, quality)
                break
        if selected is None:
            raise ValueError(f"frame[{index}] JPEG 크기가 상한을 넘었다")
        records.append(
            {
                "frame_id": f"f{index:02d}",
                "path": f"artifacts/frames/frame_{index:02d}.jpg",
                "timestamp_s": timing["actual_timestamp_s"],
                "sha256": sha256(output),
                "bytes": output.stat().st_size,
            }
        )
        commands.append(
            {
                "frame_id": f"f{index:02d}",
                "source_frame_index": timing["source_frame_index"],
                "request_timestamp_s": timing["request_timestamp_s"],
                "actual_timestamp_s": timing["actual_timestamp_s"],
                "source_frame_pts_s": timing["source_frame_pts_s"],
                "absolute_error_s": timing["absolute_error_s"],
                "source": "artifacts/source.video",
                "output": f"artifacts/frames/frame_{index:02d}.jpg",
                "scale_width_ceiling": selected[0],
                "jpeg_quality": selected[1],
                "argv": [
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "artifacts/source.video",
                    "-vf", f"select=eq(n\\,{timing['source_frame_index']}),scale='min({selected[0]},iw)':-2",
                    "-vsync", "0", "-frames:v", "1", "-q:v", str(selected[1]), "-y",
                    f"artifacts/frames/frame_{index:02d}.jpg",
                ],
            }
        )
    return records, {
        "tool": "ffmpeg",
        "tool_version": version,
        "source": "artifacts/source.video",
        "source_sha256": sha256(source),
        "time_basis": "play_relative_from_first_decoded_video_frame_pts",
        "source_first_pts_s": source_first_pts_s,
        "pts_tolerance_s": FRAME_PTS_TOLERANCE_S,
        "commands": commands,
    }


def _play(value: Any) -> dict[str, str]:
    row = _exact(
        value,
        {"task", "camera", "command_name", "command_frame", "motion_frame", "settings_sha256"},
        "play",
    )
    result = {key: _identifier(row[key], f"play.{key}") for key in ("task", "camera")}
    allowed = {
        "command_name": _COMMAND_NAMES,
        "command_frame": _COMMAND_FRAMES,
        "motion_frame": _MOTION_FRAMES,
    }
    for key, values in allowed.items():
        if row[key] not in values:
            raise ValueError(f"play.{key} 값이 등록 범위 밖이다")
        result[key] = row[key]
    result["settings_sha256"] = _hash(row["settings_sha256"], "play.settings_sha256")
    return result


def _sample(value: Any, expected_timestamp: float, index: int) -> dict[str, Any]:
    row = _exact(value, {"timestamp_s", "command", "motion", "pose"}, f"sample[{index}]")
    timestamp = _timestamp(row["timestamp_s"], f"sample[{index}].timestamp_s")
    if timestamp != expected_timestamp:
        raise ValueError("measurement 시각이 frame 시각과 다르다")
    command = _exact(row["command"], set(_VECTOR_KEYS), f"sample[{index}].command")
    motion = _exact(row["motion"], set(_VECTOR_KEYS), f"sample[{index}].motion")
    pose = _exact(row["pose"], set(_POSE_KEYS), f"sample[{index}].pose")
    return {
        "timestamp_s": timestamp,
        "command": {key: _number_or_none(command[key], f"sample[{index}].command.{key}") for key in _VECTOR_KEYS},
        "motion": {key: _number_or_none(motion[key], f"sample[{index}].motion.{key}") for key in _VECTOR_KEYS},
        "pose": {key: _number_or_none(pose[key], f"sample[{index}].pose.{key}") for key in _POSE_KEYS},
    }


def _measurement(
    value: dict[str, Any],
    *,
    run_id: str,
    video_sha: str,
    checkpoint_sha: str,
    play: dict[str, str],
    frame_timestamps: list[float],
) -> list[dict[str, Any]]:
    row = _exact(
        value,
        {
            "schema",
            "play_run_id",
            "video_sha256",
            "checkpoint_sha256",
            "command_name",
            "coordinate_frames",
            "samples",
        },
        "measurement",
    )
    frames = _exact(row["coordinate_frames"], {"command", "motion"}, "measurement.coordinate_frames")
    if (
        row["schema"] != MEASUREMENT_SCHEMA
        or row["play_run_id"] != run_id
        or row["video_sha256"] != video_sha
        or row["checkpoint_sha256"] != checkpoint_sha
        or row["command_name"] != play["command_name"]
        or frames != {"command": play["command_frame"], "motion": play["motion_frame"]}
    ):
        raise ValueError("measurement가 같은 Play 실행에 결합되지 않았다")
    samples = row["samples"]
    if not isinstance(samples, list) or len(samples) != FRAME_COUNT:
        raise ValueError("measurement에는 frame별 여섯 sample이 필요하다")
    result = [_sample(sample, frame_timestamps[index], index) for index, sample in enumerate(samples)]
    if not any(value is not None for sample in result for group in ("command", "motion", "pose")
               for value in sample[group].values()):
        raise ValueError("모든 수치가 누락이면 measurement를 null로 기록해야 한다")
    return result


def _missing_samples(frame_timestamps: list[float]) -> list[dict[str, Any]]:
    return [
        {
            "timestamp_s": timestamp,
            "command": {key: None for key in _VECTOR_KEYS},
            "motion": {key: None for key in _VECTOR_KEYS},
            "pose": {key: None for key in _POSE_KEYS},
        }
        for timestamp in frame_timestamps
    ]


def prepare(spec_path: Path, output_dir: Path) -> Path:
    """입력 파일을 해시로 봉인하고 자체 포함된 write-once 계약을 만든다."""
    spec = _exact(
        read_object(spec_path),
        {
            "schema", "play_run_id", "source_lineage_id", "parent_source_lineage_id",
            "video", "checkpoint", "frames", "measurement", "play",
        },
        "prepare spec",
    )
    if spec["schema"] != PREPARE_SCHEMA:
        raise ValueError("prepare schema가 다르다")
    run_id = _identifier(spec["play_run_id"], "play_run_id")
    lineage_id = _identifier(spec["source_lineage_id"], "source_lineage_id")
    parent_lineage_id = spec["parent_source_lineage_id"]
    if parent_lineage_id is not None:
        parent_lineage_id = _identifier(parent_lineage_id, "parent_source_lineage_id")
        if parent_lineage_id == lineage_id:
            raise ValueError("부모 계보는 현재 계보와 달라야 한다")
    video_path, video_logical = _source_file(spec["video"], "video", run_id)
    checkpoint_path, checkpoint_logical = _source_file(spec["checkpoint"], "checkpoint", run_id)
    play = _play(spec["play"])
    video_sha = sha256(video_path)
    checkpoint_sha = sha256(checkpoint_path)

    frames = spec["frames"]
    if not isinstance(frames, list) or len(frames) != FRAME_COUNT:
        raise ValueError("정확히 여섯 frame이 필요하다")
    frame_timestamps: list[float] = []
    previous = -1.0
    for index, value in enumerate(frames, 1):
        row = _exact(
            value,
            {"timestamp_s", "play_run_id", "video_sha256"},
            f"frame[{index}]",
        )
        if row["play_run_id"] != run_id or row["video_sha256"] != video_sha:
            raise ValueError("frame이 같은 Play 영상에 결합되지 않았다")
        timestamp = _timestamp(row["timestamp_s"], f"frame[{index}].timestamp_s")
        if timestamp <= previous:
            raise ValueError("frame 시각은 엄격한 오름차순이어야 한다")
        previous = timestamp
        frame_timestamps.append(timestamp)
    source_first_pts_s, resolved_frames = _resolve_frame_requests(video_path, frame_timestamps)
    frame_timestamps = [row["actual_timestamp_s"] for row in resolved_frames]

    measurement_path: Path | None = None
    measurement_logical: str | None = None
    measurement_sha: str | None = None
    if spec["measurement"] is None:
        samples = _missing_samples(frame_timestamps)
    else:
        measurement_path, measurement_logical = _source_file(spec["measurement"], "measurement", run_id)
        measurement_sha = sha256(measurement_path)
        samples = _measurement(
            read_object(measurement_path),
            run_id=run_id,
            video_sha=video_sha,
            checkpoint_sha=checkpoint_sha,
            play=play,
            frame_timestamps=frame_timestamps,
        )

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "artifacts" / "frames").mkdir(parents=True)
    shutil.copyfile(video_path, output / "artifacts" / "source.video")
    shutil.copyfile(checkpoint_path, output / "artifacts" / "checkpoint.bin")
    if measurement_path is not None:
        (output / "artifacts" / "measurements").mkdir()
        shutil.copyfile(measurement_path, output / "artifacts" / "measurements" / "trace.json")

    frame_records, extraction = _extract_frames(
        output / "artifacts" / "source.video",
        output / "artifacts" / "frames",
        source_first_pts_s,
        resolved_frames,
    )

    provenance = {
        "play_run_id": run_id,
        "source_lineage_id": lineage_id,
        "parent_source_lineage_id": parent_lineage_id,
        "video": {
            "logical_id": video_logical,
            "path": "artifacts/source.video",
            "sha256": video_sha,
            "bytes": video_path.stat().st_size,
        },
        "checkpoint": {
            "logical_id": checkpoint_logical,
            "path": "artifacts/checkpoint.bin",
            "sha256": checkpoint_sha,
            "bytes": checkpoint_path.stat().st_size,
        },
        "measurement": None if measurement_path is None else {
            "logical_id": measurement_logical,
            "path": "artifacts/measurements/trace.json",
            "sha256": measurement_sha,
            "bytes": measurement_path.stat().st_size,
        },
        "play": play,
        "frame_extraction": extraction,
        "evidence_independence": {
            "visual_source_count": 1,
            "frame_samples_from_visual_source": FRAME_COUNT,
            "measurement_source_count": 0 if measurement_path is None else 1,
            "model_interpretation_adds_independent_source": False,
        },
    }
    contract = {
        "schema": CONTRACT_SCHEMA,
        "contract_id": digest(
            {
                "run": run_id,
                "lineage": {"id": lineage_id, "parent_id": parent_lineage_id},
                "video": video_sha,
                "checkpoint": checkpoint_sha,
                "frames": [{"timestamp_s": row["timestamp_s"], "sha256": row["sha256"]} for row in frame_records],
                "measurement": measurement_sha,
                "play": play,
                "frame_extraction": extraction,
            }
        )[:24],
        "provenance": provenance,
        "frames": frame_records,
        "synchronized_samples": samples,
        "behavior_ground_truth": None,
        "fixed_evaluation_used_as_behavior_ground_truth": False,
    }
    path = output / "contract.json"
    write_new(path, contract)
    verify(path)
    return path


def _bound_file(root: Path, artifact: Any, expected_path: str, anchor: str) -> Path:
    row = _exact(artifact, {"logical_id", "path", "sha256", "bytes"}, anchor)
    if row["path"] != expected_path:
        raise ValueError(f"{anchor} 상대 경로가 다르다")
    _logical_id(row["logical_id"], f"{anchor}.logical_id")
    expected_sha = _hash(row["sha256"], f"{anchor}.sha256")
    if isinstance(row["bytes"], bool) or not isinstance(row["bytes"], int) or row["bytes"] <= 0:
        raise ValueError(f"{anchor}.bytes가 잘못됐다")
    path = root / expected_path
    if not path.is_file() or path.stat().st_size != row["bytes"] or sha256(path) != expected_sha:
        raise ValueError(f"{anchor} 파일 결합이 다르다")
    return path


def _validate_extraction(value: Any, video_sha: str, timestamps: list[float]) -> dict[str, Any]:
    receipt = _exact(
        value,
        {
            "tool", "tool_version", "source", "source_sha256", "time_basis",
            "source_first_pts_s", "pts_tolerance_s", "commands",
        },
        "frame_extraction",
    )
    first_pts = _number_or_none(receipt["source_first_pts_s"], "frame_extraction.source_first_pts_s")
    if (
        receipt["tool"] != "ffmpeg"
        or not isinstance(receipt["tool_version"], str)
        or not receipt["tool_version"].startswith("ffmpeg version ")
        or receipt["source"] != "artifacts/source.video"
        or receipt["source_sha256"] != video_sha
        or receipt["time_basis"] != "play_relative_from_first_decoded_video_frame_pts"
        or first_pts is None
        or receipt["pts_tolerance_s"] != FRAME_PTS_TOLERANCE_S
    ):
        raise ValueError("frame extraction receipt가 잘못됐다")
    commands = receipt["commands"]
    if not isinstance(commands, list) or len(commands) != FRAME_COUNT:
        raise ValueError("frame extraction command 수가 다르다")
    prior_source_index = -1
    for index, command in enumerate(commands, 1):
        command = _exact(
            command,
            {
                "frame_id", "source_frame_index", "request_timestamp_s", "actual_timestamp_s",
                "source_frame_pts_s", "absolute_error_s", "source", "output",
                "scale_width_ceiling", "jpeg_quality", "argv",
            },
            f"frame_extraction.commands[{index}]",
        )
        pair = (command["scale_width_ceiling"], command["jpeg_quality"])
        source_index = command["source_frame_index"]
        requested = _timestamp(command["request_timestamp_s"], f"frame_extraction.commands[{index}].request_timestamp_s")
        actual = _timestamp(command["actual_timestamp_s"], f"frame_extraction.commands[{index}].actual_timestamp_s")
        source_pts = _number_or_none(
            command["source_frame_pts_s"], f"frame_extraction.commands[{index}].source_frame_pts_s"
        )
        error = _number_or_none(command["absolute_error_s"], f"frame_extraction.commands[{index}].absolute_error_s")
        expected_argv = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "artifacts/source.video",
            "-vf", f"select=eq(n\\,{source_index}),scale='min({command['scale_width_ceiling']},iw)':-2",
            "-vsync", "0", "-frames:v", "1", "-q:v", str(command["jpeg_quality"]), "-y",
            f"artifacts/frames/frame_{index:02d}.jpg",
        ]
        if (
            command["frame_id"] != f"f{index:02d}"
            or isinstance(source_index, bool)
            or not isinstance(source_index, int)
            or source_index <= prior_source_index
            or actual != timestamps[index - 1]
            or source_pts is None
            or round(source_pts - first_pts, 9) != actual
            or error is None
            or error != round(abs(actual - requested), 9)
            or error > FRAME_PTS_TOLERANCE_S
            or command["source"] != "artifacts/source.video"
            or command["output"] != f"artifacts/frames/frame_{index:02d}.jpg"
            or pair not in {(640, 4), (480, 7)}
            or command["argv"] != expected_argv
        ):
            raise ValueError("frame extraction command 결합이 다르다")
        prior_source_index = source_index
    return receipt


def verify(contract_path: Path) -> dict[str, Any]:
    """계약과 모든 로컬 바이트 결합을 다시 검사한다."""
    path = Path(contract_path).resolve()
    contract = _exact(
        read_object(path),
        {
            "schema",
            "contract_id",
            "provenance",
            "frames",
            "synchronized_samples",
            "behavior_ground_truth",
            "fixed_evaluation_used_as_behavior_ground_truth",
        },
        "contract",
    )
    if (
        contract["schema"] != CONTRACT_SCHEMA
        or contract["behavior_ground_truth"] is not None
        or contract["fixed_evaluation_used_as_behavior_ground_truth"] is not False
    ):
        raise ValueError("행동 정답 경계가 잘못됐다")
    provenance = _exact(
        contract["provenance"],
        {
            "play_run_id", "source_lineage_id", "parent_source_lineage_id",
            "video", "checkpoint", "measurement", "play", "frame_extraction", "evidence_independence",
        },
        "provenance",
    )
    run_id = _identifier(provenance["play_run_id"], "provenance.play_run_id")
    lineage_id = _identifier(provenance["source_lineage_id"], "provenance.source_lineage_id")
    parent_lineage_id = provenance["parent_source_lineage_id"]
    if parent_lineage_id is not None:
        parent_lineage_id = _identifier(parent_lineage_id, "provenance.parent_source_lineage_id")
        if parent_lineage_id == lineage_id:
            raise ValueError("부모 계보는 현재 계보와 달라야 한다")
    play = _play(provenance["play"])
    root = path.parent
    _bound_file(root, provenance["video"], "artifacts/source.video", "video")
    _bound_file(root, provenance["checkpoint"], "artifacts/checkpoint.bin", "checkpoint")

    frame_rows = contract["frames"]
    if not isinstance(frame_rows, list) or len(frame_rows) != FRAME_COUNT:
        raise ValueError("계약에 여섯 frame이 필요하다")
    prior = -1.0
    frame_timestamps = []
    for index, row in enumerate(frame_rows, 1):
        row = _exact(row, {"frame_id", "path", "timestamp_s", "sha256", "bytes"}, f"frame[{index}]")
        expected_path = f"artifacts/frames/frame_{index:02d}.jpg"
        if row["frame_id"] != f"f{index:02d}" or row["path"] != expected_path:
            raise ValueError("frame 순서 또는 상대 경로가 다르다")
        timestamp = _timestamp(row["timestamp_s"], f"frame[{index}].timestamp_s")
        if timestamp <= prior:
            raise ValueError("frame 시각은 엄격한 오름차순이어야 한다")
        prior = timestamp
        frame_timestamps.append(timestamp)
        expected_sha = _hash(row["sha256"], f"frame[{index}].sha256")
        if isinstance(row["bytes"], bool) or not isinstance(row["bytes"], int):
            raise TypeError("frame bytes가 잘못됐다")
        frame_path = root / expected_path
        _validate_jpeg(frame_path, f"frame[{index}]")
        if frame_path.stat().st_size != row["bytes"] or sha256(frame_path) != expected_sha:
            raise ValueError("frame 파일 결합이 다르다")

    extraction = _validate_extraction(
        provenance["frame_extraction"],
        provenance["video"]["sha256"],
        frame_timestamps,
    )

    measurement = provenance["measurement"]
    if measurement is None:
        samples = _missing_samples(frame_timestamps)
        measurement_sha = None
    else:
        measurement_path = _bound_file(
            root,
            measurement,
            "artifacts/measurements/trace.json",
            "measurement",
        )
        measurement_sha = measurement["sha256"]
        samples = _measurement(
            read_object(measurement_path),
            run_id=run_id,
            video_sha=provenance["video"]["sha256"],
            checkpoint_sha=provenance["checkpoint"]["sha256"],
            play=play,
            frame_timestamps=frame_timestamps,
        )
    if canonical(contract["synchronized_samples"]) != canonical(samples):
        raise ValueError("동기화 sample 결합이 다르다")
    expected_independence = {
        "visual_source_count": 1,
        "frame_samples_from_visual_source": FRAME_COUNT,
        "measurement_source_count": 0 if measurement is None else 1,
        "model_interpretation_adds_independent_source": False,
    }
    if provenance["evidence_independence"] != expected_independence:
        raise ValueError("근거 독립성 경계가 다르다")
    expected_id = digest(
        {
            "run": run_id,
            "lineage": {"id": lineage_id, "parent_id": parent_lineage_id},
            "video": provenance["video"]["sha256"],
            "checkpoint": provenance["checkpoint"]["sha256"],
            "frames": [{"timestamp_s": row["timestamp_s"], "sha256": row["sha256"]} for row in frame_rows],
            "measurement": measurement_sha,
            "play": play,
            "frame_extraction": extraction,
        }
    )[:24]
    if contract["contract_id"] != expected_id:
        raise ValueError("contract_id 결합이 다르다")
    return contract


def public_model_view(contract_path: Path) -> dict[str, Any]:
    """검증된 계약에서 실제 이미지 바이트를 담은 모델 입력을 만든다."""
    contract = verify(contract_path)
    root = Path(contract_path).resolve().parent
    content: list[dict[str, Any]] = [
        {"type": "text", "text": QUESTION},
        {
            "type": "text",
            "text": json.dumps(
                {
                    "coordinate_frames": {
                        "command": contract["provenance"]["play"]["command_frame"],
                        "motion": contract["provenance"]["play"]["motion_frame"],
                    },
                    "command_name": contract["provenance"]["play"]["command_name"],
                    "samples": contract["synchronized_samples"],
                },
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ),
        },
    ]
    for row in contract["frames"]:
        frame = root / row["path"]
        data_url = "data:image/jpeg;base64," + base64.b64encode(frame.read_bytes()).decode("ascii")
        content.append(
            {"type": "text", "text": json.dumps({"frame_id": row["frame_id"], "timestamp_s": row["timestamp_s"]})}
        )
        content.append({"type": "image_url", "image_url": {"url": data_url, "detail": "low"}})
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}]
    return {
        "schema": MODEL_VIEW_SCHEMA,
        "contract_sha256": sha256(Path(contract_path)),
        "messages": messages,
        "evidence_accounting": {
            "visual_source_count": 1,
            "frame_sample_count": FRAME_COUNT,
            "measurement_source_count": contract["provenance"]["evidence_independence"]["measurement_source_count"],
            "model_interpretation_source_count": 0,
        },
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--spec", type=Path, required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
    verify_parser = commands.add_parser("verify")
    verify_parser.add_argument("--contract", type=Path, required=True)
    view_parser = commands.add_parser("model-view")
    view_parser.add_argument("--contract", type=Path, required=True)
    view_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        contract_path = prepare(args.spec, args.output)
        result = {"outcome": "prepared", "contract": str(contract_path), "sha256": sha256(contract_path)}
    elif args.command == "verify":
        contract = verify(args.contract)
        result = {"outcome": "valid", "contract_id": contract["contract_id"], "sha256": sha256(args.contract)}
    else:
        view = public_model_view(args.contract)
        write_new(args.output, view)
        result = {"outcome": "written", "model_view": str(args.output), "sha256": sha256(args.output)}
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
