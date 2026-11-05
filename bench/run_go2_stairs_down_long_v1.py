"""같은 Go2 계단 하행을 자동 리셋 없이 30초 관찰한다. 정책 교정 실험이 아니다."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from types import ModuleType
from typing import Any

BASE_SHA = "6858d73783ce02fecee6f901033660fd115c17f139d2937e07728fc25c075b5a"
CHECKPOINT_SHA = "f2aa77bf0349c10ec2a00071162e37123d0cf80f3447c3989c17a618e1b24ad2"
FRAMES = 1500
FPS = 50
SPEC_SCHEMA = "go2_stairs_down_long_spec_v1"
MANIFEST_SCHEMA = "go2_stairs_down_long_manifest_v1"
CLIP_ID = "quadruped-stairs-down-step-0p05-cmd-0p25-30s"


def sha(path: Path, *, lf: bool = False) -> str:
    raw = path.read_bytes()
    return hashlib.sha256(raw.replace(b"\r\n", b"\n") if lf else raw).hexdigest()


def read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("JSON object가 필요하다")
    return value


def make_spec(previous: dict) -> dict:
    value = copy.deepcopy(previous)
    if (value.get("robot_role") != "quadruped" or value.get("direction") != "down"
            or value.get("seed") != 200 or value.get("checkpoint", {}).get("sha256") != CHECKPOINT_SHA
            or value.get("capture", {}).get("command_x_m_s") != 0.25
            or value.get("capture", {}).get("step_height_m") != 0.05
            or value.get("capture", {}).get("step_width_m") != 0.3):
        raise ValueError("이전 하행 조건이 등록값과 다르다")
    value["schema"] = SPEC_SCHEMA
    value["clip_id"] = CLIP_ID
    value["capture"]["frames"] = FRAMES
    return value


def body_up_z(quaternion: list[float]) -> float:
    if (len(quaternion) != 4 or any(not math.isfinite(x) for x in quaternion)
            or abs(sum(x * x for x in quaternion) - 1) > 1e-3):
        raise ValueError("유한한 단위 wxyz quaternion이 필요하다")
    w, x, y, z = quaternion
    return (w * w - x * x - y * y + z * z) / sum(v * v for v in quaternion)


def configure_long(original: Any, cfg: Any, spec: dict) -> dict:
    result = original(cfg, spec)
    terms = cfg.terminations.to_dict() if hasattr(cfg.terminations, "to_dict") else vars(cfg.terminations)
    if any(value is not None and name not in {"base_contact", "time_out"} for name, value in terms.items()):
        raise ValueError("예상하지 못한 종료 조건이 있다")
    cfg.episode_length_s = FRAMES / FPS + 1
    cfg.terminations.base_contact = None
    cfg.terminations.time_out = None
    return {**result, "observation_extension": {
        "seconds": FRAMES / FPS, "automatic_reset_disabled": True,
        "disabled_termination_terms": ["base_contact", "time_out"],
        "episode_length_s": cfg.episode_length_s,
        "purpose": "넘어짐 뒤 물리 동작을 관찰하며 기립·회복으로 판정하지 않는다",
        "command_resampling_time_s": cfg.commands.base_velocity.resampling_time_range[0],
    }}


def load_base(repo: Path | None = None) -> ModuleType:
    root = repo or Path(__file__).resolve().parents[1]
    path = root / "bench/run_stairs_playback_v2.py"
    if sha(path, lf=True) != BASE_SHA:
        raise ValueError("기존 계단 runner가 바뀌었다")
    module_spec = importlib.util.spec_from_file_location("long_stairs_base", path)
    if module_spec is None or module_spec.loader is None:
        raise ImportError(path)
    base = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(base)
    base.__dict__.update(FRAMES=FRAMES, FPS=FPS, STEP_HEIGHT_M=0.05, STEP_WIDTH_M=0.3,
                         COMMAND_X_M_S=0.25, SCHEMA=SPEC_SCHEMA, MANIFEST_SCHEMA=MANIFEST_SCHEMA)
    configure, tensor_row, parse = base.configure_env, base._tensor_row, base._read_spec

    def checked_spec(path: Path) -> dict:
        value = parse(path)
        if (value["robot_role"] != "quadruped" or value["direction"] != "down"
                or value["seed"] != 200 or value["clip_id"] != CLIP_ID
                or value["checkpoint"]["sha256"] != CHECKPOINT_SHA):
            raise ValueError("30초 Go2 하행 조건이 다르다")
        return value

    def trace_row(env: Any, frame: int, terminated: bool) -> dict:
        row = tensor_row(env, frame, terminated)
        row["body_up_world_z"] = body_up_z(row["root_quat_w"])
        row["episode_length_step"] = int(env.episode_length_buf[0].item())
        return row

    base.__dict__.update(configure_env=lambda cfg, spec: configure_long(configure, cfg, spec),
                         _tensor_row=trace_row, _read_spec=checked_spec)
    return base


def verify_record(out: Path, spec: dict) -> dict:
    manifest = read(out / "manifest.json")
    if (manifest.get("schema") != MANIFEST_SCHEMA or manifest.get("clip_id") != CLIP_ID
            or manifest.get("checkpoint_sha256") != CHECKPOINT_SHA
            or manifest.get("source") != spec["source"]):
        raise ValueError("manifest/source/checkpoint가 다르다")
    trace = out / "state-trace.jsonl"
    if sha(trace) != manifest["state_trace"]["sha256"]:
        raise ValueError("trace SHA가 다르다")
    rows = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    if len(rows) != FRAMES + 1:
        raise ValueError("30초 trace 1501행이 필요하다")
    for i, row in enumerate(rows):
        if (row.get("frame") != i or row.get("episode_length_step") != i
                or row.get("sim_time_s") != i / FPS or row.get("terminated") is not False
                or abs(body_up_z(row["root_quat_w"]) - row["body_up_world_z"]) > 1e-10):
            raise ValueError("연속 관찰의 프레임·시간·리셋·자세 계약이 다르다")
    extension = manifest["terrain"].get("observation_extension", {})
    if extension.get("automatic_reset_disabled") is not True or extension.get("seconds") != 30:
        raise ValueError("리셋 없는 30초 관찰 기록이 아니다")
    if manifest["render_sync"].get("physics_state_unchanged") is not True:
        raise ValueError("렌더 비간섭 증거가 없다")
    video_ref = Path(manifest["video"]["path"])
    if video_ref.is_absolute() or ".." in video_ref.parts:
        raise ValueError("영상 경로가 record 밖이다")
    video = out / video_ref
    if sha(video) != manifest["video"]["sha256"]:
        raise ValueError("영상 SHA가 다르다")
    encoder = read(video.with_suffix(video.suffix + ".exact-frames.json"))
    sequence = encoder.get("frame_sequence", [])
    if (encoder.get("frames_received") != FRAMES or encoder.get("frames_written") != FRAMES
            or encoder.get("fps") != FPS or len(sequence) != FRAMES
            or [r.get("index") for r in sequence] != list(range(FRAMES))
            or encoder.get("video_sha256") != sha(video)):
        raise ValueError("원본 1500프레임 인코딩 증거가 다르다")
    negatives = [row["sim_time_s"] for row in rows if row["body_up_world_z"] < 0]
    return {
        "schema": "go2_stairs_down_long_observation_v1", "status": "complete",
        "frames": FRAMES, "fps": FPS, "duration_s": FRAMES / FPS,
        "trace_rows": len(rows), "reset_count": 0,
        "root_dx_m": rows[-1]["root_pos_w"][0] - rows[0]["root_pos_w"][0],
        "root_dz_m": rows[-1]["root_pos_w"][2] - rows[0]["root_pos_w"][2],
        "first_inverted_time_s": negatives[0] if negatives else None,
        "inverted_samples": len(negatives),
        "last_body_up_world_z": rows[-1]["body_up_world_z"],
        "minimum_body_up_world_z": min(row["body_up_world_z"] for row in rows),
        "video_sha256": sha(video), "manifest_sha256": sha(out / "manifest.json"),
        "trace_sha256": sha(trace), "encoder_sha256": sha(video.with_suffix(video.suffix + ".exact-frames.json")),
        "new_training": False, "model_api": False, "agent_recovery_evidence": False,
        "observation_only_no_reset": True,
    }


def compare_prefix(old_trace: Path, new_trace: Path) -> dict:
    old = [json.loads(line) for line in old_trace.read_text(encoding="utf-8").splitlines()]
    new = [json.loads(line) for line in new_trace.read_text(encoding="utf-8").splitlines()]
    if len(old) != 626 or len(new) < 626:
        raise ValueError("이전 12.5초 626행이 필요하다")
    mismatch = next((i for i, row in enumerate(old) if row != {key: new[i].get(key) for key in row}), None)
    return {"rows_compared": 626, "original_fields_exact_match": mismatch is None,
            "first_different_frame": mismatch, "old_trace_sha256": sha(old_trace),
            "new_trace_sha256": sha(new_trace), "same_prefix_not_assumed": True}
