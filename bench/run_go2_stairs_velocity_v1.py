"""공식 Go2 계단 재생을 0.25/0.75m/s 동일 조건 비교로 좁힌다."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Any

BASE_RUNNER_SHA256_LF = "6858d73783ce02fecee6f901033660fd115c17f139d2937e07728fc25c075b5a"
INTEGRITY_SOURCE_SHA256_LF = "ce5b675850791aed4b9b90eced16f31a9782b4b8f7f0a227199582b35b66f8a8"
CHECKPOINT_SHA256 = "f2aa77bf0349c10ec2a00071162e37123d0cf80f3447c3989c17a618e1b24ad2"
ALLOWED_COMMANDS_M_S = (0.25, 0.75)
STEP_HEIGHT_M = 0.05
STEP_WIDTH_M = 0.30
SEED = 200
FRAMES = 625
FPS = 50
TECHNICAL_RECEIPT_KEYS = {"child_exit_code", "shutdown_verified", "exact_encoder_verified"}


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _sha_lf(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("JSON object가 필요하다")
    return value


def find_repo(start: Path) -> Path:
    for candidate in (Path(start), *Path(start).parents):
        runner = candidate / "bench/run_stairs_playback_v2.py"
        integrity = candidate / "evals/stairs_motion_integrity_v1.py"
        if (runner.is_file() and integrity.is_file()
                and _sha_lf(runner) == BASE_RUNNER_SHA256_LF
                and _sha_lf(integrity) == INTEGRITY_SOURCE_SHA256_LF):
            return candidate
    raise FileNotFoundError("동결된 계단 runner와 upright audit를 찾지 못했다")


def _load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path.resolve(strict=True))
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_base(repo: Path) -> ModuleType:
    path = Path(repo) / "bench/run_stairs_playback_v2.py"
    if _sha_lf(path) != BASE_RUNNER_SHA256_LF:
        raise ValueError("공개 stairs runner LF SHA가 다르다")
    return _load("go2_velocity_public_stairs_v2", path)


def load_integrity(repo: Path) -> ModuleType:
    path = Path(repo) / "evals/stairs_motion_integrity_v1.py"
    if _sha_lf(path) != INTEGRITY_SOURCE_SHA256_LF:
        raise ValueError("upright audit LF SHA가 다르다")
    return _load("go2_velocity_stairs_integrity_v1", path)


def validate_spec(path: Path, base: ModuleType) -> dict[str, Any]:
    raw = _read(path)
    capture = raw.get("capture")
    command = capture.get("command_x_m_s") if isinstance(capture, dict) else None
    if command not in ALLOWED_COMMANDS_M_S:
        raise ValueError("명령 속도는 0.25 또는 0.75m/s여야 한다")
    base.__dict__["STEP_HEIGHT_M"] = STEP_HEIGHT_M
    base.__dict__["STEP_WIDTH_M"] = STEP_WIDTH_M
    base.__dict__["COMMAND_X_M_S"] = command
    value = base._read_spec(Path(path))
    if (value.get("robot_role") != "quadruped"
            or value.get("seed") != SEED
            or value.get("checkpoint", {}).get("sha256") != CHECKPOINT_SHA256):
        raise ValueError("Go2 seed/checkpoint 등록 조건이 다르다")
    if value.get("direction") not in {"up", "down"}:
        raise ValueError("계단 방향은 up/down이어야 한다")
    prefix = f"quadruped-stairs-{value['direction']}-step-0p05"
    suffix = value.get("clip_id", "")[len(prefix):] if isinstance(value.get("clip_id"), str) else None
    allowed_suffixes = {0.25: {"", "-cmd-0p25"}, 0.75: {"-cmd-0p75"}}
    if (not isinstance(value.get("clip_id"), str) or not value["clip_id"].startswith(prefix)
            or not isinstance(suffix, str) or suffix not in allowed_suffixes[command]):
        raise ValueError("clip id가 방향·단차·명령 조건을 표시하지 않는다")
    return value


def prepare_base(spec_path: Path, *, repo: Path | None = None) -> tuple[ModuleType, dict[str, Any]]:
    actual_repo = Path(repo).resolve() if repo is not None else find_repo(Path(__file__).resolve())
    base = load_base(actual_repo)
    return base, validate_spec(Path(spec_path), base)


def execute(spec_path: Path, out: Path) -> dict[str, Any]:
    base, _value = prepare_base(spec_path)
    result = base.execute(Path(spec_path).resolve(strict=True), Path(out).resolve(strict=False))
    if not isinstance(result, dict):
        raise TypeError("공개 runner 결과가 object가 아니다")
    return result


def _technical_receipt(value: dict[str, Any]) -> dict[str, Any]:
    if (not isinstance(value, dict) or set(value) != TECHNICAL_RECEIPT_KEYS
            or not isinstance(value.get("child_exit_code"), int)
            or isinstance(value.get("child_exit_code"), bool)
            or type(value.get("shutdown_verified")) is not bool
            or type(value.get("exact_encoder_verified")) is not bool):
        raise ValueError("부모 technical receipt 형식이 다르다")
    return value


def _relative_file(root: Path, relative: Any, label: str) -> Path:
    pure = PurePosixPath(relative) if isinstance(relative, str) else None
    if pure is None or pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"{label} 상대 경로가 다르다")
    path = (root / pure.as_posix()).resolve(strict=True)
    if root.resolve() not in path.parents:
        raise ValueError(f"{label}가 record 밖이다")
    return path


def _verify_encoder(out: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    video = manifest.get("video")
    if not isinstance(video, dict):
        raise TypeError("video manifest가 없다")
    video_path = _relative_file(out, video.get("path"), "video")
    if _sha(video_path) != video.get("sha256"):
        raise ValueError("video SHA가 다르다")
    receipt_path = video_path.with_suffix(video_path.suffix + ".exact-frames.json")
    receipt = _read(receipt_path)
    rows = receipt.get("frame_sequence")
    if (receipt.get("schema") != "local_stairs_exact_frame_encoder_v1"
            or receipt.get("frames_received") != FRAMES
            or receipt.get("frames_written") != FRAMES
            or receipt.get("fps") != FPS
            or receipt.get("video_file") != video_path.name
            or receipt.get("video_sha256") != video["sha256"]
            or not isinstance(rows, list) or len(rows) != FRAMES
            or [row.get("index") for row in rows] != list(range(FRAMES))):
        raise ValueError("625-frame exact encoder receipt가 다르다")
    return {"video_sha256": video["sha256"], "encoder_receipt_sha256": _sha(receipt_path)}


def _incomplete_record(spec: dict[str, Any], out: Path, technical: dict[str, Any]) -> dict[str, Any]:
    failure_path = out / "failure.json"
    if not failure_path.is_file() or technical["child_exit_code"] == 0:
        raise ValueError("완전한 manifest도 검증 가능한 실패 증거도 없다")
    failure = _read(failure_path)
    partial = failure.get("partial_trace")
    trace_rows = 0
    trace_sha256 = None
    if partial is not None:
        if not isinstance(partial, dict):
            raise ValueError("partial trace 형식이 다르다")
        trace = _relative_file(out, partial.get("path"), "partial trace")
        trace_rows = trace.read_bytes().count(b"\n")
        trace_sha256 = _sha(trace)
        if partial.get("rows") != trace_rows or partial.get("sha256") != trace_sha256:
            raise ValueError("partial trace 행/SHA가 다르다")
    return {
        "schema": "go2_stairs_velocity_verification_v1",
        "status": "technical_incomplete",
        "clip_id": spec["clip_id"],
        "command_x_m_s": spec["capture"]["command_x_m_s"],
        "technical_complete": False,
        "motion_accepted": False,
        "motion_failure_preserved": True,
        "partial_trace_rows": trace_rows,
        "partial_trace_sha256": trace_sha256,
        "failure_sha256": _sha(failure_path),
        "body_up_world_z": None,
        "claims": _claims(),
    }


def _claims() -> dict[str, bool]:
    return {
        "development_seen_source": True,
        "heldout": False,
        "new_training": False,
        "model_api": False,
        "recovery_evidence": False,
        "fall_cause_confirmed": False,
        "speed_sweep": False,
    }


def verify_record(spec_path: Path, out: Path, *, technical_receipt: dict[str, Any],
                  repo: Path | None = None) -> dict[str, Any]:
    actual_out = Path(out).resolve(strict=True)
    actual_repo = Path(repo).resolve() if repo is not None else find_repo(Path(__file__).resolve())
    base, spec = prepare_base(spec_path, repo=actual_repo)
    technical = _technical_receipt(technical_receipt)
    manifest_path = actual_out / "manifest.json"
    if not manifest_path.is_file():
        return _incomplete_record(spec, actual_out, technical)
    if technical != {"child_exit_code": 0, "shutdown_verified": True, "exact_encoder_verified": True}:
        raise ValueError("완전한 record의 wrapper/encoder 종료 검증이 없다")
    manifest = _read(manifest_path)
    if (manifest.get("schema") != base.MANIFEST_SCHEMA
            or manifest.get("clip_id") != spec["clip_id"]
            or manifest.get("robot_role") != "quadruped"
            or manifest.get("direction") != spec["direction"]
            or manifest.get("checkpoint_sha256") != CHECKPOINT_SHA256
            or manifest.get("source") != spec["source"]):
        raise ValueError("manifest spec/source/checkpoint 결합이 다르다")
    terrain = manifest.get("terrain")
    command = spec["capture"]["command_x_m_s"]
    if (not isinstance(terrain, dict)
            or terrain.get("step_height_m") != STEP_HEIGHT_M
            or terrain.get("step_width_m") != STEP_WIDTH_M
            or terrain.get("terrain_seed") != SEED
            or terrain.get("command", {}).get("lin_vel_x") != command):
        raise ValueError("manifest terrain/command 조건이 다르다")
    encoder = _verify_encoder(actual_out, manifest)
    trace_ref = manifest.get("state_trace")
    if not isinstance(trace_ref, dict):
        raise TypeError("state trace ref가 없다")
    trace_path = _relative_file(actual_out, trace_ref.get("path"), "state trace")
    trace_sha = _sha(trace_path)
    if trace_ref.get("sha256") != trace_sha:
        raise ValueError("state trace SHA가 다르다")
    assessment = manifest.get("assessment")
    if not isinstance(assessment, dict):
        raise TypeError("motion assessment가 없다")
    if assessment.get("termination_seen") is True or assessment.get("outcome") == "fall":
        if (assessment.get("termination_seen") is not True
                or assessment.get("outcome") != "fall"
                or assessment.get("trace_rows") != FRAMES + 1
                or trace_path.read_bytes().count(b"\n") != FRAMES + 1):
            raise ValueError("termination motion failure 증거가 서로 다르다")
        return {
            "schema": "go2_stairs_velocity_verification_v1",
            "status": "complete",
            "clip_id": spec["clip_id"],
            "direction": spec["direction"],
            "command_x_m_s": command,
            "technical_complete": True,
            "motion_accepted": False,
            "motion_failure_preserved": True,
            "assessment": assessment,
            "body_up_world_z": None,
            "integrity_result": "not_evaluated_registered_termination",
            "manifest_sha256": _sha(manifest_path),
            "trace_sha256": trace_sha,
            **encoder,
            "claims": _claims(),
        }
    integrity = load_integrity(actual_repo).audit(
        manifest_path, trace_path,
        expected_manifest_sha256=_sha(manifest_path),
        expected_trace_sha256=trace_sha,
    )
    upright = integrity.get("body_up_world_z")
    if not isinstance(upright, dict) or not isinstance(upright.get("negative_count"), int):
        raise TypeError("upright audit 결과가 없다")
    accepted = (
        assessment.get("outcome") == "completed"
        and assessment.get("progress_condition_pass") is True
        and assessment.get("height_condition_pass") is True
        and assessment.get("termination_seen") is False
        and upright["negative_count"] == 0
    )
    return {
        "schema": "go2_stairs_velocity_verification_v1",
        "status": "complete",
        "clip_id": spec["clip_id"],
        "direction": spec["direction"],
        "command_x_m_s": command,
        "technical_complete": True,
        "motion_accepted": accepted,
        "motion_failure_preserved": not accepted,
        "assessment": assessment,
        "body_up_world_z": upright,
        "integrity_result": integrity.get("integrity_result"),
        "manifest_sha256": _sha(manifest_path),
        "trace_sha256": trace_sha,
        **encoder,
        "claims": _claims(),
    }


def _prepared(spec_path: Path, repo: Path | None) -> dict[str, Any]:
    _base, spec = prepare_base(spec_path, repo=repo)
    return {
        "status": "ready",
        "clip_id": spec["clip_id"],
        "direction": spec["direction"],
        "command_x_m_s": spec["capture"]["command_x_m_s"],
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "frames": FRAMES,
        "fps": FPS,
        "automatic_retry": False,
        "parameter_sweep": False,
        "claims": _claims(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--spec", required=True, type=Path)
    prepare.add_argument("--repo", type=Path)
    record = sub.add_parser("record")
    record.add_argument("--spec", required=True, type=Path)
    record.add_argument("--out", required=True, type=Path)
    verify = sub.add_parser("verify")
    verify.add_argument("--spec", required=True, type=Path)
    verify.add_argument("--out", required=True, type=Path)
    verify.add_argument("--technical-receipt", required=True, type=Path)
    verify.add_argument("--repo", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        result = _prepared(args.spec, args.repo)
    elif args.command == "record":
        result = execute(args.spec, args.out)
    else:
        result = verify_record(args.spec, args.out,
                               technical_receipt=_read(args.technical_receipt), repo=args.repo)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
