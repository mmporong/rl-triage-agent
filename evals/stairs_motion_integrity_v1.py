"""계단 재생 trace의 몸체 위쪽 방향과 기존 완료 판정을 사후 감사한다."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path, PurePosixPath
from typing import Any

HEX64 = re.compile(r"[0-9a-f]{64}\Z")


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("JSON 중복 키")
        value[key] = item
    return value


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_unique,
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           ValueError(f"비유한 JSON 값: {token}")))
    if not isinstance(value, dict):
        raise TypeError("JSON object가 필요하다")
    return value


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label}은 유한 수여야 한다")
    return float(value)


def _finite_tree(value: Any, label: str) -> None:
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        _finite_number(value, label)
        return
    if isinstance(value, list):
        for item in value:
            _finite_tree(item, label)
        return
    raise ValueError(f"{label}에는 유한 수·bool·배열만 허용된다")


def _trace_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"trace 빈 행: {number}")
            value = json.loads(line, object_pairs_hook=_unique,
                               parse_constant=lambda token: (_ for _ in ()).throw(
                                   ValueError(f"비유한 JSON 값: {token}")))
            if not isinstance(value, dict):
                raise TypeError("trace 행은 object여야 한다")
            rows.append(value)
    if len(rows) != 626:
        raise ValueError("trace는 frame 0..625의 626행이어야 한다")
    return rows


def _summary(values: list[float], times: list[float]) -> dict[str, float | int]:
    return {"samples": len(values), "start_time_s": times[0], "end_time_s": times[-1],
            "min": min(values), "max": max(values), "mean": sum(values) / len(values),
            "first": values[0], "last": values[-1],
            "negative_count": sum(value < 0 for value in values)}


def _vector(value: Any, length: int, label: str) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{label} 차원이 다르다")
    return [_finite_number(item, label) for item in value]


def audit(manifest_path: Path, trace_path: Path, *, expected_manifest_sha256: str,
          expected_trace_sha256: str) -> dict[str, Any]:
    """원문 SHA와 626행 trace를 검증하고 body-up world-z 관측을 반환한다."""
    manifest_file = Path(manifest_path).resolve(strict=True)
    trace_file = Path(trace_path).resolve(strict=True)
    if (not HEX64.fullmatch(expected_manifest_sha256)
            or not HEX64.fullmatch(expected_trace_sha256)
            or _sha(manifest_file) != expected_manifest_sha256
            or _sha(trace_file) != expected_trace_sha256):
        raise ValueError("manifest/trace 기대 SHA가 다르다")
    manifest = _read(manifest_file)
    state = manifest.get("state_trace")
    if manifest.get("schema") != "stairs_playback_manifest_v2" or not isinstance(state, dict):
        raise ValueError("계단 재생 manifest가 아니다")
    relative = state.get("path")
    pure = PurePosixPath(relative) if isinstance(relative, str) else None
    if (pure is None or pure.is_absolute() or ".." in pure.parts
            or trace_file != (manifest_file.parent / pure.as_posix()).resolve()
            or state.get("sha256") != expected_trace_sha256):
        raise ValueError("manifest state_trace 경로/SHA 결합이 다르다")
    checkpoint = manifest.get("checkpoint_sha256")
    assessment = manifest.get("assessment")
    if not isinstance(checkpoint, str) or not HEX64.fullmatch(checkpoint):
        raise ValueError("checkpoint SHA가 없다")
    if not isinstance(assessment, dict) or assessment.get("outcome") not in {"completed", "failed"}:
        raise ValueError("기존 계단 판정이 없다")
    if assessment.get("trace_rows") != 626:
        raise ValueError("manifest trace_rows가 다르다")
    _finite_number(assessment.get("root_dx_m"), "assessment root_dx_m")
    _finite_number(assessment.get("root_dz_m"), "assessment root_dz_m")
    if any(type(assessment.get(key)) is not bool for key in (
            "height_condition_pass", "progress_condition_pass", "termination_seen")):
        raise ValueError("assessment bool 판정이 없다")
    rows = _trace_rows(trace_file)
    up_values: list[float] = []
    norms: list[float] = []
    times: list[float] = []
    terminated: list[bool] = []
    for expected_frame, row in enumerate(rows):
        if set(row) != {"frame", "sim_time_s", "root_pos_w", "root_quat_w",
                        "root_lin_vel_w", "terrain_origin_w", "terminated"}:
            raise ValueError("trace 행 필드가 다르다")
        _finite_tree(list(row.values()), "trace row")
        if row.get("frame") != expected_frame:
            raise ValueError("trace frame 순서가 0..625가 아니다")
        time_s = _finite_number(row.get("sim_time_s"), "sim_time_s")
        expected_time = expected_frame / 50
        if not math.isclose(time_s, expected_time, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("trace sim_time은 50 fps의 0..12.5초여야 한다")
        _vector(row.get("root_pos_w"), 3, "root_pos_w")
        _vector(row.get("root_lin_vel_w"), 3, "root_lin_vel_w")
        terrain = row.get("terrain_origin_w")
        if not isinstance(terrain, list) or len(terrain) != 1:
            raise ValueError("terrain_origin_w 차원이 다르다")
        _vector(terrain[0], 3, "terrain_origin_w")
        w, x, y, z = _vector(row.get("root_quat_w"), 4, "root_quat_w")
        norm_sq = w * w + x * x + y * y + z * z
        if norm_sq <= 0:
            raise ValueError("root quaternion norm은 0보다 커야 한다")
        norm = math.sqrt(norm_sq)
        up_z = 1.0 - 2.0 * (x * x + y * y) / norm_sq
        if not math.isfinite(up_z):
            raise ValueError("body-up world-z를 계산할 수 없다")
        if type(row.get("terminated")) is not bool:
            raise ValueError("terminated는 bool이어야 한다")
        up_values.append(up_z)
        norms.append(norm)
        times.append(time_s)
        terminated.append(row["terminated"])
    termination_seen = any(terminated)
    if (assessment.get("termination_seen") is not termination_seen
            or manifest.get("record_video_contract", {}).get("video_length") != 625
            or manifest.get("record_video_contract", {}).get("executed_steps") != 625):
        raise ValueError("termination/video/trace 계약이 다르다")
    negative_indices = [index for index, value in enumerate(up_values) if value < 0]
    original = assessment["outcome"]
    if original == "failed":
        result = "original_failure_preserved"
    elif negative_indices:
        result = "upright_completion_contradicted"
    else:
        result = "no_inversion_observed_not_full_stability_proof"
    end_values, end_times = up_values[-13:], times[-13:]
    return {"schema": "stairs_motion_integrity_audit_v1", "status": "complete",
            "clip_id": manifest.get("clip_id"), "robot_role": manifest.get("robot_role"),
            "direction": manifest.get("direction"), "checkpoint_sha256": checkpoint,
            "manifest_sha256": expected_manifest_sha256, "trace_sha256": expected_trace_sha256,
            "trace": {"frames": 626, "frame_first": 0, "frame_last": 625,
                      "sim_time_first_s": 0.0, "sim_time_last_s": 12.5, "fps": 50},
            "quaternion": {"order": "wxyz", "raw_norm_min": min(norms),
                           "raw_norm_max": max(norms), "normalization_applied": True},
            "body_up_world_z": {"min": min(up_values), "last": up_values[-1],
                                "negative_count": len(negative_indices),
                                "first_negative_time_s": (times[negative_indices[0]]
                                                          if negative_indices else None),
                                "end_window_last_13_samples_0p24s": _summary(end_values, end_times)},
            "termination_seen": termination_seen, "original_assessment": {
                "outcome": original, "root_dx_m": assessment.get("root_dx_m"),
                "root_dz_m": assessment.get("root_dz_m"),
                "height_condition_pass": assessment.get("height_condition_pass"),
                "progress_condition_pass": assessment.get("progress_condition_pass")},
            "integrity_result": result,
            "interpretation": {"negative_up_z_observation": "body_up_world_z_negative_observed",
                               "fall_cause_confirmed": False,
                               "termination_false_means_healthy_walking": False,
                               "post_exposure_development_audit": True,
                               "improvement_measured": False,
                               "model_discovered": False}}


def _write_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    spec = _read(args.spec)
    if spec.get("schema") != "stairs_motion_integrity_batch_spec_v1" \
            or not isinstance(spec.get("items"), list):
        raise ValueError("audit batch spec가 다르다")
    output = args.out.resolve()
    if output.exists():
        raise FileExistsError("audit output은 fresh 경로여야 한다")
    rows = [audit(Path(item["manifest"]), Path(item["trace"]),
                  expected_manifest_sha256=item["manifest_sha256"],
                  expected_trace_sha256=item["trace_sha256"]) for item in spec["items"]]
    output.mkdir(parents=True)
    _write_new(output / "audit.json", {"schema": "stairs_motion_integrity_batch_v1",
                                        "status": "complete", "rows": rows,
                                        "claims": {"cause_confirmed": False,
                                                   "improvement_measured": False,
                                                   "model_discovered": False}})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
