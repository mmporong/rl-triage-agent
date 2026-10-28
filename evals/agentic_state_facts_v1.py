"""same-Play 상태 sidecar와 native trace를 결합해 영상 모델 주장 원장용 관측 수치(fact)를 만든다.

- 결합: 기존 ``record_agentic_state_sidecar_v1.load_sidecar``·``join_with_trace``와
  ``record_vision_evidence_v1.bind_actual_pts``가 확인한 checkpoint·frame·step·시각 일치만 받는다.
- window: ``record_vision_evidence_v1.WINDOWS``의 선언 sample 시각(초)과 실제 video PTS가 정확히 같은
  frame만 쓴다(V01 measurement와 같은 선택). 새 구간·허용 오차·seed를 만들지 않는다.
- 좌표: trace의 command·motion은 몸체 좌표(body)다. 월드 좌표(root_position_world_m)와 섞지 않는다.
  명령 오차는 몸체 xy 벡터 차의 norm이며 두 norm의 차가 아니다.
- 관절·토크: sidecar에 기록된 값만 요약한다. 없는 필드는 unavailable로 전파하고 0으로 채우지 않는다.
  computed·applied torque를 추정하거나 보정하지 않는다.
- 판정: 전진·가속·넘어짐·보행·원인·회복 라벨과 임계값을 만들지 않는다. 0 분모는 null과 이유로 남긴다.
- 공개 결과에는 고장 ID·논리 checkpoint ID·job ID를 넣지 않는다. 결합 식별자는 별도 metadata에 둔다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _folder in (ROOT / "bench", ROOT / "evals", ROOT / "src"):
    if str(_folder) not in sys.path:
        sys.path.insert(0, str(_folder))

import record_agentic_state_sidecar_v1 as SIDECAR
import record_vision_evidence_v1 as R

SCHEMA = "agentic_state_facts_v1"
METADATA_SCHEMA = "agentic_state_facts_metadata_v1"
PUBLIC_NAME = "state_facts.json"
METADATA_NAME = "state_facts_metadata.json"
# trace state command·motion 성분 순서(record_vision_evidence_v1.build_trace_and_v01_specs와 같음)
VECTOR_COMPONENTS = ("linear_x_mps", "linear_y_mps", "yaw_rate_rps")
JOINT_FIELDS = {"joint_vel": "rad/s", "computed_torque": "N*m", "applied_torque": "N*m"}
UNSUPPORTED_CLAIMS = (
    ("gait_or_walking_state", "접촉 값은 요약하지 않고 lazy 갱신 frame은 비어 있어 보행 여부를 단정할 근거가 아니다"),
    ("forward_progress_judgement", "전진 판정 임계값을 만들지 않는다"),
    ("acceleration_judgement", "가속 판정 임계값을 만들지 않는다"),
    ("fall_judgement", "넘어짐 판정 임계값을 만들지 않는다"),
    ("fault_cause_or_location", "관측 수치는 원인·위치를 판정하지 않는다"),
    ("single_leg_fault_localization", "등록 모터 변경은 12관절 전체 함수라 한쪽 다리로 해석하지 않는다"),
    ("recovery_or_behavior_label", "행동 정답·회복 라벨을 만들지 않는다"),
    ("world_frame_displacement", "몸체 좌표 속도와 월드 좌표 위치를 섞지 않으며 이 fact는 월드 이동량을 다루지 않는다"),
    ("torque_estimation", "기록된 computed·applied torque만 요약하고 추정값을 만들지 않는다"),
)


class FactsError(ValueError):
    """결합·형식·원문 계약 위반. 부분 결과를 만들지 않는다."""


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _finite_vector(value: Any, length: int, anchor: str) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise FactsError(f"{anchor} 길이가 {length}이 아니다")
    result = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item):
            raise FactsError(f"{anchor}[{index}]가 유한 수가 아니다")
        result.append(float(item))
    return result


def _stats(values: Sequence[float], anchor: str) -> dict[str, float]:
    if not values:
        raise FactsError(f"{anchor} sample이 없다")
    mean = math.fsum(values) / len(values)
    result = {"mean": mean, "min": min(values), "max": max(values)}
    if not all(math.isfinite(item) for item in result.values()):
        raise FactsError(f"{anchor} 통계가 유한 수가 아니다")
    return result


def _check_trace(trace: Mapping[str, Any]) -> list[dict[str, Any]]:
    profile = R.fixed_profile()
    if (trace.get("schema") != R.TRACE_SCHEMA or trace.get("method") != R.METHOD
            or trace.get("profile") != profile or trace.get("profile_sha256") != R.digest(profile)
            or trace.get("time_basis") != profile["time_basis"] or trace.get("frame_count") != R.FRAME_COUNT):
        raise FactsError("native trace가 고정 same-Play 625 frame·50 fps 계약과 다르다")
    if profile["command"]["frame"] != "body" or profile["motion_frame"] != "body":
        raise FactsError("command·motion 좌표가 몸체 좌표가 아니다")
    records = trace.get("records")
    if not isinstance(records, list) or len(records) != R.FRAME_COUNT:
        raise FactsError("trace record 수가 625가 아니다")
    pts = [row.get("actual_timestamp_s") if isinstance(row, Mapping) else None for row in records]
    receipt = trace.get("timing_receipt")
    if (not isinstance(receipt, Mapping) or receipt.get("source_sha256") != trace.get("video_sha256")
            or receipt.get("frame_count") != R.FRAME_COUNT or receipt.get("relative_actual_pts_s") != pts):
        raise FactsError("trace timing receipt가 video·frame PTS와 다르다")
    stripped = [{key: value for key, value in row.items() if key != "actual_timestamp_s"} for row in records]
    try:
        rebound = R.bind_actual_pts(stripped, [_finite_vector([value], 1, "actual_timestamp_s")[0] for value in pts])
    except (KeyError, TypeError, ValueError) as error:
        raise FactsError(f"trace frame·step·시각 순서가 계약과 다르다: {error}") from error
    if rebound != records:
        raise FactsError("trace record가 PTS 결합 결과와 다르다")
    for index, row in enumerate(records):
        state = row.get("state")
        if not isinstance(state, Mapping):
            raise FactsError(f"trace frame {index} state가 없다")
        _finite_vector(state.get("command"), 3, f"frame {index} command")
        _finite_vector(state.get("motion"), 3, f"frame {index} motion")
    return records


def _check_sidecar(manifest: Mapping[str, Any], records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if manifest.get("frame_count") != R.FRAME_COUNT or len(records) != R.FRAME_COUNT:
        raise FactsError("sidecar가 625 frame 완료본이 아니다")
    joints = manifest.get("joint_names")
    if not isinstance(joints, list) or not joints or not all(isinstance(name, str) and name for name in joints):
        raise FactsError("sidecar joint_names가 없다")
    available = manifest.get("available_fields")
    unavailable = manifest.get("unavailable_fields")
    if not isinstance(available, Mapping) or not isinstance(unavailable, Mapping):
        raise FactsError("sidecar 필드 가용성 선언이 없다")
    status: dict[str, dict[str, Any]] = {}
    for name, unit in JOINT_FIELDS.items():
        if name in available:
            spec = available[name]
            if not isinstance(spec, Mapping) or spec.get("unit") != unit:
                raise FactsError(f"sidecar {name} 단위가 {unit}이 아니다")
            status[name] = {"status": "available", "unit": unit}
        elif name in unavailable:
            status[name] = {"status": "unavailable", "reason": str(unavailable[name])}
        else:
            raise FactsError(f"sidecar {name} 가용성이 선언되지 않았다")
    for index, row in enumerate(records):
        fields = row.get("fields")
        if not isinstance(fields, Mapping):
            raise FactsError(f"sidecar frame {index} fields가 없다")
        for name, info in status.items():
            if info["status"] == "available":
                if name not in fields:
                    raise FactsError(f"sidecar frame {index}에서 {name}이 빠졌다")
                _finite_vector(fields[name], len(joints), f"frame {index} {name}")
    contact = {name: ("available" if name in available else f"unavailable: {unavailable.get(name)}")
               for name, _, _, _ in SIDECAR.CONTACT_FIELDS}
    return {"joints": list(joints), "fields": status, "contact": contact}


def _window_indices(records: Sequence[Mapping[str, Any]]) -> list[list[int]]:
    windows = []
    for timestamps in R.WINDOWS:
        indices = []
        for timestamp in timestamps:
            matches = [index for index, row in enumerate(records) if row["actual_timestamp_s"] == timestamp]
            if len(matches) != 1:
                raise FactsError(f"선언 sample 시각 {timestamp}s와 같은 PTS frame이 하나가 아니다")
            indices.append(matches[0])
        windows.append(indices)
    return windows


def _per_joint(values: Sequence[Sequence[float]], joints: Sequence[str], anchor: str,
               keys: Sequence[str] = ("mean", "min", "max")) -> dict[str, dict[str, float]]:
    result = {}
    for column, joint in enumerate(joints):
        stats = _stats([row[column] for row in values], f"{anchor} {joint}")
        result[joint] = {key: stats[key] for key in keys}
    return result


def _window_facts(window_id: str, indices: Sequence[int], trace: Sequence[Mapping[str, Any]],
                  sidecar: Sequence[Mapping[str, Any]], checked: Mapping[str, Any]) -> dict[str, Any]:
    commands = [_finite_vector(trace[i]["state"]["command"], 3, "command") for i in indices]
    motions = [_finite_vector(trace[i]["state"]["motion"], 3, "motion") for i in indices]
    command_norm = [math.hypot(cmd[0], cmd[1]) for cmd in commands]
    speed_norm = [math.hypot(mot[0], mot[1]) for mot in motions]
    error_norm = [math.hypot(cmd[0] - mot[0], cmd[1] - mot[1]) for cmd, mot in zip(commands, motions, strict=True)]
    zero = sum(norm == 0.0 for norm in command_norm)
    facts: dict[str, Any] = {
        f"{window_id}.command_horizontal_norm": {
            "unit": "m/s", "frame": "body_xy", "definition": "hypot(command.linear_x, command.linear_y)",
            "per_sample": command_norm, **_stats(command_norm, "command norm")},
        f"{window_id}.body_horizontal_speed_norm": {
            "unit": "m/s", "frame": "body_xy", "definition": "hypot(motion.linear_x, motion.linear_y)",
            "per_sample": speed_norm, **_stats(speed_norm, "speed norm")},
        f"{window_id}.command_tracking_error_norm": {
            "unit": "m/s", "frame": "body_xy",
            "definition": "hypot(command.linear_x - motion.linear_x, command.linear_y - motion.linear_y); 두 norm의 차가 아니다",
            "per_sample": error_norm, **_stats(error_norm, "error norm")},
    }
    if zero:
        facts[f"{window_id}.command_tracking_error_relative"] = {
            "unit": "1", "value": None, "reason": f"command_horizontal_norm_zero_in_{zero}_of_{len(indices)}_samples"}
    else:
        relative = [err / norm for err, norm in zip(error_norm, command_norm, strict=True)]
        facts[f"{window_id}.command_tracking_error_relative"] = {
            "unit": "1", "definition": "command_tracking_error_norm / command_horizontal_norm",
            "per_sample": relative, **_stats(relative, "relative error")}
    joints = checked["joints"]
    status = checked["fields"]

    def rows(name: str) -> list[list[float]]:
        return [_finite_vector(sidecar[i]["fields"][name], len(joints), name) for i in indices]

    def unavailable(*names: str) -> dict[str, Any]:
        reasons = {name: status[name].get("reason") for name in names if status[name]["status"] != "available"}
        return {"value": None, "status": "unavailable", "reason": reasons}

    if status["joint_vel"]["status"] == "available":
        absolute = [[abs(item) for item in row] for row in rows("joint_vel")]
        facts[f"{window_id}.joint_abs_velocity"] = {
            "unit": "rad/s", "per_joint": _per_joint(absolute, joints, "joint_vel", ("mean", "max"))}
    else:
        facts[f"{window_id}.joint_abs_velocity"] = unavailable("joint_vel")
    for name in ("computed_torque", "applied_torque"):
        key = f"{window_id}.{name}"
        facts[key] = ({"unit": "N*m", "source": "recorded", "per_joint": _per_joint(rows(name), joints, name)}
                      if status[name]["status"] == "available" else unavailable(name))
    if status["computed_torque"]["status"] == "available" and status["applied_torque"]["status"] == "available":
        difference = [[abs(c - a) for c, a in zip(cr, ar, strict=True)]
                      for cr, ar in zip(rows("computed_torque"), rows("applied_torque"), strict=True)]
        facts[f"{window_id}.torque_clip_abs_difference"] = {
            "unit": "N*m", "definition": "|computed_torque - applied_torque| (둘 다 기록값)",
            "per_joint": _per_joint(difference, joints, "torque difference", ("mean", "max"))}
    else:
        facts[f"{window_id}.torque_clip_abs_difference"] = unavailable("computed_torque", "applied_torque")
    return facts


def build_state_facts(
    sidecar_dir: Path,
    trace_path: Path,
    native_manifest_path: Path,
    *,
    expected: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """검증된 결합만 받아 ``{"public": ..., "metadata": ...}``를 돌려준다. 실패하면 FactsError."""
    sidecar_dir, trace_path, native_manifest_path = Path(sidecar_dir), Path(trace_path), Path(native_manifest_path)
    if (sidecar_dir / SIDECAR.FAILURE_NAME).exists():
        raise FactsError("sidecar가 실패로 봉인됐다")
    try:
        manifest, sidecar_records = SIDECAR.load_sidecar(sidecar_dir)
        join = SIDECAR.join_with_trace(sidecar_dir, trace_path, native_manifest_path)
        trace = R.read_object(trace_path)
    except (OSError, TypeError, ValueError, KeyError, SIDECAR.SidecarError) as error:
        raise FactsError(f"sidecar·trace 결합이 검증되지 않았다: {error}") from error
    inputs = {
        "raw_trace_sha256": _sha256_file(trace_path),
        "native_capture_manifest_sha256": _sha256_file(native_manifest_path),
        "sidecar_manifest_sha256": _sha256_file(sidecar_dir / SIDECAR.MANIFEST_NAME),
        "sidecar_records_sha256": manifest["records"]["sha256"],
    }
    for key, value in (expected or {}).items():
        if key not in inputs or inputs[key] != value:
            raise FactsError(f"입력 {key}가 등록 SHA와 다르다")
    if (join["trace_sha256"] != inputs["raw_trace_sha256"]
            or join["native_manifest_sha256"] != inputs["native_capture_manifest_sha256"]
            or join["sidecar_manifest_sha256"] != inputs["sidecar_manifest_sha256"]):
        raise FactsError("join 결과와 입력 SHA가 다르다")
    trace_records = _check_trace(trace)
    source = manifest.get("source")
    if not isinstance(source, Mapping) or source.get("source_id") != trace.get("job_id"):
        raise FactsError("sidecar source와 trace job이 다르다")
    checked = _check_sidecar(manifest, sidecar_records)
    windows = _window_indices(trace_records)
    facts: dict[str, Any] = {}
    window_rows = []
    for number, indices in enumerate(windows, 1):
        window_id = f"w{number}"
        facts.update(_window_facts(window_id, indices, trace_records, sidecar_records, checked))
        window_rows.append({
            "window_id": window_id,
            "declared_sample_timestamps_s": list(R.WINDOWS[number - 1]),
            "sample_count": len(indices),
            "samples": [{"timestamp_s": trace_records[i]["actual_timestamp_s"], "frame_index": i,
                         "simulation_step": trace_records[i]["simulation_step"],
                         "simulation_time_s": trace_records[i]["state"]["simulation_time_s"]} for i in indices],
        })
    binding = {"job_id": trace["job_id"], "sidecar_source": dict(source),
               "checkpoint_sha256": trace["checkpoint_sha256"], "video_sha256": trace["video_sha256"]}
    public = {
        "schema": SCHEMA,
        "inputs": {**inputs, "sidecar_source_sha256_lf": manifest["sidecar_source"]["sha256_lf"],
                   "video_sha256": trace["video_sha256"]},
        "binding": {"join_schema": join["schema"], "join_keys": join["join_keys"], "frame_count": R.FRAME_COUNT,
                    "fps": R.FPS, "binding_digest": R.digest(binding)},
        "time_basis": {
            "window_sample": f"{trace['time_basis']}; 선언 sample 시각과 PTS 정확 일치",
            "simulation": "trace·sidecar 공통 simulation_time_s(s)와 simulation_step",
        },
        "coordinate_frames": {"command": "body", "motion": "body", "world_position_used": False,
                              "joints": "sidecar joint_names 순서"},
        "units": {"linear_velocity": "m/s", "joint_velocity": "rad/s", "torque": "N*m", "relative": "1"},
        "windows": window_rows,
        "field_availability": {
            "command": "available", "motion": "available",
            **{name: info["status"] if info["status"] == "available" else f"unavailable: {info['reason']}"
               for name, info in checked["fields"].items()},
            **{name: f"{value}; 요약하지 않음" for name, value in checked["contact"].items()},
        },
        "joint_names": checked["joints"],
        "facts": facts,
        "unsupported_claims": [{"claim": claim, "reason": reason} for claim, reason in UNSUPPORTED_CLAIMS],
        "thresholds_created": 0,
        "behavior_label": None,
        "used_for_decision": False,
    }
    metadata = {"schema": METADATA_SCHEMA, "public_sha256": R.digest(public), **binding,
                "binding_digest": public["binding"]["binding_digest"]}
    return {"public": public, "metadata": metadata}


def write_state_facts(result: Mapping[str, Any], out_dir: Path) -> dict[str, Path]:
    out = Path(out_dir)
    if out.exists():
        raise FileExistsError(f"state facts 출력이 이미 있다: {out}")
    out.mkdir(parents=True)
    public, metadata = out / PUBLIC_NAME, out / METADATA_NAME
    R.write_new(public, result["public"])
    R.write_new(metadata, result["metadata"])
    return {"public": public, "metadata": metadata}


def _expected_from_final(path: Path) -> dict[str, str]:
    """JOB09 agentic-play-final.json의 join SHA를 등록값으로 쓴다(최종 기술 완료본만)."""
    final = R.read_object(Path(path))
    join = final.get("join")
    if final.get("final_technical_complete") is not True or not isinstance(join, Mapping):
        raise FactsError("agentic play final이 기술 완료본이 아니다")
    return {"raw_trace_sha256": join["trace_sha256"],
            "native_capture_manifest_sha256": join["native_manifest_sha256"],
            "sidecar_manifest_sha256": join["sidecar_manifest_sha256"]}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sidecar", required=True, type=Path)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--native-manifest", required=True, type=Path)
    parser.add_argument("--final", type=Path, help="agentic-play-final.json (join SHA 대조)")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    expected = _expected_from_final(args.final) if args.final is not None else None
    result = build_state_facts(args.sidecar, args.trace, args.native_manifest, expected=expected)
    paths = write_state_facts(result, args.out)
    print(json.dumps({key: str(value) for key, value in paths.items()}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
