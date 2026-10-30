"""H1·Go2 계단 녹화 산출물을 공통 진단 입력용 관측 수치(fact)로 읽는 CPU 어댑터.

- 입력: 녹화기 interface ``run_stairs_playback_v2.py``(저장소 밖, protocol이 SHA로 결합)가 쓴 clip 폴더와 spec.
  녹화기를 다시 구현하지 않고 녹화기가 실제로 쓰는 spec·intent·manifest·shutdown marker·state trace 필드만 읽는다.
- 결합: spec SHA·checkpoint·runner·Isaac Lab source·robot/task/direction·video·trace SHA를 서로 대조한다.
  하나라도 다르면 부분 결과 없이 거부한다. clip 묶음에서는 robot별 checkpoint 일관성도 본다.
- 수치: trace의 root 위치·속도(world)·terrain origin·done만 쓴다. joint·contact·학습 telemetry·frame별 명령은
  기록되지 않으므로 unavailable과 이유를 낸다. done 행은 자동 reset 뒤 상태일 수 있어 이동 수치에서 뺀다.
- 좌표: 명령 좌표계와 quaternion 성분 순서가 기록에 없으므로 몸체 속도·추종 오차를 만들지 않는다.
- 판정: done만으로 넘어짐, root 높이 변화만으로 계단 성공·교정 성공을 단정하지 않는다. 녹화기 outcome은
  녹화기 규칙 라벨로만 옮기고 결정에 쓰지 않는다. 임계값을 만들지 않는다.
- 적용성: 진단 준비도·검사 적용성·교정 적용성을 따로 낸다. rough task에는 등록 probe·교정이 없어 모두 HOLD다.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _folder in (ROOT / "bench", ROOT / "evals", ROOT / "src"):
    if str(_folder) not in sys.path:
        sys.path.insert(0, str(_folder))

import agentic_state_facts_v1 as F

from rl_triage import correction_registry as CR
from rl_triage import probe_loop as L

R = F.R
FactsError = F.FactsError
SCHEMA = "rough_play_state_facts_v1"
METADATA_SCHEMA = "rough_play_state_facts_metadata_v1"
CONTRACT_SCHEMA = "rough_play_state_contract_v1"
CONTRACT = ROOT / "bench/protocols/rough_play_state_contract_20261030_v1.json"
PUBLIC_NAME = "rough_play_facts.json"
METADATA_NAME = "rough_play_facts_metadata.json"
CONTRACT_KEYS = ("recorder_interface", "clips", "registered_checkpoints", "coordinate_frames", "field_availability",
                 "fact_ids", "done_row_policy", "unsupported_claims", "applicability")
SHA = re.compile(r"[0-9a-f]{64}")
COMMIT = re.compile(r"[0-9a-f]{40}")


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise FactsError(reason)


def _read(path: Path, label: str) -> dict[str, Any]:
    try:
        return R.read_object(path)
    except (OSError, TypeError, ValueError) as error:
        raise FactsError(f"{label}를 읽지 못했다: {error}") from error


def _fields(value: Any, keys: Sequence[str], label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping) and set(value) == set(keys), f"{label} 필드가 녹화기 계약과 다르다")
    return value


def _inside(clip: Path, name: Any, label: str) -> Path:
    _require(isinstance(name, str) and bool(name) and "\\" not in name and ":" not in name
             and not Path(name).is_absolute() and ".." not in Path(name).parts, f"{label} 경로가 clip 상대경로가 아니다")
    path = clip / name
    _require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(clip.resolve()),
             f"{label} 파일이 clip 안에 없다")
    return path


def load_contract(path: Path = CONTRACT) -> dict[str, Any]:
    contract = _read(Path(path), "protocol")
    _require(contract.get("schema") == CONTRACT_SCHEMA and all(key in contract for key in CONTRACT_KEYS),
             "rough play protocol 형식이 다르다")
    return contract


def _check_spec(spec: Mapping[str, Any], contract: Mapping[str, Any]) -> None:
    iface = contract["recorder_interface"]
    _fields(spec, iface["spec_fields"], "spec")
    _require(spec["schema"] == iface["spec_schema"], "spec schema가 다르다")
    role, direction = spec["robot_role"], spec["direction"]
    _require({"robot_role": role, "direction": direction} in contract["clips"], "robot_role/direction이 등록 clip 밖이다")
    _require({"training_task": spec["training_task"], "play_task": spec["play_task"]} == iface["robot_tasks"][role],
             "robot task가 등록값과 다르다")
    _require(type(spec["seed"]) is int, "seed는 정수여야 한다")
    checkpoint = _fields(spec["checkpoint"], ("path", "sha256"), "spec checkpoint")
    _require(isinstance(checkpoint["sha256"], str) and bool(SHA.fullmatch(checkpoint["sha256"])),
             "checkpoint SHA 형식이 다르다")
    registered = contract["registered_checkpoints"][role]
    _require(registered is None or registered == checkpoint["sha256"], "checkpoint가 robot 등록 SHA와 다르다")
    source = _fields(spec["source"], ("isaaclab_commit", "isaaclab_files_lf", "runner_sha256_lf"), "spec source")
    _require(source["runner_sha256_lf"] == iface["sha256_lf"], "녹화기 runner SHA가 등록 interface와 다르다")
    _require(isinstance(source["isaaclab_commit"], str) and bool(COMMIT.fullmatch(source["isaaclab_commit"])),
             "Isaac Lab commit 형식이 다르다")
    files = source["isaaclab_files_lf"]
    required = set(iface["common_isaaclab_sources"]) | {iface["role_isaaclab_source"][role]}
    _require(isinstance(files, Mapping) and set(files) == required
             and all(isinstance(value, str) and SHA.fullmatch(value) for value in files.values()),
             "Isaac Lab source closure가 다르다")
    _require(spec["capture"] == iface["capture"], "capture 고정값이 다르다")


def _check_intent(clip: Path, spec_path: Path, spec: Mapping[str, Any], iface: Mapping[str, Any]) -> None:
    intent = _fields(_read(clip / "intent.json", "intent"), iface["intent_fields"], "intent")
    _require(intent["spec_sha256"] == F._sha256_file(spec_path), "intent가 다른 spec으로 만들어졌다")
    _require(intent["checkpoint_sha256"] == spec["checkpoint"]["sha256"], "intent checkpoint가 spec과 다르다")
    _require(all(type(intent[key]) is int and intent[key] == 0 for key in ("new_training", "model_api"))
             and intent["diagnosis_or_recovery_claim"] is False, "intent가 관측 전용 재생이 아니다")


def _check_manifest(manifest: Mapping[str, Any], spec: Mapping[str, Any], iface: Mapping[str, Any]) -> None:
    _fields(manifest, iface["manifest_fields"], "manifest")
    _require(manifest["schema"] == iface["manifest_schema"], "manifest schema가 다르다")
    for key in ("clip_id", "robot_role", "direction", "training_task", "play_task", "source"):
        _require(manifest[key] == spec[key], f"manifest {key}가 spec과 다르다")
    _require(manifest["checkpoint_sha256"] == spec["checkpoint"]["sha256"], "manifest checkpoint가 spec과 다르다")
    direction, terrain, capture = spec["direction"], manifest["terrain"], iface["capture"]
    _require(isinstance(terrain, Mapping) and terrain.get("terrain_id") == iface["terrain_for_direction"][direction]
             and terrain.get("direction_geometry") == iface["direction_geometry"][direction],
             "terrain이 direction과 다르다")
    expected = {"terrain_id": iface["terrain_for_direction"][direction], "terrain_seed": spec["seed"],
                "step_height_m": capture["step_height_m"], "step_width_m": capture["step_width_m"],
                **iface["terrain_fixed"], "command": iface["configured_command"],
                "direction_geometry": iface["direction_geometry"][direction]}
    _require(terrain == expected, "terrain 고정값이 spec·녹화기 계약과 다르다")
    _require(manifest["isaaclab_source"] == {"commit": spec["source"]["isaaclab_commit"],
                                             "files_lf": spec["source"]["isaaclab_files_lf"]},
             "실행 Isaac Lab source가 spec과 다르다")
    _require(manifest["camera"] == iface["camera"], "camera 고정값이 다르다")
    render = _fields(manifest["render_sync"], ("method", "calls", "physics_state_unchanged"), "render_sync")
    _require(render["method"] == iface["render_sync_method"] and render["physics_state_unchanged"] is True
             and type(render["calls"]) is int and render["calls"] >= 0, "render 동기화 기록이 다르다")
    video = manifest["record_video_contract"]
    _require(isinstance(video, Mapping) and video.get("video_length") == capture["frames"]
             and video.get("frames_before_close") == capture["frames"]
             and video.get("close_finalizes_recording") is True,
             "RecordVideo 계약 기록이 다르다")


def _check_markers(clip: Path, manifest_sha256: str, iface: Mapping[str, Any]) -> None:
    folder = clip / "shutdown"
    names = [f"{number:02d}-{event}.json" for number, event in enumerate(iface["shutdown_events"], 1)]
    _require(folder.is_dir() and sorted(path.name for path in folder.iterdir()) == names,
             "shutdown marker가 정상 종료 순서와 다르다")
    markers = [_read(folder / name, "shutdown marker") for name in names]
    for sequence, (event, marker) in enumerate(zip(iface["shutdown_events"], markers, strict=True), 1):
        _require(marker.get("schema") == iface["shutdown_marker_schema"] and marker.get("event") == event
                 and marker.get("sequence") == sequence and marker.get("pid") == markers[0].get("pid"),
                 "shutdown marker 내용이 다르다")
    clocks = [marker.get("monotonic_ns") for marker in markers]
    _require(all(type(value) is int for value in clocks) and clocks == sorted(set(clocks)),
             "shutdown marker 시각 순서가 다르다")
    _require(markers[1].get("manifest_sha256") == manifest_sha256, "result-ready marker가 manifest와 다르다")


def _reject_constant(value: str) -> None:
    raise FactsError(f"비유한 JSON 수 {value}는 허용되지 않는다")


def _trace_rows(path: Path, iface: Mapping[str, Any]) -> list[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8").split("\n")
    _require(lines[-1] == "", "state trace가 줄바꿈으로 끝나지 않는다")
    rows = []
    for index, line in enumerate(lines[:-1]):
        try:
            row = json.loads(line, parse_constant=_reject_constant)
        except json.JSONDecodeError as error:
            raise FactsError(f"trace 행 {index}가 JSON이 아니다") from error
        _fields(row, iface["trace_row_fields"], f"trace 행 {index}")
        _require(type(row["frame"]) is int and row["frame"] == index and row["sim_time_s"] == index / iface["fps"],
                 f"trace 행 {index} frame·시각이 녹화기 순서와 다르다")
        for name, length in (("root_pos_w", 3), ("root_quat_w", 4), ("root_lin_vel_w", 3), ("terrain_origin_w", 3)):
            row[name] = F._finite_vector(row[name], length, f"trace 행 {index} {name}")
        _require(type(row["terminated"]) is bool, f"trace 행 {index} terminated가 bool이 아니다")
        rows.append(row)
    done = [index for index, row in enumerate(rows) if row["terminated"]]
    _require(2 <= len(rows) <= iface["frames"] + 1, "trace 길이가 녹화기 계약과 다르다")
    _require(done in ([], [len(rows) - 1]) and (bool(done) or len(rows) == iface["frames"] + 1),
             "done 위치나 trace 길이가 녹화기 순서와 다르다")
    _require(all(row["terrain_origin_w"] == rows[0]["terrain_origin_w"] for row in rows),
             "단일 terrain origin이 trace 안에서 바뀌었다")
    return rows


def _check_assessment(assessment: Any, rows: Sequence[Mapping[str, Any]], iface: Mapping[str, Any]) -> dict[str, Any]:
    """녹화기 수치가 trace와 같은지만 본다. outcome 규칙은 다시 계산하지 않는다."""
    _fields(assessment, iface["assessment_fields"], "manifest assessment")
    first, last = rows[0]["root_pos_w"], rows[-1]["root_pos_w"]
    recomputed = {"root_dx_m": last[0] - first[0], "root_dz_m": last[2] - first[2],
                  "termination_seen": rows[-1]["terminated"], "trace_rows": len(rows)}
    _require(all(assessment[key] == value for key, value in recomputed.items()),
             "manifest assessment 수치가 trace와 다르다")
    _require(assessment["outcome"] in iface["recorder_label_rule"]["outcomes"]
             and all(type(assessment[key]) is bool for key in ("height_condition_pass", "progress_condition_pass")),
             "녹화기 라벨 형식이 다르다")
    return {"source": "manifest.assessment", "rule_owner": iface["recorder_label_rule"]["owner"],
            **{key: assessment[key] for key in ("outcome", "height_condition_pass", "progress_condition_pass")},
            "inputs_verified_against_trace": sorted(recomputed), "done_row_included": rows[-1]["terminated"],
            "used_for_decision": False}


def _unavailable(reason: Any) -> dict[str, Any]:
    return {"value": None, "status": "unavailable", "reason": reason}


def _clip_facts(rows: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any],
                contract: Mapping[str, Any]) -> dict[str, Any]:
    availability = contract["field_availability"]
    done = rows[-1]["terminated"]
    used = rows[:-1] if done else rows
    facts: dict[str, Any] = {}
    if len(used) < 2:
        for key in ("root.world_displacement", "root.world_xy_path_length", "root.height_above_terrain_origin",
                    "root.world_horizontal_speed", "root.world_vertical_velocity"):
            facts[key] = _unavailable(f"done 전 행이 {len(used)}개라 구간 수치를 만들 수 없다")
    else:
        span = {"first_frame": used[0]["frame"], "last_frame": used[-1]["frame"], "count": len(used)}
        position = [row["root_pos_w"] for row in used]
        velocity = [row["root_lin_vel_w"] for row in used]
        delta = [position[-1][axis] - position[0][axis] for axis in range(3)]
        height = [point[2] - used[0]["terrain_origin_w"][2] for point in position]
        facts["root.world_displacement"] = {
            "unit": "m", "frame": "world", "definition": "root_pos_w[last] - root_pos_w[first] (rows_used 구간)",
            "rows_used": span, "x": delta[0], "y": delta[1], "z": delta[2],
            "horizontal_norm": math.hypot(delta[0], delta[1])}
        facts["root.world_xy_path_length"] = {
            "unit": "m", "frame": "world_xy", "definition": "인접 행 root_pos_w xy 거리의 합", "rows_used": span,
            "value": math.fsum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in itertools.pairwise(position))}
        facts["root.height_above_terrain_origin"] = {
            "unit": "m", "frame": "world_z", "definition": "root_pos_w.z - terrain_origin_w.z; 발밑 지면 높이가 아니다",
            "rows_used": span, "first": height[0], "last": height[-1], **F._stats(height, "root height")}
        facts["root.world_horizontal_speed"] = {
            "unit": "m/s", "frame": "world_xy", "definition": "hypot(root_lin_vel_w.x, root_lin_vel_w.y)",
            "rows_used": span, **F._stats([math.hypot(v[0], v[1]) for v in velocity], "horizontal speed")}
        facts["root.world_vertical_velocity"] = {
            "unit": "m/s", "frame": "world_z", "definition": "root_lin_vel_w.z",
            "rows_used": span, **F._stats([v[2] for v in velocity], "vertical velocity")}
    facts["command.configured"] = {
        "unit": {"lin_vel_x": "m/s", "lin_vel_y": "m/s", "ang_vel_z": "rad/s"},
        "source": "manifest.terrain.command 설정값",
        "value": dict(manifest["terrain"]["command"]), "frame": None, "frame_status": availability["command_frame"],
        "per_frame_status": availability["command_per_frame"]}
    facts["command.tracking_error"] = _unavailable(
        {key: availability[key] for key in ("command_frame", "command_per_frame", "root_quat_w")})
    facts["episode.done"] = {
        "seen": done, "first_done_frame": rows[-1]["frame"] if done else None,
        "cause": None, "cause_status": availability["termination_cause"],
        "done_row_in_motion_facts": False, "done_row_policy": contract["done_row_policy"]}
    facts["trace.coverage"] = {"rows": len(rows), "registered_rows": contract["recorder_interface"]["frames"] + 1,
                               "last_sim_time_s": rows[-1]["sim_time_s"]}
    _require(list(facts) == contract["fact_ids"], "어댑터 fact 목록이 protocol과 다르다")
    return facts


def read_clip(clip_dir: Path, spec_path: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    """한 clip 폴더와 spec을 교차 검증해 clip 수치를 돌려준다. 실패하면 FactsError."""
    clip, spec_path = Path(clip_dir), Path(spec_path)
    iface = contract["recorder_interface"]
    _require(not (clip / "failure.json").exists(), "녹화가 실패로 봉인됐다")
    spec = _read(spec_path, "spec")
    _check_spec(spec, contract)
    _check_intent(clip, spec_path, spec, iface)
    manifest_path = clip / "manifest.json"
    manifest = _read(manifest_path, "manifest")
    _check_manifest(manifest, spec, iface)
    manifest_sha256 = F._sha256_file(manifest_path)
    _check_markers(clip, manifest_sha256, iface)
    video = _fields(manifest["video"], ("path", "sha256"), "manifest video")
    video_path = _inside(clip, video["path"], "video")
    _require(video_path.parent == clip / "video" and video_path.suffix == ".mp4"
             and len(list((clip / "video").glob("*.mp4"))) == 1, "RecordVideo mp4가 하나가 아니다")
    _require(F._sha256_file(video_path) == video["sha256"], "video SHA가 manifest와 다르다")
    trace = _fields(manifest["state_trace"], ("path", "sha256"), "manifest state_trace")
    _require(trace["path"] == iface["trace_path"], "state trace 경로가 녹화기 계약과 다르다")
    trace_path = _inside(clip, trace["path"], "state trace")
    _require(F._sha256_file(trace_path) == trace["sha256"], "state trace SHA가 manifest와 다르다")
    rows = _trace_rows(trace_path, iface)
    recorder = _check_assessment(manifest["assessment"], rows, iface)
    return {
        "clip_id": spec["clip_id"], "robot_role": spec["robot_role"], "direction": spec["direction"],
        "training_task": spec["training_task"], "play_task": spec["play_task"],
        "inputs": {"spec_sha256": F._sha256_file(spec_path), "intent_sha256": F._sha256_file(clip / "intent.json"),
                   "manifest_sha256": manifest_sha256, "state_trace_sha256": trace["sha256"],
                   "video_sha256": video["sha256"], "checkpoint_sha256": spec["checkpoint"]["sha256"],
                   "runner_sha256_lf": spec["source"]["runner_sha256_lf"],
                   "isaaclab_commit": spec["source"]["isaaclab_commit"],
                   "isaaclab_files_digest": R.digest(spec["source"]["isaaclab_files_lf"])},
        "coordinate_frames": dict(contract["coordinate_frames"]),
        "field_availability": dict(contract["field_availability"]),
        "facts": _clip_facts(rows, manifest, contract),
        "recorder_interpretation": recorder,
    }


def _check_set(clips: Sequence[Mapping[str, Any]]) -> None:
    _require(bool(clips), "clip이 없다")
    ids = [clip["clip_id"] for clip in clips]
    _require(len(set(ids)) == len(ids), "clip_id가 중복됐다")
    pairs = [(clip["robot_role"], clip["direction"]) for clip in clips]
    _require(len(set(pairs)) == len(pairs), "같은 robot·direction clip이 중복됐다")
    by_role: dict[str, set[str]] = {}
    for clip in clips:
        by_role.setdefault(clip["robot_role"], set()).add(clip["inputs"]["checkpoint_sha256"])
    _require(all(len(values) == 1 for values in by_role.values()), "같은 robot의 clip이 다른 checkpoint를 쓴다")
    owners = [next(iter(values)) for values in by_role.values()]
    _require(len(set(owners)) == len(owners), "다른 robot clip이 같은 checkpoint를 쓴다")
    _require(len({clip["inputs"]["isaaclab_commit"] for clip in clips}) == 1, "clip마다 Isaac Lab commit이 다르다")


def applicability(contract: Mapping[str, Any]) -> dict[str, Any]:
    """진단 준비도·검사 적용성·교정 적용성을 분리한다. rough task 등록이 없으므로 요청을 만들지 않는다."""
    rules = contract["applicability"]
    return {
        "diagnosis_readiness": dict(rules["diagnosis_readiness"]),
        "probe_applicability": {name: {"status": CR.HOLD, "reason": rules["probe_reason"]} for name in L.PROBES},
        "repair_applicability": {mechanism: {"correction_id": row["correction_id"], "status": CR.HOLD,
                                             "reason": rules["correction_reason"]}
                                 for mechanism, row in CR.REGISTRY.items()},
        "requests_created": 0,
    }


def check_repair(public: Mapping[str, Any], correction_id: str) -> dict[str, Any]:
    """rough clip 수치에 대한 교정 요청은 등록 밖이다. 원장에 쓰지 않고 HOLD를 돌려준다."""
    _require(isinstance(public, Mapping) and public.get("schema") == SCHEMA, "rough play facts가 아니다")
    flat = {row["correction_id"] for row in public["applicability"]["repair_applicability"].values()}
    rule = ("flat_registered_correction_not_registered_for_rough_task" if correction_id in flat
            else "unregistered_correction")
    return {"action": CR.HOLD, "correction_id": correction_id, "rule": rule,
            "requests_created": 0, "ledger_written": False}


def build_rough_play_facts(pairs: Sequence[tuple[Path, Path]], *, contract_path: Path = CONTRACT) -> dict[str, Any]:
    """(clip 폴더, spec) 쌍들을 검증해 ``{"public": ..., "metadata": ...}``를 돌려준다. 실패하면 FactsError."""
    contract = load_contract(contract_path)
    try:
        clips = [read_clip(Path(clip), Path(spec), contract) for clip, spec in pairs]
    except FactsError:
        raise
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise FactsError(f"clip 입력이 녹화기 계약과 다르다: {error!r}") from error
    _check_set(clips)
    public = {
        "schema": SCHEMA,
        "contract": {"name": Path(contract_path).name, "sha256_lf": R.sha256_file(Path(contract_path), lf=True),
                     "status": contract.get("status"),
                     "recorder_interface_sha256_lf": contract["recorder_interface"]["sha256_lf"]},
        "adapter_sha256_lf": R.sha256_file(Path(__file__), lf=True),
        "clips": clips,
        "applicability": applicability(contract),
        "unsupported_claims": [dict(row) for row in contract["unsupported_claims"]],
        "evidence_accounting": {"clip_count": len(clips), "simulator_fact_source_count": len(clips),
                                "recorder_rule_label_count": len(clips), "model_interpretation_source_count": 0,
                                "independent_confirmation_sources_added": 0, "video_frames_decoded": 0},
        "thresholds_created": 0,
        "behavior_label": None,
        "used_for_decision": False,
    }
    metadata = {"schema": METADATA_SCHEMA, "public_sha256": R.digest(public),
                "inputs": [{"clip_dir": str(Path(clip)), "spec": str(Path(spec))} for clip, spec in pairs]}
    return {"public": public, "metadata": metadata}


def write_rough_play_facts(result: Mapping[str, Any], out_dir: Path) -> dict[str, Path]:
    out = Path(out_dir)
    if out.exists():
        raise FileExistsError(f"rough play facts 출력이 이미 있다: {out}")
    out.mkdir(parents=True)
    public, metadata = out / PUBLIC_NAME, out / METADATA_NAME
    R.write_new(public, result["public"])
    R.write_new(metadata, result["metadata"])
    return {"public": public, "metadata": metadata}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", required=True, nargs=2, action="append", type=Path, metavar=("CLIP_DIR", "SPEC"),
                        help="녹화기 --out 폴더와 그 --spec 파일(clip마다 반복)")
    parser.add_argument("--protocol", type=Path, default=CONTRACT)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    result = build_rough_play_facts([(clip, spec) for clip, spec in args.clip], contract_path=args.protocol)
    paths = write_rough_play_facts(result, args.out)
    print(json.dumps({key: str(value) for key, value in paths.items()}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
