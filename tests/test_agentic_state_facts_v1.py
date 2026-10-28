from __future__ import annotations

import json
import math
import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest  # pyright: ignore[reportMissingImports]

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / "bench", ROOT / "evals", ROOT / "src"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import agentic_state_facts_v1 as F  # pyright: ignore[reportMissingImports]
import record_agentic_state_sidecar_v1 as S  # pyright: ignore[reportMissingImports]
import record_vision_evidence_v1 as R  # pyright: ignore[reportMissingImports]

CK = "a" * 64
OTHER_CK = "b" * 64
JOB_ID = "agentic-play-20261028a-go2-s2033-f2"
JOINTS = ["FL_hip_joint", "FR_hip_joint"]
REAL_F1 = ROOT / "evals/results/vision_transferred_pull_record_dev_20261027b_v3/f1"
Vector = Callable[[int], list[float]]


def _sim_time(index: int) -> float:
    return 0.01 + 0.02 * index


class Reader:
    """synthetic state reader(CPU fixture). 실제 Isaac 값이 아니다."""

    def __init__(self, joint_fn: Callable[[int], dict[str, list[float]]], unavailable: dict[str, str]) -> None:
        self.joint_fn = joint_fn
        self.unavailable = dict(unavailable)
        self.now: tuple[float, int] = (_sim_time(0), 0)
        self.index = 0

    def describe(self) -> dict[str, Any]:
        available: dict[str, Any] = {}
        unavailable: dict[str, str] = {}
        for name, _, unit in S.ARTICULATION_FIELDS:
            if name in self.unavailable:
                unavailable[name] = self.unavailable[name]
            else:
                available[name] = {"unit": unit}
        for name, _, _, _ in S.CONTACT_FIELDS:
            unavailable[name] = "scene_contact_sensor_missing"
        return {"reader": "synthetic_fixture_v1", "joint_names": list(JOINTS), "contact_body_names": None,
                "available": available, "unavailable": unavailable, "not_recorded": dict(S.NOT_RECORDED),
                "contact_force_threshold_n": None, "actuators": None}

    def clock(self) -> tuple[float, int]:
        return self.now

    def read(self) -> dict[str, Any]:
        values = self.joint_fn(self.index)
        fields = {name: list(values.get(name, [0.5, -0.5])) for name, _, _ in S.ARTICULATION_FIELDS
                  if name not in self.unavailable}
        return {"fields": fields, "frame_unavailable": {}}


def _default_joints(index: int) -> dict[str, list[float]]:
    return {"joint_vel": [-0.001 * index, 2.0], "computed_torque": [30.0, -30.0], "applied_torque": [23.5, -23.5]}


def build_inputs(
    tmp: Path,
    *,
    command: Vector = lambda _i: [1.0, 0.0, 0.0],
    motion: Vector = lambda _i: [0.5, 0.0, 0.0],
    joints: Callable[[int], dict[str, list[float]]] = _default_joints,
    unavailable: dict[str, str] | None = None,
    sidecar_checkpoint: str = CK,
    source_id: str = JOB_ID,
    frames: int = R.FRAME_COUNT,
) -> tuple[Path, Path, Path]:
    reader = Reader(joints, unavailable or {})
    source = S.SourceIdentity(source_id=source_id, checkpoint_sha256=sidecar_checkpoint, task=R.TASK)
    recorder = S.SidecarRecorder(tmp / "sidecar", source, expected_frames=frames)
    recorder.start(reader)
    for index in range(frames):
        reader.index = index
        reader.now = (_sim_time(index), index)
        recorder.record(index, capture_ordinal=index)
    recorder.finalize()
    records = [{
        "frame_index": index, "capture_ordinal": index, "simulation_step": index,
        "state": {"command": command(index), "motion": motion(index), "pose": {"base_height_m": 0.3, "up_axis_z": 1.0},
                  "root_position_world_m": [0.01 * index, 0.0, 0.3], "simulation_time_s": _sim_time(index),
                  "common_step_counter": index},
        "camera": {}, "render_timing": {}, "rgb_sha256": "0" * 64, "actual_timestamp_s": index / R.FPS,
    } for index in range(R.FRAME_COUNT)]
    trace = {
        "schema": R.TRACE_SCHEMA, "method": R.METHOD, "job_id": JOB_ID, "profile": R.fixed_profile(),
        "profile_sha256": R.digest(R.fixed_profile()), "video_sha256": "c" * 64, "checkpoint_sha256": CK,
        "time_basis": R.fixed_profile()["time_basis"], "frame_count": R.FRAME_COUNT, "records": records,
        "encoder_receipt": {}, "render_alignment": {},
        "timing_receipt": {"source_sha256": "c" * 64, "frame_count": R.FRAME_COUNT,
                           "relative_actual_pts_s": [row["actual_timestamp_s"] for row in records]},
    }
    trace_path = tmp / "raw_trace.json"
    trace_path.write_text(json.dumps(trace), encoding="utf-8")
    manifest_path = tmp / "native_capture_manifest.json"
    manifest_path.write_text(json.dumps({"checkpoint_sha256": CK, "frame_count": R.FRAME_COUNT}), encoding="utf-8")
    return tmp / "sidecar", trace_path, manifest_path


def _rehash(sidecar: Path, mutate_line: Callable[[int, dict[str, Any]], None]) -> None:
    """생산자 결함 흉내: records를 바꾼 뒤 manifest 해시까지 맞춘다(load_sidecar는 통과)."""
    path = sidecar / S.RECORDS_NAME
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    for index, row in enumerate(rows):
        mutate_line(index, row)
    path.write_text("".join(json.dumps(row, allow_nan=True) + "\n" for row in rows), encoding="utf-8", newline="\n")
    manifest_path = sidecar / S.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["records"].update(sha256=S._sha256_path(path), bytes=path.stat().st_size)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def _facts(tmp: Path, **kwargs: Any) -> dict[str, Any]:
    return F.build_state_facts(*build_inputs(tmp, **kwargs))


# --- 계산 의미 -------------------------------------------------------------------------------


def test_tracking_error_is_vector_difference_not_norm_difference(tmp_path: Path) -> None:
    result = _facts(tmp_path, command=lambda _i: [1.0, 0.0, 0.0], motion=lambda _i: [0.0, 1.0, 0.0])
    facts = result["public"]["facts"]
    assert facts["w1.command_horizontal_norm"]["mean"] == 1.0
    assert facts["w1.body_horizontal_speed_norm"]["mean"] == 1.0  # norm 차는 0
    assert facts["w1.command_tracking_error_norm"]["mean"] == pytest.approx(math.sqrt(2.0), abs=1e-15)
    assert facts["w2.command_tracking_error_relative"]["max"] == pytest.approx(math.sqrt(2.0), abs=1e-15)
    assert facts["w1.command_tracking_error_norm"]["frame"] == "body_xy"


def test_window_samples_are_declared_exact_pts(tmp_path: Path) -> None:
    public = _facts(tmp_path)["public"]
    assert [row["declared_sample_timestamps_s"] for row in public["windows"]] == [list(w) for w in R.WINDOWS]
    assert [row["sample_count"] for row in public["windows"]] == [6, 6]
    assert [s["frame_index"] for s in public["windows"][0]["samples"]] == [50, 100, 150, 200, 250, 300]
    assert public["windows"][1]["samples"][-1] == {"timestamp_s": 12.0, "frame_index": 600,
                                                   "simulation_step": 600, "simulation_time_s": _sim_time(600)}
    assert len(public["facts"]["w1.command_horizontal_norm"]["per_sample"]) == 6


def test_zero_command_relative_error_is_null_with_reason(tmp_path: Path) -> None:
    facts = _facts(tmp_path, command=lambda _i: [0.0, 0.0, 0.0],
                   motion=lambda _i: [0.3, 0.4, 0.0])["public"]["facts"]
    relative = facts["w1.command_tracking_error_relative"]
    assert relative["value"] is None and relative["reason"] == "command_horizontal_norm_zero_in_6_of_6_samples"
    assert "mean" not in relative
    assert facts["w1.command_tracking_error_norm"]["mean"] == pytest.approx(0.5, abs=1e-15)


def test_joint_and_recorded_torque_statistics(tmp_path: Path) -> None:
    facts = _facts(tmp_path)["public"]["facts"]
    joint = facts["w1.joint_abs_velocity"]["per_joint"]
    assert joint["FL_hip_joint"]["max"] == pytest.approx(0.3, abs=1e-15)
    assert joint["FL_hip_joint"]["mean"] == pytest.approx(0.001 * (50 + 100 + 150 + 200 + 250 + 300) / 6, abs=1e-15)
    assert joint["FR_hip_joint"] == {"mean": 2.0, "max": 2.0}
    assert facts["w2.computed_torque"]["per_joint"]["FR_hip_joint"] == {"mean": -30.0, "min": -30.0, "max": -30.0}
    assert facts["w2.applied_torque"]["source"] == "recorded"
    assert facts["w1.torque_clip_abs_difference"]["per_joint"]["FL_hip_joint"] == {"mean": 6.5, "max": 6.5}


def test_unavailable_field_propagates_without_zero_fill(tmp_path: Path) -> None:
    public = _facts(tmp_path, unavailable={"computed_torque": "articulation_attribute_missing"})["public"]
    facts = public["facts"]
    for key in ("w1.computed_torque", "w2.torque_clip_abs_difference"):
        assert facts[key]["value"] is None and facts[key]["status"] == "unavailable"
        assert "per_joint" not in facts[key]
    assert facts["w1.torque_clip_abs_difference"]["reason"] == {"computed_torque": "articulation_attribute_missing"}
    assert public["field_availability"]["computed_torque"] == "unavailable: articulation_attribute_missing"
    assert public["field_availability"]["applied_torque"] == "available"
    assert public["field_availability"]["contact_net_force_w"].startswith("unavailable")


# --- 결합 거부 -------------------------------------------------------------------------------


def test_wrong_checkpoint_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(F.FactsError, match="결합"):
        _facts(tmp_path, sidecar_checkpoint=OTHER_CK)


def test_sidecar_source_must_match_trace_job(tmp_path: Path) -> None:
    with pytest.raises(F.FactsError, match="source"):
        _facts(tmp_path, source_id="agentic-play-20261028a-other")


def test_wrong_simulation_time_or_pts_is_rejected(tmp_path: Path) -> None:
    sidecar, trace, manifest = build_inputs(tmp_path)
    original = trace.read_text(encoding="utf-8")
    value = json.loads(original)
    value["records"][100]["state"]["simulation_time_s"] += 0.001
    trace.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(F.FactsError, match="결합"):
        F.build_state_facts(sidecar, trace, manifest)
    value = json.loads(original)
    for row in value["records"]:
        row["actual_timestamp_s"] = row["frame_index"] / R.FPS + 0.001
    value["timing_receipt"]["relative_actual_pts_s"] = [row["actual_timestamp_s"] for row in value["records"]]
    trace.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(F.FactsError, match="PTS"):
        F.build_state_facts(sidecar, trace, manifest)


@pytest.mark.parametrize("case", ["shape", "non-finite", "missing", "bool"])
def test_malformed_sidecar_values_are_rejected(tmp_path: Path, case: str) -> None:
    sidecar, trace, manifest = build_inputs(tmp_path)

    def mutate(index: int, row: dict[str, Any]) -> None:
        if index != 150:
            return
        if case == "shape":
            row["fields"]["joint_vel"] = [0.1]
        elif case == "non-finite":
            row["fields"]["applied_torque"] = [math.inf, 0.0]
        elif case == "missing":
            del row["fields"]["applied_torque"]
        else:
            row["fields"]["joint_vel"] = [True, 0.0]

    _rehash(sidecar, mutate)
    with pytest.raises(F.FactsError):
        F.build_state_facts(sidecar, trace, manifest)


def test_wrong_unit_is_rejected(tmp_path: Path) -> None:
    sidecar, trace, manifest = build_inputs(tmp_path)
    path = sidecar / S.MANIFEST_NAME
    value = json.loads(path.read_text(encoding="utf-8"))
    value["available_fields"]["joint_vel"]["unit"] = "deg/s"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(F.FactsError, match="단위"):
        F.build_state_facts(sidecar, trace, manifest)


def test_non_finite_trace_value_is_rejected(tmp_path: Path) -> None:
    sidecar, trace, manifest = build_inputs(tmp_path)
    value = json.loads(trace.read_text(encoding="utf-8"))
    value["records"][200]["state"]["motion"][0] = math.nan
    trace.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(F.FactsError):
        F.build_state_facts(sidecar, trace, manifest)


def test_tampered_sources_are_rejected(tmp_path: Path) -> None:
    sidecar, trace, manifest = build_inputs(tmp_path)
    original = F.build_state_facts(sidecar, trace, manifest)["public"]["inputs"]
    value = json.loads(trace.read_text(encoding="utf-8"))
    value["records"][250]["state"]["command"] = [0.9, 0.0, 0.0]  # 형식은 맞지만 원문이 바뀜
    trace.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(F.FactsError, match="등록 SHA"):
        F.build_state_facts(sidecar, trace, manifest, expected={"raw_trace_sha256": original["raw_trace_sha256"]})
    value["timing_receipt"]["relative_actual_pts_s"][3] = 9.0
    trace.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(F.FactsError, match="timing"):
        F.build_state_facts(sidecar, trace, manifest)
    records = sidecar / S.RECORDS_NAME
    records.write_text(records.read_text(encoding="utf-8").replace("23.5", "23.4", 1), encoding="utf-8")
    with pytest.raises(F.FactsError, match="결합"):
        F.build_state_facts(sidecar, trace, manifest)


def test_incomplete_or_failed_sidecar_is_rejected(tmp_path: Path) -> None:
    sidecar, trace, manifest = build_inputs(tmp_path / "short", frames=624)
    with pytest.raises(F.FactsError):
        F.build_state_facts(sidecar, trace, manifest)
    failed = tmp_path / "failed"
    recorder = S.SidecarRecorder(failed / "sidecar", S.SourceIdentity(source_id=JOB_ID, checkpoint_sha256=CK,
                                                                       task=R.TASK))
    recorder.fail(RuntimeError("native failed"), stage="native_record")
    _, trace2, manifest2 = build_inputs(tmp_path / "other")
    with pytest.raises(F.FactsError, match="실패"):
        F.build_state_facts(failed / "sidecar", trace2, manifest2)


# --- 공개 범위 -------------------------------------------------------------------------------


def test_public_facts_do_not_leak_job_or_fault_ids(tmp_path: Path) -> None:
    result = _facts(tmp_path)
    text = json.dumps(result["public"], ensure_ascii=False)
    for secret in (JOB_ID, "go2-s2033-f2", "go2-s2033"):
        assert secret not in text
    assert result["metadata"]["job_id"] == JOB_ID and result["metadata"]["sidecar_source"]["source_id"] == JOB_ID
    assert result["metadata"]["binding_digest"] == result["public"]["binding"]["binding_digest"]
    assert all(re.fullmatch(r"w[12]\.[a-z_]+", key) for key in result["public"]["facts"])
    public = result["public"]
    assert public["thresholds_created"] == 0 and public["behavior_label"] is None
    claims = {row["claim"] for row in public["unsupported_claims"]}
    assert {"gait_or_walking_state", "single_leg_fault_localization", "fall_judgement"} <= claims
    assert public["coordinate_frames"] == {"command": "body", "motion": "body", "world_position_used": False,
                                           "joints": "sidecar joint_names 순서"}


def test_cli_writes_once_and_checks_final_record(tmp_path: Path) -> None:
    sidecar, trace, manifest = build_inputs(tmp_path)
    inputs = F.build_state_facts(sidecar, trace, manifest)["public"]["inputs"]
    final = tmp_path / "agentic-play-final.json"
    final.write_text(json.dumps({"final_technical_complete": True, "join": {
        "trace_sha256": inputs["raw_trace_sha256"], "native_manifest_sha256": inputs["native_capture_manifest_sha256"],
        "sidecar_manifest_sha256": inputs["sidecar_manifest_sha256"]}}), encoding="utf-8")
    args = ["--sidecar", str(sidecar), "--trace", str(trace), "--native-manifest", str(manifest),
            "--final", str(final), "--out", str(tmp_path / "facts")]
    assert F.main(args) == 0
    written = json.loads((tmp_path / "facts" / F.PUBLIC_NAME).read_text(encoding="utf-8"))
    assert written["schema"] == F.SCHEMA
    with pytest.raises(FileExistsError):
        F.main(args)
    final.write_text(json.dumps({"final_technical_complete": False, "join": {}}), encoding="utf-8")
    with pytest.raises(F.FactsError, match="완료"):
        F.main([*args[:-1], str(tmp_path / "facts2")])


@pytest.mark.skipif(not (REAL_F1 / "evidence/raw_trace.json").is_file(), reason="기존 F1 trace 원문 없음")
def test_real_f1_trace_shape_is_accepted_with_synthetic_sidecar(tmp_path: Path) -> None:
    """기존 F1 native trace(읽기 전용)와 같은 시각·step의 synthetic sidecar로 필드 형태를 대조한다."""
    trace_path = REAL_F1 / "evidence/raw_trace.json"
    manifest_path = REAL_F1 / "native_capture_manifest.json"
    trace = R.read_object(trace_path)
    reader = Reader(_default_joints, {})
    source = S.SourceIdentity(source_id=trace["job_id"], checkpoint_sha256=trace["checkpoint_sha256"], task=R.TASK)
    recorder = S.SidecarRecorder(tmp_path / "sidecar", source, expected_frames=R.FRAME_COUNT)
    recorder.start(reader)
    for row in trace["records"]:
        reader.index = row["frame_index"]
        reader.now = (row["state"]["simulation_time_s"], row["simulation_step"])
        recorder.record(row["frame_index"], capture_ordinal=row["capture_ordinal"])
    recorder.finalize()
    before = (S._sha256_path(trace_path), S._sha256_path(manifest_path))
    public = F.build_state_facts(tmp_path / "sidecar", trace_path, manifest_path)["public"]
    assert (S._sha256_path(trace_path), S._sha256_path(manifest_path)) == before
    assert public["facts"]["w1.command_horizontal_norm"]["per_sample"] == [1.0] * 6
    assert [s["timestamp_s"] for s in public["windows"][1]["samples"]] == list(R.WINDOWS[1])
    assert trace["job_id"] not in json.dumps(public)


def test_cpu_import_does_not_load_isaac_or_torch() -> None:
    script = (
        "import sys; sys.dont_write_bytecode = True; "
        f"[sys.path.insert(0, p) for p in {[str(ROOT / name) for name in ('src', 'bench', 'evals')]!r}]; "
        "import agentic_state_facts_v1; "
        "heavy = ('isaaclab', 'isaacsim', 'omni', 'carb', 'pxr', 'torch', 'gymnasium', 'rsl_rl'); "
        "print(sorted(name for name in sys.modules if name.split('.')[0] in heavy))"
    )
    result = subprocess.run([sys.executable, "-B", "-c", script], capture_output=True, text=True,
                            check=True, timeout=120)
    assert result.stdout.strip() == "[]"
