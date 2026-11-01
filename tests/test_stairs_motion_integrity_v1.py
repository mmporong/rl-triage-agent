# pyright: reportMissingImports=false
"""계단 몸체 방향 무결성 감사의 수치·입력 거부 경계를 검증한다."""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))
import stairs_motion_integrity_v1 as S


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path, *, final_angle: float = 0.0,
             outcome: str = "completed") -> tuple[Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    trace = tmp_path / "state-trace.jsonl"
    with trace.open("w", encoding="utf-8", newline="\n") as stream:
        for frame in range(626):
            angle = final_angle if frame >= 613 else 0.0
            row = {"frame": frame, "sim_time_s": frame / 50,
                   "root_pos_w": [frame / 100, 0.0, 0.5],
                   "root_quat_w": [math.cos(angle / 2), math.sin(angle / 2), 0.0, 0.0],
                   "root_lin_vel_w": [0.2, 0.0, 0.0],
                   "terrain_origin_w": [[0.0, 0.0, 0.0]], "terminated": False}
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    manifest = tmp_path / "manifest.json"
    value = {"schema": "stairs_playback_manifest_v2", "clip_id": "fixture",
             "robot_role": "quadruped", "direction": "down", "checkpoint_sha256": "a" * 64,
             "assessment": {"outcome": outcome, "root_dx_m": 6.25, "root_dz_m": -0.3,
                            "height_condition_pass": True, "progress_condition_pass": True,
                            "termination_seen": False, "trace_rows": 626},
             "state_trace": {"path": "state-trace.jsonl", "sha256": _sha(trace)},
             "record_video_contract": {"video_length": 625, "executed_steps": 625}}
    manifest.write_text(json.dumps(value), encoding="utf-8")
    return manifest, trace


def _audit(manifest: Path, trace: Path) -> dict:
    return S.audit(manifest, trace, expected_manifest_sha256=_sha(manifest),
                   expected_trace_sha256=_sha(trace))


def test_completed_trace_with_negative_body_up_is_contradicted(tmp_path):
    manifest, trace = _fixture(tmp_path, final_angle=2.2)
    result = _audit(manifest, trace)
    assert result["integrity_result"] == "upright_completion_contradicted"
    assert result["body_up_world_z"]["negative_count"] == 13
    assert result["body_up_world_z"]["first_negative_time_s"] == 12.26
    assert result["body_up_world_z"]["last"] < 0
    assert result["interpretation"]["fall_cause_confirmed"] is False


def test_no_inversion_is_not_promoted_to_full_stability_and_failure_is_preserved(tmp_path):
    stable_manifest, stable_trace = _fixture(tmp_path / "stable")
    stable = _audit(stable_manifest, stable_trace)
    assert stable["integrity_result"] == "no_inversion_observed_not_full_stability_proof"
    assert stable["body_up_world_z"]["negative_count"] == 0
    failed_manifest, failed_trace = _fixture(tmp_path / "failed", final_angle=2.2,
                                             outcome="failed")
    failed = _audit(failed_manifest, failed_trace)
    assert failed["integrity_result"] == "original_failure_preserved"


@pytest.mark.parametrize("mutation", ["zero_quaternion", "missing_quaternion", "nan_time"])
def test_invalid_or_unknown_orientation_is_rejected_without_zero_fabrication(tmp_path, mutation):
    manifest, trace = _fixture(tmp_path)
    rows = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    if mutation == "zero_quaternion":
        rows[-1]["root_quat_w"] = [0.0, 0.0, 0.0, 0.0]
    elif mutation == "missing_quaternion":
        del rows[-1]["root_quat_w"]
    else:
        rows[-1]["sim_time_s"] = float("nan")
    trace.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    value = json.loads(manifest.read_text(encoding="utf-8"))
    value["state_trace"]["sha256"] = _sha(trace)
    manifest.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError):
        _audit(manifest, trace)


def test_hash_path_frame_and_time_contracts_are_strict(tmp_path):
    manifest, trace = _fixture(tmp_path)
    with pytest.raises(ValueError, match="기대 SHA"):
        S.audit(manifest, trace, expected_manifest_sha256="0" * 64,
                expected_trace_sha256=_sha(trace))
    rows = trace.read_text(encoding="utf-8").splitlines()
    changed = json.loads(rows[10])
    changed["frame"] = 11
    rows[10] = json.dumps(changed)
    trace.write_text("\n".join(rows) + "\n", encoding="utf-8")
    value = json.loads(manifest.read_text(encoding="utf-8"))
    value["state_trace"]["sha256"] = _sha(trace)
    manifest.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="frame"):
        _audit(manifest, trace)

