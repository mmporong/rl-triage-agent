from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evals"))
import vision_transferred_input_acceptance_v1 as M


@pytest.mark.parametrize("receipt", [
    {"all_match": False, "files_verified": 3865},
    {"all_match": True, "files_verified": 3864},
    {"all_match": 1, "files_verified": 3865},
    {"all_match": True, "files_verified": True},
])
def test_unverified_transfers_are_rejected(receipt: dict) -> None:
    with pytest.raises(ValueError, match="게이트"):
        M.transfer_gate(receipt)


def test_original_usage_flags_cannot_be_promoted() -> None:
    original = {key: False for key in M.FLAGS}
    assert M.original_flags(original) == original
    for key in M.FLAGS:
        changed = dict(original, **{key: True})
        with pytest.raises(ValueError, match="승인 경계"):
            M.original_flags(changed)
    assert all(value is False for value in original.values())


def fixture(tmp_path: Path) -> tuple[dict, Path, Path, Path]:
    video, checkpoint, measurement = (tmp_path / name for name in ("video.mp4", "model.pt", "measurement.json"))
    video.write_bytes(b"video input")
    checkpoint.write_bytes(b"checkpoint input")
    frames = [float(i) for i in range(1, 7)]
    play = {"task": "task", "camera": "camera", "command_name": "base_velocity",
            "command_frame": "body", "motion_frame": "body", "settings_sha256": "a" * 64}
    vector = {key: 0.0 for key in ("linear_x_mps", "linear_y_mps", "yaw_rate_rps")}
    data = {"schema": M.V01.MEASUREMENT_SCHEMA, "play_run_id": "same-play",
            "video_sha256": M.V01.sha256(video), "checkpoint_sha256": M.V01.sha256(checkpoint),
            "command_name": "base_velocity", "coordinate_frames": {"command": "body", "motion": "body"},
            "samples": [{"timestamp_s": t, "command": vector, "motion": vector,
                         "pose": {"base_height_m": 0.3, "up_axis_z": 1.0}} for t in frames]}
    measurement.write_text(json.dumps(data), encoding="utf-8")
    spec = {"schema": M.V01.PREPARE_SCHEMA, "play_run_id": "same-play", "source_lineage_id": "lineage",
            "parent_source_lineage_id": None, "play": play,
            "frames": [{"timestamp_s": t, "play_run_id": "same-play", "video_sha256": M.V01.sha256(video)}
                       for t in frames]}
    for key in ("video", "checkpoint", "measurement"):
        spec[key] = {"path": "/remote/" + key, "logical_id": key, "play_run_id": "same-play"}
    return spec, video, checkpoint, measurement


def test_translation_preserves_original_and_only_rebinds_paths(tmp_path: Path) -> None:
    spec, video, checkpoint, measurement = fixture(tmp_path)
    before = deepcopy(spec)
    translated = M.local_spec(spec, video, checkpoint, measurement)
    assert spec == before
    assert translated["checkpoint"]["path"] == str(checkpoint.resolve())
    for key in ("video", "checkpoint", "measurement"):
        translated[key]["path"] = before[key]["path"]
    assert translated == before


@pytest.mark.parametrize("field,value", [
    ("play_run_id", "other-policy"), ("checkpoint_sha256", "b" * 64), ("video_sha256", "c" * 64),
])
def test_cross_play_measurements_are_rejected(tmp_path: Path, field: str, value: str) -> None:
    spec, video, checkpoint, measurement = fixture(tmp_path)
    data = json.loads(measurement.read_text())
    data[field] = value
    measurement.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="같은 Play"):
        M.local_spec(spec, video, checkpoint, measurement)


def test_measurement_time_cannot_be_shifted_to_a_nearby_frame(tmp_path: Path) -> None:
    spec, video, checkpoint, measurement = fixture(tmp_path)
    data = json.loads(measurement.read_text())
    data["samples"][0]["timestamp_s"] += 0.02
    measurement.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="시각"):
        M.local_spec(spec, video, checkpoint, measurement)


def test_output_outside_owned_tree_is_rejected_before_io(tmp_path: Path) -> None:
    output = tmp_path / "outside_unwritten"
    with pytest.raises(ValueError):
        M.prepare_pair(tmp_path / "missing", tmp_path / "missing", output,
                       owned_root=tmp_path / "owned")
    assert not output.exists()
