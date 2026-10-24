from __future__ import annotations

import base64
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from evals import vision_evidence_contract_v1 as V


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_inputs(
    tmp_path: Path,
    *,
    measurement: bool = True,
    rate: int = 2,
    timestamps: list[float] | None = None,
    video_filter: str | None = None,
) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    video = tmp_path / "fault_seed_999.mp4"
    checkpoint = tmp_path / "model_fault_seed_999.pt"
    video_command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
        f"testsrc=duration=6:size=64x64:rate={rate}",
    ]
    if video_filter is not None:
        video_command.extend(["-vf", video_filter, "-fps_mode", "vfr"])
    video_command.extend(["-c:v", "mpeg4", "-y", str(video)])
    subprocess.run(
        video_command,
        check=True,
        capture_output=True,
    )
    checkpoint.write_bytes(b"checkpoint bytes")
    frames = []
    timestamps = timestamps or [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    for timestamp in timestamps:
        frames.append({
            "timestamp_s": timestamp,
            "play_run_id": "private-fault-seed-999",
            "video_sha256": _sha(video.read_bytes()),
        })
    trace_path = tmp_path / "fault_seed_999_trace.json"
    trace = {
        "schema": V.MEASUREMENT_SCHEMA,
        "play_run_id": "private-fault-seed-999",
        "video_sha256": _sha(video.read_bytes()),
        "checkpoint_sha256": _sha(checkpoint.read_bytes()),
        "command_name": "base_velocity",
        "coordinate_frames": {"command": "body", "motion": "world"},
        "samples": [
            {
                "timestamp_s": timestamp,
                "command": {"linear_x_mps": 1.0, "linear_y_mps": 0.0, "yaw_rate_rps": None},
                "motion": {"linear_x_mps": 0.1 * index, "linear_y_mps": 0.0, "yaw_rate_rps": None},
                "pose": {"base_height_m": 0.32, "up_axis_z": 0.99},
            }
            for index, timestamp in enumerate(timestamps, 1)
        ],
    }
    trace_path.write_text(json.dumps(trace), encoding="utf-8")
    spec = {
        "schema": V.PREPARE_SCHEMA,
        "play_run_id": "private-fault-seed-999",
        "source_lineage_id": "lineage-999",
        "parent_source_lineage_id": "lineage-998",
        "video": {
            "path": str(video), "logical_id": "private/fault_seed_999.mp4",
            "play_run_id": "private-fault-seed-999",
        },
        "checkpoint": {
            "path": str(checkpoint), "logical_id": "private/model_fault_seed_999.pt",
            "play_run_id": "private-fault-seed-999",
        },
        "frames": frames,
        "measurement": None if not measurement else {
            "path": str(trace_path), "logical_id": "private/fault_trace.json",
            "play_run_id": "private-fault-seed-999",
        },
        "play": {
            "task": "Isaac-Velocity-Flat-Unitree-Go2-v0",
            "camera": "follow_camera",
            "command_name": "base_velocity",
            "command_frame": "body",
            "motion_frame": "world",
            "settings_sha256": "a" * 64,
        },
    }
    spec_path = tmp_path / "prepare.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    return spec_path


def _prepared(tmp_path: Path, *, measurement: bool = True) -> Path:
    return V.prepare(_write_inputs(tmp_path, measurement=measurement), tmp_path / "prepared")


def test_prepare_binds_six_ordered_frames_and_same_play_measurements(tmp_path):
    contract_path = _prepared(tmp_path)
    contract = V.verify(contract_path)
    assert [row["frame_id"] for row in contract["frames"]] == [f"f{i:02d}" for i in range(1, 7)]
    assert [row["timestamp_s"] for row in contract["frames"]] == [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    assert contract["provenance"]["evidence_independence"] == {
        "visual_source_count": 1,
        "frame_samples_from_visual_source": 6,
        "measurement_source_count": 1,
        "model_interpretation_adds_independent_source": False,
    }
    assert contract["behavior_ground_truth"] is None
    assert contract["fixed_evaluation_used_as_behavior_ground_truth"] is False
    assert contract["provenance"]["source_lineage_id"] == "lineage-999"
    assert contract["provenance"]["parent_source_lineage_id"] == "lineage-998"
    assert contract["provenance"]["frame_extraction"]["tool"] == "ffmpeg"
    assert len(contract["provenance"]["frame_extraction"]["commands"]) == 6
    actual = [row["actual_timestamp_s"] for row in contract["provenance"]["frame_extraction"]["commands"]]
    assert actual == [row["timestamp_s"] for row in contract["frames"]]
    assert actual == [row["timestamp_s"] for row in contract["synchronized_samples"]]
    assert not any(Path(value["path"]).is_absolute() for value in (
        contract["provenance"]["video"],
        contract["provenance"]["checkpoint"],
        contract["provenance"]["measurement"],
    ))


def test_model_view_contains_actual_frame_bytes_but_no_private_labels_or_names(tmp_path):
    contract_path = _prepared(tmp_path)
    view = V.public_model_view(contract_path)
    encoded = json.dumps(view, ensure_ascii=False)
    for forbidden in ("private-fault", "seed", "model_fault", "fixed_eval", "healthy", "unhealthy", "arm"):
        assert forbidden not in encoded.lower()
    content = view["messages"][1]["content"]
    images = [item["image_url"]["url"] for item in content if item["type"] == "image_url"]
    assert len(images) == 6
    assert base64.b64decode(images[0].split(",", 1)[1]) == (
        contract_path.parent / "artifacts/frames/frame_01.jpg"
    ).read_bytes()
    assert view["evidence_accounting"] == {
        "visual_source_count": 1,
        "frame_sample_count": 6,
        "measurement_source_count": 1,
        "model_interpretation_source_count": 0,
    }
    numeric = json.loads(content[1]["text"])
    assert numeric["coordinate_frames"] == {"command": "body", "motion": "world"}
    assert numeric["samples"][0]["motion"]["linear_x_mps"] == 0.1


def test_missing_measurements_stay_explicit_null_and_are_never_zero_filled(tmp_path):
    contract_path = _prepared(tmp_path, measurement=False)
    contract = V.verify(contract_path)
    assert contract["provenance"]["measurement"] is None
    assert contract["provenance"]["evidence_independence"]["measurement_source_count"] == 0
    values = [value for sample in contract["synchronized_samples"] for group in ("command", "motion", "pose")
              for value in sample[group].values()]
    assert values and all(value is None for value in values)
    view = V.public_model_view(contract_path)
    assert view["evidence_accounting"]["measurement_source_count"] == 0


@pytest.mark.parametrize("target", ["video", "checkpoint", "frame", "measurement"])
def test_verify_rejects_changed_bound_artifact(tmp_path, target):
    contract_path = _prepared(tmp_path)
    root = contract_path.parent
    paths = {
        "video": root / "artifacts/source.video",
        "checkpoint": root / "artifacts/checkpoint.bin",
        "frame": root / "artifacts/frames/frame_03.jpg",
        "measurement": root / "artifacts/measurements/trace.json",
    }
    paths[target].write_bytes(b"changed")
    with pytest.raises(ValueError, match="결합|JPEG"):
        V.verify(contract_path)


def test_prepare_rejects_time_reversal_and_measurement_from_another_run(tmp_path):
    spec_path = _write_inputs(tmp_path)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec["frames"][3]["timestamp_s"] = 1.0
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    with pytest.raises(ValueError, match="오름차순"):
        V.prepare(spec_path, tmp_path / "bad-time")

    spec_path = _write_inputs(tmp_path / "other")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    trace_path = Path(spec["measurement"]["path"])
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    trace["play_run_id"] = "another-play"
    trace_path.write_text(json.dumps(trace), encoding="utf-8")
    with pytest.raises(ValueError, match="같은 Play"):
        V.prepare(spec_path, tmp_path / "bad-run")


def test_prepare_rejects_frame_declared_from_another_play(tmp_path):
    spec_path = _write_inputs(tmp_path)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec["frames"][2]["play_run_id"] = "another-play"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    with pytest.raises(ValueError, match="같은 Play 영상"):
        V.prepare(spec_path, tmp_path / "bad-frame-run")

    spec_path = _write_inputs(tmp_path / "path-injection")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec["frames"][0]["path"] = str(tmp_path / "caller.jpg")
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    with pytest.raises(ValueError, match="필드"):
        V.prepare(spec_path, tmp_path / "bad-frame-path")


def test_prepare_rejects_checkpoint_or_video_binding_from_another_play(tmp_path):
    spec_path = _write_inputs(tmp_path)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    trace_path = Path(spec["measurement"]["path"])
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    trace["checkpoint_sha256"] = "b" * 64
    trace_path.write_text(json.dumps(trace), encoding="utf-8")
    with pytest.raises(ValueError, match="같은 Play"):
        V.prepare(spec_path, tmp_path / "bad-checkpoint")

    spec_path = _write_inputs(tmp_path / "declared")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec["checkpoint"]["play_run_id"] = "another-play"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    with pytest.raises(ValueError, match="checkpoint.*같은 Play"):
        V.prepare(spec_path, tmp_path / "bad-checkpoint-run")


def test_prepare_rejects_user_absolute_logical_id_from_public_contract(tmp_path):
    spec_path = _write_inputs(tmp_path)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec["video"]["logical_id"] = "C:/Users/example/private.mp4"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    with pytest.raises(ValueError, match="절대"):
        V.prepare(spec_path, tmp_path / "absolute-logical-id")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("command_name", "healthy_seed_arm"),
        ("command_frame", "healthy_seed_arm"),
        ("motion_frame", "healthy_seed_arm"),
    ],
)
def test_model_exposed_play_fields_are_closed_enums(tmp_path, field, value):
    spec_path = _write_inputs(tmp_path)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec["play"][field] = value
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    with pytest.raises(ValueError, match=f"play.{field}.*등록 범위"):
        V.prepare(spec_path, tmp_path / f"bad-{field}")


@pytest.mark.parametrize("field", ["source_lineage_id", "parent_source_lineage_id"])
def test_lineage_and_parent_lineage_tampering_breaks_contract_id(tmp_path, field):
    contract_path = _prepared(tmp_path)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["provenance"][field] = f"different-{field}"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    with pytest.raises(ValueError, match="contract_id"):
        V.verify(contract_path)


def test_marker_only_jpeg_is_not_decodable(tmp_path):
    marker = tmp_path / "marker-only.jpg"
    marker.write_bytes(b"\xff\xd8not-an-image\xff\xd9")
    with pytest.raises(ValueError, match="디코딩"):
        V._validate_jpeg(marker, "marker")


def test_truncated_jpeg_scan_is_rejected_by_full_decode(tmp_path):
    contract_path = _prepared(tmp_path)
    source = contract_path.parent / "artifacts/frames/frame_01.jpg"
    payload = source.read_bytes()
    marker = payload.index(b"\xff\xda")
    header_end = marker + 2 + int.from_bytes(payload[marker + 2:marker + 4], "big")
    truncated = tmp_path / "truncated-scan.jpg"
    truncated.write_bytes(payload[:header_end] + b"\xff\xd9")
    metadata = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name", "-of", "json", str(truncated)],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert metadata.returncode == 0 and json.loads(metadata.stdout)["streams"][0]["codec_name"] == "mjpeg"
    with pytest.raises(ValueError, match="전체 scan"):
        V._validate_jpeg(truncated, "truncated")


def test_low_fps_requested_times_outside_fixed_pts_tolerance_are_rejected(tmp_path):
    spec = _write_inputs(
        tmp_path,
        measurement=False,
        rate=1,
        timestamps=[0.49, 0.50, 0.51, 2.0, 3.0, 4.0],
    )
    with pytest.raises(ValueError, match="실제 PTS 차이"):
        V.prepare(spec, tmp_path / "low-fps")


def test_vfr_requested_time_outside_fixed_pts_tolerance_is_rejected(tmp_path):
    spec = _write_inputs(
        tmp_path,
        measurement=False,
        rate=4,
        timestamps=[0.15, 0.25, 0.5625, 1.0, 1.5625, 2.25],
        video_filter="setpts=N*N/(16*TB)",
    )
    with pytest.raises(ValueError, match="실제 PTS 차이"):
        V.prepare(spec, tmp_path / "vfr")


def test_measurement_samples_must_use_resolved_actual_pts(tmp_path):
    requested = [0.51, 1.51, 2.51, 3.51, 4.51, 5.51]
    spec_path = _write_inputs(tmp_path, rate=10, timestamps=requested)
    with pytest.raises(ValueError, match="measurement 시각.*frame 시각"):
        V.prepare(spec_path, tmp_path / "measurement-request-time")

    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    trace_path = Path(spec["measurement"]["path"])
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    actual = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    for sample, timestamp in zip(trace["samples"], actual, strict=True):
        sample["timestamp_s"] = timestamp
    trace_path.write_text(json.dumps(trace), encoding="utf-8")
    contract_path = V.prepare(spec_path, tmp_path / "measurement-actual-time")
    contract = V.verify(contract_path)
    assert [row["timestamp_s"] for row in contract["frames"]] == actual
    assert [row["timestamp_s"] for row in contract["synchronized_samples"]] == actual


def test_verify_rejects_contract_frame_reordering_even_when_files_exist(tmp_path):
    contract_path = _prepared(tmp_path)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["frames"][0], contract["frames"][1] = contract["frames"][1], contract["frames"][0]
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    with pytest.raises(ValueError, match="순서"):
        V.verify(contract_path)


def test_verify_rejects_boolean_substituted_for_numeric_sample(tmp_path):
    contract_path = _prepared(tmp_path)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    assert contract["synchronized_samples"][0]["command"]["linear_x_mps"] == 1.0
    contract["synchronized_samples"][0]["command"]["linear_x_mps"] = True
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    with pytest.raises(ValueError, match="동기화 sample"):
        V.verify(contract_path)


def test_cli_prepare_verify_and_model_view_are_cpu_only_and_write_once(tmp_path):
    spec = _write_inputs(tmp_path)
    output = tmp_path / "cli-prepared"
    module = "evals.vision_evidence_contract_v1"
    prepared = subprocess.run(
        [sys.executable, "-m", module, "prepare", "--spec", str(spec), "--output", str(output)],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert json.loads(prepared.stdout)["outcome"] == "prepared"
    verified = subprocess.run(
        [sys.executable, "-m", module, "verify", "--contract", str(output / "contract.json")],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert json.loads(verified.stdout)["outcome"] == "valid"
    model_view = tmp_path / "model-view.json"
    subprocess.run(
        [sys.executable, "-m", module, "model-view", "--contract", str(output / "contract.json"),
         "--output", str(model_view)],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert json.loads(model_view.read_text(encoding="utf-8"))["schema"] == V.MODEL_VIEW_SCHEMA
    with pytest.raises(FileExistsError):
        V.write_new(model_view, {})
