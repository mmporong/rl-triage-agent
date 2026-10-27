from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import vision_pull_record_worker_v3 as W  # pyright: ignore[reportMissingImports]


def _write(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _actual_build_fixture(
    tmp_path: Path,
) -> tuple[dict[str, Any], dict[str, Path], Path, list[dict[str, Any]]]:
    output = tmp_path / "output"
    output.mkdir(parents=True)
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    video = output / "same_play.mp4"
    video.write_bytes(b"encoded video")
    video_sha = W.JOB._sha(video)
    pts = [index / W.RECORD.FPS for index in range(W.RECORD.FRAME_COUNT)]
    timing = {
        "source_sha256": video_sha,
        "frame_count": W.RECORD.FRAME_COUNT,
        "relative_actual_pts_s": pts,
    }
    encoder = {"tool": "fixture", "frames_written": W.RECORD.FRAME_COUNT}
    bound = [
        {
            "frame_index": index,
            "capture_ordinal": index,
            "simulation_step": 100 + index,
            "actual_timestamp_s": pts[index],
            "render_timing": {
                "read_times": {
                    "durationDenominator": 18446744073709551615,
                    "durationNumerator": 0,
                    "externalTimeOfSimNs": -1,
                    "frameNumber": 46 + index,
                    "rationalTimeOfSimDenominator": 50,
                    "rationalTimeOfSimNumerator": 46 + index,
                    "sampleTimeOffsetInSimFrames": 18446744073709551615,
                    "simulationTime": pts[index],
                    "simulationTimeMonotonic": pts[index],
                    "swhFrameNumber": 0,
                    "systemTime": 1793090838.0 + pts[index],
                },
                "reference_time": {
                    "referenceTimeDenominator": 50,
                    "referenceTimeNumerator": 46 + index,
                },
            },
            "state": {
                "simulation_time_s": pts[index],
                "command": [1.0, 0.0, 0.0],
                "motion": [1.0, 0.0, 0.0],
                "pose": {"x": pts[index]},
            },
        }
        for index in range(W.RECORD.FRAME_COUNT)
    ]
    validated = {
        "output_directory": output,
        "job_path": _write(tmp_path / "job.json", {}),
        "checkpoint_sha256": "c" * 64,
        "checkpoint_path": checkpoint,
        "job_sha256": "j" * 64,
        "job": {
            "job_id": "fixture-job",
            "checkpoint": {"logical_id": "source-s2220-none"},
        },
    }
    built = W.RECORD.build_trace_and_v01_specs(
        validated, video, bound, encoder, timing, output / "evidence"
    )
    assert set(built) == W.PRODUCED_KEYS
    return validated, built, video, bound


def _write_pull_receipts(validated: dict[str, Any], bound: list[dict[str, Any]]) -> None:
    stage = Path(validated["job_path"]).parent
    previous = W.PREVIOUS.PREVIOUS
    for ordinal, record in enumerate(bound):
        npz, raw_path, receipt_path = previous._receipt_paths(stage, ordinal)
        npz.write_bytes(f"npz-{ordinal}".encode())
        timing = record["render_timing"]
        raw_times = {**timing["read_times"], "execOut": 1}
        _write(raw_path, {
            "schema": W.CAPTURE.RAW_SCHEMA,
            "capture_ordinal": ordinal,
            "npz": previous._descriptor(npz),
            "raw_fields": list(W.CAPTURE.ANNOTATORS),
            "read_times": raw_times,
            "reference_time": timing["reference_time"],
            "reserved_marker_count": 0,
        })
        physics = {"simulation_time_s": record["state"]["simulation_time_s"]}
        _write(receipt_path, {
            "schema": W.CAPTURE.SCHEMA,
            "capture_ordinal": ordinal,
            "annotator_order": list(W.CAPTURE.ANNOTATORS),
            "complete_capture": True,
            "reserved_marker_count": 0,
            "atomicity_claimed": False,
            "capture_approved": False,
            "used_for_decision": False,
            "raw_npz": previous._descriptor(npz),
            "raw_receipt": previous._descriptor(raw_path),
            "app_update_counter": {
                "source": "omni.kit.app.IApp.get_update_number",
                "fail_closed": True,
                "pull_window_updates": 0,
                "render_window_updates": 1,
                "after_each_pull": [1000 + ordinal] * 5,
            },
            "step": {"call_count": 1, "delta_time_s": 0.0},
            "physics_before": physics,
            "physics_after": physics,
            "frame_number": 46 + ordinal,
            "rational_time": [46 + ordinal, 50],
        })


def test_actual_build_five_outputs_normalize_to_strict_eight_and_report(
    tmp_path: Path,
) -> None:
    validated, built, video, bound = _actual_build_fixture(tmp_path)
    artifacts = W._normalized_artifacts(validated, built, video)
    assert set(artifacts) == W.PREVIOUS.RECORD_VERIFY.ARTIFACT_KEYS
    trace = W.RECORD.read_object(artifacts["raw_trace"])
    assert W.RECORD.read_object(artifacts["encoder_receipt"]) == trace["encoder_receipt"]
    assert W.RECORD.read_object(artifacts["timing_receipt"]) == trace["timing_receipt"]
    assert W.PREVIOUS.RECORD_VERIFY._record_artifacts(validated, artifacts)

    _write_pull_receipts(validated, bound)
    manifest_value = W._manifest(validated, artifacts)
    assert manifest_value["frame_count"] == W.RECORD.FRAME_COUNT
    assert manifest_value["frames"][-1]["ordinal"] == W.RECORD.FRAME_COUNT - 1
    manifest = _write(
        Path(validated["output_directory"]) / W.MANIFEST_NAME,
        manifest_value,
    )
    report = W._report(validated, artifacts, manifest)
    assert report["schema"] == W.SCHEMA
    assert report["technical_recording_complete"] is True
    assert set(report["artifacts"]) == W.PREVIOUS.RECORD_VERIFY.ARTIFACT_KEYS


def test_normalizer_rejects_video_content_mismatch_and_outside_artifact(
    tmp_path: Path,
) -> None:
    validated, built, video, _bound = _actual_build_fixture(tmp_path)
    video.write_bytes(b"changed after trace")
    with pytest.raises(ValueError, match="primary video 결합"):
        W._normalized_artifacts(validated, built, video)

    validated, built, video, _bound = _actual_build_fixture(tmp_path / "outside-case")
    outside = _write(tmp_path / "outside.json", {})
    built["measurement_1"] = outside
    with pytest.raises(ValueError, match="output 경계"):
        W._normalized_artifacts(validated, built, video)


def test_normalizer_rejects_symlink_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    validated, built, video, _bound = _actual_build_fixture(tmp_path)
    target = built["measurement_1"]
    link = target.with_name("linked-measurement.json")
    link.write_bytes(target.read_bytes())
    built["measurement_1"] = link
    regular = W.JOB._regular

    def reject_link(path: Path, anchor: str) -> Path:
        if Path(path) == link:
            raise ValueError(f"{anchor}는 단일 regular file이어야 한다")
        return regular(path, anchor)

    monkeypatch.setattr(W.JOB, "_regular", reject_link)
    with pytest.raises(ValueError, match="regular file"):
        W._normalized_artifacts(validated, built, video)


def test_preexisting_timing_receipt_is_never_deleted(tmp_path: Path) -> None:
    validated, built, video, _bound = _actual_build_fixture(tmp_path)
    timing = built["trace"].parent / "timing_receipt.json"
    timing.write_bytes(b"pre-existing evidence")
    before = timing.read_bytes(), W.JOB._sha(timing)
    with pytest.raises(FileExistsError, match="이미 있다"):
        W._normalized_artifacts(validated, built, video)
    assert (timing.read_bytes(), W.JOB._sha(timing)) == before
    assert not (built["trace"].parent / "encoder_receipt.json").exists()
