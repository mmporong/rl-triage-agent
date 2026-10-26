from __future__ import annotations

import copy
import importlib
import io
import json
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
import record_vision_evidence_v1 as R  # pyright: ignore[reportMissingImports]
import run_vision_prepared_record_v1 as W  # pyright: ignore[reportMissingImports]


def _write(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return path


def _descriptor(path: Path) -> dict[str, Any]:
    return {"file": path.name, "sha256": W._sha(path), "bytes": path.stat().st_size}


def _validated(tmp_path: Path) -> dict[str, Any]:
    job = _write(tmp_path / "job.json", {"job": True})
    return {
        "job_path": job,
        "job_sha256": W._sha(job),
        "output_directory": tmp_path / "output",
        "checkpoint_sha256": "c" * 64,
        "job": {
            "job_id": "prepared-record-fixture-checkpoint",
            "mode": W.JOB.MODE,
            "python": sys.executable,
            "isaaclab": str(tmp_path),
        },
        "resource": {"resource_id": "owned", "gpu_uuid": "GPU-owned"},
    }


def _artifacts(value: dict[str, Any]) -> dict[str, Any]:
    output = value["output_directory"]
    evidence = output / "evidence"
    video = output / "same_play.mp4"
    video.parent.mkdir(parents=True, exist_ok=True)
    video.write_bytes(b"video")
    points = [index / R.FPS for index in range(R.FRAME_COUNT)]
    records = [
        {
            "frame_index": index,
            "capture_ordinal": index,
            "simulation_step": 100 + index,
            "actual_timestamp_s": points[index],
        }
        for index in range(R.FRAME_COUNT)
    ]
    encoder = {"natural_exit": True, "frames_written": R.FRAME_COUNT}
    timing = {
        "source_sha256": W._sha(video),
        "frame_count": R.FRAME_COUNT,
        "relative_actual_pts_s": points,
    }
    trace = {
        "schema": R.TRACE_SCHEMA,
        "profile": R.fixed_profile(),
        "profile_sha256": R.digest(R.fixed_profile()),
        "checkpoint_sha256": value["checkpoint_sha256"],
        "video_sha256": W._sha(video),
        "frame_count": R.FRAME_COUNT,
        "records": records,
        "encoder_receipt": encoder,
        "timing_receipt": timing,
    }
    paths = {
        "video": video,
        "raw_trace": _write(evidence / "raw_trace.json", trace),
        "encoder_receipt": _write(evidence / "encoder_receipt.json", encoder),
        "timing_receipt": _write(evidence / "timing_receipt.json", timing),
    }
    for window, timestamps in enumerate(R.WINDOWS, 1):
        play_id = f"{value['job']['job_id']}-w{window}"
        measurement = {
            "schema": R.MEASUREMENT_SCHEMA,
            "play_run_id": play_id,
            "video_sha256": W._sha(video),
            "checkpoint_sha256": value["checkpoint_sha256"],
            "samples": [{"timestamp_s": timestamp} for timestamp in timestamps],
        }
        spec = {
            "schema": R.PREPARE_SCHEMA,
            "play_run_id": play_id,
            "source_lineage_id": f"{value['job']['job_id']}-same-play",
            "parent_source_lineage_id": None,
            "video": {"play_run_id": play_id},
            "checkpoint": {"play_run_id": play_id},
            "measurement": {"play_run_id": play_id},
            "frames": [{"timestamp_s": timestamp} for timestamp in timestamps],
        }
        paths[f"measurement_{window}"] = _write(
            evidence / f"window_{window}_measurements.json", measurement
        )
        paths[f"spec_{window}"] = _write(
            evidence / f"window_{window}_v01_spec.json", spec
        )
    return {key: _descriptor(path) for key, path in paths.items()}


def _envelope(value: dict[str, Any], *, sealed: bool = True) -> dict[str, Any]:
    artifacts = _artifacts(value)
    callbacks = [
        {
            "raw_index": index,
            "classification": "valid",
        }
        for index in range(R.FRAME_COUNT + 1)
    ]
    manifests = [
        {
            "ordinal": index,
            "frame_number": index + 11,
            "callback_ref": f"callback-{index}",
            "commit_receipt": {"ordinal": index},
        }
        for index in range(R.FRAME_COUNT)
    ]
    lifecycle: dict[str, Any] = dict.fromkeys(W.LIFECYCLE_FLAGS, True)
    lifecycle["arms_completed"] = R.FRAME_COUNT
    lifecycle["envelope_sealed"] = sealed
    lifecycle["errors"] = []
    return {
        "schema": "vision_prepared_record_sdk_v1",
        "prepared_record": {
            "schema": W.PREPARED.SCHEMA,
            "approved": True,
            "state": "sealed",
            "expected_count": R.FRAME_COUNT,
            "committed_count": R.FRAME_COUNT,
            "raw_callback_count": R.FRAME_COUNT + 1,
            "raw_callback_limit": R.FRAME_COUNT + 2,
            "callback_rows_dropped": 0,
            "preparation": {"bootstrap_empty_observed": False},
            "callbacks": callbacks,
            "manifests": manifests,
            "failure": None,
        },
        "artifacts": artifacts,
        "lifecycle": lifecycle,
        "renderer": {"profile": "fixed"},
        "setup_evidence": {"phase_order": ["prepare", "record"]},
        "record_result": {
            "frame_count": R.FRAME_COUNT,
            "fps": R.FPS,
            "video": artifacts["video"],
        },
        "approved": True,
    }


def test_accepts_625_frame_record_and_rejects_pts_or_label_leak(tmp_path: Path) -> None:
    value = _validated(tmp_path)
    value["output_directory"].mkdir()
    envelope = _envelope(value)
    assert W._successful_sdk_envelope(value, envelope, sealed=True) is True

    timing = value["output_directory"] / "evidence" / "timing_receipt.json"
    changed = json.loads(timing.read_text(encoding="utf-8"))
    changed["relative_actual_pts_s"][20] += 0.01
    timing.write_text(json.dumps(changed), encoding="utf-8")
    assert W._successful_sdk_envelope(value, envelope, sealed=True) is False

    value2 = _validated(tmp_path / "label")
    value2["output_directory"].mkdir()
    envelope2 = _envelope(value2)
    spec = value2["output_directory"] / "evidence" / "window_1_v01_spec.json"
    changed_spec = json.loads(spec.read_text(encoding="utf-8"))
    changed_spec["fixed_eval_label"] = "healthy"
    spec.write_text(json.dumps(changed_spec), encoding="utf-8")
    assert W._successful_sdk_envelope(value2, envelope2, sealed=True) is False


def test_sdk_child_writes_terminal_before_app_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = _validated(tmp_path)
    monkeypatch.setattr(W.JOB, "validate_job", lambda *_args, **_kwargs: value)

    class App:
        def close(self) -> None:
            assert (value["output_directory"] / "result.json").is_file()

    def run_sdk(_value: Any, _app: Any, *, seal_report: Any) -> dict[str, Any]:
        envelope = _envelope(value)
        sealed = copy.deepcopy(envelope)
        sealed["lifecycle"]["envelope_sealed"] = False
        seal_report(sealed)
        return envelope

    assert (
        W.sdk_child(value["job_path"], app_factory=lambda _: App(), run_sdk=run_sdk)
        == 0
    )
    result = W._read(value["output_directory"] / "result.json")
    assert result["record_completed"] is True and result["frame_count"] == 625


def test_actual_sdk_fixture_round_trips_through_parent_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sdk = importlib.import_module("vision_prepared_record_sdk_v1")
    fixture = importlib.import_module("test_vision_prepared_record_sdk_v1")
    value = _validated(tmp_path)
    monkeypatch.setattr(W.JOB, "validate_job", lambda *_args, **_kwargs: value)
    monkeypatch.setattr(W, "_record_artifacts", lambda *_args: True)
    monkeypatch.setattr(sdk.R, "WIDTH", fixture._Rep.width)
    monkeypatch.setattr(sdk.R, "HEIGHT", fixture._Rep.height)
    log: list[str] = []
    dependencies = fixture._dependencies(tmp_path, log)
    fixture_run_real = dependencies["run_real"]

    def run_real(*args: Any, **kwargs: Any) -> tuple[dict[str, Path], Path]:
        paths, video = fixture_run_real(*args, **kwargs)
        for window in (1, 2):
            for kind, expected in (
                ("measurement", f"window_{window}_measurements.json"),
                ("spec", f"window_{window}_v01_spec.json"),
            ):
                key = f"{kind}_{window}"
                renamed = paths[key].with_name(expected)
                paths[key].rename(renamed)
                paths[key] = renamed
        return paths, video

    dependencies["run_real"] = run_real

    class App(fixture._App):
        def close(self) -> None:
            super().close()
            assert (value["output_directory"] / "result.json").is_file()

    def run_sdk(validated: Any, app: Any, *, seal_report: Any) -> dict[str, Any]:
        return sdk.run_prepared_record(
            validated,
            app,
            seal_report=seal_report,
            dependencies=dependencies,
        )

    assert (
        W.sdk_child(value["job_path"], app_factory=lambda _: App(log), run_sdk=run_sdk)
        == 0
    )
    result = W._read(value["output_directory"] / "result.json")
    assert result["sdk"]["prepared_record"]["committed_count"] == R.FRAME_COUNT
    assert set(result["sdk"]["artifacts"]) == W.ARTIFACT_KEYS


def test_exit_zero_without_terminal_is_not_completed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = _validated(tmp_path)
    monkeypatch.setattr(W.JOB, "validate_job", lambda *_args, **_kwargs: value)
    monkeypatch.setattr(W.BASE, "_gpu_reference", lambda *_args: {"path": "gpu"})

    class Reader:
        def __init__(self) -> None:
            self.last_receipt = {"path": "gpu"}

        def __call__(self) -> list[int]:
            return []

    class Gate(io.BytesIO):
        def close(self) -> None:
            self.accepted = self.getvalue()
            super().close()

    class Process:
        pid = 123
        returncode: int | None = None

        def __init__(self) -> None:
            self.stdin = Gate()

        def wait(self, *, timeout: float) -> int:
            assert timeout <= 400 and self.stdin.accepted == b"GO\n"
            self.returncode = 0
            return 0

    receipt = W.run_job(
        value["job_path"],
        popen=lambda *_args, **_kwargs: Process(),
        gpu_reader=Reader(),
        identify=lambda pid: {"pid": pid, "start_ticks": 1},
        cleanup=lambda *_args: True,
    )
    assert receipt["exit_code"] == 0
    assert receipt["completed"] is False
    assert receipt["error_type"] in {"ValueError", "FileNotFoundError"}


def test_prepare_delegates_checkpoint_selection_to_job_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = tmp_path / "stage" / "job.json"
    calls: list[tuple[Any, ...]] = []
    monkeypatch.setattr(
        W.JOB,
        "create_job",
        lambda *args: calls.append(args) or expected,
    )
    actual = W.prepare_job(
        tmp_path / "protocol.json",
        tmp_path / "resource.json",
        "a" * 64,
        tmp_path / "IsaacLab",
        tmp_path / "python",
        tmp_path / "stage",
        "pilot-checkpoint",
    )
    assert actual == expected
    assert calls[0][-1] == "pilot-checkpoint"


@pytest.mark.skipif(W.os.name == "posix", reason="Windows 실제 실행 guard 회귀")
def test_default_launcher_cannot_bypass_owned_linux_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = _validated(tmp_path)
    monkeypatch.setattr(W.JOB, "validate_job", lambda *_args, **_kwargs: value)
    with pytest.raises(RuntimeError, match="소유 Linux process group"):
        W.run_job(value["job_path"])
