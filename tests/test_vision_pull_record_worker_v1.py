from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import vision_pull_record_worker_v1 as W  # pyright: ignore[reportMissingImports]


def _write(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _validated(tmp_path: Path) -> dict[str, Any]:
    job = _write(tmp_path / "job.json", {})
    return {
        "job_path": job, "job_sha256": W.JOB._sha(job),
        "output_directory": tmp_path / "output", "checkpoint_sha256": "c" * 64,
        "job": {"checkpoint": {"logical_id": "source-s2220-none"}},
    }


def _capture_files(value: dict[str, Any], count: int, *, marker: int = 0) -> dict[str, Path]:
    stage = Path(value["job_path"]).parent
    output = Path(value["output_directory"])
    output.mkdir()
    records = []
    for ordinal in range(count):
        npz, raw_path, receipt_path = W._receipt_paths(stage, ordinal)
        npz.write_bytes(f"npz-{ordinal}".encode())
        read = {"frameNumber": ordinal + 10, "simulationTime": ordinal * 0.02}
        reference = {"referenceTimeNumerator": ordinal + 1, "referenceTimeDenominator": 50}
        _write(raw_path, {
            "schema": W.CAPTURE.RAW_SCHEMA, "capture_ordinal": ordinal,
            "npz": W._descriptor(npz), "raw_fields": list(W.CAPTURE.ANNOTATORS),
            "read_times": read, "reference_time": reference,
            "reserved_marker_count": marker, "atomicity_claimed": False,
            "capture_approved": False, "used_for_decision": False,
        })
        _write(receipt_path, {
            "schema": W.CAPTURE.SCHEMA, "capture_ordinal": ordinal,
            "annotator_order": list(W.CAPTURE.ANNOTATORS), "complete_capture": True,
            "reserved_marker_count": marker,
            "step": {"call_count": 1, "delta_time_s": 0.0},
            "app_update_counter": {
                "source": "omni.kit.app.IApp.get_update_number", "fail_closed": True,
                "render_window_updates": 1, "pull_window_updates": 0,
                "after_each_pull": [101 + ordinal] * 5,
            },
            "physics_before": {"simulation_time_s": ordinal * 0.02},
            "physics_after": {"simulation_time_s": ordinal * 0.02},
            "frame_number": ordinal + 10, "rational_time": [ordinal + 1, 50],
            "raw_npz": W._descriptor(npz), "raw_receipt": W._descriptor(raw_path),
            "atomicity_claimed": False, "capture_approved": False, "used_for_decision": False,
        })
        records.append({
            "capture_ordinal": ordinal,
            "state": {"simulation_time_s": ordinal * 0.02},
            "render_timing": {"read_times": read, "reference_time": reference},
        })
    trace = _write(output / "raw_trace.json", {"records": records})
    video = output / "same_play.mp4"
    video.write_bytes(b"video")
    artifacts = {"raw_trace": trace, "video": video}
    for name in W.RECORD_VERIFY.ARTIFACT_KEYS - artifacts.keys():
        artifacts[name] = _write(output / f"{name}.json", {"name": name})
    return artifacts


def test_manifest_binds_five_actual_pulls_and_rejects_marker_residue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = _validated(tmp_path)
    monkeypatch.setattr(W.RECORD, "FRAME_COUNT", 2)
    artifacts = _capture_files(value, 2)
    manifest = W._manifest(value, artifacts)
    assert manifest["frame_count"] == 2
    assert manifest["raw_file_count"] == 4
    assert manifest["receipt_count"] == 2
    assert manifest["annotators"][-1] == "ReferenceTime"
    assert manifest["native_pull_window_updates"] == 0
    assert manifest["clean_reserved_marker_count"] == 0

    marker = _validated(tmp_path / "marker")
    marked = _capture_files(marker, 2, marker=1)
    with pytest.raises(ValueError, match="receipt"):
        W._manifest(marker, marked)


def test_run_policy_record_wires_factory_to_real_worker_and_seals_after_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = _validated(tmp_path)
    value["output_directory"].mkdir()
    factory, app = object(), object()
    artifacts = {key: tmp_path / f"{key}.bin" for key in W.RECORD_VERIFY.ARTIFACT_KEYS}
    for path in artifacts.values():
        path.write_bytes(b"x")
    artifacts["video"] = tmp_path / "video.bin"
    artifacts["video"].write_bytes(b"video")
    manifest = {"schema": W.MANIFEST_SCHEMA}
    monkeypatch.setattr(W, "_manifest", lambda *_args: manifest)
    monkeypatch.setattr(W, "_report", lambda *_args: {"technical_recording_complete": True})
    seen: list[Any] = []

    def run_real(validated: Any, actual_app: Any, **kwargs: Any) -> tuple[dict[str, Path], Path]:
        assert validated is value and actual_app is app
        assert kwargs == {"camera_capture_factory": factory}
        seen.append("record")
        return artifacts, artifacts["video"]

    result = W.run_policy_record(
        value, app, seal_report=lambda report: seen.append(dict(report)),
        run_real=run_real,
        capture_factory=lambda validated, actual_app: factory,
        prepare_app=lambda actual_app: {
            "before": {"orchestrator_enabled": False, "capture_on_play": True},
            "after": {"orchestrator_enabled": True, "capture_on_play": False},
        },
    )
    assert result["technical_recording_complete"] is True
    assert result["startup"]["after"] == {
        "orchestrator_enabled": True, "capture_on_play": False,
    }
    assert seen == ["record", result]
    assert json.loads((value["output_directory"] / W.MANIFEST_NAME).read_text()) == manifest


def test_cold_record_app_configures_real_startup_before_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    value = _validated(tmp_path)
    events: list[str] = []
    values = {W.STARTUP.ORCHESTRATOR_ENABLED: False, W.STARTUP.CAPTURE_ON_PLAY: True}

    class Settings:
        def get(self, key: str) -> Any:
            return values[key]

        def set_bool(self, key: str, item: bool) -> None:
            values[key] = item
            events.append("enable_orchestrator")

    def capture_on_play(item: bool) -> None:
        values[W.STARTUP.CAPTURE_ON_PLAY] = item
        events.append("disable_capture_on_play")

    app = SimpleNamespace(update=lambda: events.append("app_update"))
    rep = SimpleNamespace(orchestrator=SimpleNamespace(set_capture_on_play=capture_on_play))
    monkeypatch.setattr(W.SDK, "_sdk_enable_extension", lambda name: events.append(name) or True)
    monkeypatch.setattr(W.STARTUP, "_load_settings", lambda: Settings())
    monkeypatch.setattr(W.STARTUP, "_load_replicator", lambda: rep)

    def factory(*_args: Any) -> Any:
        assert values[W.STARTUP.ORCHESTRATOR_ENABLED] is True
        assert values[W.STARTUP.CAPTURE_ON_PLAY] is False
        events.append("factory_before_environment")
        raise RuntimeError("environment sentinel")

    sealed: list[dict[str, Any]] = []
    with pytest.raises(W.PullRecordFailure, match="environment sentinel"):
        W.run_policy_record(value, app, seal_report=lambda report: sealed.append(dict(report)), capture_factory=factory)
    assert events == [
        "isaacsim.core.nodes", "app_update", "enable_orchestrator",
        "disable_capture_on_play", "factory_before_environment",
    ]
    assert sealed[0]["startup"]["after"]["orchestrator_enabled"] is True
    assert sealed[0]["technical_recording_complete"] is False


@pytest.mark.parametrize("failure", ["extension", "readback"])
def test_cold_start_failure_seals_before_factory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    from types import SimpleNamespace

    value = _validated(tmp_path)
    events: list[str] = []
    monkeypatch.setattr(W.SDK, "_sdk_enable_extension", lambda _name: failure != "extension")
    monkeypatch.setattr(W.STARTUP, "_load_settings", lambda: SimpleNamespace(
        get=lambda _key: False, set_bool=lambda *_args: None,
    ))
    monkeypatch.setattr(W.STARTUP, "_load_replicator", lambda: object())
    app = SimpleNamespace(update=lambda: events.append("app_update"))
    sealed: list[dict[str, Any]] = []
    with pytest.raises(W.PullRecordFailure):
        W.run_policy_record(
            value, app, seal_report=lambda report: sealed.append(dict(report)),
            capture_factory=lambda *_args: events.append("factory"),
        )
    assert "factory" not in events
    assert sealed[0]["technical_recording_complete"] is False
    assert sealed[0]["error_type"] == "builtins.RuntimeError"
    assert ("활성화 실패" if failure == "extension" else "readback") in sealed[0]["error_message"]


@pytest.mark.parametrize("after", [
    {"orchestrator_enabled": False, "capture_on_play": False},
    {"orchestrator_enabled": True, "capture_on_play": True},
    {"orchestrator_enabled": 1, "capture_on_play": 0},
])
def test_startup_report_requires_observed_boolean_settings(after: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="startup readback"):
        W.verify_startup({"before": {}, "after": after})
