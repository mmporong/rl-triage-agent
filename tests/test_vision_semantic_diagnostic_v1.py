from __future__ import annotations

import importlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
D = importlib.import_module("vision_semantic_diagnostic_v1")
R = importlib.import_module("record_vision_evidence_v1")


@pytest.fixture(autouse=True)
def _fixed_test_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(R, "WIDTH", 4)
    monkeypatch.setattr(R, "HEIGHT", 3)


def _times(frame: int = 55) -> dict[str, Any]:
    values: dict[str, Any] = {
        "durationDenominator": 60,
        "durationNumerator": 1,
        "externalTimeOfSimNs": 2_000_000_000,
        "frameNumber": frame,
        "rationalTimeOfSimDenominator": 30,
        "rationalTimeOfSimNumerator": frame,
        "sampleTimeOffsetInSimFrames": 0,
        "simulationTime": 2.0,
        "simulationTimeMonotonic": 2.0,
        "swhFrameNumber": 0,
        "systemTime": 1000.0,
    }
    assert set(values) == set(R.ISAAC_READ_TIMES_FIELDS)
    return values


def _payload(*, empty: bool = False, rp: str = "RenderProduct_0") -> dict[str, Any]:
    semantic = (
        np.array([], dtype=np.float64) if empty else np.ones((3, 4), dtype=np.uint32)
    )
    return {
        "swhFrameNumber": 77,
        "reference_time": (55, 30),
        "distribution_outputs": {},
        "trigger_outputs": {},
        "named_outputs": {},
        "annotators": {
            "camera": {rp: R.fixed_profile()["camera"]["camera_prim_path"]},
            "resolution": {rp: np.array([4, 3], dtype=np.int32)},
            "rgb": {rp: {"data": np.zeros((3, 4, 4), dtype=np.uint8)}},
            "semantic_segmentation": {
                rp: {
                    "data": semantic,
                    "idToLabels": {"1": {"class": "private-label"}},
                }
            },
            "CameraParams": {
                rp: {
                    "cameraViewTransform": np.arange(16, dtype=np.float64),
                    "cameraProjection": np.arange(16, dtype=np.float32) + 1,
                    "renderProductResolution": np.array([4, 3], dtype=np.int32),
                }
            },
            "IsaacReadTimes": {rp: _times()},
        },
    }


def _collect(sequence: list[dict[str, Any]]) -> dict[str, Any]:
    observer = D.DiagnosticObserver()

    def run_event(index: int) -> None:
        observer.observe(sequence[index])

    return D.collect_diagnostic(
        observer,
        run_event=run_event,
        physics_snapshot=lambda: {"step": 8, "time": 2.0},
        event_count=len(sequence),
    )


@pytest.mark.parametrize("order", [(True, False), (False, True)])
def test_empty_and_valid_callbacks_are_both_preserved_without_approval(
    order: tuple[bool, bool],
) -> None:
    report = _collect([_payload(empty=value) for value in order])

    assert [row["ordinal"] for row in report["callbacks"]] == [0, 1]
    assert [row["window"] for row in report["callbacks"]] == [
        "event_0_running",
        "event_1_running",
    ]
    classifications = [row["classification"] for row in report["callbacks"]]
    assert classifications == ["invalid" if value else "valid" for value in order]
    empty_row = report["callbacks"][order.index(True)]
    assert empty_row["strict_error_type"] == "builtins.ValueError"
    assert empty_row["counterfactual_schema_check"] == {
        "attempted": True,
        "passed": True,
        "reason": "semantic_float64_empty_replaced_for_schema_check",
        "error_type": None,
    }
    assert report["observations"] == {
        "valid_observed": True,
        "empty_observed": True,
        "mixed_valid_empty": True,
        "continued_empty": False,
        "invalid_observed": False,
        "trigger_zero": False,
        "physics_changed": False,
        "overflow": False,
        "source_error": False,
    }
    assert report["no_capture_approved"] is True
    assert report["events"][0]["pre_callback_count"] == 0
    assert report["events"][0]["post_callback_count"] == 1
    assert report["events"][1]["pre_callback_count"] == 1
    assert report["events"][1]["post_callback_count"] == 2


def test_callback_outside_events_is_preserved_without_event_causality_claim() -> None:
    observer = D.DiagnosticObserver()
    observer.observe(_payload())
    report = D.collect_diagnostic(
        observer,
        run_event=lambda _index: None,
        physics_snapshot=lambda: {"step": 3},
        event_count=1,
    )

    assert report["callbacks"][0]["window"] == "after_attach"
    assert report["events"][0]["pre_callback_count"] == 1
    assert report["events"][0]["post_callback_count"] == 1
    assert report["observations"]["trigger_zero"] is True
    assert report["callback_event_causality_proven"] is False
    assert report["step_return_is_complete_drain"] is False
    assert report["collection_scope"] == {
        "starts": "after_writer_attach_before_first_run_event",
        "ends": "immediately_after_last_run_event_return_and_post_physics_snapshot",
        "late_callbacks_after_returned_report_included": False,
        "detach_before_final_evidence_seal_required_by_runner": True,
        "all_future_callbacks_collected": False,
    }


def test_multiple_callbacks_in_one_event_are_preserved_by_ordinal() -> None:
    observer = D.DiagnosticObserver()

    def run_event(_index: int) -> None:
        observer.observe(_payload(empty=True))
        observer.observe(_payload())
        observer.observe(_payload(empty=True))

    report = D.collect_diagnostic(
        observer,
        run_event=run_event,
        physics_snapshot=lambda: {"step": 9},
        event_count=1,
    )
    assert report["callbacks_observed"] == 3
    assert report["callbacks_preserved"] == 3
    assert [row["ordinal"] for row in report["callbacks"]] == [0, 1, 2]
    assert {row["window"] for row in report["callbacks"]} == {"event_0_running"}


def test_malformed_schema_is_not_classified_as_semantic_empty() -> None:
    raw = _payload(empty=True)
    raw["annotators"].pop("camera")
    report = _collect([raw])

    callback = report["callbacks"][0]
    assert callback["classification"] == "invalid"
    assert callback["counterfactual_schema_check"]["attempted"] is True
    assert callback["counterfactual_schema_check"]["passed"] is False
    assert report["observations"]["empty_observed"] is False
    assert report["observations"]["invalid_observed"] is True


def test_mismatched_render_product_is_invalid_not_empty() -> None:
    raw = _payload(empty=True)
    semantic = raw["annotators"].pop("semantic_segmentation")
    raw["annotators"]["semantic_segmentation"] = {
        "Other": next(iter(semantic.values()))
    }
    report = _collect([raw])

    callback = report["callbacks"][0]
    assert callback["counterfactual_schema_check"]["attempted"] is False
    assert report["observations"]["empty_observed"] is False
    assert report["observations"]["invalid_observed"] is True


def test_physics_change_is_observed_with_full_snapshot_hashes() -> None:
    observer = D.DiagnosticObserver()
    state = {"step": 1, "nested": {"position": [0.0, 1.0]}}

    def run_event(_index: int) -> None:
        observer.observe(_payload())
        state["nested"]["position"][1] = 2.0

    report = D.collect_diagnostic(
        observer,
        run_event=run_event,
        physics_snapshot=lambda: state,
        event_count=1,
    )
    event = report["events"][0]
    assert event["physics_equal"] is False
    assert event["physics_before_sha256"] != event["physics_after_sha256"]
    assert report["observations"]["physics_changed"] is True


def test_callback_overflow_is_fail_closed_and_bounded() -> None:
    observer = D.DiagnosticObserver(max_callbacks=2)

    def run_event(_index: int) -> None:
        for _ in range(3):
            observer.observe(_payload())

    report = D.collect_diagnostic(
        observer,
        run_event=run_event,
        physics_snapshot=lambda: {"step": 1},
        event_count=2,
    )
    assert report["events_completed"] == 1
    assert report["events_recorded"] == 1
    assert report["events_attempted"] == 1
    assert report["callbacks_observed"] == 3
    assert report["callbacks_preserved"] == 2
    assert report["observations"]["overflow"] is True
    assert "overflow" in report["hold_observations"]
    assert report["no_capture_approved"] is True


def test_source_error_finishes_diagnostic_report() -> None:
    observer = D.DiagnosticObserver()

    def run_event(_index: int) -> None:
        observer.observe(_payload(empty=True))
        raise RuntimeError("injected diagnostic failure")

    with pytest.raises(D.DiagnosticFailure) as caught:
        D.collect_diagnostic(
            observer,
            run_event=run_event,
            physics_snapshot=lambda: {"step": 1},
            event_count=2,
        )
    report = caught.value.report
    assert isinstance(caught.value.primary, RuntimeError)
    assert caught.value.__cause__ is caught.value.primary
    assert report["events_recorded"] == 1
    assert report["events_attempted"] == 1
    assert report["events_completed"] == 0
    assert report["events"][0]["run_event_error_type"] == "builtins.RuntimeError"
    assert report["observations"]["source_error"] is True
    assert report["callbacks_preserved"] == 1
    assert report["no_capture_approved"] is True


def test_event_bound_is_checked_before_source_call() -> None:
    calls: list[int] = []
    with pytest.raises(ValueError, match="event 상한"):
        D.collect_diagnostic(
            D.DiagnosticObserver(),
            run_event=calls.append,
            physics_snapshot=dict,
            event_count=3,
        )
    assert calls == []

    report = D.collect_diagnostic(
        D.DiagnosticObserver(),
        run_event=calls.append,
        physics_snapshot=dict,
    )
    assert calls == [0, 1]
    assert report["events_completed"] == 2
    assert report["observations"]["trigger_zero"] is True


def test_writer_factory_uses_only_injected_sdk_surface() -> None:
    requested: list[tuple[str, dict[str, Any]]] = []

    class Registry:
        @staticmethod
        def get_annotator(name: str, **kwargs: Any) -> tuple[str, dict[str, Any]]:
            requested.append((name, kwargs))
            return name, kwargs

    class Writer:
        pass

    observer = D.DiagnosticObserver()
    rep = SimpleNamespace(Writer=Writer, AnnotatorRegistry=Registry)
    writer = D.create_diagnostic_writer(rep, observer)
    writer.write(_payload(empty=True))

    assert writer.data_structure == "annotator"
    assert [name for name, _kwargs in requested] == [
        "rgb",
        "semantic_segmentation",
        "CameraParams",
        "IsaacReadTimes",
    ]
    assert requested[1][1] == {"init_params": {"colorize": False}}
    assert observer.total_callbacks == 1


def test_report_write_is_canonical_write_once_and_namespaced(tmp_path: Path) -> None:
    report = _collect([_payload()])
    target = tmp_path / "one-view.semantic-diagnostic.json"
    written = D.write_report(target, report)

    assert written == target.resolve()
    assert stat.S_ISREG(target.stat().st_mode)
    assert target.stat().st_nlink == 1
    assert json.loads(target.read_text(encoding="utf-8")) == report
    with pytest.raises(ValueError, match="write-once"):
        D.write_report(target, report)
    with pytest.raises(ValueError, match="namespace"):
        D.write_report(tmp_path / "one-view.capture-000001.json", report)


def test_descriptor_and_raw_metadata_exclude_pixel_and_label_bodies() -> None:
    raw = _payload(empty=True)
    raw["annotators"]["IsaacReadTimes"]["RenderProduct_0"]["systemTime"] = float("nan")
    observer = D.DiagnosticObserver()
    observer.observe(raw)
    callback = observer.callbacks[0]
    encoded = json.dumps(callback, allow_nan=False)

    assert "private-label" not in encoded
    assert "semantic_float64_empty_replaced_for_schema_check" in encoded
    assert callback["raw_frame_time"]["swhFrameNumber"] == {
        "type": "builtins.int",
        "numeric": True,
        "value": 77,
    }
    assert callback["raw_frame_time"]["reference_time"] == [
        {"type": "builtins.int", "numeric": True, "value": 55},
        {"type": "builtins.int", "numeric": True, "value": 30},
    ]
    assert callback["raw_frame_time"]["isaac_read_times"]["systemTime"] == {
        "type": "builtins.float",
        "numeric": False,
    }


def test_classifier_failure_still_preserves_callback_evidence() -> None:
    raw = _payload(empty=True)
    raw["annotators"]["resolution"]["RenderProduct_0"] = np.asarray(4)
    observer = D.DiagnosticObserver()

    observer.observe(raw)

    assert observer.total_callbacks == 1
    assert len(observer.callbacks) == 1
    callback = observer.callbacks[0]
    assert callback["ordinal"] == 0
    assert callback["classification"] == "invalid"
    assert callback["classification_error_type"] == "builtins.TypeError"
    assert callback["descriptor"]["schema"] == "vision_writer_payload_descriptor_v1"


@pytest.mark.parametrize("cancel", [KeyboardInterrupt, SystemExit])
def test_process_cancellation_is_not_converted_to_diagnostic_report(
    cancel: type[BaseException],
) -> None:
    def run_event(_index: int) -> None:
        raise cancel()

    with pytest.raises(cancel):
        D.collect_diagnostic(
            D.DiagnosticObserver(),
            run_event=run_event,
            physics_snapshot=lambda: {"step": 1},
            event_count=1,
        )


def test_physics_before_failure_prevents_event_and_carries_partial_report() -> None:
    calls: list[int] = []

    with pytest.raises(D.DiagnosticFailure) as caught:
        D.collect_diagnostic(
            D.DiagnosticObserver(),
            run_event=calls.append,
            physics_snapshot=lambda: {"bad": float("nan")},
            event_count=1,
        )

    assert calls == []
    assert isinstance(caught.value.primary, ValueError)
    event = caught.value.report["events"][0]
    assert caught.value.report["events_recorded"] == 1
    assert caught.value.report["events_attempted"] == 0
    assert caught.value.report["events_completed"] == 0
    assert event["diagnostic_error_phase"] == "physics_before"
    assert event["pre_callback_count"] == event["post_callback_count"] == 0


def test_physics_after_failure_preserves_callback_in_carried_report() -> None:
    observer = D.DiagnosticObserver()
    snapshots = iter(({"step": 1}, {"bad": float("nan")}))

    def run_event(_index: int) -> None:
        observer.observe(_payload(empty=True))

    with pytest.raises(D.DiagnosticFailure) as caught:
        D.collect_diagnostic(
            observer,
            run_event=run_event,
            physics_snapshot=lambda: next(snapshots),
            event_count=1,
        )

    assert caught.value.report["callbacks_preserved"] == 1
    assert caught.value.report["events_recorded"] == 1
    assert caught.value.report["events_attempted"] == 1
    assert caught.value.report["events_completed"] == 0
    assert caught.value.report["callbacks"][0]["ordinal"] == 0
    assert caught.value.report["events"][0]["diagnostic_error_phase"] == (
        "physics_after"
    )


def test_observer_is_single_use_and_second_collection_calls_no_source() -> None:
    observer = D.DiagnosticObserver()
    calls: list[int] = []
    D.collect_diagnostic(
        observer,
        run_event=calls.append,
        physics_snapshot=dict,
        event_count=2,
    )

    with pytest.raises(ValueError, match="single-use"):
        D.collect_diagnostic(
            observer,
            run_event=calls.append,
            physics_snapshot=dict,
            event_count=1,
        )
    assert calls == [0, 1]


def test_returned_report_is_immutable_when_late_callback_arrives() -> None:
    observer = D.DiagnosticObserver()
    report = D.collect_diagnostic(
        observer,
        run_event=lambda _index: observer.observe(_payload()),
        physics_snapshot=dict,
        event_count=1,
    )

    observer.observe(_payload(empty=True))

    assert report["callbacks_observed"] == 1
    assert report["callbacks_preserved"] == 1
    assert len(report["callbacks"]) == 1
    assert observer.total_callbacks == 2
    assert observer.callbacks[1]["window"] == "collection_finished"
    assert report["collection_scope"][
        "late_callbacks_after_returned_report_included"
    ] is (False)


def test_raw_time_allows_only_numeric_whitelisted_values() -> None:
    raw = _payload()
    raw["reference_time"] = ("reference-secret", 30)
    read_times = raw["annotators"]["IsaacReadTimes"]["RenderProduct_0"]
    read_times["frameNumber"] = "frame-secret"
    read_times["unexpectedField"] = "unexpected-secret"
    observer = D.DiagnosticObserver()
    observer.observe(raw)
    metadata = observer.callbacks[0]["raw_frame_time"]
    encoded = json.dumps(metadata)

    assert "reference-secret" not in encoded
    assert "frame-secret" not in encoded
    assert "unexpected-secret" not in encoded
    assert metadata["reference_time"][0] == {
        "type": "builtins.str",
        "numeric": False,
    }
    assert metadata["isaac_read_times"]["frameNumber"] == {
        "type": "builtins.str",
        "numeric": False,
    }
    assert metadata["unexpected_isaac_read_times_fields"] == [
        {"name": "unexpectedField", "type": "builtins.str"}
    ]


def test_serialization_failure_creates_no_report_file(tmp_path: Path) -> None:
    target = tmp_path / "bad.semantic-diagnostic.json"
    report = _collect([_payload()])
    report["bad"] = float("nan")

    with pytest.raises(ValueError):
        D.write_report(target, report)
    assert not target.exists()


def test_report_rejects_symlink_ancestor(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    if os.name == "nt":
        script = tmp_path / "create-junction.ps1"
        script.write_text(
            """param(
    [Parameter(Mandatory=$true)][string]$JunctionPath,
    [Parameter(Mandatory=$true)][string]$TargetPath
)
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
New-Item -ItemType Junction -Path $JunctionPath -Target $TargetPath | Out-Null
""",
            encoding="utf-8-sig",
        )
        subprocess.run(
            [
                "powershell.exe",
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(script),
                "-JunctionPath",
                str(linked),
                "-TargetPath",
                str(real),
            ],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="strict",
        )
        cleanup = linked.rmdir
    else:
        linked.symlink_to(real, target_is_directory=True)
        cleanup = linked.unlink
    target = linked / "nested.semantic-diagnostic.json"

    try:
        with pytest.raises(ValueError, match="symlink"):
            D.write_report(target, _collect([_payload()]))
        assert not (real / target.name).exists()
    finally:
        cleanup()
