from __future__ import annotations

import importlib
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import test_vision_frame_sync_v1 as FRAME_FIXTURE
import test_vision_writer_payload_v1 as PAYLOAD_FIXTURE

S = importlib.import_module("vision_atomic_smoke_v1")
P = importlib.import_module("vision_writer_payload_v1")
F = FRAME_FIXTURE.S
R = PAYLOAD_FIXTURE.R


@pytest.fixture(autouse=True)
def _fixed_test_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(R, "WIDTH", 10)
    monkeypatch.setattr(R, "HEIGHT", 10)


def _fixture(tmp_path: Path):
    job = tmp_path / "job.json"
    job.write_text("{}\n", encoding="utf-8")
    sync = F.AtomicFrameWriter(F.SMOKE_MODE, 4, render_product_ref="RenderProduct_0")
    state = {"root_position_world_m": [0.0, 0.0, 0.3], "step": 7}
    rows = []
    for ordinal, label in enumerate(S.ORDER):
        array, semantic = FRAME_FIXTURE._marker_arrays(ordinal, background=ordinal * 3)
        raw = PAYLOAD_FIXTURE._payload(frame=55 + ordinal)
        annotations = raw["annotators"]
        annotations["rgb"]["RenderProduct_0"] = np.concatenate(
            [array, np.full((10, 10, 1), 255, dtype=np.uint8)], axis=2
        )
        annotations["semantic_segmentation"]["RenderProduct_0"] = {
            "data": semantic["mask"].astype(np.uint32),
            "info": {"idToLabels": semantic["idToLabels"]},
        }
        annotations["CameraParams"]["RenderProduct_0"].update(
            {
                "cameraViewTransform": FRAME_FIXTURE._view(
                    "B" if ordinal == 2 else "A"
                ).reshape(16),
                "cameraProjection": np.eye(4).reshape(16),
                "renderProductResolution": np.array([10, 10]),
            }
        )
        annotations["resolution"]["RenderProduct_0"] = [10, 10]
        decoded = P.decode_writer_payload(raw)
        sync.arm(ordinal, ordinal, state, 2.0)
        sync.before_step(ordinal)
        sync.callback(P.frame_payload(decoded))
        bundle = sync.close(ordinal, state, 2.0)
        P.write_atomic_journal(job, ordinal, F.SMOKE_MODE, decoded, bundle)
        offset = [-3.0, 3.0, 2.0] if ordinal == 2 else [3.0, 3.0, 2.0]
        target = R.fixed_profile()["camera"]["lookat_offset_m"]
        rows.append(
            {
                "label": label,
                "rgb": decoded["rgb"].tobytes(order="C"),
                "state_before": deepcopy(state),
                "state_after": deepcopy(state),
                "simulation_time_before_s": 2.0,
                "simulation_time_after_s": 2.0,
                "camera": {
                    "eye_world_m": [
                        offset[i] + state["root_position_world_m"][i] for i in range(3)
                    ],
                    "lookat_world_m": [
                        target[i] + state["root_position_world_m"][i] for i in range(3)
                    ],
                },
                "atomic_bundle": bundle,
                **bundle["timing"],
            }
        )
    sync.finish()
    return job, rows


def _collect(job, rows):
    iterator = iter(rows)
    return S.collect(lambda _label, _offset: next(iterator), job)


def test_collect_and_reopen_actual_atomic_journals_allow_background_noise(tmp_path):
    job, rows = _fixture(tmp_path)
    captures, report = _collect(job, rows)
    assert captures == rows and report["status"] == "pass"
    assert rows[1]["rgb"] != rows[3]["rgb"]
    report_path = tmp_path / "smoke.json"
    R.write_new(report_path, report)
    paths = {}
    for ordinal in (1, 2, 3):
        path = tmp_path / f"audit{ordinal}.rgb"
        path.write_bytes(rows[ordinal]["rgb"])
        paths[f"audit_rgb_{ordinal}"] = path
    assert S.verify_report(report_path, job, paths) == report
    paths["audit_rgb_3"].write_bytes(rows[1]["rgb"])
    with pytest.raises(ValueError, match="audit RGB"):
        S.verify_report(report_path, job, paths)


@pytest.mark.parametrize("field", ["rgb", "timing", "state", "camera", "bundle"])
def test_collector_rejects_raw_row_or_registration_mismatch(tmp_path, field):
    job, rows = _fixture(tmp_path)
    if field == "rgb":
        rows[3]["rgb"] = rows[1]["rgb"]
    elif field == "timing":
        rows[3]["read_times"]["frameNumber"] += 1
    elif field == "state":
        rows[3]["state_after"]["step"] += 1
    elif field == "camera":
        rows[2]["camera"]["eye_world_m"][0] = 3.0
    else:
        rows[3]["atomic_bundle"]["adapter_callback_ref"] = "invented"
    with pytest.raises(ValueError):
        _collect(job, rows)


def test_smoke_report_verifier_rejects_a_rewritten_registration_claim(tmp_path):
    job, rows = _fixture(tmp_path)
    _, report = _collect(job, rows)
    report["universal_zero_lag_claim"] = True
    path = tmp_path / "smoke.json"
    R.write_new(path, report)
    with pytest.raises(ValueError, match="report"):
        S.verify_report(path, job, {})


def test_selection_sources_reopens_atomic_report_without_old_pixel_equality(tmp_path, monkeypatch):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evals"))
    sources = importlib.import_module("vision_selection_sources_v1")
    job, rows = _fixture(tmp_path)
    _, report = _collect(job, rows)
    path = tmp_path / "smoke.json"
    R.write_new(path, report)
    artifacts = {"render_sync_smoke": path}
    for ordinal in (1, 2, 3):
        raw = tmp_path / f"audit{ordinal}.rgb"
        raw.write_bytes(rows[ordinal]["rgb"])
        artifacts[f"audit_rgb_{ordinal}"] = raw
    capture = {"artifact_paths": artifacts, "paths": {"job": job},
               "result": {"video_sha256": None, "primary_artifact_sha256": R.sha256_file(path)}}
    monkeypatch.setattr(sources.RECORD, "WIDTH", 640)
    monkeypatch.setattr(sources.RECORD, "HEIGHT", 480)
    with pytest.raises(ValueError, match="resolution"):
        sources._verify_smoke(capture)
    monkeypatch.setattr(sources.RECORD, "WIDTH", 10)
    monkeypatch.setattr(sources.RECORD, "HEIGHT", 10)
    assert sources._verify_smoke(capture) == report
    capture["result"]["primary_artifact_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="primary"):
        sources._verify_smoke(capture)
