from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
P = importlib.import_module("vision_writer_payload_v1")
F = importlib.import_module("vision_frame_sync_v1")
R = importlib.import_module("record_vision_evidence_v1")


@pytest.fixture(autouse=True)
def _fixed_test_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(R, "WIDTH", 4)
    monkeypatch.setattr(R, "HEIGHT", 3)


def _times(frame: int = 55, *, swh: int = 0) -> dict[str, Any]:
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
        "swhFrameNumber": swh,
        "systemTime": 1000.0,
    }
    assert set(values) == set(R.ISAAC_READ_TIMES_FIELDS)
    return values


def _payload(
    *, frame: int = 55, top_swh: int = 77, rp: str = "RenderProduct_0"
) -> dict[str, Any]:
    rgba = np.zeros((3, 4, 4), dtype=np.uint8)
    rgba[:, :, :3] = (5, 6, 7)
    rgba[:, :, 3] = 255
    mask = np.ones((3, 4), dtype=np.uint32)
    camera = {
        "cameraViewTransform": np.arange(16, dtype=np.float64),
        "cameraProjection": np.arange(16, dtype=np.float32) + 1,
        "renderProductResolution": np.array([4, 3], dtype=np.int32),
        "cameraFocalLength": np.float32(24.0),
    }
    annotations = {
        "camera": {rp: R.fixed_profile()["camera"]["camera_prim_path"]},
        "resolution": {rp: [4, 3]},
        "rgb": {rp: rgba},
        "semantic_segmentation": {
            rp: {"data": mask, "info": {"idToLabels": {1: {"class": "robot/body"}}}}
        },
        "CameraParams": {rp: camera},
        "IsaacReadTimes": {rp: _times(frame)},
    }
    return {
        "swhFrameNumber": top_swh,
        "reference_time": (frame, 30),
        "distribution_outputs": {},
        "trigger_outputs": {},
        "named_outputs": {},
        "annotators": annotations,
    }


def _bundle(decoded: dict[str, Any]) -> dict[str, Any]:
    writer = F.AtomicFrameWriter(
        F.RECORD_MODE, 1, render_product_ref=decoded["render_product_ref"]
    )
    writer.arm(0, None, {"step": 1}, 2.0)
    writer.before_step(0)
    writer.callback(P.frame_payload(decoded))
    return writer.close(0, {"step": 1}, 2.0)


def test_decoder_copies_rgba_mask_flat16_and_preserves_top_swh_metadata() -> None:
    raw = _payload(top_swh=91)
    decoded = P.decode_writer_payload(raw)
    raw["annotators"]["rgb"]["RenderProduct_0"][:] = 255
    raw["annotators"]["semantic_segmentation"]["RenderProduct_0"]["data"][:] = 8
    raw["annotators"]["CameraParams"]["RenderProduct_0"]["cameraViewTransform"][:] = -1

    assert decoded["rgb"].shape == (3, 4, 3)
    assert decoded["rgb"][0, 0].tolist() == [5, 6, 7]
    assert decoded["semantic"]["mask"][0, 0] == 1
    assert decoded["camera"]["view_matrix"][0] == [0.0, 1.0, 2.0, 3.0]
    assert decoded["writer_swh_frame_number"] == 91
    assert decoded["read_times"]["swhFrameNumber"] == 0
    assert decoded["writer_reference_time"] == [55, 30]


def test_decoder_preserves_unsupported_exec_port_as_typed_metadata() -> None:
    class ExecutionPort:
        pass

    raw = _payload()
    raw["annotators"]["IsaacReadTimes"]["RenderProduct_0"]["execOut"] = ExecutionPort()
    decoded = P.decode_writer_payload(raw)
    assert "execOut" not in decoded["read_times"]
    assert decoded["read_times_raw"]["fields"]["execOut"] == {
        "type": f"{ExecutionPort.__module__}.{ExecutionPort.__qualname__}",
        "encoding": "unsupported",
        "value": None,
    }


@pytest.mark.parametrize("mutation", ["top", "rp", "rgba", "mask", "camera"])
def test_decoder_rejects_malformed_or_cross_render_product_payload(
    mutation: str,
) -> None:
    raw = _payload()
    if mutation == "top":
        raw["unknown"] = 1
    elif mutation == "rp":
        raw["annotators"]["rgb"] = {
            "other": raw["annotators"]["rgb"]["RenderProduct_0"]
        }
    elif mutation == "rgba":
        raw["annotators"]["rgb"]["RenderProduct_0"] = np.zeros(
            (3, 4, 3), dtype=np.uint8
        )
    elif mutation == "mask":
        raw["annotators"]["semantic_segmentation"]["RenderProduct_0"]["data"] = (
            np.zeros((3, 4), dtype=np.int32)
        )
    else:
        raw["annotators"]["CameraParams"]["RenderProduct_0"]["cameraProjection"] = (
            np.eye(4)
        )
    with pytest.raises((TypeError, ValueError)):
        P.decode_writer_payload(raw)


def test_custom_writer_stores_graph_callback_error_for_host_rethrow() -> None:
    class Writer:
        pass

    class Registry:
        @staticmethod
        def get_annotator(name: str, **kwargs: Any) -> str:
            return f"{name}:{kwargs}"

    rep = SimpleNamespace(Writer=Writer, AnnotatorRegistry=Registry)

    received: list[dict[str, Any]] = []
    writer = P.create_payload_writer(rep, received.append)
    writer.write({})
    assert writer.callback_count == 1 and received == []
    with pytest.raises(ValueError, match="top-level"):
        writer.raise_callback_error()


def test_atomic_journal_round_trip_binds_bundle_and_raw_arrays(tmp_path: Path) -> None:
    job = tmp_path / "job.json"
    job.write_text("{}\n", encoding="utf-8")
    decoded = P.decode_writer_payload(_payload())
    bundle = _bundle(decoded)
    json_path, npz_path = P.write_atomic_journal(job, 0, F.RECORD_MODE, decoded, bundle)
    journal, loaded_bundle, raw = P.load_capture_journal(job, 0)

    assert json_path.name == "job.atomic-000000.json"
    assert npz_path.name == "job.atomic-000000.npz"
    assert journal["schema"] == P.SCHEMA
    assert loaded_bundle == bundle
    assert raw[0] == decoded["rgb"].tobytes(order="C")
    assert raw[1] == decoded["semantic"]["mask"].tobytes(order="C")
    assert journal["writer_payload"]["swh_frame_number"] == 77
    with pytest.raises(FileExistsError):
        P.write_atomic_journal(job, 0, F.RECORD_MODE, decoded, bundle)


def test_atomic_journal_rejects_npz_or_bundle_tamper(tmp_path: Path) -> None:
    job = tmp_path / "job.json"
    job.write_text("{}\n", encoding="utf-8")
    decoded = P.decode_writer_payload(_payload())
    P.write_atomic_journal(job, 0, F.RECORD_MODE, decoded, _bundle(decoded))
    npz = tmp_path / "job.atomic-000000.npz"
    original = npz.read_bytes()
    npz.write_bytes(original + b"tamper")
    with pytest.raises(ValueError, match="SHA"):
        P.load_capture_journal(job, 0)

    npz.write_bytes(original)
    journal_path = tmp_path / "job.atomic-000000.json"
    body = R.read_object(journal_path)
    body["bundle"]["rgb"]["sha256"] = "0" * 64
    journal_path.write_text(
        __import__("json").dumps(body, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="RGB|rgb|결합"):
        P.load_capture_journal(job, 0)


@pytest.mark.parametrize(
    "mutation",
    ["callback", "timing", "reference", "camera", "resolution", "camera_path"],
)
def test_atomic_journal_rebinds_writer_metadata_to_bundle(
    tmp_path: Path, mutation: str
) -> None:
    job = tmp_path / "job.json"
    job.write_text("{}\n", encoding="utf-8")
    decoded = P.decode_writer_payload(_payload())
    P.write_atomic_journal(job, 0, F.RECORD_MODE, decoded, _bundle(decoded))
    path = tmp_path / "job.atomic-000000.json"
    body = R.read_object(path)
    writer = body["writer_payload"]
    if mutation == "callback":
        writer["callback_ref"] = "other-callback"
    elif mutation == "timing":
        writer["isaac_read_times"]["fields"]["frameNumber"]["value"] += 1
    elif mutation == "reference":
        writer["writer_reference_time"][0] += 1
    elif mutation == "camera":
        writer["camera_params"]["cameraViewTransform"]["values"][0] += 1
    elif mutation == "camera_path":
        writer["camera"] = "/World/WrongCamera"
    else:
        writer["resolution"] = [99, 99]
    path.write_text(json.dumps(body, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="identity|timing|CameraParams|camera path|resolution"):
        P.load_capture_journal(job, 0)


def test_atomic_journal_accepts_unsupported_execution_port_as_non_timing_metadata(
    tmp_path: Path,
) -> None:
    class ExecutionPort:
        pass

    job = tmp_path / "job.json"
    job.write_text("{}\n", encoding="utf-8")
    raw = _payload()
    raw["annotators"]["IsaacReadTimes"]["RenderProduct_0"]["execOut"] = ExecutionPort()
    decoded = P.decode_writer_payload(raw)
    bundle = _bundle(decoded)
    P.write_atomic_journal(job, 0, F.RECORD_MODE, decoded, bundle)
    assert P.load_capture_journal(job, 0)[1] == bundle
