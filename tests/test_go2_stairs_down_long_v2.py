from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
SPEC = importlib.util.spec_from_file_location("long_encoder_v2", ROOT / "bench/run_go2_stairs_down_long_v2.py")
assert SPEC and SPEC.loader
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def test_new_encoder_consumes_1500_unique_inputs_once_without_using_625_guard(tmp_path, monkeypatch):
    Recorder: Any = type("Recorder", (), {})
    rendering = NS(RecordVideo=Recorder)
    monkeypatch.setitem(sys.modules, "gymnasium.wrappers.rendering", rendering)
    written = []

    class Writer:
        def __init__(self, filename, size, fps, *, codec):
            self.path = Path(filename)
            assert size == (2, 2) and fps == 50 and codec == "libx264"

        def write_frame(self, frame):
            written.append(frame.copy())

        def close(self):
            self.path.write_bytes(b"unit fixture; not a video")

    frames = [np.full((2, 2, 3), i % 256, dtype=np.uint8) for i in range(1500)]
    recorder = NS(recording=True, recorded_frames=frames, frames_per_sec=50,
                  video_folder=str(tmp_path), _video_name="clip", gc_trigger=None)
    M.install_exact_frame_encoder(Writer)
    Recorder.stop_recording(recorder)
    assert len(written) == 1500
    assert all(np.array_equal(a, b) for a, b in zip(frames, written))
    receipt = M.read(tmp_path / "clip.mp4.exact-frames.json")
    assert receipt["frames_received"] == receipt["frames_written"] == 1500
    assert receipt["schema"] == M.ENCODER_SCHEMA and len(receipt["frame_sequence"]) == 1500
    assert recorder.recording is False and recorder.recorded_frames == []


def test_new_encoder_rejects_old_625_frame_recording(monkeypatch):
    Recorder: Any = type("Recorder", (), {})
    monkeypatch.setitem(sys.modules, "gymnasium.wrappers.rendering", NS(RecordVideo=Recorder))
    M.install_exact_frame_encoder()
    with pytest.raises(ValueError, match="1500"):
        Recorder.stop_recording(NS(recording=True, recorded_frames=[None] * 625, frames_per_sec=50))
