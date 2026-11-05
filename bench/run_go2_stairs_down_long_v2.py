"""30초 하행 관찰의 프레임 수에 맞춘 원본 프레임 인코더."""
from __future__ import annotations

import gc
import hashlib
import importlib
import json
from pathlib import Path
from typing import Any

import run_go2_stairs_down_long_v1 as V1

sha, read, make_spec = V1.sha, V1.read, V1.make_spec
load_base, compare_prefix = V1.load_base, V1.compare_prefix
CHECKPOINT_SHA = V1.CHECKPOINT_SHA
ENCODER_SCHEMA = "go2_long_exact_frame_encoder_v1"


def install_exact_frame_encoder(writer_factory: Any = None) -> None:
    rendering = importlib.import_module("gymnasium.wrappers.rendering")
    record_video = rendering.RecordVideo

    def stop_recording(self: Any) -> None:
        if not self.recording or len(self.recorded_frames) != V1.FRAMES or self.frames_per_sec != V1.FPS:
            raise ValueError("등록된 1500-frame/50fps encoder 입력이 아니다")
        import numpy as np

        factory = writer_factory
        if factory is None:
            factory = importlib.import_module("moviepy.video.io.ffmpeg_writer").FFMPEG_VideoWriter
        first = self.recorded_frames[0]
        height, width = first.shape[:2]
        video = Path(self.video_folder) / f"{self._video_name}.mp4"
        writer = factory(str(video), (width, height), self.frames_per_sec, codec="libx264")
        rows = []
        try:
            for i, frame in enumerate(self.recorded_frames):
                if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8
                        or frame.ndim != 3 or frame.shape[2] not in (3, 4)
                        or frame.shape[:2] != (height, width)):
                    raise ValueError("RGB 원본 프레임의 shape/dtype가 다르다")
                rgb = np.ascontiguousarray(frame[:, :, :3])
                digest = hashlib.sha256()
                digest.update(str(rgb.dtype).encode("ascii"))
                digest.update(json.dumps(list(rgb.shape), separators=(",", ":")).encode("ascii"))
                digest.update(rgb.tobytes(order="C"))
                writer.write_frame(rgb)
                rows.append({"index": i, "sha256": digest.hexdigest()})
        finally:
            writer.close()
        value = {"schema": ENCODER_SCHEMA, "frames_received": len(self.recorded_frames),
                 "frames_written": len(rows), "fps": self.frames_per_sec,
                 "width": width, "height": height, "frame_sequence": rows,
                 "video_file": video.name, "video_sha256": sha(video)}
        with video.with_suffix(video.suffix + ".exact-frames.json").open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        self.recorded_frames = []
        self.recording = False
        self._video_name = None
        if self.gc_trigger and self.gc_trigger(self.episode_id):
            gc.collect()

    record_video.stop_recording = stop_recording


def verify_record(out: Path, spec: dict) -> dict:
    result = V1.verify_record(out, spec)
    manifest = read(out / "manifest.json")
    video = out / manifest["video"]["path"]
    if read(video.with_suffix(video.suffix + ".exact-frames.json")).get("schema") != ENCODER_SCHEMA:
        raise ValueError("30초 encoder schema가 다르다")
    return result
