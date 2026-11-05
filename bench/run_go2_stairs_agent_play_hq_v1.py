"""같은 계단 정책을 1440p와 옆쪽 시점으로 재녹화한다. 학습·판정은 바꾸지 않는다."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import run_go2_stairs_agent_play_v1 as BASE

RESOLUTION = (2560, 1440)
EYE = (3.5, 6.0, 2.2)
LOOKAT = (0.0, 0.0, 0.5)
sha, read, make_spec = BASE.sha, BASE.read, BASE.make_spec


def configure_viewer(cfg: Any) -> dict:
    cfg.viewer.resolution = RESOLUTION
    cfg.viewer.eye = EYE
    cfg.viewer.lookat = LOOKAT
    return {"origin_type": cfg.viewer.origin_type, "asset_name": cfg.viewer.asset_name,
            "env_index": cfg.viewer.env_index, "eye": list(cfg.viewer.eye),
            "lookat": list(cfg.viewer.lookat), "resolution": list(cfg.viewer.resolution)}


def load_base(repo: Path | None = None):
    base = BASE.load_base(repo)
    original_configure, original_run = base.configure_env, base._run
    observed = {}

    def configure(cfg, spec):
        terrain = original_configure(cfg, spec)
        observed.update(configure_viewer(cfg))
        return terrain

    def run(spec, out, app):
        result = original_run(spec, out, app)
        if not observed:
            raise ValueError("실제 환경의 viewer 설정을 관측하지 못했다")
        result["camera"] = dict(observed)
        result["render_profile"] = {"schema": "stairs_hq_render_profile_v1",
                                    "resolution": list(RESOLUTION), "crf": 16,
                                    "preset": "medium", "new_training": 0, "model_api": 0}
        return result

    base.configure_env, base._run = configure, run
    return base


def install_exact_frame_encoder(writer_factory: Any = None) -> None:
    def high_quality_writer(path, size, fps, *, codec):
        if size != RESOLUTION or fps != 50 or codec != "libx264":
            raise ValueError("1440p·50fps 원본 렌더 계약 불일치")
        if writer_factory is not None:
            return writer_factory(path, size, fps, codec=codec, preset="medium",
                                  ffmpeg_params=["-crf", "16", "-movflags", "+faststart"])
        from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
        return FFMPEG_VideoWriter(path, size, fps, codec=codec, preset="medium",
                                 ffmpeg_params=["-crf", "16", "-movflags", "+faststart"])

    BASE.install_exact_frame_encoder(high_quality_writer)


def verify_record(out: Path, spec: dict) -> dict:
    verified = BASE.verify_record(out, spec)
    manifest = read(out / "manifest.json")
    camera = manifest["camera"]
    expected = {"origin_type": "asset_root", "asset_name": "robot", "env_index": 0,
                "eye": list(EYE), "lookat": list(LOOKAT), "resolution": list(RESOLUTION)}
    if camera != expected:
        raise ValueError("고화질 카메라 설정 불일치")
    encoder = read(out / "video/rl-video-step-0.mp4.exact-frames.json")
    if [encoder["width"], encoder["height"]] != list(RESOLUTION):
        raise ValueError("인코더가 실제 1440p RGB 프레임을 받지 않았다")
    profile = manifest.get("render_profile", {})
    if profile != {"schema": "stairs_hq_render_profile_v1", "resolution": list(RESOLUTION),
                   "crf": 16, "preset": "medium", "new_training": 0, "model_api": 0}:
        raise ValueError("고화질 렌더 영수증 불일치")
    return {**verified, "camera": camera, "render_profile": profile}
