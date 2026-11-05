from pathlib import Path
import sys
from types import SimpleNamespace as NS

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
import run_go2_stairs_agent_play_hq_v1 as HQ


def test_only_viewer_changes_and_records_real_settings():
    physics = {"command": .25, "seed": 200, "step_height": .05, "fps": 50}
    cfg = NS(viewer=NS(origin_type="asset_root", asset_name="robot", env_index=0,
                       resolution=(1280, 720), eye=(6, 6, 3), lookat=(0, 0, .5)),
             physics=physics)
    actual = HQ.configure_viewer(cfg)
    assert cfg.physics is physics and cfg.physics == {"command": .25, "seed": 200, "step_height": .05, "fps": 50}
    assert actual["resolution"] == [2560, 1440]
    assert actual["eye"] == [3.5, 6.0, 2.2]
    assert actual["lookat"] == [0, 0, .5]


def test_encoder_rejects_upscaled_source_and_fixes_quality_parameters(monkeypatch):
    captured = []
    monkeypatch.setattr(HQ.BASE, "install_exact_frame_encoder", captured.append)
    calls = []
    HQ.install_exact_frame_encoder(lambda *a, **kw: calls.append((a, kw)))
    factory = captured[0]
    with pytest.raises(ValueError, match="1440p"):
        factory("out.mp4", (1280, 720), 50, codec="libx264")
    assert not calls
    factory("out.mp4", (2560, 1440), 50, codec="libx264")
    assert calls == [(("out.mp4", (2560, 1440), 50),
                     {"codec": "libx264", "preset": "medium",
                      "ffmpeg_params": ["-crf", "16", "-movflags", "+faststart"]})]
