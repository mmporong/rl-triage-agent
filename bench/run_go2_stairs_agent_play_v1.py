# pyright: reportMissingImports=false
"""에이전트 계단 적응 전후를 동일0.25m/s·30초 과제에서 관찰한다."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import run_go2_stairs_down_long_v1 as LONG
import run_go2_stairs_down_long_v2 as ENCODER

ROOT = Path(__file__).resolve().parents[1]
sha, read = LONG.sha, LONG.read
install_exact_frame_encoder = ENCODER.install_exact_frame_encoder
SCHEMA = "stairs_agent_play_spec_v1"
MANIFEST_SCHEMA = "stairs_agent_play_manifest_v1"


def load_base(repo: Path | None = None):
    root = repo or ROOT
    path = root / "bench/run_stairs_playback_v2.py"
    if sha(path, lf=True) != LONG.BASE_SHA:
        raise ValueError("기존 계단 runner SHA 불일치")
    spec = importlib.util.spec_from_file_location("agent_stairs_base", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    base.__dict__.update(FRAMES=1500, FPS=50, STEP_HEIGHT_M=0.05, STEP_WIDTH_M=0.3,
                         COMMAND_X_M_S=0.25, SCHEMA=SCHEMA, MANIFEST_SCHEMA=MANIFEST_SCHEMA)
    original, tensor_row = base.configure_env, base._tensor_row
    base.__dict__["configure_env"] = lambda cfg, s: LONG.configure_long(original, cfg, s)

    def trace_row(env, frame, terminated):
        row = tensor_row(env, frame, terminated)
        row.update(body_up_world_z=LONG.body_up_z(row["root_quat_w"]),
                   episode_length_step=int(env.episode_length_buf[0].item()),
                   command_body=env.command_manager.get_command("base_velocity")[0].detach().cpu().tolist())
        return row

    base.__dict__["_tensor_row"] = trace_row
    return base


def make_spec(previous: dict, checkpoint: Path, seed: int) -> dict:
    import copy
    value = copy.deepcopy(previous)
    if value["capture"]["command_x_m_s"] != 0.25 or value["direction"] != "down":
        raise ValueError("원 실패 과제 불일치")
    value.update(schema=SCHEMA, seed=seed, clip_id=f"go2-stairs-agent-s{seed}",
                 checkpoint={"path": str(checkpoint.resolve()), "sha256": sha(checkpoint)})
    value["capture"]["frames"] = 1500
    return value


def verify_record(out: Path, spec: dict) -> dict:
    import json
    import sys
    sys.path.insert(0, str(ROOT / "evals"))
    import stairs_agent_repair_v1 as CONTRACT
    m = read(out / "manifest.json")
    if (m["schema"] != MANIFEST_SCHEMA or m["source"] != spec["source"]
            or m["checkpoint_sha256"] != spec["checkpoint"]["sha256"]
            or m["clip_id"] != spec["clip_id"]
            or m["render_sync"]["physics_state_unchanged"] is not True):
        raise ValueError("실제 manifest와 등록 spec 불일치")
    terrain = m["terrain"]
    extension = terrain.get("observation_extension", {})
    if (terrain.get("terrain_id") != "pyramid_stairs" or terrain.get("step_height_m") != 0.05
            or terrain.get("step_width_m") != 0.3
            or terrain.get("command") != {"lin_vel_x": 0.25, "lin_vel_y": 0.0, "ang_vel_z": 0.0}
            or extension.get("seconds") != 30 or extension.get("automatic_reset_disabled") is not True):
        raise ValueError("같은 하행 계단·명령·장시간 관찰 계약이 아니다")
    trace = out / "state-trace.jsonl"
    if sha(trace) != m["state_trace"]["sha256"]:
        raise ValueError("trace SHA 불일치")
    rows = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    result = CONTRACT.assess(rows)
    video_rel = Path(m["video"]["path"])
    if video_rel.is_absolute() or ".." in video_rel.parts:
        raise ValueError("영상 경로 계약 위반")
    video = out / video_rel
    e = read(video.with_suffix(video.suffix+".exact-frames.json"))
    if (sha(video) != m["video"]["sha256"] or e["video_sha256"] != sha(video)
            or e["frames_written"] != 1500 or e["frames_received"] != 1500
            or [v["index"] for v in e["frame_sequence"]] != list(range(1500))
            or e["fps"] != 50):
        raise ValueError("원본 프레임/영상 결합 위반")
    return {"assessment": result, "manifest_sha256": sha(out/"manifest.json"),
            "trace_sha256": sha(trace), "video_sha256": sha(video),
            "encoder_sha256": sha(video.with_suffix(video.suffix+".exact-frames.json")),
            "frames": 1500, "fps": 50, "checkpoint_sha256": spec["checkpoint"]["sha256"]}
