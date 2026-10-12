"""기존 T1 정책 하나를 새 Isaac 영상·GIF로 녹화한다. 진단·회복 평가는 아니다."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess

import record_evidence_media as M
import run_runtime_readback_smoke as P

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "bench/protocols/checkpoint_preview_t1_v1.json"
METADATA = ROOT / "bench/private/e01_s2028.meta.json"
CODE = {"bench/record_checkpoint_preview.py", "bench/record_evidence_media.py",
        "bench/run_runtime_readback_smoke.py"}
PLAY = "scripts/reinforcement_learning/rsl_rl/play.py"
FIXED = {"schema": "checkpoint_preview_v1", "scope": "existing_policy_stock_playback",
         "task": M.EXPECTED_TASK, "seed": 42, "num_envs": 1, "video_length": 300,
         "kit_args": "--/app/vulkan=false --/app/window/hideUi=true",
         "overrides": ["env.viewer.eye=[3.0,3.0,2.0]", "env.viewer.lookat=[0.0,0.0,0.35]",
                       "env.viewer.origin_type=asset_root", "env.viewer.asset_name=robot"],
         "video_contract": M.EXPECTED_VIDEO_CONTRACT,
         "gpu_admission": M.EXPECTED_GPU_ADMISSION,
         "training_seed": 2028, "training_num_envs": 4096, "training_iterations": 300,
         "isaaclab_commit": "90b79bb2d44feb8d833f260f2bf37da3487180ba",
         "model_calls": 0, "diagnosis_or_recovery_evidence": False,
         "retraining": False, "gif_fps": 15, "gif_width": 720}


def inputs():
    metadata = json.loads(METADATA.read_bytes())
    if (metadata["seed"] != FIXED["training_seed"] or metadata["num_envs"] != 4096
            or metadata["max_iterations"] != 300 or metadata["train_exit_code"] != 0
            or metadata["task"] != "Isaac-Velocity-Flat-Unitree-Go2-v0"):
        raise ValueError("original policy source changed")
    run = Path(metadata["run_dir"])
    return run / "model_299.pt", run / "params"


def prepare(isaaclab):
    checkpoint, params = inputs()
    protocol = deepcopy(FIXED)
    protocol.update(source_files={name: M.sha256_lf_file(ROOT / name) for name in sorted(CODE)},
                    metadata_sha256=M.sha256_file(METADATA),
                    checkpoint_sha256=M.sha256_file(checkpoint),
                    params_env_sha256=M.sha256_file(params / "env.yaml"),
                    params_agent_sha256=M.sha256_file(params / "agent.yaml"),
                    play_py_sha256=M.sha256_file(isaaclab / PLAY))
    M._write_new_json(PROTOCOL, protocol)
    return {"status": "prepared_commit_before_recording", "protocol": str(PROTOCOL)}


def verify(isaaclab):
    protocol = json.loads(PROTOCOL.read_bytes())
    if any(protocol[key] != value for key, value in FIXED.items()):
        raise ValueError("recording contract changed")
    if protocol["source_files"] != {name: M.sha256_lf_file(ROOT / name) for name in sorted(CODE)}:
        raise ValueError("recorder source changed")
    checkpoint, params = inputs()
    for path, key in [(METADATA, "metadata_sha256"), (checkpoint, "checkpoint_sha256"),
                      (params / "env.yaml", "params_env_sha256"),
                      (params / "agent.yaml", "params_agent_sha256"), (isaaclab / PLAY, "play_py_sha256")]:
        M._check_hash(path, protocol[key], key)
    if P.S._git_head(isaaclab) != FIXED["isaaclab_commit"] or not P.S._git_tracked_clean(isaaclab):
        raise ValueError("SDK checkout changed")
    commit = P.verify_committed(ROOT, CODE | {PROTOCOL.relative_to(ROOT).as_posix()})
    return protocol, checkpoint, params, commit


def record(tag, isaaclab):
    tag = M.validate_tag(tag)
    protocol, checkpoint, params, commit = verify(isaaclab)
    private = ROOT / "bench/private/checkpoint_preview" / tag
    public = ROOT / "docs/media" / f"go2-policy-{tag}"
    if private.exists() or public.exists():
        raise FileExistsError("recording tag is already used")
    before = M.query_gpu()
    if before["busy"]:
        return {"status": "gpu_busy_no_launch", "model_calls": 0, "gpu_runs": 0}
    private.mkdir(parents=True, exist_ok=False)
    M._write_new_json(private / "gpu_before.json", before)
    clip = {"id": "policy", "checkpoint": "model_299.pt", "overrides": protocol["overrides"],
            **{key: protocol[key] for key in ("checkpoint_sha256", "params_env_sha256", "params_agent_sha256")}}
    result = M._record_clip(protocol, {"clip": clip, "checkpoint": checkpoint, "params_dir": params},
                            isaaclab, private, 180)
    after = M.query_gpu()
    M._write_new_json(private / "gpu_after.json", after)
    if after["busy"]:
        raise RuntimeError("GPU post-recording admission failed")
    verify(isaaclab)
    public.mkdir(parents=True, exist_ok=False)
    video, gif, poster = public / "simulation.mp4", public / "simulation.gif", public / "poster.png"
    shutil.copy2(result["video"], video)
    common = {"check": True, "timeout": 60, "creationflags": M.CREATE_NO_WINDOW,
              "stdout": subprocess.DEVNULL, "stderr": subprocess.PIPE}
    subprocess.run(["ffmpeg", "-nostdin", "-n", "-i", str(video), "-vf",
                    "fps=15,scale=720:-2:flags=lanczos,split[a][b];[a]palettegen=max_colors=128:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=3",
                    "-loop", "0", str(gif)], **common)
    subprocess.run(["ffmpeg", "-nostdin", "-n", "-ss", "2", "-i", str(video),
                    "-frames:v", "1", str(poster)], **common)
    manifest = {"schema": "checkpoint_preview_artifact_v1", "scope": FIXED["scope"],
                "source_commit_sha": commit, "protocol_sha256": M.sha256_file(PROTOCOL),
                "checkpoint_sha256": protocol["checkpoint_sha256"],
                "training_seed": 2028, "play_seed": 42, "num_envs": 1,
                "task": protocol["task"], "overrides": protocol["overrides"],
                "native_video": result["probe"], "playback_wall_s": result["elapsed_s"],
                "process_cleanup_verified": result["own_pid_cleanup"],
                "artifacts": {path.name: {"sha256": M.sha256_file(path), "bytes": path.stat().st_size}
                              for path in (video, gif, poster)},
                "model_calls": 0, "gpu_exclusive": False, "performance_measurement": False,
                "diagnosis_or_recovery_evidence": False, "retraining": False,
                "interpretation": "Existing frozen T1 policy in the stock Play environment. New capture; no AI-selected measurement, diagnosis, correction or recovery is shown."}
    M._write_new_json(public / "manifest.json", manifest)
    return {"status": "recorded_visual_review_pending", "folder": str(public),
            "duration_s": result["probe"]["duration_s"], "cleanup_verified": result["own_pid_cleanup"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "record"))
    parser.add_argument("--tag")
    parser.add_argument("--isaaclab", type=Path, default=Path.home() / "IsaacLab")
    args = parser.parse_args()
    if args.command == "record" and args.tag is None:
        parser.error("record requires --tag")
    result = prepare(args.isaaclab.resolve()) if args.command == "prepare" else record(args.tag, args.isaaclab.resolve())
    print(json.dumps(result, ensure_ascii=False))
