# pyright: reportMissingImports=false
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SOURCE = REPO / "bench/run_go2_stairs_velocity_v1.py"
MODULE_SPEC = importlib.util.spec_from_file_location("go2_stairs_velocity_v1", SOURCE)
assert MODULE_SPEC and MODULE_SPEC.loader
M = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(M)


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


@pytest.fixture(autouse=True)
def _checkpoint_digest_fixture(monkeypatch: pytest.MonkeyPatch) -> None:
    original = M.load_base

    def load(repo: Path):
        base = original(repo)
        base._sha = lambda _path: M.CHECKPOINT_SHA256
        return base

    monkeypatch.setattr(M, "load_base", load)


def _spec(tmp_path: Path, command: float, direction: str = "up") -> Path:
    checkpoint = tmp_path / "go2_rough_checkpoint.pt"
    checkpoint.write_bytes(b"portable checkpoint fixture")
    base = M.load_base(REPO)
    sources = base.COMMON_ISAACLAB_SOURCES | {base.ROLE_ISAACLAB_SOURCE["quadruped"]}
    value = {
        "schema": base.SCHEMA,
        "clip_id": (
        f"quadruped-stairs-{direction}-step-0p05"
        if command == 0.25 else f"quadruped-stairs-{direction}-step-0p05-cmd-0p75"
        ),
        "robot_role": "quadruped",
        "direction": direction,
        "training_task": "Isaac-Velocity-Rough-Unitree-Go2-v0",
        "play_task": "Isaac-Velocity-Rough-Unitree-Go2-Play-v0",
        "seed": 200,
        "checkpoint": {"path": str(checkpoint.resolve()), "sha256": M.CHECKPOINT_SHA256},
        "source": {
            "isaaclab_commit": "9" * 40,
            "isaaclab_files_lf": {name: "8" * 64 for name in sources},
            "runner_sha256_lf": M.BASE_RUNNER_SHA256_LF,
        },
        "capture": {
            "frames": 625,
            "fps": 50,
            "step_height_m": 0.05,
            "step_width_m": 0.3,
            "command_x_m_s": command,
            "camera": "asset_root_three_quarter",
        },
    }
    path = tmp_path / f"{direction}-{command}.json"
    _write(path, value)
    return path


def _trace(path: Path, *, negative_last: bool = False, terminated_last: bool = False) -> str:
    rows = []
    for frame in range(626):
        rows.append({
            "frame": frame,
            "sim_time_s": frame / 50,
            "root_pos_w": [2.0 * frame / 625, 0.0, 0.2 + 0.5 * frame / 625],
            "root_quat_w": [0.0, 1.0, 0.0, 0.0] if negative_last and frame == 625
            else [1.0, 0.0, 0.0, 0.0],
            "root_lin_vel_w": [0.25, 0.0, 0.0],
            "terrain_origin_w": [[0.0, 0.0, 0.0]],
            "terminated": terminated_last and frame == 625,
        })
    path.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
                    encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _complete_record(tmp_path: Path, spec_path: Path, *, negative_last: bool = False,
                     terminated_last: bool = False) -> Path:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    out = tmp_path / "record"
    video = out / "video/rl-video-step-0.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"video-fixture")
    video_sha = hashlib.sha256(video.read_bytes()).hexdigest()
    receipt = {
        "schema": "local_stairs_exact_frame_encoder_v1",
        "frames_received": 625,
        "frames_written": 625,
        "fps": 50,
        "width": 16,
        "height": 16,
        "frame_sequence": [{"index": index, "sha256": "a" * 64} for index in range(625)],
        "video_file": video.name,
        "video_sha256": video_sha,
    }
    _write(video.with_suffix(video.suffix + ".exact-frames.json"), receipt)
    trace = out / "state-trace.jsonl"
    trace_sha = _trace(trace, negative_last=negative_last, terminated_last=terminated_last)
    manifest = {
        "schema": "stairs_playback_manifest_v2",
        "clip_id": spec["clip_id"],
        "robot_role": "quadruped",
        "direction": spec["direction"],
        "training_task": spec["training_task"],
        "play_task": spec["play_task"],
        "checkpoint_sha256": M.CHECKPOINT_SHA256,
        "source": spec["source"],
        "terrain": {
            "terrain_id": "pyramid_stairs_inv",
            "terrain_seed": 200,
            "step_height_m": 0.05,
            "step_width_m": 0.3,
            "command": {"lin_vel_x": spec["capture"]["command_x_m_s"]},
        },
        "assessment": {
            "outcome": "fall" if terminated_last else "completed",
            "root_dx_m": 2.0,
            "root_dz_m": 0.5,
            "height_condition_pass": True,
            "progress_condition_pass": True,
            "termination_seen": terminated_last,
            "trace_rows": 626,
        },
        "video": {"path": "video/rl-video-step-0.mp4", "sha256": video_sha},
        "record_video_contract": {"video_length": 625, "executed_steps": 625},
        "state_trace": {"path": "state-trace.jsonl", "sha256": trace_sha},
    }
    _write(out / "manifest.json", manifest)
    return out


def test_original_command_and_new_command_use_same_frozen_source(tmp_path: Path) -> None:
    base, old = M.prepare_base(_spec(tmp_path, 0.25), repo=REPO)
    assert old["capture"]["command_x_m_s"] == 0.25
    assert old["checkpoint"]["sha256"] == M.CHECKPOINT_SHA256
    assert M._sha_lf(REPO / "bench/run_stairs_playback_v2.py") == M.BASE_RUNNER_SHA256_LF
    assert base.COMMAND_X_M_S == 0.25


def test_prepare_accepts_only_registered_velocity_pair_without_mutating_input(tmp_path: Path) -> None:
    path = _spec(tmp_path, 0.75)
    before = path.read_bytes()
    base, value = M.prepare_base(path, repo=REPO)
    assert value["capture"] == {
        "frames": 625,
        "fps": 50,
        "step_height_m": 0.05,
        "step_width_m": 0.3,
        "command_x_m_s": 0.75,
        "camera": "asset_root_three_quarter",
    }
    assert base.STEP_HEIGHT_M == 0.05
    assert base.STEP_WIDTH_M == 0.3
    assert base.COMMAND_X_M_S == 0.75
    assert path.read_bytes() == before
    invalid = json.loads(path.read_text(encoding="utf-8"))
    invalid["capture"]["command_x_m_s"] = 0.5
    _write(path, invalid)
    with pytest.raises(ValueError, match="0.25 또는 0.75"):
        M.prepare_base(path, repo=REPO)


def test_prepare_rejects_checkpoint_seed_and_step_changes(tmp_path: Path) -> None:
    for field, value in (("seed", 201), ("checkpoint", {"path": "x", "sha256": "0" * 64})):
        path = _spec(tmp_path, 0.75)
        spec = json.loads(path.read_text(encoding="utf-8"))
        spec[field] = value
        _write(path, spec)
        with pytest.raises((ValueError, FileNotFoundError)):
            M.prepare_base(path, repo=REPO)
    path = _spec(tmp_path, 0.75)
    spec = json.loads(path.read_text(encoding="utf-8"))
    spec["capture"]["step_height_m"] = 0.1
    _write(path, spec)
    with pytest.raises(ValueError, match="capture 고정값"):
        M.prepare_base(path, repo=REPO)


def test_verify_complete_requires_wrapper_receipt_and_upright_motion(tmp_path: Path) -> None:
    spec = _spec(tmp_path, 0.75)
    out = _complete_record(tmp_path, spec)
    receipt = {"child_exit_code": 0, "shutdown_verified": True, "exact_encoder_verified": True}
    result = M.verify_record(spec, out, technical_receipt=receipt, repo=REPO)
    assert result["technical_complete"] is True
    assert result["motion_accepted"] is True
    assert result["body_up_world_z"]["negative_count"] == 0
    assert result["claims"]["heldout"] is False
    with pytest.raises(ValueError, match="wrapper/encoder"):
        M.verify_record(spec, out, technical_receipt={**receipt, "shutdown_verified": False}, repo=REPO)


def test_verify_negative_body_up_preserves_motion_failure_without_cause_claim(tmp_path: Path) -> None:
    spec = _spec(tmp_path, 0.75)
    out = _complete_record(tmp_path, spec, negative_last=True)
    result = M.verify_record(
        spec, out,
        technical_receipt={"child_exit_code": 0, "shutdown_verified": True,
                           "exact_encoder_verified": True},
        repo=REPO,
    )
    assert result["technical_complete"] is True
    assert result["motion_accepted"] is False
    assert result["motion_failure_preserved"] is True
    assert result["body_up_world_z"]["negative_count"] == 1
    assert result["claims"]["fall_cause_confirmed"] is False


def test_full_length_termination_is_preserved_as_motion_failure(tmp_path: Path) -> None:
    spec = _spec(tmp_path, 0.75)
    out = _complete_record(tmp_path, spec, terminated_last=True)
    result = M.verify_record(
        spec, out,
        technical_receipt={"child_exit_code": 0, "shutdown_verified": True,
                           "exact_encoder_verified": True},
        repo=REPO,
    )
    assert result["technical_complete"] is True
    assert result["motion_accepted"] is False
    assert result["motion_failure_preserved"] is True
    assert result["assessment"]["outcome"] == "fall"
    assert result["integrity_result"] == "not_evaluated_registered_termination"


def test_early_child_failure_is_not_relabelled_as_infrastructure_success(tmp_path: Path) -> None:
    spec = _spec(tmp_path, 0.75)
    out = tmp_path / "early"
    out.mkdir()
    trace = out / "state-trace.jsonl"
    trace.write_text('{"frame":0}\n{"frame":1}\n', encoding="utf-8")
    _write(out / "failure.json", {
        "error_type": "RuntimeError",
        "error_message": "terminated early",
        "partial_evidence_preserved": True,
        "partial_trace": {"path": "state-trace.jsonl", "sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
                          "rows": 2},
    })
    result = M.verify_record(
        spec, out,
        technical_receipt={"child_exit_code": 1, "shutdown_verified": False,
                           "exact_encoder_verified": False},
        repo=REPO,
    )
    assert result["status"] == "technical_incomplete"
    assert result["technical_complete"] is False
    assert result["motion_accepted"] is False
    assert result["motion_failure_preserved"] is True
    assert result["partial_trace_rows"] == 2


def test_complete_artifact_tamper_is_rejected_not_downgraded(tmp_path: Path) -> None:
    spec = _spec(tmp_path, 0.25)
    out = _complete_record(tmp_path, spec)
    trace = out / "state-trace.jsonl"
    trace.write_text(trace.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="state trace SHA"):
        M.verify_record(
            spec, out,
            technical_receipt={"child_exit_code": 0, "shutdown_verified": True,
                               "exact_encoder_verified": True},
            repo=REPO,
        )


def test_import_and_help_cli_do_not_load_sdk(tmp_path: Path) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = ""
    code = (
        "import importlib.util,json,sys,pathlib;"
        f"p=pathlib.Path({str(SOURCE)!r});"
        "s=importlib.util.spec_from_file_location('velocity',p);"
        "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
        "print(json.dumps(sorted(set(sys.modules)&{'isaaclab','torch','gymnasium'})))"
    )
    imported = subprocess.run([sys.executable, "-B", "-c", code], cwd=tmp_path, env=env,
                              capture_output=True, text=True, encoding="utf-8", errors="strict", check=False)
    assert imported.returncode == 0
    assert imported.stdout.strip() == "[]"
    prepared = subprocess.run(
        [sys.executable, "-B", str(SOURCE), "prepare", "--help"],
        cwd=tmp_path, env=env, capture_output=True, text=True,
        encoding="utf-8", errors="strict", check=False,
    )
    assert prepared.returncode == 0, prepared.stderr
    assert "--spec" in prepared.stdout and "--repo" in prepared.stdout
    assert prepared.stderr == ""
