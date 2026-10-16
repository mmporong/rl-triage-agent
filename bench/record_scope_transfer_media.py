"""U1과 ANYmal-C의 실패·교정·정상 정책을 같은 Play 설정으로 녹화한다."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "bench", ROOT / "evals", ROOT / "src"):
    sys.path.insert(0, str(directory))

import record_correction_media as C  # noqa: E402
import record_evidence_media as M  # noqa: E402
import run_runtime_readback_smoke as P  # noqa: E402
from rl_triage import grounded_recovery as GR  # noqa: E402
from rl_triage import recovery as RC  # noqa: E402

PROTOCOLS = {
    "u1": ROOT / "bench/protocols/scope_transfer_media_u1_s2051_v1.json",
    "anymal": ROOT / "bench/protocols/scope_transfer_media_anymal_s2047_v1.json",
}
PRIVATE_SOURCE_PATHS = {
    "u1_scope": ROOT / "bench/private/codex_support_20261015/task5/runs/u1_scope_s2051_20261015/scope.json",
    "u1_corrected_receipt": ROOT / "bench/private/codex_support_20261015/task5/runs/u1_scope_s2051_20261015/u1_family_restart/train/receipt.json",
    "anymal_none_step": ROOT / "bench/private/e2e_demo_anymal_c/anymal_c_e2e_f1_s2047_20261014/steps/01_train_NONE.json",
    "anymal_fault_step": ROOT / "bench/private/e2e_demo_anymal_c/anymal_c_e2e_f1_s2047_20261014/steps/02_train_FAULT.json",
    "anymal_corrected_receipt": ROOT / "bench/private/e2e_demo_anymal_c/anymal_c_e2e_f1_s2047_20261014/fix/restart_corrected/train/receipt.json",
    "anymal_corrected_eval": ROOT / "bench/private/e2e_demo_anymal_c/anymal_c_e2e_f1_s2047_20261014/fix/eval/restart_corrected.json",
}
CODE = {
    "bench/record_scope_transfer_media.py",
    "bench/record_evidence_media.py",
    "bench/record_correction_media.py",
    "bench/run_runtime_readback_smoke.py",
    "src/rl_triage/grounded_recovery.py",
    "src/rl_triage/recovery.py",
}
FIXED = {
    "seed": 42,
    "num_envs": 1,
    "video_length": 300,
    "kit_args": "--/app/vulkan=false --/app/window/hideUi=true",
    "overrides": ["env.viewer.eye=[3.0,3.0,2.0]", "env.viewer.lookat=[0.0,0.0,0.35]"],
    "video_contract": M.EXPECTED_VIDEO_CONTRACT,
    "gpu_admission": M.EXPECTED_GPU_ADMISSION,
    "gif_fps": 15,
    "gif_frame_count": 90,
    "gif_duration_s": 6.0,
    "gif_duration_tolerance_s": 0.1,
}
CLIP_IDS = ("before_original", "after_corrected", "after_sham")
PUBLIC_SOURCE_FIELDS = {"path", "sha256_lf"}
PRIVATE_SOURCE_FIELDS = {"logical_id", "sha256"}
CLIP_FIELDS = {
    "id", "role", "logical_run_id", "run_basename", "checkpoint",
    "checkpoint_sha256", "params_env_sha256", "params_agent_sha256",
    "fixed_eval_label", "overrides",
}
PLAY_TASK_SOURCE_FIELDS = {"role", "relative_path", "sha256", "required_symbols"}
CASE_CONTRACT = {
    "u1": {
        "task": "Isaac-Velocity-Flat-Unitree-Go2-Play-v0",
        "public": {
            "evals/results/u1_scope_extension_s2051_20261015/summary.json",
            "evals/results/u1_scope_extension_s2051_20261015/source_failure_evidence.json",
            "evals/results/u1_scope_extension_s2051_20261015/evaluations/u1_family_restart.json",
            "evals/results/u1_scope_extension_s2051_20261015/evaluations/none_family_control.json",
        },
        "private": {"u1_scope", "u1_corrected_receipt"},
        "play_sources": (
            ("registration", "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/__init__.py", ("Isaac-Velocity-Flat-Unitree-Go2-Play-v0", "UnitreeGo2FlatEnvCfg_PLAY", "UnitreeGo2FlatPPORunnerCfg")),
            ("environment", "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/flat_env_cfg.py", ("UnitreeGo2FlatEnvCfg_PLAY",)),
            ("agent", "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/agents/rsl_rl_ppo_cfg.py", ("UnitreeGo2FlatPPORunnerCfg",)),
        ),
        "clips": (
            ("before_original", "failed", "u1_failed_s2051", "2026-10-15_18-16-17_u1_s2051", "unhealthy"),
            ("after_corrected", "corrected_restart", "u1_corrected_restart_s3051", "2026-10-15_18-36-09_u1_scope_s2051_20261015_u1_family_restart", "healthy"),
            ("after_sham", "none_control", "u1_none_s2051", "2026-10-15_18-08-16_baseline_p0c_s2051", "healthy"),
        ),
    },
    "anymal": {
        "task": "Isaac-Velocity-Flat-Anymal-C-Play-v0",
        "public": {
            "evals/results/anymal_c_band_20261014/result.json",
            "evals/results/anymal_c_e2e_f1_s2047_20261014/demo_result.json",
            "evals/results/anymal_c_e2e_f1_s2047_20261014/runs/anymal_c_h02_s2047.json",
            "evals/results/anymal_c_e2e_f1_s2047_20261014/runs/anymal_c_none_s2047.json",
        },
        "private": {"anymal_none_step", "anymal_fault_step", "anymal_corrected_receipt", "anymal_corrected_eval"},
        "play_sources": (
            ("registration", "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/anymal_c/__init__.py", ("Isaac-Velocity-Flat-Anymal-C-Play-v0", "AnymalCFlatEnvCfg_PLAY", "AnymalCFlatPPORunnerCfg")),
            ("environment", "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/anymal_c/flat_env_cfg.py", ("AnymalCFlatEnvCfg_PLAY",)),
            ("agent", "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/anymal_c/agents/rsl_rl_ppo_cfg.py", ("AnymalCFlatPPORunnerCfg",)),
        ),
        "clips": (
            ("before_original", "failed", "anymal_failed_s2047", "2026-10-14_19-11-32_anymal_c_h02_s2047", "unhealthy"),
            ("after_corrected", "corrected_restart", "anymal_corrected_restart_s3047", "2026-10-14_19-27-01_anymal_c_e2e_f1_s2047_20261014_fix_restart_corrected", "healthy"),
            ("after_sham", "none_control", "anymal_none_s2047", "2026-10-14_19-01-08_anymal_c_none_s2047", "healthy"),
        ),
    },
}


def sha(path: Path, lf: bool = False) -> str:
    return M.sha256_lf_file(path) if lf else M.sha256_file(path)


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _git_blob_lf(commit: str, relative: str) -> str:
    raw = subprocess.run(
        ["git", "-C", str(ROOT), "show", f"{commit}:{relative}"],
        check=True,
        capture_output=True,
    ).stdout.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(raw).hexdigest()


def _assert_anchor_ancestor(commit: str) -> None:
    completed = subprocess.run(
        ["git", "-C", str(ROOT), "merge-base", "--is-ancestor", commit, "HEAD"],
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise ValueError("source anchor is not an ancestor of HEAD")


def _validate_protocol(protocol: dict, case: str) -> None:
    if protocol.get("schema") != "scope_transfer_media_protocol_v1" or protocol.get("case_id") != case:
        raise ValueError("protocol identity differs")
    for key, value in FIXED.items():
        if protocol.get(key) != value:
            raise ValueError("Play setting differs")
    if tuple(clip.get("id") for clip in protocol.get("clips", [])) != CLIP_IDS:
        raise ValueError("clip order differs")
    contract = CASE_CONTRACT[case]
    if protocol.get("task") != contract["task"]:
        raise ValueError("case task differs")
    if any(set(row) != PUBLIC_SOURCE_FIELDS for row in protocol.get("public_sources", [])):
        raise ValueError("public source fields differ")
    public_paths = [row.get("path") for row in protocol.get("public_sources", [])]
    if len(public_paths) != len(contract["public"]) or set(public_paths) != contract["public"]:
        raise ValueError("public source inventory differs")
    if any(set(row) != PRIVATE_SOURCE_FIELDS for row in protocol.get("private_sources", [])):
        raise ValueError("private source fields differ")
    private_ids = [row.get("logical_id") for row in protocol.get("private_sources", [])]
    if len(private_ids) != len(contract["private"]) or set(private_ids) != contract["private"]:
        raise ValueError("private source inventory differs")
    if any(set(clip) != CLIP_FIELDS for clip in protocol.get("clips", [])):
        raise ValueError("clip fields differ")
    identity = tuple(
        (clip.get("id"), clip.get("role"), clip.get("logical_run_id"), clip.get("run_basename"), clip.get("fixed_eval_label"))
        for clip in protocol["clips"]
    )
    if identity != contract["clips"] or any(clip.get("checkpoint") != "model_299.pt" for clip in protocol["clips"]):
        raise ValueError("case clip identity differs")
    if any(clip.get("overrides") != FIXED["overrides"] for clip in protocol["clips"]):
        raise ValueError("clip Play override differs")
    if any(set(row) != PLAY_TASK_SOURCE_FIELDS for row in protocol.get("play_task_sources", [])):
        raise ValueError("Play task source fields differ")
    play_sources = tuple(
        (row.get("role"), row.get("relative_path"), tuple(row.get("required_symbols", [])))
        for row in protocol.get("play_task_sources", [])
    )
    if play_sources != contract["play_sources"]:
        raise ValueError("Play task source inventory differs")
    if protocol.get("media_claim") != "visual_playback_only_fixed_eval_labels_preserved":
        raise ValueError("media claim differs")
    if protocol.get("model_calls") != 0:
        raise ValueError("model call contract differs")
    if any("path" in item for item in protocol.get("private_sources", [])):
        raise ValueError("private sources must use logical IDs only")


def _check_public_sources(protocol: dict) -> None:
    commit = protocol["source_anchor_commit"]
    _assert_anchor_ancestor(commit)
    for source in protocol["public_sources"]:
        relative = source["path"]
        path = ROOT / relative
        if not path.is_file() or sha(path, True) != source["sha256_lf"]:
            raise ValueError(f"public source differs: {relative}")
        if _git_blob_lf(commit, relative) != source["sha256_lf"]:
            raise ValueError(f"public source blob differs: {relative}")
        if _git_blob_lf("HEAD", relative) != source["sha256_lf"]:
            raise ValueError(f"public source HEAD blob differs: {relative}")
    status = subprocess.run(
        ["git", "-C", str(ROOT), "status", "--porcelain", "--", *[row["path"] for row in protocol["public_sources"]]],
        check=True,
        capture_output=True,
    )
    if status.stdout:
        raise ValueError("public result inputs are not clean")


def _check_code(protocol: dict) -> None:
    actual = {relative: sha(ROOT / relative, True) for relative in sorted(CODE)}
    if protocol.get("code_sha256_lf") != actual:
        raise ValueError("recording source differs")


def _check_private_sources(protocol: dict) -> dict[str, dict]:
    resolved = {}
    expected_ids = {item["logical_id"] for item in protocol["private_sources"]}
    if len(expected_ids) != len(protocol["private_sources"]):
        raise ValueError("private logical IDs repeat")
    for binding in protocol["private_sources"]:
        logical_id = binding["logical_id"]
        path = PRIVATE_SOURCE_PATHS.get(logical_id)
        if path is None or not path.is_file() or sha(path) != binding["sha256"]:
            raise ValueError(f"private source differs: {logical_id}")
        resolved[logical_id] = read(path)
    return resolved


def _check_play_task_sources(protocol: dict, isaaclab: Path) -> None:
    for binding in protocol["play_task_sources"]:
        path = isaaclab / binding["relative_path"]
        if not path.is_file() or sha(path) != binding["sha256"]:
            raise ValueError("Play task source differs")
        text = path.read_text(encoding="utf-8")
        if any(symbol not in text for symbol in binding["required_symbols"]):
            raise ValueError("Play task registration/config symbol differs")


def _clip_map(protocol: dict) -> dict[str, dict]:
    return {clip["logical_run_id"]: clip for clip in protocol["clips"]}


def _validate_result_bindings(protocol: dict, case: str, private: dict[str, dict]) -> None:
    clips = _clip_map(protocol)
    public = {row["path"]: read(ROOT / row["path"]) for row in protocol["public_sources"]}
    if case == "u1":
        root = "evals/results/u1_scope_extension_s2051_20261015/"
        summary = public[root + "summary.json"]
        source = public[root + "source_failure_evidence.json"]
        corrected = public[root + "evaluations/u1_family_restart.json"]
        none = public[root + "evaluations/none_family_control.json"]
        expected = (
            summary.get("status") == "complete",
            summary.get("outcome") == "recovered",
            summary.get("source_failure", {}).get("u1_label") == "unhealthy",
            summary.get("corrected_u1_restart", {}).get("label") == "healthy",
            summary.get("none_control", {}).get("label") == "healthy",
            source.get("gate", {}).get("outcome") == "failure_present",
            source.get("arms", {}).get("U1", {}).get("checkpoint", {}).get("sha256") == clips["u1_failed_s2051"]["checkpoint_sha256"],
            source.get("arms", {}).get("NONE", {}).get("checkpoint", {}).get("sha256") == clips["u1_none_s2051"]["checkpoint_sha256"],
            corrected.get("label") == clips["u1_corrected_restart_s3051"]["fixed_eval_label"] == "healthy",
            corrected.get("checkpoint", {}).get("sha256") == clips["u1_corrected_restart_s3051"]["checkpoint_sha256"],
            none.get("label") == clips["u1_none_s2051"]["fixed_eval_label"] == "healthy",
            none.get("checkpoint", {}).get("sha256") == clips["u1_none_s2051"]["checkpoint_sha256"],
            clips["u1_failed_s2051"]["fixed_eval_label"] == "unhealthy",
        )
    else:
        root = "evals/results/anymal_c_e2e_f1_s2047_20261014/"
        demo = public[root + "demo_result.json"]
        fault = public[root + "runs/anymal_c_h02_s2047.json"]
        none = public[root + "runs/anymal_c_none_s2047.json"]
        corrected_eval = private["anymal_corrected_eval"]
        band_result = public["evals/results/anymal_c_band_20261014/result.json"]
        bands = band_result.get("analysis", {}).get("bands")
        corrected_verdict = RC.verdict(RC.compare(GR._behavior(corrected_eval), GR._behavior(none)), bands)
        expected = (
            band_result.get("outcome") == "pass",
            demo.get("outcome") == "recovered",
            demo.get("demo_success") is True,
            demo.get("terminal") is None,
            demo.get("evaluation", {}).get("gate", {}).get("outcome") == "failure_present",
            demo.get("evaluation", {}).get("gate", {}).get("fault_verdict", {}).get("label") == "unhealthy",
            demo.get("evaluation", {}).get("gate", {}).get("none_walks") is True,
            demo.get("correction", {}).get("result", {}).get("verdict", {}).get("label") == "healthy",
            fault.get("checkpoint", {}).get("sha256") == clips["anymal_failed_s2047"]["checkpoint_sha256"],
            none.get("checkpoint", {}).get("sha256") == clips["anymal_none_s2047"]["checkpoint_sha256"],
            corrected_eval.get("checkpoint", {}).get("sha256") == clips["anymal_corrected_restart_s3047"]["checkpoint_sha256"],
            corrected_eval.get("name") == "restart_corrected",
            corrected_verdict.get("label") == "healthy",
            [clip["fixed_eval_label"] for clip in protocol["clips"]] == ["unhealthy", "healthy", "healthy"],
        )
    if not all(expected):
        raise ValueError("checkpoint and fixed-evaluation policy binding differs")


def _clip_path_from_source(case: str, logical_run_id: str, private: dict[str, dict]) -> Path:
    if case == "u1":
        scope = private["u1_scope"]["bindings"]
        if logical_run_id == "u1_failed_s2051":
            return Path(scope["u1_checkpoint_path"])
        if logical_run_id == "u1_none_s2051":
            return Path(scope["none_checkpoint_path"])
        if logical_run_id == "u1_corrected_restart_s3051":
            return Path(private["u1_corrected_receipt"]["checkpoint"]["path"])
    else:
        if logical_run_id == "anymal_failed_s2047":
            return Path(private["anymal_fault_step"]["artifacts"]["checkpoint"]["path"])
        if logical_run_id == "anymal_none_s2047":
            return Path(private["anymal_none_step"]["artifacts"]["checkpoint"]["path"])
        if logical_run_id == "anymal_corrected_restart_s3047":
            return Path(private["anymal_corrected_receipt"]["checkpoint"]["path"])
    raise ValueError("unknown logical run ID")


def resolve_inputs(protocol: dict, case: str, private: dict[str, dict]) -> list[dict[str, Any]]:
    rows = []
    for clip in protocol["clips"]:
        checkpoint = _clip_path_from_source(case, clip["logical_run_id"], private)
        if checkpoint.name != "model_299.pt" or checkpoint.parent.name != clip["run_basename"]:
            raise ValueError("checkpoint lineage differs")
        params = checkpoint.parent / "params"
        checks = (
            (checkpoint, clip["checkpoint_sha256"]),
            (params / "env.yaml", clip["params_env_sha256"]),
            (params / "agent.yaml", clip["params_agent_sha256"]),
        )
        if any(not path.is_file() or sha(path) != expected for path, expected in checks):
            raise ValueError("checkpoint or params digest differs")
        rows.append({"clip": clip, "checkpoint": checkpoint, "params_dir": params})
    return rows


def verify(case: str, isaaclab: Path, *, require_committed: bool) -> tuple[dict, list[dict[str, Any]], str | None]:
    if case not in PROTOCOLS:
        raise ValueError("unknown case")
    protocol_path = PROTOCOLS[case]
    protocol = read(protocol_path)
    _validate_protocol(protocol, case)
    _check_code(protocol)
    _check_public_sources(protocol)
    private = _check_private_sources(protocol)
    _validate_result_bindings(protocol, case, private)
    if sha(isaaclab / "scripts/reinforcement_learning/rsl_rl/play.py") != protocol["play_py_sha256"]:
        raise ValueError("play.py differs")
    _check_play_task_sources(protocol, isaaclab)
    if subprocess.run(["git", "-C", str(isaaclab), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip() != protocol["isaaclab_commit"]:
        raise ValueError("Isaac Lab commit differs")
    if subprocess.run(["git", "-C", str(isaaclab), "status", "--porcelain", "--untracked-files=no"], check=True, capture_output=True).stdout:
        raise ValueError("Isaac Lab tracked files are not clean")
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise ValueError(f"missing media tool: {tool}")
    inputs = resolve_inputs(protocol, case, private)
    commit = None
    if require_committed:
        commit = P.verify_committed(ROOT, CODE | {protocol_path.relative_to(ROOT).as_posix()})
    return protocol, inputs, commit


def output_paths(protocol: dict, tag: str) -> tuple[Path, Path]:
    if re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,47}", tag) is None:
        raise ValueError("tag is invalid")
    return ROOT / "docs/media" / tag, ROOT / "bench/private/scope_transfer_media" / tag


def ensure_outputs_absent(public: Path, private: Path) -> None:
    if public.exists() or private.exists():
        raise FileExistsError("media output already exists; use a new tag")


def prepare(case: str, isaaclab: Path) -> dict:
    protocol, inputs, _ = verify(case, isaaclab, require_committed=False)
    return {
        "status": "prepared",
        "case_id": case,
        "task": protocol["task"],
        "clips": [item["clip"]["logical_run_id"] for item in inputs],
        "gpu_calls": 0,
        "model_calls": 0,
    }


def record(case: str, isaaclab: Path, tag: str, timeout_s: int) -> dict:
    protocol, inputs, commit = verify(case, isaaclab, require_committed=True)
    public, private = output_paths(protocol, tag)
    ensure_outputs_absent(public, private)
    admission = M.query_gpu()
    if admission["busy"]:
        return {"status": "gpu_busy_no_launch", "clips_recorded": 0}
    private.mkdir(parents=True, exist_ok=False)
    records = []
    gpu_checks = []
    for item in inputs:
        before = M.query_gpu()
        if before["busy"]:
            raise RuntimeError("GPU became busy before clip; partial files remain private")
        before_path = private / f"gpu_before_{item['clip']['id']}.json"
        M._write_new_json(before_path, before)
        result = M._record_clip(protocol, item, isaaclab, private, timeout_s)
        after = M.query_gpu()
        after_path = private / f"gpu_after_{item['clip']['id']}.json"
        M._write_new_json(after_path, after)
        if after["busy"]:
            raise RuntimeError("GPU became busy after clip; partial files remain private")
        if result["own_pid_cleanup"] is not True or result["residual_processes_observed"] != 0:
            raise RuntimeError("owned process cleanup failed")
        verify(case, isaaclab, require_committed=True)
        records.append(result)
        gpu_checks.append({
            "clip_id": item["clip"]["id"],
            "before": {"logical_id": f"gpu_before_{item['clip']['id']}", "sha256": sha(before_path),
                       "busy": False, "utilization_percent": before["utilization_percent"], "memory_used_mib": before["memory_used_mib"]},
            "after": {"logical_id": f"gpu_after_{item['clip']['id']}", "sha256": sha(after_path),
                      "busy": False, "utilization_percent": after["utilization_percent"], "memory_used_mib": after["memory_used_mib"]},
        })

    staging = private / "public_staging"
    staging.mkdir(exist_ok=False)
    clips = []
    for result in records:
        clip = result["clip"]
        target = staging / f"{clip['id']}.mp4"
        shutil.copy2(result["video"], target)
        clips.append({
            "id": clip["id"],
            "role": clip["role"],
            "logical_run_id": clip["logical_run_id"],
            "file": target.name,
            "sha256": sha(target),
            "checkpoint_sha256": clip["checkpoint_sha256"],
            "fixed_eval_label": clip["fixed_eval_label"],
            "ffprobe": result["probe"],
            "cleanup_verified": True,
        })
    comparison, gif = C._compose(staging)
    comparison_probe = M._probe_video(comparison)
    M._validate_video_contract(comparison_probe, {**protocol["video_contract"], "width": 1920, "height": 360})
    gif_probe = M._probe_video(gif)
    C._validate_gif(gif_probe)
    manifest = {
        "schema": "scope_transfer_media_artifact_v1",
        "case_id": case,
        "source_result_disclosure": protocol["source_result_disclosure"],
        "interpretation": protocol["interpretation"],
        "media_claim": protocol["media_claim"],
        "recording_commit_sha": commit,
        "protocol": {"path": PROTOCOLS[case].relative_to(ROOT).as_posix(), "sha256_lf": sha(PROTOCOLS[case], True)},
        "public_sources": protocol["public_sources"],
        "private_sources": protocol["private_sources"],
        "playback": {key: protocol[key] for key in ("task", "seed", "num_envs", "video_length", "kit_args", "overrides")},
        "gpu_checks": gpu_checks,
        "clips": clips,
        "comparison": {
            "mp4": {"file": comparison.name, "sha256": sha(comparison), "ffprobe": comparison_probe},
            "gif": {"file": gif.name, "sha256": sha(gif), "ffprobe": gif_probe},
        },
    }
    M._write_new_json(staging / "source_manifest.json", manifest)
    public.parent.mkdir(parents=True, exist_ok=True)
    staging.replace(public)
    return {"status": "recorded", "case_id": case, "folder": public.relative_to(ROOT).as_posix(), "clips": 3}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "record"))
    parser.add_argument("case", choices=tuple(PROTOCOLS))
    parser.add_argument("--isaaclab", type=Path, default=Path.home() / "IsaacLab")
    parser.add_argument("--tag")
    parser.add_argument("--timeout-s", type=int, default=900)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare(args.case, args.isaaclab.resolve())
        else:
            if args.tag is None or args.timeout_s < 60:
                raise ValueError("record requires tag and timeout >= 60")
            result = record(args.case, args.isaaclab.resolve(), args.tag, args.timeout_s)
    except Exception as error:
        print(json.dumps({"status": "rejected", "reason": type(error).__name__, "detail": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    if args.command == "record" and result.get("status") != "recorded":
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
