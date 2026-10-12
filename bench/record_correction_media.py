"""검증된 case_01 교정 실험의 세 정책을 같은 Isaac Play 조건으로 녹화한다.

``prepare``는 완료된 비공개 실행과 공개 수치 export를 다시 검증한 뒤, 녹화
프로토콜만 새로 만든다. 프로토콜과 이 파일을 커밋한 다음에만 ``record``가
GPU를 사용할 수 있다.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import traceback
from typing import Any
import uuid


ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "bench", ROOT / "evals"):
    sys.path.insert(0, str(directory))

import record_evidence_media as M  # noqa: E402
import run_runtime_readback_smoke as P  # noqa: E402
import run_training_correction as H  # noqa: E402
import publish_training_correction as Q  # noqa: E402


PLAY = "scripts/reinforcement_learning/rsl_rl/play.py"
CODE = {
    "bench/record_correction_media.py",
    "bench/record_evidence_media.py",
    "bench/run_runtime_readback_smoke.py",
    "evals/publish_training_correction.py",
    "evals/run_training_correction.py",
}
CLIP_IDS = ("before_original", "after_corrected", "after_sham")
ALLOWED_RESULTS = {
    "restoration_observed_single_run", "source_failure_absent", "baseline_invalid",
    "no_failure_observed", "sham_shifted", "no_recovery",
}
FIXED = {
    "schema": "training_correction_media_v1",
    "scope": "verified_case_01_policy_playback",
    "case": "case_01",
    "task": M.EXPECTED_TASK,
    "seed": 42,
    "num_envs": 1,
    "video_length": 300,
    "kit_args": "--/app/vulkan=false --/app/window/hideUi=true",
    "overrides": [
        "env.viewer.eye=[3.0,3.0,2.0]",
        "env.viewer.lookat=[0.0,0.0,0.35]",
    ],
    "video_contract": M.EXPECTED_VIDEO_CONTRACT,
    "gpu_admission": M.EXPECTED_GPU_ADMISSION,
    "gif_fps": 15,
    "gif_frame_count": 90,
    "gif_duration_s": 6.0,
    "gif_duration_tolerance_s": 0.1,
    "model_calls": 0,
    "model_selected_correction": False,
    "media_claim": "policy_playback_only_result_status_preserved",
}
GENERATED_KEYS = {
    "source_tag", "result_status", "result_wording", "source_commit_sha", "source_commit_scope",
    "publisher_sources", "source_protocol_sha256_lf", "fixed_eval_sha256_lf", "code_sha256_lf",
    "play_py_sha256", "isaaclab_commit", "public_sources", "private_sources", "clips",
}
CLIP_KEYS = {
    "id", "role", "run_basename", "checkpoint", "checkpoint_sha256",
    "params_env_sha256", "params_agent_sha256", "overrides",
}
CLIP_CONTRACT = {
    "before_original": ("original_failed_source", "model_299.pt"),
    "after_corrected": ("corrected", "model_598.pt"),
    "after_sham": ("fault_sham", "model_598.pt"),
}


def _tag(value: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,39}", value) is None:
        raise ValueError("source tag must use lowercase letters, digits, underscores and hyphens")
    return value


def protocol_path(tag: str) -> Path:
    return ROOT / "bench" / "protocols" / f"training_correction_media_{_tag(tag)}.json"


def _relative(path: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(ROOT.resolve()).as_posix()
    except ValueError as error:
        raise ValueError("source artifact is outside the repository") from error


def _artifact(path: Path, *, lf: bool = False) -> dict[str, str]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"missing or unsafe source artifact: {_relative(path)}")
    return {"path": _relative(path), "sha256_lf" if lf else "sha256":
            M.sha256_lf_file(path) if lf else M.sha256_file(path)}


def _private_paths(tag: str) -> dict[str, Path]:
    private = H.folder_for(tag)
    paths = {
        "request": private / "request.json",
        "approval": private / "approval.json",
        "result": private / "result.json",
    }
    for arm in H.ARMS:
        for logical, relative in {
            "consumed": "consumed.json",
            "train_manifest": "train/manifest.json",
            "train_execution": "train/execution.json",
            "train_initial": "train/initial.json",
            "train_correction": "train/correction.json",
            "train_receipt": "train/receipt.json",
            "eval_execution": "eval/execution.json",
            "eval_report": "eval/report.json",
            "parent": "parent.json",
        }.items():
            paths[f"{arm}.{logical}"] = private / arm / relative
    return paths


def _private_artifacts(tag: str) -> list[dict[str, str]]:
    return [{"id": logical_id, "sha256": M.sha256_file(path)}
            for logical_id, path in sorted(_private_paths(tag).items())
            if path.is_file() and not path.is_symlink()]


def _verified_source(tag: str, isaaclab: Path) -> dict[str, Any]:
    """비공개 실행, 공개 export, 체크포인트를 함께 다시 검증한다."""
    tag = _tag(tag)
    private = H.folder_for(tag)
    public = ROOT / "evals" / "results" / tag
    replay = Q.replay(public)
    public_result = H.read(public / "result.json")
    numeric = public_result.get("numeric", {})
    status = numeric.get("status")
    if (public_result.get("case") != FIXED["case"] or status not in ALLOWED_RESULTS
            or replay.get("status") != status
            or numeric.get("model_selected_correction") is not False):
        raise ValueError("public correction result is invalid, partial, or from another case")
    source_commit_scope = public_result.get("source_commit_scope")
    publisher_sources = public_result.get("publisher_sources")
    if (source_commit_scope != "experiment_execution_and_verdict_code_only"
            or not isinstance(publisher_sources, dict) or not publisher_sources):
        raise ValueError("public publisher source binding is incomplete")

    source_protocol, source_commit = H.verify_protocol(isaaclab)
    request = H.read(private / "request.json")
    private_result = H.read(private / "result.json")
    if (request.get("tag") != tag or request.get("case") != FIXED["case"]
            or request.get("source_commit_sha") != source_commit
            or private_result.get("status") != status
            or private_result.get("source_commit_sha") != source_commit
            or public_result.get("source_commit_sha") != source_commit):
        raise ValueError("private and public correction results are not the same verified execution")

    receipts: dict[str, dict[str, Any]] = {}
    for arm in H.ARMS:
        training = private / arm / "train"
        receipt = H.verify_training(private, arm, H.read(training / "execution.json"))
        report = private / arm / "eval" / "report.json"
        eval_execution = H.read(private / arm / "eval" / "execution.json")
        parent = H.read(private / arm / "parent.json")
        if (not H.execution_valid(eval_execution) or parent.get("status") != "pass"
                or parent.get("training_receipt_sha256") != H.sha(training / "receipt.json")
                or parent.get("fixed_report_sha256") != H.sha(report)):
            raise ValueError(f"{arm} execution is partial or invalid")
        receipts[arm] = receipt

    clips: list[dict[str, Any]] = []
    original = H.source_paths(isaaclab, "F1")
    clips.append({
        "id": CLIP_IDS[0], "role": "original_failed_source", "run_basename": original["checkpoint"].parent.name,
        "checkpoint": original["checkpoint"].name,
        "checkpoint_sha256": M.sha256_file(original["checkpoint"]),
        "params_env_sha256": M.sha256_file(original["environment"]),
        "params_agent_sha256": M.sha256_file(original["agent"]),
        "overrides": deepcopy(FIXED["overrides"]),
    })
    for clip_id, arm in ((CLIP_IDS[1], "corrected"), (CLIP_IDS[2], "fault_sham")):
        receipt = receipts[arm]
        checkpoint = Path(receipt["checkpoint"]["path"])
        env = Path(receipt["configs"]["environment"]["path"])
        agent = Path(receipt["configs"]["agent"]["path"])
        clips.append({
            "id": clip_id, "role": arm, "run_basename": checkpoint.parent.name,
            "checkpoint": checkpoint.name, "checkpoint_sha256": receipt["checkpoint"]["sha256"],
            "params_env_sha256": receipt["configs"]["environment"]["sha256"],
            "params_agent_sha256": receipt["configs"]["agent"]["sha256"],
            "overrides": deepcopy(FIXED["overrides"]),
        })
        for path, expected in ((checkpoint, clips[-1]["checkpoint_sha256"]),
                               (env, clips[-1]["params_env_sha256"]),
                               (agent, clips[-1]["params_agent_sha256"])):
            M._check_hash(path, expected, f"{arm} source")

    public_manifest = H.read(public / "artifact_manifest.json")
    public_sources = [_artifact(public / "artifact_manifest.json")]
    for name in sorted(public_manifest["files"]):
        row = _artifact(public / name)
        if row["sha256"] != public_manifest["files"][name]:
            raise ValueError("public artifact manifest hash differs")
        public_sources.append(row)
    private_sources = _private_artifacts(tag)
    if len(private_sources) != len(_private_paths(tag)):
        raise ValueError("private correction artifact set is incomplete")
    return {
        "tag": tag, "status": status, "wording": numeric.get("wording"),
        "source_commit_sha": source_commit, "source_commit_scope": source_commit_scope,
        "publisher_sources": publisher_sources,
        "source_protocol_sha256_lf": H.sha(H.PROTOCOL, True),
        "fixed_eval_sha256_lf": source_protocol["fixed_eval_sha256_lf"],
        "public_sources": public_sources, "private_sources": private_sources, "clips": clips,
    }


def prepare(tag: str, isaaclab: Path) -> dict[str, Any]:
    source = _verified_source(tag, isaaclab)
    path = protocol_path(tag)
    protocol = {
        **deepcopy(FIXED), "source_tag": source["tag"], "result_status": source["status"],
        "result_wording": source["wording"], "source_commit_sha": source["source_commit_sha"],
        "source_commit_scope": source["source_commit_scope"],
        "publisher_sources": source["publisher_sources"],
        "source_protocol_sha256_lf": source["source_protocol_sha256_lf"],
        "fixed_eval_sha256_lf": source["fixed_eval_sha256_lf"],
        "code_sha256_lf": {name: M.sha256_lf_file(ROOT / name) for name in sorted(CODE)},
        "play_py_sha256": M.sha256_file(isaaclab / PLAY),
        "isaaclab_commit": H.FIXED["isaaclab_commit"],
        "public_sources": source["public_sources"], "private_sources": source["private_sources"],
        "clips": source["clips"],
    }
    M._write_new_json(path, protocol)
    return {"status": "prepared_commit_before_recording",
            "protocol": path.relative_to(ROOT).as_posix()}


def _resolve_clips(protocol: dict[str, Any], isaaclab: Path) -> list[dict[str, Any]]:
    private = H.folder_for(protocol["source_tag"])
    original = H.source_paths(isaaclab, "F1")
    locations = {
        CLIP_IDS[0]: (original["checkpoint"], original["environment"].parent),
        CLIP_IDS[1]: (Path(H.read(private / "corrected/train/receipt.json")["checkpoint"]["path"]),
                      Path(H.read(private / "corrected/train/receipt.json")["configs"]["environment"]["path"]).parent),
        CLIP_IDS[2]: (Path(H.read(private / "fault_sham/train/receipt.json")["checkpoint"]["path"]),
                      Path(H.read(private / "fault_sham/train/receipt.json")["configs"]["environment"]["path"]).parent),
    }
    return [{"clip": clip, "checkpoint": locations[clip["id"]][0], "params_dir": locations[clip["id"]][1]}
            for clip in protocol["clips"]]


def _safe_component(value: object) -> bool:
    return (isinstance(value, str) and value not in {"", ".", ".."}
            and not Path(value).is_absolute() and Path(value).name == value
            and "/" not in value and "\\" not in value)


def _sha(value: object, length: int = 64) -> bool:
    return isinstance(value, str) and re.fullmatch(rf"[0-9a-f]{{{length}}}", value) is not None


def _validate_protocol_shape(protocol: dict[str, Any], tag: str) -> None:
    if set(protocol) != set(FIXED) | GENERATED_KEYS or protocol.get("source_tag") != tag:
        raise ValueError("recording protocol schema or source tag changed")
    if (protocol.get("result_status") not in ALLOWED_RESULTS
            or not isinstance(protocol.get("result_wording"), str)
            or not protocol["result_wording"]
            or not _sha(protocol.get("source_commit_sha"), 40)
            or protocol.get("source_commit_scope") != "experiment_execution_and_verdict_code_only"
            or not _sha(protocol.get("source_protocol_sha256_lf"))
            or not _sha(protocol.get("fixed_eval_sha256_lf"))
            or not _sha(protocol.get("play_py_sha256"))
            or protocol.get("isaaclab_commit") != H.FIXED["isaaclab_commit"]):
        raise ValueError("recording protocol source identity changed")
    for mapping in (protocol.get("publisher_sources"), protocol.get("code_sha256_lf")):
        if (not isinstance(mapping, dict) or not mapping
                or any(not isinstance(name, str) or not _sha(digest) for name, digest in mapping.items())):
            raise ValueError("recording source hash map is malformed")
    clips = protocol.get("clips")
    if (not isinstance(clips, list) or any(not isinstance(clip, dict) for clip in clips)
            or tuple(clip.get("id") for clip in clips) != CLIP_IDS):
        raise ValueError("recording clip order changed")
    for clip in clips:
        expected_role, expected_checkpoint = CLIP_CONTRACT[clip["id"]]
        if (set(clip) != CLIP_KEYS or clip.get("role") != expected_role
                or clip.get("checkpoint") != expected_checkpoint
                or not _safe_component(clip.get("run_basename"))
                or not _safe_component(clip.get("checkpoint"))
                or clip.get("overrides") != FIXED["overrides"]
                or any(not _sha(clip.get(key)) for key in
                       ("checkpoint_sha256", "params_env_sha256", "params_agent_sha256"))):
            raise ValueError(f"recording clip contract changed: {clip.get('id')}")


def _check_artifacts(rows: list[dict[str, str]]) -> None:
    for row in rows:
        if set(row) != {"path", "sha256"}:
            raise ValueError("source artifact binding is malformed")
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("source artifact path escapes the repository")
        M._check_hash(ROOT / relative, row["sha256"], row["path"])


def _check_public_artifacts(tag: str, rows: list[dict[str, str]]) -> None:
    folder = ROOT / "evals/results" / tag
    manifest = H.read(folder / "artifact_manifest.json")
    expected = {"artifact_manifest.json", *manifest["files"]}
    prefix = folder.relative_to(ROOT).as_posix() + "/"
    bound: dict[str, str] = {}
    for row in rows:
        if set(row) != {"path", "sha256"} or not row["path"].startswith(prefix):
            raise ValueError("public source binding is malformed")
        name = row["path"][len(prefix):]
        if name in bound:
            raise ValueError("public source binding is duplicated")
        bound[name] = row["sha256"]
    if set(bound) != expected:
        raise ValueError("public source artifact set changed")
    _check_artifacts(rows)


def _check_private_artifacts(tag: str, rows: list[dict[str, str]]) -> None:
    paths = _private_paths(tag)
    if any(set(row) != {"id", "sha256"} for row in rows):
        raise ValueError("private source binding is malformed")
    bound = {row["id"]: row["sha256"] for row in rows}
    if len(bound) != len(rows) or set(bound) != set(paths):
        raise ValueError("private source artifact set changed")
    for logical_id, path in paths.items():
        M._check_hash(path, bound[logical_id], f"private source {logical_id}")


def _git_blob_lf(commit: str, relative: str) -> str:
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError("source commit is malformed")
    completed = subprocess.run(
        ["git", "-C", str(ROOT), "show", f"{commit}:{relative}"],
        check=True, capture_output=True,
    )
    raw = completed.stdout.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(raw).hexdigest()


def _runtime_fingerprint(isaaclab: Path) -> dict[str, Any]:
    script = (
        "import importlib.metadata as m,json,platform,sys;"
        "print(json.dumps({'runtime':{'implementation':platform.python_implementation(),"
        "'version':list(sys.version_info[:3]),'cache_tag':sys.implementation.cache_tag},"
        "'distributions':{n:m.version(n) for n in "
        "['torch','rsl-rl-lib','isaaclab','isaaclab_tasks','isaaclab_rl']}}))"
    )
    completed = subprocess.run(
        [str(isaaclab / "_isaac_sim/python.bat"), "-c", script], cwd=isaaclab,
        check=True, capture_output=True, text=True, timeout=30,
        creationflags=M.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise ValueError("Isaac Python runtime fingerprint is malformed") from error
    if not isinstance(value, dict):
        raise ValueError("Isaac Python runtime fingerprint is malformed")
    return value


def _check_training_runtime_sources(source_protocol: dict[str, Any], isaaclab: Path) -> None:
    current = H.source_maps(isaaclab)
    for key in ("source_files", "sdk_sources", "rsl_sources"):
        if current[key] != source_protocol[key]:
            raise ValueError(f"training runtime {key} changed")
    expected_versions = source_protocol["resume_inputs"]["runtime_versions"]
    expected_fingerprint = {
        "runtime": source_protocol["correction_source_contract"]["runtime"],
        "distributions": {name: expected_versions[name] for name in
                          ("torch", "rsl-rl-lib", "isaaclab", "isaaclab_tasks", "isaaclab_rl")},
    }
    if _runtime_fingerprint(isaaclab) != expected_fingerprint:
        raise ValueError("Isaac Python runtime distributions changed")


def _verify_frozen_source(protocol: dict[str, Any], isaaclab: Path) -> None:
    """HEAD가 전진한 뒤에도 원 실행 커밋과 고정 artifact를 재검증한다."""
    _check_public_artifacts(protocol["source_tag"], protocol["public_sources"])
    _check_private_artifacts(protocol["source_tag"], protocol["private_sources"])
    public = ROOT / "evals/results" / protocol["source_tag"]
    replay = Q.replay(public)
    public_result = H.read(public / "result.json")
    private_result = H.read(H.folder_for(protocol["source_tag"]) / "result.json")
    numeric = public_result.get("numeric", {})
    if (replay.get("status") != protocol["result_status"]
            or numeric.get("status") != protocol["result_status"]
            or numeric.get("model_selected_correction") is not False
            or private_result.get("status") != protocol["result_status"]
            or public_result.get("source_commit_sha") != protocol["source_commit_sha"]
            or private_result.get("source_commit_sha") != protocol["source_commit_sha"]
            or public_result.get("source_commit_scope") != protocol["source_commit_scope"]
            or public_result.get("publisher_sources") != protocol["publisher_sources"]):
        raise ValueError("frozen correction result binding changed")
    source_protocol_relative = H.PROTOCOL.relative_to(ROOT).as_posix()
    if (H.sha(H.PROTOCOL, True) != protocol["source_protocol_sha256_lf"]
            or _git_blob_lf(protocol["source_commit_sha"], source_protocol_relative)
            != protocol["source_protocol_sha256_lf"]):
        raise ValueError("source protocol is not present at the bound source commit")
    source_protocol = H.read(H.PROTOCOL)
    _check_training_runtime_sources(source_protocol, isaaclab)
    for relative, expected in source_protocol["source_files"].items():
        if _git_blob_lf(protocol["source_commit_sha"], relative) != expected:
            raise ValueError("source implementation differs at the bound source commit")


def verify(tag: str, isaaclab: Path) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    path = protocol_path(tag)
    protocol = H.read(path)
    _validate_protocol_shape(protocol, tag)
    for key, value in FIXED.items():
        if protocol.get(key) != value:
            raise ValueError("recording contract changed")
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise ValueError(f"missing media tool: {tool}")
    _verify_frozen_source(protocol, isaaclab)
    code = {name: M.sha256_lf_file(ROOT / name) for name in sorted(CODE)}
    if protocol.get("code_sha256_lf") != code:
        raise ValueError("recording implementation changed")
    if (protocol.get("play_py_sha256") != M.sha256_file(isaaclab / PLAY)
            or protocol.get("isaaclab_commit") != H.FIXED["isaaclab_commit"]):
        raise ValueError("Isaac playback source changed")
    if P.S._git_head(isaaclab) != H.FIXED["isaaclab_commit"] or not P.S._git_tracked_clean(isaaclab):
        raise ValueError("Isaac Lab checkout changed")
    commit = P.verify_committed(ROOT, CODE | {path.relative_to(ROOT).as_posix()})
    return protocol, _resolve_clips(protocol, isaaclab), commit


def _run_ffmpeg(args: list[str]) -> None:
    subprocess.run(args, check=True, timeout=120, creationflags=M.CREATE_NO_WINDOW if os.name == "nt" else 0,
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def _compose(public: Path) -> tuple[Path, Path]:
    inputs = [public / f"{clip_id}.mp4" for clip_id in CLIP_IDS]
    mp4, gif = public / "comparison.mp4", public / "comparison.gif"
    label = ("fontfile='C\\:/Windows/Fonts/arial.ttf':fontsize=28:fontcolor=white:"
             "box=1:boxcolor=black@0.55:boxborderw=8:x=16:y=16")
    filter_graph = (
        f"[0:v]scale=640:360,drawtext=text='Before':{label}[a];"
        f"[1:v]scale=640:360,drawtext=text='Corrected':{label}[b];"
        f"[2:v]scale=640:360,drawtext=text='Control':{label}[c];"
        "[a][b][c]hstack=inputs=3[v]"
    )
    _run_ffmpeg(["ffmpeg", "-nostdin", "-n", *sum((["-i", str(path)] for path in inputs), []),
                 "-filter_complex", filter_graph, "-map", "[v]", "-an", "-c:v", "libx264",
                 "-pix_fmt", "yuv420p", str(mp4)])
    _run_ffmpeg(["ffmpeg", "-nostdin", "-n", "-i", str(mp4), "-vf",
                 "fps=15,scale=960:-2:flags=lanczos,split[a][b];[a]palettegen=max_colors=128:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=3",
                 "-loop", "0", str(gif)])
    return mp4, gif


def _validate_gif(probe: dict[str, Any]) -> None:
    if (probe["width"] != 960 or probe["height"] != 180
            or probe["frame_count"] != FIXED["gif_frame_count"]
            or not math.isfinite(probe["duration_s"])
            or abs(probe["duration_s"] - FIXED["gif_duration_s"]) > FIXED["gif_duration_tolerance_s"]):
        raise RuntimeError("comparison GIF does not match its frame and duration contract")


def record(tag: str, isaaclab: Path, timeout_s: int = 900) -> dict[str, Any]:
    protocol, inputs, commit = verify(tag, isaaclab)
    if timeout_s < 60:
        raise ValueError("timeout must be at least 60 seconds")
    before = M.query_gpu()
    if before["busy"]:
        return {"status": "gpu_busy_no_launch", "clips_recorded": 0}
    private = ROOT / "bench/private/correction_media" / _tag(tag)
    public = ROOT / "docs/media" / f"go2-correction-{tag}"
    if private.exists() or public.exists():
        raise FileExistsError("media tag is already used")
    private.mkdir(parents=True, exist_ok=False)
    records, gpu_summaries = [], []
    for item in inputs:
        clip_id = item["clip"]["id"]
        clip_before = M.query_gpu()
        if clip_before["busy"]:
            raise RuntimeError(f"GPU became busy before {clip_id}; partial capture remains private")
        M._write_new_json(private / f"gpu_before_{clip_id}.json", clip_before)
        result = M._record_clip(protocol, item, isaaclab, private, timeout_s)
        if result.get("own_pid_cleanup") is not True or result.get("residual_processes_observed") != 0:
            raise RuntimeError(f"owned process cleanup failed for {clip_id}")
        clip_after = M.query_gpu()
        M._write_new_json(private / f"gpu_after_{clip_id}.json", clip_after)
        if clip_after["busy"]:
            raise RuntimeError(f"GPU became busy after {clip_id}; partial capture remains private")
        verify(tag, isaaclab)
        records.append(result)
        gpu_summaries.append({"clip_id": clip_id, "before_busy": False, "after_busy": False})

    staging = private / "public_staging"
    staging.mkdir(parents=True, exist_ok=False)
    clips = []
    for result in records:
        clip = result["clip"]
        target = staging / f"{clip['id']}.mp4"
        shutil.copy2(result["video"], target)
        clips.append({
            "id": clip["id"], "role": clip["role"], "file": target.name,
            "sha256": M.sha256_file(target), "bytes": target.stat().st_size,
            "checkpoint_sha256": clip["checkpoint_sha256"],
            "params_env_sha256": clip["params_env_sha256"],
            "params_agent_sha256": clip["params_agent_sha256"],
            "frame_info": result["probe"], "cleanup_verified": True,
        })
    comparison, gif = _compose(staging)
    comparison_probe = M._probe_video(comparison)
    M._validate_video_contract(comparison_probe, {
        **protocol["video_contract"], "width": 1920, "height": 360,
    })
    gif_probe = M._probe_video(gif)
    # GIF uses centisecond frame delays, so ffprobe's average-rate rational is not
    # guaranteed to print as 15/1 even though the ffmpeg fps filter is fixed at 15.
    _validate_gif(gif_probe)
    manifest = {
        "schema": "training_correction_media_artifact_v1", "source_tag": tag,
        "source_result_status": protocol["result_status"],
        "source_result_wording": protocol["result_wording"],
        "model_selected_correction": False,
        "interpretation": "세 정책을 같은 stock Play 조건에서 재생한 시각 자료다. GIF 자체는 회복 성공 주장이 아니며 판정과 수치는 결합된 fixed_eval 원자료를 따른다.",
        "gif_is_recovery_claim": False, "source_commit_sha": protocol["source_commit_sha"],
        "recording_commit_sha": commit, "protocol": {"path": protocol_path(tag).relative_to(ROOT).as_posix(),
                                                        "sha256": M.sha256_file(protocol_path(tag))},
        "provenance": {"code_sha256_lf": protocol["code_sha256_lf"],
                       "play_py_sha256": protocol["play_py_sha256"],
                       "isaaclab_commit": protocol["isaaclab_commit"],
                       "source_commit_scope": protocol["source_commit_scope"],
                       "publisher_sources": protocol["publisher_sources"],
                       "public_sources": protocol["public_sources"],
                       "private_sources": protocol["private_sources"],
                       "fixed_eval_sha256_lf": protocol["fixed_eval_sha256_lf"]},
        "playback": {key: protocol[key] for key in ("task", "seed", "num_envs", "video_length", "kit_args", "overrides")},
        "gpu_checks": gpu_summaries, "clips": clips,
        "comparison": {"mp4": {"file": comparison.name, "sha256": M.sha256_file(comparison),
                                   "bytes": comparison.stat().st_size, "frame_info": comparison_probe},
                       "gif": {"file": gif.name, "sha256": M.sha256_file(gif), "bytes": gif.stat().st_size,
                               "fps": protocol["gif_fps"], "frame_info": gif_probe}},
    }
    M._write_new_json(staging / "source_manifest.json", manifest)
    public.parent.mkdir(parents=True, exist_ok=True)
    staging.replace(public)
    return {"status": "recorded", "source_result_status": protocol["result_status"],
            "folder": public.relative_to(ROOT).as_posix(), "clips": len(clips)}


def _reason_code(error: Exception, command: str) -> str:
    if isinstance(error, FileExistsError):
        return "output_exists"
    if isinstance(error, (subprocess.SubprocessError, json.JSONDecodeError)):
        return "external_tool_or_artifact_parse_failed"
    message = str(error).casefold()
    rules = (
        (("must be committed", "source commit"), "source_commit_mismatch"),
        (("cleanup", "residual"), "owned_process_cleanup_failed"),
        (("gpu",), "gpu_admission_failed"),
        (("runtime", "distribution", "sdk_sources", "rsl_sources"), "runtime_source_drift"),
        (("clip contract", "clip order", "checkpoint"), "clip_contract_mismatch"),
        (("protocol",), "protocol_mismatch"),
        (("artifact", "source binding", "source implementation"), "artifact_binding_mismatch"),
        (("video", "gif", "ffmpeg", "ffprobe", "media tool"), "media_validation_failed"),
    )
    for terms, code in rules:
        if any(term in message for term in terms):
            return code
    return f"{command}_failed"


def _write_private_rejection(tag: str, command: str, error: Exception, reason_code: str) -> str | None:
    safe_tag = tag if re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,39}", tag or "") else "invalid_tag"
    folder = ROOT / "bench/private/correction_media_rejections" / safe_tag
    rejection_id = uuid.uuid4().hex
    try:
        M._write_new_json(folder / f"{rejection_id}.json", {
            "schema": "training_correction_media_rejection_v1",
            "rejection_id": rejection_id, "command": command, "source_tag": tag,
            "reason_code": reason_code,
            "error_type": type(error).__name__, "detail": str(error),
            "traceback": traceback.format_exc(),
        })
        return rejection_id
    except Exception:
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "record"))
    parser.add_argument("source_tag")
    parser.add_argument("--isaaclab", type=Path, default=Path.home() / "IsaacLab")
    parser.add_argument("--timeout-s", type=int, default=900)
    args = parser.parse_args(argv)
    try:
        result = (prepare(args.source_tag, args.isaaclab.resolve()) if args.command == "prepare"
                  else record(args.source_tag, args.isaaclab.resolve(), args.timeout_s))
    except Exception as error:
        reason_code = _reason_code(error, args.command)
        rejection_id = _write_private_rejection(args.source_tag, args.command, error, reason_code)
        print(json.dumps({"status": "rejected", "reason_code": reason_code,
                          "stage": args.command, "error_type": type(error).__name__,
                          "rejection_id": rejection_id},
                         ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
