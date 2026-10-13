"""검증된 v4 회복 방식 seed의 정책 여섯 개를 같은 Isaac Play 조건으로 녹화한다.

``prepare``는 비공개 실행과 공개 수치 export를 실험 커밋 기준으로 다시 검증한 뒤 녹화
프로토콜만 새로 만든다. 프로토콜과 이 파일을 커밋한 다음에만 ``record``가 GPU를 쓴다.
v3 기록기(record_correction_media.py)는 세 arm에 고정돼 있어 바꾸지 않고 범용 함수만 재사용한다.
프로토콜은 개정판 r2(``..._r2.json``)다. 같은 tag의 r1은 생성 seed가 고정 Play seed를 덮어써 녹화 전에
거부됐고, 파일을 바꾸지 않은 채 r2의 ``supersedes``로 SHA와 이유를 결합한다.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import re
import shutil
import sys
import traceback
from typing import Any
import uuid


ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "bench", ROOT / "evals"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import record_correction_media as V3M  # noqa: E402
import record_evidence_media as M  # noqa: E402
import run_runtime_readback_smoke as P  # noqa: E402
import run_training_correction_v4 as H  # noqa: E402
import publish_training_correction_v4 as Q  # noqa: E402


PLAY = V3M.PLAY
CODE = {
    "bench/record_correction_media_v4.py",
    "bench/record_correction_media.py",
    "bench/record_evidence_media.py",
    "bench/run_runtime_readback_smoke.py",
    "evals/publish_training_correction_v4.py",
    "evals/publish_training_correction.py",
    "evals/run_training_correction_v4.py",
    "evals/run_training_correction.py",
}
CLIP_IDS = ("before_original", "after_resume_corrected", "after_restart_corrected",
            "after_std_reset", "reference_clean_resume", "reference_original_none")
CLIP_CONTRACT = {
    "before_original": ("original_fault_source", "model_299.pt"),
    "after_resume_corrected": ("resume_corrected", "model_598.pt"),
    "after_restart_corrected": ("restart_corrected", "model_299.pt"),
    "after_std_reset": ("resume_corrected_std_reset", "model_598.pt"),
    "reference_clean_resume": ("clean_resume_reference", "model_598.pt"),
    "reference_original_none": ("original_clean_source", "model_299.pt"),
}
CLIP_TITLES = {
    "before_original": "Before F1", "after_resume_corrected": "Resume fixed",
    "after_restart_corrected": "Restart fixed", "after_std_reset": "Resume fixed std reset",
    "reference_clean_resume": "Clean resumed", "reference_original_none": "Clean original",
}
ALLOWED_STATUS = Q.PUBLISHABLE_STATUS
REVISION = 2
FIXED = {
    "schema": "training_correction_media_v4",
    "scope": "verified_recovery_mode_seed_policy_playback",
    # seed는 v3 공통값의 Play 재생 seed(42)다. 원 실행 seed는 생성 키 source_seed로 따로 둔다.
    **deepcopy(V3M.COMMON_FIXED),
    "revision": REVISION,
    "clip_titles": CLIP_TITLES,
    "grid": {"columns": 3, "rows": 2, "cell_width": 640, "cell_height": 360},
    "comparison_size": [1920, 720],
    "gif_size": [960, 360],
}
GENERATED_KEYS = {
    "source_tag", "source_seed", "result_status", "result_labels", "result_wording", "source_commit_sha",
    "source_commit_scope", "publisher_sources", "source_protocol_sha256_lf", "fixed_eval_sha256_lf",
    "code_sha256_lf", "play_py_sha256", "isaaclab_commit", "public_sources", "private_sources", "clips",
    "supersedes",
}
# r1 프로토콜은 생성 seed가 고정 Play seed를 덮어써 verify가 녹화 전에 거부했다. 파일은 그대로 두고 r2가 대체한다.
SUPERSEDED_REASON = ("r1은 생성 키 seed(원 실행 seed)가 고정 키 seed(Play 재생 seed 42)를 덮어써 verify가 "
                     "'recording contract changed'로 녹화 전에 거부했다(GPU 미사용). r2는 원 실행 seed를 source_seed로 둔다")
CLIP_KEYS = V3M.CLIP_KEYS
PRIVATE_FILES = {
    "consumed": "consumed.json", "train_manifest": "train/manifest.json",
    "train_execution": "train/execution.json", "train_initial": "train/initial.json",
    "train_correction": "train/correction.json", "train_receipt": "train/receipt.json",
    "eval_execution": "eval/execution.json", "eval_report": "eval/report.json", "parent": "parent.json",
}


def superseded_protocol_path(tag: str) -> Path:
    return ROOT / "bench" / "protocols" / f"training_correction_media_v4_{V3M._tag(tag)}.json"


def protocol_path(tag: str) -> Path:
    return ROOT / "bench" / "protocols" / f"training_correction_media_v4_{V3M._tag(tag)}_r{REVISION}.json"


def _supersedes(tag: str) -> dict[str, str] | None:
    """같은 tag의 r1 프로토콜이 있으면 경로·SHA·대체 이유를 결합한다. 파일은 바꾸지 않는다."""
    old = superseded_protocol_path(tag)
    if not old.is_file():
        return None
    return {"path": old.relative_to(ROOT).as_posix(), "sha256": M.sha256_file(old), "reason": SUPERSEDED_REASON}


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError as error:
        raise ValueError("source artifact is outside the repository") from error


def _artifact(path: Path) -> dict[str, str]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"missing or unsafe source artifact: {_relative(path)}")
    return {"path": _relative(path), "sha256": M.sha256_file(path)}


def _private_paths(tag: str) -> dict[str, Path]:
    private = H.folder_for(tag)
    paths = {"request": private / "request.json", "approval": private / "approval.json",
             "result": private / "result.json"}
    for arm in H.ARMS:
        for logical, relative in PRIVATE_FILES.items():
            paths[f"{arm}.{logical}"] = private / arm / relative
    return paths


def _private_artifacts(tag: str) -> list[dict[str, str]]:
    return [{"id": logical_id, "sha256": M.sha256_file(path)}
            for logical_id, path in sorted(_private_paths(tag).items())
            if path.is_file() and not path.is_symlink()]


def _validate_result_lineage(*, tag: str, seed: int, source_commit: str, source_protocol_sha: str,
                             status: str, labels: dict, request: dict, public_result: dict,
                             private_result: dict) -> None:
    numeric = public_result.get("numeric")
    if (not isinstance(numeric, dict)
            or any(value.get("tag") != tag or value.get("seed") != seed
                   or value.get("source_commit_sha") != source_commit
                   or value.get("protocol_sha256_lf") != source_protocol_sha
                   for value in (request, public_result, private_result))
            or numeric.get("status") != status or numeric.get("labels") != labels
            or private_result.get("status") != status or private_result.get("labels") != labels):
        raise ValueError("public, private, and request result lineage differs")


def _verified_source(tag: str, isaaclab: Path) -> dict[str, Any]:
    """비공개 실행, 공개 export, 체크포인트를 실험 커밋 기준으로 함께 다시 검증한다."""
    tag = V3M._tag(tag)
    private = H.folder_for(tag)
    public = ROOT / "evals" / "results" / tag
    replay = Q.replay(public)
    public_result = H.read(public / "result.json")
    numeric = public_result.get("numeric", {})
    status, seed, labels = numeric.get("status"), public_result.get("seed"), numeric.get("labels")
    if (seed not in H.SEEDS or status not in ALLOWED_STATUS or replay.get("status") != status
            or numeric.get("completed_arms") != list(H.ARMS)
            or numeric.get("model_selected_correction") is not False):
        raise ValueError("public v4 result is invalid, partially executed, or from another seed")
    source_commit_scope = public_result.get("source_commit_scope")
    publisher_sources = public_result.get("publisher_sources")
    if (source_commit_scope != Q.SOURCE_COMMIT_SCOPE
            or not isinstance(publisher_sources, dict) or not publisher_sources):
        raise ValueError("public publisher source binding is incomplete")
    request = H.read(private / "request.json")
    receipts: dict[str, dict[str, Any]] = {}
    with Q.frozen_source(request.get("source_commit_sha", "")):
        source_protocol, source_commit = H.verify_protocol(isaaclab)
        source_protocol_sha = H.sha(H.PROTOCOL, True)
        private_result = H.read(private / "result.json")
        _validate_result_lineage(tag=tag, seed=seed, source_commit=source_commit,
                                 source_protocol_sha=source_protocol_sha, status=status, labels=labels,
                                 request=request, public_result=public_result, private_result=private_result)
        for arm in H.ARMS:
            training = private / arm / "train"
            receipt = H.verify_training(private, arm, H.read(training / "execution.json"))
            report = private / arm / "eval" / "report.json"
            parent = H.read(private / arm / "parent.json")
            if (not H.execution_valid(H.read(private / arm / "eval" / "execution.json"))
                    or parent.get("status") != "pass"
                    or parent.get("training_receipt_sha256") != H.sha(training / "receipt.json")
                    or parent.get("fixed_report_sha256") != H.sha(report)):
                raise ValueError(f"{arm} execution is partial or invalid")
            fault, correction, mode = H.selection(arm)
            if (receipt.get("fault"), receipt.get("correction_id"), receipt.get("recovery_mode")) != (fault, correction, mode):
                raise ValueError(f"{arm} receipt is bound to another arm selection")
            receipts[arm] = receipt
    clips = []
    for clip_id in CLIP_IDS:
        role, _checkpoint = CLIP_CONTRACT[clip_id]
        if role.startswith("original_"):
            source = H.source_paths(isaaclab, seed, H.FAULT if role == "original_fault_source" else "NONE")
            checkpoint, env, agent = source["checkpoint"], source["environment"], source["agent"]
            row = {"checkpoint_sha256": M.sha256_file(checkpoint), "params_env_sha256": M.sha256_file(env),
                   "params_agent_sha256": M.sha256_file(agent)}
        else:
            receipt = receipts[role]
            checkpoint = Path(receipt["checkpoint"]["path"])
            env, agent = Path(receipt["configs"]["environment"]["path"]), Path(receipt["configs"]["agent"]["path"])
            row = {"checkpoint_sha256": receipt["checkpoint"]["sha256"],
                   "params_env_sha256": receipt["configs"]["environment"]["sha256"],
                   "params_agent_sha256": receipt["configs"]["agent"]["sha256"]}
            for path, key in ((checkpoint, "checkpoint_sha256"), (env, "params_env_sha256"), (agent, "params_agent_sha256")):
                M._check_hash(path, row[key], f"{role} source")
        clips.append({"id": clip_id, "role": role, "run_basename": checkpoint.parent.name,
                      "checkpoint": checkpoint.name, **row, "overrides": deepcopy(FIXED["overrides"])})
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
    return {"tag": tag, "seed": seed, "status": status, "labels": labels,
            "wording": numeric.get("seed_wording"), "source_commit_sha": source_commit,
            "source_commit_scope": source_commit_scope, "publisher_sources": publisher_sources,
            "source_protocol_sha256_lf": source_protocol_sha,
            "fixed_eval_sha256_lf": source_protocol["fixed_eval_sha256_lf"],
            "public_sources": public_sources, "private_sources": private_sources, "clips": clips}


def prepare(tag: str, isaaclab: Path) -> dict[str, Any]:
    source = _verified_source(tag, isaaclab)
    path = protocol_path(tag)
    protocol = {
        **deepcopy(FIXED), "source_tag": source["tag"], "source_seed": source["seed"], "supersedes": _supersedes(tag),
        "result_status": source["status"], "result_labels": source["labels"],
        "result_wording": source["wording"], "source_commit_sha": source["source_commit_sha"],
        "source_commit_scope": source["source_commit_scope"], "publisher_sources": source["publisher_sources"],
        "source_protocol_sha256_lf": source["source_protocol_sha256_lf"],
        "fixed_eval_sha256_lf": source["fixed_eval_sha256_lf"],
        "code_sha256_lf": {name: M.sha256_lf_file(ROOT / name) for name in sorted(CODE)},
        "play_py_sha256": M.sha256_file(isaaclab / PLAY), "isaaclab_commit": H.FIXED["isaaclab_commit"],
        "public_sources": source["public_sources"], "private_sources": source["private_sources"],
        "clips": source["clips"],
    }
    M._write_new_json(path, protocol)
    return {"status": "prepared_commit_before_recording", "protocol": path.relative_to(ROOT).as_posix()}


def _resolve_clips(protocol: dict[str, Any], isaaclab: Path) -> list[dict[str, Any]]:
    private = H.folder_for(protocol["source_tag"])
    seed = protocol["source_seed"]
    locations = {}
    for clip_id in CLIP_IDS:
        role = CLIP_CONTRACT[clip_id][0]
        if role.startswith("original_"):
            source = H.source_paths(isaaclab, seed, H.FAULT if role == "original_fault_source" else "NONE")
            locations[clip_id] = (source["checkpoint"], source["environment"].parent)
        else:
            receipt = H.read(private / role / "train/receipt.json")
            locations[clip_id] = (Path(receipt["checkpoint"]["path"]),
                                  Path(receipt["configs"]["environment"]["path"]).parent)
    return [{"clip": clip, "checkpoint": locations[clip["id"]][0], "params_dir": locations[clip["id"]][1]}
            for clip in protocol["clips"]]


def _validate_protocol_shape(protocol: dict[str, Any], tag: str) -> None:
    if set(protocol) != set(FIXED) | GENERATED_KEYS or protocol.get("source_tag") != tag:
        raise ValueError("recording protocol schema or source tag changed")
    labels = protocol.get("result_labels")
    supersedes = protocol.get("supersedes")
    if supersedes is not None and (
            not isinstance(supersedes, dict) or set(supersedes) != {"path", "sha256", "reason"}
            or supersedes.get("path") != superseded_protocol_path(tag).relative_to(ROOT).as_posix()
            or not V3M._sha(supersedes.get("sha256")) or supersedes.get("reason") != SUPERSEDED_REASON):
        raise ValueError("superseded recording protocol binding is malformed")
    if (protocol.get("source_seed") not in H.SEEDS or protocol.get("result_status") not in ALLOWED_STATUS
            or not isinstance(labels, dict) or set(labels) != set(H.TREATMENT_ARMS)
            or any(value not in H.LABELS for value in labels.values())
            or not isinstance(protocol.get("result_wording"), str) or not protocol["result_wording"]
            or not V3M._sha(protocol.get("source_commit_sha"), 40)
            or protocol.get("source_commit_scope") != Q.SOURCE_COMMIT_SCOPE
            or not V3M._sha(protocol.get("source_protocol_sha256_lf"))
            or not V3M._sha(protocol.get("fixed_eval_sha256_lf"))
            or not V3M._sha(protocol.get("play_py_sha256"))
            or protocol.get("isaaclab_commit") != H.FIXED["isaaclab_commit"]):
        raise ValueError("recording protocol source identity changed")
    for mapping in (protocol.get("publisher_sources"), protocol.get("code_sha256_lf")):
        if (not isinstance(mapping, dict) or not mapping
                or any(not isinstance(name, str) or not V3M._sha(digest) for name, digest in mapping.items())):
            raise ValueError("recording source hash map is malformed")
    clips = protocol.get("clips")
    if (not isinstance(clips, list) or any(not isinstance(clip, dict) for clip in clips)
            or tuple(clip.get("id") for clip in clips) != CLIP_IDS):
        raise ValueError("recording clip order changed")
    for clip in clips:
        expected_role, expected_checkpoint = CLIP_CONTRACT[clip["id"]]
        if (set(clip) != CLIP_KEYS or clip.get("role") != expected_role
                or clip.get("checkpoint") != expected_checkpoint
                or not V3M._safe_component(clip.get("run_basename"))
                or not V3M._safe_component(clip.get("checkpoint"))
                or clip.get("overrides") != FIXED["overrides"]
                or any(not V3M._sha(clip.get(key)) for key in
                       ("checkpoint_sha256", "params_env_sha256", "params_agent_sha256"))):
            raise ValueError(f"recording clip contract changed: {clip.get('id')}")


def _check_artifacts(rows: list[dict[str, str]]) -> None:
    for row in rows:
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


def _check_training_runtime_sources(source_protocol: dict[str, Any], isaaclab: Path) -> None:
    current = H.source_maps(isaaclab)
    for key in ("source_files", "sdk_sources", "rsl_sources"):
        if current[key] != source_protocol[key]:
            raise ValueError(f"training runtime {key} changed")
    versions = source_protocol["resume_inputs"]["runtime_versions"]
    expected = {"runtime": source_protocol["correction_source_contract"]["runtime"],
                "distributions": {name: versions[name] for name in
                                  ("torch", "rsl-rl-lib", "isaaclab", "isaaclab_tasks", "isaaclab_rl")}}
    if V3M._runtime_fingerprint(isaaclab) != expected:
        raise ValueError("Isaac Python runtime distributions changed")


def _verify_frozen_source(protocol: dict[str, Any], isaaclab: Path) -> None:
    """HEAD가 전진한 뒤에도 원 실행 커밋과 고정 artifact를 재검증한다."""
    tag = protocol["source_tag"]
    _check_public_artifacts(tag, protocol["public_sources"])
    _check_private_artifacts(tag, protocol["private_sources"])
    public = ROOT / "evals/results" / tag
    replay = Q.replay(public)
    public_result = H.read(public / "result.json")
    private = H.folder_for(tag)
    _validate_result_lineage(
        tag=tag, seed=protocol["source_seed"], source_commit=protocol["source_commit_sha"],
        source_protocol_sha=protocol["source_protocol_sha256_lf"], status=protocol["result_status"],
        labels=protocol["result_labels"], request=H.read(private / "request.json"),
        public_result=public_result, private_result=H.read(private / "result.json"))
    numeric = public_result.get("numeric", {})
    if (replay.get("status") != protocol["result_status"]
            or numeric.get("completed_arms") != list(H.ARMS)
            or numeric.get("model_selected_correction") is not False
            or numeric.get("seed_wording") != protocol["result_wording"]
            or public_result.get("source_commit_scope") != protocol["source_commit_scope"]
            or public_result.get("publisher_sources") != protocol["publisher_sources"]):
        raise ValueError("frozen correction result binding changed")
    for arm in H.ARMS:
        receipt = H.read(private / arm / "train/receipt.json")
        if (receipt.get("fault"), receipt.get("correction_id"), receipt.get("recovery_mode")) != H.selection(arm):
            raise ValueError("frozen arm selection binding changed")
    for fault in (H.FAULT, "NONE"):
        if H.source_paths(isaaclab, protocol["source_seed"], fault)["checkpoint"].name != "model_299.pt":
            raise ValueError("frozen original checkpoint contract changed")
    relative = H.PROTOCOL.relative_to(ROOT).as_posix()
    if (H.sha(H.PROTOCOL, True) != protocol["source_protocol_sha256_lf"]
            or V3M._git_blob_lf(protocol["source_commit_sha"], relative) != protocol["source_protocol_sha256_lf"]):
        raise ValueError("source protocol is not present at the bound source commit")
    source_protocol = H.read(H.PROTOCOL)
    _check_training_runtime_sources(source_protocol, isaaclab)
    for name, expected in source_protocol["source_files"].items():
        if V3M._git_blob_lf(protocol["source_commit_sha"], name) != expected:
            raise ValueError("source implementation differs at the bound source commit")


def verify(tag: str, isaaclab: Path) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    path = protocol_path(tag)
    protocol = H.read(path)
    _validate_protocol_shape(protocol, tag)
    if any(protocol.get(key) != value for key, value in FIXED.items()):
        raise ValueError("recording contract changed")
    if protocol.get("supersedes") != _supersedes(tag):
        raise ValueError("superseded recording protocol changed or disappeared")
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise ValueError(f"missing media tool: {tool}")
    _verify_frozen_source(protocol, isaaclab)
    if protocol.get("code_sha256_lf") != {name: M.sha256_lf_file(ROOT / name) for name in sorted(CODE)}:
        raise ValueError("recording implementation changed")
    if protocol.get("play_py_sha256") != M.sha256_file(isaaclab / PLAY):
        raise ValueError("Isaac playback source changed")
    if P.S._git_head(isaaclab) != H.FIXED["isaaclab_commit"] or not P.S._git_tracked_clean(isaaclab):
        raise ValueError("Isaac Lab checkout changed")
    commit = P.verify_committed(ROOT, CODE | {path.relative_to(ROOT).as_posix()})
    return protocol, _resolve_clips(protocol, isaaclab), commit


def _compose(public: Path) -> tuple[Path, Path]:
    inputs = [public / f"{clip_id}.mp4" for clip_id in CLIP_IDS]
    mp4, gif = public / "comparison.mp4", public / "comparison.gif"
    label = ("fontfile='C\\:/Windows/Fonts/arial.ttf':fontsize=28:fontcolor=white:"
             "box=1:boxcolor=black@0.55:boxborderw=8:x=16:y=16")
    cells = ";".join(f"[{index}:v]scale=640:360,drawtext=text='{CLIP_TITLES[clip_id]}':{label}[c{index}]"
                     for index, clip_id in enumerate(CLIP_IDS))
    filter_graph = (f"{cells};[c0][c1][c2]hstack=inputs=3[top];[c3][c4][c5]hstack=inputs=3[bottom];"
                    "[top][bottom]vstack=inputs=2[v]")
    V3M._run_ffmpeg(["ffmpeg", "-nostdin", "-n", *sum((["-i", str(path)] for path in inputs), []),
                     "-filter_complex", filter_graph, "-map", "[v]", "-an", "-c:v", "libx264",
                     "-pix_fmt", "yuv420p", str(mp4)])
    V3M._run_ffmpeg(["ffmpeg", "-nostdin", "-n", "-i", str(mp4), "-vf",
                     "fps=15,scale=960:-2:flags=lanczos,split[a][b];[a]palettegen=max_colors=128:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=3",
                     "-loop", "0", str(gif)])
    return mp4, gif


def _validate_gif(probe: dict[str, Any]) -> None:
    if ([probe["width"], probe["height"]] != FIXED["gif_size"]
            or probe["frame_count"] != FIXED["gif_frame_count"]
            or not math.isfinite(probe["duration_s"])
            or abs(probe["duration_s"] - FIXED["gif_duration_s"]) > FIXED["gif_duration_tolerance_s"]):
        raise RuntimeError("comparison GIF does not match its size, frame, and duration contract")


def record(tag: str, isaaclab: Path, timeout_s: int = 900) -> dict[str, Any]:
    protocol, inputs, commit = verify(tag, isaaclab)
    if timeout_s < 60:
        raise ValueError("timeout must be at least 60 seconds")
    if M.query_gpu()["busy"]:
        return {"status": "gpu_busy_no_launch", "clips_recorded": 0}
    private = ROOT / "bench/private/correction_media" / V3M._tag(tag)
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
        clips.append({"id": clip["id"], "role": clip["role"], "title": CLIP_TITLES[clip["id"]],
                      "file": target.name, "sha256": M.sha256_file(target), "bytes": target.stat().st_size,
                      "checkpoint_sha256": clip["checkpoint_sha256"],
                      "params_env_sha256": clip["params_env_sha256"],
                      "params_agent_sha256": clip["params_agent_sha256"],
                      "frame_info": result["probe"], "cleanup_verified": True})
    comparison, gif = _compose(staging)
    comparison_probe = M._probe_video(comparison)
    width, height = FIXED["comparison_size"]
    M._validate_video_contract(comparison_probe, {**protocol["video_contract"], "width": width, "height": height})
    gif_probe = M._probe_video(gif)
    _validate_gif(gif_probe)
    manifest = {
        "schema": "training_correction_media_artifact_v4", "source_tag": tag, "source_seed": protocol["source_seed"],
        "protocol_revision": protocol["revision"], "supersedes": protocol["supersedes"],
        "source_fault": H.FAULT, "source_result_status": protocol["result_status"],
        "source_result_labels": protocol["result_labels"], "source_result_wording": protocol["result_wording"],
        "model_selected_correction": False,
        "interpretation": "여섯 정책을 같은 stock Play 조건에서 재생한 시각 자료다. 영상은 회복 성공 주장이 아니며 판정과 수치는 결합된 fixed_eval 원자료를 따른다.",
        "gif_is_recovery_claim": False, "source_commit_sha": protocol["source_commit_sha"],
        "recording_commit_sha": commit,
        "protocol": {"path": protocol_path(tag).relative_to(ROOT).as_posix(),
                     "sha256": M.sha256_file(protocol_path(tag))},
        "provenance": {key: protocol[key] for key in (
            "code_sha256_lf", "play_py_sha256", "isaaclab_commit", "source_commit_scope",
            "publisher_sources", "public_sources", "private_sources", "fixed_eval_sha256_lf")},
        "playback": {key: protocol[key] for key in ("task", "seed", "num_envs", "video_length", "kit_args", "overrides")},
        "layout": {"grid": protocol["grid"], "order": list(CLIP_IDS)},
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


def _write_private_rejection(tag: str, command: str, error: Exception, reason_code: str) -> str | None:
    safe_tag = tag if re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,39}", tag or "") else "invalid_tag"
    rejection_id = uuid.uuid4().hex
    try:
        M._write_new_json(ROOT / "bench/private/correction_media_rejections" / safe_tag / f"{rejection_id}.json", {
            "schema": "training_correction_media_v4_rejection_v1", "rejection_id": rejection_id,
            "command": command, "source_tag": tag, "reason_code": reason_code,
            "error_type": type(error).__name__, "detail": str(error), "traceback": traceback.format_exc(),
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
        reason_code = V3M._reason_code(error, args.command)
        rejection_id = _write_private_rejection(args.source_tag, args.command, error, reason_code)
        print(json.dumps({"status": "rejected", "reason_code": reason_code, "stage": args.command,
                          "error_type": type(error).__name__, "rejection_id": rejection_id},
                         ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
