"""E4 종단 시연의 정책 세 개를 같은 Isaac Play 조건으로 녹화한다.

before = 시연 F1 실행의 model_299, after = 승인된 restart 교정 재학습(실패 seed + 1000)의 최종 체크포인트,
reference = 같은 seed NONE 실행의 model_299. 시연 판(v1 evals/e2e_reward_demo.py seed 2029, v2
evals/e2e_reward_demo_v2.py seed 2030)은 공개 demo_result.json의 schema로 고른다. ``prepare``는 시연 공개 결과·
비공개 단계 기록·교정 학습 receipt를 시연 실행 커밋 기준으로 다시 검증한 뒤 녹화 프로토콜만 새로 만든다.
프로토콜과 이 파일을 커밋한 다음에만 ``record``가 GPU를 쓴다. 판정이 난 restart 교정(recovered·not_recovered)이
없으면(예: v1 결과 correction_unknown) 녹화하지 않는다. E4 실행 파일과 v3·v4 기록기는 바꾸지 않고 함수만 재사용한다.
회복 판정은 고정 평가 수치로 하며 영상은 판정 근거가 아니다.

  python bench/record_e2e_demo_media.py prepare <demo tag> [--isaaclab <경로>]
  python bench/record_e2e_demo_media.py record <demo tag> [--isaaclab <경로>] [--timeout-s 900]
"""
from __future__ import annotations

import argparse
import contextlib
from copy import deepcopy
import importlib
import json
import math
from pathlib import Path
import re
import shutil
import sys
import traceback
from types import ModuleType, SimpleNamespace
from typing import Any, Iterator
import uuid


ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "bench", ROOT / "evals"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import demo_restart_correction as DR  # noqa: E402
import e2e_reward_demo as E  # noqa: E402
import publish_training_correction_v4 as Q4  # noqa: E402
import record_correction_media as V3M  # noqa: E402
import record_evidence_media as M  # noqa: E402
import run_runtime_readback_smoke as P  # noqa: E402


REPO = ROOT  # 공개·비공개 산출물의 상대 경로 기준. 테스트가 임시 저장소로 바꾼다
PLAY = V3M.PLAY
COMMON_CODE = {
    "bench/record_e2e_demo_media.py",
    "bench/record_correction_media.py",
    "bench/record_evidence_media.py",
    "bench/run_runtime_readback_smoke.py",
}
# 결과 schema → (판, 시연 모듈, restart 교정 모듈, 그 판의 결합 소스). v1은 실행 커밋 고정에 v4 publisher 대역을 쓴다.
DEMOS = {
    "e2e_reward_demo_result_v1": ("v1", "e2e_reward_demo", "demo_restart_correction",
                                  {"evals/e2e_reward_demo.py", "bench/demo_restart_correction.py",
                                   "evals/publish_training_correction_v4.py"}),
    "e2e_reward_demo_result_v2": ("v2", "e2e_reward_demo_v2", "demo_restart_correction_v2",
                                  {"evals/e2e_reward_demo_v2.py", "bench/demo_restart_correction_v2.py"}),
}
VERSIONS = {version: schema for schema, (version, *_rest) in DEMOS.items()}
CLIP_IDS = ("before_fault", "after_restart_fix", "reference_clean")
CLIP_CONTRACT = {
    "before_fault": ("demo_fault_source", "model_299.pt"),
    "after_restart_fix": ("approved_restart_correction", f"model_{DR.FINAL_ITERATION}.pt"),
    "reference_clean": ("demo_clean_source", "model_299.pt"),
}
CLIP_TITLES = {
    "before_fault": "Before - F1 reward fault",
    "after_restart_fix": "After - approved restart fix",
    "reference_clean": "Reference - clean policy",
}
CAPTION = "Same Play settings for all three. Recovery verdict comes from fixed-eval numbers - not from this video."
OVERLAY_TEXT = re.compile(r"[A-Za-z0-9 .\-]+")  # drawtext 이스케이프가 필요 없는 문자만 쓴다
RECORDABLE_OUTCOMES = ("recovered", "not_recovered")
# 승인 표기: 시연 모듈의 APPROVAL 문구를 그대로 쓴다(v1은 같은 뜻의 문구가 모듈에 없어 여기 둔다). 운영자 승인을 사람 승인으로 쓰지 않는다.
V1_APPROVAL = "사용자가 위임한 운영자 승인(승인자 문자열 기록, 신원 인증 아님)"
INTERPRETATION = ("실패 정책·승인된 restart 교정 정책·같은 seed 정상 정책을 같은 stock Play 조건에서 재생한 "
                  "시각 자료다. 회복 판정은 고정 평가 수치로 하며 영상은 판정 근거가 아니다.")
FIXED = {
    "schema": "e2e_demo_media_v1",
    "scope": "e2e_demo_policy_playback_before_after_reference",
    # 이 시연은 Nemotron이 교정을 제안했으므로 v3 공통값의 모델 관련 두 키 대신 녹화 단계의 호출 수만 적는다.
    **{key: value for key, value in deepcopy(V3M.COMMON_FIXED).items()
       if key not in ("model_selected_correction", "model_calls")},
    "recording_model_calls": 0,
    "clip_titles": CLIP_TITLES,
    "caption": CAPTION,
    "grid": {"columns": 3, "rows": 1, "cell_width": 640, "cell_height": 360, "caption_height": 40},
    "comparison_size": [1920, 400],
    "gif_size": [960, 200],
    "verdict_source": ("회복 판정은 시연 고정 평가 수치(demo_result.json의 correction.result)로 한다. "
                       "영상은 같은 Play 조건의 재생 자료이며 판정 근거가 아니다"),
}
SOURCE_KEYS = {
    "source_tag", "demo_version", "demo_seed", "restart_seed", "case", "outcome", "demo_success", "result_wording",
    "correction_result", "correction_proposal", "source_commit_sha", "demo_protocol_sha256_lf",
    "fixed_eval_sha256_lf", "public_sources", "private_sources", "clips",
}
GENERATED_KEYS = SOURCE_KEYS | {"code_sha256_lf", "play_py_sha256", "isaaclab_commit"}
CLIP_KEYS = V3M.CLIP_KEYS
FIX_FILES = (
    "fix/request.json", "fix/approval.json", "fix/consumed.json", "fix/result.json",
    "fix/train/manifest.json", "fix/train/initial.json", "fix/train/correction.json", "fix/train/receipt.json",
    "fix/train/execution.json", "fix/eval/jobs.json", "fix/eval/report.json", "fix/eval/execution.json",
)


def private_files(demo: SimpleNamespace) -> tuple[str, ...]:
    return (*(f"steps/{index:02d}_{name}.json" for index, name in enumerate(demo.E.STEPS)), "result.json", *FIX_FILES)


def code_for(version: str) -> set[str]:
    return COMMON_CODE | DEMOS[VERSIONS[version]][3]


def _modules(schema: str) -> SimpleNamespace:
    version, demo_module, restart_module, _ = DEMOS[schema]
    demo: ModuleType = importlib.import_module(demo_module)
    restart: ModuleType = importlib.import_module(restart_module)
    if restart.FINAL_ITERATION != DR.FINAL_ITERATION or demo.STEPS[-1] != "correction":
        raise ValueError("demo version changed the restart checkpoint or step contract")
    return SimpleNamespace(version=version, schema=schema, E=demo, DR=restart)


def demo_for(tag: str) -> SimpleNamespace:
    """공개 시연 결과의 schema로 시연 판과 모듈을 고른다."""
    path = REPO / "evals/results" / V3M._tag(tag) / "demo_result.json"
    if not path.is_file():
        raise ValueError("public demo result is missing; run the demo summarize step first")
    schema = E.read(path).get("schema")
    if schema not in DEMOS:
        raise ValueError("demo result schema is not a registered demo version")
    return _modules(schema)


def protocol_relative(tag: str) -> str:
    return f"bench/protocols/e2e_demo_media_{V3M._tag(tag)}.json"


def protocol_path(tag: str) -> Path:
    return ROOT / protocol_relative(tag)


def _relative(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(REPO.resolve()).as_posix()
    except ValueError as error:
        raise ValueError("source artifact is outside the repository") from error


def _artifact(path: Path) -> dict[str, str]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"missing or unsafe source artifact: {_relative(path)}")
    return {"path": _relative(path), "sha256": M.sha256_file(path)}


# --- 원 실행 검증 ---
def _check_demo_lineage(demo: SimpleNamespace, tag: str, public_result: dict, private_result: dict,
                        steps: dict[str, dict], step_sha: dict[str, str], fix_result: dict) -> str:
    """공개·비공개 시연 결과, 단계 SHA 사슬, 교정 결과가 한 실행으로 이어지는지 확인하고 시연 실행 커밋을 돌려준다."""
    D = demo.E
    if public_result != private_result:
        raise ValueError("public and private demo results differ")
    result = public_result
    if (result.get("schema") != demo.schema or result.get("tag") != tag
            or result.get("seed") != D.SEED or result.get("case") != D.CASES["F1"] or result.get("terminal") is not None):
        raise ValueError("demo result is from another tag, seed, or case, or ended early")
    if tuple(steps) != D.STEPS or result.get("steps_sha256") != step_sha:
        raise ValueError("demo step artifact set or hashes differ from the result")
    previous = None
    for name in D.STEPS:
        if steps[name].get("step") != name or steps[name].get("previous_sha256") != previous:
            raise ValueError(f"demo step chain is broken at {name}")
        previous = step_sha[name]
    commits = {step.get("source_commit_sha") for step in steps.values()}
    protocols = {step.get("protocol_sha256_lf") for step in steps.values()}
    commit = next(iter(commits))
    if (len(commits) != 1 or result.get("source_commit_sha") != [commit] or not V3M._sha(commit, 40)
            or len(protocols) != 1 or result.get("protocol_sha256_lf") not in protocols):
        raise ValueError("demo steps were not run from one committed protocol and source")
    correction = steps["correction"]
    public_fix = (result.get("correction") or {}).get("result") or {}
    if (correction.get("result") != fix_result or correction.get("receipt", {}).get("verdict") != result["outcome"]
            or fix_result.get("label") != result["outcome"] or public_fix.get("label") != fix_result.get("label")
            or public_fix.get("standstill_ratio") != fix_result.get("standstill_ratio")
            or fix_result.get("tag") != demo.DR.fix_tag(tag) or fix_result.get("case") != D.CASES["F1"]
            or fix_result.get("restart_seed") != D.RESTART_SEED or fix_result.get("reference_seed") != D.SEED
            or fix_result.get("correction_request_id") != correction.get("request_id")
            or fix_result.get("source_commit_sha") != commit):
        raise ValueError("restart correction result lineage differs")
    return commit


@contextlib.contextmanager
def _demo_source_commit(module: ModuleType, commit: str) -> Iterator[None]:
    """v1 시연 모듈의 커밋 확인을 v4 publisher 대역으로 바꾼다. HEAD가 전진해도 결합 파일이 같으면 실행 커밋을 돌려준다."""
    if not V3M._sha(commit, 40):
        raise ValueError("demo source commit is malformed")
    original = module.P
    module.P = Q4._FrozenCommit(original, commit)
    try:
        yield
    finally:
        module.P = original


def _verify_demo_execution(demo: SimpleNamespace, tag: str, isaaclab: Path, commit: str) -> tuple[dict, dict]:
    """시연 프로토콜과 승인 교정 학습 receipt를 시연 실행 커밋 기준으로 다시 검증한다.

    v1은 실행 커밋 고정이 없어 v4 publisher 대역으로 감싸고, v2는 verify_protocol(isaaclab, 실험 커밋)을 쓴다."""
    D, R = demo.E, demo.DR
    frozen = _demo_source_commit(D, commit) if demo.version == "v1" else contextlib.nullcontext()
    with frozen:
        protocol, verified = D.verify_protocol(isaaclab) if demo.version == "v1" else D.verify_protocol(isaaclab, commit)
        if verified != commit:
            raise ValueError("demo protocol was not verified at the demo source commit")
        folder = R.fix_folder(tag)
        none_report = D.read(REPO / D.step_record(tag, "evaluate")["reports"]["NONE"]["path"])
        receipt = R.verify_training(folder, D.read(folder / "train/execution.json"), protocol, none_report["versions"])
    if not R.execution_valid(D.read(folder / "eval/execution.json")):
        raise ValueError("restart correction fixed evaluation execution is invalid")
    return protocol, receipt


def _clip(clip_id: str, checkpoint: dict, environment: dict, agent: dict) -> tuple[dict, tuple[Path, Path]]:
    role, checkpoint_name = CLIP_CONTRACT[clip_id]
    paths = {key: Path(row["path"]).resolve()
             for key, row in (("checkpoint", checkpoint), ("env", environment), ("agent", agent))}
    params = paths["checkpoint"].parent / "params"
    if (paths["checkpoint"].name != checkpoint_name or paths["env"] != params / "env.yaml"
            or paths["agent"] != params / "agent.yaml"):
        raise ValueError(f"recording clip contract changed: {clip_id} checkpoint or params location")
    row = {"checkpoint_sha256": checkpoint["sha256"], "params_env_sha256": environment["sha256"],
           "params_agent_sha256": agent["sha256"]}
    for key, name in (("checkpoint", "checkpoint_sha256"), ("env", "params_env_sha256"), ("agent", "params_agent_sha256")):
        M._check_hash(paths[key], row[name], f"{clip_id} source")
    clip = {"id": clip_id, "role": role, "run_basename": paths["checkpoint"].parent.name,
            "checkpoint": paths["checkpoint"].name, **row, "overrides": deepcopy(FIXED["overrides"])}
    return clip, (paths["checkpoint"], params)


def _public_sources(demo: SimpleNamespace, tag: str) -> list[dict[str, str]]:
    folder = demo.E.RESULTS / tag
    files = sorted(path for path in folder.rglob("*") if path.is_file())
    if folder / "demo_result.json" not in files:
        raise ValueError("public demo result is missing")
    return [_artifact(path) for path in files]


def _private_sources(demo: SimpleNamespace, tag: str) -> list[dict[str, str]]:
    folder = demo.E.folder_for(tag)
    rows = []
    for relative in private_files(demo):
        path = folder / relative
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"private demo artifact set is incomplete: {relative}")
        rows.append({"id": relative, "sha256": M.sha256_file(path)})
    return rows


def _verified_source(tag: str, isaaclab: Path) -> dict[str, Any]:
    """시연 결과·단계·교정 학습·체크포인트를 다시 검증하고 녹화 결합값과 실제 위치를 돌려준다."""
    tag = V3M._tag(tag)
    demo = demo_for(tag)
    D, R = demo.E, demo.DR
    folder = D.folder_for(tag)
    if (folder / "terminal.json").exists():
        raise ValueError("demo ended early; there is no approved restart policy to record")
    steps, step_sha = {}, {}
    for name in D.STEPS:
        steps[name], step_sha[name] = D.step_record(tag, name), M.sha256_file(D.step_path(tag, name))
    public_result = D.read(D.RESULTS / tag / "demo_result.json")
    private_result = D.read(folder / "result.json")
    if public_result.get("outcome") not in RECORDABLE_OUTCOMES:
        raise ValueError(f"demo did not reach a judged restart correction (outcome {public_result.get('outcome')})")
    fix_result = D.read(R.fix_folder(tag) / "result.json")
    commit = _check_demo_lineage(demo, tag, public_result, private_result, steps, step_sha, fix_result)
    if (fix_result.get("training_receipt_sha256") != M.sha256_file(R.fix_folder(tag) / "train/receipt.json")
            or fix_result.get("fixed_report_sha256") != M.sha256_file(R.fix_folder(tag) / "eval/report.json")):
        raise ValueError("restart correction result is not bound to its receipt and report")
    protocol, receipt = _verify_demo_execution(demo, tag, isaaclab, commit)
    clips, locations = [], {}
    originals = {arm: D._check_artifacts(tag, arm) for arm in ("F1", "NONE")}
    for clip_id, row in (("before_fault", originals["F1"]),
                         ("after_restart_fix", {"checkpoint": receipt["checkpoint"], **receipt["configs"]}),
                         ("reference_clean", originals["NONE"])):
        clip, locations[clip_id] = _clip(clip_id, row["checkpoint"], row["environment"], row["agent"])
        clips.append(clip)
    proposal = public_result["correction"]["model_proposal"]
    return {
        "source_tag": tag, "demo_version": demo.version, "demo_seed": D.SEED, "restart_seed": D.RESTART_SEED,
        "case": D.CASES["F1"], "outcome": public_result["outcome"], "demo_success": public_result["demo_success"],
        "result_wording": public_result["wording"],
        "correction_result": {key: public_result["correction"]["result"][key] for key in ("label", "standstill_ratio")},
        "correction_proposal": {"decision": proposal.get("decision"), "accepted": proposal.get("accepted"),
                                "agrees_with_rule": proposal.get("agrees_with_rule"),
                                "unmet_checklist_count": len(proposal.get("unmet_checklist") or []),
                                "operator_approved": steps["correction"]["receipt"].get("event") == "receipt"},
        "source_commit_sha": commit, "demo_protocol_sha256_lf": public_result["protocol_sha256_lf"],
        "fixed_eval_sha256_lf": protocol["fixed_eval_sha256_lf"],
        "public_sources": _public_sources(demo, tag), "private_sources": _private_sources(demo, tag), "clips": clips,
        "_locations": locations,
    }


def build_protocol(tag: str, isaaclab: Path) -> dict[str, Any]:
    source = _verified_source(tag, isaaclab)
    demo = _modules(VERSIONS[source["demo_version"]])
    protocol = {**deepcopy(FIXED), **{key: source[key] for key in SOURCE_KEYS},
                "code_sha256_lf": {name: M.sha256_lf_file(ROOT / name) for name in sorted(code_for(demo.version))},
                "play_py_sha256": M.sha256_file(isaaclab / PLAY), "isaaclab_commit": demo.E.FIXED["isaaclab_commit"]}
    _validate_protocol_shape(protocol, source["source_tag"])
    return protocol


def prepare(tag: str, isaaclab: Path) -> dict[str, Any]:
    M._write_new_json(protocol_path(tag), build_protocol(tag, isaaclab))
    return {"status": "prepared_commit_before_recording", "protocol": protocol_relative(tag)}


def _validate_protocol_shape(protocol: dict[str, Any], tag: str) -> None:
    if set(protocol) != set(FIXED) | GENERATED_KEYS or protocol.get("source_tag") != tag:
        raise ValueError("recording protocol schema or source tag changed")
    if protocol.get("demo_version") not in VERSIONS:
        raise ValueError("recording protocol source identity changed: demo version")
    D = _modules(VERSIONS[protocol["demo_version"]]).E
    if (protocol.get("demo_seed") != D.SEED or protocol.get("restart_seed") != D.RESTART_SEED
            or protocol.get("case") != D.CASES["F1"] or protocol.get("outcome") not in RECORDABLE_OUTCOMES
            or not isinstance(protocol.get("result_wording"), str) or not protocol["result_wording"]
            or not V3M._sha(protocol.get("source_commit_sha"), 40)
            or not V3M._sha(protocol.get("demo_protocol_sha256_lf"))
            or not V3M._sha(protocol.get("fixed_eval_sha256_lf"))
            or not V3M._sha(protocol.get("play_py_sha256"))
            or protocol.get("isaaclab_commit") != D.FIXED["isaaclab_commit"]):
        raise ValueError("recording protocol source identity changed")
    code = protocol.get("code_sha256_lf")
    if (not isinstance(code, dict) or set(code) != code_for(protocol["demo_version"])
            or any(not V3M._sha(digest) for digest in code.values())):
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
                or clip.get("overrides") != FIXED["overrides"]
                or any(not V3M._sha(clip.get(key)) for key in
                       ("checkpoint_sha256", "params_env_sha256", "params_agent_sha256"))):
            raise ValueError(f"recording clip contract changed: {clip.get('id')}")


def verify(tag: str, isaaclab: Path) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    path = protocol_path(tag)
    protocol = E.read(path)
    _validate_protocol_shape(protocol, tag)
    if any(protocol.get(key) != value for key, value in FIXED.items()):
        raise ValueError("recording contract changed")
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise ValueError(f"missing media tool: {tool}")
    source = _verified_source(tag, isaaclab)
    changed = sorted(key for key in SOURCE_KEYS if protocol[key] != source[key])
    if changed:
        raise ValueError(f"recording source binding changed: {changed}")
    code = code_for(protocol["demo_version"])
    if protocol["code_sha256_lf"] != {name: M.sha256_lf_file(ROOT / name) for name in sorted(code)}:
        raise ValueError("recording implementation changed")
    if protocol["play_py_sha256"] != M.sha256_file(isaaclab / PLAY):
        raise ValueError("Isaac playback source changed")
    if P.S._git_head(isaaclab) != protocol["isaaclab_commit"] or not P.S._git_tracked_clean(isaaclab):
        raise ValueError("Isaac Lab checkout changed")
    commit = P.verify_committed(ROOT, code | {protocol_relative(tag)})
    inputs = [{"clip": clip, "checkpoint": source["_locations"][clip["id"]][0],
               "params_dir": source["_locations"][clip["id"]][1]} for clip in protocol["clips"]]
    return protocol, inputs, commit


# --- 합성 ---
def filter_graph() -> str:
    for text in (*CLIP_TITLES.values(), CAPTION):
        if not OVERLAY_TEXT.fullmatch(text):
            raise ValueError("overlay text needs drawtext escaping")
    font = "fontfile='C\\:/Windows/Fonts/arial.ttf'"
    label = f"{font}:fontsize=28:fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=8:x=16:y=16"
    grid = FIXED["grid"]
    width, height = FIXED["comparison_size"]
    caption = f"{font}:fontsize=22:fontcolor=white:x=(w-text_w)/2:y={grid['cell_height']}+({grid['caption_height']}-text_h)/2"
    cells = ";".join(f"[{index}:v]scale={grid['cell_width']}:{grid['cell_height']},drawtext=text='{CLIP_TITLES[clip_id]}':{label}[c{index}]"
                     for index, clip_id in enumerate(CLIP_IDS))
    return (f"{cells};[c0][c1][c2]hstack=inputs=3[row];"
            f"[row]pad={width}:{height}:0:0:color=black,drawtext=text='{CAPTION}':{caption}[v]")


def _compose(folder: Path) -> tuple[Path, Path]:
    inputs = [folder / f"{clip_id}.mp4" for clip_id in CLIP_IDS]
    mp4, gif = folder / "comparison.mp4", folder / "comparison.gif"
    V3M._run_ffmpeg(["ffmpeg", "-nostdin", "-n", *sum((["-i", str(path)] for path in inputs), []),
                     "-filter_complex", filter_graph(), "-map", "[v]", "-an", "-c:v", "libx264",
                     "-pix_fmt", "yuv420p", str(mp4)])
    V3M._run_ffmpeg(["ffmpeg", "-nostdin", "-n", "-i", str(mp4), "-vf",
                     f"fps={FIXED['gif_fps']},scale={FIXED['gif_size'][0]}:-2:flags=lanczos,split[a][b];"
                     "[a]palettegen=max_colors=128:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=3",
                     "-loop", "0", str(gif)])
    return mp4, gif


def _validate_outputs(mp4_probe: dict[str, Any], gif_probe: dict[str, Any], video_contract: dict[str, Any]) -> None:
    width, height = FIXED["comparison_size"]
    M._validate_video_contract(mp4_probe, {**video_contract, "width": width, "height": height})
    if ([gif_probe["width"], gif_probe["height"]] != FIXED["gif_size"]
            or gif_probe["frame_count"] != FIXED["gif_frame_count"]
            or not math.isfinite(gif_probe["duration_s"])
            or abs(gif_probe["duration_s"] - FIXED["gif_duration_s"]) > FIXED["gif_duration_tolerance_s"]):
        raise RuntimeError("comparison GIF does not match its size, frame, and duration contract")


def record(tag: str, isaaclab: Path, timeout_s: int = 900) -> dict[str, Any]:
    protocol, inputs, commit = verify(tag, isaaclab)
    if timeout_s < 60:
        raise ValueError("timeout must be at least 60 seconds")
    if M.query_gpu()["busy"]:
        return {"status": "gpu_busy_no_launch", "clips_recorded": 0}
    private = ROOT / "bench/private/e2e_demo_media" / V3M._tag(tag)
    public = ROOT / "docs/media" / f"go2-e2e-{tag}"
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
    comparison_probe, gif_probe = M._probe_video(comparison), M._probe_video(gif)
    _validate_outputs(comparison_probe, gif_probe, protocol["video_contract"])
    by_id = {clip["id"]: clip["checkpoint_sha256"] for clip in protocol["clips"]}
    manifest = {
        "schema": "e2e_demo_media_artifact_v1", "source_tag": tag, "demo_seed": protocol["demo_seed"],
        "restart_seed": protocol["restart_seed"], "case": protocol["case"], "source_fault": "F1",
        "outcome": protocol["outcome"], "demo_success": protocol["demo_success"],
        "result_wording": protocol["result_wording"], "correction_result": protocol["correction_result"],
        "correction_proposal": protocol["correction_proposal"],
        "interpretation": INTERPRETATION,
        "video_is_recovery_claim": False,
        "approval": getattr(_modules(VERSIONS[protocol["demo_version"]]).E, "APPROVAL", V1_APPROVAL),
        "after_checkpoint_equals_reference": by_id["after_restart_fix"] == by_id["reference_clean"],
        "source_commit_sha": protocol["source_commit_sha"], "recording_commit_sha": commit,
        "protocol": {"path": protocol_relative(tag), "sha256": M.sha256_file(protocol_path(tag))},
        "provenance": {key: protocol[key] for key in (
            "code_sha256_lf", "play_py_sha256", "isaaclab_commit", "demo_protocol_sha256_lf",
            "fixed_eval_sha256_lf", "public_sources", "private_sources")},
        "playback": {key: protocol[key] for key in ("task", "seed", "num_envs", "video_length", "kit_args", "overrides")},
        "layout": {"grid": protocol["grid"], "order": list(CLIP_IDS), "caption": protocol["caption"]},
        "gpu_checks": gpu_summaries, "clips": clips,
        "comparison": {"mp4": {"file": comparison.name, "sha256": M.sha256_file(comparison),
                               "bytes": comparison.stat().st_size, "frame_info": comparison_probe},
                       "gif": {"file": gif.name, "sha256": M.sha256_file(gif), "bytes": gif.stat().st_size,
                               "fps": protocol["gif_fps"], "frame_info": gif_probe}},
    }
    M._write_new_json(staging / "source_manifest.json", manifest)
    public.parent.mkdir(parents=True, exist_ok=True)
    staging.replace(public)
    return {"status": "recorded", "outcome": protocol["outcome"],
            "folder": public.relative_to(ROOT).as_posix(), "clips": len(clips)}


def _write_private_rejection(tag: str, command: str, error: Exception, reason_code: str) -> str | None:
    safe_tag = tag if re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,39}", tag or "") else "invalid_tag"
    rejection_id = uuid.uuid4().hex
    try:
        M._write_new_json(ROOT / "bench/private/e2e_demo_media_rejections" / safe_tag / f"{rejection_id}.json", {
            "schema": "e2e_demo_media_rejection_v1", "rejection_id": rejection_id,
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
