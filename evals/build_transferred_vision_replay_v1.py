"""전송 녹화와 E2E 교정 결과를 한 화면의 오프라인 재생 HTML로 만든다.

공개 비교 3파일을 항상 교차 검증한다. E2E manifest가 pending이면 결과나 미디어를
추정하지 않고 대기 상태만 표시한다. GPU·모델·네트워크를 사용하지 않는다.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
COMPARISON = Path("evals/results/vision_transferred_selection_dev_20261027_v1")
COMPARISON_FILES = ("summary.json", "model-selections.json", "probe-pairs.json")
HEX64 = re.compile(r"[0-9a-f]{64}")
SEEDS = (2220, 2221)
VIEWS = ("text_only", "video_only", "text_and_video")
SCHEMA = "transferred_vision_replay_e2e_manifest_v1"


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value or not value:
        raise ValueError("공개 입력 경로는 저장소 상대경로여야 한다")
    return path


def _bound_file(root: Path, ref: dict, *, suffixes: tuple[str, ...] | None = None) -> Path:
    if set(ref) != {"path", "sha256"} or not HEX64.fullmatch(str(ref["sha256"])):
        raise ValueError("파일 참조 형식이 다르다")
    relative = _relative(ref["path"])
    if suffixes is not None and relative.suffix.lower() not in suffixes:
        raise ValueError("파일 확장자가 허용 목록과 다르다")
    path = root / relative
    if not path.is_file() or path.is_symlink() or _sha(path) != ref["sha256"]:
        raise ValueError("파일 참조의 바이트 결합이 다르다")
    return path


def validate_comparison(root: Path = ROOT) -> dict:
    folder = root / COMPARISON
    summary, choices, pairs = (_read(folder / name) for name in COMPARISON_FILES)
    if (summary.get("schema") != "vision_transferred_selection_public_v1"
            or summary.get("status") != "complete"
            or summary.get("measurement_scope") != "registered_f1_two_seed_development_comparison"
            or summary.get("selection_outcome") != "observation_only"
            or summary.get("post_correction_e2e") != "not_part_of_this_comparison"):
        raise ValueError("비교 요약의 범위가 다르다")
    comparison = summary.get("comparison", {})
    if (comparison.get("selection_outcome") != "observation_only"
            or comparison.get("validity", summary.get("validity")) not in (None, summary.get("validity"))
            or summary.get("validity", {}).get("valid") != 24
            or summary.get("validity", {}).get("denominator") != 24
            or comparison.get("secondary_pairwise", {}).get("video_only_vs_text_only", {}).get("different_probe_or_hold") != 0
            or comparison.get("secondary_pairwise", {}).get("text_and_video_vs_video_only", {}).get("different_probe_or_hold") != 0):
        raise ValueError("비교 선택 결과가 다르다")
    if not isinstance(choices, list) or len(choices) != 24:
        raise ValueError("모델 선택 수가 다르다")
    if any(row.get("model_status") != "valid" or row.get("probe") != "P_reward"
           or row.get("view_kind") not in VIEWS for row in choices):
        raise ValueError("모델 선택 계약이 다르다")
    if {row["terminal_sha256"] for row in choices} != set(comparison.get("bindings", {}).get("terminal_sha256", [])):
        raise ValueError("선택 원문 해시 결합이 다르다")
    if set(pairs) != {"source-01", "source-02"} or pairs != summary.get("probe_pairs"):
        raise ValueError("검사 pair 결합이 다르다")
    by_seed = {}
    for index, group in enumerate(("source-01", "source-02")):
        row = pairs[group]
        seed = SEEDS[index]
        if (row.get("schema") != "vision_transferred_probe_pair_result_v1"
                or row.get("status") != "complete" or row.get("seed") != seed
                or row.get("probe") != "P_reward" or row.get("usable") is not True
                or row.get("outcomes") != {"F1": "abnormal", "NONE": "normal"}):
            raise ValueError("검사 pair 결과가 다르다")
        by_seed[seed] = row
    flags = summary.get("source_recordings", {}).get("source_flags_preserved", {})
    if flags != {"capture_approved": False, "model_ready": False, "used_for_decision": False}:
        raise ValueError("원본 녹화 사용 범위가 다르다")
    return {
        "summary": summary,
        "choices": choices,
        "pairs": by_seed,
        "files": {name: {"path": (COMPARISON / name).as_posix(), "sha256": _sha(folder / name)}
                  for name in COMPARISON_FILES},
    }


def _metric(document: dict, name: str) -> float:
    metrics = document.get("metrics", {})
    value = metrics.get(name)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"고정 평가 지표가 숫자가 아니다: {name}")
    return float(value)


def _contains(value: Any, expected: str) -> bool:
    if value == expected:
        return True
    if isinstance(value, dict):
        return any(_contains(child, expected) for child in value.values())
    if isinstance(value, list):
        return any(_contains(child, expected) for child in value)
    return False


def _post_observation(public_documents: list[dict], seed: int) -> dict | None:
    candidates = [document for document in public_documents
                  if document.get("schema") == "vision_transferred_e2e_seed_public_v1"]
    if not candidates:
        return None
    if len(candidates) != 1:
        raise ValueError("seed 공개 결과가 하나가 아니다")
    document = candidates[0]
    if document.get("seed") != seed or document.get("status") != "recovered":
        raise ValueError("seed 공개 결과의 identity가 다르다")
    result = document.get("post_observation")
    if not isinstance(result, dict):
        raise TypeError("후속 관찰 결과가 없다")
    observation = result.get("observation")
    if (result.get("used_for_decision") is not False
            or result.get("execution_location") != "local_wsl_after_brev_stop"):
        raise ValueError("후속 관찰 판정 계약이 다르다")
    if result.get("status") == "observed_HOLD":
        if observation is not None or not isinstance(result.get("reason"), str) or not result["reason"]:
            raise ValueError("후속 관찰 HOLD 계약이 다르다")
        return result
    validity = result.get("response_validity")
    if (result.get("status") != "observation_ready"
            or not isinstance(validity, list) or not validity or validity[-1] != "valid"
            or not isinstance(observation, dict)
            or any(observation.get(name) not in {"yes", "no", "unclear"}
                   for name in ("walking", "in_place", "falling"))):
        raise ValueError("후속 관찰 판정 계약이 다르다")
    expected = {f"f{index:02d}": float(index) for index in range(1, 7)}
    evidence = observation.get("evidence")
    if (not isinstance(evidence, dict)
            or set(evidence) != {"walking", "in_place", "falling"}
            or any(not isinstance(citations, list) or not citations
                   or any(not isinstance(item, dict) or set(item) != {"frame_id", "timestamp_s"}
                          or item.get("frame_id") not in expected
                          or type(item.get("timestamp_s")) not in {float, int}
                          or item["timestamp_s"] != expected[item["frame_id"]]
                          for item in citations)
                   for citations in evidence.values())):
        raise ValueError("후속 관찰 프레임 근거가 다르다")
    return result


def validate_e2e(manifest: dict, root: Path = ROOT) -> dict:
    fixed = {
        "schema": SCHEMA,
        "measurement_scope": "registered_f1_two_seed_development_e2e",
        "fault_novelty": "registered_f1_not_unseen_fault",
        "selection_outcome": "observation_only",
        "selection_change_count": 0,
        "used_for_decision": False,
    }
    if any(manifest.get(key) != value for key, value in fixed.items()):
        raise ValueError("E2E manifest 고정 범위가 다르다")
    if manifest.get("status") == "pending":
        if manifest.get("seeds") != []:
            raise ValueError("pending manifest에는 seed 결과를 넣을 수 없다")
        return {"status": "pending", "seeds": []}
    if manifest.get("status") != "complete":
        raise ValueError("E2E manifest 상태가 다르다")
    rows = manifest.get("seeds")
    if not isinstance(rows, list) or [row.get("seed") for row in rows] != list(SEEDS):
        raise ValueError("E2E seed 순서가 다르다")
    verified = []
    for row in rows:
        for name in ("source_fault_checkpoint_sha256", "corrected_checkpoint_sha256"):
            if not HEX64.fullmatch(str(row.get(name))):
                raise ValueError("checkpoint SHA가 다르다")
        if (row.get("status") != "recovered"
                or row.get("selection") != {"probe": "P_reward", "model_status": "valid", "used_for_decision": True}
                or row.get("probe") != {"name": "P_reward", "outcome": "abnormal"}
                or row.get("correction") != {"id": "restore_body_frame_reward", "recovery_mode": "restart", "registered": True}):
            raise ValueError("E2E 고리 결과가 다르다")
        fixed_eval = row.get("fixed_eval", {})
        if fixed_eval.get("outcome") != "recovered":
            raise ValueError("고정 평가 회복 결과가 다르다")
        artifact = _bound_file(root, fixed_eval.get("artifact", {}), suffixes=(".json",))
        artifact_doc = _read(artifact)
        for metric in ("lin_vel_rmse_mps", "fall_rate"):
            declared = fixed_eval.get(metric)
            if not isinstance(declared, (int, float)) or isinstance(declared, bool) or float(declared) != _metric(artifact_doc, metric):
                raise ValueError("고정 평가 수치 결합이 다르다")
        videos = row.get("videos", {})
        if set(videos) != {"fault", "corrected"}:
            raise ValueError("영상 역할이 다르다")
        for ref in videos.values():
            _bound_file(root, ref, suffixes=(".mp4",))
        frames = row.get("frame_evidence")
        if not isinstance(frames, list) or {item.get("phase") for item in frames} != {"fault", "corrected"}:
            raise ValueError("두 단계 프레임 근거가 필요하다")
        for item in frames:
            if (set(item) != {"phase", "path", "sha256", "frame_index", "pts_s"}
                    or not isinstance(item["frame_index"], int) or item["frame_index"] < 0
                    or not isinstance(item["pts_s"], (int, float)) or item["pts_s"] < 0):
                raise ValueError("프레임 근거 형식이 다르다")
            _bound_file(root, {"path": item["path"], "sha256": item["sha256"]}, suffixes=(".png", ".jpg", ".jpeg"))
        refs = row.get("public_result_refs")
        if not isinstance(refs, list) or not refs:
            raise ValueError("공개 결과 참조가 없다")
        public_documents = [_read(_bound_file(root, ref, suffixes=(".json",))) for ref in refs]
        for checkpoint_key in ("source_fault_checkpoint_sha256", "corrected_checkpoint_sha256"):
            if not any(_contains(document, row[checkpoint_key]) for document in public_documents):
                raise ValueError("공개 결과가 checkpoint SHA를 결합하지 않는다")
        verified.append({**row, "post_observation": _post_observation(public_documents, row["seed"])})
    return {"status": "complete", "seeds": verified}


def _web_path(root: Path, output: Path, relative: str) -> str:
    return Path(os.path.relpath(root / _relative(relative), (root / output).parent)).as_posix()


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _script_json(value: Any) -> str:
    """raw script text로 읽을 수 있는 JSON을 만들고 닫힘 태그 시작을 막는다."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")


def _seed_card(seed: int, pair: dict, row: dict | None, root: Path, output: Path) -> str:
    if row is None:
        error = pair["measurements"]["F1"]["track_lin_vel_xy_exp_rel_error"]
        return f"""
        <article class="seed-card pending">
          <header><span>seed {seed}</span><strong>E2E 결과 대기</strong></header>
          <div class="timeline compact">
            <div><b>관찰</b><span>등록 F1 개발 영상</span></div>
            <div><b>선택</b><span>P_reward</span></div>
            <div><b>비교 단계 검사</b><span>abnormal · 상대오차 {_escape(f'{error:.3f}')}</span></div>
            <div class="pending-step"><b>교정 이후</b><span>공개 export가 생기기 전에는 표시하지 않음</span></div>
          </div>
        </article>"""
    videos = row["videos"]
    frames = {phase: next(item for item in row["frame_evidence"] if item["phase"] == phase)
              for phase in ("fault", "corrected")}
    post = row["post_observation"]
    if post is None:
        post_text, post_state = "보조 관찰 미결합", "used_for_decision=false"
    elif post["status"] == "observed_HOLD":
        post_text, post_state = "관찰 보류", "HOLD · 판정 미사용"
    else:
        observation = post["observation"]
        post_text = (f"walking {observation['walking']} · in-place {observation['in_place']}"
                     f" · falling {observation['falling']}")
        post_state = "valid · 판정 미사용"
    return f"""
      <article class="seed-card">
        <header><span>seed {seed}</span><strong>recovered</strong></header>
        <div class="media-row">
          <figure><video muted loop controls preload="metadata" src="{_escape(_web_path(root, output, videos['fault']['path']))}"></video>
            <figcaption>관찰 · F1</figcaption></figure>
          <figure><video muted loop controls preload="metadata" src="{_escape(_web_path(root, output, videos['corrected']['path']))}"></video>
            <figcaption>후속 관찰 · corrected</figcaption></figure>
        </div>
        <div class="timeline">
          <div><b>관찰</b><img src="{_escape(_web_path(root, output, frames['fault']['path']))}" alt="seed {seed} F1 프레임"></div>
          <div><b>선택</b><span>P_reward</span></div>
          <div><b>검사</b><span>새 E2E 확인 검사 · abnormal</span></div>
          <div><b>교정</b><span>등록 보상 복원 · restart</span></div>
          <div><b>고정 평가</b><span>RMSE {_escape(f"{row['fixed_eval']['lin_vel_rmse_mps']:.3f}")} m/s · 낙상률 {_escape(f"{row['fixed_eval']['fall_rate']:.3f}")}</span></div>
          <div class="observation-step"><b>후속 관찰 · {_escape(post_state)}</b><img src="{_escape(_web_path(root, output, frames['corrected']['path']))}" alt="seed {seed} corrected 프레임"><span>{_escape(post_text)}</span></div>
        </div>
      </article>"""


def render(comparison: dict, e2e: dict, root: Path, output: Path) -> str:
    rows = {row["seed"]: row for row in e2e["seeds"]}
    cards = "".join(_seed_card(seed, comparison["pairs"][seed], rows.get(seed), root, output) for seed in SEEDS)
    state = "E2E 2/2 recovered" if e2e["status"] == "complete" else "E2E 공개 결과 대기"
    source_json = _script_json({"comparison": comparison["files"], "e2e_status": e2e["status"]})
    return f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="color-scheme" content="light">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>RL Triage · Vision E2E Replay</title>
<style>
:root {{--nv:#76b900;--ink:#17220f;--muted:#62705b;--line:#d9e4d3;--soft:#f4f8f1;--warn:#936200}}
*{{box-sizing:border-box}} html,body{{margin:0;background:#fff;color:var(--ink);font-family:Inter,"Noto Sans KR",sans-serif}}
#viewport{{position:fixed;inset:0;overflow:hidden;background:#fff}}
#stage{{position:absolute;width:1920px;height:1080px;overflow:hidden;padding:54px 64px 44px;background:linear-gradient(135deg,#fff 0 72%,#f2f8ed 100%)}}
.top{{display:flex;justify-content:space-between;align-items:flex-start;border-bottom:4px solid var(--nv);padding-bottom:20px}}
h1{{font-size:48px;line-height:1.08;margin:0 0 10px}} .kicker{{font-weight:800;color:#4d7800;letter-spacing:.08em}}
.scope{{margin:0;color:var(--muted);font-size:20px}} .badge{{background:var(--nv);color:#102000;padding:14px 22px;border-radius:999px;font-size:22px;font-weight:850}}
.evidence{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:20px 0 16px}}
.evidence div{{border:1px solid var(--line);border-radius:14px;padding:13px 16px;background:#fff;min-height:82px}}
.evidence b{{display:block;color:#4d7800;font-size:16px;margin-bottom:6px}} .evidence span{{font-size:20px;font-weight:760}}
.seeds{{display:grid;grid-template-columns:1fr 1fr;gap:22px;height:660px;overflow:hidden}}
.seed-card{{border:2px solid #cbd9c3;border-top:8px solid var(--nv);border-radius:18px;background:#fff;padding:18px 20px;overflow:hidden}}
.seed-card header{{display:flex;justify-content:space-between;font-size:24px;margin-bottom:14px}} .seed-card header strong{{color:#4d7800}}
.media-row{{display:grid;grid-template-columns:1fr 1fr;gap:10px;height:220px}} figure{{margin:0;overflow:hidden;border-radius:10px;background:#0c1209;position:relative}}
video{{width:100%;height:180px;object-fit:cover;display:block}} figcaption{{color:#fff;padding:6px 10px;font-size:15px}}
.timeline{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:13px;overflow:hidden}}
.timeline div{{min-height:122px;border:1px solid var(--line);border-radius:10px;padding:11px;background:var(--soft);overflow:hidden}}
.timeline b{{display:block;font-size:16px;color:#4d7800;margin-bottom:8px}} .timeline span{{font-size:17px;line-height:1.3}}
.timeline img{{width:100%;height:74px;object-fit:cover;border-radius:6px}} .observation-step{{position:relative}} .observation-step span{{position:absolute;left:14px;right:14px;bottom:12px;background:rgba(9,20,6,.78);color:#fff;padding:4px 6px;border-radius:5px;font-size:13px;line-height:1.15}} .compact{{grid-template-columns:repeat(2,1fr);margin-top:30px}}
.compact div{{min-height:165px;padding:20px}} .pending-step{{border-style:dashed!important;color:var(--warn)}}
footer{{position:absolute;left:64px;right:64px;bottom:18px;display:flex;justify-content:space-between;color:var(--muted);font-size:15px}}
@media(max-width:1200px),(max-height:760px){{#stage{{transform-origin:top left}}}}
</style></head><body><div id="viewport"><main id="stage">
<section class="top"><div><div class="kicker">NVIDIA · RL TRIAGE</div><h1>관찰에서 교정 후 평가까지, 근거 한 줄로 추적</h1>
<p class="scope">등록 F1 두 seed 개발판 · 처음 보는 결함 평가 아님 · 세 입력 조건 비교는 observation_only</p></div><div class="badge">{_escape(state)}</div></section>
<section class="evidence"><div><b>비교 모델 선택</b><span>24/24 유효 · P_reward</span></div><div><b>새 E2E 첫 판단</b><span>2/2 유효 · P_reward</span></div><div><b>영상에 따른 선택 변화</b><span>0회</span></div><div><b>비교 단계 검사</b><span>NONE normal · F1 abnormal</span></div></section>
<section class="seeds">{cards}</section>
<footer><span>후속 관찰 used_for_decision=false · 초기 영상+로그 선택 used_for_decision=true · 선택 변화 0회와 별도</span><span>1920 × 1080 · overflow 0</span></footer>
</main></div><script type="application/json" id="source-bindings">{source_json}</script>
<script>(()=>{{const s=document.getElementById('stage'),v=document.getElementById('viewport');function fit(){{const k=Math.min(innerWidth/1920,innerHeight/1080);s.style.transform=`scale(${{k}})`;s.style.left=`${{(innerWidth-1920*k)/2}}px`;s.style.top=`${{(innerHeight-1080*k)/2}}px`;requestAnimationFrame(()=>{{window.__REPLAY_LAYOUT__={{width:1920,height:1080,overflowX:Math.max(0,s.scrollWidth-1920),overflowY:Math.max(0,s.scrollHeight-1080)}};v.dataset.overflow=window.__REPLAY_LAYOUT__.overflowX+window.__REPLAY_LAYOUT__.overflowY===0?'0':'1'}})}}addEventListener('resize',fit);fit()}})();</script>
</body></html>\n"""


def build(root: Path, manifest_path: Path, output: Path) -> str:
    comparison = validate_comparison(root)
    manifest = _read(root / manifest_path)
    e2e = validate_e2e(manifest, root)
    for index, row in enumerate(e2e["seeds"], start=1):
        expected = comparison["summary"]["source_labels"][f"source-0{index}"]["registered-f1"]["checkpoint_sha256"]
        if row["source_fault_checkpoint_sha256"] != expected:
            raise ValueError("E2E source checkpoint와 비교 입력이 다르다")
    rendered = render(comparison, e2e, root, output)
    if ("width:1920px;height:1080px;overflow:hidden" not in rendered
            or "overflowX:Math.max(0,s.scrollWidth-1920)" not in rendered
            or "overflowY:Math.max(0,s.scrollHeight-1080)" not in rendered):
        raise ValueError("1920x1080 overflow 검증 경계가 없다")
    return rendered


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e2e-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    manifest = _relative(args.e2e_manifest)
    output = _relative(args.output)
    content = build(ROOT, manifest, output)
    target = ROOT / output
    if args.check:
        current = target.is_file() and not target.is_symlink() and target.read_text(encoding="utf-8") == content
        print(json.dumps({"status": "current" if current else "stale", "output": output.as_posix()}, ensure_ascii=False))
        return 0 if current else 1
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    target.write_text(content, encoding="utf-8", newline="\n")
    print(json.dumps({"status": "written", "output": output.as_posix()}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
