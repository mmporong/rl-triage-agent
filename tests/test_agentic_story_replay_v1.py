"""근거 재생 화면 v1: 원문 수치·배지 일치, 영상·계단 슬롯 ready 거부, 선택 주체·모델 혼합 거부, 외부 URL 0, --check."""
from __future__ import annotations

import copy
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))

import build_agentic_story_replay_v1 as S

CASE_VIDEO = "docs/media/agentic_f1_fresh_render_dev_20261029h_v1/case.mp4"
REFERENCE_VIDEO = "docs/media/agentic_f1_fresh_render_dev_20261029h_v1/reference.mp4"


@pytest.fixture(scope="module")
def built():
    data = S.build(ROOT)
    return data, S.render(data, ROOT)


def _raw(relative: str):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_public_json_hash_is_stable_across_checkout_line_endings(tmp_path):
    path = tmp_path / "result.json"
    path.write_bytes(b'{"status":"complete"}\n')
    expected = S._sha256_lf(path)
    path.write_bytes(b'{"status":"complete"}\r\n')
    assert S._sha256_lf(path) == expected


def _lane(data: dict, lane_id: str) -> dict:
    return next(lane for lane in data["lanes"] if lane["id"] == lane_id)


def _resolve(document, pointer: str):
    for part in pointer.lstrip("/").split("/"):
        document = document[int(part)] if isinstance(document, list) else document[part]
    return document


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.com",
                    "-c", "core.autocrlf=false", *args], check=True, capture_output=True)


def _rendered_videos(html: str) -> dict[str, str]:
    return {src: sha for sha, src in re.findall(
        r'<video[^>]* data-sha256="([0-9a-f]{64})" src="([^"]+)"', html
    )}


@pytest.fixture
def mirror(tmp_path, built):
    """화면이 읽은 공개 파일만 복사해 커밋한 임시 저장소."""
    data, _ = built
    root = tmp_path / "repo"
    for relative in data["sources"]:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "public results")
    return root


def test_every_cell_value_and_source_hash_comes_from_committed_raw_files(built):
    data, html = built
    cells = 0
    for lane in data["lanes"]:
        items = [item for row in lane["rows"] for stage in row["stages"].values() for item in stage]
        items += lane.get("inputs", []) + [note for note in lane["notes"] if isinstance(note, dict)]
        for item in items:
            raw = _raw(item["source"])
            for ref in [*item["refs"], *([item["basis"]] if item["basis"] else [])]:
                assert _resolve(raw, ref["pointer"]) == ref["value"], (item["source"], ref["pointer"])
            cells += 1
    assert cells > 60
    for relative, row in data["sources"].items():
        expected_sha = S._sha256_lf(ROOT / relative) if row["kind"] == "public_result" else _sha(ROOT / relative)
        assert row["status"] == "ready" and row["sha256"] == expected_sha
        assert relative in html and row["sha256"] in html


def test_r1_keeps_model_first_choice_separate_from_rule_follow_ups(built):
    data, html = built
    lane = _lane(data, "r1")
    probes = _raw("evals/results/r1_repair_s2071_20261016/probe_and_correction.json")["probes"]
    summary = _raw("evals/results/r1_repair_s2071_20261016/summary.json")
    select = lane["rows"][0]["stages"]["select"]
    assert [item["selector"] for item in select] == ["model", "rule", "rule", "rule", "rule", "rule"]
    assert [item["selector"] == "model" for item in select] == [p["selection_source"] == "model_selected" for p in probes]
    assert [item["state"] for item in lane["rows"][0]["stages"]["result"]] == [p["outcome"] for p in probes]
    assert lane["rows"][0]["final"] == summary["outcome"] == "recovered"
    corrected = summary["fixed_evaluation_metrics"]["CORRECTED"]["lin_vel_rmse_mps"]
    assert f"{corrected:.3f}" in html
    assert "모델은 첫 검사 1개만 골랐고 이후 5개 검사는 등록 규칙이 골랐다." in html


def test_outcome_badges_match_raw_results(built):
    data, _ = built
    v1 = _raw("evals/results/residual_failure_public_20261017_v2/summary.json")
    v2 = _raw("evals/results/residual_success_public_20261017/summary.json")
    assert [row["final"] for row in _lane(data, "residual_v1")["rows"]] == [case["outcome"] for case in v1["cases"]] == ["HOLD", "HOLD"]
    assert [row["stages"]["judge"][0]["state"] for row in _lane(data, "residual_v1")["rows"]] == ["not_recovered"] * 2
    assert [row["final"] for row in _lane(data, "residual_v2")["rows"]] == [case["outcome"] for case in v2["cases"]] == ["recovered"] * 2
    assert {item["selector"] for row in _lane(data, "residual_v2")["rows"] for item in row["stages"]["select"]} == {"model"}
    u1 = _lane(data, "u1")
    assert u1["model"] is None and [row["final"] for row in u1["rows"]] == ["recovered", "recovered"]
    assert {item["selector"] for row in u1["rows"] for stage in row["stages"].values() for item in stage} == {None, "host"}


def test_fresh_35_path_stops_at_model_choice_and_citation_counts(built):
    data, html = built
    lane = _lane(data, "fresh35")
    choice = _raw("evals/results/fresh_frame_probe_choice_dev_s2221_20261029_v1/summary.json")
    counts = choice["citation_counts"]
    row = lane["rows"][0]
    assert lane["model"] == choice["model"]["returned"] == S.MODEL_3_5
    assert row["stages"]["select"][0]["selector"] == "model" and choice["decision"]["probe"] in row["stages"]["select"][0]["text"]
    assert f"telemetry {counts['telemetry']} · frame {counts['frame']}" in html and (counts["telemetry"], counts["frame"]) == (6, 0)
    assert {item["state"] for key in ("result", "judge", "correct", "outcome") for item in row["stages"][key]} == {"pending"}
    assert row["final"] == "pending"
    assert [item["state"] for item in lane["inputs"]] == [None, "execution_failed"]


def test_html_references_no_external_resource_and_binds_existing_videos(built):
    data, html = built
    assert not re.search(r"https?:", html, re.IGNORECASE) and not re.search(
        r"(?:src|href)\s*=\s*[\"']?//", html
    )
    sources = _rendered_videos(html)
    assert sources["../media/agentic_f1_fresh_render_dev_20261029h_v1/reference.mp4"] == _sha(ROOT / REFERENCE_VIDEO)
    assert sources["../media/agentic_f1_fresh_render_dev_20261029h_v1/case.mp4"] == _sha(ROOT / CASE_VIDEO)
    ready_stairs = [slot for slot in data["stairs"] if slot["status"] == "ready"]
    for slot in ready_stairs:
        relative = "../" + Path(slot["video"]["path"]).relative_to("docs").as_posix()
        assert sources[relative] == slot["video"]["sha256"]
    assert len(sources) == 2 + len(ready_stairs)
    assert "교정 영상 아님" in html and html.count("<video") == len(sources)


def test_repository_stairs_snapshot_requires_commit_and_matching_sha(built):
    data, _ = built
    for slot in data["stairs"]:
        manifest = slot["manifest"]
        if manifest is None:
            assert (slot["status"], slot["reason"]) == ("waiting", "public_manifest_missing")
            continue
        tracked = subprocess.run(
            ["git", "-C", str(ROOT), "ls-files", "--error-unmatch", manifest["path"]],
            check=False, capture_output=True,
        ).returncode == 0
        if not tracked:
            assert (slot["status"], slot["reason"]) == ("waiting", "manifest_not_committed")
            continue
        assert manifest["sha256"] == _sha(ROOT / manifest["path"])
        if slot["status"] == "ready":
            video = slot["video"]
            assert video is not None and video["sha256"] == video["expected_sha256"] == _sha(ROOT / video["path"])
            assert subprocess.run(
                ["git", "-C", str(ROOT), "ls-files", "--error-unmatch", video["path"]],
                check=False, capture_output=True,
            ).returncode == 0


def test_selector_or_model_mixing_is_rejected(built):
    data, _ = built

    def rejects(mutate, match):
        changed = copy.deepcopy(data)
        mutate(changed)
        with pytest.raises(ValueError, match=match):
            S.check_story(changed, ROOT)

    rejects(lambda d: _lane(d, "r1")["rows"][0]["stages"]["select"][1].update(selector="model"), "선택 주체")
    rejects(lambda d: _lane(d, "r1")["rows"][0]["stages"]["select"][0].update(selector=None), "선택 주체")
    rejects(lambda d: _lane(d, "u1")["rows"][0]["stages"]["select"][0].update(selector="model"), "선택 주체")
    rejects(lambda d: _lane(d, "residual_v2").update(model=S.MODEL_3_5), "다른 모델 실행")
    rejects(lambda d: _lane(d, "fresh35")["rows"][0]["stages"]["outcome"].insert(
        0, copy.deepcopy(_lane(d, "u1")["rows"][0]["stages"]["outcome"][0])), "다른 모델 실행|3.5")
    rejects(lambda d: (_lane(d, "fresh35")["rows"][0]["stages"]["outcome"][0].update(state="recovered"),
                       _lane(d, "fresh35")["rows"][0].update(final="recovered")), "3.5")
    rejects(lambda d: _lane(d, "residual_v1")["rows"][0].update(final="recovered"), "최종 배지")
    with pytest.raises(ValueError, match="선택 주체"):
        S._selector("/probes/0/selection_source", "llm_guess")


def test_missing_changed_or_stale_video_is_not_ready(mirror):
    (mirror / CASE_VIDEO).unlink()
    (mirror / REFERENCE_VIDEO).write_bytes(b"synthetic_cpu changed bytes")
    data = S.build(mirror)
    videos = {video["path"]: video for video in _lane(data, "fresh35")["videos"]}
    assert (videos[CASE_VIDEO]["status"], videos[CASE_VIDEO]["reason"]) == ("waiting", "missing")
    assert (videos[REFERENCE_VIDEO]["status"], videos[REFERENCE_VIDEO]["reason"]) == ("mismatch", "modified_since_commit")
    rendered = _rendered_videos(S.render(data, mirror))
    assert "../media/agentic_f1_fresh_render_dev_20261029h_v1/case.mp4" not in rendered
    assert "../media/agentic_f1_fresh_render_dev_20261029h_v1/reference.mp4" not in rendered
    assert len(rendered) == sum(slot["status"] == "ready" for slot in data["stairs"])

    story = S.Story(ROOT)
    story.stale_videos = {_sha(ROOT / CASE_VIDEO)}
    slot = S.video_slot(story, CASE_VIDEO, _sha(ROOT / CASE_VIDEO), "input_case", "seed 2221")
    assert (slot["status"], slot["reason"]) == ("mismatch", "stale_render_clip")
    with pytest.raises(ValueError, match="영상 역할"):
        S.video_slot(story, CASE_VIDEO, _sha(ROOT / CASE_VIDEO), "corrected", "seed 2221")


def _stairs_fixture(root: Path, folder: str, role: str, direction: str, tasks: tuple[str, str]) -> Path:
    """synthetic_cpu: 실제 녹화가 아닌 계단 manifest·영상 바이트."""
    base = root / "docs/media" / folder
    (base / "video").mkdir(parents=True)
    video = base / "video/clip.mp4"
    video.write_bytes(b"synthetic_cpu stairs clip")
    manifest = {"schema": S.STAIRS_SCHEMA, "clip_id": f"synthetic_cpu_{role}_{direction}", "robot_role": role,
                "direction": direction, "training_task": tasks[0], "play_task": tasks[1],
                "video": {"path": "video/clip.mp4", "sha256": _sha(video)}}
    (base / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return base


def test_stairs_slot_is_ready_only_with_committed_manifest_and_matching_video(mirror):
    for manifest in mirror.glob("docs/media/**/manifest.json"):
        if json.loads(manifest.read_text(encoding="utf-8")).get("schema") == S.STAIRS_SCHEMA:
            shutil.rmtree(manifest.parent)
    if subprocess.run(["git", "-C", str(mirror), "status", "--porcelain"], check=False,
                      capture_output=True, text=True).stdout:
        _git(mirror, "add", "-A")
        _git(mirror, "commit", "-q", "-m", "remove inherited stairs")
    base = _stairs_fixture(mirror, "stairs_synthetic_cpu_go2_up", "quadruped", "up", S.STAIRS_TASKS["quadruped"])
    _stairs_fixture(mirror, "stairs_synthetic_cpu_h1_down", "biped", "down", S.STAIRS_TASKS["quadruped"])
    slots = {slot["id"]: slot for slot in S.build(mirror)["stairs"]}
    assert (slots["stairs_quadruped_up"]["status"], slots["stairs_quadruped_up"]["reason"]) == ("waiting", "manifest_not_committed")

    _git(mirror, "add", "-A")
    _git(mirror, "commit", "-q", "-m", "synthetic_cpu stairs")
    data = S.build(mirror)
    slots = {slot["id"]: slot for slot in data["stairs"]}
    assert slots["stairs_quadruped_up"]["status"] == "ready"
    assert (slots["stairs_biped_down"]["status"], slots["stairs_biped_down"]["reason"]) == ("mismatch", "task_mismatch")
    assert [slots[key]["status"] for key in ("stairs_biped_up", "stairs_quadruped_down")] == ["waiting", "waiting"]
    html = S.render(data, mirror)
    assert '<video muted controls preload="metadata"' in html and "../media/stairs_synthetic_cpu_go2_up/video/clip.mp4" in html

    (base / "video/clip.mp4").write_bytes(b"synthetic_cpu other bytes")
    slot = next(slot for slot in S.build(mirror)["stairs"] if slot["id"] == "stairs_quadruped_up")
    assert (slot["status"], slot["video"]["status"]) == ("mismatch", "mismatch")


def test_check_accepts_current_html_and_rejects_stale_html(mirror, capsys):
    target = mirror / S.OUTPUT
    assert S.main(["--check"], root=mirror) == 1
    assert S.main([], root=mirror) == 0
    assert S.main(["--check"], root=mirror) == 0
    target.write_text(target.read_text(encoding="utf-8").replace("recovered", "HOLD", 1), encoding="utf-8")
    assert S.main(["--check"], root=mirror) == 1
    assert '"status": "stale"' in capsys.readouterr().out
    assert S.main(["--check"]) == 0
