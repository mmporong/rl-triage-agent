"""발표 재생 화면: 승인 상태 판정, 숫자 차단, 금지 문구·경로 검사, 전 장면 렌더링 회귀."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "bench", ROOT / "evals"):
    sys.path.insert(0, str(directory))

import build_replay_page as B  # noqa: E402

PAGE = ROOT / "docs/replay"


@pytest.fixture(scope="module")
def built():
    return B.build(ROOT)


def _js_escape(text: str) -> str:
    table = {"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}
    return "".join(table.get(ch, ch) for ch in text)


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.com",
                    "-c", "core.autocrlf=false", *args], check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    (root / "bundle").mkdir(parents=True)
    _git(root, "init", "-q")
    (root / "a.json").write_text('{"x": 1}', encoding="utf-8")
    (root / "bundle/artifact_manifest.json").write_text("{}", encoding="utf-8")
    _git(root, "add", "a.json", "bundle/artifact_manifest.json")
    _git(root, "commit", "-q", "-m", "results")
    return root


def test_file_status_distinguishes_ready_waiting_and_mismatch(repo):
    sources = B.Sources(repo)
    ready = sources.file("a.json")
    assert (ready["status"], ready["reason"]) == ("ready", None) and len(ready["sha256"]) == 64
    assert B.Sources(repo).file("a.json", expected_sha256="0" * 64)["status"] == "mismatch"
    assert (sources.file("missing.json")["status"], sources.rows["missing.json"]["reason"]) == ("waiting", "missing")
    assert sources.file(None, kind="end_to_end")["reason"] == "path_not_set"
    (repo / "new.json").write_text("{}", encoding="utf-8")
    assert (sources.file("new.json")["status"], sources.rows["new.json"]["reason"]) == ("waiting", "not_committed")
    (repo / "a.json").write_text('{"x": 2}', encoding="utf-8")
    changed = B.Sources(repo).file("a.json")
    assert (changed["status"], changed["reason"]) == ("mismatch", "modified_since_commit")


def test_bundle_requires_commit_and_publisher_replay(repo):
    row, result = B.Sources(repo).bundle("bundle", lambda folder: {"status": "complete"}, kind="public_bundle_v4")
    assert row["status"] == "ready" and result == {"status": "complete"}

    def broken(folder):
        raise ValueError("publisher source LF-SHA binding differs")
    row, result = B.Sources(repo).bundle("bundle", broken, kind="public_bundle_v4")
    assert (row["status"], row["reason"], result) == ("mismatch", "replay_failed:ValueError", None)
    assert B.Sources(repo).bundle("absent", broken, kind="public_bundle_v4")[0]["reason"] == "missing"
    (repo / "fresh").mkdir()
    (repo / "fresh/artifact_manifest.json").write_text("{}", encoding="utf-8")
    assert B.Sources(repo).bundle("fresh", broken, kind="public_bundle_v4")[0]["reason"] == "not_committed"
    (repo / "bundle/extra.json").write_text("{}", encoding="utf-8")
    assert B.Sources(repo).bundle("bundle", broken, kind="public_bundle_v4")[0]["status"] == "mismatch"


def test_media_video_is_ready_only_when_bound_to_its_committed_manifest(repo):
    folder = "docs/media/go2-x"
    (repo / folder).mkdir(parents=True)
    (repo / folder / "comparison.mp4").write_bytes(b"video")
    digest = hashlib.sha256(b"video").hexdigest()
    (repo / folder / "source_manifest.json").write_text(json.dumps({"comparison": {"mp4": {"sha256": digest}}}),
                                                        encoding="utf-8")
    assert B.media_video(B.Sources(repo), folder)["reason"] == "media_manifest_not_ready"
    _git(repo, "add", f"{folder}/comparison.mp4", f"{folder}/source_manifest.json")
    _git(repo, "commit", "-q", "-m", "media")
    sources = B.Sources(repo)
    video = B.media_video(sources, folder)
    assert (video["status"], video["sha256"]) == ("ready", digest)
    assert sources.rows[f"{folder}/source_manifest.json"]["kind"] == "media_manifest"
    (repo / folder / "source_manifest.json").write_text(json.dumps({"comparison": {"mp4": {"sha256": "0" * 64}}}),
                                                        encoding="utf-8")
    _git(repo, "commit", "-q", "-am", "other manifest")
    assert (B.media_video(B.Sources(repo), folder)["status"], B.media_video(B.Sources(repo), folder)["reason"]) == (
        "mismatch", "sha_mismatch")


def test_cell_withholds_numbers_from_unapproved_sources(repo):
    sources = B.Sources(repo)
    ready, waiting = sources.file("a.json")["id"], sources.file("missing.json")["id"]
    assert B.cell(sources, 1.25, ready)["value"] == 1.25
    assert B.cell(sources, 1.25, waiting)["value"] is None


def test_r1_reproduces_documented_alert_counts_from_public_telemetry():
    key = json.loads((ROOT / "bench/answer_key_p0c.json").read_text(encoding="utf-8"))["cases"]
    faults = {case: ("NONE" if case == "baseline_p0c" else key[case]["fault"]) for case in B.CASES}
    series = {}
    for case in B.CASES:
        for seed in B.SEEDS:
            tele = json.loads((ROOT / f"bench/runs_p0c/{case}_s{seed}.telemetry.json").read_text(encoding="utf-8"))
            series[f"{case}_s{seed}"] = tele["series"]["Metrics/base_velocity/error_vel_xy"]
    alerts = B.r1_alerts(series, faults)
    warned = {case: sum(bool(alerts[f"{case}_s{seed}"]["alert_iterations"]) for seed in B.SEEDS) for case in B.CASES}
    assert [alerts[f"h02_s{seed}"]["alert_iterations"][0] for seed in B.SEEDS] == [135, 134, 135]
    assert warned["baseline_p0c"] == 0 and warned["h02"] == 3 and warned["h03"] == 0
    assert sum(warned[case] for case in B.CASES if case not in ("baseline_p0c", "h02")) == 12


@pytest.mark.parametrize("text", [
    "에이전트가 규칙보다 정확하다.", "GPU 4.3배 절감을 확인했다.", "최초의 학습 실패 진단 에이전트다.",
    "이 결과는 다른 로봇으로 일반화된다.", "LLM이 규칙보다 우월하다.", "이미 본 사례로 보류 평가를 했다.",
])
def test_claim_checker_rejects_forbidden_claims(text):
    assert B.claim_violations(text)


@pytest.mark.parametrize("text", [
    "이미 본 seed의 합성 결함 개발 결과이며 일반화나 AI 교정 선택을 입증하지 않는다.",
    "parent wall time은 GPU 커널 시간이나 다른 방법 대비 절감량이 아니다.",
    "승인된 공개 결과만 읽는 발표 재생 화면. 이미 본 합성 결함 개발 사례이며 일반화·GPU 절감 주장이 아니다.",
    "GPU 커널 시간이나 다른 방법 대비 절감량은 아닙니다.",
])
def test_claim_checker_allows_negated_limits(text):
    assert B.claim_violations(text) == []


@pytest.mark.parametrize("text", ["C:\\Users\\someone\\x.json", "D:/data/run", "/home/user/run", "bench/private/x.json",
                                  '{"pid": 12}', "https://example.com/a.js"])
def test_path_checker_rejects_private_or_network_references(text):
    assert B.path_violations(text)


def test_build_from_repository_is_publishable_and_never_mismatched(built):
    data = built
    assert [scene["number"] for scene in data["scenes"]] == list(range(1, 10))
    assert [scene["id"] for scene in data["scenes"]] == [
        "failure", "curves", "warning", "probe", "correction", "recovery", "hold", "nvidia", "cost"]
    statuses = {key: row["status"] for key, row in data["sources"].items()}
    assert "mismatch" not in statuses.values()
    for tag in ("reward_resume_dev3_20261013", "motor_resume_dev_20261013", "timeout_resume_dev_20261013"):
        assert statuses[f"evals/results/{tag}"] == "ready"
    assert all(not row["path"] or not Path(row["path"]).is_absolute() for row in data["sources"].values())
    for tag in ("reward_resume_dev3_20261013", *B._read(ROOT / B.SLOTS)["v4_tags"]):
        assert statuses[f"docs/media/go2-correction-{tag}/source_manifest.json"] == "ready"
        assert statuses[f"docs/media/go2-correction-{tag}/comparison.mp4"] == "ready"
    failure = data["scenes"][0]["blocks"][0]["rows"]
    assert failure[0][3]["value"] == pytest.approx(0.996710572588678)
    B.check_public(data)


def test_check_public_rejects_injected_claim_or_path(built):
    data = built
    claimed = copy.deepcopy(data)
    claimed["scenes"][0]["notes"].append("이 방법은 GPU 시간을 절감한다.")
    with pytest.raises(ValueError, match="not publishable"):
        B.check_public(claimed)
    leaked = copy.deepcopy(data)
    leaked["scenes"][0]["lead"] = "C:\\Users\\someone\\run"
    with pytest.raises(ValueError, match="not publishable"):
        B.check_public(leaked)


def test_written_data_matches_current_approved_results(built):
    assert (PAGE / "data.js").read_text(encoding="utf-8") == B.render(built)


def test_static_page_files_are_local_only():
    for name in ("index.html", "app.js", "style.css", "data.js", "slots.json"):
        text = (PAGE / name).read_text(encoding="utf-8")
        assert not B.path_violations(text), name
    html = (PAGE / "index.html").read_text(encoding="utf-8")
    assert 'src="data.js"' in html and 'src="app.js"' in html and 'href="style.css"' in html
    for name in ("index.html", "app.js", "style.css"):
        static_text = (PAGE / name).read_text(encoding="utf-8")
        assert not any(B.claim_violations(line) for line in static_text.splitlines()), name


def _scene_sources(scene: dict) -> set[str]:
    sources = set()
    for block in scene["blocks"]:
        sources.update(block.get("status_sources", []))
        sources.update(item["source"] for item in block.get("items", []))
        if block.get("source"):
            sources.add(block["source"])
        for row in block.get("rows", []):
            sources.update(cell["source"] for cell in row if isinstance(cell, dict))
    return sources


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required to render the static page")
def test_every_scene_renders_and_unapproved_numbers_show_status(tmp_path, built):
    data = built
    tampered = copy.deepcopy(data)
    first_cell = tampered["scenes"][0]["blocks"][0]["rows"][0][1]
    tampered["sources"][first_cell["source"]]["status"] = "mismatch"
    first_cell["value"] = None
    script = tmp_path / "render.js"
    script.write_text(
        "const app = require(process.argv[2]);\n"
        "const data = JSON.parse(require('fs').readFileSync(process.argv[3], 'utf8'));\n"
        "const out = data.scenes.map((_, i) => app.renderScene(data, i));\n"
        "process.stdout.write(JSON.stringify(out));\n", encoding="utf-8")
    outputs = {}
    for name, value in (("clean", data), ("tampered", tampered)):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        completed = subprocess.run(["node", str(script), str(PAGE / "app.js"), str(path)],
                                   check=True, capture_output=True, text=True, encoding="utf-8")
        outputs[name] = json.loads(completed.stdout)
    clean = outputs["clean"]
    assert len(clean) == 9 and all(_js_escape(scene["title"]) in html for scene, html in zip(data["scenes"], clean))
    assert "<svg" in clean[1] and "polyline" in clean[1]
    for scene, html in zip(data["scenes"], clean):
        statuses = {data["sources"][source]["status"] for source in _scene_sources(scene)}
        assert ("waiting" in statuses) == ("대기" in html), scene["id"]
        assert ("mismatch" in statuses) == ("불일치" in html), scene["id"]
    assert "../../evals/results/p0c_fixed_eval_20261009/runs/h02_s2026.json" in clean[0]
    assert '<span class="pending mismatch">불일치</span>' in outputs["tampered"][0]
