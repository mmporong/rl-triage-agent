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
import replay_d3  # noqa: E402
import run_training_correction_motor as W  # noqa: E402

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
    assert [scene["number"] for scene in data["scenes"]] == list(range(1, 16))
    assert [scene["id"] for scene in data["scenes"]] == [
        "failure", "curves", "warning", "probe", "correction", "demo", "nat_demo", "recovery", "motor", "hold", "upstream",
        "upstream_demo", "d3", "nvidia", "cost"]
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
    worded = copy.deepcopy(data)  # 상태 단어가 장면 문구에 있으면 근거가 모두 확인돼도 "대기"가 화면에 남는다
    nvidia = next(scene for scene in worded["scenes"] if scene["id"] == "nvidia")
    nvidia["blocks"][0]["rows"][0][1] = "대기 요청을 보여 준다"
    with pytest.raises(ValueError, match="status word"):
        B.check_public(worded)


def test_scene_scope_overrides_the_common_footer_only_where_the_facts_differ(built):
    scopes = {scene["id"]: scene.get("scope") for scene in built["scenes"]}
    assert built["scope"] == B.SCOPE_COMMON
    assert {k: v for k, v in scopes.items() if v} == {
        "demo": B.SCOPE_NEW_SEED_DEMO, "nat_demo": B.SCOPE_NEW_SEED_DEMO,
        "hold": B.SCOPE_HOLD, "upstream": B.SCOPE_UPSTREAM_SMOKE, "upstream_demo": B.SCOPE_UPSTREAM_DEMO,
        "d3": replay_d3.SCOPE_D3}
    assert "새 seed" in B.SCOPE_NEW_SEED_DEMO and "이미 본 합성 결함 개발 사례" not in B.SCOPE_NEW_SEED_DEMO
    assert "seed 2026" in B.SCOPE_UPSTREAM_SMOKE and "새 seed" in B.SCOPE_UPSTREAM_DEMO
    for text in (B.SCOPE_COMMON, B.SCOPE_NEW_SEED_DEMO, B.SCOPE_HOLD, B.SCOPE_UPSTREAM_SMOKE, B.SCOPE_UPSTREAM_DEMO,
                 replay_d3.SCOPE_D3):
        assert not B.claim_violations(text) and not B.path_violations(text)
    leaked = copy.deepcopy(built)  # 장면 문구 검사가 scope에도 적용된다
    leaked["scenes"][0]["scope"] = "이 방법은 GPU 시간을 절감한다."
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
    assert len(clean) == 15 and all(_js_escape(scene["title"]) in html for scene, html in zip(data["scenes"], clean))
    assert "<svg" in clean[1] and "polyline" in clean[1]
    dense = [scene["id"] for scene, html in zip(data["scenes"], clean, strict=True) if 'class="blocks dense"' in html]
    assert dense == ["correction", "demo", "nat_demo", "recovery", "upstream", "upstream_demo", "d3", "nvidia", "cost"]
    for scene, html in zip(data["scenes"], clean):
        statuses = {data["sources"][source]["status"] for source in _scene_sources(scene)}
        assert ("waiting" in statuses) == ("대기" in html), scene["id"]
        assert ("mismatch" in statuses) == ("불일치" in html), scene["id"]
    for scene, html in zip(data["scenes"], clean, strict=True):  # 하단 문구: 장면 scope가 있으면 그것, 없으면 공통
        shown = scene.get("scope") or data["scope"]
        assert _js_escape(shown) in html and (not scene.get("scope") or _js_escape(data["scope"]) not in html), scene["id"]
    assert "../../evals/results/p0c_fixed_eval_20261009/runs/h02_s2026.json" in clean[0]
    assert '<span class="pending mismatch">불일치</span>' in outputs["tampered"][0]


def test_demo_scene_puts_both_demos_side_by_side_and_costs_in_the_last_scene(built):
    scenes = {scene["id"]: scene for scene in built["scenes"]}
    slots = B._read(ROOT / B.SLOTS)
    demo = scenes["demo"]["blocks"]
    assert [block["type"] for block in demo] == ["table", "table"]
    assert demo[0]["status_sources"] == [slots["end_to_end"]] and demo[1]["status_sources"] == [slots["motor_end_to_end"]]
    assert all("seed" in block["title"] and not block.get("wide") for block in demo)
    proposals = [block for block in scenes["correction"]["blocks"] if block["title"].startswith("교정 제안 기록")]
    assert [block["status_sources"][0] for block in proposals] == [slots["correction_proposal"],
                                                                  slots["motor_correction_proposal"]]
    assert not any(block.get("wide") for block in proposals)
    costs = [block for block in scenes["cost"]["blocks"] if block.get("title", "").startswith("종단 시연 비용")]
    assert len(costs) == 1 and [row[1]["source"] for row in costs[0]["rows"]] == [
        slots["end_to_end"], slots["motor_end_to_end"], slots["nat_end_to_end"], slots["upstream_end_to_end"]]
    assert not any("종단 시연" in block.get("title", "") for block in scenes["probe"]["blocks"])


def test_nat_scene_and_nvidia_row_read_the_nat_demo_record(built):
    scenes = {scene["id"]: scene for scene in built["scenes"]}
    nat = B._read(ROOT / B.SLOTS)["nat_end_to_end"]
    steps, record = scenes["nat_demo"]["blocks"]
    assert steps["status_sources"] == [nat] and "첫 판단(NeMo Agent Toolkit 도구 호출)" in [row[0] for row in steps["rows"]]
    values = {row[0]: row[1]["value"] for row in record["rows"]}
    assert values["결정 경로"] == "NeMo Agent Toolkit 도구 호출(nat_v4)"
    assert values["진단 호출"].startswith("LLM ") and "e4b2_skill_checklist" in values["교정 워크플로 도구(설정 목록, 호출 순서 아님)"]
    assert all(built["sources"][source]["status"] == "ready" for source in record["status_sources"])
    role = next(row for row in scenes["nvidia"]["blocks"][0]["rows"] if row[0] == "NeMo Agent Toolkit")
    assert role[2] == "시연 결정 기록" and role[3]["source"] == nat and role[3]["value"].startswith("seed 2034 · 진단 LLM")


def test_demo_scene_keeps_a_waiting_slot_until_the_motor_result_is_set():
    sources = B.Sources(ROOT)
    slots = {**B._read(ROOT / B.SLOTS), "motor_end_to_end": None}
    blocks = B.scene_demo(sources, slots)["blocks"]
    assert [block["type"] for block in blocks] == ["table", "slot"]
    assert blocks[1]["source"] == "slot:motor_end_to_end" and sources.rows["slot:motor_end_to_end"]["status"] == "waiting"


def test_upstream_scenes_read_the_source_smoke_and_demo_protocol_and_wait_for_the_demo():
    sources = B.Sources(ROOT)
    scene = B.scene_upstream(sources)
    origin, smoke = scene["blocks"]
    assert origin["status_sources"] == [B.UPSTREAM_SMOKE_PROTOCOL, B.UPSTREAM_DEMO_PROTOCOL]
    values = _rows(origin)
    assert values["원전 수정"] == ["우회 커밋 58ec02e · 백엔드 수정 PR #1254(2026-08-23 병합)"]
    assert values["Isaac Lab으로 옮긴 보상 항(가중치)"] == [
        "track_lin_vel_xy_exp(1.5) · track_ang_vel_z_exp(0.75) · lin_vel_z_l2(-2.0) · ang_vel_xy_l2(-0.05)"]
    assert values["등록 교정이 되돌리는 함수"] == [
        "restore_body_frame_reward → isaaclab.envs.mdp.rewards.track_lin_vel_xy_exp(네 함수 중 하나)"]
    assert values["코드 지문으로 보이는가"][0].endswith("보이지 않았을 것[추론]")
    measured = _rows(smoke)
    assert smoke["title"] == "U1 smoke(seed 2026, 4096 env × 300회)" and smoke["status_sources"] == [B.UPSTREAM_SMOKE_RESULT]
    assert measured["결과 · 사전 예측 일치"] == ["failure_present · 예"]
    assert measured["standstill ratio"][0] == pytest.approx(1.0395543042795092)
    assert measured["주입 확인(교체된 함수)"] == ["4/4 · 확인 줄 ok"]
    assert measured["정상 학습과의 params 차이"] == ["agent.yaml:run_name"]
    assert all(sources.rows[source]["status"] == "ready" for source in (*origin["status_sources"], *smoke["status_sources"]))
    slots = {**B._read(ROOT / B.SLOTS), "upstream_end_to_end": "evals/results/upstream_demo_not_run/demo_result.json"}
    demo = B.scene_upstream_demo(sources, slots)
    steps, scope = demo["blocks"]
    protocol = B._read(ROOT / B.UPSTREAM_DEMO_PROTOCOL)["scope_gate"]  # 운영 기준 메모는 공개 프로토콜의 강제 모드 규칙이다
    assert B.UPSTREAM_OPERATING_NOTE in demo["notes"] and "재학습 전에 종료" in protocol["enforce_outcome"]
    assert protocol["mode"] == "contrast" and "검증용으로 승인" in protocol["contrast_run"]
    assert [steps["type"], scope["type"]] == ["slot", "slot"] and steps["source"] == scope["source"]
    assert sources.rows[steps["source"]]["status"] == "waiting"
    B.check_public({"scenes": [scene, demo], "sources": sources.rows})


def _motor_replay(seed: int, label: str = "not_recovered", curve: str = "complete") -> dict:
    ratios = ((299, 7.3), (350, 2.1), (450, 1.6), (550, 1.5), (598, 1.6))
    return {"status": "complete", "seed": seed, "labels": {"resume_corrected": label},
            "behavior": {"resume_corrected": {"err_xy": 0.2}, "clean_resume_reference": {"err_xy": 0.125}},
            "budget_curve": {"status": curve, "points": [{"iteration": k, "err_xy_ratio": r} for k, r in ratios]
                             if curve == "complete" else [], "first_healthy_iteration": None,
                             **({"budget_limited_candidate": True} if curve == "complete" else {})},
            "recovery_mode_rule": {"recommended": "resume", "outcome": "recommended_not_recovered"}}


class _MotorSources(B.Sources):
    """모터 공개 묶음만 replay를 통과한 것으로 두고 나머지는 실제 판정을 쓴다."""

    def __init__(self, root: Path, fake: dict):
        super().__init__(root)
        self.fake = fake

    def bundle(self, relative, replay, *, kind):
        tag = relative.rsplit("/", 1)[-1]
        if tag not in self.fake:
            return super().bundle(relative, replay, kind=kind)
        self.rows[relative] = {"id": relative, "path": f"{relative}/artifact_manifest.json", "kind": kind,
                               "sha256": "0" * 64, "status": "ready", "reason": None}
        self.replays[relative] = self.fake[tag]
        return self.rows[relative], self.fake[tag]


def _rows(block: dict) -> dict:
    return {row[0]: [cell["value"] if isinstance(cell, dict) else cell for cell in row[1:]] for row in block["rows"]}


MOTOR_TAGS = B._read(ROOT / B.SLOTS)["motor_tags"]


def test_motor_scene_waits_without_public_bundles(repo):
    sources = B.Sources(repo)
    scene = B.scene_motor(sources, {"motor_tags": MOTOR_TAGS})
    judged, budget = scene["blocks"]
    assert list(_rows(judged)) == list(_rows(budget)) == ["2026", "2028"]
    assert all(value is None for block in (judged, budget) for values in _rows(block).values() for value in values)
    assert {sources.rows[f"evals/results/{tag}"]["status"] for tag in MOTOR_TAGS} == {"waiting"}
    assert len(scene["notes"]) == 1


def test_motor_scene_shows_replayed_values_and_rule_update_only_with_both_seeds(repo, monkeypatch):
    tags = list(MOTOR_TAGS)
    calls = []

    def aggregate(folders):
        calls.append(folders)
        return {"recovered": 0, "labels": {"2026": "not_recovered", "2028": "not_recovered"},
                "prediction_held_all_seeds": True,
                "rule_update": {"decision": "change", "wording": W.FIXED["result_wording"]["rule_update"]["change"]}}
    monkeypatch.setattr(W, "aggregate_public", aggregate)
    sources = _MotorSources(repo, {tags[0]: _motor_replay(2026), tags[1]: _motor_replay(2028)})
    scene = B.scene_motor(sources, {"motor_tags": tags})
    judged, budget = (_rows(block) for block in scene["blocks"])
    assert judged["2026"][:4] == ["미회복", 0.2, 0.125, pytest.approx(1.6)]
    assert judged["2026"][4] == "resume · recommended_not_recovered"
    assert judged["두 seed"] == ["회복 0/2 · 예측 유지 예", "-", "-", "-", "규칙 갱신 change"]
    assert budget["2028"] == ["완료", 7.3, 2.1, 1.6, 1.5, 1.6, "없음", "예"]
    assert scene["blocks"][1]["columns"][2:7] == ["+0", "+51", "+151", "+251", "+299"]
    shifted = _motor_replay(2026)
    shifted["budget_curve"]["points"][1]["iteration"] = 351
    with pytest.raises(ValueError, match="budget curve"):
        B.scene_motor(_MotorSources(repo, {tags[0]: shifted}), {"motor_tags": tags})
    zero = _motor_replay(2026)
    zero["behavior"]["clean_resume_reference"]["err_xy"] = 0.0
    assert _rows(B.scene_motor(_MotorSources(repo, {tags[0]: zero}), {"motor_tags": tags})["blocks"][0])["2026"][3] == "-"
    assert scene["blocks"][1]["rows"][0][3]["digits"] == 3
    assert len(calls) == 1 and [folder.name for folder in calls[0]] == tags
    assert scene["notes"][1:] == [W.FIXED["result_wording"]["budget_curve"],
                                  W.FIXED["result_wording"]["rule_update"]["change"]]
    B.check_public({"scenes": [scene], "sources": sources.rows})

    partial = _MotorSources(repo, {tags[0]: _motor_replay(2026, "invalid", "failed")})
    scene = B.scene_motor(partial, {"motor_tags": tags})
    judged, budget = (_rows(block) for block in scene["blocks"])
    assert len(calls) == 1 and "두 seed" not in judged
    assert scene["notes"][1:] == [W.FIXED["result_wording"]["budget_curve"]]
    assert judged["2026"][0] == "판정 보류" and judged["2028"] == [None] * 5
    assert budget["2026"] == ["기록 실패", "-", "-", "-", "-", "-", "-", "-"] and budget["2028"] == [None] * 8


def test_nvidia_scene_shows_the_rescore_row_only_after_the_public_summary_replays(repo):
    slots = {"nat_rescore": "evals/results/nat_eval_rescore_v1"}
    name = "NeMo Agent Toolkit eval (nat eval --skip_workflow)"

    def rows(sources):
        return {row[0]: row[3]["value"] for row in B.scene_nvidia(sources, slots)["blocks"][0]["rows"]}
    waiting = B.Sources(repo)
    assert rows(waiting)[name] is None and waiting.rows["evals/results/nat_eval_rescore_v1"]["status"] == "waiting"
    assert rows(_MotorSources(repo, {"nat_eval_rescore_v1": {"items": 67}}))[name] == "67항목 · 판정 차이 0"
