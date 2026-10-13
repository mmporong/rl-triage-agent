"""재생 화면 D3 장면: 요약·사례 결과를 근거로 묶고, 커밋 전·프로토콜 불일치면 숫자를 쓰지 않는다(임시 git 저장소 fixture)."""
from __future__ import annotations

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
import d3_summary as SU  # noqa: E402
import replay_d3 as R  # noqa: E402

TAG = "d3_experiment_summary_20261020"
SUMMARY = f"evals/results/{TAG}/summary.json"


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.com",
                    "-c", "core.autocrlf=false", *args], check=True, capture_output=True)


def _write(repo: Path, relative: str, value) -> None:
    path = repo / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8", newline="\n")


def _case(case: str, protocol: dict, protocol_sha: str, counts: dict[str, int], top: int, *, valid: bool = True) -> dict:
    spec = protocol["cases"][case]
    labels = {f"change_{i + 1}": cid for i, cid in enumerate(spec["order"])}
    methods = {m: {"status": "confirmed", "reason": None, "confirmed": spec["harmful"],
                   "queries": [{"reverted": [], "answer": "healthy"}] * counts[m], "model": None} for m in R.METHODS}
    for m in ("B4", "B5"):
        methods[m]["model"] = {"status": "valid", "ranking_labels": [f"change_{top + 1}"], "uncertain": False, "attempts": 1,
                               "total_tokens": 100}
    return {"case": case, "aggregate": spec["aggregate"], "harmful": spec["harmful"], "labels": labels, "valid": valid,
            "invalid_reason": None if valid else "interaction_observed", "methods": methods,
            "protocol_sha256_lf": protocol_sha}


@pytest.fixture
def repo(tmp_path):
    """프로토콜(실제 v2 사본)·사례 결과 3개·요약을 커밋한 저장소. C3는 B4가 원인이 아닌 변경을 1위로 둔다."""
    root = tmp_path / "repo"
    (root / "bench/protocols").mkdir(parents=True)
    _git(root, "init", "-q")
    shutil.copy(ROOT / R.PROTOCOL, root / R.PROTOCOL)
    protocol = json.loads((root / R.PROTOCOL).read_text(encoding="utf-8"))
    protocol_sha = hashlib.sha256((root / R.PROTOCOL).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    counts = {"C1": {"B1": 4, "B2": 2, "B3": 2, "B4": 1, "B5": 3}, "C2": {"B1": 3, "B2": 7, "B3": 3, "B4": 2, "B5": 2},
              "C3": {"B1": 4, "B2": 2, "B3": 1, "B4": 1, "B5": 1}}
    rows, paths = [], {}
    for case, count in counts.items():
        harmful = protocol["cases"][case]["position"] - 1
        result = _case(case, protocol, protocol_sha, count, top=harmful)
        paths[case] = f"evals/results/d3_experiment_v2/{case}/case_result.json"
        _write(root, paths[case], result)
        rows.append({"case": case, "aggregate": True, "valid": True, "invalid_reason": None, "harmful_index": harmful,
                     "methods": {m: {**v, "confirmed": harmful} for m, v in result["methods"].items()}})
    verdict = SU.verdict(rows)
    _write(root, SUMMARY, {"schema": "d3_experiment_summary_v1", "tag": TAG, "protocol_sha256_lf": protocol_sha, **verdict,
                           "cases": paths})
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "results")
    return root


def _scene(repo: Path, slot: str | None = SUMMARY) -> tuple[dict, B.Sources]:
    sources = B.Sources(repo)
    return R.scene_d3(sources, {R.SLOT: slot}), sources


def test_the_scene_binds_the_summary_and_case_results_and_shows_the_outcome(repo):
    scene, sources = _scene(repo)
    assert [b["type"] for b in scene["blocks"]] == ["table", "table", "table"]
    retrains = scene["blocks"][1]["rows"]
    assert [row[0] for row in retrains] == ["C1", "C2", "C3"]
    assert [cell["value"] for cell in retrains[0][1:6]] == [4, 2, 2, 1, 3]
    assert retrains[0][6]["value"].endswith("= U1 (원인)")
    totals = dict((row[0], row[1]["value"]) for row in scene["blocks"][2]["rows"])
    assert totals["합 B1 · B2 · B3 · B4 · B5"] == "11 · 11 · 6 · 4 · 6"
    assert totals["결과"] == "재학습 감소(agent_fewer_retrains)" and totals["결과 문구(사전등록)"] == SU.WORDING["agent_fewer_retrains"]
    assert all(sources.rows[key]["status"] == "ready" for key in sources.rows)
    data = {"scope": B.SCOPE_COMMON, "status_labels": {}, "scenes": [scene], "sources": sources.rows}
    B.check_public(data)


def test_the_prereg_table_comes_from_the_protocol_even_without_a_summary(repo):
    scene, sources = _scene(repo, slot=None)
    assert [b["type"] for b in scene["blocks"]] == ["table", "slot"]
    assert sources.rows[scene["blocks"][1]["source"]]["status"] == "waiting"
    prereg = {row[0]: [c["value"] if isinstance(c, dict) else c for c in row[1:]] for row in scene["blocks"][0]["rows"]}
    assert prereg["C3"] == [2037, "F2", "2 / 8", "4 · 2 · 1", "집계"] and prereg["D0"][-1] == "개발(집계 제외)"


def test_a_summary_from_another_protocol_or_not_committed_shows_no_numbers(repo):
    summary = json.loads((repo / SUMMARY).read_text(encoding="utf-8"))
    _write(repo, SUMMARY, {**summary, "protocol_sha256_lf": "0" * 64})
    _git(repo, "commit", "-q", "-am", "other protocol")
    scene, sources = _scene(repo)
    assert scene["blocks"][1]["type"] == "slot" and sources.rows[SUMMARY]["reason"] == "protocol_mismatch"
    _write(repo, SUMMARY, summary)  # 커밋하지 않은 수정
    scene, sources = _scene(repo)
    assert scene["blocks"][1]["type"] == "slot" and sources.rows[SUMMARY]["status"] == "mismatch"


def test_a_case_result_from_another_protocol_withholds_only_its_cell(repo):
    path = "evals/results/d3_experiment_v2/C2/case_result.json"
    result = json.loads((repo / path).read_text(encoding="utf-8"))
    _write(repo, path, {**result, "protocol_sha256_lf": "0" * 64})
    _git(repo, "commit", "-q", "-am", "other protocol case")
    scene, sources = _scene(repo)
    c2 = scene["blocks"][1]["rows"][1]
    assert c2[1]["value"] == 3 and c2[6]["value"] is None and sources.rows[path]["status"] == "mismatch"


def test_an_excluded_case_is_shown_with_its_reason_and_no_counts(repo):
    summary = json.loads((repo / SUMMARY).read_text(encoding="utf-8"))
    del summary["costs"]["C2"]
    path = "evals/results/d3_experiment_v2/C2/case_result.json"
    result = json.loads((repo / path).read_text(encoding="utf-8"))
    _write(repo, path, {**result, "valid": False, "invalid_reason": "interaction_observed"})
    _write(repo, SUMMARY, summary)
    _git(repo, "commit", "-q", "-am", "excluded")
    scene, _ = _scene(repo)
    c2 = scene["blocks"][1]["rows"][1]
    assert [cell["value"] for cell in c2[1:6]] == [None] * 5 and c2[6]["value"] == "집계 제외 · interaction_observed"


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required to render the static page")
def test_the_filled_scene_renders_with_the_page_script(repo, tmp_path):
    scene, sources = _scene(repo)
    scene["number"] = 1
    data = {"schema": B.SCHEMA, "scope": B.SCOPE_COMMON, "status_labels": {"ready": "확인", "waiting": "대기", "mismatch": "불일치"},
            "scenes": [scene], "sources": sources.rows}
    script = tmp_path / "render.js"
    script.write_text("const app = require(process.argv[2]);\n"
                      "const data = JSON.parse(require('fs').readFileSync(process.argv[3], 'utf8'));\n"
                      "process.stdout.write(app.renderScene(data, 0));\n", encoding="utf-8")
    path = tmp_path / "data.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    html = subprocess.run(["node", str(script), str(ROOT / "docs/replay/app.js"), str(path)], check=True,
                          capture_output=True, text=True, encoding="utf-8").stdout
    assert "재학습 감소(agent_fewer_retrains)" in html and "대기" not in html and "불일치" not in html
    assert R.SCOPE_D3 in html.replace("&#39;", "'")
