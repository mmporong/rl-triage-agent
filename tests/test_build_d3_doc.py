"""D3 공개 문서 생성기: 결과 절을 요약·사례 결과에서 채우고, 다른 프로토콜의 산출물은 거부한다(임시 폴더 fixture)."""
from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "bench", ROOT / "evals"):
    sys.path.insert(0, str(directory))

import build_d3_doc as D  # noqa: E402
import build_replay_page as B  # noqa: E402
import d3_summary as SU  # noqa: E402

TAG = "d3_experiment_summary_20261020"
SUMMARY = f"evals/results/{TAG}/summary.json"
COUNTS = {"C1": {"B1": 4, "B2": 2, "B3": 2, "B4": 1, "B5": 3}, "C2": {"B1": 3, "B2": 7, "B3": 3, "B4": 2, "B5": 2},
          "C3": {"B1": 4, "B2": 2, "B3": 1, "B4": 1, "B5": 1}}


def _write(root: Path, relative: str, value) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8", newline="\n")


def _case(case: str, spec: dict, protocol_sha: str, counts: dict[str, int] | None) -> dict:
    labels = {f"change_{i + 1}": cid for i, cid in enumerate(spec["order"])}
    harmful_label = f"change_{spec['position']}"
    methods = {m: {"status": "confirmed", "reason": None, "confirmed": spec["harmful"],
                   "queries": [{"reverted": [], "answer": "healthy"}] * (counts or {}).get(m, 1),
                   "retrain_parent_wall_s": 600.0 * (counts or {}).get(m, 1),
                   "probe_parent_wall_s": 150.0 if m in ("B3", "B4") else 0.0, "model": None} for m in D.METHODS}
    for m in ("B4", "B5"):
        methods[m]["model"] = {"status": "valid", "ranking_labels": [harmful_label], "uncertain": False, "attempts": 1,
                               "total_tokens": 1000}
    return {"case": case, "aggregate": spec["aggregate"], "harmful": spec["harmful"], "labels": labels, "valid": True,
            "invalid_reason": None, "methods": methods, "protocol_sha256_lf": protocol_sha,
            "shared_parent_wall_s": {"none": 610.5, "failure": 620.0},
            "determinism": {"same": True, "source": "smoke"} if case == "D0" else None}


@pytest.fixture
def root(tmp_path):
    """실제 v2 프로토콜·smoke 프로토콜·smoke v1·v2 요약 사본에, 사례 결과 4개(D0 개발 포함)와 요약을 더한 폴더."""
    base = tmp_path / "repo"
    smoke = json.loads((ROOT / D.SMOKE_PROTOCOL).read_text(encoding="utf-8"))
    for relative in (D.PROTOCOL, D.SMOKE_PROTOCOL, D.SMOKE_V2_SUMMARY, smoke["version_history"]["v1"]["summary"]):
        (base / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / relative, base / relative)
    protocol = json.loads((base / D.PROTOCOL).read_text(encoding="utf-8"))
    protocol_sha = D.lf_sha(base / D.PROTOCOL)
    rows, paths = [], {}
    for case, spec in protocol["cases"].items():
        result = _case(case, spec, protocol_sha, COUNTS.get(case))
        paths[case] = f"evals/results/d3_experiment_v2/{case}/case_result.json"
        _write(base, paths[case], result)
        rows.append({"case": case, "aggregate": spec["aggregate"], "valid": True, "invalid_reason": None,
                     "harmful_index": spec["position"] - 1,
                     "methods": {m: {**v, "confirmed": spec["position"] - 1} for m, v in result["methods"].items()}})
    verdict = SU.verdict(rows)
    _write(base, SUMMARY, {"schema": "d3_experiment_summary_v1", "tag": TAG, "protocol_sha256_lf": protocol_sha, **verdict,
                           "cases": paths, "forbidden_claims": list(SU.FORBIDDEN_CLAIMS)})
    return base


def _public_problems(text: str) -> list[str]:
    """공개 문서 검사: 재생 화면과 같은 주장 검사, 사용자 경로·비공개 경로, "직접". 외부 출처 URL은 문서에서 허용한다."""
    local = re.sub(r"https?://\S+", "", text)
    paths = [p.pattern for p in B.PRIVATE_PATTERNS if p.search(local)]
    return B.claim_violations(text) + paths + (["직접"] if "직접" in text else [])


def test_the_result_section_is_filled_from_the_summary_and_case_results(root):
    text = D.build(root, SUMMARY)
    assert "| C1 | 4 | 2 | 2 | 1 | 3 | change_2 = U1 |" in text
    assert "| D0 | - | - | - | - | - | 집계 제외: 개발 사례 |" in text
    assert "| 확인 재학습 합 | 11 | 11 | 6 | 4 | 6 |" in text
    assert "| 확인 재학습 부모 벽시계 합(초) | 6600.0 | 6600.0 | 3600.0 | 2400.0 | 3600.0 |" in text
    assert "| 기준·고리 probe 부모 벽시계 합(초) | 0.0 | 0.0 | 450.0 | 450.0 | 0.0 |" in text
    assert "| 모델 토큰 합 | - | - | - | 3000 | 3000 |" in text
    assert "부모 벽시계): C1 610.5 · 620.0초, C2 610.5 · 620.0초, C3 610.5 · 620.0초." in text
    assert f"**결과: 재학습 감소(agent_fewer_retrains).** {SU.WORDING['agent_fewer_retrains']}" in text
    assert "체크포인트 SHA가 같았다(대조 출처 smoke)" in text
    assert "본실험 요약이 아직 없다" not in text
    assert _public_problems(text) == []


def test_the_pending_doc_has_the_preregistration_but_no_result_numbers(root):
    text = D.build(root, None)
    assert "본실험 요약이 아직 없다" in text and "**결과:" not in text and "| 확인 재학습 합 |" not in text
    assert "| 합 | 11 | 11 | 6 |" in text and "| C2 | 2036 | F1 | 7 / 8 | 집계 |" in text
    assert "개발 사례 D0에서 한 번 대조한다" in text
    assert _public_problems(text) == []


def test_a_summary_or_case_result_from_another_protocol_is_refused(root):
    summary = json.loads((root / SUMMARY).read_text(encoding="utf-8"))
    _write(root, SUMMARY, {**summary, "protocol_sha256_lf": "0" * 64})
    with pytest.raises(ValueError, match="요약이 이 프로토콜"):
        D.build(root, SUMMARY)
    _write(root, SUMMARY, summary)
    path = summary["cases"]["C2"]
    result = json.loads((root / path).read_text(encoding="utf-8"))
    _write(root, path, {**result, "protocol_sha256_lf": "0" * 64})
    with pytest.raises(ValueError, match="C2 사례 결과"):
        D.build(root, SUMMARY)


def test_an_excluded_case_shows_its_reason_and_is_left_out_of_the_totals(root):
    summary = json.loads((root / SUMMARY).read_text(encoding="utf-8"))
    del summary["costs"]["C2"]
    summary["excluded_cases"] = {"C2": "interaction_observed"}
    path = summary["cases"]["C2"]
    result = json.loads((root / path).read_text(encoding="utf-8"))
    _write(root, path, {**result, "valid": False, "invalid_reason": "interaction_observed"})
    _write(root, SUMMARY, summary)
    text = D.build(root, SUMMARY)
    assert "| C2 | - | - | - | - | - | 집계 제외: interaction_observed |" in text
    assert "D0 결정성 대조 기록이" not in text  # D0 결과에 대조 기록이 있으면 결과 문장으로 쓴다
    assert "| 모델 토큰 합 | - | - | - | 2000 | 2000 |" in text and "- 집계 제외: C2(interaction_observed)" in text


def test_a_filled_doc_without_the_d0_determinism_record_says_so(root):
    summary = json.loads((root / SUMMARY).read_text(encoding="utf-8"))
    path = summary["cases"]["D0"]
    _write(root, path, {**json.loads((root / path).read_text(encoding="utf-8")), "determinism": None})
    text = D.build(root, SUMMARY)
    assert "D0 결정성 대조 기록이 요약이 가리키는 사례 결과에 없다." in text and "한 번 대조한다" not in text


def test_check_mode_reports_stale_and_current(root, monkeypatch, capsys):
    monkeypatch.setattr(D, "ROOT", root)
    assert D.main(["--summary", SUMMARY, "--check"]) == 1
    assert D.main(["--summary", SUMMARY]) == 0
    assert D.main(["--summary", SUMMARY, "--check"]) == 0
    assert D.main(["--check"]) == 1
    assert '"status": "current"' in capsys.readouterr().out
    _write(root, SUMMARY, {**json.loads((root / SUMMARY).read_text(encoding="utf-8")), "protocol_sha256_lf": "0" * 64})
    assert D.main(["--summary", SUMMARY, "--check"]) == 2


def test_the_published_doc_is_current_with_the_files_it_names():
    """저장소의 docs/D3-MULTI-CHANGE.md가 생성기 출력과 같다. 결과 절이 있으면 그 요약으로, 없으면 대기 판으로 다시 만든다."""
    text = (ROOT / D.OUTPUT).read_text(encoding="utf-8")
    tag = re.search(r"^요약: `([^`]+)`$", text, re.M)
    expected = D.build(ROOT, f"evals/results/{tag.group(1)}/summary.json" if tag else None)
    assert text == expected
    assert _public_problems(text) == []
