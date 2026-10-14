"""ANYmal-C 이식 공개 문서 생성기: 단계 0·1 결과와 단계 2 사전등록·시연 결과에서 문서를 만들고, 다른 프로토콜의 결과나 단계 2가
결합한 것과 다른 단계 1 결과는 거부한다(커밋된 파일 사본으로 만든 임시 폴더)."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))

import build_anymal_doc as D  # noqa: E402

EVAL_PROTOCOL = "bench/protocols/fixed_eval_v1_anymal_c.json"
DEMO = "evals/results/anymal_c_e2e_f1_s2047_test/demo_result.json"
V4_DEMO = ROOT / "evals/results/e2e_demo_v4_f2_s2046_20261014/demo_result.json"


def _write(root: Path, relative: str, value) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8", newline="\n")
    return path


@pytest.fixture
def root(tmp_path):
    base = tmp_path / "repo"
    for relative in (D.STAGE0["protocol"], D.STAGE0["result"], D.STAGE1["protocol"], D.STAGE1["result"], D.STAGE2["protocol"],
                     EVAL_PROTOCOL):
        (base / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / relative, base / relative)
    return base


def demo_result(root: Path, **change) -> dict:
    """시연 v4 공개 결과의 모양을 빌려 이 시연 결과처럼 만든 단위 시험용 본문(보고서는 ANYmal-C 단계 0 보고서 사본)."""
    demo = copy.deepcopy(json.loads(V4_DEMO.read_text(encoding="utf-8")))
    reports = {}
    for arm in ("NONE", "FAULT"):
        relative = f"evals/results/anymal_c_e2e_f1_s2047_test/runs/{arm}.json"
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / "evals/results/anymal_c_gate_20261014/runs/anymal_c_gate_none_s2043.json", root / relative)
        reports[arm] = {"path": relative, "sha256_lf": D.lf_sha(root / relative)}
    demo.update(schema=D.DEMO_SCHEMA, protocol_sha256_lf=D.lf_sha(root / D.STAGE2["protocol"]), seed=2047,
                case="anymal_c_h02_s2047", fault="F1", mechanism="reward", source_commit_sha=["b" * 40])
    demo["evaluation"]["reports"] = reports
    demo.update(change)
    return demo


def test_without_a_demo_result_the_stage2_section_waits(root):
    text = D.build(root, None)
    assert "| 2 F1 시연 | 결함 하나를 끝까지 연결하는가 | 진행 중 |" in text and "시연이 아직 끝나지 않았다" in text
    bands = json.loads((root / D.STAGE1["result"]).read_text(encoding="utf-8"))["analysis"]["bands"]
    assert f"선속도 오차 비 ≤ {D.r(bands['err_xy_ratio_max'])}" in text and "12개 중 12개가 normal" in text
    assert "R1이 울린 정상 실행은 0개다" in text and text == D.build(root, None)
    assert "직접" not in text


def test_a_result_from_another_protocol_is_refused(root):
    path = root / D.STAGE0["result"]
    result = json.loads(path.read_text(encoding="utf-8"))
    _write(root, D.STAGE0["result"], {**result, "protocol_sha256_lf": "0" * 64})
    with pytest.raises(ValueError, match="결과가 아니다"):
        D.build(root, None)


def test_a_stage1_result_other_than_the_one_bound_by_stage2_is_refused(root):
    path = root / D.STAGE1["result"]
    result = json.loads(path.read_text(encoding="utf-8"))
    _write(root, D.STAGE1["result"], {**result, "wording": result["wording"] + " "})
    with pytest.raises(ValueError, match="단계 1 결과"):
        D.build(root, None)


def test_a_recovered_demo_fills_the_result_section(root):
    _write(root, DEMO, demo_result(root))
    text = D.build(root, DEMO)
    assert "결과는 **`recovered`, 시연 성공**이다" in text and "| 2 F1 시연 | 결함 하나를 끝까지 연결하는가 | `recovered`, 시연 성공 |" in text
    assert "사전 예측 7개 중 적중 7, 빗나감 0, 그 단계까지 가지 못함 0" in text and "성공 조건 12/12 참." in text
    assert "| 검사 `58ff030724c2` | `P_reward` normal(track_lin_vel_xy_exp_rel_error 1.8e-08), GPU 벽시계 19.3 s |" in text
    assert "| 첫 판단 | 예 | 0회 | 예 | 1 | 2,837·186 | 2.92 s |" in text and f"--demo {DEMO} --check" in text


def test_a_demo_that_ended_early_renders_without_later_steps(root):
    demo = demo_result(root, outcome="model_invalid", demo_success=False, diagnosis=None, loop=None, specificity=None,
                       correction=None, terminal={"outcome": "model_invalid", "at": "2026-10-14T10:00:00+00:00"},
                       prediction_results={k: None for k in D.PREDICTION_NAMES},
                       model_feedback={"diagnosis": None, "correction": None},
                       unmet_conditions=["fresh_diagnosis_call"])
    _write(root, DEMO, demo)
    text = D.build(root, DEMO)
    assert "시연은 `model_invalid`로 끝났다" in text and "| 첫 판단 | — | — | — | — | — | — |" in text
    assert "그 단계까지 가지 못함 7" in text and "(거짓: fresh_diagnosis_call)" in text


def test_a_demo_from_another_protocol_or_with_changed_reports_is_refused(root):
    _write(root, DEMO, demo_result(root, protocol_sha256_lf="0" * 64))
    with pytest.raises(ValueError, match="단계 2 프로토콜"):
        D.build(root, DEMO)
    demo = demo_result(root)
    _write(root, DEMO, demo)
    _write(root, demo["evaluation"]["reports"]["FAULT"]["path"], {"metrics": {}})
    with pytest.raises(ValueError, match="SHA"):
        D.build(root, DEMO)


def test_check_reports_stale_and_current(root, monkeypatch, capsys):
    monkeypatch.setattr(D, "ROOT", root)
    assert D.main(["--check"]) == 1
    assert D.main([]) == 0 and D.main(["--check"]) == 0
    assert '"status": "current"' in capsys.readouterr().out
