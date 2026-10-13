"""재생 화면의 종단 시연 자리: 승인 상태별 자리 표시, 승인된 시연 결과의 표, 금지 문구·경로, 렌더링."""
from __future__ import annotations

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
import replay_demo_slots as DS  # noqa: E402

RESULT = "evals/results/demo_x/demo_result.json"
SCHEMA = "e2e_reward_demo_result_v1"
PROTOCOL = DS.DEMO_PROTOCOLS[SCHEMA]


def demo_result(outcome: str = "recovered", schema: str = SCHEMA) -> dict:
    """evals/e2e_reward_demo.py summarize 산출물과 같은 형식의 합성 결과(값은 가짜)."""
    return {
        "schema": schema, "tag": "demo_x", "seed": 2029, "case": "h02_s2029", "outcome": outcome,
        "demo_success": outcome == "recovered",
        "success_conditions": {"protocol_committed": True, "seed_unused": True, "restart_recovered": outcome == "recovered"},
        "unmet_conditions": [] if outcome == "recovered" else ["restart_recovered"],
        "wording": "보상 결함을 확정하고 승인된 restart 교정까지 실행한 통합 시연 결과 문구다.",
        "common_wording": "통합 시연이다. 진단 정확도·일반화·교정 성공률·GPU 절감의 근거가 아니다.",
        "steps_sha256": {}, "terminal": None, "protocol_sha256_lf": "a" * 64, "source_commit_sha": ["b" * 40],
        "predictions": {}, "prediction_results": {}, "approver_identity_attested": False,
        "training_parent_wall_s": {"NONE": 700.4, "F1": 500.2}, "alarm": {}, "evaluation": {},
        "reference_parent_wall_s": 90.0,
        "diagnosis": {"decision": {"action": "continue", "ranking": ["reward", "actuator", "exploration", "optimizer"],
                                   "probe": "P_reward"},
                      "started": {}, "call": {"response_id": "r", "input_tokens": 2834, "output_tokens": 212}},
        "loop": {"status": "confirmed", "hypotheses": ["reward"], "observed": {"P_reward": "abnormal"},
                 "requests": [{"request_id": "x", "probe": "P_reward", "state": "done", "approver": "op", "gpu_s": 20.9}],
                 "probe_costs": []},
        "correction": {"request_id": "y",
                       "model_proposal": {"accepted": True, "decision": "propose", "agrees_with_rule": True,
                                          "reasons": [], "unmet_checklist": [], "rule_visible_to_model": True},
                       "call": {"input_tokens": 1500, "output_tokens": 180},
                       "receipt": {"event": "receipt", "verdict": outcome, "gpu_s": 711.0},
                       "result": {"label": outcome, "verdict": {"label": "healthy"}, "standstill_ratio": 0.131,
                                  "costs": {"train_parent_wall_s": 650.0, "eval_parent_wall_s": 61.0}}},
    }


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.com",
                    "-c", "core.autocrlf=false", *args], check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    (root / "evals/results/demo_x").mkdir(parents=True)
    (root / "bench/protocols").mkdir(parents=True)
    _git(root, "init", "-q")
    (root / RESULT).write_text(json.dumps(demo_result(), ensure_ascii=False), encoding="utf-8")
    (root / PROTOCOL).write_text(json.dumps({"correction": {
        "correction_id": "restore_body_frame_reward", "registered_recovery_mode": "restart"}}), encoding="utf-8")
    _git(root, "add", RESULT, PROTOCOL)
    _git(root, "commit", "-q", "-m", "demo")
    return root


def _values(blocks: list[dict]) -> dict:
    return {row[0]: row[1]["value"] for block in blocks for row in block.get("rows", [])}


def test_unset_or_unapproved_paths_keep_the_original_slot(repo):
    for relative, reason in ((None, "path_not_set"), ("evals/results/demo_x/missing.json", "missing")):
        sources = B.Sources(repo)
        assert DS.end_to_end_blocks(sources, relative) == [
            {"type": "slot", "label": DS.END_TO_END_LABEL, "source": relative or "slot:end_to_end"}]
        assert sources.rows[relative or "slot:end_to_end"]["reason"] == reason
    sources = B.Sources(repo)
    assert DS.correction_proposal_blocks(sources, None) == [
        {"type": "slot", "wide": True, "label": DS.PROPOSAL_LABEL, "source": "slot:correction_proposal"}]
    (repo / RESULT).write_text(json.dumps(demo_result("not_recovered")), encoding="utf-8")
    sources = B.Sources(repo)
    assert DS.end_to_end_blocks(sources, RESULT)[0]["type"] == "slot"
    assert (sources.rows[RESULT]["status"], sources.rows[RESULT]["reason"]) == ("mismatch", "modified_since_commit")


def test_approved_result_fills_steps_costs_and_proposal(repo):
    sources = B.Sources(repo)
    blocks = DS.end_to_end_blocks(sources, RESULT)
    assert [block["type"] for block in blocks] == ["table", "table"] and blocks[0]["status_sources"] == [RESULT]
    values = _values(blocks)
    assert values["시연 결과"] == "recovered · 성공 조건 3/3"
    assert values["첫 판단(새 Nemotron 호출)"] == "continue · 상위 3 reward, actuator, exploration · 제안 검사 P_reward"
    assert values["승인된 검사"] == "1건 실행 · P_reward abnormal · confirmed"
    assert values["교정 제안(새 Nemotron 호출)"] == "propose · 계약 통과 예 · 규칙과 일치 예"
    assert values["승인 뒤 교정 실행 receipt"] == "receipt · recovered"
    assert (values["교정 정책 고정 평가 판정"], values["교정 정책 standstill ratio"]) == ("recovered", 0.131)
    assert (values["NONE 학습 wall(s)"], values["승인 검사 GPU wall(s)"], values["교정 재학습 wall(s)"]) == (700.4, 20.9, 650.0)
    assert values["진단 호출 입력·출력 토큰"] == "2834 / 212" and values["교정 호출 입력·출력 토큰"] == "1500 / 180"
    proposal = DS.correction_proposal_blocks(sources, RESULT)
    assert proposal[0]["status_sources"] == [RESULT, PROTOCOL] and proposal[0]["wide"] is True
    assert _values(proposal) == {
        "등록 교정 · 회복 방식(시연 프로토콜)": "restore_body_frame_reward · restart", "모델 제안 결정": "propose",
        "공개 계약 검증 통과": "예", "결정적 규칙과 일치": "예", "규칙을 모델 입력에 보여 줌": "예",
        "미충족 체크리스트 항목 수": 0, "승인 뒤 1회 소비": "예"}


def test_unjudged_correction_shows_no_verdict_instead_of_waiting(repo):
    """v1 공개 결과처럼 교정 실행이 판정 없이 끝나면 대기가 아니라 판정 없음·기록 없음으로 보인다."""
    unknown = demo_result("correction_unknown")
    unknown["correction"].update(receipt={"event": "receipt", "verdict": "unknown", "gpu_s": None},
                                 result={"label": None, "verdict": None, "standstill_ratio": None, "costs": None})
    (repo / RESULT).write_text(json.dumps(unknown, ensure_ascii=False), encoding="utf-8")
    _git(repo, "add", RESULT)
    _git(repo, "commit", "-q", "-m", "unknown")
    values = _values(DS.end_to_end_blocks(B.Sources(repo), RESULT))
    assert values["시연 결과"] == "correction_unknown · 성공 조건 2/3"
    assert values["승인 뒤 교정 실행 receipt"] == "receipt · unknown"
    assert (values["교정 정책 고정 평가 판정"], values["교정 정책 standstill ratio"]) == (DS.NO_VERDICT, DS.NO_RECORD)
    assert (values["교정 재학습 wall(s)"], values["교정 고정 평가 wall(s)"]) == (DS.NO_RECORD, DS.NO_RECORD)
    assert None not in values.values()


def test_v2_result_reads_the_v2_demo_protocol(repo):
    v2_schema = "e2e_reward_demo_result_v2"
    v2_protocol = DS.DEMO_PROTOCOLS[v2_schema]
    (repo / RESULT).write_text(json.dumps({**demo_result(schema=v2_schema), "seed": 2030}), encoding="utf-8")
    (repo / v2_protocol).write_text(json.dumps({"correction": {
        "correction_id": "restore_body_frame_reward", "registered_recovery_mode": "restart"}}), encoding="utf-8")
    _git(repo, "add", RESULT, v2_protocol)
    _git(repo, "commit", "-q", "-m", "v2")
    blocks = DS.correction_proposal_blocks(B.Sources(repo), RESULT)
    assert blocks[0]["status_sources"] == [RESULT, v2_protocol] and blocks[0]["title"].endswith("seed 2030)")
    assert _values(blocks)["등록 교정 · 회복 방식(시연 프로토콜)"] == "restore_body_frame_reward · restart"


def test_unexpected_schema_is_a_mismatch_not_a_table(repo):
    (repo / RESULT).write_text(json.dumps({**demo_result(), "schema": "other"}), encoding="utf-8")
    _git(repo, "add", RESULT)
    _git(repo, "commit", "-q", "-m", "other")
    sources = B.Sources(repo)
    assert DS.end_to_end_blocks(sources, RESULT)[0]["type"] == "slot"
    assert (sources.rows[RESULT]["status"], sources.rows[RESULT]["reason"]) == ("mismatch", "unexpected_schema")


def test_missing_demo_protocol_withholds_only_the_registered_correction(repo):
    _git(repo, "rm", "-q", PROTOCOL)
    _git(repo, "commit", "-q", "-m", "drop")
    values = _values(DS.correction_proposal_blocks(B.Sources(repo), RESULT))
    assert values["등록 교정 · 회복 방식(시연 프로토콜)"] is None and values["모델 제안 결정"] == "propose"


def test_block_text_has_no_forbidden_claim_or_private_path(repo):
    sources = B.Sources(repo)
    texts = []
    for block in DS.end_to_end_blocks(sources, RESULT) + DS.correction_proposal_blocks(sources, RESULT):
        texts.append(block.get("title", ""))
        for row in block["rows"]:
            texts.append(row[0])
            texts += [str(cell["value"]) for cell in row[1:] if isinstance(cell, dict)]
    texts += [DS.END_TO_END_LABEL, DS.PROPOSAL_LABEL]
    assert not [text for text in texts if "사람 승인" in text]
    assert not [problem for text in texts for problem in B.claim_violations(text) + B.path_violations(text)]


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required to render the static page")
def test_ready_and_waiting_blocks_render(repo, tmp_path):
    sources = B.Sources(repo)
    blocks = DS.end_to_end_blocks(sources, RESULT) + DS.correction_proposal_blocks(sources, RESULT) \
        + DS.end_to_end_blocks(sources, None)
    data = {"scope": "s", "status_labels": {"ready": "확인", "waiting": "대기", "mismatch": "불일치"},
            "sources": sources.rows, "scenes": [{"number": 1, "title": "t", "lead": "l", "blocks": blocks, "notes": []}]}
    (tmp_path / "data.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    (tmp_path / "render.js").write_text(
        "const app = require(process.argv[2]);\n"
        "const data = JSON.parse(require('fs').readFileSync(process.argv[3], 'utf8'));\n"
        "process.stdout.write(app.renderScene(data, 0));\n", encoding="utf-8")
    html = subprocess.run(["node", str(tmp_path / "render.js"), str(ROOT / "docs/replay/app.js"), str(tmp_path / "data.json")],
                          check=True, capture_output=True, text=True, encoding="utf-8").stdout
    assert "recovered · 성공 조건 3/3" in html and "0.131" in html and "restore_body_frame_reward · restart" in html
    assert "대기" in html and DS.END_TO_END_LABEL.split(":")[0] in html
