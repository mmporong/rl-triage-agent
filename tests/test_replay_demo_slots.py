"""재생 화면의 종단 시연 자리: 승인 상태별 자리 표시, 승인된 시연 결과의 표, 금지 문구·경로, 렌더링."""
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


def _cost_values(sources, entries) -> list:
    return [[row[0], *(cell["value"] for cell in row[1:])] for row in DS.cost_block(sources, entries)["rows"]]


def test_approved_result_fills_steps_costs_and_proposal(repo):
    sources = B.Sources(repo)
    blocks = DS.end_to_end_blocks(sources, RESULT)
    assert [block["type"] for block in blocks] == ["table"] and blocks[0]["status_sources"] == [RESULT]
    assert blocks[0]["title"] == "종단 시연 seed 2029: 경고부터 교정 판정까지"
    values = _values(blocks)
    assert values["시연 결과"] == "recovered · 성공 조건 3/3"
    assert values["첫 판단(새 Nemotron 호출)"] == "continue · 상위 3 reward, actuator, exploration · 제안 검사 P_reward"
    assert values["승인된 검사"] == "1건 실행 · P_reward abnormal · confirmed" and "특이성 관문" not in values
    assert values["교정 제안(새 Nemotron 호출)"] == "propose · 계약 통과 예 · 규칙과 일치 예"
    assert values["승인 뒤 교정 실행 receipt"] == "receipt · recovered"
    assert (values["교정 정책 고정 평가 판정"], values["교정 정책 standstill ratio"]) == ("recovered", 0.131)
    cost = DS.cost_block(sources, [(RESULT, "end_to_end", "보상 결함")])
    assert cost["columns"] == DS.COST_COLUMNS and cost["status_sources"] == [RESULT] and cost["wide"] is True
    assert _cost_values(sources, [(RESULT, "end_to_end", "보상 결함")]) == [
        ["보상 결함 seed 2029", "700 / 500", 20.9, "650 / 61", "2834 / 212 · 1500 / 180"]]
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
    sources = B.Sources(repo)
    values = _values(DS.end_to_end_blocks(sources, RESULT))
    assert values["시연 결과"] == "correction_unknown · 성공 조건 2/3"
    assert values["승인 뒤 교정 실행 receipt"] == "receipt · unknown"
    assert (values["교정 정책 고정 평가 판정"], values["교정 정책 standstill ratio"]) == (DS.NO_VERDICT, DS.NO_RECORD)
    assert None not in values.values()
    costs = _cost_values(sources, [(RESULT, "end_to_end", "보상 결함")])[0]
    assert costs[3] == DS.NO_RECORD and None not in costs  # 교정 재학습·평가 기록 없음


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


MOTOR_SCHEMA = "e2e_demo_result_v3_1"
MOTOR_PROTOCOL = "bench/protocols/e2e_demo_v3_1_f2.json"


def motor_result(variant: str = "f2") -> dict:
    """evals/e2e_demo_v3_1.py summarize 산출물과 같은 형식의 모터 결함 합성 결과(값은 가짜)."""
    result = demo_result(schema=MOTOR_SCHEMA)
    result.update(seed=2033, variant=variant, fault="F2", mechanism="actuator",
                  training_parent_wall_s={"NONE": 553.6, "FAULT": 592.4},
                  specificity={"required": ["P_noise", "P_value"], "passed": True,
                               "outcomes": {"P_value": "normal", "P_noise": "normal"}})
    result["correction"]["request"] = {"correction_id": "restore_motor_torque_speed", "recovery_mode": "restart"}
    return result


def _commit_motor(repo: Path, result: dict) -> None:
    (repo / RESULT).write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    (repo / MOTOR_PROTOCOL).write_text(json.dumps({"correction": {
        "correction_id": "restore_motor_torque_speed", "modes": {"restart": {}, "resume": {}}}}), encoding="utf-8")
    _git(repo, "add", RESULT, MOTOR_PROTOCOL)
    _git(repo, "commit", "-q", "-m", "motor")


def test_v3_1_motor_result_shows_its_fault_gate_rule_chosen_mode_and_costs(repo):
    _commit_motor(repo, motor_result())
    sources = B.Sources(repo)
    steps = DS.end_to_end_blocks(sources, RESULT, kind="motor_end_to_end", label=DS.MOTOR_END_TO_END_LABEL)
    assert steps[0]["title"] == "종단 시연 seed 2033 F2·actuator: 경고부터 교정 판정까지"
    values = _values(steps)
    assert values["특이성 관문"] == "통과 · P_noise normal, P_value normal"
    assert list(values).index("특이성 관문") == list(values).index("승인된 검사") + 1
    proposal = DS.correction_proposal_blocks(sources, RESULT, kind="motor_correction_proposal",
                                             label=DS.MOTOR_PROPOSAL_LABEL, wide=False)
    assert proposal[0]["status_sources"] == [RESULT, MOTOR_PROTOCOL] and proposal[0]["wide"] is False
    proposed = _values(proposal)
    assert proposed["등록 교정 · 회복 방식(시연 프로토콜)"] == "restore_motor_torque_speed · restart/resume"
    assert proposed["등록 규칙이 고른 회복 방식"] == "restart"
    assert _cost_values(sources, [(RESULT, "motor_end_to_end", "모터 결함")])[0][:2] == ["모터 결함 seed 2033", "554 / 592"]


def test_unknown_v3_1_variant_is_a_mismatch(repo):
    _commit_motor(repo, motor_result(variant="f9"))
    sources = B.Sources(repo)
    assert DS.end_to_end_blocks(sources, RESULT, kind="motor_end_to_end")[0]["type"] == "slot"
    assert (sources.rows[RESULT]["status"], sources.rows[RESULT]["reason"]) == ("mismatch", "unexpected_variant")


def test_unset_motor_slots_wait_under_their_own_kind(repo):
    sources = B.Sources(repo)
    assert DS.end_to_end_blocks(sources, None, kind="motor_end_to_end", label=DS.MOTOR_END_TO_END_LABEL) == [
        {"type": "slot", "label": DS.MOTOR_END_TO_END_LABEL, "source": "slot:motor_end_to_end"}]
    assert DS.correction_proposal_blocks(sources, None, kind="motor_correction_proposal",
                                         label=DS.MOTOR_PROPOSAL_LABEL, wide=False)[0]["wide"] is False
    row = DS.cost_block(sources, [(None, "motor_end_to_end", "모터 결함")])["rows"][0]
    assert row[0] == "모터 결함" and all(cell["value"] is None and cell["source"] == "slot:motor_end_to_end" for cell in row[1:])
    assert sources.rows["slot:motor_end_to_end"]["status"] == "waiting"


NAT_PROTOCOL = "bench/protocols/e2e_demo_v3_1_f1.json"
NAT_TOOLS = {"diagnosis": ["e4b4_get_facts", "e4b4_submit_decision"],
             "correction": ["e4b2_skill_checklist", "e4b4_submit_proposal"]}


def nat_result(decision_path: str = "nat_v4") -> dict:
    """NAT 결정 경로 시연(v3.1, F1) 합성 결과. 공개 결과처럼 호출 기록에 도구 호출 순서가 없다."""
    result = motor_result(variant="f1")
    result.update(seed=2034, fault="F1", mechanism="reward", decision_path=decision_path,
                  specificity={"required": [], "outcomes": None, "passed": True})
    result["diagnosis"]["call"] = {"path": decision_path, "input_tokens": 47359, "output_tokens": 1632,
                                   "execution_wall_s": 28.73, "llm_calls": 10, "precheck_calls": 1}
    result["correction"]["call"] = {"path": decision_path, "input_tokens": 22247, "output_tokens": 1171,
                                    "execution_wall_s": 18.84, "llm_calls": 7, "precheck_calls": 1}
    return result


def _commit_nat(repo: Path, result: dict, *, stale_sha: bool = False) -> None:
    source_files = {}
    for stage, path in DS.NAT_CONFIGS["nat_v4"].items():
        text = f"workflow:\n  _type: tool_calling_agent\n  tool_names: [{', '.join(NAT_TOOLS[stage])}]\n"
        (repo / path).parent.mkdir(parents=True, exist_ok=True)
        (repo / path).write_text(text, encoding="utf-8")
        source_files[path] = "0" * 64 if stale_sha else hashlib.sha256(text.encode("utf-8")).hexdigest()
    (repo / RESULT).write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    (repo / NAT_PROTOCOL).write_text(json.dumps({
        "correction": {"correction_id": "restore_body_frame_reward", "modes": {"restart": {}, "resume": {}}},
        "nat_adoption": {"decision": "adopt"}, "source_files": source_files}), encoding="utf-8")
    _git(repo, "add", RESULT, NAT_PROTOCOL, *DS.NAT_CONFIGS["nat_v4"].values())
    _git(repo, "commit", "-q", "-m", "nat")


def test_nat_result_shows_its_decision_path_calls_and_bound_tool_lists(repo):
    _commit_nat(repo, nat_result())
    sources = B.Sources(repo)
    values = _values(DS.end_to_end_blocks(sources, RESULT, kind="nat_end_to_end", label=DS.NAT_END_TO_END_LABEL))
    assert "첫 판단(NeMo Agent Toolkit 도구 호출)" in values and "교정 제안(NeMo Agent Toolkit 도구 호출)" in values
    assert values["특이성 관문"] == "등록된 보조 검사 없음"
    (record,) = DS.nat_record_blocks(sources, RESULT)
    assert record["status_sources"] == [RESULT, NAT_PROTOCOL, *DS.NAT_CONFIGS["nat_v4"].values()]
    assert _values([record]) == {
        "결정 경로": "NeMo Agent Toolkit 도구 호출(nat_v4)", "경로 채택 근거(시연 프로토콜)": "NAT 경로 비교 요약 판정 adopt",
        "진단 호출": "LLM 10회 · 제출 전 점검 1회 · 실행 28.7 s", "진단 토큰(입력 / 출력)": "47359 / 1632",
        "교정 호출": "LLM 7회 · 제출 전 점검 1회 · 실행 18.8 s", "교정 토큰(입력 / 출력)": "22247 / 1171",
        "진단 워크플로 도구(설정 목록, 호출 순서 아님)": "e4b4_get_facts, e4b4_submit_decision",
        "교정 워크플로 도구(설정 목록, 호출 순서 아님)": "e4b2_skill_checklist, e4b4_submit_proposal"}
    assert DS.nat_role_cell(sources, RESULT)["value"] == "seed 2034 · 진단 LLM 10회 · 교정 LLM 7회"


def test_nat_tool_lists_wait_or_mismatch_unless_bound_by_the_demo_protocol(repo):
    _commit_nat(repo, nat_result(), stale_sha=True)
    sources = B.Sources(repo)
    record = _values(DS.nat_record_blocks(sources, RESULT))
    assert record["진단 워크플로 도구(설정 목록, 호출 순서 아님)"] is None and record["진단 호출"].startswith("LLM 10회")
    config = DS.NAT_CONFIGS["nat_v4"]["diagnosis"]
    assert (sources.rows[config]["status"], sources.rows[config]["reason"]) == ("mismatch", "demo_protocol_sha_mismatch")


def test_nat_tool_lists_wait_while_the_demo_protocol_is_not_committed(repo):
    _commit_nat(repo, nat_result())
    _git(repo, "rm", "-q", "--cached", NAT_PROTOCOL)
    _git(repo, "commit", "-q", "-m", "protocol out")
    sources = B.Sources(repo)
    record = _values(DS.nat_record_blocks(sources, RESULT))
    assert record["교정 워크플로 도구(설정 목록, 호출 순서 아님)"] is None and record["경로 채택 근거(시연 프로토콜)"] is None
    config = DS.NAT_CONFIGS["nat_v4"]["correction"]
    assert (sources.rows[config]["status"], sources.rows[config]["reason"]) == ("waiting", "demo_protocol_not_ready")


def test_a_single_call_result_cannot_fill_the_nat_slots(repo):
    _commit_nat(repo, nat_result(decision_path="single"))
    sources = B.Sources(repo)
    assert DS.nat_record_blocks(sources, RESULT)[0]["type"] == "slot"
    assert (sources.rows[RESULT]["status"], sources.rows[RESULT]["reason"]) == ("mismatch", "not_nat_decision_path")
    assert DS.nat_role_cell(B.Sources(repo), RESULT)["value"] is None
    _commit_nat(repo, nat_result(decision_path="nat_v9"))
    sources = B.Sources(repo)
    assert DS.end_to_end_blocks(sources, RESULT, kind="nat_end_to_end")[0]["type"] == "slot"
    assert sources.rows[RESULT]["reason"] == "unexpected_decision_path"


UPSTREAM_PROTOCOL = DS.DEMO_PROTOCOLS[DS.UPSTREAM_SCHEMA]
READBACK_NOTE = "등록 표적 복원됨, 그 밖 함수 상태는 보조 지문 기록(aux_fingerprints, 판정에 쓰지 않음)"


def upstream_result(contrast: str | None = "insufficient_and_not_recovered") -> dict:
    """evals/e2e_demo_u1.py summarize 산출물과 같은 형식의 U1 합성 결과(값은 가짜)."""
    result = demo_result("not_recovered", schema=DS.UPSTREAM_SCHEMA)
    result.update(seed=2040, variant="u1", fault="U1", mechanism="reward", decision_path="single",
                  training_parent_wall_s={"NONE": 560.0, "FAULT": 571.0},
                  specificity={"required": [], "passed": True, "outcomes": None},
                  scope_check={"mode": "contrast", "correction_id": "restore_body_frame_reward", "agent_judgment": False,
                               "proceed_to_correction": True,
                               "result": {"verdict": "scope_insufficient", "probe": "P_reward", "threshold": 0.05,
                                          "rel_errors": {"track_lin_vel_xy_exp": 0.93, "track_ang_vel_z_exp": 0.97},
                                          "abnormal_terms": ["track_ang_vel_z_exp", "track_lin_vel_xy_exp"],
                                          "registered_targets": ["track_lin_vel_xy_exp"],
                                          "uncovered_terms": ["track_ang_vel_z_exp"]}},
                  scope_contrast=None if contrast is None else {"category": contrast, "wording": "대조 문구."},
                  prediction_results={"scope_check": True, "correction": True, "scope_contrast": True})
    result["correction"]["request"] = {"correction_id": "restore_body_frame_reward", "recovery_mode": "restart"}
    result["correction"]["scope_check_bound"] = True
    result["correction"]["result"]["correction_readback"] = {
        "note": READBACK_NOTE, "registered_target_state": "restored_to_clean",
        "aux_fingerprints": {"used_for_verdict": False,
                             "differs_from_clean": ["ang_vel_xy_l2", "lin_vel_z_l2", "track_ang_vel_z_exp"]}}
    return result


def _commit_upstream(repo: Path, result: dict) -> None:
    (repo / RESULT).write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    (repo / UPSTREAM_PROTOCOL).write_text(json.dumps({"correction": {
        "correction_id": "restore_body_frame_reward", "modes": {"restart": {}, "resume": {}}},
        "scope_gate": {"mode": "contrast", "contrast_run": "운영자는 부족 판정을 보고 검증용으로 승인한다",
                       "enforce_outcome": "강제 모드에서 재학습 전에 종료"}}), encoding="utf-8")
    _git(repo, "add", RESULT, UPSTREAM_PROTOCOL)
    _git(repo, "commit", "-q", "-m", "upstream")


def test_upstream_steps_keep_the_shared_rows_and_put_the_scope_in_its_own_table(repo):
    _commit_upstream(repo, upstream_result())
    steps = DS.end_to_end_blocks(B.Sources(repo), RESULT, kind="upstream_end_to_end", label=DS.UPSTREAM_END_TO_END_LABEL)
    assert steps[0]["title"] == "종단 시연 seed 2040 U1·reward: 경고부터 교정 판정까지"
    values = _values(steps)
    assert values["특이성 관문"] == "등록된 보조 검사 없음" and values["교정 정책 고정 평가 판정"] == "not_recovered"
    assert not [name for name in values if "범위" in name or "readback" in name]  # 범위 점검·readback은 대조 표에 둔다


def test_scope_block_shows_the_check_the_judgement_and_the_contrast(repo):
    _commit_upstream(repo, upstream_result())
    sources = B.Sources(repo)
    blocks = DS.scope_blocks(sources, RESULT)
    assert blocks[0]["title"] == "교정 범위 점검과 재학습 대조(seed 2040)"
    assert blocks[0]["status_sources"] == [RESULT, UPSTREAM_PROTOCOL]
    assert _values(blocks) == {
        "점검 방식": "대조 모드: 판정 기록 뒤 재학습 1회 · 결정적 검사",
        "확정 검사 항별 상대 오차(문턱 0.05)": "track_ang_vel_z_exp 0.97 · track_lin_vel_xy_exp 0.93",
        "등록 교정과 되돌리는 함수": "restore_body_frame_reward → track_lin_vel_xy_exp",
        "점검 판정 · 등록 표적 밖 이상 항": "scope_insufficient · track_ang_vel_z_exp",
        "대조 범주(점검 × 고정 평가 판정)": "insufficient_and_not_recovered",
        "교정 승인 성격": ("대조 측정용 승인(운영 기준 예외) · 요청 근거 scope_insufficient · 체크리스트 미충족 없음 · "
                     "강제 모드였다면 재학습 전 종료"),
        "사전 예측 적중(점검 · 교정 · 대조)": "예 · 예 · 예",
        "교정 readback(등록 표적만 대조)": f"restored_to_clean · {READBACK_NOTE}",
        "보조 지문: 원본과 다른 함수(판정에 쓰지 않음)": "ang_vel_xy_l2, lin_vel_z_l2, track_ang_vel_z_exp"}
    proposal = _values(DS.correction_proposal_blocks(sources, RESULT))
    assert proposal["등록 규칙이 고른 회복 방식"] == "restart"
    assert _cost_values(sources, [(RESULT, "upstream_end_to_end", "설계자 밖 결함 U1")])[0][:2] == [
        "설계자 밖 결함 U1 seed 2040", "560 / 571"]


@pytest.mark.parametrize("change,expected", [
    ({}, "대조 측정용 승인(운영 기준 예외) · 요청 근거 scope_insufficient · 체크리스트 미충족 3·5·6 · 강제 모드였다면 재학습 전 종료"),
    ({"verdict": "scope_sufficient"}, "승인 · 요청 근거 scope_sufficient · 체크리스트 미충족 3·5·6"),
    ({"bound": False}, "승인 · 범위 점검 미결합"),
    ({"receipt": None}, "-")])
def test_approval_nature_comes_only_from_the_bound_request_and_the_protocol(repo, change, expected):
    result = upstream_result()
    result["correction"]["model_proposal"]["unmet_checklist"] = ["3", "5", "6"]
    result["correction"]["scope_check_bound"] = change.get("bound", True)
    if "verdict" in change:
        result["scope_check"]["result"]["verdict"] = change["verdict"]
    if "receipt" in change:
        result["correction"]["receipt"] = change["receipt"]
    if change.get("bound") is False:
        expected = "승인 · 범위 점검 미결합"
    _commit_upstream(repo, result)
    assert _values(DS.scope_blocks(B.Sources(repo), RESULT))["교정 승인 성격"] == expected


def test_approval_nature_waits_for_the_demo_protocol(repo):
    _commit_upstream(repo, upstream_result())
    _git(repo, "rm", "-q", "--cached", UPSTREAM_PROTOCOL)
    _git(repo, "commit", "-q", "-m", "protocol out")
    sources = B.Sources(repo)
    row = next(r for r in DS.scope_blocks(sources, RESULT)[0]["rows"] if r[0] == "교정 승인 성격")
    assert row[1] == {"value": None, "source": UPSTREAM_PROTOCOL, "digits": None}
    assert sources.rows[UPSTREAM_PROTOCOL]["status"] == "waiting"


def test_scope_block_of_a_demo_that_ended_before_the_check_shows_no_record(repo):
    ended = upstream_result(contrast=None)
    ended.update(outcome="loop_not_confirmed", scope_check=None, prediction_results={"scope_check": None})
    ended["correction"] = None
    _commit_upstream(repo, ended)
    values = _values(DS.scope_blocks(B.Sources(repo), RESULT))
    assert values["점검 판정 · 등록 표적 밖 이상 항"] == "-" and values["대조 범주(점검 × 고정 평가 판정)"] == "-"
    assert values["점검 방식"] == "-" and values["교정 readback(등록 표적만 대조)"] == "-"
    assert values["사전 예측 적중(점검 · 교정 · 대조)"] == "- · - · -"


def test_scope_slot_waits_without_a_result_and_rejects_other_demos(repo):
    sources = B.Sources(repo)
    assert DS.scope_blocks(sources, None) == [{"type": "slot", "label": DS.SCOPE_LABEL, "source": "slot:upstream_end_to_end"}]
    assert sources.rows["slot:upstream_end_to_end"]["status"] == "waiting"
    sources = B.Sources(repo)  # 보상 시연(v1) 결과는 범위 점검 자리에 들어가지 않는다
    assert DS.scope_blocks(sources, RESULT)[0]["type"] == "slot"
    assert (sources.rows[RESULT]["status"], sources.rows[RESULT]["reason"]) == ("mismatch", "not_upstream_result")


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
    texts = list(DS.COST_COLUMNS)
    blocks = DS.end_to_end_blocks(sources, RESULT) + DS.correction_proposal_blocks(sources, RESULT)
    for block in blocks + [DS.cost_block(sources, [(RESULT, "end_to_end", "보상 결함")])]:
        texts.append(block.get("title", ""))
        for row in block["rows"]:
            texts.append(row[0])
            texts += [str(cell["value"]) for cell in row[1:] if isinstance(cell, dict)]
    texts += [DS.END_TO_END_LABEL, DS.PROPOSAL_LABEL, DS.MOTOR_END_TO_END_LABEL, DS.MOTOR_PROPOSAL_LABEL,
              DS.UPSTREAM_END_TO_END_LABEL, DS.SCOPE_LABEL, *DS.SCOPE_MODES.values()]
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
