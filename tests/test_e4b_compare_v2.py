"""E4b v2 비교기: 두 입력 세트 결합·실행 명령·seed 2030 세트로만 하는 채택 판정·점검 호출 수 재검증. 모델·GPU 없음.
보조 함수(결정·제안·사건 행)는 v1 테스트(test_e4b_compare.py)의 것을 쓴다."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / "src", ROOT / "bench", ROOT / "evals", ROOT / "tests"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from rl_triage import active_diagnosis as D  # noqa: E402
import correction_model_call as CM  # noqa: E402
import nat_decision_call as W1  # noqa: E402
import nat_decision_call_v2 as W2  # noqa: E402
from test_e4b_compare import X as V1, decision, events_for, proposal, sha, write  # noqa: E402

spec = importlib.util.spec_from_file_location("e4b_compare_v2_under_test", ROOT / "evals/e4b_compare_v2.py")
Y = importlib.util.module_from_spec(spec)
sys.modules["e4b_compare_v2_under_test"] = Y
spec.loader.exec_module(Y)


def make_inputs(model):
    import run_active_diagnosis as RA
    src = {"telemetry": {"path": "bench/runs_p0c/h02_s2027.telemetry.json"},
           "reference_telemetry": {"path": "bench/runs_p0c/baseline_p0c_s2027.telemetry.json"}}
    dpacket = RA.input_packet({"sources": src}, {"max_probes": 4, "probe_execution_budget_s": 180})
    dpath = write(model / "diagnosis.packet.json", dpacket)
    write(model / "diagnosis.call1.json", {"schema": "active_model_call_v1", "packet_sha256": sha(dpath),
                                            "content": json.dumps(decision(dpacket)), "input_tokens": 3000,
                                            "output_tokens": 300, "execution_wall_s": 9.0})
    rule = {"mode": "restart", "rule": "r", "correction_id": "restore_body_frame_reward", "standstill_ratio": 1.0}
    report = {"metrics": {"lin_vel_rmse_mps": 1.2}, "standstill": {"lin_vel_rmse_mps": 1.2}}
    cpacket = CM.build_packet(case="h02", loop_state_digest="s", receipts_digest="r", mechanism="reward",
                              confirming={"probe": "P_reward", "request_id": "q", "outcome": "abnormal",
                                          "measurement": {}}, observed={"P_reward": "abnormal"}, rule=rule,
                              source_report=report, source_report_sha256="aa", checkpoint_sha256="bb")
    cpath = write(model / "correction.packet.json", cpacket)
    write(model / "correction.call1.json", {"schema": CM.RECORD_SCHEMA, "packet_digest": CM.packet_digest(cpacket),
                                             "content": json.dumps(proposal()), "input_tokens": 2000,
                                             "output_tokens": 200, "execution_wall_s": 7.0})
    return {"dpacket": dpacket, "dpath": dpath, "cpacket": cpacket, "cpath": cpath}


@pytest.fixture
def world(tmp_path, monkeypatch):
    """임시 ROOT 아래에 두 입력 세트(seed 2030·2029)의 E4 산출물을 만든다."""
    monkeypatch.setattr(Y, "ROOT", tmp_path)
    monkeypatch.setattr(Y, "PRIVATE", tmp_path / "bench/private/e4b_v2")
    monkeypatch.setattr(Y, "RESULTS", tmp_path / "evals/results")
    monkeypatch.setattr(Y, "E4_PRIVATE", tmp_path / "bench/private/e2e_reward_demo")
    monkeypatch.setattr(Y, "PROTOCOL", write(tmp_path / "protocol.json", {"x": 1}))
    sets = {name: make_inputs(tmp_path / "bench/private/e2e_reward_demo" / tag / "model")
            for name, tag in Y.INPUT_SETS.items()}
    protocol = {**Y.FIXED, "source_files": {name: Y.sha_lf(ROOT / name) for name in sorted(Y.CODE)},
                "inputs": {name: Y.bind_inputs(tag) for name, tag in Y.INPUT_SETS.items()}}
    monkeypatch.setattr(Y, "verify_protocol", lambda: (protocol, "c" * 40))
    return {"protocol": protocol, "sets": sets}


def nat_record(world, set_name, kind, run, attempt, status, body=None, error_type=None, checks=1):
    """load_run 재검증을 통과하는 가짜 v2 기록(사건·도구·궤적·제출 파일 포함)."""
    path = Y.run_path(set_name, kind, run, attempt)
    inputs = world["protocol"]["inputs"][set_name]
    packet_path = Y.ROOT / inputs[kind]["packet"]
    rows = events_for(run)
    write(path.with_suffix(".events.json"), {"events": rows})
    check_tool = "e4b2_check_decision" if kind == "diagnosis" else "e4b2_check_proposal"
    path.with_suffix(".tools.jsonl").write_text(
        "".join(json.dumps({"tool": name}) + "\n" for name in ["e4b2_list_fact_ids"] + [check_tool] * checks),
        encoding="utf-8")
    write(path.with_suffix(".atif.json"), {"steps": [1]})
    decision_value = None
    if status == "valid":
        write(path.with_suffix(".submission.json"), body)
        decision_value = W1.validate_submission(kind, json.loads(packet_path.read_text(encoding="utf-8")), body)
    source = world["protocol"]["source_files"]
    record = {"schema": W2.RECORD_SCHEMA, "kind": kind, "model": Y.MODEL, "temperature": 0, "nat_version": "1.9.0",
              "semantic_retries": 0, "precheck_limit": Y.PRECHECK_LIMIT, "precheck_calls": checks,
              "packet_sha256": sha(packet_path), "config_sha256_lf": source[f"configs/nat_demo_v2_{kind}.yml"],
              "tools_module_sha256_lf": source["src/rl_triage/nat_demo_tools_v2.py"],
              "base_tools_module_sha256_lf": source["src/rl_triage/nat_demo_tools.py"],
              "system_prompt_sha256": W1.text_sha(W2.system_prompt(kind)),
              "user_message_sha256": W1.text_sha(W1.user_message(kind, json.loads(packet_path.read_text(encoding="utf-8")))),
              "events_sha256": sha(path.with_suffix(".events.json")), "status": status, "decision": decision_value,
              "submission_sha256": sha(path.with_suffix(".submission.json")) if status == "valid" else None,
              "error_type": error_type, "wall_s": 20.0 + run, "tools_trace_sha256": sha(path.with_suffix(".tools.jsonl")),
              "atif_sha256": sha(path.with_suffix(".atif.json")), **W1.summarize_events(rows)}
    write(path, record)
    return path


def body(world, set_name, kind):
    packets = world["sets"][set_name]
    if kind == "diagnosis":
        return {"kind": "diagnosis", "decision": D.validate_decision(decision(packets["dpacket"]), packets["dpacket"])}
    return {"kind": "correction", "proposal": proposal(),
            "validation": CM.validate_model_proposal(proposal(), packets["cpacket"])}


def fill(world, set_name, valid_runs):
    for kind in Y.KINDS:
        for run in range(1, Y.RUNS_PER_KIND + 1):
            ok = run <= valid_runs
            nat_record(world, set_name, kind, run, 1, "valid" if ok else "invalid",
                       body(world, set_name, kind) if ok else None, None if ok else "SubmissionRejected")


def test_protocol_keeps_v1_numbers_and_binds_both_input_sets(world):
    dynamic = {"source_files": {"a": "1"}, "inputs": {"s2030": {}}}
    Y.check_protocol({**Y.FIXED, **dynamic}, dynamic)
    with pytest.raises(ValueError):
        Y.check_protocol({**Y.FIXED, **dynamic}, {**dynamic, "inputs": {"s2030": {"changed": True}}})
    assert Y.ADOPTION == V1.ADOPTION and Y.RUNS_PER_KIND == V1.RUNS_PER_KIND
    assert Y.MAX_ATTEMPTS_PER_RUN == V1.MAX_ATTEMPTS_PER_RUN
    assert Y.FIXED["input_sets"]["s2030"]["role"] == "adoption" and Y.ADOPTION_SET == "s2030"
    assert all((ROOT / name).is_file() for name in Y.CODE)
    assert not ({"src/rl_triage/nat_demo_tools.py", "evals/nat_decision_call.py"} - Y.CODE)  # 재사용한 v1 모듈도 결합
    inputs = world["protocol"]["inputs"]
    assert set(inputs) == {"s2030", "s2029"} and inputs["s2030"]["diagnosis"]["packet_sha256"] == sha(
        world["sets"]["s2030"]["dpath"])
    world["sets"]["s2029"]["cpath"].write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):  # 결합 뒤 입력이 바뀌면 거부
        Y.bind_inputs(Y.INPUT_SETS["s2029"])


def test_plan_emits_one_v2_command_per_set_kind_run(world):
    out = Y.plan()
    assert len(out["commands"]) == len(Y.INPUT_SETS) * len(Y.KINDS) * Y.RUNS_PER_KIND
    assert all("nat_decision_call_v2.py" in c for c in out["commands"])
    assert sum("/e4b_v2/s2029/correction/run5_a1.json" in c for c in out["commands"]) == 1


def test_adoption_uses_only_the_seed_2030_set(world):
    fill(world, "s2030", valid_runs=5)
    fill(world, "s2029", valid_runs=1)
    out = Y.summarize()
    assert out["adoption"]["decision"] == "adopt" and out["wording"] == Y.FIXED["wording"]["adopt"]
    assert out["sets"]["s2029"]["meets_numeric_rule"] is False  # 별도 분모로만 보고한다
    d = out["sets"]["s2030"]["kinds"]["diagnosis"]
    assert d["counts"]["valid"] == 5 and d["runs_using_precheck"] == 5 and d["precheck_calls"]["median"] == 1
    assert d["agreement_with_single_call"] == {"action": 5, "top1": 5, "top3_set": 5, "probe": 5}
    public = (Y.RESULTS / Y.RESULT_TAG / "summary.json").read_text(encoding="utf-8")
    assert str(Y.ROOT) not in public


def test_seed_2029_success_cannot_adopt_and_incomplete_sets_do_not_decide(world, monkeypatch):
    fill(world, "s2030", valid_runs=3)
    out = Y.summarize()  # seed 2029 세트가 아직 없다
    assert out["adoption"]["decision"] is None and not (Y.RESULTS / Y.RESULT_TAG).exists()
    fill(world, "s2029", valid_runs=5)
    out = Y.summarize()
    assert out["adoption"]["decision"] == "do_not_adopt" and out["sets"]["s2029"]["meets_numeric_rule"] is True


def test_tampered_precheck_count_or_v1_record_is_refused(world):
    path = nat_record(world, "s2030", "correction", 1, 1, "valid", body(world, "s2030", "correction"), checks=2)
    inputs = world["protocol"]["inputs"]["s2030"]
    record = json.loads(path.read_text(encoding="utf-8"))
    record["precheck_calls"] = 0  # 점검을 쓰지 않은 것처럼 기록
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="점검 호출 수"):
        Y.load_run("s2030", "correction", 1, inputs, world["protocol"])
    record.update(precheck_calls=2, schema=W1.RECORD_SCHEMA)  # v1 기록을 v2로 들이밀기
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="등록 입력"):
        Y.load_run("s2030", "correction", 1, inputs, world["protocol"])


def test_infra_retry_rule_matches_v1(world):
    inputs = world["protocol"]["inputs"]["s2030"]
    nat_record(world, "s2030", "diagnosis", 1, 1, "infra_error", error_type="APITimeoutError")
    assert Y.load_run("s2030", "diagnosis", 1, inputs, world["protocol"]) == {
        "run": 1, "status": "missing", "pending": "infra_retry"}
    nat_record(world, "s2030", "diagnosis", 1, 2, "valid", body(world, "s2030", "diagnosis"))
    assert Y.load_run("s2030", "diagnosis", 1, inputs, world["protocol"])["attempt"] == 2
    nat_record(world, "s2030", "diagnosis", 2, 1, "invalid", error_type="SubmissionRejected")
    nat_record(world, "s2030", "diagnosis", 2, 2, "valid", body(world, "s2030", "diagnosis"))
    with pytest.raises(ValueError, match="재시도"):  # 무효 뒤에 골라 뽑은 재시도
        Y.load_run("s2030", "diagnosis", 2, inputs, world["protocol"])
