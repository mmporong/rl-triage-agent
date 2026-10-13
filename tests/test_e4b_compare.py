"""E4b 비교기: 입력 결합·인프라 재시도 선택·일치·채택 판정·변조 탐지·공개 요약. 모델·GPU 없음."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / "src", ROOT / "bench", ROOT / "evals"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from rl_triage import active_diagnosis as D  # noqa: E402
from rl_triage import probe_loop as L  # noqa: E402
import correction_model_call as CM  # noqa: E402
import nat_decision_call as W  # noqa: E402

spec = importlib.util.spec_from_file_location("e4b_compare_under_test", ROOT / "evals/e4b_compare.py")
X = importlib.util.module_from_spec(spec)
sys.modules["e4b_compare_under_test"] = X
spec.loader.exec_module(X)

TAG = "e2e_t"
RANKING = ["reward", "actuator", "exploration", "optimizer", "physics", "termination"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def decision(packet, probe="P_reward"):
    fact = sorted(packet["facts"])[0]
    return {"ranking": RANKING, "probe": probe, "action": "continue", "reason": "r",
            "claims": [{"id": fact, "value": packet["facts"][fact]}],
            "expected_outcomes": {m: L.PROBES[probe]["expect"][m] for m in RANKING[:3]}}


def proposal(mode="restart"):
    return {"correction_id": "restore_body_frame_reward", "recovery_mode": mode, "cited_facts": ["standstill_ratio"],
            "checklist": {"1": "eval_metrics", "2": "confirmed_mechanism", "3": "confirming_probe", "4": "eval_metrics",
                          "5": "unmet", "6": "registered_correction"}, "hold_reason": None}


@pytest.fixture
def world(tmp_path, monkeypatch):
    """임시 ROOT 아래에 E4 산출물(두 packet과 단일 호출 기록)을 만든다."""
    import run_active_diagnosis as RA
    monkeypatch.setattr(X, "ROOT", tmp_path)
    monkeypatch.setattr(X, "PRIVATE", tmp_path / "bench/private/e4b")
    monkeypatch.setattr(X, "RESULTS", tmp_path / "evals/results")
    monkeypatch.setattr(X, "E4_PRIVATE", tmp_path / "bench/private/e2e_reward_demo")
    monkeypatch.setattr(X, "PROTOCOL", write(tmp_path / "protocol.json", {"x": 1}))
    protocol = {**X.FIXED, "source_files": {name: X.sha_lf(ROOT / name) for name in sorted(X.CODE)}}
    monkeypatch.setattr(X, "verify_protocol", lambda: (protocol, "c" * 40))
    model = tmp_path / "bench/private/e2e_reward_demo" / TAG / "model"
    src = {"telemetry": {"path": "bench/runs_p0c/h02_s2027.telemetry.json"},
           "reference_telemetry": {"path": "bench/runs_p0c/baseline_p0c_s2027.telemetry.json"}}
    dpacket = RA.input_packet({"sources": src}, {"max_probes": 4, "probe_execution_budget_s": 180})
    dpath = write(model / "diagnosis.packet.json", dpacket)
    write(model / "diagnosis.call1.json", {"schema": "active_model_call_v1", "packet_sha256": sha(dpath),
                                            "content": json.dumps(decision(dpacket)), "input_tokens": 3000,
                                            "output_tokens": 300, "execution_wall_s": 9.0})
    rule = {"mode": "restart", "rule": "r", "correction_id": "restore_body_frame_reward", "standstill_ratio": 1.0}
    report = {"metrics": {"lin_vel_rmse_mps": 1.2}, "standstill": {"lin_vel_rmse_mps": 1.2}}
    cpacket = CM.build_packet(case="h02_s2029", loop_state_digest="s", receipts_digest="r", mechanism="reward",
                              confirming={"probe": "P_reward", "request_id": "q", "outcome": "abnormal",
                                          "measurement": {}}, observed={"P_reward": "abnormal"}, rule=rule,
                              source_report=report, source_report_sha256="aa", checkpoint_sha256="bb")
    cpath = write(model / "correction.packet.json", cpacket)
    write(model / "correction.call1.json", {"schema": CM.RECORD_SCHEMA, "packet_digest": CM.packet_digest(cpacket),
                                             "content": json.dumps(proposal()), "input_tokens": 2000,
                                             "output_tokens": 200, "execution_wall_s": 7.0})
    return {"protocol": protocol, "dpacket": dpacket, "dpath": dpath, "cpacket": cpacket, "cpath": cpath}


def events_for(run):
    """LLM 3회(토큰 합 9000+run / 30)와 도구 2회의 사건 행."""
    rows = []
    for i, tokens in enumerate((3000 + run, 3000, 3000)):
        rows += [{"type": "LLM_START", "uuid": f"l{i}", "name": "m", "t": float(i), "span_t": None, "usage": None},
                 {"type": "LLM_END", "uuid": f"l{i}", "name": "m", "t": i + 0.5, "span_t": float(i),
                  "usage": {"prompt_tokens": tokens, "completion_tokens": 10, "total_tokens": tokens + 10}}]
    for j, tool in enumerate(("e4b_list_fact_ids", "e4b_submit_decision")):
        rows += [{"type": "FUNCTION_START", "uuid": f"f{j}", "name": tool, "t": 0.6 + j, "span_t": None, "usage": None},
                 {"type": "FUNCTION_END", "uuid": f"f{j}", "name": tool, "t": 0.7 + j, "span_t": None, "usage": None}]
    return rows


def nat_record(world, kind, run, attempt, status, body=None, error_type=None, rejected=False):
    """load_run 재검증을 통과하는 가짜 NAT 기록(사건·도구·궤적·제출·거절 파일 포함)."""
    path = X.run_path(TAG, kind, run, attempt)
    packet_path = world["dpath"] if kind == "diagnosis" else world["cpath"]
    rows = events_for(run)
    write(path.with_suffix(".events.json"), {"events": rows})
    path.with_suffix(".tools.jsonl").write_text('{"tool": "e4b_list_fact_ids"}\n', encoding="utf-8")
    write(path.with_suffix(".atif.json"), {"steps": [1]})
    decision_value = None
    if status == "valid":
        write(path.with_suffix(".submission.json"), body)
        packet = world["dpacket"] if kind == "diagnosis" else world["cpacket"]
        decision_value = W.validate_submission(kind, packet, body)
    record = {"schema": W.RECORD_SCHEMA, "kind": kind, "model": X.MODEL, "temperature": 0, "nat_version": "1.9.0",
              "semantic_retries": 0, "packet_sha256": sha(packet_path),
              "config_sha256_lf": world["protocol"]["source_files"][f"configs/nat_demo_{kind}.yml"],
              "tools_module_sha256_lf": world["protocol"]["source_files"]["src/rl_triage/nat_demo_tools.py"],
              "system_prompt_sha256": X.expected_prompt_sha(kind),
              "user_message_sha256": W.text_sha(W.user_message(kind, json.loads(packet_path.read_text(encoding="utf-8")))),
              "events_sha256": sha(path.with_suffix(".events.json")), "status": status, "decision": decision_value,
              "submission_sha256": sha(path.with_suffix(".submission.json")) if status == "valid" else None,
              "error_type": error_type, "wall_s": 20.0 + run,
              "tools_trace_sha256": sha(path.with_suffix(".tools.jsonl")),
              "atif_sha256": sha(path.with_suffix(".atif.json")), **W.summarize_events(rows)}
    if rejected:
        write(path.with_suffix(".submission.rejected.json"), {"reasons": ["x"]})
        record["rejection_sha256"] = sha(path.with_suffix(".submission.rejected.json"))
    write(path, record)
    return path


def diag_body(world, probe="P_reward"):
    return {"kind": "diagnosis", "decision": D.validate_decision(decision(world["dpacket"], probe), world["dpacket"])}


def corr_body(world, mode="restart"):
    verdict = CM.validate_model_proposal(proposal(mode), world["cpacket"])
    return {"kind": "correction", "proposal": proposal(mode), "validation": verdict}


def test_protocol_contract_and_change_detection():
    dynamic = {"source_files": {"a": "1"}}
    X.check_protocol({**X.FIXED, **dynamic}, dynamic)
    with pytest.raises(ValueError):
        X.check_protocol({**X.FIXED, **dynamic, "runs_per_kind": 1}, dynamic)
    with pytest.raises(ValueError):
        X.check_protocol({**X.FIXED, **dynamic}, {"source_files": {"a": "2"}})
    assert X.FIXED["adoption_rule"]["min_valid_runs"] == 4 and "일치율은 채택 조건에 쓰지 않는다" in X.FIXED["adoption_rule"]["rule"]
    assert all((ROOT / name).is_file() for name in X.CODE)


def test_plan_binds_inputs_and_emits_one_command_per_run(world):
    out = X.plan(TAG)
    assert out["inputs"]["diagnosis"]["packet_sha256"] == sha(world["dpath"])
    assert len(out["commands"]) == 2 * X.RUNS_PER_KIND and all("--kind" in c for c in out["commands"])
    assert X.plan(TAG)["inputs"] == out["inputs"]  # 다시 계획해도 같은 결합
    world["dpath"].write_text(world["dpath"].read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError):  # 결합 뒤 E4 입력이 바뀌었다
        X.plan(TAG)


def test_summary_adopts_with_infra_retry_and_counts_agreement(world):
    X.plan(TAG)
    for run in range(1, 5):
        nat_record(world, "diagnosis", run, 1, "valid", diag_body(world, "P_noise" if run == 4 else "P_reward"))
    nat_record(world, "diagnosis", 5, 1, "infra_error", error_type="APITimeoutError")
    nat_record(world, "diagnosis", 5, 2, "invalid", error_type="SubmissionRejected", rejected=True)
    for run in range(1, 6):
        nat_record(world, "correction", run, 1, "valid", corr_body(world, "resume" if run == 5 else "restart"))
    out = X.summarize(TAG)
    d, c = out["kinds"]["diagnosis"], out["kinds"]["correction"]
    assert d["counts"] == {"valid": 4, "invalid": 1, "infra_error": 0, "missing": 0} and d["non_infra_runs"] == 5
    assert d["runs"][4]["attempt"] == 2  # 인프라 오류 뒤 두 번째 시도
    assert d["agreement_with_single_call"] == {"action": 4, "top1": 4, "top3_set": 4, "probe": 3}
    assert c["agreement_with_single_call"]["recovery_mode"] == 4 and c["agreement_denominator"] == 5
    assert d["single_call_costs"]["input_tokens"] == 3000 and d["input_tokens"]["median"] == 9003  # 실행 5개 9001~9005
    assert out["adoption"]["decision"] == "adopt" and out["wording"] == X.FIXED["wording"]["adopt"]
    public = (world["dpath"].parents[5] / "evals/results" / f"e4b_{TAG}" / "summary.json").read_text(encoding="utf-8")
    assert str(world["dpath"].parents[5]) not in public


def test_incomplete_runs_do_not_decide_and_low_validity_does_not_adopt(world):
    X.plan(TAG)
    for run in range(1, 5):
        nat_record(world, "diagnosis", run, 1, "valid", diag_body(world))
    for run in range(1, 6):
        nat_record(world, "correction", run, 1, "valid", corr_body(world))
    out = X.summarize(TAG)
    assert out["adoption"]["decision"] is None and not (X.RESULTS / f"e4b_{TAG}").exists()
    nat_record(world, "diagnosis", 5, 1, "infra_error", error_type="APITimeoutError")
    out = X.summarize(TAG)  # 첫 시도만 인프라 오류: 재시도 대기라 판정하지 않는다
    assert out["kinds"]["diagnosis"]["runs"][4] == {"run": 5, "status": "missing", "pending": "infra_retry"}
    assert out["adoption"]["decision"] is None and not (X.RESULTS / f"e4b_{TAG}").exists()
    nat_record(world, "diagnosis", 5, 2, "infra_error", error_type="APITimeoutError")
    out = X.summarize(TAG)  # 인프라 오류를 빼면 4회, 유효 4회 → 채택
    assert out["kinds"]["diagnosis"]["counts"]["infra_error"] == 1 and out["adoption"]["decision"] == "adopt"


def test_low_validity_and_not_reached(world):
    X.plan(TAG)
    for run in range(1, 6):
        nat_record(world, "diagnosis", run, 1, "valid" if run <= 3 else "invalid",
                   diag_body(world) if run <= 3 else None)
        nat_record(world, "correction", run, 1, "valid", corr_body(world))
    assert X.summarize(TAG)["adoption"]["decision"] == "do_not_adopt"
    out = X.adoption({"diagnosis": {"status": "not_reached"}, "correction": {"complete": True, "non_infra_runs": 5,
                                                                              "counts": {"valid": 5}}})
    assert out["decision"] == "do_not_adopt"


def test_tampered_submission_or_foreign_record_is_rejected(world):
    X.plan(TAG)
    path = nat_record(world, "diagnosis", 1, 1, "valid", diag_body(world))
    sub = path.with_suffix(".submission.json")
    sub.write_text(sub.read_text(encoding="utf-8").replace("P_reward", "P_noise"), encoding="utf-8")
    inputs = json.loads((X.PRIVATE / TAG / "inputs.json").read_text(encoding="utf-8"))
    with pytest.raises(ValueError):
        X.load_run(TAG, "diagnosis", 1, inputs, world["protocol"])
    record = json.loads(path.read_text(encoding="utf-8"))
    record["config_sha256_lf"] = "0" * 64
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError):
        X.load_run(TAG, "diagnosis", 1, inputs, world["protocol"])


def test_retry_outside_rule_events_tamper_and_valid_with_rejection_are_refused(world):
    X.plan(TAG)
    inputs = json.loads((X.PRIVATE / TAG / "inputs.json").read_text(encoding="utf-8"))
    nat_record(world, "diagnosis", 1, 1, "invalid", error_type="SubmissionRejected")
    nat_record(world, "diagnosis", 1, 2, "valid", diag_body(world))  # 무효 뒤에 골라 뽑은 재시도
    with pytest.raises(ValueError, match="재시도"):
        X.load_run(TAG, "diagnosis", 1, inputs, world["protocol"])
    path = nat_record(world, "diagnosis", 2, 1, "valid", diag_body(world))
    record = json.loads(path.read_text(encoding="utf-8"))
    record["input_tokens"] = 1  # 사건 파일과 다른 토큰
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="사건 파일"):
        X.load_run(TAG, "diagnosis", 2, inputs, world["protocol"])
    nat_record(world, "diagnosis", 3, 1, "valid", diag_body(world), rejected=True)
    with pytest.raises(ValueError, match="거절 기록"):
        X.load_run(TAG, "diagnosis", 3, inputs, world["protocol"])
    path = nat_record(world, "diagnosis", 4, 1, "valid", diag_body(world))
    record = json.loads(path.read_text(encoding="utf-8"))
    record["system_prompt_sha256"] = "0" * 64  # 다른 프롬프트로 만든 기록
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="프롬프트"):
        X.load_run(TAG, "diagnosis", 4, inputs, world["protocol"])


def test_missing_e4_packets_are_not_reached(tmp_path, monkeypatch):
    monkeypatch.setattr(X, "E4_PRIVATE", tmp_path / "e4")
    inputs = X.bind_inputs(TAG)
    assert inputs["diagnosis"] == {"not_reached": True} and inputs["correction"] == {"not_reached": True}
