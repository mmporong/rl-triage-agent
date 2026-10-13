"""E4b v3 비교기: 새 채택 입력(h02_s2027, 공개 기록에서 결정적 생성)·개발 분모·채택 판정·기록 재검증. 모델·GPU 없음.
보조 함수(결정·사건 행)는 v1·v2 테스트의 것을 쓴다."""
import importlib.util
import json
import shutil
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
import nat_decision_call_v3 as W3  # noqa: E402
from test_e4b_compare import X as V1, decision, events_for, sha, write  # noqa: E402
from test_e4b_compare_v2 import Y as V2, make_inputs  # noqa: E402

spec = importlib.util.spec_from_file_location("e4b_compare_v3_under_test", ROOT / "evals/e4b_compare_v3.py")
Z = importlib.util.module_from_spec(spec)
sys.modules["e4b_compare_v3_under_test"] = Z
spec.loader.exec_module(Z)

CHECKLIST = {"1": "eval_metrics", "2": "confirmed_mechanism", "3": "confirming_probe", "4": "eval_metrics",
             "5": "unmet", "6": "registered_correction"}


@pytest.fixture
def world(tmp_path, monkeypatch):
    """임시 ROOT에 공개 원천(고리 기록·P0-C 평가)을 복사하고 e2e 세트 입력을 만든 뒤 prepare한다."""
    for source in Z.INPUT_SOURCES:
        (tmp_path / source).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / source, tmp_path / source)
    monkeypatch.setattr(Z, "ROOT", tmp_path)
    monkeypatch.setattr(Z, "PRIVATE", tmp_path / "bench/private/e4b_v3")
    monkeypatch.setattr(Z, "RESULTS", tmp_path / "evals/results")
    monkeypatch.setattr(Z, "E4_PRIVATE", tmp_path / "bench/private/e2e_reward_demo")
    monkeypatch.setattr(Z, "PROTOCOL", tmp_path / "bench/protocols/e4b_nat_compare_v3.json")
    for name, spec_ in Z.INPUT_SETS.items():
        if spec_["source"] == "e2e":
            make_inputs(tmp_path / "bench/private/e2e_reward_demo" / spec_["tag"] / "model")
    Z.prepare()
    protocol = json.loads(Z.PROTOCOL.read_text(encoding="utf-8"))
    monkeypatch.setattr(Z, "verify_protocol", lambda: (protocol, "c" * 40))
    return {"protocol": protocol}


def proposal_for(packet):
    if packet["mechanism"] == "reward":
        return {"correction_id": "restore_body_frame_reward", "recovery_mode": "restart", "cited_facts": ["standstill_ratio"],
                "checklist": dict(CHECKLIST), "hold_reason": None}
    return {"correction_id": None, "recovery_mode": None, "cited_facts": ["confirmed_mechanism"],
            "checklist": {**CHECKLIST, "6": "unmet"}, "hold_reason": "exploration has no registered correction"}


def body(world, set_name, kind):
    packet = json.loads((Z.ROOT / world["protocol"]["inputs"][set_name][kind]["packet"]).read_text(encoding="utf-8"))
    if kind == "diagnosis":
        return {"kind": "diagnosis", "decision": D.validate_decision(decision(packet), packet)}
    proposal = proposal_for(packet)
    return {"kind": "correction", "proposal": proposal, "validation": CM.validate_model_proposal(proposal, packet)}


def nat_record(world, set_name, kind, run, attempt, status, payload=None, error_type=None, checks=1):
    path = Z.run_path(set_name, kind, run, attempt)
    inputs = world["protocol"]["inputs"][set_name]
    packet_path = Z.ROOT / inputs[kind]["packet"]
    rows = events_for(run)
    write(path.with_suffix(".events.json"), {"events": rows})
    check_tool = "e4b2_check_decision" if kind == "diagnosis" else "e4b3_check_proposal"
    path.with_suffix(".tools.jsonl").write_text(
        "".join(json.dumps({"tool": name}) + "\n" for name in ["e4b2_list_fact_ids"] + [check_tool] * checks),
        encoding="utf-8")
    write(path.with_suffix(".atif.json"), {"steps": [1]})
    decision_value = None
    if status == "valid":
        write(path.with_suffix(".submission.json"), payload)
        decision_value = W1.validate_submission(kind, json.loads(packet_path.read_text(encoding="utf-8")), payload)
    source = world["protocol"]["source_files"]
    record = {"schema": W3.RECORD_SCHEMA, "kind": kind, "model": Z.MODEL, "temperature": 0, "nat_version": "1.9.0",
              "semantic_retries": 0, "precheck_limit": Z.PRECHECK_LIMIT, "precheck_calls": checks,
              "packet_sha256": sha(packet_path), "config_sha256_lf": source[Z.code_rel(W3.CONFIGS[kind])],
              "tools_modules_sha256_lf": {n: source[Z.code_rel(p)] for n, p in W3.TOOLS_MODULES.items()},
              "system_prompt_sha256": W1.text_sha(W3.system_prompt(kind)),
              "user_message_sha256": W1.text_sha(W1.user_message(kind, json.loads(packet_path.read_text(encoding="utf-8")))),
              "events_sha256": sha(path.with_suffix(".events.json")), "status": status, "decision": decision_value,
              "submission_sha256": sha(path.with_suffix(".submission.json")) if status == "valid" else None,
              "error_type": error_type, "wall_s": 20.0 + run, "tools_trace_sha256": sha(path.with_suffix(".tools.jsonl")),
              "atif_sha256": sha(path.with_suffix(".atif.json")), **W1.summarize_events(rows)}
    write(path, record)
    return path


def fill(world, set_name, valid_runs, kinds=None):
    for kind in kinds or Z.INPUT_SETS[set_name]["kinds"]:
        for run in range(1, Z.RUNS_PER_KIND + 1):
            ok = run <= valid_runs
            nat_record(world, set_name, kind, run, 1, "valid" if ok else "invalid",
                       body(world, set_name, kind) if ok else None, None if ok else "SubmissionRejected")


def test_protocol_keeps_v1_v2_numbers_and_uses_a_new_adoption_input():
    assert Z.ADOPTION == V1.ADOPTION == V2.ADOPTION and Z.RUNS_PER_KIND == V2.RUNS_PER_KIND
    assert Z.MAX_ATTEMPTS_PER_RUN == V2.MAX_ATTEMPTS_PER_RUN and Z.FIXED["attempts"] == V2.FIXED["attempts"]
    assert Z.INPUT_SETS[Z.ADOPTION_SET]["role"] == "adoption" and Z.ADOPTION_SET not in V2.INPUT_SETS
    assert {name for name, s in Z.INPUT_SETS.items() if s["role"] != "adoption"} == {"s2030", "s2029", "h05_s2027"}
    assert all((ROOT / name).is_file() for name in Z.CODE | Z.INPUT_SOURCES)
    dynamic = {"source_files": {"a": "1"}, "input_sources": {}, "inputs": {}}
    Z.check_protocol({**Z.FIXED, **dynamic}, dynamic)
    with pytest.raises(ValueError):
        Z.check_protocol({**Z.FIXED, **dynamic, "adoption_set": "s2030"}, dynamic)


def test_prepare_builds_public_inputs_deterministically_and_detects_tampering(world):
    inputs = world["protocol"]["inputs"]
    Z.check_protocol(world["protocol"], Z.dynamic_fields())  # 파일로 쓴(JSON 왕복한) 프로토콜이 그대로 통과한다
    assert set(inputs["h02_s2027"]) == {"diagnosis", "correction"} and inputs["h02_s2027"]["correction"]["single_call"] is None
    assert set(inputs["h05_s2027"]) == {"correction"} and inputs["s2030"]["correction"]["single_call"]
    assert Z.bind_set("h02_s2027") == inputs["h02_s2027"]  # 같은 원천에서 다시 만들면 같다
    stored = Z.input_path("h02_s2027", "correction")
    stored.write_text(stored.read_text(encoding="utf-8").replace("restart", "resume"), encoding="utf-8")
    with pytest.raises(ValueError, match="다시 만든"):
        Z.bind_set("h02_s2027")
    out = Z.plan()
    assert len(out["commands"]) == (2 + 1 + 1 + 1) * Z.RUNS_PER_KIND
    assert all("nat_decision_call_v3.py" in c for c in out["commands"])


def test_adoption_uses_only_the_new_input(world):
    fill(world, "h02_s2027", valid_runs=5)
    for name in ("s2030", "s2029", "h05_s2027"):
        fill(world, name, valid_runs=0)
    out = Z.summarize()
    assert out["adoption"]["decision"] == "adopt" and out["sets"]["s2030"]["meets_numeric_rule"] is False
    assert out["sets"]["h02_s2027"]["kinds"]["correction"]["decisions"] == {"propose": 5}
    assert out["sets"]["h02_s2027"]["kinds"]["correction"]["agreement_with_single_call"] is None  # 단일 호출 기록 없음
    assert str(Z.ROOT) not in (Z.RESULTS / Z.RESULT_TAG / "summary.json").read_text(encoding="utf-8")


def test_development_success_cannot_adopt_and_hold_branch_is_counted(world):
    fill(world, "h02_s2027", valid_runs=3, kinds=("diagnosis",))
    fill(world, "h02_s2027", valid_runs=5, kinds=("correction",))
    for name in ("s2030", "s2029"):
        fill(world, name, valid_runs=5)
    out = Z.summarize()  # h05_s2027 세트가 아직 없다
    assert out["adoption"]["decision"] is None and not (Z.RESULTS / Z.RESULT_TAG).exists()
    fill(world, "h05_s2027", valid_runs=5)
    out = Z.summarize()
    assert out["adoption"]["decision"] == "do_not_adopt"
    assert out["sets"]["h05_s2027"]["kinds"]["correction"]["decisions"] == {"HOLD": 5}


def test_tampered_records_are_refused(world):
    payload = body(world, "h02_s2027", "correction")
    path = nat_record(world, "h02_s2027", "correction", 1, 1, "valid", payload, checks=2)
    inputs, protocol = world["protocol"]["inputs"]["h02_s2027"], world["protocol"]
    record = json.loads(path.read_text(encoding="utf-8"))
    for key, value, message in (("precheck_calls", 0, "점검 호출 수"), ("schema", "e4b_nat_decision_v2", "등록 입력"),
                                ("tools_modules_sha256_lf", {"v3": "0" * 64}, "등록 입력")):
        path.write_text(json.dumps({**record, key: value}), encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            Z.load_run("h02_s2027", "correction", 1, inputs, protocol)
