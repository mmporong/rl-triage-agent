"""E4b v4 비교기: 사전등록 뒤 생기는 시연 v3 F2 첫 사례 입력의 결합(bind)·개발 분모·채택 판정(no_input 포함)·기록 재검증.
모델·GPU·git 없음(실험 커밋 대조는 가짜 함수로 바꾼다). 보조 함수(결정·사건 행)는 v1 테스트의 것을 쓴다."""
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
from rl_triage import correction_registry as R  # noqa: E402
from rl_triage import nat_demo_tools_v4 as T4  # noqa: E402
import correction_model_call as CM  # noqa: E402
import nat_decision_call as W1  # noqa: E402
import nat_decision_call_v4 as W4  # noqa: E402
from test_e4b_compare import X as V1, decision, events_for, sha, write  # noqa: E402

spec = importlib.util.spec_from_file_location("e4b_compare_v4_under_test", ROOT / "evals/e4b_compare_v4.py")
Z = importlib.util.module_from_spec(spec)
sys.modules["e4b_compare_v4_under_test"] = Z
spec.loader.exec_module(Z)
spec3 = importlib.util.spec_from_file_location("e4b_compare_v3_for_v4", ROOT / "evals/e4b_compare_v3.py")
V3 = importlib.util.module_from_spec(spec3)
sys.modules["e4b_compare_v3_for_v4"] = V3
spec3.loader.exec_module(V3)

CHECKLIST = {"1": "eval_metrics", "2": "confirmed_mechanism", "3": "confirming_probe", "4": "eval_metrics",
             "5": "unmet", "6": "registered_correction"}
REGISTERED = "e" * 40  # 이 프로토콜이 들어 있는 가짜 실험 커밋
WIDENED = {m: {**e, "recovery_modes": sorted(set(e["recovery_modes"]) | ({"restart"} if m == "actuator" else set()))}
           for m, e in R.REGISTRY.items()}  # 시연 v3처럼 모터 회복 방식을 넓힌 표(기본 표에는 restart가 없다)


@pytest.fixture
def world(tmp_path, monkeypatch):
    """임시 ROOT에 공개 원천을 복사하고 prepare한다. 시연 v3 F2 프로토콜도 만든다(시연 산출물은 각 테스트가 만든다)."""
    for source in Z.INPUT_SOURCES:
        (tmp_path / source).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / source, tmp_path / source)
    monkeypatch.setattr(Z, "ROOT", tmp_path)
    monkeypatch.setattr(Z, "PRIVATE", tmp_path / "bench/private/e4b_v4")
    monkeypatch.setattr(Z, "RESULTS", tmp_path / "evals/results")
    monkeypatch.setattr(Z, "DEMO_PRIVATE", tmp_path / "bench/private/e2e_demo_v3")
    monkeypatch.setattr(Z, "PROTOCOL", tmp_path / "bench/protocols/e4b_nat_compare_v4.json")
    Z.prepare()
    protocol = json.loads(Z.PROTOCOL.read_text(encoding="utf-8"))
    monkeypatch.setattr(Z, "verify_protocol", lambda: (protocol, "c" * 40))
    monkeypatch.setattr(Z, "committed_sha_lf", lambda commit, relative: Z.sha_lf(Z.PROTOCOL) if commit == REGISTERED else None)
    demo_protocol = write(tmp_path / Z.INPUT_SETS[Z.ADOPTION_SET]["demo_protocol"],
                          {"variant": "f2", "seed": 2031, "registry": WIDENED})
    return {"protocol": protocol, "demo_protocol": demo_protocol}


def make_demo(world, tag, at, *, variant="f2", commit=REGISTERED, decision_path="single", correction=True,
              terminal=None):
    """시연 v3 폴더: begin·packet·loop_start 단계, 진단 packet·단일 호출, (선택) 교정 packet·단일 호출·종료 기록."""
    import run_active_diagnosis as RA
    folder = Z.DEMO_PRIVATE / tag
    model = folder / "model"
    write(folder / "steps/00_begin.json", {"step": "begin", "at": at, "variant": variant, "decision_path": decision_path,
                                           "source_commit_sha": commit, "protocol_sha256_lf": Z.sha_lf(world["demo_protocol"])})
    src = {"telemetry": {"path": "bench/runs_p0c/h02_s2027.telemetry.json"},
           "reference_telemetry": {"path": "bench/runs_p0c/baseline_p0c_s2027.telemetry.json"}}
    dpacket = RA.input_packet({"sources": src}, {"max_probes": 4, "probe_execution_budget_s": 180})
    dpath = write(model / "diagnosis.packet.json", dpacket)
    write(folder / "steps/06_packet.json", {"step": "packet", "packet": {"path": Z.rel(dpath), "sha256": sha(dpath)}})
    call = write(model / "diagnosis.call1.json", {"schema": "active_model_call_v1", "packet_sha256": sha(dpath),
                                                   "content": json.dumps(decision(dpacket)), "input_tokens": 3000,
                                                   "output_tokens": 300, "execution_wall_s": 9.0})
    write(folder / "steps/07_loop_start.json", {"step": "loop_start", "call": {"record": Z.rel(call), "record_sha256": sha(call)}})
    if correction:
        rule = {"mode": "restart", "rule": "r", "correction_id": "restore_motor_torque_speed", "standstill_ratio": 0.5}
        cpacket = CM.build_packet(case="h06_s2031", loop_state_digest="s", receipts_digest="r", mechanism="actuator",
                                  confirming={"probe": "P_torque", "request_id": "q", "outcome": "abnormal", "measurement": {}},
                                  observed={"P_torque": "abnormal"}, rule=rule,
                                  source_report={"metrics": {"lin_vel_rmse_mps": 1.2}, "standstill": {"lin_vel_rmse_mps": 1.2}},
                                  source_report_sha256="aa", checkpoint_sha256="bb", registry=WIDENED)
        write(model / "correction.packet.json", cpacket)
        write(model / "correction.call1.json", {"schema": CM.RECORD_SCHEMA, "packet_digest": CM.packet_digest(cpacket),
                                                 "content": json.dumps(proposal_for()), "input_tokens": 2000,
                                                 "output_tokens": 200, "execution_wall_s": 7.0})
    if terminal:
        write(folder / "terminal.json", {"outcome": terminal})
    return folder


def proposal_for(mode="restart"):
    return {"correction_id": "restore_motor_torque_speed", "recovery_mode": mode, "cited_facts": ["standstill_ratio"],
            "checklist": dict(CHECKLIST), "hold_reason": None}


def entry_for(world, set_name, kind):
    if set_name == Z.ADOPTION_SET:
        return Z.adoption_inputs()["kinds"][kind]
    return world["protocol"]["development_inputs"][set_name][kind]


def body(world, set_name, kind):
    packet = json.loads((Z.ROOT / entry_for(world, set_name, kind)["packet"]).read_text(encoding="utf-8"))
    if kind == "diagnosis":
        return {"kind": "diagnosis", "decision": D.validate_decision(decision(packet), packet)}
    proposal = proposal_for()
    return {"kind": "correction", "proposal": proposal,
            "validation": CM.validate_model_proposal(proposal, packet, packet["registry"])}


def nat_record(world, set_name, kind, run, attempt, status, payload=None, error_type=None, checks=1):
    path = Z.run_path(set_name, kind, run, attempt)
    entry = entry_for(world, set_name, kind)
    packet_path = Z.ROOT / entry["packet"]
    rows = events_for(run)
    write(path.with_suffix(".events.json"), {"events": rows})
    check_tool = T4.PRECHECK_TOOLS[kind][0]
    path.with_suffix(".tools.jsonl").write_text(
        "".join(json.dumps({"tool": name}) + "\n" for name in ["e4b2_list_fact_ids"] + [check_tool] * checks),
        encoding="utf-8")
    write(path.with_suffix(".atif.json"), {"steps": [1]})
    decision_value = None
    if status == "valid":
        write(path.with_suffix(".submission.json"), payload)
        decision_value = W4.validate_submission(kind, json.loads(packet_path.read_text(encoding="utf-8")), payload)
    source = world["protocol"]["source_files"]
    record = {"schema": W4.RECORD_SCHEMA, "kind": kind, "model": Z.MODEL, "temperature": 0, "nat_version": "1.9.0",
              "semantic_retries": 0, "precheck_limit": Z.PRECHECK_LIMIT, "precheck_calls": checks,
              "packet_sha256": sha(packet_path), "config_sha256_lf": source[Z.code_rel(W4.CONFIGS[kind])],
              "tools_modules_sha256_lf": {n: source[Z.code_rel(p)] for n, p in W4.TOOLS_MODULES.items()},
              "system_prompt_sha256": W1.text_sha(W4.system_prompt(kind)),
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


def test_protocol_keeps_earlier_numbers_and_preregisters_the_selection_rule():
    assert Z.ADOPTION == V1.ADOPTION == V3.ADOPTION and Z.RUNS_PER_KIND == V3.RUNS_PER_KIND
    assert Z.MAX_ATTEMPTS_PER_RUN == V3.MAX_ATTEMPTS_PER_RUN and Z.FIXED["attempts"] == V3.FIXED["attempts"]
    assert Z.FIXED["status_definitions"] == V3.FIXED["status_definitions"]
    assert json.loads(json.dumps(Z.FIXED)) == Z.FIXED  # 파일로 쓴 뒤 그대로 비교된다(튜플 없음)
    assert Z.ADOPTION_SET not in V3.INPUT_SETS and Z.INPUT_SETS[Z.ADOPTION_SET]["role"] == "adoption"
    assert {n for n, s in Z.INPUT_SETS.items() if s["role"] != "adoption"} == {"h02_s2027"}
    assert Z.DISPLAY_DIGITS == T4.DISPLAY_DIGITS and Z.PRECHECK_LIMIT == T4.PRECHECK_LIMIT
    assert Z.FIXED["precheck"]["tools"] == {k: v[0] for k, v in T4.PRECHECK_TOOLS.items()}
    assert {Z.code_rel(p) for p in [*W4.CONFIGS.values(), *W4.TOOLS_MODULES.values()]} <= Z.CODE
    assert all((ROOT / name).is_file() for name in Z.CODE | Z.INPUT_SOURCES)
    dynamic = {"source_files": {"a": "1"}, "input_sources": {}, "development_inputs": {}}
    Z.check_protocol({**Z.FIXED, **dynamic}, dynamic)
    with pytest.raises(ValueError):
        Z.check_protocol({**Z.FIXED, **dynamic, "adoption_set": "h02_s2027"}, dynamic)


def test_before_the_demo_only_development_runs_are_planned(world):
    Z.check_protocol(world["protocol"], Z.dynamic_fields())  # 파일로 쓴 프로토콜이 다시 만든 값과 같다
    out = Z.plan()
    assert len(out["commands"]) == Z.RUNS_PER_KIND and out["adoption_inputs"].startswith("unbound")
    assert all("nat_decision_call_v4.py" in c and "h02_s2027" in c for c in out["commands"])
    assert Z.summarize()["adoption"]["reason"].startswith("채택 입력이 아직")
    with pytest.raises(ValueError, match="시작되지 않았다"):
        Z.bind()
    stored = Z.input_path("h02_s2027", "diagnosis")
    stored.write_text(stored.read_text(encoding="utf-8").replace("0", "1", 1), encoding="utf-8")
    with pytest.raises(ValueError, match="다시 만든"):
        Z.dynamic_fields()


def test_bind_waits_for_packets_and_requires_a_preregistered_first_demo(world):
    make_demo(world, "f1_demo", "2026-10-14T00:00:00+00:00", variant="f1")
    first = make_demo(world, "f2_first", "2026-10-14T01:00:00+00:00", correction=False)
    make_demo(world, "f2_later", "2026-10-14T02:00:00+00:00")  # 늦게 시작한 F2 시연은 고르지 않는다
    with pytest.raises(ValueError, match="교정 packet이 아직"):
        Z.bind()
    make_demo(world, "f2_first", "2026-10-14T01:00:00+00:00")  # 교정 packet·호출이 생겼다
    begin = first / "steps/00_begin.json"
    original = begin.read_text(encoding="utf-8")
    for key, value, message in (("source_commit_sha", "d" * 40, "실험 커밋"), ("decision_path", "nat_v3", "결정 경로"),
                                ("protocol_sha256_lf", "0" * 64, "begin 기록과")):
        begin.write_text(json.dumps({**json.loads(original), key: value}), encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            Z.bind()
    begin.write_text(original, encoding="utf-8")
    out = Z.bind()
    assert out["tag"] == "f2_first" and out["kinds"] == {"diagnosis": "bound", "correction": "bound"}
    with pytest.raises(FileExistsError):
        Z.bind()
    assert len(Z.plan()["commands"]) == 3 * Z.RUNS_PER_KIND
    packet = first / "model/correction.packet.json"
    packet.write_text(packet.read_text(encoding="utf-8").replace("restart", "resume"), encoding="utf-8")
    with pytest.raises(ValueError):
        Z.adoption_inputs()


def test_correction_packet_must_carry_the_demo_registry(world):
    folder = make_demo(world, "f2_first", "2026-10-14T01:00:00+00:00")
    packet_path = folder / "model/correction.packet.json"
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    packet_path.write_text(json.dumps({**packet, "registry": {m: {"correction_id": e["correction_id"],
                                                                  "recovery_modes": e["recovery_modes"]}
                                                              for m, e in R.REGISTRY.items()}}), encoding="utf-8")
    with pytest.raises(ValueError, match="등록 표"):
        Z.bind()


def test_demo_ending_before_correction_gives_no_input(world):
    make_demo(world, "f2_first", "2026-10-14T01:00:00+00:00", correction=False, terminal="loop_not_confirmed")
    assert Z.bind()["kinds"] == {"diagnosis": "bound", "correction": "not_reached"}
    out = Z.plan()
    assert len(out["commands"]) == 2 * Z.RUNS_PER_KIND and out["not_reached"] == {"f2_s2031/correction": "loop_not_confirmed"}
    fill(world, "f2_s2031", valid_runs=5, kinds=("diagnosis",))
    assert Z.summarize()["adoption"]["decision"] is None  # 개발 세트가 아직 없다
    fill(world, "h02_s2027", valid_runs=5)
    out = Z.summarize()
    assert out["adoption"]["decision"] == "no_input" and out["wording"] == Z.FIXED["wording"]["no_input"]
    assert out["sets"]["f2_s2031"]["kinds"]["correction"]["not_reached"] is True


def test_adoption_uses_only_the_demo_input_and_the_packet_registry(world):
    make_demo(world, "f2_first", "2026-10-14T01:00:00+00:00")
    Z.bind()
    packet = json.loads((Z.ROOT / entry_for(world, "f2_s2031", "correction")["packet"]).read_text(encoding="utf-8"))
    assert not CM.validate_model_proposal(proposal_for(), packet)["accepted"]  # 기본 표로는 restart가 무효다
    fill(world, "f2_s2031", valid_runs=5)
    fill(world, "h02_s2027", valid_runs=0)
    out = Z.summarize()
    assert out["adoption"]["decision"] == "adopt" and out["sets"]["h02_s2027"]["meets_numeric_rule"] is False
    kinds = out["sets"]["f2_s2031"]["kinds"]
    assert kinds["correction"]["decisions"] == {"propose": 5}
    assert kinds["correction"]["agreement_with_single_call"]["recovery_mode"] == 5  # 단일 호출도 packet 표로 검증한다
    assert kinds["diagnosis"]["agreement_with_single_call"]["action"] == 5
    assert out["adoption_input"]["tag"] == "f2_first"
    assert str(Z.ROOT) not in (Z.RESULTS / Z.RESULT_TAG / "summary.json").read_text(encoding="utf-8")


def test_development_success_cannot_adopt(world):
    make_demo(world, "f2_first", "2026-10-14T01:00:00+00:00")
    Z.bind()
    fill(world, "f2_s2031", valid_runs=3, kinds=("diagnosis",))
    fill(world, "f2_s2031", valid_runs=5, kinds=("correction",))
    fill(world, "h02_s2027", valid_runs=5)
    assert Z.summarize()["adoption"]["decision"] == "do_not_adopt"


def test_tampered_records_are_refused(world):
    make_demo(world, "f2_first", "2026-10-14T01:00:00+00:00")
    Z.bind()
    path = nat_record(world, "f2_s2031", "correction", 1, 1, "valid", body(world, "f2_s2031", "correction"), checks=2)
    entry = entry_for(world, "f2_s2031", "correction")
    assert Z.load_run("f2_s2031", "correction", 1, entry, world["protocol"])["status"] == "valid"
    record = json.loads(path.read_text(encoding="utf-8"))
    for key, value, message in (("precheck_calls", 0, "점검 호출 수"), ("schema", "e4b_nat_decision_v3", "등록 입력"),
                                ("tools_modules_sha256_lf", {"v4": "0" * 64}, "등록 입력")):
        path.write_text(json.dumps({**record, key: value}), encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            Z.load_run("f2_s2031", "correction", 1, entry, world["protocol"])
