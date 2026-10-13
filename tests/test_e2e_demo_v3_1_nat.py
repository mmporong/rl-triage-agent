"""시연 v3.1 NAT 결정 경로(nat_v4) 통합: 실제 E4b v4 worker(가짜 LLM 스트림, 네트워크 차단)가 쓴 진단·교정 기록이
v3.1의 고리 시작·교정 제안·승인·교정 실행을 지나가는지와 인프라 오류 재시도. GPU·모델 없음."""
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "bench", ROOT / "evals", ROOT / "tests"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import demo_correction_v3_1 as DC  # noqa: E402
import e2e_demo_v3_1 as E  # noqa: E402
import nat_decision_call_v4 as W4  # noqa: E402
from rl_triage import nat_demo_tools_v4 as T4  # noqa: E402
from rl_triage.correction_ledger import LedgerError  # noqa: E402
from test_e2e_demo_v3_1 import EXPERIMENT, fake_executor, ready_for_correction, use_tmp  # noqa: E402
from test_nat_demo import decision, deny_external_network, scripted_stream  # noqa: E402
from test_nat_demo_v3 import propose  # noqa: E402

NAT = {"decision_path": "nat_v4"}
REF_PROBES = ROOT / "evals/results/e2e_demo_v3_f2_s2031_20261013_ref/probes/baseline_p0c_s2031.json"
ENV_KEYS = ("E4B_KIND", "E4B_PACKET", "E4B_PACKET_SHA256", "E4B_SUBMISSION", "E4B_TRACE")


@pytest.fixture
def demo(tmp_path, monkeypatch):
    return use_tmp(E, tmp_path, monkeypatch)


def nat_call(monkeypatch, kind, packet_path, output, turns=None, runner=None):
    """실제 W4 worker로 결정 기록을 만든다. turns가 가짜 스트림, runner가 있으면 그 실행기(인프라 오류 재현)."""
    deny_external_network(monkeypatch)
    monkeypatch.setenv("NVIDIA_API_KEY", "offline-unused-placeholder")
    for key in ENV_KEYS:  # worker가 os.environ에 쓰는 값을 테스트 뒤 되돌린다
        monkeypatch.setenv(key, "")
    if turns is not None:
        scripted_stream(monkeypatch, turns, [])
    kwargs = {"require_posix": False} | ({"runner": runner} if runner else {})
    code = W4.call(kind, packet_path, output, **kwargs)
    return code, json.loads(output.read_text(encoding="utf-8"))


async def provider_timeout(*args):
    raise TimeoutError("provider timeout")


def ready_for_loop(demo, monkeypatch, tmp_path, tag):
    """f1 nat_v4 시연을 packet 단계까지 만든다. 진단 packet 경로를 돌려준다."""
    import run_active_diagnosis as RA
    ctx = demo.ctx_of("f1")
    demo._write_step(tag, "begin", demo._guard(tag, "begin"), EXPERIMENT, ctx.protocol,
                     {"variant": "f1", "decision_path": "nat_v4", "seed_unused": {"hits": []}})
    monkeypatch.setattr(demo, "verify_for", lambda t, isaaclab: (NAT, EXPERIMENT))
    monkeypatch.setattr(demo, "_check_artifacts", lambda t, arm: {"checkpoint": {"sha256": "c" * 64}})
    shutil.copyfile(REF_PROBES, tmp_path / "ref.json")
    sources = {"telemetry": {"path": "bench/runs_p0c/h02_s2027.telemetry.json"},
               "reference_telemetry": {"path": "bench/runs_p0c/baseline_p0c_s2027.telemetry.json"}}
    packet_path = demo.folder_for(tag) / "model" / "diagnosis.packet.json"
    packet_path.parent.mkdir(parents=True)
    packet_path.write_text(json.dumps(RA.input_packet({"sources": sources}, {"max_probes": 4,
                                                                             "probe_execution_budget_s": 180})),
                           encoding="utf-8")
    records = {"evaluate": {"gate": {"outcome": "failure_present", "standstill_ratio": {"NONE": 0.13, "FAULT": 0.95}}},
               "reference": {"report": {"path": "ref.json"}},
               "packet": {"packet": {"path": demo._rel(packet_path), "sha256": demo.sha(packet_path)}}}
    for name in demo.STEPS[1:demo.STEPS.index("loop_start")]:
        demo._write_step(tag, name, demo._guard(tag, name), EXPERIMENT, ctx.protocol, records.get(name, {}))
    return ctx, packet_path


def diagnosis_turns(packet):
    fid = next(k for k, v in sorted(packet["facts"].items()) if isinstance(v, float) and len(repr(v)) > 12)
    args = json.dumps({**decision(packet), "claims": [{"id": fid, "value": T4.display(packet["facts"][fid])}]})
    return fid, [("e4b4_check_decision", args, ""), ("e4b4_submit_decision", args, ""), (None, None, "Submitted.")]


def test_nat_diagnosis_record_starts_the_loop_after_one_infra_retry(demo, monkeypatch, tmp_path):
    ctx, packet_path = ready_for_loop(demo, monkeypatch, tmp_path, "n1")
    model = packet_path.parent
    code, record = nat_call(monkeypatch, "diagnosis", packet_path, model / "diagnosis.nat1.json", runner=provider_timeout)
    assert code == 2 and record["status"] == "infra_error"
    with pytest.raises(ValueError, match="diagnosis.nat1.json"):  # 기록 이름은 결정 경로를 따른다(call1이 아니다)
        demo.loop_start("n1", ROOT, model / "diagnosis.call1.json")
    assert demo.loop_start("n1", ROOT, model / "diagnosis.nat1.json") == {"status": "retry_allowed", "attempt": 1}
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    fid, turns = diagnosis_turns(packet)
    code, record = nat_call(monkeypatch, "diagnosis", packet_path, model / "diagnosis.nat2.json", turns)
    assert code == 0 and record["status"] == "valid"
    tampered = model / "tampered.nat.json"  # 제출 파일과 다른 결정을 적은 기록은 받지 않는다
    tampered.write_text(json.dumps({**record, "decision": {**record["decision"], "action": "hold"}}), encoding="utf-8")
    shutil.copyfile(model / "diagnosis.nat2.submission.json", tampered.with_suffix(".submission.json"))
    with pytest.raises(ValueError, match="다시 검증"):
        demo.review_nat_record(tampered, packet_path, "diagnosis")
    for path in (tampered, tampered.with_suffix(".submission.json")):
        path.unlink()
    out = demo.loop_start("n1", ROOT, model / "diagnosis.nat2.json")
    assert out["decision"]["claims"] == [{"id": fid, "value": packet["facts"][fid]}]  # 6자리 표기가 저장 값으로 치환됐다
    call = out["call"]
    assert call["path"] == "nat_v4" and call["record"].endswith("diagnosis.nat2.json") and call["precheck_calls"] == 1
    assert all(isinstance(call[k], (int, float)) for k in ("execution_wall_s", "input_tokens", "output_tokens"))
    assert out["started"]["model_probe_used"] is True and out["started"]["request_id"]
    assert [json.loads(p.read_text(encoding="utf-8"))["status"] for p in sorted(model.glob("diagnosis.attempt*.json"))] == [
        "infra_error", "valid"]
    steps = {name: demo.step_record("n1", name) for name in demo.STEPS if demo.step_path("n1", name).exists()}
    assert demo.conditions(steps, "in_progress")["fresh_diagnosis_call"] is True


def test_second_nat_diagnosis_infra_error_ends_the_demo(demo, monkeypatch, tmp_path):
    _, packet_path = ready_for_loop(demo, monkeypatch, tmp_path, "n3")
    model = packet_path.parent
    for n in (1, 2):
        nat_call(monkeypatch, "diagnosis", packet_path, model / f"diagnosis.nat{n}.json", runner=provider_timeout)
        out = demo.loop_start("n3", ROOT, model / f"diagnosis.nat{n}.json")
    assert out["outcome"] == "model_infra_failed"


def test_nat_correction_record_is_proposed_approved_and_run(demo, monkeypatch, tmp_path):
    ctx, _ = ready_for_correction(demo, monkeypatch, tmp_path, "n2", "f1")
    monkeypatch.setattr(demo, "verify_for", lambda t, isaaclab: (NAT, EXPERIMENT))
    model = demo.folder_for("n2") / "model"
    packet_path = model / "correction.packet.json"
    code, _ = nat_call(monkeypatch, "correction", packet_path, model / "correction.nat1.json", runner=provider_timeout)
    assert code == 2
    with pytest.raises(ValueError, match="인프라"):  # 인프라 오류 기록으로는 원장에 쓰지 않는다
        demo.correction_propose("n2", ROOT, model / "correction.nat1.json", budget_gpu_s=2400)
    assert demo.correction_status("n2")["requests"] == {}
    turns = [("e4b4_check_proposal", json.dumps(propose()), ""), ("e4b4_submit_proposal", json.dumps(propose()), ""),
             (None, None, "Submitted.")]
    code, record = nat_call(monkeypatch, "correction", packet_path, model / "correction.nat2.json", turns)
    assert code == 0 and record["decision"]["decision"] == "propose"
    tampered = model / "correction.nat3.json"
    for key, value in (("schema", "e4b_nat_decision_v3"), ("config_sha256_lf", "0" * 64),
                       ("tools_modules_sha256_lf", {"v4": "0" * 64}), ("system_prompt_sha256", "0" * 64)):
        tampered.write_text(json.dumps({**record, key: value}), encoding="utf-8")
        with pytest.raises(ValueError, match="NAT 기록"):  # 등록 설정·도구·프롬프트와 다른 기록은 받지 않는다
            demo.review_nat_record(tampered, packet_path, "correction", demo.registry_v3())
    tampered.unlink()
    proposed = demo.correction_propose("n2", ROOT, model / "correction.nat2.json", budget_gpu_s=2400)
    review = proposed["evidence"]["model_proposal"]
    assert review["schema"] == "e2e_nat_correction_review_v3_1" and review["accepted"] and review["agrees_with_rule"]
    assert review["precheck_calls"] == 1 and proposed["recovery_mode"] == "restart"
    rid = proposed["request_id"]
    demo.approve_correction("n2", rid, "operator")
    monkeypatch.setattr(DC, "make_executor", fake_executor([]))
    isaaclab = tmp_path / "isaaclab"
    (isaaclab / "logs/rsl_rl/unitree_go2_flat").mkdir(parents=True)
    with pytest.raises(ValueError, match="결합된"):  # 요청에 묶인 기록(nat2)이 아니면 실행하지 않는다
        demo.correction("n2", isaaclab, rid, model / "correction.nat1.json")
    step = demo.correction("n2", isaaclab, rid, model / "correction.nat2.json")
    assert step["call"]["path"] == "nat_v4" and step["model_proposal"]["accepted"] is True
    assert step["receipt"]["verdict"] == "recovered"
    steps = {name: demo.step_record("n2", name) for name in demo.STEPS}
    assert demo.conditions(steps, "recovered")["correction_call_accepted"] is True
    assert all(isinstance(step["call"][k], (int, float)) for k in ("execution_wall_s", "input_tokens", "output_tokens"))
    with pytest.raises(LedgerError):
        demo.run_correction(ctx, rid, fake_executor([])("n2", isaaclab), "n2_fix")


def test_nat_path_preconditions_for_the_next_demo(tmp_path, monkeypatch):
    adoption = E.nat_adoption("nat_v4")  # 공개 E4b v4 요약이 adopt여야 한다
    assert adoption["decision"] == "adopt" and adoption["summary"] == "evals/results/e4b_nat_compare_v4/summary.json"
    assert E.nat_adoption("single") is None
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"adoption": {"decision": "do_not_adopt"}}), encoding="utf-8")
    monkeypatch.setattr(E, "E4B_SUMMARY", summary)
    with pytest.raises(ValueError, match="adopt"):
        E.nat_adoption("nat_v4")
    monkeypatch.setattr(E, "E4B_SUMMARY", tmp_path / "missing.json")
    with pytest.raises(ValueError, match="E4b v4"):
        E.nat_adoption("nat_v4")
    fixed = E.fixed("f1", "nat_v4")
    assert fixed["decision_path"] == "nat_v4" and fixed["seed"] == 2034 and fixed["restart_seed"] == 3034
    assert all("nat_decision_call_v4.py" in text for text in (fixed["diagnosis"]["call"], fixed["correction"]["model_call"]))
    assert {"evals/nat_decision_call_v4.py", "src/rl_triage/nat_demo_tools_v4.py", "configs/nat_demo_v4_diagnosis.yml",
            "configs/nat_demo_v4_correction.yml"} <= E.CODE
