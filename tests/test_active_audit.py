import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from rl_triage import active_diagnosis as D
from rl_triage import probe_loop as L


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("active_audit", ROOT / "evals/audit_active_diagnosis.py")
assert SPEC is not None and SPEC.loader is not None
A = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(A)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def sha(path, raw=False):
    data = path.read_bytes()
    return hashlib.sha256(data if raw else data.replace(b"\r\n", b"\n")).hexdigest()


def rank(first):
    return [first] + [item for item in L.MECHANISMS if item != first]


def expected(probe, ranking):
    return {item: L.PROBES[probe]["expect"][item] for item in ranking[:3]}


def model_files(folder, stage, packet, decision, model):
    packet_path = write(folder / "model" / f"{stage}.packet.json", packet)
    decision_path = write(folder / "model" / f"{stage}.decision.json", decision)
    record = {
        "schema": "active_model_call_v1",
        "packet_sha256": sha(packet_path, raw=True),
        "messages_sha256": L.digest(D.decision_messages(packet)),
        "request": {"model": model, "temperature": 0, "max_tokens": 16384},
        "saved_response_cached": False,
        "client_cache_used": False,
        "response_id": f"response-{stage}",
        "finish_reason": "stop",
        "reasoning_content": None,
        "content": json.dumps(decision),
        "input_tokens": 100,
        "output_tokens": 20,
        "total_tokens": 120,
        "execution_wall_s": 1.5,
        "cpu_s": 0.1,
        "exit_code": 0,
    }
    attempt = write(folder / "model" / f"{stage}.attempt1.json", record)
    return packet_path, decision_path, attempt, record


def fixture(tmp_path):
    tag, case, model = "audit_test", "case0", "provider/model"
    source_dir = tmp_path / "inputs"
    telemetry = write(source_dir / "telemetry.json", {"public": 1})
    reference_telemetry = write(source_dir / "reference.json", {"public": 2})
    probe_reference = write(source_dir / "probe_reference.json", {
        "probes": {"P_reward": {"measurement": {
            "track_lin_vel_xy_exp_rel_error": 0.0,
            "track_ang_vel_z_exp_rel_error": 0.0,
        }}}
    })
    def source(path):
        return {"path": path.relative_to(tmp_path).as_posix(), "sha256_lf": sha(path)}

    checkpoint = "c" * 64
    cases = [{
        "case": case,
        "checkpoint_sha256": checkpoint,
        "sources": {
            "telemetry": source(telemetry),
            "reference_telemetry": source(reference_telemetry),
            "probe_reference": source(probe_reference),
        },
    }]
    for index in range(1, 8):
        cases.append({"case": f"missing{index}", "checkpoint_sha256": checkpoint,
                      "sources": cases[0]["sources"]})
    protocol = {
        "schema": "active_diagnosis_protocol_v1",
        "model": model,
        "approver": "operator-unverified",
        "max_probes": 2,
        "probe_execution_budget_s": 120,
        "cases": cases,
    }
    from evals.run_active_diagnosis import CODE
    code_hashes = {}
    for relative in CODE:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())
        code_hashes[relative] = sha(target)
    protocol["code_sha256_lf"] = code_hashes
    protocol_path = write(tmp_path / "protocol.json", protocol)
    bound = {**protocol, "_audit_sha256_lf": sha(protocol_path)}

    folder = tmp_path / "evals" / "results" / tag / "cases" / case
    facts = {"telemetry:return": -1.0}
    initial_packet = {
        "phase": "initial", "facts": facts, "observed": [], "budget_left": 1,
        "execution_budget_s": 120,
    }
    before_ranking = rank("actuator")
    before = {
        "ranking": before_ranking, "probe": "P_reward", "action": "continue",
        "reason": "공개 텔레메트리에서 reward 검사를 선택했다.",
        "claims": [{"id": "telemetry:return", "value": -1.0}],
        "expected_outcomes": expected("P_reward", before_ranking),
    }
    initial_files = model_files(folder, "initial", initial_packet, before, model)

    binding = {
        "controller": "model",
        "packet_sha256_lf": sha(initial_files[0]),
        "model_call_sha256_lf": sha(initial_files[2]),
        "decision_sha256_lf": sha(initial_files[1]),
        "protocol_sha256_lf": bound["_audit_sha256_lf"],
    }
    envelope = {
        "case": case, "checkpoint_sha256": checkpoint,
        "protocol_sha256_lf": bound["_audit_sha256_lf"], "stage": "1",
        "reference_source": cases[0]["sources"]["probe_reference"],
        "prior_receipt_digests": [], "binding": binding, "budget_probes": 2,
        "probe": "P_reward", "args": L.PROBES["P_reward"]["args"],
        "thresholds": L.THRESHOLDS,
    }
    prereg_digest = L.digest(envelope)
    rid = L.digest({"prereg": prereg_digest, "probe": "P_reward",
                    "args": L.PROBES["P_reward"]["args"], "n": 0})[:12]

    canonical_measurement = {
        "track_lin_vel_xy_exp_rel_error": 0.4,
        "track_ang_vel_z_exp_rel_error": 0.0,
    }
    provisional = {
        "request_id": "placeholder", "probe": "P_reward", "outcome": "abnormal",
        "measurement": canonical_measurement, "gpu_s": 10.0, "exit_code": 0,
        "over_budget": False,
    }
    after_ranking = rank("reward")
    after = {
        "ranking": after_ranking, "probe": None, "action": "confirm",
        "reason": "실측 reward 오차가 frozen 확정 규칙을 만족했다.",
        "claims": [{"id": "receipt:RID:track_lin_vel_xy_exp_rel_error", "value": 0.4}],
        "expected_outcomes": {},
    }
    shadow = {
        "ranking": before_ranking, "probe": None, "action": "hold",
        "reason": "receipt를 가린 조건에서는 보류한다.",
        "claims": [{"id": "telemetry:return", "value": -1.0}],
        "expected_outcomes": {},
    }

    provisional["request_id"] = rid
    after["claims"][0]["id"] = f"receipt:{rid}:track_lin_vel_xy_exp_rel_error"
    observed_for_packet = dict(provisional)
    after_packet = {
        "phase": "after_receipt",
        "facts": {**facts, **D.receipt_facts(observed_for_packet)},
        "observed": [observed_for_packet],
        "budget_left": 0,
        "execution_budget_s": 120,
        "previous_decision": before,
    }
    masked_packet = {**after_packet, "facts": facts, "observed": []}
    after_files = model_files(folder, "after_1", after_packet, after, model)
    masked_files = model_files(folder, "without_receipt_1", masked_packet, shadow, model)

    ledger = L.Ledger(folder / "ledger.jsonl")
    actual_rid = ledger.propose(envelope, "P_reward", 120)["request_id"]
    assert actual_rid == rid
    write(folder / "approvals" / f"{rid}.json", envelope)
    ledger.approve(rid, protocol["approver"])
    ledger.consume(rid, envelope, "P_reward")
    canonical = ledger.receipt(rid, "abnormal", canonical_measurement, 10.0, 0)

    base = tmp_path / "evals" / "results" / tag
    report = {
        "checkpoint": {"sha256": checkpoint},
        "probes": {"P_reward": {"measurement": canonical_measurement, "exit": 0}},
    }
    report_path = write(base / "probes" / f"{case}__P_reward.json", report)
    report_sha = sha(report_path)
    execution = {
        "case": case, "probe": "P_reward", "checkpoint_sha256": checkpoint,
        "report_sha256_lf": report_sha, "gpu_wall_s": 10.0,
        "execution_wall_s": 10.0, "exit_code": 0, "cpu_s": 2.0,
        "simulator_steps": 512,
    }
    resources = {
        "case": case, "probe": "P_reward", "checkpoint_sha256": checkpoint,
        "report_sha256_lf": report_sha, "cpu_s": 2.0, "simulator_steps": 512,
    }
    write(base / "execution" / f"{case}__P_reward.json", execution)
    write(base / "resources" / f"{case}__P_reward.json", resources)
    augmented = {
        **canonical, "receipt_digest": L.digest(canonical), "probe": "P_reward",
        "checkpoint_sha256": checkpoint, "report_sha256_lf": report_sha,
    }
    contrast = D.contrast(before, after, shadow, [augmented])
    records = [initial_files[3], after_files[3], masked_files[3]]
    result = {
        "schema": "active_diagnosis_case_v1", "case": case, "tag": tag,
        "status": "completed", "checkpoint_sha256": checkpoint,
        "controller": "model", "protocol_sha256_lf": bound["_audit_sha256_lf"],
        "approval_actor": protocol["approver"], "approval_identity_attested": False,
        "human_review_s": None, "gpu_device_busy_s": None,
        "decisions": [before, after], "observed": [augmented],
        "contrasts": [contrast], "without_receipt": [shadow],
        "model_action": "confirm",
        "evidence_state": D.evidence_state([augmented], 1),
        "costs": {
            "model_calls": 3, "input_tokens": 300, "output_tokens": 60,
            "model_wall_s": 4.5, "model_cpu_s": 0.3, "isaac_wall_s": 10.0,
            "isaac_cpu_s": 2.0, "isaac_simulator_steps": 512,
            "human_review_s": None, "gpu_device_busy_s": None,
        },
    }
    write(folder / "result.json", result)
    return {"root": tmp_path, "tag": tag, "case": case, "protocol": bound,
            "protocol_path": protocol_path, "folder": folder, "result": result,
            "rid": rid, "report": report_path}


def test_complete_public_chain_verifies_changed_judgment(tmp_path):
    case = fixture(tmp_path)
    audit = A.audit_case(case["root"], case["tag"], case["case"], case["protocol"])
    assert audit == {
        "case": "case0", "valid": True, "errors": [], "proof_changed": True,
        "verified_stage_indices": [0, 1], "result_sha256_lf": sha(case["folder"] / "result.json"),
    }


def test_execution_cpu_includes_outer_wrapper_and_resource_cpu(tmp_path):
    case = fixture(tmp_path)
    path = case["folder"].parents[1] / "execution" / f"{case['case']}__P_reward.json"
    execution = json.loads(path.read_text())
    execution.update(parent_cpu_s=0.25, cpu_s=2.25)
    write(path, execution)
    result = case["result"]
    result["costs"]["isaac_cpu_s"] = 2.25
    write(case["folder"] / "result.json", result)
    audit = A.audit_case(case["root"], case["tag"], case["case"], case["protocol"])
    assert audit["valid"] and audit["proof_changed"]


def test_auditor_source_change_cannot_produce_verified_evidence(tmp_path):
    case = fixture(tmp_path)
    target = case["root"] / "evals/audit_active_diagnosis.py"
    target.write_bytes(target.read_bytes() + b"\n# modified after registration\n")
    audit = A.audit_case(case["root"], case["tag"], case["case"], case["protocol"])
    assert not audit["valid"] and not audit["proof_changed"]
    assert any("코드 변경" in error for error in audit["errors"])


@pytest.mark.parametrize("tamper", ["probe", "receipt_digest", "checkpoint", "report_sha"])
def test_binding_and_artifact_tampering_fails_closed(tmp_path, tamper):
    case = fixture(tmp_path)
    result_path = case["folder"] / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if tamper == "probe":
        result["observed"][0]["probe"] = "P_value"
    elif tamper == "receipt_digest":
        result["observed"][0]["receipt_digest"] = "0" * 64
    elif tamper == "checkpoint":
        result["observed"][0]["checkpoint_sha256"] = "0" * 64
    else:
        result["observed"][0]["report_sha256_lf"] = "0" * 64
    write(result_path, result)
    audit = A.audit_case(case["root"], case["tag"], case["case"], case["protocol"])
    assert not audit["valid"] and not audit["proof_changed"] and audit["errors"]


def test_missing_case_is_preserved_as_hold_and_cli_output_is_exclusive(tmp_path, monkeypatch):
    case = fixture(tmp_path)
    missing = A.audit_case(case["root"], case["tag"], "missing1", case["protocol"])
    assert not missing["valid"] and missing["errors"]
    monkeypatch.setattr(A, "ROOT", case["root"])
    out = tmp_path / "audit.json"
    assert A.main(["--tag", case["tag"], "--protocol", str(case["protocol_path"]), "--out", str(out)]) == 3
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert [row["case"] for row in saved["cases"]] == [item["case"] for item in case["protocol"]["cases"]]
    assert A.main(["--tag", case["tag"], "--protocol", str(case["protocol_path"]), "--out", str(out)]) == 2


def test_masked_packet_or_model_prompt_hash_tampering_is_detected(tmp_path):
    case = fixture(tmp_path)
    masked = case["folder"] / "model" / "without_receipt_1.packet.json"
    value = json.loads(masked.read_text(encoding="utf-8"))
    value["previous_decision"] = None
    write(masked, value)
    record_path = case["folder"] / "model" / "without_receipt_1.attempt1.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["packet_sha256"] = sha(masked, raw=True)
    record["messages_sha256"] = L.digest(D.decision_messages(value))
    write(record_path, record)
    audit = A.audit_case(case["root"], case["tag"], case["case"], case["protocol"])
    assert not audit["valid"]
    assert any("masked packet" in error for error in audit["errors"])


def test_duplicate_ledger_transition_breaks_exactly_once_chain(tmp_path):
    case = fixture(tmp_path)
    ledger_path = case["folder"] / "ledger.jsonl"
    rows = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines()]
    rows.insert(2, dict(rows[1]))
    ledger_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    audit = A.audit_case(case["root"], case["tag"], case["case"], case["protocol"])
    assert not audit["valid"]
    assert any("exactly-once" in error for error in audit["errors"])


def test_malformed_optional_objects_fail_closed_without_type_crash(tmp_path):
    case = fixture(tmp_path)
    result_path = case["folder"] / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["observed"] = None
    result["decisions"] = {"ranking": []}
    write(result_path, result)
    audit = A.audit_case(case["root"], case["tag"], case["case"], case["protocol"])
    assert not audit["valid"] and not audit["proof_changed"]
    assert any("객체 목록" in error for error in audit["errors"])
