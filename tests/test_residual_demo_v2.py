"""새 선택 경로와 기존 승인·교정 receipt를 함께 CPU로 검증한다."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "bench"), str(ROOT / "tests")]
import residual_demo_v2 as D
import test_residual_demo_v1 as BASE


def caller(order="reward_first", *, invalid=False, invalid_second=False, uninformative=False,
           hold=False, infra=False, correction_invalid=False):
    calls = []
    correction = BASE.model_caller(invalid=correction_invalid)

    def complete(messages, params):
        calls.append(messages)
        bodies = []
        for row in messages:
            try:
                bodies.append(json.loads(row["content"]))
            except (ValueError, TypeError):
                pass
        body = next((row for row in bodies if isinstance(row, dict) and "candidate_groups" in row), None)
        if body is None:
            return correction(messages, params)
        if infra:
            return {"infra_error": "offline_fixture", "content": None, "wall_s": .1}
        first_reward = order == "reward_first"
        probe = ("P_reward" if first_reward else D.C.NEW_PROBE) if body["epoch"] == 1 else (
                 D.C.NEW_PROBE if first_reward else "P_reward")
        if uninformative and body["epoch"] == 2:
            probe = "P_reward" if first_reward else D.C.NEW_PROBE
        value = {"action": "hold" if hold else "continue", "probe": None if hold else probe,
                 "fact_ids": [next(iter(body["evidence_facts"]))]}
        if invalid or invalid_second and body["epoch"] == 2:
            value = {"ranking": list(D.C.registry()["mechanisms"]), "expected_outcomes": {}}
        return {"response_id": f"cpu_{len(calls)}", "provider_model": D.C.MODEL,
                "finish_reason": "stop", "content": json.dumps(value), "reasoning_content": None,
                "input_tokens": 20, "output_tokens": 10, "total_tokens": 30, "wall_s": .1}

    setattr(complete, "calls", calls)
    return complete


@pytest.fixture
def setup(tmp_path, monkeypatch):
    protocol = {**D.fixed_protocol(), "source_sha256_lf": {}, "runtime_source_sha256_lf": {}}
    monkeypatch.setattr(D, "PRIVATE", tmp_path / "runs")
    original = D.sha
    monkeypatch.setattr(D, "sha", lambda path, lf=False: "b" * 64 if Path(path) == D.PROTOCOL else original(path, lf))
    monkeypatch.setattr(D, "verify", lambda *args, **kwargs: (protocol, "c" * 40))
    return protocol


def execute(backend, model):
    return D.execute("residual_v2_20261017", Path("unused"), "operator", True, backend=backend, caller=model,
                     seed_guard=lambda *args, **kwargs: {"status": "fresh", "seeds": list(D.SEEDS)})


@pytest.mark.parametrize("order", ["reward_first", "optimizer_first"])
def test_valid_models_keep_both_orders_and_approved_receipts(setup, order):
    result = execute(BASE.Backend(), caller(order))
    assert result["status"] == "complete" and result["demo_success"]
    assert result["api"]["reserved"] == 8
    assert result["decision_metrics"]["model_validity"]["rate"] == 1
    assert result["decision_metrics"]["rule_transition"]["rate"] == 0
    assert result["end_to_end_recovery"] == {"numerator": 2, "denominator": 2, "rate": 1}
    for case in result["cases"]:
        assert [e["selection"]["selected_by"] for e in case["epochs"]] == ["model", "model"]
        assert [e["verified_receipt"]["verdict"] for e in case["epochs"]] == ["not_recovered", "recovered"]


@pytest.mark.parametrize("mode", ["invalid", "invalid_second", "uninformative"])
def test_invalid_models_switch_only_probe_selection_to_registered_rules(setup, mode):
    result = execute(BASE.Backend(), caller(invalid=mode == "invalid",
        invalid_second=mode == "invalid_second", uninformative=mode == "uninformative"))
    assert result["demo_success"] and result["api"]["reserved"] <= 20
    expected = 0 if mode == "invalid" else .5
    assert result["decision_metrics"]["model_validity"]["rate"] == expected
    assert result["decision_metrics"]["rule_transition"]["rate"] == 1 - expected
    assert result["decision_metrics"]["rule_transition_given_invalid"]["rate"] == 1
    for case in result["cases"]:
        epoch = case["epochs"][1]
        assert epoch["selection"]["selected_by"] == "registered_rule"
        assert epoch["selection"]["host_decision"]["claims"] == []
        assert epoch["probe_steps"][0]["probe"] == "P_value"
        expected_probe = "P_reward" if epoch["confirmed_component"] == "reward" else D.C.NEW_PROBE
        assert epoch["probe_steps"][-1]["probe"] == expected_probe
        assert epoch["probe_steps"][-1]["outcome"] == "abnormal"
        ledger = D.CorrectionLedger(D.PRIVATE / "residual_v2_20261017" / f"s{case['seed']}" /
                                    "correction.ledger.jsonl", D.C.correction_registry())
        assert len(ledger.requests()) == 2
        for request_id in ledger.requests():
            assert [e["event"] for e in ledger.events() if e.get("request_id") == request_id] == [
                "proposed", "approved", "consumed", "receipt"]


@pytest.mark.parametrize("faults", [(), ("F1",), ("R1",)])
def test_none_and_single_fault_rule_replay_have_no_unnecessary_second_epoch(setup, faults):
    folder = D.PRIVATE / "single" / "s2120"
    result = D.run_case(folder, "residual_v2_20261017", 2120, Path("unused"), setup, "c" * 40,
                        "operator", folder.parent / "api", backend=BASE.Backend(),
                        caller=caller(invalid=True), original_faults=faults)
    assert result["outcome"] == ("recovered_after_one" if faults else "source_failure_absent")
    assert result["second_epoch_requests"] == 0
    assert not (folder / "epoch2").exists()
    if not faults:
        assert D.MC.audit_budget(folder.parent / "api")["reserved"] == 0


@pytest.mark.parametrize("mode", ["hold", "infra", "unknown_probe", "correction_invalid"])
def test_uncertainty_and_correction_model_failure_do_not_authorize_repair(setup, mode):
    backend = BASE.Backend(unknown_probe=mode == "unknown_probe")
    model = caller(invalid=mode == "unknown_probe", hold=mode == "hold", infra=mode == "infra",
                   correction_invalid=mode == "correction_invalid")
    result = execute(backend, model)
    assert all(case["outcome"] == "HOLD" for case in result["cases"])
    assert not any(row[0] == "train" and row[1] in {"FIRST", "BOTH"} for row in backend.calls)
    if mode in {"hold", "infra"}:
        assert result["decision_metrics"]["rule_transition"]["numerator"] == 0


@pytest.mark.parametrize("target", ["selection", "checkpoint", "api"])
def test_tampering_after_first_receipt_blocks_epoch2_and_no_fallback(setup, monkeypatch, target):
    original = D.verify_repair
    calls = 0
    def mutate_after_verification(path, *args, **kwargs):
        nonlocal calls
        result = original(path, *args, **kwargs)
        calls += 1
        if calls == 1:
            bundle = D.read(path)
            changed = (path.parent / "model/selection.json" if target == "selection" else
                       Path(bundle["artifact"]["checkpoint"]["path"]) if target == "checkpoint" else
                       Path(bundle["api_root"]) / "responses/0001.json")
            changed.write_bytes(changed.read_bytes() + b" ")
        return result
    monkeypatch.setattr(D, "verify_repair", mutate_after_verification)
    with pytest.raises(ValueError):
        execute(BASE.Backend(), caller(invalid_second=True))
    folder = D.PRIVATE / "residual_v2_20261017"
    assert D.read(folder / "result.json")["status"] == "invalid"
    assert not (folder / "s2120/epoch2").exists()


def test_selection_replay_rejects_forged_origin_even_with_updated_inventory(setup, monkeypatch):
    original = D.seal_repair
    def forged(path, **kwargs):
        value = original(path, **kwargs)
        selected = path.parent / "model/selection.json"
        data = D.read(selected)
        data["selection"]["selected_by"] = "registered_rule"
        selected.write_text(json.dumps(data), encoding="utf-8")
        value["inventory_sha256"][str(selected.resolve())] = D.sha(selected)
        path.write_text(json.dumps(value), encoding="utf-8")
        return value
    monkeypatch.setattr(D, "seal_repair", forged)
    with pytest.raises(ValueError, match="선택 주체"):
        execute(BASE.Backend(), caller())
    assert not (D.PRIVATE / "residual_v2_20261017/s2120/epoch2").exists()


def test_corrupted_packet_is_not_treated_as_model_failure(setup, monkeypatch):
    original = D.diagnosis_packet
    def forged(*args):
        packet = original(*args)
        packet["remaining_candidates"] = ["reward"]
        return packet
    monkeypatch.setattr(D, "diagnosis_packet", forged)
    model, backend = caller(), BASE.Backend()
    with pytest.raises(ValueError):
        execute(backend, model)
    assert getattr(model, "calls") == []
    assert not any(row[0] == "train" and row[1] == "FIRST" for row in backend.calls)


@pytest.mark.parametrize("target", ["source_checkpoint", "source_telemetry", "reference_telemetry"])
def test_sealed_diagnosis_artifact_cannot_substitute_another_source(setup, monkeypatch, target):
    original = D.seal_repair
    def forged(path, **kwargs):
        value = deepcopy(original(path, **kwargs))
        field = "checkpoint" if target == "source_checkpoint" else "telemetry"
        owner = "reference_artifact" if target.startswith("reference") else "source"
        value[owner][field]["sha256"] = "0" * 64
        # 봉인 파일의 SHA는 이후 정상 correction receipt가 새로 계산한다.
        path.write_text(json.dumps(value), encoding="utf-8")
        return value
    monkeypatch.setattr(D, "seal_repair", forged)
    with pytest.raises(ValueError, match="artifact"):
        execute(BASE.Backend(), caller())
    assert not (D.PRIVATE / "residual_v2_20261017/s2120/epoch2").exists()


def test_model_input_binds_checkpoint_and_telemetry_but_keeps_input_bindings_out_of_prompt(setup):
    execute(BASE.Backend(), caller())
    folder = D.PRIVATE / "residual_v2_20261017/s2120"
    first = D.read(folder / "epoch1/repair_receipt.json")
    second = D.read(folder / "epoch2/model/diagnosis.packet.json")
    assert second["input_bindings"]["case"] == D.input_binding(first["artifact"])
    assert second["input_bindings"]["reference"] == D.input_binding(D.read(folder / "artifact_none.json"))
    assert second["prior_receipt"]["checkpoint_sha256"] == second["input_bindings"]["case"]["checkpoint_sha256"]
    body = json.loads(D.SELECT.diagnosis_messages(second)[1]["content"])
    assert "input_bindings" not in body


def test_parent_binding_is_reconstructed_from_external_expectation(setup, monkeypatch):
    execute(BASE.Backend(), caller())
    folder = D.PRIVATE / "residual_v2_20261017/s2120"
    path = folder / "epoch2/repair_receipt.json"
    child = D.read(path)
    first = D.read(folder / "epoch1/repair_receipt.json")
    initial = D.read(folder / "artifact_compound.json")
    seen = []
    actual = D.D1.verify_repair
    def capture(path, ledger, **kwargs):
        seen.append((path.parent.name, deepcopy(kwargs["expected_binding"]), kwargs["expected_input_sha"]))
        return actual(path, ledger, **kwargs)
    monkeypatch.setattr(D.D1, "verify_repair", capture)
    ledger = D.CorrectionLedger(folder / "correction.ledger.jsonl", D.C.correction_registry())
    D.verify_repair(path, ledger, expected_epoch=2, expected_applied=child["applied_before"],
        expected_input_sha=first["artifact"]["checkpoint"]["sha256"], previous=D.sha(folder / "epoch1/repair_receipt.json"),
        reference=child["reference"], bands=D.read(D.ROOT / D.BANDS)["bands"], expected_binding=child["binding"],
        seed=2120, backend=BASE.Backend())
    parent = next(item for item in seen if item[0] == "epoch1")
    assert parent[1] == {**child["binding"], "applied_corrections": child["applied_before"],
                        "source_checkpoint_sha256": initial["checkpoint"]["sha256"]}
    assert parent[2] == initial["checkpoint"]["sha256"]


def test_no_approval_no_gpu_backend_invocation(setup):
    backend = BASE.Backend()
    with pytest.raises(ValueError):
        D.execute("residual_v2_20261017", Path("unused"), "operator", False, backend=backend, caller=caller())
    assert backend.calls == []


def test_existing_tag_is_never_reused(setup):
    backend = BASE.Backend(source_healthy=True)
    execute(backend, caller())
    previous = deepcopy(backend.calls)
    with pytest.raises(FileExistsError):
        execute(backend, caller())
    assert backend.calls == previous
