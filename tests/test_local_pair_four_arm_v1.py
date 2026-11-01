# pyright: reportMissingImports=false
from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("evals", "tests", "src")]
import local_pair_four_arm_v1 as M
import test_local_same_play_pair_provider_v1 as F


def source(tmp: Path) -> dict:
    pending, approval, spec, lab, state, protocol = F.fixture(tmp / "input")
    out = tmp / "provider"
    F.M.prepare(pending, approval, spec, lab, state, protocol, "a" * 40, out,
                allow_fixture=True, materializer=F.materializer,
                input_inspector=F.inspector(pending))
    return M.load_pair(out / "result.json")


def contract(tmp: Path) -> Path:
    path = tmp / "fixture-protocol.json"
    M.write(path, {"fixture": True, "max_probes_per_arm": 3})
    return path


def fake_probe(snapshot: dict, choice: dict, *, out: Path) -> dict:
    spec = M.C.request_spec(snapshot, choice)
    ledger = M.C.S.M.RegistryLedger(Path(snapshot["inputs"]["ledger_path"]), M.C.RC.registry())
    request = ledger.propose(spec["prereg"], spec["probe"], spec["budget_gpu_s"])
    ledger.approve(request["request_id"], "fixture_operator")
    ledger.consume(request["request_id"], spec["prereg"], spec["probe"])
    outcome = "abnormal" if spec["probe"] == "P_reward" else "normal"
    ledger.receipt(request["request_id"], outcome, {}, 1, 0)
    path = out / "result.json"
    M.write(path, {"probe": spec["probe"], "outcome": outcome,
                   "request_id": request["request_id"], "owned_child_wall_s": 0.0,
                   "evidence_kind": "cpu_fixture"})
    return {"result_ref": M.ref(path)}


def verify(path: Path, _snapshot: dict, _choice: dict) -> dict:
    return M.read(path)


def valid_model(request: dict) -> dict:
    content = request["messages"][1]["content"]
    text = content[0]["text"] if isinstance(content, list) else content
    view = json.loads(text)
    fact = next(iter(view["evidence_facts"]))
    return {"status": "completed", "content": json.dumps({"action": "run_probe",
            "target": "P_reward", "fact_ids": [fact]})}


def run(tmp: Path, data: dict, arm: str, complete=valid_model, execute=fake_probe) -> dict:
    return M.run_arm(data, arm=arm, run_id="fixture_" + arm, output=tmp / arm,
                     protocol_path=contract(tmp), execution_anchor="a" * 40,
                     complete=complete, execute_probe=execute, verify_probe=verify,
                     allow_fixture=True)


@pytest.mark.parametrize("arm", M.ARMS)
def test_four_modes_follow_real_controller_ledger_to_confirmation_or_bound(tmp_path: Path, arm: str):
    result = run(tmp_path, source(tmp_path), arm)
    assert result["scope"] == "cpu_fixture" and len(result["steps"]) <= 3
    assert result["model_calls"] == (1 if arm in M.MODEL_ARMS else 0)
    if arm in M.MODEL_ARMS:
        assert result["end_state"] == "confirmed" and result["confirmed_subtype"] == "reward"
    assert result["repair"] is False and result["recovery"] is False


def test_modes_share_numbers_and_roles_and_only_frames_differ(tmp_path: Path):
    data = source(tmp_path)
    packet = data["packet"]
    snap = M.C.V.build_snapshot(packet, "fixture_compare", "model_adaptive",
                               M.C.V.file_digest(M.C.V.PROTOCOL), tmp_path / "history.jsonl")
    only, only_msgs = M._context(data, snap, "logs_only", 3)
    plus, plus_msgs = M._context(data, snap, "logs_plus_frames", 3)
    assert M.P.PAIRED._context_common(only) == M.P.PAIRED._context_common(plus)
    assert len(only["model_view"]["evidence_facts"]) == 102
    assert len(plus["model_view"]["evidence_facts"]) == 108
    assert isinstance(only_msgs[1]["content"], str)
    assert sum(x["type"] == "image_url" for x in plus_msgs[1]["content"]) == 6


@pytest.mark.parametrize("content,reason", [
    ('{"action":"run_probe","target":"P_reward","fact_ids":["invented"]}', "model_contract_invalid"),
    ('{"action":"run_probe","target":null,"fact_ids":[]}', "model_contract_invalid"),
])
def test_semantic_failure_is_hold_without_rule_fallback_or_probe(tmp_path: Path, content: str, reason: str):
    result = run(tmp_path, source(tmp_path), "logs_only",
                 complete=lambda _: {"status": "completed", "content": content})
    assert result["hold_reason"] == reason and result["steps"] == []
    assert result["semantic_invalid"] == 1 and result["model_calls"] == 1


def test_infrastructure_unknown_does_not_retry_or_become_semantic_failure(tmp_path: Path):
    result = run(tmp_path, source(tmp_path), "logs_only",
                 complete=lambda _: {"status": "infra_error", "content": None})
    assert result["hold_reason"] == "model_transport_unknown" and result["steps"] == []
    assert result["infra_unknown"] == 1 and result["semantic_invalid"] == 0
    assert result["model_calls"] == 1


def test_wrong_provider_unknown_is_separate_from_network_failure(tmp_path: Path):
    result = run(tmp_path, source(tmp_path), "logs_only",
                 complete=lambda _: {"status": "unknown", "content": "{}"})
    assert result["provider_unknown"] == 1 and result["infra_unknown"] == 0
    assert result["hold_reason"] == "model_provider_unknown"


def test_transport_directory_reference_checks_receipt_digest_and_original_request(tmp_path: Path):
    call = tmp_path / "calls/0001"
    M.write(call / "receipt.json", {"fixture": True})
    request = {"request_id": "fixture"}
    response = {"status": "completed", "content": "{}", "usage": {}, "wall_s": 0.1,
                "call_ref": {"path": str(call), "sha256": M.sha(call / "receipt.json")}}
    checked = {**response, "request": request, "live": True}
    assert M._verify_transport(response, request, lambda path: checked) == response["call_ref"]
    with pytest.raises(ValueError, match="request"):
        M._verify_transport(response, {"different": True}, lambda path: checked)
    M.write(call / "unrelated.json", {})
    (call / "receipt.json").write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA"):
        M._verify_transport(response, request, lambda path: checked)


def test_self_reported_receipt_without_ledger_settlement_is_rejected(tmp_path: Path):
    def no_settlement(snapshot, choice, *, out):
        path = out / "result.json"
        M.write(path, {"probe": choice["probe"], "outcome": "abnormal"})
        return {"result_ref": M.ref(path)}
    with pytest.raises(ValueError, match="원장"):
        run(tmp_path, source(tmp_path), "logs_only", execute=no_settlement)


def test_model_can_cite_frame_but_it_is_not_a_confirmed_visual_cause(tmp_path: Path):
    def model(_request):
        return {"status": "completed", "content": json.dumps({"action": "run_probe",
                "target": "P_reward", "fact_ids": ["vision:frame:f1"]})}
    result = run(tmp_path, source(tmp_path), "logs_plus_frames", complete=model)
    assert result["frame_fact_citations"] == 1
    assert result["confirmed_subtype"] == "reward" and result["recovery"] is False


def test_public_packet_does_not_reintroduce_hidden_runtime_facts(tmp_path: Path):
    data = source(tmp_path)
    assert len(data["backend_packet"]["facts"]) == 64
    assert len(data["packet"]["facts"]) == 48
    changed = deepcopy(data)
    changed["packet"]["facts"]["runtime:case:available"] = True
    with pytest.raises(ValueError):
        run(tmp_path, changed, "registry_next")


def test_second_model_request_sees_the_first_verified_normal_receipt(tmp_path: Path):
    observed = []
    def model(request):
        text = request["messages"][1]["content"]
        view = json.loads(text)
        observed.append(deepcopy(view["observations"]))
        return {"status": "completed", "content": json.dumps({"action": "run_probe",
                "target": "P_episode" if not view["observations"] else "P_reward",
                "fact_ids": [next(iter(view["evidence_facts"]))]})}
    result = run(tmp_path, source(tmp_path), "logs_only", complete=model)
    assert observed == [[], [{"probe": "P_episode", "outcome": "normal"}]]
    assert [row["probe"] for row in result["steps"]] == ["P_episode", "P_reward"]
    assert result["model_calls"] == 2 and result["confirmed_subtype"] == "reward"
