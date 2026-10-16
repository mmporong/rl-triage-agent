from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "evals"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import nemotron_upgrade_compare_v1 as C  # noqa: E402
REAL_LOAD_PLAN = C.load_plan


def diagnosis_verdict(label: str = "reward") -> dict:
    return {"status": "valid", "decision": {"action": "continue", "ranking": [label], "probe": "P_reward"}}


def correction_verdict(label: str = "restore") -> dict:
    return {"status": "valid", "decision": {"decision": "propose", "proposal": {
        "correction_id": label, "recovery_mode": "restart"}, "agrees_with_rule": True}}


def test_protocol_contract_has_66_alternating_single_calls_and_80_total_cap():
    protocol = C.read(C.PROTOCOL)
    assert protocol["schema"] == "nemotron_upgrade_compare_protocol_v1"
    assert protocol["base_calls"] == 66 < protocol["api_cap"] == 80
    assert len(protocol["inputs"]) == 11
    assert {row["kind"] for row in protocol["inputs"]} == {"diagnosis", "correction"}
    assert protocol["registered_tag"] == C.REGISTERED_TAG
    assert len(protocol["call_order_sha256"]) == 64
    assert "bench/private" not in json.dumps(protocol["inputs"])


def test_private_input_binding_when_registered_packets_are_available():
    first = C.N2.packet_path(C.N2.SET_ORDER[0], C.N2.INPUT_SETS[C.N2.SET_ORDER[0]]["kinds"][0])
    if not first.is_file():
        pytest.skip("비공개 N2 packet이 없는 공개 checkout")
    protocol = C.verify_protocol()
    order = C.call_order()
    assert len(order) == protocol["base_calls"] == 66
    assert all(sum(row["packet_id"] == packet["packet_id"] and row["model"] == model for row in order) == 3
               for packet in protocol["inputs"] for model in C.MODELS)
    for run in range(1, 4):
        block = [row for row in order if row["run"] == run]
        for packet_index in range(11):
            pair = [row["model"] for row in block if row["packet_index"] == packet_index]
            assert pair[0] == C.MODELS[(run + packet_index) % 2]


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    packets = []
    packet_paths = {}
    for index in range(11):
        kind = "diagnosis" if index < 6 else "correction"
        set_name = f"set{index:02d}"
        path = tmp_path / "packets" / f"{set_name}_{kind}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
        packet_paths[(set_name, kind)] = path
        packets.append({"packet_id": f"{set_name}_{kind}", "set_name": set_name, "kind": kind,
                        "raw_sha256": C.sha_raw(path), "messages_sha256": f"{index:064x}"})
    order = []
    for run in range(1, 4):
        for packet_index, packet in enumerate(packets):
            first = (run + packet_index) % 2
            for model in (C.MODELS[first], C.MODELS[1 - first]):
                order.append({"call": len(order) + 1, "run": run, "packet_index": packet_index,
                              "model": model, **packet})
    monkeypatch.setattr(C, "packet_rows", lambda: packets)
    monkeypatch.setattr(C, "call_order", lambda: order)
    monkeypatch.setattr(C.N2, "packet_path", lambda set_name, kind: packet_paths[(set_name, kind)])
    monkeypatch.setattr(C, "PRIVATE_BASE", tmp_path / "runner")
    monkeypatch.setattr(C, "load_plan", lambda tag: {"experiment_commit": "a" * 40})
    monkeypatch.setattr(C, "verify_inputs", lambda: None)
    monkeypatch.setattr(C, "verify_protocol", lambda protocol=None: {"inputs": C.packet_rows()})
    monkeypatch.setattr(C.F, "validate", lambda kind, content, packet: json.loads(content))
    plan = C.plan_path(C.REGISTERED_TAG)
    C.write_new(plan, {"schema": "nemotron_upgrade_compare_plan_v1", "tag": C.REGISTERED_TAG,
                       "experiment_commit": "a" * 40, "head_at_plan": "b" * 40,
                       "protocol_sha256_lf": C.sha_lf(C.PROTOCOL),
                       "call_order_sha256": C.canonical_sha(order), "created_at": "test"})
    return C.REGISTERED_TAG


def _response(entry: dict, status: str, verdict=None, *, tokens=(2, 3, 5)) -> dict:
    if status == "infra_error":
        content, error_type, status_code, infra_error = None, "APITimeoutError", None, "APITimeoutError"
    elif status == "call_error":
        content, error_type, status_code, infra_error = None, "BadRequestError", 400, None
    else:
        content, error_type, status_code, infra_error = json.dumps(verdict), None, None, None
    return {"schema": "nemotron_upgrade_compare_response_v1", "entry": entry, "attempt": 1,
            "reservation_sha256": "pending", "status": status, "verdict": verdict,
            "response": {"response_id": "test", "provider_model": entry["model"], "finish_reason": "stop",
                         "content": content, "reasoning_content": None,
                         "input_tokens": tokens[0], "output_tokens": tokens[1], "total_tokens": tokens[2],
                         "wall_s": 1.0, "error_type": error_type, "status_code": status_code,
                         "infra_error": infra_error},
            "measured_wall_s": 1.0}


def write_attempt(tag: str, entry: dict, status: str, verdict=None, attempt: int = 1, *, tokens=(2, 3, 5)):
    reservation = C.reservation_path(tag, entry, attempt)
    packet = next(row for row in C.packet_rows() if row["packet_id"] == entry["packet_id"])
    C.write_new(reservation, {"schema": "nemotron_upgrade_compare_reservation_v1", "tag": tag,
                              "experiment_commit": "a" * 40, "entry": entry, "attempt": attempt,
                              "packet_sha256": packet["raw_sha256"], "messages_sha256": packet["messages_sha256"],
                              "request": C.F.request_params(entry["model"]), "reserved_at": "test"})
    row = _response(entry, status, verdict, tokens=tokens)
    row["attempt"] = attempt
    row["reservation_sha256"] = C.sha_raw(reservation)
    C.write_new(C.response_path(tag, entry, attempt), row)


def test_reservation_orphan_is_fail_closed_and_not_in_remaining_commands(isolated):
    entry = C.call_order()[0]
    C.write_new(C.reservation_path(isolated, entry, 1), {"entry": entry})
    state = C.remaining(isolated)
    assert state["status"] == "orphaned_reservations"
    assert entry["call"] in state["orphaned_calls"]
    assert not any(f"--call {entry['call']} " in command for command in state["commands"])


def test_api_cap_counts_write_once_reservations_before_any_remote_call(isolated, monkeypatch):
    entry = C.call_order()[0]
    for index in range(C.API_CAP):
        C.write_new(C.run_dir(isolated) / "reservations" / f"prior_{index:02d}.json", {"index": index})
    called = False

    def complete(_messages, _params):
        nonlocal called
        called = True
        return {}

    with pytest.raises(ValueError, match="상한|실행할 수 없는"):
        C.execute_call(isolated, entry["call"], 1, complete)
    assert called is False and C.used_attempts(isolated) == C.API_CAP


def test_execute_reserves_before_call_and_only_infra_gets_one_retry(isolated, monkeypatch):
    monkeypatch.setattr(C.F, "first_messages", lambda kind, data: (
        [{"role": "user", "content": "x"}], {"messages_sha256": "1" * 64}))
    monkeypatch.setattr(C.F, "validate", lambda kind, content, packet: diagnosis_verdict())
    entry = C.call_order()[0]
    packet = C.N2.packet_path(entry["set_name"], entry["kind"])
    seen = []

    def infra(messages, params):
        seen.append(C.reservation_path(isolated, entry, 1).exists())
        return {"provider_model": entry["model"], "content": "", "finish_reason": "stop",
                "infra_error": "APITimeoutError", "wall_s": 1.0}

    assert C.execute_call(isolated, entry["call"], 1, infra)["status"] == "infra_error"
    assert seen == [True]
    state = C.remaining(isolated)
    assert any(f"--call {entry['call']} --attempt 2" in command for command in state["commands"])

    def valid(_messages, _params):
        return {"provider_model": entry["model"], "content": "{}", "finish_reason": "stop", "wall_s": 1.0}

    assert C.execute_call(isolated, entry["call"], 2, valid)["status"] == "valid"
    assert C.final_response(isolated, entry)["attempt"] == 2
    assert packet.is_file()


def test_invalid_is_not_retried_and_call_error_stops_following_calls(isolated):
    first, second = C.call_order()[:2]
    write_attempt(isolated, first, "invalid", {"status": "invalid"})
    state = C.remaining(isolated)
    assert not any(f"--call {first['call']} " in command for command in state["commands"])
    write_attempt(isolated, second, "call_error", None)
    state = C.remaining(isolated)
    assert state["status"] == "stopped_inconclusive" and state["remaining"] == 0


def test_http_400_empty_body_is_call_error_without_retry(isolated, monkeypatch):
    monkeypatch.setattr(C.F, "first_messages", lambda kind, data: (
        [{"role": "user", "content": "x"}], {"messages_sha256": "1" * 64}))
    entry = C.call_order()[0]
    result = C.execute_call(isolated, entry["call"], 1, lambda messages, params: {
        "error_type": "BadRequestError", "status_code": 400, "content": None, "wall_s": 0.1})
    assert result["status"] == "call_error"
    assert C.remaining(isolated)["status"] == "stopped_inconclusive"


def test_second_attempt_call_error_stops_and_fixed_order_rejects_later_call(isolated, monkeypatch):
    monkeypatch.setattr(C.F, "first_messages", lambda kind, data: (
        [{"role": "user", "content": "x"}], {"messages_sha256": "1" * 64}))
    first, later = C.call_order()[0], C.call_order()[2]
    write_attempt(isolated, first, "infra_error", None)
    write_attempt(isolated, first, "call_error", None, attempt=2)
    assert C.remaining(isolated)["status"] == "stopped_inconclusive"
    with pytest.raises(ValueError, match="고정 순서"):
        C.execute_call(isolated, later["call"], 1, lambda messages, params: {})


def test_provider_model_mismatch_is_a_nonretryable_call_error(isolated, monkeypatch):
    monkeypatch.setattr(C.F, "first_messages", lambda kind, data: (
        [{"role": "user", "content": "x"}], {"messages_sha256": "1" * 64}))
    entry = C.call_order()[0]
    result = C.execute_call(isolated, entry["call"], 1, lambda messages, params: {
        "provider_model": "different/model", "content": "{}", "finish_reason": "stop", "wall_s": 1.0})
    assert result["status"] == "call_error"
    assert C.remaining(isolated)["status"] == "stopped_inconclusive"


def test_commit_gate_allows_outside_commit_but_rejects_dirty_bound_blob(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=repo, check=True)
    (repo / "source.py").write_text("value = 1\n", encoding="utf-8")
    (repo / "protocol.json").write_text("{}\n", encoding="utf-8")
    subprocess.run(["git", "add", "source.py", "protocol.json"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "bound"], cwd=repo, check=True)
    bound = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True,
                           text=True).stdout.strip()
    (repo / "outside.txt").write_text("outside\n", encoding="utf-8")
    subprocess.run(["git", "add", "outside.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "outside"], cwd=repo, check=True)
    monkeypatch.setattr(C, "ROOT", repo)
    monkeypatch.setattr(C, "PROTOCOL", repo / "protocol.json")
    monkeypatch.setattr(C, "SOURCE_FILES", {"source.py"})
    monkeypatch.setattr(C, "verify_protocol", lambda protocol=None: {
        "source_files": {"source.py": C.sha_lf(repo / "source.py")}})
    assert len(C.verify_commit_binding(bound)) == 40
    (repo / "source.py").write_text("value = 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="깨끗하지 않다"):
        C.verify_commit_binding(bound)


def test_registered_tag_cannot_be_cloned_and_plan_order_is_bound(isolated, monkeypatch):
    with pytest.raises(ValueError, match="단일 등록 tag"):
        C.run_dir("cloned_tag")
    monkeypatch.setattr(C, "verify_commit_binding", lambda commit: "b" * 40)
    assert REAL_LOAD_PLAN(isolated)["call_order_sha256"] == C.canonical_sha(C.call_order())
    plan = C.read(C.plan_path(isolated))
    plan["call_order_sha256"] = "0" * 64
    C.plan_path(isolated).write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(ValueError, match="호출 순서"):
        REAL_LOAD_PLAN(isolated)


def populate_all(tag: str, *, invalid_candidate_call: int | None = None, missing_usage: bool = False,
                 disagree_candidate: bool = False, candidate_hold: bool = False):
    for entry in C.call_order():
        if entry["kind"] == "diagnosis":
            label = "optimizer" if disagree_candidate and entry["model"] == C.CANDIDATE else "reward"
            verdict = diagnosis_verdict(label)
            if candidate_hold and entry["model"] == C.CANDIDATE:
                verdict = {"status": "valid", "decision": {"action": "hold", "ranking": [label], "probe": None}}
        else:
            verdict = correction_verdict()
        status = "invalid" if entry["call"] == invalid_candidate_call else "valid"
        tokens = (None, None, None) if missing_usage and entry["call"] == 1 else (2, 3, 5)
        write_attempt(tag, entry, status, verdict if status == "valid" else {"status": "invalid"}, tokens=tokens)


def test_summary_adoption_gate_invalid_counts_and_cross_model_agreement(isolated):
    candidate_call = next(row["call"] for row in C.call_order() if row["model"] == C.CANDIDATE)
    populate_all(isolated, invalid_candidate_call=candidate_call)
    report = C.summarize(isolated)
    assert report["outcome"] == "retain_current"
    assert report["model_summary"][C.CANDIDATE]["statuses"] == {"invalid": 1, "valid": 32}
    assert report["cross_model_decision_agreement"]["comparable"] == 32
    assert report["conditions"]["kind_valid_not_below_baseline"] is False


def test_summary_passes_only_same_contract_and_missing_usage_is_not_zero(isolated):
    populate_all(isolated, missing_usage=True)
    report = C.summarize(isolated)
    assert report["outcome"] == "conditional_future_demo_candidate"
    assert report["cross_model_decision_agreement"] == {"agreements": 33, "comparable": 33, "rate": 1.0}
    assert report["cost_all_attempts"]["input_tokens"] == {"known": 65, "missing": 1, "sum_known": 130}
    assert report["attempts"] == 66
    assert C.summarize(isolated) == report


def test_stable_hold_regression_cannot_be_adopted(isolated):
    populate_all(isolated, candidate_hold=True)
    report = C.summarize(isolated)
    assert report["model_summary"][C.CANDIDATE]["by_kind"]["diagnosis"]["action_counts"] == {"hold": 18}
    assert report["conditions"]["diagnosis_continue_not_below_baseline"] is False
    assert report["outcome"] == "retain_current"


def test_raw_response_mutation_and_reservation_mutation_are_rejected(isolated):
    populate_all(isolated)
    entry = C.call_order()[0]
    response_path = C.response_path(isolated, entry, 1)
    response = C.read(response_path)
    response["response"]["content"] = json.dumps(diagnosis_verdict("optimizer"))
    response_path.write_text(json.dumps(response), encoding="utf-8")
    with pytest.raises(ValueError, match="재검증"):
        C.summarize(isolated)

    response["response"]["content"] = json.dumps(response["verdict"])
    response_path.write_text(json.dumps(response), encoding="utf-8")
    reservation_path = C.reservation_path(isolated, entry, 1)
    reservation = C.read(reservation_path)
    reservation["request"]["max_tokens"] = 1
    reservation_path.write_text(json.dumps(reservation), encoding="utf-8")
    with pytest.raises(ValueError, match="예약 내용"):
        C.summarize(isolated)


def test_orphan_and_infra_make_summary_incomplete(isolated):
    first, second = C.call_order()[:2]
    C.write_new(C.reservation_path(isolated, first, 1), {"entry": first})
    write_attempt(isolated, second, "infra_error", None)
    report = C.summarize(isolated)
    assert report["outcome"] == "incomplete_inconclusive"
    assert report["conditions"]["complete"] is False


def test_in_progress_summary_is_preview_only_and_does_not_freeze_file(isolated):
    report = C.summarize(isolated)
    assert report["terminal"] is False and report["run_state"] == "ready"
    assert not (C.run_dir(isolated) / "summary.json").exists()


def test_dry_run_uses_mock_completion_and_makes_no_real_api_call(isolated, monkeypatch):
    monkeypatch.setattr(C.F, "first_messages", lambda kind, data: (
        [{"role": "user", "content": "x"}], {"messages_sha256": "1" * 64}))
    monkeypatch.setattr(C.F, "validate", lambda kind, content, packet: diagnosis_verdict())
    entry = C.call_order()[0]
    real = C.F.openai_complete
    calls = []

    def mock_complete(messages, params):
        calls.append((messages, params))
        return {"provider_model": entry["model"], "content": "{}", "finish_reason": "stop",
                "input_tokens": 1, "output_tokens": 1, "total_tokens": 2, "wall_s": 0.01}

    assert C.execute_call(isolated, entry["call"], 1, mock_complete)["status"] == "valid"
    assert len(calls) == 1 and C.F.openai_complete is real
