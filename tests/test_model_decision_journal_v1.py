from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from evals import model_decision_journal_v1 as J
from evals import ppo_selection_adapter_v1 as B

MESSAGES = [{"role": "user", "content": "choose"}]
PARAMS = {"model": "fixture", "temperature": 0}


def binding(run_id: str = "run-1") -> dict:
    return {
        "run_id": run_id,
        "source_anchor": "a" * 40,
        "protocol_sha256": "b" * 64,
        "checkpoint_sha256": "c" * 64,
        "snapshot_sha256": "d" * 64,
        "initial_messages_sha256": J.digest(MESSAGES),
    }


class Clock:
    def __init__(self) -> None:
        self.value = datetime(2026, 10, 25, 0, 0, tzinfo=timezone.utc)  # noqa: UP017 - Python 3.10 호환

    def __call__(self) -> datetime:
        return self.value


def retry_engine(completion) -> dict:
    first = completion(MESSAGES, PARAMS)
    if J.DecisionJournal._retryable(first):
        second = completion(MESSAGES, PARAMS)
        selected = "hold" if J.DecisionJournal._retryable(second) else "continue"
        return {"stop_reason": "infra_error" if selected == "hold" else "completed",
                "selection": {"action": selected}, "results": [first, second]}
    return {"stop_reason": "completed", "selection": {"action": "continue"}, "results": [first]}


def one_call_engine(completion) -> dict:
    result = completion(MESSAGES, PARAMS)
    return {"stop_reason": "completed", "selection": {"action": "continue"}, "results": [result]}


def test_infra_pause_survives_process_reconstruction_and_calls_once_after_deadline(tmp_path: Path) -> None:
    root, api, clock = tmp_path / "decision", tmp_path / "api", Clock()
    calls = []

    def provider(messages, params):
        calls.append((messages, params))
        return ({"status_code": 500, "infra_error": "InternalServerError", "wall_s": 0.1}
                if len(calls) == 1 else {"content": "ok", "wall_s": 0.2})

    first = J.DecisionJournal(root, api, binding(), PARAMS, retry_delay_s=30).advance(
        retry_engine, provider, clock)
    assert first["status"] == "paused_infra" and len(calls) == 1
    assert first["api"] == {"reserved": 1, "pending": 0, "remaining": 19}

    clock.value += timedelta(seconds=29)
    second = J.DecisionJournal(root, api, binding(), PARAMS, retry_delay_s=30).advance(
        retry_engine, provider, clock)
    assert second["status"] == "paused_infra" and len(calls) == 1

    clock.value += timedelta(seconds=1)
    third = J.DecisionJournal(root, api, binding(), PARAMS, retry_delay_s=30).advance(
        retry_engine, provider, clock)
    assert third["status"] == "choice_ready" and len(calls) == 2
    assert third["api"] == {"reserved": 2, "pending": 0, "remaining": 18}

    def forbidden(*_args):
        raise AssertionError("terminal replay에서 provider를 호출하면 안 된다")

    replay = J.DecisionJournal(root, api, binding(), PARAMS).advance(retry_engine, forbidden, clock)
    assert replay["status"] == "choice_ready" and replay["engine_result"] == third["engine_result"]


def test_new_infra_response_always_requires_a_later_advance_even_with_zero_delay(tmp_path: Path) -> None:
    root, api, clock = tmp_path / "decision", tmp_path / "api", Clock()
    rows = iter(({"status_code": 500, "infra_error": "InternalServerError"}, {"content": "ok"}))
    journal = J.DecisionJournal(root, api, binding(), PARAMS, retry_delay_s=0)
    first = journal.advance(retry_engine, lambda *_: next(rows), clock)
    assert first["status"] == "paused_infra" and first["api"]["reserved"] == 1
    second = J.DecisionJournal(root, api, binding(), PARAMS, retry_delay_s=0).advance(
        retry_engine, lambda *_: next(rows), clock)
    assert second["status"] == "choice_ready" and second["api"]["reserved"] == 2


def test_pending_reservation_from_crash_blocks_all_decisions(tmp_path: Path) -> None:
    root, api, clock = tmp_path / "decision", tmp_path / "api", Clock()
    journal = J.DecisionJournal(root, api, binding(), PARAMS)
    reservation = {
        "schema": "model_decision_reservation_v1", "index": 1,
        "manifest_sha256": journal.manifest_sha256, "binding_sha256": journal.binding_sha256,
        "run_id": "run-1", "ordinal": 1, "messages_sha256": J.digest(MESSAGES),
        "params_sha256": journal.params_sha256, "reserved_at": clock().isoformat(),
    }
    J._write_new(api / "reservations/0001.json", reservation)
    called = False

    def provider(*_args):
        nonlocal called
        called = True
        return {}

    result = J.DecisionJournal(root, api, binding(), PARAMS).advance(one_call_engine, provider, clock)
    assert result["status"] == "blocked_pending" and result["reason"] == "pending_reservation"
    assert called is False


def test_response_mutation_is_rejected_before_replay(tmp_path: Path) -> None:
    root, api, clock = tmp_path / "decision", tmp_path / "api", Clock()
    journal = J.DecisionJournal(root, api, binding(), PARAMS)
    assert journal.advance(one_call_engine, lambda *_: {"content": "ok"}, clock)["status"] == "choice_ready"
    response = api / "responses/0001.json"
    value = json.loads(response.read_text(encoding="utf-8"))
    value["result"]["content"] = "tampered"
    response.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(J.JournalIntegrityError):
        J.DecisionJournal(root, api, binding(), PARAMS).advance(one_call_engine, lambda *_: {}, clock)


def test_changed_binding_and_old_terminal_are_not_adopted(tmp_path: Path) -> None:
    api, clock = tmp_path / "api", Clock()
    root1, root2 = tmp_path / "one", tmp_path / "two"
    first = J.DecisionJournal(root1, api, binding("one"), PARAMS)
    first.advance(one_call_engine, lambda *_: {"content": "ok"}, clock)
    changed = binding("one")
    changed["snapshot_sha256"] = "e" * 64
    with pytest.raises(J.JournalIntegrityError):
        J.DecisionJournal(root1, api, changed, PARAMS)

    second = J.DecisionJournal(root2, api, binding("two"), PARAMS)
    (root2 / "terminal.json").write_bytes((root1 / "terminal.json").read_bytes())
    called = False

    def provider(*_args):
        nonlocal called
        called = True
        return {}

    with pytest.raises(J.JournalIntegrityError):
        second.advance(one_call_engine, provider, clock)
    assert called is False


def test_global_cap_is_shared_by_distinct_decisions(tmp_path: Path) -> None:
    api, clock = tmp_path / "api", Clock()
    calls = []

    def provider(*_args):
        calls.append(1)
        return {"content": "ok"}

    for ordinal in (1, 2):
        result = J.DecisionJournal(tmp_path / f"d{ordinal}", api, binding(f"d{ordinal}"), PARAMS,
                                   api_cap=2).advance(one_call_engine, provider, clock)
        assert result["status"] == "choice_ready"
    stopped = J.DecisionJournal(tmp_path / "d3", api, binding("d3"), PARAMS, api_cap=2).advance(
        one_call_engine, provider, clock)
    assert stopped["status"] == "terminal_hold"
    assert stopped["engine_result"]["stop_reason"] == "budget_exhausted"
    assert len(calls) == 2
    replay = J.DecisionJournal(tmp_path / "d3", api, binding("d3"), PARAMS, api_cap=2).advance(
        one_call_engine, provider, clock)
    assert replay == stopped and len(calls) == 2


def test_active_lease_prevents_double_run(tmp_path: Path) -> None:
    root, api, clock = tmp_path / "decision", tmp_path / "api", Clock()
    journal = J.DecisionJournal(root, api, binding(), PARAMS)
    J._write_new(root / ".active", {
        "schema": "model_decision_active_lease_v1", "scope": "decision", "pid": 999,
        "thread_id": 1, "nonce": "existing", "manifest_sha256": journal.manifest_sha256,
    })
    result = journal.advance(one_call_engine, lambda *_: {"content": "no"}, clock)
    assert result["status"] == "blocked_pending" and result["reason"] == "active_lease"


def test_corrupt_active_lease_is_integrity_error(tmp_path: Path) -> None:
    root, api, clock = tmp_path / "decision", tmp_path / "api", Clock()
    journal = J.DecisionJournal(root, api, binding(), PARAMS)
    J._write_new(root / ".active", {"schema": "wrong"})
    with pytest.raises(J.JournalIntegrityError, match="active lease"):
        journal.advance(one_call_engine, lambda *_: {}, clock)


def test_engine_params_and_initial_messages_are_exact(tmp_path: Path) -> None:
    root, api, clock = tmp_path / "decision", tmp_path / "api", Clock()
    journal = J.DecisionJournal(root, api, binding(), PARAMS)

    def changed_params(completion):
        completion(MESSAGES, {**PARAMS, "temperature": 1})
        return {}

    with pytest.raises(J.JournalIntegrityError, match="params"):
        journal.advance(changed_params, lambda *_: {}, clock)
    assert not list((api / "reservations").glob("*.json"))

    def changed_messages(completion):
        completion([{"role": "user", "content": "changed"}], PARAMS)
        return {}

    with pytest.raises(J.JournalIntegrityError, match="첫 messages"):
        journal.advance(changed_messages, lambda *_: {}, clock)
    assert not list((api / "reservations").glob("*.json"))


def test_terminal_replay_rejects_second_feedback_message_drift(tmp_path: Path) -> None:
    root, api, clock = tmp_path / "decision", tmp_path / "api", Clock()
    feedback = [*MESSAGES, {"role": "user", "content": "feedback one"}]

    def original(completion):
        first = completion(MESSAGES, PARAMS)
        second = completion(feedback, PARAMS)
        return {"stop_reason": "completed", "selection": {"action": "continue"},
                "results": [first, second]}

    values = iter(({"content": "invalid"}, {"content": "valid"}))
    journal = J.DecisionJournal(root, api, binding(), PARAMS)
    assert journal.advance(original, lambda *_: next(values), clock)["status"] == "choice_ready"

    changed_feedback = [*MESSAGES, {"role": "user", "content": "feedback two"}]

    def changed(completion):
        first = completion(MESSAGES, PARAMS)
        second = completion(changed_feedback, PARAMS)
        return {"stop_reason": "completed", "selection": {"action": "continue"},
                "results": [first, second]}

    with pytest.raises(J.JournalIntegrityError, match="cached completion"):
        J.DecisionJournal(root, api, binding(), PARAMS).advance(changed, lambda *_: {}, clock)


def test_provider_exception_persists_only_exception_type(tmp_path: Path) -> None:
    root, api, clock = tmp_path / "decision", tmp_path / "api", Clock()

    def provider(*_args):
        raise TimeoutError("credential-like detail")

    result = J.DecisionJournal(root, api, binding(), PARAMS).advance(retry_engine, provider, clock)
    assert result["status"] == "paused_infra"
    response = json.loads((api / "responses/0001.json").read_text(encoding="utf-8"))
    assert response["result"]["error_type"] == "TimeoutError"
    assert "credential-like" not in (api / "responses/0001.json").read_text(encoding="utf-8")


def test_terminal_replay_in_fresh_python_process_makes_no_provider_call(tmp_path: Path) -> None:
    root, api, clock = tmp_path / "decision", tmp_path / "api", Clock()
    journal = J.DecisionJournal(root, api, binding(), PARAMS)
    assert journal.advance(one_call_engine, lambda *_: {"content": "ok"}, clock)["status"] == "choice_ready"
    script = r'''
import json,sys
from datetime import datetime,timezone
from pathlib import Path
from evals import model_decision_journal_v1 as J
messages=[{"role":"user","content":"choose"}]
params={"model":"fixture","temperature":0}
binding={"run_id":"run-1","source_anchor":"a"*40,"protocol_sha256":"b"*64,
 "checkpoint_sha256":"c"*64,"snapshot_sha256":"d"*64,
 "initial_messages_sha256":J.digest(messages)}
def engine(completion):
 result=completion(messages,params)
 return {"stop_reason":"completed","selection":{"action":"continue"},"results":[result]}
def forbidden(*args):
 raise AssertionError("provider called")
result=J.DecisionJournal(Path(sys.argv[1]),Path(sys.argv[2]),binding,params).advance(
 engine,forbidden,lambda:datetime(2026,10,25,tzinfo=timezone.utc))
print(json.dumps(result,sort_keys=True))
'''
    completed = subprocess.run(
        [sys.executable, "-c", script, str(root), str(api)], cwd=Path(__file__).resolve().parents[1],
        check=True, capture_output=True, text=True, encoding="utf-8",
    )
    assert json.loads(completed.stdout)["status"] == "choice_ready"


def test_semantic_hold_is_terminal_and_not_resumable(tmp_path: Path) -> None:
    root, api, clock = tmp_path / "decision", tmp_path / "api", Clock()

    def hold_engine(completion):
        result = completion(MESSAGES, PARAMS)
        return {"stop_reason": "completed", "selection": {"action": "hold"}, "results": [result]}

    journal = J.DecisionJournal(root, api, binding(), PARAMS)
    first = journal.advance(hold_engine, lambda *_: {"content": "invalid"}, clock)
    assert first["status"] == "terminal_hold"
    replay = J.DecisionJournal(root, api, binding(), PARAMS).advance(
        hold_engine, lambda *_: pytest.fail("terminal HOLD가 provider를 다시 호출했다"), clock)
    assert replay == first


def test_dotnet_seven_digit_timestamp_parser_is_accepted() -> None:
    parsed = J._parse_stamp("2026-10-25T01:02:03.1234567+00:00")
    assert parsed.microsecond == 123456


def test_actual_ppo_engine_preserves_infra_pause_then_terminal_invalid_hold(tmp_path: Path) -> None:
    packet = {
        "schema": B.C.PACKET_SCHEMA,
        "facts": {"telemetry:error": 1.2, "telemetry:alive": 20},
        "observed": [],
        "budget_left": B.C.MAX_PROBES,
        "execution_budget_s": 1200,
    }
    snapshot = B.build_snapshot(packet)
    params = B.CALL.F.request_params(B.MODEL)
    initial_messages = B.messages(snapshot)
    bound = binding("ppo-engine")
    bound["initial_messages_sha256"] = J.digest(initial_messages)
    root, api, clock = tmp_path / "decision", tmp_path / "api", Clock()
    calls = []

    def provider(_messages, _params):
        calls.append(1)
        if len(calls) == 1:
            return {"infra_error": "InternalServerError"}
        return {
            "response_id": f"r{len(calls)}",
            "provider_model": B.MODEL,
            "finish_reason": "stop",
            "content": json.dumps({"unexpected": True}),
            "reasoning_content": None,
            "input_tokens": 2,
            "output_tokens": 3,
            "total_tokens": 5,
            "wall_s": 0.1,
        }

    engine = lambda completion: B.engine(snapshot, completion)
    first = J.DecisionJournal(root, api, bound, params, retry_delay_s=30).advance(
        engine, provider, clock)
    assert first["status"] == "paused_infra" and len(calls) == 1

    clock.value += timedelta(seconds=30)
    second = J.DecisionJournal(root, api, bound, params, retry_delay_s=30).advance(
        engine, provider, clock)
    assert second["status"] == "terminal_hold" and len(calls) == 3
    assert second["engine_result"]["status"] == "terminal_hold"
    assert second["engine_result"]["reason"] == "selection_invalid"
    assert second["engine_result"]["selection"] is None
