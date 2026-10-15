"""D3 hunk 인용 계약의 결합, 적대 입력, 호출 예산을 검사한다."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))
import d3_hunk_contract as H  # noqa: E402
import run_d3_hunk_contract as R  # noqa: E402


def sample_packet() -> dict:
    return {
        "schema": "change_rank_packet_v1",
        "changes": [{"label": "change_1", "diff": (
            "--- a/x.py\n+++ b/x.py\n@@ -1,3 +1,3 @@\n context = 1\n-old_value = policy.std\n"
            "+new_value = policy.log_std\n tail = 2\n@@ -10,2 +10,2 @@\n-other = 3\n+other = 4\n")}],
        "facts": {"telemetry:case:x": 1.0},
        "probe_observations": [{"probe": "P_value"}],
    }


def decision(**reason) -> dict:
    base = {"change": "change_1", "facts": ["telemetry:case:x"], "probes": ["P_value"],
            "hunk": "H1", "quote": "new_value = policy.log_std"}
    return {"ranking": ["change_1"], "reasons": [base | reason], "uncertain": False}


def test_hunks_are_numbered_per_change_and_messages_do_not_request_line_numbers():
    transformed = H.build_packet(sample_packet())
    assert [hunk["hunk"] for hunk in transformed["changes"][0]["hunks"]] == ["H1", "H2"]
    assert H.validate(decision(), sample_packet()) == []
    prompt = json.dumps(H.messages(sample_packet()), ensure_ascii=False)
    assert "diff_lines" not in prompt and "line numbers" not in prompt and "H1" in prompt


@pytest.mark.parametrize(("mutation", "problem"), [
    ({"quote": "new_value = policy.imagined_std"}, "reason0_quote"),
    ({"hunk": "H2", "quote": "new_value = policy.log_std"}, "reason0_quote"),
    ({"quote": "context = 1"}, "reason0_quote"),
    ({"quote": "policy"}, "reason0_quote"),
    ({"hunk": "H9"}, "reason0_hunk"),
    ({"hunk": ["H1"]}, "reason0_hunk"),
])
def test_invented_mismatched_context_only_and_short_quotes_are_rejected(mutation, problem):
    assert problem in H.validate(decision(**mutation), sample_packet())


@pytest.mark.parametrize(("mutation", "problem"), [
    ({"change": {"label": "change_1"}}, "reason0_shape"),
    ({"change": ["change_1"]}, "reason0_shape"),
    ({"facts": [{"id": "telemetry:case:x"}]}, "reason0_facts"),
    ({"probes": [{"probe": "P_value"}]}, "reason0_probes"),
])
def test_malformed_json_types_return_problems_instead_of_raising(mutation, problem):
    assert problem in H.validate(decision(**mutation), sample_packet())


def test_non_string_model_content_is_not_parsed():
    assert H.parse({"ranking": []}) is None


def test_quote_must_be_contiguous_and_include_a_changed_line():
    assert "reason0_quote" in H.validate(decision(quote="context = 1\ntail = 2"), sample_packet())
    assert H.validate(decision(quote="context = 1\nold_value = policy.std"), sample_packet()) == []


def test_protocol_binds_eight_packets_call_order_and_budget():
    protocol = R.read_protocol()
    R.verify_protocol(protocol)
    order = R.call_order()
    assert len(order) == 48 and {row["arm"] for row in order} == {"old", "hunk"}
    assert all(sum(row["packet_id"] == pid and row["arm"] == arm for row in order) == 3
               for pid in protocol["packets"] for arm in ("old", "hunk"))
    assert R.BASE_CALLS + R.INFRA_RETRY_CAP == R.TOTAL_API_CAP == 60


def test_committed_gate_accepts_checkout_line_ending_normalization(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=repo, check=True)
    subprocess.run(["git", "config", "core.autocrlf", "true"], cwd=repo, check=True)
    source = repo / "source.py"
    source.write_bytes(b"value = 1\n")
    subprocess.run(["git", "add", "source.py"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "test"], cwd=repo, check=True)
    source.write_bytes(b"value = 1\r\n")
    monkeypatch.setattr(R, "ROOT", repo)
    monkeypatch.setattr(R, "CODE", {"source.py"})
    assert len(R.verify_committed()) == 40
    source.write_bytes(b"value = 2\r\n")
    with pytest.raises(ValueError, match="커밋"):
        R.verify_committed()


def _client(content: str):
    sent = []
    def create(**kwargs):
        sent.append(kwargs)
        return SimpleNamespace(id="r1", model=R.MODEL,
                               choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason="stop")],
                               usage=SimpleNamespace(prompt_tokens=2, completion_tokens=3, total_tokens=5))
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return lambda: client, sent


def test_call_writes_immutable_record_without_secret(tmp_path, monkeypatch):
    entry = R.call_order()[0]
    monkeypatch.setattr(R, "PRIVATE", tmp_path / "task4")
    monkeypatch.setattr(R, "verify_committed", lambda: "a" * 40)
    monkeypatch.setenv("NVIDIA_API_KEY", "test-secret")
    output = R.record_path(entry, 1)
    factory, sent = _client(json.dumps(decision()))
    assert R.call(entry, 1, output, client_factory=factory, require_posix=False) == 0
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["schema"] == "d3_hunk_contract_call_v1"
    assert record["status"] == "answered" and sent[0]["temperature"] == 0
    assert "test-secret" not in output.read_text(encoding="utf-8")
    ledger = (R.PRIVATE / "ledger.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(ledger) == 2
    assert [json.loads(line)["event"] for line in ledger] == ["reserved", "completed"]
    assert json.loads(ledger[1])["record_sha256"] == R.sha(output)
    assert R._used_attempts() == 1 and not R.lock_path().exists()
    with pytest.raises(FileExistsError):
        R.call(entry, 1, output, client_factory=factory, require_posix=False)


def test_semantic_invalid_is_not_retryable_and_denominator_keeps_it(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "PRIVATE", tmp_path / "task4")
    entry = R.call_order()[0]
    packet_path = next(path for path in R.packet_paths() if R.packet_id(path) == entry["packet_id"])
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    record = {"entry": entry, "packet_sha256": R.read_protocol()["packets"][entry["packet_id"]],
              "protocol_sha256": R.sha(R.PROTOCOL), "request": R.REQUEST,
              "messages_sha256": R.canonical_sha(R.messages(entry["arm"], packet)),
              "status": "answered", "content": "refusal"}
    assert R.review(entry, record)["status"] == "invalid"


def test_ledger_rejects_reordering_or_artifact_tampering(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "PRIVATE", tmp_path / "task4")
    reservation = R.PRIVATE / "reservations/a.json"
    R._write_new_durable(reservation, {"reserved": True})
    R._append_ledger({"schema": "d3_hunk_contract_ledger_v1", "event": "reserved", "call": 1, "attempt": 1,
                      "reservation": R.portable_path(reservation), "reservation_sha256": R.sha(reservation)})
    assert len(R.verify_ledger()) == 1
    reservation.write_text('{"changed": true}', encoding="utf-8")
    with pytest.raises(ValueError, match="예약·호출"):
        R.verify_ledger()


def test_reservation_precedes_remote_call_and_crash_is_terminal_unknown(tmp_path, monkeypatch):
    entry = R.call_order()[0]
    monkeypatch.setattr(R, "PRIVATE", tmp_path / "task4")
    monkeypatch.setattr(R, "verify_committed", lambda: "a" * 40)
    monkeypatch.setenv("NVIDIA_API_KEY", "test-secret")

    class HardStop:
        class Completions:
            def create(self, **_kwargs):
                assert R.reservation_path(entry, 1).is_file()
                assert R._used_attempts() == 1
                raise KeyboardInterrupt
        chat = SimpleNamespace(completions=Completions())

    with pytest.raises(KeyboardInterrupt):
        R.call(entry, 1, R.record_path(entry, 1), client_factory=lambda: HardStop(), require_posix=False)
    assert R.reservation_path(entry, 1).is_file() and not R.record_path(entry, 1).exists()
    assert not R.lock_path().exists()
    plan = R.plan()
    assert plan["status"] == "orphaned_reservations" and plan["orphaned_reservations"][0]["call"] == entry["call"]
    summary = R.summarize(tmp_path / "not_written.json")
    orphan = next(row for row in summary["rows"] if row["call"] == entry["call"])
    assert summary["status"] == "incomplete" and orphan["status"] == "orphaned_reservation"


def test_global_lock_rejects_concurrent_call_and_requires_sha_bound_resolution(tmp_path, monkeypatch):
    entry = R.call_order()[0]
    monkeypatch.setattr(R, "PRIVATE", tmp_path / "task4")
    monkeypatch.setattr(R, "verify_committed", lambda: "a" * 40)
    monkeypatch.setenv("NVIDIA_API_KEY", "test-secret")
    R._write_new_durable(R.lock_path(), {"schema": "d3_hunk_contract_call_lock_v1", "pid": 999})
    assert R.plan()["status"] == "locked"
    with pytest.raises(RuntimeError, match="lock"):
        R.call(entry, 1, R.record_path(entry, 1), client_factory=lambda: None, require_posix=False)
    lock_sha = R.sha(R.lock_path())
    with pytest.raises(ValueError):
        R.resolve_lock("0" * 64, "프로세스 부재 확인")
    resolved = R.resolve_lock(lock_sha, "운영자가 PID 부재를 확인함")
    assert resolved["status"] == "resolved" and not R.lock_path().exists()
    assert len(list((R.PRIVATE / "lock_resolutions").glob("*.json"))) == 1


def test_record_without_completed_ledger_is_not_validated_or_retried(tmp_path, monkeypatch):
    entry = R.call_order()[0]
    monkeypatch.setattr(R, "PRIVATE", tmp_path / "task4")
    monkeypatch.setattr(R, "verify_committed", lambda: "a" * 40)
    monkeypatch.setenv("NVIDIA_API_KEY", "test-secret")
    original_append = R._append_ledger
    def interrupted_completion(row):
        if row["event"] == "completed":
            raise KeyboardInterrupt
        return original_append(row)
    monkeypatch.setattr(R, "_append_ledger", interrupted_completion)
    factory, _ = _client("{}")
    with pytest.raises(KeyboardInterrupt):
        R.call(entry, 1, R.record_path(entry, 1), client_factory=factory, require_posix=False)
    assert R.record_path(entry, 1).exists() and R._used_attempts() == 1
    assert R.plan()["status"] == "orphaned_reservations"
    report = R.summarize(tmp_path / "not_written.json")
    assert next(r for r in report["rows"] if r["call"] == entry["call"])["status"] == "orphaned_reservation"


def test_reservation_count_stops_before_remote_call_at_global_cap(tmp_path, monkeypatch):
    entry = R.call_order()[0]
    monkeypatch.setattr(R, "PRIVATE", tmp_path / "task4")
    monkeypatch.setattr(R, "verify_committed", lambda: "a" * 40)
    monkeypatch.setenv("NVIDIA_API_KEY", "test-secret")
    for index in range(R.TOTAL_API_CAP):
        R._write_new_durable(R.PRIVATE / "reservations" / f"prior_{index:02d}.json", {"index": index})
    called = False
    def client_factory():
        nonlocal called
        called = True
        return None
    with pytest.raises(ValueError, match="상한"):
        R.call(entry, 1, R.record_path(entry, 1), client_factory=client_factory, require_posix=False)
    assert not called and R._used_attempts() == 60 and not R.lock_path().exists()
