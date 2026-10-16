from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from evals import vl_aux_observer_v1 as O


SHA_A = "a" * 64
SHA_B = "b" * 64
ALIAS = "private/nvidia/nemotron-3.5-super-120b-a12b"


def test_budget_constants_cover_three_expected_and_four_maximum_policies():
    assert O.EXPECTED_POLICY_COUNT == 3
    assert O.EXPECTED_API_CAP == 6
    assert O.MAX_POLICY_COUNT == 4
    assert O.TOTAL_API_CAP == 8


def _packet(tmp_path: Path, monkeypatch, name: str = "one") -> Path:
    video = tmp_path / f"{name}.mp4"
    video.write_bytes(("video-" + name).encode())

    def fake_extract(source, destination):
        rows = []
        for index in range(6):
            payload = f"jpeg-{name}-{index}".encode()
            path = destination / f"frame_{index + 1:02d}.jpg"
            path.write_bytes(payload)
            rows.append({"file": path.name, "timestamp_s": index + .5,
                         "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)})
        return rows

    monkeypatch.setattr(O.VISION, "_extract_frames", fake_extract)
    return O.prepare(video, tmp_path / f"prepared-{name}", {
        "checkpoint_sha256": SHA_A, "fixed_eval_sha256": SHA_B})


def _valid(content='{"walking":"yes","in_place":"no","falling":"no"}', provider=ALIAS):
    return {"provider_model": provider, "content": content, "finish_reason": "stop",
            "input_tokens": 10, "output_tokens": 4, "total_tokens": 14, "wall_s": .2}


def test_prepare_binds_six_frames_and_keeps_private_metadata_out_of_messages(tmp_path, monkeypatch):
    packet_path = _packet(tmp_path, monkeypatch)
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    assert len(packet["frames"]) == 6
    assert packet["clip_id"] not in {"NONE", "FAULT", "corrected"}
    assert packet["binding"] == {"checkpoint_sha256": SHA_A, "fixed_eval_sha256": SHA_B}
    case, public = O._case(packet_path)
    sent = json.dumps(case["messages"], ensure_ascii=False)
    assert SHA_A not in sent and SHA_B not in sent
    assert all(word not in sent.lower() for word in ("none", "fault", "corrected", "seed", "fixed_eval"))
    assert sum(item.get("type") == "image_url" for item in case["messages"][1]["content"]) == 6
    assert case["request_params"] == O.VISION._request_params()
    assert public["clip_id"] == packet["clip_id"]


def test_call_accepts_registered_provider_alias_and_does_not_send_binding(tmp_path, monkeypatch):
    packet = _packet(tmp_path, monkeypatch)
    seen = {}

    def complete(messages, params):
        seen["messages"] = messages
        seen["params"] = params
        return _valid()

    record = O.call(packet, tmp_path / "result.json", tmp_path / "ledger", complete=complete)
    assert record["status"] == "valid" and record["requests"] == 1
    result = record["attempts"][0]["result"]
    assert result["provider_model_raw"] == ALIAS
    assert result["provider_identity_registered"] is True
    assert result["identity_policy"] == "registered_endpoint_alias_v3_not_weights_attestation"
    encoded = json.dumps(seen, ensure_ascii=False)
    assert SHA_A not in encoded and SHA_B not in encoded
    assert seen["params"]["model"] == O.MODEL
    assert seen["params"]["temperature"] == 0 and seen["params"]["max_tokens"] == 300
    assert seen["params"]["extra_body"] == {"reasoning_effort": "none"}


def test_semantic_invalid_has_no_retry_and_wrong_alias_is_call_error(tmp_path, monkeypatch):
    packet = _packet(tmp_path, monkeypatch, "invalid")
    calls = []
    record = O.call(packet, tmp_path / "invalid.json", tmp_path / "ledger1",
                    complete=lambda *_: calls.append(1) or _valid('{"walking":"maybe"}'))
    assert record["status"] == "invalid" and record["requests"] == 1 and calls == [1]

    second = _packet(tmp_path, monkeypatch, "alias")
    record = O.call(second, tmp_path / "alias.json", tmp_path / "ledger2",
                    complete=lambda *_: _valid(provider="private/nvidia/wrong"))
    assert record["status"] == "call_error" and record["requests"] == 1


def test_only_infrastructure_error_retries_once(tmp_path, monkeypatch):
    packet = _packet(tmp_path, monkeypatch, "retry")
    replies = iter([
        {"error_type": "APIStatusError", "status_code": 503, "infra_error": "APIStatusError"},
        _valid(),
    ])
    record = O.call(packet, tmp_path / "retry.json", tmp_path / "ledger", complete=lambda *_: next(replies))
    assert record["status"] == "valid" and record["requests"] == 2
    assert [row["transport_status"] for row in record["attempts"]] == ["infra_error", "ok"]
    assert len(O.RequestLedger(tmp_path / "ledger").audit()) == 2


def test_duplicate_packet_quota_and_crash_unknown_are_fail_closed(tmp_path, monkeypatch):
    ledger_root = tmp_path / "ledger"
    packet = _packet(tmp_path, monkeypatch, "duplicate")
    O.call(packet, tmp_path / "first.json", ledger_root, complete=lambda *_: _valid())
    with pytest.raises(O.LedgerStop, match="같은 packet"):
        O.call(packet, tmp_path / "second.json", ledger_root, complete=lambda *_: _valid())

    for index in range(1, 8):
        next_packet = _packet(tmp_path, monkeypatch, f"quota{index}")
        O.call(next_packet, tmp_path / f"quota{index}.json", ledger_root, complete=lambda *_: _valid())
    assert len(O.RequestLedger(ledger_root).audit()) == 8
    overflow = _packet(tmp_path, monkeypatch, "overflow")
    with pytest.raises(O.LedgerStop, match="상한"):
        O.call(overflow, tmp_path / "overflow.json", ledger_root, complete=lambda *_: _valid())

    crash_root = tmp_path / "crash-ledger"
    crash_ledger = O.RequestLedger(crash_root)
    crash_ledger.reserve("c" * 64, "opaque", 1, "d" * 64)
    other = _packet(tmp_path, monkeypatch, "after-crash")
    with pytest.raises(O.LedgerStop, match="crash-unknown"):
        O.call(other, tmp_path / "after-crash.json", crash_root, complete=lambda *_: _valid())
    assert O.RequestLedger(crash_root).audit()[0]["response"] is None


def test_tampered_frame_and_forbidden_binding_are_rejected(tmp_path, monkeypatch):
    packet = _packet(tmp_path, monkeypatch, "tamper")
    (packet.parent / "frames/frame_01.jpg").write_bytes(b"changed")
    with pytest.raises(ValueError, match="prepared frame binding"):
        O.call(packet, tmp_path / "tampered.json", tmp_path / "ledger", complete=lambda *_: _valid())
    with pytest.raises(ValueError, match="binding"):
        O.prepare(tmp_path / "tamper.mp4", tmp_path / "bad-binding", {
            "checkpoint_sha256": SHA_A, "fixed_eval_sha256": SHA_B, "arm": "FAULT"})


def test_invoke_delegates_to_owned_worker_and_reverifies_record(tmp_path, monkeypatch):
    packet = _packet(tmp_path, monkeypatch, "invoke")
    output = tmp_path / "invoke.json"
    ledger = tmp_path / "invoke-ledger"
    captured = []

    def invoke_worker(kind, packet_arg, output_arg, ledger_arg, *, runner):
        captured.append((kind, packet_arg, output_arg, ledger_arg, runner))
        O.call(packet, output, ledger, complete=lambda *_: _valid())

    runner = object()
    monkeypatch.setattr(O.WORKER, "invoke_worker", invoke_worker)
    record = O.invoke(packet, output, ledger, runner=runner)
    assert record["status"] == "valid"
    assert captured == [("observation", packet, output, ledger, runner)]


def test_invoke_rejects_forged_valid_output_without_ledger(tmp_path, monkeypatch):
    packet = _packet(tmp_path, monkeypatch, "forged")
    output = tmp_path / "forged.json"
    ledger = tmp_path / "forged-ledger"

    def invoke_worker(*_args, **_kwargs):
        output.write_text(json.dumps({"schema": O.RECORD_SCHEMA, "status": "valid"}), encoding="utf-8")

    monkeypatch.setattr(O.WORKER, "invoke_worker", invoke_worker)
    with pytest.raises(ValueError, match="원장 수"):
        O.invoke(packet, output, ledger, runner=object())


def test_verify_record_rejects_record_and_ledger_semantic_mismatch(tmp_path, monkeypatch):
    packet = _packet(tmp_path, monkeypatch, "mismatch")
    output = tmp_path / "mismatch.json"
    ledger = tmp_path / "mismatch-ledger"
    O.call(packet, output, ledger, complete=lambda *_: _valid())
    record = json.loads(output.read_text(encoding="utf-8"))
    record["status"] = "invalid"
    output.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="재계산"):
        O.verify_record(packet, output, ledger)

    output.unlink()
    O._write_new(output, {**record, "status": "valid"})
    response_path = ledger / "responses/0001.json"
    response = json.loads(response_path.read_text(encoding="utf-8"))
    response["transport_status"] = "infra_error"
    response_path.write_text(json.dumps(response), encoding="utf-8")
    with pytest.raises(ValueError, match="transport status"):
        O.verify_record(packet, output, ledger)


def test_verify_record_rejects_unrelated_crash_unknown_reservation(tmp_path, monkeypatch):
    packet = _packet(tmp_path, monkeypatch, "orphan")
    output = tmp_path / "orphan.json"
    ledger_root = tmp_path / "orphan-ledger"
    O.call(packet, output, ledger_root, complete=lambda *_: _valid())
    O.RequestLedger(ledger_root).reserve("c" * 64, "other", 1, "d" * 64)
    with pytest.raises(ValueError, match="완료 응답 없는"):
        O.verify_record(packet, output, ledger_root)
