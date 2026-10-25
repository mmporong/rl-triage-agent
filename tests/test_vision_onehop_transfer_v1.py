from __future__ import annotations

import base64
import importlib
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
O = importlib.import_module("vision_onehop_transfer_v1")
UTC = timezone.utc  # noqa: UP017 - Python 3.10 호환
NOW = datetime(2026, 10, 25, 13, 0, 20, tzinfo=UTC)
TARGET = {"id": "owned-id", "name": "owned-vm", "instance_type": "l40s-48gb.1x"}


def inventory(owner: dict = TARGET, status: str = "RUNNING") -> bytes:
    row = {
        "build_status": "READY",
        "gpu": "L40S",
        "health_status": "HEALTHY",
        **owner,
        "instance_kind": "GPU",
        "shell_status": "READY",
        "status": status,
    }
    return json.dumps({"workspaces": [row]}, separators=(",", ":")).encode()


def packet(
    *,
    raw: bytes | None = None,
    payload: bytes = b"echo '$HOME;$(id)'\n",
    attempt: str = "attempt-01",
) -> tuple[dict, str]:
    raw = inventory() if raw is None else raw
    receipt = {
        "command": ["brev", "ls", "--json"],
        "request_before_utc": "2026-10-25T13:00:18+00:00",
        "response_after_utc": "2026-10-25T13:00:20+00:00",
        "exit_code": 0,
        "stdout_path": "/historical/native-ls/stdout.json",
        "stdout_sha256": O.sha256_bytes(raw),
        "stderr_path": "/historical/native-ls/stderr.txt",
        "stderr_sha256": O.sha256_bytes(b""),
    }
    value = O.build_packet(
        native_receipt=receipt,
        raw_stdout=raw,
        raw_stderr=b"",
        source_binding={"source_sha256": "1" * 64},
        protocol_binding={"protocol_sha256": "2" * 64},
        target=TARGET,
        attempt_id=attempt,
        stage_id="registered-stage",
        stage_bytes=payload,
    )
    return value, O.encode_packet(value)


def execute(
    encoded: str,
    tmp_path: Path,
    run_stage,
    *,
    now: datetime = NOW,
    dry_run: bool = False,
    stage_sha: str | None = None,
):
    payload_sha = stage_sha or O.decode_packet(encoded)[0]["stage"]["sha256"]
    return O.execute_packet(
        encoded,
        state_dir=tmp_path,
        expected_target=TARGET,
        expected_source_binding={"source_sha256": "1" * 64},
        expected_protocol_binding={"protocol_sha256": "2" * 64},
        expected_stage_id="registered-stage",
        registered_stage_sha256=payload_sha,
        run_stage=run_stage,
        now_utc=now,
        dry_run=dry_run,
    )


def pending_intent(encoded: str, *, dry_run: bool = False) -> bytes:
    value, packet_sha = O.decode_packet(encoded)
    return (
        O.canonical_json(
            {
                "schema": "vision_onehop_intent_v1",
                "attempt_id": value["attempt_id"],
                "packet_sha256": packet_sha,
                "payload_sha256": value["stage"]["sha256"],
                "dry_run": dry_run,
            }
        )
        + b"\n"
    )


def test_packet_roundtrip_and_tamper_rejected() -> None:
    original, encoded = packet()
    decoded, digest = O.decode_packet(encoded)
    assert decoded == original and digest == O.sha256_bytes(O.canonical_json(original))
    envelope = json.loads(base64.b64decode(encoded))
    envelope["packet"]["target"]["id"] = "other-id"
    tampered = base64.b64encode(O.canonical_json(envelope)).decode()
    with pytest.raises(ValueError, match="packet SHA"):
        O.decode_packet(tampered)


def test_packet_requires_native_stream_sha_and_empty_stderr() -> None:
    raw = inventory()
    receipt = {
        "command": ["brev", "ls", "--json"],
        "request_before_utc": "2026-10-25T13:00:18+00:00",
        "response_after_utc": "2026-10-25T13:00:20+00:00",
        "exit_code": 0,
        "stdout_path": "/historical/stdout",
        "stdout_sha256": "0" * 64,
        "stderr_path": "/historical/stderr",
        "stderr_sha256": O.sha256_bytes(b"failure"),
    }
    with pytest.raises(ValueError, match="stdout SHA"):
        O.build_packet(
            native_receipt=receipt,
            raw_stdout=raw,
            raw_stderr=b"failure",
            source_binding={"source_sha256": "1" * 64},
            protocol_binding={"protocol_sha256": "2" * 64},
            target=TARGET,
            attempt_id="attempt-native",
            stage_id="registered-stage",
            stage_bytes=b"stage",
        )


def test_other_owner_and_stale_stop_before_stage(tmp_path: Path) -> None:
    other = {**TARGET, "id": "other-id"}
    with pytest.raises(ValueError, match="exact-owned"):
        packet(raw=inventory(other))
    _, encoded = packet()
    calls = []
    with pytest.raises(ValueError, match="freshness"):
        execute(
            encoded,
            tmp_path,
            lambda *args: calls.append(args),
            now=NOW + timedelta(seconds=31),
        )
    assert calls == [] and not (tmp_path / "attempt-01/intent.json").exists()


def test_shell_metacharacters_are_preserved_as_payload_bytes(tmp_path: Path) -> None:
    payload = b"printf '%s\\n' '$HOME' ';' '$(touch marker)'\n"
    _, encoded = packet(payload=payload)
    seen = []
    result = execute(
        encoded, tmp_path, lambda body, _packet: seen.append(body) or {"exit_code": 0}
    )
    assert seen == [payload] and result["stage_executed"] is True


def test_exact_footer_parser_and_missing_or_extra_footer_fail(tmp_path: Path) -> None:
    _, encoded = packet()
    result = execute(encoded, tmp_path, lambda *_: {"exit_code": 0})
    packet_sha = result["packet_sha256"]
    stdout = O.marker_line(result) + "\n" + TARGET["name"] + "\n"
    assert (
        O.parse_native_stdout(
            stdout,
            expected_target_name=TARGET["name"],
            expected_attempt_id="attempt-01",
            expected_packet_sha256=packet_sha,
            expected_payload_sha256=result["payload_sha256"],
        )
        == result
    )
    for invalid in (
        O.marker_line(result) + "\n",
        stdout + TARGET["name"] + "\n",
        "noise\n" + stdout,
        O.marker_line(result) + "\nwrong-name\n",
    ):
        with pytest.raises(ValueError, match="sentinel|footer"):
            O.parse_native_stdout(
                invalid,
                expected_target_name=TARGET["name"],
                expected_attempt_id="attempt-01",
                expected_packet_sha256=packet_sha,
                expected_payload_sha256=result["payload_sha256"],
            )
    tampered = {**result, "result_sha256": "0" * 64}
    with pytest.raises(ValueError, match="result SHA"):
        O.parse_native_stdout(
            O.marker_line(tampered) + "\n" + TARGET["name"] + "\n",
            expected_target_name=TARGET["name"],
            expected_attempt_id="attempt-01",
            expected_packet_sha256=packet_sha,
            expected_payload_sha256=result["payload_sha256"],
        )
    with pytest.raises(ValueError, match="binding"):
        O.parse_native_stdout(
            stdout,
            expected_target_name=TARGET["name"],
            expected_attempt_id="attempt-01",
            expected_packet_sha256=packet_sha,
            expected_payload_sha256="f" * 64,
        )


def test_same_attempt_replays_complete_without_stage_call(tmp_path: Path) -> None:
    _, encoded = packet()
    calls = []
    runner = lambda *_: calls.append(1) or {"exit_code": 0}
    first = execute(encoded, tmp_path, runner)
    second = execute(encoded, tmp_path, runner, now=NOW + timedelta(hours=1))
    assert second == first and calls == [1]


def test_partial_attempt_requires_reconcile_without_stage_call(tmp_path: Path) -> None:
    value, encoded = packet()
    packet_sha = O.sha256_bytes(O.canonical_json(value))
    attempt = tmp_path / "attempt-01"
    attempt.mkdir(parents=True)
    intent = {
        "schema": "vision_onehop_intent_v1",
        "attempt_id": "attempt-01",
        "packet_sha256": packet_sha,
        "payload_sha256": value["stage"]["sha256"],
        "dry_run": False,
    }
    (attempt / "intent.json").write_bytes(O.canonical_json(intent) + b"\n")
    calls = []
    result = execute(encoded, tmp_path, lambda *_: calls.append(1))
    assert result["status"] == "requires_reconcile" and calls == []

    stale = execute(
        encoded,
        tmp_path,
        lambda *_: pytest.fail("partial must not run"),
        now=NOW + timedelta(hours=1),
    )
    assert stale == result


@pytest.mark.skipif(os.name == "nt", reason="POSIX symlink/hardlink 경계")
def test_attempt_and_state_parent_symlinks_are_rejected(tmp_path: Path) -> None:
    _, encoded = packet()
    outside = tmp_path / "outside"
    outside.mkdir()
    state = tmp_path / "state"
    state.mkdir()
    (state / "attempt-01").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="attempt state"):
        execute(encoded, state, lambda *_: pytest.fail("symlink attempt must not run"))
    assert not (outside / "intent.json").exists()

    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(ValueError, match="lexical parent"):
        execute(
            encoded, linked_parent / "state", lambda *_: pytest.fail("parent symlink")
        )


@pytest.mark.skipif(os.name == "nt", reason="POSIX symlink/hardlink 경계")
@pytest.mark.parametrize("state_file", ["intent.json", "result.json", "attempt.lock"])
def test_attempt_state_file_symlinks_are_rejected(
    tmp_path: Path, state_file: str
) -> None:
    _, encoded = packet()
    attempt = tmp_path / "attempt-01"
    attempt.mkdir()
    outside = tmp_path / f"outside-{state_file}"
    outside.write_bytes(pending_intent(encoded))
    if state_file != "intent.json":
        (attempt / "intent.json").write_bytes(pending_intent(encoded))
    (attempt / state_file).symlink_to(outside)
    with pytest.raises(ValueError, match="symlink|안전하게|regular"):
        execute(encoded, tmp_path, lambda *_: pytest.fail("symlink state must not run"))


@pytest.mark.skipif(os.name == "nt", reason="POSIX symlink/hardlink 경계")
def test_replay_rejects_hardlinked_result_and_lock(tmp_path: Path) -> None:
    _, encoded = packet()
    execute(encoded, tmp_path, lambda *_: {"exit_code": 0})
    result = tmp_path / "attempt-01/result.json"
    os.link(result, tmp_path / "result-alias.json")
    with pytest.raises(ValueError, match="regular"):
        execute(
            encoded,
            tmp_path,
            lambda *_: pytest.fail("hardlinked result must not replay"),
        )

    other_root = tmp_path / "other-state"
    other_attempt = other_root / "attempt-01"
    other_attempt.mkdir(parents=True)
    lock_source = tmp_path / "lock-source"
    lock_source.write_bytes(b"0")
    os.link(lock_source, other_attempt / "attempt.lock")
    with pytest.raises(ValueError, match="lock.*regular"):
        execute(
            encoded, other_root, lambda *_: pytest.fail("hardlinked lock must not run")
        )


def test_same_attempt_mismatched_payload_is_rejected(tmp_path: Path) -> None:
    _, first = packet()
    execute(first, tmp_path, lambda *_: {})
    _, changed = packet(payload=b"different\n")
    with pytest.raises(ValueError, match="attempt payload|binding"):
        execute(
            changed, tmp_path, lambda *_: pytest.fail("mismatched attempt must not run")
        )


def test_dry_run_uses_same_guards_and_never_calls_stage(tmp_path: Path) -> None:
    payload = b"registered bytes\n"
    _, encoded = packet(payload=payload, attempt="dry-01")
    result = execute(
        encoded,
        tmp_path,
        lambda *_: pytest.fail("dry-run must not execute stage"),
        dry_run=True,
    )
    assert result["status"] == "complete" and result["dry_run"] is True
    assert result["stage_executed"] is False


def test_unregistered_stage_and_wrong_result_binding_fail_closed(
    tmp_path: Path,
) -> None:
    _, encoded = packet()
    with pytest.raises(ValueError, match="외부 등록값"):
        execute(encoded, tmp_path, lambda *_: pytest.fail("stage"), stage_sha="f" * 64)
    result = {
        "schema": O.RESULT_SCHEMA,
        "attempt_id": "wrong",
        "packet_sha256": "a" * 64,
    }
    stdout = O.marker_line(result) + "\n" + TARGET["name"] + "\n"
    with pytest.raises(ValueError, match="binding"):
        O.parse_native_stdout(
            stdout,
            expected_target_name=TARGET["name"],
            expected_attempt_id="attempt-01",
            expected_packet_sha256="a" * 64,
            expected_payload_sha256="b" * 64,
        )


@pytest.mark.parametrize("binding", ["target", "source", "protocol"])
def test_external_expected_bindings_are_required(tmp_path: Path, binding: str) -> None:
    _, encoded = packet()
    kwargs = {
        "state_dir": tmp_path,
        "expected_target": TARGET,
        "expected_source_binding": {"source_sha256": "1" * 64},
        "expected_protocol_binding": {"protocol_sha256": "2" * 64},
        "expected_stage_id": "registered-stage",
        "registered_stage_sha256": O.decode_packet(encoded)[0]["stage"]["sha256"],
        "run_stage": lambda *_: pytest.fail("wrong binding must not run"),
        "now_utc": NOW,
    }
    if binding == "target":
        kwargs["expected_target"] = {**TARGET, "id": "other-id"}
    elif binding == "source":
        kwargs["expected_source_binding"] = {"source_sha256": "3" * 64}
    else:
        kwargs["expected_protocol_binding"] = {"protocol_sha256": "4" * 64}
    with pytest.raises(ValueError, match="외부 등록값"):
        O.execute_packet(encoded, **kwargs)


def test_exec_transfer_validates_native_intent_packet_and_readback(
    tmp_path: Path,
) -> None:
    packet_value, encoded = packet(attempt="transfer-01", payload=b"registered stage\n")
    result = execute(encoded, tmp_path, lambda *_: {"exit_code": 0})
    template = (
        b"#!/usr/bin/env python3\nPACKET_B64 = '__RL_TRIAGE_ONEHOP_PACKET_B64__'\n"
    )
    script = O.render_exec_script(
        template,
        encoded,
        expected_template_sha256=O.sha256_bytes(template),
    )
    snapshot = "/historical/onehop.py"
    command = ["brev", "exec", TARGET["name"], "@" + snapshot]
    intent = (
        O.canonical_json(
            {
                "schema": "vision_atomic_exec_script_intent_v1",
                "source_path": "/historical/source/onehop.py",
                "source_sha256": O.sha256_bytes(script),
                "snapshot_path": snapshot,
                "command": command,
            }
        )
        + b"\n"
    )
    native_stdout = (O.marker_line(result) + "\n" + TARGET["name"] + "\n").encode()
    receipt = (
        O.canonical_json(
            {
                "command": command,
                "request_before_utc": "2026-10-25T13:00:20+00:00",
                "response_after_utc": "2026-10-25T13:00:21+00:00",
                "exit_code": 0,
                "stdout_path": "/historical/exec/stdout",
                "stdout_sha256": O.sha256_bytes(native_stdout),
                "stderr_path": "/historical/exec/stderr",
                "stderr_sha256": O.sha256_bytes(b""),
            }
        )
        + b"\n"
    )
    readback = (
        O.canonical_json(
            {
                "schema": "vision_onehop_artifact_readback_v1",
                "attempt_id": "transfer-01",
                "packet_sha256": result["packet_sha256"],
                "stage_id": packet_value["stage"]["id"],
                "stage_sha256": result["payload_sha256"],
                "result_sha256": O.sha256_bytes(O.canonical_json(result)),
            }
        )
        + b"\n"
    )
    envelope = O.build_exec_transfer_envelope(
        attempt_id="transfer-01",
        target_name=TARGET["name"],
        native_exec_receipt_bytes=receipt,
        native_exec_stdout_bytes=native_stdout,
        native_exec_stderr_bytes=b"",
        intent_bytes=intent,
        template_bytes=template,
        expected_template_sha256=O.sha256_bytes(template),
        script_bytes=script,
        payload_bytes=encoded.encode("ascii"),
        expected_packet_sha256=result["packet_sha256"],
        expected_payload_sha256=result["payload_sha256"],
        artifact_readback_bytes=readback,
        result=result,
    )
    assert envelope["schema"] == O.TRANSFER_SCHEMA
    assert set(envelope["artifacts"]) == {
        "native_exec_receipt",
        "native_exec_stdout",
        "native_exec_stderr",
        "intent",
        "template",
        "script",
        "payload",
        "artifact_readback",
        "result",
    }
    assert all(
        set(row) == {"bytes", "sha256"} for row in envelope["artifacts"].values()
    )
    assert envelope["claims"] == {
        "artifact_readback_content_bound": True,
        "artifact_readback_provenance": "not_verified",
    }

    broken = json.loads(readback)
    broken["stage_id"] = "other-stage"
    with pytest.raises(ValueError, match="artifact readback"):
        O.build_exec_transfer_envelope(
            attempt_id="transfer-01",
            target_name=TARGET["name"],
            native_exec_receipt_bytes=receipt,
            native_exec_stdout_bytes=native_stdout,
            native_exec_stderr_bytes=b"",
            intent_bytes=intent,
            template_bytes=template,
            expected_template_sha256=O.sha256_bytes(template),
            script_bytes=script,
            payload_bytes=encoded.encode("ascii"),
            expected_packet_sha256=result["packet_sha256"],
            expected_payload_sha256=result["payload_sha256"],
            artifact_readback_bytes=O.canonical_json(broken),
            result=result,
        )

    unrelated = b"#!/usr/bin/env python3\n# valid hash but no packet literal\n"
    unrelated_intent = json.loads(intent)
    unrelated_intent["source_sha256"] = O.sha256_bytes(unrelated)
    with pytest.raises(ValueError, match="template.*packet 결합"):
        O.build_exec_transfer_envelope(
            attempt_id="transfer-01",
            target_name=TARGET["name"],
            native_exec_receipt_bytes=receipt,
            native_exec_stdout_bytes=native_stdout,
            native_exec_stderr_bytes=b"",
            intent_bytes=O.canonical_json(unrelated_intent),
            template_bytes=template,
            expected_template_sha256=O.sha256_bytes(template),
            script_bytes=unrelated,
            payload_bytes=encoded.encode("ascii"),
            expected_packet_sha256=result["packet_sha256"],
            expected_payload_sha256=result["payload_sha256"],
            artifact_readback_bytes=readback,
            result=result,
        )
