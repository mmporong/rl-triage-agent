from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from evals import grounded_vision_observer_v1 as G
from evals import model_decision_journal_v1 as J
from evals import vision_evidence_contract_v1 as V


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _contract(tmp_path: Path) -> Path:
    source = tmp_path / "input"
    source.mkdir()
    video = source / "source.mp4"
    checkpoint = source / "policy.pt"
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
            "testsrc=duration=6:size=64x64:rate=2", "-c:v", "mpeg4", "-y", str(video),
        ],
        check=True,
        capture_output=True,
    )
    checkpoint.write_bytes(b"checkpoint")
    run_id = "play-fixture-one"
    timestamps = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    frames = []
    for index, timestamp in enumerate(timestamps, 1):
        frames.append({
            "timestamp_s": timestamp,
            "play_run_id": run_id, "video_sha256": _sha(video.read_bytes()),
        })
    spec = {
        "schema": V.PREPARE_SCHEMA,
        "play_run_id": run_id,
        "source_lineage_id": "lineage-fixture-one",
        "parent_source_lineage_id": None,
        "video": {"path": str(video), "logical_id": "fixture/source.mp4", "play_run_id": run_id},
        "checkpoint": {
            "path": str(checkpoint), "logical_id": "fixture/policy.pt", "play_run_id": run_id,
        },
        "frames": frames,
        "measurement": None,
        "play": {
            "task": "Isaac-Velocity-Flat-Unitree-Go2-v0", "camera": "follow_camera",
            "command_name": "base_velocity", "command_frame": "body", "motion_frame": "world",
            "settings_sha256": "a" * 64,
        },
    }
    spec_path = source / "spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    return V.prepare(spec_path, tmp_path / "contract")


def _content(contract: Path, *, walking: str = "yes", frame: str = "f01",
             timestamp: float | None = None) -> str:
    frames = {row["frame_id"]: row["timestamp_s"] for row in V.verify(contract)["frames"]}
    cited_timestamp = frames[frame] if timestamp is None and frame in frames else timestamp
    return json.dumps({
        "walking": walking,
        "in_place": "no",
        "falling": "no",
        "evidence": {
            "walking": [{"frame_id": frame, "timestamp_s": cited_timestamp}],
            "in_place": [{"frame_id": "f02", "timestamp_s": frames["f02"]}],
            "falling": [{"frame_id": "f03", "timestamp_s": frames["f03"]}],
        },
    })


def _provider(content: str, *, model: str = "private/nvidia/nemotron-3.5-super-120b-a12b") -> dict:
    return {
        "response_id": "fixture-response", "provider_model": model, "finish_reason": "stop",
        "content": content, "input_tokens": 100, "output_tokens": 40, "total_tokens": 140,
        "wall_s": 0.2,
    }


def _binding(contract: Path) -> dict:
    return G.binding_template(
        contract, run_id="observation-fixture", source_anchor="a" * 40,
        protocol_sha256="b" * 64,
    )


class Clock:
    def __init__(self) -> None:
        self.value = datetime(2026, 10, 25, tzinfo=timezone.utc)  # noqa: UP017 - Python 3.10 호환

    def __call__(self) -> datetime:
        return self.value


def test_request_contains_real_images_but_not_private_provenance(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    view = G.request_view(contract)
    encoded = json.dumps(view, ensure_ascii=False)
    assert encoded.count("data:image/jpeg;base64,") == 6
    assert "play-fixture-one" not in encoded
    assert "lineage-fixture-one" not in encoded
    assert "policy.pt" not in encoded
    assert view["evidence_accounting"]["model_interpretation_source_count"] == 0
    assert G.REQUEST_PARAMS == {
        "model": G.MODEL, "temperature": 0, "max_tokens": 1024, "stream": False,
        "extra_body": {"reasoning_effort": "none"},
    }


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        (lambda value: value.update(extra=True), "F_FIELDS"),
        (lambda value: value.update(walking=True), "F_STATE"),
        (lambda value: value.update(walking=float("nan")), "F_JSON"),
        (lambda value: value["evidence"].update(walking=[]), "F_EVIDENCE"),
        (lambda value: value["evidence"]["walking"][0].update(note="invented"), "F_CITATION"),
        (lambda value: value["evidence"]["walking"][0].update(frame_id="f99"), "S_FRAME_UNKNOWN"),
        (lambda value: value["evidence"]["walking"][0].update(timestamp_s=0.6), "S_FRAME_UNKNOWN"),
    ],
)
def test_validator_rejects_extra_types_nan_and_invented_citations(tmp_path: Path, mutation, code: str) -> None:
    contract = _contract(tmp_path)
    value = json.loads(_content(contract))
    mutation(value)
    payload = json.dumps(value, allow_nan=True)
    with pytest.raises(ValueError, match=code):
        G.validate_observation(contract, payload)


def test_valid_observation_is_model_interpretation_and_never_decision(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    result = G.run(
        tmp_path / "run", contract, _binding(contract), lambda *_: _provider(_content(contract)),
        now_utc=Clock(),
    )
    assert result["status"] == "observation_ready"
    engine = result["engine_result"]
    assert engine["observation"]["walking"] == "yes"
    assert engine["provenance"] == {
        "evidence_type": "model_interpretation",
        "used_for_decision": False,
        "contract_sha256": V.sha256(contract),
        "requested_model": G.MODEL,
        "provider_models_raw": ["private/nvidia/nemotron-3.5-super-120b-a12b"],
        "provider_identity_policy": "registered_endpoint_alias_v3_not_weights_attestation",
    }
    assert result["api"] == {"reserved": 1, "pending": 0, "remaining": 5}


def test_one_semantic_feedback_then_hold_and_completed_response_replays(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    calls = []

    def provider(messages, _params):
        calls.append(messages)
        return _provider('{"walking":"yes"}')

    first = G.run(tmp_path / "run", contract, _binding(contract), provider, now_utc=Clock())
    assert first["status"] == "terminal_hold"
    assert first["engine_result"]["reason"] == "observation_invalid"
    assert len(calls) == 2
    assert "F_FIELDS" in calls[1][-1]["content"]
    replay = G.run(
        tmp_path / "run", contract, _binding(contract),
        lambda *_: (_ for _ in ()).throw(AssertionError("cached response를 다시 호출함")),
        now_utc=Clock(),
    )
    assert replay == first


def test_unregistered_provider_alias_and_api_cap_fail_closed(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    wrong_alias = G.run(
        tmp_path / "alias", contract, _binding(contract),
        lambda *_: _provider(_content(contract), model="private/unregistered-model"),
        now_utc=Clock(),
    )
    assert wrong_alias["status"] == "terminal_hold"
    assert wrong_alias["engine_result"]["reason"] == "transport_unknown"

    calls = []

    def invalid(*_args):
        calls.append(1)
        return _provider('{"walking":"yes"}')

    capped = G.run(
        tmp_path / "capped", contract, _binding(contract), invalid,
        now_utc=Clock(), api_cap=1,
    )
    assert capped["status"] == "terminal_hold"
    assert capped["engine_result"]["stop_reason"] == "budget_exhausted"
    assert calls == [1]


def test_infra_pause_pending_fail_closed_and_global_cap(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    clock = Clock()
    calls = []

    def provider(*_args):
        calls.append(1)
        if len(calls) == 1:
            return {"status_code": 500, "infra_error": "InternalServerError"}
        return _provider(_content(contract))

    first = G.run(tmp_path / "run", contract, _binding(contract), provider, now_utc=clock)
    assert first["status"] == "paused_infra" and len(calls) == 1
    clock.value += timedelta(seconds=30)
    second = G.run(tmp_path / "run", contract, _binding(contract), provider, now_utc=clock)
    assert second["status"] == "observation_ready" and len(calls) == 2

    pending_root = tmp_path / "pending"
    journal = J.DecisionJournal(
        pending_root / "decision", pending_root / "api", _binding(contract), G.REQUEST_PARAMS,
        api_cap=1,
    )
    J._write_new(pending_root / "api/reservations/0001.json", {
        "schema": "model_decision_reservation_v1", "index": 1,
        "manifest_sha256": journal.manifest_sha256, "binding_sha256": journal.binding_sha256,
        "run_id": "observation-fixture", "ordinal": 1,
        "messages_sha256": J.digest(G.request_view(contract)["messages"]),
        "params_sha256": journal.params_sha256, "reserved_at": clock().isoformat(),
    })
    blocked = G.run(
        pending_root, contract, _binding(contract),
        lambda *_: (_ for _ in ()).throw(AssertionError("pending에서 호출됨")),
        now_utc=clock, api_cap=1,
    )
    assert blocked["status"] == "blocked_pending"


def test_binding_rejects_changed_checkpoint_snapshot_and_messages(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    for key in ("checkpoint_sha256", "snapshot_sha256", "initial_messages_sha256"):
        changed = _binding(contract)
        changed[key] = "f" * 64
        with pytest.raises(ValueError, match=key):
            G.checked_binding(contract, changed)
