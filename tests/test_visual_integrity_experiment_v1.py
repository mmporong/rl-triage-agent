from __future__ import annotations

import copy
import hashlib
import json
import struct
import zlib
from pathlib import Path

import pytest

from evals import visual_integrity_experiment_v1 as V


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _png(path: Path, rgb: tuple[int, int, int]) -> None:
    width, height = 640, 480
    raw = b"".join(b"\x00" + bytes(rgb) * width for _ in range(height))

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)

    data = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _fixture(tmp_path: Path):
    catalog_root = tmp_path / "input"
    cases = []
    for number, case_id in enumerate(V.CASE_IDS, 1):
        case_dir = catalog_root / case_id
        samples, frames = [], []
        for index, frame_id in enumerate(V.FRAME_IDS, 1):
            frame_path = case_dir / "frames" / f"{frame_id}.png"
            _png(frame_path, (number * 30, index * 20, 10))
            samples.append({
                "frame_id": frame_id, "timestamp_s": index,
                "root_position_world_m": [float(index), 0.0, 0.4],
                "expected_root_pixel": [100.0 if number <= 2 else 300.0, 240.0],
                "camera_eye_world_m": [float(index) - 3.0, 0.0, 2.0],
            })
            frames.append({
                "frame_id": frame_id, "file": f"frames/{frame_id}.png",
                "sha256": hashlib.sha256(frame_path.read_bytes()).hexdigest(), "bytes": frame_path.stat().st_size,
            })
        packet = {
            "schema": "visual_integrity_case_v1", "image_size": [640, 480],
            "camera_mode": "follows_simulated_root_with_fixed_offset", "samples": samples, "frames": frames,
            "numeric_binding_checks": {
                "frames": 625, "camera_trace_eye_max_error_m": 1e-12, "timestamps_exact": True,
            },
        }
        _write(case_dir / "packet.json", packet)
        cases.append({
            "case_id": case_id, "packet": f"{case_id}/packet.json",
            "packet_sha256": hashlib.sha256((case_dir / "packet.json").read_bytes()).hexdigest(),
        })
    catalog_path = catalog_root / "catalog.json"
    _write(catalog_path, {"schema": "visual_integrity_catalog_v1", "cases": cases})
    labels = {
        "schema": "visual_integrity_labels_v1",
        "labels": {
            "clip-01": {"expected_status": "inconsistent", "lineage_id": "a", "source_refs": {"raw": "1" * 64}},
            "clip-02": {"expected_status": "inconsistent", "lineage_id": "b", "source_refs": {"raw": "2" * 64}},
            "clip-03": {"expected_status": "consistent", "lineage_id": "a", "source_refs": {"raw": "3" * 64}},
            "clip-04": {"expected_status": "consistent", "lineage_id": "c", "source_refs": {"raw": "4" * 64}},
        },
    }
    labels_path = tmp_path / "labels.json"
    _write(labels_path, labels)
    source_map = {relative: V._lf_sha_bytes((V.ROOT / relative).read_bytes()) for relative in V.SOURCE_FILES}
    protocol = {
        "schema": "visual_integrity_experiment_protocol_v1", "status": "registered",
        "source_anchor": "0" * 40, "source_sha256_lf": source_map,
        "catalog_sha256": hashlib.sha256(catalog_path.read_bytes()).hexdigest(),
        "labels_sha256": hashlib.sha256(labels_path.read_bytes()).hexdigest(),
        "model": V.MODEL, "request_params": copy.deepcopy(V.REQUEST_PARAMS),
        "budget": {"cases": 4, "repeats": 2, "request_cap": 8, "automatic_retries": 0},
        "scope": "development_render_integrity_only", "prediction": "fixed before provider calls",
    }
    protocol_path = tmp_path / "protocol.json"
    _write(protocol_path, protocol)
    return catalog_path, labels_path, protocol_path


def test_packet_messages_are_blind_and_mutations_fail(tmp_path):
    catalog, _, _ = _fixture(tmp_path)
    case = catalog.parent / "clip-01"
    messages = V.messages_for_case(case)
    text = json.dumps(messages)
    assert "clip-01" not in text and "expected_status" not in text and "checkpoint" not in text
    assert sum(block.get("type") == "image_url" for block in messages[1]["content"]) == 6

    packet_path = case / "packet.json"
    original = json.loads(packet_path.read_text())
    mutations = []
    value = copy.deepcopy(original); value["label"] = "stale"; mutations.append(value)
    value = copy.deepcopy(original); value["samples"][0]["timestamp_s"] = 2; mutations.append(value)
    value = copy.deepcopy(original); value["samples"][0]["timestamp_s"] = True; mutations.append(value)
    value = copy.deepcopy(original); value["samples"][0]["frame_id"] = "f02"; mutations.append(value)
    value = copy.deepcopy(original); value["samples"][0]["root_position_world_m"] = [1, 2]; mutations.append(value)
    value = copy.deepcopy(original); value["frames"][0]["sha256"] = "0" * 64; mutations.append(value)
    value = copy.deepcopy(original); value["frames"][0], value["frames"][1] = value["frames"][1], value["frames"][0]; mutations.append(value)
    value = copy.deepcopy(original); value["numeric_binding_checks"]["camera_trace_eye_max_error_m"] = 1e-7; mutations.append(value)
    for mutation in mutations:
        _write(packet_path, mutation)
        with pytest.raises(ValueError):
            V.validate_packet(case)
    _write(packet_path, original)
    assert V.validate_packet(case)["schema"] == "visual_integrity_case_v1"


def test_response_validator_separates_format_and_semantics():
    valid = {"status": "consistent", "frame_ids": ["f01", "f02"],
             "reason_code": "robot_matches_projection", "explanation": "The body matches the projection."}
    assert V.validate_response(json.dumps(valid))["valid"]
    bad_format = copy.deepcopy(valid); bad_format["extra"] = 1
    verdict = V.validate_response(json.dumps(bad_format))
    assert verdict["format_reasons"] == ["response_fields_mismatch"] and not verdict["semantic_reasons"]
    bad_semantic = copy.deepcopy(valid); bad_semantic["frame_ids"] = ["f99"]
    verdict = V.validate_response(json.dumps(bad_semantic))
    assert not verdict["format_reasons"] and "unknown_frame_id" in verdict["semantic_reasons"]
    line_ref = copy.deepcopy(valid); line_ref["explanation"] = "See line 12."
    assert "line_reference_forbidden" in V.validate_response(json.dumps(line_ref))["semantic_reasons"]
    for key in ("status", "reason_code"):
        wrong_type = copy.deepcopy(valid); wrong_type[key] = []
        verdict = V.validate_response(json.dumps(wrong_type))
        assert verdict["format_reasons"] and not verdict["valid"]


def test_prepare_execute_eight_fake_calls_score_and_no_duplicate_send(tmp_path, monkeypatch):
    catalog, labels, protocol = _fixture(tmp_path)
    monkeypatch.setattr(V, "_verify_source_guard", lambda value: None)
    run = tmp_path / "run"
    plan = V.prepare(catalog, protocol, run)
    assert len(plan["slots"]) == 8
    assert (run / "cases" / "catalog.json").read_bytes() == catalog.read_bytes()
    assert plan["slots"][0]["messages_sha256"] == plan["slots"][1]["messages_sha256"]
    assert plan["slots"][0]["ledger_packet_sha256"] != plan["slots"][1]["ledger_packet_sha256"]
    calls = []

    def fake(messages, params):
        calls.append((messages, params))
        sample = json.loads(messages[1]["content"][1]["text"])
        if sample["expected_root_pixel"][0] < 200:
            answer = {"status": "inconsistent", "frame_ids": ["f01"],
                      "reason_code": "robot_displaced_from_state", "explanation": "Body is far from projection."}
        else:
            answer = {"status": "consistent", "frame_ids": ["f01", "f02"],
                      "reason_code": "robot_matches_projection", "explanation": "Body remains near projection."}
        return {"response_id": f"r{len(calls)}", "provider_model": V.ENDPOINT.PRIVATE_CANDIDATE_ALIAS,
                "content": json.dumps(answer), "finish_reason": "stop", "input_tokens": 10,
                "output_tokens": 5, "total_tokens": 15, "wall_s": 0.1}

    protocol_sha = hashlib.sha256(protocol.read_bytes()).hexdigest()
    result = V.execute(run, protocol, protocol_sha, sender=fake)
    assert result["execution_kind"] == "cpu_fake" and result["live_evidence"] is False
    assert len(calls) == 8
    records = [json.loads(path.read_text()) for path in sorted((run / "records").glob("*.json"))]
    assert sum(row["gate"] == "quarantine_visual_evidence" for row in records) == 4
    assert sum(row["gate"] == "reviewed_consistent" for row in records) == 4
    tampered = copy.deepcopy(records[0])
    tampered["provider"]["provider_model_raw"] = "not-registered"
    tampered["record_sha256"] = V._digest({key: value for key, value in tampered.items() if key != "record_sha256"})
    with pytest.raises(ValueError, match="provider alias"):
        V._validate_record(tampered, plan["slots"][0])
    V.execute(run, protocol, protocol_sha, sender=fake)
    assert len(calls) == 8
    out = tmp_path / "score.json"
    score = V.score(run, protocol, labels, out)
    assert score["outcome"] == "synthetic_cpu_only"
    assert score["execution_kind"] == "cpu_fake" and score["live_evidence"] is False
    assert score["coverage"] == {"unique_cases": 4, "checkpoint_lineages": 3, "repeated_calls": 8,
                                  "scope": "development_render_integrity_only"}
    assert score["used_for_behavior_or_recovery_decision"] is False

    record_path = run / "records" / "clip-01-r1.json"
    record = json.loads(record_path.read_text())
    changed = {"status": "inconsistent", "frame_ids": ["f02"],
               "reason_code": "robot_not_visible", "explanation": "Robot is not visible."}
    record["provider"]["content"] = json.dumps(changed)
    record["validator"] = V.validate_response(record["provider"]["content"])
    record["gate"] = V._gate(record["validator"])
    record["record_sha256"] = V._digest({key: value for key, value in record.items() if key != "record_sha256"})
    _write(record_path, record)
    with pytest.raises(ValueError, match="요청 원장"):
        V.score(run, protocol, labels, tmp_path / "score-tampered.json")


def test_copied_catalog_and_recomputed_plan_prevent_swaps(tmp_path, monkeypatch):
    catalog, _, protocol = _fixture(tmp_path)
    monkeypatch.setattr(V, "_verify_source_guard", lambda value: None)
    run = tmp_path / "run"
    plan = V.prepare(catalog, protocol, run)
    protocol_value = json.loads(protocol.read_text())
    plan["cases"][0], plan["cases"][1] = plan["cases"][1], plan["cases"][0]
    plan["plan_sha256"] = V._digest({key: value for key, value in plan.items() if key != "plan_sha256"})
    _write(run / "plan.json", plan)
    with pytest.raises(ValueError, match="frozen catalog"):
        V._validate_plan(run, protocol_value, hashlib.sha256(protocol.read_bytes()).hexdigest())

    copied = json.loads((run / "cases" / "catalog.json").read_text())
    copied["cases"][0]["packet_sha256"] = "0" * 64
    _write(run / "cases" / "catalog.json", copied)
    with pytest.raises(ValueError, match="catalog SHA"):
        V._validate_plan(run, protocol_value, hashlib.sha256(protocol.read_bytes()).hexdigest())


def test_alias_hold_pending_and_missing_record_fail_closed(tmp_path, monkeypatch):
    catalog, _, protocol = _fixture(tmp_path)
    monkeypatch.setattr(V, "_verify_source_guard", lambda value: None)
    run = tmp_path / "run"
    plan = V.prepare(catalog, protocol, run)
    protocol_sha = hashlib.sha256(protocol.read_bytes()).hexdigest()
    pending = V.AUX.RequestLedger(run / "api")
    slot = plan["slots"][0]
    pending.reserve(slot["ledger_packet_sha256"], slot["slot_id"], 1, "a" * 64)
    with pytest.raises(V.AUX.LedgerStop):
        V.execute(run, protocol, protocol_sha, sender=lambda *_: {})

    run2 = tmp_path / "run2"
    V.prepare(catalog, protocol, run2)
    calls = 0

    def wrong_alias(*_):
        nonlocal calls
        calls += 1
        return {"provider_model": "not-registered", "content": "{}", "wall_s": 0.1}

    V.execute(run2, protocol, protocol_sha, sender=wrong_alias)
    records = [json.loads(path.read_text()) for path in sorted((run2 / "records").glob("*.json"))]
    assert calls == 8 and all(row["gate"] == "HOLD_visual_evidence" for row in records)
    (run2 / "records" / "clip-01-r1.json").unlink()
    with pytest.raises(V.AUX.LedgerStop):
        V.execute(run2, protocol, protocol_sha, sender=wrong_alias)
    assert calls == 8


def test_finish_reason_length_is_transport_hold(tmp_path, monkeypatch):
    catalog, _, protocol = _fixture(tmp_path)
    monkeypatch.setattr(V, "_verify_source_guard", lambda value: None)
    run = tmp_path / "run"
    V.prepare(catalog, protocol, run)

    def truncated(*_):
        answer = {"status": "consistent", "frame_ids": ["f01", "f02"],
                  "reason_code": "robot_matches_projection", "explanation": "Body matches."}
        return {"provider_model": V.MODEL, "content": json.dumps(answer), "finish_reason": "length", "wall_s": 0.1}

    V.execute(run, protocol, hashlib.sha256(protocol.read_bytes()).hexdigest(), sender=truncated)
    records = [json.loads(path.read_text()) for path in sorted((run / "records").glob("*.json"))]
    assert all(not row["transport_ok"] and row["gate"] == "HOLD_visual_evidence" for row in records)
