from __future__ import annotations

import copy
import hashlib
import json
import struct
import zlib
from pathlib import Path

import pytest

from evals import visual_localization_experiment_v2 as V


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _png(path: Path, rgb: tuple[int, int, int]) -> None:
    raw = b"".join(b"\x00" + bytes(rgb) * 640 for _ in range(480))

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 640, 480, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def _fixture(tmp_path: Path):
    root = tmp_path / "input"
    cases = []
    for number, case_id in enumerate(V.V1.CASE_IDS, 1):
        case_dir = root / case_id
        samples, frames = [], []
        for index, frame_id in enumerate(V.V1.FRAME_IDS, 1):
            image = case_dir / "frames" / f"{frame_id}.png"
            _png(image, (number * 30, index * 20, 10))
            samples.append({"frame_id": frame_id, "timestamp_s": index,
                            "root_position_world_m": [float(index), 0.0, 0.4],
                            "expected_root_pixel": [320.0, 278.0],
                            "camera_eye_world_m": [float(index) - 3.0, 0.0, 2.0]})
            frames.append({"frame_id": frame_id, "file": f"frames/{frame_id}.png",
                           "sha256": hashlib.sha256(image.read_bytes()).hexdigest(), "bytes": image.stat().st_size})
        packet = {"schema": "visual_integrity_case_v1", "image_size": [640, 480],
                  "camera_mode": "follows_simulated_root_with_fixed_offset", "samples": samples, "frames": frames,
                  "numeric_binding_checks": {"frames": 625, "camera_trace_eye_max_error_m": 1e-12,
                                             "timestamps_exact": True}}
        _write(case_dir / "packet.json", packet)
        cases.append({"case_id": case_id, "packet": f"{case_id}/packet.json",
                      "packet_sha256": hashlib.sha256((case_dir / "packet.json").read_bytes()).hexdigest()})
    catalog = root / "catalog.json"
    _write(catalog, {"schema": "visual_integrity_catalog_v1", "cases": cases})
    labels = tmp_path / "labels.json"
    _write(labels, {"schema": "visual_integrity_labels_v1", "labels": {
        "clip-01": {"expected_status": "inconsistent", "lineage_id": "a", "source_refs": {"raw": "1" * 64}},
        "clip-02": {"expected_status": "inconsistent", "lineage_id": "b", "source_refs": {"raw": "2" * 64}},
        "clip-03": {"expected_status": "consistent", "lineage_id": "a", "source_refs": {"raw": "3" * 64}},
        "clip-04": {"expected_status": "consistent", "lineage_id": "c", "source_refs": {"raw": "4" * 64}},
    }})
    source_map = {relative: V.V1._lf_sha_bytes((V.ROOT / relative).read_bytes()) for relative in V.SOURCE_FILES}
    protocol = tmp_path / "protocol.json"
    _write(protocol, {"schema": "visual_localization_experiment_protocol_v2", "status": "registered",
                      "source_anchor": "0" * 40, "source_sha256_lf": source_map,
                      "catalog_sha256": hashlib.sha256(catalog.read_bytes()).hexdigest(),
                      "labels_sha256": hashlib.sha256(labels.read_bytes()).hexdigest(), "model": V.MODEL,
                      "request_params": copy.deepcopy(V.REQUEST_PARAMS),
                      "budget": {"cases": 4, "timestamps_s": [3, 6], "request_cap": 8,
                                 "automatic_retries": 0},
                      "scope": "development_blind_visual_localization_only", "image_detail": None})
    return catalog, labels, protocol


def _localization(near: bool, frame_id: str = "f03") -> dict:
    bbox = [450, 529, 550, 629] if near else [100, 100, 200, 200]
    return {"frame_id": frame_id, "visibility": "visible", "bbox_1000": bbox}


def test_messages_are_blind_and_localization_validator_rejects_bbox_counterexamples(tmp_path):
    catalog, _, _ = _fixture(tmp_path)
    messages = V.messages_for_case(catalog.parent / "clip-01", "f03")
    text = json.dumps(messages)
    for forbidden in ("expected_root_pixel", "root_position_world_m", "camera_eye_world_m", "clip-01"):
        assert forbidden not in text
    assert sum(block.get("type") == "image_url" for block in messages[1]["content"]) == 1
    image = next(block["image_url"] for block in messages[1]["content"] if block.get("type") == "image_url")
    assert set(image) == {"url"}
    assert V.validate_localization(json.dumps(_localization(True)))["valid"]
    mutations = []
    value = _localization(True); value["frame_id"] = "f02"; mutations.append(value)
    value = _localization(True); value["bbox_1000"] = [False, 1, 2, 3]; mutations.append(value)
    value = _localization(True); value["bbox_1000"] = [500, 1, 400, 3]; mutations.append(value)
    value = _localization(True); value["bbox_1000"] = [1, 2, 3, 1001]; mutations.append(value)
    value = _localization(True); value["visibility"] = "absent"; mutations.append(value)
    value = _localization(True); value["extra"] = 1; mutations.append(value)
    for mutation in mutations:
        assert not V.validate_localization(json.dumps(mutation))["valid"]


def test_host_geometry_thresholds_and_absence():
    packet = {"samples": [{"frame_id": frame_id, "expected_root_pixel": [320.0, 278.0]}
                          for frame_id in V.V1.FRAME_IDS]}
    assert V.derive_host_decision(packet, _localization(True))["status"] == "consistent"
    assert V.derive_host_decision(packet, _localization(False))["status"] == "inconsistent"
    absent = {"frame_id": "f03", "visibility": "absent", "bbox_1000": None}
    assert V.derive_host_decision(packet, absent)["gate"] == "quarantine_visual_evidence"
    unclear = {"frame_id": "f03", "visibility": "unclear", "bbox_1000": None}
    assert V.derive_host_decision(packet, unclear)["gate"] == "HOLD_visual_evidence"


def test_fake_eight_calls_score_and_reviewed_payload_gate(tmp_path, monkeypatch):
    catalog, labels, protocol = _fixture(tmp_path)
    monkeypatch.setattr(V, "_verify_source_guard", lambda value: None)
    run = tmp_path / "run"
    plan = V.prepare(catalog, protocol, run)
    calls = []

    def sender(messages, params):
        calls.append((messages, params))
        metadata = json.loads(messages[1]["content"][0]["text"])
        answer = _localization(near=len(calls) > 4, frame_id=metadata["frame_id"])
        return {"response_id": f"r{len(calls)}", "provider_model": V.V1.ENDPOINT.PRIVATE_CANDIDATE_ALIAS,
                "content": json.dumps(answer), "finish_reason": "stop", "input_tokens": 10,
                "output_tokens": 5, "total_tokens": 15, "wall_s": 0.1}

    protocol_sha = hashlib.sha256(protocol.read_bytes()).hexdigest()
    V.execute(run, protocol, protocol_sha, sender=sender)
    assert len(calls) == 8
    records = [json.loads(path.read_text()) for path in sorted((run / "records").glob("*.json"))]
    assert sum(row["gate"] == "quarantine_visual_evidence" for row in records) == 4
    assert sum(row["gate"] == "reviewed_consistent" for row in records) == 4
    result = V.score(run, protocol, labels, tmp_path / "score.json")
    assert result["outcome"] == "synthetic_cpu_only" and result["primary_criterion"] == {
        "valid_localizations": {"value": 8, "required": 8},
        "stale_host_quarantined": {"value": 4, "required": 4},
        "fresh_host_consistent": {"value": 4, "required": 4},
    }
    assert result["case_gates"] == {
        "clip-01": "quarantine_visual_evidence", "clip-02": "quarantine_visual_evidence",
        "clip-03": "reviewed_consistent", "clip-04": "reviewed_consistent",
    }
    assert result["intervention_bundle"]["individual_causal_attribution"] is False
    consumer_calls = []
    assert V.reviewed_payload(run, protocol, "clip-01", lambda value: consumer_calls.append(value)) is None
    assert consumer_calls == []
    assert V.reviewed_payload(run, protocol, "clip-03", lambda value: consumer_calls.append(value) or "ok") == "ok"
    assert len(consumer_calls) == 1
    blocks = consumer_calls[0][0]["content"]
    assert sum(block["type"] == "image_url" for block in blocks) == 2
    metadata = [json.loads(block["text"]) for block in blocks if block["type"] == "text"]
    assert [row["frame_id"] for row in metadata] == ["f03", "f06"]
    assert all("expected_root_pixel" not in row for row in metadata)
    assert [slot["slot_id"] for slot in plan["slots"][:4]] == [
        "clip-01-f03", "clip-01-f06", "clip-02-f03", "clip-02-f06"
    ]
    mixed = [copy.deepcopy(records[4]), copy.deepcopy(records[5])]
    mixed[1]["derived_host_decision"]["status"] = "unclear"
    assert V._case_gate(mixed) == "HOLD_visual_evidence"


def test_source_plan_and_ledger_tampering_fail_closed(tmp_path, monkeypatch):
    catalog, labels, protocol = _fixture(tmp_path)
    protocol_value = json.loads(protocol.read_text())
    missing_source = copy.deepcopy(protocol_value)
    missing_source["source_sha256_lf"].pop(next(iter(missing_source["source_sha256_lf"])))
    with pytest.raises(ValueError, match="source map"):
        V._validate_protocol_shape(missing_source)
    wrong_detail = copy.deepcopy(protocol_value); wrong_detail["image_detail"] = "high"
    with pytest.raises(ValueError, match="고정 조건"):
        V._validate_protocol_shape(wrong_detail)
    monkeypatch.setattr(V, "_verify_source_guard", lambda value: None)
    run = tmp_path / "run"
    V.prepare(catalog, protocol, run)
    count = 0

    def sender(*_):
        nonlocal count
        count += 1
        frame_id = "f03" if count % 2 else "f06"
        return {"provider_model": V.MODEL, "content": json.dumps(_localization(True, frame_id)),
                "finish_reason": "stop", "wall_s": 0.1}

    protocol_sha = hashlib.sha256(protocol.read_bytes()).hexdigest()
    V.execute(run, protocol, protocol_sha, sender=sender)
    V.execute(run, protocol, protocol_sha, sender=sender)
    assert count == 8
    record_path = run / "records" / "clip-01-f03.json"
    record = json.loads(record_path.read_text())
    record["provider"]["content"] = json.dumps(_localization(False, record["frame_id"]))
    record["localization_validator"] = V.validate_localization(record["provider"]["content"])
    packet = V.V1.validate_packet(run / "cases" / "clip-01")
    record["derived_host_decision"] = V.derive_host_decision(packet, record["localization_validator"]["localization"])
    record["gate"] = record["derived_host_decision"]["gate"]
    record["record_sha256"] = V._digest({key: value for key, value in record.items() if key != "record_sha256"})
    _write(record_path, record)
    with pytest.raises(ValueError, match="요청 원장"):
        V.score(run, protocol, labels, tmp_path / "score-tampered.json")

    plan = json.loads((run / "plan.json").read_text())
    plan["slots"][0], plan["slots"][1] = plan["slots"][1], plan["slots"][0]
    plan["plan_sha256"] = V._digest({key: value for key, value in plan.items() if key != "plan_sha256"})
    _write(run / "plan.json", plan)
    with pytest.raises(ValueError, match="frozen"):
        V._validate_plan(run, protocol_value, protocol_sha)
