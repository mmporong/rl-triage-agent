"""V03 영상 선택 입력 계약의 CPU 회귀."""
# pyright: reportMissingImports=false
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "src")]
import selection_comparison_controller_v1 as C
import vision_evidence_contract_v1 as V
import vision_selection_contract_v1 as S

VIEW_SHA = C.V.file_digest(C.V.PROTOCOL)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepared_case(tmp_path: Path) -> tuple[Path, dict]:
    source = tmp_path / "source"
    source.mkdir(parents=True)
    video, checkpoint, trace = source / "play.mp4", source / "policy.pt", source / "trace.json"
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
        "testsrc=duration=6:size=64x64:rate=2", "-c:v", "mpeg4", "-y", str(video),
    ], check=True, capture_output=True)
    checkpoint.write_bytes(b"vision-selection-checkpoint")
    timestamps = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    trace.write_text(json.dumps({
        "schema": V.MEASUREMENT_SCHEMA,
        "play_run_id": "play-contract-1",
        "video_sha256": _sha(video),
        "checkpoint_sha256": _sha(checkpoint),
        "command_name": "base_velocity",
        "coordinate_frames": {"command": "body", "motion": "world"},
        "samples": [{
            "timestamp_s": timestamp,
            "command": {"linear_x_mps": 1.0, "linear_y_mps": 0.0, "yaw_rate_rps": None},
            "motion": {"linear_x_mps": index / 10, "linear_y_mps": 0.0, "yaw_rate_rps": None},
            "pose": {"base_height_m": 0.31, "up_axis_z": 0.99},
        } for index, timestamp in enumerate(timestamps, 1)],
    }), encoding="utf-8")
    spec = {
        "schema": V.PREPARE_SCHEMA,
        "play_run_id": "play-contract-1",
        "source_lineage_id": "lineage-contract-1",
        "parent_source_lineage_id": None,
        "video": {"path": str(video), "logical_id": "clips/play.mp4", "play_run_id": "play-contract-1"},
        "checkpoint": {"path": str(checkpoint), "logical_id": "models/policy.pt",
                       "play_run_id": "play-contract-1"},
        "frames": [{"timestamp_s": timestamp, "play_run_id": "play-contract-1",
                    "video_sha256": _sha(video)} for timestamp in timestamps],
        "measurement": {"path": str(trace), "logical_id": "logs/trace.json",
                        "play_run_id": "play-contract-1"},
        "play": {"task": "Isaac-Velocity-Flat-Unitree-Go2-v0", "camera": "follow_camera",
                 "command_name": "base_velocity", "command_frame": "body", "motion_frame": "world",
                 "settings_sha256": "a" * 64},
    }
    spec_path = source / "prepare.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    contract = V.prepare(spec_path, tmp_path / "contract")
    binding = {"contract_sha256": V.sha256(contract), "source_lineage_id": "lineage-contract-1"}
    return contract, binding


def selection_snapshot(tmp_path: Path, checkpoint_sha: str, *, arm: str = "case") -> dict:
    packet = C.S.build_packet(
        epoch=1,
        applied_corrections=[],
        prior_receipt=None,
        facts={"config:action_scale": 0.25},
        input_bindings={
            "case": {"checkpoint_sha256": checkpoint_sha, "telemetry_sha256": "2" * 64},
            "reference": {"checkpoint_sha256": "3" * 64, "telemetry_sha256": "4" * 64},
        },
        execution_budget_s=1200,
    )
    return C.V.build_snapshot(
        packet, f"vision_{arm}", "model_adaptive", VIEW_SHA,
        tmp_path / arm / "probes.ledger.jsonl",
    )


def setup_case(tmp_path: Path) -> tuple[Path, dict, dict]:
    contract, binding = prepared_case(tmp_path)
    checkpoint_sha = V.verify(contract)["provenance"]["checkpoint"]["sha256"]
    return contract, binding, selection_snapshot(tmp_path, checkpoint_sha)


def valid_content(bundle: dict, fact_id: str | None = None) -> str:
    facts = bundle["selection_view"]["evidence_facts"]
    return json.dumps({
        "action": "continue",
        "probe": bundle["selection_view"]["eligible_probes"][0],
        "fact_ids": [fact_id or next(iter(facts))],
    })


def test_three_views_share_operational_base_and_expose_only_registered_evidence(tmp_path: Path) -> None:
    contract, binding, snapshot = setup_case(tmp_path)
    bundles = {kind: S.prepare(snapshot, contract, binding, kind) for kind in S.VIEW_KINDS}
    assert len({row["operational_base_digest"] for row in bundles.values()}) == 1
    assert bundles["text_only"]["operational_base"]["registry_digest"] == C.S.M.registry_digest(C.RC.registry())
    binding_checkpoint = V.verify(contract)["provenance"]["checkpoint"]["sha256"]
    assert bundles["text_only"]["operational_base"]["case_checkpoint_sha256"] == binding_checkpoint
    assert binding_checkpoint not in json.dumps(S.messages(
        bundles["text_only"], snapshot, contract, binding,
    ))
    for key in ("candidate_categories", "remaining_candidates", "probe_catalog", "eligible_probes",
                "budget_left", "execution_budget_s"):
        assert len({json.dumps(row["operational_base"][key], sort_keys=True) for row in bundles.values()}) == 1
    text_facts = bundles["text_only"]["selection_view"]["evidence_facts"]
    video_facts = bundles["video_only"]["selection_view"]["evidence_facts"]
    combined = bundles["text_and_video"]["selection_view"]["evidence_facts"]
    common_ids = {key for key in video_facts if key.startswith("vision:play:")}
    assert common_ids and all({key: row["selection_view"]["evidence_facts"][key] for key in common_ids}
                              == {key: video_facts[key] for key in common_ids}
                              for row in bundles.values())
    text_telemetry = {key: value for key, value in text_facts.items() if not key.startswith("vision:")}
    combined_telemetry = {key: value for key, value in combined.items() if not key.startswith("vision:")}
    assert text_telemetry == combined_telemetry and text_telemetry
    assert not any(not key.startswith("vision:") for key in video_facts)
    assert video_facts and all(key.startswith("vision:") for key in video_facts)
    assert combined == {**video_facts, **text_telemetry}
    assert [key for key in video_facts if key.startswith("vision:frame:")] == [
        f"vision:frame:f{i:02d}" for i in range(1, 7)
    ]
    assert not any(key.startswith("vision:frame:") for key in text_facts)
    counts = [{key: bundle["evidence_accounting"][key] for key in (
        "sample_count", "measurement_source_count", "measurement_non_null_count", "measurement_null_count",
    )} for bundle in bundles.values()]
    assert counts == [counts[0]] * 3
    assert counts[0] == {"sample_count": 6, "measurement_source_count": 1,
                         "measurement_non_null_count": 36, "measurement_null_count": 12}
    assert all(bundle["evidence_accounting"]["independent_confirmation_sources_added"] == 0
               for bundle in bundles.values())
    play_contexts = [bundle["selection_view"]["play_context"] for bundle in bundles.values()]
    assert play_contexts == [play_contexts[0]] * 3
    assert play_contexts[0] == {
        "command_name": "base_velocity",
        "coordinate_frames": {"command": "body", "motion": "world"},
        "sample_timestamps_s": [0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
    }


def test_payload_has_six_real_images_only_in_video_views_and_leaks_no_provenance(tmp_path: Path) -> None:
    contract, binding, snapshot = setup_case(tmp_path)
    for kind, expected_images in (("text_only", 0), ("video_only", 6), ("text_and_video", 6)):
        bundle = S.prepare(snapshot, contract, binding, kind)
        messages = S.messages(bundle, snapshot, contract, binding)
        encoded = json.dumps(messages, ensure_ascii=False).lower()
        images = [part for part in messages[1]["content"] if part["type"] == "image_url"]
        assert len(images) == expected_images
        for forbidden in ("lineage-contract", "play-contract", "policy.pt", "checkpoint_sha256",
                          "fixed_eval", "healthy", "unhealthy", '"arm"', '"seed"'):
            assert forbidden not in encoded
        assert "walking" not in encoded
        assert '"args"' not in encoded and '"expect"' not in encoded and "approval_granted" not in encoded
        assert '"view_kind"' not in encoded and '"evidence_accounting"' not in encoded
        assert '"source"' not in encoded and '"metadata"' not in encoded


def test_exact_choice_accepts_visible_frame_fact_and_rejects_fake_fact_or_fields(tmp_path: Path) -> None:
    contract, binding, snapshot = setup_case(tmp_path)
    bundle = S.prepare(snapshot, contract, binding, "video_only")
    decision = S.validate_choice(bundle, valid_content(bundle, "vision:frame:f01"))
    projected = S.project_choice(bundle, decision, snapshot, contract, binding)
    assert projected["fact_ids"] == ["vision:frame:f01"]
    assert projected["approval_granted"] is False and projected["selected_by"] == "model"
    fake = json.loads(valid_content(bundle))
    fake["fact_ids"] = ["vision:frame:f99"]
    with pytest.raises(S.ChoiceError, match="S_FACT_UNKNOWN"):
        S.validate_choice(bundle, json.dumps(fake))
    fake = json.loads(valid_content(bundle))
    fake["args"] = {"epochs": 99}
    with pytest.raises(S.ChoiceError, match="F_FIELDS"):
        S.validate_choice(bundle, json.dumps(fake))
    with pytest.raises(S.ChoiceError, match="F_JSON_DUPLICATE_KEY"):
        S.validate_choice(bundle, '{"action":"hold","action":"continue","probe":null,"fact_ids":["vision:frame:f01"]}')

    text = S.prepare(snapshot, contract, binding, "text_only")
    play_fact = "vision:play:f01:timestamp_s"
    assert S.validate_choice(text, valid_content(text, play_fact))["fact_ids"] == [play_fact]
    with pytest.raises(S.ChoiceError, match="S_FACT_UNKNOWN"):
        S.validate_choice(text, valid_content(text, "vision:frame:f01"))


def test_contract_checkpoint_lineage_and_bound_bytes_are_checked_before_messages(tmp_path: Path) -> None:
    contract, binding, snapshot = setup_case(tmp_path)
    wrong = deepcopy(binding)
    wrong["source_lineage_id"] = "another-lineage"
    with pytest.raises(ValueError, match="lineage"):
        S.prepare(snapshot, contract, wrong, "video_only")
    wrong_snapshot = selection_snapshot(tmp_path, "f" * 64, arm="wrong")
    with pytest.raises(ValueError, match="checkpoint"):
        S.prepare(wrong_snapshot, contract, binding, "video_only")
    frame = contract.parent / "artifacts/frames/frame_01.jpg"
    frame.write_bytes(frame.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="JPEG|결합"):
        S.prepare(snapshot, contract, binding, "video_only")


def test_bundle_or_measurement_tampering_is_rejected(tmp_path: Path) -> None:
    contract, binding, snapshot = setup_case(tmp_path)
    bundle = S.prepare(snapshot, contract, binding, "text_and_video")
    changed = deepcopy(bundle)
    changed["selection_view"]["evidence_facts"]["vision:play:f01:motion:linear_x_mps"] = 99.0
    with pytest.raises(ValueError, match="bundle"):
        S.verify(changed, snapshot, contract, binding)
    original_contract = contract.read_bytes()
    payload = json.loads(original_contract)
    payload["frames"][0]["timestamp_s"] = 0.6
    contract.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="시각|결합|SHA"):
        S.messages(bundle, snapshot, contract, binding)
    contract.write_bytes(original_contract)
    trace = contract.parent / "artifacts/measurements/trace.json"
    payload = json.loads(trace.read_text(encoding="utf-8"))
    payload["samples"][0]["motion"]["linear_x_mps"] = 99.0
    trace.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="결합"):
        S.messages(bundle, snapshot, contract, binding)
