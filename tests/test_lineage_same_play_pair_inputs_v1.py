from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest  # pyright: ignore[reportMissingImports]

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("evals", "bench", "src", "tests")]

import lineage_same_play_pair_inputs_v1 as M  # pyright: ignore[reportMissingImports]
import test_lineage_five_policy_driver_v1 as DRIVER_FIXTURE  # pyright: ignore[reportMissingImports]


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_quality_receipt_quarantines_failed_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    chosen = [{"frame_id": f"f{i}", "label": {}, "image": {}} for i in range(1, 7)]
    monkeypatch.setattr(M.QUALITY, "_image_bytes", lambda _blocks: b"frame")
    values = iter([20.0, 20.0, 9.0, 20.0, 20.0, 20.0])
    monkeypatch.setattr(M.QUALITY, "laplacian_variance", lambda _raw: next(values))
    receipt = M._quality_receipt(chosen)
    assert receipt["visual_outcome"] == "unknown"
    assert receipt["frame_refs"] == [] and receipt["image_count"] == 0


def test_prepare_pair_preserves_numeric_arm_when_capture_unsupported(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    protocol = {"schema": M.SCHEMA, "case_id": "case-09", "claims": {
        "heldout": False, "recovery": False, "video_benefit": False,
        "model_improvement": False, "capture_approved": False, "used_for_decision": False}}
    protocol_path = tmp_path / "protocol.json"
    _write(protocol_path, protocol)
    source = {"packet": {"facts": {f"training:{i}": i for i in range(48)}},
              "source_binding": {"x": 1}}
    monkeypatch.setattr(M.DRIVER, "prepare_source", lambda *_a, **_k: source)
    capture = {"status": "hold_unsupported"}
    out = tmp_path / "out"
    result = M.prepare_pair(tmp_path / "index.json", "case-09", capture, capture, out,
                            protocol_path=protocol_path, execution_anchor="a" * 40,
                            capture_protocols={"case": tmp_path, "reference": tmp_path},
                            lab=tmp_path, allow_fixture=True)
    assert result["status"] == "hold_visual_unavailable"
    assert result["numeric_training_fact_count"] == 48
    assert result["providers"] is None and result["replacement_capture"] is False


def test_prepare_pair_builds_common54_and_six_frame_delta(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    protocol = {"schema": M.SCHEMA, "case_id": "case-03", "claims": {
        "heldout": False, "recovery": False, "video_benefit": False,
        "model_improvement": False, "capture_approved": False, "used_for_decision": False}}
    protocol_path = tmp_path / "protocol.json"
    _write(protocol_path, protocol)
    base = DRIVER_FIXTURE.source(tmp_path / "driver-source")
    source = {**base, "providers": None, "visual_evidence": None}

    def driver_source(*_args, providers=None, visual_evidence=None, **_kwargs):
        return {**source, "providers": providers, "visual_evidence": visual_evidence}

    monkeypatch.setattr(M.DRIVER, "prepare_source", driver_source)
    monkeypatch.setattr(M.DRIVER, "checked_source", lambda value, **_kwargs: value)
    monkeypatch.setattr(M.QUALITY, "_image_bytes", lambda _blocks: b"image")
    def play_facts(role, *_args):
        offset = 0 if role == "case" else 3
        result = {}
        for local in range(1, 4):
            frame = f"f{offset + local}"
            result[f"vision:play:{frame}:timestamp_s"] = float(local)
            for group, names in {
                "command": ("linear_x_mps", "linear_y_mps", "yaw_rate_rps"),
                "motion": ("linear_x_mps", "linear_y_mps", "yaw_rate_rps"),
                "pose": ("base_height_m", "up_axis_z"),
            }.items():
                for name in names:
                    result[f"vision:play:{frame}:{group}:{name}"] = 0.0
        return result
    monkeypatch.setattr(M.PAIR_INPUT, "_play_facts", play_facts)

    chosen = [{"role": "case" if i <= 3 else "reference", "frame_id": f"f{i}",
               "timestamp_s": float(i), "label": {"type": "text", "text": f"f{i}"},
               "image": {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}}
              for i in range(1, 7)]
    def materializer(*_args):
        return chosen, {"case": {}, "reference": {}}

    def pair_builder(_prepared, out):
        out.mkdir()
        for name, provider in base["providers"].items():
            _write(out / f"{name}.provider.json", provider)
        _write(out / "pair.json", {"schema": "fixture-pair"})

    monkeypatch.setattr(M.PROVIDER, "_prepare_pair", pair_builder)
    artifact = {"status": "available", "frames": 625, "fps": 50,
                "capture_approved": False, "model_ready": False, "used_for_decision": False,
                "checkpoint_sha256": "a" * 64,
                "v01_specs": [{"path": "unused", "sha256": "b" * 64}],
                "proofs": {"raw_trace": {"path": "unused", "sha256": "c" * 64}}}
    quality = {"schema": "agentic_frame_quality_guard_v1_evidence_receipt",
               "approval_granted": False, "visual_outcome": "available",
               "frame_refs": [{}] * 6, "image_count": 6,
               "quality_receipts": [{"outcome": "available"}] * 6}
    result = M.prepare_pair(tmp_path / "index.json", "case-03", artifact, artifact,
                            tmp_path / "out", protocol_path=protocol_path,
                            execution_anchor="a" * 40,
                            capture_protocols={"case": tmp_path, "reference": tmp_path},
                            lab=tmp_path, allow_fixture=True, materializer=materializer,
                            quality_builder=lambda _chosen: quality)
    assert result["status"] == "prepared"
    assert result["same_play_numeric_fact_count"] == 54 and result["image_count"] == 6
    assert result["labels_exposed"] is False


def test_load_pair_rejects_provider_tamper(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    provider = tmp_path / "provider.json"
    _write(provider, {"context": {}, "messages": [], "params": {}})
    result = {"schema": M.RESULT_SCHEMA, "status": "prepared", "case_id": "case-03",
              "providers": {"logs_only": {"path": str(provider), "sha256": "0" * 64}},
              "visual_evidence": {}}
    result_path = tmp_path / "result.json"
    _write(result_path, result)
    monkeypatch.setattr(M.DRIVER, "prepare_source", lambda *_a, **_k: {})
    with pytest.raises(ValueError, match="provider SHA"):
        M.load_pair(result_path, tmp_path / "index.json", allow_fixture=True)


def test_production_load_rechecks_capture_source_proofs(tmp_path: Path) -> None:
    index, pair_protocol = tmp_path / "index.json", tmp_path / "pair-protocol.json"
    capture_protocol = tmp_path / "capture-protocol.json"
    raw = tmp_path / "raw-trace.json"
    for path, value in ((index, {"index": 1}), (pair_protocol, {"pair": 1}),
                        (capture_protocol, {"capture": 1}), (raw, {"records": [1]})):
        _write(path, value)
    artifact = {"status": "available", "proofs": {"raw_trace": M._ref(raw)}}
    provenance = {
        "schema": "lineage_same_play_pair_provenance_v1", "fixture": False,
        "execution_anchor": "a" * 40, "lab_root": str(tmp_path),
        "pair_protocol": M._ref(pair_protocol), "source_index": M._ref(index),
        "capture_protocols": {"case": M._ref(capture_protocol),
                              "reference": M._ref(capture_protocol)},
        "artifacts": {role: {"value": artifact, "canonical_sha256": M._digest(artifact)}
                      for role in ("case", "reference")},
    }
    result = {"schema": M.RESULT_SCHEMA, "case_id": "case-03", "provenance": provenance}

    def verify_pair(*_args, **_kwargs):
        return {"ok": True}

    def verify_capture(value, *_args, **_kwargs):
        ref = value["proofs"]["raw_trace"]
        if M._sha(Path(ref["path"])) != ref["sha256"]:
            raise ValueError("raw trace changed")
        return dict(value)

    checked = M._verify_provenance(result, index, pair_protocol_verifier=verify_pair,
                                   capture_verifier=verify_capture)
    assert set(checked) == {"case", "reference"}
    _write(raw, {"records": [2]})
    with pytest.raises(ValueError, match="raw trace changed"):
        M._verify_provenance(result, index, pair_protocol_verifier=verify_pair,
                             capture_verifier=verify_capture)


def test_write_new_does_not_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "value.json"
    M._write_new(path, {"a": 1})
    with pytest.raises(FileExistsError):
        M._write_new(path, {"a": 2})


def test_model_messages_reject_private_metadata_and_paths() -> None:
    clean = {"logs_only": {"messages": [{"role": "user", "content": "training facts"}]}}
    M._assert_model_messages_safe(clean)
    with pytest.raises(ValueError, match="private source key"):
        M._assert_model_messages_safe({"logs_only": {"messages": [
            {"role": "user", "content": {"seed": 2552}}]}})
    with pytest.raises(ValueError, match="path exposed"):
        M._assert_model_messages_safe({"logs_only": {"messages": [
            {"role": "user", "content": "C:\\Users\\name\\secret"}]}})


def test_protocol_has_no_private_source_metadata(monkeypatch: pytest.MonkeyPatch,
                                                 tmp_path: Path) -> None:
    monkeypatch.setattr(M.INPUT, "make_case_packet", lambda *_a, **_k: {
        "source_binding": {"digest": "safe"},
        "private_metadata": {"classification": {"fault": "F1", "seed": 2552}}})
    monkeypatch.setattr(M, "SOURCE_FILES", ())
    index = tmp_path / "index.json"
    _write(index, {})
    value = M.protocol_value(index, "case-03")
    raw = json.dumps(value)
    assert "F1" not in raw and "2552" not in raw and "private_metadata" not in raw
