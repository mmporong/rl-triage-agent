from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "src")]

import lineage_post_diagnosis_epoch_v1 as E2


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")


def test_protocol_binds_typed_backend_and_epoch2_sources(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    backend = tmp_path / "post-backend.json"
    _write(backend, {"schema": E2.POST_BACKEND.PROTOCOL_SCHEMA})
    monkeypatch.setattr(E2, "_relative", lambda _path: "bench/protocols/post-backend.json")
    value = E2.prepare_protocol(backend)
    assert value["schema"] == E2.PROTOCOL_SCHEMA
    assert value["input_contract"]["prior_outcome"] == "not_recovered"
    assert value["input_contract"]["case_source_type"] == "post_observation"
    assert value["execution_contract"]["result_relabelling_after_write"] is False
    assert set(value["source_sha256_lf"]) == set(E2.SOURCE_FILES)


def test_adapter_scope_and_base_result_are_epoch2_at_creation() -> None:
    originals = (E2.E0.INPUT, E2.E0.P, E2.E0._source_binding, E2.E0._base_result)
    with E2.adapter_scope():
        assert (E2.E0.INPUT, E2.E0.P) == (E2._POST_INPUT, E2.POST_BACKEND)
        assert E2.E0._source_binding is E2._checked_post_binding
        assert E2.E0._base_result is E2._epoch2_base_result
    assert (E2.E0.INPUT, E2.E0.P, E2.E0._source_binding, E2.E0._base_result) == originals

    binding = {
        "case": {"prior_not_recovered_receipt_sha256": "a" * 64,
                 "applied_corrections": ["restore_body_frame_reward"]},
        "reference": {},
        "second_epoch_gate": {"input_checkpoint_sha": "b" * 64, "epoch": 2},
    }
    result = E2._epoch2_base_result(
        run_id="post", policy="registry_next", case_id="case-07", gate_sha256="c" * 64,
        gate_case={"label": "eligible"}, binding=binding, max_decisions=3,
        execution_budget_s=1200, evidence_scope="actual_backend",
    )
    assert (result["schema"], result["epoch"], result["epoch2_registered"]) == (
        E2.SCHEMA, 2, True)
    assert result["prior_not_recovered_receipt_sha256"] == "a" * 64


def test_checked_post_binding_rejects_checkpoint_or_telemetry_swap(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    post_path, reference_path, receipt_path = (
        tmp_path / "post.json", tmp_path / "reference.json", tmp_path / "receipt.json")
    _write(post_path, {})
    _write(reference_path, {})
    _write(receipt_path, {"seed": 2307, "correction": "restore_body_frame_reward",
                          "new_checkpoint": {"sha256": "1" * 64}})
    post = {
        "source_binding": {"checkpoint_sha256": "1" * 64, "telemetry_sha256": "2" * 64,
                           "repair_receipt_sha256": "3" * 64},
        "private_metadata": {
            "sources": {"repair_receipt": str(receipt_path)},
            "second_epoch_gate": {"input_checkpoint_sha": "1" * 64,
                                  "prior_valid_not_recovered_receipt_sha": "3" * 64},
        },
    }
    reference = {"private_source_map": {"seed": 2307, "arm": "NONE", "job_id": "none-s2307"},
                 "source_binding": {"checkpoint_sha256": "4" * 64,
                                    "telemetry_sha256": "5" * 64}}
    monkeypatch.setattr(E2.POST, "checked_observation", lambda _path: (post, {}))
    monkeypatch.setattr(E2.INPUT0, "_checked_collected", lambda _path: (reference, {}))
    envelope = {
        "private_metadata": {
            "post_observation": {"path": str(post_path), "sha256": E2._sha(post_path)},
            "reference_collected": {"path": str(reference_path), "sha256": E2._sha(reference_path)},
        },
        "packet": {
            "epoch": 2, "applied_corrections": ["restore_body_frame_reward"],
            "prior_receipt": {"verdict": "not_recovered"},
            "input_bindings": {
                "case": {"checkpoint_sha256": "1" * 64, "telemetry_sha256": "2" * 64},
                "reference": {"checkpoint_sha256": "4" * 64, "telemetry_sha256": "5" * 64},
            },
        },
    }
    gate = {"case_id": "case-07", "arm": "COMPOUND", "seed": 2307,
            "reference_job": "none-s2307"}
    assert E2._checked_post_binding(envelope, gate)["case"]["job_id"] == "post_observation"
    envelope["packet"]["input_bindings"]["case"]["telemetry_sha256"] = "9" * 64
    with pytest.raises(ValueError, match="checkpoint/telemetry"):
        E2._checked_post_binding(envelope, gate)


class _NullScope:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *_args: object) -> None:
        return None


def test_run_epoch_preserves_actual_default_path_and_restores_globals(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    post, reference, probe, epoch_protocol = (
        tmp_path / "post.json", tmp_path / "reference.json",
        tmp_path / "probe.json", tmp_path / "epoch.json")
    for path in (post, reference, probe, epoch_protocol):
        _write(path, {})
    protocol = {"component_post_backend_protocol": {"path": "probe.json", "sha256_lf": "a" * 64}}
    monkeypatch.setattr(E2, "verify_protocol", lambda *_args, **_kwargs: protocol)
    monkeypatch.setattr(E2, "ROOT", tmp_path)
    monkeypatch.setattr(E2.POST, "checked_observation", lambda _path: ({}, {}))
    monkeypatch.setattr(E2.GPU_WAIT.B, "lease_context", lambda: {"status": "active"})
    monkeypatch.setattr(E2.RETRY, "lease_read_scope", lambda _path: _NullScope())
    observed = {}

    def epoch0(*_args: object, **kwargs: object) -> dict:
        observed.update(kwargs)
        assert E2.E0.INPUT is E2._POST_INPUT and E2.E0.P is E2.POST_BACKEND
        return {"schema": E2.SCHEMA, "epoch": 2, "epoch2_registered": True,
                "evidence_scope": "actual_backend"}

    monkeypatch.setattr(E2.E0, "run_epoch", epoch0)
    result = E2.run_epoch(
        tmp_path / "out", post, reference, "registry_next",
        epoch_protocol=epoch_protocol, anchor="a" * 40, lab=tmp_path,
        probe_protocol=probe, run_id="post", gate_path=tmp_path / "gate.json",
        gate_sha256="b" * 64, case_id="case-07", max_decisions=3,
        execution_budget_s=1200,
    )
    assert result["evidence_scope"] == "actual_backend"
    assert "probe_executor" not in observed and "complete_factory" not in observed
    assert E2.E0.INPUT is E2._ORIGINAL_INPUT and E2.E0.P is E2._ORIGINAL_PROBE
    with pytest.raises(ValueError, match="주입"):
        E2.run_epoch(tmp_path / "other", post, reference, "registry_next",
                     epoch_protocol=epoch_protocol, anchor="a" * 40,
                     probe_executor=lambda: None)
    with pytest.raises(ValueError, match="새 세 slot"):
        E2.run_epoch(tmp_path / "model", post, reference, "model_adaptive",
                     epoch_protocol=epoch_protocol, anchor="a" * 40, lab=tmp_path,
                     probe_protocol=probe, transport_root=None, slot_ids=())
