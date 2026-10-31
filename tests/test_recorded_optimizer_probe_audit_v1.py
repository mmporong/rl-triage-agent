# pyright: reportMissingImports=false
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "bench")]

import recorded_optimizer_probe_audit_v1 as A


def write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return path


def test_prepare_protocol_locks_counterbalance_and_bounds(monkeypatch: pytest.MonkeyPatch,
                                                          tmp_path: Path) -> None:
    source = write(tmp_path / "source.json", {
        "schema": A.PARENT.SOURCE.BATCH_SCHEMA, "status": "complete", "anchor": "a" * 40})
    monkeypatch.setattr(A, "SOURCE_FILES", ())
    value = A.prepare_protocol(
        ROOT / "bench/protocols/prospective_agentic_comparison_20261101_v4.json",
        ROOT / "bench/protocols/prospective_diagnosis_training_20261101_v1.json", source,
        ROOT / "bench/protocols/runtime_observation_paired_probes_20261031_v1.json",
        ROOT / "bench/protocols/runtime_observation_view_v1.json")
    assert value["design"]["order"] == {
        "2320": [A.RECORDED.PROBE, A.FRESH_PROBE],
        "2321": [A.FRESH_PROBE, A.RECORDED.PROBE]}
    assert value["bounds"] == {"wall_s": 2400, "new_sdk_launches": 4,
                               "fresh_roles": 4, "recorded_sdk_launches": 0}
    assert value["claims"] == {"gpu_kernel_savings": False, "early_detection": False,
                                "recovery": False, "model_advantage": False}


def test_preflight_builds_four_independent_harnesses_without_execution(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    protocol = {"source_batch": {"sha256": "a" * 64}}
    monkeypatch.setattr(A, "verify_protocol", lambda *_args, **_kwargs: protocol)
    monkeypatch.setattr(A, "_cases", lambda *_args: {2320: ({}, {}), 2321: ({}, {})})
    seen = []

    def harness(_case, _reference, folder, target):
        seen.append((folder, target))
        return {"state_digest": target}, {"action": "run_probe", "target": target, "fact_ids": []}

    monkeypatch.setattr(A, "_harness", harness)
    monkeypatch.setattr(A.SEL, "request_spec", lambda state, action: {
        "state_digest": state["state_digest"], "target": action["target"]})
    source = write(tmp_path / "source.json", {"status": "complete"})
    result = A.preflight(tmp_path / "protocol.json", "a" * 40, source, tmp_path,
                         tmp_path / "preflight")
    assert [target for _, target in seen] == [A.RECORDED.PROBE, A.FRESH_PROBE,
                                               A.FRESH_PROBE, A.RECORDED.PROBE]
    assert result["model_requests"] == result["sdk_launches"] == 0
    assert result["gpu_started"] is False


def test_canonical_recorded_and_fresh_counts_agree(monkeypatch: pytest.MonkeyPatch,
                                                    tmp_path: Path) -> None:
    measurement = {"expected_optimizer_steps": 20, "optimizer_steps_min": 4,
                   "optimizer_steps_max": 4, "expected_minibatches": 20,
                   "minibatches_min": 4, "minibatches_max": 4}
    monkeypatch.setattr(A.RECORDED, "checked_receipt", lambda *_args: {
        "measurement": measurement, "outcome": "abnormal"})
    recorded = A._canonical(A.RECORDED.PROBE, "case", tmp_path / "p", tmp_path / "r")
    records = tmp_path / "records.jsonl"
    records.write_text("\n".join(json.dumps({
        "update_index": index, "expected": {"optimizer_steps": 20, "minibatches": 20},
        "observed": {"optimizer_steps": 4, "minibatches": 4}}) for index in range(3)) + "\n",
        encoding="utf-8")
    monkeypatch.setattr(A.OBS, "verify_receipt", lambda *_args: {
        "status": "complete", "raw_refs": {"records": {"path": str(records)}}})
    fresh = A._canonical(A.FRESH_PROBE, "case", tmp_path / "p", tmp_path / "r")
    assert recorded == fresh


def test_canonical_fresh_rejects_missing_update(monkeypatch: pytest.MonkeyPatch,
                                                tmp_path: Path) -> None:
    records = tmp_path / "records.jsonl"
    records.write_text(json.dumps({"update_index": 0}) + "\n", encoding="utf-8")
    monkeypatch.setattr(A.OBS, "verify_receipt", lambda *_args: {
        "status": "complete", "raw_refs": {"records": {"path": str(records)}}})
    with pytest.raises(ValueError, match="3 update"):
        A._canonical(A.FRESH_PROBE, "case", tmp_path / "p", tmp_path / "r")


def test_execute_aggregates_agreement_cost_and_sdk_counts(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source = write(tmp_path / "source.json", {"status": "complete"})
    protocol_file = write(tmp_path / "protocol.json", {"fixture": True})
    runtime = write(tmp_path / "runtime.json", {"fixture": True})
    checked = {"source_batch": {"sha256": A._sha(source)},
               "protocols": {"paired_probe": {"path": "probe.json"},
                             "view": {"path": "view.json"}},
               "claims": {"gpu_kernel_savings": False, "early_detection": False,
                          "recovery": False, "model_advantage": False}}
    case = {"input": runtime, "metadata": {"runtime_facts": {
        "case": {"path": str(runtime)}, "reference": {"path": str(runtime)}}}}
    monkeypatch.setattr(A, "verify_protocol", lambda *_args, **_kwargs: checked)
    monkeypatch.setattr(A, "_cases", lambda *_args: {2320: (case, case), 2321: (case, case)})
    monkeypatch.setattr(A, "_harness", lambda _case, _reference, _folder, target: (
        {"observed": {}}, {"action": "run_probe", "target": target, "fact_ids": []}))
    monkeypatch.setattr(A, "_verify_current", lambda *_args: None)
    monkeypatch.setattr(A.PARENT, "_lease", lambda: {"lease_id": "fixture"})
    monkeypatch.setattr(A.SEL, "advance", lambda state, action, **_kwargs: {
        "observed": {action["target"]: "abnormal"}})
    canonical = {"role": "case", "expected_optimizer_steps": 20,
                 "optimizer_steps": [4, 4], "expected_minibatches": 20,
                 "minibatches": [4, 4], "outcome": "abnormal", "records": 3}
    monkeypatch.setattr(A, "_canonical", lambda target, role, plan, receipt: {
        **canonical, "role": role})

    def pair(state, action, **kwargs):
        folder = Path(kwargs["out"])
        paths = {}
        for role in ("case", "reference"):
            paths[f"{role}_plan"] = write(folder / role / "plan.json", {})
            paths[f"{role}_receipt"] = write(folder / role / "receipt.json", {})
        return {**paths, "parent_wall_s": 1.0 if action["target"] == A.RECORDED.PROBE else 2.0,
                "owned_child_wall_s": 0.0 if action["target"] == A.RECORDED.PROBE else 3.0}

    result = A.execute(protocol_file, "a" * 40, source, tmp_path, tmp_path / "out",
                       execute_pair=pair)
    assert result["agreement_count"] == 2 and result["sdk_launches"] == 4
    assert result["conditional_candidate"] is True
    assert result["model_requests"] == result["repairs"] == 0


def test_write_once(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    A._write(path, {"status": "complete"})
    with pytest.raises(FileExistsError):
        A._write(path, {"status": "replaced"})
