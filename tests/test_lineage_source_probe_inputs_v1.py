# pyright: reportMissingImports=false
from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("bench", "evals", "src")]

import lineage_source_probe_inputs_v1 as M


def save(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def telemetry() -> dict:
    return {"summary": {
        field: {"n": index + 1, "mean_last_20pct": float(index)}
        for index, field in enumerate(M.INPUT.SELECT.V.TELEMETRY_FIELDS)
    }}


def source_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *,
                   arm: str = "F2") -> tuple[Path, Path, dict, dict, dict]:
    job_id, source_id = "f2-s2554", "source-01"
    stage = tmp_path / "batch" / job_id
    log = tmp_path / "logs" / job_id
    checkpoint = log / "model_299.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"source-v2-checkpoint")
    event = log / "events.out.tfevents.fixture"
    event.write_bytes(b"source-v2-event")
    event_copy = tmp_path / "inputs" / "sources" / job_id / "source.events.out.tfevents.raw"
    event_copy.parent.mkdir(parents=True)
    event_copy.write_bytes(event.read_bytes())
    telemetry_path = save(event_copy.parent / "telemetry.json", telemetry())
    extract_receipt = save(event_copy.parent / "extract.receipt.json", {"returncode": 0})
    protocol = save(tmp_path / "protocol.json", {"schema": M.BATCH.SCHEMA})

    manifest = {
        "schema": M.BATCH.CHILD.SCHEMA, "purpose": "source", "job_id": job_id,
        "case_id": "case-05", "arm": arm, "seed": 2554,
        "run_name": "f2_s2554", "task": M.BATCH.CHILD.TASK,
        "num_envs": 4096, "max_iterations": 300,
        "source_commit": "a" * 40, "protocol_relative": "protocol.json",
        "protocol_sha256_lf": M.sha(protocol, lf=True),
        "source_sha256_lf": {name: "1" * 64 for name in M.BATCH.CHILD.SOURCE_FILES},
        "runtime_source_sha256_lf": {name: "2" * 64 for name in M.BASE_PROBE.RUNTIME_PINS},
    }
    manifest_path = save(stage / "manifest.json", manifest)
    owned = {"exit_code": 0, "natural_exit": True, "cleanup_verified": True,
             "job_assigned_before_release": True, "timed_out": False, "error_type": None}
    save(stage / "owned-result.json", owned)
    evidence = {
        "schema": "lineage_source_training_evidence_v1", "arm": arm, "seed": 2554,
        "learning_completed": True, "learn_returned": True,
        "recorded_before_app_close": True, "update_count": 300, "log_dir": str(log),
        "checkpoint": {"path": str(checkpoint), "sha256": M.sha(checkpoint),
                       "log_dir": str(log)},
        "ppo_readback": {"configured": {"learning_epochs": 5, "mini_batches": 4},
                         "effective": {"learning_epochs": 5, "mini_batches": 4}},
        "patch_readback": {"configured_epochs": 5, "effective_epochs": 5,
                           "r1_applied": False},
    }
    evidence_path = save(stage / "training" / M.BATCH.GUARD.EVIDENCE, evidence)
    readback = {
        "update_count": 300, "checkpoint": evidence["checkpoint"],
        "ppo_readback": evidence["ppo_readback"], "patch_readback": evidence["patch_readback"],
    }
    records_path = stage / "training" / M.BATCH.GUARD.RECORDS
    records_path.write_text(json.dumps(readback) + "\n", encoding="utf-8")
    fixed_path = save(stage / "fixed-result.json", {"fixed": True})

    binding = {
        "checkpoint_sha256": M.sha(checkpoint), "telemetry_sha256": M.sha(telemetry_path),
        "manifest_sha256": M.sha(manifest_path), "evidence_sha256": M.sha(evidence_path),
        "records_sha256": M.sha(records_path), "fixed_result_sha256": M.sha(fixed_path),
    }
    selected = {
        "job_id": job_id, "purpose": "source", "arm": arm, "seed": 2554,
        "role": "case", "checkpoint": evidence["checkpoint"], "runtime_facts": None,
        "evidence_sha256": binding["evidence_sha256"],
        "records_sha256": binding["records_sha256"],
        "fixed26": {"result_sha256": binding["fixed_result_sha256"]},
    }
    accepted_path = save(stage / "accepted.json", selected)
    rows = [selected] + [{"job_id": f"other-{index}"} for index in range(19)]
    result = {
        "schema": M.BATCH.RESULT_SCHEMA, "status": "complete", "mode": "all",
        "source_jobs_completed": 20, "fixed26_completed": 20, "model_api": 0,
        "repairs": 0, "runtime_facts_fabricated": False, "records": rows,
    }
    result_path = save(tmp_path / "batch" / "result.json", result)
    index_path = save(tmp_path / "inputs" / "collected.json", {"schema": M.INPUT.SCHEMA})
    private = {
        "public_id": source_id, "binding": binding,
        "material": {
            "stage": str(stage), "event": M.ref(event), "event_copy": M.ref(event_copy),
            "telemetry": M.ref(telemetry_path), "receipt": M.ref(extract_receipt),
        },
        "manifest": manifest, "effective_fault_metadata": {"registered_arm": arm},
    }
    public = {"sources": {source_id: binding}}
    metadata = {
        "sources": {job_id: private}, "protocol": M.ref(protocol),
        "source_result": M.ref(result_path), "anchor": "a" * 40,
        "runtime_observation": {
            "fabricated_counts": False,
            "actual_3_update_check": "unavailable_requires_separate_gpu_run",
        },
    }
    monkeypatch.setattr(M.INPUT, "checked", lambda _path: (deepcopy(public), deepcopy(metadata)))
    monkeypatch.setattr(M.BATCH.CHILD, "validate_manifest", lambda value: deepcopy(value))
    monkeypatch.setattr(M.BATCH, "completed", lambda *_args: deepcopy(evidence))
    return index_path, accepted_path, public, metadata, evidence


def test_collect_and_check_pin_source_v2_without_runtime_count_fabrication(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    index, _accepted, _public, _metadata, _evidence = source_fixture(tmp_path, monkeypatch)
    out = tmp_path / "bridge" / "collected.json"
    value = M.collect(index, out, job_id="f2-s2554")
    checked, raw = M.checked(out)
    assert value == checked and raw["manifest"]["schema"] == M.BATCH.CHILD.SCHEMA
    assert value["source_kind"] == M.SOURCE_KIND
    assert value["runtime_observation"] == {
        "available": False, "facts": "role_scoped_false_and_null",
        "actual_3_update_check": "unavailable_requires_separate_gpu_run",
        "configured_or_effective_counts_used_as_actual": False,
    }
    assert value["scope"] == {"cpu_input_bridge_only": True, "approval": False,
                              "execution_allowed": False, "gpu": False, "api": False}


def test_source_id_alias_and_raw_tamper_fail_closed(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    index, _accepted, _public, metadata, _evidence = source_fixture(tmp_path, monkeypatch)
    out = tmp_path / "bridge.json"
    M.collect(index, out, source_id="source-01")
    telemetry_path = Path(metadata["sources"]["f2-s2554"]["material"]["telemetry"]["path"])
    telemetry_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="telemetry|ref SHA"):
        M.checked(out)


def test_job_and_source_id_must_select_same_unique_source(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    index, _accepted, _public, _metadata, _evidence = source_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="동일한 source 하나"):
        M.collect(index, tmp_path / "bad.json", job_id="f2-s2554", source_id="source-99")


def test_isolated_source_projects_role_null_runtime_facts_and_preserves_globals(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    index, _accepted, _public, _metadata, _evidence = source_fixture(tmp_path, monkeypatch)
    out = tmp_path / "bridge.json"
    M.collect(index, out, job_id="f2-s2554")
    monkeypatch.setattr(M.BATCH.CHILD, "verify_binding", lambda *_args, **_kwargs: None)
    _collected, _metadata, source = M._source(out, tmp_path, "reference")
    assert source["kind"] == M.SOURCE_KIND and source["source_arm"] == "F2"
    assert source["runtime_facts"] == M.RUNTIME.unavailable_facts("reference")
    assert source["original_faults"] == ["F2"] and source["remaining_faults"] == ["F2"]
    assert M._PROBE._source is M._source
    assert M.BASE_PROBE._source is not M._source
    assert M._HOST.OBS is M._PROBE and M.BASE_HOST.OBS is M.BASE_PROBE


def test_prepare_legacy_plan_reuses_existing_contract_with_source_v2_validator(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    index, _accepted, _public, _metadata, _evidence = source_fixture(tmp_path, monkeypatch)
    bridge = tmp_path / "bridge.json"
    M.collect(index, bridge, job_id="f2-s2554")
    lab = tmp_path / "lab"
    python = lab / "_isaac_sim/python.bat"
    python.parent.mkdir(parents=True)
    python.touch()
    wrapper = {"component_probe_protocol": {
        "path": "evals/lineage_probe_backend_v1.py",
        "sha256_lf": M.sha(Path(str(M.BASE_PROBE.P1.__file__)), lf=True)}}
    base = {
        "source_sha256_lf": {name: "1" * 64
                             for name in M.BASE_PROBE.P1.RESIDUAL.MINIMUM_BOUND_SOURCES},
        "runtime_source_sha256_lf": {name: "2" * 64 for name in M.BASE_PROBE.RUNTIME_PINS},
        "watchdogs_s": {"legacy": 600, "optimizer": 900},
        "isaac_python_relative": "_isaac_sim/python.bat",
    }
    monkeypatch.setattr(M.BATCH.CHILD, "verify_binding", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(M._PROBE, "verify_protocol", lambda *_args, **_kwargs: wrapper)
    monkeypatch.setattr(M._PROBE.P1, "verify_protocol", lambda *_args, **_kwargs: base)
    monkeypatch.setattr(M._PROBE.POST, "_admit", lambda *_args, **_kwargs: {"status": "clear"})
    plan = M.prepare_probe(
        bridge, "P_reward", stage=tmp_path / "probe", lab=lab,
        protocol_path=Path(str(M.__file__)), anchor="a" * 40, role="case")
    assert plan["source_kind"] == M.SOURCE_KIND and plan["source_arm"] == "F2"
    assert plan["runtime_facts"] == M.RUNTIME.unavailable_facts("case")
    assert M.read(Path(plan["raw_files"]["jobs"]))["fault"] == "F2"
    assert plan["scope"]["approval_created"] is False


def test_source_closure_and_optimizer_manifest_keep_existing_three_update_contract() -> None:
    assert "evals/lineage_source_probe_inputs_v1.py" in M.SOURCE_FILES
    assert "tests/test_lineage_source_probe_inputs_v1.py" in M.SOURCE_FILES
    assert set(M.BASE_PROBE.SOURCE_FILES) <= set(M.SOURCE_FILES)
    assert M._PROBE.execute_probe.__code__.co_code == M.BASE_PROBE.execute_probe.__code__.co_code
    assert M._HOST.execute_pair.__code__.co_code == M.BASE_HOST.execute_pair.__code__.co_code
    assert M._HOST._ledger_start.__code__.co_code == M.BASE_HOST._ledger_start.__code__.co_code
    source = {"source_arm": "F2", "seed": 2554,
              "checkpoint": {"sha256": "c" * 64}}
    protocol = {
        "source_sha256_lf": {name: "1" * 64
                             for name in M.BASE_PROBE.P1.RESIDUAL.MINIMUM_BOUND_SOURCES},
        "runtime_source_sha256_lf": {name: "2" * 64 for name in M.BASE_PROBE.RUNTIME_PINS},
    }
    manifest = M._residual_manifest(
        protocol, source, Path("probe"), "a" * 40,
        ROOT / "evals/lineage_source_probe_inputs_v1.py")
    assert manifest["max_iterations"] == 3 and manifest["arm"] == "NONE"
    assert manifest["original_faults"] == [] and manifest["applied_corrections"] == []
