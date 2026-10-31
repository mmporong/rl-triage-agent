from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "bench"), str(ROOT / "evals"), str(ROOT / "src")]

import runtime_observation_probe_v1 as M

MODULE = Path(M.__file__ or "")


def save(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def base_protocol() -> dict:
    return {
        "source_sha256_lf": {name: "1" * 64 for name in M.P1.RESIDUAL.MINIMUM_BOUND_SOURCES},
        "runtime_source_sha256_lf": {name: "2" * 64 for name in M.RUNTIME_PINS},
        "watchdogs_s": {"legacy": 600, "optimizer": 900},
        "isaac_python_relative": "_isaac_sim/python.bat",
    }


def fixture_source(
    tmp_path: Path, *, arm: str = "FIRST", remaining: list[str] | None = None,
) -> tuple[Path, dict, dict, dict]:
    remaining = ["R1"] if remaining is None else remaining
    checkpoint = tmp_path / "model_299.pt"
    checkpoint.write_bytes(b"checkpoint")
    telemetry = save(tmp_path / "telemetry.json", {"summary": {}})
    manifest = save(tmp_path / "manifest.json", {"arm": arm, "seed": 2307})
    evidence_dir = tmp_path / "training"
    evidence_dir.mkdir()
    owned = save(tmp_path / "owned-result.json", {"natural_exit": True})
    readback = save(tmp_path / "readback.json", {"private": True})
    collected_path = save(tmp_path / "collected.json", {"schema": "fixture"})
    collected = {
        "source_binding": {
            "checkpoint_sha256": M.sha(checkpoint),
            "telemetry_sha256": M.sha(telemetry),
            "binding_digest": "b" * 64,
        },
    }
    arm_contract = {
        "NONE": ([], []),
        "COMPOUND": (["F1", "R1"], []),
        "FIRST": (["F1", "R1"], ["restore_body_frame_reward"]),
        "BOTH": (["F1", "R1"], ["restore_body_frame_reward",
                                  "restore_configured_ppo_epochs"]),
    }
    original_faults, corrections = arm_contract[arm]
    metadata = {
        "lab": str(tmp_path / "lab"), "manifest": str(manifest),
        "evidence_dir": str(evidence_dir), "owned_result": str(owned),
        "telemetry": str(telemetry),
        "raw_sources": {"checkpoint": str(checkpoint), "readback": str(readback)},
        "private_training": {
            "arm": arm, "seed": 2307, "original_faults": original_faults,
            "applied_corrections": corrections,
            "remaining_faults": remaining,
        },
    }
    facts = {
        "runtime:case:available": True, "runtime:case:updates_observed": 3,
        "runtime:case:expected_optimizer_steps": 20,
        "runtime:case:optimizer_steps_min": 4, "runtime:case:optimizer_steps_max": 4,
        "runtime:case:expected_minibatches": 20,
        "runtime:case:minibatches_min": 4, "runtime:case:minibatches_max": 4,
    }
    return collected_path, collected, metadata, facts


def prepared(
    tmp_path: Path, probe: str, *, role: str = "case", arm: str = "FIRST",
    remaining: list[str] | None = None,
) -> tuple[dict, Path, dict, dict]:
    collected_path, collected, metadata, facts = fixture_source(
        tmp_path, arm=arm, remaining=remaining)
    if role == "reference":
        facts = {key.replace("runtime:case:", "runtime:reference:"): value
                 for key, value in facts.items()}
    lab = Path(metadata["lab"])
    (lab / "_isaac_sim").mkdir(parents=True)
    (lab / "_isaac_sim/python.bat").touch()
    stage = tmp_path / "stage"
    wrapper = {"component_probe_protocol": {
        "path": "evals/lineage_probe_backend_v1.py", "sha256_lf": M.sha(M.P1.__file__, lf=True)}}
    with patch.object(M, "verify_protocol", return_value=wrapper), \
            patch.object(M.P1, "verify_protocol", return_value=base_protocol()), \
            patch.object(M.INPUT, "checked", return_value=(collected, metadata)), \
            patch.object(M.FACT, "build", return_value=({"model_view": {"facts": facts}}, {})), \
            patch.object(M.POST, "_admit", return_value={"status": "clear"}):
        plan = M.prepare_probe(
            collected_path, probe, stage=stage, lab=lab, protocol_path=MODULE,
            anchor="a" * 40, role=role)
    return plan, stage / "plan.json", collected, metadata


def test_generic_first_case_legacy_plan_keeps_hidden_labels_private(tmp_path: Path) -> None:
    plan, _path, _collected, _metadata = prepared(tmp_path, "P_reward")
    jobs = M.read(Path(plan["raw_files"]["jobs"]))
    assert jobs["fault"] == "NONE"
    assert plan["source_arm"] == "FIRST"
    assert plan["runtime_facts"]["runtime:case:optimizer_steps_min"] == 4
    assert plan["projection"]["hidden_fault_mask_exposed_to_model"] is False
    assert plan["scope"]["model_hidden_fault_mask_exposed"] is False
    assert "remaining_faults" not in plan["runtime_facts"]
    assert plan["private_lineage"]["remaining_faults"] == ["R1"]


@pytest.mark.parametrize(("arm", "remaining", "fault"), [
    ("NONE", [], "NONE"), ("COMPOUND", ["F1", "R1"], "F1"),
    ("FIRST", ["R1"], "NONE"), ("BOTH", [], "NONE"),
])
def test_registered_training_arms_project_to_frozen_legacy_fault(
    tmp_path: Path, arm: str, remaining: list[str], fault: str,
) -> None:
    plan, _path, _collected, _metadata = prepared(
        tmp_path, "P_episode", arm=arm, remaining=remaining)
    assert M.read(Path(plan["raw_files"]["jobs"]))["fault"] == fault


def test_optimizer_plan_uses_three_update_registered_arm_manifest(tmp_path: Path) -> None:
    plan, _path, _collected, _metadata = prepared(
        tmp_path, M.OPTIMIZER_PROBE, arm="BOTH", remaining=[])
    manifest = M.read(Path(plan["raw_files"]["manifest"]))
    assert manifest["mode"] == "inspection" and manifest["max_iterations"] == 3
    assert manifest["arm"] == "BOTH"
    assert manifest["source_checkpoint_sha256"] == plan["checkpoint"]["sha256"]
    assert plan["projection"]["backend"] == "optimizer_observability_3_update"


def test_verify_plan_rebuilds_collector_and_runtime_facts(tmp_path: Path) -> None:
    plan, plan_path, collected, metadata = prepared(tmp_path, "P_reward")
    wrapper = {"component_probe_protocol": {
        "path": "evals/lineage_probe_backend_v1.py", "sha256_lf": M.sha(M.P1.__file__, lf=True)}}
    facts = deepcopy(plan["runtime_facts"])
    with patch.object(M, "verify_protocol", return_value=wrapper), \
            patch.object(M.P1, "verify_protocol", return_value=base_protocol()), \
            patch.object(M.INPUT, "checked", return_value=(collected, metadata)), \
            patch.object(M.FACT, "build", return_value=({"model_view": {"facts": facts}}, {})):
        assert M._verify_plan(plan_path)["checkpoint"] == plan["checkpoint"]
        changed = deepcopy(plan)
        changed["runtime_facts"]["runtime:case:optimizer_steps_min"] = 20
        save(plan_path, changed)
        with pytest.raises(ValueError, match="facts"):
            M._verify_plan(plan_path)


def test_authorization_rejects_snapshot_runtime_fact_mismatch(tmp_path: Path) -> None:
    plan, plan_path, _collected, _metadata = prepared(tmp_path, "P_reward")
    auth = {"snapshot": {"inputs": {"packet": {"facts": deepcopy(plan["runtime_facts"])}}}}
    authorization = save(tmp_path / "authorization.json", {})
    with patch.object(M, "_BASE_AUTHORIZATION", return_value=auth):
        assert M._authorization(authorization, plan_path, plan) is auth
        auth["snapshot"]["inputs"]["packet"]["facts"][
            "runtime:case:optimizer_steps_min"] = 20
        with pytest.raises(ValueError, match="runtime facts"):
            M._authorization(authorization, plan_path, plan)


def test_done_authorization_reads_current_done_ledger_and_facts(tmp_path: Path) -> None:
    plan, plan_path, _collected, _metadata = prepared(tmp_path, "P_reward")
    ledger_path = save(tmp_path / "probes.ledger.jsonl", {})
    spec = {
        "probe": plan["probe"], "args": M.P1.C.RC.registry()["probes"][plan["probe"]]["args"],
        "checkpoint_sha256": plan["checkpoint"]["sha256"],
        "reference_checkpoint_sha256": "f" * 64,
        "prereg": {"epoch": 1},
    }
    snapshot = {
        "binding": {"epoch": 1},
        "inputs": {"ledger_path": str(ledger_path), "packet": {
            "facts": deepcopy(plan["runtime_facts"]), "execution_budget_s": 1200,
            "input_bindings": {"case": {
                "checkpoint_sha256": plan["checkpoint"]["sha256"],
                "telemetry_sha256": plan["telemetry_sha256"]}},
        }},
    }
    choice = {"probe": plan["probe"]}
    authorization = {
        "schema": M.AUTH_SCHEMA, "plan_sha256": {"case": M.sha(plan_path),
                                                   "reference": "e" * 64},
        "snapshot": snapshot, "choice": choice, "request_spec": spec,
        "ledger_path": str(ledger_path), "request_id": "request-1",
    }
    authorization_path = save(tmp_path / "done-authorization.json", authorization)
    events = [
        {"event": name, "request_id": "request-1"}
        for name in ("proposed", "approved", "consumed", "receipt")
    ]

    class Ledger:
        def requests(self):
            return {"request-1": {"state": "done", "probe": plan["probe"],
                    "prereg_digest": M.P1.C.L.digest(spec["prereg"]),
                    "args_digest": M.P1.C.L.digest(spec["args"])}}

        def events(self):
            return events

    with patch.object(M.P1.C, "_frozen_request_spec", return_value=spec), \
            patch.object(M.P1.C.V, "read_history", return_value=([{
                "probe": plan["probe"]}], "digest", None)), \
            patch.object(M.P1.C.S.M, "RegistryLedger", return_value=Ledger()):
        checked, rows = M._done_authorization(authorization_path, plan_path, plan)
        assert checked == authorization and rows[0]["probe"] == plan["probe"]
        authorization["snapshot"]["inputs"]["packet"]["facts"][
            "runtime:case:optimizer_steps_min"] = 20
        save(tmp_path / "bad-authorization.json", authorization)
        with pytest.raises(ValueError, match="runtime facts"):
            M._done_authorization(tmp_path / "bad-authorization.json", plan_path, plan)


def test_backend_scope_restores_authorization_and_receipt_schema() -> None:
    originals = (M.P1._verify_plan, M.P1._verify_auxiliary_inputs, M.P1._legacy_measurement,
                 M.P1._authorization, M.P1.RECEIPT_SCHEMA)
    with M._backend_scope():
        assert M.P1._verify_plan is M._verify_plan
        assert M.P1._authorization is M._authorization
        assert M.P1.RECEIPT_SCHEMA == M.RECEIPT_SCHEMA
    assert originals == (M.P1._verify_plan, M.P1._verify_auxiliary_inputs,
                         M.P1._legacy_measurement, M.P1._authorization, M.P1.RECEIPT_SCHEMA)


def test_execute_reuses_p1_owned_lease_gpu_and_receipt_boundary(tmp_path: Path) -> None:
    plan, plan_path, _collected, _metadata = prepared(tmp_path, "P_reward")
    auth = save(tmp_path / "authorization.json", {})
    calls: dict[str, object] = {}

    def fake_execute(path, authorization, *, owned, check_lease, check_gpu):
        calls.update(path=path, authorization=authorization, owned=owned,
                     lease=check_lease, gpu=check_gpu)
        return {"schema": M.RECEIPT_SCHEMA, "status": "complete"}

    owned = lambda *_args: {"exit_code": 0}
    gpu = lambda *_args: {"physical_busy": False, "processes": 0}
    with patch.object(M, "_verify_plan", return_value=plan), \
            patch.object(M.POST, "_admit", return_value={"status": "clear"}), \
            patch.object(M.POST, "_lease_reader", return_value=lambda: {"leaseId": "0" * 36}), \
            patch.object(M.P1, "execute_probe", side_effect=fake_execute):
        receipt = M.execute_probe(plan_path, auth, owned=owned, gpu_reader=gpu)
    assert receipt["status"] == "complete"
    assert calls["owned"] is owned and calls["gpu"] is gpu
    assert calls["path"] == plan_path.resolve() and calls["authorization"] == auth


def receipt_fixture(tmp_path: Path, *, status: str) -> tuple[dict, Path, Path, dict]:
    plan, plan_path, _collected, _metadata = prepared(tmp_path, "P_reward")
    save(Path(plan["raw_files"]["report"]), {"probes": {}})
    execution = {
        "exit_code": 0, "natural_exit": True, "timed_out": False, "error_type": None,
        "cleanup_verified": True, "job_assigned_before_release": True,
    }
    owned_result = save(Path(plan["stage"]) / "owned-result.json", execution)
    spec = {"prereg": {"epoch": 1}, "args": {"metric": "fixture"}}
    authorization = {
        "request_id": "request-1", "request_spec": spec,
        "snapshot": {"inputs": {"packet": {}}},
    }
    authorization_path = save(tmp_path / "authorization.json", authorization)
    claim_body = {
        "role": plan["role"], "request_id": "request-1",
        "plan_sha256": M.sha(plan_path),
        "checkpoint_sha256": plan["checkpoint"]["sha256"],
        "telemetry_sha256": plan["telemetry_sha256"],
        "context_digest": M.P1.C.L.digest(spec["prereg"]),
        "args_digest": M.P1.C.L.digest(spec["args"]),
    }
    claim_path = save(tmp_path / "claim.json", claim_body)
    raw_refs = {
        name: M._file_ref(Path(path)) for name, path in plan["raw_files"].items()
        if Path(path).is_file()
    }
    raw_refs["owned_result"] = M._file_ref(owned_result)
    measurement = {"status": "normal", "value": 1.0}
    receipt = {
        "schema": M.RECEIPT_SCHEMA, "status": status,
        "plan": M._file_ref(plan_path),
        "authorization": {**M._file_ref(authorization_path), "request_id": "request-1"},
        "claim": {**M._file_ref(claim_path), "body": claim_body},
        "probe": plan["probe"], "role": plan["role"], "source_arm": plan["source_arm"],
        "checkpoint_sha256": plan["checkpoint"]["sha256"], "seed": plan["seed"],
        "source_refs": {
            "collected_input": plan["collected_input"],
            "source_manifest_sha256": plan["source_manifest_sha256"],
            "auxiliary_inputs": plan["auxiliary_inputs"],
        },
        "projection": plan["projection"],
        "measurement": measurement if status == "complete" else None,
        "execution": execution if status == "complete" else {},
        "raw_refs": raw_refs,
        "error_type": None if status == "complete" else "RuntimeError",
    }
    receipt_path = save(Path(plan["stage"]) / "probe_receipt.json", receipt)
    return plan, plan_path, receipt_path, authorization


def test_verify_receipt_recomputes_complete_raw_proof(tmp_path: Path) -> None:
    plan, plan_path, receipt_path, authorization = receipt_fixture(tmp_path, status="complete")
    measurement = {"status": "normal", "value": 1.0}
    with patch.object(M, "_verify_plan", return_value=plan), \
            patch.object(M, "_done_authorization", return_value=(authorization, [{
                "probe": plan["probe"]}])), \
            patch.object(M, "_legacy_measurement", return_value=measurement):
        verified = M.verify_receipt(plan_path, receipt_path)
    assert verified["status"] == "complete" and verified["correction_allowed"] is True
    assert verified["measurement"] == measurement
    Path(plan["raw_files"]["report"]).write_text("tampered\n", encoding="utf-8")
    with patch.object(M, "_verify_plan", return_value=plan), \
            patch.object(M, "_done_authorization", return_value=(authorization, [])), \
            pytest.raises(ValueError, match="raw refs"):
        M.verify_receipt(plan_path, receipt_path)


def test_verify_receipt_preserves_unknown_and_forbids_correction(tmp_path: Path) -> None:
    plan, plan_path, receipt_path, authorization = receipt_fixture(tmp_path, status="unknown")
    with patch.object(M, "_verify_plan", return_value=plan), \
            patch.object(M, "_done_authorization", return_value=(authorization, [])):
        verified = M.verify_receipt(plan_path, receipt_path)
    assert verified["status"] == "unknown"
    assert verified["correction_allowed"] is False and verified["measurement"] is None


def test_protocol_wrapper_is_not_original_p1_disguise(tmp_path: Path) -> None:
    base = save(tmp_path / "base.json", {"schema": M.P1.PROTOCOL_SCHEMA})
    with patch.object(M, "ROOT", tmp_path), patch.object(M, "SOURCE_FILES", ("source.py",)):
        (tmp_path / "source.py").write_text("source\n", encoding="utf-8")
        protocol = M.prepare_protocol(base)
    assert protocol["schema"] == M.PROTOCOL_SCHEMA
    assert protocol["purpose"] == M.PROTOCOL_PURPOSE
    assert protocol["component_probe_protocol"]["sha256_lf"] == M.sha(base, lf=True)
    assert M.PROTOCOL_FIELDS == set(protocol)


def test_numeric_classification_remains_frozen_controller_function() -> None:
    assert M.P1.C.classify_raw is M.classify_raw
