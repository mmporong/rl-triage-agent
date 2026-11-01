# pyright: reportMissingImports=false
from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("bench", "evals", "tests", "src")]

import five_policy_recovery_epoch2_v1 as M
import test_local_pair_four_arm_v1 as F


def save(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def first_component_fixture(tmp_path: Path, second_seeds: dict, *, name="first-protocol.json") -> Path:
    from test_five_policy_recovery_candidate_v1 import conditional_fixture

    first_seeds = {key: 7000 + index for index, key in enumerate(second_seeds)}
    design = conditional_fixture(tmp_path, first_seeds, second_seeds)
    return save(tmp_path / name, {"schema": M.FIRST.PROTOCOL_SCHEMA,
                "recovery_seeds": first_seeds,
                "components": {"conditional_design": {
                    "path": M.FIRST.CONDITIONAL_DESIGN_RELATIVE,
                    "sha256_lf": M.sha(design, lf=True)}}})


def test_second_seed_cannot_change_after_first_component_is_fixed(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "ROOT", tmp_path)
    monkeypatch.setattr(M.FIRST, "ROOT", tmp_path)
    seeds = {f"{case}:{policy}": 9000 + index
             for index, (case, policy) in enumerate(M.TARGETS)}
    first = first_component_fixture(tmp_path, seeds)
    seeds["case-03:logs_only"] += 100
    probe = save(tmp_path / "probe.json", {"schema": M.BASE_PROBE.PROTOCOL_SCHEMA})
    with pytest.raises(ValueError, match="second seed"):
        M.protocol_value(first, probe, [], seeds)


def test_mixed_source_dispatches_by_real_schema_without_disguise(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    case = save(tmp_path / "case.json", {"schema": M.TRAINING_INPUT.SCHEMA})
    reference = save(tmp_path / "reference.json", {"schema": M.SOURCE_INPUT.SCHEMA})
    calls = []
    monkeypatch.setattr(M.BASE_PROBE, "_source", lambda path, lab, role: calls.append(
        ("first", path, lab, role)) or ({}, {}, {"kind": "first"}))
    monkeypatch.setattr(M.SOURCE_INPUT, "_source", lambda path, lab, role: calls.append(
        ("source-v2", path, lab, role)) or ({}, {}, {"kind": "source-v2"}))
    assert M._mixed_source(case, tmp_path, "case")[2]["kind"] == "first"
    assert M._mixed_source(reference, tmp_path, "reference")[2]["kind"] == "source-v2"
    assert [row[0] for row in calls] == ["first", "source-v2"]
    with pytest.raises(ValueError, match="FIRST collected"):
        M._mixed_source(reference, tmp_path, "case")


def material_fixture(tmp_path: Path) -> dict:
    epoch = save(tmp_path / "epoch2.json", {})
    receipt = save(tmp_path / "receipt.json", {})
    candidate = save(tmp_path / "candidate.json", {})
    first_protocol = save(tmp_path / "first-protocol.json", {})
    collected = save(tmp_path / "collected.json", {})
    reference = save(tmp_path / "reference.json", {})
    cp = save(tmp_path / "model.pt", {"checkpoint": True})
    return {
        "input": {}, "input_path": epoch, "receipt": {}, "receipt_path": receipt,
        "receipt_sha256": M.sha(receipt), "candidate_path": candidate,
        "first_protocol_path": first_protocol,
        "candidate": {
            "case_id": "case-07", "policy": "logs_only", "source_arm": "COMPOUND",
            "correction": "restore_body_frame_reward", "original_faults": ["F1", "R1"],
            "inputs": {"source_index": {"path": "index", "sha256": "1" * 64}}},
        "checkpoint_sha256": M.sha(cp), "collected_path": collected,
        "case_collected": {},
        "case_metadata": {"telemetry": str(save(tmp_path / "case-telemetry.json", {}))},
        "case_source": {
            "checkpoint": {"path": str(cp), "sha256": M.sha(cp)},
            "telemetry_sha256": "2" * 64, "runtime_facts": {
                f"runtime:case:{name}": False if name == "available" else None
                for name in M.RUNTIME.FACT_NAMES},
            "remaining_faults": ["R1"]},
        "reference_path": reference, "reference_collected": {},
        "reference_metadata": {"raw": {"telemetry": {
            "path": str(save(tmp_path / "reference-telemetry.json", {}))}}},
        "reference_source": {
            "checkpoint": {"path": "reference.pt", "sha256": "3" * 64},
            "telemetry_sha256": "4" * 64,
            "runtime_facts": M.RUNTIME.unavailable_facts("reference")},
    }


def test_prepare_source_builds_epoch2_public48_and_backend16_hidden(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    material = material_fixture(tmp_path)
    monkeypatch.setattr(M, "_first_material", lambda *_args, **_kwargs: deepcopy(material))

    def projection(role, _telemetry):
        return ({f"telemetry:{role}:field-{index}": float(index) for index in range(24)}, {})

    monkeypatch.setattr(M.TRAINING_INPUT.INPUT, "_telemetry_projection", projection)
    source = M._prepare_source_value(material["input_path"], lab=tmp_path)
    assert len(source["packet"]["facts"]) == 48
    assert len(source["backend_packet"]["facts"]) == 64
    assert not any(key.startswith("runtime:") for key in source["packet"]["facts"])
    assert source["packet"]["prior_receipt"] == {
        "verdict": "not_recovered", "correction_id": "restore_body_frame_reward",
        "result_sha256": material["receipt_sha256"],
        "checkpoint_sha256": material["checkpoint_sha256"]}
    assert source["backend_packet"]["facts"]["runtime:reference:available"] is False


def test_first_material_rejects_swapped_checkpoint_and_runtime_facts(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cp = save(tmp_path / "model.pt", {"cp": 1})
    collected = save(tmp_path / "collected.json", {})
    runtime_case = save(tmp_path / "runtime-case.json", {})
    runtime_reference = save(tmp_path / "runtime-reference.json", {})
    reference = save(tmp_path / "reference.json", {})
    candidate_path = save(tmp_path / "candidate.json", {})
    candidate = {
        "case_id": "case-07", "policy": "fixed_eligible", "source_arm": "COMPOUND",
        "correction": "restore_body_frame_reward", "original_faults": ["F1", "R1"],
        "source_checkpoint_sha256": "s" * 64,
        "inputs": {"reference_probe_input": M.ref(reference), "anchor": "a" * 40}}
    first_protocol = save(tmp_path / "first-protocol.json", {})
    evidence = save(tmp_path / "training-evidence.json", {
        "fresh_initialization": True, "checkpoint_loaded": False, "resume_requested": False,
        "source_checkpoint_sha256": "s" * 64, "remaining_faults": ["R1"]})
    receipt = {
        "schema": M.FIRST.RECEIPT_SCHEMA, "status": "not_recovered",
        "recovery": {"outcome": "not_recovered"}, "candidate": M.ref(candidate_path),
        "case_id": "case-07", "policy": "fixed_eligible",
        "correction": "restore_body_frame_reward",
        "protocol": M.ref(first_protocol), "source_checkpoint_sha256": "s" * 64,
        "source_checkpoint_loaded": False,
        "new_checkpoint": M.ref(cp), "training": {
            "evidence": M.ref(evidence),
            "checkpoint": M.ref(cp),
            "collected": M.ref(collected), "runtime_facts": {
                "case": M.ref(runtime_case), "reference": M.ref(runtime_reference)}},
        "automatic_retry": False, "second_epoch_registered": False}
    receipt_path = save(tmp_path / "receipt.json", receipt)
    epoch = {
        "schema": M.FIRST.EPOCH2_SCHEMA, "status": "input_only",
        "new_protocol_required": True, "approval_granted": False,
        "execution_allowed": False, "automatic_second_execution": False,
        "prior_receipt": M.ref(receipt_path), "candidate": M.ref(candidate_path),
        "checkpoint": M.ref(cp), "collected": M.ref(collected),
        "runtime_facts": {"case": M.ref(runtime_case), "reference": M.ref(runtime_reference)},
        "remaining_faults": ["R1"]}
    epoch_path = save(tmp_path / "epoch.json", epoch)
    monkeypatch.setattr(M.TRAINING_INPUT, "checked", lambda _path: (
        {"source_binding": {"checkpoint_sha256": M.sha(cp)}}, {}))
    case_facts = M.RUNTIME.unavailable_facts("case")
    monkeypatch.setattr(M.RUNTIME, "checked", lambda *_args, **_kwargs: ({}, case_facts))
    monkeypatch.setattr(M.FIRST, "verify_protocol", lambda *_args, **_kwargs: {})

    def mixed(path, _lab, role):
        if role == "case":
            return {"source_binding": {"checkpoint_sha256": M.sha(cp)}}, {
                "telemetry": "case.json"}, {
                "checkpoint": {"sha256": M.sha(cp)}, "runtime_facts": case_facts,
                "original_faults": ["F1", "R1"],
                "applied_corrections": ["restore_body_frame_reward"],
                "remaining_faults": ["R1"]}
        return {}, {"raw": {}}, {
            "source_arm": "NONE", "original_faults": [], "applied_corrections": [],
            "remaining_faults": [], "runtime_facts": M.RUNTIME.unavailable_facts("reference")}

    monkeypatch.setattr(M, "_mixed_source", mixed)
    material = M._first_material(
        epoch_path, lab=tmp_path, candidate_checker=lambda _path: deepcopy(candidate))
    assert material["checkpoint_sha256"] == M.sha(cp)
    swapped = save(tmp_path / "swapped.pt", {"cp": 2})
    epoch["checkpoint"] = M.ref(swapped)
    swapped_epoch = save(tmp_path / "swapped-epoch.json", epoch)
    with pytest.raises(ValueError, match="receipt/epoch2 input"):
        M._first_material(
            swapped_epoch, lab=tmp_path, candidate_checker=lambda _path: deepcopy(candidate))


def test_loop_protocol_keeps_cap32_targets_and_separate_first_component(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(M, "ROOT", tmp_path)
    monkeypatch.setattr(M, "SOURCE_FILES", ("epoch2.py", "first.py", "transport.py"))
    for name in M.SOURCE_FILES:
        save(tmp_path / name, {"name": name})
    probe = save(tmp_path / "probe-protocol.json", {"schema": M.BASE_PROBE.PROTOCOL_SCHEMA})
    view = save(tmp_path / "view.json", {})
    monkeypatch.setattr(M.DRIVER.C.V, "PROTOCOL", view)
    monkeypatch.setattr(M, "checked_source", lambda source: deepcopy(source))
    seeds = {f"{case}:{policy}": 9000 + index
             for index, (case, policy) in enumerate(M.TARGETS)}
    monkeypatch.setattr(M.FIRST, "ROOT", tmp_path)
    first = first_component_fixture(tmp_path, seeds)
    source = {"case_id": "case-07", "policy": "logs_only"}
    value = M.protocol_value(first, probe, [source], seeds)
    assert value["api_cap"] == 32 and value["automatic_retries"] == 0
    assert value["first_correction_component"]["path"] == "first-protocol.json"
    assert value["source_digests"]["case-07:logs_only"] == M.digest(source)
    assert value["denominator_changed"] is value["new_cases_added"] is False
    assert value["missing_target_success_credit"] is False


def test_loop_protocol_verifier_rechecks_component_anchor_blobs(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(M, "ROOT", tmp_path)
    probe = save(tmp_path / "probe.json", {"schema": M.BASE_PROBE.PROTOCOL_SCHEMA})
    view = save(tmp_path / "view.json", {"schema": "view"})
    loop = save(tmp_path / "loop.json", {})
    monkeypatch.setattr(M.DRIVER.C.V, "PROTOCOL", view)
    seeds = {f"{case}:{policy}": 10 + index
             for index, (case, policy) in enumerate(M.TARGETS)}
    monkeypatch.setattr(M.FIRST, "ROOT", tmp_path)
    first = first_component_fixture(tmp_path, seeds, name="first.json")
    design_path = tmp_path / M.FIRST.CONDITIONAL_DESIGN_RELATIVE
    value = {
        "policies": list(M.POLICIES), "max_probes_per_arm": 3,
        "targets": [f"{case}:{policy}" for case, policy in M.TARGETS],
        "second_seeds": seeds, "source_digests": {},
        "denominator_changed": False, "new_cases_added": False,
        "missing_target_success_credit": False,
        "first_component_authority_inherited": False,
        "first_correction_component": {"path": "first.json", "sha256_lf": M.sha(first, lf=True)},
        "conditional_design": {"path": M.FIRST.CONDITIONAL_DESIGN_RELATIVE,
                               "sha256_lf": M.sha(design_path, lf=True)},
        "probe_protocol": {"path": "probe.json", "sha256_lf": M.sha(probe, lf=True)},
        "view_protocol": {"path": "view.json", "sha256_lf": M.sha(view, lf=True)}}
    monkeypatch.setattr(M.TRANSPORT, "_verify_protocol", lambda *_args: deepcopy(value))
    raw = {"first.json": first.read_bytes(), "probe.json": probe.read_bytes(),
           "view.json": view.read_bytes(), M.FIRST.CONDITIONAL_DESIGN_RELATIVE: design_path.read_bytes()}
    monkeypatch.setattr(M.TRANSPORT, "_git", lambda _repo, _op, spec: raw[spec.split(":", 1)[1]])
    assert M.verify_loop_protocol(loop, "a" * 40)["second_seeds"] == seeds
    raw["view.json"] = b"changed\n"
    with pytest.raises(ValueError, match="anchor selection"):
        M.verify_loop_protocol(loop, "a" * 40)


def source_fixture(tmp_path: Path) -> tuple[Path, dict]:
    epoch = save(tmp_path / "epoch.json", {})
    case = save(tmp_path / "case.json", {})
    reference = save(tmp_path / "reference.json", {})
    candidate = save(tmp_path / "first-candidate.json", {})
    receipt = save(tmp_path / "first-receipt.json", {})
    source = {
        "schema": M.SOURCE_SCHEMA, "case_id": "case-07", "policy": "logs_only",
        "qualification": {"arm": "COMPOUND", "label": "eligible", "diagnosis_eligible": True},
        "packet": {"input_bindings": {"case": {"checkpoint_sha256": "c" * 64}}},
        "backend_packet": {}, "first_correction": "restore_body_frame_reward",
        "original_faults": ["F1", "R1"], "remaining_faults": ["R1"],
        "inputs": {"epoch2_input": M.ref(epoch), "case_collected": M.ref(case),
                   "reference_source_v2": M.ref(reference), "first_candidate": M.ref(candidate),
                   "first_receipt": M.ref(receipt)},
        "providers": None, "visual_evidence": None, "approval_granted": False,
        "api_authorized": False, "lab": str(tmp_path), "source_binding": {}, "index": {}}
    return save(tmp_path / "source.json", source), source


def test_second_candidate_requires_different_registered_actual_abnormal(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source_path, source = source_fixture(tmp_path)
    result = {
        "schema": M.DIAGNOSIS_RESULT_SCHEMA, "scope": "actual_backend",
        "case_id": "case-07", "arm": "logs_only", "source_digest": M.digest(source),
        "end_state": "confirmed", "hold_reason": None,
        "confirmed_subtype": "optimizer_update_count",
        "scoring": {"correct_any_family": True, "wrong_confirmation": False},
        "steps": [{"probe": M.DRIVER.C.RC.NEW_PROBE, "outcome": "abnormal"}]}
    result_path = save(tmp_path / "diagnosis/result.json", result)
    protocol = save(tmp_path / "loop.json", {})
    monkeypatch.setattr(M, "checked_source", lambda value: deepcopy(value))
    monkeypatch.setattr(M, "verify_loop_protocol", lambda *_args, **_kwargs: {
        "source_digests": {"case-07:logs_only": M.digest(source)}})
    final = {"view": "final"}
    monkeypatch.setattr(M.FIRST, "_verify_steps", lambda *_args, **_kwargs: {
        "steps": [], "ledger": {}, "final_snapshot": final})
    monkeypatch.setattr(M.DRIVER.C, "registered_correction", lambda snapshot, applied: (
        "optimizer_update_count", "restore_configured_ppo_epochs"))
    value = M.prepare_second_candidate(
        source_path, result_path, protocol_path=protocol, anchor="a" * 40, lab=tmp_path)
    assert value["status"] == "validated_candidate"
    assert value["second_correction"] == "restore_configured_ppo_epochs"
    assert value["approval_granted"] is value["execution_allowed"] is False
    source["remaining_faults"] = []
    save(source_path, source)
    result["source_digest"] = M.digest(source)
    save(result_path, result)
    with pytest.raises(ValueError, match="original/remaining"):
        M.prepare_second_candidate(
            source_path, result_path, protocol_path=protocol, anchor="a" * 40, lab=tmp_path)
    source["remaining_faults"] = ["R1"]
    save(source_path, source)
    result.update(source_digest=M.digest(source), confirmed_subtype="reward",
                  steps=[{"probe": "P_reward", "outcome": "abnormal"}])
    save(result_path, result)
    monkeypatch.setattr(M.DRIVER.C, "registered_correction", lambda snapshot, applied: (
        "reward", "restore_body_frame_reward"))
    with pytest.raises(ValueError, match="original/remaining"):
        M.prepare_second_candidate(
            source_path, result_path, protocol_path=protocol, anchor="a" * 40, lab=tmp_path)


def test_isolated_driver_fixture_runs_real_controller_ledger_and_r4_registration_key(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base = F.source(tmp_path / "pair")
    packet, backend = deepcopy(base["packet"]), deepcopy(base["backend_packet"])
    packet["budget_left"] = backend["budget_left"] = 3
    epoch = save(tmp_path / "epoch2.json", {"fixture": True})
    source = {
        "schema": M.SOURCE_SCHEMA, "case_id": "case-07", "policy": "fixed_eligible",
        "packet": packet, "backend_packet": backend,
        "qualification": {"arm": "COMPOUND", "label": "eligible",
                          "diagnosis_eligible": True},
        "source_binding": {"fixture": True}, "index": base["prepared"],
        "inputs": {"epoch2_input": M.ref(epoch)}, "lab": str(tmp_path),
        "first_correction": "restore_body_frame_reward",
        "original_faults": ["F1", "R1"], "remaining_faults": ["R1"],
        "providers": None, "visual_evidence": None,
        "approval_granted": False, "api_authorized": False}
    monkeypatch.setattr(M, "_prepare_source_value", lambda *_args, **_kwargs: deepcopy(source))
    protocol = tmp_path / "fixture.json"
    M.DRIVER.BASE.write(protocol, {"fixture": True})
    result = M._DRIVER.run_arm(
        source, arm="fixed_eligible", run_id="r4_epoch2_fixture",
        output=tmp_path / "out", protocol_path=protocol, execution_anchor="a" * 40,
        complete=None, execute_probe=F.fake_probe, verify_probe=F.verify,
        allow_fixture=True)
    assert result["scope"] == "cpu_fixture" and result["steps"]
    assert (tmp_path / "out/probes.ledger.jsonl").is_file()
    assert M._DRIVER.source_registration_key(source, "fixed_eligible") == (
        "case-07:fixed_eligible")
    with pytest.raises(ValueError, match="source/policy"):
        M._DRIVER.source_registration_key(source, "logs_only")


def test_execute_second_candidate_is_explicit_both_fresh_and_has_no_epoch3(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    reference = save(tmp_path / "reference.json", {})
    candidate = {
        "schema": M.SECOND_CANDIDATE_SCHEMA, "status": "validated_candidate",
        "case_id": "case-07", "policy": "logs_only",
        "first_correction": "restore_body_frame_reward",
        "second_correction": "restore_configured_ppo_epochs",
        "original_faults": ["F1", "R1"], "source_checkpoint_sha256": "c" * 64,
        "inputs": {"reference_source_v2": M.ref(reference)}}
    candidate_path = save(tmp_path / "second-candidate.json", candidate)
    protocol_path = save(tmp_path / "loop.json", {})
    monkeypatch.setattr(M, "checked_second_candidate", lambda _path: deepcopy(candidate))
    monkeypatch.setattr(M, "verify_loop_protocol", lambda *_args, **_kwargs: {
        "second_seeds": {"case-07:logs_only": 9907}})
    manifest_call = {}

    def build_manifest(*args, **kwargs):
        manifest_call.update(kwargs)
        return {"manifest": True}

    monkeypatch.setattr(M.TRAINING, "build_manifest", build_manifest)
    evidence = save(tmp_path / "evidence.json", {
        "fresh_initialization": True, "checkpoint_loaded": False, "resume_requested": False,
        "source_checkpoint_sha256": "c" * 64})
    checkpoint = save(tmp_path / "new-model.pt", {"cp": True})
    monkeypatch.setattr(M.TRAINING, "run_training", lambda *_args, **_kwargs: {
        "evidence": M.ref(evidence), "checkpoint": M.ref(checkpoint),
        "collected": None, "runtime_facts": None})
    monkeypatch.setattr(M.TRAINING, "run_fixed", lambda *_args, **_kwargs: (
        {"name": "corrected"}, {"exit_code": 0}))
    fixed = save(tmp_path / "fixed.json", {"report": {"name": "reference"}})
    monkeypatch.setattr(M.SOURCE_INPUT, "checked", lambda _path: ({}, {
        "raw": {"fixed_result": M.ref(fixed)}}))
    monkeypatch.setattr(M.FIRST.GATE.RESIDUAL, "recovery", lambda *_args: {
        "verdict": {"outcome": "recovered"}})
    bands = save(tmp_path / "bands.json", {"bands": {}})
    monkeypatch.setattr(M.FIRST.OLD_RECOVERY, "BANDS", bands)
    lease = {"leaseId": "lease-2"}
    approval = save(tmp_path / "approval.json", {
        "schema": M.SECOND_APPROVAL_SCHEMA, "decision": "approve", "approver": "operator",
        "candidate": M.ref(candidate_path), "protocol": M.ref(protocol_path),
        "anchor": "a" * 40, "registered_seed": 9907, "lease_id": "lease-2"})
    result = M.execute_second_candidate(
        candidate_path, protocol_path, approval, anchor="a" * 40, lab=tmp_path,
        isaac_python=tmp_path / "python", registered_seed=9907, out=tmp_path / "out",
        runner=lambda *_args, **_kwargs: {}, wait=lambda *_args, **_kwargs: None,
        lease_reader=lambda: lease)
    assert manifest_call["arm"] == "BOTH"
    assert manifest_call["applied_corrections"] == [
        "restore_body_frame_reward", "restore_configured_ppo_epochs"]
    assert result["outcome"] == "recovered" and result["epoch3_allowed"] is False
    receipt = M.read(tmp_path / "out/receipt.json")
    assert receipt["source_checkpoint_loaded"] is False
    assert receipt["next_step"] == "stop_no_epoch3_implemented"
