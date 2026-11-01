# pyright: reportMissingImports=false
from __future__ import annotations

import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("bench", "evals", "src")]

import five_policy_recovery_candidate_v1 as M


def save(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def conditional_fixture(tmp_path: Path, first_seeds: dict, second_seeds: dict | None = None) -> Path:
    value = json.loads((ROOT / M.CONDITIONAL_DESIGN_RELATIVE).read_text(encoding="utf-8"))
    study = save(tmp_path / "study.json", {"fixture": True})
    value["study_design"] = {"path": "study.json", "sha256_lf": M.sha(study, lf=True)}
    value["first_seeds"] = first_seeds
    value["second_seeds"] = second_seeds or {
        key: 17000 + index for index, key in enumerate(first_seeds)
    }
    return save(tmp_path / M.CONDITIONAL_DESIGN_RELATIVE, value)


def candidate_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *,
                      case_id: str = "case-03", policy: str = "fixed_eligible",
                      family: str | None = "reward", hold_reason: str | None = None) -> dict:
    arm = M.TARGETS[(case_id, policy)]
    index = save(tmp_path / "index.json", {"schema": "source-index"})
    case_input = save(tmp_path / "case-input.json", {"schema": "bridge"})
    reference_input = save(tmp_path / "reference-input.json", {"schema": "bridge"})
    source = {
        "packet": {"input_bindings": {}}, "index": M.ref(index),
        "qualification": {"arm": arm, "label": "eligible", "diagnosis_eligible": True},
    }
    sources = {case_id: {"source": source,
                         "inputs": {"case": case_input, "reference": reference_input}}}
    protocol = save(tmp_path / "five.json", {"schema": M.DRIVER.SCHEMA})
    config = save(tmp_path / "config.json", {"schema": "config"})
    row_result = {
        "schema": M.DRIVER.RESULT_SCHEMA, "scope": "actual_backend", "case_id": case_id,
        "arm": policy, "source_digest": M.DRIVER.BASE.digest(source),
        "qualification": source["qualification"], "host_storage_policy": policy,
        "run_id": "fixture", "steps": [], "hold_reason": hold_reason,
        "end_state": "confirmed" if family is not None and hold_reason is None else "HOLD",
        "confirmed_subtype": family,
        "scoring": {"correct_any_family": family in M.CORRECTIONS,
                    "wrong_confirmation": family not in M.CORRECTIONS if family else False},
    }
    result_path = save(tmp_path / "row/result.json", row_result)
    rows = [{"case_id": case_id, "arm": policy, "result": M.ref(result_path)}]
    rows.extend({"case_id": f"other-{index}", "arm": "other"} for index in range(49))
    batch = {"schema": "lineage_five_policy_batch_result_v1", "status": "complete",
             "planned_cases": 10, "planned_rows": 50, "rows": rows,
             "repair": False, "recovery": False}
    batch_path = save(tmp_path / "batch.json", batch)
    case_raw = {"manifest": {"arm": arm, "seed": 2552 if arm == "F1" else 2556},
                "raw": {"protocol": M.ref(save(tmp_path / "source-protocol.json", {}))}}
    reference_raw = {"manifest": {"arm": "NONE", "seed": case_raw["manifest"]["seed"]}}
    bridges = {
        str(case_input): ({"index": M.ref(index), "binding": {"checkpoint_sha256": "c" * 64}}, case_raw),
        str(reference_input): ({"index": M.ref(index), "binding": {"checkpoint_sha256": "r" * 64}},
                               reference_raw),
    }
    monkeypatch.setattr(M.SOURCE_INPUT, "checked", lambda path: deepcopy(bridges[str(path)]))
    monkeypatch.setattr(M, "_verify_steps", lambda *_args, **_kwargs: {
        "steps": [], "ledger": None, "final_snapshot": {"fixture": True}})
    monkeypatch.setattr(M.DRIVER.C, "registered_correction",
                        lambda _snapshot, applied: (family, M.CORRECTIONS.get(family)))

    def preflight(_protocol, _anchor, _config):
        return {}, {}, sources

    return {"batch": batch_path, "protocol": protocol, "config": config,
            "preflight": preflight, "case_id": case_id, "policy": policy}


@pytest.mark.parametrize(("case_id", "policy", "family", "correction", "faults"), [
    ("case-03", "fixed_eligible", "reward", "restore_body_frame_reward", ["F1"]),
    ("case-07", "logs_only", "optimizer_update_count",
     "restore_configured_ppo_epochs", ["F1", "R1"]),
])
def test_actual_confirmed_family_builds_unapproved_first_candidate(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case_id: str, policy: str,
        family: str, correction: str, faults: list[str]) -> None:
    fixture = candidate_fixture(tmp_path, monkeypatch, case_id=case_id, policy=policy, family=family)
    value = M.prepare_candidate(
        fixture["batch"], protocol_path=fixture["protocol"], config_path=fixture["config"],
        anchor="a" * 40, case_id=case_id, policy=policy, preflight=fixture["preflight"])
    assert value["status"] == "validated_candidate" and value["correction"] == correction
    assert value["original_faults"] == faults
    assert value["approval_granted"] is value["execution_allowed"] is False
    assert value["source_checkpoint_role"] == "provenance_only_never_loaded"


@pytest.mark.parametrize(("family", "reason"), [
    (None, "probe_outcome_unknown"), ("actuator", None), (None, "model_hold"),
])
def test_hold_unknown_or_wrong_family_preserves_hold_without_replacement(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        family: str | None, reason: str | None) -> None:
    fixture = candidate_fixture(tmp_path, monkeypatch, family=family, hold_reason=reason)
    value = M.prepare_candidate(
        fixture["batch"], protocol_path=fixture["protocol"], config_path=fixture["config"],
        anchor="a" * 40, case_id=fixture["case_id"], policy=fixture["policy"],
        preflight=fixture["preflight"])
    assert value["status"] == "HOLD" and value["correction"] is None
    assert value["replacement_allowed"] is value["replacement_used"] is False
    assert value["training_executed"] is False


def test_unregistered_case_or_policy_is_rejected_before_preflight(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="사전 고정"):
        M.prepare_candidate(tmp_path / "missing", protocol_path=tmp_path / "p",
                            config_path=tmp_path / "c", anchor="a" * 40,
                            case_id="case-04", policy="fixed_eligible")


def test_historical_snapshot_uses_host_ledger_prefix_not_current_snapshot(
        tmp_path: Path) -> None:
    packet = M.DRIVER.C.S.build_packet(
        epoch=1, applied_corrections=[], prior_receipt=None,
        facts={"config:action_scale": 0.25},
        input_bindings={"case": {"checkpoint_sha256": "1" * 64, "telemetry_sha256": "2" * 64},
                        "reference": {"checkpoint_sha256": "3" * 64,
                                      "telemetry_sha256": "4" * 64}},
        budget_left=3, execution_budget_s=1200)
    root, ledger = tmp_path / "row", tmp_path / "row/probes.ledger.jsonl"
    before = M.DRIVER.C.V.build_snapshot(
        packet, "prefix_test", "fixed_eligible",
        M.DRIVER.C.V.file_digest(M.DRIVER.C.V.PROTOCOL), ledger)
    choice = M.DRIVER.C.choose(before, "fixed_eligible")
    spec = M.DRIVER.C.request_spec(before, choice)
    context = M.DRIVER.A.prepare(
        before, goal=M.DRIVER.BASE.P.PAIR_GOAL, decisions_left=3)
    registry = M.DRIVER.C.S.M.RegistryLedger(ledger, M.DRIVER.C.RC.registry())
    request = registry.propose(spec["prereg"], spec["probe"], spec["budget_gpu_s"])
    registry.approve(request["request_id"], "fixture")
    registry.consume(request["request_id"], spec["prereg"], spec["probe"])
    registry.receipt(request["request_id"], "normal", {}, 1.0, 0)
    prefix = {"byte_length": len(ledger.read_bytes()),
              "sha256": hashlib.sha256(ledger.read_bytes()).hexdigest()}
    save(root / "step01/context.json", context)
    save(root / "step01/choice.json", choice)
    host_result = {"public_ledger": {"path": str(ledger), "prefix": prefix},
                   "public_choice": choice, "public_snapshot_sha256": M.DRIVER.BASE.digest(before)}
    host_path = save(root / "step01/probe/result.json", host_result)
    final = M.DRIVER.C.V.build_snapshot(
        packet, "prefix_test", "fixed_eligible",
        M.DRIVER.C.V.file_digest(M.DRIVER.C.V.PROTOCOL), ledger)
    result = {"host_storage_policy": "fixed_eligible", "run_id": "prefix_test",
              "arm": "fixed_eligible", "final_view_digest": final["view_digest"],
              "steps": [{"step": 1, "probe": spec["probe"], "outcome": "normal",
                         "probe_result": M.ref(host_path), "model_call": None}]}
    result_path = save(root / "result.json", result)
    calls = []

    def verify(path, snapshot, selected):
        calls.append((path, snapshot, selected))
        return {"probe": spec["probe"], "outcome": "normal"}

    checked = M._verify_steps(result_path, result, {"packet": packet}, host_verify=verify)
    assert calls[0][1] == before and checked["ledger"]["final_prefix"] == prefix


def protocol_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path]:
    monkeypatch.setattr(M, "ROOT", tmp_path)
    monkeypatch.setattr(M, "SOURCE_FILES", ("bridge.py",))
    monkeypatch.setattr(M.TRAINING.TRAIN, "BOUND_SOURCES", frozenset())
    monkeypatch.setattr(M.SOURCE_INPUT.BASE_PROBE, "RUNTIME_PINS", {"runtime.py"})
    save(tmp_path / "bridge.py", {})
    fixed = save(tmp_path / "fixed.json", {})
    bands = save(tmp_path / "bands.json", {"bands": {}})
    monkeypatch.setattr(M.OLD_RECOVERY, "FIXED_PROTOCOL", fixed)
    monkeypatch.setattr(M.OLD_RECOVERY, "BANDS", bands)
    base = save(tmp_path / "base-probe.json", {"runtime_source_sha256_lf": {"runtime.py": "9" * 64}})
    save(tmp_path / "paired.json", {
        "component_probe_protocol": {"path": "base-probe.json", "sha256_lf": M.sha(base, lf=True)}})
    five = save(tmp_path / "five.json", {"schema": M.DRIVER.SCHEMA,
                "tool_protocols": {"paired_probe": {"path": "paired.json"}},
                "study_design": {"path": "study.json", "sha256_lf": M.sha(
                    save(tmp_path / "study.json", {"fixture": True}), lf=True)}})
    source = save(tmp_path / "source.json", {"schema": M.SOURCE_BATCH.SCHEMA})
    conditional_fixture(tmp_path, {f"{case}:{policy}": 7000 + index
                        for index, (case, policy) in enumerate(M.TARGETS)})
    return five, source, fixed


def test_protocol_generation_binds_four_registered_seeds_and_runtime9_shape(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    five, source, fixed = protocol_fixture(tmp_path, monkeypatch)
    seeds = {f"{case}:{policy}": 7000 + index
             for index, (case, policy) in enumerate(M.TARGETS)}
    value = M.prepare_protocol(five, source, seeds, fixed)
    assert value["recovery_seeds"] == dict(sorted(seeds.items()))
    assert value["training"] == {"arm": "FIRST", "mode": "fresh_restart", "updates": 300,
                                  "source_checkpoint": "provenance_only", "runner_load_calls": 0}
    assert set(value["runtime_source_sha256_lf"]) == {"runtime.py"}
    assert value["authority"]["training"] is False
    assert value["components"]["conditional_design"]["path"] == M.CONDITIONAL_DESIGN_RELATIVE


def test_changed_first_seed_is_rejected_before_recovery_registration(tmp_path, monkeypatch):
    five, source, fixed = protocol_fixture(tmp_path, monkeypatch)
    seeds = {f"{case}:{policy}": 7000 + index
             for index, (case, policy) in enumerate(M.TARGETS)}
    seeds["case-03:logs_only"] += 500
    with pytest.raises(ValueError, match="first seed"):
        M.prepare_protocol(five, source, seeds, fixed)


@pytest.mark.parametrize(("field", "changed"), [
    ("second_requires_valid_not_recovered_receipt", False),
    ("case_replacement", True), ("epoch3", True),
    ("second_gpu_phase", {"sdk_cap": 32, "api_cap": 33, "wall_cap_s": 43200}),
])
def test_conditional_branch_cannot_weaken_after_first_result(tmp_path, monkeypatch, field, changed):
    monkeypatch.setattr(M, "ROOT", tmp_path)
    seeds = {f"{case}:{policy}": 7000 + index
             for index, (case, policy) in enumerate(M.TARGETS)}
    path = conditional_fixture(tmp_path, seeds)
    value = M.read(path)
    value[field] = changed
    save(path, value)
    with pytest.raises(ValueError, match="분기·실행 한도"):
        M.conditional_design(path, first_seeds=seeds)


@pytest.mark.parametrize("outcome", ["recovered", "not_recovered"])
def test_execute_reuses_training_fixed_and_stops_or_emits_epoch2_input(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: str) -> None:
    monkeypatch.setattr(M, "ROOT", tmp_path)
    source_protocol = save(tmp_path / "source.json", {})
    protocol_path = save(tmp_path / "recovery.json", {})
    candidate = {
        "schema": M.SCHEMA, "status": "validated_candidate", "case_id": "case-03",
        "policy": "fixed_eligible", "source_seed": 2552, "original_faults": ["F1"],
        "correction": "restore_body_frame_reward", "confirmed_family": "reward",
        "source_checkpoint_sha256": "c" * 64,
        "inputs": {"source_protocol": M.ref(source_protocol)},
    }
    candidate_path = save(tmp_path / "candidate.json", candidate)
    protocol = {"recovery_seeds": {"case-03:fixed_eligible": 7001},
                "components": {"source_collection": {"path": "source.json"}}}
    monkeypatch.setattr(M, "checked_candidate", lambda *_args, **_kwargs: deepcopy(candidate))
    monkeypatch.setattr(M, "verify_protocol", lambda *_args, **_kwargs: deepcopy(protocol))
    monkeypatch.setattr(M.TRAINING, "build_manifest", lambda *_args, **_kwargs: {"manifest": True})
    evidence_path = save(tmp_path / "training-evidence.json", {
        "fresh_initialization": True, "checkpoint_loaded": False, "resume_requested": False,
        "source_checkpoint_sha256": "c" * 64, "remaining_faults": [],
    })
    checkpoint = {"path": str(tmp_path / "model_299.pt"), "sha256": "d" * 64}
    collected = save(tmp_path / "collected.json", {})
    runtime = save(tmp_path / "runtime.json", {})
    monkeypatch.setattr(M.TRAINING, "run_training", lambda *_args, **_kwargs: {
        "evidence": M.ref(evidence_path), "checkpoint": checkpoint,
        "collected": M.ref(collected), "runtime_facts": {"case": M.ref(runtime)}})
    monkeypatch.setattr(M.TRAINING, "run_fixed", lambda *_args, **_kwargs: (
        {"name": "corrected"}, {"exit_code": 0}))
    fixed_result = save(tmp_path / "fixed-result.json", {"report": {"name": "reference"}})
    reference_input = save(tmp_path / "reference-input.json", {})
    candidate["inputs"]["reference_probe_input"] = M.ref(reference_input)
    save(candidate_path, candidate)
    monkeypatch.setattr(M.SOURCE_INPUT, "checked", lambda _path: ({}, {
        "raw": {"fixed_result": M.ref(fixed_result)}}))
    monkeypatch.setattr(M.GATE.RESIDUAL, "recovery", lambda *_args: {
        "verdict": {"outcome": outcome}})
    bands = save(tmp_path / "bands.json", {"bands": {}})
    monkeypatch.setattr(M.OLD_RECOVERY, "BANDS", bands)
    lease = {"leaseId": "lease-1"}
    approval_path = save(tmp_path / "approval.json", {
        "schema": M.APPROVAL_SCHEMA, "decision": "approve", "approver": "operator",
        "candidate": M.ref(candidate_path), "protocol": M.ref(protocol_path),
        "anchor": "a" * 40, "registered_seed": 7001, "lease_id": "lease-1"})
    result = M.execute_candidate(
        candidate_path, protocol_path, approval_path, anchor="a" * 40,
        lab=tmp_path, isaac_python=tmp_path / "python", registered_seed=7001,
        out=tmp_path / "out", runner=lambda *_: {}, wait=lambda *_args, **_kwargs: None,
        lease_reader=lambda: lease)
    assert result["outcome"] == outcome
    assert (tmp_path / "out/epoch2-input.json").exists() is (outcome == "not_recovered")
    if outcome == "not_recovered":
        epoch2 = M.read(tmp_path / "out/epoch2-input.json")
        assert epoch2["new_protocol_required"] is True
        assert epoch2["execution_allowed"] is epoch2["automatic_second_execution"] is False
