# pyright: reportMissingImports=false
from __future__ import annotations

import json
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import test_selection_comparison_controller_v1 as LEGACY

from evals import evidence_guided_probe_host_v1 as H


def test_legacy_identifier_passes_actual_epoch_binding_contract() -> None:
    state = {"state_digest": "a" * 64}
    identifier = H.legacy_run_id(state)
    bound = H.C.V.epoch_binding(LEGACY.packet(), identifier, "registry_next", LEGACY.VIEW_SHA)
    assert bound["run_id"] == "evidence_guided_" + "a" * 16
    with pytest.raises(ValueError, match="tag"):
        H.C.V.epoch_binding(LEGACY.packet(), "evidence-guided-" + "a" * 16,
                            "registry_next", LEGACY.VIEW_SHA)


def _write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return path


def _facts(case_steps: int = 4, reference_steps: int = 20) -> dict[str, object]:
    result: dict[str, object] = {"telemetry:case:Train/mean_reward": 1.0}
    for role, observed in (("case", case_steps), ("reference", reference_steps)):
        prefix = f"runtime:{role}:"
        result.update({
            prefix + "available": True, prefix + "updates_observed": 3,
            prefix + "expected_optimizer_steps": 20,
            prefix + "optimizer_steps_min": observed,
            prefix + "optimizer_steps_max": observed,
            prefix + "expected_minibatches": 20,
            prefix + "minibatches_min": observed,
            prefix + "minibatches_max": observed,
            f"derived:{role}:step_count_matches_config": observed == 20,
            f"derived:{role}:minibatch_count_matches_config": observed == 20,
        })
    return result


@pytest.fixture
def source_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    checkpoint = {"case": "a" * 64, "reference": "b" * 64}
    runtime_digest = {"case": "c" * 64, "reference": "d" * 64}
    prospective: dict[str, tuple[dict, dict]] = {}
    base_values: dict[str, tuple[dict, dict]] = {}
    runtime_values: dict[str, tuple[dict, dict]] = {}
    paths: dict[str, Path] = {}
    for role in ("case", "reference"):
        root = tmp_path / role
        base = _write(root / "base-collected.json", {"role": role})
        manifest, evidence, owned = root / "manifest.json", root / "evidence", root / "owned.json"
        evidence.mkdir()
        sources = {
            "manifest": manifest, "owned_result": owned,
            "training_evidence": evidence / "training-evidence.json",
            "guard_records": evidence / "guard-records.jsonl",
            "observer_records": evidence / "records.jsonl",
            "checkpoint": evidence / "checkpoint.pt",
            "close_marker": evidence / "close.marker", "readback": evidence / "readback.json",
        }
        for path in sources.values():
            path.write_text(path.name, encoding="utf-8")
        runtime_dir = root / "runtime"
        metadata = _write(runtime_dir / H.RECORDED.FACT.METADATA_NAME, {
            "role": role, "sources": {name: str(path) for name, path in sources.items()},
            "raw_source_sha256": {name: H._sha(path) for name, path in sources.items()},
        })
        runtime = _write(runtime_dir / "artifact.json", {
            "envelope": {"source_binding": {
                "checkpoint_sha256": checkpoint[role], "binding_digest": runtime_digest[role]}},
            "private_metadata_ref": {"file": H.RECORDED.FACT.METADATA_NAME,
                                     "sha256": H._sha(metadata)},
        })
        prospective_path = _write(root / "prospective.json", {"role": role})
        prospective[str(prospective_path.resolve())] = (
            {"source_binding": {"checkpoint_sha256": checkpoint[role]}},
            {"base_collected": {"path": str(base.resolve()), "sha256": H._sha(base)},
             "runtime_facts": {role: {"path": str(runtime.resolve()), "sha256": H._sha(runtime)}}},
        )
        base_values[str(base.resolve())] = (
            {"source_binding": {"checkpoint_sha256": checkpoint[role],
                                "telemetry_sha256": ("e" if role == "case" else "f") * 64,
                                "binding_digest": ("1" if role == "case" else "2") * 64}},
            {"manifest": str(manifest), "evidence_dir": str(evidence),
             "owned_result": str(owned), "private_training": {"seed": 2320, "arm": "R1"}},
        )
        runtime_values[str(runtime.resolve())] = (
            {"source_binding": {"checkpoint_sha256": checkpoint[role],
                                "binding_digest": runtime_digest[role]}},
            {key: value for key, value in _facts().items() if key.startswith(f"runtime:{role}:")},
        )
        paths[f"{role}_input"] = prospective_path
        paths[f"{role}_base"] = base
        paths[f"{role}_runtime"] = runtime

    monkeypatch.setattr(H.INPUT, "checked", lambda path: deepcopy(prospective[str(Path(path).resolve())]))
    monkeypatch.setattr(H.BASE, "checked", lambda path: deepcopy(base_values[str(Path(path).resolve())]))
    monkeypatch.setattr(
        H.RECORDED.FACT, "checked",
        lambda path, **kwargs: deepcopy(runtime_values[str(Path(path).resolve())]),
    )
    fact_ledger = _write(tmp_path / "facts.json", {"facts": True})
    ledger_checked = {
        "source_binding": {
            role: {"checkpoint_sha256": checkpoint[role], "binding_digest": runtime_digest[role]}
            for role in ("case", "reference")},
        "model_view": {"facts": _facts()},
    }
    monkeypatch.setattr(H.SEL.LEDGER, "checked", lambda path: deepcopy(ledger_checked))
    packet = H.SEL.S.build_packet(
        epoch=1, applied_corrections=[], prior_receipt=None,
        facts={"telemetry:case:Train/mean_reward": 1.0},
        input_bindings={role: {
            "checkpoint_sha256": checkpoint[role],
            "telemetry_sha256": ("e" if role == "case" else "f") * 64,
        } for role in ("case", "reference")},
        budget_left=3, execution_budget_s=1200,
    )
    state = H.SEL.initialize(packet, fact_ledger)

    @contextmanager
    def view_scope(*args, **kwargs):
        yield {}

    monkeypatch.setattr(H.RV, "view_scope", view_scope)
    return {**paths, "state": state, "checkpoint": checkpoint}


def _verifier(calls: list[dict]):
    def verify(value: dict) -> dict:
        calls.append(deepcopy(value))
        return {"approved": True}
    return verify


def test_recorded_pair_uses_real_plan_authorize_execute_and_done_ledgers(
    source_fixture: dict[str, Any], tmp_path: Path
) -> None:
    state = H.SEL.advance(source_fixture["state"], H.SEL.select_strong_rule(source_fixture["state"]))
    action = H.SEL.select_strong_rule(state)
    calls: list[dict] = []
    result = H.execute_pair(
        state, action,
        case_input=source_fixture["case_input"], reference_input=source_fixture["reference_input"],
        case_runtime=source_fixture["case_runtime"],
        reference_runtime=source_fixture["reference_runtime"],
        probe_protocol=tmp_path / "unused.json", view_protocol=tmp_path / "view.json",
        anchor="a" * 40, lab=tmp_path / "lab", out=tmp_path / "recorded",
        verify_context=_verifier(calls),
    )
    assert result["scope"] == "recorded_verified_training_cpu_reuse"
    assert result["owned_child_wall_s"] == 0.0
    assert [row["phase"] for row in calls] == [
        "before_role", "before_role", "before_execution", "before_execution"]
    for role in ("case", "reference"):
        plan = Path(result[f"{role}_plan"])
        receipt = Path(result[f"{role}_receipt"])
        checked = H.RECORDED.checked_receipt(plan, receipt)
        assert checked["role"] == role
        assert [row["event"] for row in H.RECORDED._events(
            plan.parent.parent / "ledger.jsonl", checked["request_id"]
        )] == ["proposed", "approved", "consumed", "receipt"]


def test_recorded_rejects_cross_runtime_before_any_plan(
    source_fixture: dict[str, Any], tmp_path: Path
) -> None:
    state = H.SEL.advance(source_fixture["state"], H.SEL.select_strong_rule(source_fixture["state"]))
    action = H.SEL.select_strong_rule(state)
    with pytest.raises(ValueError, match="runtime artifact"):
        H.execute_pair(
            state, action, case_input=source_fixture["case_input"],
            reference_input=source_fixture["reference_input"],
            case_runtime=source_fixture["reference_runtime"],
            reference_runtime=source_fixture["reference_runtime"],
            probe_protocol=tmp_path / "unused", view_protocol=tmp_path / "view",
            anchor="a" * 40, lab=tmp_path / "lab", out=tmp_path / "bad",
            verify_context=lambda value: {"approved": True},
        )
    assert not (tmp_path / "bad").exists()


def test_legacy_pair_builds_real_plans_authorization_settlement_and_four_paths(
    source_fixture: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = source_fixture["state"]
    target = H.SEL.context(state)["eligible_probes"][0]
    action = {"action": "run_probe", "target": target,
              "fact_ids": ["telemetry:case:Train/mean_reward"]}
    source_by_role = {}
    for role in ("case", "reference"):
        base = source_fixture[f"{role}_base"]
        base_value, _metadata = H.BASE.checked(base)
        source_by_role[role] = {
            "kind": "registered_training_observation", "seed": 2320,
            "source_arm": "R1" if role == "case" else "NONE",
            "checkpoint": {"path": str(tmp_path / f"{role}.pt"),
                           "sha256": base_value["source_binding"]["checkpoint_sha256"]},
            "telemetry_sha256": base_value["source_binding"]["telemetry_sha256"],
            "binding_digest": ("c" if role == "case" else "d") * 64,
            "source_manifest_sha256": "9" * 64,
            "runtime_facts": {key: value for key, value in _facts().items()
                              if key.startswith(f"runtime:{role}:")},
            "original_faults": ["R1"] if role == "case" else [],
            "applied_corrections": [], "remaining_faults": ["R1"] if role == "case" else [],
            "readback": str(tmp_path / f"{role}-readback.json"),
        }
        Path(source_by_role[role]["checkpoint"]["path"]).write_bytes(role.encode())
    # 실제 plan/authorization/ledger 소비를 검사하되 Isaac 원문 검증은 source fixture로 격리한다.
    monkeypatch.setattr(H, "_source", lambda path, runtime, role, spec: (
        source_fixture[f"{role}_base"], source_fixture[f"{role}_runtime"]))
    monkeypatch.setattr(H.OBS, "verify_protocol", lambda *args: {
        "component_probe_protocol": {
            "path": "evals/runtime_observation_probe_v1.py", "sha256_lf": "0" * 64}})
    monkeypatch.setattr(H.OBS.P1, "verify_protocol", lambda *args: {
        "isaac_python_relative": "python.bat", "watchdogs_s": {"legacy": 10, "optimizer": 10},
        "source_sha256_lf": {}, "runtime_source_sha256_lf": {},
    })
    monkeypatch.setattr(H.OBS, "_source", lambda path, lab, role: ({}, {}, deepcopy(source_by_role[role])))
    monkeypatch.setattr(H.OBS, "_verify_auxiliary_inputs", lambda plan: None)
    monkeypatch.setattr(H.OBS.P1, "_auxiliary_inputs", lambda probe, checkpoint: {})
    monkeypatch.setattr(H.OBS, "_effective_fault", lambda source: "NONE")
    monkeypatch.setattr(H.OBS.POST, "_admit", lambda *args: {})
    monkeypatch.setattr(H.C.V, "checked", lambda snapshot: deepcopy(snapshot))
    monkeypatch.setattr(H.C.V, "build_snapshot", lambda *args, **kwargs: {
        "schema": "selection_snapshot_v1", "inputs": {"packet": state["initial_packet"],
        "ledger_path": str((tmp_path / "legacy/probes.ledger.jsonl").resolve())},
        "binding": {"epoch": 1}, "view": {"eligible_probes": [target], "status": "open"},
        "view_digest": "1" * 64})
    monkeypatch.setattr(H.C, "choose", lambda snapshot, policy: {
        "context_digest": H.C.L.digest(snapshot["binding"]), "view_digest": snapshot["view_digest"],
        "policy": policy, "selected_by": "registered_rule", "action": "continue",
        "probe": target, "fact_ids": [], "model_status": "not_used", "codes": [],
        "reason": "registered_policy", "used_for_decision": True, "approval_granted": False})
    monkeypatch.setattr(H.C, "request_spec", lambda snapshot, choice: {
        "prereg": snapshot["binding"], "probe": target,
        "args": deepcopy(H.C.RC.registry()["probes"][target]["args"]),
        "checkpoint_sha256": source_by_role["case"]["checkpoint"]["sha256"],
        "reference_checkpoint_sha256": source_by_role["reference"]["checkpoint"]["sha256"],
        "budget_gpu_s": 1200, "approval_granted": False})
    monkeypatch.setattr(H.C, "_frozen_request_spec", H.C.request_spec)

    def execute(plan: Path, authorization: Path, **kwargs) -> dict:
        value = H.OBS.read(plan)
        receipt = {
            "status": "complete", "measurement": {"probe_value": 1.0},
            "execution": {"wall_s": 1.25}, "role": value["role"], "probe": target,
        }
        H.OBS.write_new(Path(value["stage"]) / "probe_receipt.json", receipt)
        return receipt

    monkeypatch.setattr(H.OBS, "execute_probe", execute)
    monkeypatch.setattr(H.OBS, "verify_receipt", lambda plan, receipt: H.OBS.read(receipt))
    probe_protocol = _write(tmp_path / "probe.json", {})
    view_protocol = _write(tmp_path / "view.json", {})
    calls: list[dict] = []
    result = H.execute_pair(
        state, action, case_input=source_fixture["case_input"],
        reference_input=source_fixture["reference_input"],
        case_runtime=source_fixture["case_runtime"],
        reference_runtime=source_fixture["reference_runtime"],
        probe_protocol=probe_protocol, view_protocol=view_protocol,
        anchor="a" * 40, lab=tmp_path / "lab", out=tmp_path / "legacy",
        verify_context=_verifier(calls),
    )
    assert result["scope"] == "legacy_registered_paired_probe"
    assert result["owned_child_wall_s"] == 2.5
    assert all(Path(result[key]).is_file() for key in (
        "case_plan", "case_receipt", "reference_plan", "reference_receipt"))
    ledger = H.C.S.M.RegistryLedger(tmp_path / "legacy/probes.ledger.jsonl", H.C.RC.registry())
    assert [row["event"] for row in ledger.events()] == [
        "proposed", "approved", "consumed", "receipt"]


def test_context_denial_stops_before_authorization(
    source_fixture: dict[str, Any], tmp_path: Path
) -> None:
    state = H.SEL.advance(source_fixture["state"], H.SEL.select_strong_rule(source_fixture["state"]))
    action = H.SEL.select_strong_rule(state)
    with pytest.raises(ValueError, match="승인되지 않았다"):
        H.execute_pair(
            state, action, case_input=source_fixture["case_input"],
            reference_input=source_fixture["reference_input"],
            case_runtime=source_fixture["case_runtime"],
            reference_runtime=source_fixture["reference_runtime"],
            probe_protocol=tmp_path / "probe", view_protocol=tmp_path / "view",
            anchor="a" * 40, lab=tmp_path / "lab", out=tmp_path / "denied",
            verify_context=lambda value: {"approved": False},
        )
    assert not (tmp_path / "denied/root-approval.json").exists()
