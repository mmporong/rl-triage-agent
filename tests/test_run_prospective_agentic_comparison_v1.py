# pyright: reportMissingImports=false
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "bench")]

import run_prospective_agentic_comparison_v1 as RUN


def write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return path


def ref(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def test_prepare_protocol_locks_registered_design(monkeypatch: pytest.MonkeyPatch,
                                                   tmp_path: Path) -> None:
    study = {
        "schema": "evidence_guided_study_v1", "seeds": [2320, 2321],
        "categories": ["NONE", "F1", "R1_ONLY", "COMPOUND"],
        "comparison": {"policies": list(RUN.COMP.POLICIES), "paired_probe_budget": 3,
                       "model_action_budget": 4, "semantic_feedback_retries_per_action": 1,
                       "global_api_request_cap": 96},
        "source_protocol": "training.json",
        "repair_followup": {"automatic_repair": False},
    }
    training = {"schema": RUN.TRAIN.SCHEMA, "seeds": study["seeds"],
                "categories": study["categories"], "runtime_source_sha256_lf": {"x": "a" * 64}}
    files = {
        "study": write(tmp_path / "study.json", study),
        "training": write(tmp_path / "training.json", training),
        "probe": write(tmp_path / "probe.json", {"schema": RUN.OBS.PROTOCOL_SCHEMA}),
        "view": write(tmp_path / "view.json", {"schema": RUN.RV.SCHEMA}),
        "transport": write(tmp_path / "transport.json", {"schema": RUN.T4.SCHEMA,
            "plan": {"case_ids": [f"c{i:02d}" for i in range(1, 9)],
                     "global_request_cap": 96, "per_slot_max": 3,
                     "logical_slots": [{"case_id": f"c{case:02d}",
                       "slot_id": f"c{case:02d}.{policy}.d{action}"}
                       for case in range(1, 9) for policy in RUN.COMP.POLICIES[:2]
                       for action in range(1, 5)]}}),
    }
    monkeypatch.setattr(RUN, "SOURCE_FILES", ())
    monkeypatch.setattr(RUN, "_public_ref", lambda path: {
        "path": Path(path).name, "sha256_lf": RUN._sha(path, lf=True)})
    value = RUN.prepare_protocol(files["study"], files["training"], files["probe"],
                                 files["view"], files["transport"])
    assert value["design"]["opaque_cases"] == [f"c{i:02d}" for i in range(1, 9)]
    assert value["design"]["policy_order"][1] == list(RUN.COMP.policy_order(1))
    assert value["bounds"] == {"total_wall_s": 14400, "global_api_requests": 96,
                               "paired_probes": 72, "legacy_role_executions_worst_case": 144}
    study["comparison"]["paired_probe_budget"] = 4
    write(tmp_path / "bad-study.json", study)
    with pytest.raises(ValueError, match="상한"):
        RUN.prepare_protocol(tmp_path / "bad-study.json", files["training"], files["probe"],
                             files["view"], files["transport"])


def source_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                   *, bad_input_binding: bool = False) -> tuple[Path, dict]:
    protocol_file = write(tmp_path / "training-protocol.json", {"schema": "x"})
    study = write(tmp_path / "study.json", {
        "schema": "evidence_guided_study_v1", "seeds": [2320, 2321],
        "categories": ["NONE", "F1", "R1_ONLY", "COMPOUND"],
        "comparison": {"policies": list(RUN.COMP.POLICIES), "paired_probe_budget": 3,
                       "model_action_budget": 4, "semantic_feedback_retries_per_action": 1,
                       "global_api_request_cap": 96},
        "repair_followup": {"automatic_repair": False},
    })
    protocol = {"protocols": {"study": {"path": study.name},
                              "training": {"path": protocol_file.name}}}
    smoke = write(tmp_path / "smoke.json", {"schema": "prospective_diagnosis_smoke_v1",
                                             "status": "complete"})
    stages: list[dict] = [{"stage": "smoke", "result": ref(smoke)}]
    metadata: dict[str, dict] = {}
    for seed in (2320, 2321):
        for category in ("NONE", "F1", "R1_ONLY", "COMPOUND"):
            label = f"s{seed}-{category.lower()}"
            checkpoint = {"path": f"{label}.pt", "sha256": hashlib.sha256(label.encode()).hexdigest()}
            binding = {"checkpoint_sha256": checkpoint["sha256"]}
            manifest = write(tmp_path / label / "manifest.json", {"source_commit_sha": "a" * 40})
            training = write(tmp_path / label / "training.json", {
                "manifest": ref(manifest), "execution": {"exit_code": 0,
                    "natural_exit": True, "cleanup_verified": True, "timed_out": False}})
            fixed = write(tmp_path / label / "fixed.json", {"name": label})
            result = write(tmp_path / label / "result.json", {
                "schema": RUN.TRAIN.RESULT_SCHEMA, "status": "complete", "seed": seed,
                "category": category, "training": ref(training), "checkpoint": checkpoint,
                "fixed_report": ref(fixed), "fixed_execution": {"exit_code": 0,
                    "natural_exit": True, "cleanup_verified": True, "timed_out": False},
                "report": {"name": label}, "claims": {"recovery": False,
                    "diagnosis": False, "model_advantage": False},
            })
            collected_path = write(tmp_path / label / "collected.json", {"label": label})
            metadata[str(collected_path.resolve())] = {
                "collected": {"source_binding": binding},
                "metadata": {"runtime_facts": {
                    "case": {"path": str(fixed)}, "reference": {"path": str(fixed)}}},
            }
            stages.append({"stage": label, "result": ref(result), "checkpoint": checkpoint,
                           "input": ref(collected_path),
                           "input_binding": ({"checkpoint_sha256": "0" * 64}
                                             if bad_input_binding and label == "s2320-f1" else binding)})
    batch = write(tmp_path / "batch.json", {
        "schema": RUN.SOURCE.BATCH_SCHEMA, "status": "complete", "anchor": "a" * 40,
        "protocol": ref(protocol_file), "seeds": [2320, 2321],
        "categories": ["NONE", "F1", "R1_ONLY", "COMPOUND"], "stages": stages,
        "automatic_retry": False, "model_requests": 0, "repairs": 0,
    })
    monkeypatch.setattr(RUN.INPUT, "checked", lambda path: (
        metadata[str(Path(path).resolve())]["collected"], metadata[str(Path(path).resolve())]["metadata"]))
    return batch, protocol


def test_source_inventory_rejects_rebound_input(monkeypatch: pytest.MonkeyPatch,
                                                 tmp_path: Path) -> None:
    batch, protocol = source_fixture(tmp_path, monkeypatch, bad_input_binding=True)
    monkeypatch.setattr(RUN, "ROOT", tmp_path)
    # ROOT 변경 뒤 protocol 상대 경로를 fixture root 기준으로 다시 쓴다.
    protocol["protocols"]["study"]["path"] = "study.json"
    protocol["protocols"]["training"]["path"] = "training-protocol.json"
    with pytest.raises(ValueError, match="input binding"):
        RUN._source_cases(batch, protocol, "b" * 40)


def test_source_inventory_rejects_resealed_wrong_source_anchor(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    batch, protocol = source_fixture(tmp_path, monkeypatch)
    manifest_path = tmp_path / "s2320-f1" / "manifest.json"
    training_path = tmp_path / "s2320-f1" / "training.json"
    result_path = tmp_path / "s2320-f1" / "result.json"
    write(tmp_path / "replacement-manifest.json", {"source_commit_sha": "b" * 40})
    replacement = tmp_path / "replacement-manifest.json"
    training = json.loads(training_path.read_text())
    training["manifest"] = ref(replacement)
    training_path.write_text(json.dumps(training) + "\n", encoding="utf-8")
    result = json.loads(result_path.read_text())
    result["training"] = ref(training_path)
    result_path.write_text(json.dumps(result) + "\n", encoding="utf-8")
    batch_value = json.loads(batch.read_text())
    batch_value["stages"][2]["result"] = ref(result_path)
    batch.write_text(json.dumps(batch_value) + "\n", encoding="utf-8")
    assert manifest_path.is_file()
    monkeypatch.setattr(RUN, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="anchor"):
        RUN._source_cases(batch, protocol, "c" * 40)


def test_gate_keeps_fault_absence_in_denominator(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(RUN.C.D1.V.R1.U, "walks", lambda report: True)
    monkeypatch.setattr(RUN.C.D1, "recovery", lambda report, reference, bands: {
        "outcome": "recovered"})
    reference = {"fixed_report": {"kind": "none"}, "category": "NONE"}
    fault = {"fixed_report": {"kind": "r1"}, "category": "R1_ONLY"}
    gate = RUN._gate(fault, reference, {})
    assert gate == {"run": False, "reason": "healthy_gate", "source_outcome": "recovered",
                    "analysis_exclusion": "source_failure_absent"}


def test_score_does_not_promote_compound_partial() -> None:
    partial = RUN._score("COMPOUND", {"confirmed_family": "optimizer_update_count",
                                       "decision_outcome": "confirmed"})
    assert partial["compound_partial"] is True
    assert partial["correct_confirmed_family"] is True
    assert partial["compound_complete"] is False
    assert partial["correct_complete_family"] is False
    assert RUN._score("R1_ONLY", {"confirmed_family": "optimizer_update_count",
                                  "decision_outcome": "confirmed"})["correct_complete_family"] is True


def test_validity_separates_first_pass_and_feedback(tmp_path: Path) -> None:
    folder = tmp_path / "epoch"
    dialogue = {"calls": [{"review": {"status": "invalid"}},
                           {"review": {"status": "valid"}}],
                "model_valid": True, "semantic_feedback_used": True}
    path = write(folder / "step-01" / "dialogue.json", dialogue)
    epoch = {"steps": [{"dialogue_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}]}
    assert RUN._validity(folder, epoch) == {"first_pass_valid_actions": 0,
                                            "post_feedback_valid_actions": 1,
                                            "semantic_feedback_actions": 1}
    dialogue["model_valid"] = False
    write(folder / "step-02" / "dialogue.json", dialogue)
    epoch["steps"].append({"dialogue_sha256": "0" * 64})
    with pytest.raises(ValueError, match="SHA"):
        RUN._validity(folder, epoch)


def test_context_verifier_passes_actual_host_context_only_after_role_binding(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source = write(tmp_path / "source.json", {"anchor": "a" * 40})
    runtime = write(tmp_path / "runtime.json", {"runtime": True})
    prospective = write(tmp_path / "prospective.json", {"input": True})
    checkpoint, binding_digest = "b" * 64, "c" * 64
    monkeypatch.setattr(RUN, "_verify_current_protocol", lambda *_args: None)
    monkeypatch.setattr(RUN, "_lease", lambda: {"owner": RUN.OWNER, "lease_id": "lease"})
    monkeypatch.setattr(RUN.T4, "status", lambda _root: {"requests": 0})
    monkeypatch.setattr(RUN.INPUT, "checked", lambda _path: (
        {"source_binding": {"checkpoint_sha256": checkpoint}},
        {"runtime_facts": {"case": ref(runtime)}}))
    monkeypatch.setattr(RUN.HOST.RECORDED.FACT, "checked", lambda *_args, **_kwargs: (
        {"source_binding": {"binding_digest": binding_digest}}, {}))
    active = {"case": {"input": prospective, "runtime": runtime,
                       "checkpoint_sha256": checkpoint}}
    callback = RUN._context_verifier(
        protocol_path=tmp_path / "protocol.json", protocol={}, lab=tmp_path,
        source_batch_result=source, source_sha=RUN._sha(source), lease_id="lease",
        transport_root=tmp_path / "transport", active_inputs=active,
        started=RUN.time.monotonic())
    spec = {"source_binding": {"case": {"checkpoint_sha256": checkpoint,
                                          "binding_digest": binding_digest}}}
    approved = RUN.HOST._context(
        callback, phase="before_role", role="case", state={"state_digest": "d" * 64},
        action={"action": "run_probe"}, spec=spec)
    assert approved["approved"] is True and approved["checked_role"] == "case"
    spec["source_binding"]["case"]["checkpoint_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="checkpoint"):
        RUN.HOST._context(
            callback, phase="before_role", role="case", state={"state_digest": "d" * 64},
            action={"action": "run_probe"}, spec=spec)


def test_execution_preserves_24_row_latin_denominator(monkeypatch: pytest.MonkeyPatch,
                                                       tmp_path: Path) -> None:
    protocol_path = write(tmp_path / "protocol.json", {"fixture": True})
    source_path = write(tmp_path / "source.json", {"fixture": True, "anchor": "0" * 40})
    runtime = write(tmp_path / "runtime.json", {"fixture": True})
    protocol = {"protocols": {
        "paired_probe": {"path": "probe.json"}, "view": {"path": "view.json"}},
        "claims": {"model_version_advantage": False, "recovery": False,
                   "unseen_family_generalization": False, "gpu_started_by_parent": False}}
    categories = ("NONE", "F1", "R1_ONLY", "COMPOUND") * 2
    cases = [{"case_id": f"c{index:02d}", "seed": 2320 if index <= 4 else 2321,
              "category": category, "input": runtime,
              "checkpoint": {"sha256": f"{index:064x}"}, "fixed_report": {},
              "metadata": {"runtime_facts": {"case": {"path": str(runtime)},
                                               "reference": {"path": str(runtime)}}}}
             for index, category in enumerate(categories, 1)]
    monkeypatch.setattr(RUN, "verify_protocol", lambda *_args: protocol)
    monkeypatch.setattr(RUN, "_verify_current_protocol", lambda *_args: None)
    monkeypatch.setattr(RUN, "_source_cases", lambda *_args: cases)
    monkeypatch.setattr(RUN, "_gate", lambda *_args: {
        "run": True, "reason": None, "source_outcome": "not_recovered"})
    monkeypatch.setattr(RUN, "_lease", lambda: {"owner": RUN.OWNER, "lease_id": "fixture"})
    monkeypatch.setattr(RUN.T4, "status", lambda _root: {"requests": 0})
    monkeypatch.setattr(RUN.COMP, "transport_factory", lambda _root: lambda *_args: None)
    seen: list[tuple[str, str]] = []

    def epoch(_case, _reference, *, policy, run_id, out, **_kwargs):
        Path(out).mkdir(parents=True)
        seen.append((run_id, policy))
        return {"status": "complete", "decision_outcome": "HOLD", "request_count": 0,
                "model_actions": 0, "probe_count": 0, "parent_probe_wall_s": 0.0,
                "owned_child_wall_s": 0.0, "confirmed_family": None, "steps": []}

    result = RUN.execution(protocol_path, "a" * 40, source_path, tmp_path,
                           tmp_path / "out", tmp_path / "transport", run_epoch=epoch)
    assert len(result["rows"]) == 24 and result["planned_policy_rows"] == 24
    assert [policy for _, policy in seen[:3]] == list(RUN.COMP.policy_order(0))
    assert [policy for _, policy in seen[3:6]] == list(RUN.COMP.policy_order(1))
    assert all(summary["planned"] == 8 for summary in result["coverage"].values())
    assert result["automatic_repair"] is False


def test_write_is_fresh(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    RUN._write(path, {"status": "complete"})
    with pytest.raises(FileExistsError):
        RUN._write(path, {"status": "replaced"})
