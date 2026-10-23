from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))
import export_ppo_ratio_repair_v2 as EXPORT  # noqa: E402


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def committed_anchor() -> str:
    return str(EXPORT.git("log", "-1", "--format=%H", "--", EXPORT.PROTOCOL.relative_to(ROOT).as_posix()))


def base_result(outcome: str = "invalid") -> dict:
    protocol = EXPORT.read(EXPORT.PROTOCOL)
    return {
        "schema": "ppo_ratio_repair_demo_result_v2",
        "tag": EXPORT.TAG,
        "seed": protocol["seed"],
        "restart_seed": protocol["restart_seed"],
        "experiment_commit": committed_anchor(),
        "protocol_sha256_lf": EXPORT.sha256(EXPORT.PROTOCOL, True),
        "conditions": {
            "committed_source_and_protocol": True,
            "fresh_seeds": True,
            "launch_paths_checked": True,
            "costs_recorded": False,
        },
        "outcome": outcome,
        "model": {},
        "probe_steps": [],
        "costs": {},
        "api_requests": 0,
        "api_requests_in_saved_records": 0,
        "pending_model_records": [],
        "transport_errors": [],
        "demo_success": False,
        "wording": protocol["outcome_wording"][outcome],
        "design_coupling": protocol["design_coupling"],
    }


def preflight(run: Path, result: dict) -> None:
    write(run / "begin.json", {
        "tag": EXPORT.TAG,
        "experiment_commit": result["experiment_commit"],
        "protocol_sha256_lf": result["protocol_sha256_lf"],
        "approver": "fixture-operator",
        "identity_attested": False,
        "delegation": "승인된 등록 검사와 교정을 순서대로 실행하는 운영자 위임",
    })
    write(run / "seed_freshness.json", {
        "schema": "preregistered_seed_freshness_v1",
        "status": "fresh",
        "seeds": [2163, 3163],
        "evidence": [],
    })
    write(run / "launch_paths.json", {"processes_started": 0, "gpu_queries": 0})


def training(run: Path, result: dict, arm: str, checkpoint: str, source_checkpoint: str | None = None) -> dict:
    protocol = EXPORT.read(EXPORT.PROTOCOL)
    seed, folder_name, run_name, _source_kind = EXPORT.EXPECTED_ARMS[arm]
    manifest = {
        "schema": EXPORT.TRAINING_MANIFEST_SCHEMA,
        "mode": "train",
        "arm": arm,
        "seed": seed,
        "run_name": run_name,
        "task": protocol["task"],
        "num_envs": protocol["num_envs"],
        "max_iterations": protocol["training_updates"],
        "configured_learning_epochs": protocol["inspection"]["epochs"],
        "configured_mini_batches": protocol["inspection"]["mini_batches"],
        "runtime_updates_to_observe": protocol["inspection"]["updates"],
        "source_commit_sha": result["experiment_commit"],
        "protocol_path": EXPORT.PROTOCOL.relative_to(ROOT).as_posix(),
        "protocol_sha256_lf": result["protocol_sha256_lf"],
        "source_sha256_lf": protocol["source_sha256_lf"],
        "runtime_source_sha256_lf": protocol["runtime_source_sha256_lf"],
        "records_path": str((run / folder_name / "records.jsonl").resolve()),
        "readback_path": str((run / folder_name / "readback.json").resolve()),
        "source_checkpoint_sha256": source_checkpoint,
    }
    write(run / folder_name / "manifest.json", manifest)
    observer = EXPORT.TRAINING.observer_manifest(manifest)
    observe_module = EXPORT.EXECUTION.GRADIENT.O
    gradient_status = "abnormal" if arm == EXPORT.DEMO.FAULT else "normal"
    summary = {
        "max_abs_error": 1.0 if gradient_status == "abnormal" else 0.0,
        "mismatch_count": 1 if gradient_status == "abnormal" else 0,
        "nonzero_expected_count": 1,
        "finite": True,
        "n": 1,
    }
    rows = []
    for index in range(3):
        rows.append({
            "schema": observe_module.SCHEMA,
            "manifest_sha256": observe_module.manifest_fingerprint(observer),
            "binding": observe_module.manifest_binding(observer),
            "update_index": index,
            "expected": {"optimizer_steps": 20, "minibatches": 20},
            "observed": {
                "optimizer_steps": 20,
                "minibatches": 20,
                EXPORT.EXECUTION.GRADIENT.PRIMITIVE: {
                    "status": gradient_status,
                    "batches": [summary] * 20,
                },
            },
            "error": None,
        })
    records_path = run / folder_name / "records.jsonl"
    records_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8", newline="\n"
    )
    measured = EXPORT.EXECUTION.inspection(records_path, manifest)
    inspection = {
        key: measured[key]
        for key in (
            "count_status", "gradient_status", "first_mismatch_update", "gradient_batches",
            "optimizer_steps", "minibatches",
        )
    }
    mutation_arm = {"NONE": "NONE", EXPORT.DEMO.FAULT: "INVERSE_RATIO", "CORRECTED": "RESTORED"}[arm]
    readback = {
        "arm": arm,
        "configuration": {"learning_epochs": 5, "mini_batches": 4, "clip_param": 0.2},
        "mutation": {
            "arm": mutation_arm,
            "mutation": "inverse_importance_ratio_v1",
            "changed_ratio_assignments": 1,
            "original_ast_sha256": "1" * 64,
            "mutated_ast_sha256": "2" * 64,
            "active_mutation": arm == EXPORT.DEMO.FAULT,
            "restored_original": arm == "CORRECTED",
            "restoration_before_training": arm == "CORRECTED",
            "installed_ast_sha256": "2" * 64 if arm == EXPORT.DEMO.FAULT else "1" * 64,
        },
    }
    write(run / folder_name / "readback.json", readback)
    checkpoint_path = run / folder_name / "model_299.pt"
    first_checkpoint_path = run / folder_name / "model_0.pt"
    checkpoint_path.write_bytes(f"{arm}-final".encode())
    first_checkpoint_path.write_bytes(f"{arm}-first".encode())
    checkpoint = EXPORT.sha256(checkpoint_path)
    checkpoint_evidence = {
        "device": "cpu",
        "first": {"path": str(first_checkpoint_path), "sha256": EXPORT.sha256(first_checkpoint_path), "iteration": 0},
        "final": {"path": str(checkpoint_path), "sha256": checkpoint, "iteration": 299},
    }
    write(run / folder_name / "checkpoint.json", checkpoint_evidence)
    artifact = {
        "arm": arm,
        "seed": seed,
        "run_name": run_name,
        "checkpoint": {"path": str(checkpoint_path), "sha256": checkpoint},
        "inspection": inspection,
        "configuration": {
            "seed": seed,
            "max_iterations": protocol["training_updates"],
            "num_learning_epochs": protocol["inspection"]["epochs"],
            "num_mini_batches": protocol["inspection"]["mini_batches"],
        },
        "execution": {
            "natural_exit": True,
            "timed_out": False,
            "exit_code": 0,
            "cleanup_verified": True,
        },
        "readback": readback,
        "records_path": str(records_path),
        "report": measured["report"],
        "checkpoint_evidence": checkpoint_evidence,
    }
    write(run / f"artifact_{arm.lower()}.json", artifact)
    return artifact


def raw_report(source: str, name: str, checkpoint: str) -> dict:
    report = EXPORT.read(ROOT / f"evals/results/upstream_r1_smoke_20261014/runs/{source}.json")
    report["name"] = name
    report["checkpoint"] = {"file": "model_299.pt", "sha256": checkpoint}
    return report


def response(content: dict) -> dict:
    return {
        "provider_model": EXPORT.MODEL_CALL.MODEL,
        "finish_reason": "stop",
        "content": json.dumps(content, ensure_ascii=False),
        "reasoning_content": None,
        "input_tokens": 12,
        "output_tokens": 8,
        "total_tokens": 20,
        "wall_s": 0.2,
    }


def save_dialogue(run: Path, kind: str, packet: dict, decision: dict) -> dict:
    model = run / "model"
    packet_path = model / f"{kind}.packet.json"
    write(packet_path, packet)
    data = packet_path.read_bytes()
    reservation = EXPORT.MODEL_CALL.reservation_for(kind, data)
    record_path = model / f"{kind}.feedback1.json"
    write(record_path.with_suffix(".reserved.json"), reservation)
    raw = EXPORT.MODEL_CALL.dialogue(kind, data, lambda *_: response(decision))
    raw["reservation_sha256"] = EXPORT.CONTRACT.L.digest(reservation)
    raw["execution_wall_s"] = 0.25
    raw["exit_code"] = 0
    write(record_path, raw)
    review = EXPORT.MODEL_CALL.review(raw, data, kind, reservation)
    result = {
        "review": review,
        "attempts": [{
            "record": record_path.name,
            "record_sha256": EXPORT.sha256(record_path),
            "review": review,
            "calls": [EXPORT._call_cost(call) for call in raw["calls"]],
        }],
    }
    write(model / f"{kind}.review1.json", result["attempts"][0])
    return result


def complete_fixture(tmp_path: Path) -> Path:
    run, result = tmp_path / "run", base_result("recovered")
    preflight(run, result)
    none_sha, fault_sha, corrected_sha = "a" * 64, "b" * 64, "c" * 64
    artifacts = {}
    artifacts["NONE"] = training(run, result, "NONE", none_sha)
    artifacts[EXPORT.DEMO.FAULT] = training(run, result, EXPORT.DEMO.FAULT, fault_sha)
    none_sha = artifacts["NONE"]["checkpoint"]["sha256"]
    fault_sha = artifacts[EXPORT.DEMO.FAULT]["checkpoint"]["sha256"]
    artifacts["CORRECTED"] = training(run, result, "CORRECTED", corrected_sha, fault_sha)
    corrected_sha = artifacts["CORRECTED"]["checkpoint"]["sha256"]
    reports = {
        "NONE": raw_report("upstream_r1_none_s2042", f"{EXPORT.TAG}_none", none_sha),
        EXPORT.DEMO.FAULT: raw_report("upstream_r1_s2042", f"{EXPORT.TAG}_inverse_ratio", fault_sha),
        "CORRECTED": raw_report("upstream_r1_none_s2042", f"{EXPORT.TAG}_fixed_corrected", corrected_sha),
    }
    write(run / "source_evaluation/none.json", reports["NONE"])
    write(run / "source_evaluation/inverse_ratio.json", reports[EXPORT.DEMO.FAULT])
    write(run / "correction_evaluation/corrected.json", reports["CORRECTED"])
    bands = EXPORT.read(EXPORT.BANDS)["bands"]
    result["source_gate"] = EXPORT.DEMO.source_gate(reports, bands)
    write(run / "source_gate.json", result["source_gate"])
    write(run / "reference/report.json", {"schema": "fixture_reference"})

    ranking = ["optimizer", "reward", "actuator", "exploration", "physics", "termination"]
    diagnosis_packet = {
        "schema": EXPORT.CONTRACT.PACKET_SCHEMA,
        "facts": {"telemetry:case:error": 1.2, "telemetry:reference:error": 0.2},
        "observed": [],
        "budget_left": EXPORT.CONTRACT.MAX_PROBES,
        "execution_budget_s": 1200,
    }
    diagnosis_decision = {
        "ranking": ranking,
        "probe": EXPORT.CONTRACT.GRADIENT_PROBE,
        "action": "continue",
        "reason": "surrogate 계산 검사를 선택한다",
        "claims": [{"id": "telemetry:case:error", "value": 1.2}],
        "expected_outcomes": {
            mechanism: EXPORT.CONTRACT.public_catalog()[EXPORT.CONTRACT.GRADIENT_PROBE]["expect"][mechanism]
            for mechanism in ranking[:3]
        },
    }
    result["model"]["diagnosis"] = save_dialogue(run, "diagnosis", diagnosis_packet, diagnosis_decision)

    probes = EXPORT.CONTRACT.M.RegistryLedger(run / "probes.ledger.jsonl", EXPORT.CONTRACT.registry())
    prereg = {
        "checkpoint_sha256": fault_sha,
        "seed": 2163,
        "protocol_sha256_lf": result["protocol_sha256_lf"],
        "experiment_commit": result["experiment_commit"],
        "registry_digest": probes.registry_digest,
    }
    request = probes.propose(prereg, EXPORT.CONTRACT.GRADIENT_PROBE, 1200)
    probes.approve(request["request_id"], "fixture-operator")
    probes.consume(request["request_id"], prereg, EXPORT.CONTRACT.GRADIENT_PROBE)
    measurement = {
        "case": {"count_status": "normal", "gradient_status": "abnormal", "first_mismatch_update": 0,
                 "optimizer_steps": [20, 20, 20], "minibatches": [20, 20, 20]},
        "control": {"count_status": "normal", "gradient_status": "normal", "first_mismatch_update": None,
                    "optimizer_steps": [20, 20, 20], "minibatches": [20, 20, 20]},
        "source_checkpoint_sha256": fault_sha,
        "checkpoint_loaded": False,
        "execution_evidence_sha256": "d" * 64,
    }
    receipt = probes.receipt(request["request_id"], "abnormal", measurement, 2.0, 0)
    result["probe_steps"] = [{**receipt, "probe": EXPORT.CONTRACT.GRADIENT_PROBE}]
    result["loop"] = EXPORT.CONTRACT.advance(
        EXPORT.CONTRACT.initial_state(diagnosis_decision), EXPORT.CONTRACT.GRADIENT_PROBE, "abnormal"
    )
    write(run / "loop_01.json", result["loop"])

    correction_packet = EXPORT.CONTRACT.correction_packet(
        result["loop"], result["probe_steps"][0], reports[EXPORT.DEMO.FAULT], fault_sha,
        EXPORT.sha256(run / "source_evaluation/inverse_ratio.json"),
    )
    correction_decision = {
        "correction_id": EXPORT.CONTRACT.CORRECTION,
        "recovery_mode": "restart",
        "cited_facts": ["confirmed_subtype"],
        "checklist": {key: "rule_mode" for key in correction_packet["checklist"]},
        "hold_reason": None,
    }
    result["model"]["correction"] = save_dialogue(run, "correction", correction_packet, correction_decision)

    verdict = EXPORT.REC.verdict(
        EXPORT.REC.compare(EXPORT.G._behavior(reports["CORRECTED"]), EXPORT.G._behavior(reports["NONE"])), bands
    )
    recovery = {
        "outcome": "recovered",
        "verdict": verdict,
        "source_metrics": reports[EXPORT.DEMO.FAULT]["metrics"],
        "reference_metrics": reports["NONE"]["metrics"],
        "corrected_metrics": reports["CORRECTED"]["metrics"],
        "checkpoint_sha256": corrected_sha,
    }
    write(run / "recovery.json", recovery)
    corrections = EXPORT.CorrectionLedger(run / "correction.ledger.jsonl", EXPORT.CONTRACT.correction_registry())
    request = corrections.propose(
        EXPORT.CONTRACT.L.digest(result["loop"]), "optimizer", EXPORT.CONTRACT.L.digest(result["probe_steps"]),
        EXPORT.CONTRACT.CORRECTION, "restart", {"case": "optimizer_surrogate_ratio", "arm": "corrected"},
        3600, evidence={
            "packet_digest": EXPORT.CONTRACT.L.digest(correction_packet),
            "model_review_digest": EXPORT.CONTRACT.L.digest(result["model"]["correction"]),
            "protocol_sha256_lf": result["protocol_sha256_lf"],
            "checkpoint_sha256": fault_sha,
        },
    )
    corrections.approve(request["request_id"], "fixture-operator")
    corrections.consume(request["request_id"], EXPORT.CONTRACT.L.digest(result["loop"]))
    corrections.receipt(request["request_id"], EXPORT.TAG, "recovered", EXPORT.sha256(run / "recovery.json"), 30.0, 0)
    result["recovery"] = recovery
    result["costs"] = {
        "train_NONE": 10.0,
        f"train_{EXPORT.DEMO.FAULT}": 10.0,
        "source_evaluation": 2.0,
        "reference": 1.0,
        f"probe_1_{EXPORT.CONTRACT.GRADIENT_PROBE}": 2.0,
        "correction_training": 10.0,
        "correction_evaluation": 2.0,
    }
    result["api_requests"] = result["api_requests_in_saved_records"] = 2
    result["conditions"] = {key: True for key in EXPORT.read(EXPORT.PROTOCOL)["success_conditions"]}
    result["demo_success"] = True
    result["wording"] = EXPORT.read(EXPORT.PROTOCOL)["outcome_wording"]["recovered"]
    write(run / "result.json", result)
    assert artifacts["CORRECTED"]["inspection"]["gradient_status"] == "normal"
    return run


def test_complete_fixture_recomputes_all_gates_and_exports_allowlist(tmp_path):
    run = complete_fixture(tmp_path)
    payloads = EXPORT.verify_and_build(run)
    assert set(payloads) == EXPORT.PUBLIC_FILES
    assert payloads["summary.json"]["outcome"] == "recovered"
    assert payloads["summary.json"]["demo_success"] is True
    assert set(payloads["evaluations.json"]["arms"]) == {"NONE", EXPORT.DEMO.FAULT, "CORRECTED"}
    assert len(payloads["model_evidence.json"]["dialogues"]) == 2
    assert payloads["probe_correction.json"]["probes"][0]["probe"] == EXPORT.CONTRACT.GRADIENT_PROBE
    assert payloads["probe_correction.json"]["corrections"][0]["verdict"] == "recovered"


@pytest.mark.parametrize("claimed", ["source_failure_absent", "diagnosis_hold"])
def test_outcome_is_recomputed_instead_of_trusting_a_nominal_stop(tmp_path, claimed):
    run = complete_fixture(tmp_path)
    stored = EXPORT.read(run / "result.json")
    stored["outcome"] = claimed
    stored["wording"] = EXPORT.read(EXPORT.PROTOCOL)["outcome_wording"][claimed]
    stored["demo_success"] = False
    write(run / "result.json", stored)
    with pytest.raises(ValueError, match="stored outcome|stored recovery"):
        EXPORT.verify_and_build(run)


def test_current_head_blob_is_checked_even_if_worktree_bytes_match(tmp_path, monkeypatch):
    run = complete_fixture(tmp_path)
    original = EXPORT.git

    def staged_reversion(*args, binary=False):
        if args[0] == "show" and str(args[1]).startswith("HEAD:"):
            return b"staged-reversion-does-not-match-head"
        return original(*args, binary=binary)

    monkeypatch.setattr(EXPORT, "git", staged_reversion)
    with pytest.raises(ValueError, match="current HEAD"):
        EXPORT.verify_and_build(run)


@pytest.mark.parametrize("target", ["records", "readback", "checkpoint"])
def test_training_artifact_is_recomputed_from_raw_execution_files(tmp_path, target):
    run = complete_fixture(tmp_path)
    if target == "records":
        path = run / "training/none/records.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        rows[0]["observed"]["optimizer_steps"] = 19
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    elif target == "readback":
        path = run / "training/none/readback.json"
        value = EXPORT.read(path)
        value["mutation"]["installed_ast_sha256"] = "2" * 64
        write(path, value)
    else:
        artifact = EXPORT.read(run / "artifact_none.json")
        Path(artifact["checkpoint"]["path"]).write_bytes(b"tampered-checkpoint")
    with pytest.raises(ValueError):
        EXPORT.verify_and_build(run)


def test_pending_reservation_transport_and_worker_identity_are_packet_bound(tmp_path):
    run, model = tmp_path / "run", tmp_path / "run/model"
    packet_path = model / "diagnosis.packet.json"
    write(packet_path, {"fixture": "packet"})
    packet_data = packet_path.read_bytes()
    reservation_path = model / "diagnosis.feedback1.reserved.json"
    write(reservation_path, EXPORT.MODEL_CALL.reservation_for("diagnosis", packet_data))
    write(model / "diagnosis.feedback1.pid.json", {
        "pid": 42,
        "start_ticks": "12345",
        "packet_sha256": EXPORT.sha256(packet_path),
    })
    write(model / "diagnosis.transport_error.json", {
        "type": "dialogue_timeout",
        "worker_cleanup": "verified",
    })
    result = {
        "model": {},
        "api_requests": None,
        "api_requests_in_saved_records": 0,
        "pending_model_records": [reservation_path.name],
        "transport_errors": ["diagnosis.transport_error.json"],
    }
    public = EXPORT._models(run, result, {})
    assert public["pending_record_count"] == 1
    assert public["transport_error_count"] == 1

    reservation = EXPORT.read(reservation_path)
    reservation["packet_sha256"] = "0" * 64
    write(reservation_path, reservation)
    with pytest.raises(ValueError, match="reservation"):
        EXPORT._models(run, result, {})

    write(reservation_path, EXPORT.MODEL_CALL.reservation_for("diagnosis", packet_data))
    (model / "diagnosis.transport_error.json").write_text("", encoding="utf-8")
    with pytest.raises((ValueError, json.JSONDecodeError)):
        EXPORT._models(run, result, {})


def test_failed_terminal_probe_keeps_unknown_receipt_without_fabricating_loop_artifact(tmp_path):
    run = tmp_path / "run"
    decision = {
        "ranking": ["reward", "optimizer", "exploration", "actuator", "physics", "termination"],
        "probe": "P_reward",
        "action": "continue",
        "reason": "fixture",
        "claims": [],
        "expected_outcomes": {
            mechanism: EXPORT.CONTRACT.public_catalog()["P_reward"]["expect"][mechanism]
            for mechanism in ("reward", "optimizer", "exploration")
        },
    }
    result = base_result("invalid")
    preflight(run, result)
    result["model"] = {"diagnosis": {"review": {"status": "valid", "decision": decision}}}
    ledger = EXPORT.CONTRACT.M.RegistryLedger(run / "probes.ledger.jsonl", EXPORT.CONTRACT.registry())
    prereg = {
        "checkpoint_sha256": "f" * 64,
        "seed": 2163,
        "protocol_sha256_lf": result["protocol_sha256_lf"],
        "experiment_commit": result["experiment_commit"],
        "registry_digest": ledger.registry_digest,
    }
    state = EXPORT.CONTRACT.initial_state(decision)
    stored_steps = []
    for index in range(1, 5):
        probe = state["next_probe"]
        request = ledger.propose(prereg, probe, 1200)
        ledger.approve(request["request_id"], "fixture-operator")
        ledger.consume(request["request_id"], prereg, probe)
        outcome, exit_code = ("unknown", 1) if index == 4 else ("normal", 0)
        receipt = ledger.receipt(request["request_id"], outcome, {}, 0.0, exit_code)
        state = EXPORT.CONTRACT.advance(state, probe, outcome)
        if index < 4:
            stored_steps.append({**receipt, "probe": probe})
            write(run / f"loop_{index:02d}.json", state)
    result["probe_steps"] = stored_steps
    result["failure"] = {
        "stage": f"probe_4_{probe}",
        "type": "RuntimeError",
        "site": {"file": "evals/r1_repair_execution.py", "line": 89},
    }
    probe_public, replayed = EXPORT._probe_correction(
        run,
        result,
        {},
        EXPORT.read(EXPORT.PROTOCOL),
        {EXPORT.DEMO.FAULT: {"checkpoint": {"sha256": "f" * 64}}},
    )
    assert [row["outcome"] for row in probe_public["probes"]] == ["normal", "normal", "normal", "unknown"]
    assert replayed == state


@pytest.mark.parametrize(("relative", "message"), [
    ("begin.json", "begin binding"),
    ("training/none/manifest.json", "training manifest"),
    ("artifact_inverse_ratio.json", "training artifact"),
    ("source_gate.json", "source gate artifact"),
    ("source_evaluation/none.json", "기대값"),
    ("loop_01.json", "stored loop artifact"),
    ("model/correction.feedback1.json", "model"),
    ("model/diagnosis.review1.json", "model attempt"),
    ("correction.ledger.jsonl", "correction"),
])
def test_complete_outcome_rejects_tampered_or_missing_stage_evidence(tmp_path, relative, message):
    run = complete_fixture(tmp_path)
    target = run / relative
    if target.suffix == ".jsonl":
        rows = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]
        rows[0]["registry_digest"] = "0" * 64
        target.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    elif "feedback" in target.name or "review" in target.name:
        target.unlink()
    else:
        value = EXPORT.read(target)
        if relative == "begin.json":
            value["experiment_commit"] = "0" * 40
        elif "manifest" in relative:
            value["configured_learning_epochs"] = 4
        elif relative == "artifact_inverse_ratio.json":
            value["inspection"]["gradient_status"] = "normal"
        elif relative == "source_gate.json":
            value["outcome"] = "source_failure_absent"
        elif relative == "loop_01.json":
            value["status"] = "open"
        else:
            value["name"] = "report_supplied_expectation"
        write(target, value)
    with pytest.raises((ValueError, FileNotFoundError), match=message):
        EXPORT.verify_and_build(run)


@pytest.mark.parametrize(("name", "message"), [
    ("diagnosis.feedback2.pid.json", "orphan model PID"),
    ("diagnosis.review2.json", "orphan model review"),
])
def test_model_inventory_rejects_orphan_pid_and_review(tmp_path, name, message):
    run = complete_fixture(tmp_path)
    write(run / "model" / name, {})
    with pytest.raises(ValueError, match=message):
        EXPORT.verify_and_build(run)


def test_loop_condition_and_recovery_receipt_tampering_are_rejected(tmp_path):
    run = complete_fixture(tmp_path)
    stored = EXPORT.read(run / "result.json")
    stored["loop"]["status"] = "open"
    write(run / "result.json", stored)
    with pytest.raises(ValueError, match="correction request|stored loop"):
        EXPORT.verify_and_build(run)

    run = complete_fixture(tmp_path / "second")
    rows = [json.loads(line) for line in (run / "correction.ledger.jsonl").read_text(encoding="utf-8").splitlines()]
    rows[-1]["verdict"] = "not_recovered"
    (run / "correction.ledger.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="recovery outcome"):
        EXPORT.verify_and_build(run)

    run = complete_fixture(tmp_path / "third")
    ledger = run / "correction.ledger.jsonl"
    rows = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
    rows[-1]["result_sha"] = "0" * 64
    ledger.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="result binding"):
        EXPORT.verify_and_build(run)


def test_unknown_correction_receipt_is_kept_separate_from_completed_outcome(tmp_path):
    run = complete_fixture(tmp_path)
    ledger = run / "correction.ledger.jsonl"
    rows = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
    rows[-1].update(verdict="unknown", gpu_s=4001.0, over_budget=True)
    ledger.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    stored = EXPORT.read(run / "result.json")
    stored["outcome"] = "correction_unknown"
    stored["wording"] = EXPORT.read(EXPORT.PROTOCOL)["outcome_wording"]["correction_unknown"]
    stored["demo_success"] = False
    write(run / "result.json", stored)
    payloads = EXPORT.verify_and_build(run)
    assert payloads["summary.json"]["outcome"] == "correction_unknown"
    assert payloads["probe_correction.json"]["corrections"][0]["verdict"] == "unknown"


def test_failure_record_is_exported_but_success_boolean_without_evidence_is_rejected(tmp_path):
    run, stored = tmp_path / "run", base_result("invalid")
    preflight(run, stored)
    write(run / "result.json", stored)
    payloads = EXPORT.verify_and_build(run)
    assert payloads["summary.json"]["outcome"] == "invalid"
    assert payloads["evaluations.json"]["arms"] == {}

    false_success = base_result("recovered")
    false_success["conditions"] = {key: True for key in EXPORT.read(EXPORT.PROTOCOL)["success_conditions"]}
    false_success["demo_success"] = True
    preflight(run, false_success)
    write(run / "result.json", false_success)
    with pytest.raises(ValueError, match="stored outcome|required source"):
        EXPORT.verify_and_build(run)


def test_exact_export_inventory_bytes_audit_and_public_scrub(tmp_path):
    run = complete_fixture(tmp_path)
    payloads = EXPORT.verify_and_build(run)
    output, audit = tmp_path / "public", tmp_path / "audit"
    EXPORT.write_new(payloads, output, audit)
    assert set(EXPORT.verify_export(output, run, EXPORT.PROTOCOL, audit)) == EXPORT.PUBLIC_FILES

    with (output / "summary.json").open("a", encoding="utf-8") as handle:
        handle.write(" ")
    with pytest.raises(ValueError, match="artifact differs"):
        EXPORT.verify_export(output, run, EXPORT.PROTOCOL, audit)

    (output / "summary.json").write_bytes(EXPORT.serialized(payloads["summary.json"]))
    write(output / "extra.json", {})
    with pytest.raises(ValueError, match="inventory"):
        EXPORT.verify_export(output, run, EXPORT.PROTOCOL, audit)

    (output / "extra.json").unlink()
    write(audit / "extra.json", {})
    with pytest.raises(ValueError, match="audit inventory"):
        EXPORT.verify_export(output, run, EXPORT.PROTOCOL, audit)

    (audit / "extra.json").unlink()
    receipt = EXPORT.read(audit / "receipt.json")
    receipt["public_artifact_sha256"]["summary.json"] = "0" * 64
    write(audit / "receipt.json", receipt)
    with pytest.raises(ValueError, match="audit differs"):
        EXPORT.verify_export(output, run, EXPORT.PROTOCOL, audit)

    bad = deepcopy(payloads)
    bad["summary.json"]["leak"] = r"C:\Users\person\run.json"
    with pytest.raises(ValueError):
        EXPORT.verify_public_payloads(bad)
    bad = deepcopy(payloads)
    bad["model_evidence.json"]["response_id"] = "hidden"
    with pytest.raises(ValueError):
        EXPORT.verify_public_payloads(bad)
