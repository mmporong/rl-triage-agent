from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

UTC_TZ = timezone.utc  # noqa: UP017 - Python 3.10 compatibility

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("vision_selection_execution_v1", ROOT / "evals/vision_selection_execution_v1.py")
assert SPEC and SPEC.loader
E = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = E
SPEC.loader.exec_module(E)


def _context(tmp_path: Path) -> dict:
    stage = tmp_path / ".triage-runtime/vision-selection/run"
    stage.mkdir(parents=True)
    rows = [{"call_id": f"choice-{i:02d}", "capsule_sha256": f"{i:064x}",
             "view_kind": "text_only", "checkpoint_sha256": "a" * 64} for i in range(1, 25)]
    public = [{"call_id": row["call_id"], "slot_id": f"input-{(i - 1) // 3 + 1:02d}",
               "view_kind": row["view_kind"], "messages_sha256": f"{i + 30:064x}",
               "method_protocol_sha256": "b" * 64, "source_protocol_sha256": "c" * 64,
               "seed_group": "source-01", "source_role": "normal"} for i, row in enumerate(rows, 1)]
    protocol = tmp_path / "protocol.json"; protocol.write_text("{}", encoding="utf-8")
    locations = tmp_path / "locations.json"; locations.write_text("{}", encoding="utf-8")
    return {"repository": tmp_path, "stage": stage,
            "manifest": {"api_cap": 72, "rows": rows},
            "prepared": {"protocol": {"rows": public}}, "protocol_path": protocol,
            "protocol_sha256_lf": "d" * 64, "source_anchor": "e" * 40,
            "input_locations": locations}


def _patch_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(E.CALL, "_row", lambda repo, manifest, call_id: (
        0, next(row for row in manifest["rows"] if row["call_id"] == call_id), {"messages": []}))


def _patch_readiness(monkeypatch: pytest.MonkeyPatch) -> None:
    def verify(_root: Path, path: Path, _protocol: dict) -> dict:
        role = "registered-f1" if "registered-f1" in path.as_posix() else "normal"
        return {"checkpoint_sha256": "b" * 64 if role == "registered-f1" else "a" * 64,
                "source_seed": 4101, "source_role": role}
    monkeypatch.setattr(E.INPUTS, "verify_model_readiness", verify)


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ref(root: Path, path: Path) -> dict[str, str]:
    return {"path": path.relative_to(root).as_posix(), "sha256": _sha(path)}


def _gpu_observation(folder: Path, ordinal: int) -> Path:
    prefix = f"{ordinal:06d}"
    gpu, processes = folder / f"{prefix}.gpu.csv", folder / f"{prefix}.process.csv"
    gpu.write_text("GPU-fixture\n", encoding="utf-8", newline="")
    processes.write_text("", encoding="utf-8", newline="")
    receipt = folder / f"{prefix}.receipt.json"
    _write(receipt, {
        "schema": "vision_remote_gpu_observation_v1", "ordinal": ordinal,
        "resource_id": "brev-fixture", "gpu_uuid": "GPU-fixture",
        "gpu_command": E.REMOTE.GPU_COMMAND, "gpu_exit_code": 0,
        "gpu_stdout_file": gpu.name, "gpu_stdout_sha256": _sha(gpu), "gpu_stderr": "",
        "process_command": E.REMOTE.PROCESS_COMMAND, "process_exit_code": 0,
        "process_stdout_file": processes.name, "process_stdout_sha256": _sha(processes),
        "process_stderr": "",
    })
    return receipt


def _brev_sources(root: Path, folder: Path, side: str, checkpoint: str,
                  measurement: dict, *, seed: int = 4101) -> tuple[dict, dict[str, Path]]:
    stage = f"/remote/probes/{side}"
    run_name = f"brev_{side}"
    report_path = f"{stage}/report.json"
    checkpoint_path = f"/remote/checkpoints/{side}_model.pt"
    expected = _expected_brev_source()
    environment = expected["environment"]
    source = {
        "lab_head": environment["isaaclab_sha"], "lab_tracked_clean": True,
        "source_anchor": expected["source_anchor"],
        "protocol": expected["protocol"],
        "environment": environment,
        "selected_interpreter": {"python": "3.10.12", "executable": "/remote/venv/bin/python",
                                 "packages": {key: value for key, value in environment.items()
                                              if key not in {"python", "isaaclab_sha", "cuda_available",
                                                             "cuda_device_count"}}},
        "source_sha256_lf": expected["source_sha256_lf"],
        "runtime_source_sha256_lf": expected["runtime_source_sha256_lf"],
        "python_package_source_sha256_lf": expected["python_package_source_sha256_lf"],
    }
    job = {
        "schema": "brev_recovery_worker_job_v1", "mode": "probe", "stage": stage,
        "lab": "/remote/IsaacLab", "repo": "/remote/repo",
        "protocol_path": "/remote/repo/bench/protocols/brev_recovery_v4.json",
        "protocol_sha256_lf": source["protocol"]["sha256_lf"], "source_commit_sha": source["source_anchor"],
        "source_sha256_lf": source["source_sha256_lf"],
        "runtime_source_sha256_lf": source["runtime_source_sha256_lf"],
        "python_package_source_sha256_lf": source["python_package_source_sha256_lf"],
        "arm": "F1" if side == "case" else "NONE", "seed": seed, "run_name": run_name,
        "probe": "P_episode", "checkpoint_path": checkpoint_path, "checkpoint_sha256": checkpoint,
        "report_path": report_path, "training_manifest_path": f"{stage}/training.manifest.json",
        "inspection_manifest_path": None, "records_path": None, "readback_path": None,
    }
    report = {
        "name": run_name, "checkpoint": {"file": f"{side}_model.pt", "sha256": checkpoint},
        "task": "Isaac-Velocity-Flat-Unitree-Go2-v0", "seed": seed, "num_envs": 64,
        "probes": {"P_episode": {"measurement": measurement, "exit": 0, "elapsed_s": 1.0}},
    }
    jobs = {"fault": "F1" if side == "case" else "NONE", "seed": seed, "task": report["task"],
            "jobs": [{"name": run_name, "checkpoint": checkpoint_path,
                      "probes": ["P_episode"], "output": report_path}]}
    execution = {
        "schema": "brev_recovery_posix_execution_v1", "mode": "probe", "completed": True,
        "started_at": "2027-01-01T00:00:00+00:00", "finished_at": "2027-01-01T00:00:02+00:00",
        "exit_code": 0, "timed_out": False, "cleanup_verified": True,
        "launch": {"session_id": 1}, "gpu_before": [], "gpu_after": [], "wall_s": 2.0,
        "source_before": source, "source_after": source,
    }
    side_folder = folder / side
    paths = {name: side_folder / filename for name, filename in {
        "job": "job.json", "worker": "worker.json", "execution": "execution.json",
        "backend_result": "backend_result.json", "report_jobs": "report.jobs.json", "report": "report.json",
    }.items()}
    _write(paths["job"], job); _write(paths["report_jobs"], jobs); _write(paths["report"], report)
    worker = {"schema": "brev_recovery_probe_worker_v1", "completed": True, "probe": "P_episode",
              "measurement": measurement, "report": {"path": report_path, "sha256": _sha(paths["report"])},
              "inspection_manifest": None, "wall_s": 1.5}
    _write(paths["worker"], worker); _write(paths["execution"], execution)
    _write(paths["backend_result"], {"execution": execution, "measurement": measurement, "report": report})
    descriptor = {"schema": "vision_selection_brev_probe_evidence_v1", "side": side,
                  "probe": "P_episode", "checkpoint_sha256": checkpoint,
                  "sources": {name: _ref(root, path) for name, path in paths.items()}}
    return descriptor, paths


def _expected_brev_source() -> dict:
    protocol = ROOT / "bench/protocols/brev_recovery_v4.json"
    return E._probe_execution_source(ROOT, {
        "source_anchor": "fa0c8f2707eaee93c84ff273a4510c164ef3fd90",
        "protocol": {"path": protocol.relative_to(ROOT).as_posix(),
                     "sha256_lf": E.WORKER._sha_lf(protocol)},
    })


def _brev_probe_fixture(tmp_path: Path) -> tuple[dict, dict, dict[str, Path]]:
    root, work = tmp_path, tmp_path / "retrieved"
    case_checkpoint, reference_checkpoint = "b" * 64, "a" * 64
    bindings = {"case": {"checkpoint_sha256": case_checkpoint, "telemetry_sha256": "d" * 64},
                "reference": {"checkpoint_sha256": reference_checkpoint, "telemetry_sha256": "c" * 64}}
    packet = E.CONTROLLER.S.build_packet(epoch=1, applied_corrections=[], prior_receipt=None,
                                         facts={"config:action_scale": 0.25}, input_bindings=bindings,
                                         execution_budget_s=1200)
    snapshot = E.CONTROLLER.V.build_snapshot(
        packet, "portable_actual", "fixed_eligible", E.CONTROLLER.V.file_digest(E.CONTROLLER.V.PROTOCOL),
        work / "registered-f1/epoch1/probes.ledger.jsonl", None,
    )
    choice = E.CONTROLLER.choose(snapshot, "fixed_eligible")
    assert choice["probe"] == "P_episode"
    raw = work / "raw"; raw.mkdir(parents=True)
    case_measurement = {"timeout_count": 6, "median_timeout_step": 100.0, "timeout_ratio": 0.1}
    reference_measurement = {"timeout_count": 64, "median_timeout_step": 1000.0, "timeout_ratio": 1.0}
    case_raw, reference_raw = raw / "case.json", raw / "reference.json"
    _write(case_raw, case_measurement); _write(reference_raw, reference_measurement)
    seal_ref = E.CONTROLLER.record_probe(work, "registered-f1", 1, snapshot, choice,
                                         case_raw, reference_raw, "cpu", 2.0)
    seal_path = Path(seal_ref["path"]); seal = json.loads(seal_path.read_text(encoding="utf-8"))
    auth_path = Path(seal["authorization_path"]); auth = json.loads(auth_path.read_text(encoding="utf-8"))
    case_descriptor, case_paths = _brev_sources(root, work / "brev", "case", case_checkpoint, case_measurement)
    ref_descriptor, ref_paths = _brev_sources(root, work / "brev", "reference", reference_checkpoint,
                                               reference_measurement)
    case_path, reference_path = work / "case_descriptor.json", work / "reference_descriptor.json"
    _write(case_path, case_descriptor); _write(reference_path, ref_descriptor)
    auth["evidence_kind"] = E.CONTROLLER.ACTUAL_EVIDENCE
    auth["execution_plan"] = {
        "stage": "/remote/probes/case", "run_name": "brev_case", "checkpoint_sha256": case_checkpoint,
        "seed": 4101, "args": auth["execution_plan"]["args"], "gpu_contract": "residual_v1",
        "reference": {"evidence_path": "/remote/reference_descriptor.json",
                      "evidence_sha256": _sha(reference_path), "checkpoint_sha256": reference_checkpoint,
                      "stage": "/remote/probes/reference", "run_name": "brev_reference", "seed": 4101,
                      "gpu_contract": "residual_v1"},
    }
    _write(auth_path, auth)
    seal.update(evidence_kind=E.CONTROLLER.ACTUAL_EVIDENCE, authorization_sha256=_sha(auth_path),
                case_path="/remote/case_descriptor.json", case_sha256=_sha(case_path),
                reference_path="/remote/reference_descriptor.json", reference_sha256=_sha(reference_path))
    _write(seal_path, seal)
    ledger = auth_path.parent / "probes.ledger.jsonl"
    item = {name: _ref(root, path) for name, path in {
        "seal": seal_path, "authorization": auth_path, "ledger": ledger,
        "case": case_path, "reference": reference_path,
    }.items()}
    paths = {"seal": seal_path, "authorization": auth_path, "ledger": ledger,
             "case": case_path, "reference": reference_path,
             **{f"case_{key}": value for key, value in case_paths.items()},
             **{f"reference_{key}": value for key, value in ref_paths.items()}}
    return item, snapshot, paths


def _evaluation_fixture(tmp_path: Path) -> tuple[Path, dict, dict, list[dict], dict, Path]:
    root, folder = tmp_path, tmp_path / "evidence"
    folder.mkdir()
    source_sha = "1" * 64
    source_protocol = {
        "_verified_sha256_lf": source_sha,
        "evaluation": {"bands_path": "bands.json"},
        "fixed_eval_protocol": "fixed_eval.json",
        "source_action_order": [
            {"action_id": "train-none", "kind": "train", "job_id": "job-none"},
            {"action_id": "train-f1", "kind": "train", "job_id": "job-f1"},
            {"action_id": "eval-4101", "kind": "evaluate", "seed": 4101,
             "none_job_id": "job-none", "f1_job_id": "job-f1", "run_name": "eval-4101"},
        ],
        "training_plan": [
            {"job_id": "job-none", "arm": "NONE", "seed": 4101},
            {"job_id": "job-f1", "arm": "F1", "seed": 4101},
        ],
        "resources": {
            "action_limits": {"evaluate": {"max_wall_s": 900, "max_compute_usd": 1.0}},
            "cost_scope": "fixture_cost", "compute_quote_usd_per_h": 2.0,
            "whole_batch_budget_usd": 20.0, "deadline_utc": "2027-01-02T00:00:00+00:00",
        },
    }
    bands = json.loads((ROOT / "evals/results/p0c_fixed_eval_20261009/summary.json").read_text(encoding="utf-8"))["bands"]
    _write(root / "bands.json", {"bands": bands})
    (root / "fixed_eval.json").write_bytes((ROOT / "bench/protocols/fixed_eval_v1.json").read_bytes())
    rows = []
    report_templates = {
        "NONE": json.loads((ROOT / "evals/results/reward_v4_s2026_20261013/original_reports/clean_reference.json")
                           .read_text(encoding="utf-8")),
        "F1": json.loads((ROOT / "evals/results/reward_v4_s2026_20261013/original_reports/fault.json")
                         .read_text(encoding="utf-8")),
    }
    artifacts = {}
    role_data = (("normal", "NONE", "a" * 64, "c" * 64),
                 ("registered-f1", "F1", "b" * 64, "d" * 64))
    for role, arm, checkpoint, telemetry in role_data:
        for index in range(3):
            rows.append({"seed_group": "source-01", "source_role": role, "source_seed": 4101,
                         "checkpoint_sha256": checkpoint, "call_id": f"{role}-{index}"})
        path = folder / f"{arm.lower()}-artifact.json"
        _write(path, {"arm": arm, "seed": 4101,
                      "checkpoint": {"path": f"/remote/{arm.lower()}_model.pt", "sha256": checkpoint},
                      "telemetry": {"path": f"/remote/{arm.lower()}_telemetry.json", "sha256": telemetry}})
        artifacts[arm] = path
        report_templates[arm]["name"] = arm
        report_templates[arm]["checkpoint"] = {"file": f"{arm.lower()}_model.pt", "sha256": checkpoint}
    locations = []
    for row in rows:
        arm = "NONE" if row["source_role"] == "normal" else "F1"
        telemetry = "c" * 64 if arm == "NONE" else "d" * 64
        item = folder / row["call_id"]
        snapshot = item / "snapshot.json"
        _write(snapshot, {"inputs": {"packet": {"input_bindings": {
            "case": {"checkpoint_sha256": row["checkpoint_sha256"], "telemetry_sha256": telemetry},
            "reference": {"checkpoint_sha256": "a" * 64, "telemetry_sha256": "c" * 64},
        }}}})
        readiness = item / "readiness.json"
        _write(readiness, {"source_training_artifact": _ref(root, artifacts[arm])})
        placeholders = {}
        for name in ("contract", "trace", "camera_pose_audit"):
            path = item / f"{name}.json"; _write(path, {"name": name}); placeholders[name] = _ref(root, path)
        manifest = item / "input_manifest.json"
        _write(manifest, {"schema": "vision_selection_input_v1", "snapshot": _ref(root, snapshot),
                          "model_readiness": _ref(root, readiness), **placeholders})
        capsule = item / "capsule.json"; _write(capsule, {"call_id": row["call_id"]})
        locations.append({"call_id": row["call_id"], "input_manifest": _ref(root, manifest),
                          "capsule": _ref(root, capsule)})
    locations_path = root / "input_locations.json"
    _write(locations_path, {"schema": "vision_selection_input_locations_v1",
                            "execution_id": "fixture", "rows": locations})
    resource_path = folder / "resource.json"
    _write(resource_path, {"resource_id": "brev-fixture", "gpu_uuid": "GPU-fixture",
                           "compute_started_at_utc": "2027-01-01T00:00:00+00:00"})
    before, after = _gpu_observation(folder, 1), _gpu_observation(folder, 2)
    intent_path = folder / "intent.json"
    _write(intent_path, {
        "schema": E.SOURCE.INTENT_SCHEMA, "action_id": "eval-4101", "kind": "evaluate", "ordinal": 2,
        "protocol_sha256_lf": source_sha, "resource_sha256": _sha(resource_path),
        "cost_before_usd": 2.0, "started_at_utc": "2027-01-01T01:00:00+00:00",
        "reserved_max_wall_s": 900, "reserved_max_compute_usd": 1.0, "cost_scope": "fixture_cost",
    })
    raw_path = folder / "raw.json"
    execution = {
        "schema": "brev_recovery_posix_execution_v1", "mode": "evaluate", "completed": True,
        "exit_code": 0, "timed_out": False, "cleanup_verified": True, "gpu_before": [], "gpu_after": [],
        "gpu_before_observation": {"path": "/remote/gpu/000001.receipt.json", "sha256": _sha(before)},
        "gpu_after_observation": {"path": "/remote/gpu/000002.receipt.json", "sha256": _sha(after)},
        "source_before": {"head": "x"}, "source_after": {"head": "x"},
        "launch": {"pid": 123}, "wall_s": 60.0,
    }
    none_report, f1_report = report_templates["NONE"], report_templates["F1"]
    reports = {"NONE": none_report, "F1": f1_report}
    _write(raw_path, {"kind": "evaluate", "action_id": "eval-4101", "seed": 4101,
                      "evaluation": {"reports": reports, "execution": execution},
                      "source_gate": E.RECOVERY.recovery(f1_report, none_report, bands)})
    receipt_path = folder / "receipt.json"
    _write(receipt_path, {
        "schema": E.SOURCE.RECEIPT_SCHEMA, "action_id": "eval-4101",
        "intent_sha256": _sha(intent_path), "raw_sha256": _sha(raw_path),
        "cost_after_usd": 2.0 + 2.0 / 60.0, "finished_at_utc": "2027-01-01T01:01:00+00:00",
    })
    evaluation = {"intent": _ref(root, intent_path), "raw": _ref(root, raw_path),
                  "receipt": _ref(root, receipt_path), "resource": _ref(root, resource_path),
                  "gpu_before_observation": _ref(root, before),
                  "gpu_after_observation": _ref(root, after)}
    return root, {"protocol_sha256_lf": "f" * 64, "input_locations": locations_path}, \
        source_protocol, rows, evaluation, folder


def test_unstarted_returns_one_exact_worker_argv_without_writes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    context = _context(tmp_path); _patch_rows(monkeypatch)
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    result = E.remaining_commands(context, datetime(2026, 10, 25, tzinfo=UTC_TZ))
    after = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    assert result["status"] == "runnable" and result["next_call_id"] == "choice-01"
    assert len(result["commands"]) == 1 and result["commands"][0][0] == "setsid"
    assert result["commands"][0][-2:] == ["--call-id", "choice-01"]
    assert result["api"] == {"cap": 72, "reserved": 0, "pending": 0, "remaining": 72}
    assert before == after


def test_terminal_replay_is_provider_free_and_completion_has_no_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    context = _context(tmp_path); _patch_rows(monkeypatch)
    for row in context["manifest"]["rows"]:
        path = context["stage"] / "decisions" / row["call_id"] / "terminal.json"
        path.parent.mkdir(parents=True)
        path.write_text("{}", encoding="utf-8")
    called = []
    monkeypatch.setattr(E.CALL, "_replay", lambda *args: called.append(args[2]["call_id"]) or {
        "engine_result": {"model_status": "valid", "selection": {"action": "hold", "probe": None, "fact_ids": ["x"]}}})
    monkeypatch.setattr(E, "_existing_journal", lambda *args: None)
    result = E.remaining_commands(context)
    assert result["status"] == "complete" and result["commands"] == []
    assert called == [f"choice-{i:02d}" for i in range(1, 25)]
    assert result["counts"]["terminal"] == 24


@pytest.mark.parametrize("owner", ["worker.identity.json", ".active", "api/.active"])
def test_owner_or_lease_blocks_without_deletion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, owner: str) -> None:
    context = _context(tmp_path); _patch_rows(monkeypatch)
    path = context["stage"] / owner; path.parent.mkdir(parents=True, exist_ok=True); path.write_text("owner", encoding="utf-8")
    result = E.remaining_commands(context)
    assert result["status"] == "blocked_pending" and result["commands"] == [] and path.read_text() == "owner"


def test_paused_infra_then_same_call_after_retry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    context = _context(tmp_path); _patch_rows(monkeypatch)
    decision = context["stage"] / "decisions/choice-01"; decision.mkdir(parents=True)
    (decision / "manifest.json").write_text("{}", encoding="utf-8")
    api = context["stage"] / "api"; api.mkdir(); (api / "manifest.json").write_text("{}", encoding="utf-8")
    retry = "2026-10-25T00:01:00+00:00"
    audit = {"reserved": 1, "pending": 0, "remaining": 71, "rows": [],
             "own": [{"response": {"retry_not_before": retry}}]}
    class J:
        def _verify_prepared(self): pass
        def _audit_api(self): return audit
    monkeypatch.setattr(E.CALL, "_journal", lambda *args: J())
    early = E.remaining_commands(context, datetime(2026, 10, 25, tzinfo=UTC_TZ))
    late = E.remaining_commands(context, datetime(2026, 10, 25, 0, 2, tzinfo=UTC_TZ))
    assert early["status"] == "paused_infra" and early["commands"] == []
    assert late["status"] == "runnable" and late["next_call_id"] == "choice-01"


def test_orphan_and_cap_exhaustion_never_emit_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    context = _context(tmp_path); _patch_rows(monkeypatch)
    decision = context["stage"] / "decisions/choice-01"; decision.mkdir(parents=True)
    (decision / "manifest.json").write_text("{}", encoding="utf-8")
    api = context["stage"] / "api"; api.mkdir(); (api / "manifest.json").write_text("{}", encoding="utf-8")
    class J:
        def __init__(self) -> None:
            self.audit = {"reserved": 1, "pending": 1, "remaining": 71, "rows": [], "own": []}
        def _verify_prepared(self): pass
        def _audit_api(self): return self.audit
    journal = J(); monkeypatch.setattr(E.CALL, "_journal", lambda *args: journal)
    assert E.remaining_commands(context)["status"] == "blocked_pending"
    journal.audit = {"reserved": 72, "pending": 0, "remaining": 0, "rows": [], "own": []}
    result = E.remaining_commands(context)
    assert result["status"] == "cap_exhausted" and result["commands"] == []


def test_project_results_keeps_semantic_and_infrastructure_distinct(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    context = _context(tmp_path); _patch_rows(monkeypatch)
    results = [
        {"model_status": "invalid", "hold_reason": "selection_invalid", "selection": {"action": "hold"}},
        {"model_status": "invalid", "hold_reason": "transport_timeout", "selection": {"action": "hold"}},
    ]
    for row, result in zip(context["manifest"]["rows"], results):
        path = context["stage"] / "decisions" / row["call_id"] / "terminal.json"; path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"engine_result": result}), encoding="utf-8")
    monkeypatch.setattr(E, "inspect_run", lambda *_: {"status": "x", "api": {"pending": 0}})
    monkeypatch.setattr(E.CALL, "_read_canonical", lambda path: json.loads(path.read_text()))
    rows = E.project_results(context)
    assert [row["model_status"] for row in rows] == ["semantic_invalid", "infrastructure_failed"]
    assert all("final_content" not in json.dumps(row) for row in rows)


def test_finalize_is_write_once_and_preview_writes_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    context = _context(tmp_path); evidence = tmp_path / "evidence.json"; evidence.write_text("{}", encoding="utf-8")
    out = tmp_path / "out"
    state = {"status": "complete", "api": {"remaining": 30}, "counts": {"terminal": 24}}
    monkeypatch.setattr(E, "inspect_run", lambda *_: state)
    monkeypatch.setattr(E, "project_results", lambda *_: [])
    monkeypatch.setattr(E, "_evidence", lambda *_: ({}, {}, "c" * 64))
    monkeypatch.setattr(E, "_usage", lambda *_: {"requests": 1, "tokens": {"known": 0, "missing": 1, "sum": 0}, "provider_wall_s_sum": 1.0})
    monkeypatch.setattr(E.COMPARISON, "summarize", lambda *args, **kwargs: {"schema": "summary", "execution_status": "inconclusive"})
    E.summarize(context, evidence, out, False)
    assert not out.exists()
    first = E.summarize(context, evidence, out, True)
    second = E.summarize(context, evidence, out, True)
    assert first == second and first["finalized"] is True
    (out / "choice_results.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError): E.summarize(context, evidence, out, True)


def test_evidence_manifest_protocol_sha_tamper_fails_before_summary(tmp_path: Path) -> None:
    context = _context(tmp_path)
    evidence = tmp_path / "evidence.json"
    evidence.write_text(json.dumps({"schema": E.EVIDENCE_SCHEMA, "protocol_sha256_lf": "0" * 64,
                                    "probe_execution": {}, "source_groups": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="actual protocol"):
        E._evidence(context, evidence)


def test_usage_reads_actual_safe_result_token_fields(tmp_path: Path) -> None:
    context = _context(tmp_path)
    responses = context["stage"] / "api/responses"; responses.mkdir(parents=True)
    rows = [
        {"result": {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10, "wall_s": 0.25}},
        {"result": {"infra_error": "TimeoutError", "wall_s": 0.5}},
    ]
    for index, row in enumerate(rows, 1):
        (responses / f"{index:04d}.json").write_text(json.dumps(row), encoding="utf-8")
    assert E._usage(context) == {
        "requests": 2, "tokens": {"known": 1, "missing": 1, "sum": 10},
        "provider_wall_s_sum": 0.75,
    }


def test_output_pair_refuses_partial_existing_state(tmp_path: Path) -> None:
    output = tmp_path / "out"; output.mkdir()
    (output / "comparison_summary.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="부분 생성"):
        E._check_output_pair(output, {"schema": "choice"}, {"schema": "summary"})
    assert not (output / "choice_results.json").exists()


def test_production_shaped_evaluate_evidence_binds_seed_artifacts_and_gpu_release(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_readiness(monkeypatch)
    root, context, source, rows, evaluation, _ = _evaluation_fixture(tmp_path)
    reports, artifacts = E._evaluate_evidence(root, context, "source-01", evaluation, source, rows)
    assert set(reports) == {"NONE", "F1"}
    assert artifacts["NONE"]["checkpoint"]["sha256"] == "a" * 64
    assert artifacts["F1"]["telemetry"]["sha256"] == "d" * 64


@pytest.mark.parametrize("mutation,match", [
    ("seed", "실행 완료"),
    ("incomplete", "실행 완료"),
    ("gpu", "GPU 관측"),
])
def test_evaluate_evidence_rejects_seed_incomplete_and_gpu_seal_tamper(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str, match: str) -> None:
    _patch_readiness(monkeypatch)
    root, context, source, rows, evaluation, folder = _evaluation_fixture(tmp_path)
    if mutation in {"seed", "incomplete"}:
        raw_path = root / evaluation["raw"]["path"]
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        if mutation == "seed":
            raw["seed"] = 4102
        else:
            raw["evaluation"]["execution"]["completed"] = False
        _write(raw_path, raw)
        evaluation["raw"] = _ref(root, raw_path)
        receipt_path = root / evaluation["receipt"]["path"]
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["raw_sha256"] = _sha(raw_path)
        _write(receipt_path, receipt)
        evaluation["receipt"] = _ref(root, receipt_path)
    else:
        (folder / "000002.process.csv").write_text("999, python\n", encoding="utf-8", newline="")
    with pytest.raises(ValueError, match=match):
        E._evaluate_evidence(root, context, "source-01", evaluation, source, rows)


def test_evaluate_evidence_rejects_f1_report_checkpoint_tamper_after_receipt_rehash(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_readiness(monkeypatch)
    root, context, source, rows, evaluation, _ = _evaluation_fixture(tmp_path)
    raw_path = root / evaluation["raw"]["path"]
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    raw["evaluation"]["reports"]["F1"]["checkpoint"]["sha256"] = "0" * 64
    _write(raw_path, raw); evaluation["raw"] = _ref(root, raw_path)
    receipt_path = root / evaluation["receipt"]["path"]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["raw_sha256"] = _sha(raw_path)
    _write(receipt_path, receipt); evaluation["receipt"] = _ref(root, receipt_path)
    with pytest.raises(ValueError, match="checkpoint.sha256"):
        E._evaluate_evidence(root, context, "source-01", evaluation, source, rows)


def test_portable_probe_verifier_accepts_production_shaped_brev_actual_receipt(tmp_path: Path) -> None:
    item, snapshot, paths = _brev_probe_fixture(tmp_path)
    verified, digest = E._portable_probe(
        tmp_path, item, snapshot, "b" * 64, "a" * 64, _expected_brev_source())
    assert verified["outcome"] == "abnormal" and digest == _sha(paths["seal"])
    assert verified["evidence_kind"] == E.CONTROLLER.ACTUAL_EVIDENCE


def test_evidence_accepts_stored_brev_v4_originals_with_separate_probe_protocol(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture_path = os.environ.get("RL_TRIAGE_BREV_PROBE_FIXTURE")
    if fixture_path is None:
        pytest.skip("저장된 Brev 원문 통합 검사에는 RL_TRIAGE_BREV_PROBE_FIXTURE가 필요하다")
    stored = Path(fixture_path).resolve(strict=True)
    retrieved = tmp_path / "actual"
    shutil.copytree(stored, retrieved)
    protocol_path = tmp_path / "bench/protocols/brev_recovery_v4.json"
    protocol_path.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / "bench/protocols/brev_recovery_v4.json", protocol_path)
    jobs = {side: json.loads((retrieved / side / "job.json").read_text(encoding="utf-8"))
            for side in ("case", "reference")}
    workers = {side: json.loads((retrieved / side / "worker.json").read_text(encoding="utf-8"))
               for side in ("case", "reference")}
    bindings = {
        "case": {"checkpoint_sha256": jobs["case"]["checkpoint_sha256"], "telemetry_sha256": "d" * 64},
        "reference": {"checkpoint_sha256": jobs["reference"]["checkpoint_sha256"],
                      "telemetry_sha256": "c" * 64},
    }
    work = tmp_path / "approval"
    packet = E.CONTROLLER.S.build_packet(epoch=1, applied_corrections=[], prior_receipt=None,
                                         facts={"config:action_scale": 0.25}, input_bindings=bindings,
                                         execution_budget_s=1200)
    snapshot = E.CONTROLLER.V.build_snapshot(
        packet, "stored_actual", "fixed_eligible", E.CONTROLLER.V.file_digest(E.CONTROLLER.V.PROTOCOL),
        work / "registered-f1/epoch1/probes.ledger.jsonl", None,
    )
    choice = E.CONTROLLER.choose(snapshot, "fixed_eligible")
    assert choice["probe"] == jobs["case"]["probe"] == "P_episode"
    raw = work / "raw"; raw.mkdir(parents=True)
    case_raw, reference_raw = raw / "case.json", raw / "reference.json"
    _write(case_raw, workers["case"]["measurement"])
    _write(reference_raw, workers["reference"]["measurement"])
    seal_ref = E.CONTROLLER.record_probe(
        work, "registered-f1", 1, snapshot, choice, case_raw, reference_raw, "cpu", 23.0,
    )
    seal_path = Path(seal_ref["path"])
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    auth_path = Path(seal["authorization_path"])
    auth = json.loads(auth_path.read_text(encoding="utf-8"))
    descriptors = {}
    descriptor_paths = {}
    filenames = {"job": "job.json", "worker": "worker.json", "execution": "execution.json",
                 "backend_result": "backend_result.json", "report_jobs": "report.jobs.json",
                 "report": "report.json"}
    for side in ("case", "reference"):
        descriptor = {
            "schema": "vision_selection_brev_probe_evidence_v1", "side": side,
            "probe": jobs[side]["probe"], "checkpoint_sha256": jobs[side]["checkpoint_sha256"],
            "sources": {name: _ref(tmp_path, retrieved / side / filename)
                        for name, filename in filenames.items()},
        }
        descriptor_path = tmp_path / f"{side}_descriptor.json"
        _write(descriptor_path, descriptor)
        descriptors[side] = descriptor
        descriptor_paths[side] = descriptor_path
    auth["evidence_kind"] = E.CONTROLLER.ACTUAL_EVIDENCE
    auth["execution_plan"] = {
        "stage": jobs["case"]["stage"], "run_name": jobs["case"]["run_name"],
        "checkpoint_sha256": jobs["case"]["checkpoint_sha256"], "seed": jobs["case"]["seed"],
        "args": auth["execution_plan"]["args"], "gpu_contract": "residual_v1",
        "reference": {"evidence_path": "/retrieved/reference_descriptor.json",
                      "evidence_sha256": _sha(descriptor_paths["reference"]),
                      "checkpoint_sha256": jobs["reference"]["checkpoint_sha256"],
                      "stage": jobs["reference"]["stage"], "run_name": jobs["reference"]["run_name"],
                      "seed": jobs["reference"]["seed"], "gpu_contract": "residual_v1"},
    }
    _write(auth_path, auth)
    seal.update(evidence_kind=E.CONTROLLER.ACTUAL_EVIDENCE, authorization_sha256=_sha(auth_path),
                case_path="/retrieved/case_descriptor.json", case_sha256=_sha(descriptor_paths["case"]),
                reference_path="/retrieved/reference_descriptor.json",
                reference_sha256=_sha(descriptor_paths["reference"]))
    _write(seal_path, seal)
    probe_item = {name: _ref(tmp_path, path) for name, path in {
        "seal": seal_path, "authorization": auth_path,
        "ledger": auth_path.parent / "probes.ledger.jsonl",
        "case": descriptor_paths["case"], "reference": descriptor_paths["reference"],
    }.items()}
    source_protocol = tmp_path / "source_protocol.json"
    bands = tmp_path / "bands.json"; _write(bands, {"bands": {}})
    _write(source_protocol, {"evaluation": {"bands_path": "bands.json"}})
    source_protocol_sha = E.WORKER._sha_lf(source_protocol)
    rows = []
    for group in E.COMPARISON.SEED_GROUPS:
        rows.extend([
            {"seed_group": group, "source_role": "normal",
             "checkpoint_sha256": jobs["reference"]["checkpoint_sha256"],
             "source_protocol_sha256": source_protocol_sha},
            {"seed_group": group, "source_role": "registered-f1",
             "checkpoint_sha256": jobs["case"]["checkpoint_sha256"],
             "source_protocol_sha256": source_protocol_sha},
        ])
    context = {"protocol_sha256_lf": "f" * 64, "prepared": {"protocol": {
        "rows": rows, "source_protocol": {"path": "source_protocol.json", "sha256_lf": source_protocol_sha},
    }}}
    evidence_path = tmp_path / "evidence.json"
    _write(evidence_path, {
        "schema": E.EVIDENCE_SCHEMA, "protocol_sha256_lf": context["protocol_sha256_lf"],
        "probe_execution": {"source_anchor": _expected_brev_source()["source_anchor"],
                            "protocol": {"path": "bench/protocols/brev_recovery_v4.json",
                                         "sha256_lf": E.WORKER._sha_lf(protocol_path)}},
        "source_groups": {group: {"evaluation": {}, "probe_evidence": probe_item}
                          for group in E.COMPARISON.SEED_GROUPS},
    })
    monkeypatch.setattr(E, "ROOT", tmp_path)
    monkeypatch.setattr(E, "_evaluate_evidence", lambda *_: ({"NONE": {}, "F1": {}}, {}))
    monkeypatch.setattr(E.RECOVERY, "recovery", lambda *_: {"outcome": "not_recovered"})
    labels, probes, digest = E._evidence(context, evidence_path)
    assert set(labels) == set(probes) == set(E.COMPARISON.SEED_GROUPS)
    assert all(row["outcome"] == "normal" and row["usable"] for row in probes.values())
    assert digest == source_protocol_sha


def test_portable_probe_rejects_malformed_registry_ledger(tmp_path: Path) -> None:
    item, snapshot, paths = _brev_probe_fixture(tmp_path)
    rows = paths["ledger"].read_text(encoding="utf-8").splitlines()
    event = json.loads(rows[1]); event["registry_digest"] = "0" * 64; rows[1] = json.dumps(event)
    paths["ledger"].write_text("\n".join(rows) + "\n", encoding="utf-8")
    item["ledger"] = _ref(tmp_path, paths["ledger"])
    with pytest.raises((ValueError, RuntimeError)):
        E._portable_probe(tmp_path, item, snapshot, "b" * 64, "a" * 64, _expected_brev_source())


def test_portable_probe_rejects_brev_actual_provenance_tamper(tmp_path: Path) -> None:
    item, snapshot, paths = _brev_probe_fixture(tmp_path)
    execution = json.loads(paths["case_execution"].read_text(encoding="utf-8"))
    execution["cleanup_verified"] = False
    _write(paths["case_execution"], execution)
    descriptor = json.loads(paths["case"].read_text(encoding="utf-8"))
    descriptor["sources"]["execution"] = _ref(tmp_path, paths["case_execution"])
    _write(paths["case"], descriptor)
    auth = json.loads(paths["authorization"].read_text(encoding="utf-8"))
    auth["execution_plan"]["reference"]["evidence_sha256"] = _sha(paths["reference"])
    _write(paths["authorization"], auth)
    seal = json.loads(paths["seal"].read_text(encoding="utf-8"))
    seal["authorization_sha256"] = _sha(paths["authorization"])
    seal["case_sha256"] = _sha(paths["case"])
    _write(paths["seal"], seal)
    item["case"] = _ref(tmp_path, paths["case"]); item["seal"] = _ref(tmp_path, paths["seal"])
    with pytest.raises(ValueError, match="Brev actual probe"):
        E._portable_probe(tmp_path, item, snapshot, "b" * 64, "a" * 64, _expected_brev_source())


def test_portable_probe_rejects_rehashed_foreign_source_and_stage(tmp_path: Path) -> None:
    item, snapshot, paths = _brev_probe_fixture(tmp_path)
    for side in ("case", "reference"):
        job = json.loads(paths[f"{side}_job"].read_text(encoding="utf-8"))
        execution = json.loads(paths[f"{side}_execution"].read_text(encoding="utf-8"))
        job["source_commit_sha"] = "9" * 40
        source = execution["source_before"]
        source["source_anchor"] = "9" * 40
        execution["source_after"] = json.loads(json.dumps(source))
        if side == "case":
            job["stage"] = "/foreign/probes/case"
        _write(paths[f"{side}_job"], job); _write(paths[f"{side}_execution"], execution)
        backend = json.loads(paths[f"{side}_backend_result"].read_text(encoding="utf-8"))
        backend["execution"] = execution; _write(paths[f"{side}_backend_result"], backend)
        descriptor = json.loads(paths[side].read_text(encoding="utf-8"))
        for name in ("job", "execution", "backend_result"):
            descriptor["sources"][name] = _ref(tmp_path, paths[f"{side}_{name}"])
        _write(paths[side], descriptor)
    auth = json.loads(paths["authorization"].read_text(encoding="utf-8"))
    auth["execution_plan"]["reference"]["evidence_sha256"] = _sha(paths["reference"])
    _write(paths["authorization"], auth)
    seal = json.loads(paths["seal"].read_text(encoding="utf-8"))
    seal["authorization_sha256"] = _sha(paths["authorization"])
    seal["case_sha256"] = _sha(paths["case"]); seal["reference_sha256"] = _sha(paths["reference"])
    _write(paths["seal"], seal)
    item["authorization"] = _ref(tmp_path, paths["authorization"])
    item["case"] = _ref(tmp_path, paths["case"]); item["reference"] = _ref(tmp_path, paths["reference"])
    item["seal"] = _ref(tmp_path, paths["seal"])
    with pytest.raises(ValueError, match="Brev actual probe"):
        E._portable_probe(tmp_path, item, snapshot, "b" * 64, "a" * 64, _expected_brev_source())


def test_portable_probe_rejects_rehashed_foreign_task(tmp_path: Path) -> None:
    item, snapshot, paths = _brev_probe_fixture(tmp_path)
    for side in ("case", "reference"):
        jobs = json.loads(paths[f"{side}_report_jobs"].read_text(encoding="utf-8"))
        report = json.loads(paths[f"{side}_report"].read_text(encoding="utf-8"))
        jobs["task"] = "Foreign-Task-v0"
        report["task"] = "Foreign-Task-v0"
        _write(paths[f"{side}_report_jobs"], jobs)
        _write(paths[f"{side}_report"], report)
        worker = json.loads(paths[f"{side}_worker"].read_text(encoding="utf-8"))
        worker["report"]["sha256"] = _sha(paths[f"{side}_report"])
        _write(paths[f"{side}_worker"], worker)
        backend = json.loads(paths[f"{side}_backend_result"].read_text(encoding="utf-8"))
        backend["report"] = report
        _write(paths[f"{side}_backend_result"], backend)
        descriptor = json.loads(paths[side].read_text(encoding="utf-8"))
        for name in ("worker", "backend_result", "report_jobs", "report"):
            descriptor["sources"][name] = _ref(tmp_path, paths[f"{side}_{name}"])
        _write(paths[side], descriptor)
    auth = json.loads(paths["authorization"].read_text(encoding="utf-8"))
    auth["execution_plan"]["reference"]["evidence_sha256"] = _sha(paths["reference"])
    _write(paths["authorization"], auth)
    seal = json.loads(paths["seal"].read_text(encoding="utf-8"))
    seal["authorization_sha256"] = _sha(paths["authorization"])
    seal["case_sha256"] = _sha(paths["case"])
    seal["reference_sha256"] = _sha(paths["reference"])
    _write(paths["seal"], seal)
    for name in ("authorization", "case", "reference", "seal"):
        item[name] = _ref(tmp_path, paths[name])
    with pytest.raises(ValueError, match="Brev actual probe"):
        E._portable_probe(tmp_path, item, snapshot, "b" * 64, "a" * 64, _expected_brev_source())


def test_datetime_compatibility_avoids_python311_utc_symbol() -> None:
    forbidden = "from datetime import " + "UTC"
    assert forbidden not in (ROOT / "evals/vision_selection_execution_v1.py").read_text(encoding="utf-8")
    assert forbidden not in Path(__file__).read_text(encoding="utf-8")
