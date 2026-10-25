from __future__ import annotations

import importlib
import importlib.util
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / "evals", ROOT / "bench", ROOT / "src", ROOT / "tests"):
    sys.path.insert(0, str(folder))

A = importlib.import_module("vision_selection_brev_adapter_v1")
CALL = importlib.import_module("vision_selection_call_v1")
C = importlib.import_module("vision_selection_e2e_caller_v1")
F = importlib.import_module("test_vision_selection_contract_v1")
G = importlib.import_module("test_grounded_vision_observer_v1")
AF = importlib.import_module("test_vision_selection_brev_adapter_v1")
GRF = importlib.import_module("test_grounded_recovery")


def write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return path


def setup_caller(tmp_path: Path) -> tuple[Any, dict, Path, dict, dict]:
    contract, binding, snapshot = F.setup_case(tmp_path / "vision")
    frozen = CALL.freeze(snapshot, contract, binding, "text_and_video", tmp_path / "capsule")
    selection_plan = CALL.initialize(
        tmp_path, "caller-fixture", "e2e", "a" * 40, "b" * 64,
        [{"call_id": f"e2e-source-0{index}",
          "run_id": f"vision-selection-caller-fixture-e2e-source-0{index}", "capsule": frozen}
         for index in (1, 2)],
    )
    prepared = CALL.prepared(frozen)
    checkpoint = prepared["checkpoint_sha256"]
    rows = []
    for index in (1, 2):
        seed = 2200 + index
        rows.append({
            "target": {"slot_id": f"input-0{index}", "seed_group": f"source-0{index}",
                       "source_role": "registered-f1", "window": "window-01",
                       "view_kind": "text_and_video"},
            "call_id": f"e2e-source-0{index}", "source_call_id": f"choice-{index:02d}",
            "source_seed": seed, "source_job_id": f"s{seed}-f1",
            "record_job_id": f"record-{seed}-f1",
            "source_lineage_id": f"record-{seed}-f1-same-play",
            "corrected_job_id": f"vision-s{seed}-corrected",
            "checkpoint_sha256": checkpoint, "capsule_sha256": frozen["sha256"],
            "reference_checkpoint_sha256": f"{index + 8:064x}",
            "eligible_probes": list(prepared["bundle"]["selection_view"]["eligible_probes"]),
            "source_protocol_sha256_lf": f"{index + 12:064x}",
        })
    plan = {
        "schema": "vision_selection_e2e_plan_v1", "api_cap": 32,
        "execution_id": "caller-fixture", "source_anchor": "a" * 40,
        "selection_plan": selection_plan, "rows": rows,
    }
    plan_path = write(tmp_path / "e2e-plan.json", plan)
    corrected_checkpoint = tmp_path / "corrected/model_299.pt"
    corrected_checkpoint.parent.mkdir()
    corrected_checkpoint.write_bytes(b"registered-corrected-checkpoint")
    corrected_sha = A.file_sha(corrected_checkpoint)
    service_file = tmp_path / "fixture_service.py"
    service_file.write_text(
        "class FixtureService:\n"
        "    def __init__(self, artifact=None, evaluation=None, record_value=None, training_evidence_value=None):\n"
        "        self.artifact=artifact; self.evaluation=evaluation; self.record_value=record_value; self.training_evidence_value=training_evidence_value; self.backend=self; self.train_calls=0; self.evaluate_calls=0\n"
        "    @classmethod\n"
        "    def from_runtime(cls, *args, **kwargs):\n"
        "        return cls()\n"
        "    def preflight(self):\n"
        "        return True\n"
        "    def train(self, job_id, *, approval_reference):\n"
        "        self.train_calls += 1; assert approval_reference['request_id']; return self.artifact\n"
        "    def training_evidence(self, job_id, artifact):\n"
        "        assert artifact == self.artifact; return self.training_evidence_value\n"
        "    def evaluate(self, artifact):\n"
        "        self.evaluate_calls += 1; assert artifact == self.artifact; return self.evaluation\n"
        "    def record(self, job):\n"
        "        assert job['checkpoint']['sha256'] == self.artifact['checkpoint']['sha256']\n"
        "        return self.record_value\n",
        encoding="utf-8",
    )
    groups = [{
        "group": row["target"]["seed_group"], "seed": row["source_seed"],
        "source_protocol_sha256_lf": row["source_protocol_sha256_lf"],
        "checkpoints": {"NONE": row["reference_checkpoint_sha256"],
                        "F1": row["checkpoint_sha256"]},
        "corrected_job_id": row["corrected_job_id"], "probe_budget_gpu_s": 120,
        "action_limits": {name: {"max_wall_s": 3600, "max_compute_usd": 1.8}
                          for name in ("probe_pair", "correction", "record")},
        "execution": {
            "NONE": {"stage": f"none-stage-{index}", "run_name": f"none-run-{index}"},
            "F1": {"stage": f"f1-stage-{index}", "run_name": f"f1-run-{index}"},
        },
    } for index, row in enumerate(rows, 1)]
    corrected_groups = [{"group": row["target"]["seed_group"], "job": {
        "job_id": f"record-{row['source_seed']}-corrected",
        "source_job_id": row["corrected_job_id"],
        "checkpoint": {"path": str(corrected_checkpoint), "sha256": corrected_sha},
    }} for row in rows]
    context = AF.context(
        tmp_path, groups=groups, source_plan_sha256=A.file_sha(plan_path),
        selection_plan_sha256=A.file_sha(Path(selection_plan["path"])),
        source_plan_path=plan_path, selection_plan_path=Path(selection_plan["path"]),
        service_file=service_file, corrected_groups=corrected_groups,
    )
    caller = C.E2ECaller(
        tmp_path, plan_path, context,
        tmp_path / ".triage-runtime/vision-selection-e2e/caller-fixture",
    )
    spec = importlib.util.spec_from_file_location("fixture_service", service_file)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    expected_source = A._expected_correction_source(context)
    source_identity = {
        key: expected_source[key] for key in (
            "lab_head", "lab_tracked_clean", "source_anchor", "protocol", "environment",
            "source_sha256_lf", "runtime_source_sha256_lf",
            "python_package_source_sha256_lf",
        )
    }
    source_identity["selected_interpreter"] = {
        "python": expected_source["selected_python"],
        "packages": expected_source["selected_packages"],
        "launcher": "/registered/python", "executable": "/registered/python",
        "prefix": "/registered", "base_prefix": "/base",
    }
    def execution(wall_s: float) -> dict:
        return {"completed": True, "exit_code": 0, "timed_out": False,
                "cleanup_verified": True, "gpu_before": [], "gpu_after": [],
                "source_before": source_identity, "source_after": source_identity,
                "wall_s": wall_s, "finished_at": "2026-10-25T01:00:00+00:00"}
    training_dir = tmp_path / ".triage-runtime/fixture-training"
    telemetry = write(training_dir / "telemetry.json", {"seed": rows[0]["source_seed"]})
    manifest = write(training_dir / "manifest.json", {"kind": "fixture-training"})
    readback = write(training_dir / "readback.json", {"ok": True})
    records = training_dir / "records.jsonl"
    records.write_text("{}\n", encoding="utf-8")
    training_execution = write(training_dir / "execution.json", execution(10.0))
    artifact = {"arm": "CORRECTED", "seed": rows[0]["source_seed"],
                "source_checkpoint_sha256": rows[0]["checkpoint_sha256"],
                "checkpoint": {"path": str(corrected_checkpoint), "sha256": corrected_sha},
                "telemetry": {"path": str(telemetry), "sha256": A.file_sha(telemetry)},
                "manifest_path": str(manifest), "manifest_sha256": A.file_sha(manifest),
                "manifest": C._read(manifest), "readback": C._read(readback),
                "execution": C._read(training_execution)}
    artifact_path = write(training_dir / "adapter_artifact.json", artifact)
    inventory = {
        "checkpoint": {"path": str(corrected_checkpoint), "sha256": corrected_sha},
        "telemetry": {"path": str(telemetry), "sha256": A.file_sha(telemetry)},
        "manifest": {"path": str(manifest), "sha256": A.file_sha(manifest)},
        "required_files": {name: {"path": str(path), "sha256": A.file_sha(path)}
                           for name, path in (("readback", readback), ("records", records),
                                              ("execution", training_execution))},
    }
    completed_event = {"schema": "vision_brev_training_ledger_event_v1",
                       "event": "completed", "job_id": rows[0]["corrected_job_id"],
                       "plan_sha256": "f" * 64, "artifact_path": str(artifact_path),
                       "artifact_sha256": A.file_sha(artifact_path), "inventory": inventory}
    ledger = training_dir / "training.ledger.jsonl"
    ledger_rows = [
        {"schema": "vision_brev_training_ledger_event_v1", "event": event,
         "job_id": rows[0]["corrected_job_id"], "plan_sha256": "f" * 64}
        for event in ("reserved", "started")
    ] + [completed_event]
    ledger.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in ledger_rows),
                      encoding="utf-8")
    training_evidence = {
        "schema": "vision_selection_training_evidence_v1",
        "ledger": A._portable_reference(tmp_path, ledger),
        "completed_event_sha256": A.digest(completed_event),
        "files": {
            "artifact": A._portable_reference(tmp_path, artifact_path),
            "checkpoint": A._portable_reference(tmp_path, corrected_checkpoint),
            "telemetry": A._portable_reference(tmp_path, telemetry),
            "manifest": A._portable_reference(tmp_path, manifest),
            "readback": A._portable_reference(tmp_path, readback),
            "records": A._portable_reference(tmp_path, records),
            "execution": A._portable_reference(tmp_path, training_execution),
        },
    }
    protocol_sha = context["protocol"]["execution_bindings"]["fixed_eval_protocol"]["sha256_lf"]
    reports = {
        "none": GRF._report("none", checkpoint=rows[0]["reference_checkpoint_sha256"]),
        "corrected": GRF._report("corrected", checkpoint=corrected_sha),
    }
    for report in reports.values():
        report["protocol_sha256_lf"] = protocol_sha
    fixed_dir = tmp_path / ".triage-runtime/fixture-fixed"
    none_report = write(fixed_dir / "reports/NONE.json", reports["none"])
    corrected_report = write(fixed_dir / "reports/CORRECTED.json", reports["corrected"])
    fixed_execution = write(fixed_dir / "execution.json", execution(5.0))
    fixed_job = write(fixed_dir / "job.json", {
        "mode": "evaluate",
        "artifacts": {
            "NONE": {"checkpoint": {"sha256": rows[0]["reference_checkpoint_sha256"]}},
            "CORRECTED": {"checkpoint": {"sha256": corrected_sha}},
        },
    })
    fixed_worker = write(fixed_dir / "worker.json", {
        "completed": True,
        "reports": {
            "NONE": {"path": str(none_report), "sha256": A.file_sha(none_report)},
            "CORRECTED": {"path": str(corrected_report), "sha256": A.file_sha(corrected_report)},
        },
    })
    fixed_evidence = {name: A._portable_reference(tmp_path, path) for name, path in (
        ("job", fixed_job), ("worker", fixed_worker), ("execution", fixed_execution),
        ("none_report", none_report), ("corrected_report", corrected_report),
    )}
    service = module.FixtureService(
        artifact, {"reports": reports, "execution": execution(5.0), "evidence": fixed_evidence},
        training_evidence_value=training_evidence,
    )
    caller._fixture_service = service
    return caller, prepared, contract, binding, snapshot


def valid_provider(prepared: dict) -> dict:
    bundle = prepared["bundle"]
    fact_id = next(iter(bundle["selection_view"]["evidence_facts"]))
    probe = "P_reward" if "P_reward" in bundle["selection_view"]["eligible_probes"] \
        else bundle["selection_view"]["eligible_probes"][0]
    return G._provider(json.dumps({"action": "continue", "probe": probe, "fact_ids": [fact_id]}))


def canonical_observer_contract(caller: Any, tmp_path: Path, group: str = "source-01") -> Path:
    source = tmp_path / f"observer-{group}"
    source.mkdir()
    prepared = G._contract(source)
    destination = caller.runtime / "post-observation" / group / "evidence"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(prepared.parent), destination)
    return destination / "contract.json"


def selected(caller: Any, prepared: dict, contract: Path,
             binding: dict, snapshot: dict) -> dict:
    return caller.run_initial(
        "source-01", lambda *_: valid_provider(prepared), live_snapshot=snapshot,
        live_contract=contract, source_binding=binding,
    )


def reward_pair(caller: Any, selection: dict) -> dict:
    row = caller._row("source-01")
    common = {
        "job_sha256": "1" * 64, "worker_sha256": "2" * 64,
        "execution_sha256": "3" * 64, "report_jobs_sha256": "4" * 64,
        "report_sha256": "5" * 64, "gpu_before_sha256": "6" * 64,
        "gpu_after_sha256": "7" * 64, "authorization_sha256": "8" * 64,
        "seal_sha256": "9" * 64,
    }
    return {
        "schema": A.PAIR_RESULT_SCHEMA, "status": "complete", "probe": "P_reward",
        "request_sha256": "d" * 64, "seed": row["source_seed"], "gpu_runs": 2,
        "NONE": {"outcome": "normal", "measurement": {}},
        "F1": {"outcome": "abnormal", "measurement": {}},
        "none": {**common, "status": "complete", "outcome": "normal",
                 "seed": row["source_seed"], "arm": "NONE",
                 "checkpoint_sha256": row["reference_checkpoint_sha256"]},
        "f1": {**common, "status": "complete", "outcome": "abnormal",
               "seed": row["source_seed"], "arm": "F1",
               "checkpoint_sha256": row["checkpoint_sha256"]},
        "selection_sha256": C.digest(selection),
    }


def recovered_correction(caller: Any, group: str = "source-01") -> dict:
    selection = C._read(caller._selection_path(group))
    proposal = caller.propose_correction(group, reward_pair(caller, selection))
    assert proposal is not None
    service = caller._fixture_service
    with pytest.raises(ValueError, match="correction backend"):
        caller.approve_and_run_correction(
            group, proposal["request_id"], approver="fixture",
            backend_reference={"path": "evals/foreign.py", "sha256_lf": "0" * 64},
            train=service.train, evaluate=service.evaluate,
        )
    assert caller.corrections.requests()[proposal["request_id"]]["state"] == "pending"
    with pytest.raises(ValueError, match="correction service"):
        caller.approve_and_run_correction(
            group, proposal["request_id"], approver="fixture",
            backend_reference=caller.context["protocol"]["execution_bindings"]["correction_backend"],
            train=lambda *_args, **_kwargs: service.artifact,
            evaluate=lambda _artifact: service.evaluation,
        )
    assert caller.corrections.requests()[proposal["request_id"]]["state"] == "approved"
    result = caller.run_approved_correction(
        group, proposal["request_id"],
        backend_reference=caller.context["protocol"]["execution_bindings"]["correction_backend"],
        train=service.train, evaluate=service.evaluate,
    )
    cached = caller.approve_and_run_correction(
        group, proposal["request_id"], approver="fixture",
        backend_reference=caller.context["protocol"]["execution_bindings"]["correction_backend"],
        train=service.train, evaluate=service.evaluate,
    )
    assert cached == result
    return result


def test_initial_selection_replays_without_second_api_side_effect(tmp_path: Path) -> None:
    caller, prepared, contract, binding, snapshot = setup_caller(tmp_path)
    first = selected(caller, prepared, contract, binding, snapshot)
    assert first["model_status"] == "valid" and first["probe"] == "P_reward"
    assert caller.api_requests() == 1
    replay = caller.run_initial(
        "source-01", lambda *_: (_ for _ in ()).throw(AssertionError("terminal replay")),
        live_snapshot=snapshot, live_contract=contract, source_binding=binding,
    )
    assert replay == first and caller.api_requests() == 1


def test_caller_keeps_job_lineage_ids_separate_and_rejects_protocol_mismatch(tmp_path: Path) -> None:
    caller, *_ = setup_caller(tmp_path)
    row = caller._row("source-01")
    assert len({row["source_job_id"], row["record_job_id"], row["source_lineage_id"],
                row["corrected_job_id"]}) == 4
    mismatched = dict(caller.context)
    mismatched["protocol"] = json.loads(json.dumps(caller.context["protocol"]))
    mismatched["protocol"]["source_anchor"] = "f" * 40
    with pytest.raises(ValueError, match="selection plan"):
        C.E2ECaller(caller.root, caller.plan_path, mismatched, caller.runtime)
    plan = json.loads(caller.plan_path.read_text(encoding="utf-8"))
    plan["rows"][0]["corrected_job_id"] = plan["rows"][0]["source_job_id"]
    caller.plan_path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(ValueError, match="plan SHA"):
        C.E2ECaller(caller.root, caller.plan_path, caller.context, caller.runtime)


def test_only_reward_abnormal_pair_can_create_and_consume_registered_correction(tmp_path: Path) -> None:
    caller, prepared, contract, binding, snapshot = setup_caller(tmp_path)
    selection = selected(caller, prepared, contract, binding, snapshot)
    pair = reward_pair(caller, selection)
    other = json.loads(json.dumps(pair)); other["probe"] = "P_value"
    assert caller.propose_correction("source-01", other) is None
    tampered = json.loads(json.dumps(pair)); tampered["f1"]["checkpoint_sha256"] = "0" * 64
    assert caller.propose_correction("source-01", tampered) is None
    result = recovered_correction(caller)
    assert result["fixed_eval"] == {"none": "healthy", "corrected": "healthy"}
    events = caller.corrections.events()
    assert [row["event"] for row in events] == ["proposed", "approved", "consumed", "receipt"]


def test_caller_admission_roundtrip_writes_pair_prefix_and_rejects_mock_cache(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    caller, prepared, contract, binding, snapshot = setup_caller(tmp_path)
    assert caller.remaining_commands()[0]["kind"] == "initial_selection"
    selection = selected(caller, prepared, contract, binding, snapshot)
    assert caller.remaining_commands()[0]["kind"] == "probe_pair"
    dispatched = {"count": 0}

    def admitted(context: dict, reference: dict, commands: dict, expected_source: dict,
                 stage: Path, **_: object) -> dict:
        del commands, expected_source, stage
        dispatched["count"] += 1
        request = A.verify_pair_prefix(context["root"], reference)
        A._registered_pair(context, request)
        result = reward_pair(caller, selection)
        result["request_sha256"] = A.digest(request)
        return result

    monkeypatch.setattr(A, "run_probe_pair", admitted)
    command = caller.remaining_commands()[0]
    stage = caller.runtime / "probes/source-01"
    services = {
        "commands": {
            "NONE": {"argv": ["fixture-none"], "descriptor": str(stage / "none.json")},
            "F1": {"argv": ["fixture-f1"], "descriptor": str(stage / "f1.json")},
        }, "expected_source": caller._expected_probe_source(), "approver": "fixture",
    }
    pending = caller.advance(command, {**services, "approver": None})
    assert pending["status"] == "approval_required" and dispatched["count"] == 0
    assert [row["event"] for row in caller.pairs.events()] == ["proposed"]
    with pytest.raises(ValueError, match="approver"):
        caller.advance(command, {**services, "approver": "  "})
    assert [row["event"] for row in caller.pairs.events()] == ["proposed"]
    assert dispatched["count"] == 0
    result = caller.advance(command, services)
    assert result["status"] == "complete" and dispatched["count"] == 1
    assert [row["event"] for row in caller.pairs.events()] == [
        "proposed", "approved", "consumed", "receipt",
    ]
    with pytest.raises((KeyError, TypeError, ValueError)):
        caller.remaining_commands()
    with pytest.raises((TypeError, ValueError), match="descriptor"):
        caller.run_probe_pair("source-01", {"NONE": {}, "F1": {}},
                              caller._expected_probe_source(), approver="fixture")
    assert dispatched["count"] == 1


def test_partial_probe_and_selection_tamper_never_write_approval_or_dispatch(
        tmp_path: Path) -> None:
    caller, prepared, contract, binding, snapshot = setup_caller(tmp_path)
    selected(caller, prepared, contract, binding, snapshot)
    stage = caller.runtime / "probes/source-01"
    stage.mkdir(parents=True); (stage / "orphan.json").write_text("{}\n", encoding="utf-8")
    result = caller.run_probe_pair(
        "source-01", {"NONE": {}, "F1": {}}, caller._expected_probe_source(),
        approver="fixture", runner=lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError()),
    )
    assert result["status"] == "HOLD" and result["reason"] == "partial_stage"
    assert caller.pairs.events() == []
    assert not (caller.runtime / "probe_pairs/source-01.json").exists()
    assert caller.inspect()["probes"] == {}
    assert caller.remaining_commands()[0]["kind"] == "probe_pair"

    selection_path = caller._selection_path("source-01")
    tampered = json.loads(selection_path.read_text(encoding="utf-8"))
    tampered["terminal_sha256"] = "0" * 64
    write(selection_path, tampered)
    with pytest.raises(ValueError, match="terminal/activation"):
        caller.remaining_commands()
    assert caller.pairs.events() == []


def test_blank_correction_approver_does_not_write_ledger(tmp_path: Path) -> None:
    caller, prepared, contract, binding, snapshot = setup_caller(tmp_path)
    selection = selected(caller, prepared, contract, binding, snapshot)
    proposal = caller.propose_correction("source-01", reward_pair(caller, selection))
    assert proposal is not None
    before = caller.corrections.events()
    with pytest.raises(ValueError, match="approver"):
        caller.approve_correction(proposal["request_id"], " ")
    assert caller.corrections.events() == before


def test_recovered_record_and_aux_vl_share_api32_but_never_decide_recovery(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    caller, prepared, contract, binding, snapshot = setup_caller(tmp_path)
    selected(caller, prepared, contract, binding, snapshot)
    recovered_correction(caller)
    observer_contract = canonical_observer_contract(caller, tmp_path)
    callbacks = {"input": 0}
    record_job = caller._record_job("source-01", caller._cached_correction("source-01"))
    write(tmp_path / "record/job.json", record_job)
    record_result_path = write(tmp_path / "record/result.json", {
        "completed": True,
        "checkpoint_sha256": caller._fixture_service.artifact["checkpoint"]["sha256"],
    })
    record_execution_path = write(tmp_path / "record/execution.json", {
        "completed": True, "exit_code": 0, "timed_out": False, "cleanup_verified": True,
        "gpu_before": [], "gpu_after": [], "source_before": {"anchor": "fixture"},
        "source_after": {"anchor": "fixture"}, "wall_s": 2.0,
        "finished_at": "2026-10-25T01:00:00+00:00",
        "result_sha256": A.file_sha(record_result_path),
    })
    caller._fixture_service.record_value = {
        "execution": AF.ref(tmp_path, record_execution_path),
        "result": AF.ref(tmp_path, record_result_path),
    }

    def observation_input(value: dict) -> tuple[Path, dict]:
        callbacks["input"] += 1
        return observer_contract, G.G.binding_template(
            observer_contract, run_id="caller-fixture-post-source-01", source_anchor=caller.plan["source_anchor"],
            protocol_sha256=caller.context["protocol_sha256"],
        )

    with pytest.raises(ValueError, match="record verifier"):
        caller.run_post_observation(
            "source-01", record=caller._fixture_service.record, observation_input=observation_input,
            record_plan_reference=caller.context["protocol"]["execution_bindings"]["corrected_render_plan"],
            complete=lambda *_: G._provider(G._content(observer_contract)),
            verifier_reference={"path": "evals/foreign.py", "sha256_lf": "0" * 64},
        )
    assert callbacks == {"input": 0}
    result = caller.run_post_observation(
        "source-01", record=caller._fixture_service.record, observation_input=observation_input,
        record_plan_reference=caller.context["protocol"]["execution_bindings"]["corrected_render_plan"],
        complete=lambda *_: G._provider(G._content(observer_contract)),
        verifier_reference=caller.context["protocol"]["execution_bindings"]["record_verifier"],
    )
    assert result["status"] == "complete" and result["used_for_decision"] is False
    assert result["observation"]["used_for_decision"] is False
    assert callbacks == {"input": 1}
    assert caller.api_requests() == 2
    replay = caller.run_post_observation(
        "source-01",
        record=caller._fixture_service.record,
        record_plan_reference=caller.context["protocol"]["execution_bindings"]["corrected_render_plan"],
        observation_input=observation_input,
        complete=lambda *_: (_ for _ in ()).throw(AssertionError("cached API")),
        verifier_reference=caller.context["protocol"]["execution_bindings"]["record_verifier"],
    )
    assert replay == result and caller.api_requests() == 2 and callbacks == {"input": 1}

    class ExpiredDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            value = datetime(2028, 1, 1, tzinfo=timezone.utc)  # noqa: UP017 - Python 3.10
            return value if tz is None else value.astimezone(tz)

    monkeypatch.setattr(A, "datetime", ExpiredDateTime)
    assert caller.inspect()["observations"]["source-01"] == result
    assert caller.summarize()["observation_count"] == 1
    with pytest.raises(ValueError, match="만료"):
        caller.run_initial(
            "source-02", lambda *_: (_ for _ in ()).throw(AssertionError()),
            live_snapshot=snapshot, live_contract=contract, source_binding=binding,
        )
    monkeypatch.undo()

    cached_path = caller.runtime / "observations/source-01.json"
    tampered = json.loads(cached_path.read_text(encoding="utf-8"))
    tampered["observation"]["used_for_decision"] = True
    write(cached_path, tampered)
    with pytest.raises(ValueError, match="replay"):
        caller.summarize()


def test_arbitrary_correction_cache_cannot_unlock_record_or_vl(tmp_path: Path) -> None:
    caller, prepared, contract, binding, snapshot = setup_caller(tmp_path)
    selected(caller, prepared, contract, binding, snapshot)
    write(caller.runtime / "corrections/source-01.json", {
        "status": "complete", "checkpoint_sha256": "e" * 64, "training": {},
        "fixed_eval": {"none": "healthy", "corrected": "unhealthy", "result_sha256": "f" * 64},
    })
    with pytest.raises(ValueError, match="cached correction identity/schema"):
        caller.run_post_observation(
            "source-01", record=caller._fixture_service.record,
            record_plan_reference=caller.context["protocol"]["execution_bindings"]["corrected_render_plan"],
            observation_input=lambda *_: (_ for _ in ()).throw(AssertionError("blocked input")),
            complete=lambda *_: (_ for _ in ()).throw(AssertionError("blocked API")),
            verifier_reference=caller.context["protocol"]["execution_bindings"]["record_verifier"],
        )
    assert caller.api_requests() == 1


def test_completed_observation_rejects_rehashed_gpu_cleanup_tamper(tmp_path: Path) -> None:
    caller, prepared, contract, binding, snapshot = setup_caller(tmp_path)
    selected(caller, prepared, contract, binding, snapshot)
    recovered_correction(caller)
    observer_contract = canonical_observer_contract(caller, tmp_path)
    correction = caller._cached_correction("source-01")
    assert correction is not None
    record_job = caller._record_job("source-01", correction)
    write(tmp_path / "record/job.json", record_job)
    result_path = write(tmp_path / "record/result.json", {
        "completed": True, "checkpoint_sha256": correction["checkpoint_sha256"],
    })
    execution_path = write(tmp_path / "record/execution.json", {
        "completed": True, "exit_code": 0, "timed_out": False, "cleanup_verified": True,
        "gpu_before": [], "gpu_after": [], "source_before": {"anchor": "fixture"},
        "source_after": {"anchor": "fixture"}, "wall_s": 2.0,
        "finished_at": "2026-10-25T01:00:00+00:00", "result_sha256": A.file_sha(result_path),
    })
    caller._fixture_service.record_value = {
        "execution": AF.ref(tmp_path, execution_path), "result": AF.ref(tmp_path, result_path),
    }

    def materialized(_value: dict) -> tuple[Path, dict]:
        return observer_contract, G.G.binding_template(
            observer_contract, run_id="caller-fixture-post-source-01", source_anchor=caller.plan["source_anchor"],
            protocol_sha256=caller.context["protocol_sha256"],
        )

    caller.run_post_observation(
        "source-01", record=caller._fixture_service.record,
        record_plan_reference=caller.context["protocol"]["execution_bindings"]["corrected_render_plan"],
        observation_input=materialized, complete=lambda *_: G._provider(G._content(observer_contract)),
        verifier_reference=caller.context["protocol"]["execution_bindings"]["record_verifier"],
    )
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    execution["source_after"] = {"anchor": "foreign"}
    write(execution_path, execution)
    cached_path = caller.runtime / "observations/source-01.json"
    cached = json.loads(cached_path.read_text(encoding="utf-8"))
    cached["record"]["execution"]["sha256"] = A.file_sha(execution_path)
    write(cached_path, cached)
    with pytest.raises(ValueError, match="cleanup receipt"):
        caller.run_post_observation(
            "source-01", record=caller._fixture_service.record,
            record_plan_reference=caller.context["protocol"]["execution_bindings"]["corrected_render_plan"],
            observation_input=materialized, complete=lambda *_: (_ for _ in ()).throw(AssertionError()),
            verifier_reference=caller.context["protocol"]["execution_bindings"]["record_verifier"],
        )


def test_correction_cache_reopens_training_and_fixed_report_originals(tmp_path: Path) -> None:
    caller, prepared, contract, binding, snapshot = setup_caller(tmp_path)
    selected(caller, prepared, contract, binding, snapshot)
    recovered_correction(caller)
    telemetry = Path(caller._fixture_service.artifact["telemetry"]["path"])
    original = telemetry.read_bytes()
    telemetry.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="training telemetry SHA"):
        caller.inspect()
    telemetry.write_bytes(original)
    report_ref = caller._fixture_service.evaluation["evidence"]["corrected_report"]
    report = tmp_path / report_ref["path"]
    raw = report.read_bytes()
    report.write_bytes(b"{}\n")
    with pytest.raises(ValueError, match="fixed evaluation corrected_report SHA"):
        caller.inspect()
    report.write_bytes(raw)
    assert caller.inspect()["corrections"]["source-01"]["status"] == "complete"

    result = json.loads(json.dumps(caller._cached_correction("source-01")))
    assert isinstance(result, dict)
    foreign = "f" * 40
    for section in (result["training"]["execution"], result["evaluation"]["execution"]):
        changed = json.loads(json.dumps(section["source_before"]))
        changed["source_anchor"] = foreign
        section["source_before"] = changed
        section["source_after"] = changed
    training_evidence = result["training_evidence"]
    training_execution = tmp_path / training_evidence["files"]["execution"]["path"]
    write(training_execution, result["training"]["execution"] )
    training_evidence["files"]["execution"]["sha256"] = A.file_sha(training_execution)
    artifact_path = tmp_path / training_evidence["files"]["artifact"]["path"]
    write(artifact_path, result["training"] )
    training_evidence["files"]["artifact"]["sha256"] = A.file_sha(artifact_path)
    ledger_path = tmp_path / training_evidence["ledger"]["path"]
    ledger_rows = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines()]
    completed = ledger_rows[-1]
    completed["artifact_sha256"] = A.file_sha(artifact_path)
    completed["inventory"]["required_files"]["execution"]["sha256"] = A.file_sha(training_execution)
    ledger_rows[-1] = completed
    ledger_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in ledger_rows),
                           encoding="utf-8")
    training_evidence["ledger"]["sha256"] = A.file_sha(ledger_path)
    training_evidence["completed_event_sha256"] = A.digest(completed)
    eval_evidence = result["evaluation"]["evidence"]
    eval_execution = tmp_path / eval_evidence["execution"]["path"]
    write(eval_execution, result["evaluation"]["execution"] )
    eval_evidence["execution"]["sha256"] = A.file_sha(eval_execution)
    with pytest.raises(ValueError, match="source identity"):
        A.validate_correction_result(caller.context, "source-01", result)


def test_correction_rechecks_window_before_evaluation_and_can_resume(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    caller, prepared, contract, binding, snapshot = setup_caller(tmp_path)
    selection = selected(caller, prepared, contract, binding, snapshot)
    proposal = caller.propose_correction("source-01", reward_pair(caller, selection))
    assert proposal is not None
    caller.approve_correction(proposal["request_id"], "fixture")
    checks = {"count": 0}
    real = A.require_action_window
    def window(context, entry, action):
        checks["count"] += 1
        if checks["count"] == 3:
            raise ValueError("evaluation window unavailable")
        return real(context, entry, action)
    monkeypatch.setattr(A, "require_action_window", window)
    service = caller._fixture_service
    result = caller.run_approved_correction(
        "source-01", proposal["request_id"],
        backend_reference=caller.context["protocol"]["execution_bindings"]["correction_backend"],
        train=service.train, evaluate=service.evaluate,
    )
    assert result["status"] == "HOLD" and result["reason"] == "fixed_evaluation_window_insufficient"
    assert service.train_calls == 1 and service.evaluate_calls == 0
    assert caller.corrections.requests()[proposal["request_id"]]["state"] == "running"
    monkeypatch.setattr(A, "require_action_window", real)
    resumed = caller.run_approved_correction(
        "source-01", proposal["request_id"],
        backend_reference=caller.context["protocol"]["execution_bindings"]["correction_backend"],
        train=service.train, evaluate=service.evaluate,
    )
    assert resumed["status"] == "complete" and service.evaluate_calls == 1


def test_first_probe_arm_hold_keeps_consumed_partial_state_without_redispatch(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    caller, prepared, contract, binding, snapshot = setup_caller(tmp_path)
    selected(caller, prepared, contract, binding, snapshot)
    stage = caller.runtime / "probes/source-01"
    commands = {
        arm: {"argv": ["fixture", arm],
              "descriptor": str(stage / arm.lower() / "descriptor.json")}
        for arm in ("NONE", "F1")
    }
    pending = caller.run_probe_pair(
        "source-01", commands, caller._expected_probe_source(), approver=None,
    )
    assert pending["reason"] == "approval_required"
    windows = {"count": 0}
    dispatches = {"count": 0}
    def window(*_args, **_kwargs):
        windows["count"] += 1
        if windows["count"] == 2:
            raise ValueError("second arm window unavailable")
    def runner(argv, **_kwargs):
        dispatches["count"] += 1
        descriptor = Path(commands[str(argv[1])]["descriptor"])
        descriptor.parent.mkdir(parents=True, exist_ok=True)
        write(descriptor, {"schema": "production-shaped-fixture", "sources": {}})
        return __import__("subprocess").CompletedProcess(argv, 0, "", "")
    def actual(_root, _descriptor, *, side, **_kwargs):
        entry = next(row for row in caller.context["protocol"]["groups"]
                     if row["group"] == "source-01")
        arm = "NONE" if side == "reference" else "F1"
        return {"seed": entry["seed"], "stage": entry["execution"][arm]["stage"],
                "run_name": entry["execution"][arm]["run_name"],
                "measurement": {"reward": 1.0}, "gpu_s": 17.0}
    monkeypatch.setattr(A, "require_action_window", window)
    monkeypatch.setattr(A.EXECUTION, "_brev_actual_probe", actual)
    result = caller.run_probe_pair(
        "source-01", commands, caller._expected_probe_source(), approver="fixture", runner=runner,
    )
    assert result["status"] == "HOLD" and result["gpu_s"] == 17.0
    assert dispatches["count"] == 1
    assert [row["event"] for row in caller.pairs.events()] == [
        "proposed", "approved", "consumed",
    ]
    assert not (caller.runtime / "probe_pairs/source-01.json").exists()
    inspected = caller.inspect()["probes"]["source-01"]
    assert inspected["gpu_s"] == 17.0 and inspected["actual"]["run_name"].startswith("none-run")
    expected_cost = 17.0 * caller.context["protocol"]["resources"]["compute_quote_usd_per_h"] / 3600
    assert caller.summarize()["verified_compute_usd"] == pytest.approx(expected_cost)
    assert all(command.get("group") != "source-01" for command in caller.remaining_commands())
    replay = caller.run_probe_pair(
        "source-01", commands, caller._expected_probe_source(), approver="fixture", runner=runner,
    )
    assert replay == result and dispatches["count"] == 1



def test_pair_authorization_sha_is_same_at_execution_and_later_two_group_replay(
        tmp_path: Path) -> None:
    ledger = C.PairLedger(tmp_path / "two-group.ledger.jsonl")
    references = []
    results = []
    for ordinal in (1, 2):
        request = {"schema": A.PAIR_SCHEMA, "group": f"source-0{ordinal}"}
        reference = ledger.authorize(request, f"approver-{ordinal}")
        result = {"status": "complete", "ordinal": ordinal}
        ledger.receipt(reference["request_id"], result)
        references.append(reference); results.append(result)
    for reference, result in zip(references, results, strict=True):
        replay = ledger.completed_reference(reference["request_id"], result)
        assert replay["sha256"] == reference["sha256"]
        assert replay["prefix_sha256"] == reference["prefix_sha256"]
