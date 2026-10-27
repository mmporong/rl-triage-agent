"""완료 학습의 evaluation-only 재개 경계 CPU 회귀."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("evals", "bench", "src")]

import test_grounded_recovery as GRF
import vision_transferred_training_resume_v2 as R


def write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return path


def old_unknown(repo: Path, seed: int = 2220) -> tuple[Path, str, Path]:
    ledger = R.CorrectionLedger(repo / f"runtime/corrections-{seed}.ledger.jsonl")
    proposed = ledger.propose("a" * 64, "reward", "b" * 64, "restore_body_frame_reward", "restart",
        {"case": "case_01", "arm": "corrected"}, 3600,
        evidence={"source_job_id": f"vision-s{seed}-f1", "source_checkpoint_sha256": "c" * 64})
    ledger.approve(proposed["request_id"], "unit")
    ledger.consume(proposed["request_id"], "a" * 64)
    ledger.receipt(proposed["request_id"], "source-01", "unknown", None, 1.0, 1,
                   reason="portable_checkpoint_reference_failed")
    failure = write(repo / f"runtime/failures/source-{1 if seed == 2220 else 2:02d}.json",
                    {"status": "HOLD", "reason": "portable"})
    return ledger.path, proposed["request_id"], failure


class Backend:
    def __init__(self, repo: Path, artifact: dict, event: dict):
        self.artifact = artifact
        self.event = event
        self.ledger_path = repo / f"runtime/training-{artifact['seed']}.ledger.jsonl"
        rows = [{"event": name, "job_id": event["job_id"], "plan_sha256": "d" * 64}
                for name in ("reserved", "started")]
        rows.append(event)
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self.ledger_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        self.completed_calls = 0

    def _completed(self, job_id: str) -> dict:
        assert job_id == self.event["job_id"]
        self.completed_calls += 1
        return self.artifact

    def _events(self) -> list[dict]:
        return [json.loads(line) for line in self.ledger_path.read_text().splitlines()]


class Service:
    def __init__(self, backend: Backend, fixed: dict):
        self.backend = backend
        self.fixed = fixed
        self.preflight_calls = 0
        self.train_calls = 0
        self.evaluate_calls = 0

    def preflight(self) -> bool:
        self.preflight_calls += 1
        return True

    def train(self, *_: object, **__: object) -> dict:
        self.train_calls += 1
        raise AssertionError("evaluation-only resume가 train을 호출함")

    def training_evidence(self, *_: object, **__: object) -> dict:
        raise AssertionError("실패한 기존 portable-reference 경로를 호출함")

    def evaluate(self, artifact: dict) -> dict:
        assert artifact == self.backend.artifact
        self.evaluate_calls += 1
        return self.fixed


def resume_case(tmp_path: Path) -> tuple[Path, Path, dict, Service, Path, Path, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    external = tmp_path / "external/model_299.pt"
    external.parent.mkdir()
    external.write_bytes(b"completed-corrected-checkpoint")
    checkpoint_sha = R.E.ADAPTER.file_sha(external)
    stage = repo / "runtime/training/vision-s2220-corrected"
    telemetry = write(stage / "telemetry.json", {"ok": True})
    manifest = write(stage / "manifest.json", {"ok": True})
    job = write(stage / "job.json", {"mode": "train", "seed": 2220})
    worker = write(stage / "worker.json", {"completed": True})
    readback = write(stage / "readback.json", {"ok": True})
    records = write(stage / "records.json", {"ok": True})
    execution_body = {"wall_s": 10.0, "completed": True, "exit_code": 0, "cleanup_verified": True}
    execution = write(stage / "execution.json", execution_body)
    artifact = {"arm": "CORRECTED", "seed": 2220, "source_checkpoint_sha256": "c" * 64,
        "checkpoint": {"path": str(external), "sha256": checkpoint_sha},
        "telemetry": {"path": str(telemetry), "sha256": R.E.ADAPTER.file_sha(telemetry)},
        "manifest": {"ok": True}, "readback": {"ok": True}, "execution": execution_body}
    artifact_path = write(stage / "adapter_artifact.json", artifact)
    inventory = {"manifest": {"path": str(manifest), "sha256": R.E.ADAPTER.file_sha(manifest)},
        "required_files": {name: {"path": str(path), "sha256": R.E.ADAPTER.file_sha(path)}
            for name, path in (("readback", readback), ("records", records), ("execution", execution))}}
    event = {"event": "completed", "job_id": "vision-s2220-corrected", "plan_sha256": "d" * 64,
             "artifact_path": str(artifact_path), "artifact_sha256": R.E.ADAPTER.file_sha(artifact_path),
             "inventory": inventory}
    backend = Backend(repo, artifact, event)
    fixed = {"reports": {"none": {}, "corrected": {}}, "execution": {"wall_s": 5.0},
             "evidence": {"job": {}, "worker": {}, "execution": {}, "none_report": {}, "corrected_report": {}}}
    service = Service(backend, fixed)
    ledger, request_id, failure = old_unknown(repo)
    execution_protocol = write(repo / "execution-protocol.json", {"protocol_id": "unit-e2e"})
    external2 = tmp_path / "external/model_299_s2221.pt"; external2.write_bytes(b"completed-s2221")
    checkpoint2 = R.E.ADAPTER.file_sha(external2)
    stage2 = repo / "runtime/training/vision-s2221-corrected"
    job2 = write(stage2 / "job.json", {"mode": "train", "seed": 2221})
    worker2 = write(stage2 / "worker.json", {"completed": True})
    execution2 = write(stage2 / "execution.json", execution_body)
    telemetry2 = write(stage2 / "telemetry.json", {"ok": True})
    manifest2 = write(stage2 / "manifest.json", {"ok": True})
    readback2 = write(stage2 / "readback.json", {"ok": True})
    records2 = write(stage2 / "records.json", {"ok": True})
    artifact2 = {"arm": "CORRECTED", "seed": 2221, "source_checkpoint_sha256": "c" * 64,
        "checkpoint": {"path": str(external2), "sha256": checkpoint2},
        "telemetry": {"path": str(telemetry2), "sha256": R.E.ADAPTER.file_sha(telemetry2)},
        "manifest": {"ok": True}, "readback": {"ok": True}, "execution": execution_body}
    artifact_path2 = write(stage2 / "adapter_artifact.json", artifact2)
    inventory2 = {"manifest": {"path": str(manifest2), "sha256": R.E.ADAPTER.file_sha(manifest2)},
        "required_files": {name: {"path": str(path), "sha256": R.E.ADAPTER.file_sha(path)}
                           for name, path in (("readback", readback2), ("records", records2),
                                              ("execution", execution2))}}
    event2 = {"event": "completed", "job_id": "vision-s2221-corrected", "plan_sha256": "e" * 64,
        "artifact_path": str(artifact_path2), "artifact_sha256": R.E.ADAPTER.file_sha(artifact_path2),
        "inventory": inventory2}
    backend2 = Backend(repo, artifact2, event2)
    old2, request2, failure2 = old_unknown(repo, 2221)
    fixed2 = repo / "runtime/fixed-source02"
    none2 = write(fixed2 / "reports/NONE.json", {"name": "NONE"})
    corrected2 = write(fixed2 / "reports/CORRECTED.json", {"name": "CORRECTED"})
    fixed_job2 = write(fixed2 / "job.json", {"mode": "evaluate"})
    fixed_execution2 = write(fixed2 / "execution.json", execution_body)
    fixed_worker2 = write(fixed2 / "worker.json", {"completed": True, "reports": {
        "NONE": {"sha256": R.E.ADAPTER.file_sha(none2)},
        "CORRECTED": {"sha256": R.E.ADAPTER.file_sha(corrected2)}}})
    cached_group = {"group": "source-02", "seed": 2221, "mode": "cached_fixed_evaluation",
        "corrected_job_id": "vision-s2221-corrected",
        "completed_training": {"artifact": R._ref(repo, artifact_path2), "ledger": R._ref(repo, backend2.ledger_path),
            "job": R._ref(repo, job2), "worker": R._ref(repo, worker2), "execution": R._ref(repo, execution2),
            "checkpoint_sha256": checkpoint2},
        "old_failure": {"ledger": R._ref(repo, old2), "request_id": request2,
            "unknown_receipt_sha256": R.E.ADAPTER.digest(R.CorrectionLedger(old2).requests()[request2]["receipt"]),
            "failure": R._ref(repo, failure2), "source_checkpoint_sha256": "c" * 64},
        "fixed_evaluation": {"job": R._ref(repo, fixed_job2), "worker": R._ref(repo, fixed_worker2),
            "execution": R._ref(repo, fixed_execution2), "none_report": R._ref(repo, none2),
            "corrected_report": R._ref(repo, corrected2)},
        "checkpoint_copy_relative": "runtime/resume/checkpoint/model_299_s2221.pt",
        "output_namespace_relative": "runtime/resume/output-source02", "observed_development_result": True}
    current = datetime.now(timezone.utc)
    protocol = {"schema": R.SCHEMA, "protocol_id": "unit",
        "registered_at_utc": (current - timedelta(minutes=1)).isoformat(),
        "deadline_utc": (current + timedelta(hours=1)).isoformat(), "source_commit": "a" * 40,
        "source_sha256_lf": {}, "execution_protocol": R._ref(repo, execution_protocol),
        "group": R.GROUP, "seed": R.SEED,
        "corrected_job_id": "vision-s2220-corrected",
        "completed_training": {"artifact": R._ref(repo, artifact_path), "ledger": R._ref(repo, backend.ledger_path),
                               "job": R._ref(repo, job), "worker": R._ref(repo, worker),
                               "execution": R._ref(repo, execution), "checkpoint_sha256": checkpoint_sha},
        "old_failure": {"ledger": R._ref(repo, ledger), "request_id": request_id,
            "unknown_receipt_sha256": R.E.ADAPTER.digest(R.CorrectionLedger(ledger).requests()[request_id]["receipt"]),
            "failure": R._ref(repo, failure), "source_checkpoint_sha256": "c" * 64},
        "checkpoint_copy_relative": "runtime/resume/checkpoint/model_299.pt",
        "output_namespace_relative": "runtime/resume/output", "limits": {"training_runs": 0, "model_calls": 0,
        "new_fixed_evaluations": {"source-01": 1, "source-02": 0}},
        "evaluation_approval": {"approved": True, "approver": "unit",
        "purpose": "completed_training_fixed_evaluation_only"}, "cached_group": cached_group}
    protocol_path = write(repo / "protocol.json", protocol)
    return repo, protocol_path, protocol, service, external, ledger, failure


def test_protocol_binds_completed_job_worker_execution_and_rejects_mutation(tmp_path: Path) -> None:
    repo, protocol_path, protocol, _service, _external, _ledger, _failure = resume_case(tmp_path)
    for relative in (R.BRIDGE_RELATIVE, R.HANDOFF_RELATIVE):
        target = repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())
    protocol["source_sha256_lf"] = {
        R.BRIDGE_RELATIVE: R.E.ADAPTER.file_sha(repo / R.BRIDGE_RELATIVE, lf=True),
        R.HANDOFF_RELATIVE: R.E.ADAPTER.file_sha(repo / R.HANDOFF_RELATIVE, lf=True),
    }
    write(protocol_path, protocol)
    class Evidence:
        def __init__(self, _repo: Path):
            pass
        def file(self, _commit: str, relative: str) -> tuple[str, bytes]:
            return "0" * 40, (repo / relative).read_bytes()
    checked = R.validate_protocol(protocol_path, repo=repo, evidence_factory=Evidence)
    assert set(checked["completed_training"]) == {"artifact", "ledger", "job", "worker", "execution",
                                                   "checkpoint_sha256"}
    worker = repo / checked["completed_training"]["worker"]["path"]
    write(worker, {"completed": False})
    protocol["completed_training"]["worker"]["sha256"] = R.E.ADAPTER.file_sha(worker)
    write(protocol_path, protocol)
    with pytest.raises(ValueError, match="job/worker/execution"):
        R.validate_protocol(protocol_path, repo=repo, evidence_factory=Evidence)


def test_resume_evaluates_completed_training_once_without_train_or_old_ledger_mutation(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo, protocol_path, protocol, service, external, ledger, failure = resume_case(tmp_path)
    old_ledger, old_failure = ledger.read_bytes(), failure.read_bytes()
    monkeypatch.setattr(R, "validate_protocol", lambda *_args, **_kwargs: protocol)
    monkeypatch.setattr(R.E.ADAPTER, "verify_correction_service", lambda *_: None)
    monkeypatch.setattr(R.E.ADAPTER, "require_action_window", lambda *_: None)
    monkeypatch.setattr(R.E.ADAPTER, "_execution_gpu_s",
                        lambda value, label, context, **kwargs: 10.0 if "training" in label else 5.0)
    monkeypatch.setattr(R, "fixed_labels_backend_names",
                        lambda *_: ({"none": "healthy", "corrected": "healthy"},
                                    {"none": {"checked": True}, "corrected": {"checked": True}}))
    monkeypatch.setattr(R.E.ADAPTER, "_action_cost", lambda *_: 0.01)
    monkeypatch.setattr(R, "validate_correction_result_backend_names", lambda _c, _g, value: value)
    context = {"root": str(repo), "protocol_reference": protocol["execution_protocol"],
        "protocol": {"groups": [{"group": R.GROUP,
        "checkpoints": {"NONE": "n" * 64, "F1": "c" * 64}}]}}
    result = R.resume_evaluation(protocol_path, context, service, external_checkpoint=external,
        result_output=repo / "runtime/resume/output/correction.result.json",
        receipt_output=repo / "runtime/resume/output/evaluation.receipt.json")
    receipt = R._read(repo / "runtime/resume/output/evaluation.receipt.json")
    assert result["fixed_eval"] == {"none": "healthy", "corrected": "healthy"}
    assert service.preflight_calls == service.evaluate_calls == service.backend.completed_calls == 1
    assert service.train_calls == 0 and receipt["training_runs"] == receipt["model_calls"] == 0
    assert receipt["new_fixed_evaluations"] == 1 and receipt["verdict"] == "recovered"
    copied = repo / protocol["checkpoint_copy_relative"]
    assert copied.read_bytes() == external.read_bytes()
    assert result["training_evidence"]["files"]["checkpoint"] == R._ref(repo, copied)
    assert result["training"]["checkpoint"]["path"] == str(external)
    assert ledger.read_bytes() == old_ledger and failure.read_bytes() == old_failure
    with pytest.raises(ValueError, match="새 namespace"):
        R.resume_evaluation(protocol_path, context, service, external_checkpoint=external,
            result_output=repo / "runtime/resume/output/correction.result.json",
            receipt_output=repo / "runtime/resume/output/evaluation.receipt.json")


def test_source02_rebuilds_from_completed_fixed_eval_with_zero_gpu_calls(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo, protocol_path, protocol, _service1, _external1, _ledger1, _failure1 = resume_case(tmp_path)
    bound = protocol["cached_group"]
    artifact = R._read(repo / bound["completed_training"]["artifact"]["path"])
    events = [json.loads(line) for line in
              (repo / bound["completed_training"]["ledger"]["path"]).read_text().splitlines()]
    backend = Backend(repo, artifact, events[-1])
    backend.ledger_path = repo / bound["completed_training"]["ledger"]["path"]
    service = Service(backend, {})
    external = Path(artifact["checkpoint"]["path"])
    old_ledger = repo / bound["old_failure"]["ledger"]["path"]
    old_failure = repo / bound["old_failure"]["failure"]["path"]
    old_bytes = old_ledger.read_bytes(), old_failure.read_bytes()
    monkeypatch.setattr(R, "validate_protocol", lambda *_args, **_kwargs: protocol)
    monkeypatch.setattr(R.E.ADAPTER, "verify_correction_service", lambda *_: None)
    monkeypatch.setattr(R.E.ADAPTER, "_execution_gpu_s",
                        lambda value, label, context, **kwargs: 10.0 if "training" in label else 5.0)
    monkeypatch.setattr(R, "fixed_labels_backend_names",
                        lambda *_: ({"none": "healthy", "corrected": "healthy"},
                                    {"none": {"name": "NONE"}, "corrected": {"name": "CORRECTED"}}))
    monkeypatch.setattr(R.E.ADAPTER, "_action_cost", lambda *_: 0.01)
    monkeypatch.setattr(R, "validate_correction_result_backend_names", lambda _c, _g, value: value)
    context = {"root": str(repo), "protocol_reference": protocol["execution_protocol"],
        "protocol": {"groups": [{"group": "source-02",
            "checkpoints": {"NONE": "n" * 64, "F1": "c" * 64}}]}}
    result = R.evaluate_cached_group(protocol_path, context, service, external_checkpoint=external)
    receipt = R._read(repo / bound["output_namespace_relative"] / "evaluation.receipt.json")
    assert result["fixed_eval"] == {"none": "healthy", "corrected": "healthy"}
    assert service.evaluate_calls == service.train_calls == 0
    assert service.preflight_calls == service.backend.completed_calls == 1
    assert receipt["new_fixed_evaluations"] == 0
    assert receipt["evaluation_origin"] == "registered_completed_cache"
    assert (old_ledger.read_bytes(), old_failure.read_bytes()) == old_bytes


def test_checkpoint_copy_rejects_wrong_sha_and_existing_target(tmp_path: Path) -> None:
    repo = tmp_path / "repo"; repo.mkdir()
    source = tmp_path / "model.pt"; source.write_bytes(b"checkpoint")
    with pytest.raises(ValueError, match="SHA"):
        R._copy_checkpoint(repo, source, "copy/model.pt", "0" * 64)
    sha = R.E.ADAPTER.file_sha(source)
    target = R._copy_checkpoint(repo, source, "copy/model.pt", sha)
    assert R.E.ADAPTER.file_sha(target) == sha
    with pytest.raises(ValueError, match="새 경로"):
        R._copy_checkpoint(repo, source, "copy/model.pt", sha)


def test_backend_uppercase_names_are_strictly_mapped_without_relaxing_report_validation() -> None:
    none_sha, corrected_sha = "1" * 64, "3" * 64
    context = {"protocol": {"execution_bindings": {
        "fixed_eval_protocol": {"sha256_lf": GRF.PROTO_SHA}}}}
    reports = {"none": GRF._report("NONE", checkpoint=none_sha),
               "corrected": GRF._report("CORRECTED", checkpoint=corrected_sha)}
    labels, checked = R.fixed_labels_backend_names(context, reports, none_sha, corrected_sha)
    assert labels == {"none": "healthy", "corrected": "healthy"}
    assert checked["none"]["name"] == "NONE" and checked["corrected"]["name"] == "CORRECTED"
    reports["none"]["name"] = "none"
    with pytest.raises(ValueError, match="backend arm"):
        R.fixed_labels_backend_names(context, reports, none_sha, corrected_sha)


def test_old_unknown_contract_rejects_recovered_or_changed_ledger(tmp_path: Path) -> None:
    repo = tmp_path / "repo"; repo.mkdir()
    ledger, request_id, _ = old_unknown(repo)
    sha = R.E.ADAPTER.file_sha(ledger)
    request, receipt = R._old_unknown(ledger, request_id, sha)
    assert request["state"] == "done" and receipt["verdict"] == "unknown"
    ledger.write_text(ledger.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA"):
        R._old_unknown(ledger, request_id, sha)


def test_handoff_cli_injects_resume_runtime_factory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                    capsys: pytest.CaptureFixture[str]) -> None:
    captured: dict[str, object] = {}
    def prepare(*args: object, **kwargs: object) -> dict:
        captured.update(kwargs)
        output = Path(args[-1])
        write(output, {"ticket": True})
        return {"expires_at_utc": "2026-10-28T00:00:00+00:00"}
    monkeypatch.setattr(R.HANDOFF, "prepare", prepare)
    monkeypatch.setattr(R, "verify_native_resume_closure", lambda *_args, **_kwargs: {})
    names = ("repo", "plan", "scope", "execution-protocol", "lease", "bound-protocol",
             "recording-registration-ref", "output")
    paths = {name: tmp_path / f"{name}.json" for name in names}
    path_args = [part for name, path in paths.items() for part in (f"--{name}", str(path))]
    result = R.main(["handoff-prepare", *path_args,
        "--execution-protocol-sha256", "a" * 64, "--lease-sha256", "b" * 64,
        "--recording-registration-anchor", "c" * 40, "--group", "source-01"])
    assert result == 0 and captured["runtime_factory"] is R.ResumeRuntime
    assert "ticket_ready" in capsys.readouterr().out


def test_native_handoff_rejects_protocol_without_resume_bridge(tmp_path: Path) -> None:
    repo = tmp_path / "repo"; repo.mkdir()
    protocol = write(repo / "native.json", {"source_commit": "a" * 40, "source_sha256_lf": {}})
    with pytest.raises(ValueError, match="resume bridge"):
        R.verify_native_resume_closure(repo, protocol)


def test_resume_runtime_uses_fixed_protocol_path_without_expanding_execution_bindings(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = tmp_path / "repo"; repo.mkdir()
    execution = write(repo / "execution.json", {"protocol_id": "existing-strict-schema"})
    reference = R._ref(repo, execution)
    resume = write(repo / R.RESUME_PROTOCOL_RELATIVE, {"placeholder": True})
    runtime = object.__new__(R.ResumeRuntime)
    runtime.root = repo
    runtime.context = {"protocol_reference": reference, "protocol": {"execution_bindings": {"unchanged": True}}}
    monkeypatch.setattr(R, "validate_protocol",
                        lambda path, **_: {"execution_protocol": reference} if Path(path) == resume else {})
    assert runtime._resume_protocol() == resume.resolve()
    assert runtime.context["protocol"]["execution_bindings"] == {"unchanged": True}


def test_resume_runtime_exports_both_groups_from_new_receipts(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo, protocol_path, protocol, _service, _external, ledger, _failure = resume_case(tmp_path)
    runtime = object.__new__(R.ResumeRuntime)
    runtime.root = repo
    runtime.path = repo / "runtime/e2e"
    runtime.path.mkdir(parents=True)
    runtime.context = {"root": str(repo)}
    runtime.corrections = R.CorrectionLedger(runtime.path / "corrections.ledger.jsonl")
    result1 = repo / protocol["output_namespace_relative"] / "correction.result.json"
    result1.parent.mkdir(parents=True)
    correction1 = {"fixed_eval": {"none": "healthy", "corrected": "healthy"},
                   "job_id": protocol["corrected_job_id"]}
    write(result1, correction1)
    result2 = repo / protocol["cached_group"]["output_namespace_relative"] / "correction.result.json"
    correction2 = {"fixed_eval": {"none": "healthy", "corrected": "healthy"},
                   "job_id": protocol["cached_group"]["corrected_job_id"]}
    write(result2, correction2)
    ledger_before = ledger.read_bytes()
    monkeypatch.setattr(runtime, "refresh", lambda: None)
    monkeypatch.setattr(runtime, "_resume_protocol", lambda: protocol_path)
    monkeypatch.setattr(R, "validate_resume_result",
                        lambda _context, _path, group: ((result1, correction1, {"verdict": "recovered"})
                                                       if group == "source-01"
                                                       else (result2, correction2, {"verdict": "recovered"})))
    monkeypatch.setattr(R, "validate_protocol", lambda *_args, **_kwargs: protocol)
    bindings = runtime.recording_bindings()
    approval = R._read(bindings["corrected-s2220"]["approval_receipt"])
    assert bindings["corrected-s2220"]["correction_result"] == result1
    assert bindings["corrected-s2221"]["correction_result"] == result2
    assert approval["request_id"].startswith("evaluation-resume-")
    assert approval["correction_result_sha256"] == R.E.ADAPTER.file_sha(result1)
    assert ledger.read_bytes() == ledger_before

