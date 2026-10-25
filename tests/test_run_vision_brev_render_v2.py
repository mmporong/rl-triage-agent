from __future__ import annotations

import importlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / "bench", ROOT / "evals", ROOT / "src", ROOT / "tests"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

D = importlib.import_module("run_vision_brev_render_v2")
V1 = importlib.import_module("test_run_vision_brev_batch_v1")
UTC = timezone.utc  # noqa: UP017 - Python 3.10


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


class Clock:
    def __init__(self) -> None:
        self.value = datetime(2027, 1, 1, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += timedelta(seconds=seconds)


class FakeRenderBackend(V1.FakeBackend):
    def __init__(self, runtime: Path, clock: Clock, *, busy_once: bool = False):
        super().__init__(runtime, permit_busy_once=busy_once)
        self.clock = clock
        self.smoke_runs: list[str] = []
        self.smokes: dict[str, dict[str, Any]] = {}
        self.raise_after_smoke_once = False
        self.hold_smoke_once = False

    def smoke(self, job: dict[str, Any]) -> dict[str, Any]:
        job_id = job["job_id"]
        if self.hold_smoke_once:
            self.hold_smoke_once = False
            self.smoke_runs.append(job_id)
            return {"status": "HOLD", "reason": "fixture_incomplete"}
        if job_id not in self.smokes:
            self.smoke_runs.append(job_id)
            self.clock.advance(120)
            self.smokes[job_id] = {
                "status": "render_sync_smoke_complete",
                "model_use_ready": False,
                "result": {"job_id": job_id, "mode": job["mode"]},
                "execution": {"completed": True, "wall_s": 1.0},
            }
            stage = self.runtime / "record" / job_id
            _write(stage / "job.json", job)
            _write(stage / "output/result.json", self.smokes[job_id]["result"])
            _write(stage / "execution.json", self.smokes[job_id]["execution"])
            if self.raise_after_smoke_once:
                self.raise_after_smoke_once = False
                raise RuntimeError("simulated controller interruption")
        return self.smokes[job_id]

    def record(self, job: dict[str, Any]) -> dict[str, Any]:
        value = super().record(job)
        _write(self.runtime / "record" / job["job_id"] / "job.json", job)
        _write(self.runtime / "record" / job["job_id"] / "output/result.json", value["result"])
        _write(self.runtime / "record" / job["job_id"] / "execution.json", value["execution"])
        return value

    def capture_references(self, job: dict[str, Any]) -> dict[str, dict[str, str]]:
        stage = self.runtime / "record" / job["job_id"]
        result, execution = stage / "output/result.json", stage / "execution.json"
        job_path = stage / "job.json"
        admission = Path(job["remote_gpu_admission"]["path"])
        if not result.is_file() or not execution.is_file():
            raise ValueError("capture incomplete")
        return {
            "job": {"path": job_path.relative_to(self.runtime.parents[2]).as_posix(),
                    "sha256": D.SOURCE.sha256(job_path)},
            "admission": {"path": admission.relative_to(self.runtime.parents[2]).as_posix(),
                          "sha256": D.SOURCE.sha256(admission)},
            "result": {"path": result.relative_to(self.runtime.parents[2]).as_posix(),
                       "sha256": D.SOURCE.sha256(result)},
            "execution": {"path": execution.relative_to(self.runtime.parents[2]).as_posix(),
                          "sha256": D.SOURCE.sha256(execution)},
        }


def _rebind_resource(tmp_path: Path, protocol_sha: str, whole_budget_usd: float) -> None:
    raw_path, budget_path = tmp_path / "raw-gpu.json", tmp_path / "budget.json"
    raw, budget = D.SOURCE.read(raw_path), D.SOURCE.read(budget_path)
    raw["protocol_sha256"] = protocol_sha
    budget["protocol_sha256"] = protocol_sha
    budget["whole_batch_budget_usd"] = whole_budget_usd
    _write(raw_path, raw)
    _write(budget_path, budget)
    resource_path = tmp_path / "resource.json"
    resource = D.SOURCE.read(resource_path)
    resource.update({
        "protocol_sha256": protocol_sha,
        "raw_gpu_receipt_sha256": D.SOURCE.sha256(raw_path),
        "budget_admission_sha256": D.SOURCE.sha256(budget_path),
    })
    _write(resource_path, resource)


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, outcome: str = "not_recovered",
             busy_once: bool = False) -> tuple[Any, FakeRenderBackend, Clock, Path, Path]:
    repo, lab, source_path, _, _ = V1._fixture(tmp_path)
    source = D.SOURCE.read(source_path)
    source["resources"]["whole_batch_budget_usd"] = 10.0
    source["resources"]["deadline_utc"] = "2027-01-01T12:00:00+00:00"
    for relative in D.REQUIRED_CAPTURE_SOURCES:
        target = repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
        digest = D.SOURCE.sha256(target, lf=True)
        source["source_sha256_lf"][relative] = digest
        source["recording"]["source_files_lf"][relative] = digest
    _write(source_path, source)
    source_sha = D.SOURCE.sha256(source_path, lf=True)
    render_path = repo / "bench/protocols/vision_render.json"
    source_ref = {"path": source_path.relative_to(repo).as_posix(), "sha256_lf": source_sha}
    _write(render_path, D.build_protocol(source, source_ref))
    render_sha = D.SOURCE.sha256(render_path, lf=True)
    _rebind_resource(tmp_path, source_sha, 10.0)
    subprocess.run(["git", "-C", str(repo), "add", "--", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "render fixture"], check=True)
    anchor = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    clock = Clock()
    holder: dict[str, FakeRenderBackend] = {}

    def factory(lab_arg: Path, python: Path, protocol: dict[str, Any], anchor_arg: str,
                runtime: Path, plan: list[dict[str, Any]], **kwargs: Any) -> FakeRenderBackend:
        assert lab_arg == lab and anchor_arg == anchor and len(plan) == 6
        assert kwargs["resource_proof"]["resource_id"] == "brev-instance-1"
        backend = FakeRenderBackend(runtime, clock, busy_once=busy_once)
        holder["backend"] = backend
        return backend

    monkeypatch.setattr(D.SOURCE.RECOVERY, "recovery", lambda *_: {"outcome": outcome})
    monkeypatch.setattr(D.SOURCE.RECOVERY.V.R1.U, "walks", lambda _report: True)
    finalizer = importlib.import_module("vision_render_readiness_v2")

    def fake_material(_repo: Path, _source: Path, _sha: str, _resource: Path,
                      _runtime: Path, record_job_id: str, record_refs: dict[str, Any],
                      smoke_refs: dict[str, Any], **_kwargs: Any) -> dict[str, Any]:
        return {"schema": "fixture", "record_job_id": record_job_id,
                "record": record_refs, "smoke": smoke_refs}

    monkeypatch.setattr(finalizer, "build_material", fake_material)
    monkeypatch.setattr(finalizer, "verify_material", lambda *_args, **_kwargs: {"verified": True})
    batch = D.RenderBatch(
        repo, source_path, source_sha, render_path, render_sha, anchor,
        tmp_path / "resource.json", D.SOURCE.sha256(tmp_path / "resource.json"),
        lab, Path(sys.executable), backend_factory=factory,
        seed_guard=lambda seeds, *_args, **_kwargs: {
            "schema": "preregistered_seed_freshness_v1", "status": "fresh", "seeds": list(seeds),
        },
        now=clock,
    )
    return batch, holder.setdefault("backend", batch._backend()), clock, source_path, render_path


def _advance_source(batch: Any) -> None:
    for count in range(1, 11):
        result = batch.advance()
        assert result["status"] == "advanced"
        assert result["completed"] == count


def test_protocol_derives_four_smokes_and_reserves_within_existing_budget(tmp_path: Path) -> None:
    repo, _, source_path, _, source_sha = V1._fixture(tmp_path)
    source = D.SOURCE.read(source_path)
    reference = {"path": source_path.relative_to(repo).as_posix(), "sha256_lf": source_sha}
    protocol = D.build_protocol(source, reference)
    assert [row["record_job_id"] for row in protocol["smoke_plan"]] == [
        "record-4101-none", "record-4101-f1", "record-4102-none", "record-4102-f1",
    ]
    assert protocol["smoke_limits"] == {"max_wall_s": 900, "max_compute_usd": 0.5}
    changed = json.loads(json.dumps(protocol))
    changed["smoke_limits"]["max_wall_s"] = 901
    with pytest.raises(ValueError, match="source 파생 계약"):
        D.validate_protocol(changed, source, reference)


def test_lf_bound_json_accepts_lf_and_exact_crlf_copy_only(tmp_path: Path) -> None:
    lf = tmp_path / "bands-lf.json"
    crlf = tmp_path / "bands-crlf.json"
    changed = tmp_path / "bands-changed.json"
    body = b'{\n  "bands": {"fixture": true}\n}\n'
    lf.write_bytes(body)
    crlf.write_bytes(body.replace(b"\n", b"\r\n"))
    changed.write_bytes(body.replace(b"true", b"false").replace(b"\n", b"\r\n"))
    expected = D.SOURCE.sha256(lf)
    assert D._read_lf_bound_json(lf, expected, "bands")["bands"]["fixture"] is True
    assert D._read_lf_bound_json(crlf, expected, "bands")["bands"]["fixture"] is True
    with pytest.raises(ValueError, match="LF SHA"):
        D._read_lf_bound_json(changed, expected, "bands")


def test_archived_original_path_maps_posix_provenance_and_rejects_escape(tmp_path: Path) -> None:
    repo = tmp_path / "archive"
    parent = repo / "runtime/parent.json"
    _write(parent, {"fixture": True})
    finalizer = importlib.import_module("vision_render_readiness_v2")
    assert D._archived_original_path(
        repo, "/origin/repo", "/origin/repo/runtime/parent.json", finalizer._archived_path,
    ) == parent
    with pytest.raises(ValueError, match="원본 저장소|정규형"):
        D._archived_original_path(
            repo, "/origin/repo", "/origin/repo/../outside.json", finalizer._archived_path,
        )
    with pytest.raises(ValueError):
        D._archived_original_path(
            repo, "/origin/repo", "/origin/other/parent.json", finalizer._archived_path,
        )


def test_offline_export_partial_journal_holds_without_backend_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch, backend, _, _, _ = _fixture(tmp_path, monkeypatch)
    result = D.export_materials(
        batch.repo, batch.protocol_path, batch.protocol_sha,
        batch.render_protocol_path, batch.render_protocol_sha, batch.anchor,
        batch.resource_path, batch.resource_sha,
    )
    assert result["status"] == "HOLD" and result["reason"] == "source_or_render_journal_incomplete"
    assert backend.train_runs == [] and backend.record_runs == [] and backend.smoke_runs == []


def test_fixed_source_then_four_smokes_once_and_elapsed_includes_attested_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch, backend, _, _, _ = _fixture(tmp_path, monkeypatch)
    assert batch.verify()["smoke_actions"] == 4
    _advance_source(batch)
    result: dict[str, Any] = {}
    for completed in range(1, 5):
        result = batch.advance()
        assert result["status"] == "advanced"
        assert result["completed_smokes"] == completed
    materials_reference = result["materials"]
    assert D.SOURCE.sha256(batch.repo / materials_reference["path"]) == materials_reference["sha256"]
    terminal = batch.advance()
    assert terminal["status"] == "needs_readiness_finalize"
    assert terminal["materials"] == materials_reference
    assert backend.record_runs == [
        "record-4101-none", "record-4101-f1", "record-4102-none", "record-4102-f1",
    ]
    assert backend.smoke_runs == [
        "smoke-record-4101-none", "smoke-record-4101-f1",
        "smoke-record-4102-none", "smoke-record-4102-f1",
    ]
    assert D.SOURCE.read(batch._smoke_folder(0) / "receipt.json")["elapsed_wall_s"] == 120
    assert batch.advance()["status"] == "needs_readiness_finalize"
    assert len(backend.smoke_runs) == 4
    finalizer = importlib.import_module("vision_render_readiness_v2")

    def archived(_repo: Path, _runtime: Path, job_id: str) -> dict[str, dict[str, str]]:
        job = D.SOURCE.read(batch.runtime / "record" / job_id / "job.json")
        references = backend.capture_references(job)
        return references

    monkeypatch.setattr(finalizer, "archive_capture_references", archived)
    monkeypatch.setattr(D, "_verify_finalize_row", lambda *_args, **_kwargs: (Path("unused"), {}))
    before = (list(backend.train_runs), list(backend.record_runs), list(backend.smoke_runs))
    exported = D.export_materials(
        batch.repo, batch.protocol_path, batch.protocol_sha,
        batch.render_protocol_path, batch.render_protocol_sha, batch.anchor,
        batch.resource_path, batch.resource_sha,
    )
    assert exported["status"] == "needs_readiness_finalize"
    assert before == (backend.train_runs, backend.record_runs, backend.smoke_runs)

    record_action = batch.protocol["source_action_order"][6]
    record_folder = batch._action_folder(6, record_action)
    record_raw_path, record_receipt_path = record_folder / "raw.json", record_folder / "receipt.json"
    record_raw = D.SOURCE.read(record_raw_path)
    record_raw["record"]["execution"]["wall_s"] = 0.0
    _write(record_raw_path, record_raw)
    record_receipt = D.SOURCE.read(record_receipt_path)
    record_receipt["raw_sha256"] = D.SOURCE.sha256(record_raw_path)
    _write(record_receipt_path, record_receipt)
    with pytest.raises(ValueError, match="record raw.*capture"):
        D.export_materials(
            batch.repo, batch.protocol_path, batch.protocol_sha,
            batch.render_protocol_path, batch.render_protocol_sha, batch.anchor,
            batch.resource_path, batch.resource_sha,
        )


def test_offline_source_journal_recomputes_gate_lifecycle_and_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch, _, _, _, _ = _fixture(tmp_path, monkeypatch)
    _advance_source(batch)
    assert D._verify_source_journals(
        batch.repo, batch.runtime, batch.protocol, batch.protocol_sha,
        batch.resource, batch.resource_sha,
    )
    folder = batch._action_folder(4, batch.protocol["source_action_order"][4])
    intent_path, raw_path, receipt_path = (
        folder / "intent.json", folder / "raw.json", folder / "receipt.json"
    )
    baseline_intent = D.SOURCE.read(intent_path)
    baseline_raw = D.SOURCE.read(raw_path)
    baseline_receipt = D.SOURCE.read(receipt_path)

    raw = json.loads(json.dumps(baseline_raw))
    raw["source_gate"] = {"outcome": "recovered"}
    _write(raw_path, raw)
    receipt = json.loads(json.dumps(baseline_receipt))
    receipt["raw_sha256"] = D.SOURCE.sha256(raw_path)
    _write(receipt_path, receipt)
    with pytest.raises(ValueError, match="gate 재계산"):
        D._verify_source_journals(
            batch.repo, batch.runtime, batch.protocol, batch.protocol_sha,
            batch.resource, batch.resource_sha,
        )

    _write(raw_path, baseline_raw)
    intent = json.loads(json.dumps(baseline_intent))
    intent["cost_before_usd"] += 1.0
    _write(intent_path, intent)
    receipt = json.loads(json.dumps(baseline_receipt))
    receipt.update({
        "intent_sha256": D.SOURCE.sha256(intent_path),
        "raw_sha256": D.SOURCE.sha256(raw_path),
    })
    _write(receipt_path, receipt)
    with pytest.raises(ValueError, match="비용·deadline"):
        D._verify_source_journals(
            batch.repo, batch.runtime, batch.protocol, batch.protocol_sha,
            batch.resource, batch.resource_sha,
        )

    deadline = D.SOURCE._stamp(batch.protocol["resources"]["deadline_utc"])
    started, finished = deadline, deadline + timedelta(seconds=1)
    compute_started = D.SOURCE._stamp(batch.resource["compute_started_at_utc"])
    quote = float(batch.protocol["resources"]["compute_quote_usd_per_h"])
    intent = json.loads(json.dumps(baseline_intent))
    intent.update({
        "started_at_utc": started.isoformat(),
        "cost_before_usd": (started - compute_started).total_seconds() * quote / 3600.0,
    })
    _write(intent_path, intent)
    receipt = json.loads(json.dumps(baseline_receipt))
    receipt.update({
        "intent_sha256": D.SOURCE.sha256(intent_path),
        "raw_sha256": D.SOURCE.sha256(raw_path),
        "finished_at_utc": finished.isoformat(),
        "cost_after_usd": (finished - compute_started).total_seconds() * quote / 3600.0,
    })
    _write(receipt_path, receipt)
    with pytest.raises(ValueError, match="비용·deadline"):
        D._verify_source_journals(
            batch.repo, batch.runtime, batch.protocol, batch.protocol_sha,
            batch.resource, batch.resource_sha,
        )


def test_source_failure_absent_stops_before_record_and_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch, backend, _, _, _ = _fixture(tmp_path, monkeypatch, outcome="recovered")
    for _ in range(6):
        assert batch.advance()["status"] == "advanced"
    result = batch.advance()
    assert result["status"] == "source_failure_absent"
    assert backend.record_runs == [] and backend.smoke_runs == []
    assert not (batch.runtime / "render-actions").exists()


def test_partial_smoke_intent_is_preserved_without_backend_reexecution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch, backend, _, _, _ = _fixture(tmp_path, monkeypatch)
    _advance_source(batch)
    backend.hold_smoke_once = True
    assert batch.advance()["status"] == "HOLD"
    folder = batch._smoke_folder(0)
    before = list(backend.smoke_runs)
    result = batch.advance()
    assert result["status"] == "HOLD" and result["reason"] == "render_action_pending"
    assert backend.smoke_runs == before
    assert (folder / "intent.json").is_file() and not (folder / "raw.json").exists()


@pytest.mark.parametrize("tamper", ["missing_protocol", "intent_field"])
def test_pending_intent_input_or_identity_tamper_is_integrity_error_without_reexecution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tamper: str,
) -> None:
    batch, backend, _, _, _ = _fixture(tmp_path, monkeypatch)
    _advance_source(batch)
    backend.hold_smoke_once = True
    assert batch.advance()["status"] == "HOLD"
    before = list(backend.smoke_runs)
    if tamper == "missing_protocol":
        (batch.runtime / "render-inputs/smoke-record-4101-none/protocol.json").unlink()
    else:
        path = batch._smoke_folder(0) / "intent.json"
        value = D.SOURCE.read(path)
        value["job_id"] = "wrong"
        _write(path, value)
    with pytest.raises(ValueError, match="protocol/admission|intent 결합"):
        batch.advance()
    assert backend.smoke_runs == before


def test_orphaned_smoke_inputs_hold_before_intent_and_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch, backend, _, _, _ = _fixture(tmp_path, monkeypatch)
    _advance_source(batch)
    orphan = batch.runtime / "render-inputs/smoke-record-4101-none"
    orphan.mkdir(parents=True)
    _write(orphan / "protocol.json", {"orphan": True})
    result = batch.advance()
    assert result["status"] == "HOLD" and result["reason"] == "render_input_orphaned"
    assert backend.smoke_runs == [] and not batch._smoke_folder(0).exists()


def test_complete_cached_capture_closes_intent_without_second_smoke_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch, backend, _, _, _ = _fixture(tmp_path, monkeypatch)
    _advance_source(batch)
    backend.raise_after_smoke_once = True
    with pytest.raises(RuntimeError, match="controller interruption"):
        batch.advance()
    folder = batch._smoke_folder(0)
    assert (folder / "intent.json").is_file() and not (folder / "raw.json").exists()
    assert backend.smoke_runs == ["smoke-record-4101-none"]
    result = batch.advance()
    assert result["status"] == "advanced" and result["recovered_from_cached_capture"] is True
    assert backend.smoke_runs == ["smoke-record-4101-none"]
    assert (folder / "raw.json").is_file() and (folder / "receipt.json").is_file()


def test_exact_raw_without_receipt_is_closed_from_read_only_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch, backend, _, _, _ = _fixture(tmp_path, monkeypatch)
    _advance_source(batch)
    assert batch.advance()["status"] == "advanced"
    folder = batch._smoke_folder(0)
    (folder / "receipt.json").unlink()
    before = list(backend.smoke_runs)
    result = batch.advance()
    assert result["status"] == "advanced" and result["recovered_from_cached_capture"] is True
    assert backend.smoke_runs == before


def test_complete_cache_tamper_is_rejected_instead_of_hidden_as_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch, backend, _, _, _ = _fixture(tmp_path, monkeypatch)
    _advance_source(batch)
    assert batch.advance()["status"] == "advanced"
    folder = batch._smoke_folder(0)
    (folder / "receipt.json").unlink()
    result_path = batch.runtime / "record/smoke-record-4101-none/output/result.json"
    _write(result_path, {"tampered": True})
    before = list(backend.smoke_runs)
    with pytest.raises(ValueError, match="pending smoke raw"):
        batch.advance()
    assert backend.smoke_runs == before


def test_busy_and_expired_or_over_budget_write_no_smoke_intent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch, backend, clock, _, _ = _fixture(tmp_path, monkeypatch)
    _advance_source(batch)
    backend.permit_busy_once = True
    with pytest.raises(Exception, match="GPUExecutionBusy"):
        batch.advance()
    assert not (batch.runtime / "render-actions").exists()
    clock.value = D.SOURCE._stamp(batch.protocol["resources"]["deadline_utc"]) - timedelta(seconds=899)
    result = batch._advance_smoke()
    assert result["status"] == "HOLD" and result["reason"] == "render_authorization_expired"
    assert not (batch.runtime / "render-actions").exists()
    clock.value = D.SOURCE._stamp(batch.resource["expires_at_utc"]) - timedelta(seconds=899)
    result = batch._advance_smoke()
    assert result["status"] == "HOLD" and result["reason"] == "render_authorization_expired"
    assert not (batch.runtime / "render-actions").exists()
    clock.value = D.SOURCE._stamp(batch.resource["compute_started_at_utc"]) + timedelta(hours=26)
    result = batch._advance_smoke()
    assert result["status"] == "HOLD" and result["reason"] in {
        "render_authorization_expired", "whole_batch_budget_reached",
    }
    assert not (batch.runtime / "render-actions").exists()


def test_cached_smoke_receipt_tamper_and_bound_source_change_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch, _, _, _, _ = _fixture(tmp_path, monkeypatch)
    _advance_source(batch)
    assert batch.advance()["status"] == "advanced"
    receipt_path = batch._smoke_folder(0) / "receipt.json"
    receipt = D.SOURCE.read(receipt_path)
    receipt["raw_sha256"] = "0" * 64
    _write(receipt_path, receipt)
    with pytest.raises(ValueError, match="receipt SHA"):
        batch.advance()

    fresh, _, _, _, render_path = _fixture(tmp_path / "fresh", monkeypatch)
    render_path.write_text(render_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="render protocol LF SHA"):
        fresh.verify()

    dirty, _, _, _, _ = _fixture(tmp_path / "dirty", monkeypatch)
    capture_source = dirty.repo / "bench/run_vision_brev_render_v2.py"
    capture_source.write_text(capture_source.read_text(encoding="utf-8") + "# changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source LF SHA|clean"):
        dirty.verify()


def test_finalize_uses_saved_materials_without_constructing_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch, _, _, source_path, _ = _fixture(tmp_path, monkeypatch)
    calls: list[Path] = []

    def finalize_pair(_repo: Path, _source: Path, _sha: str,
                      material: dict[str, Any], output: Path) -> dict[str, str]:
        calls.append(output)
        return {"path": output.relative_to(batch.repo).as_posix(),
                "sha256": D.SOURCE.digest(material)}

    finalizer = importlib.import_module("vision_render_readiness_v2")
    monkeypatch.setattr(finalizer, "finalize_pair", finalize_pair)
    materials = batch.repo / "materials.json"
    archived_resource = batch.repo / "archive/resource.json"
    archived_resource.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(batch.resource_path, archived_resource)
    resource_ref = {"path": archived_resource.relative_to(batch.repo).as_posix(),
                    "sha256": D.SOURCE.sha256(archived_resource)}
    pairs = []
    for index, plan in enumerate(batch.render_protocol["smoke_plan"]):
        base = batch.repo / "archive" / str(index)
        checkpoint = {"path": f"/task/checkpoint-{index}.pt", "sha256": str(index + 1) * 64,
                      "logical_id": plan["checkpoint_logical_id"]}
        record_job = base / "record-job.json"
        smoke_job = base / "smoke-job.json"
        _write(record_job, {"job_id": plan["record_job_id"], "mode": D.RECORD.RECORD_MODE,
                            "checkpoint": checkpoint})
        _write(smoke_job, {"job_id": plan["job_id"], "mode": D.RECORD.RENDER_SYNC_SMOKE_MODE,
                           "checkpoint": checkpoint})

        def ref(path: Path) -> dict[str, str]:
            return {"path": path.relative_to(batch.repo).as_posix(), "sha256": D.SOURCE.sha256(path)}

        smoke_refs = {"job": ref(smoke_job), "unique": {"path": f"archive/{index}/artifact.bin",
                                                            "sha256": "f" * 64}}
        material = {"record": {"job": ref(record_job), "unique": index},
                    "smoke": smoke_refs, "resource_parent": resource_ref, "unique": index}
        started = datetime(2027, 1, 1, 1, index, tzinfo=UTC)
        finished = started + timedelta(seconds=120)
        compute_started = D.SOURCE._stamp(batch.resource["compute_started_at_utc"])
        cost_before = (started - compute_started).total_seconds() * 2.0 / 3600.0
        cost_after = (finished - compute_started).total_seconds() * 2.0 / 3600.0
        intent = {
            "schema": D.SMOKE_INTENT_SCHEMA, "action_id": plan["action_id"], "job_id": plan["job_id"],
            "ordinal": index, "source_protocol_sha256_lf": batch.protocol_sha,
            "render_protocol_sha256_lf": batch.render_protocol_sha,
            "resource_sha256": resource_ref["sha256"], "job_binding_sha256": "d" * 64,
            "cost_before_usd": cost_before, "started_at_utc": started.isoformat(),
            "reserved_max_wall_s": 900, "reserved_max_compute_usd": 0.5,
        }
        intent_path, raw_path, receipt_path = base / "intent.json", base / "raw.json", base / "receipt.json"
        _write(intent_path, intent)
        _write(raw_path, {
            "schema": D.SMOKE_RAW_SCHEMA, "action_id": plan["action_id"], "job_id": plan["job_id"],
            "job_binding_sha256": "d" * 64,
            "smoke": {"status": "render_sync_smoke_complete", "model_use_ready": False},
            "references": smoke_refs,
        })
        _write(receipt_path, {
            "schema": D.SMOKE_RECEIPT_SCHEMA, "action_id": plan["action_id"], "job_id": plan["job_id"],
            "source_protocol_sha256_lf": batch.protocol_sha,
            "render_protocol_sha256_lf": batch.render_protocol_sha,
            "resource_sha256": resource_ref["sha256"], "job_binding_sha256": "d" * 64,
            "intent_sha256": D.SOURCE.sha256(intent_path), "raw_sha256": D.SOURCE.sha256(raw_path),
            "started_at_utc": started.isoformat(), "finished_at_utc": finished.isoformat(),
            "elapsed_wall_s": 120.0, "cost_before_usd": cost_before,
            "cost_after_usd": cost_after,
        })
        pairs.append({
            **plan, "output_path": (f".triage-runtime/vision-brev/{batch.protocol['protocol_id']}"
                                    f"/readiness/{plan['record_job_id']}.json"),
            "material": material,
            "smoke_journal": {"intent": ref(intent_path), "raw": ref(raw_path),
                              "receipt": ref(receipt_path)},
        })
    _write(materials, {
        "schema": D.MATERIALS_SCHEMA,
        "source_protocol": {"path": source_path.relative_to(batch.repo).as_posix(),
                            "sha256_lf": batch.protocol_sha},
        "render_protocol": {"path": batch.render_protocol_path.relative_to(batch.repo).as_posix(),
                            "sha256_lf": batch.render_protocol_sha},
        "pairs": pairs,
    })
    result = D.finalize_materials(
        batch.repo, source_path, batch.protocol_sha,
        batch.render_protocol_path, batch.render_protocol_sha, materials,
    )
    assert result["status"] == "source_data_ready" and len(calls) == 4
    assert batch._backend().train_runs == [] and batch._backend().smoke_runs == []
    baseline = D.SOURCE.read(materials)
    first = baseline["pairs"][0]
    intent_path = batch.repo / first["smoke_journal"]["intent"]["path"]
    receipt_path = batch.repo / first["smoke_journal"]["receipt"]["path"]
    raw_path = batch.repo / first["smoke_journal"]["raw"]["path"]
    baseline_intent, baseline_receipt = D.SOURCE.read(intent_path), D.SOURCE.read(receipt_path)

    def reject_timing(started: datetime, finished: datetime) -> None:
        changed = json.loads(json.dumps(baseline))
        intent = json.loads(json.dumps(baseline_intent))
        receipt = json.loads(json.dumps(baseline_receipt))
        compute = D.SOURCE._stamp(batch.resource["compute_started_at_utc"])
        before = (started - compute).total_seconds() * 2.0 / 3600.0
        after = (finished - compute).total_seconds() * 2.0 / 3600.0
        intent.update({"started_at_utc": started.isoformat(), "cost_before_usd": before})
        _write(intent_path, intent)
        receipt.update({
            "started_at_utc": started.isoformat(), "finished_at_utc": finished.isoformat(),
            "elapsed_wall_s": (finished - started).total_seconds(), "cost_before_usd": before,
            "cost_after_usd": after, "intent_sha256": D.SOURCE.sha256(intent_path),
            "raw_sha256": D.SOURCE.sha256(raw_path),
        })
        _write(receipt_path, receipt)
        changed["pairs"][0]["smoke_journal"]["intent"] = {
            "path": intent_path.relative_to(batch.repo).as_posix(), "sha256": D.SOURCE.sha256(intent_path),
        }
        changed["pairs"][0]["smoke_journal"]["receipt"] = {
            "path": receipt_path.relative_to(batch.repo).as_posix(), "sha256": D.SOURCE.sha256(receipt_path),
        }
        _write(materials, changed)
        with pytest.raises(ValueError, match="lifecycle"):
            D.finalize_materials(
                batch.repo, source_path, batch.protocol_sha,
                batch.render_protocol_path, batch.render_protocol_sha, materials,
            )
        _write(intent_path, baseline_intent)
        _write(receipt_path, baseline_receipt)
        _write(materials, baseline)

    compute = D.SOURCE._stamp(batch.resource["compute_started_at_utc"])
    reject_timing(compute - timedelta(seconds=60), compute + timedelta(seconds=60))
    reject_timing(compute + timedelta(hours=6), compute + timedelta(hours=6, seconds=120))
    expiry = D.SOURCE._stamp(batch.resource["expires_at_utc"])
    reject_timing(expiry - timedelta(seconds=60), expiry + timedelta(seconds=60))

    wrong_resource = D.SOURCE.read(archived_resource)
    wrong_resource["lease_id"] = "other-lease"
    wrong_path = batch.repo / "archive/wrong-resource.json"
    _write(wrong_path, wrong_resource)
    wrong = json.loads(json.dumps(baseline))
    wrong["pairs"][0]["material"]["resource_parent"] = {
        "path": wrong_path.relative_to(batch.repo).as_posix(), "sha256": D.SOURCE.sha256(wrong_path),
    }
    _write(materials, wrong)
    with pytest.raises(ValueError, match="resource proof"):
        D.finalize_materials(
            batch.repo, source_path, batch.protocol_sha,
            batch.render_protocol_path, batch.render_protocol_sha, materials,
        )
    _write(materials, baseline)
    duplicate = D.SOURCE.read(materials)
    duplicate["pairs"][1]["output_path"] = duplicate["pairs"][0]["output_path"]
    _write(materials, duplicate)
    with pytest.raises(ValueError, match="파생값|중복"):
        D.finalize_materials(
            batch.repo, source_path, batch.protocol_sha,
            batch.render_protocol_path, batch.render_protocol_sha, materials,
        )
