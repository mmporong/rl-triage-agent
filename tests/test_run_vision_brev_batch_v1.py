from __future__ import annotations

import hashlib
import importlib
import json
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / "bench", ROOT / "evals", ROOT / "src"):
    sys.path.insert(0, str(folder))
D = importlib.import_module("run_vision_brev_batch_v1")
R = importlib.import_module("record_vision_evidence_v1")
B = importlib.import_module("brev_recovery_backend_v4")
UTC_TZ = timezone.utc  # noqa: UP017 - Python 3.10


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _sha(path: Path, lf: bool = False) -> str:
    data = path.read_bytes().replace(b"\r\n", b"\n") if lf else path.read_bytes()
    return hashlib.sha256(data).hexdigest()


def _plan() -> list[dict]:
    rows = []
    for seed in (4101, 4102):
        for arm in ("NONE", "F1", "CORRECTED"):
            job_id = f"s{seed}-{arm.lower()}"
            rows.append({"job_id": job_id, "arm": arm, "seed": seed, "run_name": job_id,
                         "stage_relative": f"training/{job_id}", "iterations": 300, "num_envs": 4096,
                         "source_job_id": f"s{seed}-f1" if arm == "CORRECTED" else None})
    return rows


def _actions() -> list[dict]:
    actions = []
    for seed in (4101, 4102):
        for arm in ("none", "f1"):
            actions.append({"action_id": f"train-{seed}-{arm}", "kind": "train",
                            "job_id": f"s{seed}-{arm}"})
    for seed in (4101, 4102):
        actions.append({"action_id": f"eval-{seed}", "kind": "evaluate", "seed": seed,
                        "none_job_id": f"s{seed}-none", "f1_job_id": f"s{seed}-f1",
                        "run_name": f"eval-{seed}"})
    for seed in (4101, 4102):
        for arm in ("none", "f1"):
            actions.append({"action_id": f"record-{seed}-{arm}", "kind": "record",
                            "job_id": f"record-{seed}-{arm}", "source_job_id": f"s{seed}-{arm}",
                            "checkpoint_logical_id": f"policy-{seed}-{arm}"})
    return actions


def _fixture(tmp_path: Path, *, actions: list[dict] | None = None,
             expired: bool = False, compute_age_hours: float = 1.0) -> tuple[Path, Path, Path, str, str]:
    repo = tmp_path / "repo"
    (repo / "bench/protocols").mkdir(parents=True)
    (repo / "evals").mkdir()
    (repo / "src").mkdir()
    lab = tmp_path / "lab"
    (lab / "logs").mkdir(parents=True)
    source = repo / "bench/source.py"
    source.write_text("value=1\n", encoding="utf-8")
    bands = repo / "bench/bands.json"
    _write(bands, {"bands": {"fixture": True}})
    source_hash = _sha(source, lf=True)
    protocol = {
        "schema": D.SCHEMA, "protocol_id": "vision-source-test",
        "registered_before_measurement": True,
        "record_profile_sha256": R.digest(R.fixed_profile()),
        "training_plan": _plan(), "source_action_order": actions or _actions(),
        "evaluation": {"bands_path": "bench/bands.json", "bands_sha256": _sha(bands)},
        "recording": {"source_files_lf": {"bench/source.py": source_hash}},
        "resources": {"deadline_utc": "2030-01-01T00:00:00+00:00",
                      "whole_batch_budget_usd": 50.0, "compute_quote_usd_per_h": 2.0,
                      "owned_instance_id": "brev-instance-1", "max_owned_instances": 1,
                      "action_limits": {
                          "train": {"max_wall_s": 1800, "max_compute_usd": 1.0},
                          "evaluate": {"max_wall_s": 900, "max_compute_usd": 0.5},
                          "record": {"max_wall_s": 900, "max_compute_usd": 0.5}},
                      "reservation_method": "owned_instance_elapsed_plus_next_max_v1",
                      "cost_scope": "owned_instance_elapsed_wall_includes_idle_setup_transfer_excludes_storage_v1"},
        "source_sha256_lf": {"bench/source.py": source_hash},
        "runtime_source_sha256_lf": {"train.py": "1" * 64},
        "python_package_source_sha256_lf": {"ppo.py": "2" * 64},
        "environment": {"python": "3.10"}, "fixed_eval_protocol": "bench/fixed.json",
    }
    protocol_path = repo / "bench/protocols/vision_source.json"
    _write(protocol_path, protocol)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "t@example.com"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "t"], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "bench/source.py", "bench/bands.json",
                    "bench/protocols/vision_source.json"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "fixture"], check=True)
    anchor = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True,
                            capture_output=True, text=True).stdout.strip()
    protocol_sha = _sha(protocol_path, lf=True)
    now = datetime(2027, 1, 1, tzinfo=UTC_TZ)
    issued = (now - timedelta(hours=1)).isoformat()
    expires = (now - timedelta(seconds=1) if expired else now + timedelta(days=1)).isoformat()
    compute_started = (now - timedelta(hours=compute_age_hours)).isoformat()
    provider_path = tmp_path / "provider.json"
    provider_row = {key: None for key in D.REMOTE.PROVIDER_ROW_KEYS}
    provider_row.update({"id": "brev-instance-1", "name": "fixture", "status": "RUNNING"})
    _write(provider_path, {"workspaces": [provider_row]})
    gpu_path, process_path = tmp_path / "gpu.csv", tmp_path / "process.csv"
    gpu_path.write_text("GPU-fixture-1\n", encoding="utf-8", newline="")
    process_path.write_text("", encoding="utf-8", newline="")
    common = {"status": "granted", "resource_id": "brev-instance-1", "gpu_uuid": "GPU-fixture-1",
              "lease_id": "lease-1", "issued_at_utc": issued, "expires_at_utc": expires,
              "compute_started_at_utc": compute_started, "protocol_sha256": protocol_sha,
              "provider_inventory_sha256": _sha(provider_path),
              "nvidia_gpu_csv_sha256": _sha(gpu_path),
              "nvidia_process_csv_sha256": _sha(process_path)}
    raw_path, budget_path = tmp_path / "raw-gpu.json", tmp_path / "budget.json"
    _write(raw_path, {"schema": D.RAW_GPU_RECEIPT_SCHEMA, **common})
    _write(budget_path, {"schema": D.BUDGET_ADMISSION_SCHEMA, **common,
                         "whole_batch_budget_usd": 50.0, "compute_quote_usd_per_h": 2.0,
                         "max_owned_instances": 1,
                         "cost_scope": "owned_instance_elapsed_wall_includes_idle_setup_transfer_excludes_storage_v1"})
    resource = {"schema": D.RESOURCE_SCHEMA, "status": "granted", "resource_kind": "remote_gpu",
                "lease_id": "lease-1", "protocol_sha256": protocol_sha,
                "issued_at_utc": issued, "expires_at_utc": expires,
                "compute_started_at_utc": compute_started,
                "resource_id": "brev-instance-1", "gpu_uuid": "GPU-fixture-1",
                "raw_gpu_receipt_path": str(raw_path.resolve()),
                "raw_gpu_receipt_sha256": _sha(raw_path),
                "budget_admission_path": str(budget_path.resolve()),
                "budget_admission_sha256": _sha(budget_path),
                "provider_inventory_path": str(provider_path.resolve()),
                "provider_inventory_sha256": _sha(provider_path),
                "nvidia_gpu_csv_path": str(gpu_path.resolve()),
                "nvidia_gpu_csv_sha256": _sha(gpu_path),
                "nvidia_process_csv_path": str(process_path.resolve()),
                "nvidia_process_csv_sha256": _sha(process_path)}
    resource_path = tmp_path / "resource.json"
    _write(resource_path, resource)
    return repo, lab, protocol_path, anchor, protocol_sha


class FakeBackend:
    def __init__(self, runtime: Path, *, fail_job: str | None = None, model_ready: bool = False,
                 train_wall_s: float = 10.0, permit_busy_once: bool = False):
        self.runtime, self.fail_job, self.model_ready = runtime, fail_job, model_ready
        self.train_wall_s = train_wall_s
        self.train_runs: list[str] = []
        self.eval_runs: list[str] = []
        self.record_runs: list[str] = []
        self.artifacts: dict[str, dict] = {}
        self.records: dict[str, dict] = {}
        self.permit_busy_once = permit_busy_once

    @contextmanager
    def action_admission(self):
        if self.permit_busy_once:
            self.permit_busy_once = False
            raise B.BackendError("GPUExecutionBusy")
        yield

    def train(self, job_id: str) -> dict:
        if job_id in self.artifacts:
            return self.artifacts[job_id]
        self.train_runs.append(job_id)
        if job_id == self.fail_job:
            raise B.BackendError("ForeignGPUBusy")
        arm = "F1" if job_id.endswith("f1") else "NONE"
        seed = int(job_id.split("-")[0][1:])
        folder = self.runtime / "fake-training" / job_id
        folder.mkdir(parents=True)
        checkpoint, telemetry, manifest = folder / "model.pt", folder / "telemetry.json", folder / "manifest.json"
        checkpoint.write_bytes(job_id.encode())
        _write(telemetry, {"job_id": job_id})
        _write(manifest, {"job_id": job_id})
        artifact = {"arm": arm, "seed": seed,
                    "checkpoint": {"path": str(checkpoint), "sha256": _sha(checkpoint)},
                    "telemetry": {"path": str(telemetry), "sha256": _sha(telemetry), "summary": {}},
                    "manifest_path": str(manifest), "manifest_sha256": _sha(manifest),
                    "execution": {"completed": True, "wall_s": self.train_wall_s}}
        self.artifacts[job_id] = artifact
        return artifact

    def evaluate(self, stage: Path, artifacts: dict, run_name: str) -> dict:
        self.eval_runs.append(run_name)
        return {"reports": {"NONE": {"arm": "NONE", "seed": artifacts["NONE"]["seed"]},
                            "F1": {"arm": "F1", "seed": artifacts["F1"]["seed"]}},
                "execution": {"completed": True, "wall_s": 5.0}}

    def verify_execution_gpu_observations(self, execution: dict) -> None:
        for key in ("gpu_before_observation", "gpu_after_observation"):
            reference = execution.get(key)
            if reference is not None:
                D.REMOTE.verify_observation_reference(reference, "brev-instance-1", "GPU-fixture-1")

    def record(self, job: dict) -> dict:
        job_id = job["job_id"]
        if job_id in self.records:
            return self.records[job_id]
        self.record_runs.append(job_id)
        stage = self.runtime / "record" / job_id
        stage.mkdir(parents=True, exist_ok=True)
        _write(stage / "execution.json", {"completed": True})
        value = {"status": "ready" if self.model_ready else "capture_complete_gpu_smoke_pending",
                 "model_use_ready": self.model_ready,
                 "result": {"zero_update_pixel_lag_confirmed": False,
                            "gpu_smoke_required_before_model_use": True},
                 "execution": {"completed": True, "wall_s": 2.0}}
        self.records[job_id] = value
        return value


def _batch(tmp_path: Path, *, fail_job: str | None = None, model_ready: bool = False,
           actions: list[dict] | None = None, expired: bool = False,
           train_wall_s: float = 10.0, now=None, compute_age_hours: float = 1.0,
           permit_busy_once: bool = False):
    repo, lab, protocol, anchor, protocol_sha = _fixture(
        tmp_path, actions=actions, expired=expired, compute_age_hours=compute_age_hours
    )
    resource = tmp_path / "resource.json"
    holder = {}

    def factory(lab_arg, python, protocol_body, anchor_arg, runtime, plan, **kwargs):
        assert lab_arg == lab and anchor_arg == anchor and len(plan) == 6
        assert kwargs["resource_proof"]["resource_id"] == "brev-instance-1"
        backend = FakeBackend(runtime, fail_job=fail_job, model_ready=model_ready,
                              train_wall_s=train_wall_s, permit_busy_once=permit_busy_once)
        holder["backend"] = backend
        return backend

    fresh_calls = []

    def fresh(seeds, isaaclab, repo):
        fresh_calls.append(tuple(seeds))
        return {"schema": "preregistered_seed_freshness_v1", "status": "fresh", "seeds": list(seeds)}

    batch = D.SourceBatch(repo, protocol, protocol_sha, anchor, resource, _sha(resource), lab,
                          Path(sys.executable), backend_factory=factory, seed_guard=fresh,
                          now=now or (lambda: datetime(2027, 1, 1, tzinfo=UTC_TZ)))
    return batch, holder, fresh_calls


def test_dispatches_fixed_order_preserves_source_gates_and_finishes_smoke_pending(tmp_path: Path, monkeypatch) -> None:
    gates = []

    def recovery(fault, normal, bands):
        value = {"outcome": "source_failure_absent" if fault["seed"] == 4102 else "source_failure_present"}
        gates.append(value)
        return value

    monkeypatch.setattr(D.RECOVERY, "recovery", recovery)
    batch, holder, fresh_calls = _batch(tmp_path)
    assert batch.verify()["status"] == "verified"
    for expected in range(1, 11):
        result = batch.advance()
        assert result["status"] == "advanced" and result["completed"] == expected
    final = batch.advance()
    assert final["status"] == "needs_render_sync_smoke"
    assert final["model_use_ready"] is False and final["source_data_ready"] is False
    backend = holder["backend"]
    assert backend.train_runs == ["s4101-none", "s4101-f1", "s4102-none", "s4102-f1"]
    assert backend.eval_runs == ["eval-4101", "eval-4102"]
    assert backend.record_runs == ["record-4101-none", "record-4101-f1",
                                   "record-4102-none", "record-4102-f1"]
    assert fresh_calls == [(4101, 4102)]
    assert [row["outcome"] for row in final["source_gates"]] == [
        "source_failure_present", "source_failure_absent"
    ]
    before = (list(backend.train_runs), list(backend.eval_runs), list(backend.record_runs))
    assert batch.advance()["status"] == "needs_render_sync_smoke"
    assert before == (backend.train_runs, backend.eval_runs, backend.record_runs)
    assert fresh_calls == [(4101, 4102)]


def test_busy_gpu_before_action_writes_no_intent_and_same_action_can_retry(tmp_path: Path) -> None:
    batch, holder, fresh_calls = _batch(tmp_path, permit_busy_once=True)
    with pytest.raises(B.BackendError, match="GPUExecutionBusy"):
        batch.advance()
    assert not (batch.runtime / "actions").exists()
    assert not (batch.runtime / "freshness.json").exists()
    assert fresh_calls == []
    assert batch.advance()["status"] == "advanced"
    assert holder["backend"].train_runs == ["s4101-none"]
    assert fresh_calls == [(4101, 4102)]


def test_cache_revalidates_raw_artifacts_and_receipt_hash(tmp_path: Path) -> None:
    batch, _, _ = _batch(tmp_path)
    assert batch.advance()["status"] == "advanced"
    checkpoint = batch.runtime / "fake-training/s4101-none/model.pt"
    checkpoint.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="변조"):
        batch.advance()

    batch2, _, _ = _batch(tmp_path / "other")
    assert batch2.advance()["status"] == "advanced"
    receipt = batch2.runtime / "actions/00_train-4101-none/receipt.json"
    body = json.loads(receipt.read_text())
    body["raw_sha256"] = "0" * 64
    _write(receipt, body)
    with pytest.raises(ValueError, match="SHA 결합"):
        batch2.advance()


def test_protocol_rejects_corrected_source_action_and_resource_mismatch(tmp_path: Path) -> None:
    actions = _actions()
    actions[0]["job_id"] = "s4101-corrected"
    repo, lab, protocol, anchor, protocol_sha = _fixture(tmp_path, actions=actions)
    resource = tmp_path / "resource.json"
    with pytest.raises(ValueError, match="NONE/F1"):
        D.SourceBatch(repo, protocol, protocol_sha, anchor, resource, _sha(resource), lab, Path(sys.executable))

    with pytest.raises(ValueError, match="유효 시간"):
        _batch(tmp_path / "expired", expired=True)

    batch, _, _ = _batch(tmp_path / "mutated")
    resource_path = tmp_path / "mutated/resource.json"
    body = json.loads(resource_path.read_text())
    body["gpu_uuid"] = "GPU-changed"
    _write(resource_path, body)
    with pytest.raises(ValueError, match="원본 SHA"):
        batch.verify()


def test_model_ready_claim_before_render_sync_smoke_is_rejected(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(D.RECOVERY, "recovery", lambda *args: {"outcome": "source_failure_present"})
    batch, holder, _ = _batch(tmp_path, model_ready=True)
    for _ in range(6):
        assert batch.advance()["status"] == "advanced"
    with pytest.raises(ValueError, match="model-ready"):
        batch.advance()
    assert holder["backend"].record_runs == ["record-4101-none"]
    assert batch.advance()["status"] == "HOLD"


def test_resource_proof_rechecks_raw_gpu_and_budget_artifacts(tmp_path: Path) -> None:
    batch, _, _ = _batch(tmp_path)
    resource_path = tmp_path / "resource.json"
    resource = json.loads(resource_path.read_text(encoding="utf-8"))
    raw_path = Path(resource["raw_gpu_receipt_path"])
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    raw["gpu_uuid"] = "GPU-other"
    raw_path.unlink()
    _write(raw_path, raw)
    resource["raw_gpu_receipt_sha256"] = _sha(raw_path)
    resource_path.unlink()
    _write(resource_path, resource)
    with pytest.raises(ValueError, match="raw GPU receipt"):
        batch._verify_resource(resource)

    batch2, _, _ = _batch(tmp_path / "budget")
    resource2 = json.loads((tmp_path / "budget/resource.json").read_text(encoding="utf-8"))
    budget_path = Path(resource2["budget_admission_path"])
    budget = json.loads(budget_path.read_text(encoding="utf-8"))
    budget["whole_batch_budget_usd"] = 999.0
    budget_path.unlink()
    _write(budget_path, budget)
    resource2["budget_admission_sha256"] = _sha(budget_path)
    with pytest.raises(ValueError, match="budget admission"):
        batch2._verify_resource(resource2)


def test_action_reservation_and_measured_cost_fail_closed(tmp_path: Path) -> None:
    batch, holder, _ = _batch(tmp_path, train_wall_s=1801.0)
    result = batch.advance()
    assert result["status"] == "HOLD" and result["reason"] == "action_compute_exceeded"
    folder = batch.runtime / "actions/00_train-4101-none"
    assert (folder / "raw.json").is_file() and (folder / "invalid.json").is_file()
    assert not (folder / "receipt.json").exists()
    assert holder["backend"].train_runs == ["s4101-none"]
    assert batch.advance()["reason"] == "source_action_pending"


def test_protocol_rejects_total_reserved_compute_above_batch_cap(tmp_path: Path) -> None:
    _, _, protocol_path, _, _ = _fixture(tmp_path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol["resources"]["whole_batch_budget_usd"] = 1.0
    with pytest.raises(ValueError, match="최대 compute 예약"):
        D.validate_protocol(protocol)


def test_idle_and_setup_elapsed_cost_blocks_next_action_before_intent(tmp_path: Path) -> None:
    batch, holder, _ = _batch(tmp_path, compute_age_hours=25.0)
    result = batch.advance()
    assert result == {"status": "HOLD", "reason": "whole_batch_budget_reached", "completed": 0}
    assert holder["backend"].train_runs == []
    assert not (batch.runtime / "actions").exists()


def test_record_cache_preserves_original_admission_with_advancing_clock(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(D.RECOVERY, "recovery", lambda *args: {"outcome": "source_failure_present"})
    current = datetime(2027, 1, 1, tzinfo=UTC_TZ)

    def tick() -> datetime:
        return current

    batch, holder, _ = _batch(tmp_path, now=tick)
    for _ in range(10):
        assert batch.advance()["status"] == "advanced"
    admissions = sorted(batch.runtime.glob("record-inputs/*/admission.json"))
    before = {path: path.read_bytes() for path in admissions}
    runs = list(holder["backend"].record_runs)
    current += timedelta(hours=1)
    terminal = batch.advance()
    assert terminal["status"] == "needs_render_sync_smoke"
    assert terminal["lifecycle_cost_status"] == "open"
    assert batch.advance()["status"] == "needs_render_sync_smoke"
    assert {path: path.read_bytes() for path in admissions} == before
    assert holder["backend"].record_runs == runs
    first_stage = batch.runtime / "record/record-4101-none/execution.json"
    first_stage.unlink()
    with pytest.raises(ValueError, match="오래됐다"):
        batch.advance()


@pytest.mark.parametrize("target", ["receipt", "csv"])
def test_evaluation_cache_reaudits_gpu_observation_originals(
    tmp_path: Path, monkeypatch, target: str
) -> None:
    monkeypatch.setattr(D.RECOVERY, "recovery", lambda *args: {"outcome": "source_failure_present"})
    batch, _, _ = _batch(tmp_path)
    for _ in range(5):
        assert batch.advance()["status"] == "advanced"

    def run(command, **kwargs):
        stdout = "GPU-fixture-1\n" if command == D.REMOTE.GPU_COMMAND else ""
        return subprocess.CompletedProcess(command, 0, stdout, "")

    capture = {key: batch.resource[key] for key in D.REMOTE.CAPTURE_KEYS}
    reader = D.REMOTE.StrictRemoteGpuReader(capture, tmp_path / "evaluation-gpu", run=run)
    reader()
    reference = dict(reader.last_receipt or {})
    action = batch.runtime / "actions/04_eval-4101"
    raw_path, receipt_path = action / "raw.json", action / "receipt.json"
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    raw["evaluation"]["execution"].update({
        "gpu_before_observation": reference, "gpu_after_observation": reference,
    })
    raw_path.unlink()
    _write(raw_path, raw)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["raw_sha256"] = _sha(raw_path)
    receipt_path.unlink()
    _write(receipt_path, receipt)
    if target == "receipt":
        Path(reference["path"]).unlink()
    else:
        (tmp_path / "evaluation-gpu/000001.process.csv").write_text(
            "GPU-fixture-1, 99\n", encoding="utf-8", newline=""
        )
    with pytest.raises(ValueError, match="GPU 관측"):
        batch.advance()


def test_terminal_cache_holds_when_instance_lifecycle_crosses_budget(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(D.RECOVERY, "recovery", lambda *args: {"outcome": "source_failure_present"})
    current = datetime(2027, 1, 1, tzinfo=UTC_TZ)
    batch, holder, _ = _batch(tmp_path, now=lambda: current, compute_age_hours=24.0)
    for _ in range(10):
        assert batch.advance()["status"] == "advanced"
    assert batch.advance()["status"] == "needs_render_sync_smoke"
    runs = (list(holder["backend"].train_runs), list(holder["backend"].eval_runs),
            list(holder["backend"].record_runs))
    current += timedelta(hours=1, minutes=6)
    assert batch.advance() == {
        "status": "HOLD", "reason": "whole_batch_budget_reached", "completed": 10,
    }
    assert (holder["backend"].train_runs, holder["backend"].eval_runs,
            holder["backend"].record_runs) == runs
