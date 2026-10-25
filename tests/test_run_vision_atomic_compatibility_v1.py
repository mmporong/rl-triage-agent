from __future__ import annotations

import contextlib
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / "bench", ROOT / "evals"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import run_vision_atomic_compatibility_v1 as C  # pyright: ignore[reportMissingImports]

NOW = datetime(2026, 10, 25, 10, 0, tzinfo=timezone.utc)  # noqa: UP017 - Python 3.10
REAL_GIT = C.SOURCE._git


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class Clock:
    def __init__(self, value: datetime):
        self.value = value

    def __call__(self) -> datetime:
        return self.value

    def advance(self, seconds: int) -> None:
        self.value += timedelta(seconds=seconds)


class FakeBackend:
    def __init__(self, repo: Path, runtime: Path, now):
        self.repo, self.runtime, self.now = repo, runtime, now
        self.calls: list[str] = []
        self.admissions = 0

    @contextlib.contextmanager
    def action_admission(self):
        self.admissions += 1
        yield

    def _run(self, job: dict, kind: str) -> dict:
        self.calls.append(kind)
        stage = self.runtime / "record" / job["job_id"]
        finished = self.now().astimezone(timezone.utc) + timedelta(seconds=10)  # noqa: UP017
        write(stage / "execution.json", {"completed": True})
        write(stage / "output/result.json", {"completed": True})
        write(stage / "attestation/receipt.json", {"attested_at_utc": finished.isoformat()})
        if isinstance(self.now, Clock):
            self.now.advance(11)
        return {"status": f"{kind}_complete", "job_id": job["job_id"]}

    def smoke(self, job: dict) -> dict:
        return self._run(job, "smoke") if not (self.runtime / "record" / job["job_id"] / "execution.json").exists() \
            else {"status": "smoke_complete", "job_id": job["job_id"]}

    def record(self, job: dict) -> dict:
        return self._run(job, "record") if not (self.runtime / "record" / job["job_id"] / "execution.json").exists() \
            else {"status": "record_complete", "job_id": job["job_id"]}

    def capture_references(self, job: dict) -> dict:
        path = self.runtime / "record" / job["job_id"] / "attestation/receipt.json"
        if not path.is_file():
            raise ValueError("capture incomplete")
        return {"attestation": C._reference(self.repo, path)}


class FakeBatch:
    _record_job = C.SOURCE.SourceBatch._record_job

    def __init__(self, repo: Path, source: dict, resource: dict, resource_path: Path, now, backend_factory):
        self.repo = repo
        self.protocol = source
        self.anchor = "a" * 40
        self.resource = resource
        self.resource_path = resource_path
        self.resource_sha = C._sha(resource_path)
        self.runtime = repo / ".triage-runtime/vision-brev/compat-source"
        self.now = now
        self.backend_factory = backend_factory
        self.backend = None

    def verify(self):
        return {"status": "verified"}

    def _verify_resource(self, value):
        if value != self.resource:
            raise ValueError("resource changed")
        return value

    def _backend(self):
        if self.backend is None:
            self.backend = self.backend_factory(
                self.repo / "IsaacLab", self.repo / "python",
                {**self.protocol, "protocol_path": "fixture", "protocol_sha256_lf": "f" * 64},
                self.anchor, self.runtime, self.protocol["training_plan"], resource_proof=self.resource,
                now=self.now,
            )
        return self.backend


@pytest.fixture
def fx(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    source_path = repo / "bench/protocols/source.json"
    source_path.parent.mkdir(parents=True)
    write(source_path, {"fixture": "source"})
    quote = 1.8
    source = {
        "protocol_id": "compat-source",
        "resources": {
            "compute_quote_usd_per_h": quote,
            "deadline_utc": (NOW + timedelta(hours=2)).isoformat(),
            "whole_batch_budget_usd": 13.389896,
            "cost_scope": "owned_instance_lifecycle",
        },
        "recording": {"source_files_lf": {"bench/worker.py": "1" * 64}},
        "training_plan": [{"job_id": f"job-{i}"} for i in range(6)],
    }
    raw_path = repo / "archive/raw.json"
    receipt_path = repo / "archive/receipt.json"
    stop_path = repo / "archive/stop-receipt.json"
    checkpoint_sha = "6" * 64
    write(raw_path, {
        "stage": "train_none20", "job_id": "preflight-train-s2208",
        "request": {"seed": 2208},
        "artifact": {
            "checkpoint": {"sha256": checkpoint_sha},
            "execution": {"source_before": {"protocol": {"sha256_lf": "e" * 64}}},
        },
    })
    write(receipt_path, {"stage": "train_none20", "job_id": "preflight-train-s2208", "raw_sha256": C._sha(raw_path)})
    write(stop_path, {"schema": "v04_owned_vm_stop_receipt_v1", "status": "STOPPED",
                      "cost_estimate": {"estimated_compute_cost_usd": "6.610104"}})
    closure = {}
    for relative in {
        "bench/run_vision_atomic_compatibility_v1.py",
        "tests/test_run_vision_atomic_compatibility_v1.py",
        *C.BACKEND.SOURCE_FILES,
    }:
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative in {
            "bench/run_vision_atomic_compatibility_v1.py",
            "tests/test_run_vision_atomic_compatibility_v1.py",
        }:
            path.write_bytes((ROOT / relative).read_bytes())
        else:
            path.write_text(relative + "\n", encoding="utf-8")
        closure[relative] = C._sha(path, lf=True)
    source["source_sha256_lf"] = {
        relative: closure[relative] for relative in C.BACKEND.SOURCE_FILES
    }
    protocol = {
        "schema": C.SCHEMA, "protocol_id": "compat-source", "registered_before_measurement": True,
        "source_anchor": "a" * 40,
        "source_protocol": {"path": source_path.relative_to(repo).as_posix(), "sha256_lf": C._sha(source_path, lf=True)},
        "historical_checkpoint": {
            "seed": 2208, "raw": C._reference(repo, raw_path), "receipt": C._reference(repo, receipt_path),
            "checkpoint_sha256": checkpoint_sha, "prior_protocol_sha256_lf": "e" * 64,
        },
        "stages": [
            {"stage": "atomic_smoke", "job_id": "atomic-smoke-s2208", "checkpoint_logical_id": "compat-s2208",
             "mode": C.RECORD.RENDER_SYNC_SMOKE_MODE, "max_wall_s": 900, "max_compute_usd": quote * 0.25},
            {"stage": "atomic_record", "job_id": "atomic-record-s2208", "checkpoint_logical_id": "compat-s2208",
             "mode": C.RECORD.RECORD_MODE, "max_wall_s": 900, "max_compute_usd": quote * 0.25},
        ],
        "resources": {"prior_lifecycle_compute_usd": 6.610104, "fresh_lifecycle_budget_usd": 13.389896,
                      "total_program_budget_usd": 20.0, "prior_stop_receipt": C._reference(repo, stop_path),
                      "compute_quote_usd_per_h": quote, "deadline_utc": source["resources"]["deadline_utc"],
                      "fresh_admission_max_age_s": 30},
        "expected": {"new_training_actions": 0, "api_calls": 0, "smoke_captures": 4,
                     "record_frames": 625, "stage_order": list(C.STAGE_ORDER)},
        "controller_sources_lf": closure,
    }
    protocol_path = repo / "bench/protocols/compat.json"
    write(protocol_path, protocol)
    resource = {
        "issued_at_utc": (NOW - timedelta(seconds=5)).isoformat(),
        "expires_at_utc": (NOW + timedelta(hours=2)).isoformat(),
        "compute_started_at_utc": (NOW - timedelta(seconds=20)).isoformat(),
        "resource_id": "resource-1", "gpu_uuid": "GPU-fixture", "lease_id": "lease-1",
        "protocol_sha256": protocol["source_protocol"]["sha256_lf"],
        "raw_gpu_receipt_sha256": "2" * 64, "budget_admission_sha256": "3" * 64,
    }
    resource_path = repo / "resource.json"
    write(resource_path, resource)
    clock = Clock(NOW)
    holder = SimpleNamespace(batch=None, forwarded_protocols=[])

    monkeypatch.setattr(C.SOURCE, "validate_protocol", lambda _: source)
    def fake_git(target, *args, check=True):
        if args[0] == "status":
            return SimpleNamespace(returncode=0, stdout="")
        if args[0] == "ls-files":
            return SimpleNamespace(returncode=0, stdout=args[-1] + "\n")
        relative = args[1].split(":", 1)[1]
        return SimpleNamespace(returncode=0, stdout=(target / relative).read_text(encoding="utf-8"))

    monkeypatch.setattr(C.SOURCE, "_git", fake_git)

    def batch_factory(*args, **kwargs):
        selected_path = Path(args[4])
        selected_resource = C._read(selected_path)
        holder.batch = FakeBatch(
            repo, source, selected_resource, selected_path, kwargs["now"], kwargs["backend_factory"]
        )
        return holder.batch

    def backend_factory(lab, python, forwarded, anchor, runtime, training_plan, **kwargs):
        holder.forwarded_protocols.append(forwarded)
        return FakeBackend(repo, runtime, kwargs.get("now", lambda: NOW))

    monkeypatch.setattr(C.SOURCE, "SourceBatch", batch_factory)
    validated_jobs = []
    monkeypatch.setattr(C.RECORD, "validate_job", lambda path, now=None: validated_jobs.append(C._read(path)) or {})
    controller = C.AtomicCompatibilityController(
        repo, protocol_path, C._sha(protocol_path, lf=True), resource_path, C._sha(resource_path),
        repo / "IsaacLab", repo / "python", now=clock, backend_factory=backend_factory,
    )
    checkpoint = repo / "remote/model_19.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    protocol["historical_checkpoint"]["checkpoint_sha256"] = C._sha(checkpoint)
    raw = C._read(raw_path)
    raw["artifact"]["checkpoint"]["sha256"] = C._sha(checkpoint)
    write(raw_path, raw)
    protocol["historical_checkpoint"]["raw"] = C._reference(repo, raw_path)
    receipt = C._read(receipt_path)
    receipt["raw_sha256"] = C._sha(raw_path)
    write(receipt_path, receipt)
    protocol["historical_checkpoint"]["receipt"] = C._reference(repo, receipt_path)
    write(protocol_path, protocol)
    controller = C.AtomicCompatibilityController(
        repo, protocol_path, C._sha(protocol_path, lf=True), resource_path, C._sha(resource_path),
        repo / "IsaacLab", repo / "python", now=clock, backend_factory=backend_factory,
    )
    return SimpleNamespace(repo=repo, protocol=protocol, protocol_path=protocol_path, resource=resource,
                           resource_path=resource_path, checkpoint=checkpoint, controller=controller,
                           holder=holder, validated_jobs=validated_jobs, clock=clock)


def armed(fx):
    fx.controller.register_checkpoint(fx.checkpoint)
    fx.controller.approve("operator")


def test_register_approve_then_one_action_smoke_and_record_without_training_or_api(fx):
    armed(fx)
    first = fx.controller.advance_one()
    second = fx.controller.advance_one()
    assert first["stage"] == "atomic_smoke" and second["stage"] == "atomic_record"
    assert second["new_training_actions"] == second["api_calls"] == 0
    assert fx.holder.batch.backend.calls == ["smoke", "record"]
    validated_modes = [job["mode"] for job in fx.validated_jobs]
    assert validated_modes[0] == C.RECORD.RENDER_SYNC_SMOKE_MODE
    assert validated_modes[-1] == C.RECORD.RECORD_MODE
    assert set(validated_modes) == {C.RECORD.RENDER_SYNC_SMOKE_MODE, C.RECORD.RECORD_MODE}
    assert fx.controller.verify()["status"] == "complete"
    assert len(fx.holder.forwarded_protocols) == 1
    forwarded = fx.holder.forwarded_protocols[0]
    assert C.BACKEND.SOURCE_FILES <= set(forwarded["source_sha256_lf"])
    assert all(
        forwarded["source_sha256_lf"][relative]
        == fx.protocol["controller_sources_lf"][relative]
        for relative in C.BACKEND.SOURCE_FILES
    )


def test_approval_is_required_before_any_job_or_intent(fx):
    fx.controller.register_checkpoint(fx.checkpoint)
    with pytest.raises((FileNotFoundError, ValueError)):
        fx.controller.advance_one()
    assert not (fx.controller.root / "stages").exists()


def test_remote_checkpoint_tamper_is_rejected_before_registration(fx):
    fx.checkpoint.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checkpoint SHA"):
        fx.controller.register_checkpoint(fx.checkpoint)
    assert not fx.controller.registration_path.exists()


def test_historical_raw_receipt_tamper_is_rejected(fx):
    raw = fx.repo / fx.protocol["historical_checkpoint"]["raw"]["path"]
    raw.write_text(raw.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="historical raw 원문 SHA"):
        C.validate_protocol(fx.protocol, fx.repo)


def test_historical_receipt_tamper_is_rejected(fx):
    receipt = fx.repo / fx.protocol["historical_checkpoint"]["receipt"]["path"]
    receipt.write_text(receipt.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="historical receipt 원문 SHA"):
        C.validate_protocol(fx.protocol, fx.repo)


def test_source_closure_missing_is_rejected(fx):
    protocol = dict(fx.protocol)
    protocol["controller_sources_lf"] = {"bench/run_vision_atomic_compatibility_v1.py": C._sha(ROOT / "bench/run_vision_atomic_compatibility_v1.py", lf=True)}
    with pytest.raises(ValueError, match="source 결합"):
        C.validate_protocol(protocol, fx.repo)


def test_referenced_source_protocol_must_include_matching_atomic_closure(fx):
    source_map = fx.controller.source["source_sha256_lf"]
    relative = next(iter(C.BACKEND.SOURCE_FILES))
    original = source_map.pop(relative)
    with pytest.raises(ValueError, match="referenced source protocol"):
        C.validate_protocol(fx.protocol, fx.repo)
    source_map[relative] = "0" * 64
    with pytest.raises(ValueError, match="closure SHA"):
        C.validate_protocol(fx.protocol, fx.repo)
    source_map[relative] = original


def test_non_625_record_contract_is_rejected(fx):
    protocol = json.loads(json.dumps(fx.protocol))
    protocol["expected"]["record_frames"] = 624
    with pytest.raises(ValueError, match="기대 실행량"):
        C.validate_protocol(protocol, fx.repo)


def test_stale_resource_admission_blocks_before_intent(fx):
    armed(fx)
    stale = {**fx.resource, "issued_at_utc": (NOW - timedelta(seconds=31)).isoformat()}
    stale_path = fx.repo / "resource-stale.json"
    write(stale_path, stale)
    controller = C.AtomicCompatibilityController(
        fx.repo, fx.protocol_path, C._sha(fx.protocol_path, lf=True), stale_path, C._sha(stale_path),
        fx.repo / "IsaacLab", fx.repo / "python", now=lambda: NOW,
        backend_factory=fx.controller.backend_factory,
    )
    with pytest.raises(ValueError, match="30초"):
        controller.advance_one()
    assert not (controller.root / "stages").exists()


def test_record_cannot_start_if_approval_disappears_after_smoke(fx):
    armed(fx)
    fx.controller.advance_one()
    fx.controller.approval_path.unlink()
    with pytest.raises((FileNotFoundError, ValueError)):
        fx.controller.advance_one()
    assert not fx.controller._stage(1).exists()


def test_smoke_failure_is_terminal_and_record_is_never_created(fx, monkeypatch: pytest.MonkeyPatch):
    armed(fx)
    monkeypatch.setattr(fx.holder.batch.backend, "smoke", lambda job: (_ for _ in ()).throw(RuntimeError("boom")))
    failed = fx.controller.advance_one()
    held = fx.controller.advance_one()
    assert failed["reason"] == "compatibility_stage_failed"
    assert held["reason"] == "compatibility_stage_failed"
    assert not fx.controller._stage(1).exists()


def test_smoke_raw_sha_tamper_blocks_replay_and_record(fx):
    armed(fx)
    fx.controller.advance_one()
    raw = fx.controller._stage(0) / "raw.json"
    raw.write_text(raw.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="receipt SHA"):
        fx.controller.advance_one()
    assert fx.holder.batch.backend.calls == ["smoke"]


def test_completed_replay_reverifies_cache_without_new_capture(fx):
    armed(fx)
    fx.controller.advance_one()
    fx.controller.advance_one()
    before = list(fx.holder.batch.backend.calls)
    result = fx.controller.advance_one()
    assert result == {"status": "complete", "completed_stages": 2, "remaining_stages": 0,
                      "new_training_actions": 0, "api_calls": 0, "cached": True}
    assert fx.holder.batch.backend.calls == before


def test_smoke_over_45_seconds_then_fresh_record_proof_preserves_both_stage_bindings(fx):
    armed(fx)
    smoke = fx.controller.advance_one()
    assert smoke["stage"] == "atomic_smoke"
    old_sha = C._sha(fx.resource_path)

    later = NOW + timedelta(seconds=55)
    resource2 = {
        **fx.resource,
        "lease_id": "lease-2",
        "issued_at_utc": (later - timedelta(seconds=5)).isoformat(),
        "expires_at_utc": (later + timedelta(hours=2)).isoformat(),
        "raw_gpu_receipt_sha256": "7" * 64,
        "budget_admission_sha256": "8" * 64,
    }
    resource2_path = fx.repo / "resource-2.json"
    write(resource2_path, resource2)
    controller = C.AtomicCompatibilityController(
        fx.repo, fx.protocol_path, C._sha(fx.protocol_path, lf=True),
        resource2_path, C._sha(resource2_path), fx.repo / "IsaacLab", fx.repo / "python",
        now=lambda: later, backend_factory=fx.controller.backend_factory,
    )
    record = controller.advance_one()
    assert record["stage"] == "atomic_record" and record["status"] == "complete"
    smoke_intent = C._read(controller._stage(0) / "intent.json")
    record_intent = C._read(controller._stage(1) / "intent.json")
    assert smoke_intent["resource_proof"] == {"path": str(fx.resource_path), "sha256": old_sha}
    assert record_intent["resource_proof"] == {
        "path": str(resource2_path), "sha256": C._sha(resource2_path),
    }
    assert controller.verify()["status"] == "complete"


def test_rollover_cannot_reset_lifecycle_clock_or_budget(fx):
    armed(fx)
    fx.controller.advance_one()
    later = NOW + timedelta(seconds=55)
    reset = {
        **fx.resource,
        "lease_id": "lease-reset",
        "issued_at_utc": (later - timedelta(seconds=5)).isoformat(),
        "expires_at_utc": (later + timedelta(hours=2)).isoformat(),
        "compute_started_at_utc": (later - timedelta(seconds=10)).isoformat(),
        "raw_gpu_receipt_sha256": "7" * 64,
        "budget_admission_sha256": "8" * 64,
    }
    reset_path = fx.repo / "resource-reset.json"
    write(reset_path, reset)
    controller = C.AtomicCompatibilityController(
        fx.repo, fx.protocol_path, C._sha(fx.protocol_path, lf=True),
        reset_path, C._sha(reset_path), fx.repo / "IsaacLab", fx.repo / "python",
        now=lambda: later, backend_factory=fx.controller.backend_factory,
    )
    with pytest.raises(ValueError, match="등록 lifecycle"):
        controller.advance_one()
    assert not controller._stage(1).exists()


def test_completed_stages_reverify_after_resource_and_deadline_expiry_without_gpu_action(fx):
    armed(fx)
    fx.controller.advance_one()
    fx.controller.advance_one()
    calls_before = list(fx.holder.batch.backend.calls)

    expired = NOW + timedelta(hours=3)
    controller = C.AtomicCompatibilityController(
        fx.repo, fx.protocol_path, C._sha(fx.protocol_path, lf=True),
        fx.resource_path, C._sha(fx.resource_path), fx.repo / "IsaacLab", fx.repo / "python",
        now=lambda: expired, backend_factory=fx.controller.backend_factory,
    )
    result = controller.verify()

    assert result["status"] == "complete"
    assert result["stages"] == ["complete", "complete"]
    assert fx.holder.batch.backend.calls == []
    assert calls_before == ["smoke", "record"]


@pytest.mark.parametrize("name", ["protocol.json", "admission.json", "resource_parent.json"])
def test_completed_record_replay_rejects_missing_input_without_recreating_it(fx, name):
    armed(fx)
    fx.controller.advance_one()
    fx.controller.advance_one()
    path = fx.holder.batch.runtime / "record-inputs/atomic_record" / name
    path.unlink()

    with pytest.raises(ValueError, match="record input"):
        fx.controller.verify()

    assert not path.exists()


@pytest.mark.parametrize("name", [None, "protocol.json", "admission.json", "resource_parent.json"])
def test_new_record_rejects_orphan_inputs_without_completing_them_or_calling_backend(fx, name):
    armed(fx)
    fx.controller.advance_one()
    inputs = fx.holder.batch.runtime / "record-inputs/atomic_record"
    inputs.mkdir(parents=True)
    if name is not None:
        write(inputs / name, {"orphan": True})
    calls_before = list(fx.holder.batch.backend.calls)

    with pytest.raises(ValueError, match="orphan"):
        fx.controller.advance_one()

    assert not fx.controller._stage(1).exists()
    assert fx.holder.batch.backend.calls == calls_before
    assert sorted(path.name for path in inputs.iterdir()) == ([] if name is None else [name])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("cost_before_usd", 99.0),
        ("cost_after_usd", 99.0),
        ("elapsed_wall_s", 11.0),
        ("finished_at_utc", (NOW + timedelta(seconds=11)).isoformat()),
        ("recovered_from_cache", True),
        ("unexpected", "field"),
    ],
)
def test_receipt_cost_recovery_flag_and_extra_field_tamper_are_rejected(fx, field, value):
    armed(fx)
    fx.controller.advance_one()
    path = fx.controller._stage(0) / "receipt.json"
    receipt = C._read(path)
    receipt[field] = value
    write(path, receipt)
    with pytest.raises(ValueError, match="receipt|recovery"):
        fx.controller.verify()


def test_pending_completed_capture_writes_cache_recovery_proof_once(fx):
    armed(fx)
    fx.controller.advance_one()
    receipt_path = fx.controller._stage(0) / "receipt.json"
    receipt_path.unlink()
    recovered = fx.controller.advance_one()
    receipt = C._read(receipt_path)
    assert recovered["recovered_from_cache"] is True
    assert receipt["recovered_from_cache"] is True
    assert receipt["cache_recovery_sha256"] == C._sha(fx.controller._stage(0) / "cache_recovery.json")
    assert fx.controller.verify()["stages"][0] == "complete"


def test_pending_settle_reuses_matching_write_once_cache_recovery_after_interruption(fx):
    armed(fx)
    fx.controller.advance_one()
    folder = fx.controller._stage(0)
    (folder / "receipt.json").unlink()
    job = C._read(folder / "job.precheck.json")
    recovery = {
        "schema": "vision_atomic_compatibility_cache_recovery_v1",
        "stage": "atomic_smoke",
        "intent_sha256": C._sha(folder / "intent.json"),
        "job_sha256": C._digest(job),
        "execution_sha256": C._sha(fx.holder.batch.runtime / "record/atomic-smoke-s2208/execution.json"),
        "result_sha256": C._sha(fx.holder.batch.runtime / "record/atomic-smoke-s2208/output/result.json"),
    }
    write(folder / "cache_recovery.json", recovery)

    result = fx.controller.advance_one()

    assert result["status"] == "advanced"
    assert result["recovered_from_cache"] is True
    assert C._read(folder / "cache_recovery.json") == recovery


def test_pending_settle_rejects_mismatched_existing_cache_recovery(fx):
    armed(fx)
    fx.controller.advance_one()
    folder = fx.controller._stage(0)
    (folder / "receipt.json").unlink()
    write(folder / "cache_recovery.json", {"schema": "wrong"})

    result = fx.controller.advance_one()

    assert result["status"] == "HOLD"
    assert result["reason"] == "compatibility_stage_invalid"


def test_real_git_code_then_source_then_compatibility_commits_have_no_cycle(
    fx, monkeypatch: pytest.MonkeyPatch
):
    protocol_bytes = fx.protocol_path.read_bytes()
    source_path = fx.repo / fx.protocol["source_protocol"]["path"]
    source_bytes = source_path.read_bytes()
    fx.protocol_path.unlink()
    source_path.unlink()
    subprocess.run(["git", "init", "-q", str(fx.repo)], check=True)
    subprocess.run(["git", "-C", str(fx.repo), "config", "user.email", "fixture@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(fx.repo), "config", "user.name", "fixture"], check=True)
    files = [
        str(path.relative_to(fx.repo))
        for path in fx.repo.rglob("*")
        if path.is_file() and ".git" not in path.parts
    ]
    subprocess.run(["git", "-C", str(fx.repo), "add", "--", *files], check=True)
    subprocess.run(["git", "-C", str(fx.repo), "commit", "-q", "-m", "code"], check=True)
    code_anchor = subprocess.run(
        ["git", "-C", str(fx.repo), "rev-parse", "HEAD"], check=True, text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()

    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(source_bytes)
    subprocess.run(
        ["git", "-C", str(fx.repo), "add", "--", str(source_path.relative_to(fx.repo))],
        check=True,
    )
    subprocess.run(["git", "-C", str(fx.repo), "commit", "-q", "-m", "source protocol"], check=True)
    source_anchor = subprocess.run(
        ["git", "-C", str(fx.repo), "rev-parse", "HEAD"], check=True, text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    assert source_anchor != code_anchor

    protocol = json.loads(protocol_bytes)
    protocol["source_anchor"] = source_anchor
    write(fx.protocol_path, protocol)
    subprocess.run(
        ["git", "-C", str(fx.repo), "add", "--", str(fx.protocol_path.relative_to(fx.repo))],
        check=True,
    )
    subprocess.run(["git", "-C", str(fx.repo), "commit", "-q", "-m", "protocol"], check=True)
    monkeypatch.setattr(C.SOURCE, "_git", REAL_GIT)

    controller = C.AtomicCompatibilityController(
        fx.repo, fx.protocol_path, C._sha(fx.protocol_path, lf=True),
        fx.resource_path, C._sha(fx.resource_path), fx.repo / "IsaacLab", fx.repo / "python",
        now=lambda: NOW, backend_factory=fx.controller.backend_factory,
    )
    assert controller.protocol["source_anchor"] == source_anchor
    source_blob = subprocess.run(
        ["git", "-C", str(fx.repo), "show", f"{source_anchor}:{source_path.relative_to(fx.repo).as_posix()}"],
        check=True, stdout=subprocess.PIPE,
    ).stdout.replace(b"\r\n", b"\n")
    assert C.hashlib.sha256(source_blob).hexdigest() == fx.protocol["source_protocol"]["sha256_lf"]
    assert subprocess.run(
        ["git", "-C", str(fx.repo), "rev-list", "--count", "HEAD"], check=True, text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip() == "3"
