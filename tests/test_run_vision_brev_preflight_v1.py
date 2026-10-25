from __future__ import annotations

import importlib
import json
import shutil
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / "bench", ROOT / "evals", ROOT / "src", ROOT / "tests"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

P = importlib.import_module("run_vision_brev_preflight_v1")
RT = importlib.import_module("test_run_vision_brev_render_v2")
V1 = RT.V1
VB = importlib.import_module("vision_brev_backend_v1")
B = importlib.import_module("brev_recovery_backend_v4")
UTC = timezone.utc  # noqa: UP017 - Python 3.10
SEED = 4199
CORE_FILES = ("training.ledger.jsonl", "actions", "render-actions", "freshness.json", "manifest.json")


class ControllerCrash(BaseException):
    """controller process가 receipt 전에 끊긴 상황."""


def _fake_refs(repo: Path, runtime: Path, job_id: str) -> dict[str, dict[str, str]]:
    """archive_capture_references 대신 fixture capture 원문의 ref만 같은 형식으로 만든다."""
    stage = runtime / "record" / job_id
    job = P.SOURCE.read(stage / "job.json")
    files = {"job": stage / "job.json", "admission": Path(job["remote_gpu_admission"]["path"]),
             "result": stage / "output/result.json", "execution": stage / "execution.json",
             "attestation": stage / "attestation/attestation.json"}
    return {name: {"path": path.resolve().relative_to(Path(repo).resolve()).as_posix(), "sha256": V1._sha(path)}
            for name, path in files.items()}


def _gpu_run(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
    stdout = "GPU-fixture-1\n" if command == P.REMOTE.GPU_COMMAND else ""
    return subprocess.CompletedProcess(command, 0, stdout, "")


class FakeRecovery:
    """SafeRecoveryBackend.train의 20회 dry-run stage 원문 구조를 CPU에서 만든다."""

    def __init__(self, runtime: Path, lab: Path, clock: Any, anchor: str, source_sha: str,
                 resource: dict[str, Any]):
        self.runtime, self.lab, self.clock = runtime, lab, clock
        self.calls: list[dict[str, Any]] = []
        self.wall_s, self.clock_s = 600.0, 600.0
        self.observed_iterations = 20
        self.fail = self.crash_before = self.crash_after = False
        capture = {key: resource[key] for key in P.REMOTE.CAPTURE_KEYS}
        self.reader = P.REMOTE.StrictRemoteGpuReader(capture, runtime / "gpu-observations", run=_gpu_run)
        self.snapshot = {"source_anchor": anchor,
                         "protocol": {"path": "bench/protocols/vision_source.json", "sha256_lf": source_sha}}
        self.anchor, self.source_sha = anchor, source_sha

    def train(self, stage: Path, arm: str, seed: int, run_name: str, corrections: list[str] | None = None,
              *, iterations: int = 300, source_checkpoint_sha256: str | None = None) -> dict[str, Any]:
        self.calls.append({"stage": Path(stage), "arm": arm, "seed": seed, "run_name": run_name,
                           "iterations": iterations, "corrections": corrections,
                           "source_checkpoint_sha256": source_checkpoint_sha256})
        if self.crash_before:
            self.crash_before = False
            raise ControllerCrash
        stage = Path(stage).resolve()
        stage.mkdir(parents=True)
        files = {name: stage / name for name in (
            "job.json", "worker.json", "execution.json", "dryrun.train.json", "training.manifest.json",
            "training.readback.json", "optimizer.records.jsonl", "telemetry.json")}
        RT._write(files["job.json"], {
            "schema": "brev_recovery_worker_job_v1", "mode": "train", "stage": str(stage), "arm": arm,
            "seed": seed, "run_name": run_name, "iterations": iterations,
            "source_checkpoint_sha256": source_checkpoint_sha256, "protocol_sha256_lf": self.source_sha,
            "source_commit_sha": self.anchor, "manifest_path": str(files["training.manifest.json"]),
            "records_path": str(files["optimizer.records.jsonl"]),
            "readback_path": str(files["training.readback.json"]),
            "telemetry_path": str(files["telemetry.json"]),
        })
        self.reader()
        before = dict(self.reader.last_receipt or {})
        started_at = self.clock().isoformat()
        self.clock.advance(self.clock_s)
        base_execution = {
            "schema": "brev_recovery_posix_execution_v1", "mode": "train", "started_at": started_at,
            "launch": {"pid": 1234, "start_ticks": 1}, "gpu_before": [], "wall_s": self.wall_s,
            "gpu_before_observation": before, "source_before": self.snapshot,
            "finished_at": self.clock().isoformat(),
        }
        if self.fail:
            self.reader()
            execution = {**base_execution, "completed": False, "exit_code": 1, "timed_out": False,
                         "cleanup_verified": True, "gpu_after": [],
                         "gpu_after_observation": dict(self.reader.last_receipt or {}),
                         "source_after": self.snapshot, "error_type": "InvalidExecutionEvidence"}
            RT._write(files["execution.json"], execution)
            raise B.BackendError("InvalidExecutionEvidence", execution)
        checkpoint = self.lab / "logs" / run_name / "model_19.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(f"{run_name}:{seed}".encode())
        checkpoint_ref = {"path": str(checkpoint), "sha256": V1._sha(checkpoint)}
        RT._write(files["dryrun.train.json"], {
            "schema": "brev_compute_probe_train_v1", "mode": "train", "completed": True, "task": B.TASK,
            "observed_task": B.TASK, "seed": seed, "run_name": run_name, "num_envs": 4096,
            "observed_num_envs": 4096, "requested_iterations": iterations, "initial_learning_iteration": 0,
            "final_learning_iteration": self.observed_iterations - 1,
            "observed_iterations": self.observed_iterations, "checkpoint_path": checkpoint_ref["path"],
            "checkpoint_sha256_raw": checkpoint_ref["sha256"], "learn_wall_s": 300.0, "wall_s": 400.0,
        })
        RT._write(files["telemetry.json"], {"summary": {}})
        manifest = {"schema": "brev_recovery_dryrun_manifest_v1", "arm": "NONE", "seed": seed, "iterations": 20}
        readback = {"schema": "brev_recovery_dryrun_readback_v1", "remaining_faults": []}
        RT._write(files["training.manifest.json"], manifest)
        RT._write(files["training.readback.json"], readback)
        files["optimizer.records.jsonl"].write_bytes(b"")
        telemetry = {"path": str(files["telemetry.json"]), "sha256": V1._sha(files["telemetry.json"]),
                     "summary": {}}
        manifest_sha = V1._sha(files["training.manifest.json"])
        RT._write(files["worker.json"], {
            "schema": "brev_recovery_train_worker_v1", "completed": True, "arm": "NONE", "seed": seed,
            "iterations": 20, "learn_wall_s": 300.0, "wall_s": 400.0, "checkpoint": checkpoint_ref,
            "telemetry": telemetry, "manifest_path": str(files["training.manifest.json"]),
            "manifest_sha256": manifest_sha,
            "records": {"path": str(files["optimizer.records.jsonl"]),
                        "sha256": V1._sha(files["optimizer.records.jsonl"])},
            "readback": readback, "readback_sha256": V1._sha(files["training.readback.json"]),
        })
        self.reader()
        execution = {**base_execution, "completed": True, "exit_code": 0, "timed_out": False,
                     "cleanup_verified": True, "gpu_after": [],
                     "gpu_after_observation": dict(self.reader.last_receipt or {}),
                     "source_after": self.snapshot}
        RT._write(files["execution.json"], execution)
        if self.crash_after:
            self.crash_after = False
            raise ControllerCrash
        return {"checkpoint": checkpoint_ref, "telemetry": telemetry, "seed": seed, "arm": arm,
                "original_faults": [], "applied_corrections": [], "readback": readback,
                "execution": execution, "manifest_path": str(files["training.manifest.json"]),
                "manifest_sha256": manifest_sha, "manifest": manifest, "source_checkpoint_sha256": None}


class FakePreflightBackend(RT.FakeRenderBackend):
    """실제 gpu-execution.lock을 쓰는 fixture backend."""

    def __init__(self, runtime: Path, clock: Any, recovery: FakeRecovery, forbid: Any):
        super().__init__(runtime, clock)
        self.recovery = recovery
        self.forbid = forbid
        self.crash_after_smoke = False

    @contextmanager
    def action_admission(self):
        if self.forbid():
            raise AssertionError("backend admission은 offline 마감에서 호출되면 안 된다")
        with VB._gpu_lock(self.runtime / "gpu-execution.lock"):
            if self.permit_busy_once:
                self.permit_busy_once = False
                raise B.BackendError("ForeignGPUBusy")
            yield

    def smoke(self, job: dict[str, Any]) -> dict[str, Any]:
        started = self.clock()
        value = super().smoke(job)
        attestation = self.runtime / "record" / job["job_id"] / "attestation/attestation.json"
        if not attestation.exists():
            RT._write(attestation, {"execute_started_at_utc": started.isoformat(),
                                    "execute_finished_at_utc": self.clock().isoformat(),
                                    "attested_at_utc": self.clock().isoformat()})
        if self.crash_after_smoke:
            self.crash_after_smoke = False
            raise ControllerCrash
        return value

    def capture_references(self, job: dict[str, Any]) -> dict[str, dict[str, str]]:
        return _fake_refs(self.runtime.parents[2], self.runtime, job["job_id"])


def _setup(base: Path, monkeypatch: pytest.MonkeyPatch, *, compute_age_hours: float = 1.0,
           whole_budget: float = 20.0, deadline: str = "2027-01-01T12:00:00+00:00") -> SimpleNamespace:
    base.mkdir(parents=True, exist_ok=True)
    repo, lab, source_path, _, _ = V1._fixture(base, compute_age_hours=compute_age_hours)
    source = P.SOURCE.read(source_path)
    source["resources"]["whole_batch_budget_usd"] = whole_budget
    source["resources"]["deadline_utc"] = deadline
    for relative in sorted(RT.D.REQUIRED_CAPTURE_SOURCES | P.CONTROLLER_SOURCES):
        target = repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    for relative in RT.D.REQUIRED_CAPTURE_SOURCES:
        digest = P.SOURCE.sha256(repo / relative, lf=True)
        source["source_sha256_lf"][relative] = digest
        source["recording"]["source_files_lf"][relative] = digest
    RT._write(source_path, source)
    source_sha = P.SOURCE.sha256(source_path, lf=True)
    source_ref = {"path": source_path.relative_to(repo).as_posix(), "sha256_lf": source_sha}
    render_path = repo / "bench/protocols/vision_render.json"
    render = RT.D.build_protocol(source, source_ref)
    RT._write(render_path, render)
    render_sha = P.SOURCE.sha256(render_path, lf=True)
    render_ref = {"path": render_path.relative_to(repo).as_posix(), "sha256_lf": render_sha}
    RT._rebind_resource(base, source_sha, whole_budget)
    preflight = P.build_protocol(source, source_ref, render, render_ref, SEED, P.controller_source_hashes(repo))
    preflight_path = repo / "bench/protocols/vision_preflight.json"
    RT._write(preflight_path, preflight)
    preflight_sha = P.SOURCE.sha256(preflight_path, lf=True)
    subprocess.run(["git", "-C", str(repo), "add", "--", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "preflight fixture"], check=True)
    anchor = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True,
                            capture_output=True, text=True).stdout.strip()
    clock = RT.Clock()
    resource_path = base / "resource.json"
    runtime = repo.resolve() / ".triage-runtime/vision-brev/vision-source-test"
    fx = SimpleNamespace(repo=repo, lab=lab, clock=clock, source=source, source_path=source_path,
                         source_sha=source_sha, source_ref=source_ref, render=render, render_path=render_path,
                         render_sha=render_sha, render_ref=render_ref, preflight=preflight,
                         preflight_path=preflight_path, preflight_sha=preflight_sha, anchor=anchor,
                         resource_path=resource_path, runtime=runtime, backends=[], guard_calls=[],
                         capture_checks=[], fresh=True, recovery=None, forbid_backend=False)

    def factory(lab_arg: Path, python: Path, protocol: dict[str, Any], anchor_arg: str, runtime_arg: Path,
                plan: list[dict[str, Any]], **kwargs: Any) -> FakePreflightBackend:
        if fx.forbid_backend:
            raise AssertionError("backend factory는 offline 마감에서 호출되면 안 된다")
        assert lab_arg == lab.resolve() and anchor_arg == anchor and len(plan) == 6
        assert Path(runtime_arg) == runtime and protocol["protocol_sha256_lf"] == source_sha
        if fx.recovery is None:
            fx.recovery = FakeRecovery(runtime, lab, clock, anchor, source_sha, kwargs["resource_proof"])
        backend = FakePreflightBackend(runtime, clock, fx.recovery, lambda: fx.forbid_backend)
        fx.backends.append(backend)
        return backend

    def guard(seeds: list[int], _lab: Path, repo: Path | None = None) -> dict[str, Any]:
        fx.guard_calls.append(list(seeds))
        status = "fresh" if fx.fresh else "used"
        return {"schema": "preregistered_seed_freshness_v1", "status": status, "seeds": list(seeds)}

    def capture_check(repo_arg: Path, references: dict[str, Any], *_args: Any) -> None:
        assert repo_arg == repo.resolve() and {"job", "admission", "result", "execution"} <= set(references)
        fx.capture_checks.append(sorted(references))

    monkeypatch.setattr(P, "_verify_capture_files", capture_check)
    monkeypatch.setattr(P, "_capture_references_offline", _fake_refs)
    fx.factory, fx.guard = factory, guard
    fx.batch = lambda: P.PreflightBatch(
        repo, source_path, source_sha, render_path, render_sha, preflight_path, preflight_sha, anchor,
        resource_path, P.SOURCE.sha256(resource_path), lab, Path(sys.executable),
        backend_factory=factory, seed_guard=guard, now=clock,
    )
    fx.inspect = lambda now=None: P.inspect_preflight(
        repo, source_path, source_sha, render_path, render_sha, preflight_path, preflight_sha, anchor,
        resource_path, P.SOURCE.sha256(resource_path), now=now or clock(),
    )
    fx.settle = lambda now=None: P.settle_preflight(
        repo, source_path, source_sha, render_path, render_sha, preflight_path, preflight_sha, anchor,
        resource_path, P.SOURCE.sha256(resource_path), now=now or clock(),
    )
    return fx


def _complete(fx: SimpleNamespace) -> Any:
    batch = fx.batch()
    assert batch.advance()["status"] == "advanced"
    assert batch.advance()["status"] == "preflight_complete"
    return batch


def _stage(fx: SimpleNamespace, index: int) -> Path:
    return fx.runtime / "preflight/stages" / f"{index:02d}_{P.STAGES[index]}"


def test_protocol_fixes_none20_smoke_order_limits_and_rejects_changes(tmp_path: Path, monkeypatch) -> None:
    fx = _setup(tmp_path, monkeypatch)
    protocol = fx.preflight
    assert P.preflight_ids(SEED) == {
        "train_job_id": "preflight-train-s4199", "run_name": "preflight-none20-s4199",
        "smoke_job_id": "preflight-smoke-s4199", "checkpoint_logical_id": "preflight-policy-s4199",
    }
    assert protocol["training"] == {
        "job_id": "preflight-train-s4199", "arm": "NONE", "seed": SEED, "run_name": "preflight-none20-s4199",
        "iterations": 20, "num_envs": 4096, "stage_relative": "preflight/training/preflight-train-s4199",
    }
    assert protocol["order"] == ["train_none20", "render_sync_smoke"] and protocol["model_use_ready"] is False
    assert protocol["limits"]["total_max_wall_s"] == 2700
    assert protocol["limits"]["total_max_compute_usd"] == pytest.approx(1.5)
    assert protocol["reservation"]["source_actions_usd"] == pytest.approx(7.0)
    assert protocol["reservation"]["render_smokes_usd"] == pytest.approx(2.0)
    args = (fx.source, fx.source_ref, fx.render, fx.render_ref)
    mutations = [
        ("training", "iterations", 300), ("training", "num_envs", 1024), ("training", "arm", "F1"),
        ("training", "run_name", "s4101-none"), ("smoke", "policy_steps", 1),
        ("limits", "total_max_wall_s", 2701), (None, "model_use_ready", True),
        (None, "record_profile_sha256", "0" * 64), (None, "order", ["render_sync_smoke", "train_none20"]),
    ]
    for section, key, value in mutations:
        changed = json.loads(json.dumps(protocol))
        (changed[section] if section else changed)[key] = value
        with pytest.raises(ValueError):
            P.validate_protocol(changed, *args)
    missing = json.loads(json.dumps(protocol))
    del missing["controller_sources_lf"]["bench/run_vision_brev_preflight_v1.py"]
    with pytest.raises(ValueError, match="controller"):
        P.validate_protocol(missing, *args)
    hashes = P.controller_source_hashes(fx.repo)
    with pytest.raises(ValueError, match="core 학습 seed"):
        P.build_protocol(fx.source, fx.source_ref, fx.render, fx.render_ref, 4101, hashes)
    tight = json.loads(json.dumps(fx.source))
    tight["resources"]["whole_batch_budget_usd"] = 10.0
    with pytest.raises(ValueError, match="whole batch 상한"):
        P.build_protocol(tight, fx.source_ref, RT.D.build_protocol(tight, fx.source_ref), fx.render_ref,
                         SEED, hashes)
    colliding = json.loads(json.dumps(fx.source))
    colliding["source_action_order"][6]["checkpoint_logical_id"] = "preflight-policy-s4199"
    with pytest.raises(ValueError, match="충돌"):
        P.build_protocol(colliding, fx.source_ref, RT.D.build_protocol(colliding, fx.source_ref),
                         fx.render_ref, SEED, hashes)


def test_none20_then_smoke_share_runtime_and_leave_core_ledger_untouched(tmp_path: Path, monkeypatch) -> None:
    fx = _setup(tmp_path, monkeypatch)
    batch = fx.batch()
    verified = batch.verify()
    assert verified["preflight_protocol_sha256_lf"] == fx.preflight_sha and verified["model_use_ready"] is False
    first = batch.advance()
    assert first == {"status": "advanced", "stage": "train_none20", "completed_stages": 1,
                     "remaining_stages": 1, "recovered_from_cache": False, "model_use_ready": False,
                     "ready_for_source_batch": False}
    assert fx.recovery.calls == [{
        "stage": fx.runtime / "preflight/training/preflight-train-s4199", "arm": "NONE", "seed": SEED,
        "run_name": "preflight-none20-s4199", "iterations": 20, "corrections": None,
        "source_checkpoint_sha256": None,
    }]
    assert fx.guard_calls == [[SEED]] and fx.backends[0].smoke_runs == []
    second = batch.advance()
    assert second["status"] == "preflight_complete" and second["ready_for_source_batch"] is True
    assert second["model_use_ready"] is False and second["readiness_reasons"] == []
    assert fx.backends[0].smoke_runs == ["preflight-smoke-s4199"]
    smoke_job = P.SOURCE.read(fx.runtime / "record/preflight-smoke-s4199/job.json")
    train_raw = P.SOURCE.read(_stage(fx, 0) / "raw.json")
    assert smoke_job["checkpoint"] == {**train_raw["artifact"]["checkpoint"],
                                       "logical_id": "preflight-policy-s4199"}
    assert smoke_job["mode"] == "render_sync_smoke" and len(fx.capture_checks) >= 1
    for name in CORE_FILES:
        assert not (fx.runtime / name).exists(), name
    completion = P.SOURCE.read(fx.repo / second["completion"]["path"])
    assert completion["model_use_ready"] is False and completion["scope"] == P.SCOPE
    assert [row["stage"] for row in completion["stages"]] == ["train_none20", "render_sync_smoke"]
    again = batch.advance()
    assert again["status"] == "preflight_complete" and len(fx.recovery.calls) == 1
    assert fx.backends[0].smoke_runs == ["preflight-smoke-s4199"]
    status = fx.inspect()
    assert status["status"] == "preflight_complete" and status["ready_for_source_batch"] is True
    assert status["completion"] == second["completion"]
    source_batch = RT.D.RenderBatch(
        fx.repo, fx.source_path, fx.source_sha, fx.render_path, fx.render_sha, fx.anchor, fx.resource_path,
        P.SOURCE.sha256(fx.resource_path), fx.lab, Path(sys.executable), backend_factory=fx.factory,
        seed_guard=fx.guard, now=fx.clock,
    )
    started = source_batch.advance()
    assert started["status"] == "advanced" and started["action_id"] == "train-4101-none"
    assert fx.guard_calls[-1] == [4101, 4102]
    assert fx.inspect()["source_batch_started"] is True


def test_same_gpu_lock_and_busy_gpu_write_no_freshness_or_intent(tmp_path: Path, monkeypatch) -> None:
    fx = _setup(tmp_path, monkeypatch)
    batch = fx.batch()
    with VB._gpu_lock(fx.runtime / "gpu-execution.lock"), pytest.raises(B.BackendError, match="GPUExecutionBusy"):
        batch.advance()
    assert not (fx.runtime / "preflight").exists() and fx.guard_calls == []
    batch._backend().permit_busy_once = True
    with pytest.raises(B.BackendError, match="ForeignGPUBusy"):
        batch.advance()
    assert not (fx.runtime / "preflight").exists() and fx.recovery.calls == []
    assert batch.advance()["status"] == "advanced"


@pytest.mark.parametrize(("case", "reason"), [
    ("budget", "preflight_budget_insufficient"), ("window", "preflight_window_insufficient"),
    ("seed", "preflight_seed_not_fresh"), ("source_started", "source_batch_already_started"),
])
def test_gates_hold_before_any_intent_or_train(tmp_path: Path, monkeypatch, case: str, reason: str) -> None:
    options: dict[str, Any] = {}
    if case == "budget":
        options["compute_age_hours"] = 5.0
    if case == "window":
        options["deadline"] = "2027-01-01T00:20:00+00:00"
    fx = _setup(tmp_path, monkeypatch, **options)
    fx.fresh = case != "seed"
    if case == "source_started":
        (fx.runtime / "actions").mkdir(parents=True)
    result = fx.batch().advance()
    assert result["status"] == "HOLD" and result["reason"] == reason
    assert result["model_use_ready"] is False and result["ready_for_source_batch"] is False
    assert not (_stage(fx, 0) / "intent.json").exists()
    assert not (fx.runtime / "preflight/freshness.json").exists()
    assert fx.recovery is None or fx.recovery.calls == []


def test_backend_failure_is_preserved_and_never_retried(tmp_path: Path, monkeypatch) -> None:
    fx = _setup(tmp_path, monkeypatch)
    batch = fx.batch()
    batch._backend()
    fx.recovery.fail = True
    result = batch.advance()
    assert result["status"] == "HOLD" and result["reason"] == "preflight_stage_failed"
    failure = P.SOURCE.read(_stage(fx, 0) / "failure.json")
    assert failure["reason"] == "InvalidExecutionEvidence"
    fx.recovery.fail = False
    assert batch.advance()["reason"] == "preflight_stage_failed"
    assert len(fx.recovery.calls) == 1 and fx.inspect()["status"] == "HOLD"


def test_intent_only_holds_then_becomes_invalid_without_resend(tmp_path: Path, monkeypatch) -> None:
    fx = _setup(tmp_path, monkeypatch)
    batch = fx.batch()
    batch._backend()
    fx.recovery.crash_before = True
    with pytest.raises(ControllerCrash):
        batch.advance()
    assert (_stage(fx, 0) / "intent.json").exists()
    fx.forbid_backend = True
    assert fx.batch().advance()["reason"] == "preflight_stage_pending"
    assert batch.advance()["reason"] == "preflight_stage_pending"
    fx.clock.advance(1801)
    result = fx.settle()
    assert result is not None
    assert result["reason"] == "preflight_stage_invalid" and result["detail"] == "pending_unresolved_after_window"
    assert batch.advance()["reason"] == "preflight_stage_invalid"
    assert len(fx.recovery.calls) == 1 and len(fx.backends) == 1


def test_complete_caches_close_pending_stages_without_second_execution(tmp_path: Path, monkeypatch) -> None:
    fx = _setup(tmp_path, monkeypatch)
    batch = fx.batch()
    batch._backend()
    fx.recovery.crash_after = True
    with pytest.raises(ControllerCrash):
        batch.advance()
    fx.clock.advance(1500)
    fx.forbid_backend = True
    recovered = fx.batch().advance()
    assert recovered["status"] == "advanced" and recovered["recovered_from_cache"] is True
    receipt = P.SOURCE.read(_stage(fx, 0) / "receipt.json")
    execution = P.SOURCE.read(fx.runtime / "preflight/training/preflight-train-s4199/execution.json")
    assert receipt["finished_at_utc"] == execution["finished_at"] and receipt["elapsed_wall_s"] == 600.0
    assert len(fx.recovery.calls) == 1 and len(fx.backends) == 1
    fx.forbid_backend = False
    fx.backends[0].crash_after_smoke = True
    with pytest.raises(ControllerCrash):
        batch.advance()
    fx.clock.advance(800)
    fx.forbid_backend = True
    finished = batch.advance()
    assert finished["status"] == "preflight_complete" and finished["recovered_from_cache"] is True
    assert P.SOURCE.read(_stage(fx, 1) / "receipt.json")["elapsed_wall_s"] == 120.0
    assert fx.backends[0].smoke_runs == ["preflight-smoke-s4199"] and len(fx.recovery.calls) == 1


@pytest.mark.parametrize("stage", ["train", "smoke"])
def test_expired_lease_closes_complete_cache_offline_with_zero_backend_calls(
    tmp_path: Path, monkeypatch, stage: str,
) -> None:
    fx = _setup(tmp_path, monkeypatch)
    batch = fx.batch()
    batch._backend()
    if stage == "train":
        fx.recovery.crash_after = True
    else:
        assert batch.advance()["status"] == "advanced"
        fx.backends[0].crash_after_smoke = True
    with pytest.raises(ControllerCrash):
        batch.advance()
    late = datetime(2027, 1, 2, 1, tzinfo=UTC)
    fx.clock.value = late
    fx.forbid_backend = True
    with pytest.raises(ValueError, match="resource proof 유효 시간"):
        fx.batch()
    result = fx.settle()
    assert result is not None and result["recovered_from_cache"] is True
    assert result["status"] == ("advanced" if stage == "train" else "preflight_complete")
    assert result["ready_for_source_batch"] is False and result["model_use_ready"] is False
    if stage == "smoke":
        assert "resource_gate_failed" in result["readiness_reasons"] and result["completion"] is not None
    status = fx.inspect(late)
    assert status["status"] == ("train_complete" if stage == "train" else "preflight_complete")
    assert len(fx.recovery.calls) == 1 and len(fx.backends) == 1
    assert fx.backends[0].smoke_runs == ([] if stage == "train" else ["preflight-smoke-s4199"])


def test_wrong_observed_iterations_and_stage_overrun_are_invalid(tmp_path: Path, monkeypatch) -> None:
    fx = _setup(tmp_path / "iterations", monkeypatch)
    batch = fx.batch()
    batch._backend()
    fx.recovery.observed_iterations = 19
    result = batch.advance()
    assert result["reason"] == "preflight_stage_invalid" and result["detail"] == "train_evidence_invalid"
    assert not (_stage(fx, 0) / "raw.json").exists()
    slow = _setup(tmp_path / "overrun", monkeypatch)
    slow_batch = slow.batch()
    slow_batch._backend()
    slow.recovery.clock_s = 1900.0
    result = slow_batch.advance()
    assert result["reason"] == "preflight_stage_invalid" and result["detail"] == "preflight_stage_limit_exceeded"
    assert (_stage(slow, 0) / "raw.json").exists() and not (_stage(slow, 0) / "receipt.json").exists()
    assert len(slow.recovery.calls) == 1 and slow.backends[0].smoke_runs == []


@pytest.mark.parametrize("target", ["checkpoint", "dryrun", "smoke_result", "gpu_observation", "receipt"])
def test_completed_cache_tamper_is_rejected(tmp_path: Path, monkeypatch, target: str) -> None:
    fx = _setup(tmp_path, monkeypatch)
    batch = _complete(fx)
    raw = P.SOURCE.read(_stage(fx, 0) / "raw.json")
    if target == "checkpoint":
        Path(raw["artifact"]["checkpoint"]["path"]).write_bytes(b"changed")
    elif target == "dryrun":
        path = fx.runtime / "preflight/training/preflight-train-s4199/dryrun.train.json"
        body = P.SOURCE.read(path)
        body["observed_num_envs"] = 2048
        RT._write(path, body)
    elif target == "smoke_result":
        RT._write(fx.runtime / "record/preflight-smoke-s4199/output/result.json", {"mode": "changed"})
    elif target == "gpu_observation":
        (fx.runtime / "gpu-observations/000001.process.csv").write_text(
            "GPU-fixture-1, 99\n", encoding="utf-8", newline="")
    else:
        receipt_path = _stage(fx, 1) / "receipt.json"
        receipt = P.SOURCE.read(receipt_path)
        receipt["cost_after_usd"] = 0.0
        receipt_path.unlink()
        RT._write(receipt_path, receipt)
    with pytest.raises(ValueError):
        batch.advance()
    with pytest.raises(ValueError):
        fx.inspect()


def test_expired_lease_still_reads_completion_but_blocks_source_and_new_runs(tmp_path: Path, monkeypatch) -> None:
    fx = _setup(tmp_path, monkeypatch)
    _complete(fx)
    late = datetime(2027, 1, 2, 1, tzinfo=UTC)
    status = fx.inspect(late)
    assert status["status"] == "preflight_complete" and status["completion"] is not None
    assert status["ready_for_source_batch"] is False and status["model_use_ready"] is False
    assert {"resource_gate_failed", "source_first_action_window_insufficient"} <= set(status["readiness_reasons"])
    near_deadline = fx.inspect(datetime(2027, 1, 1, 11, 40, tzinfo=UTC))
    assert near_deadline["resource_gate_error"] is None
    assert "source_first_action_window_insufficient" in near_deadline["readiness_reasons"]
    assert near_deadline["ready_for_source_batch"] is False
    fx.clock.value = late
    with pytest.raises(ValueError, match="resource proof 유효 시간"):
        fx.batch()


def test_status_before_start_and_partial_train(tmp_path: Path, monkeypatch) -> None:
    fx = _setup(tmp_path, monkeypatch)
    initial = fx.inspect()
    assert initial["status"] == "not_started" and initial["ready_for_source_batch"] is False
    assert initial["readiness_reasons"] == ["preflight_incomplete"]
    fx.batch().advance()
    partial = fx.inspect()
    assert partial["status"] == "train_complete"
    assert [row["state"] for row in partial["stages"]] == ["complete", "absent"]


def test_cli_build_prints_protocol_and_status_exit_codes(tmp_path: Path, monkeypatch, capsys) -> None:
    fx = _setup(tmp_path, monkeypatch)
    common = ["--repo", str(fx.repo), "--protocol", str(fx.source_path), "--protocol-sha256-lf", fx.source_sha,
              "--render-protocol", str(fx.render_path), "--render-protocol-sha256-lf", fx.render_sha]
    assert P.main(["build", *common, "--seed", str(SEED)]) == 0
    assert json.loads(capsys.readouterr().out) == fx.preflight

    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz: Any = None) -> datetime:  # type: ignore[override]
            return datetime(2027, 1, 1, tzinfo=UTC)

    monkeypatch.setattr(P, "datetime", FixedDatetime)
    status_args = [*common, "--preflight-protocol", str(fx.preflight_path), "--preflight-protocol-sha256-lf",
                   fx.preflight_sha, "--source-anchor", fx.anchor, "--resource-proof", str(fx.resource_path),
                   "--resource-sha256", P.SOURCE.sha256(fx.resource_path)]
    assert P.main(["status", *status_args]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "not_started"
    batch = fx.batch()
    batch._backend()
    fx.recovery.fail = True
    batch.advance()
    assert P.main(["status", *status_args]) == 2
    assert json.loads(capsys.readouterr().out)["stages"][0]["state"] == "failed"
    with pytest.raises(SystemExit):
        P.main(["advance", *status_args])
