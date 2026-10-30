from __future__ import annotations

import hashlib
import importlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "bench"), str(ROOT / "evals")]
B = importlib.import_module("lineage_source_batch_v1")


BUDGET = {
    "smoke_updates": 1,
    "source_training_runs": 18,
    "source_updates": 5400,
    "model_api_requests": 0,
    "per_job_parent_wall_s": 1800,
    "total_parent_wall_s": 21600,
    "automatic_retries": 0,
    "paid_cloud_gpu_starts": 0,
}


def _sha(path: Path, *, lf: bool = False) -> str:
    raw = path.read_bytes()
    if lf:
        raw = raw.replace(b"\r\n", b"\n")
    return hashlib.sha256(raw).hexdigest()


def _protocol() -> dict:
    cases, training, smoke = B.jobs()
    return {
        "schema": B.SCHEMA,
        "phase": "source_collection_only",
        "cases": cases,
        "training_jobs": training,
        "smoke_job": smoke,
        "source_sha256_lf": {name: "a" * 64 for name in B.CHILD.SOURCE_FILES},
        "runtime_source_sha256_lf": {name: "b" * 64 for name in B.RUNTIME_FILES},
        "budget": dict(BUDGET),
    }


@pytest.fixture
def batch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    repo, lab = tmp_path / "repo", tmp_path / "lab"
    protocol_path = repo / B.PROTOCOL
    protocol_path.parent.mkdir(parents=True)
    value = _protocol()
    protocol_path.write_text(json.dumps(value), encoding="utf-8")
    monkeypatch.setattr(B, "ROOT", repo)
    monkeypatch.setattr(B, "source_files", lambda: tuple(B.CHILD.SOURCE_FILES))
    binding_calls: list[str] = []

    def bound(fixed, *, repo, lab):
        assert repo == B.ROOT and Path(lab) == batch_lab
        binding_calls.append(fixed["job_id"])

    batch_lab = lab
    monkeypatch.setattr(B.CHILD, "verify_binding", bound)
    return {"repo": repo, "lab": lab, "value": value, "bindings": binding_calls,
            "root": tmp_path / "batch", "python": tmp_path / "python.exe"}


def _lease() -> dict:
    return {"owner": B.OWNER, "leaseId": "11111111-1111-1111-1111-111111111111",
            "state": "running"}


def _completion(stage: Path, argv: list[str], lab: Path, timeout: float, *,
                missing: bool = False, tamper_checkpoint: bool = False) -> dict:
    del lab, timeout
    assert not (stage / "launch.json").exists(), "parent가 owned runner의 launch.json을 선점했다"
    manifest_path = Path(argv[argv.index("--manifest") + 1])
    output = Path(argv[argv.index("--out") + 1])
    fixed = json.loads(manifest_path.read_text(encoding="utf-8"))
    B.GUARD.durable_new(stage / "launch.json", {"argv": argv, "owner": "fake-run-owned"})
    if not missing:
        log_dir = stage / "logs"
        log_dir.mkdir()
        checkpoint = log_dir / f"model_{fixed['max_iterations'] - 1}.pt"
        checkpoint.write_bytes((fixed["job_id"] + "-checkpoint").encode())
        checkpoint_sha = _sha(checkpoint)
        manifest_sha = B.CHILD.manifest_fingerprint(fixed)
        records = output / B.GUARD.RECORDS
        records.parent.mkdir(parents=True)
        records.write_text(json.dumps({"job_id": fixed["job_id"]}) + "\n", encoding="utf-8")
        evidence = {
            "schema": "lineage_source_training_evidence_v1",
            "manifest_sha256": manifest_sha,
            "updates": fixed["max_iterations"],
            "update_count": fixed["max_iterations"],
            "arm": fixed["arm"],
            "seed": fixed["seed"],
            "learning_completed": True,
            "recorded_before_app_close": True,
            "learn_returned": True,
            "log_dir": str(log_dir),
            "checkpoint": {"path": str(checkpoint), "sha256": checkpoint_sha},
        }
        B.GUARD.durable_new(output / B.GUARD.EVIDENCE, evidence)
        B.GUARD.durable_new(output / B.GUARD.CLOSE_READY, {
            "schema": B.GUARD.CLOSE_SCHEMA,
            "manifest_sha256": manifest_sha,
            "updates": fixed["max_iterations"],
            "evidence_sha256_lf": _sha(output / B.GUARD.EVIDENCE, lf=True),
            "records_sha256_lf": _sha(records, lf=True),
        })
        if tamper_checkpoint:
            checkpoint.write_bytes(b"tampered-after-evidence")
    return {"exit_code": 0, "natural_exit": True, "cleanup_verified": True,
            "job_assigned_before_release": True, "timed_out": False,
            "error_type": None, "wall_s": 0.25}


def test_jobs_are_18_for_10_cases_with_same_seed_references() -> None:
    cases, training, smoke = B.jobs()
    assert len(cases) == 10 and len(training) == 18
    assert smoke == {"job_id": "smoke-none-s2399", "case_id": "preflight", "arm": "NONE",
                     "seed": 2399, "run_name": "lineage_v1_smoke_s2399", "task": B.CHILD.TASK,
                     "num_envs": 4096, "max_iterations": 1}
    by_id = {row["job_id"]: row for row in training}
    for case in cases:
        source, reference = by_id[case["source_job"]], by_id[case["reference_job"]]
        assert source["seed"] == reference["seed"] == case["seed"]
        assert reference["arm"] == "NONE"
    assert len({case["seed"] for case in cases} | {smoke["seed"]}) == 11


def test_protocol_rejects_budget_job_and_purpose_tamper(batch: dict) -> None:
    value = batch["value"]
    B.validate_protocol(value)
    tampered = json.loads(json.dumps(value))
    tampered["budget"]["source_updates"] -= 1
    with pytest.raises(ValueError, match="예산"):
        B.validate_protocol(tampered)
    tampered = json.loads(json.dumps(value))
    tampered["training_jobs"][0]["max_iterations"] = 299
    with pytest.raises(ValueError, match="실행 순서"):
        B.validate_protocol(tampered)
    with pytest.raises(ValueError, match="purpose"):
        B.manifest(value, value["smoke_job"], "a" * 40, purpose="diagnosis")
    with pytest.raises(ValueError, match="mode"):
        B.run_sequence(batch["root"], value, "a" * 40, batch["lab"], batch["python"],
                       mode="source", runner=_completion, check_lease=_lease)


def test_smoke_success_has_bound_order_and_owned_launch(batch: dict) -> None:
    result = B.run_sequence(batch["root"], batch["value"], "a" * 40,
                            batch["lab"], batch["python"], mode="smoke",
                            runner=_completion, check_lease=_lease)
    assert result["status"] == "complete" and result["source_jobs_completed"] == 0
    assert batch["bindings"] == ["smoke-none-s2399"]
    assert json.loads((batch["root"] / "smoke-none-s2399" / "launch.json").read_text())["owner"] == "fake-run-owned"


@pytest.mark.parametrize("failure", ["missing", "false_exit"])
def test_false_completion_is_rejected_and_next_job_never_starts(batch: dict, failure: str) -> None:
    calls = []

    def runner(stage, argv, lab, timeout):
        calls.append(stage.name)
        receipt = _completion(stage, argv, lab, timeout, missing=failure == "missing")
        if failure == "false_exit":
            receipt["exit_code"] = 1
        return receipt

    with pytest.raises((ValueError, RuntimeError)):
        B.run_sequence(batch["root"], batch["value"], "a" * 40,
                       batch["lab"], batch["python"], mode="all",
                       runner=runner, check_lease=_lease)
    assert calls == ["smoke-none-s2399"]
    assert (batch["root"] / "failure.json").is_file()


def test_runner_failure_and_lease_drop_stop_before_next_job(batch: dict) -> None:
    calls = []

    def failed(stage, argv, lab, timeout):
        del argv, lab, timeout
        calls.append(stage.name)
        raise RuntimeError("owned runner failed")

    with pytest.raises(RuntimeError, match="owned runner failed"):
        B.run_sequence(batch["root"], batch["value"], "a" * 40,
                       batch["lab"], batch["python"], mode="all",
                       runner=failed, check_lease=_lease)
    assert calls == ["smoke-none-s2399"]

    second_root = batch["root"].with_name("lease-drop")
    calls.clear()
    lease_checks = 0

    def dropping_lease():
        nonlocal lease_checks
        lease_checks += 1
        current = _lease()
        if lease_checks >= 3:
            current["leaseId"] = "22222222-2222-2222-2222-222222222222"
        return current

    def complete(stage, argv, lab, timeout):
        calls.append(stage.name)
        return _completion(stage, argv, lab, timeout)

    with pytest.raises(ValueError, match="lease ID"):
        B.run_sequence(second_root, batch["value"], "a" * 40,
                       batch["lab"], batch["python"], mode="all",
                       runner=complete, check_lease=dropping_lease)
    assert calls == ["smoke-none-s2399"]


def test_existing_output_and_checkpoint_byte_tamper_are_rejected(batch: dict) -> None:
    batch["root"].mkdir()
    with pytest.raises(FileExistsError):
        B.run_sequence(batch["root"], batch["value"], "a" * 40,
                       batch["lab"], batch["python"], mode="smoke",
                       runner=_completion, check_lease=_lease)
    root = batch["root"].with_name("checkpoint-tamper")
    with pytest.raises(ValueError, match="checkpoint"):
        B.run_sequence(root, batch["value"], "a" * 40,
                       batch["lab"], batch["python"], mode="smoke",
                       runner=lambda *args: _completion(*args, tamper_checkpoint=True),
                       check_lease=_lease)


def test_validate_lease_accepts_7_digit_timestamp_and_rejects_external_pid() -> None:
    caller, wrapper = 40201, 40200
    metadata = {"owner": B.OWNER, "state": "running", "childPid": caller,
                "wrapperPid": wrapper, "completedUtc": None,
                "acquiredUtc": "2026-10-30T12:00:00.1234567Z",
                "heartbeatUtc": "2026-10-30T12:00:30.7654321Z",
                "leaseId": "11111111-1111-1111-1111-111111111111"}
    ancestors = [{"pid": caller, "parent": wrapper, "name": "python.exe"},
                 {"pid": wrapper, "parent": 1, "name": "pwsh.exe"}]
    now = datetime(2026, 10, 30, 12, 0, 31, tzinfo=timezone.utc)
    assert B.validate_lease(metadata, ancestors, caller, now) is metadata
    with pytest.raises(ValueError, match="신원"):
        B.validate_lease(metadata, ancestors, caller + 1, now)


def test_all_can_reuse_accepted_smoke_reference_without_rerun(batch: dict) -> None:
    smoke_root = batch["root"].with_name("smoke")
    smoke = B.run_sequence(smoke_root, batch["value"], "a" * 40,
                           batch["lab"], batch["python"], mode="smoke",
                           runner=_completion, check_lease=_lease)
    assert smoke["status"] == "complete"
    reference = B.verify_smoke(smoke_root, batch["value"], "a" * 40)
    calls = []

    def runner(stage, argv, lab, timeout):
        calls.append(stage.name)
        return _completion(stage, argv, lab, timeout)

    result = B.run_sequence(batch["root"], batch["value"], "a" * 40,
                            batch["lab"], batch["python"], mode="all", runner=runner,
                            check_lease=_lease, smoke_reference=reference)
    assert len(calls) == 18 and "smoke-none-s2399" not in calls
    assert result["source_jobs_completed"] == 18
    assert result["smoke_reference"] == reference
