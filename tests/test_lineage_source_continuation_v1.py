from __future__ import annotations

import hashlib
import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT / "bench"), str(PROJECT / "evals")]
E = importlib.import_module("lineage_source_continuation_v1")


def _sha(path: Path, *, lf: bool = False) -> str:
    raw = path.read_bytes()
    if lf:
        raw = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(raw).hexdigest()


def _source_protocol() -> dict:
    cases, training, smoke = E.B.jobs()
    return {"schema": E.B.SCHEMA, "phase": "source_collection_only", "cases": cases,
            "training_jobs": training, "smoke_job": smoke,
            "source_sha256_lf": {name: "a" * 64 for name in E.B.CHILD.SOURCE_FILES},
            "runtime_source_sha256_lf": {name: "b" * 64 for name in E.B.RUNTIME_FILES},
            "budget": {"smoke_updates": 1, "source_training_runs": 18, "source_updates": 5400,
                       "model_api_requests": 0, "per_job_parent_wall_s": 1800,
                       "total_parent_wall_s": 21600, "automatic_retries": 0,
                       "paid_cloud_gpu_starts": 0}}


def _completion(stage: Path, argv: list[str], _lab: Path, _timeout: float) -> dict:
    assert not (stage / "launch.json").exists()
    manifest_path = Path(argv[argv.index("--manifest") + 1])
    output = Path(argv[argv.index("--out") + 1])
    fixed = json.loads(manifest_path.read_text(encoding="utf-8"))
    E.GUARD.durable_new(stage / "launch.json", {"owner": "fixture-owned", "argv": argv})
    log_dir = stage / "logs"
    log_dir.mkdir()
    checkpoint = log_dir / "model_299.pt"
    checkpoint.write_bytes((fixed["job_id"] + "-checkpoint").encode())
    records = output / E.GUARD.RECORDS
    records.parent.mkdir(parents=True)
    records.write_text(json.dumps({"job_id": fixed["job_id"]}) + "\n", encoding="utf-8")
    fingerprint = E.B.CHILD.manifest_fingerprint(fixed)
    evidence = {"schema": "lineage_source_training_evidence_v1", "manifest_sha256": fingerprint,
                "updates": 300, "update_count": 300, "arm": fixed["arm"], "seed": fixed["seed"],
                "learning_completed": True, "recorded_before_app_close": True, "learn_returned": True,
                "log_dir": str(log_dir),
                "checkpoint": {"path": str(checkpoint), "sha256": _sha(checkpoint)}}
    E.GUARD.durable_new(output / E.GUARD.EVIDENCE, evidence)
    E.GUARD.durable_new(output / E.GUARD.CLOSE_READY, {
        "schema": E.GUARD.CLOSE_SCHEMA, "manifest_sha256": fingerprint, "updates": 300,
        "evidence_sha256_lf": _sha(output / E.GUARD.EVIDENCE, lf=True),
        "records_sha256_lf": _sha(records, lf=True)})
    return {"exit_code": 0, "natural_exit": True, "cleanup_verified": True,
            "job_assigned_before_release": True, "timed_out": False,
            "error_type": None, "wall_s": 0.25}


def _lease(identifier: str = "11111111-1111-1111-1111-111111111111") -> dict:
    return {"owner": E.B.OWNER, "leaseId": identifier, "state": "running"}


@pytest.fixture
def continuation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    repo, lab, prior = tmp_path / "repo", tmp_path / "lab", tmp_path / "private/runs/sources-v2"
    repo.mkdir(); lab.mkdir(); prior.mkdir(parents=True)
    (lab / "logs/rsl_rl/unitree_go2_flat").mkdir(parents=True)
    value = _source_protocol()
    source_protocol_path = repo / E.B.PROTOCOL
    source_protocol_path.parent.mkdir(parents=True)
    source_protocol_path.write_text(json.dumps(value), encoding="utf-8")
    for name in E.SOURCE_FILES:
        path = repo / name
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(name, encoding="utf-8")
    monkeypatch.setattr(E, "ROOT", repo)
    monkeypatch.setattr(E.B, "ROOT", repo)
    monkeypatch.setattr(E.B, "verify", lambda actual_lab, anchor: (
        value if Path(actual_lab) == lab and anchor == E.SOURCE_ANCHOR
        else (_ for _ in ()).throw(ValueError("binding"))))
    bindings = []
    monkeypatch.setattr(E.B.CHILD, "verify_binding", lambda fixed, **_kwargs: bindings.append(fixed["job_id"]))
    monkeypatch.setattr(E, "_seed_guard", lambda _lab: {
        "checked_seeds": [2308, 2309, 2310], "allowed_consumed": [2308],
        "required_unused": [2309, 2310], "evidence": [{"seed": 2308, "path": "fixture"}]})
    prefix, remaining = E._jobs()
    records = []
    for job in prefix:
        stage = prior / job["job_id"]
        stage.mkdir()
        fixed = E.B.manifest(value, job, E.SOURCE_ANCHOR, purpose="source")
        E.GUARD.durable_new(stage / "manifest.json", fixed)
        receipt = _completion(stage, ["python", "--manifest", str(stage / "manifest.json"),
                                      "--out", str(stage / "training")], lab, 1800)
        E.GUARD.durable_new(stage / "owned-result.json", receipt)
        evidence = E.B.completed(stage, fixed, receipt)
        record = {"job_id": job["job_id"], "purpose": "source", "arm": job["arm"],
                  "seed": job["seed"], "checkpoint": evidence["checkpoint"],
                  "evidence_sha256": _sha(stage / "training" / E.GUARD.EVIDENCE),
                  "owned_parent_wall_s": receipt["wall_s"]}
        E.GUARD.durable_new(stage / "accepted.json", record)
        records.append(record)
    failed = prior / remaining[0]["job_id"]
    failed.mkdir()
    E.GUARD.durable_new(failed / "gpu_before.json", {"busy": True,
                                                      "busy_reasons": ["unapproved_gpu_process:94284"]})
    E.GUARD.durable_new(prior / "intent.json", {
        "mode": "all", "anchor": E.SOURCE_ANCHOR,
        "protocol_sha256_lf": E.B.digest(repo / E.B.PROTOCOL, lf=True)})
    E.GUARD.durable_new(prior / "failure.json", {
        "status": "invalid", "completed_jobs": records, "error_type": "RuntimeError",
        "error": "GPU unapproved"})
    preflight = prior.parents[1] / "recovery-preflight-20261031-v1"
    preflight.mkdir()
    lease = {"state": "failed", "completedUtc": "2026-10-31T00:00:00Z"}
    E.GUARD.durable_new(preflight / "lease-completed.json", lease)
    preflight_value = {"prior_failure_sha256": _sha(prior / "failure.json"),
                       "prior_completed_jobs": 13, "lease_sha256": _sha(preflight / "lease-completed.json"),
                       "failed_stage_prelaunch_only": True, "policy_unchanged": True,
                       "all_clear": True, "new_gpu_jobs": 0, "api": 0}
    E.GUARD.durable_new(preflight / "result.json", preflight_value)
    monkeypatch.setattr(E, "PREFLIGHT_SHA256", _sha(preflight / "result.json"))
    protocol_value = E.prepare_protocol(prior, lab)
    protocol_path = repo / "bench/protocols/lineage_source_continuation_v1.json"
    protocol_path.write_text(json.dumps(protocol_value), encoding="utf-8")
    python = lab / "_isaac_sim/python.bat"
    python.parent.mkdir(exist_ok=True)
    python.write_text("fixture", encoding="utf-8")
    monkeypatch.setattr(E, "_verify_registration", lambda path, anchor, root, actual_lab: (
        protocol_value if Path(path) == protocol_path and anchor == "e" * 40
        and Path(root) == prior and Path(actual_lab) == lab
        else (_ for _ in ()).throw(ValueError("registration"))))
    return {"repo": repo, "lab": lab, "prior": prior, "value": value,
            "protocol": protocol_path, "protocol_value": protocol_value, "python": python,
            "out": tmp_path / "continuation", "bindings": bindings, "remaining": remaining}


def _tree_sha(root: Path) -> dict[str, str]:
    return {path.relative_to(root).as_posix(): _sha(path) for path in root.rglob("*") if path.is_file()}


def test_prepare_reuses_exact_13_prefix_and_registers_only_remaining_five(continuation: dict) -> None:
    value = E.prepare_protocol(continuation["prior"], continuation["lab"])
    assert value["prior_prefix_count"] == 13
    assert [row["job_id"] for row in value["remaining_jobs"]] == [
        "compound-s2308", "none-s2309", "f5-s2309", "none-s2310", "f3-s2310"]
    assert value["failed_prelaunch"]["infrastructure_prelaunch_only"] is True
    assert value["budget"]["automatic_retries"] == value["budget"]["model_api_requests"] == 0
    assert value["claims"]["single_uninterrupted_batch"] is False
    serialized = json.dumps(value)
    assert str(continuation["prior"]) not in serialized and str(continuation["lab"]) not in serialized
    assert "bench/private" not in serialized and "seed_guard.evidence" not in serialized
    assert set(value["seed_guard"]) == {"checked_seeds", "allowed_consumed", "required_unused",
                                        "evidence_count", "evidence_sha256"}
    assert value["seed_guard"]["evidence_count"] == 1
    assert value["seed_guard"]["evidence_sha256"] == E._canonical_digest([{
        "seed": 2308, "path": "fixture"}])
    assert all(set(row["checkpoint"]) == {"sha256", "size_bytes", "iteration"}
               for row in value["prior_jobs"])


def test_execute_runs_only_five_and_preserves_prior_tree(continuation: dict) -> None:
    before = _tree_sha(continuation["prior"])
    manifest = E.prepare_inputs(continuation["prior"], continuation["protocol"], "e" * 40,
                                continuation["out"], continuation["lab"], continuation["python"])
    calls = []

    def runner(stage, argv, lab, timeout):
        calls.append((stage.name, timeout))
        return _completion(stage, argv, lab, timeout)

    result = E.execute(continuation["out"] / "continuation-manifest.json", runner=runner,
                       check_lease=_lease, check_gpu=lambda stage: E.GUARD.durable_new(
                           stage / "gpu_before.json", {"busy": False, "busy_reasons": []}))
    assert [row[0] for row in calls] == [job["job_id"] for job in continuation["remaining"]]
    assert result["source_jobs_completed"] == 18 and len(result["records"]) == 18
    assert result["continuation"]["reused_prior_jobs"] == 13
    assert result["continuation"]["new_jobs"] == 5
    assert result["continuation"]["single_uninterrupted_batch"] is False
    assert _tree_sha(continuation["prior"]) == before
    assert manifest["source_anchor"] == E.SOURCE_ANCHOR
    assert manifest["seed_guard_evidence"] == [{"seed": 2308, "path": "fixture"}]


def test_prior_checkpoint_tamper_duplicate_or_failed_child_artifact_is_rejected(
        continuation: dict) -> None:
    first = continuation["prior"] / E._jobs()[0][0]["job_id"] / "logs/model_299.pt"
    first.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checkpoint|완료"):
        E.prepare_inputs(continuation["prior"], continuation["protocol"], "e" * 40,
                         continuation["out"], continuation["lab"], continuation["python"])


def test_duplicate_prefix_record_and_prelaunch_child_file_are_rejected(continuation: dict) -> None:
    failure_path = continuation["prior"] / "failure.json"
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    failure["completed_jobs"][1] = failure["completed_jobs"][0]
    failure_path.write_text(json.dumps(failure), encoding="utf-8")
    with pytest.raises(ValueError, match="prefix 순서"):
        E.prepare_protocol(continuation["prior"], continuation["lab"])


def test_prelaunch_stage_with_child_manifest_is_rejected(continuation: dict) -> None:
    failed = continuation["prior"] / "compound-s2308"
    (failed / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="gpu_before만|prelaunch"):
        E.prepare_protocol(continuation["prior"], continuation["lab"])


def test_existing_output_and_lease_change_block_launch(continuation: dict) -> None:
    continuation["out"].mkdir()
    with pytest.raises(FileExistsError, match="fresh"):
        E.prepare_inputs(continuation["prior"], continuation["protocol"], "e" * 40,
                         continuation["out"], continuation["lab"], continuation["python"])
    continuation["out"].rmdir()
    E.prepare_inputs(continuation["prior"], continuation["protocol"], "e" * 40,
                     continuation["out"], continuation["lab"], continuation["python"])
    checks, calls = 0, []

    def changed():
        nonlocal checks
        checks += 1
        return _lease("1" * 36 if checks == 1 else "2" * 36)

    with pytest.raises(ValueError, match="lease ID"):
        E.execute(continuation["out"] / "continuation-manifest.json",
                  runner=lambda *args: calls.append(args), check_lease=changed, check_gpu=None)
    assert calls == []


def test_powershell_preview_has_one_shared_lease_and_no_execution(continuation: dict) -> None:
    script = PROJECT / "bench/run_lineage_source_continuation_v1.ps1"
    command = ["powershell.exe", "-NoProfile", "-NonInteractive", "-File", str(script),
               "-ExecutionAnchor", "e" * 40, "-Protocol", str(continuation["protocol"]),
               "-PriorRoot", str(continuation["prior"]), "-Output", str(continuation["out"])]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    preview = json.loads(result.stdout)
    assert preview["mode"] == "preview" and preview["remainingJobs"] == 5
    assert preview["leaseOwner"] == E.B.OWNER and preview["retries"] == preview["API"] == 0
