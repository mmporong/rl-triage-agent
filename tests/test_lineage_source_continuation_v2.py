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
E = importlib.import_module("lineage_source_continuation_v2")


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
    fixed = json.loads(Path(argv[argv.index("--manifest") + 1]).read_text(encoding="utf-8"))
    output = Path(argv[argv.index("--out") + 1])
    E.GUARD.durable_new(stage / "launch.json", {"owner": "fixture", "argv": argv})
    log_dir = stage / "logs"; log_dir.mkdir()
    checkpoint = log_dir / "model_299.pt"; checkpoint.write_bytes(fixed["job_id"].encode())
    records = output / E.GUARD.RECORDS; records.parent.mkdir(parents=True)
    records.write_text("{}\n", encoding="utf-8")
    fingerprint = E.B.CHILD.manifest_fingerprint(fixed)
    evidence = {"schema": "lineage_source_training_evidence_v1", "manifest_sha256": fingerprint,
                "updates": 300, "update_count": 300, "arm": fixed["arm"], "seed": fixed["seed"],
                "learning_completed": True, "recorded_before_app_close": True, "learn_returned": True,
                "log_dir": str(log_dir), "checkpoint": {"path": str(checkpoint),
                                                            "sha256": _sha(checkpoint)}}
    E.GUARD.durable_new(output / E.GUARD.EVIDENCE, evidence)
    E.GUARD.durable_new(output / E.GUARD.CLOSE_READY, {
        "schema": E.GUARD.CLOSE_SCHEMA, "manifest_sha256": fingerprint, "updates": 300,
        "evidence_sha256_lf": _sha(output / E.GUARD.EVIDENCE, lf=True),
        "records_sha256_lf": _sha(records, lf=True)})
    return {"exit_code": 0, "natural_exit": True, "cleanup_verified": True,
            "job_assigned_before_release": True, "timed_out": False,
            "error_type": None, "wall_s": 0.25}


def _stage(root: Path, job: dict, source: dict, lab: Path) -> dict:
    stage = root / job["job_id"]; stage.mkdir()
    fixed = E.B.manifest(source, job, E.SOURCE_ANCHOR, purpose="source")
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
    return record


def _lease(identifier="11111111-1111-1111-1111-111111111111") -> dict:
    return {"owner": E.B.OWNER, "leaseId": identifier, "state": "running"}


@pytest.fixture
def cont(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    repo, lab = tmp_path / "repo", tmp_path / "lab"
    origin, first = tmp_path / "private/runs/sources-v2", tmp_path / "private/runs/sources-v3"
    repo.mkdir(); lab.mkdir(); origin.mkdir(parents=True); first.mkdir(parents=True)
    (lab / "logs/rsl_rl/unitree_go2_flat").mkdir(parents=True)
    source = _source_protocol()
    source_path = repo / E.B.PROTOCOL; source_path.parent.mkdir(parents=True)
    source_path.write_text(json.dumps(source), encoding="utf-8")
    for name in E.SOURCE_FILES:
        path = repo / name
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True); path.write_text(name, encoding="utf-8")
    v1_path = repo / E.V1_PROTOCOL; v1_path.parent.mkdir(parents=True, exist_ok=True)
    v1_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(E, "ROOT", repo); monkeypatch.setattr(E.B, "ROOT", repo)
    verify_calls = []

    def verified(actual_lab, anchor):
        verify_calls.append((Path(actual_lab), anchor))
        return source

    monkeypatch.setattr(E.B, "verify", verified)
    binding_calls = []
    monkeypatch.setattr(E.B.CHILD, "verify_binding",
                        lambda fixed, **_kwargs: binding_calls.append(fixed["job_id"]))
    monkeypatch.setattr(E, "_fresh_2310", lambda _lab: {
        "checked_seeds": [2310], "evidence_count": 0,
        "evidence_sha256": E._canonical_sha([]), "private_evidence": []})
    origin_jobs, continued_jobs, remaining = E._job_sets()
    origin_records = [_stage(origin, job, source, lab) for job in origin_jobs]
    failed0 = origin / continued_jobs[0]["job_id"]; failed0.mkdir()
    E.GUARD.durable_new(failed0 / "gpu_before.json", {"busy": True, "busy_reasons": ["pid"]})
    E.GUARD.durable_new(origin / "intent.json", {"anchor": E.SOURCE_ANCHOR,
                                                   "protocol_sha256_lf": E.B.digest(source_path, lf=True)})
    E.GUARD.durable_new(origin / "failure.json", {"status": "invalid",
                                                    "completed_jobs": origin_records})
    continued_records = [_stage(first, job, source, lab) for job in continued_jobs]
    failed1 = first / remaining[0]["job_id"]; failed1.mkdir()
    E.GUARD.durable_new(failed1 / "gpu_before.json", {"busy": True, "busy_reasons": ["photos"]})
    E.GUARD.durable_new(first / "intent.json", {"anchor": E.SOURCE_ANCHOR,
        "protocol_sha256_lf": E.B.digest(source_path, lf=True),
        "execution_anchor": E.V1_EXECUTION_ANCHOR,
        "continuation_protocol_sha256_lf": E.V1_PROTOCOL_SHA256_LF})
    E.GUARD.durable_new(first / "failure.json", {"status": "invalid", "automatic_retry": False,
                                                  "completed_new_jobs": continued_records})
    v1_manifest = {"schema": "lineage_source_continuation_manifest_v1",
                   "prior_root": str(origin.resolve()), "prior_failure_sha256": _sha(origin / "failure.json"),
                   "protocol_path": str(v1_path.resolve()), "protocol_sha256_lf": E.V1_PROTOCOL_SHA256_LF,
                   "source_anchor": E.SOURCE_ANCHOR, "execution_anchor": E.V1_EXECUTION_ANCHOR,
                   "seed_guard_evidence": [], "seed_guard_evidence_sha256": E._canonical_sha([]),
                   "lab": str(lab), "isaac_python": str(lab / "python.bat"),
                   "output_root": str(first.resolve())}
    E.GUARD.durable_new(first / "continuation-manifest.json", v1_manifest)
    monkeypatch.setattr(E, "_verify_v1_registration", lambda manifest, root: {
        "prior_failure_sha256": _sha(origin / "failure.json")})
    prior = E.collect_prior(origin, first, lab)
    protocol_value = E.protocol_from_prior(prior)
    protocol = repo / "bench/protocols/lineage_source_continuation_v2.json"
    protocol.write_text(json.dumps(protocol_value), encoding="utf-8")
    python = lab / "python.bat"; python.write_text("fixture", encoding="utf-8")
    monkeypatch.setattr(E, "_verify_registration", lambda path, anchor, value: (
        value if Path(path) == protocol and anchor == "e" * 40 else (_ for _ in ()).throw(ValueError())))
    verify_calls.clear()
    return {"repo": repo, "lab": lab, "origin": origin, "first": first, "source": source,
            "prior": prior, "protocol": protocol, "protocol_value": protocol_value,
            "python": python, "out": tmp_path / "v2", "remaining": remaining,
            "verify_calls": verify_calls, "binding_calls": binding_calls}


def _tree(root: Path) -> dict[str, str]:
    return {path.relative_to(root).as_posix(): _sha(path) for path in root.rglob("*") if path.is_file()}


def _wait(stage: Path, **kwargs) -> dict:
    assert kwargs == {"max_wait_s": 600, "consecutive_clear": 2}
    folder = stage / "gpu-wait"; folder.mkdir()
    receipt = folder / "result.json"
    E.GUARD.durable_new(receipt, {"status": "clear"})
    return {"status": "clear", "receipt_path": str(receipt), "receipt_sha256": _sha(receipt)}


def test_collect_reuses_exact_16_with_one_common_source_verify(cont: dict) -> None:
    prior = E.collect_prior(cont["origin"], cont["first"], cont["lab"])
    assert len(prior["records"]) == 16 and [row["job_id"] for row in prior["remaining_jobs"]] == [
        "none-s2310", "f3-s2310"]
    assert cont["verify_calls"] == [(cont["lab"], E.SOURCE_ANCHOR)]
    public = E.protocol_from_prior(prior)
    raw = json.dumps(public)
    assert str(cont["origin"]) not in raw and str(cont["first"]) not in raw and "bench/private" not in raw
    assert public["budget"]["controller_wall_s"] == 5400


def test_execute_runs_only_last_two_and_preserves_both_roots(cont: dict) -> None:
    before = (_tree(cont["origin"]), _tree(cont["first"]))
    E.prepare_inputs(cont["origin"], cont["first"], cont["protocol"], "e" * 40,
                     cont["out"], cont["lab"], cont["python"])
    calls = []

    def runner(stage, argv, lab, timeout):
        calls.append(stage.name); return _completion(stage, argv, lab, timeout)

    result = E.execute(cont["out"] / "continuation-manifest.json", runner=runner,
                       wait_gpu=_wait, check_lease=_lease)
    assert calls == ["none-s2310", "f3-s2310"]
    assert result["source_jobs_completed"] == 18 and len(result["records"]) == 18
    assert result["continuation"]["reused_origin_jobs"] == 13
    assert result["continuation"]["reused_v1_jobs"] == 3
    assert result["continuation"]["new_v2_jobs"] == 2
    assert cont["binding_calls"] == ["none-s2310", "f3-s2310"]
    assert (_tree(cont["origin"]), _tree(cont["first"])) == before


@pytest.mark.parametrize("target", ["origin", "continuation", "checkpoint"])
def test_two_root_tamper_and_missing_checkpoint_are_rejected(cont: dict, target: str) -> None:
    if target == "origin":
        path = cont["origin"] / "failure.json"; value = json.loads(path.read_text()); value["x"] = 1
        path.write_text(json.dumps(value))
    elif target == "continuation":
        path = cont["first"] / "failure.json"; value = json.loads(path.read_text())
        value["completed_new_jobs"].reverse(); path.write_text(json.dumps(value))
    else:
        checkpoint = Path(cont["prior"]["records"][0]["checkpoint"]["path"]); checkpoint.unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        E.collect_prior(cont["origin"], cont["first"], cont["lab"])


def test_existing_seventeenth_child_or_target_log_is_rejected(cont: dict) -> None:
    stage = cont["first"] / "f3-s2310"; stage.mkdir(); (stage / "manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="17/18번째"):
        E.collect_prior(cont["origin"], cont["first"], cont["lab"])
    stage.rmdir() if not any(stage.iterdir()) else (stage / "manifest.json").unlink()
    if stage.exists(): stage.rmdir()
    (cont["lab"] / "logs/rsl_rl/unitree_go2_flat/2026_lineage_v1_s2310_f3").mkdir()
    with pytest.raises(ValueError, match="run_name"):
        E.collect_prior(cont["origin"], cont["first"], cont["lab"])


def test_lease_change_blocks_runner_before_new_training(cont: dict) -> None:
    E.prepare_inputs(cont["origin"], cont["first"], cont["protocol"], "e" * 40,
                     cont["out"], cont["lab"], cont["python"])
    checks, calls = 0, []

    def changed():
        nonlocal checks
        checks += 1
        return _lease("1" * 36 if checks == 1 else "2" * 36)

    with pytest.raises(ValueError, match="lease ID"):
        E.execute(cont["out"] / "continuation-manifest.json",
                  runner=lambda *args: calls.append(args), wait_gpu=_wait, check_lease=changed)
    assert calls == []


def test_powershell_preview_is_two_jobs_and_one_shared_lease(cont: dict) -> None:
    script = PROJECT / "bench/run_lineage_source_continuation_v2.ps1"
    result = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-File", str(script),
                             "-ExecutionAnchor", "e" * 40, "-Protocol", str(cont["protocol"]),
                             "-OriginRoot", str(cont["origin"]), "-ContinuationRoot", str(cont["first"]),
                             "-Output", str(cont["out"])], check=True, capture_output=True, text=True)
    preview = json.loads(result.stdout)
    assert preview["reusedJobs"] == 16 and preview["remainingJobs"] == 2
    assert preview["parentWallSeconds"] == 5400 and preview["leaseOwner"] == E.B.OWNER


def test_budget_expired_during_gpu_wait_does_not_launch_sdk(cont: dict) -> None:
    E.prepare_inputs(cont["origin"], cont["first"], cont["protocol"], "e" * 40,
                     cont["out"], cont["lab"], cont["python"])
    now, calls = [0.0], []

    def expired_wait(stage, **kwargs):
        receipt = _wait(stage, **kwargs)
        now[0] = E.TOTAL_WALL_S
        return receipt

    with pytest.raises(TimeoutError, match="대기 중"):
        E.execute(cont["out"] / "continuation-manifest.json", runner=lambda *args: calls.append(args),
                  wait_gpu=expired_wait, check_lease=_lease, clock=lambda: now[0])
    assert calls == []
    assert not (cont["out"] / "none-s2310/manifest.json").exists()


def test_lease_change_after_wait_does_not_launch_sdk(cont: dict) -> None:
    E.prepare_inputs(cont["origin"], cont["first"], cont["protocol"], "e" * 40,
                     cont["out"], cont["lab"], cont["python"])
    checks, calls = 0, []

    def changing():
        nonlocal checks
        checks += 1
        return _lease() if checks < 3 else _lease("2" * 36)

    with pytest.raises(ValueError, match="대기 뒤 lease ID"):
        E.execute(cont["out"] / "continuation-manifest.json", runner=lambda *args: calls.append(args),
                  wait_gpu=_wait, check_lease=changing)
    assert calls == []
