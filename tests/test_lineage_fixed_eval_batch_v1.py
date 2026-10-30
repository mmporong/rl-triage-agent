from __future__ import annotations

import copy
import hashlib
import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT / "bench"), str(PROJECT / "evals"), str(PROJECT / "src")]
E = importlib.import_module("lineage_fixed_eval_batch_v1")

SOURCE_PROTOCOL = PROJECT / "bench/protocols/lineage_source_collection_20261030_v2.json"
FIXED = PROJECT / "bench/protocols/fixed_eval_v1.json"
BANDS = PROJECT / "evals/results/p0c_fixed_eval_20261009/summary.json"
REAL_REPORT = PROJECT / "evals/results/brev_recovery_v4_s2196_20261024/source_fixed_none.json"


def _sha(path: Path, *, lf: bool = False) -> str:
    raw = path.read_bytes()
    if lf:
        raw = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(raw).hexdigest()


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    repo, lab, source_root = tmp_path / "repo", tmp_path / "lab", tmp_path / "lab/source"
    repo.mkdir(); lab.mkdir(); source_root.mkdir(parents=True)
    monkeypatch.setattr(E, "ROOT", repo)
    monkeypatch.setattr(E.SOURCE, "ROOT", repo)
    source_protocol = json.loads(SOURCE_PROTOCOL.read_text(encoding="utf-8"))
    runtime = {}
    for index, name in enumerate(source_protocol["runtime_source_sha256_lf"]):
        path = lab / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"runtime-{index}".encode())
        runtime[name] = _sha(path, lf=True)
    source_protocol["runtime_source_sha256_lf"] = runtime
    source_path = repo / E.SOURCE_PROTOCOL
    source_path.parent.mkdir(parents=True)
    source_path.write_text(json.dumps(source_protocol), encoding="utf-8")
    fixed = repo / E.FIXED_PROTOCOL
    fixed.parent.mkdir(parents=True, exist_ok=True)
    fixed.write_bytes(FIXED.read_bytes())
    bands = repo / E.BANDS
    bands.parent.mkdir(parents=True, exist_ok=True)
    bands.write_bytes(BANDS.read_bytes())
    _git(repo, "init")
    _git(repo, "config", "user.email", "fixture@example.invalid")
    _git(repo, "config", "user.name", "fixture")
    _git(repo, "add", E.SOURCE_PROTOCOL, E.FIXED_PROTOCOL, E.BANDS)
    _git(repo, "commit", "-m", "source collection")
    source_anchor = _git(repo, "rev-parse", "HEAD")
    for name in E.SOURCE_FILES:
        path = repo / name
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(name, encoding="utf-8")
    records = []
    for index, job in enumerate(source_protocol["training_jobs"]):
        folder = source_root / job["job_id"]
        folder.mkdir()
        checkpoint = folder / "model_299.pt"
        checkpoint.write_bytes(("duplicate" if index in {2, 3} else job["job_id"]).encode())
        records.append({"job_id": job["job_id"], "purpose": "source", "arm": job["arm"],
                        "seed": job["seed"], "checkpoint": {"path": str(checkpoint),
                                                               "sha256": _sha(checkpoint)}})
    result = {"status": "complete", "mode": "all", "source_jobs_completed": 18,
              "records": records}
    (source_root / "result.json").write_text(json.dumps(result), encoding="utf-8")
    (source_root / "intent.json").write_text(json.dumps({
        "anchor": source_anchor, "protocol_sha256_lf": _sha(source_path, lf=True),
    }), encoding="utf-8")
    isaac_python = lab / "_isaac_sim/python.bat"
    isaac_python.parent.mkdir(exist_ok=True)
    isaac_python.write_text("fixture", encoding="utf-8")
    protocol_path = repo / "bench/protocols/lineage_fixed_eval_v1.json"
    protocol_path.write_text(json.dumps(E.prepare_protocol(lab)), encoding="utf-8")
    _git(repo, "add", *E.SOURCE_FILES, protocol_path.relative_to(repo).as_posix())
    _git(repo, "commit", "-m", "eval controller")
    anchor = _git(repo, "rev-parse", "HEAD")

    def verify_source(_lab: Path, candidate: str) -> dict:
        if Path(_lab).resolve() != lab.resolve() or candidate != source_anchor:
            raise ValueError("source verify fixture binding mismatch")
        return source_protocol

    monkeypatch.setattr(E.SOURCE, "verify", verify_source)
    return {"repo": repo, "lab": lab, "source_root": source_root, "anchor": anchor,
            "source_anchor": source_anchor,
            "isaac_python": isaac_python, "protocol": protocol_path,
            "out": tmp_path / "evaluation", "source_protocol": source_protocol}


def _lease(identifier: str = "11111111-1111-1111-1111-111111111111") -> dict:
    return {"owner": E.SOURCE.OWNER, "leaseId": identifier, "state": "running"}


def _real_worker_runner(env: dict, monkeypatch: pytest.MonkeyPatch, calls: list) -> callable:
    template = json.loads(REAL_REPORT.read_text(encoding="utf-8"))

    def fake_checked(_argv):
        job = E.WORKER.validate_job(json.loads((env["out"] / "owned-evaluate/job.json").read_text()))
        jobs = json.loads(Path(job["jobs_path"]).read_text(encoding="utf-8"))["jobs"]
        for row in jobs:
            report = copy.deepcopy(template)
            report["name"] = row["name"]
            report["checkpoint"] = {"file": Path(row["checkpoint"]).name,
                                    "sha256": _sha(Path(row["checkpoint"]))}
            Path(row["output"]).write_text(json.dumps(report), encoding="utf-8")

    monkeypatch.setattr(E.WORKER, "_run_checked", fake_checked)

    def runner(stage, argv, cwd, timeout):
        calls.append({"stage": stage, "argv": argv, "cwd": cwd, "timeout": timeout})
        assert not (stage / "launch.json").exists()
        (stage / "launch.json").write_text(json.dumps({"args": argv, "cwd": str(cwd)}), encoding="utf-8")
        assert argv[0] == str(env["isaac_python"])
        assert Path(argv[argv.index("--job") + 1]) == stage / "job.json"
        job = E.WORKER.validate_job(json.loads((stage / "job.json").read_text(encoding="utf-8")))
        E.WORKER.run_evaluate(job, Path(argv[argv.index("--output") + 1]))
        return {"exit_code": 0, "natural_exit": True, "cleanup_verified": True,
                "job_assigned_before_release": True, "timed_out": False,
                "error_type": None, "wall_s": 12.5}

    return runner


def test_prepare_protocol_is_public_eval_only_and_reuses_pins(env: dict) -> None:
    value = E.prepare_protocol(env["lab"])
    assert value["phase"] == "eval_only" and value["task"] == E.GATE.TASK
    assert value["budget"] == {"owned_workers": 1, "evaluation_jobs": 18,
                                "training_updates": 0, "model_api_requests": 0,
                                "automatic_retries": 0, "whole_controller_wall_s": 3600,
                                "basis": "기존 동일 fixed_eval 보고서 약 13초×18보다 큰 단일 owned 상한"}
    assert value["fixed_eval"]["conditions"] == 26
    assert value["source_next_stage_was_registered"] is False
    assert all(not Path(text).is_absolute() for text in (
        value["source_collection_protocol"]["path"], value["fixed_eval"]["path"], value["bands"]["path"]))
    assert value["source_collection_protocol"]["source_tree_sha256"] == E._tree(
        env["source_protocol"]["source_sha256_lf"])
    for name in ("tests/test_lineage_evaluation_gate_v1.py", E.FIXED_PROTOCOL, E.BANDS):
        assert value["source_sha256_lf"][name] == _sha(env["repo"] / name, lf=True)
    assert E.verify(env["protocol"], lab=env["lab"], anchor=env["anchor"]) == value


def test_prepare_inputs_builds_real_v2_evaluate_job_and_rejects_reuse(env: dict) -> None:
    manifest = E.prepare_inputs(env["source_root"], env["protocol"], env["anchor"], env["out"],
                                env["lab"], env["isaac_python"])
    job = E.WORKER.validate_job(manifest["worker_job"])
    assert job["mode"] == "evaluate" and len(job["artifacts"]) == 18
    assert set(job["python_package_source_sha256_lf"]) == set(E.RSL_PACKAGE_FILES)
    assert job["source_commit_sha"] == env["anchor"]
    assert manifest["eval_execution_anchor"] == env["anchor"]
    assert manifest["source_collection_anchor"] == env["source_anchor"]
    assert manifest["training_seed_scope"].endswith("eval_seed 2026")
    with pytest.raises(FileExistsError, match="fresh"):
        E.prepare_inputs(env["source_root"], env["protocol"], env["anchor"], env["out"],
                         env["lab"], env["isaac_python"])


def test_rejects_source_collection_anchor_from_another_git_branch(
        env: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    unrelated = _git(env["repo"], "commit-tree", "HEAD^{tree}", "-m", "unrelated source")
    intent_path = env["source_root"] / "intent.json"
    intent = json.loads(intent_path.read_text(encoding="utf-8"))
    intent["anchor"] = unrelated
    intent_path.write_text(json.dumps(intent), encoding="utf-8")
    monkeypatch.setattr(E.SOURCE, "verify", lambda _lab, _anchor: env["source_protocol"])
    with pytest.raises(ValueError, match="Git 결합"):
        E.prepare_inputs(env["source_root"], env["protocol"], env["anchor"], env["out"],
                         env["lab"], env["isaac_python"])


def test_verify_rejects_uncommitted_protocol_or_registered_source_blob(env: dict) -> None:
    value = json.loads(env["protocol"].read_text(encoding="utf-8"))
    env["protocol"].write_text(json.dumps(value, indent=2), encoding="utf-8")
    with pytest.raises(ValueError, match="anchor blob"):
        E.verify(env["protocol"], lab=env["lab"], anchor=env["anchor"])

    env["protocol"].write_bytes(
        subprocess.run(["git", "-C", str(env["repo"]), "show",
                        f"{env['anchor']}:bench/protocols/lineage_fixed_eval_v1.json"],
                       check=True, capture_output=True).stdout)
    changed = env["repo"] / "evals/lineage_evaluation_gate_v1.py"
    changed.write_text("changed", encoding="utf-8")
    env["protocol"].write_text(json.dumps(E.prepare_protocol(env["lab"])), encoding="utf-8")
    with pytest.raises(ValueError, match="lineage_evaluation_gate_v1.py"):
        E.verify(env["protocol"], lab=env["lab"], anchor=env["anchor"])


def test_execute_runs_one_owned_v2_flow_and_passes_actual_reports_to_gate(
        env: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = E.prepare_inputs(env["source_root"], env["protocol"], env["anchor"], env["out"],
                                env["lab"], env["isaac_python"])
    calls = []
    runner = _real_worker_runner(env, monkeypatch, calls)
    gpu_calls = []
    result = E.execute(env["out"] / "manifest.json", runner=runner, check_lease=_lease,
                       check_gpu=lambda stage: gpu_calls.append(stage))
    assert result["status"] == "complete" and result["evaluation_jobs"] == 18
    assert result["owned_workers"] == 1 and result["training_updates"] == 0
    assert len(calls) == 1 and calls[0]["timeout"] == 3600
    assert gpu_calls == [env["out"] / "owned-evaluate"]
    assert result["gate"]["planned_cases"] == 10
    index = json.loads((env["out"] / "report-index.json").read_text(encoding="utf-8"))
    assert len(index) == 18 and all(set(row) == {
        "report_path", "report_sha256", "checkpoint_sha256"} for row in index.values())
    assert manifest["source_artifact_count"] == 18


def test_false_owned_completion_and_missing_worker_receipt_are_rejected(env: dict) -> None:
    E.prepare_inputs(env["source_root"], env["protocol"], env["anchor"], env["out"],
                     env["lab"], env["isaac_python"])

    def false_runner(*_args):
        return {"exit_code": 0, "natural_exit": True, "cleanup_verified": True,
                "job_assigned_before_release": True, "timed_out": False,
                "error_type": None, "wall_s": 0.1}

    with pytest.raises((FileNotFoundError, ValueError)):
        E.execute(env["out"] / "manifest.json", runner=false_runner,
                  check_lease=_lease, check_gpu=None)
    assert (env["out"] / "failure.json").is_file()


def test_checkpoint_byte_mutation_is_rejected_before_runner(env: dict) -> None:
    manifest = E.prepare_inputs(env["source_root"], env["protocol"], env["anchor"], env["out"],
                                env["lab"], env["isaac_python"])
    checkpoint = Path(next(iter(manifest["worker_job"]["artifacts"].values()))["checkpoint"]["path"])
    checkpoint.write_bytes(b"mutated")
    calls = []
    with pytest.raises(ValueError, match="checkpoint"):
        E.execute(env["out"] / "manifest.json", runner=lambda *args: calls.append(args),
                  check_lease=_lease, check_gpu=None)
    assert calls == []


def test_source_result_and_protocol_tamper_are_rejected(env: dict) -> None:
    result_path = env["source_root"] / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["records"].pop()
    result_path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(ValueError, match="18개"):
        E.prepare_inputs(env["source_root"], env["protocol"], env["anchor"], env["out"],
                         env["lab"], env["isaac_python"])
    protocol = json.loads(env["protocol"].read_text(encoding="utf-8"))
    protocol["budget"]["evaluation_jobs"] = 17
    env["protocol"].write_text(json.dumps(protocol), encoding="utf-8")
    with pytest.raises(ValueError, match="프로토콜"):
        E.verify(env["protocol"], lab=env["lab"], anchor=env["anchor"])


def test_lease_change_blocks_owned_launch(env: dict) -> None:
    E.prepare_inputs(env["source_root"], env["protocol"], env["anchor"], env["out"],
                     env["lab"], env["isaac_python"])
    checks = 0
    calls = []

    def changed():
        nonlocal checks
        checks += 1
        return _lease("1" * 36 if checks == 1 else "2" * 36)

    with pytest.raises(ValueError, match="lease ID"):
        E.execute(env["out"] / "manifest.json", runner=lambda *args: calls.append(args),
                  check_lease=changed, check_gpu=None)
    assert calls == []


def test_completed_controller_cannot_be_executed_again(env: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    E.prepare_inputs(env["source_root"], env["protocol"], env["anchor"], env["out"],
                     env["lab"], env["isaac_python"])
    runner = _real_worker_runner(env, monkeypatch, [])
    E.execute(env["out"] / "manifest.json", runner=runner, check_lease=_lease, check_gpu=None)
    with pytest.raises(FileExistsError, match="재사용"):
        E.execute(env["out"] / "manifest.json", runner=runner, check_lease=_lease, check_gpu=None)
