"""v5 계약 보존·완료 증거·1-update 선행 관문을 검사한다."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "bench")]
import ppo_gradient_check_v5 as Q
import ppo_gradient_training_v5 as T


def test_measurement_contract_is_preserved_except_observer_and_cohort():
    old, new = Q.V4.fixed_protocol(), Q.fixed_protocol()
    assert {key for key in old if old[key] != new[key]} == {"schema", "seeds", "runs", "supersedes", "dry_run"}
    assert set(new) - set(old) == {"observer_implementation"}
    assert {k:v for k,v in new["dry_run"].items() if k != "spec"} == {k:v for k,v in old["dry_run"].items() if k != "spec"}
    assert new["supersedes"]["outcome"] == "observer_cost_exceeded"
    assert new["supersedes"]["status"] == "complete"
    assert len(new["supersedes"]["update_wall_ratios"]) == 3
    assert set(Q.SEEDS).isdisjoint({*Q.V4.SEEDS, Q.dry_spec()["seed"]})
    assert len(Q.specs()) == 8 and Q.dry_spec()["updates"] == 1


def test_dry_manifest_and_args_have_one_update():
    protocol = Q.fixed_protocol() | {"source_sha256_lf": {"fixture.py": "a" * 64}}
    dry = Q.dry_spec()
    manifest = Q.manifest(dry, protocol, "b" * 40)
    assert manifest["max_iterations"] == manifest["runtime_updates_to_observe"] == 1
    arguments = Q.training_args(Path("isaac"), Path("output"), dry, "ppo_gradient_v5_20261017_dry")
    assert arguments[arguments.index("--max_iterations") + 1] == "1"
    assert arguments[arguments.index("--num_envs") + 1] == "4096"


def test_run_all_stops_before_batch_if_dry_fails(monkeypatch):
    calls = []
    def run(tag, isaaclab, *, dry_run=False):
        calls.append((tag, dry_run))
        raise ValueError("dry rejected")
    monkeypatch.setattr(Q, "run", run)
    with pytest.raises(ValueError, match="dry rejected"):
        Q.main(["run-all"])
    assert calls == [("ppo_gradient_v5_20261017_dry", True)]


def test_batch_requires_saved_dry_run_before_seed_consumption(tmp_path, monkeypatch):
    monkeypatch.setattr(Q, "PRIVATE", tmp_path)
    monkeypatch.setattr(Q, "verify", lambda *a, **k: ({}, "b" * 40))
    monkeypatch.setattr(Q, "lease_context", lambda: {"leaseId": "fixture"})
    def unexpected(*args, **kwargs):
        pytest.fail("dry 증거 없이는 seed 소비·GPU 실행 경로를 열면 안 된다")
    monkeypatch.setattr(Q.SEED, "assert_fresh_seeds", unexpected)
    monkeypatch.setattr(Q, "execute_owned", unexpected)
    with pytest.raises(FileNotFoundError):
        Q.run("ppo_gradient_v5_20261017", tmp_path)
    assert not (tmp_path / "consumed").exists()


@pytest.mark.parametrize("tag,dry_run", [("ppo_gradient_v5_20261017abcde", False),
    ("ppo_gradient_v5_20261017abcde_dry", True), ("ppo_gradient_v5_20261017", True)])
def test_invalid_tag_is_rejected_before_source_or_lease_work(tag, dry_run, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("tag 검사가 실행 계층보다 먼저여야 한다")
    monkeypatch.setattr(Q, "verify", unexpected)
    with pytest.raises(ValueError, match="tag"):
        Q.run(tag, Path("isaac"), dry_run=dry_run)


@pytest.fixture
def saved_dry(tmp_path, monkeypatch):
    monkeypatch.setattr(Q, "PRIVATE", tmp_path)
    anchor = "b" * 40
    monkeypatch.setattr(Q, "verify", lambda *args, **kwargs: ({}, anchor))
    folder = tmp_path / "dry_runs/ppo_gradient_v5_20261017_dry"
    arm = folder / Q.dry_spec()["id"]
    arm.mkdir(parents=True)
    execution = {"job_assigned_before_release": True, "natural_exit": True, "timed_out": False,
        "error_type": None, "exit_code": 0, "cleanup_verified": True, "owned_process_identities": {"1": 1}, "wall_s": 1.0}
    Q.write_new(arm / "execution.json", execution)
    row = {"id": Q.dry_spec()["id"], "inspection": {"status": "normal"}, "losses": [{}], "owned_process_wall_s": 1.0}
    monkeypatch.setattr(Q, "inspect_run", lambda *args: deepcopy(row))
    result = Q.summarize_dry([row]) | {"dry_run": True, "tag": folder.name,
        "protocol_sha256_lf": Q.sha(Q.PROTOCOL), "experiment_commit": anchor,
        "artifact_sha256_lf": {p.relative_to(folder).as_posix(): Q.sha(p) for p in folder.rglob("*") if p.is_file()}}
    Q.write_new(folder / "result.json", result)
    return folder


def test_saved_dry_requires_original_receipts_and_source_binding(saved_dry):
    assert Q.verify_dry_run(saved_dry.name, Path("isaac"))["tag"] == saved_dry.name
    receipt = saved_dry / Q.dry_spec()["id"] / "execution.json"
    receipt.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="산출물"):
        Q.verify_dry_run(saved_dry.name, Path("isaac"))


def test_saved_dry_rejects_spoofed_summary(saved_dry):
    path = saved_dry / "result.json"
    result = Q.read(path)
    result["outcome"] = "invalid"
    path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(ValueError, match="요약"):
        Q.verify_dry_run(saved_dry.name, Path("isaac"))


@pytest.mark.skipif(os.name != "nt", reason="실제 Windows owned child 회귀")
def test_exit_zero_without_completion_is_invalid_and_records_parent_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(Q, "PRIVATE", tmp_path / "private")
    monkeypatch.setattr(Q, "PROTOCOL", tmp_path / "protocol.json")
    monkeypatch.setattr(Q, "ROOT", tmp_path)
    protocol = Q.fixed_protocol() | {"source_sha256_lf": {"fixture.py": "a" * 64}}
    Q.write_new(Q.PROTOCOL, protocol)
    monkeypatch.setattr(Q, "verify", lambda *args, **kwargs: (protocol, "b" * 40))
    monkeypatch.setattr(Q, "lease_context", lambda: {"leaseId": "fixture"})
    monkeypatch.setattr(Q.SEED, "assert_fresh_seeds", lambda *args: {"status": "fixture"})
    monkeypatch.setattr(Q.OLD.GPU, "query_gpu", lambda *args: {"busy": False, "processes": []})
    monkeypatch.setattr(Q.OLD, "validate_gpu_snapshot", lambda value: value)
    monkeypatch.setattr(Q, "training_args", lambda *args: [sys.executable, "-c", "pass"])
    tag = "ppo_gradient_v5_20261017_dry"
    with pytest.raises(ValueError, match="marker"):
        Q.run(tag, tmp_path, dry_run=True)
    target = Q.PRIVATE / "dry_runs" / tag
    arm = target / Q.dry_spec()["id"]
    assert Q.read(arm / "execution.json")["exit_code"] == 0
    assert Q.read(arm / "training_failure.json")["stage"] == "parent_completion_check"
    assert Q.read(target / "result.json")["status"] == "invalid"
    assert not (Q.PRIVATE / "runs").exists()


@pytest.mark.skipif(os.name != "nt", reason="Isaac Windows Python CPU 회귀")
def test_actual_isaac_entrypoint_failure_has_nonzero_and_evidence(tmp_path, monkeypatch):
    launcher = Path.home() / "IsaacLab/_isaac_sim/python.bat"
    if not launcher.is_file():
        pytest.skip("Isaac Python launcher 없음")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    monkeypatch.setenv("PYTHONUTF8", "1")
    Q.write_new(tmp_path / "bad_manifest.json", {})
    args = [str(launcher), "-B", str(ROOT / "bench/ppo_gradient_training_v5.py"),
            "--manifest", str(tmp_path / "bad_manifest.json"), "--output", str(tmp_path)]
    with pytest.raises(RuntimeError):
        Q.execute_owned(tmp_path, args, tmp_path, 30)
    execution = Q.read(tmp_path / "execution.json")
    assert execution["exit_code"] != 0 and execution["cleanup_verified"]
    failure = Q.read(tmp_path / "training_failure.json")
    assert failure["stage"] == "training_entrypoint" and failure["traceback"]
