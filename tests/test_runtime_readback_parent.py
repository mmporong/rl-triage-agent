import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import subprocess

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "bench"))
SPEC = importlib.util.spec_from_file_location("readback_parent_test", ROOT / "bench/run_runtime_readback_smoke.py")
assert SPEC and SPEC.loader
P = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(P)


def fixture(tmp_path, monkeypatch):
    config = tmp_path / "configuration.yaml"
    config.write_text("seed: 20261011\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"schema":"test"}\n', encoding="utf-8")
    proto = {"scope": "actual_sdk_api_integration_not_diagnosis_or_recovery",
             "checkpoint_sha256": "a" * 64, "registry_sha256": "b" * 64,
             "isaacsim_version": "4.5.0",
             "source_files": {"src/rl_triage/runtime_observers.py": "c" * 64}}
    validated = {"protocol": proto, "protocol_sha256": "d" * 64,
                 "paths": {"configuration_path": config}, "manifest_path": manifest,
                 "source_receipt": {"task_binding_sha256": "e" * 64}}
    verdict = {"smoke_status": "pass", "validation_reasons": []}
    monkeypatch.setattr(P.S, "assess_smoke", lambda *_: verdict)
    config_sha = P.S._sha256_file(config)
    body = {"schema": "runtime_readback_smoke_body_v1", "completed": True,
            "scope": proto["scope"], "task": P.S.TASK,
            "protocol_sha256": validated["protocol_sha256"],
            "manifest_sha256": P.S._sha256_file(manifest), "project_commit_sha": "f" * 40,
            "cleanup_confirmed": False, "cleanup_confirmation_required_from_parent_receipt": True,
            "process_id": 321, "process_creation_time_filetime": 1234567,
            "source_receipt": validated["source_receipt"],
            "checkpoint": {"sha256": proto["checkpoint_sha256"]},
            "configuration": {"sha256": config_sha},
            "sdk_versions": {"isaaclab_commit": P.S.ISAACLAB_COMMIT, "isaacsim": "4.5.0"},
            "collector_report": {"measurement_status": "complete", "source_binding": {
                "task_sha256": "e" * 64, "checkpoint_sha256": "a" * 64,
                "configuration_sha256": config_sha, "observer_sha256": "c" * 64,
                "registry_sha256": "b" * 64}},
            "sdk_clock_receipt": {}, "smoke_validation": verdict,
            "smoke_status": "pass", "measurement_status": "complete"}
    execution = {"owned_process_identities": {"321": 1234567}}
    return body, validated, execution


def test_parent_rejects_pid_reuse_and_unobserved_process(tmp_path, monkeypatch):
    body, validated, execution = fixture(tmp_path, monkeypatch)
    assert P.verify_body(body, validated, execution, "f" * 40) == []
    body["process_creation_time_filetime"] += 1
    assert "body_pid_creation_not_observed_in_owned_job" in P.verify_body(body, validated, execution, "f" * 40)
    body["process_id"] = 999
    assert "body_pid_creation_not_observed_in_owned_job" in P.verify_body(body, validated, execution, "f" * 40)


@pytest.mark.parametrize("field", ["manifest_sha256", "project_commit_sha", "protocol_sha256", "schema"])
def test_parent_rejects_unbound_body(tmp_path, monkeypatch, field):
    body, validated, execution = fixture(tmp_path, monkeypatch)
    body[field] = "mismatch"
    assert f"body_{field}_mismatch" in P.verify_body(body, validated, execution, "f" * 40)


def test_parent_rejects_configuration_changed_or_absent(tmp_path, monkeypatch):
    body, validated, execution = fixture(tmp_path, monkeypatch)
    config = validated["paths"]["configuration_path"]
    config.write_text("seed: 1\n", encoding="utf-8")
    reasons = P.verify_body(body, validated, execution, "f" * 40)
    assert "body_configuration_sha_mismatch" in reasons
    assert "collector_source_binding_mismatch" in reasons
    config.unlink()
    with pytest.raises(FileNotFoundError):
        P.verify_body(body, validated, execution, "f" * 40)


def test_parent_rejects_checkpoint_source_and_masked_verdict_mismatch(tmp_path, monkeypatch):
    body, validated, execution = fixture(tmp_path, monkeypatch)
    body["checkpoint"]["sha256"] = "0" * 64
    body["source_receipt"] = copy.deepcopy(body["source_receipt"])
    body["source_receipt"]["task_binding_sha256"] = "1" * 64
    body["smoke_validation"] = {"smoke_status": "invalid"}
    reasons = P.verify_body(body, validated, execution, "f" * 40)
    assert "body_checkpoint_sha_mismatch" in reasons
    assert "body_source_receipt_mismatch" in reasons
    assert "measurement_or_clock_validation_failed" in reasons


def test_committed_gate_rejects_ignored_or_untracked_files(tmp_path):
    import subprocess
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    (tmp_path / "untracked.py").write_text("pass\n", encoding="utf-8")
    with pytest.raises(subprocess.CalledProcessError):
        P.verify_committed(tmp_path, {"untracked.py"})


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object integration")
def test_owned_cpu_worker_exits_and_job_is_empty(tmp_path):
    # This process test invokes no SDK, model or GPU job.
    execution = P.run_owned(tmp_path, [sys.executable, "-c", "import time; time.sleep(0.2)"], tmp_path, 15)
    assert execution["job_assigned_before_release"]
    assert execution["natural_exit"]
    assert execution["exit_code"] == 0
    assert execution["cleanup_verified"]
    assert execution["owned_process_identities"]


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object integration")
def test_owned_cpu_watchdog_cleans_only_its_job(tmp_path):
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"],
                             creationflags=P.M.CREATE_NO_WINDOW)
    try:
        execution = P.run_owned(tmp_path, [sys.executable, "-c", "import time; time.sleep(30)"], tmp_path, 0.5)
        assert execution["job_assigned_before_release"]
        assert execution["timed_out"]
        assert not execution["natural_exit"]
        assert execution["cleanup_verified"]
        assert execution["exit_code"] != 0
        assert other.poll() is None
        assert str(other.pid) not in execution["owned_process_identities"]
    finally:
        other.terminate()
        other.wait(timeout=10)
