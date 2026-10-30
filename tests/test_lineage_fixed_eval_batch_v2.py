from __future__ import annotations

import hashlib
import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT / "tests"), str(PROJECT / "bench"),
                str(PROJECT / "evals"), str(PROJECT / "src")]
V = importlib.import_module("lineage_fixed_eval_batch_v2")
V1_TEST = importlib.import_module("test_lineage_fixed_eval_batch_v1")
pytest_plugins = ("test_lineage_fixed_eval_batch_v1",)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Clock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value

    def sleep(self, seconds):
        self.value += seconds


def lease(identifier: str = "11111111-1111-1111-1111-111111111111") -> dict:
    return {"owner": V.V1.SOURCE.OWNER, "state": "running", "leaseId": identifier}


@pytest.fixture
def v2_env(env: dict, monkeypatch: pytest.MonkeyPatch) -> dict:
    repo = env["repo"]
    monkeypatch.setattr(V, "ROOT", repo)
    for name in V.SOURCE_FILES:
        path = repo / name
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(name, encoding="utf-8")
    protocol = repo / "bench/protocols/lineage_fixed_eval_v2.json"
    protocol.write_text(json.dumps(V.prepare_protocol(env["lab"])), encoding="utf-8")
    _git(repo, "add", *V.SOURCE_FILES, protocol.relative_to(repo).as_posix())
    _git(repo, "commit", "-m", "fixed eval v2")
    value = dict(env)
    value.update(protocol=protocol, anchor=_git(repo, "rev-parse", "HEAD"))
    return value


def clear_waiter(clock: Clock, calls: list):
    def wait(stage, **kwargs):
        calls.append({"stage": stage, **kwargs})
        clock.value += 100
        path = stage / "gpu-wait/result.json"
        path.parent.mkdir()
        path.write_text(json.dumps({"status": "clear"}), encoding="utf-8")
        return {"status": "clear", "receipt_path": str(path), "receipt_sha256": _sha(path)}
    return wait


def test_protocol_extends_v1_closure_without_private_paths_and_restores_globals(v2_env) -> None:
    env = v2_env
    before = (V.V1.SCHEMA, V.V1.SOURCE_FILES, V.V1.prepare_protocol)
    value = V.prepare_protocol(env["lab"])
    assert value["schema"] == V.SCHEMA
    assert set(V.BASE_SOURCE_FILES) < set(value["source_sha256_lf"])
    assert set(value["source_sha256_lf"]) == set(V.SOURCE_FILES)
    assert {"evals/lineage_fixed_eval_batch_v2.py", "tests/test_lineage_fixed_eval_batch_v2.py",
            "bench/run_lineage_fixed_eval_v2.ps1", "evals/lineage_gpu_wait_v1.py",
            "tests/test_lineage_gpu_wait_v1.py"} <= set(value["source_sha256_lf"])
    assert value["budget"]["whole_controller_wall_s"] == 3600
    assert value["budget"]["admission_wait_max_s"] == 600
    assert value["budget"]["gpu_budget_increase"] == 0
    assert value["gpu_wait"]["consecutive_clear"] == 2
    assert all(not Path(name).is_absolute() for name in value["source_sha256_lf"])
    assert V.verify(env["protocol"], lab=env["lab"], anchor=env["anchor"]) == value
    assert (V.V1.SCHEMA, V.V1.SOURCE_FILES, V.V1.prepare_protocol) == before


def test_reentrant_patch_and_exception_restore_v1_surface() -> None:
    source = PROJECT / "evals/lineage_fixed_eval_batch_v1.py"
    original = source.read_bytes()
    before = (V.V1.SCHEMA, V.V1.SOURCE_FILES, V.V1.prepare_protocol)
    with pytest.raises(RuntimeError), V._v1_surface(route_protocol=True):
        assert V.V1.SCHEMA == V.SCHEMA
        with V._v1_surface(route_protocol=False):
            assert V.V1.SOURCE_FILES == V.SOURCE_FILES
        raise RuntimeError("fixture")
    assert (V.V1.SCHEMA, V.V1.SOURCE_FILES, V.V1.prepare_protocol) == before
    assert source.read_bytes() == original


def test_prepare_inputs_and_actual_v1_worker_gate_flow_use_wait_and_reduced_timeout(
        v2_env, monkeypatch) -> None:
    env = v2_env
    manifest = V.prepare_inputs(env["source_root"], env["protocol"], env["anchor"], env["out"],
                                env["lab"], env["isaac_python"])
    assert manifest["worker_job"]["source_sha256_lf"] == V.prepare_protocol(env["lab"])["source_sha256_lf"]
    owned_calls = []
    base_runner = V1_TEST._real_worker_runner(env, monkeypatch, owned_calls)
    clock, wait_calls = Clock(), []
    result = V.execute(env["out"] / "manifest.json", runner=base_runner, check_lease=lease,
                       waiter=clear_waiter(clock, wait_calls), clock=clock, sleep=clock.sleep)
    assert result["status"] == "complete"
    assert result["base_result"]["gate"]["planned_cases"] == 10
    assert len(wait_calls) == 1 and wait_calls[0]["max_wait_s"] == 600
    assert wait_calls[0]["consecutive_clear"] == 2
    assert len(owned_calls) == 1 and owned_calls[0]["timeout"] == 3495
    assert result["owned_timeout_s"] == 3495
    assert result["whole_controller_wall_s"] == 100
    assert result["gpu_budget_increase"] == result["model_api_requests"] == 0
    assert (env["out"] / "gate.json").is_file()


def test_wait_lease_change_or_timeout_blocks_sdk_and_preserves_failure(v2_env) -> None:
    env = v2_env
    V.prepare_inputs(env["source_root"], env["protocol"], env["anchor"], env["out"],
                     env["lab"], env["isaac_python"])
    clock, checks, sdk_calls = Clock(), 0, []

    def changed():
        nonlocal checks
        checks += 1
        return lease() if checks < 3 else lease("22222222-2222-2222-2222-222222222222")

    def actual_wait(stage, **kwargs):
        return V.WAIT.await_clear(
            stage, max_wait_s=kwargs["max_wait_s"], interval_s=2, consecutive_clear=2,
            lease_check=kwargs["lease_check"], query=lambda: {
                "admission_policy": V.WAIT.GPU.GPU_PROFILE, "busy": False,
                "busy_reasons": [], "compute_process_count": 0},
            clock=kwargs["clock"], sleep=kwargs["sleep"])

    with pytest.raises(V.WAIT.GpuWaitError, match="lease"):
        V.execute(env["out"] / "manifest.json", runner=lambda *args: sdk_calls.append(args),
                  check_lease=changed, waiter=actual_wait, clock=clock, sleep=clock.sleep)
    assert sdk_calls == []
    failure = json.loads((env["out"] / "v2-failure.json").read_text(encoding="utf-8"))
    assert failure["sdk_retry"] is failure["automatic_retry"] is False
    assert (env["out"] / "failure.json").is_file()


def test_missing_expanded_source_and_wrong_registered_blob_are_rejected(v2_env) -> None:
    env = v2_env
    missing = env["repo"] / "evals/lineage_gpu_wait_v1.py"
    original = missing.read_bytes()
    missing.unlink()
    try:
        with pytest.raises(FileNotFoundError):
            V.prepare_protocol(env["lab"])
    finally:
        missing.write_bytes(original)
    changed = env["repo"] / "tests/test_lineage_gpu_wait_v1.py"
    changed.write_text("changed", encoding="utf-8")
    updated = V.prepare_protocol(env["lab"])
    env["protocol"].write_text(json.dumps(updated), encoding="utf-8")
    with pytest.raises(ValueError, match="anchor blob"):
        V.verify(env["protocol"], lab=env["lab"], anchor=env["anchor"])


def test_launcher_prepares_before_lease_and_executes_v2_inside_same_owner() -> None:
    text = (PROJECT / "bench/run_lineage_fixed_eval_v2.ps1").read_text(encoding="utf-8")
    assert "lineage_fixed_eval_batch_v2.py" in text
    assert text.index("& $python @prepareArgs") < text.index("& $lease -Owner")
    assert "'rl-triage-lineage-sources-v1'" in text
    assert "-WaitTimeoutSeconds 3600" in text
    assert "admissionWaitMaxSeconds=600" in text
