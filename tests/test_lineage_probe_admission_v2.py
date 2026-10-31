from __future__ import annotations

import hashlib
import importlib
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "bench"), str(ROOT / "src")]
A = importlib.import_module("lineage_probe_admission_v2")


class Clock:
    def __init__(self, jump=2):
        self.value = 0.0
        self.jump = jump

    def __call__(self):
        return self.value

    def sleep(self, _seconds):
        self.value += self.jump


def lease(identifier="11111111-1111-1111-1111-111111111111"):
    return {"owner": A.GPU_WAIT.B.OWNER, "state": "running", "leaseId": identifier}


def sample(*, busy=False):
    return {
        "admission_policy": A.GPU_WAIT.GPU.GPU_PROFILE,
        "busy": busy,
        "busy_reasons": (["unapproved_gpu_process:139312:svchost.exe"] if busy else []),
        "compute_process_count": 1 if busy else 0,
    }


def dependencies(values, lease_check=lease, *, jump=2):
    rows = iter(values)
    clock = Clock(jump)
    return patch.multiple(A, _LEASE_CHECK=lease_check, _GPU_QUERY=lambda: next(rows),
                          _CLOCK=clock, _SLEEP=clock.sleep)


def test_busy_clear_clear_runs_role_once_and_preserves_svchost_block(tmp_path):
    calls = []
    stage = tmp_path / "case"
    with dependencies([sample(busy=True), sample(), sample()]), \
            patch.object(A.P, "prepare_probe", side_effect=lambda *args, **kwargs: calls.append(kwargs["role"]) or {"ok": True}):
        assert A.prepare_probe(tmp_path / "input.json", "P_reward", stage=stage,
                               lab=tmp_path, protocol_path=tmp_path / "p.json",
                               anchor="a" * 40, role="case") == {"ok": True}
    assert calls == ["case"]
    wait = stage.parent / "case-admission-before-prepare/gpu-wait"
    assert len(list((wait / "samples").glob("*.json"))) == 3
    assert json.loads((wait / "samples/0001.json").read_text())["busy_reasons"] == [
        "unapproved_gpu_process:139312:svchost.exe"
    ]
    assert "svchost.exe" not in A.GPU_WAIT.GPU.policy()["allowed_display_process_names"]


def test_persistent_busy_stops_before_claim_owned_or_later_role(tmp_path):
    calls = []
    plan = tmp_path / "case/plan.json"
    plan.parent.mkdir()
    plan.write_text(json.dumps({"stage": str(plan.parent.resolve())}), encoding="utf-8")
    with dependencies([sample(busy=True)], jump=600), \
            patch.object(A.P, "execute_probe", side_effect=lambda *args, **kwargs: calls.append("claim")), \
            pytest.raises(A.ProbeAdmissionBlocked):
        A.execute_probe(plan, tmp_path / "authorization.json")
    assert calls == []
    failure_path = tmp_path / "case-admission-before-claim/probe-admission-failure.json"
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    assert failure["claim_created"] is False and failure["owned_processes_started"] == 0
    assert failure["post_block_roles_started"] == 0 and failure["phase"] == "execute"
    wait_path = Path(failure["wait_receipt"]["path"])
    assert failure["wait_receipt"]["sha256"] == hashlib.sha256(wait_path.read_bytes()).hexdigest()
    assert "GpuWaitError" in failure["traceback"]


@pytest.mark.parametrize("mode", ["malformed", "changed"])
def test_malformed_or_changed_lease_is_write_once_blocked_before_delegate(tmp_path, mode):
    if mode == "malformed":
        lease_check = lambda: {"owner": A.GPU_WAIT.B.OWNER, "state": "running", "leaseId": "bad"}
    else:
        leases = iter([lease(), lease("22222222-2222-2222-2222-222222222222")])
        lease_check = lambda: next(leases)
    delegated = []
    stage = tmp_path / "reference"
    with dependencies([sample()], lease_check=lease_check), \
            patch.object(A.P, "prepare_probe", side_effect=lambda *args, **kwargs: delegated.append(True)), \
            pytest.raises(A.ProbeAdmissionBlocked):
        A.prepare_probe(tmp_path / "input.json", "P_reward", stage=stage,
                        lab=tmp_path, protocol_path=tmp_path / "p.json",
                        anchor="a" * 40, role="reference")
    assert delegated == []
    root = tmp_path / "reference-admission-before-prepare"
    failure = json.loads((root / "probe-admission-failure.json").read_text(encoding="utf-8"))
    assert failure["error_type"] == "GpuWaitError"
    before = (root / "probe-admission-failure.json").read_bytes()
    with pytest.raises(FileExistsError):
        A._admit(root, "prepare")
    assert (root / "probe-admission-failure.json").read_bytes() == before


def test_probe_admission_blocked_escapes_exception_and_stops_second_role(tmp_path):
    roles = []

    def fake_epoch(*args, **kwargs):
        A.E.P.prepare_probe(tmp_path / "input", "P_reward", stage=tmp_path / "case",
                            lab=tmp_path, protocol_path=tmp_path / "base.json",
                            anchor="a" * 40, role="case")
        roles.append("reference")

    assert not issubclass(A.ProbeAdmissionBlocked, Exception)
    with dependencies([sample(busy=True)], jump=600), patch.object(A, "ROOT", tmp_path), \
            patch.object(A, "verify_protocol", return_value={
            "component_probe_protocol": {"path": "base.json"}
         }), patch.object(A.E, "run_epoch", side_effect=fake_epoch), \
            pytest.raises(A.ProbeAdmissionBlocked):
        A.run_epoch(wrapper_protocol=tmp_path / "wrapper.json", anchor="a" * 40,
                    probe_protocol=tmp_path / "base.json")
    assert roles == [] and A.E.P is A.P


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True,
                          text=True, encoding="utf-8").stdout.strip()


def registered_protocol(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    for relative in A.SOURCE_FILES:
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{relative}\n", encoding="utf-8")
    base = repo / "bench/protocols/base.json"
    base.parent.mkdir(parents=True, exist_ok=True)
    base.write_text(json.dumps({"schema": A.P.PROTOCOL_SCHEMA}), encoding="utf-8")
    _git(repo, "add", "--", *A.SOURCE_FILES, "bench/protocols/base.json")
    _git(repo, "-c", "user.name=fixture", "-c", "user.email=fixture@example.invalid",
         "commit", "-qm", "code")
    wrapper = repo / "bench/protocols/wrapper.json"
    with patch.object(A, "ROOT", repo):
        wrapper.write_text(json.dumps(A.prepare_protocol(base)), encoding="utf-8")
    _git(repo, "add", "--", "bench/protocols/wrapper.json")
    _git(repo, "-c", "user.name=fixture", "-c", "user.email=fixture@example.invalid",
         "commit", "-qm", "protocol")
    return repo, base, wrapper, _git(repo, "rev-parse", "HEAD")


def test_protocol_binds_adapter_wait_epoch_backend_and_anchor_blobs(tmp_path):
    repo, _base, wrapper, anchor = registered_protocol(tmp_path)
    with patch.object(A, "ROOT", repo):
        value = A.verify_protocol(wrapper, anchor=anchor)
    assert set(value["source_sha256_lf"]) == set(A.SOURCE_FILES)
    assert value["admission"] == A.GPU_WAIT.policy()
    assert value["claims"] == {"measurement": False, "correction": False, "recovery": False}


def test_protocol_rejects_changed_source_blob_and_component(tmp_path):
    repo, base, wrapper, anchor = registered_protocol(tmp_path)
    target = repo / A.SOURCE_FILES[0]
    target.write_text("changed\n", encoding="utf-8")
    with patch.object(A, "ROOT", repo), pytest.raises(ValueError, match="내용 또는 source|현재 source"):
        A.verify_protocol(wrapper, anchor=anchor)
    target.write_text(A.SOURCE_FILES[0] + "\n", encoding="utf-8")
    base.write_text(json.dumps({"schema": A.P.PROTOCOL_SCHEMA, "changed": True}), encoding="utf-8")
    with patch.object(A, "ROOT", repo), pytest.raises(ValueError, match="내용 또는 source"):
        A.verify_protocol(wrapper, anchor=anchor)


def test_adapter_scope_reentrant_and_always_restores():
    assert A.E.P is A.P
    with pytest.raises(RuntimeError, match="boom"), A.adapter_scope():
        assert A.E.P is A
        with A.adapter_scope():
            assert A.E.P is A
        raise RuntimeError("boom")
    assert A.E.P is A.P
