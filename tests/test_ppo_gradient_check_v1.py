"""계산 검사 배치의 출처·비간섭·비용·lease 관문을 CPU에서 확인한다."""
from __future__ import annotations

import ast
import ctypes
from ctypes import wintypes
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "bench")]
import ppo_gradient_check_v1 as Q
import ppo_gradient_training_v1 as T
import ppo_ratio_mutation_v1 as M


SOURCE = """def update(self):
    before = 1
    ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
    return ratio + before
"""


def state():
    return {key: "a" * 64 for key in (
        "policy_sha256", "optimizer_sha256", "cpu_rng_sha256", "cuda_rng_sha256")} | {"learning_rate": .001}


def losses():
    return [{"value_function": .1, "surrogate": -.2, "entropy": .3} for _ in range(3)]


def batch_rows():
    rows = []
    for spec in Q.specs():
        status = "abnormal" if spec["arm"] == "INVERSE_RATIO" else "normal"
        rows.append(spec | {"state": state(), "losses": losses(), "update_wall_s": [1., 1., 1.],
            "inspection": {"status": status, "primitives": {
                "optimizer_step_count": {"status": "normal"}, "minibatch_count": {"status": "normal"},
                Q.G.PRIMITIVE: {"status": status}}} if spec["observed"] else None})
    return rows


def test_mutation_changes_only_one_registered_ratio_assignment():
    tree, evidence = M.mutation_tree(SOURCE)
    expected = ast.parse(SOURCE.replace(
        "actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch)",
        "torch.squeeze(old_actions_log_prob_batch) - actions_log_prob_batch"))
    assert M.ast_sha(tree) == M.ast_sha(expected)
    assert evidence["changed_ratio_assignments"] == 1
    assert evidence["original_ast_sha256"] != evidence["mutated_ast_sha256"]
    for bad in (SOURCE.replace("torch.exp", "torch.log"), SOURCE.replace("ratio =", "other ="),
                SOURCE.replace("return ratio + before", "ratio = torch.exp(0)\n    return ratio")):
        with pytest.raises(ValueError):
            M.mutation_tree(bad)


def test_ast_binding_ignores_empty_python_version_specific_type_parameters():
    tree = ast.parse(SOURCE)
    expected = M.ast_sha(tree)
    function = tree.body[0]
    if hasattr(function, "type_params"):
        delattr(function, "type_params")
    else:
        setattr(function, "type_params", [])
    assert M.ast_sha(tree) == expected


def test_windows_exclusive_lock_requires_sharing_violation(tmp_path):
    path = tmp_path / "lock"
    path.write_bytes(b"")
    with pytest.raises(ValueError):
        Q.assert_exclusive_lock(path)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
        wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.CreateFileW(str(path), 0xC0000000, 0, None, 3, 0, None)
    assert handle != ctypes.c_void_p(-1).value
    try:
        Q.assert_exclusive_lock(path)
    finally:
        kernel.CloseHandle(handle)
    with pytest.raises(OSError):
        Q.assert_exclusive_lock(tmp_path / "absent")


class FakePpo:
    def update(self):
        return "original"


def test_restoration_installs_then_restores_before_learning(monkeypatch):
    mutant = lambda self: "mutant"
    _, evidence = M.mutation_tree(SOURCE)
    monkeypatch.setattr(M, "mutated_update", lambda _: (mutant, evidence))
    ppo = FakePpo()
    cleanup, readback = M.install(ppo, "INVERSE_RATIO")
    assert ppo.update() == "mutant"
    cleanup()
    assert ppo.update() == "original"
    cleanup, readback = M.install(ppo, "RESTORED")
    assert readback["restoration_before_training"] and readback["restored_original"]
    assert ppo.update.__func__ is FakePpo.update
    cleanup()
    assert "update" not in ppo.__dict__


@pytest.mark.parametrize("status", ["unknown", "abnormal"])
def test_normal_gate_rejects_uncertain_or_abnormal_baseline(status):
    rows = batch_rows()[:6]
    next(row for row in rows if row["observed"])["inspection"]["status"] = status
    result = Q.summarize(rows)
    assert not result["checks"]["normal_3_seeds"]
    assert result["outcome"] == "normal_baseline_unconfirmed"


def test_summary_requires_exact_inventory_and_all_registration_conditions():
    rows = batch_rows()
    assert Q.summarize(rows)["outcome"] == "confirmed"
    for bad in (rows[:5], rows[:7], rows[::-1], rows[:7] + [rows[0]]):
        with pytest.raises(ValueError):
            Q.summarize(bad)
    rows[1]["state"]["policy_sha256"] = "b" * 64
    assert Q.summarize(rows[:6])["outcome"] == "observer_interference"
    rows = batch_rows()
    rows[1]["update_wall_s"] = [1.51] * 3
    assert Q.summarize(rows[:6])["outcome"] == "observer_cost_exceeded"
    rows = batch_rows()
    rows[-1]["state"]["policy_sha256"] = "b" * 64
    assert Q.summarize(rows)["outcome"] == "not_confirmed"


@pytest.fixture
def bound(tmp_path, monkeypatch):
    monkeypatch.setattr(Q, "ROOT", tmp_path)
    monkeypatch.setattr(Q, "PRIVATE", tmp_path / "private")
    monkeypatch.setattr(Q, "PROTOCOL", tmp_path / "bench/protocols/fixture.json")
    protocol = Q.fixed_protocol() | {
        "source_sha256_lf": {"evals/ppo_gradient_invariant_v1.py": "c" * 64},
        "mutation_readback": M.mutation_tree(SOURCE)[1]}
    Q.write_new(Q.PROTOCOL, protocol)
    return protocol, "a" * 40


def make_run(folder, spec, protocol, anchor):
    fixed = Q.manifest(spec, protocol, anchor)
    Q.write_new(folder / "manifest.json", fixed)
    evidence = {"schema": "ppo_gradient_training_evidence_v1", "updates": 3,
        "manifest_sha256": Q.O.manifest_fingerprint(fixed), "runtime_configuration": {
            "epochs": 5, "minibatches": 4, "clip_param": .2}, "method_cleanup": True,
        "arm": spec["arm"], "observed": spec["observed"], "torch_version": "2.7.0+cu128",
        "device_type": "cuda", "update_wall_s": [1.] * 3, "state": state(), "losses": losses(),
        "readback": protocol["mutation_readback"] | {"arm": spec["arm"], "active_mutation": False,
            "restored_original": False, "restoration_before_training": False,
            "installed_ast_sha256": protocol["mutation_readback"]["original_ast_sha256"]}}
    Q.write_new(folder / "training_evidence.json", evidence)
    records = []
    if spec["observed"]:
        for index in range(3):
            records.append({"schema": Q.O.SCHEMA, "manifest_sha256": Q.O.manifest_fingerprint(fixed),
                "binding": Q.O.manifest_binding(fixed), "update_index": index,
                "expected": {"optimizer_steps": 20, "minibatches": 20}, "error": None,
                "observed": {"optimizer_steps": 20, "minibatches": 20, Q.G.PRIMITIVE: {
                    "status": "normal", "batches": [{"max_abs_error": 0., "mismatch_count": 0,
                        "nonzero_expected_count": 1, "finite": True, "n": 4}] * 20}}})
    (folder / "records.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    return evidence


def test_inspection_recomputes_records_and_rejects_evidence_substitution(bound, tmp_path):
    protocol, anchor = bound
    spec = Q.specs()[1]
    folder = tmp_path / "arm"
    evidence = make_run(folder, spec, protocol, anchor)
    assert Q.inspect_run(folder, spec, protocol, anchor)["inspection"]["status"] == "normal"
    mutations = [lambda x: x.update(updates=True), lambda x: x.update(method_cleanup=False),
        lambda x: x.update(device_type="cpu"), lambda x: x.update(update_wall_s=[float("nan")] * 3),
        lambda x: x["readback"].update(active_mutation=True),
        lambda x: x["state"].update(policy_sha256="missing"), lambda x: x.update(losses=[]),
        lambda x: x["losses"][0].update(surrogate=float("nan"))]
    for mutate in mutations:
        bad = deepcopy(evidence)
        mutate(bad)
        (folder / "training_evidence.json").write_text(json.dumps(bad), encoding="utf-8")
        with pytest.raises(ValueError):
            Q.inspect_run(folder, spec, protocol, anchor)


def test_lease_requires_current_parent_chain_active_owner_and_fresh_heartbeat():
    now = datetime.now(timezone.utc)
    meta = {"owner": Q.LEASE_OWNER, "state": "running", "childPid": 2, "wrapperPid": 1,
        "completedUtc": None, "heartbeatUtc": now.isoformat(), "acquiredUtc": now.isoformat(),
        "leaseId": "a" * 8 + "-" + "b" * 4 + "-" + "c" * 4 + "-" + "d" * 4 + "-" + "e" * 12}
    ancestors = [{"pid": 3, "parent": 2, "name": "python.exe"},
        {"pid": 2, "parent": 1, "name": "python.exe"}, {"pid": 1, "parent": 0, "name": "pwsh.exe"}]
    assert Q.validate_lease(meta, ancestors, 2, now)["childPid"] == 2
    for changes in ({"owner": "other"}, {"childPid": 4}, {"state": "completed"},
                    {"wrapperPid": 4}, {"heartbeatUtc": "2020-01-01T00:00:00+00:00"}):
        with pytest.raises(ValueError):
            Q.validate_lease(meta | changes, ancestors, 2, now)
    with pytest.raises(ValueError):
        Q.validate_lease(meta, ancestors[:1], 2, now)


def test_lease_context_accepts_venv_launcher_ancestor_but_rejects_sibling(monkeypatch):
    now = datetime.now(timezone.utc).isoformat()
    meta = {"owner": Q.LEASE_OWNER, "state": "running", "childPid": 2, "wrapperPid": 1,
        "completedUtc": None, "heartbeatUtc": now, "acquiredUtc": now, "leaseId": "a" * 36}
    chain = [{"pid": 3, "parent": 2, "name": "python.exe"},
        {"pid": 2, "parent": 1, "name": "python.exe"}, {"pid": 1, "parent": 0, "name": "pwsh.exe"}]
    monkeypatch.setattr(Q.os, "getpid", lambda: 3)
    monkeypatch.setattr(Q, "read", lambda _: meta)
    monkeypatch.setattr(Q.subprocess, "run", lambda *_, **__: SimpleNamespace(stdout=json.dumps(chain)))
    monkeypatch.setattr(Q, "assert_exclusive_lock", lambda _: None)
    receipt = Q.lease_context()
    assert receipt["childPid"] == 2 and receipt["callerPid"] == 3
    chain[:] = [{"pid": 3, "parent": 4, "name": "python.exe"},
        {"pid": 4, "parent": 0, "name": "pwsh.exe"}]
    with pytest.raises(ValueError):
        Q.lease_context()


def test_training_checks_bound_arguments_and_lease_before_initialization(bound, monkeypatch):
    protocol, anchor = bound
    monkeypatch.setattr(Q, "verify", lambda *_: bound)
    spec = Q.specs()[0]
    tag = "ppo_gradient_v1_20261017"
    folder = Q.PRIVATE / "runs" / tag / spec["id"]
    Q.write_new(folder.parent / "plan.json", {"lease": {"childPid": 2, "leaseId": "fixture"}})
    monkeypatch.setattr(Q, "lease_context", lambda *_: {"leaseId": "fixture"})
    fixed = Q.manifest(spec, protocol, anchor)
    args = Q.training_args(Path("runtime"), folder, spec, tag)
    args = args[args.index("--") + 1:]
    assert T.verify_training(fixed, folder, args, Path("runtime")) == spec
    for bad in (args + ["--max_iterations", "300"], args[:-1] + ["different"]):
        with pytest.raises(ValueError):
            T.verify_training(fixed, folder, bad, Path("runtime"))
    monkeypatch.setattr(Q, "lease_context", lambda *_: {"leaseId": "other"})
    with pytest.raises(ValueError):
        T.verify_training(fixed, folder, args, Path("runtime"))


@pytest.mark.parametrize("failure", ["baseline", "cost", "interference", "execution"])
def test_batch_stops_before_fault_or_next_run_when_gate_fails(bound, monkeypatch, failure):
    protocol, anchor = bound
    monkeypatch.setattr(Q, "verify", lambda *_: bound)
    monkeypatch.setattr(Q, "lease_context", lambda *_: {"leaseId": "fixture", "childPid": 2})
    monkeypatch.setattr(Q.SEED, "assert_fresh_seeds", lambda *_: {})
    monkeypatch.setattr(Q.OLD.GPU, "query_gpu", lambda *_: {"busy": False, "processes": []})
    monkeypatch.setattr(Q.OLD, "validate_gpu_snapshot", lambda row: row)
    monkeypatch.setattr(Q.OLD, "verify_execution", lambda _: None)
    rows = batch_rows()
    if failure == "baseline": rows[1]["inspection"]["status"] = "unknown"
    if failure == "cost": rows[1]["update_wall_s"] = [2.] * 3
    if failure == "interference": rows[1]["state"]["policy_sha256"] = "b" * 64
    launched = []
    def launch(folder, *_):
        launched.append(folder.name)
        if failure == "execution": raise RuntimeError("fixture launch failure")
        return {"wall_s": 1., "owned_process_identities": {}}
    monkeypatch.setattr(Q.OLD.OWNED, "run_owned", launch)
    monkeypatch.setattr(Q, "inspect_run", lambda folder, *_: deepcopy(next(row for row in rows if row["id"] == folder.name)))
    tag = "ppo_gradient_v1_20261017"
    if failure == "execution":
        with pytest.raises(RuntimeError): Q.run(tag, Path("runtime"))
        assert len(launched) == 1
    else:
        assert Q.run(tag, Path("runtime"))["completed_runs"] == 6
        assert len(launched) == 6
    result = Q.read(Q.PRIVATE / "runs" / tag / "result.json")
    assert result["outcome"] != "confirmed"
    with pytest.raises(FileExistsError):
        Q.run(tag, Path("runtime"))
