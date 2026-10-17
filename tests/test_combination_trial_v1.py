from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from evals import combination_trial_v1 as T
from bench import combination_training_v1 as TRAINING


def test_fixed_plan_stops_before_conditional_restart_and_uses_no_api():
    protocol = T.fixed()
    assert protocol["sequence"].index("source_gate") < protocol["sequence"].index("family_restart_train")
    assert protocol["budget"]["api_calls"] == 0
    assert protocol["budget"]["maximum_gpu_runs"] == 5
    assert protocol["stop_rule"].startswith("source_failure_absent")
    assert protocol["selection"]["task"] == T.TASK
    assert protocol["predictions"]["family_restart"] == "recovered"
    assert protocol["retry_policy"]["semantic_retries"] == 0
    assert protocol["success_conditions"]["correction_consumptions"] == 1
    assert protocol["budget"]["maximum_parent_wall_s"] == 2700
    assert protocol["budget"]["maximum_owned_gpu_process_wall_s"] == 2400


@pytest.mark.parametrize("outcome,action", [
    ("source_failure_absent", "stop"), ("baseline_invalid", "stop"), ("failure_present", "contract")])
def test_source_gate_controls_conditional_restart(outcome, action):
    assert T._source_next(outcome) == action


def test_budget_exceeded_receipt_cannot_publish_recovered():
    assert T._receipt_outcome("recovered", {"verdict": "unknown", "over_budget": True}) == "invalid"
    assert T._receipt_outcome("recovered", {"verdict": "recovered", "over_budget": False}) == "recovered"


def test_total_time_budget_stops_the_next_stage(monkeypatch):
    protocol = T.fixed()
    monkeypatch.setattr(T.time, "monotonic", lambda: 2800.0)
    with pytest.raises(RuntimeError, match="총 시간 상한"):
        T._assert_total_budget(0.0, {}, protocol)
    monkeypatch.setattr(T.time, "monotonic", lambda: 100.0)
    with pytest.raises(RuntimeError, match="총 시간 상한"):
        T._assert_total_budget(0.0, {"source": 2400.1}, protocol)


def test_public_protocol_binds_registered_correction_noop():
    protocol = json.loads(T.PROTOCOL.read_text(encoding="utf-8"))
    noop = protocol["cpu_semantics"]["none_noop"]
    assert noop["selected_correction"] == T.FAMILY.CORRECTION_ID
    assert noop["before_fault"] == noop["after_fault"] == noop["after_correction"]


def test_manifest_fixes_anymal_task_seeds_and_restart_mode(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "PROTOCOL", tmp_path / "protocol.json")
    T.PROTOCOL.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(T, "PRIVATE", tmp_path / "private")
    manifest = T.manifest_for(T.TAG, "family_restart", tmp_path / "IsaacLab", "a" * 40)
    assert manifest["task"] == "Isaac-Velocity-Flat-Anymal-C-v0"
    assert manifest["seed"] == 3150
    assert manifest["recovery_mode"] == "restart"
    assert manifest["correction_id"] == T.FAMILY.CORRECTION_ID
    assert manifest["max_iterations"] == 300


def test_only_ready_contract_can_create_consumable_restart(monkeypatch, tmp_path):
    folder = tmp_path / "run"
    folder.mkdir()
    monkeypatch.setattr(T, "PROTOCOL", tmp_path / "protocol.json")
    T.PROTOCOL.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(T, "folder_for", lambda _: folder)
    monkeypatch.setattr(T, "verify", lambda *a, **k: ({}, None))
    monkeypatch.setattr(T, "verify_source_anchor", lambda anchor, *_: anchor)
    # 결합 파일 부재 또는 ready 밖 상태는 corrected manifest 검증 전에 닫힌다.
    manifest = {key: None for key in T.TRAIN.MANIFEST_FIELDS}
    manifest.update(T.manifest_for(T.TAG, "family_restart", tmp_path, "a" * 40))
    path = folder / "family_restart/train/manifest.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        T.validate_manifest(path)


def test_source_arm_does_not_require_contract_request(monkeypatch, tmp_path):
    monkeypatch.setattr(T, "PROTOCOL", tmp_path / "protocol.json")
    T.PROTOCOL.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(T, "PRIVATE", tmp_path / "private")
    monkeypatch.setattr(T, "verify", lambda *a, **k: ({}, None))
    monkeypatch.setattr(T, "verify_source_anchor", lambda anchor, *_: anchor)
    path = T.folder_for(T.TAG) / "source_none/train/manifest.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(T.manifest_for(T.TAG, "source_none", tmp_path, "a" * 40)), encoding="utf-8")
    manifest, _ = T.validate_manifest(path)
    assert manifest["correction_id"] is None


@pytest.mark.parametrize("fault,correction,changed,restored", [
    ("NONE", None, set(), False),
    ("U1", None, set(T.FAMILY.FUNCTIONS), False),
    ("U1", T.FAMILY.CORRECTION_ID, set(T.FAMILY.FUNCTIONS), True),
])
def test_readback_requires_exact_registered_family(fault, correction, changed, restored):
    clean = {name: str(index) * 64 for index, name in enumerate(T.FAMILY.FUNCTIONS, 1)}
    injected = {name: ("f" * 64 if name in changed else clean[name]) for name in T.FAMILY.FUNCTIONS}
    row = {
        "schema": "reward_family_correction_readback_v1", "selected_correction": correction,
        "registered_functions": list(T.FAMILY.FUNCTIONS), "before_fault": clean,
        "after_fault": injected, "after_correction": clean if restored else injected,
        "unrelated_changed_after_fault": [], "unrelated_changed_after_correction": [],
    }
    T.validate_readback(row, fault, correction)
    bad = deepcopy(row)
    bad["after_fault"][T.FAMILY.FUNCTIONS[0]] = clean[T.FAMILY.FUNCTIONS[0]]
    if fault == "U1":
        with pytest.raises(ValueError):
            T.validate_readback(bad, fault, correction)


def test_owned_launcher_records_intent_separately(monkeypatch, tmp_path):
    stage = tmp_path / "stage"
    args = ["python", "worker.py"]
    monkeypatch.setattr(T, "_admit", lambda _: None)
    leases = []
    monkeypatch.setattr(T, "assert_lease", lambda lease_id: leases.append(lease_id))
    def owned(folder, argv, cwd, timeout):
        T.write_new(folder / "launch.json", {"args": argv, "cwd": str(cwd.resolve())})
        return {"exit_code": 0, "natural_exit": True, "cleanup_verified": True, "timed_out": False,
                "error_type": None, "job_assigned_before_release": True}
    monkeypatch.setattr(T.OWNED, "run_owned", owned)
    monkeypatch.setattr(T.V3, "execution_valid", lambda row: row["exit_code"] == 0)
    result = T._execute_owned(stage, args, tmp_path, 10, "lease-a")
    assert result["exit_code"] == 0
    assert T.read(stage / "execution.json") == result
    assert leases == ["lease-a", "lease-a"]


@pytest.mark.skipif(sys.platform != "win32", reason="Windows Job launcher 계약")
def test_actual_owned_launcher_runs_cpu_child_and_records_launch(tmp_path):
    stage = tmp_path / "owned"
    stage.mkdir()
    args = [sys.executable, "-c", "print('owned-cpu-ok')"]
    result = T.OWNED.run_owned(stage, args, tmp_path, 30)
    assert T.V3.execution_valid(result)
    assert T.read(stage / "launch.json") == {"args": args, "cwd": str(tmp_path)}
    assert (stage / "stdout.log").read_text(encoding="utf-8").strip() == "owned-cpu-ok"


def _anchor_fixture(monkeypatch, tmp_path, working=b"clean\n", committed=b"clean\n"):
    protocol = tmp_path / "protocol.json"
    protocol.write_bytes(working)
    monkeypatch.setattr(T, "ROOT", tmp_path)
    monkeypatch.setattr(T, "PROTOCOL", protocol)
    monkeypatch.setattr(T, "CODE", set())
    monkeypatch.setattr(T, "_rel", lambda _: "protocol.json")
    monkeypatch.setattr(T, "read", lambda _: {"source_sha256_lf": {}})
    monkeypatch.setattr(T, "_git_bytes", lambda *args: committed)


def test_source_anchor_accepts_ancestor_only_when_bound_bytes_match(monkeypatch, tmp_path):
    _anchor_fixture(monkeypatch, tmp_path)
    seen = []
    def git(*args):
        seen.append(args)
        if args[:2] == ("rev-parse", "HEAD"):
            return "b" * 40
        if args[0] == "rev-parse":
            return "c" * 40
        return ""
    monkeypatch.setattr(T, "_git", git)
    assert T.verify_source_anchor("a" * 40) == "b" * 40
    assert ("merge-base", "--is-ancestor", "a" * 40, "b" * 40) in seen
    assert sum(args[0] == "rev-parse" for args in seen) == 3


def test_source_anchor_rejects_worktree_bytes_changed_after_verify(monkeypatch, tmp_path):
    _anchor_fixture(monkeypatch, tmp_path, working=b"dirty\n", committed=b"clean\n")
    monkeypatch.setattr(T, "_git", lambda *args: "b" * 40 if args[:2] == ("rev-parse", "HEAD") else
                        "c" * 40 if args[0] == "rev-parse" else "")
    with pytest.raises(ValueError, match="bytes"):
        T.verify_source_anchor("a" * 40)


def test_tag_and_paths_are_fixed():
    for tag in ("../escape", T.TAG + "_retry", "combination_anymal_u1_s2151_20261017"):
        with pytest.raises(ValueError):
            T.folder_for(tag)


def test_active_lease_reuses_validation_and_restores_owner(monkeypatch):
    original = T.LEASE.LEASE_OWNER
    seen = {}
    def context():
        seen["owner"] = T.LEASE.LEASE_OWNER
        return {"owner": seen["owner"], "state": "running"}
    monkeypatch.setattr(T.LEASE, "lease_context", context)
    assert T.active_lease()["owner"] == "rl-triage-combination-trial-v1"
    assert T.LEASE.LEASE_OWNER == original


def test_stage_lease_rejects_replacement_or_stale_context(monkeypatch):
    monkeypatch.setattr(T, "active_lease", lambda: {"leaseId": "replacement"})
    with pytest.raises(RuntimeError, match="바뀌었다"):
        T.assert_lease("original")
    monkeypatch.setattr(T, "active_lease", lambda: (_ for _ in ()).throw(ValueError("stale heartbeat")))
    with pytest.raises(ValueError, match="stale heartbeat"):
        T.assert_lease("original")


def test_protocol_cpu_runtime_contract_is_consistent_and_rejects_synthetic_mismatch():
    protocol = json.loads(T.PROTOCOL.read_text(encoding="utf-8"))
    correction = protocol["correction_source_contract"]
    assert correction == protocol["cpu_semantics"]["source_contract"]
    T.verify_cpu_contract(protocol, correction["source_sha256"])
    bad = deepcopy(protocol)
    bad["correction_source_contract"]["runtime"]["cache_tag"] = "cpython-312"
    with pytest.raises(ValueError, match="Isaac Python 3.10"):
        T.verify_cpu_contract(bad, correction["source_sha256"])


def test_consumption_rejects_ledger_row_from_other_manifest(monkeypatch, tmp_path):
    folder = tmp_path / "run"
    folder.mkdir()
    manifest = {"evidence": [{"path": "a", "sha256_lf": "1" * 64}]}
    state = {"status": "ready", "manifest_digest": "manifest-a", "restoration_targets": ["x"]}
    request = {"request_id": "r1", "manifest_digest": "manifest-a",
               "proposed": {"event": "proposed"}, "approved": {"event": "approved"},
               "consumed": {"event": "consumed"}}
    (folder / "combination_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (folder / "contract_request.json").write_text(json.dumps(request), encoding="utf-8")

    fake_contract = SimpleNamespace(
        evaluate=lambda *_args, **_kwargs: state,
        state_digest=lambda *_args: "current-state",
        registry=lambda: {"reward_family": {"protocol_case": "combination_u1_family_v1"}},
        L=SimpleNamespace(digest=lambda _value: "receipts"),
    )
    row = {"state": "running", "correction_id": T.FAMILY.CORRECTION_ID,
           "recovery_mode": "restart", "loop_state_digest": "other-manifest-state"}
    fake_ledger = SimpleNamespace(
        requests=lambda: {"r1": row}, events=lambda: [], registry_digest="registry",
        effect_key=lambda *_args: "effect")
    monkeypatch.setattr(T, "_contract_module", lambda: fake_contract)
    monkeypatch.setattr(T, "CorrectionLedger", lambda *_args, **_kwargs: fake_ledger)
    with pytest.raises(ValueError, match="승인·소비"):
        T._validate_contract_consumption(folder, "a" * 40)


@pytest.mark.parametrize("correction_applied", [False, True])
def test_execute_failure_preserves_invalid_result_and_consumption(monkeypatch, tmp_path, correction_applied):
    folder = tmp_path / "run"
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({"wording": {"invalid": "invalid"}}), encoding="utf-8")
    consumed = tmp_path / "consumed.json"
    monkeypatch.setattr(T, "PROTOCOL", protocol)
    monkeypatch.setattr(T, "folder_for", lambda _: folder)
    monkeypatch.setattr(T, "consumed_path", lambda: consumed)
    def fail(tag, isaaclab):
        folder.mkdir()
        T.write_new(consumed, {"tag": tag, "protocol_sha256_lf": T.sha(protocol, True),
                               "source_commit_sha": "a" * 40})
        if correction_applied:
            T.write_new(folder / "family_restart/train/correction.json", {"selected_correction": T.FAMILY.CORRECTION_ID})
        raise RuntimeError("synthetic failure")
    monkeypatch.setattr(T, "_execute", fail)
    with pytest.raises(RuntimeError, match="synthetic"):
        T.execute(T.TAG, tmp_path)
    result = T.read(folder / "result.json")
    assert result["status"] == result["outcome"] == "invalid"
    assert result["failure"]["error_type"] == "RuntimeError"
    assert result["correction_executed"] is correction_applied


def test_execute_never_writes_into_preexisting_unfinished_folder(monkeypatch, tmp_path):
    folder = tmp_path / "run"
    folder.mkdir()
    existing = folder / "partial.json"
    existing.write_text('{"kept":true}\n', encoding="utf-8")
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({"wording": {"invalid": "invalid"}}), encoding="utf-8")
    monkeypatch.setattr(T, "PROTOCOL", protocol)
    monkeypatch.setattr(T, "folder_for", lambda _: folder)
    monkeypatch.setattr(T, "consumed_path", lambda: tmp_path / "missing-consumed.json")
    monkeypatch.setattr(T, "_execute", lambda *_args: (_ for _ in ()).throw(FileExistsError("existing")))
    before = {path.name: path.read_bytes() for path in folder.iterdir()}
    with pytest.raises(FileExistsError, match="existing"):
        T.execute(T.TAG, tmp_path)
    assert {path.name: path.read_bytes() for path in folder.iterdir()} == before
    assert not (folder / "result.json").exists()


class _Helper:
    @staticmethod
    def sha(path, lf=False):
        import hashlib
        raw = Path(path).read_bytes()
        if lf:
            raw = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        return hashlib.sha256(raw).hexdigest()

    @staticmethod
    def validate_manifest(path):
        return json.loads(Path(path).read_text(encoding="utf-8")), {}


def _guard_manifest(tmp_path):
    root = tmp_path / "isaac"
    upstream = root / "scripts/reinforcement_learning/rsl_rl/train.py"
    upstream.parent.mkdir(parents=True)
    upstream.write_text("""
class App:
    def close(self):
        return None
simulation_app = App()
if __name__ == '__main__':
    simulation_app.close()
""", encoding="utf-8")
    output = tmp_path / "output"
    output.mkdir()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"isaaclab_root": str(root), "output_dir": str(output)}), encoding="utf-8")
    return manifest, output


def test_guarded_training_refuses_terminal_close_without_receipt(monkeypatch, tmp_path):
    manifest, output = _guard_manifest(tmp_path)
    def execute(path, helper):
        return TRAINING.RUNNER.runpy.run_path(
            Path(json.loads(path.read_text(encoding="utf-8"))["isaaclab_root"])
            / "scripts/reinforcement_learning/rsl_rl/train.py", run_name="__main__")
    monkeypatch.setattr(TRAINING.RUNNER, "execute", execute)
    with pytest.raises(RuntimeError, match="receipt"):
        TRAINING.guarded_execute(manifest, _Helper)
    assert (output / "training_failure.json").is_file()
    assert not (output / "close_ready.json").exists()


def test_guarded_training_seals_completion_before_close(monkeypatch, tmp_path):
    manifest, output = _guard_manifest(tmp_path)
    def execute(path, helper):
        TRAINING.GUARD.durable_new(output / "receipt.json", {
            "completed": True, "manifest_sha256": helper.sha(path), "actual_control_steps": 7200})
        TRAINING.RUNNER.runpy.run_path(
            Path(json.loads(path.read_text(encoding="utf-8"))["isaaclab_root"])
            / "scripts/reinforcement_learning/rsl_rl/train.py", run_name="__main__")
        return {"completed": True}
    monkeypatch.setattr(TRAINING.RUNNER, "execute", execute)
    assert TRAINING.guarded_execute(manifest, _Helper)["completed"] is True
    TRAINING.GUARD.validate_completion(output, _Helper.sha(manifest))


def _sealed_child_output(tmp_path):
    output = tmp_path / "train"
    output.mkdir()
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"fixed":true}\n', encoding="utf-8")
    manifest_sha = T.sha(manifest, False)
    T.write_new(output / T.GUARD.EVIDENCE, {
        "schema": "combination_training_evidence_v1", "manifest_sha256": manifest_sha, "updates": 300})
    (output / T.GUARD.RECORDS).write_bytes(b"")
    T.write_new(output / T.GUARD.CLOSE_READY, {
        "schema": T.GUARD.CLOSE_SCHEMA, "manifest_sha256": manifest_sha, "updates": 300,
        "evidence_sha256_lf": T.sha(output / T.GUARD.EVIDENCE, True),
        "records_sha256_lf": T.sha(output / T.GUARD.RECORDS, True),
    })
    return output, manifest


def test_parent_revalidates_child_close_seal_and_rejects_missing_marker(tmp_path):
    output, manifest = _sealed_child_output(tmp_path)
    T.validate_child_completion(output, manifest)
    (output / T.GUARD.CLOSE_READY).unlink()
    with pytest.raises(ValueError, match="close marker"):
        T.validate_child_completion(output, manifest)


def test_parent_rejects_child_close_seal_hash_mutation(tmp_path):
    output, manifest = _sealed_child_output(tmp_path)
    (output / T.GUARD.RECORDS).write_bytes(b"mutated")
    with pytest.raises(ValueError, match="close marker 결합"):
        T.validate_child_completion(output, manifest)
