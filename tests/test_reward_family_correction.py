from __future__ import annotations

import json
from pathlib import Path
import sys
from types import ModuleType
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "bench", ROOT / "evals"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import reward_family_correction as F  # noqa: E402
import demo_correction_u1_family as FAMILY_RUNNER  # noqa: E402
import u1_scope_extension as E  # noqa: E402
import u1_scope_source as SOURCE  # noqa: E402


def _install_fake(tmp_path, monkeypatch):
    root = tmp_path / "IsaacLab"
    source = root / F.SOURCE
    source.parent.mkdir(parents=True)
    body = """from __future__ import annotations
def track_lin_vel_xy_exp(env, std, command_name, asset_cfg=None): return ('lin', env)
def track_ang_vel_z_exp(env, std, command_name, asset_cfg=None): return ('ang', env)
def lin_vel_z_l2(env, asset_cfg=None): return ('z', env)
def ang_vel_xy_l2(env, asset_cfg=None): return ('xy', env)
def unrelated(env): return ('other', env)
"""
    source.write_text(body, encoding="utf-8", newline="\n")
    module = ModuleType(F.MODULE)
    module.__file__ = str(source)
    module.__dict__["__name__"] = F.MODULE
    exec(compile(body, str(source), "exec"), module.__dict__)
    monkeypatch.setitem(sys.modules, F.MODULE, module)
    return root, module


def _mutate(module, names):
    def replacement(env, std=None, command_name=None, asset_cfg=None):
        return ("fault", env)
    for name in names:
        getattr(module, name).__code__ = replacement.__code__


def _bind_fake_protocol(monkeypatch, contract):
    monkeypatch.setattr(E, "fixed_protocol", lambda: {"correction_source_contract": contract})


def test_full_family_restores_all_four_and_preserves_unrelated(tmp_path, monkeypatch):
    root, module = _install_fake(tmp_path, monkeypatch)
    contract = F.source_contract(root)
    _bind_fake_protocol(monkeypatch, contract)
    snapshot = F.capture(root, contract)
    _mutate(module, F.FUNCTIONS)
    row = F.apply(snapshot, F.CORRECTION_ID)
    assert set(row["restored_targets"]) == set(F.FUNCTIONS)
    assert row["after_correction"] == row["before_fault"]
    assert row["unrelated_changed_after_fault"] == row["unrelated_changed_after_correction"] == []
    E.validate_readback(row, "U1", F.CORRECTION_ID)


def test_none_control_is_a_noop_with_exact_readback(tmp_path, monkeypatch):
    root, _ = _install_fake(tmp_path, monkeypatch)
    contract = F.source_contract(root)
    _bind_fake_protocol(monkeypatch, contract)
    snapshot = F.capture(root, contract)
    row = F.apply(snapshot, F.CORRECTION_ID)
    assert row["changed_targets"] == row["restored_targets"] == []
    E.validate_readback(row, "NONE", F.CORRECTION_ID)


def test_environment_readback_requires_clean_functions_and_actual_term_binding(tmp_path, monkeypatch):
    root, module = _install_fake(tmp_path, monkeypatch)
    contract = F.source_contract(root)

    class Manager:
        active_terms = list(F.FUNCTIONS)

        def get_term_cfg(self, name):
            return SimpleNamespace(func=getattr(module, name))

    env = SimpleNamespace(unwrapped=SimpleNamespace(reward_manager=Manager()))
    assert F.environment_readback(env, contract)["ok"] is True
    Manager.get_term_cfg = lambda self, name: SimpleNamespace(func=(lambda: None))
    with pytest.raises(RuntimeError, match="not clean and bound"):
        F.environment_readback(env, contract)


def test_readback_rejects_forged_clean_hash_or_source_binding(tmp_path, monkeypatch):
    root, module = _install_fake(tmp_path, monkeypatch)
    contract = F.source_contract(root)
    _bind_fake_protocol(monkeypatch, contract)
    snapshot = F.capture(root, contract)
    _mutate(module, F.FUNCTIONS)
    row = F.apply(snapshot, F.CORRECTION_ID)
    forged = json.loads(json.dumps(row))
    forged["before_fault"][F.FUNCTIONS[0]] = "a" * 64
    with pytest.raises(ValueError, match="fully restored"):
        E.validate_readback(forged, "U1", F.CORRECTION_ID)
    forged = json.loads(json.dumps(row))
    forged["source_binding"]["source_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="source binding"):
        E.validate_readback(forged, "U1", F.CORRECTION_ID)


def test_unregistered_scope_and_unrelated_mutation_fail_closed(tmp_path, monkeypatch):
    root, module = _install_fake(tmp_path, monkeypatch)
    snapshot = F.capture(root, F.source_contract(root))
    with pytest.raises(ValueError, match="unregistered"):
        F.apply(snapshot, "restore_one_term")
    snapshot = F.capture(root, F.source_contract(root))
    _mutate(module, ["unrelated"])
    with pytest.raises(RuntimeError, match="outside"):
        F.apply(snapshot, F.CORRECTION_ID)
    with pytest.raises(ValueError, match="unregistered fault family"):
        E.choose_correction("U1", "scope_insufficient", ["track_ang_vel_z_exp"], "other")
    with pytest.raises(ValueError, match="unregistered reward function"):
        E.choose_correction("U1", "scope_insufficient", ["made_up"],
                            F.REGISTRY[F.CORRECTION_ID]["fault_family"])


def test_selection_and_preregistered_protocol_are_fixed():
    protocol = E.validate_protocol(E.fixed_protocol())
    selection = protocol["selection_contract"]
    assert E.choose_correction("U1", selection["required_scope_verdict"], selection["registered_uncovered_terms"],
                               selection["fault_family"]) == F.CORRECTION_ID
    assert protocol["status"] == "preregistered_unmeasured"
    assert (protocol["fault_seed"], protocol["restart_seed"]) == (2051, 3051)
    assert protocol["predictions"]["u1_family_restart"].startswith("결과 미측정")
    assert (protocol["action_scale"], protocol["rsl_rl_version"], protocol["isaaclab_commit"]) == (
        0.25, "2.3.3", "90b79bb2d44feb8d833f260f2bf37da3487180ba")
    assert protocol["watchdogs_s"] == {"train": 1800, "fixed_eval": 360}


def test_strict_v4_verifier_is_wired_to_the_new_contract(monkeypatch, tmp_path):
    seen = {}

    def verify(folder, arm, execution, protocol, versions):
        seen.update(folder=folder, arm=arm, execution=execution, protocol=protocol, versions=versions)
        assert E.STRICT.validate_manifest is E.validate_manifest
        assert E.STRICT.validate_readback is E.validate_readback
        assert E.STRICT.validate_training_configuration is E.validate_training_configuration
        assert E.STRICT.UPSTREAM_TARGETS == {}
        return {"checkpoint": {"path": "p", "sha256": "a" * 64}}

    monkeypatch.setattr(E.STRICT, "verify_training", verify)
    old = (E.STRICT.validate_manifest, E.STRICT.validate_readback, E.STRICT.validate_training_configuration,
           E.STRICT.UPSTREAM_TARGETS)
    result = E._strict_verify(tmp_path, "none_family_control", {"exit_code": 0}, E.fixed_protocol(), {"torch": "x"})
    assert result["checkpoint"]["sha256"] == "a" * 64 and seen["arm"] == "none_family_control"
    assert (E.STRICT.validate_manifest, E.STRICT.validate_readback, E.STRICT.validate_training_configuration,
            E.STRICT.UPSTREAM_TARGETS) == old


def test_child_runner_passes_the_complete_h_interface(monkeypatch, tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    seen = {}
    monkeypatch.setattr(FAMILY_RUNNER, "install_fault", lambda path: "NONE")

    def execute(path, harness):
        seen.update(path=path, harness=harness)
        for name in ("validate_manifest", "validate_readback", "validate_training_configuration", "sha", "write_new"):
            assert callable(getattr(harness, name))
        protocol = harness.fixed_protocol()
        for key in ("action_scale", "rsl_rl_version", "isaaclab_commit", "correction_source_contract"):
            assert key in protocol

    monkeypatch.setattr(FAMILY_RUNNER.RUNNER, "execute", execute)
    assert FAMILY_RUNNER.main(["--manifest", str(manifest)]) == 0
    assert seen == {"path": manifest, "harness": E}


def test_source_phase_redirects_the_existing_u1_lifecycle():
    old = (SOURCE.E.VARIANTS["u1"], SOURCE.E.PRIVATE, SOURCE.E.RESULTS, SOURCE.E.PROTOCOL_DIR,
           set(SOURCE.E.CODE), SOURCE.E.fixed)
    try:
        SOURCE.configure()
        assert SOURCE.E.VARIANTS["u1"]["seed"] == 2051
        assert SOURCE.E.PRIVATE == SOURCE.SOURCE_PRIVATE and SOURCE.E.RESULTS == SOURCE.SOURCE_RESULTS
        assert SOURCE.E.PROTOCOL_DIR == SOURCE.SOURCE_PROTOCOL_DIR
        assert SOURCE.SOURCE_CODE <= SOURCE.E.CODE
        contract = SOURCE.E.fixed("u1", "single")
        assert contract["api_calls"] == 0 and contract["enabled_steps"] == SOURCE.ENABLED_STEPS
        assert contract["model_loop"]["status"] == "inactive" and contract["correction"]["status"] == "inactive"
    finally:
        SOURCE.E.VARIANTS["u1"], SOURCE.E.PRIVATE, SOURCE.E.RESULTS, SOURCE.E.PROTOCOL_DIR = old[:4]
        SOURCE.E.CODE = old[4]
        SOURCE.E.fixed = old[5]


def test_source_scope_runs_registered_probe_and_binds_actual_artifacts(monkeypatch, tmp_path):
    monkeypatch.setattr(SOURCE, "configure", lambda: None)
    monkeypatch.setattr(SOURCE, "ROOT", tmp_path)
    folder = tmp_path / "source/tag1"
    folder.mkdir(parents=True)
    ctx = SimpleNamespace(seed=2051, cases={"FAULT": "u1_s2051"}, protocol=tmp_path / "protocol.json")
    ctx.protocol.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(SOURCE.E, "folder_for", lambda tag: folder)
    monkeypatch.setattr(SOURCE.E, "write_new", E.write_new)
    reference = {"probes": {"P_reward": {"measurement": {
        "track_lin_vel_xy_exp_rel_error": 0.0, "track_ang_vel_z_exp_rel_error": 0.0}}}}
    evidence = {"ctx": ctx, "reference": reference,
                "artifacts": {"FAULT": {"checkpoint": {"sha256": "b" * 64}}}}
    monkeypatch.setattr(SOURCE, "source_evidence", lambda tag, isaaclab: evidence)
    monkeypatch.setattr(SOURCE, "_scope_bindings", lambda tag, isaaclab, probe: {"source_tag": tag})
    monkeypatch.setattr(SOURCE, "validate_scope_record", lambda path, isaaclab: json.loads(path.read_text(encoding="utf-8")))
    seen = {}

    def probe(isaaclab, **kwargs):
        seen.update(kwargs)
        execution = folder / "probe_execution.json"
        execution.write_text("{}", encoding="utf-8")
        return {"measurement": {"track_lin_vel_xy_exp_rel_error": 0.9,
                                 "track_ang_vel_z_exp_rel_error": 0.9},
                "gpu_s": 10.0, "exit_code": 0, "paths": {"execution": str(execution)}}

    monkeypatch.setattr(SOURCE.UR, "probe", probe)
    monkeypatch.setattr(SOURCE.L, "classify", lambda *args: "abnormal")
    out = SOURCE.diagnose_scope("tag1", tmp_path / "IsaacLab")
    scope = json.loads(Path(out["scope"]).read_text(encoding="utf-8"))
    assert scope["result"]["uncovered_terms"] == ["track_ang_vel_z_exp", "lin_vel_z_l2", "ang_vel_xy_l2"]
    assert scope["observed_probe_terms"] == ["track_lin_vel_xy_exp", "track_ang_vel_z_exp"]
    assert scope["source_registered_terms"] == list(F.FUNCTIONS)
    assert seen["work_dir"] == folder / "exec/scope_probe" and seen["fault"] == "U1" and seen["seed"] == 2051


def test_owned_evaluation_uses_absolute_isaac_argv_and_consumes_once(monkeypatch, tmp_path):
    monkeypatch.setattr(E, "PRIVATE", tmp_path / "private")
    folder = E._folder("eval1")
    folder.mkdir(parents=True)
    (folder / "eval").mkdir()
    monkeypatch.setattr(E, "write_eval_jobs", lambda tag: {"jobs": "ok"})
    seen = {}

    def run_owned(work, argv, cwd, watchdog):
        seen.update(work=work, argv=argv, cwd=cwd, watchdog=watchdog)
        return {"exit_code": 0, "wall_s": 1.0, "owned_process_identities": {"123": "abc"},
                "timed_out": False, "cleanup": {"remaining_owned_processes": []}}

    monkeypatch.setattr(E.P, "run_owned", run_owned)
    monkeypatch.setattr(E.V3, "execution_valid", lambda row: True)
    profiles = []
    def query_gpu(profile):
        profiles.append(profile)
        return {"busy": False, "processes": [], "admission_policy": profile}
    monkeypatch.setattr(E.MEDIA, "query_gpu", query_gpu)
    monkeypatch.setattr(E, "judge", lambda tag: {"outcome": "recovered"})
    isaac = tmp_path / "IsaacLab"
    (isaac / "_isaac_sim").mkdir(parents=True)
    assert E.run_evaluation("eval1", isaac) == {"outcome": "recovered"}
    assert Path(seen["argv"][0]).is_absolute() and Path(seen["argv"][1]).is_absolute()
    assert seen["watchdog"] == 360 and (folder / "eval/execution.json").is_file()
    assert profiles == [E.GPU_PROFILE, E.GPU_PROFILE]
    with pytest.raises(FileExistsError):
        E.run_evaluation("eval1", isaac)


def test_judge_refuses_missing_owned_execution(monkeypatch, tmp_path):
    monkeypatch.setattr(E, "PRIVATE", tmp_path / "private")
    folder = E._folder("judge1")
    folder.mkdir(parents=True)
    reference = ROOT / "evals/results/p0c_fixed_eval_20261009/runs/baseline_p0c_s2026.json"
    (folder / "prereg.json").write_text(json.dumps({"reference_report": {"path": str(reference),
                                                                           "sha256": E.sha(reference)}}),
                                          encoding="utf-8")
    with pytest.raises(ValueError, match="execution receipt"):
        E.judge("judge1")


def test_contract_extracts_real_isaac_reward_functions_when_source_is_available():
    isaaclab = Path.home() / "IsaacLab"
    source = isaaclab / F.SOURCE
    assert source.is_file()
    contract = F.source_contract(isaaclab)
    assert contract["source_sha256"] == E.fixed_protocol()["correction_source_contract"]["source_sha256"]
    assert set(contract["functions"]) == set(F.FUNCTIONS)
    assert all(len(value) == 64 for value in contract["functions"].values())


def test_manifest_writer_is_write_once_and_uses_only_registered_arms(tmp_path, monkeypatch):
    monkeypatch.setattr(E, "PRIVATE", tmp_path / "private")
    monkeypatch.setattr(E, "PROTOCOL", ROOT / "bench/protocols/u1_scope_extension_v1.json")
    monkeypatch.setattr(E, "validate_protocol", lambda protocol, isaaclab=None: protocol)
    monkeypatch.setattr(E.P, "verify_committed", lambda root, code: "a" * 40)
    env, agent = tmp_path / "env.yaml", tmp_path / "agent.yaml"
    env.write_text("seed: 2051\n", encoding="utf-8")
    agent.write_text("seed: 2051\n", encoding="utf-8")
    reference = tmp_path / "none_report.json"
    reference.write_text("{}", encoding="utf-8")
    checkpoint = tmp_path / "model_299.pt"
    checkpoint.write_bytes(b"checkpoint")
    scope = tmp_path / "scope.json"
    bindings = {"source_commit_sha": "a" * 40,
                "source_env_config_path": str(env.resolve()), "source_env_config_sha256": E.sha(env),
                "source_agent_config_path": str(agent.resolve()), "source_agent_config_sha256": E.sha(agent),
                "none_fixed_report_path": str(reference.resolve()), "none_fixed_report_sha256": E.sha(reference),
                "none_checkpoint_path": str(checkpoint.resolve()), "none_checkpoint_sha256": E.sha(checkpoint)}
    scope_row = {"schema": "u1_scope_extension_scope_v1", "fault": "U1", "seed": E.FAULT_SEED,
                 "fault_family": "root_body_velocity_reward_family",
                 "observed_probe_terms": ["track_lin_vel_xy_exp", "track_ang_vel_z_exp"],
                 "source_registered_terms": list(F.FUNCTIONS),
                 "result": {"verdict": "scope_insufficient",
                            "uncovered_terms": ["track_ang_vel_z_exp", "lin_vel_z_l2", "ang_vel_xy_l2"]},
                 "bindings": bindings}
    scope.write_text(json.dumps(scope_row), encoding="utf-8")
    approval = tmp_path / "approval.json"
    approval.write_text(json.dumps({"schema": "human_correction_approval_v1", "decision": "approve",
                                    "correction_id": F.CORRECTION_ID, "scope_sha256": E.sha(scope),
                                    "identity_attested": False}), encoding="utf-8")
    monkeypatch.setattr(E, "validate_authorization", lambda scope_path, approval_path, isaaclab: (scope_row, {}))
    monkeypatch.setattr(E.C, "validate_fixed_report", lambda report, expected: {"checkpoint": expected,
                                                                                "versions": {}})
    monkeypatch.setattr(E, "_validate_source_inputs", lambda prereg, isaaclab: (scope_row, {}))
    out = E.write_manifests("u1_scope_v1", tmp_path / "isaaclab", scope, approval)
    assert set(out["manifests"]) == set(E.ARMS)
    for arm, path in out["manifests"].items():
        row = json.loads(Path(path).read_text(encoding="utf-8"))
        assert row["arm"] == arm and row["correction_id"] == F.CORRECTION_ID
        assert row["recovery_mode"] == "restart" and row["final_iteration"] == 299
    with pytest.raises(FileExistsError):
        E.write_manifests("u1_scope_v1", tmp_path / "isaaclab", scope, approval)


def test_authorization_rejects_unbound_or_unregistered_scope(tmp_path):
    scope = tmp_path / "scope.json"
    scope.write_text(json.dumps({"schema": "u1_scope_extension_scope_v1", "fault": "U1", "seed": 2051,
                                 "fault_family": "root_body_velocity_reward_family",
                                 "result": {"verdict": "scope_insufficient", "uncovered_terms": ["made_up"]}}),
                     encoding="utf-8")
    approval = tmp_path / "approval.json"
    approval.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="unregistered reward function"):
        E.validate_authorization(scope, approval, tmp_path / "IsaacLab")


def test_gpu_admission_profile_is_registered_and_query_receives_its_name(monkeypatch, tmp_path):
    profile = E.fixed_protocol()["gpu_admission_profile"]
    assert E.MEDIA._gpu_admission_profile(profile)["policy"] == E.GPU_PROFILE
    with pytest.raises(E.MEDIA.PreflightError):
        E.MEDIA._gpu_admission_profile({"devices": [0]})


def test_source_fixed_eval_uses_private_jobs_and_low_level_sdk_entry(monkeypatch, tmp_path):
    monkeypatch.setattr(SOURCE, "SOURCE_RESULTS", tmp_path / "private_results")
    folder = tmp_path / "source/tag"
    ctx = SimpleNamespace(cases={"NONE": "baseline_p0c_s2051", "FAULT": "u1_s2051"},
                          protocol=tmp_path / "protocol.json")
    ctx.protocol.write_text("{}", encoding="utf-8")
    checkpoints = {}
    for arm in ("NONE", "FAULT"):
        path = tmp_path / f"{arm}.pt"
        path.write_bytes(arm.encode())
        checkpoints[arm] = {"path": str(path), "sha256": E.sha(path)}
    protocol = {"action_scale": 0.25, "fixed_eval_sha256_lf": "f" * 64,
                "bands": {}, "watchdogs_s": {"fixed_eval": 360}}
    monkeypatch.setattr(SOURCE.E, "ctx_for", lambda tag: ctx)
    monkeypatch.setattr(SOURCE.E, "_guard", lambda tag, step: None)
    monkeypatch.setattr(SOURCE.E, "verify_for", lambda tag, isaac: (protocol, "a" * 40))
    monkeypatch.setattr(SOURCE.E, "_check_artifacts", lambda tag, arm: {"checkpoint": checkpoints[arm]})
    monkeypatch.setattr(SOURCE.E.V2, "_admit_gpu", lambda protocol: {"busy": False})
    monkeypatch.setattr(SOURCE.E, "folder_for", lambda tag: folder)
    monkeypatch.setattr(SOURCE.E, "write_new", E.write_new)
    monkeypatch.setattr(SOURCE.E.V3, "execution_valid", lambda row: True)
    monkeypatch.setattr(SOURCE, "_validate_report", lambda *args: {"versions": {}})
    monkeypatch.setattr(SOURCE.E, "source_gate", lambda *args: {"outcome": "failure_present"})
    monkeypatch.setattr(SOURCE.E, "_write_step", lambda tag, step, previous, commit, path, row: row)
    monkeypatch.setattr(SOURCE.E, "_rel", lambda path: str(Path(path).resolve()))
    seen = {}

    def execute(tag, step, argv, watchdog):
        seen.update(argv=argv, watchdog=watchdog)
        jobs = json.loads((folder / "jobs/fixed_eval.json").read_text(encoding="utf-8"))
        for job in jobs["jobs"]:
            Path(job["output"]).parent.mkdir(parents=True, exist_ok=True)
            Path(job["output"]).write_text("{}", encoding="utf-8")
        return {"exit_code": 0}, 1

    monkeypatch.setattr(SOURCE.E, "_execute", execute)
    isaaclab = tmp_path / "IsaacLab"
    result = SOURCE._private_fixed_evaluate("tag", isaaclab)
    jobs = json.loads((folder / "jobs/fixed_eval.json").read_text(encoding="utf-8"))
    assert all(Path(job["output"]).is_relative_to(SOURCE.SOURCE_RESULTS) for job in jobs["jobs"])
    assert Path(seen["argv"][1]).name == "fixed_eval.py" and "run_fixed_eval.py" not in seen["argv"]
    assert result["gate"]["outcome"] == "failure_present"


def test_scope_authorization_requires_source_validator_and_distinct_hashes(monkeypatch, tmp_path):
    scope = tmp_path / "scope.json"
    row = {"schema": "u1_scope_extension_scope_v1", "fault": "U1", "seed": 2051,
           "fault_family": "root_body_velocity_reward_family",
           "observed_probe_terms": ["track_lin_vel_xy_exp", "track_ang_vel_z_exp"],
           "source_registered_terms": list(F.FUNCTIONS),
           "result": {"verdict": "scope_insufficient",
                      "uncovered_terms": ["track_ang_vel_z_exp", "lin_vel_z_l2", "ang_vel_xy_l2"]},
           "bindings": {"source_tag": "source2051", "source_commit_sha": "a" * 40,
                        "none_checkpoint_sha256": "b" * 64, "u1_checkpoint_sha256": "c" * 64}}
    scope.write_text(json.dumps(row), encoding="utf-8")
    approval = tmp_path / "approval.json"
    approval.write_text(json.dumps({"schema": "human_correction_approval_v1", "decision": "approve",
                                    "correction_id": F.CORRECTION_ID, "scope_sha256": E.sha(scope),
                                    "identity_attested": False}), encoding="utf-8")
    calls = []
    monkeypatch.setattr(SOURCE, "validate_scope_record", lambda path, isaac: calls.append((path, isaac)) or row)
    E.validate_authorization(scope, approval, tmp_path / "IsaacLab")
    assert calls and row["bindings"]["none_checkpoint_sha256"] != row["bindings"]["u1_checkpoint_sha256"]
    monkeypatch.setattr(SOURCE, "validate_scope_record",
                        lambda path, isaac: (_ for _ in ()).throw(ValueError("forged source hash")))
    with pytest.raises(ValueError, match="forged source hash"):
        E.validate_authorization(scope, approval, tmp_path / "IsaacLab")


def test_scope_validator_rejects_noncanonical_or_mismatched_probe_evidence(monkeypatch, tmp_path):
    monkeypatch.setattr(SOURCE, "configure", lambda: None)
    folder = tmp_path / "source/source2051"
    folder.mkdir(parents=True)
    results = tmp_path / "source_results"
    monkeypatch.setattr(SOURCE, "SOURCE_RESULTS", results)
    monkeypatch.setattr(SOURCE.E, "folder_for", lambda tag: folder)
    case = "u1_s2051"
    paths = SOURCE.UR.probe_paths("source2051_scope", case, "P_reward", results)
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    measurement = {"track_lin_vel_xy_exp_rel_error": 0.9, "track_ang_vel_z_exp_rel_error": 0.0}
    paths["report"].write_text(json.dumps({"name": case, "checkpoint": {"sha256": "b" * 64},
                                            "probes": {"P_reward": {"exit": 0, "measurement": measurement}}}),
                                    encoding="utf-8")
    paths["execution"].write_text(json.dumps({"schema": "probe_execution_v1", "case": case,
                                               "probe": "P_reward", "checkpoint_sha256": "b" * 64,
                                               "report_sha256_lf": E.sha(paths["report"], lf=True), "exit_code": 0,
                                               "upstream_readback_ok": True}), encoding="utf-8")
    probe_path = folder / "scope_probe.json"
    probe_path.write_text(json.dumps({"schema": "u1_scope_probe_receipt_v1", "case": case, "probe": "P_reward",
                                      "measurement": measurement, "outcome": "abnormal", "exit_code": 0, "gpu_s": 1.0,
                                      "checkpoint_sha256": "b" * 64,
                                      "execution_receipt_path": str(paths["execution"].resolve()),
                                      "execution_receipt_sha256": E.sha(paths["execution"])}), encoding="utf-8")
    bindings = {"source_tag": "source2051", "probe_receipt_path": str(probe_path.resolve())}
    scope = folder / "scope.json"
    scope.write_text(json.dumps({"schema": "u1_scope_extension_scope_v1", "fault": "U1", "seed": 2051,
                                 "bindings": bindings}), encoding="utf-8")
    evidence = {"ctx": SimpleNamespace(cases={"FAULT": case}),
                "artifacts": {"FAULT": {"checkpoint": {"sha256": "b" * 64}}},
                "reference": {"probes": {"P_reward": {"measurement": {
                    "track_lin_vel_xy_exp_rel_error": 0.0, "track_ang_vel_z_exp_rel_error": 0.0}}}}}
    monkeypatch.setattr(SOURCE, "source_evidence", lambda tag, isaaclab: evidence)
    monkeypatch.setattr(SOURCE, "_scope_bindings", lambda tag, isaaclab, path: bindings)
    SOURCE.validate_scope_record(scope, tmp_path / "IsaacLab")

    forged = json.loads(paths["report"].read_text(encoding="utf-8"))
    forged["probes"]["P_reward"]["measurement"] = {"track_lin_vel_xy_exp_rel_error": 0.1,
                                                        "track_ang_vel_z_exp_rel_error": 0.0}
    paths["report"].write_text(json.dumps(forged), encoding="utf-8")
    with pytest.raises(ValueError, match="receipt differs"):
        SOURCE.validate_scope_record(scope, tmp_path / "IsaacLab")

    other = folder / "other_probe.json"
    other.write_bytes(probe_path.read_bytes())
    changed = json.loads(scope.read_text(encoding="utf-8"))
    changed["bindings"]["probe_receipt_path"] = str(other.resolve())
    scope.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical source run"):
        SOURCE.validate_scope_record(scope, tmp_path / "IsaacLab")
