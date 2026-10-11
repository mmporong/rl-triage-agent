from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import yaml

from rl_triage.reset_com_observer import BaseComResetObserver, MEASUREMENT_DEFINITIONS_SHA256


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "bench"))
SPEC = importlib.util.spec_from_file_location(
    "base_com_reset_launcher_test", ROOT / "bench/run_base_com_reset.py",
)
assert SPEC and SPEC.loader
L = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(L)
SHA = "a" * 64


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FakeTorch:
    @staticmethod
    def arange(start, stop, step, device=None):
        return list(range(start, stop, step))


class RecordingObserver:
    def __init__(self):
        self.calls = []

    def record_completed_reset(self, *, reset_sequence, reset_env_ids):
        self.calls.append((reset_sequence, list(reset_env_ids)))
        return {"reset_env_ids": list(reset_env_ids)}


class FakeBase:
    def __init__(self, num_envs=64, *, active=True):
        self.num_envs = num_envs
        self.device = "cuda:0"
        self.common_step_counter = 0
        self._sim_step_counter = 0
        self.episode_length_buf = [5] * num_envs
        self.event_manager = SimpleNamespace(
            active_terms={"reset": ["base_com"] if active else []},
        )
        self.reset_calls = []

    def _reset_idx(self, env_ids):
        ids = list(env_ids)
        self.reset_calls.append(ids)
        for env_id in ids:
            self.episode_length_buf[env_id] = 0


class FakeView:
    def __init__(self, values):
        self.values = deepcopy(values)

    def get_coms(self):
        return deepcopy(self.values)


def small_observer_fixture(num_envs=4):
    values = [
        [[0.0, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0],
         [100.0, 100.0, 1.0, 1.0, 0.0, 0.0, 0.0]]
        for _ in range(num_envs)
    ]
    view = FakeView(values)
    robot = SimpleNamespace(body_names=["base", "leg"], root_physx_view=view)
    env = SimpleNamespace(num_envs=num_envs, scene={"robot": robot})
    observer = BaseComResetObserver(
        env,
        source_binding={
            "task_sha256": SHA, "configuration_sha256": "b" * 64,
            "isaaclab_sha256": "c" * 64, "observer_sha256": "d" * 64,
        },
        expected_reset_sequences=4, expected_resets_per_env=2,
    )
    for sequence in range(4):
        ids = list(range(sequence % 2, num_envs, 2))
        for env_id in ids:
            view.values[env_id][0][0] += 0.01
        observer.record_completed_reset(reset_sequence=sequence, reset_env_ids=ids)
    return observer.aggregate()


def small_protocol():
    return {
        "num_envs": 4, "reset_sequences": 4, "resets_per_env": 2,
        "allowed_xy_range": 0.05, "tolerance": 1.0e-4,
    }


def test_fixed_contract_includes_fault_observer_launcher_and_sdk_mechanism_sources():
    assert {
        "bench/hidden_faults.py", "bench/external_faults.py",
        "bench/run_base_com_reset.py", "src/rl_triage/reset_com_observer.py",
    } <= L.PROJECT_SOURCE_FILES
    assert {
        "source/isaaclab/isaaclab/envs/mdp/events.py",
        "source/isaaclab/isaaclab/assets/articulation/articulation.py",
        "source/isaaclab/isaaclab/managers/scene_entity_cfg.py",
        "source/isaaclab/isaaclab/managers/event_manager.py",
        "source/isaaclab/isaaclab/managers/manager_base.py",
    } <= L.ISAACLAB_SOURCE_FILES
    assert L.FIXED_PROTOCOL_VALUES["num_envs"] == 64
    assert L.FIXED_PROTOCOL_VALUES["reset_sequences"] == 20
    assert L.FIXED_PROTOCOL_VALUES["resets_per_env"] == 10
    assert L.FIXED_PROTOCOL_VALUES["model_selected"] is False
    assert L.FIXED_PROTOCOL_VALUES["schema"] == "base_com_reset_observation_protocol_v3"
    assert L.FIXED_PROTOCOL_VALUES["gpu_admission_profile"] == "base_com_reset_readback_v2"
    assert L.FIXED_PROTOCOL_VALUES["source_binding_revision"] == 3


def test_collect_resets_uses_exact_twenty_alternating_subsets_and_records_sdk_state():
    base = FakeBase()
    observer = RecordingObserver()
    phase = L.collect_resets(base, observer, FakeTorch, L.FIXED_PROTOCOL_VALUES)

    assert len(base.reset_calls) == len(observer.calls) == 20
    for sequence in range(20):
        expected = list(range(sequence % 2, 64, 2))
        assert base.reset_calls[sequence] == expected
        assert observer.calls[sequence] == (sequence, expected)
        row = phase["reset_rows"][sequence]
        assert row["expected_env_ids"] == row["sdk_reset_env_ids"] == expected
        assert row["observer_reset_env_ids"] == expected
        assert row["control_delta"] == row["physics_delta"] == 0
    assert phase["base_com_active_reset_term"] is True
    assert phase["active_reset_terms"] == ["base_com"]
    assert phase["initial_sdk_state"]["episode_length"] == [5] * 64
    assert phase["initial_sdk_state"]["reset_buffer_attribute_present"] is False
    assert phase["initial_sdk_state"]["reset_buffer"] is None
    assert phase["final_sdk_state"]["episode_length"] == [0] * 64


def test_collect_resets_rejects_missing_active_event_and_size_drift():
    with pytest.raises(ValueError, match="base_com"):
        L.collect_resets(FakeBase(active=False), RecordingObserver(), FakeTorch, L.FIXED_PROTOCOL_VALUES)
    wrong = FakeBase(num_envs=63)
    with pytest.raises(ValueError, match="protocol size"):
        L.collect_resets(wrong, RecordingObserver(), FakeTorch, L.FIXED_PROTOCOL_VALUES)


def test_configuration_payload_reads_independent_resolved_manager_term():
    original = {
        "events": {"base_com": {"params": {"asset_cfg": {
            "body_names": "base", "body_ids": slice(None),
        }}}},
    }
    resolved = {"params": {"asset_cfg": {"body_names": ["base"], "body_ids": [0]}}}

    class EventManager:
        def get_term_cfg(self, name):
            assert name == "base_com"
            return resolved

    base = SimpleNamespace(cfg=original, event_manager=EventManager())
    payload = L._configuration_snapshot_payload(
        base, deepcopy(L.EXPECTED_INJECTION_READBACK), L.FIXED_PROTOCOL_VALUES, deepcopy,
    )

    assert payload["environment"]["events"]["base_com"]["params"]["asset_cfg"]["body_names"] == "base"
    assert payload["resolved_event_readback"]["base_com"]["params"]["asset_cfg"] == {
        "body_names": ["base"], "body_ids": [0],
    }
    assert payload["resolved_event_readback"]["source"] == L.RESOLVED_EVENT_READ_SOURCE
    resolved["params"]["asset_cfg"]["body_ids"][0] = 7
    assert payload["resolved_event_readback"]["base_com"]["params"]["asset_cfg"]["body_ids"] == [0]


def test_configuration_payload_fails_when_manager_term_is_missing():
    manager = SimpleNamespace(get_term_cfg=lambda _name: (_ for _ in ()).throw(KeyError("missing")))
    base = SimpleNamespace(cfg={}, event_manager=manager)
    with pytest.raises(KeyError, match="missing"):
        L._configuration_snapshot_payload(
            base, L.EXPECTED_INJECTION_READBACK, L.FIXED_PROTOCOL_VALUES, deepcopy,
        )


def test_parent_reaggregates_complete_raw_observer_report():
    report = small_observer_fixture()
    assert L.verify_observer_report(report, small_protocol()) == []


@pytest.mark.parametrize("mutate, reason", [
    (lambda report: report["reset_records"][0]["samples"][0].update(delta_x=9.0),
     "reset_raw_reaggregation_failed"),
    (lambda report: report["coverage"].update(finite_reset_measurement_count=7),
     "observer_coverage_mismatch"),
    (lambda report: report["per_environment"][0].update(declared_reset_count=1),
     "per_environment_summary_mismatch"),
    (lambda report: report["measurements"]["base_com_outside_range_fraction"].update(value=1.0),
     "aggregate_measurement_mismatch"),
    (lambda report: report.update(model_selected=True), "forbidden_claim_flag"),
])
def test_parent_reaggregation_rejects_raw_tamper_missing_coverage_and_claims(mutate, reason):
    report = small_observer_fixture()
    mutate(report)
    assert reason in L.verify_observer_report(report, small_protocol())


def make_source_fixture(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    isaac = tmp_path / "isaac"
    for relative in L.PROJECT_SOURCE_FILES:
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")
    for relative in L.ISAACLAB_SOURCE_FILES:
        path = isaac / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")
    protocol = deepcopy(L.FIXED_PROTOCOL_VALUES)
    protocol.update({
        "measurement_definitions_sha256": MEASUREMENT_DEFINITIONS_SHA256,
        "source_files": {relative: digest(repo / relative) for relative in L.PROJECT_SOURCE_FILES},
        "isaaclab_source_files": {
            relative: digest(isaac / relative) for relative in L.ISAACLAB_SOURCE_FILES
        },
    })
    monkeypatch.setattr(L.S, "_git_head", lambda root: L.ISAACLAB_COMMIT)
    monkeypatch.setattr(L.S, "_git_tracked_clean", lambda root: True)
    return repo, isaac, protocol


def test_protocol_validates_exact_sources_and_rejects_source_drift(tmp_path, monkeypatch):
    repo, isaac, protocol = make_source_fixture(tmp_path, monkeypatch)
    receipt = L.validate_protocol(protocol, repo_root=repo, isaaclab_root=isaac)
    assert set(receipt["project_source_files"]) == L.PROJECT_SOURCE_FILES
    assert set(receipt["isaaclab_source_files"]) == L.ISAACLAB_SOURCE_FILES
    assert len(receipt["task_binding_sha256"]) == 64

    (repo / "bench/hidden_faults.py").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="source SHA mismatch"):
        L.validate_protocol(protocol, repo_root=repo, isaaclab_root=isaac)


def test_protocol_rejects_extra_field_sdk_drift_and_dirty_checkout(tmp_path, monkeypatch):
    repo, isaac, protocol = make_source_fixture(tmp_path, monkeypatch)
    altered = deepcopy(protocol)
    altered["extra"] = True
    with pytest.raises(ValueError, match="fields differ"):
        L.validate_protocol(altered, repo_root=repo, isaaclab_root=isaac)

    monkeypatch.setattr(L.S, "_git_head", lambda root: "0" * 40)
    with pytest.raises(ValueError, match="pinned commit"):
        L.validate_protocol(protocol, repo_root=repo, isaaclab_root=isaac)
    monkeypatch.setattr(L.S, "_git_head", lambda root: L.ISAACLAB_COMMIT)
    monkeypatch.setattr(L.S, "_git_tracked_clean", lambda root: False)
    with pytest.raises(ValueError, match="tracked changes"):
        L.validate_protocol(protocol, repo_root=repo, isaaclab_root=isaac)


def manifest_fixture(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    run = repo / "bench/private/base_com_reset/tag1"
    run.mkdir(parents=True)
    protocol_path = repo / "bench/protocols/base_com_reset_observation_v3.json"
    protocol_path.parent.mkdir(parents=True)
    protocol_path.write_text("{}", encoding="utf-8")
    manifest = run / "manifest.json"
    manifest.write_text(json.dumps({
        "schema": "base_com_reset_manifest_v1",
        "protocol_path": str(protocol_path.resolve()),
        "protocol_sha256": digest(protocol_path), "arm": "X1",
        "output_path": str((run / "body.json").resolve()),
        "configuration_path": str((run / "configuration.yaml").resolve()),
    }), encoding="utf-8")
    monkeypatch.setattr(L, "validate_protocol", lambda *args, **kwargs: {"source": "ok"})
    monkeypatch.setattr(L.S, "_git_head", lambda root: "f" * 40)
    return repo, run, manifest


def test_manifest_confines_outputs_and_rejects_overwrite(tmp_path, monkeypatch):
    repo, run, manifest = manifest_fixture(tmp_path, monkeypatch)
    validated = L.validate_manifest(manifest.resolve(), repo_root=repo, isaaclab_root=tmp_path)
    assert validated["paths"]["output_path"] == (run / "body.json").resolve()

    (run / "body.json").write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError, match="output_path"):
        L.validate_manifest(manifest.resolve(), repo_root=repo, isaaclab_root=tmp_path)
    assert (run / "body.json").read_text(encoding="utf-8") == "preserve"


def test_manifest_rejects_protocol_copy_and_output_escape(tmp_path, monkeypatch):
    repo, _, manifest = manifest_fixture(tmp_path, monkeypatch)
    body = json.loads(manifest.read_text(encoding="utf-8"))
    copy = tmp_path / "copy.json"
    copy.write_text("{}", encoding="utf-8")
    body["protocol_path"] = str(copy.resolve())
    body["protocol_sha256"] = digest(copy)
    manifest.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(ValueError, match="fixed repository"):
        L.validate_manifest(manifest.resolve(), repo_root=repo, isaaclab_root=tmp_path)

    body["protocol_path"] = str((repo / "bench/protocols/base_com_reset_observation_v3.json").resolve())
    body["protocol_sha256"] = digest(repo / "bench/protocols/base_com_reset_observation_v3.json")
    body["output_path"] = str((tmp_path / "escaped.json").resolve())
    manifest.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(ValueError, match="output_path"):
        L.validate_manifest(manifest.resolve(), repo_root=repo, isaaclab_root=tmp_path)


def body_fixture(tmp_path):
    configuration = tmp_path / "configuration.yaml"
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    report = small_observer_fixture()
    protocol = {
        **small_protocol(), "scope": "scope", "isaacsim_version": "4.5.0",
        "task": L.TASK, "seed": 20261012, "reset_pattern": "alternating_even_odd",
        "device": "cuda:0",
        "measurement_definitions_sha256": MEASUREMENT_DEFINITIONS_SHA256,
        "source_files": {"src/rl_triage/reset_com_observer.py": "d" * 64},
    }
    runtime_contract = {
        key: protocol[key] for key in (
            "task", "seed", "num_envs", "reset_sequences", "resets_per_env",
            "reset_pattern", "device", "allowed_xy_range", "tolerance",
        )
    }
    configuration.write_text(json.dumps({
        "runtime_contract": runtime_contract,
        "injection_readback": L.EXPECTED_INJECTION_READBACK,
        "environment": {"events": {"base_com": {
            "mode": "reset", "params": {
                "com_range": L.EXPECTED_COM_RANGE,
                "asset_cfg": {
                    "name": "robot", "body_names": "base", "body_ids": [None, None, None],
                },
            },
        }}},
        "resolved_event_readback": {
            "source": L.RESOLVED_EVENT_READ_SOURCE,
            "base_com": {"mode": "reset", "params": {
                "com_range": L.EXPECTED_COM_RANGE,
                "asset_cfg": {"name": "robot", "body_names": ["base"], "body_ids": [0]},
            }},
        },
    }), encoding="utf-8")
    receipt = {
        "task_binding_sha256": SHA,
        "isaaclab_source_binding_sha256": "c" * 64,
    }
    report["source_binding"].update({
        "task_sha256": SHA, "configuration_sha256": digest(configuration),
        "isaaclab_sha256": "c" * 64, "observer_sha256": "d" * 64,
        "measurement_definitions_sha256": MEASUREMENT_DEFINITIONS_SHA256,
    })
    reset_rows = []
    episode = [5] * 4
    for sequence in range(4):
        ids = list(range(sequence % 2, 4, 2))
        after_episode = list(episode)
        for env_id in ids:
            after_episode[env_id] = 0
        reset_rows.append({
            "reset_sequence": sequence,
            "expected_env_ids": ids, "sdk_reset_env_ids": ids,
            "observer_reset_env_ids": ids,
            "control_before": 0, "control_after": 0,
            "physics_before": 0, "physics_after": 0,
            "control_delta": 0, "physics_delta": 0,
            "episode_length_before": episode, "episode_length_after": after_episode,
            "reset_buffer_attribute_present_before": False,
            "reset_buffer_attribute_present_after": False,
            "reset_buffer_before": None, "reset_buffer_after": None,
        })
        episode = after_episode
    phase = {
        "phase": "caller_completed_reset_readback", "base_com_active_reset_term": True,
        "active_reset_terms": ["base_com"],
        "initial_sdk_state": {
            "control_step": 0, "physics_step": 0,
            "episode_length": [5] * 4,
            "reset_buffer_attribute_present": False, "reset_buffer": None,
        },
        "final_sdk_state": {
            "control_step": 0, "physics_step": 0,
            "episode_length": [0] * 4,
            "reset_buffer_attribute_present": False, "reset_buffer": None,
        },
        "reset_rows": reset_rows,
    }
    validated = {
        "protocol": protocol, "protocol_sha256": "e" * 64,
        "manifest": {"arm": "X1"}, "manifest_path": manifest,
        "paths": {"configuration_path": configuration}, "source_receipt": receipt,
    }
    execution = {"owned_process_identities": {"321": 1234567}}
    body = {
        "schema": "base_com_reset_body_v1", "completed": True,
        "scope": "scope", "task": L.TASK, "arm": "X1",
        "protocol_sha256": "e" * 64, "manifest_sha256": digest(manifest),
        "project_commit_sha": "f" * 40,
        "process_id": 321, "process_creation_time_filetime": 1234567,
        "source_receipt": receipt,
        "configuration": {"sha256": digest(configuration)},
        "injection_readback": deepcopy(L.EXPECTED_INJECTION_READBACK),
        "observer_report": report, "phase_provenance": phase,
        "sdk_versions": {"isaaclab_commit": L.ISAACLAB_COMMIT, "isaacsim": "4.5.0"},
        "model_selected": False, "diagnosis_confirmed": False,
        "cause_confirmed": False, "recovery_confirmed": False, "cleanup_confirmed": False,
    }
    return body, validated, execution


def rebind_configuration(body, validated):
    configuration = validated["paths"]["configuration_path"]
    configuration_sha = digest(configuration)
    body["configuration"]["sha256"] = configuration_sha
    body["observer_report"]["source_binding"]["configuration_sha256"] = configuration_sha


def test_body_verification_rejects_pid_reuse_and_configuration_drift(tmp_path):
    body, validated, execution = body_fixture(tmp_path)
    assert L.verify_body(body, validated, execution, "f" * 40) == []

    reused = deepcopy(execution)
    reused["owned_process_identities"]["321"] = 7654321
    assert "body_pid_creation_not_observed_in_owned_job" in L.verify_body(
        body, validated, reused, "f" * 40,
    )
    body["configuration"]["sha256"] = "0" * 64
    assert "body_configuration_sha_mismatch" in L.verify_body(
        body, validated, execution, "f" * 40,
    )


def test_body_verification_rejects_sdk_reset_id_and_clock_tamper(tmp_path):
    body, validated, execution = body_fixture(tmp_path)
    body["phase_provenance"]["reset_rows"][0]["sdk_reset_env_ids"] = [1, 3]
    assert "phase_reset_sequence_mismatch" in L.verify_body(
        body, validated, execution, "f" * 40,
    )

    body, validated, execution = body_fixture(tmp_path)
    row = body["phase_provenance"]["reset_rows"][0]
    row["control_after"] = 2
    row["control_delta"] = 2
    assert "phase_reset_sequence_mismatch" in L.verify_body(
        body, validated, execution, "f" * 40,
    )

    body, validated, execution = body_fixture(tmp_path)
    body["phase_provenance"]["reset_rows"][0]["episode_length_after"] = [999] * 4
    assert "phase_reset_sequence_mismatch" in L.verify_body(
        body, validated, execution, "f" * 40,
    )

    body, validated, execution = body_fixture(tmp_path)
    body["phase_provenance"]["reset_rows"][1]["episode_length_before"] = [0] * 4
    assert "phase_reset_sequence_mismatch" in L.verify_body(
        body, validated, execution, "f" * 40,
    )

    body, validated, execution = body_fixture(tmp_path)
    body["phase_provenance"]["final_sdk_state"]["physics_step"] = 1
    assert "phase_final_state_mismatch" in L.verify_body(
        body, validated, execution, "f" * 40,
    )


def test_body_verification_rejects_forged_reset_buffer_presence(tmp_path):
    body, validated, execution = body_fixture(tmp_path)
    initial = body["phase_provenance"]["initial_sdk_state"]
    initial["reset_buffer_attribute_present"] = True
    initial["reset_buffer"] = [False] * 4
    assert "phase_sdk_state_mismatch" in L.verify_body(
        body, validated, execution, "f" * 40,
    )

    body, validated, execution = body_fixture(tmp_path)
    row = body["phase_provenance"]["reset_rows"][0]
    row["reset_buffer_attribute_present_after"] = False
    row["reset_buffer_after"] = [False] * 4
    assert "phase_reset_sequence_mismatch" in L.verify_body(
        body, validated, execution, "f" * 40,
    )

    body, validated, execution = body_fixture(tmp_path)
    body["phase_provenance"]["reset_rows"][0]["control_delta"] = 1
    assert "phase_reset_sequence_mismatch" in L.verify_body(
        body, validated, execution, "f" * 40,
    )


def test_parent_reaggregation_rejects_numeric_overflow():
    report = small_observer_fixture()
    report["baseline_base_xy"][0]["x"] = 10**10000
    assert "baseline_rows_snapshot_mismatch" in L.verify_observer_report(report, small_protocol())


def test_parent_rejects_forged_baseline_hash_body_binding_and_snapshot_xy(tmp_path):
    body, validated, execution = body_fixture(tmp_path)
    body["observer_report"]["source_binding"]["baseline_all_env_all_body_com_sha256"] = "0" * 64
    assert "observer_source_binding_mismatch" in L.verify_body(
        body, validated, execution, "f" * 40,
    )

    body, validated, execution = body_fixture(tmp_path)
    body["observer_report"]["body_binding"]["base_body_index"] = 1
    assert "observer_body_binding_mismatch" in L.verify_body(
        body, validated, execution, "f" * 40,
    )

    body, validated, execution = body_fixture(tmp_path)
    body["observer_report"]["baseline_snapshot"]["coms"][0][0][0] = 0.25
    reasons = L.verify_body(body, validated, execution, "f" * 40)
    assert "observer_source_binding_mismatch" in reasons
    assert "baseline_rows_snapshot_mismatch" in reasons


@pytest.mark.parametrize("mutate, reason", [
    (lambda report: report["measurement_definitions"].clear(),
     "observer_measurement_definitions_mismatch"),
    (lambda report: report["sample_field_definitions"].clear(),
     "observer_sample_field_definitions_mismatch"),
    (lambda report: report["caller_provenance"].update(
        baseline_capture_timing_verified_by_observer=True),
     "observer_caller_provenance_mismatch"),
    (lambda report: report["threshold"].update(tolerance=0.0),
     "observer_threshold_mismatch"),
    (lambda report: report["measurements"].update(extra={}),
     "aggregate_measurement_mismatch"),
])
def test_parent_rejects_forged_observer_contract_fields(mutate, reason):
    report = small_observer_fixture()
    mutate(report)
    assert reason in L.verify_observer_report(report, small_protocol())


def test_parent_rejects_malformed_configuration_and_injection_mismatch(tmp_path):
    body, validated, execution = body_fixture(tmp_path)
    validated["paths"]["configuration_path"].write_text(
        "!!python/object/new:os.system ['echo unsafe']\n", encoding="utf-8",
    )
    assert "configuration_yaml_invalid" in L.verify_body(
        body, validated, execution, "f" * 40,
    )

    body, validated, execution = body_fixture(tmp_path)
    body["injection_readback"]["checks"]["base_com_in_reset"] = False
    assert "body_injection_readback_mismatch" in L.verify_body(
        body, validated, execution, "f" * 40,
    )

    body, validated, execution = body_fixture(tmp_path)
    config = json.loads(validated["paths"]["configuration_path"].read_text(encoding="utf-8"))
    config["runtime_contract"]["seed"] = 0
    validated["paths"]["configuration_path"].write_text(json.dumps(config), encoding="utf-8")
    assert "configuration_runtime_contract_mismatch" in L.verify_body(
        body, validated, execution, "f" * 40,
    )

    body, validated, execution = body_fixture(tmp_path)
    config = json.loads(validated["paths"]["configuration_path"].read_text(encoding="utf-8"))
    config["injection_readback"]["checks"]["base_com_in_reset"] = False
    validated["paths"]["configuration_path"].write_text(json.dumps(config), encoding="utf-8")
    assert "configuration_injection_readback_mismatch" in L.verify_body(
        body, validated, execution, "f" * 40,
    )

    body, validated, execution = body_fixture(tmp_path)
    config = json.loads(validated["paths"]["configuration_path"].read_text(encoding="utf-8"))
    config["environment"]["events"]["base_com"]["params"]["asset_cfg"]["body_names"] = ["leg"]
    validated["paths"]["configuration_path"].write_text(json.dumps(config), encoding="utf-8")
    assert "configuration_base_com_contract_mismatch" in L.verify_body(
        body, validated, execution, "f" * 40,
    )


@pytest.mark.parametrize("body_names, body_ids", [
    (["base"], [1]),
    (["base"], [False]),
    (["base"], [0, 1]),
    (["leg"], [0]),
])
def test_parent_rejects_wrong_resolved_base_binding(tmp_path, body_names, body_ids):
    body, validated, execution = body_fixture(tmp_path)
    path = validated["paths"]["configuration_path"]
    config = json.loads(path.read_text(encoding="utf-8"))
    asset = config["resolved_event_readback"]["base_com"]["params"]["asset_cfg"]
    asset["body_names"] = body_names
    asset["body_ids"] = body_ids
    path.write_text(json.dumps(config), encoding="utf-8")
    rebind_configuration(body, validated)
    assert "configuration_base_com_contract_mismatch" in L.verify_body(
        body, validated, execution, "f" * 40,
    )


def test_post_init_yaml_resolved_base_binding_and_safe_python_sequence_tags(tmp_path):
    body, validated, execution = body_fixture(tmp_path)
    path = validated["paths"]["configuration_path"]
    config = json.loads(path.read_text(encoding="utf-8"))
    term = config["environment"]["events"]["base_com"]
    term["params"]["com_range"] = {
        axis: tuple(bounds) for axis, bounds in L.EXPECTED_COM_RANGE.items()
    }
    term["params"]["asset_cfg"]["body_ids"] = slice(None)
    path.write_text(yaml.dump(config, sort_keys=True), encoding="utf-8")
    rebind_configuration(body, validated)

    assert L.verify_body(body, validated, execution, "f" * 40) == []


def test_parent_rejects_missing_or_wrong_resolved_event_snapshot(tmp_path):
    body, validated, execution = body_fixture(tmp_path)
    path = validated["paths"]["configuration_path"]
    config = json.loads(path.read_text(encoding="utf-8"))
    del config["resolved_event_readback"]
    path.write_text(json.dumps(config), encoding="utf-8")
    rebind_configuration(body, validated)
    assert "configuration_base_com_contract_mismatch" in L.verify_body(
        body, validated, execution, "f" * 40,
    )

    body, validated, execution = body_fixture(tmp_path)
    path = validated["paths"]["configuration_path"]
    config = json.loads(path.read_text(encoding="utf-8"))
    config["resolved_event_readback"]["base_com"]["mode"] = "interval"
    path.write_text(json.dumps(config), encoding="utf-8")
    rebind_configuration(body, validated)
    assert "configuration_base_com_contract_mismatch" in L.verify_body(
        body, validated, execution, "f" * 40,
    )


def test_parent_missing_output_writes_invalid_receipt_without_quality_claim(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    isaac = tmp_path / "isaac"
    protocol = repo / "bench/protocols/base_com_reset_observation_v3.json"
    protocol.parent.mkdir(parents=True)
    protocol.write_text(json.dumps({
        "scope": "scope", "launcher_watchdog_s": 1,
        "gpu_admission_profile": "base_com_reset_readback_v2",
    }), encoding="utf-8")
    monkeypatch.setattr(L, "ROOT", repo)
    monkeypatch.setattr(L, "protocol_location", lambda path, repo_root=repo: protocol.resolve())
    monkeypatch.setattr(L, "validate_protocol", lambda *args, **kwargs: {"source": "ok"})
    monkeypatch.setattr(L.P, "verify_committed", lambda *args: "f" * 40)
    gpu = {"busy": False, "processes": []}
    queried_profiles = []
    monkeypatch.setattr(
        L.M, "query_gpu",
        lambda profile: queried_profiles.append(profile) or deepcopy(gpu),
    )

    def validate(manifest_path, **kwargs):
        manifest = L.S._read_json(manifest_path)
        folder = manifest_path.parent
        return {
            "protocol": {
                "scope": "scope", "launcher_watchdog_s": 1,
                "gpu_admission_profile": "base_com_reset_readback_v2",
            },
            "protocol_sha256": manifest["protocol_sha256"], "manifest": manifest,
            "manifest_path": manifest_path,
            "paths": {"output_path": folder / "body.json", "configuration_path": folder / "configuration.yaml"},
            "source_receipt": {"source": "ok"},
        }

    monkeypatch.setattr(L, "validate_manifest", validate)
    execution = {
        "job_assigned_before_release": True, "natural_exit": True, "exit_code": 0,
        "cleanup_verified": True, "timed_out": False, "error_type": None,
        "owned_process_identities": {},
    }
    monkeypatch.setattr(L.P, "run_owned", lambda *args, **kwargs: deepcopy(execution))
    receipt = L.run("missing_body", "X1", isaac, protocol)

    assert receipt["status"] == "invalid"
    assert queried_profiles == ["base_com_reset_readback_v2", "base_com_reset_readback_v2"]
    assert receipt["validation_reasons"] == ["body_validation_FileNotFoundError"]
    assert receipt["diagnosis_or_recovery_evidence"] is False
    stored = json.loads((repo / "bench/private/base_com_reset/missing_body/parent_receipt.json").read_text())
    assert stored["status"] == "invalid"
    with pytest.raises(FileExistsError, match="existing run"):
        L.run("missing_body", "X1", isaac, protocol)
