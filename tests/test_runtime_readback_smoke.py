import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from rl_triage import runtime_observers as R
from rl_triage.runtime_observers import RuntimeReadbackCollector


ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location(
    "runtime_readback_smoke_test", ROOT / "bench/runtime_readback_smoke.py",
)
assert SPEC is not None and SPEC.loader is not None
SMOKE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SMOKE)
SHA = "a" * 64


class Values:
    def __init__(self, values):
        self.values = values

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.values


class RewardManager:
    def get_active_iterable_terms(self, env_idx):
        return [("tracking", [1.0 + env_idx])]


class FakeEnv:
    def __init__(self, *, bad_physics_clock=False):
        self.num_envs = 16
        self.bad_physics_clock = bad_physics_clock
        self.step_count = 0
        self.unwrapped = SimpleNamespace(
            num_envs=16,
            common_step_counter=0,
            _sim_step_counter=0,
            cfg=SimpleNamespace(decimation=4),
            scene={
                "robot": SimpleNamespace(data=SimpleNamespace(
                    root_com_pos_w=[[0.0, 0.0, 0.5] for _ in range(16)],
                    root_com_lin_vel_w=[[0.0, 0.0, 0.0] for _ in range(16)],
                    body_com_pos_b=[[[0.0, 0.0, 0.1]] for _ in range(16)],
                )),
                "contact_forces": SimpleNamespace(data=SimpleNamespace(
                    net_forces_w=[[[0.0, 0.0, 1.0]] for _ in range(16)],
                )),
            },
            reward_manager=RewardManager(),
        )

    def get_observations(self):
        return Values([[0.0] for _ in range(16)]), {}

    def step(self, actions):
        assert actions == "actions"
        self.step_count += 1
        self.unwrapped.common_step_counter += 1
        self.unwrapped._sim_step_counter += 3 if self.bad_physics_clock else 4
        dones = [0] * 16
        if self.step_count == 1:
            dones[0] = 1
        return Values([[0.0] for _ in range(16)]), Values([float(self.step_count)] * 16), Values(dones), {}


def protocol(steps=2):
    return {
        "num_envs": 16,
        "observed_env_indices": [0, 1, 2, 3],
        "completed_control_steps": steps,
    }


def collector(env, steps=2):
    return RuntimeReadbackCollector(
        env.unwrapped,
        observed_env_indices=[0, 1, 2, 3],
        max_completed_control_samples=steps,
        source_binding={
            "task_sha256": SHA,
            "checkpoint_sha256": "b" * 64,
            "configuration_sha256": "c" * 64,
            "observer_sha256": "d" * 64,
        },
    )


def test_completed_step_reward_clocks_and_auto_reset_phase_are_bound():
    env = FakeEnv()
    readback = collector(env)
    receipt = SMOKE.collect_window(env, lambda obs: "actions", readback, protocol())
    result = readback.aggregate()

    assert receipt["expected_control_delta"] == 1
    assert receipt["expected_physics_delta"] == 4
    assert [row["physics_after"] for row in receipt["clock_rows"]] == [4, 8]
    assert receipt["phase_counts_over_observed_environments"] == {
        "completed_control_no_reset": 7,
        "post_auto_reset": 1,
    }
    first_reset = result["samples"][0]["environments"][0]
    assert (first_reset["episode_id"], first_reset["phase"], first_reset["reset_occurred"]) == (
        0, "post_auto_reset", True,
    )
    assert first_reset["measurements"]["completed_reward"]["value"] == 1.0
    assert first_reset["measurements"]["contact_force_max"]["state"] == "unavailable"
    second_episode = result["samples"][1]["environments"][0]
    assert (second_episode["episode_id"], second_episode["phase"]) == (
        1, "completed_control_no_reset",
    )
    assert result["scope"]["control_step_indices"] == [1, 2]
    assert result["scope"]["actual_physics_step_indices"] == [4, 8]
    verdict = SMOKE.assess_smoke(result, receipt, protocol())
    assert verdict == {
        "smoke_status": "pass",
        "validation_reasons": [],
        "expected_completed_control_samples": 2,
        "expected_value_count_per_measurement": 8,
        "post_auto_reset_value_count": 1,
        "known_post_auto_reset_contact_unavailable_is_preserved": True,
        "quality_or_recovery_judgment": False,
    }


def test_malformed_sdk_clock_fails_before_a_sample_is_accepted():
    env = FakeEnv(bad_physics_clock=True)
    readback = collector(env, steps=1)
    with pytest.raises(RuntimeError, match="cfg.decimation"):
        SMOKE.collect_window(env, lambda obs: "actions", readback, protocol(steps=1))
    assert readback.sample_count == 0


def test_malformed_wrapper_done_value_is_rejected():
    env = FakeEnv()
    original_step = env.step

    def malformed_step(actions):
        obs, rewards, dones, extras = original_step(actions)
        dones.values[0] = "reset"
        return obs, rewards, dones, extras

    env.step = malformed_step
    readback = collector(env, steps=1)
    with pytest.raises(ValueError, match=r"done\[0\]"):
        SMOKE.collect_window(env, lambda obs: "actions", readback, protocol(steps=1))
    assert readback.sample_count == 0


def test_missing_api_value_makes_smoke_invalid_without_converting_it_to_zero():
    env = FakeEnv()
    env.unwrapped.scene["robot"].data.root_com_pos_w[0][2] = None
    readback = collector(env, steps=1)
    receipt = SMOKE.collect_window(env, lambda obs: "actions", readback, protocol(steps=1))
    result = readback.aggregate()
    verdict = SMOKE.assess_smoke(result, receipt, protocol(steps=1))

    assert verdict["smoke_status"] == "invalid"
    assert "com_position_z:nonfinite_or_unavailable" in verdict["validation_reasons"]
    measurement = result["read_only_checks"]["runtime_com_state"]["measurements"]["com_position_z"]
    assert measurement["finite_value_count"] == 3
    assert measurement["unavailable_value_count"] == 1
    assert measurement["unavailable_reason_counts"] == {"null_or_missing": 1}


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_source_mismatch_is_rejected_before_runtime_import(monkeypatch, tmp_path):
    repo = tmp_path / "repo"
    isaac = tmp_path / "isaac"
    for relative in SMOKE.PROJECT_SOURCE_FILES:
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")
    sdk = isaac / "source/sdk.py"
    for relative in SMOKE.ISAACLAB_SOURCE_FILES:
        sdk = isaac / relative
        sdk.parent.mkdir(parents=True, exist_ok=True)
        sdk.write_text(relative, encoding="utf-8")
    source_files = {relative: _digest(repo / relative) for relative in SMOKE.PROJECT_SOURCE_FILES}
    source_files["src/rl_triage/runtime_observers.py"] = "0" * 64
    proto = {
        "schema": "runtime_readback_smoke_protocol_v1",
        "scope": "actual_sdk_api_integration_not_diagnosis_or_recovery",
        "task": SMOKE.TASK,
        "seed": 20261011,
        "num_envs": 16,
        "completed_control_steps": 20,
        "observed_env_indices": [0, 1, 2, 3],
        "device": "cuda:0",
        "isaaclab_commit": SMOKE.ISAACLAB_COMMIT,
        "isaacsim_version": "4.5.0",
        "checkpoint_sha256": SMOKE.CHECKPOINT_SHA256,
        "registry_sha256": SHA,
        "source_files": source_files,
        "isaaclab_source_files": {
            relative: _digest(isaac / relative) for relative in SMOKE.ISAACLAB_SOURCE_FILES
        },
        "launcher_watchdog_s": 360,
        "semantic_retries": 0,
    }
    imported = []
    monkeypatch.setattr(SMOKE, "_load_runtime_observer", lambda: imported.append(True))
    monkeypatch.setattr(SMOKE, "_git_head", lambda path: SMOKE.ISAACLAB_COMMIT)
    monkeypatch.setattr(SMOKE, "_git_tracked_clean", lambda path: True)

    with pytest.raises(ValueError, match="source SHA mismatch"):
        SMOKE._validate_protocol(proto, repo_root=repo, isaaclab_root=isaac)
    assert imported == []


def test_manifest_rejects_output_overwrite_before_protocol_validation(tmp_path):
    repo = tmp_path / "repo"
    protocol_path = repo / "bench/protocols/runtime_readback_smoke_v1.json"
    checkpoint = tmp_path / "model_99.pt"
    run = repo / "bench/private/runtime_readback/test-manifest-overwrite"
    run.mkdir(parents=True, exist_ok=True)
    output = run / "body.json"
    configuration = run / "configuration.yaml"
    manifest = run / "manifest.json"
    protocol_path.parent.mkdir(parents=True, exist_ok=True)
    protocol_path.write_text("{}", encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint")
    output.write_text("preserve", encoding="utf-8")
    body = {
        "schema": "runtime_readback_smoke_manifest_v1",
        "protocol_path": str(protocol_path.resolve()),
        "protocol_sha256": _digest(protocol_path),
        "checkpoint_path": str(checkpoint.resolve()),
        "output_path": str(output.resolve()),
        "configuration_path": str(configuration.resolve()),
    }
    manifest.write_text(json.dumps(body), encoding="utf-8")

    with pytest.raises(FileExistsError, match="output_path"):
        SMOKE.validate_manifest(manifest, repo_root=repo)
    assert output.read_text(encoding="utf-8") == "preserve"


def test_manifest_rejects_protocol_outside_fixed_public_path(tmp_path):
    repo = tmp_path / "repo"
    run = repo / "bench/private/runtime_readback/tag1"
    run.mkdir(parents=True)
    wrong_protocol = tmp_path / "copy.json"
    wrong_protocol.write_text("{}", encoding="utf-8")
    checkpoint = tmp_path / "model_99.pt"
    checkpoint.write_bytes(b"checkpoint")
    manifest = run / "manifest.json"
    manifest.write_text(json.dumps({
        "schema": "runtime_readback_smoke_manifest_v1",
        "protocol_path": str(wrong_protocol.resolve()),
        "protocol_sha256": _digest(wrong_protocol),
        "checkpoint_path": str(checkpoint.resolve()),
        "output_path": str((run / "body.json").resolve()),
        "configuration_path": str((run / "configuration.yaml").resolve()),
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="repository runtime readback smoke protocol"):
        SMOKE.validate_manifest(manifest, repo_root=repo)


def test_tracked_sdk_change_fails_protocol_preflight(monkeypatch, tmp_path):
    proto = {
        "schema": "runtime_readback_smoke_protocol_v1",
        "scope": "actual_sdk_api_integration_not_diagnosis_or_recovery",
        "task": SMOKE.TASK,
        "seed": 20261011,
        "num_envs": 16,
        "completed_control_steps": 20,
        "observed_env_indices": [0, 1, 2, 3],
        "device": "cuda:0",
        "isaaclab_commit": SMOKE.ISAACLAB_COMMIT,
        "isaacsim_version": "4.5.0",
        "checkpoint_sha256": SMOKE.CHECKPOINT_SHA256,
        "registry_sha256": R.REGISTRY_SHA256,
        "source_files": {name: SHA for name in SMOKE.PROJECT_SOURCE_FILES},
        "isaaclab_source_files": {name: SHA for name in SMOKE.ISAACLAB_SOURCE_FILES},
        "launcher_watchdog_s": 360,
        "semantic_retries": 0,
    }
    monkeypatch.setattr(
        SMOKE, "_verify_source_map",
        lambda root, value, label, exact_paths=None: dict(value),
    )
    monkeypatch.setattr(SMOKE, "_git_head", lambda path: SMOKE.ISAACLAB_COMMIT)
    monkeypatch.setattr(SMOKE, "_git_tracked_clean", lambda path: False)

    with pytest.raises(ValueError, match="tracked changes"):
        SMOKE._validate_protocol(proto, repo_root=tmp_path, isaaclab_root=tmp_path)


def test_no_overwrite_publisher_is_exclusive_and_strict_json_ready(tmp_path):
    output = tmp_path / "body.json"
    SMOKE._publish_bytes_no_overwrite(output, b'{"completed":true}\n')
    assert json.loads(output.read_text(encoding="utf-8"))["completed"] is True
    with pytest.raises(FileExistsError):
        SMOKE._publish_bytes_no_overwrite(output, b"replacement")
    assert output.read_bytes() == b'{"completed":true}\n'


def test_process_identity_binds_pid_and_injected_windows_filetime(monkeypatch):
    monkeypatch.setattr(SMOKE.os, "getpid", lambda: 4321)
    assert SMOKE.process_identity(creation_time_reader=lambda: 987654321) == {
        "process_id": 4321,
        "process_creation_time_filetime": 987654321,
    }
    with pytest.raises(ValueError, match="positive integer"):
        SMOKE.process_identity(creation_time_reader=lambda: 0)
