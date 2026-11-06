from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
import train_generated_ppo_adapter_v2 as V2

PUBLIC = ROOT / "evals/results/diagnostic_generated_patch_cpu_20261106_v1"
FAULT = json.loads((PUBLIC / "source-cases.json").read_text())["variants"]["case-b"]
PATCH = json.loads((PUBLIC / "runs/api/case-b/revision/response-01.json").read_text())["proposal"]["replacement_source"]
HEALTHY = "def build_algorithm_kwargs(config):\n    options = dict(config)\n    return options\n"


def config():
    return {**V2.V1.ADAPTER_PROBE_CONFIG, "class_name": "PPO", "entropy_coef": 0.01,
            "clip_param": 0.2, "lam": 0.95, "desired_kl": 0.01}


def shape(arm, source):
    return {"schema": V2.TRAINING_SCHEMA, "arm": arm, "seed": 5753, "task": V2.TASK,
            "adapter_source": source, "adapter_source_sha256": V2.V1.CONTRACT.source_sha256(source),
            "source_commit": "a" * 40,
            "source_sha256_lf": {key: "b" * 64 for key in V2.REQUIRED_SOURCES},
            "protocol": {"relative_path": "bench/protocols/test.json", "sha256_lf": "c" * 64},
            "training": {"num_envs": 4096, "updates": 300}}


def test_original_model_patch_and_fault_are_exact_saved_sources():
    assert V2.V1.CONTRACT.source_sha256(PATCH) == "02fa206fbedf0bf374b7bafab534efdb498dfb3e51fbae8a7fbd6ff674e59295"
    assert V2.V1.CONTRACT.source_sha256(FAULT).startswith("c4c710f1")


def test_missing_keys_remain_missing_until_ppo_constructor():
    before = config()
    resolution = V2.adapter_resolution(FAULT, before, "fault")
    assert before == config()
    assert set(resolution["raw_adapter_kwargs"]) == V2.FAULT_KEYS
    assert set(resolution["runner_kwargs"]) == V2.FAULT_KEYS | {"class_name"}
    assert resolution["metadata_restored_by_host"] == ["class_name"]
    assert "num_learning_epochs" not in resolution["runner_kwargs"]
    assert resolution["effective_parameters"]["num_learning_epochs"] == 1
    assert resolution["effective_parameters"]["gamma"] == 0.998
    assert resolution["effective_parameters"]["entropy_coef"] == 0.0


@pytest.mark.parametrize(("arm", "source"), [("healthy", HEALTHY), ("patched", PATCH)])
def test_healthy_and_patch_preserve_every_config_key(arm, source):
    resolution = V2.adapter_resolution(source, config(), arm)
    assert resolution["runner_kwargs"] == config()
    assert resolution["raw_adapter_kwargs"] == config()
    assert resolution["metadata_restored_by_host"] == []
    assert resolution["omitted_algorithm_keys"] == []


@pytest.mark.parametrize(("arm", "source"), [("healthy", HEALTHY), ("fault", FAULT), ("patched", PATCH)])
def test_new_spec_shape_and_scope_restoration(arm, source):
    old_train, old_schema = V2.V1.train, V2.V1.PROTOCOL_SCHEMA
    assert V2.validate_spec_shape(shape(arm, source))["arm"] == arm
    assert V2.V1.train is old_train
    assert V2.V1.PROTOCOL_SCHEMA == old_schema


def test_original_v1_still_rejects_omission():
    with pytest.raises(V2.V1.TrainingContractError, match="key 집합"):
        V2.V1._apply_adapter_checked(FAULT, config(), "fault")


@pytest.mark.parametrize("key", ["learning_rate", "schedule", "num_mini_batches"])
def test_fault_cannot_change_preserved_values(key):
    source = FAULT.replace(f'config["{key}"]', "0")
    with pytest.raises(ValueError, match="전달한 값"):
        V2.adapter_resolution(source, config(), "fault")


def test_patch_cannot_restore_epochs_but_drop_other_keys():
    with pytest.raises(ValueError, match="보존"):
        V2.adapter_resolution(FAULT, config(), "patched")


def test_unknown_default_missing_key_is_rejected():
    before = {**config(), "unregistered_argument": 7}
    with pytest.raises(ValueError, match="기본값이 없는"):
        V2.adapter_resolution(FAULT, before, "fault")


def test_real_ppo_parameters_check_values_beyond_step_count():
    raw = V2.adapter_resolution(FAULT, config(), "fault")["runner_kwargs"]
    effective = V2.effective_parameters(raw)
    V2.verify_runtime_parameters(SimpleNamespace(**effective), raw)
    changed = copy.deepcopy(effective)
    changed["gamma"] = 0.99
    with pytest.raises(ValueError, match="gamma"):
        V2.verify_runtime_parameters(SimpleNamespace(**changed), raw)


def test_scope_restores_on_exception_and_rejects_nested_replacement():
    original = V2.V1.train
    with pytest.raises(RuntimeError), V2.training_scope():
        assert V2.V1.train is V2.train
        with pytest.raises(ValueError, match="이미 교체"), V2.training_scope():
            pytest.fail("nested scope must be refused")
        raise RuntimeError("fixture")
    assert V2.V1.train is original


def test_constructor_default_mismatch_is_rejected():
    class WrongPPO:
        def __init__(self, policy, num_learning_epochs=5):
            self.policy = policy

    with pytest.raises(ValueError, match="생성자 기본값"):
        V2.verify_constructor(WrongPPO)
