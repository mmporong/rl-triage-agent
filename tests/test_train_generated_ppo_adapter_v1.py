# pyright: reportMissingImports=false
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "bench" / "train_generated_ppo_adapter_v1.py"
HEALTHY = "def build_algorithm_kwargs(config):\n    result = dict(config)\n    return result\n"
FAULT = (
    "def build_algorithm_kwargs(config):\n"
    "    result = dict(config)\n"
    "    result['num_learning_epochs'] = 1\n"
    "    return result\n"
)
PATCHED = (
    "def build_algorithm_kwargs(config):\n"
    "    result = dict(config)\n"
    "    result['num_learning_epochs'] = config['num_learning_epochs']\n"
    "    return result\n"
)


def _load():
    name = "train_generated_ppo_adapter_v1_under_test"
    spec = importlib.util.spec_from_file_location(name, SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _digest(source: str) -> str:
    return hashlib.sha256(source.encode()).hexdigest()


def _shape(module, *, schema=None, arm="healthy", source=HEALTHY):
    schema = schema or module.TRAINING_SCHEMA
    return {
        "schema": schema,
        "arm": arm,
        "seed": 2311,
        "task": module.TASK,
        "adapter_source": source,
        "adapter_source_sha256": _digest(source),
        "source_commit": "a" * 40,
        "source_sha256_lf": {name: "b" * 64 for name in module.REQUIRED_SOURCES},
        "protocol": {"relative_path": "bench/protocols/example.json", "sha256_lf": "c" * 64},
        "training": dict(module.TRAINING_BY_SCHEMA[schema]),
    }


def test_import_does_not_load_sdk_modules():
    watched = {"torch", "gymnasium", "isaaclab", "isaaclab_tasks", "carb", "omni"}
    before = {name for name in sys.modules if name.split(".", 1)[0] in watched}
    module = _load()
    after = {name for name in sys.modules if name.split(".", 1)[0] in watched}
    assert module.TASK.endswith("Unitree-Go2-v0")
    assert after == before


@pytest.mark.parametrize(
    ("arm", "source", "epochs"),
    [("healthy", HEALTHY, 5), ("fault", FAULT, 1), ("patched", PATCHED, 5)],
)
def test_adapter_arm_semantics_are_checked(arm, source, epochs):
    module = _load()
    validated = module.validate_spec_shape(_shape(module, arm=arm, source=source))
    result = module._apply_adapter_checked(source, module.ADAPTER_PROBE_CONFIG, arm)
    assert validated["arm"] == arm
    assert result["num_learning_epochs"] == epochs
    assert {key: value for key, value in result.items() if key != "num_learning_epochs"} == {
        key: value for key, value in module.ADAPTER_PROBE_CONFIG.items()
        if key != "num_learning_epochs"
    }


def test_smoke_is_a_separate_one_update_schema():
    module = _load()
    value = module.validate_spec_shape(_shape(module, schema=module.SMOKE_SCHEMA))
    assert value["training"] == {"num_envs": 4096, "updates": 1}


def test_task_term_hash_handles_callables_without_memory_addresses():
    module = _load()

    def reward_term():
        return None

    left = {"term": {"func": reward_term, "params": [1.0, True]}}
    right = {"term": {"params": [1.0, True], "func": reward_term}}
    assert module._json_sha256(left) == module._json_sha256(right)


def test_task_term_hash_rejects_unknown_object_instead_of_using_repr():
    module = _load()
    with pytest.raises(TypeError, match="지원하지 않는 형식"):
        module._json_sha256({"term": object()})


@pytest.mark.parametrize(
    "mutate",
    [
        lambda module, value: value.update(training={"num_envs": 4096, "updates": 2}),
        lambda module, value: value.update(task="other"),
        lambda module, value: value.update(seed=True),
        lambda module, value: value.update(adapter_source_sha256="0" * 64),
        lambda module, value: value["source_sha256_lf"].pop(next(iter(module.REQUIRED_SOURCES))),
    ],
)
def test_shape_rejects_unregistered_variations(mutate):
    module = _load()
    value = _shape(module)
    mutate(module, value)
    with pytest.raises((module.TrainingContractError, module.CONTRACT.PatchContractError)):
        module.validate_spec_shape(value)


def test_adapter_cannot_change_other_algorithm_values():
    module = _load()
    source = (
        "def build_algorithm_kwargs(config):\n"
        "    result = dict(config)\n"
        "    result['num_learning_epochs'] = 1\n"
        "    result['learning_rate'] = 0.5\n"
        "    return result\n"
    )
    with pytest.raises(module.CONTRACT.PatchContractError):
        module.validate_spec_shape(_shape(module, arm="fault", source=source))


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()


def test_validate_spec_binds_current_files_commit_blobs_and_protocol(tmp_path):
    module = _load()
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Test")
    value = _shape(module)
    protocol_path = repo / value["protocol"]["relative_path"]
    protocol_path.parent.mkdir(parents=True)
    protocol = {
        "schema": module.PROTOCOL_SCHEMA,
        "jobs": [{
            "arm": value["arm"],
            "seed": value["seed"],
            "adapter_source_sha256": value["adapter_source_sha256"],
            "training": value["training"],
        }],
    }
    protocol_path.write_text(json.dumps(protocol) + "\n", encoding="utf-8")
    value["protocol"]["sha256_lf"] = module._sha256_lf(protocol_path)
    for relative in module.REQUIRED_SOURCES:
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fixture for {relative}\n", encoding="utf-8")
        value["source_sha256_lf"][relative] = module._sha256_lf(path)
    _git(repo, "add", *sorted([value["protocol"]["relative_path"], *module.REQUIRED_SOURCES]))
    _git(repo, "commit", "-m", "fixture")
    value["source_commit"] = _git(repo, "rev-parse", "HEAD")
    assert module.validate_spec(value, repo=repo)["seed"] == 2311
    changed = repo / next(iter(module.REQUIRED_SOURCES))
    changed.write_text("changed\n", encoding="utf-8")
    with pytest.raises(module.TrainingContractError, match="현재 파일 SHA"):
        module.validate_spec(value, repo=repo)


def test_preparse_requires_headless_and_new_output(tmp_path):
    module = _load()
    spec = tmp_path / "spec.json"
    spec.write_text("{}\n", encoding="utf-8")
    out = tmp_path / "out"
    args = module._preparse(["--spec", str(spec), "--out", str(out), "--headless"])
    assert args.headless is True
    out.mkdir()
    with pytest.raises(module.TrainingContractError, match="이미 존재"):
        module._preparse(["--spec", str(spec), "--out", str(out), "--headless"])


def test_launcher_arguments_have_one_fixed_kit_args_value():
    module = _load()
    value = module.launcher_arguments(
        Namespace(spec=Path("s"), out=Path("o"), headless=True, kit_args="old", fast_shutdown=True)
    )
    assert value["headless"] is True
    assert value["kit_args"] == "--/app/vulkan=false --/app/window/hideUi=true"
    assert value["fast_shutdown"] is False


@pytest.mark.parametrize(
    ("epochs", "minibatches"), [(5, 4), (1, 4)]
)
def test_runtime_parameters_match_adapter_fields_separately(epochs, minibatches):
    module = _load()
    expected = {"num_learning_epochs": epochs, "num_mini_batches": minibatches}
    module.verify_runtime_parameters(Namespace(**expected), expected)


@pytest.mark.parametrize(
    ("epochs", "minibatches"), [(4, 5), (10, 2), (True, 20), (5, 4.0)]
)
def test_runtime_parameters_reject_wrong_pair_even_with_same_step_product(epochs, minibatches):
    module = _load()
    with pytest.raises(module.TrainingContractError, match="실제 PPO"):
        module.verify_runtime_parameters(
            Namespace(num_learning_epochs=epochs, num_mini_batches=minibatches),
            {"num_learning_epochs": 5, "num_mini_batches": 4},
        )
