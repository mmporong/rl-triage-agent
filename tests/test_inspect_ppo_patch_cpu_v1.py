from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "bench" / "inspect_ppo_patch_cpu_v1.py"


def _load():
    name = "inspect_ppo_patch_cpu_v1_under_test"
    spec = importlib.util.spec_from_file_location(name, SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _config(**changes):
    value = {
        "num_mini_batches": 4,
        "learning_rate": 0.001,
        "schedule": "fixed",
    }
    value.update(changes)
    return value


def test_import_does_not_load_sdk_or_simulation_modules():
    watched = ("torch", "rsl_rl", "isaac", "carb", "omni")
    before = {name for name in sys.modules if name.split(".", 1)[0] in watched}
    module = _load()
    after = {name for name in sys.modules if name.split(".", 1)[0] in watched}
    assert module.SCHEMA == "rsl_ppo_patch_cpu_inspection_v1"
    assert after == before


def test_missing_epochs_is_preserved_for_runtime_signature_resolution():
    module = _load()
    assert module.validate_config(_config()) == _config()


@pytest.mark.parametrize("epochs", [1, 4, 8])
def test_explicit_epoch_bounds_are_retained(epochs):
    module = _load()
    assert module.validate_config(_config(num_learning_epochs=epochs))["num_learning_epochs"] == epochs


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"num_learning_epochs": 0}, "num_learning_epochs"),
        ({"num_learning_epochs": 9}, "num_learning_epochs"),
        ({"num_learning_epochs": True}, "num_learning_epochs"),
        ({"num_mini_batches": 0}, "num_mini_batches"),
        ({"num_mini_batches": 5}, "num_mini_batches"),
        ({"learning_rate": 0.0}, "learning_rate"),
        ({"schedule": "adaptive"}, "schedule"),
        ({"unknown": 1}, "unsupported"),
    ],
)
def test_invalid_config_is_rejected(changes, message):
    module = _load()
    with pytest.raises(module.InspectionError, match=message):
        module.validate_config(_config(**changes))


def test_updates_are_bounded_before_sdk_import(monkeypatch):
    module = _load()
    monkeypatch.setattr(module, "_loaded_simulation_modules", list)
    for value in (0, 4, True):
        with pytest.raises(module.InspectionError, match="updates"):
            module.inspect(_config(), updates=value)


def test_pure_validation_does_not_consult_runtime(monkeypatch):
    module = _load()

    def forbidden():
        raise AssertionError("runtime boundary crossed")

    monkeypatch.setattr(module, "_loaded_simulation_modules", forbidden)
    assert module.validate_config(_config(num_learning_epochs=5))["num_learning_epochs"] == 5
