"""R1 epoch correction that preserves the registered source reward environment."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import re
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _folder in (ROOT, ROOT / "bench", ROOT / "evals", ROOT / "src"):
    if str(_folder) not in sys.path:
        sys.path.insert(0, str(_folder))

from bench import correction_adapter as ADAPTER
from bench import hidden_faults as FAULTS
from bench import r1_repair_training as BASE

SCHEMA = "registered_epoch_environment_preservation_v1"
EXPECTED_SCHEMA = "registered_epoch_reward_recipe_v1"
PROVENANCE_KIND = "registered_source_recipe_not_historical_runtime_observation"
MODES = {"nominal_body_frame", "world_frame_reward"}
TARGET = {
    "module": "isaaclab.envs.mdp.rewards",
    "qualname": "track_lin_vel_xy_exp",
    "source": "source/isaaclab/isaaclab/envs/mdp/rewards.py",
}
SECONDARY_TARGET = {
    "module": "isaaclab.actuators.actuator_pd",
    "qualname": "DCMotor._clip_effort",
    "source": "source/isaaclab/isaaclab/actuators/actuator_pd.py",
}


def _sha(path: Path, *, lf: bool = False) -> str:
    raw = path.read_bytes()
    if lf:
        raw = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(raw).hexdigest()


def _hex(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("JSON object required")
    return value


def _write_new(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(dict(value), stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
        stream.write("\n")


def _reference(value: Any, label: str, *, parse_json: bool = True) -> tuple[dict[str, str], dict[str, Any]]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} reference fields")
    raw_path = Path(value["path"])
    if raw_path.is_symlink():
        raise ValueError(f"{label} reference changed")
    path = raw_path.resolve(strict=True)
    if not _hex(value["sha256"]) or _sha(path) != value["sha256"]:
        raise ValueError(f"{label} reference changed")
    body = _read(path) if parse_json else {}
    return {"path": str(path), "sha256": value["sha256"]}, body


def _target(value: Any, identity: Mapping[str, str] = TARGET) -> dict[str, str]:
    fields = {*identity, "source_sha256_lf", "clean_fingerprint", "expected_runtime_fingerprint"}
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("environment target fields")
    target = dict(value)
    if any(target[key] != expected for key, expected in identity.items()):
        raise ValueError("environment target identity")
    if any(not _hex(target[key]) for key in (
            "source_sha256_lf", "clean_fingerprint", "expected_runtime_fingerprint")):
        raise ValueError("environment target fingerprints")
    return target


def validate_environment_binding(value: Mapping[str, Any]) -> dict[str, Any]:
    fields = {"schema", "mode", "source_setup", "source_env_config", "expected_reward",
              "code_contract", "target", "secondary_target"}
    if not isinstance(value, Mapping) or set(value) != fields or value.get("schema") != SCHEMA:
        raise ValueError("environment preservation fields")
    if value["mode"] not in MODES:
        raise ValueError("environment preservation mode")
    target = _target(value["target"])
    secondary_target = _target(value["secondary_target"], SECONDARY_TARGET)
    setup_ref, setup = _reference(value["source_setup"], "source setup")
    env_ref, _ = _reference(value["source_env_config"], "source environment configuration",
                            parse_json=False)
    contract_ref, contract = _reference(value["code_contract"], "environment recipe code contract")
    expected_ref, expected = _reference(value["expected_reward"], "expected reward recipe")
    expected_fields = {"schema", "provenance_kind", "task", "source_arm", "source_seed", "training_seed",
                       "source_checkpoint_sha256", "source_setup_sha256", "source_env_config_sha256",
                       "code_contract_sha256", "mode", "target", "secondary_target"}
    if (set(expected) != expected_fields
            or expected.get("schema") != EXPECTED_SCHEMA
            or expected.get("provenance_kind") != PROVENANCE_KIND):
        raise ValueError("source environment evidence schema")
    allowed_arms = ({"NONE", "R1", "F2"} if value["mode"] == "nominal_body_frame"
                    else {"F1", "COMPOUND"})
    if (expected["source_arm"] not in allowed_arms or setup.get("arm") != expected["source_arm"]
            or setup.get("task") != expected["task"]
            or setup.get("seed") != expected["source_seed"]
            or expected["source_setup_sha256"] != setup_ref["sha256"]
            or expected["source_env_config_sha256"] != env_ref["sha256"]
            or expected["code_contract_sha256"] != contract_ref["sha256"]):
        raise ValueError("source setup is not the registered source recipe")
    if (expected["mode"] != value["mode"] or expected["target"] != target
            or expected["secondary_target"] != secondary_target):
        raise ValueError("environment binding differs from evidence")
    if (value["mode"] == "nominal_body_frame"
            and target["expected_runtime_fingerprint"] != target["clean_fingerprint"]):
        raise ValueError("nominal reward recipe is not clean")
    if (expected["source_arm"] != "F2"
            and secondary_target["expected_runtime_fingerprint"] != secondary_target["clean_fingerprint"]):
        raise ValueError("non-F2 motor recipe is not clean")
    motor_key = "preserve_source_envelope" if expected["source_arm"] == "F2" else "nominal"
    if (contract.get("schema") != "registered_environment_recipe_code_contract_v1"
            or contract.get("provenance_kind") != "compiled_code_contract_not_historical_runtime_observation"
            or contract.get("reward_targets", {}).get(value["mode"]) != target
            or contract.get("motor_targets", {}).get(motor_key) != secondary_target
            or contract.get("runtime", {}).get("implementation") != "CPython"
            or contract.get("runtime", {}).get("version", [])[:2] != [3, 10]
            or contract.get("module_imports") != {"carb": False, "isaaclab": False,
                                                   "omni": False, "torch": False}
            or contract.get("gpu_launches") != 0 or contract.get("model_api_calls") != 0):
        raise ValueError("environment recipe code contract differs")
    source_hashes = setup.get("source_sha256_lf")
    runtime_hashes = setup.get("runtime_source_sha256_lf")
    hidden_relative = "bench/hidden_faults.py"
    if (not isinstance(source_hashes, dict)
            or source_hashes.get(hidden_relative) != contract.get("hidden_recipe_source_sha256_lf")
            or source_hashes.get(hidden_relative) != _sha(ROOT / hidden_relative, lf=True)
            or not isinstance(runtime_hashes, dict)
            or runtime_hashes.get(target["source"]) != target["source_sha256_lf"]
            or runtime_hashes.get(secondary_target["source"]) != secondary_target["source_sha256_lf"]):
        raise ValueError("source setup recipe/runtime source binding differs")
    if (expected["task"] != BASE.TASK
            or any(type(expected[key]) is not int or expected[key] < 0
                   for key in ("source_seed", "training_seed"))
            or not _hex(expected["source_checkpoint_sha256"])):
        raise ValueError("source environment provenance")
    return {"schema": SCHEMA, "mode": value["mode"], "source_setup": setup_ref,
            "source_env_config": env_ref, "code_contract": contract_ref,
            "expected_reward": expected_ref, "target": target,
            "secondary_target": secondary_target,
            "provenance_kind": PROVENANCE_KIND,
            "source_provenance": {key: expected[key] for key in (
                "task", "source_arm", "source_seed", "training_seed", "source_checkpoint_sha256")}}


def _reward_term(env: Any) -> Any:
    cfg = getattr(getattr(env, "unwrapped", env), "cfg", None)
    rewards = getattr(cfg, "rewards", None)
    term = getattr(rewards, "track_lin_vel_xy_exp", None)
    function = getattr(term, "func", None)
    if function is None:
        raise ValueError("runtime reward term is unavailable")
    return function


def _seal_final_readback(
        state: dict[str, Any], binding: Mapping[str, Any], readback_path: Path,
        final_readback_path: Path) -> None:
    """Seal the preserved recipe before Isaac application shutdown can end the process."""
    if state.get("final_readback_written") is True:
        return
    function, motor_class = state.get("function"), state.get("motor_class")
    if (function is None or motor_class is None or state.get("term_function") is None
            or state.get("readback_written") is not True):
        raise ValueError("training environment was not initialized")
    if state.get("learn_calls") != 1 or state.get("learn_returned") is not True:
        raise ValueError("training did not return successfully exactly once")
    final = {
        "schema": "registered_epoch_environment_final_readback_v1",
        "mode": binding["mode"],
        "runtime_fingerprint": ADAPTER.fingerprint(function),
        "target_identity_preserved": state["term_function"] is function,
        "motor_runtime_fingerprint": ADAPTER.fingerprint(motor_class._clip_effort),
        "motor_target_identity_preserved": motor_class._clip_effort is state["original_motor"],
        "initial_readback": {"path": str(readback_path.resolve()), "sha256": _sha(readback_path)},
    }
    expected_motor_identity = binding["source_provenance"]["source_arm"] != "F2"
    if (final["runtime_fingerprint"] != binding["target"]["expected_runtime_fingerprint"]
            or final["target_identity_preserved"] is not True
            or final["motor_runtime_fingerprint"]
            != binding["secondary_target"]["expected_runtime_fingerprint"]
            or final["motor_target_identity_preserved"] is not expected_motor_identity):
        raise ValueError("environment recipe changed during training")
    _write_new(final_readback_path, final)
    state["final_readback_written"] = True
    state["final_readback"] = final


def _install_close_seal(
        env: Any, state: dict[str, Any], binding: Mapping[str, Any], readback_path: Path,
        final_readback_path: Path) -> None:
    """Seal once immediately before delegating to the environment's original close method."""
    original_close = getattr(env, "close", None)
    if not callable(original_close):
        raise TypeError("training environment close is unavailable")
    namespace = getattr(env, "__dict__", None)
    if isinstance(namespace, dict):
        had_instance_close = "close" in namespace
        instance_close = namespace.get("close")
    else:
        had_instance_close = False
        instance_close = None

    def close(*args, **kwargs):
        if state["close_calls"] != 0:
            raise ValueError("training environment closed more than once")
        state["close_calls"] += 1
        _seal_final_readback(state, binding, readback_path, final_readback_path)
        return original_close(*args, **kwargs)

    env.close = close
    state.update({"environment": env, "original_close": original_close,
                  "had_instance_close": had_instance_close, "instance_close": instance_close})


def _install_learn_observer(state: dict[str, Any]) -> None:
    runners = importlib.import_module("rsl_rl.runners")
    runner_class = runners.OnPolicyRunner
    original_learn = runner_class.learn

    def learn(self, *args, **kwargs):
        if state["learn_calls"] != 0:
            raise ValueError("training learn called more than once")
        state["learn_calls"] += 1
        result = original_learn(self, *args, **kwargs)
        state["learn_returned"] = True
        return result

    runner_class.learn = learn
    state.update({"runner_class": runner_class, "original_learn": original_learn})


def _restore_close(state: Mapping[str, Any]) -> None:
    env = state.get("environment")
    if env is None:
        return
    if state.get("had_instance_close"):
        env.close = state["instance_close"]
    else:
        delattr(env, "close")


def _restore_learn(state: Mapping[str, Any]) -> None:
    runner_class = state.get("runner_class")
    if runner_class is not None:
        runner_class.learn = state["original_learn"]


@contextmanager
def environment_scope(
        binding: Mapping[str, Any], readback_path: Path,
        final_readback_path: Path | None = None) -> Iterator[dict[str, Any]]:
    """Keep the registered body/world reward implementation active for the whole child run."""
    checked = validate_environment_binding(binding)
    source = (Path.cwd() / checked["target"]["source"]).resolve()
    motor_source = (Path.cwd() / checked["secondary_target"]["source"]).resolve()
    cwd = Path.cwd().resolve()
    try:
        source.relative_to(cwd)
        motor_source.relative_to(cwd)
    except ValueError as exc:
        raise ValueError("reward target source escapes runtime root") from exc
    if not source.is_file() or _sha(source, lf=True) != checked["target"]["source_sha256_lf"]:
        raise ValueError("reward target source changed")
    if (not motor_source.is_file()
            or _sha(motor_source, lf=True) != checked["secondary_target"]["source_sha256_lf"]):
        raise ValueError("motor target source changed")

    gymnasium = importlib.import_module("gymnasium")
    original_make = gymnasium.make
    source_arm = checked["source_provenance"]["source_arm"]
    state: dict[str, Any] = {"make_calls": 0, "readback_written": False, "recipe_applied": False,
                             "function": None, "original_code": None,
                             "motor_class": None, "original_motor": None, "term_function": None,
                             "close_calls": 0, "final_readback_written": False,
                             "environment": None, "learn_calls": 0, "learn_returned": False,
                             "runner_class": None}

    def make(*args, **kwargs):
        if state["recipe_applied"]:
            raise ValueError("more than one training environment was created")
        # Isaac modules are imported only after the upstream AppLauncher has initialized and called gym.make.
        rewards = importlib.import_module("isaaclab.envs.mdp.rewards")
        actuator = importlib.import_module("isaaclab.actuators.actuator_pd")
        function = rewards.track_lin_vel_xy_exp
        motor_class = actuator.DCMotor
        motor_function = motor_class._clip_effort
        if ADAPTER.fingerprint(function) != checked["target"]["clean_fingerprint"]:
            raise ValueError("reward target is not clean before preservation")
        if ADAPTER.fingerprint(motor_function) != checked["secondary_target"]["clean_fingerprint"]:
            raise ValueError("motor target is not clean before preservation")
        expected_code = (function.__code__ if checked["mode"] == "nominal_body_frame"
                         else FAULTS._track_lin_vel_world.__code__)
        if ADAPTER.fingerprint(expected_code) != checked["target"]["expected_runtime_fingerprint"]:
            raise ValueError("registered runtime reward fingerprint differs")
        state.update({"function": function, "original_code": function.__code__,
                      "motor_class": motor_class, "original_motor": motor_function})
        function.__code__ = expected_code
        if source_arm == "F2":
            FAULTS._before_env("F2")
        state["recipe_applied"] = True
        env = original_make(*args, **kwargs)
        state["make_calls"] += 1
        term_function = _reward_term(env)
        state["term_function"] = term_function
        runtime_motor = motor_class._clip_effort
        row = {
            "schema": "registered_epoch_environment_readback_v1",
            "mode": checked["mode"],
            "source_setup": checked["source_setup"],
            "expected_reward": checked["expected_reward"],
            "source_env_config": checked["source_env_config"],
            "code_contract": checked["code_contract"],
            "provenance_kind": checked["provenance_kind"],
            "module": getattr(term_function, "__module__", None),
            "qualname": getattr(term_function, "__qualname__", None),
            "runtime_fingerprint": ADAPTER.fingerprint(term_function),
            "target_identity_preserved": term_function is function,
            "motor_module": getattr(runtime_motor, "__module__", None),
            "motor_qualname": getattr(runtime_motor, "__qualname__", None),
            "motor_runtime_fingerprint": ADAPTER.fingerprint(runtime_motor),
            "motor_target_identity_preserved": runtime_motor is motor_function,
            "make_call": state["make_calls"],
        }
        if (row["module"] != checked["target"]["module"]
                or row["qualname"] != checked["target"]["qualname"]
                or row["runtime_fingerprint"] != checked["target"]["expected_runtime_fingerprint"]
                or row["target_identity_preserved"] is not True
                or row["motor_runtime_fingerprint"]
                != checked["secondary_target"]["expected_runtime_fingerprint"]
                or row["motor_target_identity_preserved"] is not (source_arm != "F2")):
            raise ValueError("runtime reward environment differs from registered source")
        if state["readback_written"]:
            raise ValueError("more than one training environment was created")
        _write_new(readback_path, row)
        state["readback_written"] = True
        state["readback"] = row
        if final_readback_path is not None:
            _install_learn_observer(state)
            _install_close_seal(env, state, checked, readback_path, final_readback_path)
        return env

    try:
        gymnasium.__dict__["make"] = make
        yield state
        if state["make_calls"] != 1 or state["readback_written"] is not True:
            raise ValueError("training environment readback was not sealed exactly once")
    finally:
        _restore_close(state)
        _restore_learn(state)
        gymnasium.__dict__["make"] = original_make
        if state["function"] is not None:
            state["function"].__code__ = state["original_code"]
        if state["motor_class"] is not None:
            state["motor_class"]._clip_effort = state["original_motor"]


def execute(manifest_path: Path, environment_binding_path: Path) -> dict[str, Any]:
    manifest_path = Path(manifest_path).resolve(strict=True)
    binding_path = Path(environment_binding_path).resolve(strict=True)
    expected_binding_path = (manifest_path.parent.parent
                             / f".{manifest_path.parent.name}-environment-binding.json").resolve()
    if binding_path != expected_binding_path:
        raise ValueError("environment binding path differs from the registered sibling")
    manifest = BASE.validate_manifest(_read(manifest_path))
    raw_binding = _read(binding_path)
    binding = validate_environment_binding(raw_binding)
    provenance = binding["source_provenance"]
    if (manifest["task"] != provenance["task"] or manifest["seed"] != provenance["training_seed"]
            or manifest["source_checkpoint_sha256"] != provenance["source_checkpoint_sha256"]
            or manifest["arm"] != "CORRECTED" or manifest["mode"] != "train"):
        raise ValueError("environment evidence differs from corrected training manifest")
    readback_path = manifest_path.parent / "environment-readback.json"
    if readback_path.exists():
        raise FileExistsError("environment readback already exists")
    final_readback_path = manifest_path.parent / "environment-final-readback.json"
    if final_readback_path.exists():
        raise FileExistsError("environment final readback already exists")
    with environment_scope(raw_binding, readback_path, final_readback_path) as state:
        BASE.execute(manifest_path)
        # CPU fixtures and alternate launchers may return without calling env.close().
        _seal_final_readback(state, binding, readback_path, final_readback_path)
    return dict(state["readback"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--environment-binding", required=True, type=Path)
    args = parser.parse_args(argv)
    execute(args.manifest, args.environment_binding)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
