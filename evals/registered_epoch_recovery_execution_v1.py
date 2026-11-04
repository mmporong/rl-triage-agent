"""Registered R1 correction runner that preserves the source reward environment."""
from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import yaml  # pyright: ignore[reportMissingModuleSource]

ROOT = Path(__file__).resolve().parents[1]
for _folder in (ROOT, ROOT / "bench", ROOT / "evals", ROOT / "src"):
    if str(_folder) not in sys.path:
        sys.path.insert(0, str(_folder))

from bench import registered_epoch_recovery_training_v1 as CHILD
from evals import r1_repair_execution as V1
from evals import r1_repair_execution_v2 as V2

RUNNER_ID = "r1_epochs_preserving_environment_v1"
ENVIRONMENT_FIELD = "environment_preservation"
PRESERVATION_SCHEMA = "registered_epoch_environment_evidence_v1"


def _write_new(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(dict(value), stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
        stream.write("\n")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ref(path: Path) -> dict[str, str]:
    resolved = path.resolve(strict=True)
    return {"path": str(resolved), "sha256": _sha(resolved)}


def _read_ref(value: Any, label: str, *, json_body: bool = True) -> tuple[Path, Any]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} reference fields")
    raw = Path(value["path"])
    if raw.is_symlink():
        raise ValueError(f"{label} reference changed")
    path = raw.resolve(strict=True)
    if _sha(path) != value["sha256"]:
        raise ValueError(f"{label} reference changed")
    if not json_body:
        return path, None
    body = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(body, dict):
        raise TypeError(f"{label} body")
    return path, body


def _owned_adapter(owned: Callable[..., Mapping[str, Any]] | None,
                   binding_path: Path, state: dict[str, int]):
    runner = owned or V1.OWNED.run_owned

    def execute(folder: Path, args: list[str], cwd: Path, watchdog_s: float):
        argv = list(args)
        base_entry = str((ROOT / "bench/r1_repair_training.py").resolve())
        if len(argv) >= 4 and argv[2] == base_entry and argv[3] == "--manifest":
            if state["training_replacements"] != 0 or len(argv) != 5:
                raise ValueError("corrected training child argv differs")
            argv[2] = str(Path(CHILD.__file__).resolve())
            argv.extend(["--environment-binding", str(binding_path)])
            state["training_replacements"] += 1
        elif "--manifest" in argv:
            raise ValueError("unexpected manifest child entry")
        return runner(folder, argv, cwd, watchdog_s)

    return execute


def train(stage: Path, isaaclab: Path, arm: str, seed: int, run_name: str,
          binding: Mapping[str, Any], owned=None) -> dict[str, Any]:
    if arm != "CORRECTED" or not isinstance(binding, Mapping) or ENVIRONMENT_FIELD not in binding:
        raise ValueError("environment-preserving runner accepts one corrected binding")
    stage = Path(stage)
    if stage.exists():
        raise FileExistsError("training stage already exists")
    environment = CHILD.validate_environment_binding(binding[ENVIRONMENT_FIELD])
    provenance = environment["source_provenance"]
    if seed != provenance["training_seed"]:
        raise ValueError("training seed differs from source environment")
    base_binding = {key: value for key, value in binding.items() if key != ENVIRONMENT_FIELD}
    stage.parent.mkdir(parents=True, exist_ok=True)
    binding_path = stage.parent / f".{stage.name}-environment-binding.json"
    _write_new(binding_path, {key: environment[key] for key in (
        "schema", "mode", "source_setup", "source_env_config", "expected_reward", "target",
        "secondary_target", "code_contract")})
    state = {"training_replacements": 0}
    result = V1.train(stage, isaaclab, arm, seed, run_name, base_binding,
                      _owned_adapter(owned, binding_path, state))
    if state["training_replacements"] != 1:
        raise RuntimeError("corrected training child was not replaced exactly once")
    readback_path = stage / "environment-readback.json"
    if not readback_path.is_file():
        raise RuntimeError("environment preservation readback is missing")
    readback = json.loads(readback_path.read_text(encoding="utf-8"))
    if (not isinstance(readback, dict) or readback.get("mode") != environment["mode"]
            or readback.get("runtime_fingerprint") != environment["target"]["expected_runtime_fingerprint"]
            or readback.get("target_identity_preserved") is not True):
        raise ValueError("environment preservation readback differs")
    run_dir = Path(result.get("run_dir", "")).resolve(strict=True)
    saved_env = yaml.load((run_dir / "params/env.yaml").read_text(encoding="utf-8"),
                          Loader=V1._TolerantLoader)
    source_env = yaml.load(Path(environment["source_env_config"]["path"]).read_text(encoding="utf-8"),
                           Loader=V1._TolerantLoader)
    if (not isinstance(saved_env, dict) or not isinstance(source_env, dict)
            or "rewards" not in source_env or saved_env.get("rewards") != source_env["rewards"]):
        raise ValueError("saved reward configuration differs from source environment")
    checkpoint = result.get("checkpoint")
    if (not isinstance(checkpoint, dict) or set(checkpoint) < {"path", "sha256"}
            or _sha(Path(checkpoint["path"]).resolve(strict=True)) != checkpoint["sha256"]):
        raise ValueError("training checkpoint differs")
    saved_check_path = stage / "environment-saved-config-check.json"
    _write_new(saved_check_path, {
        "schema": "registered_epoch_saved_environment_check_v1",
        "source_env_config": environment["source_env_config"],
        "saved_env_config": _ref(run_dir / "params/env.yaml"),
        "rewards_subtree_match": True,
        "checkpoint": {"path": str(Path(checkpoint["path"]).resolve()), "sha256": checkpoint["sha256"]},
    })
    preservation = {"binding": _ref(binding_path), "runtime_readback": _ref(readback_path),
                    "final_runtime_readback": _ref(stage / "environment-final-readback.json"),
                    "saved_config_check": _ref(saved_check_path)}
    verify_preservation(preservation, checkpoint)
    return {**result, "environment_preservation": preservation}


def verify_preservation(evidence: Mapping[str, Any], checkpoint_ref: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute the environment-preservation receipt without importing the SDK."""
    if not isinstance(evidence, Mapping) or set(evidence) != {
            "binding", "runtime_readback", "final_runtime_readback", "saved_config_check"}:
        raise ValueError("environment preservation evidence fields")
    _, binding_body = _read_ref(evidence["binding"], "environment binding")
    binding = CHILD.validate_environment_binding(binding_body)
    _, readback = _read_ref(evidence["runtime_readback"], "environment runtime readback")
    _, final_readback = _read_ref(evidence["final_runtime_readback"],
                                  "environment final runtime readback")
    _, saved_check = _read_ref(evidence["saved_config_check"], "saved environment check")
    checkpoint_path, _ = _read_ref(checkpoint_ref, "new checkpoint", json_body=False)
    if (readback.get("schema") != "registered_epoch_environment_readback_v1"
            or readback.get("mode") != binding["mode"]
            or readback.get("source_setup") != binding["source_setup"]
            or readback.get("source_env_config") != binding["source_env_config"]
            or readback.get("expected_reward") != binding["expected_reward"]
            or readback.get("code_contract") != binding["code_contract"]
            or readback.get("provenance_kind") != binding["provenance_kind"]
            or readback.get("runtime_fingerprint") != binding["target"]["expected_runtime_fingerprint"]
            or readback.get("target_identity_preserved") is not True
            or readback.get("motor_runtime_fingerprint")
            != binding["secondary_target"]["expected_runtime_fingerprint"]
            or readback.get("motor_target_identity_preserved")
            is not (binding["source_provenance"]["source_arm"] != "F2")
            or readback.get("make_call") != 1):
        raise ValueError("environment runtime readback differs")
    expected_motor_identity = binding["source_provenance"]["source_arm"] != "F2"
    if (final_readback.get("schema") != "registered_epoch_environment_final_readback_v1"
            or final_readback.get("mode") != binding["mode"]
            or final_readback.get("runtime_fingerprint")
            != binding["target"]["expected_runtime_fingerprint"]
            or final_readback.get("target_identity_preserved") is not True
            or final_readback.get("motor_runtime_fingerprint")
            != binding["secondary_target"]["expected_runtime_fingerprint"]
            or final_readback.get("motor_target_identity_preserved") is not expected_motor_identity
            or final_readback.get("initial_readback") != evidence["runtime_readback"]):
        raise ValueError("environment final runtime readback differs")
    if (saved_check.get("schema") != "registered_epoch_saved_environment_check_v1"
            or saved_check.get("source_env_config") != binding["source_env_config"]
            or saved_check.get("rewards_subtree_match") is not True
            or saved_check.get("checkpoint") != {"path": str(checkpoint_path),
                                                  "sha256": checkpoint_ref["sha256"]}):
        raise ValueError("saved environment check differs")
    source_path, _ = _read_ref(binding["source_env_config"], "source env", json_body=False)
    saved_path, _ = _read_ref(saved_check.get("saved_env_config"), "saved env", json_body=False)
    source_env = yaml.load(source_path.read_text(encoding="utf-8"), Loader=V1._TolerantLoader)
    saved_env = yaml.load(saved_path.read_text(encoding="utf-8"), Loader=V1._TolerantLoader)
    if (not isinstance(source_env, dict) or not isinstance(saved_env, dict)
            or "rewards" not in source_env or saved_env.get("rewards") != source_env["rewards"]):
        raise ValueError("saved reward configuration differs from source environment")
    return {"schema": PRESERVATION_SCHEMA, "mode": binding["mode"],
            "source_arm": binding["source_provenance"]["source_arm"],
            "checkpoint_sha256": checkpoint_ref["sha256"], "refs": dict(evidence)}


fixed_evaluate = V2.fixed_evaluate
