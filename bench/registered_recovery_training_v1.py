"""Registered F2 actuator recovery child using the frozen correction training hooks.

This adapter supplies a new-stage manifest validator to ``train_with_correction``.  It
does not call the legacy experiment CLI or write below its historical private root.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _folder in (ROOT, ROOT / "bench", ROOT / "evals", ROOT / "src"):
    if str(_folder) not in sys.path:
        sys.path.insert(0, str(_folder))

from bench import correction_adapter as ADAPTER
from bench import train_with_correction as CHILD
from evals import r1_repair_execution as V1
from evals import r1_repair_execution_v2 as V2
from evals import run_training_correction as LEGACY

SCHEMA = "registered_actuator_recovery_binding_v1"
SCHEMA_V2 = "registered_actuator_recovery_binding_v2"
SCHEMA_V3 = "registered_actuator_recovery_binding_v3"
NAMESPACE_PROFILE = "balanced_resource_remeasurement_v1"
NAMESPACE_RECEIPT_SCHEMA = "registered_actuator_run_namespace_v1"
CORRECTION = "restore_motor_torque_speed"


def _sha(path: Path, lf: bool = False) -> str:
    raw = path.read_bytes()
    if lf:
        raw = raw.replace(b"\r\n", b"\n")
    return hashlib.sha256(raw).hexdigest()


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("JSON object required")
    return value


def _write_new(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
        stream.write("\n")


def validate_prepared_binding(binding: Mapping[str, Any]) -> dict:
    """Validate immutable fields and refs in host Python without importing the SDK."""
    fields = {"schema", "source_commit_sha", "task", "seed", "num_envs", "max_iterations",
              "resume_iteration", "resume_checkpoint", "resume_env_config", "resume_agent_config",
              "correction_source_contract", "action_scale", "rsl_rl_version", "isaaclab_commit",
              "num_steps_per_env"}
    schema = binding.get("schema") if isinstance(binding, Mapping) else None
    if schema in {SCHEMA_V2, SCHEMA_V3}:
        fields.add("namespace_profile")
    if (not isinstance(binding, Mapping) or set(binding) != fields
            or schema not in {SCHEMA, SCHEMA_V2, SCHEMA_V3}):
        raise ValueError("actuator recovery binding fields")
    if schema in {SCHEMA_V2, SCHEMA_V3} \
            and binding["namespace_profile"] != NAMESPACE_PROFILE:
        raise ValueError("actuator recovery namespace profile")
    if binding["max_iterations"] != 300 or binding["resume_iteration"] != 299:
        raise ValueError("actuator recovery iteration binding")
    if binding["task"] != V1.TASK or type(binding["seed"]) is not int or type(binding["num_envs"]) is not int:
        raise ValueError("actuator recovery task/seed/env binding")
    for key in ("resume_checkpoint", "resume_env_config", "resume_agent_config"):
        ref = binding[key]
        if not isinstance(ref, dict) or set(ref) != {"path", "sha256"}:
            raise ValueError(f"{key} reference fields")
        path = Path(ref["path"]).resolve(strict=True)
        if _sha(path) != ref["sha256"]:
            raise ValueError(f"{key} changed")
    contract = binding["correction_source_contract"]
    if (not isinstance(contract, dict) or contract.get("schema") != "correction_source_contract_v1"
            or not isinstance(contract.get("runtime"), dict) or not isinstance(contract.get("targets"), dict)):
        raise ValueError("correction source contract fields")
    return dict(binding)


def validate_binding(binding: Mapping[str, Any], isaaclab: Path) -> dict:
    """In SDK Python, additionally bind the live source and interpreter ABI."""
    prepared = validate_prepared_binding(binding)
    if prepared["correction_source_contract"] != ADAPTER.source_contract(isaaclab):
        raise ValueError("correction source contract changed")
    return prepared


def validate_host_binding(binding: Mapping[str, Any], isaaclab: Path) -> dict:
    """Host-side source/ref check; Python ABI and code fingerprints remain child-owned."""
    prepared = validate_prepared_binding(binding)
    root = Path(isaaclab).resolve()
    contract = prepared["correction_source_contract"]
    targets = contract["targets"]
    if set(targets) != set(ADAPTER.TARGETS):
        raise ValueError("correction source contract target set differs")
    for correction_id, static in ADAPTER.TARGETS.items():
        row = targets[correction_id]
        if (not isinstance(row, dict)
                or any(row.get(key) != value for key, value in static.items())
                or not isinstance(row.get("clean_fingerprint"), str)
                or not re.fullmatch(r"[0-9a-f]{64}", row["clean_fingerprint"])
                or not isinstance(row.get("source_sha256"), str)
                or not re.fullmatch(r"[0-9a-f]{64}", row["source_sha256"])):
            raise ValueError("correction source contract target metadata differs")
        path = (root / static["source"]).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError("correction source path escapes Isaac Lab root") from exc
        if (not path.is_file()
                or hashlib.sha256(ADAPTER._lf_bytes(path)).hexdigest() != row["source_sha256"]):
            raise ValueError("correction target source changed")
    return prepared


class _Host:
    """Narrow helper ABI consumed by the frozen child implementation."""

    def __init__(self, binding: dict, manifest_path: Path):
        self.binding = binding
        self.manifest_path = manifest_path.resolve()

    @staticmethod
    def sha(path: Path, lf: bool = False) -> str:
        return _sha(Path(path), lf)

    @staticmethod
    def write_new(path: Path, value: dict) -> None:
        _write_new(Path(path), value)

    def validate_manifest(self, path: Path) -> tuple[dict, dict]:
        if Path(path).resolve() != self.manifest_path:
            raise ValueError("manifest path differs")
        manifest = _read(self.manifest_path)
        namespace = self.validate_namespace_receipt() if self.binding["schema"] == SCHEMA_V3 else None
        expected = manifest_for(self.manifest_path.parent, self.binding,
                                Path(manifest["isaaclab_root"]), manifest["run_name"],
                                original_run_name=(namespace["original_run_name"]
                                                   if namespace is not None
                                                   else manifest.get("original_run_name")))
        if manifest != expected:
            raise ValueError("manifest differs from registered actuator binding")
        protocol = {"correction_source_contract": self.binding["correction_source_contract"],
                    "action_scale": self.binding["action_scale"],
                    "rsl_rl_version": self.binding["rsl_rl_version"],
                    "isaaclab_commit": self.binding["isaaclab_commit"]}
        return manifest, protocol

    def validate_namespace_receipt(self) -> dict:
        path = self.manifest_path.parent / "run-namespace.json"
        row = _read(path)
        fields = {"schema", "namespace_profile", "original_run_name", "effective_run_name",
                  "binding_sha256", "manifest_sha256"}
        if (set(row) != fields or row["schema"] != NAMESPACE_RECEIPT_SCHEMA
                or row["namespace_profile"] != NAMESPACE_PROFILE
                or row["binding_sha256"] != _sha(self.manifest_path.parent / "binding.json")
                or row["manifest_sha256"] != _sha(self.manifest_path)
                or row["effective_run_name"] != training_run_name(
                    row["original_run_name"], self.binding["seed"], self.binding)):
            raise ValueError("actuator recovery namespace receipt differs")
        return row

    def validate_training_configuration(self, log_dir: Path, manifest: dict) -> None:
        source_env = _read_yaml(Path(manifest["resume_env_config_path"]))
        source_agent = _read_yaml(Path(manifest["resume_agent_config_path"]))
        current_env = _read_yaml(Path(log_dir) / "params/env.yaml")
        current_agent = _read_yaml(Path(log_dir) / "params/agent.yaml")
        if current_env != source_env:
            raise ValueError("environment configuration changed outside the actuator correction")
        allowed = {"run_name", "resume", "load_run", "load_checkpoint", "max_iterations"}
        if ({key: value for key, value in current_agent.items() if key not in allowed}
                != {key: value for key, value in source_agent.items() if key not in allowed}):
            raise ValueError("agent configuration changed outside resume arguments")
        expected = {"run_name": manifest["run_name"], "resume": True,
                    "load_run": manifest["load_run_regex"],
                    "load_checkpoint": manifest["checkpoint_regex"],
                    "max_iterations": manifest["max_iterations"]}
        if any(current_agent.get(key) != value for key, value in expected.items()):
            raise ValueError("agent resume configuration differs")

    @staticmethod
    def validate_readback(row: dict, fault: str, correction: str | None) -> None:
        LEGACY.validate_readback(row, fault, correction)


def _read_yaml(path: Path) -> dict:
    return LEGACY.load_config(path)


def training_run_name(request_id: str, seed: int, binding: Mapping[str, Any]) -> str:
    """Return the historical request name or a V2 source-bound namespace."""
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", request_id):
        raise ValueError("registered actuator request identity")
    if type(seed) is not int or seed != binding.get("seed"):
        raise ValueError("actuator recovery seed differs from binding")
    schema = binding.get("schema")
    if schema == SCHEMA:
        return request_id
    if schema not in {SCHEMA_V2, SCHEMA_V3} \
            or binding.get("namespace_profile") != NAMESPACE_PROFILE:
        raise ValueError("actuator recovery namespace binding")
    checkpoint = binding.get("resume_checkpoint")
    if not isinstance(checkpoint, Mapping) or not isinstance(checkpoint.get("sha256"), str):
        raise TypeError("actuator recovery checkpoint namespace binding")
    identity = {
        "namespace_profile": NAMESPACE_PROFILE,
        "request_id": request_id,
        "seed": seed,
        "source_checkpoint_sha256": checkpoint["sha256"],
        "source_commit_sha": binding.get("source_commit_sha"),
    }
    digest = hashlib.sha256(json.dumps(
        identity, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()[:24]
    prefix = request_id[:32].rstrip("_-")
    return f"{prefix}_ns_{digest}"


def manifest_for(stage: Path, binding: Mapping[str, Any], isaaclab: Path, run_name: str,
                 *, original_run_name: str | None = None) -> dict:
    checkpoint = Path(binding["resume_checkpoint"]["path"]).resolve()
    manifest = {
        "schema": "training_correction_manifest_v1", "tag": run_name, "case": "case_02",
        "arm": "corrected", "source_commit_sha": binding["source_commit_sha"],
        "protocol_sha256_lf": _sha(Path(__file__), True), "isaaclab_root": str(isaaclab.resolve()),
        "protocol_path": str(Path(__file__).resolve()), "fault": "F2", "correction_id": CORRECTION,
        "seed": binding["seed"], "num_envs": binding["num_envs"],
        "max_iterations": binding["max_iterations"], "task": binding["task"], "run_name": run_name,
        "output_dir": str(stage.resolve()), "resume_checkpoint_path": str(checkpoint),
        "resume_checkpoint_sha256": binding["resume_checkpoint"]["sha256"],
        "resume_iteration": binding["resume_iteration"],
        "load_run_regex": "^" + re.escape(checkpoint.parent.name) + "$",
        "checkpoint_regex": "^" + re.escape(checkpoint.name) + "$",
        "resume_env_config_path": str(Path(binding["resume_env_config"]["path"]).resolve()),
        "resume_agent_config_path": str(Path(binding["resume_agent_config"]["path"]).resolve()),
    }
    if binding["schema"] in {SCHEMA_V2, SCHEMA_V3}:
        if original_run_name is None or run_name != training_run_name(
                original_run_name, binding["seed"], binding):
            raise ValueError("v2 actuator manifest namespace differs")
        if binding["schema"] == SCHEMA_V2:
            manifest.update({
                "namespace_profile": binding["namespace_profile"],
                "original_run_name": original_run_name,
                "effective_run_name": run_name,
            })
    elif original_run_name is not None:
        raise ValueError("v1 actuator manifest forbids namespace identity")
    return manifest


def child_main(manifest_path: Path, binding_path: Path) -> dict:
    binding = validate_binding(_read(binding_path), Path.cwd())
    host = _Host(binding, Path(manifest_path))
    if binding["schema"] == SCHEMA_V3:
        host.validate_namespace_receipt()
    return CHILD.execute(Path(manifest_path), host)


def train(stage: Path, isaaclab: Path, arm: str, seed: int, run_name: str,
          binding: Mapping[str, Any], owned=None) -> dict[str, Any]:
    if arm != "CORRECTED" or not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", run_name):
        raise ValueError("registered actuator recovery accepts one corrected request identity")
    bound = validate_host_binding(binding, Path(isaaclab).resolve())
    if seed != bound["seed"] or Path(stage).exists():
        raise ValueError("actuator recovery seed or fresh stage differs")
    effective_run_name = training_run_name(run_name, seed, bound)
    Path(stage).mkdir(parents=True, exist_ok=False)
    binding_path = Path(stage) / "binding.json"
    manifest_path = Path(stage) / "manifest.json"
    _write_new(binding_path, bound)
    manifest = manifest_for(
        Path(stage), bound, Path(isaaclab), effective_run_name,
        original_run_name=run_name if bound["schema"] in {SCHEMA_V2, SCHEMA_V3} else None,
    )
    _write_new(manifest_path, manifest)
    namespace_receipt = None
    if bound["schema"] == SCHEMA_V3:
        namespace_receipt = {
            "schema": NAMESPACE_RECEIPT_SCHEMA,
            "namespace_profile": bound["namespace_profile"],
            "original_run_name": run_name,
            "effective_run_name": effective_run_name,
            "binding_sha256": _sha(binding_path),
            "manifest_sha256": _sha(manifest_path),
        }
        _write_new(Path(stage) / "run-namespace.json", namespace_receipt)
        if _Host(bound, manifest_path).validate_namespace_receipt() != namespace_receipt:
            raise ValueError("actuator recovery namespace receipt self-check differs")
    runner = V2._prepared_owned(owned)
    execution = runner(Path(stage) / "owned", [str(Path(isaaclab) / "_isaac_sim/python.bat"), "-B",
        str(Path(__file__).resolve()), "--manifest", str(manifest_path), "--binding", str(binding_path)],
        Path(isaaclab), V1.WATCHDOG["train"])
    if (namespace_receipt is not None
            and _Host(bound, manifest_path).validate_namespace_receipt() != namespace_receipt):
        raise ValueError("actuator recovery namespace receipt post-run differs")
    _write_new(Path(stage) / "execution.json", dict(execution))
    if not V1._finished(execution):
        raise RuntimeError("registered actuator training did not finish cleanly")
    receipt = _read(Path(stage) / "receipt.json")
    readback = _read(Path(stage) / "correction.json")
    LEGACY.validate_readback(readback, "F2", CORRECTION)
    checkpoint = receipt["checkpoint"]
    if _sha(Path(checkpoint["path"])) != checkpoint["sha256"]:
        raise ValueError("actuator recovery checkpoint changed")
    configuration: dict[str, Any] = {
        "source_checkpoint_consumed": True, "optimizer_state_resumed": True,
    }
    if bound["schema"] in {SCHEMA_V2, SCHEMA_V3}:
        configuration.update({
            "namespace_profile": bound["namespace_profile"],
            "original_run_name": run_name,
            "effective_run_name": effective_run_name,
        })
    return {"arm": arm, "checkpoint": checkpoint, "readback": readback,
            "execution": dict(execution), "receipt": receipt, "configuration": configuration}


fixed_evaluate = V2.fixed_evaluate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--binding", required=True, type=Path)
    args = parser.parse_args(argv)
    child_main(args.manifest, args.binding)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
