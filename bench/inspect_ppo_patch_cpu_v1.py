"""Run a small, actual RSL-RL PPO update on CPU and report its work.

The module deliberately imports Torch and RSL-RL only inside :func:`inspect`.
This keeps configuration validation usable in environments where the Isaac
Lab SDK is not installed.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect as python_inspect
import json
import math
import os
import random
import sys
import tempfile
from importlib import metadata
from pathlib import Path
from typing import Any

SCHEMA = "rsl_ppo_patch_cpu_inspection_v1"
FORBIDDEN_SIMULATION_PREFIXES = ("isaac", "carb", "omni")
_REQUIRED_KEYS = {"num_mini_batches", "learning_rate", "schedule"}
_OPTIONAL_KEYS = {
    "num_learning_epochs",
    "clip_param",
    "gamma",
    "lam",
    "value_loss_coef",
    "entropy_coef",
    "max_grad_norm",
    "use_clipped_value_loss",
    "desired_kl",
    "normalize_advantage_per_mini_batch",
}


class InspectionError(ValueError):
    """Raised when an inspection input or runtime violates the contract."""


def _integer(value: object, *, name: str, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InspectionError(f"{name} must be an integer")
    if not low <= value <= high:
        raise InspectionError(f"{name} must be in [{low}, {high}]")
    return value


def _finite_number(value: object, *, name: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InspectionError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number) or (positive and number <= 0.0):
        qualifier = "positive " if positive else ""
        raise InspectionError(f"{name} must be a {qualifier}finite number")
    return number


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate the bounded subset forwarded to ``rsl_rl.algorithms.PPO``.

    ``num_learning_epochs`` is intentionally optional. Its absence is one of
    the implementation faults under study, so the installed PPO constructor
    must resolve the value rather than this pure validator inventing one.
    """

    if not isinstance(config, dict):
        raise InspectionError("config must be an object")
    missing = sorted(_REQUIRED_KEYS - config.keys())
    unknown = sorted(config.keys() - _REQUIRED_KEYS - _OPTIONAL_KEYS)
    if missing:
        raise InspectionError(f"missing config keys: {missing}")
    if unknown:
        raise InspectionError(f"unsupported config keys: {unknown}")

    normalized = dict(config)
    if "num_learning_epochs" in normalized:
        normalized["num_learning_epochs"] = _integer(
            normalized["num_learning_epochs"],
            name="num_learning_epochs",
            low=1,
            high=8,
        )
    normalized["num_mini_batches"] = _integer(
        normalized["num_mini_batches"],
        name="num_mini_batches",
        low=1,
        high=4,
    )
    normalized["learning_rate"] = _finite_number(
        normalized["learning_rate"], name="learning_rate", positive=True
    )
    if normalized["schedule"] != "fixed":
        raise InspectionError("schedule must be 'fixed' for deterministic inspection")

    numeric = {
        "clip_param": True,
        "gamma": True,
        "lam": True,
        "value_loss_coef": False,
        "entropy_coef": False,
        "max_grad_norm": True,
    }
    for name, positive in numeric.items():
        if name in normalized:
            normalized[name] = _finite_number(
                normalized[name], name=name, positive=positive
            )
    if "desired_kl" in normalized and normalized["desired_kl"] is not None:
        normalized["desired_kl"] = _finite_number(
            normalized["desired_kl"], name="desired_kl", positive=True
        )
    for name in ("use_clipped_value_loss", "normalize_advantage_per_mini_batch"):
        if name in normalized and not isinstance(normalized[name], bool):
            raise InspectionError(f"{name} must be a boolean")
    return normalized


def _canonical_sha256(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _loaded_simulation_modules() -> list[str]:
    return sorted(
        name
        for name in sys.modules
        if name.split(".", 1)[0] in FORBIDDEN_SIMULATION_PREFIXES
    )


def inspect(config: dict[str, Any], *, updates: int = 3) -> dict[str, Any]:
    """Measure real PPO minibatches and optimizer steps on deterministic CPU data."""

    forwarded = validate_config(config)
    updates = _integer(updates, name="updates", low=1, high=3)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise InspectionError("CUDA_VISIBLE_DEVICES must be -1 before SDK import")
    before_simulation_modules = _loaded_simulation_modules()
    if before_simulation_modules:
        raise InspectionError(
            f"simulation modules were loaded before inspection: {before_simulation_modules}"
        )

    # SDK-only imports remain inside this boundary.
    import torch  # type: ignore[import-not-found]
    from rsl_rl.algorithms import PPO  # type: ignore[import-not-found]
    from rsl_rl.runners import OnPolicyRunner  # type: ignore[import-not-found]

    cuda_before = bool(torch.cuda.is_initialized())
    if cuda_before:
        raise InspectionError("CUDA was initialized before the CPU measurement")

    signature = python_inspect.signature(PPO.__init__)
    epochs_parameter = signature.parameters.get("num_learning_epochs")
    if epochs_parameter is None or epochs_parameter.default is python_inspect.Parameter.empty:
        raise InspectionError("installed PPO has no num_learning_epochs default")
    installed_default_epochs = _integer(
        epochs_parameter.default,
        name="installed PPO num_learning_epochs default",
        low=1,
        high=8,
    )
    resolved_epochs = forwarded.get("num_learning_epochs", installed_default_epochs)

    seed = 575
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 1)

    class DeterministicTensorEnv:
        num_envs = 32
        num_actions = 12
        device = "cpu"
        max_episode_length = 10_000

        def __init__(self) -> None:
            self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long)
            self._observations = torch.randn(
                self.num_envs, 48, generator=generator, device="cpu"
            )

        def get_observations(self):
            return self._observations.clone(), {"observations": {}}

        def step(self, actions):
            self.episode_length_buf += 1
            drift = 0.015 * torch.tanh(actions.mean(dim=-1, keepdim=True))
            noise = 0.01 * torch.randn(
                self.num_envs, 48, generator=generator, device="cpu"
            )
            self._observations = 0.97 * self._observations + drift + noise
            rewards = -actions.square().mean(dim=-1) + 0.05 * self._observations[:, 0]
            dones = torch.zeros(self.num_envs, dtype=torch.long, device="cpu")
            return self._observations.clone(), rewards, dones, {"observations": {}}

    runner_config = {
        "num_steps_per_env": 8,
        "save_interval": 999,
        "empirical_normalization": False,
        "policy": {
            "class_name": "ActorCritic",
            "actor_hidden_dims": [64, 32],
            "critic_hidden_dims": [64, 32],
            "activation": "elu",
            "init_noise_std": 0.8,
        },
        "algorithm": {"class_name": "PPO", **forwarded},
    }
    temporary_log = tempfile.TemporaryDirectory(prefix="rl-triage-ppo-cpu-")
    runner = OnPolicyRunner(
        DeterministicTensorEnv(), runner_config, log_dir=temporary_log.name, device="cpu"
    )
    algorithm = runner.alg
    policy = algorithm.policy

    def parameter_snapshot() -> dict[str, Any]:
        tensors = [value.detach().cpu().clone() for value in policy.parameters()]
        digest = hashlib.sha256()
        for value in tensors:
            digest.update(value.contiguous().numpy().tobytes())
        return {"tensors": tensors, "sha256": digest.hexdigest()}

    before = parameter_snapshot()
    counters = {"minibatches": 0, "optimizer_steps": 0}
    per_update: list[dict[str, Any]] = []
    original_generator = algorithm.storage.mini_batch_generator
    original_step = algorithm.optimizer.step
    original_update = algorithm.update

    def measured_generator(*args: Any, **kwargs: Any):
        for batch in original_generator(*args, **kwargs):
            counters["minibatches"] += 1
            yield batch

    def measured_step(*args: Any, **kwargs: Any):
        counters["optimizer_steps"] += 1
        return original_step(*args, **kwargs)

    def measured_update(*args: Any, **kwargs: Any):
        minibatches_before = counters["minibatches"]
        steps_before = counters["optimizer_steps"]
        result = original_update(*args, **kwargs)
        per_update.append(
            {
                "index": len(per_update) + 1,
                "minibatches": counters["minibatches"] - minibatches_before,
                "optimizer_steps": counters["optimizer_steps"] - steps_before,
                "return_values_finite": all(
                    math.isfinite(float(value))
                    for value in result
                    if isinstance(value, (int, float))
                ),
            }
        )
        return result

    algorithm.storage.mini_batch_generator = measured_generator
    algorithm.optimizer.step = measured_step
    algorithm.update = measured_update
    runner.learn(num_learning_iterations=updates, init_at_random_ep_len=False)
    if runner.writer is not None:
        runner.writer.close()
    after = parameter_snapshot()
    temporary_log.cleanup()

    changed_tensors = sum(
        not torch.equal(left, right)
        for left, right in zip(before["tensors"], after["tensors"], strict=True)
    )
    parameter_values_finite = all(
        bool(torch.isfinite(value).all().item()) for value in after["tensors"]
    )
    cuda_after = bool(torch.cuda.is_initialized())
    after_simulation_modules = _loaded_simulation_modules()
    expected_per_update = int(resolved_epochs) * int(forwarded["num_mini_batches"])
    expected_total = updates * expected_per_update
    counts_match = (
        counters["minibatches"] == expected_total
        and counters["optimizer_steps"] == expected_total
        and all(
            row["minibatches"] == expected_per_update
            and row["optimizer_steps"] == expected_per_update
            for row in per_update
        )
    )
    status = (
        "passed"
        if counts_match
        and changed_tensors > 0
        and parameter_values_finite
        and not cuda_after
        and not after_simulation_modules
        else "failed"
    )
    return {
        "schema": SCHEMA,
        "status": status,
        "runtime": {
            "python": sys.version.split()[0],
            "torch": str(torch.__version__),
            "rsl_rl": metadata.version("rsl-rl-lib"),
            "ppo_class": f"{PPO.__module__}.{PPO.__name__}",
            "ppo_constructor_signature": str(signature),
        },
        "input": {
            "config": forwarded,
            "config_sha256": _canonical_sha256(forwarded),
            "updates": updates,
            "seed": seed,
        },
        "resolved": {
            "num_learning_epochs": int(resolved_epochs),
            "num_mini_batches": int(forwarded["num_mini_batches"]),
            "num_learning_epochs_source": (
                "config" if "num_learning_epochs" in forwarded else "installed_ppo_signature_default"
            ),
            "installed_ppo_signature_default": installed_default_epochs,
        },
        "fixture": {
            "kind": "deterministic_tensor_abi_not_simulation_result",
            "num_envs": 32,
            "rollout_steps": 8,
            "observation_dim": 48,
            "action_dim": 12,
            "device": "cpu",
        },
        "measurement": {
            "updates": len(per_update),
            "minibatches": counters["minibatches"],
            "optimizer_steps": counters["optimizer_steps"],
            "expected_per_update": expected_per_update,
            "expected_total": expected_total,
            "counts_match": counts_match,
            "per_update": per_update,
            "policy_sha256_before": before["sha256"],
            "policy_sha256_after": after["sha256"],
            "changed_parameter_tensors": changed_tensors,
            "policy_parameters_finite": parameter_values_finite,
        },
        "isolation": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cuda_initialized_before": cuda_before,
            "cuda_initialized_after": cuda_after,
            "simulation_modules_before": before_simulation_modules,
            "simulation_modules_after": after_simulation_modules,
        },
    }


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise InspectionError("config JSON must contain an object")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error(f"output already exists: {args.out}")
    if not args.out.parent.is_dir():
        parser.error(f"output parent does not exist: {args.out.parent}")
    result = inspect(_read_json(args.config), updates=3)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "out": str(args.out)}))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
