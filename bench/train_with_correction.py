"""사전등록 manifest대로 숨은 결함을 주입·복원한 뒤 상류 RSL-RL 학습을 실행한다."""
from __future__ import annotations

import argparse
import hashlib
import importlib
from importlib import metadata
import json
import os
from pathlib import Path
import runpy
import sys
from typing import Any, Callable, Mapping


ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "bench"
EVALS = ROOT / "evals"
MANIFEST_FIELDS = {
    "schema", "tag", "case", "arm", "source_commit_sha", "protocol_sha256_lf",
    "isaaclab_root", "protocol_path", "fault", "correction_id", "seed", "num_envs",
    "max_iterations", "task", "run_name", "output_dir",
    "resume_checkpoint_path", "resume_checkpoint_sha256", "resume_iteration",
    "load_run_regex", "checkpoint_regex", "resume_env_config_path", "resume_agent_config_path",
}


def _json_scalar(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return {"float_hex": value.hex()}
    raise TypeError(f"unsupported state scalar {type(value).__qualname__}")


def _state_material(value: Any, path: str, tensors: list[dict], scalars: list[dict]) -> Any:
    if all(hasattr(value, name) for name in ("detach", "cpu", "contiguous", "numpy")):
        tensor = value.detach().cpu().contiguous()
        raw = tensor.numpy().tobytes(order="C")
        row = {
            "path": path, "dtype": str(tensor.dtype), "shape": [int(item) for item in tensor.shape],
            "content_sha256": hashlib.sha256(raw).hexdigest(),
        }
        tensors.append(row)
        return {"tensor": row}
    if isinstance(value, Mapping):
        rows = []
        for key, child in value.items():
            key_material = _json_scalar(key)
            key_text = json.dumps(key_material, sort_keys=True, separators=(",", ":"))
            rows.append((key_text, key_material, child))
        rows.sort(key=lambda row: row[0])
        return {"mapping": [
            {"key": key, "value": _state_material(child, f"{path}[{key_text}]", tensors, scalars)}
            for key_text, key, child in rows
        ]}
    if isinstance(value, (list, tuple)):
        return {"sequence_type": type(value).__name__, "items": [
            _state_material(child, f"{path}[{index}]", tensors, scalars)
            for index, child in enumerate(value)
        ]}
    scalar = _json_scalar(value)
    scalars.append({"path": path, "type": type(value).__name__, "value": scalar})
    return {"scalar": scalar, "type": type(value).__name__}


def state_digest(value: Any) -> dict[str, Any]:
    """tensor 내용·dtype·shape와 스칼라를 모두 포함한 결정적 상태 지문."""
    tensors: list[dict] = []
    scalars: list[dict] = []
    material = _state_material(value, "$", tensors, scalars)
    raw = json.dumps(material, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "tensor_count": len(tensors), "scalar_count": len(scalars),
        "tensors": tensors, "scalars": scalars,
    }


def _checkpoint_state(path: Path, expected_iteration: int, torch_module) -> dict[str, Any]:
    payload = torch_module.load(str(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("iter") != expected_iteration:
        raise ValueError("checkpoint iteration differs from the bound training stage")
    if "model_state_dict" not in payload or "optimizer_state_dict" not in payload:
        raise ValueError("checkpoint is missing policy or optimizer state")
    policy = state_digest(payload["model_state_dict"])
    optimizer = state_digest(payload["optimizer_state_dict"])
    return {
        "iteration": expected_iteration,
        "policy_state_sha256": policy["sha256"],
        "optimizer_state_sha256": optimizer["sha256"],
    }


def _rng_state_hashes(torch_module) -> dict[str, Any]:
    cpu = state_digest(torch_module.get_rng_state())
    cuda_states = torch_module.cuda.get_rng_state_all()
    return {
        "cpu_sha256": cpu["sha256"],
        "cuda_all_sha256": [state_digest(state)["sha256"] for state in cuda_states],
        "cuda_device_count": len(cuda_states),
    }


def _value(mapping: Any, key: str) -> Any:
    if isinstance(mapping, Mapping):
        return mapping.get(key)
    return getattr(mapping, key, None)


def _runtime_versions() -> dict[str, str]:
    def distribution(name: str) -> str:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            return "unavailable"

    return {
        "python": ".".join(str(value) for value in sys.version_info[:3]),
        "python_cache_tag": str(sys.implementation.cache_tag),
        "torch_distribution": distribution("torch"),
        "rsl_rl_distribution": distribution("rsl-rl-lib"),
        "isaaclab_distribution": distribution("isaaclab"),
    }


def _runner_observations(runner: Any, train_cfg: Any, manifest: dict,
                         expected_action_scale: float | None) -> dict[str, Any]:
    env_cfg = runner.env.unwrapped.cfg
    action_scale = env_cfg.actions.joint_pos.scale
    if isinstance(action_scale, bool) or not isinstance(action_scale, (int, float)):
        raise ValueError("runtime joint position action scale must be numeric")
    observed = {
        "environment_seed": getattr(env_cfg, "seed", None),
        "environment_num_envs": getattr(env_cfg.scene, "num_envs", None),
        "action_scale": float(action_scale),
        "agent_seed": _value(train_cfg, "seed"),
        "agent_max_iterations": _value(train_cfg, "max_iterations"),
        "runner_num_steps_per_env": getattr(runner, "num_steps_per_env", None),
    }
    expected = {
        "environment_seed": manifest["seed"], "environment_num_envs": manifest["num_envs"],
        "agent_seed": manifest["seed"], "agent_max_iterations": manifest["max_iterations"],
    }
    if any(observed[key] != value for key, value in expected.items()):
        raise ValueError("runner seed, environment count, or iteration configuration differs from manifest")
    if expected_action_scale is not None and observed["action_scale"] != expected_action_scale:
        raise ValueError("runtime action scale differs from the fixed protocol")
    steps = observed["runner_num_steps_per_env"]
    if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
        raise ValueError("runner num_steps_per_env must be a positive integer")
    return observed


def _install_runner_hooks(manifest: dict, H, runtime: dict[str, Any], *,
                          expected_action_scale: float | None = None,
                          expected_rsl_version: str | None = None,
                          expected_isaaclab_commit: str | None = None,
                          on_learning_complete: Callable[[], None] | None = None) -> tuple[Any, Any, Any, type]:
    runners = importlib.import_module("rsl_rl.runners")
    runner_class = runners.OnPolicyRunner
    original_init = runner_class.__init__
    original_load = runner_class.load
    original_learn = runner_class.learn
    output_dir = Path(manifest["output_dir"])

    def measured_init(self, *args, **kwargs):
        if runtime.get("runner_initialized"):
            raise RuntimeError("training created more than one OnPolicyRunner")
        original_init(self, *args, **kwargs)
        train_cfg = args[1] if len(args) > 1 else kwargs.get("train_cfg")
        observations = _runner_observations(self, train_cfg, manifest, expected_action_scale)
        versions = _runtime_versions()
        if expected_isaaclab_commit is not None:
            versions["isaaclab_source_commit"] = expected_isaaclab_commit
        if expected_rsl_version is not None and versions["rsl_rl_distribution"] != expected_rsl_version:
            raise ValueError("runtime RSL-RL version differs from the fixed protocol")
        log_dir = kwargs.get("log_dir", args[2] if len(args) > 2 else getattr(self, "log_dir", None))
        if not isinstance(log_dir, (str, os.PathLike)):
            raise ValueError("runner log_dir is unavailable")
        runtime.update({
            "runner_initialized": True, "runner": self, "log_dir": Path(log_dir).resolve(),
            "observed": observations, "runtime_versions": versions,
        })

    def measured_load(self, *args, **kwargs):
        if not runtime.get("runner_initialized") or runtime.get("checkpoint_loaded"):
            raise RuntimeError("runner.load must occur exactly once after runner initialization")
        path_value = args[0] if args else kwargs.get("path")
        load_optimizer = kwargs.get("load_optimizer", args[1] if len(args) > 1 else True)
        expected_path = Path(manifest["resume_checkpoint_path"]).resolve()
        if not isinstance(path_value, (str, os.PathLike)) or Path(path_value).resolve() != expected_path:
            raise ValueError("runner.load checkpoint path differs from the resume manifest")
        if load_optimizer is not True:
            raise ValueError("runner.load must restore optimizer state")
        if H.sha(expected_path, lf=False) != manifest["resume_checkpoint_sha256"]:
            raise ValueError("resume checkpoint SHA differs immediately before load")
        import torch
        source_state = _checkpoint_state(expected_path, manifest["resume_iteration"], torch)
        result = original_load(self, *args, **kwargs)
        if H.sha(expected_path, lf=False) != manifest["resume_checkpoint_sha256"]:
            raise ValueError("resume checkpoint changed during load")
        if self.current_learning_iteration != manifest["resume_iteration"]:
            raise ValueError("loaded checkpoint iteration differs from the resume manifest")
        current_policy = state_digest(self.alg.policy.state_dict())
        current_optimizer = state_digest(self.alg.optimizer.state_dict())
        if (current_policy["sha256"] != source_state["policy_state_sha256"]
                or current_optimizer["sha256"] != source_state["optimizer_state_sha256"]):
            raise ValueError("loaded policy or optimizer state differs from the source checkpoint")
        runtime.update({
            "checkpoint_loaded": True,
            "resume_checkpoint_path": expected_path,
            "resume_checkpoint_sha256": manifest["resume_checkpoint_sha256"],
            "resume_iteration": self.current_learning_iteration,
            "load_optimizer": True,
            "source_checkpoint_state": source_state,
        })
        return result

    def measured_learn(self, *args, **kwargs):
        if runtime.get("learn_called"):
            raise RuntimeError("runner.learn was called more than once")
        if not runtime.get("checkpoint_loaded"):
            raise RuntimeError("runner.learn requires the registered resume checkpoint")
        iterations = kwargs.get("num_learning_iterations", args[0] if args else None)
        random_ep = kwargs.get("init_at_random_ep_len", args[1] if len(args) > 1 else None)
        if iterations != manifest["max_iterations"] or random_ep is not True:
            raise ValueError("runner.learn arguments differ from the fixed training budget")
        H.validate_training_configuration(runtime["log_dir"], manifest)
        import torch
        rng_states = _rng_state_hashes(torch)
        runtime["rng_state_before_learn"] = rng_states
        initial = {
            "schema": "training_correction_initial_v1",
            "policy_state": state_digest(self.alg.policy.state_dict()),
            "optimizer_state": state_digest(self.alg.optimizer.state_dict()),
            "observed": runtime["observed"],
            "runtime_versions": runtime["runtime_versions"],
            "log_dir": str(runtime["log_dir"]),
            "resume_checkpoint": {
                "path": str(runtime["resume_checkpoint_path"]),
                "sha256": runtime["resume_checkpoint_sha256"],
                "load_optimizer": runtime["load_optimizer"],
            },
            "resume_iteration": runtime["resume_iteration"],
            "init_at_random_ep_len": True,
            "source_checkpoint_state": runtime["source_checkpoint_state"],
            "rng_state_before_learn": rng_states,
            "full_environment_state_restored": False,
            "rng_state_restored_from_checkpoint": False,
        }
        initial_path = output_dir / "initial.json"
        H.write_new(initial_path, initial)
        runtime["initial_path"] = initial_path
        expected_steps = iterations * self.num_steps_per_env
        original_step = self.env.step
        counter = {"successful": 0}

        def measured_step(*step_args, **step_kwargs):
            result = original_step(*step_args, **step_kwargs)
            counter["successful"] += 1
            return result

        runtime["learn_called"] = True
        self.env.step = measured_step
        try:
            result = original_learn(self, *args, **kwargs)
        finally:
            self.env.step = original_step
        if counter["successful"] != expected_steps:
            raise RuntimeError("successful environment steps differ from the fixed training budget")
        runtime.update({
            "learning_completed": True, "actual_control_steps": counter["successful"],
            "expected_control_steps": expected_steps,
        })
        if on_learning_complete is not None:
            on_learning_complete()
        return result

    runner_class.__init__ = measured_init
    runner_class.load = measured_load
    runner_class.learn = measured_learn
    return original_init, original_load, original_learn, runner_class


def _upstream_argv(upstream: Path, manifest: dict) -> list[str]:
    return [
        str(upstream), "--task", manifest["task"], "--seed", str(manifest["seed"]),
        "--num_envs", str(manifest["num_envs"]), "--max_iterations", str(manifest["max_iterations"]),
        "--headless", "--run_name", manifest["run_name"], "--resume",
        "--load_run", manifest["load_run_regex"], "--checkpoint", manifest["checkpoint_regex"],
    ]


def _artifact(path: Path, H) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path.resolve()), "sha256": H.sha(path, lf=False)}


def execute(manifest_path: Path, H) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    manifest, protocol = H.validate_manifest(manifest_path)
    if set(manifest) != MANIFEST_FIELDS:
        raise ValueError("training correction manifest fields differ from the fixed schema")
    isaaclab_root = Path(manifest["isaaclab_root"]).resolve()
    if Path.cwd().resolve() != isaaclab_root:
        raise ValueError("training correction bootstrap must run from the Isaac Lab root")
    upstream = isaaclab_root / "scripts" / "reinforcement_learning" / "rsl_rl" / "train.py"
    if not upstream.is_file():
        raise FileNotFoundError(upstream)
    protocol_path = Path(manifest["protocol_path"]).resolve()
    protocol_sha = H.sha(protocol_path, lf=True)
    if protocol_sha != manifest["protocol_sha256_lf"]:
        raise ValueError("protocol LF SHA differs from the validated manifest")
    output_dir = Path(manifest["output_dir"]).resolve()
    initial_path = output_dir / "initial.json"
    correction_path = output_dir / "correction.json"
    receipt_path = output_dir / "receipt.json"
    if any(path.exists() for path in (initial_path, correction_path, receipt_path)):
        raise FileExistsError("training correction artifacts are write-once")

    if str(BENCH) not in sys.path:
        sys.path.insert(0, str(BENCH))
    import correction_adapter as A
    import hidden_faults

    original_before = hidden_faults._before_env
    runtime: dict[str, Any] = {"environment_make_calls": 0}
    runner_hooks = None

    def record_learning_complete() -> None:
        # Isaac fast shutdown can end the process without returning from runpy.
        # The parent separately requires a clean process exit and owned-job cleanup.
        runtime["receipt"] = _write_receipt(manifest, manifest_path, runtime, H)

    def corrected_before(fault: str) -> None:
        nonlocal runner_hooks
        if runtime.get("correction_applied"):
            raise RuntimeError("correction hook was called more than once")
        snapshot = A.capture(isaaclab_root, protocol["correction_source_contract"])
        original_before(fault)
        readback = A.apply(snapshot, manifest["correction_id"])
        H.validate_readback(readback, manifest["fault"], manifest["correction_id"])
        H.write_new(correction_path, readback)
        runner_hooks = _install_runner_hooks(
            manifest, H, runtime,
            expected_action_scale=protocol["action_scale"],
            expected_rsl_version=protocol["rsl_rl_version"],
            expected_isaaclab_commit=protocol["isaaclab_commit"],
            on_learning_complete=record_learning_complete,
        )
        runtime["correction_applied"] = True

    hidden_faults._before_env = corrected_before
    hidden_faults.arm(manifest["fault"])
    import gymnasium
    armed_make = gymnasium.make

    def one_environment_make(*args, **kwargs):
        runtime["environment_make_calls"] += 1
        if runtime["environment_make_calls"] != 1:
            raise RuntimeError("training created more than one gym environment")
        return armed_make(*args, **kwargs)

    gymnasium.make = one_environment_make
    old_argv = sys.argv
    old_path = sys.path[:]
    sys.argv = _upstream_argv(upstream, manifest)
    sys.path.insert(0, str(upstream.parent))
    try:
        runpy.run_path(str(upstream), run_name="__main__")
    finally:
        sys.argv = old_argv
        sys.path[:] = old_path
        gymnasium.make = armed_make
        hidden_faults._before_env = original_before
        if runner_hooks is not None:
            original_init, original_load, original_learn, runner_class = runner_hooks
            runner_class.__init__ = original_init
            runner_class.load = original_load
            runner_class.learn = original_learn

    if "receipt" not in runtime:
        raise RuntimeError("upstream training returned without a completion receipt")
    return runtime["receipt"]


def _write_receipt(manifest: dict, manifest_path: Path, runtime: dict[str, Any], H) -> dict[str, Any]:
    """학습과 최종 저장을 검증한 뒤 Isaac 종료 전에 기록한다."""
    if (not runtime.get("correction_applied") or not runtime.get("runner_initialized")
            or not runtime.get("checkpoint_loaded") or not runtime.get("learning_completed")):
        raise RuntimeError("training did not complete correction and learning")
    output_dir = Path(manifest["output_dir"]).resolve()
    protocol_path = Path(manifest["protocol_path"]).resolve()
    protocol_sha = H.sha(protocol_path, lf=True)
    if protocol_sha != manifest["protocol_sha256_lf"]:
        raise ValueError("protocol changed during learning")
    initial_path = output_dir / "initial.json"
    correction_path = output_dir / "correction.json"
    receipt_path = output_dir / "receipt.json"
    log_dir = runtime["log_dir"]
    final_iteration = manifest["resume_iteration"] + manifest["max_iterations"] - 1
    checkpoint = log_dir / f"model_{final_iteration}.pt"
    import torch
    final_checkpoint_state = _checkpoint_state(checkpoint, final_iteration, torch)
    configs = {
        "environment": _artifact(log_dir / "params" / "env.yaml", H),
        "agent": _artifact(log_dir / "params" / "agent.yaml", H),
    }
    import runtime_readback_smoke as S
    receipt = {
        "schema": "training_correction_receipt_v1", "completed": True,
        "tag": manifest["tag"], "case": manifest["case"], "arm": manifest["arm"],
        "fault": manifest["fault"], "correction_id": manifest["correction_id"],
        "source_commit_sha": manifest["source_commit_sha"],
        "protocol_sha256_lf": manifest["protocol_sha256_lf"],
        "protocol": {"path": str(protocol_path), "sha256_lf": protocol_sha},
        "manifest_sha256": H.sha(manifest_path, lf=False),
        "initial_sha256": H.sha(initial_path, lf=False),
        "correction_sha256": H.sha(correction_path, lf=False),
        "actual_control_steps": runtime["actual_control_steps"],
        "resume_checkpoint": {
            "path": str(runtime["resume_checkpoint_path"]),
            "sha256": runtime["resume_checkpoint_sha256"],
        },
        "resume_iteration": runtime["resume_iteration"], "final_iteration": final_iteration,
        "source_checkpoint_state": runtime["source_checkpoint_state"],
        "final_checkpoint_state": final_checkpoint_state,
        "budget": {
            "seed": manifest["seed"], "num_envs": manifest["num_envs"],
            "max_iterations": manifest["max_iterations"],
            "num_steps_per_env": runtime["observed"]["runner_num_steps_per_env"],
            "expected_control_steps": runtime["expected_control_steps"],
        },
        "aggregate_env_steps": runtime["actual_control_steps"] * manifest["num_envs"],
        "environment_make_calls": runtime["environment_make_calls"],
        "task": manifest["task"], "run_name": manifest["run_name"],
        "observed": runtime["observed"],
        "actual_action_scale": runtime["observed"]["action_scale"],
        "runtime_versions": runtime["runtime_versions"],
        "rng_state_before_learn": runtime["rng_state_before_learn"],
        "resume_scope": {
            "policy_optimizer_restored": True,
            "full_environment_state_restored": False,
            "rng_state_restored_from_checkpoint": False,
        },
        "process_identity": S.process_identity(),
        "checkpoint": _artifact(checkpoint, H), "configs": configs,
    }
    H.write_new(receipt_path, receipt)
    return receipt


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args(argv)
    if str(EVALS) not in sys.path:
        sys.path.insert(0, str(EVALS))
    import run_training_correction as H
    execute(args.manifest, H)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
