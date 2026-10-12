"""v4 manifest의 회복 방식(resume, resume_std_reset, restart)으로 교정 학습을 실행한다.

v3 진입점 train_with_correction.py는 공개 결과의 소스 결합을 지키려고 바꾸지 않는다.
상태 지문·체크포인트·RNG·runner 관측 함수는 그 모듈의 것을 그대로 쓴다.
"""
from __future__ import annotations

import argparse
import importlib
import math
import os
from pathlib import Path
import runpy
import struct
import sys
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "bench"
EVALS = ROOT / "evals"
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))

import train_with_correction as T  # noqa: E402

RECOVERY_MODES = ("resume", "resume_std_reset", "restart")
RESUME_FIELDS = ("resume_checkpoint_path", "resume_checkpoint_sha256", "resume_iteration",
                 "load_run_regex", "checkpoint_regex")
MANIFEST_FIELDS = {
    "schema", "tag", "seed", "arm", "recovery_mode", "source_commit_sha", "protocol_sha256_lf",
    "isaaclab_root", "protocol_path", "fault", "correction_id", "num_envs", "max_iterations", "task",
    "run_name", "output_dir", "final_iteration", *RESUME_FIELDS,
    "source_env_config_path", "source_agent_config_path", "config_source_fault", "init_noise_std",
}
STD_PATH = '$["std"]'


def float32(value: float) -> float:
    """RSL-RL std 파라미터(float32)에 저장될 값."""
    return struct.unpack("<f", struct.pack("<f", value))[0]


def validate_manifest_mode(manifest: dict) -> str:
    """회복 방식과 재개 필드·최종 iteration·std 초기값의 조합을 확인한다."""
    if set(manifest) != MANIFEST_FIELDS:
        raise ValueError("recovery mode manifest fields differ from the fixed schema")
    mode = manifest["recovery_mode"]
    if mode not in RECOVERY_MODES:
        raise ValueError("unregistered recovery mode")
    if mode == "restart":
        if any(manifest[key] is not None for key in RESUME_FIELDS):
            raise ValueError("restart manifest must not carry resume arguments")
        expected_final = manifest["max_iterations"] - 1
    else:
        if any(manifest[key] is None for key in RESUME_FIELDS):
            raise ValueError("resume manifest is missing resume arguments")
        expected_final = manifest["resume_iteration"] + manifest["max_iterations"] - 1
    if manifest["final_iteration"] != expected_final:
        raise ValueError("final iteration differs from the recovery mode budget")
    std = manifest["init_noise_std"]
    if mode == "resume_std_reset":
        if isinstance(std, bool) or not isinstance(std, (int, float)) or not std > 0:
            raise ValueError("std reset requires a positive registered init_noise_std")
    elif std is not None:
        raise ValueError("only the std reset arm may carry init_noise_std")
    return mode


def _tensor_rows(digest: dict) -> dict[str, tuple]:
    return {row["path"]: (row["dtype"], tuple(row["shape"]), row["content_sha256"]) for row in digest["tensors"]}


def _optimizer_index(optimizer: Any, parameter: Any) -> int:
    """torch Optimizer.state_dict()가 쓰는 param_groups 순서의 정수 index."""
    flat = [item for group in optimizer.param_groups for item in group["params"]]
    indices = [index for index, item in enumerate(flat) if item is parameter]
    if len(indices) != 1:
        raise ValueError("policy std parameter must appear exactly once in the optimizer")
    return indices[0]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def reset_action_std(policy: Any, optimizer: Any, init_noise_std: float, torch_module) -> dict[str, Any]:
    """action std 파라미터를 같은 객체 그대로 등록 초기값으로 채우고, 그 파라미터의 Adam 모멘트만 지운다.

    다른 파라미터의 optimizer 상태와 param_groups(lr 포함)는 유지한다. 파라미터 객체를 바꾸면
    optimizer가 옛 참조를 계속 학습하므로 in-place로만 바꾼다.
    """
    noise_type = getattr(policy, "noise_std_type", None)
    if noise_type == "scalar":
        name, target_value = "std", float(init_noise_std)
    elif noise_type == "log":
        name, target_value = "log_std", math.log(init_noise_std)
    else:
        raise ValueError("std reset supports only scalar or log noise std parameters")
    parameter = getattr(policy, name)
    index = _optimizer_index(optimizer, parameter)
    policy_before = T.state_digest(policy.state_dict())
    optimizer_dict = optimizer.state_dict()
    optimizer_before = T.state_digest(optimizer_dict)
    expected_optimizer = T.state_digest({
        **optimizer_dict,
        "state": {key: value for key, value in optimizer_dict["state"].items() if key != index},
    })
    moment_before = parameter in optimizer.state and bool(optimizer.state[parameter])
    before = [float(value) for value in parameter.detach().cpu().tolist()]
    with torch_module.no_grad():
        parameter.fill_(target_value)
    optimizer.state.pop(parameter, None)
    if getattr(policy, name) is not parameter or _optimizer_index(optimizer, parameter) != index:
        raise RuntimeError("std reset replaced the parameter object")
    after = [float(value) for value in parameter.detach().cpu().tolist()]
    moment_after = parameter in optimizer.state and bool(optimizer.state[parameter])
    policy_after = T.state_digest(policy.state_dict())
    optimizer_after = T.state_digest(optimizer.state_dict())
    target = float32(target_value)
    if not after or len(after) != len(before) or any(value != target for value in after):
        raise ValueError("std readback differs from the registered initial value")
    if moment_after or optimizer_after["sha256"] != expected_optimizer["sha256"]:
        raise ValueError("std reset changed optimizer state beyond the std moments")
    rows_before, rows_after = _tensor_rows(policy_before), _tensor_rows(policy_after)
    if set(rows_before) != set(rows_after):
        raise ValueError("std reset changed the policy state layout")
    path = f'$["{name}"]'
    changed = sorted(item for item in rows_after if rows_after[item] != rows_before[item])
    if changed not in ([], [path]) or bool(changed) != (before != after):
        raise ValueError("std reset changed a tensor other than the action std")
    effective = (lambda values: values) if name == "std" else (lambda values: [math.exp(v) for v in values])
    return {
        "schema": "action_std_reset_v2", "parameter": name, "noise_std_type": noise_type,
        "init_noise_std": init_noise_std, "parameter_target_float32": target,
        "before": before, "after": after, "after_matches_init": True,
        "std_mean_before": _mean(effective(before)), "std_mean_after": _mean(effective(after)),
        "policy_state_before_reset": policy_before["sha256"],
        "policy_state_after_reset": policy_after["sha256"],
        "changed_tensor_paths": changed, "parameter_in_optimizer": True,
        "optimizer_param_index": index,
        "optimizer_std_moment_present_before": moment_before,
        "optimizer_std_moment_present_after": moment_after,
        "optimizer_state_before_reset": optimizer_before["sha256"],
        "optimizer_state_expected_after_reset": expected_optimizer["sha256"],
        "optimizer_state_after_reset": optimizer_after["sha256"],
        "optimizer_other_state_preserved": True,
    }


def _install_runner_hooks(manifest: dict, H, runtime: dict[str, Any], *,
                          expected_action_scale: float | None = None,
                          expected_rsl_version: str | None = None,
                          expected_isaaclab_commit: str | None = None,
                          on_learning_complete: Callable[[], None] | None = None) -> tuple[Any, Any, Any, type]:
    mode = validate_manifest_mode(manifest)
    runners = importlib.import_module("rsl_rl.runners")
    runner_class = runners.OnPolicyRunner
    original_init = runner_class.__init__
    original_load = runner_class.load
    original_learn = runner_class.learn
    output_dir = Path(manifest["output_dir"])
    runtime.setdefault("checkpoint_load_calls", 0)

    def measured_init(self, *args, **kwargs):
        if runtime.get("runner_initialized"):
            raise RuntimeError("training created more than one OnPolicyRunner")
        original_init(self, *args, **kwargs)
        train_cfg = args[1] if len(args) > 1 else kwargs.get("train_cfg")
        observations = T._runner_observations(self, train_cfg, manifest, expected_action_scale)
        versions = T._runtime_versions()
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
        runtime["checkpoint_load_calls"] += 1
        if mode == "restart":
            raise RuntimeError("restart arm must not load a checkpoint")
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
        source_state = T._checkpoint_state(expected_path, manifest["resume_iteration"], torch)
        result = original_load(self, *args, **kwargs)
        if H.sha(expected_path, lf=False) != manifest["resume_checkpoint_sha256"]:
            raise ValueError("resume checkpoint changed during load")
        if self.current_learning_iteration != manifest["resume_iteration"]:
            raise ValueError("loaded checkpoint iteration differs from the resume manifest")
        current_policy = T.state_digest(self.alg.policy.state_dict())
        current_optimizer = T.state_digest(self.alg.optimizer.state_dict())
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
        if mode == "resume_std_reset":
            runtime["std_reset"] = reset_action_std(
                self.alg.policy, self.alg.optimizer, manifest["init_noise_std"], torch)
            if runtime["std_reset"]["policy_state_before_reset"] != source_state["policy_state_sha256"]:
                raise ValueError("std reset did not start from the loaded checkpoint policy")
        return result

    def measured_learn(self, *args, **kwargs):
        if runtime.get("learn_called"):
            raise RuntimeError("runner.learn was called more than once")
        if mode == "restart":
            if runtime["checkpoint_load_calls"] or runtime.get("checkpoint_loaded"):
                raise RuntimeError("restart arm loaded a checkpoint")
            if self.current_learning_iteration != 0:
                raise ValueError("restart arm must start from iteration 0")
        elif not runtime.get("checkpoint_loaded"):
            raise RuntimeError("runner.learn requires the registered resume checkpoint")
        if mode == "resume_std_reset" and "std_reset" not in runtime:
            raise RuntimeError("std reset arm reached learning without the registered reset")
        iterations = kwargs.get("num_learning_iterations", args[0] if args else None)
        random_ep = kwargs.get("init_at_random_ep_len", args[1] if len(args) > 1 else None)
        if iterations != manifest["max_iterations"] or random_ep is not True:
            raise ValueError("runner.learn arguments differ from the fixed training budget")
        H.validate_training_configuration(runtime["log_dir"], manifest)
        import torch
        rng_states = T._rng_state_hashes(torch)
        runtime["rng_state_before_learn"] = rng_states
        runtime["start_iteration"] = self.current_learning_iteration
        resumed = mode != "restart"
        initial = {
            "schema": "training_correction_initial_v2",
            "recovery_mode": mode,
            "policy_state": T.state_digest(self.alg.policy.state_dict()),
            "optimizer_state": T.state_digest(self.alg.optimizer.state_dict()),
            "observed": runtime["observed"],
            "runtime_versions": runtime["runtime_versions"],
            "log_dir": str(runtime["log_dir"]),
            "resume_checkpoint": {
                "path": str(runtime["resume_checkpoint_path"]),
                "sha256": runtime["resume_checkpoint_sha256"],
                "load_optimizer": runtime["load_optimizer"],
            } if resumed else None,
            "resume_iteration": runtime["resume_iteration"] if resumed else None,
            "start_iteration": runtime["start_iteration"],
            "init_at_random_ep_len": True,
            "source_checkpoint_state": runtime["source_checkpoint_state"] if resumed else None,
            "std_reset": runtime.get("std_reset"),
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
    argv = [
        str(upstream), "--task", manifest["task"], "--seed", str(manifest["seed"]),
        "--num_envs", str(manifest["num_envs"]), "--max_iterations", str(manifest["max_iterations"]),
        "--headless", "--run_name", manifest["run_name"],
    ]
    if manifest["recovery_mode"] != "restart":
        argv += ["--resume", "--load_run", manifest["load_run_regex"], "--checkpoint", manifest["checkpoint_regex"]]
    return argv


def execute(manifest_path: Path, H) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    manifest, protocol = H.validate_manifest(manifest_path)
    validate_manifest_mode(manifest)
    isaaclab_root = Path(manifest["isaaclab_root"]).resolve()
    if Path.cwd().resolve() != isaaclab_root:
        raise ValueError("training correction bootstrap must run from the Isaac Lab root")
    upstream = isaaclab_root / "scripts" / "reinforcement_learning" / "rsl_rl" / "train.py"
    if not upstream.is_file():
        raise FileNotFoundError(upstream)
    protocol_path = Path(manifest["protocol_path"]).resolve()
    if H.sha(protocol_path, lf=True) != manifest["protocol_sha256_lf"]:
        raise ValueError("protocol LF SHA differs from the validated manifest")
    output_dir = Path(manifest["output_dir"]).resolve()
    correction_path = output_dir / "correction.json"
    if any((output_dir / name).exists() for name in ("initial.json", "correction.json", "receipt.json")):
        raise FileExistsError("training correction artifacts are write-once")

    import correction_adapter as A
    import hidden_faults

    original_before = hidden_faults._before_env
    runtime: dict[str, Any] = {"environment_make_calls": 0}
    runner_hooks = None

    def record_learning_complete() -> None:
        # Isaac fast shutdown can end the process without returning from runpy.
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
    mode = validate_manifest_mode(manifest)
    resumed = mode != "restart"
    if (not runtime.get("correction_applied") or not runtime.get("runner_initialized")
            or not runtime.get("learning_completed")):
        raise RuntimeError("training did not complete correction and learning")
    if resumed and not runtime.get("checkpoint_loaded"):
        raise RuntimeError("resume arm completed without its checkpoint")
    if not resumed and runtime.get("checkpoint_load_calls"):
        raise RuntimeError("restart arm loaded a checkpoint")
    output_dir = Path(manifest["output_dir"]).resolve()
    protocol_path = Path(manifest["protocol_path"]).resolve()
    protocol_sha = H.sha(protocol_path, lf=True)
    if protocol_sha != manifest["protocol_sha256_lf"]:
        raise ValueError("protocol changed during learning")
    log_dir = runtime["log_dir"]
    final_iteration = manifest["final_iteration"]
    checkpoint = log_dir / f"model_{final_iteration}.pt"
    import torch
    final_checkpoint_state = T._checkpoint_state(checkpoint, final_iteration, torch)
    configs = {
        "environment": T._artifact(log_dir / "params" / "env.yaml", H),
        "agent": T._artifact(log_dir / "params" / "agent.yaml", H),
    }
    import runtime_readback_smoke as S
    receipt = {
        "schema": "training_correction_receipt_v2", "completed": True,
        "tag": manifest["tag"], "seed": manifest["seed"], "arm": manifest["arm"],
        "recovery_mode": mode, "fault": manifest["fault"], "correction_id": manifest["correction_id"],
        "source_commit_sha": manifest["source_commit_sha"],
        "protocol_sha256_lf": manifest["protocol_sha256_lf"],
        "protocol": {"path": str(protocol_path), "sha256_lf": protocol_sha},
        "manifest_sha256": H.sha(manifest_path, lf=False),
        "initial_sha256": H.sha(output_dir / "initial.json", lf=False),
        "correction_sha256": H.sha(output_dir / "correction.json", lf=False),
        "actual_control_steps": runtime["actual_control_steps"],
        "resume_checkpoint": {
            "path": str(runtime["resume_checkpoint_path"]),
            "sha256": runtime["resume_checkpoint_sha256"],
        } if resumed else None,
        "resume_iteration": runtime["resume_iteration"] if resumed else None,
        "start_iteration": runtime["start_iteration"], "final_iteration": final_iteration,
        "checkpoint_load_calls": runtime["checkpoint_load_calls"],
        "source_checkpoint_state": runtime["source_checkpoint_state"] if resumed else None,
        "final_checkpoint_state": final_checkpoint_state,
        "std_reset_applied": "std_reset" in runtime,
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
            "policy_optimizer_restored": resumed,
            "full_environment_state_restored": False,
            "rng_state_restored_from_checkpoint": False,
            "fresh_initialization": not resumed,
            "action_std_reset": mode == "resume_std_reset",
        },
        "process_identity": S.process_identity(),
        "checkpoint": T._artifact(checkpoint, H), "configs": configs,
    }
    H.write_new(output_dir / "receipt.json", receipt)
    return receipt


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args(argv)
    if str(EVALS) not in sys.path:
        sys.path.insert(0, str(EVALS))
    import run_training_correction_v4 as H
    execute(args.manifest, H)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
