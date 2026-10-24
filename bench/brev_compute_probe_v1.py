"""Measure a pinned Isaac Lab Go2 training workload on a provisioned Brev host.

This worker records environment compatibility and throughput evidence.  It does
not diagnose training failures or claim recovery performance.  A parent process
owns wall-clock timeout enforcement and must validate both this process exit and
the write-once JSON report.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import re
import runpy
import sys
import time
from pathlib import Path
from typing import Any, Callable

SCHEMAS = {
    "metadata": "brev_compute_metadata_v1",
    "empty-stage": "brev_compute_empty_stage_v1",
    "train": "brev_compute_train_v1",
}
TASK = "Isaac-Velocity-Flat-Unitree-Go2-v0"
NUM_ENVS = 4096
ITERATION_BUDGETS = (20, 300)
PACKAGE_NAMES = ("isaacsim", "isaaclab", "isaaclab_rl", "torch", "rsl-rl-lib", "gymnasium")
RUN_NAME_PATTERN = re.compile(r"[a-z0-9](?:[a-z0-9_-]{0,78}[a-z0-9])?\Z")


def _package_versions() -> dict[str, str | None]:
    from importlib.metadata import PackageNotFoundError, version

    packages: dict[str, str | None] = {}
    for name in PACKAGE_NAMES:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return packages


def _python_metadata() -> dict[str, str]:
    return {
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
    }


def _torch_metadata() -> dict[str, Any]:
    try:
        torch = importlib.import_module("torch")
    except (ImportError, OSError):
        return {"imported": False, "cuda_available": False, "device_count": 0, "devices": []}

    available = bool(torch.cuda.is_available())
    count = int(torch.cuda.device_count()) if available else 0
    devices = []
    for index in range(count):
        properties = torch.cuda.get_device_properties(index)
        devices.append({
            "index": index,
            "name": str(properties.name),
            "compute_capability": [int(properties.major), int(properties.minor)],
            "total_memory_bytes": int(properties.total_memory),
        })
    get_arch_list = getattr(torch.cuda, "get_arch_list", None)
    return {
        "imported": True,
        "version": str(torch.__version__),
        "built_cuda_version": getattr(torch.version, "cuda", None),
        "cuda_available": available,
        "device_count": count,
        "compiled_architectures": list(get_arch_list()) if available and get_arch_list is not None else [],
        "devices": devices,
    }


def _base_report(mode: str, timeout_s: float) -> dict[str, Any]:
    return {
        "schema": SCHEMAS[mode],
        "mode": mode,
        "completed": False,
        "scope": "compatibility_and_throughput_only_not_diagnostic_or_recovery_evidence",
        "timeout_s": timeout_s,
        "timeout_enforcement": "parent_controller",
        "python": _python_metadata(),
        "packages": _package_versions(),
    }


def _typed_error(error: BaseException, stage: str) -> dict[str, str]:
    """Return a bounded error classification without exception text or paths."""
    if isinstance(error, SystemExit):
        kind = "UpstreamEarlyExit"
    elif isinstance(error, (ImportError, ModuleNotFoundError)):
        kind = "DependencyImportError"
    elif isinstance(error, FileNotFoundError):
        kind = "RequiredFileMissing"
    elif isinstance(error, FileExistsError):
        kind = "OutputCollision"
    else:
        kind = type(error).__name__
    return {"type": kind, "stage": stage}


def _write_new(output: Path, report: dict[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read_completion(output: Path) -> bool:
    try:
        body = json.loads(output.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    return body.get("completed") is True


def _failure(output: Path, mode: str, timeout_s: float, error: BaseException, stage: str,
             started: float) -> None:
    if output.exists():
        return
    report = _base_report(mode, timeout_s)
    report.update({
        "error": _typed_error(error, stage),
        "wall_s": time.perf_counter() - started,
        "wall_scope": "worker_start_to_failure_report_before_app_close",
        "shutdown_verification": "parent_exit_and_cleanup_receipt_required",
    })
    _write_new(output, report)


def run_metadata(output: Path, timeout_s: float) -> None:
    started = time.perf_counter()
    report = _base_report("metadata", timeout_s)
    report.update({
        "completed": True,
        "torch": _torch_metadata(),
        "wall_s": time.perf_counter() - started,
        "wall_scope": "worker_start_to_report_write",
        "shutdown_verification": "parent_exit_required",
    })
    _write_new(output, report)


def run_empty_stage(output: Path, timeout_s: float) -> None:
    started = time.perf_counter()
    try:
        from isaaclab.app import AppLauncher  # pyright: ignore[reportMissingImports]
    except BaseException as error:
        _failure(output, "empty-stage", timeout_s, error, "dependency_import", started)
        raise

    app = AppLauncher(headless=True).app
    sim = None
    try:
        from isaaclab.sim import SimulationCfg, SimulationContext  # pyright: ignore[reportMissingImports]

        sim = SimulationContext(SimulationCfg(dt=0.01))
        sim.reset()
        sim.step()
        report = _base_report("empty-stage", timeout_s)
        report.update({
            "completed": True,
            "torch": _torch_metadata(),
            "checkpoint_loaded": False,
            "robot_environments": 0,
            "empty_stage_physics_steps": 1,
            "wall_s": time.perf_counter() - started,
            "wall_scope": "worker_start_to_report_before_app_close",
            "shutdown_verification": "parent_exit_and_cleanup_receipt_required",
        })
        _write_new(output, report)
    except BaseException as error:
        _failure(output, "empty-stage", timeout_s, error, "empty_stage", started)
        raise
    finally:
        try:
            if sim is not None:
                try:
                    sim.clear_all_callbacks()
                finally:
                    sim.clear_instance()
        finally:
            app.close()


def _learn_iterations(args: tuple[Any, ...], kwargs: dict[str, Any]) -> int | None:
    value = kwargs.get("num_learning_iterations", args[0] if args else None)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _latest_checkpoint(log_dir: Path) -> Path:
    candidates: list[tuple[int, Path]] = []
    for path in log_dir.glob("model_*.pt"):
        match = re.fullmatch(r"model_(\d+)\.pt", path.name)
        if match and path.is_file():
            candidates.append((int(match.group(1)), path))
    if not candidates:
        raise FileNotFoundError("checkpoint unavailable")
    return max(candidates, key=lambda item: item[0])[1]


def _runner_log_dir(runner: Any) -> Path:
    value = getattr(runner, "log_dir", None)
    if not isinstance(value, (str, os.PathLike)):
        raise RuntimeError("runner log directory unavailable")
    return Path(value)


def _observed_environment(runner: Any) -> tuple[str | None, int | None]:
    env = getattr(runner, "env", None)
    num_envs = getattr(env, "num_envs", None)
    unwrapped = getattr(env, "unwrapped", env)
    spec = getattr(unwrapped, "spec", None)
    task = getattr(spec, "id", None)
    return (task if isinstance(task, str) else None,
            num_envs if isinstance(num_envs, int) and not isinstance(num_envs, bool) else None)


def _training_argv(upstream: Path, seed: int, iterations: int, run_name: str) -> list[str]:
    return [
        str(upstream),
        "--task", TASK,
        "--seed", str(seed),
        "--num_envs", str(NUM_ENVS),
        "--max_iterations", str(iterations),
        "--headless",
        "--run_name", run_name,
    ]


def run_train(output: Path, isaaclab: Path, seed: int, iterations: int, run_name: str,
              timeout_s: float, run_path: Callable[..., dict[str, Any]] = runpy.run_path) -> None:
    started = time.perf_counter()
    isaaclab = isaaclab.resolve()
    upstream = isaaclab / "scripts" / "reinforcement_learning" / "rsl_rl" / "train.py"
    if not upstream.is_file():
        error = FileNotFoundError("upstream trainer unavailable")
        _failure(output, "train", timeout_s, error, "preflight", started)
        raise error

    try:
        runner_class = importlib.import_module("rsl_rl.runners").OnPolicyRunner
    except BaseException as error:
        _failure(output, "train", timeout_s, error, "dependency_import", started)
        raise
    original_learn = runner_class.learn
    old_argv, old_path, old_cwd = sys.argv[:], sys.path[:], Path.cwd()
    state: dict[str, Any] = {"learn_calls": 0}

    def measured_learn(self: Any, *args: Any, **kwargs: Any) -> Any:
        state["learn_calls"] += 1
        if state["learn_calls"] != 1:
            error = RuntimeError("unexpected runner count")
            _failure(output, "train", timeout_s, error, "learn_contract", started)
            raise error
        observed_argument = _learn_iterations(args, kwargs)
        if observed_argument != iterations:
            error = ValueError("learn iteration budget mismatch")
            _failure(output, "train", timeout_s, error, "learn_contract", started)
            raise error
        learn_started = time.perf_counter()
        initial_iteration = getattr(self, "current_learning_iteration", None)
        try:
            result = original_learn(self, *args, **kwargs)
        except BaseException as error:
            _failure(output, "train", timeout_s, error, "learn", started)
            raise
        learn_finished = time.perf_counter()
        final_iteration = getattr(self, "current_learning_iteration", None)
        observed_iterations = (
            final_iteration - initial_iteration + 1
            if isinstance(initial_iteration, int) and isinstance(final_iteration, int)
            else None
        )
        if observed_iterations is None or observed_iterations != iterations:
            error = RuntimeError("observed iteration count mismatch")
            _failure(output, "train", timeout_s, error, "learn_contract", started)
            raise error
        observed_task, observed_num_envs = _observed_environment(self)
        if observed_task != TASK or observed_num_envs != NUM_ENVS:
            error = RuntimeError("observed environment contract mismatch")
            _failure(output, "train", timeout_s, error, "learn_contract", started)
            raise error
        num_steps_per_env = getattr(self, "num_steps_per_env", None)
        if not isinstance(num_steps_per_env, int) or isinstance(num_steps_per_env, bool) or num_steps_per_env <= 0:
            error = RuntimeError("runner rollout length unavailable")
            _failure(output, "train", timeout_s, error, "learn_contract", started)
            raise error
        checkpoint = _latest_checkpoint(_runner_log_dir(self))
        aggregate_environment_steps = iterations * num_steps_per_env * observed_num_envs
        learn_wall_s = learn_finished - learn_started
        report = _base_report("train", timeout_s)
        report.update({
            "completed": True,
            "torch": _torch_metadata(),
            "task": TASK,
            "observed_task": observed_task,
            "seed": seed,
            "run_name": run_name,
            "num_envs": NUM_ENVS,
            "observed_num_envs": observed_num_envs,
            "requested_iterations": iterations,
            "learn_arguments": {
                "num_learning_iterations": observed_argument,
                "init_at_random_ep_len": kwargs.get("init_at_random_ep_len", args[1] if len(args) > 1 else None),
            },
            "initial_learning_iteration": initial_iteration,
            "final_learning_iteration": final_iteration,
            "observed_iterations": observed_iterations,
            "num_steps_per_env": num_steps_per_env,
            "aggregate_environment_steps": aggregate_environment_steps,
            "aggregate_environment_steps_scope": "num_envs_times_rollout_steps_times_observed_iterations",
            "aggregate_environment_steps_per_s": aggregate_environment_steps / learn_wall_s,
            "iterations_per_s": observed_iterations / learn_wall_s,
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256_raw": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "script_start_to_learn_start_s": learn_started - started,
            "learn_wall_s": learn_wall_s,
            "learn_wall_scope": "OnPolicyRunner.learn_call_only",
            "wall_s": learn_finished - started,
            "wall_scope": "worker_start_to_learn_return_before_upstream_app_close",
            "shutdown_verification": "parent_exit_and_cleanup_receipt_required",
        })
        _write_new(output, report)
        return result

    runner_class.learn = measured_learn
    sys.argv = _training_argv(upstream, seed, iterations, run_name)
    sys.path.insert(0, str(upstream.parent))
    try:
        os.chdir(isaaclab)
        run_path(str(upstream), run_name="__main__")
        if not output.exists():
            error = RuntimeError("upstream returned before learning")
            _failure(output, "train", timeout_s, error, "upstream", started)
            raise error
    except BaseException as error:
        _failure(output, "train", timeout_s, error, "upstream", started)
        raise
    finally:
        runner_class.learn = original_learn
        sys.argv, sys.path[:] = old_argv, old_path
        os.chdir(old_cwd)


def _positive_timeout(value: str) -> float:
    parsed = float(value)
    if not 0 < parsed < float("inf"):
        raise argparse.ArgumentTypeError("timeout must be finite and positive")
    return parsed


def _run_name(value: str) -> str:
    if len(value) > 80 or RUN_NAME_PATTERN.fullmatch(value) is None:
        raise argparse.ArgumentTypeError("run name must be a slug of at most 80 characters")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=tuple(SCHEMAS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--isaaclab", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--run-name", type=_run_name)
    parser.add_argument("--timeout-s", type=_positive_timeout, required=True)
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.output.exists():
        parser.error("output already exists")
    if args.mode != "train":
        forbidden = (args.seed, args.num_envs, args.iterations, args.run_name)
        if any(value is not None for value in forbidden):
            parser.error("training arguments are valid only in train mode")
        return
    if args.isaaclab is None or args.seed is None or args.iterations is None or args.run_name is None:
        parser.error("train requires --isaaclab, --seed, --iterations, and --run-name")
    if args.num_envs != NUM_ENVS:
        parser.error("train requires --num-envs 4096")
    if args.iterations not in ITERATION_BUDGETS:
        parser.error("train iterations must be 20 or 300")


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)
    started = time.perf_counter()
    try:
        if args.mode == "metadata":
            run_metadata(args.output, args.timeout_s)
        elif args.mode == "empty-stage":
            run_empty_stage(args.output, args.timeout_s)
        else:
            run_train(args.output, args.isaaclab, args.seed, args.iterations, args.run_name, args.timeout_s)
    except BaseException as error:
        try:
            _failure(args.output, args.mode, args.timeout_s, error, "mode_entrypoint", started)
        except BaseException:
            return 1
        return 0 if isinstance(error, SystemExit) and error.code == 0 and _read_completion(args.output) else 1
    return 0 if _read_completion(args.output) else 1


if __name__ == "__main__":
    raise SystemExit(main())
