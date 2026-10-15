"""학습 루프 계측 실행기와 기록 판정 CLI.

CPU 판정:
  python bench/inspect_optimizer_observability.py inspect --manifest manifest.json --records records.jsonl --output report.json

Isaac Lab 학습(메인 GPU 실행 전 protocol·코드 commit 필요):
  _isaac_sim/python.bat <repo>/bench/inspect_optimizer_observability.py train --manifest manifest.json \
    --records records.jsonl --runtime-set num_learning_epochs=1 -- --task ...
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import runpy
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))
import optimizer_observability as O  # noqa: E402

PROTOCOL = ROOT / "bench/protocols/optimizer_observability_v1.json"
BOUND_SOURCES = (ROOT / "evals/optimizer_observability.py", Path(__file__).resolve(),
                 ROOT / "bench/hidden_faults.py", PROTOCOL)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_new(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def sha256_lf(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def build_manifest(epochs: int, mini_batches: int, updates: int, *, arm: str, seed: int, task: str,
                   num_envs: int, max_iterations: int) -> dict[str, Any]:
    if min(epochs, mini_batches, updates, num_envs, max_iterations) <= 0:
        raise ValueError("epochs, mini-batches, updates, envs, iterations는 양수여야 한다")
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], check=True,
                            capture_output=True, text=True).stdout.strip()
    expected = epochs * mini_batches
    return {
        "schema": O.MANIFEST_SCHEMA,
        "configured_learning_epochs": epochs,
        "configured_mini_batches": mini_batches,
        "expected_optimizer_steps_per_update": expected,
        "expected_minibatches_per_update": expected,
        "runtime_updates_to_observe": updates,
        "inspection_primitives": list(O.PRIMITIVES),
        "arm": arm,
        "seed": seed,
        "task": task,
        "num_envs": num_envs,
        "max_iterations": max_iterations,
        "source_commit_sha": commit,
        "protocol_path": PROTOCOL.relative_to(ROOT).as_posix(),
        "protocol_sha256_lf": sha256_lf(PROTOCOL),
        "source_sha256_lf": {path.relative_to(ROOT).as_posix(): sha256_lf(path) for path in BOUND_SOURCES[:-1]},
    }


def verify_commitment(manifest: dict[str, Any], require_committed: bool = False) -> None:
    O.validate_manifest(manifest)
    if manifest.get("protocol_path") != PROTOCOL.relative_to(ROOT).as_posix():
        raise ValueError("protocol path commitment가 다르다")
    if manifest.get("protocol_sha256_lf") != sha256_lf(PROTOCOL):
        raise ValueError("protocol hash commitment가 다르다")
    expected = {path.relative_to(ROOT).as_posix(): sha256_lf(path) for path in BOUND_SOURCES[:-1]}
    if manifest.get("source_sha256_lf") != expected:
        raise ValueError("source hash commitment가 다르다")
    if require_committed:
        paths = [path.relative_to(ROOT).as_posix() for path in BOUND_SOURCES]
        tracked = subprocess.run(["git", "-C", str(ROOT), "ls-files", "--error-unmatch", *paths],
                                 capture_output=True).returncode == 0
        clean = subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet", "HEAD", "--", *paths]).returncode == 0
        head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], check=True,
                              capture_output=True, text=True).stdout.strip()
        if not tracked or not clean or manifest.get("source_commit_sha") != head:
            raise ValueError("GPU 실행 전 결합 소스·protocol을 같은 HEAD에 커밋해야 한다")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"records {number}번째 줄이 JSON이 아님") from error
    return rows


def _runtime_assignment(text: str | None) -> tuple[str, Any] | None:
    if text is None:
        return None
    if "=" not in text:
        raise ValueError("--runtime-set은 이름=JSON값 형식이어야 한다")
    name, raw = text.split("=", 1)
    if not name.isidentifier() or name.startswith("_"):
        raise ValueError("runtime field 이름이 안전하지 않다")
    return name, json.loads(raw)


def _upstream_value(arguments: list[str], name: str) -> str | None:
    matches = []
    for index, item in enumerate(arguments):
        if item == name:
            matches.append(arguments[index + 1] if index + 1 < len(arguments) else None)
        if item.startswith(name + "="):
            matches.append(item.split("=", 1)[1])
    if len(matches) > 1:
        raise ValueError(f"중복 학습 인자: {name}")
    return matches[0] if matches else None


def verify_training_binding(manifest: dict[str, Any], runtime_set: str | None, hidden_fault: str | None,
                            upstream_args: list[str]) -> None:
    allowed = {"--task", "--seed", "--num_envs", "--max_iterations", "--run_name"}
    index = 0
    while index < len(upstream_args):
        option = upstream_args[index]
        if option == "--headless":
            index += 1
        elif option.split("=", 1)[0] in allowed:
            index += 1 if "=" in option else 2
        else:
            raise ValueError(f"등록하지 않은 학습 인자: {option}")
    values = {"task": _upstream_value(upstream_args, "--task"),
              "seed": _upstream_value(upstream_args, "--seed"),
              "num_envs": _upstream_value(upstream_args, "--num_envs"),
              "max_iterations": _upstream_value(upstream_args, "--max_iterations")}
    expected = {"task": str(manifest["task"]), "seed": str(manifest["seed"]),
                "num_envs": str(manifest["num_envs"]), "max_iterations": str(manifest["max_iterations"])}
    if values != expected:
        raise ValueError(f"학습 인자가 manifest binding과 다르다: {values} != {expected}")
    arm_options = {"NONE": (None, "NONE"), "R1": ("num_learning_epochs=1", None),
                   "h01": (None, "F5"), "h02": (None, "F1"), "h03": (None, "F6"),
                   "h04": (None, "F4"), "h05": (None, "F3"), "h06": (None, "F2")}
    if manifest["arm"] not in arm_options or (runtime_set, hidden_fault) != arm_options[manifest["arm"]]:
        raise ValueError("arm과 runtime/hidden fault 옵션이 manifest와 다르다")


def train(manifest_path: Path, records_path: Path, runtime_set: str | None, hidden_fault: str | None,
          upstream_args: list[str]) -> None:
    """PPO 생성 때 observer를 붙인 뒤 Isaac Lab 상류 train.py를 실행한다."""
    manifest = O.validate_manifest(read_json(manifest_path))
    verify_commitment(manifest, require_committed=True)
    assignment = _runtime_assignment(runtime_set)
    if assignment is not None and hidden_fault is not None:
        raise ValueError("runtime field 변경과 hidden fault는 한 arm에 함께 쓰지 않는다")
    verify_training_binding(manifest, runtime_set, hidden_fault, upstream_args)
    if records_path.exists():
        raise FileExistsError(f"기록 파일이 이미 있다: {records_path}")
    upstream = Path.cwd() / "scripts" / "reinforcement_learning" / "rsl_rl" / "train.py"
    if not upstream.is_file():
        raise FileNotFoundError(f"Isaac Lab 폴더에서 실행해야 한다: {upstream}")
    records_path.parent.mkdir(parents=True, exist_ok=True)
    import gymnasium
    original_make = gymnasium.make
    original_path, original_argv = sys.path[:], sys.argv[:]
    fault_restores = []
    state = {"installed": False, "observers": []}

    def sink(record: dict[str, Any]) -> None:
        with records_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")

    def make(*args, **kwargs):
        first_environment = not state["installed"]
        if not state["installed"]:
            state["installed"] = True
            ppo_class = importlib.import_module("rsl_rl.algorithms.ppo").PPO
            original_init = ppo_class.__init__

            def observed_init(self, *init_args, **init_kwargs):
                original_init(self, *init_args, **init_kwargs)
                if assignment is not None:
                    setattr(self, assignment[0], assignment[1])
                observer = O.RuntimeObserver(self, manifest, sink).install()
                state["observers"].append(observer)

            ppo_class.__init__ = observed_init
            state["ppo_class"], state["original_init"] = ppo_class, original_init
            if hidden_fault not in (None, "NONE"):
                targets = {
                    "F1": ("isaaclab.envs.mdp.rewards", "track_lin_vel_xy_exp", "__code__"),
                    "F2": ("isaaclab.actuators.actuator_pd", "DCMotor", "_clip_effort"),
                    "F3": ("rsl_rl.modules.actor_critic", "ActorCritic", "act"),
                    "F4": ("rsl_rl.runners", "OnPolicyRunner", "__init__"),
                    "F6": ("isaaclab.envs.mdp.terminations", "time_out", "__code__"),
                }
                if hidden_fault in targets:
                    mod, obj, attr = targets[hidden_fault]
                    target = getattr(importlib.import_module(mod), obj)
                    fault_restores.append((target, attr, getattr(target, attr)))
                fault_module._before_env(hidden_fault)
        env = original_make(*args, **kwargs)
        if hidden_fault is not None and first_environment:
            readback = fault_module._after_env(hidden_fault, env)
            print("[hidden_fault] " + json.dumps({"fault": hidden_fault, **readback}), flush=True)
        return env

    try:
        fault_module = None
        if hidden_fault is not None:
            sys.path.insert(0, str(ROOT / "bench"))
            fault_module = importlib.import_module("hidden_faults")
        gymnasium.make = make
        sys.path.insert(0, str(upstream.parent))
        sys.argv = [str(upstream), *upstream_args]
        runpy.run_path(str(upstream), run_name="__main__")
    finally:
        gymnasium.make = original_make
        for observer in state["observers"]:
            observer.remove()
        if "ppo_class" in state:
            state["ppo_class"].__init__ = state["original_init"]
        for target, attribute, previous in reversed(fault_restores):
            setattr(target, attribute, previous)
        sys.argv = original_argv
        sys.path[:] = original_path


def checkpoint_steps(path: Path) -> dict[str, Any]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    optimizer = payload.get("optimizer_state_dict")
    if not isinstance(optimizer, dict) or not isinstance(optimizer.get("state"), dict):
        raise ValueError("optimizer_state_dict.state가 없다")
    steps = {}
    for key, state in optimizer["state"].items():
        if not isinstance(state, dict) or "step" not in state:
            continue
        value = state["step"]
        steps[str(key)] = float(value.item() if hasattr(value, "item") else value)
    return {"path": str(path), "steps": steps}


def verify_runner_checkpoint_semantics(path: Path, expected_sha256_lf: str) -> bool:
    if not path.is_file() or sha256_lf(path) != expected_sha256_lf:
        return False
    text = path.read_text(encoding="utf-8")
    update = text.find("loss_dict = self.alg.update()")
    assign = text.find("self.current_learning_iteration = it", update)
    save = text.find('f"model_{it}.pt"', assign)
    return update >= 0 and update < assign < save


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("inspect")
    inspect.add_argument("--manifest", type=Path, required=True)
    inspect.add_argument("--records", type=Path, required=True)
    inspect.add_argument("--output", type=Path, required=True)
    prepare = sub.add_parser("prepare-manifest")
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--epochs", type=int, default=5)
    prepare.add_argument("--mini-batches", type=int, default=4)
    prepare.add_argument("--updates", type=int, default=3)
    prepare.add_argument("--arm", choices=("NONE", "R1", "h01", "h02", "h03", "h04", "h05", "h06"), required=True)
    prepare.add_argument("--seed", type=int, required=True)
    prepare.add_argument("--task", required=True)
    prepare.add_argument("--num-envs", type=int, required=True)
    prepare.add_argument("--max-iterations", type=int, required=True)
    compare = sub.add_parser("compare-checkpoints")
    compare.add_argument("--before", type=Path, required=True)
    compare.add_argument("--after", type=Path, required=True)
    compare.add_argument("--metadata", type=Path, required=True)
    compare.add_argument("--runner-source", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("train")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--records", type=Path, required=True)
    run.add_argument("--runtime-set")
    run.add_argument("--hidden-fault", choices=("NONE", "F1", "F2", "F3", "F4", "F5", "F6"))
    run.add_argument("upstream", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare-manifest":
            write_new(args.output, build_manifest(args.epochs, args.mini_batches, args.updates, arm=args.arm,
                                                  seed=args.seed, task=args.task, num_envs=args.num_envs,
                                                  max_iterations=args.max_iterations))
        elif args.command == "inspect":
            manifest = read_json(args.manifest)
            verify_commitment(manifest)
            write_new(args.output, O.assemble_inspection(read_jsonl(args.records), manifest))
        elif args.command == "compare-checkpoints":
            metadata = read_json(args.metadata)
            metadata["checkpoint_semantics_source_verified"] = verify_runner_checkpoint_semantics(
                args.runner_source, metadata.get("runner_source_sha256_lf", ""))
            report = O.compare_checkpoint_adam_steps(checkpoint_steps(args.before), checkpoint_steps(args.after), metadata)
            write_new(args.output, report)
        else:
            upstream = args.upstream[1:] if args.upstream[:1] == ["--"] else args.upstream
            train(args.manifest, args.records, args.runtime_set, args.hidden_fault, upstream)
    except (ValueError, FileNotFoundError, FileExistsError, json.JSONDecodeError,
            subprocess.CalledProcessError) as error:
        print(f"거부: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
