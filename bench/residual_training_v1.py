"""F1+R1 복합 결함과 최대 두 등록 교정을 같은 학습에서 재현한다."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from itertools import permutations
from pathlib import Path
import re
import runpy
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
for _directory in (ROOT / "bench", ROOT / "evals"):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

import optimizer_observability as OBSERVE  # noqa: E402
import residual_faults_v1 as FAULTS  # noqa: E402

SCHEMA = "residual_training_manifest_v1"
TASK = "Isaac-Velocity-Flat-Unitree-Go2-v0"
ARMS = ("NONE", "F1", "R1", "COMPOUND", "FIRST", "BOTH")
MODES = {"train": 300, "inspection": 3}
FIXED = {"num_envs": 4096, "configured_learning_epochs": 5,
         "configured_mini_batches": 4, "runtime_updates_to_observe": 3}
REQUIRED = {
    "schema", "mode", "arm", "seed", "run_name", "task", "num_envs", "max_iterations",
    "configured_learning_epochs", "configured_mini_batches", "runtime_updates_to_observe",
    "source_commit_sha", "protocol_path", "protocol_sha256_lf", "source_sha256_lf",
    "runtime_source_sha256_lf", "records_path", "readback_path", "source_checkpoint_sha256",
    "original_faults", "applied_corrections",
}
MINIMUM_BOUND_SOURCES = {
    "bench/residual_faults_v1.py", "bench/residual_training_v1.py", "bench/residual_probe_v1.py",
    "bench/r1_repair_training.py", "bench/hidden_faults.py", "bench/optimizer_faults.py",
    "evals/optimizer_observability.py",
}


def _hex(value: Any, length: int) -> bool:
    try:
        return isinstance(value, str) and len(value) == length and int(value, 16) >= 0
    except ValueError:
        return False


def sha256_lf(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _safe_relative(value: Any) -> bool:
    return (isinstance(value, str) and bool(value) and not Path(value).is_absolute()
            and "\\" not in value and ".." not in Path(value).parts)


def _arm_contract(arm: str, faults: tuple[str, ...], corrections: tuple[str, ...]) -> bool:
    if arm == "NONE":
        return faults == () and corrections == ()
    if arm in {"F1", "R1"}:
        return faults == (arm,) and corrections == ()
    if arm == "COMPOUND":
        return faults == ("F1", "R1") and corrections == ()
    if arm == "FIRST":
        return (bool(faults) and len(corrections) == 1
                and FAULTS.CORRECTIONS[corrections[0]] in faults)
    return arm == "BOTH" and faults == ("F1", "R1") and set(corrections) == set(FAULTS.CORRECTIONS)


def validate_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    data = dict(value)
    if set(data) != REQUIRED:
        raise ValueError(f"manifest 필드가 고정 계약과 다르다: {sorted(set(data) ^ REQUIRED)}")
    if data["schema"] != SCHEMA or data["mode"] not in MODES or data["arm"] not in ARMS:
        raise ValueError("manifest schema, mode 또는 arm이 잘못됐다")
    faults = FAULTS.normalize_faults(data["original_faults"])
    corrections = FAULTS.normalize_corrections(data["applied_corrections"])
    if list(faults) != data["original_faults"] or list(corrections) != data["applied_corrections"]:
        raise ValueError("fault/correction 배열의 정렬 또는 형식이 고정 계약과 다르다")
    if not _arm_contract(data["arm"], faults, corrections):
        raise ValueError("arm과 original_faults/applied_corrections가 일치하지 않는다")
    if type(data["seed"]) is not int or data["seed"] < 0:
        raise ValueError("seed는 0 이상의 정수여야 한다")
    if not isinstance(data["run_name"], str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_]{0,79}", data["run_name"]):
        raise ValueError("run_name은 영문·숫자·밑줄만 쓴다")
    expected = {**FIXED, "max_iterations": MODES[data["mode"]]}
    if data["task"] != TASK or any(data.get(key) != item for key, item in expected.items()):
        raise ValueError("task, 학습 예산 또는 optimizer 설정이 고정 계약과 다르다")
    if not _hex(data["source_commit_sha"], 40) or not _hex(data["protocol_sha256_lf"], 64):
        raise ValueError("source commit 또는 protocol SHA 형식이 잘못됐다")
    for key in ("source_sha256_lf", "runtime_source_sha256_lf"):
        hashes = data[key]
        if not isinstance(hashes, dict) or not hashes or any(
                not _safe_relative(path) or not _hex(digest, 64) for path, digest in hashes.items()):
            raise ValueError(f"{key}가 비었거나 형식이 잘못됐다")
    if not _safe_relative(data["protocol_path"]):
        raise ValueError("protocol_path는 저장소 상대경로여야 한다")
    for key in ("records_path", "readback_path"):
        if not isinstance(data[key], str) or not Path(data[key]).is_absolute():
            raise ValueError(f"{key}는 private 절대경로여야 한다")
    checkpoint = data["source_checkpoint_sha256"]
    needs_checkpoint = data["mode"] == "inspection" or data["arm"] in {"FIRST", "BOTH"}
    if (needs_checkpoint and not _hex(checkpoint, 64)) or (not needs_checkpoint and checkpoint is not None):
        raise ValueError("source checkpoint SHA provenance가 arm/mode 계약과 다르다")
    if not MINIMUM_BOUND_SOURCES <= set(data["source_sha256_lf"]):
        raise ValueError("필수 복합 학습 source binding이 없다")
    return data


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", "-C", str(ROOT), *args], check=True, capture_output=True,
                          text=True, encoding="utf-8", errors="strict")


def verify_source_binding(manifest: Mapping[str, Any], runtime_root: Path | None = None) -> None:
    data = validate_manifest(manifest)
    commit = data["source_commit_sha"]
    _git("cat-file", "-e", f"{commit}^{{commit}}")
    if subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", commit, "HEAD"]).returncode:
        raise ValueError("source commit이 현재 HEAD의 조상이 아니다")
    for relative, expected in {data["protocol_path"]: data["protocol_sha256_lf"], **data["source_sha256_lf"]}.items():
        path = ROOT / relative
        if not path.is_file() or sha256_lf(path) != expected:
            raise ValueError(f"현재 source hash가 binding과 다르다: {relative}")
        blob = _git("show", f"{commit}:{relative}").stdout.encode().replace(b"\r\n", b"\n")
        if hashlib.sha256(blob).hexdigest() != expected:
            raise ValueError(f"source commit blob이 binding과 다르다: {relative}")
    base = (runtime_root or Path.cwd()).resolve()
    for relative, expected in data["runtime_source_sha256_lf"].items():
        path = base / relative
        if not path.is_file() or sha256_lf(path) != expected:
            raise ValueError(f"Isaac Lab runtime source가 binding과 다르다: {relative}")


def observer_manifest(data: Mapping[str, Any]) -> dict[str, Any]:
    expected = data["configured_learning_epochs"] * data["configured_mini_batches"]
    return OBSERVE.validate_manifest({
        "schema": OBSERVE.MANIFEST_SCHEMA,
        "configured_learning_epochs": data["configured_learning_epochs"],
        "configured_mini_batches": data["configured_mini_batches"],
        "expected_optimizer_steps_per_update": expected,
        "expected_minibatches_per_update": expected,
        "runtime_updates_to_observe": data["runtime_updates_to_observe"],
        "inspection_primitives": list(OBSERVE.PRIMITIVES),
        **{key: data[key] for key in OBSERVE.REQUIRED_BINDING},
    })


def cpu_check() -> dict[str, Any]:
    """Isaac 앱 없이 상태 의미, 순서 독립성, no-op과 결함 보존을 검증한다."""
    def body(*_args: Any) -> str:
        return "body"

    baseline = body.__code__
    rows = []
    for faults in ((), ("F1",), ("R1",), ("F1", "R1")):
        options = [()] + [(item,) for item in FAULTS.CORRECTIONS]
        options += list(permutations(FAULTS.CORRECTIONS, 2))
        for corrections in options:
            body.__code__ = baseline
            ppo = SimpleNamespace(num_learning_epochs=5, num_mini_batches=4,
                                  schedule="adaptive", desired_kl=0.01)
            reward = FAULTS.configure_reward(body, faults, corrections)
            optimizer = FAULTS.configure_ppo(ppo, faults, corrections)
            remaining = list(FAULTS.remaining_faults(faults, corrections))
            if reward["fault_present_after"] != ("F1" in remaining):
                raise AssertionError("F1 remaining 상태가 다르다")
            if optimizer["fault_present_after"] != ("R1" in remaining):
                raise AssertionError("R1 remaining 상태가 다르다")
            rows.append({"original_faults": list(faults), "applied_corrections": list(corrections),
                         "remaining_faults": remaining})
    body.__code__ = baseline
    both = [row for row in rows if row["original_faults"] == ["F1", "R1"]
            and len(row["applied_corrections"]) == 2]
    if len(both) != 2 or any(row["remaining_faults"] for row in both):
        raise AssertionError("두 교정 순서 독립성이 성립하지 않는다")
    source_paths = sorted(MINIMUM_BOUND_SOURCES)
    snapshots = {path: sha256_lf(ROOT / path) for path in source_paths if (ROOT / path).is_file()}
    return {"schema": "residual_training_cpu_check_v1", "status": "pass", "cases": len(rows),
            "order_independent_both": True, "none_noop": True, "other_fault_preserved": True,
            "source_sha256_lf": snapshots}


def execute(manifest_path: Path) -> None:
    data = validate_manifest(json.loads(manifest_path.read_text(encoding="utf-8")))
    verify_source_binding(data)
    records, readback = Path(data["records_path"]), Path(data["readback_path"])
    private_root = manifest_path.resolve().parent
    if private_root not in records.resolve().parents or private_root not in readback.resolve().parents:
        raise ValueError("records/readback은 private manifest 폴더 아래에 있어야 한다")
    if records.exists() or readback.exists():
        raise FileExistsError("records/readback은 덮어쓰지 않는다")
    records.parent.mkdir(parents=True, exist_ok=True)
    readback.parent.mkdir(parents=True, exist_ok=True)
    upstream = Path.cwd() / "scripts/reinforcement_learning/rsl_rl/train.py"
    if not upstream.is_file():
        raise FileNotFoundError("Isaac Lab 상류 train.py가 없다")
    import gymnasium
    original_make, original_path, original_argv = gymnasium.make, sys.path[:], sys.argv[:]
    state: dict[str, Any] = {"installed": False, "observers": []}

    def sink(row: dict[str, Any]) -> None:
        with records.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")

    def write_receipt() -> None:
        if readback.exists() or not all(key in state for key in ("reward", "optimizer", "manager")):
            return
        row = FAULTS.receipt(data["original_faults"], data["applied_corrections"],
                             state["reward"], state["optimizer"], state["manager"])
        row["observer_manifest"] = observer_manifest(data)
        with readback.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(row, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")

    def make(*args: Any, **kwargs: Any) -> Any:
        if not state["installed"]:
            state["installed"] = True
            rewards = importlib.import_module("isaaclab.envs.mdp.rewards")
            target = rewards.track_lin_vel_xy_exp
            state["reward_target"] = target
            state["reward_original_code"] = target.__code__
            state["reward"] = FAULTS.configure_reward(
                target, data["original_faults"], data["applied_corrections"])
            ppo_class = importlib.import_module("rsl_rl.algorithms.ppo").PPO
            original_init = ppo_class.__init__

            def observed_init(self: Any, *init_args: Any, **init_kwargs: Any) -> None:
                original_init(self, *init_args, **init_kwargs)
                state["optimizer"] = FAULTS.configure_ppo(
                    self, data["original_faults"], data["applied_corrections"])
                observer = OBSERVE.RuntimeObserver(self, observer_manifest(data), sink).install()
                state["observers"].append(observer)
                write_receipt()

            ppo_class.__init__ = observed_init
            state["ppo_class"], state["original_init"] = ppo_class, original_init
        env = original_make(*args, **kwargs)
        if "manager" not in state:
            state["manager"] = FAULTS.reward_manager_binding(env, state["reward_target"])
            write_receipt()
        return env

    argv = [str(upstream), "--task", data["task"], "--seed", str(data["seed"]),
            "--num_envs", str(data["num_envs"]), "--max_iterations", str(data["max_iterations"]),
            "--headless", "--run_name", data["run_name"]]
    try:
        gymnasium.make = make
        sys.path.insert(0, str(upstream.parent))
        sys.argv = argv
        runpy.run_path(str(upstream), run_name="__main__")
    finally:
        gymnasium.make = original_make
        for observer in state["observers"]:
            observer.remove()
        if "ppo_class" in state:
            state["ppo_class"].__init__ = state["original_init"]
        if "reward_target" in state:
            state["reward_target"].__code__ = state["reward_original_code"]
        sys.path[:], sys.argv = original_path, original_argv


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="F1+R1 복합 결함과 두 등록 교정 학습")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--manifest", type=Path)
    group.add_argument("--cpu-check", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.cpu_check:
            if args.output is None:
                parser.error("--cpu-check에는 --output이 필요하다")
            with args.output.resolve().open("x", encoding="utf-8", newline="\n") as stream:
                json.dump(cpu_check(), stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write("\n")
        else:
            if args.output is not None:
                parser.error("--manifest에는 --output을 쓰지 않는다")
            execute(args.manifest.resolve())
    except (ValueError, FileNotFoundError, FileExistsError, json.JSONDecodeError,
            subprocess.CalledProcessError) as error:
        print(f"거부: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
