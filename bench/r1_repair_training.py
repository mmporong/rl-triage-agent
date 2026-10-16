"""R1 종단 비교용 새 학습·3 update 검사 진입점.

private manifest 하나만 받아 Isaac Lab 상류 train.py를 실행한다. PPO 생성 직후
optimizer_faults.apply로 R1 의미를 적용하며 CORRECTED는 같은 적용 뒤 등록값 5를 복원한다.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import runpy
import subprocess
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
for _directory in (ROOT / "bench", ROOT / "evals"):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

import optimizer_faults as FAULTS  # noqa: E402
import optimizer_observability as OBSERVE  # noqa: E402

SCHEMA = "r1_repair_training_manifest_v1"
TASK = "Isaac-Velocity-Flat-Unitree-Go2-v0"
ARMS = ("NONE", "R1", "CORRECTED")
MODES = {"train": 300, "inspection": 3}
FIXED = {"num_envs": 4096, "configured_learning_epochs": 5,
         "configured_mini_batches": 4, "runtime_updates_to_observe": 3}
REQUIRED = {
    "schema", "mode", "arm", "seed", "run_name", "task", "num_envs", "max_iterations",
    "configured_learning_epochs", "configured_mini_batches", "runtime_updates_to_observe",
    "source_commit_sha", "protocol_path", "protocol_sha256_lf", "source_sha256_lf",
    "runtime_source_sha256_lf", "records_path", "readback_path", "source_checkpoint_sha256",
}
MINIMUM_BOUND_SOURCES = {
    "bench/r1_repair_training.py", "bench/optimizer_faults.py", "evals/optimizer_observability.py",
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


def validate_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    data = dict(value)
    if set(data) != REQUIRED:
        raise ValueError(f"manifest 필드가 고정 계약과 다르다: {sorted(set(data) ^ REQUIRED)}")
    if data["schema"] != SCHEMA or data["mode"] not in MODES or data["arm"] not in ARMS:
        raise ValueError("manifest schema, mode 또는 arm이 잘못됐다")
    if data["mode"] == "inspection" and data["arm"] not in {"NONE", "R1"}:
        raise ValueError("3 update 검사는 NONE 또는 R1 arm만 허용한다")
    if type(data["seed"]) is not int or data["seed"] < 0:
        raise ValueError("seed는 0 이상의 정수여야 한다")
    if not isinstance(data["run_name"], str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_]{0,79}", data["run_name"]):
        raise ValueError("run_name은 영문·숫자·밑줄만 쓴다")
    if not isinstance(data["task"], str) or not data["task"]:
        raise ValueError("task가 비었다")
    if data["task"] != TASK:
        raise ValueError("task가 고정 allowlist와 다르다")
    expected = {**FIXED, "max_iterations": MODES[data["mode"]]}
    if any(data.get(key) != item for key, item in expected.items()):
        raise ValueError("학습 예산 또는 optimizer 설정이 고정 계약과 다르다")
    if not _hex(data["source_commit_sha"], 40) or not _hex(data["protocol_sha256_lf"], 64):
        raise ValueError("source commit 또는 protocol SHA 형식이 잘못됐다")
    for key in ("source_sha256_lf", "runtime_source_sha256_lf"):
        hashes = data[key]
        if not isinstance(hashes, dict) or not hashes or any(
                not _safe_relative(path) or not _hex(digest, 64)
                for path, digest in hashes.items()):
            raise ValueError(f"{key}가 비었거나 형식이 잘못됐다")
    if not _safe_relative(data["protocol_path"]):
        raise ValueError("protocol_path는 저장소 상대경로여야 한다")
    for key in ("records_path", "readback_path"):
        if not isinstance(data[key], str) or not Path(data[key]).is_absolute():
            raise ValueError(f"{key}는 private 절대경로여야 한다")
    source_checkpoint = data["source_checkpoint_sha256"]
    if source_checkpoint is not None and not _hex(source_checkpoint, 64):
        raise ValueError("source checkpoint SHA 형식이 잘못됐다")
    if data["mode"] == "train":
        if data["arm"] == "CORRECTED" and source_checkpoint is None:
            raise ValueError("CORRECTED 학습은 source checkpoint SHA provenance가 필요하다")
        if data["arm"] in {"NONE", "R1"} and source_checkpoint is not None:
            raise ValueError("NONE/R1 학습은 source checkpoint를 받지 않는다")
    elif source_checkpoint is None:
        raise ValueError("inspection은 source checkpoint SHA 계보가 필요하다")
    if not MINIMUM_BOUND_SOURCES <= set(data["source_sha256_lf"]):
        raise ValueError("필수 wrapper/observer/optimizer source binding이 없다")
    return data


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", "-C", str(ROOT), *args], check=True, capture_output=True,
                          text=True, encoding="utf-8", errors="strict")


def verify_repository_binding(manifest: Mapping[str, Any]) -> None:
    data = validate_manifest(manifest)
    commit = data["source_commit_sha"]
    _git("cat-file", "-e", f"{commit}^{{commit}}")
    if subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", commit, "HEAD"]).returncode != 0:
        raise ValueError("source commit이 현재 HEAD의 조상이 아니다")
    commitments = {data["protocol_path"]: data["protocol_sha256_lf"], **data["source_sha256_lf"]}
    for relative, expected in commitments.items():
        path = ROOT / relative
        if not path.is_file() or sha256_lf(path) != expected:
            raise ValueError(f"현재 source hash가 binding과 다르다: {relative}")
        blob = _git("show", f"{commit}:{relative}").stdout.encode("utf-8").replace(b"\r\n", b"\n")
        if hashlib.sha256(blob).hexdigest() != expected:
            raise ValueError(f"source commit blob이 binding과 다르다: {relative}")


def verify_runtime_binding(manifest: Mapping[str, Any], runtime_root: Path) -> None:
    data = validate_manifest(manifest)
    for relative, expected in data["runtime_source_sha256_lf"].items():
        path = runtime_root / relative
        if not path.is_file() or sha256_lf(path) != expected:
            raise ValueError(f"Isaac Lab runtime source가 binding과 다르다: {relative}")


def verify_source_binding(manifest: Mapping[str, Any]) -> None:
    verify_repository_binding(manifest)
    verify_runtime_binding(manifest, Path.cwd())


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


def configure_ppo(ppo: Any, arm: str, configured_epochs: int = 5, mini_batches: int = 4) -> dict[str, Any]:
    """결함 적용과 교정 복원을 수행하고 적용 전후 값을 반환한다."""
    if arm not in ARMS:
        raise ValueError("arm은 NONE/R1/CORRECTED 중 하나여야 한다")
    if ppo.num_learning_epochs != configured_epochs or ppo.num_mini_batches != mini_batches:
        raise ValueError("런타임 PPO 설정이 등록값과 다르다")
    before = {"learning_epochs": ppo.num_learning_epochs, "mini_batches": ppo.num_mini_batches}
    applied = FAULTS.apply("NONE" if arm == "NONE" else "R1", ppo)
    after_fault = {"learning_epochs": ppo.num_learning_epochs, "mini_batches": ppo.num_mini_batches}
    restored = None
    if arm == "CORRECTED":
        ppo.num_learning_epochs = configured_epochs
        restored = {"registry_value": configured_epochs, "runtime_value": ppo.num_learning_epochs}
    after = {"learning_epochs": ppo.num_learning_epochs, "mini_batches": ppo.num_mini_batches}
    effective = 1 if arm == "R1" else configured_epochs
    if after != {"learning_epochs": effective, "mini_batches": mini_batches}:
        raise ValueError("optimizer 결함/교정 readback이 기대값과 다르다")
    return {"arm": arm, "before": before, "fault_application": applied,
            "after_fault": after_fault, "restoration": restored, "after": after}


def inspect_checkpoint(final_path: Path, first_path: Path, expected_iteration: int) -> dict[str, Any]:
    """체크포인트 두 개를 CPU로 읽어 iteration과 state dictionary를 검증한다."""
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    import torch

    def load(path: Path, expected: int) -> dict[str, Any]:
        if not path.is_file():
            raise FileNotFoundError(f"체크포인트가 없다: {path.name}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or payload.get("iter") != expected:
            raise ValueError(f"{path.name} iteration이 {expected}와 다르다")
        for key in ("model_state_dict", "optimizer_state_dict"):
            if not isinstance(payload.get(key), dict) or not payload[key]:
                raise ValueError(f"{path.name}에 {key}가 없다")
        return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "iteration": payload["iter"], "model_state_keys": len(payload["model_state_dict"]),
                "optimizer_state_keys": len(payload["optimizer_state_dict"])}

    return {"device": "cpu", "first": load(first_path, 0), "final": load(final_path, expected_iteration)}


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
    state: dict[str, Any] = {"installed": False, "observers": [], "readbacks": []}

    def sink(row: dict[str, Any]) -> None:
        with records.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")

    def make(*args, **kwargs):
        if not state["installed"]:
            state["installed"] = True
            ppo_class = importlib.import_module(FAULTS.PPO_MODULE).PPO
            original_init = ppo_class.__init__

            def observed_init(self, *init_args, **init_kwargs):
                original_init(self, *init_args, **init_kwargs)
                row = configure_ppo(self, data["arm"])
                state["readbacks"].append(row)
                observer = OBSERVE.RuntimeObserver(self, observer_manifest(data), sink).install()
                state["observers"].append(observer)
                with readback.open("x", encoding="utf-8", newline="\n") as stream:
                    json.dump(row, stream, ensure_ascii=False, indent=2, allow_nan=False)
                    stream.write("\n")

            ppo_class.__init__ = observed_init
            state["ppo_class"], state["original_init"] = ppo_class, original_init
        return original_make(*args, **kwargs)

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
        sys.path[:], sys.argv = original_path, original_argv


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--inspect-checkpoint", type=Path)
    parser.add_argument("--first-checkpoint", type=Path)
    parser.add_argument("--expected-final-iteration", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.manifest is not None and all(item is None for item in (
                args.inspect_checkpoint, args.first_checkpoint, args.expected_final_iteration, args.output)):
            execute(args.manifest.resolve())
        elif (args.manifest is None and args.inspect_checkpoint is not None and args.first_checkpoint is not None
              and args.expected_final_iteration is not None and args.output is not None):
            receipt = inspect_checkpoint(args.inspect_checkpoint.resolve(), args.first_checkpoint.resolve(),
                                         args.expected_final_iteration)
            with args.output.resolve().open("x", encoding="utf-8", newline="\n") as stream:
                json.dump(receipt, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write("\n")
        else:
            parser.error("--manifest 또는 checkpoint 검사 인자 전부를 제공해야 한다")
    except (ValueError, FileNotFoundError, FileExistsError, json.JSONDecodeError,
            subprocess.CalledProcessError) as error:
        print(f"거부: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
