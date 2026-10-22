"""PPO ratio 계산 오류 종단 실행용 새 학습·3 update 검사 진입점.

private manifest 하나만 받아 Isaac Lab 상류 train.py를 실행한다. PPO 생성 직후 등록 변이
(ppo_ratio_mutation_v1)를 arm에 맞춰 설치하고, 첫 3 update에 surrogate gradient 관측기를 붙인다.
CORRECTED는 같은 변이를 설치한 뒤 원래 update로 되돌린 상태로 학습한다(복원 readback 포함).
checkpoint CPU 검사는 R1 진입점(bench/r1_repair_training.py)의 함수를 그대로 쓴다.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
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

import optimizer_observability as OBSERVE  # noqa: E402
import ppo_gradient_invariant_v2 as GRADIENT  # noqa: E402
import ppo_ratio_mutation_v1 as MUTATION  # noqa: E402
import r1_repair_training as R1  # noqa: E402

SCHEMA = "ppo_ratio_repair_training_manifest_v1"
TASK = R1.TASK
ARMS = ("NONE", "INVERSE_RATIO", "CORRECTED")
# 학습 arm → 등록 변이 arm. CORRECTED는 변이를 설치했다가 원래 계산으로 되돌린다.
MUTATION_ARMS = {"NONE": "NONE", "INVERSE_RATIO": "INVERSE_RATIO", "CORRECTED": "RESTORED"}
MODES = R1.MODES
FIXED = R1.FIXED
REQUIRED = R1.REQUIRED
INVARIANT = {"schema": GRADIENT.GRADIENT_SCHEMA, "atol": GRADIENT.ATOL, "rtol": GRADIENT.RTOL,
             "scale": GRADIENT.SCALE, "clip_param": GRADIENT.CLIP_PARAM}
PRIMITIVES = ["optimizer_step_count", "minibatch_count", GRADIENT.PRIMITIVE]
MINIMUM_BOUND_SOURCES = {
    "bench/ppo_ratio_repair_training.py", "bench/ppo_ratio_mutation_v1.py", "bench/r1_repair_training.py",
    "evals/ppo_gradient_invariant_v1.py", "evals/ppo_gradient_invariant_v2.py", "evals/optimizer_observability.py",
}


def validate_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    data = dict(value)
    if set(data) != REQUIRED:
        raise ValueError(f"manifest 필드가 고정 계약과 다르다: {sorted(set(data) ^ REQUIRED)}")
    if data["schema"] != SCHEMA or data["mode"] not in MODES or data["arm"] not in ARMS:
        raise ValueError("manifest schema, mode 또는 arm이 잘못됐다")
    if data["mode"] == "inspection" and data["arm"] not in {"NONE", "INVERSE_RATIO"}:
        raise ValueError("3 update 검사는 NONE 또는 INVERSE_RATIO arm만 허용한다")
    if type(data["seed"]) is not int or data["seed"] < 0:
        raise ValueError("seed는 0 이상의 정수여야 한다")
    if not isinstance(data["run_name"], str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_]{0,79}", data["run_name"]):
        raise ValueError("run_name은 영문·숫자·밑줄만 쓴다")
    if data["task"] != TASK:
        raise ValueError("task가 고정 allowlist와 다르다")
    expected = {**FIXED, "max_iterations": MODES[data["mode"]]}
    if any(data.get(key) != item for key, item in expected.items()):
        raise ValueError("학습 예산 또는 optimizer 설정이 고정 계약과 다르다")
    if not R1._hex(data["source_commit_sha"], 40) or not R1._hex(data["protocol_sha256_lf"], 64):
        raise ValueError("source commit 또는 protocol SHA 형식이 잘못됐다")
    for key in ("source_sha256_lf", "runtime_source_sha256_lf"):
        hashes = data[key]
        if not isinstance(hashes, dict) or not hashes or any(
                not R1._safe_relative(path) or not R1._hex(digest, 64) for path, digest in hashes.items()):
            raise ValueError(f"{key}가 비었거나 형식이 잘못됐다")
    if not R1._safe_relative(data["protocol_path"]):
        raise ValueError("protocol_path는 저장소 상대경로여야 한다")
    for key in ("records_path", "readback_path"):
        if not isinstance(data[key], str) or not Path(data[key]).is_absolute():
            raise ValueError(f"{key}는 private 절대경로여야 한다")
    source_checkpoint = data["source_checkpoint_sha256"]
    if source_checkpoint is not None and not R1._hex(source_checkpoint, 64):
        raise ValueError("source checkpoint SHA 형식이 잘못됐다")
    if data["mode"] == "train":
        if data["arm"] == "CORRECTED" and source_checkpoint is None:
            raise ValueError("CORRECTED 학습은 source checkpoint SHA provenance가 필요하다")
        if data["arm"] in {"NONE", "INVERSE_RATIO"} and source_checkpoint is not None:
            raise ValueError("NONE/INVERSE_RATIO 학습은 source checkpoint를 받지 않는다")
    elif source_checkpoint is None:
        raise ValueError("inspection은 source checkpoint SHA 계보가 필요하다")
    if not set(data["source_sha256_lf"]) >= MINIMUM_BOUND_SOURCES:
        raise ValueError("필수 wrapper/observer/mutation source binding이 없다")
    return data


def verify_repository_binding(manifest: Mapping[str, Any]) -> None:
    data = validate_manifest(manifest)
    commit = data["source_commit_sha"]
    R1._git("cat-file", "-e", f"{commit}^{{commit}}")
    if subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", commit, "HEAD"]).returncode != 0:
        raise ValueError("source commit이 현재 HEAD의 조상이 아니다")
    commitments = {data["protocol_path"]: data["protocol_sha256_lf"], **data["source_sha256_lf"]}
    for relative, expected in commitments.items():
        path = ROOT / relative
        if not path.is_file() or R1.sha256_lf(path) != expected:
            raise ValueError(f"현재 source hash가 binding과 다르다: {relative}")
        blob = R1._git("show", f"{commit}:{relative}").stdout.encode("utf-8").replace(b"\r\n", b"\n")
        if hashlib.sha256(blob).hexdigest() != expected:
            raise ValueError(f"source commit blob이 binding과 다르다: {relative}")


def verify_runtime_binding(manifest: Mapping[str, Any], runtime_root: Path) -> None:
    data = validate_manifest(manifest)
    for relative, expected in data["runtime_source_sha256_lf"].items():
        path = runtime_root / relative
        if not path.is_file() or R1.sha256_lf(path) != expected:
            raise ValueError(f"Isaac Lab runtime source가 binding과 다르다: {relative}")


def observer_manifest(data: Mapping[str, Any]) -> dict[str, Any]:
    expected = data["configured_learning_epochs"] * data["configured_mini_batches"]
    return GRADIENT.validate_manifest({
        "schema": OBSERVE.MANIFEST_SCHEMA,
        "configured_learning_epochs": data["configured_learning_epochs"],
        "configured_mini_batches": data["configured_mini_batches"],
        "expected_optimizer_steps_per_update": expected,
        "expected_minibatches_per_update": expected,
        "runtime_updates_to_observe": data["runtime_updates_to_observe"],
        "gradient_invariant": dict(INVARIANT), "inspection_primitives": list(PRIMITIVES),
        **{key: data[key] for key in OBSERVE.REQUIRED_BINDING},
    })


def configure_ppo(ppo: Any, arm: str, configured_epochs: int = 5, mini_batches: int = 4):
    """등록 변이를 arm에 맞춰 설치하고 (복원 함수, readback)을 돌려준다."""
    if arm not in ARMS:
        raise ValueError("arm은 NONE/INVERSE_RATIO/CORRECTED 중 하나여야 한다")
    configuration = {"learning_epochs": ppo.num_learning_epochs, "mini_batches": ppo.num_mini_batches,
                     "clip_param": ppo.clip_param}
    if configuration != {"learning_epochs": configured_epochs, "mini_batches": mini_batches,
                         "clip_param": GRADIENT.CLIP_PARAM}:
        raise ValueError("런타임 PPO 설정이 등록값과 다르다")
    restore, mutation = MUTATION.install(ppo, MUTATION_ARMS[arm])
    return restore, {"arm": arm, "configuration": configuration, "mutation": mutation}


def validate_readback(value: Any, arm: str) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("arm") != arm:
        raise ValueError("ratio readback arm이 다르다")
    if value.get("configuration") != {"learning_epochs": 5, "mini_batches": 4, "clip_param": GRADIENT.CLIP_PARAM}:
        raise ValueError("ratio readback의 PPO 구성이 등록값과 다르다")
    mutation = value.get("mutation")
    if not isinstance(mutation, dict) or mutation.get("arm") != MUTATION_ARMS[arm]:
        raise ValueError("ratio 변이 arm readback이 다르다")
    if (mutation.get("mutation") != MUTATION.MUTATION or mutation.get("changed_ratio_assignments") != 1
            or mutation.get("active_mutation") is not (arm == "INVERSE_RATIO")
            or mutation.get("restored_original") is not (arm == "CORRECTED")):
        raise ValueError("ratio 변이 설치·복원 readback이 등록 arm과 다르다")
    key = "mutated_ast_sha256" if arm == "INVERSE_RATIO" else "original_ast_sha256"
    if mutation.get("installed_ast_sha256") != mutation.get(key):
        raise ValueError("설치된 update AST가 arm과 다르다")
    return value


def execute(manifest_path: Path) -> None:
    data = validate_manifest(json.loads(manifest_path.read_text(encoding="utf-8")))
    verify_repository_binding(data)
    verify_runtime_binding(data, Path.cwd())
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
    state: dict[str, Any] = {"installed": False, "observers": [], "restores": []}

    def sink(row: dict[str, Any]) -> None:
        with records.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")

    def make(*args, **kwargs):
        if not state["installed"]:
            state["installed"] = True
            ppo_class = importlib.import_module("rsl_rl.algorithms.ppo").PPO
            original_init = ppo_class.__init__

            def observed_init(self, *init_args, **init_kwargs):
                original_init(self, *init_args, **init_kwargs)
                if state["restores"]:
                    raise ValueError("PPO 인스턴스는 실행당 하나여야 한다")
                restore, row = configure_ppo(self, data["arm"])
                state["restores"].append(restore)
                observer = GRADIENT.GradientObserver(self, observer_manifest(data), sink).install()
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
        for restore in state["restores"]:
            restore()
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
            receipt = R1.inspect_checkpoint(args.inspect_checkpoint.resolve(), args.first_checkpoint.resolve(),
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
