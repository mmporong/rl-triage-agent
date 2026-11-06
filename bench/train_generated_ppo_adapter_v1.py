# pyright: reportMissingImports=false
"""Train a Go2 policy from scratch with a registered PPO kwargs adapter."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import traceback
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _directory in (ROOT / "evals", ROOT / "bench"):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

import ppo_patch_contract_v1 as CONTRACT

TRAINING_SCHEMA = "generated_ppo_adapter_training_v1"
SMOKE_SCHEMA = "generated_ppo_adapter_training_smoke_v1"
PROTOCOL_SCHEMA = "generated_ppo_adapter_training_protocol_v1"
RESULT_SCHEMA = "generated_ppo_adapter_training_result_v1"
TASK = "Isaac-Velocity-Flat-Unitree-Go2-v0"
ARMS = ("healthy", "fault", "patched")
TRAINING_BY_SCHEMA = {
    TRAINING_SCHEMA: {"num_envs": 4096, "updates": 300},
    SMOKE_SCHEMA: {"num_envs": 4096, "updates": 1},
}
SPEC_FIELDS = {
    "schema",
    "arm",
    "seed",
    "task",
    "adapter_source",
    "adapter_source_sha256",
    "source_commit",
    "source_sha256_lf",
    "protocol",
    "training",
}
REQUIRED_SOURCES = {
    "bench/train_generated_ppo_adapter_v1.py",
    "tests/test_train_generated_ppo_adapter_v1.py",
    "evals/ppo_patch_contract_v1.py",
    "tests/test_ppo_patch_contract_v1.py",
    "evals/lineage_lease_context_v2.py",
}
ADAPTER_PROBE_CONFIG = {
    "num_learning_epochs": 5,
    "num_mini_batches": 4,
    "learning_rate": 0.001,
    "schedule": "fixed",
    "gamma": 0.99,
}
_HEX40 = re.compile(r"[0-9a-f]{40}")
_HEX64 = re.compile(r"[0-9a-f]{64}")


class TrainingContractError(ValueError):
    """Raised before the SDK starts when a registered input is invalid."""


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise TrainingContractError(reason)


def _sha256_lf(path: Path) -> str:
    raw = Path(path).read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(raw).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _stable_value(value: object) -> object:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        _require(math.isfinite(value), "설정에 유한하지 않은 수치가 있다")
        return value
    if isinstance(value, dict):
        _require(all(isinstance(key, str) for key in value), "설정 dictionary key가 문자열이 아니다")
        return {key: _stable_value(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_stable_value(item) for item in value]
    if isinstance(value, slice):
        return {
            "slice": [
                _stable_value(value.start),
                _stable_value(value.stop),
                _stable_value(value.step),
            ]
        }
    if isinstance(value, Enum):
        return {
            "enum": f"{type(value).__module__}.{type(value).__qualname__}",
            "name": value.name,
        }
    if callable(value):
        return {
            "callable": f"{getattr(value, '__module__', type(value).__module__)}."
            f"{getattr(value, '__qualname__', type(value).__qualname__)}"
        }
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        converted = to_dict()
        if converted is value:
            raise TypeError("to_dict()가 원래 설정 객체를 반환했다")
        return {
            "config_type": f"{type(value).__module__}.{type(value).__qualname__}",
            "value": _stable_value(converted),
        }
    raise TypeError(
        f"설정 정규화를 지원하지 않는 형식이다: {type(value).__module__}.{type(value).__qualname__}"
    )


def _json_sha256(value: object) -> str:
    raw = json.dumps(
        _stable_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _relative(value: object) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and "\\" not in value
        and not Path(value).is_absolute()
        and ".." not in Path(value).parts
    )


def _hash_map(value: object) -> dict[str, str]:
    if not isinstance(value, dict) or not value:
        raise TrainingContractError("source_sha256_lf가 비었다")
    result: dict[str, str] = {}
    for path, digest in value.items():
        if not (_relative(path) and isinstance(digest, str) and _HEX64.fullmatch(digest)):
            raise TrainingContractError("source_sha256_lf 형식이 잘못됐다")
        assert isinstance(path, str)
        result[path] = digest
    return result


def _apply_adapter_checked(source: str, config: dict[str, Any], arm: str) -> dict[str, Any]:
    before = copy.deepcopy(config)
    after = CONTRACT.apply_adapter(source, config)
    _require(set(after) == set(before), "adapter가 algorithm key 집합을 바꿨다")
    _require(
        {key: value for key, value in after.items() if key != "num_learning_epochs"}
        == {key: value for key, value in before.items() if key != "num_learning_epochs"},
        "adapter가 epoch 외 algorithm 값을 바꿨다",
    )
    expected_epochs = 1 if arm == "fault" else before.get("num_learning_epochs")
    _require(
        isinstance(expected_epochs, int)
        and not isinstance(expected_epochs, bool)
        and after.get("num_learning_epochs") == expected_epochs,
        "adapter arm의 num_learning_epochs 결과가 등록 의미와 다르다",
    )
    return after


def validate_spec_shape(value: Mapping[str, Any]) -> dict[str, Any]:
    _require(isinstance(value, Mapping), "spec은 JSON 객체여야 한다")
    data = dict(value)
    _require(set(data) == SPEC_FIELDS, "spec 필드가 고정 계약과 다르다")
    _require(data["schema"] in TRAINING_BY_SCHEMA, "spec schema가 잘못됐다")
    _require(data["arm"] in ARMS, "arm이 잘못됐다")
    _require(type(data["seed"]) is int and 0 <= data["seed"] <= 2_147_483_647, "seed가 잘못됐다")
    _require(data["task"] == TASK, "task가 Go2 flat 등록값과 다르다")
    _require(isinstance(data["adapter_source"], str), "adapter_source가 문자열이 아니다")
    _require(
        isinstance(data["adapter_source_sha256"], str)
        and _HEX64.fullmatch(data["adapter_source_sha256"]) is not None,
        "adapter source SHA 형식이 잘못됐다",
    )
    _require(
        CONTRACT.source_sha256(data["adapter_source"]) == data["adapter_source_sha256"],
        "adapter source SHA가 원문과 다르다",
    )
    CONTRACT.validate_source(data["adapter_source"])
    _apply_adapter_checked(data["adapter_source"], ADAPTER_PROBE_CONFIG, data["arm"])
    _require(
        isinstance(data["source_commit"], str)
        and _HEX40.fullmatch(data["source_commit"]) is not None,
        "source_commit 형식이 잘못됐다",
    )
    data["source_sha256_lf"] = _hash_map(data["source_sha256_lf"])
    _require(REQUIRED_SOURCES <= set(data["source_sha256_lf"]), "필수 source closure가 누락됐다")
    protocol = data["protocol"]
    _require(isinstance(protocol, dict) and set(protocol) == {"relative_path", "sha256_lf"},
             "protocol 필드가 잘못됐다")
    _require(_relative(protocol["relative_path"]), "protocol 경로가 저장소 상대경로가 아니다")
    _require(isinstance(protocol["sha256_lf"], str)
             and _HEX64.fullmatch(protocol["sha256_lf"]) is not None,
             "protocol SHA 형식이 잘못됐다")
    _require(protocol["relative_path"] not in data["source_sha256_lf"],
             "protocol은 source_sha256_lf와 별도 필드로 한 번만 결합해야 한다")
    _require(data["training"] == TRAINING_BY_SCHEMA[data["schema"]],
             "환경 수 또는 update 예산이 schema와 다르다")
    return data


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=check, capture_output=True
    )


def _verify_commit_file(repo: Path, commit: str, relative: str, expected: str) -> None:
    path = repo / relative
    _require(path.is_file() and _sha256_lf(path) == expected,
             f"현재 파일 SHA가 결합과 다르다: {relative}")
    blob = _git(repo, "show", f"{commit}:{relative}").stdout
    normalized = blob.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    _require(hashlib.sha256(normalized).hexdigest() == expected,
             f"커밋 blob SHA가 결합과 다르다: {relative}")


def _protocol_job(protocol: object, spec: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(protocol, dict) or protocol.get("schema") != PROTOCOL_SCHEMA:
        raise TrainingContractError("protocol schema가 잘못됐다")
    jobs = protocol.get("jobs")
    if not isinstance(jobs, list):
        raise TrainingContractError("protocol jobs가 배열이 아니다")
    matches = [
        row for row in jobs
        if isinstance(row, dict)
        and row.get("arm") == spec["arm"]
        and row.get("seed") == spec["seed"]
        and row.get("adapter_source_sha256") == spec["adapter_source_sha256"]
        and row.get("training") == spec["training"]
    ]
    _require(len(matches) == 1, "protocol에서 유일한 adapter 학습 job을 찾지 못했다")
    return dict(matches[0])


def validate_spec(spec: dict[str, Any], *, repo: Path = ROOT) -> dict[str, Any]:
    """Validate adapter semantics and the committed source/protocol binding."""

    data = validate_spec_shape(spec)
    repo = Path(repo).resolve()
    commit = data["source_commit"]
    try:
        _git(repo, "cat-file", "-e", f"{commit}^{{commit}}")
    except subprocess.CalledProcessError as error:
        raise TrainingContractError("source_commit을 찾지 못했다") from error
    _require(
        _git(repo, "merge-base", "--is-ancestor", commit, "HEAD", check=False).returncode == 0,
        "source_commit이 현재 HEAD의 조상이 아니다",
    )
    commitments = {
        data["protocol"]["relative_path"]: data["protocol"]["sha256_lf"],
        **data["source_sha256_lf"],
    }
    for relative, expected in commitments.items():
        _verify_commit_file(repo, commit, relative, expected)
    protocol = json.loads((repo / data["protocol"]["relative_path"]).read_text(encoding="utf-8"))
    _protocol_job(protocol, data)
    return data


def _write_new(path: Path, value: Mapping[str, Any]) -> None:
    raw = json.dumps(
        dict(value), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
    ).encode("utf-8") + b"\n"
    with Path(path).open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def _model_digest(model: Any) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _finite_result(value: object) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, dict):
        return all(_finite_result(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_result(item) for item in value)
    if hasattr(value, "isfinite"):
        try:
            return bool(value.isfinite().all().item())  # type: ignore[union-attr]
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return False
    return True


def launcher_arguments(args: argparse.Namespace) -> dict[str, Any]:
    """Produce one AppLauncher kwargs mapping without duplicate kit arguments."""

    value = vars(args).copy()
    value.update(
        headless=True,
        kit_args="--/app/vulkan=false --/app/window/hideUi=true",
        fast_shutdown=False,
    )
    return value


def train(spec: dict[str, Any], *, out: Path) -> dict[str, Any]:
    """Run the registered from-scratch training after the application starts."""

    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401
    import lineage_lease_context_v2 as LEASE
    import torch
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
    from rsl_rl.runners import OnPolicyRunner

    LEASE.lease_context()
    torch.manual_seed(spec["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(spec["seed"])
    env_cfg = parse_env_cfg(
        spec["task"], device="cuda:0", num_envs=spec["training"]["num_envs"], use_fabric=True
    )
    env_cfg.seed = spec["seed"]
    registered = env_cfg.to_dict()
    task_terms_before = {
        "rewards_sha256": _json_sha256(registered["rewards"]),
        "terminations_sha256": _json_sha256(registered["terminations"]),
    }
    agent = load_cfg_from_registry(spec["task"], "rsl_rl_cfg_entry_point")
    agent.device = "cuda:0"
    agent.seed = spec["seed"]
    agent.max_iterations = spec["training"]["updates"]
    agent.save_interval = spec["training"]["updates"] + 1
    runner_cfg = agent.to_dict()
    algorithm_before = copy.deepcopy(runner_cfg["algorithm"])
    algorithm_after = _apply_adapter_checked(
        spec["adapter_source"], algorithm_before, spec["arm"]
    )
    runner_cfg["algorithm"] = algorithm_after
    env = RslRlVecEnvWrapper(
        gym.make(spec["task"], cfg=env_cfg), clip_actions=agent.clip_actions
    )
    runner = None
    try:
        task_terms_after = env_cfg.to_dict()
        task_terms_after = {
            "rewards_sha256": _json_sha256(task_terms_after["rewards"]),
            "terminations_sha256": _json_sha256(task_terms_after["terminations"]),
        }
        _require(task_terms_after == task_terms_before,
                 "등록 task reward 또는 termination 설정이 바뀌었다")
        runner = OnPolicyRunner(env, runner_cfg, log_dir=str(out / "logs"), device=agent.device)
        policy_before = _model_digest(runner.alg.policy)
        counts = {"updates": 0, "minibatches": 0, "optimizer_steps": 0}
        first_three: list[dict[str, Any]] = []
        original_generator = runner.alg.storage.mini_batch_generator
        original_step = runner.alg.optimizer.step
        original_update = runner.alg.update
        updates_stream = (out / "updates.jsonl").open("xb")

        def measured_generator(*args: Any, **kwargs: Any):
            for batch in original_generator(*args, **kwargs):
                counts["minibatches"] += 1
                yield batch

        def measured_step(*args: Any, **kwargs: Any):
            counts["optimizer_steps"] += 1
            return original_step(*args, **kwargs)

        def measured_update(*args: Any, **kwargs: Any):
            before = dict(counts)
            result = original_update(*args, **kwargs)
            counts["updates"] += 1
            row = {
                "index": counts["updates"],
                "minibatches": counts["minibatches"] - before["minibatches"],
                "optimizer_steps": counts["optimizer_steps"] - before["optimizer_steps"],
                "return_values_finite": _finite_result(result),
                "learning_rate": float(runner.alg.learning_rate),
                "mean_policy_std": float(runner.alg.policy.std.mean().item()),
            }
            if counts["updates"] <= 3:
                first_three.append(dict(row))
            updates_stream.write(
                json.dumps(row, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n"
            )
            updates_stream.flush()
            os.fsync(updates_stream.fileno())
            if counts["updates"] % 25 == 0:
                LEASE.lease_context()
            return result

        runner.alg.storage.mini_batch_generator = measured_generator
        runner.alg.optimizer.step = measured_step
        runner.alg.update = measured_update
        try:
            runner.learn(
                num_learning_iterations=spec["training"]["updates"],
                init_at_random_ep_len=True,
            )
        finally:
            updates_stream.close()
        LEASE.lease_context()
        final = out / "final.pt"
        runner.save(str(final))
        policy_after = _model_digest(runner.alg.policy)
        actual_epochs = int(runner.alg.num_learning_epochs)
        actual_minibatches = int(runner.alg.num_mini_batches)
        expected_steps = spec["training"]["updates"] * actual_epochs * actual_minibatches
        _require(
            counts == {
                "updates": spec["training"]["updates"],
                "minibatches": expected_steps,
                "optimizer_steps": expected_steps,
            },
            "실제 PPO update/minibatch/optimizer step 수가 설정과 다르다",
        )
        _require(policy_before != policy_after, "from-scratch 정책 가중치가 갱신되지 않았다")
        _require(
            all(bool(torch.isfinite(value).all().item()) for value in runner.alg.policy.state_dict().values()),
            "최종 정책에 유한하지 않은 값이 있다",
        )
        result = {
            "schema": RESULT_SCHEMA,
            "status": "complete",
            "pid": os.getpid(),
            "arm": spec["arm"],
            "seed": spec["seed"],
            "task": spec["task"],
            "training": dict(spec["training"]),
            "adapter_source_sha256": spec["adapter_source_sha256"],
            "algorithm_before": algorithm_before,
            "algorithm_after": algorithm_after,
            "actual_ppo": {
                "num_learning_epochs": actual_epochs,
                "num_mini_batches": actual_minibatches,
            },
            "counts": counts,
            "first_three_updates": first_three,
            "task_terms_before": task_terms_before,
            "task_terms_after": task_terms_after,
            "policy_sha256_before": policy_before,
            "policy_sha256_after": policy_after,
            "policy_parameters_finite": True,
            "checkpoint": "final.pt",
            "checkpoint_sha256": _sha256(final),
            "protocol_sha256_lf": spec["protocol"]["sha256_lf"],
            "source_commit": spec["source_commit"],
            "policy_recovery_verified": False,
        }
        _write_new(out / "training-result.json", result)
        return result
    finally:
        env.close()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(value, dict), "spec JSON은 객체여야 한다")
    return value


def _preparse(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--headless", action="store_true", required=True)
    args, _ = parser.parse_known_args(argv)
    _require(args.headless, "--headless가 필요하다")
    _require(not args.out.exists(), "출력 경로가 이미 존재한다")
    _require(args.out.parent.is_dir(), "출력 상위 경로가 없다")
    return args


def main(argv: list[str] | None = None) -> int:
    preparsed = _preparse(argv)
    spec = validate_spec(_read_json(preparsed.spec), repo=ROOT)

    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(argv)
    _require(args.headless is True, "--headless가 필요하다")
    args.out.mkdir(parents=False, exist_ok=False)
    app = None
    try:
        app = AppLauncher(launcher_arguments(args)).app
        _write_new(args.out / "app-startup.json", {"pid": os.getpid()})
        train(spec, out=args.out)
        import carb

        settings = carb.settings.get_settings()
        settings.set_bool("/app/fastShutdown", True)
        _require(settings.get_as_bool("/app/fastShutdown") is True,
                 "학습 증거 봉인 후 종료 설정 readback이 실패했다")
        _write_new(
            args.out / "app-close-entered.json",
            {
                "pid": os.getpid(),
                "fast_shutdown": True,
                "wait_for_replicator": False,
                "training_result_sha256": _sha256(args.out / "training-result.json"),
            },
        )
        app.close(wait_for_replicator=False)
        return 0
    except BaseException as error:  # noqa: BLE001 - SDK 종료가 예외를 숨기기 전에 봉인한다.
        if args.out.is_dir() and not (args.out / "training-failure.json").exists():
            _write_new(
                args.out / "training-failure.json",
                {
                    "status": "invalid",
                    "type": type(error).__name__,
                    "traceback": traceback.format_exc(),
                },
            )
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        os._exit(1)


if __name__ == "__main__":
    raise SystemExit(main())
