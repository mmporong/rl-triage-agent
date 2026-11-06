"""재작성 패치와 설정 누락 결함을 설치된 PPO 기본값으로 비교한다."""
from __future__ import annotations

import copy
import inspect
import json
from contextlib import contextmanager
from pathlib import Path
from threading import RLock
from typing import Any

import train_generated_ppo_adapter_v1 as V1

ROOT = V1.ROOT
ABI_PATH = ROOT / "bench/fixtures/ppo_config_omission_v1/ppo_defaults.json"
ABI = json.loads(ABI_PATH.read_text(encoding="utf-8"))
DEFAULTS = ABI["constructor_defaults"]
TRAINING_SCHEMA = "generated_ppo_adapter_omission_training_v2"
SMOKE_SCHEMA = "generated_ppo_adapter_omission_smoke_v2"
PROTOCOL_SCHEMA = "generated_ppo_adapter_omission_protocol_v2"
RESULT_SCHEMA = "generated_ppo_adapter_omission_result_v2"
TASK = V1.TASK
TRAINING_BY_SCHEMA = {
    TRAINING_SCHEMA: {"num_envs": 4096, "updates": 300},
    SMOKE_SCHEMA: {"num_envs": 4096, "updates": 1},
}
REQUIRED_SOURCES = V1.REQUIRED_SOURCES | {
    "bench/train_generated_ppo_adapter_v2.py",
    "tests/test_train_generated_ppo_adapter_v2.py",
    "bench/fixtures/ppo_config_omission_v1/ppo_defaults.json",
}
FAULT_KEYS = {"num_mini_batches", "learning_rate", "schedule"}
CHECKED_PARAMETERS = (
    "num_learning_epochs", "num_mini_batches", "clip_param", "gamma", "lam",
    "value_loss_coef", "entropy_coef", "max_grad_norm", "use_clipped_value_loss",
    "schedule", "desired_kl", "normalize_advantage_per_mini_batch",
)
_LOCK = RLock()
_ORIGINAL_TRAIN = V1.train
_RESOLUTIONS: list[dict[str, Any]] = []


def effective_parameters(kwargs: dict[str, Any]) -> dict[str, Any]:
    """PPO 전달 인자는 유지하고 생성자 기본값으로 검증할 값을 계산한다."""
    return {key: kwargs.get(key, DEFAULTS[key]) for key in CHECKED_PARAMETERS}


def adapter_resolution(source: str, config: dict[str, Any], arm: str) -> dict[str, Any]:
    before = copy.deepcopy(config)
    raw = V1.CONTRACT.apply_adapter(source, config)
    if arm == "fault":
        V1._require(set(raw) == FAULT_KEYS, "설정 누락 결함의 전달 key 집합이 다르다")
        V1._require(raw == {key: before[key] for key in FAULT_KEYS},
                    "설정 누락 결함이 전달한 값도 바꿨다")
    else:
        V1._require(arm in {"healthy", "patched"} and raw == before,
                    "정상 또는 재작성 adapter가 설정을 보존하지 않았다")
    # class_name은 PPO 학습 인자가 아닌 OnPolicyRunner의 알고리즘 선택 메타데이터다.
    # 학습 인자는 보충하지 않아, 누락된 값은 실제 PPO 생성자가 정한다.
    forwarded = dict(raw)
    if "class_name" in before:
        V1._require(before["class_name"] == "PPO", "등록 알고리즘이 PPO가 아니다")
        forwarded["class_name"] = before["class_name"]
    V1._require(set(before) - set(forwarded) <= set(DEFAULTS),
                "설치된 PPO 기본값이 없는 인자가 누락됐다")
    return {
        "raw_adapter_kwargs": raw,
        "runner_kwargs": forwarded,
        "metadata_restored_by_host": ["class_name"] if "class_name" in before and "class_name" not in raw else [],
        "omitted_algorithm_keys": sorted(set(before) - set(forwarded)),
        "effective_parameters": effective_parameters(forwarded),
    }


def _apply_adapter_checked(source: str, config: dict[str, Any], arm: str) -> dict[str, Any]:
    resolution = adapter_resolution(source, config, arm)
    _RESOLUTIONS.append(resolution)
    return resolution["runner_kwargs"]


def verify_runtime_parameters(ppo: Any, expected: dict[str, Any]) -> None:
    for key, value in effective_parameters(expected).items():
        actual = getattr(ppo, key, None)
        V1._require(type(actual) is type(value) and actual == value,
                    f"실제 PPO {key}가 전달값 또는 설치된 기본값과 다르다")


def verify_constructor(ppo_class: Any) -> None:
    defaults = {
        name: parameter.default
        for name, parameter in inspect.signature(ppo_class.__init__).parameters.items()
        if parameter.default is not inspect.Parameter.empty
        and parameter.kind not in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
    }
    V1._require(defaults == DEFAULTS, "실제 PPO 생성자 기본값이 등록 ABI와 다르다")
    source = inspect.getsourcefile(ppo_class)
    V1._require(source is not None and V1._sha256_lf(Path(source)) == ABI["source_sha256_lf"],
                "실제 PPO 생성자 소스가 등록 ABI와 다르다")


def train(spec: dict[str, Any], *, out: Path) -> dict[str, Any]:
    from rsl_rl.algorithms import PPO

    verify_constructor(PPO)
    _RESOLUTIONS.clear()
    result = _ORIGINAL_TRAIN(spec, out=out)
    V1._require(len(_RESOLUTIONS) == 1, "학습 adapter 적용 횟수가 하나가 아니다")
    resolution = _RESOLUTIONS[0]
    V1._require(result["algorithm_after"] == resolution["runner_kwargs"],
                "실제 runner 인자와 adapter 영수증이 다르다")
    V1._write_new(out / "adapter-resolution.json", {
        "schema": "generated_ppo_adapter_resolution_v2",
        "adapter_source_sha256": spec["adapter_source_sha256"],
        "constructor_abi_sha256": V1._sha256(ABI_PATH),
        "constructor_source_sha256_lf": ABI["source_sha256_lf"],
        **resolution,
    })
    return result


@contextmanager
def training_scope():
    """이 프로세스에서 v1의 학습·종료·lease 구현을 재사용하고 원복한다."""
    replacements = {
        "TRAINING_SCHEMA": TRAINING_SCHEMA, "SMOKE_SCHEMA": SMOKE_SCHEMA,
        "PROTOCOL_SCHEMA": PROTOCOL_SCHEMA, "RESULT_SCHEMA": RESULT_SCHEMA,
        "TRAINING_BY_SCHEMA": TRAINING_BY_SCHEMA, "REQUIRED_SOURCES": REQUIRED_SOURCES,
        "_apply_adapter_checked": _apply_adapter_checked,
        "verify_runtime_parameters": verify_runtime_parameters, "train": train,
    }
    with _LOCK:
        original = {name: getattr(V1, name) for name in replacements}
        V1._require(original["train"] is _ORIGINAL_TRAIN, "학습 scope가 이미 교체돼 있다")
        for name, value in replacements.items():
            setattr(V1, name, value)
        try:
            yield
        finally:
            for name, value in original.items():
                setattr(V1, name, value)


def validate_spec_shape(value):
    with training_scope():
        return V1.validate_spec_shape(value)


def validate_spec(value, *, repo=ROOT):
    with training_scope():
        return V1.validate_spec(value, repo=repo)


def main(argv=None) -> int:
    with training_scope():
        return V1.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
