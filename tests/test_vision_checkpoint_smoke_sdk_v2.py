from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest  # pyright: ignore[reportMissingImports]

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
import vision_checkpoint_smoke_sdk_v2 as S  # pyright: ignore[reportMissingImports]


class _Scalar:
    def __init__(self, value: bool) -> None:
        self.value = value

    def all(self) -> _Scalar:
        return self

    def item(self) -> bool:
        return self.value


class _Tensor:
    def __init__(self, shape: tuple[int, ...], data: bytes) -> None:
        self.shape = shape
        self.data = data
        self.dtype = "torch.float32"

    def detach(self) -> _Tensor:
        return self

    def to(self, **_kwargs: Any) -> _Tensor:
        return self

    def contiguous(self) -> _Tensor:
        return self

    def numpy(self) -> Any:
        return SimpleNamespace(tobytes=lambda **_kwargs: self.data)

    def new_tensor(self, _value: Any) -> _Tensor:
        return _Tensor((3,), b"command")


class _NoGrad:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *_args: object) -> None:
        return None


class _Torch:
    float32 = "torch.float32"

    @staticmethod
    def isfinite(_value: Any) -> _Scalar:
        return _Scalar(True)

    @staticmethod
    def no_grad() -> _NoGrad:
        return _NoGrad()


class _Command:
    def __init__(self) -> None:
        self.writes: list[Any] = []

    def __setitem__(self, key: Any, value: Any) -> None:
        self.writes.append((key, value))

    def new_tensor(self, value: Any) -> _Tensor:
        return _Tensor((len(value),), b"command")


class _Env:
    def __init__(self, log: list[str], *, step_error: BaseException | None = None,
                 close_error: BaseException | None = None) -> None:
        self.log = log
        self.unwrapped = self
        self.device = "cuda:0"
        self.common_step_counter = 0
        self.command = _Command()
        self.command_manager = SimpleNamespace(
            get_command=lambda name: self.command if name == "base_velocity" else None
        )
        self.step_error = step_error
        self.close_error = close_error

    def get_observations(self) -> tuple[_Tensor, dict[str, Any]]:
        self.log.append("observation")
        return _Tensor((1, 48), b"observation"), {}

    def step(self, _action: Any) -> None:
        self.log.append("policy_step")
        if self.step_error is not None:
            raise self.step_error
        self.common_step_counter += 1

    def close(self) -> None:
        self.log.append("environment_close")
        if self.close_error is not None:
            raise self.close_error


def _validated(tmp_path: Path) -> dict[str, Any]:
    job = tmp_path / "job.json"
    checkpoint = tmp_path / "model.pt"
    output = tmp_path / "output"
    job.write_text("{}", encoding="utf-8")
    checkpoint.write_bytes(b"model")
    output.mkdir()
    return {
        "job": {"mode": S.MODE},
        "job_path": job,
        "job_sha256": hashlib.sha256(job.read_bytes()).hexdigest(),
        "output_directory": output,
        "checkpoint_path": checkpoint,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    }


def _deps() -> dict[str, Any]:
    return {name: object() for name in S.PREPARED._DEPENDENCY_KEYS}


def _factory(env: _Env, policy: Any) -> Any:
    def factory(validated: dict[str, Any], _make_env: Any, state: dict[str, bool],
                holder: dict[str, Any]) -> Any:
        def create(task: str, _cfg: Any) -> Any:
            cfg = SimpleNamespace(device=None, clip_actions=1.0, to_dict=dict)

            class Runner:
                def __init__(self, *_args: Any, **_kwargs: Any) -> None:
                    pass

                def load(self, path: str, *, load_optimizer: bool) -> None:
                    assert path == str(validated["checkpoint_path"])
                    assert load_optimizer is False

                def get_inference_policy(self, *, device: str) -> Any:
                    assert device == "cuda:0"
                    return policy

            return S._policy_environment(
                validated, task, env, state, holder, torch=_Torch,
                wrapper_type=lambda raw, **_kwargs: raw,
                load_cfg=lambda *_args: cfg, runner_type=Runner,
            )
        return create
    return factory


def test_capture_and_detach_precede_single_policy_step(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    log: list[str] = []
    env = _Env(log)
    sealed: list[dict[str, Any]] = []
    deps = _deps()
    deps["make_env"] = lambda *_args: env

    class Writer:
        def detach(self) -> None:
            log.append("writer_detach")

        def raise_callback_error(self) -> None:
            log.append("writer_clean")

    deps["writer"] = lambda *_args: Writer()

    def prepared(validated: dict[str, Any], _app: Any, *, seal_report: Any,
                 dependencies: dict[str, Any]) -> dict[str, Any]:
        assert validated["job"]["mode"] == S.PREPARED.MODE
        selected = dependencies["make_env"]("task", object())
        assert env.common_step_counter == 0
        log.append("capture_complete")
        mux = SimpleNamespace(
            state="detaching", _prepare={"valid": True}, _arms=[{}] * 4,
            _failure=None,
        )
        writer = dependencies["writer"](object(), mux)
        writer.detach()
        writer.raise_callback_error()
        assert env.common_step_counter == 0
        selected.close()
        assert env.common_step_counter == 1
        before = {"schema": S.PREPARED.SDK_REPORT_SCHEMA, "approved": True,
                  "lifecycle": {"envelope_sealed": False}}
        seal_report(before)
        return {"schema": S.PREPARED.SDK_REPORT_SCHEMA, "approved": True,
                "lifecycle": {"envelope_sealed": True}}

    monkeypatch.setattr(S.PREPARED, "run_prepared_smoke", prepared)
    result = S.run_checkpoint_smoke(
        _validated(tmp_path), object(),
        seal_report=lambda value: sealed.append(dict(value)), dependencies=deps,
        policy_env_factory=_factory(env, lambda _obs: _Tensor((1, 12), b"action")),
    )
    assert result["approved"] is True
    assert result["policy_probe"]["action"]["shape"] == [1, 12]
    assert log.index("capture_complete") < log.index("writer_detach")
    assert log.index("writer_clean") < log.index("policy_step")
    assert log.index("policy_step") < log.index("environment_close")
    assert sealed[0]["policy_probe"]["writer_detached_before_policy_step"] is True


def test_policy_step_is_refused_before_complete_detach(tmp_path: Path) -> None:
    env = _Env([])
    proxy = S._PolicyEnvProxy(
        env, lambda _obs: _Tensor((1, 12), b"action"), _Torch,
        _validated(tmp_path)["checkpoint_sha256"],
        {"capture_completed": True, "writer_detached": False}, {},
    )
    with pytest.raises(RuntimeError, match="Writer detach 전에"):
        proxy.close()
    assert env.common_step_counter == 0
    assert env.log == ["environment_close"]


def test_step_failure_preserves_primary_when_close_also_fails(tmp_path: Path) -> None:
    primary = ValueError("policy failed")
    env = _Env([], step_error=primary, close_error=OSError("close failed"))
    proxy = S._PolicyEnvProxy(
        env, lambda _obs: _Tensor((1, 12), b"action"), _Torch,
        _validated(tmp_path)["checkpoint_sha256"],
        {"capture_completed": True, "writer_detached": True}, {},
    )
    with pytest.raises(ValueError, match="policy failed") as caught:
        proxy.close()
    assert caught.value is primary
    assert primary.__dict__["checkpoint_probe_cleanup_error"] == {
        "error_type": "OSError", "error_message": "close failed"
    }


def test_writer_proxy_preserves_duplicate_callback_failure() -> None:
    state = {"capture_completed": False, "writer_detached": False}

    class Writer:
        def detach(self) -> None:
            pass

        def raise_callback_error(self) -> None:
            raise RuntimeError("callback_cardinality_invalid")

    mux = SimpleNamespace(
        state="detaching", _prepare={}, _arms=[{}] * 4, _failure=None
    )
    writer = S._WriterProxy(Writer(), mux, state)
    writer.detach()
    with pytest.raises(RuntimeError, match="callback_cardinality_invalid"):
        writer.raise_callback_error()
    assert state == {"capture_completed": False, "writer_detached": False}


def test_checkpoint_load_failure_closes_environment_and_preserves_first_error(
    tmp_path: Path,
) -> None:
    env = _Env([])

    class Runner:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def load(self, *_args: Any, **_kwargs: Any) -> None:
            raise FileNotFoundError("runner checkpoint payload missing")

    cfg = SimpleNamespace(device=None, clip_actions=1.0, to_dict=dict)
    with pytest.raises(FileNotFoundError, match="runner checkpoint payload missing"):
        S._policy_environment(
            _validated(tmp_path), "task", env,
            {"capture_completed": False, "writer_detached": False}, {},
            torch=_Torch, wrapper_type=lambda raw, **_kwargs: raw,
            load_cfg=lambda *_args: cfg, runner_type=Runner,
        )
    assert env.log == ["environment_close"]
