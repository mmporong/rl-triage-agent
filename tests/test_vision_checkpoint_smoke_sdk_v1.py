from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest  # pyright: ignore[reportMissingImports]

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
import vision_checkpoint_smoke_sdk_v1 as S  # pyright: ignore[reportMissingImports]


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


def _factory(validated: dict[str, Any], original: Any) -> Any:
    assert original is not None

    def create(*_args: Any, **_kwargs: Any) -> Any:
        return SimpleNamespace(
            _rl_triage_checkpoint_probe={
                "schema": "vision_checkpoint_policy_probe_v1",
                "checkpoint_sha256": validated["checkpoint_sha256"],
                "command": [1.0, 0.0, 0.0],
                "observation": {
                    "shape": [1, 48],
                    "dtype": "torch.float32",
                    "sha256": "a" * 64,
                },
                "action": {
                    "shape": [1, 12],
                    "dtype": "torch.float32",
                    "sha256": "b" * 64,
                },
                "environment_steps": 1,
                "step_counter_before": 0,
                "step_counter_after": 1,
                "reset_required_before_capture": True,
                "actual_environment_observation_used": True,
                "pure_checkpoint_read_only": False,
            }
        )

    return create


def test_wraps_policy_step_and_existing_smoke(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sealed: list[dict[str, Any]] = []

    def inner(
        validated: dict[str, Any],
        _app: Any,
        *,
        seal_report: Any,
        dependencies: dict[str, Any],
    ) -> dict[str, Any]:
        assert validated["job"]["mode"] == S.PREPARED.MODE
        dependencies["make_env"]("task", object())
        before = {
            "schema": S.PREPARED.SDK_REPORT_SCHEMA,
            "approved": True,
            "lifecycle": {"envelope_sealed": False},
        }
        seal_report(before)
        return {
            "schema": S.PREPARED.SDK_REPORT_SCHEMA,
            "approved": True,
            "lifecycle": {"envelope_sealed": True},
        }

    monkeypatch.setattr(S.PREPARED, "run_prepared_smoke", inner)
    result = S.run_checkpoint_smoke(
        _validated(tmp_path),
        object(),
        seal_report=lambda value: sealed.append(dict(value)),
        dependencies=_deps(),
        policy_env_factory=_factory,
    )
    assert result["approved"] is True
    assert result["policy_probe"]["environment_steps"] == 1
    assert result["policy_probe"]["pure_checkpoint_read_only"] is False
    assert sealed[0]["approved"] is True
    assert sealed[0]["prepared_smoke"]["lifecycle"]["envelope_sealed"] is False


def test_missing_policy_probe_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def bad_factory(_validated: dict[str, Any], _original: Any) -> Any:
        return lambda *_args, **_kwargs: SimpleNamespace()

    def inner(
        _validated: dict[str, Any],
        _app: Any,
        *,
        seal_report: Any,
        dependencies: dict[str, Any],
    ) -> dict[str, Any]:
        dependencies["make_env"]("task", object())
        raise AssertionError("unreachable")

    monkeypatch.setattr(S.PREPARED, "run_prepared_smoke", inner)
    with pytest.raises(S.CheckpointSmokeSDKFailure) as caught:
        S.run_checkpoint_smoke(
            _validated(tmp_path),
            object(),
            seal_report=lambda _value: None,
            dependencies=_deps(),
            policy_env_factory=bad_factory,
        )
    assert caught.value.report["approved"] is False
    assert caught.value.report["policy_probe"] is None


def test_checkpoint_load_failure_closes_environment_and_preserves_first_error(
    tmp_path: Path,
) -> None:
    validated = _validated(tmp_path)
    raw = SimpleNamespace(closed=False)
    raw.close = lambda: setattr(raw, "closed", True)
    env = SimpleNamespace(close=raw.close)

    class Runner:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def load(self, *_args: Any, **_kwargs: Any) -> None:
            raise FileNotFoundError("runner checkpoint payload missing")

    cfg = SimpleNamespace(
        device=None,
        clip_actions=1.0,
        to_dict=dict,
    )
    with pytest.raises(FileNotFoundError, match="runner checkpoint payload missing"):
        S._policy_step(
            validated,
            "task",
            raw,
            torch=object(),
            wrapper_type=lambda *_args, **_kwargs: env,
            load_cfg=lambda *_args: cfg,
            runner_type=Runner,
        )
    assert raw.closed is True
