# pyright: reportMissingImports=false
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import pytest

from rl_triage import optimizer_trace as T


class Optimizer:
    def __init__(self) -> None:
        self.calls = 0

    def step(self) -> None:
        self.calls += 1


class Storage:
    def __init__(self, learner: Learner) -> None:
        self.learner = learner
        self.recurrent = False
        self.fail_after: int | None = None

    def mini_batch_generator(self):
        yield from self._items()

    def recurrent_mini_batch_generator(self):
        yield from self._items()

    def _items(self):
        count = self.learner.num_learning_epochs * self.learner.num_mini_batches
        for index in range(count):
            if self.fail_after is not None and index == self.fail_after:
                raise RuntimeError("training failed")
            yield (index,)


class Learner:
    def __init__(self, epochs: int = 2, minibatches: int = 3) -> None:
        self.num_learning_epochs = epochs
        self.num_mini_batches = minibatches
        self.optimizer = Optimizer()
        self.storage = Storage(self)

    def update(self) -> str:
        generator = (self.storage.recurrent_mini_batch_generator()
                     if self.storage.recurrent else self.storage.mini_batch_generator())
        for _batch in generator:
            self.optimizer.step()
        return "updated"


def test_real_dummy_consumer_counts_standard_and_late_config_change_and_restores_hooks(
    tmp_path: Path
) -> None:
    learner = Learner()
    original_update = learner.update
    original_step = learner.optimizer.step
    original_generator = learner.storage.mini_batch_generator
    source = tmp_path / "runner.py"
    source.write_text("runner", encoding="utf-8")
    out = tmp_path / "trace"
    with T.observe_optimizer(
        learner, out=out, run_id="user-run-01", source_files=[source],
        config_manifest={"algorithm": "PPO"}, checkpoint_sha256="a" * 64,
    ) as session:
        assert session["scope"]["diagnostic_authority"] is False
        assert session["checkpoint_association"]["authority"] == "caller_supplied_association"
        assert learner.update() == "updated"
        learner.num_learning_epochs = 1
        learner.num_mini_batches = 4
        assert learner.update() == "updated"

    report = T.inspect_trace(out)
    assert report["result"] == "consistent"
    assert report["counts"] == [
        {"update_index": 0, "optimizer_steps": 6, "minibatches_yielded": 6,
         "declared_expected_optimizer_steps": 6, "comparison": "consistent"},
        {"update_index": 1, "optimizer_steps": 4, "minibatches_yielded": 4,
         "declared_expected_optimizer_steps": 4, "comparison": "consistent"},
    ]
    manifest = json.loads((out / T.MANIFEST).read_text(encoding="utf-8"))
    assert manifest["source_files"] == [{
        "source_id": "source-01", "sha256": T._sha(source), "bytes": 6}]
    assert str(source.resolve()) not in json.dumps(manifest)
    assert learner.update.__func__ is original_update.__func__
    assert learner.optimizer.step.__func__ is original_step.__func__
    assert learner.storage.mini_batch_generator.__func__ is original_generator.__func__
    assert not hasattr(learner, T._MARKER)


def test_recurrent_generator_is_counted_without_tensor_values(tmp_path: Path) -> None:
    learner = Learner(epochs=2, minibatches=2)
    learner.storage.recurrent = True
    out = tmp_path / "trace"
    with T.observe_optimizer(learner, out=out, run_id="recurrent"):
        learner.update()
    report = T.inspect_trace(out)
    assert report["counts"][0]["optimizer_steps"] == 4
    encoded = (out / T.RECORDS).read_text(encoding="utf-8")
    assert "tensor" not in encoded.lower()
    assert "(0,)" not in encoded


def test_update_exception_records_incomplete_restores_hooks_and_rethrows(tmp_path: Path) -> None:
    learner = Learner()
    learner.storage.fail_after = 1
    original_update = learner.update
    out = tmp_path / "trace"
    with pytest.raises(RuntimeError, match="training failed"), \
            T.observe_optimizer(learner, out=out, run_id="failed-update"):
        learner.update()
    assert learner.update.__func__ is original_update.__func__
    report = T.inspect_trace(out)
    assert report["result"] == "unknown"
    assert report["incomplete_updates"] == 1
    row = json.loads((out / T.RECORDS).read_text(encoding="utf-8"))
    assert row["status"] == "incomplete"
    assert row["exception_type"] == "RuntimeError"
    assert "training failed" not in json.dumps(row)


def test_missing_declared_config_is_unknown_not_zero(tmp_path: Path) -> None:
    class UnknownStorage:
        def mini_batch_generator(self):
            return iter((1,))

    class UnknownLearner:
        def __init__(self) -> None:
            self.optimizer = Optimizer()
            self.storage = UnknownStorage()

        def update(self) -> None:
            for _ in self.storage.mini_batch_generator():
                self.optimizer.step()

    learner = UnknownLearner()
    out = tmp_path / "trace"
    with T.observe_optimizer(learner, out=out, run_id="unknown-config"):
        learner.update()
    row = json.loads((out / T.RECORDS).read_text(encoding="utf-8"))
    assert row["declared_expected_optimizer_steps"] is None
    assert row["comparison"] == "unknown"
    assert row["config_snapshot"]["declared_learning_epochs"] is None


def test_inconsistent_counts_and_duplicate_instrumentation_fail_closed(tmp_path: Path) -> None:
    learner = Learner()
    def missing_steps() -> str:
        for _ in learner.storage.mini_batch_generator():
            pass
        return "updated"

    cast(Any, learner).update = missing_steps
    out = tmp_path / "trace"
    with T.observe_optimizer(learner, out=out, run_id="inconsistent"):
        with pytest.raises(ValueError, match="중복"), \
                T.observe_optimizer(learner, out=tmp_path / "second", run_id="duplicate"):
            pass
        learner.update()
    assert T.inspect_trace(out)["result"] == "inconsistent"


def test_caller_exception_without_update_closes_unknown_session(tmp_path: Path) -> None:
    learner = Learner()
    out = tmp_path / "trace"
    with pytest.raises(KeyError), T.observe_optimizer(learner, out=out, run_id="caller-error"):
        raise KeyError("outside update")
    report = T.inspect_trace(out)
    assert report["result"] == "unknown"
    readback = json.loads((out / T.READBACK).read_text(encoding="utf-8"))
    assert readback["session_status"] == "caller_exception"
    assert readback["caller_exception_type"] == "KeyError"


def test_hash_chain_readback_and_cli_reject_tamper(tmp_path: Path) -> None:
    learner = Learner(1, 1)
    out = tmp_path / "trace"
    with T.observe_optimizer(learner, out=out, run_id="inspect-cli"):
        learner.update()
        learner.update()
    command = [sys.executable, "-m", "rl_triage.optimizer_trace", "inspect", "--out", str(out)]
    result = subprocess.run(command, cwd=Path(__file__).resolve().parents[1], check=True,
                            capture_output=True, encoding="utf-8", errors="replace")
    assert json.loads(result.stdout)["updates_recorded"] == 2
    lines = (out / T.RECORDS).read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[1])
    row["optimizer_steps"] = 99
    lines[1] = json.dumps(row)
    (out / T.RECORDS).write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="record SHA"):
        T.inspect_trace(out)


def test_dynamic_optimizer_and_storage_replacement_is_observed(tmp_path: Path) -> None:
    learner = Learner(1, 2)
    out = tmp_path / "trace"
    with T.observe_optimizer(learner, out=out, run_id="dynamic"):
        learner.update()
        learner.optimizer = Optimizer()
        learner.storage = Storage(learner)
        learner.update()
    report = T.inspect_trace(out)
    assert [row["comparison"] for row in report["counts"]] == ["consistent", "consistent"]


def test_import_does_not_load_torch_or_isaac_modules() -> None:
    code = (
        "import sys; import rl_triage.optimizer_trace; "
        "print(any(n == 'torch' or n.startswith(('torch.', 'isaaclab', 'omni.')) for n in sys.modules))"
    )
    result = subprocess.run([sys.executable, "-c", code],
                            cwd=Path(__file__).resolve().parents[1], check=True,
                            capture_output=True, encoding="utf-8", errors="replace")
    assert result.stdout.strip() == "False"
