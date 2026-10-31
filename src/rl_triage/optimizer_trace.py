"""결함 label 없이 RSL-style PPO update의 optimizer/minibatch 횟수를 기록한다."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

SESSION_SCHEMA = "optimizer_trace_session_v1"
RECORD_SCHEMA = "optimizer_trace_record_v1"
READBACK_SCHEMA = "optimizer_trace_readback_v1"
INSPECTION_SCHEMA = "optimizer_trace_inspection_v1"
MANIFEST = "session.json"
RECORDS = "records.jsonl"
READBACK = "readback.json"
_MARKER = "_rl_triage_optimizer_trace_session_v1"


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def _canonical(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode("utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=lambda token: (
        _ for _ in ()).throw(ValueError(f"비유한 JSON 숫자: {token}")))
    _require(isinstance(value, dict), f"{Path(path).name}은 JSON 객체여야 한다")
    return value


def _write_new(path: Path, value: object) -> None:
    path = Path(path).resolve()
    _require(not path.exists(), f"{path.name}은 write-once 파일이어야 한다")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical(value))


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()  # noqa: UP017 - Isaac Python 3.10


def _descriptor(value: object) -> dict[str, str]:
    kind = f"{type(value).__module__}.{type(value).__qualname__}"
    try:
        encoded = _canonical(value)
    except (TypeError, ValueError):
        encoded = repr(value).encode("utf-8", errors="replace")
    return {"type": kind, "value_sha256": hashlib.sha256(encoded).hexdigest()}


def _config_value(learner: object, name: str) -> object | None:
    if hasattr(learner, name):
        return getattr(learner, name)
    config = getattr(learner, "cfg", None)
    if isinstance(config, dict):
        return config.get(name)
    if config is not None and hasattr(config, name):
        return getattr(config, name)
    return None


def _positive_int(value: object) -> int | None:
    return value if type(value) is int and value > 0 else None


def _config_snapshot(learner: object, config_manifest: object | None) -> dict[str, Any]:
    epochs = _config_value(learner, "num_learning_epochs")
    minibatches = _config_value(learner, "num_mini_batches")
    declared_epochs, declared_minibatches = _positive_int(epochs), _positive_int(minibatches)
    expected = (declared_epochs * declared_minibatches
                if declared_epochs is not None and declared_minibatches is not None else None)
    return {
        "num_learning_epochs": _descriptor(epochs),
        "num_mini_batches": _descriptor(minibatches),
        "declared_learning_epochs": declared_epochs,
        "declared_mini_batches": declared_minibatches,
        "declared_expected_optimizer_steps": expected,
        "config_manifest": None if config_manifest is None else _descriptor(config_manifest),
    }


def _source_descriptors(source_files: object | None) -> list[dict[str, object]]:
    if source_files is None:
        return []
    _require(isinstance(source_files, (list, tuple)), "source_files는 경로 목록이어야 한다")
    rows = []
    for index, item in enumerate(cast(Iterable[Path | str], source_files), 1):
        path = Path(cast(Path | str, item)).resolve()
        _require(path.is_file(), f"source file {index}가 없다")
        rows.append({"source_id": f"source-{index:02d}", "sha256": _sha(path),
                     "bytes": path.stat().st_size})
    return rows


def _classification(status: str, steps: int, minibatches: int,
                    expected: int | None) -> str:
    if status != "complete":
        return "unknown"
    if expected is None:
        return "inconsistent" if steps != minibatches else "unknown"
    return "consistent" if steps == minibatches == expected else "inconsistent"


class _Patch:
    def __init__(self, owner: object, name: str, replacement: object):
        self.owner, self.name = owner, name
        namespace = getattr(owner, "__dict__", {})
        self.had_instance = name in namespace
        self.instance_value = namespace.get(name)
        self.original = getattr(owner, name)
        setattr(owner, name, replacement)

    def restore(self) -> None:
        if self.had_instance:
            setattr(self.owner, self.name, self.instance_value)
        else:
            delattr(self.owner, self.name)


class _Session:
    def __init__(self, learner: object, out: Path, run_id: str, source_files: object | None,
                 config_manifest: object | None, checkpoint_sha256: str | None):
        _require(re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", run_id) is not None, "run_id 형식")
        _require(callable(getattr(learner, "update", None)), "learner.update가 callable이 아니다")
        _require(not hasattr(learner, _MARKER), "같은 learner의 중복 instrumentation은 허용하지 않는다")
        if checkpoint_sha256 is not None:
            _require(re.fullmatch(r"[0-9a-f]{64}", checkpoint_sha256) is not None,
                     "checkpoint SHA 형식")
        self.learner, self.out = learner, Path(out).resolve()
        _require(not self.out.exists(), "trace out은 fresh 디렉터리여야 한다")
        self.run_id, self.config_manifest = run_id, config_manifest
        self.patches: list[_Patch] = []
        self.dynamic_owners: tuple[object | None, object | None] = (None, None)
        self.active: dict[str, Any] | None = None
        self.sequence = 0
        self.previous: str | None = None
        self.closed = False
        self.out.mkdir(parents=True, exist_ok=False)
        manifest = {
            "schema": SESSION_SCHEMA, "run_id": run_id, "created_utc": _utc(),
            "source_files": _source_descriptors(source_files),
            "config_manifest": None if config_manifest is None else _descriptor(config_manifest),
            "checkpoint_association": None if checkpoint_sha256 is None else {
                "sha256": checkpoint_sha256, "authority": "caller_supplied_association"},
            "scope": {"evidence": "caller_instrumented_self_reported",
                      "approval": False, "diagnostic_authority": False,
                      "recovery_authority": False, "owned_gpu_receipt": False,
                      "fault_label_required": False, "tensor_values_recorded": False},
        }
        _write_new(self.out / MANIFEST, manifest)
        setattr(learner, _MARKER, self)
        original_update = cast(Any, learner).update

        def update_wrapper(*args: object, **kwargs: object) -> object:
            self._refresh_dynamic_hooks()
            _require(self.active is None, "learner.update 재진입은 허용하지 않는다")
            self.active = {"steps": 0, "minibatches": 0,
                           "utc_started": _utc(), "monotonic_started_ns": time.monotonic_ns(),
                           "config": _config_snapshot(self.learner, self.config_manifest)}
            try:
                value = original_update(*args, **kwargs)
            except BaseException as error:
                self._record("incomplete", type(error).__name__)
                raise
            self._record("complete", None)
            return value

        try:
            self.patches.append(_Patch(learner, "update", update_wrapper))
            self._refresh_dynamic_hooks()
        except BaseException:
            for patch in reversed(self.patches):
                patch.restore()
            self.patches.clear()
            if getattr(learner, _MARKER, None) is self:
                delattr(learner, _MARKER)
            raise

    def _refresh_dynamic_hooks(self) -> None:
        optimizer = getattr(self.learner, "optimizer", None)
        storage = getattr(self.learner, "storage", None)
        if (optimizer, storage) == self.dynamic_owners:
            return
        while len(self.patches) > 1:
            self.patches.pop().restore()
        self.dynamic_owners = (optimizer, storage)
        if optimizer is not None and callable(getattr(optimizer, "step", None)):
            original = optimizer.step

            def step_wrapper(*args: object, **kwargs: object) -> object:
                if self.active is not None:
                    self.active["steps"] += 1
                return original(*args, **kwargs)

            self.patches.append(_Patch(optimizer, "step", step_wrapper))
        if storage is not None:
            for name in ("mini_batch_generator", "recurrent_mini_batch_generator"):
                method = getattr(storage, name, None)
                if not callable(method):
                    continue

                def generator_wrapper(*args: object, _method=method,
                                      **kwargs: object) -> Iterator[object]:
                    for item in _method(*args, **kwargs):
                        if self.active is not None:
                            self.active["minibatches"] += 1
                        yield item

                self.patches.append(_Patch(storage, name, generator_wrapper))

    def _record(self, status: str, exception_type: str | None) -> None:
        active = self.active
        _require(isinstance(active, dict), "active update가 없다")
        active = cast(dict[str, Any], active)
        self.sequence += 1
        steps, minibatches = int(active["steps"]), int(active["minibatches"])
        config = cast(dict[str, Any], active["config"])
        body = {
            "schema": RECORD_SCHEMA, "sequence": self.sequence,
            "previous_record_sha256": self.previous, "update_index": self.sequence - 1,
            "status": status, "optimizer_steps": steps,
            "minibatches_yielded": minibatches,
            "declared_expected_optimizer_steps": config["declared_expected_optimizer_steps"],
            "comparison": _classification(
                status, steps, minibatches, config["declared_expected_optimizer_steps"]),
            "config_snapshot": config,
            "timing": {"utc_started": active["utc_started"], "utc_finished": _utc(),
                       "monotonic_started_ns": active["monotonic_started_ns"],
                       "monotonic_finished_ns": time.monotonic_ns()},
            "exception_type": exception_type,
        }
        row = {**body, "record_sha256": _digest(body)}
        with (self.out / RECORDS).open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(_canonical(row).decode("utf-8"))
        self.previous = row["record_sha256"]
        self.active = None

    def close(self, caller_exception: BaseException | None) -> None:
        if self.closed:
            return
        self.closed = True
        for patch in reversed(self.patches):
            patch.restore()
        self.patches.clear()
        if getattr(self.learner, _MARKER, None) is self:
            delattr(self.learner, _MARKER)
        records = self.out / RECORDS
        if not records.exists():
            records.write_bytes(b"")
        rows = _records(records)
        comparisons = [row["comparison"] for row in rows]
        result = ("inconsistent" if "inconsistent" in comparisons else
                  "consistent" if rows and set(comparisons) == {"consistent"} else "unknown")
        readback = {
            "schema": READBACK_SCHEMA, "run_id": self.run_id,
            "manifest_sha256": _sha(self.out / MANIFEST), "records_sha256": _sha(records),
            "updates_recorded": len(rows), "completed_updates": sum(
                row["status"] == "complete" for row in rows),
            "incomplete_updates": sum(row["status"] == "incomplete" for row in rows),
            "result": result,
            "session_status": "caller_exception" if caller_exception is not None else "closed",
            "caller_exception_type": None if caller_exception is None else type(caller_exception).__name__,
            "closed_utc": _utc(),
            "scope": {"diagnostic_authority": False, "recovery_authority": False},
        }
        _write_new(self.out / READBACK, readback)


def _records(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    previous = None
    for sequence, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        row = json.loads(line, parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
        _require(isinstance(row, dict) and row.get("schema") == RECORD_SCHEMA
                 and row.get("sequence") == sequence
                 and row.get("previous_record_sha256") == previous,
                 "optimizer trace record 순서/hash chain이 다르다")
        expected = row.get("declared_expected_optimizer_steps")
        _require(expected is None or (type(expected) is int and expected > 0),
                 "declared expected optimizer steps 형식이 다르다")
        claimed = row.get("record_sha256")
        body = {key: value for key, value in row.items() if key != "record_sha256"}
        _require(claimed == _digest(body), "optimizer trace record SHA가 다르다")
        _require(row.get("status") in {"complete", "incomplete"}
                 and type(row.get("optimizer_steps")) is int and row["optimizer_steps"] >= 0
                 and type(row.get("minibatches_yielded")) is int and row["minibatches_yielded"] >= 0
                 and row.get("comparison") == _classification(
                     row["status"], row["optimizer_steps"], row["minibatches_yielded"],
                     cast(int | None, expected)),
                 "optimizer trace record count/status 판정이 다르다")
        timing = row.get("timing")
        _require(isinstance(timing, dict)
                 and type(timing.get("monotonic_started_ns")) is int
                 and type(timing.get("monotonic_finished_ns")) is int
                 and timing["monotonic_finished_ns"] >= timing["monotonic_started_ns"],
                 "optimizer trace monotonic timing이 다르다")
        previous = cast(str, claimed)
        rows.append(cast(dict[str, Any], row))
    return rows


@contextmanager
def observe_optimizer(
    learner: object, *, out: Path, run_id: str, source_files: object | None = None,
    config_manifest: object | None = None, checkpoint_sha256: str | None = None,
) -> Iterator[dict[str, Any]]:
    """`runner.learn()` 바깥에서 사용하며 모든 hook을 finally에서 원복한다."""
    session = _Session(learner, Path(out), run_id, source_files,
                       config_manifest, checkpoint_sha256)
    try:
        yield deepcopy(_read(Path(out) / MANIFEST))
    except BaseException as error:
        session.close(error)
        raise
    else:
        session.close(None)


def inspect_trace(out: Path) -> dict[str, Any]:
    """session/readback/JSONL hash chain을 현재 bytes에서 다시 검사한다."""
    out = Path(out).resolve()
    manifest = _read(out / MANIFEST)
    _require(manifest.get("schema") == SESSION_SCHEMA
             and manifest.get("scope") == {
                 "evidence": "caller_instrumented_self_reported", "approval": False,
                 "diagnostic_authority": False, "recovery_authority": False,
                 "owned_gpu_receipt": False, "fault_label_required": False,
                 "tensor_values_recorded": False}, "optimizer trace session scope가 다르다")
    rows = _records(out / RECORDS)
    readback = _read(out / READBACK)
    _require(readback.get("schema") == READBACK_SCHEMA
             and readback.get("run_id") == manifest.get("run_id")
             and readback.get("manifest_sha256") == _sha(out / MANIFEST)
             and readback.get("records_sha256") == _sha(out / RECORDS)
             and readback.get("updates_recorded") == len(rows)
             and readback.get("completed_updates") == sum(
                 row["status"] == "complete" for row in rows)
             and readback.get("incomplete_updates") == sum(
                 row["status"] == "incomplete" for row in rows),
             "optimizer trace readback이 현재 원문과 다르다")
    comparisons = [row["comparison"] for row in rows]
    result = ("inconsistent" if "inconsistent" in comparisons else
              "consistent" if rows and set(comparisons) == {"consistent"} else "unknown")
    _require(readback.get("result") == result, "optimizer trace 종합 판정이 다르다")
    return {
        "schema": INSPECTION_SCHEMA, "run_id": manifest["run_id"], "result": result,
        "updates_recorded": len(rows), "completed_updates": readback["completed_updates"],
        "incomplete_updates": readback["incomplete_updates"],
        "counts": [{"update_index": row["update_index"],
                    "optimizer_steps": row["optimizer_steps"],
                    "minibatches_yielded": row["minibatches_yielded"],
                    "declared_expected_optimizer_steps": row["declared_expected_optimizer_steps"],
                    "comparison": row["comparison"]} for row in rows],
        "scope": deepcopy(manifest["scope"]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("inspect")
    inspect.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = inspect_trace(args.out)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
