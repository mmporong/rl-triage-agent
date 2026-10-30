"""NONE 대조군의 실제 300회 학습 완료와 SDK 종료 증거를 결합한다."""

from __future__ import annotations

import argparse
import itertools
import os
import time
from pathlib import Path
from typing import Any

from local_f1_restart_base_v2 import read, sha, write_new

SETTING = "/app/fastShutdown"
_SEQ = itertools.count(1)


class TerminalEvidenceError(ValueError):
    pass


def marker(folder: Path, event: str, **fields: Any) -> Path:
    sequence = next(_SEQ)
    path = folder / f"{sequence:02d}-{event}.json"
    write_new(path, {"schema": "local_same_seed_none_terminal_v1", "event": event,
                     "pid": os.getpid(), "sequence": sequence,
                     "monotonic_ns": time.monotonic_ns(), **fields})
    return path


def training_evidence(output: Path, old_checkpoints: set[str]) -> dict[str, Any]:
    value = read(output)
    checkpoint = value.get("checkpoint", {})
    target = Path(checkpoint.get("path", ""))
    manifest = Path(value.get("manifest_path", ""))
    if (value.get("completed") is not True or value.get("arm") != "NONE"
            or value.get("seed") != 3221 or value.get("iterations") != 300
            or target.name != "model_299.pt" or not target.is_file()
            or sha(target) != checkpoint.get("sha256")
            or checkpoint.get("sha256") in old_checkpoints
            or not manifest.is_file() or sha(manifest) != value.get("manifest_sha256")):
        raise TerminalEvidenceError("NONE model_299 완료 증거가 유효하지 않다")
    descriptor = read(manifest)
    if (descriptor.get("arm") != "NONE" or descriptor.get("seed") != 3221
            or descriptor.get("max_iterations") != 300
            or descriptor.get("source_checkpoint_sha256") is not None
            or descriptor.get("original_faults") != []
            or descriptor.get("applied_corrections") != []
            or value.get("readback", {}).get("remaining_faults") != []):
        raise TerminalEvidenceError("NONE 대조군의 학습/결함/재개 계약이 다르다")
    return {"worker_sha256": sha(output), "checkpoint_sha256": checkpoint["sha256"],
            "checkpoint_file": target.name, "seed": 3221, "iterations": 300,
            "manifest_sha256": sha(manifest), "arm": "NONE"}


class AppProxy:
    def __init__(self, app: Any, settings: Any, output: Path, terminal: Path, old: set[str]) -> None:
        self._app, self._settings = app, settings
        self._output, self._terminal, self._old = output, terminal, old
        if settings.get_as_bool(SETTING) is not False:
            raise TerminalEvidenceError("startup fastShutdown이 false가 아니다")
        marker(terminal, "app-startup", startup_fast_shutdown=False)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._app, name)

    def close(self) -> Any:
        policy = None
        try:
            evidence = training_evidence(self._output, self._old)
            ready = marker(self._terminal, "training-result-ready", evidence=evidence)
            if self._settings.get_as_bool(SETTING) is not False:
                raise TerminalEvidenceError("terminal 전 fastShutdown이 false가 아니다")
            self._settings.set_bool(SETTING, True)
            if self._settings.get_as_bool(SETTING) is not True:
                raise TerminalEvidenceError("fastShutdown readback 실패")
            policy = marker(self._terminal, "shutdown-policy", before=False, after=True,
                            wait_for_replicator=False, evidence_sha256=sha(ready))
        except BaseException as error:  # noqa: BLE001
            marker(self._terminal, "shutdown-refusal", error_type=type(error).__name__, reason=str(error))
        marker(self._terminal, "sdk-close-entered", policy_sha256=None if policy is None else sha(policy))
        result = self._app.close(wait_for_replicator=False)
        marker(self._terminal, "sdk-close-returned")
        return result


def verify(folder: Path, output: Path, child_exit: int, old: set[str]) -> dict[str, Any]:
    if child_exit != 0:
        raise TerminalEvidenceError(f"child exit={child_exit}")
    paths = sorted(folder.glob("*.json"))
    values = [read(path) for path in paths]
    events = [value.get("event") for value in values]
    required = ["app-startup", "training-result-ready", "shutdown-policy", "sdk-close-entered"]
    if events[:4] != required or events[4:] not in ([], ["sdk-close-returned"]):
        raise TerminalEvidenceError(f"terminal 사건이 다르다: {events}")
    if len({value.get("pid") for value in values}) != 1 or any(
            left["sequence"] >= right["sequence"] or left["monotonic_ns"] > right["monotonic_ns"]
            for left, right in itertools.pairwise(values)):
        raise TerminalEvidenceError("terminal PID/순서가 다르다")
    evidence = training_evidence(output, old)
    if (values[1].get("evidence") != evidence
            or values[3].get("policy_sha256") != sha(paths[2])
            or values[2].get("evidence_sha256") != sha(paths[1])
            or values[2].get("before") is not False or values[2].get("after") is not True
            or values[2].get("wait_for_replicator") is not False):
        raise TerminalEvidenceError("terminal evidence/policy 결합이 다르다")
    return {**evidence, "natural_exit_before_close_return": len(values) == 4}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--terminal", required=True, type=Path)
    parser.add_argument("--old-checkpoint-sha256", required=True, action="append")
    args = parser.parse_args()
    import brev_recovery_worker_v3 as worker
    import carb.settings
    import isaaclab.app as app_module

    original = app_module.AppLauncher

    class BoundLauncher:
        add_app_launcher_args = staticmethod(original.add_app_launcher_args)

        def __init__(self, *values: Any, **kwargs: Any) -> None:
            kwargs["fast_shutdown"] = False
            self._launcher = original(*values, **kwargs)
            self.app = AppProxy(self._launcher.app, carb.settings.get_settings(),
                                args.output.resolve(), args.terminal.resolve(),
                                set(args.old_checkpoint_sha256))

        def __getattr__(self, name: str) -> Any:
            return getattr(self._launcher, name)

    app_module.AppLauncher = BoundLauncher
    try:
        return worker.main(["--job", str(args.job.resolve()), "--output", str(args.output.resolve())])
    finally:
        app_module.AppLauncher = original


if __name__ == "__main__":
    raise SystemExit(main())
