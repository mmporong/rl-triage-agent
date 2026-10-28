"""JOB4 입장 뒤 Isaac Lab 카메라 App startup/update/close만 분리해 기록한다."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
import time
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

SCHEMA = "agentic_isaac_camera_startup_v1"
ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = frozenset({
    "bench/probe_isaac_camera_startup_v1.py",
    "tests/test_isaac_camera_startup_v1.py",
})
LAUNCHER = {"headless": True, "enable_cameras": True, "fast_shutdown": False}
MARKERS = (
    "startup-entered", "startup-returned", "update-entered", "update-returned",
    "close-entered", "close-returned", "close-error",
)
FORBIDDEN_IMPORT_PREFIXES = ("isaaclab", "isaacsim", "torch", "carb", "omni")


class CameraStartupError(ValueError):
    pass


def require(condition: Any, reason: str) -> None:
    if not condition:
        raise CameraStartupError(reason)


def sha(path: Path, *, lf: bool = False) -> str:
    raw = path.read_bytes()
    return hashlib.sha256(raw.replace(b"\r\n", b"\n") if lf else raw).hexdigest()


def write_new(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        stream.write("\n")


def _fixture_modules(fixture: Mapping[str, Any] | None) -> tuple[Any, Any]:
    require(fixture is not None and set(fixture) == {"job4", "record"}, "explicit fixture modules required")
    checked = cast(Mapping[str, Any], fixture)
    return checked["job4"], checked["record"]


def _actual_modules() -> tuple[Any, Any]:
    before = set(sys.modules)
    job4 = importlib.import_module("agentic_play_job_v4")
    record = importlib.import_module("record_vision_evidence_v1")
    imported = set(sys.modules) - before
    require(not any(name == prefix or name.startswith(prefix + ".")
                    for name in imported for prefix in FORBIDDEN_IMPORT_PREFIXES),
            "input validation imported SDK module")
    return job4, record


def validate_inputs(
    protocol_path: Path,
    resource_proof_path: Path,
    resource_proof_sha256: str,
    *,
    repo: Path = ROOT,
    now: datetime | None = None,
    allow_fixture: bool = False,
    fixture: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    require(type(allow_fixture) is bool, "fixture flag must be bool")
    if allow_fixture:
        job4, record = _fixture_modules(fixture)
    else:
        require(fixture is None and os.name == "posix", "actual validation requires POSIX and no fixture")
        job4, record = _actual_modules()
    current = now or datetime.now(timezone.utc)  # noqa: UP017 - Python 3.10 target
    protocol_path = Path(protocol_path).absolute()
    proof_path = Path(resource_proof_path).absolute()
    checked = job4.validate_protocol(protocol_path, repo=Path(repo), now=current, require_current=True)
    protocol_sha = sha(protocol_path, lf=True)
    proof = job4.validate_resource_proof(
        proof_path, resource_proof_sha256, protocol=checked, protocol_sha=protocol_sha,
        now=current, require_fresh=True,
    )
    lease = job4.verify_lease(proof["lease"], protocol=checked, protocol_sha=protocol_sha, now=current)
    sources = checked.get("source_sha256_lf")
    require(isinstance(sources, dict) and SOURCE_FILES <= sources.keys(), "camera startup source closure missing")
    for relative in SOURCE_FILES:
        require(sources[relative] == sha(Path(repo) / relative, lf=True), "camera startup source blob changed")
    launcher = record.fixed_profile().get("launcher")
    require(launcher == LAUNCHER, "registered vision launcher changed")
    return {
        "schema": SCHEMA + "_validated_inputs",
        "protocol_sha256_lf": protocol_sha,
        "resource_proof_sha256": resource_proof_sha256,
        "resource_id": checked["_recording_target"]["resource_id"],
        "lease_reference": dict(lease["reference"]),
        "launcher": dict(launcher),
        "authority_issued": False,
    }


class _Markers:
    def __init__(self, output: Path):
        self.output = output
        self.sequence = 0

    def write(self, event: str, **fields: Any) -> None:
        require(event in MARKERS, "known startup marker")
        self.sequence += 1
        write_new(self.output / "markers" / f"{event}.json", {
            "schema": SCHEMA + "_marker", "event": event, "sequence": self.sequence,
            "pid": os.getpid(), "monotonic_ns": time.monotonic_ns(), **fields,
        })


def _run_sdk(
    output: Path,
    launcher: Mapping[str, Any],
    *,
    allow_fixture: bool = False,
    app_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    require(output.is_absolute() and output.is_dir(), "validated output directory required")
    require(dict(launcher) == LAUNCHER, "launcher contract changed")
    if allow_fixture:
        require(app_factory is not None, "fixture app factory required")
    else:
        require(app_factory is None and os.name == "posix", "actual SDK requires POSIX and no factory")
        def actual_app_factory(**options: Any) -> Any:
            from isaaclab.app import AppLauncher  # type: ignore[import-not-found]

            return AppLauncher(**options).app

        app_factory = actual_app_factory
    assert app_factory is not None
    markers = _Markers(output)
    app = None
    stage = "startup"
    failure_stage = stage
    failure: BaseException | None = None
    try:
        markers.write("startup-entered", launcher=dict(launcher))
        app = app_factory(**dict(launcher))
        markers.write("startup-returned")
        stage = "update"
        markers.write("update-entered", update_count=1)
        app.update()
        markers.write("update-returned", update_count=1)
        write_new(output / "preclose-result.json", {
            "schema": SCHEMA + "_preclose_result", "status": "exit_unconfirmed",
            "startup_returned": True, "updates_returned": 1,
            "parent_must_verify_exit0_cleanup": True,
        })
    except BaseException as error:  # noqa: BLE001 - SDK SystemExit must fail closed
        failure = error
        failure_stage = stage
    finally:
        if app is not None:
            stage = "close"
            markers.write("close-entered")
            try:
                app.close()
                markers.write("close-returned")
            except BaseException as close_error:  # noqa: BLE001 - close SystemExit must fail closed
                markers.write("close-error", error_type=type(close_error).__name__, reason=str(close_error))
                if failure is None:
                    failure = close_error
                    failure_stage = "close"
    if failure is not None:
        failure_receipt = {
            "schema": SCHEMA + "_failure", "status": "failed", "stage": failure_stage,
            "error_type": type(failure).__name__, "reason": str(failure), "success_eligible": False,
        }
        if isinstance(failure, SystemExit):
            failure_receipt["exit_code"] = failure.code
        write_new(output / "failure.json", failure_receipt)
        if isinstance(failure, SystemExit):
            raise CameraStartupError(f"SDK raised SystemExit({failure.code!r})") from failure
        raise failure
    result = {
        "schema": SCHEMA + "_child_result", "status": "awaiting_parent_exit_receipt",
        "startup_returned": True, "updates_returned": 1, "close_returned": True,
        "success_eligible_without_parent": False, "parent_must_verify_exit0_cleanup": True,
    }
    write_new(output / "child-result.json", result)
    return result


def execute(
    protocol_path: Path,
    resource_proof_path: Path,
    resource_proof_sha256: str,
    output: Path,
    *,
    repo: Path = ROOT,
    now: datetime | None = None,
    allow_fixture: bool = False,
    fixture: Mapping[str, Any] | None = None,
    app_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    output = Path(output).absolute()
    require(not output.exists() and output.parent.is_dir(), "new output directory required")
    validated = validate_inputs(
        protocol_path, resource_proof_path, resource_proof_sha256,
        repo=repo, now=now, allow_fixture=allow_fixture, fixture=fixture,
    )
    output.mkdir()
    write_new(output / "validated-inputs.json", validated)
    return _run_sdk(output, validated["launcher"], allow_fixture=allow_fixture, app_factory=app_factory)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--resource-proof", type=Path, required=True)
    parser.add_argument("--resource-proof-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=ROOT)
    args = parser.parse_args()
    result = execute(
        args.protocol, args.resource_proof, args.resource_proof_sha256, args.output, repo=args.repo,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
