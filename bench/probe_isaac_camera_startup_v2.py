"""Camera startup/update terminal을 봉인한 뒤 등록된 fast-shutdown 정책으로 SDK를 닫는다."""
from __future__ import annotations

import argparse
import importlib
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Mapping

import probe_isaac_camera_startup_v1 as V1
import record_vision_evidence_v1 as R
import vision_shutdown_policy_v1 as POLICY

SCHEMA = "agentic_isaac_camera_startup_v2"
SOURCE_FILES = frozenset({"bench/probe_isaac_camera_startup_v2.py", "tests/test_isaac_camera_startup_v2.py"})
REQUIRED_MARKERS = ("startup-entered", "startup-returned", "update-entered", "update-returned",
                    "terminal-sealed", "close-entered")
OPTIONAL_MARKER = "close-returned"


def validate_inputs(*args: Any, **kwargs: Any) -> dict[str, Any]:
    value = V1.validate_inputs(*args, **kwargs)
    protocol = json.loads(Path(args[0]).read_text(encoding="utf-8"))
    sources = protocol.get("source_sha256_lf", {})
    V1.require(SOURCE_FILES <= sources.keys(), "camera startup v2 source closure missing")
    repo = Path(kwargs.get("repo", V1.ROOT))
    for relative in SOURCE_FILES:
        V1.require(sources[relative] == V1.sha(repo / relative, lf=True), "camera startup v2 source blob changed")
    return value


class Markers:
    def __init__(self, output: Path):
        self.output, self.sequence = output, 0

    def write(self, event: str, **fields: Any) -> None:
        V1.require(event in REQUIRED_MARKERS or event == OPTIONAL_MARKER, "known startup v2 marker")
        self.sequence += 1
        V1.write_new(self.output / "markers" / f"{event}.json", {
            "schema": SCHEMA + "_marker", "event": event, "sequence": self.sequence,
            "pid": os.getpid(), "monotonic_ns": time.monotonic_ns(), **fields,
        })


def run_sdk(output: Path, launcher: Mapping[str, Any], *, app_factory: Callable[..., Any], settings: Any = None,
            settings_factory: Callable[[], Any] | None = None) -> dict[str, Any]:
    V1.require(dict(launcher) == V1.LAUNCHER and output.is_dir(), "camera v2 launcher/output")
    markers = Markers(output)
    markers.write("startup-entered", launcher=dict(launcher))
    app = app_factory(**dict(launcher))
    markers.write("startup-returned")
    if settings is None:
        if settings_factory is None:
            raise V1.CameraStartupError("settings factory required after App creation")
        settings = settings_factory()
    else:
        V1.require(settings_factory is None, "settings and settings factory are mutually exclusive")
    proxy = POLICY.AppProxy(app, {
        "output_directory": str(output), "job_path": str(output / "camera-job.json"),
        "job_sha256": R.sha256_file(output / "camera-job.json"),
    }, settings)
    markers.write("update-entered", update_count=1)
    proxy.update()
    markers.write("update-returned", update_count=1)
    terminal = output / "result.json"
    V1.write_new(terminal, {"schema": SCHEMA + "_terminal", "status": "startup_update_returned",
                            "startup_returned": True, "updates_returned": 1,
                            "physics_settings_changed": False, "policy_runs": 0})
    markers.write("terminal-sealed", terminal_sha256=R.sha256_file(terminal))
    markers.write("close-entered", wait_for_replicator=False)
    proxy.close()
    markers.write("close-returned")
    result = {"schema": SCHEMA + "_child_result", "status": "awaiting_parent_exit_receipt",
              "terminal_sha256": R.sha256_file(terminal),
              "shutdown_policy_sha256": R.sha256_file(output / POLICY.SIDECAR),
              "close_returned": True, "success_eligible_without_parent": False}
    V1.write_new(output / "child-result.json", result)
    return result


def execute(protocol: Path, proof: Path, proof_sha: str, output: Path, *, repo: Path = V1.ROOT,
            app_factory: Callable[..., Any] | None = None, settings: Any = None,
            validate_kwargs: dict[str, Any] | None = None) -> dict[str, Any]:
    output = Path(output).absolute()
    V1.require(not output.exists() and output.parent.is_dir(), "new output directory required")
    validated = validate_inputs(protocol, proof, proof_sha, repo=repo, **(validate_kwargs or {}))
    output.mkdir()
    V1.write_new(output / "validated-inputs.json", validated)
    V1.write_new(output / "camera-job.json", {"protocol_sha256_lf": validated["protocol_sha256_lf"],
                                                "resource_proof_sha256": validated["resource_proof_sha256"]})
    if app_factory is None:
        from isaaclab.app import AppLauncher  # pyright: ignore[reportMissingImports]
        def actual_app_factory(**options: Any) -> Any:
            return AppLauncher(**options).app
        app_factory = actual_app_factory
    settings_factory = None
    if settings is None:
        def actual_settings_factory() -> Any:
            return importlib.import_module("carb.settings").get_settings()
        settings_factory = actual_settings_factory
    return run_sdk(output, validated["launcher"], app_factory=app_factory, settings=settings,
                   settings_factory=settings_factory)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--resource-proof", type=Path, required=True)
    parser.add_argument("--resource-proof-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=V1.ROOT)
    args = parser.parse_args()
    print(json.dumps(execute(args.protocol, args.resource_proof, args.resource_proof_sha256,
                             args.output, repo=args.repo), sort_keys=True))


if __name__ == "__main__":
    main()
