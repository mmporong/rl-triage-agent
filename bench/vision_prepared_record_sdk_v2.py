"""V1 prepared record 앞에 ffmpeg/ffprobe provenance 검사를 추가한다."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import traceback
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import vision_prepared_record_sdk_v1 as V1

PREFLIGHT_SCHEMA = "vision_prepared_record_runtime_preflight_v2"
PRIMARY_ERROR_SCHEMA = "vision_prepared_record_primary_error_v2"
VERSION_TIMEOUT_S = 5.0
TOOLS = ("ffmpeg", "ffprobe")
_V2_KEYS = frozenset({"resolve_executable", "run_tool_version"})
_DEPENDENCY_KEYS = V1._DEPENDENCY_KEYS | _V2_KEYS

PreparedRecordSDKFailure = V1.PreparedRecordSDKFailure


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_new(path: Path, value: Mapping[str, Any]) -> None:
    raw = _canonical(dict(value))
    parent = path.parent.resolve(strict=True)
    if parent != path.parent.absolute() or path.is_symlink() or path.exists():
        raise FileExistsError(f"prepared record V2 sidecar 경계가 다르다: {path.name}")
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _type_name(error: BaseException) -> str:
    kind = type(error)
    return f"{kind.__module__}.{kind.__qualname__}"


def _exception_tree(
    error: BaseException,
    *,
    seen: set[int] | None = None,
    depth: int = 0,
) -> dict[str, Any]:
    active = set() if seen is None else seen
    if id(error) in active or depth > 8:
        return {"type": _type_name(error), "message": str(error), "cycle_or_depth_limit": True}
    active.add(id(error))
    frames = [
        {
            "file": frame.filename,
            "line": frame.lineno,
            "function": frame.name,
            "source": frame.line,
        }
        for frame in traceback.extract_tb(error.__traceback__)
    ]
    cause = error.__cause__
    context = error.__context__ if cause is None and not error.__suppress_context__ else None
    return {
        "type": _type_name(error),
        "message": str(error),
        "traceback_frames": frames,
        "cause": (
            _exception_tree(cause, seen=active, depth=depth + 1)
            if cause is not None
            else None
        ),
        "context": (
            _exception_tree(context, seen=active, depth=depth + 1)
            if context is not None
            else None
        ),
    }


def _tool_row(
    name: str,
    resolve: Callable[[str], str | None],
    run_version: Callable[..., subprocess.CompletedProcess[str]],
) -> tuple[dict[str, Any], Exception | None]:
    try:
        requested = resolve(name)
    except Exception as error:  # noqa: BLE001 - locator 원오류를 sidecar에 보존한다.
        return {
            "name": name,
            "status": "resolve_failed",
            "requested_path": None,
            "error_type": _type_name(error),
        }, error
    if requested is None:
        error = FileNotFoundError(f"required executable not found: {name}")
        return {"name": name, "status": "missing", "requested_path": None}, error
    requested_path = Path(requested)
    try:
        resolved = requested_path.resolve(strict=True)
    except OSError as error:
        return {
            "name": name,
            "status": "invalid",
            "requested_path": str(requested_path),
            "error_type": _type_name(error),
        }, error
    if (
        not resolved.is_file()
        or resolved.stat().st_nlink != 1
        or not os.access(resolved, os.X_OK)
    ):
        error = PermissionError(f"required executable is not executable: {resolved}")
        return {
            "name": name,
            "status": "invalid",
            "requested_path": str(requested_path),
            "resolved_path": str(resolved),
        }, error
    argv = [str(resolved), "-version"]
    try:
        completed = run_version(
            argv,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="strict",
            timeout=VERSION_TIMEOUT_S,
        )
    except Exception as error:  # noqa: BLE001 - 주입 runner의 원래 오류 타입을 보존한다.
        return {
            "name": name,
            "status": "version_failed",
            "requested_path": str(requested_path),
            "resolved_path": str(resolved),
            "argv": argv,
            "error_type": _type_name(error),
        }, error
    stdout = completed.stdout
    stderr = completed.stderr
    first_line = stdout.splitlines()[0] if stdout.splitlines() else ""
    expected_prefix = f"{name} version "
    row = {
        "name": name,
        "status": "ready" if completed.returncode == 0 and first_line.startswith(expected_prefix) else "invalid",
        "requested_path": str(requested_path),
        "resolved_path": str(resolved),
        "sha256": _sha(resolved),
        "bytes": resolved.stat().st_size,
        "argv": argv,
        "returncode": completed.returncode,
        "version_first_line": first_line,
        "stdout_sha256": hashlib.sha256(stdout.encode("utf-8")).hexdigest(),
        "stderr_sha256": hashlib.sha256(stderr.encode("utf-8")).hexdigest(),
        "timeout_s": VERSION_TIMEOUT_S,
    }
    if row["status"] != "ready":
        return row, RuntimeError(f"{name} executable version check failed")
    return row, None


def _preflight(
    resolve: Callable[[str], str | None],
    run_version: Callable[..., subprocess.CompletedProcess[str]],
) -> tuple[dict[str, Any], Exception | None]:
    rows: list[dict[str, Any]] = []
    primary: Exception | None = None
    for name in TOOLS:
        row, error = _tool_row(name, resolve, run_version)
        rows.append(row)
        if error is not None:
            primary = error
            break
    return {
        "schema": PREFLIGHT_SCHEMA,
        "status": "complete" if primary is None else "failed",
        "tools": rows,
        "required_tools": list(TOOLS),
        "version_timeout_s": VERSION_TIMEOUT_S,
    }, primary


def _default_dependencies() -> dict[str, Any]:
    return {
        **V1._load_dependencies(),
        "resolve_executable": shutil.which,
        "run_tool_version": subprocess.run,
    }


def run_prepared_record(
    validated: Mapping[str, Any],
    app: Any,
    *,
    seal_report: Callable[[Mapping[str, Any]], None],
    dependencies: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """동결 V1 envelope를 유지하며 encoder 도구와 원오류를 sidecar로 봉인한다."""
    deps = dict(dependencies) if dependencies is not None else _default_dependencies()
    if set(deps) != _DEPENDENCY_KEYS:
        raise ValueError("prepared record SDK V2 dependency 표면이 다르다")
    job_path = validated.get("job_path")
    output = validated.get("output_directory")
    if not isinstance(job_path, Path) or not isinstance(output, Path):
        raise TypeError("prepared record SDK V2 job/output 경로가 Path가 아니다")
    preflight_path = job_path.parent / "record-runtime-preflight.json"
    primary_path = job_path.parent / "primary-error.json"
    original_run_real = deps["run_real"]

    def save_primary(error: Exception, phase: str) -> None:
        value = {
            "schema": PRIMARY_ERROR_SCHEMA,
            "phase": phase,
            "exception": _exception_tree(error),
            "preflight": {
                "path": preflight_path.name,
                "sha256": _sha(preflight_path),
            },
            "output_absent": not output.exists(),
        }
        try:
            _write_new(primary_path, value)
        except Exception as sidecar_error:  # noqa: BLE001 - 원오류를 가리지 않는다.
            traceback.print_exception(sidecar_error, file=sys.stderr)

    def guarded_run_real(
        selected: Mapping[str, Any],
        selected_app: Any,
        *,
        camera_capture_factory: Any,
    ) -> Any:
        receipt, preflight_error = _preflight(
            deps["resolve_executable"], deps["run_tool_version"]
        )
        _write_new(preflight_path, receipt)
        if preflight_error is not None:
            save_primary(preflight_error, "runtime_preflight")
            traceback.print_exception(preflight_error, file=sys.stderr)
            raise preflight_error
        try:
            return original_run_real(
                selected,
                selected_app,
                camera_capture_factory=camera_capture_factory,
            )
        except Exception as error:
            save_primary(error, "run_real")
            traceback.print_exception(error, file=sys.stderr)
            raise

    v1_dependencies = {key: deps[key] for key in V1._DEPENDENCY_KEYS}
    v1_dependencies["run_real"] = guarded_run_real
    return V1.run_prepared_record(
        validated,
        app,
        seal_report=seal_report,
        dependencies=v1_dependencies,
    )


__all__: Sequence[str] = ("PreparedRecordSDKFailure", "run_prepared_record")
