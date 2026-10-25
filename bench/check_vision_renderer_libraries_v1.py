"""Isaac Sim Iray 공유 라이브러리 의존성을 GPU 없이 점검한다."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = "vision_renderer_library_check_v1"
SCOPE = "shared_library_dependencies_only_not_renderer_validation"
LIBRARY_GLOB = "extscache/omni.iray.libs-*/bin/iray/libneuray.so"
TIMEOUT_S = 30


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")  # noqa: UP017 - Python 3.10


def _text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def inspect_libraries(
    isaacsim_root: Path,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    root = Path(isaacsim_root).resolve()
    matches = sorted(path.resolve() for path in root.glob(LIBRARY_GLOB) if path.is_file())
    reasons: list[str] = []
    library: dict[str, Any] | None = None
    ldd: dict[str, Any] = {
        "argv": None,
        "timeout_s": TIMEOUT_S,
        "exit_code": None,
        "stdout": "",
        "stderr": "",
        "timed_out": False,
    }

    if len(matches) != 1:
        reasons.append("library_missing" if not matches else "library_count_not_one")
    else:
        path = matches[0]
        library = {
            "path": str(path),
            "sha256_raw": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        argv = ["ldd", "--", str(path)]
        ldd["argv"] = argv
        try:
            completed = runner(
                argv,
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=TIMEOUT_S,
                shell=False,
            )
            ldd.update({
                "exit_code": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            })
            if completed.returncode != 0:
                reasons.append("ldd_nonzero")
            if "not found" in completed.stdout:
                reasons.append("dependency_not_found")
        except subprocess.TimeoutExpired as error:
            ldd.update({
                "stdout": _text(error.stdout),
                "stderr": _text(error.stderr),
                "timed_out": True,
            })
            reasons.append("ldd_timeout")
        except OSError as error:
            ldd.update({
                "error_type": type(error).__name__,
                "error_message": str(error),
            })
            reasons.append("ldd_os_error")

    return {
        "schema": SCHEMA,
        "status": "ready" if not reasons else "not_ready",
        "scope": SCOPE,
        "model_use_ready": False,
        "queried_at_utc": _utc_stamp(),
        "isaacsim_root": str(root),
        "discovery": {
            "glob": LIBRARY_GLOB,
            "match_count": len(matches),
            "matches": [str(path) for path in matches],
        },
        "library": library,
        "ldd": ldd,
        "reasons": reasons,
    }


def write_report(path: Path, report: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--isaacsim-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    report = inspect_libraries(args.isaacsim_root)
    write_report(args.output, report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["status"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
