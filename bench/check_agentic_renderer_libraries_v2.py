"""Iray·USD MaterialX 공유 라이브러리 의존성을 앱/GPU 초기화 없이 확인한다."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import check_vision_renderer_libraries_v1 as BASE

SCHEMA = "agentic_renderer_library_check_v2"
LIBRARY_GLOBS = (
    BASE.LIBRARY_GLOB,
    "extscache/omni.usd.libs-*/bin/libMaterialXRenderHw.so",
    "extscache/omni.usd.libs-*/bin/libMaterialXRenderGlsl.so",
    "extscache/omni.usd.libs-*/bin/libusdBakeMtlx.so",
)
SOURCE_FILES = frozenset({
    "bench/check_agentic_renderer_libraries_v2.py",
    "tests/test_check_agentic_renderer_libraries_v2.py",
    "bench/check_vision_renderer_libraries_v1.py",
})


def inspect_libraries(
    isaacsim_root: Path, *, runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    root = Path(isaacsim_root).resolve()
    entries: list[dict[str, Any]] = []
    for pattern in LIBRARY_GLOBS:
        matches = sorted(path for path in root.glob(pattern) if path.is_file())
        row: dict[str, Any] = {"glob": pattern, "matches": [str(p) for p in matches], "library": None,
                               "ldd": None, "reasons": []}
        entries.append(row)
        if len(matches) != 1:
            row["reasons"].append("library_missing" if not matches else "library_count_not_one")
            continue
        path = matches[0].resolve()
        if not path.is_relative_to(root):
            row["reasons"].append("library_outside_package")
            continue
        row["library"] = {"path": str(path), "sha256_raw": hashlib.sha256(path.read_bytes()).hexdigest()}
        argv = ["ldd", "--", str(path)]
        detail: dict[str, Any] = {"argv": argv, "timeout_s": BASE.TIMEOUT_S, "exit_code": None,
                                  "stdout": "", "stderr": "", "timed_out": False}
        row["ldd"] = detail
        try:
            result = runner(argv, check=False, capture_output=True, text=True, encoding="utf-8",
                            errors="replace", timeout=BASE.TIMEOUT_S, shell=False)
            detail.update(exit_code=result.returncode, stdout=BASE._text(result.stdout), stderr=BASE._text(result.stderr))
            if result.returncode != 0:
                row["reasons"].append("ldd_nonzero")
            if "not found" in detail["stdout"] or "not found" in detail["stderr"]:
                row["reasons"].append("dependency_not_found")
        except subprocess.TimeoutExpired as error:
            detail.update(stdout=BASE._text(error.stdout), stderr=BASE._text(error.stderr), timed_out=True)
            row["reasons"].append("ldd_timeout")
        except OSError as error:
            detail.update(error_type=type(error).__name__, error_message=str(error))
            row["reasons"].append("ldd_os_error")
    return {"schema": SCHEMA, "status": "not_ready" if any(e["reasons"] for e in entries) else "ready",
            "scope": BASE.SCOPE, "model_use_ready": False, "renderer_initialized": False,
            "queried_at_utc": BASE._utc_stamp(), "isaacsim_root": str(root), "libraries": entries,
            "reasons": [{"glob": e["glob"], "reasons": e["reasons"]} for e in entries if e["reasons"]]}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isaacsim-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("기존 의존성 보고서는 덮어쓰지 않는다")
    report = inspect_libraries(args.isaacsim_root)
    BASE.write_report(args.output, report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["status"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
