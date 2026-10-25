from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
C = importlib.import_module("check_vision_renderer_libraries_v1")


def _library(root: Path, version: str = "1.0.0") -> Path:
    path = root / "extscache" / f"omni.iray.libs-{version}" / "bin" / "iray" / "libneuray.so"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"fixture-neuray")
    return path.resolve()


def test_ready_records_exact_ldd_invocation_and_library_digest(tmp_path: Path) -> None:
    library = _library(tmp_path)
    calls: list[tuple[list[str], dict[str, object]]] = []

    def runner(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, "libGLU.so.1 => /usr/lib/libGLU.so.1\n", "")

    report = C.inspect_libraries(tmp_path, runner=runner)

    assert report["status"] == "ready" and report["model_use_ready"] is False
    assert report["scope"] == C.SCOPE and report["reasons"] == []
    assert report["library"]["path"] == str(library)
    assert len(report["library"]["sha256_raw"]) == 64
    assert calls == [(["ldd", "--", str(library)], {
        "check": False, "capture_output": True, "text": True, "encoding": "utf-8",
        "errors": "replace", "timeout": 30, "shell": False,
    })]


@pytest.mark.parametrize("count", [0, 2])
def test_missing_or_multiple_libraries_are_not_ready_without_ldd(tmp_path: Path, count: int) -> None:
    for index in range(count):
        _library(tmp_path, str(index))

    def forbidden(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise AssertionError((args, kwargs))

    report = C.inspect_libraries(tmp_path, runner=forbidden)

    assert report["status"] == "not_ready"
    assert report["discovery"]["match_count"] == count
    assert report["reasons"] == (["library_missing"] if count == 0 else ["library_count_not_one"])
    assert report["ldd"]["argv"] is None


@pytest.mark.parametrize(
    ("returncode", "stdout", "reason"),
    [(0, "libGLU.so.1 => not found\n", "dependency_not_found"), (1, "", "ldd_nonzero")],
)
def test_ldd_failure_is_not_ready(
    tmp_path: Path, returncode: int, stdout: str, reason: str,
) -> None:
    _library(tmp_path)

    def runner(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(argv, returncode, stdout, "diagnostic")

    report = C.inspect_libraries(tmp_path, runner=runner)

    assert report["status"] == "not_ready" and reason in report["reasons"]
    assert report["ldd"]["exit_code"] == returncode
    assert report["ldd"]["stdout"] == stdout and report["ldd"]["stderr"] == "diagnostic"


def test_ldd_timeout_is_not_ready_and_preserves_partial_output(tmp_path: Path) -> None:
    _library(tmp_path)

    def runner(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(argv, 30, output=b"partial\xff", stderr=b"late\xff")

    report = C.inspect_libraries(tmp_path, runner=runner)

    assert report["status"] == "not_ready" and report["reasons"] == ["ldd_timeout"]
    assert report["ldd"]["timed_out"] is True and report["ldd"]["exit_code"] is None
    assert "partial" in report["ldd"]["stdout"] and "late" in report["ldd"]["stderr"]


def test_ldd_os_error_is_recorded_as_not_ready(tmp_path: Path) -> None:
    _library(tmp_path)

    def runner(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise PermissionError(13, "permission denied", argv[0])

    report = C.inspect_libraries(tmp_path, runner=runner)

    assert report["status"] == "not_ready" and report["reasons"] == ["ldd_os_error"]
    assert report["ldd"]["error_type"] == "PermissionError"
    assert "permission denied" in report["ldd"]["error_message"]
    assert report["ldd"]["exit_code"] is None and report["ldd"]["timed_out"] is False


def test_cli_refuses_to_overwrite_output_before_inspection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "report.json"
    output.write_text('{"preserve":true}\n', encoding="utf-8")
    monkeypatch.setattr(C, "inspect_libraries", lambda root: pytest.fail(str(root)))

    with pytest.raises(FileExistsError):
        C.main(["--isaacsim-root", str(tmp_path), "--output", str(output)])

    assert json.loads(output.read_text(encoding="utf-8")) == {"preserve": True}
