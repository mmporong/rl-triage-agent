"""Iray만 통과하는 libXt 누락을 새 CPU 의존성 검사에서 거부한다."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
import check_agentic_renderer_libraries_v2 as C


def package(tmp_path):
    root = tmp_path / "isaacsim"
    for pattern in C.LIBRARY_GLOBS:
        path = root / pattern.replace("*", "fixture")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic library fixture")
    return root


def clean(argv, **kwargs):
    assert argv[:2] == ["ldd", "--"] and kwargs["shell"] is False
    assert kwargs["encoding"] == "utf-8" and kwargs["timeout"] == 30
    return subprocess.CompletedProcess(argv, 0, "libc.so.6 => /lib/libc.so.6\n", "")


def test_all_four_dependencies_ready_creates_no_renderer_or_model_approval(tmp_path):
    result = C.inspect_libraries(package(tmp_path), runner=clean)
    assert result["status"] == "ready" and len(result["libraries"]) == 4
    assert not result["model_use_ready"] and not result["renderer_initialized"]


def test_iray_ready_does_not_hide_materialx_missing_libxt(tmp_path):
    def ldd(argv, **kwargs):
        if "libMaterialX" in argv[-1]:
            return subprocess.CompletedProcess(argv, 0, "libXt.so.6 => not found\n", "")
        return clean(argv, **kwargs)
    result = C.inspect_libraries(package(tmp_path), runner=ldd)
    assert result["status"] == "not_ready" and len(result["reasons"]) == 2
    assert result["libraries"][0]["reasons"] == []
    assert all(row["reasons"] == ["dependency_not_found"] for row in result["reasons"])


@pytest.mark.parametrize("pattern", C.LIBRARY_GLOBS)
def test_missing_registered_library_is_not_ready(tmp_path, pattern):
    root = package(tmp_path)
    next(root.glob(pattern)).unlink()
    result = C.inspect_libraries(root, runner=clean)
    assert result["status"] == "not_ready" and result["reasons"] == [{"glob": pattern, "reasons": ["library_missing"]}]


def test_multiple_package_versions_are_ambiguous(tmp_path):
    root = package(tmp_path)
    extra = root / C.LIBRARY_GLOBS[1].replace("*", "other")
    extra.parent.mkdir(parents=True)
    extra.write_bytes(b"another synthetic library")
    result = C.inspect_libraries(root, runner=clean)
    assert result["status"] == "not_ready" and result["reasons"][0]["reasons"] == ["library_count_not_one"]


@pytest.mark.parametrize("mode", ["nonzero", "stderr_missing", "timeout", "oserror"])
def test_ldd_failures_keep_raw_evidence_and_refuse_ready(tmp_path, mode):
    def fail(argv, **_kwargs):
        if mode == "timeout":
            raise subprocess.TimeoutExpired(argv, 30, output=b"partial", stderr=b"raw timeout")
        if mode == "oserror":
            raise FileNotFoundError("synthetic missing ldd")
        return subprocess.CompletedProcess(argv, 1 if mode == "nonzero" else 0, "", "libXt.so.6 => not found")
    result = C.inspect_libraries(package(tmp_path), runner=fail)
    assert result["status"] == "not_ready" and len(result["reasons"]) == 4
    if mode == "timeout":
        assert result["libraries"][0]["ldd"]["stdout"] == "partial"
    if mode == "stderr_missing":
        assert result["libraries"][0]["reasons"] == ["dependency_not_found"]


def test_report_output_is_exclusive(tmp_path):
    path = tmp_path / "report.json"
    C.BASE.write_report(path, {"synthetic": True})
    with pytest.raises(FileExistsError):
        C.BASE.write_report(path, {"synthetic": False})
    assert json.loads(path.read_text(encoding="utf-8")) == {"synthetic": True}
