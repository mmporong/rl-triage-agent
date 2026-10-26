from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / "bench", ROOT / "evals", ROOT / "src", ROOT / "tests"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

M = importlib.import_module("vision_prepared_record_sdk_v2")
V1 = importlib.import_module("vision_prepared_record_sdk_v1")
F = importlib.import_module("test_vision_prepared_record_sdk_v1")


def _tools(tmp_path: Path) -> dict[str, Path]:
    result = {}
    for name in M.TOOLS:
        path = tmp_path / "tools" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((name + " fixture executable\n").encode())
        path.chmod(0o755)
        result[name] = path
    return result


def _version_runner(
    argv: list[str], **kwargs: Any
) -> subprocess.CompletedProcess[str]:
    assert kwargs == {
        "check": False,
        "capture_output": True,
        "text": True,
        "encoding": "utf-8",
        "errors": "strict",
        "timeout": M.VERSION_TIMEOUT_S,
    }
    name = Path(argv[0]).name
    assert argv == [argv[0], "-version"]
    return subprocess.CompletedProcess(argv, 0, f"{name} version fixture\n", "")


def _dependencies(tmp_path: Path, log: list[str]) -> dict[str, Any]:
    tools = _tools(tmp_path)
    return {
        **F._dependencies(tmp_path, log),
        "resolve_executable": lambda name: str(tools[name]),
        "run_tool_version": _version_runner,
    }


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_missing_ffmpeg_is_preserved_before_run_real_and_output_creation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    log: list[str] = []
    validated = F._validated(tmp_path)
    deps = _dependencies(tmp_path, log)
    ffprobe = deps["resolve_executable"]("ffprobe")
    deps["resolve_executable"] = lambda name: None if name == "ffmpeg" else ffprobe
    app = F._App(log)
    with pytest.raises(V1.PreparedRecordSDKFailure) as caught:
        M.run_prepared_record(
            validated,
            app,
            seal_report=lambda _report: None,
            dependencies=deps,
        )
    assert isinstance(caught.value.primary, FileNotFoundError)
    assert str(caught.value.primary) == "required executable not found: ffmpeg"
    assert "run_real" not in log
    assert not validated["output_directory"].exists()
    assert app.close_calls == 0
    preflight = _read(tmp_path / "record-runtime-preflight.json")
    assert preflight["status"] == "failed"
    assert preflight["tools"] == [
        {"name": "ffmpeg", "status": "missing", "requested_path": None}
    ]
    primary = _read(tmp_path / "primary-error.json")
    assert primary["phase"] == "runtime_preflight"
    assert primary["exception"]["message"] == str(caught.value.primary)
    assert primary["output_absent"] is True
    assert str(caught.value.primary) in capsys.readouterr().err


def test_original_run_real_file_not_found_keeps_message_and_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    log: list[str] = []
    validated = F._validated(tmp_path)
    deps = _dependencies(tmp_path, log)

    def missing_encoder(*_args: Any, **_kwargs: Any) -> None:
        try:
            raise RuntimeError("encoder process lookup failed")
        except RuntimeError as cause:
            raise FileNotFoundError(
                "[Errno 2] No such file or directory: 'ffmpeg'"
            ) from cause

    deps["run_real"] = missing_encoder
    app = F._App(log)
    with pytest.raises(V1.PreparedRecordSDKFailure) as caught:
        M.run_prepared_record(
            validated,
            app,
            seal_report=lambda _report: None,
            dependencies=deps,
        )
    assert isinstance(caught.value.primary, FileNotFoundError)
    assert str(caught.value.primary) == "[Errno 2] No such file or directory: 'ffmpeg'"
    assert not validated["output_directory"].exists()
    assert app.close_calls == 0
    assert _read(tmp_path / "record-runtime-preflight.json")["status"] == "complete"
    primary = _read(tmp_path / "primary-error.json")
    assert primary["phase"] == "run_real"
    frames = primary["exception"]["traceback_frames"]
    assert any(row["function"] == "missing_encoder" for row in frames)
    assert primary["exception"]["cause"]["message"] == "encoder process lookup failed"
    assert primary["exception"]["context"] is None
    assert str(caught.value.primary) in capsys.readouterr().err


def test_success_preserves_v1_envelope_and_writes_exact_tool_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log: list[str] = []
    validated = F._validated(tmp_path)
    deps = _dependencies(tmp_path, log)
    monkeypatch.setattr(V1.R, "WIDTH", F._Rep.width)
    monkeypatch.setattr(V1.R, "HEIGHT", F._Rep.height)
    sealed: list[dict[str, Any]] = []
    result = M.run_prepared_record(
        validated,
        F._App(log),
        seal_report=lambda report: sealed.append(dict(report)),
        dependencies=deps,
    )
    assert result["schema"] == V1.SDK_REPORT_SCHEMA
    assert result["approved"] is True
    assert set(result) == set(sealed[0])
    preflight = _read(tmp_path / "record-runtime-preflight.json")
    assert preflight["status"] == "complete"
    assert [row["name"] for row in preflight["tools"]] == list(M.TOOLS)
    assert all(row["status"] == "ready" for row in preflight["tools"])
    assert all(Path(row["resolved_path"]).is_absolute() for row in preflight["tools"])
    assert not (tmp_path / "primary-error.json").exists()
