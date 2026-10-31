from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "bench"), str(ROOT / "evals"), str(ROOT / "src")]
T = importlib.import_module("lineage_recovery_execution_v2")


class Input:
    def write(self, _value):
        return None

    def flush(self):
        return None

    def close(self):
        return None


class Process:
    def __init__(self):
        self.stdin = Input()
        self.pid = 1234

    def poll(self):
        return 0

    def wait(self, timeout=None):
        return 0

    def terminate(self):
        return None


class Job:
    def assign(self, _process):
        return None

    def active_processes(self):
        return 0

    def terminate(self, _code):
        return None

    def wait_empty(self, _timeout):
        return True

    def close(self):
        return None


def actual_owned_runner(folder: Path):
    with patch.object(T.OWNED.M, "WindowsJob", Job), \
            patch.object(T.OWNED.subprocess, "Popen", return_value=Process()), \
            patch.object(T.OWNED, "job_identities", return_value={}):
        return T.OWNED.run_owned(folder, ["python", "worker.py"], folder.parent, 10)


def test_v1_baseline_reproduces_missing_owned_folder_before_spawn(tmp_path):
    folder = tmp_path / "training/owned"
    with patch.object(T.OWNED.M, "WindowsJob", Job), \
            patch.object(T.OWNED.subprocess, "Popen", return_value=Process()), \
            patch.object(T.OWNED, "job_identities", return_value={}), \
            pytest.raises(FileNotFoundError, match="launch.json"):
        T.OWNED.run_owned(folder, ["python", "worker.py"], tmp_path, 10)
    assert not folder.exists()


@pytest.mark.parametrize("role", ["training", "evaluation"])
def test_adapter_creates_fresh_owned_folder_then_actual_run_owned_writes_launch(tmp_path, role):
    folder = tmp_path / role / "owned"
    wrapped = T._owned_runner(lambda target, _args, _cwd, _watchdog: actual_owned_runner(target))
    result = wrapped(folder, ["ignored"], tmp_path, 10)
    assert result["exit_code"] == 0 and result["natural_exit"] is True
    assert result["cleanup_verified"] is True and result["job_assigned_before_release"] is True
    assert json.loads((folder / "launch.json").read_text(encoding="utf-8"))["args"] == [
        "python", "worker.py"
    ]
    assert (folder / "stdout.log").is_file() and (folder / "stderr.log").is_file()
    with pytest.raises(FileExistsError):
        wrapped(folder, ["ignored"], tmp_path, 10)


def test_protocol_uses_v2_schema_and_exact_extended_source_closure(tmp_path, monkeypatch):
    batch = tmp_path / "batch.json"
    batch.write_text(json.dumps({
        "schema": T.V1.BATCH.SCHEMA,
        "runtime_source_sha256_lf": {f"runtime/{index}.py": "a" * 64 for index in range(9)},
    }), encoding="utf-8")
    monkeypatch.setattr(T.V1, "_relative", lambda path: (
        "batch.json" if Path(path).resolve() == batch.resolve()
        else Path(path).resolve().relative_to(ROOT).as_posix()
    ))
    before = (T.V1.SCHEMA, T.V1.SOURCE_FILES)
    value = T.prepare_protocol(batch)
    assert value["schema"] == T.SCHEMA
    assert set(value["source_sha256_lf"]) == set(T.SOURCE_FILES)
    assert set(T.SOURCE_FILES) == set(T.V1.SOURCE_FILES) | {
        "evals/lineage_recovery_execution_v2.py",
        "tests/test_lineage_recovery_execution_v2.py",
    }
    assert (T.V1.SCHEMA, T.V1.SOURCE_FILES) == before


def test_verify_and_execution_scopes_restore_v1_after_success_and_failure(tmp_path):
    before = (T.V1.SCHEMA, T.V1.SOURCE_FILES)
    observed = []

    def verify(*_args, **_kwargs):
        observed.append((T.V1.SCHEMA, T.V1.SOURCE_FILES))
        return {"ok": True}

    with patch.object(T.V1, "verify_protocol", side_effect=verify):
        assert T.verify_protocol(tmp_path / "protocol.json", anchor="a" * 40, lab=tmp_path) == {"ok": True}
    assert observed == [(T.SCHEMA, T.SOURCE_FILES)]
    assert (T.V1.SCHEMA, T.V1.SOURCE_FILES) == before

    def fail(*_args, **_kwargs):
        assert T.V1.SCHEMA == T.SCHEMA and T.V1.SOURCE_FILES == T.SOURCE_FILES
        raise RuntimeError("stop")

    with patch.object(T.V1, "smoke", side_effect=fail), pytest.raises(RuntimeError, match="stop"):
        T.smoke(tmp_path / "manifest.json", tmp_path / "out")
    assert (T.V1.SCHEMA, T.V1.SOURCE_FILES) == before


def test_scope_is_same_thread_reentrant_and_rejects_external_binding():
    before = (T.V1.SCHEMA, T.V1.SOURCE_FILES)
    with T._v1_scope():
        assert (T.V1.SCHEMA, T.V1.SOURCE_FILES) == (T.SCHEMA, T.SOURCE_FILES)
        with T._v1_scope():
            assert (T.V1.SCHEMA, T.V1.SOURCE_FILES) == (T.SCHEMA, T.SOURCE_FILES)
    assert (T.V1.SCHEMA, T.V1.SOURCE_FILES) == before
    with patch.object(T.V1, "SCHEMA", "other"), pytest.raises(RuntimeError, match="등록 원본"), \
            T._v1_scope():
        pass


def test_cli_rejects_unregistered_command():
    with pytest.raises(SystemExit):
        T.main(["other"])
