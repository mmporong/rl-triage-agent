from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest  # pyright: ignore[reportMissingImports]

ROOT = next(path for path in Path(__file__).resolve().parents if (path / "AGENTS.md").is_file())
sys.path[:0] = [str(ROOT / name) for name in ("bench", "evals", "src")]

import run_agentic_brev_probe_parent_v4 as P  # noqa: E402  # pyright: ignore[reportMissingImports]


@pytest.mark.parametrize("version", ["3.10", "3.12"])
def test_actual_parent_cli_imports_from_unrelated_cwd_with_empty_pythonpath(
    tmp_path: Path, version: str
) -> None:
    uv = shutil.which("uv")
    assert uv is not None
    env = dict(os.environ)
    env["PYTHONPATH"] = ""
    env["UV_OFFLINE"] = "1"
    result = subprocess.run(
        [uv, "run", "--offline", "--isolated", "--no-project", "--python", version,
         "--with", "pyyaml", "--with", "numpy", "python", str(ROOT / P.SOURCE), "--help"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--request" in result.stdout and "--execute" in result.stdout
    assert "ModuleNotFoundError" not in result.stderr


def test_parent4_keeps_registered_worker_and_bounded_popen_guards() -> None:
    assert P.WORKER1.name == "brev_recovery_worker_v1.py"
    assert P.BACKEND.WORKER == P.WORKER2
    P.verify_probe_equivalence()
    expected = ["python", "-B", str(P.WORKER1), "--job", "j", "--output", "o"]
    bounded = P.V2.V1.BoundedPopen(lambda: None, 60, expected)
    with pytest.raises(ValueError, match="unapproved child command"):
        bounded([*expected[:-1], "different"], start_new_session=True)
