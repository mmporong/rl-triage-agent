from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
CANDIDATE = HERE.parent
ROOT = next(path for path in HERE.parents if (path / "AGENTS.md").is_file())
OLD = ROOT / "bench/run_agentic_brev_probe_parent_v8.py"
NEW = CANDIDATE / "bench/run_agentic_brev_probe_parent_v9.py"


def run(path: Path, cwd: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    return subprocess.run(
        [sys.executable, "-B", str(path), "--help"], cwd=cwd, env=env,
        text=True, encoding="utf-8", errors="replace", capture_output=True, check=False,
    )


def test_actual_subprocess_bootstraps_before_parent8_import() -> None:
    with tempfile.TemporaryDirectory() as raw:
        cwd = Path(raw)
        old, new = run(OLD, cwd), run(NEW, cwd)
    assert old.returncode != 0
    assert "ModuleNotFoundError" in old.stderr
    assert "agentic_investigation_contract_v2" in old.stderr
    assert new.returncode == 0, new.stderr
    assert "--request" in new.stdout


def test_parent9_declares_its_public_source_closure() -> None:
    sys.path[:0] = [str(CANDIDATE / "bench"), str(ROOT / "evals"),
                    str(ROOT / "bench"), str(ROOT / "src")]
    import run_agentic_brev_probe_parent_v9 as parent  # pyright: ignore[reportMissingImports]

    assert parent.SOURCE in parent.SOURCES
    assert parent.TEST_SOURCE in parent.SOURCES
    assert parent.SCHEMA == parent.V8.SCHEMA
