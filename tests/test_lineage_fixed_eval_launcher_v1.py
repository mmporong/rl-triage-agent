from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "bench/run_lineage_fixed_eval_v1.ps1"
ANCHOR = "10942fcdde672156ed751b26c3de2270c59c06bb"


@pytest.fixture
def powershell() -> str:
    executable = shutil.which("pwsh") or shutil.which("powershell")
    if os.name != "nt" or executable is None:
        pytest.skip("Windows GPU launcher의 실제 PowerShell CPU 검증")
    return executable


def invoke(powershell: str, output: Path, source: Path, protocol: Path, *, execute=False):
    args = [powershell, "-NoProfile", "-NonInteractive", "-File", str(SCRIPT),
            "-Anchor", ANCHOR, "-Protocol", str(protocol), "-SourceRoot", str(source),
            "-Output", str(output)]
    if execute:
        args.append("-Execute")
    return subprocess.run(args, check=False, cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=30,
                          env={**os.environ, "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1"},
                          creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


def test_preview_has_no_side_effects_and_preserves_prepare_execute_arguments(powershell, tmp_path):
    output, source, protocol = tmp_path / "new-output", tmp_path / "source", tmp_path / "protocol.json"
    child = invoke(powershell, output, source, protocol)
    assert child.returncode == 0, child.stderr
    preview = json.loads(child.stdout)
    assert preview["mode"] == "preview" and preview["phase"] == "fixed_evaluation_only"
    assert preview["sourceJobsRequired"] == preview["evaluationJobs"] == 18
    assert preview["trainingUpdates"] == preview["API"] == preview["paidCloudStarts"] == 0
    assert preview["owner"] == "rl-triage-lineage-sources-v1"
    assert preview["prepare"][2] == "prepare-inputs" and preview["execute"][2] == "execute"
    assert preview["prepare"][preview["prepare"].index("--source-root") + 1] == str(source)
    assert preview["execute"][preview["execute"].index("--manifest") + 1] == str(output / "manifest.json")
    assert not output.exists() and not source.exists() and not protocol.exists()


def test_existing_output_is_rejected_before_any_preparation_or_lease(powershell, tmp_path):
    output = tmp_path / "existing-output"
    output.mkdir()
    original = output / "original.json"
    original.write_bytes(b'{"keep":true}\n')
    child = invoke(powershell, output, tmp_path / "no-source", tmp_path / "no-protocol", execute=True)
    assert child.returncode != 0
    assert list(output.iterdir()) == [original]
    assert original.read_bytes() == b'{"keep":true}\n'


def test_incomplete_source_stops_before_prepare_and_gpu_lease(powershell, tmp_path):
    output, source = tmp_path / "new-output", tmp_path / "incomplete-source"
    source.mkdir()
    protocol = ROOT / "bench/protocols/lineage_source_collection_20261030_v2.json"
    child = invoke(powershell, output, source, protocol, execute=True)
    assert child.returncode != 0
    assert not output.exists() and list(source.iterdir()) == []
