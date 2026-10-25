from __future__ import annotations

import hashlib
import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))
G = importlib.import_module("vision_remote_gpu_v1")


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, str):
        path.write_text(value, encoding="utf-8", newline="")
    else:
        path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _proof(tmp_path: Path, *, status: str = "RUNNING", processes: str = "") -> dict:
    provider, gpu, process = tmp_path / "provider.json", tmp_path / "gpu.csv", tmp_path / "process.csv"
    row = {key: None for key in G.PROVIDER_ROW_KEYS}
    row.update({"id": "instance-1", "name": "fixture", "status": status})
    _write(provider, {"workspaces": [row]})
    _write(gpu, "GPU-fixture\n")
    _write(process, processes)
    return {
        "resource_id": "instance-1", "gpu_uuid": "GPU-fixture",
        "provider_inventory_path": str(provider.resolve()),
        "provider_inventory_sha256": _sha(provider),
        "nvidia_gpu_csv_path": str(gpu.resolve()), "nvidia_gpu_csv_sha256": _sha(gpu),
        "nvidia_process_csv_path": str(process.resolve()),
        "nvidia_process_csv_sha256": _sha(process),
    }


def test_capture_proof_requires_running_provider_and_exact_single_gpu(tmp_path: Path) -> None:
    G.verify_capture_proof(_proof(tmp_path))
    with pytest.raises(ValueError, match="RUNNING"):
        G.verify_capture_proof(_proof(tmp_path / "stopped", status="STOPPED"))
    proof = _proof(tmp_path / "two")
    Path(proof["nvidia_gpu_csv_path"]).write_text("GPU-fixture\nGPU-other\n", encoding="utf-8")
    proof["nvidia_gpu_csv_sha256"] = _sha(Path(proof["nvidia_gpu_csv_path"]))
    with pytest.raises(ValueError, match="단일"):
        G.verify_capture_proof(proof)
    with pytest.raises(ValueError, match="기존 compute process"):
        G.verify_capture_proof(_proof(tmp_path / "busy", processes="GPU-fixture, 41\n"))


@pytest.mark.parametrize("value", ["N/A", "Unknown", "GPU-other, 42", "GPU-fixture, x", "42"])
def test_process_csv_rejects_unknown_malformed_or_other_gpu(value: str) -> None:
    with pytest.raises(ValueError, match="process CSV"):
        G.parse_process_csv(value, "GPU-fixture")


def test_reader_records_raw_sha_and_rejects_query_failure(tmp_path: Path) -> None:
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        stdout = "GPU-fixture\n" if command == G.GPU_COMMAND else "GPU-fixture, 22\n"
        return subprocess.CompletedProcess(command, 0, stdout, "")

    reader = G.StrictRemoteGpuReader(_proof(tmp_path), tmp_path / "audit", run=run)
    assert reader() == [22]
    receipt = json.loads((tmp_path / "audit/000001.receipt.json").read_text())
    assert receipt["gpu_stdout_sha256"] == _sha(tmp_path / "audit/000001.gpu.csv")
    assert receipt["process_stdout_sha256"] == _sha(tmp_path / "audit/000001.process.csv")
    assert all(call[1] == {"capture_output": True, "text": True, "check": False, "timeout": 30}
               for call in calls)
    assert reader.last_receipt == {
        "path": str((tmp_path / "audit/000001.receipt.json").resolve()),
        "sha256": _sha(tmp_path / "audit/000001.receipt.json"),
    }

    def failed(command, **kwargs):
        return subprocess.CompletedProcess(command, 1, "N/A\n", "query failed")

    broken = G.StrictRemoteGpuReader(_proof(tmp_path / "broken"), tmp_path / "broken-audit", run=failed)
    with pytest.raises(ValueError, match="깨끗하게"):
        broken()
    assert (tmp_path / "broken-audit/000001.receipt.json").is_file()


def test_capture_proof_rejects_tampered_original(tmp_path: Path) -> None:
    proof = _proof(tmp_path)
    Path(proof["provider_inventory_path"]).write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA"):
        G.verify_capture_proof(proof)


def test_observation_reference_rechecks_missing_and_coherent_rewrite(tmp_path: Path) -> None:
    def run(command, **kwargs):
        stdout = "GPU-fixture\n" if command == G.GPU_COMMAND else ""
        return subprocess.CompletedProcess(command, 0, stdout, "")

    reader = G.StrictRemoteGpuReader(_proof(tmp_path), tmp_path / "audit", run=run)
    assert reader() == []
    reference = dict(reader.last_receipt or {})
    G.verify_observation_reference(reference, "instance-1", "GPU-fixture")
    process_path = tmp_path / "audit/000001.process.csv"
    process_path.unlink()
    with pytest.raises(ValueError, match="원본 파일"):
        G.verify_observation_reference(reference, "instance-1", "GPU-fixture")

    reader2 = G.StrictRemoteGpuReader(_proof(tmp_path / "rewrite"), tmp_path / "rewrite-audit", run=run)
    reader2()
    reference2 = dict(reader2.last_receipt or {})
    process_path = tmp_path / "rewrite-audit/000001.process.csv"
    process_path.write_text("GPU-fixture, 91\n", encoding="utf-8", newline="")
    receipt_path = Path(reference2["path"])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["process_stdout_sha256"] = _sha(process_path)
    receipt_path.unlink()
    _write(receipt_path, receipt)
    reference2["sha256"] = _sha(receipt_path)
    with pytest.raises(ValueError, match="compute process"):
        G.verify_observation_reference(reference2, "instance-1", "GPU-fixture")
