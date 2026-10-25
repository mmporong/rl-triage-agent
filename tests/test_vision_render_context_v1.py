from __future__ import annotations

import copy
import hashlib
import importlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
C = importlib.import_module("vision_render_context_v1")


def _sha(path: Path, *, lf: bool = False) -> str:
    data = path.read_bytes()
    if lf:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path) -> tuple[Path, dict, Path, dict]:
    repo = (tmp_path / "repo").resolve()
    (repo / "bench/protocols").mkdir(parents=True)
    writer = repo / C.WRITER_RELATIVE
    writer.parent.mkdir(parents=True, exist_ok=True)
    assert C.__file__ is not None
    writer.write_bytes(Path(C.__file__).read_bytes())
    protocol = repo / "bench/protocols/source.json"
    protocol.write_text('{"schema":"source"}\n', encoding="utf-8")
    dependency = repo / "bench/dependency.py"
    dependency.write_text("VALUE = 1\n", encoding="utf-8")
    source_hashes = {
        C.WRITER_RELATIVE: _sha(writer, lf=True),
        "bench/dependency.py": _sha(dependency, lf=True),
    }

    venv = (tmp_path / "venv").resolve()
    base = (tmp_path / "base").resolve()
    (venv / "bin").mkdir(parents=True)
    base.mkdir()
    python = venv / "bin/python"
    python.write_bytes(b"fixture interpreter")
    packages = {
        "isaacsim": "4.5.0.0", "isaaclab": "0.41.3", "isaaclab_rl": "0.1.8",
        "torch": "2.7.0+cu128", "rsl-rl-lib": "2.3.3", "gymnasium": "1.2.0",
    }
    environment = {
        "python": "3.10.12", "isaaclab_sha": "1" * 40, **packages,
        "cuda_available": True, "cuda_device_count": 1,
    }
    snapshot = {
        "lab_head": "1" * 40,
        "lab_tracked_clean": True,
        "source_anchor": "2" * 40,
        "protocol": {"path": "bench/protocols/source.json", "sha256_lf": _sha(protocol, lf=True)},
        "environment": environment,
        "selected_interpreter": {
            "launcher": str(python), "python": "3.10.12", "executable": str(python),
            "prefix": str(venv), "base_prefix": str(base), "packages": packages,
        },
        "source_sha256_lf": source_hashes,
        "runtime_source_sha256_lf": {"scripts/train.py": "3" * 64},
        "python_package_source_sha256_lf": {"rsl_rl/algorithms/ppo.py": "4" * 64},
    }
    job = {
        "schema": C.JOB_SCHEMA, "method": C.JOB_METHOD, "mode": "record",
        "job_id": "record-1", "profile": {},
        "source": {"repo_root": str(repo), "commit": "2" * 40, "files_lf": dict(source_hashes)},
        "protocol": {"path": str(protocol), "sha256": _sha(protocol)},
        "checkpoint": {"path": str(protocol), "sha256": _sha(protocol), "logical_id": "policy"},
        "resource": {"resource_id": "brev-one", "gpu_uuid": "GPU-fixture"},
        "remote_gpu_admission": {}, "output_directory": str(tmp_path / "unused"),
    }
    job_path = (tmp_path / "job.json").resolve()
    _write(job_path, job)
    return job_path, snapshot, python, job


def _torch(**updates: object) -> dict[str, object]:
    value: dict[str, object] = {
        "version": "2.7.0+cu128", "built_cuda_version": "12.8",
        "device_name": "L40S", "compute_capability": [8, 9],
        "device_count": 1, "cuda_available": True,
    }
    value.update(updates)
    return value


class Runner:
    def __init__(self, gpu: str = "GPU-fixture, NVIDIA L40S, 570.10\n",
                 torch: dict[str, object] | None = None) -> None:
        self.gpu = gpu
        self.torch = _torch() if torch is None else torch
        self.calls: list[list[str]] = []

    def __call__(self, argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert kwargs == {
            "capture_output": True, "text": True, "encoding": "utf-8", "errors": "strict",
            "check": False, "timeout": 30,
        }
        self.calls.append(argv)
        if argv == C.GPU_COMMAND:
            return subprocess.CompletedProcess(argv, 0, self.gpu, "")
        return subprocess.CompletedProcess(
            argv, 0, json.dumps(self.torch, sort_keys=True, separators=(",", ":")) + "\n", "",
        )


def _capture(tmp_path: Path, runner: Runner | None = None):
    job_path, snapshot, python, job = _fixture(tmp_path)
    runner = runner or Runner()
    receipt = C.capture_context(job_path, snapshot, python, (tmp_path / "context").resolve(), runner=runner)
    fingerprint = C.read_context(receipt, job, _sha(job_path), snapshot)
    return receipt, fingerprint, job_path, snapshot, python, job, runner


def test_capture_and_read_bind_raw_gpu_torch_source_and_selected_python(tmp_path: Path) -> None:
    receipt, fingerprint, _, snapshot, python, _, runner = _capture(tmp_path)
    body = json.loads(receipt.read_text(encoding="utf-8"))
    assert runner.calls == [C.GPU_COMMAND, [str(python), "-I", "-c", C.TORCH_SCRIPT]]
    assert body["source_snapshot"] == snapshot
    assert C._utc_stamp(body["finished_at_utc"], "finished") >= C._utc_stamp(
        body["captured_at_utc"], "captured",
    )
    assert body["gpu_raw"]["stdout_file"] == C.GPU_FILE
    assert body["torch_raw"]["stdout_file"] == C.TORCH_FILE
    assert fingerprint["gpu"] == {
        "uuid": "GPU-fixture", "name": "NVIDIA L40S", "normalized_name": "L40S",
        "driver_version": "570.10",
    }
    assert fingerprint["torch"]["normalized_device_name"] == "L40S"
    assert fingerprint["source_snapshot_sha256"] == C._digest(snapshot)
    assert "captured_at_utc" not in fingerprint and "finished_at_utc" not in fingerprint


def test_capture_records_fixed_utc_start_and_finish_without_changing_fingerprint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    moments = iter([
        datetime(2026, 10, 25, 1, 2, 3, 123456, tzinfo=timezone.utc),  # noqa: UP017
        datetime(2026, 10, 25, 1, 2, 4, 654321, tzinfo=timezone.utc),  # noqa: UP017
    ])
    monkeypatch.setattr(C, "_utc_now", lambda: next(moments))
    receipt, fingerprint, _, _, _, _, _ = _capture(tmp_path)
    body = json.loads(receipt.read_text(encoding="utf-8"))
    assert body["captured_at_utc"] == "2026-10-25T01:02:03.123456+00:00"
    assert body["finished_at_utc"] == "2026-10-25T01:02:04.654321+00:00"
    assert set(fingerprint).isdisjoint({"captured_at_utc", "finished_at_utc"})


@pytest.mark.parametrize(("captured", "finished", "match"), [
    ("2026-10-25T01:02:03", "2026-10-25T01:02:04+00:00", "UTC ISO"),
    ("2026-10-25T01:02:03+09:00", "2026-10-25T01:02:04+09:00", "UTC timezone"),
    ("2026-10-25T01:02:05+00:00", "2026-10-25T01:02:04+00:00", "빠르다"),
])
def test_read_rejects_invalid_or_reversed_context_timestamps(
    tmp_path: Path, captured: str, finished: str, match: str,
) -> None:
    receipt, _, job_path, snapshot, _, job, _ = _capture(tmp_path)
    body = json.loads(receipt.read_text(encoding="utf-8"))
    body.update(captured_at_utc=captured, finished_at_utc=finished)
    receipt.unlink()
    _write(receipt, body)
    with pytest.raises(ValueError, match=match):
        C.read_context(receipt, job, _sha(job_path), snapshot)


def test_read_accepts_python310_dotnet_seven_digit_utc_fraction(tmp_path: Path) -> None:
    receipt, expected, job_path, snapshot, _, job, _ = _capture(tmp_path)
    body = json.loads(receipt.read_text(encoding="utf-8"))
    body.update(
        captured_at_utc="2026-10-25T01:02:03.1234567+00:00",
        finished_at_utc="2026-10-25T01:02:04.7654321+00:00",
    )
    receipt.unlink()
    _write(receipt, body)
    assert C.read_context(receipt, job, _sha(job_path), snapshot) == expected


@pytest.mark.parametrize("gpu", [
    "", "GPU-fixture, NVIDIA L40S\n", "GPU-fixture,NVIDIA L40S,570.10\n",
    "GPU-fixture, NVIDIA L40S, 570.10\nGPU-other, NVIDIA L40S, 570.10\n",
    "GPU-other, NVIDIA L40S, 570.10\n", "GPU-fixture, Unknown, 570.10\n",
    "GPU-fixture, NVIDIA L40S, N/A\n",
])
def test_capture_rejects_malformed_multiple_wrong_or_unknown_gpu_csv(tmp_path: Path, gpu: str) -> None:
    job_path, snapshot, python, _ = _fixture(tmp_path)
    with pytest.raises(ValueError, match="GPU"):
        C.capture_context(job_path, snapshot, python, (tmp_path / "context").resolve(), runner=Runner(gpu=gpu))
    assert (tmp_path / "context/gpu.csv").is_file()
    assert not (tmp_path / "context/context.receipt.json").exists()


@pytest.mark.parametrize(("updates", "match"), [
    ({"version": "wrong"}, "버전"),
    ({"built_cuda_version": None}, "built CUDA"),
    ({"device_name": "N/A"}, "device name"),
    ({"device_name": "RTX 4090"}, "GPU 이름"),
    ({"compute_capability": [8, True]}, "capability"),
    ({"compute_capability": [8, -1]}, "capability"),
    ({"device_count": True}, "GPU 한 개"),
    ({"device_count": 2}, "GPU 한 개"),
    ({"cuda_available": 1}, "GPU 한 개"),
])
def test_capture_rejects_invalid_torch_types_or_semantics(
    tmp_path: Path, updates: dict[str, object], match: str,
) -> None:
    job_path, snapshot, python, _ = _fixture(tmp_path)
    with pytest.raises(ValueError, match=match):
        C.capture_context(
            job_path, snapshot, python, (tmp_path / "context").resolve(),
            runner=Runner(torch=_torch(**updates)),
        )
    assert (tmp_path / "context/torch.json").is_file()
    assert not (tmp_path / "context/context.receipt.json").exists()


def test_read_rejects_forged_job_sha_and_raw_tamper(tmp_path: Path) -> None:
    receipt, _, job_path, snapshot, _, job, _ = _capture(tmp_path)
    with pytest.raises(ValueError, match="결합"):
        C.read_context(receipt, job, "f" * 64, snapshot)
    raw = receipt.parent / C.GPU_FILE
    raw.write_text("GPU-fixture, NVIDIA L40S, 570.11\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA"):
        C.read_context(receipt, job, _sha(job_path), snapshot)


def test_read_rejects_receipt_source_snapshot_and_raw_filename_tamper(tmp_path: Path) -> None:
    receipt, _, job_path, snapshot, _, job, _ = _capture(tmp_path)
    body = json.loads(receipt.read_text(encoding="utf-8"))
    body["gpu_raw"]["stdout_file"] = "../gpu.csv"
    receipt.unlink()
    _write(receipt, body)
    with pytest.raises(ValueError, match="실행 계약"):
        C.read_context(receipt, job, _sha(job_path), snapshot)


def test_no_overwrite_and_failed_capture_cannot_retry_same_output(tmp_path: Path) -> None:
    job_path, snapshot, python, _ = _fixture(tmp_path)
    output = (tmp_path / "context").resolve()
    with pytest.raises(ValueError, match="GPU"):
        C.capture_context(job_path, snapshot, python, output, runner=Runner(gpu="bad"))
    with pytest.raises(FileExistsError):
        C.capture_context(job_path, snapshot, python, output, runner=Runner())


def test_nonzero_and_stderr_preserve_partial_raw_without_receipt(tmp_path: Path) -> None:
    job_path, snapshot, python, _ = _fixture(tmp_path)

    def runner(argv: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(argv, 7, "partial", "query failed")

    output = (tmp_path / "context").resolve()
    with pytest.raises(ValueError, match="깨끗하게"):
        C.capture_context(job_path, snapshot, python, output, runner=runner)
    assert (output / C.GPU_FILE).read_text(encoding="utf-8") == "partial"
    assert not (output / C.RECEIPT_FILE).exists()


def test_source_snapshot_requires_writer_and_covers_job_source(tmp_path: Path) -> None:
    job_path, snapshot, python, job = _fixture(tmp_path)
    missing = copy.deepcopy(snapshot)
    missing["source_sha256_lf"].pop(C.WRITER_RELATIVE)
    with pytest.raises(ValueError, match="writer"):
        C.capture_context(job_path, missing, python, (tmp_path / "missing").resolve(), runner=Runner())
    job["source"]["files_lf"]["extra.py"] = "a" * 64
    _write(job_path, job)
    with pytest.raises(ValueError, match="포함되지"):
        C.capture_context(job_path, snapshot, python, (tmp_path / "uncovered").resolve(), runner=Runner())


def test_selected_python_binding_must_match_snapshot(tmp_path: Path) -> None:
    job_path, snapshot, _, _ = _fixture(tmp_path)
    other = (tmp_path / "other/python").resolve()
    other.parent.mkdir()
    other.write_bytes(b"other")
    with pytest.raises(ValueError, match="launcher"):
        C.capture_context(job_path, snapshot, other, (tmp_path / "context").resolve(), runner=Runner())


def test_source_protocol_is_separate_from_source_map_and_its_bytes_are_checked(tmp_path: Path) -> None:
    job_path, snapshot, python, job = _fixture(tmp_path)
    assert snapshot["protocol"]["path"] not in snapshot["source_sha256_lf"]
    protocol = Path(job["source"]["repo_root"]) / snapshot["protocol"]["path"]
    protocol.write_text('{"schema":"changed"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="source protocol LF SHA"):
        C.capture_context(
            job_path, snapshot, python, (tmp_path / "context").resolve(), runner=Runner(),
        )


def test_capture_allows_selected_venv_python_symlink_when_supported(tmp_path: Path) -> None:
    job_path, snapshot, python, _ = _fixture(tmp_path)
    target = python.with_name("python-real")
    python.rename(target)
    try:
        python.symlink_to(target.name)
    except OSError:
        target.rename(python)
        pytest.skip("이 환경은 symlink 생성을 허용하지 않는다")
    receipt = C.capture_context(
        job_path, snapshot, python, (tmp_path / "context").resolve(), runner=Runner(),
    )
    assert receipt.is_file()


def test_read_is_portable_across_nonexistent_remote_source_and_interpreter_paths(tmp_path: Path) -> None:
    receipt, _, _, snapshot, _, job, _ = _capture(tmp_path)
    remote_snapshot = copy.deepcopy(snapshot)
    remote_snapshot["selected_interpreter"].update({
        "launcher": "/remote/vision-venv/bin/python",
        "executable": "/remote/vision-venv/bin/python",
        "prefix": "/remote/vision-venv",
        "base_prefix": "/usr",
    })
    remote_job = copy.deepcopy(job)
    remote_job["source"]["repo_root"] = "/remote/rl-triage-agent"
    body = json.loads(receipt.read_text(encoding="utf-8"))
    body["job_sha256"] = "a" * 64
    body["source_snapshot"] = remote_snapshot
    body["source_snapshot_sha256"] = C._digest(remote_snapshot)
    body["torch_raw"]["argv"][0] = remote_snapshot["selected_interpreter"]["launcher"]
    receipt.unlink()
    _write(receipt, body)
    fingerprint = C.read_context(receipt, remote_job, "a" * 64, remote_snapshot)
    assert fingerprint["selected_interpreter"]["launcher"] == "/remote/vision-venv/bin/python"


def test_job_hardlink_and_context_raw_hardlink_are_rejected(tmp_path: Path) -> None:
    job_path, snapshot, python, _ = _fixture(tmp_path)
    os.link(job_path, tmp_path / "job-hardlink.json")
    with pytest.raises(ValueError, match="단일 link"):
        C.capture_context(job_path, snapshot, python, (tmp_path / "context-a").resolve(), runner=Runner())

    fresh = tmp_path / "fresh"
    fresh.mkdir()
    job_path, snapshot, python, job = _fixture(fresh)
    receipt = C.capture_context(
        job_path, snapshot, python, (fresh / "context").resolve(), runner=Runner(),
    )
    os.link(receipt.parent / C.GPU_FILE, fresh / "gpu-hardlink.csv")
    with pytest.raises(ValueError, match="단일 link"):
        C.read_context(receipt, job, _sha(job_path), snapshot)


def test_symlink_job_and_raw_are_rejected_when_supported(tmp_path: Path) -> None:
    job_path, snapshot, python, job = _fixture(tmp_path)
    linked_job = tmp_path / "linked-job.json"
    try:
        linked_job.symlink_to(job_path)
    except OSError:
        pytest.skip("이 환경은 symlink 생성을 허용하지 않는다")
    with pytest.raises(ValueError, match="symlink"):
        C.capture_context(linked_job, snapshot, python, (tmp_path / "context-a").resolve(), runner=Runner())

    receipt = C.capture_context(
        job_path, snapshot, python, (tmp_path / "context-b").resolve(), runner=Runner(),
    )
    raw = receipt.parent / C.GPU_FILE
    saved = receipt.parent / "saved.csv"
    raw.rename(saved)
    raw.symlink_to(saved)
    with pytest.raises(ValueError, match="symlink"):
        C.read_context(receipt, job, _sha(job_path), snapshot)


def test_empty_raw_file_is_rejected_even_if_receipt_sha_is_rewritten(tmp_path: Path) -> None:
    receipt, _, job_path, snapshot, _, job, _ = _capture(tmp_path)
    raw = receipt.parent / C.GPU_FILE
    raw.write_bytes(b"")
    body = json.loads(receipt.read_text(encoding="utf-8"))
    body["gpu_raw"]["stdout_sha256"] = _sha(raw)
    receipt.unlink()
    _write(receipt, body)
    with pytest.raises(ValueError, match="한 GPU"):
        C.read_context(receipt, job, _sha(job_path), snapshot)


def test_module_import_does_not_import_torch() -> None:
    source = (
        "import importlib,sys\n"
        f"sys.path.insert(0,{str(ROOT / 'bench')!r})\n"
        "importlib.import_module('vision_render_context_v1')\n"
        "assert 'torch' not in sys.modules\n"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", source], capture_output=True, text=True,
        encoding="utf-8", errors="strict", env={**os.environ, "PYTHONUTF8": "0"}, check=False,
    )
    assert result.returncode == 0, result.stderr


def test_public_api_runs_under_python_utf8_zero_subprocess(tmp_path: Path) -> None:
    script = tmp_path / "probe.py"
    script.write_text(
        "import sys\n"
        f"sys.path.insert(0,{str(ROOT / 'bench')!r})\n"
        "import vision_render_context_v1 as C\n"
        "assert C.SCHEMA == 'vision_render_runtime_context_v1'\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, "-I", str(script)], capture_output=True, text=True,
        encoding="utf-8", errors="strict", env={**os.environ, "PYTHONUTF8": "0"}, check=False,
    )
    assert result.returncode == 0, result.stderr
