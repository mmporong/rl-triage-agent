from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest  # pyright: ignore[reportMissingImports]

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / "bench", ROOT / "evals", ROOT / "src"):
    sys.path.insert(0, str(folder))

import vision_transferred_pull_job_v1 as J  # pyright: ignore[reportMissingImports]

NOW = datetime(2026, 10, 27, 12, 0, tzinfo=timezone.utc)  # noqa: UP017 - Python 3.10
TARGET = {"resource_id": "new-l40s-id", "resource_name": "new-recording-vm",
          "instance_type": "l40s-48gb.1x"}


def _write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, str):
        path.write_text(value, encoding="utf-8", newline="\n")
    else:
        path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8", newline="\n")
    return path


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(repo), *args], check=True,
                            capture_output=True, text=True, encoding="utf-8")
    return result.stdout.strip()


def _historical(tmp_path: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, str]]]:
    lab = tmp_path / "isaaclab"
    rows = []
    for identity in J.CHECKPOINT_IDS:
        path = lab / "logs" / f"{identity}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(identity.encode())
        rows.append({"logical_id": identity,
                     "isaaclab_relative": path.relative_to(lab).as_posix(),
                     "sha256": J._sha(path)})
    source = {"protocol_id": "vision-source-20261026x",
              "resources": {"owned_instance_id": "old-source-id"}}
    return source, {"checkpoints": rows}, rows


def _repo(tmp_path: Path) -> tuple[Path, str, dict[str, str]]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "test")
    for relative in J.REQUIRED_SOURCE_FILES:
        _write(repo / relative, f"# {relative}\n")
    _git(repo, "add", *sorted(J.REQUIRED_SOURCE_FILES))
    _git(repo, "commit", "-m", "fixture")
    commit = _git(repo, "rev-parse", "HEAD")
    sources = {relative: J._sha(repo / relative, lf=True)
               for relative in J.REQUIRED_SOURCE_FILES}
    return repo, commit, sources


def _protocol(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, dict[str, Any], Path]:
    repo, commit, sources = _repo(tmp_path)
    historical = _historical(tmp_path)
    monkeypatch.setattr(J, "_historical", lambda _repo: historical)
    value = J.build_protocol(
        protocol_id="transferred-record-20261027a",
        registered_at_utc=(NOW - timedelta(minutes=1)).isoformat(),
        source_commit=commit,
        source_sha256_lf=sources,
        recording_target=TARGET,
        compute_quote_usd_per_h="1.74",
        batch_budget_usd="2.00",
        deadline_utc=(NOW + timedelta(hours=2)).isoformat(),
        repo=repo,
    )
    path = _write(tmp_path / "protocol.json", value)
    return path, value, repo


def _proof(tmp_path: Path, protocol: dict[str, Any], protocol_path: Path, *,
           process_text: str = "", source_sha: str = J.SOURCE_PROTOCOL["sha256_lf"],
           observed: datetime = NOW, expires: datetime | None = None) -> Path:
    provider = _write(tmp_path / "provider.json", {"workspaces": [{
        "build_status": "READY", "gpu": "L40S", "health_status": "HEALTHY",
        "id": TARGET["resource_id"], "instance_kind": "gpu",
        "instance_type": TARGET["instance_type"], "name": TARGET["resource_name"],
        "shell_status": "READY", "status": "RUNNING",
    }]})
    gpu = _write(tmp_path / "gpu.csv", "GPU-NEW\n")
    processes = _write(tmp_path / "processes.csv", process_text)
    capture = {
        "resource_id": TARGET["resource_id"], "gpu_uuid": "GPU-NEW",
        "provider_inventory_path": str(provider.resolve()),
        "provider_inventory_sha256": J._sha(provider),
        "nvidia_gpu_csv_path": str(gpu.resolve()), "nvidia_gpu_csv_sha256": J._sha(gpu),
        "nvidia_process_csv_path": str(processes.resolve()),
        "nvidia_process_csv_sha256": J._sha(processes),
    }
    value = {
        "schema": J.RESOURCE_SCHEMA, "status": "granted",
        "observed_at_utc": observed.isoformat(),
        "expires_at_utc": (expires or NOW + timedelta(hours=1)).isoformat(),
        "recording_protocol_sha256_lf": J._sha(protocol_path, lf=True),
        "source_protocol_sha256_lf": source_sha, "recording_target": TARGET,
        "source_resource_reused": False, "capture": capture,
    }
    return _write(tmp_path / "resource.json", value)


def _job(tmp_path: Path, protocol_path: Path, protocol: dict[str, Any], repo: Path,
         proof: Path, *, omit_output: bool = False) -> Path:
    stage = tmp_path / "stage"
    stage.mkdir()
    lab = tmp_path / "isaaclab"
    row = protocol["checkpoints"][0]
    python = _write(tmp_path / "python", "python")
    job = {
        "schema": J.JOB_SCHEMA, "mode": J.MODE,
        "job_id": f"{protocol['protocol_id']}-{row['logical_id']}",
        "source": {"repo_root": str(repo.resolve()), "commit": protocol["source_commit"],
                   "files_lf": protocol["source_sha256_lf"]},
        "protocol": {"path": str(protocol_path.resolve()),
                     "sha256_lf": J._sha(protocol_path, lf=True)},
        "resource_proof": {"path": str(proof.resolve()), "sha256": J._sha(proof)},
        "isaaclab": str(lab.resolve()), "python": str(python.resolve()),
        "output_directory": str((stage / "output").resolve()),
        "profile": J.RECORD.fixed_profile(),
        "checkpoint": {"path": str((lab / row["isaaclab_relative"]).resolve()),
                       "sha256": row["sha256"], "logical_id": row["logical_id"]},
    }
    if omit_output:
        del job["output_directory"]
    return _write(stage / "job.json", job)


def _allow_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(J.COMMON, "_python_executable", lambda *_args: None)
    monkeypatch.setattr(J.COMMON, "_verify_runtime_sources", lambda *_args: None)


def test_new_source_and_target_pass_without_mutating_historical_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_path = ROOT / J.SOURCE_PROTOCOL["path"]
    base_path = ROOT / J.BASE_PROTOCOL["path"]
    before = (source_path.read_bytes(), base_path.read_bytes())
    protocol_path, protocol, repo = _protocol(tmp_path, monkeypatch)
    proof = _proof(tmp_path, protocol, protocol_path)
    job = _job(tmp_path, protocol_path, protocol, repo, proof)
    _allow_runtime(monkeypatch)

    checked = J.validate_job(job, now=NOW)

    assert checked["resource"]["resource_id"] == TARGET["resource_id"]
    assert checked["protocol"]["gpu_validation_status"] == "pending_live_gpu_execution"
    assert checked["protocol"]["source_lineage"]["historical_resource_used_for_recording"] is False
    assert before == (source_path.read_bytes(), base_path.read_bytes())


@pytest.mark.parametrize(("budget", "accepted"), [("0.698720", True), ("0.698719", False)])
def test_exact_wall_budget_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, budget: str, accepted: bool
) -> None:
    path, protocol, repo = _protocol(tmp_path, monkeypatch)
    protocol["resources"].update(compute_quote_usd_per_h="2.09616", batch_budget_usd=budget)
    _write(path, protocol)
    if accepted:
        J.validate_protocol(path, repo=repo, now=NOW)
    else:
        with pytest.raises(ValueError, match="batch budget"):
            J.validate_protocol(path, repo=repo, now=NOW)


def test_resource_builder_uses_only_new_target_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol_path, _protocol_value, _repo_value = _protocol(tmp_path, monkeypatch)
    provider = _write(tmp_path / "provider-build.json", {"workspaces": [{
        "build_status": "READY", "gpu": "L40S", "health_status": "HEALTHY",
        "id": TARGET["resource_id"], "instance_kind": "gpu",
        "instance_type": TARGET["instance_type"], "name": TARGET["resource_name"],
        "shell_status": "READY", "status": "RUNNING",
    }]})
    gpu = _write(tmp_path / "gpu-build.csv", "GPU-NEW\n")
    processes = _write(tmp_path / "process-build.csv", "")
    proof = J.build_resource_proof(
        protocol_path, provider_inventory_path=provider, nvidia_gpu_csv_path=gpu,
        nvidia_process_csv_path=processes, observed_at_utc=NOW.isoformat(),
        expires_at_utc=(NOW + timedelta(hours=1)).isoformat(), repo=_repo_value,
    )
    assert proof["recording_target"] == TARGET
    assert proof["capture"]["resource_id"] == TARGET["resource_id"]
    assert proof["source_resource_reused"] is False


def test_builder_rejects_historical_resource_as_new_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, commit, sources = _repo(tmp_path)
    monkeypatch.setattr(J, "_historical", lambda _repo: _historical(tmp_path))
    with pytest.raises(ValueError, match="historical source resource"):
        J.build_protocol(
            protocol_id="transferred-record-20261027a",
            registered_at_utc=(NOW - timedelta(minutes=1)).isoformat(),
            source_commit=commit, source_sha256_lf=sources,
            recording_target={**TARGET, "resource_id": "old-source-id"},
            compute_quote_usd_per_h="1.74", batch_budget_usd="2.00",
            deadline_utc=(NOW + timedelta(hours=2)).isoformat(), repo=repo,
        )


@pytest.mark.parametrize("case", ["old", "tampered", "expired", "busy", "source_mismatch"])
def test_resource_proof_rejects_old_tampered_expired_busy_and_source_mismatch(
    case: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol_path, protocol, repo = _protocol(tmp_path, monkeypatch)
    proof = _proof(
        tmp_path, protocol, protocol_path,
        process_text="GPU-NEW, 123\n" if case == "busy" else "",
        source_sha="f" * 64 if case == "source_mismatch" else J.SOURCE_PROTOCOL["sha256_lf"],
        observed=NOW - timedelta(minutes=2) if case == "expired" else NOW,
    )
    if case == "old":
        value = json.loads(proof.read_text())
        value["recording_target"]["resource_id"] = "old-source-id"
        value["capture"]["resource_id"] = "old-source-id"
        _write(proof, value)
    job = _job(tmp_path, protocol_path, protocol, repo, proof)
    if case == "tampered":
        Path(json.loads(proof.read_text())["capture"]["nvidia_gpu_csv_path"]).write_text(
            "GPU-TAMPERED\n", encoding="utf-8")
    else:
        body = json.loads(job.read_text())
        body["resource_proof"]["sha256"] = J._sha(proof)
        _write(job, body)
    _allow_runtime(monkeypatch)
    with pytest.raises(ValueError):
        J.validate_job(job, now=NOW)


def test_job_rejects_missing_output_lifecycle_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol_path, protocol, repo = _protocol(tmp_path, monkeypatch)
    proof = _proof(tmp_path, protocol, protocol_path)
    job = _job(tmp_path, protocol_path, protocol, repo, proof, omit_output=True)
    _allow_runtime(monkeypatch)
    with pytest.raises(ValueError, match="필드가 계약과 다르다"):
        J.validate_job(job, now=NOW)


def test_fresh_job_rejects_preexisting_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol_path, protocol, repo = _protocol(tmp_path, monkeypatch)
    proof = _proof(tmp_path, protocol, protocol_path)
    job = _job(tmp_path, protocol_path, protocol, repo, proof)
    (job.parent / "output").mkdir()
    _allow_runtime(monkeypatch)
    with pytest.raises(FileExistsError):
        J.validate_job(job, now=NOW)
