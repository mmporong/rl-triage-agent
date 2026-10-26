from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
import vision_semantic_job_v1 as J


def _sha(path: Path, *, lf: bool = False) -> str:
    data = path.read_bytes().replace(b"\r\n", b"\n") if lf else path.read_bytes()
    return hashlib.sha256(data).hexdigest()


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True,
        encoding="utf-8",
    ).stdout.strip()


def _fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    *, metadata_contract: dict[str, object] | None = None,
) -> dict[str, object]:
    repo = tmp_path / "repo"
    repo.mkdir(parents=True)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "test")
    parent_path = repo / "bench/protocols/parent.json"
    parent_source = repo / "bench/parent_source.py"
    parent_source.parent.mkdir(parents=True)
    parent_source.write_text("PARENT = True\n", encoding="utf-8")
    parent = {
        "protocol_id": "vision-source-parent",
        "source_sha256_lf": {"bench/parent_source.py": _sha(parent_source, lf=True)},
        "runtime_source_sha256_lf": {"runtime.py": hashlib.sha256(b"runtime\n").hexdigest()},
        "python_package_source_sha256_lf": {"package.py": "1" * 64},
        "resources": {
            "owned_instance_id": "3655a80oo", "max_owned_instances": 1,
            "compute_quote_usd_per_h": 1.74, "whole_batch_budget_usd": 11.499326,
        },
    }
    _write(parent_path, parent)
    source_paths = set(J.REQUIRED_SOURCE_FILES) | {"bench/parent_source.py"}
    for relative in source_paths - {"bench/parent_source.py"}:
        target = repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if relative == J.HISTORY_PROTOCOL_PATH and metadata_contract is not None:
            _write(target, {"resources": {"metadata_lifecycle_contract": metadata_contract}})
        else:
            target.write_text(f"# {relative}\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "anchor")
    commit = _git(repo, "rev-parse", "HEAD")
    source_map = {relative: _sha(repo / relative, lf=True) for relative in source_paths}
    protocol = {
        "schema": J.PROTOCOL_SCHEMA,
        "protocol_id": "semantic-diagnostic-v1",
        "mode": J.MODE,
        "registered_before_measurement": True,
        "source_commit": commit,
        "source_sha256_lf": source_map,
        "parent_source_protocol": {
            "path": "bench/protocols/parent.json", "sha256_lf": _sha(parent_path, lf=True),
        },
        "profile_sha256": J.RECORD.digest(J.RECORD.fixed_profile()),
        "limits": dict(J.LIMITS),
        "resources": {
            "owned_instance_id": "3655a80oo",
            "owned_instance_name": "rl-triage-e2e-20261024-c9f4",
            "compute_quote_usd_per_h": 1.74,
            "max_owned_instances": 1,
            "prior_compute_estimate_usd": "8.500674",
            "fresh_lifecycle_budget_usd": "11.499326",
            "total_program_budget_usd": "20.000000",
            "prior_stop_receipts": [{"path": "stop.json", "sha256": "2" * 64}],
        },
        "preregistration": {
            "prediction": "semantic annotator 입력 구조를 관찰한다.",
            "observation_phrases": {"empty": "영역이 비어 있음", "present": "영역이 존재함"},
            "fixed_flags": dict(J.FIXED_FLAGS),
        },
    }
    stop = {
        "status": "STOPPED",
        "owned_instance": {"id": "3655a80oo", "name": "rl-triage-e2e-20261024-c9f4"},
        "execution_identity": {"run_id": "prior-run"},
        "cost_estimate": {"estimated_compute_cost_usd": "8.500674"},
    }
    _write(repo / "stop.json", stop)
    protocol["resources"]["prior_stop_receipts"][0]["sha256"] = _sha(repo / "stop.json")
    protocol_path = repo / "bench/protocols/semantic.json"
    _write(protocol_path, protocol)
    _git(repo, "add", "bench/protocols/semantic.json")
    _git(repo, "commit", "-qm", "register protocol")
    monkeypatch.setattr(J.SOURCE, "validate_protocol", lambda value: parent)
    return {"repo": repo, "parent": parent, "protocol": protocol,
            "protocol_path": protocol_path, "commit": commit}


def _job_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    fixture = _fixture(tmp_path, monkeypatch)
    repo = fixture["repo"]
    assert isinstance(repo, Path)
    protocol = fixture["protocol"]
    protocol_path = fixture["protocol_path"]
    assert isinstance(protocol, dict) and isinstance(protocol_path, Path)
    now = datetime(2026, 10, 26, tzinfo=timezone.utc)  # noqa: UP017 - Python 3.10
    resource = tmp_path / "resource.json"
    _write(resource, {"issued_at_utc": (now - timedelta(seconds=10)).isoformat()})
    lab = tmp_path / "IsaacLab"
    lab.mkdir()
    (lab / "runtime.py").write_text("runtime\n", encoding="utf-8")
    python = tmp_path / "python"
    shutil.copyfile(sys.executable, python)
    monkeypatch.setattr(J.sys, "executable", str(python.absolute()))
    stage = tmp_path / "stage"
    stage.mkdir()
    job = {
        "schema": J.JOB_SCHEMA,
        "mode": J.MODE,
        "job_id": "semantic-job-1",
        "source": {
            "repo_root": str(repo.resolve()), "commit": fixture["commit"],
            "files_lf": protocol["source_sha256_lf"],
        },
        "protocol": {"path": str(protocol_path.resolve()), "sha256_lf": _sha(protocol_path, lf=True)},
        "resource_proof": {"path": str(resource.resolve()), "sha256": _sha(resource)},
        "isaaclab": str(lab.resolve()),
        "python": str(python.resolve()),
        "output_directory": str((stage / "output").resolve()),
    }
    job_path = stage / "job.json"
    _write(job_path, job)

    def verify_resource(_batch: object, value: dict[str, object]) -> dict[str, object]:
        return {
            **value, "resource_id": "3655a80oo", "gpu_uuid": "GPU-fixture",
            "expires_at_utc": (now + timedelta(hours=1)).isoformat(),
        }

    monkeypatch.setattr(J.SOURCE.SourceBatch, "_verify_resource", verify_resource)
    fixture.update({"job": job, "job_path": job_path, "resource": resource, "now": now, "lab": lab})
    return fixture


def _commit_protocol(fixture: dict[str, object], message: str = "update protocol") -> None:
    repo, protocol_path = fixture["repo"], fixture["protocol_path"]
    assert isinstance(repo, Path) and isinstance(protocol_path, Path)
    _git(repo, "add", protocol_path.relative_to(repo).as_posix())
    _git(repo, "commit", "-qm", message)


def test_protocol_and_job_validate_fixed_diagnostic_contract(tmp_path: Path, monkeypatch) -> None:
    fixture = _job_fixture(tmp_path, monkeypatch)
    protocol = J.validate_protocol(fixture["protocol_path"], repo=fixture["repo"])
    assert protocol["limits"] == J.LIMITS
    assert protocol["limits"]["training_actions"] == 0
    assert protocol["limits"]["model_api_requests"] == 0
    assert protocol["limits"]["full_recordings"] == 0
    result = J.validate_job(fixture["job_path"], now=fixture["now"])
    assert result["job_path"] == fixture["job_path"].resolve()
    assert result["output_directory"] == fixture["job_path"].parent / "output"
    assert result["repo"] == fixture["repo"].resolve()


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("mode", "record", "schema/mode"),
        ("registered_before_measurement", False, "등록 상태"),
        ("limits", {**J.LIMITS, "model_api_requests": 1}, "실행 상한"),
    ],
)
def test_protocol_rejects_unregistered_mode_and_caps(
    tmp_path: Path, monkeypatch, field: str, value: object, match: str,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    body = fixture["protocol"]
    assert isinstance(body, dict)
    body[field] = value
    _write(fixture["protocol_path"], body)
    _commit_protocol(fixture)
    with pytest.raises(ValueError, match=match):
        J.validate_protocol(fixture["protocol_path"], repo=fixture["repo"])


def test_protocol_rejects_budget_owner_and_source_tamper(tmp_path: Path, monkeypatch) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    body = fixture["protocol"]
    assert isinstance(body, dict)
    body["resources"]["prior_compute_estimate_usd"] = "8.500675"
    _write(fixture["protocol_path"], body)
    _commit_protocol(fixture)
    with pytest.raises(ValueError, match="budget"):
        J.validate_protocol(fixture["protocol_path"], repo=fixture["repo"])

    fixture2 = _fixture(tmp_path / "owner", monkeypatch)
    body2 = fixture2["protocol"]
    body2["resources"]["owned_instance_id"] = "other"
    _write(fixture2["protocol_path"], body2)
    _commit_protocol(fixture2)
    with pytest.raises(ValueError, match="budget"):
        J.validate_protocol(fixture2["protocol_path"], repo=fixture2["repo"])

    fixture3 = _fixture(tmp_path / "source", monkeypatch)
    (fixture3["repo"] / "bench/parent_source.py").write_text("TAMPER = True\n", encoding="utf-8")
    with pytest.raises(ValueError, match="LF SHA"):
        J.validate_protocol(fixture3["protocol_path"], repo=fixture3["repo"])


def test_protocol_requires_clean_head_blob_but_allows_later_unrelated_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    dirty = _fixture(tmp_path / "dirty", monkeypatch)
    dirty_body = dirty["protocol"]
    assert isinstance(dirty_body, dict)
    dirty_body["protocol_id"] = "dirty-protocol"
    _write(dirty["protocol_path"], dirty_body)
    with pytest.raises(ValueError, match="HEAD committed LF blob"):
        J.validate_protocol(dirty["protocol_path"], repo=dirty["repo"])

    untracked = _fixture(tmp_path / "untracked", monkeypatch)
    untracked_repo, untracked_path = untracked["repo"], untracked["protocol_path"]
    assert isinstance(untracked_repo, Path) and isinstance(untracked_path, Path)
    _git(untracked_repo, "rm", "--cached", untracked_path.relative_to(untracked_repo).as_posix())
    _git(untracked_repo, "commit", "-qm", "remove registered protocol")
    with pytest.raises(ValueError, match="HEAD에 커밋되지 않았다"):
        J.validate_protocol(untracked_path, repo=untracked_repo)

    later = _fixture(tmp_path / "later", monkeypatch)
    later_repo = later["repo"]
    assert isinstance(later_repo, Path)
    (later_repo / "unrelated.txt").write_text("later\n", encoding="utf-8")
    _git(later_repo, "add", "unrelated.txt")
    _git(later_repo, "commit", "-qm", "later unrelated change")
    assert J.validate_protocol(later["protocol_path"], repo=later_repo)["protocol_id"] == (
        "semantic-diagnostic-v1"
    )


def test_protocol_uses_pinned_mixed_history_for_metadata_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = {
        "owner": {
            "id": "3655a80oo", "name": "rl-triage-e2e-20261024-c9f4",
            "instance_type": "l40s-48gb.1x",
        },
        "allowed_inspection_script_sha256": ["a" * 64],
        "runs": {
            "metadata-run": {
                "envelope_sha256": "b" * 64,
                "start_receipt_sha256": "c" * 64,
                "stop_receipt_sha256": "d" * 64,
            },
        },
    }
    fixture = _fixture(tmp_path, monkeypatch, metadata_contract=contract)
    repo = fixture["repo"]
    protocol = fixture["protocol"]
    protocol_path = fixture["protocol_path"]
    assert isinstance(repo, Path) and isinstance(protocol, dict) and isinstance(protocol_path, Path)

    history_path = repo / J.HISTORY_PROTOCOL_PATH
    history_sha = _sha(history_path, lf=True)
    monkeypatch.setattr(J, "HISTORY_PROTOCOL_SHA256_LF", history_sha)

    metadata_path = repo / "metadata-envelope.json"
    _write(metadata_path, {"schema": J.METADATA_LIFECYCLE_SCHEMA})
    references = [
        *protocol["resources"]["prior_stop_receipts"],
        {"path": "metadata-envelope.json", "sha256": _sha(metadata_path)},
    ]
    protocol["resources"]["prior_stop_receipts"] = references
    _write(protocol_path, protocol)
    _commit_protocol(fixture, "register metadata protocol")

    captured: dict[str, object] = {}

    def validate_mixed(resources: dict[str, object], source: dict[str, object], root: Path):
        captured.update(resources=resources, source=source, repo=root)
        return datetime(2026, 10, 26, tzinfo=timezone.utc), set()  # noqa: UP017

    monkeypatch.setattr(J.ATOMIC, "_validate_mixed_prior_stops", validate_mixed)
    J.validate_protocol(protocol_path, repo=repo)
    mixed = captured["resources"]
    assert isinstance(mixed, dict)
    assert mixed["prior_stop_receipts"] == references
    assert mixed["prior_lifecycle_compute_usd"] == "8.500674"
    assert mixed["metadata_lifecycle_contract"] == contract
    assert captured["source"] == fixture["parent"]
    assert captured["repo"] == repo.resolve()

    monkeypatch.setattr(J, "HISTORY_PROTOCOL_SHA256_LF", "0" * 64)
    with pytest.raises(ValueError, match="source LF SHA"):
        J.validate_protocol(protocol_path, repo=repo)


def test_job_freshness_can_be_disabled_only_for_completion_recheck(tmp_path: Path, monkeypatch) -> None:
    fixture = _job_fixture(tmp_path, monkeypatch)
    stale = fixture["now"] + timedelta(seconds=31)
    with pytest.raises(ValueError, match="30초 freshness"):
        J.validate_job(fixture["job_path"], now=stale)
    (fixture["job_path"].parent / "output").mkdir()
    assert J.validate_job(fixture["job_path"], now=stale, require_fresh=False)["resource"]


def test_job_rejects_output_escape_resource_tamper_and_runtime_pin(
    tmp_path: Path, monkeypatch,
) -> None:
    fixture = _job_fixture(tmp_path, monkeypatch)
    job = fixture["job"]
    assert isinstance(job, dict)
    job["output_directory"] = str((tmp_path / "outside").resolve())
    _write(fixture["job_path"], job)
    with pytest.raises(ValueError, match="job 옆 output"):
        J.validate_job(fixture["job_path"], now=fixture["now"])

    fixture2 = _job_fixture(tmp_path / "resource", monkeypatch)
    fixture2["resource"].write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="resource proof SHA"):
        J.validate_job(fixture2["job_path"], now=fixture2["now"])

    fixture3 = _job_fixture(tmp_path / "runtime", monkeypatch)
    (fixture3["lab"] / "runtime.py").write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="runtime source SHA"):
        J.validate_job(fixture3["job_path"], now=fixture3["now"])


def test_job_rejects_source_ref_tamper_and_symlink(tmp_path: Path, monkeypatch) -> None:
    fixture = _job_fixture(tmp_path, monkeypatch)
    job = fixture["job"]
    assert isinstance(job, dict)
    job["source"]["files_lf"]["bench/parent_source.py"] = "0" * 64
    _write(fixture["job_path"], job)
    with pytest.raises(ValueError, match="job source"):
        J.validate_job(fixture["job_path"], now=fixture["now"])

    fixture2 = _job_fixture(tmp_path / "link", monkeypatch)
    protocol_path = fixture2["protocol_path"]
    assert isinstance(protocol_path, Path)
    linked = tmp_path / "protocol-parent-link"
    if os.name == "nt":
        script = tmp_path / "create-junction.ps1"
        script.write_text(
            """param(
    [Parameter(Mandatory=$true)][string]$JunctionPath,
    [Parameter(Mandatory=$true)][string]$TargetPath
)
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
New-Item -ItemType Junction -Path $JunctionPath -Target $TargetPath | Out-Null
""",
            encoding="utf-8-sig",
        )
        subprocess.run(
            ["powershell.exe", "-NoLogo", "-NoProfile", "-NonInteractive",
             "-ExecutionPolicy", "Bypass", "-File", str(script),
             "-JunctionPath", str(linked), "-TargetPath", str(protocol_path.parent)],
            check=True, capture_output=True, text=True, encoding="utf-8", errors="strict",
        )
        cleanup = linked.rmdir
    else:
        linked.symlink_to(protocol_path.parent, target_is_directory=True)
        cleanup = linked.unlink
    raw = json.loads(fixture2["job_path"].read_text(encoding="utf-8"))
    raw["protocol"]["path"] = str((linked / protocol_path.name).absolute())
    fixture2["job_path"].unlink()
    _write(fixture2["job_path"], raw)
    try:
        with pytest.raises(ValueError, match="symlink/reparse|regular"):
            J.validate_job(fixture2["job_path"], now=fixture2["now"])
    finally:
        cleanup()


def test_write_new_is_canonical_finite_and_write_once(tmp_path: Path) -> None:
    output = tmp_path / "nested/result.json"
    J.write_new(output, {"b": 2, "a": 1})
    assert output.read_bytes() == b'{"a":1,"b":2}\n'
    with pytest.raises(FileExistsError):
        J.write_new(output, {"other": True})
    with pytest.raises(ValueError):
        J.write_new(tmp_path / "nan.json", {"value": float("nan")})


def test_validation_deadline_clamps_git_rejects_expiry_and_resets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    timeouts: list[float] = []

    def run(*args, **kwargs):
        timeouts.append(kwargs["timeout"])
        return subprocess.CompletedProcess(args[0], 0, "", "")

    monkeypatch.setattr(J.subprocess, "run", run)
    monkeypatch.setattr(J.time, "monotonic", lambda: 100.0)
    with J.validation_deadline(105.0):
        J._git(tmp_path, "status")
    assert timeouts == [5.0]

    with J.validation_deadline(99.0), pytest.raises(TimeoutError, match="deadline"):
        J._git(tmp_path, "status")
    assert timeouts == [5.0]

    readings = iter((100.0, 106.0))
    monkeypatch.setattr(J.time, "monotonic", lambda: next(readings))
    with J.validation_deadline(105.0), pytest.raises(TimeoutError, match="deadline"):
        J._git(tmp_path, "status")

    monkeypatch.setattr(J.time, "monotonic", lambda: 500.0)
    J._git(tmp_path, "status")
    assert timeouts == [5.0, 5.0, 30.0]


@pytest.mark.skipif(os.name == "nt", reason="POSIX venv Python은 말단 symlink를 사용한다")
def test_python_executable_preserves_posix_venv_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "python-real"
    target.write_bytes(Path(sys.executable).read_bytes())
    target.chmod(0o755)
    venv_python = tmp_path / "venv/bin/python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(target)
    monkeypatch.setattr(J.sys, "executable", str(venv_python.absolute()))
    assert J._python_executable(venv_python.absolute()) == venv_python.absolute()

    other = tmp_path / "other-python"
    other.write_bytes(target.read_bytes())
    other.chmod(0o755)
    monkeypatch.setattr(J.sys, "executable", str(other.absolute()))
    with pytest.raises(ValueError, match="현재 실행 Python"):
        J._python_executable(venv_python.absolute())
