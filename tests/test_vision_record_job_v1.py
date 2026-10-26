from __future__ import annotations

import shutil
import sys
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
import test_vision_prepared_job_v1 as F

from bench import vision_record_job_v1 as J


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    value = F._fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(J.PREPARED, "ANCHOR", deepcopy(F.J.ANCHOR))
    repo = value["repo"]
    for relative in J.REQUIRED_SOURCE_FILES - F.J.REQUIRED_SOURCE_FILES:
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {relative}\n", encoding="utf-8")
    F.F._git(repo, "add", ".")
    F.F._git(repo, "commit", "-qm", "fixture record sources")
    commit = F.F._git(repo, "rev-parse", "HEAD")
    lab = tmp_path / "lab"
    lab.mkdir()
    relative = "logs/rsl_rl/pilot/model_19.pt"
    checkpoint = lab / relative
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"test checkpoint bytes")
    body = value["protocol"]
    body.update(
        schema=J.PROTOCOL_SCHEMA,
        mode=J.MODE,
        limits=dict(J.LIMITS),
        source_commit=commit,
        source_sha256_lf={
            name: F.F._sha(repo / name, lf=True) for name in J.REQUIRED_SOURCE_FILES
        },
        checkpoints=[
            {
                "logical_id": "pilot",
                "isaaclab_relative": relative,
                "sha256": F.F._sha(checkpoint),
            }
        ],
        preregistration={
            "prediction": "개발 정책 녹화의 같은 실행·프레임 결합을 확인한다.",
            "observation_phrases": {
                key: key
                for key in (
                    "record_complete",
                    "callback_hold",
                    "alignment_hold",
                    "infrastructure_failure",
                )
            },
            "fixed_flags": dict(J.FIXED_FLAGS),
        },
    )
    path = repo / "bench/protocols/record.json"
    F.F._write(path, body)
    F.F._git(repo, "add", path.relative_to(repo).as_posix())
    F.F._git(repo, "commit", "-qm", "fixture record preregistration")
    now = datetime.now(timezone.utc)  # noqa: UP017 - Python 3.10
    resource = tmp_path / "resource.json"
    F.F._write(resource, {"issued_at_utc": now.isoformat()})
    python = tmp_path / "python"
    shutil.copyfile(sys.executable, python)
    monkeypatch.setattr(J.COMMON.sys, "executable", str(python))
    monkeypatch.setattr(J, "ROOT", repo)
    monkeypatch.setattr(
        J.SOURCE.SourceBatch, "_verify_resource", lambda _batch, raw: raw
    )
    value.update(
        protocol=body,
        protocol_path=path,
        commit=commit,
        lab=lab,
        checkpoint=checkpoint,
        resource=resource,
        python=python,
        now=now,
    )
    return value


def _save(value: dict[str, Any]) -> None:
    F.F._write(value["protocol_path"], value["protocol"])
    F.F._git(
        value["repo"],
        "add",
        value["protocol_path"].relative_to(value["repo"]).as_posix(),
    )
    F.F._git(value["repo"], "commit", "-qm", "fixture record update")


def _job(value: dict[str, Any]) -> Path:
    return J.create_job(
        value["protocol_path"],
        value["resource"],
        F.F._sha(value["resource"]),
        value["lab"],
        value["python"],
        value["repo"].parent / "record-stage",
        "pilot",
    )


def test_actual_job_creator_binds_registered_checkpoint_and_profile(
    tmp_path, monkeypatch
):
    value = _fixture(tmp_path, monkeypatch)
    path = _job(value)
    result = J.validate_job(path)
    assert result["checkpoint_path"] == value["checkpoint"]
    assert result["checkpoint_sha256"] == F.F._sha(value["checkpoint"])
    assert result["job"]["job_id"] == f"{value['protocol']['protocol_id']}-pilot"
    assert result["job"]["profile"] == J.RECORD.fixed_profile()
    assert result["protocol"]["limits"]["training_actions"] == 0
    assert result["protocol"]["limits"]["frames"] == 625
    with pytest.raises(FileExistsError):
        _job(value)
    with pytest.raises(ValueError, match="30초"):
        J.validate_job(path, now=value["now"] + timedelta(seconds=31))
    result["output_directory"].mkdir()
    assert J.validate_job(
        path, now=value["now"] + timedelta(seconds=31), require_fresh=False
    )


@pytest.mark.parametrize(
    "mutation",
    ["changed_bytes", "changed_path", "identity", "profile", "mode", "source", "proof"],
)
def test_job_rejects_unregistered_or_changed_evidence(tmp_path, monkeypatch, mutation):
    value = _fixture(tmp_path, monkeypatch)
    path = _job(value)
    job = J._read(path)
    if mutation == "changed_bytes":
        value["checkpoint"].write_bytes(b"different checkpoint")
    elif mutation == "changed_path":
        alternate = value["lab"] / "other.pt"
        shutil.copyfile(value["checkpoint"], alternate)
        job["checkpoint"]["path"] = str(alternate)
    elif mutation == "identity":
        job["job_id"] = "other"
    elif mutation == "profile":
        job["profile"]["camera"]["rt_subframes"] = 2
    elif mutation == "mode":
        job["mode"] = "record"
    elif mutation == "source":
        job["source"]["commit"] = "f" * 40
    else:
        job["resource_proof"]["sha256"] = "f" * 64
    F.F._write(path, job)
    with pytest.raises(ValueError):
        J.validate_job(path)


@pytest.mark.parametrize("mutation", ["duplicate", "path", "sha", "limits", "phrases"])
def test_protocol_rejects_invalid_registration(tmp_path, monkeypatch, mutation):
    value = _fixture(tmp_path, monkeypatch)
    body = value["protocol"]
    if mutation == "duplicate":
        body["checkpoints"].append(deepcopy(body["checkpoints"][0]))
    elif mutation == "path":
        body["checkpoints"][0]["isaaclab_relative"] = "../outside.pt"
    elif mutation == "sha":
        body["checkpoints"][0]["sha256"] = "short"
    elif mutation == "limits":
        body["limits"]["frames"] = 626
    else:
        phrases = body["preregistration"]["observation_phrases"]
        phrases["record_complete"] = phrases["callback_hold"]
    _save(value)
    with pytest.raises(ValueError):
        J.validate_protocol(value["protocol_path"], repo=value["repo"])


def test_record_history_can_append_a_record_plate_without_resetting_cost(
    tmp_path, monkeypatch
):
    value = _fixture(tmp_path, monkeypatch)
    previous = deepcopy(value["protocol"])
    previous_ref = {
        "path": value["protocol_path"].relative_to(value["repo"]).as_posix(),
        "sha256_lf": F.F._sha(value["protocol_path"], lf=True),
    }
    stop = {
        "status": "STOPPED",
        "owned_instance": {"id": "3655a80oo", "name": "rl-triage-e2e-20261024-c9f4"},
        "execution_identity": {
            "run_id": previous["protocol_id"],
            "protocol_sha256": previous["parent_source_protocol"]["sha256_lf"],
        },
        "cost_estimate": {"estimated_compute_cost_usd": "0.1"},
    }
    stop_path = value["repo"] / "third-stop.json"
    F.F._write(stop_path, stop)
    ref = {"path": "third-stop.json", "sha256": F.F._sha(stop_path)}
    current = deepcopy(previous)
    current["resources"].update(
        prior_compute_estimate_usd="9.085493", fresh_lifecycle_budget_usd="10.914507"
    )
    current["resources"]["prior_stop_receipts"].append(ref)
    current["predecessor"] = {"semantic": previous_ref, "appended_stop": ref}
    parent = deepcopy(value["parent"])
    parent["resources"]["whole_batch_budget_usd"] = "10.914507"
    commit = F.F._git(value["repo"], "rev-parse", "HEAD")
    J.validate_cumulative_resources(
        current, parent, repo=value["repo"], source_commit=commit
    )
    current["resources"]["prior_stop_receipts"].pop(0)
    with pytest.raises(ValueError, match="append-only"):
        J.validate_cumulative_resources(
            current, parent, repo=value["repo"], source_commit=commit
        )
