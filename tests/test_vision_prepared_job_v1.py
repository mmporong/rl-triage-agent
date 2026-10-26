from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
import test_vision_semantic_job_v1 as F

from bench import vision_prepared_job_v1 as J


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    repo = tmp_path / "repo"
    repo.mkdir(parents=True)
    F._git(repo, "init", "-q")
    F._git(repo, "config", "user.email", "test@example.com")
    F._git(repo, "config", "user.name", "test")
    for name in J.REQUIRED_SOURCE_FILES:
        target = repo / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"# {name}\n", encoding="utf-8")

    def parent(name: str, fresh: str) -> dict[str, Any]:
        return {
            "protocol_id": name,
            "source_sha256_lf": {
                "bench/vision_prepared_job_v1.py": F._sha(
                    repo / "bench/vision_prepared_job_v1.py", lf=True
                )
            },
            "runtime_source_sha256_lf": {},
            "resources": {
                "owned_instance_id": "3655a80oo",
                "max_owned_instances": 1,
                "compute_quote_usd_per_h": "1.74",
                "whole_batch_budget_usd": fresh,
            },
        }

    old_parent_path = repo / "bench/protocols/old_source.json"
    F._write(old_parent_path, parent("source-previous", "11.267092"))
    parent_path = repo / "bench/protocols/source.json"
    current_parent = parent("source-current", "11.014507")
    F._write(parent_path, current_parent)
    old_stop = {
        "status": "STOPPED",
        "owned_instance": {"id": "3655a80oo", "name": "rl-triage-e2e-20261024-c9f4"},
        "execution_identity": {"run_id": "historical"},
        "cost_estimate": {"estimated_compute_cost_usd": "8.732908"},
    }
    F._write(repo / "old-stop.json", old_stop)
    old_ref = {"path": "old-stop.json", "sha256": F._sha(repo / "old-stop.json")}
    stop = deepcopy(old_stop)
    stop["execution_identity"].update(
        run_id="previous-diagnostic", protocol_sha256=F._sha(old_parent_path, lf=True)
    )
    stop["cost_estimate"]["estimated_compute_cost_usd"] = "0.252585"
    F._write(repo / "new-stop.json", stop)
    stop_ref = {"path": "new-stop.json", "sha256": F._sha(repo / "new-stop.json")}
    resources = {
        "owned_instance_id": "3655a80oo",
        "owned_instance_name": "rl-triage-e2e-20261024-c9f4",
        "compute_quote_usd_per_h": "1.74",
        "max_owned_instances": 1,
        "prior_compute_estimate_usd": "8.732908",
        "fresh_lifecycle_budget_usd": "11.267092",
        "total_program_budget_usd": "20",
        "prior_stop_receipts": [old_ref],
    }
    previous = {
        "schema": J.COMMON.PROTOCOL_SCHEMA,
        "protocol_id": "previous-diagnostic",
        "parent_source_protocol": {
            "path": old_parent_path.relative_to(repo).as_posix(),
            "sha256_lf": F._sha(old_parent_path, lf=True),
        },
        "resources": deepcopy(resources),
    }
    anchor_path = repo / J.ANCHOR["path"]
    F._write(anchor_path, previous)
    anchor = {
        "path": anchor_path.relative_to(repo).as_posix(),
        "sha256_lf": F._sha(anchor_path, lf=True),
    }
    monkeypatch.setattr(J, "ANCHOR", anchor)
    F._git(repo, "add", ".")
    F._git(repo, "commit", "-qm", "fixture source")
    commit = F._git(repo, "rev-parse", "HEAD")
    resources.update(
        prior_compute_estimate_usd="8.985493",
        fresh_lifecycle_budget_usd="11.014507",
        prior_stop_receipts=[old_ref, stop_ref],
    )
    protocol = {
        "schema": J.PROTOCOL_SCHEMA,
        "protocol_id": "prepared-new",
        "mode": J.MODE,
        "registered_before_measurement": True,
        "source_commit": commit,
        "source_sha256_lf": {
            name: F._sha(repo / name, lf=True) for name in J.REQUIRED_SOURCE_FILES
        },
        "parent_source_protocol": {
            "path": parent_path.relative_to(repo).as_posix(),
            "sha256_lf": F._sha(parent_path, lf=True),
        },
        "profile_sha256": J.RECORD.digest(J.RECORD.fixed_profile()),
        "limits": dict(J.LIMITS),
        "resources": resources,
        "predecessor": {"semantic": anchor, "appended_stop": stop_ref},
        "preregistration": {
            "prediction": "개발 smoke: 준비 뒤 네 장을 기존 기준으로 검증한다.",
            "observation_phrases": {
                key: key
                for key in (
                    "sync_pass",
                    "callback_hold",
                    "sync_hold",
                    "infrastructure_failure",
                )
            },
            "fixed_flags": dict(J.FIXED_FLAGS),
        },
    }
    path = repo / "bench/protocols/prepared.json"
    F._write(path, protocol)
    F._git(repo, "add", path.relative_to(repo).as_posix())
    F._git(repo, "commit", "-qm", "fixture preregistration")
    monkeypatch.setattr(J.SOURCE, "validate_protocol", lambda value: value)
    return {
        "repo": repo,
        "protocol": protocol,
        "protocol_path": path,
        "commit": commit,
        "parent": current_parent,
    }


def _save(value: dict[str, Any]) -> None:
    F._write(value["protocol_path"], value["protocol"])
    F._git(
        value["repo"],
        "add",
        value["protocol_path"].relative_to(value["repo"]).as_posix(),
    )
    F._git(value["repo"], "commit", "-qm", "fixture update")


def test_protocol_keeps_source_and_append_only_budget(tmp_path, monkeypatch):
    value = _fixture(tmp_path, monkeypatch)
    observed = J.validate_protocol(value["protocol_path"], repo=value["repo"])
    assert observed["resources"]["prior_compute_estimate_usd"] == "8.985493"
    assert observed["limits"]["logical_frames"] == 4
    assert (
        observed["limits"]["training_actions"]
        == observed["limits"]["model_api_requests"]
        == 0
    )
    (value["repo"] / "unrelated.txt").write_text("unrelated\n", encoding="utf-8")
    F._git(value["repo"], "add", "unrelated.txt")
    F._git(value["repo"], "commit", "-qm", "unrelated")
    assert (
        J.validate_protocol(value["protocol_path"], repo=value["repo"])["source_commit"]
        == value["commit"]
    )


@pytest.mark.parametrize(
    "change",
    [
        "omit",
        "reverse",
        "duplicate",
        "rebind_run",
        "rebind_protocol",
        "cost",
        "anchor",
        "extra",
        "negative",
        "nan",
        "zero",
        "bool",
        "owner",
        "quote",
        "limit",
        "flags",
        "phrases",
        "source",
    ],
)
def test_rejects_history_budget_or_contract_mutation(tmp_path, monkeypatch, change):
    value = _fixture(tmp_path, monkeypatch)
    body, repo = value["protocol"], value["repo"]
    resources = body["resources"]
    if change == "omit":
        resources["prior_stop_receipts"] = resources["prior_stop_receipts"][1:]
        resources.update(
            prior_compute_estimate_usd="0.252585",
            fresh_lifecycle_budget_usd="19.747415",
        )
        # parent fresh를 바꿔도 history 자체가 누락을 거부한다.
        with pytest.raises(ValueError, match="append-only"):
            J._history(body, repo, value["commit"], set())
        return
    if change == "reverse":
        resources["prior_stop_receipts"].reverse()
    elif change == "duplicate":
        resources["prior_stop_receipts"].append(resources["prior_stop_receipts"][-1])
    elif change in {"rebind_run", "rebind_protocol", "cost"}:
        stop = J._read(repo / "new-stop.json")
        if change == "rebind_run":
            stop["execution_identity"]["run_id"] = "different"
        elif change == "rebind_protocol":
            stop["execution_identity"]["protocol_sha256"] = "f" * 64
        else:
            stop["cost_estimate"]["estimated_compute_cost_usd"] = "0.252584"
        F._write(repo / "new-stop.json", stop)
        digest = F._sha(repo / "new-stop.json")
        body["predecessor"]["appended_stop"]["sha256"] = digest
        resources["prior_stop_receipts"][-1]["sha256"] = digest
    elif change == "anchor":
        body["predecessor"]["semantic"]["sha256_lf"] = "f" * 64
    elif change == "extra":
        resources["allow_overspend"] = True
    elif change == "negative":
        resources["prior_compute_estimate_usd"] = "-1"
    elif change == "nan":
        resources["fresh_lifecycle_budget_usd"] = "NaN"
    elif change == "zero":
        resources["fresh_lifecycle_budget_usd"] = "0"
    elif change == "bool":
        resources["max_owned_instances"] = True
    elif change == "owner":
        resources["owned_instance_id"] = "other"
    elif change == "quote":
        resources["compute_quote_usd_per_h"] = "1.75"
    elif change == "limit":
        body["limits"]["events"] = 6
    elif change == "flags":
        body["preregistration"]["fixed_flags"]["used_for_decision"] = True
    elif change == "phrases":
        phrases = body["preregistration"]["observation_phrases"]
        phrases["sync_hold"] = phrases["sync_pass"]
    else:
        (repo / "bench/vision_prepared_job_v1.py").write_text(
            "changed\n", encoding="utf-8"
        )
    if change != "source":
        _save(value)
    with pytest.raises(ValueError):
        J.validate_protocol(value["protocol_path"], repo=repo)


def test_rejects_dirty_protocol_and_uncommitted_source(tmp_path, monkeypatch):
    value = _fixture(tmp_path, monkeypatch)
    value["protocol"]["protocol_id"] = "dirty"
    F._write(value["protocol_path"], value["protocol"])
    with pytest.raises(ValueError, match="HEAD committed"):
        J.validate_protocol(value["protocol_path"], repo=value["repo"])


def _job_fixture(tmp_path, monkeypatch):
    import shutil
    import sys

    value = _fixture(tmp_path, monkeypatch)
    now = datetime(2026, 10, 26, tzinfo=timezone.utc)  # noqa: UP017
    resource = tmp_path / "resource.json"
    F._write(resource, {"issued_at_utc": (now - timedelta(seconds=10)).isoformat()})
    lab = tmp_path / "lab"
    lab.mkdir()
    python = tmp_path / "python"
    shutil.copyfile(sys.executable, python)
    monkeypatch.setattr(J.COMMON.sys, "executable", str(python))
    stage = tmp_path / "stage"
    stage.mkdir()
    job = {
        "schema": J.JOB_SCHEMA,
        "mode": J.MODE,
        "job_id": value["protocol"]["protocol_id"],
        "source": {
            "repo_root": str(value["repo"]),
            "commit": value["commit"],
            "files_lf": value["protocol"]["source_sha256_lf"],
        },
        "protocol": {
            "path": str(value["protocol_path"]),
            "sha256_lf": F._sha(value["protocol_path"], lf=True),
        },
        "resource_proof": {"path": str(resource), "sha256": F._sha(resource)},
        "isaaclab": str(lab),
        "python": str(python),
        "output_directory": str(stage / "output"),
    }
    F._write(stage / "job.json", job)
    monkeypatch.setattr(
        J.SOURCE.SourceBatch, "_verify_resource", lambda _batch, raw: raw
    )
    value.update(job=job, job_path=stage / "job.json", now=now, resource=resource)
    return value


def test_job_admission_and_postflight_preserve_30_second_boundary(
    tmp_path, monkeypatch
):
    value = _job_fixture(tmp_path, monkeypatch)
    result = J.validate_job(value["job_path"], now=value["now"])
    assert result["job_sha256"] == F._sha(value["job_path"])
    with pytest.raises(ValueError, match="30초"):
        J.validate_job(value["job_path"], now=value["now"] + timedelta(seconds=21))
    result["output_directory"].mkdir()
    assert J.validate_job(
        value["job_path"], now=value["now"] + timedelta(seconds=31), require_fresh=False
    )
    with pytest.raises(FileExistsError):
        J.validate_job(value["job_path"], now=value["now"])


@pytest.mark.parametrize(
    "change", ["source", "proof", "output", "schema", "naive", "identity"]
)
def test_job_rejects_mutated_bindings(tmp_path, monkeypatch, change):
    value = _job_fixture(tmp_path, monkeypatch)
    job = value["job"]
    if change == "source":
        job["source"]["commit"] = "f" * 40
    elif change == "proof":
        job["resource_proof"]["sha256"] = "f" * 64
    elif change == "output":
        job["output_directory"] += "-elsewhere"
    elif change == "schema":
        job["mode"] = "record"
    elif change == "identity":
        job["job_id"] = "different-slug"
    else:
        value["now"] = value["now"].replace(tzinfo=None)
    F._write(value["job_path"], job)
    with pytest.raises(ValueError):
        J.validate_job(value["job_path"], now=value["now"])
