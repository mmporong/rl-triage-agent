from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest  # pyright: ignore[reportMissingImports]

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / "evals", ROOT / "bench", ROOT / "src"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

E = importlib.import_module("vision_record_selection_e2e_v2")


def test_e2e_v2_has_distinct_schema_and_v2_dependencies() -> None:
    assert E.SCHEMA == "vision_record_selection_e2e_plan_v2"
    assert E.API_CAP == 32
    assert E.RECORD_WORKER.__name__ == "vision_record_selection_worker_v2"
    assert E.RECORD_EXECUTION.__name__ == "vision_record_selection_execution_v2"
    assert E.READINESS.SCHEMA == "vision_record_source_readiness_v2"


def _write(path: Path, value: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ref(repo: Path, path: Path, *, frozen: bool = False) -> dict[str, str]:
    value = {"path": path.relative_to(repo).as_posix(), "sha256": _sha(path)}
    if frozen:
        value["status"] = "frozen"
    return value


def _case(tmp_path: Path) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any], str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    checkpoint_sha = "1" * 64
    training = _write(
        repo / "evidence/training-action.json",
        {
            "kind": "train",
            "job_id": "vision-s2220-f1",
            "artifact": {"checkpoint": {"sha256": checkpoint_sha}},
        },
    )
    record = _write(repo / "evidence/record-job.json", {"job_id": "record-s2220-f1"})
    readiness = _write(
        repo / "evidence/readiness.json",
        {
            "schema": E.READINESS.SCHEMA,
            "model_use_ready": True,
            "labels_inferred_from_recording": False,
            "training_action_raw": _ref(repo, training),
            "record_job": _ref(repo, record),
        },
    )
    manifest = _write(
        repo / "inputs/input-manifest.json",
        {"schema": "fixture", "model_readiness": _ref(repo, readiness)},
    )
    location = {"input_manifest": _ref(repo, manifest, frozen=True)}
    source = {
        "training_plan": [
            {"job_id": "vision-s2220-f1", "seed": 2220, "arm": "F1", "source_job_id": None},
            {
                "job_id": "vision-s2220-corrected",
                "seed": 2220,
                "arm": "CORRECTED",
                "source_job_id": "vision-s2220-f1",
            },
        ]
    }
    row = {
        "source_seed": 2220,
        "source_role": "registered-f1",
        "checkpoint_sha256": checkpoint_sha,
    }
    return repo, location, source, row, _sha(manifest)


def test_first_class_readiness_supplies_training_and_record_jobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, location, source, row, manifest_sha = _case(tmp_path)
    calls: list[Path] = []

    def verify(_repo: Path, path: Path, _source: dict[str, Any]) -> dict[str, Any]:
        calls.append(path)
        return {
            "checkpoint_sha256": row["checkpoint_sha256"],
            "source_seed": 2220,
            "source_role": "registered-f1",
        }

    monkeypatch.setattr(E.READINESS, "verify_model_readiness", verify)
    jobs = E._training_jobs(repo, location, source, row, manifest_sha)
    assert jobs == {
        "source_training_job_id": "vision-s2220-f1",
        "record_job_id": "record-s2220-f1",
        "corrected_job_id": "vision-s2220-corrected",
    }
    assert calls == [(repo / "evidence/readiness.json").resolve()]


def test_legacy_aliases_are_not_synthesized_or_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, location, source, row, _manifest_sha = _case(tmp_path)
    manifest_path = repo / location["input_manifest"]["path"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    readiness_path = repo / manifest["model_readiness"]["path"]
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    readiness["training_action"] = {"raw": readiness.pop("training_action_raw")}
    readiness["record"] = {"job": readiness.pop("record_job")}
    _write(readiness_path, readiness)
    manifest["model_readiness"] = _ref(repo, readiness_path)
    _write(manifest_path, manifest)
    location["input_manifest"] = _ref(repo, manifest_path, frozen=True)
    monkeypatch.setattr(
        E.READINESS,
        "verify_model_readiness",
        lambda *_args: {
            "checkpoint_sha256": row["checkpoint_sha256"],
            "source_seed": 2220,
            "source_role": "registered-f1",
        },
    )
    with pytest.raises(TypeError, match="training_action_raw/record_job"):
        E._training_jobs(repo, location, source, row, _sha(manifest_path))


def test_existing_hold_state_machine_and_new_execution_path_are_preserved() -> None:
    plan = {
        "schema": E.SCHEMA,
        "api_cap": E.API_CAP,
        "rows": [
            {
                "target": dict(E.TARGETS[0]),
                "call_id": "e2e-source-01",
                "source_seed": 2220,
                "checkpoint_sha256": "1" * 64,
                "eligible_probes": ["P_reward"],
            },
            {
                "target": dict(E.TARGETS[1]),
                "call_id": "e2e-source-02",
                "source_seed": 2221,
                "checkpoint_sha256": "2" * 64,
                "eligible_probes": ["P_reward"],
            },
        ],
    }
    state = {
        "api_used": 2,
        "selections": {
            "source-01": {"model_status": "invalid"},
            "source-02": {"model_status": "invalid"},
        },
    }
    assert E.commands(plan, state) == []
    assert E.execution_module() is E.RECORD_EXECUTION
    assert E.prepare_plan.__kwdefaults__["production_prepare"] is E.RECORD_WORKER.prepare
