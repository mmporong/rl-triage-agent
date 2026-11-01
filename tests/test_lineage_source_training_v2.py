# pyright: reportMissingImports=false
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
import lineage_source_training_v1 as V1
import lineage_source_training_v2 as T


def _job(index: int = 0, *, purpose: str = "source", arm: str = "NONE") -> dict:
    return {"job_id": f"job-{index}", "case_id": f"case-{index}", "arm": arm,
            "seed": 100 + index, "run_name": f"lineage_v2_{index}", "task": T.TASK,
            "num_envs": 4096, "max_iterations": 1 if purpose == "smoke" else 300}


def _manifest(*, purpose: str = "source", arm: str = "NONE") -> dict:
    return {"schema": T.SCHEMA, "purpose": purpose, **_job(arm=arm, purpose=purpose),
            "source_commit": "a" * 40, "protocol_relative": "protocol.json",
            "protocol_sha256_lf": "b" * 64,
            "source_sha256_lf": {name: "c" * 64 for name in T.SOURCE_FILES},
            "runtime_source_sha256_lf": {"runtime.py": "d" * 64}}


def test_v2_contract_requires_twenty_jobs_and_keeps_v1_globals_unchanged() -> None:
    value = _manifest()
    protocol = {"schema": T.PROTOCOL_SCHEMA, "smoke_job": _job(purpose="smoke"),
                "training_jobs": [_job(index) for index in range(20)]}
    assert T._protocol_job(protocol, value)["job_id"] == value["job_id"]
    protocol["training_jobs"].pop()
    with pytest.raises(ValueError, match="20개"):
        T._protocol_job(protocol, value)
    clone = T._isolated_v1()
    assert clone.SCHEMA == T.SCHEMA
    assert clone.validate_manifest is T.validate_manifest
    assert V1.SCHEMA != T.SCHEMA
    assert V1.validate_manifest is not T.validate_manifest


@pytest.mark.parametrize("change", ["source", "budget", "smoke_fault", "extra"])
def test_manifest_rejects_missing_closure_and_policy_changes(change: str) -> None:
    value = _manifest(purpose="smoke" if change == "smoke_fault" else "source")
    if change == "source":
        value["source_sha256_lf"].pop(T.SOURCE_FILES[0])
    elif change == "budget":
        value["max_iterations"] -= 1
    elif change == "smoke_fault":
        value["arm"] = "F2"
    else:
        value["resume"] = True
    with pytest.raises(ValueError):
        T.validate_manifest(value)


def test_validate_only_does_not_execute_training(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(_manifest()), encoding="utf-8")
    assert T.main(["--manifest", str(manifest), "--lab", str(tmp_path),
                   "--out", str(tmp_path / "out"), "--validate-only"]) == 0
    assert not (tmp_path / "out").exists()
