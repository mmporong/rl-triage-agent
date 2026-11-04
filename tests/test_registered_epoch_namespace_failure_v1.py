from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals import registered_epoch_namespace_failure_v1 as failure


def _fixture(tmp_path: Path) -> dict:
    refs = {}
    for role, sha in failure.REFERENCE_SHA256.items():
        path = tmp_path / f"{role}.json"
        path.write_bytes(role.encode())
        refs[role] = {"path": str(path), "sha256": sha}
    body = {
        "schema": "registered_run_name_collision_evidence_v1",
        "status": "pre_sdk_infrastructure_failure",
        "source_anchor": failure.SOURCE_ANCHOR,
        "registration_anchor": failure.REGISTRATION_ANCHOR,
        "case_id": "case-07", "method": "one_shot_llm",
        "request_id": failure.REQUEST_ID, "registered_training_seed": 6066,
        "planned_next_restart_seeds_not_executed": [6266, 6267],
        "training_executed": False, "fixed_evaluation_executed": False,
        "same_consumed_request_reexecute_allowed": False,
        "prior_record_preserved": True, "SDK": 0, "API": 0,
        "CPU_helper_children": 0,
        "observed_error": {"error_type": "FileExistsError", "reexecute_allowed": False},
        "references": refs,
    }
    path = tmp_path / "context.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    return {"path": str(path), "sha256": failure.CONTEXT_SHA256}


def test_actual_registered_context_verifies() -> None:
    root = Path(__file__).resolve().parents[1]
    path = (root / "bench/private/claude_reward_track_20261013/codex_main_20261025/vision_goal"
            / "ri_upgrade_451_20261104/v12_run_name_failure_528/context.json")
    verified = failure.verify_failure_context(
        {"path": str(path), "sha256": failure.CONTEXT_SHA256})
    assert verified["context"]["SDK"] == 0


def test_context_sha_and_raw_reference_are_required(tmp_path: Path, monkeypatch) -> None:
    reference = _fixture(tmp_path)
    original = failure._sha

    def fake_sha(path: Path) -> str:
        if path.name == "context.json":
            return failure.CONTEXT_SHA256
        return failure.REFERENCE_SHA256[path.stem]

    monkeypatch.setattr(failure, "_sha", fake_sha)
    assert failure.verify_failure_context(reference)["context"]["SDK"] == 0
    body = json.loads(Path(reference["path"]).read_text(encoding="utf-8"))
    body["references"].pop("error")
    Path(reference["path"]).write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(failure.NamespaceFailureError, match="raw references"):
        failure.verify_failure_context(reference)
    monkeypatch.setattr(failure, "_sha", original)
