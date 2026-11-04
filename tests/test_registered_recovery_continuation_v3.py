import json
import subprocess
from pathlib import Path

import pytest

from evals import registered_recovery_continuation_v3 as continuation


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def test_historical_closure_uses_committed_blob_when_current_file_changed(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "fixture@example.test")
    _git(repo, "config", "user.name", "fixture")
    source = repo / "source.py"
    source.write_text("old = True\n", encoding="utf-8")
    _git(repo, "add", "source.py")
    _git(repo, "commit", "-m", "fixture")
    anchor = _git(repo, "rev-parse", "HEAD")
    old_sha = continuation._sha(source, lf=True)
    source.write_text("old = False\n", encoding="utf-8")

    result = continuation._historical_closure(repo, anchor, {"source.py": old_sha})

    assert result["historical_source_authoritative"] is True
    assert result["historical_files_verified"] == 1
    assert result["current_difference_paths"] == ["source.py"]


def test_reference_mutation_and_write_once_are_rejected(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    _write(source, {"value": 1})
    reference = continuation._ref(source)
    source.write_text('{"value":2}\n', encoding="utf-8")
    with pytest.raises(continuation.ContinuationProofError, match="reference changed"):
        continuation._validate_ref(reference, label="source")

    output = tmp_path / "proof.json"
    continuation._write_new(output, {"schema": "fixture"})
    with pytest.raises(FileExistsError):
        continuation._write_new(output, {"schema": "replacement"})


def test_completed_training_builder_exposes_exact_static_identity(tmp_path: Path) -> None:
    stage = tmp_path / "stage"
    recovery = tmp_path / "recovery"
    external = tmp_path / "external"
    external.mkdir()
    refs = {}
    for name in ("source.pt", "source-env.yaml", "source-agent.yaml", "final.pt", "env.yaml", "agent.yaml"):
        path = external / name
        path.write_bytes(name.encode())
        refs[name] = continuation._ref(path)
    binding = {
        "source_commit_sha": continuation.SOURCE_ANCHOR,
        "seed": 2664,
        "task": "registered-task",
        "resume_checkpoint": refs["source.pt"],
        "resume_env_config": refs["source-env.yaml"],
        "resume_agent_config": refs["source-agent.yaml"],
    }
    _write(stage / "binding.json", binding)
    _write(stage / "manifest.json", {"run_name": "request-1", "correction_id": continuation.CORRECTION})
    _write(stage / "receipt.json", {"checkpoint": refs["final.pt"], "configs": {
        "environment": refs["env.yaml"], "agent": refs["agent.yaml"]}})
    for relative in set(continuation._STAGE_FILES) - {"binding.json", "manifest.json", "receipt.json"}:
        _write(stage / relative, {"fixture": relative})
    for relative in continuation._RECOVERY_FILES:
        if relative == "manifest.json":
            _write(recovery / relative, {"recovery_context": {"protocol_body": {
                "diagnostic": {"session_manifest_sha256": "d" * 64}}}})
        elif relative.endswith(".jsonl"):
            (recovery / relative).parent.mkdir(parents=True, exist_ok=True)
            (recovery / relative).write_text("{}\n", encoding="utf-8")
        else:
            _write(recovery / relative, {"fixture": relative})
    for relative in ("approvals/request-1.json", "admissions/request-1.json",
                     "intents/request-1.json", "resources/request-1.json",
                     "dispatch_errors/request-1.json"):
        _write(recovery / relative, {"fixture": relative})

    proof_ref = continuation.build_completed_training_proof(
        stage, recovery, tmp_path / "completed-training.json"
    )
    proof = continuation._json(Path(proof_ref["path"]))

    assert proof["schema"] == continuation.TRAINING_SCHEMA
    assert proof["case_id"] == "case-05" and proof["method"] == "static_rule"
    assert proof["identity"] == {
        "diagnostic_manifest_sha256": "d" * 64,
        "source_checkpoint_sha256": refs["source.pt"]["sha256"],
        "final_checkpoint_sha256": refs["final.pt"]["sha256"],
        "seed": 2664,
        "task": "registered-task",
        "correction_id": continuation.CORRECTION,
        "source_anchor": continuation.SOURCE_ANCHOR,
        "case_id": "case-05",
        "method": "static_rule",
    }
    assert proof["claims"]["new_training_launched"] is False
