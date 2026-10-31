# pyright: reportMissingImports=false
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from evals import export_prospective_agentic_comparison_v2 as X


def write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return path


def eager_epoch(tmp_path: Path, *, host_digest: str = "d" * 64,
                omit_actual_after: bool = False) -> tuple[Path, dict[str, Any]]:
    root = tmp_path / "comparison"
    epoch_root = root / "c01/eager_model"
    before = {"state_digest": "d" * 64, "history": [X.HOST_EAGER_ACTION],
              "status": "open"}
    after = {"state_digest": "e" * 64, "history": [X.HOST_EAGER_ACTION],
             "status": "confirmed", "hypotheses": ["reward"],
             "observed": {"P_reward": "abnormal"}}
    dialogue = {"calls": [{"review": {"status": "valid"}}], "model_valid": True,
                "semantic_feedback_used": False}
    dialogue_path = write(epoch_root / "step-02/dialogue.json", dialogue)
    if not omit_actual_after:
        write(epoch_root / "step-02/state-after.json", after)
    write(epoch_root / "final-state.json", after)
    epoch = {
        "schema": "prospective_agentic_epoch_v1", "status": "complete",
        "policy": "eager_model", "decision_outcome": "confirmed",
        "confirmed_family": "reward", "request_count": 1, "model_actions": 1,
        "probe_count": 1, "parent_probe_wall_s": 2.0, "owned_child_wall_s": 1.0,
        "final_status": "confirmed", "remaining_candidates": ["reward"],
        "observations": {"P_reward": "abnormal"},
        "steps": [
            {"kind": "host_eager_exposure", "action": X.HOST_EAGER_ACTION,
             "state_digest": host_digest, "model_request": False},
            {"state_before": before, "dialogue_sha256": X.V1._sha(dialogue_path),
             "state_after_digest": "e" * 64},
        ],
    }
    write(epoch_root / "result.json", epoch)
    score = X.RUN._score("F1", epoch)
    validity = X.RUN._validity(epoch_root, epoch)
    row = {"case_id": "c01", "policy": "eager_model", "gate": {"run": True},
           "decision_outcome": "confirmed", "model_requests": 1, "model_actions": 1,
           "paired_probes": 1, "parent_probe_wall_s": 2.0, "owned_child_wall_s": 1.0,
           **score, **validity}
    return root, row


def test_exact_eager_host_exposure_replays_into_first_actual_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, row = eager_epoch(tmp_path)
    monkeypatch.setattr(X.SEL, "context", lambda state: {})
    epoch, dialogues = X._verify_epoch(root, row, "F1")
    assert epoch is not None
    assert [path.parent.name for path in dialogues] == ["step-02"]


def test_eager_host_digest_mismatch_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, row = eager_epoch(tmp_path, host_digest="x" * 64)
    monkeypatch.setattr(X.SEL, "context", lambda state: {})
    with pytest.raises(ValueError, match="digest"):
        X._verify_epoch(root, row, "F1")


def test_missing_state_after_for_actual_step_is_not_allowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, row = eager_epoch(tmp_path, omit_actual_after=True)
    monkeypatch.setattr(X.SEL, "context", lambda state: {})
    with pytest.raises(ValueError, match="일반 step state-after"):
        X._verify_epoch(root, row, "F1")


def test_non_eager_missing_first_state_after_is_not_allowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, row = eager_epoch(tmp_path)
    epoch_root = root / "c01/eager_model"
    row["policy"] = "evidence_guided_model"
    target = root / "c01/evidence_guided_model"
    target.parent.mkdir(parents=True, exist_ok=True)
    epoch_root.rename(target)
    value = json.loads((target / "result.json").read_text(encoding="utf-8"))
    value["policy"] = "evidence_guided_model"
    write(target / "result.json", value)
    monkeypatch.setattr(X.SEL, "context", lambda state: {})
    with pytest.raises(ValueError, match="eager host exposure"):
        X._verify_epoch(root, row, "F1")


def test_cli_requires_fresh_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict[str, object] = {}
    monkeypatch.setattr(X, "export", lambda *args, **kwargs: called.update(kwargs))
    assert X.main(["--source-batch", str(tmp_path / "source.json"),
                   "--comparison", str(tmp_path / "comparison.json"),
                   "--protocol", str(tmp_path / "protocol.json"),
                   "--anchor", "a" * 40, "--out", str(tmp_path / "out")]) == 0
    assert called["anchor"] == "a" * 40


def test_script_help_bootstraps_repository_package() -> None:
    result = subprocess.run(
        [sys.executable, "-B",
         str(X.ROOT / "evals/export_prospective_agentic_comparison_v2.py"), "--help"],
        cwd=X.ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
        check=False)
    assert result.returncode == 0, result.stderr
    assert "--comparison" in result.stdout
