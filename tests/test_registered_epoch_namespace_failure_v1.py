from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals import registered_epoch_namespace_failure_v1 as failure
from tests.registered_failure_evidence_fixture_v1 import namespace_fixture


def test_registered_context_contract_verifies_with_synthetic_raw(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = namespace_fixture(tmp_path, monkeypatch)
    verified = failure.verify_failure_context(reference)
    assert verified["context"]["SDK"] == 0


def test_context_sha_and_raw_reference_are_required(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = namespace_fixture(tmp_path, monkeypatch)
    assert failure.verify_failure_context(reference)["context"]["SDK"] == 0
    body = json.loads(Path(reference["path"]).read_text(encoding="utf-8"))
    body["references"].pop("error")
    Path(reference["path"]).write_text(json.dumps(body), encoding="utf-8")
    changed_sha = failure._sha(Path(reference["path"]))
    monkeypatch.setattr(failure, "CONTEXT_SHA256", changed_sha)
    with pytest.raises(failure.NamespaceFailureError, match="raw references"):
        failure.verify_failure_context({"path": reference["path"], "sha256": changed_sha})
