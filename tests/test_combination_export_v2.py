import json
from pathlib import Path
import shutil

import pytest

from evals import export_combination_trial_v2 as X


def test_public_result_verifies_and_reports_recovery():
    out = X.verify()
    assert out == {"status": "valid", "files": 30, "outcome": "recovered", "contract_status": "ready"}
    summary = json.loads((X.OUTPUT / "summary.json").read_text(encoding="utf-8"))
    assert summary["seeds"] == [2151, 3151]
    assert summary["model_calls"] == 0
    assert summary["selection_actor"] == "registered_rule"
    assert summary["source_gate"]["outcome"] == "failure_present"
    assert len(summary["restoration_targets"]) == 4


def test_public_files_have_no_absolute_paths():
    for path in X.OUTPUT.rglob("*"):
        if path.is_file():
            assert not X.ABSOLUTE.search(path.read_text(encoding="utf-8")), path.name


@pytest.mark.parametrize("relative", ["summary.json", "result.json", "family_restart/train/receipt.json"])
def test_tampered_public_file_is_rejected(tmp_path, relative):
    folder = tmp_path / "public"
    shutil.copytree(X.OUTPUT, folder)
    target = folder / relative
    original = target.read_bytes()
    target.write_bytes(original + b" ")
    assert target.read_bytes() != original
    with pytest.raises(ValueError):
        X.verify(folder)


def test_extra_public_file_is_rejected(tmp_path):
    folder = tmp_path / "public"
    shutil.copytree(X.OUTPUT, folder)
    (folder / "extra.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        X.verify(folder)


def test_scrub_replaces_known_roots_and_rejects_unknown_ones():
    prefixes = [("C:\\work\\repo", "<repo>")]
    assert X._scrub({"p": "C:\\work\\repo\\a\\b.json"}, prefixes) == {"p": "<repo>/a/b.json"}
    assert X._scrub(["plain"], prefixes) == ["plain"]
    with pytest.raises(ValueError):
        X._scrub("D:\\other\\file", prefixes)


def test_export_refuses_to_overwrite_existing_public_folder():
    with pytest.raises(FileExistsError):
        X.export(Path.home() / "IsaacLab")
