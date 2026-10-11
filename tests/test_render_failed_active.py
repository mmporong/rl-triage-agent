from pathlib import Path

import pytest

from evals import render_failed_active as R


def test_public_error_redacts_plain_and_repr_windows_paths(monkeypatch):
    monkeypatch.setattr(R, "ROOT", Path("C:/repo"))
    assert R.redact("C:/repo/evals/a.json") == "evals/a.json"
    # On Windows Path string separators are backslashes, including repr escapes.
    root_text = str(R.ROOT)
    assert R.redact(root_text + "\\evals\\a.json") == "evals/a.json"
    doubled = root_text.replace("\\", "\\\\") + "\\\\evals\\\\a.json"
    assert R.redact(doubled) == "evals/a.json"


def test_public_error_rejects_unmapped_absolute_path():
    for message in ("C:\\other\\secret.json", "/tmp/secret.json", "missing '/home/user/secret'"):
        with pytest.raises(ValueError):
            R.reject_absolute_paths({"errors": [message]})


def test_failed_publication_removes_only_new_artifacts(tmp_path, monkeypatch):
    first, second = tmp_path / "audit.json", tmp_path / "summary.json"
    original = Path.open

    def open_path(path, *args, **kwargs):
        if path == second:
            raise OSError("disk failure")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_path)
    with pytest.raises(OSError):
        R.publish_new({first: "new", second: "new"})
    assert not first.exists() and not second.exists()


def test_publication_never_replaces_existing_artifact(tmp_path):
    existing, new = tmp_path / "audit.json", tmp_path / "summary.json"
    existing.write_text("preserved", encoding="utf-8")
    with pytest.raises(FileExistsError):
        R.publish_new({new: "new", existing: "replacement"})
    assert existing.read_text(encoding="utf-8") == "preserved"
    assert not new.exists()
