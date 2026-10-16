"""새 공개 근거 검증과 기존 생성기 경계 보존을 확인한다."""
import json
from pathlib import Path
import shutil
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))
import build_anymal_doc_v2 as D


def test_existing_sections_are_preserved_and_registered_document_is_current():
    old = D.BASE.build(ROOT, D.DEMO)
    new = D.build(ROOT)
    assert new == (ROOT / D.OUTPUT).read_text(encoding="utf-8")
    old_start = old.index("## 바꾼 것")
    old_end = old.index("## 주장하지 않는 것")
    assert old[old_start:old_end] in new


def test_changed_evidence_is_rejected_before_document_output(tmp_path, monkeypatch):
    shutil.copytree(ROOT / D.PUBLIC, tmp_path / D.PUBLIC)
    path = tmp_path / D.PUBLIC / "summary.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["may_preregister_demo"] = True
    path.write_text(json.dumps(value), encoding="utf-8")
    monkeypatch.setattr(D, "ROOT", tmp_path)
    monkeypatch.setattr(D.BASE, "build", lambda *_: "## 주장하지 않는 것\n")
    assert D.main(["--output", "out.md"]) == 2
    assert not (tmp_path / "out.md").exists()


def test_check_detects_stale_document_without_writing(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "ROOT", tmp_path)
    monkeypatch.setattr(D, "build", lambda *_: "verified document\n")
    assert D.main(["--check"]) == 1
    assert not (tmp_path / D.OUTPUT).exists()
    assert D.main([]) == 0
    assert D.main(["--check"]) == 0


def test_missing_public_evidence_is_rejected(tmp_path):
    with pytest.raises((ValueError, FileNotFoundError)):
        D.smoke_section(tmp_path)
