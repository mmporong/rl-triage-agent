import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
import record_checkpoint_preview as R


def test_unverified_source_never_reaches_gpu_or_player(monkeypatch, tmp_path):
    def reject(_sdk):
        raise ValueError("source changed")
    monkeypatch.setattr(R, "verify", reject)
    monkeypatch.setattr(R.M, "query_gpu", lambda: pytest.fail("source failure reached GPU gate"))
    with pytest.raises(ValueError, match="source changed"):
        R.record("fresh", tmp_path)


def fixture(monkeypatch, tmp_path):
    monkeypatch.setattr(R, "ROOT", tmp_path)
    monkeypatch.setattr(R, "verify", lambda _sdk: (R.FIXED, tmp_path / "checkpoint", tmp_path / "params", "a" * 40))
    monkeypatch.setattr(R.M, "_record_clip", lambda *_args: pytest.fail("preflight failure launched SDK"))


def test_busy_gpu_never_launches_sdk_or_creates_output(monkeypatch, tmp_path):
    fixture(monkeypatch, tmp_path)
    monkeypatch.setattr(R.M, "query_gpu", lambda: {"busy": True})
    result = R.record("fresh", tmp_path)
    assert result == {"status": "gpu_busy_no_launch", "model_calls": 0, "gpu_runs": 0}
    assert not (tmp_path / "bench/private/checkpoint_preview/fresh").exists()
    assert not (tmp_path / "docs/media/go2-policy-fresh").exists()


@pytest.mark.parametrize("used", ["bench/private/checkpoint_preview/fresh", "docs/media/go2-policy-fresh"])
def test_used_tag_never_queries_gpu_or_overwrites_artifacts(monkeypatch, tmp_path, used):
    fixture(monkeypatch, tmp_path)
    marker = tmp_path / used / "sentinel"
    marker.parent.mkdir(parents=True)
    marker.write_text("existing evidence", encoding="utf-8")
    monkeypatch.setattr(R.M, "query_gpu", lambda: pytest.fail("used tag reached GPU gate"))
    with pytest.raises(FileExistsError, match="already used"):
        R.record("fresh", tmp_path)
    assert marker.read_text(encoding="utf-8") == "existing evidence"
