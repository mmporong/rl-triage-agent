import sys
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evals"))
import runtime_observation_development_v2 as D


def test_registered_compatibility_scope_covers_original_consumer_and_restores(tmp_path, monkeypatch):
    events = []
    monkeypatch.setattr(D, "verify_protocol", lambda *_args: {
        "component_development": {"path": "old.json"}})

    @contextmanager
    def scope():
        events.append("entered")
        try:
            yield
        finally:
            events.append("restored")

    monkeypatch.setattr(D.T, "compatibility_scope", scope)

    def original(*_args):
        assert events == ["entered"]
        return {"scope": "development_only"}

    monkeypatch.setattr(D.D, "run", original)
    assert D.run(tmp_path / "p", tmp_path / "i", "a" * 40, tmp_path / "q", tmp_path / "o") == {
        "scope": "development_only"}
    assert events == ["entered", "restored"]
