import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))

import prove_policies as P  # noqa: E402


@pytest.fixture
def repo(tmp_path, monkeypatch):
    shutil.copytree(ROOT / "policies", tmp_path / "policies", ignore=shutil.ignore_patterns("candidates"))
    monkeypatch.setattr(P, "ROOT", tmp_path)
    monkeypatch.setattr(P, "POL", tmp_path / "policies")
    monkeypatch.setattr(P, "OUT", tmp_path / "policies" / "candidates")
    return tmp_path


def _fake_prove(result_for):
    def prove(candidate: Path) -> dict:
        name = "triage_agent" if candidate.name == "triage_agent.yaml" else candidate.name.removesuffix(".merged.yaml")
        return {"exit_code": 0, "result": result_for(name), "counterexample": None,
                "coverage": None, "reason": None, "stderr": ""}
    return prove


def test_missing_prover_exits_without_overwriting_proofs(repo, monkeypatch, capsys):
    monkeypatch.setattr(P, "prove", _fake_prove(lambda name: None))
    with pytest.raises(SystemExit) as exc:
        P.main()
    assert exc.value.code == 1
    assert "result=unavailable" in capsys.readouterr().out
    assert not (repo / "evals" / "results" / "policy_proofs.json").exists()


def test_missing_shell_is_reported_as_no_result(monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError("wsl")
    monkeypatch.setattr(P.subprocess, "run", missing)
    r = P.prove(ROOT / "policies" / "triage_agent.yaml")
    assert r["result"] is None
    assert "FileNotFoundError" in r["stderr"]


def test_candidates_and_proofs_are_written_with_lf(repo, monkeypatch):
    monkeypatch.setattr(P, "prove", _fake_prove(P.EXPECT.get))
    with pytest.raises(SystemExit) as exc:
        P.main()
    assert exc.value.code == 0
    written = [*sorted((repo / "policies" / "candidates").glob("*.merged.yaml")),
               repo / "evals" / "results" / "policy_proofs.json"]
    assert len(written) == len(list((repo / "policies" / "proposals").glob("*.yaml"))) + 1
    for path in written:
        assert b"\r\n" not in path.read_bytes(), path.name
