# pyright: reportMissingImports=false
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from evals import export_prospective_reasoning_confirmation_v3 as E


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n",
                    encoding="utf-8")


def _initial(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[dict, Path]:
    corpus_ledger = tmp_path / "corpus-ledger.json"
    runtime_ledger = tmp_path / "epoch/fact-ledger.json"
    corpus_ledger.parent.mkdir(parents=True, exist_ok=True)
    runtime_ledger.parent.mkdir(parents=True, exist_ok=True)
    corpus_ledger.write_text("same-ledger", encoding="utf-8")
    runtime_ledger.write_text("same-ledger", encoding="utf-8")
    source = {role: {"checkpoint_sha256": char * 64,
                     "binding_digest": ("c" if role == "case" else "d") * 64}
              for role, char in (("case", "a"), ("reference", "b"))}
    monkeypatch.setattr(E.BASE.R.SEL.LEDGER, "checked",
                        lambda _path: {"source_binding": deepcopy(source)})
    monkeypatch.setattr(E.BASE.R.SEL.S, "validate_packet", lambda value: deepcopy(value))
    packet = {"facts": {"telemetry:case:mean_reward": 1.0},
              "remaining_candidates": list(E.BASE.R.SEL.S.C1.registry()["mechanisms"]),
              "observed": {}, "budget_left": 3,
              "input_bindings": {
                  "case": {"checkpoint_sha256": "a" * 64, "telemetry_sha256": "e" * 64},
                  "reference": {"checkpoint_sha256": "b" * 64,
                                "telemetry_sha256": "f" * 64}}}
    initial = E.BASE.R.SEL.initialize(packet, corpus_ledger)
    return initial, runtime_ledger


def _epoch(initial: dict, runtime_ledger: Path) -> dict:
    expected = deepcopy(initial)
    expected["fact_ledger"]["path"] = str(runtime_ledger.resolve())
    expected = E.BASE.R.SEL._seal(expected)
    return {"initial_state_digest": expected["state_digest"],
            "source_binding": {
                "case": {"checkpoint_sha256": "a" * 64, "telemetry_sha256": "e" * 64},
                "reference": {"checkpoint_sha256": "b" * 64,
                              "telemetry_sha256": "f" * 64}}}


def test_runtime_initial_rebinds_only_to_identical_sealed_ledger(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    initial, ledger = _initial(tmp_path, monkeypatch)
    epoch = _epoch(initial, ledger)
    actual = E._runtime_rule_initial(ledger.parent, initial, epoch)
    assert actual["state_digest"] == epoch["initial_state_digest"]
    assert actual["fact_ledger"]["path"] == str(ledger.resolve())
    assert actual["initial_packet"] == initial["initial_packet"]


@pytest.mark.parametrize("mutation,match", [
    ("ledger", "ledger"), ("source", "source"), ("digest", "digest")])
def test_runtime_initial_rejects_ledger_source_or_digest_tampering(
        mutation: str, match: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    initial, ledger = _initial(tmp_path, monkeypatch)
    epoch = _epoch(initial, ledger)
    if mutation == "ledger":
        ledger.write_text("changed-ledger", encoding="utf-8")
    elif mutation == "source":
        epoch["source_binding"]["case"]["telemetry_sha256"] = "0" * 64
    else:
        epoch["initial_state_digest"] = "0" * 64
    with pytest.raises(ValueError, match=match):
        E._runtime_rule_initial(ledger.parent, initial, epoch)


def test_release_injects_runtime_collect_without_changing_outer_arguments(
        monkeypatch: pytest.MonkeyPatch) -> None:
    seen = {}

    def release(*args, **kwargs):
        seen["args"] = args
        seen["kwargs"] = kwargs
        return {"status": "complete"}

    monkeypatch.setattr(E.RELEASE2, "export", release)
    args = tuple(Path(name) for name in
                 ("execution", "protocol", "design", "bridge", "source", "corpus",
                  "bridge-root", "failure", "audit"))
    assert E.export(*args, anchor="a", out=Path("out")) == {"status": "complete"}
    assert seen["args"] == args
    assert seen["kwargs"]["anchor"] == "a"
    assert seen["kwargs"]["base_export_fn"] is E._base_export


def test_base_export_forces_runtime_collect(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = {}

    def base(*args, **kwargs):
        seen.update(kwargs)
        return {"status": "complete"}

    monkeypatch.setattr(E.BASE, "export", base)
    args = tuple(Path(name) for name in
                 ("result", "protocol", "design", "bridge", "source", "corpus",
                  "bridge-root"))
    assert E._base_export(*args, anchor="a", out=Path("out")) == {"status": "complete"}
    assert seen["collect_fn"] is E._collect
