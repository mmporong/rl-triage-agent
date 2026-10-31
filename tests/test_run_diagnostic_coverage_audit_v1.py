# pyright: reportMissingImports=false
from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))

import run_diagnostic_coverage_audit_v1 as A


def _state() -> dict:
    return {"visible_facts": {name: float(index + 1) for index, name in enumerate(A.FACT_IDS)}}


def test_operator_action_uses_fixed_visible_scalars(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = {}
    monkeypatch.setattr(A.SEL, "request_spec", lambda state, action: seen.setdefault(
        "spec", {"action": action, "host_args": {"roles": ["case", "reference"]}}))
    action = A._operator_action(_state())
    assert action == {"action": "run_probe", "target": "P_reward",
                      "fact_ids": list(A.FACT_IDS)}
    assert seen["spec"]["action"] == action


def test_operator_action_rejects_missing_or_nonfinite_fact(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(A.SEL, "request_spec", lambda *_: pytest.fail("must reject first"))
    state = _state()
    state["visible_facts"][A.FACT_IDS[0]] = None
    with pytest.raises(ValueError, match="scalar fact"):
        A._operator_action(state)


@pytest.mark.parametrize("outcome,expected", [
    ("abnormal", "could distinguish"),
    ("normal", "other probes remain untested"),
    ("unknown", "indeterminate"),
])
def test_reporting_rules_do_not_infer_other_probes(outcome: str, expected: str) -> None:
    assert expected in A._interpret(outcome)


def test_epoch_requires_fixed_hold_sequence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "epoch.json"
    value = {"schema": "reasoning_diagnostic_loop_result_v1", "case_id": "c02",
             "mode": "enabled_low", "status": "complete", "decision_outcome": "HOLD",
             "final_status": "hold", "steps": [
                 {"action": {"action": "read_evidence"}},
                 {"action": {"action": "run_probe", "target": "P_reward"}},
                 {"action": {"action": "hold"}}]}
    path.write_text(json.dumps(value), encoding="utf-8")
    monkeypatch.setattr(A.BASE, "_sha", lambda _path: A.PARENT_EPOCH_SHA256)
    with pytest.raises(ValueError, match="read-count-HOLD"):
        A._epoch(path)


def test_prepare_protocol_keeps_operator_scope_and_zero_api(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    parent = tmp_path / "parent.json"
    parent.write_text(json.dumps({"schema": A.R.SCHEMA, "source_sha256_lf": {},
                                  "inherited_source_sha256_lf": {}}), encoding="utf-8")
    monkeypatch.setattr(A, "ROOT", tmp_path)
    monkeypatch.setattr(A.BASE, "ROOT", tmp_path)
    for relative in A.SOURCE_FILES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x", encoding="utf-8")
    protocol = A.prepare_protocol(parent)
    assert protocol["design"]["selection_actor"] == A.APPROVER
    assert protocol["design"]["resource_lease_owner"] == A.RUN.OWNER
    assert protocol["design"]["proof_cache"] == {
        "kind": "immutable_git_show", "scope": "one_audit",
        "mutable_or_authority_cached": False}
    assert protocol["design"]["model_selected_probe"] is False
    assert protocol["bounds"] == {"wall_s": 1800, "paired_probes": 1,
                                  "sdk_role_executions": 2, "api_requests": 0}
    assert protocol["claims"]["model_selection_failure"] is False


def test_execute_runs_one_operator_pair_and_preserves_model_hold(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    parent = tmp_path / "parent.json"
    probe = tmp_path / "probe-protocol.json"
    view = tmp_path / "view-protocol.json"
    files = [tmp_path / name for name in ("protocol.json", "source.json", "mode.json",
                                           "corpus.json", "epoch.json", "case.json",
                                           "reference.json", "case-runtime.json",
                                           "reference-runtime.json")]
    for path in [parent, probe, view, *files]:
        path.write_text("{}", encoding="utf-8")
    case, reference = ({"case_id": "c02", "input": files[5],
                        "metadata": {"runtime_facts": {"case": {"path": str(files[7])}}}},
                       {"case_id": "c01", "input": files[6],
                        "metadata": {"runtime_facts": {
                            "reference": {"path": str(files[8])}}}})
    inputs = {"spec": {}, "case": case, "reference": reference,
              "first": {"state": {"state_digest": "state"}},
              "action": {"action": "run_probe", "target": "P_reward", "fact_ids": []}}
    monkeypatch.setattr(A, "ROOT", tmp_path)
    monkeypatch.setattr(A, "verify_protocol", lambda *_: {
        "parent_reasoning_protocol": {"path": "parent.json"}})
    monkeypatch.setattr(A, "inspect_inputs", lambda *_: inputs)
    monkeypatch.setattr(A.R, "_base_protocol", lambda *_: {
        "protocols": {"paired_probe": {"path": "probe-protocol.json"},
                      "view": {"path": "view-protocol.json"}}})
    monkeypatch.setattr(A.RUN, "_lease", lambda: {"leaseId": "lease-1"})
    monkeypatch.setattr(A.SEL, "advance", lambda *_args, **_kwargs: {
        "observed": {"P_reward": "abnormal"}, "state_digest": "after"})
    calls = []

    def host(_state, _action, **kwargs):
        calls.append(kwargs["approver"])
        assert kwargs["verify_context"]({"role": "case"})["approved"] is True
        out = kwargs["out"]
        out.mkdir(parents=True)
        A.BASE._write(out / "host-result.json", {"schema": "fixture"})
        refs = {}
        for name in ("case_plan", "case_receipt", "reference_plan", "reference_receipt"):
            path = out / f"{name}.json"
            path.write_text("{}", encoding="utf-8")
            refs[name] = str(path)
        return refs

    result = A.execute(files[0], "a" * 40, files[1], files[2], files[3], files[4],
                       tmp_path / "lab", tmp_path / "out", execute_host=host)
    assert calls == [A.APPROVER]
    assert result["outcome"] == "abnormal"
    assert result["original_model_outcome"] == "HOLD"
    assert result["api_requests"] == 0 and result["paired_probes"] == 1


def test_cache_wrapper_preserves_execute_result(monkeypatch: pytest.MonkeyPatch) -> None:
    events = []

    @contextmanager
    def scope():
        events.append("enter")
        yield {"approval": False, "execution_allowed": False}
        events.append("exit")

    expected = {"schema": "fixture", "context_digest": "unchanged"}
    monkeypatch.setattr(A.CACHE, "proof_scope", scope)
    monkeypatch.setattr(A, "_execute", lambda *_args, **_kwargs: expected)
    actual = A.execute(Path("p"), "a" * 40, Path("s"), Path("m"), Path("c"),
                       Path("e"), Path("l"), Path("o"))
    assert actual is expected
    assert events == ["enter", "exit"]
