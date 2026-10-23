from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))
import selection_comparison_flow_v1 as F  # type: ignore[import-not-found]  # noqa: E402


def artifact(name: str, *, policy="shared_reference", epoch=0, corrections=None):
    return {"checkpoint": {"sha256": (name[0] if name else "a") * 64},
            "corrections": list(corrections or []), "policy": policy, "epoch": epoch,
            "backend": {"artifact": {"name": name}, "arm": "COMPOUND"}}


class Caller:
    def __init__(self):
        self.finished = []
        self.finalized = []

    def __call__(self, snapshot, condition, policy, epoch, ordinal):
        return {"condition": condition, "policy": policy, "epoch": epoch, "ordinal": ordinal}

    def finish_epoch(self, policy, epoch, snapshot, reason):
        self.finished.append((policy, epoch, reason, snapshot is None))

    def finalize(self, *, status, arms):
        self.finalized.append((status, tuple(arms)))
        return {"status": "complete", "closed": True}


def patch_common(monkeypatch, tmp_path, *, policies=("registry_next",), gate=None):
    reference = artifact("reference")
    raw = {"NONE": {"x": 1}, "COMPOUND": {"x": 2}}
    reports = {"NONE": {}, "COMPOUND": {}}
    gate = gate or {"none_walks": True, "outcome": "not_recovered"}
    monkeypatch.setattr(F, "_source", lambda *a, **k: (
        raw, reference, reports, gate, {"source": 1.0},
        {"reports": {"NONE": {"path": str(tmp_path / "none.json")}}}))
    monkeypatch.setattr(F, "_wall", lambda raw: 1.0)
    refs = {name: tmp_path / f"{name}.json" for name in [*F.CONTROL.L.PROBES, F.CONTROL.RC.NEW_PROBE]}
    monkeypatch.setattr(F, "_shared_references", lambda *a, **k: (refs, {"reference": 1.0}))
    monkeypatch.setattr(F.CONTROL, "POLICIES", policies)
    monkeypatch.setattr(F.CONTROL, "adapt_actual_artifact", lambda *a, **k: artifact(
        k["arm"], policy=k["policy"], epoch=k["epoch"], corrections=k["corrections"]))
    monkeypatch.setattr(F.DEMO, "write_new", lambda *a, **k: None)
    monkeypatch.setattr(F.DEMO, "binding", lambda *a, **k: {})
    monkeypatch.setattr(F.DEMO, "read", lambda *a, **k: {"bands": {}})
    return reference


def test_case_probe_authorizes_and_validates_before_backend(monkeypatch, tmp_path):
    trace = []
    snapshot = {"view": {"status": "open"}, "inputs": {"packet": {"input_bindings": {
        "reference": {"checkpoint_sha256": "r" * 64}}}}}
    choice = {"probe": "P_reward"}
    case = artifact("case", policy="registry_next")
    reference = tmp_path / "reference.json"
    reference.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(F.CONTROL, "request_spec", lambda *_: {"args": {"registered": True}})
    monkeypatch.setattr(F, "_reference_plan", lambda *a, **k: {"bound": True})
    monkeypatch.setattr(F.CONTROL, "authorize_probe", lambda *a, **k: trace.append("authorize") or {
        "path": str(tmp_path / "authorization_01.json"), "sha256": "a" * 64, "request_id": "r"})
    monkeypatch.setattr(F.CONTROL, "validate_probe_authorization",
                        lambda *a, **k: trace.append("validate") or {})
    monkeypatch.setattr(F.DEMO, "verify", lambda *a, **k: trace.append("verify"))
    monkeypatch.setattr(F.DEMO, "binding", lambda *a, **k: {})
    monkeypatch.setattr(F, "_verified", lambda stage, kind, raw: raw)
    monkeypatch.setattr(F.CONTROL, "actual_legacy_probe_evidence", lambda *a, **k: {"ok": True})
    def write_new(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(F.DEMO, "write_new", write_new)
    monkeypatch.setattr(F, "_wall", lambda raw: 1.0)
    monkeypatch.setattr(F.CONTROL, "seal_probe", lambda *a, **k: trace.append("seal") or {
        "path": str(tmp_path / "probe_01.json"), "outcome": "normal"})
    monkeypatch.setattr(F.CONTROL, "verify_probe", lambda *a, **k: trace.append("verify_seal") or {})

    class Backend:
        def legacy_probe(self, *args, **kwargs):
            trace.append("backend")
            return {}

    F._case_probe(tmp_path, "tag", tmp_path, {}, "anchor", Backend(), case, "registry_next", 1,
                  snapshot, choice, "operator", {"P_reward": reference}, 1)
    assert trace.index("authorize") < trace.index("validate") < trace.index("backend") < trace.index("seal")


def test_run_performs_two_repairs_and_keeps_policy_paths_isolated(monkeypatch, tmp_path):
    patch_common(monkeypatch, tmp_path, policies=("registry_next", "fixed_eligible"))
    states = {}

    def snapshot(_folder, _tag, policy, epoch, *_):
        count = states.get((policy, epoch), 0)
        return {"view": {"status": "open" if count == 0 else "confirmed"},
                "inputs": {"policy": policy}}

    monkeypatch.setattr(F, "_snapshot", snapshot)
    monkeypatch.setattr(F.CONTROL, "dialogue_input", lambda snap: {"needs_model": False})
    monkeypatch.setattr(F.CONTROL, "choose", lambda snap, policy, dialogue=None: {
        "action": "continue", "probe": "P_reward"})

    probe_paths = []

    def case_probe(folder, tag, lab, protocol, anchor, backend, current, policy, epoch, *args):
        states[(policy, epoch)] = 1
        probe_paths.append(folder / "arms" / policy / f"epoch{epoch}")
        return {"path": str(probe_paths[-1] / "probe.json"), "outcome": "normal", "owned_wall_s": 1.0}

    monkeypatch.setattr(F, "_case_probe", case_probe)
    monkeypatch.setattr(F.CONTROL, "next_epoch", lambda parent, **k: parent is not None)
    repairs = []

    def repair(folder, tag, lab, protocol, anchor, backend, policy, epoch, applied, current, *args):
        path = folder / "arms" / policy / f"epoch{epoch}/repair.json"
        outcome = "not_recovered" if epoch == 1 else "recovered"
        after = [*applied, f"c{epoch}"]
        output = artifact(policy + str(epoch), policy=policy, epoch=epoch, corrections=after)
        verified = {"outcome": outcome, "applied_after": after, "result_sha256": str(epoch) * 64,
                    "owned_wall_s": 2.0}
        repairs.append((policy, epoch, tuple(applied), path))
        return path, output, verified

    monkeypatch.setattr(F, "_repair", repair)
    caller = Caller()
    result = F.run(tmp_path, "tag", tmp_path, "operator", {}, "anchor",
                   backend=object(), dialogue_caller=caller)
    assert result["status"] == "complete"
    assert all(result["arms"][policy]["status"] == "recovered" for policy in F.CONTROL.POLICIES)
    assert [(row[0], row[1]) for row in repairs] == [
        ("registry_next", 1), ("registry_next", 2), ("fixed_eligible", 1), ("fixed_eligible", 2)]
    assert len(set(probe_paths)) == 4
    assert caller.finalized == [("complete", ("registry_next", "fixed_eligible"))]


def test_model_hold_continues_other_policy_and_closes_absent_epoch(monkeypatch, tmp_path):
    patch_common(monkeypatch, tmp_path, policies=("model_adaptive", "registry_next"))
    counts = {}

    def snapshot(_folder, _tag, policy, epoch, *_):
        count = counts.get((policy, epoch), 0)
        return {"view": {"status": "open" if count == 0 else "confirmed"},
                "inputs": {"policy": policy}}

    monkeypatch.setattr(F, "_snapshot", snapshot)
    monkeypatch.setattr(F.CONTROL, "dialogue_input", lambda snap: {
        "needs_model": snap["inputs"]["policy"] == "model_adaptive"})
    monkeypatch.setattr(F.CONTROL, "choose", lambda snap, policy, dialogue=None: (
        {"action": "hold", "probe": None, "reason": "model_invalid"}
        if policy == "model_adaptive" else {"action": "continue", "probe": "P_reward"}))

    def case_probe(*args):
        policy, epoch = args[7], args[8]
        counts[(policy, epoch)] = 1
        return {"path": str(tmp_path / "probe.json"), "outcome": "normal", "owned_wall_s": 1.0}

    monkeypatch.setattr(F, "_case_probe", case_probe)
    monkeypatch.setattr(F.CONTROL, "next_epoch", lambda *a, **k: False)
    monkeypatch.setattr(F, "_repair", lambda *a, **k: (
        tmp_path / "repair.json", artifact("out", policy="registry_next", epoch=1, corrections=["c"]),
        {"outcome": "recovered", "applied_after": ["c"], "result_sha256": "f" * 64,
         "owned_wall_s": 2.0}))
    caller = Caller()
    result = F.run(tmp_path, "tag", tmp_path, "operator", {}, "anchor", backend=object(), dialogue_caller=caller)
    assert result["arms"]["model_adaptive"]["status"] == "HOLD"
    assert result["arms"]["registry_next"]["status"] == "recovered"
    assert ("model_adaptive", 2, "absent", True) in caller.finished


@pytest.mark.parametrize(
    ("gate", "expected"),
    [({"none_walks": False, "outcome": "not_recovered"}, "baseline_invalid"),
     ({"none_walks": True, "outcome": "recovered"}, "source_failure_absent")],
)
def test_source_gate_stops_before_reference_or_policy(monkeypatch, tmp_path, gate, expected):
    patch_common(monkeypatch, tmp_path, gate=gate)
    monkeypatch.setattr(F, "_shared_references", lambda *a, **k: pytest.fail("reference should not run"))
    caller = Caller()
    result = F.run(tmp_path, "tag", tmp_path, "operator", {}, "anchor", backend=object(), dialogue_caller=caller)
    assert result["status"] == expected and result["arms"] == {}
    assert caller.finalized == [(expected, ())]


def test_backend_invalid_halts_remaining_policies_and_preserves_unknown(monkeypatch, tmp_path):
    patch_common(monkeypatch, tmp_path, policies=("registry_next", "fixed_eligible"))
    monkeypatch.setattr(F, "_snapshot", lambda *a, **k: {
        "view": {"status": "open"}, "inputs": {"policy": a[2]}})
    monkeypatch.setattr(F.CONTROL, "dialogue_input", lambda snap: {"needs_model": False})
    monkeypatch.setattr(F.CONTROL, "choose", lambda *a, **k: {"action": "continue", "probe": "P_reward"})
    calls = []

    def broken(*args, **kwargs):
        calls.append(args[7])
        raise RuntimeError("gpu receipt invalid")

    monkeypatch.setattr(F, "_case_probe", broken)
    caller = Caller()
    result = F.run(tmp_path, "tag", tmp_path, "operator", {}, "anchor", backend=object(), dialogue_caller=caller)
    assert result["status"] == "invalid"
    assert calls == ["registry_next"]
    assert result["failure"]["type"] == "RuntimeError"
    assert caller.finalized[0][0] == "invalid"


def test_execute_checks_fresh_seed_once_before_creating_tag(monkeypatch, tmp_path):
    monkeypatch.setattr(F.DEMO, "PRIVATE", tmp_path)
    monkeypatch.setattr(F.DEMO, "verify", lambda *_: ({}, "a" * 40))
    monkeypatch.setattr(F.DEMO, "sha", lambda *a, **k: "b" * 64)
    calls = []

    def guard(seeds, isaaclab, *, repo):
        calls.append((tuple(seeds), isaaclab, repo))
        return {"fresh": True}

    monkeypatch.setattr(F, "run", lambda *a, **k: {
        "schema": "selection_comparison_result_v1", "status": "complete"})
    result = F.execute("selection_comparison_s2170_20261023", tmp_path, "operator", True,
                       backend=object(), dialogue_caller=Caller(), seed_guard=guard)
    assert result["status"] == "complete" and len(calls) == 1
    assert (tmp_path / "selection_comparison_s2170_20261023/result.json").is_file()

    blocked = tmp_path / "selection_comparison_s2170_20261023_retry"
    with pytest.raises(RuntimeError, match="seed used"):
        F.execute(blocked.name, tmp_path, "operator", True, backend=object(),
                  dialogue_caller=Caller(), seed_guard=lambda *a, **k: (_ for _ in ()).throw(
                      RuntimeError("seed used")))
    assert not blocked.exists()


def test_snapshot_builds_diagnosis_from_raw_backend_artifacts(monkeypatch, tmp_path):
    captured = {}
    current, reference = artifact("case"), artifact("reference")

    def packet(case_raw, reference_raw, epoch, applied, prior):
        captured.update(case=case_raw, reference=reference_raw, epoch=epoch)
        return {"packet": True}

    monkeypatch.setattr(F.RESIDUAL, "diagnosis_packet", packet)
    monkeypatch.setattr(F.CONTROL.V, "build_snapshot", lambda packet_value, *a: packet_value)
    assert F._snapshot(tmp_path, "tag", "registry_next", 1, current, reference, [], None) == {
        "packet": True}
    assert captured == {"case": current["backend"]["artifact"],
                        "reference": reference["backend"]["artifact"], "epoch": 1}


def test_repair_passes_backend_artifact_mapping_to_fixed_evaluate(monkeypatch, tmp_path):
    current, reference = artifact("case", policy="registry_next"), artifact("reference")
    raw, output = {"arm": "FIRST"}, artifact("output", policy="registry_next", epoch=1,
                                               corrections=["restore"])
    monkeypatch.setattr(F.CONTROL, "authorize_repair", lambda *a, **k: {
        "request_id": "r", "correction_id": "restore"})
    monkeypatch.setattr(F.DEMO, "binding", lambda *a, **k: {})
    monkeypatch.setattr(F.DEMO, "verify", lambda *a, **k: None)
    monkeypatch.setattr(F, "_verified", lambda stage, kind, value: value)
    monkeypatch.setattr(F.CONTROL, "adapt_actual_artifact", lambda *a, **k: output)
    monkeypatch.setattr(F.CONTROL, "fixed_report", lambda *a, **k: {})
    monkeypatch.setattr(F.CONTROL, "seal_repair", lambda *a, **k: None)
    monkeypatch.setattr(F.CONTROL, "verify_repair", lambda *a, **k: {
        "outcome": "recovered", "applied_after": ["restore"]})
    monkeypatch.setattr(F, "_wall", lambda *a, **k: 1.0)

    class Backend:
        def train(self, *args):
            return raw

        def fixed_evaluate(self, stage, lab, artifacts, *args):
            assert artifacts == {"FIRST": raw}
            return {"report_paths": {"FIRST": str(tmp_path / "first.json")}}

    _, _, verified = F._repair(
        tmp_path, "tag", tmp_path, {}, "anchor", Backend(), "registry_next", 1, [], current,
        reference, tmp_path / "none.json", {"view": {"status": "confirmed"}}, [], "operator", {}, None)
    assert verified["owned_wall_s"] == 2.0


def test_shared_legacy_reference_uses_default_and_accepts_exact_sorted_six(monkeypatch, tmp_path):
    calls = []
    with pytest.raises(ValueError, match="명시 probe"):
        F.EXECUTION.RESIDUAL.legacy_probe(
            tmp_path / "frozen_reject", tmp_path, {"NONE": {"arm": "NONE"}},
            "frozen_reference", {}, probes=sorted(F.CONTROL.L.PROBES))
    monkeypatch.setattr(F.DEMO, "binding", lambda *a, **k: {})
    monkeypatch.setattr(F.DEMO, "verify", lambda *a, **k: None)
    monkeypatch.setattr(F, "_verified", lambda stage, kind, value: value)
    monkeypatch.setattr(F.CONTROL, "actual_legacy_probe_evidence", lambda *a, **k: {"kind": "legacy"})
    monkeypatch.setattr(F.CONTROL, "actual_optimizer_probe_evidence", lambda *a, **k: {"kind": "optimizer"})
    monkeypatch.setattr(F.DEMO, "write_new", lambda *a, **k: None)
    monkeypatch.setattr(F, "_wall", lambda *a, **k: 1.0)

    class Backend:
        def legacy_probe(self, *args, probes=None):
            calls.append(probes)
            return {"probes": sorted(F.CONTROL.L.PROBES)}

        def inspect(self, *args):
            return {}

    reference = artifact("reference")
    reference["checkpoint"]["sha256"] = "a" * 64
    F._shared_references(tmp_path, "tag", tmp_path, {}, "anchor", Backend(), {}, reference)
    assert calls == [None] and len(sorted(F.CONTROL.L.PROBES)) == 6


@pytest.mark.parametrize("returned", [
    sorted(F.CONTROL.L.PROBES)[:-1],
    list(reversed(sorted(F.CONTROL.L.PROBES))),
])
def test_shared_legacy_reference_rejects_missing_or_reversed_six(monkeypatch, tmp_path, returned):
    monkeypatch.setattr(F.DEMO, "binding", lambda *a, **k: {})
    monkeypatch.setattr(F.DEMO, "verify", lambda *a, **k: None)
    monkeypatch.setattr(F, "_verified", lambda stage, kind, value: value)

    class Backend:
        def legacy_probe(self, *args, probes=None):
            assert probes is None
            return {"probes": returned}

    with pytest.raises(ValueError, match="정렬 6종"):
        F._shared_references(tmp_path, "tag", tmp_path, {}, "anchor", Backend(), {}, artifact("reference"))


def test_invalid_model_finalize_is_propagated(monkeypatch, tmp_path):
    patch_common(monkeypatch, tmp_path, gate={"none_walks": False, "outcome": "not_recovered"})

    class InvalidCaller(Caller):
        def finalize(self, *, status, arms):
            return {"status": "invalid", "pending": 1}

    result = F.run(tmp_path, "tag", tmp_path, "operator", {}, "anchor",
                   backend=object(), dialogue_caller=InvalidCaller())
    assert result["status"] == "invalid" and result["failure"]["type"] == "ModelAuditInvalid"


def test_batch_end_closes_running_probe_with_null_cost(monkeypatch, tmp_path):
    path = tmp_path / "arms/registry_next/epoch1/probes.ledger.jsonl"
    ledger = F.CONTROL.S.M.RegistryLedger(path, F.CONTROL.RC.registry())
    probe = next(iter(F.CONTROL.RC.registry()["probes"]))
    prereg = {"case": "fixture"}
    request = ledger.propose(prereg, probe, 1200)
    ledger.approve(request["request_id"], "operator")
    ledger.consume(request["request_id"], prereg, probe)
    F._close_incomplete_ledgers(tmp_path)
    finished = ledger.requests()[request["request_id"]]
    assert finished["state"] == "done"
    assert finished["receipt"]["outcome"] == "unknown"
    assert finished["receipt"]["gpu_s"] is None
    assert F._ledger_audit(tmp_path)["pending_count"] == 0
