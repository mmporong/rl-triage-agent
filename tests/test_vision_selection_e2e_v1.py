from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / "evals", ROOT / "bench", ROOT / "tests"):
    sys.path.insert(0, str(folder))
E = importlib.import_module("vision_selection_e2e_v1")
SF = importlib.import_module("test_vision_selection_sources_v1")
MF = importlib.import_module("test_vision_selection_materialize_v1")


def _plan() -> dict:
    rows = []
    for index, target in enumerate(E.TARGETS, 1):
        rows.append({"target": dict(target), "call_id": f"e2e-source-0{index}",
                     "source_call_id": f"choice-{3 if index == 1 else 21:02d}",
                     "source_seed": 2210 + index - 1, "source_job_id": f"s{2210 + index - 1}-f1",
                     "record_job_id": f"record-{2210 + index - 1}-f1",
                     "source_lineage_id": f"record-{2210 + index - 1}-f1-same-play",
                     "corrected_job_id": f"vision-s{2210 + index - 1}-corrected",
                     "checkpoint_sha256": f"{index:064x}", "capsule_sha256": f"{index + 2:064x}",
                     "reference_checkpoint_sha256": f"{index + 8:064x}",
                     "eligible_probes": ["P_reward", "P_value"],
                     "source_protocol_sha256_lf": "a" * 64})
    return {"schema": E.SCHEMA, "api_cap": 32, "rows": rows}


def _selection(probe: str = "P_reward") -> dict:
    return {"model_status": "valid", "action": "continue", "probe": probe,
            "terminal_sha256": "b" * 64, "activation_bridge_sha256": "c" * 64}


def _pair(plan: dict, group: str, outcome: str = "abnormal") -> dict:
    row = E._row(plan, group)
    common = {field: "d" * 64 for field in E.PROBE_RECEIPT_FIELDS
              if field not in {"status", "outcome", "seed", "arm", "checkpoint_sha256"}}
    return {"probe": "P_reward", "none": {**common, "status": "complete", "outcome": "normal",
            "seed": row["source_seed"], "arm": "NONE",
            "checkpoint_sha256": row["reference_checkpoint_sha256"]},
            "f1": {**common, "status": "complete", "outcome": outcome, "seed": row["source_seed"],
                   "arm": "F1", "checkpoint_sha256": row["checkpoint_sha256"]}}


def test_state_machine_runs_only_exact_reward_path() -> None:
    plan = _plan()
    assert E.commands(plan, {"api_used": 0}) == [{"kind": "initial_selection", "group": "source-01",
                                                    "call_id": "e2e-source-01", "api_cap": 32}]
    selections = {"source-01": _selection(), "source-02": {"model_status": "invalid"}}
    command = E.commands(plan, {"api_used": 1, "selections": selections})
    assert command == [{"kind": "probe_pair", "group": "source-01", "probe": "P_reward",
                        "arms": ["NONE", "F1"], "checkpoint_sha256": "1".zfill(64)}]
    pair = _pair(plan, "source-01")
    command = E.commands(plan, {"api_used": 1, "selections": selections,
                                "probes": {"source-01": pair}})
    assert command[0]["kind"] == "propose_correction"
    assert set(command[0]["evidence"]) == E.APPROVAL_FIELDS


@pytest.mark.parametrize("selection,pair", [
    ({"model_status": "invalid"}, None),
    ({"model_status": "valid", "action": "hold", "probe": None}, None),
    (_selection(), "normal"),
    (_selection(), "unknown"),
])
def test_hold_other_probe_or_nonabnormal_never_corrects(selection: dict, pair: str | None) -> None:
    plan = _plan()
    state = {"api_used": 1, "selections": {"source-01": selection, "source-02": {"model_status": "invalid"}}}
    if pair is not None:
        state["probes"] = {"source-01": _pair(plan, "source-01", pair)}
    assert all(row["kind"] not in {"propose_correction", "consume_and_run_correction"}
               for row in E.commands(plan, state))


def test_other_registered_probe_runs_pair_but_never_correction() -> None:
    plan = _plan(); selections = {"source-01": _selection("P_value"),
                                  "source-02": {"model_status": "invalid"}}
    command = E.commands(plan, {"api_used": 1, "selections": selections})
    assert command[0]["kind"] == "probe_pair" and command[0]["probe"] == "P_value"
    pair = _pair(plan, "source-01"); pair["probe"] = "P_value"
    assert E.commands(plan, {"api_used": 1, "selections": selections,
                             "probes": {"source-01": pair}}) == []


def test_checkpoint_bound_approval_then_fixed_eval_then_aux_observation() -> None:
    plan = _plan(); group = "source-01"; selection = _selection(); pair = _pair(plan, group)
    evidence = E.approval_candidate(plan, group, selection, pair)
    base = {"api_used": 2, "selections": {group: selection, "source-02": {"model_status": "invalid"}},
            "probes": {group: pair}}
    approved = {**base, "approvals": {group: {"state": "approved", "evidence": evidence}}}
    assert E.commands(plan, approved)[0]["kind"] == "consume_and_run_correction"
    consumed = {**base, "approvals": {group: {"state": "consumed", "evidence": evidence}},
                "corrections": {group: {"status": "complete", "checkpoint_sha256": "f" * 64,
                                         "fixed_eval": {"none": "healthy", "corrected": "healthy"}}}}
    command = E.commands(plan, consumed)[0]
    assert command["kind"] == "post_observation" and command["used_for_decision"] is False
    broken = json.loads(json.dumps(consumed)); broken["approvals"][group]["evidence"]["seed"] += 1
    with pytest.raises(ValueError, match="승인"):
        E.commands(plan, broken)


def test_shared_api_cap_blocks_only_model_calls() -> None:
    plan = _plan()
    assert E.commands(plan, {"api_used": 32}) == []
    with pytest.raises(ValueError, match="cap 32"):
        E.commands(plan, {"api_used": 33})


def test_one_remaining_api_slot_emits_only_first_post_observation() -> None:
    plan = _plan()
    selections = {group: _selection() for group in ("source-01", "source-02")}
    probes = {group: _pair(plan, group) for group in selections}
    evidence = {group: E.approval_candidate(plan, group, selections[group], probes[group])
                for group in selections}
    state = {
        "api_used": 31, "selections": selections, "probes": probes,
        "approvals": {group: {"state": "consumed", "evidence": evidence[group]}
                      for group in selections},
        "corrections": {group: {"status": "complete", "checkpoint_sha256": "f" * 64,
                                  "fixed_eval": {"none": "healthy", "corrected": "healthy"}}
                        for group in selections},
    }
    assert E.commands(plan, state) == [{
        "kind": "post_observation", "group": "source-01", "checkpoint_sha256": "f" * 64,
        "used_for_decision": False, "shared_api_cap": 32,
    }]


def test_prepare_plan_uses_actual_materializer_and_call_initialize(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    actual = SF.full_case.__wrapped__(tmp_path_factory)
    repo = actual["repo"]
    manifest = MF._production_manifest(actual)
    manifest_body = json.loads(manifest.read_text(encoding="utf-8"))
    method_path = repo / manifest_body["method_protocol"]["path"]
    method = json.loads(method_path.read_text(encoding="utf-8"))
    method["evidence_sha256_lf"] = {"bench/bands.json": MF._sha(repo / "bench/bands.json", lf=True)}
    MF._write(method_path, method)
    manifest_body["method_protocol"]["sha256_lf"] = MF._sha(method_path, lf=True)
    MF._write(manifest, manifest_body)
    materialized = MF.M.materialize(
        repo, manifest, repo / "materialized/e2e-inputs",
        repo / "bench/protocols/e2e-materialized-run.json",
    )
    protocol = repo / materialized["protocol"]["path"]
    locations = repo / materialized["input_locations"]["path"]
    SF._git(repo, "add", ".")
    SF._git(repo, "commit", "-m", "materialize e2e inputs")
    anchor = SF._git(repo, "rev-parse", "HEAD")
    body = E.prepare_plan(repo, protocol, locations, repo / "private/e2e.json",
                          execution_id="v05-e2e",
                          source_anchor=anchor, protocol_sha256=E._sha(protocol))
    assert [row["target"]["slot_id"] for row in body["rows"]] == ["input-03", "input-07"]
    for row in body["rows"]:
        assert len({row["source_job_id"], row["record_job_id"], row["source_lineage_id"],
                    row["corrected_job_id"]}) == 4
    plan = E.CALL._plan(body["selection_plan"])[2]
    assert plan["phase"] == "e2e" and plan["api_cap"] == 32 and len(plan["rows"]) == 2
    assert body["comparison"]["execution_root"].endswith("selection-materialized-production-v1")
    polluted = json.loads(protocol.read_text(encoding="utf-8"))
    polluted["rows"][0]["test_only_extra"] = True
    polluted_protocol = repo / "private/e2e-source-protocol-polluted.json"
    polluted_protocol.write_text(json.dumps(polluted, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="protocol row"):
        E.prepare_plan(
            repo, polluted_protocol, locations, repo / "private/polluted.json",
            execution_id="polluted",
            source_anchor=anchor, protocol_sha256=E._sha(polluted_protocol),
        )
    rebound = json.loads(protocol.read_text(encoding="utf-8"))
    rebound["source_protocol"]["sha256_lf"] = "0" * 64
    for row in rebound["rows"]:
        row["source_protocol_sha256"] = "0" * 64
    rebound_protocol = repo / "private/e2e-source-protocol-rebound.json"
    rebound_protocol.write_text(json.dumps(rebound, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source protocol LF SHA"):
        E.prepare_plan(repo, rebound_protocol, locations, repo / "private/rebound.json",
                       execution_id="rebound", source_anchor=anchor,
                       protocol_sha256=E._sha(rebound_protocol))
    comparison_output = repo / body["comparison"]["output_root"]
    comparison_output.mkdir(parents=True)
    (comparison_output / "comparison_summary.json").write_text("{}")
    with pytest.raises(ValueError, match="comparison 결과"):
        E.prepare_plan(repo, protocol, locations, repo / "private/later.json", execution_id="later",
                       source_anchor=anchor, protocol_sha256=E._sha(protocol))
