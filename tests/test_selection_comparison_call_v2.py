# pyright: reportMissingImports=false
"""실제 API 없이 호출·전송·원장·피드백 경계를 검증한다."""

from __future__ import annotations

import itertools
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))
import selection_comparison_call_v2 as C  # noqa: E402
import test_selection_comparison_view_v1 as fixtures  # noqa: E402


class CallerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "batch"
        self.sha = C.V.file_digest(C.PROTOCOL)
        C.initialize(self.root, "caller_fixture", self.sha, fixture=True)

    def snapshot(self, policy="model_adaptive", value=None, name="live"):
        return C.V.build_snapshot(fixtures.packet() if value is None else value, "caller_fixture", policy,
                                  C.V.file_digest(C.V.PROTOCOL), self.root / (name + ".jsonl"))

    def append(self, snapshot, probe="P_value", outcome="normal", measurement=None):
        ledger = C.V.M.RegistryLedger(Path(snapshot["inputs"]["ledger_path"]), C.V.C.registry())
        request = ledger.propose(snapshot["binding"], probe, snapshot["view"]["execution_budget_s"])
        ledger.approve(request["request_id"], "caller_cpu_fixture")
        ledger.consume(request["request_id"], snapshot["binding"], probe)
        ledger.receipt(request["request_id"], outcome, measurement or {}, 1, 0)
        return C.V.build_snapshot(**snapshot["inputs"])

    def response(self, probe="P_reward", *, action="continue", facts=None, **changes):
        content = json.dumps({"action": action, "probe": probe if action == "continue" else None,
                              "fact_ids": [fixtures.FACT] if facts is None else facts})
        return {"provider_model": C.MODEL, "content": content, "finish_reason": "stop",
                "input_tokens": 10, "output_tokens": 5, "total_tokens": 15, "wall_s": 0.1, **changes}

    def run_call(self, responses, *, snapshot=None, condition="production"):
        source = self.snapshot() if snapshot is None else snapshot
        path = C.freeze(self.root, source, condition)
        complete = C.FixtureResponses(responses)
        result = C.decide(self.root, path, complete)
        return path, result, complete.sent

    def test_valid_production_reuses_default_params_and_records_bound_receipts(self):
        path, record, sent = self.run_call([self.response()])
        self.assertEqual(record["selection"]["selected_by"], "model")
        self.assertEqual(record["requests"], 1)
        self.assertEqual(sent[0][1], C.F.request_params(C.MODEL))
        body = sent[0][0][-1]["content"]
        for secret in (str(self.root), "caller_fixture", "session_digest", "a" * 64, '"args"'):
            self.assertNotIn(secret, body)
        self.assertEqual(C.review(self.root, path, self.root / "calls" / path.name), record)
        self.assertEqual(record["costs"]["total_tokens"], 15)
        self.assertEqual(C.audit(self.root)["reserved"], 1)
        self.assertFalse(record["approval_granted"])

    def test_fixed_and_first_only_after_observation_make_no_api_calls(self):
        for policy in ("registry_next", "fixed_eligible", "model_first_then_rule"):
            snapshot = self.snapshot(policy, name=policy)
            if policy == "model_first_then_rule":
                snapshot = self.append(snapshot)
            path, record, sent = self.run_call([], snapshot=snapshot)
            self.assertEqual((record["requests"], sent), (0, []))
            self.assertEqual(C.review(self.root, path, self.root / "calls" / path.name), record)
        self.assertEqual(C.audit(self.root)["reserved"], 0)

    def test_invalid_feedback_then_terminal_hold_never_calls_registry_rule(self):
        bad = self.response(content='{ "ranking": [] }')
        with patch.object(C.V, "select", side_effect=AssertionError("선택 fallback 금지")) as select, \
                patch.object(C.V, "rule_probe", side_effect=AssertionError("규칙 fallback 금지")) as fallback:
            _, record, sent = self.run_call([bad, bad])
        select.assert_not_called()
        fallback.assert_not_called()
        self.assertEqual(record["selection"]["selected_by"], "hold")
        self.assertEqual(record["selection"]["reason"], "model_invalid_hold")
        self.assertEqual(record["requests"], 2)
        self.assertTrue(record["feedback_used"])
        self.assertIn("action, probe, and fact_ids", sent[1][0][-1]["content"])
        self.assertNotIn("permutation of all six", sent[1][0][-1]["content"])
        self.assertEqual(sent[1][0][-2]["content"], bad["content"])

    def test_semantic_feedback_corrects_unknown_fact_without_inventing_values(self):
        _, record, sent = self.run_call([self.response(facts=["missing"]), self.response()])
        self.assertEqual(record["selection"]["model_status"], "valid")
        self.assertIn("S_FACT_UNKNOWN", sent[1][0][-1]["content"])
        self.assertNotIn(fixtures.FACT, sent[1][0][-1]["content"])

    def test_infra_and_feedback_share_three_request_bound(self):
        sequences = (
            [TimeoutError("secret credential"), self.response(content="bad"), self.response()],
            [self.response(content="bad"), TimeoutError("secret credential"), self.response(content="bad")],
        )
        for index, sequence in enumerate(sequences):
            s = self.snapshot(name=f"infra{index}", value={**fixtures.packet(), "facts": {fixtures.FACT: index + 1}})
            _, record, sent = self.run_call(sequence, snapshot=s)
            self.assertEqual(record["requests"], 3)
            self.assertTrue(record["feedback_used"] and record["infra_retry_used"])
            self.assertEqual(len(sent), 3)
            self.assertNotIn("secret credential", json.dumps(record))
            self.assertIsNone(record["costs"]["total_tokens"])

    def test_hold_provider_mismatch_and_api_error_do_not_fall_back(self):
        rows = [self.response(action="hold"), self.response(provider_model="wrong_model"),
                self.response(error_type="PermissionError"), self.response(content=3),
                self.response(total_tokens=-1), self.response(finish_reason="tool_calls")]
        for index, response in enumerate(rows):
            s = self.snapshot(value={**fixtures.packet(), "facts": {fixtures.FACT: index + 1}}, name=f"hold{index}")
            _, record, _ = self.run_call([response], snapshot=s)
            self.assertEqual(record["selection"]["selected_by"], "hold")
            self.assertEqual(record["requests"], 1)
            self.assertFalse(record["feedback_used"])

    def test_registered_provider_alias_is_accepted_and_unregistered_alias_is_rejected(self):
        alias = "private/nvidia/nemotron-3.5-super-120b-a12b"
        _, accepted, _ = self.run_call([self.response(provider_model=alias)])
        self.assertEqual(accepted["selection"]["selected_by"], "model")
        snapshot = self.snapshot(value={**fixtures.packet(), "facts": {fixtures.FACT: 2}}, name="bad_alias")
        _, rejected, _ = self.run_call([self.response(provider_model="private/nvidia/unknown")], snapshot=snapshot)
        self.assertEqual(rejected["selection"]["selected_by"], "hold")
        self.assertEqual(rejected["selection"]["model_status"], "unknown")
        audit = C.audit(self.root)
        self.assertEqual((audit["ok"], audit["error"], audit["timeout"]), (1, 1, 0))

    def test_capsule_is_relocatable_and_old_history_remains_reviewable(self):
        snapshot = self.snapshot()
        path = C.freeze(self.root, snapshot)
        capsule = C.MC.read(path)
        self.assertNotIn("ledger_path", capsule["snapshot_inputs"])
        self.assertNotIn("\\", capsule["history_relative"])
        self.append(snapshot)
        frozen = C.prepared(self.root, path)[1]
        self.assertEqual(frozen["view"]["observations"], [])
        relocated = Path(self.tmp.name) / "relocated"
        shutil.copytree(self.root, relocated)
        other = C.prepared(relocated, relocated / "capsules" / path.name)
        self.assertEqual(C.prepared(self.root, path)[0], other[0])
        self.assertEqual(C.prepared(self.root, path)[3], other[3])
        record = C.decide(self.root, path, C.FixtureResponses([self.response()]))
        current = C.V.build_snapshot(**snapshot["inputs"])
        old_reply = {"context_digest": record["selection"]["context_digest"],
                     "view_digest": record["selection"]["view_digest"],
                     "content": record["calls"][-1]["result"]["content"], "status": "completed"}
        with self.assertRaises(ValueError):
            C.V.request_spec(current, record["selection"], old_reply)

    def test_cross_session_capsule_and_path_escape_are_rejected(self):
        path = C.freeze(self.root, self.snapshot())
        owner = C.MC.read(self.root / "session.json")
        owner["run_id"] = "other_run"
        (self.root / "session.json").write_text(json.dumps(owner), encoding="utf-8")
        with self.assertRaises(ValueError):
            C.prepared(self.root, path)
        for relative in ("../outside", "/absolute", "a/../outside", "a\\b", "x:y", "a//b"):
            with self.assertRaises(ValueError):
                C.inside(self.root, relative)

    def test_immutable_history_or_capsule_params_tampering_stops_before_api(self):
        path = C.freeze(self.root, self.snapshot())
        capsule = C.MC.read(path)
        history = C.inside(self.root, capsule["history_relative"])
        history.write_bytes(b'changed')
        with self.assertRaises(ValueError):
            C.prepared(self.root, path)
        history.write_bytes(b'')
        capsule["params"]["temperature"] = 1
        raw = (json.dumps(capsule) + "\n").encode()
        changed = path.parent / (C.raw_sha(raw) + ".json")
        changed.write_bytes(raw)
        with self.assertRaises(ValueError):
            C.prepared(self.root, changed)

    def test_saved_choice_and_receipt_cost_tampering_is_detected_by_replay(self):
        path, record, _ = self.run_call([self.response()])
        output = self.root / "calls" / path.name
        bad = deepcopy(record)
        bad["selection"]["probe"] = "P_physics"
        output.write_text(json.dumps(bad), encoding="utf-8")
        with self.assertRaises(ValueError):
            C.review(self.root, path, output)
        output.write_text(json.dumps(record), encoding="utf-8")
        response = self.root / "api/responses/0001.json"
        row = C.MC.read(response)
        row["result"]["wall_s"] = 99
        response.write_text(json.dumps(row), encoding="utf-8")
        with self.assertRaises(ValueError):
            C.review(self.root, path, output)

    def test_duplicate_logical_call_and_pending_reservation_do_not_call_again(self):
        path, _record, _sent = self.run_call([self.response()])
        before = C.audit(self.root)["reserved"]
        with self.assertRaises(ValueError):
            C.decide(self.root, path, C.FixtureResponses([]))
        self.assertEqual(C.audit(self.root)["reserved"], before)
        s = self.snapshot(value={**fixtures.packet(), "facts": {fixtures.FACT: 22}})
        pending_path = C.freeze(self.root, s)
        sha = C.MC.file_sha(pending_path)
        messages = C.prepared(self.root, pending_path)[3]
        C.reserve(self.root / "api", "diagnosis", sha, C.V.L.digest(messages), f"selection/{sha}/request1")
        with self.assertRaises(ValueError):
            C.decide(self.root, pending_path, C.FixtureResponses([]))
        self.assertEqual(C.audit(self.root)["pending"], 1)

    def test_json_whitespace_cannot_create_a_second_logical_call(self):
        path, _record, _sent = self.run_call([self.response()])
        raw = json.dumps(C.MC.read(path), indent=4).encode()
        duplicate = path.parent / (C.raw_sha(raw) + ".json")
        duplicate.write_bytes(raw)
        with self.assertRaises(ValueError):
            C.decide(self.root, duplicate, C.FixtureResponses([]))
        self.assertEqual(C.audit(self.root)["reserved"], 1)

    def test_global_cap_counts_all_conditions_and_budget_stop_is_preserved(self):
        for index in range(20):
            s = self.snapshot(value={**fixtures.packet(), "facts": {fixtures.FACT: index}}, name=f"cap{index}")
            self.run_call([self.response()], snapshot=s)
        s = self.snapshot(value={**fixtures.packet(), "facts": {fixtures.FACT: 999}}, name="cap_last")
        _, record, sent = self.run_call([], snapshot=s)
        self.assertEqual(record["stop_reason"], "budget_exhausted")
        self.assertEqual((record["requests"], sent), (0, []))
        self.assertEqual(C.audit(self.root)["reserved"], 20)

    def test_candidate_pruning_is_not_scored_as_model_observation_use(self):
        s = self.append(self.snapshot())
        masked = C.V.build_snapshot(**{**s["inputs"], "masked": True})
        self.assertEqual(C.pair_disposition(s, masked), "combined_host_view_change")
        terminal = self.append(s, "P_reward", "abnormal")
        other = C.V.build_snapshot(**{**terminal["inputs"], "masked": True})
        self.assertEqual(C.pair_disposition(terminal, other), "host_transition_only")

    def test_every_partitioning_registered_transition_changes_candidates_or_closes(self):
        registry = C.V.C.registry()
        transitions = 0
        for size in range(1, len(registry["mechanisms"]) + 1):
            for candidates in itertools.combinations(registry["mechanisms"], size):
                for probe in registry["probes"]:
                    if not C.V.eligible(registry, list(candidates), {}, probe):
                        continue
                    for outcome in ("normal", "abnormal"):
                        remaining, _ = C.V.M.update(registry, list(candidates), probe, outcome)
                        status = C.V.M._status(registry, remaining, {probe: outcome}, 6)
                        self.assertTrue(remaining != list(candidates) or status != "open")
                        transitions += 1
        self.assertGreater(transitions, 100)

    def test_auxiliary_pair_holds_surface_fixed_and_never_enables_execution(self):
        s = self.append(self.snapshot(), "P_reward", "abnormal",
                        {"track_lin_vel_xy_exp_rel_error": 2})
        a = C.freeze(self.root, s, "exposure_full")
        b = C.freeze(self.root, s, "exposure_masked")
        va, vb = C.prepared(self.root, a)[2], C.prepared(self.root, b)[2]
        for field in va:
            if field not in ("evidence_facts", "observations"):
                self.assertEqual(va[field], vb[field])
        self.assertFalse(va["used_for_decision"])
        self.assertNotIn("observation:0:P_reward:outcome", vb["evidence_facts"])
        self.assertEqual(C.exposure_summary(self.root, a, b)["disposition"], "pair_incomplete")
        C.decide(self.root, b, C.FixtureResponses([self.response(probe="P_value")]))
        C.decide(self.root, a, C.FixtureResponses([
            self.response(action="hold", facts=["observation:0:P_reward:outcome"])]))
        summary = C.exposure_summary(self.root, a, b)
        self.assertTrue(summary["view_sensitivity"])
        self.assertTrue(summary["explicit_latest_fact_use"])
        self.assertFalse(summary["model_advantage"] or summary["recovery_comparison"] or summary["used_for_decision"])
        self.assertGreater(summary["first_request_indices"][0], summary["first_request_indices"][1])

    def test_unknown_and_nonadaptive_states_do_not_allow_auxiliary_queries(self):
        for policy, outcome in (("model_adaptive", "unknown"), ("model_first_then_rule", "normal")):
            s = self.append(self.snapshot(policy, name=policy), outcome=outcome)
            with self.assertRaises(ValueError):
                C.query_view(s, "exposure_full")

    def test_unregistered_completion_and_nvidia_in_fixture_are_blocked_before_reservation(self):
        path = C.freeze(self.root, self.snapshot())
        with patch.object(C.F, "openai_complete", side_effect=AssertionError("허용 안 된 네트워크")) as network:
            for complete in (C.F.openai_complete, lambda *args: C.F.openai_complete(*args)):
                with self.assertRaises(ValueError):
                    C.decide(self.root, path, complete)
            with self.assertRaises(ValueError):
                C.FixtureResponses([C.F.openai_complete])
            with self.assertRaises(ValueError):
                C.NvidiaCompletion(self.root, path, C.PROTOCOL, "a" * 40)
            network.assert_not_called()
        self.assertEqual(C.audit(self.root)["reserved"], 0)

    def test_response_projection_drops_unregistered_headers_and_malformed_fields_before_storage(self):
        secret = "cpu_test_secret_sentinel"
        _, record, _ = self.run_call([self.response(headers={"Authorization": secret}, api_key=secret,
                                                  reasoning_content={"unexpected": secret})])
        for path in self.root.rglob("*.json"):
            self.assertNotIn(secret, path.read_text(encoding="utf-8"))
        self.assertEqual(record["selection"]["model_status"], "unknown")
        self.assertEqual(record["calls"][0]["result"]["error_type"], "ProviderResponseContractError")

    def test_invalid_auxiliary_selection_cannot_claim_latest_fact_use_or_sensitivity(self):
        s = self.append(self.snapshot(), "P_reward", "abnormal")
        a, b = (C.freeze(self.root, s, condition) for condition in ("exposure_full", "exposure_masked"))
        invalid = self.response(probe="not_registered", facts=["observation:0:P_reward:outcome"])
        C.decide(self.root, a, C.FixtureResponses([invalid, invalid]))
        C.decide(self.root, b, C.FixtureResponses([self.response(action="hold")]))
        result = C.exposure_summary(self.root, a, b)
        self.assertEqual(result["contract_valid"], [False, True])
        invalid_selection = C.review(self.root, a, self.root / "calls" / a.name)["selection"]
        self.assertEqual((invalid_selection["action"], invalid_selection["probe"], invalid_selection["fact_ids"]),
                         ("hold", None, []))
        self.assertIn("S_PROBE_INELIGIBLE", invalid_selection["codes"])
        self.assertIsNone(result["explicit_latest_fact_use"])
        self.assertIsNone(result["view_sensitivity"])
        other = C.freeze(self.root, s, "exposure_masked", replicate=2)
        with self.assertRaises(ValueError):
            C.exposure_summary(self.root, a, other)

    def plan(self, capsules):
        rows = []
        for index, capsule in enumerate(capsules, 1):
            state = C.planned_state(capsule)
            rows.append({"row_id": f"row{index}", "pair_id": None if state["condition"] == "production" else "pair1",
                         "order": index, "state": state, "max_requests": 3})
        return {"schema": "selection_call_execution_v2", "run_id": "caller_fixture", "caller_protocol_sha": self.sha,
                "model": C.MODEL, "accepted_provider_models": sorted(C.ACCEPTED_PROVIDER_MODELS),
                "params": C.F.request_params(C.MODEL), "api_cap": 20,
                "conditions": sorted({row["state"]["condition"] for row in rows}),
                "predictions": {
                    "development_disclosure": C.DEVELOPMENT_DISCLOSURE,
                    "bundle_sha256": C.INPUT_BUNDLE_SHA256,
                    "archive_integrity_sha256": C.ARCHIVE_INTEGRITY_SHA256,
                    "runner_sha256_lf": "1" * 64,
                    "tests_sha256_lf": "2" * 64,
                    "final_model_contract_valid": {"production": "2/2", "exposure": "4/4"},
                    "global_execution_claim": "semantic_plan_digest_single_start_across_roots",
                    "request_bound": 18,
                    "new_GPU_runs": 0,
                    "new_training_runs": 0,
                    "historical_costs_combined_with_API": False,
                    "used_for_recovery_decision": False,
                },
                "result_wording": C.RESULT_WORDING,
                "trusted_host_input_required": True, "planned_calls": rows}

    def test_execution_plan_seals_pair_input_replicate_order_and_worst_case_requests(self):
        s = self.append(self.snapshot(), "P_reward", "normal")
        a, b = (C.freeze(self.root, s, condition) for condition in ("exposure_full", "exposure_masked"))
        plan = self.plan([C.MC.read(a), C.MC.read(b)])
        self.assertEqual(C.validate_plan(plan, self.sha), plan)
        changes = []
        bad = deepcopy(plan)
        bad["planned_calls"][1]["order"] = 1
        changes.append(bad)
        bad = deepcopy(plan)
        bad["planned_calls"][1]["state"]["history_sha256"] = "b" * 64
        changes.append(bad)
        bad = deepcopy(plan)
        bad["planned_calls"][1]["state"]["replicate"] = 2
        changes.append(bad)
        bad = deepcopy(plan)
        bad["planned_calls"][0]["max_requests"] = 4
        changes.append(bad)
        bad = deepcopy(plan)
        bad["planned_calls"].pop()
        changes.append(bad)
        bad = deepcopy(plan)
        bad["planned_calls"] *= 4
        changes.append(bad)
        for bad in changes:
            with self.assertRaises(ValueError):
                C.validate_plan(bad, self.sha)
        second = deepcopy(plan["planned_calls"])
        for index, row in enumerate(second, 3):
            row.update(row_id=f"row{index}", pair_id="pair2", order=index)
            row["state"]["replicate"] = 2
        unbalanced = {**plan, "planned_calls": plan["planned_calls"] + second}
        with self.assertRaises(ValueError):
            C.validate_plan(unbalanced, self.sha)
        second.reverse()
        for index, row in enumerate(second, 3):
            row["order"] = index
        balanced = {**plan, "planned_calls": plan["planned_calls"] + second}
        self.assertEqual(C.validate_plan(balanced, self.sha), balanced)

    def test_plan_predictions_and_result_wording_are_exact(self):
        s = self.append(self.snapshot(), "P_reward", "normal")
        capsules = [C.MC.read(C.freeze(self.root, s, condition))
                    for condition in ("exposure_full", "exposure_masked")]
        plan = self.plan(capsules)
        mutations = []
        for field, value in (("request_bound", 17), ("new_GPU_runs", 1),
                             ("historical_costs_combined_with_API", True),
                             ("bundle_sha256", "invalid")):
            changed = deepcopy(plan)
            changed["predictions"][field] = value
            mutations.append(changed)
        changed = deepcopy(plan)
        changed["result_wording"]["runtime_verified"] = "changed"
        mutations.append(changed)
        for changed in mutations:
            with self.assertRaises(ValueError):
                C.validate_plan(changed, self.sha)
        for field in ("bundle_sha256", "archive_integrity_sha256"):
            changed = deepcopy(plan)
            changed["predictions"][field] = "0" * 64
            with self.assertRaises(ValueError):
                C.validate_registered_inputs(changed, self.sha)

    def test_real_capability_requires_committed_plan_order_and_active_meter_reservation(self):
        experiment = Path(self.tmp.name) / "execution.json"
        experiment.write_text('{}', encoding="utf-8")
        owner = C.MC.read(self.root / "session.json")
        owner.update(fixture=False, experiment_sha=C.MC.file_sha(experiment), anchor="a" * 40)
        (self.root / "session.json").write_text(json.dumps(owner), encoding="utf-8")
        a = C.freeze(self.root, self.snapshot(), replicate=1)
        b = C.freeze(self.root, self.snapshot(), replicate=2)
        plan = self.plan([C.MC.read(a), C.MC.read(b)])
        C.validate_plan(plan, self.sha)
        with patch.object(C, "is_wsl", return_value=True), patch.object(C, "execution_manifest", return_value=plan), \
                patch.object(C, "verify_sources", return_value={}), \
                patch.object(C.F, "openai_complete", return_value=self.response()) as network:
            with self.assertRaises(FileNotFoundError):
                C.NvidiaCompletion(self.root, a, experiment, "a" * 40)
            C.authorize_controller(self.root, plan, experiment, "a" * 40)
            with self.assertRaises(FileNotFoundError):
                C.NvidiaCompletion(self.root, b, experiment, "a" * 40)
            complete = C.NvidiaCompletion(self.root, a, experiment, "a" * 40)
            with self.assertRaises(FileNotFoundError):
                complete([], C.F.request_params(C.MODEL))
            with self.assertRaises(ValueError):
                C.decide(self.root, a, C.FixtureResponses([self.response()]))
            record = C.decide(self.root, a, complete)
            self.assertEqual(record["selection"]["model_status"], "valid")
            with self.assertRaises(FileExistsError):
                C.NvidiaCompletion(self.root, a, experiment, "a" * 40)
            other = C.NvidiaCompletion(self.root, b, experiment, "a" * 40)
            self.assertEqual(C.decide(self.root, b, other)["requests"], 1)
            self.assertEqual(network.call_count, 2)

    def test_controller_authorization_binds_exact_root_plan_experiment_and_anchor(self):
        experiment = Path(self.tmp.name) / "execution.json"
        experiment.write_text('{}', encoding="utf-8")
        capsules = [C.MC.read(C.freeze(self.root, self.snapshot(name=f"auth{n}"), replicate=n))
                    for n in (1, 2)]
        plan = self.plan(capsules)
        receipt = C.authorize_controller(self.root, plan, experiment, "a" * 40)
        self.assertTrue(C.MC.read(receipt)["explicit_approval"])
        self.assertEqual(C.verify_controller_authorization(
            self.root, plan, experiment, "a" * 40)["authority_scope"],
            "local_controller_receipt_not_authenticated_host_identity")
        wrong = Path(self.tmp.name) / "wrong_root"
        C.MC.write_new(wrong / "controller_authorization.json", C.MC.read(receipt))
        with self.assertRaises(ValueError):
            C.verify_controller_authorization(wrong, plan, experiment, "a" * 40)
        with self.assertRaises(ValueError):
            C.verify_controller_authorization(self.root, {**plan, "api_cap": 19}, experiment, "a" * 40)
        with self.assertRaises(ValueError):
            C.verify_controller_authorization(self.root, plan, experiment, "b" * 40)

    def test_replay_rejects_forged_early_budget_exhaustion(self):
        path, record, _ = self.run_call([self.response(content="bad"), self.response()])
        response = self.root / "api/responses/0002.json"
        reservation = self.root / "api/reservations/0002.json"
        response.unlink()
        reservation.unlink()
        record.update(requests=1, stop_reason="budget_exhausted")
        (self.root / "calls" / path.name).write_text(json.dumps(record), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "공용 상한 전에"):
            C.review(self.root, path, self.root / "calls" / path.name)

    def test_supervisor_preserves_timeout_and_missing_record_without_auto_retry(self):
        experiment = Path(self.tmp.name) / "execution.json"
        experiment.write_text('{}', encoding="utf-8")
        owner = C.MC.read(self.root / "session.json")
        owner.update(fixture=False, experiment_sha=C.MC.file_sha(experiment), anchor="a" * 40)
        (self.root / "session.json").write_text(json.dumps(owner), encoding="utf-8")
        paths = [C.freeze(self.root, self.snapshot(), replicate=n) for n in (1, 2)]
        execution = self.plan([C.MC.read(path) for path in paths])
        C.authorize_controller(self.root, execution, experiment, "a" * 40)
        for index, path in enumerate(paths):
            failure = (subprocess.TimeoutExpired("cpu_fixture", 1) if index == 0
                       else SimpleNamespace(returncode=0))
            with patch.object(C, "execution_manifest", return_value=execution), \
                    patch.object(C, "verify_sources", return_value={}), \
                    patch.object(C, "os", SimpleNamespace(name="nt")), \
                    patch.object(C.MC, "wsl_path", side_effect=lambda p: str(p)), \
                    patch.object(C, "cleanup_worker", return_value="verified") as cleanup, \
                    patch.object(C.subprocess, "run", side_effect=failure if index == 0 else None,
                                 return_value=failure) as process:
                with self.assertRaises(TimeoutError if index == 0 else FileNotFoundError):
                    C.launch(self.root, path, experiment, "a" * 40)
                self.assertEqual(process.call_count, 1)
                self.assertEqual(cleanup.call_count, 1 if index == 0 else 0)
                row = C.MC.read(self.root / "calls" / (path.stem + ".transport_failure.json"))
                self.assertEqual(row["phase"], "worker_timeout" if index == 0 else "worker_exit_or_review")
                self.assertNotIn("stdout", row)

    def test_source_protocol_behavior_fields_are_checked_before_any_call(self):
        original = C.MC.read(C.PROTOCOL)
        for field, value in (("request_limit_per_logical_call", 4), ("semantic_feedback_limit", 2),
                             ("infra_retry_limit", 2), ("conditions", ["production"]),
                             ("accepted_provider_models", [C.MODEL])):
            bad = {**original, field: value}
            path = Path(self.tmp.name) / (field + ".json")
            path.write_text(json.dumps(bad), encoding="utf-8")
            with patch.object(C, "PROTOCOL", path), self.assertRaises(ValueError):
                C.verify_sources(C.V.file_digest(path))

    def test_source_commit_and_index_checks_are_read_only_and_exact(self):
        def git(argv, **_kwargs):
            if argv[1] == "merge-base":
                return SimpleNamespace(returncode=0, stdout=b'')
            relative = argv[-1].split(":", 1)[1]
            return SimpleNamespace(returncode=0, stdout=(ROOT / relative).read_bytes())
        with patch.object(subprocess, "run", side_effect=git):
            C.verify_sources(self.sha, "a" * 40)
        def wrong_index(argv, **kwargs):
            if argv[-1] == ":evals/selection_comparison_call_v2.py":
                return SimpleNamespace(returncode=0, stdout=b'changed index')
            return git(argv, **kwargs)
        with patch.object(subprocess, "run", side_effect=wrong_index), self.assertRaises(ValueError):
            C.verify_sources(self.sha, "a" * 40)
        def wrong_head(argv, **kwargs):
            if argv[-1] == "HEAD:evals/selection_comparison_call_v2.py":
                return SimpleNamespace(returncode=0, stdout=b'changed head')
            return git(argv, **kwargs)
        with patch.object(subprocess, "run", side_effect=wrong_head), self.assertRaises(ValueError):
            C.verify_sources(self.sha, "a" * 40)

    def test_cpu_contract_protocol_is_not_an_execution_preregistration(self):
        with patch.object(C, "verify_sources", return_value={}), self.assertRaises(ValueError):
            C.execution_manifest(C.PROTOCOL, "a" * 40, self.sha)

    def test_execution_manifest_rejects_runtime_protocol_missing_from_current_head(self):
        runtime = ROOT / "bench/protocols/selection_comparison_runtime_v2.json"
        relative = runtime.relative_to(ROOT).as_posix()
        raw = runtime.read_bytes()
        def git(argv, **_kwargs):
            if argv[-1] == f"HEAD:{relative}":
                return SimpleNamespace(returncode=0, stdout=b"staged_reversion")
            return SimpleNamespace(returncode=0, stdout=raw)
        with patch.object(C, "verify_sources", return_value={}), \
                patch.object(C.subprocess, "run", side_effect=git), self.assertRaises(ValueError):
            C.execution_manifest(runtime, "a" * 40, self.sha)


if __name__ == "__main__":
    unittest.main(verbosity=2)
