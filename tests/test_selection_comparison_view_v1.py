"""공통뷰 계약의 합성 CPU 회귀. 모델·GPU·Isaac·저장 checkpoint를 호출하지 않는다."""

from __future__ import annotations

import itertools
import json
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))
import selection_comparison_view_v1 as V  # noqa: E402

FACT = "telemetry:case:Metrics/base_velocity/error_vel_xy"


def packet(budget=7, epoch=1):
    applied = [] if epoch == 1 else [V.C.REWARD]
    prior = None if epoch == 1 else {
        "verdict": "not_recovered", "correction_id": V.C.REWARD,
        "result_sha256": "e" * 64, "checkpoint_sha256": "a" * 64,
    }
    return V.S.build_packet(
        epoch=epoch, applied_corrections=applied, prior_receipt=prior,
        facts={FACT: 1.2, "telemetry:reference:Metrics/base_velocity/error_vel_xy": 0.15,
               "config:action_scale": 0.25},
        input_bindings={"case": {"checkpoint_sha256": "a" * 64, "telemetry_sha256": "b" * 64},
                        "reference": {"checkpoint_sha256": "c" * 64, "telemetry_sha256": "d" * 64}},
        budget_left=budget, execution_budget_s=100,
    )


class SharedViewTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.protocol_sha = V.file_digest(V.PROTOCOL)

    def snapshot(self, policy="model_adaptive", *, value=None, name="arm", parent=None, masked=False):
        return V.build_snapshot(packet() if value is None else value, "fixture_run", policy,
                                self.protocol_sha, self.directory / (name + ".jsonl"), parent, masked)

    def append(self, snapshot, probe="P_value", outcome="normal", measurement=None,
               seconds=1, stage="done", approver="synthetic_fixture"):
        ledger = V.M.RegistryLedger(Path(snapshot["inputs"]["ledger_path"]), V.C.registry())
        binding = snapshot["binding"]
        req = ledger.propose(binding, probe, int(snapshot["view"]["execution_budget_s"]))
        if stage == "pending":
            return ledger
        ledger.approve(req["request_id"], approver)
        if stage == "approved":
            return ledger
        ledger.consume(req["request_id"], binding, probe)
        if stage == "running":
            return ledger
        ledger.receipt(req["request_id"], outcome, {} if measurement is None else measurement, seconds, 0)
        return ledger

    def fresh(self, snapshot, **changes):
        return V.build_snapshot(**{**snapshot["inputs"], **changes})

    def reply(self, snapshot, *, probe="P_value", action="continue", facts=None, status="completed"):
        return {"context_digest": V.L.digest(snapshot["binding"]), "view_digest": snapshot["view_digest"],
                "status": status, "content": json.dumps({"action": action,
                "probe": probe if action == "continue" else None, "fact_ids": [FACT] if facts is None else facts})}

    def rewrite(self, snapshot, transform):
        path = Path(snapshot["inputs"]["ledger_path"])
        events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        transform(events)
        path.write_text("".join(json.dumps(row) + "\n" for row in events), encoding="utf-8")

    def test_four_policies_have_identical_information_without_host_bindings(self):
        snapshots = [self.snapshot(policy, name=policy) for policy in V.POLICIES]
        self.assertEqual(len({V.canonical(s["view"]) for s in snapshots}), 1)
        self.assertEqual(len({json.dumps(V.messages(s)) for s in snapshots}), 1)
        body = V.messages(snapshots[0])[-1]["content"]
        for secret in ("fixture_run", "model_adaptive", str(self.directory), "a" * 64,
                       "restore_body_frame_reward", '"args"', '"prior_receipt"'):
            self.assertNotIn(secret, body)
        self.assertTrue(all("expect" in row for row in snapshots[0]["view"]["probe_catalog"]))
        self.assertIn("null means", V.SYSTEM)
        self.assertIn("uniquely abnormal", V.SYSTEM)
        self.assertFalse(Path(snapshots[0]["inputs"]["ledger_path"]).exists())

    def test_adapter_matches_legacy_rule_for_all_candidate_and_observed_subsets(self):
        registry = V.C.registry()
        view = self.snapshot()["view"]
        probes = list(registry["probes"])
        comparisons = 0
        for n in range(1, len(registry["mechanisms"]) + 1):
            for candidates in itertools.combinations(registry["mechanisms"], n):
                for bits in itertools.product((False, True), repeat=len(probes)):
                    observed = {name: "normal" for name, used in zip(probes, bits, strict=True) if used}
                    status = V.M._status(registry, list(candidates), observed, 7)
                    current = {**view, "remaining_candidates": list(candidates), "status": status,
                               "observations": [{"probe": p, "outcome": o} for p, o in observed.items()],
                               "eligible_probes": [p for p in probes if V.S._probe_semantic(
                                   p, list(candidates), observed)[0]]}
                    expected = V.M._next(registry, list(candidates), observed) if status == "open" else None
                    self.assertEqual(V.rule_probe(current, "registry_next"), expected)
                    fixed = next((p for p in registry["fixed_order"]
                                  if p in current["eligible_probes"]), None) if status == "open" else None
                    self.assertEqual(V.rule_probe(current, "fixed_eligible"), fixed)
                    comparisons += 1
        self.assertEqual(comparisons, 16256)
        with patch.object(V.C, "registry", side_effect=AssertionError("공개뷰 밖 registry 읽기")):
            self.assertEqual(V.rule_probe(view, "registry_next"), "P_value")
            self.assertEqual(V.rule_probe(view, "fixed_eligible"), "P_episode")

    def test_projection_adds_public_numbers_without_raw_paths_or_metadata(self):
        s = self.snapshot()
        self.append(s, measurement={"case": {"critic_change": 0.2, "actor_change": 0.3},
                                   "reference": {"critic_change": 0.4}, "source": "secret_checkpoint_path",
                                   "seed": 2120, "trace": {"answer": "hidden"}})
        current = self.fresh(s)
        self.assertNotIn("optimizer_value", current["view"]["remaining_candidates"])
        self.assertEqual(current["view"]["evidence_facts"]["observation:0:P_value:case/critic_change"], 0.2)
        self.assertNotIn("secret_checkpoint_path", V.messages(current)[-1]["content"])
        self.assertEqual(V.rule_probe(current["view"], "registry_next"), "P_physics")

    def test_latest_outcome_and_measurement_changes_do_not_leak_into_shadow(self):
        controls, full = [], []
        for name, outcome, metric in (("normal", "normal", 0.0), ("abnormal", "abnormal", 999.0)):
            s = self.snapshot(name=name)
            self.append(s, "P_reward", outcome, {"track_lin_vel_xy_exp_rel_error": metric}, seconds=2)
            full.append(self.fresh(s))
            controls.append(self.fresh(s, masked=True))
        self.assertNotEqual(full[0]["view_digest"], full[1]["view_digest"])
        self.assertEqual(V.messages(controls[0]), V.messages(controls[1]))
        self.assertEqual(controls[0]["pre_receipt_anchor"], controls[1]["pre_receipt_anchor"])
        self.assertEqual(controls[0]["pre_receipt_anchor"], full[0]["pre_receipt_anchor"])
        self.assertEqual(controls[0]["view"]["remaining_candidates"], packet()["remaining_candidates"])
        self.assertEqual(controls[0]["view"]["budget_left"], 6)
        self.assertEqual(controls[0]["view"]["recorded_probe_gpu_s"], 2)
        self.assertNotIn("P_reward", controls[0]["view"]["eligible_probes"])
        self.assertEqual(controls[0]["view"]["evidence_facts"], packet()["facts"])
        choice = V.select(controls[0], self.reply(controls[0]))
        self.assertFalse(controls[0]["proposal_allowed"])
        with self.assertRaises(ValueError):
            V.request_spec(controls[0], choice, self.reply(controls[0]))

    def test_shadow_keeps_earlier_observations_but_rolls_back_latest_candidates(self):
        s = self.snapshot()
        self.append(s)
        before = self.fresh(s)
        self.append(before, "P_physics", "abnormal", {"mismatch_count": 9})
        shadow = self.fresh(s, masked=True)
        self.assertEqual(shadow["view"]["remaining_candidates"], before["view"]["remaining_candidates"])
        self.assertIn("observation:0:P_value:outcome", shadow["view"]["evidence_facts"])
        self.assertFalse(any(k.startswith("observation:1:") for k in shadow["view"]["evidence_facts"]))
        self.assertNotIn("P_physics", shadow["view"]["eligible_probes"])

    def test_adaptive_calls_model_again_but_first_only_uses_registered_rule(self):
        for policy in ("model_first_then_rule", "model_adaptive"):
            s = self.snapshot(policy, name=policy)
            self.append(s)
            s = self.fresh(s)
            if policy == "model_adaptive":
                self.assertEqual(V.select(s)["reason"], "model_unavailable")
                self.assertEqual(V.select(s, self.reply(s, probe="P_reward"))["selected_by"], "model")
            else:
                choice = V.select(s)
                self.assertEqual((choice["selected_by"], choice["probe"]), ("registered_rule", "P_physics"))
                with self.assertRaises(ValueError):
                    V.select(s, self.reply(s))

    def test_only_proven_invalid_model_output_uses_fallback(self):
        s = self.snapshot()
        for reply, expected in ((None, "not_used"), (self.reply(s, status="infra_error"), "infra_error"),
                                (self.reply(s, status="unknown"), "unknown")):
            choice = V.select(s, reply)
            self.assertEqual(choice["selected_by"], "hold")
            self.assertEqual(choice["model_status"], expected)
        hold = V.select(s, self.reply(s, action="hold"))
        self.assertEqual((hold["selected_by"], hold["model_status"]), ("hold", "valid"))
        for content in ("not JSON", json.dumps({"action": "continue", "probe": "P_value", "fact_ids": [FACT],
                                               "args": {"unregistered": 1}})):
            reply = {**self.reply(s), "content": content}
            choice = V.select(s, reply)
            self.assertEqual((choice["selected_by"], choice["model_status"], choice["probe"]),
                             ("registered_rule", "invalid", "P_value"))
        choice = V.select(s, self.reply(s, facts=["missing"]))
        self.assertEqual(choice["codes"], ["S_FACT_UNKNOWN"])

    def test_other_context_stale_view_and_claimed_valid_envelopes_are_rejected(self):
        s = self.snapshot()
        for field, value in (("context_digest", "f" * 64), ("view_digest", "e" * 64), ("status", "valid")):
            with self.assertRaises(ValueError):
                V.select(s, {**self.reply(s), field: value})
        self.append(s)
        with self.assertRaises(ValueError):
            V.select(s, self.reply(s))
        new = self.fresh(s)
        with self.assertRaises(ValueError):
            V.select(new, self.reply(s))

    def test_request_candidate_uses_host_args_and_does_not_approve_or_write(self):
        s = self.snapshot()
        reply = self.reply(s)
        choice = V.select(s, reply)
        spec = V.request_spec(s, choice, reply)
        self.assertEqual(spec["args"], V.C.registry()["probes"]["P_value"]["args"])
        self.assertFalse(spec["approval_granted"])
        self.assertFalse(s["execution_authorized"])
        self.assertEqual(spec["budget_gpu_s"], s["view"]["execution_budget_s"])
        self.assertFalse(Path(s["inputs"]["ledger_path"]).exists())
        with self.assertRaises(ValueError):
            V.request_spec(s, {**choice, "probe": "P_reward"}, reply)

    def test_pending_last_slot_is_valid_but_holds_at_all_lifecycle_stages(self):
        for stage in ("pending", "approved", "running"):
            s = self.snapshot(value=packet(budget=1), name=stage)
            self.append(s, stage=stage)
            current = self.fresh(s)
            self.assertEqual((current["view"]["status"], current["view"]["budget_left"]), ("pending", 0))
            self.assertEqual(V.select(current)["selected_by"], "hold")
            self.assertFalse(current["proposal_allowed"])

    def test_unknown_confirmed_and_empty_budget_never_request_another_probe(self):
        for name, outcome, expected in (("unknown", "unknown", "unidentifiable"),
                                        ("abnormal", "abnormal", "confirmed")):
            s = self.snapshot(name=name)
            self.append(s, outcome=outcome)
            current = self.fresh(s)
            self.assertEqual(current["view"]["status"], expected)
            self.assertEqual(current["view"]["eligible_probes"], [])
            self.assertEqual(V.select(current)["selected_by"], "hold")
            self.append(current, "P_physics")
            with self.assertRaises(ValueError):
                self.fresh(s)
        self.assertFalse(self.snapshot(value=packet(budget=0), name="empty")["proposal_allowed"])

    def test_single_remaining_candidate_requires_its_confirmation(self):
        s = self.snapshot()
        for probe in V.C.registry()["probes"]:
            if probe != "P_episode":
                self.append(s, probe)
                s = self.fresh(s)
        self.assertEqual(s["view"]["remaining_candidates"], ["termination"])
        self.assertEqual(s["view"]["eligible_probes"], ["P_episode"])
        self.assertEqual(V.select(s, self.reply(s, probe="P_episode"))["model_status"], "valid")
        self.append(s, "P_episode", "abnormal")
        self.assertEqual(self.fresh(s)["view"]["status"], "confirmed")

    def test_epoch_two_parent_is_same_arm_and_output_checkpoint_but_not_authenticated(self):
        value = packet(epoch=2)
        parent = {"run_id": "fixture_run", "policy": "model_adaptive", "epoch": 1, **value["prior_receipt"]}
        s = self.snapshot(value=value, parent=parent)
        self.assertEqual(s["binding"]["parent"], parent)
        for field, changed in (("run_id", "other_run"), ("policy", "registry_next"),
                                ("result_sha256", "f" * 64), ("checkpoint_sha256", "f" * 64)):
            with self.assertRaises(ValueError):
                self.snapshot(value=value, parent={**parent, field: changed})
        value["input_bindings"]["case"]["checkpoint_sha256"] = "f" * 64
        with self.assertRaises(ValueError):
            self.snapshot(value=value, parent=parent)

    def test_context_policy_and_arguments_cannot_be_mixed_in_history(self):
        s = self.snapshot()
        self.append(s)
        for changes in ({"run_id": "other_run"}, {"policy": "registry_next"}):
            with self.assertRaises(ValueError):
                self.fresh(s, **changes)
        self.rewrite(s, lambda rows: rows[0].update(args_digest="f" * 64))
        with self.assertRaises(ValueError):
            self.fresh(s)

    def test_invalid_approval_event_order_or_receipt_is_rejected(self):
        mutations = (
            lambda rows: rows[1].update(approver=" "),
            lambda rows: rows[2].update(effect_key="bad"),
            lambda rows: rows.reverse(),
            lambda rows: rows[3].update(exit_code=1),
            lambda rows: rows[3].update(gpu_s=float("nan")),
            lambda rows: rows.append({**rows[3], "request_id": "orphan"}),
            lambda rows: rows.insert(1, deepcopy(rows[0])),
        )
        for index, mutation in enumerate(mutations):
            with self.subTest(index=index):
                s = self.snapshot(name=f"invalid{index}")
                self.append(s)
                self.rewrite(s, mutation)
                with self.assertRaises(ValueError):
                    self.fresh(s)

    def test_duplicate_probe_and_pending_followed_by_another_request_are_rejected(self):
        for index, stage in enumerate(("done", "pending")):
            s = self.snapshot(name=f"duplicate{index}")
            ledger = self.append(s, stage=stage)
            if stage == "done":
                self.append(s)
            else:
                ledger.propose(s["binding"], "P_physics", 100)
            with self.assertRaises(ValueError):
                self.fresh(s)

    def test_base_facts_and_numeric_projection_do_not_accept_metadata_strings(self):
        for key, value in (("config:path", 1), ("config:seed", 2026),
                           ("config:training_seed", 2026), ("config:fault_id", 1),
                           ("config:checkpoint_path", 123), ("telemetry:case:unregistered_stat", 2),
                           (FACT, "secret"), (FACT, float("inf"))):
            bad = packet()
            bad["facts"][key] = value
            with self.assertRaises(ValueError):
                self.snapshot(value=bad)
        for index, value in enumerate(("secret", True, {"bad": 1})):
            s = self.snapshot(name=f"projection{index}")
            self.append(s, measurement={"critic_change": value})
            with self.assertRaises(ValueError):
                self.fresh(s)

    def test_snapshot_and_original_input_mutations_do_not_change_frozen_messages(self):
        original = packet()
        s = self.snapshot(value=original)
        original["facts"][FACT] = 999
        self.assertEqual(s["view"]["evidence_facts"][FACT], 1.2)
        changed = deepcopy(s)
        changed["view"]["remaining_candidates"] = ["reward"]
        with self.assertRaises(ValueError):
            V.select(changed, self.reply(changed))

    def test_protocol_and_source_changes_fail_closed_before_selection(self):
        with self.assertRaises(ValueError):
            V.epoch_binding(packet(), "fixture_run", "model_adaptive", "0" * 64)
        original = V.file_digest
        def changed(path):
            return "0" * 64 if path == ROOT / "evals/residual_selection_v2.py" else original(path)
        with patch.object(V, "file_digest", side_effect=changed), self.assertRaises(ValueError):
            self.snapshot()

    def test_protocol_parser_uses_only_the_bytes_whose_hash_was_checked(self):
        raw = V.PROTOCOL.read_bytes()
        changing_path = SimpleNamespace(read_bytes=lambda: raw,
                                        read_text=lambda **_kwargs: '{"schema":"changed_between_reads"}')
        with patch.object(V, "PROTOCOL", changing_path):
            self.assertEqual(V.verify_protocol(self.protocol_sha)["measurement_kind"],
                             "synthetic_cpu_contract_only")

    def test_fractional_packet_and_different_ledger_budgets_are_rejected(self):
        for budget in (0.5, 100.0, 100.9):
            bad = packet()
            bad["execution_budget_s"] = budget
            with self.assertRaises(ValueError):
                self.snapshot(value=bad)
        for budget in (0, 50, 101):
            s = self.snapshot(name=f"budget{budget}")
            ledger = V.M.RegistryLedger(Path(s["inputs"]["ledger_path"]), V.C.registry())
            ledger.propose(s["binding"], "P_value", budget)
            with self.assertRaises(ValueError):
                self.fresh(s)


if __name__ == "__main__":
    unittest.main(verbosity=2)
