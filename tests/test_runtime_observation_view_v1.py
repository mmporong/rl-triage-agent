"""검증 runtime facts를 기존 V/C/CALL2 경로에 추가하는 CPU 회귀."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))

import runtime_observation_view_v1 as RV
import selection_comparison_call_v2 as CALL2
import selection_comparison_controller_v1 as C

V = RV.V
OLD_FACT = "telemetry:case:Metrics/base_velocity/error_vel_xy"
NEW_FACT = "runtime:case:optimizer_steps_min"


def packet(epoch: int = 1, *, include_runtime: bool = True) -> dict:
    facts = {
        OLD_FACT: 1.2,
        "telemetry:reference:Metrics/base_velocity/error_vel_xy": 0.15,
        "config:action_scale": 0.25,
    }
    if include_runtime:
        facts.update({
            f"runtime:{role}:{name}": (
                True if name == "available" else 3 if name == "updates_observed" else 15)
            for role in ("case", "reference") for name in RV.FACT.FACT_NAMES
        })
    prior = None if epoch == 1 else {
        "verdict": "not_recovered", "correction_id": V.C.REWARD,
        "result_sha256": "e" * 64, "checkpoint_sha256": "a" * 64,
    }
    return V.S.build_packet(
        epoch=epoch, applied_corrections=[] if epoch == 1 else [V.C.REWARD],
        prior_receipt=prior, facts=facts,
        input_bindings={
            "case": {"checkpoint_sha256": "a" * 64, "telemetry_sha256": "b" * 64},
            "reference": {"checkpoint_sha256": "c" * 64, "telemetry_sha256": "d" * 64},
        },
        budget_left=7, execution_budget_s=100,
    )


class RuntimeObservationViewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.protocol = self.directory / "runtime_observation_view_v1.json"
        self.protocol.write_bytes(RV.canonical(RV.prepare_protocol()))
        self.protocol_sha = RV.file_digest(self.protocol)

    def snapshot(self, *, value: dict | None = None, parent: dict | None = None) -> dict:
        return V.build_snapshot(
            packet() if value is None else value, "fixture_run", "model_adaptive",
            self.protocol_sha, self.directory / "ledger.jsonl", parent,
        )

    @staticmethod
    def complete(content: dict):
        row = {
            "provider_model": CALL2.MODEL, "finish_reason": "stop",
            "content": json.dumps(content), "wall_s": 0.1,
        }
        return CALL2.FixtureResponses([row, row])

    def dialogue(self, snapshot: dict, content: dict) -> dict:
        inputs = C.dialogue_input(snapshot)
        engine = CALL2.engine(
            snapshot, inputs["view"], inputs["messages"], "production", self.complete(content))
        return {
            "schema": "selection_controller_dialogue_v1",
            "input_digest": V.L.digest(inputs),
            "safe_results": [row["result"] for row in engine["calls"]],
            "engine": engine,
        }

    def test_full_v_controller_call_contract_accepts_new_fact_and_preserves_old_facts(self):
        with RV.view_scope(self.protocol, None, fixture=True):
            snapshot = self.snapshot()
            checked = V.checked(snapshot)
            self.assertIn(NEW_FACT, checked["view"]["evidence_facts"])
            self.assertIn(OLD_FACT, checked["view"]["evidence_facts"])
            messages = V.messages(checked)
            inputs = C.dialogue_input(checked)
            self.assertEqual(messages, inputs["messages"])
            content = {"action": "continue", "probe": "P_value", "fact_ids": [NEW_FACT, OLD_FACT]}
            dialogue = self.dialogue(checked, content)
            choice = C.choose(checked, "model_adaptive", dialogue)
            spec = C.request_spec(checked, choice)
            self.assertEqual(choice["fact_ids"], [NEW_FACT, OLD_FACT])
            self.assertEqual(spec["probe"], "P_value")
            self.assertFalse(spec["approval_granted"])
            self.assertFalse(checked["execution_authorized"])

    def test_absent_or_unknown_fact_id_holds_and_cannot_make_request(self):
        with RV.view_scope(self.protocol, None, fixture=True):
            snapshot = self.snapshot(value=packet(include_runtime=False))
            for fact_id in (NEW_FACT, "runtime:case:not_registered"):
                with self.subTest(fact_id=fact_id):
                    content = {"action": "continue", "probe": "P_value", "fact_ids": [fact_id]}
                    choice = C.choose(snapshot, "model_adaptive", self.dialogue(snapshot, content))
                    self.assertEqual((choice["action"], choice["model_status"]), ("hold", "invalid"))
                    with self.assertRaises(ValueError):
                        C.request_spec(snapshot, choice)

    def test_protocol_scope_rejects_other_sha_dirty_source_and_restores_globals(self):
        old = (V.BASE_FACT_IDS, V.PROTOCOL, V.verify_protocol)
        with RV.view_scope(self.protocol, None, fixture=True):
            with self.assertRaises(ValueError):
                V.build_snapshot(packet(), "fixture_run", "model_adaptive", "0" * 64,
                                 self.directory / "wrong.jsonl")
            value = json.loads(self.protocol.read_text(encoding="utf-8"))
            value["source_sha256_lf"]["evals/optimizer_runtime_facts_v1.py"] = "0" * 64
            dirty = self.directory / "dirty.json"
            dirty.write_bytes(RV.canonical(value))
            with self.assertRaises(ValueError):
                RV.verify_protocol(dirty, None, fixture=True)
        self.assertEqual(V.BASE_FACT_IDS, old[0])
        self.assertIs(V.PROTOCOL, old[1])
        self.assertIs(V.verify_protocol, old[2])

    def test_production_requires_anchor_and_rejects_anchor_blob_mismatch(self):
        with self.assertRaises(ValueError):
            RV.verify_protocol(self.protocol, None)
        with patch.object(RV, "PROTOCOL", self.protocol), patch.object(
                RV, "_git", side_effect=[b"", b"wrong"]), self.assertRaises(ValueError):
            RV.verify_protocol(self.protocol, "a" * 40)

    def test_epoch_two_requires_explicit_post_receipt_parent(self):
        value = packet(epoch=2)
        parent = {"run_id": "fixture_run", "policy": "model_adaptive", "epoch": 1,
                  **value["prior_receipt"]}
        with RV.view_scope(self.protocol, None, fixture=True):
            accepted = self.snapshot(value=value, parent=parent)
            self.assertEqual(accepted["binding"]["parent"], parent)
            with self.assertRaises(ValueError):
                self.snapshot(value=value, parent=None)
            with self.assertRaises(ValueError):
                self.snapshot(value=value, parent={**parent, "result_sha256": "f" * 64})

    def test_legacy_ledger_bounds_and_pure_view_authority_are_unchanged(self):
        with RV.view_scope(self.protocol, None, fixture=True):
            snapshot = self.snapshot()
            ledger = V.M.RegistryLedger(Path(snapshot["inputs"]["ledger_path"]), V.C.registry())
            ledger.propose(snapshot["binding"], "P_value", 99)
            with self.assertRaises(ValueError):
                V.checked(snapshot)
            self.assertFalse(snapshot["approval_granted"])
            self.assertFalse(snapshot["execution_authorized"])
            self.assertTrue(snapshot["proposal_allowed"])

    def test_fixture_never_claims_actual_source_ownership(self):
        with RV.view_scope(self.protocol, None, fixture=True) as protocol:
            self.assertFalse(protocol["scope"]["fixture_claims_actual_ownership"])
            self.assertFalse(protocol["claims"]["runtime_fact_values_verified_by_view"])
            self.assertEqual(protocol["scope"]["actual_fact_validation"],
                             "optimizer_runtime_facts_v1.checked_and_registered_main_runner")


if __name__ == "__main__":
    unittest.main(verbosity=2)
