"""실제 API 없이 준비·상태 봉인·fixture 표시·오류 보존을 확인한다."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))
import selection_comparison_runtime_v1 as R  # noqa: E402

sys.path.insert(0, str(R.ROOT / "tests"))
import test_selection_comparison_view_v1 as fixtures  # noqa: E402


def bundle(path: Path) -> Path:
    rows = []
    for index, probe in enumerate(("P_reward", R.C.V.C.NEW_PROBE), 1):
        packet = fixtures.packet()
        packet["execution_budget_s"] = 1200
        packet["facts"] = {fixtures.FACT: float(index)}
        state = R.C.V.epoch_binding(packet, R.RUN_ID, "model_adaptive", R.C.V.file_digest(R.C.V.PROTOCOL))
        ledger = R.C.V.M.RegistryLedger(path.parent / f"fixture_{index}.jsonl", R.C.V.C.registry())
        request = ledger.propose(state, probe, 1200)
        ledger.approve(request["request_id"], "archive_replay_fixture")
        ledger.consume(request["request_id"], state, probe)
        ledger.receipt(request["request_id"], "abnormal", {}, 10 + index, 0)
        rows.append({"context_id": f"c{index}", "packet": packet,
                     "history_utf8": ledger.path.read_text(encoding="utf-8"),
                     "archive_packet_sha256": str(index) * 64, "archive_receipt_sha256": str(index + 2) * 64})
    R.C.MC.write_new(path, {"schema": "selection_runtime_inputs_v1", "run_id": R.RUN_ID,
                          "archive_integrity_sha256": "f" * 64, "cases": rows})
    return path


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)
        self.bundle = bundle(self.folder / "bundle.json")
        self.root = self.folder / "prepared"
        self.plan = R.execution_plan(self.bundle, self.root)
        self.paths = [next(path for path in (self.root / "capsules").glob("*.json")
                           if R.C.planned_state(R.C.MC.read(path)) == row["state"])
                      for row in self.plan["planned_calls"]]

    def test_fixed_order_pairs_and_budget_are_sealed(self):
        self.assertEqual(len(self.paths), 6)
        self.assertEqual(sum(row["max_requests"] for row in self.plan["planned_calls"]), 18)
        self.assertEqual([row["state"]["condition"] for row in self.plan["planned_calls"]],
                         [condition for _name, condition in R.ORDER])
        self.assertEqual(R.C.validate_plan(self.plan, self.plan["caller_protocol_sha"]), self.plan)
        for path in self.paths:
            body = R.C.prepared(self.root, path)[3][-1]["content"]
            self.assertNotIn("archive_", body)
            self.assertNotIn("fixture", body)

    def test_real_owner_changes_capsule_sha_but_not_preregistered_state(self):
        other = self.folder / "real_owner"
        owner = R.C.MC.read(self.root / "session.json")
        owner.update(fixture=False, experiment_sha="e" * 64, anchor="a" * 40)
        R.C.MC.write_new(other / "session.json", owner)
        paths = R.materialize(other, R.read_bundle(self.bundle))
        for old, new, row in zip(self.paths, paths, self.plan["planned_calls"], strict=True):
            self.assertNotEqual(old.name, new.name)
            self.assertEqual(R.C.planned_state(R.C.prepared(other, new)[0]), row["state"])
        with self.assertRaises(FileNotFoundError):
            R.summarize(other, paths, self.plan)

    def test_fixture_results_cannot_claim_real_runtime_or_add_historical_costs(self):
        response = {"provider_model": R.C.MODEL, "content": json.dumps({"action": "hold", "probe": None,
                    "fact_ids": [fixtures.FACT]}), "finish_reason": "stop", "input_tokens": 10,
                    "output_tokens": 5, "total_tokens": 15, "wall_s": 0.25}
        with patch.object(R.C.F, "openai_complete", side_effect=AssertionError("네트워크 금지")) as network:
            for path in self.paths:
                R.C.decide(self.root, path, R.C.FixtureResponses([response]))
            result = R.summarize(self.root, self.paths, self.plan)
            network.assert_not_called()
        self.assertEqual(result["status"], "complete")
        self.assertFalse(result["runtime_verified"])
        self.assertEqual(result["measurement_kind"], "synthetic_fixture")
        self.assertTrue(result["model_contract_prediction_met"])
        self.assertEqual(result["api_costs"]["wall_s"], 1.5)
        self.assertEqual(result["api_requests"], 6)
        self.assertTrue(all(not row["view_sensitivity"] for row in result["exposure_pairs"]))
        self.assertFalse(any(result["claims"].values()))

    def test_bundle_change_and_repeated_materialization_are_rejected(self):
        with self.assertRaises(ValueError):
            R.read_bundle(self.bundle, "0" * 64)
        with self.assertRaises(FileExistsError):
            R.materialize(self.root, R.read_bundle(self.bundle))
        self.assertEqual(R.C.audit(self.root)["reserved"], 0)

    def test_execution_requires_explicit_approval_before_manifest_or_network(self):
        with patch.object(R.C, "execution_manifest") as manifest, patch.object(R.C, "launch") as launch:
            with self.assertRaises(ValueError):
                R.execute(self.bundle, self.folder / "denied", "a" * 40, False)
            manifest.assert_not_called()
            launch.assert_not_called()

    def execute_fixture_failure(self, *, bad_audit=False, pending=False):
        root = self.folder / ("audit_failure" if bad_audit else "timeout")
        protocol = self.folder / ("audit_execution.json" if bad_audit else "timeout_execution.json")
        R.C.MC.write_new(protocol, self.plan)
        original = R.C.initialize
        def initialize(where, run_id, sha, **_kwargs):
            return original(where, run_id, sha, fixture=True)
        with patch.object(R, "PROTOCOL", protocol), \
                patch.object(R, "CLAIMS", self.folder / "claims"), \
                patch.object(R.C, "execution_manifest", return_value=self.plan), \
                patch.object(R, "verify_controller"), \
                patch.object(R.C, "initialize", side_effect=initialize), \
                patch.object(R.C, "launch", side_effect=TimeoutError("credential_sentinel")) as launch:
            if bad_audit:
                with patch.object(R.C.MC, "audit_budget", side_effect=ValueError("credential_sentinel")):
                    result = R.execute(self.bundle, root, "a" * 40, True)
            elif pending:
                with patch.object(R.C.MC, "audit_budget", return_value={"reserved": 1, "pending": 1}):
                    result = R.execute(self.bundle, root, "a" * 40, True)
            else:
                result = R.execute(self.bundle, root, "a" * 40, True)
            self.assertEqual(launch.call_count, 1)
        self.assertEqual(result["logical_calls_planned"], 6)
        self.assertEqual(result["logical_calls_completed"], 0)
        self.assertEqual(result["logical_calls_not_completed"], 6)
        self.assertEqual(result["status"], "invalid")
        self.assertEqual(set(result["api_costs"]), {"input_tokens", "output_tokens", "total_tokens", "wall_s"})
        self.assertTrue(all(value is None for value in result["api_costs"].values()))
        self.assertNotIn("credential_sentinel", json.dumps(result))
        return result

    def test_worker_failure_preserves_denominator_and_does_not_retry(self):
        self.assertEqual(self.execute_fixture_failure()["api_requests"], 0)

    def test_corrupt_budget_preserves_unknown_cost_instead_of_inventing_zero(self):
        result = self.execute_fixture_failure(bad_audit=True)
        self.assertIsNone(result["api_requests"])
        self.assertEqual(result["api_audit_error_type"], "ValueError")

    def test_pending_request_preserves_unknown_cost_and_stops(self):
        result = self.execute_fixture_failure(pending=True)
        self.assertEqual((result["api_requests"], result["api_pending"]), (1, 1))

    def test_same_execution_cannot_start_again_at_another_root(self):
        with patch.object(R, "CLAIMS", self.folder / "claims"):
            path = R.claim_execution(self.plan, self.folder / "first_root")
            original = path.read_bytes()
            with self.assertRaises(FileExistsError):
                R.claim_execution(self.plan, self.folder / "second_root")
            self.assertEqual(path.read_bytes(), original)

    def test_controller_commit_and_index_must_match_worktree(self):
        def git(argv, **_kwargs):
            relative = argv[-1].split(":", 1)[1]
            return SimpleNamespace(returncode=0, stdout=(R.ROOT / relative).read_bytes())
        with patch.object(R.C.subprocess, "run", side_effect=git):
            R.verify_controller(self.plan, "a" * 40)
        def wrong(argv, **kwargs):
            if argv[-1].startswith(":"):
                return SimpleNamespace(returncode=0, stdout=b"modified_index")
            return git(argv, **kwargs)
        with patch.object(R.C.subprocess, "run", side_effect=wrong), self.assertRaises(ValueError):
            R.verify_controller(self.plan, "a" * 40)

    def test_uncommitted_plan_cannot_open_real_launch(self):
        with patch.object(R.C, "launch") as launch, self.assertRaises(ValueError):
            R.execute(self.bundle, self.folder / "uncommitted", "f0ec12947095246a246a0e497d847d4e9cf4e665", True)
        launch.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
