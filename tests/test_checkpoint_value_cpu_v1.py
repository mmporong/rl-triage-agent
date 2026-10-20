from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))
import checkpoint_value_cpu_v1 as P


def tensor_pair(root, name="case", *, scale=2.0, zero=False):
    import torch
    folder = root / name
    folder.mkdir()
    initial = {"critic.weight": torch.tensor([[0.0, 0.0] if zero else [1.0, 2.0]]),
               "critic.bias": torch.tensor([0.0 if zero else 3.0]),
               "actor.weight": torch.tensor([[0.0, 0.0] if zero else [2.0, 4.0]])}
    final = {key: (torch.ones_like(value) if zero else value * scale) for key, value in initial.items()}
    first, last = folder / "model_0.pt", folder / "model_299.pt"
    torch.save({"model_state_dict": initial}, first)
    torch.save({"model_state_dict": final}, last)
    return {"first": P.descriptor(first), "last": P.descriptor(last)}


def fixture(root):
    plan = {"schema": P.PLAN_SCHEMA, **P.SCOPE, "source_run": str(root / "archive"),
            "pairs": {"case": tensor_pair(root), "reference": tensor_pair(root, "reference")},
            "contexts": [{"id": "fixture_e1", "seed": 1, "epoch": 1, "applied_before": [],
                          "packet_sha256": "a" * 64, "packet_digest": "b" * 64,
                          "jobs": {"case": "case", "reference": "reference"}}]}
    path = root / "fixture_plan.json"
    P.write_new(path, plan)
    return path, plan


def archive_artifact(root):
    training, run_dir = root / "training", root / "checkpoint_run"
    training.mkdir()
    run_dir.mkdir()
    manifest = training / "manifest.json"
    P.write_new(manifest, {"seed": 1, "max_iterations": 300})
    first, last = run_dir / "model_0.pt", run_dir / "model_299.pt"
    first.write_bytes(b"original first bytes, not a tensor measurement")
    last.write_bytes(b"original last bytes, not a tensor measurement")
    evidence = {"device": "cpu", "first": {**P.descriptor(first), "iteration": 0},
                "final": {**P.descriptor(last), "iteration": 299}}
    P.write_new(training / "checkpoint.json", evidence)
    artifact = {"manifest_path": str(manifest), "run_dir": str(run_dir),
                "checkpoint": P.descriptor(last), "checkpoint_evidence": evidence}
    return artifact, {str(manifest): P.sha(manifest)}


class CpuArchiveTests(unittest.TestCase):
    def test_first_and_last_are_bound_to_historical_embedded_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            artifact, inventory = archive_artifact(Path(folder))
            result = P.anchored_pair(artifact, inventory, artifact["checkpoint"]["sha256"])
            self.assertEqual(result["first"]["sha256"], artifact["checkpoint_evidence"]["first"]["sha256"])
            self.assertEqual(result["historical_evidence_digest"], P.L.digest(artifact["checkpoint_evidence"]))

    def test_first_modified_before_plan_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            artifact, inventory = archive_artifact(Path(folder))
            Path(artifact["checkpoint_evidence"]["first"]["path"]).write_bytes(b"replacement")
            with self.assertRaisesRegex(ValueError, "체크포인트 SHA 불일치"):
                P.anchored_pair(artifact, inventory, artifact["checkpoint"]["sha256"])

    def test_changed_checkpoint_json_cannot_replace_sealed_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            artifact, inventory = archive_artifact(Path(folder))
            evidence_path = Path(artifact["manifest_path"]).with_name("checkpoint.json")
            changed = deepcopy(artifact["checkpoint_evidence"])
            changed["first"]["sha256"] = "0" * 64
            evidence_path.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checkpoint_evidence 불일치"):
                P.anchored_pair(artifact, inventory, artifact["checkpoint"]["sha256"])

    def test_other_run_path_iteration_and_packet_binding_are_rejected(self):
        for mutation in ("first_path", "iteration", "packet_sha"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as folder:
                artifact, inventory = archive_artifact(Path(folder))
                bound = artifact["checkpoint"]["sha256"]
                if mutation == "first_path":
                    artifact["checkpoint_evidence"]["first"]["path"] = str(Path(folder) / "other_run/model_0.pt")
                elif mutation == "iteration":
                    artifact["checkpoint_evidence"]["first"]["iteration"] = 1
                else:
                    bound = "0" * 64
                evidence_path = Path(artifact["manifest_path"]).with_name("checkpoint.json")
                evidence_path.write_text(json.dumps(artifact["checkpoint_evidence"]), encoding="utf-8")
                with self.assertRaises(ValueError):
                    P.anchored_pair(artifact, inventory, bound)

    def test_unsealed_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            artifact, inventory = archive_artifact(Path(folder))
            Path(artifact["manifest_path"]).write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "manifest inventory 불일치"):
                P.anchored_pair(artifact, inventory, artifact["checkpoint"]["sha256"])


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "Isaac 번들 CPU torch 환경에서 실행")
class CpuDefinitionTests(unittest.TestCase):
    def test_finite_measurement_matches_frozen_function_exactly(self):
        import probes as frozen
        import torch
        for zero in (False, True):
            with self.subTest(zero=zero), tempfile.TemporaryDirectory() as folder:
                pair = tensor_pair(Path(folder), zero=zero)
                env = SimpleNamespace(unwrapped=SimpleNamespace(scene={"robot": None}))
                runner = SimpleNamespace(alg=SimpleNamespace(policy=None))
                original = frozen.measure("P_value", env, runner, Path(pair["last"]["path"]))
                self.assertEqual(P.load_pair(pair), original)
                self.assertFalse(torch.cuda.is_initialized())

    def test_changed_checkpoint_is_rejected_before_loading(self):
        import torch
        with tempfile.TemporaryDirectory() as folder:
            pair = tensor_pair(Path(folder))
            Path(pair["last"]["path"]).write_bytes(b"changed")
            with patch.object(torch, "load") as loader:
                with self.assertRaisesRegex(ValueError, "승인 뒤 체크포인트 변경"):
                    P.load_pair(pair)
                loader.assert_not_called()

    def test_nonfinite_checkpoint_is_invalid_not_normal(self):
        import torch
        with tempfile.TemporaryDirectory() as folder:
            pair = tensor_pair(Path(folder))
            torch.save({"model_state_dict": {"critic.weight": torch.tensor([float("nan")]),
                                            "actor.weight": torch.tensor([1.0])}}, pair["first"]["path"])
            pair["first"] = P.descriptor(Path(pair["first"]["path"]))
            torch.save({"model_state_dict": {"critic.weight": torch.tensor([1.0]),
                                            "actor.weight": torch.tensor([2.0])}}, pair["last"]["path"])
            pair["last"] = P.descriptor(Path(pair["last"]["path"]))
            with self.assertRaisesRegex(ValueError, "비유한"):
                P.load_pair(pair)

    def test_cuda_already_initialized_is_rejected(self):
        import torch
        with patch.object(torch.cuda, "is_initialized", return_value=True):
            with self.assertRaisesRegex(ValueError, "CUDA 초기화"):
                P.load_pair({})

    def test_checkpoint_loader_never_falls_back_to_unrestricted_pickle(self):
        import torch
        with tempfile.TemporaryDirectory() as folder:
            pair = tensor_pair(Path(folder))
            with patch.object(torch, "load", wraps=torch.load) as loader:
                P.load_pair(pair)
                self.assertEqual(loader.call_count, 2)
                self.assertTrue(all(call.kwargs.get("weights_only") is True for call in loader.call_args_list))
            torch.save({"model_state_dict": {}, "infos": SimpleNamespace(untrusted=True)}, pair["last"]["path"])
            pair["last"] = P.descriptor(Path(pair["last"]["path"]))
            with self.assertRaises(P.MeasurementFailure) as failed:
                P.load_pair(pair)
            self.assertEqual(failed.exception.code, "weights_only_load_failed")

    def test_deserialization_uses_the_hashed_bytes_when_path_changes_during_load(self):
        import torch
        with tempfile.TemporaryDirectory() as folder:
            pair = tensor_pair(Path(folder))
            expected = P.load_pair(pair)
            original_load = torch.load
            def replace_path_then_load(snapshot, **kwargs):
                Path(pair["last"]["path"]).write_bytes(b"replacement after hash")
                return original_load(snapshot, **kwargs)
            with patch.object(torch, "load", side_effect=replace_path_then_load):
                self.assertEqual(P.load_pair(pair), expected)

    def test_malformed_parameter_structures_have_explicit_failure_codes(self):
        import torch
        states = [("model_state_dict_missing", {"policy": {}}),
                  ("parameter_prefix_missing", {"model_state_dict": {"actor.weight": torch.ones(1)}}),
                  ("parameter_structure_invalid", {"model_state_dict": {"actor.weight": torch.ones(1),
                                                                       "critic.weight": torch.ones(3)}})]
        for code, saved in states:
            with self.subTest(code=code), tempfile.TemporaryDirectory() as folder:
                pair = tensor_pair(Path(folder))
                torch.save(saved, pair["first"]["path"])
                pair["first"] = P.descriptor(Path(pair["first"]["path"]))
                with self.assertRaises(P.MeasurementFailure) as failed:
                    P.load_pair(pair)
                self.assertEqual(failed.exception.code, code)


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "Isaac 번들 CPU torch 환경에서 실행")
class CpuApprovalTests(unittest.TestCase):
    def setUp(self):
        protocol_folder = tempfile.TemporaryDirectory()
        self.addCleanup(protocol_folder.cleanup)
        protocol = Path(protocol_folder.name) / "cpu_fixture_protocol.json"
        P.write_new(protocol, {"schema": "cpu_fixture_only"})
        replacement = patch.object(P, "PROTOCOL", protocol)
        replacement.start()
        self.addCleanup(replacement.stop)

    def test_missing_approval_blocks_before_tensor_measurement(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            plan_path, plan = fixture(root)
            with patch.object(P, "verify_sources", return_value="fixture-anchor"), \
                    patch.object(P, "archive_plan", return_value=plan), \
                    patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "-1"}):
                output = root / "run"
                P.propose(plan_path, output)
                with patch.object(P, "load_pair") as loader:
                    with self.assertRaisesRegex(ValueError, "승인된 새 요청"):
                        P.execute(output)
                    loader.assert_not_called()
                self.assertFalse((output / "result.json").exists())

    def test_approved_cpu_measurement_and_single_consumption(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            plan_path, plan = fixture(root)
            with patch.object(P, "verify_sources", return_value="fixture-anchor"), \
                    patch.object(P, "archive_plan", return_value=plan), \
                    patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "-1"}):
                output = root / "run"
                P.propose(plan_path, output)
                P.approve(output, "fixture-controller")
                result = P.execute(output)
                self.assertEqual(result["status"], "complete")
                self.assertEqual(result["gpu_s"], 0)
                self.assertEqual(result["contexts"][0]["receipt"]["outcome"], "normal")
                for key, expected in P.SCOPE.items():
                    self.assertEqual(result[key], expected)
                    self.assertEqual(P.prereg(plan, plan["contexts"][0], "fixture-anchor")[key], expected)
                before = (output / "ledger.jsonl").read_bytes()
                with self.assertRaisesRegex(ValueError, "기존 결과"):
                    P.execute(output)
                self.assertEqual((output / "ledger.jsonl").read_bytes(), before)

    def test_plan_change_after_approval_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            plan_path, plan = fixture(root)
            with patch.object(P, "verify_sources", return_value="fixture-anchor"), \
                    patch.object(P, "archive_plan", return_value=plan):
                output = root / "run"
                P.propose(plan_path, output)
                P.approve(output, "fixture-controller")
                changed = deepcopy(plan)
                changed["contexts"][0]["packet_digest"] = "f" * 64
                (output / "plan.json").write_text(json.dumps(changed), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "plan·입력 변경"):
                    P.checked_batch(output)

    def test_failed_measurement_seals_invalid_and_unknown_receipt(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            plan_path, plan = fixture(root)
            with patch.object(P, "verify_sources", return_value="fixture-anchor"), \
                    patch.object(P, "archive_plan", return_value=plan), \
                    patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "-1"}):
                output = root / "run"
                P.propose(plan_path, output)
                P.approve(output, "fixture-controller")
                with patch.object(P, "load_pair", side_effect=ValueError("fixture failure")):
                    with self.assertRaisesRegex(ValueError, "fixture failure"):
                        P.execute(output)
                self.assertEqual(P.read(output / "result.json")["status"], "invalid")
                ledger = P.M.RegistryLedger(output / "ledger.jsonl", P.C.registry())
                request = next(iter(ledger.requests().values()))
                self.assertEqual(request["state"], "done")
                self.assertEqual(request["receipt"]["outcome"], "unknown")
                self.assertEqual(request["receipt"]["exit_code"], 1)
                failure = P.read(output / "result.json")["failure"]
                self.assertEqual(failure["failure_code"], "unclassified_execution_error")
                self.assertEqual(failure["failed_context_id"], "fixture_e1")
                self.assertEqual(failure["failed_request_id"], request["request_id"])
                self.assertEqual(request["receipt"]["measurement"]["failure"], failure)

    def test_zero_reference_change_is_invalid_even_when_case_is_also_zero(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            plan_path, plan = fixture(root)
            plan["pairs"] = {key: tensor_pair(root, "unchanged_" + key, scale=1.0)
                             for key in ("case", "reference")}
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            with patch.object(P, "verify_sources", return_value="fixture-anchor"), \
                    patch.object(P, "archive_plan", return_value=plan), \
                    patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "-1"}), \
                    patch.object(P.L, "classify") as classifier:
                output = root / "run"
                P.propose(plan_path, output)
                P.approve(output, "fixture-controller")
                with self.assertRaises(P.MeasurementFailure) as failed:
                    P.execute(output)
                self.assertEqual(failed.exception.code, "reference_critic_change_not_positive")
                classifier.assert_not_called()
                result = P.read(output / "result.json")
                self.assertEqual(result["status"], "invalid")
                self.assertEqual(result["failure"]["failed_pair_key"], "reference")
                ledger = P.M.RegistryLedger(output / "ledger.jsonl", P.C.registry())
                receipt = next(iter(ledger.requests().values()))["receipt"]
                self.assertEqual(receipt["outcome"], "unknown")

    def test_failure_codes_keep_stage_request_context_and_pair_without_raw_error(self):
        for code in ("checkpoint_sha_mismatch", "weights_only_load_failed", "parameter_nonfinite"):
            with self.subTest(code=code), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                plan_path, plan = fixture(root)
                with patch.object(P, "verify_sources", return_value="fixture-anchor"), \
                        patch.object(P, "archive_plan", return_value=plan), \
                        patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "-1"}):
                    output = root / "run"
                    batch = P.propose(plan_path, output)
                    P.approve(output, "fixture-controller")
                    with patch.object(P, "load_pair", side_effect=P.MeasurementFailure(code)):
                        with self.assertRaises(P.MeasurementFailure):
                            P.execute(output)
                    result = P.read(output / "result.json")
                    failure = result["failure"]
                    self.assertEqual(failure, {"failure_code": code, "failure_stage": P.FAILURES[code][0],
                        "failed_context_id": "fixture_e1", "failed_pair_key": "case",
                        "failed_request_id": batch["requests"][0]["request_id"]})
                    self.assertNotIn(str(root), json.dumps(result))

    def test_middle_approval_event_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            plan_path, plan = fixture(root)
            with patch.object(P, "verify_sources", return_value="fixture-anchor"), \
                    patch.object(P, "archive_plan", return_value=plan):
                output = root / "run"
                P.propose(plan_path, output)
                P.approve(output, "fixture-controller")
                path = output / "ledger.jsonl"
                events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
                events[1]["effect_key"] = "changed"
                path.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "중간 원장 결합"):
                    P.checked_batch(output)

    def test_blank_middle_approver_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            plan_path, plan = fixture(root)
            with patch.object(P, "verify_sources", return_value="fixture-anchor"), \
                    patch.object(P, "archive_plan", return_value=plan):
                output = root / "run"
                P.propose(plan_path, output)
                P.approve(output, "fixture-controller")
                path = output / "ledger.jsonl"
                events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
                events[1]["approver"] = " "
                path.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "승인 주체 누락"):
                    P.checked_batch(output)

    def test_changed_gpu_budget_is_rejected_before_consumption(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            plan_path, plan = fixture(root)
            with patch.object(P, "verify_sources", return_value="fixture-anchor"), \
                    patch.object(P, "archive_plan", return_value=plan):
                output = root / "run"
                P.propose(plan_path, output)
                path = output / "ledger.jsonl"
                events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
                events[0]["budget_gpu_s"] = 1
                path.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "원장 요청 결합"):
                    P.checked_batch(output)


if __name__ == "__main__":
    unittest.main()
