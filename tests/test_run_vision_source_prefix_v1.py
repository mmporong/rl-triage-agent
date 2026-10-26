from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / "bench", ROOT / "evals", ROOT / "src"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

S = importlib.import_module("run_vision_source_prefix_v1")


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _actions() -> list[dict[str, Any]]:
    actions = []
    for seed in S.SEEDS:
        for arm in ("none", "f1"):
            actions.append(
                {
                    "action_id": f"train-{seed}-{arm}", "kind": "train",
                    "job_id": f"vision-s{seed}-{arm}",
                }
            )
    for seed in S.SEEDS:
        actions.append(
            {
                "action_id": f"eval-{seed}", "kind": "evaluate", "seed": seed,
                "none_job_id": f"vision-s{seed}-none", "f1_job_id": f"vision-s{seed}-f1",
                "run_name": f"eval-{seed}",
            }
        )
    for seed in S.SEEDS:
        for arm in ("none", "f1"):
            actions.append(
                {
                    "action_id": f"record-{seed}-{arm}", "kind": "record",
                    "job_id": f"record-{seed}-{arm}",
                    "source_job_id": f"vision-s{seed}-{arm}",
                    "checkpoint_logical_id": f"policy-{seed}-{arm}",
                }
            )
    return actions


class FakeBatch:
    def __init__(self, root: Path):
        self.runtime = root / "runtime"
        self.lab = root / "lab"
        self.runtime.mkdir(parents=True)
        self.lab.mkdir(parents=True)
        self.protocol_sha = "1" * 64
        self.anchor = "2" * 40
        self.resource_sha = "3" * 64
        self.resource = {
            "resource_id": "owned-1", "gpu_uuid": "GPU-one", "lease_id": "lease-one"
        }
        plan = []
        for seed in S.SEEDS:
            for arm in ("NONE", "F1", "CORRECTED"):
                plan.append(
                    {
                        "job_id": f"vision-s{seed}-{arm.lower()}", "seed": seed, "arm": arm,
                        "source_job_id": f"vision-s{seed}-f1" if arm == "CORRECTED" else None,
                    }
                )
        self.protocol = {
            "protocol_id": "source-prefix-fixture",
            "training_plan": plan,
            "source_action_order": _actions(),
        }
        self.advance_calls = 0
        self.record_calls = 0
        self._raw_values: dict[int, dict[str, Any]] = {}

    def verify(self) -> dict[str, Any]:
        return {"status": "verified", "protocol_id": self.protocol["protocol_id"]}

    def _action_folder(self, index: int, action: dict[str, Any]) -> Path:
        return self.runtime / "actions" / f"{index:02d}_{action['action_id']}"

    def _raw(self, index: int) -> dict[str, Any]:
        action = self.protocol["source_action_order"][index]
        folder = self._action_folder(index, action)
        receipt, raw = folder / "receipt.json", folder / "raw.json"
        body = json.loads(receipt.read_text(encoding="utf-8"))
        if body != {"action_id": action["action_id"], "raw_sha256": _sha(raw)}:
            raise ValueError("fixture receipt SHA 결합이 다르다")
        return self._raw_values[index]

    def advance(self) -> dict[str, Any]:
        index = self.advance_calls
        action = self.protocol["source_action_order"][index]
        self.advance_calls += 1
        if action["kind"] == "record":
            self.record_calls += 1
            raise AssertionError("legacy record가 호출됐다")
        folder = self._action_folder(index, action)
        folder.mkdir(parents=True)
        if action["kind"] == "train":
            checkpoint = self.lab / "logs/rsl_rl" / action["job_id"] / "model.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(action["job_id"].encode())
            raw_value = {
                "kind": "train", "action_id": action["action_id"], "job_id": action["job_id"],
                "artifact": {"checkpoint": {"path": str(checkpoint), "sha256": _sha(checkpoint)}},
            }
        else:
            raw_value = {
                "kind": "evaluate", "action_id": action["action_id"], "job_id": action["action_id"],
                "seed": action["seed"], "source_gate": {"outcome": f"gate-{action['seed']}"},
            }
        self._raw_values[index] = raw_value
        raw = folder / "raw.json"
        _write(raw, raw_value)
        _write(folder / "receipt.json", {"action_id": action["action_id"], "raw_sha256": _sha(raw)})
        return {"status": "advanced", "completed": index + 1, "action_id": action["action_id"]}


def test_runs_only_six_actions_and_seals_replayable_handoff(tmp_path: Path) -> None:
    batch = FakeBatch(tmp_path)
    controller = S.SourcePrefixController(batch)  # type: ignore[arg-type]
    for completed in range(1, 6):
        result = controller.advance()
        assert result["status"] == "advanced"
        assert result["completed"] == completed
    result = controller.advance()
    assert result["status"] == "await_checkpoint_registration"
    assert batch.advance_calls == 6 and batch.record_calls == 0
    assert len(result["action_receipts"]) == 6
    assert len(result["checkpoints"]) == 4
    assert {row["isaaclab_relative"] for row in result["checkpoints"]} == {
        f"logs/rsl_rl/vision-s{seed}-{arm}/model.pt"
        for seed in S.SEEDS
        for arm in ("none", "f1")
    }
    assert [row["seed"] for row in result["evaluations"]] == list(S.SEEDS)
    handoff_bytes = controller.handoff_path.read_bytes()
    replay = controller.advance()
    assert replay == result
    assert controller.handoff_path.read_bytes() == handoff_bytes
    assert batch.advance_calls == 6


def test_pending_or_failed_action_holds_without_further_execution(tmp_path: Path) -> None:
    batch = FakeBatch(tmp_path)
    controller = S.SourcePrefixController(batch)  # type: ignore[arg-type]
    action = batch.protocol["source_action_order"][0]
    folder = batch._action_folder(0, action)
    folder.mkdir(parents=True)
    _write(folder / "intent.json", {"pending": True})
    result = controller.advance()
    assert result["status"] == "HOLD"
    assert result["reason"] == "source_prefix_action_pending_or_failed"
    assert batch.advance_calls == 0


def test_tampered_completed_receipt_and_legacy_record_state_fail_closed(tmp_path: Path) -> None:
    batch = FakeBatch(tmp_path)
    controller = S.SourcePrefixController(batch)  # type: ignore[arg-type]
    assert controller.advance()["status"] == "advanced"
    receipt = batch._action_folder(0, batch.protocol["source_action_order"][0]) / "receipt.json"
    _write(receipt, {"action_id": "wrong", "raw_sha256": "0" * 64})
    with pytest.raises(ValueError, match="receipt SHA"):
        controller.advance()
    assert batch.advance_calls == 1

    other = FakeBatch(tmp_path / "other")
    record = other.protocol["source_action_order"][6]
    other._action_folder(6, record).mkdir(parents=True)
    with pytest.raises(ValueError, match="legacy record"):
        S.SourcePrefixController(other).verify()  # type: ignore[arg-type]
    assert other.advance_calls == 0


def test_rejects_wrong_seed_contract_before_any_action(tmp_path: Path) -> None:
    batch = FakeBatch(tmp_path)
    batch.protocol["training_plan"][0]["seed"] = 9999
    with pytest.raises(ValueError, match="2210"):
        S.SourcePrefixController(batch)  # type: ignore[arg-type]
    assert batch.advance_calls == 0


def test_source_prefix_batch_blocks_record_dispatch_before_backend(tmp_path: Path) -> None:
    batch = object.__new__(S.SourcePrefixBatch)
    with pytest.raises(RuntimeError, match="legacy record"):
        batch._execute({"kind": "record"}, [])
