# pyright: reportMissingImports=false
from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("evals", "tests", "src")]

import five_policy_recovery_epoch2_v1 as SECOND
import local_pair_probe_host_v1 as H
import recovery_recorded_pair_reuse_v2 as M
import test_recovery_recorded_pair_reuse_v1 as F


def current_snapshot(data: dict, ledger: Path) -> dict:
    old = data["snapshot"]
    return H.C.V.build_snapshot(
        old["inputs"]["packet"], old["inputs"]["run_id"], old["inputs"]["policy"],
        old["inputs"]["protocol_sha256"], ledger,
        parent=old["inputs"]["parent"], masked=old["inputs"]["masked"])


def install_delegate(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    probes: list[str] = []

    def delegate(snapshot: dict, choice: dict, **kwargs: object) -> dict:
        checked = H.C.V.checked(snapshot)
        spec = H.C.request_spec(checked, choice)
        ledger = H.C.S.M.RegistryLedger(
            Path(snapshot["inputs"]["ledger_path"]), H.C.RC.registry())
        request = ledger.propose(spec["prereg"], spec["probe"], spec["budget_gpu_s"])
        ledger.approve(request["request_id"], "fixture_operator")
        ledger.consume(request["request_id"], spec["prereg"], spec["probe"])
        ledger.receipt(request["request_id"], "normal", {}, 1.0, 0)
        out = Path(kwargs["out"])
        result_path = F.save(out / "result.json", {
            "probe": spec["probe"], "outcome": "normal", "delegated": True})
        probes.append(spec["probe"])
        return {"result_ref": H.file_ref(result_path), "delegated": True}

    monkeypatch.setattr(SECOND, "execute_pair", delegate)
    return probes


def test_first_reuse_then_same_case_second_and_third_delegate(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = F.physical_pair(tmp_path, monkeypatch)
    monkeypatch.setattr(SECOND, "_HOST", H)
    delegated = install_delegate(monkeypatch)
    ledger = tmp_path / "run/public.ledger.jsonl"
    first = F.new_snapshot(data, ledger)
    with M.recorded_pair_reuse_scope(
            original_result=data["result"], original_context=data["context"],
            original_choice=data["choice_path"]):
        replayed = SECOND.execute_pair(
            first, data["choice"], **F.arguments(data, first, tmp_path / "run/step01"))
        assert replayed["physical_execution_reused"] is True and delegated == []

        second = current_snapshot(data, ledger)
        second_choice = H.C.choose(second, "fixed_eligible")
        second_result = SECOND.execute_pair(
            second, second_choice, **F.arguments(data, second, tmp_path / "run/step02"))
        assert second_result["delegated"] is True

        third = current_snapshot(data, ledger)
        third_choice = H.C.choose(third, "fixed_eligible")
        third_result = SECOND.execute_pair(
            third, third_choice, **F.arguments(data, third, tmp_path / "run/step03"))
        assert third_result["delegated"] is True
    assert delegated[0] == "P_noise" and len(delegated) == 2


def test_followup_before_replay_and_empty_duplicate_are_rejected_without_delegate(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = F.physical_pair(tmp_path, monkeypatch)
    monkeypatch.setattr(SECOND, "_HOST", H)
    delegated = install_delegate(monkeypatch)
    ledger = tmp_path / "run/public.ledger.jsonl"
    first = F.new_snapshot(data, ledger)
    synthetic = deepcopy(first)
    synthetic["view"]["observations"] = [{"probe": "P_episode", "outcome": "normal"}]
    with M.recorded_pair_reuse_scope(
            original_result=data["result"], original_context=data["context"],
            original_choice=data["choice_path"]):
        with pytest.raises(ValueError, match="건너뛴"):
            SECOND.execute_pair(
                synthetic, data["choice"], **F.arguments(data, synthetic, tmp_path / "skip"))
        SECOND.execute_pair(
            first, data["choice"], **F.arguments(data, first, tmp_path / "run/step01"))
        duplicate = F.new_snapshot(data, tmp_path / "duplicate.ledger")
        with pytest.raises(ValueError, match="한 번"):
            SECOND.execute_pair(
                duplicate, data["choice"],
                **F.arguments(data, duplicate, tmp_path / "duplicate"))
    assert delegated == []


def test_followup_validates_current_ledger_choice_and_source_before_delegate(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = F.physical_pair(tmp_path, monkeypatch)
    monkeypatch.setattr(SECOND, "_HOST", H)
    delegated = install_delegate(monkeypatch)
    ledger = tmp_path / "run/public.ledger.jsonl"
    first = F.new_snapshot(data, ledger)
    with M.recorded_pair_reuse_scope(
            original_result=data["result"], original_context=data["context"],
            original_choice=data["choice_path"]):
        SECOND.execute_pair(
            first, data["choice"], **F.arguments(data, first, tmp_path / "run/step01"))
        second = current_snapshot(data, ledger)
        choice = H.C.choose(second, "fixed_eligible")

        changed = deepcopy(second)
        changed["view_digest"] = "0" * 64
        with pytest.raises(ValueError):
            SECOND.execute_pair(
                changed, choice, **F.arguments(data, changed, tmp_path / "bad-view"))

        changed = deepcopy(second)
        changed["inputs"]["parent"] = {"changed": True}
        with pytest.raises(ValueError, match="source|parent"):
            SECOND.execute_pair(
                changed, choice, **F.arguments(data, changed, tmp_path / "bad-parent"))

        kwargs = F.arguments(data, second, tmp_path / "bad-source")
        kwargs["case_input"] = tmp_path / "other.json"
        with pytest.raises(ValueError, match="source"):
            SECOND.execute_pair(second, choice, **kwargs)

        other_ledger = current_snapshot(data, ledger)
        other_ledger["inputs"]["ledger_path"] = str((tmp_path / "swapped.ledger").resolve())
        with pytest.raises(ValueError, match="ledger"):
            SECOND.execute_pair(
                other_ledger, choice,
                **F.arguments(data, other_ledger, tmp_path / "bad-ledger"))
    assert delegated == []


def test_scope_rejects_incomplete_original_receipt(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = F.physical_pair(tmp_path, monkeypatch)
    monkeypatch.setattr(SECOND, "_HOST", H)
    receipt_path = Path(H.read(data["result"])["receipts"]["reference"]["path"])
    receipt = H.read(receipt_path)
    receipt["execution"]["natural_exit"] = False
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="receipt SHA|종료"), M.recorded_pair_reuse_scope(
            original_result=data["result"], original_context=data["context"],
            original_choice=data["choice_path"]):
        pass
