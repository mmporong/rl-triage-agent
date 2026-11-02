# pyright: reportMissingImports=false
from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("evals", "tests", "src")]

import five_policy_recovery_epoch2_v1 as SECOND
import local_pair_probe_host_v1 as H
import recovery_recorded_pair_reuse_v1 as M
import test_local_pair_probe_host_v1 as F


def save(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return path


def install_physical_fixture(monkeypatch: pytest.MonkeyPatch) -> None:
    @contextmanager
    def scope(path: Path, *_args: object, **_kwargs: object):
        previous = H.C.V.PROTOCOL, H.C.V.verify_protocol, H.C.V.BASE_FACT_IDS
        expected = H.C.V.file_digest(Path(path))
        H.C.V.PROTOCOL = Path(path)
        H.C.V.BASE_FACT_IDS = H.RV.BASE_FACT_IDS
        H.C.V.verify_protocol = lambda value: {} if value == expected else (_ for _ in ()).throw(
            ValueError("fixture protocol SHA"))
        try:
            yield {}
        finally:
            H.C.V.PROTOCOL, H.C.V.verify_protocol, H.C.V.BASE_FACT_IDS = previous

    def prepare(input_path: Path, probe: str, *, stage: Path, role: str,
                lab: Path, **_kwargs: object) -> dict:
        stage.mkdir(parents=True)
        body = {
            "schema": "fixture_obs_plan", "role": role, "probe": probe,
            "stage": str(stage), "checkpoint_sha256": "1" * 64,
            "collected_input": {"path": str(Path(input_path).resolve()), "sha256": "a" * 64},
            "lab": str(Path(lab).resolve()),
        }
        H.write_new(stage / "plan.json", body)
        return body

    def execute(plan_path: Path, authorization: Path, **_kwargs: object) -> dict:
        plan = H.read(plan_path)
        assert H.read(authorization)["request_spec"]["probe"] == plan["probe"]
        value = {
            "schema": "fixture_obs_receipt", "status": "complete", "role": plan["role"],
            "probe": plan["probe"], "measurement": F.measurement(plan["probe"], "normal"),
            "execution": {"wall_s": 1.0, "exit_code": 0, "natural_exit": True,
                          "cleanup_verified": True, "timed_out": False},
        }
        H.write_new(Path(plan["stage"]) / "probe_receipt.json", value)
        return value

    def verify(plan_path: Path, receipt_path: Path) -> dict:
        plan, receipt = H.read(plan_path), H.read(receipt_path)
        if receipt.get("role") != plan.get("role") or receipt.get("probe") != plan.get("probe"):
            raise ValueError("fixture receipt binding")
        return receipt

    monkeypatch.setattr(H.RV, "view_scope", scope)
    monkeypatch.setattr(H.OBS, "prepare_probe", prepare)
    monkeypatch.setattr(H.OBS, "execute_probe", execute)
    monkeypatch.setattr(H.OBS, "verify_receipt", verify)


def physical_pair(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    install_physical_fixture(monkeypatch)
    old = tmp_path / "old"
    case_input = save(tmp_path / "inputs/case.json", {"role": "case"})
    reference_input = save(tmp_path / "inputs/reference.json", {"role": "reference"})
    probe_protocol, view_protocol = F.protocols(tmp_path / "protocols")
    snapshot = F.snapshot(old)
    choice = H.C.choose(snapshot, "fixed_eligible")
    result = H.execute_pair(
        snapshot, choice, backend_packet=F.packet(True), case_input=case_input,
        reference_input=reference_input, probe_protocol=probe_protocol,
        view_protocol=view_protocol, anchor="a" * 40, lab=tmp_path,
        out=old / "pair", verify_context=lambda _value: {"approved": True})
    context = save(old / "context.json", {"snapshot": snapshot})
    choice_path = save(old / "choice.json", choice)
    return {
        "snapshot": snapshot, "choice": choice, "result": Path(result["result_ref"]["path"]),
        "context": context, "choice_path": choice_path, "case_input": case_input,
        "reference_input": reference_input, "probe_protocol": probe_protocol,
        "view_protocol": view_protocol, "backend_packet": F.packet(True), "lab": tmp_path,
        "old_root": old,
    }


def new_snapshot(data: dict, ledger: Path) -> dict:
    value = deepcopy(data["snapshot"])
    value["inputs"]["ledger_path"] = str(ledger.resolve())
    return value


def arguments(data: dict, snapshot: dict, out: Path) -> dict:
    return {
        "backend_packet": deepcopy(data["backend_packet"]),
        "case_input": data["case_input"], "reference_input": data["reference_input"],
        "probe_protocol": data["probe_protocol"], "view_protocol": data["view_protocol"],
        "anchor": "a" * 40, "lab": data["lab"], "out": out,
        "verify_context": lambda _value: {"approved": True},
    }


def inventory(root: Path) -> dict[str, str]:
    return {str(path.relative_to(root)): H.sha(path)
            for path in sorted(root.rglob("*")) if path.is_file()}


def scope(data: dict):
    return M.recorded_pair_reuse_scope(
        original_result=data["result"], original_context=data["context"],
        original_choice=data["choice_path"])


def test_replays_one_logical_pair_and_original_host_verifies(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = physical_pair(tmp_path, monkeypatch)
    monkeypatch.setattr(SECOND, "_HOST", H)
    before = inventory(data["old_root"])
    snapshot = new_snapshot(data, tmp_path / "new/public.ledger.jsonl")
    with scope(data):
        result = SECOND.execute_pair(
            snapshot, data["choice"], **arguments(data, snapshot, tmp_path / "new/pair"))
    assert result["physical_execution_reused"] is True
    assert result["prior_physical_roles"] == 2 and result["new_physical_roles"] == 0
    assert result["new_sdk_execution_count"] == 0
    assert result["parent_wall_s"] > 0
    assert inventory(data["old_root"]) == before
    rows = [json.loads(line) for line in Path(snapshot["inputs"]["ledger_path"]).read_text().splitlines()]
    assert [row["event"] for row in rows] == ["proposed", "approved", "consumed", "receipt"]
    verified = H.verify_pair(Path(result["result_ref"]["path"]), snapshot, data["choice"])
    assert verified["outcome"] == "normal" and verified["backend_ledger"] == result["backend_ledger"]
    receipt = H.read(tmp_path / "new/pair/reuse-receipt.json")
    assert receipt["logical_event_count"] == 4
    assert receipt["charged_cost_scope"] == "original_measurement_reused_once_not_new_gpu_time"


def test_rejects_second_use_and_same_target_mismatches_without_delegate(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = physical_pair(tmp_path, monkeypatch)
    monkeypatch.setattr(SECOND, "_HOST", H)
    snapshot = new_snapshot(data, tmp_path / "first.ledger.jsonl")
    with scope(data):
        SECOND.execute_pair(snapshot, data["choice"], **arguments(data, snapshot, tmp_path / "first"))
        second = new_snapshot(data, tmp_path / "second.ledger.jsonl")
        with pytest.raises(ValueError, match="한 번"):
            SECOND.execute_pair(second, data["choice"], **arguments(data, second, tmp_path / "second"))

    changes = []
    wrong = new_snapshot(data, tmp_path / "wrong-cp.ledger")
    wrong["inputs"]["packet"]["input_bindings"]["case"]["checkpoint_sha256"] = "9" * 64
    changes.append((wrong, data["choice"], {}, "snapshot"))
    wrong = new_snapshot(data, tmp_path / "wrong-parent.ledger")
    wrong["inputs"]["parent"] = {"checkpoint_sha256": "9" * 64}
    changes.append((wrong, data["choice"], {}, "snapshot"))
    wrong = new_snapshot(data, tmp_path / "wrong-view.ledger")
    wrong["view"]["remaining_candidates"] = ["termination"]
    changes.append((wrong, data["choice"], {}, "snapshot"))
    wrong_choice = deepcopy(data["choice"]); wrong_choice["fact_ids"] = ["changed"]
    changes.append((new_snapshot(data, tmp_path / "wrong-choice.ledger"), wrong_choice, {}, "choice"))
    changes.append((new_snapshot(data, tmp_path / "wrong-source.ledger"), data["choice"],
                    {"case_input": tmp_path / "other.json"}, "source"))
    changes.append((new_snapshot(data, tmp_path / "wrong-protocol.ledger"), data["choice"],
                    {"anchor": "b" * 40}, "protocol"))
    wrong_backend = deepcopy(data["backend_packet"])
    wrong_backend["input_bindings"]["case"]["checkpoint_sha256"] = "9" * 64
    changes.append((new_snapshot(data, tmp_path / "wrong-backend.ledger"), data["choice"],
                    {"backend_packet": wrong_backend}, "backend"))
    with scope(data):
        for index, (candidate, choice, overrides, message) in enumerate(changes):
            kwargs = arguments(data, candidate, tmp_path / f"bad-{index}")
            kwargs.update(overrides)
            with pytest.raises(ValueError, match=message):
                SECOND.execute_pair(candidate, choice, **kwargs)

        candidate = new_snapshot(data, tmp_path / "wrong-args.ledger")
        original_request_spec = H.C.request_spec

        def wrong_args(snapshot: dict, choice: dict) -> dict:
            value = original_request_spec(snapshot, choice)
            value["args"] = {"changed": True}
            return value

        monkeypatch.setattr(H.C, "request_spec", wrong_args)
        with pytest.raises(ValueError, match="args"):
            SECOND.execute_pair(
                candidate, data["choice"],
                **arguments(data, candidate, tmp_path / "bad-args"))


def test_other_pair_delegates_and_scope_restores(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = physical_pair(tmp_path, monkeypatch)
    monkeypatch.setattr(SECOND, "_HOST", H)
    calls: list[tuple] = []

    def delegate(*args: object, **kwargs: object) -> dict:
        calls.append((args, kwargs))
        return {"delegated": True}

    monkeypatch.setattr(SECOND, "execute_pair", delegate)
    other = new_snapshot(data, tmp_path / "other.ledger")
    other["binding"]["run_id"] = "other_run"
    with scope(data):
        assert SECOND.execute_pair(other, data["choice"], out=tmp_path / "other")["delegated"]
    assert SECOND.execute_pair is delegate and len(calls) == 1


def test_scope_rejects_incomplete_physical_receipt(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = physical_pair(tmp_path, monkeypatch)
    monkeypatch.setattr(SECOND, "_HOST", H)
    receipt_path = Path(H.read(data["result"])["receipts"]["case"]["path"])
    receipt = H.read(receipt_path)
    receipt["execution"]["cleanup_verified"] = False
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="receipt SHA|cleanup"), scope(data):
        pass
