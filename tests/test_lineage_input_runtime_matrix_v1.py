# pyright: reportMissingImports=false
from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("bench", "evals", "tests", "src")]

import lineage_input_runtime_matrix_v1 as M
import test_lineage_five_policy_driver_v1 as FIX


def save(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def fixture(tmp_path: Path, *, eligible: bool = True) -> tuple[dict, dict[str, object]]:
    source = FIX.source(tmp_path / "source", kind="F1")
    source["providers"] = None
    if not eligible:
        source["qualification"] = {"arm": "F1", "label": "baseline_invalid",
                                   "diagnosis_eligible": False}
    source_path = save(tmp_path / "source.json", source)
    index_path = Path(source["index"]["path"])
    protocol_path = save(tmp_path / "fixture-protocol.json", {"fixture": True, "api_cap": 72})
    transport_state: dict[str, object] = {"cached": None, "completed": 0}

    def initialize(root: Path, **_kwargs) -> dict:
        root.mkdir(parents=True)
        save(root / "manifest.json", {"schema": "fixture-transport", "automatic_retries": 0})
        return {"schema": "fixture-transport"}

    def complete(root: Path, request: dict, *, send) -> dict:
        del root
        if transport_state["cached"] is None:
            send({}, Path("fixture-call"))
            transport_state["cached"] = {
                "schema": M.TRANSPORT.VERIFIED_SCHEMA, "status": "completed",
                "outcome": "semantic_valid", "live": False, "content": '{"action":"hold"}',
                "usage": {}, "wall_s": 0.0,
                "call_ref": {"path": "fixture-call", "sha256": "f" * 64},
                "request": deepcopy(request)}
            transport_state["completed"] = 1
        cached = transport_state["cached"]
        assert isinstance(cached, dict)
        return deepcopy(cached)

    def status(_root: Path) -> dict:
        return {"completed": transport_state["completed"], "automatic_retries": 0,
                "pending_no_retry": 0, "reservations_charged": transport_state["completed"]}

    validators = {
        "source": lambda value: M.DRIVER.checked_source(value, fixture=True),
        "protocol": lambda path, _anchor: M.read(path),
        "transport_initialize": initialize, "transport_complete": complete,
        "transport_status": status,
        "transport_verify": lambda call: {
            "schema": M.TRANSPORT.VERIFIED_SCHEMA, "status": "completed",
            "outcome": "semantic_valid", "live": True, "content": '{"action":"hold"}',
            "call_ref": {"path": str(call), "sha256": M.sha(call / "receipt.json")}},
    }
    return {"source": source_path, "index": index_path, "protocol": protocol_path,
            "anchor": "a" * 40}, validators


def by_case(result: dict) -> dict[str, dict]:
    return {row["case"]: row for row in result["rows"]}


def test_fixture_matrix_consumes_real_driver_guard_path_and_separates_evidence_classes(
        tmp_path: Path) -> None:
    inputs, validators = fixture(tmp_path)
    before = {name: M.sha(path) for name, path in inputs.items() if isinstance(path, Path)}
    result = M.run_matrix(
        inputs["source"], inputs["index"], inputs["protocol"], inputs["anchor"],
        tmp_path / "matrix", allow_fixture=True, validators=validators)
    rows = by_case(result)
    assert result["scope"] == "development_fixture" and result["row_count"] == 10
    assert result["actual_api_calls"] == result["gpu_operations"] == 0
    assert rows["actual_model_sdk_receipts"]["status"] == "not_run"
    assert rows["blur_frame_quarantine"]["result"]["evidence"]["image_count"] == 0
    assert rows["blur_frame_quarantine"]["result"]["decision"] == "HOLD"
    assert rows["visual_numeric_conflict"]["result"]["reason"] == (
        "verified_visual_numeric_conflict")
    assert rows["visual_numeric_unknown"]["result"]["reason"] == (
        "visual_or_numeric_unknown")
    assert rows["gpu_busy_pre_dispatch"]["evidence_class"] == "controlled_injection"
    assert "driver_scope=cpu_fixture" in rows["gpu_busy_pre_dispatch"]["notes"]
    assert "dispatch_calls=0" in rows["gpu_busy_pre_dispatch"]["notes"]
    assert (tmp_path / "matrix/driver-busy/step01/busy-gate.json").is_file()
    assert (tmp_path / "matrix/driver-busy/result.json").is_file()
    assert not (tmp_path / "matrix/driver-busy/probes.ledger.jsonl").exists()
    after = {name: M.sha(path) for name, path in inputs.items() if isinstance(path, Path)}
    assert before == after and result["source_files_mutated"] is False


def test_input_timeout_pending_and_completed_reuse_have_no_retry_or_duplicate_send(
        tmp_path: Path) -> None:
    inputs, validators = fixture(tmp_path)
    result = M.run_matrix(
        inputs["source"], inputs["index"], inputs["protocol"], inputs["anchor"],
        tmp_path / "matrix", allow_fixture=True, validators=validators)
    rows = by_case(result)
    assert rows["registered_input_missing"]["result"]["reason"] == (
        "registered_input_unavailable")
    assert rows["registered_input_permission"]["result"]["reason"] == (
        "registered_input_unavailable")
    assert rows["typed_transport_timeout"]["result"]["reason"] == "typed_transport_error"
    assert rows["pending_no_automatic_resend"]["result"]["reason"] == (
        "transport_pending_no_resend")
    assert "callback_calls=1" in rows["pending_no_automatic_resend"]["notes"]
    reuse = rows["completed_receipt_reuse"]
    assert reuse["evidence_class"] == "fixture_transport"
    assert reuse["result"]["send_calls"] == 1
    assert reuse["result"]["actual_backend_admissible"] is False
    assert reuse["result"]["transport_status"]["automatic_retries"] == 0


def test_actual_call_input_is_read_only_verified_not_an_outage_claim(tmp_path: Path) -> None:
    inputs, validators = fixture(tmp_path)
    call = tmp_path / "actual-call"
    save(call / "receipt.json", {"actual": True})
    before = M.sha(call / "receipt.json")
    result = M.run_matrix(
        inputs["source"], inputs["index"], inputs["protocol"], inputs["anchor"],
        tmp_path / "matrix", actual_call_dirs=(call,),
        allow_fixture=True, validators=validators)
    row = by_case(result)["actual_model_sdk_receipts"]
    assert row["evidence_class"] == "actual_model_sdk_raw" and row["status"] == "verified"
    assert row["result"]["calls"][0]["live"] is True
    assert result["claims"]["actual_api_outage"] is False
    assert M.sha(call / "receipt.json") == before


def test_missing_driver_precondition_is_explicit_hold_not_fake_busy_measurement(
        tmp_path: Path) -> None:
    inputs, validators = fixture(tmp_path, eligible=False)
    result = M.run_matrix(
        inputs["source"], inputs["index"], inputs["protocol"], inputs["anchor"],
        tmp_path / "matrix", allow_fixture=True, validators=validators)
    row = by_case(result)["gpu_busy_pre_dispatch"]
    assert row["status"] == "HOLD"
    assert row["result"]["stage"] == "busy_not_reached"
    assert row["result"]["reason"] == "source_baseline_invalid"
    assert "dispatch_calls=0" in row["notes"]


def test_output_must_be_fresh_and_fixture_validators_are_explicit(tmp_path: Path) -> None:
    inputs, validators = fixture(tmp_path)
    out = tmp_path / "matrix"
    out.mkdir()
    with pytest.raises(ValueError, match="fresh"):
        M.run_matrix(inputs["source"], inputs["index"], inputs["protocol"],
                     inputs["anchor"], out, allow_fixture=True, validators=validators)
    out.rmdir()
    with pytest.raises(ValueError, match="validator"):
        M.run_matrix(inputs["source"], inputs["index"], inputs["protocol"],
                     inputs["anchor"], out, allow_fixture=True)


def test_cli_has_no_fixture_flag_and_passes_actual_call_dirs(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}
    monkeypatch.setattr(M, "run_matrix", lambda *args, **kwargs: captured.update(
        args=args, kwargs=kwargs) or {})
    paths = [tmp_path / name for name in ("source", "index", "protocol", "out", "call")]
    assert M.main([
        "--source", str(paths[0]), "--index", str(paths[1]),
        "--protocol", str(paths[2]), "--anchor", "a" * 40,
        "--out", str(paths[3]), "--actual-call-dir", str(paths[4])]) == 0
    assert captured["args"] == (paths[0], paths[1], paths[2], "a" * 40, paths[3])
    assert captured["kwargs"] == {"actual_call_dirs": (paths[4],)}
