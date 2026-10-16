from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import re

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "evals/export_r1_repair_v2.py"
SPEC = importlib.util.spec_from_file_location("export_r1_repair_v2", MODULE_PATH)
assert SPEC and SPEC.loader
EXPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORT)
HAS_PRIVATE = all(path.is_file() for path in (*EXPORT.INPUTS.values(), *EXPORT.VALIDATION_INPUTS.values()))


@pytest.fixture(scope="module")
def public_payloads() -> dict[str, dict]:
    payloads = {relative: json.loads((EXPORT.OUTPUT / relative).read_text(encoding="utf-8"))
                for relative in EXPORT.PUBLIC_FILES}
    EXPORT.verify_public_payloads(payloads)
    return payloads


def test_public_summary_has_eight_allowlisted_files(public_payloads):
    assert set(public_payloads) == EXPORT.PUBLIC_FILES
    summary = public_payloads["summary.json"]
    assert summary["status"] == "complete"
    assert summary["outcome"] == "recovered" and summary["demo_success"] is True
    assert len(summary["completion_conditions"]) == 13
    assert all(summary["completion_conditions"].values())
    assert summary["source_failure"] == {
        "none_walks": True, "r1_label": "unhealthy", "failed_checks": ["err_xy_ratio", "err_yaw_ratio"],
    }
    assert summary["recovery"]["corrected_label"] == "healthy"
    assert summary["interpretation"].startswith("새 seed 단일 R1")
    assert "기존 23회" in summary["design_coupling"]
    assert summary["inspection_provenance"]["mode"] == "fresh_three_update_inspection"
    assert summary["inspection_provenance"]["checkpoint_loaded"] is False


def test_fixed_evaluations_preserve_all_conditions_and_expected_metrics(public_payloads):
    expected = {
        "none": ("healthy", 0.14635629776820577),
        "r1": ("unhealthy", 1.1691575140582355),
        "corrected": ("healthy", 0.14817000429733868),
    }
    for arm, (label, lin_rmse) in expected.items():
        report = public_payloads[f"evaluations/{arm}.json"]
        assert report["label"] == label
        assert report["metrics"]["lin_vel_rmse_mps"] == lin_rmse
        assert report["condition_count"] == len(report["conditions"]) == 26
        assert all(len(row["target_velocity"]) == 3 and row["num_envs"] == 40 for row in report["conditions"])


def test_training_readback_counts_and_clean_termination_are_preserved(public_payloads):
    arms = public_payloads["training_evidence.json"]["arms"]
    assert arms["NONE"]["first_three_updates"]["optimizer_steps"] == [20, 20, 20]
    assert arms["R1"]["first_three_updates"]["optimizer_steps"] == [4, 4, 4]
    assert arms["CORRECTED"]["first_three_updates"]["optimizer_steps"] == [20, 20, 20]
    assert arms["CORRECTED"]["fresh_restart"] is True
    assert arms["CORRECTED"]["readback"]["after_fault"]["learning_epochs"] == 1
    assert arms["CORRECTED"]["readback"]["restoration"] == {"registry_value": 5, "runtime_value": 5}
    assert arms["CORRECTED"]["readback"]["after"]["learning_epochs"] == 5
    for arm in arms.values():
        assert arm["checkpoint"]["first_iteration"] == 0 and arm["checkpoint"]["final_iteration"] == 299
        assert arm["checkpoint"]["verification_device"] == "cpu"
        assert all(row["exit_code"] == 0 and row["natural_exit"] and row["cleanup_verified"]
                   for row in arm["execution"].values())


def test_model_choice_registered_probe_sequence_and_correction_are_auditable(public_payloads):
    model = public_payloads["model_evidence.json"]
    assert model["diagnosis"]["decision"]["ranking"][0] == "reward"
    assert model["diagnosis"]["decision"]["ranking"].index("optimizer") == 3
    assert model["diagnosis"]["decision"]["probe"] == "P_reward"
    assert model["diagnosis"]["review_status"] == model["correction"]["review_status"] == "valid"
    assert model["diagnosis"]["feedback"] is model["correction"]["feedback"] is None
    assert model["correction"]["decision"]["proposal"]["correction_id"] == "restore_configured_ppo_epochs"
    for row in (model["diagnosis"], model["correction"]):
        call = row["call"]
        assert call["total_tokens"] == call["input_tokens"] + call["output_tokens"]
        assert call["wall_s"] > 0 and row["execution_wall_s"] > 0

    evidence = public_payloads["probe_and_correction.json"]
    probes = evidence["probes"]
    assert [row["outcome"] for row in probes] == ["normal"] * 5 + ["abnormal"]
    assert [row["selection_source"] for row in probes] == [
        "model_selected", "registered_next", "registered_next", "registered_next", "registered_next",
        "registered_new_probe",
    ]
    assert probes[-1]["measurement"]["case"]["optimizer_steps"] == [4, 4, 4]
    assert probes[-1]["measurement"]["control"]["optimizer_steps"] == [20, 20, 20]
    assert probes[-1]["measurement"]["checkpoint_loaded"] is False
    assert evidence["correction"]["approval_events"] == ["proposed", "approved", "consumed", "receipt"]
    assert evidence["correction"]["verdict"] == "recovered"


def test_public_payload_rejects_sensitive_paths_nonfinite_and_execution_fields(public_payloads):
    cases = [
        ("summary.json", "leak", r"C:\\Users\\someone\\secret.json"),
        ("summary.json", "leak", "bench/private/run.json"),
        ("summary.json", "api_key", "redacted"),
        ("summary.json", "command", ["python", "worker.py"]),
        ("summary.json", "traceback", "raw stack"),
        ("summary.json", "leak", "../private.json"),
        ("summary.json", "leak", float("nan")),
    ]
    for filename, key, value in cases:
        bad = deepcopy(public_payloads)
        bad[filename][key] = value
        with pytest.raises(ValueError):
            EXPORT.verify_public_payloads(bad)


def test_exclusive_export_and_verification_use_only_public_payloads(tmp_path, public_payloads):
    output = tmp_path / "public"
    audit = tmp_path / "private-audit"
    EXPORT.write_new(public_payloads, output, audit)
    assert {path.relative_to(output).as_posix() for path in output.rglob("*.json")} == EXPORT.PUBLIC_FILES
    text = "\n".join(path.read_text(encoding="utf-8") for path in output.rglob("*.json"))
    assert re.search(r"[A-Za-z]:[\\/]|/(?:home|Users|mnt)/|bench[\\/]private|codex_support", text,
                     re.IGNORECASE) is None
    receipt = json.loads((audit / "receipt.json").read_text(encoding="utf-8"))
    assert set(receipt["public_artifact_sha256"]) == EXPORT.PUBLIC_FILES
    with pytest.raises(FileExistsError):
        EXPORT.write_new(public_payloads, output, audit)


def test_checked_in_public_export_has_stable_file_hashes(public_payloads):
    hashes = {relative: EXPORT.sha256(EXPORT.OUTPUT / relative) for relative in public_payloads}
    assert set(hashes) == EXPORT.PUBLIC_FILES
    assert all(re.fullmatch(r"[0-9a-f]{64}", value) for value in hashes.values())


@pytest.mark.skipif(not HAS_PRIVATE, reason="private source artifacts are unavailable in a public clone")
def test_private_sources_recompute_checked_in_public_export():
    hashes = EXPORT.verify_export()
    assert set(hashes) == EXPORT.PUBLIC_FILES
    assert all(re.fullmatch(r"[0-9a-f]{64}", value) for value in hashes.values())


@pytest.mark.skipif(not HAS_PRIVATE, reason="private source artifacts are unavailable in a public clone")
def test_model_raw_content_tamper_is_rejected_even_with_updated_file_hash(tmp_path, monkeypatch):
    record = EXPORT.read(EXPORT.INPUTS["model_diagnosis"])
    record["calls"][0]["content"] = "{}"
    changed = tmp_path / "diagnosis.json"
    changed.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setitem(EXPORT.INPUTS, "model_diagnosis", changed)
    monkeypatch.setitem(EXPORT.EXPECTED_SHA256, "model_diagnosis", EXPORT.sha256(changed))
    with pytest.raises(ValueError):
        EXPORT.verify_and_build()


@pytest.mark.skipif(not HAS_PRIVATE, reason="private source artifacts are unavailable in a public clone")
def test_probe_ledger_cross_request_and_registry_digest_tamper_are_rejected(tmp_path, monkeypatch):
    original = EXPORT.read_jsonl(EXPORT.INPUTS["probe_ledger"])
    variants = []
    crossed = deepcopy(original)
    crossed[1]["request_id"] = crossed[4]["request_id"]
    variants.append(crossed)
    registry_changed = deepcopy(original)
    registry_changed[0]["registry_digest"] = "0" * 64
    variants.append(registry_changed)
    for index, events in enumerate(variants):
        changed = tmp_path / f"probe_{index}.jsonl"
        changed.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in events), encoding="utf-8")
        monkeypatch.setitem(EXPORT.INPUTS, "probe_ledger", changed)
        monkeypatch.setitem(EXPORT.EXPECTED_SHA256, "probe_ledger", EXPORT.sha256(changed))
        with pytest.raises(ValueError):
            EXPORT.verify_and_build()


@pytest.mark.skipif(not HAS_PRIVATE, reason="private source artifacts are unavailable in a public clone")
def test_correction_ledger_semantic_tamper_is_rejected_with_updated_file_hash(tmp_path, monkeypatch):
    events = EXPORT.read_jsonl(EXPORT.INPUTS["correction_ledger"])
    events[0]["confirmed_mechanism"] = "reward"
    changed = tmp_path / "correction.jsonl"
    changed.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in events), encoding="utf-8")
    monkeypatch.setitem(EXPORT.INPUTS, "correction_ledger", changed)
    monkeypatch.setitem(EXPORT.EXPECTED_SHA256, "correction_ledger", EXPORT.sha256(changed))
    with pytest.raises(ValueError):
        EXPORT.verify_and_build()
