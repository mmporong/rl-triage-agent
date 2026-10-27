from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import vision_annotator_pull_job_v2 as J  # pyright: ignore[reportMissingImports]


def test_contract_adds_native_counter_closure_without_changing_mode() -> None:
    assert J.MODE == J.PREVIOUS.MODE
    assert J.VARIANTS == ("annotator_pull",)
    assert J.FIXED_FLAGS["app_update_counter_source"] == (
        "omni.kit.app.IApp.get_update_number"
    )
    assert J.FIXED_FLAGS["app_update_counter_fail_closed"] is True
    assert J.FIXED_FLAGS["aaba_camera_sequence"] == ["A", "A", "B", "A"]
    assert J.FIXED_FLAGS["orchestrator_step_calls_per_arm"] == 1
    assert J.FIXED_FLAGS["isaac_frame_numbers_strictly_increasing"] is True
    assert J.FIXED_FLAGS["physics_state_time_unchanged"] is True
    assert J.EXPECTED_PRIOR + J.EXPECTED_FRESH == J.EXPECTED_TOTAL
    assert J.REQUIRED_SOURCE_FILES == J.PREVIOUS.REQUIRED_SOURCE_FILES | {
        "bench/vision_annotator_pull_worker_v2.py",
        "bench/vision_annotator_pull_checkpoint_v2.py",
        "bench/vision_annotator_pull_job_v2.py",
        "bench/run_vision_annotator_pull_v2.py",
        "tests/test_vision_annotator_pull_worker_v2.py",
        "tests/test_vision_annotator_pull_checkpoint_v2.py",
        "tests/test_vision_annotator_pull_job_v2.py",
        "tests/test_run_vision_annotator_pull_v2.py",
    }


def test_registered_final_stop_closes_actual_budget() -> None:
    previous_path = ROOT / "bench/protocols/vision_annotator_pull_20261027_v1.json"
    previous = json.loads(previous_path.read_text(encoding="utf-8"))
    previous["_protocol_path"] = previous_path
    previous["_protocol_sha256"] = J._sha(previous_path, lf=True)
    appended = [dict(row) for row in J.EXPECTED_STOPS]
    resources = deepcopy(previous["resources"])
    resources.update(
        {
            "prior_compute_estimate_usd": "18.330565",
            "fresh_lifecycle_budget_usd": "1.669435",
            "prior_stop_receipts": [
                *resources["prior_stop_receipts"],
                *appended,
            ],
        }
    )
    protocol = {
        "resources": resources,
        "predecessor": {
            "diagnostic": {
                "path": previous_path.relative_to(ROOT).as_posix(),
                "sha256_lf": previous["_protocol_sha256"],
            },
            "appended_stops": appended,
        },
    }

    J._budget(protocol, previous, ROOT)


@pytest.mark.parametrize("mutation", [None, "cost", "run_id", "sha"])
def test_budget_appends_only_ah_stop_and_preserves_total(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str | None
) -> None:
    previous_source = ROOT / "bench/protocols/vision_annotator_pull_20261027_v1.json"
    previous_path = tmp_path / "bench/protocols/vision_annotator_pull_20261027_v1.json"
    previous_path.parent.mkdir(parents=True)
    previous_path.write_bytes(previous_source.read_bytes())
    previous = json.loads(previous_path.read_text(encoding="utf-8"))
    previous["_protocol_path"] = previous_path
    previous["_protocol_sha256"] = J._sha(previous_path, lf=True)

    stop_path = tmp_path / ".triage-runtime/vision-brev/lifecycle/ah-stop.json"
    stop_path.parent.mkdir(parents=True)
    stop = {
        "status": "STOPPED",
        "execution_identity": {
            "run_id": previous["protocol_id"],
            "protocol_sha256": previous["source_protocol"]["sha256_lf"],
        },
        "cost_estimate": {"estimated_compute_cost_usd": "0.119061"},
    }
    if mutation == "cost":
        stop["cost_estimate"]["estimated_compute_cost_usd"] = "0.119062"
    if mutation == "run_id":
        stop["execution_identity"]["run_id"] = "wrong"
    stop_path.write_text(json.dumps(stop), encoding="utf-8")
    stop_ref = {
        "path": stop_path.relative_to(tmp_path).as_posix(),
        "sha256": J._sha(stop_path),
    }
    monkeypatch.setattr(J, "EXPECTED_STOPS", (stop_ref,))
    appended = [dict(stop_ref)]
    if mutation == "sha":
        appended[0]["sha256"] = "0" * 64
    resources = deepcopy(previous["resources"])
    resources.update(
        {
            "prior_compute_estimate_usd": "18.330565",
            "fresh_lifecycle_budget_usd": "1.669435",
            "prior_stop_receipts": [
                *resources["prior_stop_receipts"],
                *appended,
            ],
        }
    )
    protocol = {
        "resources": resources,
        "predecessor": {
            "diagnostic": {
                "path": previous_path.relative_to(tmp_path).as_posix(),
                "sha256_lf": previous["_protocol_sha256"],
            },
            "appended_stops": appended,
        },
    }
    if mutation is not None:
        with pytest.raises(ValueError, match="STOP"):
            J._budget(protocol, previous, tmp_path)
        return

    J._budget(protocol, previous, tmp_path)
    assert len(resources["prior_stop_receipts"]) == (
        len(previous["resources"]["prior_stop_receipts"]) + 1
    )
