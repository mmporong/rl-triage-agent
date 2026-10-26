from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest  # pyright: ignore[reportMissingImports]

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
import vision_callback_boundary_job_v1 as J  # pyright: ignore[reportMissingImports]


def test_contract_is_two_bounded_non_approving_variants() -> None:
    assert J.MODE == "checkpoint_callback_boundary_diagnostic_only"
    assert J.VARIANTS == ("custom_event", "manual_schedule")
    assert J.LIMITS == {
        "variants": 2,
        "runs_per_variant": 1,
        "max_wall_s_per_variant": 600,
        "max_batch_compute_usd": "1.000000",
        "training_actions": 0,
        "model_api_requests": 0,
        "capture_approval": False,
    }


def test_budget_appends_actual_z_stop_without_relabeling_source_budget() -> None:
    base = json.loads((ROOT / "bench/protocols/vision_prepared_record_20261026_v4.json").read_text(encoding="utf-8"))
    stop_ref = {
        "path": ".triage-runtime/vision-brev/lifecycle/prepared-policy-record-20261026z-stop.json",
        "sha256": "a5b8e08a90599cb3568401c06d870d63c311e837b6c4380e9b5069652e189f4c",
    }
    resources = deepcopy(base["resources"])
    resources.update({
        "prior_compute_estimate_usd": "15.611064",
        "fresh_lifecycle_budget_usd": "4.388936",
        "prior_stop_receipts": [*resources["prior_stop_receipts"], stop_ref],
    })
    protocol = {
        "resources": resources,
        "predecessor": {"record": {"path": "bench/protocols/vision_prepared_record_20261026_v4.json", "sha256_lf": "001bf823c2776d58b7fb7e47adf34e3f6071d0ade6fb3441ae4012be805d831e"}, "appended_stop": stop_ref},
        "base_record_protocol": {"path": "bench/protocols/vision_prepared_record_20261026_v4.json", "sha256_lf": "001bf823c2776d58b7fb7e47adf34e3f6071d0ade6fb3441ae4012be805d831e"},
    }
    J._budget(protocol, base, ROOT)
    protocol["resources"]["fresh_lifecycle_budget_usd"] = "7.866279"
    with pytest.raises(ValueError, match="resource|누적"):
        J._budget(protocol, base, ROOT)


def test_protocol_requires_fixed_source_checkpoint_and_non_decision_flags() -> None:
    assert J.PROTOCOL_KEYS >= {"source_protocol", "base_record_protocol", "resources", "predecessor", "checkpoint_logical_id"}
    assert J.REQUIRED_SOURCE_FILES >= {
        "bench/vision_callback_boundary_diagnostic_v1.py",
        "bench/vision_callback_boundary_job_v1.py",
        "bench/run_vision_callback_boundary_v1.py",
    }
