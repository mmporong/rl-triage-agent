from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
import vision_checkpoint_smoke_job_v2 as J  # pyright: ignore[reportMissingImports]


def test_checkpoint_smoke_limits_keep_one_step_and_no_recording() -> None:
    assert J.MODE == "prepared_checkpoint_smoke_after_capture_only"
    assert J.LIMITS == {
        "max_wall_s": 600,
        "logical_frames": 4,
        "policy_environment_steps": 1,
        "callbacks": 6,
        "training_actions": 0,
        "model_api_requests": 0,
        "full_recordings": 0,
    }


def test_protocol_contract_requires_checkpoint_specific_preregistration() -> None:
    assert J.PROTOCOL_KEYS == {
        "schema",
        "mode",
        "protocol_id",
        "registered_before_measurement",
        "source_commit",
        "source_sha256_lf",
        "base_record_protocol",
        "limits",
        "preregistration",
    }
