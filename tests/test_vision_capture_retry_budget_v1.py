import hashlib
import json
from decimal import Decimal
from types import SimpleNamespace

import pytest

from bench.vision_capture_retry_budget_v1 import apply_budget, validate_retry


def test_budget_counts_failed_attempt_and_restores_original_constants():
    module = SimpleNamespace(
        PRIOR=Decimal("20.087924"), FRESH=Decimal("4.912076"),
        TOTAL=Decimal(25), QUOTE=Decimal("1.74"),
    )
    retry = {"budget": {
        "prior_upper_bound_usd": "20.245996",
        "remaining_lower_bound_usd": "4.754004",
        "program_cap_usd": "25",
        "compute_quote_usd_per_h": "1.74",
    }}
    with pytest.raises(RuntimeError, match="stage failure"), apply_budget(module, retry):
        assert module.PRIOR == Decimal("20.245996")
        assert module.FRESH == Decimal("4.754004")
        raise RuntimeError("stage failure")
    assert module.PRIOR == Decimal("20.087924")
    assert module.FRESH == Decimal("4.912076")


def test_budget_rejects_inconsistent_or_nonfinite_allowance():
    module = SimpleNamespace(TOTAL=Decimal(25), QUOTE=Decimal("1.74"))
    retry = {"budget": {
        "prior_upper_bound_usd": "20.245996",
        "remaining_lower_bound_usd": "4.754005",
        "program_cap_usd": "25",
        "compute_quote_usd_per_h": "1.74",
    }}
    with pytest.raises(ValueError), apply_budget(module, retry):
        pass
    retry["budget"]["remaining_lower_bound_usd"] = "NaN"
    with pytest.raises(ValueError), apply_budget(module, retry):
        pass


def test_retry_requires_original_failure_and_correct_prior_cost(tmp_path):
    def save(name, value):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = (json.dumps(value) + "\n").encode()
        path.write_bytes(raw)
        return {"path": name, "sha256_lf": hashlib.sha256(raw).hexdigest()}

    protocol = save("bench/protocol.json", {
        "protocol_id": "capture", "resources": {"owned_instance_id": "id", "owned_instance_name": "name"},
        "limits": {"max_policies": 2, "training_actions": 0, "model_api_requests": 0},
        "checkpoints": [{"id": "one"}],
    })
    authorization = save("bench/auth.json", {
        "execution_protocol": protocol, "resource_id": "id", "resource_name": "name",
        "checkpoint_catalog": [{"id": "one"}],
    })
    prior = save("evals/prior.json", {
        "protocol_id": "capture", "protocol": protocol, "status": "HOLD",
        "failure": {"stage": "instance_readiness"}, "cleanup": {"vm_status": "STOPPED"},
        "actual_recordings": 0, "new_training_actions": 0, "model_api_requests": 0,
        "cost": {"cumulative_estimate_including_upper_bound_usd": "20.245996",
                 "remaining_program_budget_lower_bound_usd": "4.754004", "program_cap_usd": "25"},
    })
    guard = save("bench/vision_capture_retry_budget_v1.py", "guard")
    retry = {
        "schema": "vision_capture_infrastructure_retry_v1", "registered_before_execution": True,
        "registered_at_utc": "2026-01-01T00:00:00Z", "retry_reason": "infrastructure_readiness_timeout",
        "max_start_requests": 1, "new_training_actions": 0, "model_api_requests": 0,
        "max_full_recordings": 2, "resource": {"instance_id": "id", "instance_name": "name"},
        "record_protocol": protocol, "resource_authorization": authorization, "prior_result": prior,
        "budget_guard": guard, "budget": {"prior_upper_bound_usd": "20.245996",
            "remaining_lower_bound_usd": "4.754004", "program_cap_usd": "25",
            "compute_quote_usd_per_h": "1.74", "reserved_for_two_recordings_usd": "0.58"},
    }
    retry_path = tmp_path / "retry.json"
    retry_path.write_text(json.dumps(retry), encoding="utf-8")
    validate_retry(retry_path, repo=tmp_path, committed=False)
    retry["budget"]["prior_upper_bound_usd"] = "20.087924"
    retry_path.write_text(json.dumps(retry), encoding="utf-8")
    with pytest.raises(ValueError, match="선행 비용"):
        validate_retry(retry_path, repo=tmp_path, committed=False)
