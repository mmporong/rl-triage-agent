from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest  # pyright: ignore[reportMissingImports]

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
import vision_semantic_settle_job_v1 as J  # pyright: ignore[reportMissingImports]


def test_contract_is_one_bounded_non_approving_candidate() -> None:
    assert J.MODE == "checkpoint_semantic_settle_diagnostic_only"
    assert J.VARIANTS == ("manual_semantic_settle",)
    assert J.LIMITS == {
        "variants": 1,
        "runs_per_variant": 1,
        "max_wall_s_per_variant": 600,
        "max_batch_compute_usd": "1.000000",
        "training_actions": 0,
        "model_api_requests": 0,
        "capture_approval": False,
    }
    assert J.REQUIRED_SOURCE_FILES >= {
        "bench/vision_semantic_settle_checkpoint_v1.py",
        "bench/vision_semantic_settle_job_v1.py",
        "bench/run_vision_semantic_settle_v1.py",
    }


@pytest.mark.parametrize("mismatch", [None, "protocol_sha256", "run_id"])
def test_budget_contract_appends_actual_ab_stop_as_fifteenth_receipt(
    tmp_path: Path, mismatch: str | None
) -> None:
    relative = Path("bench/protocols/vision_callback_boundary_20261026_v2.json")
    previous_path = tmp_path / relative
    previous_path.parent.mkdir(parents=True)
    previous_path.write_bytes((ROOT / relative).read_bytes())
    previous = json.loads(previous_path.read_text(encoding="utf-8"))
    previous["_protocol_path"] = previous_path
    previous["_protocol_sha256"] = J._sha(previous_path, lf=True)
    stop_path = (
        tmp_path
        / ".triage-runtime/vision-brev/lifecycle/callback-boundary-20261026ab-stop.json"
    )
    stop = {
        "status": "STOPPED",
        "execution_identity": {
            "protocol_sha256": previous["source_protocol"]["sha256_lf"],
            "resource_proof_sha256": "2" * 64,
            "lease_id": "v04-semantic-settle-test",
            "run_id": previous["protocol_id"],
        },
        "cost_estimate": {"estimated_compute_cost_usd": "0.215613"},
    }
    if mismatch is not None:
        stop["execution_identity"][mismatch] = "wrong-source-or-run"
    stop_path.parent.mkdir(parents=True)
    stop_path.write_text(json.dumps(stop), encoding="utf-8")
    stop_ref = {
        "path": stop_path.relative_to(tmp_path).as_posix(),
        "sha256": J._sha(stop_path),
    }
    resources = deepcopy(previous["resources"])
    resources.update(
        {
            "prior_compute_estimate_usd": "16.023895",
            "fresh_lifecycle_budget_usd": "3.976105",
            "prior_stop_receipts": [*resources["prior_stop_receipts"], stop_ref],
        }
    )
    protocol = {
        "resources": resources,
        "predecessor": {
            "diagnostic": {
                "path": previous_path.relative_to(tmp_path).as_posix(),
                "sha256_lf": previous["_protocol_sha256"],
            },
            "appended_stop": stop_ref,
        },
    }
    if mismatch is not None:
        with pytest.raises(ValueError, match="15 STOP"):
            J._budget(protocol, previous, tmp_path)
        return
    J._budget(protocol, previous, tmp_path)
    assert len(resources["prior_stop_receipts"]) == 15
