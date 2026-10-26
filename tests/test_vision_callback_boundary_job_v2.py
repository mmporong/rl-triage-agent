from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest  # pyright: ignore[reportMissingImports]

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
import vision_callback_boundary_job_v2 as J  # pyright: ignore[reportMissingImports]


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


@pytest.mark.parametrize("mismatch", [None, "protocol_sha256", "run_id"])
def test_budget_contract_accepts_actual_aa_stop_shape_and_fourteen_rows(
    tmp_path: Path, mismatch: str | None
) -> None:
    relative = Path("bench/protocols/vision_callback_boundary_20261026_v1.json")
    previous_path = tmp_path / relative
    previous_path.parent.mkdir(parents=True)
    previous_path.write_bytes((ROOT / relative).read_bytes())
    previous = json.loads(previous_path.read_text(encoding="utf-8"))
    previous["_protocol_path"] = previous_path
    previous["_protocol_sha256"] = J._sha(previous_path, lf=True)
    stop_path = (
        tmp_path
        / ".triage-runtime/vision-brev/lifecycle/callback-boundary-20261026aa-stop.json"
    )
    # 실제 AA receipt의 공개 가능한 필드만 재구성한다. 로컬 비공개 원문은 필요 없다.
    stop = {
        "status": "STOPPED",
        "execution_identity": {
            "protocol_sha256": previous["source_protocol"]["sha256_lf"],
            "resource_proof_sha256": "cebc4fd1d74e70c71b5b3937ce62b188c194c2e819af7f4c0750dbeb5c39b39c",
            "lease_id": "v04-a1d028159ea70b5d8605",
            "run_id": previous["protocol_id"],
        },
        "cost_estimate": {"estimated_compute_cost_usd": "0.197218"},
    }
    if mismatch is not None:
        stop["execution_identity"][mismatch] = "wrong-source-or-run"
    stop_path.parent.mkdir(parents=True)
    stop_path.write_text(json.dumps(stop), encoding="utf-8")
    stop_ref = {
        "path": stop_path.relative_to(tmp_path).as_posix(),
        "sha256": J._sha(stop_path),
    }
    resources = deepcopy(previous["resources"]); resources.update({
        "prior_compute_estimate_usd": "15.808282",
        "fresh_lifecycle_budget_usd": "4.191718",
        "prior_stop_receipts": [*resources["prior_stop_receipts"], stop_ref],
    })
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
        with pytest.raises(ValueError, match="14 STOP"):
            J._budget(protocol, previous, tmp_path)
        return
    J._budget(protocol, previous, tmp_path)
    assert len(protocol["resources"]["prior_stop_receipts"]) == 14
    assert J.PROTOCOL_SCHEMA == "vision_callback_boundary_protocol_v2"
    assert J.JOB_SCHEMA == "vision_callback_boundary_job_v2"


def test_protocol_requires_fixed_source_checkpoint_and_non_decision_flags() -> None:
    assert J.PROTOCOL_KEYS >= {"source_protocol", "base_record_protocol", "resources", "predecessor", "checkpoint_logical_id"}
    assert J.REQUIRED_SOURCE_FILES >= {
        "bench/vision_callback_boundary_diagnostic_v2.py",
        "bench/vision_callback_boundary_job_v2.py",
        "bench/run_vision_callback_boundary_v2.py",
    }
