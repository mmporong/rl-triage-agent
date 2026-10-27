from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest  # pyright: ignore[reportMissingImports]

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
import vision_annotator_pull_job_v1 as J  # pyright: ignore[reportMissingImports]


def test_contract_is_one_non_approving_writer_free_candidate() -> None:
    assert J.MODE == "checkpoint_annotator_pull_diagnostic_only"
    assert J.VARIANTS == ("annotator_pull",)
    assert J.LIMITS["capture_approval"] is False
    assert J.LIMITS["training_actions"] == 0
    assert J.LIMITS["model_api_requests"] == 0
    assert J.REQUIRED_SOURCE_FILES >= {
        "bench/vision_annotator_pull_collector_v1.py",
        "bench/vision_annotator_pull_validator_v1.py",
        "bench/vision_annotator_pull_worker_v1.py",
        "bench/vision_annotator_pull_checkpoint_v1.py",
        "bench/vision_annotator_pull_job_v1.py",
        "bench/run_vision_annotator_pull_v1.py",
        "tests/test_vision_annotator_pull_collector_v1.py",
        "tests/test_vision_annotator_pull_checkpoint_v1.py",
        "tests/test_vision_annotator_pull_job_v1.py",
        "tests/test_run_vision_annotator_pull_v1.py",
    }
    assert len(J.REQUIRED_SOURCE_FILES) == len(J.PREVIOUS.REQUIRED_SOURCE_FILES) + 10


@pytest.mark.parametrize("bad_stop", [None, 0, 1])
def test_budget_appends_retry2_and_genuine_ag_as_stops_19_and_20(
    tmp_path: Path, bad_stop: int | None
) -> None:
    relative = Path("bench/protocols/vision_frame_trigger_20261027_v3.json")
    previous_path = tmp_path / relative
    previous_path.parent.mkdir(parents=True)
    previous_path.write_bytes((ROOT / relative).read_bytes())
    previous = json.loads(previous_path.read_text(encoding="utf-8"))
    previous["_protocol_path"] = previous_path
    previous["_protocol_sha256"] = J._sha(previous_path, lf=True)
    appended = []
    for index, expected in enumerate(J.EXPECTED_STOPS):
        source = ROOT / expected["path"]
        target = tmp_path / expected["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        row = dict(expected)
        if bad_stop == index:
            value = json.loads(target.read_text(encoding="utf-8"))
            value["execution_identity"]["run_id"] = "wrong-run"
            target.write_text(json.dumps(value), encoding="utf-8")
            row["sha256"] = J._sha(target)
        appended.append(row)
    resources = deepcopy(previous["resources"])
    resources.update(
        {
            "prior_compute_estimate_usd": "18.211504",
            "fresh_lifecycle_budget_usd": "1.788496",
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
    if bad_stop is not None:
        with pytest.raises(ValueError, match="STOP"):
            J._budget(protocol, previous, tmp_path)
        return
    J._budget(protocol, previous, tmp_path)
    assert len(resources["prior_stop_receipts"]) == 20
