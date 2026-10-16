import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))
import build_additional_checks_replay as R


def inputs():
    protocol = json.loads((ROOT / R.PROTOCOL).read_text(encoding="utf-8"))
    return {name: json.loads((ROOT / name).read_text(encoding="utf-8")) for name in protocol["inputs_sha256"]}


@pytest.mark.parametrize("field,value", [("scenes", []), ("validation", {}), ("scope", "unregistered"),
                                         ("model_calls", 1), ("gpu_calls", 1), ("inputs_sha256", {})])
def test_protocol_contract_changes_are_rejected(field, value):
    protocol = json.loads((ROOT / R.PROTOCOL).read_text(encoding="utf-8"))
    protocol[field] = value
    with pytest.raises(ValueError): R.validate_protocol(protocol)


def test_public_results_cross_validate_and_keep_control_sets_separate():
    rows = inputs()
    assert R.validate_optimizer(rows) == {"r1_steps": 4, "none_steps": 20, "specificity": 21}
    u1 = R.validate_u1(rows)
    assert u1["before"]["lin_vel_rmse_mps"] > u1["after"]["lin_vel_rmse_mps"]


@pytest.mark.parametrize("mutation", ["step", "count", "missing", "scope", "seed", "readback"])
def test_cross_validation_rejects_inconsistent_published_numbers(mutation):
    rows = inputs()
    validator = R.validate_optimizer
    if mutation == "step": rows[R.OPT + "runs/r1_s2042.json"]["updates"][1]["observed_optimizer_steps"] = 20
    elif mutation == "count": rows[R.OPT + "summary.json"]["classification_counts"]["normal"] = 21
    elif mutation == "missing": rows[R.OPT + "runs/h06_s2028.json"]["updates"].pop()
    else:
        validator = R.validate_u1
        if mutation == "scope": rows[R.U1 + "scope_evidence.json"]["measured_probe_terms"].append("lin_vel_z_l2")
        elif mutation == "seed": rows[R.U1 + "evaluations/u1_family_restart.json"]["training"]["seed"] = 2051
        else: rows[R.U1 + "correction_readback.json"]["arms"]["u1_family_restart"]["restored_functions"].pop()
    with pytest.raises(ValueError): validator(rows)


def test_changed_input_blocks_scene_numbers(tmp_path, monkeypatch):
    protocol = json.loads((ROOT / R.PROTOCOL).read_text(encoding="utf-8"))
    for relative in [R.PROTOCOL, *protocol["inputs_sha256"]]:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / relative).read_bytes())
    monkeypatch.setattr(R.B.Sources, "_committed_state", lambda *args: ("ready", None))
    target = tmp_path / (R.U1 + "summary.json")
    target.write_text('{}', encoding="utf-8")
    data = R.build(tmp_path)
    cells = [cell for block in data["scenes"][0]["blocks"] for row in block["rows"] for cell in row if isinstance(cell, dict)]
    assert cells and all(cell["value"] is None for cell in cells)
    assert data["sources"][R.U1 + "summary.json"]["status"] == "mismatch"
    assert data["scenes"][1]["blocks"][0]["rows"][1][2]["value"] == 4


def test_generated_snapshot_matches_committed_source_inputs():
    data = R.build()
    assert len(data["scenes"]) == 2 and len(data["sources"]) == 30
    assert {row["status"] for row in data["sources"].values()} == {"ready"}
    assert R.B.render(data) == (ROOT / R.OUTPUT).read_text(encoding="utf-8")
