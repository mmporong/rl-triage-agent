from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "bench") not in sys.path:
    sys.path.insert(0, str(ROOT / "bench"))

import anymal_termination_semantics_v1 as S

ISAACLAB = Path.home() / "IsaacLab"
HAS_SOURCES = (ISAACLAB / "source/isaaclab/isaaclab/envs/mdp/terminations.py").is_file()


def protocol() -> dict:
    return json.loads(S.PROTOCOL.read_text(encoding="utf-8"))


@pytest.mark.skipif(not HAS_SOURCES, reason="Isaac Lab source checkout is not available")
def test_real_anymal_timeout_configuration_is_derived_from_source():
    row = S.inspect_static(ISAACLAB)
    assert row["task"] == S.TASK_ID
    assert row["episode_length_s"] == 20.0
    assert row["physics_dt_s"] == 0.005
    assert row["decimation"] == 4
    assert row["control_dt_s"] == 0.02
    assert row["configured_max_episode_length"] == 1000
    assert row["fault_timeout_step"] == 50
    assert row["fault_timeout_s"] == 1.0
    assert all(value is True for key, value in row["structural"].items() if key.endswith("matches") or key.endswith("post_init") or key == "termination_uses_mdp_time_out")


def test_protocol_preregisters_cpu_only_semantics_and_behavior_boundary():
    row = protocol()
    assert row["status"] == "preregistered"
    assert row["prediction"]["semantic_gate"] == "applicable"
    assert row["prediction"]["baseline"] == [False, False, False, True]
    assert row["prediction"]["fault"] == [False, True, True, True]
    assert row["prediction"]["restored"] == row["prediction"]["baseline"]
    assert "측정하지 않는다" in row["limitations"]["behavior"]
    assert row["api_calls"] == 0 and row["gpu_calls"] == 0


@pytest.mark.skipif(not HAS_SOURCES, reason="Isaac Lab source checkout is not available")
def test_protocol_rejects_bound_source_sha_mutation():
    row = protocol()
    row["sources"]["termination"]["sha256_lf"] = "0" * 64
    with pytest.raises(ValueError, match="termination source differs"):
        S.validate_protocol(row, ISAACLAB)


@pytest.mark.skipif(not HAS_SOURCES, reason="Isaac Lab source checkout is not available")
def test_protocol_rejects_registered_value_mutation():
    row = protocol()
    row["configuration"]["fault_timeout_step"] = 51
    with pytest.raises(ValueError, match="configuration differs"):
        S.validate_protocol(row, ISAACLAB)


@pytest.mark.skipif(not HAS_SOURCES, reason="Isaac Lab source checkout is not available")
def test_protocol_rejects_correction_contract_mutation():
    row = protocol()
    row["correction"]["id"] = "unknown"
    with pytest.raises(ValueError, match="correction registration differs"):
        S.validate_protocol(row, ISAACLAB)


def test_write_new_refuses_overwrite(tmp_path: Path):
    target = tmp_path / "result.json"
    S.write_new(target, {"first": True})
    with pytest.raises(FileExistsError):
        S.write_new(target, {"second": True})


def test_smoke_binding_is_deterministic_and_rejects_semantic_mutation():
    # 실제 측정 대신 공개 계약으로 만든 합성 입력으로 검증기의 거부 동작을 검사한다.
    p = protocol()
    fingerprint = p["correction_source_contract"]["targets"][S.CORRECTION_ID]["clean_fingerprint"]
    report = {"schema": "anymal_termination_semantics_result_v1", "outcome": "applicable",
        "prediction": "applicable", "prediction_match": True, "configuration": p["configuration"],
        "structural": p["structural"],
        "probe": {"steps": p["probe_steps"], **{k: p["prediction"][k] for k in ("baseline", "fault", "restored")}},
        "source_sha256_lf": {k: v["sha256_lf"] for k, v in p["sources"].items()},
        "protocol_sha256_lf": S.sha_lf(S.PROTOCOL), "api_calls": 0, "gpu_calls": 0,
        "correction_readback": {"selected_correction": S.CORRECTION_ID, "registered_mode": "code",
            "changed_targets": [S.CORRECTION_ID], "restored_targets": [S.CORRECTION_ID],
            "clean_fingerprint": fingerprint, "fault_fingerprint": "f" * 64, "restored_fingerprint": fingerprint}}
    first = S.smoke_binding(report)
    assert first == S.smoke_binding(deepcopy(report))
    assert first["applicable"] is True and first["all_checks_passed"] is True
    changed = deepcopy(report)
    changed["probe"]["fault"][1] = False
    with pytest.raises(ValueError, match="outputs differ"):
        S.smoke_binding(changed)
    changed = deepcopy(report)
    changed["source_sha256_lf"]["termination"] = "0" * 64
    with pytest.raises(ValueError, match="source binding differs"):
        S.smoke_binding(changed)
    changed = deepcopy(report)
    changed["protocol_sha256_lf"] = "0" * 64
    with pytest.raises(ValueError, match="protocol binding differs"):
        S.smoke_binding(changed)


@pytest.mark.skipif(not HAS_SOURCES, reason="Isaac Lab source checkout is not available")
@pytest.mark.parametrize("field", ["prediction", "gate", "limitations", "structural"])
def test_protocol_rejects_decision_contract_mutation(field):
    row = protocol()
    row[field] = {}
    with pytest.raises(ValueError):
        S.validate_protocol(row, ISAACLAB)


def test_public_files_have_no_user_path_or_forbidden_word():
    windows_user_root = "C:" + "\\Users\\"
    source_file = S.__file__
    assert source_file is not None
    for path in (S.PROTOCOL, Path(source_file), Path(__file__)):
        text = path.read_text(encoding="utf-8")
        assert windows_user_root not in text
        assert "직" + "접" not in text
