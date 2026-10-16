"""R1 v2의 경로 사전 점검과 판 교체가 모델·회복 계약을 바꾸지 않는지 검증한다."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "tests")]
import r1_repair_demo as V1
import r1_repair_demo_v2 as D
import test_r1_repair_demo as FIXTURE


class Backend(FIXTURE.Backend):
    def preflight(self, isaaclab):
        self.calls.append(("preflight", str(isaaclab)))
        return {"processes_started": 0, "gpu_queries": 0}


@pytest.fixture
def setup(tmp_path, monkeypatch):
    protocol = {**D.fixed_protocol(), "source_sha256_lf": {}, "runtime_source_sha256_lf": {}}
    monkeypatch.setattr(D, "PRIVATE", tmp_path / "runs")
    original_sha = D.E.sha
    monkeypatch.setattr(D.E, "sha", lambda p, lf=False: "b" * 64 if p == D.PROTOCOL else original_sha(p, lf))
    monkeypatch.setattr(D, "verify", lambda *args, **kwargs: (protocol, "c" * 40))
    return lambda *args, **kwargs: {"status": "fresh", "seeds": [2071, 3071]}


def run(setup, backend, caller=None):
    return D.execute("r1_repair_s2071_20261016", Path("unused"), "operator", True,
                     backend=backend, caller=caller or FIXTURE.model_caller(), seed_guard=setup)


def test_v2_protocol_preserves_decision_measurement_and_recovery_contract():
    original = deepcopy(V1.fixed_protocol())
    new = D.fixed_protocol()
    for field in ("model", "loop_registry", "loop_rule", "correction_registry", "inspection",
                  "prediction", "outcome_wording", "failure_policy", "gpu", "approval"):
        assert new[field] == original[field]
    old_recovery, new_recovery = original["recovery_rule"], deepcopy(new["recovery_rule"])
    new_recovery["reference"] = old_recovery["reference"]
    assert new_recovery == old_recovery
    assert new["success_conditions"] == original["success_conditions"] + ["launch_paths_checked"]
    assert D.SEED == 2071 and D.RESTART_SEED == 3071
    assert V1.SEED == 2070 and V1.RESTART_SEED == 3070
    assert D.C is V1.C and D.MODEL_CALL is V1.MODEL_CALL
    assert V1.CODE <= D.CODE and D.PROTOCOL != V1.PROTOCOL and D.PRIVATE != V1.PRIVATE


def test_complete_v2_keeps_v1_reference_and_consumes_correction_once(setup):
    backend = Backend()
    result = run(setup, backend)
    assert result["outcome"] == "recovered" and result["demo_success"]
    assert backend.calls[0][0] == "preflight"
    assert result["conditions"]["launch_paths_checked"] is True
    assert result["api_requests"] == 2
    folder = D.PRIVATE / result["tag"]
    assert D.read(folder / "artifact_none.json")["seed"] == 2071
    assert D.read(folder / "artifact_r1.json")["seed"] == 2071
    assert D.read(folder / "artifact_corrected.json")["seed"] == 3071
    assert result["recovery"]["reference_metrics"] == D.read(folder / "source_evaluation/NONE.json")["metrics"]
    ledger = D.CorrectionLedger(folder / "correction.ledger.jsonl", D.C.correction_registry())
    assert [event["event"] for event in ledger.events()] == ["proposed", "approved", "consumed", "receipt"]
    with pytest.raises(FileExistsError):
        run(setup, backend)


def test_missing_launch_path_stops_before_seed_or_attempt_creation(setup):
    class Missing(Backend):
        def preflight(self, isaaclab):
            raise FileNotFoundError("missing python")
    backend = Missing()
    with pytest.raises(FileNotFoundError, match="missing python"):
        run(setup, backend)
    assert not backend.calls and not D.PRIVATE.exists()


def test_existing_seed_is_not_exempted_for_infrastructure_retry(setup):
    backend = Backend()
    def reused(*args, **kwargs):
        raise ValueError("seed already recorded")
    with pytest.raises(ValueError, match="already recorded"):
        run(reused, backend)
    assert backend.calls == [("preflight", "unused")]
    assert not D.PRIVATE.exists()


def test_old_tag_is_not_accepted_by_v2(setup):
    with pytest.raises(ValueError):
        D.execute("r1_repair_s2070_20261016", Path("unused"), "operator", True, backend=Backend())
    assert not D.PRIVATE.exists()


@pytest.mark.parametrize("kind", ["diagnosis_invalid", "correction_hold", "count_unknown"])
def test_v2_keeps_fail_closed_model_and_observation_gates(setup, kind):
    backend = Backend(count_status="unknown" if kind == "count_unknown" else "abnormal")
    caller = FIXTURE.model_caller(invalid=kind == "diagnosis_invalid", correction_hold=kind == "correction_hold")
    result = run(setup, backend, caller)
    assert not result["demo_success"] and ("train", "CORRECTED") not in backend.calls
    assert result["outcome"] == {
        "diagnosis_invalid": "model_invalid", "correction_hold": "correction_not_proposed",
        "count_unknown": "loop_not_confirmed",
    }[kind]
