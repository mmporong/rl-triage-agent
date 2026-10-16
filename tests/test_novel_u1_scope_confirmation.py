from __future__ import annotations

import json
from pathlib import Path
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "bench", ROOT / "evals"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import novel_u1_scope_confirmation as C
import novel_u1_scope_source as S


@pytest.fixture(autouse=True)
def restore_reused_module_globals():
    e = S.BASE.E
    source_state = (S.BASE.SOURCE_PRIVATE, S.BASE.SOURCE_RESULTS, S.BASE.SOURCE_PROTOCOL_DIR,
                    set(S.BASE.SOURCE_CODE), S.BASE.configure, S.BASE._scope_bindings,
                    e.PRIVATE, e.RESULTS, e.PROTOCOL_DIR, set(e.CODE), e.fixed, dict(e.VARIANTS))
    confirmation_state = (C.BASE.PROTOCOL, C.BASE.PRIVATE, C.BASE.FAULT_SEED, C.BASE.RESTART_SEED,
                          dict(C.BASE.ARMS), set(C.BASE.CODE), C.BASE.fixed_protocol,
                          C.BASE.validate_protocol, C.BASE.validate_authorization, C.BASE.validate_manifest)
    yield
    (S.BASE.SOURCE_PRIVATE, S.BASE.SOURCE_RESULTS, S.BASE.SOURCE_PROTOCOL_DIR,
     source_code, S.BASE.configure, S.BASE._scope_bindings,
     e.PRIVATE, e.RESULTS, e.PROTOCOL_DIR, e_code, e.fixed, variants) = source_state
    S.BASE.SOURCE_CODE = source_code
    e.CODE = e_code
    e.VARIANTS.clear(); e.VARIANTS.update(variants)
    (C.BASE.PROTOCOL, C.BASE.PRIVATE, C.BASE.FAULT_SEED, C.BASE.RESTART_SEED,
     arms, code, C.BASE.fixed_protocol, C.BASE.validate_protocol,
     C.BASE.validate_authorization, C.BASE.validate_manifest) = confirmation_state
    C.BASE.ARMS = arms
    C.BASE.CODE = code


def test_protocol_fixes_novel_seeds_and_predictions():
    protocol = C.validate_protocol(C.fixed_protocol())
    assert (protocol["fault_seed"], protocol["restart_seed"]) == (2062, 3062)
    assert protocol["status"] == "preregistered_unmeasured"
    assert protocol["predictions"]["u1_family_restart"] == "recovered"
    assert protocol["predictions"]["none_family_control"] == "healthy 유지"
    assert protocol["novel_seed_contract"] == {
        "source_seed": 2062,
        "restart_seed": 3062,
        "freshness_check": "required_once_before_first_gpu_step",
        "previous_result_observed": True,
        "write_once": True,
        "same_seed_narrow_control": False,
        "limitation": "같은 seed에서 단일 항 교정과 네 항 교정을 무작위 비교하지 않는다",
    }
    assert protocol["api_calls"] == 0
    assert protocol["scope_evidence_contract"]["model_loop"] == "inactive"


def test_source_contract_uses_new_paths_and_cases():
    S.configure()
    row = S.source_fixed("u1", "single")
    assert (row["seed"], row["restart_seed"]) == (2062, 3062)
    assert row["predictions"]["source_failure"].startswith("NONE은 걷고")
    assert S.BASE.E.VARIANTS["u1"]["seed"] == 2062
    assert S.BASE.E.ctx_of("u1").cases == {"NONE": "baseline_p0c_s2062", "FAULT": "u1_s2062"}
    assert "task11/u1/source" in S.BASE.E.PRIVATE.as_posix()


def test_manifest_is_restart_and_write_scoped(tmp_path, monkeypatch):
    C.configure()
    monkeypatch.setattr(C.BASE, "PRIVATE", tmp_path)
    monkeypatch.setattr(C, "PRIVATE", tmp_path)
    manifest = C.BASE.manifest_for("novel_u1", "u1_family_restart", Path("C:/IsaacLab"),
                                   tmp_path / "env.yaml", tmp_path / "agent.yaml", "a" * 40)
    assert manifest["seed"] == 3062
    assert manifest["fault"] == "U1"
    assert manifest["recovery_mode"] == "restart"
    assert manifest["max_iterations"] == 300
    assert manifest["run_name"] == "novel_u1_u1_family_restart"


def test_source_anchor_uses_ancestor_and_bound_blob_check(monkeypatch):
    seen = {}
    monkeypatch.setattr(C, "_VERIFY_COMMITTED", lambda root, code: "b" * 40)
    def frozen(experiment, head, bound):
        seen.update(experiment=experiment, head=head, bound=set(bound))
        return experiment
    monkeypatch.setattr(C.V2, "frozen_commit", frozen)
    assert C.verify_source_anchor("a" * 40) == "a" * 40
    assert seen["head"] == "b" * 40
    assert "evals/u1_scope_extension.py" in seen["bound"]
    assert "evals/u1_scope_source.py" in seen["bound"]
    assert "bench/demo_correction_u1_family.py" in seen["bound"]


def test_authorization_rejects_wrong_scope_seed(tmp_path):
    scope = tmp_path / "scope.json"
    approval = tmp_path / "approval.json"
    scope.write_text(json.dumps({
        "schema": "novel_u1_scope_confirmation_scope_v1", "seed": 2051, "fault": "U1",
        "fault_family": "root_body_velocity_reward_family",
        "result": {"verdict": "scope_insufficient", "uncovered_terms": [
            "track_ang_vel_z_exp", "lin_vel_z_l2", "ang_vel_xy_l2"]},
    }), encoding="utf-8")
    approval.write_text("{}", encoding="utf-8")
    try:
        C.validate_authorization(scope, approval, tmp_path)
    except ValueError as error:
        assert "novel-seed" in str(error)
    else:
        raise AssertionError("wrong source seed was accepted")


def test_public_protocol_has_no_machine_specific_path():
    raw = C.PROTOCOL.read_text(encoding="utf-8")
    assert "C:\\Users\\" not in raw
    assert "bench/private" not in raw
    source_protocol = S.SOURCE_PROTOCOL_DIR / "e2e_demo_u1_v1.json"
    source_raw = source_protocol.read_text(encoding="utf-8")
    assert "C:\\Users\\" not in source_raw
    assert json.loads(source_raw)["seed"] == 2062
