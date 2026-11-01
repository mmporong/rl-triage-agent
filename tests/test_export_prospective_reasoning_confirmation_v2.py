# pyright: reportMissingImports=false
from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals import export_prospective_reasoning_confirmation_v2 as E


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                    encoding="utf-8")


def _failure() -> dict:
    return {
        "schema": "prospective_confirmation_execution_failure_public_v1",
        "status": "incomplete_execution_failure", "assessment": "indeterminate",
        "planned_rows": 24, "completed_rows": 0, "model_requests_consumed": 1,
        "provider_responses_preserved": 1, "paired_probes_completed": 0,
        "sdk_probe_executions": 0,
        "failure": {"category": "harness_callback_contract_mismatch"},
        "first_proposal": {"action": "run_probe", "case_id": "c02",
                           "confirmation": None, "policy": "legacy_none",
                           "target": "P_reward"},
        "retry": {"new_http_requests_for_preserved_first_slot": 0,
                  "policy_or_threshold_change": False, "preserve_prior_response": True,
                  "requires_new_registered_execution": True},
    }


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    monkeypatch.setattr(E, "ROOT", tmp_path)
    paths = {name: tmp_path / filename for name, filename in {
        "v3": "bench/protocols/v3.json", "v2": "bench/protocols/v2.json",
        "v1": "bench/protocols/v1.json", "design": "bench/protocols/design.json",
        "bridge": "bench/protocols/bridge.json", "source": "inputs/source.json",
        "corpus": "inputs/corpus.json", "bridge_root": "run/bridge",
        "execution": "run/execution-v3", "failure": "evals/results/failure/summary.json",
        "audit": "inputs/rule-audit.json",
        "out": "public/result"}.items()}
    for key in ("v2", "v1", "bridge", "source", "corpus"):
        _write(paths[key], {key: True})
    _write(paths["design"], {"rule_policy": ["read_evidence", "P_optimizer_updates",
                                                   "if_count_normal_then_P_reward"]})
    _write(paths["failure"], _failure())
    protocol = {
        "schema": E.V3.PROTOCOL_SCHEMA, "status": "registered_only_after_commit",
        "base_v2_protocol": {"path": paths["v2"].relative_to(tmp_path).as_posix()},
        "base_v1_protocol": {"path": paths["v1"].relative_to(tmp_path).as_posix()},
        "design_protocol": {"path": paths["design"].relative_to(tmp_path).as_posix()},
        "base_v2_anchor": E.V3.V2_ANCHOR, "base_v1_anchor": E.V3.V2.STAGE_B_ANCHOR,
        "continuation": {"classification": "mechanical_infrastructure_continuation",
                         "exposure": "post_response_host_contract_fix_same_source_development_confirmation"},
        "failed_attempt": {"api_requests_consumed": 1, "sdk_role_executions": 0,
                           "assessment": "indeterminate",
                           "artifacts": {"request": "a" * 64, "response": "b" * 64,
                                         "claim": "c" * 64, "slot_claim": "c" * 64,
                                         "session": "d" * 64}},
        "inherited": {"api_requests": 96},
        "claims": {"llm_superiority": False, "default_mode_adoption": False,
                   "new_fault_family_generalization": False, "repair": False,
                   "recovery": False, "pure_unexposed_performance": False},
        "source_sha256_lf": {"evals/evidence_guided_selector_v1.py": "e" * 64},
    }
    _write(paths["v3"], protocol)
    raw = {"schema": E.V3.R.RESULT_SCHEMA, "status": "complete", "planned_rows": 24,
           "rows": [{"row": index} for index in range(24)], "model_requests": 7,
           "paired_probes": 3, "sdk_role_executions_upper_bound": 6,
           "assessment": {"outcome": "mixed"}}
    _write(paths["execution"] / "result.json", raw)
    rule_result = paths["execution"] / "c02/strong_registered_rule/result.json"
    _write(rule_result, {"rule": "actual"})
    _write(paths["audit"], {
        "schema": "registered_rule_description_execution_audit_v1",
        "scope": "posthoc_development_audit_no_policy_mutation",
        "design_sha256": E._sha(paths["design"]),
        "registered_text_rule_policy": ["read_evidence", "P_optimizer_updates",
                                        "if_count_normal_then_P_reward"],
        "actual_rule_implementation": "evals/evidence_guided_selector_v1.py::select_strong_rule",
        "implementation_lf_sha256": "e" * 64, "case_id": "c02",
        "actual_actions": [
            {"action": "read_evidence", "target": "evidence:optimizer-counts", "fact_ids": []},
            {"action": "run_probe", "target": "P_value", "fact_ids": []}],
        "result_sha256": E._sha(rule_result),
        "must_report_rule_code_vs_text_discrepancy": True,
        "rule_arm_omitted_or_resampled": False,
        "model_only_mode_comparison_policy_changed": False, "new_API_GPU": 0})
    _write(paths["execution"] / "v3-result.json", {
        "schema": E.V3.RESULT_SCHEMA, "status": "complete",
        "classification": protocol["continuation"]["classification"],
        "exposure": protocol["continuation"]["exposure"],
        "prior_attempt": protocol["failed_attempt"],
        "base_result_sha256": E._sha(paths["execution"] / "result.json"),
        "model_requests_including_replay": 7, "replayed_provider_requests": 1,
        "new_provider_requests": 6, "paired_probes": 3,
        "sdk_role_executions_upper_bound": 6, "assessment": raw["assessment"],
        "claims": protocol["claims"]})
    return paths


def _verify(protocol_path: Path, _anchor: str, _v2: Path, _v1: Path, _design: Path,
            _bridge: Path, _source: Path, _corpus: Path, _bridge_root: Path) -> dict:
    return json.loads(protocol_path.read_text(encoding="utf-8"))


def _base_export(result: Path, _protocol: Path, _design: Path, _bridge: Path,
                 _source: Path, _corpus: Path, _bridge_root: Path, *, anchor: str,
                 out: Path) -> dict:
    assert result.name == "result.json" and anchor == E.V3.V2.STAGE_B_ANCHOR
    summary = {
        "schema": E.BASE.SUMMARY_SCHEMA, "status": "complete",
        "assessment": {"outcome": "mixed"}, "planned_rows": 24, "complete_rows": 24,
        "uncertainty": {"independent_units": 2, "p_value_reported": False,
                        "small_sample": True, "independent_unit": "source_seed",
                        "compound_partial_separate": True}}
    _write(out / "summary.json", summary)
    _write(out / "artifact-manifest.json", {"artifact_count": 2})
    return summary


def _export(paths: dict[str, Path], **kwargs):
    return E.export(paths["execution"], paths["v3"], paths["design"], paths["bridge"],
                    paths["source"], paths["corpus"], paths["bridge_root"],
                    paths["failure"], paths["audit"], anchor="f" * 40, out=paths["out"],
                    verify_protocol_fn=kwargs.get("verify", _verify),
                    base_export_fn=kwargs.get("base", _base_export))


def test_release_wraps_verified_base_and_preserves_continuation_scope(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    summary = _export(paths)
    assert summary["assessment"] == {"outcome": "mixed"}
    assert summary["provider_request_accounting"] == {
        "cap_including_replay": 96, "model_requests_including_replay": 7,
        "replayed_preserved_requests": 1, "new_provider_requests": 6,
        "first_response_reselected": False}
    assert summary["claims"]["llm_superiority"] is False
    assert summary["claims"]["pure_unexposed_performance"] is False
    assert summary["uncertainty"]["independent_units"] == 2
    assert summary["policy_audit"]["status"] == "registered_rule_prose_code_mismatch"
    assert summary["policy_audit"]["rule_arm_unqualified_for_stated_sequence"] is True
    assert summary["comparison_scope"]["original_four_arm_complete"] is False
    assert summary["comparison_scope"]["heldout_confirmation"] is False
    assert (paths["out"] / "verified-base/summary.json").is_file()
    assert (paths["out"] / "release-manifest.json").is_file()
    encoded = "".join(path.read_text(encoding="utf-8")
                      for path in paths["out"].rglob("*.json"))
    assert str(tmp_path) not in encoded and ("bench" + "/private") not in encoded


@pytest.mark.parametrize("mutation,match", [
    (lambda paths: (paths["execution"] / "v3-result.json").unlink(), "모두 필요"),
    (lambda paths: _mutate(paths["execution"] / "v3-result.json",
                           "replayed_provider_requests", 0), "결합 불일치"),
    (lambda paths: _mutate(paths["execution"] / "v3-result.json",
                           "base_result_sha256", "0" * 64), "결합 불일치"),
    (lambda paths: _mutate(paths["failure"], "assessment", "complete"), "실패 summary"),
    (lambda paths: _mutate(paths["audit"], "implementation_lf_sha256", "0" * 64),
     "규칙 설명/실제 코드 audit"),
])
def test_release_rejects_partial_or_tampered_inputs_before_output(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation, match: str) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    mutation(paths)
    with pytest.raises(ValueError, match=match):
        _export(paths)
    assert not paths["out"].exists()


def _mutate(path: Path, key: str, value: object) -> None:
    current = json.loads(path.read_text(encoding="utf-8"))
    current[key] = value
    _write(path, current)


def test_release_delegates_source_closure_validation_before_output(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _fixture(tmp_path, monkeypatch)

    def reject(*_args):
        raise ValueError("source closure mismatch")

    with pytest.raises(ValueError, match="source closure"):
        _export(paths, verify=reject)
    assert not paths["out"].exists()


def test_release_refuses_output_overwrite(tmp_path: Path,
                                          monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    _export(paths)
    with pytest.raises(ValueError, match="fresh"):
        _export(paths)
