"""R3 진단·R4 recovery 공개 replay의 분모, gate, 비용 분리 회귀."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("evals", "bench", "src")]

import build_replay_page as B


POLICIES = ("registry_next", "fixed_eligible", "logs_only", "logs_plus_frames",
            "optimizer_first_registered_rule")


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=t",
                    "-c", "user.email=t@example.com", "-c", "core.autocrlf=false", *args],
                   check=True, capture_output=True)


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _bundle(repo: Path, name: str, *, manifest_schema: str, summary: dict, cases: dict) -> None:
    folder = repo / name
    _write(folder / "summary.json", summary)
    _write(folder / "cases.json", cases)
    _write(folder / "provenance.json", {"schema": "fixture_provenance_v1", "status": "complete"})
    files = {}
    for file_name in ("summary.json", "cases.json", "provenance.json"):
        path = folder / file_name
        files[file_name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "size_bytes": path.stat().st_size}
    _write(folder / "manifest.json", {
        "schema": manifest_schema, "status": "complete", "files": files,
        "private_paths_in_public_files": False, "overwrite_allowed": False,
    })


def _observations_bundle(repo: Path, name: str, *, canonical_sha256: str) -> None:
    probes = (
        ("case-03", "fixed_eligible", ("P_episode", "P_noise", "P_torque"), "HOLD"),
        ("case-03", "logs_only", ("P_episode", "P_optimizer_updates", "P_value"), "HOLD"),
        ("case-07", "fixed_eligible", ("P_episode", "P_noise", "P_torque"), "HOLD"),
        ("case-07", "logs_only", ("P_episode", "P_value", "P_optimizer_updates"), "recovered"),
    )
    rows = []
    for case_id, policy, names, final in probes:
        rows.append({
            "case_id": case_id, "policy": policy, "first_outcome": "not_recovered",
            "final_outcome": final,
            "candidate_status": "validated_candidate" if final == "recovered" else "HOLD",
            "confirmed_family": "optimizer_update_count" if final == "recovered" else None,
            "second_correction": "restore_configured_ppo_epochs" if final == "recovered" else None,
            "model_calls": 3 if policy == "logs_only" else 0,
            "probe_owned_wall_s": 3.0,
            "steps": [{
                "step": index, "probe": probe,
                "outcome": "abnormal" if final == "recovered" and index == 3 else "normal",
                "selected_by": "model" if policy == "logs_only" else "fixed_order",
                "fact_ids": [f"fact-{index}"], "pair_result_sha256": str(index) * 64,
                "pair_parent_wall_s": 1.0, "owned_wall_s": 1.0,
            } for index, probe in enumerate(names, 1)],
        })
    count = lambda value, digest: {
        "records_sha256": digest * 64, "updates": [0, 1, 2],
        "expected_steps": [20, 20, 20], "observed_steps": value,
        "observed_minibatches": value,
    }
    summary = {
        "schema": "five_policy_recovery_observations_v1", "status": "complete",
        "canonical_manifest_sha256": canonical_sha256, "rows": rows,
        "optimizer_counts": {
            "first_training": count([4, 4, 4], "a"),
            "fresh_three_update_inspection": {
                "case": count([4, 4, 4], "b"),
                "reference": count([20, 20, 20], "c")},
            "second_training": count([20, 20, 20], "d"),
        },
        "probe_physical_roles": [{"role": index + 1} for index in range(24)],
        "probe_physical_roles_count": 24, "probe_owned_wall_sum_s": 12.0,
        "first_training_fixed_physical_roles": 8,
        "second_training_fixed_physical_roles": 2,
        "second_total_physical_roles": 26, "R4_total_physical_roles": 34,
        "second_model_requests": 6, "continuation_new_model_or_probe_calls": 0,
        "prior_failed_parent_wall_s": 10783.745328,
        "continuation_parent_wall_s": 1865.516,
    }
    folder = repo / name
    _write(folder / "summary.json", summary)
    digest = hashlib.sha256((folder / "summary.json").read_bytes()).hexdigest()
    _write(folder / "manifest.json", {
        "schema": "five_policy_recovery_observations_manifest_v1", "status": "complete",
        "files": {"summary.json": {"sha256": digest,
                                    "size_bytes": (folder / "summary.json").stat().st_size}},
    })


def _diagnosis_fixture() -> tuple[dict, dict]:
    metrics = {}
    rows = []
    for policy_index, policy in enumerate(POLICIES):
        metrics[policy] = {
            "planned_cases": 10, "independent_source_lineages": 10,
            "normal_cases": 2, "qualified_fault_cases": 8,
            "correct_any_family": 5 + (policy_index % 2),
            "compound_partial": 1, "compound_full": policy_index % 2,
            "wrong_confirmation": policy_index, "normal_model_shadow_proposal": policy_index % 2,
            "normal_model_observed": 2, "normal_model_unobserved": 0,
            "normal_host_gate": 2, "hold_rows": 3,
            "infra_unknown_rows": 1 if policy == "fixed_eligible" else 0,
            "paired_probes": 20 + policy_index, "model_calls": 8 + policy_index,
            "provider_tokens": {"prompt_tokens": 100, "completion_tokens": 20,
                                "total_tokens": 120 + policy_index},
            "provider_usage_unknown_calls": 0, "provider_unknown_events": 0,
            "provider_wall_s": 1.25 + policy_index,
            "parent_wall_s": (5289.112526100129 if policy == "fixed_eligible"
                              else 11.5 + policy_index),
        }
        for case_number in range(1, 11):
            arm = "NONE" if case_number <= 2 else "F5" if case_number >= 9 else "F1"
            rows.append({
                "case_id": f"case-{case_number:02d}", "policy": policy,
                "qualification": {"arm": arm, "diagnosis_eligible": arm != "NONE"},
                "scoring": {"repair_unsupported": arm == "F5"},
                "correct_any_family": arm == "F5" or case_number in (3, 4, 5),
                "wrong_confirmation": False, "compound_partial": False,
                "compound_full": False, "normal_shadow_proposal": False,
                "normal_shadow_observed": arm == "NONE", "normal_host_gate": arm == "NONE",
                "hold": arm in {"NONE", "F5"},
                "infra_unknown_events": (1 if policy == "fixed_eligible"
                                         and case_number == 4 else 0),
                "technical_missing": policy == "fixed_eligible" and case_number == 4,
                "hold_reason": "repair_unsupported" if arm == "F5" else
                "normal_shadow_only_host_gate" if arm == "NONE" else None,
            })
    summary = {
        "schema": "lineage_five_policy_summary_v1", "status": "complete",
        "scope": "actual_backend", "planned_cases": 10, "planned_policies": 5,
        "planned_rows": 50,
        "denominator": {"unit": "independent_source_lineage", "planned_cases": 10,
                        "valid_result_cases": 10, "normal_cases": 2, "independent_cases": 10,
                        "source_cp_independence_verified": True, "qualified_fault_cases": 8,
                        "repeated_frames_or_calls_increase_denominator": False},
        "policies": metrics,
        "primary": {"outcome": "same_accuracy_lower_probe_count",
                    "preregistered_contrast": ["logs_only", "fixed_eligible"]},
        "secondary": {"outcome": "no_observed_gain", "video_advantage_confirmed": False,
                      "preregistered_contrast": ["logs_plus_frames", "logs_only"]},
        "claims": {"model_version_gain": False, "probe_time_converted_to_gpu_savings": False},
    }
    cases = {
        "schema": "lineage_five_policy_public_cases_v1", "status": "complete",
        "scope": "actual_backend", "planned_cases": 10, "planned_policies": 5,
        "planned_rows": 50, "independent_unit": "source_lineage_not_row_or_call_or_frame",
        "rows": rows,
    }
    return summary, cases


def _recovery_fixture() -> tuple[dict, dict]:
    targets = (("case-03", "fixed_eligible", "F1", 2552),
               ("case-03", "logs_only", "F1", 2552),
               ("case-07", "fixed_eligible", "COMPOUND", 2556),
               ("case-07", "logs_only", "COMPOUND", 2556))
    rows = []
    for index, (case_id, policy, arm, seed) in enumerate(targets):
        second_required = index in (1, 3)
        rows.append({
            "case_id": case_id, "policy": policy, "source_arm": arm, "source_seed": seed,
            "original_source_checkpoint_sha256": ("a" if arm == "F1" else "b") * 64,
            "first_candidate_status": "validated_candidate", "first_outcome":
            "not_recovered" if second_required else "recovered",
            "first_training": {"training_execution_wall_s": 21.0 + index,
                               "fixed_execution_wall_s": 4.0 + index},
            "second_required": second_required,
            "second_outcome": "recovered" if second_required else None,
            "second_diagnosis_cost": ({"paired_probe_sdk_executions": 2,
                                       "model_api_calls": 1,
                                       "provider_tokens": {"total_tokens": 44 + index},
                                       "provider_wall_s": 0.5 + index}
                                      if second_required else None),
            "second_training": ({"training_execution_wall_s": 31.0 + index,
                                  "fixed_execution_wall_s": 5.0 + index}
                                 if second_required else None),
            "final_outcome": "recovered", "hold_reason": None,
        })
    summary = {
        "schema": "five_policy_recovery_public_summary_v1", "status": "complete",
        "scope": "actual_backend", "pipeline_denominator": 4,
        "independent_source_lineages": 2,
        "targets": [f"{case}:{policy}" for case, policy, _arm, _seed in targets],
        "final_outcomes": {"recovered": 4, "not_recovered": 0, "HOLD": 0},
        "first_parent_wall_s": 100.0, "second_parent_wall_s": 80.0,
        "second_required_pipelines": 2, "second_executed": True,
        "automatic_retry": False, "replacement_targets": 0, "epoch3_allowed": False,
    }
    return summary, {"schema": "five_policy_recovery_public_cases_v1",
                     "status": "complete", "rows": rows}


def _values(scene: dict) -> list[object]:
    return [item["value"] for block in scene["blocks"] for row in block["rows"]
            for item in row if isinstance(item, dict)]


def test_r3_waits_then_preserves_case_denominator_normal_gate_and_f5(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    summary, cases = _diagnosis_fixture()
    _bundle(repo, "r3", manifest_schema="lineage_five_policy_public_manifest_v1",
            summary=summary, cases=cases)
    slots = {"five_policy_diagnosis": "r3"}

    waiting = B.scene_five_policy_diagnosis(B.Sources(repo), slots)
    assert all(value is None for value in _values(waiting))

    paths = [path.relative_to(repo).as_posix() for path in (repo / "r3").iterdir()]
    _git(repo, "add", "--", *paths)
    _git(repo, "commit", "-q", "-m", "r3 fixture")
    sources = B.Sources(repo)
    scene = B.scene_five_policy_diagnosis(sources, slots)
    outcomes = {row[0]: [item["value"] for item in row[1:]] for row in scene["blocks"][0]["rows"]}
    boundaries = {row[0]: [item["value"] for item in row[1:]] for row in scene["blocks"][1]["rows"]}
    assert outcomes["등록 규칙"] == ["5/8", 0, 3, "1 / 0", 20, 8, 11.5]
    assert outcomes["고정 순서"] == [
        "6/8", 1, 3, "1 / 1", 21, 9,
        "5289.11s 기록 · 동일 범위 비교 불가",
    ]
    assert [outcomes[label][-1] for label in (
        "등록 규칙", "3.5 수치", "3.5 + 프레임", "optimizer 우선 등록 규칙"
    )] == [11.5, 13.5, 14.5, 15.5]
    assert boundaries["등록 규칙"] == ["0 / 2", 2, "2 진단 · 2 repair_unsupported", 120, 1.25]
    assert "독립 case 10개" in scene["lead"] and "no_observed_gain" in scene["lead"]
    assert scene["title"] == "다섯 정책 진단: 확정·HOLD와 실행 기록"
    assert scene["blocks"][0]["title"] == "정책별 진단과 실행 기록 범위"
    assert any("동일 범위 비용으로 비교하지 않는다" in note for note in scene["notes"])

    (repo / "r3/cases.json").write_text("{}", encoding="utf-8")
    rejected = B.scene_five_policy_diagnosis(B.Sources(repo), slots)
    assert all(value is None for value in _values(rejected))


def test_r4_keeps_four_pipelines_two_sources_and_separates_first_second_cost(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    summary, cases = _recovery_fixture()
    _bundle(repo, "r4", manifest_schema="five_policy_recovery_public_manifest_v1",
            summary=summary, cases=cases)
    slots = {"five_policy_recovery": "r4",
             "five_policy_recovery_observations": "r4-observations"}
    waiting = B.scene_five_policy_recovery(B.Sources(repo), slots)
    assert all(value is None for value in _values(waiting))

    paths = [path.relative_to(repo).as_posix() for path in (repo / "r4").iterdir()]
    _git(repo, "add", "--", *paths)
    _git(repo, "commit", "-q", "-m", "r4 fixture")

    scene = B.scene_five_policy_recovery(B.Sources(repo), slots)
    pipelines = scene["blocks"][0]["rows"]
    costs = scene["blocks"][1]["rows"]
    assert len(pipelines) == len(costs) == 4
    assert {row[1]["value"] for row in pipelines} == {"F1 · seed 2552", "COMPOUND · seed 2556"}
    second = next(row for row in pipelines if row[0] == "case-03 · logs_only")
    assert [item["value"] for item in second[2:]] == [
        None, "not_recovered", "필요", "recovered", "recovered"]
    second_cost = next(row for row in costs if row[0] == "case-03 · logs_only")
    assert [item["value"] for item in second_cost[1:]] == [
        "학습 22.0s · 평가 5.0s", "검사 2 · API 1", "tokens 45 · wall 1.5s",
        "학습 32.0s · 평가 6.0s"]
    assert "source 계보 2개" in scene["lead"] and "pipeline 4개" in scene["lead"]

    (repo / "r4/summary.json").write_text("{}", encoding="utf-8")
    rejected = B.scene_five_policy_recovery(B.Sources(repo), slots)
    assert all(value is None for value in _values(rejected))


def test_r4_observations_require_committed_matching_canonical_manifest(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    summary, cases = _recovery_fixture()
    _bundle(repo, "r4", manifest_schema="five_policy_recovery_public_manifest_v1",
            summary=summary, cases=cases)
    canonical_sha = hashlib.sha256((repo / "r4/manifest.json").read_bytes()).hexdigest()
    _observations_bundle(repo, "observations", canonical_sha256=canonical_sha)
    slots = {"five_policy_recovery": "r4",
             "five_policy_recovery_observations": "observations"}

    waiting_sources = B.Sources(repo)
    waiting = B.scene_five_policy_recovery(waiting_sources, slots)
    assert all(row[2]["value"] is None for row in waiting["blocks"][0]["rows"])
    assert waiting_sources.rows["observations"]["status"] == "waiting"

    paths = [path.relative_to(repo).as_posix() for folder in (repo / "r4", repo / "observations")
             for path in folder.iterdir()]
    _git(repo, "add", "--", *paths)
    _git(repo, "commit", "-q", "-m", "r4 observations fixture")
    sources = B.Sources(repo)
    scene = B.scene_five_policy_recovery(sources, slots)
    rows = {row[0]: row for row in scene["blocks"][0]["rows"]}
    assert rows["case-03 · fixed_eligible"][2]["value"] == (
        "episode 정상 → noise 정상 → torque 정상")
    assert rows["case-07 · logs_only"][2]["value"] == (
        "episode 정상 → value 정상 → updates 이상")
    assert rows["case-07 · logs_only"][2]["source"] == "observations"
    assert "case [4, 4, 4], reference [20, 20, 20], 교정 뒤 [20, 20, 20]" in scene["lead"]
    assert sources.rows["observations"]["status"] == "ready"


def test_r4_observations_reject_canonical_manifest_substitution(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    summary, cases = _recovery_fixture()
    _bundle(repo, "r4", manifest_schema="five_policy_recovery_public_manifest_v1",
            summary=summary, cases=cases)
    _observations_bundle(repo, "observations", canonical_sha256="f" * 64)
    paths = [path.relative_to(repo).as_posix() for folder in (repo / "r4", repo / "observations")
             for path in folder.iterdir()]
    _git(repo, "add", "--", *paths)
    _git(repo, "commit", "-q", "-m", "substituted observations fixture")
    sources = B.Sources(repo)
    scene = B.scene_five_policy_recovery(sources, {
        "five_policy_recovery": "r4",
        "five_policy_recovery_observations": "observations"})
    assert all(row[2]["value"] is None for row in scene["blocks"][0]["rows"])
    assert all(row[3]["value"] is not None for row in scene["blocks"][0]["rows"])
    assert sources.rows["observations"]["status"] == "mismatch"
    assert sources.rows["observations"]["reason"] == "canonical_manifest_sha256_mismatch"
