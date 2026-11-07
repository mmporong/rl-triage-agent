from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from evals import ppo_patch_contract_v1 as CONTRACT
from evals import ppo_patch_gate_quality_v1 as QUALITY


def _all_strings(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [item for child in value.values() for item in _all_strings(child)]
    if isinstance(value, list):
        return [item for child in value for item in _all_strings(child)]
    return []


def test_fixed_corpus_has_zero_classification_errors() -> None:
    summary, rows = QUALITY.run_quality_evaluation()

    assert summary["quality_status"] == "passed"
    assert summary["counts"] == {
        "total": 14,
        "expected_accept": 4,
        "expected_reject": 10,
        "admitted": 4,
        "rejected": 10,
        "false_accept": 0,
        "false_reject": 0,
        "ast_rejected_sources_executed": 0,
        "security_sources_executed": 0,
        "contract_ineligible_sources_executed": 0,
    }
    assert summary["unnecessary_patch_blocks"] == 1
    assert {row["case_id"] for row in rows if row["admitted"]} == {
        "correct-dict-copy",
        "correct-copy-method",
        "correct-module-docstring",
        "correct-function-docstring",
    }


def test_correct_candidates_preserve_all_configs_and_missing_epoch() -> None:
    correct = [case for case in QUALITY.DEV_CASES if case.expected_admission]
    assert len(correct) == 4

    for case in correct:
        for config in QUALITY.CONFIG_CORPUS:
            result = CONTRACT.apply_adapter(case.source, dict(config))
            assert result == config
    assert "num_learning_epochs" not in QUALITY.CONFIG_CORPUS[-1]


@pytest.mark.parametrize(
    "case_id, reason",
    [
        ("wrong-hardcoded-epoch", "forwarding_mismatch"),
        ("wrong-default-insertion", "forwarding_mismatch"),
        ("wrong-dropped-kwargs", "forwarding_mismatch"),
        ("wrong-other-kwarg-literal", "forwarding_mismatch"),
    ],
)
def test_ast_safe_semantic_errors_are_rejected(case_id: str, reason: str) -> None:
    case = next(case for case in QUALITY.DEV_CASES if case.case_id == case_id)
    row = QUALITY.evaluate_candidate(case)

    assert row["ast_contract_valid"] is True
    assert row["source_execution_attempted"] is True
    assert row["admitted"] is False
    assert row["reason"] == reason


def test_ast_rejected_candidates_are_separated_and_never_executed() -> None:
    security = [
        case for case in QUALITY.DEV_CASES if case.category.startswith("security_")
    ]
    contract_ineligible = [
        case
        for case in QUALITY.DEV_CASES
        if case.category.startswith("contract_ineligible_")
    ]
    assert len(security) == 3
    assert len(contract_ineligible) == 2

    for case in security + contract_ineligible:
        row = QUALITY.evaluate_candidate(case)
        assert row["ast_contract_valid"] is False
        assert row["source_execution_attempted"] is False
        assert row["configs_checked"] == 0
        assert row["reason"] == "ast_contract_rejected"


@pytest.mark.parametrize(
    "configs",
    [[], (), {}, "not-a-config-corpus", [{"num_learning_epochs": 5}, []]],
)
def test_empty_or_non_dictionary_config_corpus_is_rejected(configs: object) -> None:
    case = QUALITY.DEV_CASES[0]

    with pytest.raises(QUALITY.QualityGateError):
        QUALITY.evaluate_candidate(case, configs=configs)  # type: ignore[arg-type]
    with pytest.raises(QUALITY.QualityGateError):
        QUALITY.run_quality_evaluation([case], configs=configs)  # type: ignore[arg-type]


def test_healthy_reference_modification_is_blocked_without_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = next(
        case
        for case in QUALITY.DEV_CASES
        if case.case_id == "healthy-reference-modification"
    )

    def fail_if_called(source: str, config: dict) -> dict:
        raise AssertionError((source, config))

    monkeypatch.setattr(CONTRACT, "apply_adapter", fail_if_called)
    row = QUALITY.evaluate_candidate(case)

    assert row["ast_contract_valid"] is True
    assert row["source_execution_attempted"] is False
    assert row["admitted"] is False
    assert row["reason"] == "unnecessary_patch"


def test_ground_truth_is_explicit_not_inferred_from_ast() -> None:
    original = QUALITY.DEV_CASES[0]
    flipped = QUALITY.CorpusCase(
        case_id="ground-truth-flipped-fixture",
        category=original.category,
        source=original.source,
        expected_admission=False,
    )

    summary, rows = QUALITY.run_quality_evaluation([flipped])

    assert rows[0]["ast_contract_valid"] is True
    assert rows[0]["admitted"] is True
    assert summary["false_accept_ids"] == ["ground-truth-flipped-fixture"]
    assert summary["counts"]["false_accept"] == 1


def test_scope_excludes_optimizer_gpu_and_policy_recovery() -> None:
    summary, _ = QUALITY.run_quality_evaluation()

    assert summary["scope"] == {
        "algorithm_dictionary_forwarding_only": True,
        "actual_ppo_optimizer_executed": False,
        "gpu_used": False,
        "policy_recovery_measured": False,
        "quality_measurement_is_policy_normalization": False,
        "reference_status_is_development_corpus_label": True,
        "current_source_is_not_external_fault_verification": True,
        "ast_rejected_execution_definition": (
            "source_execution_attempted after validate_source"
        ),
    }


def test_cli_writes_new_public_reports_with_source_bindings(tmp_path: Path) -> None:
    out_dir = tmp_path / "quality-report"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "evals.ppo_patch_gate_quality_v1",
            "--out",
            str(out_dir),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    assert completed.returncode == 0, completed.stderr
    assert {path.name for path in out_dir.iterdir()} == {
        "summary.json",
        "cases.json",
        "provenance.json",
    }
    reports = {
        path.name: json.loads(path.read_text(encoding="utf-8"))
        for path in out_dir.iterdir()
    }
    assert reports["summary.json"]["quality_status"] == "passed"
    bindings = reports["provenance.json"]["source_bindings"]
    assert set(bindings) == {
        "evals/ppo_patch_gate_quality_v1.py",
        "evals/ppo_patch_contract_v1.py",
    }
    output_text = "\n".join(
        item for report in reports.values() for item in _all_strings(report)
    )
    assert str(tmp_path) not in output_text
    assert "bench/private" not in output_text
    assert "Users\\" not in output_text

    repeated = subprocess.run(
        [
            sys.executable,
            "-m",
            "evals.ppo_patch_gate_quality_v1",
            "--out",
            str(out_dir),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert repeated.returncode != 0


def test_module_import_does_not_load_training_or_simulation_packages() -> None:
    command = (
        "import sys; import evals.ppo_patch_gate_quality_v1; "
        "blocked=('torch','isaaclab','omni','carb'); "
        "assert all(name not in sys.modules for name in blocked)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", command],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert completed.returncode == 0, completed.stderr
