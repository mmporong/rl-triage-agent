"""D4 v2 사전등록과 source closure 회귀."""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("evals", "bench", "src")]
import selection_comparison_demo_v2 as D  # type: ignore[import-not-found]  # noqa: E402


def test_v2_keeps_decision_contract_and_records_infrastructure_retry():
    protocol = D.fixed_protocol()
    retry = protocol["infrastructure_retry"]
    assert protocol["schema"] == "selection_comparison_demo_protocol_v2"
    assert protocol["seed"] == 2171
    assert retry == {
        "prior_tag": "selection_comparison_s2170_20261023",
        "prior_seed": 2170,
        "prior_result_sha256": "9fcd81d66aa2ee5eabbc1fb23de9828e8ffcdf822bf3bd5f6fcde958b9380fd7",
        "prior_status": "invalid",
        "cause": "probe run_name이 공통 80자 상한을 넘어 authorization 전에 중단",
        "consumed": {"training_runs": 3, "model_calls": 2},
        "meaning": "판정·예산·모델·arm·receipt 계약은 유지하고 실행 이름만 고친 새 seed 재시도",
    }
    assert protocol["policies"] == list(D.CONTROL.POLICIES)
    assert protocol["budgets"]["owned_parent_wall_s_max"] == 16200
    assert protocol["api"]["http_cap"] == 60
    assert protocol["selection"]["invalid_or_infra"] == "HOLD_without_rule_fallback"
    assert not any(value in json.dumps(protocol) for value in (
        "C:\\\\Users", Path.home().name, "bench/private"))


def test_v2_source_closure_contains_every_repo_import():
    missing = set()
    for name in sorted(D.CODE):
        if not name.endswith(".py"):
            continue
        for node in ast.walk(ast.parse((ROOT / name).read_text(encoding="utf-8-sig"))):
            if isinstance(node, ast.Import):
                modules = [item.name for item in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module, *(node.module + "." + item.name for item in node.names)]
            else:
                continue
            for module in modules:
                for directory in ("evals", "bench", "src"):
                    stem = ROOT / directory / module.replace(".", "/")
                    for candidate in (stem.with_suffix(".py"), stem / "__init__.py"):
                        relative = candidate.relative_to(ROOT).as_posix()
                        if candidate.is_file() and relative not in D.CODE:
                            missing.add(relative)
    assert not missing, sorted(missing)


def test_v2_runtime_files_are_in_bound_source_set():
    required = {
        "evals/selection_comparison_demo_v2.py",
        "evals/selection_comparison_flow_v2.py",
        "evals/selection_comparison_worker_v2.py",
        "bench/run_selection_comparison_demo_v2.ps1",
        "tests/test_selection_comparison_demo_v2.py",
        "tests/test_selection_comparison_flow_v2.py",
        "tests/test_selection_comparison_worker_v2.py",
    }
    assert required <= D.CODE
