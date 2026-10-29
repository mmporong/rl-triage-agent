from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(__file__).resolve().parents[1] / "bench/prepare_selected_reward_probe_v1.py"
sys.path[:0] = [str(ROOT / name) for name in ("bench", "evals", "tests", "src")]


def load():
    spec = importlib.util.spec_from_file_location("prepare_selected_reward_probe_v1", SOURCE)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M = load()


def test_actual_production_unit_prepare_accepts_case_only_reward(tmp_path):
    from test_agentic_brev_probe_unit_v1 import NOW, world  # pyright: ignore[reportMissingImports]

    w = world(tmp_path / "unit", probe="P_reward", side="case")
    prepared = M.UNIT.prepare(w["plan_ref"], w["auth_ref"], allow_fixture=True, home=w["home"], now=NOW)
    identity = prepared["plan"]["identity"]
    assert prepared["status"] == "prepared"
    assert identity["side"] == "case"
    assert identity["arm"] == "F1"
    assert identity["probe"] == "P_reward"
    assert M.UNIT.INITIAL["case"]["checkpoint_sha256"] == M.CASE["checkpoint_sha256"]
    assert M.UNIT.INITIAL["case"]["checkpoint_sha256"] != M.REFERENCE_CHECKPOINT
