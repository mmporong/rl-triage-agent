from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = next(parent for parent in Path(__file__).resolve().parents
            if (parent / "AGENTS.md").is_file() and (parent / "bench").is_dir())
sys.path[:0] = [str(ROOT / name) for name in ("bench", "evals", "tests", "src")]
from test_agentic_brev_probe_parent_v1 import NOW, Services, world


def load():
    path = ROOT / "bench/run_agentic_brev_probe_case_v1.py"
    spec = importlib.util.spec_from_file_location("ri119_case", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M = load()


def test_case_only_executes_f1_once_and_never_reference(tmp_path):
    fixture = world(tmp_path, probe="P_reward")
    services = Services(fixture)
    result = M.run(fixture["request_ref"], execute=True, allow_fixture=True,
                   home=fixture["home"], now=NOW, services=services)
    assert result["status"] == "complete", result
    assert result["reference_execution_count"] == 0
    assert result["model_api_calls"] == 0
    assert len(services.calls) == 1
    assert len(result["completed"]) == 1
    plan = M.P.U.read_ref(result["completed"][0]["plan"])
    assert plan["identity"]["arm"] == "F1"
    assert plan["identity"]["probe"] == "P_reward"
    assert not (fixture["repo"] / "parents/operations" / ("1" * 64)).exists()


def test_prepare_has_no_external_effect_and_rejects_other_probe(tmp_path):
    fixture = world(tmp_path / "ok", probe="P_reward")
    result = M.run(fixture["request_ref"], allow_fixture=True,
                   home=fixture["home"], now=NOW)
    assert result["status"] == "prepared"
    assert result["reference_execution_count"] == 0
    assert not fixture["folder"].exists()

    wrong = world(tmp_path / "wrong", probe="P_episode")
    try:
        M.run(wrong["request_ref"], allow_fixture=True,
              home=wrong["home"], now=NOW)
    except ValueError as error:
        assert "P_reward" in str(error)
    else:
        raise AssertionError("P_episode accepted")


def test_existing_incomplete_intent_is_lookup_only(tmp_path):
    fixture = world(tmp_path, probe="P_reward")
    fixture["folder"].mkdir(parents=True)
    M.P.E.write_new(fixture["folder"] / "intent.json", {
        "schema": M.SCHEMA + "_intent", "request_ref": fixture["request_ref"],
        "authority": M.P.prepare(fixture["request_ref"], allow_fixture=True,
                                 home=fixture["home"], now=NOW)["authority"],
        "order": ["F1"], "gpu_binding": M.P.GPU_BINDING,
        "reference_execution_count": 0, "model_api_calls": 0,
    })
    result = M.run(fixture["request_ref"], execute=True, allow_fixture=True,
                   home=fixture["home"], now=NOW, services=Services(fixture))
    assert result["status"] == "pending_unknown"
    assert result["redispatch"] is False
