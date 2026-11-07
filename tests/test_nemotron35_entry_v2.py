import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from rl_triage import grounded_bounded_v2 as B
from rl_triage import nemotron35_entry_v1 as V1
from rl_triage import nemotron35_entry_v2 as V2


def fixture_module():
    path = Path(__file__).with_name("test_nemotron35_entry_v1.py")
    spec = importlib.util.spec_from_file_location("_entry_v1_test_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v2_config_loads_new_tools_and_does_not_modify_v1():
    before = V1.CONFIG
    module = V2.entry()
    config = module.check_config()
    assert config["ok"], config
    assert config["path"] == "configs/triage_nemotron35_v2.yml"
    assert module.CONFIG != V1.CONFIG and V1.CONFIG == before
    assert V1.check_config()["ok"]


def test_missing_worker_is_a_preflight_failure(monkeypatch, tmp_path):
    module = V2.entry()
    monkeypatch.setattr(module, "WORKER", tmp_path / "missing.py")
    assert "grounded_worker_not_packaged" in module.check_config()["problems"]


@pytest.mark.parametrize("name", list(V2.FUNCTION_TYPES))
def test_unbounded_tool_substitution_fails_preflight(tmp_path, name):
    module = V2.entry()
    config = yaml.safe_load(V2.CONFIG.read_text(encoding="utf-8"))
    config["functions"][name]["_type"] = "triage_grounded_evidence_group_page"
    path = tmp_path / "tampered.yml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    assert "bounded_function_types_changed" in module.check_config(path)["problems"]


@pytest.mark.parametrize("key,value", [("max_iterations", 20), ("max_history", 200),
                                       ("handle_tool_errors", True),
                                       ("truncation_retry", {"max_retries": 1, "token_increment": 100})])
def test_workflow_budget_changes_are_not_accepted(tmp_path, key, value):
    module = V2.entry()
    config = yaml.safe_load(V2.CONFIG.read_text(encoding="utf-8"))
    config["workflow"][key] = value
    path = tmp_path / "tampered.yml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    assert not module.check_config(path)["ok"]


def test_invalid_packaged_worker_is_rejected_before_output(tmp_path, monkeypatch):
    module = V2.entry()
    monkeypatch.setattr(module, "WORKER", tmp_path / "missing.py")
    with pytest.raises(ValueError, match="configuration or packaged worker"):
        module.prepare(tmp_path / "run", tmp_path / "task.json", tmp_path / "out", "execute")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("mode", ["success", "extra_page", "registry_id"])
def test_real_nat_session_uses_bounded_tools_and_writes_receipt(tmp_path, monkeypatch, mode):
    fixture = fixture_module()
    run, task = fixture._inputs(tmp_path)
    module = V2.entry()

    class Scripted(fixture._ScriptedNemotron):
        @staticmethod
        def _next_call(step, results):
            if mode == "extra_page" and step in {3, 4}:
                return "grounded_evidence_group_page", {
                    "case_id": module.CASE_ID, "catalog_sha256": results[1]["catalog_sha256"],
                    "group_id": results[1]["groups"][0]["group_id"],
                }
            name, args = fixture._ScriptedNemotron._next_call(step, results)
            if name == "write_grounded_active_assessment":
                check = args["next_check"].pop("registry_id")
                args["next_check"]["check_id"] = "grounded_active_v1" if mode == "registry_id" else check
            return name, args

    model = Scripted()
    fixture._stub_transport(monkeypatch, model)
    monkeypatch.setenv(module.KEY_ENV, fixture.FAKE_KEY)
    before = hashlib.sha256((run / "telemetry.json").read_bytes()).hexdigest()
    prepared = module.prepare(run, task, tmp_path / "out", "execute")
    result = module.execute(prepared)
    receipt = json.loads((tmp_path / "out/bounded-session.json").read_text(encoding="utf-8"))
    assert B.CURRENT_SESSION.get() is None
    assert receipt["approval_created"] is False and receipt["gpu_runs"] == 0
    assert before == hashlib.sha256((run / "telemetry.json").read_bytes()).hexdigest()
    if mode == "success":
        assert result["status"] == "hypotheses_only", result
        assert receipt["completed"]
        assert result["tools"]["budget_valid"]
    else:
        assert result["status"] == "HOLD"
        assert not receipt["completed"]
        assert not (tmp_path / "out/workspace/preregistrations/grounded_active_assessments/user_run.json").exists()
    if mode == "extra_page":
        assert receipt["executed_calls"]["grounded_evidence_group_page"] == 2
        assert receipt["events"][-1]["backend_entered"] is False
    schemas = model.requests[0]["body"]["tools"]
    write = next(row for row in schemas if row["function"]["name"] == "write_grounded_active_assessment")
    assert "check_id" in json.dumps(write) and "grounded_active_v1" not in json.dumps(write)
