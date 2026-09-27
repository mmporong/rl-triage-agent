import json

import pytest

from rl_triage import triage_tools as T

ENV_YAML = """\
episode_length_s: 20.0
actions:
  joint_pos:
    scale: 0.25
events:
  add_base_mass:
    params:
      mass_distribution_params: !!python/tuple
      - -1.0
      - 3.0
"""
AGENT_YAML = "algorithm:\n  gamma: 0.99\n"


def _tele(reward_last):
    series = {"Train/mean_reward": [0.0, 1.0, reward_last], "Train/mean_episode_length": [10, 500, 1000]}
    summary = {k: {"n": 3, "mean_last_20pct": v[-1]} for k, v in series.items()}
    return {"summary": summary, "series": series}


@pytest.fixture()
def ws(tmp_path, monkeypatch):
    (tmp_path / "reference" / "params").mkdir(parents=True)
    (tmp_path / "reference" / "params" / "env.yaml").write_text(ENV_YAML, encoding="utf-8")
    (tmp_path / "reference" / "params" / "agent.yaml").write_text(AGENT_YAML, encoding="utf-8")
    (tmp_path / "reference" / "telemetry.json").write_text(json.dumps(_tele(4.0)), encoding="utf-8")
    case = tmp_path / "cases" / "c01"
    case.mkdir(parents=True)
    (case / "telemetry.json").write_text(json.dumps(_tele(-2.0)), encoding="utf-8")
    (case / "case.json").write_text(json.dumps({"case_id": "c01", "overrides": [
        "agent.algorithm.gamma=0.5",
        "env.events.add_base_mass.params.mass_distribution_params=[25.0,30.0]",
        "env.actions.joint_pos.scale=1.5"]}), encoding="utf-8")
    monkeypatch.setattr(T, "WORKSPACE", tmp_path)
    return tmp_path


def test_list_changes_reads_old_values_from_reference_params(ws):
    ch = T.list_changes("c01")
    assert [c["change_id"] for c in ch] == ["ch1", "ch2", "ch3"]
    assert ch[0]["old"] == 0.99 and ch[0]["new"] == 0.5
    assert ch[1]["old"] == [-1.0, 3.0] and ch[1]["new"] == [25.0, 30.0]
    assert ch[2]["old"] == 0.25


def test_overview_ratio_against_reference(ws):
    rows = {r["tag"]: r for r in T.telemetry_overview("c01")["rows"]}
    assert rows["Train/mean_reward"]["ratio"] == -0.5


def test_workspace_escape_is_denied(ws):
    with pytest.raises(PermissionError):
        T.list_changes("../../etc")


def test_run_analysis_sees_run_and_ref(ws):
    out = T.run_analysis("print(run['Train/mean_reward'][-1] - ref['Train/mean_reward'][-1])", "c01")
    assert out["exit_code"] == 0 and out["stdout"].strip() == "-6.0"


def test_preregistration_rejects_unknown_change_id(ws):
    res = T.write_preregistration("c01", "ch9", ["ch9"], "h", "v", "g", "s")
    assert "error" in res


def test_preregistration_saved(ws):
    res = T.write_preregistration("c01", "ch3", ["ch3", "ch1", "ch2"], "h", "revert ch3", "reward>0", "sig")
    assert (ws / "preregistrations" / "c01.json").exists() and res["suspected_change_id"] == "ch3"


def test_write_diagnosis_validates_mechanisms(ws):
    assert "error" in T.write_diagnosis("c01", ["gravity"], "e", "n")
    res = T.write_diagnosis("c01", ["termination", "reward"], "episode length 50", "restore episode_length_s")
    assert (ws / "diagnoses" / "c01.json").exists() and res["mechanism_ranking"][0] == "termination"
