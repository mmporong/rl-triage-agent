import json

import pytest

from rl_triage import single_run as S
from rl_triage import triage_tools as T


def _telemetry():
    series = {
        "Train/mean_reward": [-2.0, -1.0, 1.0, 3.0, 4.0],
        "Custom/contact_force": [1.0, 2.0, 5.0],
    }
    summary = {
        "Train/mean_reward": {
            "n": 5, "first": -2.0, "last": 4.0, "min": -2.0, "max": 4.0,
            "mean_first_20pct": -2.0, "mean_last_20pct": 4.0,
        },
        "Custom/contact_force": {
            "n": 3, "first": 1.0, "last": 5.0, "min": 1.0, "max": 5.0,
            "mean_first_20pct": 1.0,
        },
    }
    return {"summary": summary, "series": series}


@pytest.fixture()
def single_ws(tmp_path, monkeypatch):
    reference = tmp_path / "reference"
    reference.mkdir()
    (reference / "telemetry.json").write_text(json.dumps({
        "reference_status": "absent", "summary": {}, "series": {},
    }), encoding="utf-8")
    case = tmp_path / "cases" / "one"
    case.mkdir(parents=True)
    (case / "telemetry.json").write_text(json.dumps(_telemetry()), encoding="utf-8")
    monkeypatch.setattr(T, "WORKSPACE", tmp_path)
    return tmp_path


def test_overview_has_raw_statistics_without_reference_ratio_or_missing_zero(single_ws):
    result = S.telemetry_overview("one")
    assert result["reference_status"] == "absent"
    assert all("ratio" not in row and "reference" not in row for row in result["rows"])
    contact = next(row for row in result["rows"] if row["tag"] == "Custom/contact_force")
    assert contact["mean_last_20pct"] is None


def test_get_series_returns_only_the_actual_curve_and_count(single_ws):
    result = S.get_series("one", "Train/mean_reward", points=3)
    assert result == {"tag": "Train/mean_reward", "this_run": [-2.0, 1.0, 4.0], "n": 5}


def test_reference_with_data_is_rejected(single_ws):
    path = single_ws / "reference" / "telemetry.json"
    path.write_text(json.dumps({
        "reference_status": "absent", "summary": {"x": {"n": 1}}, "series": {"x": [1]},
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="빈 reference"):
        S.telemetry_overview("one")


def test_write_assessment_accepts_free_form_hypotheses_and_refuses_overwrite(single_ws):
    hypotheses = ["sensor saturation", "unmodeled contact interaction"]
    first = S.write_assessment("one", hypotheses, "force max=5", "no body contact label", "measure contacts")
    assert first["status"] == "hypotheses_only" and first["hypotheses"] == hypotheses
    saved = json.loads((single_ws / "preregistrations" / "assessments" / "one.json").read_text(encoding="utf-8"))
    assert saved["status"] == "hypotheses_only"
    second = S.write_assessment("one", ["another"], "e", "l", "n")
    assert "덮어쓸 수 없습니다" in second["error"]


def test_case_path_escape_is_rejected(single_ws):
    with pytest.raises(PermissionError):
        S.telemetry_overview("../../outside")


@pytest.mark.parametrize("call", [
    lambda: S.telemetry_overview("../reference"),
    lambda: S.get_series("../reference", "Train/mean_reward"),
    lambda: S.run_analysis("print(1)", "../reference"),
    lambda: S.write_assessment("../reference", ["h"], "e", "l", "n"),
])
def test_reference_directory_cannot_be_used_as_a_case(single_ws, call):
    with pytest.raises(PermissionError, match="case 경계"):
        call()


def test_run_analysis_sees_empty_reference(single_ws, monkeypatch):
    monkeypatch.setenv("TRIAGE_ANALYSIS_SANDBOX", "off")
    result = S.run_analysis("print(len(run), ref)", "one")
    assert result["exit_code"] == 0
    assert result["stdout"].strip() == "2 {}"
