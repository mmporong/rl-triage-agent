import copy
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))
import build_ppo_gradient_replay_v1 as R


def payloads():
    folder = ROOT / R.PUBLIC
    return tuple(json.loads((folder / name).read_text(encoding="utf-8"))
                 for name in ("result.json", "v4_comparison.json", "cpu_profile.json"))


def test_public_export_replays_before_scene_values_are_used():
    assert R.E.verify_public(ROOT / R.PUBLIC)["status"] == "valid"
    result = R.validate_payloads(*payloads())
    assert result["fault"]["inspection"]["primitives"] == {
        "optimizer_step_count": {"status": "normal", "first_mismatch_update": None},
        "minibatch_count": {"status": "normal", "first_mismatch_update": None},
        "ppo_surrogate_gradient": {"status": "abnormal", "first_mismatch_update": 0, "batches": 60},
    }
    assert result["restored"]["inspection"]["primitives"]["ppo_surrogate_gradient"]["status"] == "normal"


@pytest.mark.parametrize("mutation", ["outcome", "cost", "fault", "restored", "profile"])
def test_cross_validation_rejects_changed_public_claims(mutation):
    result, comparison, profile = map(copy.deepcopy, payloads())
    if mutation == "outcome":
        result["outcome"] = "not_confirmed"
    elif mutation == "cost":
        result["pairs"][0]["update_wall_ratio"] = 1.5
    elif mutation == "fault":
        next(row for row in result["runs"] if row["id"] == "inverse_ratio_s2144_on")["inspection"]["primitives"]["ppo_surrogate_gradient"]["status"] = "normal"
    elif mutation == "restored":
        next(row for row in result["runs"] if row["id"] == "restored_s2144_on")["inspection"]["primitives"]["optimizer_step_count"]["status"] = "abnormal"
    else:
        next(row for row in profile["runs"] if row["label"] == "V5_ON")["counts"]["item"] = 727
    with pytest.raises(ValueError):
        R.validate_payloads(result, comparison, profile)


def test_ready_build_has_two_factual_scenes(monkeypatch):
    monkeypatch.setattr(R.B.Sources, "_committed_state", lambda *args: ("ready", None))
    data = R.build()
    assert [scene["id"] for scene in data["scenes"]] == [
        "ppo-gradient-invariant", "ppo-gradient-observer-cost",
    ]
    invariant = data["scenes"][0]["blocks"][0]["rows"]
    assert invariant[0][1]["value"] == "normal"
    assert invariant[0][3]["value"] == "abnormal"
    assert invariant[0][4]["value"] == 0
    assert invariant[1][1]["value"] == invariant[1][2]["value"] == invariant[1][3]["value"] == "normal"
    cost = data["scenes"][1]["blocks"][0]["rows"]
    assert [row[2]["value"] for row in cost[:3]] == list(R.V4_RATIOS.values())
    assert [row[2]["value"] for row in cost[3:]] == list(R.RATIOS.values())
    assert all(row[3]["value"] == "예" for row in cost[3:])


def test_precommit_build_does_not_call_untracked_results_ready(monkeypatch):
    calls = []
    monkeypatch.setattr(R.B.Sources, "_committed_state", lambda *args: ("waiting", "not_committed"))
    monkeypatch.setattr(R.E, "verify_public", lambda *args: calls.append(args))
    data = R.build()
    assert calls == []
    assert {row["status"] for row in data["sources"].values()} == {"waiting"}
    assert data["scenes"][0]["blocks"][0]["rows"][0][1]["value"] is None


def test_generated_snapshot_can_refresh_after_sources_become_committed(tmp_path, monkeypatch):
    monkeypatch.setattr(R.B.Sources, "_committed_state", lambda *args: ("waiting", "not_committed"))
    waiting = R.build()
    monkeypatch.setattr(R.B.Sources, "_committed_state", lambda *args: ("ready", None))
    ready = R.build()
    output = tmp_path / "ppo-gradient-data.js"
    monkeypatch.setattr(R, "OUTPUT", output)
    monkeypatch.setattr(R, "build", lambda: waiting)
    assert R.main([]) == 0
    first = output.read_text(encoding="utf-8")
    monkeypatch.setattr(R, "build", lambda: ready)
    assert R.main([]) == 0
    second = output.read_text(encoding="utf-8")
    assert first != second
    assert second == R.B.render(ready)


def test_public_copy_has_no_private_or_user_path():
    data = R.build()
    text = json.dumps(data, ensure_ascii=False)
    assert "bench/" + "private" not in text
    assert "LIM" + "MM" not in text
    assert "직" + "접" not in text
    assert ":\\" not in text


def test_snapshot_html_and_fixed_viewport_are_current():
    assert R.B.render(R.build()) == (ROOT / R.OUTPUT).read_text(encoding="utf-8")
    html = (ROOT / "docs/replay/ppo-gradient.html").read_text(encoding="utf-8")
    assert '<meta name="color-scheme" content="light">' in html
    assert '<script src="ppo-gradient-data.js"></script>' in html
    assert '<script src="app.js"></script>' in html
    style = (ROOT / "docs/replay/style.css").read_text(encoding="utf-8")
    app = (ROOT / "docs/replay/app.js").read_text(encoding="utf-8")
    assert "width: 1920px; height: 1080px" in style
    assert "#viewport { position: fixed; inset: 0; overflow: hidden" in style
    assert ".blocks {" in style and "overflow: hidden" in style
    assert "window.innerWidth / 1920" in app and "window.innerHeight / 1080" in app
