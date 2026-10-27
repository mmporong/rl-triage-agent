import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))
import build_transferred_vision_replay_v1 as R


def _actual_manifest() -> dict:
    path = ROOT / "evals/results/vision_transferred_e2e_dev_20261027_v1/e2e-manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _actual_seed(seed: int) -> dict:
    path = ROOT / f"evals/results/vision_transferred_e2e_dev_20261027_v1/seed-{seed}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_actual_e2e_binds_two_valid_auxiliary_post_observations():
    result = R.validate_e2e(_actual_manifest(), ROOT)
    observations = [row["post_observation"] for row in result["seeds"]]
    assert len(observations) == 2
    assert all(row["status"] == "observation_ready" for row in observations)
    assert all(row["used_for_decision"] is False for row in observations)
    assert [row["observation"] for row in observations] == [
        {
            "walking": "yes",
            "in_place": "no",
            "falling": "no",
            "evidence": row["observation"]["evidence"],
        }
        for row in observations
    ]


def test_post_observation_rejects_changed_frame_citation():
    document = copy.deepcopy(_actual_seed(2220))
    document["post_observation"]["observation"]["evidence"]["walking"][0]["timestamp_s"] = 1.5
    with pytest.raises(ValueError, match="프레임 근거"):
        R._post_observation([document], 2220)


def test_actual_render_separates_comparison_and_e2e_model_counts():
    comparison = R.validate_comparison(ROOT)
    e2e = R.validate_e2e(_actual_manifest(), ROOT)
    rendered = R.render(comparison, e2e, ROOT, Path("docs/replay/transferred-vision-e2e-v1.html"))
    assert "비교 모델 선택</b><span>24/24 유효 · P_reward" in rendered
    assert "새 E2E 첫 판단</b><span>2/2 유효 · P_reward" in rendered
    assert rendered.count("walking yes · in-place no · falling no") == 2
    assert rendered.count("valid · 판정 미사용") == 2
    assert "height:660px;overflow:hidden" in rendered


def test_valid_observation_values_are_rendered_from_bound_results():
    document = copy.deepcopy(_actual_seed(2220))
    observation = document["post_observation"]["observation"]
    observation.update(walking="unclear", in_place="yes", falling="unclear")
    post = R._post_observation([document], 2220)
    comparison = R.validate_comparison(ROOT)
    e2e = R.validate_e2e(_actual_manifest(), ROOT)
    e2e["seeds"][0]["post_observation"] = post
    rendered = R.render(comparison, e2e, ROOT, Path("docs/replay/test.html"))
    assert "walking unclear · in-place yes · falling unclear" in rendered


def test_observation_hold_does_not_become_valid_or_change_fixed_verdict():
    document = copy.deepcopy(_actual_seed(2220))
    document["post_observation"].update(status="observed_HOLD", observation=None,
                                         reason="contract_invalid", response_validity=["invalid"])
    post = R._post_observation([document], 2220)
    comparison = R.validate_comparison(ROOT)
    e2e = R.validate_e2e(_actual_manifest(), ROOT)
    e2e["seeds"][0]["post_observation"] = post
    rendered = R.render(comparison, e2e, ROOT, Path("docs/replay/test.html"))
    assert "HOLD · 판정 미사용" in rendered
    assert "관찰 보류" in rendered
    assert rendered.count("valid · 판정 미사용") == 1
    assert "E2E 2/2 recovered" in rendered
