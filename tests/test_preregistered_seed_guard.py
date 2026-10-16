import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))
import preregistered_seed_guard as G


@pytest.fixture
def workspace(tmp_path):
    repo, sdk = tmp_path / "repo", tmp_path / "sdk"
    for directory in (repo / "bench", repo / "evals", sdk / "logs"):
        directory.mkdir(parents=True)
    return repo, sdk


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_unexecuted_protocol_and_plan_are_not_prior_measurements(workspace):
    repo, sdk = workspace
    put(repo / "bench/protocols/future.json", {"seed": 2062})
    put(repo / "bench/private/task11/plan.json", {"seeds": [2062, 3062]})
    receipt = G.assert_fresh_seeds([2062, 3062], sdk, repo=repo)
    assert receipt["status"] == "fresh" and receipt["evidence"] == []
    assert str(repo) not in json.dumps(receipt) and str(sdk) not in json.dumps(receipt)


@pytest.mark.parametrize("relative,payload", [
    ("bench/runs_p0c/new_s2062.telemetry.json", {}),
    ("bench/runs_p0c/renamed.telemetry.json", {"run_dir_name": "date_case_s2062"}),
    ("bench/private/renamed.meta.json", {"seed": 2062}),
    ("bench/private/run/train/receipt.json", {"seed": 2062}),
    ("bench/private/run/steps/01_train_NONE.json", {"artifacts": {"run": "case_s2062"}}),
    ("bench/private/generic/steps/00_begin.json", {"seed": 2062}),
    ("bench/private/generic/request.json", {"training_seed": 2062}),
    ("bench/private/generic/failure.json", {"seed": 2062}),
    ("bench/private/generic/train/launch.json", {"args": ["python", "train.py", "--seed", "2062"]}),
    ("bench/private/generic/train/launch.json", {"argv": ["train.py", "--seed=2062"]}),
    ("bench/private/generic/train/launch.json", {"command": "python train.py --seed 2062 --headless"}),
    ("evals/results/renamed/result.json", {"training_seed": 2062}),
    ("evals/loops/renamed/state.json", {"seed": 2062}),
])
def test_used_seed_is_rejected_across_execution_records(workspace, relative, payload):
    repo, sdk = workspace
    put(repo / relative, payload)
    with pytest.raises(G.SeedReuseError):
        G.assert_fresh_seeds([2062], sdk, repo=repo)


def test_numeric_metric_and_different_seed_are_not_seed_reuse(workspace):
    repo, sdk = workspace
    put(repo / "evals/results/old/result.json", {"seed": 12062, "steps": 2062})
    put(repo / "bench/runs_p0c/old_s20620.telemetry.json", {"series": {"steps": [2062]}})
    put(repo / "bench/private/generic/launch.json", {"args": ["train.py", "--seed=20620", "--steps", "2062"]})
    assert G.assert_fresh_seeds([2062], sdk, repo=repo)["status"] == "fresh"


@pytest.mark.parametrize("named", [True, False])
def test_sdk_log_names_and_configuration_seeds_are_checked(workspace, named):
    repo, sdk = workspace
    run = sdk / "logs/rsl_rl/task" / ("date_case_s2062" if named else "renamed")
    run.mkdir(parents=True)
    if not named:
        (run / "agent.yaml").write_text("seed: 2062\n", encoding="utf-8")
    with pytest.raises(G.SeedReuseError):
        G.assert_fresh_seeds([2062], sdk, repo=repo)


def test_corrupt_execution_record_fails_closed(workspace):
    repo, sdk = workspace
    path = repo / "bench/runs_p0c/corrupt.telemetry.json"
    path.parent.mkdir()
    path.write_text("not JSON", encoding="utf-8")
    with pytest.raises(ValueError, match="읽기"):
        G.assert_fresh_seeds([2062], sdk, repo=repo)


@pytest.mark.parametrize("seeds", [[], [True], [-1], [2062, 2062]])
def test_invalid_seed_request_is_rejected(workspace, seeds):
    repo, sdk = workspace
    with pytest.raises(ValueError):
        G.assert_fresh_seeds(seeds, sdk, repo=repo)
