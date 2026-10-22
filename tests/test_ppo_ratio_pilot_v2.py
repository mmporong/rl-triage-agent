"""파일럿 v2가 v1과 같은 측정을 새 seed·backend v2로 실행하고 인프라 재시도를 기록하는지 검증한다."""
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "tests")]
import ppo_ratio_pilot_v1 as P1  # noqa: E402
import ppo_ratio_pilot_v2 as P  # noqa: E402
from test_ppo_ratio_pilot_v1 import Backend  # noqa: E402

TAG = "ppo_ratio_pilot_s2162_20261022"


@pytest.fixture
def setup(tmp_path, monkeypatch):
    protocol = {**P.fixed_protocol(), "source_sha256_lf": {}, "runtime_source_sha256_lf": {}}
    monkeypatch.setattr(P, "PRIVATE", tmp_path / "runs")
    original_sha = P.E.sha
    monkeypatch.setattr(P.E, "sha", lambda p, lf=False: "b" * 64 if p == P.PROTOCOL else original_sha(p, lf))
    monkeypatch.setattr(P, "verify", lambda *args, **kwargs: (protocol, "c" * 40))
    return lambda *args, **kwargs: {"status": "fresh", "seeds": [P.SEED]}


def test_v2_changes_only_seed_gpu_profile_and_retry_record():
    old, new = P1.fixed_protocol(), P.fixed_protocol()
    assert new["seed"] == 2162 and old["seed"] == 2160
    assert new["gpu"]["profile"] == P.E.GPU_PROFILE and new["gpu"]["base_profile"] == old["gpu"]["profile"]
    assert new["infrastructure_retry"]["previous_training_started"] is False
    changed = {"schema", "seed", "gpu", "infrastructure_retry"}
    assert {k: v for k, v in new.items() if k not in changed} == {k: v for k, v in old.items() if k not in changed}
    assert P1.CODE <= P.CODE and "bench/protocols/ppo_ratio_pilot_v1.json" in P.CODE
    assert all((ROOT / name).is_file() for name in P.CODE)


def test_v2_runs_the_same_measurements_with_the_new_seed(setup):
    backend = Backend()
    result = P.execute(TAG, Path("unused"), backend=backend, seed_guard=setup)
    assert result["status"] == "completed" and result["seed"] == 2162 and result["api_requests"] == 0
    assert {row["conclusion"] for row in result["loop_replay"].values()} == {P.C.SURROGATE}
    assert sorted(call[2] for call in backend.calls if call[:2] == ("legacy_probe", "INVERSE_RATIO")) == [
        (probe,) for probe in sorted(P.L.PROBES)]
    with pytest.raises(FileExistsError):
        P.execute(TAG, Path("unused"), backend=backend, seed_guard=setup)


def test_v1_tag_and_seed_are_not_accepted(setup):
    with pytest.raises(ValueError):
        P.execute("ppo_ratio_pilot_s2160_20261022", Path("unused"), backend=Backend(), seed_guard=setup)
    assert not P.PRIVATE.exists()


def test_binding_points_to_the_v2_protocol(setup):
    protocol = P.fixed_protocol() | {"source_sha256_lf": {}, "runtime_source_sha256_lf": {}}
    bound = P.binding(protocol, "c" * 40)
    assert bound["protocol_path"] == "bench/protocols/ppo_ratio_pilot_v2.json" and bound["protocol_sha256_lf"] == "b" * 64
