"""PPO ratio 설계 파일럿의 실행 순서·판정·실패 보존을 GPU·API 없이 검증한다."""
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))
import ppo_ratio_pilot_v1 as P  # noqa: E402

TAG = "ppo_ratio_pilot_s2160_20261022"
NORMAL = {"P_value": {"critic_change": 1.0}, "P_noise": {"noise_ratio": 1.0}, "P_episode": {"timeout_ratio": 1.0},
          "P_physics": {"mismatch_count": 0}, "P_torque": {"low_speed_saturation": 0.0},
          "P_reward": {"track_lin_vel_xy_exp_rel_error": 0.0, "track_ang_vel_z_exp_rel_error": 0.0}}


class Backend:
    def __init__(self, fault_measurements=None, control_gradient="normal", fail_stage=None):
        self.calls = []
        self.fault = {**NORMAL, **(fault_measurements or {})}
        self.control_gradient = control_gradient
        self.fail_stage = fail_stage

    def preflight(self, isaaclab):
        self.calls.append(("preflight",))
        return {"processes_started": 0}

    def train(self, stage, isaaclab, arm, seed, run_name, binding):
        self.calls.append(("train", arm, binding["max_iterations"]))
        if self.fail_stage == ("train", arm):
            raise RuntimeError("mock training failure")
        stage.mkdir(parents=True)
        gradient = "abnormal" if arm == "INVERSE_RATIO" else "normal"
        return {"arm": arm, "seed": seed, "checkpoint": {"path": str(stage / "model_299.pt"), "sha256": "a" * 64},
                "execution": {"wall_s": 600.0}, "readback": {"arm": arm},
                "inspection": {"count_status": "normal", "gradient_status": gradient}}

    def fixed_evaluate(self, stage, isaaclab, artifacts, run_name, binding):
        self.calls.append(("evaluate", tuple(artifacts)))
        stage.mkdir(parents=True)
        reports = {}
        for arm in artifacts:
            source = "upstream_r1_none_s2042" if arm == "NONE" else "upstream_r1_s2042"
            report = P.read(ROOT / f"evals/results/upstream_r1_smoke_20261014/runs/{source}.json")
            report["name"] = run_name + "_" + arm.lower()
            report["checkpoint"]["sha256"] = "a" * 64
            reports[arm] = report
        return {"reports": reports, "report_paths": {}, "execution": {"wall_s": 30.0}}

    def legacy_probe(self, stage, isaaclab, artifacts, run_name, binding, probes=None):
        arm = next(iter(artifacts))
        self.calls.append(("legacy_probe", arm, tuple(probes or ())))
        stage.mkdir(parents=True)
        values = NORMAL if arm == "NONE" else self.fault
        return {"report": {"probes": {name: {"measurement": values[name], "exit": 0}
                                      for name in (probes or sorted(P.L.PROBES))}}, "execution": {"wall_s": 20.0}}

    def inspect(self, stage, isaaclab, arm, seed, run_name, binding):
        self.calls.append(("inspect", arm, binding["max_iterations"], binding["source_checkpoint_sha256"]))
        stage.mkdir(parents=True)
        gradient = "abnormal" if arm == "INVERSE_RATIO" else self.control_gradient
        return {"measurement": {"count_status": "normal", "gradient_status": gradient,
                                "optimizer_steps": [20] * 3, "minibatches": [20] * 3}, "execution": {"wall_s": 45.0}}


@pytest.fixture
def setup(tmp_path, monkeypatch):
    protocol = {**P.fixed_protocol(), "source_sha256_lf": {}, "runtime_source_sha256_lf": {}}
    monkeypatch.setattr(P, "PRIVATE", tmp_path / "runs")
    original_sha = P.E.sha
    monkeypatch.setattr(P.E, "sha", lambda p, lf=False: "b" * 64 if p == P.PROTOCOL else original_sha(p, lf))
    monkeypatch.setattr(P, "verify", lambda *args, **kwargs: (protocol, "c" * 40))
    return lambda *args, **kwargs: {"status": "fresh", "seeds": [P.SEED]}


def run(setup, backend):
    return P.execute(TAG, Path("unused"), backend=backend, seed_guard=setup)


def test_pilot_runs_every_registered_observation_once_without_model_calls(setup):
    backend = Backend()
    result = run(setup, backend)
    assert result["status"] == "completed" and result["api_requests"] == 0
    assert result["source_gate"]["outcome"] == "failure_present"
    fault_probes = [call[2] for call in backend.calls if call[:2] == ("legacy_probe", "INVERSE_RATIO")]
    assert sorted(fault_probes) == [(probe,) for probe in sorted(P.L.PROBES)]
    assert ("legacy_probe", "NONE", ()) in backend.calls
    assert [call for call in backend.calls if call[0] == "inspect"] == [
        ("inspect", "NONE", 3, "a" * 64), ("inspect", "INVERSE_RATIO", 3, "a" * 64)]
    assert result["outcomes"] == {**{probe: "normal" for probe in P.L.PROBES},
                                  P.C.COUNT_PROBE: "normal", P.C.GRADIENT_PROBE: "abnormal"}
    assert {row["conclusion"] for row in result["loop_replay"].values()} == {P.C.SURROGATE}
    assert result["training_observer"]["INVERSE_RATIO"]["gradient_status"] == "abnormal"
    assert P.read(P.PRIVATE / TAG / "result.json") == result
    with pytest.raises(FileExistsError):
        run(setup, backend)


def test_pilot_records_a_competing_legacy_signal_in_the_loop_replay(setup):
    result = run(setup, Backend({"P_value": {"critic_change": 0.0}}))
    assert result["outcomes"]["P_value"] == "abnormal"
    assert result["loop_replay"]["P_reward"]["conclusion"] == P.C.VALUE
    assert result["loop_replay"][P.C.GRADIENT_PROBE]["conclusion"] == P.C.SURROGATE


def test_inspection_needs_a_normal_same_seed_control(setup):
    result = run(setup, Backend(control_gradient="abnormal"))
    assert result["inspection"][P.C.GRADIENT_PROBE]["outcome"] == "unknown"
    assert result["loop_replay"][P.C.GRADIENT_PROBE]["status"] == "unidentifiable"


def test_failure_is_recorded_with_stage_and_not_retried(setup):
    with pytest.raises(RuntimeError):
        run(setup, Backend(fail_stage=("train", "INVERSE_RATIO")))
    saved = P.read(P.PRIVATE / TAG / "result.json")
    assert saved["status"] == "failed" and saved["failure"] == {"stage": "train_INVERSE_RATIO", "type": "RuntimeError"}
    with pytest.raises(FileExistsError):
        run(setup, Backend())


def test_tag_seed_and_preflight_gates_stop_before_any_folder(setup):
    with pytest.raises(ValueError):
        P.execute("ppo_ratio_pilot_s2161_20261022", Path("unused"), backend=Backend(), seed_guard=setup)

    def reused(*args, **kwargs):
        raise ValueError("seed used")

    backend = Backend()
    with pytest.raises(ValueError, match="seed used"):
        P.execute(TAG, Path("unused"), backend=backend, seed_guard=reused)
    assert backend.calls == [("preflight",)] and not P.PRIVATE.exists()


def test_protocol_binds_new_entry_points_and_declares_no_api():
    protocol = P.fixed_protocol()
    assert protocol["api_requests"] == 0 and protocol["seed"] == 2160
    assert protocol["loop_registry"] == P.C.registry()
    assert {"bench/ppo_ratio_repair_training.py", "bench/ppo_ratio_mutation_v1.py",
            "evals/ppo_ratio_repair_execution.py"} <= P.CODE
    assert all((ROOT / name).is_file() for name in P.CODE | set(P.EVIDENCE))
