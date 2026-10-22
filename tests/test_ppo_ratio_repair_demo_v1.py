"""네트워크·GPU 없이 PPO ratio 종단 판의 상태 전이, 승인 소비, 실패 중단을 실행한다."""
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))
import ppo_ratio_repair_demo_v1 as D  # noqa: E402

C, M = D.C, D.MODEL_CALL
TAG = "ppo_ratio_repair_s2161_20261022"
NORMAL = {"P_value": {"critic_change": 1.0}, "P_noise": {"noise_ratio": 1.0}, "P_episode": {"timeout_ratio": 1.0},
          "P_physics": {"mismatch_count": 0}, "P_torque": {"low_speed_saturation": 0.0},
          "P_reward": {"track_lin_vel_xy_exp_rel_error": 0.0, "track_ang_vel_z_exp_rel_error": 0.0}}


class Backend:
    def __init__(self, corrected_healthy=True, fault_gradient="abnormal", control_gradient="normal",
                 fault_measurements=None, corrected_gradient="normal", fail_corrected=False):
        self.calls = []
        self.corrected_healthy, self.fail_corrected = corrected_healthy, fail_corrected
        self.fault_gradient, self.control_gradient = fault_gradient, control_gradient
        self.corrected_gradient = corrected_gradient
        self.fault = {**NORMAL, **(fault_measurements or {})}

    def preflight(self, isaaclab):
        self.calls.append(("preflight",))
        return {"processes_started": 0}

    def train(self, stage, isaaclab, arm, seed, run_name, binding):
        self.calls.append(("train", arm, seed))
        if arm == "CORRECTED" and self.fail_corrected:
            raise RuntimeError("mock training failure")
        stage.mkdir(parents=True)
        telemetry = stage / "telemetry.json"
        value = 1.2 if arm == D.FAULT else 0.15
        D.E.write_new(telemetry, {"summary": {"Train/mean_reward": {"mean_last_20pct": value}}})
        gradient = {"NONE": "normal", D.FAULT: "abnormal", "CORRECTED": self.corrected_gradient}[arm]
        return {"arm": arm, "seed": seed, "run_name": run_name,
                "checkpoint": {"path": str(stage / "model_299.pt"), "sha256": "a" * 64},
                "telemetry": {"path": str(telemetry), "sha256": D.E.sha(telemetry)},
                "execution": {"wall_s": 600.0},
                "inspection": {"count_status": "normal", "gradient_status": gradient,
                               "optimizer_steps": [20] * 3, "minibatches": [20] * 3}}

    def fixed_evaluate(self, stage, isaaclab, artifacts, run_name, binding):
        self.calls.append(("evaluate", tuple(artifacts)))
        stage.mkdir(parents=True)
        reports, paths = {}, {}
        for arm in artifacts:
            healthy = arm == "NONE" or (arm == "CORRECTED" and self.corrected_healthy)
            source = "upstream_r1_none_s2042" if healthy else "upstream_r1_s2042"
            report = D.read(ROOT / f"evals/results/upstream_r1_smoke_20261014/runs/{source}.json")
            report["name"] = run_name + "_" + arm.lower()
            report["checkpoint"]["sha256"] = "a" * 64
            path = stage / f"{arm}.json"
            D.E.write_new(path, report)
            reports[arm], paths[arm] = report, str(path)
        return {"reports": reports, "report_paths": paths, "execution": {"wall_s": 30.0}}

    def legacy_probe(self, stage, isaaclab, artifacts, run_name, binding, probes=None):
        arm = next(iter(artifacts))
        self.calls.append(("legacy_probe", arm, tuple(probes or ())))
        stage.mkdir(parents=True)
        values = NORMAL if arm == "NONE" else self.fault
        return {"report": {"probes": {name: {"measurement": values[name], "exit": 0}
                                      for name in (probes or sorted(C.L.PROBES))}}, "execution": {"wall_s": 20.0}}

    def inspect(self, stage, isaaclab, arm, seed, run_name, binding):
        self.calls.append(("inspect", arm, run_name))
        stage.mkdir(parents=True)
        assert binding["source_checkpoint_sha256"] == "a" * 64 and binding["max_iterations"] == 3
        gradient = self.fault_gradient if arm == D.FAULT else self.control_gradient
        return {"measurement": {"count_status": "normal", "gradient_status": gradient,
                                "first_mismatch_update": 0 if gradient == "abnormal" else None,
                                "optimizer_steps": [20] * 3, "minibatches": [20] * 3}, "execution": {"wall_s": 45.0}}


def model_caller(first_probe="P_reward", invalid=False, correction_hold=False):
    def caller(kind, data):
        packet = json.loads(data)
        if invalid:
            content = "{}"
        elif kind == "diagnosis":
            ranking = ["reward", "actuator", "optimizer", "exploration", "physics", "termination"]
            key = next(iter(packet["facts"]))
            content = json.dumps({"ranking": ranking, "probe": first_probe, "action": "continue", "reason": "검사",
                "claims": [{"id": key, "value": packet["facts"][key]}],
                "expected_outcomes": {h: C.public_catalog()[first_probe]["expect"][h] for h in ranking[:3]}})
        else:
            content = json.dumps({"correction_id": None if correction_hold else C.CORRECTION,
                "recovery_mode": None if correction_hold else "restart", "cited_facts": ["confirmed_subtype"],
                "checklist": {key: "rule_mode" for key in packet["checklist"]},
                "hold_reason": "보류" if correction_hold else None})
        return M.dialogue(kind, data, lambda *_: {"content": content, "provider_model": C.MODEL,
            "finish_reason": "stop", "input_tokens": 20, "output_tokens": 10, "total_tokens": 30, "wall_s": 1.0})
    return caller


@pytest.fixture
def setup(tmp_path, monkeypatch):
    protocol = {**D.fixed_protocol(), "source_sha256_lf": {}, "runtime_source_sha256_lf": {}}
    monkeypatch.setattr(D, "PRIVATE", tmp_path / "runs")
    original_sha = D.E.sha
    monkeypatch.setattr(D.E, "sha", lambda p, lf=False: "b" * 64 if p == D.PROTOCOL else original_sha(p, lf))
    monkeypatch.setattr(D, "verify", lambda *args, **kwargs: (protocol, "c" * 40))
    return lambda *args, **kwargs: {"status": "fresh", "seeds": [D.SEED, D.RESTART_SEED]}


def run(setup, backend, caller=None):
    return D.execute(TAG, Path("unused"), "operator", True, backend=backend,
                     caller=caller or model_caller(), seed_guard=setup)


def test_complete_path_confirms_surrogate_restores_ratio_and_consumes_once(setup):
    backend = Backend()
    result = run(setup, backend)
    assert result["outcome"] == "recovered" and result["demo_success"]
    assert result["api_requests"] == 2 and result["loop"]["hypotheses"] == [C.SURROGATE]
    assert [row["probe"] for row in result["probe_steps"]] == C.replay_loop(
        "P_reward", {**{p: "normal" for p in C.registry()["probes"]}, C.GRADIENT_PROBE: "abnormal"})["probes"]
    count_step = next(row for row in result["probe_steps"] if row["probe"] == C.COUNT_PROBE)
    assert count_step["outcome"] == "normal" and count_step["measurement"]["case"]["gradient_status"] == "abnormal"
    assert ("train", "CORRECTED", D.RESTART_SEED) in backend.calls
    folder = D.PRIVATE / TAG
    ledger = D.CorrectionLedger(folder / "correction.ledger.jsonl", C.correction_registry())
    assert [event["event"] for event in ledger.events()] == ["proposed", "approved", "consumed", "receipt"]
    assert ledger.events()[0]["correction_id"] == C.CORRECTION
    assert result["recovery"]["reference_metrics"] == D.read(folder / "source_evaluation/NONE.json")["metrics"]
    with pytest.raises(FileExistsError):
        run(setup, backend)


def test_model_choosing_the_gradient_probe_first_confirms_in_one_step(setup):
    result = run(setup, Backend(), model_caller(C.GRADIENT_PROBE))
    assert [row["probe"] for row in result["probe_steps"]] == [C.GRADIENT_PROBE]
    assert result["outcome"] == "recovered"


def test_torque_symptom_chosen_first_does_not_confirm_actuator(setup):
    backend = Backend(fault_measurements={"P_torque": {"low_speed_saturation": 0.75}})
    result = run(setup, backend, model_caller("P_torque"))
    assert [row["probe"] for row in result["probe_steps"]] == ["P_torque", "P_noise", C.GRADIENT_PROBE]
    assert result["probe_steps"][0]["outcome"] == "abnormal"
    assert result["loop"]["hypotheses"] == [C.SURROGATE] and result["outcome"] == "recovered"


@pytest.mark.parametrize(("backend", "outcome"), [
    (dict(fault_measurements={"P_value": {"critic_change": 0.0}}), "correction_scope_hold"),
    (dict(control_gradient="abnormal"), "loop_not_confirmed"),
    (dict(fault_gradient="normal"), "loop_not_confirmed"),
])
def test_competing_or_unverified_observations_never_restore_the_ratio(setup, backend, outcome):
    fake = Backend(**backend)
    result = run(setup, fake)
    assert result["outcome"] == outcome and not result["demo_success"]
    assert not any(call[:2] == ("train", "CORRECTED") for call in fake.calls)


@pytest.mark.parametrize(("caller", "outcome", "requests"), [
    (model_caller(invalid=True), "model_invalid", 2),
    (model_caller(correction_hold=True), "correction_not_proposed", 2),
])
def test_model_invalid_or_hold_does_not_fall_back_to_rule(setup, caller, outcome, requests):
    backend = Backend()
    result = run(setup, backend, caller)
    assert result["outcome"] == outcome and result["api_requests"] == requests
    assert not any(call[:2] == ("train", "CORRECTED") for call in backend.calls)


def test_unrestored_gradient_after_correction_is_correction_unknown(setup):
    with pytest.raises(ValueError, match="복원"):
        run(setup, Backend(corrected_gradient="abnormal"))
    result = D.read(D.PRIVATE / TAG / "result.json")
    assert result["outcome"] == "correction_unknown" and not result["demo_success"]
    ledger = D.CorrectionLedger(D.PRIVATE / TAG / "correction.ledger.jsonl", C.correction_registry())
    assert ledger.events()[-1]["verdict"] == "unknown" and ledger.orphans() == []


def test_healthy_gradient_does_not_override_failed_policy(setup):
    result = run(setup, Backend(corrected_healthy=False))
    assert result["conditions"]["restored_surrogate_gradient"]
    assert result["outcome"] == "not_recovered" and not result["demo_success"]


@pytest.mark.parametrize("outcome", ["baseline_invalid", "source_failure_absent"])
def test_source_gate_stops_before_model_or_probes(setup, monkeypatch, outcome):
    backend = Backend()
    monkeypatch.setattr(D, "source_gate", lambda *args: {"outcome": outcome})
    result = run(setup, backend)
    assert result["outcome"] == outcome and result["api_requests"] == 0
    assert not any(call[0] in ("legacy_probe", "inspect") for call in backend.calls)


def test_tag_approval_seed_and_preflight_gates_stop_before_training(setup):
    backend = Backend()
    with pytest.raises(ValueError):
        D.execute(TAG, Path("unused"), "operator", False, backend=backend, seed_guard=setup)
    with pytest.raises(ValueError):
        D.execute("ppo_ratio_repair_s2160_20261022", Path("unused"), "operator", True, backend=backend,
                  seed_guard=setup)

    def reused(*args, **kwargs):
        raise ValueError("seed used")

    with pytest.raises(ValueError, match="seed used"):
        D.execute(TAG, Path("unused"), "operator", True, backend=backend, seed_guard=reused)
    assert backend.calls == [("preflight",)] and not D.PRIVATE.exists()


def test_protocol_cannot_be_prepared_before_pilot_binding(monkeypatch):
    monkeypatch.setattr(D, "PILOT_RESULT", {"tag": None, "result_sha256": None})
    with pytest.raises(ValueError, match="파일럿"):
        D.prepare(Path("unused"))


def test_protocol_declares_cap_profile_and_binds_existing_files():
    protocol = D.fixed_protocol()
    assert protocol["model"]["total_http_request_cap"] == 8 and protocol["gpu"]["profile"] == D.E.GPU_PROFILE
    assert protocol["loop_registry"] == C.registry() and protocol["success_conditions"] == D.SUCCESS_CONDITIONS
    assert all((ROOT / name).is_file() for name in D.CODE | set(D.EVIDENCE))
