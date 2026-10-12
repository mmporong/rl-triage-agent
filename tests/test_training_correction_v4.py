"""v4 회복 방식(resume·resume_std_reset·restart) 교정 실험의 자식 hook, 부모 검증·판정, v3 회귀."""
from __future__ import annotations

import contextlib
import copy
import hashlib
import json
from pathlib import Path
import re
import struct
import sys
import types
from types import SimpleNamespace

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "evals", ROOT / "src", ROOT / "bench"):
    sys.path.insert(0, str(directory))

import run_training_correction as V3  # noqa: E402
import run_training_correction_v4 as H  # noqa: E402
import train_with_correction as T  # noqa: E402
import train_with_recovery_mode as R  # noqa: E402
import publish_training_correction as PUB  # noqa: E402
from rl_triage import grounded_recovery as C  # noqa: E402


SHA = "1" * 64
OTHER_SHA = "2" * 64
VERSIONS = {"torch": "2.7", "gpu": "GPU", "isaaclab": "x", "isaaclab_tasks": "x",
            "isaaclab_rl": "x", "rsl-rl-lib": "2.3.3", "isaacsim": None}
BANDS = {"survival_ratio_min": 0.9, "err_xy_ratio_max": 1.1513,
         "err_yaw_ratio_max": 1.1719, "fall_frac_max": 0.05}
WALKING, STILL = 0.2, 1.17


# ---------------------------------------------------------------- 판정 fixture

def _report(name: str, *, lin: float = WALKING, yaw: float = 0.2, checkpoint_file: str = "model_598.pt") -> dict:
    alive = 40_000
    rows = []
    for command in C.GRID:
        rows.append({"command": list(command), "fall_rate": 0.0,
                     "lin_vel_rmse_mps": lin, "yaw_rate_rmse_radps": yaw,
                     "num_envs": 40, "fall_count": 0, "alive_steps_sum": alive,
                     "lin_error_sq_sum": lin * lin * alive, "yaw_error_sq_sum": yaw * yaw * alive})
    return {"protocol": C.PROTOCOL, "protocol_sha256_lf": H.sha(H.FIXED_EVAL, True),
            "task": C.TASK, "eval_seed": C.EVAL_SEED, "num_envs": C.NUM_ENVS,
            "horizon_steps": C.HORIZON_STEPS, "step_dt": C.STEP_DT,
            "action_scale": H.FIXED["action_scale"], "standstill": dict(C.STANDSTILL),
            "versions": dict(VERSIONS), "name": name,
            "checkpoint": {"file": checkpoint_file, "sha256": SHA},
            "metrics": {"fall_rate": 0.0, "mean_survival_s": alive * len(rows) / C.NUM_ENVS * C.STEP_DT,
                        "lin_vel_rmse_mps": lin, "yaw_rate_rmse_radps": yaw},
            "by_condition": rows, "elapsed_s": 10.0,
            "started_at": "2026-10-13T00:00:00+00:00", "finished_at": "2026-10-13T00:00:10+00:00"}


def _initials() -> dict:
    rng = {"cpu_sha256": "6" * 64, "cuda_all_sha256": ["7" * 64], "cuda_device_count": 1}
    versions = {"python": "3.10.15", "python_cache_tag": "cpython-310", "torch_distribution": "2.7.0",
                "isaaclab_distribution": "0.41.3", "rsl_rl_distribution": "2.3.3",
                "isaaclab_source_commit": H.FIXED["isaaclab_commit"]}
    failed = {"path": "C:/f1/model_299.pt", "sha256": "5" * 64, "load_optimizer": True}
    source = {"iteration": 299, "policy_state_sha256": "3" * 64, "optimizer_state_sha256": "4" * 64}
    common = {"rng_state_before_learn": rng, "runtime_versions": versions}
    return {
        "clean_resume_reference": {**copy.deepcopy(common), "recovery_mode": "resume",
                                   "resume_checkpoint": {**failed, "path": "C:/none/model_299.pt", "sha256": "8" * 64},
                                   "policy_state": {"sha256": "9" * 64}, "optimizer_state": {"sha256": "a" * 64}},
        "resume_corrected": {**copy.deepcopy(common), "recovery_mode": "resume", "resume_checkpoint": dict(failed),
                             "source_checkpoint_state": dict(source),
                             "policy_state": {"sha256": "3" * 64}, "optimizer_state": {"sha256": "4" * 64}},
        "resume_corrected_std_reset": {**copy.deepcopy(common), "recovery_mode": "resume_std_reset",
                                       "resume_checkpoint": dict(failed), "source_checkpoint_state": dict(source),
                                       "policy_state": {"sha256": "b" * 64}, "optimizer_state": {"sha256": "c" * 64},
                                       "std_reset": {"policy_state_before_reset": "3" * 64,
                                                     "optimizer_state_before_reset": "4" * 64}},
        "restart_corrected": {**copy.deepcopy(common), "recovery_mode": "restart", "resume_checkpoint": None,
                              "source_checkpoint_state": None,
                              "policy_state": {"sha256": "d" * 64}, "optimizer_state": {"sha256": "e" * 64}},
    }


def _summary_inputs(*, clean=WALKING, resume=STILL, reset=WALKING, restart=WALKING,
                    original_none=WALKING, original_fault=STILL):
    lins = {"clean_resume_reference": clean, "resume_corrected": resume,
            "resume_corrected_std_reset": reset, "restart_corrected": restart}
    reports = {arm: _report(arm, lin=lin, checkpoint_file=f"model_{H.FINAL_ITERATION[H.ARM_SPECS[arm]['recovery_mode']]}.pt")
               for arm, lin in lins.items()}
    original = {"clean_reference": _report("baseline_p0c_s2026", lin=original_none, checkpoint_file="model_299.pt"),
                "fault": _report("h02_s2026", lin=original_fault, checkpoint_file="model_299.pt")}
    return reports, _initials(), original, ["err_xy_ratio"]


def test_arm_order_modes_and_fixed_contract():
    assert H.ARMS == ("clean_resume_reference", "resume_corrected", "resume_corrected_std_reset", "restart_corrected")
    assert [H.ARM_SPECS[arm]["recovery_mode"] for arm in H.ARMS] == ["resume", "resume", "resume_std_reset", "restart"]
    assert H.selection("clean_resume_reference") == ("NONE", None, "resume")
    assert H.selection("restart_corrected") == ("F1", "restore_body_frame_reward", "restart")
    assert H.FINAL_ITERATION == {"resume": 598, "resume_std_reset": 598, "restart": 299}
    assert H.PREDICTIONS == {"resume_corrected": "not_recovered", "resume_corrected_std_reset": None,
                             "restart_corrected": "recovered"}
    assert H.FIXED["approval_actor"] == "claude_operator_user_authorized_test"
    assert H.FIXED["model_calls"] == 0 and H.FIXED["model_selected_correction"] is False
    assert H.FIXED["seeds"] == [2026, 2028] and "2027" in H.FIXED["excluded_seeds"]
    assert H.FIXED["allowed_agent_config_changes"]["restart"] == ["run_name"]
    assert H.CODE == V3.CODE | {"evals/run_training_correction_v4.py", "bench/train_with_recovery_mode.py"}
    roundtrip = json.loads(json.dumps(H.FIXED, ensure_ascii=False))
    assert all(type(roundtrip[k]) is type(v) and roundtrip[k] == v for k, v in H.FIXED.items())


def test_summarize_records_expected_predictions_and_rule_hit():
    reports, initials, original, failed = _summary_inputs()
    result = H.summarize(2026, reports, initials, original, BANDS, failed)
    assert result["status"] == "complete"
    assert result["labels"] == {"resume_corrected": "not_recovered", "resume_corrected_std_reset": "recovered",
                                "restart_corrected": "recovered"}
    assert result["prediction_match"] == {"resume_corrected": True, "resume_corrected_std_reset": None,
                                          "restart_corrected": True}
    assert result["references"]["resume"]["valid"] and result["references"]["restart"]["valid"]
    assert result["restart_sham_counterpart"]["run"] == "h02_s2026"
    assert result["restart_sham_counterpart"]["reused_original_run"] is True
    rule = result["recovery_mode_rule"]
    assert (rule["recommended"], rule["outcome"], rule["rule_hit"]) == ("restart", "hit", True)
    signal = result["mechanical_signal"]
    standstill = C.STANDSTILL["lin_vel_rmse_mps"]
    assert signal["values"]["original_fault"] == STILL / standstill
    assert signal["values"]["restart_corrected"] == WALKING / standstill
    assert set(signal["values"]) == {"original_none", "original_fault", *H.ARMS}
    assert signal["prediction_met"] == {"original_fault": True, "original_none": True}
    assert result["wording"]["restart_corrected"] == H.FIXED["result_wording"]["recovered"]
    assert result["generalization_proven"] is False and result["gpu_saving_claimed"] is False
    assert result["resume_treatment_rng_hashes_match"] is True


def test_restart_uses_original_none_as_primary_reference():
    # restart가 clean_resume_reference를 기준으로 쓰면 걷지 못하는 이어 학습 기준 때문에 invalid가 된다.
    reports, initials, original, failed = _summary_inputs(clean=0.7)
    result = H.summarize(2026, reports, initials, original, BANDS, failed)
    assert result["labels"]["resume_corrected"] == "invalid"
    assert result["label_reasons"]["resume_corrected_std_reset"] == "resume_reference_invalid"
    assert result["labels"]["restart_corrected"] == "recovered"
    assert result["status"] == "partial"
    assert result["recovery_mode_rule"]["outcome"] == "undetermined"
    assert result["recovery_mode_rule"]["rule_hit"] is None


def test_restart_reference_must_walk():
    reports, initials, original, failed = _summary_inputs(original_none=0.7, restart=0.7)
    result = H.summarize(2026, reports, initials, original, BANDS, failed)
    assert result["labels"]["restart_corrected"] == "invalid"
    assert result["label_reasons"]["restart_corrected"] == "restart_reference_invalid"


@pytest.mark.parametrize(("changes", "outcome"), [
    ({"resume": WALKING}, "both_recovered"),
    ({"restart": STILL}, "recommended_not_recovered"),
])
def test_rule_outcomes(changes, outcome):
    reports, initials, original, failed = _summary_inputs(**changes)
    result = H.summarize(2026, reports, initials, original, BANDS, failed)
    assert result["recovery_mode_rule"]["outcome"] == outcome
    assert result["recovery_mode_rule"]["rule_hit"] is False


def test_rule_recommendation_requires_reward_mechanism_and_threshold():
    assert H.rule_recommendation("reward", 0.9) == "restart"
    assert H.rule_recommendation("reward", 0.89) == "resume"
    assert H.rule_recommendation("motor", 1.0) == "resume"
    labels = {"resume_corrected": "recovered", "restart_corrected": "not_recovered"}
    assert H.rule_outcome("resume", labels)["outcome"] == "hit"


def test_partial_prefix_labels_failed_and_unrun_arms_invalid():
    reports, initials, original, failed = _summary_inputs()
    for arm in ("resume_corrected_std_reset", "restart_corrected"):
        reports.pop(arm)
        initials.pop(arm)
    result = H.summarize(2028, reports, initials, original, BANDS, failed, "resume_corrected_std_reset")
    assert result["status"] == "partial"
    assert result["labels"] == {"resume_corrected": "not_recovered", "resume_corrected_std_reset": "invalid",
                                "restart_corrected": "invalid"}
    assert result["label_reasons"]["resume_corrected_std_reset"] == "execution_failed"
    assert result["label_reasons"]["restart_corrected"] == "not_executed"


def test_summarize_rejects_non_prefix_arm_set():
    reports, initials, original, failed = _summary_inputs()
    for arm in ("resume_corrected", "resume_corrected_std_reset"):
        reports.pop(arm)
        initials.pop(arm)
    with pytest.raises(ValueError, match="prefix"):
        H.summarize(2026, reports, initials, original, BANDS, failed)


def test_resume_treatment_rng_mismatch_invalidates_only_std_reset_arm():
    reports, initials, original, failed = _summary_inputs()
    initials["resume_corrected_std_reset"]["rng_state_before_learn"]["cpu_sha256"] = OTHER_SHA
    result = H.summarize(2026, reports, initials, original, BANDS, failed)
    assert result["resume_treatment_rng_hashes_match"] is False
    assert result["labels"] == {"resume_corrected": "not_recovered", "resume_corrected_std_reset": "invalid",
                                "restart_corrected": "recovered"}
    assert result["label_reasons"]["resume_corrected_std_reset"] == "resume_treatment_rng_mismatch"
    assert result["status"] == "partial" and result["recovery_mode_rule"]["outcome"] == "hit"


@pytest.mark.parametrize(("mutation", "message"), [
    ("optimizer", "beyond the std reset"),
    ("rng_malformed", "RNG observation"),
    ("checkpoint", "same checkpoint"),
    ("restart_loaded", "checkpoint usage"),
    ("clean_failed_checkpoint", "clean reference"),
])
def test_summarize_rejects_initial_state_mismatch(mutation, message):
    reports, initials, original, failed = _summary_inputs()
    if mutation == "optimizer":
        initials["resume_corrected_std_reset"]["std_reset"]["optimizer_state_before_reset"] = OTHER_SHA
    elif mutation == "rng_malformed":
        initials["restart_corrected"]["rng_state_before_learn"]["cuda_all_sha256"] = []
    elif mutation == "checkpoint":
        initials["resume_corrected_std_reset"]["resume_checkpoint"]["sha256"] = OTHER_SHA
    elif mutation == "restart_loaded":
        initials["restart_corrected"]["resume_checkpoint"] = initials["resume_corrected"]["resume_checkpoint"]
    else:
        initials["clean_resume_reference"]["resume_checkpoint"] = initials["resume_corrected"]["resume_checkpoint"]
    with pytest.raises(ValueError, match=message):
        H.summarize(2026, reports, initials, original, BANDS, failed)


def test_summarize_rejects_changed_source_failure():
    reports, initials, original, _ = _summary_inputs()
    with pytest.raises(ValueError, match="original failure"):
        H.summarize(2026, reports, initials, original, BANDS, [])


def test_aggregate_counts_labels_predictions_and_rule():
    reports, initials, original, failed = _summary_inputs()
    first = H.summarize(2026, reports, initials, original, BANDS, failed)
    reports, initials, original, failed = _summary_inputs(restart=STILL)
    second = H.summarize(2028, reports, initials, original, BANDS, failed)
    for row in (first, second):
        row["protocol_sha256_lf"] = SHA
    summary = H.aggregate([first, second])
    assert summary["complete_seed_set"] is True
    assert summary["arms"]["restart_corrected"]["recovered"] == 1
    assert summary["arms"]["restart_corrected"]["prediction_held_all_seeds"] is False
    assert summary["arms"]["resume_corrected"]["prediction_held_all_seeds"] is True
    assert summary["arms"]["resume_corrected_std_reset"]["prediction_held_all_seeds"] is None
    assert summary["recovery_mode_rule_outcomes"] == {"2026": "hit", "2028": "recommended_not_recovered"}
    second["protocol_sha256_lf"] = OTHER_SHA
    with pytest.raises(ValueError, match="different protocols"):
        H.aggregate([first, second])
    with pytest.raises(ValueError, match="distinct"):
        H.aggregate([first, first])


# ---------------------------------------------------------------- 자식 fake

def _f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


class FakeArray:
    def __init__(self, raw: bytes, *, dtype="float32", shape=(1,)):
        self.raw, self.dtype, self.shape = raw, dtype, shape

    def detach(self): return self
    def cpu(self): return self
    def contiguous(self): return self
    def numpy(self): return SimpleNamespace(tobytes=lambda order="C": self.raw)


class FakeParam:
    dtype = "float32"

    def __init__(self, values):
        self.values = [_f32(value) for value in values]

    @property
    def shape(self): return (len(self.values),)
    def detach(self): return self
    def cpu(self): return self
    def contiguous(self): return self
    def tolist(self): return list(self.values)
    def numpy(self):
        raw = struct.pack(f"<{len(self.values)}f", *self.values)
        return SimpleNamespace(tobytes=lambda order="C": raw)

    def fill_(self, value):
        self.values = [_f32(value)] * len(self.values)
        return self


class FakePolicy:
    def __init__(self, noise_std_type="scalar"):
        self.noise_std_type = noise_std_type
        self.weight = FakeParam([0.1, -0.2])
        if noise_std_type == "log":
            self.log_std = FakeParam([-0.9, -0.7])
        else:
            self.std = FakeParam([0.4, 0.5])

    def _noise(self):
        return ("log_std", self.log_std) if self.noise_std_type == "log" else ("std", self.std)

    def parameters(self):
        return [self.weight, self._noise()[1]]

    def state_dict(self):
        name, value = self._noise()
        return {name: value, "actor.0.weight": self.weight}


class FakeOptimizer:
    def __init__(self, params):
        self.param_groups = [{"lr": 0.001, "params": list(params)}]
        self.state = {param: {"step": FakeArray(b"step"), "exp_avg": FakeArray(b"avg" + bytes([index]))}
                      for index, param in enumerate(params)}

    def state_dict(self):
        flat = [param for group in self.param_groups for param in group["params"]]
        index = {id(param): position for position, param in enumerate(flat)}
        return {"state": {index[id(param)]: value for param, value in self.state.items()},
                "param_groups": [{"lr": group["lr"], "params": [index[id(param)] for param in group["params"]]}
                                 for group in self.param_groups]}


def _fake_torch(monkeypatch):
    torch = types.ModuleType("torch")

    def load(path, map_location=None, weights_only=None):
        policy = FakePolicy()
        return {"iter": int(re.fullmatch(r"model_(\d+)\.pt", Path(path).name)[1]),
                "model_state_dict": policy.state_dict(),
                "optimizer_state_dict": FakeOptimizer(policy.parameters()).state_dict()}

    torch.load = load
    torch.no_grad = contextlib.nullcontext
    torch.get_rng_state = lambda: FakeArray(b"cpu-rng", dtype="uint8", shape=(7,))
    torch.cuda = SimpleNamespace(get_rng_state_all=lambda: [FakeArray(b"cuda-rng", dtype="uint8", shape=(8,))])
    monkeypatch.setitem(sys.modules, "torch", torch)
    return torch


def test_std_reset_fills_same_parameter_and_drops_only_its_adam_moments():
    policy = FakePolicy()
    optimizer = FakeOptimizer(policy.parameters())
    parameter, weight_state = policy.std, optimizer.state[policy.weight]
    readback = R.reset_action_std(policy, optimizer, 1.0, SimpleNamespace(no_grad=contextlib.nullcontext))
    assert policy.std is parameter and policy.std.values == [1.0, 1.0]
    assert parameter not in optimizer.state and optimizer.state[policy.weight] is weight_state
    assert readback["before"] == [_f32(0.4), 0.5] and readback["after"] == [1.0, 1.0]
    assert readback["std_mean_before"] == (_f32(0.4) + 0.5) / 2 and readback["std_mean_after"] == 1.0
    assert readback["changed_tensor_paths"] == ['$["std"]']
    assert readback["optimizer_param_index"] == 1
    assert readback["optimizer_std_moment_present_before"] is True
    assert readback["optimizer_std_moment_present_after"] is False
    assert readback["optimizer_state_after_reset"] == readback["optimizer_state_expected_after_reset"]
    assert readback["optimizer_state_after_reset"] != readback["optimizer_state_before_reset"]
    source = {"policy_state_sha256": readback["policy_state_before_reset"],
              "optimizer_state_sha256": readback["optimizer_state_before_reset"]}
    initial = {"policy_state": {"sha256": readback["policy_state_after_reset"]},
               "optimizer_state": {"sha256": readback["optimizer_state_after_reset"]}}
    H.validate_std_reset(readback, source, initial, 1.0)


def test_std_reset_supports_log_parameter_and_rejects_unoptimized_std():
    policy = FakePolicy("log")
    readback = R.reset_action_std(policy, FakeOptimizer(policy.parameters()), 1.0,
                                  SimpleNamespace(no_grad=contextlib.nullcontext))
    assert readback["parameter"] == "log_std" and policy.log_std.values == [0.0, 0.0]
    assert readback["std_mean_after"] == 1.0 and readback["changed_tensor_paths"] == ['$["log_std"]']
    policy = FakePolicy()
    with pytest.raises(ValueError, match="exactly once"):
        R.reset_action_std(policy, FakeOptimizer([policy.weight]), 1.0, SimpleNamespace(no_grad=contextlib.nullcontext))


@pytest.mark.parametrize("mutation", ["optimizer_after", "other_tensor", "value", "log_type", "moment_after"])
def test_validate_std_reset_rejects_tampered_readback(mutation):
    policy = FakePolicy()
    readback = R.reset_action_std(policy, FakeOptimizer(policy.parameters()), 1.0,
                                  SimpleNamespace(no_grad=contextlib.nullcontext))
    source = {"policy_state_sha256": readback["policy_state_before_reset"],
              "optimizer_state_sha256": readback["optimizer_state_before_reset"]}
    initial = {"policy_state": {"sha256": readback["policy_state_after_reset"]},
               "optimizer_state": {"sha256": readback["optimizer_state_after_reset"]}}
    if mutation == "optimizer_after":
        initial["optimizer_state"]["sha256"] = OTHER_SHA
    elif mutation == "other_tensor":
        readback["changed_tensor_paths"] = ['$["actor.0.weight"]', '$["std"]']
    elif mutation == "value":
        readback["after"] = [0.9, 1.0]
    elif mutation == "log_type":
        readback["noise_std_type"] = "log"
    else:
        readback["optimizer_std_moment_present_after"] = True
    with pytest.raises(ValueError):
        H.validate_std_reset(readback, source, initial, 1.0)


def _child_manifest(root: Path, arm: str, *, seed: int = 2026) -> dict:
    mode = H.ARM_SPECS[arm]["recovery_mode"]
    resumed = mode != "restart"
    failed = root / "failed_run" / "model_299.pt"
    failed.parent.mkdir(parents=True, exist_ok=True)
    if not failed.exists():
        failed.write_bytes(b"resume")
    isaaclab = root / "IsaacLab"
    output = root / "private" / "tag" / arm / "train"
    return {
        "schema": "training_correction_manifest_v2", "tag": "tag", "seed": seed, "arm": arm, "recovery_mode": mode,
        "source_commit_sha": "a" * 40, "protocol_sha256_lf": "b" * 64, "isaaclab_root": str(isaaclab),
        "protocol_path": str(root / "protocol.json"), "fault": H.ARM_SPECS[arm]["fault"],
        "correction_id": H.ARM_SPECS[arm]["correction"], "num_envs": H.FIXED["num_envs"],
        "max_iterations": H.FIXED["max_iterations"], "task": "Task-v0", "run_name": f"tag_{arm}",
        "output_dir": str(output.resolve()), "final_iteration": H.FINAL_ITERATION[mode],
        "resume_checkpoint_path": str(failed.resolve()) if resumed else None,
        "resume_checkpoint_sha256": hashlib.sha256(b"resume").hexdigest() if resumed else None,
        "resume_iteration": 299 if resumed else None,
        "load_run_regex": "^failed_run$" if resumed else None,
        "checkpoint_regex": "^model_299\\.pt$" if resumed else None,
        "source_env_config_path": str(root / "failed_run/params/env.yaml"),
        "source_agent_config_path": str(root / "failed_run/params/agent.yaml"),
        "config_source_fault": H.ARM_SPECS[arm]["fault"],
        "init_noise_std": 1.0 if mode == "resume_std_reset" else None,
    }


def test_upstream_argv_restart_drops_resume_flags_and_resume_matches_v3(tmp_path):
    upstream = tmp_path / "train.py"
    restart = _child_manifest(tmp_path, "restart_corrected")
    argv = R._upstream_argv(upstream, restart)
    assert not {"--resume", "--load_run", "--checkpoint"} & set(argv)
    assert argv == [str(upstream), "--task", "Task-v0", "--seed", "2026", "--num_envs", "4096",
                    "--max_iterations", "300", "--headless", "--run_name", "tag_restart_corrected"]
    resume = _child_manifest(tmp_path, "resume_corrected")
    assert R._upstream_argv(upstream, resume) == T._upstream_argv(upstream, resume)


@pytest.mark.parametrize("mutation", ["restart_resume_path", "final", "std_on_resume", "missing_std", "fields"])
def test_manifest_mode_contract(tmp_path, mutation):
    manifest = _child_manifest(tmp_path, "restart_corrected")
    if mutation == "restart_resume_path":
        manifest["resume_checkpoint_path"] = "x"
    elif mutation == "final":
        manifest["final_iteration"] = 598
    elif mutation == "std_on_resume":
        manifest = _child_manifest(tmp_path, "resume_corrected")
        manifest["init_noise_std"] = 1.0
    elif mutation == "missing_std":
        manifest = _child_manifest(tmp_path, "resume_corrected_std_reset")
        manifest["init_noise_std"] = None
    else:
        manifest["case"] = "case_01"
    with pytest.raises(ValueError):
        R.validate_manifest_mode(manifest)


class FullH:
    """자식 실행기에 주입하는 v4 부모 대역."""

    def __init__(self, manifest, protocol, events):
        self.manifest, self.protocol, self.events = manifest, protocol, events

    def validate_manifest(self, path):
        return self.manifest, self.protocol

    def validate_readback(self, readback, fault, correction):
        self.events.append(("validate", fault, correction))

    def write_new(self, path, value):
        path = Path(path)
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
        self.events.append(("write", path.name))

    def sha(self, path, lf=False):
        path = Path(path)
        if path == Path(self.manifest["protocol_path"]):
            return self.manifest["protocol_sha256_lf"]
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def validate_training_configuration(self, log_dir, manifest):
        assert (Path(log_dir) / "params/env.yaml").is_file() and (Path(log_dir) / "params/agent.yaml").is_file()
        self.events.append(("validate", "configuration"))


def _protocol() -> dict:
    return {"correction_source_contract": {"schema": "portable", "targets": {},
                                           "runtime": {"version": [3, 10, 15], "cache_tag": "cpython-310"}},
            "action_scale": 0.25, "rsl_rl_version": "2.3.3", "isaaclab_commit": H.FIXED["isaaclab_commit"],
            "resume_inputs": {"runtime_versions": {"torch": "2.7.0+cu128", "isaaclab": "0.41.3"}}}


class FakeRunner:
    def __init__(self, env, train_cfg, log_dir=None, device=None):
        self.env, self.num_steps_per_env, self.log_dir = env, H.FIXED["num_steps_per_env"], log_dir
        policy = FakePolicy()
        self.alg = SimpleNamespace(policy=policy, optimizer=FakeOptimizer(policy.parameters()))
        self.current_learning_iteration = 0

    def load(self, path, load_optimizer=True):
        self.current_learning_iteration = 299
        return {"loaded": True}

    def learn(self, num_learning_iterations, init_at_random_ep_len):
        for index in range(num_learning_iterations * self.num_steps_per_env):
            self.env.step(index)
        final = self.current_learning_iteration + num_learning_iterations - 1
        (Path(self.log_dir) / f"model_{final}.pt").write_bytes(f"final {final}".encode())
        return "done"


def _install_child_fakes(monkeypatch, root: Path, manifest: dict, events: list):
    gym = types.ModuleType("gymnasium")
    env = SimpleNamespace(unwrapped=SimpleNamespace(cfg=SimpleNamespace(
        seed=manifest["seed"], scene=SimpleNamespace(num_envs=manifest["num_envs"]),
        actions=SimpleNamespace(joint_pos=SimpleNamespace(scale=0.25)))), step=lambda value: value)
    gym.make = lambda *args, **kwargs: events.append(("env", "create")) or env
    monkeypatch.setitem(sys.modules, "gymnasium", gym)
    hidden = types.ModuleType("hidden_faults")
    hidden._before_env = lambda fault: events.append(("fault", fault))

    def arm(fault):
        original = gym.make
        gym.make = lambda *args, **kwargs: (hidden._before_env(fault), original(*args, **kwargs))[1]
    hidden.arm = arm
    monkeypatch.setitem(sys.modules, "hidden_faults", hidden)
    adapter = types.ModuleType("correction_adapter")
    adapter.capture = lambda isaaclab, contract: "snapshot"
    adapter.apply = lambda snapshot, correction: events.append(("apply", correction)) or {"selected_correction": correction}
    monkeypatch.setitem(sys.modules, "correction_adapter", adapter)
    runners = types.ModuleType("rsl_rl.runners")
    runners.OnPolicyRunner = FakeRunner
    monkeypatch.setattr(FakeRunner, "__init__", FakeRunner.__init__)
    monkeypatch.setattr(FakeRunner, "load", FakeRunner.load)
    monkeypatch.setattr(FakeRunner, "learn", FakeRunner.learn)
    monkeypatch.setitem(sys.modules, "rsl_rl.runners", runners)
    _fake_torch(monkeypatch)
    smoke = types.ModuleType("runtime_readback_smoke")
    smoke.process_identity = lambda: {"process_id": 41, "process_creation_time_filetime": 42}
    monkeypatch.setitem(sys.modules, "runtime_readback_smoke", smoke)
    monkeypatch.setattr(R.T, "_runtime_versions", lambda: {
        "python": "3.10.15", "python_cache_tag": "cpython-310", "torch_distribution": "2.7.0+cu128",
        "rsl_rl_distribution": "2.3.3", "isaaclab_distribution": "0.41.3"})
    isaaclab = Path(manifest["isaaclab_root"])
    upstream = isaaclab / "scripts/reinforcement_learning/rsl_rl/train.py"
    upstream.parent.mkdir(parents=True, exist_ok=True)
    upstream.write_text("# fake", encoding="utf-8")
    log_dir = isaaclab / "logs/rsl_rl/unitree_go2_flat" / f"2026-10-13_00-00-00_{manifest['run_name']}"

    def fake_run(path, run_name):
        env_made = gym.make("Task-v0")
        runner = runners.OnPolicyRunner(env_made, {"seed": manifest["seed"], "max_iterations": manifest["max_iterations"]},
                                        log_dir=str(log_dir), device="cpu")
        if "--resume" in sys.argv:
            runner.load(manifest["resume_checkpoint_path"])
        (log_dir / "params").mkdir(parents=True)
        (log_dir / "params/env.yaml").write_text("env", encoding="utf-8")
        (log_dir / "params/agent.yaml").write_text("agent", encoding="utf-8")
        runner.learn(num_learning_iterations=manifest["max_iterations"], init_at_random_ep_len=True)
    monkeypatch.setattr(R.runpy, "run_path", fake_run)
    monkeypatch.chdir(isaaclab)
    return log_dir


def _run_child(monkeypatch, tmp_path, arm):
    manifest, events = _child_manifest(tmp_path, arm), []
    log_dir = _install_child_fakes(monkeypatch, tmp_path, manifest, events)
    manifest_path = Path(manifest["output_dir"]) / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    receipt = R.execute(manifest_path, FullH(manifest, _protocol(), events))
    return manifest, receipt, log_dir, events


@pytest.mark.parametrize("arm", H.ARMS)
def test_child_execute_writes_mode_specific_receipt_and_parent_verifies_it(monkeypatch, tmp_path, arm):
    manifest, receipt, log_dir, events = _run_child(monkeypatch, tmp_path, arm)
    mode = manifest["recovery_mode"]
    final = H.FINAL_ITERATION[mode]
    assert events.count(("env", "create")) == 1
    assert receipt["final_iteration"] == final and Path(receipt["checkpoint"]["path"]).name == f"model_{final}.pt"
    assert receipt["final_checkpoint_state"]["iteration"] == final
    assert receipt["checkpoint_load_calls"] == (0 if mode == "restart" else 1)
    assert receipt["std_reset_applied"] is (mode == "resume_std_reset")
    initial = json.loads((Path(manifest["output_dir"]) / "initial.json").read_text(encoding="utf-8"))
    assert initial["start_iteration"] == (0 if mode == "restart" else 299)
    if mode == "restart":
        assert initial["resume_checkpoint"] is None and receipt["resume_scope"]["fresh_initialization"] is True
    if mode == "resume_std_reset":
        assert initial["std_reset"]["after"] == [1.0, 1.0]
        assert initial["std_reset"]["optimizer_std_moment_present_after"] is False

    protocol = _protocol()
    monkeypatch.setattr(H, "validate_manifest", lambda path: (manifest, protocol))
    monkeypatch.setattr(H, "validate_readback", lambda row, fault, correction: None)
    monkeypatch.setattr(H, "validate_training_configuration", lambda log_dir, manifest: None)
    execution = {"job_assigned_before_release": True, "natural_exit": True, "exit_code": 0,
                 "cleanup_verified": True, "timed_out": False, "error_type": None,
                 "owned_process_identities": {"41": 42}}
    folder = Path(manifest["output_dir"]).parents[1]
    assert H.verify_training(folder, arm, execution) == receipt


def test_parent_rejects_restart_receipt_that_loaded_a_checkpoint(monkeypatch, tmp_path):
    manifest, receipt, _, _ = _run_child(monkeypatch, tmp_path, "restart_corrected")
    path = Path(manifest["output_dir"]) / "receipt.json"
    path.write_text(json.dumps({**receipt, "checkpoint_load_calls": 1}), encoding="utf-8")
    monkeypatch.setattr(H, "validate_manifest", lambda path: (manifest, _protocol()))
    execution = {"job_assigned_before_release": True, "natural_exit": True, "exit_code": 0,
                 "cleanup_verified": True, "timed_out": False, "error_type": None,
                 "owned_process_identities": {"41": 42}}
    with pytest.raises(ValueError, match="restart arm loaded"):
        H.verify_training(Path(manifest["output_dir"]).parents[1], "restart_corrected", execution)


def test_restart_hooks_reject_any_checkpoint_load(monkeypatch, tmp_path):
    manifest, events = _child_manifest(tmp_path, "restart_corrected"), []
    _install_child_fakes(monkeypatch, tmp_path, manifest, events)
    runtime = {}
    R._install_runner_hooks(manifest, FullH(manifest, _protocol(), events), runtime, expected_action_scale=0.25)
    env = sys.modules["gymnasium"].make("Task-v0")
    runner = FakeRunner(env, {"seed": 2026, "max_iterations": 300}, log_dir=str(tmp_path / "logs"))
    with pytest.raises(RuntimeError, match="must not load"):
        runner.load(str(tmp_path / "failed_run/model_299.pt"))
    with pytest.raises(RuntimeError, match="restart arm loaded"):
        runner.learn(num_learning_iterations=300, init_at_random_ep_len=True)


def test_std_reset_hook_runs_after_verified_load_and_before_learning(monkeypatch, tmp_path):
    manifest, events = _child_manifest(tmp_path, "resume_corrected_std_reset"), []
    _install_child_fakes(monkeypatch, tmp_path, manifest, events)
    runtime = {}
    R._install_runner_hooks(manifest, FullH(manifest, _protocol(), events), runtime, expected_action_scale=0.25)
    env = sys.modules["gymnasium"].make("Task-v0")
    runner = FakeRunner(env, {"seed": 2026, "max_iterations": 300}, log_dir=str(tmp_path / "logs"))
    parameter = runner.alg.policy.std
    with pytest.raises(RuntimeError, match="requires the registered resume checkpoint"):
        runner.learn(num_learning_iterations=300, init_at_random_ep_len=True)
    runner.load(manifest["resume_checkpoint_path"])
    assert runner.alg.policy.std is parameter and parameter.values == [1.0, 1.0]
    assert parameter not in runner.alg.optimizer.state
    assert runtime["std_reset"]["policy_state_before_reset"] == runtime["source_checkpoint_state"]["policy_state_sha256"]
    assert runtime["std_reset"]["optimizer_state_before_reset"] == runtime["source_checkpoint_state"]["optimizer_state_sha256"]


# ---------------------------------------------------------------- 부모 입력·설정

def test_manifest_for_binds_mode_specific_resume_fields(monkeypatch, tmp_path):
    source_run = tmp_path / "logs" / "2026-10-09_02-59-28_h02_s2026"
    paths = {"checkpoint": source_run / "model_299.pt", "environment": source_run / "params/env.yaml",
             "agent": source_run / "params/agent.yaml"}
    monkeypatch.setattr(H, "source_paths", lambda isaaclab, seed, fault: paths)
    monkeypatch.setattr(H, "read", lambda path: {"resume_inputs": {"init_noise_std": 1.0, "seeds": {"2026": {"runs": {
        "F1": {"hashes": {"checkpoint": SHA}}, "NONE": {"hashes": {"checkpoint": OTHER_SHA}}}}}}})
    request = {"seed": 2026, "source_commit_sha": "c" * 40, "protocol_sha256_lf": SHA, "isaaclab_root": str(tmp_path)}
    folder = tmp_path / "tag"
    manifests = {arm: H.manifest_for(folder, request, arm) for arm in H.ARMS}
    for arm, manifest in manifests.items():
        assert set(manifest) == R.MANIFEST_FIELDS
        assert R.validate_manifest_mode(manifest) == H.ARM_SPECS[arm]["recovery_mode"]
    restart = manifests["restart_corrected"]
    assert all(restart[key] is None for key in R.RESUME_FIELDS) and restart["final_iteration"] == 299
    assert restart["config_source_fault"] == "F1" and restart["init_noise_std"] is None
    assert manifests["resume_corrected"]["load_run_regex"] == r"^2026\-10\-09_02\-59\-28_h02_s2026$"
    assert manifests["resume_corrected"]["final_iteration"] == 598
    assert manifests["resume_corrected_std_reset"]["init_noise_std"] == 1.0
    assert manifests["clean_resume_reference"]["resume_checkpoint_sha256"] == OTHER_SHA
    assert manifests["clean_resume_reference"]["correction_id"] is None
    with pytest.raises(ValueError, match="unregistered seed"):
        H.manifest_for(folder, {**request, "seed": 2027}, "restart_corrected")


def _write_yaml(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=True), encoding="utf-8")


def _configuration_fixture(tmp_path: Path, mode: str) -> tuple[Path, dict, dict]:
    source_env = {"seed": 2026, "scene": {"num_envs": 4096}, "observations": {"policy": "fixed"}}
    source_agent = {"seed": 2026, "num_steps_per_env": 24, "max_iterations": 300, "run_name": "h02_s2026",
                    "resume": False, "load_run": ".*", "load_checkpoint": "model_.*.pt",
                    "algorithm": {"learning_rate": 0.001},
                    "policy": {"init_noise_std": 1.0, "noise_std_type": "scalar"}}
    env_path, agent_path = tmp_path / "source/env.yaml", tmp_path / "source/agent.yaml"
    _write_yaml(env_path, source_env)
    _write_yaml(agent_path, source_agent)
    resumed = mode != "restart"
    manifest = {"protocol_path": str(tmp_path / "protocol.json"), "seed": 2026, "recovery_mode": mode,
                "config_source_fault": "F1", "source_env_config_path": str(env_path),
                "source_agent_config_path": str(agent_path), "run_name": f"tag_{mode}",
                "load_run_regex": "^source$" if resumed else None,
                "checkpoint_regex": "^model_299\\.pt$" if resumed else None,
                "max_iterations": 300, "init_noise_std": 1.0 if mode == "resume_std_reset" else None}
    protocol = {"resume_inputs": {"seeds": {"2026": {"runs": {"F1": {"hashes": {
        "environment": H.sha(env_path), "agent": H.sha(agent_path)}}}}}}}
    Path(manifest["protocol_path"]).write_text(json.dumps(protocol), encoding="utf-8")
    current = copy.deepcopy(source_agent)
    current["run_name"] = manifest["run_name"]
    if resumed:
        current.update({"resume": True, "load_run": manifest["load_run_regex"],
                        "load_checkpoint": manifest["checkpoint_regex"]})
    log_dir = tmp_path / "target"
    _write_yaml(log_dir / "params/env.yaml", source_env)
    _write_yaml(log_dir / "params/agent.yaml", current)
    return log_dir, manifest, current


@pytest.mark.parametrize("mode", H.R.RECOVERY_MODES)
def test_training_configuration_allows_only_mode_arguments(tmp_path, mode):
    log_dir, manifest, _ = _configuration_fixture(tmp_path, mode)
    H.validate_training_configuration(log_dir, manifest)


@pytest.mark.parametrize("mutation", ["resume_flag", "learning_rate", "iterations"])
def test_restart_configuration_rejects_resume_or_budget_drift(tmp_path, mutation):
    log_dir, manifest, current = _configuration_fixture(tmp_path, "restart")
    if mutation == "resume_flag":
        current["resume"] = True
    elif mutation == "learning_rate":
        current["algorithm"]["learning_rate"] = 0.002
    else:
        current["max_iterations"] = 600
    _write_yaml(log_dir / "params/agent.yaml", current)
    with pytest.raises(ValueError):
        H.validate_training_configuration(log_dir, manifest)


def test_std_reset_configuration_requires_source_init_noise_std(tmp_path):
    log_dir, manifest, _ = _configuration_fixture(tmp_path, "resume_std_reset")
    manifest["init_noise_std"] = 0.8
    with pytest.raises(ValueError, match="std reset value"):
        H.validate_training_configuration(log_dir, manifest)


def test_gpu_busy_returns_before_sdk_launch(monkeypatch, tmp_path):
    isaaclab = tmp_path / "IsaacLab"
    (isaaclab / "logs/rsl_rl/unitree_go2_flat").mkdir(parents=True)
    output = tmp_path / "private" / "busy_tag"
    monkeypatch.setattr(H, "verify_protocol", lambda root: ({"fixed": True}, OTHER_SHA))
    monkeypatch.setattr(H, "folder_for", lambda tag: output)
    monkeypatch.setattr(H, "PRIVATE", tmp_path / "private")
    monkeypatch.setattr(H, "sha", lambda path, lf=False: SHA)
    monkeypatch.setattr(H.M, "query_gpu", lambda profile: {"busy": True})
    monkeypatch.setattr(H.P, "run_owned", lambda *args, **kwargs: pytest.fail("SDK process launched"))
    assert H.execute("busy_tag", 2026, isaaclab) == {"status": "gpu_busy_no_launch", "sdk_jobs_launched": 0}
    assert not output.exists()
    with pytest.raises(ValueError, match="seed is not registered"):
        H.execute("busy_tag", 2027, isaaclab)


def test_folder_root_is_separate_from_v3():
    assert H.PRIVATE != V3.PRIVATE and H.folder_for("reward_v4_s2026").parent == H.PRIVATE
    with pytest.raises(ValueError):
        H.folder_for("../escape")


# ---------------------------------------------------------------- 커밋된 프로토콜·v3 회귀

def test_committed_v4_protocol_binds_two_seed_inputs_and_preregistration():
    protocol = json.loads(H.PROTOCOL.read_text(encoding="utf-8"))
    assert set(protocol) == set(H.FIXED) | H.DYNAMIC
    assert all(protocol[k] == v for k, v in H.FIXED.items())
    seeds = protocol["resume_inputs"]["seeds"]
    assert set(seeds) == {"2026", "2028"} and protocol["resume_inputs"]["init_noise_std"] == 1.0
    for seed in ("2026", "2028"):
        runs = seeds[seed]["runs"]
        assert runs["NONE"]["original_label"] == "healthy" and runs["F1"]["original_label"] == "unhealthy"
        assert runs["F1"]["standstill_ratio"] >= 0.9 and runs["NONE"]["standstill_ratio"] <= 0.3
        assert seeds[seed]["rule_recommendation"] == "restart"
    assert protocol["source_files"]["evals/run_training_correction_v4.py"] == H.sha(
        ROOT / "evals/run_training_correction_v4.py", True)
    assert protocol["rule_derivation"] == H.rule_derivation()
    assert protocol["cpu_checks"]["passed_count"] == 6


def test_v3_code_protocol_and_public_replays_are_unchanged():
    assert V3.PROTOCOL == ROOT / "bench/protocols/training_correction_v3.json"
    v3 = json.loads(V3.PROTOCOL.read_text(encoding="utf-8"))
    assert H.sha(V3.PROTOCOL, True) == H.FIXED["previous_protocol"]["protocol_sha256_lf"]
    assert {name: H.sha(ROOT / name, True) for name in v3["source_files"]} == v3["source_files"]
    for tag in ("reward_resume_dev3_20261013", "motor_resume_dev_20261013", "timeout_resume_dev_20261013"):
        assert PUB.replay(ROOT / "evals/results" / tag)["status"] == "no_recovery"
