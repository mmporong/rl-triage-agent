"""PPO surrogate gradient 불변식의 고정 계약과 CPU 관찰 경로를 검증한다."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))
import optimizer_observability as O  # noqa: E402
import ppo_gradient_invariant_v1 as G  # noqa: E402


def manifest(**changes):
    data = {
        "schema": O.MANIFEST_SCHEMA,
        "configured_learning_epochs": 1,
        "configured_mini_batches": 1,
        "expected_optimizer_steps_per_update": 1,
        "expected_minibatches_per_update": 1,
        "runtime_updates_to_observe": 1,
        "inspection_primitives": ["optimizer_step_count", "minibatch_count", G.PRIMITIVE],
        "gradient_invariant": {
            "schema": G.GRADIENT_SCHEMA,
            "atol": G.ATOL,
            "rtol": G.RTOL,
            "scale": G.SCALE,
            "clip_param": G.CLIP_PARAM,
        },
        "arm": "NONE",
        "seed": 1,
        "task": "CpuPpoFixture-v0",
        "num_envs": 4,
        "max_iterations": 1,
        "source_commit_sha": "a" * 40,
        "protocol_path": "bench/protocols/fake.json",
        "protocol_sha256_lf": "b" * 64,
        "source_sha256_lf": {"evals/ppo_gradient_invariant_v1.py": "c" * 64},
    }
    return data | changes


def _record(bound, status, update_index=0, *, batches=1, error=None):
    if status == "unknown":
        gradient = {"status": "unknown", "batches": [], "reason": "missing_evidence"}
    else:
        summary = {
            "max_abs_error": 0.0 if status == "normal" else 1.0,
            "mismatch_count": 0 if status == "normal" else 1,
            "nonzero_expected_count": 1,
            "finite": True,
            "n": 1,
        }
        gradient = {"status": status, "batches": [summary] * batches}
    return {
        "schema": O.SCHEMA,
        "manifest_sha256": O.manifest_fingerprint(bound),
        "binding": O.manifest_binding(bound),
        "update_index": update_index,
        "expected": {"optimizer_steps": 1, "minibatches": 1},
        "observed": {
            "optimizer_steps": 1,
            "minibatches": 1,
            G.PRIMITIVE: gradient,
        },
        "error": error,
    }


def _raises_value_error(callable_):
    try:
        callable_()
    except ValueError:
        return
    raise AssertionError("ValueError가 발생해야 한다")


def test_fixed_manifest_contract_and_update_limit_are_enforced():
    G.validate_manifest(manifest())
    bad_tolerance = manifest()
    bad_tolerance["gradient_invariant"] = bad_tolerance["gradient_invariant"] | {"atol": 1.0e-4}
    _raises_value_error(lambda: G.validate_manifest(bad_tolerance))
    _raises_value_error(lambda: G.validate_manifest(manifest(
        runtime_updates_to_observe=4, max_iterations=4)))
    _raises_value_error(lambda: G.validate_manifest(manifest(
        inspection_primitives=["optimizer_step_count", G.PRIMITIVE])))
    without_primitives = manifest()
    del without_primitives["inspection_primitives"]
    _raises_value_error(lambda: G.validate_manifest(without_primitives))


def test_registry_keeps_count_primitives_and_adds_one_decisive_gradient_primitive():
    specs = {spec.name: spec for spec in G.registry().specs()}
    assert specs["optimizer_step_count"].decisive
    assert specs["minibatch_count"].decisive
    assert specs[G.PRIMITIVE].decisive
    assert G.PRIMITIVE not in O.PRIMITIVES


def test_inspector_is_fail_closed_for_missing_evidence_and_decisive_for_mismatch():
    bound = G.validate_manifest(manifest())
    assert G.inspect_records([_record(bound, "normal")], bound)["status"] == "normal"
    abnormal = G.inspect_records([_record(bound, "abnormal")], bound)
    assert abnormal["status"] == "abnormal"
    assert abnormal["primitives"][G.PRIMITIVE]["first_mismatch_update"] == 0
    assert G.inspect_records([_record(bound, "unknown", batches=0)], bound)["status"] == "unknown"
    assert G.inspect_records([], bound)["status"] == "unknown"
    forged = _record(bound, "normal")
    forged["observed"][G.PRIMITIVE]["status"] = "abnormal"
    _raises_value_error(lambda: G.inspect_records([forged], bound))


def test_incomplete_records_stay_unknown_even_when_available_batch_is_abnormal():
    bound = G.validate_manifest(manifest(runtime_updates_to_observe=2, max_iterations=2))
    report = G.inspect_records([_record(bound, "abnormal")], bound)
    assert report["status"] == "unknown"
    assert report["primitives"][G.PRIMITIVE]["status"] == "unknown"
    assert report["primitives"][G.PRIMITIVE]["first_mismatch_update"] is None


def test_torch_observer_in_isaac_cpu_python():
    python = Path.home() / "IsaacLab" / "_isaac_sim" / "kit" / "python" / "python.exe"
    assert python.is_file(), f"Isaac CPU Python이 없다: {python}"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    env["PYTHONIOENCODING"] = "utf-8"
    process = subprocess.run(
        [str(python), str(Path(__file__).resolve()), "--torch-helper"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
    )
    assert process.returncode == 0, f"stdout:\n{process.stdout}\nstderr:\n{process.stderr}"
    result = json.loads(process.stdout.strip().splitlines()[-1])
    assert result == {"cuda": False, "status": "normal", "batches": 1}


def _torch_helper():
    import torch

    torch.set_default_device("cpu")
    assert not torch.cuda.is_available()

    class Policy(torch.nn.Module):
        is_recurrent = False

        def __init__(self):
            super().__init__()
            self.log_prob = torch.nn.Parameter(torch.zeros(4, dtype=torch.float32))

        def get_actions_log_prob(self, _actions):
            return self.log_prob

    class Storage:
        def __init__(self, advantages, old_log_prob):
            self.advantages = advantages
            self.old_log_prob = old_log_prob

        def mini_batch_generator(self, _batches, _epochs):
            placeholder = torch.zeros((4, 1), dtype=torch.float32)
            yield (
                placeholder, placeholder, placeholder, placeholder,
                self.advantages, placeholder, self.old_log_prob,
                placeholder, placeholder, (None, None), None, None,
            )

        recurrent_mini_batch_generator = mini_batch_generator

    class Optimizer:
        def __init__(self, parameter):
            self.parameter = parameter
            self.state = {parameter: {"step": 0.0}}
            self.param_groups = [{"params": [parameter], "lr": 0.01}]

        def zero_grad(self):
            self.parameter.grad = None

        def step(self):
            with torch.no_grad():
                self.parameter.add_(self.parameter.grad, alpha=-self.param_groups[0]["lr"])
            self.state[self.parameter]["step"] += 1.0

    class PPO:
        def __init__(self, *, extra_loss=False, fail=False, recurrent=False,
                     log_prob_calls=1, zero_advantages=False):
            self.policy = Policy()
            self.policy.is_recurrent = recurrent
            advantages = torch.tensor([[1.0], [1.0], [-1.0], [-1.0]], dtype=torch.float32)
            if zero_advantages:
                advantages.zero_()
            ratios = torch.tensor([1.0, 1.3, 1.0, 0.7], dtype=torch.float32)
            self.storage = Storage(advantages, (-torch.log(ratios)).unsqueeze(1))
            self.optimizer = Optimizer(self.policy.log_prob)
            self.num_learning_epochs = 1
            self.num_mini_batches = 1
            self.clip_param = 0.2
            self.symmetry = None
            self.rnd = None
            self.is_multi_gpu = False
            self.normalize_advantage_per_mini_batch = False
            self.extra_loss = extra_loss
            self.fail = fail
            self.log_prob_calls = log_prob_calls

        def update(self):
            loss = None
            for batch in self.storage.mini_batch_generator(1, 1):
                advantages, old_log_prob = torch.squeeze(batch[4]), torch.squeeze(batch[6])
                new_log_prob = self.policy.get_actions_log_prob(batch[2])
                for _ in range(self.log_prob_calls - 1):
                    self.policy.get_actions_log_prob(batch[2])
                ratio = torch.exp(new_log_prob - old_log_prob)
                surrogate = -advantages * ratio
                clipped = -advantages * torch.clamp(ratio, 0.8, 1.2)
                loss = torch.max(surrogate, clipped).mean()
                if self.extra_loss:
                    loss = loss + new_log_prob.mean()
                self.optimizer.zero_grad()
                loss.backward()
                if self.fail:
                    raise RuntimeError("fixture failure")
                self.optimizer.step()
            assert loss is not None
            return float(loss.detach().item())

    bound = G.validate_manifest(manifest())
    plain, observed, rows = PPO(), PPO(), []
    observer = G.GradientObserver(observed, bound, rows.append).install()
    try:
        plain_result = plain.update()
        observed_result = observed.update()
    finally:
        observer.remove()
    assert plain_result == observed_result
    assert torch.equal(plain.policy.log_prob.detach(), observed.policy.log_prob.detach())
    assert "update" not in vars(observed)
    assert "mini_batch_generator" not in vars(observed.storage)
    assert "get_actions_log_prob" not in vars(observed.policy)
    evidence = rows[0]["observed"][G.PRIMITIVE]
    assert evidence["status"] == "normal"
    summary = evidence["batches"][0]
    assert summary["mismatch_count"] == 0
    assert summary["nonzero_expected_count"] == 2
    assert summary["finite"] is True and summary["n"] == 4
    assert math.isclose(summary["clipped_gradient_fraction"], 0.5)
    report = G.inspect_records(rows, bound)
    assert report["status"] == "normal"
    observed.policy.log_prob.grad = None
    observed.policy.log_prob.sum().backward()
    assert len(evidence["batches"]) == 1  # update 종료 뒤 제거된 hook은 다시 호출되지 않는다.

    bad_rows = []
    bad = PPO(extra_loss=True)
    bad_observer = G.GradientObserver(bad, bound, bad_rows.append).install()
    try:
        bad.update()
    finally:
        bad_observer.remove()
    assert bad_rows[0]["observed"][G.PRIMITIVE]["status"] == "abnormal"
    assert G.inspect_records(bad_rows, bound)["status"] == "abnormal"

    unsupported_rows = []
    unsupported = PPO(recurrent=True)
    unsupported_observer = G.GradientObserver(unsupported, bound, unsupported_rows.append).install()
    try:
        unsupported.update()
    finally:
        unsupported_observer.remove()
    assert unsupported_rows[0]["observed"][G.PRIMITIVE] == {
        "status": "unknown", "batches": [], "reason": "recurrent_policy_unsupported"
    }

    duplicate_rows = []
    duplicate = PPO(log_prob_calls=2)
    duplicate_observer = G.GradientObserver(duplicate, bound, duplicate_rows.append).install()
    try:
        duplicate.update()
    finally:
        duplicate_observer.remove()
    assert duplicate_rows[0]["observed"][G.PRIMITIVE]["status"] == "unknown"
    assert duplicate_rows[0]["observed"][G.PRIMITIVE]["reason"] == "multiple_log_prob_calls_per_batch"

    zero_rows = []
    zero = PPO(zero_advantages=True)
    zero_observer = G.GradientObserver(zero, bound, zero_rows.append).install()
    try:
        zero.update()
    finally:
        zero_observer.remove()
    assert zero_rows[0]["observed"][G.PRIMITIVE]["status"] == "unknown"
    assert zero_rows[0]["observed"][G.PRIMITIVE]["reason"] == "all_expected_gradients_zero"

    failed_rows = []
    failed = PPO(fail=True)
    failed_observer = G.GradientObserver(failed, bound, failed_rows.append).install()
    try:
        failed.update()
    except RuntimeError as exc:
        assert str(exc) == "fixture failure"
    else:
        raise AssertionError("fixture failure가 발생해야 한다")
    finally:
        failed_observer.remove()
    assert "mini_batch_generator" not in vars(failed.storage)
    assert "get_actions_log_prob" not in vars(failed.policy)
    assert failed_rows[0]["error"]["type"] == "RuntimeError"
    assert failed_rows[0]["observed"][G.PRIMITIVE]["status"] == "unknown"
    assert failed_rows[0]["observed"][G.PRIMITIVE]["reason"] == "update_exception"
    assert G.inspect_records(failed_rows, bound)["status"] == "unknown"

    print(json.dumps({"cuda": torch.cuda.is_available(), "status": report["status"],
                      "batches": report["primitives"][G.PRIMITIVE]["batches"]}))


if __name__ == "__main__" and sys.argv[1:] == ["--torch-helper"]:
    _torch_helper()
