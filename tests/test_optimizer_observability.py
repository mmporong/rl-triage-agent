"""학습 루프 계측 기본 단위와 fail-closed 판정."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))
import optimizer_observability as O  # noqa: E402
sys.path.insert(0, str(ROOT / "bench"))
import inspect_optimizer_observability as CLI  # noqa: E402


def manifest(**changes):
    data = {
        "schema": O.MANIFEST_SCHEMA,
        "configured_learning_epochs": 2,
        "configured_mini_batches": 2,
        "expected_optimizer_steps_per_update": 4,
        "expected_minibatches_per_update": 4,
        "runtime_updates_to_observe": 2,
        "inspection_primitives": list(O.PRIMITIVES),
        "arm": "NONE",
        "seed": 1,
        "task": "FakeTask-v0",
        "num_envs": 4,
        "max_iterations": 2,
        "source_commit_sha": "a" * 40,
        "protocol_path": "bench/protocols/fake.json",
        "protocol_sha256_lf": "b" * 64,
        "source_sha256_lf": {"evals/fake.py": "c" * 64},
    }
    return data | changes


class Scalar:
    def __init__(self, value):
        self.value = float(value)

    def item(self):
        return self.value


class Grad:
    def __init__(self, value):
        self.value = float(value)

    def detach(self):
        return self

    def norm(self, order):
        assert order == 2
        return Scalar(abs(self.value))


class Parameter:
    def __init__(self):
        self.value = 0.0
        self.grad = None


class Policy:
    is_recurrent = False

    def __init__(self):
        self.weight = Parameter()

    def named_parameters(self):
        return [("actor.weight", self.weight)]

    def parameters(self):
        return [self.weight]


class Storage:
    def mini_batch_generator(self, batches, epochs):
        for _ in range(batches * epochs):
            yield None


class Optimizer:
    def __init__(self, parameter):
        self.parameter = parameter
        self.state = {}
        self.param_groups = [{"params": [parameter], "lr": 0.01}]

    def step(self):
        parameter = self.parameter
        if parameter.grad is not None:
            current = self.state.setdefault(parameter, {"step": Scalar(0)})["step"]
            current.value += 1
            parameter.value -= self.param_groups[0]["lr"] * parameter.grad.value


class PPO:
    def __init__(self, epochs=2, fail=False):
        self.policy = Policy()
        self.storage = Storage()
        self.optimizer = Optimizer(self.policy.weight)
        self.num_learning_epochs = epochs
        self.num_mini_batches = 2
        self.fail = fail

    def update(self):
        for _ in self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs):
            self.policy.weight.grad = Grad(3.0)
            self.optimizer.step()
            if self.fail:
                raise RuntimeError("boom")
        return "ok"


def observed(epochs=2, fail=False, bound=None):
    ppo, rows = PPO(epochs, fail), []
    observer = O.RuntimeObserver(ppo, O.validate_manifest(bound or manifest()), rows.append).install()
    return ppo, rows, observer


def test_runtime_observer_counts_without_changing_training_result_and_removes_cleanly():
    plain = PPO()
    bound = manifest(runtime_updates_to_observe=1)
    instrumented, rows, observer = observed(bound=bound)
    try:
        assert plain.update() == instrumented.update() == "ok"
        assert plain.policy.weight.value == instrumented.policy.weight.value == -0.12
        assert rows[0]["observed"]["optimizer_steps"] == 4
        assert rows[0]["observed"]["minibatches"] == 4
        assert rows[0]["observed"]["pre_step_gradient_norm"] == [3.0] * 4
        assert rows[0]["observed"]["learning_rates"] == [[0.01]] * 4
        assert rows[0]["observed"]["approx_kl"] is None
        report = O.assemble_inspection(rows, bound)
        assert report["status"] == "normal"
        assert report["primitives"]["adam_step_delta"]["status"] == "unknown"  # 첫 update 전 Adam state 없음
    finally:
        observer.remove()
    assert "update" not in vars(instrumented)


def test_runtime_manifest_not_faulty_runtime_value_drives_r1_decision():
    bound = manifest(runtime_updates_to_observe=1)
    ppo, rows, observer = observed(epochs=1, bound=bound)
    try:
        ppo.update()
    finally:
        observer.remove()
    report = O.assemble_inspection(rows, bound)
    assert report["status"] == "abnormal"
    assert report["primitives"]["optimizer_step_count"] == {"status": "abnormal", "first_mismatch_update": 0}
    assert rows[0]["runtime_configuration"]["learning_epochs"] == 1
    assert rows[0]["expected"]["optimizer_steps"] == 4


def test_incomplete_normal_records_are_unknown_but_early_mismatch_is_abnormal():
    normal, normal_rows, normal_observer = observed()
    faulty, faulty_rows, faulty_observer = observed(epochs=1)
    try:
        normal.update()
        faulty.update()
    finally:
        normal_observer.remove()
        faulty_observer.remove()
    assert O.assemble_inspection(normal_rows, manifest())["status"] == "unknown"
    assert O.assemble_inspection(faulty_rows, manifest())["status"] == "abnormal"


def test_runtime_exception_is_recorded_and_all_temporary_hooks_are_removed():
    ppo, rows, observer = observed(fail=True)
    with pytest.raises(RuntimeError, match="boom"):
        ppo.update()
    observer.remove()
    assert "update" not in vars(ppo)
    assert "mini_batch_generator" not in vars(ppo.storage)
    assert "step" not in vars(ppo.optimizer)
    assert rows[0]["error"]["type"] == "RuntimeError"
    assert O.assemble_inspection(rows, manifest())["status"] == "unknown"


def test_second_update_has_complete_adam_delta_and_gradient_evidence():
    ppo, rows, observer = observed()
    try:
        ppo.update()
        ppo.update()
    finally:
        observer.remove()
    assert rows[1]["observed"]["adam_step_delta"] == {"actor.weight": 4.0}
    report = O.assemble_inspection(rows, manifest())
    assert report["primitives"]["adam_step_delta"]["status"] == "unknown"  # 첫 update 전 state가 없어 전체는 보수적으로 미판정


@pytest.mark.parametrize("field", ["checkpoint_semantics_source_verified", "optimizer_state_continuous",
                                    "frozen_parameters_known", "parameter_set_complete"])
def test_checkpoint_comparison_refuses_normal_without_continuity_evidence(field):
    metadata = {name: True for name in ("checkpoint_semantics_source_verified", "optimizer_state_continuous",
                                        "frozen_parameters_known", "parameter_set_complete")}
    metadata["completed_update_delta"] = 3
    metadata["expected_optimizer_steps_per_update"] = 20
    metadata[field] = False
    result = O.compare_checkpoint_adam_steps({"steps": {"0": 20}}, {"steps": {"0": 80}}, metadata)
    assert result["status"] == "unknown" and field in result["reasons"]


def test_checkpoint_comparison_handles_model_zero_as_one_completed_update_only_with_source_evidence():
    metadata = {"checkpoint_semantics_source_verified": True, "optimizer_state_continuous": True,
                "frozen_parameters_known": True, "parameter_set_complete": True,
                "before_completed_updates": 1, "completed_update_delta": 3,
                "expected_optimizer_steps_per_update": 20}
    assert O.compare_checkpoint_adam_steps({"steps": {"0": 20}}, {"steps": {"0": 80}}, metadata)["status"] == "normal"
    metadata["checkpoint_semantics_source_verified"] = False
    assert O.compare_checkpoint_adam_steps({"steps": {"0": 20}}, {"steps": {"0": 80}}, metadata)["status"] == "unknown"


def test_manifest_rejects_runtime_derived_or_inconsistent_expectations():
    with pytest.raises(ValueError, match="epoch×mini-batch"):
        O.validate_manifest(manifest(expected_optimizer_steps_per_update=2))
    with pytest.raises(ValueError, match="필드 누락"):
        O.validate_manifest({"schema": O.MANIFEST_SCHEMA})


def test_cli_inspect_writes_new_report(tmp_path):
    manifest_path, records_path, output = tmp_path / "manifest.json", tmp_path / "records.jsonl", tmp_path / "out.json"
    bound = CLI.build_manifest(2, 2, 1, arm="NONE", seed=1, task="FakeTask-v0", num_envs=4,
                               max_iterations=1)
    manifest_path.write_text(json.dumps(bound), encoding="utf-8")
    ppo, rows = PPO(), []
    observer = O.RuntimeObserver(ppo, O.validate_manifest(bound), rows.append).install()
    try:
        ppo.update()
    finally:
        observer.remove()
    records_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    process = subprocess.run([sys.executable, str(ROOT / "bench/inspect_optimizer_observability.py"), "inspect",
                              "--manifest", str(manifest_path), "--records", str(records_path), "--output", str(output)],
                             capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "normal"
    bound["source_sha256_lf"]["evals/optimizer_observability.py"] = "0" * 64
    manifest_path.write_text(json.dumps(bound), encoding="utf-8")
    rejected = subprocess.run([sys.executable, str(ROOT / "bench/inspect_optimizer_observability.py"), "inspect",
                               "--manifest", str(manifest_path), "--records", str(records_path),
                               "--output", str(tmp_path / "rejected.json")], capture_output=True, text=True)
    assert rejected.returncode == 2 and "source hash commitment" in rejected.stderr


def test_record_expected_cannot_replace_manifest_expectation():
    bound = manifest(runtime_updates_to_observe=1)
    ppo, rows, observer = observed(epochs=1, bound=bound)
    try:
        ppo.update()
    finally:
        observer.remove()
    forged = {**rows[0], "expected": {"optimizer_steps": 2, "minibatches": 2}}
    with pytest.raises(ValueError, match="expected 값"):
        O.assemble_inspection([forged], bound)


def test_records_from_another_arm_or_seed_cannot_be_mixed():
    ppo, rows, observer = observed()
    try:
        ppo.update()
    finally:
        observer.remove()
    other_manifest = manifest(arm="h01", seed=2026, runtime_updates_to_observe=1)
    with pytest.raises(ValueError, match="manifest commitment"):
        O.assemble_inspection(rows, other_manifest)


def test_new_registered_primitive_can_be_composed_without_changing_assembler():
    selected = manifest(runtime_updates_to_observe=1,
                        inspection_primitives=[*O.PRIMITIVES, "record_count"])
    ppo, rows = PPO(), []
    observer = O.RuntimeObserver(ppo, O.validate_manifest(selected), rows.append).install()
    try:
        ppo.update()
    finally:
        observer.remove()
    registry = O.PrimitiveRegistry(iter(O.DEFAULT_REGISTRY.specs()))
    registry.register(O.PrimitiveSpec("record_count", lambda ctx: {"status": "observed", "value": len(ctx.records)}))
    report = O.assemble_inspection(rows, selected, registry)
    assert report["primitives"]["record_count"] == {"status": "observed", "value": 1}


def test_training_binding_rejects_arm_seed_or_budget_mixture():
    bound = manifest(arm="R1", seed=2042, task="Go2", num_envs=4096, max_iterations=3,
                     runtime_updates_to_observe=3)
    arguments = ["--task", "Go2", "--seed", "2042", "--num_envs", "4096", "--max_iterations", "3"]
    CLI.verify_training_binding(bound, "num_learning_epochs=1", None, arguments)
    with pytest.raises(ValueError, match="학습 인자"):
        CLI.verify_training_binding(bound, "num_learning_epochs=1", None,
                                    [*arguments[:-1], "4"])
    with pytest.raises(ValueError, match="arm"):
        CLI.verify_training_binding(bound, None, "NONE", arguments)
    for option, value in (("--seed", "999"), ("--task", "Other"), ("--num_envs", "8"), ("--max_iterations", "99")):
        with pytest.raises(ValueError, match="중복"):
            CLI.verify_training_binding(bound, "num_learning_epochs=1", None, [*arguments, option, value])
        with pytest.raises(ValueError, match="중복"):
            CLI.verify_training_binding(bound, "num_learning_epochs=1", None, [*arguments, f"{option}={value}"])
    with pytest.raises(ValueError, match="등록하지 않은"):
        CLI.verify_training_binding(bound, "num_learning_epochs=1", None, [*arguments, "env.seed=999"])


def test_launcher_restores_globals_when_upstream_raises(tmp_path, monkeypatch):
    from types import SimpleNamespace
    upstream = tmp_path / "scripts/reinforcement_learning/rsl_rl/train.py"
    upstream.parent.mkdir(parents=True)
    upstream.write_text("raise RuntimeError('test')", encoding="utf-8")
    bound = manifest(arm="NONE", seed=1, task="FakeTask-v0", num_envs=4, max_iterations=1,
                     runtime_updates_to_observe=1)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(bound), encoding="utf-8")
    fake_make = lambda *a, **kw: None
    gym = SimpleNamespace(make=fake_make)
    monkeypatch.setitem(sys.modules, "gymnasium", gym)
    monkeypatch.setitem(sys.modules, "hidden_faults", SimpleNamespace())
    monkeypatch.setattr(CLI, "verify_commitment", lambda *a, **kw: None)
    monkeypatch.chdir(tmp_path)
    before_path, before_argv = sys.path[:], sys.argv[:]
    with pytest.raises(RuntimeError, match="test"):
        CLI.train(path, tmp_path / "out.jsonl", None, "NONE",
                  ["--task", "FakeTask-v0", "--seed", "1", "--num_envs", "4", "--max_iterations", "1"])
    assert gym.make is fake_make
    assert sys.path == before_path and sys.argv == before_argv


def test_protocol_marks_seen_cases_as_development_and_freezes_unavailable_metrics():
    protocol = json.loads((ROOT / "bench/protocols/optimizer_observability_v1.json").read_text(encoding="utf-8"))
    assert protocol["status"] == "preregistered_before_measurement"
    assert protocol["specificity_measurement"]["kind"] == "development"
    assert protocol["primitives"]["approx_kl"].endswith("unavailable")
    assert protocol["reference_manifest"]["model_0_completed_updates"] == 1


def test_manifest_builder_binds_protocol_and_sources_without_using_runtime_ppo():
    data = CLI.build_manifest(5, 4, 3, arm="R1", seed=2042, task="Go2", num_envs=4096,
                              max_iterations=3)
    assert data["expected_optimizer_steps_per_update"] == 20
    assert data["expected_minibatches_per_update"] == 20
    assert set(data["source_sha256_lf"]) == {
        "evals/optimizer_observability.py", "bench/inspect_optimizer_observability.py", "bench/hidden_faults.py"}
    CLI.verify_commitment(data)
    data["source_sha256_lf"]["evals/optimizer_observability.py"] = "0" * 64
    with pytest.raises(ValueError, match="source hash"):
        CLI.verify_commitment(data)


def test_hidden_fault_source_is_required_in_manifest_commitment():
    data = CLI.build_manifest(5, 4, 3, arm="h04", seed=2026, task="Go2", num_envs=4096, max_iterations=3)
    del data["source_sha256_lf"]["bench/hidden_faults.py"]
    with pytest.raises(ValueError, match="source hash"):
        CLI.verify_commitment(data)


def test_runner_source_verification_requires_hash_and_update_before_model_zero_save(tmp_path):
    source = tmp_path / "on_policy_runner.py"
    source.write_text('loss_dict = self.alg.update()\nself.current_learning_iteration = it\nself.save(f"model_{it}.pt")\n',
                      encoding="utf-8")
    digest = CLI.sha256_lf(source)
    assert CLI.verify_runner_checkpoint_semantics(source, digest)
    assert not CLI.verify_runner_checkpoint_semantics(source, "0" * 64)
    source.write_text('self.save(f"model_{it}.pt")\nloss_dict = self.alg.update()\nself.current_learning_iteration = it\n',
                      encoding="utf-8")
    assert not CLI.verify_runner_checkpoint_semantics(source, CLI.sha256_lf(source))
