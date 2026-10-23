# pyright: reportMissingImports=false
"""D4 owned-process 누적 예산과 fail-closed audit를 CPU에서 검증한다."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "bench"), str(ROOT / "evals")]
import selection_comparison_execution_v1 as E  # noqa: E402


class Clock:
    def __init__(self, value: float = 100.0):
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def receipt(wall_s=1.0, **changes):
    value = {
        "job_assigned_before_release": True,
        "natural_exit": True,
        "timed_out": False,
        "error_type": None,
        "exit_code": 0,
        "cleanup_verified": True,
        "owned_process_identities": {},
        "wall_s": wall_s,
    }
    value.update(changes)
    return value


def apps_receipt(kind="initial"):
    profile = E.GPU_BACKEND.GPU_PROFILE
    clear = {
        "busy": False,
        "busy_reasons": [],
        "admission_policy": profile,
        "gpu_exclusive": False,
        "performance_measurement": False,
        "utilization_percent": 0,
        "memory_used_mib": 100,
        "compute_process_count": 0,
        "processes": [],
        "active_engines": [],
    }
    if kind == "initial":
        samples, identities, accepted, reason = [clear], [], True, "initial_clear"
    elif kind == "resampled":
        pid = 1320
        transient = {
            **clear,
            "busy": True,
            "busy_reasons": [
                f"unapproved_active_engine:{pid}:3D",
                f"unverified_active_engine_process:{pid}",
            ],
            "active_engines": [{
                "name": f"pid_{pid}_luid_0x0_phys_0_eng_0_engtype_3D",
                "utilization_percent": 1,
                "process_name": "csrss.exe",
            }],
        }
        identity = {"ProcessId": pid, "CreationDate": "2026-10-23T00:00:00.0000000Z", "SessionId": 1}
        samples, identities, accepted, reason = (
            [transient, clear, clear],
            [identity, identity, identity],
            True,
            "two_consecutive_strict_clear",
        )
    else:
        pid = 77556
        rejected = {
            **clear,
            "busy": True,
            "busy_reasons": [
                f"unapproved_gpu_process:{pid}",
                f"unapproved_active_engine:{pid}:3D",
                f"unverified_active_engine_process:{pid}",
            ],
            "compute_process_count": 1,
            "processes": [{"pid": pid, "name": "kit.exe", "type": "C", "known_workload": True}],
            "active_engines": [{
                "name": f"pid_{pid}_x_phys_0_eng_0_engtype_3D",
                "utilization_percent": 12,
                "process_name": "kit.exe",
            }],
        }
        samples, identities, accepted, reason = [rejected], [], False, "strict_busy_not_registered_transient"
    bounded = {
        "attempted": len(samples) > 1,
        "accepted": accepted,
        "reason": reason,
        "interval_s": 2,
        "max_samples": 3,
        "sample_count": len(samples),
        "required_consecutive_clear": 2,
        "identities": identities,
    }
    return {**samples[-1], "bounded_resample": bounded, "samples": samples}


def stage_fixture(stage: Path, kind: str, *, gpu_after="resampled"):
    stage.mkdir()
    raw = {}
    for name in E._execution_names(kind):
        value = receipt()
        (stage / name).write_text(json.dumps(value), encoding="utf-8")
        raw[name.removesuffix(".json")] = value
    (stage / "gpu_before.json").write_text(json.dumps(apps_receipt()), encoding="utf-8")
    (stage / "gpu_after.json").write_text(json.dumps(apps_receipt(gpu_after)), encoding="utf-8")
    return raw


def work(tmp_path: Path, name: str) -> Path:
    path = tmp_path / name
    path.mkdir()
    return path


def test_success_reserves_before_runner_and_caps_each_watchdog(tmp_path):
    clock = Clock()
    calls = []

    def runner(folder, args, cwd, watchdog):
        assert folder.is_dir() and not (folder / "launch.json").exists()
        calls.append((args, cwd, watchdog))
        (folder / "launch.json").open("x", encoding="utf-8").write("{}")
        clock.advance(3.0)
        return receipt(2.0)

    budget = E.BudgetedOwned(
        tmp_path / "audit",
        runner=runner,
        clock=clock,
        owned_parent_wall_cap_s=10,
        overall_elapsed_cap_s=20,
    )
    first = budget.for_stage("source_none")(work(tmp_path, "owned1"), ["python", "one"], tmp_path, 9)
    second = budget.run("source_fault", work(tmp_path, "owned2"), ["python", "two"], tmp_path, 9)
    assert first["exit_code"] == second["exit_code"] == 0
    assert [call[2] for call in calls] == [9.0, 7.0]
    assert budget.consumed_owned_parent_wall_s == 6.0
    assert sorted(path.name for path in (tmp_path / "audit").iterdir()) == [
        "0001_source_none.completed.json",
        "0001_source_none.execution.json",
        "0001_source_none.reservation.json",
        "0002_source_fault.completed.json",
        "0002_source_fault.execution.json",
        "0002_source_fault.reservation.json",
        "batch.json",
    ]


def test_budget_exhaustion_stops_before_next_runner(tmp_path):
    clock = Clock()
    calls = []

    def runner(*args):
        calls.append(args)
        clock.advance(2)
        return receipt(2)

    budget = E.BudgetedOwned(
        tmp_path / "audit",
        runner=runner,
        clock=clock,
        owned_parent_wall_cap_s=2,
        overall_elapsed_cap_s=20,
    )
    budget.run("first", work(tmp_path, "one"), ["python"], tmp_path, 10)
    with pytest.raises(RuntimeError, match="예산이 소진"):
        budget.run("second", work(tmp_path, "two"), ["python"], tmp_path, 10)
    assert len(calls) == 1
    assert not any(path.name.startswith("0002_") for path in (tmp_path / "audit").iterdir())


def test_pending_reservation_is_fail_closed_after_reopen(tmp_path):
    clock = Clock()
    audit = tmp_path / "audit"
    E.BudgetedOwned(audit, runner=lambda *_: receipt(), clock=clock)
    (audit / "0001_interrupted.reservation.json").write_text("{}", encoding="utf-8")
    reopened = E.BudgetedOwned(audit, runner=lambda *_: receipt(), clock=clock)
    with pytest.raises(RuntimeError, match="pending"):
        reopened.run("later", work(tmp_path, "owned"), ["python"], tmp_path, 1)


def test_runner_exception_records_observed_consumption_and_blocks_batch(tmp_path):
    clock = Clock()

    def broken(*_):
        clock.advance(1.25)
        raise ChildProcessError("child failed")

    budget = E.BudgetedOwned(tmp_path / "audit", runner=broken, clock=clock)
    with pytest.raises(ChildProcessError):
        budget.run("train", work(tmp_path, "owned"), ["python"], tmp_path, 10)
    failed = json.loads((tmp_path / "audit/0001_train.failed.json").read_text(encoding="utf-8"))
    assert failed["error_type"] == "ChildProcessError"
    assert failed["charged_owned_wall_s"] == 1.25
    with pytest.raises(RuntimeError, match="fail-closed"):
        budget.run("later", work(tmp_path, "later"), ["python"], tmp_path, 10)


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"timed_out": True, "natural_exit": False, "exit_code": 124}, "watchdog_timeout"),
        ({"cleanup_verified": False}, "cleanup_not_verified"),
        ({"natural_exit": False}, "execution_not_finished"),
    ],
)
def test_timeout_cleanup_or_unfinished_receipt_stops_batch(tmp_path, changes, reason):
    clock = Clock()

    def runner(*_):
        clock.advance(1)
        return receipt(1, **changes)

    budget = E.BudgetedOwned(tmp_path / "audit", runner=runner, clock=clock)
    with pytest.raises(RuntimeError, match=reason):
        budget.run("probe", work(tmp_path, "owned"), ["python"], tmp_path, 10)
    failed = json.loads((tmp_path / "audit/0001_probe.failed.json").read_text(encoding="utf-8"))
    assert failed["reason"] == reason and failed["charged_owned_wall_s"] == 1.0
    with pytest.raises(RuntimeError, match="fail-closed"):
        budget.run("later", work(tmp_path, "later"), ["python"], tmp_path, 10)


def test_missing_wall_is_unquantified_instead_of_zero_and_remains_blocked(tmp_path):
    clock = Clock()
    no_wall = receipt()
    del no_wall["wall_s"]
    budget = E.BudgetedOwned(tmp_path / "audit", runner=lambda *_: no_wall, clock=clock)
    with pytest.raises(ValueError, match="벽시계 소비"):
        budget.run("probe", work(tmp_path, "owned"), ["python"], tmp_path, 10)
    failed = json.loads((tmp_path / "audit/0001_probe.failed.json").read_text(encoding="utf-8"))
    assert failed["charged_owned_wall_s"] is None
    reopened = E.BudgetedOwned(tmp_path / "audit", runner=lambda *_: receipt(), clock=clock)
    with pytest.raises(RuntimeError, match="fail-closed"):
        reopened.run("later", work(tmp_path, "later"), ["python"], tmp_path, 10)


def test_completed_audit_tamper_is_rejected_on_reopen(tmp_path):
    clock = Clock()

    def runner(*_):
        clock.advance(1)
        return receipt(1)

    audit = tmp_path / "audit"
    budget = E.BudgetedOwned(audit, runner=runner, clock=clock)
    budget.run("probe", work(tmp_path, "owned"), ["python"], tmp_path, 10)
    execution = audit / "0001_probe.execution.json"
    data = json.loads(execution.read_text(encoding="utf-8"))
    data["wall_s"] = 0.5
    execution.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="execution SHA"):
        E.BudgetedOwned(audit, runner=runner, clock=clock)


def test_overall_elapsed_cap_is_checked_after_runner(tmp_path):
    clock = Clock()

    def runner(*_):
        clock.advance(3)
        return receipt(3)

    budget = E.BudgetedOwned(
        tmp_path / "audit",
        runner=runner,
        clock=clock,
        owned_parent_wall_cap_s=10,
        overall_elapsed_cap_s=2,
    )
    with pytest.raises(RuntimeError, match="overall_elapsed_cap_exceeded"):
        budget.run("slow", work(tmp_path, "owned"), ["python"], tmp_path, 10)
    failed = json.loads((tmp_path / "audit/0001_slow.failed.json").read_text(encoding="utf-8"))
    assert failed["reason"] == "overall_elapsed_cap_exceeded"


@pytest.mark.skipif(sys.platform != "win32", reason="Windows Job 기반 run_owned 확인")
def test_real_run_owned_executes_one_short_cpu_process(tmp_path):
    folder = work(tmp_path, "owned")
    budget = E.BudgetedOwned(tmp_path / "audit")
    result = budget.run(
        "cpu_smoke",
        folder,
        [sys.executable, "-c", "print('owned-ok')"],
        tmp_path,
        20,
    )
    assert E.RESIDUAL.V1._finished(result)
    assert (folder / "launch.json").is_file()
    assert budget.consumed_owned_parent_wall_s > 0


@pytest.mark.parametrize("kind", ["train", "inspect", "fixed_evaluate", "legacy_probe"])
def test_stage_seal_revalidates_all_owned_receipts_and_bounded_gpu(tmp_path, kind):
    stage = tmp_path / kind
    raw = stage_fixture(stage, kind)
    seal = E.seal_stage(stage, kind, raw)
    assert seal["gpu_contract"] == "apps_bounded_v1"
    assert seal["gpu_receipts"]["gpu_before.json"]["status"] == "initial_clear"
    assert seal["gpu_receipts"]["gpu_after.json"]["status"] == "resampled_clear"
    assert set(seal["execution_receipts"]) == set(E._execution_names(kind))
    assert json.loads((stage / "stage_seal.json").read_text(encoding="utf-8")) == seal
    assert E.verify_stage_seal(stage, kind, raw, seal) == seal


def test_verify_stage_seal_is_read_only_and_rejects_stored_or_raw_tamper(tmp_path):
    stage = tmp_path / "verified"
    raw = stage_fixture(stage, "inspect")
    seal = E.seal_stage(stage, "inspect", raw)
    before = (stage / "stage_seal.json").read_bytes()
    assert E.verify_stage_seal(stage, "inspect", raw, seal) == seal
    assert (stage / "stage_seal.json").read_bytes() == before

    altered = {**seal, "kind": "train"}
    with pytest.raises(ValueError, match="재계산값"):
        E.verify_stage_seal(stage, "inspect", raw, altered)

    changed_raw = {**raw, "execution": {**raw["execution"], "wall_s": 9.0}}
    with pytest.raises(ValueError):
        E.verify_stage_seal(stage, "inspect", changed_raw, seal)


@pytest.mark.parametrize("mutation", ["raw", "missing", "cleanup", "kit"])
def test_stage_seal_rejects_raw_tamper_missing_receipt_cleanup_and_kit(tmp_path, mutation):
    stage = tmp_path / mutation
    raw = stage_fixture(stage, "train")
    if mutation == "raw":
        raw["execution"] = {**raw["execution"], "wall_s": 9.0}
    elif mutation == "missing":
        (stage / "extract_execution.json").unlink()
    elif mutation == "cleanup":
        value = {**raw["checkpoint_execution"], "cleanup_verified": False}
        (stage / "checkpoint_execution.json").write_text(json.dumps(value), encoding="utf-8")
        raw["checkpoint_execution"] = value
    else:
        (stage / "gpu_after.json").write_text(json.dumps(apps_receipt("kit")), encoding="utf-8")
    with pytest.raises(ValueError):
        E.seal_stage(stage, "train", raw)


def test_backend_uses_real_gpu_context_only_during_call_and_seals_train(tmp_path, monkeypatch):
    clock = Clock()
    original_snapshot = E.RESIDUAL.V1._snapshot

    def runner(_folder, _args, _cwd, _watchdog):
        clock.advance(1)
        return receipt(1)

    budget = E.BudgetedOwned(tmp_path / "audit", runner=runner, clock=clock)
    backend = E.SelectionComparisonBackend(budget)

    def fake_train(stage, isaaclab, arm, seed, run_name, binding, owned=None):
        assert E.RESIDUAL.V1._snapshot is E.GPU_BACKEND.snapshot
        assert owned is not None
        stage.mkdir()
        result = {}
        for index, name in enumerate(E._execution_names("train"), 1):
            owned_folder = stage / f"owned_{index}"
            owned_folder.mkdir()
            value = owned(owned_folder, ["python", str(index)], tmp_path, 20)
            (stage / name).write_text(json.dumps(value), encoding="utf-8")
            result[name.removesuffix(".json")] = value
        (stage / "gpu_before.json").write_text(json.dumps(apps_receipt()), encoding="utf-8")
        (stage / "gpu_after.json").write_text(json.dumps(apps_receipt()), encoding="utf-8")
        return {**result, "arm": arm, "seed": seed, "run_name": run_name, "binding": binding, "isaaclab": str(isaaclab)}

    monkeypatch.setattr(E.RESIDUAL, "train", fake_train)
    stage = tmp_path / "train_arm"
    result = backend.train(stage, tmp_path, "FIRST", 2166, "first_s2166", {"bound": True})
    assert result["stage_seal"]["kind"] == "train"
    assert result["stage_seal"]["gpu_contract"] == "apps_bounded_v1"
    assert E.RESIDUAL.V1._snapshot is original_snapshot
    assert budget.consumed_owned_parent_wall_s == 3


def test_backend_restores_gpu_context_after_backend_exception(tmp_path, monkeypatch):
    original_snapshot = E.RESIDUAL.V1._snapshot
    budget = E.BudgetedOwned(tmp_path / "audit", runner=lambda *_: receipt())
    backend = E.SelectionComparisonBackend(budget)

    def broken(*args, **kwargs):
        assert E.RESIDUAL.V1._snapshot is E.GPU_BACKEND.snapshot
        raise RuntimeError("backend failed")

    monkeypatch.setattr(E.RESIDUAL, "legacy_probe", broken)
    with pytest.raises(RuntimeError, match="backend failed"):
        backend.legacy_probe(tmp_path / "probe", tmp_path, {}, "probe_s2166", {}, probes=["P_reward"])
    assert E.RESIDUAL.V1._snapshot is original_snapshot
    assert json.loads((tmp_path / "audit/batch_failure.json").read_text(encoding="utf-8"))["reason"] == (
        "legacy_probe:RuntimeError"
    )
    reopened = E.BudgetedOwned(tmp_path / "audit", runner=lambda *_: receipt())
    with pytest.raises(RuntimeError, match="batch_failure"):
        reopened.run("later", work(tmp_path, "later"), ["python"], tmp_path, 10)


def test_backend_rejects_other_gpu_contract_and_module_identity(tmp_path, monkeypatch):
    budget = E.BudgetedOwned(tmp_path / "audit", runner=lambda *_: receipt())
    with pytest.raises(ValueError, match="GPU contract"):
        E.SelectionComparisonBackend(budget, gpu_contract="residual_v1")
    monkeypatch.setattr(E.GPU_BACKEND, "FROZEN", object())
    with pytest.raises(RuntimeError, match="같은 동결 모듈"):
        E.SelectionComparisonBackend(budget)
