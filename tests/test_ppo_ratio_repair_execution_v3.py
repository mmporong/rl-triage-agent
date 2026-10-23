"""C3 backend의 제한 재표본과 동결 hook 복원을 검증한다."""
import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "bench"), str(ROOT / "evals")]
import ppo_ratio_repair_execution_v3 as E
import r1_repair_execution as FROZEN

PID = 1320
IDENTITY = {"ProcessId": PID, "CreationDate": "2026-10-23T00:00:00.0000000Z", "SessionId": 1}
OLD_SHA = {
    "evals/ppo_ratio_repair_execution_v2.py": "26aa7283df00c83655bb98f71b653d320daa82af5981ea6637fae987020be357",
    "evals/ppo_ratio_repair_demo_v2.py": "2bb585572d509053800b502e68a15c738b80a49064278ff9ae3630fe1b45338a",
    "tests/test_ppo_ratio_repair_execution_v2.py": "3ed55292c59908d473d2534a36b50dc0733158d51fbf6e8b46bfd46767057039",
    "tests/test_ppo_ratio_repair_demo_v2.py": "6b297a73c541f0e19d1a826819c81f91758d7f0c4a1795b4aca3e9ed49b66768",
    "bench/protocols/ppo_ratio_repair_demo_v2.json": "09d94ce18361f10859609bdac41399d69e522e0a19434893da9b5981fac8b07e",
    "bench/run_ppo_ratio_repair_demo_v2.ps1": "4ab6e7f8fa774c984c218627b14a6ee811376df02b8b81a55ffd02e7ea0234a0",
}


def clear(memory=2534, *, engines=None, processes=None):
    return {"busy": False, "busy_reasons": [], "admission_policy": E.GPU_PROFILE,
            "gpu_exclusive": False, "performance_measurement": False, "utilization_percent": 0,
            "memory_used_mib": memory, "compute_process_count": 0,
            "processes": processes or [], "active_engines": engines or []}


def transient(*, utilization=1, reasons=None, engine_type="3D", memory=2534, processes=None):
    expected = [f"unapproved_active_engine:{PID}:3D", f"unverified_active_engine_process:{PID}"]
    return {"busy": True, "busy_reasons": expected if reasons is None else reasons,
            "admission_policy": E.GPU_PROFILE, "gpu_exclusive": False, "performance_measurement": False,
            "utilization_percent": 0, "memory_used_mib": memory, "compute_process_count": 0,
            "processes": processes or [], "active_engines": [{
                "name": f"pid_{PID}_luid_0x0_phys_0_eng_0_engtype_{engine_type}",
                "utilization_percent": utilization, "process_name": "csrss.exe"}]}


def patch_samples(monkeypatch, values, identities=None):
    iterator = iter(values)
    monkeypatch.setattr(E, "query_gpu", lambda: next(iterator))
    identities = iter(identities or [dict(IDENTITY)] * 3)
    monkeypatch.setattr(E, "process_identity", lambda pid: next(identities))
    sleeps = []
    monkeypatch.setattr(E.time, "sleep", sleeps.append)
    return sleeps


def test_c2_observed_fixture_is_strict_busy_and_standalone_stays_single(monkeypatch):
    sample = transient()
    assert E._candidate_pid(sample) == PID
    calls = []
    monkeypatch.setattr(E.V2, "query_gpu", lambda: calls.append(1) or sample)
    assert E.query_gpu()["busy"] is True and calls == [1]


def test_busy_then_two_clear_samples_pass_and_are_all_preserved(tmp_path, monkeypatch):
    sleeps = patch_samples(monkeypatch, [transient(), clear(), clear()])
    receipt = E.snapshot(tmp_path, "after")
    assert receipt["busy"] is False and receipt["bounded_resample"]["accepted"] is True
    assert len(receipt["samples"]) == 3 and len(receipt["bounded_resample"]["identities"]) == 3
    assert sleeps == [2, 2]
    assert json.loads((tmp_path / "gpu_after.json").read_text(encoding="utf-8")) == receipt


@pytest.mark.parametrize("samples,identities", [
    ([transient(), clear(), clear()], [IDENTITY, {**IDENTITY, "ProcessId": 1321}]),
    ([transient(), clear(), clear()], [IDENTITY, {**IDENTITY, "CreationDate": "changed"}]),
    ([transient(), clear(), clear()], [IDENTITY, {**IDENTITY, "SessionId": 2}]),
    ([transient(engine_type="Copy"), clear(), clear()], None),
    ([transient(utilization=2), clear(), clear()], None),
    ([transient(reasons=[f"unapproved_active_engine:{PID}:3D"]), clear(), clear()], None),
    ([transient(), clear(memory=2535), clear()], None),
    ([transient(processes=[{"pid": PID}]), clear(), clear()], None),
    ([transient(), transient(), clear()], None),
])
def test_unregistered_or_changed_evidence_blocks(tmp_path, monkeypatch, samples, identities):
    patch_samples(monkeypatch, samples, identities)
    with pytest.raises(RuntimeError):
        E.snapshot(tmp_path, "after")
    receipt = json.loads((tmp_path / "gpu_after.json").read_text(encoding="utf-8"))
    assert receipt["bounded_resample"]["accepted"] is False


def test_followup_two_percent_compute_and_xml_pid_block(tmp_path, monkeypatch):
    bad = [
        clear(engines=[{"name": "pid_9_x_phys_0_eng_0_engtype_3D", "utilization_percent": 2,
                        "process_name": "chrome.exe"}]),
        {**clear(), "compute_process_count": 1},
        clear(processes=[{"pid": PID}]),
    ]
    for index, second in enumerate(bad):
        stage = tmp_path / str(index)
        stage.mkdir()
        patch_samples(monkeypatch, [transient(), second, clear()])
        with pytest.raises(RuntimeError):
            E.snapshot(stage, "after")


def test_initial_clear_is_one_sample_without_identity_query(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(E, "query_gpu", lambda: clear())
    monkeypatch.setattr(E, "process_identity", lambda pid: calls.append(pid))
    receipt = E.snapshot(tmp_path, "before")
    assert receipt["bounded_resample"]["reason"] == "initial_clear"
    assert len(receipt["samples"]) == 1 and calls == []


def test_busy_clear_clear_backend_runs_subprocess_once_and_restores_both_hooks(tmp_path, monkeypatch):
    original_query, original_snapshot = FROZEN._query_gpu, FROZEN._snapshot
    calls = []
    patch_samples(monkeypatch, [transient(), clear(), clear()])

    def fake(*args, **kwargs):
        assert FROZEN._query_gpu is E.query_gpu and FROZEN._snapshot is E.snapshot
        FROZEN._snapshot(tmp_path, "after")
        calls.append("subprocess")
        return {"done": True}

    monkeypatch.setattr(E.V1, "train", fake)
    assert E.train("stage") == {"done": True} and calls == ["subprocess"]
    assert FROZEN._query_gpu is original_query and FROZEN._snapshot is original_snapshot

    monkeypatch.setattr(E.V1, "train", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("failed")))
    with pytest.raises(RuntimeError):
        E.train("stage")
    assert FROZEN._query_gpu is original_query and FROZEN._snapshot is original_snapshot


def test_missing_before_identity_blocks_and_preserves_first_sample(tmp_path, monkeypatch):
    patch_samples(monkeypatch, [transient(), clear(), clear()])
    monkeypatch.setattr(E, "process_identity", lambda pid: (_ for _ in ()).throw(ValueError("missing")))
    with pytest.raises(RuntimeError):
        E.snapshot(tmp_path, "after")
    receipt = json.loads((tmp_path / "gpu_after.json").read_text(encoding="utf-8"))
    assert len(receipt["samples"]) == 1 and receipt["bounded_resample"]["accepted"] is False


def test_v2_sources_and_result_are_unchanged():
    for relative, expected in OLD_SHA.items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected
