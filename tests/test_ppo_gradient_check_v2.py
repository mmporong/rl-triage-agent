"""실제 Windows owned launcher로 v2 launch 소유권 회귀를 검사한다."""
from __future__ import annotations

from copy import deepcopy
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "bench")]
import ppo_gradient_check_v2 as Q
import ppo_gradient_training_v2 as T


@pytest.fixture
def cpu_command(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    monkeypatch.setenv("PYTHONUTF8", "1")
    monkeypatch.setenv("PYTHONIOENCODING", "utf-8")
    child = tmp_path / "cpu child.py"
    child.write_text("""import json, os, pathlib, sys
assert os.environ['CUDA_VISIBLE_DEVICES'] == '-1'
with pathlib.Path(sys.argv[1]).open('x', encoding='utf-8') as out:
    json.dump({'pid': os.getpid(), 'cwd': os.getcwd(), 'argument': sys.argv[2], 'gpu': False}, out)
print('CPU child 완료', flush=True)
""", encoding="utf-8")
    def command(folder, isaac=False):
        launcher = Path.home() / "IsaacLab/_isaac_sim/python.bat" if isaac else Path(sys.executable)
        assert launcher.is_file()
        return [str(launcher), "-B", str(child), str(folder / "cpu_receipt.json"), "공백 포함 인자"]
    return command


def assert_owned_execution(folder, execution):
    Q.OLD.verify_execution(execution)
    assert execution["owned_process_identities"]
    assert Q.read(folder / "launch_intent.json") == Q.read(folder / "launch.json")
    receipt = Q.read(folder / "cpu_receipt.json")
    assert receipt["gpu"] is False and receipt["argument"] == "공백 포함 인자"
    assert Path(receipt["cwd"]).resolve() == folder.resolve()
    # 빠르게 끝난 child PID는 sampling에 안 잡힐 수 있으나 Job 자체는 비어 있어야 한다.
    assert execution["cleanup_verified"] and execution["natural_exit"]
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    for pid, creation in execution["owned_process_identities"].items():
        if Q.OLD.OWNED._process_creation_filetime(int(pid)) != creation:
            continue
        handle = kernel.OpenProcess(0x100000, False, int(pid))
        if handle:
            try:
                # 종료된 프로세스 객체도 다른 핸들이 남아 있으면 creation time을 읽을 수 있다.
                assert kernel.WaitForSingleObject(handle, 0) == 0
            finally:
                kernel.CloseHandle(handle)


def test_real_run_owned_reproduces_v1_collision_before_child_starts(tmp_path, cpu_command):
    folder = tmp_path / "v1"
    folder.mkdir()
    args = cpu_command(folder)
    original = {"args": args, "cwd": str(folder)}
    Q.write_new(folder / "launch.json", original)
    with pytest.raises(FileExistsError):
        Q.OLD.OWNED.run_owned(folder, args, folder, 15)
    assert Q.read(folder / "launch.json") == original
    assert not (folder / "cpu_receipt.json").exists()
    assert not (folder / "stdout.log").exists()
    assert not (folder / "stderr.log").exists()


@pytest.mark.parametrize("isaac", [False, True])
def test_v2_real_owned_launcher_with_existing_controller_metadata(tmp_path, cpu_command, isaac):
    folder = tmp_path / "v2 arm"
    folder.mkdir()
    Q.write_new(folder / "manifest.json", {"fixture": "cpu_only"})
    Q.write_new(folder / "gpu_before.json", {"fixture": "not_queried"})
    execution = Q.execute_owned(folder, cpu_command(folder, isaac), folder, 30)
    assert_owned_execution(folder, execution)
    before = {path.name: path.read_bytes() for path in folder.iterdir()}
    with pytest.raises(FileExistsError):
        Q.execute_owned(folder, cpu_command(folder, isaac), folder, 30)
    assert before == {path.name: path.read_bytes() for path in folder.iterdir()}


@pytest.mark.parametrize("name", ["launch.json", "stdout.log", "stderr.log", "execution.json", "launch_intent.json"])
def test_reserved_file_collision_is_rejected_without_touching_folder(tmp_path, cpu_command, name):
    (tmp_path / name).write_bytes(b"preserved")
    with pytest.raises(FileExistsError):
        Q.execute_owned(tmp_path, cpu_command(tmp_path), tmp_path, 15)
    assert (tmp_path / name).read_bytes() == b"preserved"
    assert not (tmp_path / "cpu_receipt.json").exists()


def test_missing_folder_is_rejected_before_owned_launch(tmp_path, cpu_command):
    folder = tmp_path / "missing"
    with pytest.raises(FileNotFoundError):
        Q.execute_owned(folder, cpu_command(folder), tmp_path, 15)
    assert not folder.exists()


def test_nonzero_real_child_preserves_failure_receipt_and_stops(tmp_path):
    with pytest.raises(RuntimeError):
        Q.execute_owned(tmp_path, [sys.executable, "-B", "-c", "raise SystemExit(7)"], tmp_path, 15)
    execution = Q.read(tmp_path / "execution.json")
    assert execution["exit_code"] == 7 and execution["cleanup_verified"]
    assert Q.read(tmp_path / "launch_intent.json") == Q.read(tmp_path / "launch.json")


@pytest.fixture
def bound(tmp_path, monkeypatch):
    monkeypatch.setattr(Q, "ROOT", tmp_path)
    monkeypatch.setattr(Q, "PRIVATE", tmp_path / "private")
    monkeypatch.setattr(Q, "PROTOCOL", tmp_path / "bench/protocols/fixture_v2.json")
    protocol = Q.fixed_protocol() | {"source_sha256_lf": {"evals/ppo_gradient_check_v2.py": "c" * 64}}
    Q.write_new(Q.PROTOCOL, protocol)
    monkeypatch.setattr(Q, "verify", lambda *_: (protocol, "a" * 40))
    monkeypatch.setattr(Q, "lease_context", lambda *_: {"leaseId": "fixture", "childPid": os.getpid()})
    monkeypatch.setattr(Q.SEED, "assert_fresh_seeds", lambda seeds, _: {"status": "fresh", "seeds": list(seeds)})
    return protocol, "a" * 40


@pytest.mark.parametrize("child_failure", [False, True])
def test_controller_with_real_run_owned_completes_or_stops_on_child_failure(bound, tmp_path, monkeypatch, cpu_command, child_failure):
    # GPU/source/measurements는 fixture다. execute_owned, run_owned, WindowsJob, Popen은 실제다.
    monkeypatch.setattr(Q.OLD.GPU, "query_gpu", lambda *_: {"busy": False, "processes": []})
    monkeypatch.setattr(Q.OLD, "validate_gpu_snapshot", lambda x: x)
    launched = []
    def training_args(_lab, folder, *_):
        launched.append(folder.name)
        if child_failure and len(launched) == 2:
            return [sys.executable, "-B", "-c", "raise SystemExit(7)"]
        return cpu_command(folder)
    monkeypatch.setattr(Q, "training_args", training_args)
    received = []
    def inspect(folder, spec, *_):
        execution = Q.read(folder / "execution.json")
        Q.OLD.verify_execution(execution)
        assert (folder / "cpu_receipt.json").is_file()
        assert Q.read(folder / "launch_intent.json") == Q.read(folder / "launch.json")
        received.append(spec["id"])
        status = "abnormal" if spec["arm"] == "INVERSE_RATIO" else "normal"
        return spec | {"state": {"fixture": "equal"}, "losses": [0.] * 3, "update_wall_s": [1.] * 3,
            "inspection": {"status": status, "primitives": {
                "optimizer_step_count": {"status": "normal"}, "minibatch_count": {"status": "normal"},
                Q.G.PRIMITIVE: {"status": status}}} if spec["observed"] else None}
    monkeypatch.setattr(Q, "inspect_run", inspect)
    target = Q.PRIVATE / "runs/ppo_gradient_v2_20261017"
    if child_failure:
        with pytest.raises(RuntimeError):
            Q.run("ppo_gradient_v2_20261017", tmp_path)
        result = Q.read(target / "result.json")
        assert result["status"] == "invalid" and len(result["runs"]) == 1
        assert len(launched) == 2 and len(received) == 1
        failed = target / Q.specs()[1]["id"]
        assert Q.read(failed / "execution.json")["exit_code"] == 7
    else:
        result = Q.run("ppo_gradient_v2_20261017", tmp_path)
        assert result["outcome"] == "confirmed" and result["completed_runs"] == 8
        assert received == [spec["id"] for spec in Q.specs()]
    assert len(list(target.glob("*/launch.json"))) == len(launched)
    for relative, digest in result["artifact_sha256_lf"].items():
        assert Q.sha(target / relative) == digest
    with pytest.raises(FileExistsError):
        Q.run("ppo_gradient_v2_20261017", tmp_path)
    assert len(received) == (1 if child_failure else 8)


def test_training_requires_v2_intent_and_actual_launch_agree(bound):
    protocol, anchor = bound
    lab = Path("runtime").resolve()
    spec = Q.specs()[0]
    tag = "ppo_gradient_v2_20261017"
    folder = Q.PRIVATE / "runs" / tag / spec["id"]
    folder.mkdir(parents=True)
    fixed = Q.manifest(spec, protocol, anchor)
    args = Q.training_args(lab, folder, spec, tag)
    expected = {"args": args, "cwd": str(lab)}
    Q.write_new(folder / "launch_intent.json", expected)
    Q.write_new(folder / "launch.json", expected)
    Q.write_new(folder.parent / "plan.json", {"lease": {"childPid": os.getpid(), "leaseId": "fixture"}})
    assert T.verify_training(fixed, folder, args[args.index("--") + 1:], lab) == spec
    altered = deepcopy(expected)
    altered["args"][-1] = "another_run"
    (folder / "launch.json").write_text(json.dumps(altered), encoding="utf-8")
    with pytest.raises(ValueError):
        T.verify_training(fixed, folder, args[args.index("--") + 1:], lab)


def test_v2_keeps_v1_measurement_contract_but_uses_fresh_cohort():
    old, new = Q.V1.fixed_protocol(), Q.fixed_protocol()
    changed = {key for key in old if old[key] != new[key]}
    assert changed == {"schema", "seeds", "runs"}
    assert set(new) - set(old) == {"supersedes", "launch_contract"}
    assert set(Q.SEEDS).isdisjoint(Q.V1.SEEDS)
