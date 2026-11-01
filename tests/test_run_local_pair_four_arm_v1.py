# pyright: reportMissingImports=false
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("bench", "evals", "src")]
import run_local_pair_four_arm_v1 as B


def test_worker_uses_existing_ubuntu_fixed_python_and_exact_paths(tmp_path: Path):
    root = tmp_path / "a path" / "nim"
    call, protocol = root / "calls/0001", ROOT / "bench/protocols/example.json"
    argv = B.worker_command(root, call, protocol, "a" * 40)
    assert argv[:6] == ["wsl.exe", "--distribution", "Ubuntu", "--exec", "bash", "-lc"]
    assert B.mounted(root) in argv[-1] and B.mounted(call) in argv[-1]
    assert '"$HOME/.config/nvidia/env"' in argv[-1]
    assert '"$HOME/.venvs/rl-triage/bin/python"' in argv[-1]


@pytest.mark.parametrize("identity", [{"pid": 1, "proc_start_ticks": "123"},
                                      {"pid": 99, "proc_start_ticks": "bad"}, {}])
def test_cleanup_rejects_missing_or_system_process_identity(identity):
    with pytest.raises(ValueError, match="신원"):
        B.worker_cleanup_command(identity)


def test_preflight_rejects_changed_private_config_before_source_or_lease(tmp_path: Path, monkeypatch):
    config = tmp_path / "config.json"
    B.M.write(config, {"schema": "changed"})
    monkeypatch.setattr(B.M, "verify_protocol", lambda *_: {
        "sources_lf": {x: "1" * 64 for x in B.SOURCE_FILES}, "execution_config_sha256": "0" * 64})
    monkeypatch.setattr(B.NIM, "_verify_protocol", lambda *_: {})
    with pytest.raises(ValueError, match="config SHA"):
        B.preflight(tmp_path / "protocol.json", "a" * 40, config)


def test_one_shot_parent_collects_process_exit_and_raw_response(tmp_path: Path, monkeypatch):
    call = tmp_path / "nim/calls/0001"
    call.mkdir(parents=True)
    identity = {"pid": 4200, "proc_start_ticks": "999", "wsl": True}
    raw = {"live": False, "http_status": 200, "provider_response": {"fixture": True}}
    class Child:
        pid = 12345
        returncode = 0
        def __enter__(self): return self
        def __exit__(self, *_args): return False
        def communicate(self, timeout):
            B.M.write(call / "worker-process.json", identity)
            B.M.write(call / "worker-exit.json", {"identity": identity, "exit_code": 0})
            B.M.write(call / "worker-response.json", raw)
            return "CPU 한글 완료".encode(), b""
    monkeypatch.setattr(B.subprocess, "Popen", lambda *_a, **_kw: Child())
    result = B._worker_send(tmp_path / "nim", ROOT / "bench/protocols/example.json",
                            "a" * 40, {}, call)
    assert result == raw and B.M.read(call / "parent-exit.json")["exit_code"] == 0
    assert (call / "parent-stdout.txt").read_bytes() == "CPU 한글 완료".encode()
    assert B.M.read(call / "parent-process.json")["pid"] == 12345


def test_cli_import_and_help_finish_without_sdk_or_provider(tmp_path: Path):
    completed = subprocess.run([sys.executable, "-B", str(ROOT / "bench/run_local_pair_four_arm_v1.py"), "--help"],
        cwd=ROOT, capture_output=True, encoding="utf-8", errors="replace", check=False, timeout=30)
    assert completed.returncode == 0, completed.stderr
    assert "--preflight" in completed.stdout and "--anchor" in completed.stdout


def test_public_lease_metadata_uses_the_existing_canonical_normalizer():
    raw = {"leaseId": "30c23b3a-f098-4002-b22c-bb9dddae3ec4",
           "owner": B.R.SOURCE.OWNER, "state": "running", "completedUtc": None}
    normalized = B.R2.normalize_lease_metadata(raw)
    assert normalized["lease_id"] == raw["leaseId"] and "lease_id" not in raw
    with pytest.raises(ValueError, match="충돌"):
        B.R2.normalize_lease_metadata({**raw, "lease_id": "different"})
