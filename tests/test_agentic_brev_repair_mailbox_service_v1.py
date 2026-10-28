"""Source identity and generated RPC syntax; actual CLI/network calls are absent."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from test_agentic_brev_repair_transport_v1 import S, T, host, world


def test_generated_python_and_bash_bodies_compile_without_execution(tmp_path):
    text = S.remote_script(["/venv/bin/python", "-I", "-B", "/repo/bench/agentic_brev_repair_transport_remote_v1.py", "--mode", "lookup"],
                           {"synthetic_cpu": True}, helper_sha256="a" * 64, timeout=120)
    python = text.split("<<'PY'\n", 1)[1].rsplit("\nPY\n", 1)[0]
    compile(python, "generated-repair-rpc.py", "exec")
    script = tmp_path / "generated-repair-rpc.sh"
    script.write_text(text, encoding="utf-8", newline="\n")
    bash = Path("C:/Program Files/Git/bin/bash.exe")
    if not bash.is_file():
        bash = Path("/bin/bash")
    assert bash.is_file(), "bash-n requires an installed native parser, never WSL"
    result = subprocess.run([str(bash), "--noprofile", "--norc", "-n", str(script)], capture_output=True, check=False)
    assert result.returncode == 0, result.stderr.decode(errors="replace")


def test_default_service_production_identity_and_foreign_rpc_injection_guard():
    service = object.__new__(S.BrevRepairMailboxService)
    service.allow_fixture, service.fixture_rpc = False, None
    service.production_identity()
    service.fixture_rpc = lambda *_: {"status": "completed"}
    with pytest.raises(ValueError, match="production CLI"):
        service.production_identity()


@pytest.mark.parametrize("bad", ["stop", "deadline", "source"])
def test_stop_deadline_or_source_mutation_blocks_before_remote_admission(tmp_path, bad):
    data = world(tmp_path)
    if bad == "stop":
        Path(data.config["lifecycle_stop_intent_path"]).write_text("{}", encoding="utf-8")
    elif bad == "deadline":
        from datetime import timedelta

        data.engine.remote.clock += timedelta(hours=2)
    else:
        copied = data.repo / T.REMOTE_SOURCE
        copied.write_bytes(copied.read_bytes() + b"\n# private fixture mutation\n")
    with pytest.raises(ValueError), host(data) as controller:
        authorization = controller.approve(data.operation, data.intent)["authorization"]
        controller.dispatch(data.operation, data.intent, authorization)
    assert data.calls == [] and data.engine.methods == [] and not data.engine.owner_live


def test_forged_owner_release_is_never_exposed_as_completed(tmp_path):
    data = world(tmp_path)

    def corrupt(response):
        if response["mode"] == "execute":
            response["state"]["owner_current"] = True

    data.engine.bad_reply = corrupt
    with host(data) as controller:
        authorization = controller.approve(data.operation, data.intent)["authorization"]
        with pytest.raises(ValueError, match="not terminated"):
            controller.dispatch(data.operation, data.intent, authorization)
    assert data.calls == ["train", "evaluate"]
    assert not list(data.runtime.folder.rglob("repair-seal.json"))
