"""OpenShell 샌드박스 안에서만 실행하는 커널 경계 테스트(policies/triage_agent.yaml 적용 상태).

호스트에서는 건너뛴다. 샌드박스에서: OPENSHELL_SANDBOX=1 pytest tests/sandbox
각 테스트는 에이전트가 run_analysis로 실행할 수 있는 코드가 실제로 막히는지 확인한다.
"""
import os
import socket
import urllib.request

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("OPENSHELL_SANDBOX") != "1", reason="OpenShell 샌드박스 전용")
WS = "/sandbox/workspace"


def test_cannot_write_reference_params():
    with pytest.raises(PermissionError):
        open(f"{WS}/reference/params/env.yaml", "a").write("# tampered\n")


def test_cannot_write_case_telemetry():
    with pytest.raises(PermissionError):
        open(f"{WS}/cases/c01/telemetry.json", "a").write("x")


def test_can_write_scratch_only():
    p = f"{WS}/scratch/ok.txt"
    open(p, "w").write("ok")
    assert os.path.exists(p)


def test_can_write_diagnosis():
    from rl_triage import triage_tools

    result = triage_tools.write_diagnosis("c01", ["physics"], "boundary fixture", "fixture check")
    assert result["saved"] == "diagnoses/c01.json"
    assert os.path.isfile(f"{WS}/diagnoses/c01.json")


def test_cannot_write_workspace_root():
    with pytest.raises(PermissionError):
        open(f"{WS}/unregistered-output.json", "w").write("{}")


def test_cannot_read_outside_policy():
    with pytest.raises(PermissionError):
        os.listdir("/home")


def test_exfiltration_host_blocked():
    with pytest.raises(Exception):
        urllib.request.urlopen("https://huggingface.co", timeout=10)


def test_cloud_metadata_blocked():
    with pytest.raises(OSError):
        socket.create_connection(("169.254.169.254", 80), timeout=5)
