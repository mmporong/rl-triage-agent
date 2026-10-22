"""backend v2의 GPU 공유 프로필이 허용 화면 앱 네 개만 더하고 호출 뒤 원상태로 돌아가는지 검증한다."""
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "bench"), str(ROOT / "evals")]
import ppo_ratio_repair_execution_v2 as E  # noqa: E402
import r1_repair_execution as FROZEN  # noqa: E402
import record_evidence_media as GPU  # noqa: E402

XML = """<nvidia_smi_log><gpu><utilization><gpu_util>1 %</gpu_util></utilization>
<fb_memory_usage><used>2534 MiB</used></fb_memory_usage><processes>{rows}</processes></gpu></nvidia_smi_log>"""
ROW = "<process_info><pid>{pid}</pid><type>C+G</type><process_name>C:\\Apps\\{name}</process_name></process_info>"


def inventory(names):
    rows = "".join(ROW.format(pid=10 + i, name=name) for i, name in enumerate(names))
    processes = [{"ProcessId": 10 + i, "Name": name, "ExecutablePath": f"C:\\Apps\\{name}",
                  "CommandLine": f"\"C:\\Apps\\{name}\""} for i, name in enumerate(names)]
    return XML.format(rows=rows), {"processes": processes, "engines": []}


def run_query(monkeypatch, names):
    xml, windows = inventory(names)
    outputs = iter([xml, json.dumps(windows)])
    monkeypatch.setattr(GPU, "_gpu_command", lambda args: next(outputs))
    return E.query_gpu()


def test_profile_adds_only_four_display_names_and_keeps_every_other_rule():
    base = GPU.EXPECTED_GPU_ADMISSION
    assert set(E.GPU_ADMISSION["allowed_display_process_names"]) - set(base["allowed_display_process_names"]) == {
        "antigravity.exe", "claude.exe", "cursor.exe", "msedge.exe"}
    assert {k: v for k, v in E.GPU_ADMISSION.items() if k not in ("policy", "allowed_display_process_names")} == {
        k: v for k, v in base.items() if k not in ("policy", "allowed_display_process_names")}


def test_display_apps_pass_while_compute_and_unknown_processes_still_block(monkeypatch):
    clear = run_query(monkeypatch, ["claude.exe", "msedge.exe", "cursor.exe", "antigravity.exe", "chrome.exe"])
    assert clear["busy"] is False and clear["admission_policy"] == E.GPU_PROFILE
    blocked = run_query(monkeypatch, ["claude.exe", "python.exe"])
    assert blocked["busy"] is True and any("unapproved_gpu_process" in r for r in blocked["busy_reasons"])
    unknown = run_query(monkeypatch, ["game.exe"])
    assert unknown["busy"] is True
    assert E.GPU_PROFILE not in GPU.GPU_ADMISSION_PROFILES


def test_base_profile_still_rejects_the_same_display_apps(monkeypatch):
    xml, windows = inventory(["claude.exe", "msedge.exe", "cursor.exe"])
    assert GPU.classify_gpu(xml, windows, E.BASE_PROFILE)["busy"] is True


def test_backend_calls_use_the_profile_only_during_the_call(monkeypatch):
    original = FROZEN._query_gpu
    seen = []

    def fake(*args, **kwargs):
        seen.append(FROZEN._query_gpu is E.query_gpu)
        return {"done": True}

    for name in ("train", "inspect", "fixed_evaluate", "legacy_probe"):
        monkeypatch.setattr(E.V1, name, fake)
        assert getattr(E, name)("stage") == {"done": True}
    assert seen == [True] * 4 and FROZEN._query_gpu is original

    def fail(*args, **kwargs):
        raise RuntimeError("stage failed")

    monkeypatch.setattr(E.V1, "train", fail)
    with pytest.raises(RuntimeError):
        E.train("stage")
    assert FROZEN._query_gpu is original


def test_registered_profile_name_collision_is_refused(monkeypatch):
    monkeypatch.setitem(GPU.GPU_ADMISSION_PROFILES, E.GPU_PROFILE, E.GPU_ADMISSION)
    with pytest.raises(ValueError, match="이미 등록"):
        E.query_gpu()
