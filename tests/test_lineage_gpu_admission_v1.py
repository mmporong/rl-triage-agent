from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "bench"), str(ROOT / "evals")]
G = importlib.import_module("lineage_gpu_admission_v1")


def _xml(rows: list[tuple[int, str, str]], *, memory: int = 1024, utilization: int = 0) -> str:
    processes = "".join(
        f"<process_info><pid>{pid}</pid><type>{kind}</type><process_name>{path}</process_name></process_info>"
        for pid, kind, path in rows
    )
    return (
        "<nvidia_smi_log><gpu><utilization>"
        f"<gpu_util>{utilization} %</gpu_util></utilization>"
        f"<fb_memory_usage><used>{memory} MiB</used></fb_memory_usage>"
        f"<processes>{processes}</processes></gpu></nvidia_smi_log>"
    )


def _windows(rows: list[tuple[int, str, str]], *, engines: list[dict] | None = None) -> dict:
    return {
        "processes": [
            {"ProcessId": pid, "Name": name, "ExecutablePath": path, "CommandLine": name}
            for pid, name, path in rows
        ],
        "engines": [] if engines is None else engines,
    }


def test_policy_changes_only_profile_and_registered_display_names() -> None:
    base, lineage = G.BASE.GPU_ADMISSION, G.policy()
    assert set(lineage) == set(base)
    assert lineage["policy"] == G.GPU_PROFILE and lineage["policy"] != base["policy"]
    added = set(lineage["allowed_display_process_names"]) - set(base["allowed_display_process_names"])
    assert added == set(G.EXTRA_DISPLAY_NAMES) == {
        "applicationframehost.exe", "systemsettings.exe", "codex-computer-use.exe"}
    for key in set(base) - {"policy", "allowed_display_process_names"}:
        assert lineage[key] == base[key]
    assert lineage["memory_used_limit_mib"] == base["memory_used_limit_mib"]
    assert lineage["allowed_process_types"] == base["allowed_process_types"]
    assert lineage["allowed_active_engine_types"] == base["allowed_active_engine_types"]
    assert lineage["unclassified_activity"] == base["unclassified_activity"] == "block"


def test_query_registers_temporary_profile_and_restores_base(monkeypatch: pytest.MonkeyPatch) -> None:
    media = G.BASE.V2.GPU
    original = G.BASE.query_gpu
    observed = []

    def fake_query(profile):
        observed.append((profile, media.GPU_ADMISSION_PROFILES[profile]))
        return {"admission_policy": profile, "busy": False}

    monkeypatch.setattr(media, "query_gpu", fake_query)
    assert G.query_gpu() == {"admission_policy": G.GPU_PROFILE, "busy": False}
    assert observed == [(G.GPU_PROFILE, G.policy())]
    assert G.GPU_PROFILE not in media.GPU_ADMISSION_PROFILES
    assert G.BASE.query_gpu is original


def test_profile_and_base_query_are_restored_after_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    media = G.BASE.V2.GPU
    original = G.BASE.query_gpu

    def fail(_profile):
        assert G.GPU_PROFILE in media.GPU_ADMISSION_PROFILES
        raise RuntimeError("fake query failed")

    monkeypatch.setattr(media, "query_gpu", fail)
    with pytest.raises(RuntimeError, match="fake query failed"):
        G.query_gpu()
    assert G.GPU_PROFILE not in media.GPU_ADMISSION_PROFILES
    assert G.BASE.query_gpu is original


def test_snapshot_uses_scoped_query_and_restores(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    media = G.BASE.V2.GPU
    original = G.BASE.query_gpu

    def fake_snapshot(stage, name):
        assert stage == tmp_path and name == "before"
        assert G.GPU_PROFILE in media.GPU_ADMISSION_PROFILES
        assert G.BASE.query_gpu is not original
        return G.BASE.query_gpu()

    monkeypatch.setattr(media, "query_gpu",
                        lambda profile: {"admission_policy": profile, "busy": False})
    monkeypatch.setattr(G.BASE, "snapshot", fake_snapshot)
    assert G.snapshot(tmp_path) == {"admission_policy": G.GPU_PROFILE, "busy": False}
    assert G.GPU_PROFILE not in media.GPU_ADMISSION_PROFILES
    assert G.BASE.query_gpu is original


def test_media_classifier_allows_two_os_windows_but_blocks_c_and_python_compute() -> None:
    media = G.BASE.V2.GPU
    display = [
        (101, "C+G", "C:/Windows/System32/ApplicationFrameHost.exe"),
        (102, "G", "C:/Windows/ImmersiveControlPanel/SystemSettings.exe"),
    ]
    windows = _windows([
        (101, "applicationframehost.exe", "C:/Windows/System32/ApplicationFrameHost.exe"),
        (102, "systemsettings.exe", "C:/Windows/ImmersiveControlPanel/SystemSettings.exe"),
    ])
    with G._scope():
        clear = media.classify_gpu(_xml(display), windows, G.GPU_PROFILE)
        c_type = media.classify_gpu(
            _xml([(102, "C", "C:/Windows/ImmersiveControlPanel/SystemSettings.exe")]),
            _windows([(102, "systemsettings.exe", "C:/Windows/ImmersiveControlPanel/SystemSettings.exe")]),
            G.GPU_PROFILE,
        )
        python_path = "C:/Python310/python.exe"
        python_windows = _windows(
            [(103, "python.exe", python_path)],
            engines=[{"Name": "pid_103_luid_0x0_phys_0_eng_0_engtype_Compute",
                      "UtilizationPercentage": 1}],
        )
        compute = media.classify_gpu(_xml([(103, "C", python_path)]), python_windows, G.GPU_PROFILE)
    assert clear["busy"] is False and clear["c_plus_g_contexts_present"] is True
    assert c_type["busy"] is True
    assert any(reason.startswith("unapproved_process_type:102:C") for reason in c_type["busy_reasons"])
    assert compute["busy"] is True and compute["compute_process_count"] == 1
    assert any("unapproved_active_engine" in reason for reason in compute["busy_reasons"])
