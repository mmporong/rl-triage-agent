"""Unicode live 조회·원 SHA·Windows 자원 경계의 CPU 회귀."""
# pyright: reportMissingImports=false
from __future__ import annotations

import json
import sys
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "bench")]
import vision_gpu_admission_v1 as G

NOW = datetime(2026, 10, 25, tzinfo=timezone.utc)  # noqa: UP017


def _xml(process: str = "") -> bytes:
    return ("<nvidia_smi_log><gpu><uuid>GPU-fixture</uuid><utilization><gpu_util>0 %</gpu_util></utilization>"
            "<fb_memory_usage><used>10 MiB</used></fb_memory_usage><processes>" + process
            + "</processes></gpu></nvidia_smi_log>").encode("utf-8")


def _query(*, xml=None, cim=None, clock=lambda: NOW):
    raw_xml = _xml() if xml is None else xml
    raw_cim = {"processes": [], "engines": []} if cim is None else cim
    observed = []
    def runner(arguments):
        observed.append(arguments)
        return raw_xml if arguments[0] == "nvidia-smi" else json.dumps(raw_cim, ensure_ascii=False).encode("utf-8")
    return G.query_windows(run=runner, clock=clock), observed


def _check(ref, *, at=NOW, kind="local_windows", uuid="GPU-fixture"):
    return G.verify_fresh(ref["snapshot"], ref["sha256"], expected_gpu_uuid=uuid, resource_kind=kind, now=at)


def test_utf8_preserves_korean_and_keeps_frozen_workload_block() -> None:
    process = "<process_info><pid>42</pid><type>C+G</type><process_name>kit.exe</process_name></process_info>"
    cim = {"processes": [{"ProcessId": 42, "Name": "kit.exe", "ExecutablePath": "kit.exe",
                           "CommandLine": "kit.exe --이름 보행 실험"}], "engines": []}
    ref, calls = _query(xml=_xml(process), cim=cim)
    assert "보행 실험" in ref["snapshot"]["raw"]["cim_utf8"]
    assert "[Console]::OutputEncoding" in calls[1][-1]
    with pytest.raises(ValueError, match="사용 중"):
        _check(ref)


def test_clear_local_query_cannot_grant_lease_or_become_brev_admission() -> None:
    ref, _ = _query()
    assert _check(ref)["lease_granted"] is False
    for kind, uuid in (("brev_remote", "GPU-fixture"), ("local_windows", "GPU-other")):
        with pytest.raises(ValueError, match="자원"):
            _check(ref, kind=kind, uuid=uuid)


def test_saved_raw_cannot_be_restamped_as_fresh_and_no_caller_timestamp_argument() -> None:
    ref, _ = _query()
    with pytest.raises(ValueError, match="오래됐거나 미래"):
        _check(ref, at=NOW + timedelta(days=100))
    changed = deepcopy(ref)
    changed["snapshot"]["capture_started_utc"] = (NOW + timedelta(days=100)).isoformat()
    changed["snapshot"]["capture_finished_utc"] = changed["snapshot"]["capture_started_utc"]
    with pytest.raises(ValueError, match="원 SHA"):
        _check(changed, at=NOW + timedelta(days=100))
    with pytest.raises(TypeError):
        G.query_windows(observed_at=NOW.isoformat())  # pyright: ignore[reportCallIssue]


def test_unknown_raw_non_utf8_or_duplicate_cim_key_fail_closed() -> None:
    for xml in (b"{", b"\xff\xfe", b"", _xml().replace(b"<processes></processes>", b"")):
        with pytest.raises((ValueError, UnicodeError, G.ET.ParseError, G.FROZEN.PreflightError)):
            _query(xml=xml)
    with pytest.raises(ValueError, match="중복"):
        G.query_windows(run=lambda args: _xml() if args[0] == "nvidia-smi" else b'{"processes":[],"processes":[],"engines":[]}', clock=lambda: NOW)


def test_clock_window_and_changed_snapshot_are_rejected() -> None:
    times = iter([NOW, NOW + timedelta(seconds=31)])
    with pytest.raises(ValueError, match="30초"):
        _query(clock=lambda: next(times))
    ref, _ = _query()
    for delta in (-1, 31):
        with pytest.raises(ValueError, match="오래됐거나 미래"):
            _check(ref, at=NOW + timedelta(seconds=delta))
    changed = deepcopy(ref)
    changed["snapshot"]["classified"]["busy"] = True
    with pytest.raises(ValueError, match="원 SHA"):
        _check(changed)


def test_unknown_or_duplicate_process_inventory_is_not_an_empty_clear_table() -> None:
    for body in ("query failed", "N/A", "<unexpected/>", "<process_info/>query failed"):
        xml = _xml().replace(b"<processes></processes>", f"<processes>{body}</processes>".encode())
        with pytest.raises(ValueError, match="알 수 없는"):
            _query(xml=xml)
    xml = _xml().replace(b"</gpu>", b"<processes></processes></gpu>")
    with pytest.raises(ValueError, match="중복"):
        _query(xml=xml)


def test_capture_finished_in_future_is_not_fresh_even_if_started_in_past() -> None:
    times = iter([NOW - timedelta(seconds=1), NOW + timedelta(seconds=29)])
    ref, _ = _query(clock=lambda: next(times))
    with pytest.raises(ValueError, match="오래됐거나 미래"):
        _check(ref)
