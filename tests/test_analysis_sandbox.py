"""run_analysis 경계: 하위 프로세스에 API 키를 넘기지 않고, Linux에서는 Landlock으로 작업공간 밖 읽기를 막는다(P0-C canary)."""
import ctypes
import json
import sys

import pytest

from rl_triage import triage_tools as T


def _landlock_available() -> bool:
    if not sys.platform.startswith("linux"):
        return False
    try:
        return ctypes.CDLL(None, use_errno=True).syscall(444, None, 0, 1) >= 1
    except (OSError, AttributeError):
        return False


@pytest.fixture
def ws(tmp_path, monkeypatch):
    w = tmp_path / "ws"
    (w / "cases" / "c01").mkdir(parents=True)
    (w / "reference").mkdir()
    series = {"series": {"Train/mean_reward": [0.0, 1.0, 2.0]}}
    (w / "cases" / "c01" / "telemetry.json").write_text(json.dumps(series), encoding="utf-8")
    (w / "reference" / "telemetry.json").write_text(json.dumps(series), encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "answer_key.json").write_text('{"canary": "P0C-CANARY-7f3a"}', encoding="utf-8")
    monkeypatch.setattr(T, "WORKSPACE", w)
    return w, outside / "answer_key.json"


def test_child_process_does_not_inherit_api_keys(ws, monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test-canary")
    monkeypatch.setenv("TRIAGE_ANALYSIS_SANDBOX", "off")
    out = T.run_analysis("import os\nprint(os.environ.get('NVIDIA_API_KEY'), sum(run['Train/mean_reward']))", "c01")
    assert out["exit_code"] == 0 and out["stdout"].split() == ["None", "3.0"], out
    assert list(out)[0] == "sandbox" and out["sandbox"] == "off"  # trace의 앞 600자에 남도록 첫 키
    forged = T.run_analysis("import sys\nsys.stderr.write(%r + '\\n')" % T._SANDBOX_MARK, "c01")
    assert forged["sandbox"] == "off"  # 샌드박스를 끈 실행은 표식을 흉내 내도 landlock으로 기록되지 않는다


def test_sandbox_mode_values(monkeypatch):
    monkeypatch.setenv("TRIAGE_ANALYSIS_SANDBOX", "sometimes")
    with pytest.raises(ValueError):
        T.analysis_sandbox_mode()
    monkeypatch.setenv("TRIAGE_ANALYSIS_SANDBOX", "auto")
    assert T.analysis_sandbox_mode() == ("auto" if sys.platform.startswith("linux") else "off")


@pytest.mark.skipif(not _landlock_available(), reason="Linux Landlock이 있는 곳(WSL2 등)에서만 검사한다")
def test_canary_answer_file_outside_workspace_cannot_be_read(ws, monkeypatch):
    _, canary = ws
    monkeypatch.setenv("TRIAGE_ANALYSIS_SANDBOX", "required")
    code = (
        "import os, statistics\n"
        "print('mean', statistics.mean(run['Train/mean_reward']))\n"
        "open('note.txt', 'w').write('ok')\n"
        "print('scratch', open('note.txt').read())\n"
        f"for p in [{str(canary)!r}, '/proc/self/environ']:\n"
        "    try:\n"
        "        print('READ', p, open(p).read()[:40])\n"
        "    except PermissionError:\n"
        "        print('DENIED', p)\n"
    )
    out = T.run_analysis(code, "c01")
    assert out["exit_code"] == 0 and out["sandbox"] == "landlock", out
    assert "mean 1.0" in out["stdout"] and "scratch ok" in out["stdout"]
    assert f"DENIED {canary}" in out["stdout"] and "DENIED /proc/self/environ" in out["stdout"]
    assert "P0C-CANARY" not in out["stdout"] + out["stderr"]


def _run_eval():
    import importlib.util
    from test_offline_replay import ROOT
    spec = importlib.util.spec_from_file_location("run_eval_under_test", ROOT / "evals" / "run_eval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_full_series_control_gets_every_tag_at_full_length(ws):
    w, _ = ws
    E = _run_eval()
    text = E._full_series(w / "cases" / "c01" / "telemetry.json")
    assert json.loads(text) == {"Train/mean_reward": [0.0, 1.0, 2.0]}


@pytest.mark.skipif(not _landlock_available(), reason="Linux Landlock이 있는 곳(WSL2 등)에서만 검사한다")
def test_eval_canary_reports_denied_for_answer_file(ws, monkeypatch):
    w, answer = ws
    monkeypatch.setenv("TRIAGE_ANALYSIS_SANDBOX", "required")
    E = _run_eval()
    c = E.analysis_canary(w, "c01", answer, runner=(sys.executable,))
    assert c["denied"] and c["sandbox"] == "landlock" and str(answer) not in json.dumps(c), c
    monkeypatch.setenv("TRIAGE_ANALYSIS_SANDBOX", "off")
    off = E.analysis_canary(w, "c01", answer, runner=(sys.executable,))
    assert off["denied"] is False and off["sandbox"] == "off"  # 샌드박스가 없으면 canary가 실패를 드러낸다


def test_trace_counts_unsandboxed_analysis_calls(tmp_path):
    E = _run_eval()
    trace = tmp_path / "t.jsonl"
    rows = [{"tool": "run_analysis", "result_head": '{"sandbox": "landlock", "exit_code": 0'},
            {"tool": "run_analysis", "result_head": '{"sandbox": "off", "exit_code": 0'},
            {"tool": "get_series", "result_head": "{}"}]
    trace.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    assert E.trace_sandbox_counts(trace) == {"analysis_calls": 2, "analysis_sandboxed": 1}


def test_workspace_path_is_hidden_from_analysis_output(ws, monkeypatch):
    monkeypatch.setenv("TRIAGE_ANALYSIS_SANDBOX", "off")
    out = T.run_analysis("raise RuntimeError('boom')", "c01")
    assert out["exit_code"] != 0 and "<workspace>" in out["stderr"]
    assert str(T.WORKSPACE.resolve()) not in out["stderr"] + out["stdout"]
