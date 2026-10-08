"""P0-A offline replay: 공개 자료만으로 과거 점수 재계산·작업공간 생성·정답 유출 경계를 검사한다.

replay와 작업공간 생성은 모델 클라이언트·NAT·torch import와 네트워크 연결을 막은 하위 프로세스에서 돌린다.
"""
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from rl_triage import leakcheck as L
from rl_triage import triage_tools as T

ROOT = Path(__file__).resolve().parents[1]
KEY = json.loads((ROOT / "bench" / "answer_key.json").read_text(encoding="utf-8"))
CASES = json.loads((ROOT / "bench" / "cases.json").read_text(encoding="utf-8"))
MANIFEST = json.loads((ROOT / "bench" / "reference" / "manifest.json").read_text(encoding="utf-8"))
USER_PATH = re.compile(r"[A-Za-z]:[\\/]+Users[\\/]|/home/[^/\s]+/|/Users/[^/\s]+/|/mnt/[a-z]/Users/")

# (전체 행, 인프라 행, task, {mode: (칸, top-1, top-2)}). 앞의 두 폴더가 docs/IMPLEMENTATION-ORDER.md 2절 G 표이고,
# blind_v1_1_dev_truncated(G절의 blind 비인프라 65행 중 5행)와 heldout_changes(2절 B의 10/10 대 10/10)는 추가 검증이다.
EXPECTED = {
    "evals/results/blind_v1": (47, 7, "blind", {"agent": (20, 10, 12), "control": (20, 8, 10)}),
    "evals/results/heldout_blind": (22, 2, "blind", {"agent": (10, 6, 6), "control": (10, 2, 6)}),
    "evals/results/blind_v1_1_dev_truncated": (5, 0, "blind", {"agent": (5, 0, 0)}),
    "evals/results/heldout_changes": (22, 2, "changes", {"agent": (10, 10, 10), "control": (10, 10, 10)}),
}

OFFLINE_RUNNER = r"""
import runpy, socket, sys
BLOCKED = {"openai", "nat", "langchain", "langchain_core", "langchain_openai", "torch", "httpx", "requests"}
class _Block:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ImportError(f"blocked import in offline run: {name}")
        return None
sys.meta_path.insert(0, _Block())
def _no_net(*a, **k):
    raise OSError("network disabled in offline run")
for attr in ("connect", "connect_ex", "sendto", "sendmsg"):
    setattr(socket.socket, attr, _no_net)
socket.getaddrinfo = socket.create_connection = _no_net
script = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name="__main__")
"""


def _offline(script: str, *args) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k not in ("NVIDIA_API_KEY", "OPENAI_API_KEY")}
    env.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8", CUDA_VISIBLE_DEVICES="")
    return subprocess.run([sys.executable, "-c", OFFLINE_RUNNER, script, *map(str, args)], cwd=ROOT, env=env,
                          capture_output=True, text=True, encoding="utf-8", timeout=300)


def _replay(*args) -> subprocess.CompletedProcess:
    return _offline("evals/replay.py", *args)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------- 재채점 ----------

def test_replay_reproduces_saved_scores_without_model_network_or_gpu():
    proc = _replay(*EXPECTED, "--traces", "evals/results/traces", "--json")
    assert proc.returncode == 0, proc.stderr
    assert "blocked import" not in proc.stderr
    report = json.loads(proc.stdout)
    assert report["status"] == "pass"
    by_path = {f["path"]: f for f in report["inputs"]["results"]}
    for path, (rows, infra, task, modes) in EXPECTED.items():
        f = by_path[path]
        assert (f["rows"], f["infra_rows"], f["mismatches"]) == (rows, infra, [])
        for mode, want in modes.items():
            m = f["tasks"][task][mode]
            assert (m["cells"], m["top1"], m["top2"]) == want, (path, mode)
    assert report["inputs"]["answer_key"]["sha256_lf"] == MANIFEST["w0_check"]["answer_key"]["sha256_clean_checkout_lf"]
    assert report["trace_scan"]["files"] >= 50 and report["trace_scan"]["hits"] == []
    assert report["environment"]["model_calls"] == 0
    assert not USER_PATH.search(proc.stdout)
    assert str(ROOT) not in proc.stdout and ROOT.as_posix() not in proc.stdout


def _blind_row(case_id="c01", ranking=("termination", "reward"), infra=False, **kw):
    ranking = list(ranking)
    truth = KEY["cases"].get(case_id, {}).get("category", "termination")
    row = {"case_id": case_id, "mode": "agent", "seed": 42, "task": "blind", "bench": "v1", "attempt": 1,
           "infra_error": infra, "ranking": ranking, "truth": truth,
           "correct": bool(ranking) and ranking[0] == truth, "top2": truth in ranking[:2]}
    row.update(kw)
    return row


def _write(tmp_path, *rows, raw=None) -> Path:
    d = tmp_path / "results"
    d.mkdir(exist_ok=True)
    text = raw if raw is not None else "".join(json.dumps(r) + "\n" for r in rows)
    (d / "seed42.jsonl").write_text(text, encoding="utf-8")
    return d


def _agent_cell(proc):
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)["inputs"]["results"][0]["tasks"]["blind"]["agent"]


def test_empty_ranking_counts_as_wrong(tmp_path):
    m = _agent_cell(_replay(_write(tmp_path, _blind_row(ranking=[])), "--json"))
    assert (m["cells"], m["top1"], m["top2"]) == (1, 0, 0)


def test_infra_only_cell_counts_as_wrong(tmp_path):
    m = _agent_cell(_replay(_write(tmp_path, _blind_row(ranking=[], infra=True)), "--json"))
    assert (m["cells"], m["top1"], m["infra_only_cells"]) == (1, 0, ["s42/c01"])


def test_last_non_infra_row_decides_the_cell(tmp_path):
    rows = [_blind_row(ranking=["reward", "termination"]), _blind_row(ranking=["termination"], attempt=2),
            _blind_row(ranking=[], infra=True, attempt=3)]
    m = _agent_cell(_replay(_write(tmp_path, *rows), "--json"))
    assert (m["cells"], m["top1"], m["top2"]) == (1, 1, 1)


def test_stored_score_that_disagrees_with_answer_key_fails(tmp_path):
    proc = _replay(_write(tmp_path, _blind_row(correct=False)))
    assert proc.returncode == 1 and "MISMATCH" in proc.stderr and "correct" in proc.stderr


@pytest.mark.parametrize("make, message", [
    (lambda t: t / "missing", "없다"),
    (lambda t: (t / "empty").mkdir() or t / "empty", "jsonl 결과 파일이 없다"),
    (lambda t: _write(t, raw='{"case_id": "c01"\n'), "잘못된 JSON"),
    (lambda t: _write(t, _blind_row(case_id="c99")), "정답표에 없는 case_id"),
    (lambda t: _write(t, {k: v for k, v in _blind_row().items() if k != "infra_error"}), "필드 누락"),
    (lambda t: _write(t, {**_blind_row(), "ranking": "reward"}), "문자열 목록"),
])
def test_bad_input_is_an_error_not_a_score(tmp_path, make, message):
    proc = _replay(make(tmp_path))
    assert proc.returncode == 2 and message in proc.stderr, proc.stderr


def test_existing_results_folder_is_not_overwritten():
    target = ROOT / "evals" / "results" / "blind_v1"
    before = {p.name: _sha(p) for p in target.iterdir()}
    proc = _replay("evals/results/blind_v1", "--tag", "blind_v1")
    assert proc.returncode == 2 and "덮어쓰지 않는다" in proc.stderr
    assert {p.name: _sha(p) for p in target.iterdir()} == before


# ---------- 공개 reference와 작업공간 ----------

def test_reference_params_match_manifest_and_have_no_user_paths():
    for seed, ref in MANIFEST["seeds"].items():
        d = ROOT / "bench" / "reference" / "params" / f"seed{seed}"
        for name, want in ref["sha256"].items():
            assert _sha(d / name) == want, f"seed{seed}/{name}"
            assert not USER_PATH.search((d / name).read_text(encoding="utf-8"))
        tele = json.loads((ROOT / ref["telemetry"]).read_text(encoding="utf-8"))
        assert tele["run_dir_name"] == ref["source_run_dir_name"]


def test_workspace_builder_does_not_read_private_files():
    assert "private" not in (ROOT / "bench" / "build_workspace.py").read_text(encoding="utf-8")


def test_blind_workspace_from_public_files_has_no_answer_information(tmp_path):
    proc = _offline("bench/build_workspace.py", "123", "--blind", "--out", tmp_path)
    assert proc.returncode == 0, proc.stderr
    ws = tmp_path / "seed123"
    files = sorted(p.relative_to(ws).as_posix() for p in ws.rglob("*") if p.is_file())
    assert files == sorted([*L.REFERENCE_FILES, *(f"cases/{c['case_id']}/telemetry.json" for c in CASES)])
    assert L.check_blind_workspace(ws, CASES, KEY, MANIFEST["seeds"]["123"]["sha256"]) == []


def test_changes_workspace_resolves_old_values_from_public_params(tmp_path, monkeypatch):
    proc = _offline("bench/build_workspace.py", "42", "--out", tmp_path)
    assert proc.returncode == 0, proc.stderr
    monkeypatch.setattr(T, "WORKSPACE", tmp_path / "seed42")
    assert T.list_cases() == [c["case_id"] for c in CASES]
    for c in CASES:
        changes = T.list_changes(c["case_id"])
        assert len(changes) == 3 and all(ch["old"] != "<없음>" for ch in changes), changes


def test_unknown_seed_fails_with_a_reason(tmp_path):
    proc = _offline("bench/build_workspace.py", "99", "--blind", "--out", tmp_path)
    assert proc.returncode != 0 and "기준 실행 없음" in proc.stderr
    assert not (tmp_path / "seed99").exists()


# ---------- 유출 검사 자체 ----------

@pytest.fixture()
def blind_ws(tmp_path):
    assert _offline("bench/build_workspace.py", "7", "--blind", "--out", tmp_path).returncode == 0
    return tmp_path / "seed7"


def test_leakcheck_flags_change_list(blind_ws):
    # override 문자열이 없어도 case.json 자체가 blind 입력에 있으면 안 된다(구조 검사)
    (blind_ws / "cases" / "c01" / "case.json").write_text('{"case_id": "c01"}', encoding="utf-8")
    assert L.check_blind_workspace(blind_ws, CASES, KEY) == ["허용 목록 밖 파일: cases/c01/case.json"]


def test_leakcheck_flags_override_text_and_changed_reference(blind_ws):
    agent_yaml = blind_ws / "reference" / "params" / "agent.yaml"
    agent_yaml.write_text(agent_yaml.read_text(encoding="utf-8") + "# env.sim.dt=0.02\n", encoding="utf-8")
    problems = L.check_blind_workspace(blind_ws, CASES, KEY, MANIFEST["seeds"]["7"]["sha256"])
    assert any("env.sim.dt=0.02" in p for p in problems) and any("SHA256" in p for p in problems)


def test_leakcheck_ignores_agent_outputs_but_flags_unknown_inputs(blind_ws):
    (blind_ws / "diagnoses").mkdir()
    (blind_ws / "diagnoses" / "c01.json").write_text('{"evidence": "env.sim.dt"}', encoding="utf-8")
    assert L.check_blind_workspace(blind_ws, CASES, KEY) == []
    (blind_ws / "notes.txt").write_text("hint", encoding="utf-8")
    assert L.check_blind_workspace(blind_ws, CASES, KEY) == ["허용 목록 밖 파일: notes.txt"]


def test_trace_scan_finds_answer_file_access(tmp_path):
    line = {"tool": "run_analysis", "args": {"code": r"open(r'..\..\..\bench\answer_key.json').read()"}}
    (tmp_path / "blind_seed1_c01_1.jsonl").write_text(json.dumps(line) + "\n", encoding="utf-8")
    hits = L.scan_traces(tmp_path)["hits"]
    assert any("answer_key" in h for h in hits) and any("../.." in h for h in hits)
