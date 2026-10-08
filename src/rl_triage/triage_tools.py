"""RL 학습 실패 원인 추적 도구 모음 (프레임워크 독립).

NeMo Agent Toolkit 함수 래퍼(agent/nat_functions.py)와 대조군 스크립트가 같은 구현을 쓴다.
모든 파일 접근은 WORKSPACE 아래로 제한된다. OpenShell 샌드박스에서는 정책이 같은 경계를 커널에서 강제한다.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import yaml

_DEFAULT_WS = Path("/sandbox/workspace") if Path("/sandbox/workspace/cases").exists() else Path(__file__).resolve().parents[2] / "workspace"
WORKSPACE = Path(os.environ.get("TRIAGE_WORKSPACE", _DEFAULT_WS))

# 학습 실패 진단에 먼저 보는 지표. 나머지는 get_series로 조회한다.
KEY_TAGS = [
    "Train/mean_reward", "Train/mean_episode_length",
    "Episode_Termination/base_contact", "Episode_Termination/time_out",
    "Metrics/base_velocity/error_vel_xy", "Metrics/base_velocity/error_vel_yaw",
    "Loss/value_function", "Loss/surrogate", "Loss/entropy", "Loss/learning_rate",
    "Policy/mean_noise_std",
]


# ---------- 파일 접근 ----------

def _safe(path: Path) -> Path:
    p = path.resolve()
    if WORKSPACE.resolve() not in (p, *p.parents):
        raise PermissionError(f"workspace 밖 접근 거부: {p}")
    return p


def _case_dir(case_id: str) -> Path:
    return _safe(WORKSPACE / "cases" / case_id)


def _load_json(path: Path):
    return json.loads(_safe(path).read_text(encoding="utf-8"))


class _TolerantLoader(yaml.SafeLoader):
    """Isaac Lab params yaml의 !!python/* 태그를 일반 값으로 읽는다."""


def _construct_any(loader, tag_suffix, node):
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node, deep=True)
    return loader.construct_scalar(node)


_TolerantLoader.add_multi_constructor("tag:yaml.org,2002:python/", _construct_any)


def _load_yaml(path: Path):
    return yaml.load(_safe(path).read_text(encoding="utf-8"), Loader=_TolerantLoader)


def _lookup(cfg, dotted: str):
    cur = cfg
    for part in dotted.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return "<없음>"
    return cur


# ---------- 도구 ----------

def list_cases() -> list[str]:
    """분석 가능한 케이스 ID 목록."""
    return sorted(p.name for p in _safe(WORKSPACE / "cases").iterdir() if p.is_dir())


def list_changes(case_id: str) -> list[dict]:
    """이번 학습에서 정상 기준 실행 대비 바뀐 설정 목록(이전 값 → 새 값)."""
    case = _load_json(_case_dir(case_id) / "case.json")
    ref = WORKSPACE / "reference" / "params"
    env_cfg, agent_cfg = _load_yaml(ref / "env.yaml"), _load_yaml(ref / "agent.yaml")
    out = []
    for i, ov in enumerate(case["overrides"]):
        key, new = ov.split("=", 1)
        root, rest = key.split(".", 1)
        old = _lookup(env_cfg if root == "env" else agent_cfg, rest)
        out.append({"change_id": f"ch{i + 1}", "key": key, "old": old, "new": yaml.safe_load(new)})
    return out


def _last(summary, tag):
    s = summary.get(tag)
    return None if s is None else s["mean_last_20pct"]


def telemetry_overview(case_id: str) -> dict:
    """핵심 지표의 학습 후반 평균을 정상 기준 실행과 나란히 보여준다."""
    run = _load_json(_case_dir(case_id) / "telemetry.json")["summary"]
    ref = _load_json(WORKSPACE / "reference" / "telemetry.json")["summary"]
    rows = []
    for tag in KEY_TAGS:
        a, b = _last(run, tag), _last(ref, tag)
        ratio = None if a is None or b in (None, 0) else round(a / b, 3)
        rows.append({"tag": tag, "this_run_last20": a, "reference_last20": b, "ratio": ratio})
    reward_terms = sorted(t for t in run if t.startswith("Episode_Reward/"))
    for tag in reward_terms:
        a, b = _last(run, tag), _last(ref, tag)
        rows.append({"tag": tag, "this_run_last20": a, "reference_last20": b,
                     "ratio": None if a is None or b in (None, 0) else round(a / b, 3)})
    return {"case_id": case_id, "iterations": run["Train/mean_reward"]["n"], "rows": rows,
            "other_tags": sorted(set(run) - set(KEY_TAGS) - set(reward_terms))}


def get_series(case_id: str, tag: str, points: int = 20) -> dict:
    """지표 한 개의 학습 곡선(균등 간격 샘플)과 같은 지표의 정상 기준 곡선."""
    def sample(vals):
        if not vals:
            return []
        step = max(1, math.ceil(len(vals) / points))
        return [round(v, 5) for v in vals[::step]]
    run = _load_json(_case_dir(case_id) / "telemetry.json")["series"]
    ref = _load_json(WORKSPACE / "reference" / "telemetry.json")["series"]
    if tag not in run:
        return {"error": f"없는 지표: {tag}"}
    return {"tag": tag, "this_run": sample(run[tag]), "reference": sample(ref.get(tag, []))}


# run_analysis 하위 프로세스가 텔레메트리를 읽은 뒤 자기 자신을 묶는 Linux Landlock 코드(x86_64 syscall 444~446).
# 에이전트가 쓴 코드는 이 제한 뒤에 실행된다. 파이썬 설치 경로 읽기와 scratch 쓰기만 남고, 정답 파일·저장소·
# /mnt/c·/proc(다른 프로세스 환경변수)은 열 수 없다. 네트워크는 Landlock ABI 3에서 막지 못한다.
_LANDLOCK = r'''
def _triage_landlock(read_dirs, write_dirs):
    import ctypes, os
    libc = ctypes.CDLL(None, use_errno=True)
    abi = libc.syscall(444, None, 0, 1)
    if abi < 1:
        raise OSError(ctypes.get_errno(), "Landlock을 쓸 수 없다")
    handled = (1 << 13) - 1
    if abi >= 2:
        handled |= 1 << 13
    if abi >= 3:
        handled |= 1 << 14
    class _Attr(ctypes.Structure):
        _fields_ = [("handled_access_fs", ctypes.c_uint64)]
    class _Beneath(ctypes.Structure):
        _pack_ = 1
        _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]
    attr = _Attr(handled)
    rfd = libc.syscall(444, ctypes.byref(attr), ctypes.sizeof(attr), 0)
    if rfd < 0:
        raise OSError(ctypes.get_errno(), "landlock_create_ruleset")
    def add(path, access):
        fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
        try:
            rule = _Beneath(access & handled, fd)
            if libc.syscall(445, rfd, 1, ctypes.byref(rule), 0) != 0:
                raise OSError(ctypes.get_errno(), "landlock_add_rule " + path)
        finally:
            os.close(fd)
    for p in read_dirs:
        if os.path.isdir(p):
            add(p, (1 << 0) | (1 << 2) | (1 << 3))
    for p in write_dirs:
        add(p, handled)
    if libc.prctl(38, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "PR_SET_NO_NEW_PRIVS")
    if libc.syscall(446, rfd, 0) != 0:
        raise OSError(ctypes.get_errno(), "landlock_restrict_self")
    os.close(rfd)
'''
# Landlock이 걸린 뒤 하위 프로세스가 stderr에 쓰는 표식. run_analysis 결과의 sandbox 필드로 바뀌어 trace에 남는다.
_SANDBOX_MARK = "[analysis-sandbox:landlock]"
# 하위 프로세스에 넘기는 환경변수. API 키 등 부모 환경은 넘기지 않는다.
_CHILD_ENV_KEYS = ("PATH", "SYSTEMROOT", "LANG", "LC_ALL", "TMPDIR", "TEMP", "TMP")


def analysis_sandbox_mode() -> str:
    """TRIAGE_ANALYSIS_SANDBOX: required(Landlock이 안 걸리면 실행 거부) | auto(Linux면 시도) | off."""
    mode = os.environ.get("TRIAGE_ANALYSIS_SANDBOX", "auto")
    if mode not in ("required", "auto", "off"):
        raise ValueError(f"TRIAGE_ANALYSIS_SANDBOX는 required·auto·off 중 하나여야 한다: {mode!r}")
    return "off" if mode == "auto" and not sys.platform.startswith("linux") else mode


def run_analysis(code: str, case_id: str, timeout_s: int = 30) -> dict:
    """텔레메트리에 대해 파이썬 분석 코드를 실행한다.

    코드 안에서 `run`(이 케이스 series dict)과 `ref`(정상 기준 series dict)를 쓸 수 있고,
    결과는 print로 출력한다. 별도 프로세스·시간 제한·최소 환경변수로 실행한다. Linux에서는 텔레메트리를 읽은 뒤
    Landlock으로 파이썬 설치 경로 읽기와 scratch 쓰기만 남기고(TRIAGE_ANALYSIS_SANDBOX), OpenShell 샌드박스에서는
    정책이 같은 경계를 커널에서 강제한다.
    """
    mode = analysis_sandbox_mode()
    run_path = _case_dir(case_id) / "telemetry.json"
    ref_path = _safe(WORKSPACE / "reference" / "telemetry.json")
    scratch = _safe(WORKSPACE / "scratch")
    scratch.mkdir(exist_ok=True)
    prelude = (
        "import json, math, statistics, sys\n"
        f"run = json.load(open(r'{run_path}', encoding='utf-8'))['series']\n"
        f"ref = json.load(open(r'{ref_path}', encoding='utf-8'))['series']\n"
    )
    if mode != "off":
        reads = sorted({p for d in (sys.prefix, sys.base_prefix, sys.exec_prefix) for p in (d, os.path.realpath(d))}
                       | {"/usr", "/lib", "/lib64", "/etc"})
        prelude += _LANDLOCK + (
            "try:\n"
            f"    _triage_landlock({reads!r}, [{str(scratch)!r}])\n"
            f"    sys.stderr.write({_SANDBOX_MARK!r} + '\\n')\n"
            "except Exception as _e:\n"
            f"    if {mode!r} == 'required':\n"
            "        raise SystemExit('analysis sandbox required but unavailable: %s' % _e)\n"
            "    print('[analysis sandbox unavailable: %s]' % _e, file=sys.stderr)\n"
            "del _triage_landlock\n"
        )
    with tempfile.NamedTemporaryFile("w", suffix=".py", dir=scratch, delete=False, encoding="utf-8") as f:
        f.write(prelude + code)
        script = f.name
    env = {k: os.environ[k] for k in _CHILD_ENV_KEYS if k in os.environ}
    env.update(PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    try:
        proc = subprocess.run([sys.executable, script], capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=timeout_s, cwd=scratch, env=env)
        # 표식은 에이전트 코드보다 먼저 찍힌다. required에서는 Landlock이 실패하면 에이전트 코드가 아예 실행되지 않는다.
        # off면 Landlock 코드를 넣지 않았으므로 표식을 읽지 않는다(에이전트 코드가 표식을 흉내 낼 수 있다).
        lines = proc.stderr.splitlines(keepends=True)
        sandboxed = mode != "off" and bool(lines) and lines[0].strip() == _SANDBOX_MARK
        stderr = "".join(lines[1:] if sandboxed else lines)
        sandbox = "landlock" if sandboxed else ("off" if mode == "off" else "unavailable")
        # traceback에 찍히는 작업공간 절대경로(사용자 이름 포함)를 가린다. 모델 입력과 trace 모두에 남지 않는다.
        ws = str(WORKSPACE.resolve())
        hide = lambda s: s.replace(ws, "<workspace>")  # noqa: E731
        # sandbox를 첫 키로 둬 trace(result_head 600자)에 남게 한다.
        return {"sandbox": sandbox, "exit_code": proc.returncode, "stdout": hide(proc.stdout)[-4000:],
                "stderr": hide(stderr)[-2000:]}
    except subprocess.TimeoutExpired:
        return {"sandbox": "timeout", "exit_code": None, "stdout": "", "stderr": f"timeout {timeout_s}s"}
    finally:
        os.unlink(script)


def query_ledger(case_id: str) -> list[dict]:
    """이 학습 계열에서 이미 시도했다가 기각된 개입 기록."""
    path = _case_dir(case_id) / "ledger.json"
    return _load_json(path) if path.exists() else []


def write_preregistration(case_id: str, suspected_change_id: str, ranking: list[str], hypothesis: str,
                          single_experimental_variable: str, acceptance_gate: str,
                          expected_signature: str) -> dict:
    """다음 실험을 사전등록 JSON으로 남긴다. 한 번에 변수 하나만 바꾸는 실험이어야 한다."""
    changes = {c["change_id"] for c in list_changes(case_id)}
    bad = [c for c in [suspected_change_id, *ranking] if c not in changes]
    if bad:
        return {"error": f"없는 change_id: {bad}. 가능한 값: {sorted(changes)}"}
    doc = {
        "case_id": case_id,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "suspected_change_id": suspected_change_id,
        "ranking": ranking,
        "hypothesis": hypothesis,
        "single_experimental_variable": single_experimental_variable,
        "acceptance_gate": acceptance_gate,
        "expected_signature": expected_signature,
        "frozen_for_next_run": True,
    }
    out_dir = _safe(WORKSPACE / "preregistrations")
    out_dir.mkdir(exist_ok=True)
    (out_dir / f"{case_id}.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"saved": f"preregistrations/{case_id}.json", **doc}


MECHANISMS = ["reward", "actuator", "exploration", "optimizer", "physics", "termination"]


def write_diagnosis(case_id: str, mechanism_ranking: list[str], evidence: str, next_check: str) -> dict:
    """원인을 모르는 실패에서 메커니즘 순위를 저장한다(과제 B). 설정 변경 목록 없이 텔레메트리만으로 판단한다."""
    bad = [m for m in mechanism_ranking if m not in MECHANISMS]
    if bad or not mechanism_ranking:
        return {"error": f"알 수 없는 메커니즘: {bad}. 가능한 값: {MECHANISMS}"}
    doc = {"case_id": case_id, "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "mechanism_ranking": mechanism_ranking, "evidence": evidence, "next_check": next_check}
    out_dir = _safe(WORKSPACE / "diagnoses")
    out_dir.mkdir(exist_ok=True)
    (out_dir / f"{case_id}.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"saved": f"diagnoses/{case_id}.json", **doc}
