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

WORKSPACE = Path(os.environ.get("TRIAGE_WORKSPACE", Path(__file__).resolve().parents[1] / "workspace"))

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


def run_analysis(code: str, case_id: str, timeout_s: int = 30) -> dict:
    """텔레메트리에 대해 파이썬 분석 코드를 실행한다.

    코드 안에서 `run`(이 케이스 series dict)과 `ref`(정상 기준 series dict)를 쓸 수 있고,
    결과는 print로 출력한다. 별도 프로세스·시간 제한으로 실행하며 OpenShell 샌드박스에서는
    네트워크와 workspace 밖 쓰기가 커널에서 막힌다.
    """
    run_path = _case_dir(case_id) / "telemetry.json"
    ref_path = _safe(WORKSPACE / "reference" / "telemetry.json")
    prelude = (
        "import json, math, statistics\n"
        f"run = json.load(open(r'{run_path}', encoding='utf-8'))['series']\n"
        f"ref = json.load(open(r'{ref_path}', encoding='utf-8'))['series']\n"
    )
    scratch = _safe(WORKSPACE / "scratch")
    scratch.mkdir(exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".py", dir=scratch, delete=False, encoding="utf-8") as f:
        f.write(prelude + code)
        script = f.name
    try:
        proc = subprocess.run([sys.executable, script], capture_output=True, text=True,
                              timeout=timeout_s, cwd=scratch)
        return {"exit_code": proc.returncode, "stdout": proc.stdout[-4000:], "stderr": proc.stderr[-2000:]}
    except subprocess.TimeoutExpired:
        return {"exit_code": None, "stdout": "", "stderr": f"timeout {timeout_s}s"}
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
