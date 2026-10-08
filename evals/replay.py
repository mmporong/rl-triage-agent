"""저장된 평가 결과를 공개 정답표로 다시 채점한다(offline replay). 모델·NAT·네트워크·GPU를 쓰지 않는다.

사용:
  python evals/replay.py evals/results/blind_v1 evals/results/heldout_blind \
      [--traces evals/results/traces] [--tag <새 폴더>] [--json]

집계 규칙
- 입력은 결과 폴더의 *.jsonl(파일 이름순, 파일 안은 기록 순서) 또는 jsonl 파일 하나다.
- (task, seed, case_id, mode) 칸마다 infra_error가 아닌 마지막 행을 그 칸의 결과로 쓴다.
  비인프라 행이 없는 칸은 오답으로 세고 infra_only_cells에 남긴다.
- truth·top-1·top-2는 저장값을 쓰지 않고 bench/answer_key.json(과제 A는 bench/cases.json도)으로 다시 계산한다.
  과제 B는 ranking의 1위·2위, 과제 A는 suspected(1위)와 ranking 2위까지를 본다. 빈 ranking은 오답이다.
- 다시 계산한 truth·correct(과제 B는 top2도)가 저장값과 하나라도 다르면 실패한다(다른 정답표로 채점됐을 가능성).

종료 코드: 0 통과, 1 저장값 불일치 또는 trace 유출 표식 발견, 2 입력 오류(파일 없음·잘못된 JSON·모르는 case·필드 누락).
--tag를 주면 evals/results/<tag>/replay.json에 표·입력 SHA256(CRLF→LF 정규화)·코드 버전·명령·실행 환경을 남긴다.
기존 폴더는 덮어쓰지 않는다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rl_triage.leakcheck import scan_traces  # noqa: E402
from rl_triage.scoring import score_blind, score_changes  # noqa: E402

KEY_PATH = ROOT / "bench" / "answer_key.json"
# 평가가 끝난 뒤 공개하는 다른 벤치 정답표. 파일이 있을 때만 그 벤치 행을 채점한다.
OTHER_KEYS = {"p0c": ROOT / "bench" / "answer_key_p0c.json"}
CASES_PATH = ROOT / "bench" / "cases.json"
CODE_FILES = ("evals/replay.py", "src/rl_triage/scoring.py", "src/rl_triage/leakcheck.py")
MODES = ("agent", "control", "control_full", "rule_prior", "rule_features", "rule_template", "rule_b0")  # rule_*: evals/baselines.py
RULE = ("per (task, seed, case_id, mode): last row with infra_error=false, in file-name then line order; "
        "a cell with only infra rows counts as wrong; truth/top-1/top-2 recomputed from bench/answer_key.json "
        "(Task A also bench/cases.json); empty ranking is wrong; any stored truth/correct/top2 that differs fails")


class ReplayError(Exception):
    """입력 오류(종료 코드 2)."""


def _sha256_lf(path: Path) -> str:
    """CRLF를 LF로 바꾼 뒤의 SHA256. .gitattributes(eol=lf)의 clean checkout과 같은 값이라 Windows 작업 사본과 비교할 수 있다."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _show(path: Path) -> str:
    """기록용 경로. 저장소 밖 경로는 사용자 경로가 남지 않게 이름만 쓴다."""
    path = Path(path).resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else f"<external>/{path.name}"


def load_rows(path: Path) -> tuple[list[dict], list[dict]]:
    path = Path(path)
    if path.is_dir():
        files = sorted(path.glob("*.jsonl"))
        if not files:
            raise ReplayError(f"{_show(path)}: jsonl 결과 파일이 없다")
    elif path.is_file():
        files = [path]
    else:
        raise ReplayError(f"{_show(path)}: 결과 폴더·파일이 없다")
    rows, infos = [], []
    for f in files:
        n_rows = 0
        for n, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            where = f"{_show(f)}:{n}"
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ReplayError(f"{where}: 잘못된 JSON ({e.msg})") from None
            if not isinstance(row, dict):
                raise ReplayError(f"{where}: 행이 JSON 객체가 아니다")
            rows.append((where, row))
            n_rows += 1
        infos.append({"path": _show(f), "sha256_lf": _sha256_lf(f), "rows": n_rows})
    if not rows:
        raise ReplayError(f"{_show(path)}: 결과 행이 없다")
    return rows, infos


def _require(where: str, row: dict, fields) -> None:
    missing = [k for k in fields if k not in row]
    if missing:
        raise ReplayError(f"{where}: 필드 누락 {missing}")


def rescore(where: str, row: dict, keys: dict, cases: dict) -> tuple[dict, list[str]]:
    """행 하나를 다시 채점한다. (재계산 결과, 저장값 불일치 목록)."""
    _require(where, row, ("case_id", "mode", "seed", "infra_error", "truth", "correct"))
    cid, task = row["case_id"], row.get("task") or "changes"
    bench = row.get("bench", "v1")
    if bench not in keys:
        raise ReplayError(f"{where}: {bench!r} 벤치는 공개 정답표가 없다")
    key = keys[bench]
    if cid not in key["cases"]:
        raise ReplayError(f"{where}: 정답표에 없는 case_id {cid!r}")
    if row["mode"] not in MODES:
        raise ReplayError(f"{where}: 모르는 mode {row['mode']!r}")
    if not isinstance(row["seed"], int) or not isinstance(row["infra_error"], bool):
        raise ReplayError(f"{where}: seed는 정수, infra_error는 true/false여야 한다")
    ranking = row.get("ranking")
    if ranking is not None and not (isinstance(ranking, list) and all(isinstance(x, str) for x in ranking)):
        raise ReplayError(f"{where}: ranking이 문자열 목록이 아니다")
    ranking = ranking or []
    if task == "blind":
        _require(where, row, ("ranking", "top2"))
        s = score_blind(cid, ranking, key)
        got, compared = {"truth": s["truth"], "correct": s["correct"], "top2": s["top2"]}, ("truth", "correct", "top2")
    elif task == "changes":
        _require(where, row, ("suspected",))
        if bench != "v1":
            raise ReplayError(f"{where}: 과제 A는 v1만 있다")
        if cid not in cases:
            raise ReplayError(f"{where}: bench/cases.json에 없는 case_id {cid!r}")
        s = score_changes(cid, row["suspected"], cases[cid]["overrides"], key)
        got, compared = {"truth": s["truth"], "correct": s["correct"], "top2": s["truth"] in ranking[:2]}, ("truth", "correct")
    else:
        raise ReplayError(f"{where}: 모르는 task {task!r}")
    mismatch = [f"{where}: {k} 저장값 {row[k]!r} != 재계산 {got[k]!r}" for k in compared if row[k] != got[k]]
    return {"task": task, **got}, mismatch


def summarize(path: Path, keys: dict, cases: dict) -> dict:
    rows, infos = load_rows(path)
    cells, mismatches, infra = {}, [], 0
    for where, row in rows:
        res, mm = rescore(where, row, keys, cases)
        mismatches += mm
        cell = (res["task"], row["seed"], row["case_id"], row["mode"])
        if row["infra_error"]:
            infra += 1
            cells.setdefault(cell, None)
        else:
            cells[cell] = res
    tasks = {}
    for (task, seed, cid, mode), res in sorted(cells.items(), key=lambda kv: kv[0]):
        t = tasks.setdefault(task, {"seeds": []})
        if seed not in t["seeds"]:
            t["seeds"].append(seed)
        m = t.setdefault(mode, {"cells": 0, "top1": 0, "top2": 0, "infra_only_cells": []})
        m["cells"] += 1
        if res is None:
            m["infra_only_cells"].append(f"s{seed}/{cid}")
        else:
            m["top1"] += res["correct"]
            m["top2"] += res["top2"]
    return {"path": _show(path), "files": infos, "rows": len(rows), "infra_rows": infra,
            "mismatches": mismatches, "tasks": tasks}


def code_version(extra_paths: list[str], code_files=CODE_FILES) -> dict:
    head = dirty = None
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=30)
        head = out.stdout.strip() or None
        if head:
            st = subprocess.run(["git", "status", "--porcelain", "--", *code_files, "bench/answer_key.json",
                                 "bench/cases.json", *extra_paths], cwd=ROOT, capture_output=True, text=True, timeout=30)
            dirty = bool(st.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return {"git_head": head, "git_dirty": dirty, "sha256_lf": {f: _sha256_lf(ROOT / f) for f in code_files}}


def _trace_digest(trace_dir: Path) -> str:
    lines = "".join(f"{f.name}\t{_sha256_lf(f)}\n" for f in sorted(trace_dir.glob("*.jsonl")))
    return hashlib.sha256(lines.encode("utf-8")).hexdigest()


def _print_table(report: dict) -> None:
    for f in report["inputs"]["results"]:
        print(f"{f['path']}: rows={f['rows']} infra={f['infra_rows']} mismatches={len(f['mismatches'])}")
        for task, t in f["tasks"].items():
            for mode in MODES:
                if mode in t:
                    m = t[mode]
                    extra = f"  infra_only={m['infra_only_cells']}" if m["infra_only_cells"] else ""
                    print(f"  {task:8s} {mode:13s} top-1 {m['top1']}/{m['cells']}  top-2 {m['top2']}/{m['cells']}"
                          f"  seeds={','.join(map(str, t['seeds']))}{extra}")
        for line in f["mismatches"]:
            print(f"  MISMATCH {line}", file=sys.stderr)
    tr = report.get("trace_scan")
    if tr:
        print(f"{tr['path']}: trace files={tr['files']} leak hits={len(tr['hits'])}")
        for h in tr["hits"]:
            print(f"  LEAK {h}", file=sys.stderr)
    print(f"status={report['status']}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="저장된 평가 결과를 공개 정답표로 재채점한다(모델·네트워크·GPU 없음)")
    ap.add_argument("inputs", nargs="+", type=Path, help="결과 폴더(evals/results/<tag>) 또는 jsonl 파일")
    ap.add_argument("--traces", type=Path, help="실행 기록 폴더. 정답 파일·작업공간 밖 경로 표식을 검사한다")
    ap.add_argument("--tag", help="evals/results/<tag>/replay.json에 기록한다(기존 폴더는 덮어쓰지 않음)")
    ap.add_argument("--json", action="store_true", help="표 대신 전체 보고서 JSON을 stdout에 쓴다")
    args = ap.parse_args(argv)
    try:
        out_dir = None
        if args.tag:
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.tag):
                raise ReplayError(f"--tag는 영문·숫자·._- 로 된 폴더 이름이어야 한다: {args.tag!r}")
            out_dir = ROOT / "evals" / "results" / args.tag
            if out_dir.exists():
                raise ReplayError(f"evals/results/{args.tag}가 이미 있다. 결과는 덮어쓰지 않는다")
        keys = {"v1": json.loads(KEY_PATH.read_text(encoding="utf-8"))}
        keys.update({b: json.loads(p.read_text(encoding="utf-8")) for b, p in OTHER_KEYS.items() if p.exists()})
        cases = {c["case_id"]: c for c in json.loads(CASES_PATH.read_text(encoding="utf-8"))}
        folders = [summarize(p, keys, cases) for p in args.inputs]
        traces = None
        if args.traces:
            if not args.traces.is_dir():
                raise ReplayError(f"{_show(args.traces)}: trace 폴더가 없다")
            traces = {"path": _show(args.traces), "sha256_lf_of_listing": _trace_digest(args.traces),
                      **scan_traces(args.traces)}
            if traces["files"] == 0:
                raise ReplayError(f"{traces['path']}: trace jsonl이 없다")
    except ReplayError as e:
        print(f"replay 입력 오류: {e}", file=sys.stderr)
        return 2
    failed = any(f["mismatches"] for f in folders) or bool(traces and traces["hits"])
    inside = [p for p in (*args.inputs, *([args.traces] if args.traces else [])) if p.resolve().is_relative_to(ROOT)]
    command = ["python", "evals/replay.py", *map(_show, args.inputs)]
    if args.traces:
        command += ["--traces", _show(args.traces)]
    if args.tag:
        command += ["--tag", args.tag]
    report = {
        "kind": "offline_replay", "status": "fail" if failed else "pass", "rule": RULE,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": " ".join(command),
        "inputs": {"answer_key": {"path": _show(KEY_PATH), "sha256_lf": _sha256_lf(KEY_PATH)},
                   "other_answer_keys": {b: {"path": _show(p), "sha256_lf": _sha256_lf(p)}
                                         for b, p in OTHER_KEYS.items() if p.exists()},
                   "cases": {"path": _show(CASES_PATH), "sha256_lf": _sha256_lf(CASES_PATH)},
                   "results": folders},
        "code": code_version([_show(p) for p in inside]),
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "model_calls": 0, "network": "not used", "gpu": "not used"},
    }
    if traces:
        report["trace_scan"] = traces
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=1))
    else:
        _print_table(report)
    if out_dir:
        out_dir.mkdir(parents=True)
        (out_dir / "replay.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        if not args.json:
            print(f"saved {_show(out_dir / 'replay.json')}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
