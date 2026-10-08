"""정답 유출 경계 검사. 표준 라이브러리만 쓴다.

- check_blind_workspace: 과제 B 작업공간의 입력 파일에 변경 목록·override·정답표 정보가 없는지 본다.
- scan_traces: 저장된 실행 기록(evals/results/traces)에서 정답 파일이나 작업공간 밖 경로를 건드린 흔적을 찾는다.

작업공간 안 검사일 뿐이다. 같은 checkout의 정답 파일을 에이전트 코드가 여는 것은 막지 못하므로,
새 채점 평가는 정답 파일이 없는 실행 위치에서 돌려야 한다(docs/IMPLEMENTATION-ORDER.md 2절 A).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

# 에이전트가 쓰는 출력 폴더. 입력이 아니므로 검사에서 뺀다.
OUTPUT_DIRS = {"scratch", "diagnoses", "preregistrations"}
REFERENCE_FILES = ("reference/telemetry.json", "reference/params/env.yaml", "reference/params/agent.yaml")
ANSWER_KEY_FIELDS = ("harmful_override", "harmful_id", "benign_ids", "secret_seed")
TRACE_MARKERS = ("answer_key", "cases.json", "case.json", "bench/private", "secret_seed", "harmful_override", "../..")


def blind_markers(cases: list[dict], key: dict | None = None) -> list[str]:
    """blind 입력에 나오면 안 되는 문자열: 모든 override(키=값과 키), 정답표 필드 이름·해로운 변경·설명."""
    marks = set(ANSWER_KEY_FIELDS)
    for case in cases:
        for ov in case["overrides"]:
            marks.update((ov, ov.split("=", 1)[0]))
    for entry in (key or {}).get("cases", {}).values():
        marks.add(entry["harmful_override"])
        if entry.get("note"):
            marks.add(entry["note"])
    return sorted(marks)


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_blind_workspace(ws: Path, cases: list[dict], key: dict | None = None,
                          reference_sha256: dict | None = None) -> list[str]:
    """위반 목록을 돌려준다. 빈 목록이면 통과.

    reference_sha256({"env.yaml": ..., "agent.yaml": ...})를 주면 기준 params가 공개 사본과 같은지도 본다.
    실패 실행의 params가 reference로 들어가면 문자열 검사로는 못 잡기 때문이다.
    """
    ws = Path(ws)
    if not (ws / "cases").is_dir():
        return [f"cases 폴더 없음: {ws.name}"]
    marks = blind_markers(cases, key)
    problems = []
    for p in sorted(ws.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(ws).as_posix()
        parts = rel.split("/")
        if parts[0] in OUTPUT_DIRS:
            continue
        if rel not in REFERENCE_FILES and not (len(parts) == 3 and parts[0] == "cases" and parts[2] == "telemetry.json"):
            problems.append(f"허용 목록 밖 파일: {rel}")
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        problems += [f"{rel}: 정답 정보 문자열 {m!r}" for m in marks if m in text]
    for name, want in (reference_sha256 or {}).items():
        p = ws / "reference" / "params" / name
        if not p.is_file():
            problems.append(f"reference/params/{name} 없음")
        elif sha256(p) != want:
            problems.append(f"reference/params/{name} SHA256이 공개 사본과 다름")
    return problems


def _strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield str(k)
            yield from _strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _strings(v)


def scan_traces(trace_dir: Path) -> dict:
    """trace jsonl의 모든 문자열 값에서 TRACE_MARKERS를 찾는다. 역슬래시는 /로 바꿔 Windows 경로도 잡는다."""
    files = sorted(Path(trace_dir).glob("*.jsonl"))
    hits = []
    for f in files:
        for n, line in enumerate(f.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if not line.strip():
                continue
            try:
                texts = list(_strings(json.loads(line)))
            except json.JSONDecodeError:
                texts = [line]
            found = {m for t in texts for m in TRACE_MARKERS if m in t.replace("\\", "/")}
            hits += [f"{f.name}:{n}: {m!r}" for m in sorted(found)]
    return {"files": len(files), "markers": list(TRACE_MARKERS), "hits": hits}
