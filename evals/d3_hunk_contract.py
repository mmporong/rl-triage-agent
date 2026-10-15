"""D3 변경 순위의 hunk 인용 계약과 의미 검증기.

줄 번호 대신 change 라벨에 묶인 hunk id와 diff에서 복사한 연속 코드 조각을 요구한다.
코드 조각은 해당 hunk 안에 정확히 존재하고 추가·삭제 줄을 하나 이상 포함해야 한다.
"""
from __future__ import annotations

import json
import re
from typing import Any


SCHEMA = "change_rank_hunk_packet_v1"
OUTPUT_KEYS = {"ranking", "reasons", "uncertain"}
REASON_KEYS = {"change", "facts", "probes", "hunk", "quote"}
HUNK_HEADER = re.compile(r"^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@.*$")
IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
SYSTEM = (
    "You review source changes for a legged-robot reinforcement-learning run. Rank every neutral change label from "
    "most to least likely to have caused the observed failure. Return only one JSON object with keys ranking, reasons, "
    "and uncertain. ranking must contain every label exactly once. Each reason must contain exactly change, facts, "
    "probes, hunk, and quote. hunk must be an H-number shown for that change. quote must copy a contiguous code fragment "
    "from that hunk exactly, without diff +/- prefixes, and must include at least one added or removed line. Do not invent, "
    "paraphrase, or repair a quote. Copy complete code lines with indentation retained. The quote must contain at least "
    "20 non-padding characters and an identifier of at least 3 characters. Include adjacent lines when one changed line "
    "is too short. The first reason must cite the first ranked change. facts and probes may cite only ids "
    "present in the input. uncertain is a boolean."
)


def parse_hunks(diff: str) -> list[dict[str, Any]]:
    """unified diff를 검증 가능한 hunk 목록으로 바꾼다."""
    hunks: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for raw in diff.splitlines():
        if HUNK_HEADER.match(raw):
            current = {"hunk": f"H{len(hunks) + 1}", "header": raw, "lines": []}
            hunks.append(current)
        elif current is not None and raw[:1] in {" ", "+", "-"} and not raw.startswith(("+++", "---")):
            current["lines"].append({"kind": raw[0], "code": raw[1:]})
    if not hunks or any(not hunk["lines"] for hunk in hunks):
        raise ValueError("unified diff hunk를 읽을 수 없다")
    return hunks


def build_packet(packet: dict) -> dict:
    if packet.get("schema") != "change_rank_packet_v1":
        raise ValueError("기존 D3 packet 형식이 아니다")
    changes = []
    for change in packet.get("changes", []):
        changes.append({"label": change["label"], "hunks": parse_hunks(change["diff"])})
    return {"schema": SCHEMA, "changes": changes, "facts": packet.get("facts", {}),
            "probe_observations": packet.get("probe_observations", [])}


def messages(packet: dict) -> list[dict[str, str]]:
    transformed = build_packet(packet)
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": json.dumps(transformed, ensure_ascii=False, sort_keys=True)}]


def parse(content: str):
    if not isinstance(content, str):
        return None
    text = (content or "").strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.S)
    try:
        return json.loads(fenced.group(1) if fenced else text)
    except (json.JSONDecodeError, AttributeError):
        return None


def _quote_match(quote: str, hunk: dict) -> bool:
    """인용이 한 hunk의 연속 코드 줄이고 그 구간에 변경 줄이 있는지 확인한다."""
    if not isinstance(quote, str) or quote != quote.strip("\n"):
        return False
    if len(quote.strip()) < 20 or not IDENTIFIER.search(quote):
        return False
    quoted = quote.split("\n")
    lines = hunk["lines"]
    for start in range(len(lines) - len(quoted) + 1):
        window = lines[start:start + len(quoted)]
        if [line["code"] for line in window] == quoted and any(line["kind"] in {"+", "-"} for line in window):
            return True
    return False


def validate(output: Any, packet: dict) -> list[str]:
    """새 계약 위반 목록. 오류가 하나라도 있으면 의미 재시도 대상이 아니다."""
    if not isinstance(output, dict) or set(output) != OUTPUT_KEYS:
        return ["keys"]
    transformed = build_packet(packet)
    labels = [change["label"] for change in transformed["changes"]]
    ranking, reasons = output["ranking"], output["reasons"]
    problems: list[str] = []
    if (not isinstance(ranking, list) or not all(isinstance(item, str) for item in ranking)
            or len(ranking) != len(labels) or sorted(ranking) != sorted(labels)):
        problems.append("ranking_not_a_permutation")
    if not isinstance(output["uncertain"], bool):
        problems.append("uncertain_not_bool")
    if not isinstance(reasons, list) or not reasons or not all(isinstance(reason, dict) for reason in reasons):
        return problems + ["reasons"]
    observed = {row.get("probe") for row in transformed["probe_observations"] if isinstance(row, dict)}
    hunks = {change["label"]: {hunk["hunk"]: hunk for hunk in change["hunks"]}
             for change in transformed["changes"]}
    for index, reason in enumerate(reasons):
        prefix = f"reason{index}"
        change = reason.get("change")
        if set(reason) != REASON_KEYS or not isinstance(change, str) or change not in hunks:
            problems.append(f"{prefix}_shape")
            continue
        if (not isinstance(reason["facts"], list)
                or not all(isinstance(fact, str) for fact in reason["facts"])
                or any(fact not in transformed["facts"] for fact in reason["facts"])):
            problems.append(f"{prefix}_facts")
        if (not isinstance(reason["probes"], list)
                or not all(isinstance(probe, str) for probe in reason["probes"])
                or any(probe not in observed for probe in reason["probes"])):
            problems.append(f"{prefix}_probes")
        hunk_id = reason["hunk"]
        hunk = hunks[change].get(hunk_id) if isinstance(hunk_id, str) else None
        if hunk is None:
            problems.append(f"{prefix}_hunk")
        elif not _quote_match(reason["quote"], hunk):
            problems.append(f"{prefix}_quote")
    if (isinstance(ranking, list) and ranking and reasons and isinstance(reasons[0], dict)
            and reasons[0].get("change") != ranking[0]):
        problems.append("top_reason")
    return problems
