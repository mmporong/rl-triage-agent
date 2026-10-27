"""V05 선택 원장을 생성 없이 감사하는 두 번째 읽기 전용 inspector."""

from __future__ import annotations

import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import model_decision_journal_v1 as JOURNAL
import vision_selection_call_v1 as CALL
import vision_selection_execution_v1 as V1

UTC_TZ = timezone.utc
_DECISION_NAMES = {"manifest.json", "events", "terminal.json", ".active"}
_API_NAMES = {"manifest.json", "reservations", "responses", "receipts", ".active"}


def load_context(protocol: Path, protocol_sha_lf: str, source_anchor: str,
                 input_locations: Path) -> dict[str, Any]:
    """기존 production prepare/source 검증을 그대로 재사용한다."""
    return V1.load_context(protocol, protocol_sha_lf, source_anchor, input_locations)


def _expected_manifest(stage: Path, manifest: dict[str, Any], row: dict[str, Any],
                       frozen: dict[str, Any]) -> dict[str, Any]:
    binding = {
        "run_id": row["run_id"], "source_anchor": manifest["source_anchor"],
        "protocol_sha256": JOURNAL.file_sha(stage / "manifest.json"),
        "checkpoint_sha256": row["checkpoint_sha256"],
        "snapshot_sha256": row["capsule_sha256"],
        "initial_messages_sha256": JOURNAL.digest(frozen["messages"]),
    }
    checked = JOURNAL._binding(binding)
    params = deepcopy(CALL.OBSERVER.REQUEST_PARAMS)
    return {
        "schema": JOURNAL.SCHEMA, "binding": checked,
        "binding_sha256": JOURNAL.digest(checked), "params": params,
        "params_sha256": JOURNAL.digest(params), "retry_delay_s": 30.0,
        "api_cap": manifest["api_cap"],
    }


def _children(path: Path, allowed: set[str], label: str) -> set[str]:
    if path.is_symlink() or not path.is_dir():
        raise JOURNAL.JournalIntegrityError(f"{label} root가 canonical directory가 아니다")
    names = {child.name for child in path.iterdir()}
    unknown = names - allowed
    if unknown:
        raise JOURNAL.JournalIntegrityError(f"{label} root의 등록되지 않은 항목: {sorted(unknown)}")
    return names


def _journals(context: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None,
                                                 dict[str, str], dict[str, dict[str, Any]]]:
    """등록 decision 상태와 전역 API를 읽기만 하며 부분·외부 상태를 거부한다."""
    repository = Path(context["repository"])
    stage = Path(context["stage"])
    manifest = context["manifest"]
    plan_repository, plan_stage, live_manifest = CALL._plan(context["prepared"]["plan"])
    if plan_repository != repository or plan_stage != stage or live_manifest != manifest:
        raise JOURNAL.JournalIntegrityError("inspection context와 봉인 plan이 다르다")

    decisions = stage / "decisions"
    if decisions.exists() or decisions.is_symlink():
        _children(decisions, {row["call_id"] for row in manifest["rows"]}, "decision inventory")
    api = stage / "api"
    api_exists = api.exists() or api.is_symlink()
    if api_exists:
        names = _children(api, _API_NAMES, "API")
        if "manifest.json" not in names:
            raise JOURNAL.JournalIntegrityError("전역 API root에 manifest가 없다")

    expected_sha: dict[str, str] = {}
    started: dict[str, dict[str, Any]] = {}
    for row in manifest["rows"]:
        _, checked_row, frozen = CALL._row(repository, manifest, row["call_id"])
        expected = _expected_manifest(stage, manifest, checked_row, frozen)
        expected_digest = JOURNAL.digest(expected)
        expected_sha[row["call_id"]] = expected_digest
        root = decisions / row["call_id"]
        if not root.exists() and not root.is_symlink():
            continue
        names = _children(root, _DECISION_NAMES, f"decision {row['call_id']}")
        if "manifest.json" not in names:
            raise JOURNAL.JournalIntegrityError(
                f"decision {row['call_id']}에 상태가 있지만 manifest가 없다")
        path = root / "manifest.json"
        if JOURNAL._read(path) != expected or JOURNAL.file_sha(path) != expected_digest:
            raise JOURNAL.JournalIntegrityError(f"decision {row['call_id']} manifest가 다르다")
        started[row["call_id"]] = {"row": checked_row, "frozen": frozen, "root": root}

    if started and not api_exists:
        raise JOURNAL.JournalIntegrityError("decision manifest는 있지만 전역 API root가 없다")
    if api_exists and not started:
        raise JOURNAL.JournalIntegrityError("전역 API manifest의 소유 decision이 없다")
    if not started:
        return {"rows": [], "own": [], "reserved": 0, "pending": 0,
                "remaining": manifest["api_cap"]}, None, expected_sha, started

    audit = None
    journals: dict[str, Any] = {}
    for call_id, item in started.items():
        journal = CALL._journal(stage, manifest, item["row"], item["frozen"])
        journal._verify_prepared()
        current = journal._audit_api()
        if audit is not None and [(r["reservation"], r["response"]) for r in current["rows"]] != [
                (r["reservation"], r["response"]) for r in audit["rows"]]:
            raise JOURNAL.JournalIntegrityError("decision별 전역 API 감사 결과가 다르다")
        audit = current
        journals[call_id] = journal
    assert audit is not None
    started_shas = {journal.manifest_sha256: call_id for call_id, journal in journals.items()}
    all_expected = set(expected_sha.values())
    for item in audit["rows"]:
        owner = item["reservation"]["manifest_sha256"]
        if owner not in started_shas:
            if owner in all_expected:
                raise JOURNAL.JournalIntegrityError("API reservation 소유 decision manifest가 없다")
            raise JOURNAL.JournalIntegrityError("등록 밖 decision manifest의 API reservation이다")
    return audit, journals, expected_sha, started


def inspect_run(context: dict[str, Any], now_utc: datetime | None = None) -> dict[str, Any]:
    """원장을 쓰지 않고 첫 미완료 decision과 전역 API 상태를 판정한다."""
    now = (now_utc or datetime.now(UTC_TZ)).astimezone(UTC_TZ)
    stage, manifest, repository = Path(context["stage"]), context["manifest"], Path(context["repository"])
    if V1._owner_present(stage):
        result = V1._inspection("blocked_pending", manifest, None, None, 0, 0, 0, 0)
        result["decision"] = "HOLD"
        return result
    audit, journals, _expected, started = _journals(context)
    terminal = semantic = infra = 0
    first: tuple[dict[str, Any], dict[str, Any]] | None = None
    terminal_gap = False
    for row in manifest["rows"]:
        _, checked_row, frozen = CALL._row(repository, manifest, row["call_id"])
        terminal_path = stage / "decisions" / row["call_id"] / "terminal.json"
        if terminal_path.is_file():
            if first is not None:
                terminal_gap = True
                break
            replay = CALL._replay(stage, manifest, checked_row, frozen)
            terminal += 1
            engine_result = replay["engine_result"]
            if engine_result.get("hold_reason") == "selection_invalid":
                semantic += 1
            if (str(engine_result.get("hold_reason", "")).startswith("transport_")
                    or engine_result.get("stop_reason") == "budget_exhausted"):
                infra += 1
            continue
        if first is None:
            first = checked_row, frozen
    if terminal_gap:
        raise JOURNAL.JournalIntegrityError("완료 decision 순서에 gap이 있다")
    if first is None:
        return V1._inspection("complete", manifest, None, None, terminal, semantic, infra,
                              audit["reserved"], audit)
    row, _frozen = first
    decision = stage / "decisions" / row["call_id"]
    if V1._owner_present(stage, decision):
        result = V1._inspection("blocked_pending", manifest, row["call_id"], None,
                                terminal, semantic, infra, audit["reserved"], audit)
        result["decision"] = "HOLD"
        return result
    if audit["pending"]:
        result = V1._inspection("blocked_pending", manifest, row["call_id"], None,
                                terminal, semantic, infra, audit["reserved"], audit)
        result["decision"] = "HOLD"
        return result
    current = journals.get(row["call_id"]) if journals else None
    if current is not None:
        own = audit["own"] if len(started) == 1 else current._audit_api()["own"]
        if own and own[-1]["response"] is not None:
            retry = own[-1]["response"].get("retry_not_before")
            if retry is not None and now < JOURNAL._parse_stamp(retry):
                return V1._inspection("paused_infra", manifest, row["call_id"], retry,
                                      terminal, semantic, infra, audit["reserved"], audit)
    if audit["remaining"] <= 0:
        return V1._inspection("cap_exhausted", manifest, row["call_id"], None,
                              terminal, semantic, infra, audit["reserved"], audit)
    return V1._inspection("runnable", manifest, row["call_id"], None,
                          terminal, semantic, infra, audit["reserved"], audit)


def remaining_commands(context: dict[str, Any], now_utc: datetime | None = None) -> dict[str, Any]:
    """기존 명령 형식을 유지하면서 v2 inspection 결과를 사용한다."""
    state = inspect_run(context, now_utc)
    commands: list[list[str]] = []
    if state["status"] == "runnable":
        commands.append([
            "setsid", str(Path(sys.executable).resolve()),
            str((V1.ROOT / "evals/vision_selection_worker_v1.py").resolve()),
            "advance", "--protocol", str(context["protocol_path"]),
            "--protocol-sha256-lf", context["protocol_sha256_lf"],
            "--source-anchor", context["source_anchor"],
            "--input-locations", str(context["input_locations"]),
            "--call-id", state["next_call_id"],
        ])
    return {**state, "commands": commands}
