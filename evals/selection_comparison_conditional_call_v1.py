"""D4의 조건부 20-slot 선택 호출과 60-request 불변 원장."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))

import selection_comparison_call_v2 as C  # noqa: E402  # pyright: ignore[reportMissingImports]

V, MC = C.V, C.MC
MODEL = C.MODEL
ACCEPTED_PROVIDER_MODELS = C.ACCEPTED_PROVIDER_MODELS
API_CAP = 60
SLOT_COUNT = 20
SLOT_FIELDS = {
    "slot_id", "policy", "arm", "epoch", "ordinal", "condition", "max_requests",
    "prerequisite", "plan_sha256",
}
DISPOSITIONS = {
    "executed", "skipped_terminal", "skipped_epoch_absent", "no_model_required",
}
TERMINAL_REASONS = {
    "view_terminal", "model_hold", "model_invalid", "model_infra", "batch_invalid", "budget",
}
RECEIPT_FIELDS = {
    "schema", "slot_id", "ordinal", "plan_sha256", "snapshot_digest", "capsule_sha256",
    "disposition", "terminal_reason", "terminal_receipt_sha256",
}
CAPSULE_FIELDS = {
    "schema", "slot", "disposition", "snapshot_inputs", "snapshot_digest", "history_utf8",
    "history_sha256", "input_digest", "terminal_reason", "terminal_receipt_sha256",
}
RESERVATION_SCHEMA = "selection_conditional_api_reservation_v1"
RESPONSE_SCHEMA = "selection_conditional_api_response_v1"
_EXCEPTION = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,79}")


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _raw_sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_new(path: Path, value: object) -> None:
    MC.write_new(path, value)


def _read(path: Path) -> dict:
    return MC.read(path)


def _slot_specs() -> list[dict]:
    rows: list[dict] = []

    def add(slot_id: str, policy: str, epoch: int, condition: str, prerequisite: dict) -> None:
        rows.append({
            "slot_id": slot_id,
            "policy": policy,
            "arm": policy,
            "epoch": epoch,
            "ordinal": len(rows) + 1,
            "condition": condition,
            "max_requests": 3,
            "prerequisite": prerequisite,
        })

    for epoch in (1, 2):
        add(f"model_first_then_rule:e{epoch}:first", "model_first_then_rule", epoch, "production",
            {"kind": "epoch_first_open", "observation_count": 0})
    for epoch in (1, 2):
        for observation_count in range(7):
            add(f"model_adaptive:e{epoch}:open{observation_count}", "model_adaptive", epoch, "production",
                {"kind": "open_state", "observation_count": observation_count})
    for epoch in (1, 2):
        for condition in ("exposure_full", "exposure_masked"):
            add(f"model_adaptive:e{epoch}:{condition}", "model_adaptive", epoch, condition,
                {"kind": "adaptive_observation", "minimum_observations": 1})
    require(len(rows) == SLOT_COUNT, "정적 slot 수")
    return rows


def _plan_digest(rows: list[dict]) -> str:
    return _digest({
        "schema": "selection_conditional_plan_body_v1",
        "model": MODEL,
        "accepted_provider_models": sorted(ACCEPTED_PROVIDER_MODELS),
        "request_params": C.F.request_params(MODEL),
        "api_cap": API_CAP,
        "slots": rows,
    })


def plan_value() -> dict:
    bare = _slot_specs()
    plan_sha = _plan_digest(bare)
    slots = [{**row, "plan_sha256": plan_sha} for row in bare]
    return {
        "schema": "selection_conditional_plan_v1",
        "model": MODEL,
        "accepted_provider_models": sorted(ACCEPTED_PROVIDER_MODELS),
        "request_params": C.F.request_params(MODEL),
        "api_cap": API_CAP,
        "slot_count": SLOT_COUNT,
        "plan_sha256": plan_sha,
        "slots": slots,
    }


def validate_plan(plan: dict) -> dict:
    require(plan == plan_value(), "정적 20-slot plan drift")
    return deepcopy(plan)


def _slot(plan: dict, slot_id: str) -> dict:
    validate_plan(plan)
    matches = [row for row in plan["slots"] if row["slot_id"] == slot_id]
    require(len(matches) == 1 and set(matches[0]) == SLOT_FIELDS, "등록 slot 식별자")
    return deepcopy(matches[0])


def _portable_snapshot(snapshot: dict) -> dict:
    value = deepcopy(snapshot)
    require(isinstance(value.get("inputs"), dict), "snapshot inputs")
    value["inputs"].pop("ledger_path", None)
    return value


def _dialogue_input(snapshot: dict, condition: str) -> dict:
    checked = V.checked(snapshot)
    view = C.query_view(checked, condition)
    extra = ("\nNon-executable observation query: candidates and eligible_probes are held at the state before "
             "the latest outcome. Use the exposed evidence to choose hold or the next probe. This query never executes."
             if condition != "production" else "")
    messages = [{"role": "system", "content": V.SYSTEM + extra},
                {"role": "user", "content": V.canonical(view).decode()}]
    policy = checked["inputs"]["policy"]
    needs_model = condition != "production" or (
        view["status"] == "open" and view["budget_left"] > 0
        and (policy == "model_adaptive" or (policy == "model_first_then_rule" and not view["observations"]))
    )
    return {"view": view, "messages": messages, "messages_sha256": V.L.digest(messages),
            "params": C.F.request_params(MODEL), "needs_model": needs_model}


def _capsule(slot: dict, snapshot: dict | None, disposition: str, *, terminal_reason: str | None = None,
             terminal_receipt_sha256: str | None = None) -> tuple[dict, str]:
    if snapshot is None:
        value = {"schema": "selection_conditional_capsule_v1", "slot": deepcopy(slot),
                 "disposition": disposition, "snapshot_inputs": None, "snapshot_digest": None,
                 "history_utf8": None, "history_sha256": None, "input_digest": None,
                 "terminal_reason": terminal_reason, "terminal_receipt_sha256": terminal_receipt_sha256}
        return value, _digest(value)
    checked = V.checked(snapshot)
    ledger_path = Path(checked["inputs"]["ledger_path"])
    history = ledger_path.read_bytes() if ledger_path.exists() else b""
    try:
        history_utf8 = history.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("probe ledger는 UTF-8이어야 한다") from error
    inputs = deepcopy(checked["inputs"])
    inputs.pop("ledger_path")
    dialogue = None if disposition == "skipped_terminal" else _dialogue_input(checked, slot["condition"])
    value = {
        "schema": "selection_conditional_capsule_v1",
        "slot": deepcopy(slot),
        "disposition": disposition,
        "snapshot_inputs": inputs,
        "snapshot_digest": _digest(_portable_snapshot(checked)),
        "history_utf8": history_utf8,
        "history_sha256": _raw_sha(history),
        "input_digest": (None if dialogue is None else
                         _digest({"view": dialogue["view"], "messages": dialogue["messages"],
                                  "params": dialogue["params"]})),
        "terminal_reason": terminal_reason,
        "terminal_receipt_sha256": terminal_receipt_sha256,
    }
    return value, _digest(value)


def materialize_slot(slot: dict, snapshot: dict | None) -> dict:
    require(isinstance(slot, dict) and isinstance(slot.get("slot_id"), str), "slot 형식")
    expected = _slot(plan_value(), slot["slot_id"])
    require(slot == expected, "slot 변조 또는 plan drift")
    if snapshot is None:
        disposition = "skipped_epoch_absent"
        terminal_reason = terminal_receipt_sha256 = None
    else:
        checked = V.checked(snapshot)
        view = checked["view"]
        require(checked["inputs"]["policy"] == slot["policy"]
                and checked["inputs"]["packet"]["epoch"] == slot["epoch"], "snapshot arm·epoch 결합")
        observations = len(view["observations"])
        prerequisite = slot["prerequisite"]
        dialogue = _dialogue_input(checked, slot["condition"])
        terminal_reason = terminal_receipt_sha256 = None
        if prerequisite["kind"] == "adaptive_observation":
            require(slot["condition"] != "production"
                    and observations >= prerequisite["minimum_observations"], "노출 slot 관찰 전제")
            disposition = "executed"
        elif view["status"] != "open" or view["budget_left"] <= 0:
            disposition = "skipped_terminal"
            terminal_reason = "view_terminal"
        elif prerequisite["kind"] == "open_state":
            require(observations == prerequisite["observation_count"], "slot 관찰 순서 결합")
            disposition = "executed"
        elif prerequisite["kind"] == "epoch_first_open":
            require(prerequisite["observation_count"] == 0, "첫 선택 slot 전제")
            disposition = "executed" if dialogue["needs_model"] else "no_model_required"
        elif not dialogue["needs_model"]:
            disposition = "no_model_required"
        else:
            raise ValueError("등록되지 않은 slot 전제")
    capsule, capsule_sha = _capsule(slot, snapshot, disposition, terminal_reason=terminal_reason,
                                    terminal_receipt_sha256=terminal_receipt_sha256)
    return {
        "schema": "selection_conditional_slot_materialization_v1",
        "slot_id": slot["slot_id"],
        "ordinal": slot["ordinal"],
        "plan_sha256": slot["plan_sha256"],
        "snapshot_digest": capsule["snapshot_digest"],
        "capsule_sha256": capsule_sha,
        "disposition": disposition,
        "terminal_reason": terminal_reason,
        "terminal_receipt_sha256": terminal_receipt_sha256,
        "capsule": capsule,
    }


def materialize_skipped_slot(slot: dict, snapshot: dict, reason: str,
                             terminal_receipt_sha256: str) -> dict:
    """열린 view의 future slot을 검증된 selector terminal receipt에 묶어 닫는다."""
    require(reason in TERMINAL_REASONS - {"view_terminal"}, "등록 terminal reason")
    require(V.S._valid_sha(terminal_receipt_sha256), "terminal receipt SHA")
    checked = V.checked(snapshot)
    require(isinstance(slot, dict) and isinstance(slot.get("slot_id"), str), "slot 형식")
    expected = _slot(plan_value(), slot["slot_id"])
    require(slot == expected and checked["inputs"]["policy"] == slot["policy"]
            and checked["inputs"]["packet"]["epoch"] == slot["epoch"], "terminal skip snapshot 결합")
    capsule, capsule_sha = _capsule(slot, checked, "skipped_terminal", terminal_reason=reason,
                                    terminal_receipt_sha256=terminal_receipt_sha256)
    return {"schema": "selection_conditional_slot_materialization_v1", "slot_id": slot["slot_id"],
            "ordinal": slot["ordinal"], "plan_sha256": slot["plan_sha256"],
            "snapshot_digest": capsule["snapshot_digest"], "capsule_sha256": capsule_sha,
            "disposition": "skipped_terminal", "terminal_reason": reason,
            "terminal_receipt_sha256": terminal_receipt_sha256, "capsule": capsule}


def initialize(root: Path, plan: dict) -> None:
    root = Path(root).resolve()
    value = validate_plan(plan)
    path = root / "plan.json"
    if path.exists():
        require(_read(path) == value, "실행 root plan drift")
    else:
        _write_new(path, value)
    for name in ("slots", "capsules", "histories", "dialogues", "api/reservations", "api/responses"):
        (root / name).mkdir(parents=True, exist_ok=True)


def _root_plan(root: Path, plan: dict) -> Path:
    root = Path(root).resolve()
    validate_plan(plan)
    require((root / "plan.json").is_file() and _read(root / "plan.json") == plan, "실행 root plan 결합")
    return root


def _slot_path(root: Path, ordinal: int) -> Path:
    return root / "slots" / f"{ordinal:02d}.json"


def record_slot(root: Path, plan: dict, materialized: dict) -> dict:
    root = _root_plan(root, plan)
    require(set(materialized) == RECEIPT_FIELDS | {"capsule"}
            and materialized["schema"] == "selection_conditional_slot_materialization_v1", "slot materialization 형식")
    slot = _slot(plan, materialized["slot_id"])
    require(materialized["ordinal"] == slot["ordinal"]
            and materialized["plan_sha256"] == plan["plan_sha256"]
            and materialized["disposition"] in DISPOSITIONS
            and _digest(materialized["capsule"]) == materialized["capsule_sha256"], "slot capsule 결합")
    capsule = materialized["capsule"]
    require(capsule["slot"] == slot and capsule["disposition"] == materialized["disposition"]
            and capsule["snapshot_digest"] == materialized["snapshot_digest"], "slot capsule 내용")
    require(capsule["terminal_reason"] == materialized["terminal_reason"]
            and capsule["terminal_receipt_sha256"] == materialized["terminal_receipt_sha256"]
            and ((materialized["disposition"] == "skipped_terminal"
                  and materialized["terminal_reason"] in TERMINAL_REASONS)
                 or (materialized["disposition"] != "skipped_terminal"
                     and materialized["terminal_reason"] is None
                     and materialized["terminal_receipt_sha256"] is None)), "terminal skip receipt 결합")
    if materialized["terminal_reason"] == "view_terminal":
        require(materialized["terminal_receipt_sha256"] is None, "view terminal은 외부 receipt를 쓰지 않는다")
    elif materialized["disposition"] == "skipped_terminal":
        require(V.S._valid_sha(materialized["terminal_receipt_sha256"]), "외부 terminal receipt SHA")
    capsule_path = root / "capsules" / f"{materialized['capsule_sha256']}.json"
    _write_new(capsule_path, capsule)
    if capsule["history_utf8"] is not None:
        raw = capsule["history_utf8"].encode("utf-8")
        require(_raw_sha(raw) == capsule["history_sha256"], "동결 history SHA")
        history = root / "histories" / f"{capsule['history_sha256']}.jsonl"
        if history.exists():
            require(history.read_bytes() == raw, "동일 history SHA의 내용 불일치")
        else:
            with history.open("xb") as handle:
                handle.write(raw)
    receipt = {key: deepcopy(materialized[key]) for key in RECEIPT_FIELDS}
    receipt["schema"] = "selection_conditional_slot_receipt_v1"
    _write_new(_slot_path(root, slot["ordinal"]), receipt)
    return receipt


def _load_receipt(root: Path, plan: dict, slot_id: str) -> tuple[dict, dict]:
    root = _root_plan(root, plan)
    slot = _slot(plan, slot_id)
    receipt = _read(_slot_path(root, slot["ordinal"]))
    require(set(receipt) == RECEIPT_FIELDS
            and receipt["schema"] == "selection_conditional_slot_receipt_v1"
            and receipt["slot_id"] == slot_id and receipt["ordinal"] == slot["ordinal"]
            and receipt["plan_sha256"] == plan["plan_sha256"]
            and receipt["disposition"] in DISPOSITIONS
            and ((receipt["disposition"] == "skipped_terminal" and receipt["terminal_reason"] in TERMINAL_REASONS)
                 or (receipt["disposition"] != "skipped_terminal" and receipt["terminal_reason"] is None
                     and receipt["terminal_receipt_sha256"] is None)), "slot receipt 결합")
    if receipt["terminal_reason"] == "view_terminal":
        require(receipt["terminal_receipt_sha256"] is None, "view terminal receipt 결합")
    elif receipt["disposition"] == "skipped_terminal":
        require(V.S._valid_sha(receipt["terminal_receipt_sha256"]), "외부 terminal receipt 결합")
    capsule_path = root / "capsules" / f"{receipt['capsule_sha256']}.json"
    require(capsule_path.is_file(), "slot capsule 누락")
    capsule = _read(capsule_path)
    require(set(capsule) == CAPSULE_FIELDS and capsule["schema"] == "selection_conditional_capsule_v1"
            and _digest(capsule) == receipt["capsule_sha256"] and capsule["slot"] == slot
            and capsule["disposition"] == receipt["disposition"]
            and capsule["snapshot_digest"] == receipt["snapshot_digest"]
            and capsule["terminal_reason"] == receipt["terminal_reason"]
            and capsule["terminal_receipt_sha256"] == receipt["terminal_receipt_sha256"], "slot capsule 결합")
    if receipt["snapshot_digest"] is None:
        require(receipt["disposition"] == "skipped_epoch_absent"
                and all(capsule[key] is None for key in (
                    "snapshot_inputs", "history_utf8", "history_sha256", "input_digest")), "epoch absent capsule 결합")
    elif receipt["disposition"] == "skipped_terminal":
        require(capsule["input_digest"] is None, "terminal skip은 model input을 만들지 않는다")
    else:
        require(V.S._valid_sha(capsule["input_digest"]), "model input digest 결합")
    return slot, receipt


def _rebuild_snapshot(root: Path, slot: dict, receipt: dict) -> tuple[dict, dict]:
    capsule_path = root / "capsules" / f"{receipt['capsule_sha256']}.json"
    require(capsule_path.is_file(), "slot capsule 누락")
    capsule = _read(capsule_path)
    require(_digest(capsule) == receipt["capsule_sha256"] and capsule["slot"] == slot
            and capsule["disposition"] == receipt["disposition"]
            and capsule["snapshot_digest"] == receipt["snapshot_digest"]
            and capsule["terminal_reason"] == receipt["terminal_reason"]
            and capsule["terminal_receipt_sha256"] == receipt["terminal_receipt_sha256"], "slot capsule 변조")
    snapshot_inputs = capsule["snapshot_inputs"]
    history_utf8 = capsule["history_utf8"]
    require(isinstance(snapshot_inputs, dict) and isinstance(history_utf8, str),
            "snapshot 없는 slot은 대화할 수 없다")
    raw = history_utf8.encode("utf-8")
    require(_raw_sha(raw) == capsule["history_sha256"], "capsule history 변조")
    history = root / "histories" / f"{capsule['history_sha256']}.jsonl"
    require(history.is_file() and history.read_bytes() == raw, "root 상대 동결 history 불일치")
    snapshot = V.build_snapshot(**snapshot_inputs, ledger_path=history)
    require(_digest(_portable_snapshot(snapshot)) == receipt["snapshot_digest"], "portable snapshot digest 불일치")
    if receipt["disposition"] == "skipped_terminal":
        require(capsule["input_digest"] is None, "terminal skip은 model input을 만들지 않는다")
        return snapshot, {}
    inputs = _dialogue_input(snapshot, slot["condition"])
    require(_digest({"view": inputs["view"], "messages": inputs["messages"], "params": inputs["params"]})
            == capsule["input_digest"], "동결 model view 불일치")
    return snapshot, inputs


def audit_api(root: Path, plan: dict) -> dict:
    root = _root_plan(root, plan)
    reservations = sorted((root / "api/reservations").glob("[0-9][0-9][0-9][0-9].json"))
    responses = {path.name: path for path in (root / "api/responses").glob("*.json")}
    require(len(reservations) <= API_CAP, "전역 API 요청 상한 초과")
    counts = {"reserved": len(reservations), "completed": 0, "pending": 0,
              "ok": 0, "error": 0, "timeout": 0}
    rows, attempts = [], {}
    required = {"schema", "request_index", "plan_sha256", "slot_id", "capsule_sha256",
                "snapshot_sha256", "messages_sha256", "attempt", "reserved_at"}
    for index, path in enumerate(reservations, 1):
        reservation = _read(path)
        require(path.name == f"{index:04d}.json" and set(reservation) == required
                and reservation["schema"] == RESERVATION_SCHEMA
                and reservation["request_index"] == index
                and reservation["plan_sha256"] == plan["plan_sha256"]
                and isinstance(reservation["slot_id"], str) and bool(reservation["slot_id"])
                and all(V.S._valid_sha(reservation[key]) for key in (
                    "plan_sha256", "capsule_sha256", "snapshot_sha256", "messages_sha256"))
                and type(reservation["attempt"]) is int and 1 <= reservation["attempt"] <= 3
                and isinstance(reservation["reserved_at"], str) and bool(reservation["reserved_at"]),
                "조건부 API reservation 형식·순서")
        slot, receipt = _load_receipt(root, plan, reservation["slot_id"])
        require(receipt["disposition"] == "executed"
                and reservation["capsule_sha256"] == receipt["capsule_sha256"]
                and reservation["snapshot_sha256"] == receipt["snapshot_digest"], "API reservation slot 결합")
        seen = attempts.setdefault(slot["slot_id"], [])
        seen.append(reservation["attempt"])
        require(seen == list(range(1, len(seen) + 1)) and len(seen) <= slot["max_requests"], "slot attempt 순서")
        response_path = responses.pop(path.name, None)
        response = _read(response_path) if response_path else None
        if response is None:
            counts["pending"] += 1
        else:
            expected = {key: reservation[key] for key in required - {"schema", "reserved_at"}}
            allowed_result = {"response_id", "provider_model", "finish_reason", "content", "reasoning_content",
                              "input_tokens", "output_tokens", "total_tokens", "wall_s", "error_type",
                              "infra_error", "status_code", "response_contract_fields"}
            require(set(response) == {"schema", *expected, "reservation_sha256", "status", "result", "finished_at"}
                    and response["schema"] == RESPONSE_SCHEMA
                    and all(response.get(key) == value for key, value in expected.items())
                    and response["reservation_sha256"] == MC.digest(reservation)
                    and response["status"] in {"ok", "error", "timeout"}
                    and isinstance(response["result"], dict)
                    and set(response["result"]) <= allowed_result
                    and all(response["result"].get(key) is None
                            or (isinstance(response["result"][key], str)
                                and bool(_EXCEPTION.fullmatch(response["result"][key])))
                            for key in ("error_type", "infra_error"))
                    and response["status"] == C.response_status(response["result"])
                    and isinstance(response["finished_at"], str) and bool(response["finished_at"]),
                    "조건부 API response 결합")
            counts["completed"] += 1
            counts[response["status"]] += 1
        rows.append({"reservation": reservation, "response": response})
    require(not responses, "reservation 없는 조건부 API response")
    return {"schema": "selection_conditional_api_audit_v1", "cap": API_CAP, **counts,
            "unknown_consumed": counts["pending"], "remaining": API_CAP - counts["reserved"], "rows": rows}


def _reserve(root: Path, plan: dict, receipt: dict, messages: list, attempt: int) -> tuple[Path, dict]:
    audit = audit_api(root, plan)
    if audit["pending"]:
        raise MC.BudgetStop("미완료 요청은 unknown 소비이며 새 요청을 막는다")
    if audit["remaining"] <= 0:
        raise MC.BudgetStop("전역 60-request 상한")
    index = audit["reserved"] + 1
    reservation = {
        "schema": RESERVATION_SCHEMA,
        "request_index": index,
        "plan_sha256": plan["plan_sha256"],
        "slot_id": receipt["slot_id"],
        "capsule_sha256": receipt["capsule_sha256"],
        "snapshot_sha256": receipt["snapshot_digest"],
        "messages_sha256": V.L.digest(messages),
        "attempt": attempt,
        "reserved_at": MC.now(),
    }
    path = root / "api/reservations" / f"{index:04d}.json"
    _write_new(path, reservation)
    audit_api(root, plan)
    return path, reservation


class FixtureProvider:
    def __init__(self, rows: list[dict | Exception]):
        require(isinstance(rows, list), "fixture provider rows")
        self.rows = iter(deepcopy(rows))
        self.sent: list[tuple[list, dict]] = []

    def __call__(self, messages: list, params: dict) -> dict:
        self.sent.append((deepcopy(messages), deepcopy(params)))
        row = next(self.rows)
        if isinstance(row, Exception):
            raise row
        require(isinstance(row, dict), "fixture provider 결과")
        return deepcopy(row)


class NvidiaProvider:
    def __init__(self) -> None:
        require(sys.platform.startswith("linux") and bool(os.environ.get("WSL_DISTRO_NAME")),
                "실제 provider worker는 WSL에서만 실행")

    def __call__(self, messages: list, params: dict) -> dict:
        return C.F.openai_complete(messages, params)


class _Meter:
    def __init__(self, root: Path, plan: dict, receipt: dict, provider: FixtureProvider | NvidiaProvider):
        self.root, self.plan, self.receipt, self.provider = root, plan, receipt, provider
        self.safe_results: list[dict] = []
        self.receipts: list[dict] = []

    def __call__(self, messages: list, params: dict) -> dict:
        require(params == C.F.request_params(MODEL), "provider params drift")
        attempt = len(self.safe_results) + 1
        require(attempt <= 3, "logical engine request 상한")
        with MC._active_lease(self.root / "api"):
            path, reservation = _reserve(self.root, self.plan, self.receipt, messages, attempt)
            started = time.perf_counter()
            try:
                result = self.provider(deepcopy(messages), deepcopy(params))
                require(isinstance(result, dict), "provider 결과는 객체여야 한다")
                safe = C.safe_result(result)
            except Exception as error:  # noqa: BLE001 - provider 경계는 예외 class만 영속한다.
                name = type(error).__name__
                require(bool(_EXCEPTION.fullmatch(name)), "provider 예외 class")
                safe = {"error_type": name, "wall_s": time.perf_counter() - started}
                if MC._retryable_exception(error):
                    safe["infra_error"] = name
            safe.setdefault("wall_s", time.perf_counter() - started)
            response = {
                "schema": RESPONSE_SCHEMA,
                **{key: reservation[key] for key in set(reservation) - {"schema", "reserved_at"}},
                "reservation_sha256": MC.digest(reservation),
                "status": C.response_status(safe),
                "result": deepcopy(safe),
                "finished_at": MC.now(),
            }
            response_path = self.root / "api/responses" / path.name
            _write_new(response_path, response)
        self.safe_results.append(deepcopy(safe))
        self.receipts.append({"request_index": reservation["request_index"],
                              "reservation_sha256": MC.digest(reservation),
                              "response_sha256": MC.file_sha(response_path)})
        return safe


def run_dialogue(root: Path, plan: dict, slot_receipt: dict, provider: FixtureProvider | NvidiaProvider) -> dict:
    root = _root_plan(root, plan)
    slot, stored = _load_receipt(root, plan, slot_receipt.get("slot_id", ""))
    require(slot_receipt == stored and stored["disposition"] == "executed", "executed slot receipt 필요")
    output = root / "dialogues" / f"{slot['ordinal']:02d}.json"
    require(not output.exists(), "slot dialogue duplicate")
    prior = [row for row in audit_api(root, plan)["rows"] if row["reservation"]["slot_id"] == slot["slot_id"]]
    require(not prior, "dialogue 없는 slot의 기존 API 소비")
    snapshot, inputs = _rebuild_snapshot(root, slot, stored)
    require(inputs["needs_model"], "no-model slot 실행 거부")
    require(type(provider) in {FixtureProvider, NvidiaProvider}, "등록 provider만 허용")
    meter = _Meter(root, plan, stored, provider)
    engine = C.engine(snapshot, inputs["view"], inputs["messages"], slot["condition"], meter)
    record = {
        "schema": "selection_conditional_dialogue_v1",
        "slot_id": slot["slot_id"],
        "plan_sha256": plan["plan_sha256"],
        "capsule_sha256": stored["capsule_sha256"],
        "input_digest": _digest({"view": inputs["view"], "messages": inputs["messages"],
                                 "params": inputs["params"]}),
        "safe_results": meter.safe_results,
        "request_receipts": meter.receipts,
        "engine": engine,
    }
    require(1 <= engine["requests"] <= slot["max_requests"], "logical dialogue 요청 수")
    _write_new(output, record)
    return review_dialogue(root, plan, stored)


def review_dialogue(root: Path, plan: dict, slot_receipt: dict) -> dict:
    root = _root_plan(root, plan)
    slot, stored = _load_receipt(root, plan, slot_receipt.get("slot_id", ""))
    require(slot_receipt == stored and stored["disposition"] == "executed", "review slot receipt")
    snapshot, inputs = _rebuild_snapshot(root, slot, stored)
    path = root / "dialogues" / f"{slot['ordinal']:02d}.json"
    record = _read(path)
    require(set(record) == {"schema", "slot_id", "plan_sha256", "capsule_sha256", "input_digest",
                            "safe_results", "request_receipts", "engine"}
            and record["schema"] == "selection_conditional_dialogue_v1"
            and record["slot_id"] == slot["slot_id"]
            and record["plan_sha256"] == plan["plan_sha256"]
            and record["capsule_sha256"] == stored["capsule_sha256"]
            and record["input_digest"] == _digest({"view": inputs["view"], "messages": inputs["messages"],
                                                   "params": inputs["params"]}), "dialogue record 결합")
    rows = [row for row in audit_api(root, plan)["rows"] if row["reservation"]["slot_id"] == slot["slot_id"]]
    require(not any(row["response"] is None for row in rows)
            and len(rows) == len(record["safe_results"]) == len(record["request_receipts"]), "dialogue API 원장 완결")
    for attempt, (row, safe, receipt) in enumerate(
            zip(rows, record["safe_results"], record["request_receipts"], strict=True), 1):
        require(row["reservation"]["attempt"] == attempt
                and row["response"]["result"] == safe
                and receipt == {"request_index": row["reservation"]["request_index"],
                                "reservation_sha256": MC.digest(row["reservation"]),
                                "response_sha256": MC.file_sha(root / "api/responses" /
                                                                f"{row['reservation']['request_index']:04d}.json")},
                "dialogue request receipt 결합")
    fixture = C.FixtureResponses(record["safe_results"])
    expected = C.engine(snapshot, inputs["view"], inputs["messages"], slot["condition"], fixture)
    require(expected == record["engine"] and len(fixture.sent) == len(rows)
            and all(row["reservation"]["messages_sha256"] == V.L.digest(sent[0])
                    for row, sent in zip(rows, fixture.sent, strict=True)), "engine replay equality")
    return deepcopy(record)


def finalize(root: Path, plan: dict) -> dict:
    root = _root_plan(root, plan)
    receipts = []
    for slot in plan["slots"]:
        path = _slot_path(root, slot["ordinal"])
        require(path.is_file(), "20 slot 전부 닫혀야 한다")
        _slot_value, receipt = _load_receipt(root, plan, slot["slot_id"])
        if receipt["snapshot_digest"] is not None:
            _rebuild_snapshot(root, slot, receipt)
        receipts.append(receipt)
    audit = audit_api(root, plan)
    if audit["pending"]:
        return {"schema": "selection_conditional_finalize_v1", "status": "invalid",
                "reason": "api_pending_unknown_consumed", "slots_closed": len(receipts),
                "api": {key: audit[key] for key in ("reserved", "completed", "pending", "unknown_consumed")},
                "costs": {key: None for key in ("input_tokens", "output_tokens", "total_tokens", "wall_s")}}
    executed = [row for row in receipts if row["disposition"] == "executed"]
    for receipt in executed:
        review_dialogue(root, plan, receipt)
    dialogue_slots = {path.stem for path in (root / "dialogues").glob("[0-9][0-9].json")}
    require(dialogue_slots == {f"{row['ordinal']:02d}" for row in executed}, "dialogue inventory 불일치")
    responses = [row["response"]["result"] for row in audit["rows"]]
    costs = {key: (sum(row[key] for row in responses)
                   if all(isinstance(row.get(key), (int, float)) and not isinstance(row.get(key), bool)
                          and row[key] >= 0 for row in responses) else None)
             for key in ("input_tokens", "output_tokens", "total_tokens", "wall_s")}
    return {"schema": "selection_conditional_finalize_v1", "status": "complete",
            "slots_closed": len(receipts), "executed": len(executed),
            "dispositions": {name: sum(row["disposition"] == name for row in receipts) for name in sorted(DISPOSITIONS)},
            "api": {key: audit[key] for key in ("reserved", "completed", "pending", "ok", "error", "timeout")},
            "costs": costs}


def verify_worker_binding(plan: dict, expected_plan_sha: str, expected_source_sha: str,
                          protocol: Path, expected_protocol_sha: str) -> None:
    validate_plan(plan)
    require(plan["plan_sha256"] == expected_plan_sha, "root demo plan SHA 결합")
    require(V.file_digest(Path(__file__)) == expected_source_sha, "root demo caller source SHA 결합")
    protocol = Path(protocol)
    require(protocol.is_file() and V.file_digest(protocol) == expected_protocol_sha,
            "root demo protocol SHA 결합")


def _load_plan(path: Path) -> dict:
    return validate_plan(_read(path))


def _bound(args, plan: dict) -> None:
    verify_worker_binding(plan, args.expected_plan_sha, args.expected_caller_source_sha,
                          args.protocol, args.expected_protocol_sha)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    make_plan = commands.add_parser("plan")
    make_plan.add_argument("--output", type=Path, required=True)
    for name in ("materialize", "execute", "audit", "review", "finalize"):
        command = commands.add_parser(name)
        command.add_argument("--root", type=Path, required=True)
        command.add_argument("--plan", type=Path, required=True)
        command.add_argument("--expected-plan-sha", required=True)
        command.add_argument("--expected-caller-source-sha", required=True)
        command.add_argument("--protocol", type=Path, required=True)
        command.add_argument("--expected-protocol-sha", required=True)
        if name in {"materialize", "execute", "review"}:
            command.add_argument("--slot-id", required=True)
        if name == "materialize":
            command.add_argument("--snapshot", type=Path)
            command.add_argument("--epoch-absent", action="store_true")
            command.add_argument("--terminal-reason", choices=sorted(TERMINAL_REASONS - {"view_terminal"}))
            command.add_argument("--terminal-receipt-sha")
        if name == "execute":
            command.add_argument("--approve-model-call", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "plan":
        _write_new(args.output, plan_value())
        return 0
    plan = _load_plan(args.plan)
    _bound(args, plan)
    initialize(args.root, plan)
    if args.command == "materialize":
        require(args.epoch_absent != bool(args.snapshot), "snapshot 또는 epoch-absent 중 하나 필요")
        snapshot = None if args.epoch_absent else _read(args.snapshot)
        if args.terminal_reason is not None:
            if snapshot is None or not isinstance(args.terminal_receipt_sha, str):
                raise ValueError("terminal skip에는 snapshot과 receipt SHA 필요")
            materialized = materialize_skipped_slot(
                _slot(plan, args.slot_id), snapshot, args.terminal_reason, args.terminal_receipt_sha)
        else:
            require(args.terminal_receipt_sha is None, "terminal reason 없는 receipt SHA")
            materialized = materialize_slot(_slot(plan, args.slot_id), snapshot)
        result = record_slot(args.root, plan, materialized)
    elif args.command == "execute":
        require(args.approve_model_call, "실제 모델 호출 명시 승인 필요")
        _slot_value, receipt = _load_receipt(args.root, plan, args.slot_id)
        result = run_dialogue(args.root, plan, receipt, NvidiaProvider())
    elif args.command == "audit":
        result = audit_api(args.root, plan)
    elif args.command == "review":
        _slot_value, receipt = _load_receipt(args.root, plan, args.slot_id)
        result = review_dialogue(args.root, plan, receipt)
    else:
        result = finalize(args.root, plan)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
