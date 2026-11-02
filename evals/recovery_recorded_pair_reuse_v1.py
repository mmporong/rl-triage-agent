"""검증된 paired runtime 측정을 새 public logical ledger에 한 번 재생한다."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from threading import RLock
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("evals", "src")]

import five_policy_recovery_epoch2_v1 as SECOND

SCHEMA = "recovery_recorded_pair_reuse_receipt_v1"
SOURCE_FILES = (
    "evals/recovery_recorded_pair_reuse_v1.py",
    "tests/test_recovery_recorded_pair_reuse_v1.py",
)
ROLES = ("reference", "case")
_LOCK = RLock()


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def _snapshot_without_ledger(snapshot: dict[str, Any]) -> dict[str, Any]:
    value = deepcopy(snapshot)
    inputs = value.get("inputs")
    require(isinstance(inputs, dict), "snapshot inputs 형식")
    inputs = cast(dict[str, Any], inputs)
    require(isinstance(inputs.get("ledger_path"), str), "snapshot ledger path 형식")
    inputs["ledger_path"] = "<recorded-pair-public-ledger>"
    return value


def _call(arguments: tuple[Any, ...], keywords: dict[str, Any]) -> tuple[dict, dict, dict]:
    require(len(arguments) <= 2, "execute_pair positional 인자 범위")
    snapshot = arguments[0] if arguments else keywords.get("snapshot")
    choice = arguments[1] if len(arguments) == 2 else keywords.get("choice")
    require(isinstance(snapshot, dict) and isinstance(choice, dict), "snapshot/choice 형식")
    return cast(dict, snapshot), cast(dict, choice), keywords


def _ref(host: Any, path: Path) -> dict[str, str]:
    return cast(dict[str, str], host.file_ref(Path(path)))


def _verified_roles(host: Any, result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    verified: dict[str, dict[str, Any]] = {}
    for role in ROLES:
        plan = Path(result["plans"][role]["path"])
        receipt = Path(result["receipts"][role]["path"])
        value = host.OBS.verify_receipt(plan, receipt)
        raw = host.read(receipt)
        execution = raw.get("execution")
        require(value.get("status") == "complete" and isinstance(value.get("measurement"), dict),
                f"old {role} physical receipt가 complete가 아니다")
        require(isinstance(execution, dict)
                and execution.get("exit_code") == 0
                and execution.get("natural_exit") is True
                and execution.get("cleanup_verified") is True
                and execution.get("timed_out") is False,
                f"old {role} physical 실행 종료/cleanup이 완전하지 않다")
        verified[role] = deepcopy(value)
    return verified


def _validate_call(host: Any, old_result: dict[str, Any], old_snapshot: dict[str, Any],
                   old_choice: dict[str, Any], snapshot: dict[str, Any], choice: dict[str, Any],
                   keywords: dict[str, Any]) -> tuple[dict[str, Any], Path, Path]:
    old_binding = old_snapshot.get("binding", {})
    new_binding = snapshot.get("binding", {})
    same_target = (
        isinstance(old_binding, dict) and isinstance(new_binding, dict)
        and old_binding.get("run_id") == new_binding.get("run_id")
        and old_binding.get("policy") == new_binding.get("policy")
    )
    if not same_target:
        return {}, Path(), Path()
    require(old_result.get("probe") == old_choice.get("probe") == choice.get("probe") == "P_episode",
            "recorded pair probe/choice가 다르다")
    require(choice == old_choice, "recorded pair choice가 다르다")
    require(_snapshot_without_ledger(snapshot) == _snapshot_without_ledger(old_snapshot),
            "recorded pair snapshot view/source/parent/protocol이 다르다")
    old_spec = host.C._frozen_request_spec(old_snapshot, old_choice)
    new_spec = host.C.request_spec(snapshot, choice)
    require(new_spec == old_spec and host.digest(new_spec) == old_result["request_spec_sha256"],
            "recorded pair request args/checkpoint/prior receipt가 다르다")
    backend_packet = keywords.get("backend_packet")
    require(isinstance(backend_packet, dict)
            and host.digest(backend_packet) == old_result.get("backend_packet_sha256"),
            "recorded pair backend source가 다르다")
    probe_protocol = Path(keywords.get("probe_protocol", "")).resolve()
    view_protocol = Path(keywords.get("view_protocol", "")).resolve()
    require(_ref(host, probe_protocol) == old_result.get("probe_protocol")
            and _ref(host, view_protocol) == old_result.get("view_protocol")
            and keywords.get("anchor") == old_result.get("view_anchor"),
            "recorded pair protocol/anchor가 다르다")
    lab = Path(keywords.get("lab", "")).resolve()
    inputs = {"case": keywords.get("case_input"), "reference": keywords.get("reference_input")}
    for role in ROLES:
        plan = host.read(Path(old_result["plans"][role]["path"]))
        require(Path(inputs[role]).resolve() == Path(plan["collected_input"]["path"]).resolve(),
                f"recorded pair {role} source가 다르다")
        require(lab == Path(plan["lab"]).resolve(), "recorded pair lab source가 다르다")
    ledger = Path(snapshot["inputs"]["ledger_path"]).resolve()
    out = Path(keywords.get("out", "")).resolve()
    require(not out.exists(), "recorded pair output은 fresh여야 한다")
    require(not ledger.exists() or (ledger.is_file() and ledger.stat().st_size == 0),
            "recorded pair public ledger는 fresh/empty여야 한다")
    return new_spec, ledger, out


def _replay(host: Any, original_result: Path, original_context: Path,
            original_choice: Path, old_result: dict[str, Any], old_snapshot: dict[str, Any],
            old_choice: dict[str, Any], snapshot: dict[str, Any], choice: dict[str, Any],
            keywords: dict[str, Any], spec: dict[str, Any], ledger_path: Path,
            out: Path) -> dict[str, Any]:
    verified_roles = _verified_roles(host, old_result)
    approver = keywords.get("approver", "codex-main-user-delegated")
    require(isinstance(approver, str) and bool(approver.strip()), "approver")
    ledger, request_id = host._ledger_start(ledger_path, spec, approver, fresh=False)
    outcome, receipt = host.DIAG._settle(
        ledger, request_id, spec["probe"], verified_roles, old_result["parent_wall_s"])
    require(outcome == old_result["outcome"], "recorded pair logical settlement가 달라졌다")
    done = host._done(
        ledger_path, snapshot["binding"], snapshot["inputs"]["packet"], request_id, spec,
        prior=snapshot["view"]["observations"], outcome=outcome, allow_suffix=False)
    events = [row for row in ledger.events() if row.get("request_id") == request_id]
    require([row.get("event") for row in events]
            == ["proposed", "approved", "consumed", "receipt"],
            "recorded pair logical ledger event가 네 개가 아니다")

    out.mkdir(parents=True, exist_ok=False)
    reuse_receipt = {
        "schema": SCHEMA,
        "status": "complete",
        "physical_execution_reused": True,
        "model_response_reused": False,
        "new_sdk_execution_count": 0,
        "prior_physical_roles": 2,
        "new_physical_roles": 0,
        "source_result": _ref(host, original_result),
        "source_context": _ref(host, original_context),
        "source_choice": _ref(host, original_choice),
        "old_public_ledger": deepcopy(old_result["public_ledger"]),
        "new_public_ledger": deepcopy(done),
        "old_snapshot_sha256": host.digest(old_snapshot),
        "new_snapshot_sha256": host.digest(snapshot),
        "choice_sha256": host.digest(choice),
        "logical_events": deepcopy(events),
        "logical_event_count": 4,
        "measurement_source": "verified_original_physical_receipts",
        "charged_parent_wall_s": old_result["parent_wall_s"],
        "charged_cost_scope": "original_measurement_reused_once_not_new_gpu_time",
        "original_backend_ledger": deepcopy(old_result["backend_ledger"]),
        "original_backend_ledger_receipt": deepcopy(old_result["backend_ledger_receipt"]),
        "original_authorization": deepcopy(old_result["authorization"]),
        "original_plans": deepcopy(old_result["plans"]),
        "original_receipts": deepcopy(old_result["receipts"]),
    }
    reuse_path = out / "reuse-receipt.json"
    host.write_new(reuse_path, reuse_receipt)

    result = deepcopy(old_result)
    result.update(
        public_snapshot_sha256=host.digest(snapshot),
        request_id=request_id,
        public_ledger=done,
        public_ledger_receipt=receipt,
        reused_original_result=_ref(host, original_result),
        physical_execution_reused=True,
        prior_physical_roles=2,
        new_physical_roles=0,
        new_sdk_execution_count=0,
        reuse_receipt=_ref(host, reuse_path),
    )
    result_path = out / "result.json"
    host.write_new(result_path, result)
    host.verify_pair(result_path, snapshot, choice)
    returned = deepcopy(result)
    returned["result_ref"] = _ref(host, result_path)
    return returned


@contextmanager
def recorded_pair_reuse_scope(*, original_result: Path, original_context: Path,
                              original_choice: Path) -> Iterator[dict[str, Any]]:
    """exact 첫 P_episode에만 검증된 physical pair를 새 public ledger로 재생한다."""
    with _LOCK:
        result_path = Path(original_result).resolve(strict=True)
        context_path = Path(original_context).resolve(strict=True)
        choice_path = Path(original_choice).resolve(strict=True)
        host = SECOND._HOST
        old_context = host.read(context_path)
        old_snapshot = old_context.get("snapshot")
        old_choice = host.read(choice_path)
        require(isinstance(old_snapshot, dict), "original context snapshot 형식")
        old_snapshot = cast(dict[str, Any], old_snapshot)
        old_result = host.verify_pair(result_path, old_snapshot, old_choice)
        require(old_result.get("status") == "complete" and old_result.get("outcome") != "unknown",
                "original pair는 complete physical 측정이어야 한다")
        _verified_roles(host, old_result)
        original_execute = SECOND.execute_pair
        used = False

        def execute(*arguments: Any, **keywords: Any) -> dict[str, Any]:
            nonlocal used
            snapshot, choice, call_keywords = _call(arguments, keywords)
            spec, ledger, out = _validate_call(
                host, old_result, old_snapshot, old_choice, snapshot, choice, call_keywords)
            if not spec:
                return cast(dict[str, Any], original_execute(*arguments, **keywords))
            require(not used, "recorded physical pair는 scope에서 한 번만 재사용할 수 있다")
            used = True
            return _replay(
                host, result_path, context_path, choice_path, old_result, old_snapshot, old_choice,
                snapshot, choice, call_keywords, spec, ledger, out)

        SECOND.__dict__["execute_pair"] = execute
        try:
            yield {
                "source_result": _ref(host, result_path),
                "source_context": _ref(host, context_path),
                "source_choice": _ref(host, choice_path),
                "physical_roles_available": 2,
                "new_sdk_execution_count": 0,
            }
        finally:
            current = SECOND.execute_pair
            SECOND.__dict__["execute_pair"] = original_execute
            if current is not execute:
                raise RuntimeError("recorded pair reuse scope가 외부에서 바뀌었다")
