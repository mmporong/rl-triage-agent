"""R4 epoch-2 완료 영수증의 public/backend snapshot 경로를 분리한다."""

from __future__ import annotations

import sys
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from threading import RLock
from types import ModuleType
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("evals", "src")]

import five_policy_recovery_candidate_v1 as FIRST
import five_policy_recovery_epoch2_v1 as SECOND

SOURCE_FILES = (
    "evals/recovery_receipt_snapshot_routes_v1.py",
    "tests/test_recovery_receipt_snapshot_routes_v1.py",
)
_LOCK = RLock()
_BASE_HOST = SECOND._HOST
_BASE_SECOND_CONTROLLER = SECOND._DRIVER.C
_BASE_FIRST_CONTROLLER = FIRST.DRIVER.C
_BASE_SNAPSHOT = FIRST._snapshot


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def _facade(module: ModuleType, name: str) -> ModuleType:
    value = ModuleType(name)
    value.__dict__.update(module.__dict__)
    value.__name__ = name
    return value


def _registrations(
    sources: Sequence[dict[str, Any]],
    parents: Mapping[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    require(1 <= len(sources) <= 4, "R4 receipt source 수 범위")
    targets = {f"{source.get('case_id')}:{source.get('policy')}" for source in sources}
    require(len(targets) == len(sources) and set(parents) == targets,
            "R4 receipt source/parent target 결합")
    result: dict[str, dict[str, Any]] = {}
    for source in sources:
        case_id, arm = source.get("case_id"), source.get("policy")
        require((case_id, arm) in SECOND.TARGETS, "등록되지 않은 R4 receipt target")
        case_id, arm = cast(str, case_id), cast(str, arm)
        packet = source.get("packet")
        require(isinstance(packet, dict) and packet.get("epoch") == 2,
                "receipt public packet은 epoch-2여야 한다")
        packet = cast(dict[str, Any], packet)
        prior = packet.get("prior_receipt")
        require(isinstance(prior, dict) and prior.get("verdict") == "not_recovered",
                "receipt public packet prior가 비회복이 아니다")
        prior = cast(dict[str, Any], prior)
        run_id = f"r4_{case_id.replace('-', '_')}_{arm}"
        policy = SECOND._DRIVER._storage_policy(arm)
        parent = deepcopy(parents[f"{case_id}:{arm}"])
        expected = {"run_id": run_id, "policy": policy, "epoch": 1, **deepcopy(prior)}
        require(parent == expected, "receipt public parent가 source/arm과 다르다")
        require(
            parent.get("checkpoint_sha256")
            == packet.get("input_bindings", {}).get("case", {}).get("checkpoint_sha256"),
            "receipt public parent checkpoint가 case input과 다르다",
        )
        packet_digest = SECOND.digest(packet)
        require(packet_digest not in result, "서로 다른 receipt target의 public packet이 같다")
        result[packet_digest] = {
            "target": f"{case_id}:{arm}",
            "packet": deepcopy(packet),
            "run_id": run_id,
            "policy": policy,
            "parent": parent,
        }
    return result


def _registered(
    registrations: Mapping[str, dict[str, Any]],
    packet: dict[str, Any],
) -> dict[str, Any] | None:
    value = registrations.get(SECOND.digest(packet))
    if value is None:
        return None
    require(packet == value["packet"], "receipt public packet 원문이 바뀌었다")
    return value


def _validate_call(
    registration: Mapping[str, Any],
    run_id: str,
    policy: str,
    protocol_sha256: str,
    parent: dict[str, Any] | None,
    view: ModuleType,
) -> dict[str, Any]:
    require(run_id == registration["run_id"], "receipt public run_id가 target과 다르다")
    require(policy == registration["policy"], "receipt public policy가 arm과 다르다")
    expected_protocol = view.file_digest(view.PROTOCOL)
    require(protocol_sha256 == expected_protocol, "receipt public selection protocol이 다르다")
    expected_parent = cast(dict[str, Any], registration["parent"])
    require(parent is None or parent == expected_parent, "receipt public parent가 등록 계보와 다르다")
    return deepcopy(expected_parent)


def _epoch2_snapshot(
    packet: dict[str, Any],
    result: dict[str, Any],
    ledger_path: Path,
    events: list[dict[str, Any]],
    registration: Mapping[str, Any],
    view: ModuleType,
) -> dict[str, Any]:
    policy = result["host_storage_policy"]
    protocol_sha = view.file_digest(view.PROTOCOL)
    parent = _validate_call(
        registration, result["run_id"], policy, protocol_sha, None, view)
    binding = view.epoch_binding(packet, result["run_id"], policy, protocol_sha, parent)
    rows = FIRST._rows(events)
    rendered = view._view(packet, rows, None, False)
    pre = {
        "binding": binding,
        "prior_observations": rows[:-1],
        "executed_probe": rows[-1]["probe"] if rows else None,
    }
    inputs = {
        "packet": packet,
        "run_id": result["run_id"],
        "policy": policy,
        "protocol_sha256": protocol_sha,
        "ledger_path": str(ledger_path),
        "parent": parent,
        "masked": False,
    }
    return {
        "schema": "selection_snapshot_v1",
        "inputs": inputs,
        "binding": binding,
        "ledger_digest": FIRST.DRIVER.C.L.digest(events),
        "pre_receipt_anchor": FIRST.DRIVER.C.L.digest(pre),
        "view": rendered,
        "view_digest": FIRST.DRIVER.C.L.digest(rendered),
        "observation_masked_control": False,
        "proposal_allowed": rendered["status"] == "open",
        "execution_authorized": False,
        "approval_granted": False,
    }


@contextmanager
def receipt_snapshot_routes_scope(
    sources: Sequence[dict[str, Any]],
    *,
    parents: Mapping[str, dict[str, Any]],
) -> Iterator[dict[str, dict[str, Any]]]:
    """활성 parent/probe scope에 public receipt snapshot 경로만 추가한다."""
    with _LOCK:
        require(SECOND._DRIVER.C is not _BASE_SECOND_CONTROLLER,
                "epoch-2 parent binding scope가 활성 상태가 아니다")
        require(SECOND._HOST is not _BASE_HOST,
                "epoch-2 probe boundary scope가 활성 상태가 아니다")
        registrations = _registrations(sources, parents)
        host = SECOND._HOST
        original_host_controller = host.C
        original_second_controller = SECOND._DRIVER.C
        original_first_controller = FIRST.DRIVER.C
        original_snapshot = FIRST._snapshot
        host_build = original_host_controller.V.build_snapshot
        second_build = original_second_controller.V.build_snapshot

        driver_view = _facade(
            original_first_controller.V, "_recovery_receipt_driver_view_v1")
        driver_controller = _facade(
            original_first_controller, "_recovery_receipt_driver_controller_v1")

        def driver_snapshot(
            packet: dict,
            run_id: str,
            policy: str,
            protocol_sha256: str,
            ledger_path: str | Path,
            parent: dict | None = None,
            masked: bool = False,
        ) -> dict:
            registration = _registered(registrations, packet)
            if registration is None:
                require(packet.get("epoch") != 2, "미등록 epoch-2 public packet")
                return original_first_controller.V.build_snapshot(
                    packet, run_id, policy, protocol_sha256, ledger_path,
                    parent=parent, masked=masked)
            expected_parent = _validate_call(
                registration, run_id, policy, protocol_sha256, parent,
                original_first_controller.V)
            return second_build(
                packet, run_id, policy, protocol_sha256, ledger_path,
                parent=expected_parent, masked=masked)

        driver_view.__dict__["build_snapshot"] = driver_snapshot
        driver_controller.__dict__["V"] = driver_view

        host_view = _facade(original_host_controller.V, "_recovery_receipt_host_view_v1")
        host_controller = _facade(
            original_host_controller, "_recovery_receipt_host_controller_v1")

        def host_snapshot(
            packet: dict,
            run_id: str,
            policy: str,
            protocol_sha256: str,
            ledger_path: str | Path,
            parent: dict | None = None,
            masked: bool = False,
        ) -> dict:
            registration = _registered(registrations, packet)
            if registration is None:
                return host_build(
                    packet, run_id, policy, protocol_sha256, ledger_path,
                    parent=parent, masked=masked)
            expected_parent = _validate_call(
                registration, run_id, policy, protocol_sha256, parent,
                original_first_controller.V)
            return driver_snapshot(
                packet, run_id, policy, protocol_sha256, ledger_path,
                parent=expected_parent, masked=masked)

        host_view.__dict__["build_snapshot"] = host_snapshot
        host_controller.__dict__["V"] = host_view

        def candidate_snapshot(
            packet: dict[str, Any],
            result: dict[str, Any],
            ledger_path: Path,
            events: list[dict[str, Any]],
        ) -> dict[str, Any]:
            registration = _registered(registrations, packet)
            if registration is None:
                require(packet.get("epoch") != 2, "미등록 epoch-2 candidate packet")
                return original_snapshot(packet, result, ledger_path, events)
            return _epoch2_snapshot(
                packet, result, Path(ledger_path), events,
                registration, original_first_controller.V)

        FIRST.DRIVER.__dict__["C"] = driver_controller
        host.__dict__["C"] = host_controller
        FIRST.__dict__["_snapshot"] = candidate_snapshot
        try:
            yield {
                row["target"]: {
                    "run_id": row["run_id"],
                    "policy": row["policy"],
                    "parent": deepcopy(row["parent"]),
                }
                for row in registrations.values()
            }
        finally:
            current = (FIRST.DRIVER.C, host.C, FIRST._snapshot)
            FIRST.DRIVER.__dict__["C"] = original_first_controller
            host.__dict__["C"] = original_host_controller
            FIRST.__dict__["_snapshot"] = original_snapshot
            if current != (driver_controller, host_controller, candidate_snapshot):
                raise RuntimeError("receipt snapshot route가 scope 중 외부에서 바뀌었다")
