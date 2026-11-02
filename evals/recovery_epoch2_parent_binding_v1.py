"""R4 epoch-2 진단의 구조적 parent를 기존 snapshot 경로에 결합한다.

parent는 계보를 나타내는 구조적 metadata다. 실제 receipt 권한과 원문 인증은
기존 source/controller 검증의 책임이며 이 adapter는 그 검증을 대체하지 않는다.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from threading import RLock
from types import ModuleType
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("evals", "src")]

import five_policy_recovery_epoch2_v1 as SECOND

MAX_TARGETS = 4
_LOCK = RLock()
_BASE_CONTROLLER = SECOND._DRIVER.C


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def _clone(module: ModuleType, name: str) -> ModuleType:
    value = ModuleType(name)
    value.__dict__.update(module.__dict__)
    value.__name__ = name
    return value


def _run_id(case_id: str, arm: str) -> str:
    return f"r4_{case_id.replace('-', '_')}_{arm}"


def _registration(source: dict[str, Any]) -> dict[str, Any]:
    checked = SECOND.checked_source(deepcopy(source))
    require(checked == source, "등록 source packet이 원문과 다르다")
    case_id, arm = checked["case_id"], checked["policy"]
    require((case_id, arm) in SECOND.TARGETS, "등록되지 않은 R4 target")

    packet = checked["packet"]
    prior = packet.get("prior_receipt")
    binding = checked.get("source_binding")
    inputs = checked.get("inputs")
    require(packet.get("epoch") == 2, "epoch-2 prior receipt가 없다")
    require(isinstance(prior, dict), "epoch-2 prior receipt가 없다")
    prior = cast(dict[str, Any], prior)
    require(prior.get("verdict") == "not_recovered", "비회복 receipt가 아니다")
    require(isinstance(binding, dict) and isinstance(inputs, dict), "source 결합이 없다")
    binding = cast(dict[str, Any], binding)
    inputs = cast(dict[str, Any], inputs)
    first_receipt = inputs.get("first_receipt")
    require(isinstance(first_receipt, dict), "first receipt ref가 없다")
    first_receipt = cast(dict[str, Any], first_receipt)
    require(
        binding.get("first_receipt_sha256")
        == first_receipt.get("sha256")
        == prior.get("result_sha256"),
        "source/packet first receipt SHA가 다르다",
    )
    case_checkpoint = packet["input_bindings"]["case"]["checkpoint_sha256"]
    require(
        binding.get("case_checkpoint_sha256")
        == case_checkpoint
        == prior.get("checkpoint_sha256"),
        "source/packet prior checkpoint가 다르다",
    )
    require(
        checked.get("first_correction")
        == prior.get("correction_id")
        == packet["applied_corrections"][0],
        "source/packet first correction이 다르다",
    )

    run_id = _run_id(case_id, arm)
    policy = SECOND._DRIVER._storage_policy(arm)
    parent = {"run_id": run_id, "policy": policy, "epoch": 1, **deepcopy(prior)}
    return {
        "target": f"{case_id}:{arm}",
        "case_id": case_id,
        "arm": arm,
        "run_id": run_id,
        "policy": policy,
        "packet": deepcopy(packet),
        "packet_digest": SECOND.digest(packet),
        "parent": parent,
    }


def parent_binding(source: dict[str, Any], run_id: str, policy: str) -> dict[str, Any]:
    """등록 source에서 host가 공급할 epoch-1 parent metadata를 만든다."""
    registration = _registration(source)
    require(run_id == registration["run_id"], "다른 case 또는 arm의 run_id")
    require(policy == registration["policy"], "다른 arm의 storage policy")
    return deepcopy(registration["parent"])


@contextmanager
def epoch2_snapshot_scope(
    sources: Sequence[dict[str, Any]],
) -> Iterator[dict[str, dict[str, Any]]]:
    """격리된 SECOND driver의 snapshot 호출에만 등록 parent를 공급한다."""
    with _LOCK:
        require(1 <= len(sources) <= MAX_TARGETS, "epoch-2 source 수 범위")
        registrations = [_registration(source) for source in sources]
        targets = [item["target"] for item in registrations]
        require(len(set(targets)) == len(targets), "중복 R4 target")
        digests = [item["packet_digest"] for item in registrations]
        require(len(set(digests)) == len(digests), "서로 다른 target의 packet이 같다")
        require(SECOND._DRIVER.C is _BASE_CONTROLLER, "SECOND driver controller가 이미 바뀌었다")

        by_digest = {item["packet_digest"]: item for item in registrations}
        view = _clone(_BASE_CONTROLLER.V, "_recovery_epoch2_parent_view_v1")
        controller = _clone(_BASE_CONTROLLER, "_recovery_epoch2_parent_controller_v1")
        original_build = _BASE_CONTROLLER.V.build_snapshot

        def build_snapshot(
            packet: dict,
            run_id: str,
            policy: str,
            protocol_sha256: str,
            ledger_path: str | Path,
            parent: dict | None = None,
            masked: bool = False,
        ) -> dict:
            packet_digest = SECOND.digest(packet)
            registration = by_digest.get(packet_digest)
            if registration is None or packet != registration["packet"]:
                raise ValueError("snapshot packet이 등록 source와 다르다")
            require(run_id == registration["run_id"], "snapshot run_id가 등록 target과 다르다")
            require(policy == registration["policy"], "snapshot policy가 등록 arm과 다르다")
            expected_parent = registration["parent"]
            require(parent is None or parent == expected_parent, "snapshot parent가 등록 계보와 다르다")
            return original_build(
                packet,
                run_id,
                policy,
                protocol_sha256,
                ledger_path,
                parent=deepcopy(expected_parent),
                masked=masked,
            )

        view.__dict__["build_snapshot"] = build_snapshot
        controller.__dict__["V"] = view
        SECOND._DRIVER.__dict__["C"] = controller
        try:
            yield {item["target"]: deepcopy(item["parent"]) for item in registrations}
        finally:
            current = SECOND._DRIVER.C
            SECOND._DRIVER.__dict__["C"] = _BASE_CONTROLLER
            if current is not controller:
                raise RuntimeError("SECOND driver controller가 scope 중 외부에서 바뀌었다")

