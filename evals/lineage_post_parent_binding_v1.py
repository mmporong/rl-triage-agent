"""epoch 2 selection snapshot에 검증한 parent 결합을 공급하는 CPU scope."""

from __future__ import annotations

import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from threading import RLock
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "src")]

import lineage_post_correction_inputs_v1 as POST
import selection_comparison_controller_v1 as C

_LOCK = RLock()
_LOCAL = threading.local()
_ORIGINAL_BUILD = C.V.build_snapshot
_VIEW: Any = C.V


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def _context(post_input: Path, reference_input: Path, run_id: str, policy: str) -> dict[str, Any]:
    post_input, reference_input = Path(post_input).resolve(), Path(reference_input).resolve()
    POST.checked_observation(post_input)
    envelope = POST.make_selection_packet(post_input, reference_input)
    _require(envelope.get("schema") == POST.INPUT.PACKET_SCHEMA,
             "post selection packet envelope schema가 다르다")
    packet = C.V.S.validate_packet(envelope["packet"])
    _require(packet["epoch"] == 2 and isinstance(packet.get("prior_receipt"), dict),
             "parent scope는 검증한 epoch 2 packet만 허용한다")
    _require(isinstance(run_id, str) and bool(run_id) and policy in C.V.POLICIES,
             "parent scope run_id 또는 policy가 다르다")
    prior = packet["prior_receipt"]
    parent = {"run_id": run_id, "policy": policy, "epoch": 1, **prior}
    return {
        "packet": packet, "packet_digest": C.L.digest(packet),
        "run_id": run_id, "policy": policy, "parent": parent,
        "post_input": str(post_input), "reference_input": str(reference_input),
    }


@contextmanager
def parent_scope(post_input: Path, reference_input: Path,
                 run_id: str, policy: str) -> Iterator[dict[str, Any]]:
    """exact epoch 2 packet/run/policy의 누락 parent만 채운다."""
    context = _context(post_input, reference_input, run_id, policy)

    def bound_build(packet: dict, run_id: str, policy: str,
                    protocol_sha256: str, ledger_path: str | Path,
                    parent: dict | None = None, masked: bool = False) -> dict:
        checked = C.V.S.validate_packet(packet)
        _require(C.L.digest(checked) == context["packet_digest"]
                 and checked == context["packet"],
                 "parent scope에 다른 packet이 들어왔다")
        _require(run_id == context["run_id"] and policy == context["policy"],
                 "parent scope에 다른 run_id 또는 policy가 들어왔다")
        expected = context["parent"]
        _require(parent is None or parent == expected, "명시된 epoch 2 parent가 검증값과 다르다")
        return _ORIGINAL_BUILD(
            checked, run_id, policy, protocol_sha256, ledger_path,
            parent=deepcopy(expected), masked=masked)

    with _LOCK:
        depth = getattr(_LOCAL, "depth", 0)
        if depth:
            _require(getattr(_LOCAL, "context", None) == context,
                     "다른 parent scope가 중첩됐다")
            _LOCAL.depth = depth + 1
            try:
                yield deepcopy(context)
            finally:
                _LOCAL.depth -= 1
            return
        _require(C.V.build_snapshot is _ORIGINAL_BUILD,
                 "selection snapshot builder가 등록 원본이 아니다")
        _VIEW.build_snapshot = bound_build
        _LOCAL.depth = 1
        _LOCAL.context = context
        try:
            yield deepcopy(context)
        finally:
            _LOCAL.context = None
            _LOCAL.depth = 0
            _VIEW.build_snapshot = _ORIGINAL_BUILD
