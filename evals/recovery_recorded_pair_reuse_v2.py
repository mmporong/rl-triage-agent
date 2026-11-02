"""첫 recorded pair만 재생하고 같은 run의 후속 probe는 원 host로 넘긴다."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from threading import RLock
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("evals", "src")]

import five_policy_recovery_epoch2_v1 as SECOND
import recovery_recorded_pair_reuse_v1 as V1

SOURCE_FILES = (
    *V1.SOURCE_FILES,
    "evals/recovery_recorded_pair_reuse_v2.py",
    "tests/test_recovery_recorded_pair_reuse_v2.py",
)
_LOCK = RLock()


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def _same_target(old_snapshot: dict[str, Any], snapshot: dict[str, Any]) -> bool:
    old_binding, binding = old_snapshot.get("binding"), snapshot.get("binding")
    return (
        isinstance(old_binding, dict)
        and isinstance(binding, dict)
        and old_binding.get("run_id") == binding.get("run_id")
        and old_binding.get("policy") == binding.get("policy")
    )


def _common_call(host: Any, old_result: dict[str, Any], old_snapshot: dict[str, Any],
                 snapshot: dict[str, Any], keywords: dict[str, Any]) -> tuple[Path, Path]:
    old_inputs, inputs = old_snapshot.get("inputs"), snapshot.get("inputs")
    require(isinstance(old_inputs, dict) and isinstance(inputs, dict), "snapshot inputs 형식")
    old_inputs = cast(dict[str, Any], old_inputs)
    inputs = cast(dict[str, Any], inputs)
    require(snapshot.get("binding") == old_snapshot.get("binding"),
            "recorded pair binding/parent가 다르다")
    require({key: value for key, value in inputs.items() if key != "ledger_path"}
            == {key: value for key, value in old_inputs.items() if key != "ledger_path"},
            "recorded pair source/parent/checkpoint/protocol이 다르다")
    backend_packet = keywords.get("backend_packet")
    require(isinstance(backend_packet, dict)
            and host.digest(backend_packet) == old_result.get("backend_packet_sha256"),
            "recorded pair backend source가 다르다")
    probe_protocol = Path(keywords.get("probe_protocol", "")).resolve()
    view_protocol = Path(keywords.get("view_protocol", "")).resolve()
    require(V1._ref(host, probe_protocol) == old_result.get("probe_protocol")
            and V1._ref(host, view_protocol) == old_result.get("view_protocol")
            and keywords.get("anchor") == old_result.get("view_anchor"),
            "recorded pair protocol/anchor가 다르다")
    lab = Path(keywords.get("lab", "")).resolve()
    supplied = {"case": keywords.get("case_input"), "reference": keywords.get("reference_input")}
    for role in V1.ROLES:
        plan = host.read(Path(old_result["plans"][role]["path"]))
        require(Path(supplied[role]).resolve() == Path(plan["collected_input"]["path"]).resolve(),
                f"recorded pair {role} source가 다르다")
        require(lab == Path(plan["lab"]).resolve(), "recorded pair lab source가 다르다")
    return Path(inputs["ledger_path"]).resolve(), Path(keywords.get("out", "")).resolve()


def _first_call(host: Any, old_result: dict[str, Any], old_snapshot: dict[str, Any],
                old_choice: dict[str, Any], snapshot: dict[str, Any], choice: dict[str, Any],
                keywords: dict[str, Any]) -> tuple[dict[str, Any], Path, Path]:
    ledger, out = _common_call(host, old_result, old_snapshot, snapshot, keywords)
    require(snapshot.get("view", {}).get("observations") == [],
            "recorded pair first replay에는 empty view가 필요하다")
    require(V1._snapshot_without_ledger(snapshot) == V1._snapshot_without_ledger(old_snapshot),
            "recorded pair first snapshot view가 다르다")
    require(choice == old_choice and choice.get("probe") == old_result.get("probe") == "P_episode",
            "recorded pair first choice가 다르다")
    spec = host.C.request_spec(snapshot, choice)
    old_spec = host.C._frozen_request_spec(old_snapshot, old_choice)
    require(spec == old_spec and host.digest(spec) == old_result.get("request_spec_sha256"),
            "recorded pair first request args가 다르다")
    require(not out.exists(), "recorded pair output은 fresh여야 한다")
    require(not ledger.exists() or (ledger.is_file() and ledger.stat().st_size == 0),
            "recorded pair first public ledger는 fresh/empty여야 한다")
    return spec, ledger, out


def _followup_call(host: Any, old_result: dict[str, Any], old_snapshot: dict[str, Any],
                   snapshot: dict[str, Any], choice: dict[str, Any], keywords: dict[str, Any],
                   replay_ledger: Path | None, replay_used: bool) -> None:
    require(replay_used and replay_ledger is not None,
            "recorded pair first replay를 건너뛴 후속 probe는 거부한다")
    ledger, _out = _common_call(host, old_result, old_snapshot, snapshot, keywords)
    require(ledger == replay_ledger, "recorded pair public ledger 바꿔끼우기")
    frozen = host.C.V.checked(snapshot)
    observations = frozen.get("view", {}).get("observations")
    expected_first = {"probe": old_result["probe"], "outcome": old_result["outcome"]}
    require(isinstance(observations, list) and observations
            and observations[0] == expected_first,
            "recorded pair first logical receipt가 현재 view에 없다")
    require(choice.get("probe") != old_result["probe"],
            "recorded pair physical effect duplicate 요청")
    host.C.request_spec(frozen, choice)


@contextmanager
def recorded_pair_reuse_scope(*, original_result: Path, original_context: Path,
                              original_choice: Path) -> Iterator[dict[str, Any]]:
    """첫 empty-view P_episode만 replay하고 검증된 후속 snapshot은 원 실행기에 위임한다."""
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
        V1._verified_roles(host, old_result)
        original_execute = SECOND.execute_pair
        replay_used = False
        replay_ledger: Path | None = None

        def execute(*arguments: Any, **keywords: Any) -> dict[str, Any]:
            nonlocal replay_used, replay_ledger
            snapshot, choice, call_keywords = V1._call(arguments, keywords)
            if not _same_target(old_snapshot, snapshot):
                return cast(dict[str, Any], original_execute(*arguments, **keywords))
            observations = snapshot.get("view", {}).get("observations")
            require(isinstance(observations, list), "recorded pair snapshot observations 형식")
            if not observations:
                require(not replay_used, "recorded physical pair는 scope에서 한 번만 재사용할 수 있다")
                spec, ledger, out = _first_call(
                    host, old_result, old_snapshot, old_choice, snapshot, choice, call_keywords)
                replay_used, replay_ledger = True, ledger
                return V1._replay(
                    host, result_path, context_path, choice_path, old_result, old_snapshot,
                    old_choice, snapshot, choice, call_keywords, spec, ledger, out)
            _followup_call(
                host, old_result, old_snapshot, snapshot, choice, call_keywords,
                replay_ledger, replay_used)
            return cast(dict[str, Any], original_execute(*arguments, **keywords))

        SECOND.__dict__["execute_pair"] = execute
        try:
            yield {
                "source_result": V1._ref(host, result_path),
                "source_context": V1._ref(host, context_path),
                "source_choice": V1._ref(host, choice_path),
                "physical_roles_available": 2,
                "new_sdk_execution_count": 0,
                "followup_delegate_enabled": True,
            }
        finally:
            current = SECOND.execute_pair
            SECOND.__dict__["execute_pair"] = original_execute
            if current is not execute:
                raise RuntimeError("recorded pair reuse v2 scope가 외부에서 바뀌었다")
