"""동일 전향 비교를 lease와 host context adapter로 기계적으로 재개하는 v3 wrapper."""
# pyright: reportMissingImports=false
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "bench"), str(ROOT / "src")]

import run_prospective_reasoning_confirmation_v2 as V2  # noqa: E402

R = V2.R
PROTOCOL_SCHEMA = "prospective_reasoning_confirmation_execution_v3"
PREFLIGHT_SCHEMA = "prospective_reasoning_confirmation_preflight_v3"
RESULT_SCHEMA = "prospective_reasoning_confirmation_continuation_v3"
ERROR_SCHEMA = "prospective_reasoning_confirmation_terminal_error_v3"
V2_ANCHOR = "23e2bec8cffc65cfbf4d9bdbaa49bcdc82b6c95b"
FIRST_SLOT = "c02.legacy_none.d1.a1"
ARTIFACTS = {
    "session": "session.json",
    "request": "requests/001.json",
    "response": "responses/001.json",
    "claim": "claims/001.json",
    "slot_claim": f"slot-claims/{FIRST_SLOT}.json",
}
LOCAL_SOURCE_FILES = (
    "evals/run_prospective_reasoning_confirmation_v3.py",
    "tests/test_run_prospective_reasoning_confirmation_v3.py",
)
HEX40 = re.compile(r"[0-9a-f]{40}\Z")


def require(ok: object, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=lambda token: (
        _ for _ in ()).throw(ValueError(f"비유한 JSON 숫자: {token}")))
    require(isinstance(value, dict), f"{Path(path).name}은 JSON object여야 한다")
    return cast(dict[str, Any], value)


def _sha(path: Path, *, lf: bool = False) -> str:
    raw = Path(path).read_bytes()
    if lf:
        raw = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(raw).hexdigest()


def _digest(value: object) -> str:
    return R.BRIDGE.digest(value)


def _write_new(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def _relative(path: Path) -> str:
    path = Path(path).resolve()
    require(path.is_file() and path.is_relative_to(ROOT), "공개 protocol/source 경로가 아니다")
    return path.relative_to(ROOT).as_posix()


def _ref(path: Path) -> dict[str, str]:
    return {"path": _relative(path), "sha256_lf": _sha(path, lf=True)}


def _git(*args: str) -> bytes:
    run = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, check=False)
    require(run.returncode == 0, run.stderr.decode("utf-8", errors="replace").strip())
    return run.stdout


def _verify_blob(anchor: str, relative: str, expected: str) -> None:
    raw = _git("show", f"{anchor}:{relative}").replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    require(hashlib.sha256(raw).hexdigest() == expected, f"anchor blob 불일치: {relative}")


def _failed_artifacts(root: Path, bridge_protocol: Path) -> dict[str, Any]:
    root = Path(root).resolve()
    paths = {name: root / relative for name, relative in ARTIFACTS.items()}
    require(all(path.is_file() for path in paths.values()), "실패 bridge 원문 5종이 완결되지 않았다")
    session, request, response = (_read(paths[name]) for name in ("session", "request", "response"))
    claim, slot_claim = _read(paths["claim"]), _read(paths["slot_claim"])
    bridge_sha = _sha(bridge_protocol)
    request_sha = _sha(paths["request"])
    require(session == {"schema": R.BRIDGE.SESSION_SCHEMA,
                        "protocol": {"path": _relative(bridge_protocol),
                                     "sha256": bridge_sha},
                        "anchor": V2.STAGE_A_ANCHOR, "max_requests": R.BRIDGE.MAX_REQUESTS},
            "실패 bridge session 결합 불일치")
    require(request.get("schema") == R.BRIDGE.REQUEST_SCHEMA
            and request.get("index") == 1 and request.get("slot") == FIRST_SLOT
            and request.get("case_id") == "c02" and request.get("mode") == "legacy_none"
            and request.get("bridge_protocol_sha256") == bridge_sha
            and request.get("messages_sha256") == _digest(request.get("messages"))
            and request.get("params_sha256") == _digest(request.get("params"))
            and request.get("context_sha256") == _digest({
                "slot": FIRST_SLOT, "messages": request.get("messages"),
                "params": request.get("params")}), "실패 bridge request 결합 불일치")
    require(response.get("schema") == R.BRIDGE.RESPONSE_SCHEMA
            and response.get("request_sha256") == request_sha
            and isinstance(response.get("result"), dict)
            and response["result"].get("request_params") == request.get("params"),
            "실패 bridge response 결합 불일치")
    require(claim == slot_claim and claim.get("schema") == "registered_reasoning_worker_claim_v2"
            and claim.get("index") == 1 and claim.get("slot") == FIRST_SLOT
            and claim.get("request_sha256") == request_sha
            and claim.get("bridge_protocol_sha256") == bridge_sha,
            "실패 bridge claim 결합 불일치")
    return {
        "slot": FIRST_SLOT,
        "api_requests_consumed": 1,
        "sdk_role_executions": 0,
        "assessment": "indeterminate",
        "artifacts": {name: _sha(path) for name, path in paths.items()},
    }


def approved_host_context(execute_host: Callable[..., dict[str, Any]]) -> Callable[..., dict[str, Any]]:
    """바깥 검증의 정확한 True만 host 승인 object로 변환한다."""
    def adapted(*args: Any, verify_context: Callable[[dict[str, Any]], object],
                **kwargs: Any) -> dict[str, Any]:
        def approved(request: dict[str, Any]) -> dict[str, bool]:
            value = verify_context(deepcopy(request))
            require(value is True, "바깥 context 검증이 정확한 True를 반환하지 않았다")
            return {"approved": True}

        return execute_host(*args, verify_context=approved, **kwargs)

    return adapted


class ReplayCompleteFactory:
    """첫 소비 슬롯만 원문으로 재생하고 이후 슬롯은 기존 bridge로 넘긴다."""

    def __init__(self, root: Path, bridge_protocol: Path,
                 delegate: Callable[[Path, str, list, dict], dict[str, Any]] = R.BRIDGE.complete):
        self.root = Path(root).resolve()
        self.bridge_protocol = Path(bridge_protocol).resolve()
        self.delegate = delegate
        self.binding = _failed_artifacts(self.root, self.bridge_protocol)
        self.calls = 0
        self.replayed = 0

    def __call__(self, slot: str, _state: dict[str, Any]) -> Callable[[list, dict], dict[str, Any]]:
        def complete(messages: list, params: dict) -> dict[str, Any]:
            self.calls += 1
            if self.calls == 1:
                require(slot == FIRST_SLOT, "첫 continuation 호출이 보존 슬롯과 다르다")
            if slot != FIRST_SLOT:
                return self.delegate(self.root, slot, deepcopy(messages), deepcopy(params))
            require(self.replayed == 0, "보존 슬롯 중복 재생 금지")
            require(_failed_artifacts(self.root, self.bridge_protocol) == self.binding,
                    "재생 직전 실패 bridge 원문이 바뀌었다")
            request_path = self.root / ARTIFACTS["request"]
            response_path = self.root / ARTIFACTS["response"]
            request, response = _read(request_path), _read(response_path)
            require(request["messages"] == messages and request["params"] == params,
                    "보존 슬롯 messages/params가 현재 호출과 다르다")
            self.replayed += 1
            result = deepcopy(response["result"])
            result["transport_ref"] = {"path": str(response_path),
                                       "sha256": _sha(response_path)}
            return result

        return complete


def prepare_protocol(base_v2_protocol: Path, base_v1_protocol: Path, design: Path,
                     bridge: Path, source_batch: Path, corpus: Path,
                     failed_bridge_root: Path) -> dict[str, Any]:
    paths = [Path(value).resolve() for value in
             (base_v2_protocol, base_v1_protocol, design, bridge)]
    base_v2_path, base_v1_path, design_path, bridge_path = paths
    base_v2 = _read(base_v2_path)
    require(base_v2 == V2.prepare_protocol(base_v1_path), "base v2 execution 계약 불일치")
    require(ROOT / base_v2["base_design_protocol"]["path"] == design_path
            and ROOT / base_v2["base_bridge_protocol"]["path"] == bridge_path,
            "base v2 design/bridge 인자 불일치")
    closure = deepcopy(base_v2["source_sha256_lf"])
    closure[_relative(base_v2_path)] = _sha(base_v2_path, lf=True)
    closure.update({name: _sha(ROOT / name, lf=True) for name in LOCAL_SOURCE_FILES})
    return {
        "schema": PROTOCOL_SCHEMA,
        "status": "registered_only_after_commit",
        "base_v2_protocol": _ref(base_v2_path),
        "base_v2_anchor": V2_ANCHOR,
        "base_v1_protocol": _ref(base_v1_path),
        "base_v1_anchor": V2.STAGE_B_ANCHOR,
        "design_protocol": _ref(design_path),
        "design_anchor": V2.STAGE_A_ANCHOR,
        "bridge_protocol": _ref(bridge_path),
        "source_batch_sha256": _sha(source_batch),
        "corpus_sha256": _sha(corpus),
        "failed_attempt": _failed_artifacts(failed_bridge_root, bridge_path),
        "continuation": {
            "classification": "mechanical_infrastructure_continuation",
            "exposure": "post_response_host_contract_fix_same_source_development_confirmation",
            "first_slot_replayed_without_new_post": True,
            "prior_failure_preserved_indeterminate": True,
            "new_seed_training": False,
            "policy_prompt_threshold_case_or_budget_changed": False,
        },
        "inherited": deepcopy(base_v2["inherited"]),
        "claims": {"llm_superiority": False, "default_mode_adoption": False,
                   "new_fault_family_generalization": False, "repair": False,
                   "recovery": False, "pure_unexposed_performance": False},
        "source_sha256_lf": closure,
    }


def verify_protocol(protocol: Path, anchor: str, base_v2_protocol: Path,
                    base_v1_protocol: Path, design: Path, bridge: Path,
                    source_batch: Path, corpus: Path, failed_bridge_root: Path) -> dict[str, Any]:
    require(HEX40.fullmatch(anchor) is not None, "v3 anchor 형식")
    expected = prepare_protocol(base_v2_protocol, base_v1_protocol, design, bridge,
                                source_batch, corpus, failed_bridge_root)
    path = Path(protocol).resolve()
    value = _read(path)
    require(value == expected, "v3 protocol 현재 입력 불일치")
    for ancestor, descendant in ((V2_ANCHOR, anchor), (anchor, "HEAD")):
        _git("merge-base", "--is-ancestor", ancestor, descendant)
    _verify_blob(V2_ANCHOR, _relative(Path(base_v2_protocol)), _sha(base_v2_protocol, lf=True))
    bound = {_relative(path): _sha(path, lf=True), **value["source_sha256_lf"]}
    for relative, digest in bound.items():
        current = ROOT / relative
        require(current.is_file() and _sha(current, lf=True) == digest,
                f"현재 v3 source 불일치: {relative}")
        _verify_blob(anchor, relative, digest)
    V2.verify_protocol(Path(base_v2_protocol), V2_ANCHOR)
    R.verify_execution(Path(base_v1_protocol), V2.STAGE_B_ANCHOR, design, bridge,
                       source_batch, corpus)
    return deepcopy(value)


def preflight(protocol: Path, anchor: str, base_v2_protocol: Path, base_v1_protocol: Path,
              design: Path, bridge: Path, source_batch: Path, corpus: Path,
              failed_bridge_root: Path, out: Path) -> dict[str, Any]:
    value = verify_protocol(protocol, anchor, base_v2_protocol, base_v1_protocol, design,
                            bridge, source_batch, corpus, failed_bridge_root)
    result = {"schema": PREFLIGHT_SCHEMA, "status": "complete",
              "outer_protocol_sha256": _sha(protocol), "outer_anchor": anchor,
              "failed_attempt": deepcopy(value["failed_attempt"]),
              "api_requests": 0, "gpu_started": False,
              "same_source_development_confirmation": True}
    _write_new(Path(out).resolve(), result)
    return result


def _write_error(out: Path, error: Exception) -> None:
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    path = out / "error.json"
    if not path.exists():
        _write_new(path, {"schema": ERROR_SCHEMA, "status": "failed",
                          "error_type": type(error).__name__, "message": str(error),
                          "prior_attempt_assessment": "indeterminate",
                          "prior_api_requests": 1, "prior_sdk_role_executions": 0})


def execute(protocol: Path, anchor: str, base_v2_protocol: Path, base_v1_protocol: Path,
            design: Path, bridge: Path, source_batch: Path, corpus: Path, lab: Path,
            failed_bridge_root: Path, out: Path, *,
            v2_execute: Callable[..., dict[str, Any]] = V2.execute,
            execute_host: Callable[..., dict[str, Any]] = R.HOST.execute_pair,
            bridge_complete: Callable[[Path, str, list, dict], dict[str, Any]] = R.BRIDGE.complete,
            bridge_root: Path | None = None) -> dict[str, Any]:
    out = Path(out).resolve()
    require(not out.exists(), "v3 out은 fresh여야 한다")
    value = verify_protocol(protocol, anchor, base_v2_protocol, base_v1_protocol, design,
                            bridge, source_batch, corpus, failed_bridge_root)
    active_bridge_root = Path(bridge_root or failed_bridge_root).resolve()
    require(active_bridge_root == Path(failed_bridge_root).resolve(),
            "v3는 실패 실행과 같은 bridge-v2 root만 사용한다")
    replay = ReplayCompleteFactory(active_bridge_root, Path(bridge), bridge_complete)
    host = approved_host_context(execute_host)

    def base_execute(*args: Any) -> dict[str, Any]:
        return R.execute(*args, complete_factory=replay, execute_host=host)

    try:
        base_result = v2_execute(Path(base_v2_protocol), V2_ANCHOR,
                                 Path(base_v1_protocol), design, bridge, source_batch,
                                 corpus, lab, active_bridge_root, out,
                                 base_execute=base_execute)
        require(replay.replayed == 1, "보존된 첫 provider 응답이 정확히 한 번 재생되지 않았다")
        result = {
            "schema": RESULT_SCHEMA, "status": "complete",
            "classification": value["continuation"]["classification"],
            "exposure": value["continuation"]["exposure"],
            "prior_attempt": deepcopy(value["failed_attempt"]),
            "base_result_sha256": _sha(out / "result.json"),
            "model_requests_including_replay": base_result["model_requests"],
            "replayed_provider_requests": 1,
            "new_provider_requests": base_result["model_requests"] - 1,
            "paired_probes": base_result["paired_probes"],
            "sdk_role_executions_upper_bound": base_result["sdk_role_executions_upper_bound"],
            "assessment": deepcopy(base_result["assessment"]),
            "claims": deepcopy(value["claims"]),
        }
        _write_new(out / "v3-result.json", result)
        return result
    except Exception as error:
        _write_error(out, error)
        raise


def _common(command: argparse.ArgumentParser) -> None:
    for name in ("protocol", "base-v2-protocol", "base-v1-protocol", "design", "bridge",
                 "source-batch", "corpus", "failed-bridge-root", "out"):
        command.add_argument(f"--{name}", type=Path, required=True)
    command.add_argument("--anchor", required=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare-protocol")
    for name in ("base-v2-protocol", "base-v1-protocol", "design", "bridge",
                 "source-batch", "corpus", "failed-bridge-root", "out"):
        prepare.add_argument(f"--{name}", type=Path, required=True)
    preflight_cmd = commands.add_parser("preflight")
    _common(preflight_cmd)
    execute_cmd = commands.add_parser("execute")
    _common(execute_cmd)
    execute_cmd.add_argument("--lab", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare-protocol":
        _write_new(args.out, prepare_protocol(args.base_v2_protocol, args.base_v1_protocol,
                                              args.design, args.bridge, args.source_batch,
                                              args.corpus, args.failed_bridge_root))
    elif args.command == "preflight":
        preflight(args.protocol, args.anchor, args.base_v2_protocol, args.base_v1_protocol,
                  args.design, args.bridge, args.source_batch, args.corpus,
                  args.failed_bridge_root, args.out)
    else:
        execute(args.protocol, args.anchor, args.base_v2_protocol, args.base_v1_protocol,
                args.design, args.bridge, args.source_batch, args.corpus, args.lab,
                args.failed_bridge_root, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["ReplayCompleteFactory", "approved_host_context", "execute", "preflight",
           "prepare_protocol", "verify_protocol"]
