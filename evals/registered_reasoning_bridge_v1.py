"""등록된 reasoning mode slot을 WSL NVIDIA 호출 한 번으로 연결하는 독립 bridge."""
# pyright: reportMissingImports=false
from __future__ import annotations

import argparse
import contextlib
import hashlib
import os
import re
import shlex
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "src")]

import nemotron_reasoning_options_study_v1 as S1
import nemotron_reasoning_options_study_v2 as S2

SCHEMA = "registered_reasoning_bridge_protocol_v1"
SESSION_SCHEMA = "registered_reasoning_bridge_session_v1"
REQUEST_SCHEMA = "registered_reasoning_bridge_request_v1"
RESPONSE_SCHEMA = "registered_reasoning_bridge_response_v1"
SOURCE_FILES = ("evals/registered_reasoning_bridge_v1.py",
                "tests/test_registered_reasoning_bridge_v1.py")
TIMEOUT_S = 395
LINUX_TIMEOUT_S = 360


def _ref(path: Path) -> dict[str, str]:
    return {"path": Path(path).relative_to(ROOT).as_posix(), "sha256": S1.file_sha(path)}


def _slot(value: dict[str, Any]) -> dict[str, str]:
    S1.require(isinstance(value, dict) and set(value) == {"slot_id", "case_id", "mode"}
               and isinstance(value["slot_id"], str)
               and re.fullmatch(r"c0[2-8]\.(legacy_none|enabled_low)\.d[2-4]\.a[12]",
                                value["slot_id"]) is not None
               and value["case_id"] in {f"c{i:02d}" for i in range(2, 9)}
               and value["mode"] in S1.MODES
               and value["slot_id"].startswith(f"{value['case_id']}.{value['mode']}.") ,
               "reasoning bridge slot 불일치")
    return deepcopy(value)


def prepare_protocol(parent_mode_plan: Path, logical_slots: list[dict],
                     max_requests: int = 72) -> dict[str, Any]:
    parent = S1._read(parent_mode_plan)
    S1.require(parent.get("schema") == S2.PLAN_SCHEMA
               and parent.get("status") == "registered_only_after_commit",
               "parent reasoning v2 protocol 불일치")
    slots = [_slot(row) for row in logical_slots]
    S1.require(type(max_requests) is int and max_requests == 72
               and len(slots) == max_requests
               and len({row["slot_id"] for row in slots}) == max_requests,
               "bridge는 고정 72개 고유 slot을 요구한다")
    fault_cases = {row["case_id"] for row in parent["calls"]
                   if row["evaluation_role"] == "fault_development"}
    S1.require(len(fault_cases) == 6, "parent fault case 분모 불일치")
    expected = {f"{case_id}.{mode}.d{decision}.a{attempt}"
                for case_id in fault_cases for mode in S1.MODES
                for decision in range(2, 5) for attempt in range(1, 3)}
    S1.require({row["slot_id"] for row in slots} == expected,
               "6 fault×2 mode×3 decision×2 attempt slot이 아니다")
    closure = deepcopy(parent["source_sha256_lf"])
    closure.update({name: S1.lf_sha(ROOT / name) for name in SOURCE_FILES})
    closure[Path(parent_mode_plan).relative_to(ROOT).as_posix()] = S1.lf_sha(parent_mode_plan)
    return {"schema": SCHEMA, "status": "registered_only_after_commit",
            "parent_mode_plan": {"path": Path(parent_mode_plan).relative_to(ROOT).as_posix(),
                                 "sha256_lf": S1.lf_sha(parent_mode_plan)},
            "model": parent["model"], "accepted_provider_models": parent["accepted_provider_models"],
            "modes": deepcopy(parent["modes"]), "logical_slots": slots,
            "max_requests": max_requests, "retry": 0, "fallback": False,
            "authority": {"approval": False, "gpu": False, "repair": False, "recovery": False},
            "source_sha256_lf": closure}


def _verify_protocol(path: Path) -> dict[str, Any]:
    value = S1._read(path)
    parent = ROOT / value.get("parent_mode_plan", {}).get("path", "")
    max_requests = value.get("max_requests")
    S1.require(type(max_requests) is int, "bridge max_requests 형식")
    S1.require(value == prepare_protocol(parent, value.get("logical_slots", []),
                                         cast(int, max_requests)),
               "bridge protocol이 현재 source와 다르다")
    return value


def _verify_registered_protocol(path: Path, anchor: str) -> dict[str, Any]:
    S1.require(re.fullmatch(r"[0-9a-f]{40}", anchor) is not None, "anchor 형식")
    path = Path(path).resolve()
    S1.require(path.is_relative_to(ROOT / "bench/protocols"), "bridge protocol 공개 경로")
    value = _verify_protocol(path)
    S1._git("merge-base", "--is-ancestor", anchor, "HEAD")
    relative = path.relative_to(ROOT).as_posix()
    S1.require(S1._git("show", f"{anchor}:{relative}").replace(b"\r\n", b"\n")
               == path.read_bytes().replace(b"\r\n", b"\n"), "bridge protocol anchor blob")
    for name, expected in value["source_sha256_lf"].items():
        raw = S1._git("show", f"{anchor}:{name}").replace(b"\r\n", b"\n")
        S1.require(hashlib.sha256(raw).hexdigest() == expected, f"bridge source blob: {name}")
    return value


def initialize(root: Path, protocol: Path, anchor: str) -> dict[str, Any]:
    root = Path(root).resolve()
    spec = _verify_registered_protocol(protocol, anchor)
    session = {"schema": SESSION_SCHEMA, "protocol": _ref(Path(protocol)), "anchor": anchor,
               "max_requests": spec["max_requests"]}
    root.mkdir(parents=True, exist_ok=True)
    path = root / "session.json"
    if path.exists():
        S1.require(S1._read(path) == session, "다른 bridge session")
    else:
        S1._write_new(path, session)
    return session


def _session(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    session = S1._read(Path(root) / "session.json")
    S1.require(session.get("schema") == SESSION_SCHEMA, "bridge session schema")
    protocol = ROOT / session["protocol"]["path"]
    S1.require(S1.file_sha(protocol) == session["protocol"]["sha256"], "bridge protocol SHA")
    return session, _verify_registered_protocol(protocol, session["anchor"])


@contextlib.contextmanager
def _lock(root: Path):
    path = Path(root) / ".reserve.lock"
    started = time.monotonic()
    fd = None
    while fd is None:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if time.monotonic() - started > 5:
                raise TimeoutError("bridge reservation lock timeout") from None
            time.sleep(0.02)
    try:
        os.write(fd, str(os.getpid()).encode())
        yield
    finally:
        os.close(fd)
        path.unlink(missing_ok=True)


def _reserve(root: Path, slot: str, messages: list, params: dict,
             protocol: dict[str, Any], protocol_sha256: str) -> Path:
    slots = {row["slot_id"]: row for row in protocol["logical_slots"]}
    S1.require(slot in slots, "미등록 bridge slot")
    mode = slots[slot]["mode"]
    expected = deepcopy(protocol["modes"][mode])
    S1.require(params == expected, "slot mode와 params 불일치")
    with _lock(root):
        intents = sorted((Path(root) / "requests").glob("[0-9][0-9][0-9].json"))
        S1.require(len(intents) < protocol["max_requests"], "bridge API cap 종료")
        S1.require(all(S1._read(path).get("slot") != slot for path in intents),
                   "같은 bridge slot 재전송 금지")
        index = len(intents) + 1
        intent = {"schema": REQUEST_SCHEMA, "index": index, "slot": slot,
                  "case_id": slots[slot]["case_id"], "mode": mode,
                  "messages": deepcopy(messages), "messages_sha256": S1.digest(messages),
                  "params": deepcopy(params), "params_sha256": S1.digest(params),
                  "bridge_protocol_sha256": protocol_sha256,
                  "context_sha256": S1.digest({"slot": slot, "messages": messages,
                                               "params": params})}
        path = Path(root) / "requests" / f"{index:03d}.json"
        S1._write_new(path, intent)
        return path


def _wsl_path(path: Path) -> str:
    value = str(Path(path).resolve())
    match = re.fullmatch(r"([A-Za-z]):\\(.*)", value)
    S1.require(match is not None, "WSL 변환은 Windows 절대 경로만 허용")
    if match is None:
        raise ValueError("WSL 변환은 Windows 절대 경로만 허용")
    return f"/mnt/{match.group(1).lower()}/{match.group(2).replace(chr(92), '/')}"


def _spawn_worker(root: Path, request: Path, timeout: int) -> int:
    args = [_wsl_path(ROOT / "evals/registered_reasoning_bridge_v1.py"),
            "--worker-call", _wsl_path(request), "--root", _wsl_path(root)]
    command = ('set -eu; . "$HOME/.config/nvidia/env"; '
               f'exec /usr/bin/timeout --signal=TERM --kill-after=5s {LINUX_TIMEOUT_S}s '
               '"$HOME/.venvs/rl-triage/bin/python" -B ' + shlex.join(args))
    process = subprocess.Popen(["wsl.exe", "-d", "Ubuntu", "--", "bash", "-lc", command],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        return process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        raise TimeoutError("bridge WSL worker timeout") from None


def _response(root: Path, request: Path) -> tuple[Path, dict[str, Any]]:
    intent = S1._read(request)
    path = Path(root) / "responses" / request.name
    value = S1._read(path)
    S1.require(set(value) == {"schema", "request_sha256", "result"}
               and value["schema"] == RESPONSE_SCHEMA
               and value["request_sha256"] == S1.file_sha(request)
               and isinstance(value["result"], dict), "bridge response receipt 변조")
    S1.require(intent["params"] == value["result"].get("request_params"),
               "bridge response params 불일치")
    return path, value


def complete(root: Path, slot: str, messages: list, params: dict) -> dict[str, Any]:
    root = Path(root).resolve()
    session_value, protocol = _session(root)
    request = _reserve(root, slot, messages, params, protocol,
                       session_value["protocol"]["sha256"])
    returncode = _spawn_worker(root, request, TIMEOUT_S)
    S1.require(returncode == 0, "bridge worker 비정상 종료")
    response_path, response = _response(root, request)
    result = deepcopy(response["result"])
    result["transport_ref"] = {"path": str(response_path.resolve()),
                               "sha256": S1.file_sha(response_path)}
    return result


def worker_call(root: Path, request_path: Path) -> dict[str, Any]:
    root, request_path = Path(root).resolve(), Path(request_path).resolve()
    session_value, protocol = _session(root)
    match = re.fullmatch(r"([0-9]{3})\.json", request_path.name)
    S1.require(request_path.parent == root / "requests" and match is not None,
               "worker request 정규 경로/파일명 불일치")
    if match is None:
        raise ValueError("worker request 정규 경로/파일명 불일치")
    index = int(match.group(1))
    S1.require(1 <= index <= protocol["max_requests"], "worker request index cap 불일치")
    intent = S1._read(request_path)
    slots = {row["slot_id"]: row for row in protocol["logical_slots"]}
    S1.require(intent.get("schema") == REQUEST_SCHEMA and intent.get("slot") in slots
               and intent.get("index") == index
               and intent.get("case_id") == slots[intent["slot"]]["case_id"]
               and intent.get("mode") == slots[intent["slot"]]["mode"]
               and intent.get("bridge_protocol_sha256") == session_value["protocol"]["sha256"]
               and intent.get("params") == protocol["modes"][intent["mode"]]
               and intent.get("messages_sha256") == S1.digest(intent.get("messages"))
               and intent.get("params_sha256") == S1.digest(intent.get("params"))
               and intent.get("context_sha256") == S1.digest({"slot": intent["slot"],
                                                               "messages": intent["messages"],
                                                               "params": intent["params"]}),
               "worker intent 결합 불일치")
    response_path = root / "responses" / request_path.name
    S1.require(not response_path.exists(), "worker response 중복")
    request_sha = S1.file_sha(request_path)
    claim = {"schema": "registered_reasoning_worker_claim_v1", "index": index,
             "slot": intent["slot"], "request_sha256": request_sha,
             "bridge_protocol_sha256": session_value["protocol"]["sha256"],
             "pid": os.getpid(), "start_ticks": _start_ticks(os.getpid())}
    S1._write_new(root / "claims" / request_path.name, claim)
    S1._write_new(root / "slot-claims" / f"{intent['slot']}.json", claim)
    result = S1.openai_complete({"messages": deepcopy(intent["messages"]), **deepcopy(intent["params"])})
    result["request_params"] = deepcopy(intent["params"])
    response = {"schema": RESPONSE_SCHEMA, "request_sha256": request_sha,
                "result": result}
    S1._write_new(response_path, response)
    return response


def _start_ticks(pid: int) -> int | None:
    path = Path("/proc") / str(pid) / "stat"
    if not path.is_file():
        return None
    return int(path.read_text(encoding="utf-8").rsplit(")", 1)[1].split()[19])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-call", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    S1.require(sys.platform.startswith("linux") and bool(os.environ.get("WSL_DISTRO_NAME")),
               "bridge worker는 기존 WSL Ubuntu에서만 실행한다")
    worker_call(args.root, args.worker_call)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["complete", "initialize", "main", "prepare_protocol", "worker_call"]
