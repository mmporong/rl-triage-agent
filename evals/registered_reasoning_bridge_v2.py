"""새 source 전향 확인의 none/low slot을 WSL 단일 호출로 연결한다."""
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
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "src")]

import nemotron_reasoning_options_study_v1 as MODE

SCHEMA = "registered_reasoning_bridge_protocol_v2"
SESSION_SCHEMA = "registered_reasoning_bridge_session_v2"
REQUEST_SCHEMA = "registered_reasoning_bridge_request_v2"
RESPONSE_SCHEMA = "registered_reasoning_bridge_response_v2"
DESIGN_SCHEMA = "prospective_reasoning_confirmation_design_v1"
SOURCE_FILES = ("evals/registered_reasoning_bridge_v2.py",
                "tests/test_registered_reasoning_bridge_v2.py")
MAX_REQUESTS = 96
TIMEOUT_S = 395
LINUX_TIMEOUT_S = 360
FAULT_CASES = ("c02", "c03", "c04", "c06", "c07", "c08")


def require(ok: object, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def file_sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def lf_sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def digest(value: object) -> str:
    return MODE.digest(value)


def _read(path: Path) -> dict[str, Any]:
    value = MODE._read(Path(path))
    require(isinstance(value, dict), "JSON object가 아니다")
    return value


def _write_new(path: Path, value: object) -> None:
    MODE._write_new(Path(path), value)


def _ref(path: Path) -> dict[str, str]:
    path = Path(path).resolve()
    require(path.is_file() and path.is_relative_to(ROOT), "공개 파일 ref가 아니다")
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": file_sha(path)}


def _expected_slots() -> list[dict[str, str]]:
    return [{"slot_id": f"{case}.{mode}.d{decision}.a{attempt}",
             "case_id": case, "mode": mode}
            for case in FAULT_CASES for mode in MODE.MODES
            for decision in range(1, 5) for attempt in range(1, 3)]


def _mode_contract() -> dict[str, dict[str, Any]]:
    messages = [{"role": "user", "content": "<registered-messages>"}]
    return {mode: {key: deepcopy(value) for key, value in MODE.build_request(
        messages, mode).items() if key != "messages"} for mode in MODE.MODES}


def _parent(path: Path) -> dict[str, Any]:
    value = _read(path)
    require(value.get("schema") == DESIGN_SCHEMA
            and value.get("status") == "registered_only_after_commit",
            "전향 확인 design protocol 불일치")
    bridge = value.get("bridge")
    require(isinstance(bridge, dict)
            and bridge.get("max_requests") == MAX_REQUESTS
            and bridge.get("logical_slots") == _expected_slots()
            and bridge.get("modes") == _mode_contract(), "design bridge 계약 불일치")
    return value


def prepare_protocol(parent_design: Path) -> dict[str, Any]:
    parent_design = Path(parent_design).resolve()
    parent = _parent(parent_design)
    closure = deepcopy(parent["source_sha256_lf"])
    closure.update({name: lf_sha(ROOT / name) for name in SOURCE_FILES})
    closure[parent_design.relative_to(ROOT).as_posix()] = lf_sha(parent_design)
    return {
        "schema": SCHEMA, "status": "registered_only_after_commit",
        "parent_design": _ref(parent_design), "model": parent["model"],
        "accepted_provider_models": deepcopy(parent["accepted_provider_models"]),
        "modes": _mode_contract(), "logical_slots": _expected_slots(),
        "max_requests": MAX_REQUESTS, "retry": 0, "fallback": False,
        "authority": {"approval": False, "gpu": False, "repair": False,
                      "recovery": False},
        "source_sha256_lf": closure,
    }


def _verify_protocol(path: Path) -> dict[str, Any]:
    value = _read(path)
    parent = ROOT / value.get("parent_design", {}).get("path", "")
    require(value == prepare_protocol(parent), "bridge v2 protocol이 현재 source와 다르다")
    return value


def _git(*args: str) -> bytes:
    run = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, check=False)
    require(run.returncode == 0, f"git {' '.join(args)} 실패")
    return run.stdout


def _verify_registered(path: Path, anchor: str) -> dict[str, Any]:
    require(re.fullmatch(r"[0-9a-f]{40}", anchor) is not None, "anchor 형식")
    path = Path(path).resolve()
    require(path.is_relative_to(ROOT / "bench/protocols"), "bridge 공개 protocol 경로")
    value = _verify_protocol(path)
    _git("merge-base", "--is-ancestor", anchor, "HEAD")
    relative = path.relative_to(ROOT).as_posix()
    require(_git("show", f"{anchor}:{relative}").replace(b"\r\n", b"\n")
            == path.read_bytes().replace(b"\r\n", b"\n"), "bridge anchor blob 불일치")
    for name, expected in value["source_sha256_lf"].items():
        raw = _git("show", f"{anchor}:{name}").replace(b"\r\n", b"\n")
        require(hashlib.sha256(raw).hexdigest() == expected
                and lf_sha(ROOT / name) == expected, f"bridge source blob 불일치: {name}")
    return value


def initialize(root: Path, protocol: Path, anchor: str) -> dict[str, Any]:
    root = Path(root).resolve()
    spec = _verify_registered(protocol, anchor)
    session = {"schema": SESSION_SCHEMA, "protocol": _ref(Path(protocol)),
               "anchor": anchor, "max_requests": spec["max_requests"]}
    root.mkdir(parents=True, exist_ok=True)
    path = root / "session.json"
    if path.exists():
        require(_read(path) == session, "다른 bridge v2 session")
    else:
        _write_new(path, session)
    return session


def _session(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    session = _read(Path(root) / "session.json")
    require(session.get("schema") == SESSION_SCHEMA, "bridge v2 session schema")
    protocol = ROOT / session["protocol"]["path"]
    require(file_sha(protocol) == session["protocol"]["sha256"], "bridge protocol SHA")
    return session, _verify_registered(protocol, session["anchor"])


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
                raise TimeoutError("bridge v2 reservation lock timeout") from None
            time.sleep(0.02)
    try:
        os.write(fd, str(os.getpid()).encode())
        yield
    finally:
        os.close(fd)
        path.unlink(missing_ok=True)


def _reserve(root: Path, slot: str, messages: list, params: dict,
             protocol: dict[str, Any], protocol_sha: str) -> Path:
    slots = {row["slot_id"]: row for row in protocol["logical_slots"]}
    require(slot in slots, "미등록 bridge v2 slot")
    mode = slots[slot]["mode"]
    require(params == protocol["modes"][mode], "slot mode와 params 불일치")
    with _lock(root):
        requests = sorted((Path(root) / "requests").glob("[0-9][0-9][0-9].json"))
        require(len(requests) < protocol["max_requests"], "bridge v2 API cap 종료")
        require(all(_read(path).get("slot") != slot for path in requests), "slot 재전송 금지")
        index = len(requests) + 1
        intent = {"schema": REQUEST_SCHEMA, "index": index, "slot": slot,
                  "case_id": slots[slot]["case_id"], "mode": mode,
                  "messages": deepcopy(messages), "messages_sha256": digest(messages),
                  "params": deepcopy(params), "params_sha256": digest(params),
                  "bridge_protocol_sha256": protocol_sha,
                  "context_sha256": digest({"slot": slot, "messages": messages,
                                             "params": params})}
        path = Path(root) / "requests" / f"{index:03d}.json"
        _write_new(path, intent)
        return path


def _wsl_path(path: Path) -> str:
    match = re.fullmatch(r"([A-Za-z]):\\(.*)", str(Path(path).resolve()))
    require(match is not None, "WSL 변환은 Windows 절대 경로만 허용")
    if match is None:
        raise ValueError("WSL 경로 변환 실패")
    return f"/mnt/{match.group(1).lower()}/{match.group(2).replace(chr(92), '/')}"


def _spawn_worker(root: Path, request: Path, timeout: int) -> int:
    args = [_wsl_path(ROOT / "evals/registered_reasoning_bridge_v2.py"),
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
        raise TimeoutError("bridge v2 WSL worker timeout") from None


def _response(root: Path, request: Path) -> tuple[Path, dict[str, Any]]:
    intent = _read(request)
    path = Path(root) / "responses" / request.name
    value = _read(path)
    require(set(value) == {"schema", "request_sha256", "result"}
            and value["schema"] == RESPONSE_SCHEMA
            and value["request_sha256"] == file_sha(request)
            and isinstance(value["result"], dict), "bridge v2 response 변조")
    require(intent["params"] == value["result"].get("request_params"),
            "bridge v2 response params 불일치")
    return path, value


def complete(root: Path, slot: str, messages: list, params: dict) -> dict[str, Any]:
    root = Path(root).resolve()
    session, protocol = _session(root)
    request = _reserve(root, slot, messages, params, protocol, session["protocol"]["sha256"])
    require(_spawn_worker(root, request, TIMEOUT_S) == 0, "bridge v2 worker 비정상 종료")
    response_path, response = _response(root, request)
    result = deepcopy(response["result"])
    result["transport_ref"] = {"path": str(response_path.resolve()),
                               "sha256": file_sha(response_path)}
    return result


def _start_ticks(pid: int) -> int | None:
    path = Path("/proc") / str(pid) / "stat"
    return (int(path.read_text(encoding="utf-8").rsplit(")", 1)[1].split()[19])
            if path.is_file() else None)


def worker_call(root: Path, request_path: Path,
                complete_call: Any = MODE.openai_complete) -> dict[str, Any]:
    root, request_path = Path(root).resolve(), Path(request_path).resolve()
    session, protocol = _session(root)
    match = re.fullmatch(r"([0-9]{3})\.json", request_path.name)
    require(request_path.parent == root / "requests" and match is not None,
            "worker request 정규 경로 불일치")
    if match is None:
        raise ValueError("worker request 형식")
    index = int(match.group(1))
    require(1 <= index <= protocol["max_requests"], "worker request cap 불일치")
    intent = _read(request_path)
    slots = {row["slot_id"]: row for row in protocol["logical_slots"]}
    require(intent.get("schema") == REQUEST_SCHEMA and intent.get("slot") in slots
            and intent.get("index") == index
            and intent.get("case_id") == slots[intent["slot"]]["case_id"]
            and intent.get("mode") == slots[intent["slot"]]["mode"]
            and intent.get("bridge_protocol_sha256") == session["protocol"]["sha256"]
            and intent.get("params") == protocol["modes"][intent["mode"]]
            and intent.get("messages_sha256") == digest(intent.get("messages"))
            and intent.get("params_sha256") == digest(intent.get("params"))
            and intent.get("context_sha256") == digest({"slot": intent["slot"],
                                                         "messages": intent["messages"],
                                                         "params": intent["params"]}),
            "worker intent 결합 불일치")
    response_path = root / "responses" / request_path.name
    require(not response_path.exists(), "worker response 중복")
    request_sha = file_sha(request_path)
    claim = {"schema": "registered_reasoning_worker_claim_v2", "index": index,
             "slot": intent["slot"], "request_sha256": request_sha,
             "bridge_protocol_sha256": session["protocol"]["sha256"],
             "pid": os.getpid(), "start_ticks": _start_ticks(os.getpid())}
    _write_new(root / "claims" / request_path.name, claim)
    _write_new(root / "slot-claims" / f"{intent['slot']}.json", claim)
    result = complete_call({"messages": deepcopy(intent["messages"]),
                            **deepcopy(intent["params"])})
    result["request_params"] = deepcopy(intent["params"])
    response = {"schema": RESPONSE_SCHEMA, "request_sha256": request_sha, "result": result}
    _write_new(response_path, response)
    return response


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-call", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    require(sys.platform.startswith("linux") and bool(os.environ.get("WSL_DISTRO_NAME")),
            "bridge worker는 기존 WSL Ubuntu에서만 실행한다")
    worker_call(args.root, args.worker_call)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["MAX_REQUESTS", "SCHEMA", "complete", "initialize", "prepare_protocol",
           "worker_call"]
