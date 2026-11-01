# pyright: reportMissingImports=false
"""다섯 정책 lineage 비교용 72-call write-once NIM transport."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "evals") not in sys.path:
    sys.path.insert(0, str(ROOT / "evals"))

import agentic_investigation_call_v2 as CALL
import local_pair_nim_transport_v1 as BASE
import selection_comparison_call_v2 as MODEL_CALL

PROTOCOL_SCHEMA = "lineage_five_policy_protocol_v1"
REQUEST_SCHEMA = BASE.REQUEST_SCHEMA
MANIFEST_SCHEMA = "lineage_five_policy_nim_manifest_v1"
INTENT_SCHEMA = "lineage_five_policy_nim_intent_v1"
RESULT_SCHEMA = "lineage_five_policy_nim_result_v1"
RECEIPT_SCHEMA = "lineage_five_policy_nim_receipt_v1"
VERIFIED_SCHEMA = "lineage_five_policy_nim_verified_call_v1"
STATUS_SCHEMA = "lineage_five_policy_nim_status_v1"
API_CAP = 72
AUTHORITY = deepcopy(BASE.AUTHORITY)
SOURCE_FILES = tuple(sorted({
    *BASE.SOURCE_FILES,
    "evals/lineage_five_policy_nim_transport_v1.py",
    "tests/test_lineage_five_policy_nim_transport_v1.py",
}))
HEX40 = re.compile(r"[0-9a-f]{40}\Z")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")


def _sha(path: Path, *, lf: bool = False) -> str:
    raw = Path(path).read_bytes()
    if lf:
        raw = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(raw).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{Path(path).name}은 JSON object여야 한다")
    return value


def _git(repo: Path, *args: str) -> bytes:
    result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, check=False)
    if result.returncode:
        raise ValueError("Git protocol/source 결합을 확인할 수 없다")
    return result.stdout


def _relative(repo: Path, path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(Path(repo).resolve()).as_posix()
    except ValueError as error:
        raise ValueError("protocol은 repo 내부여야 한다") from error


def _verify_protocol(protocol_path: Path, anchor: str) -> dict[str, Any]:
    repo, path = ROOT.resolve(), Path(protocol_path).resolve()
    relative = _relative(repo, path)
    if not HEX40.fullmatch(anchor):
        raise ValueError("execution anchor 형식이 잘못됐다")
    value = _read(path)
    required = {"schema", "api_cap", "model", "accepted_provider_models", "base_params",
                "sources_lf", "backend0prefix", "authority", "automatic_retries"}
    if not required <= set(value):
        raise ValueError("five-policy protocol 필수 필드가 없다")
    if (value.get("schema") != PROTOCOL_SCHEMA
            or value.get("api_cap") != API_CAP
            or value.get("model") != CALL.PARAMS["model"]
            or value.get("accepted_provider_models") != sorted(MODEL_CALL.ACCEPTED_PROVIDER_MODELS)
            or value.get("base_params") != CALL.PARAMS
            or value.get("backend0prefix") != 0
            or value.get("authority") != AUTHORITY
            or value.get("automatic_retries") != 0
            or not isinstance(value.get("sources_lf"), dict)
            or not set(SOURCE_FILES) <= set(value["sources_lf"])):
        raise ValueError("five-policy protocol 모델·상한·authority·source 결합이 다르다")
    _git(repo, "cat-file", "-e", f"{anchor}^{{commit}}")
    _git(repo, "merge-base", "--is-ancestor", anchor, "HEAD")
    bindings = {relative: _sha(path, lf=True), **value["sources_lf"]}
    for name, expected in bindings.items():
        source = BASE._source_path(repo, name)
        if not isinstance(expected, str) or not HEX64.fullmatch(expected):
            raise ValueError("source SHA 형식이 잘못됐다")
        if not source.is_file() or _sha(source, lf=True) != expected:
            raise ValueError(f"현재 source SHA가 protocol과 다르다: {name}")
        committed = _git(repo, "show", f"{anchor}:{name}")
        normalized = committed.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        if hashlib.sha256(normalized).hexdigest() != expected:
            raise ValueError(f"anchor source blob이 protocol과 다르다: {name}")
    return deepcopy(value)


def _build_engine() -> ModuleType:
    path = ROOT / "evals/local_pair_nim_transport_v1.py"
    spec = importlib.util.spec_from_file_location("_lineage_five_policy_nim_engine_v1", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name, value in {
        "ROOT": ROOT,
        "PROTOCOL_SCHEMA": PROTOCOL_SCHEMA,
        "REQUEST_SCHEMA": REQUEST_SCHEMA,
        "MANIFEST_SCHEMA": MANIFEST_SCHEMA,
        "INTENT_SCHEMA": INTENT_SCHEMA,
        "RESULT_SCHEMA": RESULT_SCHEMA,
        "RECEIPT_SCHEMA": RECEIPT_SCHEMA,
        "SOURCE_FILES": SOURCE_FILES,
        "AUTHORITY": AUTHORITY,
        "__file__": str(Path(__file__).resolve()),
    }.items():
        module.__dict__[name] = value
    module.__dict__["_verify_protocol"] = _verify_protocol
    return module


_ENGINE = _build_engine()


def initialize(root: Path, *, protocol_path: Path, execution_anchor: str,
               api_cap: int) -> dict[str, Any]:
    """등록 protocol을 검증하고 reservation 0의 fresh 72-call 원장을 만든다."""
    target = Path(root).resolve()
    if target.exists():
        raise FileExistsError("transport root는 fresh 경로여야 한다")
    protocol = _verify_protocol(protocol_path, execution_anchor)
    if api_cap != API_CAP or api_cap != protocol["api_cap"]:
        raise ValueError("호출 API cap이 등록 protocol과 다르다")
    target.mkdir(parents=True)
    for name in ("calls", "requests", "locks"):
        (target / name).mkdir()
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "protocol": {"relative": _relative(ROOT, protocol_path),
                     "sha256_lf": _sha(protocol_path, lf=True)},
        "execution_anchor": execution_anchor,
        "api_cap": API_CAP,
        "backend0prefix": 0,
        "model": MODEL_CALL.MODEL,
        "accepted_provider_models": sorted(MODEL_CALL.ACCEPTED_PROVIDER_MODELS),
        "base_params": deepcopy(CALL.PARAMS),
        "automatic_retries": 0,
    }
    _ENGINE._write_new(target / "manifest.json", manifest)
    return deepcopy(manifest)


def complete(root: Path, request: dict, *, send) -> dict[str, Any]:
    return _verified(_ENGINE.complete(root, request, send=send))


def submit(root: Path, request: dict, *, send) -> dict[str, Any]:
    return complete(root, request, send=send)


def verify_call(call_path: Path) -> dict[str, Any]:
    return _verified(_ENGINE.verify_call(call_path))


def verify(call_path: Path) -> dict[str, Any]:
    return verify_call(call_path)


def _verified(value: dict[str, Any]) -> dict[str, Any]:
    if value.get("schema") != "local_pair_nim_verified_call_v1":
        raise ValueError("기존 transport verified 결과 schema가 다르다")
    result = deepcopy(value)
    result["schema"] = VERIFIED_SCHEMA
    return result


def status(root: Path) -> dict[str, Any]:
    target = Path(root).resolve()
    manifest = _ENGINE._manifest(target)
    calls = sorted(path for path in (target / "calls").iterdir() if path.is_dir())
    ordinals = [int(path.name) for path in calls if re.fullmatch(r"[0-9]{4}", path.name)]
    if len(ordinals) != len(calls) or ordinals != list(range(1, len(calls) + 1)):
        raise ValueError("transport call ordinal이 연속적이지 않다")
    outcomes = {"semantic_valid": 0, "semantic_invalid": 0, "infrastructure_unknown": 0}
    completed, pending, dispatched = 0, 0, 0
    for call in calls:
        intent = _read(call / "intent.json")
        if intent.get("schema") != INTENT_SCHEMA:
            raise ValueError("예약 intent schema가 다르다")
        if (call / "dispatch-claim.json").is_file():
            dispatched += 1
        if (call / "receipt.json").is_file():
            checked = verify_call(call)
            outcomes[checked["outcome"]] += 1
            completed += 1
        else:
            pending += 1
    reserved = len(calls)
    return {"schema": STATUS_SCHEMA, "api_cap": manifest["api_cap"],
            "reservations_charged": reserved, "remaining": manifest["api_cap"] - reserved,
            "dispatched": dispatched, "completed": completed, "pending_no_retry": pending,
            "outcomes": outcomes, "automatic_retries": 0, "backend0prefix": 0}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--call-dir", type=Path)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--anchor")
    args = parser.parse_args(argv)
    if not args.worker or any(value is None for value in (
            args.root, args.call_dir, args.protocol, args.anchor)):
        parser.error("worker 인자 전체가 필요하다")
    return _ENGINE._run_worker_cli(args.root, args.call_dir, args.protocol, args.anchor)


if __name__ == "__main__":
    raise SystemExit(main())
