"""V1 정책을 유지하고 camelCase GPU lease identity를 엄격히 연결하는 v2 wrapper."""
# pyright: reportMissingImports=false
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import threading
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from threading import RLock
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "bench"), str(ROOT / "src")]

import run_prospective_reasoning_confirmation_v1 as R  # noqa: E402

PROTOCOL_SCHEMA = "prospective_reasoning_confirmation_execution_v2"
PREFLIGHT_SCHEMA = "prospective_reasoning_confirmation_preflight_v2"
STAGE_A_ANCHOR = "304e456432dc0b1218fea5f38bec4f55f25bdd2c"
STAGE_B_ANCHOR = "83f9b8c6f7048f9368e617c203d81bca0e7f6b15"
LOCAL_SOURCE_FILES = (
    "evals/run_prospective_reasoning_confirmation_v2.py",
    "tests/test_run_prospective_reasoning_confirmation_v2.py",
)
HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_LOCK = RLock()
_LOCAL = threading.local()
_REGISTERED_LEASE_CONTEXT = R.SOURCE.LEASE.lease_context


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


def _write_new(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def _relative(path: Path) -> str:
    path = Path(path).resolve()
    require(path.is_file() and path.is_relative_to(ROOT), "protocol은 저장소 파일이어야 한다")
    return path.relative_to(ROOT).as_posix()


def _ref(path: Path) -> dict[str, str]:
    return {"path": _relative(path), "sha256_lf": _sha(path, lf=True)}


def _git(*args: str) -> bytes:
    run = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, check=False)
    require(run.returncode == 0, run.stderr.decode("utf-8", errors="replace").strip())
    return run.stdout


def normalize_lease_metadata(raw: dict[str, Any]) -> dict[str, Any]:
    """검증된 원 metadata를 보존하고 V1이 읽는 snake_case alias만 추가한다."""
    require(isinstance(raw, dict), "lease metadata는 object여야 한다")
    lease_id = raw.get("leaseId")
    try:
        parsed = uuid.UUID(cast(str, lease_id))
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("leaseId는 nonempty canonical UUID여야 한다") from error
    require(str(parsed) == lease_id and bool(lease_id), "leaseId는 canonical UUID여야 한다")
    snake = raw.get("lease_id")
    require(snake is None or snake == lease_id, "leaseId와 lease_id가 충돌한다")
    require(raw.get("owner") == R.SOURCE.OWNER and raw.get("state") == "running"
            and raw.get("completedUtc") is None,
            "lease owner/state/completion이 실행 계약과 다르다")
    normalized = deepcopy(raw)
    normalized["lease_id"] = lease_id
    return normalized


@contextmanager
def lease_adapter_scope() -> Iterator[None]:
    """V1 실행 구간에서만 실제 validator 결과에 lease_id alias를 추가한다."""
    module = R.SOURCE.LEASE
    with _LOCK:
        require(getattr(_LOCAL, "depth", 0) == 0, "lease adapter scope 재진입은 허용되지 않는다")
        require(module.lease_context is _REGISTERED_LEASE_CONTEXT,
                "등록 lease_context가 이미 교체됐다")
        original = module.lease_context

        def adapted(evidence_dir: Path | None = None) -> dict[str, Any]:
            raw = original(evidence_dir)
            return normalize_lease_metadata(raw)

        module.lease_context = adapted
        _LOCAL.depth = 1
        try:
            yield
        finally:
            _LOCAL.depth = 0
            module.lease_context = original


def prepare_protocol(base_execution_protocol: Path) -> dict[str, Any]:
    """동일 V1 실행 계약에 lease interface 수정 closure만 추가한다."""
    base_path = Path(base_execution_protocol).resolve()
    base = _read(base_path)
    require(base.get("schema") == R.EXECUTION_SCHEMA
            and base.get("status") == "registered_only_after_commit"
            and base.get("design_anchor") == STAGE_A_ANCHOR
            and base.get("planned_rows") == 24 and base.get("fault_rows") == 18
            and base.get("normal_host_gate_rows") == 6
            and base.get("claims") == {"new_fault_family_generalization": False,
                                       "llm_superiority": False,
                                       "default_mode_adoption": False,
                                       "repair": False, "recovery": False},
            "base V1 execution 계약 불일치")
    design_path = ROOT / base["design_protocol"]["path"]
    bridge_path = ROOT / base["bridge_protocol"]["path"]
    design = _read(design_path)
    _read(bridge_path)
    require(base["design_protocol"]["sha256"] == _sha(design_path)
            and base["bridge_protocol"]["sha256"] == _sha(bridge_path),
            "base design/bridge ref 불일치")
    closure = deepcopy(design["source_sha256_lf"])
    for path in (base_path, design_path, bridge_path,
                 ROOT / design["source_protocol"]["path"]):
        closure[_relative(path)] = _sha(path, lf=True)
    closure.update({name: _sha(ROOT / name, lf=True) for name in LOCAL_SOURCE_FILES})
    return {
        "schema": PROTOCOL_SCHEMA, "status": "registered_only_after_commit",
        "base_execution_protocol": _ref(base_path),
        "base_execution_anchor": STAGE_B_ANCHOR,
        "base_design_anchor": STAGE_A_ANCHOR,
        "base_design_protocol": _ref(design_path),
        "base_bridge_protocol": _ref(bridge_path),
        "inherited": {"policies": list(R.POLICIES), "planned_rows": 24,
                      "fault_rows": 18, "normal_host_gate_rows": 6,
                      "api_requests": R.MAX_API, "paired_probes": R.MAX_PAIRED,
                      "sdk_role_executions_upper_bound": R.MAX_SDK_ROLES,
                      "claims": deepcopy(base["claims"])},
        "change": {"lease_interface": "validated_leaseId_to_lease_id_alias",
                   "raw_metadata_mutated": False, "policy_or_prompt_changed": False,
                   "threshold_or_case_changed": False, "v1_execution_preserved": "unexecuted"},
        "source_sha256_lf": closure,
    }


def _verify_blob(anchor: str, relative: str, expected: str) -> None:
    raw = _git("show", f"{anchor}:{relative}").replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    require(hashlib.sha256(raw).hexdigest() == expected,
            f"anchor blob 불일치: {relative}")


def verify_protocol(protocol: Path, anchor: str) -> dict[str, Any]:
    require(HEX40.fullmatch(anchor) is not None, "outer anchor 형식")
    path = Path(protocol).resolve()
    value = _read(path)
    base_ref = value.get("base_execution_protocol")
    require(isinstance(base_ref, dict), "base execution ref 누락")
    base_path = ROOT / cast(dict[str, Any], base_ref)["path"]
    require(value == prepare_protocol(base_path), "outer protocol 현재 bytes 불일치")
    for ancestor, descendant in ((STAGE_A_ANCHOR, STAGE_B_ANCHOR),
                                 (STAGE_B_ANCHOR, anchor), (anchor, "HEAD")):
        _git("merge-base", "--is-ancestor", ancestor, descendant)
    _verify_blob(STAGE_B_ANCHOR, _relative(base_path), _sha(base_path, lf=True))
    _verify_blob(STAGE_A_ANCHOR, value["base_design_protocol"]["path"],
                 value["base_design_protocol"]["sha256_lf"])
    _verify_blob(STAGE_A_ANCHOR, value["base_bridge_protocol"]["path"],
                 value["base_bridge_protocol"]["sha256_lf"])
    bound = {_relative(path): _sha(path, lf=True), **value["source_sha256_lf"]}
    for relative, digest in bound.items():
        current = ROOT / relative
        require(current.is_file() and _sha(current, lf=True) == digest,
                f"현재 outer source 불일치: {relative}")
        _verify_blob(anchor, relative, digest)
    R.verify_design(ROOT / value["base_design_protocol"]["path"], STAGE_A_ANCHOR)
    R.BRIDGE._verify_registered(ROOT / value["base_bridge_protocol"]["path"], STAGE_A_ANCHOR)
    return deepcopy(value)


def preflight(protocol: Path, anchor: str, base_protocol: Path, design: Path, bridge: Path,
              source_batch: Path, corpus: Path, out: Path) -> dict[str, Any]:
    wrapper = verify_protocol(protocol, anchor)
    expected_base = ROOT / wrapper["base_execution_protocol"]["path"]
    require(Path(base_protocol).resolve() == expected_base.resolve(), "base protocol 인자 불일치")
    base = R.verify_execution(expected_base, STAGE_B_ANCHOR, design, bridge,
                              source_batch, corpus)
    value = {"schema": PREFLIGHT_SCHEMA, "status": "complete",
             "outer_protocol_sha256": _sha(protocol), "outer_anchor": anchor,
             "base_execution_sha256": _sha(expected_base),
             "base_execution_anchor": STAGE_B_ANCHOR,
             "design_anchor": base["design_anchor"], "gpu_started": False,
             "api_requests": 0, "policy_or_threshold_changed": False}
    _write_new(Path(out).resolve(), value)
    return value


def execute(protocol: Path, anchor: str, base_protocol: Path, design: Path, bridge: Path,
            source_batch: Path, corpus: Path, lab: Path, bridge_root: Path, out: Path, *,
            base_execute: Callable[..., dict[str, Any]] = R.execute) -> dict[str, Any]:
    wrapper = verify_protocol(protocol, anchor)
    expected_base = ROOT / wrapper["base_execution_protocol"]["path"]
    require(Path(base_protocol).resolve() == expected_base.resolve(), "base protocol 인자 불일치")
    with lease_adapter_scope():
        result = base_execute(expected_base, STAGE_B_ANCHOR, design, bridge, source_batch,
                              corpus, lab, bridge_root, out)
    require(result.get("schema") == R.RESULT_SCHEMA, "V1 result schema가 바뀌었다")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare-protocol")
    prepare.add_argument("--base-execution-protocol", type=Path, required=True)
    prepare.add_argument("--out", type=Path, required=True)
    for name in ("preflight", "execute"):
        command = commands.add_parser(name)
        for argument in ("protocol", "base-protocol", "design", "bridge", "source-batch",
                         "corpus", "out"):
            command.add_argument(f"--{argument}", type=Path, required=True)
        command.add_argument("--anchor", required=True)
        if name == "execute":
            command.add_argument("--lab", type=Path, required=True)
            command.add_argument("--bridge-root", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare-protocol":
        _write_new(args.out, prepare_protocol(args.base_execution_protocol))
    elif args.command == "preflight":
        preflight(args.protocol, args.anchor, args.base_protocol, args.design, args.bridge,
                  args.source_batch, args.corpus, args.out)
    else:
        execute(args.protocol, args.anchor, args.base_protocol, args.design, args.bridge,
                args.source_batch, args.corpus, args.lab, args.bridge_root, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["execute", "lease_adapter_scope", "normalize_lease_metadata",
           "preflight", "prepare_protocol", "verify_protocol"]
