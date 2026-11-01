# pyright: reportMissingImports=false
"""기존 R4 recovery CLI를 canonical bounded lease reader로 실행하는 outer host."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("bench", "evals", "src")]

import lineage_lease_context_v2 as RETRY
import run_five_policy_recovery_loop_v1 as RUN

SCHEMA = "five_policy_recovery_lease_outer_host_v1"
SOURCE_FILES = (
    "bench/run_five_policy_recovery_with_lease_v1.py",
    "tests/test_five_policy_recovery_with_lease_v1.py",
    *RETRY.SOURCE_FILES,
)
PHASE_COMMANDS = {
    "prepare-first": "first", "run-first": "first",
    "prepare-second": "second", "run-second": "second",
}
HEX40 = frozenset("0123456789abcdef")
_REGISTERED_LEASE = RUN.LEASE


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def sha(path: Path, *, lf: bool = False) -> str:
    raw = Path(path).read_bytes()
    if lf:
        raw = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(raw).hexdigest()


def read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    require(isinstance(value, dict), "outer host protocol은 JSON object여야 한다")
    return value


def write_new(path: Path, value: object) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def relative(path: Path, repo: Path) -> str:
    try:
        return Path(path).resolve(strict=True).relative_to(Path(repo).resolve()).as_posix()
    except ValueError as error:
        raise ValueError("outer host source/protocol은 저장소 내부여야 한다") from error


def _hex40(value: object) -> bool:
    return isinstance(value, str) and len(value) == 40 and set(value) <= HEX40


def _git(repo: Path, *args: str) -> bytes:
    return subprocess.run(["git", "-C", str(repo), *args], check=True,
                          capture_output=True).stdout


def _source_map(repo: Path = ROOT) -> dict[str, str]:
    return {name: sha(Path(repo) / name, lf=True) for name in sorted(set(SOURCE_FILES))}


def _phase_row(path: Path, anchor: str, repo: Path) -> dict[str, str]:
    require(_hex40(anchor), "phase source anchor 형식")
    target = Path(path).resolve(strict=True)
    return {"path": relative(target, repo), "sha256": sha(target),
            "sha256_lf": sha(target, lf=True), "source_anchor": anchor}


def build_host_protocol(stage_protocols: Mapping[str, Mapping[str, Any]], out: Path, *,
                        repo: Path = ROOT) -> dict[str, Any]:
    """first 또는 second stage protocol을 outer host source와 함께 사전등록한다."""
    require(bool(stage_protocols) and set(stage_protocols) <= {"first", "second"},
            "stage protocol mapping")
    stages = {}
    for phase, row in stage_protocols.items():
        require(isinstance(row, Mapping) and set(row) == {"path", "source_anchor"},
                f"{phase} stage protocol 입력")
        stages[phase] = _phase_row(Path(cast(str, row["path"])),
                                   cast(str, row["source_anchor"]), repo)
    runner = Path(repo) / "bench/run_five_policy_recovery_loop_v1.py"
    value = {
        "schema": SCHEMA, "status": "preregistered_outer_host",
        "source_files_lf": _source_map(repo),
        "underlying_runner": {"path": relative(runner, repo), "sha256": sha(runner),
                              "sha256_lf": sha(runner, lf=True)},
        "stage_protocols": dict(sorted(stages.items())),
        "lease_reader": {"module": "lineage_lease_context_v2",
                         "callback": "lease_context", "source_files_lf": {
                             name: sha(Path(repo) / name, lf=True) for name in RETRY.SOURCE_FILES},
                         },
        "scope": {"prepare_cli_reused": True, "run_cli_reused": True,
                  "lease_binding_only": True, "approval_changed": False,
                  "sdk_api_caps_changed": False, "lease_validation_changed": False},
    }
    write_new(out, value)
    return value


def _verify_blob(repo: Path, anchor: str, relative_path: str, expected_lf: str) -> None:
    blob = _git(repo, "show", f"{anchor}:{relative_path}")
    normalized = blob.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    require(hashlib.sha256(normalized).hexdigest() == expected_lf,
            f"anchor blob 변경: {relative_path}")


def _default_phase_verifiers() -> dict[str, Callable[[Path, str, Path], Any]]:
    return {
        "first": lambda path, anchor, lab: RUN.FIRST.verify_protocol(
            path, anchor=anchor, lab=lab),
        "second": lambda path, anchor, lab: RUN.SECOND.verify_loop_protocol(
            path, anchor, lab=lab),
    }


def validate_host_protocol(path: Path, host_anchor: str, *, phase: str, lab: Path,
                           repo: Path = ROOT,
                           phase_verifiers: Mapping[str, Callable[[Path, str, Path], Any]] | None = None
                           ) -> dict[str, Any]:
    require(_hex40(host_anchor) and phase in {"first", "second"}, "host anchor/phase")
    repo = Path(repo).resolve()
    value = read(path)
    require(set(value) == {"schema", "status", "source_files_lf", "underlying_runner",
                           "stage_protocols", "lease_reader", "scope"}
            and value.get("schema") == SCHEMA
            and value.get("status") == "preregistered_outer_host", "outer host shape/status")
    require(value["source_files_lf"] == _source_map(repo), "outer host source SHA")
    expected_retry = {name: sha(repo / name, lf=True) for name in RETRY.SOURCE_FILES}
    require(value["lease_reader"] == {"module": "lineage_lease_context_v2",
                                      "callback": "lease_context",
                                      "source_files_lf": expected_retry},
            "기존 retry helper 결합")
    runner = repo / "bench/run_five_policy_recovery_loop_v1.py"
    expected_runner = {"path": relative(runner, repo), "sha256": sha(runner),
                       "sha256_lf": sha(runner, lf=True)}
    require(value["underlying_runner"] == expected_runner, "frozen underlying runner SHA")
    require(value["scope"] == {"prepare_cli_reused": True, "run_cli_reused": True,
                               "lease_binding_only": True, "approval_changed": False,
                               "sdk_api_caps_changed": False, "lease_validation_changed": False},
            "outer host scope")
    head = _git(repo, "rev-parse", "HEAD").decode().strip()
    ancestry = subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor",
                               host_anchor, head], check=False)
    require(ancestry.returncode == 0, "outer host registration anchor ancestry")
    protocol_relative = relative(path, repo)
    _verify_blob(repo, host_anchor, protocol_relative, sha(path, lf=True))
    for name, digest in value["source_files_lf"].items():
        _verify_blob(repo, host_anchor, name, digest)
    _verify_blob(repo, host_anchor, expected_runner["path"], expected_runner["sha256_lf"])

    stages = value.get("stage_protocols")
    require(isinstance(stages, dict) and phase in stages, f"{phase} stage protocol 미등록")
    row = stages[phase]
    require(isinstance(row, dict)
            and set(row) == {"path", "sha256", "sha256_lf", "source_anchor"}
            and _hex40(row["source_anchor"]), f"{phase} stage protocol ref")
    stage_path = (repo / row["path"]).resolve(strict=True)
    require(stage_path.is_relative_to(repo) and sha(stage_path) == row["sha256"]
            and sha(stage_path, lf=True) == row["sha256_lf"], f"{phase} protocol SHA")
    _verify_blob(repo, row["source_anchor"], row["path"], row["sha256_lf"])
    verifiers = dict(_default_phase_verifiers() if phase_verifiers is None else phase_verifiers)
    require(set(verifiers) == {"first", "second"}, "phase verifier 집합")
    verifiers[phase](stage_path, row["source_anchor"], Path(lab).resolve(strict=True))
    return value


class LeaseProxy:
    """기존 LEASE 모듈 속성을 위임하고 lease_context만 bounded helper로 교체한다."""
    def __init__(self, base: Any, reader: Callable[..., dict[str, Any]]) -> None:
        self._base = base
        self.lease_context = reader

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)


@contextmanager
def retry_lease_binding():
    require(RUN.LEASE is _REGISTERED_LEASE, "underlying RUN.LEASE가 이미 교체됐다")
    RUN.LEASE = LeaseProxy(_REGISTERED_LEASE, RETRY.lease_context)
    try:
        yield
    finally:
        RUN.LEASE = _REGISTERED_LEASE


def _extract_host_args(argv: Sequence[str]) -> tuple[Path, str, Path, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--host-protocol", type=Path, required=True)
    parser.add_argument("--host-anchor", required=True)
    parser.add_argument("--host-lab", type=Path, required=True)
    known, remaining = parser.parse_known_args(list(argv))
    require(bool(remaining) and remaining[0] in PHASE_COMMANDS,
            "underlying recovery command")
    return known.host_protocol, known.host_anchor, known.host_lab, remaining


def main(argv: Sequence[str] | None = None) -> int:
    host_protocol, host_anchor, host_lab, delegated = _extract_host_args(
        sys.argv[1:] if argv is None else argv)
    command = delegated[0]
    validate_host_protocol(host_protocol, host_anchor, phase=PHASE_COMMANDS[command], lab=host_lab)
    if command.startswith("run-"):
        with retry_lease_binding():
            return int(RUN.main(delegated))
    return int(RUN.main(delegated))


if __name__ == "__main__":
    raise SystemExit(main())
