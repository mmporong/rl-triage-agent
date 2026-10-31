"""검증된 학습 runtime facts를 기존 공통 선택 view에 추가하는 CPU 경계."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))

import optimizer_runtime_facts_v1 as FACT
import selection_comparison_view_v1 as V

_LEGACY_BASE_FACT_IDS = frozenset(V.BASE_FACT_IDS)

LEGACY_PROTOCOL = ROOT / "bench/protocols/selection_comparison_view_v1.json"
PROTOCOL_RELATIVE = "bench/protocols/runtime_observation_view_v1.json"
PROTOCOL = ROOT / PROTOCOL_RELATIVE
SCHEMA = "runtime_observation_view_protocol_v1"
MEASUREMENT_KIND = "verified_training_runtime_facts"
SOURCE_FILES = tuple(sorted(set(V.SOURCE_FILES) | {
    "bench/protocols/selection_comparison_view_v1.json",
    "evals/runtime_observation_view_v1.py",
    "tests/test_runtime_observation_view_v1.py",
    "evals/optimizer_runtime_facts_v1.py",
    "tests/test_optimizer_runtime_facts_v1.py",
}))
BASE_FACT_IDS = frozenset(_LEGACY_BASE_FACT_IDS | set(FACT.FACT_IDS))

_TOP_LEVEL_FIELDS = {
    "schema", "measurement_kind", "public_module", "legacy_component",
    "public_base_fact_ids", "runtime_fact_ids", "source_sha256_lf", "scope", "claims",
}
_LEGACY_FIELDS = {
    "path", "sha256_lf", "schema", "measurement_kind", "public_base_fact_ids",
}
_SCOPE = {
    "activity": "read_only_runtime_observation_view",
    "approval": False,
    "execution_allowed": False,
    "actual_fact_validation": "optimizer_runtime_facts_v1.checked_and_registered_main_runner",
    "fixture_claims_actual_ownership": False,
}
_CLAIMS = {
    "synthetic_measurement": False,
    "legacy_source_relabelled": False,
    "runtime_fact_values_verified_by_view": False,
    "approval_granted": False,
    "execution_authorized": False,
}
_LOCK = threading.RLock()


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def canonical(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode("utf-8")


def file_digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _read_protocol(path: Path) -> tuple[bytes, dict]:
    raw = Path(path).read_bytes().replace(b"\r\n", b"\n")
    try:
        value = json.loads(raw, parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"비유한 JSON 숫자는 허용하지 않는다: {token}")))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError("runtime view protocol JSON 불일치") from error
    require(isinstance(value, dict), "runtime view protocol은 객체여야 한다")
    return raw, value


def _legacy_component() -> dict:
    legacy_sha = file_digest(LEGACY_PROTOCOL)
    raw = LEGACY_PROTOCOL.read_bytes().replace(b"\r\n", b"\n")
    legacy = json.loads(raw)
    require(hashlib.sha256(raw).hexdigest() == legacy_sha
            and legacy.get("schema") == "selection_comparison_view_protocol_v1"
            and legacy.get("measurement_kind") == "synthetic_cpu_contract_only"
            and set(legacy.get("source_sha256_lf", {})) == set(V.SOURCE_FILES)
            and all(file_digest(ROOT / name) == sha
                    for name, sha in legacy["source_sha256_lf"].items())
            and set(legacy.get("public_base_fact_ids", [])) == _LEGACY_BASE_FACT_IDS,
            "legacy view protocol 또는 source 결합 불일치")
    return {
        "path": LEGACY_PROTOCOL.relative_to(ROOT).as_posix(),
        "sha256_lf": legacy_sha,
        "schema": legacy["schema"],
        "measurement_kind": legacy["measurement_kind"],
        "public_base_fact_ids": sorted(legacy["public_base_fact_ids"]),
    }


def prepare_protocol() -> dict:
    """현재 source bytes와 동결 legacy 구성요소를 새 공개 프로토콜 값으로 만든다."""
    legacy = _legacy_component()
    return {
        "schema": SCHEMA,
        "measurement_kind": MEASUREMENT_KIND,
        "public_module": "evals/runtime_observation_view_v1.py",
        "legacy_component": legacy,
        "public_base_fact_ids": sorted(BASE_FACT_IDS),
        "runtime_fact_ids": list(FACT.FACT_IDS),
        "source_sha256_lf": {name: file_digest(ROOT / name) for name in SOURCE_FILES},
        "scope": deepcopy(_SCOPE),
        "claims": deepcopy(_CLAIMS),
    }


def _git(*args: str) -> bytes:
    result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, check=False)
    require(result.returncode == 0, "runtime view source의 commit/blob 검증 실패")
    return result.stdout.replace(b"\r\n", b"\n")


def verify_protocol(path: str | Path, anchor: str | None, *, fixture: bool = False) -> dict:
    """새 protocol, 현재 LF source와 명시 commit blob을 fail-closed로 검증한다."""
    require(type(fixture) is bool, "fixture 플래그 형식 불일치")
    protocol_path = Path(path).resolve()
    raw, value = _read_protocol(protocol_path)
    require(set(value) == _TOP_LEVEL_FIELDS
            and value.get("schema") == SCHEMA
            and value.get("measurement_kind") == MEASUREMENT_KIND
            and value.get("public_module") == "evals/runtime_observation_view_v1.py",
            "runtime view protocol schema 또는 필드 범위 불일치")
    legacy = value.get("legacy_component")
    require(isinstance(legacy, dict) and set(legacy) == _LEGACY_FIELDS
            and legacy == _legacy_component(), "legacy view 구성요소가 그대로 보존되지 않았다")
    require(value.get("runtime_fact_ids") == list(FACT.FACT_IDS)
            and value.get("public_base_fact_ids") == sorted(BASE_FACT_IDS),
            "runtime 또는 전체 공개 fact allowlist 불일치")
    require(value.get("scope") == _SCOPE and value.get("claims") == _CLAIMS,
            "runtime view 권한 또는 주장 범위 불일치")
    sources = value.get("source_sha256_lf")
    if not isinstance(sources, dict):
        raise TypeError("runtime view source allowlist 불일치")
    require(set(sources) == set(SOURCE_FILES)
            and all(isinstance(sha, str) and bool(re.fullmatch(r"[0-9a-f]{64}", sha))
                    for sha in sources.values()), "runtime view source allowlist 불일치")
    for name, sha in sources.items():
        require(file_digest(ROOT / name) == sha, f"runtime view current source 변경: {name}")

    if fixture:
        require(anchor is None, "fixture는 commit 소유권 anchor를 주장하지 않는다")
        return deepcopy(value)

    if not isinstance(anchor, str) or not re.fullmatch(r"[0-9a-f]{40}", anchor):
        raise ValueError("production runtime view에는 명시 anchor가 필요하다")
    require(protocol_path == PROTOCOL.resolve(), "등록되지 않은 runtime view protocol 경로")
    protocol_name = PROTOCOL_RELATIVE
    _git("merge-base", "--is-ancestor", anchor, "HEAD")
    bound = {**sources, protocol_name: hashlib.sha256(raw).hexdigest()}
    for name, sha in bound.items():
        for revision in (anchor, "HEAD", ""):
            spec = f"{revision}:{name}" if revision else f":{name}"
            require(hashlib.sha256(_git("show", spec)).hexdigest() == sha,
                    f"runtime view source commit/blob/index 불일치: {name}")
    return deepcopy(value)


@contextmanager
def view_scope(path: str | Path, anchor: str | None, *, fixture: bool = False) -> Iterator[dict]:
    """기존 view engine의 공개 fact/protocol 검증점만 새 계약으로 좁게 교체한다.

    CALL2.verify_sources는 legacy protocol 위치를 고정하므로 이 scope의 production transport가 아니다.
    실제 모델 transport는 등록된 main 사전계획, v2 source 검증과 outer runner 결합이 책임진다.
    """
    protocol_path = Path(path).resolve()
    verified = verify_protocol(protocol_path, anchor, fixture=fixture)
    protocol_sha = file_digest(protocol_path)
    with _LOCK:
        previous = (V.BASE_FACT_IDS, V.PROTOCOL, V.verify_protocol)

        def scoped_verify(expected: str) -> dict:
            require(expected == protocol_sha, "다른 runtime view protocol SHA")
            return verify_protocol(protocol_path, anchor, fixture=fixture)

        V.BASE_FACT_IDS = BASE_FACT_IDS
        V.PROTOCOL = protocol_path
        V.verify_protocol = scoped_verify
        try:
            yield deepcopy(verified)
        finally:
            V.BASE_FACT_IDS, V.PROTOCOL, V.verify_protocol = previous


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-protocol", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify-protocol", type=Path)
    parser.add_argument("--anchor")
    args = parser.parse_args()
    if args.prepare_protocol:
        output = args.output
        if output is None or args.verify_protocol is not None or args.anchor is not None:
            raise ValueError("protocol prepare 인자 불일치")
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("xb") as stream:
            stream.write(canonical(prepare_protocol()))
        print(file_digest(output))
        return 0
    verify_path = args.verify_protocol
    if verify_path is None or args.output is not None:
        raise ValueError("verify에는 protocol 경로가 필요하다")
    verify_protocol(verify_path, args.anchor)
    print(file_digest(verify_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
